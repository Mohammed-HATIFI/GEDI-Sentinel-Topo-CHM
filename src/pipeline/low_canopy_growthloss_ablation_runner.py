from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT = Path(r"C:\Users\Dell\Desktop\Publication_Clarck")
GROWTH_ROOT = (PROJECT / "Source" / "Training" / "growthloss_v6").resolve()
OFFICIAL_REPO = GROWTH_ROOT / "official_source" / "ECHOSAT-main"
METHOD = "LOW_CANOPY_DROP_LAMBDA_K_V1"
YEARS = tuple(range(2019, 2026))
MONTHS = (5, 6, 7, 8, 9)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AGADIR_CONFIRMATORY_MODE = False

LOW_CANOPY_CANDIDATES = {
    "D2_K2_GL005": {"drop_m": 2.0, "K": 2, "lambda_growth": 0.05},
    "D2_K2_GL010": {"drop_m": 2.0, "K": 2, "lambda_growth": 0.10},
    "D3_K2_GL005": {"drop_m": 3.0, "K": 2, "lambda_growth": 0.05},
    "D3_K2_GL010": {"drop_m": 3.0, "K": 2, "lambda_growth": 0.10},
    "D3_K3_GL010": {"drop_m": 3.0, "K": 3, "lambda_growth": 0.10},
    "D4_K2_GL010": {"drop_m": 4.0, "K": 2, "lambda_growth": 0.10},
}

IFRAN_CANDIDATES = {
    # Ifran-specific sensitivity grid: keep disturbance thresholds near the
    # established 5 m operating point instead of importing the low-canopy grid.
    "D4_K2_GL010": {"drop_m": 4.0, "K": 2, "lambda_growth": 0.10},
    # D5/K2/GL005 and D5/K2/GL010 RAW were already explored in the historical
    # 25-run Ifran campaign.  They are intentionally not retrained here.
    "D5_K3_GL010": {"drop_m": 5.0, "K": 3, "lambda_growth": 0.10},
    "D6_K2_GL010": {"drop_m": 6.0, "K": 2, "lambda_growth": 0.10},
    "D7_K2_GL010": {"drop_m": 7.0, "K": 2, "lambda_growth": 0.10},
    # Controlled Huber comparison against the existing D5/K2/GL020/HD3 run.
    "D5_K2_GL020_HD1": {"drop_m": 5.0, "K": 2, "lambda_growth": 0.20, "huber_delta": 1.0},
}

# Confirmatory Agadir grid.  It is deliberately isolated from the earlier
# exploratory family so that completed runs and VAL selections are immutable.
# GL000 is a matched temporal fine-tuning control: the disturbance parameters
# are inert because lambda_growth=0.
AGADIR_CONFIRMATORY_CANDIDATES = {
    "AG_CTRL_GL000": {"drop_m": 5.0, "K": 2, "lambda_growth": 0.0},
    "AG_D2_K2_GL001": {"drop_m": 2.0, "K": 2, "lambda_growth": 0.01},
    "AG_D2_K2_GL0025": {"drop_m": 2.0, "K": 2, "lambda_growth": 0.025},
    "AG_D3_K2_GL0025": {"drop_m": 3.0, "K": 2, "lambda_growth": 0.025},
    "AG_D4_K2_GL005": {"drop_m": 4.0, "K": 2, "lambda_growth": 0.05},
    "AG_D3_K3_GL005": {"drop_m": 3.0, "K": 3, "lambda_growth": 0.05},
    "AG_D4_K3_GL005": {"drop_m": 4.0, "K": 3, "lambda_growth": 0.05},
    # Matched D5/K2 dose-response points.  GL010 already exists below; GL005
    # and GL020 use exactly the same parent, seed, LR and stopping protocol.
    "AG_D5_K2_GL005": {"drop_m": 5.0, "K": 2, "lambda_growth": 0.05},
    "AG_D5_K2_GL010": {"drop_m": 5.0, "K": 2, "lambda_growth": 0.10},
    "AG_D5_K2_GL020": {"drop_m": 5.0, "K": 2, "lambda_growth": 0.20},
}

FORESTS = {
    "ifran": {
        "label": "Ifran", "ecosystem": "Dense",
        "catalog": PROJECT / "Data" / "Dense" / "Ifran" / "Catalogs" / "final_catalog_C15_NATIVE",
        "parent": PROJECT / "Models" / "Dense" / "Ifran" / "best_slope.ckpt",
        "parent_sha": "330cdd7b47b064072bc6224b0e23f44ac9076182f490b00c2fdbd495aca652e0",
        "huber_delta": 3.0, "height_bins": (0, 2.5, 5, 8, 10, 15, 20, 30, 40, 45),
        "expected_train_records": 134, "expected_val_records": 30,
        "expected_test_n": 5076, "eval_min": 2.0, "eval_max": 45.0,
        "max_steps": 6600, "mae_tolerance": 0.05,
        "ablation_family": "Ablations_Ifran_DropLambdaK_HuberDelta_v1",
        "baseline_run": PROJECT / "Models" / "Dense" / "Ifran" / "Phase2" / "GrowthLoss_RAW_DROP5_K2_GL020" / "IFRAN_B4_C15_PHASE2_T4_PERSISTENT_RAW_DROP5_K2_GL020_SEED42",
    },
    "maamoura": {
        "label": "Maamoura", "ecosystem": "Low_Sparsity",
        "catalog": PROJECT / "Data" / "Low_Sparsity" / "Maamoura" / "Temporal_Catalogs" / "T4_DENSE_2019_2025_C15",
        "parent": PROJECT / "Models" / "Low_Sparsity" / "Maamoura" / "best_any.ckpt",
        "parent_sha": "dd6be72a60c165b22c2f048da93cf4b418a00fa5b5d6aef6a4f296654df1e3c6",
        "huber_delta": 1.0, "height_bins": (0, 2, 5, 8, 10, 15, 20),
        "expected_train_records": 260, "expected_val_records": 60,
        "expected_test_n": 1799, "eval_min": 2.0, "eval_max": 20.0,
        "max_steps": 4400, "mae_tolerance": 0.03,
        "ablation_family": "Ablations_LowCanopy_DropLambdaK_v1",
        "baseline_run": PROJECT / "Models" / "Low_Sparsity" / "Maamoura" / "Phase2" / "GrowthLoss_RAW_DROP5_K2_GL020_DENSE_T4_2019_2025" / "MAAMOURA_B4_C15_PHASE2_DENSE2019_2025_T4_PERSISTENT_RAW_DROP5_K2_GL020_VAL2_20_SEED42",
    },
    "agadir": {
        "label": "Agadir", "ecosystem": "Sparse",
        "catalog": PROJECT / "Data" / "Sparse" / "Agadir" / "Catalogs" / "final_catalog",
        "parent": PROJECT / "Models" / "Sparse" / "Agadir" / "best_slope.ckpt",
        "parent_sha": "1ac590d552168caee40d3a888f92e250b6c0faacc2a662e5f43bae0f1b998486",
        "huber_delta": 3.0, "height_bins": (0, 2, 5, 8, 10, 15, 20),
        "expected_train_records": 191, "expected_val_records": 38,
        "expected_test_n": 6125, "eval_min": 2.0, "eval_max": 20.0,
        "max_steps": 4400, "mae_tolerance": 0.03,
        "ablation_family": "Ablations_LowCanopy_DropLambdaK_v1",
        "baseline_run": PROJECT / "Models" / "Sparse" / "Agadir" / "Phase2" / "GrowthLoss_RAW_DROP5_K2_GL020" / "AGADIR_B4_C15_PHASE2_T4_PERSISTENT_RAW_DROP5_K2_GL020_SEED42",
    },
}


def candidates_for(forest_key: str) -> dict:
    if forest_key == "agadir" and AGADIR_CONFIRMATORY_MODE:
        return AGADIR_CONFIRMATORY_CANDIDATES
    if forest_key == "ifran":
        return IFRAN_CANDIDATES
    candidates = dict(LOW_CANOPY_CANDIDATES)
    if forest_key == "maamoura":
        # Complete the matched D2/K2/Huber-delta-1 lambda sweep.  GL005 and
        # GL010 are already part of LOW_CANOPY_CANDIDATES.
        candidates["D2_K2_GL000_HD1"] = {
            "drop_m": 2.0, "K": 2, "lambda_growth": 0.0, "huber_delta": 1.0,
        }
        candidates["D2_K2_GL0025_HD1"] = {
            "drop_m": 2.0, "K": 2, "lambda_growth": 0.025, "huber_delta": 1.0,
        }
        candidates["D2_K2_GL020_HD1"] = {
            "drop_m": 2.0, "K": 2, "lambda_growth": 0.20, "huber_delta": 1.0,
        }
        candidates["D2_K2_GL005_HD3"] = {
            "drop_m": 2.0, "K": 2, "lambda_growth": 0.05, "huber_delta": 3.0,
        }
    elif forest_key == "agadir":
        candidates["D2_K2_GL005_HD1"] = {
            "drop_m": 2.0, "K": 2, "lambda_growth": 0.05, "huber_delta": 1.0,
        }
    return candidates


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_growth_modules():
    root_text = str(GROWTH_ROOT)
    sys.path[:] = [entry for entry in sys.path if str(entry).rstrip("\\/").lower() != root_text.rstrip("\\/").lower()]
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    from src.ablation_training import set_seed
    from src.b4_echosat_adapter import B4EchoSatTwoHead, load_b4_reference
    from src.b4_sequence_dataset import B4SequenceCropDataset, build_same_month_sequences, load_catalogs
    from src.corrected_test_evaluator_v5 import evaluate_full_patch_temporal_nearest
    from src.persistent_training_v6 import prediction_head_sha256, train_phase1_direct_temporal_stage
    return {
        "set_seed": set_seed, "B4EchoSatTwoHead": B4EchoSatTwoHead,
        "load_b4_reference": load_b4_reference, "B4SequenceCropDataset": B4SequenceCropDataset,
        "build_same_month_sequences": build_same_month_sequences, "load_catalogs": load_catalogs,
        "evaluate_full_patch_temporal_nearest": evaluate_full_patch_temporal_nearest,
        "prediction_head_sha256": prediction_head_sha256,
        "train": train_phase1_direct_temporal_stage,
    }


def roots(forest_key: str) -> tuple[Path, Path]:
    cfg = FORESTS[forest_key]
    family = (
        "Ablations_Agadir_Confirmatory_LowLambda_v1"
        if forest_key == "agadir" and AGADIR_CONFIRMATORY_MODE
        else cfg["ablation_family"]
    )
    run_root = PROJECT / "Models" / cfg["ecosystem"] / cfg["label"] / "Phase2" / family
    result_root = PROJECT / "Results" / cfg["ecosystem"] / cfg["label"] / "Phase2" / family
    return run_root, result_root


def run_dir(forest_key: str, candidate_id: str) -> Path:
    return roots(forest_key)[0] / f"{FORESTS[forest_key]['label'].upper()}_B4_C15_{candidate_id}_SEED42"


def build_data(forest_key: str, modules: dict, include_test: bool = False):
    cfg = FORESTS[forest_key]
    samples, shots = modules["load_catalogs"](cfg["catalog"])
    splits = ("train", "val", "test") if include_test else ("train", "val")
    records = {
        split: modules["build_same_month_sequences"](
            samples, split=split, all_years=YEARS, window_length=4, leaf_on_months=MONTHS,
        )
        for split in splits
    }
    assert len(records["train"]) == cfg["expected_train_records"], (forest_key, len(records["train"]))
    assert len(records["val"]) == cfg["expected_val_records"], (forest_key, len(records["val"]))
    first = np.load(records["train"][0].x_paths[0], mmap_mode="r", allow_pickle=False)
    assert tuple(first.shape) == (512, 512, 15), first.shape
    del first
    return samples, shots, records


def fresh_model(cfg: dict, modules: dict):
    modules["set_seed"](SEED)
    reference = modules["load_b4_reference"](
        cfg["parent"], n_channels=15, base_ch=64, dropout=0.15, device=DEVICE,
    )
    modules["set_seed"](SEED)
    model = modules["B4EchoSatTwoHead"](reference, head_mode="residual", residual_scale=1.0).to(DEVICE)
    assert torch.count_nonzero(model.prediction_head[-1].weight).item() == 0
    assert torch.count_nonzero(model.prediction_head[-1].bias).item() == 0
    return model


def archive_nonresumable(directory: Path) -> None:
    if not directory.exists() or not any(directory.iterdir()):
        return
    if (directory / "TRAIN_DONE.json").is_file() or (directory / "checkpoints" / "last.ckpt").is_file():
        return
    archive = directory.with_name(directory.name + "__FAILED_PARTIAL_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    directory.rename(archive)
    print("[ARCHIVED NON-RESUMABLE]", archive, flush=True)


def train_candidate(forest_key: str, candidate_id: str, modules: dict) -> None:
    cfg = FORESTS[forest_key]
    candidate = candidates_for(forest_key)[candidate_id]
    directory = run_dir(forest_key, candidate_id)
    if (directory / "TRAIN_DONE.json").is_file():
        print("[REUSE COMPLETED]", directory, flush=True)
        return
    archive_nonresumable(directory)
    assert cfg["catalog"].is_dir() and cfg["parent"].is_file()
    assert sha256(cfg["parent"]) == cfg["parent_sha"]
    _, shots, records = build_data(forest_key, modules)
    train_dataset = modules["B4SequenceCropDataset"](
        records["train"], shots, crop_size=96, samples_per_epoch=264, drop_channels=(), seed=SEED,
        center_on_gedi=True, balanced_height_anchors=True, height_bins=cfg["height_bins"],
    )
    val_dataset = modules["B4SequenceCropDataset"](
        records["val"], shots, crop_size=96, samples_per_epoch=max(132, 4 * len(records["val"])),
        drop_channels=(), seed=SEED + 10_000, center_on_gedi=True,
        balanced_height_anchors=False, height_bins=cfg["height_bins"],
    )
    model = fresh_model(cfg, modules)
    print("\n" + "=" * 110, flush=True)
    print("ABLATION", forest_key, candidate_id, candidate, "device", DEVICE, flush=True)
    print("RUN", directory, flush=True)
    print("=" * 110, flush=True)
    modules["train"](
        model=model, train_dataset=train_dataset, val_dataset=val_dataset,
        official_repo=OFFICIAL_REPO, phase1_checkpoint=cfg["parent"], run_dir=directory,
        device=DEVICE, seed=SEED, batch_size=1,
        max_steps=(2500 if forest_key == "agadir" and AGADIR_CONFIRMATORY_MODE else cfg["max_steps"]),
        val_every_steps=66,
        patience_evals=(15 if forest_key == "agadir" and AGADIR_CONFIRMATORY_MODE else 20),
        learning_rate=(5e-5 if forest_key == "agadir" and AGADIR_CONFIRMATORY_MODE else 1e-4),
        weight_decay=5e-3,
        lambda_growth=candidate["lambda_growth"], lambda_slope=0.0, lambda_std=0.0,
        lambda_bias=0.0, lambda_anti_zero=0.0, anti_shrink_min_points=16,
        anti_zero_min_target=2.5, anti_zero_min_prediction=2.0, anti_zero_power=2.0,
        supervised_loss_name="huber", huber_delta=candidate.get("huber_delta", cfg["huber_delta"]), height_weight_mode="none",
        slope_min=0.0, slope_max=2.0, disturbance_indicator=-1.0,
        disturbance_rule="persistent_running_max", persistent_drop_m=candidate["drop_m"],
        persistent_required_consecutive_flags=candidate["K"], full_disturbance_window=True,
        min_height=cfg["eval_min"], max_height=cfg["eval_max"],
        # Low-canopy forests cannot satisfy the old slope>=0.65 gate (especially Agadir).
        # Keep selection driven by the explicit post-hoc VAL Pareto rule below.
        checkpoint_min_slope=0.0, checkpoint_min_std_ratio=0.0,
        checkpoint_max_std_ratio=2.0, checkpoint_max_abs_bias=5.0,
        warmup_cycles=3, plateau_patience=8, plateau_factor=0.5,
        lr_min=1e-6, grad_clip=1.0, allow_resume=True, reuse_completed=True,
    )
    del model, train_dataset, val_dataset, records, shots
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def checkpoint_metrics(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    metrics = state.get("metrics", {})
    return {
        "step": state.get("step"), "n": metrics.get("n"), "kge_2009": metrics.get("kge_2009"),
        "r2": metrics.get("r2"), "mae": metrics.get("mae"), "rmse": metrics.get("rmse"),
        "correlation": metrics.get("corr"), "slope": metrics.get("slope"),
        "std_ratio": metrics.get("std_ratio"), "bias": metrics.get("bias"),
        "article_compromise_score": metrics.get("article_compromise_score"),
    }


def rank_val(forest_key: str) -> pd.DataFrame:
    cfg = FORESTS[forest_key]
    candidate_specs = candidates_for(forest_key)
    _, result_root = roots(forest_key)
    result_root.mkdir(parents=True, exist_ok=True)
    rows = []
    include_historical_baseline = not (forest_key == "agadir" and AGADIR_CONFIRMATORY_MODE)
    baseline_ckpt = cfg["baseline_run"] / "checkpoints" / "best_compromise.ckpt"
    if include_historical_baseline and baseline_ckpt.is_file():
        rows.append({
            "candidate_id": "BASELINE_D5_K2_GL020", "drop_m": 5.0, "K": 2, "lambda_growth": 0.20,
            "huber_delta": cfg["huber_delta"],
            "checkpoint": str(baseline_ckpt), **checkpoint_metrics(baseline_ckpt),
        })
    for candidate_id, spec in candidate_specs.items():
        ckpt = run_dir(forest_key, candidate_id) / "checkpoints" / "best_compromise.ckpt"
        if not ckpt.is_file():
            print("[MISSING VAL CHECKPOINT]", ckpt, flush=True)
            continue
        rows.append({
            "candidate_id": candidate_id, **spec,
            "huber_delta": spec.get("huber_delta", cfg["huber_delta"]),
            "checkpoint": str(ckpt), **checkpoint_metrics(ckpt),
        })
    ranking = pd.DataFrame(rows)
    expected_rows = len(candidate_specs) + int(include_historical_baseline)
    if len(ranking) != expected_rows:
        raise RuntimeError(f"{forest_key}: ranking incomplete {len(ranking)}/{expected_rows}")
    numeric = ("r2", "mae", "rmse", "slope", "std_ratio", "bias", "kge_2009")
    for name in numeric:
        ranking[name] = pd.to_numeric(ranking[name], errors="raise")
    best_r2, best_mae = float(ranking.r2.max()), float(ranking.mae.min())
    ranking["accuracy_eligible"] = (
        (ranking.r2 >= best_r2 - 0.01)
        & (ranking.mae <= best_mae + cfg["mae_tolerance"])
    )
    ranking["selection_score"] = (
        (1.0 - ranking.slope).abs()
        + (1.0 - ranking.std_ratio).abs()
        + 0.25 * ranking.bias.abs()
        + 0.25 * (ranking.mae - best_mae).clip(lower=0)
        + 0.25 * (best_r2 - ranking.r2).clip(lower=0)
    )
    ranking["accepted_vs_control"] = ranking["accuracy_eligible"]
    ranking["delta_r2_vs_control"] = np.nan
    ranking["mae_gain_vs_control_m"] = np.nan
    ranking["rmse_gain_vs_control_m"] = np.nan
    ranking["delta_slope_vs_control"] = np.nan
    ranking["abs_bias_gain_vs_control_m"] = np.nan
    if forest_key == "agadir" and AGADIR_CONFIRMATORY_MODE:
        control = ranking.loc[ranking.candidate_id.eq("AG_CTRL_GL000")].iloc[0]
        ranking["delta_r2_vs_control"] = ranking.r2 - float(control.r2)
        ranking["mae_gain_vs_control_m"] = float(control.mae) - ranking.mae
        ranking["rmse_gain_vs_control_m"] = float(control.rmse) - ranking.rmse
        ranking["delta_slope_vs_control"] = ranking.slope - float(control.slope)
        ranking["abs_bias_gain_vs_control_m"] = abs(float(control.bias)) - ranking.bias.abs()
        is_growthloss = ~ranking.candidate_id.eq("AG_CTRL_GL000")
        ranking["accepted_vs_control"] = (
            is_growthloss
            & (ranking.r2 >= float(control.r2) - 0.002)
            & (ranking.mae <= float(control.mae) + 0.01)
            & (ranking.rmse <= float(control.rmse) + 0.01)
            & (ranking.slope >= float(control.slope) + 0.02)
            & (ranking.bias.abs() <= abs(float(control.bias)) + 0.05)
        )
    ranking["val_rank"] = np.nan
    eligible = ranking.loc[ranking.accepted_vs_control].sort_values(
        ["selection_score", "mae", "rmse"], ascending=[True, True, True], kind="stable"
    )
    if eligible.empty and forest_key == "agadir" and AGADIR_CONFIRMATORY_MODE:
        eligible = ranking.loc[ranking.candidate_id.eq("AG_CTRL_GL000")]
    ranking.loc[eligible.index, "val_rank"] = np.arange(1, len(eligible) + 1)
    ranking = ranking.sort_values(["accuracy_eligible", "val_rank", "selection_score"], ascending=[False, True, True])
    winner = ranking.loc[ranking.val_rank.eq(1)].iloc[0]
    # A GrowthLoss is accepted only when the VAL-selected winner actually
    # carries a strictly positive growth coefficient. This also handles
    # matched GL000 controls outside the Agadir confirmatory protocol.
    growthloss_accepted = float(winner.lambda_growth) > 0.0
    selected = {
        "forest": cfg["label"], "selection_split": "VAL only", "test_used_for_selection": False,
        "rule": (
            "Against matched GL000 control on VAL: delta R2 >= -0.002, delta MAE <= +0.01 m, "
            "delta RMSE <= +0.01 m, delta slope >= +0.02, delta abs(bias) <= +0.05 m; "
            "then minimize slope/std/bias compromise"
            if forest_key == "agadir" and AGADIR_CONFIRMATORY_MODE
            else f"R2 within 0.01 of best and MAE within {cfg['mae_tolerance']:.2f} m of best; then minimize slope/std/bias compromise"
        ),
        "growthloss_accepted_on_val": growthloss_accepted,
        "candidate_id": winner.candidate_id, "checkpoint": winner.checkpoint,
        "checkpoint_sha256": sha256(Path(winner.checkpoint)),
        "metrics": {name: float(winner[name]) for name in numeric},
    }
    ranking.to_csv(result_root / "val_ranking_low_canopy_ablations.csv", index=False)
    (result_root / "selected_on_val.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print("\nVAL RANKING", cfg["label"], flush=True)
    columns = ["val_rank", "accuracy_eligible", "accepted_vs_control", "candidate_id", "drop_m", "K", "lambda_growth", "huber_delta", "r2", "mae", "rmse", "slope", "std_ratio", "bias", "kge_2009", "selection_score"]
    if forest_key == "agadir" and AGADIR_CONFIRMATORY_MODE:
        columns += ["delta_r2_vs_control", "mae_gain_vs_control_m", "rmse_gain_vs_control_m", "delta_slope_vs_control", "abs_bias_gain_vs_control_m"]
    print(ranking[columns].to_string(index=False), flush=True)
    print("[VAL WINNER]", json.dumps(selected, indent=2), flush=True)
    return ranking


def metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    corr = float(np.corrcoef(y, pred)[0, 1])
    slope = float(np.polyfit(y, pred, 1)[0])
    std_ratio = float(np.std(pred) / np.std(y))
    mean_ratio = float(np.mean(pred) / np.mean(y))
    return {
        "n": int(len(y)), "r2": float(1 - np.sum(err**2) / np.sum((y-y.mean())**2)),
        "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2))),
        "correlation": corr, "slope": slope, "std_ratio": std_ratio,
        "bias": float(np.mean(err)),
        "kge_2009": float(1 - np.sqrt((corr-1)**2 + (std_ratio-1)**2 + (mean_ratio-1)**2)),
    }


def evaluate_selected(forest_key: str, modules: dict) -> None:
    cfg = FORESTS[forest_key]
    _, result_root = roots(forest_key)
    selected_path = result_root / "selected_on_val.json"
    if not selected_path.is_file():
        raise FileNotFoundError("Run VAL ranking first: " + str(selected_path))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    checkpoint = Path(selected["checkpoint"])
    assert sha256(checkpoint) == selected["checkpoint_sha256"]
    _, shots, records = build_data(forest_key, modules, include_test=True)
    model = fresh_model(cfg, modules)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.prediction_head.load_state_dict(state["prediction_head"], strict=True)
    model.eval()
    raw, nearest = modules["evaluate_full_patch_temporal_nearest"](
        model=model, records=records["test"], shots=shots, device=DEVICE, split="test",
        drop_channels=(), min_height=cfg["eval_min"], max_height=cfg["eval_max"], progress_every=1,
    )
    nearest["aux_shot_uid"] = nearest["aux_shot_uid"].astype(str)
    assert nearest.aux_shot_uid.is_unique
    assert len(nearest) == cfg["expected_test_n"], (forest_key, len(nearest), cfg["expected_test_n"])
    phase1 = metrics(nearest.rh95, nearest.pred_off_reference)
    phase2 = metrics(nearest.rh95, nearest.pred_on_growthloss)
    table = pd.DataFrame([
        {"phase": "Phase 1", "candidate_id": "frozen_parent", **phase1},
        {"phase": "Phase 2", "candidate_id": selected["candidate_id"], **phase2},
    ])
    gains = {
        "forest": cfg["label"], "selected_on": "VAL only", "candidate_id": selected["candidate_id"],
        "delta_r2": phase2["r2"] - phase1["r2"], "mae_gain_m": phase1["mae"] - phase2["mae"],
        "rmse_gain_m": phase1["rmse"] - phase2["rmse"], "delta_slope": phase2["slope"] - phase1["slope"],
        "delta_std_ratio": phase2["std_ratio"] - phase1["std_ratio"],
        "delta_kge": phase2["kge_2009"] - phase1["kge_2009"],
    }
    test_root = result_root / "TEST_selected_on_VAL"
    test_root.mkdir(parents=True, exist_ok=True)
    raw.to_csv(test_root / "occurrences.csv.gz", index=False, compression="gzip")
    nearest.to_csv(test_root / "unique_nearest_predictions.csv.gz", index=False, compression="gzip")
    table.to_csv(test_root / "phase1_vs_selected_phase2_metrics.csv", index=False)
    pd.DataFrame([gains]).to_csv(test_root / "phase1_to_selected_phase2_gains.csv", index=False)
    protocol = {**selected, "test_used_for_selection": False, "test_n": len(nearest), "test_gains": gains}
    (test_root / "test_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    lo, hi = cfg["eval_min"], cfg["eval_max"]
    for axis, row, column in zip(axes, table.to_dict("records"), ("pred_off_reference", "pred_on_growthloss")):
        y = nearest.rh95.to_numpy(float); pred = nearest[column].to_numpy(float)
        density = axis.hexbin(y, pred, gridsize=38, extent=(0, hi, 0, hi), mincnt=1, bins="log", cmap="viridis")
        axis.plot([0, hi], [0, hi], "--", color="black", linewidth=1.2)
        fit = np.polyfit(y, pred, 1); xx = np.asarray([lo, hi])
        axis.plot(xx, fit[0]*xx+fit[1], color="crimson", linewidth=1.5)
        axis.set(xlim=(0,hi), ylim=(0,hi), xlabel="GEDI RH95 observé (m)")
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.18)
        axis.set_title(f"{row['phase']} | n={row['n']}", fontweight="bold")
        axis.text(.03,.97, f"R²={row['r2']:.4f} | KGE={row['kge_2009']:.4f}\nMAE={row['mae']:.4f} | RMSE={row['rmse']:.4f}\nslope={row['slope']:.4f} | std={row['std_ratio']:.4f}\nbias={row['bias']:+.4f}", transform=axis.transAxes, va="top", fontsize=9, bbox={"boxstyle":"round","facecolor":"white","alpha":.9})
        fig.colorbar(density, ax=axis, fraction=.046, pad=.03, label="log10(N)")
    axes[0].set_ylabel("Hauteur prédite (m)")
    fig.suptitle(f"{cfg['label']} — gagnant VAL {selected['candidate_id']} — TEST figé", fontweight="bold")
    fig.tight_layout()
    fig.savefig(test_root / "scatter_phase1_vs_selected_phase2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("\nTEST SELECTED-ON-VAL", cfg["label"], flush=True)
    print(table.to_string(index=False), flush=True)
    print("GAINS", json.dumps(gains, indent=2), flush=True)


def evaluate_all_ablations_on_test(forest_key: str, modules: dict) -> None:
    """Descriptive TEST audit of every VAL checkpoint on one immutable TEST split.

    This deliberately does not overwrite ``selected_on_val.json``.  The resulting
    TEST ranks are exploratory diagnostics only and must not be used as the
    publication selection rule.
    """
    cfg = FORESTS[forest_key]
    candidate_specs = candidates_for(forest_key)
    _, result_root = roots(forest_key)
    ranking_path = result_root / "val_ranking_low_canopy_ablations.csv"
    selected_path = result_root / "selected_on_val.json"
    if not ranking_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError("Run VAL ranking first: " + str(ranking_path))

    ranking = pd.read_csv(ranking_path)
    include_historical_baseline = not (forest_key == "agadir" and AGADIR_CONFIRMATORY_MODE)
    expected_ids = set(candidate_specs)
    if include_historical_baseline:
        expected_ids.add("BASELINE_D5_K2_GL020")
    if set(ranking["candidate_id"].astype(str)) != expected_ids:
        raise RuntimeError(
            f"{forest_key}: incomplete VAL candidate set: "
            f"{sorted(set(ranking['candidate_id'].astype(str)))}"
        )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    test_root = result_root / "TEST_all_VAL_checkpoints_EXPLORATORY"
    prediction_root = test_root / "Predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)

    _, shots, records = build_data(forest_key, modules, include_test=True)
    metric_rows, gain_rows = [], []
    canonical_uid = canonical_y = canonical_phase1 = None

    # Stable order: official Phase-1 reference, old D5 baseline, then the six new runs.
    ordered_ids = list(candidate_specs.keys())
    if include_historical_baseline:
        ordered_ids.insert(0, "BASELINE_D5_K2_GL020")
    by_id = ranking.set_index("candidate_id", drop=False)
    for candidate_id in ordered_ids:
        val_row = by_id.loc[candidate_id]
        checkpoint = Path(val_row["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hash = sha256(checkpoint)
        prediction_path = prediction_root / f"{candidate_id}_unique_nearest.csv.gz"
        lineage_path = prediction_root / f"{candidate_id}_lineage.json"
        reusable = False
        if prediction_path.is_file() and lineage_path.is_file():
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            reusable = lineage.get("checkpoint_sha256") == checkpoint_hash
        if reusable:
            nearest = pd.read_csv(prediction_path)
            print("[REUSE TEST ALL]", forest_key, candidate_id, prediction_path, flush=True)
        else:
            model = fresh_model(cfg, modules)
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(checkpoint, map_location="cpu")
            model.prediction_head.load_state_dict(state["prediction_head"], strict=True)
            model.eval()
            _, nearest = modules["evaluate_full_patch_temporal_nearest"](
                model=model, records=records["test"], shots=shots, device=DEVICE, split="test",
                drop_channels=(), min_height=cfg["eval_min"], max_height=cfg["eval_max"], progress_every=1,
            )
            nearest.to_csv(prediction_path, index=False, compression="gzip")
            lineage_path.write_text(json.dumps({
                "forest": cfg["label"], "candidate_id": candidate_id,
                "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                "split": "TEST", "selection_authority": False,
            }, indent=2), encoding="utf-8")
            del model, state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        nearest["aux_shot_uid"] = nearest["aux_shot_uid"].astype(str)
        assert nearest["aux_shot_uid"].is_unique
        assert len(nearest) == cfg["expected_test_n"], (forest_key, candidate_id, len(nearest))
        uid = nearest["aux_shot_uid"].to_numpy(str)
        y = nearest["rh95"].to_numpy(float)
        phase1_pred = nearest["pred_off_reference"].to_numpy(float)
        phase2_pred = nearest["pred_on_growthloss"].to_numpy(float)
        if canonical_uid is None:
            canonical_uid, canonical_y, canonical_phase1 = uid, y, phase1_pred
            phase1_metrics = metrics(canonical_y, canonical_phase1)
            metric_rows.append({
                "phase": "Phase 1", "candidate_id": "frozen_parent",
                "drop_m": np.nan, "K": np.nan, "lambda_growth": 0.0,
                "selected_on_val": False, **phase1_metrics,
            })
        else:
            assert np.array_equal(uid, canonical_uid), f"{forest_key}/{candidate_id}: TEST UID mismatch"
            assert np.allclose(y, canonical_y, rtol=0, atol=1e-9)
            assert np.allclose(phase1_pred, canonical_phase1, rtol=0, atol=1e-6)

        current = metrics(canonical_y, phase2_pred)
        is_selected = candidate_id == selected["candidate_id"]
        metric_rows.append({
            "phase": "Phase 2", "candidate_id": candidate_id,
            "drop_m": float(val_row["drop_m"]), "K": int(val_row["K"]),
            "lambda_growth": float(val_row["lambda_growth"]),
            "selected_on_val": is_selected, **current,
        })
        gain_rows.append({
            "forest": cfg["label"], "candidate_id": candidate_id,
            "selected_on_val": is_selected,
            "delta_r2": current["r2"] - phase1_metrics["r2"],
            "mae_gain_m": phase1_metrics["mae"] - current["mae"],
            "rmse_gain_m": phase1_metrics["rmse"] - current["rmse"],
            "delta_slope": current["slope"] - phase1_metrics["slope"],
            "delta_std_ratio": current["std_ratio"] - phase1_metrics["std_ratio"],
            "delta_kge": current["kge_2009"] - phase1_metrics["kge_2009"],
        })

    table = pd.DataFrame(metric_rows)
    gains = pd.DataFrame(gain_rows)
    phase2_mask = table["phase"].eq("Phase 2")
    table.loc[phase2_mask, "test_rank_r2_exploratory"] = (
        table.loc[phase2_mask, "r2"].rank(method="min", ascending=False).astype(int)
    )
    table.loc[phase2_mask, "test_rank_mae_exploratory"] = (
        table.loc[phase2_mask, "mae"].rank(method="min", ascending=True).astype(int)
    )
    table.to_csv(test_root / "all_ablations_same_test_metrics.csv", index=False)
    gains.to_csv(test_root / "all_ablations_same_test_gains.csv", index=False)
    (test_root / "README_PROTOCOL.json").write_text(json.dumps({
        "forest": cfg["label"], "split": "same immutable TEST",
        "test_n": int(cfg["expected_test_n"]),
        "official_candidate_selected_on_val": selected["candidate_id"],
        "warning": "Exploratory TEST comparison only. Never replace selected_on_val.json from this table.",
    }, indent=2), encoding="utf-8")

    n_panels = 1 + len(ordered_ids)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 4.7 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(nrows, ncols)
    hi = cfg["eval_max"]
    plot_specs = [("Phase 1", "frozen_parent", canonical_phase1)]
    for candidate_id in ordered_ids:
        nearest = pd.read_csv(prediction_root / f"{candidate_id}_unique_nearest.csv.gz")
        plot_specs.append(("Phase 2", candidate_id, nearest["pred_on_growthloss"].to_numpy(float)))
    for axis, (phase, candidate_id, pred) in zip(axes.ravel(), plot_specs):
        row = table.loc[table["candidate_id"].eq(candidate_id)].iloc[0]
        density = axis.hexbin(canonical_y, pred, gridsize=34, extent=(0,hi,0,hi), mincnt=1, bins="log", cmap="viridis")
        axis.plot([0,hi], [0,hi], "--", color="black", linewidth=1.0)
        fit = np.polyfit(canonical_y, pred, 1); xx = np.asarray([cfg["eval_min"],hi])
        axis.plot(xx, fit[0]*xx+fit[1], color="crimson", linewidth=1.3)
        selected_tag = " | VAL WINNER" if bool(row["selected_on_val"]) else ""
        axis.set_title(f"{candidate_id}{selected_tag}", fontsize=10, fontweight="bold")
        axis.text(.03,.97, f"R²={row.r2:.4f} | MAE={row.mae:.4f}\nRMSE={row.rmse:.4f} | slope={row.slope:.4f}\nstd={row.std_ratio:.4f} | KGE={row.kge_2009:.4f}", transform=axis.transAxes, va="top", fontsize=8, bbox={"boxstyle":"round","facecolor":"white","alpha":.88})
        axis.set(xlim=(0,hi), ylim=(0,hi)); axis.grid(alpha=.15)
    for axis in axes.ravel()[len(plot_specs):]:
        axis.axis("off")
    for axis in axes[-1, :]: axis.set_xlabel("GEDI RH95 observé (m)")
    for axis in axes[:, 0]: axis.set_ylabel("Hauteur prédite (m)")
    fig.suptitle(f"{cfg['label']} — toutes les ablations sur le même TEST (exploratoire)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(test_root / "scatter_all_ablations_same_test_EXPLORATORY.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("\nTEST ALL ABLATIONS — EXPLORATORY ONLY", cfg["label"], flush=True)
    print(table.to_string(index=False), flush=True)
    print("\nGAINS VS PHASE 1", flush=True)
    print(gains.to_string(index=False), flush=True)


def evaluate_all_checkpoint_variants_on_test(forest_key: str, modules: dict) -> None:
    """Evaluate every available checkpoint variant on one immutable TEST split.

    Variants are compared descriptively.  Missing ``best_r2.ckpt`` files from
    legacy runs are reported, because they cannot be reconstructed from a CSV
    history after the corresponding model weights have been discarded.
    """
    cfg = FORESTS[forest_key]
    _, result_root = roots(forest_key)
    ranking_path = result_root / "val_ranking_low_canopy_ablations.csv"
    selected_path = result_root / "selected_on_val.json"
    if not ranking_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError("Run VAL ranking first: " + str(ranking_path))
    ranking = pd.read_csv(ranking_path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    test_root = result_root / "TEST_all_checkpoint_variants_EXPLORATORY"
    prediction_root = test_root / "Predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)

    _, shots, records = build_data(forest_key, modules, include_test=True)
    variants = ("best", "best_compromise", "best_slope", "best_any", "best_r2")
    rows, missing_rows = [], []
    canonical_uid = canonical_y = canonical_phase1 = None
    prediction_by_sha = {}
    by_id = ranking.set_index("candidate_id", drop=False)
    ordered_ids = list(candidates_for(forest_key).keys())
    if not (forest_key == "agadir" and AGADIR_CONFIRMATORY_MODE):
        ordered_ids.insert(0, "BASELINE_D5_K2_GL020")

    for candidate_id in ordered_ids:
        val_row = by_id.loc[candidate_id]
        checkpoint_dir = Path(val_row["checkpoint"]).parent
        for variant in variants:
            checkpoint = checkpoint_dir / f"{variant}.ckpt"
            if not checkpoint.is_file():
                missing_rows.append({
                    "forest": cfg["label"], "candidate_id": candidate_id,
                    "checkpoint_variant": variant, "status": "MISSING_NOT_SAVED",
                })
                continue
            checkpoint_hash = sha256(checkpoint)
            cache_key = f"{candidate_id}__{variant}"
            prediction_path = prediction_root / f"{cache_key}_unique_nearest.csv.gz"
            lineage_path = prediction_root / f"{cache_key}_lineage.json"
            if checkpoint_hash in prediction_by_sha:
                nearest = prediction_by_sha[checkpoint_hash].copy()
                print("[DEDUP SHA TEST]", forest_key, candidate_id, variant, checkpoint_hash[:12], flush=True)
            else:
                reusable = False
                if prediction_path.is_file() and lineage_path.is_file():
                    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
                    reusable = lineage.get("checkpoint_sha256") == checkpoint_hash
                if reusable:
                    nearest = pd.read_csv(prediction_path)
                    print("[REUSE CHECKPOINT TEST]", forest_key, candidate_id, variant, flush=True)
                else:
                    model = fresh_model(cfg, modules)
                    try:
                        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    except TypeError:
                        state = torch.load(checkpoint, map_location="cpu")
                    model.prediction_head.load_state_dict(state["prediction_head"], strict=True)
                    model.eval()
                    _, nearest = modules["evaluate_full_patch_temporal_nearest"](
                        model=model, records=records["test"], shots=shots, device=DEVICE, split="test",
                        drop_channels=(), min_height=cfg["eval_min"], max_height=cfg["eval_max"],
                        progress_every=1,
                    )
                    nearest.to_csv(prediction_path, index=False, compression="gzip")
                    lineage_path.write_text(json.dumps({
                        "forest": cfg["label"], "candidate_id": candidate_id,
                        "checkpoint_variant": variant, "checkpoint": str(checkpoint),
                        "checkpoint_sha256": checkpoint_hash, "split": "TEST",
                        "selection_authority": False,
                    }, indent=2), encoding="utf-8")
                    del model, state
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                prediction_by_sha[checkpoint_hash] = nearest.copy()

            nearest["aux_shot_uid"] = nearest["aux_shot_uid"].astype(str)
            assert nearest["aux_shot_uid"].is_unique
            assert len(nearest) == cfg["expected_test_n"]
            uid = nearest["aux_shot_uid"].to_numpy(str)
            y = nearest["rh95"].to_numpy(float)
            phase1_pred = nearest["pred_off_reference"].to_numpy(float)
            phase2_pred = nearest["pred_on_growthloss"].to_numpy(float)
            if canonical_uid is None:
                canonical_uid, canonical_y, canonical_phase1 = uid, y, phase1_pred
            else:
                assert np.array_equal(uid, canonical_uid)
                assert np.allclose(y, canonical_y, rtol=0, atol=1e-9)
                assert np.allclose(phase1_pred, canonical_phase1, rtol=0, atol=1e-6)
            val_metrics = checkpoint_metrics(checkpoint)
            rows.append({
                "forest": cfg["label"], "candidate_id": candidate_id,
                "checkpoint_variant": variant, "checkpoint_sha256": checkpoint_hash,
                "same_weights_as_another_variant": sum(
                    row.get("checkpoint_sha256") == checkpoint_hash for row in rows
                ) > 0,
                "selected_candidate_on_val": candidate_id == selected["candidate_id"],
                "val_r2": val_metrics.get("r2"), "val_mae": val_metrics.get("mae"),
                "val_slope": val_metrics.get("slope"),
                **metrics(canonical_y, phase2_pred),
            })

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(f"{forest_key}: no checkpoint variant evaluated")
    table["test_rank_r2_exploratory"] = table["r2"].rank(method="min", ascending=False).astype(int)
    table["test_rank_mae_exploratory"] = table["mae"].rank(method="min", ascending=True).astype(int)
    phase1_metrics = metrics(canonical_y, canonical_phase1)
    table["delta_r2_vs_phase1"] = table["r2"] - phase1_metrics["r2"]
    table["mae_gain_vs_phase1_m"] = phase1_metrics["mae"] - table["mae"]
    table["delta_slope_vs_phase1"] = table["slope"] - phase1_metrics["slope"]
    table.to_csv(test_root / "all_checkpoint_variants_same_test_metrics.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(test_root / "missing_checkpoint_variants.csv", index=False)
    (test_root / "README_PROTOCOL.json").write_text(json.dumps({
        "forest": cfg["label"], "split": "same immutable TEST",
        "test_n": cfg["expected_test_n"], "official_candidate_selected_on_val": selected["candidate_id"],
        "warning": "Exploratory checkpoint audit only; TEST must not replace VAL selection.",
        "best_r2_note": "Available only for runs trained after best_r2 checkpoint support was added.",
    }, indent=2), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(11, 7))
    scatter = axis.scatter(table["mae"], table["r2"], c=table["slope"], s=80,
                           cmap="viridis", edgecolor="black", linewidth=.4)
    for row in table.itertuples():
        axis.annotate(f"{row.candidate_id}/{row.checkpoint_variant}", (row.mae, row.r2),
                      xytext=(4, 4), textcoords="offset points", fontsize=6)
    axis.set_xlabel("MAE TEST (m) — plus petit = meilleur")
    axis.set_ylabel("R² TEST — plus grand = meilleur")
    axis.set_title(f"{cfg['label']} — variantes de checkpoints sur le même TEST (exploratoire)")
    axis.grid(alpha=.2); fig.colorbar(scatter, ax=axis, label="Slope TEST")
    fig.tight_layout(); fig.savefig(test_root / "checkpoint_variants_test_r2_mae_slope.png", dpi=220)
    plt.close(fig)
    print("\nTEST CHECKPOINT VARIANTS — EXPLORATORY ONLY", cfg["label"], flush=True)
    print(table.sort_values(["test_rank_r2_exploratory", "test_rank_mae_exploratory"]).to_string(index=False), flush=True)
    if missing_rows:
        print("\nMISSING LEGACY VARIANTS", flush=True)
        print(pd.DataFrame(missing_rows).to_string(index=False), flush=True)


def main() -> None:
    global AGADIR_CONFIRMATORY_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--forest", choices=("ifran", "maamoura", "agadir", "all"), default="all")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--rank-val", action="store_true")
    parser.add_argument("--test-winner", action="store_true")
    parser.add_argument("--test-all", action="store_true")
    parser.add_argument("--test-checkpoints", action="store_true")
    parser.add_argument(
        "--agadir-confirmatory", action="store_true",
        help="Use the isolated Agadir low-lambda confirmatory grid and result family.",
    )
    args = parser.parse_args()
    AGADIR_CONFIRMATORY_MODE = bool(args.agadir_confirmatory)
    if AGADIR_CONFIRMATORY_MODE and args.forest not in ("agadir", "all"):
        parser.error("--agadir-confirmatory is only valid with --forest agadir or all")
    if not (args.train or args.rank_val or args.test_winner or args.test_all or args.test_checkpoints):
        parser.error("Select at least one action")
    modules = import_growth_modules()
    forest_keys = ("ifran", "maamoura", "agadir") if args.forest == "all" else (args.forest,)
    for forest_key in forest_keys:
        if args.train:
            for candidate_id in candidates_for(forest_key):
                train_candidate(forest_key, candidate_id, modules)
        if args.rank_val:
            rank_val(forest_key)
        if args.test_winner:
            evaluate_selected(forest_key, modules)
        if args.test_all:
            evaluate_all_ablations_on_test(forest_key, modules)
        if args.test_checkpoints:
            evaluate_all_checkpoint_variants_on_test(forest_key, modules)


if __name__ == "__main__":
    main()
