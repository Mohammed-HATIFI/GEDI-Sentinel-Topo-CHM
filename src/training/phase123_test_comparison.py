from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .b4_sequence_dataset import build_same_month_sequences, load_catalogs
from .corrected_test_evaluator_v5 import (
    canonical_step07_unique_nearest,
    evaluate_full_patch_temporal_nearest,
)
from .p2a_echosat_adapter import P2AEchoSatTwoHead, load_p2a_reference


DATASET_ROOT = Path(
    r"E:\shifted_Pauls_Article_Ifran\Output\Sample_Store_NPY\STEP04_IFRAN6_C11_PALSAR_AOIMASK_SPATIAL_NOLEAK_MAX45_75_15_10_NODUP_SANSNPZ_v36\DATASET_NPY_CATALOG_PATCH512_STRIDE512_TW180_GROWTHLOSS_S2_REDEDGE_ROBUSTNORM_C17_DEMSLOPE_RH0_42_ELIG_GT5_GE8M10PCT_S2V90_SPATIAL75_15_10_HEIGHTBAL_SEED42_v2"
)
PHASE1_RUN = Path(
    r"C:\Users\Dell\Desktop\Safi_3\Last_Ablation_Output\Runs_NoPalsar_Cumulative_B2_to_B5\DATASET_NPY_CATALOG_PATCH512_STRIDE512_TW180_GROWTHLOSS_S2_REDEDGE_ROBUSTNORM_C17_DEMSLOPE_RH0_42_ELIG_GT5_GE8M10PCT_S2V90_SPATIAL75_15_10_HEIGHTBAL_SEED42_v2\IFRAN6_B4_NOPALSAR_DEMSLOPE_REDEDGE_C15_PHASE1_HUBER3_P10_SEED42"
)
PIPELINE_ROOT = Path(
    r"C:\Users\Dell\Desktop\Publication_Inchallah\Phase2_P2A_then_Phase3_RAW_GL020"
)
PHASE2_RUN = (
    PIPELINE_ROOT
    / "runs_phase2"
    / DATASET_ROOT.name
    / "IFRAN6_B4_NOPALSAR_DEMSLOPE_REDEDGE_C15_PHASE1_HUBER3_P10_SEED42_P2A_TARGETED_ASYMHUBER_P8"
)
PHASE3_RUN = (
    PIPELINE_ROOT
    / "runs_phase3"
    / "P3_T4_PERSISTENT_RAW_DROP5_K2_GL020_FROM_P2A"
)
PHASE1_TEST_PREDICTIONS = (
    PHASE1_RUN
    / "step07_original_coords_minmax"
    / "best"
    / "test_predictions_original_coords.csv.gz"
)
PHASE2_BEST = PHASE2_RUN / "checkpoints" / "best.ckpt"
PHASE3_BEST = PHASE3_RUN / "checkpoints" / "best.ckpt"
OUTPUT_DIR = PIPELINE_ROOT / "comparison_test_common_w3"
EVAL_MIN = 2.5
EVAL_MAX = 45.0


def _metrics(true: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    true = np.asarray(true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(true) & np.isfinite(prediction)
    true = true[valid]
    prediction = prediction[valid]
    residual = prediction - true
    corr = float(np.corrcoef(true, prediction)[0, 1]) if len(true) > 1 else np.nan
    slope = float(np.polyfit(true, prediction, 1)[0]) if len(true) > 1 else np.nan
    true_std = float(np.std(true))
    pred_std = float(np.std(prediction))
    std_ratio = pred_std / true_std if true_std > 0 else np.nan
    mean_ratio = float(np.mean(prediction) / np.mean(true)) if np.mean(true) != 0 else np.nan
    kge = 1.0 - np.sqrt(
        (corr - 1.0) ** 2 + (std_ratio - 1.0) ** 2 + (mean_ratio - 1.0) ** 2
    )
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else np.nan
    return {
        "n": int(len(true)),
        "kge_2009": float(kge),
        "corr": corr,
        "r2": r2,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "slope": slope,
        "std_ratio": std_ratio,
        "pred_mean": float(np.mean(prediction)),
        "pred_std": pred_std,
    }


def _load_phase3_model(device: str) -> P2AEchoSatTwoHead:
    reference = load_p2a_reference(
        PHASE2_BEST,
        n_channels=15,
        base_ch=64,
        dropout=0.15,
        hidden_ch=32,
        gate_bias=-2.0,
        high_threshold=20.0,
        device=device,
    )
    model = P2AEchoSatTwoHead(reference, head_mode="residual", residual_scale=1.0)
    raw = torch.load(PHASE3_BEST, map_location="cpu")
    if "prediction_head" not in raw:
        raise KeyError(f"prediction_head absent de {PHASE3_BEST}")
    model.prediction_head.load_state_dict(raw["prediction_head"], strict=True)
    model.to(device).eval()
    return model


def _temporal_predictions(device: str, reuse: bool) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nearest_path = OUTPUT_DIR / "phase2_phase3_test_nearest_w3_2p5_45.csv.gz"
    raw_path = OUTPUT_DIR / "phase2_phase3_test_occurrences_w3_2p5_45.csv.gz"
    if reuse and nearest_path.exists():
        print("[REUSE] Prédictions temporelles :", nearest_path, flush=True)
        return pd.read_csv(nearest_path, dtype={"aux_shot_uid": str})

    samples, shots = load_catalogs(DATASET_ROOT)
    records = build_same_month_sequences(
        samples,
        split="test",
        all_years=tuple(range(2019, 2026)),
        window_length=3,
        leaf_on_months=(5, 6, 7, 8, 9),
    )
    model = _load_phase3_model(device)
    raw, nearest = evaluate_full_patch_temporal_nearest(
        model=model,
        records=records,
        shots=shots,
        device=device,
        split="test",
        drop_channels=(12, 13),
        min_height=EVAL_MIN,
        max_height=EVAL_MAX,
        progress_every=1,
    )
    raw.to_csv(raw_path, index=False, compression="gzip")
    nearest.to_csv(nearest_path, index=False, compression="gzip")
    print("[SAVED] Prédictions temporelles :", nearest_path, flush=True)
    return nearest


def _common_predictions(device: str, reuse: bool) -> pd.DataFrame:
    temporal = _temporal_predictions(device, reuse)
    temporal["aux_shot_uid"] = temporal["aux_shot_uid"].astype(str)

    phase1 = canonical_step07_unique_nearest(
        PHASE1_TEST_PREDICTIONS,
        min_height=EVAL_MIN,
        max_height=EVAL_MAX,
    ).copy()
    phase1["aux_shot_uid"] = phase1["aux_shot_uid"].astype(str)
    phase1 = phase1[
        ["aux_shot_uid", "rh95", "abs_temporal_delta_days", "prediction_original_coords"]
    ].rename(
        columns={
            "rh95": "rh95_phase1",
            "abs_temporal_delta_days": "delta_phase1",
            "prediction_original_coords": "prediction_phase1",
        }
    )

    common = temporal.merge(phase1, on="aux_shot_uid", how="inner", validate="one_to_one")
    common["rh95"] = pd.to_numeric(common["rh95"], errors="coerce")
    common["rh95_phase1"] = pd.to_numeric(common["rh95_phase1"], errors="coerce")
    if not np.allclose(common["rh95"], common["rh95_phase1"], atol=1e-5, rtol=0):
        raise AssertionError("RH95 diffère entre STEP07 Phase 1 et l’évaluation temporelle.")
    common = common.rename(
        columns={
            "pred_off_reference": "prediction_phase2",
            "pred_on_growthloss": "prediction_phase3",
        }
    )
    keep = [
        "aux_shot_uid",
        "rh95",
        "prediction_phase1",
        "prediction_phase2",
        "prediction_phase3",
        "abs_temporal_delta_days",
        "delta_phase1",
    ]
    common = common[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if not common["aux_shot_uid"].is_unique:
        raise AssertionError("Le support commun doit avoir un seul point par GEDI.")
    common.to_csv(
        OUTPUT_DIR / "phase123_predictions_test_common_w3_n4304.csv.gz",
        index=False,
        compression="gzip",
    )
    return common


def _gain_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    transitions = [(0, 1, "Phase 1 → Phase 2"), (1, 2, "Phase 2 → Phase 3")]
    for old_index, new_index, label in transitions:
        old = metrics.iloc[old_index]
        new = metrics.iloc[new_index]
        rows.append(
            {
                "transition": label,
                "delta_kge": new.kge_2009 - old.kge_2009,
                "delta_corr": new["corr"] - old["corr"],
                "delta_r2": new.r2 - old.r2,
                "mae_gain_m": old.mae - new.mae,
                "mae_gain_pct": 100.0 * (old.mae - new.mae) / old.mae,
                "rmse_gain_m": old.rmse - new.rmse,
                "rmse_gain_pct": 100.0 * (old.rmse - new.rmse) / old.rmse,
                "abs_bias_gain_m": abs(old.bias) - abs(new.bias),
                "slope_gap_gain": abs(1.0 - old.slope) - abs(1.0 - new.slope),
                "std_gap_gain": abs(1.0 - old.std_ratio) - abs(1.0 - new.std_ratio),
            }
        )
    return pd.DataFrame(rows)


def _scatter_figure(common: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    phases = [
        ("Phase 1 — B4 spatial", "prediction_phase1"),
        ("Phase 2 — P2A anti-shrink", "prediction_phase2"),
        ("Phase 3 — RAW DROP5 K2 GL020", "prediction_phase3"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharex=True, sharey=True)
    true = common["rh95"].to_numpy(float)
    for axis, (title, column), (_, row) in zip(axes, phases, metrics.iterrows()):
        prediction = common[column].to_numpy(float)
        density = axis.hexbin(
            true,
            prediction,
            gridsize=52,
            extent=(0, 45, 0, 45),
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        axis.plot([0, 45], [0, 45], "--", color="black", linewidth=1.2, label="1:1")
        fit = np.polyfit(true, prediction, 1)
        xx = np.asarray([2.5, 40.0])
        axis.plot(xx, fit[0] * xx + fit[1], color="crimson", linewidth=1.6, label="Régression")
        axis.set_title(title, fontweight="bold")
        axis.set_xlim(0, 45)
        axis.set_ylim(0, 45)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18)
        axis.set_xlabel("GEDI RH95 observé (m)")
        axis.text(
            0.035,
            0.965,
            (
                f"n={int(row.n):,}\n"
                f"KGE={row.kge_2009:.4f} | R²={row.r2:.4f}\n"
                f"MAE={row.mae:.4f} | RMSE={row.rmse:.4f}\n"
                f"slope={row.slope:.4f} | std={row.std_ratio:.4f}\n"
                f"bias={row.bias:+.4f} | r={row['corr']:.4f}"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        axis.legend(loc="lower right", fontsize=8)
        figure.colorbar(density, ax=axis, fraction=0.046, pad=0.03, label="log10(N)")
    axes[0].set_ylabel("Hauteur prédite (m)")
    figure.suptitle(
        "Comparaison TEST canonique W3 — même support GEDI — checkpoints sélectionnés sur VAL",
        fontweight="bold",
        fontsize=14,
    )
    figure.tight_layout()
    path = OUTPUT_DIR / "phase123_scatter_test_common_w3_n4304.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def run_phase123_comparison(
    *, device: str | None = None, reuse: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Path, pd.DataFrame]:
    for required in (
        DATASET_ROOT,
        PHASE1_TEST_PREDICTIONS,
        PHASE2_BEST,
        PHASE3_BEST,
        PHASE3_RUN / "TRAIN_DONE.json",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    active_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Évaluation commune Phase 1/2/3 sur :", active_device, flush=True)
    common = _common_predictions(active_device, reuse)
    print("Support TEST commun unique GEDI :", len(common), flush=True)

    definitions = [
        ("Phase 1", "B4 spatial / best.ckpt VAL", "prediction_phase1"),
        ("Phase 2", "P2A anti-shrink / best.ckpt VAL", "prediction_phase2"),
        ("Phase 3", "T4 RAW DROP5 K2 GL020 / best.ckpt VAL", "prediction_phase3"),
    ]
    metric_rows = []
    for phase, model, column in definitions:
        row = {"phase": phase, "model": model}
        row.update(_metrics(common["rh95"].to_numpy(), common[column].to_numpy()))
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    gains = _gain_table(metrics)
    metrics.to_csv(OUTPUT_DIR / "phase123_metrics_test_common_w3_n4304.csv", index=False)
    gains.to_csv(OUTPUT_DIR / "phase123_gains_test_common_w3_n4304.csv", index=False)
    figure_path = _scatter_figure(common, metrics)
    metadata = {
        "split": "test",
        "support": "common unique GEDI intersection; W3 temporal evaluation; RH95 2.5-45 m",
        "min_height_m": EVAL_MIN,
        "max_height_m": EVAL_MAX,
        "n": int(len(common)),
        "checkpoint_selection": "best.ckpt selected on VAL for all phases; no TEST selection",
        "phase1_checkpoint": str(PHASE1_RUN / "checkpoints" / "best.ckpt"),
        "phase2_checkpoint": str(PHASE2_BEST),
        "phase3_checkpoint": str(PHASE3_BEST),
    }
    (OUTPUT_DIR / "comparison_protocol_n4304.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics, gains, figure_path, common


if __name__ == "__main__":
    metric_table, gain_table, scatter_path, _ = run_phase123_comparison()
    print("\nMETRICS TEST — SUPPORT COMMUN W3")
    print(metric_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nGAINS — VALEUR POSITIVE = AMÉLIORATION")
    print(gain_table.to_string(index=False, float_format=lambda value: f"{value:+.4f}"))
    print("\nScatter :", scatter_path)
