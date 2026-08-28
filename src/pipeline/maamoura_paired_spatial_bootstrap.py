from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT = Path(r"C:\Users\Dell\Desktop\Publication_Clarck")
RUNNER_PATH = PROJECT / "Source" / "Project" / "low_canopy_growthloss_ablation_runner.py"
FOREST_KEY = "maamoura"
CONTROL_ID = "D2_K2_GL000_HD1"
CANDIDATE_ID = "D2_K2_GL0025_HD1"
BOOTSTRAP_SEED = 42
N_BOOTSTRAP = 5000
MARGINS = {"delta_r2": -0.01, "delta_mae_m": 0.03, "delta_rmse_m": 0.05}
AUDIT_VERSION = "MAAMOURA_PAIRED_SPATIAL_BOOTSTRAP_V1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("paired_bootstrap_growth_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.AGADIR_CONFIRMATORY_MODE = False
    return module


def spatial_cluster_id(frame: pd.DataFrame) -> pd.Series:
    if "patch_key" in frame.columns:
        def from_patch(value):
            parts = str(value).split("|")
            if len(parts) >= 2:
                return "|".join(parts[:2])
            return re.sub(r"\|M\d{2}\|Y\d{4}-\d{4}$", "", str(value))
        return frame["patch_key"].map(from_patch).astype(str)
    if "sample_id" not in frame.columns:
        raise KeyError("Neither patch_key nor sample_id is available for spatial clustering")
    return frame["sample_id"].astype(str).str.replace(r"_Y\d{4}_M\d{2}$", "", regex=True)


def evaluate_candidate(runner, modules, candidate_id: str, checkpoint: Path, records, shots, prediction_root: Path):
    checkpoint_hash = sha256(checkpoint)
    prediction_path = prediction_root / f"{candidate_id}_val_unique_nearest.csv.gz"
    lineage_path = prediction_root / f"{candidate_id}_lineage.json"
    reusable = False
    if prediction_path.is_file() and lineage_path.is_file():
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        reusable = (
            lineage.get("audit_version") == AUDIT_VERSION
            and lineage.get("checkpoint_sha256") == checkpoint_hash
        )
    if reusable:
        print("[REUSE VAL PREDICTIONS]", candidate_id, flush=True)
        return pd.read_csv(prediction_path)

    cfg = runner.FORESTS[FOREST_KEY]
    model = runner.fresh_model(cfg, modules)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.prediction_head.load_state_dict(state["prediction_head"], strict=True)
    model.eval()
    _, nearest = modules["evaluate_full_patch_temporal_nearest"](
        model=model, records=records, shots=shots, device=DEVICE, split="val",
        drop_channels=(), min_height=cfg["eval_min"], max_height=cfg["eval_max"],
        progress_every=5,
    )
    nearest.to_csv(prediction_path, index=False, compression="gzip")
    lineage_path.write_text(json.dumps({
        "audit_version": AUDIT_VERSION, "forest": cfg["label"], "split": "VAL only",
        "candidate_id": candidate_id, "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash, "test_used": False,
    }, indent=2), encoding="utf-8")
    del model, state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return nearest


def cluster_sufficient_statistics(frame: pd.DataFrame, pred_col: str):
    y = frame["rh95"].to_numpy(np.float64)
    pred = frame[pred_col].to_numpy(np.float64)
    err = pred - y
    return pd.Series({
        "n": len(frame), "sum_y": y.sum(), "sum_y2": np.square(y).sum(),
        "sum_pred": pred.sum(), "sum_pred2": np.square(pred).sum(),
        "sum_ypred": (y * pred).sum(), "sum_abs": np.abs(err).sum(),
        "sum_sq": np.square(err).sum(), "sum_err": err.sum(),
    })


def metrics_from_sums(sums: np.ndarray):
    # columns: n, sum_y, sum_y2, sum_pred, sum_pred2, sum_ypred, sum_abs, sum_sq, sum_err
    n, sy, sy2, sp, sp2, syp, sa, ss, se = [sums[:, i] for i in range(sums.shape[1])]
    sst = sy2 - np.square(sy) / n
    var_y_num = sst
    var_p_num = sp2 - np.square(sp) / n
    cov_num = syp - sy * sp / n
    return {
        "r2": 1.0 - ss / sst,
        "mae": sa / n,
        "rmse": np.sqrt(ss / n),
        "bias": se / n,
        "slope": cov_num / var_y_num,
        "std_ratio": np.sqrt(np.maximum(var_p_num, 0) / var_y_num),
    }


def paired_bootstrap(frame: pd.DataFrame, n_bootstrap: int, seed: int):
    grouped = frame.groupby("spatial_cluster_id", sort=True, observed=True)
    control_stats = grouped.apply(lambda g: cluster_sufficient_statistics(g, "pred_control"))
    candidate_stats = grouped.apply(lambda g: cluster_sufficient_statistics(g, "pred_candidate"))
    if not control_stats.index.equals(candidate_stats.index):
        raise RuntimeError("Cluster mismatch")
    clusters = control_stats.index.to_numpy(str)
    c0 = control_stats.to_numpy(np.float64)
    c1 = candidate_stats.to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(n_bootstrap, len(clusters)))
    multiplicities = np.zeros((n_bootstrap, len(clusters)), dtype=np.int32)
    row_ids = np.repeat(np.arange(n_bootstrap), len(clusters))
    np.add.at(multiplicities, (row_ids, draws.ravel()), 1)
    sums0 = multiplicities @ c0
    sums1 = multiplicities @ c1
    m0 = metrics_from_sums(sums0)
    m1 = metrics_from_sums(sums1)
    out = pd.DataFrame({
        "delta_r2": m1["r2"] - m0["r2"],
        "delta_mae_m": m1["mae"] - m0["mae"],
        "delta_rmse_m": m1["rmse"] - m0["rmse"],
        "delta_slope": m1["slope"] - m0["slope"],
        "delta_std_ratio": m1["std_ratio"] - m0["std_ratio"],
        "delta_abs_bias_m": np.abs(m1["bias"]) - np.abs(m0["bias"]),
    })
    return out, clusters


def point_metrics(y, pred):
    y = np.asarray(y, np.float64); pred = np.asarray(pred, np.float64)
    err = pred - y
    corr = np.corrcoef(y, pred)[0, 1]
    slope = np.polyfit(y, pred, 1)[0]
    std_ratio = np.std(pred) / np.std(y)
    return {
        "n": len(y), "r2": 1 - np.square(err).sum() / np.square(y-y.mean()).sum(),
        "mae": np.abs(err).mean(), "rmse": np.sqrt(np.square(err).mean()),
        "slope": slope, "std_ratio": std_ratio, "bias": err.mean(), "correlation": corr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    runner = load_runner()
    modules = runner.import_growth_modules()
    _, result_root = runner.roots(FOREST_KEY)
    ranking = pd.read_csv(result_root / "val_ranking_low_canopy_ablations.csv").set_index("candidate_id")
    for candidate_id in (CONTROL_ID, CANDIDATE_ID):
        if candidate_id not in ranking.index:
            raise RuntimeError(f"Missing candidate in VAL ranking: {candidate_id}")
    output_root = result_root / "VAL_Paired_Spatial_Bootstrap_GL000_vs_GL0025"
    prediction_root = output_root / "Predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    _, shots, records = runner.build_data(FOREST_KEY, modules, include_test=False)
    predictions = {}
    for candidate_id in (CONTROL_ID, CANDIDATE_ID):
        checkpoint = Path(ranking.loc[candidate_id, "checkpoint"])
        predictions[candidate_id] = evaluate_candidate(
            runner, modules, candidate_id, checkpoint, records["val"], shots, prediction_root,
        )

    control = predictions[CONTROL_ID].copy()
    candidate = predictions[CANDIDATE_ID].copy()
    for frame in (control, candidate):
        frame["aux_shot_uid"] = frame["aux_shot_uid"].astype(str)
        frame.sort_values("aux_shot_uid", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if not np.array_equal(control.aux_shot_uid, candidate.aux_shot_uid):
        raise RuntimeError("VAL UID support differs between candidates")
    if not np.allclose(control.rh95, candidate.rh95, rtol=0, atol=1e-9):
        raise RuntimeError("VAL targets differ between candidates")
    if not np.allclose(control.pred_off_reference, candidate.pred_off_reference, rtol=0, atol=1e-6):
        raise RuntimeError("Frozen Phase-1 predictions differ between candidates")

    paired = control[["aux_shot_uid", "patch_key", "sample_id", "rh95"]].copy()
    paired["pred_control"] = control["pred_on_growthloss"].to_numpy(float)
    paired["pred_candidate"] = candidate["pred_on_growthloss"].to_numpy(float)
    paired["spatial_cluster_id"] = spatial_cluster_id(paired)
    paired.to_csv(output_root / "paired_val_predictions.csv.gz", index=False, compression="gzip")

    draws, clusters = paired_bootstrap(paired, int(args.n_bootstrap), BOOTSTRAP_SEED)
    draws.to_csv(output_root / "paired_spatial_bootstrap_draws.csv.gz", index=False, compression="gzip")
    point0 = point_metrics(paired.rh95, paired.pred_control)
    point1 = point_metrics(paired.rh95, paired.pred_candidate)
    point_delta = {
        "delta_r2": point1["r2"] - point0["r2"],
        "delta_mae_m": point1["mae"] - point0["mae"],
        "delta_rmse_m": point1["rmse"] - point0["rmse"],
        "delta_slope": point1["slope"] - point0["slope"],
        "delta_std_ratio": point1["std_ratio"] - point0["std_ratio"],
        "delta_abs_bias_m": abs(point1["bias"]) - abs(point0["bias"]),
    }
    summary_rows = []
    for metric in draws.columns:
        values = draws[metric].to_numpy(float)
        summary_rows.append({
            "metric": metric, "point_delta": point_delta[metric],
            "bootstrap_mean": values.mean(), "ci_90_low": np.quantile(values, .05),
            "ci_90_high": np.quantile(values, .95), "ci_95_low": np.quantile(values, .025),
            "ci_95_high": np.quantile(values, .975),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "bootstrap_summary.csv", index=False)

    by_metric = summary.set_index("metric")
    noninferiority = {
        "r2": bool(by_metric.loc["delta_r2", "ci_90_low"] >= MARGINS["delta_r2"]),
        "mae": bool(by_metric.loc["delta_mae_m", "ci_90_high"] <= MARGINS["delta_mae_m"]),
        "rmse": bool(by_metric.loc["delta_rmse_m", "ci_90_high"] <= MARGINS["delta_rmse_m"]),
    }
    temporal_path = result_root / "VAL_Temporal_Coherence_Audit_v1" / "temporal_pareto_summary_val.csv"
    temporal = pd.read_csv(temporal_path).set_index("candidate_id")
    temporal_improved = int(temporal.loc[CANDIDATE_ID, "metrics_improved_vs_temporal_control"])
    accepted = bool(all(noninferiority.values()) and temporal_improved >= 3)
    decision = {
        "audit_version": AUDIT_VERSION, "split": "VAL only", "test_used": False,
        "control": CONTROL_ID, "candidate": CANDIDATE_ID, "n_val": int(len(paired)),
        "n_spatial_clusters": int(len(clusters)), "n_bootstrap": int(args.n_bootstrap),
        "seed": BOOTSTRAP_SEED, "noninferiority_margins": MARGINS,
        "control_metrics": point0, "candidate_metrics": point1, "point_deltas": point_delta,
        "one_sided_95pct_noninferiority": noninferiority,
        "temporal_metrics_improved_vs_control": temporal_improved,
        "growthloss_gl0025_accepted_on_val": accepted,
        "decision": (
            "Accept GL0025: spatially bootstrapped static non-inferiority and temporal improvement confirmed."
            if accepted else
            "Retain GL000: GL0025 does not satisfy every pre-specified spatial-bootstrap non-inferiority criterion."
        ),
    }
    (output_root / "final_val_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    plot_specs = [
        ("delta_r2", MARGINS["delta_r2"], "ΔR² (GL0025 − GL000)", "lower"),
        ("delta_mae_m", MARGINS["delta_mae_m"], "ΔMAE (m)", "upper"),
        ("delta_rmse_m", MARGINS["delta_rmse_m"], "ΔRMSE (m)", "upper"),
    ]
    for axis, (metric, margin, title, side) in zip(axes, plot_specs):
        values = draws[metric].to_numpy(float)
        axis.hist(values, bins=45, color="#457b9d", alpha=.82)
        axis.axvline(point_delta[metric], color="black", linewidth=1.5, label="Delta observé")
        axis.axvline(margin, color="crimson", linestyle="--", linewidth=1.5, label="Marge NI")
        axis.set_title(title); axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Maamoura — bootstrap spatial apparié VAL — GL0025 vs GL000", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_root / "bootstrap_noninferiority_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("\nMAAMOURA PAIRED SPATIAL BOOTSTRAP — VAL ONLY", flush=True)
    print("CONTROL", point0, flush=True)
    print("CANDIDATE", point1, flush=True)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    print("OUTPUT", output_root, flush=True)


if __name__ == "__main__":
    main()
