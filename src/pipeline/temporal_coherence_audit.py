from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT = Path(r"C:\Users\Dell\Desktop\Publication_Clarck")
RUNNER_PATH = PROJECT / "Source" / "Project" / "low_canopy_growthloss_ablation_runner.py"
AUDIT_VERSION = "VAL_TEMPORAL_COHERENCE_V2_TILED256"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPATIAL_STRIDE = 8
TILE_SIZE = 256
MAX_GROWTH_M_PER_YEAR = 2.0

MATCHED_IDS = {
    "ifran": ["D5_K2_GL020_HD1"],
    "maamoura": [
        "D2_K2_GL000_HD1", "D2_K2_GL0025_HD1", "D2_K2_GL005",
        "D2_K2_GL010", "D2_K2_GL020_HD1",
    ],
    "agadir": [
        "AG_CTRL_GL000", "AG_D5_K2_GL005", "AG_D5_K2_GL010", "AG_D5_K2_GL020",
    ],
}
CONTROL_ID = {
    "ifran": "PHASE1_REFERENCE",
    "maamoura": "D2_K2_GL000_HD1",
    "agadir": "AG_CTRL_GL000",
}
LOWER_IS_BETTER = (
    "growth_gt_2m_rate", "drop_gt_threshold_rate", "mean_abs_second_difference_m",
    "mean_linear_fit_rmse_m",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("publication_clarck_growth_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sequence(paths, device: str, row_slice: slice, col_slice: slice):
    cubes = []
    original_shape = None
    for path in paths:
        array = np.load(str(path), mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or array.shape[-1] != 15:
            raise RuntimeError(f"Expected native C15 [H,W,15], got {array.shape}: {path}")
        crop = np.array(array[row_slice, col_slice, :], dtype=np.float32, copy=True)
        if original_shape is None:
            original_shape = tuple(crop.shape[:2])
        tensor = torch.from_numpy(np.moveaxis(crop, -1, 0))
        pad_h = (16 - tensor.shape[-2] % 16) % 16
        pad_w = (16 - tensor.shape[-1] % 16) % 16
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
        cubes.append(tensor)
    return torch.stack(cubes, dim=0).unsqueeze(0).to(device), original_shape


class MetricAccumulator:
    def __init__(self):
        self.sums = {name: 0.0 for name in LOWER_IS_BETTER}
        self.counts = {name: 0 for name in LOWER_IS_BETTER}
        self.stable_pixels = 0
        self.disturbed_pixels = 0
        self.sequence_occurrences = 0

    def add(self, values: np.ndarray, stable: np.ndarray, disturbed: np.ndarray, drop_m: float):
        # values [T,H,W], masks [H,W]
        self.sequence_occurrences += 1
        self.stable_pixels += int(stable.sum())
        self.disturbed_pixels += int(disturbed.sum())
        if not stable.any():
            return
        delta = np.diff(values, axis=0)
        transition_mask = np.broadcast_to(stable, delta.shape)
        growth = (delta > MAX_GROWTH_M_PER_YEAR) & transition_mask
        drops = (delta < -float(drop_m)) & transition_mask
        self.sums["growth_gt_2m_rate"] += float(growth.sum())
        self.counts["growth_gt_2m_rate"] += int(transition_mask.sum())
        self.sums["drop_gt_threshold_rate"] += float(drops.sum())
        self.counts["drop_gt_threshold_rate"] += int(transition_mask.sum())

        if values.shape[0] >= 3:
            second = np.abs(np.diff(values, n=2, axis=0))
            second_mask = np.broadcast_to(stable, second.shape)
            self.sums["mean_abs_second_difference_m"] += float(second[second_mask].sum())
            self.counts["mean_abs_second_difference_m"] += int(second_mask.sum())

        t = np.arange(values.shape[0], dtype=np.float64)
        tc = t - t.mean()
        slope = np.sum(tc[:, None, None] * values, axis=0) / np.sum(tc**2)
        fitted = values.mean(axis=0, keepdims=True) + tc[:, None, None] * slope[None]
        linear_rmse = np.sqrt(np.mean((values - fitted) ** 2, axis=0))
        self.sums["mean_linear_fit_rmse_m"] += float(linear_rmse[stable].sum())
        self.counts["mean_linear_fit_rmse_m"] += int(stable.sum())

    def finish(self):
        result = {
            name: (self.sums[name] / self.counts[name] if self.counts[name] else np.nan)
            for name in LOWER_IS_BETTER
        }
        result.update({
            "stable_pixel_sequence_occurrences": self.stable_pixels,
            "disturbed_pixel_sequence_occurrences": self.disturbed_pixels,
            "sequence_occurrences": self.sequence_occurrences,
            "spatial_stride": SPATIAL_STRIDE,
        })
        return result


@torch.inference_mode()
def audit_candidate(runner, modules, forest_key: str, candidate_id: str, checkpoint: Path, spec: dict, records):
    cfg = runner.FORESTS[forest_key]
    model = runner.fresh_model(cfg, modules)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.prediction_head.load_state_dict(state["prediction_head"], strict=True)
    model.to(DEVICE).eval()

    from src.persistent_growth_loss_v6 import persistent_loss_flags

    phase1_acc = MetricAccumulator()
    phase2_acc = MetricAccumulator()
    total = len(records)
    torch.backends.cudnn.benchmark = False
    for index, record in enumerate(records, start=1):
        first = np.load(str(record.x_paths[0]), mmap_mode="r", allow_pickle=False)
        full_height, full_width = first.shape[:2]
        del first
        for row_start in range(0, full_height, TILE_SIZE):
            for col_start in range(0, full_width, TILE_SIZE):
                row_slice = slice(row_start, min(row_start + TILE_SIZE, full_height))
                col_slice = slice(col_start, min(col_start + TILE_SIZE, full_width))
                sequence, original_shape = load_sequence(
                    record.x_paths, DEVICE, row_slice, col_slice
                )
                output = model(sequence)[0].float().cpu().numpy()
                height, width = original_shape
                output = output[..., :height, :width]
                reference = output[0, :, ::SPATIAL_STRIDE, ::SPATIAL_STRIDE].astype(np.float64)
                phase2 = output[1, :, ::SPATIAL_STRIDE, ::SPATIAL_STRIDE].astype(np.float64)

                ref_tensor = torch.from_numpy(reference).unsqueeze(0).float()
                _, _, windows = persistent_loss_flags(
                    ref_tensor, drop_threshold_m=float(spec["drop_m"]),
                    consecutive_flags=int(spec["K"]),
                )
                disturbed = windows.any(dim=1)[0].numpy() if windows.shape[1] else np.zeros(reference.shape[1:], bool)
                finite = np.isfinite(reference).all(axis=0) & np.isfinite(phase2).all(axis=0)
                reference_level = np.median(reference, axis=0)
                eligible_height = (reference_level >= float(cfg["eval_min"])) & (reference_level <= float(cfg["eval_max"]))
                stable = finite & eligible_height & ~disturbed
                disturbed = finite & eligible_height & disturbed
                phase1_acc.add(reference, stable, disturbed, float(spec["drop_m"]))
                phase2_acc.add(phase2, stable, disturbed, float(spec["drop_m"]))
                del sequence, output, ref_tensor, reference, phase2, windows
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        if index % 5 == 0 or index == total:
            print(f"[TEMPORAL VAL] {forest_key} {candidate_id} {index:03d}/{total:03d}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model, state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return phase1_acc.finish(), phase2_acc.finish()


def audit_forest(forest_key: str):
    runner = load_runner()
    runner.AGADIR_CONFIRMATORY_MODE = forest_key == "agadir"
    modules = runner.import_growth_modules()
    cfg = runner.FORESTS[forest_key]
    _, result_root = runner.roots(forest_key)
    ranking_path = result_root / "val_ranking_low_canopy_ablations.csv"
    if not ranking_path.is_file():
        raise FileNotFoundError(ranking_path)
    ranking = pd.read_csv(ranking_path)
    by_id = ranking.set_index("candidate_id", drop=False)
    ids = MATCHED_IDS[forest_key]
    missing = [candidate_id for candidate_id in ids if candidate_id not in by_id.index]
    if missing:
        raise RuntimeError(f"{forest_key}: candidates absent from VAL ranking: {missing}")

    _, _, records = runner.build_data(forest_key, modules, include_test=False)
    val_records = records["val"]
    audit_root = result_root / "VAL_Temporal_Coherence_Audit_v1"
    cache_root = audit_root / "Cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for candidate_id in ids:
        val_row = by_id.loc[candidate_id]
        checkpoint = Path(val_row["checkpoint"])
        checkpoint_hash = sha256(checkpoint)
        spec = {
            "drop_m": float(val_row["drop_m"]), "K": int(val_row["K"]),
            "lambda_growth": float(val_row["lambda_growth"]),
        }
        cache_path = cache_root / f"{candidate_id}.json"
        cached = None
        if cache_path.is_file():
            candidate_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                candidate_cache.get("audit_version") == AUDIT_VERSION
                and candidate_cache.get("checkpoint_sha256") == checkpoint_hash
                and int(candidate_cache.get("spatial_stride", -1)) == SPATIAL_STRIDE
                and int(candidate_cache.get("tile_size", -1)) == TILE_SIZE
            ):
                cached = candidate_cache
        if cached is None:
            print(f"\n[TEMPORAL AUDIT START] {forest_key} {candidate_id}", flush=True)
            phase1, phase2 = audit_candidate(
                runner, modules, forest_key, candidate_id, checkpoint, spec, val_records,
            )
            cached = {
                "audit_version": AUDIT_VERSION, "forest": cfg["label"],
                "candidate_id": candidate_id, "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash, "split": "VAL",
                "spatial_stride": SPATIAL_STRIDE, "tile_size": TILE_SIZE, "spec": spec,
                "phase1": phase1, "phase2": phase2,
            }
            cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
        else:
            print(f"[REUSE TEMPORAL AUDIT] {forest_key} {candidate_id}", flush=True)

        for phase_name in ("phase1", "phase2"):
            rows.append({
                "forest": cfg["label"], "candidate_id": candidate_id,
                "phase": "Phase 1" if phase_name == "phase1" else "Phase 2",
                **spec, **cached[phase_name],
            })

    long_table = pd.DataFrame(rows)
    summaries = []
    for candidate_id in ids:
        pair = long_table[long_table.candidate_id.eq(candidate_id)].set_index("phase")
        val_row = by_id.loc[candidate_id]
        row = {
            "forest": cfg["label"], "candidate_id": candidate_id,
            "drop_m": float(val_row["drop_m"]), "K": int(val_row["K"]),
            "lambda_growth": float(val_row["lambda_growth"]),
            "val_accuracy_eligible": bool(val_row.get("accuracy_eligible", False)),
            "val_r2": float(val_row["r2"]), "val_mae": float(val_row["mae"]),
            "val_rmse": float(val_row["rmse"]), "val_slope": float(val_row["slope"]),
        }
        improvement_count = 0
        max_relative_worsening = 0.0
        for metric in LOWER_IS_BETTER:
            before = float(pair.loc["Phase 1", metric])
            after = float(pair.loc["Phase 2", metric])
            gain = before - after
            ratio = after / before if before > 0 else np.nan
            row[f"phase1_{metric}"] = before
            row[f"phase2_{metric}"] = after
            row[f"gain_{metric}"] = gain
            row[f"ratio_{metric}"] = ratio
            improvement_count += int(gain > 0)
            if np.isfinite(ratio):
                max_relative_worsening = max(max_relative_worsening, ratio - 1.0)
        row["temporal_metrics_improved"] = improvement_count
        row["max_relative_temporal_worsening"] = max_relative_worsening
        row["temporal_pareto_eligible_vs_phase1"] = bool(
            row["val_accuracy_eligible"] and improvement_count >= 3 and max_relative_worsening <= 0.02
        )
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    control_id = CONTROL_ID[forest_key]
    if control_id == "PHASE1_REFERENCE":
        for metric in LOWER_IS_BETTER:
            summary[f"gain_vs_temporal_control_{metric}"] = summary[f"gain_{metric}"]
    else:
        control = summary.loc[summary.candidate_id.eq(control_id)].iloc[0]
        for metric in LOWER_IS_BETTER:
            control_value = float(control[f"phase2_{metric}"])
            summary[f"gain_vs_temporal_control_{metric}"] = control_value - summary[f"phase2_{metric}"]
    gain_columns = [f"gain_vs_temporal_control_{metric}" for metric in LOWER_IS_BETTER]
    summary["metrics_improved_vs_temporal_control"] = (summary[gain_columns] > 0).sum(axis=1)
    summary["growthloss_temporal_accepted_on_val"] = (
        summary["lambda_growth"].gt(0)
        & summary["val_accuracy_eligible"]
        & summary["metrics_improved_vs_temporal_control"].ge(3)
    )

    audit_root.mkdir(parents=True, exist_ok=True)
    long_table.to_csv(audit_root / "temporal_metrics_phase1_vs_phase2_long.csv", index=False)
    summary.to_csv(audit_root / "temporal_pareto_summary_val.csv", index=False)
    protocol = {
        "audit_version": AUDIT_VERSION, "forest": cfg["label"], "split": "VAL only",
        "test_used": False, "spatial_stride": SPATIAL_STRIDE, "tile_size": TILE_SIZE,
        "height_domain_m": [cfg["eval_min"], cfg["eval_max"]],
        "growth_violation_threshold_m_per_year": MAX_GROWTH_M_PER_YEAR,
        "disturbance_mask": "persistent running maximum computed from frozen Phase-1 reference",
        "stable_pixel_definition": "finite C15 prediction, reference median in evaluation domain, no persistent event",
        "selection_note": "Temporal acceptance is descriptive and must be combined with the pre-existing VAL accuracy gate.",
    }
    (audit_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, metric in zip(axes.ravel(), LOWER_IS_BETTER):
        values = summary[f"ratio_{metric}"].to_numpy(float)
        axis.bar(np.arange(len(summary)), values, color=np.where(values <= 1, "#2a9d8f", "#e76f51"))
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(np.arange(len(summary)), summary.candidate_id, rotation=35, ha="right", fontsize=8)
        axis.set_ylabel("Phase 2 / Phase 1 (<1 meilleur)")
        axis.set_title(metric.replace("_", " "))
        axis.grid(axis="y", alpha=.2)
    fig.suptitle(f"{cfg['label']} — cohérence temporelle sur VAL", fontweight="bold")
    fig.tight_layout()
    fig.savefig(audit_root / "temporal_ratios_phase2_vs_phase1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6.5))
    scatter = axis.scatter(
        summary.val_mae, summary["gain_mean_linear_fit_rmse_m"],
        c=summary.lambda_growth, s=110, cmap="viridis", edgecolor="black", linewidth=.5,
    )
    for row in summary.itertuples():
        axis.annotate(row.candidate_id, (row.val_mae, row.gain_mean_linear_fit_rmse_m),
                      xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("MAE VAL statique (m) — plus petit = meilleur")
    axis.set_ylabel("Gain de linéarité temporelle (m) — plus grand = meilleur")
    axis.set_title(f"{cfg['label']} — compromis précision / cohérence sur VAL")
    axis.grid(alpha=.2)
    fig.colorbar(scatter, ax=axis, label="lambda GrowthLoss")
    fig.tight_layout()
    fig.savefig(audit_root / "pareto_static_mae_vs_temporal_gain.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"\nTEMPORAL PARETO SUMMARY — {cfg['label']} — VAL ONLY", flush=True)
    show = [
        "candidate_id", "lambda_growth", "val_accuracy_eligible", "val_r2", "val_mae", "val_slope",
        "temporal_metrics_improved", "metrics_improved_vs_temporal_control",
        "temporal_pareto_eligible_vs_phase1", "growthloss_temporal_accepted_on_val",
    ]
    print(summary[show].to_string(index=False), flush=True)
    print("OUTPUT", audit_root, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forest", choices=("ifran", "maamoura", "agadir", "all"), default="all")
    args = parser.parse_args()
    forests = ("ifran", "maamoura", "agadir") if args.forest == "all" else (args.forest,)
    for forest in forests:
        audit_forest(forest)


if __name__ == "__main__":
    main()
