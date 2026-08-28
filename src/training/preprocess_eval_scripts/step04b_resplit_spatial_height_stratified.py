# -*- coding: utf-8 -*-
from __future__ import annotations

"""
STEP04B — spatial + height-stratified resplit for Sans_NPZ.

Purpose
-------
Create a new STEP04-like dataset subdirectory BEFORE STEP05, using a fresh
spatial split with target fractions 75/15/10 and height-class balancing.

This script does not copy large .npy tensors. It keeps the existing x_path
references from the source NPY catalog, then rewrites only:
  - sample_catalog.csv
  - shot_catalog.csv.gz
  - provenance/audit summaries

That is enough for step05_build_catalog_npy.py because it trusts x_path when
present and verifies that each referenced .npy exists.

Split safety
------------
The split unit is a connected component over:
  - spatial_group_id
  - aux_shot_id

So a GEDI shot that appears in more than one spatial group forces those groups
into the same split. This protects against cross-split GEDI-shot leakage.
"""

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CHANNEL_ORDER = [
    "S2_LEAFON_B02", "S2_LEAFON_B03", "S2_LEAFON_B04", "S2_LEAFON_B08",
    "S1_ASC_VV", "S1_ASC_VH", "S1_DESC_VV", "S1_DESC_VH",
    "PALSAR_HH", "PALSAR_HV", "AOI_MASK",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic parent selection keeps manifests stable.
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def leakage_count(df: pd.DataFrame, key: str, split_col: str = "split") -> int:
    if key not in df.columns:
        return -1
    tmp = df[[key, split_col]].dropna().copy()
    tmp[key] = tmp[key].astype(str)
    return int((tmp.groupby(key, dropna=False)[split_col].nunique() > 1).sum())


def parse_bins(text: str) -> np.ndarray:
    arr = np.asarray([float(x) for x in str(text).split(",")], dtype=np.float64)
    if arr.size < 2 or not np.all(np.diff(arr) > 0):
        raise ValueError(f"Invalid height bins: {text}")
    return arr


# STEP04B_ELIGIBILITY_FILTERS_NOBSGT5_GT5M10PCT_S2V90_TW180_V1
def apply_sample_eligibility_filters(
    samples: pd.DataFrame,
    shots: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Filter patch/month samples before spatial splitting.

    Eligibility is intentionally evaluated before the 75/15/10 resplit so the
    final split ratios and height balance are optimized on the scientific
    sample universe actually used by training/evaluation.

    Strict criteria used by this notebook:
      - S2 valid AOI fraction >= 0.90
      - number of valid GEDI/RH95 observations > 5
      - fraction of valid GEDI/RH95 observations with RH95 > threshold > 0.10
      - GEDI/S2 absolute temporal delta <= 180 days when the column is present
    """
    samples = samples.copy()
    shots = shots.copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    shots["sample_id"] = shots["sample_id"].astype(str)

    input_samples = int(len(samples))
    input_shots = int(len(shots))

    if not bool(getattr(args, "apply_eligibility_filters", True)):
        table = samples[["sample_id"]].copy()
        table["eligible"] = True
        audit = {
            "enabled": False,
            "input_samples": input_samples,
            "input_shots": input_shots,
            "output_samples": input_samples,
            "output_shots": input_shots,
        }
        return samples, shots, audit, table

    required = {"sample_id", "s2_valid_aoi_fraction"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise RuntimeError(f"Eligibility filter requires sample columns missing from source: {missing}")
    required_shots = {"sample_id", "rh95"}
    missing_shots = sorted(required_shots - set(shots.columns))
    if missing_shots:
        raise RuntimeError(f"Eligibility filter requires shot columns missing from source: {missing_shots}")

    y_all = pd.to_numeric(shots["rh95"], errors="coerce")
    height_domain_mask = np.isfinite(y_all) & (y_all >= float(args.height_min)) & (y_all <= float(args.height_max))

    temporal_available = "aux_abs_temporal_delta_days" in shots.columns
    temporal_filter_enabled = bool(
        temporal_available
        and math.isfinite(float(args.max_abs_temporal_delta_days))
        and float(args.max_abs_temporal_delta_days) > 0
    )
    if temporal_filter_enabled:
        delta = pd.to_numeric(shots["aux_abs_temporal_delta_days"], errors="coerce")
        temporal_mask = np.isfinite(delta) & (delta <= float(args.max_abs_temporal_delta_days))
    else:
        temporal_mask = pd.Series(True, index=shots.index)

    valid_observation_mask = height_domain_mask & temporal_mask
    shots_valid = shots.loc[valid_observation_mask].copy()
    y_valid = pd.to_numeric(shots_valid["rh95"], errors="coerce")
    shots_valid["_elig_ge_high"] = y_valid >= float(args.gedi_high_threshold)

    grouped = shots_valid.groupby("sample_id", dropna=False).agg(
        n_rh95_observations=("rh95", "size"),
        n_rh95_ge_high=("_elig_ge_high", "sum"),
    )
    grouped["frac_rh95_ge_high"] = (
        grouped["n_rh95_ge_high"].astype(float)
        / grouped["n_rh95_observations"].replace(0, np.nan).astype(float)
    )

    keep_cols = ["sample_id", "s2_valid_aoi_fraction"]
    if "split" in samples.columns:
        keep_cols.append("split")
    if "n_valid_points" in samples.columns:
        keep_cols.append("n_valid_points")
    if "rh95_max" in samples.columns:
        keep_cols.append("rh95_max")
    if "rh95_median" in samples.columns:
        keep_cols.append("rh95_median")
    table = samples[keep_cols].copy().merge(
        grouped.reset_index(),
        on="sample_id",
        how="left",
    )
    table["n_rh95_observations"] = table["n_rh95_observations"].fillna(0).astype(int)
    table["n_rh95_ge_high"] = table["n_rh95_ge_high"].fillna(0).astype(int)
    table["frac_rh95_ge_high"] = table["frac_rh95_ge_high"].fillna(0.0).astype(float)
    table["s2_valid_aoi_fraction"] = pd.to_numeric(table["s2_valid_aoi_fraction"], errors="coerce")

    table["pass_s2_valid"] = table["s2_valid_aoi_fraction"] >= float(args.min_s2_valid_fraction)
    table["pass_min_gedi_observations"] = table["n_rh95_observations"] > int(args.min_gedi_observations_per_patch)
    table["pass_fraction_ge_high"] = table["frac_rh95_ge_high"] > float(args.min_fraction_gedi_rh95_ge10)
    table["eligible"] = (
        table["pass_s2_valid"]
        & table["pass_min_gedi_observations"]
        & table["pass_fraction_ge_high"]
    )

    eligible_ids = set(table.loc[table["eligible"], "sample_id"].astype(str))
    samples_out = samples[samples["sample_id"].astype(str).isin(eligible_ids)].copy().reset_index(drop=True)
    shots_out = shots_valid[shots_valid["sample_id"].astype(str).isin(eligible_ids)].copy()
    shots_out = shots_out.drop(columns=["_elig_ge_high"], errors="ignore").reset_index(drop=True)

    if samples_out.empty or shots_out.empty:
        raise RuntimeError(
            "Eligibility filters removed all samples/shots. "
            "Relax thresholds or inspect step04b_sample_eligibility_audit.csv."
        )

    audit = {
        "enabled": True,
        "thresholds": {
            "min_s2_valid_fraction": float(args.min_s2_valid_fraction),
            "min_gedi_observations_per_patch_strict_gt": int(args.min_gedi_observations_per_patch),
            "gedi_high_threshold_m": float(args.gedi_high_threshold),
            "min_fraction_gedi_rh95_ge_high_fraction_strict_gt": float(args.min_fraction_gedi_rh95_ge10),
            "height_domain_m": [float(args.height_min), float(args.height_max)],
            "max_abs_temporal_delta_days": float(args.max_abs_temporal_delta_days),
            "temporal_column_present": bool(temporal_available),
            "temporal_filter_enabled": bool(temporal_filter_enabled),
        },
        "input_samples": input_samples,
        "input_shots": input_shots,
        "valid_height_domain_shots": int(height_domain_mask.sum()),
        "valid_temporal_shots": int(temporal_mask.sum()) if temporal_filter_enabled else input_shots,
        "valid_height_and_temporal_shots": int(valid_observation_mask.sum()),
        "output_samples": int(len(samples_out)),
        "output_shots": int(len(shots_out)),
        "removed_samples": int(input_samples - len(samples_out)),
        "removed_shots": int(input_shots - len(shots_out)),
        "fail_counts_nonexclusive": {
            "s2_valid_lt_threshold": int((~table["pass_s2_valid"]).sum()),
            "gedi_observations_le_min": int((~table["pass_min_gedi_observations"]).sum()),
            "fraction_rh95_ge_high_le_threshold": int((~table["pass_fraction_ge_high"]).sum()),
        },
        "eligible_by_source_split": (
            table.groupby("split")["eligible"].agg(["size", "sum"]).astype(int).to_dict(orient="index")
            if "split" in table.columns
            else {}
        ),
    }
    print(
        "[ELIGIBILITY FILTER] "
        f"samples {input_samples:,} -> {len(samples_out):,} | "
        f"shots {input_shots:,} -> {len(shots_out):,} | "
        f"S2>={float(args.min_s2_valid_fraction):.2f} | "
        f"GEDI obs>{int(args.min_gedi_observations_per_patch)} | "
        f"frac(RH95>={float(args.gedi_high_threshold):g})>{float(args.min_fraction_gedi_rh95_ge10):.2f} | "
        f"|delta|<={float(args.max_abs_temporal_delta_days):g}d",
        flush=True,
    )
    print(
        "[ELIGIBILITY FAIL NON-EXCLUSIVE] "
        f"s2={audit['fail_counts_nonexclusive']['s2_valid_lt_threshold']} | "
        f"n_gedi={audit['fail_counts_nonexclusive']['gedi_observations_le_min']} | "
        f"high_frac={audit['fail_counts_nonexclusive']['fraction_rh95_ge_high_le_threshold']}",
        flush=True,
    )
    return samples_out, shots_out, audit, table


def build_components(samples: pd.DataFrame, shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dsu = DSU()
    samples = samples.copy()
    shots = shots.copy()

    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["spatial_group_id"] = samples["spatial_group_id"].astype(str)
    shots["sample_id"] = shots["sample_id"].astype(str)
    shots["aux_shot_id"] = shots["aux_shot_id"].astype(str)

    sample_to_sg = samples.set_index("sample_id")["spatial_group_id"].to_dict()
    for sg in samples["spatial_group_id"].dropna().astype(str).unique():
        dsu.add("sg:" + sg)

    shots["_spatial_group_id"] = shots["sample_id"].map(sample_to_sg).astype(str)
    for sg, shot_id in zip(shots["_spatial_group_id"], shots["aux_shot_id"]):
        if sg and sg != "nan" and shot_id and shot_id != "nan":
            dsu.union("sg:" + sg, "shot:" + shot_id)

    sg_to_component = {
        sg: dsu.find("sg:" + sg)
        for sg in samples["spatial_group_id"].dropna().astype(str).unique()
    }
    samples["_component_id"] = samples["spatial_group_id"].astype(str).map(sg_to_component)
    shots["_component_id"] = shots["_spatial_group_id"].astype(str).map(sg_to_component)
    return samples, shots


def component_table(
    samples: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    height_bins: np.ndarray,
    height_min: float,
    height_max: float,
) -> pd.DataFrame:
    y = pd.to_numeric(shots["rh95"], errors="coerce")
    valid_shots = shots[np.isfinite(y) & (y >= height_min) & (y <= height_max)].copy()
    valid_shots["rh95"] = pd.to_numeric(valid_shots["rh95"], errors="coerce")

    rows = []
    for comp, sg in samples.groupby("_component_id", dropna=False):
        shot_g = valid_shots[valid_shots["_component_id"] == comp]
        unique_shots = shot_g.drop_duplicates("aux_shot_id")
        hist, _ = np.histogram(unique_shots["rh95"].to_numpy(dtype=np.float64), bins=height_bins)
        rows.append({
            "component_id": str(comp),
            "n_samples": int(len(sg)),
            "n_patches": int(sg["patch_key"].nunique()) if "patch_key" in sg.columns else int(len(sg)),
            "n_spatial_groups": int(sg["spatial_group_id"].nunique()),
            "n_unique_shots": int(unique_shots["aux_shot_id"].nunique()),
            "rh95_p90": float(unique_shots["rh95"].quantile(0.90)) if len(unique_shots) else float("nan"),
            "height_hist": hist.astype(int).tolist(),
        })
    tab = pd.DataFrame(rows)
    if tab.empty:
        raise RuntimeError("No split components were produced.")
    return tab


def score_assignment(
    comp_tab: pd.DataFrame,
    assignment: dict[str, str],
    *,
    targets: dict[str, float],
    height_bins: np.ndarray,
    min_bin_count: int,
) -> tuple[float, dict[str, Any]]:
    splits = ["train", "val", "test"]
    target_arr = np.asarray([targets[s] for s in splits], dtype=np.float64)
    sample_counts = Counter()
    shot_counts = Counter()
    hist_by_split = {s: np.zeros(len(height_bins) - 1, dtype=np.float64) for s in splits}

    for _, r in comp_tab.iterrows():
        sp = assignment[str(r["component_id"])]
        sample_counts[sp] += int(r["n_samples"])
        shot_counts[sp] += int(r["n_unique_shots"])
        hist_by_split[sp] += np.asarray(r["height_hist"], dtype=np.float64)

    total_samples = max(1, sum(sample_counts.values()))
    total_shots = max(1, sum(shot_counts.values()))
    sample_frac = np.asarray([sample_counts[s] / total_samples for s in splits])
    shot_frac = np.asarray([shot_counts[s] / total_shots for s in splits])

    sample_score = float(np.abs(sample_frac - target_arr).sum())
    shot_score = float(np.abs(shot_frac - target_arr).sum())

    hist_matrix = np.vstack([hist_by_split[s] for s in splits])
    bin_totals = hist_matrix.sum(axis=0)
    height_score = 0.0
    missing_penalty = 0.0
    for j, total in enumerate(bin_totals):
        if total <= 0:
            continue
        frac = hist_matrix[:, j] / total
        # Rare bins are informative but should not dominate impossible splits.
        weight = min(3.0, math.sqrt(total / max(1, min_bin_count)))
        height_score += float(weight * np.abs(frac - target_arr).sum())
        if total >= min_bin_count:
            for split_idx, sp in enumerate(splits):
                if hist_matrix[split_idx, j] == 0:
                    missing_penalty += 2.0

    empty_split_penalty = sum(50.0 for sp in splits if sample_counts[sp] == 0 or shot_counts[sp] == 0)
    total_score = (
        5.0 * sample_score
        + 8.0 * shot_score
        + 1.25 * height_score
        + missing_penalty
        + empty_split_penalty
    )
    details = {
        "sample_frac": dict(zip(splits, sample_frac.round(6).tolist())),
        "shot_frac": dict(zip(splits, shot_frac.round(6).tolist())),
        "sample_counts": {s: int(sample_counts[s]) for s in splits},
        "unique_shot_counts": {s: int(shot_counts[s]) for s in splits},
        "height_score": height_score,
        "missing_penalty": missing_penalty,
    }
    return float(total_score), details


def propose_assignment(
    comp_tab: pd.DataFrame,
    *,
    targets: dict[str, float],
    height_bins: np.ndarray,
    min_bin_count: int,
    seed: int,
    n_candidates: int,
) -> tuple[dict[str, str], float, dict[str, Any]]:
    rng = random.Random(seed)
    splits = ["train", "val", "test"]
    target_arr = np.asarray([targets[s] for s in splits], dtype=np.float64)
    comp_ids = comp_tab["component_id"].astype(str).tolist()
    sample_map = dict(zip(comp_tab["component_id"].astype(str), comp_tab["n_samples"].astype(float)))
    shot_map = dict(zip(comp_tab["component_id"].astype(str), comp_tab["n_unique_shots"].astype(float)))
    hist_map = {
        str(r["component_id"]): np.asarray(r["height_hist"], dtype=np.float64)
        for _, r in comp_tab.iterrows()
    }
    total_samples = max(1.0, float(comp_tab["n_samples"].sum()))
    total_shots = max(1.0, float(comp_tab["n_unique_shots"].sum()))
    total_hist = np.maximum(1.0, np.sum(np.vstack(list(hist_map.values())), axis=0))

    def greedy_once(k: int) -> dict[str, str]:
        # Large, tall and shot-rich components first, with stochastic jitter.
        order = sorted(
            comp_ids,
            key=lambda cid: (
                shot_map[cid] * (1.0 + 0.08 * rng.random())
                + sample_map[cid] * 0.25
                + float(np.nan_to_num(comp_tab.loc[comp_tab["component_id"].astype(str) == cid, "rh95_p90"].iloc[0], nan=0.0)) * rng.random()
            ),
            reverse=True,
        )
        counts_s = Counter()
        counts_u = Counter()
        hist_s = {s: np.zeros(len(height_bins) - 1, dtype=np.float64) for s in splits}
        assignment: dict[str, str] = {}
        split_noise = {s: 0.015 * rng.random() for s in splits}
        for cid in order:
            best_split, best_local = None, float("inf")
            for sp in splits:
                cand_counts_s = counts_s.copy(); cand_counts_s[sp] += sample_map[cid]
                cand_counts_u = counts_u.copy(); cand_counts_u[sp] += shot_map[cid]
                cand_hist = {s: hist_s[s].copy() for s in splits}; cand_hist[sp] += hist_map[cid]

                sample_frac = np.asarray([cand_counts_s[s] / total_samples for s in splits])
                shot_frac = np.asarray([cand_counts_u[s] / total_shots for s in splits])
                hist_matrix = np.vstack([cand_hist[s] for s in splits])
                hist_frac = hist_matrix / total_hist[None, :]
                # Penalize exceeding target too early, but allow train to take the largest blocks.
                overshoot = np.maximum(0.0, shot_frac - target_arr - 0.025).sum()
                local = (
                    5.0 * np.abs(sample_frac - target_arr).sum()
                    + 8.0 * np.abs(shot_frac - target_arr).sum()
                    + 0.35 * np.abs(hist_frac - target_arr[:, None]).mean()
                    + 6.0 * overshoot
                    + split_noise[sp]
                )
                if local < best_local:
                    best_local = float(local)
                    best_split = sp
            assert best_split is not None
            assignment[cid] = best_split
            counts_s[best_split] += sample_map[cid]
            counts_u[best_split] += shot_map[cid]
            hist_s[best_split] += hist_map[cid]
        return assignment

    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    best_details: dict[str, Any] = {}
    for k in range(max(1, n_candidates)):
        assignment = greedy_once(k)
        score, details = score_assignment(
            comp_tab,
            assignment,
            targets=targets,
            height_bins=height_bins,
            min_bin_count=min_bin_count,
        )
        if score < best_score:
            best_assignment, best_score, best_details = assignment, score, details
            print(
                f"[RESPLIT BEST] iter={k:06d} score={best_score:.6f} "
                f"| sample_frac={details['sample_frac']} | shot_frac={details['shot_frac']}",
                flush=True,
            )
    assert best_assignment is not None
    return best_assignment, best_score, best_details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-dataset-subdir", required=True)
    parser.add_argument("--target-dataset-subdir", required=True)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--height-bins", default="0,2.5,5,10,15,20,30,40,42")
    parser.add_argument("--height-min", type=float, default=0.0)
    parser.add_argument("--height-max", type=float, default=42.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-candidates", type=int, default=50000)
    parser.add_argument("--min-bin-count", type=int, default=20)
    parser.add_argument("--apply-eligibility-filters", dest="apply_eligibility_filters", action="store_true", default=True)
    parser.add_argument("--no-eligibility-filters", dest="apply_eligibility_filters", action="store_false")
    parser.add_argument("--min-s2-valid-fraction", type=float, default=0.90)
    parser.add_argument("--min-gedi-observations-per-patch", type=int, default=5)
    parser.add_argument("--gedi-high-threshold", type=float, default=10.0)
    parser.add_argument("--min-fraction-gedi-rh95-gt-high", "--min-fraction-gedi-rh95-ge10", dest="min_fraction_gedi_rh95_ge10", type=float, default=0.10)
    parser.add_argument("--max-abs-temporal-delta-days", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dataset = args.source_root / args.source_dataset_subdir
    target_dataset = args.source_root / args.target_dataset_subdir
    sample_in = source_dataset / "sample_catalog.csv"
    shot_in = source_dataset / "shot_catalog.csv.gz"
    if not sample_in.exists():
        raise FileNotFoundError(f"Missing source sample catalog: {sample_in}")
    if not shot_in.exists():
        raise FileNotFoundError(f"Missing source shot catalog: {shot_in}")
    if target_dataset.exists() and any(target_dataset.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Target dataset already exists and is not empty: {target_dataset}\n"
            f"Use --overwrite only if you intentionally want to rebuild it."
        )
    if target_dataset.exists() and args.overwrite:
        shutil.rmtree(target_dataset)
    target_dataset.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        (target_dataset / sp / "x").mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(sample_in, low_memory=False)
    shots = pd.read_csv(shot_in, low_memory=False)
    required_samples = {"sample_id", "sample_key", "patch_key", "spatial_group_id", "x_path"}
    required_shots = {"sample_id", "sample_key", "patch_key", "aux_shot_id", "rh95"}
    missing_samples = sorted(required_samples - set(samples.columns))
    missing_shots = sorted(required_shots - set(shots.columns))
    if missing_samples:
        raise RuntimeError(f"Source sample_catalog missing columns: {missing_samples}")
    if missing_shots:
        raise RuntimeError(f"Source shot_catalog missing columns: {missing_shots}")

    samples, shots, eligibility_audit, eligibility_table = apply_sample_eligibility_filters(samples, shots, args)
    eligibility_table.to_csv(target_dataset / "step04b_sample_eligibility_audit.csv", index=False)
    write_json(target_dataset / "step04b_eligibility_audit.json", eligibility_audit)

    samples["x_exists"] = samples["x_path"].astype(str).map(lambda p: Path(p).exists())
    if not bool(samples["x_exists"].all()):
        bad = samples.loc[~samples["x_exists"], ["sample_id", "x_path"]].head(10)
        raise FileNotFoundError(f"Missing source x_path examples:\n{bad.to_string(index=False)}")
    samples = samples.drop(columns=["x_exists"])

    height_bins = parse_bins(args.height_bins)
    targets = {
        "train": float(args.train_frac),
        "val": float(args.val_frac),
        "test": float(args.test_frac),
    }
    total_target = sum(targets.values())
    targets = {k: v / total_target for k, v in targets.items()}

    samples_c, shots_c = build_components(samples, shots)
    comp_tab = component_table(
        samples_c,
        shots_c,
        height_bins=height_bins,
        height_min=float(args.height_min),
        height_max=float(args.height_max),
    )
    print(
        f"[RESPLIT INPUT] samples={len(samples_c):,} | shots={len(shots_c):,} | "
        f"components={len(comp_tab):,} | targets={targets}",
        flush=True,
    )

    assignment, score, details = propose_assignment(
        comp_tab,
        targets=targets,
        height_bins=height_bins,
        min_bin_count=int(args.min_bin_count),
        seed=int(args.seed),
        n_candidates=int(args.n_candidates),
    )
    comp_to_split = assignment
    samples_out = samples_c.copy()
    samples_out["split"] = samples_out["_component_id"].astype(str).map(comp_to_split)
    sample_to_split = samples_out.set_index("sample_id")["split"].to_dict()
    shots_out = shots_c.copy()
    shots_out["split"] = shots_out["sample_id"].astype(str).map(sample_to_split)

    if samples_out["split"].isna().any() or shots_out["split"].isna().any():
        raise RuntimeError("Split assignment produced NaNs.")
    audits = {
        "patch_key_cross_split_leaks": leakage_count(samples_out, "patch_key"),
        "sample_key_cross_split_leaks": leakage_count(samples_out, "sample_key"),
        "spatial_group_cross_split_leaks": leakage_count(samples_out, "spatial_group_id"),
        "aux_shot_id_cross_split_leaks": leakage_count(shots_out, "aux_shot_id"),
    }
    if any(v != 0 for v in audits.values()):
        raise RuntimeError(f"Cross-split leakage detected after resplit: {audits}")

    drop_cols_samples = [c for c in ["_component_id"] if c in samples_out.columns]
    drop_cols_shots = [c for c in ["_component_id", "_spatial_group_id"] if c in shots_out.columns]
    samples_out.drop(columns=drop_cols_samples).to_csv(target_dataset / "sample_catalog.csv", index=False)
    shots_out.drop(columns=drop_cols_shots).to_csv(
        target_dataset / "shot_catalog.csv.gz",
        index=False,
        compression="gzip",
    )

    y = pd.to_numeric(shots_out["rh95"], errors="coerce")
    points = shots_out[np.isfinite(y) & (y >= args.height_min) & (y <= args.height_max)].copy()
    points["rh95"] = pd.to_numeric(points["rh95"], errors="coerce")
    unique_points = points.drop_duplicates("aux_shot_id").copy()
    unique_points["hbin"] = pd.cut(unique_points["rh95"], bins=height_bins, include_lowest=True)
    height_counts = pd.crosstab(unique_points["hbin"], unique_points["split"]).reindex(columns=["train", "val", "test"]).fillna(0).astype(int)
    height_counts.to_csv(target_dataset / "step04b_height_bin_unique_shots_by_split.csv")
    split_pct_within_bin = height_counts.div(height_counts.sum(axis=1).replace(0, np.nan), axis=0).mul(100).round(2)
    split_pct_within_bin.to_csv(target_dataset / "step04b_split_pct_within_height_bin.csv")
    comp_tab.assign(split=comp_tab["component_id"].astype(str).map(comp_to_split)).to_csv(
        target_dataset / "step04b_component_assignment.csv",
        index=False,
    )

    provenance = {
        "conversion": "step04b_spatial_height_stratified_resplit_v1",
        "source_root": str(args.source_root),
        "source_dataset": str(source_dataset),
        "target_dataset": str(target_dataset),
        "channel_order": CHANNEL_ORDER,
        "schema": {
            "in_channels": len(CHANNEL_ORDER),
            "temporal_window_days": 180,
            "height_range_train": [float(args.height_min), float(args.height_max)],
        },
        "eligibility_filter": eligibility_audit,
        "split_policy": {
            "unit": "connected_components(spatial_group_id, aux_shot_id)",
            "target": targets,
            "seed": int(args.seed),
            "n_candidates": int(args.n_candidates),
            "height_bins": height_bins.astype(float).tolist(),
            "objective": "global sample/unique-shot ratio + per-height-bin split balance + no empty major bins",
        },
    }
    write_json(target_dataset / "legacy_import_provenance.json", provenance)
    summary = {
        "status": "completed",
        "score": score,
        "score_details": details,
        "target_dataset": str(target_dataset),
        "samples_by_split": samples_out["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict(),
        "shots_by_split": shots_out["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict(),
        "unique_shots_by_split": unique_points.groupby("split")["aux_shot_id"].nunique().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict(),
        "components_by_split": pd.Series(comp_to_split).value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict(),
        "leakage_audit": audits,
        "eligibility_filter": eligibility_audit,
        "outputs": {
            "sample_catalog": str(target_dataset / "sample_catalog.csv"),
            "shot_catalog": str(target_dataset / "shot_catalog.csv.gz"),
            "component_assignment": str(target_dataset / "step04b_component_assignment.csv"),
            "height_bin_counts": str(target_dataset / "step04b_height_bin_unique_shots_by_split.csv"),
        },
    }
    write_json(target_dataset / "step04b_resplit_summary.json", summary)
    print("[RESPLIT DONE]", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
