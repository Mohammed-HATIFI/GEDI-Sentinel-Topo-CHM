from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_STEP04_ROOT = Path(
    r"E:\shifted_Pauls_Article_Ifran\Output\Sample_Store_NPY"
    r"\STEP04_IFRAN6_C11_PALSAR_AOIMASK_SPATIAL_NOLEAK_MAX45_75_15_10_NODUP_SANSNPZ_v36"
)
DEFAULT_DATASET_SUBDIR = "DATASET_NPY_CATALOG_PATCH512_STRIDE512_TW180_PAULS_TRACKID"
LEGACY_STEP04_BASE = Path(r"E:\shifted_Pauls_Article_Ifran\Output\Shard_NPZ")
CHANNEL_ORDER = ["S2_LEAFON_B02", "S2_LEAFON_B03", "S2_LEAFON_B04", "S2_LEAFON_B08", "S1_ASC_VV", "S1_ASC_VH", "S1_DESC_VV", "S1_DESC_VH", "PALSAR_HH", "PALSAR_HV", "AOI_MASK"]

def stable_int64(value: Any) -> int:
    text = str(value).strip()
    digest = hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)


def gaussian_smooth(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sigma = float(sigma_bins)
    if sigma <= 0 or values.size <= 1:
        return values
    radius = max(1, int(math.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= max(float(kernel.sum()), 1e-12)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def leakage_count(frame: pd.DataFrame, key: str) -> int:
    if key not in frame.columns:
        return -1
    return int((frame.groupby(key, dropna=False)["split"].nunique() > 1).sum())


def newest_legacy_step04() -> Path | None:
    if not LEGACY_STEP04_BASE.exists():
        return None
    candidates = [
        p for p in LEGACY_STEP04_BASE.iterdir()
        if p.is_dir()
        and "IFRAN6_ONLY" in p.name.upper()
        and (p / "sample_catalog.csv").exists()
        and (p / "shot_catalog.csv.gz").exists()
        and (p / "step04_build_summary.json").exists()
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def derive_gedi_date_from_track(track: Any) -> pd.Timestamp:
    text = str(track)
    match = re.search(r"_(20\d{2})(\d{3})\d*", text)
    if not match:
        return pd.NaT
    try:
        return pd.Timestamp(year=int(match.group(1)), month=1, day=1) + pd.Timedelta(
            days=int(match.group(2)) - 1
        )
    except Exception:
        return pd.NaT


def import_legacy_npz_step04(
    legacy_root: Path,
    target_dataset_root: Path,
) -> None:
    """Convert the existing v35 NPZ Step04 once instead of rebuilding preprocessing."""
    summary = json.loads((legacy_root / "step04_build_summary.json").read_text(encoding="utf-8"))
    patches_root = Path(summary.get("patches_root", ""))
    if not patches_root.exists():
        dirs = [
            p for p in legacy_root.iterdir()
            if p.is_dir() and all((p / split).exists() for split in ("train", "val", "test"))
        ]
        if not dirs:
            raise FileNotFoundError(f"Cannot locate legacy NPZ split directories under {legacy_root}")
        patches_root = dirs[0]

    samples = pd.read_csv(legacy_root / "sample_catalog.csv", low_memory=False)
    shots = pd.read_csv(legacy_root / "shot_catalog.csv.gz", low_memory=False)
    target_dataset_root.mkdir(parents=True, exist_ok=True)

    expected_ids = set(samples["sample_id"].astype(str))
    written_ids: set[str] = set()
    for split in ("train", "val", "test"):
        x_dir = target_dataset_root / split / "x"
        x_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = sorted((patches_root / split).glob("*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"No legacy NPZ shards for split={split}: {patches_root / split}")
        for shard_no, shard_path in enumerate(shard_paths, start=1):
            with np.load(shard_path, allow_pickle=True) as z:
                if "X" not in z.files or "sample_id" not in z.files:
                    raise RuntimeError(f"{shard_path} missing X/sample_id")
                X = np.asarray(z["X"], dtype=np.float32)
                sample_ids = np.asarray(z["sample_id"]).astype(str).reshape(-1)
                if X.shape[0] != sample_ids.size:
                    raise RuntimeError(
                        f"{shard_path}: X N={X.shape[0]} != sample_id N={sample_ids.size}"
                    )
                for i, sample_id in enumerate(sample_ids):
                    if sample_id not in expected_ids:
                        raise RuntimeError(f"Unknown legacy sample_id={sample_id} in {shard_path}")
                    np.save(x_dir / f"{sample_id}.npy", X[i], allow_pickle=False)
                    written_ids.add(sample_id)
            print(
                f"[LEGACY IMPORT] split={split} shard={shard_no}/{len(shard_paths)} "
                f"| {shard_path.name}",
                flush=True,
            )

    missing = sorted(expected_ids - written_ids)
    if missing:
        raise RuntimeError(f"Legacy conversion missed {len(missing)} samples, examples={missing[:10]}")

    samples["x_path"] = [
        str((target_dataset_root / str(split) / "x" / f"{sample_id}.npy").resolve())
        for split, sample_id in zip(samples["split"], samples["sample_id"])
    ]
    if "year" not in samples.columns or "month" not in samples.columns:
        extracted = samples["sample_id"].astype(str).str.extract(r"_Y(\d{4})_M(\d{2})$")
        samples["year"] = pd.to_numeric(extracted[0], errors="coerce")
        samples["month"] = pd.to_numeric(extracted[1], errors="coerce")

    sample_dates = samples.set_index("sample_id").apply(
        lambda r: pd.Timestamp(year=int(r["year"]), month=int(r["month"]), day=15),
        axis=1,
    )
    gedi_dates = shots["aux_track_id"].map(derive_gedi_date_from_track)
    center_dates = shots["sample_id"].map(sample_dates)
    temporal_delta = (gedi_dates - center_dates).dt.days
    shots["aux_gedi_date"] = gedi_dates.dt.strftime("%Y-%m-%d").fillna("")
    shots["aux_gedi_ordinal"] = shots.groupby("sample_id").cumcount().astype(np.int64)
    shots["aux_temporal_delta_days"] = temporal_delta.astype(float)
    shots["aux_abs_temporal_delta_days"] = temporal_delta.abs().astype(float)

    samples.to_csv(target_dataset_root / "sample_catalog.csv", index=False)
    shots.to_csv(
        target_dataset_root / "shot_catalog.csv.gz",
        index=False,
        compression="gzip",
    )
    provenance = {
        "conversion": "legacy_step04_npz_to_npy_catalog_v1",
        "legacy_root": str(legacy_root),
        "legacy_patches_root": str(patches_root),
        "target_dataset_root": str(target_dataset_root),
        "n_samples": int(len(samples)),
        "n_shots": int(len(shots)),
        "channel_order": summary.get("schema", {}).get("channel_order", CHANNEL_ORDER),
    }
    write_json(target_dataset_root / "legacy_import_provenance.json", provenance)
    print(
        f"[LEGACY IMPORT COMPLETE] samples={len(samples)} shots={len(shots)} "
        f"| target={target_dataset_root}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step05 catalog builder for SAFI_2 Maamoura/Agadir NPY sample stores; no large NPZ shards."
    )
    parser.add_argument("--step04-root", type=Path, default=DEFAULT_STEP04_ROOT)
    parser.add_argument("--dataset-subdir", default=DEFAULT_DATASET_SUBDIR)
    parser.add_argument("--patch-stat", choices=["p90", "max", "mean"], default="p90")
    parser.add_argument("--patch-bins", default="0,10,20,30,45")
    parser.add_argument("--height-min", type=float, default=0.0)
    parser.add_argument("--height-max", type=float, default=45.0)
    parser.add_argument("--lds-bin-width", type=float, default=0.5)
    parser.add_argument("--lds-sigma-bins", type=float, default=2.0)
    parser.add_argument("--lds-power", type=float, default=0.5)
    parser.add_argument("--shot-weight-clip-min", type=float, default=0.25)
    parser.add_argument("--shot-weight-clip-max", type=float, default=4.0)
    parser.add_argument("--height-weight-clip-min", type=float, default=0.5)
    parser.add_argument("--height-weight-clip-max", type=float, default=4.0)
    args = parser.parse_args()

    dataset_root = args.step04_root / str(args.dataset_subdir)
    sample_path = dataset_root / "sample_catalog.csv"
    shot_path = dataset_root / "shot_catalog.csv.gz"
    if not sample_path.exists() or not shot_path.exists():
        if os.environ.get("CHM_NO_LEGACY_NPZ_FALLBACK", "0") == "1":
            raise FileNotFoundError(
                "Requested STEP04 NPY catalog is missing. "
                "Run STEP04A then STEP04B first; automatic legacy NPZ/shard fallback is disabled. "
                f"sample={sample_path} shot={shot_path}"
            )
        legacy = newest_legacy_step04()
        if legacy is not None:
            print(
                f"[STEP05 FALLBACK] Requested Sans_NPZ Step04 is missing. "
                f"Importing existing legacy Step04: {legacy}",
                flush=True,
            )
            import_legacy_npz_step04(legacy, dataset_root)
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing Step04 sample catalog: {sample_path}")
    if not shot_path.exists():
        raise FileNotFoundError(f"Missing Step04 shot catalog: {shot_path}")

    samples = pd.read_csv(sample_path, low_memory=False)
    shots = pd.read_csv(shot_path, low_memory=False)
    required_samples = {"split", "sample_id", "sample_key", "patch_id", "patch_key", "spatial_group_id"}
    required_shots = {
        "split", "sample_id", "sample_key", "patch_id", "patch_key",
        "aux_shot_id", "aux_track_id", "rh95", "local_row", "local_col",
    }
    missing_samples = sorted(required_samples - set(samples.columns))
    missing_shots = sorted(required_shots - set(shots.columns))
    if missing_samples:
        raise RuntimeError(f"sample_catalog.csv missing columns: {missing_samples}")
    if missing_shots:
        raise RuntimeError(f"shot_catalog.csv.gz missing columns: {missing_shots}")

    samples["split"] = samples["split"].astype(str).str.lower()
    shots["split"] = shots["split"].astype(str).str.lower()
    if set(samples["split"].unique()) != {"train", "val", "test"}:
        raise RuntimeError(f"Unexpected sample splits: {sorted(samples['split'].unique())}")

    if "x_path" not in samples.columns:
        samples["x_path"] = [
            str((dataset_root / sp / "x" / f"{sid}.npy").resolve())
            for sp, sid in zip(samples["split"], samples["sample_id"])
        ]
    samples["x_exists"] = samples["x_path"].map(lambda p: Path(str(p)).exists())
    if not bool(samples["x_exists"].all()):
        bad = samples.loc[~samples["x_exists"], ["sample_id", "x_path"]].head(10)
        raise FileNotFoundError(f"Missing X files, examples:\n{bad.to_string(index=False)}")

    shots["rh95"] = pd.to_numeric(shots["rh95"], errors="coerce")
    shots = shots[
        np.isfinite(shots["rh95"])
        & (shots["rh95"] >= float(args.height_min))
        & (shots["rh95"] <= float(args.height_max))
    ].copy()
    if shots.empty:
        raise RuntimeError("No valid GEDI points remain in the requested height domain.")

    shots["aux_shot_uid"] = shots["aux_shot_id"].map(stable_int64).astype(np.int64)
    shots["aux_track_id_numeric"] = shots["aux_track_id"].map(stable_int64).astype(np.int64)
    if "aux_gedi_ordinal" not in shots.columns:
        shots["aux_gedi_ordinal"] = shots.groupby("sample_id").cumcount().astype(np.int64)
    for key, fill in (
        ("aux_temporal_delta_days", np.nan),
        ("aux_abs_temporal_delta_days", np.nan),
    ):
        if key not in shots.columns:
            shots[key] = fill
        shots[key] = pd.to_numeric(shots[key], errors="coerce")

    train_shots = shots[shots["split"] == "train"].copy()
    occurrence_counts = train_shots.groupby("aux_shot_uid").size().astype(np.int64)
    shots["aux_occurrence_count_train"] = (
        shots["aux_shot_uid"].map(occurrence_counts).fillna(1).astype(np.int64)
    )
    raw_shot_weight = 1.0 / train_shots["aux_shot_uid"].map(occurrence_counts).astype(float)
    shot_scale = float(raw_shot_weight.mean()) if len(raw_shot_weight) else 1.0
    shot_scale = max(shot_scale, 1e-12)
    shots["aux_shot_weight"] = 1.0
    train_mask = shots["split"] == "train"
    shots.loc[train_mask, "aux_shot_weight"] = (
        (1.0 / shots.loc[train_mask, "aux_occurrence_count_train"].astype(float)) / shot_scale
    ).clip(float(args.shot_weight_clip_min), float(args.shot_weight_clip_max))

    unique_train = (
        train_shots.sort_values(["aux_shot_uid", "sample_id"])
        .drop_duplicates("aux_shot_uid", keep="first")
    )
    bin_width = float(args.lds_bin_width)
    edges = np.arange(
        float(args.height_min),
        float(args.height_max) + bin_width + 1e-9,
        bin_width,
        dtype=np.float64,
    )
    hist, _ = np.histogram(unique_train["rh95"].to_numpy(dtype=np.float64), bins=edges)
    smooth = np.maximum(gaussian_smooth(hist, float(args.lds_sigma_bins)), 1e-6)
    density = smooth / max(float(smooth.sum()), 1e-12)
    weights = 1.0 / np.power(density, float(args.lds_power))
    empirical = hist.astype(np.float64)
    expected = (
        float((empirical * weights).sum() / empirical.sum())
        if empirical.sum() > 0
        else float(np.mean(weights))
    )
    weights = weights / max(expected, 1e-12)
    weights = np.clip(
        weights,
        float(args.height_weight_clip_min),
        float(args.height_weight_clip_max),
    ).astype(np.float32)
    height_idx = np.searchsorted(edges, shots["rh95"].to_numpy(dtype=np.float64), side="right") - 1
    height_idx = np.clip(height_idx, 0, len(weights) - 1)
    shots["aux_height_weight"] = weights[height_idx]

    grouped = shots.groupby("sample_id")["rh95"]
    if args.patch_stat == "p90":
        patch_stat = grouped.quantile(0.90)
    elif args.patch_stat == "max":
        patch_stat = grouped.max()
    else:
        patch_stat = grouped.mean()
    patch_edges = np.asarray([float(x) for x in str(args.patch_bins).split(",")], dtype=np.float64)
    if patch_edges.size < 2 or not np.all(np.diff(patch_edges) > 0):
        raise ValueError(f"Invalid patch bins: {patch_edges.tolist()}")
    samples[f"patch_height_{args.patch_stat}"] = samples["sample_id"].map(patch_stat)
    stat_values = samples[f"patch_height_{args.patch_stat}"].to_numpy(dtype=np.float64)
    patch_bin_idx = np.searchsorted(patch_edges, stat_values, side="right") - 1
    patch_bin_idx = np.clip(patch_bin_idx, 0, len(patch_edges) - 2)
    samples["patch_height_bin"] = patch_bin_idx.astype(np.int16)
    samples["patch_height_bin_label"] = [
        f"[{patch_edges[i]:g},{patch_edges[i + 1]:g}{']' if i == len(patch_edges) - 2 else ')'}"
        for i in patch_bin_idx
    ]
    sample_counts = shots.groupby("sample_id").size()
    sample_unique = shots.groupby("sample_id")["aux_shot_uid"].nunique()
    samples["n_gedi_occurrences"] = samples["sample_id"].map(sample_counts).fillna(0).astype(np.int32)
    samples["n_unique_gedi"] = samples["sample_id"].map(sample_unique).fillna(0).astype(np.int32)

    audits = {
        "patch_key_cross_split_leaks": leakage_count(samples, "patch_key"),
        "sample_key_cross_split_leaks": leakage_count(samples, "sample_key"),
        "spatial_group_cross_split_leaks": leakage_count(samples, "spatial_group_id"),
        "gedi_shot_cross_split_leaks": leakage_count(shots, "aux_shot_uid"),
    }
    if any(v > 0 for v in audits.values()):
        raise RuntimeError(f"Cross-split leakage detected: {audits}")

    sample_out = dataset_root / "sample_catalog_step05.csv"
    shot_out = dataset_root / "shot_catalog_step05.csv.gz"
    samples.drop(columns=["x_exists"]).to_csv(sample_out, index=False)
    shots.to_csv(shot_out, index=False, compression="gzip")
    occurrence_counts.rename("occurrence_count").to_csv(
        dataset_root / "train_shot_occurrence_counts.csv.gz",
        compression="gzip",
    )

    lds_payload = {
        "edges": edges.astype(float).tolist(),
        "weights": weights.astype(float).tolist(),
        "hist": hist.astype(int).tolist(),
        "smooth": smooth.astype(float).tolist(),
        "n_train_unique_shots": int(len(unique_train)),
        "range": [float(args.height_min), float(args.height_max)],
        "bin_width": bin_width,
        "sigma_bins": float(args.lds_sigma_bins),
        "power": float(args.lds_power),
        "clip_min": float(args.height_weight_clip_min),
        "clip_max": float(args.height_weight_clip_max),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }
    write_json(dataset_root / "lds_table_train_only.json", lds_payload)

    provenance_path = dataset_root / "legacy_import_provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else {}
    )
    experiment = {
        "storage_format": "npy_catalog_v1",
        "experiment_root": str(dataset_root),
        "sample_catalog": str(sample_out),
        "shot_catalog": str(shot_out),
        "channel_order": provenance.get("channel_order", CHANNEL_ORDER),
        "schema": {
            "in_channels": 11,
            "patch_size": 512,
            "stride": 512,
            "temporal_window_days": 180,
            "max_height_m": float(args.height_max),
            "max_aux_k": 2048,
        },
        "ablation_contract": {
            "A": "balanced patch sampler only",
            "B": "A plus train-only inverse GEDI occurrence weighting",
            "C": "B plus train-only LDS/KDE continuous height weighting",
            "patch_stat": args.patch_stat,
            "patch_bins": patch_edges.astype(float).tolist(),
        },
        "train_only_statistics": {
            "shot_occurrence_counts": "train_shot_occurrence_counts.csv.gz",
            "lds_table": "lds_table_train_only.json",
        },
        "leakage_audit": audits,
    }
    write_json(dataset_root / "experiment.json", experiment)

    summary = {
        "dataset_root": str(dataset_root),
        "storage_format": "npy_catalog_v1",
        "samples_by_split": samples["split"].value_counts().to_dict(),
        "shots_by_split": shots["split"].value_counts().to_dict(),
        "unique_shots_by_split": shots.groupby("split")["aux_shot_uid"].nunique().to_dict(),
        "patch_bins_by_split": (
            samples.groupby(["split", "patch_height_bin_label"]).size().unstack(fill_value=0).to_dict(orient="index")
        ),
        "train_occurrences": int(len(train_shots)),
        "train_unique_shots": int(train_shots["aux_shot_uid"].nunique()),
        "mean_occurrences_per_train_shot": float(
            len(train_shots) / max(1, train_shots["aux_shot_uid"].nunique())
        ),
        "leakage_audit": audits,
        "outputs": {
            "sample_catalog": str(sample_out),
            "shot_catalog": str(shot_out),
            "experiment_json": str(dataset_root / "experiment.json"),
            "lds_table": str(dataset_root / "lds_table_train_only.json"),
        },
    }
    write_json(dataset_root / "step05_summary.json", summary)

    print("\nSTEP05 NPY CATALOG COMPLETE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
