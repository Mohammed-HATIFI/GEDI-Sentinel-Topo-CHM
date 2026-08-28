from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CHANNEL_ORDER = [
    "S2_LEAFON_B02", "S2_LEAFON_B03", "S2_LEAFON_B04", "S2_LEAFON_B08",
    "S1_ASC_VV", "S1_ASC_VH", "S1_DESC_VV", "S1_DESC_VH",
    "PALSAR_HH", "PALSAR_HV", "AOI_MASK",
]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.name.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table: {path}")


def scalar_at(z: np.lib.npyio.NpzFile, key: str, i: int, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    if arr.ndim == 0:
        return arr.item()
    if arr.shape[0] <= i:
        return default
    val = arr[i]
    if np.asarray(val).ndim == 0:
        return np.asarray(val).item()
    return val


def vector_at(z: np.lib.npyio.NpzFile, key: str, i: int, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    if arr.ndim == 0:
        return default
    if arr.shape[0] <= i:
        return default
    return arr[i]


def safe_file_stem(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text or "sample"


def compute_s2_valid_fractions(x_hwc: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x_hwc)
    if x.ndim != 3 or x.shape[-1] < 4:
        return float("nan"), float("nan"), float("nan")
    s2 = x[..., :4]
    finite_s2 = np.all(np.isfinite(s2), axis=-1)
    all_frac = float(finite_s2.mean())
    nonzero_frac = float((finite_s2 & (np.nanmax(np.abs(s2), axis=-1) > 0)).mean())
    if x.shape[-1] >= 11:
        aoi = np.isfinite(x[..., 10]) & (x[..., 10] > 0.5)
    else:
        aoi = np.ones(finite_s2.shape, dtype=bool)
    if bool(aoi.any()):
        aoi_frac = float(finite_s2[aoi].mean())
    else:
        aoi_frac = all_frac
    return all_frac, nonzero_frac, aoi_frac


def to_float_or_nan(v: Any) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except Exception:
        return float("nan")


def to_int_or_default(v: Any, default: int = -1) -> int:
    try:
        if pd.isna(v):
            return default
        return int(round(float(v)))
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy Maamoura/Agadir NPZ shards into a Sans_NPZ per-sample .npy catalog source.")
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--target-step04-root", type=Path, required=True)
    parser.add_argument("--target-dataset-subdir", required=True)
    parser.add_argument("--spatial-group-block-px", type=int, default=1024)
    parser.add_argument("--height-min", type=float, default=0.0)
    parser.add_argument("--height-max", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    legacy_root = args.legacy_root
    target_root = args.target_step04_root / args.target_dataset_subdir
    if not legacy_root.exists():
        raise FileNotFoundError(f"Missing legacy root: {legacy_root}")
    if target_root.exists() and any(target_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Target exists and is not empty: {target_root}\\nUse --overwrite only if you intentionally rebuild this source NPY store.")
    if target_root.exists() and args.overwrite:
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (target_root / split / "x").mkdir(parents=True, exist_ok=True)

    sample_catalog_path = first_existing([
        legacy_root / "MASTER_CATALOG" / "sample_catalog.csv",
        legacy_root / "MASTER_CATALOG" / "sample_catalog.csv.gz",
        legacy_root / "sample_catalog.csv",
        legacy_root / "sample_catalog.csv.gz",
    ])
    if sample_catalog_path is not None:
        source_samples = read_table(sample_catalog_path)
        source_samples["sample_id"] = source_samples["sample_id"].astype(str)
        source_samples_idx = source_samples.drop_duplicates("sample_id").set_index("sample_id", drop=False)
    else:
        source_samples = pd.DataFrame()
        source_samples_idx = pd.DataFrame()

    exp_cfg = read_json(legacy_root / "experiment.json", default={}) or {}
    channel_order = (
        exp_cfg.get("schema", {}).get("channel_order")
        or exp_cfg.get("channel_order")
        or DEFAULT_CHANNEL_ORDER
    )

    sample_rows: list[dict[str, Any]] = []
    shot_rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    n_shards = 0

    for split in ("train", "val", "test"):
        split_dir = legacy_root / split
        shard_paths = sorted(split_dir.glob("*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"No NPZ shards found for split={split}: {split_dir}")
        for shard_idx, shard_path in enumerate(shard_paths, start=1):
            n_shards += 1
            with np.load(shard_path, allow_pickle=True) as z:
                if "X" not in z.files or "sample_id" not in z.files:
                    raise RuntimeError(f"{shard_path} must contain X and sample_id.")
                X = np.asarray(z["X"], dtype=np.float32)
                sample_ids = np.asarray(z["sample_id"]).astype(str).reshape(-1)
                if X.shape[0] != sample_ids.size:
                    raise RuntimeError(f"{shard_path}: X N={X.shape[0]} but sample_id N={sample_ids.size}")
                for i, sid_raw in enumerate(sample_ids):
                    sid = str(sid_raw)
                    if sid in seen_samples:
                        raise RuntimeError(f"Duplicate sample_id across legacy shards: {sid}")
                    seen_samples.add(sid)

                    x = X[i]
                    x_file = target_root / split / "x" / f"{safe_file_stem(sid)}.npy"
                    np.save(x_file, x, allow_pickle=False)

                    patch_id = str(scalar_at(z, "patch_id", i, default=""))
                    patch_row = to_int_or_default(scalar_at(z, "patch_row_start", i, default=0), 0)
                    patch_col = to_int_or_default(scalar_at(z, "patch_col_start", i, default=0), 0)
                    if not patch_id or patch_id.lower() in {"nan", "none"}:
                        patch_id = f"r{patch_row}_c{patch_col}"
                    sample_key = sid
                    patch_key = patch_id
                    block = max(int(args.spatial_group_block_px), 1)
                    spatial_group_id = f"block_r{patch_row // block}_c{patch_col // block}"

                    s2_all, s2_nonzero, s2_aoi = compute_s2_valid_fractions(x)
                    aux_y = vector_at(z, "aux_y", i, default=np.asarray([], dtype=np.float32))
                    aux_y = np.asarray(aux_y, dtype=np.float32).reshape(-1)
                    aux_mask = vector_at(z, "aux_mask", i, default=np.isfinite(aux_y))
                    aux_mask = np.asarray(aux_mask).astype(bool).reshape(-1) if np.asarray(aux_mask).size else np.isfinite(aux_y)
                    valid_y_domain = aux_mask & np.isfinite(aux_y) & (aux_y >= float(args.height_min)) & (aux_y <= float(args.height_max))
                    n_valid_domain = int(valid_y_domain.sum())
                    n_gt5 = int((aux_y[valid_y_domain] > 5.0).sum()) if n_valid_domain else 0
                    ratio_gt5 = float(n_gt5 / n_valid_domain) if n_valid_domain else 0.0

                    row = {
                        "split": split,
                        "sample_id": sid,
                        "sample_key": sample_key,
                        "patch_id": patch_id,
                        "patch_key": patch_key,
                        "spatial_group_id": spatial_group_id,
                        "patch_row_start": patch_row,
                        "patch_col_start": patch_col,
                        "x_path": str(x_file.resolve()),
                        "source_legacy_root": str(legacy_root),
                        "source_legacy_shard": str(shard_path),
                        "source_legacy_split": split,
                        "s2_valid_fraction_all": s2_all,
                        "s2_valid_fraction_nonzero": s2_nonzero,
                        "s2_valid_aoi_fraction": s2_aoi,
                        "n_valid_points": n_valid_domain,
                        "n_rh95_gt5": n_gt5,
                        "ratio_rh95_gt5": ratio_gt5,
                    }
                    for key in [
                        "repr_year", "anchor_month", "anchor_date", "anchor_ordinal",
                        "s2_year_used", "s2_month_used", "s1_asc_year_used",
                        "s1_desc_year_used", "palsar_year_used", "site_id",
                        "target_median", "target_mean", "n_total_patch_shots",
                        "n_temporal_shots", "n_valid_shots",
                    ]:
                        val = scalar_at(z, key, i, default=None)
                        if val is not None:
                            row[key] = val
                    if sid in source_samples_idx.index:
                        src_row = source_samples_idx.loc[sid]
                        for key in [
                            "label_domain", "train_label_domain", "val_test_label_domain",
                            "rh95_filter_protocol", "step04_protocol", "esa_protocol",
                        ]:
                            if key in source_samples_idx.columns:
                                row[key] = src_row[key]
                    sample_rows.append(row)

                    aux_rows = np.asarray(vector_at(z, "aux_rows", i, default=np.arange(aux_y.size))).reshape(-1)
                    aux_cols = np.asarray(vector_at(z, "aux_cols", i, default=np.arange(aux_y.size))).reshape(-1)
                    aux_rows_l2a = np.asarray(vector_at(z, "aux_rows_l2a", i, default=np.full(aux_y.size, np.nan))).reshape(-1)
                    aux_cols_l2a = np.asarray(vector_at(z, "aux_cols_l2a", i, default=np.full(aux_y.size, np.nan))).reshape(-1)
                    aux_shot_id = np.asarray(vector_at(z, "aux_shot_id", i, default=np.asarray([""] * aux_y.size))).astype(str).reshape(-1)
                    aux_lon = np.asarray(vector_at(z, "aux_lon", i, default=np.full(aux_y.size, np.nan))).reshape(-1)
                    aux_lat = np.asarray(vector_at(z, "aux_lat", i, default=np.full(aux_y.size, np.nan))).reshape(-1)
                    aux_temporal = np.asarray(vector_at(z, "aux_temporal_delta_days", i, default=np.full(aux_y.size, np.nan))).reshape(-1)
                    aux_abs_temporal = np.asarray(vector_at(z, "aux_abs_temporal_delta_days", i, default=np.full(aux_y.size, np.nan))).reshape(-1)
                    aux_gedi_date = np.asarray(vector_at(z, "aux_gedi_date", i, default=np.asarray([""] * aux_y.size))).astype(str).reshape(-1)
                    aux_ordinal = np.asarray(vector_at(z, "aux_gedi_ordinal", i, default=np.arange(aux_y.size))).reshape(-1)
                    aux_source = np.asarray(vector_at(z, "aux_source_rowid", i, default=np.full(aux_y.size, -1))).reshape(-1)

                    for j in np.where(aux_mask & np.isfinite(aux_y))[0]:
                        shot_id = str(aux_shot_id[j]) if j < aux_shot_id.size else ""
                        if not shot_id or shot_id.lower() in {"nan", "none", "0"}:
                            shot_id = f"{sid}:{j}"
                        lr = to_int_or_default(aux_rows[j] if j < aux_rows.size else j, int(j))
                        lc = to_int_or_default(aux_cols[j] if j < aux_cols.size else j, int(j))
                        shot_rows.append({
                            "split": split,
                            "sample_id": sid,
                            "sample_key": sample_key,
                            "patch_id": patch_id,
                            "patch_key": patch_key,
                            "spatial_group_id": spatial_group_id,
                            "aux_shot_id": shot_id,
                            # No robust real track ID is present in these legacy shards; keep a stable surrogate.
                            "aux_track_id": shot_id,
                            "aux_gedi_ordinal": to_int_or_default(aux_ordinal[j] if j < aux_ordinal.size else j, int(j)),
                            "rh95": float(aux_y[j]),
                            "local_row": lr,
                            "local_col": lc,
                            "row_l2a": to_int_or_default(aux_rows_l2a[j] if j < aux_rows_l2a.size else np.nan, -1),
                            "col_l2a": to_int_or_default(aux_cols_l2a[j] if j < aux_cols_l2a.size else np.nan, -1),
                            "lon": to_float_or_nan(aux_lon[j] if j < aux_lon.size else np.nan),
                            "lat": to_float_or_nan(aux_lat[j] if j < aux_lat.size else np.nan),
                            "aux_temporal_delta_days": to_float_or_nan(aux_temporal[j] if j < aux_temporal.size else np.nan),
                            "aux_abs_temporal_delta_days": to_float_or_nan(aux_abs_temporal[j] if j < aux_abs_temporal.size else np.nan),
                            "aux_gedi_date": str(aux_gedi_date[j]) if j < aux_gedi_date.size else "",
                            "aux_source_rowid": to_int_or_default(aux_source[j] if j < aux_source.size else -1, -1),
                        })
            print(f"[IMPORT] split={split} shard={shard_idx}/{len(shard_paths)} | {shard_path.name}", flush=True)

    samples = pd.DataFrame(sample_rows)
    shots = pd.DataFrame(shot_rows)
    if samples.empty or shots.empty:
        raise RuntimeError("Import produced empty samples or shots.")
    samples.to_csv(target_root / "sample_catalog.csv", index=False)
    shots.to_csv(target_root / "shot_catalog.csv.gz", index=False, compression="gzip")

    leakage = {
        "sample_ids_unique": int(samples["sample_id"].nunique()) == int(len(samples)),
        "n_samples": int(len(samples)),
        "n_shots": int(len(shots)),
        "samples_by_source_split": samples["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict(),
        "shots_by_source_split": shots["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict(),
    }
    provenance = {
        "conversion": "safi2_legacy_npz_to_sansnpz_npy_catalog_v1",
        "legacy_root": str(legacy_root),
        "target_dataset_root": str(target_root),
        "channel_order": channel_order,
        "spatial_group_policy": {
            "type": "coarse_blocks_from_patch_row_col",
            "block_px": int(args.spatial_group_block_px),
            "reason": "spatially separated split unit before height balancing",
        },
        "height_domain_m": [float(args.height_min), float(args.height_max)],
        "summary": leakage,
    }
    write_json(target_root / "legacy_import_provenance.json", provenance)
    write_json(target_root / "step04a_import_summary.json", {
        "status": "completed",
        "target_dataset": str(target_root),
        "n_shards": int(n_shards),
        **leakage,
        "sample_catalog": str(target_root / "sample_catalog.csv"),
        "shot_catalog": str(target_root / "shot_catalog.csv.gz"),
    })
    print("[IMPORT COMPLETE]", json.dumps(read_json(target_root / "step04a_import_summary.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
