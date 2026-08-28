from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from b4_c15_config import (
    B4_C15_CHANNEL_ORDER,
    B4_S2_ORDER,
    RAW_S2_ORDER,
    SITES,
    STEP05_SCRIPT,
    assert_c15_contract,
)
from b4_c15_preflight import run_preflight


PATCH_SIZE = 512
SEED = 42
TARGET_FRACTIONS = {"train": 0.75, "val": 0.15, "test": 0.10}
SOURCE_S1_INDICES = (4, 5, 6, 7)
SOURCE_AOI_INDEX = 10
S2_RAW_INDEX = {name: i + 1 for i, name in enumerate(RAW_S2_ORDER)}
S2_B4_RASTER_INDEX = tuple(S2_RAW_INDEX[name] for name in B4_S2_ORDER)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _scalar(values: np.ndarray, index: int) -> Any:
    value = values[index]
    return value.item() if isinstance(value, np.generic) else value


def _optional_vector(data: Any, key: str, sample_index: int, valid: np.ndarray, fill: Any) -> np.ndarray:
    available = data.files if hasattr(data, "files") else data.keys()
    if key not in available:
        return np.full(int(valid.sum()), fill)
    return np.asarray(data[key][sample_index])[valid]


def _clean_shot_id(values: np.ndarray, fallback: np.ndarray, site: str) -> np.ndarray:
    output: list[str] = []
    for value, rowid in zip(values, fallback):
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            text = f"{site}_ROW_{rowid}"
        output.append(text)
    return np.asarray(output, dtype=object)


def scan_source(site_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only metadata from the legacy C11 shards; X is loaded later per shard."""
    cfg = SITES[site_key]
    assert cfg.source_npz_experiment is not None
    sample_rows: list[dict[str, Any]] = []
    shot_frames: list[pd.DataFrame] = []
    serial = 0

    for shard_path in sorted(cfg.source_npz_experiment.rglob("*.npz")):
        with np.load(shard_path, allow_pickle=True) as data:
            required = {
                "X", "aux_rows", "aux_cols", "aux_y", "aux_mask",
                "patch_row_start", "patch_col_start", "s2_year_used", "s2_month_used",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise RuntimeError(f"{shard_path}: missing keys {missing}")
            # NpzFile does not cache members: indexing data[key] repeatedly would
            # decompress the same large array once per sample. Load each metadata
            # member exactly once and keep the multi-GB X member untouched here.
            metadata_keys = {
                "patch_row_start", "patch_col_start", "s2_year_used", "s2_month_used", "patch_id",
                "aux_rows", "aux_cols", "aux_y", "aux_mask", "aux_source_rowid", "aux_shot_id",
                "aux_track_id", "aux_gedi_date", "aux_gedi_ordinal", "aux_temporal_delta_days",
                "aux_abs_temporal_delta_days", "aux_lon", "aux_lat",
            }
            arrays = {key: np.asarray(data[key]) for key in metadata_keys if key in data.files}
            # Do not decompress the multi-gigabyte X member during the metadata pass.
            # The C11 shape is already checked from the NPY header by the preflight.
            n_samples = int(arrays["patch_row_start"].shape[0])

            for i in range(n_samples):
                r0 = int(_scalar(arrays["patch_row_start"], i))
                c0 = int(_scalar(arrays["patch_col_start"], i))
                year = int(_scalar(arrays["s2_year_used"], i))
                month = int(_scalar(arrays["s2_month_used"], i))
                patch_id_src = str(_scalar(arrays["patch_id"], i)) if "patch_id" in arrays else f"r{r0:06d}_c{c0:06d}"
                patch_id = f"{site_key.upper()}_r{r0:06d}_c{c0:06d}"
                sample_id = f"{patch_id}_Y{year:04d}_M{month:02d}_N{serial:05d}"
                patch_key = f"{site_key}|{patch_id}"
                sample_key = f"{site_key}|{sample_id}"
                valid = np.asarray(arrays["aux_mask"][i], dtype=bool)
                rh95 = np.asarray(arrays["aux_y"][i], dtype=np.float64)
                valid &= np.isfinite(rh95) & (rh95 >= cfg.train_min) & (rh95 <= cfg.train_max)

                source_rowid = _optional_vector(arrays, "aux_source_rowid", i, valid, -1)
                shot_id = _clean_shot_id(
                    _optional_vector(arrays, "aux_shot_id", i, valid, ""),
                    source_rowid,
                    site_key,
                )
                n_valid = int(valid.sum())
                source_split = shard_path.parent.name.lower()
                sample_rows.append(
                    {
                        "serial": serial,
                        "source_shard": str(shard_path),
                        "source_index": i,
                        "source_split": source_split,
                        "site_id": cfg.label,
                        "patch_id_source": patch_id_src,
                        "patch_id": patch_id,
                        "sample_id": sample_id,
                        "patch_key": patch_key,
                        "sample_key": sample_key,
                        "patch_row_start": r0,
                        "patch_col_start": c0,
                        "year": year,
                        "month": month,
                        "n_valid_points": n_valid,
                        "rh95_median": float(np.nanmedian(rh95[valid])) if n_valid else np.nan,
                        "rh95_max": float(np.nanmax(rh95[valid])) if n_valid else np.nan,
                    }
                )

                if n_valid:
                    frame = pd.DataFrame(
                        {
                            "serial": serial,
                            "sample_id": sample_id,
                            "sample_key": sample_key,
                            "patch_id": patch_id,
                            "patch_key": patch_key,
                            "aux_shot_id": shot_id,
                            "aux_track_id": _optional_vector(arrays, "aux_track_id", i, valid, -1),
                            "rh95": rh95[valid].astype(np.float32),
                            "local_row": np.asarray(arrays["aux_rows"][i])[valid].astype(np.int32),
                            "local_col": np.asarray(arrays["aux_cols"][i])[valid].astype(np.int32),
                            "aux_gedi_date": _optional_vector(arrays, "aux_gedi_date", i, valid, ""),
                            "aux_gedi_ordinal": _optional_vector(arrays, "aux_gedi_ordinal", i, valid, -1),
                            "aux_temporal_delta_days": _optional_vector(arrays, "aux_temporal_delta_days", i, valid, np.nan),
                            "aux_abs_temporal_delta_days": _optional_vector(arrays, "aux_abs_temporal_delta_days", i, valid, np.nan),
                            "aux_lon": _optional_vector(arrays, "aux_lon", i, valid, np.nan),
                            "aux_lat": _optional_vector(arrays, "aux_lat", i, valid, np.nan),
                            "aux_source_rowid": source_rowid,
                        }
                    )
                    shot_frames.append(frame)
                serial += 1

    samples = pd.DataFrame(sample_rows)
    shots = pd.concat(shot_frames, ignore_index=True) if shot_frames else pd.DataFrame()
    if samples.empty or shots.empty:
        raise RuntimeError(f"No usable samples/shots found in {cfg.source_npz_experiment}")
    if samples["sample_id"].duplicated().any():
        raise RuntimeError("Constructed sample_id values are not unique.")
    return samples, shots


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_spatial_splits(samples: pd.DataFrame, shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Connected components of patch_key and GEDI shot IDs, then deterministic 75/15/10 search."""
    patches = sorted(samples["patch_key"].astype(str).unique())
    uf = UnionFind(patches)
    for _, group in shots.groupby("aux_shot_id", sort=False):
        keys = group["patch_key"].astype(str).unique().tolist()
        for other in keys[1:]:
            uf.union(keys[0], other)
    component_for_patch = {key: uf.find(key) for key in patches}
    samples = samples.copy()
    shots = shots.copy()
    samples["component_root"] = samples["patch_key"].map(component_for_patch)
    shots["component_root"] = shots["patch_key"].map(component_for_patch)

    component_sizes = samples.groupby("component_root").size().sort_index()
    components = component_sizes.index.to_numpy(dtype=object)
    sizes = component_sizes.to_numpy(dtype=np.int64)
    total = int(sizes.sum())
    target = np.asarray([0.75, 0.15, 0.10], dtype=np.float64)
    split_names = np.asarray(["train", "val", "test"], dtype=object)
    rng = np.random.default_rng(SEED)
    best_score = math.inf
    best_assignment: dict[str, str] | None = None

    # Height-distribution term uses unique GEDI shots per component/bin.
    height_edges = np.asarray([0.0, 2.5, 5.0, 8.0, 10.0, 15.0, 20.0], dtype=float)
    unique_shots = shots.sort_values(["aux_shot_id", "sample_id"]).drop_duplicates("aux_shot_id")
    unique_shots["height_bin"] = np.clip(
        np.searchsorted(height_edges, unique_shots["rh95"].to_numpy(float), side="right") - 1,
        0,
        len(height_edges) - 2,
    )
    comp_hist = defaultdict(lambda: np.zeros(len(height_edges) - 1, dtype=np.float64))
    for (component, height_bin), count in unique_shots.groupby(["component_root", "height_bin"]).size().items():
        comp_hist[str(component)][int(height_bin)] = float(count)
    global_hist = sum((comp_hist[str(c)] for c in components), np.zeros(len(height_edges) - 1))
    valid_bins = global_hist > 0

    n_candidates = 15000 if len(components) > 3 else 1
    for _ in range(n_candidates):
        order = rng.permutation(len(components))
        counts = np.zeros(3, dtype=np.int64)
        assignment: dict[str, str] = {}
        for pos in order:
            deficits = target * total - counts
            split_index = int(np.argmax(deficits / np.maximum(target * total, 1.0)))
            assignment[str(components[pos])] = str(split_names[split_index])
            counts[split_index] += sizes[pos]
        if np.any(counts == 0):
            continue
        sample_error = float(np.abs(counts / total - target).sum())
        height_error = 0.0
        for split_index, split_name in enumerate(split_names):
            hist = sum(
                (comp_hist[str(component)] for component, value in assignment.items() if value == split_name),
                np.zeros(len(height_edges) - 1),
            )
            if valid_bins.any():
                height_error += float(np.mean(np.abs(hist[valid_bins] / global_hist[valid_bins] - target[split_index])))
        score = sample_error + 0.35 * height_error
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Could not create three non-empty connected-component splits.")
    samples["split"] = samples["component_root"].astype(str).map(best_assignment)
    shots["split"] = shots["component_root"].astype(str).map(best_assignment)
    roots = sorted(samples["component_root"].astype(str).unique())
    group_id = {root: f"SG_{index:04d}" for index, root in enumerate(roots)}
    samples["spatial_group_id"] = samples["component_root"].astype(str).map(group_id)
    shots["spatial_group_id"] = shots["component_root"].astype(str).map(group_id)

    def leakage(frame: pd.DataFrame, column: str) -> int:
        return int((frame.groupby(column)["split"].nunique() > 1).sum())

    audits = {
        "patch_key_cross_split_leaks": leakage(samples, "patch_key"),
        "sample_key_cross_split_leaks": leakage(samples, "sample_key"),
        "spatial_group_cross_split_leaks": leakage(samples, "spatial_group_id"),
        "gedi_shot_cross_split_leaks": leakage(shots, "aux_shot_id"),
    }
    if any(audits.values()):
        raise RuntimeError(f"Split leakage detected: {audits}")
    component_table = (
        samples.groupby(["spatial_group_id", "split"], as_index=False)
        .agg(n_samples=("sample_id", "size"), n_patches=("patch_key", "nunique"))
    )
    component_table["objective_score"] = best_score
    for key, value in audits.items():
        component_table[key] = value
    return samples, shots, component_table


class RasterPool:
    def __init__(self, site_key: str, reference_path: Path) -> None:
        cfg = SITES[site_key]
        assert cfg.s2_dir is not None and cfg.dem is not None and cfg.slope is not None
        self.cfg = cfg
        self.datasets: dict[tuple[int, int], rasterio.io.DatasetReader] = {}
        self.reference = rasterio.open(reference_path)
        vrt_args = {
            "crs": self.reference.crs,
            "transform": self.reference.transform,
            "width": self.reference.width,
            "height": self.reference.height,
            "resampling": Resampling.bilinear,
            "nodata": np.nan,
        }
        self.dem_source = rasterio.open(cfg.dem)
        self.slope_source = rasterio.open(cfg.slope)
        self.dem = WarpedVRT(self.dem_source, **vrt_args)
        self.slope = WarpedVRT(self.slope_source, **vrt_args)

    def monthly(self, year: int, month: int) -> rasterio.io.DatasetReader:
        key = (int(year), int(month))
        if key not in self.datasets:
            assert self.cfg.s2_dir is not None
            path = self.cfg.s2_dir / f"S2_MONTHLY_CLEAN_Y{year:04d}_M{month:02d}_AOI.tif"
            self.datasets[key] = rasterio.open(path)
        return self.datasets[key]

    @staticmethod
    def window(r0: int, c0: int) -> Window:
        return Window(int(c0), int(r0), PATCH_SIZE, PATCH_SIZE)

    def read_s2(self, year: int, month: int, r0: int, c0: int) -> np.ndarray:
        ds = self.monthly(year, month)
        return ds.read(
            S2_B4_RASTER_INDEX,
            window=self.window(r0, c0),
            boundless=True,
            fill_value=np.nan,
            out_dtype="float32",
        ).transpose(1, 2, 0)

    @staticmethod
    def _read_warped_patch(dataset: WarpedVRT, r0: int, c0: int) -> np.ndarray:
        """Read a PATCH_SIZE window from WarpedVRT with explicit boundless padding.

        Rasterio forbids ``boundless=True`` on WarpedVRT. We therefore clip the
        requested window to the VRT grid, read only the valid intersection, and
        place it in a NaN-filled 512 x 512 output array.
        """
        r0 = int(r0)
        c0 = int(c0)
        r1 = r0 + PATCH_SIZE
        c1 = c0 + PATCH_SIZE
        src_r0 = max(0, r0)
        src_c0 = max(0, c0)
        src_r1 = min(int(dataset.height), r1)
        src_c1 = min(int(dataset.width), c1)
        output = np.full((PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
        if src_r1 <= src_r0 or src_c1 <= src_c0:
            return output
        source_window = Window(
            col_off=src_c0,
            row_off=src_r0,
            width=src_c1 - src_c0,
            height=src_r1 - src_r0,
        )
        values = dataset.read(1, window=source_window, out_dtype="float32")
        dst_r0 = src_r0 - r0
        dst_c0 = src_c0 - c0
        output[
            dst_r0 : dst_r0 + values.shape[0],
            dst_c0 : dst_c0 + values.shape[1],
        ] = values
        return output

    def read_topography(self, r0: int, c0: int) -> tuple[np.ndarray, np.ndarray]:
        dem = self._read_warped_patch(self.dem, r0, c0)
        slope = self._read_warped_patch(self.slope, r0, c0)
        return dem, slope

    def close(self) -> None:
        for dataset in self.datasets.values():
            dataset.close()
        self.dem.close()
        self.slope.close()
        self.dem_source.close()
        self.slope_source.close()
        self.reference.close()

    def __enter__(self) -> "RasterPool":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _source_x(samples: pd.DataFrame) -> Iterable[tuple[pd.Series, np.ndarray]]:
    for shard_path, group in samples.groupby("source_shard", sort=True):
        with np.load(str(shard_path), allow_pickle=True) as data:
            x = np.asarray(data["X"], dtype=np.float32)
            for _, row in group.sort_values("source_index").iterrows():
                yield row, x[int(row["source_index"])]


def _robust_stats(samples: pd.DataFrame, pool: RasterPool) -> dict[str, list[float]]:
    rng = np.random.default_rng(SEED)
    values: list[list[np.ndarray]] = [[] for _ in range(10)]
    train = samples[samples["split"] == "train"]
    for row, source_x in _source_x(train):
        aoi = source_x[:, :, SOURCE_AOI_INDEX] > 0.5
        s2 = pool.read_s2(int(row["year"]), int(row["month"]), int(row["patch_row_start"]), int(row["patch_col_start"]))
        dem, slope = pool.read_topography(int(row["patch_row_start"]), int(row["patch_col_start"]))
        stack = np.dstack([s2, dem, slope])
        flat_valid = np.flatnonzero(aoi.reshape(-1))
        if flat_valid.size > 4096:
            flat_valid = rng.choice(flat_valid, size=4096, replace=False)
        flat = stack.reshape(-1, 10)
        for band in range(10):
            chosen = flat[flat_valid, band]
            chosen = chosen[np.isfinite(chosen) & (chosen > -1000)]
            if chosen.size:
                values[band].append(chosen.astype(np.float32, copy=False))
    median: list[float] = []
    scale: list[float] = []
    q01: list[float] = []
    q99: list[float] = []
    for band_values in values:
        if not band_values:
            raise RuntimeError("No valid train pixels available for robust normalization.")
        combined = np.concatenate(band_values)
        q25, med, q75 = np.nanpercentile(combined, [25, 50, 75])
        robust_scale = max(float(q75 - q25) / 1.349, 1e-6)
        lo, hi = np.nanpercentile(combined, [1, 99])
        median.append(float(med))
        scale.append(robust_scale)
        q01.append(float(lo))
        q99.append(float(hi))
    return {
        "median": median,
        "robust_scale_iqr_over_1p349": scale,
        "q01": q01,
        "q99": q99,
        "channel_order": [*B4_S2_ORDER, "DEM", "SLOPE"],
        "fit_split": "train only",
        "clip_z": [-5.0, 5.0],
        "seed": SEED,
    }


def _normalise(raw: np.ndarray, aoi: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    output = (raw.astype(np.float32) - median) / scale
    invalid = ~np.isfinite(raw) | (raw <= -1000) | ~aoi[:, :, None]
    output[invalid] = 0.0
    return np.clip(output, -5.0, 5.0).astype(np.float32)


def materialise_catalog(site_key: str, overwrite: bool = False) -> dict[str, Any]:
    cfg = SITES[site_key]
    if site_key == "ifran":
        raise ValueError("Ifran is historical and must not be rebuilt by this script.")
    report = run_preflight(site_key)
    if not report.ready:
        raise RuntimeError(
            f"{cfg.label} preflight is not ready. Missing months: {report.missing_year_months}; errors: {report.errors}"
        )
    assert cfg.s2_dir is not None
    root = cfg.catalog_root
    summary_path = root / "build_summary.json"
    if summary_path.exists() and not overwrite:
        print(f"[REUSE] Existing completed catalogue: {root}")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if root.exists():
        if not overwrite:
            # A previous interrupted launch can leave only the target directory.
            # Recover that harmless state without requiring destructive overwrite.
            if next(root.iterdir(), None) is None:
                print(f"[RECOVER] Removing empty target directory: {root}")
                root.rmdir()
            else:
                raise FileExistsError(
                    f"Incomplete non-empty target exists; inspect it and use --overwrite explicitly: {root}"
                )
        if root.exists():
            shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)

    samples, shots = scan_source(site_key)
    samples, shots, components = assign_spatial_splits(samples, shots)
    reference_path = cfg.s2_dir / f"S2_MONTHLY_CLEAN_Y{int(samples.iloc[0]['year']):04d}_M{int(samples.iloc[0]['month']):02d}_AOI.tif"
    with RasterPool(site_key, reference_path) as pool:
        stats = _robust_stats(samples, pool)
        median = np.asarray(stats["median"], dtype=np.float32).reshape(1, 1, 10)
        scale = np.asarray(stats["robust_scale_iqr_over_1p349"], dtype=np.float32).reshape(1, 1, 10)
        s2_valid_by_serial: dict[int, float] = {}
        x_path_by_serial: dict[int, str] = {}
        for row, source_x in _source_x(samples):
            split = str(row["split"])
            x_dir = root / split / "x"
            x_dir.mkdir(parents=True, exist_ok=True)
            path = x_dir / f"{row['sample_id']}.npy"
            aoi = source_x[:, :, SOURCE_AOI_INDEX] > 0.5
            s2 = pool.read_s2(int(row["year"]), int(row["month"]), int(row["patch_row_start"]), int(row["patch_col_start"]))
            dem, slope = pool.read_topography(int(row["patch_row_start"]), int(row["patch_col_start"]))
            raw10 = np.dstack([s2, dem, slope])
            normalized10 = _normalise(raw10, aoi, median, scale)
            s1 = source_x[:, :, SOURCE_S1_INDICES].astype(np.float32, copy=True)
            s1[~np.isfinite(s1)] = 0.0
            s1[~aoi] = 0.0
            x_c15 = np.dstack(
                [normalized10[:, :, :8], s1, aoi.astype(np.float32), normalized10[:, :, 8:10]]
            ).astype(np.float32)
            if x_c15.shape != (PATCH_SIZE, PATCH_SIZE, 15):
                raise RuntimeError(f"Unexpected output shape: {x_c15.shape}")
            np.save(path, x_c15, allow_pickle=False)
            valid_s2 = np.all(np.isfinite(s2) & (s2 > -1000), axis=2)
            denominator = max(int(aoi.sum()), 1)
            serial = int(row["serial"])
            s2_valid_by_serial[serial] = float((valid_s2 & aoi).sum() / denominator)
            x_path_by_serial[serial] = str(path.resolve())

    samples["s2_valid_aoi_fraction"] = samples["serial"].map(s2_valid_by_serial)
    samples["x_path"] = samples["serial"].map(x_path_by_serial)
    samples["n_valid_points"] = samples["sample_id"].map(shots.groupby("sample_id").size()).fillna(0).astype(int)
    samples["rh95_median"] = samples["sample_id"].map(shots.groupby("sample_id")["rh95"].median())
    samples["rh95_max"] = samples["sample_id"].map(shots.groupby("sample_id")["rh95"].max())

    sample_columns = [
        "split", "site_id", "patch_id", "sample_id", "patch_key", "sample_key",
        "spatial_group_id", "patch_row_start", "patch_col_start", "year", "month",
        "n_valid_points", "rh95_median", "rh95_max", "s2_valid_aoi_fraction", "x_path",
        "source_split", "source_shard", "source_index",
    ]
    shot_columns = [
        "split", "sample_id", "sample_key", "patch_id", "patch_key", "spatial_group_id",
        "aux_shot_id", "aux_track_id", "rh95", "local_row", "local_col", "aux_gedi_date",
        "aux_gedi_ordinal", "aux_temporal_delta_days", "aux_abs_temporal_delta_days",
        "aux_lon", "aux_lat", "aux_source_rowid",
    ]
    samples[sample_columns].to_csv(root / "sample_catalog.csv", index=False)
    shots[shot_columns].to_csv(root / "shot_catalog.csv.gz", index=False, compression="gzip")
    components.to_csv(root / "component_assignment.csv", index=False)
    write_json(root / "normalization_train_only.json", stats)
    assert_c15_contract(B4_C15_CHANNEL_ORDER)
    provenance = {
        "conversion": "legacy_C11_plus_new_S2_rededge_DEM_slope_to_native_B4_C15_v1",
        "source_root": str(cfg.source_npz_experiment),
        "target_dataset": str(root),
        "channel_order": list(B4_C15_CHANNEL_ORDER),
        "in_channels": 15,
        "native_c15_no_palsar": True,
        "s2_raw_band_order": list(RAW_S2_ORDER),
        "s2_model_band_order": list(B4_S2_ORDER),
        "source_s1_channels": [4, 5, 6, 7],
        "source_aoi_channel": 10,
        "split_policy": {
            "unit": "connected_components(patch_key, aux_shot_id)",
            "target": TARGET_FRACTIONS,
            "seed": SEED,
            "objective": "sample ratio plus unique-shot height-bin balance",
        },
        "preflight": asdict(report),
    }
    write_json(root / "legacy_import_provenance.json", provenance)
    experiment = {
        "storage_format": "npy_catalog_v1",
        "experiment_root": str(root),
        "sample_catalog": str(root / "sample_catalog.csv"),
        "shot_catalog": str(root / "shot_catalog.csv.gz"),
        "channel_order": list(B4_C15_CHANNEL_ORDER),
        "schema": {
            "in_channels": 15,
            "n_channels": 15,
            "channel_order": list(B4_C15_CHANNEL_ORDER),
            "patch_size": PATCH_SIZE,
            "stride": PATCH_SIZE,
            "temporal_window_days": 180,
            "max_height_m": cfg.train_max,
            "max_aux_k": 2048,
        },
        "site": cfg.label,
        "contract": "B4 native C15: 8 S2 + 4 S1 + AOI + DEM + SLOPE; no PALSAR",
    }
    write_json(root / "experiment.json", experiment)
    split_counts = samples["split"].value_counts().reindex(["train", "val", "test"], fill_value=0)
    summary = {
        "status": "CATALOG_C15_COMPLETE_STEP04",
        "site": cfg.label,
        "catalog_root": str(root),
        "n_samples": int(len(samples)),
        "n_shot_occurrences": int(len(shots)),
        "n_unique_shots": int(shots["aux_shot_id"].nunique()),
        "sample_split_counts": {key: int(value) for key, value in split_counts.items()},
        "sample_split_fractions": {key: float(value / len(samples)) for key, value in split_counts.items()},
        "channel_order": list(B4_C15_CHANNEL_ORDER),
        "in_channels": 15,
        "palsar_in_model_input": False,
        "normalization": str(root / "normalization_train_only.json"),
    }
    write_json(summary_path, summary)
    return summary


def run_step05(site_key: str) -> None:
    cfg = SITES[site_key]
    if not STEP05_SCRIPT.exists():
        raise FileNotFoundError(f"Vendored STEP05 script missing: {STEP05_SCRIPT}")
    bins = "0,2.5,5,8,10,15,20" if cfg.train_max <= 20 else "0,2.5,5,8,10,15,20,30,40,42"
    command = [
        str(cfg_python()), str(STEP05_SCRIPT),
        "--step04-root", str(cfg.catalog_root.parent),
        "--dataset-subdir", cfg.catalog_root.name,
        "--patch-stat", "p90",
        "--patch-bins", bins,
        "--height-min", str(cfg.train_min),
        "--height-max", str(cfg.train_max),
    ]
    subprocess.run(command, check=True)
    experiment_path = cfg.catalog_root / "experiment.json"
    payload = json.loads(experiment_path.read_text(encoding="utf-8"))
    payload["channel_order"] = list(B4_C15_CHANNEL_ORDER)
    payload.setdefault("schema", {})["in_channels"] = 15
    payload["schema"]["n_channels"] = 15
    payload["schema"]["channel_order"] = list(B4_C15_CHANNEL_ORDER)
    write_json(experiment_path, payload)
    assert_c15_contract(payload["channel_order"])


def cfg_python() -> Path:
    from b4_c15_config import PYTHON

    return PYTHON if PYTHON.exists() else Path("python")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build native B4 C15 catalogues for Maamoura or Agadir.")
    parser.add_argument("--site", choices=["maamoura", "agadir"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-step05", action="store_true")
    args = parser.parse_args()
    summary = materialise_catalog(args.site, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.skip_step05:
        run_step05(args.site)


if __name__ == "__main__":
    main()
