from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import torch


PROJECT = Path(r"C:\Users\Dell\Desktop\Publication_Clarck\Natural_Sampling")
DATA_ROOT = Path(r"C:\Users\Dell\Desktop\Publication_Clarck\Data")
GROWTH_ROOT = PROJECT / "Source" / "Training" / "growthloss_v6"
OUTPUT_ROOT = PROJECT / "Inference_Harmonized_GEDIAnchored_NaturalP1"
NODATA = -9999.0
MONTHS = (5, 6, 7, 8, 9)
T = 4
MAP_YEARS = tuple(range(2018, 2026))
S2_INDEXES = (1, 2, 3, 7, 4, 5, 6, 8)
INFERENCE_CACHE = OUTPUT_ROOT / "_C15_IMAGE_ONLY_CACHE"


SITES = {
    "ifran": {
        "label": "Ifran", "ecosystem": "Dense", "target_year": 2020,
        "catalog": DATA_ROOT / "Dense" / "Ifran" / "Catalogs" / "final_catalog_C15_NATIVE",
        "s2_root": Path(r"E:\CHM\Ifran_6\DATA\S2\S2_MONTHLY_IFRAN_CLEAN_V1"),
        "s1_root": Path(r"E:\CHM\Ifran_6\DATA\S1_GEE_PREPROCESSED_2B_VALIDPIXELS\seasonal_MJJAS"),
        # The frozen Ifran cubes trace to this legacy seasonal product whose
        # filename retained the MAAMOURA token; calibration verifies R2>0.999.
        "s1_mode": "seasonal", "s1_token": "MAAMOURA",
        "phase1": PROJECT / "Models" / "Phase1" / "Ifran" / "IFRAN_PHASE1_NATURAL_SEED42_V1" / "checkpoints" / "best_slope.ckpt",
        "phase2": Path(r"C:\Users\Dell\Desktop\Publication_Clarck\Natural_Sampling\Ablations\Phase2_D_K_Lambda_Delta_VAL_Only\runs\ifran\IFRAN_B4_C15_CTRL_GL000_D5_K2_HD3_SEED42\checkpoints\best_compromise.ckpt"),
        "phase2_sha256": "41881083d90a9aeac8b8c4aa08e35d0142d94abc2b546737dead0af7c0982ffd",
        "max_height": 45.0,
    },
    "maamoura": {
        "label": "Maamoura", "ecosystem": "Low_Sparsity", "target_year": 2019,
        "catalog": DATA_ROOT / "Low_Sparsity" / "Maamoura" / "Temporal_Catalogs" / "T4_DENSE_2019_2025_C15",
        "s2_root": Path(r"E:\CHM\Maamoura\Data\S2_12_Bands"),
        "s1_root": Path(r"E:\CHM\Maamoura\Data\S1_GEE_PREPROCESSED_2B_VALIDPIXELS\seasonal_MJJAS"),
        "s1_mode": "seasonal", "s1_token": "MAAMOURA",
        "phase1": PROJECT / "Models" / "Phase1" / "Maamoura" / "MAAMOURA_PHASE1_NATURAL_SAMPLING_V2_SEED42" / "checkpoints" / "best_any.ckpt",
        "phase2": Path(r"C:\Users\Dell\Desktop\Publication_Clarck\Natural_Sampling\Ablations\Phase2_D_K_Lambda_Delta_VAL_Only\runs\maamoura\MAAMOURA_B4_C15_CTRL_GL000_D2_K2_HD3_SEED42\checkpoints\best_compromise.ckpt"),
        "phase2_sha256": "4d406f9edc55b3bda33a8c42f62d05518e816b9180de8a3abd93e71d52167bbd",
        "max_height": 20.0,
    },
    "agadir": {
        "label": "Agadir", "ecosystem": "Sparse", "target_year": 2020,
        "catalog": DATA_ROOT / "Sparse" / "Agadir" / "Catalogs" / "final_catalog",
        "s2_root": Path(r"E:\CHM\Agadir\DATA\S2_12_Bands"),
        "s1_root": Path(r"E:\CHM\Agadir\DATA\S1_GEE_PREPROCESSED_2B_VALIDPIXELS\seasonal_MJJAS"),
        "s1_mode": "seasonal", "s1_token": "AGADIR",
        "phase1": PROJECT / "Models" / "Phase1" / "Agadir" / "AGADIR_PHASE1_NATURAL_SEED42_V1" / "checkpoints" / "best_slope.ckpt",
        "phase2": Path(r"C:\Users\Dell\Desktop\Publication_Clarck\Natural_Sampling\Ablations\Phase2_D_K_Lambda_Delta_VAL_Only\runs\agadir\AGADIR_B4_C15_CTRL_GL000_D3_K3_HD3_SEED42\checkpoints\best_compromise.ckpt"),
        "phase2_sha256": "6addc2c4d752b80f69211ad498a230b86e7b45b16d20c9c51986208139283b8a",
        "max_height": 20.0,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_modules():
    root_text = str(GROWTH_ROOT.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from src.b4_echosat_adapter import B4EchoSatTwoHead, load_b4_reference
    return B4EchoSatTwoHead, load_b4_reference


def find_reference_raster(cfg: dict, year: int, month: int = 5) -> Path:
    candidates = sorted(cfg["s2_root"].rglob(f"*Y{year:04d}_M{month:02d}*.tif"))
    if not candidates:
        raise FileNotFoundError(f"S2 reference Y{year} M{month:02d}: {cfg['s2_root']}")
    return candidates[0]


def find_s1_raster(cfg: dict, orbit: str, year: int, month: int) -> Path:
    directory = cfg["s1_root"] / orbit / str(year)
    if cfg["s1_mode"] == "seasonal":
        pattern = f"S1_GEE_{cfg['s1_token']}_2B_Y{year}_*_{orbit}_*SEASONAL_{year}05_{year}09.tif"
    else:
        pattern = f"S1_GEE_{cfg['s1_token']}_2B_Y{year}_*_{orbit}_*MONTHLY_{year}{month:02d}_{year}{month:02d}.tif"
    candidates = sorted(directory.glob(pattern))
    if len(candidates) != 1:
        raise FileNotFoundError(f"S1 {orbit} Y{year} M{month:02d}: {directory}; candidates={candidates}")
    return candidates[0]


def read_patch(dataset, indexes: tuple[int, ...], r0: int, c0: int) -> np.ndarray:
    r1, c1 = r0 + 512, c0 + 512
    sr0, sc0 = max(0, r0), max(0, c0)
    sr1, sc1 = min(int(dataset.height), r1), min(int(dataset.width), c1)
    out = np.full((len(indexes), 512, 512), np.nan, np.float32)
    if sr1 <= sr0 or sc1 <= sc0:
        return out
    values = dataset.read(
        indexes,
        window=Window(sc0, sr0, sc1 - sc0, sr1 - sr0),
        out_dtype="float32",
    )
    dr0, dc0 = sr0 - r0, sc0 - c0
    out[:, dr0 : dr0 + values.shape[1], dc0 : dc0 + values.shape[2]] = values
    return out


def resolve_cube_path(cfg: dict, row) -> Path:
    path = Path(str(row.x_path))
    if not path.is_file():
        path = cfg["catalog"] / str(row.split) / "x" / f"{row.sample_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def raw_dynamic_patch(cfg: dict, year: int, month: int, r0: int, c0: int) -> np.ndarray:
    reference_path = find_reference_raster(cfg, year, month)
    with rasterio.open(reference_path) as reference:
        s2 = read_patch(reference, S2_INDEXES, r0, c0).transpose(1, 2, 0)
        vrt_args = {
            "crs": reference.crs, "transform": reference.transform,
            "width": reference.width, "height": reference.height,
            "resampling": Resampling.bilinear, "nodata": np.nan,
        }
        asc_path = find_s1_raster(cfg, "ASC", year, month)
        desc_path = find_s1_raster(cfg, "DESC", year, month)
        with rasterio.open(asc_path) as asc_src, rasterio.open(desc_path) as desc_src:
            with WarpedVRT(asc_src, **vrt_args) as asc, WarpedVRT(desc_src, **vrt_args) as desc:
                s1 = np.concatenate(
                    [read_patch(asc, (1, 2), r0, c0), read_patch(desc, (1, 2), r0, c0)], axis=0
                ).transpose(1, 2, 0)
    return np.dstack([s2, s1]).astype(np.float32)


def fit_raw_to_c15_transform(cfg: dict, samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Recover the exact affine preprocessing from raw rasters to stored C15.

    The transform is fitted only against already frozen 2019-2025 image cubes.
    It does not use GEDI targets and therefore cannot leak VAL/TEST labels.
    """
    calibration_path = INFERENCE_CACHE / cfg["label"] / "raw_to_c15_calibration.json"
    catalog_sha = sha256(cfg["catalog"] / "sample_catalog_step05.csv")
    if calibration_path.is_file():
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        if payload.get("catalog_sha256") == catalog_sha:
            return (
                np.asarray(payload["slopes"], np.float32),
                np.asarray(payload["intercepts"], np.float32),
                list(payload["diagnostics"]),
            )

    candidates = samples.sort_values(["year", "month", "patch_key"]).drop_duplicates(
        ["year", "month"], keep="first"
    ).head(12)
    pairs: list[list[np.ndarray]] = [[] for _ in range(12)]
    for row in candidates.itertuples(index=False):
        stored = np.load(resolve_cube_path(cfg, row), mmap_mode="r", allow_pickle=False)
        raw = raw_dynamic_patch(cfg, int(row.year), int(row.month), int(row.patch_row_start), int(row.patch_col_start))
        aoi = np.asarray(stored[:, :, 12]) > 0.5
        for channel in range(12):
            xx = raw[:, :, channel]
            yy = np.asarray(stored[:, :, channel])
            valid = aoi & np.isfinite(xx) & np.isfinite(yy) & (xx > -1000)
            selected = np.flatnonzero(valid.reshape(-1))[::128]
            if selected.size:
                pairs[channel].append(np.column_stack([xx.reshape(-1)[selected], yy.reshape(-1)[selected]]))
        del stored, raw

    slopes, intercepts, diagnostics = [], [], []
    for channel, chunks in enumerate(pairs):
        if not chunks:
            raise RuntimeError(f"No calibration pairs for dynamic C15 channel {channel}")
        values = np.concatenate(chunks, axis=0)
        x, y = values[:, 0], values[:, 1]
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        residual = y - fitted
        corr = float(np.corrcoef(x, y)[0, 1])
        rmse = float(np.sqrt(np.mean(residual * residual)))
        diag = {
            "channel": channel, "n": int(len(x)), "slope": float(slope),
            "intercept": float(intercept), "correlation": corr,
            "r2": corr * corr, "rmse_c15": rmse,
        }
        if not np.isfinite(corr) or corr < 0.985 or rmse > 0.20:
            raise RuntimeError(f"Unsafe raw-to-C15 reproduction for {cfg['label']}: {diag}")
        slopes.append(float(slope)); intercepts.append(float(intercept)); diagnostics.append(diag)

    payload = {
        "protocol": "raw dynamic S2/S1 affine fit against frozen C15 image cubes",
        "catalog_sha256": catalog_sha, "slopes": slopes, "intercepts": intercepts,
        "diagnostics": diagnostics, "uses_gedi_targets": False,
    }
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return np.asarray(slopes, np.float32), np.asarray(intercepts, np.float32), diagnostics


def ensure_image_only_year(cfg: dict, samples: pd.DataFrame, year: int) -> pd.DataFrame:
    """Create inference-only native C15 NPY cubes for a raw-data year."""
    if year in set(samples["year"].astype(int)):
        return samples
    for month in MONTHS:
        find_reference_raster(cfg, year, month)
        find_s1_raster(cfg, "ASC", year, month)
        find_s1_raster(cfg, "DESC", year, month)
    slopes, intercepts, diagnostics = fit_raw_to_c15_transform(cfg, samples)
    cache_root = INFERENCE_CACHE / cfg["label"] / f"Y{year}"
    templates = samples.sort_values(["patch_key", "year", "month"]).drop_duplicates("patch_key")
    rows = []
    for template in templates.itertuples(index=False):
        static_source = np.load(resolve_cube_path(cfg, template), mmap_mode="r", allow_pickle=False)
        static = np.asarray(static_source[:, :, 12:15], np.float32).copy()
        aoi = static[:, :, 0] > 0.5
        del static_source
        for month in MONTHS:
            sample_id = f"{template.patch_id}_Y{year}_M{month:02d}_INFERENCE_IMAGE_ONLY"
            path = cache_root / str(template.split) / "x" / f"{sample_id}.npy"
            if not path.is_file():
                raw = raw_dynamic_patch(
                    cfg, year, month, int(template.patch_row_start), int(template.patch_col_start)
                )
                dynamic = raw * slopes.reshape(1, 1, 12) + intercepts.reshape(1, 1, 12)
                invalid = ~np.isfinite(raw) | (raw <= -1000) | ~aoi[:, :, None]
                dynamic[invalid] = 0.0
                dynamic = np.clip(dynamic, -5.0, 5.0).astype(np.float32)
                cube = np.dstack([dynamic, static]).astype(np.float32)
                if cube.shape != (512, 512, 15) or not np.isfinite(cube).all():
                    raise RuntimeError((cfg["label"], year, month, template.patch_key, cube.shape))
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, cube, allow_pickle=False)
            row = template._asdict()
            row.update({
                "sample_id": sample_id, "sample_key": f"{cfg['label'].lower()}|{sample_id}",
                "year": year, "month": month, "x_path": str(path.resolve()),
                "n_valid_points": 0, "n_unique_gedi": 0,
            })
            rows.append(row)
    provenance = {
        "status": "READY_IMAGE_ONLY", "forest": cfg["label"], "year": year,
        "months": list(MONTHS), "channel_count": 15, "generated_rows": len(rows),
        "uses_gedi_targets": False, "source_catalog": str(cfg["catalog"]),
        "calibration_diagnostics": diagnostics,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "image_only_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return pd.concat([samples, pd.DataFrame(rows)], ignore_index=True, sort=False)


def choose_window(target_year: int, available_years: set[int]) -> tuple[int, ...]:
    candidates = []
    for start in range(target_year - T + 1, target_year + 1):
        years = tuple(range(start, start + T))
        if target_year in years and set(years).issubset(available_years):
            center_gap = abs((start + (T - 1) / 2) - target_year)
            candidates.append((center_gap, start, years))
    if not candidates:
        raise RuntimeError(f"No complete T={T} year window around {target_year}; available={sorted(available_years)}")
    return min(candidates)[2]


def raw_year_complete(cfg: dict, year: int) -> bool:
    try:
        for month in MONTHS:
            find_reference_raster(cfg, year, month)
            find_s1_raster(cfg, "ASC", year, month)
            find_s1_raster(cfg, "DESC", year, month)
    except FileNotFoundError:
        return False
    return True


def load_prediction_head(model, checkpoint: Path) -> None:
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("prediction_head", raw) if isinstance(raw, dict) else raw
    incompatible = model.prediction_head.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError((incompatible.missing_keys, incompatible.unexpected_keys))


def preflight(site_key: str) -> dict:
    cfg = SITES[site_key]
    catalog_csv = cfg["catalog"] / "sample_catalog_step05.csv"
    experiment = cfg["catalog"] / "experiment.json"
    required = [catalog_csv, experiment, cfg["phase1"], cfg["phase2"]]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing inference inputs:\n" + "\n".join(missing))
    actual_phase2_sha = sha256(cfg["phase2"])
    if actual_phase2_sha != cfg["phase2_sha256"]:
        raise RuntimeError(f"{site_key}: selected Phase-2 checkpoint hash mismatch")
    frame = pd.read_csv(catalog_csv, low_memory=False)
    exp = json.loads(experiment.read_text(encoding="utf-8"))
    channel_order = exp.get("channel_order") or exp.get("schema", {}).get("channel_order")
    if len(channel_order or []) != 15:
        raise RuntimeError(f"{site_key}: expected native C15, got {channel_order}")
    sample = Path(str(frame.iloc[0]["x_path"]))
    if not sample.is_file():
        sample = cfg["catalog"] / str(frame.iloc[0]["split"]) / "x" / f"{frame.iloc[0]['sample_id']}.npy"
    shape = np.load(sample, mmap_mode="r", allow_pickle=False).shape
    if tuple(shape) != (512, 512, 15):
        raise RuntimeError((site_key, sample, shape))
    catalog_years = set(pd.to_numeric(frame["year"], errors="raise").astype(int))
    raw_extension_years = {year for year in MAP_YEARS if year not in catalog_years and raw_year_complete(cfg, year)}
    years = catalog_years | raw_extension_years
    map_years = tuple(year for year in MAP_YEARS if year in years)
    sliding_windows = [
        tuple(range(start, start + T))
        for start in range(min(map_years), max(map_years) - T + 2)
        if set(range(start, start + T)).issubset(years)
    ]
    year_to_windows = {
        year: [window for window in sliding_windows if year in window]
        for year in map_years
    }
    if not sliding_windows or any(not windows for windows in year_to_windows.values()):
        raise RuntimeError(f"Incomplete sliding T4 coverage for {site_key}: {year_to_windows}")
    reference = find_reference_raster(cfg, int(cfg["target_year"]), 5)
    with rasterio.open(reference) as src:
        raster = {"crs": str(src.crs), "shape": [src.height, src.width], "resolution": list(map(abs, src.res))}
    return {
        "site": site_key, "label": cfg["label"], "target_year": cfg["target_year"],
        "map_years": list(map_years),
        "catalog_years": sorted(catalog_years),
        "raw_image_only_extension_years": sorted(raw_extension_years),
        "sliding_windows": [list(window) for window in sliding_windows],
        "year_to_contributing_windows": {
            str(year): [list(window) for window in windows]
            for year, windows in year_to_windows.items()
        },
        "aggregation_rule": "mean of every prediction from every complete sliding T4 window and M05-M09 containing the map year",
        "catalog_rows": len(frame),
        "unique_patches": int(frame["patch_key"].nunique()), "reference_raster": str(reference),
        "phase1_sha256": sha256(cfg["phase1"]), "phase2_sha256": actual_phase2_sha,
        "expected_phase2_sha256": cfg["phase2_sha256"],
        "raster": raster, "status": "PASS",
    }


def write_geotiff(path: Path, data: np.ndarray, reference: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference) as src:
        profile = src.profile.copy()
    profile.update(
        driver="GTiff", count=1, dtype="float32", nodata=NODATA,
        compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    output = np.where(np.isfinite(data), data, NODATA).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(output, 1)
        dst.set_band_description(1, "canopy_height_phase2_m")


def infer_site(site_key: str, overwrite: bool = False, save_monthly: bool = False) -> list[Path]:
    cfg = SITES[site_key]
    audit = preflight(site_key)
    map_years = tuple(int(year) for year in audit["map_years"])
    windows = [tuple(int(v) for v in values) for values in audit["sliding_windows"]]
    year_to_windows = {
        int(year): [tuple(int(v) for v in window) for window in values]
        for year, values in audit["year_to_contributing_windows"].items()
    }
    series_span = f"{min(map_years)}-{max(map_years)}"

    expected_paths: list[Path] = []
    all_reusable = not overwrite
    for year in map_years:
        root = OUTPUT_ROOT / cfg["ecosystem"] / cfg["label"] / "Phase2" / f"Y{year}"
        path = root / "Annual" / f"{cfg['label']}_B4_C15_Phase2_Y{year}_M05-09_T4_SLIDING_{series_span}_ENSEMBLE.tif"
        manifest_path = root / "QA" / "inference_manifest.json"
        expected_paths.append(path)
        if not (path.is_file() and manifest_path.is_file()):
            all_reusable = False
            continue
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_windows = [tuple(values) for values in old.get("contributing_windows", [])]
        if old.get("phase2_sha256") != audit["phase2_sha256"] or old_windows != year_to_windows[year]:
            all_reusable = False
    if all_reusable:
        for path in expected_paths:
            print(f"[REUSE] {path}", flush=True)
        return expected_paths

    B4EchoSatTwoHead, load_b4_reference = load_modules()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_model = load_b4_reference(cfg["phase1"], n_channels=15, base_ch=64, dropout=0.15, device=device)
    model = B4EchoSatTwoHead(reference_model, head_mode="residual", residual_scale=1.0).to(device)
    load_prediction_head(model, cfg["phase2"])
    model.eval()

    samples = pd.read_csv(cfg["catalog"] / "sample_catalog_step05.csv", low_memory=False)
    samples["year"] = pd.to_numeric(samples["year"], errors="raise").astype(int)
    samples["month"] = pd.to_numeric(samples["month"], errors="raise").astype(int)
    for year in audit["raw_image_only_extension_years"]:
        samples = ensure_image_only_year(cfg, samples, int(year))
    index = {
        (str(row.patch_key), int(row.year), int(row.month)): row
        for row in samples.itertuples(index=False)
    }
    patches = sorted(samples["patch_key"].astype(str).unique())
    references = {year: find_reference_raster(cfg, year, 5) for year in map_years}
    shapes = {}
    for year, reference in references.items():
        with rasterio.open(reference) as src:
            shapes[year] = (src.height, src.width)
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"Changing raster grid across years for {site_key}: {shapes}")
    height, width = next(iter(shapes.values()))
    annual_sum = {year: np.zeros((height, width), np.float32) for year in map_years}
    annual_count = {year: np.zeros((height, width), np.uint16) for year in map_years}
    qa_rows = {year: [] for year in map_years}
    monthly_accumulators = None
    if save_monthly:
        monthly_accumulators = {
            (year, month): [np.zeros((height, width), np.float32), np.zeros((height, width), np.uint16)]
            for year in map_years for month in MONTHS
        }

    # Every complete T4 window is evaluated, exactly as in Phase-2 training.
    # Interior years receive several temporal-context predictions; the final
    # annual map averages all window contexts and all leaf-on months.
    for sequence_years in windows:
        print(f"\n[{cfg['label']}] sliding window {sequence_years}", flush=True)
        for month in MONTHS:
            complete = [p for p in patches if all((p, year, month) in index for year in sequence_years)]
            print(f"  M{month:02d}: {len(complete)}/{len(patches)} complete patches", flush=True)
            for ordinal, patch_key in enumerate(complete, 1):
                records = [index[(patch_key, year, month)] for year in sequence_years]
                cubes = []
                for record in records:
                    path = resolve_cube_path(cfg, record)
                    cube = np.load(path, mmap_mode="r", allow_pickle=False)
                    if tuple(cube.shape) != (512, 512, 15):
                        raise RuntimeError((path, cube.shape))
                    cubes.append(np.asarray(cube, dtype=np.float32).transpose(2, 0, 1))
                tensor = torch.from_numpy(np.stack(cubes)[None]).to(device)
                with torch.inference_mode():
                    outputs = model(tensor)[0, 1].detach().cpu().numpy()
                for year in sequence_years:
                    temporal_index = sequence_years.index(year)
                    prediction = np.clip(outputs[temporal_index], 0.0, float(cfg["max_height"])).copy()
                    prediction[cubes[temporal_index][12] <= 0.5] = np.nan
                    record = records[temporal_index]
                    r0, c0 = int(record.patch_row_start), int(record.patch_col_start)
                    r1, c1 = min(r0 + 512, height), min(c0 + 512, width)
                    if r1 <= r0 or c1 <= c0:
                        continue
                    crop = prediction[: r1 - r0, : c1 - c0]
                    valid = np.isfinite(crop)
                    sum_region = annual_sum[year][r0:r1, c0:c1]
                    count_region = annual_count[year][r0:r1, c0:c1]
                    sum_region[valid] += crop[valid]
                    count_region[valid] += 1
                    if monthly_accumulators is not None:
                        month_sum, month_count = monthly_accumulators[(year, month)]
                        month_sum_region = month_sum[r0:r1, c0:c1]
                        month_count_region = month_count[r0:r1, c0:c1]
                        month_sum_region[valid] += crop[valid]
                        month_count_region[valid] += 1
                if ordinal % 10 == 0 or ordinal == len(complete):
                    print(f"    {ordinal}/{len(complete)}", flush=True)
                del tensor, outputs, cubes

            for year in sequence_years:
                qa_rows[year].append({
                    "month": month, "window": f"{sequence_years[0]}-{sequence_years[-1]}",
                    "temporal_index": sequence_years.index(year), "complete_patches": len(complete),
                })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    written_paths: list[Path] = []
    for year in map_years:
        root = OUTPUT_ROOT / cfg["ecosystem"] / cfg["label"] / "Phase2" / f"Y{year}"
        if monthly_accumulators is not None:
            for month in MONTHS:
                month_sum, month_count = monthly_accumulators[(year, month)]
                monthly = np.full((height, width), np.nan, np.float32)
                valid_month = month_count > 0
                monthly[valid_month] = month_sum[valid_month] / month_count[valid_month]
                write_geotiff(
                    root / "Monthly" / f"{cfg['label']}_B4_C15_Phase2_Y{year}_M{month:02d}_T4_SLIDING_{series_span}_ENSEMBLE.tif",
                    monthly, references[year],
                )
                del monthly
        annual = np.full((height, width), np.nan, np.float32)
        valid = annual_count[year] > 0
        annual[valid] = annual_sum[year][valid] / annual_count[year][valid]
        annual_path = root / "Annual" / f"{cfg['label']}_B4_C15_Phase2_Y{year}_M05-09_T4_SLIDING_{series_span}_ENSEMBLE.tif"
        qa_dir = root / "QA"
        write_geotiff(annual_path, annual, references[year])
        qa_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(qa_rows[year]).to_csv(qa_dir / "window_month_coverage.csv", index=False)
        manifest = {
            **audit, "target_year": year,
            "contributing_windows": [list(window) for window in year_to_windows[year]],
            "n_contributing_windows": len(year_to_windows[year]),
            "n_temporal_month_contexts": len(year_to_windows[year]) * len(MONTHS),
            "output": str(annual_path), "output_type": "catalog-footprint annual CHM",
            "aggregation": "pixelwise mean across all complete T4 contexts and M05-M09",
            "save_monthly": bool(save_monthly), "valid_annual_pixels": int(valid.sum()),
            "annual_mean_m": float(np.nanmean(annual)),
            "annual_p95_m": float(np.nanpercentile(annual, 95)), "nodata": NODATA,
            "phase2_checkpoint": str(cfg["phase2"]), "phase1_checkpoint": str(cfg["phase1"]),
        }
        (qa_dir / "inference_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written_paths.append(annual_path)
        print(f"[READY] {annual_path}", flush=True)
        del annual_sum[year], annual_count[year], annual

    series_manifest = OUTPUT_ROOT / cfg["ecosystem"] / cfg["label"] / "Phase2" / "sliding_T4_series_manifest.json"
    series_manifest.write_text(
        json.dumps({**audit, "outputs": [str(path) for path in written_paths]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    del model, reference_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact Phase-2 C15 inference for the three forest ecosystems.")
    parser.add_argument("--forest", choices=(*SITES, "all"), default="all")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-monthly", action="store_true")
    args = parser.parse_args()
    selected = tuple(SITES) if args.forest == "all" else (args.forest,)
    for site in selected:
        report = preflight(site)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        if args.run:
            infer_site(site, overwrite=args.overwrite, save_monthly=args.save_monthly)


if __name__ == "__main__":
    main()
