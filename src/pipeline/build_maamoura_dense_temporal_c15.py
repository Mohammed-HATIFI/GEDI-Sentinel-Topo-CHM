from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


PROJECT = Path(r"C:\Users\Dell\Desktop\Publication_Clarck")
SOURCE = PROJECT / "Data" / "Low_Sparsity" / "Maamoura" / "Catalogs" / "final_catalog"
TARGET = PROJECT / "Data" / "Low_Sparsity" / "Maamoura" / "Temporal_Catalogs" / "T4_DENSE_2019_2025_C15"
BUILDING = TARGET.with_name(TARGET.name + "__BUILDING")
S2_ROOT = Path(r"E:\CHM\Maamoura\Data\S2_12_Bands")
S1_ROOT = Path(r"E:\CHM\Maamoura\Data\S1_GEE_PREPROCESSED_2B_VALIDPIXELS\seasonal_MJJAS")
CANONICAL_TEST_PREDICTIONS = PROJECT / "Results" / "Low_Sparsity" / "Maamoura" / "Predictions" / "test_predictions_full_domain.csv.gz"

YEARS = tuple(range(2019, 2026))
MONTHS = (5, 6, 7, 8, 9)
T = 4
CALIBRATION_YEARS = (2020, 2021, 2022, 2024)
S2_INDEXES = (1, 2, 3, 7, 4, 5, 6, 8)
CHANNEL_ORDER = (
    "S2_LEAFON_B02", "S2_LEAFON_B03", "S2_LEAFON_B04", "S2_LEAFON_B08",
    "S2_LEAFON_B05", "S2_LEAFON_B06", "S2_LEAFON_B07", "S2_LEAFON_B8A",
    "S1_ASC_VV", "S1_ASC_VH", "S1_DESC_VV", "S1_DESC_VH",
    "AOI_MASK", "DEM", "SLOPE",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def s2_path(year: int, month: int) -> Path:
    return S2_ROOT / f"S2_MONTHLY_CLEAN_Y{year}_M{month:02d}_AOI.tif"


def s1_path(orbit: str, year: int) -> Path:
    directory = S1_ROOT / orbit / str(year)
    expected = directory / f"S1_GEE_MAAMOURA_2B_Y{year}_LEAFON_{orbit}_SIGMA0_SEASONAL_{year}05_{year}09.tif"
    if expected.is_file():
        return expected
    candidates = sorted(directory.glob("*.tif"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"S1 {orbit} {year}: expected {expected}; candidates={candidates}")


def read_patch(dataset, indexes: tuple[int, ...], r0: int, c0: int) -> np.ndarray:
    r1, c1 = r0 + 512, c0 + 512
    src_r0, src_c0 = max(0, r0), max(0, c0)
    src_r1, src_c1 = min(int(dataset.height), r1), min(int(dataset.width), c1)
    output = np.full((len(indexes), 512, 512), np.nan, dtype=np.float32)
    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return output
    values = dataset.read(
        indexes,
        window=Window(src_c0, src_r0, src_c1 - src_c0, src_r1 - src_r0),
        out_dtype="float32",
    )
    dst_r0, dst_c0 = src_r0 - r0, src_c0 - c0
    output[:, dst_r0 : dst_r0 + values.shape[1], dst_c0 : dst_c0 + values.shape[2]] = values
    return output


def s1_vrts(year: int, reference):
    asc_src = rasterio.open(s1_path("ASC", year))
    desc_src = rasterio.open(s1_path("DESC", year))
    args = {
        "crs": reference.crs,
        "transform": reference.transform,
        "width": reference.width,
        "height": reference.height,
        "resampling": Resampling.bilinear,
        "nodata": np.nan,
    }
    return asc_src, desc_src, WarpedVRT(asc_src, **args), WarpedVRT(desc_src, **args)


def calibrate_s1(samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    pairs: list[list[np.ndarray]] = [[] for _ in range(4)]
    for year in CALIBRATION_YEARS:
        rows = samples[(samples["split"] == "train") & (samples["year"] == year)].head(12)
        if rows.empty:
            continue
        with rasterio.open(s2_path(year, 5)) as reference:
            asc_src, desc_src, asc, desc = s1_vrts(year, reference)
            try:
                for _, row in rows.iterrows():
                    r0, c0 = int(row["patch_row_start"]), int(row["patch_col_start"])
                    raw = np.concatenate(
                        [read_patch(asc, (1, 2), r0, c0), read_patch(desc, (1, 2), r0, c0)], axis=0
                    ).transpose(1, 2, 0)
                    source_x = np.load(str(row["x_path"]), mmap_mode="r", allow_pickle=False)
                    if tuple(source_x.shape) != (512, 512, 15):
                        raise RuntimeError(f"Canonical Phase1 cube is not native C15: {row['x_path']} {source_x.shape}")
                    aoi = np.asarray(source_x[:, :, 12]) > 0.5
                    for channel in range(4):
                        xx = raw[:, :, channel]
                        yy = np.asarray(source_x[:, :, 8 + channel])
                        valid = aoi & np.isfinite(xx) & np.isfinite(yy) & (xx > -1000) & (yy != 0)
                        flat = np.flatnonzero(valid.reshape(-1))[::100]
                        if flat.size:
                            pairs[channel].append(np.column_stack([xx.reshape(-1)[flat], yy.reshape(-1)[flat]]))
                    del source_x
            finally:
                asc.close(); desc.close(); asc_src.close(); desc_src.close()

    slopes, intercepts, diagnostics = [], [], []
    for name, chunks in zip(CHANNEL_ORDER[8:12], pairs):
        if not chunks:
            raise RuntimeError(f"No S1 calibration pairs for {name}")
        values = np.concatenate(chunks, axis=0)
        raw, normalized = values[:, 0], values[:, 1]
        slope, intercept = np.polyfit(raw, normalized, 1)
        predicted = slope * raw + intercept
        residual = normalized - predicted
        corr = float(np.corrcoef(raw, normalized)[0, 1])
        rmse = float(np.sqrt(np.mean(residual * residual)))
        diag = {
            "channel": name, "n": int(len(raw)), "slope": float(slope), "intercept": float(intercept),
            "correlation": corr, "r2": corr * corr, "rmse_normalized": rmse,
            "abs_residual_p95": float(np.percentile(np.abs(residual), 95)),
        }
        print("S1 CALIBRATION", diag, flush=True)
        if corr < 0.995 or rmse > 0.10:
            raise RuntimeError(f"S1 normalization cannot be reproduced safely: {diag}")
        diagnostics.append(diag)
        slopes.append(float(slope)); intercepts.append(float(intercept))
    return np.asarray(slopes, np.float32), np.asarray(intercepts, np.float32), diagnostics


def hardlink_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def sequence_support(samples: pd.DataFrame) -> tuple[dict[str, int], set[str]]:
    counts: dict[str, int] = {}
    covered: set[str] = set()
    for split in ("train", "val", "test"):
        frame = samples[samples["split"].astype(str).eq(split)]
        count = 0
        for (_, _), group in frame.groupby(["patch_key", "month"]):
            by_year = {int(row.year): str(row.sample_id) for row in group.itertuples()}
            for start in range(min(YEARS), max(YEARS) - T + 2):
                window = tuple(range(start, start + T))
                if all(year in by_year for year in window):
                    count += 1
                    covered.update(by_year[year] for year in window)
        counts[split] = count
    return counts, covered


def validate_canonical_test(shots: pd.DataFrame, covered_ids: set[str]) -> dict:
    test = shots.loc[
        shots["split"].astype(str).eq("test")
        & pd.to_numeric(shots["rh95"], errors="coerce").between(2.0, 20.0, inclusive="both")
    ].copy()
    if not set(test["sample_id"].astype(str)).issubset(covered_ids):
        missing = sorted(set(test["sample_id"].astype(str)) - covered_ids)
        raise RuntimeError(f"Canonical TEST sample IDs not covered by T4: {missing[:10]}")
    test["aux_shot_uid"] = test["aux_shot_uid"].astype(str)
    nearest = test.sort_values(
        ["aux_shot_uid", "aux_abs_temporal_delta_days", "sample_id"], kind="stable"
    ).drop_duplicates("aux_shot_uid", keep="first")
    if len(nearest) != 1799:
        raise RuntimeError(f"Canonical TEST should contain n=1799 unique-nearest shots, found {len(nearest)}")

    predictions = pd.read_csv(CANONICAL_TEST_PREDICTIONS, dtype={"aux_shot_uid": str}, low_memory=False)
    predictions = predictions.loc[
        pd.to_numeric(predictions["rh95"], errors="coerce").between(2.0, 20.0, inclusive="both")
    ].sort_values(["aux_shot_uid", "abs_temporal_delta_days", "batch_index"], kind="stable")
    predictions = predictions.drop_duplicates("aux_shot_uid", keep="first")
    if len(predictions) != 1799 or set(predictions["aux_shot_uid"]) != set(nearest["aux_shot_uid"]):
        raise RuntimeError("Dense temporal TEST IDs do not match the immutable Phase1 canonical TEST IDs")
    y = predictions["rh95"].to_numpy(float)
    p = predictions["prediction_original_coords"].to_numpy(float)
    r2 = 1.0 - float(np.sum((p - y) ** 2)) / float(np.sum((y - np.mean(y)) ** 2))
    if abs(r2 - 0.6865788573822955) > 1e-9:
        raise RuntimeError(f"Canonical Phase1 R2 lineage changed: {r2}")
    return {"n_unique_nearest": 1799, "phase1_r2": r2, "aux_shot_uid_set_verified": True}


def main() -> None:
    required = [
        SOURCE / "sample_catalog_step05.csv", SOURCE / "shot_catalog_step05.csv.gz",
        SOURCE / "normalization_train_only.json", SOURCE / "experiment.json",
        CANONICAL_TEST_PREDICTIONS,
        *(s2_path(year, month) for year in YEARS for month in MONTHS),
        *(s1_path(orbit, year) for orbit in ("ASC", "DESC") for year in YEARS),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dense temporal inputs:\n" + "\n".join(missing))
    if TARGET.exists() or BUILDING.exists():
        raise FileExistsError(f"Refusing overwrite: target={TARGET.exists()} building={BUILDING.exists()}")

    samples = pd.read_csv(SOURCE / "sample_catalog_step05.csv", low_memory=False)
    shots = pd.read_csv(SOURCE / "shot_catalog_step05.csv.gz", dtype={"aux_shot_uid": str}, low_memory=False)
    samples["year"] = pd.to_numeric(samples["year"], errors="raise").astype(int)
    samples["month"] = pd.to_numeric(samples["month"], errors="raise").astype(int)
    if samples["sample_id"].duplicated().any() or samples.duplicated(["patch_key", "year", "month"]).any():
        raise RuntimeError("Canonical sample catalogue has duplicate IDs or patch/year/month rows")
    split_per_patch = samples.groupby("patch_key")["split"].nunique()
    if int(split_per_patch.max()) != 1:
        raise RuntimeError("Spatial leakage: one patch appears in multiple splits")

    source_experiment = json.loads((SOURCE / "experiment.json").read_text(encoding="utf-8"))
    stored_order = source_experiment.get("channel_order") or source_experiment.get("schema", {}).get("channel_order")
    if tuple(stored_order or ()) != CHANNEL_ORDER:
        raise RuntimeError(f"Canonical C15 order mismatch: {stored_order}")

    slopes, intercepts, calibration = calibrate_s1(samples)
    stats = json.loads((SOURCE / "normalization_train_only.json").read_text(encoding="utf-8"))
    median = np.asarray(stats["median"][:8], np.float32).reshape(1, 1, 8)
    scale = np.asarray(stats["robust_scale_iqr_over_1p349"][:8], np.float32).reshape(1, 1, 8)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError("Invalid train-only S2 normalization scale")

    BUILDING.mkdir(parents=True, exist_ok=False)
    output_rows: list[pd.Series] = []
    link_modes = {"hardlink": 0, "copy": 0}
    source_by_key = {
        (str(row.patch_key), int(row.year), int(row.month)): row
        for row in samples.itertuples(index=False)
    }
    templates = samples.sort_values(["patch_key", "year", "month"]).drop_duplicates("patch_key")

    # First preserve every canonical sample ID and cube in a self-contained NPY catalogue.
    for _, row in samples.iterrows():
        target_x = BUILDING / str(row["split"]) / "x" / Path(str(row["x_path"])).name
        mode = hardlink_or_copy(Path(str(row["x_path"])), target_x)
        link_modes[mode] += 1
        updated = row.copy()
        updated["x_path"] = str((TARGET / target_x.relative_to(BUILDING)).resolve())
        output_rows.append(updated)

    generated = 0
    for year in YEARS:
        s2_datasets = {month: rasterio.open(s2_path(year, month)) for month in MONTHS}
        reference = s2_datasets[5]
        asc_src, desc_src, asc, desc = s1_vrts(year, reference)
        try:
            for _, template in templates.iterrows():
                patch_key = str(template["patch_key"])
                missing_months = [month for month in MONTHS if (patch_key, year, month) not in source_by_key]
                if not missing_months:
                    continue
                r0, c0 = int(template["patch_row_start"]), int(template["patch_col_start"])
                template_x = np.load(str(template["x_path"]), mmap_mode="r", allow_pickle=False)
                if tuple(template_x.shape) != (512, 512, 15):
                    raise RuntimeError(f"Template is not C15: {template['x_path']} {template_x.shape}")
                aoi = np.asarray(template_x[:, :, 12]) > 0.5
                static = np.asarray(template_x[:, :, 12:15], dtype=np.float32).copy()
                del template_x

                raw_s1 = np.concatenate(
                    [read_patch(asc, (1, 2), r0, c0), read_patch(desc, (1, 2), r0, c0)], axis=0
                ).transpose(1, 2, 0)
                s1 = raw_s1 * slopes.reshape(1, 1, 4) + intercepts.reshape(1, 1, 4)
                invalid_s1 = ~np.isfinite(raw_s1) | (raw_s1 <= -1000) | ~aoi[:, :, None]
                s1[invalid_s1] = 0.0
                s1 = np.clip(s1, -5.0, 5.0).astype(np.float32)

                for month in missing_months:
                    raw_s2 = read_patch(s2_datasets[month], S2_INDEXES, r0, c0).transpose(1, 2, 0)
                    s2 = (raw_s2 - median) / scale
                    invalid_s2 = ~np.isfinite(raw_s2) | (raw_s2 <= -1000) | ~aoi[:, :, None]
                    s2[invalid_s2] = 0.0
                    s2 = np.clip(s2, -5.0, 5.0).astype(np.float32)
                    cube = np.dstack([s2, s1, static]).astype(np.float32)
                    if cube.shape != (512, 512, 15) or not np.isfinite(cube).all():
                        raise RuntimeError(f"Invalid dense cube: patch={patch_key}, year={year}, month={month}")

                    split = str(template["split"])
                    sample_id = f"{template['patch_id']}_Y{year}_M{month:02d}_TEMPORAL_IMAGE_ONLY"
                    target_x = BUILDING / split / "x" / f"{sample_id}.npy"
                    target_x.parent.mkdir(parents=True, exist_ok=True)
                    np.save(target_x, cube, allow_pickle=False)

                    row = template.copy()
                    row["sample_id"] = sample_id
                    row["sample_key"] = f"maamoura|{sample_id}"
                    row["year"] = year
                    row["month"] = month
                    row["x_path"] = str((TARGET / target_x.relative_to(BUILDING)).resolve())
                    for name in ("n_valid_points", "n_unique_gedi"):
                        if name in row.index:
                            row[name] = 0
                    for name in ("rh95_median", "rh95_max"):
                        if name in row.index:
                            row[name] = np.nan
                    if "s2_valid_aoi_fraction" in row.index:
                        valid_s2 = np.all(np.isfinite(raw_s2) & (raw_s2 > -1000), axis=2)
                        row["s2_valid_aoi_fraction"] = float((valid_s2 & aoi).sum() / max(int(aoi.sum()), 1))
                    if "source_split" in row.index:
                        row["source_split"] = split
                    if "source_shard" in row.index:
                        row["source_shard"] = "DENSE_TEMPORAL_IMAGE_ONLY"
                    if "source_index" in row.index:
                        row["source_index"] = -1
                    output_rows.append(row)
                    generated += 1
                    print(f"WROTE {generated:03d}/389 {target_x}", flush=True)
        finally:
            asc.close(); desc.close(); asc_src.close(); desc_src.close()
            for dataset in s2_datasets.values():
                dataset.close()

    combined = pd.DataFrame(output_rows).sort_values(["split", "patch_key", "year", "month"]).reset_index(drop=True)
    expected_rows = len(templates) * len(YEARS) * len(MONTHS)
    if len(combined) != expected_rows or generated != expected_rows - len(samples):
        raise RuntimeError((len(combined), expected_rows, generated))
    if combined["sample_id"].duplicated().any() or combined.duplicated(["patch_key", "year", "month"]).any():
        raise RuntimeError("Dense catalogue has duplicate sample IDs or patch/year/month rows")
    counts_per_patch = combined.groupby("patch_key").size()
    if not counts_per_patch.eq(len(YEARS) * len(MONTHS)).all():
        raise RuntimeError(f"Incomplete dense patches: {counts_per_patch.to_dict()}")

    support, covered_ids = sequence_support(combined)
    expected_support = {
        split: int(combined.loc[combined["split"].eq(split), "patch_key"].nunique() * len(MONTHS) * (len(YEARS) - T + 1))
        for split in ("train", "val", "test")
    }
    if support != expected_support or not set(samples["sample_id"].astype(str)).issubset(covered_ids):
        raise RuntimeError(f"T4 coverage mismatch support={support} expected={expected_support}")
    canonical_test = validate_canonical_test(shots, covered_ids)

    combined.to_csv(BUILDING / "sample_catalog_step05.csv", index=False)
    combined.to_csv(BUILDING / "sample_catalog.csv", index=False)
    for name in (
        "shot_catalog_step05.csv.gz", "shot_catalog.csv.gz", "spatial_split_manifest.csv",
        "normalization_train_only.json", "train_shot_occurrence_counts.csv.gz",
        "lds_table_train_only.json", "component_assignment.csv", "build_summary.json",
        "fixed_split_domain_provenance.json", "fixed_v11_split_provenance.json",
        "legacy_import_provenance.json", "step05_summary.json",
    ):
        source = SOURCE / name
        if source.is_file():
            shutil.copy2(source, BUILDING / name)

    experiment = dict(source_experiment)
    experiment["experiment_root"] = str(TARGET)
    experiment["sample_catalog"] = str(TARGET / "sample_catalog_step05.csv")
    experiment["shot_catalog"] = str(TARGET / "shot_catalog_step05.csv.gz")
    experiment["channel_order"] = list(CHANNEL_ORDER)
    experiment["temporal_extension"] = {
        "protocol": "dense_same_month_T4_2019_2025_C15",
        "years": list(YEARS), "months": list(MONTHS), "window_length": T,
        "source_supervised_samples": int(len(samples)), "image_only_samples": int(generated),
        "supervised_gedi_for_generated_samples": False,
        "split_policy": "inherit immutable Phase1 patch split; never resplit",
        "purpose": "complete temporal context for frozen-parent GrowthLoss; no pseudo GEDI labels",
    }
    (BUILDING / "experiment.json").write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")

    provenance = {
        "status": "READY_DENSE_T4_2019_2025_C15",
        "source_catalog": str(SOURCE), "target_catalog": str(TARGET),
        "source_sample_catalog_sha256": sha256(SOURCE / "sample_catalog_step05.csv"),
        "years": list(YEARS), "months": list(MONTHS), "window_length": T,
        "channel_order": list(CHANNEL_ORDER), "drop_channels": [], "contains_npz": False,
        "source_rows_preserved": int(len(samples)), "generated_image_only_rows": int(generated),
        "total_rows": int(len(combined)), "source_npy_materialization": link_modes,
        "s1_calibration_years": list(CALIBRATION_YEARS), "s1_calibration": calibration,
        "s2_normalization_source": str(SOURCE / "normalization_train_only.json"),
        "aoi_dem_slope": "copied per immutable Phase1 patch",
        "split_policy": "same Phase1 spatial split; no resplitting and no cross-split patch overlap",
        "generated_targets": "none; shot catalogue unchanged and generated sample IDs have no GEDI rows",
        "t4_sequence_support": support, "all_source_sample_ids_covered_by_t4": True,
        "canonical_test": canonical_test,
    }
    (BUILDING / "temporal_catalog_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (BUILDING / "DENSE_T4_READY.ok").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # Paths were written against the final directory name; atomic rename publishes the catalogue.
    BUILDING.rename(TARGET)
    print("DENSE TEMPORAL CATALOG READY", TARGET, flush=True)
    print(json.dumps(provenance, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
