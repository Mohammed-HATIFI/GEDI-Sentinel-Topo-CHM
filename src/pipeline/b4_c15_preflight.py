from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio

from b4_c15_config import RAW_S2_ORDER, SOURCE_C11_CHANNEL_ORDER, SITES


MONTH_PATTERN = re.compile(r"S2_MONTHLY_CLEAN_Y(?P<year>20\d{2})_M(?P<month>\d{2})_AOI\.tif$", re.I)


@dataclass
class PreflightReport:
    site: str
    ready: bool
    expected_year_months: list[str]
    completed_year_months: list[str]
    missing_year_months: list[str]
    partial_downloads: list[str]
    errors: list[str]
    warnings: list[str]


def _read_array_header(npz_path: Path, key: str) -> tuple[int, ...] | None:
    filename = key + ".npy"
    with zipfile.ZipFile(npz_path) as archive:
        try:
            info = archive.getinfo(filename)
        except KeyError:
            return None
        with archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(stream)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(stream)
    return tuple(int(v) for v in shape)


def _source_channels(experiment_root: Path) -> list[str]:
    payload = json.loads((experiment_root / "experiment.json").read_text(encoding="utf-8"))
    schema = payload.get("schema", {})
    return list(payload.get("channel_order") or schema.get("channel_order") or [])


def _expected_year_months(experiment_root: Path) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for npz_path in sorted(experiment_root.rglob("*.npz")):
        with np.load(npz_path, allow_pickle=True) as data:
            if "s2_year_used" not in data.files or "s2_month_used" not in data.files:
                raise RuntimeError(f"Missing s2_year_used/s2_month_used: {npz_path}")
            years = np.asarray(data["s2_year_used"]).reshape(-1)
            months = np.asarray(data["s2_month_used"]).reshape(-1)
            result.update((int(y), int(m)) for y, m in zip(years, months))
    return result


def run_preflight(site_key: str) -> PreflightReport:
    cfg = SITES[site_key]
    errors: list[str] = []
    warnings: list[str] = []
    expected: set[tuple[int, int]] = set()
    completed: set[tuple[int, int]] = set()
    partials: list[str] = []

    if site_key == "ifran":
        for path, label in [
            (cfg.existing_experiment_root, "Ifran B4 experiment"),
            (cfg.existing_run_dir, "Ifran completed run"),
        ]:
            if path is None or not path.exists():
                errors.append(f"{label} missing: {path}")
        return PreflightReport(site_key, not errors, [], [], [], [], errors, warnings)

    required = [
        (cfg.source_npz_experiment, "source C11 NPZ experiment"),
        (cfg.s2_dir, "new S2 directory"),
        (cfg.dem, "DEM"),
        (cfg.slope, "slope"),
    ]
    for path, label in required:
        if path is None or not path.exists():
            errors.append(f"{label} missing: {path}")
    if errors:
        return PreflightReport(site_key, False, [], [], [], [], errors, warnings)

    assert cfg.source_npz_experiment is not None
    assert cfg.s2_dir is not None
    channels = _source_channels(cfg.source_npz_experiment)
    if tuple(channels) != SOURCE_C11_CHANNEL_ORDER:
        errors.append(f"C11 source channel order mismatch: {channels}")

    shards = sorted(cfg.source_npz_experiment.rglob("*.npz"))
    if not shards:
        errors.append(f"No NPZ shards found: {cfg.source_npz_experiment}")
    else:
        shape = _read_array_header(shards[0], "X")
        if not shape or len(shape) != 4 or shape[-1] != 11:
            errors.append(f"Expected NHWC C11 in {shards[0]}, got {shape}")
        try:
            expected = _expected_year_months(cfg.source_npz_experiment)
        except Exception as exc:
            errors.append(f"Cannot read source year/month metadata: {type(exc).__name__}: {exc}")

    exact_files: dict[tuple[int, int], Path] = {}
    for path in cfg.s2_dir.glob("S2_MONTHLY_CLEAN_Y*_M*_AOI.tif"):
        match = MONTH_PATTERN.match(path.name)
        if match:
            exact_files[(int(match.group("year")), int(match.group("month")))] = path
    partials = sorted(path.name for path in cfg.s2_dir.glob("*_S*.tif"))
    completed = set(exact_files)

    reference_signature = None
    for ym in sorted(expected & completed):
        path = exact_files[ym]
        try:
            with rasterio.open(path) as ds:
                signature = (ds.count, ds.crs.to_string() if ds.crs else None, ds.width, ds.height, tuple(ds.transform))
                if ds.count != len(RAW_S2_ORDER):
                    errors.append(f"{path.name}: expected {len(RAW_S2_ORDER)} bands {RAW_S2_ORDER}, got {ds.count}")
                if ds.res[0] != 10 or abs(ds.res[1]) != 10:
                    errors.append(f"{path.name}: expected 10 m pixels, got {ds.res}")
                if reference_signature is None:
                    reference_signature = signature
                elif signature != reference_signature:
                    errors.append(f"Grid mismatch: {path.name}")
        except Exception as exc:
            errors.append(f"Cannot open {path}: {type(exc).__name__}: {exc}")

    missing = sorted(expected - completed)
    if missing:
        warnings.append(
            "Download incomplete. Training/build will stay blocked until every required monthly monolithic TIFF exists."
        )
    if partials:
        warnings.append(f"{len(partials)} split download parts detected; they are ignored until merged into monthly AOI.tif files.")

    ready = not errors and not missing and bool(expected)
    return PreflightReport(
        site=site_key,
        ready=ready,
        expected_year_months=[f"{y}-{m:02d}" for y, m in sorted(expected)],
        completed_year_months=[f"{y}-{m:02d}" for y, m in sorted(expected & completed)],
        missing_year_months=[f"{y}-{m:02d}" for y, m in missing],
        partial_downloads=partials,
        errors=errors,
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=sorted(SITES), required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = run_preflight(args.site)
    payload = asdict(report)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
