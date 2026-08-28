#!/usr/bin/env python
"""Rebuild hashes and sizes for the public data-release manifest."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

UPSTREAM = [
    ("gedi_l2a", "upstream_labels", "all", "raw/GEDI_L2A", "NASA_Earthdata", "follow_NASA_terms", "user_download_required"),
    ("sentinel1_grd", "upstream_predictor", "all", "raw/Sentinel-1", "Copernicus_Data_Space", "follow_Copernicus_terms", "user_download_required"),
    ("sentinel2_l2a", "upstream_predictor", "all", "raw/Sentinel-2", "Copernicus_Data_Space", "follow_Copernicus_terms", "user_download_required"),
    ("srtm_dem", "upstream_predictor", "all", "raw/SRTM", "USGS_EarthExplorer", "follow_USGS_terms", "user_download_required"),
    ("esa_worldcover", "study_area_audit", "all", "raw/ESA_WorldCover", "ESA_WorldCover", "follow_ESA_terms", "user_download_required"),
    ("dynamic_world", "study_area_audit", "all", "raw/Dynamic_World", "Google_Earth_Engine", "follow_provider_terms", "user_download_required"),
    ("derived_tensors", "model_inputs", "all", "derived/tensors", "research_archive", "deposit_required", "pending_archive"),
    ("model_checkpoints", "trained_models", "all", "models/checkpoints", "research_archive", "deposit_required", "pending_archive"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role_for(path: Path) -> str:
    posix = path.as_posix()
    if "/processed/gedi/" in f"/{posix}":
        return "filtered_gedi_catalog"
    if "/processed/evaluation/" in f"/{posix}":
        return "evaluation_support"
    if "/geospatial/" in f"/{posix}":
        return "spatial_partition_layer"
    if "/splits/" in f"/{posix}":
        return "spatial_split_manifest"
    return "release_metadata"


def site_for(path: Path) -> str:
    lower = path.as_posix().lower()
    for site in ("ifran", "maamoura", "agadir"):
        if site in lower:
            return site.capitalize()
    return "all"


def main() -> None:
    rows = []
    for logical_id, role, site, rel, source, redistribution, status in UPSTREAM:
        rows.append([logical_id, role, site, rel, "", "", source, redistribution, status])

    roots = [DATA / "processed", DATA / "geospatial", DATA / "splits"]
    files = sorted(path for root in roots if root.exists() for path in root.rglob("*") if path.is_file())
    for path in files:
        rel = path.relative_to(DATA).as_posix()
        logical_id = rel.replace("/", "_").replace(".", "_")
        rows.append([
            logical_id,
            role_for(path.relative_to(DATA)),
            site_for(path.relative_to(DATA)),
            rel,
            sha256(path),
            path.stat().st_size,
            "repository",
            "derived_research_data",
            "included",
        ])

    destination = DATA / "manifest.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["logical_id", "role", "site", "relative_path", "sha256", "size_bytes", "source_or_archive", "redistribution", "status"])
        writer.writerows(rows)
    print(f"Wrote {destination} with {len(rows)} rows")


if __name__ == "__main__":
    main()
