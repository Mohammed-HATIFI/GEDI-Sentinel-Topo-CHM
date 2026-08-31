#!/usr/bin/env python
"""Fast, data-free integrity checks for the GitHub release."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md", "CITATION.cff", "requirements.txt", "environment.yml",
    "configs/final_models.json", "configs/phase1_models.json", "configs/paths.example.yaml",
    "data/manifest.csv", "models/manifest.json", "docs/REPRODUCIBILITY.md",
    "notebooks/official/README.md", "src/pipeline", "src/training",
    "data/splits/ifran_spatial_split_manifest.csv",
    "data/splits/maamoura_spatial_split_manifest.csv",
    "data/splits/agadir_spatial_split_manifest.csv",
    "data/DATA_DICTIONARY.md",
    "data/processed/gedi/ifran/shot_catalog_step05.csv.gz",
    "data/processed/gedi/maamoura/shot_catalog_step05.csv.gz",
    "data/processed/gedi/agadir/shot_catalog_step05.csv.gz",
    "data/processed/evaluation/external_chm_strict_common_support.csv.gz",
    "data/geospatial/ifran_spatial_partitions.geojson",
    "data/geospatial/maamoura_spatial_partitions.geojson",
    "data/geospatial/agadir_spatial_partitions.geojson",
]

EXPECTED_NOTEBOOKS = {
    "01_GEDI_Preprocessing_and_Catalog_Audit.ipynb",
    "02_Study_Areas_Spatial_Splits_and_GEDI_Support.ipynb",
    "03_Phase1_Spatial_Canopy_Height_Model.ipynb",
    "04_Phase2_MultiYear_Residual_Refinement.ipynb",
    "05_Phase2_TEST_Diagnostics_and_Table2.ipynb",
    "06_Annual_CHM_Inference_and_Qualitative_Maps.ipynb",
    "07_External_CHM_Benchmark.ipynb",
    "08_Height_Stratified_MAE_and_Table_D1.ipynb",
    "09_MSE_Decomposition_Appendix_D.ipynb",
}


def main() -> int:
    failures = []
    for item in REQUIRED:
        if not (ROOT / item).exists():
            failures.append(f"missing: {item}")

    for path in ROOT.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"invalid JSON: {path.relative_to(ROOT)} ({exc})")
    for path in (ROOT / "configs").glob("*.yaml"):
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"invalid YAML: {path.relative_to(ROOT)} ({exc})")
    for path in ROOT.rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                next(csv.reader(handle))
        except Exception as exc:
            failures.append(f"invalid/empty CSV: {path.relative_to(ROOT)} ({exc})")

    registry = ROOT / "configs/final_models.json"
    if registry.exists():
        data = json.loads(registry.read_text(encoding="utf-8"))
        if data.get("test_used_for_selection") is not False:
            failures.append("final registry must state test_used_for_selection=false")
        if set(data.get("models", {})) != {"ifran", "maamoura", "agadir"}:
            failures.append("final registry must contain exactly three sites")
        if data.get("source_patch_shape") != [512, 512, 15]:
            failures.append("source patch contract must be [512, 512, 15]")
        if data.get("phase2_training_crop_shape") != [4, 15, 96, 96]:
            failures.append("Phase 2 training crop contract must be [4, 15, 96, 96]")

    phase1_registry = ROOT / "configs/phase1_models.json"
    if phase1_registry.exists():
        data = json.loads(phase1_registry.read_text(encoding="utf-8"))
        if data.get("test_used_for_selection") is not False:
            failures.append("Phase 1 registry must state test_used_for_selection=false")
        if set(data.get("models", {})) != {"ifran", "maamoura", "agadir"}:
            failures.append("Phase 1 registry must contain exactly three sites")

    notebook_dir = ROOT / "notebooks/official"
    notebooks = {path.name for path in notebook_dir.glob("*.ipynb")}
    if notebooks != EXPECTED_NOTEBOOKS:
        missing = sorted(EXPECTED_NOTEBOOKS - notebooks)
        extra = sorted(notebooks - EXPECTED_NOTEBOOKS)
        failures.append(f"official notebook set mismatch; missing={missing}, extra={extra}")

    model_manifest = ROOT / "models/manifest.json"
    if model_manifest.exists():
        data = json.loads(model_manifest.read_text(encoding="utf-8"))
        models = data.get("models", {})
        if len(models) != 6:
            failures.append("model manifest must contain three Phase 1 and three Phase 2 checkpoints")
        expected = {f"{site}_phase{phase}" for phase in (1, 2) for site in ("ifran", "maamoura", "agadir")}
        if set(models) != expected:
            failures.append("model manifest phase/site coverage is incomplete")

    forbidden_config_tokens = ("C:\\\\Users\\\\", "E:\\\\")
    for path in (ROOT / "configs").glob("*"):
        if path.is_file() and path.name != "paths.local.yaml":
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden_config_tokens:
                if token in text:
                    failures.append(f"machine-specific path in public config: {path.relative_to(ROOT)}")

    public_provenance = ROOT / "results/provenance/external_chm_raster_crs_provenance.csv"
    if public_provenance.exists():
        provenance_text = public_provenance.read_text(encoding="utf-8-sig", errors="ignore")
        for token in ("C:\\Users\\", "E:\\"):
            if token in provenance_text:
                failures.append(
                    "machine-specific path in public external-CHM provenance table"
                )

    manifest = ROOT / "data/manifest.csv"
    if manifest.exists():
        text = manifest.read_text(encoding="utf-8-sig")
        if "PLACEHOLDER" in text:
            failures.append("data manifest still contains a PLACEHOLDER row")

    executable = [ROOT / "code/phase1_training.py"]
    for path in executable:
        if path.exists() and "placeholder" in path.read_text(encoding="utf-8", errors="ignore").lower():
            failures.append(f"placeholder presented as executable: {path.relative_to(ROOT)}")

    if failures:
        print("RELEASE CHECK FAILED")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("RELEASE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
