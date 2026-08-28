from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2] if "Source" in Path(__file__).parts else Path(r"C:\Users\Dell\Desktop\Publication_Clarck")
CONFIG = json.loads((ROOT / "Config" / "pipeline_config.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(frame: pd.DataFrame, lower: float, upper: float) -> dict[str, float]:
    frame = frame.loc[frame.rh95.between(lower, upper, inclusive="both")]
    y = frame.rh95.to_numpy(float)
    p = frame.prediction_original_coords.to_numpy(float)
    error = p - y
    corr = float(np.corrcoef(y, p)[0, 1])
    alpha = float(p.std(ddof=0) / y.std(ddof=0))
    beta = float(p.mean() / y.mean())
    return {
        "n": len(frame),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / np.sum((y - y.mean())**2)),
        "kge": float(1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)),
    }


for site, cfg in CONFIG["sites"].items():
    checkpoint = Path(cfg["checkpoint"])
    assert checkpoint.is_file(), checkpoint
    assert sha256(checkpoint) == cfg["checkpoint_sha256"], checkpoint

    catalog = Path(cfg["catalog"])
    samples = pd.read_csv(catalog / "sample_catalog_step05.csv")
    shots = pd.read_csv(catalog / "shot_catalog_step05.csv.gz")
    missing = [path for path in samples.x_path.drop_duplicates() if not Path(path).is_file()]
    assert not missing, f"{site}: {len(missing)} missing NPY arrays"
    for key in ("patch_key", "spatial_group_id"):
        if key not in samples:
            continue
        groups = {split: set(samples.loc[samples.split.eq(split), key].dropna()) for split in ("train", "val", "test")}
        assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]), (site, key)
    groups = {split: set(shots.loc[shots.split.eq(split), "aux_shot_uid"].dropna()) for split in ("train", "val", "test")}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]), (site, "GEDI")

    predictions = pd.read_csv(cfg["test_predictions"])
    predictions = predictions.sort_values(
        ["aux_shot_uid", "abs_temporal_delta_days", "batch_index"], kind="stable"
    ).drop_duplicates("aux_shot_uid")
    expected = cfg["selected_test_metrics"]
    actual = metrics(predictions, *cfg["primary_test_domain_m"])
    assert actual["n"] == expected["n"], (site, actual, expected)
    for key in ("mae", "rmse", "r2", "kge"):
        assert math.isclose(actual[key], expected[key], rel_tol=0, abs_tol=5e-10), (site, key, actual[key], expected[key])
    print(f"[OK] {site}: checkpoint, NPY, split leakage, and TEST metrics")

# Phase1 publication figures: exactly 16 triplets under the explicit Figures folders.
for extension in (".png", ".pdf", ".svg"):
    files = [file for file in (ROOT / "Results").rglob(f"*{extension}") if "Figures" in file.parts]
    assert len(files) == 16, (extension, len(files))
    print(f"[OK] {len(files)} Phase1 publication figures in {extension}")

# Phase2 comparison plots are PNG-only and live outside the Phase1 Figures contract.
phase2_png = [
    file for file in (ROOT / "Results").rglob("*.png")
    if "Canonical_Spatial_Split" in file.parts
]
assert all(file.name.startswith("phase1_vs_phase2_scatter_") for file in phase2_png), phase2_png
print(f"[OK] {len(phase2_png)} Phase2 comparison PNG(s); no stale file mixed with Phase1 figures")

dense = ROOT / "Data" / "Low_Sparsity" / "Maamoura" / "Temporal_Catalogs" / "T4_DENSE_2019_2025_C15"
provenance = json.loads((dense / "temporal_catalog_provenance.json").read_text(encoding="utf-8"))
assert provenance["status"] == "READY_DENSE_T4_2019_2025_C15"
assert provenance["total_rows"] == 630
assert provenance["generated_image_only_rows"] == 389
assert provenance["t4_sequence_support"] == {"train": 260, "val": 60, "test": 40}
assert provenance["canonical_test"]["n_unique_nearest"] == 1799
assert math.isclose(provenance["canonical_test"]["phase1_r2"], 0.6865788573822955, abs_tol=1e-12)
assert provenance["drop_channels"] == [] and provenance["contains_npz"] is False
print("[OK] Maamoura dense T4 C15: 630 rows, 260/60/40 sequences, canonical TEST n=1799")

for notebook_name in (
    "B4_C15_Final_Three_Ecosystems.ipynb",
    "B4_C15_Phase2_GrowthLoss_Three_Ecosystems.ipynb",
):
    notebook = ROOT / "Notebooks" / notebook_name
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    kernelspec = payload["metadata"]["kernelspec"]
    assert kernelspec["name"] in {".venv310", "python3"}
    assert ".venv310" in kernelspec.get("display_name", "")
    assert all("id" in cell for cell in payload["cells"])
    assert not any(output.get("output_type") == "error" for cell in payload["cells"] for output in cell.get("outputs", []))
    print(f"[OK] clean notebook JSON: {notebook}")
print("FINAL PIPELINE VALIDATION PASSED")
