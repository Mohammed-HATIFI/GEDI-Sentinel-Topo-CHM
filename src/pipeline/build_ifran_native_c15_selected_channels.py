from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\Dell\Desktop\Publication_Clarck")
SOURCE = ROOT / "Data" / "Dense" / "Ifran" / "Catalogs" / "final_catalog"
TARGET = ROOT / "Data" / "Dense" / "Ifran" / "Catalogs" / "final_catalog_C15_NATIVE"
STAGING = TARGET.with_name(TARGET.name + "__BUILDING")
TARGET_CHANNELS = (
    "S2_LEAFON_B02", "S2_LEAFON_B03", "S2_LEAFON_B04", "S2_LEAFON_B08",
    "S2_LEAFON_B05", "S2_LEAFON_B06", "S2_LEAFON_B07", "S2_LEAFON_B8A",
    "S1_ASC_VV", "S1_ASC_VH", "S1_DESC_VV", "S1_DESC_VH",
    "AOI_MASK", "DEM", "SLOPE",
)


def read_catalog(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or [])


def write_catalog(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


assert SOURCE.is_dir(), SOURCE
assert not TARGET.exists(), f"Catalogue cible déjà présent: {TARGET}"
assert not STAGING.exists(), f"Build partiel déjà présent: {STAGING}"

experiment = json.loads((SOURCE / "experiment.json").read_text(encoding="utf-8"))
source_channels = tuple(experiment.get("channel_order") or experiment.get("schema", {}).get("channel_order") or ())
assert len(source_channels) >= len(TARGET_CHANNELS), source_channels
assert all(channel in source_channels for channel in TARGET_CHANNELS)
selected_indices = tuple(source_channels.index(channel) for channel in TARGET_CHANNELS)
assert tuple(source_channels[index] for index in selected_indices) == TARGET_CHANNELS
assert not any("PALSAR" in channel.upper() for channel in TARGET_CHANNELS)

rows, fields = read_catalog(SOURCE / "sample_catalog_step05.csv")
assert rows and "x_path" in fields
source_paths = [Path(row["x_path"]).resolve() for row in rows]
assert len(source_paths) == len(set(source_paths))
assert all(path.is_file() and path.is_relative_to(SOURCE.resolve()) for path in source_paths)

source_bytes = sum(path.stat().st_size for path in source_paths)
estimated = int(source_bytes * len(TARGET_CHANNELS) / len(source_channels) * 1.08)
free = shutil.disk_usage(TARGET.parent).free
assert free > estimated, f"Espace insuffisant: libre={free / 2**30:.2f} GiB, requis~={estimated / 2**30:.2f} GiB"
print(
    f"Ifran C15 natif | {len(rows)} NPY | source={source_bytes / 2**30:.2f} GiB "
    f"| cible~={estimated / 2**30:.2f} GiB | libre={free / 2**30:.2f} GiB",
    flush=True,
)
print("Sélection directe:", list(zip(selected_indices, TARGET_CHANNELS)), flush=True)

STAGING.mkdir(parents=True)
new_rows = []
try:
    for number, (row, source_path) in enumerate(zip(rows, source_paths), 1):
        relative = source_path.relative_to(SOURCE.resolve())
        target_path = STAGING / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_array = np.load(source_path, mmap_mode="r", allow_pickle=False)
        assert source_array.ndim == 3 and source_array.shape[-1] == len(source_channels), (source_path, source_array.shape)
        # Select only the model channels at dataset construction time.
        selected = np.asarray(source_array[..., selected_indices])
        assert selected.shape[-1] == 15
        with target_path.open("wb") as stream:
            np.save(stream, selected, allow_pickle=False)
        check = np.load(target_path, mmap_mode="r", allow_pickle=False)
        assert check.shape[-1] == 15
        del check, selected, source_array
        updated = dict(row)
        updated["x_path"] = str(TARGET / relative)
        new_rows.append(updated)
        if number == 1 or number % 25 == 0 or number == len(rows):
            print(f"[C15] {number}/{len(rows)} | {relative}", flush=True)

    for source_item in SOURCE.iterdir():
        if source_item.is_dir() or source_item.name in {"experiment.json", "sample_catalog_step05.csv"}:
            continue
        shutil.copy2(source_item, STAGING / source_item.name)
    write_catalog(STAGING / "sample_catalog_step05.csv", new_rows, fields)

    experiment["experiment_root"] = str(TARGET)
    experiment["sample_catalog"] = str(TARGET / "sample_catalog_step05.csv")
    if experiment.get("shot_catalog"):
        experiment["shot_catalog"] = str(TARGET / Path(experiment["shot_catalog"]).name)
    experiment["channel_order"] = list(TARGET_CHANNELS)
    experiment.setdefault("schema", {})["in_channels"] = 15
    experiment["schema"]["n_channels"] = 15
    experiment["schema"]["channel_order"] = list(TARGET_CHANNELS)
    experiment["native_c15_build"] = {
        "source_catalog": str(SOURCE),
        "selection_mode": "target_channel_names_at_dataset_construction",
        "selected_source_indices": list(selected_indices),
        "selected_channel_order": list(TARGET_CHANNELS),
        "runtime_drop_channels": [],
        "palsar_written_to_target": False,
    }
    (STAGING / "experiment.json").write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")

    final_rows, _ = read_catalog(STAGING / "sample_catalog_step05.csv")
    assert len(final_rows) == len(rows)
    for row in final_rows:
        staged = STAGING / Path(row["x_path"]).relative_to(TARGET)
        assert staged.is_file()
        check = np.load(staged, mmap_mode="r", allow_pickle=False)
        assert check.shape[-1] == 15
        del check
    STAGING.rename(TARGET)
except Exception:
    print("Build interrompu; staging conservé:", STAGING, flush=True)
    raise

workflow_script = ROOT / "Source" / "Project" / "build_ifran_native_c15_selected_channels.py"
shutil.copy2(Path(__file__).resolve(), workflow_script)
print("[DONE]", TARGET, flush=True)
print("[SCRIPT]", workflow_script, flush=True)
