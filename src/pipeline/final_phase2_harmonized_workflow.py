from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


NATURAL_ROOT = Path(r"C:\Users\Dell\Desktop\Publication_Clarck\Natural_Sampling")
PROJECT_SOURCE = NATURAL_ROOT / "Source" / "Project"
BASE_WORKFLOW_PATH = PROJECT_SOURCE / "final_phase2_aoi_workflow.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("clarck_base_phase2_aoi", BASE_WORKFLOW_PATH)
_original_configure = base.configure


def configure(engine, forest: str):
    candidate_id, _ = _original_configure(engine, forest)

    def isolated_roots(forest_key: str):
        cfg = engine.FORESTS[forest_key]
        model_root = (
            NATURAL_ROOT / "Models" / "Phase2_Harmonized_GEDIAnchored_NaturalP1"
            / cfg["ecosystem"] / cfg["label"]
        )
        result_root = (
            NATURAL_ROOT / "Results" / "Phase2_Harmonized_GEDIAnchored_NaturalP1"
            / cfg["ecosystem"] / cfg["label"]
        )
        return model_root, result_root

    engine.roots = isolated_roots
    engine.FORESTS[forest]["ablation_family"] = "Phase2_Harmonized_GEDIAnchored_NaturalP1"
    return candidate_id, engine.run_dir(forest, candidate_id)


base.configure = configure


def install_harmonized_build_data(engine):
    def record_has_valid_gedi(record, valid_sample_ids: set[str]) -> bool:
        return any(str(sample_id) in valid_sample_ids for sample_id in record.sample_ids)

    def harmonized_build_data(forest_key: str, modules: dict, include_test: bool = False):
        cfg = engine.FORESTS[forest_key]
        samples, shots = modules["load_catalogs"](cfg["catalog"])
        valid = shots.copy()
        valid["rh95"] = pd.to_numeric(valid["rh95"], errors="coerce")
        valid = valid[valid["rh95"].notna() & (valid["rh95"] > 0)].copy()
        valid_sample_ids = set(valid["sample_id"].astype(str))
        splits = ("train", "val", "test") if include_test else ("train", "val")
        raw_records = {
            split: modules["build_same_month_sequences"](
                samples,
                split=split,
                all_years=engine.YEARS,
                window_length=4,
                leaf_on_months=engine.MONTHS,
            )
            for split in splits
        }
        records = {
            split: [
                record for record in split_records
                if record_has_valid_gedi(record, valid_sample_ids)
            ]
            for split, split_records in raw_records.items()
        }
        for split in splits:
            removed = len(raw_records[split]) - len(records[split])
            print(
                f"[HARMONIZED GEDI-ANCHORED RECORDS] {forest_key} {split}: "
                f"kept={len(records[split])} removed_image_only={removed} raw={len(raw_records[split])}",
                flush=True,
            )
            if not records[split]:
                raise RuntimeError(f"{forest_key}/{split}: no GEDI-anchored T4 sequence remains")
        first = np.load(records["train"][0].x_paths[0], mmap_mode="r", allow_pickle=False)
        if tuple(first.shape) != (512, 512, 15):
            raise RuntimeError(first.shape)
        del first
        return samples, shots, records

    engine.build_data = harmonized_build_data


def preflight(engine, forest: str):
    candidate_id, run_dir = configure(engine, forest)
    modules = base.prepare_modules(engine)
    _, shots, records = engine.build_data(forest, modules, include_test=True)
    cfg = engine.FORESTS[forest]
    parent = Path(cfg["parent"])
    selected = Path(base.FINAL_MODELS[forest]["selected_checkpoint"])
    selected_hash = base.sha256_file(selected) if selected.is_file() else None
    report = {
        "forest": forest,
        "candidate_id": candidate_id,
        "catalog": str(cfg["catalog"]),
        "phase1_parent": str(parent),
        "phase1_parent_sha256": engine.sha256(parent),
        "expected_phase1_parent_sha256": cfg["parent_sha"],
        "training_sequence_rule": "complete same-month T4 containing at least one valid GEDI shot",
        "image_only_annual_inputs_allowed_inside_anchored_T4": True,
        "fully_image_only_T4_used_for_training": False,
        "records": {split: len(values) for split, values in records.items()},
        "run_dir": str(run_dir),
        "run_already_complete": (run_dir / "TRAIN_DONE.json").is_file(),
        "selected_checkpoint": str(selected),
        "selected_checkpoint_sha256": selected_hash,
        "expected_selected_checkpoint_sha256": base.FINAL_MODELS[forest]["selected_checkpoint_sha256"],
    }
    report["status"] = "PASS" if (
        parent.is_file()
        and report["phase1_parent_sha256"] == cfg["parent_sha"]
        and selected.is_file()
        and selected_hash == report["expected_selected_checkpoint_sha256"]
    ) else "FAIL"
    del shots, records
    return report


def evaluate_test(engine, forest: str):
    final = base.FINAL_MODELS[forest]
    candidate_id, run_dir = configure(engine, forest)
    checkpoint = Path(final["selected_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if base.sha256_file(checkpoint) != final["selected_checkpoint_sha256"]:
        raise RuntimeError(f"Selected Phase-2 checkpoint hash mismatch: {checkpoint}")
    modules = base.prepare_modules(engine)
    cfg = engine.FORESTS[forest]
    _, shots, records = engine.build_data(forest, modules, include_test=True)
    model = engine.fresh_model(cfg, modules)
    try:
        state = engine.torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        state = engine.torch.load(checkpoint, map_location="cpu")
    model.prediction_head.load_state_dict(state["prediction_head"], strict=True)
    model.eval()
    _, nearest = modules["evaluate_full_patch_temporal_nearest"](
        model=model,
        records=records["test"],
        shots=shots,
        device=engine.DEVICE,
        split="test",
        drop_channels=(),
        min_height=cfg["eval_min"],
        max_height=cfg["eval_max"],
        progress_every=1,
    )
    if not nearest["aux_shot_uid"].is_unique:
        raise RuntimeError(f"{forest}: duplicate TEST shot identifiers")
    output_dir = (
        NATURAL_ROOT / "Results" / "Final_Article_Harmonized_GEDIAnchored_NaturalP1"
        / "Phase2" / cfg["label"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "test_unique_nearest.csv.gz"
    nearest.to_csv(prediction_path, index=False, compression="gzip")
    y = nearest["rh95"].to_numpy(float)
    metrics = {
        "phase1_same_test": engine.metrics(y, nearest["pred_off_reference"].to_numpy(float)),
        "phase2_harmonized": engine.metrics(y, nearest["pred_on_growthloss"].to_numpy(float)),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lineage = {
        "forest": cfg["label"],
        "candidate_id": candidate_id,
        "training_sequence_rule": "complete same-month T4 containing at least one valid GEDI shot",
        "fully_image_only_T4_used_for_training": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": engine.sha256(checkpoint),
        "phase1_parent": str(cfg["parent"]),
        "phase1_parent_sha256": engine.sha256(cfg["parent"]),
        "split": "TEST unique-nearest GEDI",
        "n": len(nearest),
    }
    (output_dir / "lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    del model, state
    gc.collect()
    if engine.torch.cuda.is_available():
        engine.torch.cuda.empty_cache()
    return prediction_path


def main():
    parser = argparse.ArgumentParser(description="Harmonized GEDI-anchored Phase-2 workflow")
    parser.add_argument("--forest", choices=(*base.FINAL_MODELS, "all"), default="all")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    engine = base.load_engine()
    install_harmonized_build_data(engine)
    forests = tuple(base.FINAL_MODELS) if args.forest == "all" else (args.forest,)
    for forest in forests:
        report = preflight(engine, forest)
        print(json.dumps(report, indent=2), flush=True)
        if report["status"] != "PASS":
            raise RuntimeError(f"Harmonized Phase-2 preflight failed for {forest}")
        if args.train:
            base.train(engine, forest)
        if args.evaluate_test:
            evaluate_test(engine, forest)


if __name__ == "__main__":
    main()
