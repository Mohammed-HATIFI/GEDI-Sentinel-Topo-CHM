from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


NATURAL_ROOT = Path(r"C:\Users\Dell\Desktop\Publication_Clarck\Natural_Sampling")
PROJECT_SOURCE = NATURAL_ROOT / "Source" / "Project"
ENGINE_PATH = PROJECT_SOURCE / "low_canopy_growthloss_ablation_runner.py"
ADAPTER_PATH = PROJECT_SOURCE / "aoi_masked_phase2_adapter.py"
FINAL_SELECTION_PATH = PROJECT_SOURCE / "final_selected_phase2_models.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


FINAL_SELECTION = json.loads(FINAL_SELECTION_PATH.read_text(encoding="utf-8"))
_SELECTED = FINAL_SELECTION["models"]

FINAL_MODELS = {
    "ifran": {
        "candidate_id": "CTRL_GL000_D5_K2_HD3",
        "spec": {"drop_m": 5.0, "K": 2, "lambda_growth": 0.0, "huber_delta": 3.0},
        "checkpoint_variant": "best_compromise.ckpt", "test_domain_m": [2.0, 45.0],
    },
    "maamoura": {
        "candidate_id": "CTRL_GL000_D2_K2_HD3",
        "spec": {"drop_m": 2.0, "K": 2, "lambda_growth": 0.0, "huber_delta": 3.0},
        "checkpoint_variant": "best_compromise.ckpt", "test_domain_m": [2.0, 20.0],
    },
    "agadir": {
        "candidate_id": "D3_K3_GL010_HD3",
        "spec": {"drop_m": 3.0, "K": 3, "lambda_growth": 0.10, "huber_delta": 3.0},
        "checkpoint_variant": "best_compromise.ckpt", "test_domain_m": [2.0, 20.0],
    },
}

for _forest, _entry in FINAL_MODELS.items():
    _entry["selected_checkpoint"] = _SELECTED[_forest]["checkpoint"]
    _entry["selected_checkpoint_sha256"] = _SELECTED[_forest]["checkpoint_sha256"]
    _entry["product"] = _SELECTED[_forest]["product"]


PHASE1_NATURAL_PARENTS = {
    "ifran": {
        "path": NATURAL_ROOT / "Models/Phase1/Ifran/IFRAN_PHASE1_NATURAL_SEED42_V1/checkpoints/best_slope.ckpt",
        "sha256": "072a8735973e3511564cf3ef7907f5b33105df08895c78917fb817de6198afb1",
    },
    "maamoura": {
        "path": NATURAL_ROOT / "Models/Phase1/Maamoura/MAAMOURA_PHASE1_NATURAL_SAMPLING_V2_SEED42/checkpoints/best_any.ckpt",
        "sha256": "d83a5493715451a61c997a66a25f361080a56e7ceade60eb886f2ae76f6bd7f4",
    },
    "agadir": {
        "path": NATURAL_ROOT / "Models/Phase1/Agadir/AGADIR_PHASE1_NATURAL_SEED42_V1/checkpoints/best_slope.ckpt",
        "sha256": "f72f06e33455852cbc94cd00ecb81e192f68e7f67b89de007fef4a610fa25b2e",
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_engine():
    return load_module("clarck_phase2_aoi_engine", ENGINE_PATH)


def configure(engine, forest: str):
    final = FINAL_MODELS[forest]
    candidate_id = final["candidate_id"]
    engine.AGADIR_CONFIRMATORY_MODE = False
    parent = PHASE1_NATURAL_PARENTS[forest]
    parent_path = Path(parent["path"])
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    parent_hash = engine.sha256(parent_path)
    if parent_hash != parent["sha256"]:
        raise RuntimeError(f"Natural Sampling Phase-1 hash mismatch: {parent_path}")
    engine.FORESTS[forest]["parent"] = parent_path
    engine.FORESTS[forest]["parent_sha"] = parent_hash
    engine.FORESTS[forest]["ablation_family"] = "Final_Article_Workflow_AOI_Masked_NaturalP1"
    if forest == "ifran":
        engine.IFRAN_CANDIDATES = {candidate_id: dict(final["spec"])}
    else:
        engine.LOW_CANOPY_CANDIDATES = {candidate_id: dict(final["spec"])}

    def isolated_roots(forest_key: str):
        cfg = engine.FORESTS[forest_key]
        model_root = NATURAL_ROOT / "Models" / "Phase2_Harmonized_GEDIAnchored_NaturalP1" / cfg["ecosystem"] / cfg["label"]
        result_root = NATURAL_ROOT / "Results" / "Phase2_Harmonized_GEDIAnchored_NaturalP1" / cfg["ecosystem"] / cfg["label"]
        return model_root, result_root

    engine.roots = isolated_roots
    return candidate_id, engine.run_dir(forest, candidate_id)


def prepare_modules(engine):
    adapter = load_module("clarck_phase2_aoi_adapter", ADAPTER_PATH)
    return adapter.prepare_aoi_masked_modules(engine)


def preflight(engine, forest: str):
    candidate_id, run_dir = configure(engine, forest)
    cfg = engine.FORESTS[forest]
    checkpoint = Path(cfg["parent"])
    selected = Path(FINAL_MODELS[forest]["selected_checkpoint"])
    selected_hash = sha256_file(selected) if selected.is_file() else None
    report = {
        "forest": forest, "candidate_id": candidate_id,
        "catalog": str(cfg["catalog"]), "catalog_exists": Path(cfg["catalog"]).is_dir(),
        "phase1_parent": str(checkpoint), "phase1_parent_exists": checkpoint.is_file(),
        "phase1_parent_sha256": engine.sha256(checkpoint) if checkpoint.is_file() else None,
        "expected_phase1_parent_sha256": cfg["parent_sha"],
        "run_dir": str(run_dir), "run_already_complete": (run_dir / "TRAIN_DONE.json").is_file(),
        "temporal_support": "AOI channel 12 > 0.5",
        "final_spec": FINAL_MODELS[forest],
        "selected_checkpoint": str(selected),
        "selected_checkpoint_exists": selected.is_file(),
        "selected_checkpoint_sha256": selected_hash,
        "expected_selected_checkpoint_sha256": FINAL_MODELS[forest]["selected_checkpoint_sha256"],
    }
    report["status"] = "PASS" if (
        report["catalog_exists"] and report["phase1_parent_exists"]
        and report["phase1_parent_sha256"] == report["expected_phase1_parent_sha256"]
        and report["selected_checkpoint_exists"]
        and report["selected_checkpoint_sha256"] == report["expected_selected_checkpoint_sha256"]
    ) else "FAIL"
    return report


def train(engine, forest: str):
    raise RuntimeError(
        "Final checkpoints are frozen in final_selected_phase2_models.json. "
        "Do not retrain from the publication workflow; run evaluation/inference only."
    )


def evaluate_test(engine, forest: str):
    final = FINAL_MODELS[forest]
    candidate_id, run_dir = configure(engine, forest)
    checkpoint = Path(final["selected_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_hash = sha256_file(checkpoint)
    if actual_hash != final["selected_checkpoint_sha256"]:
        raise RuntimeError(f"Selected Phase-2 checkpoint hash mismatch: {checkpoint}")
    modules = prepare_modules(engine)
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
        model=model, records=records["test"], shots=shots, device=engine.DEVICE,
        split="test", drop_channels=(), min_height=cfg["eval_min"], max_height=cfg["eval_max"],
        progress_every=1,
    )
    if not nearest["aux_shot_uid"].is_unique or len(nearest) != cfg["expected_test_n"]:
        raise RuntimeError((forest, len(nearest), cfg["expected_test_n"]))
    output_dir = NATURAL_ROOT / "Results" / "Final_Article_Harmonized_GEDIAnchored_NaturalP1" / "Phase2" / cfg["label"]
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "test_unique_nearest.csv.gz"
    nearest.to_csv(prediction_path, index=False, compression="gzip")
    y = nearest["rh95"].to_numpy(float)
    metrics = {
        "phase1_same_test": engine.metrics(y, nearest["pred_off_reference"].to_numpy(float)),
        "phase2_aoi_masked": engine.metrics(y, nearest["pred_on_growthloss"].to_numpy(float)),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lineage = {
        "forest": cfg["label"], "candidate_id": candidate_id,
        "temporal_support": "AOI channel 12 > 0.5",
        "checkpoint": str(checkpoint), "checkpoint_sha256": engine.sha256(checkpoint),
        "phase1_parent": str(cfg["parent"]), "phase1_parent_sha256": engine.sha256(cfg["parent"]),
        "split": "TEST unique-nearest GEDI", "n": len(nearest),
    }
    (output_dir / "lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    del model, state
    gc.collect()
    if engine.torch.cuda.is_available():
        engine.torch.cuda.empty_cache()
    return prediction_path


def main():
    parser = argparse.ArgumentParser(description="AOI-masked final Phase-2 workflow")
    parser.add_argument("--forest", choices=(*FINAL_MODELS, "all"), default="all")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    engine = load_engine()
    forests = tuple(FINAL_MODELS) if args.forest == "all" else (args.forest,)
    for forest in forests:
        report = preflight(engine, forest)
        print(json.dumps(report, indent=2), flush=True)
        if report["status"] != "PASS":
            raise RuntimeError(f"AOI Phase-2 preflight failed for {forest}")
        if args.train:
            train(engine, forest)
        if args.evaluate_test:
            evaluate_test(engine, forest)


if __name__ == "__main__":
    main()
