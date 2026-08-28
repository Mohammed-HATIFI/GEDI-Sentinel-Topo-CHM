from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path


PROJECT = Path(r"C:\Users\Dell\Desktop\Publication_Clarck")
ENGINE_PATH = PROJECT / "Source" / "Project" / "low_canopy_growthloss_ablation_runner.py"

FINAL_MODELS = {
    "ifran": {
        "candidate_id": "FINAL_D5_K2_GL020_HD3",
        "spec": {"drop_m": 5.0, "K": 2, "lambda_growth": 0.20, "huber_delta": 3.0},
        "checkpoint_variant": "best_slope.ckpt",
        "test_domain_m": [2.0, 45.0],
    },
    "maamoura": {
        "candidate_id": "FINAL_D2_K2_GL005_HD3",
        "spec": {"drop_m": 2.0, "K": 2, "lambda_growth": 0.05, "huber_delta": 3.0},
        "checkpoint_variant": "best_r2.ckpt",
        "test_domain_m": [2.0, 20.0],
    },
    "agadir": {
        "candidate_id": "FINAL_D3_K3_GL010_HD3",
        "spec": {"drop_m": 3.0, "K": 3, "lambda_growth": 0.10, "huber_delta": 3.0},
        "checkpoint_variant": "best_slope.ckpt",
        "test_domain_m": [2.0, 20.0],
    },
}


def load_engine():
    if not ENGINE_PATH.is_file():
        raise FileNotFoundError(ENGINE_PATH)
    spec = importlib.util.spec_from_file_location("clarck_phase2_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure(engine, forest: str) -> tuple[str, Path]:
    final = FINAL_MODELS[forest]
    candidate_id = final["candidate_id"]
    engine.AGADIR_CONFIRMATORY_MODE = False
    engine.FORESTS[forest]["ablation_family"] = "Final_Article_Workflow"
    parent = Path(engine.FORESTS[forest]["parent"])
    lineage_path = parent.parent / "phase1_final_model.json"
    if lineage_path.is_file():
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        if Path(lineage["canonical_checkpoint"]).resolve() != parent.resolve():
            raise RuntimeError(f"Phase-1 lineage points to another checkpoint: {lineage_path}")
        current_hash = engine.sha256(parent)
        if current_hash != lineage["sha256"]:
            raise RuntimeError(f"Phase-1 checkpoint hash mismatch: {parent}")
        engine.FORESTS[forest]["parent_sha"] = current_hash
    if forest == "ifran":
        engine.IFRAN_CANDIDATES = {candidate_id: dict(final["spec"])}
    else:
        engine.LOW_CANOPY_CANDIDATES = {candidate_id: dict(final["spec"])}
    run_dir = engine.run_dir(forest, candidate_id)
    return candidate_id, run_dir


def preflight(engine, forest: str) -> dict:
    candidate_id, run_dir = configure(engine, forest)
    cfg = engine.FORESTS[forest]
    lineage_path = Path(cfg["parent"]).parent / "phase1_final_model.json"
    report = {
        "forest": forest,
        "candidate_id": candidate_id,
        "catalog": str(cfg["catalog"]),
        "catalog_exists": cfg["catalog"].is_dir(),
        "phase1_parent": str(cfg["parent"]),
        "phase1_parent_exists": cfg["parent"].is_file(),
        "phase1_parent_sha256": engine.sha256(cfg["parent"]) if cfg["parent"].is_file() else None,
        "expected_phase1_parent_sha256": cfg["parent_sha"],
        "phase1_lineage": str(lineage_path),
        "phase1_lineage_exists": lineage_path.is_file(),
        "run_dir": str(run_dir),
        "final_spec": FINAL_MODELS[forest],
    }
    report["status"] = (
        "PASS" if report["catalog_exists"] and report["phase1_parent_exists"]
        and report["phase1_parent_sha256"] == report["expected_phase1_parent_sha256"] else "FAIL"
    )
    return report


def evaluate_test(engine, forest: str) -> Path:
    """Evaluate the frozen final checkpoint and publish one row per TEST GEDI shot."""
    final = FINAL_MODELS[forest]
    candidate_id, run_dir = configure(engine, forest)
    checkpoint = run_dir / "checkpoints" / final["checkpoint_variant"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = engine.sha256(checkpoint)
    cfg = engine.FORESTS[forest]
    output_dir = PROJECT / "Results" / "Final_Article" / "Phase2" / cfg["label"]
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "test_unique_nearest.csv.gz"
    lineage_path = output_dir / "lineage.json"
    metrics_path = output_dir / "test_metrics.json"
    reusable = False
    if prediction_path.is_file() and lineage_path.is_file():
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        reusable = lineage.get("checkpoint_sha256") == checkpoint_hash
    if reusable:
        # A cache is complete only when its tabular metrics can also be rebuilt.
        # This keeps an interrupted/report-only execution fully self-documenting.
        nearest = engine.pd.read_csv(prediction_path)
        if "aux_shot_uid" in nearest and not nearest["aux_shot_uid"].is_unique:
            raise RuntimeError(f"Duplicate TEST GEDI shots in cached predictions for {forest}")
        if len(nearest) != cfg["expected_test_n"]:
            raise RuntimeError((forest, len(nearest), cfg["expected_test_n"]))
        y_true = nearest["rh95"].to_numpy(float)
        phase1_metrics = engine.metrics(y_true, nearest["pred_off_reference"].to_numpy(float))
        phase2_metrics = engine.metrics(y_true, nearest["pred_on_growthloss"].to_numpy(float))
        metrics_path.write_text(
            json.dumps({"phase1_same_test": phase1_metrics, "phase2": phase2_metrics}, indent=2),
            encoding="utf-8",
        )
        print(f"[REUSE FINAL TEST + METRICS READY] {prediction_path}", flush=True)
        return prediction_path

    modules = engine.import_growth_modules()
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
        raise RuntimeError(f"Duplicate TEST GEDI shots for {forest}")
    if len(nearest) != cfg["expected_test_n"]:
        raise RuntimeError((forest, len(nearest), cfg["expected_test_n"]))
    nearest.to_csv(prediction_path, index=False, compression="gzip")
    y_true = nearest["rh95"].to_numpy(float)
    phase1_metrics = engine.metrics(y_true, nearest["pred_off_reference"].to_numpy(float))
    phase2_metrics = engine.metrics(y_true, nearest["pred_on_growthloss"].to_numpy(float))
    lineage = {
        "forest": cfg["label"],
        "candidate_id": candidate_id,
        "checkpoint_variant": final["checkpoint_variant"].removesuffix(".ckpt"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "phase1_parent": str(cfg["parent"]),
        "phase1_parent_sha256": engine.sha256(cfg["parent"]),
        "split": "TEST unique-nearest GEDI",
        "n": len(nearest),
        "test_used_for_further_tuning": False,
    }
    lineage_path.write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    metrics_path.write_text(
        json.dumps({"phase1_same_test": phase1_metrics, "phase2": phase2_metrics}, indent=2),
        encoding="utf-8",
    )
    print(f"[FINAL TEST READY] {prediction_path}", flush=True)
    del model, state
    gc.collect()
    if engine.torch.cuda.is_available():
        engine.torch.cuda.empty_cache()
    return prediction_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Final reproducible Phase-2 workflow for the three forests.")
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
            raise RuntimeError(f"Phase-2 preflight failed for {forest}")
        if args.train:
            candidate_id, run_dir = configure(engine, forest)
            completed_config = run_dir / "config.json"
            if (run_dir / "TRAIN_DONE.json").is_file() and completed_config.is_file():
                saved = json.loads(completed_config.read_text(encoding="utf-8"))
                saved_parent_hash = saved.get("phase1_parent", {}).get("source_sha256")
                current_parent_hash = engine.sha256(engine.FORESTS[forest]["parent"])
                if saved_parent_hash != current_parent_hash:
                    raise RuntimeError(
                        f"Refusing to reuse {run_dir}: its Phase-1 parent hash differs from "
                        "the freshly promoted parent. Move the old Phase-2 run aside and retrain."
                    )
            modules = engine.import_growth_modules()
            engine.train_candidate(forest, candidate_id, modules)
            checkpoint = run_dir / "checkpoints" / FINAL_MODELS[forest]["checkpoint_variant"]
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Expected final checkpoint was not produced: {checkpoint}. "
                    "Inspect the saved VAL checkpoint variants before any TEST evaluation."
                )
            print(f"[FINAL CHECKPOINT] {checkpoint}", flush=True)
        if args.evaluate_test:
            evaluate_test(engine, forest)


if __name__ == "__main__":
    main()
