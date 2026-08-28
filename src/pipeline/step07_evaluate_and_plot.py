from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from b4_c15_config import (
    B4_C15_CHANNEL_ORDER,
    PYTHON,
    SITES,
    STEP07_SCRIPT,
    assert_c15_contract,
)


FINAL_CHECKPOINT_VARIANT = {
    "ifran": "best_slope.ckpt",
    "maamoura": "best_any.ckpt",
    "agadir": "best_slope.ckpt",
}


def _effective_channels(site_key: str) -> tuple[list[str], list[int]]:
    cfg = SITES[site_key]
    experiment_path = cfg.catalog_root / "experiment.json"
    if not experiment_path.exists():
        raise FileNotFoundError(experiment_path)
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    raw = list(experiment.get("channel_order") or experiment.get("schema", {}).get("channel_order") or [])
    drops = list(cfg.historical_drop_channels)
    effective = [name for index, name in enumerate(raw) if index not in drops]
    assert_c15_contract(effective)
    if cfg.native_c15 and drops:
        raise RuntimeError("Native C15 evaluation must not drop channels.")
    return effective, drops


def evaluation_command(site_key: str, split: str = "test") -> list[str]:
    cfg = SITES[site_key]
    _, drops = _effective_channels(site_key)
    python = PYTHON if PYTHON.exists() else Path("python")
    bins = "2.5,5,8,10,15,20" if cfg.eval_max <= 20 else "2.5,5,8,10,15,20,30,40"
    command = [
        str(python), str(STEP07_SCRIPT),
        "--experiment-root", str(cfg.catalog_root),
        "--runs-root", str(cfg.runs_root),
        "--run-name", cfg.run_name,
        "--split", split,
        "--batch-size", str(cfg.batch_size),
        "--max-aux-k", "2048",
        "--bins", bins,
        "--min-height", str(cfg.eval_min),
        "--max-height", str(cfg.eval_max),
        "--checkpoint-name", FINAL_CHECKPOINT_VARIANT[site_key],
        "--no-clip-predictions",
        "--plots",
    ]
    if drops:
        command.extend(["--drop-channels", ",".join(map(str, drops))])
    return command


def _source_output(site_key: str, split: str) -> tuple[Path, Path, Path]:
    cfg = SITES[site_key]
    root = cfg.run_dir / "step07_original_coords_minmax"
    # Older B4 evaluations were grouped under the evaluated checkpoint stem
    # (for example step07_original_coords_minmax/best), while the frozen
    # compatible evaluator writes directly below step07_original_coords_minmax.
    candidates = [root, root / "best", root / "best_phase1", root / "best_any", root / "best_compromise", root / "best_slope", root / "best_r2"]
    for candidate in candidates:
        metrics = candidate / f"{split}_metrics_original_coords.json"
        predictions = candidate / f"{split}_predictions_original_coords.csv.gz"
        figures = candidate / f"figures_{split}"
        if metrics.exists() and predictions.exists() and figures.exists():
            return metrics, predictions, figures
    return (
        root / f"{split}_metrics_original_coords.json",
        root / f"{split}_predictions_original_coords.csv.gz",
        root / f"figures_{split}",
    )


def publish_existing(site_key: str, split: str = "test") -> list[Path]:
    cfg = SITES[site_key]
    metrics, predictions, figures = _source_output(site_key, split)
    if not metrics.exists() or not predictions.exists() or not figures.exists():
        return []
    out = cfg.figures_root / "step07" / split
    out.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for source in sorted(figures.glob("*.png")):
        destination = out / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        generated.append(destination)
    report_dir = cfg.reports_root / "step07" / split
    report_dir.mkdir(parents=True, exist_ok=True)
    for source in [metrics, predictions, metrics.parent / f"{split}_metrics_by_height.csv", metrics.parent / f"{split}_prediction_pathology_by_height.csv"]:
        if source.exists():
            destination = report_dir / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
    canonical_dir = (
        Path(r"C:\Users\Dell\Desktop\Publication_Clarck")
        / "Results" / "Final_Article" / "Phase1" / cfg.label
    )
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_metrics = canonical_dir / f"{split}_metrics_original_coords.json"
    canonical_predictions = canonical_dir / f"{split}_predictions_original_coords.csv.gz"
    shutil.copy2(metrics, canonical_metrics)
    shutil.copy2(predictions, canonical_predictions)
    payload = {
        "site": cfg.label,
        "split": split,
        "model_input": "B4 C15 no-PALSAR",
        "channel_order": list(B4_C15_CHANNEL_ORDER),
        "source_metrics": str(metrics),
        "source_predictions": str(predictions),
        "canonical_metrics": str(canonical_metrics),
        "canonical_predictions": str(canonical_predictions),
        "checkpoint_variant": FINAL_CHECKPOINT_VARIANT[site_key],
        "published_figures": [str(path) for path in generated],
    }
    (report_dir / "publication_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return generated


def run(site_key: str, split: str = "test", execute_if_missing: bool = False) -> list[Path]:
    if not STEP07_SCRIPT.exists():
        raise FileNotFoundError(f"Vendored STEP07 evaluator missing: {STEP07_SCRIPT}")
    existing = publish_existing(site_key, split)
    if existing:
        print(f"[REUSE] {len(existing)} STEP07 figures already available for {site_key}/{split}.")
        return existing
    command = evaluation_command(site_key, split)
    cfg = SITES[site_key]
    cfg.reports_root.mkdir(parents=True, exist_ok=True)
    command_path = cfg.reports_root / f"step07_{split}_command.json"
    command_path.write_text(
        json.dumps(
            {
                "site": cfg.label,
                "split": split,
                "run_dir": str(cfg.run_dir),
                "command": command,
                "windows_command_line": subprocess.list2cmdline(command),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(subprocess.list2cmdline(command))
    if not execute_if_missing:
        print("[SKIP] STEP07 artefacts are absent and evaluation execution is disabled.")
        return []
    subprocess.run(command, check=True)
    generated = publish_existing(site_key, split)
    if not generated:
        raise RuntimeError(f"STEP07 finished but no figures were found for {site_key}/{split}.")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatible STEP07 evaluation and publication copier.")
    parser.add_argument("--site", choices=sorted(SITES), required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--execute-if-missing", action="store_true")
    args = parser.parse_args()
    run(args.site, args.split, execute_if_missing=args.execute_if_missing)


if __name__ == "__main__":
    main()
