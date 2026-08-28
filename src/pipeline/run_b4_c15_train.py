from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from b4_c15_config import (
    B4_C15_CHANNEL_ORDER,
    PYTHON,
    SITES,
    TRAIN_SCRIPT,
    assert_c15_contract,
)


FINAL_CHECKPOINT_VARIANT = {
    "ifran": "best_slope.ckpt",
    "maamoura": "best_any.ckpt",
    "agadir": "best_slope.ckpt",
}

FINAL_MODEL_ROOT = {
    "ifran": Path(r"C:\Users\Dell\Desktop\Publication_Clarck\Models\Dense\Ifran"),
    "maamoura": Path(r"C:\Users\Dell\Desktop\Publication_Clarck\Models\Low_Sparsity\Maamoura"),
    "agadir": Path(r"C:\Users\Dell\Desktop\Publication_Clarck\Models\Sparse\Agadir"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote_phase1_checkpoint(site_key: str) -> Path:
    """Promote the VAL-selected checkpoint to the immutable Phase-2 parent path."""
    cfg = SITES[site_key]
    variant = FINAL_CHECKPOINT_VARIANT[site_key]
    candidates = (cfg.run_dir / "checkpoints" / variant, cfg.run_dir / variant)
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            f"The required VAL checkpoint {variant} was not produced by {cfg.run_dir}. "
            "Phase 2 is forbidden until Phase 1 checkpoint selection is complete."
        )
    destination_root = FINAL_MODEL_ROOT[site_key]
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / variant
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    source_hash = _sha256(source)
    destination_hash = _sha256(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"Checkpoint promotion hash mismatch: {source} -> {destination}")
    payload = {
        "site": site_key,
        "selection_split": "VAL only",
        "checkpoint_variant": variant,
        "source_run_dir": str(cfg.run_dir),
        "source_checkpoint": str(source),
        "canonical_checkpoint": str(destination),
        "sha256": destination_hash,
        "phase2_parent_ready": True,
    }
    (destination_root / "phase1_final_model.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[PHASE1 PROMOTED] {destination} sha256={destination_hash}", flush=True)
    return destination


def _best_checkpoint(run_dir: Path) -> Path | None:
    for folder in (run_dir / "checkpoints", run_dir):
        for name in ("best.ckpt", "best_any.ckpt", "best_compromise.ckpt", "best_slope.ckpt", "last.ckpt"):
            path = folder / name
            if path.exists():
                return path
    return None


def _load_contract(site_key: str) -> tuple[list[str], list[int]]:
    cfg = SITES[site_key]
    experiment = json.loads((cfg.catalog_root / "experiment.json").read_text(encoding="utf-8"))
    channels = list(experiment.get("channel_order") or experiment.get("schema", {}).get("channel_order") or [])
    if not cfg.native_c15:
        raise RuntimeError(f"{cfg.label}: le workflow final exige des NPY physiquement C15.")
    if cfg.historical_drop_channels:
        raise RuntimeError(f"{cfg.label}: aucun drop runtime n'est autorisé: {cfg.historical_drop_channels}")
    assert_c15_contract(channels)
    shard = next((cfg.catalog_root / "train" / "x").glob("*.npy"), None)
    if shard is None:
        raise FileNotFoundError(f"No native C15 train arrays: {cfg.catalog_root}")
    shape = np.load(shard, mmap_mode="r").shape
    if len(shape) != 3 or shape[-1] != 15:
        raise RuntimeError(f"Native C15 array mismatch: {shard} -> {shape}")
    return channels, []


def phase1_command(site_key: str) -> list[str]:
    cfg = SITES[site_key]
    _, drops = _load_contract(site_key)
    python = PYTHON if PYTHON.exists() else Path("python")
    profiles = {
        "ifran": {"delta": "3.0", "patience": "15", "audit_min": "2.5", "bins": "2,2.5,5,8,10,15,20,30,40,45", "height_bins": "0,2.5,5,8,10,15,20,30,40,45"},
        "maamoura": {"delta": "1.0", "patience": "15", "audit_min": "2.5", "bins": "2.5,5,8,10,15,20", "height_bins": "0,2.5,5,8,10,15,20"},
        "agadir": {"delta": "3.0", "patience": "15", "audit_min": "0.0", "bins": "2,5,8,10,15,20", "height_bins": "0,2,5,8,10,15,20"},
    }
    profile = profiles[site_key]
    bins = profile["bins"]
    height_bins = profile["height_bins"]
    command = [
        str(python), str(TRAIN_SCRIPT),
        "--experiment-root", str(cfg.catalog_root),
        "--runs-root", str(cfg.runs_root),
        "--run-name", cfg.run_name,
        "--max-aux-k", "2048",
        "--step1-max-steps", "999999",
        "--step2-max-steps", "0",
        "--step1-epochs", "9999",
        "--step1-patience-evals", profile["patience"],
        "--training-mode", "step",
        "--val-every-steps", "66",
        "--eval-primary-min-height", str(cfg.eval_min),
        "--eval-primary-max-height", str(cfg.eval_max),
        "--eval-audit-min-height", profile["audit_min"],
        "--eval-audit-max-height", str(cfg.train_max),
        "--no-eval-test-at-end",
        "--regression-loss", "huber", "--huber-beta", profile["delta"],
        "--alpha-reg", "1.0", "--beta-ord", "0.0", "--gamma-cls", "0.0", "--tv-weight", "0.0",
        "--model-type", "hytec", "--base-ch", "64", "--dropout", "0.15",
        "--batch-size", str(cfg.batch_size), "--num-workers", "0", "--seed", "42",
        "--optimizer", "adamw", "--lr", "1e-4", "--weight-decay", "5e-3", "--grad-clip", "1.0",
        "--lr-warmup-epochs", "3", "--plateau-patience", "8",
        "--plateau-factor", "0.5", "--lr-min", "1e-6",
        "--monitor", "article_compromise_score", "--monitor-mode", "min",
        "--phase1-monitor", "article_compromise_score", "--phase1-monitor-mode", "min",
        "--checkpoint-eligibility-mode", "on", "--checkpoint-gate-domain", "primary",
        "--checkpoint-min-slope", "0.65", "--checkpoint-min-std-ratio", "0.75",
        "--checkpoint-max-std-ratio", "1.25", "--checkpoint-max-abs-bias", "2.50",
        "--official-best-min-rel-gain", "0.0", "--official-best-min-abs-gain", "0.0",
        "--bins", bins,
        "--weights-mode", "none", "--patch-weight-mode", "equal",
        "--phase1-rebalance-strategy", "plain_mae",
        "--train-sampler-mode", "balanced_height",
        "--height-sampler-bins", height_bins,
        "--height-sampler-stat", "p90",
        "--phase1-lambda-slope-loss", "0.0",
        "--phase1-lambda-std-loss", "0.0",
        "--phase1-lambda-bias-loss", "0.0",
        "--phase1-lambda-anti-zero-loss", "0.0",
    ]
    if drops:
        raise RuntimeError("Invariant violé: aucun --drop-channels n'est autorisé dans le workflow C15 natif.")
    return command


def _command_payload(site_key: str, command: list[str]) -> dict[str, Any]:
    cfg = SITES[site_key]
    return {
        "site": cfg.label,
        "run_name": cfg.run_name,
        "run_dir": str(cfg.run_dir),
        "model_input": "B4 C15 no-PALSAR",
        "effective_channel_order": list(B4_C15_CHANNEL_ORDER),
        "stored_channels": 15,
        "runtime_drop_channels": [],
        "command": command,
        "windows_command_line": subprocess.list2cmdline(command),
    }


def run(site_key: str, execute: bool = False) -> Path | None:
    # === B4_INCOMPLETE_RUN_AUTO_RESUME_V2 ===
    cfg = SITES[site_key]
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Vendored B4 trainer missing: {TRAIN_SCRIPT}")
    command = phase1_command(site_key)
    cfg.reports_root.mkdir(parents=True, exist_ok=True)

    checkpoint = _best_checkpoint(cfg.run_dir)
    log_path = cfg.run_dir / "console_stream.log"
    clean_finish = False
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        clean_finish = "[TRAIN DONE]" in log_text or "[EARLY STOPPING]" in log_text
    if checkpoint is not None and clean_finish:
        payload_path = cfg.reports_root / "b4_c15_phase1_command.json"
        payload_path.write_text(json.dumps(_command_payload(site_key, command), indent=2), encoding="utf-8")
        print(f"[REUSE] Cleanly completed run checkpoint: {checkpoint}")
        return promote_phase1_checkpoint(site_key)

    resume_checkpoint: Path | None = None
    if cfg.run_dir.exists():
        for folder in (cfg.run_dir / "checkpoints", cfg.run_dir):
            for name in ("last.ckpt", "last_phase1.ckpt"):
                candidate = folder / name
                if candidate.is_file():
                    resume_checkpoint = candidate
                    break
            if resume_checkpoint is not None:
                break
        if resume_checkpoint is None:
            raise FileExistsError(
                "Incomplete run directory exists but no last checkpoint can be resumed safely: "
                f"{cfg.run_dir}"
            )
        command.extend(["--resume", str(resume_checkpoint)])
        print(f"[RESUME] Incomplete {site_key} run from: {resume_checkpoint}")

    payload_path = cfg.reports_root / "b4_c15_phase1_command.json"
    payload_path.write_text(json.dumps(_command_payload(site_key, command), indent=2), encoding="utf-8")
    print(subprocess.list2cmdline(command))
    if not execute:
        if resume_checkpoint is None:
            print("[DRY-RUN] Training was not launched. Add --execute or enable the notebook site flag.")
        else:
            print("[DRY-RUN] Resume was prepared but not launched. Enable the notebook site flag.")
        return None

    if resume_checkpoint is None:
        cfg.run_dir.mkdir(parents=True, exist_ok=False)
        log_mode = "w"
    else:
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        log_mode = "a"
    log_path = cfg.run_dir / "console_stream.log"
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open(log_mode, encoding="utf-8", buffering=1) as log:
        if resume_checkpoint is not None:
            marker = f"\n[LAUNCHER RESUME] checkpoint={resume_checkpoint}\n"
            print(marker, end="")
            log.write(marker)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    checkpoint = _best_checkpoint(cfg.run_dir)
    if checkpoint is None:
        raise RuntimeError(f"Training ended without a reusable checkpoint: {cfg.run_dir}")
    return promote_phase1_checkpoint(site_key)
    # === END B4_INCOMPLETE_RUN_AUTO_RESUME_V2 ===

def main() -> None:
    parser = argparse.ArgumentParser(description="Safe B4 C15 Phase-1 launcher.")
    parser.add_argument("--site", choices=sorted(SITES), required=True)
    parser.add_argument("--execute", action="store_true", help="Actually launch training; default is dry-run.")
    args = parser.parse_args()
    run(args.site, execute=args.execute)


if __name__ == "__main__":
    main()
