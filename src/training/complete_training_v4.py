from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .ablation_training import (
    load_official_growth_loss,
    safe_copy_reference,
    set_seed,
    sha256_file,
    sparse_prediction_loss,
)


def _finite(value, default=float("nan")) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _metrics_from_arrays(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    min_height: float = 2.5,
    max_height: float = 40.0,
) -> Dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = (
        np.isfinite(truth)
        & np.isfinite(prediction)
        & (truth >= float(min_height))
        & (truth <= float(max_height))
    )
    y = truth[valid]
    p = prediction[valid]
    if not len(y):
        return {k: float("nan") for k in (
            "mae", "rmse", "r2", "bias", "slope", "std_ratio", "pred_std",
            "true_std", "pred_max", "true_max", "mae_20_40", "mae_30_40"
        )} | {"n": 0.0}
    residual = p - y
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    true_std = float(np.std(y))
    pred_std = float(np.std(p))
    std_ratio = pred_std / true_std if true_std > 0 else float("nan")
    if len(y) >= 2 and float(np.var(y)) > 0:
        slope = float(np.cov(y, p, ddof=0)[0, 1] / np.var(y))
    else:
        slope = float("nan")

    def slice_mae(low: float, high: float) -> float:
        mask = (y >= low) & (y <= high)
        return float(np.mean(np.abs(residual[mask]))) if bool(mask.any()) else float("nan")

    return {
        "n": float(len(y)),
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "bias": float(np.mean(residual)),
        "slope": slope,
        "std_ratio": std_ratio,
        "pred_std": pred_std,
        "true_std": true_std,
        "pred_max": float(np.max(p)),
        "true_max": float(np.max(y)),
        "mae_20_40": slice_mae(20.0, 40.0),
        "mae_30_40": slice_mae(30.0, 40.0),
    }


def article_compromise_score(metrics: Dict[str, float]) -> float:
    """Same article-oriented compromise formula used by the B4 trainer."""
    mae = _finite(metrics.get("mae"), math.inf)
    rmse = _finite(metrics.get("rmse"), math.inf)
    r2 = _finite(metrics.get("r2"), -math.inf)
    slope = _finite(metrics.get("slope"), -math.inf)
    std_ratio = _finite(metrics.get("std_ratio"), math.inf)
    bias = _finite(metrics.get("bias"), math.inf)
    if not all(math.isfinite(v) for v in (mae, rmse, r2, slope, std_ratio, bias)):
        return math.inf
    return float(
        mae
        + 0.25 * rmse
        + 2.00 * max(0.0, 0.90 - slope)
        + 0.75 * abs(1.0 - std_ratio)
        + 0.20 * min(abs(bias), 2.0)
        + 0.50 * max(0.0, 0.70 - r2)
    )


def checkpoint_eligibility(
    metrics: Dict[str, float],
    *,
    min_slope: float = 0.65,
    min_std_ratio: float = 0.75,
    max_std_ratio: float = 1.25,
    max_abs_bias: float = 2.50,
) -> tuple[bool, list[str]]:
    reasons = []
    slope = _finite(metrics.get("slope"))
    std_ratio = _finite(metrics.get("std_ratio"))
    bias = _finite(metrics.get("bias"))
    if not math.isfinite(slope) or slope < float(min_slope):
        reasons.append(f"slope={slope:.3f} < {float(min_slope):.3f}")
    if not math.isfinite(std_ratio) or std_ratio < float(min_std_ratio):
        reasons.append(f"std_ratio={std_ratio:.3f} < {float(min_std_ratio):.3f}")
    if not math.isfinite(std_ratio) or std_ratio > float(max_std_ratio):
        reasons.append(f"std_ratio={std_ratio:.3f} > {float(max_std_ratio):.3f}")
    if not math.isfinite(bias) or abs(bias) > float(max_abs_bias):
        reasons.append(f"|bias|={abs(bias):.3f} > {float(max_abs_bias):.3f}")
    return not reasons, reasons


def _progress_bar(position: int, total: int, width: int = 26) -> str:
    total = max(1, int(total))
    position = min(max(0, int(position)), total)
    filled = int(round(width * position / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _state_payload(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    cycle: int,
    metrics: Dict[str, float],
    train_metrics: Dict[str, float],
    config: Dict,
    checkpoint_kind: str,
) -> Dict:
    return {
        "prediction_head": model.prediction_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "step": int(step),
        "cycle": int(cycle),
        "metrics": {k: _finite(v) for k, v in metrics.items()},
        "train_metrics": {k: _finite(v) for k, v in train_metrics.items()},
        "article_compromise_score": article_compromise_score(metrics),
        "checkpoint_kind": str(checkpoint_kind),
        "config": config,
    }


def _atomic_torch_save(payload: Dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _save_checkpoint(
    *,
    path: Path,
    kind: str,
    model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    cycle: int,
    metrics: Dict[str, float],
    train_metrics: Dict[str, float],
    config: Dict,
) -> None:
    payload = _state_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=step,
        cycle=cycle,
        metrics=metrics,
        train_metrics=train_metrics,
        config=config,
        checkpoint_kind=kind,
    )
    _atomic_torch_save(payload, path)
    meta = {
        "checkpoint": str(path.resolve()),
        "kind": kind,
        "step": int(step),
        "cycle": int(cycle),
        "metrics": payload["metrics"],
        "article_compromise_score": payload["article_compromise_score"],
    }
    path.with_name(path.stem + "_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


@torch.no_grad()
def validate_complete(
    *,
    model,
    loader,
    disturbance_loss,
    device: str,
    supervised_loss_name: str,
    huber_delta: float,
    height_weight_mode: str,
    lambda_growth: float,
    min_height: float,
    max_height: float,
) -> Dict[str, float]:
    model.eval()
    truths = []
    predictions = []
    huber_values = []
    growth_values = []
    total_values = []
    amp_enabled = str(device).startswith("cuda")
    for x, target, _ in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            out = model(x)
            direct = sparse_prediction_loss(
                out,
                target,
                loss_name=supervised_loss_name,
                huber_delta=huber_delta,
                height_weight_mode=height_weight_mode,
            )
            growth = disturbance_loss(out, target)
            total = direct + float(lambda_growth) * growth
        valid = torch.isfinite(target) & (target != 0)
        if bool(valid.any()):
            truths.append(target[valid].detach().float().cpu().numpy())
            predictions.append(out[:, 1][valid].detach().float().cpu().numpy())
        huber_values.append(float(direct.detach().cpu()))
        growth_values.append(float(growth.detach().cpu()))
        total_values.append(float(total.detach().cpu()))
    y = np.concatenate(truths) if truths else np.array([], dtype=np.float32)
    p = np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)
    metrics = _metrics_from_arrays(y, p, min_height=min_height, max_height=max_height)
    metrics.update(
        {
            "direct_loss": float(np.mean(huber_values)) if huber_values else float("nan"),
            "growth_loss": float(np.mean(growth_values)) if growth_values else float("nan"),
            "total_loss": float(np.mean(total_values)) if total_values else float("nan"),
        }
    )
    metrics["article_compromise_score"] = article_compromise_score(metrics)
    return metrics


def train_complete_temporal_stage(
    *,
    model,
    train_dataset,
    val_dataset,
    official_repo: Path,
    phase1_checkpoint: Path,
    run_dir: Path,
    device: str,
    seed: int = 42,
    batch_size: int = 1,
    max_steps: int = 999_999,
    val_every_steps: int = 66,
    patience_evals: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 5e-3,
    lambda_growth: float = 0.1,
    supervised_loss_name: str = "huber",
    huber_delta: float = 3.0,
    height_weight_mode: str = "none",
    slope_min: float = 0.0,
    slope_max: float = 2.0,
    disturbance_indicator: float = -1.0,
    full_disturbance_window: bool = True,
    min_height: float = 2.5,
    max_height: float = 40.0,
    checkpoint_min_slope: float = 0.65,
    checkpoint_min_std_ratio: float = 0.75,
    checkpoint_max_std_ratio: float = 1.25,
    checkpoint_max_abs_bias: float = 2.50,
    warmup_cycles: int = 3,
    plateau_patience: int = 8,
    plateau_factor: float = 0.5,
    lr_min: float = 1e-6,
    grad_clip: float = 1.0,
    allow_resume: bool = True,
    reuse_completed: bool = True,
):
    """Complete isolated Phase-2 run with B4-style checkpoint semantics.

    Phase 1 is the already completed scratch B4 Huber run.  The copied Phase-1
    checkpoint is immutable provenance.  This function trains the full temporal
    Conv3D prediction head with direct sparse Huber + official ECHOSAT GrowthLoss.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    table_dir = run_dir / "tables"
    done_path = run_dir / "TRAIN_DONE.json"
    best_path = ckpt_dir / "best.ckpt"
    best_any_path = ckpt_dir / "best_any.ckpt"
    best_compromise_path = ckpt_dir / "best_compromise.ckpt"
    best_slope_path = ckpt_dir / "best_slope.ckpt"
    last_path = ckpt_dir / "last.ckpt"

    if done_path.exists() and reuse_completed:
        print(f"♻️ [REUSE COMPLETED RUN] {run_dir}")
        chosen = best_path if best_path.exists() else best_compromise_path
        if not chosen.exists():
            raise FileNotFoundError(f"Completed marker exists but no reusable checkpoint: {run_dir}")
        state = torch.load(chosen, map_location="cpu")
        model.prediction_head.load_state_dict(state["prediction_head"], strict=True)
        history_path = table_dir / "training_history.csv"
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        return history, chosen, None

    if run_dir.exists() and any(run_dir.iterdir()) and not (allow_resume and last_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite non-resumable run directory: {run_dir}. Change RUN_NAME."
        )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    phase1_copy = ckpt_dir / "best_phase1.ckpt"
    phase1_provenance = safe_copy_reference(Path(phase1_checkpoint), phase1_copy)
    (ckpt_dir / "best_phase1_meta.json").write_text(
        json.dumps({"kind": "immutable B4 scratch Huber Phase-1 parent", **phase1_provenance}, indent=2),
        encoding="utf-8",
    )

    GrowthLoss, source_path = load_official_growth_loss(official_repo)
    disturbance_loss = GrowthLoss(
        disturbance_indicator=disturbance_indicator,
        slope_min=slope_min,
        slope_max=slope_max,
        full_disturbance_window=full_disturbance_window,
        use_l2=True,
        max_intercept_after_disturbance=100,
        disturbance_factor=1,
        no_disturbance_factor=1,
        slope_no_disturbance=-0.0,
    ).to(device)

    config = {
        "protocol": "B4 scratch Huber Phase-1 -> complete ECHOSAT temporal Phase-2",
        "phase1_parent": phase1_provenance,
        "reference_frozen": True,
        "train_scope": "full official-style Conv3D temporal prediction head",
        "sequence_policy": "same-month leaf-on sequences; no annual median; no month mixing",
        "supervised_loss": supervised_loss_name,
        "huber_delta": float(huber_delta),
        "lambda_growth": float(lambda_growth),
        "official_growth_loss_source": str(Path(source_path).resolve()),
        "official_growth_loss_sha256": sha256_file(source_path),
        "optimizer": "AdamW",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "warmup_cycles": int(warmup_cycles),
        "plateau_patience": int(plateau_patience),
        "plateau_factor": float(plateau_factor),
        "lr_min": float(lr_min),
        "grad_clip": float(grad_clip),
        "max_steps": int(max_steps),
        "val_every_steps": int(val_every_steps),
        "patience_evals": int(patience_evals),
        "checkpoint_monitor": "article_compromise_score",
        "checkpoint_gate": {
            "min_slope": float(checkpoint_min_slope),
            "min_std_ratio": float(checkpoint_min_std_ratio),
            "max_std_ratio": float(checkpoint_max_std_ratio),
            "max_abs_bias": float(checkpoint_max_abs_bias),
        },
        "evaluation_domain_m": [float(min_height), float(max_height)],
        "seed": int(seed),
        "trainable_parameters": int(model.trainable_parameter_count()),
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError("Resume config differs from the existing isolated run config.")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    model.to(device)
    initial_lr = (
        float(learning_rate) / max(1, int(warmup_cycles))
        if int(warmup_cycles) > 0
        else float(learning_rate)
    )
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=initial_lr, weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(plateau_factor),
        patience=int(plateau_patience),
        min_lr=float(lr_min),
    )
    amp_enabled = str(device).startswith("cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=amp_enabled,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=amp_enabled,
    )

    history = []
    history_path = table_dir / "training_history.csv"
    step = 0
    cycle = 0
    epoch = 0
    no_improve = 0
    best_any_score = math.inf
    best_official_score = math.inf
    best_slope_key = (math.inf, math.inf)
    start_time = time.time()

    if allow_resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu")
        model.prediction_head.load_state_dict(state["prediction_head"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        step = int(state.get("step", 0))
        cycle = int(state.get("cycle", 0))
        history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
        for row in history:
            score = _finite(row.get("val_article_compromise_score"), math.inf)
            best_any_score = min(best_any_score, score)
            eligible = bool(row.get("checkpoint_eligible", False))
            if eligible:
                best_official_score = min(best_official_score, score)
            slope = _finite(row.get("val_slope"))
            mae = _finite(row.get("val_mae"), math.inf)
            if math.isfinite(slope):
                best_slope_key = min(best_slope_key, (abs(1.0 - slope), mae))
        print(f"🔁 [RESUME] step={step:05d} | cycle={cycle:04d} | {last_path}")

    print("\n" + "=" * 118)
    print("🚀 COMPLETE TEMPORAL TRAINING — B4 SCRATCH HUBER -> HUBER + ECHOSAT GROWTHLOSS")
    print("=" * 118)
    print(f"📁 [RUN] {run_dir}")
    print(f"🧬 [PHASE-1 PARENT] {phase1_checkpoint}")
    print(f"🔒 [REFERENCE] frozen B4 | trainable temporal parameters={model.trainable_parameter_count():,}")
    print(f"🗓️ [SEQUENCES] same-month leaf-on only | train={len(train_dataset.records):,} | val={len(val_dataset.records):,}")
    print(f"🧪 [LOSS] Huber(delta={huber_delta:g}) + {lambda_growth:g} x official GrowthLoss")
    print(f"⚙️ [OPTIMIZER] AdamW lr={learning_rate:.2e} wd={weight_decay:.2e} | grad_clip={grad_clip:g}")
    print(f"🛡️ [GATE] slope>={checkpoint_min_slope:.2f} | std_ratio=[{checkpoint_min_std_ratio:.2f},{checkpoint_max_std_ratio:.2f}] | |bias|<={checkpoint_max_abs_bias:.2f} m")
    print(f"⏱️ [SCHEDULE] val_every={val_every_steps} | patience={patience_evals} | technical max={max_steps}")
    print("=" * 118, flush=True)

    window_truth = []
    window_pred = []
    window_direct = []
    window_growth = []
    window_total = []
    stop = False
    while step < int(max_steps) and not stop:
        train_dataset.set_epoch(epoch)
        for x, target, _ in train_loader:
            if step >= int(max_steps) or stop:
                break
            model.train()
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                out = model(x)
                direct = sparse_prediction_loss(
                    out,
                    target,
                    loss_name=supervised_loss_name,
                    huber_delta=huber_delta,
                    height_weight_mode=height_weight_mode,
                )
                growth = disturbance_loss(out, target)
                total = direct + float(lambda_growth) * growth
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            step += 1

            valid = torch.isfinite(target) & (target != 0)
            if bool(valid.any()):
                window_truth.append(target[valid].detach().float().cpu().numpy())
                window_pred.append(out[:, 1][valid].detach().float().cpu().numpy())
            window_direct.append(float(direct.detach().cpu()))
            window_growth.append(float(growth.detach().cpu()))
            window_total.append(float(total.detach().cpu()))

            position = ((step - 1) % int(val_every_steps)) + 1
            if step == 1 or position % 10 == 0 or position == int(val_every_steps):
                y_now = np.concatenate(window_truth) if window_truth else np.array([])
                p_now = np.concatenate(window_pred) if window_pred else np.array([])
                tm = _metrics_from_arrays(y_now, p_now, min_height=min_height, max_height=max_height)
                print(
                    f"🚂 [P2|C{cycle + 1:03d}] {_progress_bar(position, val_every_steps)} "
                    f"{position:02d}/{int(val_every_steps):02d} | step={step:05d} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} | total={np.mean(window_total):.4f} | "
                    f"huber={np.mean(window_direct):.4f} | growth={np.mean(window_growth):.4f} | "
                    f"mae={_finite(tm.get('mae')):.3f} | bias={_finite(tm.get('bias')):.3f} | "
                    f"slope={_finite(tm.get('slope')):.3f} | pts={int(tm.get('n', 0)):,}",
                    flush=True,
                )

            if step % int(val_every_steps) != 0 and step != int(max_steps):
                continue

            cycle += 1
            y_window = np.concatenate(window_truth) if window_truth else np.array([])
            p_window = np.concatenate(window_pred) if window_pred else np.array([])
            train_metrics = _metrics_from_arrays(
                y_window, p_window, min_height=min_height, max_height=max_height
            )
            train_metrics.update(
                {
                    "direct_loss": float(np.mean(window_direct)),
                    "growth_loss": float(np.mean(window_growth)),
                    "total_loss": float(np.mean(window_total)),
                }
            )
            val_metrics = validate_complete(
                model=model,
                loader=val_loader,
                disturbance_loss=disturbance_loss,
                device=device,
                supervised_loss_name=supervised_loss_name,
                huber_delta=huber_delta,
                height_weight_mode=height_weight_mode,
                lambda_growth=lambda_growth,
                min_height=min_height,
                max_height=max_height,
            )
            score = float(val_metrics["article_compromise_score"])
            eligible, gate_reasons = checkpoint_eligibility(
                val_metrics,
                min_slope=checkpoint_min_slope,
                min_std_ratio=checkpoint_min_std_ratio,
                max_std_ratio=checkpoint_max_std_ratio,
                max_abs_bias=checkpoint_max_abs_bias,
            )

            print("\n" + "·" * 118)
            print(f"🔎 VALIDATION @ STEP {step:05d} | CYCLE {cycle:04d}")
            print("·" * 118)
            print(
                f"[TRAIN-WINDOW] mae={train_metrics['mae']:.4f} | rmse={train_metrics['rmse']:.4f} | "
                f"r2={train_metrics['r2']:.4f} | bias={train_metrics['bias']:.4f} | "
                f"slope={train_metrics['slope']:.3f} | stdr={train_metrics['std_ratio']:.3f} | "
                f"huber={train_metrics['direct_loss']:.4f} | growth={train_metrics['growth_loss']:.4f} | "
                f"n={int(train_metrics['n']):,}"
            )
            print(
                f"[VAL-CURRENT ] mae={val_metrics['mae']:.4f} | rmse={val_metrics['rmse']:.4f} | "
                f"r2={val_metrics['r2']:.4f} | bias={val_metrics['bias']:.4f} | "
                f"slope={val_metrics['slope']:.3f} | stdr={val_metrics['std_ratio']:.3f} | "
                f"mae20-40={val_metrics['mae_20_40']:.4f} | mae30-40={val_metrics['mae_30_40']:.4f} | "
                f"n={int(val_metrics['n']):,}"
            )
            print(
                f"[VAL-LOSSES  ] total={val_metrics['total_loss']:.5f} | "
                f"huber={val_metrics['direct_loss']:.5f} | growth={val_metrics['growth_loss']:.5f}"
            )
            print(f"[MONITOR     ] article_compromise_score={score:.6f} | lower=better")
            if eligible:
                print("✅ [CKPT-GATE] eligible=YES | primary-domain calibration gate passed")
            else:
                print("🛑 [CKPT-GATE] eligible=NO  | " + "; ".join(gate_reasons))

            row = {
                "step": int(step),
                "cycle": int(cycle),
                "epoch": int(epoch),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "checkpoint_eligible": bool(eligible),
                "gate_reasons": "; ".join(gate_reasons),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)
            pd.DataFrame(history).to_csv(history_path, index=False)

            save_kwargs = dict(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step,
                cycle=cycle,
                metrics=val_metrics,
                train_metrics=train_metrics,
                config=config,
            )
            _save_checkpoint(path=last_path, kind="last", **save_kwargs)

            improved_any = math.isfinite(score) and score < best_any_score
            if improved_any:
                previous = best_any_score
                best_any_score = score
                no_improve = 0
                _save_checkpoint(path=best_any_path, kind="best_any_raw_monitor", **save_kwargs)
                _save_checkpoint(path=best_compromise_path, kind="best_compromise", **save_kwargs)
                old_txt = "none" if not math.isfinite(previous) else f"{previous:.6f}"
                print(f"🌟 [MONITOR IMPROVED] {old_txt} -> {score:.6f} | patience reset to 0")
                print(f"⚖️ [BEST COMPROMISE CHECKPOINT SAVED] {best_compromise_path}")
                print(f"🧾 [BEST ANY CHECKPOINT SAVED] {best_any_path}")
            else:
                no_improve += 1
                print(f"⏳ [NO MONITOR IMPROVEMENT] patience={no_improve}/{patience_evals}")

            slope_key = (abs(1.0 - _finite(val_metrics.get("slope"))), _finite(val_metrics.get("mae"), math.inf))
            if all(math.isfinite(v) for v in slope_key) and slope_key < best_slope_key:
                best_slope_key = slope_key
                _save_checkpoint(path=best_slope_path, kind="best_slope", **save_kwargs)
                print(
                    f"📐 [BEST SLOPE CHECKPOINT SAVED] slope={val_metrics['slope']:.4f} | "
                    f"|1-slope|={slope_key[0]:.4f} | mae={val_metrics['mae']:.4f} | {best_slope_path}"
                )

            if eligible and math.isfinite(score) and score < best_official_score:
                previous = best_official_score
                best_official_score = score
                _save_checkpoint(path=best_path, kind="official_best_eligible_compromise", **save_kwargs)
                old_txt = "none" if not math.isfinite(previous) else f"{previous:.6f}"
                print(
                    f"💾 [BEST CHECKPOINT SAVED] official eligible score {old_txt} -> {score:.6f} | {best_path}"
                )
            elif best_path.exists():
                kept = torch.load(best_path, map_location="cpu")
                kept_metrics = kept.get("metrics", {})
                print(
                    f"👁️ [BEST CHECKPOINT KEPT] step={int(kept.get('step', 0)):05d} | "
                    f"mae={_finite(kept_metrics.get('mae')):.4f} | "
                    f"slope={_finite(kept_metrics.get('slope')):.3f} | "
                    f"stdr={_finite(kept_metrics.get('std_ratio')):.3f}"
                )
            else:
                print("👁️ [BEST CHECKPOINT KEPT] no eligible official checkpoint yet")

            if cycle <= int(warmup_cycles):
                # B4 convention: C001=1/3 LR, C002=2/3 LR, C003=full LR.
                warm_lr = float(learning_rate) * min(
                    1.0, (cycle + 1) / max(1, int(warmup_cycles))
                )
                for group in optimizer.param_groups:
                    group["lr"] = warm_lr
                print(f"🔥 [LR WARMUP] cycle={cycle}/{warmup_cycles} | next_lr={warm_lr:.2e}")
            else:
                old_lr = float(optimizer.param_groups[0]["lr"])
                scheduler.step(score)
                new_lr = float(optimizer.param_groups[0]["lr"])
                if new_lr < old_lr:
                    print(f"📉 [LR REDUCED] {old_lr:.2e} -> {new_lr:.2e}")

            print(f"🧭 [PATIENCE] no_improve={no_improve}/{patience_evals}")
            print("·" * 118 + "\n", flush=True)
            window_truth.clear()
            window_pred.clear()
            window_direct.clear()
            window_growth.clear()
            window_total.clear()
            model.train()

            if no_improve >= int(patience_evals):
                print(f"🛑 [EARLY STOP] patience reached at step={step:05d} | cycle={cycle:04d}")
                stop = True
                break
        epoch += 1

    if not best_compromise_path.exists():
        raise RuntimeError("Training ended without a valid best_compromise.ckpt")
    if not best_path.exists():
        shutil.copy2(best_compromise_path, best_path)
        fallback_meta = {
            "kind": "fallback_best_from_best_compromise",
            "warning": "No checkpoint passed the B4 calibration gate; do not call this an eligible official best.",
            "source": str(best_compromise_path.resolve()),
        }
        (ckpt_dir / "best_meta.json").write_text(json.dumps(fallback_meta, indent=2), encoding="utf-8")
        print("⚠️ [BEST FALLBACK] no candidate passed the gate; best.ckpt copies best_compromise.ckpt for audit only")

    elapsed = float(time.time() - start_time)
    done = {
        "status": "complete",
        "global_step": int(step),
        "cycles": int(cycle),
        "elapsed_seconds": elapsed,
        "best_checkpoint": str(best_path.resolve()),
        "best_compromise_checkpoint": str(best_compromise_path.resolve()),
        "best_slope_checkpoint": str(best_slope_path.resolve()) if best_slope_path.exists() else None,
        "phase1_parent_sha256": phase1_provenance["source_sha256"],
    }
    done_path.write_text(json.dumps(done, indent=2), encoding="utf-8")
    print("=" * 118)
    print(f"🏁 [TRAIN DONE] total_time_sec={elapsed:.1f} | global_step={step} | cycles={cycle}")
    print(f"💾 [OFFICIAL BEST] {best_path}")
    print(f"⚖️ [BEST COMPROMISE] {best_compromise_path}")
    print(f"📐 [BEST SLOPE] {best_slope_path if best_slope_path.exists() else 'not available'}")
    print(f"🧬 [PHASE-1 IMMUTABLE COPY] {phase1_copy}")
    print("=" * 118)

    state = torch.load(best_path, map_location="cpu")
    model.prediction_head.load_state_dict(state["prediction_head"], strict=True)
    model.to(device).eval()
    return pd.DataFrame(history), best_path, disturbance_loss
