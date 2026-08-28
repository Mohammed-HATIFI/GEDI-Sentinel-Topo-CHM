from __future__ import annotations

import hashlib
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
    safe_copy_reference,
    set_seed,
    sparse_prediction_loss,
)
from .persistent_growth_loss_v6 import build_growth_loss


def prediction_head_sha256(model) -> str:
    """Stable fingerprint of the Conv3D prediction-head state."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.prediction_head.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _assert_conv3d_only_trainable(model) -> None:
    head_ids = {id(parameter) for parameter in model.prediction_head.parameters()}
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    frozen_head = [
        name
        for name, parameter in model.prediction_head.named_parameters()
        if not parameter.requires_grad
    ]
    if unexpected or frozen_head:
        raise RuntimeError(
            "Phase-2 isolation failure: only the Conv3D prediction head may be trainable; "
            f"unexpected_trainable={unexpected}, frozen_head={frozen_head}"
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


def anti_shrinkage_penalties(
    out: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_slope: float = 0.0,
    lambda_std: float = 0.0,
    lambda_bias: float = 0.0,
    lambda_anti_zero: float = 0.0,
    target_slope: float = 1.0,
    target_std_ratio: float = 1.0,
    min_points: int = 16,
    eps: float = 1e-6,
    anti_zero_min_target: float = 2.5,
    anti_zero_min_prediction: float = 2.0,
    anti_zero_power: float = 2.0,
) -> Dict[str, torch.Tensor | float | bool]:
    """Differentiable anti-compression penalties on sparse GEDI-labelled pixels.

    This is the v4 residual-head adaptation of the penalties audited in the
    B4 C15 Phase-2 anti-shrinkage experiment.  Targets equal to zero remain the
    catalogue no-label sentinel.  Moment terms are skipped for small or nearly
    constant batches; the anti-zero term remains independently available.
    """
    prediction = out[:, 1]
    zero = prediction.sum() * 0.0
    valid = torch.isfinite(target) & torch.isfinite(prediction) & (target != 0)
    n_valid = int(valid.sum().detach().cpu())

    slope_loss = zero
    std_loss = zero
    bias_loss = zero
    moment_active = False
    if n_valid >= int(min_points):
        pred = prediction[valid].float().reshape(-1)
        truth = target[valid].float().reshape(-1)
        pred_centered = pred - pred.mean()
        truth_centered = truth - truth.mean()
        truth_var = (truth_centered * truth_centered).mean()
        if bool(torch.isfinite(truth_var)) and float(truth_var.detach().cpu()) >= float(eps):
            pred_var = (pred_centered * pred_centered).mean().clamp_min(float(eps))
            truth_var = truth_var.clamp_min(float(eps))
            covariance = (pred_centered * truth_centered).mean()
            slope = covariance / truth_var
            std_ratio = torch.sqrt(pred_var) / torch.sqrt(truth_var)
            bias = (pred - truth).mean()
            slope_loss = (slope - float(target_slope)).pow(2)
            std_loss = (std_ratio - float(target_std_ratio)).pow(2)
            bias_loss = bias.pow(2)
            moment_active = True

    anti_zero_loss = zero
    if float(lambda_anti_zero) > 0.0 and n_valid > 0:
        pred = prediction[valid].float().reshape(-1)
        truth = target[valid].float().reshape(-1)
        anti_zero_mask = (truth >= float(anti_zero_min_target)) & (
            pred < float(anti_zero_min_prediction)
        )
        if bool(anti_zero_mask.any()):
            penalty = torch.relu(float(anti_zero_min_prediction) - pred[anti_zero_mask])
            anti_zero_loss = (
                penalty.mean()
                if abs(float(anti_zero_power) - 1.0) < 1e-8
                else penalty.pow(float(anti_zero_power)).mean()
            )

    weighted_total = (
        float(lambda_slope) * slope_loss
        + float(lambda_std) * std_loss
        + float(lambda_bias) * bias_loss
        + float(lambda_anti_zero) * anti_zero_loss
    )
    return {
        "total": weighted_total.to(dtype=prediction.dtype),
        "slope": slope_loss.to(dtype=prediction.dtype),
        "std": std_loss.to(dtype=prediction.dtype),
        "bias": bias_loss.to(dtype=prediction.dtype),
        "anti_zero": anti_zero_loss.to(dtype=prediction.dtype),
        "n_valid": float(n_valid),
        "moment_active": bool(moment_active),
    }


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
        "checkpoint_eligible": bool(metrics.get("checkpoint_eligible", False)),
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
    lambda_slope: float,
    lambda_std: float,
    lambda_bias: float,
    lambda_anti_zero: float,
    anti_shrink_min_points: int,
    anti_zero_min_target: float,
    anti_zero_min_prediction: float,
    anti_zero_power: float,
    min_height: float,
    max_height: float,
) -> Dict[str, float]:
    model.eval()
    truths = []
    predictions = []
    huber_values = []
    growth_values = []
    anti_values = []
    slope_penalty_values = []
    std_penalty_values = []
    bias_penalty_values = []
    anti_zero_penalty_values = []
    moment_active_values = []
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
            anti = anti_shrinkage_penalties(
                out,
                target,
                lambda_slope=lambda_slope,
                lambda_std=lambda_std,
                lambda_bias=lambda_bias,
                lambda_anti_zero=lambda_anti_zero,
                min_points=anti_shrink_min_points,
                anti_zero_min_target=anti_zero_min_target,
                anti_zero_min_prediction=anti_zero_min_prediction,
                anti_zero_power=anti_zero_power,
            )
            total = direct + float(lambda_growth) * growth + anti["total"]
        valid = torch.isfinite(target) & (target != 0)
        if bool(valid.any()):
            truths.append(target[valid].detach().float().cpu().numpy())
            predictions.append(out[:, 1][valid].detach().float().cpu().numpy())
        huber_values.append(float(direct.detach().cpu()))
        growth_values.append(float(growth.detach().cpu()))
        anti_values.append(float(anti["total"].detach().cpu()))
        slope_penalty_values.append(float(anti["slope"].detach().cpu()))
        std_penalty_values.append(float(anti["std"].detach().cpu()))
        bias_penalty_values.append(float(anti["bias"].detach().cpu()))
        anti_zero_penalty_values.append(float(anti["anti_zero"].detach().cpu()))
        moment_active_values.append(float(bool(anti["moment_active"])))
        total_values.append(float(total.detach().cpu()))
    y = np.concatenate(truths) if truths else np.array([], dtype=np.float32)
    p = np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)
    metrics = _metrics_from_arrays(y, p, min_height=min_height, max_height=max_height)
    metrics.update(
        {
            "direct_loss": float(np.mean(huber_values)) if huber_values else float("nan"),
            "growth_loss": float(np.mean(growth_values)) if growth_values else float("nan"),
            "anti_shrink_loss": float(np.mean(anti_values)) if anti_values else float("nan"),
            "slope_penalty_raw": float(np.mean(slope_penalty_values)) if slope_penalty_values else float("nan"),
            "std_penalty_raw": float(np.mean(std_penalty_values)) if std_penalty_values else float("nan"),
            "bias_penalty_raw": float(np.mean(bias_penalty_values)) if bias_penalty_values else float("nan"),
            "anti_zero_penalty_raw": float(np.mean(anti_zero_penalty_values)) if anti_zero_penalty_values else float("nan"),
            "moment_active_fraction": float(np.mean(moment_active_values)) if moment_active_values else 0.0,
            "total_loss": float(np.mean(total_values)) if total_values else float("nan"),
        }
    )
    metrics["article_compromise_score"] = article_compromise_score(metrics)
    return metrics


@torch.no_grad()
def evaluate_sparse_prediction_metrics(
    *,
    model,
    dataset,
    device: str,
    min_height: float = 2.5,
    max_height: float = 40.0,
) -> Dict[str, float]:
    """Evaluate prediction metrics only on a canonical shared dataset.

    No pseudo-target or temporal loss is evaluated here. This lets candidates
    trained with different temporal lengths be selected on identical sparse
    GEDI observations and an identical inference length.
    """
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=str(device).startswith("cuda"),
    )
    truths = []
    predictions = []
    model.eval()
    for x, target, _ in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        out = model(x)
        valid = torch.isfinite(target) & (target != 0)
        if bool(valid.any()):
            truths.append(target[valid].detach().float().cpu().numpy())
            predictions.append(out[:, 1][valid].detach().float().cpu().numpy())
    truth = np.concatenate(truths) if truths else np.array([], dtype=np.float32)
    prediction = (
        np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)
    )
    metrics = _metrics_from_arrays(
        truth,
        prediction,
        min_height=min_height,
        max_height=max_height,
    )
    metrics["article_compromise_score"] = article_compromise_score(metrics)
    return metrics


def train_persistent_temporal_stage(
    *,
    model,
    train_dataset,
    val_dataset,
    official_repo: Path,
    phase1_checkpoint: Path,
    warmstart_prediction_head_checkpoint: Path | None = None,
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
    lambda_slope: float = 0.0,
    lambda_std: float = 0.0,
    lambda_bias: float = 0.0,
    lambda_anti_zero: float = 0.0,
    anti_shrink_min_points: int = 16,
    anti_zero_min_target: float = 2.5,
    anti_zero_min_prediction: float = 2.0,
    anti_zero_power: float = 2.0,
    supervised_loss_name: str = "huber",
    huber_delta: float = 3.0,
    height_weight_mode: str = "none",
    slope_min: float = 0.0,
    slope_max: float = 2.0,
    disturbance_indicator: float = -1.0,
    disturbance_rule: str = "official_echosat",
    persistent_drop_m: float = 5.0,
    persistent_required_consecutive_flags: int = 2,
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
    parent_stage_name: str = "B4 Phase-1",
):
    """Train an isolated Conv3D temporal head with configurable GrowthLoss.

    With ``warmstart_prediction_head_checkpoint=None`` the run is a strict
    Phase-1-direct experiment: B4 stays frozen and the fresh Conv3D head passed
    by the caller is optimized.  A non-null checkpoint retains the historical
    warm-start behaviour for archived notebooks only.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    table_dir = run_dir / "tables"
    done_path = run_dir / "TRAIN_DONE.json"
    best_path = ckpt_dir / "best.ckpt"
    best_any_path = ckpt_dir / "best_any.ckpt"
    best_compromise_path = ckpt_dir / "best_compromise.ckpt"
    best_slope_path = ckpt_dir / "best_slope.ckpt"
    best_r2_path = ckpt_dir / "best_r2.ckpt"
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
    _assert_conv3d_only_trainable(model)

    parent_stage_name = str(parent_stage_name).strip() or "B4 Phase-1"
    is_phase1_parent = parent_stage_name.lower() == "b4 phase-1"
    phase1_copy = ckpt_dir / ("best_phase1.ckpt" if is_phase1_parent else "parent_reference.ckpt")
    phase1_provenance = safe_copy_reference(Path(phase1_checkpoint), phase1_copy)
    parent_meta_name = "best_phase1_meta.json" if is_phase1_parent else "parent_reference_meta.json"
    (ckpt_dir / parent_meta_name).write_text(
        json.dumps(
            {
                "kind": f"immutable {parent_stage_name} parent",
                "parent_stage_name": parent_stage_name,
                **phase1_provenance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if warmstart_prediction_head_checkpoint is None:
        prediction_head_class = model.prediction_head.__class__.__name__
        initialization_kind = (
            "phase1_direct_fresh_swin3d"
            if "swin" in prediction_head_class.lower()
            else "phase1_direct_fresh_conv3d"
        )
        initialization_provenance = {
            "kind": initialization_kind,
            "prediction_head_class": prediction_head_class,
            "prediction_head_initial_sha256": prediction_head_sha256(model),
            "external_prediction_head_checkpoint": None,
        }
    else:
        warmstart_prediction_head_checkpoint = Path(warmstart_prediction_head_checkpoint)
        if not warmstart_prediction_head_checkpoint.exists():
            raise FileNotFoundError(warmstart_prediction_head_checkpoint)
        warmstart_state = torch.load(warmstart_prediction_head_checkpoint, map_location="cpu")
        if "prediction_head" not in warmstart_state:
            raise KeyError(
                "Warm-start checkpoint has no prediction_head: "
                f"{warmstart_prediction_head_checkpoint}"
            )
        model.prediction_head.load_state_dict(warmstart_state["prediction_head"], strict=True)
        warmstart_copy = ckpt_dir / "warmstart_prediction_head_READ_ONLY_COPY.ckpt"
        warmstart_provenance = safe_copy_reference(
            warmstart_prediction_head_checkpoint, warmstart_copy
        )
        initialization_kind = "external_prediction_head_warmstart"
        initialization_provenance = {
            "kind": initialization_kind,
            **warmstart_provenance,
            "prediction_head_initial_sha256": prediction_head_sha256(model),
        }

    disturbance_loss, growth_loss_provenance = build_growth_loss(
        official_repo=official_repo,
        disturbance_rule=disturbance_rule,
        persistent_drop_m=persistent_drop_m,
        persistent_required_consecutive_flags=persistent_required_consecutive_flags,
        disturbance_indicator=disturbance_indicator,
        slope_min=slope_min,
        slope_max=slope_max,
        full_disturbance_window=full_disturbance_window,
        use_l2=True,
        max_intercept_after_disturbance=100,
        disturbance_factor=1,
        no_disturbance_factor=1,
        slope_no_disturbance=-0.0,
    )
    disturbance_loss = disturbance_loss.to(device)

    config = {
        "protocol": (
            f"{parent_stage_name} frozen -> fresh {initialization_provenance['prediction_head_class']} "
            "-> Huber + disturbance-rule GrowthLoss"
            if initialization_kind.startswith("phase1_direct_fresh_")
            else "archived external-head warm-start -> Huber + disturbance-rule GrowthLoss"
        ),
        "phase1_parent": phase1_provenance,
        "parent_stage_name": parent_stage_name,
        "parent_checkpoint": phase1_provenance,
        "prediction_head_initialization": initialization_provenance,
        "reference_frozen": True,
        "train_scope": "full official-style Conv3D temporal prediction head",
        "sequence_policy": "same-month leaf-on sequences; no annual median; no month mixing",
        "supervised_loss": supervised_loss_name,
        "huber_delta": float(huber_delta),
        "lambda_growth": float(lambda_growth),
        "anti_shrinkage": {
            "lambda_slope": float(lambda_slope),
            "lambda_std": float(lambda_std),
            "lambda_bias": float(lambda_bias),
            "lambda_anti_zero": float(lambda_anti_zero),
            "target_slope": 1.0,
            "target_std_ratio": 1.0,
            "min_points": int(anti_shrink_min_points),
            "anti_zero_min_target": float(anti_zero_min_target),
            "anti_zero_min_prediction": float(anti_zero_min_prediction),
            "anti_zero_power": float(anti_zero_power),
        },
        "growth_loss_provenance": growth_loss_provenance,
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
    best_r2_value = -math.inf
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
            r2 = _finite(row.get("val_r2"), -math.inf)
            best_r2_value = max(best_r2_value, r2)
        if not best_r2_path.exists():
            # Legacy partial run: historical weights are unavailable.  Start
            # tracking best-R2 from the first post-upgrade validation onward.
            best_r2_value = -math.inf
            print("⚠️ [BEST R2 RESUME] legacy history has no best_r2.ckpt; tracking restarts now")
        print(f"🔁 [RESUME] step={step:05d} | cycle={cycle:04d} | {last_path}")

    print("\n" + "=" * 118)
    print("🚀 ISOLATED PHASE-2 — FROZEN B4 + TRAINABLE CONV3D TEMPORAL HEAD")
    print("=" * 118)
    print(f"📁 [RUN] {run_dir}")
    print(f"🧬 [PHASE-1 PARENT] {phase1_checkpoint}")
    print(f"🌱 [HEAD INITIALIZATION] {initialization_kind}")
    print(f"🔏 [INITIAL HEAD SHA256] {initialization_provenance['prediction_head_initial_sha256']}")
    print(f"🔒 [REFERENCE] frozen B4 | trainable temporal parameters={model.trainable_parameter_count():,}")
    print(f"🗓️ [SEQUENCES] same-month leaf-on only | train={len(train_dataset.records):,} | val={len(val_dataset.records):,}")
    print(
        f"🧪 [LOSS] Huber(delta={huber_delta:g}) + {lambda_growth:g} x GrowthLoss "
        f"[disturbance_rule={disturbance_rule}, persistent_drop={persistent_drop_m:g}m, "
        f"persistent_K={persistent_required_consecutive_flags}] "
        f"+ anti-shrink(slope={lambda_slope:g}, std={lambda_std:g}, "
        f"bias={lambda_bias:g}, anti-zero={lambda_anti_zero:g})"
    )
    print(f"⚙️ [OPTIMIZER] AdamW lr={learning_rate:.2e} wd={weight_decay:.2e} | grad_clip={grad_clip:g}")
    print(f"🛡️ [GATE] slope>={checkpoint_min_slope:.2f} | std_ratio=[{checkpoint_min_std_ratio:.2f},{checkpoint_max_std_ratio:.2f}] | |bias|<={checkpoint_max_abs_bias:.2f} m")
    print(f"⏱️ [SCHEDULE] val_every={val_every_steps} | patience={patience_evals} | technical max={max_steps}")
    print("=" * 118, flush=True)

    window_truth = []
    window_pred = []
    window_direct = []
    window_growth = []
    window_anti = []
    window_moment_active = []
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
                anti = anti_shrinkage_penalties(
                    out,
                    target,
                    lambda_slope=lambda_slope,
                    lambda_std=lambda_std,
                    lambda_bias=lambda_bias,
                    lambda_anti_zero=lambda_anti_zero,
                    min_points=anti_shrink_min_points,
                    anti_zero_min_target=anti_zero_min_target,
                    anti_zero_min_prediction=anti_zero_min_prediction,
                    anti_zero_power=anti_zero_power,
                )
                total = direct + float(lambda_growth) * growth + anti["total"]
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
            window_anti.append(float(anti["total"].detach().cpu()))
            window_moment_active.append(float(bool(anti["moment_active"])))
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
                    f"anti={np.mean(window_anti):.4f} | "
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
                    "anti_shrink_loss": float(np.mean(window_anti)),
                    "moment_active_fraction": float(np.mean(window_moment_active)),
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
                lambda_slope=lambda_slope,
                lambda_std=lambda_std,
                lambda_bias=lambda_bias,
                lambda_anti_zero=lambda_anti_zero,
                anti_shrink_min_points=anti_shrink_min_points,
                anti_zero_min_target=anti_zero_min_target,
                anti_zero_min_prediction=anti_zero_min_prediction,
                anti_zero_power=anti_zero_power,
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
            val_metrics["checkpoint_eligible"] = bool(eligible)

            print("\n" + "·" * 118)
            print(f"🔎 VALIDATION @ STEP {step:05d} | CYCLE {cycle:04d}")
            print("·" * 118)
            print(
                f"[TRAIN-WINDOW] mae={train_metrics['mae']:.4f} | rmse={train_metrics['rmse']:.4f} | "
                f"r2={train_metrics['r2']:.4f} | bias={train_metrics['bias']:.4f} | "
                f"slope={train_metrics['slope']:.3f} | stdr={train_metrics['std_ratio']:.3f} | "
                f"huber={train_metrics['direct_loss']:.4f} | growth={train_metrics['growth_loss']:.4f} | "
                f"anti={train_metrics['anti_shrink_loss']:.4f} | "
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
                f"huber={val_metrics['direct_loss']:.5f} | growth={val_metrics['growth_loss']:.5f} | "
                f"anti={val_metrics['anti_shrink_loss']:.5f} | "
                f"moment-active={val_metrics['moment_active_fraction']:.2%}"
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

            current_r2 = _finite(val_metrics.get("r2"), -math.inf)
            if math.isfinite(current_r2) and current_r2 > best_r2_value:
                previous_r2 = best_r2_value
                best_r2_value = current_r2
                _save_checkpoint(path=best_r2_path, kind="best_r2", **save_kwargs)
                old_r2 = "none" if not math.isfinite(previous_r2) else f"{previous_r2:.6f}"
                print(f"📈 [BEST R2 CHECKPOINT SAVED] VAL R2 {old_r2} -> {current_r2:.6f} | {best_r2_path}")

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
            window_anti.clear()
            window_moment_active.clear()
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
        "best_r2_checkpoint": str(best_r2_path.resolve()) if best_r2_path.exists() else None,
        "phase1_parent_sha256": phase1_provenance["source_sha256"],
        "prediction_head_initialization": initialization_kind,
        "prediction_head_initial_sha256": initialization_provenance[
            "prediction_head_initial_sha256"
        ],
    }
    done_path.write_text(json.dumps(done, indent=2), encoding="utf-8")
    print("=" * 118)
    print(f"🏁 [TRAIN DONE] total_time_sec={elapsed:.1f} | global_step={step} | cycles={cycle}")
    print(f"💾 [OFFICIAL BEST] {best_path}")
    print(f"⚖️ [BEST COMPROMISE] {best_compromise_path}")
    print(f"📐 [BEST SLOPE] {best_slope_path if best_slope_path.exists() else 'not available'}")
    print(f"📈 [BEST R2] {best_r2_path if best_r2_path.exists() else 'not available'}")
    print(f"🧬 [PHASE-1 IMMUTABLE COPY] {phase1_copy}")
    print("=" * 118)

    state = torch.load(best_path, map_location="cpu")
    model.prediction_head.load_state_dict(state["prediction_head"], strict=True)
    model.to(device).eval()
    return pd.DataFrame(history), best_path, disturbance_loss


def train_phase1_direct_temporal_stage(**kwargs):
    """Strict entry point used by corrected Phase-2 ablation notebooks.

    It rejects any external prediction-head checkpoint, including V4.  Existing
    completed directories are reused only when their config proves the same
    Phase-1-direct initialization protocol.
    """
    forbidden = kwargs.pop("warmstart_prediction_head_checkpoint", None)
    if forbidden is not None:
        raise ValueError(
            "Phase1-direct training forbids warmstart_prediction_head_checkpoint."
        )
    run_dir = Path(kwargs["run_dir"])
    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        initialization = existing.get("prediction_head_initialization", {})
        if not str(initialization.get("kind", "")).startswith("phase1_direct_fresh_"):
            raise RuntimeError(
                f"Refusing to reuse a non-Phase1-direct run directory: {run_dir}"
            )
        if initialization.get("external_prediction_head_checkpoint") is not None:
            raise RuntimeError(
                f"Refusing to reuse a run with an external head checkpoint: {run_dir}"
            )
    return train_persistent_temporal_stage(
        warmstart_prediction_head_checkpoint=None,
        **kwargs,
    )
