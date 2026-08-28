from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_copy_reference(source: Path, destination: Path) -> Dict[str, str]:
    """Copy once; never overwrite an existing checkpoint."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    if destination.exists():
        destination_hash = sha256_file(destination)
        if destination_hash != source_hash:
            raise FileExistsError(
                f"Refusing to overwrite {destination}: existing SHA256 differs from source."
            )
    else:
        shutil.copy2(source, destination)
        destination_hash = sha256_file(destination)
    return {
        "source": str(source),
        "copy": str(destination),
        "source_sha256": source_hash,
        "copy_sha256": destination_hash,
    }


def load_official_growth_loss(official_repo: Path):
    source = (
        Path(official_repo)
        / "fine-tuning"
        / "losses"
        / "regression_loss_disturbance_2heads.py"
    )
    if not source.exists():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("echosat_official_growth_loss", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DisturbanceRegressionLoss2Heads, source


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sparse_reference_l1(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    reference = out[:, 0]
    valid = torch.isfinite(target) & (target != 0)
    if not bool(valid.any()):
        return reference.sum() * 0.0
    return torch.abs(reference[valid] - target[valid]).mean()


def sparse_prediction_loss(
    out: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str = "huber",
    huber_delta: float = 3.0,
    height_weight_mode: str = "none",
) -> torch.Tensor:
    """Direct GEDI supervision for the prediction head on sparse target pixels.

    ``target == 0`` is the catalogue no-label sentinel.  The optional height
    weighting is deliberately simple and fixed before looking at TEST results.
    """
    prediction = out[:, 1]
    valid = torch.isfinite(target) & (target != 0)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    pred = prediction[valid]
    truth = target[valid]
    name = str(loss_name).lower()
    if name == "huber":
        per_pixel = F.huber_loss(pred, truth, reduction="none", delta=float(huber_delta))
    elif name == "l1":
        per_pixel = torch.abs(pred - truth)
    else:
        raise ValueError(f"Unsupported sparse prediction loss: {loss_name!r}")

    weight_mode = str(height_weight_mode).lower()
    if weight_mode == "none":
        weights = torch.ones_like(truth)
    elif weight_mode == "tall":
        weights = torch.where(
            truth >= 30,
            torch.full_like(truth, 4.0),
            torch.where(
                truth >= 20,
                torch.full_like(truth, 2.5),
                torch.where(truth >= 10, torch.full_like(truth, 1.5), torch.ones_like(truth)),
            ),
        )
    else:
        raise ValueError(f"Unsupported height weighting: {height_weight_mode!r}")
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def sparse_prediction_metrics(out: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    prediction = out[:, 1]
    valid = torch.isfinite(target) & (target != 0)
    if not bool(valid.any()):
        return {"prediction_n": 0.0, "prediction_mae": float("nan"), "prediction_bias": float("nan")}
    residual = prediction[valid] - target[valid]
    return {
        "prediction_n": float(valid.sum().cpu()),
        "prediction_mae": float(torch.abs(residual).mean().cpu()),
        "prediction_bias": float(residual.mean().cpu()),
    }


@torch.no_grad()
def temporal_metrics(out: torch.Tensor, disturbance_loss) -> Dict[str, float]:
    pred = out[:, 1]
    ref = out[:, 0]
    disturbance_idx, _ = disturbance_loss.get_disturbance_idx(ref)
    disturbed = disturbance_idx > 0
    delta = pred[:, 1:] - pred[:, :-1]
    non_disturbed = (~disturbed).unsqueeze(1).expand_as(delta)
    violations = (delta < 0) & non_disturbed
    return {
        "median_total_change_m": float(torch.median(pred[:, -1] - pred[:, 0]).cpu()),
        "mean_annual_change_m": float(delta.mean().cpu()),
        "negative_step_fraction_non_disturbed": float(
            violations.float().sum().cpu() / non_disturbed.float().sum().clamp_min(1).cpu()
        ),
        "disturbed_pixel_fraction": float(disturbed.float().mean().cpu()),
        "reference_median_total_change_m": float(torch.median(ref[:, -1] - ref[:, 0]).cpu()),
    }


@torch.no_grad()
def validate_growth(
    model,
    loader,
    disturbance_loss,
    device,
    max_batches: int | None = None,
    *,
    supervised_loss_name: str = "none",
    huber_delta: float = 3.0,
    height_weight_mode: str = "none",
    lambda_growth: float = 3.0,
):
    model.eval()
    rows = []
    losses = []
    supervised_losses = []
    for batch_idx, (x, target, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
            out = model(x)
            growth = disturbance_loss(out, target)
            if str(supervised_loss_name).lower() == "none":
                supervised = sparse_reference_l1(out, target)
            else:
                supervised = sparse_prediction_loss(
                    out,
                    target,
                    loss_name=supervised_loss_name,
                    huber_delta=huber_delta,
                    height_weight_mode=height_weight_mode,
                )
        losses.append(float(growth.detach().cpu()))
        supervised_losses.append(float(supervised.detach().cpu()))
        rows.append({**temporal_metrics(out, disturbance_loss), **sparse_prediction_metrics(out, target)})
    summary = pd.DataFrame(rows).mean(numeric_only=True).to_dict() if rows else {}
    summary["growth_loss"] = float(np.mean(losses)) if losses else float("nan")
    summary["prediction_supervised_loss"] = (
        float(np.mean(supervised_losses)) if supervised_losses else float("nan")
    )
    summary["selection_loss"] = summary["prediction_supervised_loss"] + float(lambda_growth) * summary["growth_loss"]
    return summary


def train_growth_head(
    *,
    model,
    train_dataset,
    val_dataset,
    official_repo: Path,
    run_dir: Path,
    device: str,
    seed: int = 42,
    batch_size: int = 1,
    max_steps: int = 1000,
    steps_per_epoch: int = 128,
    val_every_steps: int = 100,
    patience_evals: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    lambda_regression: float = 3.0,
    slope_min: float = 0.0,
    slope_max: float = 2.0,
    disturbance_indicator: float = -1.0,
    full_disturbance_window: bool = True,
    supervised_loss_name: str = "none",
    huber_delta: float = 3.0,
    height_weight_mode: str = "none",
    selection_metric: str | None = None,
    lr_schedule: str = "constant",
    warmup_fraction: float = 0.10,
    allow_existing_run: bool = False,
):
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not allow_existing_run:
        raise FileExistsError(
            f"Isolated run already exists: {run_dir}. Change RUN_NAME; no file will be overwritten."
        )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    set_seed(seed)

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

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    schedule_name = str(lr_schedule).lower()
    if schedule_name == "constant":
        scheduler = None
    elif schedule_name == "warmup_linear":
        warmup_steps = max(1, int(float(warmup_fraction) * int(max_steps)))

        def lr_factor(current_step: int) -> float:
            if current_step < warmup_steps:
                return max(1e-3, float(current_step + 1) / float(warmup_steps))
            remaining = max(0, int(max_steps) - current_step)
            return max(1e-3, float(remaining) / float(max(1, int(max_steps) - warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    else:
        raise ValueError(f"Unsupported lr_schedule: {lr_schedule!r}")
    amp_enabled = str(device).startswith("cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        # Compatibility with the PyTorch version used by the existing B4 environment.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=str(device).startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=str(device).startswith("cuda"),
    )

    config = {
        "protocol": "ECHOSAT exact GrowthLoss mechanics adapted to B4 features",
        "official_loss_source": str(source_path.resolve()),
        "official_loss_sha256": sha256_file(source_path),
        "reference_frozen": True,
        "prediction_head": "official 3-layer Conv3D head",
        "lambda_regression": float(lambda_regression),
        "supervised_loss_name": str(supervised_loss_name),
        "huber_delta": float(huber_delta),
        "height_weight_mode": str(height_weight_mode),
        "selection_metric": selection_metric or (
            "growth_loss" if str(supervised_loss_name).lower() == "none" else "prediction_mae"
        ),
        "slope_min": float(slope_min),
        "slope_max": float(slope_max),
        "disturbance_indicator": float(disturbance_indicator),
        "disturbance_rule": "drop >50% and >4m; min of next two reference heights <=10m; 3x3 dilation",
        "full_disturbance_window": bool(full_disturbance_window),
        "max_steps": int(max_steps),
        "val_every_steps": int(val_every_steps),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "lr_schedule": schedule_name,
        "warmup_fraction": float(warmup_fraction),
        "seed": int(seed),
        "trainable_parameters": int(model.trainable_parameter_count()),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best = math.inf
    best_path = run_dir / "checkpoints" / "best_prediction_head.ckpt"
    no_improve = 0
    step = 0
    epoch = 0
    while step < int(max_steps):
        train_dataset.set_epoch(epoch)
        for x, target, _ in train_loader:
            if step >= int(max_steps):
                break
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                out = model(x)
                l1_reference = sparse_reference_l1(out, target)
                growth = disturbance_loss(out, target)
                if str(supervised_loss_name).lower() == "none":
                    supervised = l1_reference
                else:
                    supervised = sparse_prediction_loss(
                        out,
                        target,
                        loss_name=supervised_loss_name,
                        huber_delta=huber_delta,
                        height_weight_mode=height_weight_mode,
                    )
                total = supervised + float(lambda_regression) * growth
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            step += 1

            if step == 1 or step % int(val_every_steps) == 0 or step == int(max_steps):
                val = validate_growth(
                    model,
                    val_loader,
                    disturbance_loss,
                    device,
                    max_batches=32,
                    supervised_loss_name=supervised_loss_name,
                    huber_delta=huber_delta,
                    height_weight_mode=height_weight_mode,
                    lambda_growth=lambda_regression,
                )
                row = {
                    "step": step,
                    "epoch": epoch,
                    "train_total": float(total.detach().cpu()),
                    "train_reference_l1_constant": float(l1_reference.detach().cpu()),
                    "train_prediction_supervised_loss": float(supervised.detach().cpu()),
                    "train_growth_loss": float(growth.detach().cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    **{f"val_{k}": v for k, v in val.items()},
                }
                history.append(row)
                pd.DataFrame(history).to_csv(run_dir / "tables" / "training_history.csv", index=False)
                metric_name = selection_metric or (
                    "growth_loss" if str(supervised_loss_name).lower() == "none" else "prediction_mae"
                )
                score = float(val.get(metric_name, math.inf))
                if math.isfinite(score) and score < best:
                    best = score
                    no_improve = 0
                    torch.save(
                        {
                            "prediction_head": model.prediction_head.state_dict(),
                            "step": step,
                            "val_growth_loss": score,
                            "selection_metric": metric_name,
                            "selection_score": score,
                            "config": config,
                        },
                        best_path,
                    )
                else:
                    no_improve += 1
                print(
                    f"step={step:05d} total={row['train_total']:.5f} "
                    f"growth={row['train_growth_loss']:.5f} val_{metric_name}={score:.5f} "
                    f"best={best:.5f} patience={no_improve}/{patience_evals}"
                )
                model.train()
                if no_improve >= int(patience_evals):
                    step = int(max_steps)
                    break
        epoch += 1

    if not best_path.exists():
        raise RuntimeError("Training ended without a valid isolated checkpoint.")
    best_state = torch.load(best_path, map_location="cpu")
    model.prediction_head.load_state_dict(best_state["prediction_head"], strict=True)
    model.to(device).eval()
    return pd.DataFrame(history), best_path, disturbance_loss


def regression_metrics(frame: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    clean = frame[["rh95", pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
    y = clean["rh95"].to_numpy(float)
    p = clean[pred_col].to_numpy(float)
    if not len(clean):
        return {k: float("nan") for k in ["n", "mae", "rmse", "r2", "bias", "slope", "std_ratio"]}
    residual = p - y
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / sst) if sst > 0 else float("nan"),
        "bias": float(np.mean(residual)),
        "slope": float(np.polyfit(y, p, 1)[0]) if len(y) > 1 else float("nan"),
        "std_ratio": float(np.std(p) / np.std(y)) if np.std(y) > 0 else float("nan"),
    }


@torch.no_grad()
def collect_sparse_loader_predictions(
    *,
    model,
    loader,
    device: str,
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Collect sparse GEDI predictions from a fixed crop loader for screening.

    This is a VAL-only, deterministic screening diagnostic. It is not a
    substitute for the independent unique-shot TEST evaluation.
    """
    model.eval()
    rows = []
    for batch_idx, (x, target, meta) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
            out = model(x)
        prediction = out[:, 1]
        valid = torch.isfinite(target) & (target != 0)
        if not bool(valid.any()):
            continue
        truth = target[valid].float().cpu().numpy()
        pred = prediction[valid].float().cpu().numpy()
        rows.extend({"rh95": float(y), "prediction": float(p)} for y, p in zip(truth, pred))
    return pd.DataFrame(rows, columns=["rh95", "prediction"])


@torch.no_grad()
def evaluate_sparse_crops(
    *,
    model,
    records,
    shots: pd.DataFrame,
    device: str,
    crop_size: int = 96,
    halo: int = 16,
    drop_channels: Sequence[int] = (12, 13),
) -> pd.DataFrame:
    """Evaluate every selected GEDI occurrence once using haloed sequence crops."""
    model.eval()
    shot_groups = {
        str(sample_id): group.copy()
        for sample_id, group in shots.groupby(shots["sample_id"].astype(str), sort=False)
    }
    rows = []
    drop_channels = tuple(sorted(set(int(v) for v in drop_channels)))

    for record in records:
        first = np.load(record.x_paths[0], mmap_mode="r", allow_pickle=False)
        height, width, channels = first.shape
        keep = [c for c in range(channels) if c not in drop_channels]
        for core_r0 in range(0, height, int(crop_size)):
            for core_c0 in range(0, width, int(crop_size)):
                core_r1 = min(height, core_r0 + int(crop_size))
                core_c1 = min(width, core_c0 + int(crop_size))
                in_r0 = max(0, core_r0 - int(halo))
                in_c0 = max(0, core_c0 - int(halo))
                in_r1 = min(height, core_r1 + int(halo))
                in_c1 = min(width, core_c1 + int(halo))

                cubes = []
                for x_path in record.x_paths:
                    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
                    crop = np.asarray(x[in_r0:in_r1, in_c0:in_c1, keep], dtype=np.float32)
                    cube = torch.from_numpy(np.moveaxis(crop, -1, 0))
                    pad_h = (16 - cube.shape[-2] % 16) % 16
                    pad_w = (16 - cube.shape[-1] % 16) % 16
                    cube = F.pad(cube, (0, pad_w, 0, pad_h), mode="replicate")
                    cubes.append(cube)
                sequence = torch.stack(cubes).unsqueeze(0).to(device)
                with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                    out = model(sequence)[0].float().cpu().numpy()

                for year_idx, (year, month, sample_id) in enumerate(
                    zip(record.years, record.months, record.sample_ids)
                ):
                    group = shot_groups.get(str(sample_id))
                    if group is None:
                        continue
                    rr = pd.to_numeric(group["local_row"], errors="coerce")
                    cc = pd.to_numeric(group["local_col"], errors="coerce")
                    yy = pd.to_numeric(group["rh95"], errors="coerce")
                    valid = (
                        rr.notna()
                        & cc.notna()
                        & yy.notna()
                        & (yy > 0)
                        & (rr >= core_r0)
                        & (rr < core_r1)
                        & (cc >= core_c0)
                        & (cc < core_c1)
                    )
                    for idx in group.index[valid]:
                        local_r = int(rr.loc[idx]) - in_r0
                        local_c = int(cc.loc[idx]) - in_c0
                        rows.append(
                            {
                                "split": record.split,
                                "patch_key": record.patch_key,
                                "sample_id": sample_id,
                                "aux_shot_uid": str(group.loc[idx, "aux_shot_uid"]),
                                "year": int(year),
                                "month": int(month),
                                "rh95": float(yy.loc[idx]),
                                "pred_off_reference": float(out[0, year_idx, local_r, local_c]),
                                "pred_on_growthloss": float(out[1, year_idx, local_r, local_c]),
                            }
                        )
    result = pd.DataFrame(rows)
    if len(result):
        result = (
            result.assign(abs_error_off=lambda x: (x.pred_off_reference - x.rh95).abs())
            .sort_values(["aux_shot_uid", "abs_error_off"])
            .drop_duplicates("aux_shot_uid")
            .reset_index(drop=True)
        )
    return result
