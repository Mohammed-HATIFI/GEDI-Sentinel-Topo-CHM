from __future__ import annotations

# vPAULS_TRACK_LEVEL_SHIFTED_HUBER — Pauls-style track-level LS + Huber
# Old-best Ifran loss behavior: asymmetric weighted MAE + anti-shrink moment loss via Step06 args.
# No prediction floor and no anti-collapse penalty. GE2/unique evaluation upper bound defaults to 40 m for Ifran.

import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.common import (
    BINS_DEFAULT,
    HEIGHT_CLASS_KEYS,
)
from training.data import augment_patch_with_aux

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

try:
    _STDOUT_IS_TTY = bool(getattr(sys.stdout, "isatty", lambda: False)())
except Exception:
    _STDOUT_IS_TTY = False

_FORCE_TQDM = str(os.environ.get("CHM_FORCE_TQDM", "0")).strip().lower() in {"1", "true", "yes", "y"}
_SIMPLE_LOGS = str(os.environ.get("CHM_SIMPLE_LOGS", "0")).strip().lower() in {"1", "true", "yes", "y"}
_PROGRESS_PRINT_EVERY_DEFAULT = max(1, int(os.environ.get("CHM_PROGRESS_EVERY", "10")))


def _should_use_tqdm() -> bool:
    if _SIMPLE_LOGS:
        return False
    return bool(_HAS_TQDM and (_STDOUT_IS_TTY or _FORCE_TQDM))


def _progress_bar(step: int, total: Optional[int], width: int = 26) -> str:
    if total is None or total <= 0:
        return "[" + "." * width + "]"
    ratio = max(0.0, min(1.0, float(step) / float(total)))
    n_fill = int(round(ratio * width))
    n_fill = max(0, min(width, n_fill))
    return "[" + "#" * n_fill + "-" * (width - n_fill) + "]"


def _print_progress_line(
    *,
    prefix: str,
    step: int,
    total_steps: Optional[int],
    metrics: Dict[str, object],
) -> None:
    total_txt = "?" if total_steps is None or total_steps <= 0 else str(int(total_steps))
    bar = _progress_bar(step, total_steps)
    bits = [f"{prefix} {bar} {step}/{total_txt}"]

    def _is_empty(v: object) -> bool:
        return v in (None, "")

    def _is_nan_text(v: object) -> bool:
        try:
            if isinstance(v, str) and v.strip().lower() in {"nan", "none"}:
                return True
            x = float(v)
            return not math.isfinite(x)
        except Exception:
            return False

    pts_val = metrics.get("pts", None)
    try:
        pts_int = int(float(pts_val)) if pts_val is not None and str(pts_val).strip() != "" else None
    except Exception:
        pts_int = None

    # When a short training window contains no valid GEDI points, printing a full
    # line full of nan metrics is misleading. Show the useful status only.
    if pts_int is not None and pts_int <= 0:
        for k in ("lr", "pts", "best", "pat"):
            if k in metrics and not _is_empty(metrics[k]):
                bits.append(f"{k}={metrics[k]}")
        bits.append("status=no_valid_gedi_in_window")
        print(" | ".join(bits), flush=True)
        return

    for k in ("lr", "tot", "reg", "hub", "mae", "rmse", "r2", "bias", "stdr", "slope", "corr", "pts", "best", "pat"):
        if k in metrics and (not _is_empty(metrics[k])) and (not _is_nan_text(metrics[k])):
            bits.append(f"{k}={metrics[k]}")

    print(" | ".join(bits), flush=True)


def _maybe_print_cuda_status(device: torch.device, prefix: str) -> None:
    if str(device) != "cuda" or not torch.cuda.is_available():
        print(f"[{prefix}] device={device}", flush=True)
        return

    try:
        idx = int(torch.cuda.current_device())
        name = torch.cuda.get_device_name(idx)
        alloc = torch.cuda.memory_allocated(idx) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)
        print(
            f"[{prefix}] device=cuda:{idx} | name={name} | "
            f"mem_alloc={alloc:.1f} MiB | mem_reserved={reserved:.1f} MiB",
            flush=True,
        )
    except Exception as e:
        print(f"[{prefix}] device=cuda | status_unavailable={e!r}", flush=True)


# -------------------------------------------------------------------------------------------------
# Small safe helper
# -------------------------------------------------------------------------------------------------
def _sf(v, default: float = float("nan")) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


# -------------------------------------------------------------------------------------------------
# Optional Pauls track-shift export for QGIS
# -------------------------------------------------------------------------------------------------
# The shifted-Huber loss selects a single candidate offset per GEDI track during
# training/evaluation.  By default this decision exists only in memory.  The
# helpers below export that decision as CSV files when CHM_EXPORT_TRACK_SHIFTS=1.
#
# Recommended environment variables from Step06:
#   CHM_EXPORT_TRACK_SHIFTS=1
#   CHM_TRACK_SHIFT_EXPORT_DIR=<run_dir>/track_shift_exports
#   CHM_TRACK_SHIFT_EXPORT_SPLITS=val,test   # default: val
#   CHM_TRACK_SHIFT_EXPORT_PIXEL_SIZE_M=10   # default: 10 m
#
# CSVs are intentionally used because QGIS can load them directly as delimited
# text layers using aux_lon/aux_lat.  The files contain both per-shot rows and
# one per-track summary row, making the export suitable for visual checks and
# later conversion to GeoPackage.

_TRACK_SHIFT_EXPORT_WARNED = False
_TRACK_SHIFT_EXPORT_PRINTED = False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_csv_env(name: str, default: str) -> set[str]:
    raw = os.environ.get(name, default)
    return {x.strip().lower() for x in str(raw).split(",") if x.strip()}


def _track_shift_export_enabled(context: Optional[Dict[str, Any]]) -> bool:
    if not _env_flag("CHM_EXPORT_TRACK_SHIFTS", default=False):
        return False
    ctx = context or {}
    split = str(ctx.get("split", ctx.get("mode", "unknown"))).strip().lower()
    phase = str(ctx.get("phase", "all")).strip().lower()

    allowed_splits = _split_csv_env("CHM_TRACK_SHIFT_EXPORT_SPLITS", "val")
    allowed_phases = _split_csv_env("CHM_TRACK_SHIFT_EXPORT_PHASES", "all")
    if "all" not in allowed_splits and split not in allowed_splits:
        return False
    if "all" not in allowed_phases and phase not in allowed_phases:
        return False

    every_n = max(1, int(_sf(os.environ.get("CHM_TRACK_SHIFT_EXPORT_EVERY_N_CYCLES", "1"), default=1)))
    cycle = int(_sf(ctx.get("cycle_idx", ctx.get("cycle", 0)), default=0))
    return (cycle % every_n) == 0


def _track_shift_export_dir() -> Path:
    raw = os.environ.get("CHM_TRACK_SHIFT_EXPORT_DIR", "track_shift_exports")
    return Path(raw).expanduser()


def _track_shift_pixel_size_m() -> float:
    px = _sf(os.environ.get("CHM_TRACK_SHIFT_EXPORT_PIXEL_SIZE_M", "10"), default=10.0)
    return float(px if math.isfinite(px) and px > 0 else 10.0)


def _candidate_offsets(radius_px: int) -> List[Tuple[int, int]]:
    r = max(0, int(radius_px))
    return [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1)]


def _to_numpy_1d(x: torch.Tensor, dtype=None) -> np.ndarray:
    arr = x.detach().cpu().numpy().reshape(-1)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _json_safe_scalar(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        vf = float(v)
        return vf if math.isfinite(vf) else ""
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8")
        except Exception:
            return str(v)
    if torch.is_tensor(v):
        try:
            return _json_safe_scalar(v.detach().cpu().item())
        except Exception:
            return str(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else ""
    if v is None:
        return ""
    return v


def _meta_values_for_points(
    meta: Optional[Dict[str, Any]],
    key: str,
    batch_index_np: np.ndarray,
    point_index_np: np.ndarray,
    *,
    default: Any = "",
) -> List[Any]:
    n = int(batch_index_np.size)
    if not isinstance(meta, dict) or key not in meta:
        return [default] * n
    try:
        v = meta[key]
        if torch.is_tensor(v):
            arr = v.detach().cpu().numpy()
        else:
            arr = np.asarray(v)
        vals = arr[batch_index_np, point_index_np]
        vals = np.asarray(vals).reshape(-1).tolist()
        return [_json_safe_scalar(x) for x in vals]
    except Exception:
        return [default] * n


def _write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fieldnames), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _export_track_shift_decisions(
    *,
    target: torch.Tensor,
    pred_exact: torch.Tensor,
    pred_shifted: torch.Tensor,
    group_track_ids: torch.Tensor,
    orig_track_ids: torch.Tensor,
    aux_rows: torch.Tensor,
    aux_cols: torch.Tensor,
    batch_indices: torch.Tensor,
    point_indices: torch.Tensor,
    details: Dict[str, torch.Tensor],
    radius_px: int,
    meta: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
) -> None:
    """Export per-shot and per-track Pauls shift decisions as CSV.

    The export never changes the loss.  If CHM_TRACK_SHIFT_EXPORT_STRICT=1,
    an export failure raises an exception; otherwise it prints a warning once and
    continues training.
    """
    global _TRACK_SHIFT_EXPORT_WARNED, _TRACK_SHIFT_EXPORT_PRINTED

    if not _track_shift_export_enabled(context):
        return

    try:
        ctx = context or {}
        split = str(ctx.get("split", ctx.get("mode", "unknown"))).strip().lower()
        phase = int(_sf(ctx.get("phase", 0), default=0))
        cycle = int(_sf(ctx.get("cycle_idx", ctx.get("cycle", 0)), default=0))
        batch_step = int(_sf(ctx.get("batch_step", ctx.get("step", 0)), default=0))
        pixel_size_m = _track_shift_pixel_size_m()

        out_dir = _track_shift_export_dir()
        prefix = f"track_shifts_{split}_phase{phase}_cycle{cycle:04d}"
        shot_csv = out_dir / f"{prefix}_shots.csv"
        track_csv = out_dir / f"{prefix}_tracks.csv"

        target_np = _to_numpy_1d(target, np.float32)
        pred_exact_np = _to_numpy_1d(pred_exact, np.float32)
        pred_shifted_np = _to_numpy_1d(pred_shifted, np.float32)
        group_tid_np = _to_numpy_1d(group_track_ids.long(), np.int64)
        orig_tid_np = _to_numpy_1d(orig_track_ids.long(), np.int64)
        row_np = _to_numpy_1d(aux_rows.long(), np.int64)
        col_np = _to_numpy_1d(aux_cols.long(), np.int64)
        batch_np = _to_numpy_1d(batch_indices.long(), np.int64)
        point_np = _to_numpy_1d(point_indices.long(), np.int64)

        selected_idx_np = _to_numpy_1d(details["selected_idx"].long(), np.int64)
        track_n_np = _to_numpy_1d(details["track_n_points"].long(), np.int64)
        loss_exact_np = _to_numpy_1d(details["track_loss_exact"].float(), np.float32)
        loss_sel_np = _to_numpy_1d(details["track_loss_selected"].float(), np.float32)
        loss_gain_np = _to_numpy_1d(details["track_loss_gain"].float(), np.float32)
        was_shifted_np = _to_numpy_1d(details["was_shifted"].long(), np.int64)
        center_idx = int(_sf(details.get("center_idx", torch.tensor(0)), default=0))

        offsets = _candidate_offsets(radius_px)
        dr_np = np.zeros_like(selected_idx_np, dtype=np.int64)
        dc_np = np.zeros_like(selected_idx_np, dtype=np.int64)
        for i, idx in enumerate(selected_idx_np):
            if 0 <= int(idx) < len(offsets):
                dr_np[i], dc_np[i] = offsets[int(idx)]

        shot_id = _meta_values_for_points(meta, "aux_shot_id", batch_np, point_np, default="")
        shot_uid = _meta_values_for_points(meta, "aux_shot_uid", batch_np, point_np, default="")
        source_rowid = _meta_values_for_points(meta, "aux_source_rowid", batch_np, point_np, default="")
        gedi_ordinal = _meta_values_for_points(meta, "aux_gedi_ordinal", batch_np, point_np, default="")
        gedi_date = _meta_values_for_points(meta, "aux_gedi_date", batch_np, point_np, default="")
        temporal_delta = _meta_values_for_points(meta, "aux_temporal_delta_days", batch_np, point_np, default="")
        abs_temporal_delta = _meta_values_for_points(meta, "aux_abs_temporal_delta_days", batch_np, point_np, default="")
        lon = _meta_values_for_points(meta, "aux_lon", batch_np, point_np, default="")
        lat = _meta_values_for_points(meta, "aux_lat", batch_np, point_np, default="")

        shot_fields = [
            "split", "phase", "cycle", "batch_step", "batch_index", "point_index",
            "group_track_id", "aux_track_id", "track_n_points",
            "candidate_idx", "center_candidate_idx", "shift_dr_px", "shift_dc_px",
            "shift_dx_m", "shift_dy_m", "track_was_shifted",
            "track_loss_exact", "track_loss_shifted", "track_loss_gain",
            "aux_row", "aux_col", "shifted_row", "shifted_col",
            "aux_lon", "aux_lat", "aux_shot_id", "aux_shot_uid",
            "aux_source_rowid", "aux_gedi_ordinal", "aux_gedi_date",
            "aux_temporal_delta_days", "aux_abs_temporal_delta_days",
            "rh95", "pred_exact", "pred_shifted", "point_abs_error_exact", "point_abs_error_shifted",
        ]
        shot_rows: List[Dict[str, Any]] = []
        for i in range(int(target_np.size)):
            shifted_row = int(row_np[i] + dr_np[i])
            shifted_col = int(col_np[i] + dc_np[i])
            shot_rows.append({
                "split": split,
                "phase": phase,
                "cycle": cycle,
                "batch_step": batch_step,
                "batch_index": int(batch_np[i]),
                "point_index": int(point_np[i]),
                "group_track_id": int(group_tid_np[i]),
                "aux_track_id": int(orig_tid_np[i]),
                "track_n_points": int(track_n_np[i]),
                "candidate_idx": int(selected_idx_np[i]),
                "center_candidate_idx": int(center_idx),
                "shift_dr_px": int(dr_np[i]),
                "shift_dc_px": int(dc_np[i]),
                "shift_dx_m": float(dc_np[i] * pixel_size_m),
                "shift_dy_m": float(-dr_np[i] * pixel_size_m),
                "track_was_shifted": int(was_shifted_np[i]),
                "track_loss_exact": float(loss_exact_np[i]),
                "track_loss_shifted": float(loss_sel_np[i]),
                "track_loss_gain": float(loss_gain_np[i]),
                "aux_row": int(row_np[i]),
                "aux_col": int(col_np[i]),
                "shifted_row": shifted_row,
                "shifted_col": shifted_col,
                "aux_lon": lon[i],
                "aux_lat": lat[i],
                "aux_shot_id": shot_id[i],
                "aux_shot_uid": shot_uid[i],
                "aux_source_rowid": source_rowid[i],
                "aux_gedi_ordinal": gedi_ordinal[i],
                "aux_gedi_date": gedi_date[i],
                "aux_temporal_delta_days": temporal_delta[i],
                "aux_abs_temporal_delta_days": abs_temporal_delta[i],
                "rh95": float(target_np[i]),
                "pred_exact": float(pred_exact_np[i]),
                "pred_shifted": float(pred_shifted_np[i]),
                "point_abs_error_exact": float(abs(pred_exact_np[i] - target_np[i])),
                "point_abs_error_shifted": float(abs(pred_shifted_np[i] - target_np[i])),
            })
        _write_csv_rows(shot_csv, shot_rows, shot_fields)

        track_fields = [
            "split", "phase", "cycle", "batch_step", "group_track_id", "aux_track_id",
            "track_n_points", "candidate_idx", "center_candidate_idx", "shift_dr_px", "shift_dc_px",
            "shift_dx_m", "shift_dy_m", "track_was_shifted",
            "track_loss_exact", "track_loss_shifted", "track_loss_gain",
            "mean_lon", "mean_lat", "mean_rh95", "mean_pred_exact", "mean_pred_shifted",
        ]
        track_rows: List[Dict[str, Any]] = []
        for gid in np.unique(group_tid_np):
            m = group_tid_np == gid
            first = int(np.where(m)[0][0])
            lon_vals = np.asarray([_sf(x, default=np.nan) for x, ok in zip(lon, m) if ok], dtype=float)
            lat_vals = np.asarray([_sf(x, default=np.nan) for x, ok in zip(lat, m) if ok], dtype=float)
            track_rows.append({
                "split": split,
                "phase": phase,
                "cycle": cycle,
                "batch_step": batch_step,
                "group_track_id": int(gid),
                "aux_track_id": int(orig_tid_np[first]),
                "track_n_points": int(track_n_np[first]),
                "candidate_idx": int(selected_idx_np[first]),
                "center_candidate_idx": int(center_idx),
                "shift_dr_px": int(dr_np[first]),
                "shift_dc_px": int(dc_np[first]),
                "shift_dx_m": float(dc_np[first] * pixel_size_m),
                "shift_dy_m": float(-dr_np[first] * pixel_size_m),
                "track_was_shifted": int(was_shifted_np[first]),
                "track_loss_exact": float(loss_exact_np[first]),
                "track_loss_shifted": float(loss_sel_np[first]),
                "track_loss_gain": float(loss_gain_np[first]),
                "mean_lon": float(np.nanmean(lon_vals)) if np.isfinite(lon_vals).any() else "",
                "mean_lat": float(np.nanmean(lat_vals)) if np.isfinite(lat_vals).any() else "",
                "mean_rh95": float(np.nanmean(target_np[m])),
                "mean_pred_exact": float(np.nanmean(pred_exact_np[m])),
                "mean_pred_shifted": float(np.nanmean(pred_shifted_np[m])),
            })
        _write_csv_rows(track_csv, track_rows, track_fields)

        if not _TRACK_SHIFT_EXPORT_PRINTED:
            print(
                f"[TRACK-SHIFT EXPORT] enabled | dir={out_dir} | splits={os.environ.get('CHM_TRACK_SHIFT_EXPORT_SPLITS', 'val')} | "
                "files=track_shifts_<split>_phase<phase>_cycleXXXX_{shots,tracks}.csv",
                flush=True,
            )
            _TRACK_SHIFT_EXPORT_PRINTED = True
    except Exception as exc:
        if _env_flag("CHM_TRACK_SHIFT_EXPORT_STRICT", default=False):
            raise
        if not _TRACK_SHIFT_EXPORT_WARNED:
            print(f"[TRACK-SHIFT EXPORT WARNING] export failed but training continues: {type(exc).__name__}: {exc}", flush=True)
            _TRACK_SHIFT_EXPORT_WARNED = True


def _eval_max_height_m() -> float:
    """Legacy upper bound used by GE2/unique evaluation masks."""
    return _sf(os.environ.get("CHM_EVAL_MAX_HEIGHT", "40"), default=40.0)


def _eval_primary_min_height_m() -> float:
    """Lower bound for the generic primary evaluation domain."""
    return _sf(
        os.environ.get("CHM_EVAL_PRIMARY_MIN_HEIGHT", os.environ.get("CHM_EVAL_MIN_HEIGHT", "2.0")),
        default=2.0,
    )


def _eval_primary_max_height_m() -> float:
    """Upper bound for the generic primary evaluation domain."""
    return _sf(
        os.environ.get("CHM_EVAL_PRIMARY_MAX_HEIGHT", os.environ.get("CHM_EVAL_MAX_HEIGHT", "40")),
        default=_eval_max_height_m(),
    )


def _eval_audit_min_height_m() -> float:
    """Lower bound for an optional audit-tail evaluation domain."""
    return _sf(os.environ.get("CHM_EVAL_AUDIT_MIN_HEIGHT", "nan"), default=float("nan"))


def _eval_audit_max_height_m() -> float:
    """Upper bound for an optional audit-tail evaluation domain."""
    return _sf(os.environ.get("CHM_EVAL_AUDIT_MAX_HEIGHT", "nan"), default=float("nan"))


def _domain_mask_np(
    y_true: np.ndarray,
    lo: float,
    hi: float,
    *,
    inclusive_hi: bool = True,
) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.float64)
    finite = np.isfinite(y)
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or float(hi) <= float(lo):
        return finite & False
    if inclusive_hi:
        return finite & (y >= float(lo)) & (y <= float(hi))
    return finite & (y >= float(lo)) & (y < float(hi))


def _domain_mask_torch(
    y: torch.Tensor,
    lo: float,
    hi: float,
    *,
    inclusive_hi: bool = True,
) -> torch.Tensor:
    finite = torch.isfinite(y)
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or float(hi) <= float(lo):
        return finite & torch.zeros_like(finite, dtype=torch.bool)
    if inclusive_hi:
        return finite & (y >= float(lo)) & (y <= float(hi))
    return finite & (y >= float(lo)) & (y < float(hi))



def _anti_shrinkage_brief(metrics: Dict[str, object]) -> str:
    return (
        f"bias={_sf(metrics.get('bias')):.3f} | "
        f"stdr={_sf(metrics.get('std_ratio')):.3f} | "
        f"slope={_sf(metrics.get('slope')):.3f} | "
        f"corr={_sf(metrics.get('corr')):.3f}"
    )


def _resolve_cycle_index(*, cycle_idx: Optional[int] = None, epoch_idx: Optional[int] = None) -> int:
    """
    Resolve the step-based cycle index used in logs.

    We keep ``epoch_idx`` as a compatibility alias because older callers still
    pass that keyword even when the training loop is actually step-based.
    """
    if cycle_idx is not None:
        return int(cycle_idx)
    if epoch_idx is not None:
        return int(epoch_idx)
    return 0


# -------------------------------------------------------------------------------------------------
# Streaming regression metrics
# -------------------------------------------------------------------------------------------------

class RunningReg:
    """Streaming exact MAE / MSE / RMSE / R² / Bias + anti-shrinkage diagnostics."""

    def __init__(self):
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sse = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_pred = 0.0
        self.sum_pred2 = 0.0
        self.sum_pred_y = 0.0
        self.pred_min = math.inf
        self.pred_max = -math.inf
        self.true_min = math.inf
        self.true_max = -math.inf

    def update(self, pred: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        pred = pred.detach().float().view(-1)
        y = y.detach().float().view(-1)

        if mask is not None:
            mask = mask.detach().view(-1).bool()
            if not bool(mask.any()):
                return
            pred = pred[mask]
            y = y[mask]

        if y.numel() == 0:
            return

        diff = y - pred
        self.n += int(y.numel())
        self.sum_abs += float(diff.abs().sum().item())
        self.sum_sse += float((diff * diff).sum().item())
        self.sum_y += float(y.sum().item())
        self.sum_y2 += float((y * y).sum().item())
        self.sum_pred += float(pred.sum().item())
        self.sum_pred2 += float((pred * pred).sum().item())
        self.sum_pred_y += float((pred * y).sum().item())
        self.pred_min = min(self.pred_min, float(pred.min().item()))
        self.pred_max = max(self.pred_max, float(pred.max().item()))
        self.true_min = min(self.true_min, float(y.min().item()))
        self.true_max = max(self.true_max, float(y.max().item()))

    def compute(self, global_mean: Optional[float] = None) -> Dict[str, float]:
        """
        Compute streaming regression metrics.

        Scientific convention used here:
        - MAE       = mean(|y - y_hat|)
        - MSE       = mean((y - y_hat)^2)
        - RMSE      = sqrt(MSE)
        - Bias      = mean(y_hat - y)
        - R²        = 1 - SSE / SST
        - Slope     = slope of least-squares regression y_hat ~ a + b*y
        - Corr      = Pearson correlation between y_hat and y
        - StdRatio  = std(y_hat) / std(y)

        Notes
        -----
        If ``global_mean`` is None, SST is computed with the *local* mean of the
        current sample set. This is the standard definition for a split-level or
        subgroup-level R².

        If ``global_mean`` is provided, SST is computed against that external mean.
        This is useful only as an auxiliary "global-reference R²" diagnostic.
        It should not replace the standard within-group R².
        """
        if self.n <= 0:
            return {
                "mae": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "bias": np.nan,
                "pred_mean": np.nan,
                "pred_std": np.nan,
                "pred_min": np.nan,
                "pred_max": np.nan,
                "true_mean": np.nan,
                "true_std": np.nan,
                "true_min": np.nan,
                "true_max": np.nan,
                "std_ratio": np.nan,
                "slope": np.nan,
                "corr": np.nan,
                "n": 0,
            }

        n = float(self.n)
        mae = self.sum_abs / n
        mse = self.sum_sse / n
        rmse = float(np.sqrt(mse))

        mean_y_local = self.sum_y / n
        mean_pred = self.sum_pred / n
        bias = (self.sum_pred - self.sum_y) / n

        if global_mean is not None:
            ref_mean = float(global_mean)
        else:
            ref_mean = float(mean_y_local)

        sst = self.sum_y2 - 2.0 * ref_mean * self.sum_y + n * (ref_mean ** 2)
        if sst <= 0.0 or not math.isfinite(sst):
            r2 = float("nan")
        else:
            r2 = 1.0 - (self.sum_sse / sst)

        var_y_sum = self.sum_y2 - n * (mean_y_local ** 2)
        var_pred_sum = self.sum_pred2 - n * (mean_pred ** 2)
        cov_sum = self.sum_pred_y - n * mean_pred * mean_y_local

        true_std = float(np.sqrt(max(var_y_sum / n, 0.0)))
        pred_std = float(np.sqrt(max(var_pred_sum / n, 0.0)))
        std_ratio = float(pred_std / max(true_std, 1e-8)) if math.isfinite(pred_std) and math.isfinite(true_std) else float("nan")

        if var_y_sum > 0.0 and math.isfinite(var_y_sum):
            slope = float(cov_sum / var_y_sum)
        else:
            slope = float("nan")

        denom = var_pred_sum * var_y_sum
        if denom > 0.0 and math.isfinite(denom):
            corr = float(cov_sum / np.sqrt(denom))
            corr = float(np.clip(corr, -1.0, 1.0))
        else:
            corr = float("nan")

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "bias": float(bias),
            "pred_mean": float(mean_pred),
            "pred_std": float(pred_std),
            "pred_min": float(self.pred_min),
            "pred_max": float(self.pred_max),
            "true_mean": float(mean_y_local),
            "true_std": float(true_std),
            "true_min": float(self.true_min),
            "true_max": float(self.true_max),
            "std_ratio": float(std_ratio),
            "slope": float(slope),
            "corr": float(corr),
            "n": int(self.n),
        }


# -------------------------------------------------------------------------------------------------
# AUROC / AUPRC reservoir sampling
# -------------------------------------------------------------------------------------------------
class Reservoir:

    def __init__(self, k: int, seed: int = 123):
        self.k = int(k)
        self.rng = np.random.default_rng(int(seed))
        self.n_seen = 0
        self.scores = np.empty((0,), dtype=np.float32)
        self.labels = np.empty((0,), dtype=np.int8)

    def add(self, scores: np.ndarray, labels: np.ndarray):
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        labels = np.asarray(labels, dtype=np.int8).reshape(-1)

        for s, y in zip(scores, labels):
            self.n_seen += 1
            if self.scores.size < self.k:
                self.scores = np.append(self.scores, s)
                self.labels = np.append(self.labels, y)
            else:
                j = self.rng.integers(0, self.n_seen)
                if j < self.k:
                    self.scores[j] = s
                    self.labels[j] = y

    def get(self):
        return self.scores, self.labels


def _auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)

    if scores.size == 0:
        return float("nan")

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)

    s_sorted = scores[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while (j + 1) < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1

    return float(((ranks[labels == 1]).sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _auc_pr(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)

    if scores.size == 0:
        return float("nan")

    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    if int((labels == 0).sum()) == 0:
        return 1.0

    order = np.argsort(-scores)
    y = labels[order]
    tp = 0
    fp = 0
    ap = 0.0
    prev_r = 0.0

    for yi in y:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        r = tp / n_pos
        p = tp / max(1, tp + fp)
        if yi == 1:
            ap += p * (r - prev_r)
            prev_r = r

    return float(ap)


class RunningBinaryCls:
    def __init__(self, th: float, auc_reservoir: int = 200_000, seed: int = 123):
        self.th = float(th)
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.res = Reservoir(k=int(auc_reservoir), seed=seed)

    def update(self, pred: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None):
        pred = pred.detach().float().view(-1)
        y = y.detach().float().view(-1)

        if mask is not None:
            mask = mask.detach().view(-1).bool()
            if not bool(mask.any()):
                return
            pred = pred[mask]
            y = y[mask]

        if y.numel() == 0:
            return

        yt = y > self.th
        pt = pred > self.th

        self.tp += int((pt & yt).sum().item())
        self.fp += int((pt & (~yt)).sum().item())
        self.fn += int(((~pt) & yt).sum().item())
        self.tn += int(((~pt) & (~yt)).sum().item())

        self.res.add(
            (pred - self.th).cpu().numpy().astype(np.float32),
            yt.cpu().numpy().astype(np.int8),
        )

    def compute(self) -> Dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = tp + fp + fn + tn

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        acc = (tp + tn) / total if total > 0 else float("nan")

        fin_rec = rec if math.isfinite(rec) else 0.0
        fin_spec = spec if math.isfinite(spec) else 0.0
        bal = ((fin_rec + fin_spec) / 2.0) if total > 0 else float("nan")

        f1 = (
            2 * prec * rec / (prec + rec)
            if (math.isfinite(prec) and math.isfinite(rec) and (prec + rec) > 0)
            else float("nan")
        )
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")

        sc, lb = self.res.get()
        return {
            "prec": float(prec),
            "rec": float(rec),
            "f1": float(f1),
            "spec": float(spec),
            "acc": float(acc),
            "bal_acc": float(bal),
            "iou": float(iou),
            "dice": float(dice),
            "auroc": _auc_roc(sc, lb),
            "auprc": _auc_pr(sc, lb),
        }


# -------------------------------------------------------------------------------------------------
# Dynamic group reporting
# -------------------------------------------------------------------------------------------------
def _make_group_names(edges: Sequence[float]) -> List[str]:
    """
    Build metric-safe group names from report bins.

    For legacy Eval15 reporting with edges=(0,5,10,15), we use:
      lt0, 0_5, 5_10, 10_15, ge15

    Important: bins are thresholds. The leading 0 creates an audit class lt0,
    normally empty because valid GEDI RH95 labels are >= 0. The article bins are
    therefore 0_5 / 5_10 / 10_15, with ge15 as an upper-tail audit.

    For other bin sets, this falls back to generic lt/ge names.
    """
    xs = [float(x) for x in edges]
    if len(xs) < 1:
        raise ValueError("edges must contain at least one threshold")

    if len(xs) == 4 and all(abs(a - b) < 1e-6 for a, b in zip(xs, [0.0, 5.0, 10.0, 15.0])):
        return ["lt0", "0_5", "5_10", "10_15", "ge15"]

    if len(xs) == 4 and all(abs(a - b) < 1e-6 for a, b in zip(xs, [2.0, 5.0, 10.0, 15.0])):
        return ["0_2", "2_5", "5_10", "10_15", "ge15"]

    if len(xs) == 3 and all(abs(a - b) < 1e-6 for a, b in zip(xs, [5.0, 10.0, 15.0])):
        return ["0_5", "5_10", "10_15", "ge15"]

    def _fmt(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else str(v).replace(".", "p")

    names = [f"lt{_fmt(xs[0])}"]
    for lo, hi in zip(xs[:-1], xs[1:]):
        names.append(f"{_fmt(lo)}_{_fmt(hi)}")
    names.append(f"ge{_fmt(xs[-1])}")
    return names


def group_id(y: torch.Tensor, edges: Sequence[float]) -> torch.Tensor:
    """
    Dynamic height classes with half-open intervals [lo, hi),
    and boundary -> upper class.

    Example with edges=(3,10,20,30):
      class 0 : y < 3
      class 1 : 3 <= y < 10
      class 2 : 10 <= y < 20
      class 3 : 20 <= y < 30
      class 4 : y >= 30
    """
    xs = tuple(float(e) for e in edges)
    y = y.detach().float()
    g = torch.zeros_like(y, dtype=torch.long)

    for i, thr in enumerate(xs):
        g = torch.where(y >= thr, torch.full_like(g, i + 1), g)

    return g


class PerGroupReport:
    def __init__(self, edges: Sequence[float]):
        self.edges = tuple(float(x) for x in edges)
        self.names = _make_group_names(self.edges)
        self.n_groups = len(self.names)
        self.reg = [RunningReg() for _ in range(self.n_groups)]
        self.conf = np.zeros((self.n_groups, self.n_groups), dtype=np.int64)

    def update(self, pred: torch.Tensor, y: torch.Tensor):
        pred = pred.detach().float().view(-1)
        y = y.detach().float().view(-1)
        if y.numel() == 0:
            return

        tcls = group_id(y, self.edges).view(-1)
        pcls = group_id(pred, self.edges).view(-1)

        for c in range(self.n_groups):
            self.reg[c].update(pred, y, mask=(tcls == c))

        for i in range(self.n_groups):
            ti = tcls == i
            if not bool(ti.any()):
                continue
            for j in range(self.n_groups):
                self.conf[i, j] += int((ti & (pcls == j)).sum().item())

    def compute(self, global_mean: Optional[float] = None) -> Dict[str, float]:
        """
        Group-wise metrics.

        Scientific convention:
        - g_*_r2 uses the *within-group* mean as baseline, i.e. a true subgroup R².
        - g_*_r2_globalref is additionally reported when ``global_mean`` is given;
          it compares the subgroup against the split-level mean baseline and is
          retained only as an auxiliary diagnostic / backward-compatibility aid.
        """
        out: Dict[str, float] = {}

        for c, name in enumerate(self.names):
            m_local = self.reg[c].compute(global_mean=None)
            for k in ("n", "mae", "mse", "rmse", "r2", "bias", "pred_mean", "pred_std", "pred_min", "pred_max", "true_mean", "true_std", "true_min", "true_max", "std_ratio", "slope", "corr"):
                out[f"g_{name}_{k}"] = float(m_local[k])

            if global_mean is not None:
                m_globalref = self.reg[c].compute(global_mean=global_mean)
                out[f"g_{name}_r2_globalref"] = float(m_globalref["r2"])

        total = float(self.conf.sum())
        out[f"g_acc{self.n_groups}"] = float(np.trace(self.conf) / total) if total > 0 else float("nan")

        for i in range(self.n_groups):
            for j in range(self.n_groups):
                out[f"g_cm_{i}{j}"] = float(self.conf[i, j])

        return out


class PerGroupBinaryCls:
    def __init__(
        self,
        ths: Sequence[float],
        report_bins: Sequence[float],
        auc_reservoir: int,
        seed: int,
    ):
        self.ths = tuple(float(t) for t in ths)
        self.report_bins = tuple(float(x) for x in report_bins)
        self.group_names = _make_group_names(self.report_bins)
        self.n_groups = len(self.group_names)

        self.global_cls = {
            t: RunningBinaryCls(t, auc_reservoir=auc_reservoir, seed=seed + int(t * 10))
            for t in self.ths
        }
        self.group_cls = {
            g: {
                t: RunningBinaryCls(t, auc_reservoir=auc_reservoir, seed=seed + 1000 * g + int(t * 10))
                for t in self.ths
            }
            for g in range(self.n_groups)
        }

    def update(self, pred: torch.Tensor, y: torch.Tensor):
        pred = pred.detach().float().view(-1)
        y = y.detach().float().view(-1)
        if y.numel() == 0:
            return

        g = group_id(y, self.report_bins).view(-1)
        for t in self.ths:
            self.global_cls[t].update(pred, y)
            for gi in range(self.n_groups):
                self.group_cls[gi][t].update(pred, y, mask=(g == gi))

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}

        for t in self.ths:
            k = f"cls{int(t)}" if float(t).is_integer() else f"cls{t}"
            g = self.global_cls[t].compute()
            for met in ("prec", "rec", "f1", "spec", "acc", "bal_acc", "iou", "dice", "auroc", "auprc"):
                out[f"{k}_{met}"] = g[met]

            for gi, gname in enumerate(self.group_names):
                m = self.group_cls[gi][t].compute()
                for met in ("prec", "rec", "f1", "spec", "acc", "bal_acc", "iou", "dice", "auroc", "auprc"):
                    out[f"{k}_g_{gname}_{met}"] = m[met]

        return out


# -------------------------------------------------------------------------------------------------
# Prediction map normalization
# -------------------------------------------------------------------------------------------------
def _ensure_pred_map_4d(pred) -> torch.Tensor:
    """
    Accept:
      - Tensor (B,1,H,W)
      - Tensor (B,H,W)
      - Tensor (B,H,W,1)
      - tuple/list -> first element
      - dict with out/pred/logits/yhat
    Return:
      - Tensor (B,1,H,W)
    """
    if isinstance(pred, (tuple, list)):
        if len(pred) == 0:
            raise RuntimeError("Empty tuple/list model output.")
        pred = pred[0]

    if isinstance(pred, dict):
        for k in ("out", "pred", "logits", "yhat"):
            if k in pred:
                pred = pred[k]
                break
        else:
            raise RuntimeError(f"Unsupported dict model output keys: {list(pred.keys())}")

    if not torch.is_tensor(pred):
        raise RuntimeError(f"Unsupported model output type: {type(pred)}")

    if pred.ndim == 4 and pred.shape[1] == 1:
        return pred
    if pred.ndim == 3:
        return pred.unsqueeze(1)
    if pred.ndim == 4 and pred.shape[-1] == 1:
        return pred.permute(0, 3, 1, 2)

    raise RuntimeError(f"Unsupported prediction shape: {tuple(pred.shape)}")


# -------------------------------------------------------------------------------------------------
# Sparse GEDI loss helpers
# -------------------------------------------------------------------------------------------------
def _build_valid_mask(
    aux_rows: torch.Tensor,
    aux_cols: torch.Tensor,
    aux_y: torch.Tensor,
    H: int,
    W: int,
    aux_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    valid = (
        (aux_rows >= 0)
        & (aux_cols >= 0)
        & (aux_rows < H)
        & (aux_cols < W)
        & torch.isfinite(aux_y)
    )
    if aux_mask is not None:
        valid = valid & aux_mask.bool()
    return valid


def _ensure_sparse_map_batch(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Accept:
      - None
      - (B,H,W)
      - (B,1,H,W)
    Return:
      - None or (B,H,W)
    """
    if x is None:
        return None
    if x.ndim == 4 and int(x.shape[1]) == 1:
        return x[:, 0]
    if x.ndim == 3:
        return x
    raise RuntimeError(f"Unsupported sparse-map shape: {tuple(x.shape)}")


def _gather_points_from_aux_batch(
    pred_map: torch.Tensor,          # (B,1,H,W)
    aux_rows: torch.Tensor,          # (B,K)
    aux_cols: torch.Tensor,          # (B,K)
    aux_y: torch.Tensor,             # (B,K)
    aux_mask: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], int, int]:
    """
    Gather all valid GEDI points using legacy aux coordinates.
    """
    B, _, H, W = pred_map.shape

    per_patch_preds: List[torch.Tensor] = []
    per_patch_targets: List[torch.Tensor] = []
    n_pts_total = 0
    n_empty_patches = 0

    for b in range(B):
        valid = _build_valid_mask(
            aux_rows[b],
            aux_cols[b],
            aux_y[b],
            H=H,
            W=W,
            aux_mask=None if aux_mask is None else aux_mask[b],
        )

        if not bool(valid.any()):
            n_empty_patches += 1
            continue

        r = aux_rows[b][valid].long()
        c = aux_cols[b][valid].long()
        t = aux_y[b][valid].float()
        p = pred_map[b, 0, r, c]

        per_patch_preds.append(p)
        per_patch_targets.append(t)
        n_pts_total += int(p.numel())

    return per_patch_preds, per_patch_targets, n_pts_total, n_empty_patches


def _gather_points_from_raster_sparse_batch(
    pred_map: torch.Tensor,          # (B,1,H,W)
    y_sparse: Optional[torch.Tensor],# (B,H,W) or (B,1,H,W)
    y_mask: Optional[torch.Tensor],  # (B,H,W) or (B,1,H,W)
) -> Tuple[List[torch.Tensor], List[torch.Tensor], int, int]:
    """
    Gather all valid GEDI points from pixel-wise rasterized sparse supervision.
    """
    y_sparse = _ensure_sparse_map_batch(y_sparse)
    y_mask = _ensure_sparse_map_batch(y_mask)

    if y_sparse is None or y_mask is None:
        return [], [], 0, int(pred_map.shape[0])

    if tuple(y_sparse.shape) != tuple(y_mask.shape):
        raise RuntimeError(
            f"y_sparse and y_mask shape mismatch: {tuple(y_sparse.shape)} vs {tuple(y_mask.shape)}"
        )

    B = int(pred_map.shape[0])
    per_patch_preds: List[torch.Tensor] = []
    per_patch_targets: List[torch.Tensor] = []
    n_pts_total = 0
    n_empty_patches = 0

    for b in range(B):
        valid = y_mask[b].bool() & torch.isfinite(y_sparse[b])
        if not bool(valid.any()):
            n_empty_patches += 1
            continue

        p = pred_map[b, 0][valid]
        t = y_sparse[b][valid].float()

        per_patch_preds.append(p)
        per_patch_targets.append(t)
        n_pts_total += int(p.numel())

    return per_patch_preds, per_patch_targets, n_pts_total, n_empty_patches

def _make_regression_point_weights(
    targets: torch.Tensor,
    reg_bin_edges: Sequence[float],
    reg_bin_weights: Sequence[float],
) -> torch.Tensor:
    if len(reg_bin_weights) != len(reg_bin_edges) + 1:
        raise ValueError(
            f"reg_bin_weights length must be len(edges)+1. "
            f"Got edges={len(reg_bin_edges)} weights={len(reg_bin_weights)}"
        )

    edges = tuple(float(x) for x in reg_bin_edges)
    weights = tuple(float(x) for x in reg_bin_weights)

    w = torch.full_like(targets, float(weights[0]))

    if len(edges) >= 2:
        for i in range(1, len(edges)):
            lo = float(edges[i - 1])
            hi = float(edges[i])
            w = torch.where(
                (targets >= lo) & (targets < hi),
                torch.full_like(targets, float(weights[i])),
                w,
            )

    if len(edges) >= 1:
        w = torch.where(
            targets >= float(edges[-1]),
            torch.full_like(targets, float(weights[-1])),
            w,
        )

    return w


def _class_centres_from_edges(edges: Sequence[float]) -> torch.Tensor:
    """
    Approximate class centres used for optional auxiliary CE loss.
    Dynamic version for any number of bins.

    For edges=(3,10,20,30):
      centres = [1.5, 6.5, 15.0, 25.0, 35.0]
    """
    xs = [float(x) for x in edges]
    centres: List[float] = []

    # first class
    centres.append(xs[0] / 2.0)

    # middle classes
    for i in range(1, len(xs)):
        centres.append(0.5 * (xs[i - 1] + xs[i]))

    # last class: extend by same gap as last interval
    if len(xs) == 1:
        last_gap = xs[0]
    else:
        last_gap = xs[-1] - xs[-2]
    centres.append(xs[-1] + 0.5 * last_gap)

    return torch.tensor(centres, dtype=torch.float32)



def _resolve_regression_loss_name(criterion: nn.Module) -> str:
    """
    Main regression loss used on sparse GEDI supervision.

    Default is MAE because the current CHM pipeline is closer to a FORMS-like
    sparse GEDI setup than to a super-resolution setup.
    """
    name = getattr(criterion, "regression_loss_name", None)
    if name is None:
        name = getattr(criterion, "reg_loss_name", "mae")

    name = str(name).strip().lower()
    if name in {"l1", "mae", "abs", "absolute"}:
        return "mae"
    if name in {"huber", "smoothl1", "smooth_l1"}:
        return "huber"
    return "mae"


def _huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float,
    reduction: str = "none",
) -> torch.Tensor:
    """Pauls et al. Huber loss with cutoff ``delta``.

    ``smooth_l1_loss(beta=delta)`` reproduces exactly the Pauls formula:
        h_δ(e) = 0.5 * e² / δ   if |e| < δ
               = |e| - 0.5 * δ  otherwise
    """
    delta = float(delta)
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError(f"Huber delta must be finite and > 0, got {delta!r}.")

    abs_error = torch.abs(pred - target)
    loss = torch.where(
        abs_error < delta,
        0.5 * abs_error.square() / delta,
        abs_error - 0.5 * delta,
    )

    reduction = str(reduction).strip().lower()
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported Huber reduction: {reduction!r}.")


def _point_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    regression_loss_name: str,
    huber_beta: float,
) -> torch.Tensor:
    if regression_loss_name == "huber":
        return _huber_loss(pred, target, delta=float(huber_beta), reduction="none")
    return torch.abs(pred - target)


# -------------------------------------------------------------------------------------------------
# SHIFTED HUBER SPATIAL TOLERANCE PATCH v1
# -------------------------------------------------------------------------------------------------
def _shifted_spatial_loss_enabled(criterion: nn.Module) -> bool:
    """Whether the sparse GEDI regression loss should tolerate small XY shifts.

    This is a TRAINING-LOSS option only. Metrics still use the exact prediction at
    the GEDI row/col returned by ``compute_pixelwise_loss``. The goal is to absorb
    small GEDI geolocation / footprint / Sentinel-grid mismatch without changing
    Step04/Step05 labels and without using Tolan as a label.
    """
    mode = str(getattr(criterion, "spatial_loss", "exact")).strip().lower()
    radius = int(getattr(criterion, "shift_radius_px", 0))
    return bool(radius > 0 and mode in {
        "shifted", "shifted_huber", "shifted_mae", "spatial_shifted", "spatial_shifted_huber", "pauls_track_shifted_huber", "track_shifted_huber", "track_shifted_mae", "fixed_track_shift_huber"
    })


def _shifted_spatial_loss_config(criterion: nn.Module) -> Dict[str, Any]:
    mode = str(getattr(criterion, "spatial_loss", "exact")).strip().lower()
    radius = max(0, int(getattr(criterion, "shift_radius_px", 0)))
    reduce_mode = str(getattr(criterion, "shift_mode", "softmin")).strip().lower()
    if reduce_mode not in {"min", "softmin"}:
        reduce_mode = "softmin"
    temperature = float(getattr(criterion, "shift_softmin_temperature", 0.5))
    if not math.isfinite(temperature) or temperature <= 0:
        temperature = 0.5
    min_track_points = int(getattr(criterion, "shift_min_track_points", 10))
    if min_track_points < 1:
        min_track_points = 1
    track_level = mode in {"pauls_track_shifted_huber", "track_shifted_huber", "track_shifted_mae", "fixed_track_shift_huber"}
    pauls_strict = mode == "pauls_track_shifted_huber"
    fixed_shift = mode == "fixed_track_shift_huber"
    if fixed_shift and radius < 1:
        raise ValueError("spatial_loss=fixed_track_shift_huber requires shift_radius_px>=1.")
    if pauls_strict:
        if radius != 1:
            raise ValueError(
                "spatial_loss=pauls_track_shifted_huber requires shift_radius_px=1 "
                "(the 3x3 candidate set corresponding to r=sqrt(2) at 10 m)."
            )
        if reduce_mode != "min":
            raise ValueError(
                "spatial_loss=pauls_track_shifted_huber requires shift_mode='min'. "
                "Use track_shifted_huber for an experimental softmin variant."
            )
        if min_track_points != 10:
            raise ValueError(
                "spatial_loss=pauls_track_shifted_huber requires shift_min_track_points=10, "
                "as specified by Pauls et al."
            )
    return {
        "mode": mode,
        "enabled": _shifted_spatial_loss_enabled(criterion),
        "radius_px": radius,
        "reduce_mode": reduce_mode,
        "temperature": temperature,
        "track_level": bool(track_level),
        "pauls_strict": bool(pauls_strict),
        "fixed_shift": bool(fixed_shift),
        "min_track_points": int(min_track_points),
    }


def _gather_shifted_points_from_aux_batch(
    pred_map: torch.Tensor,          # (B,1,H,W)
    aux_rows: torch.Tensor,          # (B,K)
    aux_cols: torch.Tensor,          # (B,K)
    aux_y: torch.Tensor,             # (B,K)
    aux_mask: Optional[torch.Tensor] = None,
    *,
    radius_px: int = 1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], int, int]:
    """Gather exact predictions + local candidate windows for shifted GEDI loss.

    Returns
    -------
    per_patch_exact_preds:
        exact prediction at the GEDI row/col, used for metrics and auxiliary losses.
    per_patch_targets:
        RH target values.
    per_patch_candidate_preds:
        candidate predictions in a (2r+1)x(2r+1) window around each GEDI point,
        shape (n_points_in_patch, n_candidates). Edge windows use replicate padding.
    n_pts_total, n_empty_patches
        diagnostics.
    """
    pred_map = _ensure_pred_map_4d(pred_map)
    B, _, H, W = pred_map.shape
    radius_px = max(0, int(radius_px))

    per_patch_exact_preds: List[torch.Tensor] = []
    per_patch_targets: List[torch.Tensor] = []
    per_patch_candidate_preds: List[torch.Tensor] = []
    n_pts_total = 0
    n_empty_patches = 0

    for b in range(B):
        valid = _build_valid_mask(
            aux_rows[b],
            aux_cols[b],
            aux_y[b],
            H=H,
            W=W,
            aux_mask=None if aux_mask is None else aux_mask[b],
        )
        if not bool(valid.any()):
            n_empty_patches += 1
            continue

        r = aux_rows[b][valid].long()
        c = aux_cols[b][valid].long()
        t = aux_y[b][valid].float()
        exact = pred_map[b, 0, r, c]

        if radius_px <= 0:
            cand = exact[:, None]
        else:
            # Replicate padding keeps all GEDI points usable even close to patch borders.
            # This is only for the loss candidate window; exact metrics still use r,c.
            padded = F.pad(
                pred_map[b:b + 1, 0:1],
                pad=(radius_px, radius_px, radius_px, radius_px),
                mode="replicate",
            )[0, 0]
            rr = r + radius_px
            cc = c + radius_px
            vals: List[torch.Tensor] = []
            for dr in range(-radius_px, radius_px + 1):
                for dc in range(-radius_px, radius_px + 1):
                    vals.append(padded[rr + dr, cc + dc])
            cand = torch.stack(vals, dim=1)

        per_patch_exact_preds.append(exact)
        per_patch_targets.append(t)
        per_patch_candidate_preds.append(cand)
        n_pts_total += int(t.numel())

    return per_patch_exact_preds, per_patch_targets, per_patch_candidate_preds, n_pts_total, n_empty_patches


def _reduce_shifted_candidate_losses(
    pred_candidates: torch.Tensor,   # (N,C)
    target: torch.Tensor,            # (N,)
    *,
    regression_loss_name: str,
    huber_beta: float,
    reduce_mode: str,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert candidate predictions into one per-point loss and effective pred.

    ``min`` picks the candidate with the smallest robust loss.
    ``softmin`` uses a differentiable soft selection over candidate losses.
    """
    if pred_candidates.ndim != 2:
        raise RuntimeError(f"pred_candidates must be 2D (N,C), got {tuple(pred_candidates.shape)}")
    if target.ndim != 1:
        target = target.reshape(-1)
    if int(pred_candidates.shape[0]) != int(target.shape[0]):
        raise RuntimeError(
            f"pred_candidates/target mismatch: {tuple(pred_candidates.shape)} vs {tuple(target.shape)}"
        )

    target2 = target[:, None].expand_as(pred_candidates)
    cand_losses = _point_regression_loss(
        pred_candidates,
        target2,
        regression_loss_name=regression_loss_name,
        huber_beta=huber_beta,
    )

    reduce_mode = str(reduce_mode).strip().lower()
    if reduce_mode == "min":
        point_loss, idx = torch.min(cand_losses, dim=1)
        eff_pred = pred_candidates.gather(1, idx[:, None]).squeeze(1)
        return point_loss, eff_pred

    temp = max(float(temperature), 1e-6)
    weights = torch.softmax(-cand_losses / temp, dim=1)
    point_loss = (weights * cand_losses).sum(dim=1)
    eff_pred = (weights * pred_candidates).sum(dim=1)
    return point_loss, eff_pred


# -------------------------------------------------------------------------------------------------
# Fixed train-calibrated track-shift mode
# -------------------------------------------------------------------------------------------------
# ``pauls_track_shifted_huber`` remains the dynamic Pauls-like objective: the
# best shift is selected inside the current input instance/sample.  The mode below
# is an explicit extension for a second-stage experiment where a train-calibrated
# CSV table is frozen before training/evaluation.

_FIXED_TRACK_SHIFT_CACHE: Dict[str, Dict[int, Tuple[int, int]]] = {}
_FIXED_TRACK_SHIFT_TABLE_PRINTED = False


def _fixed_track_shift_csv_path(criterion: nn.Module) -> str:
    raw = getattr(criterion, "fixed_track_shift_csv", None)
    if raw is None or str(raw).strip() == "":
        raw = os.environ.get("CHM_FIXED_TRACK_SHIFT_CSV", "")
    return str(raw).strip().strip('"').strip("'")


def _fixed_track_shift_strict_missing() -> bool:
    return _env_flag("CHM_FIXED_TRACK_SHIFT_STRICT", default=False)


def _load_fixed_track_shift_table(path_raw: str) -> Dict[int, Tuple[int, int]]:
    path_raw = str(path_raw or "").strip()
    if not path_raw:
        raise RuntimeError(
            "spatial_loss=fixed_track_shift_huber requires CHM_FIXED_TRACK_SHIFT_CSV "
            "or criterion.fixed_track_shift_csv pointing to a CSV table with "
            "aux_track_id,shift_dr_px,shift_dc_px."
        )

    path = Path(path_raw).expanduser()
    cache_key = str(path.resolve()) if path.exists() else str(path)
    if cache_key in _FIXED_TRACK_SHIFT_CACHE:
        return _FIXED_TRACK_SHIFT_CACHE[cache_key]
    if not path.exists():
        raise FileNotFoundError(f"Fixed track-shift CSV not found: {path}")

    table: Dict[int, Tuple[int, int]] = {}

    def _first_present(row: Dict[str, Any], names: Sequence[str]) -> Any:
        lower = {str(k).strip().lower(): v for k, v in row.items()}
        for name in names:
            if name.lower() in lower and str(lower[name.lower()]).strip() != "":
                return lower[name.lower()]
        return None

    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise RuntimeError(f"Fixed track-shift CSV has no header: {path}")
        for row in reader:
            tid_raw = _first_present(row, ["aux_track_id", "track_id", "gedi_track_id"])
            dr_raw = _first_present(row, ["shift_dr_px", "dr_px", "dr", "row_shift_px"])
            dc_raw = _first_present(row, ["shift_dc_px", "dc_px", "dc", "col_shift_px"])
            if tid_raw is None or dr_raw is None or dc_raw is None:
                continue
            try:
                tid = int(round(float(str(tid_raw).strip())))
                dr = int(round(float(str(dr_raw).strip())))
                dc = int(round(float(str(dc_raw).strip())))
            except Exception:
                continue
            table[tid] = (dr, dc)

    if not table:
        raise RuntimeError(
            f"Fixed track-shift CSV contains no usable rows: {path}. "
            "Expected columns aux_track_id,shift_dr_px,shift_dc_px."
        )
    _FIXED_TRACK_SHIFT_CACHE[cache_key] = table
    return table


def _fixed_shift_selected_indices_for_points(
    *,
    orig_track_ids: torch.Tensor,
    radius_px: int,
    criterion: nn.Module,
    context: Optional[Dict[str, Any]],
) -> Tuple[torch.Tensor, int, int, int]:
    """Return candidate indices from a frozen aux_track_id -> (dr, dc) table.

    If training uses geometric flips, the geographic shift vector must be
    transformed into the augmented image coordinates: horizontal flip changes the
    sign of dc, vertical flip changes the sign of dr.  This function supports
    that transformation only when the caller explicitly passes flip flags in the
    context.
    """
    device = orig_track_ids.device
    offsets = _candidate_offsets(radius_px)
    center_idx = int(len(offsets) // 2)
    offset_to_idx = {(int(dr), int(dc)): i for i, (dr, dc) in enumerate(offsets)}
    selected = torch.full(tuple(orig_track_ids.reshape(-1).shape), center_idx, device=device, dtype=torch.long)

    table = _load_fixed_track_shift_table(_fixed_track_shift_csv_path(criterion))
    strict = _fixed_track_shift_strict_missing()
    ctx = context or {}
    flip_h = bool(ctx.get("flip_h", False))
    flip_v = bool(ctx.get("flip_v", False))

    found = 0
    missing = 0
    out_of_radius = 0
    tids = orig_track_ids.detach().reshape(-1).cpu().numpy().astype(np.int64, copy=False)
    idx_values: List[int] = []
    missing_examples: List[int] = []
    oor_examples: List[Tuple[int, int, int]] = []

    for tid_np in tids:
        tid = int(tid_np)
        if tid not in table:
            missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(tid)
            idx_values.append(center_idx)
            continue
        dr, dc = table[tid]
        if flip_h:
            dc = -int(dc)
        if flip_v:
            dr = -int(dr)
        idx = offset_to_idx.get((int(dr), int(dc)))
        if idx is None:
            out_of_radius += 1
            if len(oor_examples) < 5:
                oor_examples.append((tid, int(dr), int(dc)))
            idx_values.append(center_idx)
            continue
        found += 1
        idx_values.append(int(idx))

    if strict and (missing > 0 or out_of_radius > 0):
        raise RuntimeError(
            "fixed_track_shift_huber could not apply the frozen table to all points: "
            f"missing={missing}, out_of_radius={out_of_radius}, "
            f"missing_examples={missing_examples}, out_of_radius_examples={oor_examples}."
        )

    selected = torch.as_tensor(idx_values, device=device, dtype=torch.long)
    return selected, int(found), int(missing), int(out_of_radius)


def _reduce_fixed_track_shift_candidate_losses(
    pred_candidates: torch.Tensor,   # (N,C)
    target: torch.Tensor,            # (N,)
    orig_track_ids: torch.Tensor,    # (N,), physical aux_track_id used for CSV lookup
    *,
    group_track_ids: Optional[torch.Tensor] = None,  # optional sample-local groups for diagnostics
    regression_loss_name: str,
    huber_beta: float,
    radius_px: int,
    criterion: nn.Module,
    context: Optional[Dict[str, Any]] = None,
    return_details: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply a frozen train-calibrated track shift table.

    This is intentionally not called ``pauls_track_shifted_huber``.  It is a
    separate extension: the shift is looked up from a fixed CSV table instead of
    being re-estimated by an argmin inside every forward pass.
    """
    if pred_candidates.ndim != 2:
        raise RuntimeError(f"pred_candidates must be 2D (N,C), got {tuple(pred_candidates.shape)}")
    target = target.reshape(-1)
    orig_track_ids = orig_track_ids.reshape(-1).long()
    if int(pred_candidates.shape[0]) != int(target.shape[0]) or int(orig_track_ids.shape[0]) != int(target.shape[0]):
        raise RuntimeError(
            f"pred_candidates/target/orig_track_ids mismatch: {tuple(pred_candidates.shape)} vs "
            f"{tuple(target.shape)} vs {tuple(orig_track_ids.shape)}"
        )

    target2 = target[:, None].expand_as(pred_candidates)
    cand_losses = _point_regression_loss(
        pred_candidates,
        target2,
        regression_loss_name=regression_loss_name,
        huber_beta=huber_beta,
    )

    n, c = int(cand_losses.shape[0]), int(cand_losses.shape[1])
    center_idx = c // 2
    selected_idx, found, missing, out_of_radius = _fixed_shift_selected_indices_for_points(
        orig_track_ids=orig_track_ids,
        radius_px=int(radius_px),
        criterion=criterion,
        context=context,
    )
    selected_idx = selected_idx.to(device=pred_candidates.device).clamp(0, c - 1)

    out_loss = cand_losses.gather(1, selected_idx[:, None]).squeeze(1)
    out_pred = pred_candidates.gather(1, selected_idx[:, None]).squeeze(1)

    global _FIXED_TRACK_SHIFT_TABLE_PRINTED
    if not _FIXED_TRACK_SHIFT_TABLE_PRINTED:
        print(
            f"[FIXED-TRACK-SHIFT] enabled | csv={_fixed_track_shift_csv_path(criterion)} | "
            f"radius_px={int(radius_px)} | found_points={found} | missing_points={missing} | "
            f"out_of_radius_points={out_of_radius} | strict_missing={int(_fixed_track_shift_strict_missing())}",
            flush=True,
        )
        _FIXED_TRACK_SHIFT_TABLE_PRINTED = True

    if not return_details:
        return out_loss, out_pred

    track_n_points = torch.ones((n,), device=pred_candidates.device, dtype=torch.long)
    track_loss_exact = cand_losses[:, center_idx].detach().clone()
    track_loss_selected = out_loss.detach().clone()

    group_ids = group_track_ids.reshape(-1).long() if group_track_ids is not None else orig_track_ids
    for gid in torch.unique(group_ids.detach()):
        m = group_ids == gid
        n_group = int(m.sum().item())
        if n_group <= 0:
            continue
        track_n_points[m] = n_group
        track_loss_exact[m] = cand_losses[m, center_idx].mean().detach()
        track_loss_selected[m] = out_loss[m].mean().detach()

    details = {
        "selected_idx": selected_idx.detach(),
        "center_idx": torch.as_tensor(int(center_idx), device=pred_candidates.device, dtype=torch.long),
        "track_n_points": track_n_points.detach(),
        "track_loss_exact": track_loss_exact.detach(),
        "track_loss_selected": track_loss_selected.detach(),
        "track_loss_gain": (track_loss_exact - track_loss_selected).detach(),
        "was_shifted": (selected_idx != int(center_idx)).detach().to(torch.long),
        "n_candidates": torch.as_tensor(int(c), device=pred_candidates.device, dtype=torch.long),
        "fixed_shift_found": torch.as_tensor(int(found), device=pred_candidates.device, dtype=torch.long),
        "fixed_shift_missing": torch.as_tensor(int(missing), device=pred_candidates.device, dtype=torch.long),
        "fixed_shift_out_of_radius": torch.as_tensor(int(out_of_radius), device=pred_candidates.device, dtype=torch.long),
    }
    return out_loss, out_pred, details



def _gather_track_shifted_points_from_aux_batch(
    pred_map: torch.Tensor,          # (B,1,H,W)
    aux_rows: torch.Tensor,          # (B,K)
    aux_cols: torch.Tensor,          # (B,K)
    aux_y: torch.Tensor,             # (B,K)
    aux_track_id: torch.Tensor,      # (B,K), int64
    aux_mask: Optional[torch.Tensor] = None,
    *,
    radius_px: int = 1,
) -> Tuple[
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor],
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor],
    int, int,
]:
    """Gather exact predictions, candidate windows, track groups and export metadata.

    The returned ``track_group_ids`` are unique within the current batch.  They
    deliberately represent ``(sample, aux_track_id)`` groups, not global physical
    tracks across temporal samples, because each sample has its own prediction map.

    Additional returned tensors preserve the original ``aux_track_id``, GEDI
    row/col, batch index and point index so that the selected track shift can be
    exported later with shot-level metadata for QGIS.
    """
    pred_map = _ensure_pred_map_4d(pred_map)
    B, _, H, W = pred_map.shape
    radius_px = max(0, int(radius_px))

    per_patch_exact_preds: List[torch.Tensor] = []
    per_patch_targets: List[torch.Tensor] = []
    per_patch_candidate_preds: List[torch.Tensor] = []
    per_patch_track_group_ids: List[torch.Tensor] = []
    per_patch_original_track_ids: List[torch.Tensor] = []
    per_patch_rows: List[torch.Tensor] = []
    per_patch_cols: List[torch.Tensor] = []
    per_patch_batch_indices: List[torch.Tensor] = []
    per_patch_point_indices: List[torch.Tensor] = []
    n_pts_total = 0
    n_empty_patches = 0
    next_group_id = 0

    if aux_track_id is None:
        raise RuntimeError("Pauls track-level shifted loss requires aux_track_id in the batch metadata.")
    if tuple(aux_track_id.shape) != tuple(aux_y.shape):
        try:
            aux_track_id = aux_track_id.reshape(tuple(aux_y.shape))
        except Exception as exc:
            raise RuntimeError(f"aux_track_id shape mismatch: {tuple(aux_track_id.shape)} vs aux_y {tuple(aux_y.shape)}") from exc

    for b in range(B):
        valid = _build_valid_mask(
            aux_rows[b], aux_cols[b], aux_y[b], H=H, W=W,
            aux_mask=None if aux_mask is None else aux_mask[b],
        )
        missing_track = valid & (aux_track_id[b].long() < 0)
        if bool(missing_track.any()):
            n_missing = int(missing_track.sum().item())
            n_valid = int(valid.sum().item())
            raise RuntimeError(
                "Pauls track-level shifted Huber requires a valid aux_track_id for every "
                f"valid GEDI point. Batch sample {b} has {n_missing}/{n_valid} missing IDs. "
                "Rebuild Step04/Step05 shards with aux_track_id; points must never be "
                "silently dropped from the loss."
            )
        if not bool(valid.any()):
            n_empty_patches += 1
            continue

        valid_idx = torch.where(valid)[0].long()
        r = aux_rows[b][valid].long()
        c = aux_cols[b][valid].long()
        t = aux_y[b][valid].float()
        tid = aux_track_id[b][valid].long()
        # Select the shift independently for every (input sample, GEDI track).
        # The same physical track may be reused in several temporal samples, but
        # their separate prediction maps must never be merged into one group.
        _, group_inverse = torch.unique(tid, sorted=True, return_inverse=True)
        group_tid = group_inverse.long() + int(next_group_id)
        next_group_id += int(group_inverse.max().item()) + 1
        exact = pred_map[b, 0, r, c]

        if radius_px <= 0:
            cand = exact[:, None]
        else:
            # Candidate shifts are evaluated on the prediction map. The same candidate
            # column corresponds to the same dx/dy for all points of a track.
            padded = F.pad(
                pred_map[b:b + 1, 0:1],
                pad=(radius_px, radius_px, radius_px, radius_px),
                mode="replicate",
            )[0, 0]
            rr = r + radius_px
            cc = c + radius_px
            vals: List[torch.Tensor] = []
            # radius_px=1 with this square enumeration gives the 3x3 set. This is
            # equivalent to r=sqrt(2) in Pauls et al. at 10 m resolution.
            for dr in range(-radius_px, radius_px + 1):
                for dc in range(-radius_px, radius_px + 1):
                    vals.append(padded[rr + dr, cc + dc])
            cand = torch.stack(vals, dim=1)

        per_patch_exact_preds.append(exact)
        per_patch_targets.append(t)
        per_patch_candidate_preds.append(cand)
        per_patch_track_group_ids.append(group_tid)
        per_patch_original_track_ids.append(tid)
        per_patch_rows.append(r)
        per_patch_cols.append(c)
        per_patch_batch_indices.append(torch.full_like(tid, int(b), dtype=torch.long))
        per_patch_point_indices.append(valid_idx)
        n_pts_total += int(t.numel())

    return (
        per_patch_exact_preds,
        per_patch_targets,
        per_patch_candidate_preds,
        per_patch_track_group_ids,
        per_patch_original_track_ids,
        per_patch_rows,
        per_patch_cols,
        per_patch_batch_indices,
        per_patch_point_indices,
        n_pts_total,
        n_empty_patches,
    )


def _reduce_track_shifted_candidate_losses(
    pred_candidates: torch.Tensor,   # (N,C)
    target: torch.Tensor,            # (N,)
    track_ids: torch.Tensor,         # (N,)
    *,
    regression_loss_name: str,
    huber_beta: float,
    reduce_mode: str,
    temperature: float,
    min_track_points: int = 10,
    return_details: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Pauls-style track-level shifted loss.

    For each GEDI track, compute the average candidate loss for every allowed shift,
    choose the best shift for the whole track, then assign the corresponding point
    losses/effective predictions back to all points of that track.

    Tracks with fewer than ``min_track_points`` are kept exact, following Pauls et al.
    When ``return_details=True``, the selected candidate index and track-level losses
    are returned for optional CSV/QGIS export.  Those details are detached from the
    computation graph and never affect optimization.
    """
    if pred_candidates.ndim != 2:
        raise RuntimeError(f"pred_candidates must be 2D (N,C), got {tuple(pred_candidates.shape)}")
    target = target.reshape(-1)
    track_ids = track_ids.reshape(-1).long()
    if int(pred_candidates.shape[0]) != int(target.shape[0]) or int(track_ids.shape[0]) != int(target.shape[0]):
        raise RuntimeError(
            f"pred_candidates/target/track_ids mismatch: {tuple(pred_candidates.shape)} vs {tuple(target.shape)} vs {tuple(track_ids.shape)}"
        )

    target2 = target[:, None].expand_as(pred_candidates)
    cand_losses = _point_regression_loss(
        pred_candidates,
        target2,
        regression_loss_name=regression_loss_name,
        huber_beta=huber_beta,
    )

    n, c = int(cand_losses.shape[0]), int(cand_losses.shape[1])
    # The centre candidate is the exact GEDI pixel because shifts are enumerated row-major.
    center_idx = c // 2
    out_loss = cand_losses[:, center_idx].clone()
    out_pred = pred_candidates[:, center_idx].clone()

    selected_idx = torch.full((n,), int(center_idx), device=pred_candidates.device, dtype=torch.long)
    track_n_points = torch.ones((n,), device=pred_candidates.device, dtype=torch.long)
    track_loss_exact = cand_losses[:, center_idx].detach().clone()
    track_loss_selected = cand_losses[:, center_idx].detach().clone()

    min_track_points = max(1, int(min_track_points))
    reduce_mode = str(reduce_mode).strip().lower()
    temp = max(float(temperature), 1e-6)

    # unique() is differentiability-safe because it only groups indices, not values.
    for tid in torch.unique(track_ids.detach()):
        tid_val = int(tid.item())
        if tid_val < 0:
            continue
        m = track_ids == tid
        n_track = int(m.sum().item())
        losses_t = cand_losses[m]        # (n_track, C)
        preds_t = pred_candidates[m]     # (n_track, C)
        track_loss = losses_t.mean(dim=0)
        exact_mean = track_loss[center_idx].detach()

        track_n_points[m] = int(n_track)
        track_loss_exact[m] = exact_mean

        if n_track < min_track_points:
            # Pauls et al.: do not shift tracks with fewer than ten measurements.
            track_loss_selected[m] = exact_mean
            continue

        if reduce_mode == "min":
            idx = int(torch.argmin(track_loss).item())
            out_loss[m] = losses_t[:, idx]
            out_pred[m] = preds_t[:, idx]
            selected_idx[m] = int(idx)
            track_loss_selected[m] = track_loss[idx].detach()
        else:
            # Experimental differentiable variant.  For export we record the argmin
            # as a diagnostic candidate index, while the optimized prediction uses
            # the softmin weighted combination below.
            w = torch.softmax(-track_loss / temp, dim=0)  # (C,)
            idx = int(torch.argmin(track_loss).item())
            out_loss[m] = (losses_t * w[None, :]).sum(dim=1)
            out_pred[m] = (preds_t * w[None, :]).sum(dim=1)
            selected_idx[m] = int(idx)
            track_loss_selected[m] = (track_loss * w).sum().detach()

    if not return_details:
        return out_loss, out_pred

    details = {
        "selected_idx": selected_idx.detach(),
        "center_idx": torch.as_tensor(int(center_idx), device=pred_candidates.device, dtype=torch.long),
        "track_n_points": track_n_points.detach(),
        "track_loss_exact": track_loss_exact.detach(),
        "track_loss_selected": track_loss_selected.detach(),
        "track_loss_gain": (track_loss_exact - track_loss_selected).detach(),
        "was_shifted": (selected_idx != int(center_idx)).detach().to(torch.long),
        "n_candidates": torch.as_tensor(int(c), device=pred_candidates.device, dtype=torch.long),
    }
    return out_loss, out_pred, details


def _weighted_loss_from_point_losses(
    point_losses: torch.Tensor,
    target: torch.Tensor,
    *,
    bin_edges: Sequence[float],
    class_weights: Optional[Sequence[float]],
) -> torch.Tensor:
    if class_weights is None or len(tuple(bin_edges)) < 1:
        return point_losses.mean()
    weights = _make_regression_point_weights(target, bin_edges, class_weights).to(
        device=point_losses.device,
        dtype=point_losses.dtype,
    )
    return (weights * point_losses).sum() / weights.sum().clamp_min(1e-8)


def _equal_bin_loss_from_point_losses(
    point_losses: torch.Tensor,
    target: torch.Tensor,
    *,
    bin_edges: Sequence[float],
    class_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    target_cls = group_id(target, bin_edges).long()
    n_classes = int(len(tuple(float(x) for x in bin_edges)) + 1)
    if class_weights is not None:
        if len(class_weights) != n_classes:
            raise ValueError(
                f"class_weights must have len(edges)+1 values. Got {len(class_weights)} weights for {n_classes} classes."
            )
        weights = [float(w) for w in class_weights]
    else:
        weights = [1.0] * n_classes

    losses = []
    wts = []
    for cls_idx in range(n_classes):
        m = target_cls == cls_idx
        if not bool(m.any()):
            continue
        losses.append(point_losses[m].mean())
        wts.append(float(weights[cls_idx]))
    if not losses:
        return point_losses.sum() * 0.0
    losses_t = torch.stack(losses)
    weights_t = torch.tensor(wts, device=losses_t.device, dtype=losses_t.dtype)
    return (losses_t * weights_t).sum() / weights_t.sum().clamp_min(1e-8)


def _asymmetric_weighted_loss_from_point_losses(
    point_losses: torch.Tensor,
    effective_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    criterion: nn.Module,
    phase: int,
    bin_edges: Sequence[float],
    class_weights: Optional[Sequence[float]],
) -> torch.Tensor:
    weights = torch.ones_like(point_losses)

    if class_weights is not None and len(tuple(bin_edges)) >= 1:
        weights = weights * _make_regression_point_weights(target, bin_edges, class_weights).to(
            device=point_losses.device,
            dtype=point_losses.dtype,
        )

    low_thr = _asym_value(criterion, phase, "asym_low_threshold", 5.0)
    high_thr = _asym_value(criterion, phase, "asym_high_threshold", 20.0)
    very_high_thr = _asym_value(criterion, phase, "asym_very_high_threshold", 30.0)

    low_over_w = _asym_value(criterion, phase, "asym_low_over_weight", 1.0)
    high_under_w = _asym_value(criterion, phase, "asym_high_under_weight", 1.0)
    very_high_under_w = _asym_value(criterion, phase, "asym_very_high_under_weight", 1.0)

    low_over = (target < low_thr) & (effective_pred > target)
    high_under = (target > high_thr) & (effective_pred < target)
    very_high_under = (target > very_high_thr) & (effective_pred < target)

    weights = torch.where(low_over, weights * float(low_over_w), weights)
    weights = torch.where(high_under, weights * float(high_under_w), weights)
    weights = torch.where(very_high_under, weights * float(very_high_under_w), weights)

    return (weights * point_losses).sum() / weights.sum().clamp_min(1e-8)



def _normalize_rebalance_strategy(strategy: Any, *, default: str = "plain_mae") -> str:
    """Normalize all loss-ablation strategy names to stable internal names."""
    if strategy is None:
        strategy = default
    s = str(strategy).strip().lower()
    if s in {"", "none", "off", "false", "0", "plain", "mae", "plain_mae"}:
        return "plain_mae"
    if s in {"auto", "adaptive", "data_driven", "dynamic", "auto_bins"}:
        return "auto_quantile"
    if s in {"weighted", "weighted_mae", "class_weighted", "class_weighted_mae", "equal_bin", "equal_bin_mae", "balanced_mae"}:
        return "weighted_mae"
    if s in {"asym", "asymmetric", "asymmetric_mae", "asymmetric_weighted_mae", "anti_shrink_asym", "anti_shrinkage_asym"}:
        return "asymmetric_weighted_mae"
    if s in {"forms", "forms_fixed", "fixed", "quantile", "quantiles", "auto_quantile", "phase1_quantile"}:
        return s
    return s


def _resolve_phase_rebalance_strategy(criterion: nn.Module, phase: int) -> str:
    """
    Resolve the main sparse-regression loss strategy for each phase.

    Backward-compatible behavior:
      - phase 1 defaults to plain MAE;
      - phase 2 keeps the previous phase2_rebalance_strategy behavior.

    New ablation behavior:
      - phase1_rebalance_strategy can activate weighted/asymmetric MAE from phase 1;
      - phase-specific slope/std/bias penalties are handled separately.
    """
    ph = int(phase)
    if ph < 2:
        strategy = getattr(criterion, "phase1_rebalance_strategy", "plain_mae")
        return _normalize_rebalance_strategy(strategy, default="plain_mae")

    strategy = getattr(criterion, "phase2_rebalance_strategy", None)
    if strategy is None:
        strategy = getattr(criterion, "rebalance_strategy", "auto_quantile")
    return _normalize_rebalance_strategy(strategy, default="auto_quantile")


def _is_zero_like_value(value: Any, *, atol: float = 1e-12) -> bool:
    """Return True for numeric values that are effectively zero."""
    try:
        x = float(value)
        return math.isfinite(x) and abs(x) <= float(atol)
    except Exception:
        return False


def _is_one_like_value(value: Any, *, atol: float = 1e-12) -> bool:
    """Return True for numeric values that are effectively one."""
    try:
        x = float(value)
        return math.isfinite(x) and abs(x - 1.0) <= float(atol)
    except Exception:
        return False


def _validate_pauls_strict_loss_contract(
    *,
    criterion: nn.Module,
    phase: int,
    regression_loss_name: str,
    huber_beta: float,
    tv_weight: float,
    shift_cfg: Dict[str, Any],
) -> None:
    """Hard validation for the Pauls et al. shifted-Huber training loss.

    The strict mode is intentionally conservative: it allows only the track-level
    shifted Huber term used by Pauls et al. Any auxiliary ordinal/classification,
    total-variation, class-rebalanced, asymmetric, anti-shrinkage, or anti-zero
    term would change the optimized objective and is therefore rejected.
    """
    fixed_shift_active = bool(shift_cfg.get("fixed_shift", False))
    if fixed_shift_active:
        if regression_loss_name != "huber":
            raise ValueError("spatial_loss=fixed_track_shift_huber requires regression_loss='huber'.")
        if not math.isclose(float(huber_beta), 3.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "spatial_loss=fixed_track_shift_huber requires huber_beta=3.0 m, "
                f"got {huber_beta!r}."
            )
        if not _is_one_like_value(getattr(criterion, "alpha_reg", 1.0)):
            raise ValueError("fixed_track_shift_huber requires alpha_reg=1.0.")
        if not _is_zero_like_value(getattr(criterion, "beta_ord", 0.0)):
            raise ValueError("fixed_track_shift_huber requires beta_ord=0.")
        if not _is_zero_like_value(getattr(criterion, "gamma_cls", 0.0)):
            raise ValueError("fixed_track_shift_huber requires gamma_cls=0.")
        if not _is_zero_like_value(tv_weight):
            raise ValueError("fixed_track_shift_huber requires tv_weight=0.")
        strategy = _resolve_phase_rebalance_strategy(criterion, phase)
        if strategy != "plain_mae":
            raise ValueError(
                "fixed_track_shift_huber requires phase1/phase2 rebalance strategy 'plain_mae' "
                f"for the active phase, got {strategy!r}."
            )
        ph = 1 if int(phase) < 2 else 2
        for name in [
            f"phase{ph}_lambda_slope_loss",
            f"phase{ph}_lambda_std_loss",
            f"phase{ph}_lambda_bias_loss",
            f"phase{ph}_lambda_anti_zero_loss",
        ]:
            if not _is_zero_like_value(getattr(criterion, name, 0.0)):
                raise ValueError(f"fixed_track_shift_huber requires {name}=0.")
        weights_attr = f"phase{ph}_class_weights"
        weights = getattr(criterion, weights_attr, None)
        if weights is not None and len(tuple(weights)) > 0:
            raise ValueError(f"fixed_track_shift_huber requires {weights_attr}=None or empty.")
        weights_mode = getattr(criterion, "weights_mode", None)
        if weights_mode is not None:
            wm = str(weights_mode).strip().lower()
            if wm not in {"", "none", "off", "false", "0", "plain"}:
                raise ValueError("fixed_track_shift_huber requires weights_mode='none'.")
        _load_fixed_track_shift_table(_fixed_track_shift_csv_path(criterion))
        return

    if not bool(shift_cfg.get("pauls_strict", False)):
        return

    if regression_loss_name != "huber":
        raise ValueError("spatial_loss=pauls_track_shifted_huber requires regression_loss='huber'.")
    if not math.isclose(float(huber_beta), 3.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "spatial_loss=pauls_track_shifted_huber requires huber_beta=3.0 m, "
            f"got {huber_beta!r}."
        )

    if not _is_one_like_value(getattr(criterion, "alpha_reg", 1.0)):
        raise ValueError("Pauls strict mode requires alpha_reg=1.0.")
    if not _is_zero_like_value(getattr(criterion, "beta_ord", 0.0)):
        raise ValueError("Pauls strict mode requires beta_ord=0.")
    if not _is_zero_like_value(getattr(criterion, "gamma_cls", 0.0)):
        raise ValueError("Pauls strict mode requires gamma_cls=0.")
    if not _is_zero_like_value(tv_weight):
        raise ValueError("Pauls strict mode requires tv_weight=0.")

    strategy = _resolve_phase_rebalance_strategy(criterion, phase)
    if strategy != "plain_mae":
        raise ValueError(
            "Pauls strict mode requires phase1/phase2 rebalance strategy 'plain_mae' "
            f"for the active phase, got {strategy!r}."
        )

    ph = 1 if int(phase) < 2 else 2
    forbidden_nonzero = [
        f"phase{ph}_lambda_slope_loss",
        f"phase{ph}_lambda_std_loss",
        f"phase{ph}_lambda_bias_loss",
        f"phase{ph}_lambda_anti_zero_loss",
    ]
    for name in forbidden_nonzero:
        if not _is_zero_like_value(getattr(criterion, name, 0.0)):
            raise ValueError(f"Pauls strict mode requires {name}=0.")

    weights_attr = f"phase{ph}_class_weights"
    weights = getattr(criterion, weights_attr, None)
    if weights is not None and len(tuple(weights)) > 0:
        raise ValueError(f"Pauls strict mode requires {weights_attr}=None or empty.")

    weights_mode = getattr(criterion, "weights_mode", None)
    if weights_mode is not None:
        wm = str(weights_mode).strip().lower()
        if wm not in {"", "none", "off", "false", "0", "plain"}:
            raise ValueError("Pauls strict mode requires weights_mode='none'.")



def _resolve_phase2_rebalance_strategy(criterion: nn.Module, phase: int) -> str:
    """Backward-compatible alias kept for older callers."""
    if int(phase) < 2:
        return "plain_mae"
    return _resolve_phase_rebalance_strategy(criterion, phase)


def _clean_strictly_increasing_edges(
    edges: Sequence[float],
    *,
    min_gap: float,
) -> Tuple[float, ...]:
    clean = []
    prev = None
    for e in edges:
        ef = float(e)
        if not math.isfinite(ef):
            continue
        if prev is None or (ef - prev) >= float(min_gap):
            clean.append(ef)
            prev = ef
    return tuple(clean)


def _infer_auto_phase2_bin_edges(
    targets: torch.Tensor,
    *,
    desired_n_classes: int,
    min_points_per_class: int,
    min_bin_width: float,
) -> Tuple[float, ...]:
    """
    Infer balanced phase-2 bins directly from TARGET heights.

    Design goals:
      - adapt to the actual support of the dataset (e.g. lower-height forests)
      - avoid prediction-driven bins
      - keep bins stable enough for phase-2 fine-tuning
      - degrade gracefully when the target distribution is narrow
    """
    vals = targets.detach().float()
    vals = vals[torch.isfinite(vals)]

    if vals.numel() < 2:
        return tuple()

    vmin = float(vals.min().item())
    vmax = float(vals.max().item())
    if (not math.isfinite(vmin)) or (not math.isfinite(vmax)) or (vmax - vmin) < float(min_bin_width):
        return tuple()

    desired_n_classes = max(2, int(desired_n_classes))
    min_points_per_class = max(1, int(min_points_per_class))
    min_bin_width = max(1e-4, float(min_bin_width))

    max_classes_by_points = max(1, int(vals.numel() // min_points_per_class))
    start_n_classes = min(desired_n_classes, max(2, max_classes_by_points))

    for n_classes in range(start_n_classes, 1, -1):
        qs = torch.linspace(
            0.0,
            1.0,
            steps=n_classes + 1,
            device=vals.device,
            dtype=vals.dtype,
        )[1:-1]
        if qs.numel() == 0:
            continue

        q_edges = torch.quantile(vals, qs).detach().cpu().tolist()
        edges = _clean_strictly_increasing_edges(q_edges, min_gap=min_bin_width)
        if len(edges) == (n_classes - 1):
            return edges

    # Fallback: at least split the support into two meaningful groups.
    q50 = torch.quantile(vals, torch.tensor([0.5], device=vals.device, dtype=vals.dtype))
    edges = _clean_strictly_increasing_edges(q50.detach().cpu().tolist(), min_gap=min_bin_width)
    if len(edges) >= 1:
        return edges

    mid = 0.5 * (vmin + vmax)
    edges = _clean_strictly_increasing_edges([mid], min_gap=min_bin_width)
    return edges


def _get_phase2_bin_edges(
    criterion: nn.Module,
    targets: torch.Tensor,
    phase: int,
) -> Tuple[float, ...]:
    """
    Resolve the bin edges used for the phase-2 equal-bin MAE.

    Priority order:
      1) explicit manual edges provided by the config
      2) precomputed train-set edges injected by the launcher / experiment script
      3) cached auto-detected edges for stable phase-2 fine-tuning
      4) on-the-fly target-based automatic edges
    """
    explicit = getattr(criterion, "phase2_bin_edges", None)
    if explicit is not None:
        return tuple(float(x) for x in explicit)

    precomputed = getattr(criterion, "phase2_train_bin_edges", None)
    if precomputed is None:
        precomputed = getattr(criterion, "detected_phase2_bin_edges", None)
    if precomputed is not None:
        return tuple(float(x) for x in precomputed)

    strategy = _resolve_phase2_rebalance_strategy(criterion, phase)

    if strategy in {"forms", "forms_fixed", "fixed", "equal_bin_mae"}:
        return (5.0, 25.0)

    cached = getattr(criterion, "_phase2_cached_bin_edges", None)
    cache_mode = str(getattr(criterion, "phase2_bin_cache_mode", "once")).strip().lower()
    if cached is not None and cache_mode not in {"off", "none", "false", "0", "batch", "per_batch"}:
        return tuple(float(x) for x in cached)

    if strategy in {"quantile", "quantiles", "auto_quantile", "phase1_quantile"}:
        desired_n_classes = int(
            getattr(
                criterion,
                "phase2_target_n_classes",
                getattr(criterion, "phase2_n_classes", 3),
            )
        )
        min_points_per_class = int(getattr(criterion, "phase2_min_points_per_class", 64))
        min_bin_width = float(getattr(criterion, "phase2_min_bin_width", 0.75))

        edges = _infer_auto_phase2_bin_edges(
            targets=targets,
            desired_n_classes=desired_n_classes,
            min_points_per_class=min_points_per_class,
            min_bin_width=min_bin_width,
        )
        if len(edges) > 0 and cache_mode not in {"off", "none", "false", "0", "batch", "per_batch"}:
            setattr(criterion, "_phase2_cached_bin_edges", tuple(edges))
        return tuple(edges)

    return tuple()



def _get_phase_bin_edges(
    criterion: nn.Module,
    targets: torch.Tensor,
    phase: int,
) -> Tuple[float, ...]:
    """Resolve bin edges for the requested phase."""
    ph = int(phase)
    if ph < 2:
        explicit = getattr(criterion, "phase1_bin_edges", None)
        if explicit is not None:
            return tuple(float(x) for x in explicit)

        train_edges = getattr(criterion, "phase1_train_bin_edges", None)
        if train_edges is not None:
            return tuple(float(x) for x in train_edges)

        strategy = _resolve_phase_rebalance_strategy(criterion, phase)
        if strategy in {"forms", "forms_fixed", "fixed"}:
            return (5.0, 25.0)
        if strategy in {"quantile", "quantiles", "auto_quantile", "phase1_quantile"}:
            desired_n_classes = int(getattr(criterion, "phase1_target_n_classes", 3))
            min_points_per_class = int(getattr(criterion, "phase1_min_points_per_class", 64))
            min_bin_width = float(getattr(criterion, "phase1_min_bin_width", 0.75))
            return _infer_auto_phase2_bin_edges(
                targets=targets,
                desired_n_classes=desired_n_classes,
                min_points_per_class=min_points_per_class,
                min_bin_width=min_bin_width,
            )

        # Last fallback: use the global regression/ordinal bins from HyTecLossV6.
        reg_edges = getattr(criterion, "reg_bin_edges", None)
        if reg_edges is not None:
            try:
                if torch.is_tensor(reg_edges):
                    return tuple(float(x) for x in reg_edges.detach().cpu().tolist())
                return tuple(float(x) for x in reg_edges)
            except Exception:
                return tuple()
        ord_edges = getattr(criterion, "ord_thresholds", None)
        if ord_edges is not None:
            try:
                if torch.is_tensor(ord_edges):
                    return tuple(float(x) for x in ord_edges.detach().cpu().tolist())
                return tuple(float(x) for x in ord_edges)
            except Exception:
                return tuple()
        return tuple()

    return _get_phase2_bin_edges(criterion=criterion, targets=targets, phase=phase)


def _get_phase_class_weights(criterion: nn.Module, phase: int) -> Optional[Tuple[float, ...]]:
    """Return optional class weights for the requested phase."""
    attr = "phase1_class_weights" if int(phase) < 2 else "phase2_class_weights"
    weights = getattr(criterion, attr, None)
    if weights is None:
        return None
    return tuple(float(x) for x in weights)


def _asym_value(criterion: nn.Module, phase: int, name: str, default: float) -> float:
    ph = 1 if int(phase) < 2 else 2
    return float(getattr(criterion, f"phase{ph}_{name}", default))


def _point_weighted_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    regression_loss_name: str,
    huber_beta: float,
    bin_edges: Sequence[float],
    class_weights: Optional[Sequence[float]],
) -> torch.Tensor:
    """Per-point weighted MAE/Huber, useful for ablations."""
    base_loss = _point_regression_loss(
        pred,
        target,
        regression_loss_name=regression_loss_name,
        huber_beta=huber_beta,
    )
    if class_weights is None or len(tuple(bin_edges)) < 1:
        return base_loss.mean()
    weights = _make_regression_point_weights(target, bin_edges, class_weights).to(device=base_loss.device, dtype=base_loss.dtype)
    return (weights * base_loss).sum() / weights.sum().clamp_min(1e-8)


def _asymmetric_weighted_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    criterion: nn.Module,
    phase: int,
    regression_loss_name: str,
    huber_beta: float,
    bin_edges: Sequence[float],
    class_weights: Optional[Sequence[float]],
) -> torch.Tensor:
    """
    Asymmetric anti-shrinkage MAE/Huber.

    It penalizes the two typical shrink-to-mean errors more strongly:
      - low canopy heights over-predicted;
      - high canopy heights under-predicted.
    Optional class weights can be multiplied on top.
    """
    base_loss = _point_regression_loss(
        pred,
        target,
        regression_loss_name=regression_loss_name,
        huber_beta=huber_beta,
    )
    weights = torch.ones_like(base_loss)

    if class_weights is not None and len(tuple(bin_edges)) >= 1:
        weights = weights * _make_regression_point_weights(target, bin_edges, class_weights).to(
            device=base_loss.device,
            dtype=base_loss.dtype,
        )

    low_thr = _asym_value(criterion, phase, "asym_low_threshold", 5.0)
    high_thr = _asym_value(criterion, phase, "asym_high_threshold", 20.0)
    very_high_thr = _asym_value(criterion, phase, "asym_very_high_threshold", 30.0)

    low_over_w = _asym_value(criterion, phase, "asym_low_over_weight", 1.0)
    high_under_w = _asym_value(criterion, phase, "asym_high_under_weight", 1.0)
    very_high_under_w = _asym_value(criterion, phase, "asym_very_high_under_weight", 1.0)

    low_over = (target < low_thr) & (pred > target)
    high_under = (target > high_thr) & (pred < target)
    very_high_under = (target > very_high_thr) & (pred < target)

    weights = torch.where(low_over, weights * float(low_over_w), weights)
    weights = torch.where(high_under, weights * float(high_under_w), weights)
    weights = torch.where(very_high_under, weights * float(very_high_under_w), weights)

    return (weights * base_loss).sum() / weights.sum().clamp_min(1e-8)


def _anti_shrinkage_moment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    criterion: nn.Module,
    phase: int,
) -> torch.Tensor:
    """
    Differentiable batch-level penalties against vertical compression.

    The penalty is intentionally optional and small by default. It is skipped for
    tiny or near-constant batches to avoid unstable slope estimates.
    """
    ph = 1 if int(phase) < 2 else 2
    lambda_slope = float(getattr(criterion, f"phase{ph}_lambda_slope_loss", 0.0))
    lambda_std = float(getattr(criterion, f"phase{ph}_lambda_std_loss", 0.0))
    lambda_bias = float(getattr(criterion, f"phase{ph}_lambda_bias_loss", 0.0))

    z = pred.sum() * 0.0
    if lambda_slope <= 0.0 and lambda_std <= 0.0 and lambda_bias <= 0.0:
        return z
    if pred.numel() < int(getattr(criterion, "anti_shrink_min_points", 32)):
        return z

    eps = float(getattr(criterion, "anti_shrink_eps", 1e-6))
    p = pred.float().reshape(-1)
    y = target.float().reshape(-1)
    finite = torch.isfinite(p) & torch.isfinite(y)
    if int(finite.sum().item()) < int(getattr(criterion, "anti_shrink_min_points", 32)):
        return z
    p = p[finite]
    y = y[finite]

    pc = p - p.mean()
    yc = y - y.mean()
    true_var = (yc * yc).mean()
    if not torch.isfinite(true_var) or float(true_var.detach().item()) < eps:
        return z

    pred_var = (pc * pc).mean().clamp_min(eps)
    true_var = true_var.clamp_min(eps)
    cov = (pc * yc).mean()

    slope = cov / true_var
    std_ratio = torch.sqrt(pred_var) / torch.sqrt(true_var)
    bias = (p - y).mean()

    loss = z
    target_slope = float(getattr(criterion, "anti_shrink_target_slope", 1.0))
    target_std = float(getattr(criterion, "anti_shrink_target_std_ratio", 1.0))
    if lambda_slope > 0.0:
        loss = loss + lambda_slope * (slope - target_slope).pow(2)
    if lambda_std > 0.0:
        loss = loss + lambda_std * (std_ratio - target_std).pow(2)
    if lambda_bias > 0.0:
        loss = loss + lambda_bias * bias.pow(2)
    return loss.to(dtype=pred.dtype)



def _anti_zero_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    criterion: nn.Module,
    phase: int,
) -> torch.Tensor:
    """
    Soft anti-zero penalty for forest/GEDI-valid CHM supervision.

    Purpose
    -------
    Avoid predictions close to 0 m on valid forest GEDI points, without applying
    a hard prediction floor. This keeps the model honest during evaluation while
    discouraging impossible low outputs for the training domain.

    The penalty is active only when:
      - y_true >= anti_zero_min_target_y, e.g. 2.5 m
      - y_pred < anti_zero_min_pred, e.g. 2.0 m

    Formula
    -------
      mean( relu(anti_zero_min_pred - y_pred)^power )

    The returned tensor is already multiplied by the phase-specific lambda:
      phase1_lambda_anti_zero_loss or phase2_lambda_anti_zero_loss.
    """
    ph = 1 if int(phase) < 2 else 2
    lambda_anti_zero = float(getattr(criterion, f"phase{ph}_lambda_anti_zero_loss", 0.0))

    z = pred.sum() * 0.0
    if lambda_anti_zero <= 0.0:
        return z

    p = pred.float().reshape(-1)
    y = target.float().reshape(-1)
    finite = torch.isfinite(p) & torch.isfinite(y)
    if not bool(finite.any()):
        return z

    p = p[finite]
    y = y[finite]

    min_target_y = float(getattr(criterion, "anti_zero_min_target_y", 2.5))
    min_pred = float(getattr(criterion, "anti_zero_min_pred", 2.0))
    power = float(getattr(criterion, "anti_zero_power", 2.0))
    min_points = int(getattr(criterion, "anti_zero_min_points", 1))

    mask = (y >= min_target_y) & (p < min_pred)
    if int(mask.sum().item()) < max(1, min_points):
        return z

    penalty = torch.relu(torch.as_tensor(min_pred, device=p.device, dtype=p.dtype) - p[mask])
    if abs(power - 1.0) < 1e-8:
        loss = penalty.mean()
    else:
        loss = penalty.pow(power).mean()

    return (lambda_anti_zero * loss).to(dtype=pred.dtype)

def _equal_bin_rebalanced_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    regression_loss_name: str,
    huber_beta: float,
    bin_edges: Sequence[float],
    class_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Equal-weight MAE/Huber per height class.

    This matches the FORMS fine-tuning idea:
      1) split target heights into a few broad ranges
      2) compute mean loss inside each range
      3) average the range losses with equal importance

    Important:
      - class membership is determined from TARGET heights, not predictions
      - empty classes are ignored for the current batch
    """
    target_cls = group_id(target, bin_edges).long()
    n_classes = int(len(tuple(float(x) for x in bin_edges)) + 1)

    per_class_losses = []
    per_class_weights = []

    if class_weights is not None:
        if len(class_weights) != n_classes:
            raise ValueError(
                f"class_weights must have len(edges)+1 values. "
                f"Got {len(class_weights)} weights for {n_classes} classes."
            )
        weights = [float(w) for w in class_weights]
    else:
        weights = [1.0] * n_classes

    for cls_idx in range(n_classes):
        m = (target_cls == cls_idx)
        if not bool(m.any()):
            continue

        cls_loss = _point_regression_loss(
            pred[m],
            target[m],
            regression_loss_name=regression_loss_name,
            huber_beta=huber_beta,
        ).mean()

        per_class_losses.append(cls_loss)
        per_class_weights.append(float(weights[cls_idx]))

    if len(per_class_losses) == 0:
        return pred.sum() * 0.0

    losses_t = torch.stack(per_class_losses)
    weights_t = torch.tensor(
        per_class_weights,
        device=losses_t.device,
        dtype=losses_t.dtype,
    )
    return (losses_t * weights_t).sum() / weights_t.sum().clamp_min(1e-8)


def compute_pixelwise_loss(
    pred_map: torch.Tensor,          # (B,1,H,W)
    y_center: torch.Tensor,          # compatibility only; NOT used
    aux_rows: torch.Tensor,          # (B,K)
    aux_cols: torch.Tensor,          # (B,K)
    aux_y: torch.Tensor,             # (B,K)
    criterion: nn.Module,            # HyTecLossV6-like object
    huber_beta: float,
    tv_weight: float = 0.0,
    aux_mask: Optional[torch.Tensor] = None,
    y_sparse: Optional[torch.Tensor] = None,   # (B,H,W) or (B,1,H,W)
    y_mask: Optional[torch.Tensor] = None,     # (B,H,W) or (B,1,H,W)
    y_count: Optional[torch.Tensor] = None,    # optional, currently diagnostic only
    aux_track_id: Optional[torch.Tensor] = None, # (B,K), required for Pauls track-level LS
    phase: int = 1,
    track_shift_meta: Optional[Dict[str, Any]] = None,
    track_shift_context: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor, int]:
    """
    Sparse GEDI loss over ALL valid GEDI points in the batch.

    Scientific choice adopted here:
      - Phase 1: plain MAE by default, or weighted/asymmetric/slope-aware MAE
        when phase1_* ablation arguments are enabled.
      - Phase 2: equal-bin/weighted/asymmetric fine-tuning with optional
        slope/std/bias moment penalties.

    What was removed:
      - no per-patch averaging followed by patch averaging
      - no patch-count weighting for the main regression term

    That makes the main regression loss consistent with the FORMS description:
    every valid GEDI supervision point contributes directly to the batch loss.
    """
    del y_center, y_count
    huber_beta = float(getattr(criterion, "huber_beta", huber_beta))
    regression_loss_name = _resolve_regression_loss_name(criterion)

    pred_map = _ensure_pred_map_4d(pred_map)

    shift_cfg = _shifted_spatial_loss_config(criterion)
    _validate_pauls_strict_loss_contract(
        criterion=criterion,
        phase=int(phase),
        regression_loss_name=regression_loss_name,
        huber_beta=float(huber_beta),
        tv_weight=float(tv_weight),
        shift_cfg=shift_cfg,
    )
    all_shift_candidates: Optional[torch.Tensor] = None
    all_shift_track_ids: Optional[torch.Tensor] = None
    all_shift_orig_track_ids: Optional[torch.Tensor] = None
    all_shift_rows: Optional[torch.Tensor] = None
    all_shift_cols: Optional[torch.Tensor] = None
    all_shift_batch_indices: Optional[torch.Tensor] = None
    all_shift_point_indices: Optional[torch.Tensor] = None
    shifted_point_losses: Optional[torch.Tensor] = None
    shifted_details: Optional[Dict[str, torch.Tensor]] = None

    if bool(shift_cfg["enabled"]):
        # Shifted Huber/MAE must use original aux rows/cols, not raster sparse maps,
        # because y_sparse can aggregate several GEDI shots in one 10 m pixel.
        if bool(shift_cfg.get("track_level", False)):
            if aux_track_id is None:
                raise RuntimeError(
                    "spatial_loss=pauls_track_shifted_huber requires aux_track_id. "
                    "Re-run Step04/Step05 PAULS_TRACKID and ensure training.data returns meta['aux_track_id']."
                )
            (
                per_patch_preds,
                per_patch_targets,
                per_patch_candidates,
                per_patch_track_ids,
                per_patch_orig_track_ids,
                per_patch_rows,
                per_patch_cols,
                per_patch_batch_indices,
                per_patch_point_indices,
                n_pts_total,
                n_empty_patches,
            ) = _gather_track_shifted_points_from_aux_batch(
                pred_map=pred_map,
                aux_rows=aux_rows,
                aux_cols=aux_cols,
                aux_y=aux_y,
                aux_track_id=aux_track_id,
                aux_mask=aux_mask,
                radius_px=int(shift_cfg["radius_px"]),
            )
            if len(per_patch_preds) > 0:
                all_shift_candidates = torch.cat(per_patch_candidates, dim=0)
                all_shift_track_ids = torch.cat(per_patch_track_ids, dim=0)
                all_shift_orig_track_ids = torch.cat(per_patch_orig_track_ids, dim=0)
                all_shift_rows = torch.cat(per_patch_rows, dim=0)
                all_shift_cols = torch.cat(per_patch_cols, dim=0)
                all_shift_batch_indices = torch.cat(per_patch_batch_indices, dim=0)
                all_shift_point_indices = torch.cat(per_patch_point_indices, dim=0)
        else:
            per_patch_preds, per_patch_targets, per_patch_candidates, n_pts_total, n_empty_patches = _gather_shifted_points_from_aux_batch(
                pred_map=pred_map,
                aux_rows=aux_rows,
                aux_cols=aux_cols,
                aux_y=aux_y,
                aux_mask=aux_mask,
                radius_px=int(shift_cfg["radius_px"]),
            )
            if len(per_patch_preds) > 0:
                all_shift_candidates = torch.cat(per_patch_candidates, dim=0)
            else:
                # Defensive fallback: if a future dataset lacks aux coordinates but has
                # raster sparse supervision, continue with exact raster loss instead of
                # silently producing an empty batch.
                per_patch_preds, per_patch_targets, n_pts_total, n_empty_patches = _gather_points_from_raster_sparse_batch(
                    pred_map=pred_map,
                    y_sparse=y_sparse,
                    y_mask=y_mask,
                )
    else:
        per_patch_preds, per_patch_targets, n_pts_total, n_empty_patches = _gather_points_from_raster_sparse_batch(
            pred_map=pred_map,
            y_sparse=y_sparse,
            y_mask=y_mask,
        )

        if len(per_patch_preds) == 0:
            per_patch_preds, per_patch_targets, n_pts_total, n_empty_patches = _gather_points_from_aux_batch(
                pred_map=pred_map,
                aux_rows=aux_rows,
                aux_cols=aux_cols,
                aux_y=aux_y,
                aux_mask=aux_mask,
            )

    if len(per_patch_preds) == 0:
        z = pred_map.sum() * 0.0
        empty = torch.empty((0,), device=pred_map.device, dtype=torch.float32)
        return z, z.detach(), z.detach(), 0, empty, empty, int(n_empty_patches)

    # IMPORTANT: all_preds are exact predictions at GEDI row/col. They are returned
    # for train/val metrics so evaluation stays exact. The shifted window is used
    # only to compute the main regression loss when enabled.
    all_preds = torch.cat(per_patch_preds, dim=0)
    all_targets = torch.cat(per_patch_targets, dim=0)

    main_strategy = _resolve_phase_rebalance_strategy(criterion, phase)
    phase_bin_edges = _get_phase_bin_edges(
        criterion=criterion,
        targets=all_targets,
        phase=phase,
    )
    phase_class_weights = _get_phase_class_weights(criterion, phase)

    if all_shift_candidates is not None:
        if bool(shift_cfg.get("track_level", False)):
            if all_shift_track_ids is None:
                raise RuntimeError("Internal error: Pauls track-level LS has candidates but no track IDs.")
            if bool(shift_cfg.get("fixed_shift", False)):
                if all_shift_orig_track_ids is None:
                    raise RuntimeError("Internal error: fixed_track_shift_huber has candidates but no original aux_track_id.")
                shifted_point_losses, shifted_effective_preds, shifted_details = _reduce_fixed_track_shift_candidate_losses(
                    all_shift_candidates,
                    all_targets,
                    all_shift_orig_track_ids,
                    group_track_ids=all_shift_track_ids,
                    regression_loss_name=regression_loss_name,
                    huber_beta=huber_beta,
                    radius_px=int(shift_cfg["radius_px"]),
                    criterion=criterion,
                    context=track_shift_context,
                    return_details=True,
                )
            else:
                shifted_point_losses, shifted_effective_preds, shifted_details = _reduce_track_shifted_candidate_losses(
                    all_shift_candidates,
                    all_targets,
                    all_shift_track_ids,
                    regression_loss_name=regression_loss_name,
                    huber_beta=huber_beta,
                    reduce_mode=str(shift_cfg["reduce_mode"]),
                    temperature=float(shift_cfg["temperature"]),
                    min_track_points=int(shift_cfg.get("min_track_points", 10)),
                    return_details=True,
                )
            if (
                shifted_details is not None
                and all_shift_orig_track_ids is not None
                and all_shift_rows is not None
                and all_shift_cols is not None
                and all_shift_batch_indices is not None
                and all_shift_point_indices is not None
            ):
                _export_track_shift_decisions(
                    target=all_targets.detach(),
                    pred_exact=all_preds.detach(),
                    pred_shifted=shifted_effective_preds.detach(),
                    group_track_ids=all_shift_track_ids.detach(),
                    orig_track_ids=all_shift_orig_track_ids.detach(),
                    aux_rows=all_shift_rows.detach(),
                    aux_cols=all_shift_cols.detach(),
                    batch_indices=all_shift_batch_indices.detach(),
                    point_indices=all_shift_point_indices.detach(),
                    details=shifted_details,
                    radius_px=int(shift_cfg["radius_px"]),
                    meta=track_shift_meta,
                    context=track_shift_context,
                )
        else:
            shifted_point_losses, shifted_effective_preds = _reduce_shifted_candidate_losses(
                all_shift_candidates,
                all_targets,
                regression_loss_name=regression_loss_name,
                huber_beta=huber_beta,
                reduce_mode=str(shift_cfg["reduce_mode"]),
                temperature=float(shift_cfg["temperature"]),
            )

        if main_strategy == "plain_mae":
            loss_reg_main = shifted_point_losses.mean()
        elif main_strategy in {"weighted_mae", "equal_bin_mae", "balanced_mae", "class_weighted_mae", "forms", "forms_fixed", "fixed", "quantile", "quantiles", "auto_quantile", "phase1_quantile"}:
            if len(phase_bin_edges) >= 1:
                loss_reg_main = _equal_bin_loss_from_point_losses(
                    shifted_point_losses,
                    all_targets,
                    bin_edges=phase_bin_edges,
                    class_weights=phase_class_weights,
                )
            else:
                loss_reg_main = shifted_point_losses.mean()
        elif main_strategy in {"point_weighted_mae", "per_point_weighted_mae"}:
            loss_reg_main = _weighted_loss_from_point_losses(
                shifted_point_losses,
                all_targets,
                bin_edges=phase_bin_edges,
                class_weights=phase_class_weights,
            )
        elif main_strategy == "asymmetric_weighted_mae":
            loss_reg_main = _asymmetric_weighted_loss_from_point_losses(
                shifted_point_losses,
                shifted_effective_preds,
                all_targets,
                criterion=criterion,
                phase=phase,
                bin_edges=phase_bin_edges,
                class_weights=phase_class_weights,
            )
        else:
            loss_reg_main = shifted_point_losses.mean()
    elif main_strategy == "plain_mae":
        loss_reg_main = _point_regression_loss(
            all_preds,
            all_targets,
            regression_loss_name=regression_loss_name,
            huber_beta=huber_beta,
        ).mean()
    elif main_strategy in {"weighted_mae", "equal_bin_mae", "balanced_mae", "class_weighted_mae", "forms", "forms_fixed", "fixed", "quantile", "quantiles", "auto_quantile", "phase1_quantile"}:
        if len(phase_bin_edges) >= 1:
            loss_reg_main = _equal_bin_rebalanced_regression_loss(
                pred=all_preds,
                target=all_targets,
                regression_loss_name=regression_loss_name,
                huber_beta=huber_beta,
                bin_edges=phase_bin_edges,
                class_weights=phase_class_weights,
            )
        else:
            loss_reg_main = _point_regression_loss(
                all_preds,
                all_targets,
                regression_loss_name=regression_loss_name,
                huber_beta=huber_beta,
            ).mean()
    elif main_strategy in {"point_weighted_mae", "per_point_weighted_mae"}:
        loss_reg_main = _point_weighted_regression_loss(
            all_preds,
            all_targets,
            regression_loss_name=regression_loss_name,
            huber_beta=huber_beta,
            bin_edges=phase_bin_edges,
            class_weights=phase_class_weights,
        )
    elif main_strategy == "asymmetric_weighted_mae":
        loss_reg_main = _asymmetric_weighted_regression_loss(
            all_preds,
            all_targets,
            criterion=criterion,
            phase=phase,
            regression_loss_name=regression_loss_name,
            huber_beta=huber_beta,
            bin_edges=phase_bin_edges,
            class_weights=phase_class_weights,
        )
    else:
        # Safe fallback for unknown/experimental names.
        loss_reg_main = _point_regression_loss(
            all_preds,
            all_targets,
            regression_loss_name=regression_loss_name,
            huber_beta=huber_beta,
        ).mean()

    alpha_reg = float(getattr(criterion, "alpha_reg", 1.0))
    # Historical HyTec runs used beta_ord=0.2 by default. In Pauls strict mode,
    # the default must be zero so the optimized objective remains pure shifted
    # Huber even if the launcher omits --beta-ord.
    pauls_strict_active = bool(shift_cfg.get("pauls_strict", False))
    beta_ord_default = 0.0 if pauls_strict_active else 0.2
    beta_ord = float(getattr(criterion, "beta_ord", beta_ord_default))
    gamma_cls = float(getattr(criterion, "gamma_cls", 0.0))

    loss_ord = all_preds.sum() * 0.0
    loss_cls_aux = all_preds.sum() * 0.0
    ord_thresholds = None

    if beta_ord > 0.0 or gamma_cls > 0.0:
        ord_thresholds = getattr(criterion, "ord_thresholds")
        ord_pos_weight = getattr(criterion, "ord_pos_weight")
        temperature = max(float(getattr(criterion, "temperature", 4.0)), float(getattr(criterion, "eps", 1e-6)))
        logits_clip = float(getattr(criterion, "logits_clip", 15.0))

        logits = (all_preds[:, None] - ord_thresholds[None, :]) / temperature
        if logits_clip > 0:
            logits = torch.clamp(logits, -logits_clip, logits_clip)

        targets_ord = (all_targets[:, None] >= ord_thresholds[None, :]).float()
        if beta_ord > 0.0:
            loss_ord = F.binary_cross_entropy_with_logits(
                logits,
                targets_ord,
                pos_weight=ord_pos_weight,
                reduction="mean",
            )

        if gamma_cls > 0.0:
            ord_edges = tuple(float(x) for x in ord_thresholds.detach().cpu().tolist())
            target_cls = group_id(all_targets, ord_edges).long()

            centres = _class_centres_from_edges(ord_edges).to(device=all_preds.device, dtype=all_preds.dtype)
            logits_cls = -torch.abs(all_preds[:, None] - centres[None, :])

            cbw = getattr(criterion, "cls_bin_weights", None)
            if cbw is not None:
                loss_cls_aux = F.cross_entropy(logits_cls, target_cls, weight=cbw.to(logits_cls.device))
            else:
                loss_cls_aux = F.cross_entropy(logits_cls, target_cls)

    loss_tv = pred_map.sum() * 0.0
    if float(tv_weight) > 0.0:
        dx = pred_map[:, :, :, 1:] - pred_map[:, :, :, :-1]
        dy = pred_map[:, :, 1:, :] - pred_map[:, :, :-1, :]
        loss_tv = float(tv_weight) * (dx.abs().mean() + dy.abs().mean())

    loss_anti_shrink = _anti_shrinkage_moment_loss(
        all_preds,
        all_targets,
        criterion=criterion,
        phase=phase,
    )
    loss_anti_zero = _anti_zero_loss(
        all_preds,
        all_targets,
        criterion=criterion,
        phase=phase,
    )
    loss_total = (
        alpha_reg * loss_reg_main
        + beta_ord * loss_ord
        + gamma_cls * loss_cls_aux
        + loss_tv
        + loss_anti_shrink
        + loss_anti_zero
    )

    with torch.no_grad():
        if shifted_point_losses is not None:
            # Log the unweighted spatial loss actually optimized. Exact-position
            # evaluation metrics still use all_preds/all_targets.
            pure_reg = shifted_point_losses.mean()
        elif regression_loss_name == "huber":
            pure_reg = _huber_loss(
                all_preds,
                all_targets,
                delta=float(huber_beta),
                reduction="mean",
            )
        else:
            pure_reg = F.l1_loss(all_preds, all_targets, reduction="mean")

    return (
        loss_total,
        pure_reg.detach(),
        loss_cls_aux.detach(),
        int(n_pts_total),
        all_preds.detach(),
        all_targets.detach(),
        int(n_empty_patches),
    )


# -------------------------------------------------------------------------------------------------
# Batch unpacking
# -------------------------------------------------------------------------------------------------
def _unpack_batch(batch):
    """
    Accept:
      - (x, y, aux_rows, aux_cols, aux_y)
      - (x, y, aux_rows, aux_cols, aux_y, aux_mask)
      - (x, y, aux_rows, aux_cols, aux_y, aux_mask, y_sparse, y_mask, y_count)
      - (x, y, aux_rows, aux_cols, aux_y, aux_mask, y_sparse, y_mask, y_count, meta)

    meta is optional and is used only for GEDI-unique validation/test metrics.
    It should be a dict containing tensors such as:
      aux_shot_uid, aux_gedi_ordinal, aux_temporal_delta_days,
      aux_abs_temporal_delta_days, aux_source_rowid.
    """
    if not isinstance(batch, (list, tuple)):
        raise RuntimeError("Unexpected batch format from DataLoader")

    if len(batch) >= 10:
        x, y, aux_rows, aux_cols, aux_y, aux_mask, y_sparse, y_mask, y_count, meta = batch[:10]
        return x, y, aux_rows, aux_cols, aux_y, aux_mask, y_sparse, y_mask, y_count, meta

    if len(batch) >= 9:
        x, y, aux_rows, aux_cols, aux_y, aux_mask, y_sparse, y_mask, y_count = batch[:9]
        return x, y, aux_rows, aux_cols, aux_y, aux_mask, y_sparse, y_mask, y_count, None

    if len(batch) >= 6:
        x, y, aux_rows, aux_cols, aux_y, aux_mask = batch[:6]
        return x, y, aux_rows, aux_cols, aux_y, aux_mask, None, None, None, None

    if len(batch) >= 5:
        x, y, aux_rows, aux_cols, aux_y = batch[:5]
        return x, y, aux_rows, aux_cols, aux_y, None, None, None, None, None

    raise RuntimeError("Unexpected batch format from DataLoader")

def _augment_batch_sparse_targets(
    x: torch.Tensor,
    aux_rows: torch.Tensor,
    aux_cols: torch.Tensor,
    aux_mask: Optional[torch.Tensor],
    y_sparse: Optional[torch.Tensor],
    y_mask: Optional[torch.Tensor],
    y_count: Optional[torch.Tensor],
    *,
    return_flip_flags: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Apply the same SAR-safe flips to:
      - inputs x
      - legacy aux_* sparse supervision
      - raster sparse supervision (y_sparse / y_mask / y_count)

    When ``return_flip_flags=True``, the two trailing returned booleans indicate
    whether a horizontal and/or vertical flip was applied.  This is needed by
    ``fixed_track_shift_huber`` because a frozen geographic shift vector must be
    transformed after geometric augmentation.
    """
    if x.ndim != 4:
        if return_flip_flags:
            return x, aux_rows, aux_cols, y_sparse, y_mask, y_count, False, False
        return x, aux_rows, aux_cols, y_sparse, y_mask, y_count

    _, _, H, W = x.shape
    valid_aux = (aux_rows >= 0) & (aux_cols >= 0)
    if aux_mask is not None:
        valid_aux = valid_aux & aux_mask.bool()

    do_h = torch.rand(1).item() < 0.5
    do_v = torch.rand(1).item() < 0.5

    if do_h:
        x = torch.flip(x, dims=[-1])
        aux_cols = torch.where(valid_aux, (W - 1) - aux_cols, aux_cols)
        if y_sparse is not None:
            y_sparse = torch.flip(y_sparse, dims=[-1])
        if y_mask is not None:
            y_mask = torch.flip(y_mask, dims=[-1])
        if y_count is not None:
            y_count = torch.flip(y_count, dims=[-1])

    if do_v:
        x = torch.flip(x, dims=[-2])
        aux_rows = torch.where(valid_aux, (H - 1) - aux_rows, aux_rows)
        if y_sparse is not None:
            y_sparse = torch.flip(y_sparse, dims=[-2])
        if y_mask is not None:
            y_mask = torch.flip(y_mask, dims=[-2])
        if y_count is not None:
            y_count = torch.flip(y_count, dims=[-2])

    if return_flip_flags:
        return x, aux_rows, aux_cols, y_sparse, y_mask, y_count, bool(do_h), bool(do_v)
    return x, aux_rows, aux_cols, y_sparse, y_mask, y_count



# -------------------------------------------------------------------------------------------------
# GEDI-unique validation metrics
# -------------------------------------------------------------------------------------------------
def _as_meta_tensor(
    meta: Optional[Dict[str, Any]],
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    like: torch.Tensor,
    fill_value: float | int,
) -> torch.Tensor:
    if not isinstance(meta, dict) or key not in meta:
        return torch.full(tuple(like.shape), fill_value, device=device, dtype=dtype)

    v = meta[key]
    if torch.is_tensor(v):
        t = v.to(device=device, non_blocking=True)
    else:
        t = torch.as_tensor(v, device=device)

    if t.shape != like.shape:
        try:
            t = t.reshape(tuple(like.shape))
        except Exception:
            return torch.full(tuple(like.shape), fill_value, device=device, dtype=dtype)

    return t.to(dtype=dtype)


def _collect_unique_aux_points_from_batch(
    *,
    pred_map: torch.Tensor,
    aux_rows: torch.Tensor,
    aux_cols: torch.Tensor,
    aux_y: torch.Tensor,
    aux_mask: Optional[torch.Tensor],
    meta: Optional[Dict[str, Any]],
) -> Optional[Dict[str, np.ndarray]]:
    """
    Collect predictions at original aux GEDI coordinates with numeric shot ids.

    This is intentionally based on aux_* rather than raster sparse maps because
    the raster sparse map can aggregate multiple GEDI shots falling in the same
    10 m pixel. Unique-shot metrics require shot identity.
    """
    if not isinstance(meta, dict):
        return None

    pred_map = _ensure_pred_map_4d(pred_map)
    B, _, H, W = pred_map.shape

    shot_uid = _as_meta_tensor(
        meta, "aux_shot_uid", device=pred_map.device, dtype=torch.long, like=aux_rows, fill_value=0
    )
    gedi_ord = _as_meta_tensor(
        meta, "aux_gedi_ordinal", device=pred_map.device, dtype=torch.long, like=aux_rows, fill_value=0
    )
    delta = _as_meta_tensor(
        meta, "aux_temporal_delta_days", device=pred_map.device, dtype=torch.float32, like=aux_y, fill_value=float("nan")
    )
    abs_delta = _as_meta_tensor(
        meta, "aux_abs_temporal_delta_days", device=pred_map.device, dtype=torch.float32, like=aux_y, fill_value=float("nan")
    )

    valid = _build_valid_mask(
        aux_rows=aux_rows,
        aux_cols=aux_cols,
        aux_y=aux_y,
        H=H,
        W=W,
        aux_mask=aux_mask,
    )
    valid = valid & (shot_uid > 0)

    if not bool(valid.any()):
        return None

    b_idx, k_idx = torch.where(valid)
    r = aux_rows[b_idx, k_idx].long()
    c = aux_cols[b_idx, k_idx].long()

    pred = pred_map[b_idx, 0, r, c].detach().float()
    true = aux_y[b_idx, k_idx].detach().float()

    finite = torch.isfinite(pred) & torch.isfinite(true)
    if not bool(finite.any()):
        return None

    b_idx = b_idx[finite]
    k_idx = k_idx[finite]

    return {
        "shot_uid": shot_uid[b_idx, k_idx].detach().cpu().numpy().astype(np.int64, copy=False),
        "gedi_ordinal": gedi_ord[b_idx, k_idx].detach().cpu().numpy().astype(np.int64, copy=False),
        "temporal_delta_days": delta[b_idx, k_idx].detach().cpu().numpy().astype(np.float32, copy=False),
        "abs_temporal_delta_days": abs_delta[b_idx, k_idx].detach().cpu().numpy().astype(np.float32, copy=False),
        "y_true": true[finite].detach().cpu().numpy().astype(np.float32, copy=False),
        "y_pred": pred[finite].detach().cpu().numpy().astype(np.float32, copy=False),
    }


def _np_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]

    if y.size == 0:
        keys = [
            "mae", "mse", "rmse", "r2", "bias", "pred_mean", "pred_std", "pred_min", "pred_max",
            "true_mean", "true_std", "true_min", "true_max", "std_ratio", "slope", "corr", "n"
        ]
        return {f"{k}_{prefix}": float("nan") for k in keys}

    diff = p - y
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    bias = float(np.mean(diff))

    y_mean = float(np.mean(y))
    p_mean = float(np.mean(p))
    y_std = float(np.std(y))
    p_std = float(np.std(p))
    sst = float(np.sum((y - y_mean) ** 2))
    sse = float(np.sum((y - p) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 and math.isfinite(sst) else float("nan")
    std_ratio = float(p_std / max(y_std, 1e-8)) if math.isfinite(p_std) and math.isfinite(y_std) else float("nan")

    if y.size >= 2 and np.var(y) > 0:
        cov = float(np.mean((p - p_mean) * (y - y_mean)))
        slope = float(cov / max(float(np.var(y)), 1e-12))
    else:
        slope = float("nan")

    if y.size >= 2 and p_std > 0 and y_std > 0:
        corr = float(np.corrcoef(p, y)[0, 1])
        corr = float(np.clip(corr, -1.0, 1.0))
    else:
        corr = float("nan")

    return {
        f"mae_{prefix}": mae,
        f"mse_{prefix}": mse,
        f"rmse_{prefix}": rmse,
        f"r2_{prefix}": r2,
        f"bias_{prefix}": bias,
        f"pred_mean_{prefix}": p_mean,
        f"pred_std_{prefix}": p_std,
        f"pred_min_{prefix}": float(np.min(p)),
        f"pred_max_{prefix}": float(np.max(p)),
        f"true_mean_{prefix}": y_mean,
        f"true_std_{prefix}": y_std,
        f"true_min_{prefix}": float(np.min(y)),
        f"true_max_{prefix}": float(np.max(y)),
        f"std_ratio_{prefix}": std_ratio,
        f"slope_{prefix}": slope,
        f"corr_{prefix}": corr,
        f"n_{prefix}": int(y.size),
    }


def _domain_masks_for_eval15(y_true: np.ndarray) -> Dict[str, np.ndarray]:
    """Evaluation domains for unique-shot and occurrence-level metrics."""
    y = np.asarray(y_true, dtype=np.float64)
    max_h = float(_eval_max_height_m())
    pmin = float(_eval_primary_min_height_m())
    pmax = float(_eval_primary_max_height_m())
    amin = float(_eval_audit_min_height_m())
    amax = float(_eval_audit_max_height_m())
    finite = np.isfinite(y)
    return {
        "all": finite & (y >= 0.0) & (y <= max_h),
        "eval15": finite & (y >= 0.0) & (y <= max_h),
        "low_0_2": finite & (y >= 0.0) & (y < 2.0),
        "ge2": finite & (y >= 2.0) & (y <= max_h),
        "primary": _domain_mask_np(y, pmin, pmax, inclusive_hi=True),
        "audit_tail": _domain_mask_np(y, amin, amax, inclusive_hi=True),
    }


def _update_occurrence_domain_regs(
    regs: Dict[str, RunningReg],
    pred_pts: torch.Tensor,
    tgt_pts: torch.Tensor,
) -> None:
    """Update occurrence-level domain metrics used for FORMS-T-style monitoring.

    Definitions:
      - occurrence_eval15: every valid GEDI occurrence with 0 <= RH95 <= 15 counts;
      - occurrence_ge2: every valid GEDI occurrence with 2 <= RH95 <= 15 counts;
      - occurrence_low_0_2: every valid GEDI occurrence with 0 <= RH95 < 2 counts.

    This does not average by shot_id and does not change the training loss. It only
    exposes monitor/reporting metrics such as mae_occurrence_ge2, allowing
    --monitor val_mae_occurrence_ge2.
    """
    if pred_pts is None or tgt_pts is None:
        return

    p = pred_pts.detach().float().reshape(-1)
    y = tgt_pts.detach().float().reshape(-1)
    if p.numel() == 0 or y.numel() == 0:
        return

    finite = torch.isfinite(p) & torch.isfinite(y)
    max_h = float(_eval_max_height_m())
    pmin = float(_eval_primary_min_height_m())
    pmax = float(_eval_primary_max_height_m())
    amin = float(_eval_audit_min_height_m())
    amax = float(_eval_audit_max_height_m())
    masks = {
        "eval15": finite & (y >= 0.0) & (y <= max_h),
        "ge2": finite & (y >= 2.0) & (y <= max_h),
        "low_0_2": finite & (y >= 0.0) & (y < 2.0),
        "primary": finite & _domain_mask_torch(y, pmin, pmax, inclusive_hi=True),
        "audit_tail": finite & _domain_mask_torch(y, amin, amax, inclusive_hi=True),
    }
    for name, mask in masks.items():
        if name in regs and bool(mask.any()):
            regs[name].update(p, y, mask=mask)


def _occurrence_domain_metrics(regs: Dict[str, RunningReg]) -> Dict[str, float]:
    """Return occurrence-level domain metrics with stable key names."""
    out: Dict[str, float] = {}
    for name, reg in regs.items():
        m = reg.compute()
        suffix = name
        for key, value in m.items():
            out[f"{key}_occurrence_{suffix}"] = value

    # Convenience monitor aliases used by 06_train_experiment.py.
    if "mae_occurrence_ge2" in out:
        out["val_mae_occurrence_ge2"] = out["mae_occurrence_ge2"]
        out["test_mae_occurrence_ge2"] = out["mae_occurrence_ge2"]
        out["train_mae_occurrence_ge2"] = out["mae_occurrence_ge2"]
    if "mae_occurrence_eval15" in out:
        out["val_mae_occurrence_eval15"] = out["mae_occurrence_eval15"]
        out["test_mae_occurrence_eval15"] = out["mae_occurrence_eval15"]
        out["train_mae_occurrence_eval15"] = out["mae_occurrence_eval15"]
    if "mae_occurrence_primary" in out:
        out["val_mae_primary"] = out["mae_occurrence_primary"]
        out["test_mae_primary"] = out["mae_occurrence_primary"]
        out["train_mae_primary"] = out["mae_occurrence_primary"]
        out["val_rmse_primary"] = out.get("rmse_occurrence_primary", float("nan"))
        out["val_r2_primary"] = out.get("r2_occurrence_primary", float("nan"))
        out["val_bias_primary"] = out.get("bias_occurrence_primary", float("nan"))
        out["val_slope_primary"] = out.get("slope_occurrence_primary", float("nan"))
        out["val_stdr_primary"] = out.get("std_ratio_occurrence_primary", float("nan"))
    if "mae_occurrence_audit_tail" in out:
        out["val_mae_audit_tail"] = out["mae_occurrence_audit_tail"]
        out["test_mae_audit_tail"] = out["mae_occurrence_audit_tail"]
        out["train_mae_audit_tail"] = out["mae_occurrence_audit_tail"]
    return out


def _compute_gedi_unique_metrics(chunks: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """
    Compute shot-level validation/test metrics from concatenated aux GEDI points.

    Definitions:
      - occurrence_aux: each shot occurrence in each patch-month counts.
      - unique_nearest: one occurrence per shot, nearest in time to S2 anchor.
      - unique_temporal_pred_mean: average predictions per shot, compare to shot truth.
      - unique_temporal_error_mean: average |error| per shot, then average across shots.

    The official best-checkpoint monitor is:
      mae_unique_temporal_error_mean_ge2
    """
    if not chunks:
        return {
            "gedi_unique_metadata_available": 0.0,
            "mae_unique_temporal_error_mean_ge2": float("nan"),
            "val_mae_unique_temporal_error_mean_ge2": float("nan"),
        }

    cat: Dict[str, np.ndarray] = {}
    for key in ("shot_uid", "gedi_ordinal", "temporal_delta_days", "abs_temporal_delta_days", "y_true", "y_pred"):
        xs = [np.asarray(c[key]).reshape(-1) for c in chunks if key in c and np.asarray(c[key]).size > 0]
        cat[key] = np.concatenate(xs, axis=0) if xs else np.empty((0,), dtype=np.float32)

    uid = np.asarray(cat["shot_uid"], dtype=np.int64)
    y = np.asarray(cat["y_true"], dtype=np.float64)
    p = np.asarray(cat["y_pred"], dtype=np.float64)
    abs_d = np.asarray(cat["abs_temporal_delta_days"], dtype=np.float64)
    delta = np.asarray(cat["temporal_delta_days"], dtype=np.float64)

    finite = (uid > 0) & np.isfinite(y) & np.isfinite(p)
    uid = uid[finite]
    y = y[finite]
    p = p[finite]
    abs_d = abs_d[finite] if abs_d.shape[0] == finite.shape[0] else np.full_like(y, np.nan)
    delta = delta[finite] if delta.shape[0] == finite.shape[0] else np.full_like(y, np.nan)

    out: Dict[str, float] = {
        "gedi_unique_metadata_available": 1.0 if uid.size > 0 else 0.0,
        "n_aux_occurrence_all_raw": int(uid.size),
    }

    if uid.size == 0:
        out["mae_unique_temporal_error_mean_ge2"] = float("nan")
        out["val_mae_unique_temporal_error_mean_ge2"] = float("nan")
        return out

    domains = _domain_masks_for_eval15(y)

    # Occurrence metrics directly from aux points.
    for dom, mask in domains.items():
        out.update(_np_regression_metrics(y[mask], p[mask], f"aux_occurrence_{dom}"))
        out[f"n_aux_occurrence_{dom}"] = int(mask.sum())

    # Group by shot id.
    per_shot: Dict[int, Dict[str, Any]] = {}
    for sid, yy, pp, ad, dd in zip(uid, y, p, abs_d, delta):
        sid_i = int(sid)
        rec = per_shot.get(sid_i)
        if rec is None:
            rec = {
                "ys": [],
                "ps": [],
                "abs_errors": [],
                "best_abs_delta": float("inf"),
                "best_abs_delta_isfinite": False,
                "nearest_y": float(yy),
                "nearest_p": float(pp),
                "nearest_delta": float(dd) if math.isfinite(float(dd)) else float("nan"),
            }
            per_shot[sid_i] = rec

        rec["ys"].append(float(yy))
        rec["ps"].append(float(pp))
        rec["abs_errors"].append(abs(float(pp) - float(yy)))

        ad_f = float(ad) if math.isfinite(float(ad)) else float("inf")
        cur_best = float(rec["best_abs_delta"])
        if ad_f < cur_best:
            rec["best_abs_delta"] = ad_f
            rec["best_abs_delta_isfinite"] = math.isfinite(float(ad))
            rec["nearest_y"] = float(yy)
            rec["nearest_p"] = float(pp)
            rec["nearest_delta"] = float(dd) if math.isfinite(float(dd)) else float("nan")

    nearest_y = []
    nearest_p = []
    pred_mean_y = []
    pred_mean_p = []
    error_mean_y_ref = []
    error_mean_abs = []
    pred_std = []
    n_occ = []
    nearest_abs_delta = []

    for rec in per_shot.values():
        ys = np.asarray(rec["ys"], dtype=np.float64)
        ps = np.asarray(rec["ps"], dtype=np.float64)
        ae = np.asarray(rec["abs_errors"], dtype=np.float64)

        nearest_y.append(float(rec["nearest_y"]))
        nearest_p.append(float(rec["nearest_p"]))
        nearest_abs_delta.append(float(rec["best_abs_delta"]) if math.isfinite(float(rec["best_abs_delta"])) else float("nan"))

        pred_mean_y.append(float(np.mean(ys)))
        pred_mean_p.append(float(np.mean(ps)))
        error_mean_y_ref.append(float(np.mean(ys)))
        error_mean_abs.append(float(np.mean(ae)))
        pred_std.append(float(np.std(ps)) if ps.size > 1 else 0.0)
        n_occ.append(int(ps.size))

    nearest_y = np.asarray(nearest_y, dtype=np.float64)
    nearest_p = np.asarray(nearest_p, dtype=np.float64)
    pred_mean_y = np.asarray(pred_mean_y, dtype=np.float64)
    pred_mean_p = np.asarray(pred_mean_p, dtype=np.float64)
    error_mean_y_ref = np.asarray(error_mean_y_ref, dtype=np.float64)
    error_mean_abs = np.asarray(error_mean_abs, dtype=np.float64)
    pred_std = np.asarray(pred_std, dtype=np.float64)
    n_occ = np.asarray(n_occ, dtype=np.int64)
    nearest_abs_delta = np.asarray(nearest_abs_delta, dtype=np.float64)

    out["n_unique_shots_raw"] = int(len(per_shot))
    out["temporal_occurrences_per_shot_mean"] = float(np.mean(n_occ)) if n_occ.size else float("nan")
    out["temporal_occurrences_per_shot_max"] = float(np.max(n_occ)) if n_occ.size else float("nan")

    # Unique metrics by domain.
    for dom, mask_nearest in _domain_masks_for_eval15(nearest_y).items():
        out.update(_np_regression_metrics(nearest_y[mask_nearest], nearest_p[mask_nearest], f"unique_nearest_{dom}"))
        out[f"n_unique_nearest_{dom}"] = int(mask_nearest.sum())
        if mask_nearest.any():
            out[f"nearest_abs_temporal_delta_days_mean_{dom}"] = float(np.nanmean(nearest_abs_delta[mask_nearest]))
        else:
            out[f"nearest_abs_temporal_delta_days_mean_{dom}"] = float("nan")

    for dom, mask_mean in _domain_masks_for_eval15(pred_mean_y).items():
        out.update(_np_regression_metrics(pred_mean_y[mask_mean], pred_mean_p[mask_mean], f"unique_temporal_pred_mean_{dom}"))
        out[f"n_unique_temporal_pred_mean_{dom}"] = int(mask_mean.sum())
        if mask_mean.any():
            out[f"temporal_pred_std_mean_{dom}"] = float(np.mean(pred_std[mask_mean]))
            out[f"temporal_pred_std_median_{dom}"] = float(np.median(pred_std[mask_mean]))
        else:
            out[f"temporal_pred_std_mean_{dom}"] = float("nan")
            out[f"temporal_pred_std_median_{dom}"] = float("nan")

    for dom, mask_err in _domain_masks_for_eval15(error_mean_y_ref).items():
        vals = error_mean_abs[mask_err]
        out[f"mae_unique_temporal_error_mean_{dom}"] = float(np.mean(vals)) if vals.size else float("nan")
        out[f"median_unique_temporal_error_mean_{dom}"] = float(np.median(vals)) if vals.size else float("nan")
        out[f"std_unique_temporal_error_mean_{dom}"] = float(np.std(vals)) if vals.size else float("nan")
        out[f"n_unique_temporal_error_mean_{dom}"] = int(vals.size)

    # Compatibility aliases expected by monitor/test scripts.
    for base in [
        "mae_unique_temporal_error_mean_ge2",
        "mae_unique_nearest_ge2",
        "rmse_unique_nearest_ge2",
        "r2_unique_nearest_ge2",
        "mae_unique_temporal_pred_mean_ge2",
        "rmse_unique_temporal_pred_mean_ge2",
        "r2_unique_temporal_pred_mean_ge2",
        "temporal_pred_std_mean_ge2",
        "temporal_pred_std_median_ge2",
    ]:
        if base in out:
            out[f"val_{base}"] = out[base]
            out[f"test_{base}"] = out[base]

    return out



def _fmt_train_metric(value: Any, digits: int = 4) -> str:
    """Format metric values for one-line train-window logging.

    The helper is deliberately tolerant of missing keys because different
    Step06 wrappers can enable/disable some metric families.
    """
    x = _sf(value, default=float("nan"))
    if not math.isfinite(x):
        return "nan"
    if abs(x) >= 1000:
        return f"{x:.1f}"
    return f"{x:.{int(digits)}f}"


def _print_train_window_summary(
    *,
    metrics: Dict[str, Any],
    phase: int,
    cycle: int,
    steps_seen: int,
    total_steps: Optional[int],
) -> None:
    """Print a stable, parseable train-window metric line.

    Why this exists
    ---------------
    Some Step06 wrappers show detailed validation metrics but only expose train
    loss through generic ``loss_huber`` entries.  This summary prints the same
    core regression diagnostics for the recent training window:
    MAE, RMSE, R2, bias, slope, std_ratio, prediction range, and point counts.

    Scientific convention
    ---------------------
    These metrics use exact GEDI coordinates returned by ``compute_pixelwise_loss``.
    For Pauls shifted-Huber training, the shift affects the optimized loss only;
    the metrics reported here remain exact-position diagnostics, matching the
    validation/test convention used by this trainloop.

    The ``primary_*`` fields use CHM_EVAL_PRIMARY_MIN_HEIGHT /
    CHM_EVAL_PRIMARY_MAX_HEIGHT, so Maamoura reports 2.5-20 m and Ifran reports
    2.5-40 m when the Step06 wrapper sets those environment variables.
    """
    if not _env_flag("CHM_PRINT_TRAIN_WINDOW", default=True):
        return

    total_txt = "?" if total_steps is None or int(total_steps) <= 0 else str(int(total_steps))
    parts = [
        "[TRAIN-WINDOW]",
        f"cycle={int(cycle):04d}",
        f"phase={int(phase)}",
        f"steps={int(steps_seen)}/{total_txt}",
        f"loss={_fmt_train_metric(metrics.get('loss_total'), 5)}",
        f"reg={_fmt_train_metric(metrics.get('loss_reg'), 5)}",
        f"mae={_fmt_train_metric(metrics.get('mae'), 4)}",
        f"rmse={_fmt_train_metric(metrics.get('rmse'), 4)}",
        f"r2={_fmt_train_metric(metrics.get('r2'), 4)}",
        f"bias={_fmt_train_metric(metrics.get('bias'), 4)}",
        f"slope={_fmt_train_metric(metrics.get('slope'), 4)}",
        f"stdr={_fmt_train_metric(metrics.get('std_ratio'), 4)}",
        f"pred_std={_fmt_train_metric(metrics.get('pred_std'), 4)}",
        f"max_pred={_fmt_train_metric(metrics.get('pred_max'), 2)}",
        f"max_true={_fmt_train_metric(metrics.get('true_max'), 2)}",
        f"n_pts={int(_sf(metrics.get('n_points', metrics.get('n', 0)), default=0))}",
        f"n_patches={int(_sf(metrics.get('n_patches', 0), default=0))}",
        f"empty_patches={int(_sf(metrics.get('n_empty_patches', 0), default=0))}",
        f"skipped_batches={int(_sf(metrics.get('n_skipped_batches', 0), default=0))}",
    ]

    # Primary-domain metrics are only appended if the primary domain actually has
    # valid points. This prevents cluttering the line with NaN values during rare
    # empty-window cases.
    primary_n = int(_sf(metrics.get("n_occurrence_primary", 0), default=0))
    if primary_n > 0:
        parts.extend([
            f"primary_mae={_fmt_train_metric(metrics.get('mae_occurrence_primary'), 4)}",
            f"primary_rmse={_fmt_train_metric(metrics.get('rmse_occurrence_primary'), 4)}",
            f"primary_r2={_fmt_train_metric(metrics.get('r2_occurrence_primary'), 4)}",
            f"primary_bias={_fmt_train_metric(metrics.get('bias_occurrence_primary'), 4)}",
            f"primary_slope={_fmt_train_metric(metrics.get('slope_occurrence_primary'), 4)}",
            f"primary_stdr={_fmt_train_metric(metrics.get('std_ratio_occurrence_primary'), 4)}",
            f"primary_n={primary_n}",
        ])

    print(" | ".join(parts), flush=True)

# -------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------
# Growth Loss temporal consistency
# -------------------------------------------------------------------------------------------------
def _growth_meta_vector(meta: Dict[str, Any], key: str, *, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Return a 1-D metadata tensor from a DataLoader-collated meta dict."""
    if not isinstance(meta, dict) or key not in meta:
        return None
    value = meta.get(key)
    try:
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype).view(-1)
        return torch.as_tensor(value, device=device, dtype=dtype).view(-1)
    except Exception:
        return None


def _temporal_growth_consistency_loss(
    *,
    pred_map: torch.Tensor,
    x: torch.Tensor,
    meta: Dict[str, Any],
    criterion: nn.Module,
    phase: int,
) -> Tuple[torch.Tensor, int]:
    """Dense same-month inter-annual temporal consistency regularizer.

    The loss groups predictions by (patch_hash, month, year), averages duplicate
    samples of the same patch/month/year if present, and compares only same-patch,
    same-month, consecutive-year pairs. This avoids comparing phenologically
    different observations such as May in one year with September in another.

    Stable pixels:
        ReLU(-dH - dmax)^2 + ReLU(dH - gmax)^2

    Disturbed pixels when growth_stable_mask_mode="disturbance_guard". Disturbance is detected first from ECHOSAT-inspired GEDI RH95 drop rules, optionally complemented by RS proxies:
        alpha * ReLU(dH - gmax)^2

    This means true abrupt losses due to fire/logging/mortality are allowed on
    disturbed pixels, while unrealistic positive jumps are still controlled.
    """
    # AGADIR_TRUEORD_GROWTH_TUPLE_V1
    # A true ordinal model returns (regression_map, ordinal_logits).
    # Temporal growth regularization applies only to the regression map,
    # including the zero-weight fast path that calls tensor.new_zeros().
    pred_map = _ensure_pred_map_4d(pred_map)
    ph = int(phase)
    lambda_growth = float(getattr(criterion, f"phase{ph}_lambda_growth_loss", 0.0))
    if lambda_growth <= 0.0:
        return pred_map.new_zeros(()), 0

    if pred_map.ndim == 3:
        pred = pred_map.unsqueeze(1)
    elif pred_map.ndim == 4:
        pred = pred_map[:, :1]
    else:
        return pred_map.new_zeros(()), 0

    B = int(pred.shape[0])
    if B < 2:
        return pred_map.new_zeros(()), 0

    patch_hash = _growth_meta_vector(meta, "sample_patch_hash", device=pred.device, dtype=torch.long)
    years = _growth_meta_vector(meta, "sample_year", device=pred.device, dtype=torch.long)
    months = _growth_meta_vector(meta, "sample_month", device=pred.device, dtype=torch.long)
    sample_rh95_median = _growth_meta_vector(meta, "sample_rh95_median", device=pred.device, dtype=torch.float32)
    if (
        patch_hash is None or years is None or months is None
        or patch_hash.numel() < B or years.numel() < B or months.numel() < B
    ):
        return pred_map.new_zeros(()), 0

    stride = max(1, int(getattr(criterion, "growth_loss_stride", 8)))
    pred_s = pred[..., ::stride, ::stride]

    mode = str(getattr(criterion, "growth_stable_mask_mode", "aoi")).strip().lower()
    aoi_mask: Optional[torch.Tensor] = None
    disturbance_mask: Optional[torch.Tensor] = None

    aoi_idx = int(getattr(criterion, "growth_aoi_channel_index", 10))
    aoi_thr = float(getattr(criterion, "growth_aoi_threshold", 0.5))
    if x is not None and x.ndim == 4 and 0 <= aoi_idx < int(x.shape[1]):
        aoi_mask = (x[:, aoi_idx:aoi_idx + 1, ::stride, ::stride] > aoi_thr)

    if aoi_mask is None:
        aoi_mask = torch.ones_like(pred_s, dtype=torch.bool)

    # Disturbance proxy from remote-sensing signal changes when explicit masks
    # are not yet available. With the current C13 stack, this uses:
    #   - NDVI from S2 B04/B08
    #   - Sentinel-1 VH/VV changes
    #   - PALSAR HV/HH changes
    # A later version can replace/add an explicit burned-area or forest-loss mask.
    disturbance_mode = mode in {"disturbance_guard", "disturbance", "guard"}
    if disturbance_mode and x is not None and x.ndim == 4:
        disturbance_mask = torch.zeros_like(pred_s, dtype=torch.bool)
    else:
        disturbance_mask = torch.zeros_like(pred_s, dtype=torch.bool)

    dmax = float(getattr(criterion, "growth_dmax_m_per_year", 2.0))
    gmax = float(getattr(criterion, "growth_gmax_m_per_year", 1.2))
    max_gap = int(getattr(criterion, "growth_max_year_gap", 1))
    alpha_disturbed = float(getattr(criterion, "growth_disturbed_jump_weight", 0.1))

    ndvi_drop_thr = float(getattr(criterion, "growth_ndvi_drop_threshold", 0.18))
    s1_change_thr = float(getattr(criterion, "growth_s1_change_threshold", 0.20))
    palsar_change_thr = float(getattr(criterion, "growth_palsar_change_threshold", 0.18))
    min_signals = max(1, int(getattr(criterion, "growth_disturbance_min_signals", 2)))
    gedi_drop_m = float(getattr(criterion, "growth_gedi_drop_threshold_m", 4.0))
    gedi_drop_rel = float(getattr(criterion, "growth_gedi_drop_rel_threshold", 0.50))
    gedi_final_h = float(getattr(criterion, "growth_gedi_final_height_threshold_m", 10.0))

    patch_cpu = patch_hash[:B].detach().cpu().tolist()
    year_cpu = years[:B].detach().cpu().tolist()
    month_cpu = months[:B].detach().cpu().tolist()

    by_patch_month_year: Dict[Tuple[int, int, int], list[int]] = {}
    for i, (p, y, mo) in enumerate(zip(patch_cpu, year_cpu, month_cpu)):
        try:
            p_i = int(p)
            y_i = int(y)
            mo_i = int(mo)
        except Exception:
            continue
        if p_i < 0 or y_i < 1900 or mo_i < 1 or mo_i > 12:
            continue
        by_patch_month_year.setdefault((p_i, mo_i, y_i), []).append(int(i))

    if not by_patch_month_year:
        return pred_map.new_zeros(()), 0

    pred_by_pmy: Dict[Tuple[int, int, int], torch.Tensor] = {}
    mask_by_pmy: Dict[Tuple[int, int, int], torch.Tensor] = {}
    x_by_pmy: Dict[Tuple[int, int, int], torch.Tensor] = {}
    gedi_median_by_pmy: Dict[Tuple[int, int, int], torch.Tensor] = {}
    for key, idxs in by_patch_month_year.items():
        idx_t = torch.as_tensor(idxs, device=pred_s.device, dtype=torch.long)
        pred_by_pmy[key] = pred_s.index_select(0, idx_t).mean(dim=0, keepdim=True)
        mask_by_pmy[key] = aoi_mask.index_select(0, idx_t).all(dim=0, keepdim=True)
        if sample_rh95_median is not None and sample_rh95_median.numel() >= B:
            vals = sample_rh95_median.index_select(0, idx_t)
            vals = vals[torch.isfinite(vals)]
            if vals.numel() > 0:
                gedi_median_by_pmy[key] = vals.median()
        if disturbance_mode and x is not None and x.ndim == 4:
            x_by_pmy[key] = x.index_select(0, idx_t).mean(dim=0, keepdim=True)

    years_by_patch_month: Dict[Tuple[int, int], list[int]] = {}
    for p_i, mo_i, y_i in pred_by_pmy.keys():
        years_by_patch_month.setdefault((int(p_i), int(mo_i)), []).append(int(y_i))

    def _disturbance_from_pair(k0: Tuple[int, int, int], k1: Tuple[int, int, int], out_shape: torch.Size) -> torch.Tensor:
        if not disturbance_mode or k0 not in x_by_pmy or k1 not in x_by_pmy:
            return torch.zeros(out_shape, device=pred_s.device, dtype=torch.bool)
        x0 = x_by_pmy[k0]
        x1 = x_by_pmy[k1]
        votes = torch.zeros(out_shape, device=pred_s.device, dtype=torch.float32)

        # Channel order C13:
        # 0 B02, 1 B03, 2 B04, 3 B08, 4 S1_ASC_VV, 5 S1_ASC_VH,
        # 6 S1_DESC_VV, 7 S1_DESC_VH, 8 PALSAR_HH, 9 PALSAR_HV, 10 AOI, 11 DEM, 12 SLOPE.
        eps = 1e-6
        if int(x0.shape[1]) > 3:
            ndvi0 = (x0[:, 3:4] - x0[:, 2:3]) / (x0[:, 3:4] + x0[:, 2:3] + eps)
            ndvi1 = (x1[:, 3:4] - x1[:, 2:3]) / (x1[:, 3:4] + x1[:, 2:3] + eps)
            votes = votes + ((ndvi0[..., ::stride, ::stride] - ndvi1[..., ::stride, ::stride]) > ndvi_drop_thr).float()
        if int(x0.shape[1]) > 7:
            s1_delta = 0.5 * (
                (x1[:, 5:6] - x0[:, 5:6]).abs() +
                (x1[:, 7:8] - x0[:, 7:8]).abs()
            )
            votes = votes + (s1_delta[..., ::stride, ::stride] > s1_change_thr).float()
        if int(x0.shape[1]) > 9:
            palsar_delta = 0.5 * (
                (x1[:, 8:9] - x0[:, 8:9]).abs() +
                (x1[:, 9:10] - x0[:, 9:10]).abs()
            )
            votes = votes + (palsar_delta[..., ::stride, ::stride] > palsar_change_thr).float()
        return votes >= float(min_signals)

    pair_losses = []
    for (p_i, mo_i), ys in years_by_patch_month.items():
        ys = sorted(set(int(y) for y in ys))
        if len(ys) < 2:
            continue
        for y0, y1 in zip(ys[:-1], ys[1:]):
            dy = int(y1 - y0)
            if dy <= 0:
                continue
            if max_gap > 0 and dy > max_gap:
                continue
            k0 = (int(p_i), int(mo_i), int(y0))
            k1 = (int(p_i), int(mo_i), int(y1))
            h0 = pred_by_pmy[k0]
            h1 = pred_by_pmy[k1]
            aoi_pair = mask_by_pmy[k0] & mask_by_pmy[k1]
            if not bool(aoi_pair.any().item()):
                continue

            # ECHOSAT-inspired GEDI drop guard. If sample-level GEDI RH95 shows an
            # abrupt loss (>4 m, >50%, final height <10 m by default), the decrease
            # penalty is relaxed over the AOI for that same-patch/same-month/year pair.
            gedi_disturbed = torch.zeros(h0.shape, device=pred_s.device, dtype=torch.bool)
            if k0 in gedi_median_by_pmy and k1 in gedi_median_by_pmy:
                y0 = gedi_median_by_pmy[k0]
                y1 = gedi_median_by_pmy[k1]
                abs_drop = y0 - y1
                rel_drop = abs_drop / torch.clamp(y0.abs(), min=1e-6)
                if bool((abs_drop > gedi_drop_m) and (rel_drop > gedi_drop_rel) and (y1 < gedi_final_h)):
                    gedi_disturbed = aoi_pair

            rs_disturbed = _disturbance_from_pair(k0, k1, h0.shape) & aoi_pair
            disturbed = (gedi_disturbed | rs_disturbed) & aoi_pair
            stable = aoi_pair & (~disturbed)

            dh = h1 - h0
            upper = float(gmax * dy)
            lower = float(dmax * dy)
            drop_penalty = torch.relu((-dh) - lower).pow(2)
            jump_penalty = torch.relu(dh - upper).pow(2)

            terms = []
            if bool(stable.any().item()):
                terms.append((drop_penalty[stable] + jump_penalty[stable]).mean())
            if bool(disturbed.any().item()):
                # Real losses are allowed; only unrealistic positive jumps are softly controlled.
                terms.append(float(alpha_disturbed) * jump_penalty[disturbed].mean())
            if terms:
                pair_losses.append(torch.stack(terms).mean())

    if not pair_losses:
        return pred_map.new_zeros(()), 0

    base = torch.stack(pair_losses).mean()
    return (lambda_growth * base).to(dtype=pred_map.dtype), int(len(pair_losses))


# Train
# -------------------------------------------------------------------------------------------------
def train_one_cycle(
    *,
    model,
    loader,
    criterion,
    opt,
    scaler,
    device,
    huber_beta: float,
    report_bins: Sequence[float],
    cls_thresholds: Sequence[float],
    auc_reservoir: int,
    grad_clip: float = 1.0,
    total_steps: Optional[int] = None,
    postfix_every: int = 1,
    augment: bool = True,
    tv_weight: float = 0.0,
    cycle_idx: Optional[int] = None,
    epoch_idx: Optional[int] = None,  # compatibility alias
    phase: int = 1,
    current_lr: float = 0.0,
    best_score: float = float("inf"),
    patience_counter: int = 0,
    patience_limit: int = 20,
):
    """
    Train one step-based cycle.

    A cycle is one validation window of ``total_steps`` mini-batches.
    We keep ``epoch_idx`` as a backward-compatible alias because older callers
    may still pass that keyword.
    """
    model.train()

    reg = RunningReg()
    rep = PerGroupReport(report_bins)
    cls = PerGroupBinaryCls(cls_thresholds, report_bins, auc_reservoir, seed=333)
    occ_regs = {
        "eval15": RunningReg(),
        "ge2": RunningReg(),
        "low_0_2": RunningReg(),
        "primary": RunningReg(),
        "audit_tail": RunningReg(),
    }

    sum_total = 0.0
    sum_reg = 0.0
    sum_cls_aux = 0.0
    n_patches = 0
    n_pts_total = 0
    n_empty_patches_total = 0
    n_skipped_batches = 0
    sum_growth = 0.0
    n_growth_pairs_total = 0

    amp_on = bool(getattr(scaler, "is_enabled", lambda: False)())
    cyc = _resolve_cycle_index(cycle_idx=cycle_idx, epoch_idx=epoch_idx)
    lr_disp = f"{current_lr:.2e}" if current_lr > 0 else "?"
    total_disp = str(int(total_steps)) if total_steps else "?"
    desc = f"[P{phase}|C{cyc:03d}]"
    use_tqdm = _should_use_tqdm()
    print_every = max(1, int(postfix_every)) if use_tqdm else _PROGRESS_PRINT_EVERY_DEFAULT
    cuda_status_printed = False

    it = tqdm(loader, total=total_steps, leave=True, dynamic_ncols=True, desc=desc) if use_tqdm else loader

    for step, batch in enumerate(it, start=1):
        x, y, aux_rows, aux_cols, aux_y, aux_mask, y_sparse, y_mask, y_count, meta = _unpack_batch(batch)

        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        aux_rows = aux_rows.to(device, non_blocking=True).long()
        aux_cols = aux_cols.to(device, non_blocking=True).long()
        aux_y = aux_y.to(device, non_blocking=True).float()
        aux_mask = aux_mask.to(device, non_blocking=True).bool() if aux_mask is not None else None
        y_sparse = y_sparse.to(device, non_blocking=True).float() if y_sparse is not None else None
        y_mask = y_mask.to(device, non_blocking=True).bool() if y_mask is not None else None
        y_count = y_count.to(device, non_blocking=True).float() if y_count is not None else None
        aux_track_id = _as_meta_tensor(meta, "aux_track_id", device=device, dtype=torch.long, like=aux_y, fill_value=-1)

        if (not _SIMPLE_LOGS) and (not cuda_status_printed) and (str(device) == "cuda"):
            _maybe_print_cuda_status(device, prefix=f"CUDA TRAIN C{cyc:03d}")
            cuda_status_printed = True

        fixed_shift_active = str(getattr(criterion, "spatial_loss", "")).strip().lower() == "fixed_track_shift_huber"
        fixed_shift_flip_h = False
        fixed_shift_flip_v = False
        if augment:
            if fixed_shift_active and not _env_flag("CHM_FIXED_SHIFT_ALLOW_AUGMENT", default=False):
                raise RuntimeError(
                    "fixed_track_shift_huber with geometric augmentation is disabled by default. "
                    "Either run the fixed-shift final training without --augment, or set "
                    "CHM_FIXED_SHIFT_ALLOW_AUGMENT=1 so the trainloop transforms the frozen "
                    "shift vector during horizontal/vertical flips."
                )
            if fixed_shift_active:
                x, aux_rows, aux_cols, y_sparse, y_mask, y_count, fixed_shift_flip_h, fixed_shift_flip_v = _augment_batch_sparse_targets(
                    x=x,
                    aux_rows=aux_rows,
                    aux_cols=aux_cols,
                    aux_mask=aux_mask,
                    y_sparse=y_sparse,
                    y_mask=y_mask,
                    y_count=y_count,
                    return_flip_flags=True,
                )
            else:
                x, aux_rows, aux_cols, y_sparse, y_mask, y_count = _augment_batch_sparse_targets(
                    x=x,
                    aux_rows=aux_rows,
                    aux_cols=aux_cols,
                    aux_mask=aux_mask,
                    y_sparse=y_sparse,
                    y_mask=y_mask,
                    y_count=y_count,
                )

        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(amp_on and device.type == "cuda")):
            pred_map = model(x)
            loss_total, loss_reg, loss_cls_aux, n_pts, pred_pts, tgt_pts, n_empty = compute_pixelwise_loss(
                pred_map=pred_map,
                y_center=y,
                aux_rows=aux_rows,
                aux_cols=aux_cols,
                aux_y=aux_y,
                criterion=criterion,
                huber_beta=float(huber_beta),
                tv_weight=float(tv_weight),
                aux_mask=aux_mask,
                y_sparse=y_sparse,
                y_mask=y_mask,
                y_count=y_count,
                aux_track_id=aux_track_id,
                phase=int(phase),
                track_shift_meta=meta,
                track_shift_context={
                    "split": "train",
                    "mode": "train",
                    "phase": int(phase),
                    "cycle_idx": int(cyc),
                    "batch_step": int(step),
                    "flip_h": bool(fixed_shift_flip_h),
                    "flip_v": bool(fixed_shift_flip_v),
                },
            )

            loss_growth, n_growth_pairs = _temporal_growth_consistency_loss(
                pred_map=pred_map,
                x=x,
                meta=meta,
                criterion=criterion,
                phase=int(phase),
            )
            if torch.isfinite(loss_growth):
                loss_total = loss_total + loss_growth
            else:
                loss_growth = pred_map.new_zeros(())
                n_growth_pairs = 0

        if not torch.isfinite(loss_total):
            n_skipped_batches += 1
            continue

        if amp_on and device.type == "cuda":
            scaler.scale(loss_total).backward()
            if grad_clip is not None and float(grad_clip) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(opt)
            scaler.update()
        else:
            loss_total.backward()
            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()

        bs = int(x.size(0))
        n_patches += bs
        n_pts_total += int(n_pts)
        n_empty_patches_total += int(n_empty)

        metric_weight = max(1, int(n_pts))
        sum_total += float(loss_total.item()) * metric_weight
        sum_reg += float(loss_reg.item()) * metric_weight
        sum_cls_aux += float(loss_cls_aux.item()) * metric_weight
        sum_growth += float(loss_growth.item()) * metric_weight
        n_growth_pairs_total += int(n_growth_pairs)

        if n_pts > 0:
            reg.update(pred_pts, tgt_pts)
            rep.update(pred_pts, tgt_pts)
            cls.update(pred_pts, tgt_pts)
            _update_occurrence_domain_regs(occ_regs, pred_pts, tgt_pts)

        if use_tqdm and (step % print_every == 0):
            tmp = reg.compute()
            it.set_postfix({
                "lr": lr_disp,
                "tot": f"{sum_total / max(1, n_pts_total):.4f}",
                "reg": f"{sum_reg / max(1, n_pts_total):.4f}",
                "mae": f"{_sf(tmp.get('mae')):.3f}",
                "rmse": f"{_sf(tmp.get('rmse')):.3f}",
                "r2": f"{_sf(tmp.get('r2')):.3f}",
                "bias": f"{_sf(tmp.get('bias')):.3f}",
                "stdr": f"{_sf(tmp.get('std_ratio')):.3f}",
                "slope": f"{_sf(tmp.get('slope')):.3f}",
                "maxp": f"{_sf(tmp.get('pred_max')):.2f}",
                "maxt": f"{_sf(tmp.get('true_max')):.2f}",
                "corr": f"{_sf(tmp.get('corr')):.3f}",
                "pts": int(n_pts_total),
                "best": f"{best_score:.4f}" if math.isfinite(best_score) else "?",
                "pat": f"{patience_counter}/{patience_limit}",
                "step": f"{step}/{total_disp}",
            })

        elif (not _SIMPLE_LOGS) and ((step % print_every == 0) or (step == 1) or (total_steps is not None and step == int(total_steps))):
            tmp = reg.compute()
            _print_progress_line(
                prefix=desc,
                step=step,
                total_steps=total_steps,
                metrics={
                    "lr": lr_disp,
                    "tot": f"{sum_total / max(1, n_pts_total):.4f}",
                    "reg": f"{sum_reg / max(1, n_pts_total):.4f}",
                    "mae": f"{_sf(tmp.get('mae')):.3f}",
                    "rmse": f"{_sf(tmp.get('rmse')):.3f}",
                    "r2": f"{_sf(tmp.get('r2')):.3f}",
                    "bias": f"{_sf(tmp.get('bias')):.3f}",
                    "stdr": f"{_sf(tmp.get('std_ratio')):.3f}",
                    "slope": f"{_sf(tmp.get('slope')):.3f}",
                    "corr": f"{_sf(tmp.get('corr')):.3f}",
                    "pts": int(n_pts_total),
                    "best": f"{best_score:.4f}" if math.isfinite(best_score) else "?",
                    "pat": f"{patience_counter}/{patience_limit}",
                },
            )

    n_pts_metric = max(1, n_pts_total)
    n_patches_metric = max(1, n_patches)
    _global_reg_tr = reg.compute()
    _global_mean_tr = reg.sum_y / max(1.0, float(reg.n)) if reg.n > 0 else None

    out = {
        "loss_total": sum_total / n_pts_metric,
        "loss_reg": sum_reg / n_pts_metric,
        "loss_mae": sum_reg / n_pts_metric,
        "loss_huber": sum_reg / n_pts_metric,   # compatibility alias kept for 06_train_experiment
        "loss_cls4": sum_cls_aux / n_pts_metric,  # compatibility alias
        "loss_cls_aux": sum_cls_aux / n_pts_metric,
        "loss_growth": sum_growth / n_pts_metric,
        "n_growth_pairs": int(n_growth_pairs_total),
        "avg_pts_per_patch": n_pts_total / n_patches_metric,
        "n_patches": int(n_patches),
        "n_points": int(n_pts_total),
        "n_empty_patches": int(n_empty_patches_total),
        "n_skipped_batches": int(n_skipped_batches),
        **_global_reg_tr,
        **rep.compute(global_mean=_global_mean_tr),
        **cls.compute(),
        **_occurrence_domain_metrics(occ_regs),
    }
    out["n"] = int(out.get("n", out.get("n_points", 0)))
    out["stdr"] = float(out.get("std_ratio", float("nan")))
    out["max_pred"] = float(out.get("pred_max", float("nan")))
    out["max_true"] = float(out.get("true_max", float("nan")))
    if not _SIMPLE_LOGS:
        print(f"[ANTI-SHRINKAGE][TRAIN][P{phase}|C{cyc:03d}] {_anti_shrinkage_brief(out)}", flush=True)

    _print_train_window_summary(
        metrics=out,
        phase=int(phase),
        cycle=int(cyc),
        steps_seen=int(step if 'step' in locals() else 0),
        total_steps=total_steps,
    )
    return out


def train_one_epoch(**kwargs):
    """
    Backward-compatible alias.

    Older callers may still import ``train_one_epoch``. Internally, the loop is
    step-based and the displayed index is now a cycle index (Cxxx), not an epoch
    index (Exxx).
    """
    return train_one_cycle(**kwargs)


# -------------------------------------------------------------------------------------------------
# Eval
# -------------------------------------------------------------------------------------------------
@torch.no_grad()
def eval_one_cycle(
    *,
    model,
    loader,
    criterion,
    device,
    huber_beta: float,
    report_bins: Sequence[float],
    cls_thresholds: Sequence[float],
    auc_reservoir: int,
    total_steps: Optional[int] = None,
    cycle_idx: Optional[int] = None,
    epoch_idx: Optional[int] = None,  # compatibility alias
    phase: int = 1,
):
    """
    Evaluate one step-based cycle.

    A cycle is one validation window of ``total_steps`` mini-batches.
    We keep ``epoch_idx`` as a backward-compatible alias because older callers
    may still pass that keyword.
    """
    if loader is None:
        return {}

    model.eval()

    reg = RunningReg()
    rep = PerGroupReport(report_bins)
    cls = PerGroupBinaryCls(cls_thresholds, report_bins, auc_reservoir, seed=777)
    occ_regs = {
        "eval15": RunningReg(),
        "ge2": RunningReg(),
        "low_0_2": RunningReg(),
        "primary": RunningReg(),
        "audit_tail": RunningReg(),
    }

    sum_total = 0.0
    sum_reg = 0.0
    sum_cls_aux = 0.0
    n_patches = 0
    n_pts_total = 0
    n_empty_patches_total = 0
    n_skipped_batches = 0
    unique_chunks: List[Dict[str, np.ndarray]] = []

    cyc = _resolve_cycle_index(cycle_idx=cycle_idx, epoch_idx=epoch_idx)
    use_tqdm = _should_use_tqdm()
    print_every = _PROGRESS_PRINT_EVERY_DEFAULT
    cuda_status_printed = False
    it = tqdm(loader, total=total_steps, leave=True, dynamic_ncols=True, mininterval=0.2, desc=f"VAL C{cyc:03d}") if use_tqdm else loader

    for step, batch in enumerate(it, start=1):
        x, y, aux_rows, aux_cols, aux_y, aux_mask, y_sparse, y_mask, y_count, meta = _unpack_batch(batch)

        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        aux_rows = aux_rows.to(device, non_blocking=True).long()
        aux_cols = aux_cols.to(device, non_blocking=True).long()
        aux_y = aux_y.to(device, non_blocking=True).float()
        aux_mask = aux_mask.to(device, non_blocking=True).bool() if aux_mask is not None else None
        y_sparse = y_sparse.to(device, non_blocking=True).float() if y_sparse is not None else None
        y_mask = y_mask.to(device, non_blocking=True).bool() if y_mask is not None else None
        y_count = y_count.to(device, non_blocking=True).float() if y_count is not None else None
        aux_track_id = _as_meta_tensor(meta, "aux_track_id", device=device, dtype=torch.long, like=aux_y, fill_value=-1)

        if (not _SIMPLE_LOGS) and (not cuda_status_printed) and (str(device) == "cuda"):
            _maybe_print_cuda_status(device, prefix=f"CUDA VAL C{cyc:03d}")
            cuda_status_printed = True

        pred_map = torch.clamp(model(x), 0.0, 40.0)
        loss_total, loss_reg, loss_cls_aux, n_pts, pred_pts, tgt_pts, n_empty = compute_pixelwise_loss(
            pred_map=pred_map,
            y_center=y,
            aux_rows=aux_rows,
            aux_cols=aux_cols,
            aux_y=aux_y,
            criterion=criterion,
            huber_beta=float(huber_beta),
            tv_weight=0.0,
            aux_mask=aux_mask,
            y_sparse=y_sparse,
            y_mask=y_mask,
            y_count=y_count,
            aux_track_id=aux_track_id,
            phase=int(phase),
            track_shift_meta=meta,
            track_shift_context={
                "split": "val",
                "mode": "eval",
                "phase": int(phase),
                "cycle_idx": int(cyc),
                "batch_step": int(step),
            },
        )

        if not torch.isfinite(loss_total):
            n_skipped_batches += 1
            continue

        bs = int(x.size(0))
        n_patches += bs
        n_pts_total += int(n_pts)
        n_empty_patches_total += int(n_empty)

        metric_weight = max(1, int(n_pts))
        sum_total += float(loss_total.item()) * metric_weight
        sum_reg += float(loss_reg.item()) * metric_weight
        sum_cls_aux += float(loss_cls_aux.item()) * metric_weight

        if n_pts > 0:
            reg.update(pred_pts, tgt_pts)
            rep.update(pred_pts, tgt_pts)
            cls.update(pred_pts, tgt_pts)
            _update_occurrence_domain_regs(occ_regs, pred_pts, tgt_pts)

            unique_chunk = _collect_unique_aux_points_from_batch(
                pred_map=pred_map,
                aux_rows=aux_rows,
                aux_cols=aux_cols,
                aux_y=aux_y,
                aux_mask=aux_mask,
                meta=meta,
            )
            if unique_chunk is not None:
                unique_chunks.append(unique_chunk)

        if use_tqdm and (step % print_every == 0):
            tmp = reg.compute()
            it.set_postfix({
                "tot": f"{sum_total / max(1, n_pts_total):.4f}",
                "reg": f"{sum_reg / max(1, n_pts_total):.4f}",
                "rmse": f"{_sf(tmp.get('rmse')):.3f}",
                "r2": f"{_sf(tmp.get('r2')):.3f}",
                "bias": f"{_sf(tmp.get('bias')):.3f}",
                "stdr": f"{_sf(tmp.get('std_ratio')):.3f}",
                "slope": f"{_sf(tmp.get('slope')):.3f}",
                "maxp": f"{_sf(tmp.get('pred_max')):.2f}",
                "maxt": f"{_sf(tmp.get('true_max')):.2f}",
                "corr": f"{_sf(tmp.get('corr')):.3f}",
            })
        elif (not _SIMPLE_LOGS) and ((step % print_every == 0) or (step == 1) or (total_steps is not None and step == int(total_steps))):
            tmp = reg.compute()
            _print_progress_line(
                prefix=f"[VAL|C{cyc:03d}]",
                step=step,
                total_steps=total_steps,
                metrics={
                    "tot": f"{sum_total / max(1, n_pts_total):.4f}",
                    "reg": f"{sum_reg / max(1, n_pts_total):.4f}",
                    "rmse": f"{_sf(tmp.get('rmse')):.3f}",
                    "r2": f"{_sf(tmp.get('r2')):.3f}",
                    "bias": f"{_sf(tmp.get('bias')):.3f}",
                    "stdr": f"{_sf(tmp.get('std_ratio')):.3f}",
                    "slope": f"{_sf(tmp.get('slope')):.3f}",
                    "corr": f"{_sf(tmp.get('corr')):.3f}",
                    "pts": int(n_pts_total),
                },
            )

    n_pts_metric = max(1, n_pts_total)
    n_patches_metric = max(1, n_patches)
    _global_reg_va = reg.compute()
    _global_mean_va = reg.sum_y / max(1.0, float(reg.n)) if reg.n > 0 else None

    out = {
        "loss_total": sum_total / n_pts_metric,
        "loss_reg": sum_reg / n_pts_metric,
        "loss_mae": sum_reg / n_pts_metric,
        "loss_huber": sum_reg / n_pts_metric,   # compatibility alias kept for 06_train_experiment
        "loss_cls4": sum_cls_aux / n_pts_metric,   # compatibility alias
        "loss_cls_aux": sum_cls_aux / n_pts_metric,
        "avg_pts_per_patch": n_pts_total / n_patches_metric,
        "n_patches": int(n_patches),
        "n_points": int(n_pts_total),
        "n_empty_patches": int(n_empty_patches_total),
        "n_skipped_batches": int(n_skipped_batches),
        **_global_reg_va,
        **rep.compute(global_mean=_global_mean_va),
        **cls.compute(),
        **_occurrence_domain_metrics(occ_regs),
    }
    out.update(_compute_gedi_unique_metrics(unique_chunks))
    out["n"] = int(out.get("n", out.get("n_points", 0)))
    out["stdr"] = float(out.get("std_ratio", float("nan")))
    out["max_pred"] = float(out.get("pred_max", float("nan")))
    out["max_true"] = float(out.get("true_max", float("nan")))
    return out


def eval_one_epoch(**kwargs):
    """
    Backward-compatible alias.

    Older callers may still import ``eval_one_epoch``. Internally, the loop is
    step-based and the displayed index is now a cycle index (Cxxx), not an epoch
    index (Exxx).
    """
    return eval_one_cycle(**kwargs)
