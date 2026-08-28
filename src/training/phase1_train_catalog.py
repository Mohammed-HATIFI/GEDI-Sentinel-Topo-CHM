from __future__ import annotations

# v26_OLDLOSS_CKPTSAFE_MINGAIN_ANTI_ZERO
# Trainer without prediction-floor / anti-collapse loss. It keeps the balanced anti-shrink monitor
# and checkpoint gate support, min-gain best.ckpt replacement, and soft anti-zero loss controls.

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


# === CHM POSITIVE HEIGHT HEAD PATCH v1 ===
class _CHMPositiveHeightOutputWrapper(torch.nn.Module):
    """Non-parametric positive output head for canopy height regression.

    Enabled via env CHM_POSITIVE_HEIGHT_HEAD=1.
    mode=softplus: y_hat = softplus(raw, beta, threshold) + min_height.
    This is part of the model forward used during training/evaluation/inference,
    not a plotting post-processing step.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.mode = str(os.environ.get("CHM_OUTPUT_ACTIVATION", "softplus")).strip().lower()
        self.min_height = float(os.environ.get("CHM_OUTPUT_MIN_HEIGHT", "0.0"))
        self.beta = float(os.environ.get("CHM_OUTPUT_SOFTPLUS_BETA", "1.0"))
        self.threshold = float(os.environ.get("CHM_OUTPUT_SOFTPLUS_THRESHOLD", "20.0"))

    def _activate_tensor(self, x):
        if self.mode in {"off", "none", "raw", "identity"}:
            return x
        if self.mode == "softplus":
            return torch.nn.functional.softplus(x, beta=self.beta, threshold=self.threshold) + self.min_height
        if self.mode == "relu":
            return torch.relu(x) + self.min_height
        raise RuntimeError(f"Unsupported CHM_OUTPUT_ACTIVATION={self.mode!r}. Use softplus/relu/raw.")

    def _activate_output(self, out):
        if torch.is_tensor(out):
            return self._activate_tensor(out)
        if isinstance(out, tuple):
            if len(out) == 0:
                return out
            return (self._activate_output(out[0]), *out[1:])
        if isinstance(out, list):
            if len(out) == 0:
                return out
            return [self._activate_output(out[0]), *out[1:]]
        if isinstance(out, dict):
            d = dict(out)
            for k in ("out", "pred", "prediction", "height", "yhat", "logits"):
                if k in d and torch.is_tensor(d[k]):
                    d[k] = self._activate_tensor(d[k])
                    return d
            for k, v in list(d.items()):
                if torch.is_tensor(v):
                    d[k] = self._activate_tensor(v)
                    return d
            return d
        return out

    def forward(self, *args, **kwargs):
        return self._activate_output(self.base_model(*args, **kwargs))


def _chm_positive_height_head_enabled() -> bool:
    return str(os.environ.get("CHM_POSITIVE_HEIGHT_HEAD", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _chm_apply_positive_height_head(model):
    if not _chm_positive_height_head_enabled():
        return model
    if isinstance(model, _CHMPositiveHeightOutputWrapper):
        return model
    mode = os.environ.get("CHM_OUTPUT_ACTIVATION", "softplus")
    min_h = os.environ.get("CHM_OUTPUT_MIN_HEIGHT", "0.0")
    beta = os.environ.get("CHM_OUTPUT_SOFTPLUS_BETA", "1.0")
    thr = os.environ.get("CHM_OUTPUT_SOFTPLUS_THRESHOLD", "20.0")
    print(f"[MODEL OUTPUT] PositiveHeightHead enabled: activation={mode}, min_height={min_h}, beta={beta}, threshold={thr}", flush=True)
    return _CHMPositiveHeightOutputWrapper(model)
# === END CHM POSITIVE HEIGHT HEAD PATCH v1 ===


class TrueOrdinalRegressionHead(torch.nn.Module):
    """Dual head used by Ablation D/E: continuous CHM + independent ordinal logits.

    The ordinal branch reads the final decoder features immediately before the
    1x1 regression convolution.  It therefore has its own learnable parameters
    and is not reconstructed from the scalar height prediction.
    """

    def __init__(self, base_model: torch.nn.Module, n_thresholds: int):
        super().__init__()
        if not hasattr(base_model, "outc") or not hasattr(base_model, "base_ch"):
            raise TypeError("TrueOrdinalRegressionHead currently requires CanopyHyTecModel")
        self.base_model = base_model
        self.n_thresholds = int(n_thresholds)
        self.ordinal_head = torch.nn.Conv2d(int(base_model.base_ch), self.n_thresholds, kernel_size=1)
        torch.nn.init.kaiming_normal_(self.ordinal_head.weight, mode="fan_out", nonlinearity="relu")
        if self.ordinal_head.bias is not None:
            torch.nn.init.zeros_(self.ordinal_head.bias)
        self._decoder_features = None
        self._feature_hook = self.base_model.outc.register_forward_pre_hook(self._capture_decoder_features)

    def _capture_decoder_features(self, _module, inputs):
        self._decoder_features = inputs[0]

    def forward(self, x: torch.Tensor):
        regression = self.base_model(x)
        if self._decoder_features is None:
            raise RuntimeError("Ordinal head did not receive decoder features")
        ordinal_logits = self.ordinal_head(self._decoder_features)
        return regression, ordinal_logits


class AntiShrinkTransferHead(torch.nn.Module):
    """Residual transfer-learning head for shrinkage correction on top of HyTec.

    The base HyTec checkpoint is kept as the starting predictor.  A lightweight
    feature-conditioned residual head is then learned on decoder features just
    before the original 1x1 output convolution.  This preserves the pretrained
    spatial structure while giving Phase 2 enough flexibility to:

    - increase global slope when predictions are vertically compressed;
    - expand the high-canopy tail with a dedicated high-height branch;
    - correct local under/over-shooting without replacing the full backbone.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        hidden_ch: int = 32,
        gate_bias: float = -2.0,
        high_threshold: float = 20.0,
        init_high_gain: float = 0.0,
        init_global_scale: float = 1.0,
        init_residual_scale: float = 1.0,
        freeze_base: bool = False,
    ):
        super().__init__()
        if not hasattr(base_model, "outc") or not hasattr(base_model, "base_ch"):
            raise TypeError("AntiShrinkTransferHead currently requires CanopyHyTecModel")
        self.base_model = base_model
        self.base_ch = int(base_model.base_ch)
        self.hidden_ch = max(4, int(hidden_ch))
        self.high_threshold = float(high_threshold)
        self.freeze_base = bool(freeze_base)

        self.residual_head = torch.nn.Sequential(
            torch.nn.Conv2d(self.base_ch, self.hidden_ch, kernel_size=3, padding=1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.hidden_ch, 1, kernel_size=1, bias=True),
        )
        self.gate_head = torch.nn.Sequential(
            torch.nn.Conv2d(self.base_ch, self.hidden_ch, kernel_size=1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.hidden_ch, 1, kernel_size=1, bias=True),
        )
        self.high_focus_head = torch.nn.Sequential(
            torch.nn.Conv2d(self.base_ch, self.hidden_ch, kernel_size=1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.hidden_ch, 1, kernel_size=1, bias=True),
        )

        self.global_scale = torch.nn.Parameter(torch.tensor(float(init_global_scale), dtype=torch.float32))
        self.global_bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.residual_scale = torch.nn.Parameter(torch.tensor(float(init_residual_scale), dtype=torch.float32))
        self.high_gain = torch.nn.Parameter(torch.tensor(float(init_high_gain), dtype=torch.float32))

        self._decoder_features = None
        self._feature_hook = self.base_model.outc.register_forward_pre_hook(self._capture_decoder_features)
        self._init_weights(gate_bias=float(gate_bias))

        if self.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def _init_weights(self, gate_bias: float) -> None:
        for seq in (self.residual_head, self.gate_head, self.high_focus_head):
            torch.nn.init.kaiming_normal_(seq[0].weight, mode="fan_out", nonlinearity="relu")
            if seq[0].bias is not None:
                torch.nn.init.zeros_(seq[0].bias)

        torch.nn.init.zeros_(self.residual_head[-1].weight)
        torch.nn.init.zeros_(self.residual_head[-1].bias)

        torch.nn.init.zeros_(self.gate_head[-1].weight)
        torch.nn.init.constant_(self.gate_head[-1].bias, float(gate_bias))

        torch.nn.init.zeros_(self.high_focus_head[-1].weight)
        torch.nn.init.constant_(self.high_focus_head[-1].bias, float(gate_bias) - 2.0)

    def _capture_decoder_features(self, _module, inputs):
        self._decoder_features = inputs[0]

    # PHASE1_TO_TRANSFER_HEAD_EXACT_REMAP_V1
    def load_state_dict(self, state_dict, strict: bool = True):
        """Load either a wrapped Phase-2 state or an unwrapped Phase-1 HyTeC state.

        A Phase-1 checkpoint stores keys such as ``inc.*`` and ``down1.*``.
        This wrapper stores the same tensors under ``base_model.inc.*`` and
        ``base_model.down1.*``.  Passing the Phase-1 dictionary directly to
        ``nn.Module.load_state_dict(..., strict=False)`` silently loads zero
        backbone tensors.  The explicit remap below is therefore mandatory:
        every base-model key must exist and have the expected shape.
        """
        incoming = dict(state_dict)
        if any(str(key).startswith("base_model.") for key in incoming):
            result = super().load_state_dict(incoming, strict=bool(strict))
            print(
                "[TRANSFER LOAD OK] wrapped Phase-2 checkpoint loaded | "
                f"keys={len(incoming)} | strict={bool(strict)}",
                flush=True,
            )
            return result

        expected = self.base_model.state_dict()
        remapped = {}
        shape_mismatches = []
        source_for_target = {}

        for raw_key, tensor in incoming.items():
            key = str(raw_key)
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            if key.startswith("base_model."):
                key = key[len("base_model."):]
            if key not in expected:
                continue
            if tuple(tensor.shape) != tuple(expected[key].shape):
                shape_mismatches.append(
                    f"{raw_key}: checkpoint={tuple(tensor.shape)} expected={tuple(expected[key].shape)}"
                )
                continue
            remapped[key] = tensor
            source_for_target[key] = str(raw_key)

        missing = sorted(set(expected) - set(remapped))
        if missing or shape_mismatches:
            details = [
                "Unsafe Phase-1 -> AntiShrinkTransferHead checkpoint transfer refused.",
                f"matched={len(remapped)}/{len(expected)}",
            ]
            if missing:
                details.append("missing base keys: " + ", ".join(missing[:30]))
            if shape_mismatches:
                details.append("shape mismatches: " + " | ".join(shape_mismatches[:20]))
            raise RuntimeError("\n".join(details))

        result = self.base_model.load_state_dict(remapped, strict=True)
        self._phase1_transfer_loaded = True
        self._phase1_transfer_loaded_keys = len(remapped)
        loaded_numel = sum(int(value.numel()) for value in remapped.values())
        print(
            "[PHASE1->TRANSFER LOAD OK] exact backbone remap verified | "
            f"keys={len(remapped)}/{len(expected)} | tensors_numel={loaded_numel:,} | "
            "new transfer-head parameters kept at identity initialization",
            flush=True,
        )
        return result

    def forward(self, x: torch.Tensor):
        base_pred = self.base_model(x)
        if not torch.is_tensor(base_pred):
            raise TypeError("AntiShrinkTransferHead expects a tensor regression map from the base model")
        if self._decoder_features is None:
            raise RuntimeError("Transfer head did not receive decoder features")

        feats = self._decoder_features
        residual = self.residual_head(feats)
        gate = torch.sigmoid(self.gate_head(feats))
        high_focus = torch.sigmoid(self.high_focus_head(feats))
        high_base = torch.nn.functional.softplus(base_pred - self.high_threshold, beta=1.0, threshold=20.0)

        correction = (self.residual_scale * gate * residual) + (self.high_gain * high_focus * high_base)
        return (self.global_scale * base_pred) + self.global_bias + correction


def _transfer_head_requested(args) -> bool:
    return str(getattr(args, "transfer_head", "off")).strip().lower() not in {"off", "none", "0", "false"}


def _true_ordinal_head_enabled() -> bool:
    return str(os.environ.get("CHM_TRUE_ORDINAL_HEAD", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _temporal_fusion_enabled() -> bool:
    return str(os.environ.get("CHM_TEMPORAL_FUSION", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _simple_logs_enabled() -> bool:
    return str(os.environ.get("CHM_SIMPLE_LOGS", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}

# --------------------------------------------------------------------------------------
# Make project root importable when running:
#   python pipeline\06_train_experiment.py
#
# NOTEBOOK-SAFE:
# In Jupyter/VS Code notebooks, __file__ is not defined. The old version crashed with:
#   NameError: name '__file__' is not defined
# This resolver keeps normal .py execution unchanged, but also works when the file is
# pasted/executed in a notebook cell. You can force the path with CHM_PROJECT_ROOT.
# --------------------------------------------------------------------------------------
def _resolve_project_root() -> Path:
    env_root = os.environ.get("CHM_PROJECT_ROOT") or os.environ.get("PROJECT_ROOT")
    if env_root:
        p = Path(env_root).expanduser()
        if (p / "training").exists():
            return p.resolve()

    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        pass

    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd.parent,
        cwd / "Architecture",
        cwd.parent / "Architecture",
        Path(r"C:\Users\Dell\Desktop\Article_Maroc\Architecture"),
        Path(r"E:\CHM\Architecture"),
    ]
    for cand in candidates:
        try:
            if (cand / "training").exists() and ((cand / "pipeline").exists() or (cand / "Pipeline").exists()):
                return cand.resolve()
            if (cand / "training").exists():
                return cand.resolve()
        except Exception:
            continue

    # Dernier fallback: permet au notebook de continuer, mais il faudra lancer la cellule
    # depuis le dossier racine Architecture ou définir CHM_PROJECT_ROOT.
    return cwd

PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.common import (
    AUC_RESERVOIR_DEFAULT,
    BINS_DEFAULT,
    DEFAULT_BASE_CH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_GRAD_CLIP,
    DEFAULT_NUM_WORKERS,
    DEFAULT_RUNS_ROOT,
    DEFAULT_SEED,
    EARLY_STOPPING_PATIENCE_DEFAULT,
    MAX_AUX_K,
    TV_WEIGHT_DEFAULT,
    USE_AMP_DEFAULT,
    assert_experiment_dirs_exist,
    build_run_paths,
    get_default_device,
    infer_channel_order,
    infer_in_channels,
    infer_patch_size,
    infer_stride,
    infer_temporal_window_days,
    load_experiment_config,
    parse_bins,
    set_seed,
)
from training.data import (
    PatchShardIterable,
    count_samples,
    count_sparse_targets,
    list_experiment_shards,
    preflight_experiment,
)
from training.io import (
    append_csv,
    init_csv,
    load_checkpoint,
    save_checkpoint,
    save_json_artifact,
)
try:
    from training.io import make_cycle_row
except Exception:  # backward compatibility with legacy io.py
    from training.io import make_epoch_row as make_cycle_row
from training.model import (
    CanopyClayTLModel,
    CanopyDinoV3TLModel,
    CanopyDOFATLModel,
    CanopyHyTecModel,
    CanopyPrithviTLModel,
    CanopySatlasTLModel,
    CanopyScaleMAETLModel,
    CanopyTerraMindTLModel,
    CanopyTLModel,
    HyTecLossV6,
)
try:
    from training.trainloop import eval_one_cycle, train_one_cycle
except Exception:  # backward compatibility with legacy trainloop.py
    from training.trainloop import eval_one_epoch as eval_one_cycle, train_one_epoch as train_one_cycle


# ======================================================================================
# Helpers
# ======================================================================================
def _sf(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _import_grad_scaler(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _infer_in_ch_from_first_shard(shards: List[Path]) -> Optional[int]:
    for p in shards:
        try:
            with np.load(p, allow_pickle=False) as z:
                if "X" not in z.files:
                    continue
                X = z["X"]
                if X.ndim != 4:
                    continue
                if X.shape[-1] < 128 and X.shape[1] >= 64 and X.shape[2] >= 64:
                    return int(X.shape[-1])
                if X.shape[1] < 128 and X.shape[2] >= 64 and X.shape[3] >= 64:
                    return int(X.shape[1])
                return int(min(X.shape[1], X.shape[-1]))
        except Exception:
            continue
    return None


def _parse_float_tuple(text: Optional[str], expected_len: Optional[int] = None) -> Optional[Tuple[float, ...]]:
    if text is None:
        return None
    xs = tuple(float(x.strip()) for x in str(text).split(",") if x.strip())
    if expected_len is not None and len(xs) != int(expected_len):
        raise ValueError(f"Expected {expected_len} values, got {len(xs)} from: {text}")
    return xs


def _default_run_name() -> str:
    return "RUN_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def compute_auto_weights_from_shards(
    train_shards: List[Path],
    bins: Tuple[float, ...],
    weights_mode: str = "sqrt_inv",
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    edges = np.asarray(bins, dtype=np.float32)
    n_cls = len(bins) + 1
    counts = np.zeros((n_cls,), dtype=np.int64)
    pos_c = np.zeros((len(bins),), dtype=np.int64)
    neg_c = np.zeros((len(bins),), dtype=np.int64)

    total_valid = 0
    for p in train_shards:
        try:
            with np.load(p, allow_pickle=False) as z:
                if "aux_y" in z.files:
                    aux_y = np.asarray(z["aux_y"], dtype=np.float32)
                    if aux_y.ndim == 1:
                        aux_y = aux_y[None, :]

                    if "aux_mask" in z.files:
                        aux_mask = np.asarray(z["aux_mask"]).astype(bool)
                        if aux_mask.shape == aux_y.shape:
                            y = aux_y[aux_mask]
                        else:
                            y = aux_y[np.isfinite(aux_y)]
                    elif "aux_rows" in z.files:
                        aux_rows = np.asarray(z["aux_rows"])
                        if aux_rows.shape == aux_y.shape:
                            y = aux_y[(aux_rows >= 0) & np.isfinite(aux_y)]
                        else:
                            y = aux_y[np.isfinite(aux_y)]
                    else:
                        y = aux_y[np.isfinite(aux_y)]
                elif "y" in z.files and "y_mask" in z.files:
                    y = np.asarray(z["y"], dtype=np.float32)
                    y_mask = np.asarray(z["y_mask"]).astype(bool)
                    if y.ndim == 4 and y.shape[1] == 1:
                        y = y[:, 0]
                    if y_mask.ndim == 4 and y_mask.shape[1] == 1:
                        y_mask = y_mask[:, 0]
                    y = y[y_mask & np.isfinite(y)]
                elif "y" in z.files:
                    y = np.asarray(z["y"], dtype=np.float32).reshape(-1)
                    y = y[np.isfinite(y)]
                elif "target" in z.files:
                    y = np.asarray(z["target"], dtype=np.float32).reshape(-1)
                    y = y[np.isfinite(y)]
                else:
                    continue
        except Exception:
            continue

        y = np.asarray(y, dtype=np.float32)
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue

        total_valid += int(y.size)

        cls_idx = np.digitize(y, edges, right=False).astype(np.int64)
        counts += np.bincount(cls_idx, minlength=n_cls).astype(np.int64)

        for i, thr in enumerate(edges):
            pos = int((y >= thr).sum())
            neg = int((y < thr).sum())
            pos_c[i] += pos
            neg_c[i] += neg

    if total_valid <= 0:
        counts[:] = 1
        pos_c[:] = 1
        neg_c[:] = 1

    weights_mode = str(weights_mode).strip().lower()
    c = counts.astype(np.float64).clip(min=1.0)

    if weights_mode == "none":
        reg_w_auto = np.ones_like(c, dtype=np.float64)
    elif weights_mode == "inv":
        reg_w_auto = 1.0 / c
        reg_w_auto = reg_w_auto / reg_w_auto.mean()
    else:
        reg_w_auto = 1.0 / np.sqrt(c)
        reg_w_auto = reg_w_auto / reg_w_auto.mean()

    ord_pos_w_auto = tuple(
        float(np.clip(neg_c[i] / max(1.0, float(pos_c[i])), 0.25, 25.0))
        for i in range(len(edges))
    )

    return counts.astype(np.int64), reg_w_auto.astype(np.float32), ord_pos_w_auto


def compute_auto_weights_from_values(
    y: np.ndarray,
    bins: Tuple[float, ...],
    weights_mode: str = "sqrt_inv",
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y = y[np.isfinite(y)]
    edges = np.asarray(bins, dtype=np.float32)
    n_cls = len(bins) + 1
    cls_idx = np.digitize(y, edges, right=False).astype(np.int64)
    counts = np.bincount(cls_idx, minlength=n_cls).astype(np.int64)
    c = counts.astype(np.float64).clip(min=1.0)
    mode = str(weights_mode).strip().lower()
    if mode == "none":
        reg = np.ones_like(c)
    elif mode == "inv":
        reg = 1.0 / c
        reg /= reg.mean()
    else:
        reg = 1.0 / np.sqrt(c)
        reg /= reg.mean()
    ord_pos = []
    for threshold in edges:
        pos = int((y >= threshold).sum())
        neg = int((y < threshold).sum())
        ord_pos.append(float(np.clip(neg / max(1.0, float(pos)), 0.25, 25.0)))
    return counts, reg.astype(np.float32), tuple(ord_pos)


def collect_train_targets_from_shards(train_shards: List[Path]) -> np.ndarray:
    """
    Collect all finite GEDI targets from train shards.

    Priority:
      1) raster sparse supervision: y + y_mask
      2) legacy sparse supervision: aux_y (+ aux_mask / aux_rows if present)
    """
    ys: List[np.ndarray] = []

    for p in train_shards:
        try:
            with np.load(p, allow_pickle=False) as z:
                if "y" in z.files and "y_mask" in z.files:
                    y = np.asarray(z["y"], dtype=np.float32)
                    y_mask = np.asarray(z["y_mask"]).astype(bool)

                    if y.ndim == 4 and y.shape[1] == 1:
                        y = y[:, 0]
                    if y_mask.ndim == 4 and y_mask.shape[1] == 1:
                        y_mask = y_mask[:, 0]

                    vals = y[y_mask & np.isfinite(y)]
                    if vals.size > 0:
                        ys.append(vals.astype(np.float32, copy=False))
                    continue

                if "aux_y" in z.files:
                    aux_y = np.asarray(z["aux_y"], dtype=np.float32)

                    if "aux_mask" in z.files:
                        aux_mask = np.asarray(z["aux_mask"]).astype(bool)
                        if aux_mask.shape == aux_y.shape:
                            vals = aux_y[aux_mask & np.isfinite(aux_y)]
                        else:
                            vals = aux_y[np.isfinite(aux_y)]
                    elif "aux_rows" in z.files:
                        aux_rows = np.asarray(z["aux_rows"])
                        if aux_rows.shape == aux_y.shape:
                            vals = aux_y[(aux_rows >= 0) & np.isfinite(aux_y)]
                        else:
                            vals = aux_y[np.isfinite(aux_y)]
                    else:
                        vals = aux_y[np.isfinite(aux_y)]

                    if vals.size > 0:
                        ys.append(vals.astype(np.float32, copy=False))
        except Exception:
            continue

    if len(ys) == 0:
        return np.empty((0,), dtype=np.float32)

    out = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
    return out[np.isfinite(out)]


def infer_phase2_train_bin_edges(
    y_train: np.ndarray,
    *,
    target_n_classes: int = 3,
    min_points_per_class: int = 64,
    min_bin_width: float = 0.75,
) -> Tuple[float, ...]:
    """
    Precompute stable phase-2 bin edges from TRAIN TARGETS only.
    """
    y = np.asarray(y_train, dtype=np.float32)
    y = y[np.isfinite(y)]

    if y.size < 2:
        return tuple()

    vmin = float(np.min(y))
    vmax = float(np.max(y))
    if (vmax - vmin) < float(min_bin_width):
        return tuple()

    target_n_classes = max(2, int(target_n_classes))
    min_points_per_class = max(1, int(min_points_per_class))
    min_bin_width = max(1e-4, float(min_bin_width))

    max_classes_by_points = max(1, int(y.size // min_points_per_class))
    start_n_classes = min(target_n_classes, max(2, max_classes_by_points))

    def _clean(edges):
        clean = []
        prev = None
        for e in edges:
            ef = float(e)
            if not np.isfinite(ef):
                continue
            if prev is None or (ef - prev) >= min_bin_width:
                clean.append(ef)
                prev = ef
        return tuple(clean)

    for n_classes in range(start_n_classes, 1, -1):
        qs = np.linspace(0.0, 1.0, num=n_classes + 1, dtype=np.float64)[1:-1]
        if qs.size == 0:
            continue
        q_edges = np.quantile(y, qs).tolist()
        edges = _clean(q_edges)
        if len(edges) == (n_classes - 1):
            return edges

    med = float(np.quantile(y, 0.5))
    edges = _clean([med])
    if len(edges) > 0:
        return edges

    mid = 0.5 * (vmin + vmax)
    return _clean([mid])


def _metrics_lower_map(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).strip().lower(): v for k, v in (metrics or {}).items()}


def _candidate_metric_keys(monitor_key: str) -> List[str]:
    """Generate robust aliases for val/test/train-prefixed metric names.

    trainloop.py normally returns metrics without split prefixes, while the top-level
    launcher often uses names such as:
      val_mae_unique_temporal_error_mean_ge2
      test_mae_unique_nearest_ge2

    This helper lets both styles work:
      val_mae_unique_temporal_error_mean_ge2 -> mae_unique_temporal_error_mean_ge2
      test_mae_unique_nearest_ge2            -> mae_unique_nearest_ge2
    """
    k = str(monitor_key).strip().lower()
    candidates = [k]

    for prefix in ("val_", "test_", "train_"):
        if k.startswith(prefix):
            candidates.append(k[len(prefix):])

    # Common explicit split + metric aliases.
    for prefix in ("val_", "test_", "train_"):
        if k.startswith(prefix + "mae_"):
            candidates.append("mae_" + k[len(prefix + "mae_"):])
        if k.startswith(prefix + "rmse_"):
            candidates.append("rmse_" + k[len(prefix + "rmse_"):])
        if k.startswith(prefix + "r2_"):
            candidates.append("r2_" + k[len(prefix + "r2_"):])
        if k.startswith(prefix + "bias_"):
            candidates.append("bias_" + k[len(prefix + "bias_"):])
        if k.startswith(prefix + "n_"):
            candidates.append("n_" + k[len(prefix + "n_"):])

    # Backward aliases if a metric was exported with the shorter "temporal_mean" name.
    more = []
    for c in candidates:
        if "unique_temporal_pred_mean" in c:
            more.append(c.replace("unique_temporal_pred_mean", "unique_temporal_mean"))
        if "unique_temporal_mean" in c:
            more.append(c.replace("unique_temporal_mean", "unique_temporal_pred_mean"))
    candidates.extend(more)

    out: List[str] = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _metric_value(metrics: Dict[str, Any], monitor_key: str) -> float:
    lower = _metrics_lower_map(metrics)
    candidates = _candidate_metric_keys(monitor_key)

    for cand in candidates:
        if cand in lower:
            return float(lower[cand])

    k = str(monitor_key).strip().lower()

    # strict MAE-like monitor
    if k in {"val_mae", "test_mae", "train_mae", "mae"}:
        for cand in ("mae", "loss_mae", "loss_reg", "loss_huber"):
            if cand in lower:
                return float(lower[cand])
        return float("inf")

    # loss-style aliases
    if k in {"val_loss_mae", "loss_mae", "val_loss_reg", "loss_reg", "test_loss_mae", "test_loss_reg"}:
        for cand in ("loss_reg", "loss_mae", "mae", "loss_huber"):
            if cand in lower:
                return float(lower[cand])
        return float("inf")

    if k in {"val_rmse", "test_rmse", "rmse"}:
        return float(lower.get("rmse", float("inf")))
    if k in {"val_mse", "test_mse", "mse"}:
        return float(lower.get("mse", float("inf")))
    if k in {"val_loss_total", "test_loss_total", "loss_total"}:
        return float(lower.get("loss_total", float("inf")))
    if k in {"val_loss_huber", "test_loss_huber", "loss_huber"}:
        return float(lower.get("loss_huber", float("inf")))
    if k in {"val_r2", "test_r2", "r2"}:
        return float(lower.get("r2", float("nan")))

    available = ", ".join(sorted(str(x) for x in metrics.keys())[:80])
    raise KeyError(
        f"Unknown monitor metric: {monitor_key}. "
        f"Tried aliases={candidates}. Available metrics include: {available}"
    )


def _is_better(score: float, best: float, mode: str) -> bool:
    if not math.isfinite(score):
        return False
    mode = str(mode).strip().lower()
    if mode == "max":
        return score > best
    return score < best


def _official_best_replacement_allowed(
    score: float,
    best: float,
    mode: str,
    *,
    min_rel_gain: float = 0.0,
    min_abs_gain: float = 0.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Whether a raw monitor improvement is large enough to replace best.ckpt."""
    details: Dict[str, Any] = {
        "enabled": bool(float(min_rel_gain or 0.0) > 0.0 or float(min_abs_gain or 0.0) > 0.0),
        "min_rel_gain": float(min_rel_gain or 0.0),
        "min_abs_gain": float(min_abs_gain or 0.0),
        "allowed": False,
        "reason": "unknown",
    }
    try:
        c = float(score)
        p = float(best)
        mode_l = str(mode).strip().lower()
        if not math.isfinite(c):
            details["reason"] = "candidate score is not finite"
            return False, details
        if not math.isfinite(p):
            details.update({
                "allowed": True,
                "reason": "no previous official best.ckpt",
                "gain": float("inf"),
                "gain_pct": float("inf"),
                "required_gain": 0.0,
                "required_gain_pct": 0.0,
            })
            return True, details
        gain = (p - c) if mode_l == "min" else (c - p)
        rel_req = abs(p) * max(0.0, float(min_rel_gain or 0.0))
        abs_req = max(0.0, float(min_abs_gain or 0.0))
        required = max(rel_req, abs_req)
        pct = (100.0 * gain / abs(p)) if abs(p) > 1e-12 else float("nan")
        allowed = bool(gain > 0.0 and gain >= required)
        details.update({
            "allowed": allowed,
            "reason": "gain is significant" if allowed else "gain too small to replace official best.ckpt",
            "previous_score": p,
            "candidate_score": c,
            "gain": gain,
            "gain_pct": pct,
            "required_gain": required,
            "required_gain_pct": 100.0 * max(0.0, float(min_rel_gain or 0.0)),
        })
        return allowed, details
    except Exception as exc:
        details["reason"] = f"could not evaluate replacement gain: {exc}"
        return False, details


def _fmt_official_replacement_details(details: Optional[Dict[str, Any]]) -> str:
    if not isinstance(details, dict):
        return "replacement-margin details unavailable"
    reason = str(details.get("reason", "replacement-margin rule"))
    try:
        gain = float(details.get("gain", float("nan")))
        req = float(details.get("required_gain", float("nan")))
        pct = float(details.get("gain_pct", float("nan")))
        req_pct = float(details.get("required_gain_pct", float("nan")))
        if math.isfinite(gain) and math.isfinite(req) and math.isfinite(pct) and math.isfinite(req_pct):
            return f"{reason} | gain={gain:+.6f} ({pct:+.2f}%) < required {req:.6f} ({req_pct:.2f}%)"
        if math.isfinite(gain) and math.isfinite(req):
            return f"{reason} | gain={gain:+.6f} < required {req:.6f}"
    except Exception:
        pass
    return reason


_GROUP_RE = re.compile(
    r"^(?:val_)?g_(lt(?P<lt>-?\d+(?:\.\d+)?)|(?P<lo>-?\d+(?:\.\d+)?)_(?P<hi>-?\d+(?:\.\d+)?)|ge(?P<ge>-?\d+(?:\.\d+)?))_(?P<metric>[a-z0-9_]+)$",
    re.IGNORECASE,
)


def _resolve_phase_monitor(args, phase: int) -> Tuple[str, str]:
    if int(phase) == 1:
        key = str(args.phase1_monitor if args.phase1_monitor else args.monitor)
        mode = str(args.phase1_monitor_mode if args.phase1_monitor_mode else args.monitor_mode)
    else:
        key = str(args.phase2_monitor if args.phase2_monitor else args.monitor)
        mode = str(args.phase2_monitor_mode if args.phase2_monitor_mode else args.monitor_mode)
    return key.strip(), mode.strip().lower()


def _collect_group_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for raw_k, raw_v in metrics.items():
        k = str(raw_k).strip().lower()
        m = _GROUP_RE.match(k)
        if m is None:
            continue

        if m.group("lt") is not None:
            lo = float("-inf")
            hi = float(m.group("lt"))
            spec = f"lt{hi:g}"
        elif m.group("ge") is not None:
            lo = float(m.group("ge"))
            hi = float("inf")
            spec = f"ge{lo:g}"
        else:
            lo = float(m.group("lo"))
            hi = float(m.group("hi"))
            spec = f"{lo:g}_{hi:g}"

        row = groups.setdefault(spec, {"lower": lo, "upper": hi})
        metric_name = str(m.group("metric")).lower()
        try:
            row[metric_name] = float(raw_v)
        except Exception:
            row[metric_name] = raw_v

    rows = list(groups.values())
    rows.sort(key=lambda d: (d.get("lower", float("-inf")), d.get("upper", float("inf"))))
    return rows


def _aggregate_group_metric_above_threshold(
    metrics: Dict[str, Any],
    *,
    base_metric: str,
    threshold: float,
    fallback_to_overlapping: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    rows = _collect_group_rows(metrics)
    if not rows:
        return float("nan"), {"groups_used": []}

    selected = [r for r in rows if math.isfinite(float(r.get("lower", float("nan")))) and float(r["lower"]) >= float(threshold)]
    if len(selected) == 0 and fallback_to_overlapping:
        selected = [
            r for r in rows
            if float(r.get("upper", float("-inf"))) > float(threshold)
            and float(r.get("lower", float("inf"))) < float(r.get("upper", float("-inf")))
        ]

    weighted_sum = 0.0
    weight_sum = 0.0
    groups_used: List[str] = []
    for row in selected:
        value = _sf(row.get(base_metric), default=float("nan"))
        if not math.isfinite(value):
            continue
        count = _sf(row.get("n", row.get("count", row.get("n_points", 1.0))), default=1.0)
        count = max(1.0, float(count))
        weighted_sum += count * value
        weight_sum += count
        lo = row.get("lower", float("nan"))
        hi = row.get("upper", float("nan"))
        if math.isfinite(lo) and math.isfinite(hi):
            groups_used.append(f"{lo:g}-{hi:g}")
        elif math.isfinite(lo):
            groups_used.append(f">={lo:g}")
        elif math.isfinite(hi):
            groups_used.append(f"<{hi:g}")
        else:
            groups_used.append("all")

    if weight_sum <= 0:
        return float("nan"), {"groups_used": groups_used}

    return float(weighted_sum / weight_sum), {"groups_used": groups_used}


def _compute_monitor_score(
    metrics: Dict[str, Any],
    monitor_key: str,
    *,
    phase2_high_threshold: float,
    phase2_global_weight: float,
    phase2_high_weight: float,
) -> Tuple[float, Dict[str, Any]]:
    key = str(monitor_key).strip().lower()

    # dynamic high-threshold monitors driven by grouped validation metrics
    if key in {"val_mae_high", "mae_high"}:
        high_mae, details = _aggregate_group_metric_above_threshold(
            metrics,
            base_metric="mae",
            threshold=float(phase2_high_threshold),
        )
        return high_mae, {"high_mae": high_mae, **details}

    if key in {"val_rmse_high", "rmse_high"}:
        high_rmse, details = _aggregate_group_metric_above_threshold(
            metrics,
            base_metric="rmse",
            threshold=float(phase2_high_threshold),
        )
        return high_rmse, {"high_rmse": high_rmse, **details}

    if key.startswith("val_mae_ge") or key.startswith("mae_ge"):
        thr = float(key.split("ge", 1)[1])
        high_mae, details = _aggregate_group_metric_above_threshold(metrics, base_metric="mae", threshold=thr)
        return high_mae, {"high_mae": high_mae, "threshold": thr, **details}

    if key.startswith("val_rmse_ge") or key.startswith("rmse_ge"):
        thr = float(key.split("ge", 1)[1])
        high_rmse, details = _aggregate_group_metric_above_threshold(metrics, base_metric="rmse", threshold=thr)
        return high_rmse, {"high_rmse": high_rmse, "threshold": thr, **details}

    if key in {"composite_mae_high", "val_composite_mae_high", "phase2_composite_mae_high"}:
        global_mae = _metric_value(metrics, "val_mae")
        high_mae, details = _aggregate_group_metric_above_threshold(
            metrics,
            base_metric="mae",
            threshold=float(phase2_high_threshold),
        )
        if not math.isfinite(high_mae):
            high_mae = global_mae
        gw = max(0.0, float(phase2_global_weight))
        hw = max(0.0, float(phase2_high_weight))
        if (gw + hw) <= 0:
            gw, hw = 1.0, 0.0
        score = (gw * global_mae + hw * high_mae) / (gw + hw)
        return score, {
            "global_mae": float(global_mae),
            "high_mae": float(high_mae),
            "threshold": float(phase2_high_threshold),
            "global_weight": float(gw),
            "high_weight": float(hw),
            **details,
        }

    if key in {"anti_shrinkage_score", "val_anti_shrinkage_score"}:
        global_mae = _metric_value(metrics, "val_mae")
        high_mae, details = _aggregate_group_metric_above_threshold(
            metrics,
            base_metric="mae",
            threshold=float(phase2_high_threshold),
        )
        if not math.isfinite(high_mae):
            high_mae = global_mae

        slope = _sf(metrics.get("slope"), default=float("nan"))
        std_ratio = _sf(metrics.get("std_ratio"), default=float("nan"))
        bias = _sf(metrics.get("bias"), default=float("nan"))

        slope_penalty = abs(1.0 - slope) if math.isfinite(slope) else 1.0
        std_ratio_penalty = abs(1.0 - std_ratio) if math.isfinite(std_ratio) else 1.0
        bias_penalty = abs(bias) if math.isfinite(bias) else 1.0

        # v16 BALANCED anti-shrinkage monitor:
        # - MAE/RMSE/R² remain protected because the dominant terms are MAE and HighMAE.
        # - Slope/std/bias stay as soft anti-shrinkage penalties, while the hard gate below
        #   prevents obviously compressed checkpoints from becoming official best.ckpt.
        w_global_mae = max(0.0, float(phase2_global_weight))
        w_high_mae = max(0.0, float(phase2_high_weight))
        if (w_global_mae + w_high_mae) <= 0:
            w_global_mae, w_high_mae = 0.70, 0.30

        # Balanced calibration penalties: enough to prefer non-shrinked candidates,
        # not so strong that reviewers' MAE/R² degrade unnecessarily.
        w_slope = 0.85
        w_std_ratio = 0.45
        w_bias = 0.20

        score = (
            w_global_mae * float(global_mae)
            + w_high_mae * float(high_mae)
            + w_slope * float(slope_penalty)
            + w_std_ratio * float(std_ratio_penalty)
            + w_bias * float(bias_penalty)
        )

        return score, {
            "global_mae": float(global_mae),
            "high_mae": float(high_mae),
            "threshold": float(phase2_high_threshold),
            "slope": float(slope) if math.isfinite(slope) else float("nan"),
            "std_ratio": float(std_ratio) if math.isfinite(std_ratio) else float("nan"),
            "bias": float(bias) if math.isfinite(bias) else float("nan"),
            "slope_penalty": float(slope_penalty),
            "std_ratio_penalty": float(std_ratio_penalty),
            "bias_penalty": float(bias_penalty),
            "formula": f"anti_shrinkage_score = {w_global_mae:.2f}*val_mae + {w_high_mae:.2f}*val_mae_high + {w_slope:.2f}*|1-slope| + {w_std_ratio:.2f}*|1-std_ratio| + {w_bias:.2f}*|bias|",
            **details,
        }

    if key in {"article_compromise_score", "val_article_compromise_score", "chm_article_compromise_score"}:
        # Article-oriented compromise monitor:
        # - MAE/RMSE/R² remain present in the score.
        # - slope/std/bias can dominate only when the calibration gain is significant.
        # - hard checkpoint gate still blocks non-defensible checkpoints.
        mae = _sf(metrics.get("mae"), default=float("nan"))
        if not math.isfinite(mae):
            mae = _metric_value(metrics, "val_mae")
        rmse = _sf(metrics.get("rmse"), default=float("nan"))
        if not math.isfinite(rmse):
            rmse = mae
        r2 = _sf(metrics.get("r2"), default=float("nan"))
        slope = _sf(metrics.get("slope"), default=float("nan"))
        std_ratio = _sf(metrics.get("std_ratio"), default=float("nan"))
        bias = _sf(metrics.get("bias"), default=float("nan"))

        slope_shortfall = max(0.0, 0.90 - slope) if math.isfinite(slope) else 1.0
        std_ratio_penalty = abs(1.0 - std_ratio) if math.isfinite(std_ratio) else 1.0
        bias_penalty = min(abs(bias), 2.0) if math.isfinite(bias) else 1.0
        r2_penalty = max(0.0, 0.70 - r2) if math.isfinite(r2) else 1.0

        w_mae = 1.00
        w_rmse = 0.25
        w_slope = 2.00
        w_std = 0.75
        w_bias = 0.20
        w_r2 = 0.50

        score = (
            w_mae * float(mae)
            + w_rmse * float(rmse)
            + w_slope * float(slope_shortfall)
            + w_std * float(std_ratio_penalty)
            + w_bias * float(bias_penalty)
            + w_r2 * float(r2_penalty)
        )

        return score, {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2) if math.isfinite(r2) else float("nan"),
            "slope": float(slope) if math.isfinite(slope) else float("nan"),
            "std_ratio": float(std_ratio) if math.isfinite(std_ratio) else float("nan"),
            "bias": float(bias) if math.isfinite(bias) else float("nan"),
            "slope_shortfall": float(slope_shortfall),
            "std_ratio_penalty": float(std_ratio_penalty),
            "bias_penalty": float(bias_penalty),
            "r2_penalty": float(r2_penalty),
            "formula": "article_compromise_score = 1.00*MAE + 0.25*RMSE + 2.00*max(0,0.90-slope) + 0.75*|1-std_ratio| + 0.20*min(|bias|,2) + 0.50*max(0,0.70-r2)",
        }

    if key in {"slope_priority_score", "val_slope_priority_score", "anti_shrinkage_slope_priority_score"}:
        # Publication-oriented monitor for small validation sets:
        # prioritize avoiding vertical compression, while keeping MAE/RMSE reasonable.
        # A checkpoint still must pass the hard gate (e.g. slope >= 0.80, std_ratio <= 1.12).
        global_mae = _metric_value(metrics, "val_mae")
        high_mae, details = _aggregate_group_metric_above_threshold(
            metrics,
            base_metric="mae",
            threshold=float(phase2_high_threshold),
        )
        if not math.isfinite(high_mae):
            high_mae = global_mae

        slope = _sf(metrics.get("slope"), default=float("nan"))
        std_ratio = _sf(metrics.get("std_ratio"), default=float("nan"))
        bias = _sf(metrics.get("bias"), default=float("nan"))

        slope_target_soft = 0.90
        std_ratio_deadband = 0.12
        slope_shortfall = max(0.0, slope_target_soft - slope) if math.isfinite(slope) else 1.0
        std_ratio_excess = max(0.0, abs(1.0 - std_ratio) - std_ratio_deadband) if math.isfinite(std_ratio) else 1.0
        bias_penalty = abs(bias) if math.isfinite(bias) else 1.0

        w_global_mae = 0.50
        w_high_mae = 0.10
        w_slope_shortfall = 4.00
        w_std_ratio_excess = 0.60
        w_bias = 0.15

        score = (
            w_global_mae * float(global_mae)
            + w_high_mae * float(high_mae)
            + w_slope_shortfall * float(slope_shortfall)
            + w_std_ratio_excess * float(std_ratio_excess)
            + w_bias * float(bias_penalty)
        )

        return score, {
            "global_mae": float(global_mae),
            "high_mae": float(high_mae),
            "threshold": float(phase2_high_threshold),
            "slope": float(slope) if math.isfinite(slope) else float("nan"),
            "std_ratio": float(std_ratio) if math.isfinite(std_ratio) else float("nan"),
            "bias": float(bias) if math.isfinite(bias) else float("nan"),
            "slope_target_soft": float(slope_target_soft),
            "std_ratio_deadband": float(std_ratio_deadband),
            "slope_shortfall": float(slope_shortfall),
            "std_ratio_excess": float(std_ratio_excess),
            "bias_penalty": float(bias_penalty),
            "formula": "0.50*val_mae + 0.10*val_mae_high + 4.00*max(0,0.90-slope) + 0.60*max(0,|1-std_ratio|-0.12) + 0.15*|bias|",
            **details,
        }

    score = _metric_value(metrics, monitor_key)
    return float(score), {}


def _checkpoint_eligibility_from_metrics(args, metrics: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Hard gate for best.ckpt to avoid vertically compressed checkpoints.

    When --checkpoint-eligibility-mode=on, a validation checkpoint can improve the
    official best.ckpt only if its vertical calibration passes the requested
    thresholds. A raw best-any checkpoint can still be saved separately for audit.
    """
    mode = str(getattr(args, "checkpoint_eligibility_mode", "off")).strip().lower()
    if mode not in {"on", "true", "yes", "1"}:
        return True, {"enabled": False, "eligible": True, "reasons": []}

    def _get_metric(*names: str, default: float = float("nan")) -> float:
        for name in names:
            if name in metrics:
                v = _sf(metrics.get(name), default=float("nan"))
                if math.isfinite(v):
                    return float(v)
        return float(default)

    gate_domain = str(getattr(args, "checkpoint_gate_domain", os.environ.get("CHM_CKPT_GATE_DOMAIN", "global"))).strip().lower()

    def _gate_metric_names(base: str, *aliases: str) -> Tuple[str, ...]:
        raw = (base, *aliases)
        if gate_domain in {"primary", "audit_tail"}:
            dom = gate_domain
            prefixed: List[str] = []
            for name in raw:
                prefixed.append(f"{name}_occurrence_{dom}")
                prefixed.append(f"val_{name}_{dom}")
            return tuple(prefixed + list(raw))
        return tuple(raw)

    slope = _get_metric(*_gate_metric_names("slope"))
    std_ratio = _get_metric(*_gate_metric_names("std_ratio", "stdr"))
    pred_max = _get_metric(*_gate_metric_names("pred_max", "max_pred"))
    true_max = _get_metric(*_gate_metric_names("true_max", "max_true"))
    bias = _get_metric(*_gate_metric_names("bias"))

    min_slope = float(getattr(args, "checkpoint_min_slope", float("nan")))
    min_std_ratio = float(getattr(args, "checkpoint_min_std_ratio", float("nan")))
    max_std_ratio = float(getattr(args, "checkpoint_max_std_ratio", float("nan")))
    maxpred_under_margin = float(getattr(args, "checkpoint_maxpred_under_margin", float("nan")))
    maxpred_over_margin = float(getattr(args, "checkpoint_maxpred_over_margin", float("nan")))
    max_abs_bias = float(getattr(args, "checkpoint_max_abs_bias", float("nan")))

    reasons: List[str] = []
    if math.isfinite(min_slope) and ((not math.isfinite(slope)) or slope < min_slope):
        reasons.append(f"slope {slope:.3f} < min_slope {min_slope:.3f}")
    if math.isfinite(min_std_ratio) and ((not math.isfinite(std_ratio)) or std_ratio < min_std_ratio):
        reasons.append(f"std_ratio {std_ratio:.3f} < min_std_ratio {min_std_ratio:.3f}")
    if math.isfinite(max_std_ratio) and ((not math.isfinite(std_ratio)) or std_ratio > max_std_ratio):
        reasons.append(f"std_ratio {std_ratio:.3f} > max_std_ratio {max_std_ratio:.3f}")
    if math.isfinite(max_abs_bias) and ((not math.isfinite(bias)) or abs(float(bias)) > max_abs_bias):
        reasons.append(f"abs(bias) {abs(float(bias)) if math.isfinite(bias) else float('nan'):.3f} > max_abs_bias {max_abs_bias:.3f}")
    if math.isfinite(maxpred_under_margin) and math.isfinite(pred_max) and math.isfinite(true_max):
        if pred_max < (true_max - maxpred_under_margin):
            reasons.append(f"pred_max {pred_max:.3f} < true_max-margin {(true_max - maxpred_under_margin):.3f}")
    if math.isfinite(maxpred_over_margin) and math.isfinite(pred_max) and math.isfinite(true_max):
        if pred_max > (true_max + maxpred_over_margin):
            reasons.append(f"pred_max {pred_max:.3f} > true_max+margin {(true_max + maxpred_over_margin):.3f}")

    eligible = len(reasons) == 0
    return eligible, {
        "enabled": True,
        "eligible": bool(eligible),
        "reasons": reasons,
        "slope": float(slope) if math.isfinite(slope) else float("nan"),
        "std_ratio": float(std_ratio) if math.isfinite(std_ratio) else float("nan"),
        "pred_max": float(pred_max) if math.isfinite(pred_max) else float("nan"),
        "true_max": float(true_max) if math.isfinite(true_max) else float("nan"),
        "bias": float(bias) if math.isfinite(bias) else float("nan"),
        "min_slope": min_slope,
        "min_std_ratio": min_std_ratio,
        "max_std_ratio": max_std_ratio,
        "maxpred_under_margin": maxpred_under_margin,
        "maxpred_over_margin": maxpred_over_margin,
        "max_abs_bias": max_abs_bias,
        "gate_domain": gate_domain,
    }


def _make_optimizer(model, lr: float, weight_decay: float, optimizer_name: str = "adamw"):
    optimizer_name = str(optimizer_name).strip().lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )


def _make_schedulers(
    opt,
    *,
    warmup_epochs: int,
    plateau_factor: float,
    plateau_patience: int,
    lr_min: float,
    plateau_mode: str,
):
    n_warmup = max(0, int(warmup_epochs))

    def _warmup_lambda(epoch_idx: int) -> float:
        if n_warmup <= 0:
            return 1.0
        if epoch_idx < n_warmup:
            return float(epoch_idx + 1) / float(n_warmup)
        return 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_warmup_lambda)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode=str(plateau_mode).strip().lower(),
        factor=float(plateau_factor),
        patience=int(plateau_patience),
        min_lr=float(lr_min),
    )
    return warmup_sched, plateau_sched


def _phase_ckpt_paths(paths: Dict[str, Path], phase: int) -> Dict[str, Path]:
    ckdir = Path(paths.get("checkpoints_dir", Path(paths["run_dir"]) / "checkpoints"))
    ckdir.mkdir(parents=True, exist_ok=True)
    return {
        "phase_best": ckdir / f"best_phase{phase}.ckpt",
        "phase_last": ckdir / f"last_phase{phase}.ckpt",
    }


class StepCycleLoader:
    """
    Yield exactly n_steps mini-batches by cycling over a base DataLoader.
    This lets the top-level trainer be step-budgeted without rewriting trainloop.py.
    """
    def __init__(self, base_loader, n_steps: int):
        self.base_loader = base_loader
        self.n_steps = max(0, int(n_steps))

    def __len__(self):
        return int(self.n_steps)

    def __iter__(self):
        if self.n_steps <= 0:
            return iter(())
        produced = 0
        base_it = iter(self.base_loader)
        while produced < self.n_steps:
            try:
                batch = next(base_it)
            except StopIteration:
                base_it = iter(self.base_loader)
                batch = next(base_it)
            produced += 1
            yield batch


# ======================================================================================
# Ablation A — balanced height sampler, data-side only
# ======================================================================================
def _balanced_as_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim >= 2:
        return a.reshape(a.shape[0], -1)
    raise RuntimeError(f"Unsupported sparse aux shape: {a.shape}")


def _balanced_sample_x_to_chw(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 3:
        raise RuntimeError(f"Expected one 3D patch X, got shape={a.shape}")

    # HWC -> CHW
    if a.shape[-1] < 128 and a.shape[0] >= 16 and a.shape[1] >= 16:
        return np.transpose(a, (2, 0, 1)).astype(np.float32, copy=False)

    # Already CHW
    if a.shape[0] < 128 and a.shape[1] >= 16 and a.shape[2] >= 16:
        return a.astype(np.float32, copy=False)

    raise RuntimeError(f"Cannot infer channel layout for X sample shape={a.shape}")


def _balanced_drop_channels_chw(x_chw: np.ndarray, drop_channels: Sequence[int]) -> np.ndarray:
    if not drop_channels:
        return x_chw.astype(np.float32, copy=False)
    drop = {int(i) for i in drop_channels}
    keep = [i for i in range(int(x_chw.shape[0])) if i not in drop]
    return x_chw[keep, :, :].astype(np.float32, copy=False)


def _balanced_valid_aux(aux_y: np.ndarray, aux_mask: Optional[np.ndarray]) -> np.ndarray:
    y = np.asarray(aux_y, dtype=np.float32)
    valid = np.isfinite(y)
    if aux_mask is not None:
        m = np.asarray(aux_mask).astype(bool)
        if m.shape == y.shape:
            valid &= m
    return valid


def _balanced_height_stat(aux_y: np.ndarray, aux_mask: Optional[np.ndarray], mode: str) -> float:
    valid = _balanced_valid_aux(aux_y, aux_mask)
    vals = np.asarray(aux_y, dtype=np.float32)[valid]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    mode_l = str(mode).strip().lower()
    if mode_l == "p90":
        return float(np.nanpercentile(vals, 90))
    if mode_l == "mean":
        return float(np.nanmean(vals))
    return float(np.nanmax(vals))


def _balanced_sparse_maps(
    *,
    H: int,
    W: int,
    aux_rows: np.ndarray,
    aux_cols: np.ndarray,
    aux_y: np.ndarray,
    aux_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_sum = np.zeros((H, W), dtype=np.float32)
    y_count = np.zeros((H, W), dtype=np.float32)

    rows = np.asarray(aux_rows).astype(np.int64)
    cols = np.asarray(aux_cols).astype(np.int64)
    ys = np.asarray(aux_y).astype(np.float32)
    mask = np.asarray(aux_mask).astype(bool)

    valid = (
        mask
        & np.isfinite(ys)
        & (rows >= 0)
        & (cols >= 0)
        & (rows < int(H))
        & (cols < int(W))
    )

    # Multiple GEDI points may fall in the same 10 m pixel. Keep their mean for
    # raster-sparse compatibility, while the aux_* arrays preserve shot identity.
    for r, c, y in zip(rows[valid], cols[valid], ys[valid]):
        y_sum[int(r), int(c)] += float(y)
        y_count[int(r), int(c)] += 1.0

    y_sparse = np.zeros((H, W), dtype=np.float32)
    nonzero = y_count > 0
    y_sparse[nonzero] = y_sum[nonzero] / np.maximum(y_count[nonzero], 1.0)
    y_mask = nonzero.astype(np.bool_)
    return y_sparse, y_mask, y_count


def _balanced_stable_int64_from_object_array(arr: np.ndarray) -> np.ndarray:
    """Convert string/object shot ids to stable positive int64 ids for unique metrics."""
    flat = np.asarray(arr).reshape(-1)
    out = np.zeros((flat.size,), dtype=np.int64)
    for i, v in enumerate(flat):
        if isinstance(v, (np.integer, int)):
            out[i] = int(v)
            continue
        if isinstance(v, (np.floating, float)) and np.isfinite(v):
            out[i] = int(v)
            continue
        s = str(v)
        if s == "" or s.lower() in {"nan", "none"}:
            out[i] = 0
            continue
        digest = hashlib.blake2b(s.encode("utf-8", errors="replace"), digest_size=8).digest()
        out[i] = int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)
    return out.reshape(np.asarray(arr).shape)


_BALANCED_FASTIO_KEYS = (
    "X",
    "aux_y",
    "aux_rows",
    "aux_cols",
    "aux_mask",
    "aux_track_id",
    "aux_shot_id",
    "aux_point_shot_id",
    "aux_shot_uid",
    "aux_gedi_ordinal",
    "aux_temporal_delta_days",
    "aux_abs_temporal_delta_days",
    "aux_lon",
    "aux_lat",
    "aux_source_rowid",
)


class _BalancedCachedShard(dict):
    @property
    def files(self) -> List[str]:
        return list(self.keys())


@lru_cache(maxsize=4)
def _balanced_load_shard_cached(shard_path_str: str) -> _BalancedCachedShard:
    """Decompress each large NPZ shard once, not once per sampled patch."""
    started = time.perf_counter()
    out = _BalancedCachedShard()
    with np.load(shard_path_str, allow_pickle=True) as src:
        for key in _BALANCED_FASTIO_KEYS:
            if key in src.files:
                out[key] = np.asarray(src[key])
    elapsed = time.perf_counter() - started
    x_shape = tuple(out["X"].shape) if "X" in out else None
    print(
        f"[FASTIO NPZ CACHE] loaded once | shard={Path(shard_path_str).name} "
        f"| X={x_shape} | seconds={elapsed:.1f} | cache_entries<=4",
        flush=True,
    )
    return out


def _balanced_get_sample_array(z: Any, key: str, sample_i: int, K: int, *, dtype: Optional[np.dtype] = None) -> Optional[np.ndarray]:
    """Return one per-point metadata vector padded/truncated to exactly K.

    Why this matters: Step05 shards may have different local K_shard values
    (for example 1625 in one shard and 1734 in another). DataLoader's default
    collate stacks tensors, so every sample emitted by BalancedHeightPatchShardIterable
    must have the same 1D metadata length. Padding is masked out later by aux_mask.
    """
    if key not in z.files:
        return None
    arr = _balanced_as_2d(np.asarray(z[key]))
    if sample_i >= int(arr.shape[0]):
        return None

    K = int(K)
    out = arr[sample_i][:K]

    if dtype is not None:
        out = out.astype(dtype, copy=False)
        if int(out.shape[0]) < K:
            pad = K - int(out.shape[0])
            out = np.pad(out, (0, pad), mode="constant", constant_values=0).astype(dtype, copy=False)
    else:
        # Object/string metadata, e.g. aux_shot_id. Use an empty sentinel for
        # padded positions. These positions have aux_mask=False and are ignored.
        if int(out.shape[0]) < K:
            pad = K - int(out.shape[0])
            out_obj = out.astype(object, copy=False)
            fill = np.full((pad,), "", dtype=object)
            out = np.concatenate([out_obj, fill], axis=0)

    return out


class BalancedHeightPatchShardIterable(IterableDataset):
    """Balanced patch sampler for Ablation A.

    This is intentionally data-side only: the loss remains Pauls strict / Huber /
    plain_mae / weights_mode=none. The sampler builds an index of train samples
    from Step05 shards using a patch-level height statistic, then emits samples in
    balanced logical batches across height bins. Validation/test stay unchanged.
    """

    def __init__(
        self,
        shards: Sequence[Path],
        *,
        seed: int = 42,
        in_ch: Optional[int] = None,
        max_aux_k: int = 2048,
        drop_channels: Optional[Sequence[int]] = None,
        height_bins: Sequence[float] = (0.0, 10.0, 20.0, 30.0, 45.0),
        height_stat: str = "max",
        samples_per_epoch: Optional[int] = None,
        batch_size: int = 8,
        verbose: bool = True,
    ):
        super().__init__()
        self.shards = [Path(p) for p in shards]
        self.seed = int(seed)
        self.in_ch = int(in_ch) if in_ch is not None else None
        self.max_aux_k = int(max_aux_k)
        self.drop_channels = [int(x) for x in (drop_channels or [])]
        self.height_bins = np.asarray([float(x) for x in height_bins], dtype=np.float32)
        self.height_stat = str(height_stat).strip().lower()
        self.batch_size = max(1, int(batch_size))
        self.verbose = bool(verbose)

        if self.height_stat not in {"max", "p90", "mean"}:
            raise ValueError("height_stat must be one of: max, p90, mean")
        if self.height_bins.ndim != 1 or self.height_bins.size < 2:
            raise ValueError("height_bins must contain at least two edges")
        if not np.all(np.diff(self.height_bins) > 0):
            raise ValueError(f"height_bins must be strictly increasing, got {self.height_bins.tolist()}")

        self.index_by_bin: Dict[int, List[Tuple[Path, int]]] = {
            i: [] for i in range(int(self.height_bins.size) - 1)
        }
        self._build_index()

        self.valid_bins = [b for b, items in self.index_by_bin.items() if len(items) > 0]
        if not self.valid_bins:
            raise RuntimeError("BalancedHeightPatchShardIterable: no valid sample found in the requested height bins")

        self.n_indexed = int(sum(len(v) for v in self.index_by_bin.values()))
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch is not None else self.n_indexed
        self.samples_per_epoch = int(math.ceil(self.samples_per_epoch / self.batch_size) * self.batch_size)

        if self.verbose:
            print("[TRAIN SAMPLER] mode=balanced_height", flush=True)
            print("[TRAIN SAMPLER] implementation=BalancedHeightPatchShardIterable", flush=True)
            print(f"[TRAIN SAMPLER] height_bins={self.height_bins.tolist()}", flush=True)
            print(f"[TRAIN SAMPLER] height_stat={self.height_stat}", flush=True)
            print(f"[TRAIN SAMPLER] samples_per_epoch={self.samples_per_epoch}", flush=True)
            print(f"[TRAIN SAMPLER] batch_size={self.batch_size}", flush=True)
            print(f"[TRAIN SAMPLER] drop_channels={self.drop_channels}", flush=True)
            print("[TRAIN SAMPLER] fixed aux length: all sparse vectors padded/truncated to max_aux_k", flush=True)
            print("[TRAIN SAMPLER] loss unchanged: Pauls strict / plain_mae / weights_mode=none", flush=True)
            for b in self.valid_bins:
                lo = float(self.height_bins[b])
                hi = float(self.height_bins[b + 1])
                print(f"[TRAIN SAMPLER] bin {b} [{lo:g},{hi:g}) patches={len(self.index_by_bin[b])}", flush=True)

    def _build_index(self) -> None:
        for shard_path in self.shards:
            with np.load(shard_path, allow_pickle=True) as z:
                if "aux_y" not in z.files:
                    continue
                aux_y = _balanced_as_2d(np.asarray(z["aux_y"], dtype=np.float32))
                aux_mask = None
                if "aux_mask" in z.files:
                    aux_mask = _balanced_as_2d(np.asarray(z["aux_mask"]).astype(bool))

                n = int(aux_y.shape[0])
                for i in range(n):
                    mask_i = aux_mask[i] if aux_mask is not None and aux_mask.shape == aux_y.shape else None
                    h = _balanced_height_stat(aux_y[i], mask_i, self.height_stat)
                    if not np.isfinite(h):
                        continue
                    b = int(np.searchsorted(self.height_bins, h, side="right") - 1)
                    if b < 0 or b >= int(self.height_bins.size) - 1:
                        continue
                    self.index_by_bin[b].append((shard_path, int(i)))

    def __len__(self) -> int:
        return int(self.samples_per_epoch)

    def _load_one(self, shard_path: Path, sample_i: int):
        z = _balanced_load_shard_cached(str(shard_path.resolve()))
        if z is not None:
            if "X" not in z.files:
                raise RuntimeError(f"Shard has no X array: {shard_path}")

            X = np.asarray(z["X"])
            x = _balanced_sample_x_to_chw(X[sample_i])
            x = _balanced_drop_channels_chw(x, self.drop_channels)
            C, H, W = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])

            if self.in_ch is not None and C != int(self.in_ch):
                raise RuntimeError(
                    f"BalancedHeightPatchShardIterable channel mismatch after drop: "
                    f"got C={C}, expected in_ch={self.in_ch}, shard={shard_path}, sample={sample_i}"
                )

            aux_y_all = _balanced_as_2d(np.asarray(z["aux_y"], dtype=np.float32))
            aux_rows_all = _balanced_as_2d(np.asarray(z["aux_rows"], dtype=np.int64))
            aux_cols_all = _balanced_as_2d(np.asarray(z["aux_cols"], dtype=np.int64))

            aux_y = aux_y_all[sample_i].astype(np.float32, copy=False)
            aux_rows = aux_rows_all[sample_i].astype(np.int64, copy=False)
            aux_cols = aux_cols_all[sample_i].astype(np.int64, copy=False)

            if "aux_mask" in z.files:
                aux_mask_all = _balanced_as_2d(np.asarray(z["aux_mask"]).astype(bool))
                aux_mask = aux_mask_all[sample_i].astype(np.bool_, copy=False)
            else:
                aux_mask = np.isfinite(aux_y)

            # IMPORTANT: emit fixed-length sparse vectors for every sample.
            # Different Step05 shards can have different local K_shard values
            # (for example 1625 vs 1734). Without padding, DataLoader default_collate
            # crashes with: stack expects each tensor to be equal size.
            K_raw = min(int(self.max_aux_k), int(aux_y.shape[0]))
            K = int(self.max_aux_k)

            aux_y = aux_y[:K_raw]
            aux_rows = aux_rows[:K_raw]
            aux_cols = aux_cols[:K_raw]
            aux_mask = aux_mask[:K_raw]

            if K_raw < K:
                pad = K - K_raw
                aux_y = np.pad(aux_y, (0, pad), mode="constant", constant_values=np.nan).astype(np.float32, copy=False)
                aux_rows = np.pad(aux_rows, (0, pad), mode="constant", constant_values=0).astype(np.int64, copy=False)
                aux_cols = np.pad(aux_cols, (0, pad), mode="constant", constant_values=0).astype(np.int64, copy=False)
                aux_mask = np.pad(aux_mask, (0, pad), mode="constant", constant_values=False).astype(np.bool_, copy=False)

            y_sparse, y_mask, y_count = _balanced_sparse_maps(
                H=H, W=W, aux_rows=aux_rows, aux_cols=aux_cols, aux_y=aux_y, aux_mask=aux_mask
            )

            # Sparse GEDI training does not use the dense target map, but the trainloop
            # expects this placeholder tensor. Keep the historical shape (1,H,W).
            y_center = np.zeros((1, H, W), dtype=np.float32)

            meta: Dict[str, Any] = {}

            aux_track_id = _balanced_get_sample_array(z, "aux_track_id", sample_i, K, dtype=np.int64)
            if aux_track_id is None:
                raise RuntimeError(
                    "BalancedHeightPatchShardIterable requires aux_track_id for Pauls strict track-level loss. "
                    f"Missing key in shard={shard_path}, sample={sample_i}. Rebuild Step05 PAULS_TRACKID."
                )
            meta["aux_track_id"] = torch.from_numpy(aux_track_id.astype(np.int64, copy=False))

            # Use a fixed metadata schema for every sample. This avoids DataLoader
            # default_collate crashes when one shard has optional metadata and another
            # shard does not. Missing values are filled with neutral defaults.
            numeric_meta_defaults: Dict[str, Tuple[np.dtype, float | int]] = {
                "aux_shot_uid": (np.int64, 0),
                "aux_gedi_ordinal": (np.int64, 0),
                "aux_temporal_delta_days": (np.float32, float("nan")),
                "aux_abs_temporal_delta_days": (np.float32, float("nan")),
                "aux_lon": (np.float32, float("nan")),
                "aux_lat": (np.float32, float("nan")),
                "aux_source_rowid": (np.int64, -1),
            }

            for key, (dtype, fill_value) in numeric_meta_defaults.items():
                val = _balanced_get_sample_array(z, key, sample_i, K, dtype=dtype)
                if val is None:
                    val = np.full((K,), fill_value, dtype=dtype)
                meta[key] = torch.from_numpy(val.astype(dtype, copy=False))

            # If only aux_shot_id exists, derive a stable numeric aux_shot_uid so
            # unique-shot validation metrics can still work.
            shot_id_key = "aux_shot_id" if "aux_shot_id" in z.files else "aux_point_shot_id"
            if shot_id_key in z.files:
                raw_shot = _balanced_get_sample_array(z, shot_id_key, sample_i, K, dtype=None)
                if raw_shot is not None:
                    shot_uid = _balanced_stable_int64_from_object_array(raw_shot).astype(np.int64, copy=False)
                    # Prefer an explicit non-zero aux_shot_uid if Step05 already supplied one.
                    existing = meta["aux_shot_uid"].numpy()
                    use_derived = existing == 0
                    existing[use_derived] = shot_uid[use_derived]
                    meta["aux_shot_uid"] = torch.from_numpy(existing.astype(np.int64, copy=False))

            return (
                torch.from_numpy(x.astype(np.float32, copy=False)),
                torch.from_numpy(y_center),
                torch.from_numpy(aux_rows.astype(np.int64, copy=False)),
                torch.from_numpy(aux_cols.astype(np.int64, copy=False)),
                torch.from_numpy(aux_y.astype(np.float32, copy=False)),
                torch.from_numpy(aux_mask.astype(np.bool_, copy=False)),
                torch.from_numpy(y_sparse.astype(np.float32, copy=False)),
                torch.from_numpy(y_mask.astype(np.bool_, copy=False)),
                torch.from_numpy(y_count.astype(np.float32, copy=False)),
                meta,
            )

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        rng = random.Random(self.seed + 1009 * worker_id)

        valid_bins = list(self.valid_bins)
        n_bins = max(1, len(valid_bins))
        produced = 0

        while produced < self.samples_per_epoch:
            batch_records: List[Tuple[Path, int]] = []
            seen: set[Tuple[str, int]] = set()

            for j in range(self.batch_size):
                b = valid_bins[j % n_bins]
                items = self.index_by_bin[b]
                rec = rng.choice(items)

                # Avoid duplicate patch occurrences inside the same logical batch when possible.
                if len(items) > 1:
                    for _ in range(8):
                        key = (str(rec[0]), int(rec[1]))
                        if key not in seen:
                            break
                        rec = rng.choice(items)

                seen.add((str(rec[0]), int(rec[1])))
                batch_records.append(rec)

            rng.shuffle(batch_records)

            for offset, (shard_path, sample_i) in enumerate(batch_records):
                global_pos = produced + offset
                if (global_pos % num_workers) == worker_id:
                    yield self._load_one(shard_path, sample_i)

            produced += self.batch_size


class FixedLengthPatchShardIterable(IterableDataset):
    """Sequential shard iterator with the same fixed-length GEDI metadata contract
    as BalancedHeightPatchShardIterable.

    Why this exists:
    - BalancedHeightPatchShardIterable already pads aux_y/aux_rows/aux_cols/aux_mask
      and, crucially, aux_track_id to max_aux_k.
    - The original PatchShardIterable may omit aux_track_id from meta or fill it with
      -1 during eval, which makes Pauls track-level shifted Huber crash at validation.
    - Validation/test must therefore use this deterministic sequential iterator so
      the trainloop receives valid aux_track_id for every valid GEDI point.
    """

    def __init__(
        self,
        shards: Sequence[Path],
        *,
        seed: int = 42,
        shuffle_shards: bool = False,
        shuffle_within: bool = False,
        in_ch: Optional[int] = None,
        max_aux_k: int = 2048,
        drop_channels: Optional[Sequence[int]] = None,
        verbose: bool = True,
        split_name: str = "eval",
    ):
        super().__init__()
        self.shards = [Path(p) for p in shards]
        self.seed = int(seed)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within = bool(shuffle_within)
        self.in_ch = int(in_ch) if in_ch is not None else None
        self.max_aux_k = int(max_aux_k)
        self.drop_channels = [int(x) for x in (drop_channels or [])]
        self.verbose = bool(verbose)
        self.split_name = str(split_name)

        if self.verbose:
            print(
                f"[EVAL DATASET FIX] split={self.split_name} | implementation=FixedLengthPatchShardIterable | "
                f"max_aux_k={self.max_aux_k} | drop_channels={self.drop_channels} | "
                "aux_track_id preserved/padded for Pauls strict validation",
                flush=True,
            )

    def __len__(self) -> int:
        total = 0
        for shard_path in self.shards:
            with np.load(shard_path, allow_pickle=True) as z:
                if "aux_y" not in z.files:
                    continue
                total += int(np.asarray(z["aux_y"]).shape[0])
        return int(total)

    def _load_one(self, shard_path: Path, sample_i: int):
        # Reuse the already-audited fixed-length loader from the balanced sampler.
        return BalancedHeightPatchShardIterable._load_one(self, shard_path, sample_i)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        rng = random.Random(self.seed + 9973 * worker_id)

        shard_list = list(self.shards)
        if self.shuffle_shards:
            rng.shuffle(shard_list)

        global_pos = 0
        for shard_path in shard_list:
            with np.load(shard_path, allow_pickle=True) as z:
                if "aux_y" not in z.files:
                    raise RuntimeError(f"Shard has no aux_y array: {shard_path}")
                n = int(np.asarray(z["aux_y"]).shape[0])

            indices = list(range(n))
            if self.shuffle_within:
                rng.shuffle(indices)

            for sample_i in indices:
                if (global_pos % num_workers) == worker_id:
                    yield self._load_one(shard_path, int(sample_i))
                global_pos += 1



def _growth_stable_int64_hash(value: Any) -> int:
    """Stable positive int64 hash for patch identifiers passed through DataLoader meta."""
    import hashlib
    raw = str(value).encode("utf-8", errors="replace")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFFFFFFFFFF

class CatalogNpyIterable(IterableDataset):
    """Read one uncompressed NPY image per sample plus Step05 CSV catalogs."""

    def __init__(
        self,
        experiment_root: Path,
        split: str,
        *,
        seed: int,
        in_ch: int,
        max_aux_k: int = 2048,
        drop_channels: Optional[Sequence[int]] = None,
        balanced_height: bool = False,
        batch_size: int = 8,
        samples_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        temporal_fusion: bool = False,
        temporal_dynamic_channels: int = 4,
        temporal_frames: int = 3,
        temporal_window_days: Optional[int] = None,
        temporal_patch_grouping: bool = False,
    ):
        super().__init__()
        self.experiment_root = Path(experiment_root)
        self.split = str(split).lower()
        self.seed = int(seed)
        self.in_ch = int(in_ch)
        self.max_aux_k = int(max_aux_k)
        self.drop_channels = [int(x) for x in (drop_channels or [])]
        self.balanced_height = bool(balanced_height)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.temporal_fusion = bool(temporal_fusion)
        self.temporal_dynamic_channels = int(temporal_dynamic_channels)
        self.temporal_frames = int(temporal_frames)
        if temporal_window_days is None:
            temporal_window_days = 180
            experiment_json = self.experiment_root / "experiment.json"
            if experiment_json.exists():
                try:
                    experiment_cfg = json.loads(experiment_json.read_text(encoding="utf-8"))
                    temporal_window_days = int(
                        experiment_cfg.get("schema", {}).get("temporal_window_days", temporal_window_days)
                    )
                except Exception:
                    pass
        self.temporal_window_days = int(temporal_window_days)
        self.temporal_patch_grouping = bool(temporal_patch_grouping)
        if self.temporal_fusion and self.temporal_frames != 3:
            raise ValueError("Temporal fusion currently uses exactly prev/current/next frames")
        if self.temporal_fusion and self.drop_channels:
            raise ValueError("drop_channels is not supported together with temporal fusion")
        self.shards: List[Path] = []

        samples = pd.read_csv(self.experiment_root / "sample_catalog_step05.csv", low_memory=False)
        shots = pd.read_csv(self.experiment_root / "shot_catalog_step05.csv.gz", low_memory=False)
        self.samples = samples[samples["split"].astype(str).str.lower() == self.split].copy()
        if self.temporal_patch_grouping:
            # Growth Loss compares same patch + same month + consecutive years.
            # Sorting patch/month/year maximizes useful temporal pairs inside a small batch.
            sort_cols = [c for c in ("patch_key", "patch_id", "month", "year", "sample_id") if c in self.samples.columns]
            if sort_cols:
                self.samples = self.samples.sort_values(sort_cols, kind="mergesort")
        self.samples = self.samples.reset_index(drop=True)
        self.shots = shots[shots["split"].astype(str).str.lower() == self.split].copy()
        if self.samples.empty:
            raise RuntimeError(f"CatalogNpyIterable: split={self.split} has no samples")

        self.records = self.samples.to_dict(orient="records")

        def record_year_month(record: Dict[str, Any]) -> Optional[Tuple[int, int]]:
            try:
                year = int(record.get("year"))
                month = int(record.get("month"))
                if year >= 1900 and 1 <= month <= 12:
                    return year, month
            except Exception:
                pass
            match = re.search(r"_Y(\d{4})_M(\d{1,2})(?:_|$)", str(record.get("sample_id", "")))
            if match:
                year, month = int(match.group(1)), int(match.group(2))
                if year >= 1900 and 1 <= month <= 12:
                    return year, month
            return None

        self.temporal_indices: Dict[int, Tuple[int, int, int]] = {}
        if self.temporal_fusion:
            groups: Dict[str, List[int]] = {}
            for i, record in enumerate(self.records):
                groups.setdefault(str(record["patch_id"]), []).append(i)
            for indices in groups.values():
                ordered = sorted(
                    indices,
                    key=lambda j: record_year_month(self.records[j]) or (9999, 12),
                )
                for pos, anchor in enumerate(ordered):
                    self.temporal_indices[anchor] = (
                        ordered[max(0, pos - 1)],
                        anchor,
                        ordered[min(len(ordered) - 1, pos + 1)],
                    )
        self.points_by_sample: Dict[str, Dict[str, np.ndarray]] = {}
        for sample_id, sub in self.shots.groupby("sample_id", sort=False):
            self.points_by_sample[str(sample_id)] = {
                key: sub[key].to_numpy()
                for key in sub.columns
            }

        if self.temporal_fusion:
            # Strict GEDI temporal contract for E.  A neighbouring S2 frame is
            # retained only when every supervised GEDI occurrence for the anchor
            # sample is within +/- temporal_window_days of that frame.  Otherwise
            # the central S2 frame is repeated, preserving every GEDI target.
            strict_indices: Dict[int, Tuple[int, int, int]] = {}
            kept_prev = 0
            kept_next = 0
            repeated_missing_dates = 0
            repeated_missing_sample_dates = 0

            def midpoint(record_index: int) -> Optional[pd.Timestamp]:
                source = self.records[int(record_index)]
                year_month = record_year_month(source)
                if year_month is None:
                    return None
                return pd.Timestamp(year=year_month[0], month=year_month[1], day=15)

            for anchor, (prev_i, current_i, next_i) in self.temporal_indices.items():
                sample_id = str(self.records[int(anchor)]["sample_id"])
                points = self.points_by_sample.get(sample_id, {})
                raw_delta = np.asarray(points.get("aux_temporal_delta_days", []), dtype=np.float64)
                finite_delta = raw_delta[np.isfinite(raw_delta)]
                complete_dates = bool(raw_delta.size > 0 and finite_delta.size == raw_delta.size)
                if not complete_dates:
                    repeated_missing_dates += 1

                anchor_date = midpoint(anchor)
                if anchor_date is None:
                    repeated_missing_sample_dates += 1

                def strict_or_current(candidate: int) -> int:
                    if int(candidate) == int(anchor):
                        return int(anchor)
                    candidate_date = midpoint(candidate)
                    if not complete_dates or anchor_date is None or candidate_date is None:
                        return int(anchor)
                    candidate_offset = float((candidate_date - anchor_date).days)
                    valid_for_all_gedi = np.all(
                        np.abs(finite_delta - candidate_offset) <= float(self.temporal_window_days)
                    )
                    return int(candidate) if bool(valid_for_all_gedi) else int(anchor)

                strict_prev = strict_or_current(prev_i)
                strict_next = strict_or_current(next_i)
                kept_prev += int(strict_prev != int(anchor))
                kept_next += int(strict_next != int(anchor))
                strict_indices[int(anchor)] = (strict_prev, int(current_i), strict_next)

            self.temporal_indices = strict_indices
            n_samples_temporal = max(1, len(self.temporal_indices))
            print(
                f"[TEMPORAL GEDI WINDOW] split={self.split} | window=+/-{self.temporal_window_days}d | "
                f"strict_for_all_anchor_GEDI=yes | prev_kept={kept_prev}/{n_samples_temporal} | "
                f"next_kept={kept_next}/{n_samples_temporal} | "
                f"samples_missing_GEDI_dates={repeated_missing_dates} | "
                f"samples_missing_sample_dates={repeated_missing_sample_dates} | fallback=repeat_current",
                flush=True,
            )

        self.index_by_bin: Dict[int, List[int]] = {}
        if "patch_height_bin" in self.samples.columns:
            for i, value in enumerate(self.samples["patch_height_bin"].astype(int).tolist()):
                self.index_by_bin.setdefault(int(value), []).append(int(i))
        self.valid_bins = sorted(k for k, v in self.index_by_bin.items() if v)
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch is not None else len(self.records)
        if self.balanced_height:
            self.samples_per_epoch = int(math.ceil(self.samples_per_epoch / self.batch_size) * self.batch_size)
            if not self.valid_bins:
                raise RuntimeError("Balanced catalog sampling requested but patch_height_bin is missing/empty")

        self.shot_count_map: Dict[str, int] = {}
        if self.split == "train" and "aux_shot_uid" in self.shots.columns:
            counts = self.shots.groupby("aux_shot_uid").size()
            self.shot_count_map = {str(int(k)): int(v) for k, v in counts.items()}

        self.height_lds_table: Dict[str, Any] = {}
        lds_path = self.experiment_root / "lds_table_train_only.json"
        if lds_path.exists():
            self.height_lds_table = json.loads(lds_path.read_text(encoding="utf-8"))
            self.height_lds_table["edges"] = np.asarray(self.height_lds_table["edges"], dtype=np.float32)
            self.height_lds_table["weights"] = np.asarray(self.height_lds_table["weights"], dtype=np.float32)
            self.height_lds_table["hist"] = np.asarray(self.height_lds_table.get("hist", []), dtype=np.int64)
            self.height_lds_table["smooth"] = np.asarray(self.height_lds_table.get("smooth", []), dtype=np.float32)
            self.height_lds_table["n_train_heights"] = int(
                self.height_lds_table.get("n_train_unique_shots", self.height_lds_table.get("n_train_heights", 0))
            )

        print(
            f"[CATALOG NPY] split={self.split} | samples={len(self.records)} | "
            f"points={len(self.shots)} | balanced={self.balanced_height} | bins={self.valid_bins} | "
            f"temporal_fusion={self.temporal_fusion} | temporal_patch_grouping={self.temporal_patch_grouping}",
            flush=True,
        )

    def __len__(self) -> int:
        return int(self.samples_per_epoch if self.balanced_height else len(self.records))

    def _load_one(self, index: int):
        record = self.records[int(index)]
        sample_id = str(record["sample_id"])

        def load_x(record_index: int) -> np.ndarray:
            source = self.records[int(record_index)]
            x_np = np.load(str(source["x_path"]), mmap_mode="r", allow_pickle=False)
            return _balanced_sample_x_to_chw(np.asarray(x_np, dtype=np.float32))

        if self.temporal_fusion:
            prev_i, current_i, next_i = self.temporal_indices[int(index)]
            frames = [load_x(i) for i in (prev_i, current_i, next_i)]
            dynamic = [frame[: self.temporal_dynamic_channels] for frame in frames]
            static = frames[1][self.temporal_dynamic_channels :]
            x = np.concatenate([*dynamic, static], axis=0)
        else:
            x = _balanced_drop_channels_chw(load_x(int(index)), self.drop_channels)
        if int(x.shape[0]) != self.in_ch:
            raise RuntimeError(f"{sample_id}: got C={x.shape[0]}, expected {self.in_ch}")
        _, H, W = x.shape

        points = self.points_by_sample.get(sample_id, {})
        y_values = np.asarray(points.get("rh95", []), dtype=np.float32)
        finite_y = y_values[np.isfinite(y_values)]
        sample_rh95_median_fallback = float(np.nanmedian(finite_y)) if finite_y.size else float("nan")
        sample_rh95_max_fallback = float(np.nanmax(finite_y)) if finite_y.size else float("nan")

        def _record_float_or_fallback(key: str, fallback: float) -> float:
            try:
                value = float(record.get(key, fallback))
                return value if np.isfinite(value) else float(fallback)
            except Exception:
                return float(fallback)

        rows = np.asarray(points.get("local_row", []), dtype=np.int64)
        cols = np.asarray(points.get("local_col", []), dtype=np.int64)
        n_raw = min(len(y_values), len(rows), len(cols), self.max_aux_k)
        K = self.max_aux_k
        aux_y = np.full((K,), np.nan, dtype=np.float32)
        aux_rows = np.zeros((K,), dtype=np.int64)
        aux_cols = np.zeros((K,), dtype=np.int64)
        aux_mask = np.zeros((K,), dtype=np.bool_)
        if n_raw:
            aux_y[:n_raw] = y_values[:n_raw]
            aux_rows[:n_raw] = rows[:n_raw]
            aux_cols[:n_raw] = cols[:n_raw]
            aux_mask[:n_raw] = np.isfinite(aux_y[:n_raw])

        def padded(key: str, dtype, fill):
            out = np.full((K,), fill, dtype=dtype)
            vals = np.asarray(points.get(key, []), dtype=dtype)
            n = min(len(vals), n_raw)
            if n:
                out[:n] = vals[:n]
            return out

        y_sparse, y_mask, y_count = _balanced_sparse_maps(
            H=H, W=W, aux_rows=aux_rows, aux_cols=aux_cols, aux_y=aux_y, aux_mask=aux_mask
        )
        meta = {
            "aux_track_id": torch.from_numpy(padded("aux_track_id_numeric", np.int64, -1)),
            "aux_shot_uid": torch.from_numpy(padded("aux_shot_uid", np.int64, 0)),
            "aux_gedi_ordinal": torch.from_numpy(padded("aux_gedi_ordinal", np.int64, 0)),
            "aux_temporal_delta_days": torch.from_numpy(padded("aux_temporal_delta_days", np.float32, np.nan)),
            "aux_abs_temporal_delta_days": torch.from_numpy(padded("aux_abs_temporal_delta_days", np.float32, np.nan)),
            "aux_source_rowid": torch.from_numpy(np.arange(K, dtype=np.int64)),
            "aux_shot_weight": torch.from_numpy(padded("aux_shot_weight", np.float32, 1.0)),
            "aux_height_weight": torch.from_numpy(padded("aux_height_weight", np.float32, 1.0)),
            "sample_patch_hash": torch.tensor(
                _growth_stable_int64_hash(record.get("patch_key", record.get("patch_id", sample_id))),
                dtype=torch.long,
            ),
            "sample_year": torch.tensor(int(record.get("year", 0) or 0), dtype=torch.long),
            "sample_month": torch.tensor(int(record.get("month", 0) or 0), dtype=torch.long),
            "sample_rh95_median": torch.tensor(
                _record_float_or_fallback("rh95_median", sample_rh95_median_fallback),
                dtype=torch.float32,
            ),
            "sample_rh95_max": torch.tensor(
                _record_float_or_fallback("rh95_max", sample_rh95_max_fallback),
                dtype=torch.float32,
            ),
        }
        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32).copy()),
            torch.zeros((1, H, W), dtype=torch.float32),
            torch.from_numpy(aux_rows),
            torch.from_numpy(aux_cols),
            torch.from_numpy(aux_y),
            torch.from_numpy(aux_mask),
            torch.from_numpy(y_sparse.astype(np.float32, copy=False)),
            torch.from_numpy(y_mask.astype(np.bool_, copy=False)),
            torch.from_numpy(y_count.astype(np.float32, copy=False)),
            meta,
        )

    def __iter__(self):
        info = get_worker_info()
        worker_id = 0 if info is None else int(info.id)
        num_workers = 1 if info is None else int(info.num_workers)
        rng = random.Random(self.seed + 1009 * worker_id)

        if self.balanced_height:
            produced = 0
            while produced < self.samples_per_epoch:
                batch: List[int] = []
                seen: set[int] = set()
                for j in range(self.batch_size):
                    b = self.valid_bins[j % len(self.valid_bins)]
                    candidates = self.index_by_bin[b]
                    idx = rng.choice(candidates)
                    if len(candidates) > 1:
                        for _ in range(8):
                            if idx not in seen:
                                break
                            idx = rng.choice(candidates)
                    seen.add(idx)
                    batch.append(idx)
                rng.shuffle(batch)
                for offset, idx in enumerate(batch):
                    if ((produced + offset) % num_workers) == worker_id:
                        yield self._load_one(idx)
                produced += self.batch_size
            return

        order = list(range(len(self.records)))
        if self.shuffle:
            rng.shuffle(order)
        for global_pos, idx in enumerate(order):
            if (global_pos % num_workers) == worker_id:
                yield self._load_one(idx)


def _ceil_div(a: int, b: int) -> int:
    a = int(a)
    b = max(1, int(b))
    return (a + b - 1) // b


def _normalize_optional_int(v: Optional[int]) -> Optional[int]:
    if v is None:
        return None
    return int(v)


def _resolve_step_schedule(
    args,
    *,
    nominal_train_steps_per_epoch: int,
) -> Dict[str, int]:
    training_mode = str(getattr(args, "training_mode", "step")).strip().lower()
    step1_epochs = int(args.step1_epochs if args.step1_epochs is not None else args.epochs)
    step2_epochs = max(0, int(args.step2_epochs))

    if training_mode == "epoch":
        # Honest epoch emulation on top of shards: one validation window = one full loader pass.
        step1_max_steps = max(0, int(step1_epochs * nominal_train_steps_per_epoch))
        step2_max_steps = max(0, int(step2_epochs * nominal_train_steps_per_epoch))
        val_every_steps = max(1, int(nominal_train_steps_per_epoch))

        step1_patience_evals = _normalize_optional_int(args.step1_patience_evals)
        if step1_patience_evals is None:
            step1_patience_evals = int(args.step1_patience if args.step1_patience is not None else args.patience)

        step2_patience_evals = _normalize_optional_int(args.step2_patience_evals)
        if step2_patience_evals is None:
            step2_patience_evals = int(args.step2_patience if args.step2_patience is not None else step1_patience_evals)

        return {
            "requested_training_mode": training_mode,
            "effective_training_mode": "epoch",
            "step1_epochs_fallback": int(step1_epochs),
            "step2_epochs_fallback": int(step2_epochs),
            "step1_max_steps": max(0, int(step1_max_steps)),
            "step2_max_steps": max(0, int(step2_max_steps)),
            "val_every_steps": max(1, int(val_every_steps)),
            "step1_patience_evals": max(1, int(step1_patience_evals)),
            "step2_patience_evals": max(1, int(step2_patience_evals)),
            "nominal_train_steps_per_epoch": max(1, int(nominal_train_steps_per_epoch)),
        }

    step1_max_steps = _normalize_optional_int(args.step1_max_steps)
    step2_max_steps = _normalize_optional_int(args.step2_max_steps)

    if step1_max_steps is None:
        if bool(args.forms_t_strict):
            step1_max_steps = 100_000
        else:
            step1_max_steps = int(step1_epochs * nominal_train_steps_per_epoch)

    if step2_max_steps is None:
        if bool(args.forms_t_strict):
            # Keep phase 2 genuinely useful for severe shrinkage instead of a symbolic 20-step touch-up.
            step2_max_steps = 6_400
        else:
            step2_max_steps = int(step2_epochs * nominal_train_steps_per_epoch)

    val_every_steps = _normalize_optional_int(args.val_every_steps)
    if val_every_steps is None or val_every_steps <= 0:
        val_every_steps = 640 if bool(args.forms_t_strict) else int(nominal_train_steps_per_epoch)

    step1_patience_evals = _normalize_optional_int(args.step1_patience_evals)
    if step1_patience_evals is None:
        step1_patience_evals = int(args.step1_patience if args.step1_patience is not None else args.patience)

    step2_patience_evals = _normalize_optional_int(args.step2_patience_evals)
    if step2_patience_evals is None:
        step2_patience_evals = int(args.step2_patience if args.step2_patience is not None else step1_patience_evals)

    return {
        "requested_training_mode": training_mode,
        "effective_training_mode": "step",
        "step1_epochs_fallback": int(step1_epochs),
        "step2_epochs_fallback": int(step2_epochs),
        "step1_max_steps": max(0, int(step1_max_steps)),
        "step2_max_steps": max(0, int(step2_max_steps)),
        "val_every_steps": max(1, int(val_every_steps)),
        "step1_patience_evals": max(1, int(step1_patience_evals)),
        "step2_patience_evals": max(1, int(step2_patience_evals)),
        "nominal_train_steps_per_epoch": max(1, int(nominal_train_steps_per_epoch)),
    }


def _resolve_forms_drop_channels(
    *,
    args,
    channel_order: Sequence[str],
    drop_channels_cli: List[int],
) -> Tuple[List[int], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "forms_auto_drop_applied": False,
        "auto_dropped_channels": [],
        "warning": None,
    }
    if drop_channels_cli:
        return drop_channels_cli, info

    if not bool(args.forms_t_strict):
        return drop_channels_cli, info

    palsar_idx = [i for i, ch in enumerate(channel_order) if str(ch).upper().startswith("PALSAR_")]
    if len(palsar_idx) >= 2:
        info["forms_auto_drop_applied"] = True
        info["auto_dropped_channels"] = list(sorted(palsar_idx))
        return list(sorted(palsar_idx)), info

    info["warning"] = "FORMS strict requested but no obvious PALSAR channels were found to auto-drop."
    return drop_channels_cli, info


def _resolve_phase_for_step(global_step: int, step1_max_steps: int, step2_max_steps: int) -> int:
    if int(global_step) < int(step1_max_steps):
        return 1
    return 2 if int(step2_max_steps) > 0 else 1


def _safe_monitor_value_for_print(score: float) -> str:
    return f"{float(score):.6f}" if math.isfinite(float(score)) else "nan"


def _warn_if_noncomparable_loss_monitor(args, schedule: Dict[str, int]) -> None:
    phase1_key, _ = _resolve_phase_monitor(args, 1)
    if str(phase1_key).strip().lower() not in {"val_loss_total", "loss_total"}:
        return
    if int(schedule.get("step2_max_steps", 0)) <= 0:
        return

    print(
        "[WARN] phase-1 monitor is loss_total. If phase-2 uses a different loss definition/weighting, "
        "prefer val_mae / val_rmse or a phase-2 composite monitor for a fair selection.",
        flush=True,
    )


def _make_log_row_cycle(
    *,
    cycle_index: int,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    time_sec: float,
    lr: float,
    best: float,
    model_type: str,
    is_resume: bool,
    training_mode: str,
    phase: int,
    phase_cycle_index: int,
    global_step: int,
    phase1_steps_done: int,
    phase2_steps_done: int,
    cycle_steps: int,
    step_schedule: Dict[str, Any],
) -> Dict[str, Any]:
    base_kwargs = {
        "cycle_index": int(cycle_index),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "time_sec": float(time_sec),
        "lr": float(lr),
        "best": float(best),
        "model_type": str(model_type),
        "is_resume": bool(is_resume),
    }
    extra_kwargs = {
        "training_mode": str(training_mode),
        "phase": int(phase),
        "phase_cycle_index": int(phase_cycle_index),
        "global_step": int(global_step),
        "phase1_steps_done": int(phase1_steps_done),
        "phase2_steps_done": int(phase2_steps_done),
        "cycle_steps": int(cycle_steps),
        "step_schedule": dict(step_schedule),
    }

    try:
        sig = inspect.signature(make_cycle_row)
        accepted = {k: v for k, v in {**base_kwargs, **extra_kwargs}.items() if k in sig.parameters}
        return make_cycle_row(**accepted)
    except TypeError:
        legacy_kwargs = {
            "epoch": int(cycle_index),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "time_sec": float(time_sec),
            "lr": float(lr),
            "best": float(best),
            "model_type": str(model_type),
            "is_resume": bool(is_resume),
            "training_mode": str(training_mode),
            "phase": int(phase),
            "phase_eval_index": int(phase_cycle_index),
            "global_step": int(global_step),
            "phase1_steps_done": int(phase1_steps_done),
            "phase2_steps_done": int(phase2_steps_done),
            "cycle_steps": int(cycle_steps),
            "step_schedule": dict(step_schedule),
        }
        return make_cycle_row(**legacy_kwargs)


def _print_header(title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def _print_split_metrics(name: str, metrics: Dict[str, Any], print_all_metrics: bool = True) -> None:
    print("\n" + "-" * 110)
    print(f"{name.upper()} METRICS")
    print("-" * 110)

    if not metrics:
        print(f"{name.upper():<5} | no metrics")
        return

    key_order = [
        "loss_total", "loss_reg", "loss_mae", "loss_huber", "loss_cls_aux", "loss_cls4",
        "mae", "rmse", "mse", "r2", "bias", "slope", "corr",
        "avg_pts_per_patch", "n_patches", "n_points", "n_empty_patches", "n_skipped_batches",
        "pred_mean", "pred_std", "pred_min", "pred_max", "true_mean", "true_std", "true_min", "true_max", "std_ratio",
    ]

    printed = set()
    for k in key_order:
        if k in metrics:
            print(f"{k:>28}: {metrics[k]}")
            printed.add(k)

    if print_all_metrics:
        for k in sorted(metrics.keys()):
            if k in printed:
                continue
            v = metrics[k]
            if isinstance(v, (int, float, np.integer, np.floating)):
                print(f"{k:>28}: {v}")


def _print_cycle_block(
    *,
    cycle_index: int,
    phase: int,
    phase_cycle_index: int,
    phase_best: float,
    global_best: float,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    lr: float,
    score: float,
    improved_phase: bool,
    improved_global: bool,
    monitor: str,
    patience_counter: int,
    patience_limit: int,
    print_all_metrics: bool = True,
    global_step: Optional[int] = None,
    total_max_steps: Optional[int] = None,
    phase_steps_done: Optional[int] = None,
    phase_steps_budget: Optional[int] = None,
    cycle_steps: Optional[int] = None,
    monitor_details: Optional[Dict[str, Any]] = None,
) -> None:
    title_bits = [
        f"CYCLE {cycle_index:03d}",
        f"PHASE {phase}",
        f"phase_cycle={phase_cycle_index:03d}",
    ]
    if global_step is not None and total_max_steps is not None:
        title_bits.append(f"global_step={int(global_step):06d}/{int(total_max_steps):06d}")
    if phase_steps_done is not None and phase_steps_budget is not None:
        title_bits.append(f"phase_steps={int(phase_steps_done):06d}/{int(phase_steps_budget):06d}")
    if cycle_steps is not None:
        title_bits.append(f"window_steps={int(cycle_steps):04d}")
    title_bits.extend([
        f"lr={lr:.2e}",
        f"monitor={monitor}",
        f"score={_safe_monitor_value_for_print(score)}",
        f"phase_best={_safe_monitor_value_for_print(phase_best)}",
        f"global_best={_safe_monitor_value_for_print(global_best)}",
    ])

    _print_header(" | ".join(title_bits))
    print(
        f"[STATUS] improved_phase={improved_phase} | improved_global={improved_global} | "
        f"patience={patience_counter}/{patience_limit}"
    )
    _print_monitor_score_breakdown(monitor_details, score)
    _print_split_metrics("train", train_metrics, print_all_metrics=print_all_metrics)
    _print_split_metrics("val", val_metrics, print_all_metrics=print_all_metrics)




def _pretty_sep() -> str:
    return "·" * 120


def _pretty_metric(metrics: Optional[Dict[str, Any]], key: str, default: float = float("nan")) -> float:
    if not metrics:
        return default
    return _sf(metrics.get(key), default=default)


def _pretty_npts(metrics: Optional[Dict[str, Any]]) -> int:
    if not metrics:
        return 0
    try:
        return int(metrics.get("n_points", metrics.get("n", 0)))
    except Exception:
        return 0


def _fmt_monitor_score_value(v: Optional[float]) -> str:
    try:
        x = float(v)
        if math.isfinite(x):
            return f"{x:.6f}"
    except Exception:
        pass
    return "none"


def _fmt_monitor_improvement(current: Optional[float], previous: Optional[float], mode: str) -> str:
    """Readable previous -> current monitor score delta.

    For min monitors, lower is better: improvement = previous - current.
    For max monitors, higher is better: improvement = current - previous.
    """
    try:
        c = float(current)
        p = float(previous)
        if not (math.isfinite(c) and math.isfinite(p)):
            return f"score {_fmt_monitor_score_value(previous)} -> {_fmt_monitor_score_value(current)}"
        mode_l = str(mode).strip().lower()
        gain = (p - c) if mode_l == "min" else (c - p)
        pct = (100.0 * gain / abs(p)) if abs(p) > 1e-12 else float("nan")
        better = "lower is better" if mode_l == "min" else "higher is better"
        if math.isfinite(pct):
            return f"score {p:.6f} -> {c:.6f} | gain={gain:+.6f} ({pct:+.2f}%) | {better}"
        return f"score {p:.6f} -> {c:.6f} | gain={gain:+.6f} | {better}"
    except Exception:
        return f"score {_fmt_monitor_score_value(previous)} -> {_fmt_monitor_score_value(current)}"


def _fmt_metric_delta(current_metrics: Optional[Dict[str, Any]], previous_metrics: Optional[Dict[str, Any]]) -> str:
    if not current_metrics or not previous_metrics:
        return ""
    parts = []
    specs = [
        ("mae", "MAE", "↓"),
        ("rmse", "RMSE", "↓"),
        ("r2", "R²", "↑"),
        ("slope", "slope", "→1"),
        ("std_ratio", "stdr", "→1"),
        ("bias", "bias", "→0"),
        ("pred_max", "max_pred", "safe"),
    ]
    for key, label, target in specs:
        try:
            c = float(current_metrics.get(key))
            p = float(previous_metrics.get(key))
            if not (math.isfinite(c) and math.isfinite(p)):
                continue
            d = c - p
            if key in {"mae", "rmse"}:
                good = d < 0
            elif key == "r2":
                good = d > 0
            elif key in {"slope", "std_ratio"}:
                good = abs(c - 1.0) < abs(p - 1.0)
            elif key == "bias":
                good = abs(c) < abs(p)
            else:
                good = False
            mark = "better" if good else "changed"
            parts.append(f"{label}: {p:.4f}->{c:.4f} ({d:+.4f}, {mark}, target {target})")
        except Exception:
            continue
    return " | ".join(parts)



def _fmt_float_or_nan(v: Any, digits: int = 4) -> str:
    try:
        x = float(v)
        if math.isfinite(x):
            return f"{x:.{digits}f}"
    except Exception:
        pass
    return "nan"


def _print_monitor_score_breakdown(monitor_details: Optional[Dict[str, Any]], score: Optional[float]) -> None:
    """Print only the essential monitor information.

    Compact live format: one line per validation. Detailed formulas/contributions are
    intentionally omitted from the console to keep the training log readable.
    """
    if not isinstance(monitor_details, dict) or not monitor_details:
        return

    formula = str(monitor_details.get("formula", "")).strip()
    score_txt = _fmt_float_or_nan(score, 6)

    if "anti_shrinkage_score" in formula or "0.60*val_mae" in formula:
        mae = _sf(monitor_details.get("global_mae"), default=float("nan"))
        high_mae = _sf(monitor_details.get("high_mae"), default=float("nan"))
        thr = _sf(monitor_details.get("threshold"), default=float("nan"))
        slope = _sf(monitor_details.get("slope"), default=float("nan"))
        stdr = _sf(monitor_details.get("std_ratio"), default=float("nan"))
        bias = _sf(monitor_details.get("bias"), default=float("nan"))
        print(
            f"[MONITOR SCORE] anti_shrinkage={score_txt} | lower=better | "
            f"MAE={_fmt_float_or_nan(mae)} | HighMAE≥{_fmt_float_or_nan(thr, 1)}m={_fmt_float_or_nan(high_mae)} | "
            f"slope={_fmt_float_or_nan(slope)} | stdr={_fmt_float_or_nan(stdr)} | bias={_fmt_float_or_nan(bias)}",
            flush=True,
        )
        return

    if "0.50*val_mae" in formula:
        mae = _sf(monitor_details.get("global_mae"), default=float("nan"))
        high_mae = _sf(monitor_details.get("high_mae"), default=float("nan"))
        thr = _sf(monitor_details.get("threshold"), default=float("nan"))
        slope = _sf(monitor_details.get("slope"), default=float("nan"))
        stdr = _sf(monitor_details.get("std_ratio"), default=float("nan"))
        bias = _sf(monitor_details.get("bias"), default=float("nan"))
        print(
            f"[MONITOR SCORE] slope_priority={score_txt} | lower=better | "
            f"MAE={_fmt_float_or_nan(mae)} | HighMAE≥{_fmt_float_or_nan(thr, 1)}m={_fmt_float_or_nan(high_mae)} | "
            f"slope={_fmt_float_or_nan(slope)} | stdr={_fmt_float_or_nan(stdr)} | bias={_fmt_float_or_nan(bias)}",
            flush=True,
        )
        return

    # Simple monitors such as val_mae or val_rmse_high.
    print(f"[MONITOR SCORE] score={score_txt} | lower=better", flush=True)

def _pretty_metrics_line(label: str, metrics: Optional[Dict[str, Any]]) -> str:
    return (
        f"    {label:<8}: "
        f"mae={_pretty_metric(metrics, 'mae'):.4f} | "
        f"rmse={_pretty_metric(metrics, 'rmse'):.4f} | "
        f"r2={_pretty_metric(metrics, 'r2'):.4f} | "
        f"bias={_pretty_metric(metrics, 'bias'):.4f} | "
        f"slope={_pretty_metric(metrics, 'slope'):.3f} | "
        f"stdr={_pretty_metric(metrics, 'std_ratio'):.3f} | "
        f"pred_std={_pretty_metric(metrics, 'pred_std'):.3f} | "
        f"max_pred={_pretty_metric(metrics, 'pred_max'):.2f} | "
        f"max_true={_pretty_metric(metrics, 'true_max'):.2f} | "
        f"n_pts={_pretty_npts(metrics)}"
    )


def _pretty_occurrence_domain_line(label: str, metrics: Optional[Dict[str, Any]], domain: str) -> Optional[str]:
    """Compact log line for generic occurrence-level evaluation domains."""
    if not metrics:
        return None
    d = str(domain).strip().lower()
    n = _sf(metrics.get(f"n_occurrence_{d}"), default=0.0)
    if not math.isfinite(n) or int(n) <= 0:
        return None
    return (
        f"[{label:<12}] "
        f"mae={_pretty_metric(metrics, f'mae_occurrence_{d}'):.4f} | "
        f"rmse={_pretty_metric(metrics, f'rmse_occurrence_{d}'):.4f} | "
        f"r2={_pretty_metric(metrics, f'r2_occurrence_{d}'):.4f} | "
        f"bias={_pretty_metric(metrics, f'bias_occurrence_{d}'):.4f} | "
        f"slope={_pretty_metric(metrics, f'slope_occurrence_{d}'):.3f} | "
        f"stdr={_pretty_metric(metrics, f'std_ratio_occurrence_{d}'):.3f} | "
        f"max_pred={_pretty_metric(metrics, f'pred_max_occurrence_{d}'):.2f} | "
        f"max_true={_pretty_metric(metrics, f'true_max_occurrence_{d}'):.2f} | "
        f"n_pts={int(n)}"
    )


def _best_ckpt_kept_line(best_step: int, best_phase: int, best_metrics: Optional[Dict[str, Any]]) -> str:
    """Compact status line for the official best.ckpt currently kept."""
    try:
        bs = int(best_step)
        bp = int(best_phase)
    except Exception:
        bs, bp = 0, 0

    if best_metrics is None or bs <= 0 or bp <= 0:
        return "👁️ [BEST CHECKPOINT KEPT] no official best.ckpt yet"

    return (
        f"👁️ [BEST CHECKPOINT KEPT] best_step={bs:05d} | best_phase={bp} | "
        f"mae={_pretty_metric(best_metrics, 'mae'):.4f} | "
        f"rmse={_pretty_metric(best_metrics, 'rmse'):.4f} | "
        f"r2={_pretty_metric(best_metrics, 'r2'):.4f} | "
        f"bias={_pretty_metric(best_metrics, 'bias'):.4f} | "
        f"slope={_pretty_metric(best_metrics, 'slope'):.3f} | "
        f"stdr={_pretty_metric(best_metrics, 'std_ratio'):.3f} | "
        f"max_pred={_pretty_metric(best_metrics, 'pred_max'):.2f}"
    )


def _print_pretty_window_block(
    *,
    global_step: int,
    total_max_steps: int,
    phase: int,
    phase_step: int,
    phase_step_budget: int,
    window_start_step: int,
    window_end_step: int,
    lr: float,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    improved_phase: bool,
    raw_improved_global: bool,
    improved_global: bool,
    checkpoint_eligible: bool,
    checkpoint_gate_details: Optional[Dict[str, Any]],
    patience_reset_reason: str,
    patience_counter: int,
    patience_limit: int,
    best_metrics: Optional[Dict[str, Any]],
    best_step: int,
    best_phase: int,
    raw_best_metrics: Optional[Dict[str, Any]] = None,
    raw_best_step: int = 0,
    raw_best_phase: int = 0,
    best_ckpt_path: Optional[Any] = None,
    monitor_score: Optional[float] = None,
    monitor_details: Optional[Dict[str, Any]] = None,
    official_replacement_details: Optional[Dict[str, Any]] = None,
    prev_phase_best_score: Optional[float] = None,
    prev_global_best_score: Optional[float] = None,
    prev_phase_best_metrics: Optional[Dict[str, Any]] = None,
    prev_global_best_metrics: Optional[Dict[str, Any]] = None,
    monitor_mode: str = "min",
) -> None:
    """Human-readable validation block with explicit separation between:

    1) monitor improvement (training progress / patience reset), and
    2) official best.ckpt save (hard checkpoint-safe gate passed).

    Print semantics:
    - 🌟 [MONITOR IMPROVED] means the monitored validation score improved and
      no_improve/patience was reset to 0. It prints previous score -> new score,
      absolute/percent gain, and key metric deltas.
    - 💾 [BEST CHECKPOINT SAVED] means the current model passed the hard
      checkpoint-safe gate and was written as best.ckpt.
    - 🛑 [CHECKPOINT NOT SAVED] means the raw candidate improved the monitor but
      failed one or more hard checkpoint-safe criteria. It is followed by 👁️ because
      the previously saved official best.ckpt remains kept.
    - ⏳ [NO MONITOR IMPROVEMENT] means no reset; no_improve increments.
    - 🌟 [OFFICIAL BEST IMPROVED] appears when the candidate improves the
      official safe checkpoint reference even if it is not the phase monitor best.
    - 👁️ [BEST CHECKPOINT KEPT] means the previously saved official best.ckpt
      is still the reference checkpoint after this validation, with its key metrics.
    """
    train_loss = _pretty_metric(train_metrics, "loss_total")
    train_reg = _pretty_metric(train_metrics, "loss_reg", default=_pretty_metric(train_metrics, "mae"))
    gate = checkpoint_gate_details if isinstance(checkpoint_gate_details, dict) else {}
    gate_enabled = bool(gate.get("enabled", False))
    gate_reasons = list(gate.get("reasons", []) or [])
    monitor_score_txt = "nan"
    try:
        if monitor_score is not None and math.isfinite(float(monitor_score)):
            monitor_score_txt = f"{float(monitor_score):.6f}"
    except Exception:
        pass

    print(_pretty_sep(), flush=True)
    print(
        f"[TRAIN] step={int(global_step):05d}/{int(phase_step):05d} | "
        f"phase={int(phase)} | lr={float(lr):.2e} | "
        f"loss={train_loss:.4f} | reg_loss={train_reg:.4f} | "
        f"batch_mae={_pretty_metric(train_metrics, 'mae'):.3f} | "
        f"rmse={_pretty_metric(train_metrics, 'rmse'):.3f} | "
        f"r2={_pretty_metric(train_metrics, 'r2'):.3f} | "
        f"slope={_pretty_metric(train_metrics, 'slope'):.3f} | "
        f"stdr={_pretty_metric(train_metrics, 'std_ratio'):.3f} | "
        f"max_pred={_pretty_metric(train_metrics, 'pred_max'):.2f} | "
        f"n_pts={_pretty_npts(train_metrics)}",
        flush=True,
    )
    print(_pretty_sep(), flush=True)
    print(f"VALIDATION @ STEP {int(global_step):05d}", flush=True)
    print(_pretty_sep(), flush=True)
    print(
        f"[TRAIN-WINDOW] steps={int(window_start_step):05d}-{int(window_end_step):05d} | "
        f"mae={_pretty_metric(train_metrics, 'mae'):.4f} | "
        f"rmse={_pretty_metric(train_metrics, 'rmse'):.4f} | "
        f"r2={_pretty_metric(train_metrics, 'r2'):.4f} | "
        f"bias={_pretty_metric(train_metrics, 'bias'):.4f} | "
        f"slope={_pretty_metric(train_metrics, 'slope'):.3f} | "
        f"stdr={_pretty_metric(train_metrics, 'std_ratio'):.3f} | "
        f"max_pred={_pretty_metric(train_metrics, 'pred_max'):.2f} | "
        f"max_true={_pretty_metric(train_metrics, 'true_max'):.2f} | "
        f"n_pts={_pretty_npts(train_metrics)}",
        flush=True,
    )
    print(
        f"[VAL-CURRENT ] step={int(global_step):05d} | phase={int(phase)} | "
        f"phase_step={int(phase_step):05d}/{int(phase_step_budget):05d} | "
        f"mae={_pretty_metric(val_metrics, 'mae'):.4f} | "
        f"rmse={_pretty_metric(val_metrics, 'rmse'):.4f} | "
        f"r2={_pretty_metric(val_metrics, 'r2'):.4f} | "
        f"bias={_pretty_metric(val_metrics, 'bias'):.4f} | "
        f"slope={_pretty_metric(val_metrics, 'slope'):.3f} | "
        f"stdr={_pretty_metric(val_metrics, 'std_ratio'):.3f} | "
        f"pred_std={_pretty_metric(val_metrics, 'pred_std'):.3f} | "
        f"true_std={_pretty_metric(val_metrics, 'true_std'):.3f} | "
        f"max_pred={_pretty_metric(val_metrics, 'pred_max'):.2f} | "
        f"max_true={_pretty_metric(val_metrics, 'true_max'):.2f} | "
        f"n_pts={_pretty_npts(val_metrics)}",
        flush=True,
    )
    _primary_line = _pretty_occurrence_domain_line("VAL-PRIMARY", val_metrics, "primary")
    if _primary_line:
        print(_primary_line, flush=True)
    _audit_line = _pretty_occurrence_domain_line("VAL-AUDIT", val_metrics, "audit_tail")
    if _audit_line:
        print(_audit_line, flush=True)

    if gate_enabled:
        _gate_domain_txt = str(gate.get("gate_domain", getattr(locals().get("args", object()), "checkpoint_gate_domain", "global")))
        if checkpoint_eligible:
            print(f"[CKPT-GATE ] eligible=YES | domain={_gate_domain_txt} | hard checkpoint-safe conditions passed", flush=True)
        else:
            print(
                f"[CKPT-GATE ] eligible=NO  | domain={_gate_domain_txt} | " + ("; ".join(gate_reasons) if gate_reasons else "unknown reason"),
                flush=True,
            )
    else:
        print("[CKPT-GATE ] disabled | any monitor improvement may save best.ckpt", flush=True)

    reset_flag = "YES" if str(patience_reset_reason) != "no_improvement" else "NO"
    print(
        f"[PATIENCE  ] reset={reset_flag} | reason={str(patience_reset_reason)} | "
        f"no_improve={int(patience_counter)}/{int(patience_limit)}",
        flush=True,
    )

    print("", flush=True)
    print(_pretty_sep(), flush=True)

    monitor_improvement_txt = _fmt_monitor_improvement(monitor_score, prev_phase_best_score, monitor_mode)
    monitor_metric_delta_txt = _fmt_metric_delta(val_metrics, prev_phase_best_metrics)
    official_improvement_txt = _fmt_monitor_improvement(monitor_score, prev_global_best_score, monitor_mode)
    official_metric_delta_txt = _fmt_metric_delta(val_metrics, prev_global_best_metrics)

    official_saved = bool(raw_improved_global and checkpoint_eligible and improved_global)
    official_candidate = bool(raw_improved_global)

    if improved_phase:
        print(
            f"🌟 [MONITOR IMPROVED] step={int(global_step):05d} | phase={int(phase)} | "
            f"{monitor_improvement_txt} | no_improve reset to 0",
            flush=True,
        )
        if monitor_metric_delta_txt:
            print(f"   ↳ [MONITOR DELTA] {monitor_metric_delta_txt}", flush=True)
    elif official_saved:
        # The current score may be worse than an earlier unsafe phase candidate,
        # but it is better than the official safe checkpoint reference and passed the gate.
        # Do not print ⏳ here: patience was reset because best.ckpt was updated.
        print(
            f"🌟 [OFFICIAL BEST IMPROVED] step={int(global_step):05d} | phase={int(phase)} | "
            f"official_best {official_improvement_txt} | no_improve reset to 0",
            flush=True,
        )
        if official_metric_delta_txt:
            print(f"   ↳ [OFFICIAL BEST DELTA] {official_metric_delta_txt}", flush=True)
    else:
        print(
            f"⏳ [NO MONITOR IMPROVEMENT] step={int(global_step):05d} | phase={int(phase)} | "
            f"{monitor_improvement_txt} | no_improve={int(patience_counter)}/{int(patience_limit)}",
            flush=True,
        )
        if monitor_metric_delta_txt:
            print(f"   ↳ [MONITOR GAP] {monitor_metric_delta_txt}", flush=True)

    if official_saved:
        ckpt_txt = f" | path={best_ckpt_path}" if best_ckpt_path is not None else ""
        print(
            f"💾 [BEST CHECKPOINT SAVED] step={int(global_step):05d} | phase={int(phase)} | "
            f"official_best {official_improvement_txt} | "
            f"best_step={int(best_step):05d} | best_phase={int(best_phase)}{ckpt_txt}",
            flush=True,
        )
        if official_metric_delta_txt:
            print(f"   ↳ [BEST.CKPT DELTA] {official_metric_delta_txt}", flush=True)
    elif official_candidate and not checkpoint_eligible:
        print(
            f"🛑 [CHECKPOINT NOT SAVED] candidate would update best.ckpt, "
            f"but failed checkpoint-safe gate | " + ("; ".join(gate_reasons) if gate_reasons else "unknown reason"),
            flush=True,
        )
        print(_best_ckpt_kept_line(best_step, best_phase, best_metrics), flush=True)
    elif official_candidate and checkpoint_eligible and not official_saved:
        print(
            "🟨 [CHECKPOINT NOT SAVED] candidate passed checkpoint-safe gate, "
            "but replacement gain is too small | " + _fmt_official_replacement_details(official_replacement_details),
            flush=True,
        )
        print(_best_ckpt_kept_line(best_step, best_phase, best_metrics), flush=True)
    else:
        print(_best_ckpt_kept_line(best_step, best_phase, best_metrics), flush=True)

    print(_pretty_metrics_line("CURRENT", val_metrics), flush=True)
    if isinstance(best_metrics, dict) and best_metrics:
        print(_pretty_metrics_line("BEST-OFFICIAL", best_metrics), flush=True)
    else:
        print("    BEST-OFFICIAL : none (checkpoint gate not passed yet)", flush=True)
    if isinstance(raw_best_metrics, dict) and raw_best_metrics:
        print(
            _pretty_metrics_line(f"BEST-COMPOSITE@{int(raw_best_step):05d}", raw_best_metrics),
            flush=True,
        )
    _print_monitor_score_breakdown(monitor_details, monitor_score)
    print(_pretty_sep(), flush=True)

def _print_test_block(test_metrics: Dict[str, Any], print_all_metrics: bool = True) -> None:
    _print_header("FINAL TEST METRICS")
    _print_split_metrics("test", test_metrics, print_all_metrics=print_all_metrics)


def _fmt_loss_value(v: Any) -> str:
    try:
        x = float(v)
        if math.isfinite(x):
            return f"{x:.4g}"
    except Exception:
        pass
    return str(v)


def _print_loss_block(args) -> None:
    loss_name = str(getattr(args, "regression_loss", "mae")).strip().lower()

    line1 = [f"regression_loss={loss_name}"]
    if loss_name == "huber":
        line1.append(f"huber_beta={_fmt_loss_value(getattr(args, 'huber_beta', 'N/A'))}")

    print(f"[LOSS] {' | '.join(line1)}")

    line2 = [
        f"alpha_reg={_fmt_loss_value(getattr(args, 'alpha_reg', 'N/A'))}",
        f"beta_ord={_fmt_loss_value(getattr(args, 'beta_ord', 'N/A'))}",
        f"gamma_cls={_fmt_loss_value(getattr(args, 'gamma_cls', 'N/A'))}",
        f"tv_weight={_fmt_loss_value(getattr(args, 'tv_weight', 'N/A'))}",
        f"weights_mode={getattr(args, 'weights_mode', 'N/A')}",
        f"patch_weight_mode={getattr(args, 'patch_weight_mode', 'N/A')}",
    ]
    print(f"[LOSS] {' | '.join(line2)}")

    phase1_bits = [
        f"phase1_strategy={getattr(args, 'phase1_rebalance_strategy', 'plain_mae')}",
        f"phase1_bin_edges={getattr(args, 'phase1_bin_edges', None)}",
        f"phase1_class_weights={getattr(args, 'phase1_class_weights', None)}",
        f"phase1_lambda_slope={_fmt_loss_value(getattr(args, 'phase1_lambda_slope_loss', 0.0))}",
        f"phase1_lambda_std={_fmt_loss_value(getattr(args, 'phase1_lambda_std_loss', 0.0))}",
        f"phase1_lambda_bias={_fmt_loss_value(getattr(args, 'phase1_lambda_bias_loss', 0.0))}",
        f"phase1_lambda_antizero={_fmt_loss_value(getattr(args, 'phase1_lambda_anti_zero_loss', 0.0))}",
    ]
    print(f"[LOSS] {' | '.join(phase1_bits)}")

    phase2_bits = [
        f"phase2_strategy={getattr(args, 'phase2_rebalance_strategy', 'N/A')}",
        f"phase2_bin_edges={getattr(args, 'phase2_bin_edges', None)}",
        f"phase2_class_weights={getattr(args, 'phase2_class_weights', None)}",
        f"phase2_lambda_slope={_fmt_loss_value(getattr(args, 'phase2_lambda_slope_loss', 0.0))}",
        f"phase2_lambda_std={_fmt_loss_value(getattr(args, 'phase2_lambda_std_loss', 0.0))}",
        f"phase2_lambda_bias={_fmt_loss_value(getattr(args, 'phase2_lambda_bias_loss', 0.0))}",
        f"phase2_lambda_antizero={_fmt_loss_value(getattr(args, 'phase2_lambda_anti_zero_loss', 0.0))}",
    ]
    if getattr(args, 'phase2_beta_ord', None) is not None:
        phase2_bits.append(f"phase2_beta_ord={_fmt_loss_value(args.phase2_beta_ord)}")
    if getattr(args, 'phase2_gamma_cls', None) is not None:
        phase2_bits.append(f"phase2_gamma_cls={_fmt_loss_value(args.phase2_gamma_cls)}")
    print(f"[LOSS] {' | '.join(phase2_bits)}")

    asym_bits = [
        f"p1_low_over_w={_fmt_loss_value(getattr(args, 'phase1_asym_low_over_weight', 1.0))}",
        f"p1_high_under_w={_fmt_loss_value(getattr(args, 'phase1_asym_high_under_weight', 1.0))}",
        f"p1_vhigh_under_w={_fmt_loss_value(getattr(args, 'phase1_asym_very_high_under_weight', 1.0))}",
        f"p2_low_over_w={_fmt_loss_value(getattr(args, 'phase2_asym_low_over_weight', 1.0))}",
        f"p2_high_under_w={_fmt_loss_value(getattr(args, 'phase2_asym_high_under_weight', 1.0))}",
        f"p2_vhigh_under_w={_fmt_loss_value(getattr(args, 'phase2_asym_very_high_under_weight', 1.0))}",
    ]
    print(f"[LOSS] {' | '.join(asym_bits)}")

    anti_zero_bits = [
        f"phase1_lambda_anti_zero={_fmt_loss_value(getattr(args, 'phase1_lambda_anti_zero_loss', 0.0))}",
        f"phase2_lambda_anti_zero={_fmt_loss_value(getattr(args, 'phase2_lambda_anti_zero_loss', 0.0))}",
        f"min_target_y={_fmt_loss_value(getattr(args, 'anti_zero_min_target_y', 2.5))}",
        f"min_pred={_fmt_loss_value(getattr(args, 'anti_zero_min_pred', 2.0))}",
        f"power={_fmt_loss_value(getattr(args, 'anti_zero_power', 2.0))}",
        f"min_points={getattr(args, 'anti_zero_min_points', 1)}",
    ]
    print(f"[ANTI-ZERO LOSS] {' | '.join(anti_zero_bits)}")


def _print_header_block(
    args,
    exp_cfg: Dict[str, Any],
    train_shards: List[Path],
    val_shards: List[Path],
    test_shards: List[Path],
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_train_sparse: int,
    n_val_sparse: int,
    n_test_sparse: int,
    counts: np.ndarray,
    reg_w_auto: np.ndarray,
    ord_pw_auto: Tuple[float, ...],
    reg_w_used: Tuple[float, ...],
    ord_pw_used: Tuple[float, ...],
    cls_w_used: Tuple[float, ...],
    in_ch: int,
    channel_order: Sequence[str],
    patch_size: int,
    stride: int,
    temporal_window_days: int,
    device: torch.device,
    run_dir: Path,
    schedule: Dict[str, int],
    optimizer_name: str,
    phase2_train_bin_edges: Tuple[float, ...],
    n_train_targets_for_phase2_bins: int,
    forms_notes: Dict[str, Any],
) -> None:
    phase1_monitor, phase1_monitor_mode = _resolve_phase_monitor(args, 1)
    phase2_monitor, phase2_monitor_mode = _resolve_phase_monitor(args, 2)

    if _simple_logs_enabled():
        print(
            f"[TRAIN READY] train={n_train_samples} | val={n_val_samples} | test={n_test_samples} | "
            f"sparse_train={n_train_sparse:,} | sparse_val={n_val_sparse:,} | sparse_test={n_test_sparse:,} | "
            f"in_channels={in_ch} | patch={patch_size} | stride={stride} | device={device}",
            flush=True,
        )
        print(
            f"[SCHEDULE] phase1_steps={schedule['step1_max_steps']} | phase2_steps={schedule['step2_max_steps']} | "
            f"val_every={schedule['val_every_steps']} | batch={args.batch_size}",
            flush=True,
        )
        return

    _print_header("TRAIN EXPERIMENT — HEADER")
    print(f"[EXPERIMENT] root={args.experiment_root}")
    print(f"[EXPERIMENT] policy={exp_cfg.get('policy', 'N/A')} | train={n_train_samples} | val={n_val_samples} | test={n_test_samples}")
    print(f"[SHARDS] train={len(train_shards)} | val={len(val_shards)} | test={len(test_shards)}")
    print(f"[SPARSE] train={n_train_sparse:,} | val={n_val_sparse:,} | test={n_test_sparse:,}")
    print(f"[SCHEMA] in_channels={in_ch} | patch_size={patch_size} | stride={stride} | temporal_window_days={temporal_window_days}")
    print(f"[CHANNEL_ORDER] {tuple(channel_order)}")
    print(f"[RUN] output_dir={run_dir}")
    print(f"[DEVICE] {device}")
    print(f"[BINS] thresholds={args.bins} -> {len(parse_bins(args.bins, BINS_DEFAULT)) + 1} classes")
    _print_loss_block(args)
    print(f"  counts            = {counts.tolist()}")
    print(f"  auto_reg_weights  = {[f'{w:.3f}' for w in reg_w_auto.tolist()]}")
    print(f"  auto_ord_pos_w    = {[f'{w:.2f}' for w in ord_pw_auto]}")
    print(f"  used_reg_weights  = {[f'{w:.3f}' for w in reg_w_used]}")
    print(f"  used_ord_pos_w    = {[f'{w:.2f}' for w in ord_pw_used]}")
    print(f"  used_cls_weights  = {[f'{w:.3f}' for w in cls_w_used]}")
    print(
        f"[PHASE 1] max_steps={schedule['step1_max_steps']} | "
        f"lr={float(args.step1_lr if args.step1_lr is not None else args.lr):.2e} | "
        f"wd={float(args.step1_weight_decay if args.step1_weight_decay is not None else args.weight_decay):.2e} | "
        f"patience_evals={schedule['step1_patience_evals']}"
    )
    print(
        f"[PHASE 2] max_steps={schedule['step2_max_steps']} | "
        f"lr={float(args.step2_lr if args.step2_lr is not None else max((args.step1_lr if args.step1_lr is not None else args.lr) * 0.1, args.lr_min)):.2e} | "
        f"wd={float(args.step2_weight_decay if args.step2_weight_decay is not None else (args.step1_weight_decay if args.step1_weight_decay is not None else args.weight_decay)):.2e} | "
        f"patience_evals={schedule['step2_patience_evals']}"
    )
    print(f"[STEP SCHEDULE] requested_mode={schedule.get('requested_training_mode', args.training_mode)} | effective_mode={schedule.get('effective_training_mode', args.training_mode)} | val_every_steps={schedule['val_every_steps']} | optimizer={optimizer_name}")
    print(f"[PHASE2 AUTO-BINS] strategy={args.phase2_rebalance_strategy} | target_n_classes={args.phase2_target_n_classes} | min_points_per_class={args.phase2_min_points_per_class} | min_bin_width={args.phase2_min_bin_width} | cache={args.phase2_bin_cache_mode}")
    print(f"[PHASE2 TRAIN BINS] detected={list(phase2_train_bin_edges)} | n_train_targets={int(n_train_targets_for_phase2_bins)}")
    print(
        f"[TRAIN SETTINGS] augment={'ON' if args.augment else 'off'} | use_amp={args.use_amp} | "
        f"phase1_monitor={phase1_monitor} ({phase1_monitor_mode}) | "
        f"phase2_monitor={phase2_monitor} ({phase2_monitor_mode}) | "
        f"phase2_high_threshold={float(args.phase2_monitor_high_threshold):.2f} | "
        f"phase2_global_w={float(args.phase2_monitor_global_weight):.2f} | "
        f"phase2_high_w={float(args.phase2_monitor_high_weight):.2f} | "
        f"phase2_reset_best={bool(args.phase2_reset_best_on_monitor_change)}"
    )
    if "unique_temporal_error_mean_ge2" in str(phase1_monitor).lower() or "unique_temporal_error_mean_ge2" in str(phase2_monitor).lower():
        print("[GEDI-UNIQUE MONITOR] best.ckpt uses val_mae_unique_temporal_error_mean_ge2")
        print("[GEDI-UNIQUE TEST] primary=test_mae_unique_nearest_ge2 | secondary=test_mae_unique_temporal_pred_mean_ge2")
    if str(phase1_monitor).strip().lower() in {"anti_shrinkage_score", "val_anti_shrinkage_score"} or str(phase2_monitor).strip().lower() in {"anti_shrinkage_score", "val_anti_shrinkage_score"}:
        print(f"[ANTI-SHRINKAGE MONITOR] score = {float(args.phase2_monitor_global_weight):.2f}*val_mae + {float(args.phase2_monitor_high_weight):.2f}*val_mae_high + 0.85*|1-slope| + 0.45*|1-std_ratio| + 0.20*|bias|")
    if str(phase1_monitor).strip().lower() in {"slope_priority_score", "val_slope_priority_score", "anti_shrinkage_slope_priority_score"} or str(phase2_monitor).strip().lower() in {"slope_priority_score", "val_slope_priority_score", "anti_shrinkage_slope_priority_score"}:
        print("[SLOPE-PRIORITY MONITOR] score = 0.50*val_mae + 0.10*val_mae_high + 4.00*max(0,0.90-slope) + 0.60*max(0,|1-std_ratio|-0.12) + 0.15*|bias|")
    print(
        f"[TRAIN SAMPLER] mode={getattr(args, 'train_sampler_mode', 'natural')} | "
        f"height_bins={getattr(args, 'height_sampler_bins', 'N/A')} | "
        f"height_stat={getattr(args, 'height_sampler_stat', 'N/A')}"
    )
    if str(getattr(args, 'train_sampler_mode', 'natural')).strip().lower() == 'balanced_height':
        print("[TRAIN SAMPLER] Ablation A active: data sampling changes only; Pauls strict loss remains unchanged.")

    if bool(args.forms_t_strict):
        print("[FORMS STRICT] enabled=True")
        print(f"  optimizer         = {optimizer_name}")
        print(f"  auto_drop_applied = {forms_notes.get('forms_auto_drop_applied')}")
        print(f"  auto_dropped_idx  = {forms_notes.get('auto_dropped_channels')}")
        if forms_notes.get("warning"):
            print(f"  warning           = {forms_notes['warning']}")
        if int(in_ch) != 6:
            print(f"  warning           = FORMS target is 6 channels, current in_channels={int(in_ch)}")
        print("  note              = Training is step-budgeted here, but data still comes from prebuilt shards, not true on-the-fly random tile sampling.")



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train canopy-height model from a merged experiment root.")

    # Inputs / outputs
    p.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--resume-strict", action="store_true")

    # Recipe shortcuts
    p.add_argument("--forms-t-strict", action="store_true", help="Apply a FORMS-like recipe: MAE-only, Adam, 6ch target via auto-drop if possible, monitor val_mae, fixed phase-2 bins 5/25.")
    p.add_argument("--training-mode", choices=["step", "epoch"], default="step", help="step = true step-budgeted training windows; epoch = classic full-loader epochs")

    # Model
    p.add_argument("--model-type", choices=["hytec", "tl", "b4_timm", "efficientnet", "efficientnet_b4_timm", "dinov3_tl", "dofa_tl", "satlas_tl", "clay_tl", "terramind_tl", "scalemae_tl", "prithvi_tl"], default="hytec")
    p.add_argument("--base-ch", type=int, default=DEFAULT_BASE_CH)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--freeze-encoder-epochs", type=int, default=5)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--dino-source", type=str, default=None)
    p.add_argument("--pretrained-source", type=str, default=None, help="External pretrained source/model id for DOFA/Satlas/Clay/TerraMind/Scale-MAE/Prithvi or DINO override.")
    p.add_argument("--timm-model-name", type=str, default="efficientnet_b4", help="timm model name for --model-type tl/b4_timm/efficientnet.")
    p.add_argument("--hf-trust-remote-code", action="store_true", help="Allow custom Hugging Face model code for external backbones.")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument(
        "--transfer-head",
        choices=["off", "anti_shrink"],
        default=str(os.environ.get("CHM_TRANSFER_HEAD", "off")).strip().lower(),
        help="Optional transfer-learning head attached on top of a resumed HyTec checkpoint to fight vertical shrinkage.",
    )
    p.add_argument(
        "--transfer-head-hidden-ch",
        type=int,
        default=int(os.environ.get("CHM_TRANSFER_HEAD_HIDDEN_CH", "32")),
        help="Hidden channels used by the anti-shrink transfer head.",
    )
    p.add_argument(
        "--transfer-head-gate-bias",
        type=float,
        default=float(os.environ.get("CHM_TRANSFER_HEAD_GATE_BIAS", "-2.0")),
        help="Initial bias for the residual gate of the transfer head. More negative = more conservative start.",
    )
    p.add_argument(
        "--transfer-head-high-threshold",
        type=float,
        default=float(os.environ.get("CHM_TRANSFER_HEAD_HIGH_THRESHOLD", "20.0")),
        help="Height threshold (m) above which the dedicated high-canopy transfer branch becomes active.",
    )
    p.add_argument(
        "--transfer-head-init-high-gain",
        type=float,
        default=float(os.environ.get("CHM_TRANSFER_HEAD_INIT_HIGH_GAIN", "0.0")),
        help="Initial gain of the high-canopy branch. Keep near zero for stable resume.",
    )
    p.add_argument(
        "--transfer-head-init-global-scale",
        type=float,
        default=float(os.environ.get("CHM_TRANSFER_HEAD_INIT_GLOBAL_SCALE", "1.0")),
        help="Initial multiplicative scale applied to the base prediction inside the transfer head.",
    )
    p.add_argument(
        "--transfer-head-init-residual-scale",
        type=float,
        default=float(os.environ.get("CHM_TRANSFER_HEAD_INIT_RESIDUAL_SCALE", "1.0")),
        help="Initial scale applied to the learned residual branch of the transfer head.",
    )
    p.add_argument(
        "--transfer-head-freeze-base",
        action="store_true",
        default=str(os.environ.get("CHM_TRANSFER_HEAD_FREEZE_BASE", "0")).strip().lower() in {"1", "true", "yes", "y", "on"},
        help="Freeze the pretrained HyTec backbone and train only the transfer head parameters.",
    )
    p.add_argument("--no-transfer-head-freeze-base", dest="transfer_head_freeze_base", action="store_false")

    # Data
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-aux-k", type=int, default=MAX_AUX_K)
    p.add_argument("--drop-channels", type=str, default=None, help="comma-separated channel indices to drop")
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--no-preflight", action="store_true")

    # Ablation A — balanced height sampler, data-side only.
    # Default is balanced_height for this patched Ablation-A script.
    # Use --train-sampler-mode natural to recover the original PatchShardIterable behavior.
    p.add_argument(
        "--train-sampler-mode",
        choices=["natural", "balanced_height", "temporal_patch"],
        default=str(os.environ.get("CHM_TRAIN_SAMPLER_MODE", "balanced_height")).strip().lower(),
        help="natural = original PatchShardIterable; balanced_height = balanced patch sampling by GEDI height bins; temporal_patch = keep same-patch/month years adjacent for Growth Loss.",
    )
    p.add_argument(
        "--height-sampler-bins",
        default=str(os.environ.get("CHM_HEIGHT_SAMPLER_BINS", "0,10,20,30,45")),
        help="Comma-separated edges for balanced height sampler, e.g. 0,10,20,30,45.",
    )
    p.add_argument(
        "--height-sampler-stat",
        choices=["max", "p90", "mean"],
        default=str(os.environ.get("CHM_HEIGHT_SAMPLER_STAT", "max")).strip().lower(),
        help="Patch height statistic used to assign a patch to a sampler bin.",
    )

    # Backward-compatible general training args = phase 1 defaults
    p.add_argument("--epochs", type=int, default=40, help="Fallback used only when explicit step budgets are not provided")
    p.add_argument("--lr", type=float, default=1e-4, help="Used as phase-1 lr if --step1-lr is not set")
    p.add_argument("--weight-decay", type=float, default=5e-3, help="Used as phase-1 wd if --step1-weight-decay is not set")
    p.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    p.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE_DEFAULT, help="Fallback patience if phase-specific values are not set")
    p.add_argument("--monitor", type=str, default="val_mae_unique_temporal_error_mean_ge2")
    p.add_argument("--monitor-mode", choices=["min", "max"], default="min")
    p.add_argument("--phase1-monitor", type=str, default=None, help="Optional explicit monitor for phase 1. Default: --monitor")
    p.add_argument("--phase1-monitor-mode", choices=["min", "max"], default=None, help="Optional explicit monitor mode for phase 1. Default: --monitor-mode")
    p.add_argument("--phase2-monitor", type=str, default="val_mae_unique_temporal_error_mean_ge2", help="Phase-2 monitor. Examples: val_mae_unique_temporal_error_mean_ge2, composite_mae_high, anti_shrinkage_score, val_mae_high, val_mae_ge10, val_rmse, val_mae")
    p.add_argument("--phase2-monitor-mode", choices=["min", "max"], default="min", help="Monitor mode for phase 2")
    p.add_argument("--phase2-monitor-high-threshold", type=float, default=10.0, help="Threshold used by *_high / *_ge monitors in phase 2")
    p.add_argument("--phase2-monitor-global-weight", type=float, default=0.7, help="Global MAE weight for composite_mae_high")
    p.add_argument("--phase2-monitor-high-weight", type=float, default=0.3, help="High-height MAE weight for composite_mae_high")
    p.add_argument("--phase2-reset-best-on-monitor-change", action="store_true", default=True, help="Reset best.ckpt selection when phase-2 monitor differs from phase 1")
    p.add_argument("--no-phase2-reset-best-on-monitor-change", dest="phase2_reset_best_on_monitor_change", action="store_false")

    # Checkpoint-safe anti-shrinkage gate.
    # This is a hard eligibility filter for official best.ckpt: raw improvements
    # that are vertically compressed are saved only as best_any.ckpt for audit.
    p.add_argument("--checkpoint-eligibility-mode", choices=["off", "on"], default="off")
    p.add_argument("--checkpoint-min-slope", type=float, default=float("nan"))
    p.add_argument("--checkpoint-min-std-ratio", type=float, default=float("nan"))
    p.add_argument("--checkpoint-max-std-ratio", type=float, default=float("nan"))
    p.add_argument("--checkpoint-maxpred-under-margin", type=float, default=float("nan"))
    p.add_argument("--checkpoint-maxpred-over-margin", type=float, default=float("nan"))
    p.add_argument("--checkpoint-max-abs-bias", type=float, default=float("nan"), help="Optional hard gate on absolute validation bias for official best.ckpt")
    p.add_argument("--official-best-min-rel-gain", type=float, default=0.0, help="Minimum relative monitor gain required to overwrite official best.ckpt, e.g. 0.003 = 0.3%%")
    p.add_argument("--official-best-min-abs-gain", type=float, default=0.0, help="Minimum absolute monitor gain required to overwrite official best.ckpt")
    p.add_argument("--use-amp", action="store_true", default=USE_AMP_DEFAULT)
    p.add_argument("--no-amp", dest="use_amp", action="store_false")
    p.add_argument("--tv-weight", type=float, default=TV_WEIGHT_DEFAULT)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")

    # Phase 1 / Phase 2 schedule
    p.add_argument("--step1-epochs", type=int, default=None)
    p.add_argument("--step1-lr", type=float, default=None)
    p.add_argument("--step1-weight-decay", type=float, default=None)
    p.add_argument("--step1-patience", type=int, default=None)
    p.add_argument("--step2-epochs", type=int, default=0)
    p.add_argument("--step2-lr", type=float, default=None)
    p.add_argument("--step2-weight-decay", type=float, default=None)
    p.add_argument("--step2-patience", type=int, default=None)

    # Step-budgeted controls
    p.add_argument("--step1-max-steps", type=int, default=None)
    p.add_argument("--step2-max-steps", type=int, default=None)
    p.add_argument("--val-every-steps", type=int, default=None, help="Validation/checkpoint cadence in training steps. FORMS-like default = 640")
    p.add_argument("--step1-patience-evals", type=int, default=None, help="Early-stop patience in number of validation windows during phase 1")
    p.add_argument("--step2-patience-evals", type=int, default=None, help="Early-stop patience in number of validation windows during phase 2")
    p.add_argument("--phase2-reload-best", action="store_true", default=True)
    p.add_argument("--no-phase2-reload-best", dest="phase2_reload_best", action="store_false")

    # Warmup + Plateau scheduler
    p.add_argument("--lr-warmup-epochs", type=int, default=3)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--plateau-patience", type=int, default=10)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--step2-lr-warmup-epochs", type=int, default=0)

    # Bins / loss config
    p.add_argument("--bins", type=str, default="3,10,20,30")
    p.add_argument("--weights-mode", choices=["none", "inv", "sqrt_inv"], default="sqrt_inv")
    p.add_argument("--reg-bin-weights", type=str, default=None, help="comma-separated len=len(bins)+1")
    p.add_argument("--ord-pos-weight", type=str, default=None, help="comma-separated len=len(bins)")
    p.add_argument("--cls-bin-weights", type=str, default=None, help="comma-separated len=len(bins)+1")
    p.add_argument("--alpha-reg", type=float, default=1.0)
    p.add_argument("--beta-ord", type=float, default=0.2)
    p.add_argument("--gamma-cls", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--huber-beta", type=float, default=1.0)
    p.add_argument("--patch-weight-mode", choices=["equal", "sqrt_n", "count", "clipped_count"], default="sqrt_n")
    p.add_argument("--patch-count-cap", type=float, default=16.0)
    p.add_argument("--regression-loss", choices=["mae", "huber"], default="mae")

    # Phase 1 anti-shrinkage loss ablations.
    # Default keeps the historical behavior: plain MAE in phase 1.
    p.add_argument("--phase1-rebalance-strategy", type=str, default="plain_mae", help="Phase-1 sparse regression strategy: plain_mae, weighted_mae, point_weighted_mae, asymmetric_weighted_mae, auto_quantile, forms_fixed.")
    p.add_argument("--phase1-bin-edges", type=str, default=None, help="Optional manual comma-separated edges for phase-1 weighted/asymmetric loss, e.g. 5,10,15,20,30")
    p.add_argument("--phase1-class-weights", type=str, default=None, help="Optional comma-separated class weights for phase 1, len=len(phase1-bin-edges)+1")
    p.add_argument("--phase1-target-n-classes", type=int, default=3)
    p.add_argument("--phase1-min-points-per-class", type=int, default=64)
    p.add_argument("--phase1-min-bin-width", type=float, default=0.75)
    p.add_argument("--phase1-asym-low-threshold", type=float, default=5.0)
    p.add_argument("--phase1-asym-high-threshold", type=float, default=20.0)
    p.add_argument("--phase1-asym-very-high-threshold", type=float, default=30.0)
    p.add_argument("--phase1-asym-low-over-weight", type=float, default=1.0)
    p.add_argument("--phase1-asym-high-under-weight", type=float, default=1.0)
    p.add_argument("--phase1-asym-very-high-under-weight", type=float, default=1.0)
    p.add_argument("--phase1-lambda-slope-loss", type=float, default=0.0)
    p.add_argument("--phase1-lambda-std-loss", type=float, default=0.0)
    p.add_argument("--phase1-lambda-bias-loss", type=float, default=0.0)
    p.add_argument("--phase1-lambda-anti-zero-loss", type=float, default=0.0, help="Soft penalty weight for predictions below --anti-zero-min-pred when y_true >= --anti-zero-min-target-y during phase 1")

    # Shared anti-shrinkage moment loss controls.
    p.add_argument("--anti-shrink-min-points", type=int, default=32)
    p.add_argument("--anti-shrink-eps", type=float, default=1e-6)
    p.add_argument("--anti-shrink-target-slope", type=float, default=1.0)
    p.add_argument("--anti-shrink-target-std-ratio", type=float, default=1.0)
    p.add_argument("--anti-zero-min-target-y", type=float, default=2.5, help="Apply anti-zero only on valid GEDI targets >= this height in meters")
    p.add_argument("--anti-zero-min-pred", type=float, default=2.0, help="Soft lower target for valid forest predictions; not a hard clamp")
    p.add_argument("--anti-zero-power", type=float, default=2.0, help="Power used by anti-zero penalty: relu(min_pred - pred)^power")
    p.add_argument("--anti-zero-min-points", type=int, default=1, help="Minimum number of offending points required before anti-zero loss is active")

    # Phase 2 auto bins + optional phase-specific weights
    p.add_argument("--phase2-rebalance-strategy", type=str, default="auto_quantile")
    p.add_argument("--phase2-target-n-classes", type=int, default=3)
    p.add_argument("--phase2-min-points-per-class", type=int, default=64)
    p.add_argument("--phase2-min-bin-width", type=float, default=0.75)
    p.add_argument("--phase2-bin-cache-mode", type=str, default="once")
    p.add_argument("--phase2-bin-edges", type=str, default=None, help="Optional manual comma-separated edges for phase 2")
    p.add_argument("--phase2-train-bin-edges", type=str, default=None, help="Optional precomputed train-set edges for phase 2")
    p.add_argument("--phase2-class-weights", type=str, default=None, help="Optional comma-separated class weights for phase 2 equal-bin regression")
    p.add_argument("--phase2-beta-ord", type=float, default=None)
    p.add_argument("--phase2-gamma-cls", type=float, default=None)
    p.add_argument("--phase2-asym-low-threshold", type=float, default=5.0)
    p.add_argument("--phase2-asym-high-threshold", type=float, default=20.0)
    p.add_argument("--phase2-asym-very-high-threshold", type=float, default=30.0)
    p.add_argument("--phase2-asym-low-over-weight", type=float, default=1.0)
    p.add_argument("--phase2-asym-high-under-weight", type=float, default=1.0)
    p.add_argument("--phase2-asym-very-high-under-weight", type=float, default=1.0)
    p.add_argument("--phase2-lambda-slope-loss", type=float, default=0.0)
    p.add_argument("--phase2-lambda-std-loss", type=float, default=0.0)
    p.add_argument("--phase2-lambda-bias-loss", type=float, default=0.0)
    p.add_argument("--phase2-lambda-anti-zero-loss", type=float, default=0.0, help="Soft penalty weight for predictions below --anti-zero-min-pred when y_true >= --anti-zero-min-target-y during phase 2")
    p.add_argument("--phase1-lambda-growth-loss", type=float, default=0.0, help="Dense temporal growth consistency weight during phase 1; normally 0.")
    p.add_argument("--phase2-lambda-growth-loss", type=float, default=0.0, help="Dense temporal growth consistency weight during phase 2.")
    p.add_argument("--growth-dmax-m-per-year", type=float, default=2.0, help="Allowed annual canopy-height decrease before growth penalty.")
    p.add_argument("--growth-gmax-m-per-year", type=float, default=1.2, help="Allowed annual canopy-height increase before growth penalty.")
    p.add_argument("--growth-loss-stride", type=int, default=8, help="Pixel stride for dense growth loss to limit memory; 1 means full 10 m grid.")
    p.add_argument("--growth-stable-mask-mode", choices=["all", "aoi", "disturbance_guard"], default="disturbance_guard", help="Stable mask used by growth loss. disturbance_guard = AOI minus multi-signal disturbance proxy.")
    p.add_argument("--growth-aoi-channel-index", type=int, default=10, help="Input channel index containing AOI mask, default 10 for C13 stack.")
    p.add_argument("--growth-aoi-threshold", type=float, default=0.5, help="AOI threshold for growth stable mask.")
    p.add_argument("--growth-max-year-gap", type=int, default=1, help="Only compare consecutive years when set to 1.")
    p.add_argument("--growth-disturbed-jump-weight", type=float, default=0.1, help="Weight for positive-jump penalty on disturbed pixels; no decrease penalty is applied there.")
    p.add_argument("--growth-ndvi-drop-threshold", type=float, default=0.18, help="NDVI drop threshold used by disturbance_guard.")
    p.add_argument("--growth-s1-change-threshold", type=float, default=0.20, help="Sentinel-1 change threshold used by disturbance_guard.")
    p.add_argument("--growth-palsar-change-threshold", type=float, default=0.18, help="PALSAR change threshold used by disturbance_guard.")
    p.add_argument("--growth-disturbance-min-signals", type=int, default=2, help="Minimum number of RS disturbance signals required to relax the decrease penalty.")
    p.add_argument("--growth-gedi-drop-threshold-m", type=float, default=4.0, help="ECHOSAT-inspired GEDI RH95 absolute drop threshold in metres.")
    p.add_argument("--growth-gedi-drop-rel-threshold", type=float, default=0.50, help="ECHOSAT-inspired GEDI RH95 relative drop threshold.")
    p.add_argument("--growth-gedi-final-height-threshold-m", type=float, default=10.0, help="ECHOSAT-inspired final RH95 threshold after abrupt loss.")

    # Eval / printing
    p.add_argument("--eval-test-at-end", action="store_true", default=True)
    p.add_argument("--no-eval-test-at-end", dest="eval_test_at_end", action="store_false")
    p.add_argument("--print-all-metrics", action="store_true", default=False)
    p.add_argument("--no-print-all-metrics", dest="print_all_metrics", action="store_false")

    # Generic publication evaluation domains. These make the same 06_train_experiment.py
    # and trainloop.py valid for every forest. Launchers choose the bounds; the core
    # trainer never hard-codes Maamoura/Ifran/Taroudant.
    p.add_argument("--eval-primary-min-height", type=float, default=None, help="Primary evaluation lower bound, e.g. 2.5 for Maamoura")
    p.add_argument("--eval-primary-max-height", type=float, default=None, help="Primary evaluation upper bound, e.g. 15 for Maamoura")
    p.add_argument("--eval-audit-min-height", type=float, default=None, help="Optional audit-tail lower bound, e.g. 15 for Maamoura")
    p.add_argument("--eval-audit-max-height", type=float, default=None, help="Optional audit-tail upper bound, e.g. 20 for Maamoura")
    p.add_argument("--checkpoint-gate-domain", choices=["global", "primary", "audit_tail"], default="global", help="Domain used by checkpoint-safe gate. Use primary when official evaluation is a bounded domain such as 2.5-15 m.")

    return p


def _set_phase2_criterion_attrs(
    args,
    criterion: torch.nn.Module,
    *,
    phase1_bin_edges_manual: Optional[Tuple[float, ...]] = None,
    phase1_class_weights: Optional[Tuple[float, ...]] = None,
    phase2_bin_edges_manual: Optional[Tuple[float, ...]] = None,
    phase2_train_bin_edges: Optional[Tuple[float, ...]] = None,
    phase2_class_weights: Optional[Tuple[float, ...]] = None,
) -> None:
    """Attach phase-specific loss-ablation settings to the criterion object.

    trainloop.py reads these attributes inside compute_pixelwise_loss(). Keeping
    them on the criterion avoids changing the train_one_cycle/eval_one_cycle APIs.
    """
    criterion.regression_loss_name = str(args.regression_loss).strip().lower()

    # Phase 1: historical default remains plain MAE.
    criterion.phase1_rebalance_strategy = str(args.phase1_rebalance_strategy).strip().lower()
    criterion.phase1_bin_edges = (
        tuple(float(x) for x in phase1_bin_edges_manual)
        if phase1_bin_edges_manual is not None else None
    )
    criterion.phase1_class_weights = (
        tuple(float(x) for x in phase1_class_weights)
        if phase1_class_weights is not None else None
    )
    criterion.phase1_target_n_classes = int(args.phase1_target_n_classes)
    criterion.phase1_min_points_per_class = int(args.phase1_min_points_per_class)
    criterion.phase1_min_bin_width = float(args.phase1_min_bin_width)

    # Phase 2: backward-compatible behavior.
    criterion.phase2_rebalance_strategy = str(args.phase2_rebalance_strategy).strip().lower()
    criterion.rebalance_strategy = criterion.phase2_rebalance_strategy
    criterion.phase2_target_n_classes = int(args.phase2_target_n_classes)
    criterion.phase2_n_classes = int(args.phase2_target_n_classes)
    criterion.phase2_min_points_per_class = int(args.phase2_min_points_per_class)
    criterion.phase2_min_bin_width = float(args.phase2_min_bin_width)
    criterion.phase2_bin_cache_mode = str(args.phase2_bin_cache_mode).strip().lower()

    criterion.phase2_bin_edges = (
        tuple(float(x) for x in phase2_bin_edges_manual)
        if phase2_bin_edges_manual is not None else None
    )
    criterion.phase2_train_bin_edges = (
        tuple(float(x) for x in phase2_train_bin_edges)
        if phase2_train_bin_edges is not None else None
    )
    criterion.phase2_class_weights = (
        tuple(float(x) for x in phase2_class_weights)
        if phase2_class_weights is not None else None
    )

    # Asymmetric anti-shrinkage weights.
    for ph in (1, 2):
        for name in (
            "asym_low_threshold",
            "asym_high_threshold",
            "asym_very_high_threshold",
            "asym_low_over_weight",
            "asym_high_under_weight",
            "asym_very_high_under_weight",
            "lambda_slope_loss",
            "lambda_std_loss",
            "lambda_bias_loss",
            "lambda_anti_zero_loss",
        ):
            setattr(criterion, f"phase{ph}_{name}", float(getattr(args, f"phase{ph}_{name}")))

    criterion.anti_shrink_min_points = int(args.anti_shrink_min_points)
    criterion.anti_shrink_eps = float(args.anti_shrink_eps)
    criterion.anti_shrink_target_slope = float(args.anti_shrink_target_slope)
    criterion.anti_shrink_target_std_ratio = float(args.anti_shrink_target_std_ratio)

    # Soft anti-zero CHM loss: discourages impossible 0 m outputs on valid forest GEDI points.
    criterion.anti_zero_min_target_y = float(args.anti_zero_min_target_y)
    criterion.anti_zero_min_pred = float(args.anti_zero_min_pred)
    criterion.anti_zero_power = float(args.anti_zero_power)
    criterion.anti_zero_min_points = int(args.anti_zero_min_points)

    # Dense multi-year growth consistency regularizer.
    criterion.phase1_lambda_growth_loss = float(getattr(args, "phase1_lambda_growth_loss", 0.0))
    criterion.phase2_lambda_growth_loss = float(getattr(args, "phase2_lambda_growth_loss", 0.0))
    criterion.growth_dmax_m_per_year = float(getattr(args, "growth_dmax_m_per_year", 2.0))
    criterion.growth_gmax_m_per_year = float(getattr(args, "growth_gmax_m_per_year", 1.2))
    criterion.growth_loss_stride = int(getattr(args, "growth_loss_stride", 8))
    criterion.growth_stable_mask_mode = str(getattr(args, "growth_stable_mask_mode", "aoi")).strip().lower()
    criterion.growth_aoi_channel_index = int(getattr(args, "growth_aoi_channel_index", 10))
    criterion.growth_aoi_threshold = float(getattr(args, "growth_aoi_threshold", 0.5))
    criterion.growth_max_year_gap = int(getattr(args, "growth_max_year_gap", 1))
    criterion.growth_disturbed_jump_weight = float(getattr(args, "growth_disturbed_jump_weight", 0.1))
    criterion.growth_ndvi_drop_threshold = float(getattr(args, "growth_ndvi_drop_threshold", 0.18))
    criterion.growth_s1_change_threshold = float(getattr(args, "growth_s1_change_threshold", 0.20))
    criterion.growth_palsar_change_threshold = float(getattr(args, "growth_palsar_change_threshold", 0.18))
    criterion.growth_disturbance_min_signals = int(getattr(args, "growth_disturbance_min_signals", 2))
    criterion.growth_gedi_drop_threshold_m = float(getattr(args, "growth_gedi_drop_threshold_m", 4.0))
    criterion.growth_gedi_drop_rel_threshold = float(getattr(args, "growth_gedi_drop_rel_threshold", 0.50))
    criterion.growth_gedi_final_height_threshold_m = float(getattr(args, "growth_gedi_final_height_threshold_m", 10.0))



def _apply_forms_t_strict_recipe(args) -> None:
    """
    Approximate the FORMS-T training recipe:
      - phase 1: plain MAE over GEDI points
      - no ordinal / no aux classification / no TV
      - validation monitored with val_mae
      - Adam optimizer
      - step-budgeted training with validation every 640 steps
      - phase 2: simple fixed 3-class rebalance with edges 5 m and 25 m
    """
    args.training_mode = "step"

    args.regression_loss = "mae"
    args.monitor = "val_mae"
    args.monitor_mode = "min"
    args.phase1_monitor = "val_mae"
    args.phase1_monitor_mode = "min"
    args.phase2_monitor = "val_mae"
    args.phase2_monitor_mode = "min"
    args.phase2_reset_best_on_monitor_change = False

    args.optimizer = "adam"
    if args.step1_lr is None:
        args.step1_lr = 1e-3
    if args.step2_lr is None:
        args.step2_lr = 1e-4

    if args.step1_weight_decay is None:
        args.step1_weight_decay = 0.0
    if args.step2_weight_decay is None:
        args.step2_weight_decay = 0.0

    args.weights_mode = "none"
    args.patch_weight_mode = "count"

    args.beta_ord = 0.0
    args.gamma_cls = 0.0
    args.tv_weight = 0.0

    if args.val_every_steps is None:
        args.val_every_steps = 640

    args.phase1_rebalance_strategy = "plain_mae"
    args.phase1_bin_edges = None
    args.phase1_class_weights = None
    args.phase1_lambda_slope_loss = 0.0
    args.phase1_lambda_std_loss = 0.0
    args.phase1_lambda_bias_loss = 0.0
    args.phase1_lambda_anti_zero_loss = 0.0

    args.phase2_rebalance_strategy = "forms_fixed"
    args.phase2_target_n_classes = 3
    args.phase2_bin_edges = "5,25"
    args.phase2_train_bin_edges = "5,25"
    args.phase2_class_weights = "1,1,1"
    args.phase2_beta_ord = 0.0
    args.phase2_gamma_cls = 0.0
    args.phase2_lambda_slope_loss = 0.0
    args.phase2_lambda_std_loss = 0.0
    args.phase2_lambda_bias_loss = 0.0
    args.phase2_lambda_anti_zero_loss = 0.0




def _safe_metric_lookup(metrics: Dict[str, Any], key: str) -> Optional[float]:
    try:
        v = _metric_value(metrics, key)
        return float(v) if math.isfinite(float(v)) else None
    except Exception:
        return None


def _apply_generic_eval_domain_env(args) -> None:
    """Export generic primary/audit evaluation domains for training.trainloop.

    The core files stay common to all forests. Forest-specific launchers only pass
    CLI values such as 2.5-15 for Maamoura, or broader bounds for Ifran.
    """
    mapping = [
        ("eval_primary_min_height", "CHM_EVAL_PRIMARY_MIN_HEIGHT"),
        ("eval_primary_max_height", "CHM_EVAL_PRIMARY_MAX_HEIGHT"),
        ("eval_audit_min_height", "CHM_EVAL_AUDIT_MIN_HEIGHT"),
        ("eval_audit_max_height", "CHM_EVAL_AUDIT_MAX_HEIGHT"),
    ]
    for attr, env_name in mapping:
        v = getattr(args, attr, None)
        if v is not None:
            os.environ[env_name] = str(float(v))
    # Keep legacy GE2/unique domains synchronized with the primary domain.
    # Several historical trainloops read CHM_EVAL_MAX_HEIGHT directly; exporting
    # both names prevents a silent fallback to the old 40 m Ifran cap.
    if getattr(args, "eval_primary_min_height", None) is not None:
        os.environ["CHM_EVAL_MIN_HEIGHT"] = str(float(args.eval_primary_min_height))
    if getattr(args, "eval_primary_max_height", None) is not None:
        os.environ["CHM_EVAL_MAX_HEIGHT"] = str(float(args.eval_primary_max_height))
    os.environ["CHM_CKPT_GATE_DOMAIN"] = str(getattr(args, "checkpoint_gate_domain", "global")).strip().lower()
    print(
        "[EVAL DOMAIN CONTRACT] "
        f"primary=[{os.environ.get('CHM_EVAL_PRIMARY_MIN_HEIGHT')},"
        f"{os.environ.get('CHM_EVAL_PRIMARY_MAX_HEIGHT')}] | "
        f"legacy=[{os.environ.get('CHM_EVAL_MIN_HEIGHT')},"
        f"{os.environ.get('CHM_EVAL_MAX_HEIGHT')}]",
        flush=True,
    )


def _gedi_unique_reporting_summary(metrics: Dict[str, Any], split_prefix: str) -> Dict[str, Any]:
    prefix = str(split_prefix).strip().lower()
    return {
        f"{prefix}_mae_unique_nearest_ge2": _safe_metric_lookup(metrics, f"{prefix}_mae_unique_nearest_ge2"),
        f"{prefix}_rmse_unique_nearest_ge2": _safe_metric_lookup(metrics, f"{prefix}_rmse_unique_nearest_ge2"),
        f"{prefix}_r2_unique_nearest_ge2": _safe_metric_lookup(metrics, f"{prefix}_r2_unique_nearest_ge2"),
        f"{prefix}_mae_unique_temporal_pred_mean_ge2": _safe_metric_lookup(metrics, f"{prefix}_mae_unique_temporal_pred_mean_ge2"),
        f"{prefix}_mae_unique_temporal_mean_ge2_alias": _safe_metric_lookup(metrics, f"{prefix}_mae_unique_temporal_mean_ge2"),
        f"{prefix}_mae_unique_temporal_error_mean_ge2": _safe_metric_lookup(metrics, f"{prefix}_mae_unique_temporal_error_mean_ge2"),
        f"{prefix}_mae_occurrence_ge2": _safe_metric_lookup(metrics, f"{prefix}_mae_occurrence_ge2"),
    }



def main():
    parser = build_argparser()
    # NOTEBOOK-SAFE: Jupyter injecte souvent des arguments du type -f kernel.json.
    # En script normal, on garde parse_args() strict. En notebook, on ignore seulement
    # les arguments inconnus de Jupyter pour éviter un second crash après le fix __file__.
    if "ipykernel" in sys.modules or any(str(a).endswith(".json") and "kernel" in str(a).lower() for a in sys.argv[1:]):
        args, _unknown_notebook_args = parser.parse_known_args()
        if _unknown_notebook_args:
            print(f"[NOTEBOOK] ignored unknown Jupyter args: {_unknown_notebook_args}", flush=True)
    else:
        args = parser.parse_args()

    simple_logs = _simple_logs_enabled()
    if simple_logs:
        args.no_preflight = True
        args.print_all_metrics = False

    if bool(args.forms_t_strict):
        _apply_forms_t_strict_recipe(args)

    _apply_generic_eval_domain_env(args)

    if args.phase1_monitor is None:
        args.phase1_monitor = str(args.monitor)
    if args.phase1_monitor_mode is None:
        args.phase1_monitor_mode = str(args.monitor_mode)
    if args.phase2_monitor_mode is None:
        args.phase2_monitor_mode = str(args.monitor_mode)

    args.experiment_root = Path(args.experiment_root)
    args.runs_root = Path(args.runs_root)

    assert_experiment_dirs_exist(args.experiment_root)
    exp_cfg = load_experiment_config(args.experiment_root)

    set_seed(args.seed)

    bins = parse_bins(args.bins, BINS_DEFAULT)
    cls_thresholds = bins
    report_bins = bins
    catalog_mode = (
        (args.experiment_root / "sample_catalog_step05.csv").exists()
        and (args.experiment_root / "shot_catalog_step05.csv.gz").exists()
    )

    catalog_samples: Optional[pd.DataFrame] = None
    catalog_shots: Optional[pd.DataFrame] = None
    if catalog_mode:
        catalog_samples = pd.read_csv(args.experiment_root / "sample_catalog_step05.csv", low_memory=False)
        catalog_shots = pd.read_csv(args.experiment_root / "shot_catalog_step05.csv.gz", low_memory=False)
        catalog_samples["split"] = catalog_samples["split"].astype(str).str.lower()
        catalog_shots["split"] = catalog_shots["split"].astype(str).str.lower()

        train_shards = [args.experiment_root / "sample_catalog_step05.csv"]
        val_shards = [args.experiment_root / "sample_catalog_step05.csv"]
        test_shards = [args.experiment_root / "sample_catalog_step05.csv"]
        n_train_samples = int((catalog_samples["split"] == "train").sum())
        n_val_samples = int((catalog_samples["split"] == "val").sum())
        n_test_samples = int((catalog_samples["split"] == "test").sum())
        n_train_sparse = int((catalog_shots["split"] == "train").sum())
        n_val_sparse = int((catalog_shots["split"] == "val").sum())
        n_test_sparse = int((catalog_shots["split"] == "test").sum())

        schema = exp_cfg.get("schema", {}) if isinstance(exp_cfg, dict) else {}
        channel_order = tuple(exp_cfg.get("channel_order", ()))
        patch_size = int(schema.get("patch_size", 512))
        stride = int(schema.get("stride", 512))
        temporal_window_days = int(schema.get("temporal_window_days", 180))
        in_ch_raw = int(schema.get("in_channels", len(channel_order) or 11))
        y_train_all = catalog_shots.loc[
            catalog_shots["split"] == "train", "rh95"
        ].to_numpy(dtype=np.float32)
        counts, reg_w_auto, ord_pw_auto = compute_auto_weights_from_values(
            y_train_all, bins=bins, weights_mode=args.weights_mode
        )
        print(
            f"[STORAGE] catalog NPY mode | root={args.experiment_root} | "
            f"train={n_train_samples} val={n_val_samples} test={n_test_samples}",
            flush=True,
        )
    else:
        if not args.no_preflight:
            preflight_experiment(args.experiment_root, n_shards=1, n_samples=2, max_aux_k=args.max_aux_k)

        train_shards = list_experiment_shards(args.experiment_root, "train")
        val_shards = list_experiment_shards(args.experiment_root, "val")
        test_shards = list_experiment_shards(args.experiment_root, "test")
        if len(train_shards) == 0:
            raise RuntimeError("No train shards found in experiment.")

        n_train_samples = count_samples(train_shards)
        n_val_samples = count_samples(val_shards) if val_shards else 0
        n_test_samples = count_samples(test_shards) if test_shards else 0
        n_train_sparse = count_sparse_targets(train_shards)
        n_val_sparse = count_sparse_targets(val_shards) if val_shards else 0
        n_test_sparse = count_sparse_targets(test_shards) if test_shards else 0
        channel_order = infer_channel_order(args.experiment_root)
        patch_size = infer_patch_size(args.experiment_root)
        stride = infer_stride(args.experiment_root)
        temporal_window_days = infer_temporal_window_days(args.experiment_root)
        in_ch_json = infer_in_channels(args.experiment_root, fallback=8)
        in_ch_shard = _infer_in_ch_from_first_shard(train_shards)
        in_ch_raw = int(in_ch_shard) if in_ch_shard is not None else int(in_ch_json)
        counts, reg_w_auto, ord_pw_auto = compute_auto_weights_from_shards(
            train_shards,
            bins=bins,
            weights_mode=args.weights_mode,
        )
        y_train_all = collect_train_targets_from_shards(train_shards)

    drop_channels_cli = []
    if args.drop_channels:
        drop_channels_cli = [int(x.strip()) for x in str(args.drop_channels).split(",") if x.strip()]

    drop_channels, forms_notes = _resolve_forms_drop_channels(
        args=args,
        channel_order=channel_order,
        drop_channels_cli=drop_channels_cli,
    )

    temporal_fusion = bool(catalog_mode and _temporal_fusion_enabled())
    if temporal_fusion:
        # C11 = 4 dynamic S2 channels + 4 static S1 ASC/DESC channels
        #       + 2 static PALSAR channels + 1 AOI mask.
        # E concatenates dynamic prev/current/next and keeps static channels once.
        if int(in_ch_raw) != 11:
            raise ValueError(f"Temporal fusion expects the C11 catalog, got in_ch_raw={in_ch_raw}")
        if drop_channels:
            raise ValueError("Temporal fusion requires all C11 channels; drop_channels must be empty")
        in_ch = 3 * 4 + 7
        print(
            "[TEMPORAL FUSION] prev/current/next x 4 dynamic S2 channels + "
            "S1_ASC_VV/VH + S1_DESC_VV/VH + PALSAR_HH/HV + AOI once = 19 channels",
            flush=True,
        )
    else:
        in_ch = int(in_ch_raw - len(drop_channels))
    if in_ch <= 0:
        raise ValueError(f"Invalid in_ch after drop_channels: raw={in_ch_raw}, drop={drop_channels}")

    device = get_default_device()

    reg_w_used = _parse_float_tuple(args.reg_bin_weights, expected_len=len(bins) + 1)
    ord_pw_used = _parse_float_tuple(args.ord_pos_weight, expected_len=len(bins))
    cls_w_used = _parse_float_tuple(args.cls_bin_weights, expected_len=len(bins) + 1)

    if reg_w_used is None:
        reg_w_used = tuple(float(x) for x in reg_w_auto.tolist())
    if ord_pw_used is None:
        ord_pw_used = tuple(float(x) for x in ord_pw_auto)
    if cls_w_used is None:
        cls_w_used = tuple(float(x) for x in reg_w_used)

    phase1_bin_edges_manual = _parse_float_tuple(args.phase1_bin_edges)
    phase1_class_weights = _parse_float_tuple(args.phase1_class_weights)
    if phase1_class_weights is not None and phase1_bin_edges_manual is not None:
        if len(phase1_class_weights) != len(phase1_bin_edges_manual) + 1:
            raise ValueError(
                "phase1-class-weights must contain len(phase1-bin-edges)+1 values. "
                f"Got {len(phase1_class_weights)} weights for {len(phase1_bin_edges_manual)} edges."
            )

    phase2_bin_edges_manual = _parse_float_tuple(args.phase2_bin_edges)
    phase2_train_bin_edges_cli = _parse_float_tuple(args.phase2_train_bin_edges)
    phase2_class_weights = _parse_float_tuple(args.phase2_class_weights)
    phase2_edges_for_check = phase2_train_bin_edges_cli if phase2_train_bin_edges_cli is not None else phase2_bin_edges_manual
    if phase2_class_weights is not None and phase2_edges_for_check is not None:
        if len(phase2_class_weights) != len(phase2_edges_for_check) + 1:
            raise ValueError(
                "phase2-class-weights must contain len(phase2-bin-edges)+1 values. "
                f"Got {len(phase2_class_weights)} weights for {len(phase2_edges_for_check)} edges."
            )

    if phase2_train_bin_edges_cli is not None:
        phase2_train_bin_edges = tuple(float(x) for x in phase2_train_bin_edges_cli)
    elif phase2_bin_edges_manual is not None:
        phase2_train_bin_edges = tuple(float(x) for x in phase2_bin_edges_manual)
    else:
        phase2_train_bin_edges = infer_phase2_train_bin_edges(
            y_train_all,
            target_n_classes=int(args.phase2_target_n_classes),
            min_points_per_class=int(args.phase2_min_points_per_class),
            min_bin_width=float(args.phase2_min_bin_width),
        )

    if not simple_logs:
        print("[PHASE1 LOSS ABLATION]")
        print(f"strategy                 : {args.phase1_rebalance_strategy}")
        print(f"manual_bin_edges         : {phase1_bin_edges_manual}")
        print(f"class_weights            : {phase1_class_weights}")
        print(f"lambda_slope/std/bias    : {float(args.phase1_lambda_slope_loss)}, {float(args.phase1_lambda_std_loss)}, {float(args.phase1_lambda_bias_loss)}")
        print(f"asym weights low/high/vh : {float(args.phase1_asym_low_over_weight)}, {float(args.phase1_asym_high_under_weight)}, {float(args.phase1_asym_very_high_under_weight)}")
        print("[PHASE2 BINS]")
        print(f"strategy                 : {args.phase2_rebalance_strategy}")
        print(f"target_n_classes         : {int(args.phase2_target_n_classes)}")
        print(f"min_points_per_class     : {int(args.phase2_min_points_per_class)}")
        print(f"min_bin_width            : {float(args.phase2_min_bin_width)}")
        print(f"cache_mode               : {args.phase2_bin_cache_mode}")
        print(f"manual_bin_edges         : {phase2_bin_edges_manual}")
        print(f"train_bin_edges_cli      : {phase2_train_bin_edges_cli}")
        print(f"detected_train_bin_edges : {phase2_train_bin_edges}")
        print(f"phase2_class_weights     : {phase2_class_weights}")
        print(f"n_train_targets          : {int(y_train_all.size)}")

    train_sampler_mode = str(getattr(args, "train_sampler_mode", "natural")).strip().lower()
    if train_sampler_mode not in {"natural", "balanced_height", "temporal_patch"}:
        raise ValueError(f"Unsupported train_sampler_mode={train_sampler_mode!r}")

    if catalog_mode:
        train_ds = CatalogNpyIterable(
            args.experiment_root,
            "train",
            seed=args.seed,
            in_ch=in_ch,
            max_aux_k=args.max_aux_k,
            drop_channels=drop_channels,
            balanced_height=(train_sampler_mode == "balanced_height"),
            batch_size=int(args.batch_size),
            samples_per_epoch=n_train_samples,
            shuffle=(train_sampler_mode != "temporal_patch"),
            temporal_fusion=temporal_fusion,
            temporal_patch_grouping=(train_sampler_mode == "temporal_patch"),
        )
        val_ds = CatalogNpyIterable(
            args.experiment_root,
            "val",
            seed=args.seed + 17,
            in_ch=in_ch,
            max_aux_k=args.max_aux_k,
            drop_channels=drop_channels,
            balanced_height=False,
            batch_size=int(args.batch_size),
            shuffle=False,
            temporal_fusion=temporal_fusion,
        ) if n_val_samples else None
        test_ds = CatalogNpyIterable(
            args.experiment_root,
            "test",
            seed=args.seed + 29,
            in_ch=in_ch,
            max_aux_k=args.max_aux_k,
            drop_channels=drop_channels,
            balanced_height=False,
            batch_size=int(args.batch_size),
            shuffle=False,
            temporal_fusion=temporal_fusion,
        ) if n_test_samples else None
    elif train_sampler_mode == "balanced_height":
        height_sampler_bins = tuple(
            float(x.strip())
            for x in str(args.height_sampler_bins).split(",")
            if x.strip()
        )
        train_ds = BalancedHeightPatchShardIterable(
            train_shards,
            seed=args.seed,
            in_ch=in_ch,
            max_aux_k=args.max_aux_k,
            drop_channels=drop_channels,
            height_bins=height_sampler_bins,
            height_stat=str(args.height_sampler_stat),
            samples_per_epoch=n_train_samples,
            batch_size=int(args.batch_size),
            verbose=True,
        )
    else:
        print("[TRAIN SAMPLER] mode=natural | original PatchShardIterable", flush=True)
        train_ds = PatchShardIterable(
            train_shards,
            seed=args.seed,
            shuffle_shards=True,
            shuffle_within=True,
            in_ch=in_ch,
            max_aux_k=args.max_aux_k,
            drop_channels=drop_channels,
        )
    if not catalog_mode:
        # IMPORTANT EVALFIX for historical NPZ experiments.
        val_ds = FixedLengthPatchShardIterable(
            val_shards,
            seed=args.seed + 17,
            shuffle_shards=False,
            shuffle_within=False,
            in_ch=in_ch,
            max_aux_k=args.max_aux_k,
            drop_channels=drop_channels,
            split_name="val",
            verbose=True,
        ) if val_shards else None
        test_ds = FixedLengthPatchShardIterable(
            test_shards,
            seed=args.seed + 29,
            shuffle_shards=False,
            shuffle_within=False,
            in_ch=in_ch,
            max_aux_k=args.max_aux_k,
            drop_channels=drop_channels,
            split_name="test",
            verbose=True,
        ) if test_shards else None

    loader_kwargs = dict(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
    )

    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs) if val_ds is not None else None
    test_loader = DataLoader(test_ds, **loader_kwargs) if test_ds is not None else None

    nominal_train_steps = max(1, math.ceil(n_train_samples / max(1, args.batch_size)))
    val_steps = math.ceil(n_val_samples / max(1, args.batch_size)) if n_val_samples > 0 else 0
    test_steps = math.ceil(n_test_samples / max(1, args.batch_size)) if n_test_samples > 0 else 0

    schedule = _resolve_step_schedule(args, nominal_train_steps_per_epoch=nominal_train_steps)
    step1_max_steps = int(schedule["step1_max_steps"])
    step2_max_steps = int(schedule["step2_max_steps"])
    val_every_steps = int(schedule["val_every_steps"])
    step1_patience_evals = int(schedule["step1_patience_evals"])
    step2_patience_evals = int(schedule["step2_patience_evals"])
    total_max_steps = int(step1_max_steps + step2_max_steps)
    effective_training_mode = str(schedule.get("effective_training_mode", args.training_mode)).strip().lower()
    _warn_if_noncomparable_loss_monitor(args, schedule)

    model_type = str(args.model_type).strip().lower()
    pretrained = not bool(args.no_pretrained)

    if model_type == "hytec":
        model = CanopyHyTecModel(
            n_channels=in_ch,
            n_classes=1,
            dropout=float(args.dropout),
            base_ch=int(args.base_ch),
        )
    elif model_type in {"tl", "b4_timm", "efficientnet", "efficientnet_b4_timm"}:
        model = CanopyTLModel(
            n_channels=in_ch,
            pretrained=pretrained,
            freeze_encoder_epochs=int(args.freeze_encoder_epochs),
            dropout=float(args.dropout),
            timm_model_name=str(args.timm_model_name or "efficientnet_b4"),
        )
    elif model_type == "dinov3_tl":
        model = CanopyDinoV3TLModel(
            n_channels=in_ch,
            pretrained=pretrained,
            pretrained_source=args.dino_source or args.pretrained_source,
            freeze_encoder_epochs=int(args.freeze_encoder_epochs),
            dropout=float(args.dropout),
            local_files_only=bool(args.local_files_only),
        )
    else:
        external_model_classes = {
            "dofa_tl": CanopyDOFATLModel,
            "satlas_tl": CanopySatlasTLModel,
            "clay_tl": CanopyClayTLModel,
            "terramind_tl": CanopyTerraMindTLModel,
            "scalemae_tl": CanopyScaleMAETLModel,
            "prithvi_tl": CanopyPrithviTLModel,
        }
        model_cls = external_model_classes[model_type]
        model = model_cls(
            n_channels=in_ch,
            pretrained=pretrained,
            pretrained_source=args.pretrained_source,
            freeze_encoder_epochs=int(args.freeze_encoder_epochs),
            dropout=float(args.dropout),
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.hf_trust_remote_code),
        )

    if _transfer_head_requested(args):
        if model_type != "hytec":
            raise ValueError("The anti-shrink transfer head is currently implemented for model-type=hytec")
        if _true_ordinal_head_enabled():
            raise ValueError("The anti-shrink transfer head cannot be combined with CHM_TRUE_ORDINAL_HEAD")
        if args.resume is None:
            print(
                "[WARN] transfer-head is enabled without --resume. "
                "This head is mainly intended for Phase-2 fine-tuning from an existing checkpoint.",
                flush=True,
            )
        model = AntiShrinkTransferHead(
            model,
            hidden_ch=int(args.transfer_head_hidden_ch),
            gate_bias=float(args.transfer_head_gate_bias),
            high_threshold=float(args.transfer_head_high_threshold),
            init_high_gain=float(args.transfer_head_init_high_gain),
            init_global_scale=float(args.transfer_head_init_global_scale),
            init_residual_scale=float(args.transfer_head_init_residual_scale),
            freeze_base=bool(args.transfer_head_freeze_base),
        )
        n_total_params = sum(int(p.numel()) for p in model.parameters())
        n_trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
        print(
            "[MODEL] anti-shrink transfer head enabled | "
            f"freeze_base={bool(args.transfer_head_freeze_base)} | "
            f"hidden_ch={int(args.transfer_head_hidden_ch)} | "
            f"high_threshold={float(args.transfer_head_high_threshold):.2f} m | "
            f"trainable={n_trainable_params:,}/{n_total_params:,}",
            flush=True,
        )

    if _true_ordinal_head_enabled():
        if model_type != "hytec":
            raise ValueError("The true ordinal head is currently implemented for model-type=hytec")
        model = TrueOrdinalRegressionHead(model, n_thresholds=len(bins))
        print(
            f"[MODEL] true ordinal/regression dual head enabled | thresholds={tuple(float(v) for v in bins)}",
            flush=True,
        )

    model = model.to(device)

    criterion = HyTecLossV6(
        alpha_reg=float(args.alpha_reg),
        beta_ord=float(args.beta_ord),
        ord_thresholds=bins,
        temperature=float(args.temperature),
        huber_beta=float(args.huber_beta),
        reg_bin_edges=bins,
        reg_bin_weights=reg_w_used,
        ord_pos_weight=ord_pw_used,
        patch_weight_mode=str(args.patch_weight_mode),
        patch_count_cap=float(args.patch_count_cap),
        gamma_cls=float(args.gamma_cls),
        cls_bin_weights=cls_w_used,
    ).to(device)
    criterion.true_ordinal_head = bool(_true_ordinal_head_enabled())
    criterion.ordinal_monotonic_lambda = float(os.environ.get("CHM_ORDINAL_MONOTONIC_LAMBDA", "0.02"))

    _set_phase2_criterion_attrs(
        args,
        criterion,
        phase1_bin_edges_manual=phase1_bin_edges_manual,
        phase1_class_weights=phase1_class_weights,
        phase2_bin_edges_manual=phase2_bin_edges_manual,
        phase2_train_bin_edges=phase2_train_bin_edges,
        phase2_class_weights=phase2_class_weights,
    )

    step1_lr = float(args.step1_lr if args.step1_lr is not None else args.lr)
    step1_wd = float(args.step1_weight_decay if args.step1_weight_decay is not None else args.weight_decay)
    step2_lr = float(args.step2_lr if args.step2_lr is not None else max(step1_lr * 0.1, float(args.lr_min)))
    step2_wd = float(args.step2_weight_decay if args.step2_weight_decay is not None else step1_wd)

    optimizer_name = str(args.optimizer).strip().lower()

    run_name = args.run_name or _default_run_name()
    paths = build_run_paths(args.experiment_root, runs_root=args.runs_root, run_name=run_name)

    for k, p in paths.items():
        if k.endswith("_dir"):
            Path(p).mkdir(parents=True, exist_ok=True)

    save_json_artifact(
        paths["run_config_json"],
        {
            "args": vars(args),
            "experiment_root": str(args.experiment_root),
            "experiment_config": exp_cfg,
            "bins": list(bins),
            "channel_order": list(channel_order),
            "counts": counts.tolist(),
            "reg_w_auto": reg_w_auto.tolist(),
            "ord_pw_auto": list(ord_pw_auto),
            "reg_w_used": list(reg_w_used),
            "ord_pw_used": list(ord_pw_used),
            "cls_w_used": list(cls_w_used),
            "in_ch_raw": in_ch_raw,
            "in_ch": in_ch,
            "drop_channels": drop_channels,
            "transfer_head": {
                "mode": str(args.transfer_head),
                "hidden_ch": int(args.transfer_head_hidden_ch),
                "gate_bias": float(args.transfer_head_gate_bias),
                "high_threshold": float(args.transfer_head_high_threshold),
                "init_high_gain": float(args.transfer_head_init_high_gain),
                "init_global_scale": float(args.transfer_head_init_global_scale),
                "init_residual_scale": float(args.transfer_head_init_residual_scale),
                "freeze_base": bool(args.transfer_head_freeze_base),
                "note": "Residual transfer-learning head for slope/std recovery and high-canopy de-shrinkage.",
            },
            "train_sampler": {
                "mode": str(getattr(args, "train_sampler_mode", "natural")),
                "height_bins": str(getattr(args, "height_sampler_bins", "")),
                "height_stat": str(getattr(args, "height_sampler_stat", "")),
                "note": "Ablation A: data-side balanced height sampler only; loss/trainloop unchanged.",
            },
            "forms_notes": forms_notes,
            "step_schedule": schedule,
            "requested_training_mode": str(args.training_mode),
            "effective_training_mode": effective_training_mode,
            "optimizer_name": optimizer_name,
            "phase1_monitor": str(args.phase1_monitor),
            "phase1_monitor_mode": str(args.phase1_monitor_mode),
            "phase2_monitor": str(args.phase2_monitor),
            "phase2_monitor_mode": str(args.phase2_monitor_mode),
            "phase2_monitor_high_threshold": float(args.phase2_monitor_high_threshold),
            "phase2_monitor_global_weight": float(args.phase2_monitor_global_weight),
            "phase2_monitor_high_weight": float(args.phase2_monitor_high_weight),
            "phase2_reset_best_on_monitor_change": bool(args.phase2_reset_best_on_monitor_change),
            "checkpoint_eligibility": {
                "mode": str(args.checkpoint_eligibility_mode),
                "min_slope": float(args.checkpoint_min_slope),
                "min_std_ratio": float(args.checkpoint_min_std_ratio),
                "max_std_ratio": float(args.checkpoint_max_std_ratio),
                "maxpred_under_margin": float(args.checkpoint_maxpred_under_margin),
                "maxpred_over_margin": float(args.checkpoint_maxpred_over_margin),
                "max_abs_bias": float(args.checkpoint_max_abs_bias),
                "official_best_min_rel_gain": float(args.official_best_min_rel_gain),
                "official_best_min_abs_gain": float(args.official_best_min_abs_gain),
            },
            "phase1": {
                "rebalance_strategy": str(args.phase1_rebalance_strategy),
                "manual_bin_edges": list(phase1_bin_edges_manual) if phase1_bin_edges_manual is not None else None,
                "class_weights": list(phase1_class_weights) if phase1_class_weights is not None else None,
                "target_n_classes": int(args.phase1_target_n_classes),
                "min_points_per_class": int(args.phase1_min_points_per_class),
                "min_bin_width": float(args.phase1_min_bin_width),
                "asym_low_threshold": float(args.phase1_asym_low_threshold),
                "asym_high_threshold": float(args.phase1_asym_high_threshold),
                "asym_very_high_threshold": float(args.phase1_asym_very_high_threshold),
                "asym_low_over_weight": float(args.phase1_asym_low_over_weight),
                "asym_high_under_weight": float(args.phase1_asym_high_under_weight),
                "asym_very_high_under_weight": float(args.phase1_asym_very_high_under_weight),
                "lambda_slope_loss": float(args.phase1_lambda_slope_loss),
                "lambda_std_loss": float(args.phase1_lambda_std_loss),
                "lambda_bias_loss": float(args.phase1_lambda_bias_loss),
                "lambda_anti_zero_loss": float(args.phase1_lambda_anti_zero_loss),
            },
            "anti_shrink_moment_loss": {
                "min_points": int(args.anti_shrink_min_points),
                "eps": float(args.anti_shrink_eps),
                "target_slope": float(args.anti_shrink_target_slope),
                "target_std_ratio": float(args.anti_shrink_target_std_ratio),
            },
            "anti_zero_loss": {
                "phase1_lambda": float(args.phase1_lambda_anti_zero_loss),
                "phase2_lambda": float(args.phase2_lambda_anti_zero_loss),
                "min_target_y": float(args.anti_zero_min_target_y),
                "min_pred": float(args.anti_zero_min_pred),
                "power": float(args.anti_zero_power),
                "min_points": int(args.anti_zero_min_points),
                "note": "soft penalty only; predictions are not hard-clamped",
            },
            "phase2": {
                "rebalance_strategy": str(args.phase2_rebalance_strategy),
                "target_n_classes": int(args.phase2_target_n_classes),
                "min_points_per_class": int(args.phase2_min_points_per_class),
                "min_bin_width": float(args.phase2_min_bin_width),
                "bin_cache_mode": str(args.phase2_bin_cache_mode),
                "manual_bin_edges": list(phase2_bin_edges_manual) if phase2_bin_edges_manual is not None else None,
                "train_bin_edges_cli": list(phase2_train_bin_edges_cli) if phase2_train_bin_edges_cli is not None else None,
                "detected_train_bin_edges": list(phase2_train_bin_edges) if phase2_train_bin_edges is not None else None,
                "class_weights": list(phase2_class_weights) if phase2_class_weights is not None else None,
                "beta_ord_phase1": float(args.beta_ord),
                "beta_ord_phase2": float(args.phase2_beta_ord if args.phase2_beta_ord is not None else args.beta_ord),
                "gamma_cls_phase1": float(args.gamma_cls),
                "gamma_cls_phase2": float(args.phase2_gamma_cls if args.phase2_gamma_cls is not None else args.gamma_cls),
                "asym_low_threshold": float(args.phase2_asym_low_threshold),
                "asym_high_threshold": float(args.phase2_asym_high_threshold),
                "asym_very_high_threshold": float(args.phase2_asym_very_high_threshold),
                "asym_low_over_weight": float(args.phase2_asym_low_over_weight),
                "asym_high_under_weight": float(args.phase2_asym_high_under_weight),
                "asym_very_high_under_weight": float(args.phase2_asym_very_high_under_weight),
                "lambda_slope_loss": float(args.phase2_lambda_slope_loss),
                "lambda_std_loss": float(args.phase2_lambda_std_loss),
                "lambda_bias_loss": float(args.phase2_lambda_bias_loss),
                "lambda_anti_zero_loss": float(args.phase2_lambda_anti_zero_loss),
                "n_train_targets_used_for_bins": int(y_train_all.size),
            },
        },
    )

    init_csv(paths["train_csv"], force=True)

    _print_header_block(
        args=args,
        exp_cfg=exp_cfg,
        train_shards=train_shards,
        val_shards=val_shards,
        test_shards=test_shards,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        n_train_sparse=n_train_sparse,
        n_val_sparse=n_val_sparse,
        n_test_sparse=n_test_sparse,
        counts=counts,
        reg_w_auto=reg_w_auto,
        ord_pw_auto=ord_pw_auto,
        reg_w_used=reg_w_used,
        ord_pw_used=ord_pw_used,
        cls_w_used=cls_w_used,
        in_ch=in_ch,
        channel_order=channel_order,
        patch_size=patch_size,
        stride=stride,
        temporal_window_days=temporal_window_days,
        device=device,
        run_dir=paths["run_dir"],
        schedule=schedule,
        optimizer_name=optimizer_name,
        phase2_train_bin_edges=phase2_train_bin_edges,
        n_train_targets_for_phase2_bins=int(y_train_all.size),
        forms_notes=forms_notes,
    )

    active_phase = 1
    active_monitor_key, active_monitor_mode = _resolve_phase_monitor(args, active_phase)
    phase_best = float("inf") if str(active_monitor_mode).lower() == "min" else float("-inf")
    phase_best_metrics: Optional[Dict[str, Any]] = None
    global_best = float("inf") if str(active_monitor_mode).lower() == "min" else float("-inf")
    patience_counter = 0
    is_resumed = False

    opt = _make_optimizer(model, lr=step1_lr, weight_decay=step1_wd, optimizer_name=optimizer_name)
    scaler = _import_grad_scaler(device=device, enabled=bool(args.use_amp))
    warmup_sched, plateau_sched = _make_schedulers(
        opt,
        warmup_epochs=int(args.lr_warmup_epochs),
        plateau_factor=float(args.plateau_factor),
        plateau_patience=int(args.plateau_patience),
        lr_min=float(args.lr_min),
        plateau_mode=str(active_monitor_mode),
    )

    eval_index = 0
    phase_eval_index = 0
    global_step = 0
    phase1_steps_done = 0
    phase2_steps_done = 0
    best_global_metrics: Optional[Dict[str, Any]] = None
    best_global_step = 0
    best_global_phase = 0
    best_any_score = float("inf") if str(active_monitor_mode).lower() == "min" else float("-inf")
    best_any_metrics: Optional[Dict[str, Any]] = None
    best_any_step = 0
    best_any_phase = 0

    # === SANS_NPZ2_BEST_SLOPE_CKPT_V1 ===
    # Keep an independent checkpoint for the validation model whose slope is
    # closest to 1.0.  This does NOT replace best.ckpt; it is an audit/test
    # candidate for shrinkage-sensitive comparison.
    best_slope_penalty = float("inf")
    best_slope_tiebreak_mae = float("inf")
    best_slope_metrics: Optional[Dict[str, Any]] = None
    best_slope_step = 0
    best_slope_phase = 0
    # === END SANS_NPZ2_BEST_SLOPE_CKPT_V1 ===

    # === B4_MULTI_CHECKPOINTS_BEST_R2_V1 ===
    # Independent audit checkpoint: maximum validation R², tie-break lower MAE.
    # It never drives patience or replaces the official compromise checkpoint.
    best_r2_value = float("-inf")
    best_r2_tiebreak_mae = float("inf")
    best_r2_metrics: Optional[Dict[str, Any]] = None
    best_r2_step = 0
    best_r2_phase = 0
    # === END B4_MULTI_CHECKPOINTS_BEST_R2_V1 ===

    if args.resume is not None:
        info0 = load_checkpoint(
            args.resume,
            model=model,
            opt=None,
            device=device,
            strict=bool(args.resume_strict),
            scaler=None,
            scheduler=None,
        )
        extra0 = info0.get("extra_state", {}) or {}
        resumed_phase = int(extra0.get("phase", _resolve_phase_for_step(int(extra0.get("global_step", 0)), step1_max_steps, step2_max_steps)))

        if resumed_phase == 2:
            opt = _make_optimizer(model, lr=step2_lr, weight_decay=step2_wd, optimizer_name=optimizer_name)
            scaler = _import_grad_scaler(device=device, enabled=bool(args.use_amp))
            phase2_monitor_key, phase2_monitor_mode = _resolve_phase_monitor(args, 2)
            warmup_sched, plateau_sched = _make_schedulers(
                opt,
                warmup_epochs=int(args.step2_lr_warmup_epochs),
                plateau_factor=float(args.plateau_factor),
                plateau_patience=int(args.plateau_patience),
                lr_min=float(args.lr_min),
                plateau_mode=str(phase2_monitor_mode),
            )

        info = load_checkpoint(
            args.resume,
            model=model,
            opt=opt,
            device=device,
            strict=bool(args.resume_strict),
            scaler=scaler,
            scheduler=None,
        )
        extra = info.get("extra_state", {}) or {}

        active_phase = int(extra.get("phase", resumed_phase))
        active_monitor_key, active_monitor_mode = _resolve_phase_monitor(args, active_phase)
        global_best = float(info.get("best", global_best))
        phase_best = float(extra.get("phase_best", global_best))
        if "phase_best_metrics" in extra and isinstance(extra.get("phase_best_metrics"), dict):
            phase_best_metrics = dict(extra.get("phase_best_metrics") or {})
        patience_counter = int(extra.get("patience_counter", 0))
        eval_index = int(extra.get("cycle_index", extra.get("eval_index", info.get("cycle_index", info.get("epoch", 0)))))
        phase_eval_index = int(extra.get("phase_cycle", extra.get("phase_epoch", 0)))
        global_step = int(extra.get("global_step", 0))
        phase1_steps_done = int(extra.get("phase1_steps_done", min(global_step, step1_max_steps)))
        phase2_steps_done = int(extra.get("phase2_steps_done", max(0, global_step - step1_max_steps)))
        loaded_global_metrics = extra.get("best_global_metrics", extra.get("val_metrics", None))
        best_global_metrics = (
            dict(loaded_global_metrics)
            if isinstance(loaded_global_metrics, dict) and loaded_global_metrics
            else None
        )
        best_global_step = int(extra.get("best_global_step", global_step))
        best_global_phase = int(extra.get("best_global_phase", active_phase))

        # MAAMOURA_CHECKPOINT_RESUME_SAFETY_V1
        # Restore the raw-monitor best before the next validation. Without this,
        # best_any_score restarts at +inf and the first resumed validation can
        # overwrite a genuinely better best_any/best_compromise checkpoint.
        loaded_best_any_score = _sf(extra.get("best_any_score"), default=best_any_score)
        if math.isfinite(loaded_best_any_score):
            best_any_score = float(loaded_best_any_score)
        loaded_best_any_metrics = extra.get("best_any_metrics", None)
        if isinstance(loaded_best_any_metrics, dict) and loaded_best_any_metrics:
            best_any_metrics = dict(loaded_best_any_metrics)
        best_any_step = int(extra.get("best_any_step", best_any_step))
        best_any_phase = int(extra.get("best_any_phase", best_any_phase))
        if best_any_metrics is not None:
            print(
                f"[RESUME COMPOSITE BEST] score={best_any_score:.6f} | "
                f"step={best_any_step:05d} | phase={best_any_phase}",
                flush=True,
            )

        if "best_slope_penalty" in extra:
            loaded_slope_penalty = float(extra.get("best_slope_penalty", best_slope_penalty))
            loaded_slope_mae = float(extra.get("best_slope_tiebreak_mae", best_slope_tiebreak_mae))
            # Checkpoints written before B4_MULTI_CHECKPOINTS_V2 stored NaN
            # when no gate-eligible slope model existed.  Never propagate NaN
            # into comparisons, otherwise best_slope can no longer improve.
            if math.isfinite(loaded_slope_penalty):
                best_slope_penalty = loaded_slope_penalty
            if math.isfinite(loaded_slope_mae):
                best_slope_tiebreak_mae = loaded_slope_mae
            best_slope_metrics = extra.get("best_slope_metrics", best_slope_metrics)
            best_slope_step = int(extra.get("best_slope_step", best_slope_step))
            best_slope_phase = int(extra.get("best_slope_phase", best_slope_phase))

        # Recover an explicitly backfilled/previous best-slope sidecar when the
        # resumed last.ckpt predates independent slope checkpointing.
        if not math.isfinite(best_slope_penalty):
            slope_meta_path = Path(paths.get("checkpoints_dir", Path(paths["run_dir"]) / "checkpoints")) / "best_slope_meta.json"
            if slope_meta_path.is_file():
                try:
                    slope_meta_loaded = json.loads(slope_meta_path.read_text(encoding="utf-8"))
                    meta_slope = float(slope_meta_loaded.get("slope", float("nan")))
                    meta_penalty = float(slope_meta_loaded.get("slope_penalty", abs(1.0 - meta_slope)))
                    meta_mae = float(slope_meta_loaded.get("mae", float("inf")))
                    if math.isfinite(meta_penalty):
                        best_slope_penalty = meta_penalty
                    if math.isfinite(meta_mae):
                        best_slope_tiebreak_mae = meta_mae
                    best_slope_step = int(slope_meta_loaded.get("step", best_slope_step))
                    best_slope_phase = int(slope_meta_loaded.get("phase", best_slope_phase))
                    print(
                        f"[RESUME BEST SLOPE] penalty={best_slope_penalty:.4f} | "
                        f"mae={best_slope_tiebreak_mae:.4f} | meta={slope_meta_path}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[WARN] Could not restore best_slope_meta.json: {exc}", flush=True)

        # Restore independent best-R² state when resuming a patched run.
        if "best_r2_value" in extra:
            loaded_r2 = _sf(extra.get("best_r2_value"), default=best_r2_value)
            loaded_r2_mae = _sf(extra.get("best_r2_tiebreak_mae"), default=best_r2_tiebreak_mae)
            if math.isfinite(loaded_r2):
                best_r2_value = float(loaded_r2)
            if math.isfinite(loaded_r2_mae):
                best_r2_tiebreak_mae = float(loaded_r2_mae)
            loaded_r2_metrics = extra.get("best_r2_metrics", None)
            if isinstance(loaded_r2_metrics, dict) and loaded_r2_metrics:
                best_r2_metrics = dict(loaded_r2_metrics)
            best_r2_step = int(extra.get("best_r2_step", best_r2_step))
            best_r2_phase = int(extra.get("best_r2_phase", best_r2_phase))
            if best_r2_metrics is not None:
                print(
                    f"[RESUME BEST R²] r2={best_r2_value:.4f} | mae={best_r2_tiebreak_mae:.4f} | "
                    f"step={best_r2_step:05d} | phase={best_r2_phase}",
                    flush=True,
                )

        # Resume safety for checkpoints produced before the patience fix:
        # old best.ckpt files may contain patience_counter=20 even though the same
        # validation just updated the official safe best.ckpt. Reset on resume when
        # the checkpoint itself represents an official best update.
        resume_path_name = Path(args.resume).name.lower() if args.resume is not None else ""
        resume_is_official_best_update = (
            resume_path_name == "best.ckpt"
            or bool(extra.get("improved_global_after_checkpoint_gate", False))
            or str(extra.get("patience_reset_reason", "")).strip().lower() == "official_best_ckpt"
        )
        if resume_is_official_best_update and int(patience_counter) != 0:
            print(
                f"[RESUME] reset patience_counter {int(patience_counter)} -> 0 "
                f"because resume checkpoint is an official best.ckpt update.",
                flush=True,
            )
            patience_counter = 0

        is_resumed = True

        print(
            f"[RESUME] cycle_index={eval_index} | phase={active_phase} | "
            f"global_step={global_step} | phase1_steps={phase1_steps_done} | "
            f"phase2_steps={phase2_steps_done} | global_best={_safe_monitor_value_for_print(global_best)} | "
            f"monitor={active_monitor_key} ({active_monitor_mode}) | lr={opt.param_groups[0]['lr']:.2e}",
            flush=True,
        )

    if global_step >= total_max_steps:
        print(f"[WARN] resume global_step={global_step} >= total_max_steps={total_max_steps} -> nothing to do.", flush=True)
        return

    t0_global = time.time()

    while global_step < total_max_steps:
        if phase1_steps_done < step1_max_steps:
            phase = 1
            phase_steps_done = phase1_steps_done
            phase_steps_budget = step1_max_steps
            phase_patience_limit = step1_patience_evals
            current_warmup_evals = int(args.lr_warmup_epochs)
        elif step2_max_steps > 0:
            phase = 2
            phase_steps_done = phase2_steps_done
            phase_steps_budget = step2_max_steps
            phase_patience_limit = step2_patience_evals
            current_warmup_evals = int(args.step2_lr_warmup_epochs)
        else:
            break

        current_monitor_key, current_monitor_mode = _resolve_phase_monitor(args, phase)

        if int(phase) == 1:
            criterion.beta_ord = float(args.beta_ord)
            criterion.gamma_cls = float(args.gamma_cls)
        else:
            criterion.beta_ord = float(args.phase2_beta_ord if args.phase2_beta_ord is not None else args.beta_ord)
            criterion.gamma_cls = float(args.phase2_gamma_cls if args.phase2_gamma_cls is not None else args.gamma_cls)

        if phase != active_phase:
            print("\n[PHASE SWITCH] step-based transition -> phase 2", flush=True)

            if hasattr(criterion, "_phase2_cached_bin_edges"):
                delattr(criterion, "_phase2_cached_bin_edges")

            if bool(args.phase2_reload_best):
                phase1_paths = _phase_ckpt_paths(paths, 1)
                if phase1_paths["phase_best"].exists():
                    _ = load_checkpoint(
                        phase1_paths["phase_best"],
                        model=model,
                        opt=None,
                        device=device,
                        strict=False,
                        scaler=None,
                        scheduler=None,
                    )
                elif Path(paths["best_ckpt"]).exists():
                    _ = load_checkpoint(
                        paths["best_ckpt"],
                        model=model,
                        opt=None,
                        device=device,
                        strict=False,
                        scaler=None,
                        scheduler=None,
                    )

            opt = _make_optimizer(model, lr=step2_lr, weight_decay=step2_wd, optimizer_name=optimizer_name)
            scaler = _import_grad_scaler(device=device, enabled=bool(args.use_amp))
            warmup_sched, plateau_sched = _make_schedulers(
                opt,
                warmup_epochs=int(args.step2_lr_warmup_epochs),
                plateau_factor=float(args.plateau_factor),
                plateau_patience=int(args.plateau_patience),
                lr_min=float(args.lr_min),
                plateau_mode=str(current_monitor_mode),
            )
            patience_counter = 0
            phase_best = float("inf") if str(current_monitor_mode).lower() == "min" else float("-inf")
            phase_best_metrics = None
            phase_eval_index = 0

            if bool(args.phase2_reset_best_on_monitor_change) and (
                str(current_monitor_key).strip().lower() != str(active_monitor_key).strip().lower()
                or str(current_monitor_mode).strip().lower() != str(active_monitor_mode).strip().lower()
            ):
                print(
                    f"[PHASE 2 MONITOR] resetting best selection because monitor changes "
                    f"from {active_monitor_key} ({active_monitor_mode}) to {current_monitor_key} ({current_monitor_mode})",
                    flush=True,
                )
                global_best = float("inf") if str(current_monitor_mode).lower() == "min" else float("-inf")

            active_phase = 2
            active_monitor_key, active_monitor_mode = current_monitor_key, current_monitor_mode

        remaining_phase_steps = max(0, int(phase_steps_budget - phase_steps_done))
        if remaining_phase_steps <= 0:
            if phase == 1 and step2_max_steps > 0:
                active_phase = 1
                phase1_steps_done = int(step1_max_steps)
                continue
            break

        cycle_steps = min(int(val_every_steps), int(remaining_phase_steps))
        current_lr = float(opt.param_groups[0]["lr"])
        eval_index += 1
        phase_eval_index += 1

        cycle_loader = StepCycleLoader(train_loader, cycle_steps)

        t0 = time.time()

        train_metrics = train_one_cycle(
            model=model,
            loader=cycle_loader,
            criterion=criterion,
            opt=opt,
            scaler=scaler,
            device=device,
            huber_beta=float(args.huber_beta),
            report_bins=report_bins,
            cls_thresholds=cls_thresholds,
            auc_reservoir=int(AUC_RESERVOIR_DEFAULT),
            grad_clip=float(args.grad_clip),
            total_steps=cycle_steps,
            postfix_every=1,
            augment=bool(args.augment),
            tv_weight=float(args.tv_weight),
            cycle_idx=eval_index,
            phase=phase,
            current_lr=current_lr,
            best_score=float(global_best if math.isfinite(global_best) else 0.0),
            patience_counter=int(patience_counter),
            patience_limit=int(phase_patience_limit),
        )

        global_step += int(cycle_steps)
        if phase == 1:
            phase1_steps_done += int(cycle_steps)
        else:
            phase2_steps_done += int(cycle_steps)

        val_metrics = eval_one_cycle(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            huber_beta=float(args.huber_beta),
            report_bins=report_bins,
            cls_thresholds=cls_thresholds,
            auc_reservoir=int(AUC_RESERVOIR_DEFAULT),
            total_steps=val_steps,
            cycle_idx=eval_index,
            phase=phase,
        ) if val_loader is not None else {}

        score_source = val_metrics if val_metrics else train_metrics
        score, monitor_details = _compute_monitor_score(
            score_source,
            current_monitor_key,
            phase2_high_threshold=float(args.phase2_monitor_high_threshold),
            phase2_global_weight=float(args.phase2_monitor_global_weight),
            phase2_high_weight=float(args.phase2_monitor_high_weight),
        )
        checkpoint_eligible, checkpoint_gate_details = _checkpoint_eligibility_from_metrics(args, score_source)
        prev_phase_best_score = float(phase_best)
        prev_global_best_score = float(global_best)
        prev_phase_best_metrics = dict(phase_best_metrics) if isinstance(phase_best_metrics, dict) else None
        prev_global_best_metrics = dict(best_global_metrics) if isinstance(best_global_metrics, dict) else None
        improved_phase = _is_better(score, phase_best, current_monitor_mode)
        raw_improved_global = _is_better(score, global_best, current_monitor_mode)
        official_gain_ok, official_replacement_details = _official_best_replacement_allowed(
            score,
            global_best,
            current_monitor_mode,
            min_rel_gain=float(args.official_best_min_rel_gain),
            min_abs_gain=float(args.official_best_min_abs_gain),
        )
        improved_any = _is_better(score, best_any_score, current_monitor_mode)
        improved_global = bool(raw_improved_global and checkpoint_eligible and official_gain_ok)

        # Patience policy:
        # - improved_phase tracks the raw phase monitor.
        # - improved_global tracks the official best.ckpt AFTER the checkpoint-safe gate.
        #
        # Before this fix, an official safe checkpoint could be saved (🌟 BEST MODEL UPDATED)
        # while patience_counter still increased because an earlier raw/ineligible candidate
        # had already set phase_best. This could switch to phase 2 immediately after a
        # valid best.ckpt update. Any official best.ckpt update must therefore reset
        # the no-improvement counter as well.
        if improved_phase:
            phase_best = float(score)
            phase_best_metrics = dict(score_source)

        patience_reset_reason = None
        if improved_global:
            patience_counter = 0
            patience_reset_reason = "official_best_ckpt"
        elif improved_phase:
            patience_counter = 0
            patience_reset_reason = "phase_monitor"
        else:
            patience_counter += 1
            patience_reset_reason = "no_improvement"

        if improved_any:
            best_any_score = float(score)
            best_any_metrics = dict(score_source)
            best_any_step = int(global_step)
            best_any_phase = int(phase)

        if improved_global:
            global_best = float(score)
            best_global_metrics = dict(score_source)
            best_global_step = int(global_step)
            best_global_phase = int(phase)

        # === SANS_NPZ2_BEST_SLOPE_CKPT_V1 ===
        slope_for_best = _sf(score_source.get("slope"), default=float("nan"))
        mae_for_best_slope = _sf(score_source.get("mae"), default=float("inf"))
        bias_for_best_slope = _sf(score_source.get("bias"), default=float("nan"))
        stdr_for_best_slope = _sf(score_source.get("std_ratio"), default=_sf(score_source.get("stdr"), default=float("nan")))
        slope_penalty_now = abs(1.0 - slope_for_best) if math.isfinite(slope_for_best) else float("inf")
        # === B4_MULTI_CHECKPOINTS_V2 ===
        # best_slope is deliberately independent from the official best.ckpt
        # safety gate.  It is an audit candidate selected only by closeness of
        # the validation slope to 1 (tie-break: lower MAE), and is therefore
        # saved from the first finite validation onward.
        best_slope_eligible = (
            math.isfinite(slope_penalty_now)
            and math.isfinite(mae_for_best_slope)
            and math.isfinite(bias_for_best_slope)
            and math.isfinite(stdr_for_best_slope)
            and float(stdr_for_best_slope) > 0.0
        )
        # === END B4_MULTI_CHECKPOINTS_V2 ===
        improved_best_slope = False
        if best_slope_eligible:
            improved_best_slope = (
                slope_penalty_now < (best_slope_penalty - 1e-6)
                or (
                    abs(slope_penalty_now - best_slope_penalty) <= 1e-6
                    and float(mae_for_best_slope) < float(best_slope_tiebreak_mae)
                )
            )
        if improved_best_slope:
            best_slope_penalty = float(slope_penalty_now)
            best_slope_tiebreak_mae = float(mae_for_best_slope)
            best_slope_metrics = dict(score_source)
            best_slope_step = int(global_step)
            best_slope_phase = int(phase)
        # === END SANS_NPZ2_BEST_SLOPE_CKPT_V1 ===

        # === B4_MULTI_CHECKPOINTS_BEST_R2_V1 ===
        r2_for_best = _sf(score_source.get("r2"), default=float("nan"))
        mae_for_best_r2 = _sf(score_source.get("mae"), default=float("inf"))
        best_r2_eligible = math.isfinite(r2_for_best) and math.isfinite(mae_for_best_r2)
        improved_best_r2 = False
        if best_r2_eligible:
            improved_best_r2 = (
                r2_for_best > (best_r2_value + 1e-6)
                or (
                    abs(r2_for_best - best_r2_value) <= 1e-6
                    and mae_for_best_r2 < best_r2_tiebreak_mae
                )
            )
        if improved_best_r2:
            best_r2_value = float(r2_for_best)
            best_r2_tiebreak_mae = float(mae_for_best_r2)
            best_r2_metrics = dict(score_source)
            best_r2_step = int(global_step)
            best_r2_phase = int(phase)
        # === END B4_MULTI_CHECKPOINTS_BEST_R2_V1 ===

        elapsed = time.time() - t0

        extra_state = {
            "epoch_time_sec": float(elapsed),
            "phase": int(phase),
            "phase_cycle": int(phase_eval_index),
            "phase_epoch": int(phase_eval_index),
            "cycle_index": int(eval_index),
            "eval_index": int(eval_index),
            "global_step": int(global_step),
            "phase1_steps_done": int(phase1_steps_done),
            "phase2_steps_done": int(phase2_steps_done),
            "cycle_steps": int(cycle_steps),
            "phase_best": float(phase_best),
            "phase_best_metrics": phase_best_metrics if phase_best_metrics is not None else {},
            "global_best": float(global_best),
            "best_global_metrics": best_global_metrics if best_global_metrics is not None else {},
            "best_global_step": int(best_global_step),
            "best_global_phase": int(best_global_phase),
            "patience_counter": int(patience_counter),
            "monitor": str(current_monitor_key),
            "monitor_mode": str(current_monitor_mode),
            "monitor_score": float(score) if math.isfinite(score) else float("nan"),
            "monitor_details": monitor_details,
            "checkpoint_eligibility": checkpoint_gate_details,
            "raw_improved_global": bool(raw_improved_global),
            "official_replacement_details": official_replacement_details,
            "improved_global_after_checkpoint_gate": bool(improved_global),
            "patience_reset_reason": str(patience_reset_reason),
            "best_any_score": float(best_any_score) if math.isfinite(best_any_score) else float("nan"),
            "best_any_metrics": best_any_metrics if best_any_metrics is not None else {},
            "best_any_step": int(best_any_step),
            "best_any_phase": int(best_any_phase),
            "best_slope_penalty": float(best_slope_penalty) if math.isfinite(best_slope_penalty) else float("nan"),
            "best_slope_tiebreak_mae": float(best_slope_tiebreak_mae) if math.isfinite(best_slope_tiebreak_mae) else float("nan"),
            "best_slope_metrics": best_slope_metrics if best_slope_metrics is not None else {},
            "best_slope_step": int(best_slope_step),
            "best_slope_phase": int(best_slope_phase),
            "best_r2_value": float(best_r2_value) if math.isfinite(best_r2_value) else float("nan"),
            "best_r2_tiebreak_mae": float(best_r2_tiebreak_mae) if math.isfinite(best_r2_tiebreak_mae) else float("nan"),
            "best_r2_metrics": best_r2_metrics if best_r2_metrics is not None else {},
            "best_r2_step": int(best_r2_step),
            "best_r2_phase": int(best_r2_phase),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "step_schedule": schedule,
            "training_mode": effective_training_mode,
        }

        phase_paths = _phase_ckpt_paths(paths, phase)

        save_checkpoint(
            paths["last_ckpt"],
            cycle_index=eval_index,
            best=global_best,
            model=model,
            opt=opt,
            scaler=scaler,
            scheduler=None,
            model_type=model_type,
            extra_state=extra_state,
        )
        save_checkpoint(
            phase_paths["phase_last"],
            epoch=eval_index,
            best=phase_best,
            model=model,
            opt=opt,
            scaler=scaler,
            scheduler=None,
            model_type=model_type,
            extra_state=extra_state,
        )

        if improved_phase:
            save_checkpoint(
                phase_paths["phase_best"],
                cycle_index=eval_index,
                best=phase_best,
                model=model,
                opt=opt,
                scaler=scaler,
                scheduler=None,
                model_type=model_type,
                extra_state=extra_state,
            )

        if improved_any:
            ckdir_any = Path(paths.get("checkpoints_dir", Path(paths["run_dir"]) / "checkpoints"))
            ckdir_any.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                ckdir_any / "best_any.ckpt",
                epoch=eval_index,
                best=best_any_score,
                model=model,
                opt=opt,
                scaler=scaler,
                scheduler=None,
                model_type=model_type,
                extra_state=extra_state,
            )
            # === SANS_NPZ2_BEST_COMPROMISE_ALIAS_V1 ===
            # In the clean Sans_NPZ_2 protocol, the raw monitor is an
            # article-level compromise score.  Keep a named alias so that
            # manual interruption still leaves an explicit compromise model.
            save_checkpoint(
                ckdir_any / "best_compromise.ckpt",
                epoch=eval_index,
                best=best_any_score,
                model=model,
                opt=opt,
                scaler=scaler,
                scheduler=None,
                model_type=model_type,
                extra_state={**extra_state, "checkpoint_alias": "best_compromise_raw_monitor"},
            )
            # === END SANS_NPZ2_BEST_COMPROMISE_ALIAS_V1 ===

            # === COMPOSITE_CHECKPOINT_RESUME_SAFETY_V2 ===
            # best.ckpt is reserved for a gate-eligible official model.
            # Unsafe/raw improvements already live in best_any.ckpt and
            # best_compromise.ckpt and must never masquerade as official.
            print(
                f"💾 [COMPOSITE CHECKPOINTS SAVED] step={int(global_step):05d} | "
                f"score={float(best_any_score):.6f} | "
                f"best_any={ckdir_any / 'best_any.ckpt'} | "
                f"best_compromise={ckdir_any / 'best_compromise.ckpt'}",
                flush=True,
            )
            # === END MAAMOURA_CHECKPOINT_RESUME_SAFETY_V1 ===

        # === SANS_NPZ2_BEST_SLOPE_CKPT_V1 ===
        if improved_best_slope:
            ckdir_slope = Path(paths.get("checkpoints_dir", Path(paths["run_dir"]) / "checkpoints"))
            ckdir_slope.mkdir(parents=True, exist_ok=True)
            best_slope_extra = {
                **extra_state,
                "checkpoint_alias": "best_slope_closest_to_one",
                "best_slope_penalty": float(best_slope_penalty),
                "best_slope_tiebreak_mae": float(best_slope_tiebreak_mae),
                "best_slope_step": int(best_slope_step),
                "best_slope_phase": int(best_slope_phase),
                "best_slope_metrics": best_slope_metrics if best_slope_metrics is not None else {},
            }
            save_checkpoint(
                ckdir_slope / "best_slope.ckpt",
                epoch=eval_index,
                best=best_slope_penalty,
                model=model,
                opt=opt,
                scaler=scaler,
                scheduler=None,
                model_type=model_type,
                extra_state=best_slope_extra,
            )
            slope_meta = {
                "checkpoint": str(ckdir_slope / "best_slope.ckpt"),
                "selection": "minimum abs(1-slope), tie-break lower MAE",
                "step": int(best_slope_step),
                "phase": int(best_slope_phase),
                "slope": float(slope_for_best),
                "slope_penalty": float(best_slope_penalty),
                "mae": float(mae_for_best_slope),
                "bias": float(bias_for_best_slope),
                "std_ratio": float(stdr_for_best_slope),
                "checkpoint_eligible": bool(checkpoint_eligible),
                "metrics": best_slope_metrics if best_slope_metrics is not None else {},
            }
            (ckdir_slope / "best_slope_meta.json").write_text(
                json.dumps(slope_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"📐 [BEST SLOPE CHECKPOINT SAVED] step={int(global_step):05d} | "
                f"phase={int(phase)} | slope={float(slope_for_best):.4f} | "
                f"|1-slope|={float(best_slope_penalty):.4f} | mae={float(mae_for_best_slope):.4f} | "
                f"path={ckdir_slope / 'best_slope.ckpt'}",
                flush=True,
            )
        # === END SANS_NPZ2_BEST_SLOPE_CKPT_V1 ===

        # === B4_MULTI_CHECKPOINTS_BEST_R2_V1 ===
        if improved_best_r2:
            ckdir_r2 = Path(paths.get("checkpoints_dir", Path(paths["run_dir"]) / "checkpoints"))
            ckdir_r2.mkdir(parents=True, exist_ok=True)
            best_r2_extra = {
                **extra_state,
                "checkpoint_alias": "best_r2_maximum_validation",
                "best_r2_value": float(best_r2_value),
                "best_r2_tiebreak_mae": float(best_r2_tiebreak_mae),
                "best_r2_step": int(best_r2_step),
                "best_r2_phase": int(best_r2_phase),
                "best_r2_metrics": best_r2_metrics if best_r2_metrics is not None else {},
            }
            save_checkpoint(
                ckdir_r2 / "best_r2.ckpt",
                epoch=eval_index,
                best=best_r2_value,
                model=model,
                opt=opt,
                scaler=scaler,
                scheduler=None,
                model_type=model_type,
                extra_state=best_r2_extra,
            )
            r2_meta = {
                "checkpoint": str(ckdir_r2 / "best_r2.ckpt"),
                "selection": "maximum validation R², tie-break lower MAE",
                "step": int(best_r2_step),
                "phase": int(best_r2_phase),
                "r2": float(best_r2_value),
                "mae": float(best_r2_tiebreak_mae),
                "checkpoint_eligible": bool(checkpoint_eligible),
                "metrics": best_r2_metrics if best_r2_metrics is not None else {},
            }
            (ckdir_r2 / "best_r2_meta.json").write_text(
                json.dumps(r2_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"🏆 [BEST R² CHECKPOINT SAVED] step={int(global_step):05d} | "
                f"phase={int(phase)} | r2={float(best_r2_value):.4f} | "
                f"mae={float(best_r2_tiebreak_mae):.4f} | path={ckdir_r2 / 'best_r2.ckpt'}",
                flush=True,
            )
        # === END B4_MULTI_CHECKPOINTS_BEST_R2_V1 ===

        if improved_global:
            save_checkpoint(
                paths["best_ckpt"],
                epoch=eval_index,
                best=global_best,
                model=model,
                opt=opt,
                scaler=scaler,
                scheduler=None,
                model_type=model_type,
                extra_state={
                    **extra_state,
                    "checkpoint_alias": "best_official_safe_monitor",
                    "checkpoint_is_official": True,
                    "checkpoint_selection": "official monitor improvement after checkpoint safety gate",
                },
            )

        row = _make_log_row_cycle(
            cycle_index=eval_index,
            train_metrics=train_metrics,
            val_metrics=val_metrics if val_metrics else {},
            time_sec=float(elapsed),
            lr=current_lr,
            best=global_best,
            model_type=model_type,
            is_resume=is_resumed,
            training_mode=effective_training_mode,
            phase=phase,
            phase_cycle_index=phase_eval_index,
            global_step=global_step,
            phase1_steps_done=phase1_steps_done,
            phase2_steps_done=phase2_steps_done,
            cycle_steps=cycle_steps,
            step_schedule=schedule,
        )
        append_csv(paths["train_csv"], row)

        current_phase_step = int(phase1_steps_done if phase == 1 else phase2_steps_done)
        _print_pretty_window_block(
            global_step=int(global_step),
            total_max_steps=int(total_max_steps),
            phase=int(phase),
            phase_step=int(current_phase_step),
            phase_step_budget=int(phase_steps_budget),
            window_start_step=int(global_step - cycle_steps + 1),
            window_end_step=int(global_step),
            lr=float(current_lr),
            train_metrics=train_metrics,
            val_metrics=val_metrics if val_metrics else {},
            improved_phase=bool(improved_phase),
            raw_improved_global=bool(raw_improved_global),
            improved_global=bool(improved_global),
            checkpoint_eligible=bool(checkpoint_eligible),
            checkpoint_gate_details=checkpoint_gate_details,
            patience_reset_reason=str(patience_reset_reason),
            patience_counter=int(patience_counter),
            patience_limit=int(phase_patience_limit),
            best_metrics=best_global_metrics,
            best_step=int(best_global_step),
            best_phase=int(best_global_phase),
            raw_best_metrics=best_any_metrics,
            raw_best_step=int(best_any_step),
            raw_best_phase=int(best_any_phase),
            best_ckpt_path=paths.get("best_ckpt"),
            monitor_score=float(score) if math.isfinite(score) else float("nan"),
            monitor_details=monitor_details,
            official_replacement_details=official_replacement_details,
            prev_phase_best_score=prev_phase_best_score,
            prev_global_best_score=prev_global_best_score,
            prev_phase_best_metrics=prev_phase_best_metrics,
            prev_global_best_metrics=prev_global_best_metrics,
            monitor_mode=str(current_monitor_mode),
        )

        if (not simple_logs) and bool(args.print_all_metrics):
            _print_cycle_block(
                cycle_index=eval_index,
                phase=phase,
                phase_cycle_index=phase_eval_index,
                phase_best=float(phase_best),
                global_best=float(global_best),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr,
                score=float(score) if math.isfinite(score) else float("nan"),
                improved_phase=bool(improved_phase),
                improved_global=bool(improved_global),
                monitor=str(current_monitor_key),
                patience_counter=int(patience_counter),
                patience_limit=int(phase_patience_limit),
                print_all_metrics=bool(args.print_all_metrics),
                global_step=int(global_step),
                total_max_steps=int(total_max_steps),
                phase_steps_done=int(current_phase_step),
                phase_steps_budget=int(phase_steps_budget),
                cycle_steps=int(cycle_steps),
                monitor_details=monitor_details,
            )

        try:
            if phase_eval_index <= current_warmup_evals and current_warmup_evals > 0:
                warmup_sched.step()
            elif math.isfinite(score):
                plateau_sched.step(float(score))
        except Exception:
            pass

        if int(patience_counter) >= int(phase_patience_limit):
            if phase == 1 and step2_max_steps > 0:
                print(
                    f"\n[PHASE 1 EARLY STOP] patience reached after {phase1_steps_done} steps. "
                    f"Switching to phase 2.",
                    flush=True,
                )
                phase1_steps_done = int(step1_max_steps)
                continue
            else:
                print(
                    f"\n[EARLY STOPPING] patience reached in phase {phase}: "
                    f"{patience_counter}/{phase_patience_limit}. Stopping at global_step={global_step}.",
                    flush=True,
                )
                break

    train_total_elapsed = time.time() - t0_global
    print(f"\\n[TRAIN DONE] total_time_sec={train_total_elapsed:.1f} | global_step={global_step}", flush=True)

    best_ckpt_path = Path(paths["best_ckpt"])
    best_any_ckpt_path = Path(paths.get("checkpoints_dir", Path(paths["run_dir"]) / "checkpoints")) / "best_any.ckpt"
    has_official_best = bool(
        best_ckpt_path.exists()
        and isinstance(best_global_metrics, dict)
        and bool(best_global_metrics)
        and int(best_global_step) > 0
    )
    test_ckpt_path = best_ckpt_path if has_official_best else best_any_ckpt_path
    if bool(args.eval_test_at_end) and test_loader is not None and test_ckpt_path.exists():
        if test_ckpt_path == best_ckpt_path:
            print("\n[TEST] Loading official best checkpoint for final evaluation...", flush=True)
        else:
            print(
                "\n[TEST] No gate-eligible official best.ckpt; "
                "loading best_any.ckpt for explicit audit evaluation.",
                flush=True,
            )
        info = load_checkpoint(
            test_ckpt_path,
            model=model,
            opt=None,
            device=device,
            strict=False,
            scaler=None,
            scheduler=None,
        )
        best_extra = info.get("extra_state", {}) or {}
        best_eval_phase = int(best_extra.get("phase", _resolve_phase_for_step(int(best_extra.get("global_step", global_step)), step1_max_steps, step2_max_steps)))

        if int(best_eval_phase) == 1:
            criterion.beta_ord = float(args.beta_ord)
            criterion.gamma_cls = float(args.gamma_cls)
        else:
            criterion.beta_ord = float(args.phase2_beta_ord if args.phase2_beta_ord is not None else args.beta_ord)
            criterion.gamma_cls = float(args.phase2_gamma_cls if args.phase2_gamma_cls is not None else args.gamma_cls)

        test_metrics = eval_one_cycle(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            huber_beta=float(args.huber_beta),
            report_bins=report_bins,
            cls_thresholds=cls_thresholds,
            auc_reservoir=int(AUC_RESERVOIR_DEFAULT),
            total_steps=test_steps,
            cycle_idx=0,
            phase=best_eval_phase,
        )

        test_reporting = _gedi_unique_reporting_summary(test_metrics, "test")
        save_json_artifact(
            paths["test_metrics_json"],
            {
                "experiment_root": str(args.experiment_root),
                "run_dir": str(paths["run_dir"]),
                "best_ckpt": str(test_ckpt_path),
                "checkpoint_status": "official_best" if test_ckpt_path == best_ckpt_path else "best_any_gate_fallback",
                "best_phase": int(best_eval_phase),
                "best_global_step": int(best_extra.get("global_step", global_step)),
                "best_monitor": str(best_extra.get("monitor", "")),
                "best_monitor_mode": str(best_extra.get("monitor_mode", "")),
                "best_monitor_score": float(best_extra.get("monitor_score", float("nan"))),
                "best_monitor_details": best_extra.get("monitor_details", {}),
                "reporting_protocol": {
                    "best_checkpoint_monitor": str(best_extra.get("monitor", args.monitor)),
                    "primary_test_metric": "test_mae_unique_nearest_ge2",
                    "secondary_test_metric": "test_mae_unique_temporal_pred_mean_ge2",
                    "diagnostic_test_metric": "test_mae_unique_temporal_error_mean_ge2",
                    "domain": (
                        f"RH95 in [{float(args.eval_primary_min_height):g},"
                        f"{float(args.eval_primary_max_height):g}] m"
                    ),
                },
                "test_reporting_summary": test_reporting,
                "test_metrics": test_metrics,
            },
        )

        _print_test_block(test_metrics, print_all_metrics=bool(args.print_all_metrics))
        print("[TEST REPORTING SUMMARY]", test_reporting, flush=True)
        print(f"[TEST] metrics saved -> {paths['test_metrics_json']}", flush=True)

    print(f"\\n[ARTIFACTS] run_dir={paths['run_dir']}", flush=True)
    print(f"[ARTIFACTS] best_ckpt={paths['best_ckpt']}", flush=True)
    print(f"[ARTIFACTS] last_ckpt={paths['last_ckpt']}", flush=True)
    print(f"[ARTIFACTS] best_any_ckpt={Path(paths.get('checkpoints_dir', Path(paths['run_dir']) / 'checkpoints')) / 'best_any.ckpt'}", flush=True)
    print(f"[ARTIFACTS] best_compromise_ckpt={Path(paths.get('checkpoints_dir', Path(paths['run_dir']) / 'checkpoints')) / 'best_compromise.ckpt'}", flush=True)
    print(f"[ARTIFACTS] best_slope_ckpt={Path(paths.get('checkpoints_dir', Path(paths['run_dir']) / 'checkpoints')) / 'best_slope.ckpt'}", flush=True)
    print(f"[ARTIFACTS] best_r2_ckpt={Path(paths.get('checkpoints_dir', Path(paths['run_dir']) / 'checkpoints')) / 'best_r2.ckpt'}", flush=True)
    print(f"[ARTIFACTS] train_csv={paths['train_csv']}", flush=True)


if __name__ == "__main__":
    main()

