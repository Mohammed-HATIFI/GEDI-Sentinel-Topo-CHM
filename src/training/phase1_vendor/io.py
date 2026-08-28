from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from training.common import (
    CLS_THRESH_DEFAULT,
    HEIGHT_CLASS_KEYS,
    NUM_HEIGHT_CLASSES,
)


# ======================================================================================
# CSV LOGGING — dynamic version (multi-forest, 5 classes)
# Canonical wording is now CYCLE-based for step-based training.
# Legacy aliases are preserved to avoid breaking the rest of the pipeline.
# ======================================================================================
_GROUP_NAMES = list(HEIGHT_CLASS_KEYS)
_CLS_THRESH = [float(x) for x in CLS_THRESH_DEFAULT]
_CLS_METRICS = ["prec", "rec", "f1", "spec", "acc", "bal_acc", "iou", "dice", "auroc", "auprc"]
_REG_METRICS = ["n", "mae", "mse", "rmse", "r2", "bias"]


# Prefer a step/cycle vocabulary, but keep epoch aliases where useful.
CANONICAL_INDEX_FIELD = "Cycle_Index"
LEGACY_INDEX_FIELD = "Epoch"
CANONICAL_PHASE_INDEX_FIELD = "Phase_Cycle_Index"
LEGACY_PHASE_INDEX_FIELD = "Phase_Eval_Index"



def _cls_key(th: float) -> str:
    return f"cls{int(th)}" if float(th).is_integer() else f"cls{th}"


REQUIRED_FIELDS: List[str] = [
    CANONICAL_INDEX_FIELD,
    LEGACY_INDEX_FIELD,  # legacy alias
    "Train_Loss_Huber",
    "Val_Loss_Huber",
    "Val_MAE",
    "Val_RMSE",
    "Val_MSE",
    "Val_R2",
]
for _t in _CLS_THRESH:
    _k = _cls_key(_t).replace("cls", "")
    REQUIRED_FIELDS.append(f"Val_IoU{_k}")
    REQUIRED_FIELDS.append(f"Val_Dice{_k}")


BASE_FIELDS: List[str] = [
    "Time_sec",
    "LR",
    "Best",
    "Train_Loss_Total",
    "Val_Loss_Total",
    "Train_Loss_ClsAux",
    "Val_Loss_ClsAux",
    "Train_MAE",
    "Train_RMSE",
    "Train_MSE",
    "Train_R2",
    "Train_Bias",
    "Val_Bias",
    "Train_Avg_Pts_Per_Patch",
    "Val_Avg_Pts_Per_Patch",
    "Train_N_Points",
    "Val_N_Points",
    "Val_Std_Ratio",
    "Model_Type",
    "Is_Resume",
]


STEP_FIELDS: List[str] = [
    "Training_Mode",
    "Phase",
    CANONICAL_PHASE_INDEX_FIELD,
    LEGACY_PHASE_INDEX_FIELD,  # legacy alias
    "Global_Step",
    "Phase1_Steps_Done",
    "Phase2_Steps_Done",
    "Cycle_Steps",
    "Step1_Max_Steps",
    "Step2_Max_Steps",
    "Val_Every_Steps",
]


# Height-bin CSV fields are generated dynamically from training.common. With common.py v3 they follow --bins 0,5,10,15: lt0 / 0_5 / 5_10 / 10_15 / ge15.
# GEDI-unique metrics used for model selection and paper reporting.
# These are computed by training.trainloop.eval_one_cycle from aux_shot_uid
# metadata produced by Step04 GEDIIDS_UNIQUEVAL.
UNIQUE_FIELDS: List[str] = []
for _sp in ("Train", "Val"):
    UNIQUE_FIELDS.extend([
        # FORMS-T-style occurrence-level metrics by domain.
        # These allow monitoring with val_mae_occurrence_ge2 while keeping the
        # training loss as ordinary sparse GEDI MAE.
        f"{_sp}_MAE_Occurrence_Eval15",
        f"{_sp}_RMSE_Occurrence_Eval15",
        f"{_sp}_R2_Occurrence_Eval15",
        f"{_sp}_Bias_Occurrence_Eval15",
        f"{_sp}_Slope_Occurrence_Eval15",
        f"{_sp}_Std_Ratio_Occurrence_Eval15",
        f"{_sp}_N_Occurrence_Eval15",
        f"{_sp}_MAE_Occurrence_GE2",
        f"{_sp}_RMSE_Occurrence_GE2",
        f"{_sp}_R2_Occurrence_GE2",
        f"{_sp}_Bias_Occurrence_GE2",
        f"{_sp}_Slope_Occurrence_GE2",
        f"{_sp}_Std_Ratio_Occurrence_GE2",
        f"{_sp}_N_Occurrence_GE2",
        f"{_sp}_MAE_Occurrence_Low_0_2",
        f"{_sp}_N_Occurrence_Low_0_2",

        f"{_sp}_MAE_Unique_Temporal_Error_Mean_GE2",
        f"{_sp}_Median_Unique_Temporal_Error_Mean_GE2",
        f"{_sp}_N_Unique_Temporal_Error_Mean_GE2",
        f"{_sp}_MAE_Unique_Nearest_GE2",
        f"{_sp}_RMSE_Unique_Nearest_GE2",
        f"{_sp}_R2_Unique_Nearest_GE2",
        f"{_sp}_N_Unique_Nearest_GE2",
        f"{_sp}_MAE_Unique_Temporal_Pred_Mean_GE2",
        f"{_sp}_RMSE_Unique_Temporal_Pred_Mean_GE2",
        f"{_sp}_R2_Unique_Temporal_Pred_Mean_GE2",
        f"{_sp}_Temporal_Pred_Std_Mean_GE2",
        f"{_sp}_Temporal_Pred_Std_Median_GE2",
        f"{_sp}_GEDI_Unique_Metadata_Available",
    ])


GROUP_FIELDS: List[str] = []
for _sp in ("Train", "Val"):
    for _g in _GROUP_NAMES:
        for _m in _REG_METRICS:
            GROUP_FIELDS.append(f"{_sp}_g_{_g}_{_m}")
    GROUP_FIELDS.append(f"{_sp}_g_acc{NUM_HEIGHT_CLASSES}")
    for _i in range(NUM_HEIGHT_CLASSES):
        for _j in range(NUM_HEIGHT_CLASSES):
            GROUP_FIELDS.append(f"{_sp}_g_cm_{_i}{_j}")


CLS_FIELDS: List[str] = []
for _sp in ("Train", "Val"):
    for _t in _CLS_THRESH:
        _k = _cls_key(_t)
        for _m in _CLS_METRICS:
            CLS_FIELDS.append(f"{_sp}_{_k}_{_m}")
        for _g in _GROUP_NAMES:
            for _m in _CLS_METRICS:
                CLS_FIELDS.append(f"{_sp}_{_k}_g_{_g}_{_m}")


LOG_FIELDS = REQUIRED_FIELDS + BASE_FIELDS + STEP_FIELDS + UNIQUE_FIELDS + GROUP_FIELDS + CLS_FIELDS


# ======================================================================================
# Small helpers
# ======================================================================================
def _sf(v: Any, default: Any = "") -> Any:
    try:
        x = float(v)
        return x
    except Exception:
        return default



def _bool01(v: Any) -> int:
    return 1 if bool(v) else 0



def _get(d: Dict[str, Any], key: str, default: Any = "") -> Any:
    return d.get(key, default)


# ======================================================================================
# CSV helpers
# ======================================================================================
def init_csv(path: Path, *, force: bool = False) -> None:
    """
    Crée le CSV et écrit l'entête LOG_FIELDS.
    Sécurité: si le fichier existe mais l'entête ne correspond pas à LOG_FIELDS,
    on lève une erreur (ou on réécrit si force=True).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None) or []

        if header != LOG_FIELDS:
            msg = (
                f"[CSV] Header mismatch for {path}.\n"
                f"  Found   : {len(header)} cols\n"
                f"  Expected: {len(LOG_FIELDS)} cols\n"
                f"Fix: delete the existing CSV or call init_csv(..., force=True)."
            )
            if not force:
                raise RuntimeError(msg)

            with path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()



def append_csv(path: Path, row: Dict[str, Any]) -> None:
    """
    Append une ligne au CSV en complétant les champs manquants par "".
    Sécurité: si le fichier n'existe pas, on crée l'entête.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        init_csv(path)

    full = {k: row.get(k, "") for k in LOG_FIELDS}
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writerow(full)


# ======================================================================================
# Row builders from train/val metrics
# ======================================================================================
def make_cycle_row(
    *,
    cycle_index: int,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    time_sec: float,
    lr: float,
    best: float,
    model_type: str,
    is_resume: bool,
    training_mode: str = "",
    phase: Optional[int] = None,
    phase_cycle_index: Optional[int] = None,
    global_step: Optional[int] = None,
    phase1_steps_done: Optional[int] = None,
    phase2_steps_done: Optional[int] = None,
    cycle_steps: Optional[int] = None,
    step_schedule: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Canonical row builder for step-based training.

    Notes
    -----
    - ``Cycle_Index`` is the canonical index field.
    - ``Epoch`` is preserved as a legacy alias with the same value.
    - ``Phase_Cycle_Index`` is canonical; ``Phase_Eval_Index`` is kept as alias.
    """
    step_schedule = dict(step_schedule or {})
    cycle_index = int(cycle_index)
    phase_cycle_value = int(phase_cycle_index) if phase_cycle_index is not None else ""

    row: Dict[str, Any] = {
        CANONICAL_INDEX_FIELD: cycle_index,
        LEGACY_INDEX_FIELD: cycle_index,
        "Time_sec": _sf(time_sec),
        "LR": _sf(lr),
        "Best": _sf(best),
        "Train_Loss_Total": _sf(_get(train_metrics, "loss_total")),
        "Val_Loss_Total": _sf(_get(val_metrics, "loss_total")),
        "Train_Loss_Huber": _sf(_get(train_metrics, "loss_huber")),
        "Val_Loss_Huber": _sf(_get(val_metrics, "loss_huber")),
        "Train_Loss_ClsAux": _sf(_get(train_metrics, "loss_cls_aux", _get(train_metrics, "loss_cls4"))),
        "Val_Loss_ClsAux": _sf(_get(val_metrics, "loss_cls_aux", _get(val_metrics, "loss_cls4"))),
        "Train_MAE": _sf(_get(train_metrics, "mae")),
        "Train_RMSE": _sf(_get(train_metrics, "rmse")),
        "Train_MSE": _sf(_get(train_metrics, "mse")),
        "Train_R2": _sf(_get(train_metrics, "r2")),
        "Train_Bias": _sf(_get(train_metrics, "bias")),
        "Val_MAE": _sf(_get(val_metrics, "mae")),
        "Val_RMSE": _sf(_get(val_metrics, "rmse")),
        "Val_MSE": _sf(_get(val_metrics, "mse")),
        "Val_R2": _sf(_get(val_metrics, "r2")),
        "Val_Bias": _sf(_get(val_metrics, "bias")),
        "Train_Avg_Pts_Per_Patch": _sf(_get(train_metrics, "avg_pts_per_patch")),
        "Val_Avg_Pts_Per_Patch": _sf(_get(val_metrics, "avg_pts_per_patch")),
        "Train_N_Points": int(_get(train_metrics, "n_points", 0)),
        "Val_N_Points": int(_get(val_metrics, "n_points", 0)),
        "Val_Std_Ratio": _sf(_get(val_metrics, "std_ratio")),
        "Model_Type": str(model_type),
        "Is_Resume": _bool01(is_resume),
        "Training_Mode": str(training_mode) if training_mode is not None else "",
        "Phase": int(phase) if phase is not None else "",
        CANONICAL_PHASE_INDEX_FIELD: phase_cycle_value,
        LEGACY_PHASE_INDEX_FIELD: phase_cycle_value,
        "Global_Step": int(global_step) if global_step is not None else "",
        "Phase1_Steps_Done": int(phase1_steps_done) if phase1_steps_done is not None else "",
        "Phase2_Steps_Done": int(phase2_steps_done) if phase2_steps_done is not None else "",
        "Cycle_Steps": int(cycle_steps) if cycle_steps is not None else "",
        "Step1_Max_Steps": int(step_schedule.get("step1_max_steps")) if step_schedule.get("step1_max_steps") is not None else "",
        "Step2_Max_Steps": int(step_schedule.get("step2_max_steps")) if step_schedule.get("step2_max_steps") is not None else "",
        "Val_Every_Steps": int(step_schedule.get("val_every_steps")) if step_schedule.get("val_every_steps") is not None else "",
    }

    for split_name, metrics in (("Train", train_metrics), ("Val", val_metrics)):
        row[f"{split_name}_MAE_Unique_Temporal_Error_Mean_GE2"] = _sf(_get(metrics, "mae_unique_temporal_error_mean_ge2"))
        row[f"{split_name}_Median_Unique_Temporal_Error_Mean_GE2"] = _sf(_get(metrics, "median_unique_temporal_error_mean_ge2"))
        row[f"{split_name}_N_Unique_Temporal_Error_Mean_GE2"] = _sf(_get(metrics, "n_unique_temporal_error_mean_ge2"))
        row[f"{split_name}_MAE_Unique_Nearest_GE2"] = _sf(_get(metrics, "mae_unique_nearest_ge2"))
        row[f"{split_name}_RMSE_Unique_Nearest_GE2"] = _sf(_get(metrics, "rmse_unique_nearest_ge2"))
        row[f"{split_name}_R2_Unique_Nearest_GE2"] = _sf(_get(metrics, "r2_unique_nearest_ge2"))
        row[f"{split_name}_N_Unique_Nearest_GE2"] = _sf(_get(metrics, "n_unique_nearest_ge2"))
        row[f"{split_name}_MAE_Unique_Temporal_Pred_Mean_GE2"] = _sf(_get(metrics, "mae_unique_temporal_pred_mean_ge2"))
        row[f"{split_name}_RMSE_Unique_Temporal_Pred_Mean_GE2"] = _sf(_get(metrics, "rmse_unique_temporal_pred_mean_ge2"))
        row[f"{split_name}_R2_Unique_Temporal_Pred_Mean_GE2"] = _sf(_get(metrics, "r2_unique_temporal_pred_mean_ge2"))
        row[f"{split_name}_Temporal_Pred_Std_Mean_GE2"] = _sf(_get(metrics, "temporal_pred_std_mean_ge2"))
        row[f"{split_name}_Temporal_Pred_Std_Median_GE2"] = _sf(_get(metrics, "temporal_pred_std_median_ge2"))
        row[f"{split_name}_GEDI_Unique_Metadata_Available"] = _sf(_get(metrics, "gedi_unique_metadata_available"))

    for th in _CLS_THRESH:
        k = _cls_key(th)
        num = str(int(th)) if float(th).is_integer() else str(th)
        row[f"Val_IoU{num}"] = _sf(_get(val_metrics, f"{k}_iou"))
        row[f"Val_Dice{num}"] = _sf(_get(val_metrics, f"{k}_dice"))

    for split_name, metrics in (("Train", train_metrics), ("Val", val_metrics)):
        # Occurrence-level domain metrics. Keys come from trainloop.py:
        # mae_occurrence_eval15, mae_occurrence_ge2, mae_occurrence_low_0_2, etc.
        row[f"{split_name}_MAE_Occurrence_Eval15"] = _sf(_get(metrics, "mae_occurrence_eval15"))
        row[f"{split_name}_RMSE_Occurrence_Eval15"] = _sf(_get(metrics, "rmse_occurrence_eval15"))
        row[f"{split_name}_R2_Occurrence_Eval15"] = _sf(_get(metrics, "r2_occurrence_eval15"))
        row[f"{split_name}_Bias_Occurrence_Eval15"] = _sf(_get(metrics, "bias_occurrence_eval15"))
        row[f"{split_name}_Slope_Occurrence_Eval15"] = _sf(_get(metrics, "slope_occurrence_eval15"))
        row[f"{split_name}_Std_Ratio_Occurrence_Eval15"] = _sf(_get(metrics, "std_ratio_occurrence_eval15"))
        row[f"{split_name}_N_Occurrence_Eval15"] = _sf(_get(metrics, "n_occurrence_eval15"))

        row[f"{split_name}_MAE_Occurrence_GE2"] = _sf(_get(metrics, "mae_occurrence_ge2"))
        row[f"{split_name}_RMSE_Occurrence_GE2"] = _sf(_get(metrics, "rmse_occurrence_ge2"))
        row[f"{split_name}_R2_Occurrence_GE2"] = _sf(_get(metrics, "r2_occurrence_ge2"))
        row[f"{split_name}_Bias_Occurrence_GE2"] = _sf(_get(metrics, "bias_occurrence_ge2"))
        row[f"{split_name}_Slope_Occurrence_GE2"] = _sf(_get(metrics, "slope_occurrence_ge2"))
        row[f"{split_name}_Std_Ratio_Occurrence_GE2"] = _sf(_get(metrics, "std_ratio_occurrence_ge2"))
        row[f"{split_name}_N_Occurrence_GE2"] = _sf(_get(metrics, "n_occurrence_ge2"))

        row[f"{split_name}_MAE_Occurrence_Low_0_2"] = _sf(_get(metrics, "mae_occurrence_low_0_2"))
        row[f"{split_name}_N_Occurrence_Low_0_2"] = _sf(_get(metrics, "n_occurrence_low_0_2"))

    for split_name, metrics in (("Train", train_metrics), ("Val", val_metrics)):
        for g in _GROUP_NAMES:
            for m in _REG_METRICS:
                row[f"{split_name}_g_{g}_{m}"] = _sf(_get(metrics, f"g_{g}_{m}"))

        row[f"{split_name}_g_acc{NUM_HEIGHT_CLASSES}"] = _sf(
            _get(metrics, f"g_acc{NUM_HEIGHT_CLASSES}")
        )

        for i in range(NUM_HEIGHT_CLASSES):
            for j in range(NUM_HEIGHT_CLASSES):
                row[f"{split_name}_g_cm_{i}{j}"] = _sf(_get(metrics, f"g_cm_{i}{j}"))

    for split_name, metrics in (("Train", train_metrics), ("Val", val_metrics)):
        for th in _CLS_THRESH:
            k = _cls_key(th)

            for met in _CLS_METRICS:
                row[f"{split_name}_{k}_{met}"] = _sf(_get(metrics, f"{k}_{met}"))

            for g in _GROUP_NAMES:
                for met in _CLS_METRICS:
                    row[f"{split_name}_{k}_g_{g}_{met}"] = _sf(
                        _get(metrics, f"{k}_g_{g}_{met}")
                    )

    for f in LOG_FIELDS:
        row.setdefault(f, "")

    return row



def make_epoch_row(
    *,
    epoch: int,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    time_sec: float,
    lr: float,
    best: float,
    model_type: str,
    is_resume: bool,
    training_mode: str = "",
    phase: Optional[int] = None,
    phase_eval_index: Optional[int] = None,
    global_step: Optional[int] = None,
    phase1_steps_done: Optional[int] = None,
    phase2_steps_done: Optional[int] = None,
    cycle_steps: Optional[int] = None,
    step_schedule: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Legacy alias kept for callers that still use epoch naming.
    Internally mapped to the canonical cycle-based row builder.
    """
    return make_cycle_row(
        cycle_index=int(epoch),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        time_sec=time_sec,
        lr=lr,
        best=best,
        model_type=model_type,
        is_resume=is_resume,
        training_mode=training_mode,
        phase=phase,
        phase_cycle_index=phase_eval_index,
        global_step=global_step,
        phase1_steps_done=phase1_steps_done,
        phase2_steps_done=phase2_steps_done,
        cycle_steps=cycle_steps,
        step_schedule=step_schedule,
    )



def make_step_eval_row(**kwargs: Any) -> Dict[str, Any]:
    """
    Explicit alias for step-based runs.
    """
    return make_cycle_row(**kwargs)


# ======================================================================================
# CHECKPOINTS
# ======================================================================================
def save_checkpoint(
    path: Path,
    *,
    epoch: Optional[int] = None,
    cycle_index: Optional[int] = None,
    best: float,
    model,
    opt,
    model_type: str,
    scaler=None,
    scheduler=None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save checkpoint with model/optimizer and optional scaler/scheduler.

    Canonical wording is cycle-based, but legacy ``epoch`` is kept as alias.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if cycle_index is None and epoch is None:
        cycle_index = 0
    if cycle_index is None:
        cycle_index = int(epoch)
    if epoch is None:
        epoch = int(cycle_index)

    ckpt = {
        "cycle_index": int(cycle_index),
        "epoch": int(epoch),
        "best": float(best),
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt is not None else None,
        "model_type": str(model_type),
    }

    if scaler is not None:
        try:
            ckpt["scaler"] = scaler.state_dict()
        except Exception:
            ckpt["scaler"] = None

    if scheduler is not None:
        try:
            ckpt["scheduler"] = scheduler.state_dict()
        except Exception:
            ckpt["scheduler"] = None

    if extra_state is not None:
        ckpt["extra_state"] = dict(extra_state)

    torch.save(ckpt, str(path))



def load_checkpoint(
    path: Path,
    *,
    model,
    opt=None,
    device="cpu",
    strict: bool = True,
    scaler=None,
    scheduler=None,
) -> Dict[str, Any]:
    """
    Charge un checkpoint et restaure model/opt/scaler/scheduler si fournis.
    Retourne dict avec:
      - cycle_index
      - epoch (legacy alias)
      - best
      - model_type
      - extra_state
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {path}")

    ckpt = torch.load(str(path), map_location=device)

    model.load_state_dict(ckpt["model"], strict=bool(strict))

    if opt is not None and ckpt.get("opt", None) is not None:
        try:
            opt.load_state_dict(ckpt["opt"])
        except ValueError as exc:
            print(
                "[RESUME WARNING] optimizer state incompatible with current model/optimizer; "
                f"continuing with a fresh optimizer. Details: {exc}",
                flush=True,
            )

    if scaler is not None and ckpt.get("scaler", None) is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass

    if scheduler is not None and ckpt.get("scheduler", None) is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass

    cycle_index = ckpt.get("cycle_index", ckpt.get("epoch", 0))
    epoch = ckpt.get("epoch", cycle_index)

    return {
        "cycle_index": int(cycle_index),
        "epoch": int(epoch),
        "best": float(ckpt.get("best", float("inf"))),
        "model_type": str(ckpt.get("model_type", "")),
        "extra_state": ckpt.get("extra_state", {}),
    }


# ======================================================================================
# JSON artifact helpers
# ======================================================================================
def _jsonable(obj):
    """
    Convert recursively to JSON-serializable Python objects.

    Handles:
      - Path / WindowsPath / PosixPath -> str
      - numpy scalars -> Python scalars
      - torch.device -> str
      - dict / list / tuple / set -> recursive conversion
    """
    from pathlib import Path as _Path

    try:
        import numpy as np
    except Exception:
        np = None

    if obj is None:
        return None

    if isinstance(obj, _Path):
        return str(obj)

    if isinstance(obj, torch.device):
        return str(obj)

    if np is not None and isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]

    return obj



def save_json_artifact(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, ensure_ascii=False)


# ======================================================================================
# Debug
# ======================================================================================
if __name__ == "__main__":
    print("[io] NUM_HEIGHT_CLASSES =", NUM_HEIGHT_CLASSES)
    print("[io] GROUP_NAMES        =", _GROUP_NAMES)
    print("[io] CLS_THRESH         =", _CLS_THRESH)
    print("[io] LOG_FIELDS         =", len(LOG_FIELDS))
    print("[io] First 25 fields:")
    for x in LOG_FIELDS[:25]:
        print(" -", x)
