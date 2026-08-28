from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch


# ======================================================================================
# PROJECT PATHS
# ======================================================================================
PROJECT_ROOT = Path(r"C:\Users\Dell\Desktop\Article_Maroc\Architecture")

# Expérience finale par défaut pour le training
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "exp_maamoura_gediids_uniqueval_shotaware_hytec_75_15_10_PATCH512_STRIDE512_S1ASC_DESC_C10_EVAL15_v3"

# Dossier de sortie des runs d'entraînement
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"


# ======================================================================================
# DATA / SUPERVISION DEFAULTS
# ======================================================================================
# K max de tirs GEDI par patch dans les shards
MAX_AUX_K = 512

# --------------------------------------------------------------------------------------
# BINS / CLASSES DE HAUTEUR — MAÂMORA GEDIIDS UNIQUEVAL EVAL15
#
# Domaine scientifique retenu :
#   - train peut garder la queue haute >15 m ;
#   - val/test sont évalués sur 0–15 m ;
#   - métrique officielle article/checkpoint : domaine ge2 = 2 <= RH95 <= 15.
#
# Seuils de REPORTING demandés :
#   --bins 0,5,10,15
#
# Convention du code : les bins sont des SEUILS internes. Avec un seuil 0,
# le code crée donc une classe audit "<0 m", normalement vide si les labels
# sont correctement filtrés à RH95 >= 0. Les classes utiles article restent :
#   0–5, 5–10, 10–15 ; >=15 est une queue haute/audit.
#
#   Classe 0 : h < 0       (audit invalid/normalement vide)
#   Classe 1 : 0 <= h < 5
#   Classe 2 : 5 <= h < 10
#   Classe 3 : 10 <= h < 15
#   Classe 4 : h >= 15     (audit queue haute ; normalement absent de val/test EVAL15)
# --------------------------------------------------------------------------------------
BINS_DEFAULT = (0.0, 5.0, 10.0, 15.0)
REPORT_BINS_DEFAULT = (0.0, 5.0, 10.0, 15.0)
CLS_THRESH_DEFAULT = (0.0, 5.0, 10.0, 15.0)

HEIGHT_CLASS_LABELS = ("<0m", "0-5m", "5-10m", "10-15m", ">=15m")
HEIGHT_CLASS_KEYS = ("lt0", "0_5", "5_10", "10_15", "ge15")
HEIGHT_CLASS_DISPLAY = ("<0m", "0-5m", "5-10m", "10-15m", ">=15m")

NUM_HEIGHT_CLASSES = len(HEIGHT_CLASS_LABELS)


AUC_RESERVOIR_DEFAULT = 200_000
USE_AMP_DEFAULT = True

# Windows / notebook : garder 0 est plus robuste
DEFAULT_NUM_WORKERS = 0 if (os.name == "nt" or "ipykernel" in sys.modules) else 2


# ======================================================================================
# TRAINING DEFAULTS
# ======================================================================================
EARLY_STOPPING_PATIENCE_DEFAULT = 30
TV_WEIGHT_DEFAULT = 0.005

# Architecture U-Net
DEFAULT_BASE_CH = 32
DEFAULT_DROPOUT = 0.25
DEFAULT_OUT_CHANNELS = 1

# Optimisation
DEFAULT_BATCH_SIZE = 2
DEFAULT_EPOCHS = 150
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 5e-3
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SEED = 42

# Logging / checkpoints
DEFAULT_SAVE_EVERY = 1
DEFAULT_MONITOR_METRIC = "val_mae_unique_temporal_error_mean_ge2"
DEFAULT_MONITOR_MODE = "min"

# Step-based training defaults (kept optional / non-breaking).
DEFAULT_TRAINING_MODE = "step"
DEFAULT_OPTIMIZER = "adamw"
DEFAULT_VAL_EVERY_STEPS = 640
DEFAULT_STEP1_MAX_STEPS: Optional[int] = None
DEFAULT_STEP2_MAX_STEPS: Optional[int] = None
DEFAULT_STEP1_PATIENCE_EVALS: Optional[int] = None
DEFAULT_STEP2_PATIENCE_EVALS: Optional[int] = None


# ======================================================================================
# SAFE HELPERS
# ======================================================================================
def sf(v: Any, default: float = float("nan")) -> float:
    """Safe float conversion (finite only)."""
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def si(v: Any, default: int = 0) -> int:
    """Safe int conversion."""
    try:
        return int(v)
    except Exception:
        return int(default)


def load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_bins(v: str | Sequence[float], default: Tuple[float, ...]) -> Tuple[float, ...]:
    """
    Parse:
      "3,10,20,30"      -> (3.0, 10.0, 20.0, 30.0)
      [3, 10, 20, 30]   -> (3.0, 10.0, 20.0, 30.0)
    """
    if isinstance(v, (tuple, list)):
        try:
            xs = tuple(float(x) for x in v)
            return xs if len(xs) > 0 else tuple(default)
        except Exception:
            return tuple(default)

    try:
        xs = tuple(float(x.strip()) for x in str(v).split(",") if x.strip())
        return xs if len(xs) > 0 else tuple(default)
    except Exception:
        return tuple(default)


def bins_to_class_edges(bins: Sequence[float]) -> Tuple[float, ...]:
    """
    Convertit (3,10,20,30) en bornes implicites de classes :
      (-inf, 3, 10, 20, 30, +inf)
    Retourne seulement les seuils internes triés/validés.
    """
    xs = sorted(float(x) for x in bins)
    if len(xs) == 0:
        raise ValueError("bins vide")
    for i in range(1, len(xs)):
        if not (xs[i] > xs[i - 1]):
            raise ValueError(f"bins non strictement croissants: {xs}")
    return tuple(xs)


def class_index_from_height(h: float, bins: Sequence[float] = BINS_DEFAULT) -> int:
    """
    Classe une hauteur h selon les seuils fournis.
    Convention : frontière -> classe supérieure
    Exemple avec bins=(3,10,20,30):
      h < 3      -> 0
      3 <= h <10 -> 1
      10<=h <20  -> 2
      20<=h <30  -> 3
      h >= 30    -> 4
    """
    hh = float(h)
    xs = bins_to_class_edges(bins)
    for i, thr in enumerate(xs):
        if hh < thr:
            return i
    return len(xs)


def get_height_class_names(
    bins: Sequence[float] = BINS_DEFAULT,
    style: str = "display",
) -> Tuple[str, ...]:
    """
    Génère les noms de classes depuis les bins.
    style:
      - display : <3m, 3-10m, 10-20m, 20-30m, >=30m
      - key     : lt3, 3_10, 10_20, 20_30, ge30
    """
    xs = bins_to_class_edges(bins)

    if style not in {"display", "key"}:
        raise ValueError(f"style invalide: {style}")

    names = []
    for i in range(len(xs) + 1):
        if i == 0:
            if style == "display":
                names.append(f"<{int(xs[0])}m" if float(xs[0]).is_integer() else f"<{xs[0]}m")
            else:
                names.append(f"lt{int(xs[0])}" if float(xs[0]).is_integer() else f"lt{xs[0]}")
        elif i == len(xs):
            if style == "display":
                names.append(f">={int(xs[-1])}m" if float(xs[-1]).is_integer() else f">={xs[-1]}m")
            else:
                names.append(f"ge{int(xs[-1])}" if float(xs[-1]).is_integer() else f"ge{xs[-1]}")
        else:
            lo, hi = xs[i - 1], xs[i]
            if style == "display":
                lo_s = str(int(lo)) if float(lo).is_integer() else str(lo)
                hi_s = str(int(hi)) if float(hi).is_integer() else str(hi)
                names.append(f"{lo_s}-{hi_s}m")
            else:
                lo_s = str(int(lo)) if float(lo).is_integer() else str(lo)
                hi_s = str(int(hi)) if float(hi).is_integer() else str(hi)
                names.append(f"{lo_s}_{hi_s}")

    return tuple(names)


# ======================================================================================
# SEED / REPRODUCIBILITY
# ======================================================================================
def set_seed(seed: int) -> None:
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Réglage safe / pragmatique
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


# ======================================================================================
# EXPERIMENT HELPERS
# ======================================================================================
def get_experiment_json(experiment_root: Path) -> Path:
    experiment_root = Path(experiment_root)
    p = experiment_root / "experiment.json"
    if not p.exists():
        raise FileNotFoundError(f"experiment.json introuvable: {p}")
    return p


def load_experiment_config(experiment_root: Path) -> Dict[str, Any]:
    return load_json(get_experiment_json(experiment_root))


def get_split_dir(experiment_root: Path, split_name: str) -> Path:
    split_name = str(split_name).strip().lower()
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"split_name invalide: {split_name}")

    exp = load_experiment_config(experiment_root)
    files = exp.get("files", {}) or {}

    key = f"{split_name}_dir"
    if key in files and files[key]:
        return Path(files[key])

    return Path(experiment_root) / split_name


def get_manifest_csv(experiment_root: Path) -> Path:
    exp = load_experiment_config(experiment_root)
    files = exp.get("files", {}) or {}
    p = files.get("manifest_csv")
    if p:
        return Path(p)
    return Path(experiment_root) / "shards_manifest.csv"


def get_schema(experiment_root: Path) -> Dict[str, Any]:
    exp = load_experiment_config(experiment_root)
    schema = exp.get("schema", {}) or {}
    if not isinstance(schema, dict):
        raise RuntimeError(f"Champ 'schema' invalide dans {get_experiment_json(experiment_root)}")
    return schema


def infer_in_channels(experiment_root: Path, fallback: int = 8) -> int:
    try:
        schema = get_schema(experiment_root)
        n = int(schema.get("n_channels", 0))
        if n > 0:
            return n
    except Exception:
        pass
    return int(fallback)


def infer_patch_size(experiment_root: Path, fallback: int = 512) -> int:
    try:
        schema = get_schema(experiment_root)
        n = int(schema.get("patch_size", 0))
        if n > 0:
            return n
    except Exception:
        pass
    return int(fallback)


def infer_stride(experiment_root: Path, fallback: int = 256) -> int:
    try:
        schema = get_schema(experiment_root)
        n = int(schema.get("stride", 0))
        if n > 0:
            return n
    except Exception:
        pass
    return int(fallback)


def infer_temporal_window_days(experiment_root: Path, fallback: int = 180) -> int:
    try:
        schema = get_schema(experiment_root)
        n = int(schema.get("temporal_window_days", 0))
        if n > 0:
            return n
    except Exception:
        pass
    return int(fallback)


def infer_channel_order(experiment_root: Path) -> Tuple[str, ...]:
    try:
        schema = get_schema(experiment_root)
        order = schema.get("channel_order", [])
        if isinstance(order, list) and len(order) > 0:
            return tuple(str(x) for x in order)
    except Exception:
        pass
    return tuple()


def infer_target_fracs(experiment_root: Path) -> Dict[str, float]:
    try:
        schema = get_schema(experiment_root)
        tf = schema.get("target_fracs", {}) or {}
        out = {
            "train": float(tf.get("train", 0.75)),
            "val": float(tf.get("val", 0.15)),
            "test": float(tf.get("test", 0.10)),
        }
        return out
    except Exception:
        return {"train": 0.75, "val": 0.15, "test": 0.10}


def make_run_dir(
    experiment_root: Path,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    run_name: Optional[str] = None,
) -> Path:
    experiment_root = Path(experiment_root)
    runs_root = Path(runs_root)

    exp_name = experiment_root.name
    run_name = run_name or "run_001"

    out = runs_root / exp_name / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_run_paths(
    experiment_root: Path,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    run_name: Optional[str] = None,
) -> Dict[str, Path]:
    run_dir = make_run_dir(experiment_root, runs_root=runs_root, run_name=run_name)

    return {
        "run_dir": run_dir,
        "checkpoints_dir": run_dir / "checkpoints",
        "logs_dir": run_dir / "logs",
        "artifacts_dir": run_dir / "artifacts",
        "train_csv": run_dir / "logs" / "train_log.csv",
        "best_ckpt": run_dir / "checkpoints" / "best.ckpt",
        "last_ckpt": run_dir / "checkpoints" / "last.ckpt",
        "test_metrics_json": run_dir / "artifacts" / "test_metrics.json",
        "run_config_json": run_dir / "artifacts" / "run_config.json",
    }


# ======================================================================================
# DEVICE HELPERS
# ======================================================================================
def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_amp_available(device: torch.device) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()



def summarize_training_defaults() -> Dict[str, Any]:
    """
    Compact summary of default training knobs.
    Useful for launchers and quick sanity checks.
    """
    return {
        "batch_size": int(DEFAULT_BATCH_SIZE),
        "epochs": int(DEFAULT_EPOCHS),
        "lr": float(DEFAULT_LR),
        "weight_decay": float(DEFAULT_WEIGHT_DECAY),
        "grad_clip": float(DEFAULT_GRAD_CLIP),
        "seed": int(DEFAULT_SEED),
        "monitor_metric": str(DEFAULT_MONITOR_METRIC),
        "monitor_mode": str(DEFAULT_MONITOR_MODE),
        "training_mode": str(DEFAULT_TRAINING_MODE),
        "optimizer": str(DEFAULT_OPTIMIZER),
        "val_every_steps": int(DEFAULT_VAL_EVERY_STEPS),
        "step1_max_steps": DEFAULT_STEP1_MAX_STEPS,
        "step2_max_steps": DEFAULT_STEP2_MAX_STEPS,
        "step1_patience_evals": DEFAULT_STEP1_PATIENCE_EVALS,
        "step2_patience_evals": DEFAULT_STEP2_PATIENCE_EVALS,
    }


# ======================================================================================
# SHARD / SPLIT SANITY HELPERS
# ======================================================================================
def assert_experiment_dirs_exist(experiment_root: Path) -> None:
    experiment_root = Path(experiment_root)

    train_dir = get_split_dir(experiment_root, "train")
    val_dir = get_split_dir(experiment_root, "val")
    test_dir = get_split_dir(experiment_root, "test")

    missing = [str(p) for p in [train_dir, val_dir, test_dir] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Certains dossiers de split sont introuvables:\n" + "\n".join(f" - {m}" for m in missing)
        )


def summarize_experiment(experiment_root: Path) -> Dict[str, Any]:
    experiment_root = Path(experiment_root)
    exp = load_experiment_config(experiment_root)

    return {
        "experiment_root": str(experiment_root),
        "experiment_json": str(get_experiment_json(experiment_root)),
        "train_dir": str(get_split_dir(experiment_root, "train")),
        "val_dir": str(get_split_dir(experiment_root, "val")),
        "test_dir": str(get_split_dir(experiment_root, "test")),
        "manifest_csv": str(get_manifest_csv(experiment_root)),
        "n_channels": infer_in_channels(experiment_root),
        "patch_size": infer_patch_size(experiment_root),
        "stride": infer_stride(experiment_root),
        "temporal_window_days": infer_temporal_window_days(experiment_root),
        "channel_order": infer_channel_order(experiment_root),
        "target_fracs": infer_target_fracs(experiment_root),
        "policy": exp.get("policy", None),
        "output_shards": exp.get("output_shards", {}),
        "actual_output_counts": exp.get("actual_output_counts", {}),
        "actual_output_metrics": exp.get("actual_output_metrics", {}),
    }


# ======================================================================================
# DEBUG / CLI
# ======================================================================================
if __name__ == "__main__":
    print("[common] PROJECT_ROOT            =", PROJECT_ROOT)
    print("[common] DEFAULT_EXPERIMENT_ROOT =", DEFAULT_EXPERIMENT_ROOT)
    print("[common] DEFAULT_RUNS_ROOT       =", DEFAULT_RUNS_ROOT)
    print("[common] MAX_AUX_K               =", MAX_AUX_K)
    print("[common] BINS_DEFAULT            =", BINS_DEFAULT)
    print("[common] HEIGHT_CLASS_LABELS     =", HEIGHT_CLASS_LABELS)
    print("[common] HEIGHT_CLASS_KEYS       =", HEIGHT_CLASS_KEYS)
    print("[common] NUM_HEIGHT_CLASSES      =", NUM_HEIGHT_CLASSES)

    if DEFAULT_EXPERIMENT_ROOT.exists():
        try:
            assert_experiment_dirs_exist(DEFAULT_EXPERIMENT_ROOT)
            info = summarize_experiment(DEFAULT_EXPERIMENT_ROOT)

            print("[common] experiment_json        =", info["experiment_json"])
            print("[common] train_dir             =", info["train_dir"])
            print("[common] val_dir               =", info["val_dir"])
            print("[common] test_dir              =", info["test_dir"])
            print("[common] manifest_csv          =", info["manifest_csv"])
            print("[common] n_channels            =", info["n_channels"])
            print("[common] patch_size            =", info["patch_size"])
            print("[common] stride                =", info["stride"])
            print("[common] temporal_window_days  =", info["temporal_window_days"])
            print("[common] channel_order         =", info["channel_order"])
            print("[common] target_fracs          =", info["target_fracs"])
            print("[common] policy                =", info["policy"])
            print("[common] output_shards         =", info["output_shards"])
            print("[common] actual_output_counts  =", info["actual_output_counts"])
            print("[common] actual_output_metrics =", info["actual_output_metrics"])
        except Exception as e:
            print("[common] experiment read failed =", repr(e))
    else:
        print("[common] DEFAULT_EXPERIMENT_ROOT does not exist.")