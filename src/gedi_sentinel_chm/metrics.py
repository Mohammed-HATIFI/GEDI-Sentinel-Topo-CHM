"""Evaluation metrics used by lightweight release checks."""
from __future__ import annotations
import numpy as np


def paired_finite(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if not mask.any():
        raise ValueError("No finite observed-predicted pairs")
    return observed[mask], predicted[mask]


def regression_metrics(observed, predicted):
    """Return the manuscript's core point-prediction diagnostics."""
    y, yhat = paired_finite(observed, predicted)
    error = yhat - y
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "correlation": float(np.corrcoef(y, yhat)[0, 1]) if len(y) > 1 else float("nan"),
        "bias": float(np.mean(error)),
        "slope": float(np.polyfit(y, yhat, 1)[0]) if len(y) > 1 else float("nan"),
        "std_ratio": float(np.std(yhat, ddof=1) / np.std(y, ddof=1)) if len(y) > 1 and np.std(y, ddof=1) > 0 else float("nan"),
    }
