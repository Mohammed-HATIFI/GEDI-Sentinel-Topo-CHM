from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(r"C:\Users\Dell\Desktop\Article_Maroc\Architecture")
HERE = Path(__file__).resolve().parent


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if y.size == 0:
        return {"n": 0}
    err = p - y
    y_mean = float(y.mean())
    p_mean = float(p.mean())
    y_std = float(y.std())
    p_std = float(p.std())
    cov = float(np.mean((y - y_mean) * (p - p_mean)))
    slope = cov / max(float(np.mean((y - y_mean) ** 2)), 1e-12)
    corr = cov / max(y_std * p_std, 1e-12)
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y - y_mean) ** 2))
    return {
        "n": int(y.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - sse / max(sst, 1e-12)),
        "bias": float(err.mean()),
        "slope": float(slope),
        "corr": float(corr),
        "std_ratio": float(p_std / max(y_std, 1e-12)),
        "pred_mean": p_mean,
        "pred_std": p_std,
        "pred_min": float(p.min()),
        "pred_p01": float(np.quantile(p, 0.01)),
        "pred_p05": float(np.quantile(p, 0.05)),
        "pred_p50": float(np.quantile(p, 0.50)),
        "pred_p95": float(np.quantile(p, 0.95)),
        "pred_max": float(p.max()),
        "pred_zero_frac": float(np.mean(p <= 1e-6)),
        "pred_lt1_frac": float(np.mean(p < 1.0)),
        "pred_lt2p5_frac": float(np.mean(p < 2.5)),
        "true_mean": y_mean,
        "true_std": y_std,
        "true_min": float(y.min()),
        "true_max": float(y.max()),
    }


def _finite_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = pd.to_numeric(df["rh95"], errors="coerce").to_numpy(dtype=np.float64)
    p = pd.to_numeric(df["prediction_original_coords"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    return y, p, p - y


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] saved: {path}", flush=True)


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        f = float(v)
        if math.isfinite(f):
            return f"{f:.{digits}f}"
    except Exception:
        pass
    return "nan"


def plot_metric_cards(
    *,
    occurrence: dict[str, Any],
    unique_nearest: dict[str, Any],
    title: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 3.35))
    ax.axis("off")
    cards = [
        ("MAE", occurrence.get("mae"), "min"),
        ("RMSE", occurrence.get("rmse"), "min"),
        ("R2", occurrence.get("r2"), "max"),
        ("bias", occurrence.get("bias"), "0"),
        ("slope", occurrence.get("slope"), "1"),
        ("std ratio", occurrence.get("std_ratio"), "1"),
        ("pred<=0 %", 100.0 * float(occurrence.get("pred_zero_frac", float("nan"))), "0"),
        ("pred<1 %", 100.0 * float(occurrence.get("pred_lt1_frac", float("nan"))), "low"),
        ("pred max", occurrence.get("pred_max"), "domain"),
        ("n occ.", occurrence.get("n"), ""),
    ]
    ax.text(0.01, 0.96, title, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    subtitle = (
        "Occurrence metrics shown in cards. "
        f"Unique-nearest: MAE={_fmt(unique_nearest.get('mae'), 4)} | "
        f"slope={_fmt(unique_nearest.get('slope'), 4)} | "
        f"stdr={_fmt(unique_nearest.get('std_ratio'), 4)} | "
        f"bias={_fmt(unique_nearest.get('bias'), 4)} | n={unique_nearest.get('n', 0)}"
    )
    ax.text(0.01, 0.84, subtitle, transform=ax.transAxes, fontsize=10.2, color="0.25", va="top")
    x0, y0, w, h, gap = 0.01, 0.18, 0.092, 0.42, 0.006
    for i, (name, value, target) in enumerate(cards):
        x = x0 + i * (w + gap)
        rect = plt.Rectangle((x, y0), w, h, transform=ax.transAxes, facecolor="#f5f7fb", edgecolor="#c7ceda", lw=1.0)
        ax.add_patch(rect)
        ax.text(x + 0.012, y0 + h - 0.08, name, transform=ax.transAxes, fontsize=9.5, color="0.30", va="top")
        if isinstance(value, (int, np.integer)) or name.startswith("n "):
            val_text = f"{int(value or 0):,}"
        elif "%" in name:
            val_text = f"{float(value):.1f}%" if math.isfinite(float(value)) else "nan"
        else:
            val_text = _fmt(value, 3 if name not in {"MAE", "RMSE", "bias"} else 4)
        ax.text(x + 0.012, y0 + 0.18, val_text, transform=ax.transAxes, fontsize=15, fontweight="bold", va="center")
        if target:
            ax.text(x + 0.012, y0 + 0.055, f"target {target}", transform=ax.transAxes, fontsize=8.5, color="0.42")
    _save_fig(fig, out_dir / "00_metric_cards.png")


def plot_scatter_density(
    df: pd.DataFrame,
    *,
    metric: dict[str, Any],
    title: str,
    max_height: float,
    out_dir: Path,
) -> None:
    y, p, _ = _finite_xy(df)
    if y.size == 0:
        return
    plot_mask = (y >= 0.0) & (y <= max_height) & (p >= 0.0) & (p <= max_height)
    y = y[plot_mask]
    p = p[plot_mask]
    if y.size == 0:
        return

    # Exact visual convention of the supplied publication-style Step07:
    # every point is coloured by its local 2-D-bin count, low density first.
    bins2d = 95 if float(max_height) <= 25 else 150
    density, xedges, yedges = np.histogram2d(
        y,
        p,
        bins=int(bins2d),
        range=[[0.0, float(max_height)], [0.0, float(max_height)]],
    )
    xi = np.clip(np.digitize(y, xedges) - 1, 0, density.shape[0] - 1)
    yi = np.clip(np.digitize(p, yedges) - 1, 0, density.shape[1] - 1)
    point_count = np.clip(density[xi, yi].astype(float), 1.0, None)
    order = np.argsort(point_count)
    y, p, point_count = y[order], p[order], point_count[order]

    fig, ax = plt.subplots(figsize=(7.3, 7.3), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    sc = ax.scatter(
        y,
        p,
        c=point_count,
        cmap="viridis",
        s=7.0 if float(max_height) <= 25 else 6.0,
        alpha=0.85,
        marker="o",
        linewidths=0.0,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("Count", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # Keep only the 1:1 line, exactly as in the reference style.  Slope remains
    # visible in the metric box, avoiding the misleading impression of shifted axes.
    ax.plot([0.0, max_height], [0.0, max_height], "--", color="black", lw=1.2, zorder=3, label="1:1")
    ticks = np.arange(0.0, float(max_height) + 0.1, 5.0)
    ax.set_xlim(0.0, max_height)
    ax.set_ylim(0.0, max_height)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed GEDI RH95 (m)", fontsize=13)
    ax.set_ylabel("Predicted GEDI RH95 (m)", fontsize=13)
    ax.set_title(title + " - observed vs predicted", fontsize=13)
    txt = (
        f"Domain: 0-{max_height:g} m\n"
        f"MAE = {_fmt(metric.get('mae'), 2)} m\n"
        f"RMSE = {_fmt(metric.get('rmse'), 2)} m\n"
        f"R2 = {_fmt(metric.get('r2'), 2)}\n"
        f"Bias = {_fmt(metric.get('bias'), 2)} m\n"
        f"Slope = {_fmt(metric.get('slope'), 2)}\n"
        f"Corr = {_fmt(metric.get('corr'), 2)}\n"
        f"Std ratio = {_fmt(metric.get('std_ratio'), 2)}"
    )
    ax.text(0.035, 0.965, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=10.0,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="0.55", linewidth=0.9, alpha=0.96),
            zorder=5)
    ax.grid(alpha=0.18)
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.1, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    _save_fig(fig, out_dir / "01_scatter_density_original_coords.png")


def plot_height_distribution_and_error(
    df: pd.DataFrame,
    *,
    title: str,
    max_height: float,
    out_dir: Path,
) -> None:
    y, p, err = _finite_xy(df)
    if y.size == 0:
        return
    edges = np.arange(0.0, max_height + 5.0, 5.0)
    edges[-1] = max_height
    hist_edges = np.arange(0.0, max_height + 1.0, 1.0)
    centers, box_data = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y >= lo) & ((y <= hi) if hi == edges[-1] else (y < hi))
        vals = err[mask]
        if vals.size:
            centers.append((lo + hi) / 2.0)
            box_data.append(vals)
    fig, ax_count = plt.subplots(figsize=(11.8, 3.7), dpi=160)
    ax_count.set_facecolor("#f2f2f2")
    ax_count.hist(y, bins=hist_edges, color="lightsteelblue", alpha=0.95, histtype="stepfilled", label="GEDI height")
    p_in_window = p[(p >= 0.0) & (p <= max_height)]
    ax_count.hist(p_in_window, bins=hist_edges, histtype="step", color="#ff7f0e", lw=1.6, label="Predicted values")
    ax_err = ax_count.twinx()
    if box_data:
        ax_err.boxplot(
            box_data,
            positions=centers,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            medianprops=dict(color="red", lw=1.6),
            boxprops=dict(facecolor="black", edgecolor="black", lw=1.0),
            whiskerprops=dict(color="black", lw=1.0),
            capprops=dict(color="black", lw=1.0),
        )
    ax_err.axhline(0.0, color="0.35", ls=":", lw=1.5)
    ax_count.set_xlim(0, max_height)
    ax_count.set_xticks(edges)
    ax_count.set_xticklabels([f"{int(v)}" if float(v).is_integer() else f"{v:g}" for v in edges], fontsize=10)
    ax_count.set_xlabel(r"GEDI $RH_{95}$ height bin edges (m)")
    ax_count.set_ylabel("Count")
    ax_err.set_ylabel("Error (m)")
    finite_err = err[np.isfinite(err)]
    if finite_err.size:
        q01, q99 = np.quantile(finite_err, [0.01, 0.99])
        lim = min(max(8.0, abs(q01), abs(q99)) * 1.15, 20.0)
        ax_err.set_ylim(-lim, lim)
    ax_count.legend(loc="upper right", frameon=True)
    ax_count.set_title(title + " - height-dependent error distribution")
    fig.tight_layout()
    _save_fig(fig, out_dir / "02_height_distribution_error_boxplots.png")


def plot_signed_error_distribution(
    df: pd.DataFrame,
    *,
    title: str,
    out_dir: Path,
) -> None:
    y, p, err = _finite_xy(df)
    if err.size == 0:
        return
    met = metrics(y, p)
    fig, ax = plt.subplots(figsize=(8.8, 5.3), dpi=150)
    ax.hist(err, bins=55, alpha=0.78)
    ax.axvline(0, ls="--", lw=1.1, color="black", label="Zero error")
    ax.axvline(float(np.mean(err)), ls="--", lw=1.1, label=f"Mean = {float(np.mean(err)):.2f}")
    ax.axvline(float(np.median(err)), ls="--", lw=1.1, label=f"Median = {float(np.median(err)):.2f}")
    ax.text(
        0.04,
        0.96,
        f"sigma = {float(np.std(err)):.2f} m\n"
        f"q05 = {float(np.quantile(err, 0.05)):.2f} m\n"
        f"q95 = {float(np.quantile(err, 0.95)):.2f} m\n"
        f"R2 = {_fmt(met.get('r2'), 3)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.6", alpha=0.90),
    )
    ax.set_xlabel("Signed error (m)")
    ax.set_ylabel("Number of GEDI shots")
    ax.set_title(title + " - signed errors")
    ax.grid(alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    _save_fig(fig, out_dir / "03_signed_error_distribution.png")


def plot_residuals_vs_height(
    df: pd.DataFrame,
    *,
    title: str,
    max_height: float,
    out_dir: Path,
) -> None:
    y, _, err = _finite_xy(df)
    if y.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8.9, 5.4), dpi=150)
    ax.scatter(y, err, s=7, alpha=0.36, linewidths=0)
    ax.axhline(0, color="black", ls="--", lw=1.1, label="Zero error")
    bins = np.linspace(0.0, max_height, 24)
    centers, med = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y >= lo) & ((y <= hi) if hi == bins[-1] else (y < hi))
        vals = err[mask]
        if vals.size >= 3:
            centers.append((lo + hi) / 2.0)
            med.append(float(np.median(vals)))
    if centers:
        ax.plot(centers, med, color="red", lw=1.5, label="Median residual")
    met = metrics(y, y + err)
    ax.set_xlim(0, max_height)
    ax.set_xlabel("GEDI RH95 / y_true (m)")
    ax.set_ylabel("Residual (pred - true) (m)")
    ax.set_title(title + " - residuals vs y_true")
    ax.text(
        0.02,
        0.96,
        f"Bias = {_fmt(met.get('bias'), 2)} m\n"
        f"MAE = {_fmt(met.get('mae'), 2)} m\n"
        f"RMSE = {_fmt(met.get('rmse'), 2)} m\n"
        f"R2 = {_fmt(met.get('r2'), 2)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.6", alpha=0.90),
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    _save_fig(fig, out_dir / "04_residuals_vs_true.png")


def plot_error_histogram_and_cdf(
    df: pd.DataFrame,
    *,
    title: str,
    out_dir: Path,
) -> None:
    _, _, err = _finite_xy(df)
    if err.size == 0:
        return
    abs_err = np.sort(np.abs(err))
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=150)
    cdf = np.arange(1, abs_err.size + 1, dtype=float) / max(abs_err.size, 1)
    ax.plot(abs_err, cdf, lw=2.0)
    for thr in [1, 2, 3, 5, 10]:
        pct = float(np.mean(abs_err <= thr) * 100.0)
        ax.axvline(thr, color="0.45", ls="--", lw=0.9, alpha=0.65)
        ax.text(thr, 0.04, f"{pct:.0f}% <= {thr}m", rotation=90, ha="right", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Absolute error threshold (m)")
    ax.set_ylabel("Cumulative fraction of GEDI shots")
    ax.set_title(title + " - cumulative absolute error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_fig(fig, out_dir / "06_absolute_error_cdf.png")


def plot_training_bin_error_summary(
    df: pd.DataFrame,
    *,
    title: str,
    max_height: float,
    out_dir: Path,
) -> None:
    y, _, err = _finite_xy(df)
    if y.size == 0:
        return
    bins = np.arange(0.0, float(max_height) + 5.0, 5.0, dtype=float)
    bins[-1] = float(max_height)
    labels = [f"{int(bins[i])}-{int(bins[i + 1])} m" for i in range(len(bins) - 1)]
    x = np.arange(len(labels), dtype=float)
    counts, box_data = [], []
    for i in range(len(bins) - 1):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (y >= lo) & (y < hi if i < len(bins) - 2 else y <= hi)
        counts.append(int(mask.sum()))
        box_data.append(err[mask])

    fig, ax_count = plt.subplots(figsize=(10.6, 4.8), dpi=150)
    ax_count.set_facecolor("#d9d9d9")
    ax_count.bar(x, counts, width=0.92, alpha=0.72, edgecolor="#4477AA", linewidth=0.9)
    ax_err = ax_count.twinx()
    valid_box = [values for values in box_data if len(values) > 0]
    valid_pos = [xi for xi, values in zip(x, box_data) if len(values) > 0]
    if valid_box:
        ax_err.boxplot(
            valid_box,
            positions=valid_pos,
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            medianprops=dict(color="red", linewidth=1.6),
            boxprops=dict(facecolor="white", edgecolor="0.35", linewidth=1.0),
            whiskerprops=dict(color="0.35", linewidth=1.0),
            capprops=dict(color="0.35", linewidth=1.0),
        )
    ax_err.axhline(0, ls="--", lw=0.9, color="#4A90E2")
    ax_count.set_xticks(x)
    ax_count.set_xticklabels(labels)
    ax_count.set_ylabel("Count")
    ax_err.set_ylabel("Error (pred - GEDI) [m]")
    ax_count.set_xlabel(r"GEDI $RH_{95}$ height class")
    ax_count.set_title(title + " - height-dependent errors (5 m bins)")
    ax_count.grid(True, axis="y", alpha=0.18)
    fig.tight_layout()
    _save_fig(fig, out_dir / "05_training_bin_error_summary.png")


def plot_metrics_by_height(
    bin_rows: list[dict[str, Any]],
    *,
    title: str,
    out_dir: Path,
) -> None:
    df = pd.DataFrame(bin_rows)
    if df.empty or "n" not in df:
        return
    df = df[pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(float) > 0].copy()
    if df.empty:
        return
    labels = [f"{r.height_min:g}-{r.height_max:g}" for r in df.itertuples(index=False)]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2))
    specs = [
        ("mae", "MAE (m)", None),
        ("bias", "Bias (m)", 0.0),
        ("slope", "Slope", 1.0),
        ("std_ratio", "Std ratio", 1.0),
    ]
    for ax, (col, ylabel, target) in zip(axes.ravel(), specs):
        vals = pd.to_numeric(df.get(col), errors="coerce").to_numpy(dtype=float)
        ax.bar(x, vals, color="#4c78a8", alpha=0.86)
        if target is not None:
            ax.axhline(target, color="green", ls="--", lw=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
        for xi, v, n in zip(x, vals, df["n"].to_numpy()):
            if math.isfinite(float(v)):
                ax.text(xi, v, f"n={int(n)}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.8, rotation=90)
    fig.suptitle(title + " - metrics by GEDI height bin", y=1.02, fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out_dir / "09_metrics_by_height_bin.png")


def plot_training_history(
    *,
    run_dir: Path,
    title: str,
    out_dir: Path,
) -> None:
    csv_path = run_dir / "logs" / "train_log.csv"
    if not csv_path.exists():
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[PLOT WARN] cannot read training log {csv_path}: {exc}", flush=True)
        return
    step_col = "Global_Step" if "Global_Step" in df.columns else ("step" if "step" in df.columns else None)
    if step_col is None:
        return
    metric_candidates = [
        ("Val_MAE_Occurrence_GE2", "MAE", "min", None),
        ("Val_RMSE_Occurrence_GE2", "RMSE", "min", None),
        ("Val_R2_Occurrence_GE2", "R2", "max", None),
        ("Val_Slope_Occurrence_GE2", "slope", "target=1", 1.0),
        ("Val_Std_Ratio_Occurrence_GE2", "std ratio", "target=1", 1.0),
        ("Val_Bias_Occurrence_GE2", "bias", "target=0", 0.0),
    ]
    available = [(col, name, arrow, target) for col, name, arrow, target in metric_candidates if col in df.columns]
    if not available:
        return
    ncols = 3
    nrows = int(math.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 3.3 * nrows), squeeze=False)
    x = pd.to_numeric(df[step_col], errors="coerce")
    for ax, (col, name, arrow, target) in zip(axes.ravel(), available):
        y = pd.to_numeric(df[col], errors="coerce")
        ax.plot(x, y, marker="o", ms=4, lw=1.5)
        if target is not None:
            ax.axhline(target, color="green", ls="--", lw=1.0)
        ax.set_title(f"{name} {arrow}")
        ax.set_xlabel("global step")
        ax.grid(alpha=0.22)
    for ax in axes.ravel()[len(available):]:
        ax.axis("off")
    fig.suptitle(title + " - validation history", y=1.02, fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out_dir / "10_training_history.png")




def prediction_pathology_diagnostics(df: pd.DataFrame, *, max_height: float) -> dict[str, Any]:
    """Diagnose zero spike, raw negative predictions, and low-height overprediction."""
    y = pd.to_numeric(df["rh95"], errors="coerce").to_numpy(dtype=np.float64)
    p = pd.to_numeric(df["prediction_original_coords"], errors="coerce").to_numpy(dtype=np.float64)
    raw = None
    if "prediction_raw_original_coords" in df.columns:
        raw = pd.to_numeric(df["prediction_raw_original_coords"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if raw is not None:
        raw = raw[valid]
    if y.size == 0:
        return {"global": {"n": 0}, "by_true_height": []}

    err = p - y
    out: dict[str, Any] = {
        "global": {
            "n": int(y.size),
            "pred_zero_frac": float(np.mean(p <= 1e-6)),
            "pred_lt1_frac": float(np.mean(p < 1.0)),
            "pred_lt2p5_frac": float(np.mean(p < 2.5)),
            "pred_gt40_frac": float(np.mean(p >= min(40.0, max_height) - 1e-6)),
            "over_2m_frac": float(np.mean(err > 2.0)),
            "under_2m_frac": float(np.mean(err < -2.0)),
            "bias": float(np.mean(err)),
            "mae": float(np.mean(np.abs(err))),
        },
        "by_true_height": [],
    }
    if raw is not None and np.isfinite(raw).any():
        raw_f = raw[np.isfinite(raw)]
        out["global"].update({
            "raw_pred_min": float(np.min(raw_f)),
            "raw_pred_p01": float(np.quantile(raw_f, 0.01)),
            "raw_pred_p05": float(np.quantile(raw_f, 0.05)),
            "raw_negative_frac": float(np.mean(raw_f < 0.0)),
            "clipped_to_zero_frac": float(np.mean(raw < 0.0)),
        })

    low = y < 5.0
    if low.any():
        out["global"].update({
            "true_lt5_n": int(low.sum()),
            "true_lt5_pred_gt5_frac": float(np.mean(p[low] > 5.0)),
            "true_lt5_pred_gt10_frac": float(np.mean(p[low] > 10.0)),
            "true_lt5_bias": float(np.mean(err[low])),
            "true_lt5_mae": float(np.mean(np.abs(err[low]))),
        })
    high = y >= 20.0
    if high.any():
        out["global"].update({
            "true_ge20_n": int(high.sum()),
            "true_ge20_bias": float(np.mean(err[high])),
            "true_ge20_mae": float(np.mean(np.abs(err[high]))),
            "true_ge20_pred_mean": float(np.mean(p[high])),
            "true_ge20_true_mean": float(np.mean(y[high])),
        })

    bins = [(0.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, max_height)]
    for lo, hi in bins:
        mask = (y >= lo) & ((y <= hi) if hi == max_height else (y < hi))
        if not mask.any():
            continue
        ee = err[mask]
        pp = p[mask]
        yy = y[mask]
        row = {
            "height_min": float(lo),
            "height_max": float(hi),
            "n": int(mask.sum()),
            "true_mean": float(np.mean(yy)),
            "pred_mean": float(np.mean(pp)),
            "bias": float(np.mean(ee)),
            "mae": float(np.mean(np.abs(ee))),
            "pred_zero_frac": float(np.mean(pp <= 1e-6)),
            "pred_lt1_frac": float(np.mean(pp < 1.0)),
            "pred_lt2p5_frac": float(np.mean(pp < 2.5)),
            "over_2m_frac": float(np.mean(ee > 2.0)),
            "under_2m_frac": float(np.mean(ee < -2.0)),
        }
        if raw is not None:
            rr = raw[mask]
            row["raw_negative_frac"] = float(np.mean(rr < 0.0))
        out["by_true_height"].append(row)
    return out


def plot_low_height_zero_diagnostics(
    df: pd.DataFrame,
    *,
    title: str,
    max_height: float,
    out_dir: Path,
) -> None:
    y = pd.to_numeric(df["rh95"], errors="coerce").to_numpy(dtype=np.float64)
    p = pd.to_numeric(df["prediction_original_coords"], errors="coerce").to_numpy(dtype=np.float64)
    raw = None
    if "prediction_raw_original_coords" in df.columns:
        raw = pd.to_numeric(df["prediction_raw_original_coords"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if raw is not None:
        raw = raw[valid]
    if y.size == 0:
        return
    err = p - y
    diag = prediction_pathology_diagnostics(df, max_height=max_height)
    bins_df = pd.DataFrame(diag.get("by_true_height", []))

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))

    ax = axes[0, 0]
    low = y < 5.0
    edges = np.linspace(0, 15, 61)
    ax.hist(p[low], bins=edges, histtype="stepfilled", color="#ff7f0e", alpha=0.28, label="Pred | true<5 m")
    ax.hist(y[low], bins=edges, histtype="step", color="#4c78a8", lw=1.7, label="GEDI true<5 m")
    ax.axvline(0, color="black", ls="--", lw=1.0)
    ax.axvline(5, color="red", ls=":", lw=1.2, label="5 m")
    ax.set_title("Low-height distribution: true<5 m")
    ax.set_xlabel("Height / prediction (m)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.20)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if raw is not None and np.isfinite(raw).any():
        raw_f = raw[np.isfinite(raw)]
        edges_raw = np.linspace(max(-15, np.quantile(raw_f, 0.001)), min(15, np.quantile(raw_f, 0.999)), 80)
        ax.hist(raw_f, bins=edges_raw, color="#bab0ac", alpha=0.72, label="raw model output")
        ax.axvline(0, color="black", ls="--", lw=1.0, label="clamp threshold")
        ax.set_title("Raw output before clamp")
        ax.set_xlabel("Raw prediction (m)")
        ax.set_ylabel("Count")
        raw_neg = 100.0 * float(diag["global"].get("raw_negative_frac", np.nan))
        ax.text(0.03, 0.95, f"raw<0: {raw_neg:.1f}%\\nthese become 0 after clamp",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.55", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "Raw prediction column unavailable\\nRe-run Step07 with updated script.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.grid(alpha=0.20)
    if ax.has_data():
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    if not bins_df.empty:
        labels = [f"{r.height_min:g}-{r.height_max:g}" for r in bins_df.itertuples(index=False)]
        x = np.arange(len(labels))
        ax.bar(x - 0.25, 100 * bins_df["pred_zero_frac"], width=0.25, label="pred=0 %", color="#e45756")
        ax.bar(x, 100 * bins_df["pred_lt1_frac"], width=0.25, label="pred<1 %", color="#f58518")
        ax.bar(x + 0.25, 100 * bins_df["over_2m_frac"], width=0.25, label="over +2m %", color="#54a24b")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("% points")
        ax.set_title("Zero spike and low-height overprediction by GEDI bin")
        ax.grid(axis="y", alpha=0.20)
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    zoom = y < 10.0
    ax.scatter(y[zoom], p[zoom], s=8, alpha=0.28, linewidths=0, color="#4c78a8")
    ax.plot([0, 10], [0, 10], color="black", ls="--", lw=1.1, label="1:1")
    ax.axhline(0, color="#e45756", lw=1.1, label="pred=0")
    ax.axhline(5, color="#f58518", ls=":", lw=1.1, label="pred=5 m")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.set_xlabel("GEDI RH95 true (m)")
    ax.set_ylabel("Prediction clipped (m)")
    ax.set_title("Zoom 0–10 m: zero wall and low-height false elevation")
    ax.grid(alpha=0.20)
    ax.legend(fontsize=8)

    g = diag["global"]
    fig.suptitle(
        title
        + " - zero/low-height diagnostics\\n"
        + f"pred=0: {100*g.get('pred_zero_frac', 0):.1f}% | pred<1: {100*g.get('pred_lt1_frac', 0):.1f}% | "
          f"true<5->pred>5: {100*g.get('true_lt5_pred_gt5_frac', 0):.1f}% | "
          f"true>=20 bias: {g.get('true_ge20_bias', float('nan')):+.2f} m",
        y=1.02,
        fontsize=12.5,
    )
    fig.tight_layout()
    _save_fig(fig, out_dir / "07_zero_low_height_diagnostics.png")


def plot_local3x3_shift_diagnostics(
    df: pd.DataFrame,
    *,
    title: str,
    out_dir: Path,
) -> None:
    """Diagnostic only: compare original-coordinate readout with best local 3x3 readout."""
    required = {"rh95", "prediction_original_coords", "prediction_local3x3_oracle", "local3x3_dr", "local3x3_dc"}
    if not required.issubset(set(df.columns)):
        return
    y = pd.to_numeric(df["rh95"], errors="coerce").to_numpy(dtype=np.float64)
    p0 = pd.to_numeric(df["prediction_original_coords"], errors="coerce").to_numpy(dtype=np.float64)
    p1 = pd.to_numeric(df["prediction_local3x3_oracle"], errors="coerce").to_numpy(dtype=np.float64)
    dr = pd.to_numeric(df["local3x3_dr"], errors="coerce").to_numpy(dtype=np.float64)
    dc = pd.to_numeric(df["local3x3_dc"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p0) & np.isfinite(p1)
    y, p0, p1, dr, dc = y[valid], p0[valid], p1[valid], dr[valid], dc[valid]
    if y.size == 0:
        return
    e0 = p0 - y
    e1 = p1 - y
    m0 = metrics(y, p0)
    m1 = metrics(y, p1)
    improvement = np.abs(e0) - np.abs(e1)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4))

    ax = axes[0, 0]
    labels = ["MAE", "RMSE", "|bias|", "pred=0 %"]
    orig_vals = [
        m0.get("mae", np.nan),
        m0.get("rmse", np.nan),
        abs(m0.get("bias", np.nan)),
        100 * m0.get("pred_zero_frac", np.nan),
    ]
    loc_vals = [
        m1.get("mae", np.nan),
        m1.get("rmse", np.nan),
        abs(m1.get("bias", np.nan)),
        100 * m1.get("pred_zero_frac", np.nan),
    ]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, orig_vals, width=0.36, label="original GEDI pixel", color="#4c78a8")
    ax.bar(x + 0.18, loc_vals, width=0.36, label="best local 3x3 oracle", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Original vs local 3x3 oracle")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    bins = np.linspace(-10, 20, 80)
    ax.hist(improvement, bins=bins, color="#54a24b", alpha=0.78)
    ax.axvline(0, color="black", ls="--", lw=1.1)
    ax.set_xlabel("|err original| - |err local3x3| (m)")
    ax.set_ylabel("Count")
    ax.set_title("Positive = local 3x3 fixes a spatial offset")
    ax.grid(alpha=0.20)
    ax.text(
        0.03, 0.95,
        f"median gain={np.median(improvement):+.2f} m\n"
        f"mean gain={np.mean(improvement):+.2f} m\n"
        f"% improved={100*np.mean(improvement>0):.1f}%",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.55", alpha=0.90),
    )

    ax = axes[1, 0]
    grid = np.zeros((3, 3), dtype=int)
    for rr, cc in zip(dr.astype(int), dc.astype(int)):
        if -1 <= rr <= 1 and -1 <= cc <= 1:
            grid[rr + 1, cc + 1] += 1
    im = ax.imshow(grid, cmap="YlOrRd")
    for r in range(3):
        for c in range(3):
            ax.text(c, r, str(grid[r, c]), ha="center", va="center", fontsize=11)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["dc=-1", "dc=0", "dc=+1"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["dr=-1", "dr=0", "dr=+1"])
    ax.set_title("Chosen local 3x3 offset counts")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    bins_h = np.linspace(0, 45, 19)
    centers, med0, med1 = [], [], []
    for lo, hi in zip(bins_h[:-1], bins_h[1:]):
        mask = (y >= lo) & (y < hi if hi < bins_h[-1] else y <= hi)
        if mask.sum() >= 8:
            centers.append((lo + hi) / 2)
            med0.append(float(np.median(e0[mask])))
            med1.append(float(np.median(e1[mask])))
    if centers:
        ax.plot(centers, med0, "-o", lw=1.5, ms=3.5, label="original median residual", color="#4c78a8")
        ax.plot(centers, med1, "-o", lw=1.5, ms=3.5, label="local3x3 median residual", color="#f58518")
    ax.axhline(0, color="black", ls="--", lw=1.0)
    ax.set_xlabel("GEDI RH95 (m)")
    ax.set_ylabel("Median residual pred - GEDI (m)")
    ax.set_title("Does local 3x3 reduce height-dependent bias?")
    ax.grid(alpha=0.20)
    ax.legend(fontsize=8)

    fig.suptitle(
        title
        + " - local 3x3 spatial-offset diagnostic (oracle, not official metric)\n"
        + f"MAE original={m0.get('mae', np.nan):.3f} m -> local3x3={m1.get('mae', np.nan):.3f} m | "
          f"slope original={m0.get('slope', np.nan):.3f} -> local3x3={m1.get('slope', np.nan):.3f}",
        y=1.02,
        fontsize=12.5,
    )
    fig.tight_layout()
    _save_fig(fig, out_dir / "08_local3x3_shift_diagnostics.png")


def make_step07_plots(
    *,
    frame: pd.DataFrame,
    nearest: pd.DataFrame,
    bin_rows: list[dict[str, Any]],
    occurrence: dict[str, Any],
    unique_nearest: dict[str, Any],
    run_dir: Path,
    run_name: str,
    split: str,
    checkpoint: Path,
    max_height: float,
    output_dir: Path,
) -> Path:
    out_dir = output_dir / f"figures_{split}"
    title = f"{run_name} | {split} | {checkpoint.name} | original GEDI coords"
    plot_metric_cards(occurrence=occurrence, unique_nearest=unique_nearest, title=title, out_dir=out_dir)
    plot_scatter_density(nearest, metric=unique_nearest, title=title + " | unique-nearest", max_height=max_height, out_dir=out_dir)
    plot_height_distribution_and_error(nearest, title=title + " | unique-nearest", max_height=max_height, out_dir=out_dir)
    plot_signed_error_distribution(nearest, title=title + " | unique-nearest", out_dir=out_dir)
    plot_residuals_vs_height(nearest, title=title + " | unique-nearest", max_height=max_height, out_dir=out_dir)
    plot_training_bin_error_summary(nearest, title=title + " | unique-nearest", max_height=max_height, out_dir=out_dir)
    plot_error_histogram_and_cdf(nearest, title=title + " | unique-nearest", out_dir=out_dir)
    plot_metrics_by_height(bin_rows, title=title + " | occurrence", out_dir=out_dir)
    plot_training_history(run_dir=run_dir, title=run_name, out_dir=out_dir)
    plot_low_height_zero_diagnostics(nearest, title=title + " | unique-nearest", max_height=max_height, out_dir=out_dir)
    plot_local3x3_shift_diagnostics(nearest, title=title + " | unique-nearest", out_dir=out_dir)
    return out_dir


def load_runner():
    vendor_root = HERE.parent
    # insert(0) reverses priority: keep vendor_root last so its training package wins.
    for path in (HERE, PROJECT_ROOT, vendor_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    candidates = (
        HERE / "06_train_catalog_ablation.py",
        vendor_root / "06_train_growthloss_catalog.py",
        PROJECT_ROOT / "Pipeline" / "06_train_growthloss_catalog.py",
    )
    runner_path = next((path for path in candidates if path.is_file()), None)
    if runner_path is None:
        raise FileNotFoundError(
            "Cannot locate a compatible catalog runner. Checked: "
            + ", ".join(str(path) for path in candidates)
        )
    print(f"[STEP07 RUNNER] {runner_path}")
    spec = importlib.util.spec_from_file_location("catalog_runner_eval", str(runner_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import catalog runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-aux-k", type=int, default=2048)
    parser.add_argument("--checkpoint-name", default=None, help="Explicit checkpoint filename inside the run checkpoints directory.")
    parser.add_argument("--drop-channels", default=None, help="Comma-separated raw channel indices to drop before evaluation.")
    parser.add_argument("--bins", default="0,2.5,5,10,15,20,30,40,45")
    parser.add_argument("--min-height", type=float, default=0.0)
    parser.add_argument("--max-height", type=float, default=None)
    parser.add_argument(
        "--clip-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, clamp predictions to [0,max_height]. Default is disabled to expose shrinkage/overprediction honestly.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the complete Step07 diagnostic dashboard (default: enabled).",
    )
    args = parser.parse_args()

    min_height = float(args.min_height)
    max_height = args.max_height
    experiment_json = args.experiment_root / "experiment.json"
    if max_height is None and experiment_json.exists():
        try:
            experiment_cfg = json.loads(experiment_json.read_text(encoding="utf-8"))
            max_height = float(experiment_cfg.get("schema", {}).get("max_height_m", 45.0))
        except Exception:
            max_height = 45.0
    max_height = float(45.0 if max_height is None else max_height)
    if not math.isfinite(max_height) or max_height <= 0:
        raise ValueError(f"Invalid max_height={max_height}")
    if not math.isfinite(min_height) or min_height < 0:
        raise ValueError(f"Invalid min_height={min_height}")
    if min_height >= max_height:
        raise ValueError(f"min_height must be < max_height, got {min_height} >= {max_height}")
    pred_mode = "clipped_to_eval_window" if bool(args.clip_predictions) else "raw_no_upper_clip"
    print(f"[STEP07 DOMAIN] target range=[{min_height:g},{max_height:g}] m | predictions={pred_mode}", flush=True)

    runner = load_runner()
    from training.model import CanopyHyTecModel

    run_dir = args.runs_root / args.experiment_root.name / args.run_name
    requested_checkpoint = str(args.checkpoint_name).strip() if args.checkpoint_name else ""
    if requested_checkpoint:
        checkpoint = run_dir / "checkpoints" / requested_checkpoint
        if not checkpoint.exists() and not requested_checkpoint.endswith(".ckpt"):
            checkpoint = run_dir / "checkpoints" / f"{requested_checkpoint}.ckpt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Requested checkpoint not found: {checkpoint}")
    else:
        checkpoint = run_dir / "checkpoints" / "best.ckpt"
        if not checkpoint.exists():
            fallback = run_dir / "checkpoints" / "best_any.ckpt"
            if not fallback.exists():
                raise FileNotFoundError(f"Neither best.ckpt nor best_any.ckpt exists in {checkpoint.parent}")
            checkpoint = fallback
            print(f"[STEP07] official best unavailable; using {checkpoint.name}", flush=True)
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = ckpt["model"]
    true_ordinal = "ordinal_head.weight" in state
    state_prefix = "base_model." if true_ordinal else ""
    first_weight = state.get(f"{state_prefix}inc.net.0.weight")
    if first_weight is None:
        raise RuntimeError("Cannot infer in_channels/base_ch from checkpoint")
    base_ch = int(first_weight.shape[0])
    in_ch = int(first_weight.shape[1])
    base_model = CanopyHyTecModel(
        n_channels=in_ch,
        n_classes=1,
        dropout=0.15,
        base_ch=base_ch,
        use_attention=True,
    )
    model = (
        runner.TrueOrdinalRegressionHead(base_model, n_thresholds=int(state["ordinal_head.weight"].shape[0]))
        if true_ordinal
        else base_model
    )
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    drop_channels = [int(x.strip()) for x in str(args.drop_channels).split(",") if str(x).strip()] if args.drop_channels else []
    dataset = runner.CatalogNpyIterable(
        args.experiment_root,
        args.split,
        seed=42,
        in_ch=in_ch,
        max_aux_k=args.max_aux_k,
        drop_channels=drop_channels,
        balanced_height=False,
        batch_size=args.batch_size,
        shuffle=False,
        temporal_fusion=(in_ch == 19),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x, _, aux_rows, aux_cols, aux_y, aux_mask, _, _, _, meta = batch
            model_output = model(x.to(device).float())
            pred_map_raw = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
            if pred_map_raw.ndim == 4:
                pred_map_raw = pred_map_raw[:, 0]
            if bool(args.clip_predictions):
                pred_map = torch.clamp(pred_map_raw, 0.0, max_height)
            else:
                pred_map = pred_map_raw
            B, H, W = pred_map.shape
            pred_map_cpu = pred_map.detach().cpu().numpy()
            pred_raw_cpu = pred_map_raw.detach().cpu().numpy()
            for b in range(B):
                valid = aux_mask[b].bool() & torch.isfinite(aux_y[b])
                valid &= aux_rows[b].long().ge(0) & aux_rows[b].long().lt(H)
                valid &= aux_cols[b].long().ge(0) & aux_cols[b].long().lt(W)
                indices = torch.where(valid)[0]
                if indices.numel() == 0:
                    continue
                rr_cpu = aux_rows[b, indices].long().numpy()
                cc_cpu = aux_cols[b, indices].long().numpy()
                pred = pred_map_cpu[b, rr_cpu, cc_cpu]
                pred_raw = pred_raw_cpu[b, rr_cpu, cc_cpu]
                true = aux_y[b, indices].numpy()
                local_best_pred = np.empty_like(pred, dtype=np.float64)
                local_best_dr = np.zeros_like(pred, dtype=np.int16)
                local_best_dc = np.zeros_like(pred, dtype=np.int16)
                for jj, (r0, c0, yy) in enumerate(zip(rr_cpu, cc_cpu, true)):
                    best_v = float(pred[jj])
                    best_abs = abs(best_v - float(yy))
                    best_dr = 0
                    best_dc = 0
                    for dr_ in (-1, 0, 1):
                        rr2 = int(r0) + dr_
                        if rr2 < 0 or rr2 >= H:
                            continue
                        for dc_ in (-1, 0, 1):
                            cc2 = int(c0) + dc_
                            if cc2 < 0 or cc2 >= W:
                                continue
                            v = float(pred_map_cpu[b, rr2, cc2])
                            ae = abs(v - float(yy))
                            if ae < best_abs:
                                best_v = v
                                best_abs = ae
                                best_dr = dr_
                                best_dc = dc_
                    local_best_pred[jj] = best_v
                    local_best_dr[jj] = best_dr
                    local_best_dc[jj] = best_dc
                uid = meta["aux_shot_uid"][b, indices].numpy()
                delta = meta["aux_abs_temporal_delta_days"][b, indices].numpy()
                for j in range(len(true)):
                    rows.append({
                        "batch_index": int(batch_idx),
                        "aux_shot_uid": int(uid[j]),
                        "abs_temporal_delta_days": float(delta[j]) if np.isfinite(delta[j]) else np.nan,
                        "rh95": float(true[j]),
                        "prediction_raw_original_coords": float(pred_raw[j]),
                        "prediction_original_coords": float(pred[j]),
                        "prediction_local3x3_oracle": float(local_best_pred[j]),
                        "local3x3_dr": int(local_best_dr[j]),
                        "local3x3_dc": int(local_best_dc[j]),
                    })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No valid predictions were collected")
    frame = frame[
        frame["rh95"].between(min_height, max_height)
        & np.isfinite(frame["prediction_original_coords"])
    ].copy()
    overall = metrics(frame["rh95"].to_numpy(), frame["prediction_original_coords"].to_numpy())
    prediction_pathology = prediction_pathology_diagnostics(frame, max_height=max_height)
    local3x3_oracle = metrics(frame["rh95"].to_numpy(), frame["prediction_local3x3_oracle"].to_numpy())

    nearest = (
        frame.assign(
            _delta=frame["abs_temporal_delta_days"].fillna(np.inf),
            _row=np.arange(len(frame)),
        )
        .sort_values(["aux_shot_uid", "_delta", "_row"])
        .drop_duplicates("aux_shot_uid", keep="first")
    )
    unique_nearest = metrics(
        nearest["rh95"].to_numpy(),
        nearest["prediction_original_coords"].to_numpy(),
    )
    prediction_pathology_unique = prediction_pathology_diagnostics(nearest, max_height=max_height)
    local3x3_oracle_unique = metrics(nearest["rh95"].to_numpy(), nearest["prediction_local3x3_oracle"].to_numpy())

    edges = np.asarray([float(x) for x in args.bins.split(",")], dtype=np.float64)
    bin_rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        include_hi = bool(hi == edges[-1])
        mask = (frame["rh95"] >= lo) & (
            (frame["rh95"] <= hi) if include_hi else (frame["rh95"] < hi)
        )
        result = metrics(
            frame.loc[mask, "rh95"].to_numpy(),
            frame.loc[mask, "prediction_original_coords"].to_numpy(),
        )
        result.update({"height_min": float(lo), "height_max": float(hi)})
        bin_rows.append(result)

    output_dir = run_dir / "step07_original_coords_minmax"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{args.split}_predictions_original_coords.csv.gz", index=False, compression="gzip")
    pd.DataFrame(bin_rows).to_csv(output_dir / f"{args.split}_metrics_by_height.csv", index=False)
    pd.DataFrame(prediction_pathology.get("by_true_height", [])).to_csv(
        output_dir / f"{args.split}_prediction_pathology_by_height.csv",
        index=False,
    )
    payload = {
        "evaluation_coordinates": "original_GEDI_coordinates",
        "spatial_shift_applied": False,
        "plots_use_unique_nearest_gedi": True,
        "min_height_m": min_height,
        "max_height_m": max_height,
        "prediction_clipped_to_eval_window": bool(args.clip_predictions),
        "split": args.split,
        "checkpoint": str(checkpoint),
        "occurrence": overall,
        "unique_nearest": unique_nearest,
        "metrics_by_height": bin_rows,
        "prediction_pathology": prediction_pathology,
        "prediction_pathology_unique_nearest": prediction_pathology_unique,
        "local3x3_oracle": local3x3_oracle,
        "local3x3_oracle_unique_nearest": local3x3_oracle_unique,
    }
    if args.plots:
        figures_dir = make_step07_plots(
            frame=frame,
            nearest=nearest,
            bin_rows=bin_rows,
            occurrence=overall,
            unique_nearest=unique_nearest,
            run_dir=run_dir,
            run_name=args.run_name,
            split=args.split,
            checkpoint=checkpoint,
            max_height=max_height,
            output_dir=output_dir,
        )
        payload["figures_dir"] = str(figures_dir)
        payload["figures"] = [str(path) for path in sorted(figures_dir.glob("*.png"))]
    (output_dir / f"{args.split}_metrics_original_coords.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "\n[STEP07 ORIGINAL GEDI] "
        f"run={args.run_name} | split={args.split} | checkpoint={checkpoint.name} | "
        f"mae={overall.get('mae', math.nan):.4f} | rmse={overall.get('rmse', math.nan):.4f} | "
        f"r2={overall.get('r2', math.nan):.4f} | bias={overall.get('bias', math.nan):+.4f} | "
        f"slope={overall.get('slope', math.nan):.4f} | stdr={overall.get('std_ratio', math.nan):.4f} | "
        f"pred_std={overall.get('pred_std', math.nan):.4f} | max_pred={overall.get('pred_max', math.nan):.2f} | "
        f"max_true={overall.get('true_max', math.nan):.2f} | n={overall.get('n', 0)}",
        flush=True,
    )
    print(
        "[STEP07 UNIQUE GEDI]   "
        f"mae={unique_nearest.get('mae', math.nan):.4f} | "
        f"slope={unique_nearest.get('slope', math.nan):.4f} | "
        f"stdr={unique_nearest.get('std_ratio', math.nan):.4f} | "
        f"bias={unique_nearest.get('bias', math.nan):+.4f} | n={unique_nearest.get('n', 0)}",
        flush=True,
    )
    pg = prediction_pathology.get("global", {})
    pgu = prediction_pathology_unique.get("global", {})
    print(
        "[STEP07 LOCAL3X3] occurrence "
        f"mae={overall.get('mae', math.nan):.4f}->{local3x3_oracle.get('mae', math.nan):.4f} | "
        f"slope={overall.get('slope', math.nan):.4f}->{local3x3_oracle.get('slope', math.nan):.4f} | "
        f"bias={overall.get('bias', math.nan):+.4f}->{local3x3_oracle.get('bias', math.nan):+.4f}",
        flush=True,
    )
    print(
        "[STEP07 LOCAL3X3] unique-nearest "
        f"mae={unique_nearest.get('mae', math.nan):.4f}->{local3x3_oracle_unique.get('mae', math.nan):.4f} | "
        f"slope={unique_nearest.get('slope', math.nan):.4f}->{local3x3_oracle_unique.get('slope', math.nan):.4f} | "
        f"bias={unique_nearest.get('bias', math.nan):+.4f}->{local3x3_oracle_unique.get('bias', math.nan):+.4f}",
        flush=True,
    )
    print(
        "[STEP07 ZERO/LOW]  occurrence "
        f"pred=0={100*pg.get('pred_zero_frac', math.nan):.1f}% | "
        f"pred<1={100*pg.get('pred_lt1_frac', math.nan):.1f}% | "
        f"raw<0={100*pg.get('raw_negative_frac', math.nan):.1f}% | "
        f"true<5->pred>5={100*pg.get('true_lt5_pred_gt5_frac', math.nan):.1f}% | "
        f"true>=20 bias={pg.get('true_ge20_bias', math.nan):+.2f} m",
        flush=True,
    )
    print(
        "[STEP07 ZERO/LOW]  unique-nearest "
        f"pred=0={100*pgu.get('pred_zero_frac', math.nan):.1f}% | "
        f"pred<1={100*pgu.get('pred_lt1_frac', math.nan):.1f}% | "
        f"raw<0={100*pgu.get('raw_negative_frac', math.nan):.1f}% | "
        f"true<5->pred>5={100*pgu.get('true_lt5_pred_gt5_frac', math.nan):.1f}% | "
        f"true>=20 bias={pgu.get('true_ge20_bias', math.nan):+.2f} m",
        flush=True,
    )
    for row in bin_rows:
        print(
            f"  [HEIGHT {row['height_min']:>4.1f}-{row['height_max']:<4.1f}m] "
            f"n={row.get('n', 0):>6} | mae={row.get('mae', math.nan):.4f} | "
            f"bias={row.get('bias', math.nan):+.4f} | slope={row.get('slope', math.nan):.4f}",
            flush=True,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
