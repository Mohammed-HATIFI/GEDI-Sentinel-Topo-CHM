from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from b4_c15_config import B4_C15_CHANNEL_ORDER, SITES


SPLITS = ["train", "val", "test"]
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
SPLIT_COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
PATCH_SIZE = 512


def _catalog_paths(root: Path) -> tuple[Path, Path]:
    sample_candidates = [root / "sample_catalog_step05.csv", root / "sample_catalog.csv"]
    shot_candidates = [root / "shot_catalog_step05.csv.gz", root / "shot_catalog.csv.gz"]
    sample = next((p for p in sample_candidates if p.exists()), None)
    shot = next((p for p in shot_candidates if p.exists()), None)
    if sample is None or shot is None:
        raise FileNotFoundError(f"STEP05 catalogues missing under {root}")
    return sample, shot


def _edges(site_key: str) -> np.ndarray:
    cfg = SITES[site_key]
    if cfg.train_max <= 20:
        return np.asarray([0, 2.5, 5, 8, 10, 15, 20], dtype=float)
    return np.asarray([0, 5, 10, 15, 20, 25, 30, 35, 40, cfg.train_max], dtype=float)


def _labels(edges: np.ndarray) -> list[str]:
    return [f"{a:g}-{b:g} m" for a, b in zip(edges[:-1], edges[1:])]


def _unique_points(shots: pd.DataFrame, cfg) -> pd.DataFrame:
    d = shots.copy()
    d["split"] = d["split"].astype(str).str.lower().replace({"validation": "val", "valid": "val"})
    d["height"] = pd.to_numeric(d["rh95"], errors="coerce")
    d = d[d["split"].isin(SPLITS) & d["height"].notna()].copy()
    d = d[(d["height"] >= cfg.train_min) & (d["height"] <= cfg.train_max)].copy()
    key = "aux_shot_uid" if "aux_shot_uid" in d.columns else "aux_shot_id"
    sort_cols = ["split", key]
    if "aux_abs_temporal_delta_days" in d.columns:
        sort_cols.append("aux_abs_temporal_delta_days")
    elif "sample_id" in d.columns:
        sort_cols.append("sample_id")
    return d.sort_values(sort_cols, na_position="last").drop_duplicates(["split", key], keep="first")


def _height_bin_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    return np.histogram(vals, bins=edges)[0].astype(int)


def _save_show(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    print("Saved:", path)
    plt.show()
    plt.close(fig)
    return path


def make_plots(site_key: str) -> list[Path]:
    cfg = SITES[site_key]
    sample_path, shot_path = _catalog_paths(cfg.catalog_root)
    samples = pd.read_csv(sample_path, low_memory=False)
    shots = pd.read_csv(shot_path, low_memory=False)
    samples["split"] = samples["split"].astype(str).str.lower().replace({"validation": "val", "valid": "val"})
    points = _unique_points(shots, cfg)
    edges = _edges(site_key)
    labels = _labels(edges)
    out = cfg.figures_root / "step05"
    out.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    counts = {
        split: _height_bin_counts(points.loc[points["split"] == split, "height"].to_numpy(), edges)
        for split in SPLITS
    }

    # 1. Same twin-axis grouped-bar style as Last_Ablation - Copie.
    fig, ax_train = plt.subplots(figsize=(15.5, 5.2), dpi=140)
    x = np.arange(len(labels))
    width = 0.26
    ax_other = ax_train.twinx()
    ax_train.bar(x - width, counts["train"], width=width, color=SPLIT_COLORS["train"], alpha=0.78,
                 label=f"Train (n={int(counts['train'].sum()):,})")
    ax_other.bar(x, counts["val"], width=width, color=SPLIT_COLORS["val"], alpha=0.78,
                 label=f"Validation (n={int(counts['val'].sum()):,})")
    ax_other.bar(x + width, counts["test"], width=width, color=SPLIT_COLORS["test"], alpha=0.78,
                 label=f"Test (n={int(counts['test'].sum()):,})")
    ax_train.set_xticks(x)
    ax_train.set_xticklabels(labels, rotation=25, ha="right")
    ax_train.set_ylabel("Train count")
    ax_other.set_ylabel("Validation/Test count")
    ax_train.set_xlabel("GEDI RH95 height class")
    ax_train.set_title(f"{cfg.label} — GEDI RH95 class distribution by split\nTrain uses left axis; validation/test use right axis", fontsize=13.5)
    ax_train.grid(axis="y", alpha=0.22)
    h1, l1 = ax_train.get_legend_handles_labels()
    h2, l2 = ax_other.get_legend_handles_labels()
    ax_train.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)
    generated.append(_save_show(fig, out / "01_height_class_distribution_by_split.png"))

    # 2. Same translucent colored patch-map style.
    required = {"split", "patch_row_start", "patch_col_start"}
    if required.issubset(samples.columns):
        subset = ["split", "patch_key"] if "patch_key" in samples.columns else ["split", "patch_row_start", "patch_col_start"]
        patches = samples.drop_duplicates(subset).copy()
        patches["r0"] = pd.to_numeric(patches["patch_row_start"], errors="coerce")
        patches["c0"] = pd.to_numeric(patches["patch_col_start"], errors="coerce")
        patches = patches.dropna(subset=["r0", "c0"])
        fig, ax = plt.subplots(figsize=(11.5, 10.5), dpi=145)
        for split in SPLITS:
            for _, row in patches[patches["split"] == split].iterrows():
                ax.add_patch(Rectangle((float(row["c0"]), float(row["r0"])), PATCH_SIZE, PATCH_SIZE,
                                       facecolor=SPLIT_COLORS[split], edgecolor=SPLIT_COLORS[split],
                                       linewidth=1.4, alpha=0.32))
        ax.autoscale_view()
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.set_xlabel("Column / x pixel")
        ax.set_ylabel("Row / y pixel")
        ax.set_title(f"{cfg.label} — spatially disjoint train / validation / test patches", fontsize=13.5)
        handles = [Patch(facecolor=SPLIT_COLORS[s], edgecolor=SPLIT_COLORS[s], alpha=0.35,
                         label=f"{SPLIT_LABELS[s]} patches (n={int((patches['split'] == s).sum())})") for s in SPLITS]
        ax.legend(handles=handles, loc="upper right", frameon=True)
        generated.append(_save_show(fig, out / "02_spatial_split_patch_map.png"))

    # 3. Same boxplot + step-density layout.
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.3), dpi=140)
    data = [points.loc[points["split"] == s, "height"].to_numpy(dtype=float) for s in SPLITS]
    bp = axes[0].boxplot(data, tick_labels=[SPLIT_LABELS[s] for s in SPLITS], showfliers=True,
                         patch_artist=True, medianprops=dict(color="black", linewidth=1.25),
                         whiskerprops=dict(color="0.25"), capprops=dict(color="0.25"))
    for patch, split in zip(bp["boxes"], SPLITS):
        patch.set_facecolor(SPLIT_COLORS[split])
        patch.set_alpha(0.35)
        patch.set_edgecolor(SPLIT_COLORS[split])
    means = [np.nanmean(v) if len(v) else np.nan for v in data]
    axes[0].scatter(np.arange(1, 4), means, marker="D", s=55, color="black", label="Mean", zorder=4)
    axes[0].set_ylabel("GEDI RH95 (m)")
    axes[0].set_ylim(cfg.train_min, cfg.train_max + 1)
    axes[0].set_title("RH95 boxplot by split")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=True)
    dense_bins = np.linspace(cfg.train_min, cfg.train_max, 32)
    for split, values in zip(SPLITS, data):
        axes[1].hist(values[np.isfinite(values)], bins=dense_bins, density=True, histtype="step",
                     linewidth=2.0, color=SPLIT_COLORS[split], label=f"{SPLIT_LABELS[split]} (n={len(values):,})")
    axes[1].set_xlabel("GEDI RH95 (m)")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(cfg.train_min, cfg.train_max)
    axes[1].set_title("RH95 density by split")
    axes[1].grid(alpha=0.20)
    axes[1].legend(frameon=True)
    fig.suptitle(f"{cfg.label} — RH95 distribution and mean boxplots", fontsize=14)
    generated.append(_save_show(fig, out / "03_rh95_boxplot_density_by_split.png"))

    # 4. Same YlGnBu annotated heatmap style.
    count_df = pd.DataFrame(counts, index=labels)
    pct = count_df.div(count_df.sum(axis=1).replace(0, np.nan), axis=0).mul(100)
    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=145)
    image = ax.imshow(pct[SPLITS].to_numpy(float), vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(3), [SPLIT_LABELS[s] for s in SPLITS])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Split percentage inside each RH95 height class\nIdeal target approximately 75 / 15 / 10", fontsize=13)
    for i in range(len(labels)):
        for j, split in enumerate(SPLITS):
            value = pct.iloc[i, j]
            label = "" if not np.isfinite(value) else f"{value:.1f}%\n(n={count_df.iloc[i, j]})"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(image, ax=ax).set_label("% within height bin")
    generated.append(_save_show(fig, out / "04_height_bin_split_ratio_heatmap.png"))

    patch_key = ["split", "patch_key"] if "patch_key" in samples.columns else ["split", "patch_row_start", "patch_col_start"]
    patch_df = samples.drop_duplicates(patch_key)
    rows = []
    for split in SPLITS:
        values = points.loc[points["split"] == split, "height"].to_numpy(dtype=float)
        rows.append({"site": cfg.label, "split": split,
                     "patch_count": int((patch_df["split"] == split).sum()),
                     "unique_gedi_n": int(len(values)),
                     "mean_rh95": float(np.nanmean(values)) if len(values) else np.nan,
                     "median_rh95": float(np.nanmedian(values)) if len(values) else np.nan,
                     "p90_rh95": float(np.nanpercentile(values, 90)) if len(values) else np.nan,
                     "max_rh95": float(np.nanmax(values)) if len(values) else np.nan,
                     "model_input_channels": 15, "palsar_in_model_input": False,
                     "channel_order": " | ".join(B4_C15_CHANNEL_ORDER)})
    summary_path = cfg.reports_root / "step05_protocol_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    manifest = {"site": cfg.label, "catalog": str(cfg.catalog_root), "sample_catalog": str(sample_path),
                "shot_catalog": str(shot_path), "figures": [str(p) for p in generated],
                "protocol_summary": str(summary_path), "style_reference": "Last_Ablation - Copie.ipynb / STEP05_VISUAL_AUDIT_ARTICLE_READY_V1"}
    (cfg.reports_root / "step05_visual_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Article-ready STEP05 plots for one forest.")
    parser.add_argument("--site", choices=sorted(SITES), required=True)
    args = parser.parse_args()
    make_plots(args.site)


if __name__ == "__main__":
    main()
