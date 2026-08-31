"""
Fig. 2 — compact 2x3 landscape layout of the study-area figure.

Replaces the previous 3-rows x 2-columns portrait figure (~1.5 pages) with
2 rows x 3 columns (~0.5 page) WITHOUT removing any panel:

    row 1 : spatially disjoint train/val/test patches over tree-cover density
    row 2 : unique GEDI RH95 shots per 5-m height class, by split
    columns: Ifran | Maamoura | Agadir

The spatial maps are retained in the main text on purpose: they are the visual
evidence that the partitions are spatially disjoint, which is the specific
methodological objection a locally trained model attracts.

HOW TO RUN
----------
Paste this file's contents into a new cell at the end of
02_Study_Areas_Spatial_Splits_and_GEDI_Support.ipynb,
after Section 7 has run, so that the following names already exist:

    SITE_DATA, GEDI_DISTRIBUTIONS, SPLIT_STYLE, SPLIT_LABEL, SPLIT_COLORS,
    QGIS_SPLIT_COLORS, TREE_DENSITY_CMAP, OUT_DIR, ARTICLE_IMAGES_DIR,
    EXPORT_DPI, nice_scale_length, save_figure

Output
------
    Fig_02_Study_Area_Splits_and_GEDI_2x3.{png,svg,pdf}   in OUT_DIR
    Study_Area_Spatial_Splits_ARTICLE.pdf                 in ARTICLE_IMAGES_DIR
"""

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.patches import Patch, Rectangle
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Guards: fail loudly rather than producing a half-empty figure
# ---------------------------------------------------------------------------
_required = ["SITE_DATA", "GEDI_DISTRIBUTIONS", "SPLIT_LABEL", "OUT_DIR",
             "EXPORT_DPI", "TREE_DENSITY_CMAP", "QGIS_SPLIT_COLORS"]
_missing = [n for n in _required if n not in globals()]
if _missing:
    raise NameError(
        f"Missing names from 02_Study_Areas_Spatial_Splits_and_GEDI_Support.ipynb: {_missing}. "
        "Run the notebook through Section 7 first, then execute this cell."
    )

FORESTS = ["Ifran", "Maamoura", "Agadir"]
TITLES = {
    "Ifran": "Ifran — moderately dense\nAtlas cedar forest",
    "Maamoura": "Maamoura — low-density\ncork-oak woodland",
    "Agadir": "Agadir — sparse\nargan woodland",
}

for _f in FORESTS:
    if _f not in SITE_DATA:
        raise RuntimeError(f"SITE_DATA has no entry for {_f}")
    _present = set(SITE_DATA[_f]["patch_gdf"]["split"].dropna().astype(str))
    if _present != {"train", "val", "test"}:
        raise RuntimeError(f"{_f}: expected frozen train/val/test, got {sorted(_present)}")


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------
def draw_split_map(ax, forest, data, show_density_key=False):
    """Train/val/test patches over the tree-density background, native CRS."""
    p = data["profile"]
    extent = (p["bounds"].left, p["bounds"].right,
              p["bounds"].bottom, p["bounds"].top)

    ax.imshow(data["density"], extent=extent, origin="upper",
              cmap=TREE_DENSITY_CMAP, vmin=1, vmax=100,
              aspect="equal", interpolation="nearest", zorder=0)

    for split in ("train", "val", "test"):
        part = data["patch_gdf"][data["patch_gdf"]["split"].eq(split)]
        if not len(part):
            continue
        is_test = split == "test"
        part.plot(ax=ax,
                  facecolor="#F2F2F2" if is_test else QGIS_SPLIT_COLORS[split],
                  edgecolor="black", linewidth=0.45,
                  hatch="////" if is_test else None,
                  alpha=1.0 if is_test else 0.58, zorder=20)

    if data.get("aoi_geometry") is not None:
        gpd.GeoSeries([data["aoi_geometry"]], crs=p["crs"]).boundary.plot(
            ax=ax, color="black", linewidth=0.7, zorder=30)

    # square framing with a small margin so patches clear the frame
    xmin, xmax, ymin, ymax = extent
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * 1.08 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_facecolor("#F7FAF7")
    ax.set_box_aspect(1.0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.7)

    # north arrow — identical size in all three panels
    ax.annotate("N", xy=(0.085, 0.945), xytext=(0.085, 0.795),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=7, fontweight="bold",
                arrowprops=dict(facecolor="black", edgecolor="white",
                                linewidth=0.5, width=1.5, headwidth=5.5,
                                headlength=6.0), zorder=70)

    # metric scale bar derived from the native projected width
    width_m = float(p["bounds"].right - p["bounds"].left)
    length_m = nice_scale_length(width_m)
    frac = min(0.30, max(0.14, length_m / max(width_m, 1.0)))
    x1, x0, y = 0.955, 0.955 - frac, 0.055
    ax.plot([x0, x1], [y, y], transform=ax.transAxes, color="white", lw=3.0,
            solid_capstyle="butt", zorder=71)
    ax.plot([x0, x1], [y, y], transform=ax.transAxes, color="black", lw=0.8,
            solid_capstyle="butt", zorder=72)
    label = (f"{length_m/1000:g} km" if length_m >= 1000 else f"{length_m:g} m")
    ax.text(0.5 * (x0 + x1), y + 0.030, label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5.6, color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.3),
            zorder=73)

    if show_density_key:
        cax = ax.inset_axes([1.035, 0.30, 0.030, 0.42], zorder=50)
        cbar = ColorbarBase(cax, cmap=TREE_DENSITY_CMAP,
                            norm=Normalize(vmin=1, vmax=100),
                            orientation="vertical")
        cbar.set_ticks([1, 100]); cbar.set_ticklabels(["1%", "100%"])
        cbar.ax.tick_params(labelsize=5.6, length=1.6, width=0.4, pad=1.8)
        cbar.outline.set_linewidth(0.45)
        cax.text(0.5, 1.10, "Tree cover", transform=cax.transAxes,
                 ha="center", va="bottom", fontsize=5.6)


def draw_gedi_histogram(ax, forest):
    """Unique GEDI RH95 shots per 5-m class, grouped by split."""
    table = GEDI_DISTRIBUTIONS[GEDI_DISTRIBUTIONS["forest"].eq(forest)]
    # preserve the numeric class order produced by the notebook's binning
    labels = list(dict.fromkeys(table["height_class"].tolist()))
    x = np.arange(len(labels), dtype=float)
    width = 0.26

    for offset, split in zip((-1, 0, 1), ("train", "val", "test")):
        part = (table[table["split"].eq(split)]
                .set_index("height_class").reindex(labels))
        is_test = split == "test"
        ax.bar(x + offset * width,
               np.asarray(part["count"].to_numpy(), dtype=float),
               width=width,
               color="#F2F2F2" if is_test else QGIS_SPLIT_COLORS[split],
               edgecolor="black" if is_test else "none",
               linewidth=0.7 if is_test else 0.0,
               hatch="////" if is_test else None,
               alpha=1.0 if is_test else 0.90,
               zorder=3 if is_test else 2,
               label=SPLIT_LABEL[split])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=5.8)
    ax.tick_params(axis="y", labelsize=6.0)
    ax.set_xlabel("GEDI RH95 height class", fontsize=6.6)
    ax.grid(axis="y", alpha=0.22, zorder=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------------------
# Assemble: 2 rows x 3 columns
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 3, figsize=(13.2, 8.1),
    gridspec_kw={"height_ratios": [1.42, 1.00], "hspace": 0.36, "wspace": 0.24},
)
fig.subplots_adjust(left=0.055, right=0.945, bottom=0.115, top=0.905)

for col, forest in enumerate(FORESTS):
    data = SITE_DATA[forest]
    draw_split_map(axes[0, col], forest, data,
                   show_density_key=(col == len(FORESTS) - 1))
    draw_gedi_histogram(axes[1, col], forest)

    axes[0, col].set_title(TITLES[forest], fontsize=8.6,
                           fontweight="semibold", pad=6, linespacing=1.18)
    axes[1, col].set_ylabel("Unique GEDI RH95 shots" if col == 0 else "",
                            fontsize=6.6)

    axes[0, col].text(0.015, 0.985, f"({'abc'[col]})",
                      transform=axes[0, col].transAxes, ha="left", va="top",
                      fontsize=7.4, fontweight="bold",
                      bbox=dict(facecolor="white", edgecolor="none",
                                alpha=0.80, pad=1.0), zorder=80)
    axes[1, col].text(0.015, 0.985, f"({'def'[col]})",
                      transform=axes[1, col].transAxes, ha="left", va="top",
                      fontsize=7.4, fontweight="bold",
                      bbox=dict(facecolor="white", edgecolor="none",
                                alpha=0.80, pad=1.0), zorder=80)

# one shared legend for both rows
handles = [
    Patch(facecolor="#F2F2F2" if s == "test" else QGIS_SPLIT_COLORS[s],
          edgecolor="black" if s == "test" else "none",
          linewidth=0.7 if s == "test" else 0.0,
          hatch="////" if s == "test" else None,
          alpha=1.0 if s == "test" else 0.72,
          label=SPLIT_LABEL[s])
    for s in ("train", "val", "test")
]
handles.append(Patch(facecolor="none", edgecolor="black", linewidth=0.7,
                     label="Valid AOI boundary"))
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
           fontsize=7.4, bbox_to_anchor=(0.5, 0.018), columnspacing=1.9,
           handlelength=1.9, handleheight=1.05)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
stem = "Fig_02_Study_Area_Splits_and_GEDI_2x3"
paths = {
    "png": OUT_DIR / f"{stem}.png",
    "svg": OUT_DIR / f"{stem}.svg",
    "pdf": OUT_DIR / f"{stem}.pdf",
}
fig.savefig(paths["png"], dpi=EXPORT_DPI, bbox_inches="tight",
            pad_inches=0.03, facecolor="white")
fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.03, facecolor="white")
fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.03, facecolor="white")

# Asset consumed by the manuscript. Written under a NEW name so that the
# previous 3x2 portrait figure (Study_Area_Spatial_Splits_ARTICLE.pdf) is
# preserved and the two layouts can be compared side by side.
article_pdf = ARTICLE_IMAGES_DIR / "Fig_02_Study_Area_Splits_and_GEDI_2x3.pdf"
fig.savefig(article_pdf, bbox_inches="tight", pad_inches=0.03, facecolor="white")

plt.show()
plt.close(fig)

print("Saved 2x3 landscape Fig. 2:")
for k, v in paths.items():
    print(f"  {k.upper()}: {v}")
print(f"  ARTICLE PDF: {article_pdf}")
print("\nThe manuscript references this file as")
print("  Images/Fig_02_Study_Area_Splits_and_GEDI_2x3.pdf")
print("The previous 3x2 portrait figure is left untouched.")
