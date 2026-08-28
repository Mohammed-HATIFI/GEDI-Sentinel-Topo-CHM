"""
Attention U-Net architecture figure — 15-channel input -> dense canopy-height regression.

Conventional U-Net figure style (cf. Ronneberger et al. 2015):
  * block HEIGHT encodes spatial resolution (tall = 512x512, short = 32x32)
  * channel depth annotated above each block, spatial resolution under each encoder pair
  * blue diamonds mark 3x3 Conv + BN + ReLU
  * pale-green blocks = pooled feature maps entering lower encoder levels
  * cyan dashed blocks = channel-wise concatenation in the decoder
  * orange arrows = MaxPool 2x2 (down) ; purple arrows = transposed Conv 2x2 (up)
 * grey dotted attention-filtered skip connections, explicitly labelled as skip connections
  * red block = 1x1 Conv output ; boxed legend on the right

Vertical layout is computed by stacking: each level's top edge sits LEVEL_GAP below
the previous level's bottom edge, so the pooling arrows visibly descend instead of
running almost horizontally.

Run:  python plot_attention_unet.py
Out:  Fig_AttentionUNet_Architecture.{pdf,svg,png}   (vector PDF is the LaTeX asset)
"""

from pathlib import Path
import colorsys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent
EXPORT_DPI = 600

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
GREEN       = "#2e7031"
GOLD        = "#8a6d1f"
PALE        = "#dceadb"
RED         = "#8c1f28"
CONV_BLUE   = "#3f7fbf"
POOL_ORANGE = "#e2861c"
UP_PURPLE   = "#7a4fa3"
SKIP_GREEN  = "#4c9a5b"
GATE_PINK   = "#b03a6e"
INK         = "#222222"
MUTED       = "#555555"


def shade(hex_color, factor):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, max(0.0, min(1.0, l * factor)), s)
    return (r, g, b)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
RES_LABEL = {0: "512×512", 1: "256×256", 2: "128×128", 3: "64×64", 4: "32×32"}
BLOCK_H   = {0: 2.95, 1: 2.30, 2: 1.78, 3: 1.34, 4: 1.00}

BW        = 0.30    # fallback block width
DEPTH     = 0.11    # pseudo-3D offset
BLOCK_GAP = 0.40    # leaves about 0.3 cm of visible connector after pseudo-depth
LEVEL_GAP = 0.92    # vertical clearance between one level's bottom and the next level's top

# Width encodes channel depth, while height encodes spatial resolution.  The
# mapping is deliberately compressed so that 1024-channel tensors remain
# legible without dominating the page.
CHANNEL_W = {1: 0.14, 64: 0.22, 128: 0.30, 256: 0.42,
             512: 0.58, 1024: 0.82}


def channel_width(channels):
    return CHANNEL_W[int(channels)]


def group_centres(group_center, widths, gap=BLOCK_GAP):
    total = sum(widths) + gap * (len(widths) - 1)
    cursor = group_center - total / 2
    centres = []
    for width in widths:
        centres.append(cursor + width / 2)
        cursor += width + gap
    return centres

# Stack the levels downwards so every pooling arrow has a real vertical drop.
Y_CENTER = {0: 10.55}
for _lv in range(1, 5):
    _bottom_prev = Y_CENTER[_lv - 1] - BLOCK_H[_lv - 1] / 2
    Y_CENTER[_lv] = _bottom_prev - LEVEL_GAP - BLOCK_H[_lv] / 2

ENC_STEP = 1.95     # horizontal offset per encoder level (keeps arrows steep)
DEC_STEP = 2.85     # room for channel-proportional 3-block decoder groups

ENC_X = {lv: 1.05 + lv * ENC_STEP for lv in range(4)}       # centre of the 2-block pair
BOT_X = ENC_X[3] + ENC_STEP                                  # centre of bottleneck pair
DEC_X = {3: BOT_X + DEC_STEP, 2: BOT_X + 2 * DEC_STEP,
         1: BOT_X + 3 * DEC_STEP, 0: BOT_X + 4 * DEC_STEP}


def draw_block(ax, xc, yc, h, color, top_label=None, w=BW, zorder=4,
               label_dx=0.0):
    x, y = xc - w / 2, yc - h / 2
    dx, dy = DEPTH, DEPTH * 0.62
    edge = shade(color, 0.5)

    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         facecolor=shade(color, 0.74), edgecolor=edge,
                         linewidth=0.45, zorder=zorder))
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         facecolor=shade(color, 1.24), edgecolor=edge,
                         linewidth=0.45, zorder=zorder))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=edge,
                           linewidth=0.55, zorder=zorder + 1))

    if top_label is not None:
        # Keep channel labels clear of the terminal pooling/upsampling arrows.
        ax.text(xc + dx / 2 + label_dx, y + h + dy + 0.28,
                str(top_label), ha="center",
                va="bottom", fontsize=6.3, color=INK, zorder=zorder + 2)


def draw_concat_block(ax, xc, yc, h, top_label=None, w=BW, zorder=5):
    """Concatenation: gated skip (dashed) + upsampled tensor (solid)."""
    x, y = xc - w / 2, yc - h / 2
    dx, dy = DEPTH, DEPTH * 0.62
    dash = (0, (1.6, 1.4))
    edge = "#2f9db5"
    split = x + w / 2
    # Solid right half: upsampled decoder tensor.
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         facecolor="#d6eff1", edgecolor=edge,
                         linewidth=0.8, zorder=zorder))
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         facecolor="#f5fcfc", edgecolor=edge,
                         linestyle=dash, linewidth=0.9, zorder=zorder))
    ax.add_patch(Rectangle((split, y), w / 2, h, facecolor="#82c9c6",
                           edgecolor=edge, linewidth=0.8,
                           zorder=zorder + 1))
    # Dashed left half: attention-filtered encoder skip tensor.
    ax.add_patch(Rectangle((x, y), w / 2, h, facecolor="white",
                           edgecolor=edge, linestyle=dash, linewidth=0.9,
                           zorder=zorder + 2))
    if top_label is not None:
        # A larger vertical offset prevents the purple upsampling arrow from
        # touching the channel annotation of the concatenation block.
        ax.text(xc, y + h + dy + 0.28, str(top_label), ha="center",
                va="bottom", fontsize=6.3, color=INK, zorder=zorder + 2)


def conv_marker(ax, x0, x1, y, z=3):
    """Connector carrying a blue diamond: 3x3 Conv + BN + ReLU."""
    ax.plot([x0, x1], [y, y], color=CONV_BLUE, lw=0.9, zorder=z,
            solid_capstyle="butt")
    ax.plot([(x0 + x1) / 2], [y], marker="D", markersize=3.2,
            markerfacecolor=CONV_BLUE, markeredgecolor="white",
            markeredgewidth=0.4, zorder=z + 3)


def stage_arrow(ax, x0, y0, x1, y1, color):
    """Straight connector, used only for the input arrow."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                shrinkA=0, shrinkB=0, mutation_scale=12),
                zorder=3)


def right_angle_arrow(ax, start, elbow_x, end, color, direction):
    """Three-segment U-Net transition with a vertical terminal arrow.

    The reference figure uses an orthogonal path: a short vertical exit from
    the source block, a horizontal run, then a vertical arrow into the target
    block. ``direction`` is either ``down`` (pooling) or ``up`` (upsampling).
    """
    x0, y0 = start
    x1, y1 = end
    exit_dy = -0.16 if direction == "down" else 0.16
    y_exit = y0 + exit_dy

    # Draw the orthogonal polyline explicitly.  Keeping the terminal segment
    # separate guarantees a mathematically vertical line in PDF/SVG output.
    ax.plot([x0, x0], [y0, y_exit], color=color, lw=1.5,
            solid_capstyle="butt", zorder=3)
    ax.plot([x0, x1], [y_exit, y_exit], color=color, lw=1.5,
            solid_capstyle="butt", zorder=3)
    ax.plot([x1, x1], [y_exit, y1], color=color, lw=1.5,
            solid_capstyle="butt", zorder=3)
    marker = "v" if direction == "down" else "^"
    ax.plot([x1], [y1], marker=marker, markersize=6.4,
            markerfacecolor=color, markeredgecolor=color, zorder=4)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
y_top    = Y_CENTER[0] + BLOCK_H[0] / 2
y_bottom = Y_CENTER[4] - BLOCK_H[4] / 2

fig, ax = plt.subplots(figsize=(14.2, 7.4))
ax.set_xlim(-4.20, 28.90)
ax.set_ylim(y_bottom - 1.75, y_top + 2.35)
ax.set_aspect("equal")
ax.axis("off")

ax.text(11.4, y_top + 1.95,
        "Attention U-Net — 15-channel input to dense canopy-height regression",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
ax.text(3.6, y_top + 0.95, "Encoder", ha="center", fontsize=9.5,
        fontweight="bold", color=GREEN)
ax.text(19.0, y_top + 0.95, "Attention-gated decoder", ha="center", fontsize=9.5,
        fontweight="bold", color=UP_PURPLE)

# Geometry: height represents spatial resolution; compressed width represents
# channel depth.  The pale encoder block is the pooled tensor entering a level.
ENC_CH = {0: (64, 64), 1: (128, 128), 2: (256, 256), 3: (512, 512)}
ENC_IN = {1: 64, 2: 128, 3: 256}
ENC_POS = {}
for lv in range(4):
    channels = list(ENC_CH[lv]) if lv == 0 else [ENC_IN[lv], *ENC_CH[lv]]
    widths = [channel_width(c) for c in channels]
    ENC_POS[lv] = (channels, widths, group_centres(ENC_X[lv], widths))

BOT_CHANNELS = [512, 1024, 1024]
BOT_WIDTHS = [channel_width(c) for c in BOT_CHANNELS]
BOT_CENTRES = group_centres(BOT_X, BOT_WIDTHS)

DEC_CH = {3: (1024, 512, 512), 2: (512, 256, 256),
          1: (256, 128, 128), 0: (128, 64, 64)}
UP_CH = {3: 512, 2: 256, 1: 128, 0: 64}
DEC_POS = {}
for lv in range(4):
    channels = list(DEC_CH[lv])
    widths = [channel_width(c) for c in channels]
    DEC_POS[lv] = (channels, widths, group_centres(DEC_X[lv], widths))

# ---- input -----------------------------------------------------------------
_, enc0_widths, enc0_centres = ENC_POS[0]
x_first_edge = enc0_centres[0] - enc0_widths[0] / 2
stage_arrow(ax, x_first_edge - 0.80, Y_CENTER[0],
            x_first_edge - 0.08, Y_CENTER[0], "#888888")

# Structured input description.  The executable tensor contains 14 physical
# predictors plus one binary AOI-support channel; the wording deliberately
# mirrors the compact publication style of the reference U-Net diagram.
input_right = x_first_edge - 1.90
input_y = Y_CENTER[0] + 1.12
ax.text(input_right, input_y, "Input: 15-channel image", ha="right", va="top",
        fontsize=8.6, fontweight="bold", fontstyle="italic",
        color=POOL_ORANGE)
input_lines = [
    "Sentinel-2: 8 bands",
    "Sentinel-1: 4 layers",
    "Topography: 2 layers",
    "AOI support: 1 channel",
]
for row_index, label in enumerate(input_lines, start=1):
    row_y = input_y - 0.46 * row_index
    ax.text(input_right, row_y, f"-   {label}", ha="right", va="top",
            fontsize=7.1, fontstyle="italic", color=MUTED)

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
skip_from = {}

for lv in [0, 1, 2, 3]:
    yc, h = Y_CENTER[lv], BLOCK_H[lv]
    channels, widths, centres = ENC_POS[lv]
    if lv == 0:
        x_first = None
        x_c1, x_c2 = centres
        w_c1, w_c2 = widths
        c1, c2 = channels
    else:
        x_first, x_c1, x_c2 = centres
        w_first, w_c1, w_c2 = widths
        c_first, c1, c2 = channels
        draw_block(ax, x_first, yc, h, PALE, top_label=c_first, w=w_first,
                   label_dx=-0.42)
        conv_marker(ax, x_first + w_first / 2 + DEPTH, x_c1 - w_c1 / 2, yc)

    draw_block(ax, x_c1, yc, h, GREEN, top_label=c1, w=w_c1)
    conv_marker(ax, x_c1 + w_c1 / 2 + DEPTH, x_c2 - w_c2 / 2, yc)
    draw_block(ax, x_c2, yc, h, GREEN, top_label=c2, w=w_c2)
    skip_from[lv] = x_c2 + w_c2 / 2 + DEPTH

    # Spatial size aligned with the skip/attention-gate line.
    x_left = (x_first - w_first / 2) if x_first is not None else (x_c1 - w_c1 / 2)
    ax.text(x_left - 0.18, yc, RES_LABEL[lv], ha="right",
            va="center", fontsize=6.3, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    # Max-pooling: right-angle transition, matching the reference figure.
    nxt = lv + 1
    x_target = BOT_CENTRES[0] if nxt == 4 else ENC_POS[nxt][2][0]
    x_start = x_c2
    elbow_x = x_target
    right_angle_arrow(
        ax,
        start=(x_start, yc - h / 2),
        elbow_x=elbow_x,
        end=(x_target, Y_CENTER[nxt] + BLOCK_H[nxt] / 2 + 0.04),
        color=POOL_ORANGE,
        direction="down",
    )

# ---------------------------------------------------------------------------
# Bottleneck
# ---------------------------------------------------------------------------
yc, h = Y_CENTER[4], BLOCK_H[4]
x_in, xa, xb = BOT_CENTRES
w_in, wa, wb = BOT_WIDTHS
draw_block(ax, x_in, yc, h, PALE, top_label=512, w=w_in, label_dx=-0.42)
conv_marker(ax, x_in + w_in / 2 + DEPTH, xa - wa / 2, yc)
draw_block(ax, xa, yc, h, GOLD, top_label=1024, w=wa)
conv_marker(ax, xa + wa / 2 + DEPTH, xb - wb / 2, yc)
draw_block(ax, xb, yc, h, GOLD, top_label=1024, w=wb)
ax.text(x_in - w_in / 2 - 0.18, yc, RES_LABEL[4], ha="right", va="center",
        fontsize=6.3, color=MUTED,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.text(BOT_X, yc - h / 2 - 0.70, "Bottleneck", ha="center", va="top",
        fontsize=7.6, fontweight="bold", color=GOLD)

# ---------------------------------------------------------------------------
# Decoder — transposed conv up, attention-gated concatenation, two convolutions
# ---------------------------------------------------------------------------
src_x, src_lv = xb + wb / 2 + DEPTH, 4
for lv in [3, 2, 1, 0]:
    yc, h = Y_CENTER[lv], BLOCK_H[lv]
    channels, widths, centres = DEC_POS[lv]
    c_cat, c1, c2 = channels
    w_cat, w_c1, w_c2 = widths
    x_cat, x_c1, x_c2 = centres

    # Transposed convolution: right-angle ascent, matching the reference.
    right_angle_arrow(
        ax,
        start=(src_x, Y_CENTER[src_lv] + BLOCK_H[src_lv] / 2),
        elbow_x=x_cat,
        end=(x_cat, yc - h / 2 - 0.04),
        color=UP_PURPLE,
        direction="up",
    )

    # ConvTranspose2d reduces channel depth before concatenation.
    # Label the incoming transposed-convolution tensor beside, rather than on,
    # the vertical purple arrow and leave a visible clearance under the block.
    ax.text(x_cat + 0.30, yc - h / 2 - 0.50, str(UP_CH[lv]),
            ha="left", va="top", fontsize=5.8, color=UP_PURPLE)

    draw_concat_block(ax, x_cat, yc, h, top_label=c_cat, w=w_cat)
    conv_marker(ax, x_cat + w_cat / 2 + DEPTH, x_c1 - w_c1 / 2, yc)
    draw_block(ax, x_c1, yc, h, GREEN, top_label=c1, w=w_c1)
    conv_marker(ax, x_c1 + w_c1 / 2 + DEPTH, x_c2 - w_c2 / 2, yc)
    draw_block(ax, x_c2, yc, h, GREEN, top_label=c2, w=w_c2)

    # attention-gated skip connection from the matching encoder level
    x0, x1 = skip_from[lv], x_cat - w_cat / 2 - 0.02
    ax.plot([x0, x1], [yc, yc], color="#9a9a9a", lw=1.05, zorder=3,
            linestyle=(0, (1.4, 3.0)), dash_capstyle="butt")
    # The legend identifies these dotted paths once; repeating a boxed label
    # on every level obscures the line and adds no information.

    src_x, src_lv = x_c2 + w_c2 / 2 + DEPTH, lv

# ---- output ---------------------------------------------------------------
w_out = channel_width(1)
_, dec0_widths, dec0_centres = DEC_POS[0]
dec0_right = dec0_centres[-1] + dec0_widths[-1] / 2 + DEPTH
x_out = dec0_right + 0.78 + w_out / 2
conv_marker(ax, dec0_right, x_out - w_out / 2, Y_CENTER[0])
draw_block(ax, x_out, Y_CENTER[0], BLOCK_H[0], RED, top_label=1, w=w_out)
ax.text(x_out + 0.20, Y_CENTER[0] + BLOCK_H[0] / 2 + 0.52, "1×1 Conv",
        ha="center", fontsize=6.3, color=MUTED)
ax.text(x_out + 1.30, Y_CENTER[0] + 0.36, "Canopy height", ha="left",
        fontsize=6.9, fontweight="bold", color=INK)
ax.text(x_out + 1.30, Y_CENTER[0] + 0.06, "1 × 512 × 512", ha="left",
        fontsize=6.3, color=MUTED)
ax.text(x_out + 1.30, Y_CENTER[0] - 0.24, "10 m grid", ha="left",
        fontsize=6.3, color=MUTED)

# ---------------------------------------------------------------------------
# Compact unboxed legend, placed close to the lower-right decoder branch.
# The wording follows the actual implementation: decoder upsampling uses a
# transposed convolution, not bilinear interpolation.
# ---------------------------------------------------------------------------
# Shift the unboxed legend by a further 2.5 cm to the left. With the current
# canvas and x range, 2.5 cm corresponds to approximately 2.11 data units.
lx = 18.13
y_row = Y_CENTER[3] - 0.10
rows = [
    ("conv", "3×3 Conv + BN + ReLU", CONV_BLUE),
    ("skip", "Attention-filtered skip connection", SKIP_GREEN),
    ("pool", "MaxPool 2×2", POOL_ORANGE),
    ("up", "Transposed Conv 2×2", UP_PURPLE),
    ("concat", "Channel-wise concatenation", "#e8f7f8"),
    ("box", "1×1 Conv (output)", RED),
    ("resolution", "Spatial resolution", MUTED),
    ("channels", "Number of channels", MUTED),
]
for kind, text, color in rows:
    xs, xe = lx, lx + 0.88
    if kind == "conv":
        conv_marker(ax, xs, xe, y_row, z=10)
    elif kind == "skip":
        ax.plot([xs, xe], [y_row, y_row], color="#9a9a9a", lw=1.05,
                linestyle=(0, (1.4, 3.0)), zorder=10)
    elif kind == "gate":
        ax.plot([xs, xe], [y_row, y_row], color="#9a9a9a", lw=1.05,
                linestyle=(0, (1.4, 3.0)), zorder=10)
        attention_gate(ax, (xs + xe) / 2, y_row, w=0.42, h=0.27, z=11, fontsize=5.0)
    elif kind in ("pool", "up"):
        ax.annotate("", xy=(xe, y_row), xytext=(xs, y_row),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                    mutation_scale=11), zorder=10)
    elif kind == "concat":
        mid = (xs + xe) / 2
        ax.add_patch(Rectangle((xs, y_row - 0.115), mid - xs, 0.23,
                               facecolor="white", edgecolor="#2f9db5",
                               linestyle=(0, (1.6, 1.4)), linewidth=0.9,
                               zorder=10))
        ax.add_patch(Rectangle((mid, y_row - 0.115), xe - mid, 0.23,
                               facecolor="#82c9c6", edgecolor="#2f9db5",
                               linewidth=0.8, zorder=10))
    elif kind == "resolution":
        ax.text((xs + xe) / 2, y_row, "512×512", ha="center", va="center",
                fontsize=6.6, fontweight="bold", color=MUTED, zorder=10)
    elif kind == "channels":
        ax.text((xs + xe) / 2, y_row, "64", ha="center", va="center",
                fontsize=6.6, color=MUTED, zorder=10)
    else:
        ax.add_patch(Rectangle((xs, y_row - 0.115), xe - xs, 0.23,
                               facecolor=color, edgecolor=shade(color, 0.55),
                               linewidth=0.5, zorder=10))
    label_x = xe + (0.58 if kind in ("resolution", "channels") else 0.22)
    ax.text(label_x, y_row, text, ha="left", va="center", fontsize=6.6,
            color=INK, zorder=10)
    y_row -= 0.43

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
paths = {
    "pdf": OUT_DIR / "Fig_AttentionUNet_Architecture.pdf",
    "svg": OUT_DIR / "Fig_AttentionUNet_Architecture.svg",
    "png": OUT_DIR / "Fig_AttentionUNet_Architecture.png",
}
fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.04, facecolor="white")
fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.04, facecolor="white")
fig.savefig(paths["png"], dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.04,
            facecolor="white")
plt.close(fig)

print("Saved:")
for k, v in paths.items():
    print(f"  {k.upper()}: {v}")
print("\nLevel geometry (centre / top / bottom):")
for lv in range(5):
    yc, h = Y_CENTER[lv], BLOCK_H[lv]
    print(f"  L{lv} {RES_LABEL[lv]:>8}  centre={yc:6.2f}  top={yc + h / 2:6.2f}  "
          f"bottom={yc - h / 2:6.2f}")
