"""One style for every figure of the paper, after the landing page's charts.

    from calm_paper import figstyle as fs
    fs.use()                                   # rcParams: serif, frame, inward ticks, no grid
    color, ls = fs.series(conditions)          # hue by the first split, line style by the second
    fs.jump_lines(ax, thresholds)              # dotted lines where another feature's region switches
    fs.jump_marks(ax, fs.jumps(model, X, j))   # each line's hidden jump: arrow + [α, β] (§3.3)
    fs.DIVERGING                               # blue → white → orange, for the GA²M surfaces

The page draws in SVG with two hues on a cream ground; the paper prints the
same hues on white with black ink. A feature's curves are coloured by the
first condition of their region and dashed by the second, so a depth-2 tree
reads as two hues × two line styles instead of four unrelated colours.
"""

import re

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#2b5c8a"      # the page's --s1
ORANGE = "#b85c25"    # the page's --s2
GRAY = "#6b6155"      # the page's --muted: the GAM / GA²M curve
INK = "#000000"
HUES = [BLUE, ORANGE]
STYLES = ["-", (0, (3.2, 1.6)), (0, (3.2, 1.2, 1, 1.2)), (0, (6, 1.6))]   # solid, short dash, dash-dot, long dash
DIVERGING = LinearSegmentedColormap.from_list("page", [BLUE, "#ffffff", ORANGE])
LW = 1.2


def use(size=8):
    """Set matplotlib up for the paper: serif text, a closed hairline frame,
    ticks pointing inward on all four sides, no grid, a boxed legend."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": size, "axes.labelsize": size, "axes.titlesize": size,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1, "legend.fontsize": size - 1,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.linewidth": 0.6,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
        "axes.grid": False, "axes.facecolor": "white", "figure.facecolor": "white",
        "lines.linewidth": LW, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": True, "legend.fancybox": False, "legend.framealpha": 1, "legend.edgecolor": INK,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.borderpad": 0.35, "legend.labelspacing": 0.25,
        "legend.borderaxespad": 0.4,
        "patch.linewidth": 0.5,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def assign(regions):
    """`[(color, linestyle)]` for a feature's regions, given their condition
    tuples. Regions that share a first condition share a hue and differ by
    line style; a different first condition takes the freshest hue, and when
    both hues are taken, the next line style on the one with room. One region, no condition →
    blue, solid. Up to 2 × 4 distinct looks (a depth-2 tree needs 4, a
    depth-3 tree up to 8)."""
    regions = [tuple(r) for r in regions]
    groups = []
    for conds in regions:
        if conds and conds[0] not in groups:
            groups.append(conds[0])
    free = {(h, st) for h in range(len(HUES)) for st in range(len(STYLES))}
    slot = {}
    for g in groups:
        members = [c for c in regions if c and c[0] == g]
        room = [sum((h, st) in free for st in range(len(STYLES))) for h in range(len(HUES))]
        hue = max(range(len(HUES)), key=lambda h: room[h]) if max(room) >= len(members) else None   # the freshest hue
        if hue is None:   # no hue has room for the whole group: take whatever is left, in order
            picks = sorted(free, key=lambda t: (t[1], t[0]))[:len(members)]
        else:
            picks = [(hue, st) for st in range(len(STYLES)) if (hue, st) in free][:len(members)]
        for c, pk in zip(members, picks):
            slot[c] = pk
            free.discard(pk)
    return [(HUES[slot[c][0]], STYLES[slot[c][1]]) if c else (BLUE, "-") for c in regions]


def short(condition, math=False):
    """A condition as a short legend entry: `x2 ≥ 0.00` → `x₂ ≥ 0`, `temp < 0.46` kept."""
    s = re.sub(r"(-?\d+)\.00\b", r"\1", condition)
    s = s.replace("-", "−")
    if math:
        s = re.sub(r"\bx(\d+)\b", r"$x_{\1}$", s)
    return s


def label(conditions, math=False):
    return ", ".join(short(c, math) for c in conditions) if conditions else "everywhere"


def jump_lines(ax, thresholds):
    """Dotted vertical lines where another feature's region switches, i.e.
    the hidden jumps of §3.3; drawn behind the curves."""
    for t in thresholds:
        ax.axvline(t, color=GRAY, lw=0.8, ls=(0, (1, 1.6)), zorder=0.5)


def thresholds_on(model, j):
    """Where the other features' partitions cut feature `j`, in display units,
    sorted and unique; an integer-valued feature is cut halfway between ints."""
    import numpy as np

    builder = model._view_builder()
    X = builder._X if hasattr(builder, "_X") else None
    cuts = set()
    for k, part in model.partitions_.items():
        if k == j:
            continue
        for region in part.regions:
            sub = region.rule.conditions.get(j)
            if sub is None or not hasattr(sub, "lo"):
                continue
            for v in (sub.lo, sub.hi):
                if np.isfinite(v):
                    cuts.add(float(builder._show_x(j, v)))
    if X is not None and cuts:
        col = X[:, j]
        if np.all(np.isfinite(col)) and np.all(np.equal(np.mod(col, 1), 0)):
            cuts = {np.floor(c) + 0.5 for c in cuts}
    return sorted(cuts)


ARROWS = {"↑": r"$\uparrow$", "↓": r"$\downarrow$", "↕": r"$\updownarrow$"}


def jumps(model, X, j):
    """The hidden jump at each dotted line of an integer-valued feature `j` (the hour), as §3.3
    defines it: over every row of `X`, the change in the prediction from the integer below the line
    to the one above, minus the change along the row's own curve of `j`. Returns
    `[(cut, lo, hi, arrow)]`, arrow ↑ (always > 0), ↓ (always < 0) or ↕ (it depends)."""
    import numpy as np

    name, out = X.columns[j], []
    for c in thresholds_on(model, j):
        a, b = int(np.floor(c)), int(np.ceil(c))
        d = np.asarray(model.contributions(X.assign(**{name: b}))) - np.asarray(model.contributions(X.assign(**{name: a})))
        jump = d.sum(1) - d[:, j]
        arrow = "↑" if (jump > 0).all() else "↓" if (jump < 0).all() else "↕"
        out.append((c, float(jump.min()), float(jump.max()), arrow))
    return out


def jump_text(lo, hi, arrow, ranges=True):
    num = lambda v: f"{v:.0f}".replace("-", "−")
    return f"{ARROWS[arrow]} [{num(lo)}, {num(hi)}]" if ranges else ARROWS[arrow]


def jump_marks(ax, marks, ranges=True, ha="center", dx=0.0, y=0.97, size=6.5):
    """Each dotted line's arrow at the top of the axes, with its range [α, β] when `ranges`;
    `ha`/`dx` set it on the line (center, on a white patch that cuts the line) or beside it
    (right: just left of the line; left: just right of it)."""
    box = dict(facecolor="white", edgecolor="none", pad=0.4) if ha == "center" else None
    for c, lo, hi, arrow in marks:
        ax.text(c + dx, y, jump_text(lo, hi, arrow, ranges), transform=ax.get_xaxis_transform(), ha=ha, va="top",
                fontsize=size, color=GRAY, zorder=3, bbox=box)


def legend(ax, loc="best", above=None, **kw):
    """The legend: inside the axes for one or two entries, above them in two
    columns for more (or when `above` is set), where it hides no curve."""
    handles, labels = ax.get_legend_handles_labels()
    if above is None:
        above = len(labels) > 2
    if above:
        kw = {"loc": "lower center", "bbox_to_anchor": (0.5, 1.0), "ncol": 2, "fontsize": 6,
              "columnspacing": 0.8, "handlelength": 1.4, "borderaxespad": 0.15, **kw}
        leg = ax.legend(**kw)
    else:
        leg = ax.legend(loc=loc, **kw)
    leg.get_frame().set_linewidth(0.5)
    return leg
