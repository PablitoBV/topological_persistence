# barcode.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
from filtration import Simplex
import math

import matplotlib.pyplot as plt

Interval = Tuple[int, float, Union[float, str]]  # (dim, birth, death|'inf')

def write_barcode(
    simplices: List[Simplex],
    lowest_row_of_col: List[Optional[int]],
    birth_to_death: Dict[int, int],
    outfile: str,
) -> None:
    """Write the final barcode sorted logically: (dim, birth, death)."""
    lines: List[str] = []

    # Paired intervals (birth → death)
    for birth_row, death_col in sorted(birth_to_death.items()):
        k = simplices[birth_row].dim
        b = simplices[birth_row].val
        d = simplices[death_col].val
        lines.append(f"{k} {b} {d}")

    # Unpaired (essential) intervals
    used_births = set(birth_to_death.keys())
    for col, pivot_row in enumerate(lowest_row_of_col):
        if pivot_row is None and col not in used_births:
            k = simplices[col].dim
            b = simplices[col].val
            lines.append(f"{k} {b} inf")

    # Sorting logic: first by dim, then birth, then death (inf last)
    def sort_key(line: str) -> Tuple[int, float, float]:
        dim_str, birth_str, death_str = line.split()
        dim = int(dim_str)
        birth = float(birth_str)
        death = -1 if death_str == "inf" else float(death_str)
        return (dim, birth, death)

    lines.sort(key=sort_key)

    # Write to file and print for quick inspection
    with open(outfile, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    # print("\n=== Barcode ===")
    # for line in lines:
    #     print(line)

def plot_barcode(
    intervals: List[Interval],
    title: Optional[str] = None,
    filename: Optional[str] = None,
    x_padding: float = 0.05,
    band_gap: float = 1.2,
    line_width: float = 2.0,
    dpi: int = 180,
):
    """
    Plot a barcode diagram on a single Matplotlib axes.

    Parameters
    ----------
    intervals : list of (k, birth, death)
        death can be a float, 'inf', or None meaning infinite bar.
    title : optional str
        Title on top of the plot.
    filename : optional str
        If provided, save the plot to this path.
    x_padding : float
        Fractional padding added on the left/right of the min/max finite x.
    band_gap : float
        Vertical spacing between homology bands (H0, H1, H2, ...).
    line_width : float
        Width of the barcode line segments.
    dpi : int
        Resolution for saving.
    """
    if not intervals:
        fig, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No intervals", ha="center", va="center", transform=ax.transAxes)
        if filename:
            fig.savefig(filename, bbox_inches="tight", dpi=dpi)
        plt.show()
        return

    # Normalize and collect
    cleaned: List[Tuple[int, float, Optional[float]]] = []
    min_x = math.inf
    max_x = -math.inf
    max_dim = 0
    for k, b, d in intervals:
        max_dim = max(max_dim, k)
        if isinstance(d, str) and d.lower() == "inf":
            d_val = None
        else:
            d_val = float(d) if d is not None else None
        b_val = float(b)
        cleaned.append((k, b_val, d_val))
        min_x = min(min_x, b_val)
        if d_val is not None:
            max_x = max(max_x, d_val, max_x)
        else:
            max_x = max(max_x, b_val, max_x)

    # Compute x-limits with padding
    x_span = max(1e-6, max_x - min_x)
    x_pad = x_span * x_padding
    x_left = min_x - x_pad
    x_right = max_x + x_pad

    # Prepare y mapping per dimension
    # Each dim occupies a horizontal band centered at y = k*band_gap,
    # and within it bars are vertically stacked with a small offset.
    per_dim: List[List[Tuple[float, Optional[float]]]] = [[] for _ in range(max_dim + 1)]
    for k, b, d in cleaned:
        per_dim[k].append((b, d))

    # Sort each band by birth then death for reproducibility
    for k in range(max_dim + 1):
        per_dim[k].sort(key=lambda bd: (bd[0], math.inf if bd[1] is None else bd[1]))

    # Determine per-dimension stacking (rows)
    y_positions: List[List[float]] = []
    for k in range(max_dim + 1):
        items = per_dim[k]
        rows: List[float] = []
        y_positions.append(rows)
        # assign vertical slots; simple greedy stacking
        # spacing within band:
        inner_step = 0.06 * band_gap
        base = k * band_gap
        next_row_index = 0
        for _ in items:
            y = base - 0.4 * band_gap + next_row_index * inner_step
            rows.append(y)
            next_row_index += 1

    # Plot
    fig_height = max(2.2, (max_dim + 1) * (band_gap * 0.9) + 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=dpi)

    # Draw horizontal separators and Y tick labels "H k"
    for k in range(max_dim + 1):
        band_center = k * band_gap
        # horizontal separator below each band except the last
        if k > 0:
            ax.axhline(band_center - 0.5 * band_gap, linewidth=0.5, alpha=0.4)

    ax.set_yticks([k * band_gap for k in range(max_dim + 1)])
    ax.set_yticklabels([f"H {k}" for k in range(max_dim + 1)])

    
    # Plot bars
    for k in range(max_dim + 1):
        items = per_dim[k]
        rows_y = y_positions[k]
        # Séparer les barres infinies et finies
        infinite_bars = [(b, d, idx) for idx, (b, d) in enumerate(items) if d is None]
        finite_bars = [(b, d, idx) for idx, (b, d) in enumerate(items) if d is not None]
        # Dessiner d'abord les barres infinies
        for b, d, idx in infinite_bars:
            y = rows_y[idx]
            ax.plot([b, x_right], [y, y], linewidth=line_width)
            ax.annotate(
                "",
                xy=(x_right, y),
                xytext=(x_right - 0.001 * x_span, y),
                arrowprops=dict(arrowstyle="->", lw=line_width),
            )
        # Puis les barres finies
        for b, d, idx in finite_bars:
            y = rows_y[idx]
            ax.plot([b, d], [y, y], linewidth=line_width)

    # Aesthetics
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel("Filtration value")
    if title:
        ax.set_title(title)

    # Expand y-lims to avoid clipping arrows
    ymin = -0.6 * band_gap
    ymax = max_dim * band_gap + 0.6 * band_gap
    ax.set_ylim(ymin, ymax)

    # Reduce visual clutter
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.tight_layout()
    if filename:
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)
    plt.show()


# ---- Build intervals from a reduction result (same inputs as write_barcode) ----

def build_intervals(
    simplices: List[Simplex],
    lowest_row_of_col: List[Optional[int]],
    birth_to_death: Dict[int, int],
) -> List[Interval]:
    intervals: List[Interval] = []
    # paired
    for i, j in birth_to_death.items():
        intervals.append((simplices[i].dim, simplices[i].val, simplices[j].val))
    # essentials (no double count)
    used_births = set(birth_to_death.keys())
    for j, low in enumerate(lowest_row_of_col):
        if low is None and j not in used_births:
            intervals.append((simplices[j].dim, simplices[j].val, "inf"))
    # sort (dim, birth, death), put inf last
    intervals.sort(key=lambda t: (t[0], t[1], float("inf") if t[2] == "inf" else float(t[2])))
    return intervals

# ---- Filtering & Betti ----

def filter_by_length(
    intervals: List[Interval],
    *,
    min_length: float = 0.0,
    relative: bool = False,
) -> List[Interval]:
    """Drop finite bars with length < min_length. If relative=True, threshold is fraction of global span."""
    if not intervals or min_length <= 0:
        return intervals[:]
    # global span (per-dim span is also fine; global matches your B remark)
    finite_vals = [float(x) for _, b, d in intervals for x in (b, d if d != "inf" else b)]
    if not finite_vals:
        return intervals[:]
    x_min, x_max = min(finite_vals), max(finite_vals)
    span = max(1e-12, x_max - x_min)
    thr = min_length * span if relative else min_length

    out: List[Interval] = []
    for k, b, d in intervals:
        if d == "inf":
            out.append((k, b, d))
        else:
            length = float(d) - float(b)
            if length >= thr:
                out.append((k, b, d))
    return out

def betti_from_intervals(intervals: List[Interval]) -> Dict[int, int]:
    """Count infinite bars per dimension."""
    betti: Dict[int, int] = {}
    for k, _b, d in intervals:
        if d == "inf":
            betti[k] = betti.get(k, 0) + 1
    return betti

def guess_space(betti: Dict[int, int]) -> str:
    """Heuristic guess from Betti numbers on low dims."""
    b0 = betti.get(0, 0)
    b1 = betti.get(1, 0)
    b2 = betti.get(2, 0)

    if b0 == 0:
        return "empty set (no components)"
    if b0 > 1 and b1 == 0 and b2 == 0:
        return f"{b0} disjoint contractible components"
    if b0 == 1 and b1 == 0 and b2 == 0:
        return "contractible (disk/ball/tree-like)"
    if b0 == 1 and b1 == 1 and b2 == 0:
        return "circle-like (S^1) or cylinder-ish"
    if b0 == 1 and b1 == 0 and b2 == 1:
        return "sphere-like (S^2)"
    if b0 == 1 and b1 == 2 and b2 == 1:
        return "torus (T^2)"
    if b0 == 1 and b2 == 1 and any(b1 == 2*g for g in range(2, 10)):
        g = b1 // 2
        return f"closed orientable surface of genus {g}"
    if b0 == 1 and b1 > 0 and b2 == 0:
        return f"one component with {b1} independent 1-cycles (wedge of {b1} circles?)"
    return f"undetermined from low Betti: β0={b0}, β1={b1}, β2={b2}"

# ---- Plotting (same spirit as your plot_barcode, adds log-x & min-length) ----

def plot_barcode_from_reduction(
    simplices: List[Simplex],
    lowest_row_of_col: List[Optional[int]],
    birth_to_death: Dict[int, int],
    outfile: str,
    *,
    title: Optional[str] = None,
    log_x: bool = False,
    min_length: float = 0.0,     # if >0, drop finite bars shorter than this (absolute)
    relative: bool = False,      # if True, treat min_length as fraction of global span
) -> None:
    intervals = build_intervals(simplices, lowest_row_of_col, birth_to_death)
    intervals = filter_by_length(intervals, min_length=min_length, relative=relative)
    plot_barcode(intervals, outfile=outfile, title=title, log=log_x)

def plot_barcode(
    intervals: List[Interval],
    outfile: Optional[str] = None,
    title: Optional[str] = None,
    *,
    log: bool = False,          # True => xscale='symlog' (gère x < 0)
    min_length: float = 0.0,    # filtre des barres finies; 0 => pas de filtre
    relative: bool = False,     # si True, min_length est fraction du span visible
) -> None:
    """
    Trace un barcode persistant avec **dimension max en haut** et H0 en bas.

    Paramètres
    ----------
    intervals : list[(k, b, d|'inf')]
        (dimension k, naissance b, mort d). Utiliser "inf" pour une barre infinie.
    outfile : str | None
        Si fourni, sauvegarde (PNG/SVG/PDF selon l'extension).
    title : str | None
        Titre optionnel.
    log : bool
        Si True, applique une échelle x 'symlog' (log symétrique, supporte x négatifs).
    min_length : float
        Seuil de longueur pour **barres finies** (les 'inf' sont toujours gardées).
        - Si relative=False : seuil absolu dans l’unité affichée (linéaire ou symlog).
        - Si relative=True  : seuil = min_length * (étendue visible globale).
    relative : bool
        Interprétation relative du seuil si True.
    """
    # Cas trivial
    if not intervals:
        fig, ax = plt.subplots(figsize=(8, 2), dpi=160)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No intervals", ha="center", va="center", transform=ax.transAxes)
        if outfile: fig.savefig(outfile, bbox_inches="tight", dpi=160)
        plt.show()
        return

    # Normalisation
    cleaned: List[Tuple[int, float, Optional[float]]] = []
    finite_vals: List[float] = []
    dims_set = set()
    for k, b, d in intervals:
        k = int(k)
        b = float(b)
        d_val = None if (isinstance(d, str) and d == "inf") else float(d)
        cleaned.append((k, b, d_val))
        dims_set.add(k)
        finite_vals.append(b)
        if d_val is not None:
            finite_vals.append(d_val)

    # Longueur mesurée dans le domaine d'affichage (lin ou symlog)
    def to_display_x(x: float) -> float:
        if not log:
            return x
        # approx cohérente avec perception en symlog pour le filtrage
        return math.copysign(math.log10(abs(x) + 1.0), x)

    # Seuil
    if min_length > 0.0 and finite_vals:
        xs_disp = [to_display_x(v) for v in finite_vals]
        span_disp = max(1e-12, max(xs_disp) - min(xs_disp))
        thr = (min_length * span_disp) if relative else min_length
    else:
        thr = 0.0

    # Filtrage
    kept: List[Tuple[int, float, Optional[float]]] = []
    for k, b, d in cleaned:
        if d is None:
            kept.append((k, b, d))  # garder les infinies
        else:
            if (to_display_x(d) - to_display_x(b)) >= thr:
                kept.append((k, b, d))

    if not kept:
        fig, ax = plt.subplots(figsize=(8, 2), dpi=160)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "All bars filtered out", ha="center", va="center", transform=ax.transAxes)
        if outfile: fig.savefig(outfile, bbox_inches="tight", dpi=160)
        plt.show()
        return

    # Groupement par dimension et **ordre vertical: max→...→0**
    dims = sorted({k for k, _, _ in kept}, reverse=True)  # max en premier (haut)
    per_dim = {k: [] for k in dims}
    for k, b, d in kept:
        per_dim[k].append((b, d))
    for k in dims:
        per_dim[k].sort(key=lambda bd: (bd[0], math.inf if bd[1] is None else bd[1]))

    # x-lims en domaine linéaire (puis on appliquera symlog si demandé)
    all_finite = [x for k in dims for (b, d) in per_dim[k] for x in (b, d if d is not None else b)]
    x_min, x_max = min(all_finite), max(all_finite)
    x_span = max(1e-12, x_max - x_min)
    x_pad = 0.01 * x_span
    x_left, x_right = x_min - x_pad, x_max + x_pad

    # Layout vertical sans chevauchement: chaque bande a une fenêtre ±0.4*gap
    band_gap = 1.05
    band_half = 0.4 * band_gap
    fig_h = max(2.2, len(dims) * (band_gap * 0.9) + 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=180)

    # séparateurs + ticks (de haut en bas: k = dims[0] … dims[-1])
    for i in range(1, len(dims)):
        ax.axhline(i * band_gap - 0.5 * band_gap, linewidth=0.5, alpha=0.4)
    ax.set_yticks([i * band_gap for i in range(len(dims))])
    ax.set_yticklabels([f"H {k}" for k in dims])

    # tracer
    lw = 2.0
    for i, k in enumerate(dims):  # i=0 en haut = dim max
        base = i * band_gap
        bars = per_dim[k]
        m = len(bars)
        if m == 1:
            y_positions = [base]
        else:
            top, bot = base + band_half, base - band_half
            step = (top - bot) / (m - 1)
            y_positions = [bot + j * step for j in range(m)]
        for y, (b, d) in zip(y_positions, bars):
            if d is None:
                ax.plot([b, x_right], [y, y], linewidth=lw, color="darkblue")
            else:
                ax.plot([b, d], [y, y], linewidth=lw, color="darkblue")

    # échelle x
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel("Filtration (symlog)" if log else "Filtration")
    if log:
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
    if title:
        ax.set_title(title)

    # flèches des 'inf' vers la borne droite visible
    xr = ax.get_xlim()[1]
    for i, k in enumerate(dims):
        base = i * band_gap
        bars = per_dim[k]
        m = len(bars)
        if m == 1:
            y_positions = [base]
        else:
            top, bot = base + band_half, base - band_half
            step = (top - bot) / (m - 1)
            y_positions = [bot + j * step for j in range(m)]
        for y, (b, d) in zip(y_positions, bars):
            if d is None:
                ax.annotate("", xy=(xr, y), xytext=(xr - 0.02 * (x_right - x_left), y),
                            arrowprops=dict(arrowstyle="->", lw=lw))

    ax.set_ylim(-0.6 * band_gap, (len(dims) - 1) * band_gap + 0.6 * band_gap)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, bbox_inches="tight", dpi=180)
    plt.show()
