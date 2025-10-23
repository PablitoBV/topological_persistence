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

    print("\n=== Barcode ===")
    for line in lines:
        print(line)




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
    _plot_barcode(intervals, filename=outfile, title=title, log_x=log_x)

def _plot_barcode(
    intervals: List[Interval],
    title: Optional[str] = None,
    filename: Optional[str] = None,
    x_padding: float = 0.05,
    band_gap: float = 1.2,
    line_width: float = 2.0,
    dpi: int = 180,
    log_x: bool = False,
    # --- nouveau ---
    min_length: float = 0.0,      # 0 => pas de filtre
    relative: bool = False,       # si True, min_length est une fraction du span
    filter_in_log: bool = False,  # si True, applique le filtre après pseudo-log
):
    """Trace un barcode; supporte pseudo-log (valeurs négatives) et filtrage par longueur minimale."""
    if not intervals:
        fig, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No intervals", ha="center", va="center", transform=ax.transAxes)
        if filename: fig.savefig(filename, bbox_inches="tight", dpi=dpi)
        plt.show()
        return

    def pseudo_log(x: float) -> float:
        return math.copysign(math.log10(abs(x) + 1.0), x)

    # ---- normalisation initiale (linéaire) ----
    clean_lin: List[Tuple[int, float, Optional[float]]] = []
    min_x_lin, max_x_lin, max_dim = math.inf, -math.inf, 0
    for k, b, d in intervals:
        max_dim = max(max_dim, k)
        b_lin = float(b)
        d_lin = None if (isinstance(d, str) and d.lower() == "inf") else float(d)
        clean_lin.append((k, b_lin, d_lin))
        min_x_lin = min(min_x_lin, b_lin)
        max_x_lin = max(max_x_lin, b_lin if d_lin is None else d_lin)

    # ---- filtrage longueur (linéaire ou pseudo-log) ----
    def _filter(items: List[Tuple[int, float, Optional[float]]],
                use_log: bool) -> List[Tuple[int, float, Optional[float]]]:
        if min_length <= 0:
            return items
        # calcule le span pour seuil relatif
        vals = []
        for _, b, d in items:
            xb = pseudo_log(b) if use_log else b
            vals.append(xb)
            if d is not None:
                xd = pseudo_log(d) if use_log else d
                vals.append(xd)
        if not vals:
            return items
        span = max(1e-12, max(vals) - min(vals))
        thr = (min_length * span) if relative else min_length
        out = []
        for k, b, d in items:
            if d is None:
                out.append((k, b, d))          # garde les bars infinies
            else:
                xb = pseudo_log(b) if use_log else b
                xd = pseudo_log(d) if use_log else d
                if (xd - xb) >= thr:
                    out.append((k, b, d))     # conserve valeurs originales
        return out

    filtered_lin = _filter(clean_lin, use_log=filter_in_log)

    # ---- appliquer pseudo-log pour l'affichage si demandé ----
    if log_x:
        clean = [(k, pseudo_log(b), None if d is None else pseudo_log(d)) for k, b, d in filtered_lin]
        min_x = min(pseudo_log(min_x_lin), *(x for _, x, _ in clean))
        max_x = max(pseudo_log(max_x_lin), *(x for _, x, _ in clean))
    else:
        clean = filtered_lin
        min_x, max_x = min_x_lin, max_x_lin

    # padding
    x_span = max(1e-12, max_x - min_x)
    x_pad = x_span * x_padding
    x_left, x_right = min_x - x_pad, max_x + x_pad

    # regrouper par dimension
    per_dim: List[List[Tuple[float, Optional[float]]]] = [[] for _ in range(max_dim + 1)]
    for k, b, d in clean:
        per_dim[k].append((b, d))
    for k in range(max_dim + 1):
        per_dim[k].sort(key=lambda bd: (bd[0], math.inf if bd[1] is None else bd[1]))

    # positions verticales
    y_positions: List[List[float]] = []
    for k in range(max_dim + 1):
        rows, base, inner_step = [], k * band_gap, 0.06 * band_gap
        for i in range(len(per_dim[k])):
            rows.append(base - 0.4 * band_gap + i * inner_step)
        y_positions.append(rows)

    # plot
    fig_h = max(2.2, (max_dim + 1) * (band_gap * 0.9) + 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=dpi)

    for k in range(1, max_dim + 1):
        ax.axhline(k * band_gap - 0.5 * band_gap, linewidth=0.5, alpha=0.4)
    ax.set_yticks([k * band_gap for k in range(max_dim + 1)])
    ax.set_yticklabels([f"H {k}" for k in range(max_dim + 1)])

    for k in range(max_dim + 1):
        for idx, (b, d) in enumerate(per_dim[k]):
            y = y_positions[k][idx]
            if d is None:
                ax.plot([b, x_right], [y, y], linewidth=line_width)
                ax.annotate("", xy=(x_right, y), xytext=(x_right - 0.002 * x_span, y),
                            arrowprops=dict(arrowstyle="->", lw=line_width))
            else:
                ax.plot([b, d], [y, y], linewidth=line_width)

    ax.set_xlim(x_left, x_right)
    ax.set_xlabel("Filtration index (pseudo-log)" if log_x else "Filtration index")
    if title:
        ax.set_title(title)
    ax.set_ylim(-0.6 * band_gap, max_dim * band_gap + 0.6 * band_gap)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)
    plt.show()
