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

def plot_barcode_reduction(
    simplices: List[Simplex],
    lowest_row_of_col: List[Optional[int]],
    birth_to_death: Dict[int, int],
    *,
    use_f_values: bool = True,   # True => x = f(σ); False => x = indices (i/j)
    outfile: Optional[str] = None,
    title: Optional[str] = None,
    log: bool = False,           # True => xscale='symlog' (handles <=0)
    min_length: float = 0.0,     # filter finite bars shorter than this (0 = off)
    relative: bool = False,      # if True, threshold is a fraction of visible span
) -> None:
    """
    Plot a persistence barcode from a reduced boundary matrix.

    Inputs mirror write_barcode(...):
      - simplices: sorted simplices (each has .dim and .val = f(σ))
      - lowest_row_of_col: pivot row per column (or None)
      - birth_to_death: map birth-row i -> death-column j

    X-axis choice:
      - use_f_values=True  → x = filtration values f(σ) for births/deaths
      - use_f_values=False → x = indices (i/j) in the sorted filtration

    Display:
      - Highest dimension on top, H0 at the bottom.
      - Optional symlog x-scale (true log that supports negatives).
      - Optional min-length filtering in the display domain (linear or symlog).
    """
    # Build intervals (birth, death) in chosen x domain
    intervals: List[Interval] = []
    if use_f_values:
        # paired
        for i, j in birth_to_death.items():
            intervals.append((simplices[i].dim, float(simplices[i].val), float(simplices[j].val)))
        # essential (no double count)
        used_births = set(birth_to_death.keys())
        for j, l in enumerate(lowest_row_of_col):
            if l is None and j not in used_births:
                intervals.append((simplices[j].dim, float(simplices[j].val), "inf"))
    else:
        # indices as x
        for i, j in birth_to_death.items():
            intervals.append((simplices[i].dim, float(i), float(j)))
        used_births = set(birth_to_death.keys())
        for j, l in enumerate(lowest_row_of_col):
            if l is None and j not in used_births:
                intervals.append((simplices[j].dim, float(j), "inf"))

    # Sanity: no negative lengths for finite bars
    for k, b, d in intervals:
        if d != "inf" and float(d) < float(b):
            raise ValueError(f"Death < birth on x-axis: (dim={k}, b={b}, d={d})")

    # Normalize and compute filtering threshold in display domain
    def disp_x(x: float) -> float:
        if not log:
            return x
        return math.copysign(math.log10(abs(x) + 1.0), x)  # for filtering only; axis uses true symlog

    finite_x: List[float] = []
    cleaned: List[Tuple[int, float, Optional[float]]] = []
    for k, b, d in intervals:
        k = int(k); b = float(b)
        d_val = None if d == "inf" else float(d)
        cleaned.append((k, b, d_val))
        finite_x.append(b)
        if d_val is not None:
            finite_x.append(d_val)

    if min_length > 0.0 and finite_x:
        xs = [disp_x(v) for v in finite_x]
        span = max(1e-12, max(xs) - min(xs))
        thr = (min_length * span) if relative else min_length
    else:
        thr = 0.0

    kept: List[Tuple[int, float, Optional[float]]] = []
    for k, b, d in cleaned:
        if d is None:
            kept.append((k, b, d))
        else:
            if (disp_x(d) - disp_x(b)) >= thr:
                kept.append((k, b, d))

    if not kept:
        fig, ax = plt.subplots(figsize=(8, 2), dpi=160)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "All bars filtered out", ha="center", va="center", transform=ax.transAxes)
        if outfile: fig.savefig(outfile, bbox_inches="tight", dpi=160)
        plt.show()
        return

    # Group by dimension and order vertically: max→…→0 top→bottom
    dims = sorted({k for k, _, _ in kept}, reverse=True)
    per_dim = {k: [] for k in dims}
    for k, b, d in kept:
        per_dim[k].append((b, d))
    for k in dims:
        per_dim[k].sort(key=lambda bd: (bd[0], math.inf if bd[1] is None else bd[1]))

    # X-limits from chosen x-values (never from indices unless use_f_values=False)
    all_x = [x for k in dims for (b, d) in per_dim[k] for x in (b, d if d is not None else b)]
    xmin, xmax = min(all_x), max(all_x)
    xspan = max(1e-12, xmax - xmin)
    xpad = 0.05 * xspan
    x_left, x_right = xmin - xpad, xmax + xpad

    # Vertical layout without overlap
    band_gap = 1.2
    band_half = 0.4 * band_gap
    fig_h = max(2.2, len(dims) * (band_gap * 0.9) + 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=180)

    for i in range(1, len(dims)):
        ax.axhline(i * band_gap - 0.5 * band_gap, linewidth=0.5, alpha=0.4)
    ax.set_yticks([i * band_gap for i in range(len(dims))])
    ax.set_yticklabels([f"H {k}" for k in dims])  # top: max dim

    lw = 2.0
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
                ax.plot([b, x_right], [y, y], linewidth=lw, color='darkblue')
            else:
                ax.plot([b, d], [y, y], linewidth=lw, color='darkblue')

    ax.set_xlim(x_left, x_right)
    if log:
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
        ax.set_xlabel(("f(σ)" if use_f_values else "index") + " (symlog)")
    else:
        ax.set_xlabel("f(σ)" if use_f_values else "index")
    if title:
        ax.set_title(title)

    # Arrowheads for infinite bars at the visible right boundary
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

