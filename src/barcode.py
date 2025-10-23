from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union

import math
import matplotlib.pyplot as plt # type: ignore

from matplotlib.collections import LineCollection # type: ignore
from filtration import Simplex

# A barcode interval is (dimension, birth_x, death_x or "inf")
Interval = Tuple[int, float, Union[float, str]]


# ----------------------------------------------------------------------
# Text I/O
# ----------------------------------------------------------------------

def write_barcode(
    simplices: List[Simplex],
    lowest_row_of_col: List[Optional[int]],
    birth_to_death: Dict[int, int],
    outfile: str,
) -> None:
    """
    Write the barcode intervals as lines "dim birth death".
    Sorted by (dim, birth, death) with "inf" sorted last.
    """
    lines: List[str] = []

    # Paired intervals
    for birth_row, death_col in sorted(birth_to_death.items()):
        k = simplices[birth_row].dim
        b = simplices[birth_row].val
        d = simplices[death_col].val
        lines.append(f"{k} {b} {d}")

    # Essential intervals (infinite)
    used_births = set(birth_to_death.keys())
    for col, pivot_row in enumerate(lowest_row_of_col):
        if pivot_row is None and col not in used_births:
            k = simplices[col].dim
            b = simplices[col].val
            lines.append(f"{k} {b} inf")

    def sort_key(line: str) -> Tuple[int, float, float]:
        dim_str, birth_str, death_str = line.split()
        dim = int(dim_str)
        birth = float(birth_str)
        death = float("inf") if death_str == "inf" else float(death_str)
        return (dim, birth, death)

    lines.sort(key=sort_key)

    with open(outfile, "w", encoding="utf-8") as f:
        for idx, line in enumerate(lines):
            f.write(line + "\n" if idx < len(lines) - 1 else line)


# ----------------------------------------------------------------------
# Interval helpers
# ----------------------------------------------------------------------

def build_intervals(
    simplices: List[Simplex],
    lowest_row_of_col: List[Optional[int]],
    birth_to_death: Dict[int, int],
) -> List[Interval]:
    """
    Build intervals from a reduction result.
    X values are filtration values (birth f, death f), "inf" for essentials.
    Sorted by (dim, birth, death) with "inf" last.
    """
    out: List[Interval] = []

    for i, j in birth_to_death.items():
        out.append((simplices[i].dim, simplices[i].val, simplices[j].val))

    used_births = set(birth_to_death.keys())
    for j, low in enumerate(lowest_row_of_col):
        if low is None and j not in used_births:
            out.append((simplices[j].dim, simplices[j].val, "inf"))

    out.sort(key=lambda t: (t[0], t[1], float("inf") if t[2] == "inf" else float(t[2])))
    return out


def filter_by_length(
    intervals: List[Interval],
    *,
    min_length: float = 0.0,
    relative: bool = False,
) -> List[Interval]:
    """
    Remove finite bars shorter than a threshold.
    If relative is True, the threshold is a fraction of the global span.
    """
    if not intervals or min_length <= 0:
        return list(intervals)

    finite_vals = [float(x) for _, b, d in intervals for x in (b, d if d != "inf" else b)]
    if not finite_vals:
        return list(intervals)

    x_min, x_max = min(finite_vals), max(finite_vals)
    span = max(1e-12, x_max - x_min)
    thr = min_length * span if relative else min_length

    out: List[Interval] = []
    for k, b, d in intervals:
        if d == "inf" or (float(d) - float(b)) >= thr:
            out.append((k, b, d))
    return out


def betti_from_intervals(intervals: List[Interval]) -> Dict[int, int]:
    """
    Count infinite bars per dimension.
    """
    betti: Dict[int, int] = {}
    for k, _b, d in intervals:
        if d == "inf":
            betti[k] = betti.get(k, 0) + 1
    return betti


def guess_space(betti: Dict[int, int]) -> str:
    """
    Heuristic space guess from low-dimensional Betti numbers.
    """
    b0 = betti.get(0, 0)
    b1 = betti.get(1, 0)
    b2 = betti.get(2, 0)

    if b0 == 0:
        return "empty set"
    if b0 > 1 and b1 == 0 and b2 == 0:
        return f"{b0} disjoint contractible components"
    if b0 == 1 and b1 == 0 and b2 == 0:
        return "contractible"
    if b0 == 1 and b1 == 1 and b2 == 0:
        return "circle-like (S1)"
    if b0 == 1 and b1 == 0 and b2 == 1:
        return "sphere-like (S2)"
    if b0 == 1 and b1 == 2 and b2 == 1:
        return "torus (T2)"
    if b0 == 1 and b2 == 1 and b1 % 2 == 0 and b1 >= 4:
        g = b1 // 2
        return f"closed orientable surface of genus {g}"
    if b0 == 1 and b1 > 0 and b2 == 0:
        return f"one component with {b1} independent 1-cycles"
    return f"undetermined: b0={b0}, b1={b1}, b2={b2}"


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


# -------------------------- helpers --------------------------

def _build_intervals_raw(
    simplices: List[Simplex],
    lowest_row_of_col: List[Optional[int]],
    birth_to_death: Dict[int, int],
    use_f_values: bool,
) -> List[Tuple[int, float, Union[float, str]]]:
    """
    Build (dim, x_birth, x_death|'inf') from the reduction.
    x is either filtration value or index depending on use_f_values.
    """
    out: List[Tuple[int, float, Union[float, str]]] = []

    if use_f_values:
        for i, j in birth_to_death.items():
            out.append((simplices[i].dim, float(simplices[i].val), float(simplices[j].val)))
        used = set(birth_to_death.keys())
        for j, low in enumerate(lowest_row_of_col):
            if low is None and j not in used:
                out.append((simplices[j].dim, float(simplices[j].val), "inf"))
    else:
        for i, j in birth_to_death.items():
            out.append((simplices[i].dim, float(i), float(j)))
        used = set(birth_to_death.keys())
        for j, low in enumerate(lowest_row_of_col):
            if low is None and j not in used:
                out.append((simplices[j].dim, float(j), "inf"))

    for k, b, d in out:
        if d != "inf" and float(d) < float(b):
            raise ValueError(f"death < birth: dim={k}, b={b}, d={d}")

    return out


def _make_x_transformers(x_mode: str):
    """
    Return (to_plot, to_filter) callables.

    to_plot(x): x used to plot coordinates and set xlim in the displayed domain.
    to_filter(x): x used to measure lengths for filtering in that same domain.
    """
    mode = x_mode.lower()
    if mode not in {"linear", "symlog", "log_values"}:
        raise ValueError("x_mode must be 'linear', 'symlog', or 'log_values'.")

    if mode == "linear":
        return (lambda x: x, lambda x: x)

    if mode == "symlog":
        # Plot with raw values; axis will be symlog.
        # For filtering we use a simple monotone map.
        def to_filter(x: float) -> float:
            return math.copysign(math.log10(abs(x) + 1.0), x)
        return (lambda x: x, to_filter)

    # mode == "log_values": plot log10(x) on a linear axis
    def to_plot(x: float) -> float:
        if x < 0:
            raise ValueError("log_values requires all finite x > 0.")
        if x == 0:
            return 1e-12
        return math.log10(x)
    return (to_plot, to_plot)


def _stack_y_positions(n: int, center: float, band_half: float) -> List[float]:
    """
    Evenly space n y-positions inside [center - band_half, center + band_half].
    """
    if n <= 1:
        return [center]
    lo, hi = center - band_half, center + band_half
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


# ----------------------------- plotting -----------------------------

def plot_barcode_reduction(
    simplices: List[Simplex],
    lowest_row_of_col: List[Optional[int]],
    birth_to_death: Dict[int, int],
    *,
    use_f_values: bool = True,
    x_mode: str = "linear",      # "linear" | "symlog" | "log_values"
    min_length: float = 0.0,     # 0 disables
    relative: bool = False,      # if True, threshold is fraction of span
    outfile: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot barcode from a reduced boundary.

    Layout:
      - Highest dimension at top, H0 at bottom.
      - X is either filtration values or indices.
      - X scaling:
          * linear: raw values
          * symlog: real symmetric log axis (works with x <= 0)
          * log_values: plot log10(x) on a linear axis (requires x > 0)
      - Infinite bars end at the visible right bound in the displayed domain.
      - Filtering happens in that same displayed domain.
    """
    # Build raw intervals in chosen x domain
    raw = _build_intervals_raw(simplices, lowest_row_of_col, birth_to_death, use_f_values)

    # Prepare x transforms
    to_plot, to_filter = _make_x_transformers(x_mode)

    # Convert to displayed domain for plotting and limits
    disp_intervals: List[Tuple[int, float, Optional[float]]] = []
    disp_samples: List[float] = []
    dims_set = set()

    for k, b, d in raw:
        xb = to_plot(float(b))
        if d == "inf":
            disp_intervals.append((int(k), xb, None))
            disp_samples.append(xb)
        else:
            xd = to_plot(float(d))
            disp_intervals.append((int(k), xb, xd))
            disp_samples.extend([xb, xd])
        dims_set.add(int(k))

    if not disp_intervals:
        fig, ax = plt.subplots(figsize=(8, 2), dpi=160)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No intervals", ha="center", va="center", transform=ax.transAxes)
        if outfile:
            fig.savefig(outfile, bbox_inches="tight", dpi=160)
        plt.show()
        return

    # Compute filter threshold in displayed domain
    if min_length > 0.0 and disp_samples:
        span = max(1e-12, max(disp_samples) - min(disp_samples))
        thr = (min_length * span) if relative else min_length
    else:
        thr = 0.0

    # Keep finite bars with length >= thr; keep all infinite
    kept: List[Tuple[int, float, Optional[float]]] = []
    for k, xb, xd in disp_intervals:
        if xd is None:
            kept.append((k, xb, None))
        else:
            left, right = (xb, xd) if xb <= xd else (xd, xb)
            # For symlog the visible distance is measured by to_filter
            length = (to_filter(right) - to_filter(left)) if x_mode == "symlog" else (right - left)
            if length >= thr:
                kept.append((k, left, right))

    if not kept:
        fig, ax = plt.subplots(figsize=(8, 2), dpi=160)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "All bars filtered out", ha="center", va="center", transform=ax.transAxes)
        if outfile:
            fig.savefig(outfile, bbox_inches="tight", dpi=160)
        plt.show()
        return

    # Group by dimension; draw top to bottom from highest to lowest
    dims = sorted({k for k, _, _ in kept}, reverse=True)
    per_dim: Dict[int, List[Tuple[float, Optional[float]]]] = {k: [] for k in dims}
    for k, xb, xd in kept:
        per_dim[k].append((xb, xd))
    for k in dims:
        per_dim[k].sort(key=lambda bd: (bd[0], float("inf") if bd[1] is None else bd[1]))

    # X limits in displayed domain (this fixes the "inf uses max f instead of max log f" bug)
    xs = [x for k in dims for (b, d) in per_dim[k] for x in (b, d if d is not None else b)]
    xmin, xmax = min(xs), max(xs)
    xspan = max(1e-12, xmax - xmin)
    xpad = 0.05 * xspan
    x_left, x_right = xmin - xpad, xmax + xpad

    # Figure and axes
    band_gap = 1.2
    band_half = 0.4 * band_gap
    fig_h = max(2.2, len(dims) * (band_gap * 0.9) + 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=180)

    # Y layout and labels
    for i in range(1, len(dims)):
        ax.axhline(i * band_gap - 0.5 * band_gap, linewidth=0.5, alpha=0.3)
    ax.set_yticks([i * band_gap for i in range(len(dims))])
    ax.set_yticklabels([f"H {k}" for k in dims])

    # Build line segments per band
    finite_segs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    inf_segs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    x_right = to_plot(x_right)
    
    for band_idx, k in enumerate(dims):
        base = band_idx * band_gap
        bars = per_dim[k]
        # reverse for nicer stacking (latest on top inside the band)
        bars_rev = list(reversed(bars))
        ys = _stack_y_positions(len(bars_rev), base, band_half)
        for y, (b, d) in zip(ys, bars_rev):
            if d is None:
                inf_segs.append(((b, y), (x_right, y)))
            else:
                finite_segs.append(((b, y), (d, y)))

    # Draw in batches
    if finite_segs:
        ax.add_collection(LineCollection(finite_segs, linewidths=2.0, colors='darkblue'))
    if inf_segs:
        ax.add_collection(LineCollection(inf_segs, linewidths=2.0))
        # Arrowheads at the visible right bound of the displayed domain
        for (_, y0), _ in inf_segs:
            ax.annotate(
                "",
                xy=(x_right, y0),
                xytext=(x_right - 0.02 * (x_right - x_left), y0),
                arrowprops=dict(arrowstyle="->", lw=2.0),
            )

    # X axis config
    ax.set_xlim(x_left, x_right)
    if x_mode == "symlog":
        # Values were not pre-transformed; axis does the transform.
        ax.set_xscale("symlog", linthresh=1e-9, linscale=1.0)
        xlabel = "f(.)" if use_f_values else "index"
        xlabel += " (symlog)"
    elif x_mode == "log_values":
        # Values already mapped to log10; keep a linear axis.
        xlabel = "log10 f(.)" if use_f_values else "log10 index"
    else:
        xlabel = "f(.)" if use_f_values else "index"
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)

    # Final layout
    ax.set_ylim(-0.6 * band_gap, (len(dims) - 1) * band_gap + 0.6 * band_gap)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, bbox_inches="tight", dpi=180)
    plt.show()
