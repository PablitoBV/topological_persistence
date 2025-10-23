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
    use_f_values: bool = True,   # True: x = f(σ); False: x = indices i/j
    outfile: Optional[str] = None,
    title: Optional[str] = None,
    log: bool = False,           # True => log-auto (log10 if x>0 else symlog)
    min_length: float = 0.0,
    relative: bool = False,
) -> None:
    """
    Plot a barcode from a reduction. Bands ordered top→bottom: highest dim ... → H0.
    - x uses either filtration values f(σ) or sorted indices, per `use_f_values`.
    - if log=True: auto-pick log10 (all finite x>0) else symlog (supports <=0).
    - infinite bars are drawn to the current RIGHT EDGE of the axis (after scaling).
    - finite bars can be filtered by visible length (absolute or relative).
    """
    # ---- Build intervals in chosen x domain (never infer from indices when use_f_values=True) ----
    intervals: List[Interval] = []
    if use_f_values:
        for i, j in birth_to_death.items():
            intervals.append((simplices[i].dim, float(simplices[i].val), float(simplices[j].val)))
        used_births = set(birth_to_death.keys())
        for j, l in enumerate(lowest_row_of_col):
            if l is None and j not in used_births:
                intervals.append((simplices[j].dim, float(simplices[j].val), "inf"))
    else:
        for i, j in birth_to_death.items():
            intervals.append((simplices[i].dim, float(i), float(j)))
        used_births = set(birth_to_death.keys())
        for j, l in enumerate(lowest_row_of_col):
            if l is None and j not in used_births:
                intervals.append((simplices[j].dim, float(j), "inf"))

    # Basic sanity
    for k, b, d in intervals:
        if d != "inf" and float(d) < float(b):
            raise ValueError(f"Death < birth on x-axis: (dim={k}, b={b}, d={d})")

    # ---- Normalize + collect finite x ----
    cleaned: List[Tuple[int, float, Optional[float]]] = []
    finite_x: List[float] = []
    for k, b, d in intervals:
        b = float(b)
        d_val = None if d == "inf" else float(d)
        cleaned.append((int(k), b, d_val))
        finite_x.append(b)
        if d_val is not None:
            finite_x.append(d_val)

    # ---- Choose axis transform (log auto) ----
    use_log10 = False
    use_symlog = False
    if log:
        if finite_x and min(finite_x) > 0.0:
            use_log10 = True
        else:
            use_symlog = True
    # Helper to measure visible length (for filtering) in the chosen domain
    def to_display_x(x: float) -> float:
        if not log:
            return x
        if use_log10:
            # log10 domain length; x must be > 0
            return math.log10(x)
        # symlog approximation for filtering (axis uses true symlog)
        return math.copysign(math.log10(abs(x) + 1.0), x)

    # ---- Filtering (keep ∞) ----
    if min_length > 0.0 and finite_x:
        vals_disp = [to_display_x(v) for v in finite_x if (not use_log10 or v > 0)]
        if not vals_disp:
            # if all values invalid for log10 measure (e.g., <=0), skip filtering
            thr = 0.0
        else:
            span_disp = max(1e-12, max(vals_disp) - min(vals_disp))
            thr = (min_length * span_disp) if relative else min_length
    else:
        thr = 0.0

    kept: List[Tuple[int, float, Optional[float]]] = []
    for k, b, d in cleaned:
        if d is None:
            kept.append((k, b, d))
        else:
            if (not use_log10) or (b > 0 and d > 0):
                if (to_display_x(d) - to_display_x(b)) >= thr:
                    kept.append((k, b, d))
            else:
                # log10 length undefined for non-positive; keep bar (or pre-shift outside if desired)
                kept.append((k, b, d))

    if not kept:
        fig, ax = plt.subplots(figsize=(8, 2), dpi=160)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "All bars filtered out", ha="center", va="center", transform=ax.transAxes)
        if outfile: fig.savefig(outfile, bbox_inches="tight", dpi=160)
        plt.show()
        return

    # ---- Group by dim; top→bottom = highest→lowest ----
    dims = sorted({k for k, _, _ in kept}, reverse=True)
    per_dim = {k: [] for k in dims}
    for k, b, d in kept:
        per_dim[k].append((b, d))
    for k in dims:
        per_dim[k].sort(key=lambda bd: (bd[0], math.inf if bd[1] is None else bd[1]))

    # ---- X limits from chosen x-values (not indices unless use_f_values=False) ----
    all_x = [x for k in dims for (b, d) in per_dim[k] for x in (b, d if d is not None else b)]
    xmin, xmax = min(all_x), max(all_x)
    x_left, x_right = xmin, xmax

    # ---- Vertical layout: no overlaps ----
    band_gap = 1.2
    band_half = 0.4 * band_gap
    fig_h = max(2.2, len(dims) * (band_gap * 0.9) + 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=180)

    for i in range(1, len(dims)):
        ax.axhline(i * band_gap - 0.5 * band_gap, linewidth=0.5, alpha=0.4)
    ax.set_yticks([i * band_gap for i in range(len(dims))])
    ax.set_yticklabels([f"H {k}" for k in dims])  # H(max) on top, H0 bottom

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
                # draw to data right limit; arrow will be snapped to visible right edge after scaling
                ax.plot([b, x_right], [y, y], linewidth=lw, color="darkblue")
            else:
                ax.plot([b, d], [y, y], linewidth=lw, color="darkblue")

    # ---- Set x-scale (real log), then compute visible right boundary and draw arrows to it ----
    ax.set_xlim(x_left, x_right)
    if log:
        if use_log10:
            ax.set_xscale("log")
            ax.set_xlabel(("f(σ)" if use_f_values else "index") + " (log10)")
        else:
            # tiny linthresh so 0<f<1 est bien en "log" (pas en zone linéaire)
            ax.set_xscale("symlog", linthresh=1e-9, linscale=1.0)
            ax.set_xlabel(("f(σ)" if use_f_values else "index") + " (symlog)")
    else:
        ax.set_xlabel("f(σ)" if use_f_values else "index")
    if title:
        title += " [log scale]" if log else ""
        title += f" [min length > {min_length}{' (rel)' if relative else ''}]" if min_length > 0 else ""
        ax.set_title(title)

    # snap INF arrows to the visible right boundary (AFTER scale): this makes ∞ == max displayed x
    xr_visible = ax.get_xlim()[1]  # data coordinate at the right edge (log/ symlog aware)
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
                # small backstep in DATA units to make arrowhead visible within the frame
                ax.annotate(
                    "",
                    xy=(xr_visible, y),
                    xytext=(xr_visible - 0.02 * (x_right - x_left), y),
                    arrowprops=dict(arrowstyle="->", lw=lw),
                )

    ax.set_ylim(-0.6 * band_gap, (len(dims) - 1) * band_gap + 0.6 * band_gap)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, bbox_inches="tight", dpi=180)
    plt.show()