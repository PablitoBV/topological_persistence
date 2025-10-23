from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from boundary import Boundary


@dataclass
class ReductionResult:
    """
    Result of the boundary matrix reduction.

    - lowest_row_of_col[j]: pivot row index for column j, or None if empty
    - birth_to_death: map from birth row -> death column
    """
    lowest_row_of_col: List[Optional[int]]
    birth_to_death: Dict[int, int]


@dataclass
class ReductionStats:
    """
    Simple counters used for performance tracking.
    """
    column_additions: int = 0
    pivots_finalized: int = 0


# ---------------------------------------------------------------------
# Greedy algorithm: process columns left-to-right and eliminate conflicts
# ---------------------------------------------------------------------
def reduce_greedy(boundary: Boundary) -> Tuple[ReductionResult, ReductionStats]:
    """
    Basic Gaussian elimination on Z2 without optimization.
    Each column is processed in order, adding previous columns
    when they share the same lowest one.
    """
    n = boundary.size
    lowest_row_of_col: List[Optional[int]] = [None] * n
    birth_to_death: Dict[int, int] = {}
    stats = ReductionStats()

    for col in range(n):
        while True:
            pivot_row = boundary.lowest_one(col)
            if pivot_row is None:
                # column is empty: it creates a new feature
                break

            existing_col = birth_to_death.get(pivot_row)
            if existing_col is None:
                # first time we see this pivot row: finalize
                lowest_row_of_col[col] = pivot_row
                birth_to_death[pivot_row] = col
                stats.pivots_finalized += 1
                break

            # otherwise, cancel the pivot by XORing the two columns
            boundary.add_column(col, existing_col)
            stats.column_additions += 1

    result = ReductionResult(lowest_row_of_col, dict(birth_to_death))
    return result, stats


# ---------------------------------------------------------------------
# Smart reduction: choose the sparsest column when multiple conflicts exist
# ---------------------------------------------------------------------
def reduce_smart(boundary: Boundary) -> Tuple[ReductionResult, ReductionStats]:
    """
    Optimized reduction. When several pivot conflicts occur, choose
    the column with the smallest number of nonzero entries to reduce fill-in.
    """
    n = boundary.size
    lowest_row_of_col: List[Optional[int]] = [None] * n
    birth_to_death: Dict[int, int] = {}
    stats = ReductionStats()

    for col in range(n):
        while True:
            support_rows = boundary.column_rows(col)
            if not support_rows:
                # no more 1s in this column
                break

            candidates: List[Tuple[int, int]] = []  # (nnz_count, col_index)
            for r in reversed(support_rows):
                k = birth_to_death.get(r)
                if k is not None:
                    nnz_k = len(boundary.column_rows(k))
                    candidates.append((nnz_k, k))

            if not candidates:
                # no conflicting pivot: finalize this column
                pivot_row = support_rows[-1]
                lowest_row_of_col[col] = pivot_row
                birth_to_death[pivot_row] = col
                stats.pivots_finalized += 1
                break

            # choose the sparsest column to minimize fill-in
            _, best_col = min(candidates, key=lambda t: (t[0], t[1]))
            boundary.add_column(col, best_col)
            stats.column_additions += 1

    result = ReductionResult(lowest_row_of_col, dict(birth_to_death))
    return result, stats
