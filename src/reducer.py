# reducer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from boundary import Boundary

@dataclass
class ReductionResult:
    lowest_row_of_col: List[Optional[int]]     # lowest_row_of_col[j] = pivot row, else None
    birth_to_death: Dict[int, int]             # birth row i -> death column j

@dataclass
class ReductionStats:
    column_additions: int = 0
    pivots_finalized: int = 0

def reduce_greedy(boundary: Boundary) -> Tuple[ReductionResult, ReductionStats]:
    n = boundary.size
    lowest_row_of_col: List[Optional[int]] = [None] * n
    birth_to_death: Dict[int, int] = {}
    stats = ReductionStats()

    for col in range(n):
        while True:
            pivot_row = boundary.lowest_one(col)
            if pivot_row is None:
                break
            existing_col = birth_to_death.get(pivot_row)
            if existing_col is None:
                lowest_row_of_col[col] = pivot_row
                birth_to_death[pivot_row] = col
                stats.pivots_finalized += 1
                break
            boundary.add_column(col, existing_col)
            stats.column_additions += 1

    return ReductionResult(lowest_row_of_col, dict(birth_to_death)), stats

def reduce_smart(boundary: Boundary) -> Tuple[ReductionResult, ReductionStats]:
    n = boundary.size
    lowest_row_of_col: List[Optional[int]] = [None] * n
    birth_to_death: Dict[int, int] = {}
    stats = ReductionStats()

    for col in range(n):
        while True:
            support_rows = boundary.column_rows(col)
            if not support_rows:
                break
            candidates: List[Tuple[int, int]] = []  # (nnz_of_candidate, candidate_col)
            for r in reversed(support_rows):
                k = birth_to_death.get(r)
                if k is not None:
                    nnz_k = len(boundary.column_rows(k))
                    candidates.append((nnz_k, k))
            if not candidates:
                pivot_row = support_rows[-1]
                lowest_row_of_col[col] = pivot_row
                birth_to_death[pivot_row] = col
                stats.pivots_finalized += 1
                break
            _, best_col = min(candidates, key=lambda t: (t[0], t[1]))
            boundary.add_column(col, best_col)
            stats.column_additions += 1

    return ReductionResult(lowest_row_of_col, dict(birth_to_death)), stats
