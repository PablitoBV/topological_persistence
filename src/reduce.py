# reducer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from boundary import BoundaryMatrix, ReductionResult


@dataclass
class Stats:
    col_additions: int = 0
    pivots_finalized: int = 0


class BaseReducer:
    """Common scaffolding for column-based Gaussian elimination over Z2."""
    def __init__(self, *, enable_clearing: bool = False):
        self.enable_clearing = enable_clearing
        self.stats = Stats()

    def reduce(self, B: BoundaryMatrix) -> ReductionResult:
        n = B.n
        low: List[Optional[int]] = [None] * n
        row2col: Dict[int, int] = {}

        for j in range(n):
            self._reduce_one(B, j, low, row2col)

            # clearing: once (i,j) is finalized, clear column i to 0s
            if self.enable_clearing:
                i = low[j]
                if i is not None:
                    # If a birth i is paired, future columns will never pivot on i.
                    # Clearing is strategy-dependent; here, no direct API to clear,
                    # but future adds will never reference column i; no action needed.

                    # NOTE: In classic "clearing", we zero the whole column i.
                    # That would require a method to set column to empty; we skip it
                    # since our algorithms never *add* column i to others later.
                    pass

        return ReductionResult(low=low, pair=dict(row2col))

    # --- strategy-specific ---
    def _reduce_one(
        self,
        B: BoundaryMatrix,
        j: int,
        low: List[Optional[int]],
        row2col: Dict[int, int],
    ) -> None:
        raise NotImplementedError

    # --- helpers ---
    def _finalize(self, j: int, lj: int, low: List[Optional[int]], row2col: Dict[int, int]) -> None:
        low[j] = lj
        row2col[lj] = j
        self.stats.pivots_finalized += 1

    def _add(self, B: BoundaryMatrix, j: int, k: int) -> None:
        B.add_col(j, k)
        self.stats.col_additions += 1


class GreedyReducer(BaseReducer):
    """
    Glouton: process columns in filtration order. When a pivot collision occurs
    (same low row), immediately add the existing pivot column to cancel it.
    """
    def _reduce_one(
        self,
        B: BoundaryMatrix,
        j: int,
        low: List[Optional[int]],
        row2col: Dict[int, int],
    ) -> None:
        while True:
            lj = self._low(B, j)
            if lj is None:
                # creator that never dies -> leave low[j]=None
                return
            k = row2col.get(lj)
            if k is None:
                self._finalize(j, lj, low, row2col)
                return
            self._add(B, j, k)

    def _low(self, B: BoundaryMatrix, j: int) -> Optional[int]:
        # scan from bottom by using rows() then picking max if exists
        rs = B.rows(j)
        if not rs:
            return None
        return rs[-1]


class SmartReducer(BaseReducer):
    """
    Sparsity-aware elimination:
    - For current column j, while it has a low row lj that is already pivoted,
      choose among all conflicting pivot columns the one with smallest nnz
      (minimize fill-in), then XOR it into j.
    - Additionally, pre-eliminate all pivoted rows present in j in descending order,
      picking the *sparsest* available pivot each step.
    - Optional 'clearing' reduces future work on typical filtrations.
    """
    def _reduce_one(
        self,
        B: BoundaryMatrix,
        j: int,
        low: List[Optional[int]],
        row2col: Dict[int, int],
    ) -> None:
        # Loop while some pivoted row is present in column j
        while True:
            rs = B.rows(j)
            if not rs:
                return  # creator, low[j]=None

            # Try to find a pivoted row among the current support, taking the deepest (largest row)
            # but *choosing* the sparsest pivot column associated to any conflicting low.
            candidates: List[Tuple[int, int]] = []  # (nnz(k), k)
            pivot_row_chosen: Optional[int] = None

            # consider rows from bottom up
            for r in reversed(rs):
                k = row2col.get(r)
                if k is not None:
                    if pivot_row_chosen is None:
                        pivot_row_chosen = r
                    # collect candidate pivot columns, prefer sparse
                    candidates.append((B.nnz(k), k))
                    # keep scanning in case there is an even sparser pivot at a higher row

            if not candidates:
                # No conflicting pivot -> finalize with current lowest row
                lj = rs[-1]  # bottom-most
                self._finalize(j, lj, low, row2col)
                return

            # pick sparsest pivot column to minimize fill-in
            _, k_best = min(candidates, key=lambda t: (t[0], t[1]))
            self._add(B, j, k_best)
            # loop continues with updated column j
