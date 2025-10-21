# boundary.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Iterable
from filtration import Filtration, Simplex


# ------------ small utilities (pure, stateless) ------------ #

def filtration_order_key(s: Simplex) -> Tuple[float, int, Tuple[int, ...]]:
    return (s.val, s.dim, s.vert)

def faces_of(simplex: Simplex) -> List[Tuple[int, ...]]:
    if simplex.dim <= 0:
        return []
    v = list(simplex.vert)
    return [tuple(v[j] for j in range(len(v)) if j != i) for i in range(len(v))]

def index_by_vertices(simplices: List[Simplex]) -> Dict[Tuple[int, ...], int]:
    return {s.vert: i for i, s in enumerate(simplices)}


# ------------ base class interface ------------ #

@dataclass
class ReductionResult:
    low: List[Optional[int]]           # low[j] = pivot row of col j (or None)
    pair: Dict[int, int]               # birth row i -> death column j

class BoundaryMatrix:
    """Interface for a Z2 boundary matrix tied to a Filtration."""
    def __init__(self, filtration: Filtration, *, strict: bool = False):
        self._filtration = filtration
        self._simplices: List[Simplex] = filtration.sort_by_filtration(in_place=False)
        self._n = len(self._simplices)
        self._strict = strict
        self._reduction: Optional[ReductionResult] = None

    @property
    def simplices(self) -> List[Simplex]:
        return self._simplices

    @property
    def n(self) -> int:
        return self._n
    
    def nnz(self, j: int) -> int:
        """Number of 1s in column j."""
        raise NotImplementedError

    def rows(self, j: int) -> List[int]:
        """Sorted list of row indices (ascending) where column j has 1s."""
        raise NotImplementedError

    # --- virtuals to implement ---
    def low(self, j: int) -> Optional[int]:
        raise NotImplementedError

    def add_col(self, j: int, k: int) -> None:
        raise NotImplementedError

    # --- concrete: left-to-right column reduction over Z2 ---
    def reduce(self) -> ReductionResult:
        low: List[Optional[int]] = [None] * self._n
        row2col: Dict[int, int] = {}
        for j in range(self._n):
            while True:
                lj = self.low(j)
                if lj is None:
                    break
                k = row2col.get(lj)
                if k is None:
                    low[j] = lj
                    row2col[lj] = j
                    break
                self.add_col(j, k)
        self._reduction = ReductionResult(low=low, pair=dict(row2col))
        return self._reduction

    # --- derived outputs ---
    def pairing(self) -> ReductionResult:
        return self._reduction if self._reduction is not None else self.reduce()

    def barcode(self, outfile: str) -> None:
        rr = self.pairing()
        S = self._simplices
        lines: List[str] = []
        for i, j in sorted(rr.pair.items()):
            k = S[i].dim
            b = S[i].val
            d = S[j].val
            lines.append(f"{k} {b} {d}")
        for j, l in enumerate(rr.low):
            if l is None:
                k = S[j].dim
                b = S[j].val
                lines.append(f"{k} {b} inf")
        def key_fn(s: str) -> Tuple[int, float, float]:
            ks, bs, ds = s.split()
            return (int(ks), float(bs), float('inf') if ds == 'inf' else float(ds))
        lines.sort(key=key_fn)
        with open(outfile, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")


# ------------ dense backend ------------ #

class BoundaryDense(BoundaryMatrix):
    """Binary n×n matrix B as list-of-lists."""
    def __init__(self, filtration: Filtration, *, strict: bool = False):
        super().__init__(filtration, strict=strict)
        S = self._simplices
        n = self._n
        self._B: List[List[int]] = [[0] * n for _ in range(n)]
        by_vert = index_by_vertices(S)

        for j, s in enumerate(S):
            if s.dim == 0:
                continue
            for face in faces_of(s):
                i = by_vert.get(face)
                if i is None:
                    if self._strict:
                        raise ValueError(f"Missing face {face} (of {s.vert})")
                    continue
                self._B[i][j] = 1

    def nnz(self, j: int) -> int:
        c = 0
        B = self._B
        for i in range(self._n):
            if B[i][j] == 1:
                c += 1
        return c

    def rows(self, j: int) -> List[int]:
        B = self._B
        return [i for i in range(self._n) if B[i][j] == 1]

    def low(self, j: int) -> Optional[int]:
        col = self._B
        for i in range(self._n - 1, -1, -1):
            if col[i][j] == 1:
                return i
        return None

    def add_col(self, j: int, k: int) -> None:
        B = self._B
        for i in range(self._n):
            B[i][j] ^= B[i][k]

    # optional inspection
    def print(self) -> None:
        for r in self._B:
            print(" ".join(str(x) for x in r))


# ------------ sparse backend ------------ #

class BoundarySparse(BoundaryMatrix):
    """Column dictionary: col j -> set of row indices i with B[i,j]=1."""
    def __init__(self, filtration: Filtration, *, strict: bool = False):
        super().__init__(filtration, strict=strict)
        S = self._simplices
        by_vert = index_by_vertices(S)
        Bc: Dict[int, Set[int]] = {}

        for j, s in enumerate(S):
            if s.dim == 0:
                continue
            rows: Set[int] = set()
            for face in faces_of(s):
                i = by_vert.get(face)
                if i is None:
                    if self._strict:
                        raise ValueError(f"Missing face {face} (of {s.vert})")
                    continue
                rows.add(i)
            if rows:
                Bc[j] = rows

        self._Bc: Dict[int, Set[int]] = Bc

    def nnz(self, j: int) -> int:
        return len(self._Bc.get(j, ()))

    def rows(self, j: int) -> List[int]:
        s = self._Bc.get(j)
        if not s:
            return []
        return sorted(s)  # ascending; reducers will reverse when needed

    def low(self, j: int) -> Optional[int]:
        s = self._Bc.get(j)
        if not s:
            return None
        return max(s)

    def add_col(self, j: int, k: int) -> None:
        if j == k:
            return
        sj = self._Bc.get(j, set())
        sk = self._Bc.get(k, set())
        if not sj:
            if sk:
                self._Bc[j] = set(sk)
            return
        sj ^= sk  # symmetric difference = XOR
        if sj:
            self._Bc[j] = sj
        elif j in self._Bc:
            del self._Bc[j]

    # optional inspection
    def print(self) -> None:
        for j in sorted(self._Bc):
            print(f"col {j}: rows {sorted(self._Bc[j])}")


# ------------ factory ------------ #

def make_boundary(filtration: Filtration, *, mode: str = "sparse", strict: bool = False) -> BoundaryMatrix:
    if mode == "dense":
        return BoundaryDense(filtration, strict=strict)
    if mode == "sparse":
        return BoundarySparse(filtration, strict=strict)
    raise ValueError("mode must be 'dense' or 'sparse'")
