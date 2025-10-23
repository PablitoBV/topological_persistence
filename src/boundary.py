from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from filtration import Filtration, Simplex


def _faces_of(vertices: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    Return all codimension-1 faces as vertex tuples.
    Each face is built by removing one position from the tuple.
    """
    n = len(vertices)
    if n == 0:
        return []
    return [tuple(vertices[j] for j in range(n) if j != i) for i in range(n)]


@dataclass
class Boundary:
    """
    Sparse Z2 boundary matrix attached to a sorted list of simplices.

    Storage:
      - simplices: filtration-sorted simplices
      - columns: dict col -> set of row indices where B[row, col] == 1

    Notes:
      - Coefficients are in Z2. Column addition is XOR of row index sets.
      - Rows and columns follow the same global order (the filtration order).
    """
    simplices: List[Simplex]
    columns: Dict[int, Set[int]]  # col j -> set of rows i with B[i, j] = 1

    @property
    def size(self) -> int:
        """Number of simplices (matrix is size x size)."""
        return len(self.simplices)

    @classmethod
    def from_filtration(cls, filtration: Filtration, *, strict: bool = True) -> Boundary:
        """
        Build the boundary from a filtration.
        If strict is True, raise when a face is missing from the file.
        """
        ordered = filtration.sort_by_filtration(in_place=False)
        index_of = {s.vert: i for i, s in enumerate(ordered)}
        cols: Dict[int, Set[int]] = {}

        for col, s in enumerate(ordered):
            if s.dim == 0:
                continue
            rows: Set[int] = set()
            for face in _faces_of(s.vert):
                row = index_of.get(face)
                if row is None:
                    if strict:
                        raise ValueError(f"missing face {face} for simplex {s.vert}")
                    # if not strict, skip missing faces
                    continue
                rows.add(row)
            if rows:
                cols[col] = rows

        return cls(simplices=ordered, columns=cols)

    # ---------------- reduction helpers (Z2) ----------------

    def lowest_one(self, col: int) -> Optional[int]:
        """
        Return the largest row index with a 1 in column col, or None if empty.
        """
        s = self.columns.get(col)
        return max(s) if s else None

    def add_column(self, dst_col: int, src_col: int) -> None:
        """
        dst_col <- dst_col XOR src_col (Z2 column addition).
        No-op if src == dst. Remove empty columns to keep storage tight.
        """
        if dst_col == src_col:
            return

        s_dst = self.columns.get(dst_col)
        s_src = self.columns.get(src_col)

        if not s_src:
            # adding zero does nothing
            return

        if not s_dst:
            # copy to avoid aliasing
            self.columns[dst_col] = set(s_src)
            return

        # symmetric difference is XOR on sets of row indices
        s_dst ^= s_src
        if s_dst:
            self.columns[dst_col] = s_dst
        else:
            # drop empty column
            self.columns.pop(dst_col, None)

    def column_rows(self, col: int) -> List[int]:
        """
        Return sorted row indices where column col has ones.
        """
        s = self.columns.get(col)
        return sorted(s) if s else []

    def copy(self) -> Boundary:
        """
        Deep copy of the sparse structure (simplices list is shallow-copied).
        """
        return Boundary(
            simplices=list(self.simplices),
            columns={c: set(rows) for c, rows in self.columns.items()},
        )

    # ---------------- debug / inspection ----------------

    def print_dense(self) -> None:
        """
        Print the matrix as 0/1 rows without materializing a full dense array.
        Useful for small cases and sanity checks.
        """
        n = self.size
        for i in range(n):
            bits = ("1" if i in self.columns.get(j, ()) else "0" for j in range(n))
            print(" ".join(bits))

    def __repr__(self) -> str:
        n = self.size
        nnz = sum(len(r) for r in self.columns.values())
        return f"Boundary(n={n}, nnz={nnz}, cols={len(self.columns)})"
