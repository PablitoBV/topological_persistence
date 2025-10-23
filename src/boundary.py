# boundary.py (sparse: store only ones)
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from filtration import Filtration, Simplex

def _faces(verts: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    return [tuple(verts[j] for j in range(len(verts)) if j != i) for i in range(len(verts))]

@dataclass
class Boundary:
    simplices: List[Simplex]
    columns: Dict[int, Set[int]]  # col j -> set of row indices i where B[i,j] == 1

    @property
    def size(self) -> int:
        return len(self.simplices)

    @classmethod
    def from_filtration(cls, filtration: Filtration, *, strict: bool = True) -> "Boundary":
        ordered = filtration.sort_by_filtration(in_place=False)
        index_of = {s.vert: i for i, s in enumerate(ordered)}
        cols: Dict[int, Set[int]] = {}

        for col, simplex in enumerate(ordered):
            if simplex.dim == 0:
                continue
            rows: Set[int] = set()
            for face in _faces(simplex.vert):
                row = index_of.get(face)
                if row is None:
                    if strict:
                        raise ValueError(f"Missing face {face} of {simplex.vert}")
                    continue
                rows.add(row)
            if rows:
                cols[col] = rows
        return cls(ordered, cols)

    # Impression dense (pour debug) sans matérialiser toute la matrice
    def print(self) -> None:
        n = self.size
        for i in range(n):
            line_bits = []
            for j in range(n):
                line_bits.append("1" if i in self.columns.get(j, ()) else "0")
            print(" ".join(line_bits))

    # helpers pour l'élimination sur Z2
    def lowest_one(self, col: int) -> Optional[int]:
        s = self.columns.get(col)
        return max(s) if s else None

    def add_column(self, dst_col: int, src_col: int) -> None:
        if dst_col == src_col:
            return
        s_dst = self.columns.get(dst_col, set())
        s_src = self.columns.get(src_col, set())
        if not s_dst:
            if s_src:
                self.columns[dst_col] = set(s_src)  # copy
            return
        # XOR = symmetric difference
        s_dst ^= s_src
        if s_dst:
            self.columns[dst_col] = s_dst
        elif dst_col in self.columns:
            del self.columns[dst_col]

    def column_rows(self, col: int) -> List[int]:
        s = self.columns.get(col)
        return sorted(s) if s else []

    def copy(self) -> "Boundary":
        return Boundary(self.simplices[:], {c: set(r) for c, r in self.columns.items()})
