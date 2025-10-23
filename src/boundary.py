# boundary.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from filtration import Filtration, Simplex

def _faces(verts: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    return [tuple(verts[j] for j in range(len(verts)) if j != i) for i in range(len(verts))]

@dataclass
class Boundary:
    simplices: List[Simplex]
    matrix: List[List[int]]  # 0/1

    @property
    def size(self) -> int:
        return len(self.simplices)

    @classmethod
    def from_filtration(cls, filtration: Filtration, *, strict: bool = True) -> "Boundary":
        ordered = filtration.sort_by_filtration(in_place=False)
        n = len(ordered)
        mat = [[0] * n for _ in range(n)]
        index_of = {s.vert: i for i, s in enumerate(ordered)}
        for col, simplex in enumerate(ordered):
            if simplex.dim == 0:
                continue
            for face in _faces(simplex.vert):
                row = index_of.get(face)
                if row is None:
                    if strict:
                        raise ValueError(f"Missing face {face} of {simplex.vert}")
                    continue
                mat[row][col] = 1
        return cls(ordered, mat)

    def print(self) -> None:
        for row in self.matrix:
            print(" ".join("1" if v else "0" for v in row))

    def lowest_one(self, col: int) -> Optional[int]:
        for row in range(self.size - 1, -1, -1):
            if self.matrix[row][col] == 1:
                return row
        return None

    def add_column(self, dst_col: int, src_col: int) -> None:
        for row in range(self.size):
            self.matrix[row][dst_col] ^= self.matrix[row][src_col]

    def column_rows(self, col: int) -> List[int]:
        return [row for row in range(self.size) if self.matrix[row][col] == 1]

    def copy(self) -> "Boundary":
        return Boundary(self.simplices[:], [r[:] for r in self.matrix])
