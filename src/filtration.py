from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class Simplex:
    """
    One simplex: filtration value, dimension, and sorted vertex ids.
    - val: filtration value (time of appearance)
    - dim: simplex dimension
    - vert: tuple of distinct vertex ids, ascending
    """
    val: float
    dim: int
    vert: Tuple[int, ...]  # sorted, unique

    def __post_init__(self) -> None:
        # Basic shape check: a dim-k simplex has k+1 vertices
        if self.dim != len(self.vert) - 1:
            raise ValueError(f"dim={self.dim} incompatible with vertices {self.vert}")


class Filtration:
    """
    Load and hold a filtration.
    File format: one simplex per line
        f dim v0 ... v_dim
    where f is a float, dim is an int, and vertices are ints.
    """

    def __init__(self, simplices: Iterable[Simplex]) -> None:
        self._simplices: List[Simplex] = list(simplices)

    @classmethod
    def from_file(cls, path: str | Path, *, allow_comments: bool = True) -> Filtration:
        """
        Parse a text file with lines "f dim v0 ... v_dim".
        Empty lines are skipped. If allow_comments is True, lines starting
        with '#' are skipped.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)

        simplices: List[Simplex] = []
        with p.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                if allow_comments and line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"{p}:{lineno}: need at least 2 tokens, got {len(parts)}")

                # f (float)
                try:
                    val = float(parts[0])
                except ValueError as e:
                    raise ValueError(f"{p}:{lineno}: invalid float '{parts[0]}'") from e

                # dim (int)
                try:
                    dim = int(parts[1])
                except ValueError as e:
                    raise ValueError(f"{p}:{lineno}: invalid dim '{parts[1]}'") from e

                # expect exactly k+1 vertices
                expected = 2 + (dim + 1)
                if len(parts) != expected:
                    raise ValueError(
                        f"{p}:{lineno}: expected {expected} tokens for dim={dim}, got {len(parts)}"
                    )

                # parse vertices as ints, enforce distinct ids, then sort
                verts_raw = parts[2:]
                try:
                    verts_set = {int(v) for v in verts_raw}
                except ValueError as e:
                    raise ValueError(f"{p}:{lineno}: vertices must be integers") from e
                verts = tuple(sorted(verts_set))

                # duplicates on the line are not allowed
                if len(verts) != (dim + 1):
                    raise ValueError(
                        f"{p}:{lineno}: need exactly {dim+1} distinct vertices, got {len(verts)}"
                    )

                simplices.append(Simplex(val=val, dim=dim, vert=verts))

        return cls(simplices)

    # -------- collection API --------

    def __len__(self) -> int:
        return len(self._simplices)

    def __iter__(self) -> Iterator[Simplex]:
        return iter(self._simplices)

    def simplices(self) -> List[Simplex]:
        """Return a shallow copy of the list of simplices."""
        return list(self._simplices)

    # -------- utilities --------

    def sort_by_filtration(self, *, in_place: bool = True) -> List[Simplex]:
        """
        Sort by (val, dim, vertices). This is a common stable order
        for building boundary and doing reductions.
        """
        key = lambda s: (s.val, s.dim, s.vert)
        if in_place:
            self._simplices.sort(key=key)
            return self._simplices
        return sorted(self._simplices, key=key)

    def validate_monotonicity(self) -> Optional[str]:
        """
        Check that any face present in the file does not appear later
        than its coface: for every pair (face, simplex), require
        val(simplex) >= val(face). Return None if ok, else an error string.
        """
        # fast lookup by vertex tuple
        by_vertices = {s.vert: s for s in self._simplices}

        for s in self._simplices:
            if s.dim == 0:
                continue
            verts = list(s.vert)
            for i in range(len(verts)):
                face = tuple(v for j, v in enumerate(verts) if j != i)
                face_s = by_vertices.get(face)
                if face_s is not None and s.val < face_s.val:
                    return (
                        f"monotonicity violation: simplex {s.vert} val={s.val} "
                        f"< face {face} val={face_s.val}"
                    )
        return None

    def count_by_dim(self) -> dict[int, int]:
        """Return a histogram of simplex counts per dimension."""
        hist: dict[int, int] = {}
        for s in self._simplices:
            hist[s.dim] = hist.get(s.dim, 0) + 1
        return hist

    def __repr__(self) -> str:
        n = len(self._simplices)
        dims = sorted(self.count_by_dim().items())
        dims_str = ", ".join(f"{d}:{c}" for d, c in dims)
        return f"Filtration(n={n}, dims={{ {dims_str} }})"
