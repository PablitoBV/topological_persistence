# filtration.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Iterator, Optional


@dataclass(frozen=True)
class Simplex:
    """A simplex with filtration value, dimension, and sorted vertex IDs."""
    val: float
    dim: int
    vert: Tuple[int, ...]  # sorted, unique

    def __post_init__(self):
        if self.dim != len(self.vert) - 1:
            raise ValueError(f"dim={self.dim} incompatible with vertices {self.vert}")


class Filtration:
    """Parser and container for a filtration ASCII file."""

    def __init__(self, simplices: Iterable[Simplex]):
        self._simplices: List[Simplex] = list(simplices)

    @classmethod
    def from_file(cls, path: str | Path, *, allow_comments: bool = True) -> Filtration:
        """Read 'f dim v0 ... v_dim' per line. Lines with '#' (if enabled) or blank lines are ignored."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        simplices: List[Simplex] = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                if allow_comments and line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"{path}:{lineno}: expected at least 2 fields, got {len(parts)}")

                try:
                    val = float(parts[0])
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno}: invalid float '{parts[0]}'") from e

                try:
                    dim = int(parts[1])
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno}: invalid dim '{parts[1]}'") from e

                expected = 2 + (dim + 1)
                if len(parts) != expected:
                    raise ValueError(
                        f"{path}:{lineno}: expected {expected} tokens for dim={dim}, got {len(parts)}"
                    )

                verts_raw = parts[2:]
                try:
                    verts = tuple(sorted({int(v) for v in verts_raw}))
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno}: vertices must be integers") from e

                # Re-check after deduplication: duplicates in a line are not allowed
                if len(verts) != (dim + 1):
                    raise ValueError(
                        f"{path}:{lineno}: vertex duplication detected; need exactly {dim+1} distinct IDs (got {len(verts)})"
                    )

                simplices.append(Simplex(val=val, dim=dim, vert=verts))

        return cls(simplices)

    def __len__(self) -> int:
        return len(self._simplices)

    def __iter__(self) -> Iterator[Simplex]:
        return iter(self._simplices)

    def simplices(self) -> List[Simplex]:
        """Return a shallow copy of the internal list."""
        return list(self._simplices)

    def sort_by_filtration(self, *, in_place: bool = True) -> List[Simplex]:
        """
        Sort by (val, dim, vertices) which is a common filtration order.
        Returns the sorted list; if in_place=False, returns a new list.
        """
        key = lambda s: (s.val, s.dim, s.vert)
        if in_place:
            self._simplices.sort(key=key)
            return self._simplices
        return sorted(self._simplices, key=key)

    def validate_monotonicity(self) -> Optional[str]:
        """
        Check f(σ) >= f(τ) for every face τ of σ that exists in this file.
        Returns None if OK; otherwise an error string describing the first violation found.
        """
        # Index for quick lookup
        by_vertices = {s.vert: s for s in self._simplices}

        # Generate faces for each simplex and compare values when faces are present
        for s in self._simplices:
            if s.dim == 0:
                continue
            verts = list(s.vert)
            for i in range(len(verts)):
                face = tuple(v for j, v in enumerate(verts) if j != i)
                face_simplex = by_vertices.get(face)
                if face_simplex is not None and s.val < face_simplex.val:
                    return (
                        f"Monotonicity violation: simplex {s.vert} val={s.val} "
                        f"< face {face} val={face_simplex.val}"
                    )
        return None

    def count_by_dim(self) -> dict[int, int]:
        """Histogram of simplex counts per dimension."""
        hist: dict[int, int] = {}
        for s in self._simplices:
            hist[s.dim] = hist.get(s.dim, 0) + 1
        return hist

    def __repr__(self) -> str:
        n = len(self._simplices)
        dims = sorted(self.count_by_dim().items())
        dims_str = ", ".join(f"{d}:{c}" for d, c in dims)
        return f"Filtration(n={n}, dims={{ {dims_str} }})"
