# barcode.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from filtration import Simplex


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
        death = float("inf") if death_str == "inf" else float(death_str)
        return (dim, birth, death)

    lines.sort(key=sort_key)

    # Write to file and print for quick inspection
    with open(outfile, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print("\n=== Barcode ===")
    for line in lines:
        print(line)
