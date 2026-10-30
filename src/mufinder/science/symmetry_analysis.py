"""Space-group analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass


from mufinder.domain.models import Structure
from mufinder.science.spglib_compat import (
    SYMMETRY_TOLERANCE_ANG,
    get_crystal_symmetry_dataset,
    spglib_cell,
)


@dataclass(frozen=True, slots=True)
class SpaceGroupResult:
    """Detected space-group metadata."""

    number: int | None
    symbol: str | None
    hall_symbol: str | None
    available: bool


def detect_space_group(
    structure: Structure, *, symprec: float = SYMMETRY_TOLERANCE_ANG
) -> SpaceGroupResult:
    """Detect space group metadata via spglib."""

    cell = spglib_cell(structure)
    if cell is None:
        return SpaceGroupResult(number=None, symbol=None, hall_symbol=None, available=False)

    dataset = get_crystal_symmetry_dataset(cell, symprec=float(symprec))
    if dataset is None:
        return SpaceGroupResult(number=None, symbol=None, hall_symbol=None, available=True)

    if hasattr(dataset, "number"):
        number = int(dataset.number)
        symbol = str(dataset.international)
        hall_symbol = str(dataset.hall)
    else:  # pragma: no cover - compatibility with older spglib mapping-style datasets.
        number = int(dataset["number"])
        symbol = str(dataset["international"])
        hall_symbol = str(dataset["hall"])

    return SpaceGroupResult(
        number=number,
        symbol=symbol,
        hall_symbol=hall_symbol,
        available=True,
    )


def compare_space_group(
    experimental: Structure,
    dft: Structure,
    *,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
) -> tuple[SpaceGroupResult, SpaceGroupResult, bool | None]:
    """Return both space groups and a match flag."""

    exp_group = detect_space_group(experimental, symprec=symprec)
    dft_group = detect_space_group(dft, symprec=symprec)

    if not exp_group.available or not dft_group.available:
        return exp_group, dft_group, None
    if exp_group.number is None or dft_group.number is None:
        return exp_group, dft_group, None
    return exp_group, dft_group, exp_group.number == dft_group.number
