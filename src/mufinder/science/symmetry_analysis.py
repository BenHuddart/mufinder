"""Space-group and topology analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.data import atomic_numbers

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import AtomMatching
from mufinder.science.bonding_analysis import connectivity_edges
from mufinder.science.spglib_compat import get_crystal_symmetry_dataset


@dataclass(frozen=True, slots=True)
class SpaceGroupResult:
    """Detected space-group metadata."""

    number: int | None
    symbol: str | None
    hall_symbol: str | None
    available: bool


@dataclass(frozen=True, slots=True)
class TopologyResult:
    """Connectivity-preservation summary."""

    preserved: bool | None
    added_bonds: int
    removed_bonds: int
    summary: str


def detect_space_group(structure: Structure, *, symprec: float = 1e-2) -> SpaceGroupResult:
    """Detect space group metadata via spglib."""

    try:
        numbers = [atomic_numbers[str(symbol).capitalize()] for symbol in structure.species]
    except KeyError:
        return SpaceGroupResult(number=None, symbol=None, hall_symbol=None, available=False)

    cell = (
        np.asarray(structure.lattice_vectors, dtype=float),
        np.asarray(structure.fractional_positions, dtype=float),
        np.asarray(numbers, dtype=int),
    )
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
    symprec: float = 1e-2,
) -> tuple[SpaceGroupResult, SpaceGroupResult, bool | None]:
    """Return both space groups and a match flag."""

    exp_group = detect_space_group(experimental, symprec=symprec)
    dft_group = detect_space_group(dft, symprec=symprec)

    if not exp_group.available or not dft_group.available:
        return exp_group, dft_group, None
    if exp_group.number is None or dft_group.number is None:
        return exp_group, dft_group, None
    return exp_group, dft_group, exp_group.number == dft_group.number


def compare_topology(
    experimental: Structure,
    dft: Structure,
    matching: AtomMatching,
) -> TopologyResult:
    """Check whether matched-atom connectivity is preserved."""

    if not matching.pairs:
        return TopologyResult(
            preserved=None,
            added_bonds=0,
            removed_bonds=0,
            summary="No matched atoms were available for topology comparison.",
        )

    exp_edges = connectivity_edges(experimental)
    dft_edges = connectivity_edges(dft)

    exp_to_dft = {pair.experimental_index: pair.dft_index for pair in matching.pairs}

    removed = 0
    compared = 0
    for left, right in exp_edges:
        mapped_left = exp_to_dft.get(left)
        mapped_right = exp_to_dft.get(right)
        if mapped_left is None or mapped_right is None:
            continue
        compared += 1
        mapped_edge = tuple(sorted((mapped_left, mapped_right)))
        if mapped_edge not in dft_edges:
            removed += 1

    dft_to_exp = {pair.dft_index: pair.experimental_index for pair in matching.pairs}
    added = 0
    for left, right in dft_edges:
        mapped_left = dft_to_exp.get(left)
        mapped_right = dft_to_exp.get(right)
        if mapped_left is None or mapped_right is None:
            continue
        mapped_edge = tuple(sorted((mapped_left, mapped_right)))
        if mapped_edge not in exp_edges:
            added += 1

    preserved = added == 0 and removed == 0
    summary = (
        "Topology preserved." if preserved else f"Connectivity changed: +{added} / -{removed} bonds."
    )
    return TopologyResult(
        preserved=preserved,
        added_bonds=added,
        removed_bonds=removed,
        summary=summary,
    )
