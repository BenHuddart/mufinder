"""Periodic-boundary atom matching utilities for structure comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from mufinder.domain.models import Structure


@dataclass(frozen=True, slots=True)
class AtomPair:
    """One matched atom pair and its periodic-boundary displacement."""

    experimental_index: int
    dft_index: int
    displacement: float


@dataclass(frozen=True, slots=True)
class AtomMatching:
    """Result of matching experimental and DFT atoms."""

    pairs: tuple[AtomPair, ...] = ()
    unmatched_experimental: tuple[int, ...] = ()
    unmatched_dft: tuple[int, ...] = ()
    mean_displacement: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MatchOptions:
    """Controls for the matching algorithm."""

    max_reasonable_displacement: float = 2.5


def _minimum_image_displacement(
    lhs_fractional: Iterable[float],
    rhs_fractional: Iterable[float],
    lattice_vectors: list[list[float]],
) -> float:
    lhs = np.asarray(list(lhs_fractional), dtype=float)
    rhs = np.asarray(list(rhs_fractional), dtype=float)
    lattice = np.asarray(lattice_vectors, dtype=float)

    delta_fractional = rhs - lhs
    delta_fractional -= np.round(delta_fractional)
    delta_cartesian = delta_fractional @ lattice
    return float(np.linalg.norm(delta_cartesian))


def _species_index_map(structure: Structure) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, species in enumerate(structure.species):
        grouped.setdefault(str(species), []).append(index)
    return grouped


def match_structures(
    experimental: Structure,
    dft: Structure,
    *,
    options: MatchOptions | None = None,
) -> AtomMatching:
    """Match atoms between two structures with periodic-boundary distances.

    Matching is performed separately per species using a Hungarian assignment on
    the minimum-image distance matrix.
    """

    if options is None:
        options = MatchOptions()

    warnings: list[str] = []
    pairs: list[AtomPair] = []
    unmatched_experimental: list[int] = []
    unmatched_dft: list[int] = []

    experimental_by_species = _species_index_map(experimental)
    dft_by_species = _species_index_map(dft)
    all_species = sorted(set(experimental_by_species) | set(dft_by_species))

    for species in all_species:
        experimental_indices = experimental_by_species.get(species, [])
        dft_indices = dft_by_species.get(species, [])

        if not experimental_indices:
            unmatched_dft.extend(dft_indices)
            warnings.append(
                f"DFT contains species '{species}' not present in experimental structure."
            )
            continue
        if not dft_indices:
            unmatched_experimental.extend(experimental_indices)
            warnings.append(
                f"Experimental contains species '{species}' not present in DFT structure."
            )
            continue

        distances = np.zeros((len(experimental_indices), len(dft_indices)), dtype=float)
        for row_index, experimental_index in enumerate(experimental_indices):
            experimental_position = experimental.fractional_positions[experimental_index]
            for col_index, dft_index in enumerate(dft_indices):
                dft_position = dft.fractional_positions[dft_index]
                distances[row_index, col_index] = _minimum_image_displacement(
                    experimental_position,
                    dft_position,
                    dft.lattice_vectors,
                )

        selected_rows, selected_cols = linear_sum_assignment(distances)
        matched_rows = set(int(value) for value in selected_rows)
        matched_cols = set(int(value) for value in selected_cols)

        for row_index, col_index in zip(selected_rows, selected_cols, strict=False):
            displacement = float(distances[int(row_index), int(col_index)])
            pairs.append(
                AtomPair(
                    experimental_index=experimental_indices[int(row_index)],
                    dft_index=dft_indices[int(col_index)],
                    displacement=displacement,
                )
            )

        for row_index, experimental_index in enumerate(experimental_indices):
            if row_index not in matched_rows:
                unmatched_experimental.append(experimental_index)
        for col_index, dft_index in enumerate(dft_indices):
            if col_index not in matched_cols:
                unmatched_dft.append(dft_index)

    pairs.sort(key=lambda pair: pair.experimental_index)

    if not pairs:
        warnings.append("No atoms could be matched between experimental and DFT structures.")
        return AtomMatching(
            pairs=(),
            unmatched_experimental=tuple(sorted(unmatched_experimental)),
            unmatched_dft=tuple(sorted(unmatched_dft)),
            mean_displacement=None,
            warnings=tuple(warnings),
        )

    mean_displacement = float(np.mean([pair.displacement for pair in pairs]))
    if mean_displacement > options.max_reasonable_displacement:
        warnings.append(
            "Mean matched-atom displacement is large; atom mapping may be unreliable."
        )

    if unmatched_experimental or unmatched_dft:
        warnings.append(
            "Exact atom matching was not possible; metrics are based on partial correspondences."
        )

    return AtomMatching(
        pairs=tuple(pairs),
        unmatched_experimental=tuple(sorted(unmatched_experimental)),
        unmatched_dft=tuple(sorted(unmatched_dft)),
        mean_displacement=mean_displacement,
        warnings=tuple(warnings),
    )
