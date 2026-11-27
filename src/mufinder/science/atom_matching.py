"""Periodic-boundary atom matching utilities for structure comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from mufinder.domain.models import Structure
from mufinder.science.ase_bridge import is_muon_label
from mufinder.science.lattice import nearest_image_distance_matrix
from mufinder.science.spglib_compat import species_element_symbol


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


_MUON_GROUP = "muon"


def _matching_group(label: str) -> str:
    """The group a species label is matched within: its element, or the muon.

    The element is read with :func:`species_element_symbol`, so a CASTEP species
    label (``"Co:1"``), a CIF site label (``"Co1"``) and the bare symbol all
    match one another. A muon label (:func:`is_muon_label`, such as ``"H:mu"``)
    forms its own group, so a muon is never matched to a hydrogen atom. A label
    that names no element is matched only to the same label, ignoring
    surrounding space.
    """

    if is_muon_label(label):
        return _MUON_GROUP
    element = species_element_symbol(label)
    return str(label).strip() if element is None else element


def _group_labels(structure: Structure, indices: list[int]) -> str:
    """The distinct species labels of ``indices``, as a warning names them."""

    labels = dict.fromkeys(str(structure.species[index]).strip() for index in indices)
    return ", ".join(f"'{label}'" for label in labels)


def _group_index_map(structure: Structure) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, species in enumerate(structure.species):
        grouped.setdefault(_matching_group(str(species)), []).append(index)
    return grouped


def match_structures(
    experimental: Structure,
    dft: Structure,
    *,
    options: MatchOptions | None = None,
) -> AtomMatching:
    """Match atoms between two structures with periodic-boundary distances.

    Atoms are matched within each group, their element or the muon
    (:func:`_matching_group`), by a Hungarian assignment on the nearest-image
    distance matrix (:func:`~mufinder.science.lattice.nearest_image_distance_matrix`)
    in the DFT lattice.
    """

    if options is None:
        options = MatchOptions()

    warnings: list[str] = []
    pairs: list[AtomPair] = []
    unmatched_experimental: list[int] = []
    unmatched_dft: list[int] = []

    experimental_by_group = _group_index_map(experimental)
    dft_by_group = _group_index_map(dft)
    all_groups = sorted(set(experimental_by_group) | set(dft_by_group))

    for group in all_groups:
        experimental_indices = experimental_by_group.get(group, [])
        dft_indices = dft_by_group.get(group, [])

        if not experimental_indices:
            unmatched_dft.extend(dft_indices)
            warnings.append(
                f"DFT contains species {_group_labels(dft, dft_indices)} not present in "
                "experimental structure."
            )
            continue
        if not dft_indices:
            unmatched_experimental.extend(experimental_indices)
            warnings.append(
                f"Experimental contains species {_group_labels(experimental, experimental_indices)} "
                "not present in DFT structure."
            )
            continue

        distances = nearest_image_distance_matrix(
            [experimental.fractional_positions[index] for index in experimental_indices],
            [dft.fractional_positions[index] for index in dft_indices],
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
        warnings.append("Mean matched-atom displacement is large; atom mapping may be unreliable.")

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
