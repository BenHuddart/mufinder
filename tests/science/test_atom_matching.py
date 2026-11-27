"""Tests for periodic-boundary atom matching."""

from __future__ import annotations

from itertools import product
import math

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import match_structures
from mufinder.science.lattice import minimum_image_distance, nearest_image_distance
from tests.support.symmetry_hosts import cof2_host


def _base_structure() -> Structure:
    return Structure(
        label="Base",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_match_structures_identity() -> None:
    matching = match_structures(_base_structure(), _base_structure())

    assert len(matching.pairs) == 2
    assert matching.unmatched_experimental == ()
    assert matching.unmatched_dft == ()
    assert matching.mean_displacement == pytest.approx(0.0)


def test_match_structures_wraps_periodic_boundaries() -> None:
    experimental = _base_structure()
    dft = Structure(
        label="Wrapped",
        lattice_vectors=experimental.lattice_vectors,
        species=["Si", "Si"],
        fractional_positions=[[0.98, 0.99, 0.97], [0.5, 0.5, 0.5]],
    )

    matching = match_structures(experimental, dft)

    assert len(matching.pairs) == 2
    assert matching.mean_displacement is not None
    assert matching.mean_displacement < 0.25


def test_match_structures_warns_on_unmatched_species() -> None:
    experimental = _base_structure()
    dft = Structure(
        label="Unmatched",
        lattice_vectors=experimental.lattice_vectors,
        species=["Si", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    matching = match_structures(experimental, dft)

    assert len(matching.pairs) == 1
    assert matching.unmatched_experimental == (1,)
    assert matching.unmatched_dft == (1,)
    assert matching.warnings


def test_a_labelled_structure_matches_its_unlabelled_host_on_the_element() -> None:
    host = cof2_host()
    relaxed = cof2_host()
    relaxed.species = ["Co:1", "Co:2", "F", "F", "F", "F"]

    matching = match_structures(host, relaxed)

    assert [(pair.experimental_index, pair.dft_index) for pair in matching.pairs] == [
        (index, index) for index in range(6)
    ]
    assert matching.unmatched_experimental == ()
    assert matching.unmatched_dft == ()
    assert matching.warnings == ()


@pytest.mark.parametrize("label", ["Fe1", "fe", " Fe ", "Fe:up"])
def test_site_and_species_labels_name_their_element(label: str) -> None:
    experimental = _base_structure()
    experimental.species = ["Fe", "Fe"]
    dft = _base_structure()
    dft.species = [label, "Fe"]

    matching = match_structures(experimental, dft)

    assert len(matching.pairs) == 2
    assert matching.warnings == ()


def test_a_muon_is_never_matched_to_a_host_hydrogen() -> None:
    experimental = _base_structure()
    experimental.species = ["Si", "H"]
    dft = _base_structure()
    dft.species = ["Si", "H:mu"]

    matching = match_structures(experimental, dft)

    assert [(pair.experimental_index, pair.dft_index) for pair in matching.pairs] == [(0, 0)]
    assert matching.unmatched_experimental == (1,)
    assert matching.unmatched_dft == (1,)
    assert "DFT contains species 'H:mu' not present in experimental structure." in (
        matching.warnings
    )


def test_the_displacement_is_the_true_nearest_image_in_a_hexagonal_cell() -> None:
    """A Zn atom moves from (0, 0, 0) to (0.45, 0.55, 0) in a hexagonal cell.

    The cell has a = 4 Å, c = 5 Å and γ = 120°: rows (4, 0, 0), (−2, 2√3, 0) and
    (0, 0, 5). Folding each fractional component of the difference into
    [−0.5, 0.5] gives (0.45, −0.45, 0), whose Cartesian length is 3.1177 Å; the
    nearest image is (−0.55, −0.45, 0), at 2.0298 Å.
    """

    lattice = [[4.0, 0.0, 0.0], [-2.0, 2.0 * math.sqrt(3.0), 0.0], [0.0, 0.0, 5.0]]
    experimental = Structure(
        label="exp", lattice_vectors=lattice, species=["Zn"], fractional_positions=[[0.0, 0.0, 0.0]]
    )
    dft = Structure(
        label="dft",
        lattice_vectors=lattice,
        species=["Zn"],
        fractional_positions=[[0.45, 0.55, 0.0]],
    )

    (pair,) = match_structures(experimental, dft).pairs

    assert minimum_image_distance([0.45, 0.55, 0.0], [0.0, 0.0, 0.0], lattice) == pytest.approx(
        3.117691, abs=1e-6
    )
    assert pair.displacement == pytest.approx(2.029778, abs=1e-6)


def test_matching_a_distorted_supercell_uses_the_pairwise_nearest_image_distance() -> None:
    rng = np.random.default_rng(306)
    # A skewed 3 x 3 x 2 cell, so that the lattice product and the rounding both matter.
    lattice = [[12.0, 0.4, -0.3], [-1.1, 11.5, 0.6], [0.8, -0.5, 9.0]]
    grid = np.array(list(product(range(3), range(3), range(2))), dtype=float) / [3.0, 3.0, 2.0]
    experimental = grid + rng.normal(scale=0.01, size=grid.shape)
    # Some positions wrap across the cell boundary.
    dft = (grid + rng.normal(scale=0.02, size=grid.shape)) % 1.0
    species = ["Si"] * len(grid)

    matching = match_structures(
        Structure(
            label="exp",
            lattice_vectors=lattice,
            species=species,
            fractional_positions=experimental.tolist(),
        ),
        Structure(
            label="dft", lattice_vectors=lattice, species=species, fractional_positions=dft.tolist()
        ),
    )

    assert len(matching.pairs) == len(grid)
    for pair in matching.pairs:
        assert pair.displacement == pytest.approx(
            nearest_image_distance(
                dft[pair.dft_index], experimental[pair.experimental_index], lattice
            ),
            abs=1e-12,
        )


def test_a_label_that_names_no_element_matches_only_itself() -> None:
    experimental = _base_structure()
    experimental.species = ["Si", "Q"]
    dft = _base_structure()
    dft.species = ["Si", " Q "]

    matching = match_structures(experimental, dft)

    assert len(matching.pairs) == 2

    dft.species = ["Si", "Qx"]
    assert match_structures(experimental, dft).unmatched_dft == (1,)
