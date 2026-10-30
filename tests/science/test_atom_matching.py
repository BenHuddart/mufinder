"""Tests for periodic-boundary atom matching."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import match_structures
from mufinder.science.lattice import minimum_image_distance, minimum_image_distance_matrix


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


def test_matching_a_distorted_supercell_uses_the_pairwise_minimum_image_distance() -> None:
    rng = np.random.default_rng(306)
    # A skewed 3 x 3 x 2 cell, so that the lattice product and the rounding both matter.
    lattice = [[12.0, 0.4, -0.3], [-1.1, 11.5, 0.6], [0.8, -0.5, 9.0]]
    grid = np.array(list(product(range(3), range(3), range(2))), dtype=float) / [3.0, 3.0, 2.0]
    experimental = grid + rng.normal(scale=0.01, size=grid.shape)
    # Some positions wrap across the cell boundary.
    dft = (grid + rng.normal(scale=0.02, size=grid.shape)) % 1.0
    species = ["Si"] * len(grid)

    matrix = minimum_image_distance_matrix(experimental, dft, lattice)

    for row, left in enumerate(experimental):
        for column, right in enumerate(dft):
            assert matrix[row, column] == pytest.approx(
                minimum_image_distance(right, left, lattice), abs=1e-12
            )

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
            minimum_image_distance(
                dft[pair.dft_index], experimental[pair.experimental_index], lattice
            ),
            abs=1e-12,
        )
