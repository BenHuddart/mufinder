"""Tests for periodic-boundary atom matching."""

from __future__ import annotations

import pytest

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import match_structures


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
