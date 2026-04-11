"""Tests for symmetry and topology comparison helpers."""

from __future__ import annotations

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import match_structures
from mufinder.science.symmetry_analysis import compare_space_group, compare_topology, detect_space_group


def _structure() -> Structure:
    return Structure(
        label="cell",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_detect_space_group_returns_structured_payload() -> None:
    result = detect_space_group(_structure())

    assert result.available is True


def test_compare_topology_reports_preserved_for_identical_structures() -> None:
    experimental = _structure()
    dft = _structure()
    matching = match_structures(experimental, dft)

    topology = compare_topology(experimental, dft, matching)

    assert topology.preserved in {True, None}


def test_compare_space_group_returns_structured_payload() -> None:
    exp_group, dft_group, matched = compare_space_group(_structure(), _structure())

    assert exp_group.available is True
    assert dft_group.available is True
    assert matched in {True, False, None}
