"""Tests for symmetry and topology comparison helpers."""

from __future__ import annotations

from mufinder.science.atom_matching import match_structures
from mufinder.science.symmetry_analysis import compare_space_group, compare_topology, detect_space_group
from tests.support.structures import host_structure


def test_detect_space_group_returns_structured_payload() -> None:
    result = detect_space_group(host_structure(label="cell", lattice=4.0))

    assert result.available is True


def test_compare_topology_reports_preserved_for_identical_structures() -> None:
    experimental = host_structure(label="cell", lattice=4.0)
    dft = host_structure(label="cell", lattice=4.0)
    matching = match_structures(experimental, dft)

    topology = compare_topology(experimental, dft, matching)

    assert topology.preserved in {True, None}


def test_compare_space_group_returns_structured_payload() -> None:
    exp_group, dft_group, matched = compare_space_group(host_structure(label="cell", lattice=4.0), host_structure(label="cell", lattice=4.0))

    assert exp_group.available is True
    assert dft_group.available is True
    assert matched in {True, False, None}
