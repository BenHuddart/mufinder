"""Tests for symmetry and topology comparison helpers."""

from __future__ import annotations

from unittest.mock import patch

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import AtomMatching, match_structures
import mufinder.science.symmetry_analysis as symmetry_module
from mufinder.science.symmetry_analysis import compare_space_group, compare_topology, detect_space_group
from tests.support.structures import host_structure


def _silicon() -> Structure:
    return host_structure(label="cell", lattice=4.0)


def _unknown_species() -> Structure:
    return host_structure(label="bad", lattice=4.0, species=("Xx", "Si"))


# detect_space_group


def test_detect_space_group_populates_fields_for_valid_dataset() -> None:
    result = detect_space_group(_silicon())

    assert result.available is True
    assert isinstance(result.number, int)
    assert isinstance(result.symbol, str)
    assert isinstance(result.hall_symbol, str)


def test_detect_space_group_unknown_species_marks_unavailable() -> None:
    result = detect_space_group(_unknown_species())

    assert result.available is False
    assert result.number is None
    assert result.symbol is None
    assert result.hall_symbol is None


def test_detect_space_group_none_dataset_is_available_but_empty() -> None:
    with patch.object(symmetry_module, "get_crystal_symmetry_dataset", return_value=None):
        result = detect_space_group(_silicon())

    assert result.available is True
    assert result.number is None
    assert result.symbol is None
    assert result.hall_symbol is None


# compare_space_group


def test_compare_space_group_returns_structured_payload() -> None:
    exp_group, dft_group, matched = compare_space_group(_silicon(), _silicon())

    assert exp_group.available is True
    assert dft_group.available is True
    assert matched in {True, False, None}


def test_compare_space_group_returns_none_when_unavailable() -> None:
    exp_group, _dft_group, matched = compare_space_group(_unknown_species(), _silicon())

    assert exp_group.available is False
    assert matched is None


def test_compare_space_group_returns_none_when_number_missing() -> None:
    with patch.object(symmetry_module, "get_crystal_symmetry_dataset", return_value=None):
        exp_group, dft_group, matched = compare_space_group(_silicon(), _silicon())

    assert exp_group.available is True
    assert dft_group.available is True
    assert exp_group.number is None
    assert matched is None


# compare_topology


def test_compare_topology_reports_preserved_for_identical_structures() -> None:
    experimental = _silicon()
    dft = _silicon()
    matching = match_structures(experimental, dft)

    topology = compare_topology(experimental, dft, matching)

    assert topology.preserved in {True, None}


def test_compare_topology_no_pairs_returns_unknown() -> None:
    structure = _silicon()
    result = compare_topology(structure, structure, AtomMatching())

    assert result.preserved is None
    assert result.added_bonds == 0
    assert result.removed_bonds == 0
    assert result.summary == "No matched atoms were available for topology comparison."


def test_compare_topology_detects_removed_bond() -> None:
    bonded = host_structure(label="bonded", lattice=10.0, positions=((0.0, 0.0, 0.0), (0.15, 0.0, 0.0)))
    stretched = host_structure(label="stretched", lattice=10.0, positions=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)))
    matching = match_structures(bonded, stretched)

    result = compare_topology(bonded, stretched, matching)

    assert result.preserved is False
    assert result.removed_bonds == 1
    assert result.added_bonds == 0
    assert "Connectivity changed" in result.summary
