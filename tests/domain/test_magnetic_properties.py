"""Tests for the readers of persisted magnetic metadata on ``Structure.properties``."""

from __future__ import annotations

from typing import Any

import pytest

from mufinder.domain.magnetic_properties import (
    atomic_spins_from_magnetic_moments,
    atomic_spins_from_structure,
    has_nonzero_spins,
    propagation_vector_from_structure,
)
from mufinder.domain.models import Structure


def _structure(species_count: int = 2, **properties: Any) -> Structure:
    return Structure(
        label="host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Fe"] * species_count,
        fractional_positions=[[0.5 * index, 0.0, 0.0] for index in range(species_count)],
        properties=dict(properties),
    )


# has_nonzero_spins


def test_has_nonzero_spins_treats_none_and_all_zero_as_no_spins() -> None:
    assert has_nonzero_spins(None) is False
    assert has_nonzero_spins([]) is False
    assert has_nonzero_spins([0.0, -0.0, 1e-13]) is False
    assert has_nonzero_spins([0.0, -2.0]) is True


def test_has_nonzero_spins_honours_an_explicit_tolerance() -> None:
    assert has_nonzero_spins([0.05], tolerance=0.1) is False
    assert has_nonzero_spins([0.5], tolerance=0.1) is True


# atomic_spins_from_magnetic_moments


def test_moment_spins_skip_malformed_moments_and_honour_atom_index() -> None:
    magnetic = {
        "moments": [
            "not-a-dict",
            {"atom_index": "x", "vector": [0, 0, 1]},
            {"atom_index": 99, "vector": [0, 0, 1]},
            {"atom_index": 0, "vector": [0, 0]},
            {"atom_index": 0, "vector": "nope"},
            {"atom_index": 0, "vector": [0, 0, "bad"]},
            {"atom_index": 0, "vector": [0, 0, 4.0]},
        ]
    }

    assert atomic_spins_from_magnetic_moments(magnetic, atom_count=1) == pytest.approx([4.0])


def test_moment_spins_are_none_without_a_numeric_moment() -> None:
    assert atomic_spins_from_magnetic_moments({}, atom_count=2) is None
    assert atomic_spins_from_magnetic_moments({"moments": []}, atom_count=2) is None
    assert atomic_spins_from_magnetic_moments({"moments": [{"vector": [0, 0, "x"]}]}, atom_count=2) is None


def test_moment_spins_report_all_zero_moments_as_spins_not_none() -> None:
    magnetic = {"moments": [{"vector": [1.0, 0.0, 0.0]}]}

    assert atomic_spins_from_magnetic_moments(magnetic, atom_count=2) == [0.0, 0.0]


def test_moment_spins_place_by_atom_index_only_when_asked() -> None:
    magnetic = {"moments": [{"atom_index": 1, "vector": [0, 0, 3.0]}, {"vector": [0, 0, -1.0]}]}

    assert atomic_spins_from_magnetic_moments(magnetic, atom_count=3) == [0.0, -1.0, 0.0]
    assert atomic_spins_from_magnetic_moments(magnetic, atom_count=3, honour_atom_index=False) == [3.0, -1.0, 0.0]


# atomic_spins_from_structure


def test_structure_spins_pad_to_the_species_count_by_default() -> None:
    structure = _structure(3, magnetic_structure={"moments": [{"vector": [0, 0, 2.5]}]})

    assert atomic_spins_from_structure(structure) == [2.5, 0.0, 0.0]


def test_structure_spins_honour_an_explicit_atom_count() -> None:
    moments = [{"vector": [0, 0, 1.0]}, {"vector": [0, 0, 2.0]}, {"vector": [0, 0, 3.0]}]
    structure = _structure(2, magnetic_structure={"moments": moments})

    assert atomic_spins_from_structure(structure, atom_count=4) == [1.0, 2.0, 3.0, 0.0]
    assert atomic_spins_from_structure(structure, atom_count=1) == [1.0]


def test_structure_spins_place_moments_in_list_order_ignoring_atom_index() -> None:
    structure = _structure(2, magnetic_structure={"moments": [{"atom_index": 1, "vector": [0, 0, 5.0]}]})

    assert atomic_spins_from_structure(structure) == [5.0, 0.0]


def test_structure_spins_are_none_without_magnetic_structure_or_numeric_moments() -> None:
    assert atomic_spins_from_structure(_structure()) is None
    assert atomic_spins_from_structure(_structure(magnetic_structure="not-a-dict")) is None
    assert atomic_spins_from_structure(_structure(magnetic_structure={"moments": [{"vector": None}]})) is None


# propagation_vector_from_structure


def test_propagation_vector_prefers_the_top_level_property() -> None:
    structure = _structure(
        propagation_vector=[0.5, 0.0, 0.0],
        magnetic_structure={"propagation_vector": [0.0, 0.0, 0.25]},
    )

    assert propagation_vector_from_structure(structure) == (0.5, 0.0, 0.0)
    assert propagation_vector_from_structure(structure, strict=True) == (0.5, 0.0, 0.0)


def test_propagation_vector_falls_back_to_magnetic_structure_then_zero() -> None:
    assert propagation_vector_from_structure(_structure(magnetic_structure={"propagation_vector": [0, 0, 0.25]})) == (
        0.0,
        0.0,
        0.25,
    )
    assert propagation_vector_from_structure(_structure()) == (0.0, 0.0, 0.0)
    assert propagation_vector_from_structure(_structure(), strict=True) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("stored", "lenient", "strict"),
    [
        ([0.5], (0.5, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ([0.5, "x", 0.25], (0.5, 0.0, 0.25), (0.0, 0.0, 0.0)),
        ([0.1, 0.2, 0.3, 0.4], (0.1, 0.2, 0.3), (0.0, 0.0, 0.0)),
        (["0.5", 0, 0], (0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
        ("0.5 0 0", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ({"x": 1}, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        (7, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ],
    ids=["short", "non-numeric-component", "long", "numeric-strings", "string", "mapping", "scalar"],
)
def test_propagation_vector_lenient_and_strict_readings(
    stored: object,
    lenient: tuple[float, float, float],
    strict: tuple[float, float, float],
) -> None:
    structure = _structure(propagation_vector=stored)

    assert propagation_vector_from_structure(structure) == lenient
    assert propagation_vector_from_structure(structure, strict=True) == strict
