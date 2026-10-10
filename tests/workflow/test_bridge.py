"""Tests for domain Structure to workflow StructureSpec bridging."""

from __future__ import annotations

import pytest

from mufinder.workflow.bridge import StructureBridgeError, structure_to_spec
from tests.support.structures import si_pair_host


def test_structure_to_spec_converts_basic_structure() -> None:
    structure = si_pair_host(5.43)
    structure.properties["atomic_spins"] = [1.0, -1.0]

    spec = structure_to_spec(structure)

    assert spec.label == "Si"
    assert spec.source_structure_id == structure.id
    assert spec.lattice_cart.vectors[0] == pytest.approx((5.43, 0.0, 0.0))
    assert spec.positions_frac.positions[1].species == "Si"
    assert spec.atomic_spins == pytest.approx((1.0, -1.0))


def test_structure_to_spec_rejects_species_position_count_mismatch() -> None:
    structure = si_pair_host(5.43)
    structure.species = ["Si"]

    with pytest.raises(StructureBridgeError, match="does not match position count"):
        structure_to_spec(structure)


def test_structure_to_spec_rejects_invalid_lattice_shape() -> None:
    structure = si_pair_host(5.43)
    structure.lattice_vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    with pytest.raises(StructureBridgeError, match="exactly three lattice vectors"):
        structure_to_spec(structure)


def test_structure_to_spec_rejects_lattice_vector_with_wrong_component_count() -> None:
    structure = si_pair_host(5.43)
    structure.lattice_vectors = [[1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    with pytest.raises(StructureBridgeError, match="Lattice vector 1 must contain exactly three"):
        structure_to_spec(structure)


def test_structure_to_spec_rejects_fractional_position_with_wrong_component_count() -> None:
    structure = si_pair_host(5.43)
    structure.fractional_positions = [[0.0, 0.0, 0.0], [0.25, 0.25]]

    with pytest.raises(StructureBridgeError, match="Fractional position 2 must contain exactly three"):
        structure_to_spec(structure)


def test_structure_to_spec_rejects_empty_species() -> None:
    structure = si_pair_host(5.43)
    structure.species = []
    structure.fractional_positions = []

    with pytest.raises(StructureBridgeError, match="at least one species and one position"):
        structure_to_spec(structure)


def test_structure_to_spec_collapses_duplicate_species() -> None:
    structure = si_pair_host(5.43)
    structure.species = ["Si", "Si", "O"]
    structure.fractional_positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]

    spec = structure_to_spec(structure)

    assert spec.species == ("Si", "O")


def test_structure_to_spec_rejects_spin_count_mismatch() -> None:
    structure = si_pair_host(5.43)
    structure.properties["atomic_spins"] = [1.0]

    with pytest.raises(StructureBridgeError, match="Atomic spin count"):
        structure_to_spec(structure)


@pytest.mark.parametrize(
    ("atomic_spins", "message"),
    [
        ("not-a-list", "must be a list of numeric values"),
        ([1.0, "nope"], "values must be numeric"),
    ],
    ids=["not-a-list", "non-numeric"],
)
def test_structure_to_spec_rejects_malformed_atomic_spins(atomic_spins: object, message: str) -> None:
    structure = si_pair_host(5.43)
    structure.properties["atomic_spins"] = atomic_spins

    with pytest.raises(StructureBridgeError, match=message):
        structure_to_spec(structure)


def test_structure_to_spec_uses_initial_magnetic_moments_when_atomic_spins_absent() -> None:
    structure = si_pair_host(5.43)
    structure.properties["initial_magnetic_moments"] = [0.5, -0.5]

    spec = structure_to_spec(structure)

    assert spec.atomic_spins == pytest.approx((0.5, -0.5))


def test_structure_to_spec_uses_magnetic_structure_spin_fallback() -> None:
    structure = si_pair_host(5.43)
    structure.properties["magnetic_structure"] = {
        "moments": [
            {"label": "Si1", "vector": [0.0, 0.0, 1.5], "coordinate_system": "cartesian"},
            {"label": "Si2", "vector": [0.0, 0.0, -0.5], "coordinate_system": "cartesian"},
        ]
    }

    spec = structure_to_spec(structure)

    assert spec.atomic_spins == pytest.approx((1.5, -0.5))


def test_structure_to_spec_prefers_moments_when_atomic_spins_are_all_zero() -> None:
    structure = si_pair_host(5.43)
    structure.properties["atomic_spins"] = [0.0, 0.0]
    structure.properties["magnetic_structure"] = {
        "moments": [
            {"label": "Si1", "vector": [0.0, 0.0, 2.0], "coordinate_system": "cartesian"},
            {"label": "Si2", "vector": [0.0, 0.0, -2.0], "coordinate_system": "cartesian"},
        ]
    }

    spec = structure_to_spec(structure)

    assert spec.atomic_spins == pytest.approx((2.0, -2.0))


def test_structure_to_spec_keeps_zero_spins_of_a_supercell_asset() -> None:
    structure = si_pair_host(5.43)
    # Zero explicit spins would normally give way to the nonzero magnetic
    # moments, but a saved supercell asset keeps the spins it was built with.
    structure.properties["atomic_spins"] = [0.0, 0.0]
    structure.properties["supercell_asset"] = {"path": "ignored"}
    structure.properties["magnetic_structure"] = {
        "moments": [
            {"label": "Si1", "vector": [0.0, 0.0, 3.0], "coordinate_system": "cartesian"},
            {"label": "Si2", "vector": [0.0, 0.0, -3.0], "coordinate_system": "cartesian"},
        ]
    }

    spec = structure_to_spec(structure)

    assert spec.atomic_spins == pytest.approx((0.0, 0.0))


@pytest.mark.parametrize(
    "magnetic_structure",
    [
        ["not", "a", "dict"],
        {"moments": "nope"},
        {"moments": []},
        {"moments": [{"label": "Si1"}, {"label": "Si2"}]},
        {"moments": ["not-a-dict", {"label": "Si2", "vector": [0.0, 0.0]}]},
    ],
    ids=["not-a-dict", "moments-not-a-list", "no-moments", "no-vectors", "unusable-moments"],
)
def test_structure_to_spec_ignores_magnetic_structure_without_usable_moments(
    magnetic_structure: object,
) -> None:
    structure = si_pair_host(5.43)
    structure.properties["magnetic_structure"] = magnetic_structure

    spec = structure_to_spec(structure)

    assert spec.atomic_spins is None


def test_structure_to_spec_skips_moments_with_non_numeric_components() -> None:
    structure = si_pair_host(5.43)
    structure.properties["magnetic_structure"] = {
        "moments": [
            {"label": "Si1", "vector": [0.0, 0.0, "bad"]},
            {"label": "Si2", "vector": [0.0, 0.0, 4.0]},
        ]
    }

    spec = structure_to_spec(structure)

    assert spec.atomic_spins == pytest.approx((0.0, 4.0))


def test_structure_to_spec_passes_all_zero_moments_through_as_zero_spins() -> None:
    structure = si_pair_host(5.43)
    structure.properties["magnetic_structure"] = {
        "moments": [
            {"label": "Si1", "vector": [0.0, 0.0, 0.0]},
            {"label": "Si2", "vector": [0.0, 0.0, 0.0]},
        ]
    }

    spec = structure_to_spec(structure)

    assert spec.atomic_spins == pytest.approx((0.0, 0.0))
