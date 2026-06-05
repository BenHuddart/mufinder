"""Coverage-focused tests for workflow bridge helper branches.

These exercise error paths and fallback branches in
:mod:`mufinder.workflow.bridge` not already covered by
``tests/test_workflow_bridge.py``. No network, GUI, files, sleep, or
randomness are involved.
"""

from __future__ import annotations

import pytest

from mufinder.domain.models import Structure
from mufinder.workflow.bridge import StructureBridgeError, structure_to_spec


def _host_structure() -> Structure:
    return Structure(
        label="Si",
        lattice_vectors=[[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


# --- _as_vector3 / structure validation error branches -------------------


def test_rejects_lattice_vector_with_wrong_component_count() -> None:
    structure = _host_structure()
    # Three lattice vectors, but the first has only two components.
    structure.lattice_vectors = [[1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    with pytest.raises(StructureBridgeError, match="Lattice vector 1 must contain exactly three"):
        structure_to_spec(structure)


def test_rejects_fractional_position_with_wrong_component_count() -> None:
    structure = _host_structure()
    structure.fractional_positions = [[0.0, 0.0, 0.0], [0.25, 0.25]]

    with pytest.raises(StructureBridgeError, match="Fractional position 2 must contain exactly three"):
        structure_to_spec(structure)


def test_rejects_empty_species() -> None:
    structure = _host_structure()
    structure.species = []
    structure.fractional_positions = []

    with pytest.raises(StructureBridgeError, match="at least one species and one position"):
        structure_to_spec(structure)


# --- atomic_spins validation branches ------------------------------------


def test_rejects_non_list_atomic_spins() -> None:
    structure = _host_structure()
    structure.properties["atomic_spins"] = "not-a-list"

    with pytest.raises(StructureBridgeError, match="must be a list of numeric values"):
        structure_to_spec(structure)


def test_rejects_non_numeric_atomic_spins() -> None:
    structure = _host_structure()
    structure.properties["atomic_spins"] = [1.0, "nope"]

    with pytest.raises(StructureBridgeError, match="values must be numeric"):
        structure_to_spec(structure)


def test_uses_initial_magnetic_moments_when_atomic_spins_absent() -> None:
    structure = _host_structure()
    structure.properties["initial_magnetic_moments"] = [0.5, -0.5]

    spec = structure_to_spec(structure)

    assert spec.atomic_spins == pytest.approx((0.5, -0.5))


# --- supercell_asset short-circuit branch --------------------------------


def test_supercell_asset_keeps_zero_atomic_spins_over_moment_fallback() -> None:
    structure = _host_structure()
    # Zero explicit spins that would normally be replaced by the nonzero
    # magnetic-structure fallback, but the supercell_asset guard keeps them.
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


# --- _spins_from_magnetic_structure branches -----------------------------


def test_magnetic_structure_not_a_dict_is_ignored() -> None:
    structure = _host_structure()
    structure.properties["magnetic_structure"] = ["not", "a", "dict"]

    spec = structure_to_spec(structure)

    assert spec.atomic_spins is None


def test_magnetic_structure_moments_not_list_is_ignored() -> None:
    structure = _host_structure()
    structure.properties["magnetic_structure"] = {"moments": "nope"}

    spec = structure_to_spec(structure)

    assert spec.atomic_spins is None


def test_magnetic_structure_empty_moments_is_ignored() -> None:
    structure = _host_structure()
    structure.properties["magnetic_structure"] = {"moments": []}

    spec = structure_to_spec(structure)

    assert spec.atomic_spins is None


def test_magnetic_structure_skips_non_dict_and_bad_vector_moments() -> None:
    structure = _host_structure()
    structure.properties["magnetic_structure"] = {
        "moments": [
            "not-a-dict",  # skipped (not a dict)
            {"label": "Si2", "vector": [0.0, 0.0]},  # skipped (vector too short)
        ]
    }

    spec = structure_to_spec(structure)

    # No numeric value seen -> fallback returns None.
    assert spec.atomic_spins is None


def test_magnetic_structure_skips_non_numeric_vector_component() -> None:
    structure = _host_structure()
    structure.properties["magnetic_structure"] = {
        "moments": [
            {"label": "Si1", "vector": [0.0, 0.0, "bad"]},  # float() raises -> skipped
            {"label": "Si2", "vector": [0.0, 0.0, 4.0]},  # numeric -> used
        ]
    }

    spec = structure_to_spec(structure)

    assert spec.atomic_spins == pytest.approx((0.0, 4.0))


def test_magnetic_structure_missing_vector_key_is_ignored() -> None:
    structure = _host_structure()
    structure.properties["magnetic_structure"] = {
        "moments": [{"label": "Si1"}, {"label": "Si2"}]
    }

    spec = structure_to_spec(structure)

    assert spec.atomic_spins is None


def test_all_zero_moment_fallback_does_not_set_spins() -> None:
    structure = _host_structure()
    structure.properties["magnetic_structure"] = {
        "moments": [
            {"label": "Si1", "vector": [0.0, 0.0, 0.0]},
            {"label": "Si2", "vector": [0.0, 0.0, 0.0]},
        ]
    }

    spec = structure_to_spec(structure)

    # saw_numeric is True, so the fallback becomes raw_spins (no explicit
    # atomic_spins were provided) and is parsed through as all-zero spins.
    assert spec.atomic_spins == pytest.approx((0.0, 0.0))


def test_unique_species_collapses_duplicates() -> None:
    structure = _host_structure()
    structure.species = ["Si", "Si", "O"]
    structure.fractional_positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]

    spec = structure_to_spec(structure)

    assert spec.species == ("Si", "O")
