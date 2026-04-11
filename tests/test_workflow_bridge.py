"""Tests for domain Structure to workflow StructureSpec bridging."""

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


def test_structure_to_spec_converts_basic_structure() -> None:
    structure = _host_structure()
    structure.properties["atomic_spins"] = [1.0, -1.0]

    spec = structure_to_spec(structure)

    assert spec.label == "Si"
    assert spec.source_structure_id == structure.id
    assert spec.lattice_cart.vectors[0] == pytest.approx((5.43, 0.0, 0.0))
    assert spec.positions_frac.positions[1].species == "Si"
    assert spec.atomic_spins == pytest.approx((1.0, -1.0))


def test_structure_to_spec_rejects_species_position_count_mismatch() -> None:
    structure = _host_structure()
    structure.species = ["Si"]

    with pytest.raises(StructureBridgeError, match="does not match position count"):
        structure_to_spec(structure)


def test_structure_to_spec_rejects_invalid_lattice_shape() -> None:
    structure = _host_structure()
    structure.lattice_vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    with pytest.raises(StructureBridgeError, match="exactly three lattice vectors"):
        structure_to_spec(structure)


def test_structure_to_spec_rejects_spin_count_mismatch() -> None:
    structure = _host_structure()
    structure.properties["atomic_spins"] = [1.0]

    with pytest.raises(StructureBridgeError, match="Atomic spin count"):
        structure_to_spec(structure)


def test_structure_to_spec_uses_magnetic_structure_spin_fallback() -> None:
    structure = _host_structure()
    structure.properties["magnetic_structure"] = {
        "moments": [
            {"label": "Si1", "vector": [0.0, 0.0, 1.5], "coordinate_system": "cartesian"},
            {"label": "Si2", "vector": [0.0, 0.0, -0.5], "coordinate_system": "cartesian"},
        ]
    }

    spec = structure_to_spec(structure)

    assert spec.atomic_spins == pytest.approx((1.5, -0.5))


def test_structure_to_spec_prefers_moments_when_atomic_spins_are_all_zero() -> None:
    structure = _host_structure()
    structure.properties["atomic_spins"] = [0.0, 0.0]
    structure.properties["magnetic_structure"] = {
        "moments": [
            {"label": "Si1", "vector": [0.0, 0.0, 2.0], "coordinate_system": "cartesian"},
            {"label": "Si2", "vector": [0.0, 0.0, -2.0], "coordinate_system": "cartesian"},
        ]
    }

    spec = structure_to_spec(structure)

    assert spec.atomic_spins == pytest.approx((2.0, -2.0))
