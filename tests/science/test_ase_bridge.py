"""Tests for the structure-to-ASE converter in :mod:`mufinder.science.ase_bridge`."""

from __future__ import annotations

from ase import Atoms
import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.ase_bridge import (
    StructureConversionError,
    element_symbol,
    is_muon_label,
    host_elements,
    reduced_formula,
    structure_to_ase_atoms,
)

_LATTICE = [[4.0, 0.0, 0.0], [0.5, 4.5, 0.0], [0.2, 0.3, 5.0]]


def _structure(species: list[str], **properties) -> Structure:
    return Structure(
        label="host",
        lattice_vectors=[list(row) for row in _LATTICE],
        species=species,
        fractional_positions=[[0.1 * index, 0.2, 0.3] for index in range(len(species))],
        properties=dict(properties),
    )


def test_default_conversion_matches_the_plain_atoms_construction_it_replaced() -> None:
    structure = _structure(["Fe", "O", "O"])

    atoms = structure_to_ase_atoms(structure)

    expected = Atoms(
        symbols=list(structure.species),
        scaled_positions=np.asarray(structure.fractional_positions, dtype=float),
        cell=np.asarray(structure.lattice_vectors, dtype=float),
        pbc=[True, True, True],
    )
    assert atoms.get_chemical_symbols() == expected.get_chemical_symbols()
    assert np.array_equal(atoms.cell.array, expected.cell.array)
    assert np.array_equal(atoms.get_positions(), expected.get_positions())
    assert atoms.pbc.tolist() == [True, True, True]
    assert "mufinder_species_labels" not in atoms.info


def test_structure_like_objects_convert() -> None:
    host = type(
        "_Host",
        (),
        {
            "lattice_vectors": tuple(tuple(row) for row in _LATTICE),
            "species": ("Si",),
            "fractional_positions": ((0.5, 0.5, 0.5),),
        },
    )()

    atoms = structure_to_ase_atoms(host)

    assert atoms.get_chemical_symbols() == ["Si"]
    assert atoms.get_scaled_positions()[0].tolist() == pytest.approx([0.5, 0.5, 0.5])


@pytest.mark.parametrize(
    ("host", "message"),
    [
        (
            type(
                "_Partial",
                (),
                {"lattice_vectors": _LATTICE, "species": ("Si",), "fractional_positions": None},
            )(),
            "must define lattice_vectors, species",
        ),
        (
            type(
                "_Mismatched",
                (),
                {
                    "lattice_vectors": _LATTICE,
                    "species": ("Si", "Si"),
                    "fractional_positions": ((0.0, 0.0, 0.0),),
                },
            )(),
            "matching lengths",
        ),
        (
            type(
                "_Flat",
                (),
                {
                    "lattice_vectors": [[1.0, 0.0, 0.0]],
                    "species": ("Si",),
                    "fractional_positions": ((0.0, 0.0, 0.0),),
                },
            )(),
            "3x3 matrix",
        ),
    ],
    ids=["missing-positions", "mismatched-lengths", "bad-lattice"],
)
def test_malformed_structures_raise_the_callers_error(host: object, message: str) -> None:
    class CallerError(ValueError):
        pass

    with pytest.raises(StructureConversionError, match=message):
        structure_to_ase_atoms(host)
    with pytest.raises(CallerError, match=message):
        structure_to_ase_atoms(host, error=CallerError)


def test_skip_muon_drops_muon_sites_and_records_the_kept_labels() -> None:
    structure = _structure(["Fe1", "H:mu", "O"])

    atoms = structure_to_ase_atoms(
        structure, skip_muon=True, sanitize_symbols=True, record_species_labels=True
    )

    assert atoms.get_chemical_symbols() == ["Fe", "O"]
    assert atoms.info["mufinder_species_labels"] == ["Fe1", "O"]
    assert atoms.get_scaled_positions()[1].tolist() == pytest.approx([0.2, 0.2, 0.3])


def test_sanitize_symbols_maps_the_muon_to_hydrogen() -> None:
    atoms = structure_to_ase_atoms(
        _structure(["O", "mu"]), sanitize_symbols=True, record_species_labels=True
    )

    assert atoms.get_chemical_symbols() == ["O", "H"]
    assert atoms.info["mufinder_species_labels"] == ["O", "mu"]


def test_unsanitized_labels_reach_ase_unchanged() -> None:
    with pytest.raises(KeyError):
        structure_to_ase_atoms(_structure(["Fe1"]))


@pytest.mark.parametrize("label", ["X", "h:mu", " Mu ", "MUON"])
def test_muon_labels_are_case_and_space_insensitive(label: str) -> None:
    assert is_muon_label(label) is True


def test_element_symbol_reads_the_leading_element_and_rejects_unreadable_labels() -> None:
    class CallerError(ValueError):
        pass

    assert element_symbol("Fe12") == "Fe"
    assert element_symbol(" O ") == "O"
    with pytest.raises(StructureConversionError, match="must not be empty"):
        element_symbol("   ")
    with pytest.raises(CallerError, match="Could not resolve element"):
        element_symbol("123", error=CallerError)


def test_host_elements_lists_distinct_elements_in_first_seen_order() -> None:
    assert host_elements(["Fe1", "O", "Fe2", "H:mu", "123", "O"]) == ("Fe", "O")
    assert host_elements(["X", "muon"]) == ()


def test_reduced_formula_counts_formula_units_and_skips_the_muon_and_unreadable_labels() -> None:
    assert reduced_formula(["Fe1", "Fe2", "O", "O", "O", "Fe", "Fe", "O", "O", "O", "H:mu"]) == (
        "Fe2O3",
        2,
    )
    assert reduced_formula(["Sr", "Ti", "O", "O", "O", "123"]) == ("SrTiO3", 1)
    assert reduced_formula(["X", "muon"]) == ("", 0)
