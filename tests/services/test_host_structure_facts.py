"""The facts the Host Structure page states about a structure."""

from __future__ import annotations

import pytest

from mufinder.science.supercell import build_supercell_structure
from mufinder.services.host_structure_facts import structure_facts, supercell_facts
from tests.support.structures import hexagonal_magnesium_oxide_host, host_structure
from tests.support.symmetry_hosts import cof2_host


def test_supercell_facts_give_atoms_image_distance_and_cell_lengths() -> None:
    supercell, _spins = build_supercell_structure(cof2_host(), [[2, 0, 0], [0, 2, 0], [0, 0, 3]])

    facts = supercell_facts(supercell)

    # a = b = 2 × 4.695 Å and c = 3 × 3.18 Å; the nearest image is one a edge away.
    assert facts == (
        ("Atoms", "72"),
        ("Image distance", "9.39 Å"),
        ("a", "9.3900 Å"),
        ("b", "9.3900 Å"),
        ("c", "9.5400 Å"),
    )


def test_supercell_facts_measure_a_skewed_cell_by_its_shortest_translation() -> None:
    skewed, _spins = build_supercell_structure(
        host_structure(lattice=4.0), [[3, 0, 0], [2, 1, 0], [0, 0, 3]]
    )

    facts = dict(supercell_facts(skewed))

    # The edges are 12, √80 and 12 Å, but A2 - A1 = (-1, 1, 0) cells is only √32 Å.
    assert facts["b"] == "8.9443 Å"
    assert facts["Image distance"] == "5.66 Å"


def test_supercell_facts_find_an_image_a_sheared_basis_hides() -> None:
    sheared, _spins = build_supercell_structure(
        host_structure(lattice=4.0), [[5, 2, 0], [2, 1, 0], [0, 0, 3]]
    )

    # A1 = (20, 8, 0) and A2 = (8, 4, 0) Å; A1 - 2 A2 = (4, 0, 0) is the nearest image,
    # which a search over coefficients of ±1 in this basis misses (it finds 8.94 Å).
    assert dict(supercell_facts(sheared))["Image distance"] == "4.00 Å"


def test_supercell_facts_of_a_cell_with_a_zero_length_vector_give_the_atoms_alone() -> None:
    flat = host_structure(lattice=[[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])

    assert supercell_facts(flat) == (("Atoms", "2"),)


def test_cof2_has_its_whole_cell_formula_space_group_and_lattice() -> None:
    facts = structure_facts(cof2_host())

    assert facts.formula == "Co2F4"
    assert facts.atom_count == 6
    assert (facts.space_group_symbol, facts.space_group_number) == ("P4_2/mnm", 136)
    assert facts.lattice is not None
    assert (facts.lattice.a, facts.lattice.b, facts.lattice.c) == pytest.approx(
        (4.695, 4.695, 3.18)
    )
    assert (facts.lattice.alpha, facts.lattice.beta, facts.lattice.gamma) == pytest.approx(
        (90.0, 90.0, 90.0)
    )
    assert facts.lattice.volume == pytest.approx(4.695 * 4.695 * 3.18)


def test_facts_are_two_groups_of_captioned_values_with_units_on_the_values() -> None:
    groups = structure_facts(cof2_host()).groups("imports/CoF2.cif")

    assert groups == (
        (
            ("Formula", "Co2F4"),
            ("Atoms", "6"),
            ("Space group", "P4₂/mnm (136)"),
            ("Source", "imports/CoF2.cif"),
        ),
        (
            ("a", "4.6950 Å"),
            ("b", "4.6950 Å"),
            ("c", "3.1800 Å"),
            ("α", "90°"),
            ("β", "90°"),
            ("γ", "90°"),
            ("Volume", "70.10 Å³"),
        ),
    )


def test_an_oblique_angle_keeps_its_decimals() -> None:
    contents, lattice = structure_facts(hexagonal_magnesium_oxide_host()).groups("")

    assert dict(lattice)["γ"] == "120°"
    assert dict(lattice)["Volume"] == "103.88 Å³"
    assert "Source" not in dict(contents)


def test_a_structure_spglib_cannot_classify_has_no_space_group() -> None:
    """Two atoms on one site: spglib refuses the cell, and the page leaves the group out."""

    overlapping = host_structure(species=("Si", "Si"), positions=((0.0, 0.0, 0.0),) * 2)

    facts = structure_facts(overlapping)

    assert facts.space_group_symbol is None
    assert facts.space_group_text() is None
    assert facts.groups("imports/host.cif")[0] == (
        ("Formula", "Si2"),
        ("Atoms", "2"),
        ("Source", "imports/host.cif"),
    )


def test_a_species_that_names_no_element_has_no_space_group_and_stays_out_of_the_formula() -> None:
    facts = structure_facts(host_structure(species=("Si", "123"), positions=((0, 0, 0),) * 2))

    assert facts.space_group_number is None
    assert facts.formula == "Si"
    assert facts.atom_count == 2


def test_a_cell_with_a_zero_length_vector_has_no_lattice_line() -> None:
    flat = host_structure(
        lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 0]], species=("Si",), positions=((0, 0, 0),)
    )

    facts = structure_facts(flat)

    assert facts.lattice is None
    assert len(facts.groups("flat")) == 1


def test_a_symmetry_search_that_fails_leaves_the_space_group_out(monkeypatch) -> None:
    import mufinder.services.host_structure_facts as module

    def failing(_structure):
        raise RuntimeError("spglib refused the cell")

    monkeypatch.setattr(module, "detect_space_group", failing)

    facts = structure_facts(cof2_host())

    assert facts.space_group_number is None
    assert dict(facts.groups("s")[0]) == {"Formula": "Co2F4", "Atoms": "6", "Source": "s"}
