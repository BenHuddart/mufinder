"""Tests for CASTEP CELL and -out.cell parsing utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.cell_parser import CellParseError, parse_cell_file, parse_cell_text


def test_parse_cell_text_extracts_geometry_and_scalars() -> None:
    text = """
%BLOCK LATTICE_CART
ang
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
Si 0.25 0.25 0.25
%ENDBLOCK POSITIONS_FRAC

KPOINT_MP_GRID 4 4 4
FIX_ALL_CELL TRUE
"""

    parsed = parse_cell_text(text)

    assert parsed.lattice_vectors[0] == pytest.approx((4.0, 0.0, 0.0))
    assert parsed.lattice_vectors[1] == pytest.approx((0.0, 4.0, 0.0))
    assert parsed.lattice_vectors[2] == pytest.approx((0.0, 0.0, 4.0))
    assert parsed.species == ("Si", "Si")
    assert parsed.fractional_positions[1] == pytest.approx((0.25, 0.25, 0.25))
    assert parsed.cell_file.kpoint_mp_grid == (4, 4, 4)
    assert any(item.name == "fix_all_cell" and item.value is True for item in parsed.cell_file.scalars)


def test_parse_cell_file_reads_from_disk(tmp_path: Path) -> None:
    cell_path = tmp_path / "simple.cell"
    cell_path.write_text(
        """
%BLOCK LATTICE_CART
ang
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_ABS
Si 0.0 0.0 0.0
%ENDBLOCK POSITIONS_ABS
""",
        encoding="utf-8",
    )

    parsed = parse_cell_file(cell_path)

    assert parsed.species == ("Si",)
    assert parsed.fractional_positions[0] == pytest.approx((0.0, 0.0, 0.0))


def test_parse_cell_text_preserves_hubbard_u_unit_and_rows() -> None:
    parsed = parse_cell_text(
        """
%BLOCK LATTICE_CART
ang
5 0 0
0 5 0
0 0 5
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Sm 0 0 0
Ni 0.5 0.5 0.5
%ENDBLOCK POSITIONS_FRAC

%BLOCK HUBBARD_U
eV
Sm 1 f: 6.1
Ni d: 2.4
%ENDBLOCK HUBBARD_U
"""
    )

    assert parsed.cell_file.blocks[0].name == "hubbard_u"
    assert parsed.cell_file.blocks[0].unit == "eV"
    assert parsed.cell_file.blocks[0].rows == (
        ("Sm", 1, "f:", 6.1),
        ("Ni", "d:", 2.4),
    )


def test_parse_cell_text_rejects_multiple_lattice_blocks() -> None:
    text = """
%BLOCK LATTICE_CART
ang
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART

%BLOCK LATTICE_ABC
ang
4.0 4.0 4.0
90.0 90.0 90.0
%ENDBLOCK LATTICE_ABC

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
%ENDBLOCK POSITIONS_FRAC
"""

    with pytest.raises(CellParseError, match="may not contain both LATTICE_CART and LATTICE_ABC"):
        parse_cell_text(text)


def test_parse_cell_text_allows_empty_spin_marker_in_positions_frac() -> None:
    text = """
%BLOCK LATTICE_CART
ang
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Fe 0.0 0.0 0.0 SPIN=
Fe 0.5 0.5 0.5
%ENDBLOCK POSITIONS_FRAC
"""

    parsed = parse_cell_text(text)

    assert parsed.species == ("Fe", "Fe")
    assert parsed.atomic_spins is None


def test_parse_cell_text_parses_text_scalar_values_as_text() -> None:
    text = """
%BLOCK LATTICE_CART
ang
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
%ENDBLOCK POSITIONS_FRAC

UNIT_CELL 0.0 0.0 0.0
"""

    parsed = parse_cell_text(text)

    scalar_map = {assignment.name: assignment.value for assignment in parsed.cell_file.scalars}
    assert scalar_map["unit_cell"] == "0.0 0.0 0.0"


def test_positions_abs_accepts_a_castep_species_label() -> None:
    """``H:mu`` is a CASTEP species *label*, not a chemical symbol.

    The absolute-position branch used to convert coordinates by round-tripping
    through ``ase.Atoms``, which resolves every symbol against the periodic
    table and raised ``KeyError`` on the muon label. Any muonated ``-out.cell``
    written with ``POSITIONS_ABS`` was therefore unparseable.
    """

    text = """
%BLOCK LATTICE_CART
ang
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_ABS
ang
Si 0.0 0.0 0.0
H:mu 1.0 2.0 3.0
%ENDBLOCK POSITIONS_ABS
"""

    parsed = parse_cell_text(text)

    assert parsed.species == ("Si", "H:mu")
    assert parsed.fractional_positions[1] == pytest.approx((0.25, 0.5, 0.75))


def test_positions_abs_wraps_coordinates_into_the_cell() -> None:
    text = """
%BLOCK LATTICE_CART
ang
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_ABS
ang
H:mu 5.0 -1.0 0.0
%ENDBLOCK POSITIONS_ABS
"""

    parsed = parse_cell_text(text)

    assert parsed.fractional_positions[0] == pytest.approx((0.25, 0.75, 0.0))
