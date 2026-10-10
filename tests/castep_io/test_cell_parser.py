"""Tests for CASTEP CELL and -out.cell parsing utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.cell_parser import (
    CellParseError,
    _lattice_from_abc,
    _parse_scalar_assignment,
    _parse_scalar_value,
    parse_cell_file,
    parse_cell_text,
)
from mufinder.castep_io.types import BlockSection, ValueWithUnit


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


_LATTICE = """
%BLOCK LATTICE_CART
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART
"""
_POSITIONS = """
%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
%ENDBLOCK POSITIONS_FRAC
"""


def _cell(*sections: str, lattice: str = _LATTICE, positions: str = _POSITIONS) -> str:
    return "\n".join((lattice, positions, *sections))


def test_parse_cell_file_wraps_os_errors(tmp_path: Path) -> None:
    with pytest.raises(CellParseError, match="Failed to read CELL file"):
        parse_cell_file(tmp_path / "missing.cell")


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ("%BLOCK SPECIES_MASS\n%BLOCK HUBBARD_U\n", "Nested %BLOCK is not allowed"),
        ("%ENDBLOCK SPECIES_MASS", "Found %ENDBLOCK without matching %BLOCK"),
        ("%BLOCK SPECIES_MASS\nSi 28.0\n%ENDBLOCK HUBBARD_U", "Mismatched block terminator"),
        (_LATTICE, "Duplicate block LATTICE_CART is not supported"),
        ("%BLOCK SPECIES_MASS\nSi 28.0", "Unterminated block"),
    ],
)
def test_parse_cell_text_rejects_malformed_block_structure(section: str, message: str) -> None:
    with pytest.raises(CellParseError, match=message):
        parse_cell_text(_cell(section))


def test_parse_cell_text_strips_comments_and_keeps_generic_blocks() -> None:
    parsed = parse_cell_text(
        _cell(
            "! ignored comment",
            "%BLOCK SPECIES_MASS\nbohr\n1 2 3 # inline comment\n%ENDBLOCK SPECIES_MASS",
            "KPOINT_MP_GRID 2 2 2 ! trailing",
        )
    )

    assert parsed.cell_file.blocks == (BlockSection(name="species_mass", rows=((1, 2, 3),), unit="bohr"),)
    assert parsed.cell_file.kpoint_mp_grid == (2, 2, 2)


def test_generic_block_rows_parse_units_and_typed_tokens() -> None:
    with_unit = parse_cell_text(
        _cell("%BLOCK SPECIES_MASS\neV/ang\nTRUE .false. 12 1.25 token\n%ENDBLOCK SPECIES_MASS")
    )
    assert with_unit.cell_file.blocks[0].unit == "eV/ang"
    assert with_unit.cell_file.blocks[0].rows == ((True, False, 12, 1.25, "token"),)

    without_unit = parse_cell_text(
        _cell("%BLOCK SPECIES_MASS\ncustom\nalpha beta\n%ENDBLOCK SPECIES_MASS")
    )
    assert without_unit.cell_file.blocks[0].unit is None
    assert without_unit.cell_file.blocks[0].rows == (("custom",), ("alpha", "beta"))


def test_lattice_blocks_scale_by_their_length_unit() -> None:
    cart = parse_cell_text(
        _cell(lattice="%BLOCK LATTICE_CART\nbohr\n1 0 0\n0 1 0\n0 0 1\n%ENDBLOCK LATTICE_CART")
    )
    assert cart.lattice_vectors[0][0] == pytest.approx(0.529177210903)
    assert cart.cell_file.lattice_cart is not None
    assert cart.cell_file.lattice_cart.unit == "bohr"

    abc = parse_cell_text(
        _cell(lattice="%BLOCK LATTICE_ABC\nnm\n1 1 1\n90 90 90\n%ENDBLOCK LATTICE_ABC")
    )
    assert abc.cell_file.lattice_abc is not None
    assert abc.cell_file.lattice_abc.lengths == pytest.approx((10.0, 10.0, 10.0))
    assert abc.cell_file.lattice_abc.unit == "nm"
    assert abc.lattice_vectors[0] == pytest.approx((10.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("lattice", "message"),
    [
        ("%BLOCK LATTICE_CART\n1 0 0\n0 1 0\n%ENDBLOCK LATTICE_CART", "must contain three lattice vectors"),
        ("%BLOCK LATTICE_CART\n1 0\n0 1 0\n0 0 1\n%ENDBLOCK LATTICE_CART", "must contain three values"),
        ("%BLOCK LATTICE_ABC\n1 1 1\n%ENDBLOCK LATTICE_ABC", "must contain lengths and angles rows"),
        ("%BLOCK LATTICE_ABC\n1 1\n90 90 90\n%ENDBLOCK LATTICE_ABC", "must contain three values"),
        ("", "must contain LATTICE_CART or LATTICE_ABC"),
    ],
)
def test_parse_cell_text_rejects_malformed_lattice(lattice: str, message: str) -> None:
    with pytest.raises(CellParseError, match=message):
        parse_cell_text(_cell(lattice=lattice))


def test_lattice_from_abc_requires_a_definition() -> None:
    # Defensive guard: parse_cell_text rejects a missing lattice before reaching it.
    with pytest.raises(CellParseError, match="Missing LATTICE_ABC definition"):
        _lattice_from_abc(None)


def test_positions_frac_reads_explicit_spins() -> None:
    parsed = parse_cell_text(
        _cell(positions="%BLOCK POSITIONS_FRAC\nFe 0 0 0 spin= 1.5\nFe 0.5 0.5 0.5\n%ENDBLOCK POSITIONS_FRAC")
    )

    assert parsed.cell_file.positions_abs is None
    assert parsed.atomic_spins == pytest.approx((1.5, 0.0))


def test_positions_abs_scales_by_its_length_unit() -> None:
    parsed = parse_cell_text(
        _cell(positions="%BLOCK POSITIONS_ABS\nbohr\nSi 1 0 0\n%ENDBLOCK POSITIONS_ABS")
    )

    positions_abs = parsed.cell_file.positions_abs
    assert parsed.cell_file.positions_frac is None
    assert parsed.atomic_spins is None
    assert positions_abs is not None
    assert positions_abs.unit == "bohr"
    assert positions_abs.positions[0].coordinates[0] == pytest.approx(0.529177210903)


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (
            _POSITIONS + "%BLOCK POSITIONS_ABS\nSi 0 0 0\n%ENDBLOCK POSITIONS_ABS",
            "may not contain both POSITIONS_FRAC and POSITIONS_ABS",
        ),
        ("", "must contain POSITIONS_FRAC or POSITIONS_ABS"),
        ("%BLOCK POSITIONS_FRAC\nFe 0 0 0 spin=not-a-number\n%ENDBLOCK POSITIONS_FRAC", "Invalid SPIN value"),
        (
            "%BLOCK POSITIONS_FRAC\nFe 0 0\n%ENDBLOCK POSITIONS_FRAC",
            "POSITIONS_FRAC rows must contain species and three coordinates",
        ),
        (
            "%BLOCK POSITIONS_ABS\nSi 0 0\n%ENDBLOCK POSITIONS_ABS",
            "POSITIONS_ABS rows must contain species and three coordinates",
        ),
    ],
)
def test_parse_cell_text_rejects_malformed_positions(positions: str, message: str) -> None:
    with pytest.raises(CellParseError, match=message):
        parse_cell_text(_cell(positions=positions))


def test_species_pot_and_kpoints_list_rows_are_parsed() -> None:
    parsed = parse_cell_text(
        _cell(
            "%BLOCK SPECIES_POT\nSi Si.recpot\n%ENDBLOCK SPECIES_POT",
            "%BLOCK KPOINTS_LIST\n0 0 0 1.0\n%ENDBLOCK KPOINTS_LIST",
        )
    )

    assert parsed.cell_file.species_pot[0].potential == "Si.recpot"
    assert parsed.cell_file.kpoints_list[0].weight == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ("%BLOCK SPECIES_POT\nSi\n%ENDBLOCK SPECIES_POT", "SPECIES_POT rows must contain species and potential"),
        ("%BLOCK KPOINTS_LIST\n0 0 0\n%ENDBLOCK KPOINTS_LIST", "KPOINTS_LIST rows must contain kx ky kz and weight"),
        ("FIX_ALL_CELL MAYBE", "Invalid boolean value"),
        ("KPOINT_MP_GRID 1 2", "requires three integer values"),
    ],
)
def test_parse_cell_text_rejects_malformed_rows_and_scalars(section: str, message: str) -> None:
    with pytest.raises(CellParseError, match=message):
        parse_cell_text(_cell(section))


def test_scalar_assignments_accept_bare_flags_colons_and_units() -> None:
    parsed = parse_cell_text(
        _cell(
            "SYMMETRY_GENERATE",
            "UNIT_CELL: a b c",
            "KPOINT_MP_SPACING 0.05 1/ang",
            "KPOINT_MP_OFFSET 0.25 0.25 0",
        )
    )

    assert parsed.cell_file.symmetry_generate is True
    assert {item.name: item.value for item in parsed.cell_file.scalars} == {
        "unit_cell": "a b c",
        "kpoint_mp_offset": (0.25, 0.25, 0),
    }
    assert parsed.cell_file.kpoint_mp_spacing == ValueWithUnit(0.05, "1/ang")


def test_scalar_values_follow_numeric_keyword_types() -> None:
    # The built-in cell keyword set has no plain int or number keywords; a full
    # CASTEP catalog does, so exercise those value types directly.
    int_spec = type("Spec", (), {"value_type": "int", "name": "nextra_bands"})()
    number_spec = type("Spec", (), {"value_type": "number", "name": "spin"})()

    assert _parse_scalar_value("7", int_spec) == 7
    assert _parse_scalar_value("1.25", number_spec) == pytest.approx(1.25)
    assert _parse_scalar_assignment("") is None


def test_positions_abs_on_a_hexagonal_cell_face_wraps_to_zero() -> None:
    # -b of a hexagonal cell converts to a fractional x a rounding error below
    # zero; a single ``mod`` would leave it at exactly 1.0.
    text = """
%BLOCK LATTICE_ABC
3.0 3.0 5.0
90.0 90.0 120.0
%ENDBLOCK LATTICE_ABC

%BLOCK POSITIONS_ABS
Si 1.4999999999999993 -2.598076211353316 0.0
%ENDBLOCK POSITIONS_ABS
"""

    parsed = parse_cell_text(text)

    assert all(0.0 <= value < 1.0 for value in parsed.fractional_positions[0])
    assert parsed.fractional_positions[0] == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


@pytest.mark.parametrize(
    ("block", "body"),
    [
        ("LATTICE_CART", "furlong\n4 0 0\n0 4 0\n0 0 4"),
        ("LATTICE_ABC", "furlong\n4 4 4\n90 90 90"),
        ("POSITIONS_ABS", "furlong\nSi 0 0 0"),
    ],
)
def test_unsupported_length_unit_is_named(block: str, body: str) -> None:
    lattice = "%BLOCK LATTICE_CART\n4 0 0\n0 4 0\n0 0 4\n%ENDBLOCK LATTICE_CART\n"
    positions = "%BLOCK POSITIONS_FRAC\nSi 0 0 0\n%ENDBLOCK POSITIONS_FRAC\n"
    block_text = f"%BLOCK {block}\n{body}\n%ENDBLOCK {block}\n"
    text = block_text + (lattice if block == "POSITIONS_ABS" else positions)

    with pytest.raises(CellParseError, match=f"{block} uses unsupported length unit 'furlong'"):
        parse_cell_text(text)


def test_single_number_first_row_is_a_short_row_not_a_unit() -> None:
    text = (
        "%BLOCK LATTICE_CART\n4\n0 4 0\n0 0 4\n%ENDBLOCK LATTICE_CART\n"
        "%BLOCK POSITIONS_FRAC\nSi 0 0 0\n%ENDBLOCK POSITIONS_FRAC\n"
    )

    with pytest.raises(CellParseError, match="LATTICE_CART rows must contain three values"):
        parse_cell_text(text)


_GEOMETRY_WITH_SETTINGS = """
%BLOCK LATTICE_CART
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0 SPIN=1.0
%ENDBLOCK POSITIONS_FRAC

%BLOCK SPECIES_POT
Si Si_00.usp
%ENDBLOCK SPECIES_POT

%BLOCK SPECIES_LCAO_STATES
Si 'unbalanced
%ENDBLOCK SPECIES_LCAO_STATES

KPOINT_MP_GRID 4 4 4
FIX_VOL : false
"""


def test_parse_cell_text_rejects_malformed_non_geometry_content_by_default() -> None:
    with pytest.raises(ValueError):
        parse_cell_text(_GEOMETRY_WITH_SETTINGS)
    with pytest.raises(CellParseError, match="fix_vol"):
        parse_cell_text(_GEOMETRY_WITH_SETTINGS.replace("Si 'unbalanced", "Si 2"))


def test_geometry_only_parse_skips_every_non_geometry_keyword() -> None:
    parsed = parse_cell_text(_GEOMETRY_WITH_SETTINGS, geometry_only=True)

    assert parsed.species == ("Si",)
    assert parsed.atomic_spins == (1.0,)
    assert parsed.cell_file.lattice_cart is not None
    assert parsed.cell_file.species_pot == ()
    assert parsed.cell_file.kpoint_mp_grid is None
    assert parsed.cell_file.scalars == ()
    assert parsed.cell_file.blocks == ()


def test_geometry_only_parse_still_checks_geometry_and_block_delimiters(tmp_path: Path) -> None:
    with pytest.raises(CellParseError, match="Unterminated block"):
        parse_cell_text(_GEOMETRY_WITH_SETTINGS + "%BLOCK SPECIES_MASS\nSi 28.0\n", geometry_only=True)

    with pytest.raises(CellParseError, match="must contain POSITIONS_FRAC or POSITIONS_ABS"):
        parse_cell_text(
            "%BLOCK LATTICE_CART\n4 0 0\n0 4 0\n0 0 4\n%ENDBLOCK LATTICE_CART\nFIX_VOL : false\n",
            geometry_only=True,
        )

    path = tmp_path / "relaxed-out.cell"
    path.write_text(_GEOMETRY_WITH_SETTINGS, encoding="utf-8")
    assert parse_cell_file(path, geometry_only=True).species == ("Si",)
