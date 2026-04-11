"""Additional coverage for CASTEP cell parser internals."""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.cell_parser import (
    CellParseError,
    _lattice_from_abc,
    _looks_like_unit,
    _parse_generic_block_rows,
    _parse_generic_token,
    _parse_kpoints_list,
    _parse_lattice_abc,
    _parse_lattice_cart,
    _parse_positions,
    _parse_scalar_assignment,
    _parse_scalar_value,
    _parse_species_pot,
    _partition_cell_text,
    _strip_comment,
    parse_cell_file,
)
from mufinder.castep_io.types import LatticeABC, ValueWithUnit


def test_parse_cell_file_wraps_os_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.cell"
    with pytest.raises(CellParseError, match="Failed to read CELL file"):
        parse_cell_file(missing)


def test_partition_cell_text_covers_error_and_unknown_block_paths() -> None:
    with pytest.raises(CellParseError, match="Nested %BLOCK is not allowed"):
        _partition_cell_text("%BLOCK A\n%BLOCK B\n")

    with pytest.raises(CellParseError, match="Found %ENDBLOCK without matching %BLOCK"):
        _partition_cell_text("%ENDBLOCK A")

    with pytest.raises(CellParseError, match="Mismatched block terminator"):
        _partition_cell_text("%BLOCK A\n1 2 3\n%ENDBLOCK B")

    with pytest.raises(CellParseError, match="Duplicate block LATTICE_CART is not supported"):
        _partition_cell_text(
            "%BLOCK LATTICE_CART\n1 0 0\n0 1 0\n0 0 1\n%ENDBLOCK LATTICE_CART\n"
            "%BLOCK LATTICE_CART\n1 0 0\n0 1 0\n0 0 1\n%ENDBLOCK LATTICE_CART"
        )

    known, unknown, scalars = _partition_cell_text(
        """
        ! ignored comment
        %BLOCK CUSTOM
        bohr
        1 2 3 # inline comment
        %ENDBLOCK CUSTOM

        KPOINT_MP_GRID 2 2 2 ! trailing
        """
    )

    assert known == {}
    assert unknown == [("custom", ((1, 2, 3),), "bohr")]
    assert scalars == ["KPOINT_MP_GRID 2 2 2"]

    with pytest.raises(CellParseError, match="Unterminated block"):
        _partition_cell_text("%BLOCK A\n1 2 3")


def test_lattice_parsers_cover_scaling_and_error_paths() -> None:
    lattice_cart = _parse_lattice_cart(["bohr", "1 0 0", "0 1 0", "0 0 1"])
    assert lattice_cart is not None
    assert lattice_cart.vectors[0][0] == pytest.approx(0.529177210903)
    assert lattice_cart.unit == "bohr"

    with pytest.raises(CellParseError, match="must contain three lattice vectors"):
        _parse_lattice_cart(["1 0 0", "0 1 0"])

    with pytest.raises(CellParseError, match="must contain three values"):
        _parse_lattice_cart(["1 0", "0 1 0", "0 0 1"])

    lattice_abc = _parse_lattice_abc(["nm", "1 1 1", "90 90 90"])
    assert lattice_abc is not None
    assert lattice_abc.lengths == pytest.approx((10.0, 10.0, 10.0))
    assert lattice_abc.unit == "nm"

    with pytest.raises(CellParseError, match="must contain lengths and angles rows"):
        _parse_lattice_abc(["1 1 1"])

    with pytest.raises(CellParseError, match="must contain three values"):
        _parse_lattice_abc(["1 1", "90 90 90"])

    with pytest.raises(CellParseError, match="Missing LATTICE_ABC definition"):
        _lattice_from_abc(None)

    vectors = _lattice_from_abc(LatticeABC(lengths=(4.0, 4.0, 4.0), angles=(90.0, 90.0, 90.0)))
    assert vectors[0] == pytest.approx((4.0, 0.0, 0.0))


def test_position_and_row_parsers_cover_frac_abs_and_errors() -> None:
    lattice_vectors = ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))

    with pytest.raises(CellParseError, match="may not contain both POSITIONS_FRAC and POSITIONS_ABS"):
        _parse_positions(
            positions_frac_lines=["Si 0 0 0"],
            positions_abs_lines=["Si 0 0 0"],
            lattice_vectors=lattice_vectors,
        )

    with pytest.raises(CellParseError, match="must contain POSITIONS_FRAC or POSITIONS_ABS"):
        _parse_positions(
            positions_frac_lines=None,
            positions_abs_lines=None,
            lattice_vectors=lattice_vectors,
        )

    positions_frac, positions_abs, spins = _parse_positions(
        positions_frac_lines=["Fe 0 0 0 spin= 1.5", "Fe 0.5 0.5 0.5"],
        positions_abs_lines=None,
        lattice_vectors=lattice_vectors,
    )
    assert positions_abs is None
    assert positions_frac is not None
    assert spins == pytest.approx((1.5, 0.0))

    with pytest.raises(CellParseError, match="Invalid SPIN value"):
        _parse_positions(
            positions_frac_lines=["Fe 0 0 0 spin=not-a-number"],
            positions_abs_lines=None,
            lattice_vectors=lattice_vectors,
        )

    with pytest.raises(CellParseError, match="POSITIONS_FRAC rows must contain species and three coordinates"):
        _parse_positions(
            positions_frac_lines=["Fe 0 0"],
            positions_abs_lines=None,
            lattice_vectors=lattice_vectors,
        )

    positions_frac, positions_abs, spins = _parse_positions(
        positions_frac_lines=None,
        positions_abs_lines=["bohr", "Si 1 0 0"],
        lattice_vectors=lattice_vectors,
    )
    assert positions_frac is None
    assert spins is None
    assert positions_abs is not None
    assert positions_abs.unit == "bohr"
    assert positions_abs.positions[0].coordinates[0] == pytest.approx(0.529177210903)

    with pytest.raises(CellParseError, match="POSITIONS_ABS rows must contain species and three coordinates"):
        _parse_positions(
            positions_frac_lines=None,
            positions_abs_lines=["Si 0 0"],
            lattice_vectors=lattice_vectors,
        )

    with pytest.raises(CellParseError, match="SPECIES_POT rows must contain species and potential tokens"):
        _parse_species_pot(["Si"])
    assert _parse_species_pot(["Si Si.recpot"])[0].potential == "Si.recpot"

    with pytest.raises(CellParseError, match="KPOINTS_LIST rows must contain kx ky kz and weight"):
        _parse_kpoints_list(["0 0 0"])
    assert _parse_kpoints_list(["0 0 0 1.0"])[0].weight == pytest.approx(1.0)


def test_scalar_and_generic_parsers_cover_branchy_helpers() -> None:
    assert _parse_scalar_assignment("symmetry_generate") == {"name": "symmetry_generate", "value": None}
    assert _parse_scalar_assignment("unit_cell: a b c") == {"name": "unit_cell", "value": "a b c"}
    assert _parse_scalar_assignment("") is None

    with pytest.raises(CellParseError, match="Invalid boolean value"):
        _parse_scalar_value("MAYBE", type("Spec", (), {"value_type": "bool", "name": "spin_polarized"})())
    with pytest.raises(CellParseError, match="requires three integer values"):
        _parse_scalar_value("1 2", type("Spec", (), {"value_type": "int_triplet", "name": "kpoint_mp_grid"})())

    assert _parse_scalar_value(
        "450 eV",
        type("Spec", (), {"value_type": "physical", "name": "cut_off_energy"})(),
    ) == ValueWithUnit(450.0, "eV")
    assert _parse_scalar_value("7", type("Spec", (), {"value_type": "int", "name": "nextra_bands"})()) == 7
    assert _parse_scalar_value("1.25", type("Spec", (), {"value_type": "number", "name": "spin"})()) == pytest.approx(1.25)
    assert _parse_scalar_value("true false 3", None) == (True, False, 3)

    unit, rows = _parse_generic_block_rows(["ang", "1 2 3", "alpha beta"])
    assert unit == "ang"
    assert rows == ((1, 2, 3), ("alpha", "beta"))

    assert _parse_generic_token("TRUE") is True
    assert _parse_generic_token(".false.") is False
    assert _parse_generic_token("12") == 12
    assert _parse_generic_token("1.25") == pytest.approx(1.25)
    assert _parse_generic_token("token") == "token"
    assert _looks_like_unit("eV/ang") is True
    assert _looks_like_unit("custom") is False
    assert _strip_comment("value ! comment # later") == "value"
