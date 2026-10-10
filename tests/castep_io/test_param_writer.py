"""Tests for CASTEP ``.param`` serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io import KeywordAssignment, ParamFile, ValueWithUnit, geometry_optimization, write_param
from mufinder.castep_io.param_writer import _render_value
from mufinder.castep_io.types import BlockSection


def test_write_param_serializes_ordered_keywords() -> None:
    param_file = ParamFile(
        keywords=(
            KeywordAssignment("geom_force_tol", ValueWithUnit(0.05, "eV/ang")),
            KeywordAssignment("task", "GEOMETRYOPTIMIZATION"),
            KeywordAssignment("xc_functional", "PBE"),
            KeywordAssignment("cut_off_energy", ValueWithUnit(340.0, "eV")),
        )
    )

    rendered = write_param(param_file)
    lines = rendered.splitlines()

    assert lines[0] == "TASK : GEOMETRYOPTIMIZATION"
    assert lines[1] == "XC_FUNCTIONAL : PBE"
    assert "CUT_OFF_ENERGY : 340 eV" in rendered
    assert "GEOM_FORCE_TOL : 0.05 eV/ang" in rendered


def test_geometry_optimization_preset_matches_golden_file() -> None:
    param_file = geometry_optimization(
        cut_off_energy=340.0,
        max_scf_cycles=60,
        geom_energy_tol=ValueWithUnit(2e-5, "eV"),
        geom_force_tol=ValueWithUnit(0.05, "eV/ang"),
        geom_stress_tol=ValueWithUnit(0.1, "GPa"),
        geom_max_iter=30,
    )
    param_file.keywords += (KeywordAssignment("elec_energy_tol", ValueWithUnit(1e-6, "eV")),)

    rendered = write_param(param_file)
    golden = Path("tests/golden/si_geom_opt.param").read_text(encoding="utf-8")

    assert rendered == golden


def test_write_param_supports_boolean_spin_keywords() -> None:
    param_file = ParamFile(
        keywords=(
            KeywordAssignment("task", "SINGLEPOINT"),
            KeywordAssignment("spin_polarized", True),
            KeywordAssignment("smearing_width", ValueWithUnit(0.1, "eV")),
        )
    )

    rendered = write_param(param_file)

    assert "SPIN_POLARIZED : TRUE" in rendered
    assert "SMEARING_WIDTH : 0.1 eV" in rendered
    assert write_param(ParamFile(keywords=(KeywordAssignment("spin_polarized", False),))) == (
        "SPIN_POLARIZED : FALSE\n"
    )


def test_write_param_renders_bare_int_scalar() -> None:
    """A bare integer keyword value uses the integer formatting branch."""

    param_file = ParamFile(keywords=(KeywordAssignment("iprint", 2),))

    rendered = write_param(param_file)

    assert rendered == "IPRINT : 2\n"


def test_write_param_renders_plain_text_scalar() -> None:
    """A plain string keyword value uses the ``str`` fallthrough branch."""

    param_file = ParamFile(keywords=(KeywordAssignment("length_unit", "ang"),))

    rendered = write_param(param_file)

    assert rendered == "LENGTH_UNIT : ang\n"


def test_tuple_values_render_space_separated_and_none_is_rejected() -> None:
    # No keyword in the built-in param set takes a vector value, so validation
    # stops write_param before these branches; a full CASTEP catalog reaches them.
    assert _render_value(ValueWithUnit((1, 2, 3), "bohr")) == "1 2 3 bohr"
    assert _render_value((True, 3.5, "A")) == "TRUE 3.5 A"
    with pytest.raises(ValueError, match="may not be None"):
        _render_value(None)


def test_write_param_renders_block_unit_line() -> None:
    """A block that carries a unit renders the unit on its own line."""

    param_file = ParamFile(
        keywords=(),
        blocks=(BlockSection("devel_code", rows=(("foo=1",), ("bar=2",)), unit="eV"),),
    )

    rendered = write_param(param_file)

    assert rendered == (
        "%BLOCK DEVEL_CODE\n"
        "eV\n"
        "foo=1\n"
        "bar=2\n"
        "%ENDBLOCK DEVEL_CODE\n"
    )


def test_write_param_block_only_has_no_leading_blank_line() -> None:
    """With no scalar keywords the block output starts immediately."""

    param_file = ParamFile(
        keywords=(),
        blocks=(BlockSection("devel_code", rows=(("x=1",),)),),
    )

    rendered = write_param(param_file)

    assert not rendered.startswith("\n")
    assert rendered == (
        "%BLOCK DEVEL_CODE\n"
        "x=1\n"
        "%ENDBLOCK DEVEL_CODE\n"
    )


def test_write_param_separates_multiple_blocks_with_blank_line() -> None:
    """Consecutive blocks are separated by a single blank line."""

    param_file = ParamFile(
        keywords=(),
        blocks=(
            BlockSection("devel_code", rows=(("a=1",),)),
            BlockSection("xc_definition", rows=(("PBE", "1.0"),)),
        ),
    )

    rendered = write_param(param_file)

    assert rendered == (
        "%BLOCK DEVEL_CODE\n"
        "a=1\n"
        "%ENDBLOCK DEVEL_CODE\n"
        "\n"
        "%BLOCK XC_DEFINITION\n"
        "PBE 1.0\n"
        "%ENDBLOCK XC_DEFINITION\n"
    )


def test_write_param_inserts_blank_line_between_scalars_and_blocks() -> None:
    """A blank line separates scalar keywords from the first block."""

    param_file = ParamFile(
        keywords=(KeywordAssignment("iprint", 1),),
        blocks=(BlockSection("devel_code", rows=(("y=2",),)),),
    )

    rendered = write_param(param_file)

    assert rendered == (
        "IPRINT : 1\n"
        "\n"
        "%BLOCK DEVEL_CODE\n"
        "y=2\n"
        "%ENDBLOCK DEVEL_CODE\n"
    )
