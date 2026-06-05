"""Additional coverage for CASTEP ``.param`` serialization branches.

These tests target branches of :mod:`mufinder.castep_io.param_writer` that are
not exercised by ``tests/test_castep_param_writer.py`` or
``tests/test_runtime_helper_coverage.py``:

* ``_render_block`` emitting a unit line.
* Block serialization when there are no preceding scalar keywords
  (no leading blank line).
* Multiple blocks separated by a blank line.
* The ``int`` and ``str`` fallthrough branches of ``_format_atom`` reached
  through ``write_param`` with real schema keywords.

All inputs use real schema keywords/blocks so ``validate_param_file`` passes,
keeping the tests deterministic with no network, filesystem, GUI, or sleeps.
"""

from __future__ import annotations

from mufinder.castep_io import KeywordAssignment, ParamFile, write_param
from mufinder.castep_io.param_writer import _render_block
from mufinder.castep_io.types import BlockSection


def test_render_block_emits_unit_line() -> None:
    """A block that carries a unit renders the unit on its own line."""

    block = BlockSection("devel_code", rows=(("foo=1",), ("bar=2",)), unit="eV")

    rendered = _render_block(block)

    assert rendered == (
        "%BLOCK DEVEL_CODE\n"
        "eV\n"
        "foo=1\n"
        "bar=2\n"
        "%ENDBLOCK DEVEL_CODE"
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
