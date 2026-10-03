"""Tests for CASTEP ``.param`` parsing utilities.

Every fixture here is hand-authored. Nothing is copied from a real calculation:
see ``docs/legacy_import.rst`` (Research data boundary).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.param_parser import (
    ParamParseError,
    parse_param_assignments,
    parse_param_file,
    parse_param_observations,
    parse_param_text,
    parse_param_value,
    run_is_spin_polarized,
)
from mufinder.castep_io.schema import get_keyword_spec
from mufinder.castep_io.types import ValueWithUnit


#: A ``.param`` in the shape the legacy MuFinder wrote. Hand-authored to that
#: documented shape, not taken from any calculation.
LEGACY_PARAM_TEXT = """task: geometryoptimisation
charge: +1
xc_functional: PBE
spin_polarized : false
basis_precision : FINE
write_cell_structure : true
geom_max_iter : 300
max_scf_cycles : 100
elec_energy_tol : 1E-6
metals_method : DM
continuation : default
"""


def test_parse_param_assignments_reads_a_legacy_shaped_file() -> None:
    assignments = parse_param_assignments(LEGACY_PARAM_TEXT)

    # The legacy writer used the British spelling, which CASTEP folds on read.
    assert assignments["task"] == "GEOMETRYOPTIMIZATION"
    assert assignments["charge"] == 1
    assert assignments["spin_polarized"] is False
    assert assignments["geom_max_iter"] == 300
    assert assignments["max_scf_cycles"] == 100
    assert assignments["elec_energy_tol"] == pytest.approx(1e-6)


def test_parse_param_assignments_keeps_keywords_the_schema_does_not_know() -> None:
    """A faithful view of the file includes keywords with no runtime spec.

    The runtime catalog is environment-dependent, so "unknown" is a statement
    about this machine rather than about the file. Dropping such keywords would
    silently lose what the file said.
    """

    assignments = parse_param_assignments("definitely_not_a_castep_keyword : 7\n")

    assert assignments["definitely_not_a_castep_keyword"] == 7


def test_parse_param_observations_separates_mapped_from_unmapped() -> None:
    observations = parse_param_observations(
        "task : SINGLEPOINT\ndefinitely_not_a_castep_keyword : 7\n"
    )

    mapped = {item.canonical_name for item in observations if item.mapped}
    unmapped = {item.canonical_name for item in observations if not item.mapped}

    assert "task" in mapped
    assert unmapped == {"definitely_not_a_castep_keyword"}


@pytest.mark.parametrize(
    "written, expected",
    [
        ("geometryoptimisation", "GEOMETRYOPTIMIZATION"),
        ("GEOMOPT", "GEOMETRYOPTIMIZATION"),
        ("geom", "GEOMETRYOPTIMIZATION"),
        ("spe", "SINGLEPOINT"),
        ("md", "MOLECULARDYNAMICS"),
        ("singlepoint", "SINGLEPOINT"),
    ],
)
def test_task_value_aliases_are_folded_the_way_castep_folds_them(
    written: str, expected: str
) -> None:
    """CASTEP normalises these spellings on read; a parse must match.

    Legacy MuFinder wrote ``geometryoptimisation``. Without folding, a legacy
    file parses to a value MuFinder's own validator rejects, so the rendered
    ``.param`` for an imported job could not be written at all.
    """

    assert parse_param_assignments(f"task : {written}\n")["task"] == expected


def test_an_unrecognised_enum_value_is_left_alone() -> None:
    """Never coerce a value CASTEP would not have accepted; report it as-is."""

    assert parse_param_assignments("task : not_a_real_task\n")["task"] == "not_a_real_task"


def test_parse_param_accepts_whitespace_separated_assignments() -> None:
    """CASTEP accepts ``NAME VALUE`` as well as ``NAME : VALUE``."""

    assert parse_param_assignments("MAX_SCF_CYCLES 40\n")["max_scf_cycles"] == 40


def test_parse_param_strips_comments_and_skips_block_contents() -> None:
    text = """
! a leading comment
task : SINGLEPOINT   # trailing comment
%BLOCK DEVEL_CODE
max_scf_cycles : 999
%ENDBLOCK DEVEL_CODE
max_scf_cycles : 40
"""

    assignments = parse_param_assignments(text)

    assert assignments["task"] == "SINGLEPOINT"
    # 999 lives inside a block and must not be read as an assignment.
    assert assignments["max_scf_cycles"] == 40


def test_parse_param_text_round_trips_through_the_writer() -> None:
    from mufinder.castep_io.param_writer import write_param

    parsed = parse_param_text(LEGACY_PARAM_TEXT)
    rendered = write_param(parsed.param_file)

    assert "TASK :" in rendered
    assert "MAX_SCF_CYCLES : 100" in rendered
    # Re-reading the rendered text must agree with the original parse on the
    # keywords the schema recognises.
    reparsed = parse_param_assignments(rendered)
    for name in ("charge", "geom_max_iter", "max_scf_cycles", "spin_polarized"):
        assert reparsed[name] == parse_param_assignments(LEGACY_PARAM_TEXT)[name]


def test_parse_param_text_preserves_blocks() -> None:
    text = """%BLOCK DEVEL_CODE
some raw directive
another one
%ENDBLOCK DEVEL_CODE
"""

    parsed = parse_param_text(text)

    assert len(parsed.param_file.blocks) == 1
    block = parsed.param_file.blocks[0]
    assert block.name == "devel_code"
    assert block.rows == (("some", "raw", "directive"), ("another", "one"))


@pytest.mark.parametrize(
    "text, message",
    [
        ("%BLOCK ONE\n%BLOCK TWO\n", "Nested"),
        ("%ENDBLOCK ONE\n", "without matching"),
        ("%BLOCK ONE\n%ENDBLOCK TWO\n", "Mismatched"),
        ("%BLOCK ONE\n", "Unterminated"),
    ],
)
def test_parse_param_text_rejects_malformed_blocks(text: str, message: str) -> None:
    with pytest.raises(ParamParseError, match=message):
        parse_param_text(text)


def test_strict_mode_raises_and_lenient_mode_records_the_failure() -> None:
    text = "spin_polarized : banana\n"

    with pytest.raises(ValueError):
        parse_param_assignments(text)

    observations = parse_param_observations(text, strict=False)
    assert len(observations) == 1
    assert observations[0].error is not None
    assert not observations[0].mapped
    # The raw text survives so a caller can report exactly what the file said.
    assert observations[0].raw_value == "banana"


def test_parse_param_file_reads_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "seed.param"
    path.write_text(LEGACY_PARAM_TEXT, encoding="utf-8")

    parsed = parse_param_file(path)

    assert any(item.name == "task" for item in parsed.param_file.keywords)


def test_parse_param_file_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ParamParseError, match="Failed to read"):
        parse_param_file(tmp_path / "absent.param")


class TestParseParamValue:
    """The spec drives coercion; without a spec the literal is inferred."""

    def test_bool_spec_accepts_castep_spellings(self) -> None:
        spec = get_keyword_spec("param", "spin_polarized")
        assert parse_param_value("TRUE", spec) is True
        assert parse_param_value(".false.", spec) is False

    def test_bool_spec_rejects_nonsense(self) -> None:
        spec = get_keyword_spec("param", "spin_polarized")
        with pytest.raises(ValueError, match="Invalid boolean"):
            parse_param_value("banana", spec)

    def test_physical_spec_keeps_an_explicit_unit(self) -> None:
        spec = get_keyword_spec("param", "cut_off_energy")
        assert parse_param_value("800 eV", spec) == ValueWithUnit(value=800.0, unit="eV")
        assert parse_param_value("800", spec) == pytest.approx(800.0)

    def test_int_triplet_spec_requires_three_values(self) -> None:
        spec = get_keyword_spec("cell", "kpoint_mp_grid")
        assert parse_param_value("2 3 4", spec) == (2, 3, 4)
        with pytest.raises(ValueError, match="three integer values"):
            parse_param_value("2 3", spec)

    def test_without_a_spec_a_bare_digit_stays_a_number(self) -> None:
        """``1`` is far more often a number than a boolean when untyped.

        With a bool spec it means true; without one, inferring a bool would
        corrupt any numeric keyword this machine's catalog happens to lack.
        """

        assert parse_param_value("1", None) == 1
        assert parse_param_value("true", None) is True

    def test_without_a_spec_multi_token_values_become_a_tuple(self) -> None:
        assert parse_param_value("2 3 4", None) == (2, 3, 4)
        assert parse_param_value("a b", None) == ("a", "b")

    def test_empty_value_is_none(self) -> None:
        assert parse_param_value("", None) is None


@pytest.mark.parametrize(
    ("text", "atomic_spins", "expected"),
    [
        pytest.param("SPIN_POLARIZED : TRUE", None, True, id="explicit-true"),
        pytest.param("spin_polarised : false", (0.0, 2.0), False, id="explicit-false-beats-atomic-spins"),
        pytest.param("SPIN : -2", None, True, id="negative-net-spin"),
        pytest.param("SPIN : 0", None, False, id="zero-net-spin"),
        pytest.param("TASK : SINGLEPOINT", (0.0, 2.0), True, id="atomic-spins"),
        pytest.param("TASK : SINGLEPOINT", (0.0, 0.0), False, id="zero-atomic-spins"),
        pytest.param("TASK : SINGLEPOINT", None, False, id="nothing"),
    ],
)
def test_run_is_spin_polarized_follows_castep_defaults(
    text: str, atomic_spins: tuple[float, ...] | None, expected: bool
) -> None:
    assert run_is_spin_polarized(parse_param_assignments(text), atomic_spins) is expected
