"""Tests for recovering run parameters from a ``.castep`` parameter echo.

Every fixture here is hand-authored to the layout CASTEP's own source produces
(``Source/Fundamental/parameters.f90`` and ``cell.f90``). Nothing is copied from
a real calculation: see ``docs/legacy_import.rst`` (Research data boundary).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.run_parameters import (
    RunParameterParseError,
    keyword_is_supported,
    parse_run_parameters,
    parse_run_parameters_text,
)
from mufinder.castep_io.types import ValueWithUnit


def _echo(
    *,
    task: str = "geometry optimization",
    functional: str = "Perdew Burke Ernzerhof",
    cutoff: str = "  900.0000   eV",
    charge: str = "+1.000",
    spin_line: str = "treating system as non-spin-polarized",
    metals_line: str = (
        "Method: Treating system as metallic with density mixing treatment of electrons,"
    ),
    mp_grid: str = "3  3  3",
    run_banner: str = " Run started: Fri, 10 Jan 2025 11:57:08 +0000",
) -> str:
    """Build a minimal parameter echo in CASTEP's own layout."""

    return f"""
 | Welcome to Academic Release CASTEP version 26.11 |

{run_banner}

 ************************************ Title ************************************


 ***************************** General Parameters ******************************

 output verbosity                               : normal  (1)
 type of calculation                            : {task}
 stress calculation                             : off
 write structure in CELL formatted file         : on

 *********************** Exchange-Correlation Parameters ***********************

 using functional                               : {functional}
 relativistic treatment                         : Koelling-Harmon

 **************************** Basis Set Parameters *****************************

 plane wave basis set cut-off                   : {cutoff}
 finite basis set correction                    : none

 **************************** Electronic Parameters ****************************

 number of  electrons                           :  1080.
 net charge of system                           : {charge}
 {spin_line}
 number of bands                                :        648

 ********************* Electronic Minimization Parameters **********************

 {metals_line}
 total energy / atom convergence tol.           : 0.1000E-09   eV
 max. number of SCF cycles                      :        100

 ********************** Geometry Optimization Parameters ***********************

 optimization method                            : LBFGS
 max. number of steps                           :        300
 total energy convergence tolerance             : 0.2000E-04   eV/atom
        max ionic |force| tolerance             : 0.5000E-01   eV/A
 max ionic |displacement| tolerance             : 0.1000E-02   A
   max |stress component| tolerance             : 0.1000       GPa

 *******************************************************************************

                       MP grid size for SCF calculation is  {mp_grid}
                       Number of kpoints used =            14
"""


def test_recovers_the_keywords_an_import_needs() -> None:
    values = parse_run_parameters_text(_echo()).mapped_values()

    assert values["task"] == "GEOMETRYOPTIMIZATION"
    assert values["xc_functional"] == "PBE"
    assert values["cut_off_energy"] == ValueWithUnit(value=900.0, unit="eV")
    assert values["charge"] == 1
    assert values["spin_polarized"] is False
    assert values["metals_method"] == "DM"
    assert values["max_scf_cycles"] == 100
    assert values["geom_max_iter"] == 300
    assert values["write_cell_structure"] is True
    assert values["kpoint_mp_grid"] == (3, 3, 3)


def test_recovers_the_two_values_downstream_jobs_inherit() -> None:
    """The cutoff and k-point grid are the functional payload of the whole reader.

    Legacy inputs often state neither: the cutoff is implied by
    ``BASIS_PRECISION`` and the grid never reaches the local input cell. Only
    the echo carries both.
    """

    values = parse_run_parameters_text(_echo()).mapped_values()

    assert values["cut_off_energy"] == ValueWithUnit(value=900.0, unit="eV")
    assert values["kpoint_mp_grid"] == (3, 3, 3)


@pytest.mark.parametrize(
    "printed, expected",
    [
        ("single point energy", "SINGLEPOINT"),
        ("geometry optimization", "GEOMETRYOPTIMIZATION"),
        ("transition state search", "TRANSITIONSTATESEARCH"),
        ("phonon calculation", "PHONON"),
        ("molecular dynamics", "MOLECULARDYNAMICS"),
    ],
)
def test_task_display_strings_map_back_to_keyword_values(printed: str, expected: str) -> None:
    values = parse_run_parameters_text(_echo(task=printed)).mapped_values()
    assert values["task"] == expected


@pytest.mark.parametrize(
    "printed, expected",
    [
        ("Perdew Burke Ernzerhof", "PBE"),
        ("PBE for solids (2008)", "PBESOL"),
        ("revised Perdew Burke Ernzerhof", "RPBE"),
        ("BLYP", "BLYP"),
    ],
)
def test_functional_display_strings_map_back_to_keyword_values(
    printed: str, expected: str
) -> None:
    values = parse_run_parameters_text(_echo(functional=printed)).mapped_values()
    assert values["xc_functional"] == expected


def test_a_lossy_display_string_reports_its_alternatives() -> None:
    """CASTEP prints one string for both ``LDA`` and ``LDA_PZ``.

    The echo cannot distinguish them, so the reader must surface the ambiguity
    rather than silently asserting one reading.
    """

    parsed = parse_run_parameters_text(
        _echo(functional="Perdew-Zunger Local Density Approximation")
    )

    assert parsed.mapped_values()["xc_functional"] == "LDA"
    ambiguous = parsed.ambiguous()
    assert [item.keyword for item in ambiguous] == ["xc_functional"]
    assert ambiguous[0].ambiguous_with == ("LDA_PZ",)


def test_an_unambiguous_functional_reports_no_alternatives() -> None:
    assert parse_run_parameters_text(_echo()).ambiguous() == ()


@pytest.mark.parametrize(
    "line, expected",
    [
        ("treating system as non-spin-polarized", False),
        ("treating system as spin-polarized", True),
    ],
)
def test_spin_treatment_statements_map_to_spin_polarized(line: str, expected: bool) -> None:
    values = parse_run_parameters_text(_echo(spin_line=line)).mapped_values()
    assert values["spin_polarized"] is expected


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Method: Treating system as metallic with conventional treatment of electrons,", "NONE"),
        ("Method: Treating system as metallic with density mixing treatment of electrons,", "DM"),
        ("Method: Treating system as metallic with ensemble DFT treatment of electrons,", "EDFT"),
    ],
)
def test_metals_method_statements_map_to_keyword_values(line: str, expected: str) -> None:
    values = parse_run_parameters_text(_echo(metals_line=line)).mapped_values()
    assert values["metals_method"] == expected


def test_the_non_metallic_statement_is_not_a_metals_method() -> None:
    """``non-metallic`` is emitted for fixed occupancy, not a METALS_METHOD value.

    Mapping it would assert something CASTEP did not say.
    """

    parsed = parse_run_parameters_text(
        _echo(metals_line="Method: Treating system as non-metallic,")
    )

    assert "metals_method" not in parsed.mapped_values()


def test_geometry_labels_are_only_read_inside_their_own_section() -> None:
    """``max. number of steps`` is the geometry cap and nothing else.

    Two CASTEP sections carry similar wording, so a label that is only
    meaningful in one section must not be harvested from another.
    """

    text = _echo().replace(
        " ********************** Geometry Optimization Parameters ***********************",
        " ************************** Density Mixing Parameters **************************",
    )

    parsed = parse_run_parameters_text(text)

    assert "geom_max_iter" not in parsed.mapped_values()
    assert any(item.label == "max. number of steps" for item in parsed.unmapped())


def test_a_multi_restart_output_reports_the_final_run() -> None:
    """CASTEP appends on restart, so the last echo is the one that applies."""

    first = _echo(cutoff="  500.0000   eV", mp_grid="2  2  2")
    second = _echo(cutoff="  900.0000   eV", mp_grid="4  4  4")

    values = parse_run_parameters_text(first + second).mapped_values()

    assert values["cut_off_energy"] == ValueWithUnit(value=900.0, unit="eV")
    assert values["kpoint_mp_grid"] == (4, 4, 4)


def test_unrecognised_echo_lines_are_reported_not_dropped() -> None:
    parsed = parse_run_parameters_text(_echo())

    labels = {item.label for item in parsed.unmapped()}
    assert "output verbosity" in labels
    assert "relativistic treatment" in labels
    # ...but they are never presented as keyword values.
    assert all(item.keyword is None for item in parsed.unmapped())


def test_an_unreadable_value_becomes_unmapped_rather_than_raising() -> None:
    """One surprising line must not discard an entire echo."""

    parsed = parse_run_parameters_text(_echo(cutoff="  not-a-number   eV"))

    assert "cut_off_energy" not in parsed.mapped_values()
    # The rest of the echo still parsed.
    assert parsed.mapped_values()["task"] == "GEOMETRYOPTIMIZATION"


def test_castep_version_is_captured_for_provenance() -> None:
    assert parse_run_parameters_text(_echo()).castep_version == "26.11"


def test_castep_version_ignores_the_banner_border() -> None:
    """The version banner is boxed in ``|`` and the closing bar is not always
    separated by a space, so a naive capture picks it up as part of the version.
    """

    text = " | Welcome to Academic Release CASTEP version 22.11|\n" + _echo()

    assert parse_run_parameters_text(text).castep_version == "22.11"


def test_observations_record_which_file_each_keyword_belongs_to() -> None:
    parsed = parse_run_parameters_text(_echo())
    by_keyword = {item.keyword: item for item in parsed.observations if item.mapped}

    assert by_keyword["cut_off_energy"].file == "param"
    assert by_keyword["kpoint_mp_grid"].file == "cell"


def test_keyword_is_supported_reflects_the_runtime_schema() -> None:
    parsed = parse_run_parameters_text(_echo())
    mapped = [item for item in parsed.observations if item.mapped]

    assert mapped, "expected at least one mapped observation"
    assert all(keyword_is_supported(item) for item in mapped)


def test_text_without_an_echo_yields_nothing() -> None:
    assert parse_run_parameters_text("no parameters here\n").observations == ()


def test_parse_run_parameters_reads_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "seed.castep"
    path.write_text(_echo(), encoding="utf-8")

    assert parse_run_parameters(path).mapped_values()["task"] == "GEOMETRYOPTIMIZATION"


def test_parse_run_parameters_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RunParameterParseError, match="Failed to read"):
        parse_run_parameters(tmp_path / "absent.castep")
