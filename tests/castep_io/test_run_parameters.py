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


def _mixing_block(
    *,
    scheme: str = "Broyden",
    history: str = "        20",
    amplitude: str | None = "0.2000",
    mix_cutoff: str | None = "  900.0       eV",
) -> str:
    """Build a Density Mixing Parameters block in CASTEP's own layout.

    CASTEP writes this block only when ``METALS_METHOD`` is ``DM``, and gates
    most of its lines on output verbosity: the scheme and history length are
    unconditional, the amplitudes and mixing cutoff need at least intermediate
    mode, and the g-vectors and metric coefficient need expert mode. Pass
    ``None`` for the optional lines to model a lower-verbosity run.
    """

    lines = [
        " ***************************** Density Mixing Parameters ***********************",
        "",
        f" density-mixing scheme                          : {scheme}",
        f" max. length of mixing history                  : {history}",
    ]
    if amplitude is not None:
        lines.append(f" charge density mixing amplitude                : {amplitude}")
    if mix_cutoff is not None:
        lines.append(f" cut-off energy for mixing                      : {mix_cutoff}")
    return "\n".join(lines)


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
    mixing_block: str | None = None,
) -> str:
    """Build a minimal parameter echo in CASTEP's own layout."""

    if mixing_block is None:
        mixing_block = _mixing_block()

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

{mixing_block}

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
    """The keywords an import depends on resolve against any runtime catalog.

    Asserted for the specific keywords the importer needs rather than for every
    mapped line: the translation tables are fixed, but the schema they resolve
    against is discovered at runtime and is much smaller when no CASTEP
    installation is found. Requiring *all* mapped keywords to be present would
    make this a test of whichever catalog happens to be installed.
    """

    parsed = parse_run_parameters_text(_echo())
    by_keyword = {item.keyword: item for item in parsed.observations if item.mapped}

    required = (
        "task",
        "xc_functional",
        "cut_off_energy",
        "charge",
        "max_scf_cycles",
        "geom_max_iter",
        "kpoint_mp_grid",
    )
    for keyword in required:
        assert keyword_is_supported(by_keyword[keyword]), keyword


def test_a_mapped_keyword_can_still_be_absent_from_a_thin_catalog() -> None:
    """The failures "MuFinder cannot name this line" and "this catalog does not
    carry that keyword" are different, and only the second is recoverable by
    pointing MuFinder at a CASTEP installation.

    The density-mixing keywords are the live example: the bundled starter
    catalog does not carry them, so they are mapped from the echo but resolve to
    no spec until a real catalog is found.
    """

    parsed = parse_run_parameters_text(_echo())
    by_keyword = {item.keyword: item for item in parsed.observations if item.mapped}

    observation = by_keyword["mix_charge_amp"]
    assert observation.mapped
    assert observation.value == 0.2
    # Whether it is *supported* depends on the runtime catalog, so this asserts
    # only that the question is answerable, not which answer it gives here.
    assert isinstance(keyword_is_supported(observation), bool)


def test_text_without_an_echo_yields_nothing() -> None:
    assert parse_run_parameters_text("no parameters here\n").observations == ()


def test_parse_run_parameters_reads_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "seed.castep"
    path.write_text(_echo(), encoding="utf-8")

    assert parse_run_parameters(path).mapped_values()["task"] == "GEOMETRYOPTIMIZATION"


def test_parse_run_parameters_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RunParameterParseError, match="Failed to read"):
        parse_run_parameters(tmp_path / "absent.castep")


def test_recovers_the_density_mixing_settings() -> None:
    """A legacy ``.param`` need not state these.

    Without them, whether a run's mixing settings survive an import depends on
    whether the local input file happened to carry them — so two runs of the
    same calculation can import with different parameter sets purely because one
    ``.param`` was older than the other.
    """

    values = parse_run_parameters_text(_echo()).mapped_values()

    assert values["mixing_scheme"] == "BROYDEN"
    assert values["mix_history_length"] == 20
    assert values["mix_charge_amp"] == 0.2
    assert values["mix_cut_off_energy"] == ValueWithUnit(value=900.0, unit="eV")


def test_the_mixing_amplitude_is_a_bare_number() -> None:
    """CASTEP writes it with format 94's trailing unit argument absent, so
    reading it as a physical quantity would attach a unit it never stated."""

    values = parse_run_parameters_text(_echo()).mapped_values()

    assert values["mix_charge_amp"] == 0.2
    assert not isinstance(values["mix_charge_amp"], ValueWithUnit)


@pytest.mark.parametrize(
    "printed, expected",
    [
        ("Linear", "LINEAR"),
        ("Kerker", "KERKER"),
        ("Broyden", "BROYDEN"),
        ("Pulay", "PULAY"),
    ],
)
def test_mixing_scheme_display_strings_map_back_to_keyword_values(
    printed: str, expected: str
) -> None:
    text = _echo(mixing_block=_mixing_block(scheme=printed))

    assert parse_run_parameters_text(text).mapped_values()["mixing_scheme"] == expected


def test_a_lower_verbosity_run_reports_only_what_it_printed() -> None:
    """The amplitude and mixing cutoff need intermediate output verbosity. Their
    absence is a genuine absence, not a parse failure, and must not be filled in
    from the keyword's schema default."""

    text = _echo(mixing_block=_mixing_block(amplitude=None, mix_cutoff=None))

    values = parse_run_parameters_text(text).mapped_values()

    assert values["mixing_scheme"] == "BROYDEN"
    assert values["mix_history_length"] == 20
    assert "mix_charge_amp" not in values
    assert "mix_cut_off_energy" not in values


def test_mixing_labels_are_only_read_inside_their_own_section() -> None:
    """These are the most generically-worded labels in the tables, so a future
    CASTEP reusing one elsewhere must not be mis-mapped."""

    text = _echo().replace(
        " ***************************** Density Mixing Parameters ***********************",
        " ***************************** General Parameters ******************************",
    )

    parsed = parse_run_parameters_text(text)

    assert "mix_charge_amp" not in parsed.mapped_values()
    assert any(item.label == "charge density mixing amplitude" for item in parsed.unmapped())
