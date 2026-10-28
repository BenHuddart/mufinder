"""Tests for CASTEP output parsing utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.output_parser import (
    _FORCE_BLOCK_HEADER_RE,
    parse_castep_mulliken_atomic_spins,
    parse_castep_output,
)


GOLDEN_DIR = Path("tests/golden")


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- Summary fields, completion and failure markers ------------------------------


def test_parse_castep_output_success_extracts_summary_fields() -> None:
    parsed = parse_castep_output(GOLDEN_DIR / "castep_singlepoint_success.castep")

    assert parsed.completed is True
    assert parsed.total_energy == -1234.76543
    assert parsed.enthalpy == -1234.12
    assert parsed.num_scf_cycles == 12
    assert parsed.wall_time == 12.5
    assert parsed.geom_converged is True

    assert len(parsed.forces) == 2
    assert parsed.max_force == pytest.approx(0.038284097220647635, rel=1e-12)
    assert len(parsed.stress) == 3
    assert parsed.max_stress == 3.0
    assert any("WARNING" in line for line in parsed.warnings)
    assert parsed.error_message is None


def test_parse_castep_output_failed_reports_error_state() -> None:
    parsed = parse_castep_output(GOLDEN_DIR / "castep_failed.castep")

    assert parsed.completed is False
    assert parsed.total_energy is None
    assert parsed.error_message is not None
    assert "ERROR" in parsed.error_message


def test_parse_castep_output_handles_truncated_file(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.castep"
    truncated.write_text("SCF iteration 1\nSCF iteration 2\n", encoding="utf-8")

    parsed = parse_castep_output(truncated)

    assert parsed.completed is False
    assert parsed.total_energy is None
    assert parsed.forces == ()
    assert parsed.stress == ()
    assert parsed.error_message is None


def test_completion_via_calculation_complete_marker(tmp_path: Path) -> None:
    output = _write(tmp_path, "complete.castep", "Calculation complete and clean\n")

    parsed = parse_castep_output(output)

    assert parsed.completed is True
    assert parsed.error_message is None
    assert parsed.wall_time is None
    assert parsed.forces == ()
    assert parsed.stress == ()


def test_failure_marker_did_not_converge_blocks_completion(tmp_path: Path) -> None:
    output = _write(
        tmp_path,
        "noconverge.castep",
        "SCF did not converge after 100 cycles\nTotal time = 5.0 s\n",
    )

    parsed = parse_castep_output(output)

    assert parsed.completed is False
    assert parsed.error_message is not None
    assert "did not converge" in parsed.error_message.lower()


def test_panic_marker_detected_as_error(tmp_path: Path) -> None:
    output = _write(tmp_path, "panic.castep", "internal panic in module xyz\n")

    parsed = parse_castep_output(output)

    assert parsed.completed is False
    assert parsed.error_message is not None
    assert "panic" in parsed.error_message.lower()


def test_geometry_optimization_failure_sets_geom_converged_false(tmp_path: Path) -> None:
    output = _write(
        tmp_path,
        "geom_failed.castep",
        "Geometry optimization failed\nTotal time = 4.0 s\n",
    )

    parsed = parse_castep_output(output)

    assert parsed.geom_converged is False
    # "failed" is a failure marker so completion is suppressed and an error
    # message is surfaced from the same line.
    assert parsed.completed is False
    assert parsed.error_message is not None
    assert "failed" in parsed.error_message.lower()


_ACOUSTIC_WARNING_BOX = """\
???????????????????????????????????????????????????????????????????????
                     Warning in secondd_find_acoustics
    Failed to identify 3 eigenvectors with acoustic character
    Acoustic sum rule correction will only be applied to 0 mode(s)
????????????????????????????????????????????????????????????????????????
"""


def test_a_boxed_castep_warning_does_not_fail_a_finished_run(tmp_path: Path) -> None:
    """CASTEP boxes its warnings in ``?`` rules; their wording is not a failure.

    A phonon run of a molecular crystal can find no acoustic-like modes at Gamma
    and say so with "Failed to identify ...", then finish normally.
    """

    output = _write(
        tmp_path,
        "phonon.castep",
        " Run started: Wed, 04 Mar 2020 19:36:03 +0000\n"
        + _ACOUSTIC_WARNING_BOX
        + "Final energy, E             =  -17715.64280048     eV\n"
        + "Total time          =  30603.64 s\n",
    )

    parsed = parse_castep_output(output)

    assert parsed.error_message is None
    assert parsed.completed is True
    assert any("secondd_find_acoustics" in line for line in parsed.warnings)


def test_a_stray_rule_does_not_hide_a_later_failure(tmp_path: Path) -> None:
    """An unbalanced rule must not turn the next boxed warning inside out."""

    output = _write(
        tmp_path,
        "stray.castep",
        "????????????????????????????????????????\n"
        + _ACOUSTIC_WARNING_BOX
        + "Geometry optimization failed\nTotal time = 4.0 s\n",
    )

    parsed = parse_castep_output(output)

    assert parsed.completed is False
    assert parsed.error_message == "Geometry optimization failed"


def test_a_failure_outside_a_warning_box_still_fails_the_run(tmp_path: Path) -> None:
    output = _write(
        tmp_path,
        "failed.castep",
        _ACOUSTIC_WARNING_BOX + "Geometry optimization failed\nTotal time = 4.0 s\n",
    )

    parsed = parse_castep_output(output)

    assert parsed.completed is False
    assert parsed.error_message == "Geometry optimization failed"


def test_earlier_restart_failure_does_not_shadow_later_completion(tmp_path: Path) -> None:
    # A resubmitted/continued calculation appends to the same .castep file,
    # so an early restart's transient failure (e.g. running out of geometry
    # optimization steps before a walltime-triggered continuation) must not
    # cause the job to be reported as failed once a later restart in the same
    # file completes successfully.
    output = _write(
        tmp_path,
        "resubmitted.castep",
        "Run started: Mon, 01 Jun 2026 09:00:00 +0100\n"
        "LBFGS: WARNING - Geometry optimization failed to converge after 300 steps\n"
        "Total time = 10.0 s\n"
        "Run started: Tue, 02 Jun 2026 09:00:00 +0100\n"
        "LBFGS: Geometry optimization completed successfully.\n"
        "Final energy = -42.0 eV\n"
        "Total time = 5.0 s\n",
    )

    parsed = parse_castep_output(output)

    assert parsed.completed is True
    assert parsed.error_message is None
    assert parsed.geom_converged is True


def test_wall_time_falls_back_to_hms_when_no_seconds_value(tmp_path: Path) -> None:
    output = _write(
        tmp_path,
        "hms.castep",
        "Final energy = -1.0 eV\nWall time spent 01:02:03.5 elapsed\n",
    )

    parsed = parse_castep_output(output)

    # 1*3600 + 2*60 + 3.5
    assert parsed.wall_time == pytest.approx(3723.5)
    # "Final energy" precedes the forces and stress, so without the closing
    # "Total time" summary the run is not complete.
    assert parsed.completed is False


_STRESS_ON_HEADER = (
    " Run started: Fri, 02 Oct 2026 10:00:00 +0100\n"
    " ***************************** General Parameters ******************************\n"
    " type of calculation                            : single point energy\n"
    " stress calculation                             : on\n"
)


_ENERGY_AND_FORCES = (
    "Final energy, E             =  -1363.109714972     eV\n"
    "Final free energy (E-TS)    =  -1363.109714972     eV\n"
    "NB est. 0K energy (E-0.5TS)      =  -1363.109714972     eV\n"
    "\n"
    " ******************************** Symmetrised Forces ********************************\n"
    " * Si              1      0.00000      0.00000      0.00000 *\n"
    " ************************************************************************************\n"
    "\n"
)


_STRESS_BLOCK = (
    " *********** Symmetrised Stress Tensor ***********\n"
    " *  x      1.000         0.100         0.000     *\n"
    " *  y      0.100         2.000         0.000     *\n"
    " *  z      0.000         0.000         3.000     *\n"
    " *************************************************\n"
    "\n"
)


def test_output_cut_off_during_stress_calculation_is_not_complete(tmp_path: Path) -> None:
    # Regression: "Final energy" is written before the forces and the stress
    # tensor, so a run killed (or read) mid-way through the stress calculation
    # must not be reported as a finished job.
    output = tmp_path / "stress_cut_off.castep"
    output.write_text(_STRESS_ON_HEADER + _ENERGY_AND_FORCES, encoding="utf-8")

    parsed = parse_castep_output(output)

    assert parsed.completed is False
    assert parsed.error_message is None
    assert parsed.total_energy == pytest.approx(-1363.109714972)
    assert parsed.max_stress is None


def test_final_energy_without_closing_summary_is_not_complete(tmp_path: Path) -> None:
    output = tmp_path / "no_summary.castep"
    output.write_text(_ENERGY_AND_FORCES, encoding="utf-8")

    assert parse_castep_output(output).completed is False


def test_geometry_success_line_without_closing_summary_is_not_complete(tmp_path: Path) -> None:
    # The success banner precedes the final energy, forces and stress.
    output = tmp_path / "geom_no_summary.castep"
    output.write_text(
        "LBFGS: Geometry optimization completed successfully.\n" + _ENERGY_AND_FORCES,
        encoding="utf-8",
    )

    parsed = parse_castep_output(output)

    assert parsed.completed is False
    assert parsed.geom_converged is True


def test_stress_run_with_stress_tensor_and_summary_is_complete(tmp_path: Path) -> None:
    output = tmp_path / "stress_complete.castep"
    output.write_text(
        _STRESS_ON_HEADER
        + _ENERGY_AND_FORCES
        + _STRESS_BLOCK
        + "Total time          =     32.92 s\n",
        encoding="utf-8",
    )

    parsed = parse_castep_output(output)

    assert parsed.completed is True
    assert parsed.max_stress == pytest.approx(3.0)


def test_stress_run_missing_stress_tensor_is_not_complete_even_with_summary(tmp_path: Path) -> None:
    output = tmp_path / "stress_missing.castep"
    output.write_text(
        _STRESS_ON_HEADER + _ENERGY_AND_FORCES + "Total time          =     32.92 s\n",
        encoding="utf-8",
    )

    assert parse_castep_output(output).completed is False


def test_stress_tensor_from_an_earlier_restart_does_not_complete_a_later_one(
    tmp_path: Path,
) -> None:
    # The latest restart requested stress but stopped before printing it; the
    # earlier restart's tensor must not stand in for it.
    output = tmp_path / "stress_restart.castep"
    output.write_text(
        _STRESS_ON_HEADER
        + _ENERGY_AND_FORCES
        + _STRESS_BLOCK
        + "Total time          =     32.92 s\n"
        + _STRESS_ON_HEADER
        + _ENERGY_AND_FORCES
        + "Total time          =     12.00 s\n",
        encoding="utf-8",
    )

    assert parse_castep_output(output).completed is False


# --- Forces and stress -----------------------------------------------------------


def test_parse_castep_output_reads_max_force_line_without_force_block(tmp_path: Path) -> None:
    output = tmp_path / "max_force_only.castep"
    output.write_text(
        "NB est. 0K energy (E-0.5TS) = -4.2 eV\nMaximum force = 0.1234 eV/A\nTotal time = 3.0 s\n",
        encoding="utf-8",
    )

    parsed = parse_castep_output(output)

    assert parsed.total_energy == -4.2
    assert parsed.max_force == 0.1234


def test_ase_style_force_block_with_constrained_token(tmp_path: Path) -> None:
    output = _write(
        tmp_path,
        "ase_forces.castep",
        "\n".join(
            [
                "Final energy = -10.0 eV",
                "******************** Forces ********************",
                " * H        1      0.10000      0.00000(cons'd)      0.00000 *",
                " * H        2     -0.20000      0.00000      0.00000 *",
                "*****************************************************",
                "Total time = 2.0 s",
                "",
            ]
        ),
    )

    parsed = parse_castep_output(output)

    assert len(parsed.forces) == 2
    assert parsed.forces[0] == ("H", pytest.approx(0.1), pytest.approx(0.0), pytest.approx(0.0))
    assert parsed.forces[1][0] == "H"
    # max_force is computed from the block when no explicit max-force line exists.
    assert parsed.max_force == pytest.approx(0.2)


def test_last_force_block_wins_when_multiple_blocks_present(tmp_path: Path) -> None:
    output = _write(
        tmp_path,
        "multi_force.castep",
        "\n".join(
            [
                "Final energy = -1.0 eV",
                "*** Forces ***",
                "1 H   0.5   0.0   0.0",
                "",
                "*** Forces ***",
                "1 O   0.9   0.0   0.0",
                "2 O  -0.1   0.0   0.0",
                "",
                "Total time = 1.0 s",
                "",
            ]
        ),
    )

    parsed = parse_castep_output(output)

    species = tuple(row[0] for row in parsed.forces)
    assert species == ("O", "O")
    assert parsed.max_force == pytest.approx(0.9)


def test_ase_style_stress_block_yields_three_rows_and_max_abs(tmp_path: Path) -> None:
    output = _write(
        tmp_path,
        "ase_stress.castep",
        "\n".join(
            [
                "Final energy = -5.0 eV",
                "*********** Stress Tensor ***********",
                " *  x      1.50000      0.00000      0.00000  *",
                " *  y      0.00000     -7.25000      0.00000  *",
                " *  z      0.00000      0.00000      2.00000  *",
                "*************************************",
                "Total time = 1.0 s",
                "",
            ]
        ),
    )

    parsed = parse_castep_output(output)

    assert len(parsed.stress) == 3
    # max_stress is the largest absolute component (abs(-7.25)).
    assert parsed.max_stress == pytest.approx(7.25)


def test_plain_stress_block_negative_component_sets_max_stress(tmp_path: Path) -> None:
    output = _write(
        tmp_path,
        "plain_stress.castep",
        "\n".join(
            [
                "Final energy = -2.0 eV",
                "*** Stress Tensor ***",
                "  1.0   0.0   0.0",
                "  0.0  -8.0   0.0",
                "  0.0   0.0   2.0",
                "",
                "Total time = 1.0 s",
                "",
            ]
        ),
    )

    parsed = parse_castep_output(output)

    assert len(parsed.stress) == 3
    assert parsed.max_stress == pytest.approx(8.0)


class TestBlockHeaderDetection:
    """The substring guard in front of the header regexes is a speed device only.

    It must not become a second, stricter rule. These pin the case-insensitivity
    the guard could silently have broken — a lowercase ``forces`` header would
    still match the regex but be rejected by a case-sensitive pre-filter, and the
    result would be silently missing forces rather than an error.
    """

    def _output(self, force_header: str, stress_header: str) -> str:
        return f"""
 Run started: Fri, 10 Jan 2025 11:57:08 +0000
 Final energy, E             =  -100.0     eV

 {force_header}
 * Si  1   0.10000   0.20000   0.30000 *

 {stress_header}
 *  x      1.000000      0.000000      0.000000  *
 *  y      0.000000      2.000000      0.000000  *
 *  z      0.000000      0.000000      3.000000  *

 Total time =     1.00 s
"""

    @pytest.mark.parametrize(
        "force_header",
        [
            "*********************** Forces ***********************",
            "*********************** forces ***********************",
            "*********************** FORCES ***********************",
            "************* Unconstrained Forces *************",
        ],
    )
    def test_force_headers_are_found_in_any_case(self, tmp_path: Path, force_header: str) -> None:
        path = tmp_path / "seed.castep"
        path.write_text(self._output(force_header, "***** Stress Tensor *****"), encoding="utf-8")

        result = parse_castep_output(path)

        assert result.forces == (("Si", 0.1, 0.2, 0.3),)

    @pytest.mark.parametrize(
        "stress_header",
        [
            "***** Stress Tensor *****",
            "***** stress tensor *****",
            "***** STRESS TENSOR *****",
        ],
    )
    def test_stress_headers_are_found_in_any_case(self, tmp_path: Path, stress_header: str) -> None:
        path = tmp_path / "seed.castep"
        path.write_text(self._output("***** Forces *****", stress_header), encoding="utf-8")

        result = parse_castep_output(path)

        assert result.stress[0] == pytest.approx((1.0, 0.0, 0.0))

    def test_the_guard_is_permissive_and_the_regex_still_decides(self, tmp_path: Path) -> None:
        """A line can pass the substring guard and still not be a header.

        The guard only rejects lines the regex could never match; it must not be
        allowed to become the decision. This line contains "orce" (so the guard
        lets it through) but not the plural ``Forces`` the regex requires, so the
        regex is what rejects it. A change that promoted the guard to the
        decision would wrongly treat it as a header and truncate the real block.
        """

        prose = " the net force on each ion is reported below"
        assert "orce" in prose.casefold()
        assert _FORCE_BLOCK_HEADER_RE.search(prose) is None

        path = tmp_path / "seed.castep"
        path.write_text(
            self._output("***** Forces *****", "***** Stress Tensor *****").replace(
                " * Si  1   0.10000   0.20000   0.30000 *",
                " * Si  1   0.10000   0.20000   0.30000 *\n" + prose,
            ),
            encoding="utf-8",
        )

        result = parse_castep_output(path)

        assert result.forces == (("Si", 0.1, 0.2, 0.3),)


# --- Final lattice vectors -------------------------------------------------------


def test_final_lattice_vectors_parsed_and_last_block_wins(tmp_path: Path) -> None:
    output = _write(
        tmp_path,
        "lattice.castep",
        "\n".join(
            [
                "Real Lattice",
                "   1.0   0.0   0.0",
                "   0.0   1.0   0.0",
                "   0.0   0.0   1.0",
                "some other text",
                "Unit Cell",
                "   2.0   0.0   0.0",
                "   0.0   2.0   0.0",
                "   0.0   0.0   3.0",
                "Final energy = -1.0 eV",
                "Total time = 1.0 s",
                "",
            ]
        ),
    )

    parsed = parse_castep_output(output)

    assert parsed.final_lattice_vectors == (
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 3.0),
    )


def test_lattice_header_with_leading_noise_rows_skipped_until_floats(tmp_path: Path) -> None:
    # The first rows after the header are non-numeric (no rows yet -> continue),
    # then three float rows are collected.
    output = _write(
        tmp_path,
        "lattice_noise.castep",
        "\n".join(
            [
                "Unit Cell",
                "header label without numbers",
                "   4.0   0.0   0.0",
                "   0.0   5.0   0.0",
                "   0.0   0.0   6.0",
                "Final energy = -1.0 eV",
                "Total time = 1.0 s",
                "",
            ]
        ),
    )

    parsed = parse_castep_output(output)

    assert parsed.final_lattice_vectors == (
        (4.0, 0.0, 0.0),
        (0.0, 5.0, 0.0),
        (0.0, 0.0, 6.0),
    )


def test_lattice_block_breaks_on_gap_after_partial_rows(tmp_path: Path) -> None:
    # Header then two float rows, then a non-numeric line: because rows is
    # non-empty, parsing breaks before three rows accumulate -> no candidate.
    output = _write(
        tmp_path,
        "lattice_partial.castep",
        "\n".join(
            [
                "Real Lattice",
                "   1.0   2.0   3.0",
                "   4.0   5.0   6.0",
                "not a number row at all",
                "   7.0   8.0   9.0",
                "Final energy = -1.0 eV",
                "Total time = 1.0 s",
                "",
            ]
        ),
    )

    parsed = parse_castep_output(output)

    assert parsed.final_lattice_vectors is None


def test_no_lattice_header_yields_none(tmp_path: Path) -> None:
    output = _write(
        tmp_path,
        "no_lattice.castep",
        "Final energy = -1.0 eV\nTotal time = 1.0 s\n",
    )

    parsed = parse_castep_output(output)

    assert parsed.final_lattice_vectors is None


# --- Mulliken atomic spins -------------------------------------------------------


def test_parse_castep_output_extracts_mulliken_spins_when_present(tmp_path: Path) -> None:
    output = tmp_path / "mulliken_spin.castep"
    output.write_text(
        "\n".join(
            [
                "     Atomic Populations (Mulliken)",
                "     -----------------------------",
                "Species          Ion Spin      s       p       d       f      Total   Charge(e)   Spin(hbar/2)",
                "==============================================================================================",
                "  Fe              1   up:     0.340   0.263   4.512   0.000   5.115     0.000        2.230",
                "                  1   dn:     0.359   0.400   2.126   0.000   2.885",
                "  Fe              2   up:     0.340   0.263   4.512   0.000   5.115     0.000       -2.230",
                "                  2   dn:     0.359   0.400   2.126   0.000   2.885",
                "==============================================================================================",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert parse_castep_mulliken_atomic_spins(output) == pytest.approx((2.23, -2.23))

    parsed = parse_castep_output(output)

    assert parsed.mulliken_atomic_spins == pytest.approx((2.23, -2.23))


def test_parse_castep_mulliken_spins_preserves_species_local_ion_order(tmp_path: Path) -> None:
    output = tmp_path / "mixed_species_mulliken_spin.castep"
    output.write_text(
        "\n".join(
            [
                "     Atomic Populations (Mulliken)",
                "     -----------------------------",
                "Species          Ion Spin      s       p       d       f      Total   Charge(e)   Spin(hbar/2)",
                "==============================================================================================",
                "  F               1   up:     0.973   2.775   0.000   0.000   3.748    -0.534       -0.037",
                "                  1   dn:     0.976   2.810   0.000   0.000   3.785",
                "  F               2   up:     0.976   2.810   0.000   0.000   3.786    -0.533        0.038",
                "                  2   dn:     0.973   2.775   0.000   0.000   3.748",
                "  Co              1   up:     0.153   0.167   4.976   0.000   5.295     1.067        2.658",
                "                  1   dn:     0.141   0.158   2.338   0.000   2.637",
                "  Co              2   up:     0.141   0.158   2.339   0.000   2.638     1.067       -2.656",
                "                  2   dn:     0.152   0.166   4.976   0.000   5.295",
                "==============================================================================================",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert parse_castep_mulliken_atomic_spins(output) == pytest.approx(
        (-0.037, 0.038, 2.658, -2.656)
    )


def test_mulliken_none_when_header_absent(tmp_path: Path) -> None:
    output = _write(tmp_path, "nospin.castep", "Final energy = -1.0 eV\n")

    assert parse_castep_mulliken_atomic_spins(output) is None


def test_mulliken_header_present_but_no_spin_rows_returns_none(tmp_path: Path) -> None:
    # Header exists, but the population rows have no "up:"/"dn:" channel, so
    # saw_spin stays False and the function returns None.
    output = _write(
        tmp_path,
        "nochannel.castep",
        "\n".join(
            [
                "     Atomic Populations (Mulliken)",
                "     -----------------------------",
                "Species   Ion      s       p       d       f      Total   Charge",
                "==================================================================",
                "  Fe       1     0.34    0.26    4.51    0.00    5.11     0.00",
                "==================================================================",
                "",
            ]
        ),
    )

    assert parse_castep_mulliken_atomic_spins(output) is None


def test_mulliken_stops_at_bond_section_and_skips_short_rows(tmp_path: Path) -> None:
    # Exercises: blank-line continue (before any spin), Species skip,
    # dashed/equals separator skip, short-token continue, "dn:" ignored,
    # and the "Bond" terminator break.
    output = _write(
        tmp_path,
        "bondstop.castep",
        "\n".join(
            [
                "     Atomic Populations (Mulliken)",
                "",
                "Species          Ion Spin   s   p   d   f   Total  Charge  Spin",
                "-------------------------------------------------------------",
                "  H              1   up:   0.5   0.0   0.0   0.0   0.5   0.0   0.9",
                "                 1   dn:   0.4   0.0   0.0   0.0   0.4",
                "ab",
                "  Bond                          Population      Length (A)",
                "  H 1 -- H 2                        0.50          0.74",
                "",
            ]
        ),
    )

    spins = parse_castep_mulliken_atomic_spins(output)

    assert spins == pytest.approx((0.9,))


def test_mulliken_up_row_with_unparseable_value_is_skipped(tmp_path: Path) -> None:
    # An "up:" row whose final token is not a float triggers the ValueError
    # branch (continue) without setting saw_spin; the second valid up row wins.
    output = _write(
        tmp_path,
        "badval.castep",
        "\n".join(
            [
                "     Atomic Populations (Mulliken)",
                "Species   Ion Spin   s   p   d   f   Total  Charge  Spin",
                "  Fe      1   up:   0.3   0.2   4.5   0.0   5.0   0.0   notanumber",
                "  Fe      2   up:   0.3   0.2   4.5   0.0   5.0   0.0   1.5",
                "",
            ]
        ),
    )

    spins = parse_castep_mulliken_atomic_spins(output)

    assert spins == pytest.approx((1.5,))


def test_mulliken_blank_line_after_spin_terminates_block(tmp_path: Path) -> None:
    # After a valid spin row, a blank line breaks the loop (saw_spin True),
    # so a later up row outside the block is not included.
    output = _write(
        tmp_path,
        "blankbreak.castep",
        "\n".join(
            [
                "     Atomic Populations (Mulliken)",
                "Species   Ion Spin   s   p   d   f   Total  Charge  Spin",
                "  Fe      1   up:   0.3   0.2   4.5   0.0   5.0   0.0   2.0",
                "",
                "  Fe      2   up:   0.3   0.2   4.5   0.0   5.0   0.0   9.9",
                "",
            ]
        ),
    )

    spins = parse_castep_mulliken_atomic_spins(output)

    assert spins == pytest.approx((2.0,))
