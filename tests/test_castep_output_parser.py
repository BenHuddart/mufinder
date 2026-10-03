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
        _STRESS_ON_HEADER + _ENERGY_AND_FORCES + _STRESS_BLOCK + "Total time          =     32.92 s\n",
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


def test_stress_tensor_from_an_earlier_restart_does_not_complete_a_later_one(tmp_path: Path) -> None:
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


def test_parse_castep_output_reads_max_force_line_without_force_block(tmp_path: Path) -> None:
    output = tmp_path / "max_force_only.castep"
    output.write_text(
        "NB est. 0K energy (E-0.5TS) = -4.2 eV\n"
        "Maximum force = 0.1234 eV/A\n"
        "Total time = 3.0 s\n",
        encoding="utf-8",
    )

    parsed = parse_castep_output(output)

    assert parsed.total_energy == -4.2
    assert parsed.max_force == 0.1234


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
    def test_force_headers_are_found_in_any_case(
        self, tmp_path: Path, force_header: str
    ) -> None:
        path = tmp_path / "seed.castep"
        path.write_text(
            self._output(force_header, "***** Stress Tensor *****"), encoding="utf-8"
        )

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
    def test_stress_headers_are_found_in_any_case(
        self, tmp_path: Path, stress_header: str
    ) -> None:
        path = tmp_path / "seed.castep"
        path.write_text(
            self._output("***** Forces *****", stress_header), encoding="utf-8"
        )

        result = parse_castep_output(path)

        assert result.stress[0] == pytest.approx((1.0, 0.0, 0.0))

    def test_the_guard_is_permissive_and_the_regex_still_decides(
        self, tmp_path: Path
    ) -> None:
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
