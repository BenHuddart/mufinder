"""Additional coverage for CASTEP output parsing branches and error paths.

These tests exercise parser branches not covered by
``tests/test_castep_output_parser.py``: ASE-style force/stress blocks,
constrained-force tokens, HH:MM:SS wall-time fallback, geometry-failure
detection, lattice-vector parsing edge cases, Mulliken population
short-circuits, and restart-segment isolation for resubmitted/continued
calculations. They are deterministic and use only temporary files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.output_parser import (
    parse_castep_mulliken_atomic_spins,
    parse_castep_output,
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


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


def test_wall_time_falls_back_to_hms_when_no_seconds_value(tmp_path: Path) -> None:
    output = _write(
        tmp_path,
        "hms.castep",
        "Final energy = -1.0 eV\nWall time spent 01:02:03.5 elapsed\n",
    )

    parsed = parse_castep_output(output)

    # 1*3600 + 2*60 + 3.5
    assert parsed.wall_time == pytest.approx(3723.5)
    assert parsed.completed is True


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


def test_completion_via_calculation_complete_marker(tmp_path: Path) -> None:
    output = _write(tmp_path, "complete.castep", "Calculation complete and clean\n")

    parsed = parse_castep_output(output)

    assert parsed.completed is True
    assert parsed.error_message is None
    assert parsed.wall_time is None
    assert parsed.forces == ()
    assert parsed.stress == ()


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
