"""Tests for CASTEP output parsing utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.output_parser import parse_castep_output


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
