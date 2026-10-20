"""Geometry-optimisation progress parsed from real CASTEP 26.11 output.

The fixtures in ``tests/fixtures/geometry`` were produced by CASTEP 26.11 from
the ``.cell``/``.param`` files beside them:

- ``Si2_bfgs.castep``: a BFGS relaxation of ions and cell, so every step tests
  all four criteria; it converges at iteration 7.
- ``Si2_lbfgs.castep``: an LBFGS relaxation with a fixed cell (three criteria).
  The first run stops at ``GEOM_MAX_ITER : 3`` without converging; a
  ``CONTINUATION : DEFAULT`` run with ``GEOM_MAX_ITER : 30`` appends a second
  restart to the same file and converges at its own iteration 3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.geometry_progress import (
    ConvergenceCriterion,
    is_geometry_progress_line,
    parse_geometry_progress,
    parse_geometry_progress_lines,
    parse_geometry_progress_text,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BFGS = FIXTURES / "geometry" / "Si2_bfgs.castep"
LBFGS_CONTINUED = FIXTURES / "geometry" / "Si2_lbfgs.castep"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_bfgs_history_has_every_iteration_and_criterion() -> None:
    progress = parse_geometry_progress(BFGS)

    assert progress is not None
    assert progress.method == "BFGS"
    assert progress.max_iterations == 30
    assert progress.restart == 0
    assert progress.converged is True
    assert [iteration.iteration for iteration in progress.iterations] == list(range(8))
    first = progress.iterations[0]
    assert first.enthalpy == pytest.approx(-340.682656)
    assert first.enthalpy_unit == "eV"
    assert [criterion.name for criterion in first.criteria] == ["dE/ion", "|F|max", "|dR|max", "Smax"]
    latest = progress.latest
    assert latest is not None
    assert latest.criteria == (
        ConvergenceCriterion("dE/ion", 1.585179e-05, 2.0e-05, "eV", True),
        ConvergenceCriterion("|F|max", 1.013222e-02, 2.0e-02, "eV/A", True),
        ConvergenceCriterion("|dR|max", 4.263498e-04, 1.0e-03, "A", True),
        ConvergenceCriterion("Smax", 7.684312e-02, 1.0e-01, "GPa", True),
    )
    assert latest.criteria_met == 4
    # Iteration 6 had every criterion but the displacement converged.
    assert progress.iterations[6].criteria_met == 3


def test_continued_run_reports_only_the_latest_restart() -> None:
    progress = parse_geometry_progress(LBFGS_CONTINUED)

    assert progress is not None
    assert progress.method == "LBFGS"
    assert progress.restart == 1
    # CASTEP counts from 0 again after a continuation, and the continuation's
    # own step limit and success replace the first run's.
    assert [iteration.iteration for iteration in progress.iterations] == [0, 1, 2, 3]
    assert progress.max_iterations == 30
    assert progress.converged is True
    assert [criterion.name for criterion in progress.latest.criteria] == ["dE/ion", "|F|max", "|dR|max"]
    assert progress.latest.criteria_met == 3


def test_first_run_of_the_continued_file_reports_its_failure() -> None:
    text = _text(LBFGS_CONTINUED)
    second_run = text.index("Run started", text.index("Run started") + 1)

    progress = parse_geometry_progress_text(text[: text.rindex("\n", 0, second_run)])

    assert progress is not None
    assert progress.restart == 0
    assert progress.max_iterations == 3
    assert progress.converged is False
    assert progress.latest.iteration == 3
    assert progress.latest.criteria_met < len(progress.latest.criteria)


def test_output_cut_off_inside_a_table_keeps_the_last_complete_iteration() -> None:
    text = _text(BFGS)
    fifth = text.index("BFGS: finished iteration     5")
    inside_table = text.index("|F|max", fifth)

    for cut in (fifth + 10, text.index("\n", fifth), inside_table):
        progress = parse_geometry_progress_text(text[:cut])
        assert progress is not None
        assert progress.converged is None
        assert progress.latest.iteration == 4
        assert len(progress.latest.criteria) == 4


def test_output_before_the_parameter_echo_has_no_progress() -> None:
    text = _text(BFGS)

    assert parse_geometry_progress_text(text[: text.index("Geometry Optimization Parameters")]) is None
    assert parse_geometry_progress_text("") is None


def test_run_that_has_echoed_its_method_but_not_finished_a_step_has_no_iterations() -> None:
    text = _text(BFGS)

    progress = parse_geometry_progress_text(text[: text.index("BFGS: finished iteration     0")])

    assert progress is not None
    assert progress.method == "BFGS"
    assert progress.iterations == ()
    assert progress.latest is None


def test_single_point_output_has_no_progress() -> None:
    golden = Path(__file__).resolve().parents[1] / "golden" / "castep_singlepoint_success.castep"

    assert parse_geometry_progress(golden) is None


def test_cell_only_relaxation_tests_energy_and_stress() -> None:
    progress = parse_geometry_progress(FIXTURES / "uep" / "Si8.castep")

    assert progress is not None
    assert [criterion.name for criterion in progress.latest.criteria] == ["dE/ion", "Smax"]


@pytest.mark.parametrize("path", [BFGS, LBFGS_CONTINUED])
def test_kept_lines_parse_the_same_as_the_whole_file(path: Path) -> None:
    lines = _text(path).splitlines()
    kept = [line for line in lines if is_geometry_progress_line(line)]

    assert parse_geometry_progress_lines(kept) == parse_geometry_progress_lines(lines)
    # The line-minimiser and SCF tables are not convergence rows.
    assert not any(line.rstrip().endswith(("<-- min BFGS", "<-- SCF")) for line in kept)
    assert len(kept) < len(lines) // 5
