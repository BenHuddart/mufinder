"""The CASTEP outputs ``docs_screenshots.synthetic`` writes for the screenshot projects.

The module imports no Qt, so it is imported here directly; the outputs must
read back through the parsers the application uses on real CASTEP output.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from mufinder.castep_io.geometry_progress import parse_geometry_progress_text
from mufinder.castep_io.output_parser import parse_castep_output
from mufinder.services.job_progress import GeometryProgressReader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from docs_screenshots.synthetic import bfgs_log_text, final_configuration  # noqa: E402

LATTICE = [[2.6954645, 2.6954645, 0.0], [2.6954645, 0.0, 2.6954645], [0.0, 2.6954645, 2.6954645]]
SPECIES = ["Si", "Si"]
FRACTIONAL = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]


def _flat(rows) -> list[float]:
    return [float(value) for row in rows for value in row]


def _log(**overrides: object) -> str:
    arguments: dict[str, object] = {
        "lattice": LATTICE,
        "species": SPECIES,
        "fractional": FRACTIONAL,
        "iterations": 6,
        "initial_enthalpy": -340.682656,
        "final_enthalpy": -340.953487,
        "initial_force": 1.025682,
        "final_force": 0.0101,
    }
    arguments.update(overrides)
    return bfgs_log_text(**arguments)


def test_a_finished_bfgs_log_reads_as_a_converged_geometry_optimisation(tmp_path: Path) -> None:
    castep = tmp_path / "Si2.castep"
    castep.write_text(_log(), encoding="utf-8")

    result = parse_castep_output(castep)

    assert result.completed
    assert result.geom_converged is True
    assert result.error_message is None
    assert result.total_energy == pytest.approx(-340.953487)
    assert result.enthalpy == pytest.approx(-340.953487)
    assert result.max_force == pytest.approx(0.0101, abs=1e-5)
    assert result.wall_time == pytest.approx(1843.2)
    assert _flat(result.final_lattice_vectors) == pytest.approx(_flat(LATTICE), abs=1e-6)
    lattice, species, fractional = final_configuration(castep)
    assert species == SPECIES
    assert _flat(lattice) == pytest.approx(_flat(LATTICE), abs=1e-6)
    assert _flat(fractional) == pytest.approx(_flat(FRACTIONAL), abs=1e-6)


def test_the_progress_parser_reads_every_iteration_with_falling_energy_and_force() -> None:
    progress = parse_geometry_progress_text(_log())

    assert progress is not None
    assert (progress.method, progress.max_iterations, progress.converged) == ("BFGS", 100, True)
    assert [iteration.iteration for iteration in progress.iterations] == list(range(7))
    enthalpies = [iteration.enthalpy for iteration in progress.iterations]
    assert enthalpies[0] == pytest.approx(-340.682656) and enthalpies[-1] == pytest.approx(-340.953487)
    assert all(later < earlier for earlier, later in zip(enthalpies, enthalpies[1:]))
    drops = [earlier - later for earlier, later in zip(enthalpies, enthalpies[1:])]
    assert all(later < earlier for earlier, later in zip(drops, drops[1:]))
    forces = [criterion.value for iteration in progress.iterations for criterion in iteration.criteria if criterion.name == "|F|max"]
    assert all(later < earlier for earlier, later in zip(forces, forces[1:]))
    assert [criterion.name for criterion in progress.latest.criteria] == ["dE/ion", "|F|max", "|dR|max"]
    assert progress.latest.criteria_met == 3
    assert progress.iterations[0].criteria_met == 0


def test_an_unfinished_bfgs_log_reads_as_a_running_optimisation(tmp_path: Path) -> None:
    castep = tmp_path / "Si2.castep"
    castep.write_text(_log(iterations=3, finished=False, final_force=0.2), encoding="utf-8")

    progress = GeometryProgressReader().read_local("job", [castep])
    result = parse_castep_output(castep)

    assert progress is not None
    assert progress.converged is None
    assert [iteration.iteration for iteration in progress.iterations] == [0, 1, 2, 3]
    assert progress.latest.criteria_met < 3
    assert not result.completed
    assert result.geom_converged is None


def test_a_finished_log_whose_last_force_misses_the_tolerance_is_refused() -> None:
    with pytest.raises(ValueError, match="force_tolerance"):
        _log(final_force=0.2)


@pytest.mark.parametrize("iterations", [0, 101])
def test_an_iteration_count_outside_the_run_is_refused(iterations: int) -> None:
    with pytest.raises(ValueError, match="iteration"):
        _log(iterations=iterations)
