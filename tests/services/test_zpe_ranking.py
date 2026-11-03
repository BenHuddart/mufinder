"""Tests for :mod:`mufinder.services.zpe_ranking`.

That the Quantum Corrections table, the Journal and the survey agree on it is pinned in
``tests/app/workflow_controllers/test_quantum_corrections.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Result
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.source_candidates import (
    PhononZpeSource,
    eligible_quantum_source_candidates,
)
from mufinder.services.zpe_ranking import zpe_ranking
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_PHONON_SET_CALC_TYPE
from tests.support.muon_sites import add_relaxed_muon_site


@pytest.fixture
def document(tmp_path: Path) -> ProjectDocument:
    return ProjectService().create_project(tmp_path / "zpe.mufinder", "ZPE")


def _phonon_job(document: ProjectDocument, data: dict | None) -> tuple[str, Job]:
    """One relaxed site, with a completed phonon job that read its current relaxation."""

    add_relaxed_muon_site(document, label="A")
    (candidate,) = eligible_quantum_source_candidates(document)
    job = Job(
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
        status=JobStatus.COMPLETED,
    )
    document.jobs[job.id] = job
    if data is not None:
        run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
        document.runs[run.id] = run
        result = Result(result_type=QUANTUM_PHONON_CALC_TYPE, calculation_run_id=run.id, data=data)
        document.results[result.id] = result
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_SET_CALC_TYPE,
        parameters={
            "phonon_job_ids_by_label": {candidate.key: str(job.id)},
            "phonon_source_run_ids_by_label": {candidate.key: str(candidate.run.id)},
        },
    )
    document.calculations[definition.id] = definition
    return candidate.key, job


def _ranking(document: ProjectDocument):
    return zpe_ranking(document, eligible_quantum_source_candidates(document))


def test_a_site_with_a_current_zpe_is_ranked_without_a_mark(document) -> None:
    key, _job = _phonon_job(document, {"muon_zpe_total_eV": 0.5})

    ranking = _ranking(document)

    assert ranking.sources[key] is PhononZpeSource.CURRENT
    assert ranking.zpe_eV(key) == 0.5
    assert ranking.corrected_eV(key) == 0.0
    assert (ranking.mark(key), ranking.notes(key), ranking.warned(key)) == ("", (), False)


def test_a_wrong_mass_warning_marks_the_numbers_but_keeps_its_place(document) -> None:
    key, _job = _phonon_job(
        document, {"muon_zpe_total_eV": 0.5, "phonon_warnings": ["Check the muon mass."]}
    )

    ranking = _ranking(document)

    assert ranking.warned(key) is True
    assert key not in ranking.flagged
    assert ranking.corrected_eV(key) == 0.0


def test_a_non_positive_mode_with_no_stored_warning_is_flagged_but_not_marked(document) -> None:
    """The table marks only stored warnings, so nothing else may mark more than it does."""

    key, _job = _phonon_job(
        document, {"muon_zpe_total_eV": 0.5, "muon_zpe_modes": [{"frequency_cm": -300.0}]}
    )

    ranking = _ranking(document)

    assert key in ranking.flagged
    assert ranking.warned(key) is False


@pytest.mark.parametrize("status", [JobStatus.RUNNING, JobStatus.DEFINED, JobStatus.FAILED])
def test_a_job_that_is_not_completed_gives_its_site_no_result(document, status) -> None:
    key, job = _phonon_job(document, {"muon_zpe_total_eV": 0.5})
    job.status = status

    ranking = _ranking(document)

    assert ranking.results == {}
    assert ranking.zpe_eV(key) is None
    assert ranking.corrected_eV(key) is None
    assert ranking.mark(key) == ""


def test_a_site_with_no_stored_result_has_nothing_to_rank(document) -> None:
    key, _job = _phonon_job(document, None)

    assert _ranking(document).results == {}
    assert _ranking(document).corrected_eV(key) is None


def test_a_project_with_no_definition_ranks_nothing(document) -> None:
    add_relaxed_muon_site(document, label="A")

    ranking = _ranking(document)

    assert ranking.results == {}
    assert ranking.newest_results == {}
