"""Starting a job again never deletes files another job still depends on.

A job's directory is cleared when it starts again only when no other job uses
it, unless that job is a directory companion its controller names. The one
companion is the singlepoint job an older MuFinder version ran for a Quantum
Corrections row before the phonon job, in the same seed directory: starting the
phonon job again empties that directory, so the row reruns as a fresh phonon
job, and the singlepoint keeps its record.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.domain.constants import IMPORTED_JOB_TAG
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.services.project_service import ProjectService
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE
from tests.support.jobs import set_job_state
from tests.app.test_fresh_start_safety import ALL_FILES, SEED, _job_directory, _pristine_job
from tests.support.main_window import stage_quantum_row

QUANTUM_SEED = "quantum_siteA"
QUANTUM_FILES = sorted(
    [
        f"{QUANTUM_SEED}.cell",
        f"{QUANTUM_SEED}.param",
        f"{QUANTUM_SEED}.castep",
        f"{QUANTUM_SEED}.check",
        f"{QUANTUM_SEED}.phonon",
        f"{QUANTUM_SEED}.0001.err",
    ]
)
QUANTUM_INPUTS = [f"{QUANTUM_SEED}.cell", f"{QUANTUM_SEED}.param"]


def _legacy_quantum_row(tmp_path: Path, *, singlepoint_status: JobStatus, phonon_status: JobStatus):
    """A legacy Quantum Corrections row whose singlepoint and phonon share one seed directory."""

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_again.mufinder", "QuantumAgain")
    window._document = doc
    definition, candidate, singlepoint = stage_quantum_row(
        window, service, doc, singlepoint_status=singlepoint_status
    )
    definition.parameters["job_seed_names_by_label"] = {candidate.key: QUANTUM_SEED}
    phonon = service.add_job(
        doc,
        label="quantum_phonon_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
    )
    set_job_state(phonon, phonon_status)
    definition.parameters["phonon_job_ids_by_label"] = {candidate.key: str(phonon.id)}

    directory = doc.project_root / "files" / "quantum_corrections" / QUANTUM_SEED
    directory.mkdir(parents=True, exist_ok=True)
    for name in QUANTUM_FILES:
        (directory / name).write_text("data\n", encoding="utf-8")
    assert (
        window._job_output_directory(singlepoint)
        == window._job_output_directory(phonon)
        == directory
    )
    return window, doc, definition, candidate, singlepoint, phonon, directory


def test_starting_a_legacy_phonon_again_empties_the_shared_directory(qapp, tmp_path: Path) -> None:
    window, doc, _definition, _candidate, singlepoint, phonon, directory = _legacy_quantum_row(
        tmp_path, singlepoint_status=JobStatus.COMPLETED, phonon_status="failed"
    )

    cleaned, errors = window._jobs._clear_outputs_for_fresh_start([phonon])

    # The legacy checkpoint is gone, so the rerun is a fresh phonon job.
    assert cleaned == [phonon] and errors == []
    assert sorted(path.name for path in directory.iterdir()) == QUANTUM_INPUTS
    # The singlepoint stays as a record in the Job Dashboard.
    assert doc.jobs[singlepoint.id].status == JobStatus.COMPLETED
    window.close()


def test_starting_a_phonon_again_keeps_a_folder_an_unrelated_job_uses(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, doc, _definition, _candidate, _singlepoint, phonon, directory = _legacy_quantum_row(
        tmp_path, singlepoint_status=JobStatus.COMPLETED, phonon_status="failed"
    )
    stranger = ProjectService().add_job(
        doc, label="stranger", workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS, calc_type="castep"
    )
    original_lookup = window._job_output_directory
    monkeypatch.setattr(
        window,
        "_job_output_directory",
        lambda job: directory if job.id == stranger.id else original_lookup(job),
    )

    cleaned, errors = window._jobs._clear_outputs_for_fresh_start([phonon])

    assert cleaned == []
    assert sorted(path.name for path in directory.iterdir()) == QUANTUM_FILES
    assert "another job uses the same output folder" in errors[0]
    window.close()


def test_a_folder_shared_with_an_imported_job_is_never_cleaned(qapp, tmp_path: Path) -> None:
    """An imported job never runs again, so its folder is never cleared with a native job's."""

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "imported_share.mufinder", "ImportedShare")
    window._document = doc
    native = _pristine_job(service, doc, "failed")
    imported = _pristine_job(service, doc, "failed")
    imported.tags.append(IMPORTED_JOB_TAG)
    directory = _job_directory(doc)

    cleaned, errors = window._jobs._clear_outputs_for_fresh_start([native])

    assert cleaned == []
    assert sorted(path.name for path in directory.iterdir()) == ALL_FILES
    assert f"- {SEED}: another job uses the same output folder" in errors[0]
    window.close()


def test_an_imported_job_is_never_cleared_even_when_asked(qapp, tmp_path: Path) -> None:
    """The clean-up skips an imported job itself, not only through its callers."""

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "imported_alone.mufinder", "ImportedAlone")
    window._document = doc
    imported = _pristine_job(service, doc, "failed")
    imported.tags.append(IMPORTED_JOB_TAG)
    directory = _job_directory(doc)

    cleaned, errors = window._jobs._clear_outputs_for_fresh_start([imported])

    assert cleaned == []
    assert errors == [f"- {imported.label}: it records a run imported into MuFinder"]
    assert sorted(path.name for path in directory.iterdir()) == ALL_FILES
    window.close()
