"""Reset's directory cleanup never deletes files another job still depends on.

A job's directory is cleaned on Reset only when no other job uses it, unless
that job is reset at the same time or is a directory companion its controller
names. The one companion is the singlepoint job an older MuFinder version ran
for a Quantum Corrections row before the phonon job, in the same seed
directory: resetting the phonon job with clean-up empties that directory, so
the row reruns as a fresh phonon job, and the singlepoint keeps its record.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.services.project_service import ProjectService
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE
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
    doc = service.create_project(tmp_path / "quantum_reset.mufinder", "QuantumReset")
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
    phonon.status = phonon_status
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


def _capture_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    return warnings


def test_resetting_a_legacy_phonon_with_cleanup_empties_the_shared_directory(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, doc, _definition, _candidate, singlepoint, phonon, directory = _legacy_quantum_row(
        tmp_path, singlepoint_status=JobStatus.COMPLETED, phonon_status=JobStatus.FAILED
    )
    warnings = _capture_warnings(monkeypatch)

    window._reset_failed_jobs([phonon], cleanup_choice=True)

    # The legacy checkpoint is gone, so the rerun is a fresh phonon job.
    assert warnings == []
    assert phonon.status == JobStatus.DEFINED
    assert sorted(path.name for path in directory.iterdir()) == QUANTUM_INPUTS
    # The singlepoint stays as a record in the Job Dashboard.
    assert doc.jobs[singlepoint.id].status == JobStatus.COMPLETED
    window.close()


def test_resetting_both_legacy_jobs_together_cleans_their_directory(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, _definition, _candidate, singlepoint, phonon, directory = _legacy_quantum_row(
        tmp_path, singlepoint_status=JobStatus.FAILED, phonon_status=JobStatus.FAILED
    )
    warnings = _capture_warnings(monkeypatch)

    window._reset_failed_jobs([singlepoint, phonon], cleanup_choice=True)

    assert warnings == []
    assert sorted(path.name for path in directory.iterdir()) == QUANTUM_INPUTS
    window.close()


def test_resetting_a_legacy_phonon_without_cleanup_keeps_its_checkpoint(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, _definition, _candidate, singlepoint, phonon, directory = _legacy_quantum_row(
        tmp_path, singlepoint_status=JobStatus.COMPLETED, phonon_status=JobStatus.FAILED
    )
    warnings = _capture_warnings(monkeypatch)

    window._reset_failed_jobs([phonon], cleanup_choice=False)

    # The rerun continues from the checkpoint in the kept directory.
    assert warnings == []
    assert phonon.status == JobStatus.DEFINED
    assert singlepoint.status == JobStatus.COMPLETED
    assert sorted(path.name for path in directory.iterdir()) == QUANTUM_FILES
    window.close()


def test_resetting_a_phonon_keeps_a_folder_an_unrelated_job_uses(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, doc, _definition, _candidate, _singlepoint, phonon, directory = _legacy_quantum_row(
        tmp_path, singlepoint_status=JobStatus.COMPLETED, phonon_status=JobStatus.FAILED
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
    warnings = _capture_warnings(monkeypatch)

    window._reset_failed_jobs([phonon], cleanup_choice=True)

    assert phonon.status == JobStatus.DEFINED
    assert sorted(path.name for path in directory.iterdir()) == QUANTUM_FILES
    assert (
        f"- quantum_phonon_siteA: another job uses the same output folder ({directory.resolve()}); "
        "its folder was kept"
    ) in warnings[0]
    window.close()


def test_resetting_a_relaxation_keeps_a_folder_its_replacement_uses(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "relax_reset.mufinder", "RelaxReset")
    window._document = doc
    old = _pristine_job(service, doc, JobStatus.FAILED)
    _replacement = _pristine_job(service, doc, JobStatus.COMPLETED)
    directory = _job_directory(doc)
    warnings = _capture_warnings(monkeypatch)

    window._reset_failed_jobs([old], cleanup_choice=True)

    assert old.status == JobStatus.DEFINED
    assert sorted(path.name for path in directory.iterdir()) == ALL_FILES
    assert f"- {SEED}: another job uses the same output folder" in warnings[0]
    window.close()


def test_a_folder_shared_with_an_imported_job_is_never_cleaned(qapp, tmp_path: Path) -> None:
    """Imported jobs are skipped by the cleanup, so they never count as cleaned together."""

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "imported_share.mufinder", "ImportedShare")
    window._document = doc
    native = _pristine_job(service, doc, JobStatus.FAILED)
    imported = _pristine_job(service, doc, JobStatus.FAILED)
    imported.tags.append(main_window_module.IMPORTED_JOB_TAG)
    directory = _job_directory(doc)

    reset_count, cleanup_errors = window._apply_failed_job_reset(
        [native, imported], cleanup_choice=True, detach_convergence=False
    )

    assert reset_count == 1
    assert sorted(path.name for path in directory.iterdir()) == ALL_FILES
    assert f"- {SEED}: another job uses the same output folder" in cleanup_errors[0]
    window.close()
