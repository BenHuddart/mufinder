"""Reset's directory cleanup never deletes files another job still depends on.

A Quantum Corrections phonon job continues from the stage-1 singlepoint's
checkpoint in the same seed directory (both stages write ``{seed}.*`` there), so
cleaning the phonon job's directory on Reset used to delete the completed
singlepoint's ``.check`` and ``.castep`` and leave the row unable to rerun stage
2. Reset now keeps any directory shared with a job it is not also resetting.
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
from tests.app.test_main_window_actions import _stage_quantum_row_with_singlepoint

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


def _quantum_row(tmp_path: Path, *, singlepoint_status: JobStatus, phonon_status: JobStatus):
    """A Quantum Corrections row whose singlepoint and phonon share one seed directory."""

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_reset.mufinder", "QuantumReset")
    window._document = doc
    definition, candidate, singlepoint = _stage_quantum_row_with_singlepoint(
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
    assert window._job_output_directory(singlepoint) == window._job_output_directory(phonon) == directory
    return window, doc, definition, candidate, singlepoint, phonon, directory


def _capture_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox, "warning", lambda _parent, _title, text: warnings.append(text)
    )
    return warnings


def test_resetting_a_failed_phonon_keeps_the_stage_one_checkpoint(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, _definition, _candidate, singlepoint, phonon, directory = _quantum_row(
        tmp_path, singlepoint_status=JobStatus.COMPLETED, phonon_status=JobStatus.FAILED
    )
    warnings = _capture_warnings(monkeypatch)

    window._reset_failed_jobs([phonon], cleanup_choice=True)

    # The phonon job is reset, but the shared directory is left untouched.
    assert phonon.status == JobStatus.DEFINED
    assert singlepoint.status == JobStatus.COMPLETED
    assert sorted(path.name for path in directory.iterdir()) == QUANTUM_FILES
    assert (
        f"- quantum_phonon_siteA: another job uses the same output folder ({directory.resolve()}); "
        "its folder was kept"
    ) in warnings[0]
    window.close()


def test_resetting_both_stages_together_cleans_their_directory(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, _definition, _candidate, singlepoint, phonon, directory = _quantum_row(
        tmp_path, singlepoint_status=JobStatus.FAILED, phonon_status=JobStatus.FAILED
    )
    warnings = _capture_warnings(monkeypatch)

    window._reset_failed_jobs([singlepoint, phonon], cleanup_choice=True)

    assert warnings == []
    assert sorted(path.name for path in directory.iterdir()) == [
        f"{QUANTUM_SEED}.cell",
        f"{QUANTUM_SEED}.param",
    ]
    window.close()


def test_reset_without_cleanup_never_asks_about_sharing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, _definition, _candidate, _singlepoint, phonon, directory = _quantum_row(
        tmp_path, singlepoint_status=JobStatus.COMPLETED, phonon_status=JobStatus.FAILED
    )
    warnings = _capture_warnings(monkeypatch)

    window._reset_failed_jobs([phonon], cleanup_choice=False)

    assert warnings == []
    assert phonon.status == JobStatus.DEFINED
    assert sorted(path.name for path in directory.iterdir()) == QUANTUM_FILES
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


def test_reset_to_stage_one_still_clears_its_own_directory(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, doc, definition, candidate, singlepoint, phonon, directory = _quantum_row(
        tmp_path, singlepoint_status=JobStatus.COMPLETED, phonon_status=JobStatus.FAILED
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox, "exec", lambda _self: main_window_module.QMessageBox.StandardButton.Yes
    )

    window._on_reset_quantum_corrections_to_stage_one([candidate.key])

    # Both stages belong to the row, so rewinding it removes the directory.
    assert not directory.exists()
    assert singlepoint.id not in doc.jobs and phonon.id not in doc.jobs
    window.close()


def test_reset_to_stage_one_refuses_a_directory_shared_with_another_row(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, doc, definition, candidate, singlepoint, phonon, directory = _quantum_row(
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
    monkeypatch.setattr(
        main_window_module.QMessageBox, "exec", lambda _self: main_window_module.QMessageBox.StandardButton.Yes
    )
    warnings = _capture_warnings(monkeypatch)

    window._on_reset_quantum_corrections_to_stage_one([candidate.key])

    assert sorted(path.name for path in directory.iterdir()) == QUANTUM_FILES
    assert singlepoint.id in doc.jobs and phonon.id in doc.jobs
    assert "another job uses the same output folder" in warnings[0]
    assert "the row was not reset" in warnings[0]
    window.close()
