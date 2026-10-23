"""Resubmitting relaxations asks whether to continue from the checkpoint or start again."""

from __future__ import annotations

from pathlib import Path

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.app.resubmit_choice_dialog import (
    ResubmitCandidate,
    ResubmitChoiceDialog,
    ResubmitMode,
    cause_summary,
    checkpoint_summary,
    default_resubmit_mode,
)
from mufinder.domain.enums import FailureReason, JobStatus, WorkflowStage
from mufinder.services.cluster_runner import CheckpointProbeStatus
from mufinder.services.project_service import ProjectService

FOUND = CheckpointProbeStatus.FOUND
ABSENT = CheckpointProbeStatus.ABSENT
UNAVAILABLE = CheckpointProbeStatus.UNAVAILABLE


def _candidate(
    reason: FailureReason | None,
    *,
    status: JobStatus = JobStatus.FAILED,
    checkpoint: CheckpointProbeStatus = FOUND,
) -> ResubmitCandidate:
    return ResubmitCandidate(label="job", status=status, reason=reason, checkpoint=checkpoint)


@pytest.mark.parametrize(
    "reason",
    [
        FailureReason.WALLTIME,
        FailureReason.OUT_OF_MEMORY,
        FailureReason.NODE_FAILURE,
        FailureReason.INTERRUPTED,
    ],
)
def test_interrupted_runs_default_to_continue(reason: FailureReason) -> None:
    assert default_resubmit_mode([_candidate(reason)]) == ResubmitMode.CONTINUE


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(FailureReason.CASTEP_ERROR),
        _candidate(FailureReason.SUBMISSION),
        _candidate(None),
        _candidate(None, status=JobStatus.CANCELLED),
    ],
)
def test_runs_that_went_wrong_default_to_start_again(candidate: ResubmitCandidate) -> None:
    assert default_resubmit_mode([candidate]) == ResubmitMode.START_AGAIN


def test_a_mixed_batch_defaults_to_the_choice_that_deletes_nothing() -> None:
    candidates = [_candidate(FailureReason.CASTEP_ERROR), _candidate(FailureReason.WALLTIME)]

    assert default_resubmit_mode(candidates) == ResubmitMode.CONTINUE


def test_cause_summary_counts_each_cause() -> None:
    candidates = [
        _candidate(FailureReason.WALLTIME, status=JobStatus.STOPPED),
        _candidate(FailureReason.WALLTIME, status=JobStatus.STOPPED),
        _candidate(FailureReason.CASTEP_ERROR),
        _candidate(None, status=JobStatus.CANCELLED),
        _candidate(None),
    ]

    assert cause_summary(candidates) == "2 walltime, 1 CASTEP, 1 cancelled, 1 no recorded cause"


@pytest.mark.parametrize(
    ("checkpoints", "expected"),
    [
        ([FOUND], "The job has a checkpoint to resume from."),
        ([FOUND, FOUND], "Every job has a checkpoint to resume from."),
        ([ABSENT], "No checkpoint was found, so the run will start from its initial structure."),
        (
            [ABSENT, ABSENT],
            "No checkpoints were found, so the runs will start from their initial structures.",
        ),
        (
            [UNAVAILABLE],
            "Checkpoints could not be checked; a run without one starts from its initial structure.",
        ),
        (
            [FOUND, FOUND, ABSENT, UNAVAILABLE],
            "2 have a checkpoint; 1 has none and will start from the initial structure; 1 could not be checked.",
        ),
    ],
)
def test_checkpoint_summary(checkpoints: list[CheckpointProbeStatus], expected: str) -> None:
    candidates = [
        _candidate(FailureReason.WALLTIME, checkpoint=checkpoint) for checkpoint in checkpoints
    ]

    assert checkpoint_summary(candidates) == expected


def test_dialog_preselects_the_default_and_reports_the_choice(qapp) -> None:
    dialog = ResubmitChoiceDialog([_candidate(FailureReason.WALLTIME)], other_job_count=2)

    assert dialog.mode() == ResubmitMode.CONTINUE
    dialog._start_again_radio.setChecked(True)
    assert dialog.mode() == ResubmitMode.START_AGAIN
    dialog.close()

    dialog = ResubmitChoiceDialog([_candidate(FailureReason.CASTEP_ERROR)])
    assert dialog.mode() == ResubmitMode.START_AGAIN
    dialog.close()


def test_enter_cancels_while_start_again_is_selected(qapp) -> None:
    dialog = ResubmitChoiceDialog([_candidate(FailureReason.CASTEP_ERROR)])
    assert dialog._cancel_button.isDefault()
    assert not dialog._resubmit_button.isDefault()

    dialog._continue_radio.setChecked(True)
    assert dialog._resubmit_button.isDefault()
    assert not dialog._cancel_button.isDefault()

    dialog._start_again_radio.setChecked(True)
    assert dialog._cancel_button.isDefault()
    dialog.close()


# --- Dashboard Resubmit -----------------------------------------------------


def _window_with_jobs(tmp_path: Path, specs: list[tuple[str, WorkflowStage]]):
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "resubmit.mufinder", "Resubmit")
    jobs = []
    for label, stage in specs:
        job = service.add_job(doc, label=label, workflow_stage=stage, calc_type="castep")
        job.status = JobStatus.STOPPED
        job.failure_reason = FailureReason.WALLTIME
        job.execution_target = "alpha"
        job.remote_directory = f"/remote/work/{label}"
        jobs.append(job)
    window._document = doc
    return window, jobs


def _record_cluster_submissions(monkeypatch: pytest.MonkeyPatch, window: MainWindow) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(
        window,
        "_submit_jobs_on_cluster",
        lambda jobs, **kwargs: calls.append({"jobs": jobs, **kwargs}),
    )
    return calls


def _fake_choice_dialog(
    monkeypatch: pytest.MonkeyPatch, *, accepted: bool, mode: ResubmitMode
) -> list:
    shown: list = []

    class _FakeDialog:
        def __init__(self, candidates, *, other_job_count, parent) -> None:
            shown.append((candidates, other_job_count))

        def exec(self) -> int:
            return main_window_module.QDialog.DialogCode.Accepted if accepted else 0

        def mode(self) -> ResubmitMode:
            return mode

    monkeypatch.setattr(main_window_module, "ResubmitChoiceDialog", _FakeDialog)
    return shown


def test_resubmitting_only_other_jobs_asks_nothing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, jobs = _window_with_jobs(tmp_path, [("conv_300", WorkflowStage.CONVERGENCE_CUTOFF)])
    shown = _fake_choice_dialog(monkeypatch, accepted=True, mode=ResubmitMode.START_AGAIN)
    calls = _record_cluster_submissions(monkeypatch, window)

    window._on_resubmit_from_dashboard([job.id for job in jobs])

    assert shown == []
    assert calls[0]["fresh_start_job_ids"] == set()
    window.close()


def test_start_again_marks_only_the_relaxations(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, jobs = _window_with_jobs(
        tmp_path,
        [
            ("relax_a", WorkflowStage.MUON_RELAXATION),
            ("host_relax", WorkflowStage.PRISTINE_RELAXATION),
            ("conv_300", WorkflowStage.CONVERGENCE_CUTOFF),
        ],
    )
    shown = _fake_choice_dialog(monkeypatch, accepted=True, mode=ResubmitMode.START_AGAIN)
    calls = _record_cluster_submissions(monkeypatch, window)

    window._on_resubmit_from_dashboard([job.id for job in jobs])

    candidates, other_job_count = shown[0]
    assert [candidate.label for candidate in candidates] == ["relax_a", "host_relax"]
    # Not connected to the cluster, so the checkpoints could not be checked.
    assert {candidate.checkpoint for candidate in candidates} == {UNAVAILABLE}
    assert other_job_count == 1
    assert calls[0]["fresh_start_job_ids"] == {jobs[0].id, jobs[1].id}
    window.close()


def test_continue_resubmits_in_place(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window, jobs = _window_with_jobs(tmp_path, [("relax_a", WorkflowStage.MUON_RELAXATION)])
    _fake_choice_dialog(monkeypatch, accepted=True, mode=ResubmitMode.CONTINUE)
    calls = _record_cluster_submissions(monkeypatch, window)

    window._on_resubmit_from_dashboard([jobs[0].id])

    assert calls[0]["fresh_start_job_ids"] == set()
    assert calls[0]["reuse_existing_remote_files"] is True
    window.close()


def test_cancelling_the_choice_resubmits_nothing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, jobs = _window_with_jobs(tmp_path, [("relax_a", WorkflowStage.MUON_RELAXATION)])
    _fake_choice_dialog(monkeypatch, accepted=False, mode=ResubmitMode.START_AGAIN)
    calls = _record_cluster_submissions(monkeypatch, window)

    window._on_resubmit_from_dashboard([jobs[0].id])

    assert calls == []
    assert jobs[0].status == JobStatus.STOPPED
    window.close()


def test_imported_relaxations_are_never_offered_a_fresh_start(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, jobs = _window_with_jobs(tmp_path, [("relax_a", WorkflowStage.MUON_RELAXATION)])
    jobs[0].tags.append(main_window_module.IMPORTED_JOB_TAG)

    assert MainWindow._job_resumes_from_checkpoint(jobs[0]) is False
    window.close()


def test_local_checkpoint_probe(qapp, tmp_path: Path) -> None:
    window, jobs = _window_with_jobs(
        tmp_path,
        [("relax_a", WorkflowStage.MUON_RELAXATION), ("relax_b", WorkflowStage.MUON_RELAXATION)],
    )
    found_dir, absent_dir = tmp_path / "relax_a", tmp_path / "relax_b"
    for job, job_dir in zip(jobs, (found_dir, absent_dir), strict=True):
        job.execution_target = "local"
        job_dir.mkdir()
    (found_dir / "relax_a.check").write_bytes(b"checkpoint")
    directories = {jobs[0].id: found_dir, jobs[1].id: absent_dir}
    window._job_output_directory = lambda job: directories[job.id]

    probes = window._checkpoint_probes(jobs)

    assert probes[jobs[0].id].status == FOUND
    assert probes[jobs[1].id].status == ABSENT
    window.close()


def test_walltime_tooltip_offers_the_choice_for_relaxations(qapp, tmp_path: Path) -> None:
    window, jobs = _window_with_jobs(
        tmp_path,
        [
            ("relax_a", WorkflowStage.MUON_RELAXATION),
            ("conv_300", WorkflowStage.CONVERGENCE_CUTOFF),
        ],
    )

    assert window._failure_tooltip(jobs[0]).endswith(MainWindow._RELAXATION_WALLTIME_NEXT_STEP)
    assert window._failure_tooltip(jobs[1]).endswith(MainWindow._WALLTIME_NEXT_STEP)
    window.close()
