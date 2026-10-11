"""Recovering jobs from the main window: retry retrieval, dismiss, reset, inspect failure and continue relaxation."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PySide6.QtCore import Qt

import mufinder.app.main_window as main_window_module
import mufinder.app.workflow_table_utils as workflow_table_utils
from mufinder.app import selection_snapshot as selection_snapshot_module
from mufinder.app.selection_snapshot import SelectionScope
from mufinder.app.workflow_submission import READY_WORKFLOW_ROW_STATUSES
from mufinder.app.main_window import MainWindow
from mufinder.castep_setup import CastepSetup
from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Structure
from mufinder.services.project_service import ProjectService
from tests.support.reset_menu import install_fake_menu, open_reset_menu
from tests.support.main_window import (
    add_cluster_input_job,
    patch_continuation_cluster_dialog,
    patch_continuation_cluster_runner,
)


def test_main_window_reset_failed_jobs_can_clean_output_directory(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_failed_cleanup.mufinder", "ResetFailedCleanup")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    job.error_message = "boom"
    window._document = doc

    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.cell").write_text("cell", encoding="utf-8")
    (output_dir / f"{job.label}.param").write_text("param", encoding="utf-8")
    (output_dir / "stale.err").write_text("old failure", encoding="utf-8")

    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: True)

    window._reset_failed_jobs([job])

    assert job.status == JobStatus.DEFINED
    assert sorted(path.name for path in output_dir.iterdir()) == [f"{job.label}.cell", f"{job.label}.param"]
    assert doc.provenance[-1].action == "job_reset"
    window.close()


def test_main_window_reset_failed_jobs_no_failed_shows_info(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_no_failed.mufinder", "ResetNoFailed")
    job = service.add_job(
        doc,
        label="job_defined",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._reset_failed_jobs([job])

    assert infos == ["No resettable jobs"]
    window.close()


def test_main_window_reset_failed_jobs_accepts_cancelled_jobs(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_cancelled_job.mufinder", "ResetCancelledJob")
    job = service.add_job(
        doc,
        label="job_cancelled",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.CANCELLED
    job.error_message = "stopped"
    window._document = doc

    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._reset_failed_jobs([job])

    assert job.status == JobStatus.DEFINED
    assert job.error_message is None
    window.close()


def test_main_window_reset_failed_jobs_cancelled_by_user(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_cancelled.mufinder", "ResetCancelled")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    job.error_message = "boom"
    window._document = doc

    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: None)

    window._reset_failed_jobs([job])

    assert job.status == JobStatus.FAILED
    assert job.error_message == "boom"
    window.close()


def test_main_window_reset_failed_jobs_warns_when_cleanup_fails(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_cleanup_warn.mufinder", "ResetCleanupWarn")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    window._document = doc

    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stale.log").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: True)
    monkeypatch.setattr(window, "_clear_directory_contents", lambda _path: (_ for _ in ()).throw(OSError("denied")))

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._reset_failed_jobs([job])

    assert job.status == JobStatus.DEFINED
    assert warnings == ["Directory cleanup issues"]
    window.close()


def test_main_window_inspect_failed_jobs_no_failed_shows_info(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_no_failed.mufinder", "InspectNoFailed")
    job = service.add_job(
        doc,
        label="job_defined",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._inspect_failed_jobs([job])

    assert infos == ["No failed jobs"]
    window.close()


def test_main_window_failure_diagnostics_and_inspect_dialog(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_failed.mufinder", "InspectFailed")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    job.error_message = "process failed"
    window._document = doc

    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    err_path = output_dir / f"{job.label}.err"
    err_path.write_text("line 1\nfatal error\nline 3\n", encoding="utf-8")

    diagnostics = window._failure_diagnostics_for_job(job)
    assert "Recorded failure message" in diagnostics
    assert "2: fatal error" in diagnostics

    captured_details: list[str] = []

    class _FakeMessageBox:
        class Icon:
            Information = 1

        def __init__(self, *_args, **_kwargs) -> None:
            self._details = ""

        def setIcon(self, *_args, **_kwargs) -> None:
            return None

        def setWindowTitle(self, *_args, **_kwargs) -> None:
            return None

        def setText(self, *_args, **_kwargs) -> None:
            return None

        def setInformativeText(self, *_args, **_kwargs) -> None:
            return None

        def setDetailedText(self, details: str) -> None:
            self._details = details

        def setSizeGripEnabled(self, *_args, **_kwargs) -> None:
            return None

        def setWindowFlag(self, *_args, **_kwargs) -> None:
            return None

        def exec(self) -> int:
            captured_details.append(self._details)
            return 0

    monkeypatch.setattr(main_window_module, "QMessageBox", _FakeMessageBox)

    window._inspect_failed_jobs([job])

    assert len(captured_details) == 1
    assert "job_failed.err" in captured_details[0]
    assert "2: fatal error" in captured_details[0]
    window.close()


def test_main_window_failure_diagnostics_prefers_numbered_error_files(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_numbered_err.mufinder", "InspectNumberedErr")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    window._document = doc

    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "job_failed.0001.err").write_text("rank 1 error\n", encoding="utf-8")
    (output_dir / "job_failed.0002.err").write_text("rank 2 error\n", encoding="utf-8")
    (output_dir / "unrelated.err").write_text("other error\n", encoding="utf-8")

    candidates = window._error_file_candidates(job, output_dir)
    diagnostics = window._failure_diagnostics_for_job(job)

    assert [path.name for path in candidates] == ["job_failed.0001.err", "job_failed.0002.err"]
    assert "[job_failed.0001.err]" in diagnostics
    assert "[job_failed.0002.err]" in diagnostics
    assert "unrelated.err" not in diagnostics
    window.close()


def test_main_window_failure_diagnostics_includes_slurm_output(
    main_window,
    tmp_path: Path,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_slurm_out.mufinder", "InspectSlurmOut")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    job.remote_job_id = "771300"
    window._document = doc

    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "slurm-771300.out").write_text(
        "running job\nslurm error: out of memory\n",
        encoding="utf-8",
    )

    candidates = window._error_file_candidates(job, output_dir)
    diagnostics = window._failure_diagnostics_for_job(job)

    assert [path.name for path in candidates] == ["slurm-771300.out"]
    assert "[slurm-771300.out]" in diagnostics
    assert "2: slurm error: out of memory" in diagnostics
    window.close()


def test_main_window_failure_diagnostics_without_output_directory(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_no_output.mufinder", "InspectNoOutput")
    job = service.add_job(
        doc,
        label="job_failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    window._document = doc

    diagnostics = window._failure_diagnostics_for_job(job)

    assert "Output directory: not found" in diagnostics
    window.close()


def test_main_window_error_file_candidates_fallback_and_excerpt_helpers(
    main_window, tmp_path: Path
) -> None:
    output_dir = tmp_path / "errors"
    output_dir.mkdir()
    (output_dir / "job_x.err").write_text("line 1\nerror here\nline 3\n", encoding="utf-8")
    (output_dir / "empty.err").write_text("", encoding="utf-8")

    service = ProjectService()
    doc = service.create_project(tmp_path / "helpers_doc.mufinder", "HelpersDoc")
    job = service.add_job(
        doc,
        label="job_x",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )

    window = main_window
    candidates = window._error_file_candidates(job, output_dir)
    assert [path.name for path in candidates] == ["job_x.err"]

    assert MainWindow._read_error_contexts(output_dir / "empty.err") == "[File is empty]"
    assert MainWindow._read_error_contexts(output_dir / "missing.err").startswith("Could not read file")
    assert "2: error here" in MainWindow._read_error_contexts(output_dir / "job_x.err")
    window.close()


def test_read_bounded_tail_small_file_matches_read_text(tmp_path: Path) -> None:
    path = tmp_path / "small.out"
    content = "first line\nsecond line\nthird line\n"
    path.write_text(content, encoding="utf-8")

    assert MainWindow._read_bounded_tail(path) == content
    assert MainWindow._read_error_contexts(path) != "[File is empty]"
    assert MainWindow._read_text_excerpt(path) == content.strip()


def test_read_bounded_tail_large_file_returns_only_tail(tmp_path: Path) -> None:
    path = tmp_path / "huge.out"
    limit = MainWindow._TAIL_READ_LIMIT_BYTES

    first_line = "FIRST-LINE-SENTINEL " + ("x" * 200) + "\n"
    filler_line = "y" * 500 + "\n"
    filler_count = (limit // len(filler_line)) + 10
    last_line = "LAST-LINE-SENTINEL\n"

    with path.open("w", encoding="utf-8") as handle:
        handle.write(first_line)
        for _ in range(filler_count):
            handle.write(filler_line)
        handle.write(last_line)

    assert path.stat().st_size > limit

    tail_text = MainWindow._read_bounded_tail(path)
    assert len(tail_text.encode("utf-8")) <= limit + len("[... truncated] ...\n")
    assert "FIRST-LINE-SENTINEL" not in tail_text
    assert "LAST-LINE-SENTINEL" in tail_text
    assert tail_text.startswith("[... truncated] ...\n")

    excerpt = MainWindow._read_text_excerpt(path, max_chars=10**9)
    assert "FIRST-LINE-SENTINEL" not in excerpt
    assert "LAST-LINE-SENTINEL" in excerpt

    error_contexts = MainWindow._read_error_contexts(path)
    assert "FIRST-LINE-SENTINEL" not in error_contexts


def test_main_window_prompt_failed_job_cleanup_choices(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeMessageBox:
        class Icon:
            Question = 1

        class StandardButton:
            Yes = 1
            No = 2
            Cancel = 3

        _answer = 1

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def setIcon(self, *_args, **_kwargs) -> None:
            return None

        def setWindowTitle(self, *_args, **_kwargs) -> None:
            return None

        def setText(self, *_args, **_kwargs) -> None:
            return None

        def setInformativeText(self, *_args, **_kwargs) -> None:
            return None

        def setStandardButtons(self, *_args, **_kwargs) -> None:
            return None

        def setDefaultButton(self, *_args, **_kwargs) -> None:
            return None

        def exec(self) -> int:
            return _FakeMessageBox._answer

    window = main_window
    monkeypatch.setattr(main_window_module, "QMessageBox", _FakeMessageBox)

    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.Yes
    assert window._prompt_failed_job_cleanup(2) is True

    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.No
    assert window._prompt_failed_job_cleanup(2) is False

    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.Cancel
    assert window._prompt_failed_job_cleanup(2) is None
    window.close()


def test_main_window_delete_path_retries_transient_permission_error(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    target_dir = tmp_path / "transient_dir"
    target_dir.mkdir()
    (target_dir / "inner.txt").write_text("x", encoding="utf-8")

    real_rmtree = main_window_module.shutil.rmtree
    calls = {"n": 0}

    def flaky_rmtree(path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "Access is denied", str(path))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(main_window_module.shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(main_window_module.time, "sleep", lambda _delay: None)

    window._delete_path(target_dir)

    assert calls["n"] >= 2
    assert not target_dir.exists()
    window.close()


def test_main_window_error_file_candidates_handles_missing_directory(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "missing_errors.mufinder", "MissingErrors")
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )

    assert window._error_file_candidates(job, tmp_path / "does_not_exist") == []
    window.close()


# ---------------------------------------------------------------------------
# Job-lifecycle hardening: retry retrieval (rec #4) + dismiss (rec #8)
# ---------------------------------------------------------------------------
def _connect_cluster_runner(window, cluster_name: str) -> None:
    """Put the ClusterController into a connected state for the given cluster."""

    window._cluster_runner._connected = True
    window._cluster_runner._connected_cluster_name = cluster_name


def _make_failed_cluster_job(service, doc, *, label: str, cluster: str = "scarf") -> Job:
    job = service.add_job(
        doc,
        label=label,
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    job.execution_target = cluster
    job.remote_job_id = "99123"
    job.remote_directory = "/remote/work/" + label
    return job


def test_dashboard_retry_retrieval_button_enablement(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "retry.mufinder", "Retry")

    connected_job = _make_failed_cluster_job(service, doc, label="conn", cluster="scarf")
    disconnected_job = _make_failed_cluster_job(service, doc, label="disc", cluster="other")
    no_meta_job = _make_failed_cluster_job(service, doc, label="nometa", cluster="scarf")
    no_meta_job.remote_directory = None
    local_job = service.add_job(
        doc,
        label="localjob",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    local_job.status = JobStatus.FAILED
    local_job.execution_target = "local"

    window._document = doc
    _connect_cluster_runner(window, "scarf")
    window._refresh_jobs_table()

    def retry_btn_for(job_id):
        for row in range(window._jobs_table.rowCount()):
            item = window._jobs_table.item(row, window._JOB_LABEL_COLUMN)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == str(job_id):
                return window._jobs_table.cellWidget(row, window._JOB_RETRY_RETRIEVAL_COLUMN)
        return None

    assert retry_btn_for(connected_job.id).text() == "Retry retrieval"
    assert retry_btn_for(connected_job.id).isEnabled() is True
    # Cluster not connected for this job's target -> disabled.
    assert retry_btn_for(disconnected_job.id).isEnabled() is False
    # Missing remote directory -> disabled.
    assert retry_btn_for(no_meta_job.id).isEnabled() is False
    # Local job -> disabled.
    assert retry_btn_for(local_job.id).isEnabled() is False
    window.close()


def test_dashboard_retry_retrieval_click_dispatches_to_runner(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "retry_click.mufinder", "RetryClick")
    job = _make_failed_cluster_job(service, doc, label="conn", cluster="scarf")

    window._document = doc
    _connect_cluster_runner(window, "scarf")
    monkeypatch.setattr(window, "_job_output_directory", lambda _job: tmp_path)

    retrieve_calls: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        window._cluster_runner,
        "retrieve_outputs_for",
        lambda job_id, **kwargs: retrieve_calls.append((job_id, kwargs)),
    )
    monkeypatch.setattr(window, "_request_dashboard_flush", lambda **kwargs: None)

    window._refresh_jobs_table()
    row = 0
    retry_btn = window._jobs_table.cellWidget(row, window._JOB_RETRY_RETRIEVAL_COLUMN)
    retry_btn.click()

    # The job moved to RETRIEVING and the runner was asked to re-download.
    assert job.status == JobStatus.RETRIEVING
    assert len(retrieve_calls) == 1
    called_job_id, kwargs = retrieve_calls[0]
    assert called_job_id == job.id
    assert kwargs["remote_job_id"] == "99123"
    assert kwargs["remote_directory"] == "/remote/work/conn"
    window.close()


def test_dashboard_dismiss_hides_row_after_flush(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismiss.mufinder", "Dismiss")
    failed_job = service.add_job(
        doc,
        label="failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    failed_job.status = JobStatus.FAILED
    running_job = service.add_job(
        doc,
        label="running",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    running_job.status = JobStatus.RUNNING

    window._document = doc
    window._refresh_jobs_table()
    assert window._jobs_table.rowCount() == 2

    # The running job's Dismiss button must be disabled; the failed job's enabled.
    def dismiss_btn_for(job_id):
        for row in range(window._jobs_table.rowCount()):
            item = window._jobs_table.item(row, window._JOB_LABEL_COLUMN)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == str(job_id):
                return window._jobs_table.cellWidget(row, window._JOB_DISMISS_COLUMN)
        return None

    assert dismiss_btn_for(failed_job.id).isEnabled() is True
    assert dismiss_btn_for(running_job.id).isEnabled() is False

    dismiss_btn_for(failed_job.id).click()
    # Dismissal schedules a coalesced flush; force it to rebuild the table now.
    window._flush_dashboard()

    # After dismissal + rebuild, the failed row disappears; running row remains.
    assert failed_job.dismissed_at is not None
    assert window._jobs_table.rowCount() == 1
    remaining = window._jobs_table.item(0, window._JOB_LABEL_COLUMN)
    assert remaining.data(Qt.ItemDataRole.UserRole) == str(running_job.id)
    # Provenance milestone recorded.
    assert doc.provenance[-1].action == "job_dismissed"
    window.close()


def test_dashboard_dismiss_multi_select(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismiss_multi.mufinder", "DismissMulti")
    job0 = service.add_job(doc, label="d0", workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF, calc_type="castep_convergence_cutoff")
    job1 = service.add_job(doc, label="d1", workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF, calc_type="castep_convergence_cutoff")
    job0.status = JobStatus.FAILED
    job1.status = JobStatus.STOPPED

    window._document = doc
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )
    window._refresh_jobs_table()

    dismiss_btn = window._jobs_table.cellWidget(0, window._JOB_DISMISS_COLUMN)
    assert isinstance(dismiss_btn, SelectionSnapshotPushButton)
    dismiss_btn.selection_snapshot_rows = (0, 1)
    dismiss_btn.clicked.emit()
    window._flush_dashboard()

    assert job0.dismissed_at is not None
    assert job1.dismissed_at is not None
    assert window._jobs_table.rowCount() == 0
    window.close()


# --------------------------------------------------------------------------- #
# Dismissed failed jobs stay resubmittable from their workflow row
# --------------------------------------------------------------------------- #
#
# Regression for the stranded-row bug: dismissing a failed job from the jobs
# dashboard removes its Resubmit/Reset controls, but its workflow row kept
# projecting "failed" and stayed unsubmittable forever. The row now projects
# "ready" and its submit resets the *same* job back to defined before dispatch.
def _fail_and_dismiss(window: MainWindow, service: ProjectService, job: Job) -> None:
    service.submit_job(job)
    service.fail_job(job, "castep exploded")
    window._on_dismiss_from_dashboard([job.id])
    assert job.dismissed_at is not None
    assert job.status is JobStatus.FAILED


def test_dismissed_pristine_relaxation_job_row_is_resubmittable(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismissed_pristine.mufinder", "DismissedPristine")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    row = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={})
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [{"label": "row_1", "fix_all_cell": False, "params": {}, "added_advanced": []}],
    }
    window._refresh_ui_after_project_change()

    service.submit_job(job)
    service.fail_job(job, "castep exploded")
    window._refresh_ui_after_project_change()
    status_column = window._pristine_relaxation_view._STATUS_COLUMN
    assert window._pristine_relaxation_view._table.item(0, status_column).text() == "failed"

    # Dismissing from the dashboard removes the job's only generic controls...
    window._on_dismiss_from_dashboard([job.id])
    window._refresh_ui_after_project_change()

    # ...so the workflow row becomes the recovery path.
    assert window._pristine_relaxation_view._table.item(0, status_column).text() == "ready"

    submitted: list[list[Job]] = []
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda jobs, **_kwargs: submitted.append(list(jobs))
    )
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._pristine_relaxation_controller.submit(["row_1"], target="local")

    assert [[j.id for j in jobs] for jobs in submitted] == [[job.id]]
    assert job.status is JobStatus.DEFINED
    assert job.dismissed_at is None
    assert job.error_message is None
    # The row reused its bound job — no duplicate was minted for it.
    assert len(doc.jobs) == 1
    assert definition.parameters["job_ids_by_label"] == {"row_1": str(job.id)}
    window.close()


def test_dismissed_convergence_cutoff_job_row_is_resubmittable(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismissed_cutoff.mufinder", "DismissedCutoff")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )
    window._document = doc
    window._refresh_ui_after_project_change()

    service.submit_job(job)
    service.fail_job(job, "castep exploded")
    window._refresh_ui_after_project_change()
    status_column = window._convergence_view._STATUS_COLUMN
    assert window._convergence_view._table.item(0, status_column).text() == "failed"

    window._on_dismiss_from_dashboard([job.id])
    window._refresh_ui_after_project_change()

    assert window._convergence_view._table.item(0, status_column).text() == "ready"
    # The cutoff panel's Submit actions are gated on the projected row status,
    # so they become available again for the dismissed row.
    assert not workflow_table_utils.workflow_row_is_locked(
        window._convergence_view._row_status(0)
    )

    submitted: list[list[Job]] = []
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda jobs, **_kwargs: submitted.append(list(jobs))
    )
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._convergence_controller.submit([300.0], target="local")

    assert [[j.id for j in jobs] for jobs in submitted] == [[job.id]]
    assert job.status is JobStatus.DEFINED
    assert job.dismissed_at is None
    assert len(doc.jobs) == 1
    # The lazy reset must NOT detach the sweep point from its job the way the
    # explicit "Reset failed jobs" action does, or the row would go back to
    # "ready" with no job and a second submit would mint a duplicate.
    assert definition.parameters["job_ids_by_cutoff"] == {
        main_window_module.MainWindow._cutoff_key(300.0): str(job.id)
    }
    window.close()


def test_dismissed_row_submit_aborts_when_cleanup_prompt_cancelled(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismissed_cancel.mufinder", "DismissedCancel")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )
    window._document = doc
    window._refresh_ui_after_project_change()
    _fail_and_dismiss(window, service, job)

    submitted: list[list[Job]] = []
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda jobs, **_kwargs: submitted.append(list(jobs))
    )
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: None)

    window._convergence_controller.submit([300.0], target="local")

    assert submitted == []
    assert job.status is JobStatus.FAILED
    assert job.dismissed_at is not None
    window.close()


def test_non_dismissed_failed_row_keeps_failed_status_and_is_not_reset(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed job that was never dismissed keeps its current behavior.

    Its recovery path is still the dashboard Resubmit/Reset controls, which are
    still on screen because the row was never filtered out.
    """

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "undismissed_cutoff.mufinder", "UndismissedCutoff")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )
    window._document = doc
    service.submit_job(job)
    service.fail_job(job, "castep exploded")
    window._refresh_ui_after_project_change()

    assert (
        window._convergence_view._table.item(
            0, window._convergence_view._STATUS_COLUMN
        ).text()
        == "failed"
    )

    prompted: list[int] = []
    monkeypatch.setattr(
        window, "_prompt_failed_job_cleanup", lambda count: prompted.append(count) or False
    )
    infos: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda *args, **_kwargs: infos.append((args[1], args[2])),
    )

    window._convergence_controller.submit([300.0], target="local")

    assert prompted == []
    assert job.status is JobStatus.FAILED
    assert job.dismissed_at is None
    window.close()


def test_dismissed_row_submit_regenerates_job_inputs(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resubmitted job's inputs are rebuilt from the current configuration.

    ``materialize`` only rewrites a job directory for a DEFINED job, and it runs
    while the plan is being built — before the lazy reset. Without a second
    materialize pass the row would resubmit against the stale ``.cell`` the
    failed run left behind (``_clear_directory_contents`` deliberately preserves
    ``.cell``/``.param``), diverging from the explicit reset-then-submit path.
    """

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismissed_rewrite.mufinder", "DismissedRewrite")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    window._on_pristine_relaxation_configured(
        main_window_module.PristineRelaxationConfig(
            name_prefix="pristine_relax",
            rows=(main_window_module.PristineRelaxationRow(label="row_1", parameter_values={}),),
        )
    )
    job = window._pristine_relaxation_controller.materialize(["row_1"])[0]
    cell_path = doc.project_root / "files" / "pristine_relaxation" / job.label / f"{job.label}.cell"
    assert "KPOINT_MP_GRID" not in cell_path.read_text(encoding="utf-8")

    _fail_and_dismiss(window, service, job)

    # The configuration changes while the row is stranded.
    doc.project.recommended_kpoint_grid = (4, 4, 4)
    doc.project.recommended_kpoint_lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    submitted: list[list[Job]] = []
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda jobs, **_kwargs: submitted.append(list(jobs))
    )
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._pristine_relaxation_controller.submit(["row_1"], target="local")

    assert [[j.id for j in jobs] for jobs in submitted] == [[job.id]]
    assert "KPOINT_MP_GRID 4 4 4" in cell_path.read_text(encoding="utf-8")
    assert len(doc.jobs) == 1
    window.close()


# ---------------------------------------------------------------------------
# Row-menu Cancel / Inspect Failure / Reset Failed Jobs routing
# ---------------------------------------------------------------------------
_ROW_ACTION_VIEWS = [
    ("_convergence_view", "_convergence_controller", [300.0]),
    ("_kpoint_convergence_view", "_kpoint_convergence_controller", [(2, 2, 2)]),
    ("_pristine_relaxation_view", "_pristine_relaxation_controller", ["relax_1"]),
    ("_muon_relaxation_view", "_muon_relaxation_controller", ["positivemuon:siteA"]),
    ("_uep_view", "_uep_controller", ["uep_1"]),
    ("_quantum_corrections_view", "_quantum_corrections_controller", ["site-a"]),
    ("_energy_barriers_view", "_energy_barriers_controller", ["path-a"]),
]


_ROW_ACTION_SIGNALS = [
    ("cancel_requested", "_cancel_jobs"),
    ("inspect_failed_requested", "_inspect_failed_jobs"),
    ("reset_failed_requested", "_reset_failed_jobs"),
]


@pytest.mark.parametrize(
    ("view_attr", "controller_attr", "keys", "signal_name", "handler_name"),
    [
        (*view, *signal)
        for view in _ROW_ACTION_VIEWS
        for signal in _ROW_ACTION_SIGNALS
        # Quantum rows reset through "Reset to stage 1", not a failed-job reset.
        if not (view[0] == "_quantum_corrections_view" and signal[0] == "reset_failed_requested")
    ],
)
def test_main_window_routes_workflow_row_actions_through_the_controller(
    main_window,
    monkeypatch: pytest.MonkeyPatch,
    view_attr: str,
    controller_attr: str,
    keys: list[object],
    signal_name: str,
    handler_name: str,
) -> None:
    # The signal must actually reach MainWindow: k-point and UEP once shipped a
    # declared-and-connected reset signal that no widget ever emitted. The
    # energy-barriers view is created lazily, so its wiring happens in
    # _ensure_energy_barriers_view rather than __init__.
    window = main_window
    if view_attr == "_energy_barriers_view":
        window._ensure_energy_barriers_view()
    job = Job(
        label="row-job",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_relaxation",
        status=JobStatus.FAILED,
    )

    requested_keys: list[list[object]] = []
    handled: list[list[Job]] = []
    monkeypatch.setattr(
        getattr(window, controller_attr),
        "jobs_for_keys",
        lambda values: (requested_keys.append(list(values)), [job])[1],
    )
    monkeypatch.setattr(window, handler_name, lambda jobs: handled.append(jobs))

    getattr(getattr(window, view_attr), signal_name).emit(keys)

    assert requested_keys == [keys]
    assert handled == [[job]]
    window.close()


def test_main_window_reset_refreshes_the_lazily_created_workflow_views(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Energy barriers and quantum corrections are not repainted by
    # _refresh_jobs_table, so a row reset from either table would leave a stale
    # "failed" status behind without these two refreshes.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_refresh.mufinder", "ResetRefresh")
    job = service.add_job(
        doc,
        label="neb-job",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type="castep_neb",
    )
    job.status = JobStatus.FAILED
    window._document = doc

    refreshed: list[str] = []
    monkeypatch.setattr(
        window, "_refresh_energy_barriers_view", lambda: refreshed.append("energy_barriers")
    )
    monkeypatch.setattr(
        window, "_refresh_quantum_corrections_view", lambda: refreshed.append("quantum")
    )
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._reset_failed_jobs([job])

    assert refreshed == ["energy_barriers", "quantum"]
    assert job.status == JobStatus.DEFINED
    window.close()


def test_main_window_row_reset_clears_the_displayed_status_for_a_bound_job(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-row reset must repaint the row, not just mutate the Job object.

    Cutoff convergence clears its row by detaching the job from
    ``job_ids_by_cutoff``; the other six workflows keep their label->job-id
    mapping pointing at the reset job, so the row only clears because the job is
    now ``defined`` and ``effective_row_status`` projects a submittable status.
    This drives the real refresh path (no stubbed refresh) and asserts the Status
    *cell*, which is what the user actually sees.
    """

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "row_reset_repaint.mufinder", "RowResetRepaint")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    row = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={})
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    failed_job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    failed_job.status = JobStatus.FAILED
    definition.parameters["job_ids_by_label"] = {"row_1": str(failed_job.id)}
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [{"label": "row_1", "fix_all_cell": False, "params": {}, "added_advanced": []}],
    }

    window._refresh_ui_after_project_change()

    view = window._pristine_relaxation_view
    assert view._row_status(0) == "failed"
    # The context menu offers the reset precisely because the row reads "failed".
    assert view._row_status(0) in workflow_table_utils.RESETTABLE_ROW_STATUSES

    # Decline the output cleanup so this asserts the state change alone.
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda _count: False)

    window._pristine_relaxation_view.reset_failed_requested.emit(["row_1"])

    assert failed_job.status == JobStatus.DEFINED
    # The row stays bound to the same job - no duplicate is minted - and now
    # reads as submittable again.
    assert definition.parameters["job_ids_by_label"] == {"row_1": str(failed_job.id)}
    assert view._row_status(0) not in workflow_table_utils.RESETTABLE_ROW_STATUSES
    assert view._row_status(0) in READY_WORKFLOW_ROW_STATUSES
    window.close()


# --------------------------------------------------------------------------- #
# "Continue relaxation…" — WP6 controller action wiring
# --------------------------------------------------------------------------- #
class _FakeContinueRelaxationDialog:
    """Test double for ``ContinueRelaxationDialog``: records constructor kwargs
    and returns a canned accept/cancel result without any Qt event loop."""

    captured_kwargs: list[dict[str, object]] = []
    accept: bool = True
    result_to_return: tuple[dict[str, object], tuple[str, ...]] = ({}, ())

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        type(self).captured_kwargs.append(kwargs)

    def exec(self) -> int:
        return (
            main_window_module.QDialog.DialogCode.Accepted
            if type(self).accept
            else main_window_module.QDialog.DialogCode.Rejected
        )

    def edited_parameters(self) -> tuple[dict[str, object], tuple[str, ...]]:
        return type(self).result_to_return


@pytest.fixture()
def fake_continue_dialog(monkeypatch: pytest.MonkeyPatch):
    _FakeContinueRelaxationDialog.captured_kwargs = []
    _FakeContinueRelaxationDialog.accept = True
    _FakeContinueRelaxationDialog.result_to_return = ({}, ())
    monkeypatch.setattr(
        main_window_module, "ContinueRelaxationDialog", _FakeContinueRelaxationDialog
    )
    return _FakeContinueRelaxationDialog


class _FakeClusterSubmissionDialogForContinue:
    DialogCode = SimpleNamespace(Accepted=1)

    def __init__(self, **_kwargs) -> None:
        pass

    def exec(self) -> int:
        return 1

    def config(self):
        return SimpleNamespace(
            cluster_name="alpha",
            working_directory="/remote/work",
            username="alice",
            job_name_prefix="continue",
            num_processes=16,
            walltime="01:00:00",
        )

    def password(self) -> str:
        return ""


def _wire_fake_cluster_submission(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, *, cluster_name: str = "alpha"
) -> list[dict[str, object]]:
    """Mirrors ``test_main_window_resubmits_failed_cluster_job_...``'s cluster
    doubles: connect + submit succeed, and any existence-check/overwrite-delete
    call fails the test outright, since a continuation must never reach them."""

    cluster = ClusterConfig(
        name=cluster_name,
        hostname=f"{cluster_name}.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )
    monkeypatch.setattr(
        main_window_module, "ClusterSubmissionDialog", _FakeClusterSubmissionDialogForContinue
    )
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner, "connect_to_cluster", lambda _config, password=None: None
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda _path: pytest.fail("Continue must never check for an existing remote directory"),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remove_remote_directory",
        lambda _path: pytest.fail("Continue must never delete the preserved remote directory"),
    )
    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs) or "new-remote-job-id",
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_a, **_k: None)
    return submit_calls


def _pristine_continue_fixture(tmp_path: Path):
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "continue_pristine.mufinder", "ContinuePristine")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    row = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={"geom_max_iter": 120})
    output_root = doc.project_root / "files" / "pristine_relaxation"
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=output_root,
    )
    job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [
            {"label": "row_1", "fix_all_cell": False, "params": {"geom_max_iter": 120}, "added_advanced": []}
        ],
    }
    window._refresh_ui_after_project_change()
    materialized = window._pristine_relaxation_controller.materialize(["row_1"])
    assert materialized == [job]
    param_path = output_root / job.label / f"{job.label}.param"
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")
    return window, service, doc, definition, job, output_root, param_path


def _muon_continue_fixture(tmp_path: Path):
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "continue_muon.mufinder", "ContinueMuon")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    site = MuonSite(label="Site 1", fractional_position=[0.1, 0.2, 0.3], status="staged")
    doc.sites[site.id] = site
    window._document = doc

    row = main_window_module.MuonRelaxationRow(
        site_id=site.id,
        label="Site 1",
        initial_fractional_position=(0.1, 0.2, 0.3),
        charge_state="positive_muon",
        parameter_values={"geom_max_iter": 120},
    )
    output_root = doc.project_root / "files" / "muon_relaxation"
    definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_muon_relaxation_config(
            main_window_module.MuonRelaxationConfig(rows=(row,))
        ),
        output_directory=output_root,
    )
    job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.muon_relaxation_calc_type("positive_muon"),
    )
    definition.parameters["job_ids_by_label"] = {"positivemuon:Site 1": str(job.id)}
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["muon_relaxation_form"] = window._serialize_muon_relaxation_config(
        main_window_module.MuonRelaxationConfig(rows=(row,))
    )
    window._refresh_ui_after_project_change()
    materialized = window._muon_relaxation_controller.materialize(["positivemuon:Site 1"])
    assert materialized == [job]
    seed_name = definition.parameters["job_seed_names_by_label"]["positivemuon:Site 1"]
    param_path = output_root / "positivemuon" / seed_name / f"{seed_name}.param"
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")
    return window, service, doc, definition, job, output_root, param_path


def test_continue_relaxation_pristine_cluster_dismissed_failed_row_end_to_end(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )

    job.execution_target = "alpha"
    job.status = JobStatus.FAILED
    job.remote_directory = "/remote/work/pristine_relax_row_1"
    job.remote_job_id = "old-remote-id"
    job.error_message = "previous castep failure"
    job.dismissed_at = datetime.now(UTC)
    window._refresh_ui_after_project_change()
    assert window._pristine_relaxation_view._row_status(0) == "ready"  # dismissed -> displays "ready"

    fake_continue_dialog.result_to_return = ({"geom_max_iter": 500}, ())
    submit_calls = _wire_fake_cluster_submission(window, monkeypatch, cluster_name="alpha")
    cleanup_prompted: list[int] = []
    monkeypatch.setattr(
        window,
        "_prompt_failed_job_cleanup",
        lambda count: cleanup_prompted.append(count) or pytest.fail(
            "Continue must never trigger the dismissed-job cleanup prompt"
        ),
    )

    window._on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    # Row params mutated in place.
    config = window._deserialize_pristine_relaxation_config(definition.parameters["configuration"])
    assert {row.label: row.parameter_values for row in config.rows} == {
        "row_1": {"geom_max_iter": 500}
    }
    # Same directory rewritten with the new parameters.
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 500" in rendered
    assert "GEOM_MAX_ITER : 120" not in rendered
    # No duplicate job; job_ids_by_label untouched.
    assert list(doc.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {"row_1": str(job.id)}
    # Submitted into the same remote directory without any existence/overwrite prompt.
    assert len(submit_calls) == 1
    assert submit_calls[0]["remote_directory"] == "/remote/work/pristine_relax_row_1"
    assert submit_calls[0]["allow_existing_remote_directory"] is True
    assert submit_calls[0]["reuse_existing_files"] is False
    # dismissed_at/error/old remote id cleared by prepare_job_continuation, and
    # the job is back in an active state after a successful cluster submit.
    assert job.status == JobStatus.QUEUED
    assert job.dismissed_at is None
    assert job.error_message is None
    assert job.remote_job_id == "new-remote-job-id"
    assert job.remote_directory == "/remote/work/pristine_relax_row_1"
    assert cleanup_prompted == []
    # Journal-visible provenance milestone.
    assert any(record.action == "job_continuation" for record in doc.provenance)
    window.close()


def test_continue_relaxation_muon_cluster_completed_row_end_to_end(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _muon_continue_fixture(
        tmp_path
    )

    job.execution_target = "alpha"
    job.status = JobStatus.COMPLETED
    job.remote_directory = "/remote/work/positivemuon_site1"
    window._refresh_ui_after_project_change()

    fake_continue_dialog.result_to_return = ({"geom_max_iter": 777}, ())
    submit_calls = _wire_fake_cluster_submission(window, monkeypatch, cluster_name="alpha")

    window._on_continue_relaxation_requested_from_muon_relaxation("positivemuon:Site 1")

    config = window._deserialize_muon_relaxation_config(definition.parameters["configuration"])
    assert {row.charge_state: row.parameter_values for row in config.rows} == {
        "positive_muon": {"geom_max_iter": 777}
    }
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 777" in rendered
    assert list(doc.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {"positivemuon:Site 1": str(job.id)}
    assert len(submit_calls) == 1
    assert submit_calls[0]["remote_directory"] == "/remote/work/positivemuon_site1"
    assert submit_calls[0]["allow_existing_remote_directory"] is True
    assert job.status == JobStatus.QUEUED
    assert any(record.action == "job_continuation" for record in doc.provenance)
    window.close()


def test_continue_relaxation_local_job_round_trip(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.FAILED
    job.error_message = "boom"
    assert job.execution_target == "local"

    # A local job's checkpoint probe is a plain Path check: write a real
    # ``.check`` file so the probe reports "found".
    checkpoint_path = param_path.parent / f"{job.label}.check"
    checkpoint_path.write_text("", encoding="utf-8")

    fake_continue_dialog.result_to_return = ({"geom_max_iter": 42}, ())
    local_submitted: list[list[Job]] = []
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda jobs, **_kwargs: local_submitted.append(list(jobs))
    )
    monkeypatch.setattr(
        window,
        "_submit_jobs_on_cluster",
        lambda *_a, **_k: pytest.fail("A local job must submit through the local path"),
    )

    window._on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    probe = fake_continue_dialog.captured_kwargs[-1]["checkpoint_probe"]
    assert probe.status == main_window_module.CheckpointProbeResult.found().status
    assert [[j.id for j in jobs] for jobs in local_submitted] == [[job.id]]
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 42" in rendered
    window.close()


def test_continue_relaxation_cluster_not_connected_dialog_still_opens_with_unavailable_probe(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.execution_target = "alpha"
    job.status = JobStatus.FAILED
    job.remote_directory = "/remote/work/pristine_relax_row_1"
    assert window._cluster_runner.is_connected is False

    monkeypatch.setattr(
        window._cluster_runner,
        "check_remote_files",
        lambda *_a, **_k: pytest.fail("Not connected: the checkpoint probe must not hit the network"),
    )
    fake_continue_dialog.accept = False  # cancel: no submission wiring needed

    window._on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    assert len(fake_continue_dialog.captured_kwargs) == 1
    probe = fake_continue_dialog.captured_kwargs[0]["checkpoint_probe"]
    assert probe.status == main_window_module.CheckpointProbeResult.unavailable("x").status
    assert "alpha" in probe.reason
    window.close()


def test_continue_relaxation_cancel_leaves_everything_unchanged(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.FAILED
    job.error_message = "boom"
    before_config = copy.deepcopy(definition.parameters["configuration"])

    fake_continue_dialog.accept = False
    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda *_a, **_k: pytest.fail("Cancel must not submit")
    )
    monkeypatch.setattr(
        window, "_submit_jobs_on_cluster", lambda *_a, **_k: pytest.fail("Cancel must not submit")
    )

    window._on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    assert definition.parameters["configuration"] == before_config
    assert job.status == JobStatus.FAILED
    assert job.error_message == "boom"
    assert not any(record.action == "job_continuation" for record in doc.provenance)
    window.close()


def test_continue_relaxation_gating_disabled_running_enabled_not_converged(
    qapp, tmp_path: Path
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )

    job.status = JobStatus.RUNNING
    assert window._pristine_relaxation_controller.row_is_continuable("row_1") is False

    job.status = JobStatus.QUEUED
    assert window._pristine_relaxation_controller.row_is_continuable("row_1") is False

    # A completed-but-unconverged job displays "not converged" on the row, but
    # its raw status (COMPLETED) is continuable.
    job.status = JobStatus.COMPLETED
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED, convergence_achieved=False)
    doc.runs[run.id] = run
    window._refresh_ui_after_project_change()
    assert window._pristine_relaxation_view._row_status(0) == "not converged"
    assert window._pristine_relaxation_controller.row_is_continuable("row_1") is True
    window.close()


def test_continue_relaxation_unedited_real_dialog_shows_no_parameter_changes(
    qapp, tmp_path: Path
) -> None:
    """WP4 contract check: ``old_param_text`` must be rendered through the same
    ``render_param_text`` callable applied to the row's pre-edit values -- the
    exact values seeded into the dialog's own editor -- so an unedited dialog
    reads "No parameter changes." This constructs the REAL
    ``ContinueRelaxationDialog`` (not the fake double the other tests use) to
    prove the seeding path this module wires up actually satisfies that
    contract, rather than merely asserting the fake was called correctly."""

    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.FAILED

    seed = window._pristine_relaxation_view.continue_relaxation_dialog_seed("row_1")
    assert seed is not None

    def render_param_text(values, advanced):
        return window._pristine_relaxation_controller.render_row_param_text(
            "row_1", values, advanced
        )

    resolved_values, resolved_advanced = window._resolve_continue_relaxation_seed_values(seed)
    old_param_text = render_param_text(resolved_values, resolved_advanced)

    dialog = main_window_module.ContinueRelaxationDialog(
        **seed,
        old_param_text=old_param_text,
        render_param_text=render_param_text,
        checkpoint_probe=main_window_module.CheckpointProbeResult.absent(),
        execution_target=job.execution_target,
        reused_directory=str(param_path.parent),
        parent=window,
    )
    try:
        assert dialog._diff_table.rowCount() == 0
        assert dialog._diff_no_changes_label.isHidden() is False
        assert dialog._diff_table.isHidden() is True
    finally:
        dialog.close()
    window.close()


def test_continue_relaxation_muon_unedited_real_dialog_shows_no_parameter_changes(
    qapp, tmp_path: Path
) -> None:
    """Muon counterpart of the pristine no-changes contract check above,
    using the charge-qualified key end to end (finding 7)."""

    window, service, doc, definition, job, output_root, param_path = _muon_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.FAILED

    seed = window._muon_relaxation_view.continue_relaxation_dialog_seed("positivemuon:Site 1")
    assert seed is not None

    def render_param_text(values, advanced):
        return window._muon_relaxation_controller.render_row_param_text(
            "positivemuon:Site 1", values, advanced
        )

    resolved_values, resolved_advanced = window._resolve_continue_relaxation_seed_values(seed)
    old_param_text = render_param_text(resolved_values, resolved_advanced)

    dialog = main_window_module.ContinueRelaxationDialog(
        **seed,
        old_param_text=old_param_text,
        render_param_text=render_param_text,
        checkpoint_probe=main_window_module.CheckpointProbeResult.absent(),
        execution_target=job.execution_target,
        reused_directory=str(param_path.parent),
        parent=window,
    )
    try:
        assert dialog._diff_table.rowCount() == 0
        assert dialog._diff_no_changes_label.isHidden() is False
        assert dialog._diff_table.isHidden() is True
    finally:
        dialog.close()
    window.close()


# --------------------------------------------------------------------------- #
# Opus audit follow-ups (findings 1-4)
# --------------------------------------------------------------------------- #
def test_continue_relaxation_warns_and_does_not_submit_when_materialize_fails(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    """Audit finding 1 (MainWindow half): when continue_job returns None after
    the user accepted the dialog -- e.g. because materialize swallowed an
    expansion error and rewrote nothing -- the row must not be silently
    submitted with stale on-disk inputs, and the user must be told."""

    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.FAILED
    fake_continue_dialog.result_to_return = ({"geom_max_iter": 500}, ())

    # Force the continuation's own materialize call to fail, exactly like the
    # real one does on a caught ValueError/StructureBridgeError.
    window._pristine_relaxation_controller.materialize = lambda keys: []

    monkeypatch.setattr(
        window, "_submit_jobs_locally", lambda *_a, **_k: pytest.fail("must not submit")
    )
    monkeypatch.setattr(
        window, "_submit_jobs_on_cluster", lambda *_a, **_k: pytest.fail("must not submit")
    )
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, text: warnings.append((title, text)),
    )

    window._on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    assert len(warnings) == 1
    # The job was still transitioned (prepare_job_continuation ran before the
    # failed materialize) -- it stays recoverable -- but nothing was submitted,
    # and no provenance milestone claims a continuation that never happened.
    assert job.status == JobStatus.DEFINED
    assert "GEOM_MAX_ITER : 500" not in param_path.read_text(encoding="utf-8")
    assert not any(record.action == "job_continuation" for record in doc.provenance)
    window.close()


class _FakeClusterSubmissionDialogSelectsCluster:
    """Like ``_FakeClusterSubmissionDialogForContinue``, but the user picks a
    DIFFERENT cluster than the one the row was preselected onto -- proving the
    cross-cluster misroute guard (audit finding 2)."""

    DialogCode = SimpleNamespace(Accepted=1)
    selected_cluster_name = "beta"

    def __init__(self, **_kwargs) -> None:
        pass

    def exec(self) -> int:
        return 1

    def config(self):
        return SimpleNamespace(
            cluster_name=type(self).selected_cluster_name,
            working_directory="/remote/work",
            username="bob",
            job_name_prefix="continue",
            num_processes=16,
            walltime="01:00:00",
        )

    def password(self) -> str:
        return ""


def test_continue_relaxation_cross_cluster_misroute_guard(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    """Audit finding 2: the submission dialog preselects the job's original
    cluster, but the user can still pick a different one. The continuation
    branch in ``_submit_jobs_on_cluster`` must not fire in that case -- a
    continued job's ``remote_directory`` string means nothing on a different
    cluster's filesystem. This asserts the NORMAL path is taken instead: the
    remote-directory existence check runs (it is skipped entirely for a
    same-cluster continuation), and nothing is deleted."""

    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.execution_target = "alpha"
    job.status = JobStatus.FAILED
    job.remote_directory = "/remote/work/pristine_relax_row_1"
    fake_continue_dialog.result_to_return = ({"geom_max_iter": 500}, ())

    alpha = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )
    beta = ClusterConfig(
        name="beta",
        hostname="beta.example",
        username="bob",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )
    monkeypatch.setattr(
        main_window_module, "ClusterSubmissionDialog", _FakeClusterSubmissionDialogSelectsCluster
    )
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [alpha, beta])
    monkeypatch.setattr(
        main_window_module,
        "load_cluster_config",
        lambda _settings, name: beta if name == "beta" else alpha,
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner, "connect_to_cluster", lambda _config, password=None: None
    )
    checked_paths: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda path: checked_paths.append(path) or False,
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remove_remote_directory",
        lambda _path: pytest.fail("must never delete on the misrouted path"),
    )
    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs) or "new-remote-job-id",
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_a, **_k: None)

    window._on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    # The continuation branch was NOT taken: the existence check ran, and the
    # job was submitted with allow_existing_remote_directory=False /
    # reuse_existing_files=False -- the ordinary cold-target path, not a
    # blind directory reuse onto a cluster that never saw this job.
    assert len(checked_paths) == 1
    assert len(submit_calls) == 1
    assert submit_calls[0]["allow_existing_remote_directory"] is False
    assert submit_calls[0]["reuse_existing_files"] is False
    assert job.execution_target == "beta"
    window.close()


def test_continue_relaxation_pristine_context_menu_action_invokes_the_real_handler(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    """Audit finding 3: prove the MainWindow wiring lines themselves (the
    ``set_continuable_predicate``/``continue_relaxation_requested.connect``
    calls in ``__init__``) are live, by driving the action through the REAL
    row context menu rather than calling the handler method directly."""

    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    # COMPLETED (not FAILED/STOPPED/CANCELLED): continuable but NOT resettable,
    # so the shared FakeMenu's "auto-fire the one enabled action" behavior
    # (tests/support/reset_menu.py) cannot land on "Reset Failed Jobs..."
    # instead of the action under test.
    job.status = JobStatus.COMPLETED
    window._refresh_ui_after_project_change()
    fake_continue_dialog.accept = False  # the handler running is all this proves

    install_fake_menu(monkeypatch)
    menu = open_reset_menu(window._pristine_relaxation_view._table, row=0)

    assert menu is not None
    action = menu.action("Continue relaxation…")
    assert action is not None
    assert action.enabled is True

    action.triggered.emit()

    assert len(fake_continue_dialog.captured_kwargs) == 1
    window.close()


def test_continue_relaxation_muon_context_menu_action_invokes_the_real_handler(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    """Muon counterpart of the pristine context-menu wiring check above."""

    window, service, doc, definition, job, output_root, param_path = _muon_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.COMPLETED
    window._refresh_ui_after_project_change()
    fake_continue_dialog.accept = False

    install_fake_menu(monkeypatch)
    table = window._muon_relaxation_view._tables_by_charge_state["positive_muon"]
    menu = open_reset_menu(table, row=0)

    assert menu is not None
    action = menu.action("Continue relaxation…")
    assert action is not None
    assert action.enabled is True

    action.triggered.emit()

    assert len(fake_continue_dialog.captured_kwargs) == 1
    window.close()


def test_await_remote_file_check_reports_found(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit finding 4: drive the real nested-QEventLoop wait in
    ``_await_remote_file_check`` end to end (worker thread + signal), rather
    than only unit-testing ``ClusterController.check_remote_files`` in
    isolation (already covered by ``tests/app/test_cluster_controller.py``)."""

    window = main_window
    job_id = uuid4()
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_file_exists",
        lambda path: path == "/remote/x/seed.check",
    )

    result = window._await_remote_file_check([(job_id, "/remote/x/seed.check")])

    assert result == {"existing": ["/remote/x/seed.check"], "errors": {}}
    window.close()


def test_await_remote_file_check_reports_absent(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    job_id = uuid4()
    monkeypatch.setattr(window._cluster_runner._runner, "remote_file_exists", lambda _path: False)

    result = window._await_remote_file_check([(job_id, "/remote/x/seed.check")])

    assert result == {"existing": [], "errors": {}}
    window.close()


def test_await_remote_file_check_reports_worker_exception_as_unavailable(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    job_id = uuid4()

    def _boom(_path: str) -> bool:
        raise ConnectionError("stat failed")

    monkeypatch.setattr(window._cluster_runner._runner, "remote_file_exists", _boom)

    result = window._await_remote_file_check([(job_id, "/remote/x/seed.check")])

    assert result["existing"] == []
    assert result["errors"] == {job_id: "stat failed"}

    # And the probe helper built on top of it maps that into an unavailable
    # CheckpointProbeResult rather than raising or blocking the dialog.
    job = Job(
        label="probe_job",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
        execution_target="alpha",
        remote_directory="/remote/x",
    )
    job.status = JobStatus.FAILED
    monkeypatch.setattr(window, "_is_cluster_connected_for_job", lambda _job: True)
    monkeypatch.setattr(window, "_job_seed_name", lambda _job: "seed")
    monkeypatch.setattr(window._cluster_runner._runner, "remote_file_exists", _boom)

    probe = window._continue_relaxation_checkpoint_probe(job)

    from mufinder.services.cluster_runner import CheckpointProbeStatus

    assert probe.status == CheckpointProbeStatus.UNAVAILABLE
    assert "stat failed" in probe.reason
    window.close()


def test_main_window_reset_failed_jobs_uses_supplied_cleanup_choice_without_asking(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "reset_preconfirmed.mufinder", "ResetPreconfirmed")
    job = service.add_job(
        doc,
        label="j0",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.FAILED
    window._document = doc
    asked: list[int] = []
    monkeypatch.setattr(window, "_prompt_failed_job_cleanup", lambda count: asked.append(count))
    applied: list[tuple[list[Job], bool]] = []
    monkeypatch.setattr(
        window,
        "_apply_failed_job_reset",
        lambda jobs, *, cleanup_choice, detach_convergence: applied.append((list(jobs), cleanup_choice))
        or (0, []),
    )

    window._reset_failed_jobs([job], cleanup_choice=False)

    assert asked == []
    assert applied == [([job], False)]
    window.close()


def _write_previous_run_output(job_dir: Path, label: str) -> None:
    for name in (f"{label}.castep", f"{label}.check", f"{label}-out.cell", "slurm-55501.out"):
        (job_dir / name).write_text("previous run\n", encoding="utf-8")


def test_main_window_start_again_on_cluster_deletes_its_own_directory_without_prompting(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "start_again_cluster.mufinder", "StartAgainCluster")
    window._document = doc

    job = add_cluster_input_job(service, doc, "relax_a")
    job.execution_target = "alpha"
    job.remote_directory = "/remote/work/relax_a"
    job.remote_job_id = "55501"
    job.status = JobStatus.STOPPED
    job_dir = doc.project_root / "files" / "convergence_cutoff" / "relax_a"
    _write_previous_run_output(job_dir, "relax_a")

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    scheduler_checks: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "scheduler_job_is_active",
        lambda remote_job_id: scheduler_checks.append(remote_job_id) or False,
    )
    monkeypatch.setattr(
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("starting again was already confirmed"),
    )

    window._submit_jobs_on_cluster(
        [job],
        eligible_statuses={JobStatus.STOPPED},
        reuse_existing_remote_files=True,
        fresh_start_job_ids={job.id},
    )

    # The previous run is confirmed finished before its directory is deleted.
    assert scheduler_checks == ["55501"]
    assert recorded["checked"] == []
    assert recorded["deleted"] == ["/remote/work/relax_a"]
    assert len(recorded["submitted"]) == 1
    submitted = recorded["submitted"][0]
    assert submitted["remote_directory"] == "/remote/work/relax_a"
    assert submitted["reuse_existing_files"] is False
    assert sorted(path.name for path in job_dir.iterdir()) == ["relax_a.cell", "relax_a.param"]
    window.close()


def test_main_window_continue_on_cluster_keeps_the_previous_run(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "continue_cluster.mufinder", "ContinueCluster")
    window._document = doc

    job = add_cluster_input_job(service, doc, "relax_b")
    job.execution_target = "alpha"
    job.remote_directory = "/remote/work/relax_b"
    job.status = JobStatus.STOPPED
    job_dir = doc.project_root / "files" / "convergence_cutoff" / "relax_b"
    _write_previous_run_output(job_dir, "relax_b")

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)

    window._submit_jobs_on_cluster(
        [job],
        eligible_statuses={JobStatus.STOPPED},
        reuse_existing_remote_files=True,
    )

    assert recorded["deleted"] == []
    assert recorded["submitted"][0]["reuse_existing_files"] is True
    assert (job_dir / "relax_b.check").exists()
    window.close()


def test_main_window_start_again_locally_clears_the_previous_output(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "start_again_local.mufinder", "StartAgainLocal")
    window._document = doc

    job = add_cluster_input_job(service, doc, "relax_c")
    job.execution_target = "local"
    job.status = JobStatus.STOPPED
    job_dir = doc.project_root / "files" / "convergence_cutoff" / "relax_c"
    _write_previous_run_output(job_dir, "relax_c")

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/usr/bin/castep", mpirun_path=""),
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(num_cores=1, parallel=False, max_concurrent=1)

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)
    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(window._local_runner, "submit", lambda **kwargs: submit_calls.append(kwargs))

    window._submit_jobs_locally(
        [job],
        eligible_statuses={JobStatus.STOPPED},
        fresh_start_job_ids={job.id},
    )

    assert len(submit_calls) == 1
    assert sorted(path.name for path in job_dir.iterdir()) == ["relax_c.cell", "relax_c.param"]
    window.close()


def test_main_window_start_again_skips_a_job_whose_output_cannot_be_cleared(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "start_again_locked.mufinder", "StartAgainLocked")
    window._document = doc

    job = add_cluster_input_job(service, doc, "relax_d")
    job.execution_target = "alpha"
    job.remote_directory = "/remote/work/relax_d"
    job.status = JobStatus.STOPPED

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module, "show_detailed_warning", lambda _parent, _title, text, **_kw: warnings.append(text)
    )

    def _locked(_directory: Path) -> None:
        raise PermissionError("file in use")

    monkeypatch.setattr(MainWindow, "_clear_directory_contents", staticmethod(_locked))

    window._submit_jobs_on_cluster(
        [job],
        eligible_statuses={JobStatus.STOPPED},
        reuse_existing_remote_files=True,
        fresh_start_job_ids={job.id},
    )

    # Submitting would resume from a checkpoint the user asked to discard.
    assert recorded["submitted"] == []
    assert recorded["deleted"] == []
    assert job.status == JobStatus.STOPPED
    assert "relax_d: could not clear its previous output (file in use)" in warnings[0]
    window.close()
