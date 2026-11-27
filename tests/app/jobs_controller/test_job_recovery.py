"""Recovering jobs through the jobs controller: retry retrieval, dismiss, run again, inspect failure and continue relaxation."""

from __future__ import annotations

from mufinder.app.relaxation_types import MuonRelaxationRow
from mufinder.domain.charge_states import muon_relaxation_calc_type
from mufinder.services.job_monitoring import job_status_word
import copy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt

import mufinder.app.main_window as main_window_module
import mufinder.app.jobs_controller as jobs_controller_module
import mufinder.app.workflow_table_utils as workflow_table_utils
from mufinder.app.failed_job_dialog import FailedJobAction
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app import selection_snapshot as selection_snapshot_module
from mufinder.app.selection_snapshot import SelectionScope
from mufinder.app.workflow_submission import READY_WORKFLOW_ROW_STATUSES
from mufinder.app.relaxation_types import PristineRelaxationRow
from mufinder.app.main_window import MainWindow
from mufinder.app.workflow_config_codecs import (
    deserialize_muon_relaxation_config,
    deserialize_pristine_relaxation_config,
    serialize_muon_relaxation_config,
    serialize_pristine_relaxation_config,
)
from mufinder.castep_io.input_diff import JobInputTexts, diff_job_inputs
from mufinder.castep_setup import CastepSetup
from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import EndReason, JobActivity, JobStatus, WorkflowStage
from tests.support.dashboard import dismiss_button, look_button, open_context_menu, row_actions
from mufinder.domain.models import CalculationRun, Job, MuonSite, Structure
from mufinder.services.cluster_runner import CheckpointProbeResult, CheckpointProbeStatus
from mufinder.services.job_file_cleanup import JobFileCleanup
from mufinder.services.project_service import ProjectService
from tests.support.row_menu import install_fake_menu, open_row_menu
from tests.support.main_window import (
    add_cluster_input_job,
    capture_failed_job_dialogs,
    fake_local_submission,
    patch_continuation_cluster_dialog,
    patch_continuation_cluster_runner,
)


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

    window._jobs.inspect_failed_jobs([job])

    assert infos == ["No failed jobs"]
    window.close()


def _failed_job(
    service: ProjectService, doc, label: str, *, stage=WorkflowStage.CONVERGENCE_CUTOFF
):
    job = service.add_job(
        doc,
        label=label,
        workflow_stage=stage,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    return job


def test_inspect_shows_the_failed_job_dialog_with_the_log_excerpt(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_failed.mufinder", "InspectFailed")
    job = _failed_job(service, doc, "job_failed")
    job.error_message = "process failed"
    window._document = doc
    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.err").write_text("line 1\nfatal error\nline 3\n", encoding="utf-8")
    dialogs = capture_failed_job_dialogs(monkeypatch)

    window._jobs.inspect_failed_jobs([job])

    (dialog,) = dialogs
    (report,) = dialog._reports
    assert report.headline == "CASTEP stopped with an error."
    assert "Recorded message: process failed" in report.notes
    assert "[job_failed.err]" in report.log_excerpt
    assert "2: fatal error" in dialog.log_view.toPlainText()
    # Resubmit runs the job's workflow row, and this job has none.
    assert dialog.resubmit_button is None
    assert dialog.continue_button is None
    assert dialog.open_folder_button is not None
    window.close()


def test_inspect_resubmit_runs_the_dashboard_resubmit_path(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_resubmit.mufinder", "InspectResubmit")
    jobs = [_failed_job(service, doc, f"job_{index}") for index in range(3)]
    window._document = doc
    capture_failed_job_dialogs(monkeypatch, action=FailedJobAction.RESUBMIT)
    resubmitted: list[list[object]] = []
    monkeypatch.setattr(
        window._jobs, "_on_resubmit_from_dashboard", lambda job_ids: resubmitted.append(job_ids)
    )

    window._jobs.inspect_failed_jobs(jobs)

    assert resubmitted == [[job.id for job in jobs]]
    window.close()


def test_inspect_close_does_nothing_further(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "inspect_close.mufinder", "InspectClose")
    job = _failed_job(service, doc, "job_failed")
    window._document = doc
    capture_failed_job_dialogs(monkeypatch, action=None)
    monkeypatch.setattr(
        window._jobs,
        "_on_resubmit_from_dashboard",
        lambda _ids: pytest.fail("Close must not resubmit"),
    )

    window._jobs.inspect_failed_jobs([job])

    assert job_status_word(job) == "failed"
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
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    job.execution_target = cluster
    job.remote_job_id = "99123"
    job.remote_directory = "/remote/work/" + label
    return job


def test_dashboard_check_now_is_offered_only_where_output_may_still_be_on_the_cluster(
    main_window, tmp_path: Path
) -> None:
    """Check now is for unfetched cluster output, connected or not; the refresh covers the rest."""

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "retry.mufinder", "Retry")

    connected_job = _make_failed_cluster_job(service, doc, label="conn", cluster="scarf")
    connected_job.end_reason = EndReason.OUTPUT_UNAVAILABLE
    disconnected_job = _make_failed_cluster_job(service, doc, label="disc", cluster="other")
    disconnected_job.end_reason = EndReason.INTERRUPTED
    settled_job = _make_failed_cluster_job(service, doc, label="settled", cluster="scarf")
    no_meta_job = _make_failed_cluster_job(service, doc, label="nometa", cluster="scarf")
    no_meta_job.end_reason = EndReason.OUTPUT_UNAVAILABLE
    no_meta_job.remote_directory = None
    active_job = _make_failed_cluster_job(service, doc, label="active", cluster="scarf")
    active_job.status = JobStatus.ACTIVE
    active_job.activity = JobActivity.RUNNING
    local_job = service.add_job(
        doc,
        label="localjob",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    local_job.status = JobStatus.ENDED
    local_job.end_reason = EndReason.UNKNOWN
    local_job.execution_target = "local"

    window._document = doc
    _connect_cluster_runner(window, "scarf")
    window._job_execution._job_output_directory = lambda _job: tmp_path
    window._refresh_jobs_table()

    def offers_check(job) -> bool:
        return window._job_dashboard.action_applies("check_now", job)

    assert offers_check(connected_job) is True
    # Its cluster is not connected: pressing it connects first.
    assert offers_check(disconnected_job) is True
    # Output already read to a verdict, an active job, nowhere to look, a local job.
    assert offers_check(settled_job) is False
    assert offers_check(active_job) is False
    assert offers_check(no_meta_job) is False
    assert offers_check(local_job) is False
    window.close()


def test_dashboard_check_now_asks_the_cluster_and_leaves_the_job_as_it_is(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "retry_click.mufinder", "RetryClick")
    job = _make_failed_cluster_job(service, doc, label="conn", cluster="scarf")
    job.end_reason = EndReason.OUTPUT_UNAVAILABLE

    window._document = doc
    _connect_cluster_runner(window, "scarf")
    monkeypatch.setattr(window._job_execution, "_job_output_directory", lambda _job: tmp_path)

    checks: list[tuple[list, bool]] = []
    monkeypatch.setattr(
        window._cluster_runner,
        "check_jobs",
        lambda targets, *, skip_if_busy=False: checks.append((list(targets), skip_if_busy)),
    )
    monkeypatch.setattr(window, "_request_dashboard_flush", lambda **kwargs: None)

    window._refresh_jobs_table()
    look_button(window).click()

    # The job changes only when the evidence comes back.
    assert job_status_word(job) == "failed"
    ((targets, skip_if_busy),) = checks
    assert skip_if_busy is False
    assert [(t.job_id, t.remote_job_id, t.on_request) for t in targets] == [(job.id, "99123", True)]
    assert window.statusBar().currentMessage() == "Checking 1 cluster job"
    window.close()


def test_dashboard_check_now_completes_a_local_job_whose_output_is_complete(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "check_local.mufinder", "CheckLocal")
    job = service.add_job(
        doc,
        label="localjob",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.execution_target = "local"
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.WALLTIME
    fixture = Path(__file__).parents[2] / "fixtures" / "uep" / "Si2.castep"
    (tmp_path / "localjob.castep").write_bytes(fixture.read_bytes())
    window._document = doc
    monkeypatch.setattr(window._job_execution, "_job_output_directory", lambda _job: tmp_path)
    monkeypatch.setattr(window._job_execution, "_job_seed_name", lambda job: job.label)
    monkeypatch.setattr(window, "_job_output_directory", lambda _job: tmp_path)
    monkeypatch.setattr(window, "_job_seed_name", lambda job: job.label)

    window._jobs._on_check_now_from_dashboard([job.id])

    assert job.status == JobStatus.COMPLETED
    assert len(doc.results) == 1

    job.status = JobStatus.COMPLETED
    shown: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _text: shown.append(title),
    )
    window._jobs._on_check_now_from_dashboard([job.id])
    assert shown == ["No jobs to check"]
    window.close()


def test_dashboard_dismiss_hides_row_after_flush(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismiss.mufinder", "Dismiss")
    failed_job = service.add_job(
        doc,
        label="failed",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    failed_job.status = JobStatus.ENDED
    failed_job.end_reason = EndReason.UNKNOWN
    running_job = service.add_job(
        doc,
        label="running",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    running_job.status = JobStatus.ACTIVE
    running_job.activity = JobActivity.RUNNING

    window._document = doc
    window._refresh_jobs_table()
    assert window._job_dashboard.table.rowCount() == 2

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_args, **_kwargs: True)
    assert dismiss_button(window, 1).isHidden()

    dismiss_button(window, 0).click()
    # Dismissal schedules a coalesced flush; force it to rebuild the table now.
    window._flush_dashboard()

    # After dismissal + rebuild, the failed row disappears; running row remains.
    assert failed_job.dismissed_at is not None
    assert window._job_dashboard.table.rowCount() == 1
    remaining = window._job_dashboard.table.item(0, window._job_dashboard.LABEL_COLUMN)
    assert remaining.data(Qt.ItemDataRole.UserRole) == str(running_job.id)
    # Provenance milestone recorded.
    assert doc.provenance[-1].action == "job_dismissed"
    window.close()


def test_dashboard_dismiss_multi_select(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "dismiss_multi.mufinder", "DismissMulti")
    job0 = service.add_job(
        doc,
        label="d0",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job1 = service.add_job(
        doc,
        label="d1",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job0.status = JobStatus.ENDED
    job0.end_reason = EndReason.UNKNOWN
    job1.status = JobStatus.ENDED
    job1.end_reason = EndReason.WALLTIME

    window._document = doc
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )
    window._refresh_jobs_table()

    dismiss_button(window).selection_snapshot_rows = (0, 1)
    dismiss_button(window).click()
    window._flush_dashboard()

    assert job0.dismissed_at is not None
    assert job1.dismissed_at is not None
    assert window._job_dashboard.table.rowCount() == 0
    window.close()


# --------------------------------------------------------------------------- #
# Dismissed failed jobs stay resubmittable from their workflow row
# --------------------------------------------------------------------------- #
#
# Regression for the stranded-row bug: dismissing a failed job from the jobs
# dashboard hides it, but its workflow row kept projecting "failed" and stayed
# unsubmittable forever. The row's Run returns the *same* job to defined before
# dispatch.
def _fail_and_dismiss(window: MainWindow, service: ProjectService, job: Job) -> None:
    service.submit_job(job)
    service.end_job(job, EndReason.UNKNOWN, "castep exploded")
    window._jobs._on_dismiss_from_dashboard([job.id])
    assert job.dismissed_at is not None
    assert job_status_word(job) == "failed"


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

    row = PristineRelaxationRow(label="row_1", parameter_values={})
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialize_pristine_relaxation_config(
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
    service.end_job(job, EndReason.UNKNOWN, "castep exploded")
    window._refresh_ui_after_project_change()
    status_column = window._pristine_relaxation_view._STATUS_COLUMN
    assert window._pristine_relaxation_view._table.item(0, status_column).text() == "failed"

    # Dismissing only hides the dashboard row: the workflow row still says what happened.
    window._jobs._on_dismiss_from_dashboard([job.id])
    window._refresh_ui_after_project_change()

    assert window._pristine_relaxation_view._table.item(0, status_column).text() == "failed"

    submitted = fake_local_submission(window, monkeypatch)

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
            "sweep_values": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )
    window._document = doc
    window._refresh_ui_after_project_change()

    service.submit_job(job)
    service.end_job(job, EndReason.UNKNOWN, "castep exploded")
    window._refresh_ui_after_project_change()
    status_column = window._convergence_view._STATUS_COLUMN
    assert window._convergence_view._table.item(0, status_column).text() == "failed"

    window._jobs._on_dismiss_from_dashboard([job.id])
    window._refresh_ui_after_project_change()

    assert window._convergence_view._table.item(0, status_column).text() == "failed"
    # The cutoff panel's Submit actions are gated on the projected row status,
    # which accepts a row whose job ended; its inputs can change before it runs
    # again, since the run rebuilds them.
    assert window._convergence_view._row_status(0) in READY_WORKFLOW_ROW_STATUSES
    assert not workflow_table_utils.workflow_row_is_locked(window._convergence_view._row_status(0))

    submitted = fake_local_submission(window, monkeypatch)

    window._convergence_controller.submit([300.0], target="local")

    assert [[j.id for j in jobs] for jobs in submitted] == [[job.id]]
    assert job.status is JobStatus.DEFINED
    assert job.dismissed_at is None
    assert len(doc.jobs) == 1
    # Running the row again must NOT detach the sweep point from its job, or the
    # row would go back to "ready" with no job and a second submit would mint a
    # duplicate.
    assert definition.parameters["job_ids_by_cutoff"] == {
        main_window_module.MainWindow._cutoff_key(300.0): str(job.id)
    }
    window.close()


def test_failed_row_without_a_checkpoint_starts_again_without_a_question(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answered_choices
) -> None:
    """A failed sweep row runs again from its row with no question: nothing could continue.

    The row reads ``failed``; Run cleans the job's folder (its inputs are kept,
    then rebuilt), resets the same job and submits it.
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
            "sweep_values": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )
    window._document = doc
    service.submit_job(job)
    service.end_job(job, EndReason.UNKNOWN, "castep exploded")
    window._refresh_ui_after_project_change()
    job_dir = window._job_output_directory(job)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{job.label}.castep").write_text("old run\n", encoding="utf-8")
    # A sweep's inputs never resume, so even a checkpoint is not asked about.
    (job_dir / f"{job.label}.check").write_bytes(b"checkpoint")

    assert (
        window._convergence_view._table.item(0, window._convergence_view._STATUS_COLUMN).text()
        == "failed"
    )
    submitted = fake_local_submission(window, monkeypatch)

    window._convergence_controller.submit([300.0], target="local")

    assert answered_choices == []
    assert [[j.id for j in jobs] for jobs in submitted] == [[job.id]]
    assert job.status is JobStatus.DEFINED
    assert len(doc.jobs) == 1
    assert not (job_dir / f"{job.label}.castep").exists()
    assert not (job_dir / f"{job.label}.check").exists()
    assert (job_dir / f"{job.label}.param").is_file()
    assert [record.action for record in doc.provenance][-1] == "job_reset"
    window.close()


def test_dismissed_row_submit_regenerates_job_inputs(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resubmitted job's inputs are rebuilt from the current configuration.

    ``materialize`` only rewrites a job directory for a DEFINED job, and it runs
    while the plan is being built — before the job returns to defined. Without a
    second materialize pass the row would resubmit against the stale ``.cell``
    the failed run left behind (``_clear_directory_contents`` deliberately
    preserves ``.cell``/``.param``).
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
            rows=(PristineRelaxationRow(label="row_1", parameter_values={}),),
        )
    )
    job = window._pristine_relaxation_controller.materialize(["row_1"])[0]
    cell_path = doc.project_root / "files" / "pristine_relaxation" / job.label / f"{job.label}.cell"
    assert "KPOINT_MP_GRID" not in cell_path.read_text(encoding="utf-8")

    _fail_and_dismiss(window, service, job)

    # The configuration changes while the row is stranded.
    doc.project.recommended_kpoint_grid = (4, 4, 4)
    doc.project.recommended_kpoint_lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    submitted = fake_local_submission(window, monkeypatch)

    window._pristine_relaxation_controller.submit(["row_1"], target="local")

    assert [[j.id for j in jobs] for jobs in submitted] == [[job.id]]
    assert "KPOINT_MP_GRID 4 4 4" in cell_path.read_text(encoding="utf-8")
    assert len(doc.jobs) == 1
    window.close()


# ---------------------------------------------------------------------------
# Row-menu Cancel / Inspect Failure routing
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
    ("cancel_requested", "cancel_jobs"),
    ("inspect_failed_requested", "inspect_failed_jobs"),
]


@pytest.mark.parametrize(
    ("view_attr", "controller_attr", "keys", "signal_name", "handler_name"),
    [(*view, *signal) for view in _ROW_ACTION_VIEWS for signal in _ROW_ACTION_SIGNALS],
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
    # The signal must actually reach MainWindow. The energy-barriers view is created lazily, so its wiring happens in
    # _ensure_energy_barriers_view rather than __init__.
    window = main_window
    if view_attr == "_energy_barriers_view":
        window._ensure_energy_barriers_view()
    job = Job(
        label="row-job",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_relaxation",
        status=JobStatus.ENDED,
        end_reason=EndReason.UNKNOWN,
    )

    requested_keys: list[list[object]] = []
    handled: list[list[Job]] = []
    monkeypatch.setattr(
        getattr(window, controller_attr),
        "jobs_for_keys",
        lambda values: (requested_keys.append(list(values)), [job])[1],
    )
    monkeypatch.setattr(window._jobs, handler_name, lambda jobs: handled.append(jobs))

    getattr(getattr(window, view_attr), signal_name).emit(keys)

    assert requested_keys == [keys]
    assert handled == [[job]]
    window.close()


@pytest.mark.parametrize("view_attr", [view for view, _controller, _keys in _ROW_ACTION_VIEWS])
def test_no_workflow_view_offers_a_reset_signal(main_window, view_attr: str) -> None:
    window = main_window
    if view_attr == "_energy_barriers_view":
        window._ensure_energy_barriers_view()

    assert not hasattr(getattr(window, view_attr), "reset_failed_requested")
    assert not hasattr(window._jobs, "reset_failed_jobs")
    window.close()


def test_running_jobs_again_refreshes_the_lazily_created_workflow_views(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Energy barriers and quantum corrections are not repainted by
    # _refresh_jobs_table, so a row run again from either table would leave a
    # stale "failed" status behind without these two refreshes.
    window = main_window
    refreshed: list[str] = []
    monkeypatch.setattr(
        window, "_refresh_energy_barriers_view", lambda: refreshed.append("energy_barriers")
    )
    monkeypatch.setattr(
        window, "_refresh_quantum_corrections_view", lambda: refreshed.append("quantum")
    )

    window._refresh_views_after_job_reset()

    assert refreshed == ["energy_barriers", "quantum"]
    window.close()


def test_running_a_failed_row_again_repaints_its_status_for_the_same_job(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running an ended row again must repaint the row, not just mutate the Job object.

    The row keeps its label->job-id mapping pointing at the same job, so the row
    only changes because the job is now ``defined`` and ``effective_row_status``
    projects a submittable status. This drives the real refresh path (no stubbed
    refresh) and asserts the Status *cell*, which is what the user sees.
    """

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "row_rerun_repaint.mufinder", "RowRerunRepaint")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    window._on_pristine_relaxation_configured(
        main_window_module.PristineRelaxationConfig(
            name_prefix="pristine_relax",
            rows=(PristineRelaxationRow(label="row_1", parameter_values={}),),
        )
    )
    job = window._pristine_relaxation_controller.materialize(["row_1"])[0]
    service.submit_job(job)
    service.end_job(job, EndReason.UNKNOWN, "castep exploded")
    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [{"label": "row_1", "fix_all_cell": False, "params": {}, "added_advanced": []}],
    }
    window._refresh_ui_after_project_change()

    view = window._pristine_relaxation_view
    assert view._row_status(0) == "failed"
    submitted = fake_local_submission(window, monkeypatch)

    window._pristine_relaxation_controller.submit(["row_1"], target="local")

    assert [[j.id for j in jobs] for jobs in submitted] == [[job.id]]
    assert job.status == JobStatus.DEFINED
    # The row stays bound to the same job - no duplicate is minted - and now
    # reads as submittable again.
    assert len(doc.jobs) == 1
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

    def continuation_inputs(self) -> JobInputTexts | None:
        # The canned result, rendered as the row would render it.
        values, advanced = type(self).result_to_return
        return self.kwargs["render_inputs"](values, advanced)

    def deleteLater(self) -> None:  # noqa: N802 - Qt API naming.
        return None


@pytest.fixture()
def fake_continue_dialog(monkeypatch: pytest.MonkeyPatch):
    _FakeContinueRelaxationDialog.captured_kwargs = []
    _FakeContinueRelaxationDialog.accept = True
    _FakeContinueRelaxationDialog.result_to_return = ({}, ())
    monkeypatch.setattr(
        jobs_controller_module, "ContinueRelaxationDialog", _FakeContinueRelaxationDialog
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
        jobs_controller_module, "ClusterSubmissionDialog", _FakeClusterSubmissionDialogForContinue
    )
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )
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

    row = PristineRelaxationRow(label="row_1", parameter_values={"geom_max_iter": 120})
    output_root = doc.project_root / "files" / "pristine_relaxation"
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialize_pristine_relaxation_config(
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
            {
                "label": "row_1",
                "fix_all_cell": False,
                "params": {"geom_max_iter": 120},
                "added_advanced": [],
            }
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
    site = MuonSite(label="Site 1", initial_fractional_position=[0.1, 0.2, 0.3], status="staged")
    doc.sites[site.id] = site
    window._document = doc

    row = MuonRelaxationRow(
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
        parameters=serialize_muon_relaxation_config(
            main_window_module.MuonRelaxationConfig(rows=(row,))
        ),
        output_directory=output_root,
    )
    job = service.add_job(
        doc,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=muon_relaxation_calc_type("positive_muon"),
    )
    definition.parameters["job_ids_by_label"] = {"positivemuon:Site 1": str(job.id)}
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["muon_relaxation_form"] = serialize_muon_relaxation_config(
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
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    job.remote_directory = "/remote/work/pristine_relax_row_1"
    job.remote_job_id = "old-remote-id"
    job.error_message = "previous castep failure"
    job.dismissed_at = datetime.now(UTC)
    window._refresh_ui_after_project_change()
    # Dismissal does not change what the row says.
    assert window._pristine_relaxation_view._row_status(0) == "failed"

    fake_continue_dialog.result_to_return = ({"geom_max_iter": 500}, ())
    submit_calls = _wire_fake_cluster_submission(window, monkeypatch, cluster_name="alpha")
    monkeypatch.setattr(
        window._jobs,
        "_ask_continue_or_start_again",
        lambda *_args: pytest.fail("Continue must never ask the run-again question"),
    )

    window._jobs.on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    # Row params mutated in place.
    config = deserialize_pristine_relaxation_config(definition.parameters["configuration"])
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
    # dismissed_at/error/old remote id cleared by prepare_job_continuation, and
    # the job is back in an active state after a successful cluster submit.
    assert job_status_word(job) == "queued"
    assert job.dismissed_at is None
    assert job.error_message is None
    assert job.remote_job_id == "new-remote-job-id"
    assert job.remote_directory == "/remote/work/pristine_relax_row_1"
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
    # A cluster job runs in a remote folder named after its seed.
    job.remote_directory = f"/remote/work/{param_path.parent.name}"
    window._refresh_ui_after_project_change()

    fake_continue_dialog.result_to_return = ({"geom_max_iter": 777}, ())
    submit_calls = _wire_fake_cluster_submission(window, monkeypatch, cluster_name="alpha")

    window._jobs.on_continue_relaxation_requested_from_muon_relaxation("positivemuon:Site 1")

    config = deserialize_muon_relaxation_config(definition.parameters["configuration"])
    assert {row.charge_state: row.parameter_values for row in config.rows} == {
        "positive_muon": {"geom_max_iter": 777}
    }
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 777" in rendered
    assert list(doc.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {"positivemuon:Site 1": str(job.id)}
    assert len(submit_calls) == 1
    assert submit_calls[0]["remote_directory"] == job.remote_directory
    assert submit_calls[0]["allow_existing_remote_directory"] is True
    assert job_status_word(job) == "queued"
    assert any(record.action == "job_continuation" for record in doc.provenance)
    window.close()


def test_continue_relaxation_local_job_round_trip(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    job.error_message = "boom"
    assert job.execution_target == "local"

    # A local job's checkpoint probe is a plain Path check: write a real
    # ``.check`` file so the probe reports "found".
    checkpoint_path = param_path.parent / f"{job.label}.check"
    checkpoint_path.write_text("", encoding="utf-8")

    fake_continue_dialog.result_to_return = ({"geom_max_iter": 42}, ())
    local_submitted: list[list[Job]] = []
    monkeypatch.setattr(
        window._jobs,
        "submit_jobs_locally",
        lambda jobs, **_kwargs: local_submitted.append(list(jobs)),
    )
    monkeypatch.setattr(
        window._jobs,
        "submit_jobs_on_cluster",
        lambda *_a, **_k: pytest.fail("A local job must submit through the local path"),
    )

    window._jobs.on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    probe = fake_continue_dialog.captured_kwargs[-1]["checkpoint_probe"]
    assert probe.status == jobs_controller_module.CheckpointProbeResult.found().status
    assert [[j.id for j in jobs] for jobs in local_submitted] == [[job.id]]
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 42" in rendered
    window.close()


def test_continue_relaxation_from_the_failed_job_dialog_continues_the_row(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, _service, _doc, _definition, job, _output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    job.error_message = "CASTEP exited with status 1: SCF cycles did not converge"
    dialogs = capture_failed_job_dialogs(monkeypatch, action=FailedJobAction.CONTINUE_RELAXATION)
    fake_continue_dialog.result_to_return = ({"geom_max_iter": 300}, ())
    local_submitted: list[list[Job]] = []
    monkeypatch.setattr(
        window._jobs,
        "submit_jobs_locally",
        lambda jobs, **_kwargs: local_submitted.append(list(jobs)),
    )

    window._jobs.inspect_failed_jobs([job])

    (dialog,) = dialogs
    assert dialog.continue_button is not None
    assert dialog.resubmit_button is not None
    assert dialog.open_folder_button is not None
    assert [[j.id for j in jobs] for jobs in local_submitted] == [[job.id]]
    assert "GEOM_MAX_ITER : 300" in param_path.read_text(encoding="utf-8")
    window.close()


def test_failed_job_dialog_continues_a_muon_relaxation_by_its_charge_qualified_row(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _service, _doc, _definition, job, _output_root, _param_path = _muon_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    window._refresh_ui_after_project_change()
    continued: list[object] = []
    monkeypatch.setattr(
        window._jobs, "on_continue_relaxation_requested_from_muon_relaxation", continued.append
    )
    dialogs = capture_failed_job_dialogs(monkeypatch, action=FailedJobAction.CONTINUE_RELAXATION)

    window._jobs.inspect_failed_jobs([job])

    assert dialogs[0].continue_button is not None
    assert continued == ["positivemuon:Site 1"]
    window.close()


def test_dashboard_opens_a_muon_relaxation_on_its_charge_state_row(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _service, _doc, _definition, job, _output_root, _param_path = _muon_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    window._refresh_ui_after_project_change()

    # Continue relaxation stays on the relaxation table and in the failed-job dialog.
    assert row_actions(window) == ("Resubmit…", "Inspect…", "✕")
    menu = open_context_menu(window, monkeypatch)
    assert [action.text for action in menu.actions if action.enabled] == [
        "Inspect…",
        "Resubmit…",
        "Dismiss",
        "Open in Relaxations",
    ]
    menu.action("Open in Relaxations").triggered.emit()

    assert window._project_navigator.active_route() == "muon_relaxations"
    assert window._muon_relaxation_view._selected_keys() == ["positivemuon:Site 1"]
    window.close()


def test_dashboard_offers_no_continue_relaxation(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continue relaxation is on the relaxation tables and in the failed-job dialog only."""

    window, _service, _doc, _definition, job, _output_root, _param_path = (
        _pristine_continue_fixture(tmp_path)
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.WALLTIME
    window._refresh_ui_after_project_change()

    assert row_actions(window) == ("Resubmit…", "Inspect…", "✕")
    texts = [action.text for action in open_context_menu(window, monkeypatch).actions]
    assert "Resubmit…" in texts
    assert not any(text.startswith("Continue relaxation") for text in texts)
    # The failed-job dialog still offers it for the row.
    dialogs = capture_failed_job_dialogs(monkeypatch)
    window._jobs.inspect_failed_jobs([job])
    assert dialogs[0].continue_button is not None
    assert dialogs[0].resubmit_button is not None
    window.close()


def test_failed_job_dialog_offers_no_continue_when_the_row_cannot_be_continued(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _service, _doc, _definition, job, _output_root, _param_path = (
        _pristine_continue_fixture(tmp_path)
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.WALLTIME
    monkeypatch.setattr(
        window._pristine_relaxation_controller, "row_is_continuable", lambda _key: False
    )
    dialogs = capture_failed_job_dialogs(monkeypatch)

    window._jobs.inspect_failed_jobs([job])

    assert dialogs[0].continue_button is None
    window.close()


def test_continue_relaxation_cluster_not_connected_dialog_still_opens_with_unavailable_probe(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.execution_target = "alpha"
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    job.remote_directory = "/remote/work/pristine_relax_row_1"
    assert window._cluster_runner.is_connected is False

    monkeypatch.setattr(
        window._cluster_runner,
        "check_remote_files",
        lambda *_a, **_k: pytest.fail(
            "Not connected: the checkpoint probe must not hit the network"
        ),
    )
    fake_continue_dialog.accept = False  # cancel: no submission wiring needed

    window._jobs.on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    assert len(fake_continue_dialog.captured_kwargs) == 1
    probe = fake_continue_dialog.captured_kwargs[0]["checkpoint_probe"]
    assert probe.status == jobs_controller_module.CheckpointProbeResult.unavailable("x").status
    assert "alpha" in probe.reason
    window.close()


def test_continue_relaxation_cancel_leaves_everything_unchanged(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_continue_dialog
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    job.error_message = "boom"
    before_config = copy.deepcopy(definition.parameters["configuration"])

    fake_continue_dialog.accept = False
    monkeypatch.setattr(
        window._jobs, "submit_jobs_locally", lambda *_a, **_k: pytest.fail("Cancel must not submit")
    )
    monkeypatch.setattr(
        window._jobs,
        "submit_jobs_on_cluster",
        lambda *_a, **_k: pytest.fail("Cancel must not submit"),
    )

    window._jobs.on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    assert definition.parameters["configuration"] == before_config
    assert job_status_word(job) == "failed"
    assert job.error_message == "boom"
    assert not any(record.action == "job_continuation" for record in doc.provenance)
    window.close()


@pytest.mark.parametrize("accept", [True, False], ids=["continued", "cancelled"])
def test_continue_relaxation_deletes_its_dialogs_after_closing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accept: bool
) -> None:
    """The real Continue dialog does not outlive the open."""

    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    code = main_window_module.QDialog.DialogCode
    opened: list[object] = []

    def fake_exec(dialog: object) -> object:
        opened.append(dialog)
        return code.Accepted if accept else code.Rejected

    monkeypatch.setattr(jobs_controller_module.ContinueRelaxationDialog, "exec", fake_exec)
    monkeypatch.setattr(window._jobs, "submit_jobs_locally", lambda *_a, **_k: None)
    monkeypatch.setattr(window._jobs, "submit_jobs_on_cluster", lambda *_a, **_k: None)

    for _ in range(3):
        if accept:
            job.status = JobStatus.ENDED
            job.end_reason = EndReason.UNKNOWN
        window._jobs.on_continue_relaxation_requested_from_pristine_relaxation("row_1")
    assert len(opened) == 3
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert window.findChildren(ParameterDialog) == []
    window.close()


def test_continue_relaxation_gating_disabled_running_enabled_not_converged(
    qapp, tmp_path: Path
) -> None:
    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )

    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    assert window._pristine_relaxation_controller.row_is_continuable("row_1") is False

    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.QUEUED
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


_PRISTINE_KEY = "row_1"
_MUON_KEY = "positivemuon:Site 1"


def _continue_fixture(tmp_path: Path, workflow: str):
    """Return ``(window, job, param_path, open_dialog)`` for one continuable row."""

    if workflow == "pristine":
        window, _service, _doc, _definition, job, _root, param_path = _pristine_continue_fixture(
            tmp_path
        )

        def open_dialog() -> None:
            window._jobs.on_continue_relaxation_requested_from_pristine_relaxation(_PRISTINE_KEY)

    else:
        window, _service, _doc, _definition, job, _root, param_path = _muon_continue_fixture(
            tmp_path
        )

        def open_dialog() -> None:
            window._jobs.on_continue_relaxation_requested_from_muon_relaxation(_MUON_KEY)

    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    return window, job, param_path, open_dialog


def _run_real_continue_dialog(
    monkeypatch: pytest.MonkeyPatch,
    window: MainWindow,
    open_dialog,
    *,
    edit=lambda dialog: None,
    accept: bool = False,
) -> list[tuple[str, ...]]:
    """Open the real Continue dialog through the jobs controller, apply ``edit``,
    and return the ``Parameter changes`` table's rows; then accept (with
    submission stubbed out) or cancel."""

    captured: list[tuple[str, ...]] = []

    def exec_dialog(dialog) -> int:
        edit(dialog)
        table = dialog._diff_table
        if not table.isHidden():
            captured.extend(
                tuple(table.item(row, column).text() for column in range(table.columnCount()))
                for row in range(table.rowCount())
            )
        if accept:
            return main_window_module.QDialog.DialogCode.Accepted
        return main_window_module.QDialog.DialogCode.Rejected

    monkeypatch.setattr(jobs_controller_module.ContinueRelaxationDialog, "exec", exec_dialog)
    monkeypatch.setattr(window._jobs, "submit_jobs_locally", lambda *_args, **_kwargs: None)
    open_dialog()
    return captured


def _read_inputs(param_path: Path) -> JobInputTexts:
    return JobInputTexts(
        cell=param_path.with_suffix(".cell").read_text(encoding="utf-8"),
        param=param_path.read_text(encoding="utf-8"),
    )


def _uncheck_generate_symmetry(dialog) -> None:
    row = dialog._rows.get("symmetry_generate")
    if row is not None:
        row.editor.set_value(False)


def _set_grid(dialog) -> None:
    dialog._rows["kpoint_mp_grid"].editor.set_value((3, 3, 3))


def _raise_max_iterations(dialog) -> None:
    dialog._rows["geom_max_iter"].editor.set_value(250)


@pytest.mark.parametrize("workflow", ["pristine", "muon"])
@pytest.mark.parametrize(
    "edit",
    [lambda dialog: None, _uncheck_generate_symmetry, _set_grid, _raise_max_iterations],
    ids=["unedited", "generate-symmetry", "grid", "max-iterations"],
)
def test_continue_relaxation_table_lists_exactly_what_accepting_writes(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, edit
) -> None:
    """The ``Parameter changes`` table compares the job's ``.cell`` and
    ``.param`` on disk with what accepting writes, so its rows are exactly the
    keyword changes between the files before and after."""

    window, job, param_path, open_dialog = _continue_fixture(tmp_path, workflow)
    before = _read_inputs(param_path)

    rows = _run_real_continue_dialog(monkeypatch, window, open_dialog, edit=edit, accept=True)

    written = diff_job_inputs(before, _read_inputs(param_path))
    assert [row[:3] for row in rows] == [
        ({"cell": ".cell", "param": ".param"}[change.file], change.keyword, change.change)
        for change in written
    ]
    window.close()


def test_continue_relaxation_unedited_dialog_after_a_continuation_shows_no_changes(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, job, param_path, open_dialog = _continue_fixture(tmp_path, "pristine")
    _run_real_continue_dialog(
        monkeypatch, window, open_dialog, edit=_raise_max_iterations, accept=True
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN

    rows = _run_real_continue_dialog(monkeypatch, window, open_dialog)

    assert rows == []
    window.close()


def test_continue_relaxation_lists_a_removed_cell_keyword(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchecking ``Generate Symmetry`` removes ``SYMMETRY_GENERATE`` from the
    rewritten ``.cell``; the table says so before the job restarts."""

    window, job, param_path, open_dialog = _continue_fixture(tmp_path, "pristine")
    assert "SYMMETRY_GENERATE" in param_path.with_suffix(".cell").read_text(encoding="utf-8")

    rows = _run_real_continue_dialog(
        monkeypatch, window, open_dialog, edit=_uncheck_generate_symmetry
    )

    assert (".cell", "SYMMETRY_GENERATE", "Removed", "") in rows
    window.close()


def test_continue_relaxation_muon_lists_an_added_cell_keyword(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, job, param_path, open_dialog = _continue_fixture(tmp_path, "muon")
    assert "KPOINT" not in param_path.with_suffix(".cell").read_text(encoding="utf-8")

    rows = _run_real_continue_dialog(monkeypatch, window, open_dialog, edit=_set_grid)

    assert (".cell", "KPOINT_MP_GRID", "Added", "3 3 3") in rows
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
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    fake_continue_dialog.result_to_return = ({"geom_max_iter": 500}, ())

    # Force the continuation's own materialize call to fail, exactly like the
    # real one does on a caught ValueError/StructureBridgeError.
    window._pristine_relaxation_controller.materialize = lambda keys: []

    monkeypatch.setattr(
        window._jobs, "submit_jobs_locally", lambda *_a, **_k: pytest.fail("must not submit")
    )
    monkeypatch.setattr(
        window._jobs, "submit_jobs_on_cluster", lambda *_a, **_k: pytest.fail("must not submit")
    )
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, text: warnings.append((title, text)),
    )

    window._jobs.on_continue_relaxation_requested_from_pristine_relaxation("row_1")

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
    branch in ``JobsController.submit_jobs_on_cluster`` must not fire in that case -- a
    continued job's ``remote_directory`` string means nothing on a different
    cluster's filesystem. This asserts the NORMAL path is taken instead: the
    remote-directory existence check runs (it is skipped entirely for a
    same-cluster continuation), and nothing is deleted."""

    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.execution_target = "alpha"
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
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
        jobs_controller_module,
        "ClusterSubmissionDialog",
        _FakeClusterSubmissionDialogSelectsCluster,
    )
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [alpha, beta])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_configs", lambda _settings: [alpha, beta]
    )
    monkeypatch.setattr(
        jobs_controller_module,
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

    window._jobs.on_continue_relaxation_requested_from_pristine_relaxation("row_1")

    # The continuation branch was NOT taken: the existence check ran, and the
    # job was submitted with allow_existing_remote_directory=False -- the
    # ordinary cold-target path, not a blind directory reuse onto a cluster
    # that never saw this job.
    assert len(checked_paths) == 1
    assert len(submit_calls) == 1
    assert submit_calls[0]["allow_existing_remote_directory"] is False
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
    job.status = JobStatus.COMPLETED
    window._refresh_ui_after_project_change()
    fake_continue_dialog.accept = False  # the handler running is all this proves

    install_fake_menu(monkeypatch)
    menu = open_row_menu(window._pristine_relaxation_view._table, row=0)

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
    menu = open_row_menu(table, row=0)

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

    result = window._jobs._await_remote_file_check([(job_id, "/remote/x/seed.check")])

    assert result == {"existing": ["/remote/x/seed.check"], "errors": {}}
    window.close()


def test_await_remote_file_check_reports_absent(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    job_id = uuid4()
    monkeypatch.setattr(window._cluster_runner._runner, "remote_file_exists", lambda _path: False)

    result = window._jobs._await_remote_file_check([(job_id, "/remote/x/seed.check")])

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

    result = window._jobs._await_remote_file_check([(job_id, "/remote/x/seed.check")])

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
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    monkeypatch.setattr(window._jobs, "_is_cluster_connected_for_job", lambda _job: True)
    monkeypatch.setattr(window, "_job_seed_name", lambda _job: "seed")
    monkeypatch.setattr(window._cluster_runner._runner, "remote_file_exists", _boom)

    probe = window._jobs._continue_relaxation_checkpoint_probe(job)

    from mufinder.services.cluster_runner import CheckpointProbeStatus

    assert probe.status == CheckpointProbeStatus.UNAVAILABLE
    assert "stat failed" in probe.reason
    window.close()


def _write_previous_run_output(job_dir: Path, label: str) -> None:
    for name in (f"{label}.castep", f"{label}.check", f"{label}-out.cell", "slurm-55501.out"):
        (job_dir / name).write_text("previous run\n", encoding="utf-8")


@pytest.mark.parametrize("accounting_state", ["FAILED", "RUNNING", "ERROR"])
def test_main_window_start_again_on_cluster_deletes_its_own_directory_without_prompting(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accounting_state
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "start_again_cluster.mufinder", "StartAgainCluster")
    window._document = doc

    job = add_cluster_input_job(service, doc, "relax_a")
    job.execution_target = "alpha"
    job.remote_directory = "/remote/work/relax_a"
    job.remote_job_id = "55501"
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.WALLTIME
    job_dir = doc.project_root / "files" / "convergence_cutoff" / "relax_a"
    _write_previous_run_output(job_dir, "relax_a")

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    from mufinder.services.scheduler import create_scheduler
    from tests.support.cluster_ssh import FakeSSH

    runner = window._cluster_runner._runner
    runner._ssh = FakeSSH()
    runner._scheduler = create_scheduler(
        "addqueue",
        queue_check_command_path="/opt/slurm/squeue",
        accounting_command_path="/usr/local/shared/slurm/bin/sacct",
    )
    commands = []

    def remote_command(command, timeout=30):
        commands.append(command)
        if command.startswith("/opt/slurm/squeue "):
            return 0, "", ""
        assert command.startswith("/usr/local/shared/slurm/bin/sacct ")
        if accounting_state == "ERROR":
            return 127, "", "sacct: command not found"
        return 0, f"55501|{accounting_state}|00:00:01|relax_a\n", ""

    monkeypatch.setattr(runner._ssh, "exec_command", remote_command, raising=False)
    monkeypatch.setattr(main_window_module, "show_detailed_warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        jobs_controller_module, "show_detailed_warning", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        window._jobs,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("starting again was already confirmed"),
    )

    window._jobs.submit_jobs_on_cluster(
        [job],
        eligible_statuses={JobStatus.ENDED},
        fresh_start_job_ids={job.id},
    )

    # The previous run is confirmed finished before anything is cleared, and
    # again just before its directory is deleted.
    assert len(commands) == (4 if accounting_state == "FAILED" else 2)
    if accounting_state != "FAILED":
        assert recorded["deleted"] == []
        assert recorded["submitted"] == []
        window.close()
        return
    assert recorded["checked"] == []
    assert recorded["deleted"] == ["/remote/work/relax_a"]
    assert len(recorded["submitted"]) == 1
    submitted = recorded["submitted"][0]
    assert submitted["remote_directory"] == "/remote/work/relax_a"
    assert sorted(path.name for path in job_dir.iterdir()) == ["relax_a.cell", "relax_a.param"]
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
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.WALLTIME
    job_dir = doc.project_root / "files" / "convergence_cutoff" / "relax_c"
    _write_previous_run_output(job_dir, "relax_c")

    monkeypatch.setattr(
        jobs_controller_module,
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

    monkeypatch.setattr(jobs_controller_module, "LocalSubmissionDialog", _FakeDialog)
    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._local_runner, "submit", lambda **kwargs: submit_calls.append(kwargs)
    )

    window._jobs.submit_jobs_locally(
        [job],
        eligible_statuses={JobStatus.ENDED},
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
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.WALLTIME

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module,
        "show_detailed_warning",
        lambda _parent, _title, text, **_kw: warnings.append(text),
    )
    monkeypatch.setattr(
        jobs_controller_module,
        "show_detailed_warning",
        lambda _parent, _title, text, **_kw: warnings.append(text),
    )

    def _locked(_directory: Path) -> None:
        raise PermissionError("file in use")

    monkeypatch.setattr(JobFileCleanup, "clear_directory_contents", staticmethod(_locked))

    window._jobs.submit_jobs_on_cluster(
        [job],
        eligible_statuses={JobStatus.ENDED},
        fresh_start_job_ids={job.id},
    )

    # Submitting would resume from a checkpoint the user asked to discard.
    assert recorded["submitted"] == []
    assert recorded["deleted"] == []
    assert job_status_word(job) == "stopped"
    assert "relax_d: could not clear its previous output (file in use)" in warnings[0]
    window.close()


# ---------------------------------------------------------------------------
# Running an ended row again: one question, from the row or the dashboard
# ---------------------------------------------------------------------------


def _ended_pristine_row(tmp_path: Path, *, checkpoint: bool, target: str = "local"):
    """The pristine relaxation row of ``_pristine_continue_fixture``, its job stopped.

    The job's folder holds the previous run's output, with a checkpoint when
    ``checkpoint`` is set. A cluster ``target`` records the job's remote folder.
    """

    window, service, doc, definition, job, output_root, param_path = _pristine_continue_fixture(
        tmp_path
    )
    job.execution_target = target
    service.submit_job(job)
    service.end_job(job, EndReason.WALLTIME, "walltime reached")
    if target != "local":
        job.remote_directory = f"/remote/work/{job.label}"
        job.remote_job_id = "old-remote-id"
    job_dir = param_path.parent
    (job_dir / f"{job.label}.castep").write_text("previous run\n", encoding="utf-8")
    (job_dir / f"{job.label}-out.cell").write_text("previous geometry\n", encoding="utf-8")
    if checkpoint:
        (job_dir / f"{job.label}.check").write_bytes(b"checkpoint")
    window._refresh_ui_after_project_change()
    return window, service, doc, definition, job, job_dir, param_path


def _answer_run_again(monkeypatch: pytest.MonkeyPatch, answer: str) -> list[dict[str, object]]:
    asked: list[dict[str, object]] = []

    def _answer(_parent, title, text, choices, **kwargs) -> str:
        asked.append({"title": title, "text": text, "choices": list(choices), **kwargs})
        return answer

    monkeypatch.setattr(workflow_table_utils, "confirm_choice", _answer)
    return asked


def _record_local_submissions(
    window, monkeypatch: pytest.MonkeyPatch, *, accept: bool = True
) -> list[list[Job]]:
    return fake_local_submission(window, monkeypatch, accept=accept)


@pytest.mark.parametrize("entry", ["row", "dashboard"])
def test_run_again_asks_once_when_a_checkpoint_exists_and_continue_keeps_the_files(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    window, _service, doc, _definition, job, job_dir, param_path = _ended_pristine_row(
        tmp_path, checkpoint=True
    )
    asked = _answer_run_again(monkeypatch, "continue")
    submitted = _record_local_submissions(window, monkeypatch)
    monkeypatch.setattr(
        jobs_controller_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/opt/castep", mpirun_path=""),
    )

    if entry == "row":
        window._pristine_relaxation_controller.submit(["row_1"], target="local")
    else:
        window._jobs._on_resubmit_from_dashboard([job.id])

    (question,) = asked
    assert question["title"] == "Run again"
    assert question["text"] == (
        "'pristine_relax_row_1' did not finish. Continue from the previous run's files, "
        "or start again?"
    )
    assert question["choices"] == [
        ("continue", "Continue"),
        ("start_again", "Start again"),
        ("cancel", "Cancel"),
    ]
    assert (question["default"], question["escape"]) == ("continue", "cancel")
    assert question["destructive"] == ("start_again",)
    assert "It has a checkpoint to continue from." in question["detail"]
    assert submitted == [[job]]
    assert job.status is JobStatus.DEFINED
    # Continue keeps the previous run's files, checkpoint included, and rebuilds the inputs.
    assert (job_dir / f"{job.label}.check").is_file()
    assert (job_dir / f"{job.label}.castep").is_file()
    assert "CONTINUATION : DEFAULT" in param_path.read_text(encoding="utf-8")
    assert [record.action for record in doc.provenance][-1] == "job_continuation"
    window.close()


def test_start_again_deletes_the_previous_output_and_rebuilds_the_inputs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _service, doc, _definition, job, job_dir, param_path = _ended_pristine_row(
        tmp_path, checkpoint=True
    )
    _answer_run_again(monkeypatch, "start_again")
    submitted = _record_local_submissions(window, monkeypatch)

    window._pristine_relaxation_controller.submit(["row_1"], target="local")

    assert submitted == [[job]]
    assert job.status is JobStatus.DEFINED
    assert sorted(path.name for path in job_dir.iterdir()) == [
        f"{job.label}.cell",
        f"{job.label}.param",
    ]
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")
    assert [record.action for record in doc.provenance][-1] == "job_reset"
    window.close()


def test_a_relaxation_without_a_checkpoint_starts_again_without_a_question(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answered_choices
) -> None:
    window, _service, _doc, _definition, job, job_dir, _param_path = _ended_pristine_row(
        tmp_path, checkpoint=False
    )
    submitted = _record_local_submissions(window, monkeypatch)

    window._pristine_relaxation_controller.submit(["row_1"], target="local")

    assert answered_choices == []
    assert submitted == [[job]]
    assert not (job_dir / f"{job.label}.castep").exists()
    window.close()


@pytest.mark.parametrize("entry", ["row", "dashboard"])
def test_cancelling_the_run_again_question_submits_nothing_and_changes_no_job(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    window, _service, doc, _definition, job, job_dir, param_path = _ended_pristine_row(
        tmp_path, checkpoint=True
    )
    window._pristine_relaxation_controller.update_row_parameters(
        "row_1", {"geom_max_iter": 300}, ()
    )
    _answer_run_again(monkeypatch, "cancel")
    submitted = _record_local_submissions(window, monkeypatch)
    monkeypatch.setattr(
        jobs_controller_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/opt/castep", mpirun_path=""),
    )
    before = (copy.deepcopy(job), sorted(path.name for path in job_dir.iterdir()))
    param_text = param_path.read_text(encoding="utf-8")
    journal = list(doc.provenance)

    if entry == "row":
        window._pristine_relaxation_controller.submit(["row_1"], target="local")
    else:
        window._jobs._on_resubmit_from_dashboard([job.id])

    assert submitted == []
    assert (job, sorted(path.name for path in job_dir.iterdir())) == before
    assert param_path.read_text(encoding="utf-8") == param_text
    assert list(doc.provenance) == journal
    assert job_status_word(job) == "stopped"
    window.close()


def test_run_again_asks_when_the_checkpoint_cannot_be_checked_and_warns_of_unretrieved_output(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _service, _doc, _definition, job, _job_dir, _param_path = _ended_pristine_row(
        tmp_path, checkpoint=False, target="alpha"
    )
    job.end_reason = EndReason.INTERRUPTED
    asked = _answer_run_again(monkeypatch, "cancel")

    window._pristine_relaxation_controller.submit(["row_1"], target="cluster")

    (question,) = asked
    assert (
        "Its checkpoint could not be checked (Not connected to alpha). Without one, Continue "
        "starts it again from the beginning."
    ) in question["detail"]
    assert (
        "Its output may still be on alpha and has not been retrieved. Starting again does not "
        "retrieve it; use Check now or Refresh Queue first to keep the result."
    ) in question["detail"]
    window.close()


def test_row_and_dashboard_run_an_edited_ended_row_with_the_same_inputs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered: dict[str, tuple[str, str]] = {}
    for entry in ("row", "dashboard"):
        window, _service, _doc, _definition, job, job_dir, param_path = _ended_pristine_row(
            tmp_path / entry, checkpoint=True
        )
        # The ended row is editable; the run rebuilds its inputs from the edit.
        assert not workflow_table_utils.workflow_row_is_locked(
            window._pristine_relaxation_view._row_status(0)
        )
        window._pristine_relaxation_controller.update_row_parameters(
            "row_1", {"geom_max_iter": 300}, ()
        )
        _answer_run_again(monkeypatch, "continue")
        submitted = _record_local_submissions(window, monkeypatch)
        monkeypatch.setattr(
            jobs_controller_module,
            "load_castep_setup",
            lambda _settings: CastepSetup(executable_path="/opt/castep", mpirun_path=""),
        )
        if entry == "row":
            window._pristine_relaxation_controller.submit(["row_1"], target="local")
        else:
            window._jobs._on_resubmit_from_dashboard([job.id])
        assert submitted == [[job]]
        rendered[entry] = (
            param_path.read_text(encoding="utf-8"),
            (job_dir / f"{job.label}.cell").read_text(encoding="utf-8"),
        )
        window.close()

    assert rendered["row"] == rendered["dashboard"]
    assert "GEOM_MAX_ITER : 300" in rendered["row"][0]


class _RecordingClusterSubmissionDialog(_FakeClusterSubmissionDialogForContinue):
    opened: list[dict[str, object]] = []

    def __init__(self, **kwargs) -> None:
        type(self).opened.append(kwargs)


def test_dashboard_resubmit_continues_on_the_jobs_own_cluster_with_rebuilt_inputs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continue on a cluster uploads the rebuilt inputs into the kept remote folder.

    The job keeps its ``remote_directory``, so the submission writes into it
    as-is (no existence check, nothing deleted) and uploads the new ``.cell``
    and ``.param``: the fresh ``.param`` is the marker retrieval orders the
    previous run's output against. The previous ``-out.cell`` is left alone, so
    the relaxation's own stale-output check still rejects it until this run
    writes a new one.
    """

    window, _service, doc, _definition, job, job_dir, param_path = _ended_pristine_row(
        tmp_path, checkpoint=False, target="alpha"
    )
    out_cell = job_dir / f"{job.label}-out.cell"
    out_cell_mtime = out_cell.stat().st_mtime_ns
    window._pristine_relaxation_controller.update_row_parameters(
        "row_1", {"geom_max_iter": 500}, ()
    )
    submit_calls = _wire_fake_cluster_submission(window, monkeypatch, cluster_name="alpha")
    _RecordingClusterSubmissionDialog.opened = []
    monkeypatch.setattr(
        jobs_controller_module, "ClusterSubmissionDialog", _RecordingClusterSubmissionDialog
    )
    # Not connected, so the checkpoint cannot be checked and the question is asked.
    asked = _answer_run_again(monkeypatch, "continue")

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert len(asked) == 1
    assert _RecordingClusterSubmissionDialog.opened[0]["default_cluster_name"] == "alpha"
    (call,) = submit_calls
    assert call["remote_directory"] == "/remote/work/pristine_relax_row_1"
    assert call["allow_existing_remote_directory"] is True
    assert "GEOM_MAX_ITER : 500" in param_path.read_text(encoding="utf-8")
    assert out_cell.stat().st_mtime_ns == out_cell_mtime
    assert job.remote_directory == "/remote/work/pristine_relax_row_1"
    assert job.remote_job_id == "new-remote-job-id"
    assert any(record.action == "job_continuation" for record in doc.provenance)
    window.close()


def test_dashboard_resubmit_runs_each_row_on_its_own_target(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selection goes to each job's own controller and row, once per workflow and target."""

    window, service, doc, _definition, relaxation, _job_dir, _param_path = _ended_pristine_row(
        tmp_path, checkpoint=False, target="alpha"
    )
    cutoff = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=doc.project.host_structure_id,
        parameters={
            "name_prefix": "conv",
            "sweep_values": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(cutoff.id)},
    )
    service.submit_job(cutoff)
    service.end_job(cutoff, EndReason.CANCELLED)
    window._refresh_ui_after_project_change()
    monkeypatch.setattr(
        jobs_controller_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/opt/castep", mpirun_path=""),
    )
    _wire_fake_cluster_submission(window, monkeypatch)
    calls: list[tuple[str, list[object], dict[str, object]]] = []
    for name in ("_pristine_relaxation_controller", "_convergence_controller"):
        controller = getattr(window, name)
        monkeypatch.setattr(
            controller,
            "submit",
            lambda keys, *, _name=name, **kwargs: calls.append((_name, list(keys), kwargs)),
        )

    assert window._jobs.can_resubmit(relaxation) and window._jobs.can_resubmit(cutoff)
    window._jobs._on_resubmit_from_dashboard([relaxation.id, cutoff.id])

    assert calls == [
        (
            "_pristine_relaxation_controller",
            ["row_1"],
            {"target": "cluster", "default_cluster_name": "alpha"},
        ),
        (
            "_convergence_controller",
            [window._convergence_controller.key_for_job(cutoff.id)],
            {"target": "local"},
        ),
    ]
    window.close()


def test_a_job_without_a_row_is_not_offered_resubmit(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "rowless.mufinder", "Rowless")
    window._document = doc
    failed = _failed_job(service, doc, "rowless_failed")
    cancelled = _failed_job(service, doc, "rowless_cancelled")
    cancelled.end_reason = EndReason.CANCELLED
    window._refresh_ui_after_project_change()
    dashboard = window._job_dashboard
    infos: list[str] = []
    monkeypatch.setattr(
        jobs_controller_module.QMessageBox,
        "information",
        lambda _parent, title, *_args, **_kwargs: infos.append(title),
    )
    submitted = _record_local_submissions(window, monkeypatch)

    assert not window._jobs.can_resubmit(failed)
    assert not dashboard.action_applies("resubmit", failed)
    # The run slot is left blank, not greyed out.
    assert dashboard.slot_actions(cancelled) == (None, None, "dismiss")
    assert dashboard.slot_actions(failed) == (None, "inspect", "dismiss")
    dialogs = capture_failed_job_dialogs(monkeypatch)
    window._jobs.inspect_failed_jobs([failed])
    assert dialogs[0].resubmit_button is None
    # A row that went away after the menu opened is reported, and nothing is run.
    window._jobs._on_resubmit_from_dashboard([failed.id])
    assert infos == ["Jobs not resubmitted"]
    assert submitted == []
    assert job_status_word(failed) == "failed"


def _without_castep_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        jobs_controller_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="", mpirun_path=""),
    )


def test_a_missing_executable_is_reported_before_the_run_again_question(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _service, _doc, _definition, job, _job_dir, _param_path = _ended_pristine_row(
        tmp_path, checkpoint=True
    )
    submitted = _record_local_submissions(window, monkeypatch)
    _without_castep_executable(monkeypatch)
    asked = _answer_run_again(monkeypatch, "cancel")

    window._jobs._on_resubmit_from_dashboard([job.id])

    (prompt,) = asked
    assert prompt["title"] == "CASTEP executable missing"
    assert prompt["choices"] == [("setup", "Open Setup > CASTEP…"), ("cancel", "Cancel")]
    assert submitted == []
    assert job_status_word(job) == "stopped"
    window.close()


def test_the_keyword_catalog_guard_runs_before_the_run_again_question(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _service, _doc, _definition, job, _job_dir, _param_path = _ended_pristine_row(
        tmp_path, checkpoint=True, target="alpha"
    )
    _wire_fake_cluster_submission(window, monkeypatch)
    asked = _answer_run_again(monkeypatch, "continue")
    monkeypatch.setattr(window._keyword_catalog_guard, "confirm_submission", lambda: False)

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert asked == []
    assert job_status_word(job) == "stopped"
    window.close()


def test_no_configured_cluster_offers_hpc_cluster_setup_before_the_question(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _service, _doc, _definition, job, _job_dir, _param_path = _ended_pristine_row(
        tmp_path, checkpoint=True, target="alpha"
    )
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [])
    asked = _answer_run_again(monkeypatch, "cancel")

    window._jobs._on_resubmit_from_dashboard([job.id])

    (prompt,) = asked
    assert prompt["title"] == "No configured clusters"
    assert prompt["choices"] == [("setup", "Open Setup > HPC Clusters…"), ("cancel", "Cancel")]
    window.close()


def test_local_checkpoint_probe(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "probe.mufinder", "Probe")
    window._document = doc
    jobs = [
        _failed_job(service, doc, label, stage=WorkflowStage.MUON_RELAXATION)
        for label in ("relax_a", "relax_b")
    ]
    found_dir, absent_dir = tmp_path / "relax_a", tmp_path / "relax_b"
    for job, job_dir in zip(jobs, (found_dir, absent_dir), strict=True):
        job.execution_target = "local"
        job_dir.mkdir()
    (found_dir / "relax_a.check").write_bytes(b"checkpoint")
    directories = {jobs[0].id: found_dir, jobs[1].id: absent_dir}
    window._job_output_directory = lambda job: directories[job.id]

    probes = window._jobs._checkpoint_probes(jobs)

    assert probes[jobs[0].id].status == CheckpointProbeStatus.FOUND
    assert probes[jobs[1].id].status == CheckpointProbeStatus.ABSENT


_FOUND = CheckpointProbeResult.found()


@pytest.mark.parametrize(
    ("probes", "expected"),
    [
        ([_FOUND], "It has a checkpoint to continue from."),
        ([_FOUND, _FOUND], "Each has a checkpoint to continue from."),
        (
            [CheckpointProbeResult.unavailable("Not connected to alpha")],
            "Its checkpoint could not be checked (Not connected to alpha). Without one, "
            "Continue starts it again from the beginning.",
        ),
        (
            [
                CheckpointProbeResult.unavailable("Not connected to alpha"),
                CheckpointProbeResult.unavailable("Not connected to alpha"),
            ],
            "Their checkpoints could not be checked (Not connected to alpha). A job "
            "without one starts again from the beginning.",
        ),
        (
            [_FOUND, _FOUND, CheckpointProbeResult.unavailable("")],
            "2 have a checkpoint; 1 could not be checked. A job without one starts "
            "again from the beginning.",
        ),
    ],
)
def test_checkpoint_line(probes: list[CheckpointProbeResult], expected: str) -> None:
    assert jobs_controller_module.checkpoint_line(probes) == expected


def test_run_again_question_counts_the_jobs_that_start_again_anyway(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, service, doc, _definition, job, _job_dir, _param_path = _ended_pristine_row(
        tmp_path, checkpoint=True
    )
    other = _failed_job(service, doc, "conv_other")
    asked = _answer_run_again(monkeypatch, "cancel")

    assert window._jobs.prepare_jobs_for_resubmission([job, other]) is None

    (question,) = asked
    assert "1 other job has no checkpoint and starts again." in question["detail"]
    assert job_status_word(other) == "failed"
    window.close()


# ---------------------------------------------------------------------------
# Nothing is lost before the submission is accepted
# ---------------------------------------------------------------------------


class _RejectingClusterSubmissionDialog(_FakeClusterSubmissionDialogForContinue):
    def exec(self) -> int:
        return 0


def _snapshot(job: Job, job_dir: Path, doc) -> tuple:
    return (
        copy.deepcopy(job),
        {path.name: path.read_bytes() for path in sorted(job_dir.iterdir())},
        list(doc.provenance),
    )


@pytest.mark.parametrize("answer", ["continue", "start_again"])
@pytest.mark.parametrize("target", ["local", "alpha"])
@pytest.mark.parametrize("entry", ["row", "dashboard"])
def test_cancelling_the_submission_dialog_changes_no_job_or_file(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str, target: str, answer: str
) -> None:
    """The run-again answer is applied only once the submission dialog is accepted.

    Whatever the answer, a cancelled submission dialog leaves the job ended
    (Inspect still works), its folder byte for byte as it was, and the journal
    without a reset or continuation entry.
    """

    window, _service, doc, _definition, job, job_dir, _param_path = _ended_pristine_row(
        tmp_path, checkpoint=True, target=target
    )
    window._pristine_relaxation_controller.update_row_parameters(
        "row_1", {"geom_max_iter": 300}, ()
    )
    asked = _answer_run_again(monkeypatch, answer)
    submitted = fake_local_submission(window, monkeypatch, accept=False)
    if target != "local":
        _wire_fake_cluster_submission(window, monkeypatch, cluster_name=target)
        monkeypatch.setattr(
            jobs_controller_module, "ClusterSubmissionDialog", _RejectingClusterSubmissionDialog
        )
    before = _snapshot(job, job_dir, doc)

    if entry == "row":
        window._pristine_relaxation_controller.submit(
            ["row_1"], target="local" if target == "local" else "cluster"
        )
    else:
        window._jobs._on_resubmit_from_dashboard([job.id])

    assert [question["title"] for question in asked] == ["Run again"]
    assert submitted == []
    assert _snapshot(job, job_dir, doc) == before
    assert job_status_word(job) == "stopped"
    window.close()


def _wire_scheduler(window, monkeypatch: pytest.MonkeyPatch, answer) -> dict[str, list]:
    """Answer the scheduler check before a start-again deletes the job's remote folder."""

    recorded: dict[str, list] = {"asked": [], "deleted": [], "warnings": []}

    def scheduler_job_is_active(remote_job_id: str) -> bool:
        recorded["asked"].append(remote_job_id)
        if isinstance(answer, Exception):
            raise answer
        return answer

    runner = window._cluster_runner._runner
    monkeypatch.setattr(runner, "scheduler_job_is_active", scheduler_job_is_active)
    monkeypatch.setattr(runner, "remove_remote_directory", recorded["deleted"].append)
    for module in (main_window_module, jobs_controller_module):
        monkeypatch.setattr(
            module,
            "show_detailed_warning",
            lambda _parent, title, text, **_kwargs: recorded["warnings"].append(f"{title}: {text}"),
        )
    return recorded


@pytest.mark.parametrize(
    "scheduler",
    [True, RuntimeError("sacct: command not found")],
    ids=["still-listed", "accounting-error"],
)
@pytest.mark.parametrize("entry", ["row", "dashboard"])
def test_start_again_on_a_cluster_keeps_the_folder_while_the_previous_run_may_be_alive(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str, scheduler
) -> None:
    """Start again deletes the job's own remote folder only once its previous run finished.

    An ended job can still run on the cluster (cancelled in the project while
    disconnected, or lost track of). While the scheduler still lists it, or
    cannot be asked, the folder stays and the submission is reported as failed.
    """

    window, _service, doc, _definition, job, job_dir, _param_path = _ended_pristine_row(
        tmp_path, checkpoint=False, target="alpha"
    )
    submit_calls = _wire_fake_cluster_submission(window, monkeypatch, cluster_name="alpha")
    recorded = _wire_scheduler(window, monkeypatch, scheduler)
    _answer_run_again(monkeypatch, "start_again")
    before = _snapshot(job, job_dir, doc)

    if entry == "row":
        window._pristine_relaxation_controller.submit(["row_1"], target="cluster")
    else:
        window._jobs._on_resubmit_from_dashboard([job.id])

    assert recorded["asked"] == ["old-remote-id"]
    assert recorded["deleted"] == []
    assert submit_calls == []
    (warning,) = recorded["warnings"]
    assert "pristine_relax_row_1: not started again" in warning
    assert "scheduler job old-remote-id" in warning
    # Refused before anything was cleared: the job keeps its record (and its
    # previous scheduler job ID, so a later try is checked again) and its files.
    assert _snapshot(job, job_dir, doc) == before

    window.close()


def test_start_again_on_a_cluster_deletes_the_old_folder_once_the_previous_run_finished(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _service, doc, _definition, job, job_dir, param_path = _ended_pristine_row(
        tmp_path, checkpoint=True, target="alpha"
    )
    submit_calls = _wire_fake_cluster_submission(window, monkeypatch, cluster_name="alpha")
    recorded = _wire_scheduler(window, monkeypatch, False)
    _answer_run_again(monkeypatch, "start_again")

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert recorded["deleted"] == ["/remote/work/pristine_relax_row_1"]
    (call,) = submit_calls
    assert call["remote_directory"] == "/remote/work/pristine_relax_row_1"
    assert sorted(path.name for path in job_dir.iterdir()) == [
        f"{job.label}.cell",
        f"{job.label}.param",
    ]
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")
    reset = [record for record in doc.provenance if record.action == "job_reset"]
    assert reset[-1].parameters["cleaned_output_directory"] is True
    window.close()


# ---------------------------------------------------------------------------
# Output that may still be on the cluster is never started over silently
# ---------------------------------------------------------------------------


def _ended_cutoff_row(tmp_path: Path, *, reason: EndReason, target: str):
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "ended_cutoff.mufinder", "EndedCutoff")
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
            "sweep_values": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )
    window._document = doc
    window._refresh_ui_after_project_change()
    window._convergence_controller.materialize([300.0])
    job.execution_target = target
    service.submit_job(job)
    service.end_job(job, reason, "no output from this run")
    if target != "local":
        job.remote_directory = "/remote/work/conv_ecut_300"
        job.remote_job_id = "old-remote-id"
    job_dir = window._job_output_directory(job)
    (job_dir / f"{job.label}.castep").write_text("previous run\n", encoding="utf-8")
    window._refresh_ui_after_project_change()
    return window, service, doc, job, job_dir


@pytest.mark.parametrize("entry", ["row", "dashboard"])
def test_starting_again_over_unretrieved_output_asks_first_and_cancel_changes_nothing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """A sweep job whose output may still be on the cluster is not started over silently.

    It has nothing to continue from, so the question is a confirmation to start
    again, with the unretrieved-output warning and ``Cancel`` as the default.
    """

    window, _service, doc, job, job_dir = _ended_cutoff_row(
        tmp_path, reason=EndReason.OUTPUT_UNAVAILABLE, target="alpha"
    )
    _wire_fake_cluster_submission(window, monkeypatch, cluster_name="alpha")
    asked: list[tuple] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt, **_kwargs: asked.append(prompt) or False,
    )
    before = _snapshot(job, job_dir, doc)

    if entry == "row":
        window._convergence_controller.submit([300.0], target="cluster")
    else:
        window._jobs._on_resubmit_from_dashboard([job.id])

    ((title, text, detail, accept_label),) = asked
    assert (title, text, accept_label) == (
        "Run again",
        "Start 'conv_ecut_300' again?",
        "Start again",
    )
    assert (
        "Its output may still be on alpha and has not been retrieved. Starting again does not "
        "retrieve it"
    ) in detail
    assert _snapshot(job, job_dir, doc) == before
    window.close()


def test_a_relaxation_with_unretrieved_output_defaults_the_one_question_to_cancel(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _service, _doc, _definition, job, _job_dir, _param_path = _ended_pristine_row(
        tmp_path, checkpoint=False, target="alpha"
    )
    job.end_reason = EndReason.INTERRUPTED
    asked = _answer_run_again(monkeypatch, "cancel")
    confirmed: list[tuple] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt, **_kwargs: confirmed.append(prompt) or True,
    )

    window._pristine_relaxation_controller.submit(["row_1"], target="cluster")

    (question,) = asked
    assert question["default"] == "cancel"
    assert "Starting again does not retrieve it" in question["detail"]
    # One box: the warning is folded into the run-again question.
    assert confirmed == []
    window.close()


def test_a_sweep_job_without_unretrieved_output_runs_again_without_a_question(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answered_choices
) -> None:
    window, _service, _doc, job, job_dir = _ended_cutoff_row(
        tmp_path, reason=EndReason.CASTEP_ERROR, target="local"
    )
    confirmed: list[tuple] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt, **_kwargs: confirmed.append(prompt) or True,
    )
    submitted = fake_local_submission(window, monkeypatch)

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert answered_choices == [] and confirmed == []
    assert submitted == [[job]]
    assert not (job_dir / f"{job.label}.castep").exists()
    window.close()
