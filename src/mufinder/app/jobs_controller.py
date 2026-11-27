"""The jobs controller: what happens to jobs when the user acts on them.

``JobsController`` carries out every job action the Job Dashboard, the workflow
rows and the workflow controllers ask for: local and cluster submission (with
their dialogs, the cluster connection and the waits on the cluster worker),
resubmission, Continue relaxation, cancel, inspect, ``Check now`` and
dismiss. It asks and reports through dialogs and the status bar and leaves
every change of job state to :class:`~mufinder.services.JobExecutionController`
and :class:`~mufinder.services.project_service.ProjectService`
(see :doc:`/job_lifecycle`, Jobs controller).

It holds no reference to the main window. What it needs from the window (the
open document, page refreshes, the workflow controllers and the setup dialogs)
comes through :class:`JobsControllerDeps`, late-bound, as the workflow
controllers take theirs through ``WorkflowControllerContext``.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
import posixpath
from uuid import UUID
from dataclasses import dataclass

from mufinder.app.job_dashboard import ClusterQueueStatus, JobDashboard
import logging
from mufinder.domain.models import Result
from mufinder.app.cluster_ui_utils import prompt_for_cluster_credentials, prompt_to_trust_host_key
from mufinder.app.theme import current_palette
from mufinder.app.workflow_config_codecs import num_farms_from_extras, workflow_row_configuration
from mufinder.app.workflow_controllers import merge_controller_result, validate_controller_artifacts
from mufinder.castep_io.input_diff import (
    JobInputTexts,
    overlay_job_inputs,
    read_job_inputs,
    write_job_inputs,
)
from mufinder.castep_io.output_parser import CastepResult
from mufinder.services.cluster_status import (
    ClusterConnectionTone,
    cluster_status,
    refreshable_cluster_names,
)
from mufinder.services.job_artifacts import ParsedJobArtifacts, collect_job_artifacts
from mufinder.services.job_paths import job_output_files
from mufinder.services.job_reconcile import JobEvidence
from mufinder.services.ssh_client import (
    SSHAuthenticationError,
    SSHUnknownHostKeyError,
    save_trusted_host_key,
)
from mufinder.workflow.energy_barriers import NUDGED_ELASTIC_BAND_CALC_TYPE
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE

from PySide6.QtCore import (
    QObject,
    QEventLoop,
    QSettings,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QDialog,
    QMessageBox,
    QProgressDialog,
    QStatusBar,
    QWidget,
)
from mufinder.castep_setup import load_castep_setup
from mufinder.cluster_setup import load_cluster_config, load_cluster_configs
from mufinder.domain.cluster import ClusterConfig, ClusterSubmissionConfig
from mufinder.domain.enums import EndReason, JobActivity, JobStatus, WorkflowStage
from mufinder.domain.models import (
    Job,
)
from mufinder.app.cluster_submission_dialog import ClusterSubmissionDialog
from mufinder.app.cluster_ui_utils import (
    format_cluster_connection_error,
    show_detailed_warning,
)
from mufinder.app.keyword_catalog_guard import KeywordCatalogGuard
from mufinder.app.export_dialogs import open_containing_folder
from mufinder.app.failed_job_dialog import FailedJobAction, FailedJobDialog
from mufinder.domain.text import (
    plural,
)
from mufinder.app.relaxation_types import (
    MuonRelaxationRow,
)
from mufinder.app.relaxation_widget import PristineRelaxationWidget
from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget
from mufinder.app.adapter_common import inherited_project_default_names
from mufinder.app.continue_relaxation_dialog import ContinueRelaxationDialog
from mufinder.app.geometry_progress_dialog import (
    GeometryProgressDialog,
)
from mufinder.app.submission_dialog import LocalSubmissionDialog
from mufinder.app.workflow_config_codecs import (
    deserialize_muon_relaxation_config,
    deserialize_pristine_relaxation_config,
)
from mufinder.app.workflow_submission import (
    RESUBMITTABLE_JOB_STATES,
    RUN_AGAIN_STATES,
    OnAccepted,
    RunAgainPlan,
)
from mufinder.app.workflow_controllers import (
    EnergyBarriersController,
    ConvergenceCutoffController,
    ConvergenceKpointController,
    MuonRelaxationController,
    PristineRelaxationController,
    QuantumCorrectionsController,
    WorkflowJobController,
    dedupe_jobs,
    owning_controller,
    planned_job_entry,
)
from mufinder.castep_io.geometry_progress import GeometryProgress
from mufinder.app.cluster_controller import ClusterController
from mufinder.services.cluster_runner import CheckpointProbeResult, CheckpointProbeStatus
from mufinder.services.cluster_submission_planner import (
    ClusterSubmissionPlan,
    can_reuse_cluster_connection,
    plan_cluster_submission,
    unsafe_job_labels_message,
)
from mufinder.services.job_execution_controller import (
    JobExecutionController,
    PreparedClusterSubmission,
    remember_cluster_submission,
    remember_local_cores,
    remembered_cluster_submission,
    remembered_local_cores,
)
from mufinder.services.failure_diagnosis import (
    FailureDiagnosis,
    failed_job_report,
)
from mufinder.services.job_artifacts import (
    RetrievalFlags,
)
from mufinder.services.job_file_cleanup import (
    JobFileCleanup,
)
from mufinder.services.job_paths import (
    job_directory_label,
)
from mufinder.services.job_monitoring import (
    has_unretrieved_cluster_output,
    job_is_imported,
    job_resumes_from_checkpoint,
    unretrieved_output_note,
)
from mufinder.services.job_progress import JobProgressMonitor
from mufinder.services.local_runner import LocalJobRunner
from mufinder.app import workflow_table_utils
from mufinder.services.project_service import (
    CONTINUABLE_JOB_STATUSES,
    ProjectDocument,
    ProjectService,
)
from mufinder.services.project_summary import (
    job_stage_title,
)
from mufinder.services.ssh_client import (
    SSHCredentials,
)
from mufinder.workflow.naming import cluster_workflow_directory_fragment


_LOG = logging.getLogger(__name__)


def checkpoint_line(probes: Sequence[CheckpointProbeResult]) -> str:
    """Return the run-again question's line on the jobs' checkpoints.

    ``probes`` are those of the jobs that may continue: each either has
    a checkpoint or could not be checked (an absent one is never asked about).
    The reasons a check failed are named once each.
    """

    total = len(probes)
    unknown = [probe for probe in probes if probe.status == CheckpointProbeStatus.UNAVAILABLE]
    found = total - len(unknown)
    if not unknown:
        if total == 1:
            return "It has a checkpoint to continue from."
        return "Each has a checkpoint to continue from."
    reasons = "; ".join(dict.fromkeys(probe.reason for probe in unknown if probe.reason))
    because = f" ({reasons})" if reasons else ""
    without = "A job without one starts again from the beginning."
    if not found:
        if total == 1:
            return (
                f"Its checkpoint could not be checked{because}. Without one, Continue "
                "starts it again from the beginning."
            )
        return f"Their checkpoints could not be checked{because}. {without}"
    return (
        f"{found} {'has' if found == 1 else 'have'} a checkpoint; {len(unknown)} could not be "
        f"checked{because}. {without}"
    )


@dataclass(frozen=True)
class JobsControllerDeps:
    """The window's state and helpers the jobs controller uses, late-bound."""

    document: Callable[[], ProjectDocument | None]
    #: The widget dialogs and prompts are parented to.
    dialog_parent: Callable[[], QWidget]
    status_bar: Callable[[], QStatusBar]
    job_output_directory: Callable[[Job], Path | None]
    job_seed_name: Callable[[Job], str]
    workflow_controllers: Callable[[], Sequence[WorkflowJobController]]
    convergence_controller: Callable[[], ConvergenceCutoffController]
    kpoint_convergence_controller: Callable[[], ConvergenceKpointController]
    pristine_relaxation_controller: Callable[[], PristineRelaxationController]
    muon_relaxation_controller: Callable[[], MuonRelaxationController]
    pristine_relaxation_view: Callable[[], PristineRelaxationWidget]
    muon_relaxation_view: Callable[[], MuonRelaxationWidget]
    muon_relaxation_row_key: Callable[[MuonRelaxationRow], str]
    refresh_jobs_table: Callable[[], None]
    refresh_journal_view: Callable[[], None]
    refresh_hyperfine_coupling_view: Callable[[], None]
    refresh_quantum_corrections_view: Callable[[], None]
    refresh_energy_barriers_view: Callable[[], None]
    refresh_views_after_job_reset: Callable[[], None]
    request_dashboard_flush: Callable[..., None]
    persist_document_ui_state: Callable[..., None]
    with_kpoint_status_note: Callable[[str], str]
    open_castep_setup_dialog: Callable[[], None]
    open_cluster_setup_dialog: Callable[[], None]
    edit_cluster_during_submission: Callable[..., object]
    keyword_catalog_guard: Callable[[], KeywordCatalogGuard]
    quantum_corrections_controller: Callable[[], QuantumCorrectionsController]
    energy_barriers_controller: Callable[[], EnergyBarriersController]
    job_dashboard: Callable[[], JobDashboard]
    #: Whether a coalesced dashboard refresh is waiting to run.
    dashboard_flush_pending: Callable[[], bool]


class JobsController(QObject):
    """Carries out the user's job actions; state changes go through the services."""

    def __init__(
        self,
        deps: JobsControllerDeps,
        *,
        project_service: ProjectService,
        settings: QSettings,
        job_execution: JobExecutionController,
        local_runner: LocalJobRunner,
        cluster_runner: ClusterController,
        job_file_cleanup: JobFileCleanup,
        job_progress: JobProgressMonitor,
        active_run_by_job: dict[UUID, UUID],
        cluster_status: ClusterQueueStatus,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._deps = deps
        self._project_service = project_service
        self._settings = settings
        self._job_execution = job_execution
        self._local_runner = local_runner
        self._cluster_runner = cluster_runner
        self._job_file_cleanup = job_file_cleanup
        self._job_progress = job_progress
        self._active_run_by_job = active_run_by_job
        self._geometry_progress_dialogs: dict[UUID, GeometryProgressDialog] = {}
        self._job_progress.progress_changed.connect(self._on_job_progress_changed)
        self._cluster_connection_label = cluster_status.connection_label
        self._cluster_queue_checked_label = cluster_status.queue_checked_label
        self._cluster_refresh_button = cluster_status.refresh_button
        self._cluster_refresh_button.clicked.connect(self._on_refresh_cluster_queue)
        self._cluster_last_connected_name: str | None = None
        self._cluster_lost_names: set[str] = set()
        self._cluster_status_tone = ClusterConnectionTone.NEUTRAL
        self._queue_refreshes_pending = 0
        # Every minute, check each active job: a local one against the local
        # runner and its output folder, a cluster one against the scheduler
        # and its remote folder. The handler is cheap when no job is active.
        self.check_timer = QTimer(self)
        self.check_timer.setInterval(60_000)
        self.check_timer.timeout.connect(self.check_jobs)
        self.check_timer.start()

    def connect_runners(self) -> None:
        """Apply what the local and cluster runners report to the jobs they name."""

        self._local_runner.job_started.connect(self._on_local_job_started)
        self._local_runner.job_finished.connect(self._on_local_job_finished)
        self._local_runner.job_failed.connect(self._on_local_job_failed)
        self._local_runner.job_cancelled.connect(self._on_local_job_cancelled)
        self._local_runner.status_changed.connect(self._on_local_job_status_changed)

        self._cluster_runner.job_cancelled.connect(self._on_cluster_job_cancelled)
        self._cluster_runner.job_checked.connect(self._on_cluster_job_checked)
        self._cluster_runner.connection_status_changed.connect(
            self._on_cluster_connection_status_changed
        )
        self._cluster_runner.queue_checked.connect(self._on_cluster_queue_checked)
        self._cluster_runner.poll_failed.connect(self._on_cluster_poll_failed)
        self._cluster_runner.abandoned_job_cancel_failed.connect(
            self._on_abandoned_cluster_job_cancel_failed
        )
        # Emitted while the window closes; the close saves what it records.
        self._cluster_runner.submission_cut_off.connect(
            lambda job_id: self._job_execution.note_submission_cut_off(job_id)
        )

    @property
    def _document(self) -> ProjectDocument | None:
        return self._deps.document()

    def _parent(self) -> QWidget:
        return self._deps.dialog_parent()

    def handle_dashboard_action(self, kind: str, job_ids: list[UUID]) -> None:
        """Carry out an action the Job Dashboard confirmed (all but ``open``)."""

        self._dashboard_action_handler(kind)(list(job_ids))

    def close_progress_dialogs(self) -> None:
        """Close every geometry progress dialog, as when the project closes."""

        for dialog in list(self._geometry_progress_dialogs.values()):
            dialog.close()

    def _is_cluster_connected_for_job(self, job: Job) -> bool:
        """Return True when the cluster runner is connected to this job's cluster.

        Mirrors the casefold comparison used by cluster resubmission
        (``submit_jobs_on_cluster``). Local jobs are never "cluster connected".
        """

        target = (job.execution_target or "").strip()
        if not target or target.casefold() == "local":
            return False
        if not self._cluster_runner.is_connected:
            return False
        connected = (self._cluster_runner.connected_cluster_name or "").strip()
        return bool(connected) and connected.casefold() == target.casefold()

    def _on_submit_local_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return
        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        self.submit_jobs_locally([job for job in jobs if job is not None])

    def _on_submit_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return

        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        selected = [job for job in jobs if job is not None]
        if not selected:
            return

        local_jobs = [job for job in selected if job.execution_target == "local"]
        cluster_jobs = [job for job in selected if job.execution_target != "local"]

        if local_jobs:
            self.submit_jobs_locally(local_jobs)

        if cluster_jobs:
            preferred = None
            targets = {job.execution_target for job in cluster_jobs}
            if len(targets) == 1:
                preferred = next(iter(targets))
            self.submit_jobs_on_cluster(cluster_jobs, default_cluster_name=preferred)

    def _on_resubmit_from_dashboard(self, job_ids: list[UUID]) -> None:
        """Run each ended job's workflow row again, as the row's Run does.

        Each job goes to its owning controller's
        :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.submit`
        for its row, on its own target: locally, or on its own cluster, which the
        cluster submission dialog preselects. Inputs are rebuilt from the row,
        and the row's continue-or-start-again question is asked there. A
        selection is submitted once per workflow and target. A job with no row
        (:meth:`can_resubmit`) is not offered Resubmit; one that lost its row
        since is reported and left as it is.
        """

        if self._document is None:
            return

        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        selected = [
            job for job in jobs if job is not None and job.status in RESUBMITTABLE_JOB_STATES
        ]
        if not selected:
            QMessageBox.information(
                self._parent(),
                "No resubmittable jobs",
                "Only failed, stopped, or cancelled jobs can be resubmitted from the job dashboard.",
            )
            return

        groups: dict[tuple[int, str], tuple[WorkflowJobController, str, list[object]]] = {}
        rowless: list[Job] = []
        for job in dedupe_jobs(selected):
            located = None if job_is_imported(job) else self._row_for_job(job)
            if located is None:
                rowless.append(job)
                continue
            controller, key = located
            target = (job.execution_target or "").strip() or "local"
            if target.casefold() == "local":
                target = "local"
            _controller, _target, keys = groups.setdefault(
                (id(controller), target.casefold()), (controller, target, [])
            )
            if key not in keys:
                keys.append(key)
        if rowless:
            QMessageBox.information(
                self._parent(),
                "Jobs not resubmitted",
                (
                    f"'{rowless[0].label}' has no row on its workflow page now, so it was "
                    "not resubmitted."
                    if len(rowless) == 1
                    else f"{plural(len(rowless), 'job has', 'jobs have')} no row on their "
                    "workflow pages now, so they were not resubmitted:\n\n"
                    + "\n".join(f"- {job.label}" for job in rowless)
                ),
            )

        # Check what the submission needs before the row asks how to run the
        # jobs again, so a missing executable or cluster is never reported after
        # the question.
        if any(target == "local" for _c, target, _k in groups.values()) and (
            not self._castep_executable_ready()
        ):
            groups = {key: group for key, group in groups.items() if group[1] != "local"}
        if any(target != "local" for _c, target, _k in groups.values()) and (
            not self._clusters_configured()
        ):
            groups = {key: group for key, group in groups.items() if group[1] == "local"}
        if not groups or not self._deps.keyword_catalog_guard().confirm_submission():
            return

        for controller, target, keys in groups.values():
            if target == "local":
                controller.submit(keys, target="local")
            else:
                controller.submit(keys, target="cluster", default_cluster_name=target)

    def _row_for_job(self, job: Job) -> tuple[WorkflowJobController, object] | None:
        """Return ``job``'s owning controller and its row key, or ``None`` without a row."""

        controller = owning_controller(self._deps.workflow_controllers(), job)
        if controller is None:
            return None
        key = controller.key_for_job(job.id)
        return None if key is None else (controller, key)

    def can_resubmit(self, job: Job) -> bool:
        """Return whether the Job Dashboard offers ``Resubmit…`` for ``job``.

        An ended job MuFinder ran, whose workflow row still holds it: Resubmit
        runs that row. An imported job, or one with no row, is not offered it.
        """

        return (
            job.status in RESUBMITTABLE_JOB_STATES
            and not job_is_imported(job)
            and self._row_for_job(job) is not None
        )

    def _fresh_start_refusals(self, jobs: Sequence[Job]) -> dict[UUID, str]:
        """Return, per job id, why starting that job again would be unsafe.

        Start again deletes a job's local output directory and its remote
        directory, so both must belong to that job alone (see
        :meth:`JobFileCleanup.directory_ownership_refusals`). Reconfiguring
        a relaxation row can leave an old failed job and its replacement
        sharing one directory, which is the case this exists for. A job's
        directory companions (the singlepoint an older project ran in a Quantum
        Corrections phonon job's folder) do not count: they are not run, and
        the clean-up may clear the folder they share.
        """

        controllers = self._deps.workflow_controllers()
        companions = {
            companion.id
            for job in jobs
            if (controller := owning_controller(controllers, job)) is not None
            for companion in controller.directory_companions(job)
        }
        return self._job_file_cleanup.directory_ownership_refusals(jobs, shared_with=companions)

    def _clear_outputs_for_fresh_start(self, jobs: Sequence[Job]) -> tuple[list[Job], list[str]]:
        """Empty each job's local output directory, keeping its ``.cell``/``.param``.

        A job refused by :meth:`_fresh_start_refusals`, or whose directory
        cannot be fully cleaned, is left out of the returned list: submitting it would either
        delete another job's files (its remote directory is deleted next) or
        resume from a checkpoint the user asked to discard.
        """

        refusals = self._fresh_start_refusals(jobs)
        cleaned: list[Job] = []
        errors: list[str] = list(refusals.values())
        for job in jobs:
            if job.id in refusals:
                continue
            if job_is_imported(job):
                # Defensive: the folder holds the only copy of the imported run.
                errors.append(f"- {job.label}: it records a run imported into MuFinder")
                continue
            output_dir = self._deps.job_output_directory(job)
            if output_dir is not None and output_dir.is_dir():
                try:
                    self._job_file_cleanup.clear_directory_contents(output_dir)
                except OSError as exc:
                    errors.append(f"- {job.label}: could not clear its previous output ({exc})")
                    continue
            cleaned.append(job)
        return cleaned, errors

    def _on_check_now_from_dashboard(self, job_ids: list[UUID]) -> None:
        """Check the jobs now, connecting first to each of their clusters that is not connected.

        A cluster is connected to as ``Refresh Queue`` connects (see
        :meth:`_reconnect_to_cluster`), so the button never does nothing because
        the connection is closed. The jobs on connected clusters are checked
        first, since connecting to another cluster closes the connection they use.
        """

        if self._document is None:
            return

        waiting = self._job_execution.check_now_waits_for_connection(job_ids)
        waiting_ids = {job_id for ids in waiting.values() for job_id in ids}
        updates, requested = self._job_execution.check_now(
            [job_id for job_id in job_ids if job_id not in waiting_ids]
        )
        failed_names: list[str] = []
        failures: list[str] = []
        for target, ids in waiting.items():
            name, failure = self._reconnect_to_cluster(target)
            if failure is not None:
                failed_names.append(name)
                failures.append(failure)
                continue
            more_updates, more_requested = self._job_execution.check_now(ids)
            updates += more_updates
            requested += more_requested
        if failures:
            self._deps.status_bar().showMessage("Failed to connect to " + ", ".join(failed_names))
            show_detailed_warning(
                self._parent(),
                "Check now failed",
                "Some clusters could not be connected to:\n" + "\n".join(failures),
                informative_text="Open Details to inspect the full connection failures.",
            )
        if not updates and not requested:
            if not failures:
                QMessageBox.information(
                    self._parent(),
                    "No jobs to check",
                    "Check now is for failed or stopped cluster jobs whose output may still "
                    "be on the cluster.",
                )
            return
        completed = 0
        for update in updates:
            if update.job is not None and update.finished is not None:
                self._on_local_job_finished(
                    update.job.id, update.finished.parsed, update.finished.artifacts
                )
                completed += update.job.status == JobStatus.COMPLETED
        self._deps.request_dashboard_flush(
            failure_message="Jobs were checked, but autosave failed."
        )
        if requested:
            self._deps.status_bar().showMessage(f"Checking {plural(requested, 'cluster job')}")
        elif not completed:
            self._deps.status_bar().showMessage(
                f"Checked {plural(len(updates), 'local job')}: no complete output found"
            )

    def _on_dismiss_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return

        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        dismissable = [
            job
            for job in jobs
            if job is not None and job.status == JobStatus.ENDED and job.dismissed_at is None
        ]
        if not dismissable:
            QMessageBox.information(
                self._parent(),
                "No jobs to dismiss",
                "Only failed, stopped, or cancelled jobs can be dismissed from the dashboard.",
            )
            return

        dismissed = 0
        for job in dedupe_jobs(dismissable):
            try:
                self._project_service.dismiss_job(job)
            except ValueError:
                continue
            self._project_service.record_job_dismissed(self._document, job=job)
            dismissed += 1

        if dismissed:
            self._deps.refresh_journal_view()
            self._deps.request_dashboard_flush(
                failure_message="Jobs were dismissed, but autosave failed.",
            )
            self._deps.status_bar().showMessage(
                f"Dismissed {plural(dismissed, 'job')} from the dashboard"
            )

    def _dashboard_action_handler(self, kind: str) -> Callable[[list[UUID]], None]:
        # Looked up by name at call time so a replaced handler is honoured.
        names = {
            "resubmit": "_on_resubmit_from_dashboard",
            "check_now": "_on_check_now_from_dashboard",
            "cancel": "_on_cancel_from_dashboard",
            "inspect": "_on_inspect_from_dashboard",
            "dismiss": "_on_dismiss_from_dashboard",
        }
        return getattr(self, names[kind])

    def _on_cancel_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return
        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        self.cancel_jobs([job for job in jobs if job is not None])

    def _on_inspect_from_dashboard(self, job_ids: list[UUID]) -> None:
        """Open each running job's progress and the failure diagnostics of the rest."""

        if self._document is None:
            return
        jobs = [
            job
            for job in (self._document.jobs.get(job_id) for job_id in job_ids)
            if job is not None
        ]
        for job in jobs:
            if job.activity == JobActivity.RUNNING:
                self.show_geometry_progress(job)
        others = [job.id for job in jobs if job.activity != JobActivity.RUNNING]
        if others or not jobs:
            self._on_inspect_failed_from_dashboard(others)

    def _on_inspect_failed_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return
        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        self.inspect_failed_jobs([job for job in jobs if job is not None])

    def show_geometry_progress(self, job: Job) -> GeometryProgressDialog:
        """Open (or raise) the progress dialog of a running geometry optimisation."""

        dialog = self._geometry_progress_dialogs.get(job.id)
        if dialog is None:
            dialog = GeometryProgressDialog(
                job.label, self._job_progress.progress_for(job.id), self._parent()
            )
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            dialog.refresh_requested.connect(lambda jid=job.id: self._job_progress.refresh([jid]))
            dialog.destroyed.connect(
                lambda _obj=None, jid=job.id: self._geometry_progress_dialogs.pop(jid, None)
            )
            self._geometry_progress_dialogs[job.id] = dialog
        self._job_progress.refresh([job.id])
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def _on_job_progress_changed(self, job_id: UUID, progress: GeometryProgress | None) -> None:
        dialog = self._geometry_progress_dialogs.get(job_id)
        if dialog is not None and progress is not None:
            dialog.set_progress(progress)

    def cancel_and_dismiss(self, cancel_ids: list[UUID], dismiss_ids: list[UUID]) -> None:
        """Cancel and dismiss the jobs Delete or Backspace chose, and report both."""

        if self._document is None:
            return
        active_jobs = [
            job for job_id in cancel_ids if (job := self._document.jobs.get(job_id)) is not None
        ]
        # Each action reports in the status bar; together, report both.
        status_bar = self._deps.status_bar()
        status_bar.clearMessage()
        reports: list[str] = []
        if active_jobs:
            self.cancel_jobs(active_jobs)
            reports.append(status_bar.currentMessage())
            status_bar.clearMessage()
        if dismiss_ids:
            self._on_dismiss_from_dashboard(list(dismiss_ids))
            reports.append(status_bar.currentMessage())
        reports = [report for report in reports if report]
        if len(reports) == 2:
            status_bar.showMessage(f"{reports[0]}; {reports[1][0].lower()}{reports[1][1:]}")
        elif reports:
            status_bar.showMessage(reports[0])

    def prompt_job_file_disposition(
        self,
        *,
        title: str,
        text: str,
        keep_text: str,
        include_apply_all: bool = False,
    ) -> str | None:
        choices = [("delete", "Delete files"), ("archive", "Move files")]
        if include_apply_all:
            choices += [("delete_all", "Delete all"), ("archive_all", "Move all")]
        choices.append(("cancel", "Cancel"))
        answer = workflow_table_utils.confirm_choice(
            self._parent(),
            title,
            text,
            choices,
            default="archive",
            escape="cancel",
            detail=(
                "Choose 'Delete files' to delete those files, 'Move files' to move them into a "
                f"separate 'removed_from_workflow' subfolder, {keep_text}"
            ),
            destructive=("delete", "delete_all"),
            icon=QMessageBox.Icon.Warning,
        )
        return None if answer == "cancel" else answer

    def _farm_counts_for_jobs(self, jobs: Sequence[Job]) -> dict[str, int]:
        """Map job label to configured farm count, for submission-time validation.

        Only farmed jobs are included: an empty mapping means nothing to check.
        Labels (not ids) are the key because the submission dialogs name the
        offending job back to the user.
        """

        counts: dict[str, int] = {}
        for job in jobs:
            enabled, num_farms = self._task_farming_for_job(job)
            if enabled and num_farms:
                counts[job.label] = int(num_farms)
        return counts

    def without_jobs_in_another_jobs_folder(self, jobs: list[Job]) -> list[Job]:
        """Return ``jobs`` less those that would write into a folder another job uses.

        Only a project that already holds two jobs in one folder has any: a new job gets a
        folder of its own when its inputs are written. Nothing saved changes, so the user
        decides what happens to the job.
        """

        def sharers(job: Job) -> set[UUID]:
            controller = owning_controller(self._deps.workflow_controllers(), job)
            return controller.folder_sharers(job) if controller is not None else set()

        refused = self._job_file_cleanup.jobs_in_another_jobs_folder(jobs, together=sharers)
        if not refused:
            return jobs
        project_root = self._document.project_root
        text = (
            f"'{refused[0].label}' was not started because another job uses its folder."
            if len(refused) == 1
            else f"{plural(len(refused), 'job')} were not started because other jobs "
            "use their folders."
        )
        show_detailed_warning(
            self._parent(),
            "Jobs not started",
            text,
            informative_text=(
                "No files were changed. Remove the row and add it again to give the job a folder "
                "of its own."
            ),
            details="\n".join(
                f"{job.label}: {job_directory_label(project_root, directory)}"
                for job in refused
                if (directory := self._deps.job_output_directory(job)) is not None
            ),
        )
        refused_ids = {job.id for job in refused}
        return [job for job in jobs if job.id not in refused_ids]

    def submit_jobs_locally(
        self,
        jobs: list[Job],
        *,
        eligible_statuses: Collection[JobStatus] | None = None,
        no_eligible_message: str = "Selected jobs must be ready for local submission.",
        success_message: Callable[[int], str] | None = None,
        fresh_start_job_ids: Collection[UUID] = (),
        run_again: RunAgainPlan | None = None,
        on_accepted: OnAccepted | None = None,
    ) -> None:
        """Submit ``jobs`` to the local runner after the local submission dialog.

        A workflow row's submission passes ``on_accepted`` (and ``run_again``
        when it runs ended jobs again). Once the dialog is accepted that call
        creates the planned jobs and writes the inputs, then clears the
        start-again jobs' previous output, so a failed write loses none, and
        applies the run-again plan; ``None`` from it starts nothing. The number
        of cores is remembered only then. Without ``on_accepted`` (the Job
        Dashboard's Submit, continuing a relaxation) the jobs exist already;
        those in ``fresh_start_job_ids`` have their previous output cleared
        once the dialog is accepted, so cancelling never deletes anything.
        """

        if run_again is not None and on_accepted is None:
            raise ValueError("a run-again plan is applied by the submission's on_accepted step")
        if self._document is None or not jobs:
            return

        allowed_statuses = set(eligible_statuses or {JobStatus.DEFINED})
        running_again = run_again.job_ids if run_again is not None else frozenset()
        eligible_jobs = [
            job
            for job in dedupe_jobs(jobs)
            if job.status in allowed_statuses
            or (job.status in RUN_AGAIN_STATES and job.id in running_again)
        ]
        if not eligible_jobs:
            QMessageBox.information(
                self._parent(),
                "No eligible jobs",
                no_eligible_message,
            )
            return
        eligible_jobs = self.without_jobs_in_another_jobs_folder(eligible_jobs)
        if not eligible_jobs:
            return

        if not self._castep_executable_ready():
            return
        if not self._deps.keyword_catalog_guard().confirm_submission():
            return

        setup = load_castep_setup(self._settings)
        dialog = LocalSubmissionDialog(
            job_labels=[job.label for job in eligible_jobs],
            mpirun_configured=bool(setup.mpirun_path),
            farm_counts=self._farm_counts_for_jobs(eligible_jobs),
            stage_title=job_stage_title(job.workflow_stage for job in eligible_jobs),
            initial_cores=remembered_local_cores(self._document.ui_state),
            parent=self._parent(),
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        config = dialog.config()
        if setup.mpirun_path and on_accepted is None:
            # Without mpirun the dialog pins one core, which is not a choice to keep.
            remember_local_cores(self._document.ui_state, config.num_cores)

        start_again = set(fresh_start_job_ids)
        if run_again is not None:
            start_again |= run_again.start_again_job_ids
        fresh_jobs = [job for job in eligible_jobs if job.id in start_again and on_accepted is None]
        cleaned: list[Job] = []
        if fresh_jobs:
            cleaned, cleanup_errors = self._clear_outputs_for_fresh_start(fresh_jobs)
            if cleanup_errors:
                not_cleaned = {job.id for job in fresh_jobs} - {job.id for job in cleaned}
                eligible_jobs = [job for job in eligible_jobs if job.id not in not_cleaned]
                self._warn_previous_output_not_cleared(cleanup_errors)
                if not eligible_jobs:
                    return
        if on_accepted is not None:
            accepted = on_accepted(
                eligible_jobs,
                {job.id for job in eligible_jobs if job.id in start_again},
                self._clear_outputs_for_fresh_start,
            )
            if accepted is None:
                return
            if accepted.errors:
                self._warn_previous_output_not_cleared(accepted.errors)
            eligible_jobs = accepted.jobs
            if not eligible_jobs:
                return
            if setup.mpirun_path:
                remember_local_cores(self._document.ui_state, config.num_cores)
            self._deps.refresh_views_after_job_reset()
            allowed_statuses.add(JobStatus.DEFINED)

        outcome = self._job_execution.submit_local_jobs(
            eligible_jobs,
            num_cores=config.num_cores,
            max_concurrent=config.max_concurrent if config.parallel else 1,
            castep_executable=setup.executable_path,
            mpirun_path=setup.mpirun_path,
            eligible_statuses=allowed_statuses,
            retrieval_for_job=self.retrieval_flags_for_job,
        )

        self._deps.refresh_jobs_table()
        self._deps.convergence_controller().refresh_view()
        self._deps.kpoint_convergence_controller().refresh_view()
        self._deps.refresh_hyperfine_coupling_view()
        self._deps.refresh_quantum_corrections_view()
        if outcome.journal_changed:
            self._deps.refresh_journal_view()

        if outcome.missing_paths:
            QMessageBox.warning(
                self._parent(),
                "Missing job directories",
                "Could not locate generated input directories for:\n"
                + "\n".join(f"- {label}" for label in outcome.missing_paths),
            )

        if outcome.failed_messages:
            show_detailed_warning(
                self._parent(),
                "Local submission failed",
                "Some jobs failed to submit:\n" + "\n".join(outcome.failed_messages),
                informative_text="Open Details to inspect the full submission failures.",
            )

        if outcome.submitted_count:
            self._deps.persist_document_ui_state(
                failure_message="Submitted local jobs, but autosave failed.",
            )
            message = (
                success_message(outcome.submitted_count)
                if success_message is not None
                else f"Submitted {plural(outcome.submitted_count, 'local job')}"
            )
            self._deps.status_bar().showMessage(self._deps.with_kpoint_status_note(message))

    def _warn_previous_output_not_cleared(self, errors: Sequence[str]) -> None:
        show_detailed_warning(
            self._parent(),
            "Previous output not cleared",
            "These jobs were not resubmitted because their previous output, "
            "including any checkpoint, could not be cleared:\n" + "\n".join(errors),
            informative_text="Close any program using these files, then resubmit.",
        )

    def submit_jobs_on_cluster(
        self,
        jobs: list[Job],
        *,
        default_cluster_name: str | None = None,
        eligible_statuses: Collection[JobStatus] | None = None,
        no_eligible_message: str = "Selected jobs must be ready for cluster submission.",
        success_message: Callable[[int, str], str] | None = None,
        fresh_start_job_ids: Collection[UUID] = (),
        run_again: RunAgainPlan | None = None,
        on_accepted: OnAccepted | None = None,
    ) -> None:
        """Submit ``jobs`` to a cluster chosen in the cluster submission dialog.

        The decisions live in ``services.cluster_submission_planner``; this
        method shows the dialog, connects, asks the overwrite prompts the plan
        lists and reports the outcome. Jobs in ``fresh_start_job_ids``, and the
        start-again jobs of ``run_again``, start again: on their own cluster,
        their remote directory is deleted (only once their previous scheduler
        job has finished) and their inputs uploaded afresh, without the
        overwrite prompt, and their local output is cleared (see
        :meth:`_clear_outputs_for_fresh_start`). The continuing jobs of
        ``run_again`` are submitted into their own remote directory.

        A workflow row's submission passes ``on_accepted`` (and ``run_again``
        when it runs ended jobs again). That call runs after the planner's last
        refusal and before the jobs are prepared, with the planned items' jobs
        only, so a job skipped at a prompt is never created. It writes the
        inputs first and then clears the start-again jobs' previous output (the
        planner leaves that to it), so a failed write loses none, then returns
        the run-again jobs to ``defined``; ``None`` from it starts nothing. A
        cancelled dialog, a failed connection or a declined prompt leaves every
        job as it was. The dialog's settings are remembered only then; without
        ``on_accepted`` (the Job Dashboard's Submit, continuing a relaxation)
        they are remembered once the dialog is accepted.
        """

        if run_again is not None and on_accepted is None:
            raise ValueError("a run-again plan is applied by the submission's on_accepted step")
        if self._document is None or not jobs:
            return

        allowed_statuses = set(eligible_statuses or {JobStatus.DEFINED})
        running_again = run_again.job_ids if run_again is not None else frozenset()
        eligible_jobs = [
            job
            for job in dedupe_jobs(jobs)
            if job.status in allowed_statuses
            or (job.status in RUN_AGAIN_STATES and job.id in running_again)
        ]
        if not eligible_jobs:
            QMessageBox.information(self._parent(), "No eligible jobs", no_eligible_message)
            return
        eligible_jobs = self.without_jobs_in_another_jobs_folder(eligible_jobs)
        if not eligible_jobs:
            return

        chosen = self._choose_cluster_submission(
            eligible_jobs, default_cluster_name, remember=on_accepted is None
        )
        if chosen is None:
            return
        submission, selected_cluster, dialog_credentials = chosen

        screened = self._screen_cluster_submission_jobs(eligible_jobs, selected_cluster)
        if screened is None:
            return
        eligible_jobs, seed_name_by_job = screened

        if not self._connect_for_cluster_submission(selected_cluster, dialog_credentials):
            return

        start_again = set(fresh_start_job_ids)
        if run_again is not None:
            start_again |= run_again.start_again_job_ids
        eligible_jobs, refused_restarts = self._without_previous_runs_alive(
            eligible_jobs, start_again, selected_cluster.name
        )

        # The existence check runs on the worker thread; every prompt is answered
        # here, before the batch starts, so none interrupts it.
        plan = plan_cluster_submission(
            eligible_jobs,
            cluster=selected_cluster,
            submission=submission,
            seed_name_by_job=seed_name_by_job,
            remote_job_directory=self._cluster_runner.remote_job_directory,
            output_directory=self._deps.job_output_directory,
            retrieval_flags=self.retrieval_flags_for_job,
            check_remote_directories=self._await_remote_dir_check,
            ask_overwrite=self._prompt_remote_directory_overwrite,
            confirm_bulk_delete=self._confirm_bulk_remote_delete,
            # On the deferred path the accept step clears them, after the new inputs.
            clear_fresh_outputs=(
                self._clear_outputs_for_fresh_start
                if on_accepted is None
                else lambda fresh: (list(fresh), [])
            ),
            fresh_start_job_ids=start_again,
            continue_job_ids=run_again.continue_job_ids if run_again is not None else (),
        )
        failed_submit = refused_restarts + list(plan.failed)
        if on_accepted is not None:
            going = [item.submission.job for item in plan.items]
            if going:
                accepted = on_accepted(
                    going,
                    {job.id for job in going if job.id in start_again},
                    self._clear_outputs_for_fresh_start,
                )
                if accepted is None:
                    # Nothing is submitted, but what the plan refused is still reported.
                    self._report_cluster_submission(
                        replace(plan, items=[]),
                        submitted_jobs=[],
                        failed_submit=failed_submit,
                        cluster_name=selected_cluster.name,
                        success_message=success_message,
                    )
                    return
                failed_submit.extend(accepted.errors)
                kept = {job.id for job in accepted.jobs}
                plan = replace(
                    plan, items=[item for item in plan.items if item.submission.job.id in kept]
                )
                if plan.items:
                    self._remember_cluster_submission(eligible_jobs, submission)
                self._deps.refresh_views_after_job_reset()
        submit_items, prepared_by_job = plan.prepare_batch(
            self._job_execution.prepare_cluster_submission
        )

        submitted_jobs: list[Job] = []
        if submit_items:
            submitted_jobs, batch_failed = self._run_cluster_submit_batch(
                submit_items, prepared_by_job
            )
            failed_submit.extend(batch_failed)

        self._deps.refresh_jobs_table()
        self._deps.convergence_controller().refresh_view()
        self._deps.kpoint_convergence_controller().refresh_view()
        self._deps.refresh_hyperfine_coupling_view()
        self._deps.refresh_quantum_corrections_view()
        self._deps.refresh_energy_barriers_view()

        self._report_cluster_submission(
            plan,
            submitted_jobs=submitted_jobs,
            failed_submit=failed_submit,
            cluster_name=selected_cluster.name,
            success_message=success_message,
        )

    def _castep_executable_ready(self) -> bool:
        """Return whether a local CASTEP executable is configured, offering Setup > CASTEP."""

        return self._ready_or_offer_setup(
            lambda: bool(load_castep_setup(self._settings).executable_path),
            title="CASTEP executable missing",
            text="Configure a CASTEP executable in Setup > CASTEP before submitting jobs.",
            setup_label="Open Setup > CASTEP…",
            open_setup=self._deps.open_castep_setup_dialog,
        )

    def _clusters_configured(self) -> bool:
        """Return whether at least one cluster is configured, offering Setup > HPC Clusters."""

        return self._ready_or_offer_setup(
            lambda: bool(load_cluster_configs(self._settings)),
            title="No configured clusters",
            text="Configure at least one cluster in Setup > HPC Clusters before submitting.",
            setup_label="Open Setup > HPC Clusters…",
            open_setup=self._deps.open_cluster_setup_dialog,
        )

    def _ready_or_offer_setup(
        self,
        is_ready: Callable[[], bool],
        *,
        title: str,
        text: str,
        setup_label: str,
        open_setup: Callable[[], None],
    ) -> bool:
        """Return ``is_ready()``; when false, say why and offer the setup dialog.

        The answer is true when the user opens the setup dialog and fixes it
        there. Cancel is the default.
        """

        if is_ready():
            return True
        choice = workflow_table_utils.confirm_choice(
            self._parent(),
            title,
            text,
            [("setup", setup_label), ("cancel", "Cancel")],
            default="cancel",
            escape="cancel",
            icon=QMessageBox.Icon.Warning,
        )
        if choice != "setup":
            return False
        open_setup()
        return is_ready()

    def _remember_cluster_submission(
        self, jobs: list[Job], submission: ClusterSubmissionConfig
    ) -> None:
        """Remember ``submission`` as the last one of ``jobs``' workflow, for the next dialog."""

        if self._document is not None:
            remember_cluster_submission(
                self._document.ui_state,
                submission,
                workflow_type=cluster_workflow_directory_fragment(jobs),
            )

    def _choose_cluster_submission(
        self,
        eligible_jobs: list[Job],
        default_cluster_name: str | None,
        *,
        remember: bool = True,
    ) -> tuple[ClusterSubmissionConfig, ClusterConfig, SSHCredentials | None] | None:
        """Show the cluster submission dialog and resolve the chosen cluster.

        Returns the accepted submission settings, the selected cluster (with the
        dialog's username applied) and the dialog's credentials, or ``None``
        after telling the user why when no cluster is configured, the keyword
        catalog guard declines, the dialog is cancelled or the chosen cluster
        has since been removed. With ``remember`` the accepted settings are
        kept for the next dialog of this workflow at once; otherwise the caller
        does that once the submission is accepted.
        """

        if not self._clusters_configured():
            return None
        if not self._deps.keyword_catalog_guard().confirm_submission():
            return None
        cluster_configs = load_cluster_configs(self._settings)

        workflow_type = cluster_workflow_directory_fragment(eligible_jobs)
        last_submission = (
            remembered_cluster_submission(self._document.ui_state, workflow_type=workflow_type)
            if self._document is not None
            else None
        )
        dialog = ClusterSubmissionDialog(
            job_labels=[job.label for job in eligible_jobs],
            cluster_configs=cluster_configs,
            default_cluster_name=default_cluster_name
            or (last_submission.cluster_name if last_submission is not None else None),
            initial_config=last_submission,
            project_name=self._document.project.name,
            workflow_type=workflow_type,
            connected_cluster_name=self._cluster_runner.connected_cluster_name,
            connected_username=self._cluster_runner.connected_username,
            farm_counts=self._farm_counts_for_jobs(eligible_jobs),
            edit_cluster=self._deps.edit_cluster_during_submission,
            parent=self._parent(),
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return None

        submission = dialog.config()
        if remember:
            self._remember_cluster_submission(eligible_jobs, submission)
        if hasattr(dialog, "credentials"):
            dialog_credentials = dialog.credentials()
        else:
            dialog_credentials = SSHCredentials(target_secret=dialog.password().strip() or None)
        selected_cluster = load_cluster_config(self._settings, submission.cluster_name)
        if selected_cluster is None:
            QMessageBox.warning(
                self._parent(), "Cluster not found", "Selected cluster is no longer configured."
            )
            return None
        runtime_username = submission.username.strip()
        if runtime_username and runtime_username != selected_cluster.username:
            selected_cluster = replace(selected_cluster, username=runtime_username)
        return submission, selected_cluster, dialog_credentials

    def _screen_cluster_submission_jobs(
        self, eligible_jobs: list[Job], cluster: ClusterConfig
    ) -> tuple[list[Job], dict[UUID, str]] | None:
        """Map the jobs going to ``cluster`` to their seed names.

        Returns the jobs with their seed names, or ``None`` when a seed name is
        unsafe for remote paths (which is reported and aborts the whole
        submission).
        """

        seed_name_by_job = {job.id: self._deps.job_seed_name(job) for job in eligible_jobs}
        invalid_labels = self._cluster_runner.invalid_remote_job_labels(
            list(seed_name_by_job.values())
        )
        if invalid_labels:
            show_detailed_warning(
                self._parent(),
                "Unsafe job labels",
                unsafe_job_labels_message(invalid_labels),
                informative_text="Remote paths and scheduler commands use these labels directly.",
            )
            return None
        return eligible_jobs, seed_name_by_job

    def _report_cluster_submission(
        self,
        plan: ClusterSubmissionPlan,
        *,
        submitted_jobs: list[Job],
        failed_submit: list[str],
        cluster_name: str,
        success_message: Callable[[int, str], str] | None,
    ) -> None:
        """Report a dispatched cluster submission and record what was submitted.

        Warns about the plan's missing input directories and every failed
        submission, lists the jobs skipped because their remote directory
        already exists, then records the submitted jobs in the journal, autosaves
        and shows the status-bar summary (``success_message`` when given).
        """

        if plan.missing:
            QMessageBox.warning(
                self._parent(),
                "Missing job directories",
                "Could not locate generated input directories for:\n"
                + "\n".join(f"- {label}" for label in plan.missing),
            )
        if failed_submit:
            show_detailed_warning(
                self._parent(),
                "Cluster submission failed",
                "Some jobs failed to submit:\n" + "\n".join(failed_submit),
                informative_text="Open Details to inspect the full submission failures.",
            )
        if plan.skipped:
            QMessageBox.information(
                self._parent(),
                "Cluster submission skipped",
                "Some jobs were not submitted because their remote directories already exist:\n"
                + "\n".join(plan.skipped),
            )
        if submitted_jobs:
            if self._job_execution.record_submission(submitted_jobs, cluster_name):
                self._deps.refresh_journal_view()
            self._deps.persist_document_ui_state(
                failure_message="Submitted cluster jobs, but autosave failed."
            )
            submitted = len(submitted_jobs)
            message = (
                success_message(submitted, cluster_name)
                if success_message is not None
                else f"Submitted {plural(submitted, 'cluster job')} on {cluster_name}"
            )
            self._deps.status_bar().showMessage(self._deps.with_kpoint_status_note(message))

    def _connect_for_cluster_submission(
        self, cluster: ClusterConfig, dialog_credentials: SSHCredentials | None
    ) -> bool:
        """Connect to ``cluster`` for a submission, reusing a matching open connection.

        Returns ``False`` after telling the user why when there are no
        credentials or the connection fails.
        """

        reusing_connection = can_reuse_cluster_connection(
            cluster,
            is_connected=self._cluster_runner.is_connected,
            connected_cluster_name=self._cluster_runner.connected_cluster_name,
            connected_username=self._cluster_runner.connected_username,
        )
        credentials = SSHCredentials() if reusing_connection else dialog_credentials
        if credentials is None:
            QMessageBox.warning(
                self._parent(),
                "Missing credentials",
                "Credentials are required for cluster submission.",
            )
            return False
        try:
            self._connect_to_cluster_with_trust(cluster, credentials=credentials)
        except Exception as exc:  # noqa: BLE001
            show_detailed_warning(
                self._parent(),
                "Cluster connection failed",
                format_cluster_connection_error(cluster, exc),
                informative_text="Open Details for the full connection error and recovery guidance.",
            )
            return False
        if not reusing_connection:
            QMessageBox.information(
                self._parent(),
                "Cluster connection successful",
                f"Connected to {cluster.name} as {cluster.username}.",
            )
        return True

    def _await_cluster_signal(
        self,
        *,
        start: Callable[[], None],
        signal: Signal,
        title: str,
        label: str,
    ) -> object:
        """Run a worker-thread cluster op behind a modal busy dialog.

        Calls ``start()`` (which kicks off async work on the worker thread), then
        runs a nested event loop until ``signal`` fires, returning its payload. The
        window keeps repainting while the worker performs the blocking SSH work.

        A watchdog aborts the wait after three minutes and returns a
        ``TimeoutError``: the dialog has no cancel button, so without it a
        worker that never emits (or a request stuck behind a long queue-poll
        batch) would trap the user in the modal dialog forever.
        """

        captured: dict[str, object] = {}
        loop = QEventLoop()

        def on_signal(payload: object = None) -> None:
            captured["payload"] = payload
            loop.quit()

        watchdog = QTimer(self)
        watchdog.setSingleShot(True)
        watchdog.setInterval(180_000)
        watchdog.timeout.connect(loop.quit)

        progress = QProgressDialog(label, None, 0, 0, self._parent())
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        signal.connect(on_signal)
        progress.show()
        try:
            start()
            watchdog.start()
            loop.exec()
        finally:
            watchdog.stop()
            watchdog.deleteLater()
            signal.disconnect(on_signal)
            progress.close()
        if "payload" not in captured:
            return TimeoutError(
                f"{title} did not complete within 3 minutes; the cluster worker may be busy or stuck."
            )
        return captured.get("payload")

    def _await_remote_dir_check(self, targets: list[tuple[UUID, str]]) -> dict[str, object]:
        payload = self._await_cluster_signal(
            start=lambda: self._cluster_runner.check_remote_dirs(targets),
            signal=self._cluster_runner.remote_dirs_checked,
            title="Checking remote directories",
            label="Checking existing remote job directories…",
        )
        if isinstance(payload, TimeoutError):
            # Fail every checked job rather than silently submitting over
            # directories whose state is unknown.
            return {
                "existing": [],
                "errors": {job_id: str(payload) for job_id, _target in targets},
            }
        return payload if isinstance(payload, dict) else {"existing": [], "errors": {}}

    def _await_previous_runs_check(self, targets: list[tuple[UUID, str]]) -> dict[str, object]:
        """Ask the scheduler whether each ``(job_id, remote_job_id)`` previous run has finished.

        The nested-event-loop wait of :meth:`_await_remote_dir_check`, over
        :meth:`ClusterController.check_previous_runs`. A timeout makes every
        job an error, so nothing is deleted on an unknown answer.
        """

        payload = self._await_cluster_signal(
            start=lambda: self._cluster_runner.check_previous_runs(targets),
            signal=self._cluster_runner.previous_runs_checked,
            title="Checking previous runs",
            label="Checking the previous runs are no longer queued or running…",
        )
        if isinstance(payload, TimeoutError):
            return {"active": [], "errors": {job_id: str(payload) for job_id, _id in targets}}
        return payload if isinstance(payload, dict) else {"active": [], "errors": {}}

    def _without_previous_runs_alive(
        self, jobs: list[Job], start_again: Collection[UUID], cluster_name: str
    ) -> tuple[list[Job], list[str]]:
        """Return ``jobs`` less the start-again jobs whose previous run may still be alive.

        Starting a job again on its own cluster deletes its remote folder, so
        before anything is cleared the scheduler is asked about its previous
        scheduler job. A job it still lists as queued or running, or that it
        cannot be asked about, is left out, with a line saying why: its folders
        and its record stay as they were. (An ended job can still run on the
        cluster: one cancelled in the project while disconnected, or one
        MuFinder lost track of.)
        """

        targets = [
            (job.id, (job.remote_job_id or "").strip())
            for job in jobs
            if job.id in start_again
            and (job.remote_directory or "").strip()
            and (job.remote_job_id or "").strip()
            and (job.execution_target or "").strip().casefold() == cluster_name.casefold()
        ]
        if not targets:
            return jobs, []
        result = self._await_previous_runs_check(targets)
        active = set(result.get("active", []))
        errors: Mapping[UUID, str] = result.get("errors", {})
        remote_ids = dict(targets)
        failed: list[str] = []
        kept: list[Job] = []
        for job in jobs:
            if job.id in active:
                failed.append(
                    f"- {job.label}: not started again, because its previous run (scheduler job "
                    f"{remote_ids[job.id]}) is still queued or running; its folders were kept"
                )
            elif job.id in errors:
                failed.append(
                    f"- {job.label}: not started again, because the scheduler could not say "
                    f"whether its previous run (scheduler job {remote_ids[job.id]}) has "
                    f"finished ({errors[job.id]}); its folders were kept"
                )
            else:
                kept.append(job)
        return kept, failed

    def _await_remote_file_check(self, targets: list[tuple[UUID, str]]) -> dict[str, object]:
        """Twin of :meth:`_await_remote_dir_check` for a single-file existence probe.

        Feeds the Continue-relaxation checkpoint probe
        (:meth:`_continue_relaxation_checkpoint_probe`): the same nested-event-
        loop wait ``submit_jobs_on_cluster`` already uses mid-flow before
        showing a dialog, so the probe can complete synchronously before the
        Continue dialog is constructed (WP5 requires the probe run before
        dialog construction, never inside the dialog's own ``__init__``).
        """

        payload = self._await_cluster_signal(
            start=lambda: self._cluster_runner.check_remote_files(targets),
            signal=self._cluster_runner.remote_files_checked,
            title="Checking remote checkpoint",
            label="Checking for an existing CASTEP checkpoint…",
        )
        if isinstance(payload, TimeoutError):
            return {
                "existing": [],
                "errors": {job_id: str(payload) for job_id, _target in targets},
            }
        return payload if isinstance(payload, dict) else {"existing": [], "errors": {}}

    def await_cluster_connect(
        self, config: ClusterConfig, *, credentials: SSHCredentials
    ) -> BaseException | None:
        """Connect to a cluster off the GUI thread, returning the error (or None)."""

        payload = self._await_cluster_signal(
            start=lambda: self._cluster_runner.connect_async(config, credentials=credentials),
            signal=self._cluster_runner.connect_finished,
            title="Connecting to cluster",
            label=f"Connecting to {config.name}…",
        )
        return payload if isinstance(payload, BaseException) else None

    def _run_cluster_submit_batch(
        self,
        submit_items: list[dict[str, object]],
        prepared_by_job: dict[UUID, PreparedClusterSubmission],
    ) -> tuple[list[Job], list[str]]:
        """Submit a batch on the worker thread with a cancellable progress dialog.

        Returns the successfully submitted jobs and per-job failure messages. The
        GUI stays responsive via a nested event loop; document bookkeeping for each
        result runs here on the GUI thread.
        """

        submitted_jobs: list[Job] = []
        failed: list[str] = []
        results_seen: set[UUID] = set()
        total = len(submit_items)
        loop = QEventLoop()
        timed_out = False
        # Set once the batch is handed to the worker; signals of any other batch
        # (one given up on earlier, still draining on the worker) are ignored.
        batch_id: int | None = None

        progress = QProgressDialog("Submitting cluster jobs…", "Cancel", 0, total, self._parent())
        progress.setWindowTitle("Submitting to cluster")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Inactivity watchdog: any per-job progress restarts it, so long
        # batches are fine, but a worker that stops responding entirely cannot
        # trap the user in the nested event loop forever.
        watchdog = QTimer(self)
        watchdog.setSingleShot(True)
        watchdog.setInterval(180_000)

        def on_watchdog_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            # The unseen jobs are cancelled below, so the worker must not go on
            # submitting them, and one it submits after all is cancelled on
            # the scheduler.
            if batch_id is not None:
                self._cluster_runner.abandon_batch(batch_id)
            loop.quit()

        watchdog.timeout.connect(on_watchdog_timeout)

        def on_progress(batch: int, index: int, count: int, label: str) -> None:
            if batch != batch_id:
                return
            progress.setLabelText(f"Submitting {label} ({index}/{count})")
            progress.setValue(index - 1)
            watchdog.start()

        def on_one(batch: int, job_id: object, remote_job_id: object, error: str) -> None:
            if batch != batch_id or not isinstance(job_id, UUID):
                return
            results_seen.add(job_id)
            watchdog.start()
            prepared = prepared_by_job.get(job_id)
            if prepared is None:
                return
            if remote_job_id:
                outcome = self._job_execution.finalize_cluster_submission(
                    prepared, str(remote_job_id)
                )
            else:
                outcome = self._job_execution.fail_cluster_submission(prepared, error)
            if outcome.submitted_job is not None:
                submitted_jobs.append(outcome.submitted_job)
            if outcome.failed_message is not None:
                failed.append(outcome.failed_message)
            if outcome.journal_changed:
                self._deps.refresh_journal_view()
            progress.setValue(progress.value() + 1)
            # The jobs table is refreshed once by the caller after the batch;
            # refreshing per job here made an N-job batch do N full rebuilds.

        def on_finished(batch: int) -> None:
            if batch == batch_id:
                loop.quit()

        def on_cancel() -> None:
            if batch_id is not None:
                self._cluster_runner.cancel_batch(batch_id)

        self._cluster_runner.submit_batch_progress.connect(on_progress)
        self._cluster_runner.job_submit_finished.connect(on_one)
        self._cluster_runner.submit_batch_finished.connect(on_finished)
        progress.canceled.connect(on_cancel)
        progress.show()
        try:
            batch_id = self._cluster_runner.submit_batch(submit_items)
            watchdog.start()
            loop.exec()
        finally:
            watchdog.stop()
            watchdog.deleteLater()
            self._cluster_runner.submit_batch_progress.disconnect(on_progress)
            self._cluster_runner.job_submit_finished.disconnect(on_one)
            self._cluster_runner.submit_batch_finished.disconnect(on_finished)
            progress.close()

        if timed_out:
            pending = total - len(results_seen)
            failed.append(
                f"- Cluster submission stopped responding after 3 minutes; "
                f"{plural(pending, 'pending job was', 'pending jobs were')} cancelled."
            )

        # Jobs prepared but never submitted (cancelled mid-batch or timed out)
        # are cancelled so they do not linger in the ``submitting`` state.
        for job_id, prepared in prepared_by_job.items():
            if job_id not in results_seen:
                self._job_execution.cancel_job(prepared.job.id)
        return submitted_jobs, failed

    def _prompt_remote_directory_overwrite(self, job_label: str, remote_dir: str) -> str:
        return workflow_table_utils.confirm_choice(
            self._parent(),
            "Remote directory exists",
            f"Remote directory already exists for {job_label}.",
            [
                ("yes", "Delete directory"),
                ("yes_to_all", "Delete all"),
                ("no", "Keep directory"),
                ("no_to_all", "Keep all"),
            ],
            default="no",
            escape="no",
            detail=f"Directory: {remote_dir}\n\nDelete it before submission?",
            destructive=("yes", "yes_to_all"),
            icon=QMessageBox.Icon.Warning,
        )

    def _confirm_bulk_remote_delete(self, remote_dirs: list[str]) -> bool:
        if not remote_dirs:
            return True

        unique_dirs = sorted(set(remote_dirs))
        return workflow_table_utils.confirm_destructive(
            self._parent(),
            "Confirm remote deletion",
            "Delete all existing remote job directories?",
            "The following remote directories will be deleted before submission.",
            accept_label=f"Delete {plural(len(unique_dirs), 'directory', 'directories')}",
            detailed_text="\n".join(f"- {path}" for path in unique_dirs),
        )

    def _cluster_workflow_directory_fragment(self, jobs: list[Job]) -> str:
        return cluster_workflow_directory_fragment(jobs)

    def cancel_jobs(self, jobs: list[Job]) -> None:
        if self._document is None:
            return

        cancelled = 0
        uncontacted: dict[str, str] = {}
        project_only = 0
        for job in jobs:
            if job.status != JobStatus.ACTIVE:
                continue
            if job.execution_target == "local":
                self._local_runner.cancel(job.id)
            elif not self._job_execution.can_cancel_on_cluster(job):
                # The scheduler can only be asked to cancel a job with a
                # scheduler job ID on the connected cluster. Any other job is
                # cancelled in the project, or the row could never leave its
                # active state. The job records that its scheduler was not
                # told, so a check can find it if it still runs there.
                self._record_cancel(job.id, scheduler_told=False)
                target = job.execution_target.strip()
                uncontacted.setdefault(target.casefold(), target)
                project_only += 1
                continue
            else:
                try:
                    self._cluster_runner.cancel(job.id, (job.remote_job_id or "").strip())
                except Exception as exc:  # noqa: BLE001
                    show_detailed_warning(
                        self._parent(),
                        "Cluster cancellation failed",
                        str(exc),
                        informative_text="Open Details to inspect the full cancellation error.",
                    )
                    continue
            cancelled += 1

        requested = f"Cancellation requested for {plural(cancelled, 'job')}"
        if project_only:
            where = " or ".join(uncontacted.values())
            in_project = f"ancelled {plural(project_only, 'job')} in the project"
            not_contacted = f"{where} was not contacted for jobs it could not be asked about"
            self._deps.status_bar().showMessage(
                f"{requested}; c{in_project} ({not_contacted})"
                if cancelled
                else f"C{in_project}; {not_contacted}"
            )
        elif cancelled:
            self._deps.status_bar().showMessage(requested)

    def inspect_failed_jobs(self, jobs: list[Job]) -> None:
        if self._document is None:
            return

        failed_jobs = [
            job
            for job in jobs
            if job.status == JobStatus.ENDED and job.end_reason != EndReason.CANCELLED
        ]
        if not failed_jobs:
            QMessageBox.information(
                self._parent(),
                "No failed jobs",
                "Select one or more failed or stopped jobs to inspect details.",
            )
            return

        reports = [
            failed_job_report(
                job,
                self._job_failure_diagnosis(job),
                self._deps.job_output_directory(job),
                self._deps.job_seed_name(job),
            )
            for job in failed_jobs
        ]
        continue_relaxation = (
            self.continue_relaxation_for_job(failed_jobs[0]) if len(failed_jobs) == 1 else None
        )
        dialog = FailedJobDialog(
            reports,
            # Resubmit runs the jobs' rows, so jobs without one are not offered it.
            can_resubmit=any(self.can_resubmit(job) for job in failed_jobs),
            can_continue=continue_relaxation is not None,
            open_folder=open_containing_folder,
            parent=self._parent(),
        )
        dialog.exec()
        action = dialog.action()
        if action == FailedJobAction.RESUBMIT:
            self._on_resubmit_from_dashboard([job.id for job in failed_jobs])
        elif action == FailedJobAction.CONTINUE_RELAXATION and continue_relaxation is not None:
            continue_relaxation()

    def continue_relaxation_for_job(self, job: Job) -> Callable[[], None] | None:
        """Return the row menu's ``Continue relaxation…`` for ``job``'s row.

        ``None`` when ``job`` is not a relaxation, no row of its page's active
        run holds it, or that row cannot be continued (the row menu's check).
        The row is found through the active run's job bindings, as the row
        menu resolves its row's job.
        """

        if job.workflow_stage == WorkflowStage.PRISTINE_RELAXATION:
            controller = self._deps.pristine_relaxation_controller()
            view = self._deps.pristine_relaxation_view()
            handler = self.on_continue_relaxation_requested_from_pristine_relaxation
        elif job.workflow_stage == WorkflowStage.MUON_RELAXATION:
            controller = self._deps.muon_relaxation_controller()
            view = self._deps.muon_relaxation_view()
            handler = self.on_continue_relaxation_requested_from_muon_relaxation
        else:
            return None
        key = self._relaxation_row_key_for_job(job)
        if key is None or not controller.row_is_continuable(key):
            return None
        if view.continue_relaxation_dialog_seed(key) is None:
            return None
        return partial(handler, key)

    def _relaxation_row_key_for_job(self, job: Job) -> str | None:
        """Return the key of the active relaxation run's row bound to ``job``."""

        if job.workflow_stage == WorkflowStage.PRISTINE_RELAXATION:
            controller = self._deps.pristine_relaxation_controller()
            definition = controller.active_definition()
            if definition is None:
                return None
            controller.sync_mappings(definition)
            config = deserialize_pristine_relaxation_config(
                definition.parameters.get("configuration", {})
            )
            rows = config.rows if config is not None else ()
            return next(
                (
                    row.label
                    for row in rows
                    if (bound := controller.job_for_key(definition, row.label)) is not None
                    and bound.id == job.id
                ),
                None,
            )
        controller = self._deps.muon_relaxation_controller()
        definition = controller.active_definition()
        if definition is None:
            return None
        controller.sync_mappings(definition)
        config = deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        rows = config.rows if config is not None else ()
        return next(
            (
                self._deps.muon_relaxation_row_key(row)
                for row in rows
                if (bound := controller.job_for_label(definition, row.label, row.charge_state))
                is not None
                and bound.id == job.id
            ),
            None,
        )

    def prepare_jobs_for_resubmission(self, jobs: Sequence[Job]) -> RunAgainPlan | None:
        """Decide how ended jobs run again from their workflow row; change nothing yet.

        Bound into every controller's
        :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`
        and called from the shared submit cycle before dispatch, from a row's
        Run or Run All and from the Job Dashboard's ``Resubmit…``, which runs
        the job's row.

        When a job whose inputs resume from a checkpoint (a relaxation or a
        phonon job,
        :func:`~mufinder.services.job_monitoring.job_resumes_from_checkpoint`)
        has a checkpoint, or its checkpoint could not be checked, one question
        asks whether to continue from the previous run's files or start again.
        Every other job starts again. When a job about to start again may still
        have output on its cluster that was never retrieved, the question warns
        so and defaults to ``Cancel``; with nothing to continue from it is a
        start-again-or-cancel confirmation. With neither, nothing is asked.

        A completed job among ``jobs`` is one whose row was edited (a Quantum
        Corrections row the user already confirmed replacing): it starts again
        without a question, and its result goes once the plan is applied.

        Returns the :class:`RunAgainPlan` the accept step applies once the
        submission is accepted
        (:meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.accept_submission`), or ``None``
        when the user cancels. Other jobs, and imported jobs, are never in the
        plan.
        """

        if self._document is None:
            return None

        replaced = frozenset(
            job.id for job in jobs if job.status == JobStatus.COMPLETED and not job_is_imported(job)
        )
        ended_jobs = [
            job for job in jobs if job.status == JobStatus.ENDED and not job_is_imported(job)
        ]
        if not ended_jobs:
            if replaced:
                return RunAgainPlan(start_again_job_ids=replaced, apply=self._apply_run_again)
            return RunAgainPlan()

        resumable = [job for job in ended_jobs if job_resumes_from_checkpoint(job)]
        probes = self._checkpoint_probes(resumable) if resumable else {}
        continuable = [
            job for job in resumable if probes[job.id].status != CheckpointProbeStatus.ABSENT
        ]
        continue_ids: frozenset[UUID] = frozenset()
        kept: dict[UUID, JobInputTexts] = {}
        if continuable:
            answer = self._ask_continue_or_start_again(continuable, probes, ended_jobs)
            if answer is None:
                return None
            if answer == "continue":
                continue_ids = frozenset(job.id for job in continuable)
                kept_inputs = self._kept_relaxation_inputs(continuable)
                if kept_inputs is None:
                    return None
                kept = kept_inputs
        elif any(has_unretrieved_cluster_output(job) for job in ended_jobs):
            if not self._confirm_start_again_over_unretrieved_output(ended_jobs):
                return None
        return RunAgainPlan(
            continue_job_ids=continue_ids,
            start_again_job_ids=(frozenset(job.id for job in ended_jobs) - continue_ids) | replaced,
            apply=self._apply_run_again,
            keep_continued_inputs=partial(self._write_kept_inputs, kept),
        )

    def _relaxation_controller(
        self, job: Job
    ) -> PristineRelaxationController | MuonRelaxationController | None:
        if job.workflow_stage == WorkflowStage.PRISTINE_RELAXATION:
            return self._deps.pristine_relaxation_controller()
        if job.workflow_stage == WorkflowStage.MUON_RELAXATION:
            return self._deps.muon_relaxation_controller()
        return None

    @staticmethod
    def _kept_row_inputs(
        controller: PristineRelaxationController | MuonRelaxationController,
        key: object,
        old: JobInputTexts,
    ) -> tuple[JobInputTexts, frozenset[str]] | None:
        """Return what continuing row ``key`` writes with no edit, and the names it keeps.

        The row's inputs rebuilt from the row, with the project defaults it
        inherits rather than sets (cut-off, k-point grid, functional and their
        either-or partners) kept as ``old``, the run's inputs, hold them: a
        default adopted or changed since the run does not reach a continuation.
        ``None`` when the row cannot be rendered.
        """

        settings = controller.row_settings(key)
        if settings is None:
            return None
        values, advanced = settings
        rendered = controller.render_row_inputs(key, values, advanced)
        if rendered is None:
            return None
        kept_names = inherited_project_default_names(set(values) | set(advanced))
        return overlay_job_inputs(rendered, old, kept_names), kept_names

    def _kept_relaxation_inputs(self, jobs: Sequence[Job]) -> dict[UUID, JobInputTexts] | None:
        """Return the inputs each continuing relaxation in ``jobs`` writes.

        See :meth:`_kept_row_inputs`. Jobs of other workflows are left out;
        their inputs are rebuilt from their rows. ``None``, after saying so,
        when a relaxation's inputs cannot be read or its row cannot be
        rendered: it is not continued with values its run may never have used.
        """

        kept: dict[UUID, JobInputTexts] = {}
        unreadable: list[Job] = []
        for job in jobs:
            controller = self._relaxation_controller(job)
            key = self._relaxation_row_key_for_job(job) if controller is not None else None
            if controller is None or key is None:
                continue
            output_dir = self._deps.job_output_directory(job)
            old = (
                read_job_inputs(output_dir, self._deps.job_seed_name(job))
                if output_dir is not None
                else None
            )
            result = self._kept_row_inputs(controller, key, old) if old is not None else None
            if result is None:
                unreadable.append(job)
                continue
            kept[job.id] = result[0]
        if unreadable:
            self._report_unreadable_relaxation_inputs(unreadable)
            return None
        return kept

    def _report_unreadable_relaxation_inputs(self, jobs: Sequence[Job]) -> None:
        text = (
            f"'{jobs[0].label}' was not continued because its input files could not be read."
            if len(jobs) == 1
            else f"{plural(len(jobs), 'job was', 'jobs were')} not continued because their "
            "input files could not be read."
        )
        project_root = self._document.project_root if self._document is not None else None
        details = "\n".join(
            f"{job.label}: {job_directory_label(project_root, directory)}"
            for job in jobs
            if project_root is not None
            and (directory := self._deps.job_output_directory(job)) is not None
        )
        show_detailed_warning(
            self._parent(),
            "Relaxation not continued",
            text,
            informative_text=(
                "A continued relaxation keeps the inputs its run used, so it needs the job's "
                ".cell and .param. Nothing was submitted or changed. Run the row and choose "
                "'Start again' "
                "to rebuild the inputs from the row."
            ),
            details=details or None,
        )

    def _write_kept_inputs(self, kept: Mapping[UUID, JobInputTexts], jobs: Sequence[Job]) -> None:
        """Write the inputs continuing ``jobs`` keep over the ones just rebuilt from their rows.

        A folder another job uses is left as it is, as the accept step's
        rewrite leaves it. A write that fails is reported.
        """

        failed: list[str] = []
        for job in jobs:
            inputs = kept.get(job.id)
            output_dir = self._deps.job_output_directory(job)
            controller = self._relaxation_controller(job)
            if inputs is None or output_dir is None or controller is None:
                continue
            if not controller.job_may_write_folder(job, output_dir):
                continue
            try:
                write_job_inputs(output_dir, self._deps.job_seed_name(job), inputs)
            except OSError as error:
                failed.append(f"{job.label}: {error}")
        if failed:
            show_detailed_warning(
                self._parent(),
                "Relaxation inputs not kept",
                "The inputs of the previous run could not be written back, so the job runs "
                "with inputs rebuilt from the row.",
                details="\n".join(failed),
            )

    def _apply_run_again(
        self, continued: Sequence[Job], restarted: Sequence[Job], cleaned: Collection[UUID]
    ) -> None:
        """Return jobs to ``defined``: ``continued`` keep their files, ``restarted`` reset.

        A completed job among ``restarted`` (an edited Quantum Corrections or
        Hyperfine Coupling row)
        loses its result as it is reset
        (:meth:`~mufinder.services.job_execution_controller.JobExecutionController.reset_completed_jobs`).

        The :attr:`RunAgainPlan.apply` of the plans
        :meth:`prepare_jobs_for_resubmission` returns, called by the submit
        backends once the submission is accepted. The restarted jobs' folders
        were already cleared by then (``cleaned`` names those), so the reset
        changes no file and records whether each was cleaned; a continued job
        keeps its ``remote_directory``.
        """

        changed = False
        if continued and self._job_execution.continue_ended_jobs(continued):
            changed = True
        ended = [job for job in restarted if job.status == JobStatus.ENDED]
        completed = [job for job in restarted if job.status == JobStatus.COMPLETED]
        if ended and self._job_execution.reset_ended_jobs(ended, cleaned_job_ids=cleaned):
            changed = True
        if completed and self._job_execution.reset_completed_jobs(
            completed, cleaned_job_ids=cleaned
        ):
            changed = True
        if changed:
            self._deps.refresh_journal_view()
            self._deps.persist_document_ui_state(
                failure_message="Jobs were prepared to run again, but autosave failed.",
            )

    def _ask_continue_or_start_again(
        self,
        continuable: Sequence[Job],
        probes: Mapping[UUID, CheckpointProbeResult],
        ended_jobs: Sequence[Job],
    ) -> str | None:
        """Ask whether ``continuable`` continue from their previous run's files.

        ``continuable`` are the jobs with a checkpoint, or whose checkpoint
        could not be checked, which the question says. The rest of
        ``ended_jobs`` start again whatever the answer. Returns ``"continue"``,
        ``"start_again"``, or ``None`` when the user cancels. ``Continue``,
        which deletes nothing, is the default, unless a job's output may still
        be on its cluster unretrieved: then the question says so and ``Cancel``
        is the default.
        """

        count = len(continuable)
        subject = f"'{continuable[0].label}'" if count == 1 else plural(count, "job")
        relaxations = [job for job in continuable if self._relaxation_controller(job) is not None]
        if relaxations:
            inputs_line = (
                "Continue keeps the previous run's files and inputs and resumes from the "
                "checkpoint; a setting changed on the row since then applies, a changed "
                "project default does not. Start again deletes the previous run's output, "
                "including the checkpoint, and rebuilds the inputs from the row's current "
                "settings."
            )
            if len(relaxations) < count:
                inputs_line += " Other jobs rebuild their inputs from the row either way."
        else:
            inputs_line = (
                "Continue keeps the previous run's files and resumes from the checkpoint. "
                "Start again deletes the previous run's output, including the checkpoint. "
                "Either way the inputs are rebuilt from the row's current settings."
            )
        lines = [checkpoint_line([probes[job.id] for job in continuable]), inputs_line]
        others = len(ended_jobs) - count
        if others:
            lines.append(
                f"{plural(others, 'other job has', 'other jobs have')} no checkpoint and "
                f"{'starts' if others == 1 else 'start'} again."
            )
        note = unretrieved_output_note(list(ended_jobs), action="Starting again")
        if note:
            lines.append(note)
        answer = workflow_table_utils.confirm_choice(
            self._parent(),
            "Run again",
            f"{subject} did not finish. Continue from the previous run's files, or start again?",
            [("continue", "Continue"), ("start_again", "Start again"), ("cancel", "Cancel")],
            default="cancel" if note else "continue",
            escape="cancel",
            detail="\n\n".join(lines),
            destructive=("start_again",),
        )
        return None if answer == "cancel" else answer

    def _confirm_start_again_over_unretrieved_output(self, jobs: Sequence[Job]) -> bool:
        """Confirm starting ``jobs`` again when some may have unretrieved cluster output.

        Starting again gives a cluster job a fresh remote folder, so output that
        finished there but was never retrieved could no longer be fetched.
        ``Cancel`` is the default.
        """

        subject = f"'{jobs[0].label}'" if len(jobs) == 1 else plural(len(jobs), "job")
        return workflow_table_utils.confirm_destructive(
            self._parent(),
            "Run again",
            f"Start {subject} again?",
            "\n\n".join(
                [
                    unretrieved_output_note(list(jobs), action="Starting again"),
                    "Starting again deletes the previous run's output and rebuilds the "
                    "inputs from the row's current settings.",
                ]
            ),
            "Start again",
        )

    def _continue_relaxation_checkpoint_probe(self, job: Job) -> CheckpointProbeResult:
        """Probe for ``{seed}.check`` in the directory a Continue would reuse.

        Local jobs get a plain :class:`~pathlib.Path` check. Cluster jobs are
        probed only when the cluster runner is currently connected to that
        job's cluster (:meth:`_is_cluster_connected_for_job`, the same check
        cluster resubmission uses) via the WP5 bridge
        (``ClusterController.check_remote_files``/``remote_files_checked``),
        awaited synchronously through :meth:`_await_remote_file_check` — the
        same nested-event-loop pattern ``submit_jobs_on_cluster`` already uses
        mid-flow before showing a dialog, run here before the Continue dialog
        is constructed (never inside the dialog's own ``__init__``). Not
        connected, a missing remote directory, or an SSH error all map to
        :meth:`CheckpointProbeResult.unavailable` rather than blocking the
        dialog from opening.
        """

        return self._checkpoint_probes([job])[job.id]

    def _checkpoint_probes(self, jobs: Sequence[Job]) -> dict[UUID, CheckpointProbeResult]:
        """Probe several jobs' ``{seed}.check`` files, with one remote round trip.

        Per job this is exactly :meth:`_continue_relaxation_checkpoint_probe`;
        every probed cluster path goes into a single remote check, so a batch
        resubmission waits once rather than once per job.
        """

        probes: dict[UUID, CheckpointProbeResult] = {}
        remote_paths: dict[UUID, str] = {}
        for job in jobs:
            seed_name = self._deps.job_seed_name(job)
            if (job.execution_target or "").strip().casefold() == "local":
                output_dir = self._deps.job_output_directory(job)
                if output_dir is None:
                    probes[job.id] = CheckpointProbeResult.unavailable(
                        "The job's output directory is unknown"
                    )
                elif (output_dir / f"{seed_name}.check").exists():
                    probes[job.id] = CheckpointProbeResult.found()
                else:
                    probes[job.id] = CheckpointProbeResult.absent()
                continue
            if not self._is_cluster_connected_for_job(job):
                probes[job.id] = CheckpointProbeResult.unavailable(
                    f"Not connected to {job.execution_target}"
                )
                continue
            remote_directory = (job.remote_directory or "").strip()
            if not remote_directory:
                probes[job.id] = CheckpointProbeResult.unavailable(
                    "No remote directory recorded for this job"
                )
                continue
            remote_paths[job.id] = posixpath.join(remote_directory, f"{seed_name}.check")

        if remote_paths:
            result = self._await_remote_file_check(list(remote_paths.items()))
            errors = result.get("errors", {})
            existing = result.get("existing", [])
            for job_id, remote_path in remote_paths.items():
                if job_id in errors:
                    probes[job_id] = CheckpointProbeResult.unavailable(str(errors[job_id]))
                elif remote_path in existing:
                    probes[job_id] = CheckpointProbeResult.found()
                else:
                    probes[job_id] = CheckpointProbeResult.absent()
        return probes

    def _continue_relaxation_row(
        self,
        *,
        controller: PristineRelaxationController | MuonRelaxationController,
        view: PristineRelaxationWidget | MuonRelaxationWidget,
        key: object,
        autosave_failure_message: str,
    ) -> None:
        """Open "Continue relaxation…" for one row and drive it to submission.

        Owns every side effect the workflow controller and the
        ``ContinueRelaxationDialog`` (WP4) deliberately do not: resolving the
        row's job, probing for a checkpoint, reading the job's ``.cell`` and
        ``.param`` from disk for the dialog to compare with what continuing
        would write, opening the dialog, and — on accept — running the
        controller's ``continue_job`` (row mutation -> ``prepare_job_continuation``
        -> provenance -> re-materialize) followed by submission to the job's
        ORIGINAL execution target (finding 9 of the warm-start plan: local jobs
        through the local submit path, cluster jobs through
        ``submit_jobs_on_cluster`` with that cluster preselected, mirroring
        how ``_on_resubmit_from_dashboard`` picks ``default_cluster_name``).

        Deliberately bypasses ``controller.submit()``: that template method has
        no hook for preselecting a specific cluster (``submission_groups``
        supplies no per-group ``default_cluster_name``), and — for muon
        relaxation — its ``pre_submit_check`` host-mismatch validation would be
        redundant here: the row's site already matched the active host when
        this job was first submitted, and offers no sensible resolution for
        resubmitting that same site. Because ``continue_job`` already leaves
        the job ``DEFINED`` before either submit path runs,
        the submit cycle has no ended job to run again, so the run-again
        question (``prepare_jobs_for_resubmission``) is never asked — Continue
        never asks about deleting the very directory it just preserved and
        rewrote.
        """

        if self._document is None:
            return
        jobs = controller.jobs_for_keys([key])
        if not jobs:
            return
        job = jobs[0]
        if job.status not in CONTINUABLE_JOB_STATUSES:
            return
        if not self.without_jobs_in_another_jobs_folder([job]):
            return

        seed = view.continue_relaxation_dialog_seed(key)
        if seed is None:
            return

        checkpoint_probe = self._continue_relaxation_checkpoint_probe(job)

        def render_inputs(
            values: Mapping[str, object], advanced: tuple[str, ...]
        ) -> JobInputTexts | None:
            return controller.render_row_inputs(key, values, advanced)

        # The inputs are written locally for every job before it is submitted
        # (a cluster job's files are uploaded from there), and Continue
        # rewrites them in the same folder.
        output_dir = self._deps.job_output_directory(job)
        old_inputs = (
            read_job_inputs(output_dir, self._deps.job_seed_name(job))
            if output_dir is not None
            else None
        )

        kept = (
            self._kept_row_inputs(controller, key, old_inputs) if old_inputs is not None else None
        )
        if output_dir is None or old_inputs is None or kept is None:
            self._report_unreadable_relaxation_inputs([job])
            return
        kept_inputs, kept_names = kept

        if (job.execution_target or "").strip().casefold() == "local":
            reused_directory = str(output_dir)
        else:
            reused_directory = job.remote_directory or "(directory will be computed at submission)"

        dialog = ContinueRelaxationDialog(
            **seed,
            old_inputs=old_inputs,
            render_inputs=render_inputs,
            kept_inputs=kept_inputs,
            kept_names=kept_names,
            checkpoint_probe=checkpoint_probe,
            execution_target=job.execution_target,
            reused_directory=reused_directory,
            parent=self._parent(),
        )
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            parameter_values, added_advanced = dialog.edited_parameters()
            continuation_inputs = dialog.continuation_inputs()
        finally:
            # The dialog is a child of the window; without this every open
            # would stay alive under it, editors and all.
            dialog.deleteLater()
        continued_job = (
            controller.continue_job(key, parameter_values, added_advanced)
            if continuation_inputs is not None
            else None
        )
        if continued_job is None or continuation_inputs is None:
            # The row's parameters may already be mutated (update_row_parameters
            # ran before materialize failed), but nothing was submitted: warn
            # rather than fail silently, since the user just accepted a diff
            # that promised the new parameters would run.
            QMessageBox.warning(
                self._parent(),
                "Continue relaxation failed",
                "Could not regenerate this job's input files, so it was not "
                "resubmitted. The row's job was left in a recoverable state; "
                "try Continue relaxation again.",
            )
            return
        # The job's inputs were rewritten from the row; the continuation writes
        # what the dialog's table showed: the run's values for the project
        # defaults the row inherits, and the dialog's edits.
        try:
            write_job_inputs(
                output_dir, self._deps.job_seed_name(continued_job), continuation_inputs
            )
        except OSError as error:
            show_detailed_warning(
                self._parent(),
                "Relaxation not continued",
                f"'{continued_job.label}' was not submitted because its inputs could not be "
                "written.",
                details=str(error),
            )
            return

        if (continued_job.execution_target or "").strip().casefold() == "local":
            self.submit_jobs_locally([continued_job])
        else:
            self.submit_jobs_on_cluster(
                [continued_job], default_cluster_name=continued_job.execution_target
            )

        self._deps.refresh_views_after_job_reset()
        self._deps.persist_document_ui_state(failure_message=autosave_failure_message)

    def on_continue_relaxation_requested_from_pristine_relaxation(self, key: object) -> None:
        self._continue_relaxation_row(
            controller=self._deps.pristine_relaxation_controller(),
            view=self._deps.pristine_relaxation_view(),
            key=key,
            autosave_failure_message="Relaxation was continued, but autosave failed.",
        )

    def on_continue_relaxation_requested_from_muon_relaxation(self, key: object) -> None:
        self._continue_relaxation_row(
            controller=self._deps.muon_relaxation_controller(),
            view=self._deps.muon_relaxation_view(),
            key=key,
            autosave_failure_message="Relaxation was continued, but autosave failed.",
        )

    def _job_failure_diagnosis(self, job: Job) -> FailureDiagnosis:
        """Diagnose ``job`` from its logs, keeping the cause recorded at failure time.

        The fresh diagnosis supplies the evidence line, and covers jobs that
        failed before causes were recorded. A recorded cause the logs cannot
        show (an interrupted run, a rejected submission) is kept as is.
        """

        fresh = self._job_execution.diagnose_failure(job, job.error_message)
        if job.end_reason in (None, EndReason.UNKNOWN) or fresh.reason == job.end_reason:
            return fresh
        return FailureDiagnosis(reason=job.end_reason)

    def reset_tracking_for_project_change(self) -> None:
        assert self._document is not None
        self._cluster_runner.forget_progress()
        self._cluster_lost_names.clear()
        self._queue_refreshes_pending = 0
        self._cluster_queue_checked_label.setText("Queue last checked: never")

        # Local jobs that were in progress when MuFinder closed have no process
        # now. This runs before the jobs table is rebuilt, so no such job is
        # ever shown as running. Cluster jobs are asked about on the worker.
        self.check_jobs(project_opened=True)

    def _task_farming_for_job(self, job: Job) -> tuple[bool, int | None]:
        """Return ``(farming_enabled, num_farms)`` for any farm-aware calc type.

        Both runners use this to decide which output files a finished job wrote:
        task farming renames every output to ``<seed>_farmNNN.<ext>``, so a job
        whose farm count is misreported here is read from the wrong files. The
        quantum-phonon and NEB workflows each store their farm count in the
        owning definition's per-row configuration extras; every other calc type
        is unfarmed. A job a submission has planned but not created reads its
        row from the plan (:func:`planned_job_entry`).
        """

        if job.calc_type not in {QUANTUM_PHONON_CALC_TYPE, NUDGED_ELASTIC_BAND_CALC_TYPE}:
            return False, None
        planned = planned_job_entry(self._deps.workflow_controllers(), job)
        if planned is not None:
            # A planned job is in no mapping yet: its row is known from the plan.
            definition, key = planned.definition, planned.key
        elif job.calc_type == QUANTUM_PHONON_CALC_TYPE:
            definition, key = (
                self._deps.quantum_corrections_controller().definition_and_key_for_job(job)
            )
        else:
            definition, key = self._deps.energy_barriers_controller().definition_and_key_for_job(
                job
            )
        if definition is None or key is None:
            return False, None
        _params, _advanced, extras = workflow_row_configuration(definition, key)
        num_farms = num_farms_from_extras(extras)
        return num_farms > 1, (num_farms if num_farms > 1 else None)

    def _on_local_job_status_changed(self, job_id: UUID, status_value: str) -> None:
        if self._document is None:
            return
        update = self._job_execution.mark_local_status_changed(job_id, status_value)
        if update.changed:
            self._deps.refresh_jobs_table()
            self._deps.convergence_controller().refresh_view()
            self._deps.kpoint_convergence_controller().refresh_view()
            self._deps.refresh_hyperfine_coupling_view()
            self._deps.refresh_quantum_corrections_view()
            self._deps.refresh_energy_barriers_view()
            self._deps.persist_document_ui_state(
                failure_message="Local job status changed, but autosave failed.",
            )

    def _on_local_job_started(self, job_id: UUID) -> None:
        if self._document is None:
            return
        update = self._job_execution.mark_started(job_id)
        if update.changed:
            self._deps.refresh_jobs_table()
            self._deps.persist_document_ui_state(
                failure_message="Local job start was recorded, but autosave failed.",
            )

    def _on_local_job_finished(
        self,
        job_id: UUID,
        parsed: CastepResult,
        artifacts: ParsedJobArtifacts | None = None,
    ) -> None:
        if self._document is None:
            return
        job = self._document.jobs.get(job_id)
        if job is None:
            return
        results_before = frozenset(self._document.results)
        try:
            self._record_finished_job(job, parsed, artifacts)
        except Exception as error:  # noqa: BLE001 - a job must not stay active on any error
            _LOG.exception("Could not record the result of finished job %s", job.label)
            update = self._job_execution.fail_unrecorded_finish(
                job_id, error, results_before=results_before
            )
            if update.changed:
                self._deps.request_dashboard_flush(
                    failure_message="Failed job state was recorded, but autosave failed.",
                )
                self._deps.status_bar().showMessage(f"Could not record the result of {job.label}")

    def _record_finished_job(
        self,
        job: Job,
        parsed: CastepResult,
        artifacts: ParsedJobArtifacts | None,
    ) -> None:
        """Build and store ``job``'s result from its parsed output, and complete it."""

        assert self._document is not None
        job_id = job.id

        # The runner worker thread parses the secondary outputs (relaxed cell /
        # phonon / transition state) exactly once and delivers them here as
        # ``artifacts``. Non-runner callers (reconciliation/retry paths, tests
        # that invoke this directly) pass no carrier: fall back to synchronous
        # collection through the *same* collector so parse-once still holds — the
        # file is read once, just on the GUI thread for those paths.
        if artifacts is None:
            flags = self._cluster_retrieve_flags(job)
            # The farm count must be resolved here too: a task-farmed job writes
            # <seed>_farm001.ts and no <seed>.ts, so collecting at the default of
            # one farm would report a successful farmed job's output as missing.
            _farming_enabled, farm_count = self._task_farming_for_job(job)
            artifacts = collect_job_artifacts(
                self._deps.job_output_directory(job),
                self._deps.job_seed_name(job),
                want_out_cell=bool(flags.get("retrieve_out_cell", False)),
                want_phonon=bool(flags.get("retrieve_phonon", False)),
                want_ts=bool(flags.get("retrieve_ts", False)),
                want_magres=bool(flags.get("retrieve_magres", False)),
                num_farms=farm_count or 1,
            )

        run = self._job_execution.ensure_run_for_job(job, started=True)
        if run is None:
            return

        # The owning controller rejects a completion whose outputs cannot carry
        # a result (unconverged geometry, a missing, unparseable or stale
        # secondary output). It reads the carrier, never disk, and runs before
        # any result is built.
        failure = validate_controller_artifacts(
            self._deps.workflow_controllers(), job, parsed, run, artifacts
        )
        if failure is not None:
            self._on_local_job_failed(job_id, failure)
            return

        result_payload = {
            "completed": parsed.completed,
            "total_energy": parsed.total_energy,
            "enthalpy": parsed.enthalpy,
            "forces": [list(force) for force in parsed.forces],
            "stress": [list(row) for row in parsed.stress],
            "max_force": parsed.max_force,
            "max_stress": parsed.max_stress,
            "num_scf_cycles": parsed.num_scf_cycles,
            "wall_time": parsed.wall_time,
            "warnings": list(parsed.warnings),
            "geom_converged": parsed.geom_converged,
        }

        # The owning workflow controller (if any) contributes payload, file and
        # structure additions before the Result is built: a strict no-op for a
        # calc type no controller owns and for an empty ControllerResult (cutoff
        # convergence). It consumes the pre-parsed ``artifacts``.
        result_payload, output_files, derived_structure_ids, merged_warnings = (
            merge_controller_result(
                self._deps.workflow_controllers(),
                job,
                parsed,
                run,
                artifacts,
                payload=result_payload,
                output_files=job_output_files(
                    self._document.project_root,
                    self._deps.job_output_directory(job),
                    self._deps.job_seed_name(job),
                ),
                derived_structure_ids=[],
                warnings=list(parsed.warnings),
            )
        )

        result = Result(
            calculation_run_id=run.id,
            result_type=job.calc_type,
            data=result_payload,
            output_files=output_files,
            derived_structure_ids=derived_structure_ids,
        )
        self._document.results[result.id] = result

        self._job_execution.complete_job_with_result(
            job,
            result=result,
            warnings=merged_warnings,
            convergence_achieved=parsed.geom_converged,
        )
        self._deps.request_dashboard_flush(
            failure_message="Completed job result was recorded, but autosave failed.",
        )
        self._deps.status_bar().showMessage(f"Completed local job {job.label}")

    def _on_local_job_failed(self, job_id: UUID, message: str) -> None:
        if self._document is None:
            return
        update = self._job_execution.fail_job(job_id, message)
        if update.job is None:
            return
        self._deps.request_dashboard_flush(
            failure_message="Failed job state was recorded, but autosave failed.",
        )
        self._deps.status_bar().showMessage(f"Local job failed: {update.job.label}")

    def _on_local_job_cancelled(self, job_id: UUID) -> None:
        job = self._record_cancel(job_id)
        if job is not None:
            self._deps.status_bar().showMessage(f"Cancelled local job {job.label}")

    def _record_cancel(self, job_id: UUID, *, scheduler_told: bool = True) -> Job | None:
        """End ``job_id`` as cancelled and save it; return the job, or ``None``.

        ``scheduler_told`` is False when the cancel never reached the job's
        scheduler, which the job then records so it is offered ``Check now``.
        """

        if self._document is None:
            return None
        update = self._job_execution.cancel_job(job_id)
        if update.job is None:
            return None
        if not scheduler_told:
            self._job_execution.mark_cancel_not_sent(job_id)
        self._deps.request_dashboard_flush(
            failure_message="Cancelled job state was recorded, but autosave failed.",
        )
        return update.job

    def _on_cluster_job_checked(self, job_id: UUID, evidence: JobEvidence) -> None:
        """Apply what a check saw of one cluster job."""

        if self._document is None:
            return
        update = self._job_execution.apply_evidence(job_id, evidence)
        job = update.job
        if job is None:
            return
        if update.finished is not None:
            self._on_local_job_finished(job_id, update.finished.parsed, update.finished.artifacts)
            # Recording the result can fail the job instead; that path reports itself.
            if job.status == JobStatus.COMPLETED:
                self._deps.status_bar().showMessage(f"Completed cluster job {job.label}")
        if update.changed or update.stamped:
            self._deps.request_dashboard_flush(
                failure_message="Cluster job state was recorded, but autosave failed.",
            )

    def _on_cluster_job_cancelled(self, job_id: UUID) -> None:
        job = self._record_cancel(job_id)
        if job is not None:
            self._deps.status_bar().showMessage(f"Cancelled cluster job {job.label}")

    def _on_cluster_connection_status_changed(self, connected: bool) -> None:
        if connected:
            name = self._cluster_runner.connected_cluster_name
            self._cluster_last_connected_name = name
            self._cluster_lost_names.discard((name or "").strip().casefold())
        else:
            # Polls are serial, so any poll requested before this has already reported.
            self._queue_refreshes_pending = 0
            self._deps.job_dashboard().clear_cluster_elapsed()
        self.sync_cluster_status_widgets()
        # Each row's action and stale marker depend on the connection, so the
        # table follows it at once instead of waiting for the next queue poll.
        if self._document is not None:
            self._deps.refresh_jobs_table()
            if connected:
                self._job_execution.check_cluster_jobs(skip_if_busy=True)

    def sync_cluster_status_widgets(self) -> None:
        """Show the cluster widgets only while the project has cluster jobs to follow."""

        status = cluster_status(
            self._document.jobs.values() if self._document is not None else (),
            connected_cluster=self._cluster_runner.connected_cluster_name,
            lost_clusters=self._cluster_lost_names,
        )
        self._cluster_connection_label.setText(status.connection_text)
        self._cluster_status_tone = status.tone
        self.refresh_cluster_connection_label_colour()
        self._cluster_refresh_button.setEnabled(status.can_poll)
        self._cluster_refresh_button.setToolTip(status.refresh_tooltip)
        for widget in (
            self._cluster_connection_label,
            self._cluster_queue_checked_label,
            self._cluster_refresh_button,
        ):
            widget.setVisible(status.visible)

    def refresh_cluster_connection_label_colour(self) -> None:
        palette = current_palette()
        colour = {
            ClusterConnectionTone.NEUTRAL: palette.text_muted,
            ClusterConnectionTone.CONNECTED: palette.success,
            ClusterConnectionTone.LOST: palette.danger,
        }[self._cluster_status_tone]
        # This runs on every dashboard flush; only an actual change re-styles.
        style = f"color: {colour};"
        if self._cluster_connection_label.styleSheet() != style:
            self._cluster_connection_label.setStyleSheet(style)

    def _on_cluster_queue_checked(self, checked_at) -> None:
        # A check in flight when another project opened reports the jobs of the
        # project it started in; it says nothing about the one now open.
        observed = getattr(checked_at, "job_ids", None)
        if (
            self._document is not None
            and observed
            and self._document.jobs.keys().isdisjoint(observed)
        ):
            return
        # Each job's evidence was applied before this arrives, and asked for
        # its own flush; this drains whatever is still pending.
        if self._deps.dashboard_flush_pending():
            self._deps.request_dashboard_flush(
                failure_message="Queue check was recorded, but autosave failed.",
            )
        stamp_source = getattr(checked_at, "checked_at", checked_at)
        # A pass over some jobs (Check now on one) changes their times alone.
        self._deps.job_dashboard().set_cluster_elapsed(
            getattr(checked_at, "elapsed_by_job_id", None),
            stamp_source,
            checked_job_ids=getattr(checked_at, "job_ids", None),
        )
        if hasattr(stamp_source, "strftime"):
            localized = stamp_source
            if getattr(stamp_source, "tzinfo", None) is not None and hasattr(
                stamp_source, "astimezone"
            ):
                try:
                    localized = stamp_source.astimezone()
                except Exception:
                    localized = stamp_source
            stamp = localized.strftime("%H:%M:%S")
        elif isinstance(stamp_source, str):
            stamp = stamp_source.strip() or "unknown"
        else:
            stamp = "unknown"
        self._cluster_queue_checked_label.setText(f"Queue last checked: {stamp}")
        # Only the refresh's own pass answers the button, not a check queued before it.
        if self._queue_refreshes_pending and getattr(checked_at, "refresh", False):
            self._queue_refreshes_pending -= 1
            checked = len(getattr(checked_at, "job_ids", ()))
            self._deps.status_bar().showMessage(
                f"Refreshed queue: checked {plural(checked, 'job')} at {stamp}"
            )
        # One line for everything this pass changed; it outranks the plain
        # "refreshed" line, which says less.
        summary = self._job_execution.take_check_summary(len(getattr(checked_at, "job_ids", ())))
        if summary:
            self._deps.status_bar().showMessage(summary)
        self._deps.job_dashboard().sync_runtime_timer()
        self._deps.job_dashboard().refresh_runtime_cells()

    def _on_cluster_poll_failed(self, message: str) -> None:
        # A failed poll is what closes a connection without being asked to, so
        # this is the drop the connection label reports (the runner signals the
        # disconnect first, which also happens on every deliberate reconnect).
        name = self._cluster_last_connected_name
        if name:
            self._cluster_lost_names.add(name.strip().casefold())
        self._queue_refreshes_pending = 0
        # The pass did not finish, so what it changed goes unsummarised.
        self._job_execution.take_check_summary(0)
        self.sync_cluster_status_widgets()
        self._deps.status_bar().showMessage(f"Failed to check queue on {name or 'the cluster'}")
        show_detailed_warning(
            self._parent(),
            "Queue polling failed",
            (
                f"{message}\n\n"
                "Check VPN/network access, SSH trust, authentication, and scheduler command paths."
            ),
            informative_text="Open Details to inspect the full polling failure.",
        )

    def _on_abandoned_cluster_job_cancel_failed(
        self, job_id: object, remote_job_id: str, error: str
    ) -> None:
        job = self._document.jobs.get(job_id) if self._document is not None else None
        label = job.label if job is not None else "A cancelled job"
        if job is not None and self._job_execution.mark_cancel_not_sent(job.id, remote_job_id):
            self._deps.request_dashboard_flush(
                failure_message="Cancelled job state was recorded, but autosave failed.",
            )
        show_detailed_warning(
            self._parent(),
            "Cluster cancellation failed",
            (
                f"{label} was submitted after the cluster stopped responding, and "
                f"cancelling scheduler job {remote_job_id} failed. Cancel it on the cluster.\n\n"
                f"{error}"
            ),
            informative_text="Open Details to inspect the full cancellation error.",
        )

    def _connect_to_cluster_with_trust(
        self, config: ClusterConfig, *, credentials: SSHCredentials
    ) -> None:
        trusted_endpoints: set[str] = set()
        while True:
            error = self.await_cluster_connect(config, credentials=credentials)
            if error is None:
                return
            if isinstance(error, SSHUnknownHostKeyError):
                if error.endpoint in trusted_endpoints:
                    raise error
                if not prompt_to_trust_host_key(self._parent(), config, error):
                    raise ConnectionError(
                        "Connection cancelled because the SSH host key was not trusted."
                    ) from error
                save_trusted_host_key(
                    self._settings,
                    config,
                    algorithm=error.algorithm,
                    key_base64=error.key_base64,
                    endpoint=error.endpoint,
                )
                trusted_endpoints.add(error.endpoint)
                continue
            raise error

    def _on_refresh_cluster_queue(self) -> None:
        try:
            if not self._connected_to_a_monitored_cluster():
                self._refresh_queue_for_active_dashboard_clusters()
                return
            self._request_queue_poll()
        except Exception as exc:  # noqa: BLE001
            self._queue_refreshes_pending = 0
            self._deps.status_bar().showMessage("Failed to refresh queue")
            show_detailed_warning(
                self._parent(),
                "Queue refresh failed",
                str(exc),
                informative_text="Open Details to inspect the full queue refresh error.",
            )

    def _connected_to_a_monitored_cluster(self) -> bool:
        # A connection left over from another project would poll none of this
        # project's jobs, so it counts as not connected.
        if not self._cluster_runner.is_connected:
            return False
        connected = (self._cluster_runner.connected_cluster_name or "").strip().casefold()
        return connected in {name.casefold() for name in self._refreshable_cluster_targets()}

    def _request_queue_poll(self) -> None:
        # The refresh runs on the cluster worker, after any check already there;
        # its result reaches the status bar through queue_checked or poll_failed.
        self._queue_refreshes_pending += 1
        self._job_execution.refresh_cluster_jobs()

    def _refresh_queue_for_active_dashboard_clusters(self) -> None:
        target_clusters = self._refreshable_cluster_targets()
        if not target_clusters:
            return

        refreshed: list[str] = []
        failures: list[str] = []
        failed_names: list[str] = []
        for target in target_clusters:
            name, failure = self._reconnect_to_cluster(target)
            if failure is not None:
                failed_names.append(name)
                failures.append(failure)
                continue
            self._request_queue_poll()
            refreshed.append(name)

        if refreshed:
            self._deps.status_bar().showMessage(
                "Reconnected to " + ", ".join(sorted(set(refreshed)))
            )

        if failures:
            self._deps.status_bar().showMessage(
                "Failed to check queue on " + ", ".join(failed_names)
            )
            show_detailed_warning(
                self._parent(),
                "Queue refresh failed",
                "Some clusters could not be refreshed:\n" + "\n".join(failures),
                informative_text="Open Details to inspect the full queue refresh failures.",
            )

    def _reconnect_to_cluster(self, target: str) -> tuple[str, str | None]:
        """Connect to cluster ``target`` as ``Refresh Queue`` does, asking for credentials if needed.

        Returns the cluster's configured name and ``None`` once connected, or a
        ``- <name>: <reason>`` line saying why it could not be.
        """

        config = next(
            (
                config
                for config in load_cluster_configs(self._settings)
                if config.name.strip().casefold() == target.strip().casefold()
            ),
            None,
        )
        if config is None:
            return target, f"- {target}: cluster is not configured. Set it in Setup > HPC Clusters."
        try:
            self._connect_to_cluster_with_trust(config, credentials=SSHCredentials())
            return config.name, None
        except Exception as exc:  # noqa: BLE001
            auth_method = config.auth_method.strip().lower()
            needs_credentials = auth_method == "password" or isinstance(exc, SSHAuthenticationError)
            if not needs_credentials:
                return (
                    config.name,
                    f"- {config.name}: {format_cluster_connection_error(config, exc)}",
                )
        credentials = prompt_for_cluster_credentials(self._parent(), config)
        if credentials is None:
            return (
                config.name,
                f"- {config.name}: credentials required for reconnect. Enter them when prompted.",
            )
        try:
            self._connect_to_cluster_with_trust(config, credentials=credentials)
            return config.name, None
        except Exception as retry_exc:  # noqa: BLE001
            return (
                config.name,
                f"- {config.name}: {format_cluster_connection_error(config, retry_exc)}",
            )

    def _refreshable_cluster_targets(self) -> list[str]:
        # The clusters of active jobs and of ended jobs whose output may still be there.
        if self._document is None:
            return []
        return refreshable_cluster_names(self._document.jobs.values())

    def check_jobs(self, *, project_opened: bool = False) -> None:
        """Check every active job against what can be seen of it.

        Runs on project open (``project_opened``) and off a 60 s timer. A local
        job the runner holds no process for is completed from complete output,
        and stopped otherwise. On project open a local job stopped as
        ``interrupted`` is also completed when its folder now holds complete
        output from its run. Cluster jobs are asked about on the worker, unless
        an earlier check is still running there; their evidence arrives later.
        """

        if self._document is None:
            return
        self._job_execution.check_cluster_jobs(skip_if_busy=True)
        stopped = 0
        local_updates = self._job_execution.check_local_jobs(include_interrupted=project_opened)
        for update in local_updates:
            if update.job is None:
                continue
            if update.finished is not None:
                self._on_local_job_finished(
                    update.job.id, update.finished.parsed, update.finished.artifacts
                )
            elif update.job.status == JobStatus.ENDED:
                stopped += 1
        if stopped == 0:
            return
        self._deps.request_dashboard_flush(
            failure_message="Local jobs with no process were stopped, but autosave failed."
        )
        self._deps.status_bar().showMessage(
            f"{plural(stopped, 'local job was', 'local jobs were')} stopped: MuFinder has no "
            f"process for {'it' if stopped == 1 else 'them'}"
        )

    def retrieval_flags_for_job(self, job: Job) -> RetrievalFlags:
        """Return the retrieval flags and farm count both runners need for ``job``.

        Shared by local and cluster submission and by every cluster check,
        so every path collects the same
        optional outputs (``.den_fmt`` / ``-out.cell`` / ``.phonon`` / ``.ts``)
        under the same farmed names.
        """

        task_farming_enabled, num_farms = self._task_farming_for_job(job)
        return RetrievalFlags.for_calc_type(
            job.calc_type, task_farming_enabled=task_farming_enabled, num_farms=num_farms
        )

    @staticmethod
    def _cluster_retrieve_flags(job: Job) -> dict[str, bool]:
        """Return the per-calc-type ``retrieve_*`` flags for ``job`` as a mapping."""

        flags = RetrievalFlags.for_calc_type(job.calc_type)
        return {
            "retrieve_den_fmt": flags.retrieve_den_fmt,
            "retrieve_out_cell": flags.retrieve_out_cell,
            "retrieve_phonon": flags.retrieve_phonon,
            "retrieve_ts": flags.retrieve_ts,
            "retrieve_magres": flags.retrieve_magres,
        }

    def infer_remote_job_directory(self, job: Job) -> str:
        config = load_cluster_config(self._settings, job.execution_target.strip())
        if config is None:
            return ""

        working_directory = config.working_directory_for()
        if not working_directory:
            return ""

        try:
            return self._cluster_runner.remote_job_directory(
                working_directory=working_directory,
                seed_name=self._deps.job_seed_name(job),
            )
        except Exception:
            return ""
