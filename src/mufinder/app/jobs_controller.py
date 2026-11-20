"""The jobs controller: what happens to jobs when the user acts on them.

``JobsController`` carries out every job action the Job Dashboard, the workflow
rows and the workflow controllers ask for: local and cluster submission (with
their dialogs, the cluster connection and the waits on the cluster worker),
resubmission, Continue relaxation, reset, cancel, inspect, ``Check now`` and
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
from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import EndReason, JobActivity, JobStatus, WorkflowStage
from mufinder.domain.models import (
    Job,
)
from mufinder.app.cluster_submission_dialog import ClusterSubmissionConfig, ClusterSubmissionDialog
from mufinder.app.cluster_ui_utils import (
    format_cluster_connection_error,
    show_detailed_warning,
)
from mufinder.app.keyword_catalog_guard import KeywordCatalogGuard
from mufinder.app.export_dialogs import open_containing_folder
from mufinder.app.failed_job_dialog import FailedJobAction, FailedJobDialog
from mufinder.domain.text import (
    KeptFolder,
    folder_outcome_status,
    plain_error,
    plural,
)
from mufinder.app.relaxation_types import (
    MuonRelaxationRow,
)
from mufinder.app.relaxation_widget import PristineRelaxationWidget
from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget
from mufinder.app.continue_relaxation_dialog import ContinueRelaxationDialog
from mufinder.app.resubmit_choice_dialog import (
    ResubmitCandidate,
    ResubmitChoiceDialog,
    ResubmitMode,
)
from mufinder.app.geometry_progress_dialog import (
    GeometryProgressDialog,
)
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.project_home_trail import (
    ConstructionSignWidget,  # noqa: F401 - re-export only
)
from mufinder.app.submission_dialog import LocalSubmissionDialog
from mufinder.app.workflow_config_codecs import (
    deserialize_muon_relaxation_config,
    deserialize_pristine_relaxation_config,
)
from mufinder.app.workflow_parameters import (
    kpoint_exclusive_hint,
    reorder_workflow_parameter_specs,
)
from mufinder.app.workflow_submission import (
    RESUBMITTABLE_JOB_STATES,
    job_is_imported,
    job_resumes_from_checkpoint,
)
from mufinder.app.workflow_controllers import (
    ConvergenceCutoffController,
    ConvergenceKpointController,
    MuonRelaxationController,
    PristineRelaxationController,
    WorkflowJobController,
    dedupe_jobs,
    owning_controller,
)
from mufinder.castep_io.geometry_progress import GeometryProgress
from mufinder.app.cluster_controller import ClusterController
from mufinder.services.cluster_runner import CheckpointProbeResult
from mufinder.services.cluster_submission_planner import (
    ClusterSubmissionPlan,
    can_reuse_cluster_connection,
    plan_cluster_submission,
    unsafe_job_labels_message,
)
from mufinder.services.job_execution_controller import (
    JobExecutionController,
    PreparedClusterSubmission,
    remember_local_cores,
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
    refusal_because,
)
from mufinder.services.job_paths import (
    job_directory_label,
)
from mufinder.services.job_monitoring import (
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


@dataclass(frozen=True)
class JobsControllerDeps:
    """The window's state and helpers the jobs controller uses, late-bound."""

    document: Callable[[], ProjectDocument | None]
    #: The widget dialogs and prompts are parented to.
    dialog_parent: Callable[[], QWidget]
    status_bar: Callable[[], QStatusBar]
    job_output_directory: Callable[[Job], Path | None]
    job_seed_name: Callable[[Job], str]
    retrieval_flags_for_job: Callable[[Job], RetrievalFlags]
    task_farming_for_job: Callable[[Job], tuple[bool, int | None]]
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
    show_kept_folders: Callable[..., None]
    detach_convergence_job: Callable[[UUID], None]
    open_castep_setup_dialog: Callable[[], None]
    open_cluster_setup_dialog: Callable[[], None]
    edit_cluster_during_submission: Callable[..., object]
    connect_to_cluster_with_trust: Callable[..., bool]
    on_local_job_finished: Callable[..., None]
    on_cluster_job_cancelled: Callable[[UUID], None]
    keyword_catalog_guard: Callable[[], KeywordCatalogGuard]


class JobsController(QObject):
    """Carries out the user's job actions; state changes go through the services."""

    _LAST_CLUSTER_SUBMISSION_CONFIG_KEY = "last_cluster_submission_config"

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

        # A job in a folder another job uses is refused as a submit refuses it, before
        # anything is reset or cleared.
        selected = self.without_jobs_in_another_jobs_folder(selected)
        if not selected:
            return

        refusals = self._fresh_start_refusals(
            [job for job in selected if job_resumes_from_checkpoint(job)]
        )
        if refusals:
            # Neither start these again nor continue them: their directory is
            # not theirs alone, so starting again would delete another job's
            # files and continuing would run in (and overwrite) them.
            selected = [job for job in selected if job.id not in refusals]
            QMessageBox.warning(
                self._parent(),
                "Some jobs were not resubmitted",
                "These relaxations' folders are not theirs alone, so resubmitting them "
                "would delete or overwrite another job's files. They were not "
                "resubmitted:\n\n"
                + "\n".join(refusals.values())
                + "\n\nDismiss or reset them instead.",
            )
            if not selected:
                return

        # Check what the submission needs before asking how to resubmit, so a
        # missing executable or cluster is never reported after the question.
        if any(job.execution_target == "local" for job in selected) and (
            not self._castep_executable_ready()
        ):
            selected = [job for job in selected if job.execution_target != "local"]
        if any(job.execution_target != "local" for job in selected) and (
            not self._clusters_configured()
        ):
            selected = [job for job in selected if job.execution_target == "local"]
        if not selected or not self._deps.keyword_catalog_guard().confirm_submission():
            return

        fresh_start_job_ids = self._choose_relaxation_resubmit_mode(selected)
        if fresh_start_job_ids is None:
            return

        local_jobs = [job for job in selected if job.execution_target == "local"]
        cluster_jobs = [job for job in selected if job.execution_target != "local"]

        if local_jobs:
            self.submit_jobs_locally(
                local_jobs,
                fresh_start_job_ids=fresh_start_job_ids,
                eligible_statuses=RESUBMITTABLE_JOB_STATES,
                no_eligible_message=(
                    "Selected jobs must be failed, stopped, or cancelled for local resubmission."
                ),
                success_message=lambda count: f"Resubmitted {plural(count, 'local job')}",
            )

        if cluster_jobs:
            preferred = None
            targets = {job.execution_target for job in cluster_jobs}
            if len(targets) == 1:
                preferred = next(iter(targets))
            self.submit_jobs_on_cluster(
                cluster_jobs,
                default_cluster_name=preferred,
                eligible_statuses=RESUBMITTABLE_JOB_STATES,
                no_eligible_message=(
                    "Selected jobs must be failed, stopped, or cancelled for cluster resubmission."
                ),
                reuse_existing_remote_files=True,
                fresh_start_job_ids=fresh_start_job_ids,
                success_message=lambda count, cluster: (
                    f"Resubmitted {plural(count, 'cluster job')} on {cluster}"
                ),
            )

    def _choose_relaxation_resubmit_mode(self, jobs: Sequence[Job]) -> set[UUID] | None:
        """Ask how resubmitted relaxations should treat their previous run.

        Returns the ids of the jobs to start again from a clean directory (empty
        when they continue, or when no relaxation is being resubmitted), or
        ``None`` when the user cancels.
        """

        relaxations = [job for job in jobs if job_resumes_from_checkpoint(job)]
        if not relaxations:
            return set()

        probes = self._checkpoint_probes(relaxations)
        dialog = ResubmitChoiceDialog(
            [
                ResubmitCandidate(
                    label=job.label,
                    reason=job.end_reason,
                    checkpoint=probes[job.id].status,
                )
                for job in relaxations
            ],
            other_job_count=len(jobs) - len(relaxations),
            parent=self._parent(),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        if dialog.mode() == ResubmitMode.START_AGAIN:
            return {job.id for job in relaxations}
        return set()

    def _fresh_start_refusals(self, jobs: Sequence[Job]) -> dict[UUID, str]:
        """Return, per job id, why starting that job again would be unsafe.

        Start again deletes a job's local output directory and its remote
        directory, so both must belong to that job alone (see
        :meth:`JobFileCleanup.directory_ownership_refusals`). Reconfiguring
        a relaxation row can leave an old failed job and its replacement
        sharing one directory, which is the case this exists for.
        """

        return self._job_file_cleanup.directory_ownership_refusals(jobs)

    def _clear_outputs_for_fresh_start(self, jobs: Sequence[Job]) -> tuple[list[Job], list[str]]:
        """Empty each job's local output directory, keeping its ``.cell``/``.param``.

        The same cleanup Reset offers. A job refused by
        :meth:`_fresh_start_refusals`, or whose directory cannot be fully
        cleaned, is left out of the returned list: submitting it would either
        delete another job's files (its remote directory is deleted next) or
        resume from a checkpoint the user asked to discard.
        """

        refusals = self._fresh_start_refusals(jobs)
        cleaned: list[Job] = []
        errors: list[str] = list(refusals.values())
        for job in jobs:
            if job.id in refusals:
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
        if self._document is None:
            return

        updates, requested = self._job_execution.check_now(job_ids)
        if not updates and not requested:
            QMessageBox.information(
                self._parent(),
                "No jobs to check",
                "Check now is available for cluster jobs with a remote folder while their "
                "cluster is connected, and for local jobs that did not complete.",
            )
            return
        completed = 0
        for update in updates:
            if update.job is not None and update.finished is not None:
                self._deps.on_local_job_finished(
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
            "reset": "_on_reset_failed_from_dashboard",
            "dismiss": "_on_dismiss_from_dashboard",
            "continue_relaxation": "_on_continue_relaxation_from_dashboard",
        }
        return getattr(self, names[kind])

    def _on_continue_relaxation_from_dashboard(self, job_ids: list[UUID]) -> None:
        """Open "Continue relaxation…" for one job, through its workflow row."""

        job = (
            self._document.jobs.get(job_ids[0]) if self._document is not None and job_ids else None
        )
        continue_relaxation = self.continue_relaxation_for_job(job) if job is not None else None
        if continue_relaxation is not None:
            continue_relaxation()

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

    def _on_reset_failed_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return
        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        self.reset_failed_jobs([job for job in jobs if job is not None])

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
            enabled, num_farms = self._deps.task_farming_for_job(job)
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
    ) -> None:
        """Submit ``jobs`` to the local runner after the local submission dialog.

        Jobs in ``fresh_start_job_ids`` (a relaxation the user chose to start
        again) have their previous output cleared first, once the dialog is
        accepted, so cancelling never deletes anything.
        """

        if self._document is None or not jobs:
            return

        allowed_statuses = eligible_statuses or {JobStatus.DEFINED}
        eligible_jobs = [job for job in dedupe_jobs(jobs) if job.status in allowed_statuses]
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
        if setup.mpirun_path:
            # Without mpirun the dialog pins one core, which is not a choice to keep.
            remember_local_cores(self._document.ui_state, config.num_cores)

        fresh_jobs = [job for job in eligible_jobs if job.id in fresh_start_job_ids]
        if fresh_jobs:
            cleaned, cleanup_errors = self._clear_outputs_for_fresh_start(fresh_jobs)
            if cleanup_errors:
                not_cleaned = {job.id for job in fresh_jobs} - {job.id for job in cleaned}
                eligible_jobs = [job for job in eligible_jobs if job.id not in not_cleaned]
                show_detailed_warning(
                    self._parent(),
                    "Previous output not cleared",
                    "These jobs were not resubmitted because their previous output, "
                    "including any checkpoint, could not be cleared:\n" + "\n".join(cleanup_errors),
                    informative_text="Close any program using these files, then resubmit.",
                )
                if not eligible_jobs:
                    return

        outcome = self._job_execution.submit_local_jobs(
            eligible_jobs,
            num_cores=config.num_cores,
            max_concurrent=config.max_concurrent if config.parallel else 1,
            castep_executable=setup.executable_path,
            mpirun_path=setup.mpirun_path,
            eligible_statuses=allowed_statuses,
            retrieval_for_job=self._deps.retrieval_flags_for_job,
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

    def submit_jobs_on_cluster(
        self,
        jobs: list[Job],
        *,
        default_cluster_name: str | None = None,
        eligible_statuses: Collection[JobStatus] | None = None,
        no_eligible_message: str = "Selected jobs must be ready for cluster submission.",
        reuse_existing_remote_files: bool = False,
        success_message: Callable[[int, str], str] | None = None,
        fresh_start_job_ids: Collection[UUID] = (),
    ) -> None:
        """Submit ``jobs`` to a cluster chosen in the cluster submission dialog.

        The decisions live in ``services.cluster_submission_planner``; this
        method shows the dialog, connects, asks the overwrite prompts the plan
        lists and reports the outcome. Jobs in ``fresh_start_job_ids`` start
        again: on their own cluster, their remote directory is deleted and
        their inputs uploaded afresh, without the overwrite prompt, and their
        local output is cleared as Reset would clear it.
        """

        if self._document is None or not jobs:
            return

        allowed_statuses = eligible_statuses or {JobStatus.DEFINED}
        eligible_jobs = [job for job in dedupe_jobs(jobs) if job.status in allowed_statuses]
        if not eligible_jobs:
            QMessageBox.information(self._parent(), "No eligible jobs", no_eligible_message)
            return
        eligible_jobs = self.without_jobs_in_another_jobs_folder(eligible_jobs)
        if not eligible_jobs:
            return

        chosen = self._choose_cluster_submission(eligible_jobs, default_cluster_name)
        if chosen is None:
            return
        submission, selected_cluster, dialog_credentials = chosen

        screened = self._screen_cluster_submission_jobs(eligible_jobs, selected_cluster)
        if screened is None:
            return
        eligible_jobs, seed_name_by_job = screened

        if not self._connect_for_cluster_submission(selected_cluster, dialog_credentials):
            return

        # The existence check runs on the worker thread; every prompt is answered
        # here, before the batch starts, so none interrupts it.
        plan = plan_cluster_submission(
            eligible_jobs,
            cluster=selected_cluster,
            submission=submission,
            seed_name_by_job=seed_name_by_job,
            remote_job_directory=self._cluster_runner.remote_job_directory,
            output_directory=self._deps.job_output_directory,
            retrieval_flags=self._deps.retrieval_flags_for_job,
            check_remote_directories=self._await_remote_dir_check,
            ask_overwrite=self._prompt_remote_directory_overwrite,
            confirm_bulk_delete=self._confirm_bulk_remote_delete,
            clear_fresh_outputs=self._clear_outputs_for_fresh_start,
            fresh_start_job_ids=fresh_start_job_ids,
            reuse_existing_remote_files=reuse_existing_remote_files,
        )
        failed_submit = list(plan.failed)
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

    def _choose_cluster_submission(
        self, eligible_jobs: list[Job], default_cluster_name: str | None
    ) -> tuple[ClusterSubmissionConfig, ClusterConfig, SSHCredentials | None] | None:
        """Show the cluster submission dialog and resolve the chosen cluster.

        Returns the accepted submission settings, the selected cluster (with the
        dialog's username applied) and the dialog's credentials, or ``None``
        after telling the user why when no cluster is configured, the keyword
        catalog guard declines, the dialog is cancelled or the chosen cluster
        has since been removed.
        """

        if not self._clusters_configured():
            return None
        if not self._deps.keyword_catalog_guard().confirm_submission():
            return None
        cluster_configs = load_cluster_configs(self._settings)

        workflow_type = cluster_workflow_directory_fragment(eligible_jobs)
        last_submission = self._load_last_cluster_submission_config(workflow_type=workflow_type)
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
        self._persist_last_cluster_submission_config(submission, workflow_type=workflow_type)
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
            self._deps.connect_to_cluster_with_trust(cluster, credentials=credentials)
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

    def _load_last_cluster_submission_config(
        self,
        *,
        workflow_type: str | None = None,
    ) -> ClusterSubmissionConfig | None:
        if self._document is None:
            return None

        payload = self._document.ui_state.get(self._LAST_CLUSTER_SUBMISSION_CONFIG_KEY, {})
        if not isinstance(payload, dict):
            return None

        cluster_name = str(payload.get("cluster_name", "")).strip()
        if not cluster_name:
            return None

        try:
            num_processes = max(1, int(payload.get("num_processes", 1)))
        except (TypeError, ValueError):
            num_processes = 1

        memory_raw = payload.get("memory_gb")
        memory_gb: float | None
        try:
            memory_gb = None if memory_raw in (None, "") else float(memory_raw)
        except (TypeError, ValueError):
            memory_gb = None

        mem_per_cpu_raw = payload.get("mem_per_cpu_gb")
        mem_per_cpu_gb: float | None
        try:
            mem_per_cpu_gb = None if mem_per_cpu_raw in (None, "") else float(mem_per_cpu_raw)
        except (TypeError, ValueError):
            mem_per_cpu_gb = None
        if mem_per_cpu_gb is not None and mem_per_cpu_gb <= 0:
            mem_per_cpu_gb = None

        # Saved before equal placement existed, or unreadable: the scheduler places.
        per_node_raw = payload.get("processes_per_node")
        try:
            processes_per_node = None if per_node_raw in (None, "") else int(per_node_raw)
        except (TypeError, ValueError):
            processes_per_node = None
        if processes_per_node is not None and processes_per_node < 1:
            processes_per_node = None

        persisted_workflow_type = str(payload.get("workflow_type", "")).strip()
        same_workflow = (
            not workflow_type
            or not persisted_workflow_type
            or workflow_type == persisted_workflow_type
        )

        return ClusterSubmissionConfig(
            cluster_name=cluster_name,
            working_directory=str(payload.get("working_directory", "")).strip()
            if same_workflow
            else "",
            username=str(payload.get("username", "")).strip(),
            num_processes=num_processes,
            walltime=str(payload.get("walltime", "")).strip(),
            queue_name=str(payload.get("queue_name", "")).strip(),
            memory_gb=memory_gb,
            mem_per_cpu_gb=mem_per_cpu_gb,
            comment=str(payload.get("comment", "")).strip(),
            processes_per_node=processes_per_node,
        )

    def _persist_last_cluster_submission_config(
        self,
        config: ClusterSubmissionConfig,
        *,
        workflow_type: str,
    ) -> None:
        if self._document is None:
            return
        self._document.ui_state[self._LAST_CLUSTER_SUBMISSION_CONFIG_KEY] = {
            "cluster_name": config.cluster_name,
            "working_directory": config.working_directory,
            "username": config.username,
            "num_processes": int(config.num_processes),
            "walltime": config.walltime,
            "queue_name": str(getattr(config, "queue_name", "")),
            "memory_gb": getattr(config, "memory_gb", None),
            "mem_per_cpu_gb": getattr(config, "mem_per_cpu_gb", None),
            "comment": str(getattr(config, "comment", "")),
            "processes_per_node": getattr(config, "processes_per_node", None),
            "workflow_type": workflow_type,
        }

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
            self._cluster_runner.abandon_batch()
            loop.quit()

        watchdog.timeout.connect(on_watchdog_timeout)

        def on_progress(index: int, count: int, label: str) -> None:
            progress.setLabelText(f"Submitting {label} ({index}/{count})")
            progress.setValue(index - 1)
            watchdog.start()

        def on_one(job_id: object, remote_job_id: object, error: str) -> None:
            if not isinstance(job_id, UUID):
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

        self._cluster_runner.submit_batch_progress.connect(on_progress)
        self._cluster_runner.job_submit_finished.connect(on_one)
        self._cluster_runner.submit_batch_finished.connect(loop.quit)
        progress.canceled.connect(self._cluster_runner.cancel_batch)
        progress.show()
        try:
            self._cluster_runner.submit_batch(submit_items)
            watchdog.start()
            loop.exec()
        finally:
            watchdog.stop()
            watchdog.deleteLater()
            self._cluster_runner.submit_batch_progress.disconnect(on_progress)
            self._cluster_runner.job_submit_finished.disconnect(on_one)
            self._cluster_runner.submit_batch_finished.disconnect(loop.quit)
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
                # active state.
                self._deps.on_cluster_job_cancelled(job.id)
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
            # Failed and stopped jobs can always be resubmitted.
            can_resubmit=True,
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

    def reset_failed_jobs(self, jobs: list[Job], *, cleanup_choice: bool | None = None) -> None:
        """Reset terminal jobs back to ``defined``.

        ``cleanup_choice`` is for a caller that has already confirmed the reset
        and asked whether to clean the job directories; left as ``None`` the
        user is asked here.
        """

        if self._document is None:
            return

        terminal_jobs = [job for job in jobs if job.status == JobStatus.ENDED]
        resettable_jobs = [job for job in terminal_jobs if not job_is_imported(job)]
        imported_count = len(terminal_jobs) - len(resettable_jobs)
        if imported_count:
            # Resetting an imported job would relaunch a calculation MuFinder
            # never ran, and its cleanup would delete the only copy of the
            # imported results and their archived originals.
            QMessageBox.information(
                self._parent(),
                "Imported jobs cannot be reset",
                f"{plural(imported_count, 'imported job was', 'imported jobs were')} skipped. They record calculations "
                "run outside MuFinder, and their directories hold the only copy of those "
                "results.",
            )
        if not resettable_jobs:
            QMessageBox.information(
                self._parent(),
                "No resettable jobs",
                "Only jobs in the 'failed', 'stopped', or 'cancelled' state can be reset.",
            )
            return

        if cleanup_choice is None:
            cleanup_choice = self._prompt_failed_job_cleanup(resettable_jobs)
        if cleanup_choice is None:
            return

        reset_count, cleanup_errors, folders = self._apply_failed_job_reset(
            resettable_jobs,
            cleanup_choice=cleanup_choice,
            detach_convergence=True,
        )

        self._deps.refresh_views_after_job_reset()

        cleaned, kept = folders
        if reset_count:
            self._deps.persist_document_ui_state(
                failure_message="Reset job state was recorded, but autosave failed.",
            )
            self._deps.status_bar().showMessage(
                folder_outcome_status(
                    f"Reset {plural(reset_count, 'failed job')}",
                    cleaned=len(cleaned),
                    kept=len(kept),
                )
            )
        self._deps.show_kept_folders(kept, title="Directory cleanup issues")

    def _apply_failed_job_reset(
        self,
        jobs: Sequence[Job],
        *,
        cleanup_choice: bool,
        detach_convergence: bool,
    ) -> tuple[int, list[str], tuple[list[Path], list[KeptFolder]]]:
        """Clean, reset, and record provenance for already-vetted terminal jobs.

        Shared core of the dashboard/table "Reset failed jobs" action and of the
        lazy reset the workflow-row submit cycle performs for ended jobs
        (:meth:`prepare_jobs_for_resubmission`). Callers own the "nothing to
        reset" dialog, the cleanup prompt, the view refreshes, the autosave and
        the status-bar message; this helper only mutates job state and returns
        ``(reset_count, cleanup_errors, (cleaned, kept))``, the last two naming the
        folders cleaned and those a refusal or an error kept.

        ``detach_convergence`` unbinds a reset cutoff-convergence job from its
        definition's ``job_ids_by_cutoff`` mapping. That is right for an explicit
        reset — the sweep point starts over from an empty row — but wrong for a
        resubmission, which must keep the row bound to the very job it is about
        to submit, or the row would project "ready" with no job and a second
        submit would mint a duplicate.
        """

        cleanup_errors: list[str] = []
        reset_count = 0
        # By place, since folders in different charge folders can share a name.
        cleaned_folders: dict[Path, None] = {}
        kept_folders: dict[Path | str, KeptFolder] = {}
        # Never clean a directory another job still depends on. Jobs reset
        # together may share one, and so may a job's directory companions (a
        # Quantum Corrections phonon job with the legacy singlepoint an older
        # project ran in its directory); anything else keeps its directory and
        # is reported.
        cleanable_jobs = [job for job in jobs if not job_is_imported(job)]
        cleanup_refusals = (
            self._job_file_cleanup.directory_ownership_refusals(
                cleanable_jobs,
                # Imported jobs are skipped below, never cleaned, so a folder
                # shared with one must not count as cleaned together.
                shared_with={job.id for job in cleanable_jobs}
                | {
                    companion.id
                    for job in cleanable_jobs
                    if (controller := owning_controller(self._deps.workflow_controllers(), job))
                    is not None
                    for companion in controller.directory_companions(job)
                },
                check_remote=False,
            )
            if cleanup_choice
            else {}
        )
        for job in jobs:
            if job_is_imported(job):
                # Defensive: this core is shared with the resubmission path, and
                # the directory cleanup below is irreversible.
                continue
            cleaned_output_directory = False
            if cleanup_choice and job.id in cleanup_refusals:
                cleanup_errors.append(f"{cleanup_refusals[job.id]}; its folder was kept")
                refused_dir = self._deps.job_output_directory(job)
                name = (
                    job_directory_label(self._document.project_root, refused_dir)
                    if refused_dir is not None
                    else job.label
                )
                refusal = cleanup_refusals[job.id].removeprefix(f"- {job.label}: ")
                kept_folders[refused_dir or job.label] = KeptFolder(name, refusal_because(refusal))
            elif cleanup_choice:
                output_dir = self._deps.job_output_directory(job)
                if output_dir is not None and output_dir.exists() and output_dir.is_dir():
                    try:
                        self._job_file_cleanup.clear_directory_contents(output_dir)
                        cleaned_output_directory = True
                        cleaned_folders[output_dir] = None
                    except OSError as exc:
                        cleanup_errors.append(f"- {job.label}: {exc}")
                        reason = plain_error(exc)
                        name = job_directory_label(self._document.project_root, output_dir)
                        kept_folders[output_dir] = KeptFolder(name, (reason, ""))

            try:
                self._project_service.reset_failed_job(job)
            except ValueError:
                continue

            if detach_convergence and job.workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF:
                self._deps.detach_convergence_job(job.id)

            self._active_run_by_job.pop(job.id, None)
            self._record_job_reset(job=job, cleaned_output_directory=cleaned_output_directory)
            reset_count += 1

        return reset_count, cleanup_errors, (list(cleaned_folders), list(kept_folders.values()))

    def prepare_jobs_for_resubmission(self, jobs: Sequence[Job]) -> bool:
        """Reset ended jobs so a workflow row can resubmit them.

        Bound into every controller's
        :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`
        and called from the shared submit cycle just before dispatch. A row
        whose job ended can be submitted: this runs the same stale-output
        cleanup prompt the Reset action uses, then resets the jobs back to
        ``defined`` so they clear the ``{DEFINED}`` gate in the submit backends.

        Returns ``False`` only when the user cancels the cleanup prompt, which
        aborts the submission without an error dialog. Jobs that have not ended,
        and imported jobs, are never touched here.
        """

        if self._document is None:
            return False

        resettable_jobs = [
            job for job in jobs if job.status == JobStatus.ENDED and not job_is_imported(job)
        ]
        if not resettable_jobs:
            return True

        cleanup_choice = self._prompt_failed_job_cleanup(resettable_jobs)
        if cleanup_choice is None:
            return False

        reset_count, _cleanup_errors, (_cleaned, kept) = self._apply_failed_job_reset(
            resettable_jobs,
            cleanup_choice=cleanup_choice,
            detach_convergence=False,
        )

        self._deps.refresh_views_after_job_reset()

        self._deps.show_kept_folders(kept, title="Directory cleanup issues")

        if reset_count:
            self._deps.persist_document_ui_state(
                failure_message="Reset job state was recorded, but autosave failed.",
            )
        return True

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

    @staticmethod
    def _resolve_continue_relaxation_seed_values(
        seed: Mapping[str, object],
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        """Resolve a Continue dialog seed to the values its editor will actually show.

        ``ContinueRelaxationDialog.__init__`` documents the contract precisely:
        ``old_param_text`` must be rendered from the exact values its own live
        "new" render produces when unedited, or the diff panel shows spurious
        rows for every basic-visible keyword that has no catalog
        ``default_value`` (``ParameterDialog`` still gives such a field's editor
        a sensible starting value — e.g. a dropdown defaults to its first
        allowed option — even though the row's stored/effective
        ``parameter_values`` never mentions it, so passing the seed straight to
        ``render_param_text`` would omit it from "old" while the dialog's own
        ``collect_values()`` includes it in "new"). This builds a throwaway
        ``ParameterDialog`` with the identical specs/workflow_stage/locked_values
        and the identical constructor seed (values, k-point choice, advanced
        names) → ``collect_values()`` sequence ``ContinueRelaxationDialog.__init__``
        runs internally, so the two agree keyword-for-keyword on an unedited row.
        """

        specs = reorder_workflow_parameter_specs(list(seed.get("specs", ())))
        parameter_values = dict(seed.get("parameter_values", {}))
        added_advanced = tuple(seed.get("added_advanced", ()))
        reference = ParameterDialog(
            specs=specs,
            workflow_stage=str(seed.get("workflow_stage", "")),
            locked_values=seed.get("locked_values"),
            species_options=seed.get("species_options", ()),
            values=dict(parameter_values),
            preferred_exclusive=kpoint_exclusive_hint([parameter_values]),
            advanced_names=added_advanced,
        )
        try:
            values = dict(reference.collect_values())
            values.pop("task", None)
            return values, reference.advanced_names()
        finally:
            reference.close()
            reference.deleteLater()

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
        row's job, probing for a checkpoint, rendering the old/new ``.param``
        diff text, opening the dialog, and — on accept — running the
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
        the submit cycle has no ended job to reset, so the cleanup prompt
        (``prepare_jobs_for_resubmission``) is
        never reached — Continue never asks about deleting the very directory
        it just preserved and rewrote.
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

        def render_param_text(values: Mapping[str, object], advanced: tuple[str, ...]) -> str:
            return controller.render_row_param_text(key, values, advanced)

        resolved_values, resolved_advanced = self._resolve_continue_relaxation_seed_values(seed)
        old_param_text = render_param_text(resolved_values, resolved_advanced)

        if (job.execution_target or "").strip().casefold() == "local":
            output_dir = self._deps.job_output_directory(job)
            reused_directory = str(output_dir) if output_dir is not None else job.label
        else:
            reused_directory = job.remote_directory or "(directory will be computed at submission)"

        dialog = ContinueRelaxationDialog(
            **seed,
            old_param_text=old_param_text,
            render_param_text=render_param_text,
            checkpoint_probe=checkpoint_probe,
            execution_target=job.execution_target,
            reused_directory=reused_directory,
            parent=self._parent(),
        )
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            parameter_values, added_advanced = dialog.edited_parameters()
        finally:
            # The dialog is a child of the window; without this every open
            # would stay alive under it, editors and all.
            dialog.deleteLater()
        continued_job = controller.continue_job(key, parameter_values, added_advanced)
        if continued_job is None:
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

    def _prompt_failed_job_cleanup(self, jobs: Sequence[Job]) -> bool | None:
        """Ask how to reset ``jobs``: clean their folders, keep them, or cancel.

        Shared by every Reset and by a workflow row's Submit of an ended job.
        It warns when a job's output may still be on its cluster.
        """

        detail = (
            "Choose 'Clean and reset' to clean each job directory before reset, "
            "'Reset, keep files' to keep existing files, or 'Cancel' to abort."
        )
        note = unretrieved_output_note(jobs, action="Resetting")
        answer = workflow_table_utils.confirm_choice(
            self._parent(),
            "Reset jobs",
            f"Reset {plural(len(jobs), 'job')} that did not finish so "
            f"{'it is' if len(jobs) == 1 else 'they are'} ready to run again?",
            [("clean", "Clean and reset"), ("keep", "Reset, keep files"), ("cancel", "Cancel")],
            default="keep",
            escape="cancel",
            detail=f"{note}\n\n{detail}" if note else detail,
            destructive=("clean",),
        )
        if answer == "cancel":
            return None
        return answer == "clean"

    def _record_job_reset(self, *, job: Job, cleaned_output_directory: bool) -> None:
        if self._document is None:
            return
        self._project_service.record_job_reset(
            self._document,
            job=job,
            cleaned_output_directory=cleaned_output_directory,
        )
        self._deps.refresh_journal_view()
