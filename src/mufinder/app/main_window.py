"""Primary application window for MuFinder."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, TypeVar
from uuid import UUID, uuid4

from PySide6.QtCore import (
    QByteArray,
    QSettings,
    QTimer,
    QUrl,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QDesktopServices,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QStyle,
    QWidget,
)

from mufinder._version import current_version
from mufinder.branding import (
    DOCUMENTATION_URL,
    PRODUCT_NAME,
    PRODUCT_TAGLINE,
    REPOSITORY_URL,
    app_icon,
)
from mufinder.cluster_setup import load_cluster_configs
from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import (
    MANUAL_DEFAULT_METHOD,
    CalculationRun,
    Job,
    Structure,
)
from mufinder.app.cluster_setup_dialog import ClusterSetupDialog
from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.comparison_job_dialog import run_comparison_job_dialog
from mufinder.app.comparison_table_widget import ComparisonTablesPage
from mufinder.app.blocking_call import run_blocking_call, wait_for_event
from mufinder.app.cluster_ui_utils import (
    show_detailed_warning,
)
from mufinder.app.castep_setup_dialog import CastepSetupDialog
from mufinder.app.keyword_catalog_guard import KeywordCatalogGuard
from mufinder.app.convergence_adapter import convergence_job_name
from mufinder.app.convergence_examiner import ConvergenceExaminerDialog
from mufinder.app.convergence_display import (
    format_cutoff_energy,
    format_default_criteria,
    format_kpoint_grid,
)
from mufinder.app.convergence_overview_widget import ConvergenceOverviewWidget
from mufinder.app.convergence_types import (
    SweepPointRunState,
)
from mufinder.app.convergence_widget import CutoffConvergenceWidget
from mufinder.app.energy_barriers_widget import (
    EnergyBarriersWidget,
)
from mufinder.app.kpoint_convergence_widget import KPointConvergenceWidget
from mufinder.domain.text import (
    KeptFolder,
    folder_outcome_status,
    kept_folders_message,
    plural,
)
from mufinder.app.relaxation_adapter import (
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    expand_muon_relaxation_jobs,
    expand_pristine_relaxation_jobs,
    muon_charge_state_label,
)
from mufinder.app.adapter_common import (
    IDENTITY_SUPERCELL,
    AdapterExpansion,
    KpointAdjustment,
    coerce_supercell_matrix,
    kpoint_adjustment_status,
)
from mufinder.app.relaxation_types import (
    MuonRelaxationConfig,
    PristineRelaxationConfig,
    SupercellMatrix,
)
from mufinder.app.relaxation_analysis_widget import (
    RelaxationAnalysisWidget,
)
from mufinder.app.relaxation_widget import PristineRelaxationWidget
from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget
from mufinder.app.muon_sites_summary_widget import MuonSitesSummaryWidget
from mufinder.app.uep_widget import UEPWidget
from mufinder.app.host_structure_widget import HostStructureWidget
from mufinder.app.host_switch import confirm_host_switch
from mufinder.app.icons import icon
from mufinder.app.job_dashboard import ClusterQueueStatus, JobDashboard, JobDashboardDeps
from mufinder.app.job_dashboard_tray import ClippablePageArea, JobDashboardTray
from mufinder.app.jobs_controller import JobsController, JobsControllerDeps
from mufinder.app.journal_widget import JournalWidget
from mufinder.app.magnetic_field_widget import MagneticFieldWidget
from mufinder.app.polarisation_widget import PolarisationWidget
from mufinder.app.muon_generation_widget import MuonGenerationWidget
from mufinder.app.new_project_dialog import NewProjectDialog
from mufinder.app.project_defaults_widget import ProjectDefaultsStrip, ProjectDefaultsSummary
from mufinder.app.file_dialogs import RememberedFileDialogs
from mufinder.app.home_page import HomePage
from mufinder.app.navigator_panel import NavigatorPanel
from mufinder.app.project_navigator import PROJECT_HOME_ROUTE, ProjectNavigator
from mufinder.app.hyperfine_coupling_widget import HyperfineCouplingWidget
from mufinder.app.recent_projects import RecentProjects, RecentProjectsMenu
from mufinder.app.quantum_corrections_widget import (
    QuantumCorrectionsWidget,
)
from mufinder.app.screen_utils import clamp_to_screen, clamped_size
from mufinder.app.supercell_dialog import SupercellEditorDialog
from mufinder.app.workflow_row_widget import RowTableWidget
from mufinder.app.theme import (
    THEME_DARK,
    THEME_LIGHT,
    THEME_SYSTEM,
    apply_theme,
    load_theme_mode,
    save_theme_mode,
)
from mufinder.app.workflow_config_codecs import (
    deserialize_convergence_config,
    deserialize_muon_relaxation_config,
    deserialize_pristine_relaxation_config,
    muon_relaxation_row_key,
    serialize_muon_relaxation_config,
    serialize_pristine_relaxation_config,
    stored_row_keys,
)
from mufinder.app.workflow_parameters import (
    structure_species_options,
)
from mufinder.app.workflow_submission import (
    WorkflowSubmissionCoordinator,
)
from mufinder.app.workflow_controllers.analysis_workspaces import (
    AnalysisWorkspacesController,
    AnalysisWorkspacesDeps,
)
from mufinder.app.workflow_controllers.site_generation import (
    SiteGenerationController,
    SiteGenerationDeps,
)
from mufinder.app.workflow_controllers.comparison_tables import (
    ComparisonTablesController,
    ComparisonTablesDeps,
    comparison_table_id_from_route,
)
from mufinder.app.workflow_controllers import (
    ConvergenceCutoffController,
    ConvergenceCutoffDeps,
    ConvergenceKpointController,
    ConvergenceKpointDeps,
    EnergyBarriersController,
    EnergyBarriersDeps,
    MuonRelaxationController,
    MuonRelaxationDeps,
    PristineRelaxationController,
    HyperfineCouplingController,
    HyperfineCouplingDeps,
    ImportedRelaxationsController,
    ImportedRelaxationsDeps,
    QuantumCorrectionsController,
    QuantumCorrectionsDeps,
    UepController,
    UepDeps,
    WorkflowControllerContext,
    WorkflowJobController,
    all_planned_jobs,
    dedupe_jobs,
    job_folder,
)
from mufinder.app.cluster_controller import ClusterController
from mufinder.services.job_execution_controller import (
    JobExecutionController,
)
from mufinder.services.convergence_defaults import (
    CUTOFF_SNAPSHOT,
    KPOINT_SNAPSHOT,
    store_convergence_overview_snapshot,
    validated_cutoff_overview_snapshot,
    validated_kpoint_overview_snapshot,
)
from mufinder.services.definition_mappings import (
    sweep_definition_context_matches,
)
from mufinder.services.job_file_cleanup import (
    JobCleanupItem,
    JobCleanupOutcome,
    JobFileCleanup,
)
from mufinder.services.job_paths import (
    definition_output_directory,
    job_seed_name,
)
from mufinder.services.job_results import (
    max_force_for_job,
    max_stress_for_job,
    total_energy_for_job,
)
from mufinder.services.job_progress import JobProgressMonitor
from mufinder.services.local_runner import LocalJobRunner
from mufinder.app import workflow_table_utils
from mufinder.services.hyperfine_results import (
    apply_magnetic_field_handoff,
    plan_magnetic_field_handoff,
)
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    attach_magnetic_field_results,
    remove_magnetic_field_results,
    store_magnetic_field_results,
)
from mufinder.services.generation_runs import generation_run_groups
from mufinder.domain.routes import (
    magnetic_field_workspace_id_from_route,
    magnetic_field_workspace_route,
    relaxation_analysis_workspace_id_from_route,
    relaxation_analysis_workspace_route,
)
from mufinder.services.uep_results import (
    UEP_ANALYSIS_STATES_KEY,
    uep_cluster_source_rows,
)
from mufinder.services.analysis_workspaces import analysis_workspaces
from mufinder.services.muon_sites_summary import (
    validated_muon_sites_summary_snapshot,
)
from mufinder.services.perf_trace import PerfTrace, StageTiming
from mufinder.services.site_generation_service import (
    uep_site_coordinate_warning,
)
from mufinder.services.project_service import (
    ProjectDocument,
    ProjectService,
    UnsavedChanges,
)
from mufinder.services.project_summary import (
    job_stage_key,
    job_stage_name,
    summarize_project,
)
from mufinder.services.source_candidates import (
    CompletedRelaxationAnalysisSource,
    MuonRelaxationRowJob,
    completed_muon_relaxation_sources,
    newest_completed_structure_id,
    quantum_seed_name_for_key,
)
from mufinder.services.structure_export import relaxed_structure_export_for_job_id
from mufinder.science.hyperfine import HyperfineError
from mufinder.science.structure_io import StructureLoadError
from mufinder.workflow import StructureBridgeError


_LOG = logging.getLogger(__name__)

_WorkflowKey = TypeVar("_WorkflowKey")
_STRUCTURE_CONTROLLED_CELL_KEYWORDS = {
    "lattice_cart",
    "lattice_abc",
    "positions_frac",
    "positions_abs",
}


def _ordered_string_list(payload: object) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, str):
        values = [payload]
    elif isinstance(payload, (list, tuple)):
        values = list(payload)
    else:
        values = [payload]
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


class MainWindow(QMainWindow):
    """Main shell containing project navigation and job dashboard."""

    _MAIN_WINDOW_GEOMETRY_KEY = "ui/main_window_geometry"
    _MAIN_WINDOW_STATE_KEY = "ui/main_window_state"
    # Quiet-period debounce for dashboard flushes: an isolated lifecycle event
    # (a single job completing with nothing else nearby) flushes this long
    # after the last request, so it still feels prompt.
    _DASHBOARD_FLUSH_DEBOUNCE_MS = 300
    # Floor on the gap between two flushes during a burst of lifecycle events
    # (e.g. many cluster jobs completing across a serial retrieval). Without
    # this floor, a steady stream of events spaced further apart than the
    # debounce (but closer than this) would re-arm the debounce forever and
    # never flush until the stream stopped; field traces showed 15 flushes in
    # 80s. With the floor, the burst still flushes on a bounded ~2s cadence.
    _DASHBOARD_MIN_FLUSH_INTERVAL_MS = 2000
    _ACTIVE_CONVERGENCE_DEFINITION_KEY = ConvergenceCutoffController.active_key
    _ACTIVE_KPOINT_DEFINITION_KEY = ConvergenceKpointController.active_key
    _ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY = PristineRelaxationController.active_key
    _ACTIVE_MUON_RELAXATION_DEFINITION_KEY = MuonRelaxationController.active_key
    _ACTIVE_UEP_DEFINITION_KEY = UepController.active_key
    _UEP_ANALYSIS_STATES_KEY = UEP_ANALYSIS_STATES_KEY
    _MAGNETIC_FIELD_FORM_KEY = "magnetic_field_form"
    _POLARISATION_FORM_KEY = "polarisation_form"
    _MAGNETIC_FIELD_WORKSPACES_KEY = "magnetic_field_workspaces"

    def __init__(self) -> None:
        super().__init__()
        self._project_service = ProjectService()
        self._document: ProjectDocument | None = None
        self._kpoint_adjustments: tuple[KpointAdjustment, ...] = ()
        self._kpoint_status_note = ""
        self._jobs_dock_placed = False
        # View > Job Dashboard's setting: the dock shows while a project is open and
        # this is on. The saved window state brings it back after a restart.
        self._jobs_dock_wanted = True
        # A save while closing failed, so the next close asks rather than retries.
        self._close_save_failed = False
        self._settings = QSettings("MuFinder", "MuFinder")
        self._appearance_mode = load_theme_mode(self._settings)
        self._local_runner = LocalJobRunner(self)
        # Cluster SSH work runs on a dedicated worker thread via this facade,
        # which mirrors ClusterJobRunner's public surface (see cluster_controller).
        self._cluster_runner = ClusterController(self, settings=self._settings)
        self._job_progress = JobProgressMonitor(
            local_output_paths=self._local_runner.output_candidates,
            request_remote_progress=lambda job_ids: self._job_execution.read_cluster_progress(
                job_ids
            ),
            parent=self,
        )
        self._cluster_runner.geometry_progress_read.connect(
            self._job_progress.accept_remote_progress
        )
        self._active_run_by_job: dict[UUID, UUID] = {}
        self._job_execution = JobExecutionController(
            document_getter=lambda: self._document,
            project_service=self._project_service,
            local_runner=self._local_runner,
            cluster_runner=self._cluster_runner,
            job_output_directory=lambda job: self._job_output_directory(job),
            job_seed_name=lambda job: self._job_seed_name(job),
            active_run_by_job=self._active_run_by_job,
            retrieval_for_job=lambda job: self._jobs.retrieval_flags_for_job(job),
            remote_directory_for_job=lambda job: self._jobs.infer_remote_job_directory(job),
        )
        self._job_file_cleanup = JobFileCleanup(
            document_getter=lambda: self._document,
            job_output_directory=lambda job: self._job_output_directory(job),
            job_seed_name=lambda job: self._job_seed_name(job),
            active_run_by_job=self._active_run_by_job,
            run_file_op=lambda op, description: self._run_blocking_file_op(
                op, description=description
            ),
            planned_jobs=lambda: all_planned_jobs(self._workflow_controllers),
        )
        self._workflow_submission = WorkflowSubmissionCoordinator(dedupe_jobs=self._dedupe_jobs)
        self._cluster_queue_status = ClusterQueueStatus(self)
        self._jobs = JobsController(
            self._build_jobs_controller_deps(),
            project_service=self._project_service,
            settings=self._settings,
            job_execution=self._job_execution,
            local_runner=self._local_runner,
            cluster_runner=self._cluster_runner,
            job_file_cleanup=self._job_file_cleanup,
            job_progress=self._job_progress,
            active_run_by_job=self._active_run_by_job,
            cluster_status=self._cluster_queue_status,
            parent=self,
        )
        # Job-lifecycle handlers (completion/status/cancel) coalesce their heavy
        # dashboard work — the full-document autosave AND the journal/jobs-table/
        # workflow-view rebuilds — through this timer. A burst of cluster jobs
        # finishing in one poll otherwise rebuilds every view and rewrites the
        # whole document to disk once per job, which freezes the GUI.
        self._dashboard_flush_pending = False
        self._dashboard_autosave_failure_message = ""
        # Monotonic timestamp (time.monotonic(), never wall-clock) of the last
        # completed flush, and of the deadline currently armed on
        # _dashboard_flush_timer. Both None until the first request. Used by
        # _request_dashboard_flush to enforce _DASHBOARD_MIN_FLUSH_INTERVAL_MS
        # without re-arming (and thereby postponing) an already-pending deadline.
        self._dashboard_last_flush_at: float | None = None
        self._dashboard_flush_deadline: float | None = None
        self._dashboard_flush_timer = QTimer(self)
        self._dashboard_flush_timer.setSingleShot(True)
        self._dashboard_flush_timer.setInterval(self._DASHBOARD_FLUSH_DEBOUNCE_MS)
        self._dashboard_flush_timer.timeout.connect(self._flush_dashboard)
        self._shutting_down = False
        self._recent_projects = RecentProjects(self._settings, self)
        self._file_dialogs = RememberedFileDialogs(self._settings, self)
        self._magnetic_field_workspace_widgets: dict[str, MagneticFieldWidget] = {}

        app = QApplication.instance()
        if app is not None:
            apply_theme(app, self._appearance_mode)

        self.setWindowTitle("MuFinder")
        self.setWindowIcon(app_icon())
        # Tall enough for the tallest page and about five Job Dashboard rows.
        self.resize(clamped_size(self, 1280, 960))
        saved_geometry = self._settings.value(self._MAIN_WINDOW_GEOMETRY_KEY, QByteArray())
        if (
            isinstance(saved_geometry, QByteArray)
            and not saved_geometry.isEmpty()
            and self.restoreGeometry(saved_geometry)
        ):
            clamp_to_screen(self)

        self._project_navigator = ProjectNavigator()
        self._project_navigator.route_selected.connect(self._show_route)
        self._project_navigator.home_requested.connect(self._show_project_home)
        self._project_navigator.delete_magnetic_workspace_requested.connect(
            self._delete_magnetic_field_workspace
        )

        self._main_stack = QStackedWidget()
        self._home_page = HomePage(self._recent_projects, self)
        self._home_page.new_requested.connect(self._new_project)
        self._home_page.open_requested.connect(self._open_project)
        self._home_page.recent_requested.connect(self._open_recent_project)
        self._home_page.route_requested.connect(self._set_active_route)
        self._host_structure_view = HostStructureWidget()
        self._host_structure_view.load_requested.connect(self._on_load_structure_requested)
        self._host_structure_view.magnetic_configuration_changed.connect(
            self._on_host_magnetic_configuration_changed
        )
        self._host_structure_view.supercell_editor_requested.connect(
            self._on_host_supercell_editor_requested
        )
        self._host_structure_view.active_host_change_requested.connect(
            self._on_active_host_change_requested
        )
        self._convergence_overview_view = ConvergenceOverviewWidget()
        self._convergence_overview_view.open_cutoff_requested.connect(
            lambda: self._set_active_route("convergence_cutoff")
        )
        self._convergence_overview_view.open_kpoint_requested.connect(
            lambda: self._set_active_route("convergence_kpoint")
        )
        self._convergence_view = CutoffConvergenceWidget()
        self._convergence_view.configure_requested.connect(
            lambda config: self._convergence_controller.configure(config)
        )
        self._convergence_view.examine_requested.connect(
            lambda: self._convergence_controller.examine()
        )
        self._convergence_view.form_state_changed.connect(self._on_convergence_form_state_changed)
        self._convergence_view.simulation_cell_create_requested.connect(
            lambda: self._convergence_controller.create_simulation_cell()
        )
        self._convergence_view.simulation_cell_selection_requested.connect(
            lambda payload: self._convergence_controller.select_simulation_cell(payload)
        )
        self._convergence_view.simulation_cell_use_unit_cell_requested.connect(
            lambda: self._convergence_controller.use_unit_cell()
        )
        # Per-workflow controllers own their submit → materialize → result →
        # refresh cycle. Built before signal wiring and before any job-completion
        # handler can fire (see _on_local_job_finished, which consults the
        # registry). Each migration PR adds one entry; cutoff convergence is the
        # first. See mufinder.app.workflow_controllers.
        self._convergence_controller = ConvergenceCutoffController(
            self._build_workflow_controller_context(),
            self._build_convergence_cutoff_deps(),
            self._convergence_view,
        )
        self._workflow_controllers: list[WorkflowJobController] = [
            self._convergence_controller,
        ]
        self._convergence_view.submit_local_requested.connect(
            lambda values: self._convergence_controller.submit(values, target="local")
        )
        self._convergence_view.submit_cluster_requested.connect(
            lambda values: self._convergence_controller.submit(values, target="cluster")
        )
        self._connect_workflow_row_actions(self._convergence_view, self._convergence_controller)
        self._convergence_view.remove_cutoff_requested.connect(
            self._on_remove_cutoffs_from_convergence
        )
        self._kpoint_convergence_view = KPointConvergenceWidget()
        self._kpoint_convergence_view.configure_requested.connect(
            lambda config: self._kpoint_convergence_controller.configure(config)
        )
        self._kpoint_convergence_view.examine_requested.connect(
            lambda: self._kpoint_convergence_controller.examine()
        )
        self._kpoint_convergence_view.form_state_changed.connect(self._on_kpoint_form_state_changed)
        self._kpoint_convergence_view.simulation_cell_create_requested.connect(
            lambda: self._kpoint_convergence_controller.create_simulation_cell()
        )
        self._kpoint_convergence_view.simulation_cell_selection_requested.connect(
            lambda payload: self._kpoint_convergence_controller.select_simulation_cell(payload)
        )
        self._kpoint_convergence_view.simulation_cell_use_unit_cell_requested.connect(
            lambda: self._kpoint_convergence_controller.use_unit_cell()
        )
        # K-point convergence controller: sibling of the cutoff controller. Built
        # here (after its view exists) and appended to the registry so completed
        # castep_convergence_kpoint jobs route through its build_result.
        self._kpoint_convergence_controller = ConvergenceKpointController(
            self._build_workflow_controller_context(),
            self._build_convergence_kpoint_deps(),
            self._kpoint_convergence_view,
        )
        self._workflow_controllers.append(self._kpoint_convergence_controller)
        self._kpoint_convergence_view.submit_local_requested.connect(
            lambda values: self._kpoint_convergence_controller.submit(values, target="local")
        )
        self._kpoint_convergence_view.submit_cluster_requested.connect(
            lambda values: self._kpoint_convergence_controller.submit(values, target="cluster")
        )
        self._connect_workflow_row_actions(
            self._kpoint_convergence_view, self._kpoint_convergence_controller
        )
        self._kpoint_convergence_view.remove_grid_requested.connect(
            self._on_remove_grids_from_kpoint_convergence
        )
        self._pristine_relaxation_view = PristineRelaxationWidget()
        self._pristine_relaxation_view.configure_requested.connect(
            self._on_pristine_relaxation_configured
        )
        self._pristine_relaxation_view.form_state_changed.connect(
            self._on_pristine_relaxation_form_state_changed
        )
        self._pristine_relaxation_view.row_supercell_edit_requested.connect(
            self._on_pristine_relaxation_supercell_edit_requested
        )
        # Pristine-relaxation controller: the first migration with real
        # build_result logic (relaxed lattice payload + derived structure). Built
        # here (after its view exists) and appended to the registry so completed
        # castep_pristine_relaxation jobs route through its validate_artifacts
        # and build_result.
        self._pristine_relaxation_controller = PristineRelaxationController(
            self._build_workflow_controller_context(),
            self._pristine_relaxation_view,
        )
        self._workflow_controllers.append(self._pristine_relaxation_controller)
        self._pristine_relaxation_view.submit_local_requested.connect(
            lambda values: self._pristine_relaxation_controller.submit(values, target="local")
        )
        self._pristine_relaxation_view.submit_cluster_requested.connect(
            lambda values: self._pristine_relaxation_controller.submit(values, target="cluster")
        )
        self._comparison_tables_view = ComparisonTablesPage()
        self._comparison_tables_controller = ComparisonTablesController(
            self._build_workflow_controller_context(),
            self._build_comparison_tables_deps(),
            self._comparison_tables_view,
        )
        self._comparison_tables_view.delete_requested.connect(
            self._comparison_tables_controller.delete_table
        )
        self._comparison_tables_view.title_changed.connect(
            self._comparison_tables_controller.rename_table
        )
        self._pristine_relaxation_view.generate_comparison_requested.connect(
            self._comparison_tables_controller.generate_table
        )
        self._pristine_relaxation_view.remove_requested.connect(
            self._on_remove_pristine_relaxation_rows
        )
        self._connect_workflow_row_actions(
            self._pristine_relaxation_view, self._pristine_relaxation_controller
        )
        self._pristine_relaxation_view.set_continuable_predicate(
            self._pristine_relaxation_controller.row_is_continuable
        )
        self._pristine_relaxation_view.set_relaxed_structure_export_provider(
            self._pristine_relaxation_controller.relaxed_structure_export
        )
        self._pristine_relaxation_view.continue_relaxation_requested.connect(
            self._jobs.on_continue_relaxation_requested_from_pristine_relaxation
        )
        self._uep_view = UEPWidget()
        self._uep_view.row_supercell_edit_requested.connect(self._on_uep_supercell_edit_requested)
        # UEP controller: one of the simpler migrations — no pre_submit_check, no
        # submission_groups override, and a small build_result contributing only
        # the ``density_available`` flag (no derived structures, no document
        # mutation). Built here (after its view exists) and appended to the
        # registry so completed castep_uep_source jobs route through its
        # validate_artifacts (the den_fmt missing-output failure) and build_result.
        self._uep_controller = UepController(
            self._build_workflow_controller_context(),
            self._build_uep_deps(),
            self._uep_view,
        )
        self._workflow_controllers.append(self._uep_controller)
        self._uep_view.configure_requested.connect(self._uep_controller.configure)
        self._uep_view.form_state_changed.connect(self._uep_controller.store_form_state)
        self._uep_view.source_structure_selection_requested.connect(
            self._uep_controller.select_source_structure
        )
        self._uep_view.wizard_requested.connect(self._uep_controller.open_analysis)
        self._uep_view.submit_local_requested.connect(
            lambda values: self._uep_controller.submit(values, target="local")
        )
        self._uep_view.submit_cluster_requested.connect(
            lambda values: self._uep_controller.submit(values, target="cluster")
        )
        self._connect_workflow_row_actions(self._uep_view, self._uep_controller)
        self._uep_view.remove_requested.connect(
            lambda payload: self._uep_controller.remove_labelled_rows(
                self._removed_row_labels(payload)
            )
        )
        self._muon_generation_view = MuonGenerationWidget()
        self._site_generation_controller = SiteGenerationController(
            self._build_workflow_controller_context(),
            self._build_site_generation_deps(),
            self._muon_generation_view,
        )
        generation = self._site_generation_controller
        self._muon_generation_view.generate_requested.connect(generation.generate)
        self._muon_generation_view.cancel_generation_requested.connect(generation.cancel_generation)
        self._muon_generation_view.cluster_positions_requested.connect(generation.cluster_positions)
        self._muon_generation_view.delete_candidates_requested.connect(generation.delete_sites)
        self._muon_generation_view.form_state_changed.connect(generation.store_form_state)
        self._import_relaxations_action = None
        self._remove_imported_action = None
        self._muon_relaxation_view = MuonRelaxationWidget()
        self._muon_relaxation_view.configure_requested.connect(self._on_muon_relaxation_configured)
        self._muon_relaxation_view.form_state_changed.connect(
            self._on_muon_relaxation_form_state_changed
        )
        for strip in self._project_defaults_strips():
            strip.xc_functional_change_requested.connect(
                self._on_project_xc_functional_change_requested
            )
            strip.cutoff_energy_change_requested.connect(self._on_project_cutoff_change_requested)
            strip.kpoint_grid_change_requested.connect(
                self._on_project_kpoint_grid_change_requested
            )
        self._refresh_convergence_project_defaults()
        self._muon_relaxation_view.row_supercell_edit_requested.connect(
            self._on_muon_relaxation_supercell_edit_requested
        )
        # Muon-relaxation controller: the first migration exercising the
        # pre_submit_check (staged-site/host validation) and submission_groups
        # (per-charge-state grouped submission) hooks, plus the richest
        # build_result so far (muon position payload, derived structure, and
        # source-site mutation). Built here (after its view exists) and appended
        # to the registry so completed muon-relaxation jobs route through its
        # build_result.
        self._muon_relaxation_controller = MuonRelaxationController(
            self._build_workflow_controller_context(),
            self._build_muon_relaxation_deps(),
            self._muon_relaxation_view,
        )
        self._workflow_controllers.append(self._muon_relaxation_controller)
        self._muon_generation_view.create_jobs_requested.connect(
            self._muon_relaxation_controller.stage_sites
        )
        self._muon_relaxation_view.submit_local_requested.connect(
            lambda values: self._muon_relaxation_controller.submit(values, target="local")
        )
        self._muon_relaxation_view.submit_cluster_requested.connect(
            lambda values: self._muon_relaxation_controller.submit(values, target="cluster")
        )
        self._muon_relaxation_view.remove_requested.connect(self._on_remove_muon_relaxation_rows)
        self._connect_workflow_row_actions(
            self._muon_relaxation_view, self._muon_relaxation_controller
        )
        self._muon_relaxation_view.set_continuable_predicate(
            self._muon_relaxation_controller.row_is_continuable
        )
        self._muon_relaxation_view.set_relaxed_structure_export_provider(
            self._muon_relaxation_controller.relaxed_structure_export
        )
        self._muon_relaxation_view.continue_relaxation_requested.connect(
            self._jobs.on_continue_relaxation_requested_from_muon_relaxation
        )
        self._imported_relaxations_controller = ImportedRelaxationsController(
            self._build_workflow_controller_context(),
            self._build_imported_relaxations_deps(),
            self._muon_relaxation_view,
        )
        self._relaxation_analysis_view = RelaxationAnalysisWidget()
        # The page is rebuilt only while it is showing; see
        # _refresh_relaxation_analysis_context.
        self._relaxation_analysis_context_stale = False
        self._deferring_relaxation_analysis_refresh = False
        self._relaxation_analysis_view.set_relaxed_structure_export_provider(
            lambda job_id: relaxed_structure_export_for_job_id(self._document, job_id)
        )
        self._analysis_workspaces_controller = AnalysisWorkspacesController(
            self._build_workflow_controller_context(),
            self._build_analysis_workspaces_deps(),
            self._relaxation_analysis_view,
        )
        workspaces = self._analysis_workspaces_controller
        self._muon_relaxation_view.analysis_workspace_requested.connect(workspaces.create_workspace)
        self._project_navigator.delete_analysis_requested.connect(workspaces.delete_workspace)
        self._relaxation_analysis_view.state_changed.connect(workspaces.store_workspaces)
        self._relaxation_analysis_view.add_sites_requested.connect(workspaces.add_relaxations)
        self._relaxation_analysis_view.pin_summary_requested.connect(workspaces.pin_to_summary)
        self._relaxation_analysis_view.grouping_applied.connect(workspaces.record_grouping)
        self._relaxation_analysis_view.delete_requested.connect(workspaces.delete_workspace)
        self._relaxation_analysis_view.current_workspace_changed.connect(
            self._on_relaxation_analysis_current_workspace_changed
        )
        self._muon_sites_summary_view = MuonSitesSummaryWidget()
        self._muon_sites_summary_view.open_analysis_requested.connect(
            workspaces.open_pinned_analysis
        )
        self._muon_sites_summary_view.open_analysis_root_requested.connect(
            lambda: self._set_active_route("muon_analysis")
        )
        self._muon_sites_summary_view.focus_site_requested.connect(workspaces.focus_pinned_site)
        self._muon_sites_summary_view.unpin_requested.connect(workspaces.unpin)
        self._magnetic_field_view = MagneticFieldWidget()
        self._magnetic_field_view.magnetic_structure_changed.connect(
            self._on_magnetic_field_structure_changed
        )
        self._magnetic_field_view.form_state_changed.connect(
            self._on_magnetic_field_form_state_changed
        )
        self._magnetic_field_view.pin_requested.connect(self._on_magnetic_field_pin_requested)
        self._magnetic_field_view.open_muon_sites_requested.connect(
            lambda: self._set_active_route("muon_sites")
        )
        self._magnetic_field_view.open_polarisation_requested.connect(
            lambda: self._open_polarisation_for(MAGNETIC_FIELD_PAGE_OWNER)
        )
        self._polarisation_view = PolarisationWidget()
        self._polarisation_view.form_state_changed.connect(self._on_polarisation_form_state_changed)
        self._polarisation_view.simulations_changed.connect(
            self._on_polarisation_simulations_changed
        )
        self._polarisation_view.open_magnetic_fields_requested.connect(
            lambda: self._set_active_route("magnetic_fields")
        )
        self._hyperfine_coupling_view = HyperfineCouplingWidget()
        self._hyperfine_coupling_view.source_label_renamed.connect(
            self._on_hyperfine_coupling_source_label_renamed
        )
        # Hyperfine-coupling controller: one magres job per relaxed site, in the
        # result-dispatch registry so completed castep_hyperfine jobs route
        # through its validate_artifacts and build_result.
        self._hyperfine_coupling_controller = HyperfineCouplingController(
            self._build_workflow_controller_context(),
            self._build_hyperfine_coupling_deps(),
            self._hyperfine_coupling_view,
        )
        self._workflow_controllers.append(self._hyperfine_coupling_controller)
        self._hyperfine_coupling_view.configure_requested.connect(
            self._hyperfine_coupling_controller.configure_sources
        )
        self._hyperfine_coupling_view.submit_local_requested.connect(
            lambda values: self._hyperfine_coupling_controller.submit(values, target="local")
        )
        self._hyperfine_coupling_view.submit_cluster_requested.connect(
            lambda values: self._hyperfine_coupling_controller.submit(values, target="cluster")
        )
        self._hyperfine_coupling_view.show_every_site_toggled.connect(
            self._hyperfine_coupling_controller.set_show_every_site
        )
        self._hyperfine_coupling_view.open_muon_sites_requested.connect(
            lambda: self._set_active_route("muon_sites")
        )
        self._connect_workflow_row_actions(
            self._hyperfine_coupling_view, self._hyperfine_coupling_controller
        )
        self._hyperfine_coupling_view.use_in_magnetic_fields_requested.connect(
            self._on_use_hyperfine_in_magnetic_fields
        )
        self._quantum_corrections_view = QuantumCorrectionsWidget()
        # Quantum-corrections controller: one phonon job per relaxed site. Built
        # here (after its view exists) and appended to the registry so completed
        # castep_quantum_phonon jobs (and an older project's
        # castep_quantum_singlepoint jobs) route through its validate_artifacts
        # and build_result; task farming stays in the shared submit backend.
        self._quantum_corrections_controller = QuantumCorrectionsController(
            self._build_workflow_controller_context(),
            self._build_quantum_corrections_deps(),
            self._quantum_corrections_view,
        )
        self._workflow_controllers.append(self._quantum_corrections_controller)
        self._quantum_corrections_view.submit_local_requested.connect(
            lambda values: self._quantum_corrections_controller.submit(values, target="local")
        )
        self._quantum_corrections_view.submit_cluster_requested.connect(
            lambda values: self._quantum_corrections_controller.submit(values, target="cluster")
        )
        self._quantum_corrections_view.show_every_site_toggled.connect(
            self._quantum_corrections_controller.set_show_every_site
        )
        self._quantum_corrections_view.open_muon_sites_requested.connect(
            lambda: self._set_active_route("muon_sites")
        )
        self._connect_workflow_row_actions(
            self._quantum_corrections_view, self._quantum_corrections_controller
        )
        self._quantum_corrections_view.configure_requested.connect(
            self._quantum_corrections_controller.configure_sources
        )
        self._quantum_corrections_view.source_label_renamed.connect(
            self._quantum_corrections_controller.rename_source
        )
        self._quantum_corrections_view.analysis_requested.connect(
            self._quantum_corrections_controller.open_analysis
        )
        self._quantum_corrections_view.open_energy_barriers_requested.connect(
            lambda: self._set_active_route("energy_barriers")
        )
        self._energy_barriers_view: EnergyBarriersWidget | None = None
        # Energy-barriers (NEB) controller. Its view is created lazily (only when
        # the workflow route is first opened), so the controller takes a
        # ``view_getter`` rather than a view instance. It is still constructed and
        # registered here so a completed NEB job dispatches through its
        # build_result even when the view was never opened; refresh_view no-ops
        # until the view exists. Its validate_artifacts fails a job whose ``.ts``
        # is missing or unparseable. The submit-signal wiring happens at lazy view creation
        # (``_ensure_energy_barriers_view``).
        self._energy_barriers_controller = EnergyBarriersController(
            self._build_workflow_controller_context(),
            self._build_energy_barriers_deps(),
            lambda: self._energy_barriers_view,
        )
        self._workflow_controllers.append(self._energy_barriers_controller)
        self._journal_view = JournalWidget()
        self._journal_view.set_route_checker(lambda route: self._project_navigator.has_route(route))
        self._journal_view.open_route_requested.connect(self.show_page)
        self._main_stack.addWidget(self._home_page)
        self._main_stack.addWidget(self._host_structure_view)
        self._main_stack.addWidget(self._convergence_overview_view)
        self._main_stack.addWidget(self._convergence_view)
        self._main_stack.addWidget(self._kpoint_convergence_view)
        self._main_stack.addWidget(self._pristine_relaxation_view)
        self._main_stack.addWidget(self._uep_view)
        self._main_stack.addWidget(self._muon_generation_view)
        self._main_stack.addWidget(self._muon_relaxation_view)
        self._main_stack.addWidget(self._relaxation_analysis_view)
        self._main_stack.addWidget(self._comparison_tables_view)
        self._main_stack.addWidget(self._muon_sites_summary_view)
        self._main_stack.addWidget(self._hyperfine_coupling_view)
        self._main_stack.addWidget(self._magnetic_field_view)
        self._main_stack.addWidget(self._polarisation_view)
        self._main_stack.addWidget(self._quantum_corrections_view)
        self._main_stack.addWidget(self._journal_view)

        self._job_dashboard = JobDashboard(
            JobDashboardDeps(
                document=lambda: self._document,
                staleness=lambda job, now: self._job_execution.staleness(job, now=now),
                can_resubmit=lambda job: self._jobs.can_resubmit(job),
                dialog_parent=lambda: self,
            ),
            self._job_progress,
        )
        self._job_dashboard.action_requested.connect(self._on_dashboard_action_requested)
        self._job_dashboard.cancel_and_dismiss_requested.connect(self._jobs.cancel_and_dismiss)
        self._job_dashboard.show_dismissed_toggled.connect(
            lambda _shown: self._refresh_jobs_table()
        )

        self._jobs.connect_runners()

        self._build_layout()
        self._build_menu()
        # After the File menu's own connection, so Open Recent is rebuilt first.
        self._recent_projects.changed.connect(self._home_page.refresh_recent_projects)
        self.setStatusBar(QStatusBar(self))
        self._keyword_catalog_guard = KeywordCatalogGuard(
            parent=self,
            settings=self._settings,
            open_setup=self._open_castep_setup_dialog,
        )
        self.statusBar().addPermanentWidget(self._keyword_catalog_guard.status_button)
        for widget in self._cluster_queue_status.widgets():
            self.statusBar().addPermanentWidget(widget)
        self._jobs.sync_cluster_status_widgets()
        self.statusBar().showMessage("Ready")
        # Measured with the panels still shown: the welcome page keeps a project's
        # minimum size, so opening a project never resizes the window.
        self.setMinimumSize(self.minimumSizeHint())
        self._sync_project_panels()
        # Pause the 1s runtime timer while the dock is not visible (closed,
        # obscured, window minimized or the tray collapsed), and refresh + re-arm
        # when it returns, so we do not update an invisible table every second.
        # Connected after the first sync: hiding the dock until a project opens
        # is no such pause.
        self._jobs_dock.visibilityChanged.connect(self._job_tray.set_dock_seen)

    def _build_layout(self) -> None:
        central = QWidget()
        central_layout = QHBoxLayout(central)

        # The Job Dashboard tray may cut the page's lower part off; see job_dashboard_tray.
        self._page_area = ClippablePageArea(self._main_stack)
        self._navigator_panel = NavigatorPanel(
            self._project_navigator, self._page_area, self._settings, self
        )
        self._navigator_splitter = self._navigator_panel.splitter
        central_layout.addWidget(self._navigator_splitter)
        self.setCentralWidget(central)
        # The page runs down to the Job Dashboard tray's header strip, with no band of
        # window background between them; the other sides keep the style's margins.
        style = central.style()
        central_layout.setContentsMargins(
            style.pixelMetric(QStyle.PixelMetric.PM_LayoutLeftMargin, None, central),
            style.pixelMetric(QStyle.PixelMetric.PM_LayoutTopMargin, None, central),
            style.pixelMetric(QStyle.PixelMetric.PM_LayoutRightMargin, None, central),
            0,
        )

        self._job_tray = JobDashboardTray(
            self, self._job_dashboard, self._page_area, self._settings
        )
        self._jobs_dock = self._job_tray.dock

    def _on_jobs_dock_toggled(self, shown: bool) -> None:
        # Only the user changes it with a project open, through the View action.
        # Hiding it for having no project is not the user's setting, and
        # minimising the window also unchecks the action.
        if self._document is not None and not self.isMinimized():
            self._jobs_dock_wanted = shown

    def _sync_project_panels(self) -> None:
        """Fit the window to whether a project is open.

        The navigator and the Job Dashboard show only with a project and their
        ``View`` action on. The project actions and the title follow.
        """

        has_project = self._document is not None
        self._navigator_panel.set_project_open(has_project)
        self._place_jobs_dock_once()
        self._jobs_dock.setVisible(has_project and self._jobs_dock_wanted)
        self._jobs_dock.toggleViewAction().setEnabled(has_project)
        self._save_project_action.setEnabled(has_project)
        self._import_relaxations_action.setEnabled(
            has_project and not self._imported_relaxations_controller.scan_in_progress
        )
        self._remove_imported_action.setEnabled(has_project)
        self.setWindowTitle(
            f"{self._document.project.name} \u2014 {PRODUCT_NAME}" if has_project else PRODUCT_NAME
        )
        if has_project:
            # The panels are back in the layout, which sets the minimum from here on.
            self.setMinimumSize(0, 0)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        self._place_jobs_dock_once()

    def _place_jobs_dock_once(self) -> None:
        """Show or hide the dock as it was last time, then let the tray place and fit it.

        This waits for the first time a project is on screen: a saved state would
        otherwise show a dock that stays hidden until a project opens.
        """

        if self._jobs_dock_placed or not self.isVisible() or self._document is None:
            return
        self._jobs_dock_placed = True
        saved_state = self._settings.value(self._MAIN_WINDOW_STATE_KEY, QByteArray())
        if (
            isinstance(saved_state, QByteArray)
            and not saved_state.isEmpty()
            and self.restoreState(saved_state)
        ):
            self._jobs_dock_wanted = not self._jobs_dock.isHidden()
        self._job_tray.place()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if getattr(self, "_shutting_down", False):
            super().closeEvent(event)
            return
        self._settings.setValue(self._MAIN_WINDOW_GEOMETRY_KEY, self.saveGeometry())
        if self._document is not None and self.isVisible():
            # Without a project the dock is hidden whatever the user chose, which
            # is not worth remembering.
            self._settings.setValue(self._MAIN_WINDOW_STATE_KEY, self.saveState())
        self._navigator_panel.save_width()
        job_dashboard = getattr(self, "_job_dashboard", None)
        if job_dashboard is not None:
            job_dashboard.runtime_timer.stop()
        dashboard_flush_timer = getattr(self, "_dashboard_flush_timer", None)
        if dashboard_flush_timer is not None:
            dashboard_flush_timer.stop()
        job_check_timer = getattr(getattr(self, "_jobs", None), "check_timer", None)
        if job_check_timer is not None:
            job_check_timer.stop()
        # Before the save question, whose event loop could otherwise open a scan's review.
        imported_relaxations_controller = getattr(self, "_imported_relaxations_controller", None)
        if imported_relaxations_controller is not None:
            imported_relaxations_controller.hold_for_close()
        # What the user chose for the project, so what arrives from the cluster
        # while closing is saved the same way; None when nothing was asked.
        answer: str | None = None
        if self._document is not None and self.isVisible():
            answer = self._answer_for_saving_on_close()
            if answer == "cancel":
                self._resume_after_abandoned_close()
                event.ignore()
                return
            if answer in ("save", "tidy"):
                try:
                    if answer == "tidy":
                        self._project_service.save_tidying(self._document)
                    else:
                        self._project_service.save_project(self._document)
                except (OSError, ValueError) as exc:
                    event.ignore()
                    self._close_save_failed = True
                    QMessageBox.critical(self, "Save blocked", str(exc))
                    self._resume_after_abandoned_close()
                    return
            self._shutting_down = True
            self._uep_controller.close_analysis_dialogs()
            relaxation_analysis_view = getattr(self, "_relaxation_analysis_view", None)
            if relaxation_analysis_view is not None:
                relaxation_analysis_view.shutdown()
        else:
            # No save prompt (no document or window not visible). Persist any
            # pending coalesced save so completions are not lost on this close;
            # before the autosave coalescing these were already on disk.
            self._flush_dashboard()
            self._shutting_down = True
            relaxation_analysis_view = getattr(self, "_relaxation_analysis_view", None)
            if relaxation_analysis_view is not None:
                relaxation_analysis_view.shutdown()
        site_generation_controller = getattr(self, "_site_generation_controller", None)
        if site_generation_controller is not None:
            site_generation_controller.stop_for_close()
        if imported_relaxations_controller is not None:
            imported_relaxations_controller.stop_for_close()
        self._shutdown_embedded_structure_viewers()
        local_runner = getattr(self, "_local_runner", None)
        if local_runner is not None:
            local_runner.shutdown()
        cluster_runner = getattr(self, "_cluster_runner", None)
        if cluster_runner is not None:
            cluster_runner.shutdown(wait=self._closing_wait())
            if cluster_runner.results_arrived_while_closing:
                self._save_what_arrived_while_closing(answer)
        super().closeEvent(event)

    def _closing_wait(self) -> Callable[[threading.Event, float], bool]:
        """Return the wait the cluster worker's shutdown uses: a busy dialog after a moment.

        Its second wait shows the dialog at once, so the dialog closes and opens
        again rather than leaving a gap.
        """

        waited: list[bool] = []

        def wait(stopped: threading.Event, seconds: float) -> bool:
            delay = {"busy_delay_seconds": 0.0} if waited else {}
            waited.append(True)
            return wait_for_event(
                self,
                stopped,
                seconds,
                title="Closing",
                description="Closing the cluster connection…",
                **delay,
            )

        return wait

    def _save_what_arrived_while_closing(self, answer: str | None) -> None:
        """Save what changed after the close's save, once the cluster worker stopped.

        A submission's scheduler job ID, say, applied while closing waited for
        the worker; called only when such a result arrived. Skipped when the
        user chose not to save. Nothing can be
        shown now, so a failure is only logged.
        """

        if answer is None:
            self._flush_dashboard()
            return
        if answer == "discard" or self._document is None:
            return
        changes = self._project_service.unsaved_changes(self._document)
        if changes is UnsavedChanges.NONE:
            return
        try:
            if changes is UnsavedChanges.TIDYING:
                self._project_service.save_tidying(self._document)
            else:
                self._project_service.save_project(self._document)
        except (OSError, ValueError):
            _LOG.warning(
                "Could not save the cluster results that arrived while closing", exc_info=True
            )

    def _answer_for_saving_on_close(self) -> str:
        """Return ``save``, ``tidy``, ``discard``, ``cancel`` or ``nothing``, asking only if needed.

        ``nothing`` means there was nothing to save and the user was not asked.
        """

        assert self._document is not None
        self._uep_controller.flush_analysis_dialogs()
        changes = self._project_service.unsaved_changes(self._document)
        if changes is UnsavedChanges.UI_STATE or self._close_save_failed:
            return workflow_table_utils.confirm_choice(
                self,
                "Save project?",
                "Save the current project before closing?",
                [("save", "Save"), ("discard", "Don't save"), ("cancel", "Cancel")],
                default="save",
                escape="cancel",
                destructive=("discard",),
            )
        # Nothing to decide. "nothing" writes nothing now.
        return {UnsavedChanges.ROUTINE: "save", UnsavedChanges.TIDYING: "tidy"}.get(
            changes, "nothing"
        )

    def _resume_after_abandoned_close(self) -> None:
        """Restart what ``closeEvent`` stopped, now that the window stays open."""
        # Persist/refresh the coalesced flush whose timer was stopped, so it is not
        # orphaned until the next job.
        self._flush_dashboard()
        self._job_dashboard.sync_runtime_timer()
        job_check_timer = getattr(getattr(self, "_jobs", None), "check_timer", None)
        if job_check_timer is not None:
            job_check_timer.start()
        self._imported_relaxations_controller.resume_after_cancelled_close()

    def _shutdown_embedded_structure_viewers(self) -> None:
        viewers = self.findChildren(AtomicStructureViewer)
        seen: set[int] = set()
        for viewer in viewers:
            viewer_id = id(viewer)
            if viewer_id in seen:
                continue
            seen.add(viewer_id)
            try:
                viewer.shutdown()
            except RuntimeError:
                continue

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        new_action = file_menu.addAction(icon("new_project"), "New Project…")
        new_action.setShortcuts(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_project)

        open_action = file_menu.addAction(icon("open_project"), "Open Project…")
        open_action.setShortcuts(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_project)

        self._recent_projects_menu = RecentProjectsMenu(self._recent_projects, file_menu)
        self._recent_projects_menu.open_requested.connect(self._open_recent_project)
        file_menu.addMenu(self._recent_projects_menu)

        self._save_project_action = file_menu.addAction(icon("save_project"), "Save Project")
        self._save_project_action.setShortcuts(QKeySequence.StandardKey.Save)
        self._save_project_action.triggered.connect(self._save_project)

        file_menu.addSeparator()

        self._import_relaxations_action = file_menu.addAction("Import CASTEP Relaxations…")
        self._import_relaxations_action.triggered.connect(
            self._imported_relaxations_controller.import_relaxations
        )

        self._remove_imported_action = file_menu.addAction("Remove Imported Relaxations…")
        self._remove_imported_action.triggered.connect(
            self._imported_relaxations_controller.remove_imported_relaxations
        )

        file_menu.addSeparator()

        # macOS moves a QuitRole action to the application menu and names it itself.
        quit_action = file_menu.addAction("Exit" if sys.platform == "win32" else "Quit")
        quit_action.setShortcuts(self._quit_key_bindings())
        quit_action.setMenuRole(QAction.MenuRole.QuitRole)
        quit_action.triggered.connect(self.close)

        setup_menu = self.menuBar().addMenu("Setup")
        castep_action = setup_menu.addAction(icon("settings"), "CASTEP…")
        castep_action.triggered.connect(self._open_castep_setup_dialog)
        cluster_action = setup_menu.addAction(icon("cluster"), "HPC Clusters…")
        cluster_action.triggered.connect(lambda: self._open_cluster_setup_dialog())

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self._navigator_panel.toggle_action)
        view_menu.addAction(self._jobs_dock.toggleViewAction())
        self._jobs_dock.toggleViewAction().toggled.connect(self._on_jobs_dock_toggled)
        appearance_menu = view_menu.addMenu("Appearance")
        appearance_group = QActionGroup(self)
        appearance_group.setExclusive(True)

        light_action = appearance_menu.addAction(icon("appearance_light"), "Light")
        light_action.setCheckable(True)
        light_action.triggered.connect(
            lambda _checked=False: self._set_appearance_mode(THEME_LIGHT)
        )
        appearance_group.addAction(light_action)

        dark_action = appearance_menu.addAction(icon("appearance_dark"), "Dark")
        dark_action.setCheckable(True)
        dark_action.triggered.connect(lambda _checked=False: self._set_appearance_mode(THEME_DARK))
        appearance_group.addAction(dark_action)

        system_action = appearance_menu.addAction(icon("appearance_system"), "System")
        system_action.setCheckable(True)
        system_action.triggered.connect(
            lambda _checked=False: self._set_appearance_mode(THEME_SYSTEM)
        )
        appearance_group.addAction(system_action)

        if self._appearance_mode == THEME_DARK:
            dark_action.setChecked(True)
        elif self._appearance_mode == THEME_LIGHT:
            light_action.setChecked(True)
        else:
            system_action.setChecked(True)

        help_menu = self.menuBar().addMenu("Help")
        user_guide_action = help_menu.addAction("User Guide")
        user_guide_action.triggered.connect(self._open_user_guide)
        about_action = help_menu.addAction(f"About {PRODUCT_NAME}")
        about_action.triggered.connect(self._show_about_dialog)

    @staticmethod
    def _quit_key_bindings() -> list[QKeySequence]:
        """Return the platform's Quit keys, with Ctrl+Q first except on Windows.

        What ``StandardKey.Quit`` holds depends on the platform theme, and it can be
        nothing or only the Exit key. Ctrl+Q is Cmd+Q on macOS.
        """

        bindings = QKeySequence.keyBindings(QKeySequence.StandardKey.Quit)
        if sys.platform == "win32":
            return bindings
        ctrl_q = QKeySequence("Ctrl+Q")
        return [ctrl_q, *(binding for binding in bindings if binding != ctrl_q)]

    def _open_user_guide(self) -> None:
        QDesktopServices.openUrl(QUrl(DOCUMENTATION_URL))

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            f"About {PRODUCT_NAME}",
            f"<h3>{PRODUCT_NAME} {current_version()}</h3>"
            f"<p>{PRODUCT_TAGLINE}</p>"
            f'<p><a href="{DOCUMENTATION_URL}">Documentation</a> | '
            f'<a href="{REPOSITORY_URL}">Source code</a></p>',
        )

    def _set_appearance_mode(self, mode: str) -> None:
        self._appearance_mode = mode
        save_theme_mode(self._settings, mode)
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, mode)
        # The dashboard's content-sized columns are fixed widths; refit them
        # to the new theme's metrics.
        self._job_dashboard.fit_columns()
        self._jobs.refresh_cluster_connection_label_colour()
        self._sync_project_home()

    def _new_project(self) -> None:
        dialog = NewProjectDialog(self._file_dialogs.start_dir("new_project_parent"), parent=self)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        project_name = dialog.project_name()
        project_root = dialog.project_root()
        dialog.deleteLater()
        if not accepted or project_root is None:
            return
        self._file_dialogs.remember(project_root.parent, "new_project_parent")

        # Persist any coalesced flush for the outgoing project before swapping.
        self._flush_dashboard()
        try:
            self._document = self._project_service.create_project(project_root, project_name)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Project error", str(exc))
            return

        self._recent_projects.record(project_root)
        self._refresh_ui_after_project_change()
        self.statusBar().showMessage(f"Created project at {project_root}")
        self._navigator_panel.focus_navigator_or_page()

    def _open_project(self) -> None:
        project_dir = self._file_dialogs.get_existing_directory(
            title="Select .mufinder project folder",
            context="open_project",
        )
        if not project_dir:
            return

        self._open_project_from_path(Path(project_dir))

    def open_project(self, project_path: Path) -> bool:
        """Open the project at ``project_path``, reporting failures in a dialog.

        ``project_path`` is a ``.mufinder`` project folder or the ``project.json``
        inside it. This is the path File > Open Project takes, for callers outside
        the window such as the command-line entry point.
        """

        # Resolved so a path relative to the shell's directory is recorded in
        # Open Recent as a path that still works from anywhere.
        project_root = Path(project_path).expanduser().resolve()
        if project_root.name == "project.json" and project_root.is_file():
            project_root = project_root.parent
        return self._open_project_from_path(project_root)

    def show_page(self, route: str) -> bool:
        """Show the workflow page for ``route`` in the open project.

        ``route`` is a project-tree route key such as ``host_structure`` or
        ``uep``. An unknown route, or no open project, shows a warning and leaves
        the current page in place.
        """

        if self._document is None:
            QMessageBox.warning(self, "No project open", f"Open a project to show '{route}'.")
            return False
        if not self._project_navigator.has_route(route):
            QMessageBox.warning(
                self,
                "Unknown page",
                f"MuFinder has no page named '{route}' for the open project.",
            )
            return False
        self._set_active_route(route)
        return True

    def _open_project_from_path(self, project_root: Path, *, show_errors: bool = True) -> bool:
        manifest = project_root / "project.json"
        if not manifest.exists():
            if show_errors:
                QMessageBox.critical(
                    self,
                    "Invalid project",
                    f"{project_root} does not contain project.json.",
                )
            return False

        # Persist any coalesced autosave for the outgoing project before swapping,
        # with a UEP search or view change its open analysis windows hold back.
        self._uep_controller.flush_analysis_dialogs()
        self._flush_dashboard()
        try:
            # Loading parses every entity JSON file; on a mature project over a
            # network share or cold OneDrive folder that froze the window for
            # seconds, so it runs off the GUI thread behind a busy dialog.
            self._document = self._run_blocking_file_op(
                lambda: self._project_service.load_project(project_root),
                description=f"Opening project {project_root.name}…",
            )
        except Exception as exc:  # noqa: BLE001 - surface project-load failures to users.
            if show_errors:
                QMessageBox.critical(self, "Open failed", str(exc))
            return False

        self._recent_projects.record(project_root)
        self._file_dialogs.remember(project_root, "open_project")
        # Shown first: the refresh below reports any local job it had to stop.
        self.statusBar().showMessage(f"Opened project {self._document.project.name}")
        self._refresh_ui_after_project_change()

        integrity_warnings = self._document.integrity_warnings
        if integrity_warnings:
            QMessageBox.warning(
                self,
                "Integrity warnings",
                "Project loaded with integrity warnings:\n\n"
                + "\n".join(f"- {warning}" for warning in integrity_warnings),
            )
        self._navigator_panel.focus_navigator_or_page()
        return True

    def _open_recent_project(self, project_path: str) -> None:
        project_root = Path(project_path).expanduser()
        opened = self._open_project_from_path(project_root, show_errors=False)
        if opened:
            return

        self._recent_projects.remove(project_path)
        QMessageBox.warning(
            self,
            "Recent project unavailable",
            f"Could not open recent project at {project_root}. The entry has been removed from history.",
        )

    def _save_project(self) -> None:
        # The action is disabled with no project.
        if self._document is None:
            return

        self._uep_controller.flush_analysis_dialogs()
        try:
            self._project_service.save_project(self._document)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Save blocked", str(exc))
            return
        self._close_save_failed = False
        # This wrote the whole document, so any coalesced flush is subsumed.
        self._dashboard_flush_pending = False
        self._dashboard_flush_timer.stop()
        self._dashboard_flush_deadline = None
        self._dashboard_last_flush_at = time.monotonic()
        self.statusBar().showMessage("Project saved")

    def _open_castep_setup_dialog(self) -> None:
        dialog = CastepSetupDialog(settings=self._settings, parent=self)
        # The dialog saves paths and installs catalogs as they change, and
        # only has Close.
        dialog.exec()
        changes_saved = dialog.changes_saved
        dialog.deleteLater()
        self._keyword_catalog_guard.refresh()
        if changes_saved:
            self.statusBar().showMessage("CASTEP setup saved")

    def prompt_for_keyword_catalog_if_needed(self) -> None:
        """Explain a missing CASTEP keyword catalog once the window is up."""

        self._keyword_catalog_guard.prompt_at_startup_if_needed()

    def _open_cluster_setup_dialog(
        self, cluster_name: str | None = None, *, parent: QWidget | None = None
    ) -> None:
        dialog = ClusterSetupDialog(
            settings=self._settings, parent=parent or self, cluster_name=cluster_name
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self.statusBar().showMessage("HPC cluster setup updated")

    def _edit_cluster_during_submission(
        self, parent: QWidget, cluster_name: str
    ) -> list[ClusterConfig]:
        """Open a cluster's profile over ``parent``; return the clusters as saved."""

        self._open_cluster_setup_dialog(cluster_name, parent=parent)
        return load_cluster_configs(self._settings)

    def _refresh_ui_after_project_change(self) -> None:
        if self._document is None:
            return

        # Each project opens with its dismissed jobs hidden and every job listed,
        # whatever the last project's state filter; the refresh below redraws.
        self._job_dashboard.hide_dismissed()
        self._job_dashboard.clear_state_filter()

        # First, so a project is reachable even if a refresh below fails, and the
        # jobs table is measured in view rather than over all its rows.
        self._sync_project_panels()
        trace = PerfTrace(label="project change refresh", logger=_LOG)
        # Several steps below cascade into the Relaxation Analysis page (the
        # jobs table, the muon relaxation form restore, the explicit calls).
        # Each only marks it stale here; it is rebuilt once at the end, and only
        # if it is the page showing.
        self._deferring_relaxation_analysis_refresh = True
        try:
            with trace.stage("cluster and local job tracking"):
                self._jobs.reset_tracking_for_project_change()
            with trace.stage("project titles"):
                self._apply_project_change_titles()
            with trace.stage("comparison and magnetic workspace views"):
                self._comparison_tables_controller.sync_view()
                self._sync_magnetic_field_workspace_views()
            with trace.stage("_refresh_navigator"):
                self._refresh_navigator(preserve_selection=False)
            with trace.stage("_refresh_jobs_table"):
                self._refresh_jobs_table()
            with trace.stage("_refresh_host_structure_view"):
                self._refresh_host_structure_view(force=True)
            with trace.stage("_refresh_magnetic_field_view"):
                self._refresh_magnetic_field_view()
            with trace.stage("_refresh_polarisation_view"):
                self._refresh_polarisation_view()
            with trace.stage("_refresh_muon_generation_table"):
                self._refresh_muon_generation_table()
            with trace.stage("convergence views"):
                self._convergence_view.set_form_state(self._convergence_form_restore_state())
                self._convergence_controller.refresh_simulation_cell()
                self._convergence_controller.refresh_view()
                self._kpoint_convergence_view.set_form_state(
                    self._document.ui_state.get("kpoint_convergence_form", {})
                )
                self._kpoint_convergence_controller.refresh_simulation_cell()
                self._kpoint_convergence_controller.refresh_view()
            with trace.stage("workflow forms"):
                self._pristine_relaxation_view.set_form_state(
                    self._document.ui_state.get("pristine_relaxation_form", {})
                )
                self._uep_view.set_form_state(self._document.ui_state.get("uep_form", {}))
                self._muon_generation_view.set_form_state(
                    self._document.ui_state.get("muon_generation_form", {})
                )
                self._muon_relaxation_view.set_form_state(
                    self._document.ui_state.get("muon_relaxation_form", {})
                )
            with trace.stage("relaxation rows"):
                self._refresh_convergence_project_defaults()
                self._refresh_workflow_defaults(self._pristine_relaxation_view)
                self._pristine_relaxation_controller.refresh_view()
                self._refresh_workflow_defaults(self._muon_relaxation_view)
                self._refresh_muon_relaxation_rows()
            with trace.stage("uep rows"):
                self._refresh_workflow_defaults(self._uep_view, supercells=False)
                self._uep_controller.refresh_view()
                self._uep_controller.refresh_source_structures()
                self._refresh_workflow_supercell_options(self._uep_view)
            with trace.stage("_refresh_convergence_overview"):
                self._refresh_convergence_overview()
        finally:
            self._deferring_relaxation_analysis_refresh = False
        with trace.stage("_refresh_relaxation_analysis_context"):
            self._refresh_relaxation_analysis_context()
        with trace.stage("_refresh_muon_sites_summary"):
            self._refresh_muon_sites_summary()
        with trace.stage("_refresh_journal_view"):
            self._refresh_journal_view()
        trace.log_summary(extra=self._document.project.name)
        # Filling the pages tidied some saved forms; only what the user does next counts.
        self._project_service.settle_ui_state(self._document)
        self._close_save_failed = False

    def _apply_project_change_titles(self) -> None:
        assert self._document is not None
        self._home_page.set_project_name(self._document.project.name)
        self._project_navigator.set_title(self._document.project.name)

    def _refresh_journal_view(self) -> None:
        if self._document is None:
            self._journal_view.set_records("", [])
            return
        self._journal_view.set_records(self._document.project.name, self._document.provenance)

    @staticmethod
    def _overview_placeholder_message(title: str) -> str:
        return (
            f"No adopted {title.lower()} analysis is pinned yet.\n"
            "Open the analysis dialog in the child workflow and adopt a project default to pin it here."
        )

    def _host_structure_id_text(self) -> str | None:
        if self._document is None or self._document.project.host_structure_id is None:
            return None
        return str(self._document.project.host_structure_id)

    def _cutoff_overview_summary_rows(self) -> list[tuple[str, str]]:
        if self._document is None:
            return [
                ("Project default", "Not set"),
                ("Functional", "Not recorded"),
                ("Criteria", "Not set"),
            ]
        return [
            (
                "Project default",
                format_cutoff_energy(self._document.project.recommended_cutoff_energy),
            ),
            (
                "Functional",
                self._document.project.recommended_cutoff_xc_functional or "Not recorded",
            ),
            (
                "Criteria",
                format_default_criteria(
                    method=self._document.project.recommended_cutoff_method,
                    energy_threshold=self._document.project.recommended_cutoff_energy_threshold,
                    force_threshold=self._document.project.recommended_cutoff_force_threshold,
                    stress_threshold=self._document.project.recommended_cutoff_stress_threshold,
                ),
            ),
        ]

    def _kpoint_overview_summary_rows(self) -> list[tuple[str, str]]:
        if self._document is None:
            return [
                ("Project default", "Not set"),
                ("Cutoff default", "Not set"),
                ("Functional", "Not recorded"),
                ("Criteria", "Not set"),
            ]
        return [
            ("Project default", format_kpoint_grid(self._document.project.recommended_kpoint_grid)),
            (
                "Cutoff default",
                format_cutoff_energy(self._document.project.recommended_cutoff_energy),
            ),
            (
                "Functional",
                self._document.project.recommended_kpoint_xc_functional or "Not recorded",
            ),
            (
                "Criteria",
                format_default_criteria(
                    method=self._document.project.recommended_kpoint_method,
                    energy_threshold=self._document.project.recommended_kpoint_energy_threshold,
                    force_threshold=self._document.project.recommended_kpoint_force_threshold,
                    stress_threshold=self._document.project.recommended_kpoint_stress_threshold,
                ),
            ),
        ]

    def _refresh_convergence_overview(self) -> None:
        cutoff_snapshot = validated_cutoff_overview_snapshot(self._document)
        kpoint_snapshot = validated_kpoint_overview_snapshot(self._document)
        self._convergence_overview_view.set_cutoff_panel(
            summary_rows=self._cutoff_overview_summary_rows(),
            snapshot=cutoff_snapshot,
            placeholder_message=self._overview_placeholder_message("cutoff"),
        )
        self._convergence_overview_view.set_kpoint_panel(
            summary_rows=self._kpoint_overview_summary_rows(),
            snapshot=kpoint_snapshot,
            placeholder_message=self._overview_placeholder_message("k-point"),
        )

    @staticmethod
    def _form_state_has_cutoffs(state: dict[str, object]) -> bool:
        raw_cutoffs = state.get("cutoff_energies", [])
        if isinstance(raw_cutoffs, list):
            for value in raw_cutoffs:
                try:
                    float(value)
                    return True
                except (TypeError, ValueError):
                    continue

        raw_rows = state.get("cutoff_parameter_rows", [])
        if isinstance(raw_rows, list):
            for entry in raw_rows:
                if not isinstance(entry, dict):
                    continue
                try:
                    float(entry.get("cutoff"))
                    return True
                except (TypeError, ValueError):
                    continue
        return False

    def _convergence_form_restore_state(self) -> dict[str, object]:
        if self._document is None:
            return {}

        persisted = self._document.ui_state.get("convergence_form", {})
        if not isinstance(persisted, dict):
            persisted = {}

        if self._form_state_has_cutoffs(persisted):
            return persisted

        definition = self._convergence_controller.active_definition()
        if definition is None:
            return persisted

        configuration = definition.parameters.get("configuration", {})
        if not isinstance(configuration, dict):
            return persisted

        # The definition stores its sweep as ``sweep_values``; the form keeps
        # its key list under its own field name.
        recovered = dict(configuration)
        sweep_values = recovered.pop("sweep_values", [])
        recovered[self._convergence_controller.keys_state_field] = (
            list(sweep_values) if isinstance(sweep_values, list) else []
        )

        range_state = persisted.get("range")
        if isinstance(range_state, dict):
            recovered["range"] = dict(range_state)

        cutoff_rows = persisted.get("cutoff_parameter_rows")
        if isinstance(cutoff_rows, list):
            recovered["cutoff_parameter_rows"] = cutoff_rows

        persisted_name = str(persisted.get("name_prefix", "")).strip()
        if persisted_name:
            recovered["name_prefix"] = persisted_name

        return recovered

    def _magnetic_field_workspaces_state(self) -> list[dict[str, object]]:
        if self._document is None:
            return []
        workspaces = self._document.ui_state.get(self._MAGNETIC_FIELD_WORKSPACES_KEY, [])
        if not isinstance(workspaces, list):
            return []
        return [workspace for workspace in workspaces if isinstance(workspace, dict)]

    def _magnetic_field_workspace_route(self, workspace_id: str) -> str:
        return magnetic_field_workspace_route(workspace_id)

    def _magnetic_field_workspace_id_from_route(self, route: object) -> str | None:
        return magnetic_field_workspace_id_from_route(route)

    def _relaxation_analysis_workspace_route(self, workspace_id: str) -> str:
        return relaxation_analysis_workspace_route(workspace_id)

    def _relaxation_analysis_workspace_id_from_route(self, route: object) -> str | None:
        return relaxation_analysis_workspace_id_from_route(route)

    def _refresh_navigator(self, *, preserve_selection: bool = True) -> None:
        """Refresh the navigator and project home from one summary; go home if the page is gone."""

        if self._document is None:
            return
        summary = summarize_project(self._document)
        if not self._project_navigator.refresh(
            self._document, summary, keep_current=preserve_selection
        ):
            self._main_stack.setCurrentWidget(self._home_page)
            self._sync_home_content_page()
        self._home_page.set_document(self._document, summary)

    def _refresh_jobs_table(self) -> None:
        """Redraw the Job Dashboard and the workflow rows and badges that follow job state."""

        self._job_dashboard.refresh()
        if self._document is None:
            self._jobs.close_progress_dialogs()
            self._convergence_view.set_runtime_states({})
            self._kpoint_convergence_view.set_runtime_states({})
            self._pristine_relaxation_view.set_runtime_states({})
            self._muon_relaxation_view.set_runtime_states({})
            self._uep_view.set_runtime_states({})
            self._refresh_relaxation_analysis_context()
            self._jobs.sync_cluster_status_widgets()
            return

        self._jobs.sync_cluster_status_widgets()

        self._convergence_controller.refresh_view()
        self._kpoint_convergence_controller.refresh_view()
        self._pristine_relaxation_controller.refresh_view()
        self._refresh_muon_relaxation_rows()
        self._uep_controller.refresh_view()
        self._refresh_navigator()

    def _persist_document_ui_state(self, *, failure_message: str) -> None:
        if self._document is None:
            return
        self._uep_controller.flush_analysis_dialogs()
        try:
            self._project_service.save_project(self._document)
        except (OSError, ValueError) as exc:
            _LOG.warning("Autosave failed: %s", exc)
            self.statusBar().showMessage(failure_message)
        else:
            self._close_save_failed = False

    def _request_dashboard_flush(self, *, failure_message: str) -> None:
        """Schedule a coalesced dashboard refresh + autosave for lifecycle handlers.

        Many cluster jobs can finish in a single queue poll — or a few seconds
        apart across a serial cluster retrieval — and rebuilding every view plus
        saving the whole document per job freezes the GUI. Instead, mark a flush
        pending and arm ``_dashboard_flush_timer`` to run it once via
        ``_flush_dashboard``.

        Two goals in tension: an isolated event should still flush promptly
        (the ``_DASHBOARD_FLUSH_DEBOUNCE_MS`` quiet period), but a steady stream
        of events spaced further apart than that debounce must not re-arm it
        forever and starve the flush until the stream ends. So the delay used
        is ``max(debounce, min_interval - elapsed_since_last_flush)``: during a
        burst this floors the gap between flushes at
        ``_DASHBOARD_MIN_FLUSH_INTERVAL_MS``. Critically, an already-armed
        deadline is only ever tightened (moved earlier), never pushed further
        into the future — otherwise a steady stream of requests, each re-arming
        the timer, would never let it fire.
        """

        if self._document is None:
            return
        self._dashboard_flush_pending = True
        self._dashboard_autosave_failure_message = failure_message
        if getattr(self, "_shutting_down", False):
            # Closing writes what is pending itself, as the user chose.
            return

        now = time.monotonic()
        if self._dashboard_last_flush_at is None:
            elapsed_since_flush_ms = None
        else:
            elapsed_since_flush_ms = (now - self._dashboard_last_flush_at) * 1000.0
        if elapsed_since_flush_ms is None:
            min_interval_delay_ms = 0.0
        else:
            min_interval_delay_ms = self._DASHBOARD_MIN_FLUSH_INTERVAL_MS - elapsed_since_flush_ms
        delay_ms = max(self._DASHBOARD_FLUSH_DEBOUNCE_MS, min_interval_delay_ms)
        candidate_deadline = now + delay_ms / 1000.0

        pending_deadline = self._dashboard_flush_deadline
        if pending_deadline is not None and self._dashboard_flush_timer.isActive():
            if candidate_deadline >= pending_deadline:
                # The armed deadline already covers this request (or is
                # earlier) — leave it alone so the stream cannot push it out
                # indefinitely.
                return
        self._dashboard_flush_deadline = candidate_deadline
        self._dashboard_flush_timer.start(max(0, int(round(delay_ms))))

    def _flush_dashboard(self) -> None:
        """Rebuild the dashboard views and persist the document, if a flush is due."""

        self._dashboard_flush_timer.stop()
        self._dashboard_flush_deadline = None
        if not self._dashboard_flush_pending:
            return
        self._dashboard_flush_pending = False
        if self._document is None:
            return
        self._dashboard_last_flush_at = time.monotonic()
        trace = PerfTrace(label="dashboard flush", logger=_LOG)
        # _refresh_jobs_table already refreshes the convergence/kpoint/pristine/
        # muon-relaxation/uep rows, so only the views it does not cover are added.
        with trace.stage("_refresh_jobs_table"):
            self._refresh_jobs_table()
        with trace.stage("_refresh_journal_view"):
            self._refresh_journal_view()
        with trace.stage("_refresh_host_structure_view"):
            self._refresh_host_structure_view()
        with trace.stage("_refresh_muon_generation_table"):
            self._refresh_muon_generation_table()
        with trace.stage("_refresh_hyperfine_coupling_view"):
            self._refresh_hyperfine_coupling_view()
        with trace.stage("_refresh_quantum_corrections_view"):
            self._refresh_quantum_corrections_view()
        with trace.stage("_refresh_energy_barriers_view"):
            self._refresh_energy_barriers_view()
        with trace.stage("_persist_document_ui_state"):
            self._persist_document_ui_state(
                failure_message=self._dashboard_autosave_failure_message
            )
        # The widgets the views released are deleted, and the window repainted, on
        # the next event-loop pass, still inside the same freeze. The summary is
        # logged from a zero-delay timer after that pass so its total covers them;
        # forcing the deferred deletes here would destroy objects whose handlers
        # may still be on the stack (a flush inside a confirmation's event loop).
        QTimer.singleShot(
            0, self, partial(self._log_dashboard_flush_trace, trace, time.perf_counter())
        )

    @staticmethod
    def _log_dashboard_flush_trace(trace: PerfTrace, handed_back: float) -> None:
        trace.stages.append(
            StageTiming(
                name="event loop (deferred deletes, repaint)",
                duration_ms=(time.perf_counter() - handed_back) * 1000.0,
            )
        )
        trace.log_summary()

    def _show_route(self, route: object) -> None:
        """Show the page for the navigator row the user selected."""

        target = self._home_page
        if route == "host_structure":
            target = self._host_structure_view
        elif route == "convergence":
            target = self._convergence_overview_view
        elif route == "convergence_cutoff":
            target = self._convergence_view
        elif route == "convergence_kpoint":
            target = self._kpoint_convergence_view
        elif route == "pristine_relaxation":
            target = self._pristine_relaxation_view
        elif route == "uep":
            target = self._uep_view
        elif route == "muon_candidates":
            target = self._muon_generation_view
        elif route == "muon_relaxations":
            target = self._muon_relaxation_view
        elif route == "muon_analysis":
            self._refresh_stale_relaxation_analysis_context()
            target = self._relaxation_analysis_view
        elif (
            analysis_workspace_id := self._relaxation_analysis_workspace_id_from_route(route)
        ) is not None:
            self._refresh_stale_relaxation_analysis_context(analysis_workspace_id)
            self._relaxation_analysis_view.focus_workspace(analysis_workspace_id)
            target = self._relaxation_analysis_view
        elif route == "muon_sites":
            self._refresh_muon_sites_summary()
            target = self._muon_sites_summary_view
        elif route == "magnetic_fields":
            target = self._magnetic_field_view
        elif (
            isinstance(route, str)
            and self._magnetic_field_workspace_id_from_route(route) is not None
        ):
            workspace_id = self._magnetic_field_workspace_id_from_route(route)
            if workspace_id is not None:
                target = self._magnetic_field_workspace_widgets.get(
                    workspace_id, self._magnetic_field_view
                )
        elif route == "polarisation":
            self._polarisation_view.refresh()
            target = self._polarisation_view
        elif route == "hyperfine_coupling":
            self._refresh_hyperfine_coupling_view()
            target = self._hyperfine_coupling_view
        elif route == "quantum_corrections":
            self._refresh_quantum_corrections_view()
            target = self._quantum_corrections_view
        elif route == "energy_barriers":
            target = self._ensure_energy_barriers_view()
            self._refresh_energy_barriers_view()
        elif route == "provenance_log":
            target = self._journal_view
        elif (comparison_table_id := comparison_table_id_from_route(route)) is not None:
            target = (
                self._comparison_tables_view
                if self._comparison_tables_view.show_table(comparison_table_id)
                else self._pristine_relaxation_view
            )
        self._main_stack.setCurrentWidget(target)
        # Navigation does not mutate the document, so workflow completion state
        # cannot change here. Stage icons are kept current by tree construction
        # and by the mutation-driven refresh calls, so recomputing every check
        # on each tab switch (which scans jobs/results) is wasted work.
        self._sync_project_home()

    def _sync_magnetic_field_workspace_views(self) -> None:
        for widget in self._magnetic_field_workspace_widgets.values():
            self._main_stack.removeWidget(widget)
            widget.deleteLater()
        self._magnetic_field_workspace_widgets.clear()

        if self._document is None:
            return

        for workspace in self._magnetic_field_workspaces_state():
            workspace_id = str(workspace.get("id", "")).strip()
            if not workspace_id:
                continue
            title = str(workspace.get("title", "Magnetic Fields")).strip() or "Magnetic Fields"
            magnetic_structure = workspace.get("magnetic_structure")
            form_state = workspace.get("form", {})
            widget = MagneticFieldWidget(parent=self)
            widget.set_document(
                self._document,
                magnetic_structure=magnetic_structure
                if isinstance(magnetic_structure, Mapping)
                else None,
                title=title,
                title_editable=True,
                deletable=True,
            )
            widget.set_form_state(
                attach_magnetic_field_results(self._document, workspace_id, form_state),
                preserve_edits=True,
            )
            widget.magnetic_structure_changed.connect(
                lambda payload, workspace_id=workspace_id: (
                    self._on_magnetic_field_workspace_structure_changed(
                        workspace_id,
                        payload,
                    )
                )
            )
            widget.form_state_changed.connect(
                lambda state, workspace_id=workspace_id: (
                    self._on_magnetic_field_workspace_form_state_changed(
                        workspace_id,
                        state,
                    )
                )
            )
            widget.workspace_title_changed.connect(
                lambda title, workspace_id=workspace_id: (
                    self._on_magnetic_field_workspace_title_changed(
                        workspace_id,
                        title,
                    )
                )
            )
            widget.pin_requested.connect(self._on_magnetic_field_pin_requested)
            widget.open_muon_sites_requested.connect(lambda: self._set_active_route("muon_sites"))
            widget.open_polarisation_requested.connect(
                lambda workspace_id=workspace_id: self._open_polarisation_for(workspace_id)
            )
            widget.delete_requested.connect(
                lambda workspace_id=workspace_id: self._delete_magnetic_field_workspace(
                    workspace_id
                )
            )
            self._main_stack.addWidget(widget)
            self._magnetic_field_workspace_widgets[workspace_id] = widget

    def _refresh_magnetic_field_workspace_views(self) -> None:
        if self._document is None:
            return
        for workspace in self._magnetic_field_workspaces_state():
            workspace_id = str(workspace.get("id", "")).strip()
            widget = self._magnetic_field_workspace_widgets.get(workspace_id)
            if widget is None:
                continue
            title = str(workspace.get("title", "Magnetic Fields")).strip() or "Magnetic Fields"
            magnetic_structure = workspace.get("magnetic_structure")
            form_state = workspace.get("form", {})
            widget.set_document(
                self._document,
                magnetic_structure=magnetic_structure
                if isinstance(magnetic_structure, Mapping)
                else None,
                title=title,
                title_editable=True,
                deletable=True,
            )
            widget.set_form_state(
                attach_magnetic_field_results(self._document, workspace_id, form_state),
                preserve_edits=True,
            )

    def _set_active_route(self, route: str) -> None:
        if route == PROJECT_HOME_ROUTE:
            self._show_project_home()
        elif self._project_navigator.set_current_route(route):
            self._show_route(route)

    def _show_project_home(self) -> None:
        self._project_navigator.set_current_route(PROJECT_HOME_ROUTE)
        self._main_stack.setCurrentWidget(self._home_page)
        self._sync_home_content_page()
        self._sync_project_home()

    def _sync_home_content_page(self) -> None:
        self._home_page.show_project_page(self._document is not None)

    def _sync_project_home(self) -> None:
        self._home_page.set_document(self._document)

    def _job_stage_page(self, stage: WorkflowStage) -> tuple[WorkflowJobController, object] | None:
        """Return the controller and view whose table holds jobs of ``stage``, if one does."""

        if stage == WorkflowStage.CONVERGENCE_CUTOFF:
            return self._convergence_controller, self._convergence_view
        if stage == WorkflowStage.CONVERGENCE_KPOINT:
            return self._kpoint_convergence_controller, self._kpoint_convergence_view
        if stage == WorkflowStage.PRISTINE_RELAXATION:
            return self._pristine_relaxation_controller, self._pristine_relaxation_view
        if stage == WorkflowStage.UEP:
            return self._uep_controller, self._uep_view
        if stage == WorkflowStage.MUON_RELAXATION:
            return self._muon_relaxation_controller, self._muon_relaxation_view
        if stage == WorkflowStage.HYPERFINE_COUPLING:
            return self._hyperfine_coupling_controller, self._hyperfine_coupling_view
        if stage == WorkflowStage.QUANTUM_CORRECTIONS:
            return self._quantum_corrections_controller, self._quantum_corrections_view
        if stage == WorkflowStage.ENERGY_BARRIERS:
            return self._energy_barriers_controller, self._ensure_energy_barriers_view()
        return None

    def _on_open_in_stage_from_dashboard(self, job_ids: list[UUID]) -> None:
        """Show the page of the jobs' stage with their rows selected."""

        if self._document is None:
            return
        jobs = [job for job_id in job_ids if (job := self._document.jobs.get(job_id)) is not None]
        if not jobs or not self.show_page(job_stage_key(jobs[0].workflow_stage)):
            return
        page = self._job_stage_page(jobs[0].workflow_stage)
        keys: list[object] = []
        if page is not None:
            controller, view = page
            keys = [key for job in jobs if (key := controller.key_for_job(job.id)) is not None]
            if keys and not view.select_rows_for_keys(keys):
                keys = []
        if len(keys) < len(jobs):
            name = job_stage_name(jobs[0].workflow_stage)
            subject = (
                f"{jobs[0].label} has"
                if len(jobs) == 1
                else plural(len(jobs) - len(keys), "job has", "jobs have")
            )
            self.statusBar().showMessage(f"{subject} no row on {name} now")
        else:
            # An earlier "no row" message no longer describes what is on screen.
            self.statusBar().clearMessage()

    def _on_dashboard_action_requested(self, kind: str, job_ids: list[UUID]) -> None:
        """Carry out an action the Job Dashboard confirmed."""

        if kind == "open":
            self._on_open_in_stage_from_dashboard(list(job_ids))
        else:
            self._jobs.handle_dashboard_action(kind, list(job_ids))

    @staticmethod
    def _cutoff_key(value: float) -> str:
        return f"{float(value):.12g}"

    def _ensure_energy_barriers_view(self) -> EnergyBarriersWidget:
        energy_barriers_view = self._energy_barriers_view
        if energy_barriers_view is not None:
            return energy_barriers_view

        energy_barriers_view = EnergyBarriersWidget()
        controller = self._energy_barriers_controller
        energy_barriers_view.endpoint_selection_changed.connect(controller.select_endpoints)
        energy_barriers_view.generate_paths_requested.connect(controller.generate_paths)
        energy_barriers_view.show_every_site_toggled.connect(controller.set_show_every_site)
        energy_barriers_view.open_muon_sites_requested.connect(
            lambda: self._set_active_route("muon_sites")
        )
        energy_barriers_view.remove_sources_requested.connect(controller.remove_paths)
        self._connect_workflow_row_actions(energy_barriers_view, controller)
        energy_barriers_view.configure_result_requested.connect(controller.configure_paths)
        # The controller was built and registered in ``__init__``; its view's
        # signals are wired here, when the view is first created.
        energy_barriers_view.submit_neb_local_requested.connect(
            lambda values: controller.submit(values, target="local")
        )
        energy_barriers_view.submit_neb_cluster_requested.connect(
            lambda values: controller.submit(values, target="cluster")
        )
        energy_barriers_view.path_selected.connect(controller.show_path)
        self._energy_barriers_view = energy_barriers_view
        self._main_stack.insertWidget(
            self._main_stack.indexOf(self._journal_view), energy_barriers_view
        )
        return energy_barriers_view

    def _on_remove_cutoffs_from_convergence(self, cutoff_values: list[float]) -> None:
        if not cutoff_values:
            return

        if self._document is None:
            self._convergence_view.remove_cutoffs(cutoff_values)
            return

        plan = [self._convergence_cutoff_cleanup_item(float(cutoff)) for cutoff in cutoff_values]
        multi = len(plan) > 1
        outcome = self._job_file_cleanup.execute(
            plan,
            choose_disposition=lambda item: self._prompt_cutoff_file_disposition(
                item.key[0],
                item.directory,
                include_apply_all=multi,
            ),
            stop_on_error=False,
        )
        for failure in outcome.failures:
            QMessageBox.warning(
                self,
                "File cleanup failed",
                f"Could not update files for cutoff {failure.item.key[0]:.8g} eV:\n{failure.reason}",
            )

        removed_cutoffs: list[float] = []
        for item in outcome.cleaned:
            cutoff, definition = item.key
            if definition is not None:
                self._remove_cutoff_from_equivalent_definitions(definition, cutoff)
            removed_cutoffs.append(cutoff)

        if removed_cutoffs:
            self._convergence_view.remove_cutoffs(removed_cutoffs)
            self._refresh_jobs_table()
            self._convergence_controller.refresh_view()
            removed_text = plural(
                len(removed_cutoffs), "convergence cutoff entry", "convergence cutoff entries"
            )
            self._report_cleanup_outcome(f"Removed {removed_text}", outcome)

    def _convergence_cutoff_cleanup_item(self, cutoff: float) -> JobCleanupItem:
        """Return the clean-up item for one convergence cutoff.

        Its directory is ``<definition output>/<job label>``, named after the
        cutoff's job or, without one, after the label the sweep would give it;
        archived files stay under the definition's output directory. ``key`` is
        ``(cutoff, definition)``, so the cutoff can be dropped from the
        definitions once its files are dealt with.
        """

        definition, config = self._convergence_controller.definition_for_sweep_key(cutoff)
        job = (
            self._convergence_controller.job_for_cutoff(definition, cutoff)
            if definition is not None
            else None
        )

        job_label = None
        if job is not None:
            job_label = job.label
        elif config is not None:
            job_label = convergence_job_name(
                config.name_prefix,
                cutoff,
                sweep_parameter="cut_off_energy",
            )

        output_root = (
            definition_output_directory(self._document, definition)
            if definition is not None
            else None
        )
        job_dir = output_root / job_label if output_root is not None and job_label else None
        # The cutoff leaves every equivalent definition with this item, so a job one of
        # them still binds (a reconfigured sweep's earlier run) gives up the folder too.
        releases = (
            frozenset(
                other.id
                for equivalent in self._convergence_controller.definitions()
                if sweep_definition_context_matches(self._document, equivalent, definition)
                and (other := self._convergence_controller.job_for_cutoff(equivalent, cutoff))
                is not None
            )
            if definition is not None
            else frozenset()
        )
        return JobCleanupItem(
            label=job_label or f"{cutoff:.8g} eV",
            jobs=(job,) if job is not None else (),
            directory=job_dir if self._job_file_cleanup.directory_on_disk(job_dir) else None,
            archive_root=output_root,
            key=(cutoff, definition),
            releases=releases,
        )

    def _remove_cutoff_from_equivalent_definitions(self, source_definition, cutoff: float) -> None:
        if self._document is None:
            return

        if (
            not source_definition.input_structure_ids
            or definition_output_directory(self._document, source_definition) is None
        ):
            self._convergence_controller.remove_cutoff_from_definition(source_definition, cutoff)
            return

        for definition in self._convergence_controller.definitions():
            if sweep_definition_context_matches(self._document, definition, source_definition):
                self._convergence_controller.remove_cutoff_from_definition(definition, cutoff)

    def _run_blocking_file_op(self, op: Callable[[], object], *, description: str) -> object:
        """Run a blocking filesystem operation off the GUI thread.

        Preserves the caller's synchronous semantics (returns the result,
        re-raises the exception) while pumping the event loop, so the retry
        backoff against OneDrive/antivirus file locks and large recursive
        deletes cannot freeze the window. User input is held back while the
        operation runs, and a busy dialog appears only when it takes
        noticeably long.
        """

        return run_blocking_call(self, op, title="Working on files", description=description)

    def _prompt_cutoff_file_disposition(
        self,
        cutoff: float,
        job_dir: Path,
        *,
        include_apply_all: bool = False,
    ) -> str | None:
        return self._jobs.prompt_job_file_disposition(
            title="Remove cutoff with files",
            text=f"Cutoff {cutoff:.8g} eV has associated files in {job_dir.name}.",
            keep_text="or Cancel to keep the cutoff entry.",
            include_apply_all=include_apply_all,
        )

    def _refresh_views_after_job_reset(self) -> None:
        """Repaint every workflow table that can show a job just prepared to run again.

        Called when a workflow row's Run, or the dashboard's ``Resubmit…``,
        returns an ended job to ``defined`` (continued or started again), and
        after ``Continue relaxation…``. ``_refresh_jobs_table`` already
        re-renders the dashboard plus the convergence, pristine, muon, and UEP
        tables; the convergence controllers are refreshed explicitly, and the
        two lazily-created views (energy barriers, quantum corrections) refresh
        through their own no-op-until-created wrappers. Every workflow runs
        ended jobs again from its table, so every workflow's table has to
        repaint.
        """

        self._refresh_jobs_table()
        self._convergence_controller.refresh_view()
        self._kpoint_convergence_controller.refresh_view()
        self._refresh_energy_barriers_view()
        self._refresh_hyperfine_coupling_view()
        self._refresh_quantum_corrections_view()

    _dedupe_jobs = staticmethod(dedupe_jobs)

    def _build_jobs_controller_deps(self) -> JobsControllerDeps:
        """Build the jobs controller's late-bound view of this window.

        Every entry is a small wrapper, so the controller always sees the
        current document and views, and any test-time replacement of them.
        """

        return JobsControllerDeps(
            document=lambda: self._document,
            dialog_parent=lambda: self,
            status_bar=lambda: self.statusBar(),
            job_output_directory=lambda job: self._job_output_directory(job),
            job_seed_name=lambda job: self._job_seed_name(job),
            workflow_controllers=lambda: self._workflow_controllers,
            convergence_controller=lambda: self._convergence_controller,
            kpoint_convergence_controller=lambda: self._kpoint_convergence_controller,
            pristine_relaxation_controller=lambda: self._pristine_relaxation_controller,
            muon_relaxation_controller=lambda: self._muon_relaxation_controller,
            pristine_relaxation_view=lambda: self._pristine_relaxation_view,
            muon_relaxation_view=lambda: self._muon_relaxation_view,
            muon_relaxation_row_key=muon_relaxation_row_key,
            refresh_jobs_table=lambda: self._refresh_jobs_table(),
            refresh_journal_view=lambda: self._refresh_journal_view(),
            refresh_hyperfine_coupling_view=lambda: self._refresh_hyperfine_coupling_view(),
            refresh_quantum_corrections_view=lambda: self._refresh_quantum_corrections_view(),
            refresh_energy_barriers_view=lambda: self._refresh_energy_barriers_view(),
            refresh_views_after_job_reset=lambda: self._refresh_views_after_job_reset(),
            request_dashboard_flush=lambda **kwargs: self._request_dashboard_flush(**kwargs),
            persist_document_ui_state=lambda **kwargs: self._persist_document_ui_state(**kwargs),
            with_kpoint_status_note=lambda message: self._with_kpoint_status_note(message),
            open_castep_setup_dialog=lambda: self._open_castep_setup_dialog(),
            open_cluster_setup_dialog=lambda: self._open_cluster_setup_dialog(),
            edit_cluster_during_submission=lambda *args, **kwargs: (
                self._edit_cluster_during_submission(*args, **kwargs)
            ),
            keyword_catalog_guard=lambda: self._keyword_catalog_guard,
            quantum_corrections_controller=lambda: self._quantum_corrections_controller,
            energy_barriers_controller=lambda: self._energy_barriers_controller,
            job_dashboard=lambda: self._job_dashboard,
            dashboard_flush_pending=lambda: self._dashboard_flush_pending,
        )

    def _build_workflow_controller_context(self) -> WorkflowControllerContext:
        """Build the narrow, late-bound context shared by all workflow controllers.

        Every callable is a small wrapper so the controller always sees the
        current document and any test-time monkeypatching of these methods
        (see the ``monkeypatch.setattr(window, "_submit_jobs_*", ...)`` tests).
        Workflow-specific handles never go here — they live in a per-controller
        deps dataclass (see :meth:`_build_convergence_cutoff_deps`).
        """

        return WorkflowControllerContext(
            document=lambda: self._document,
            project_service=self._project_service,
            job_execution=self._job_execution,
            submission_coordinator=self._workflow_submission,
            submit_jobs_locally=lambda *args, **kwargs: self._jobs.submit_jobs_locally(
                *args, **kwargs
            ),
            submit_jobs_on_cluster=lambda *args, **kwargs: self._jobs.submit_jobs_on_cluster(
                *args, **kwargs
            ),
            prepare_jobs_for_resubmission=lambda jobs: self._jobs.prepare_jobs_for_resubmission(
                jobs
            ),
            show_info=lambda title, text: QMessageBox.information(self, title, text),
            request_dashboard_flush=lambda **kwargs: self._request_dashboard_flush(**kwargs),
            persist_ui_state=lambda **kwargs: self._persist_document_ui_state(**kwargs),
            status_message=lambda message: self.statusBar().showMessage(message),
            jobs_free_to_start=lambda jobs: self._jobs.without_jobs_in_another_jobs_folder(jobs),
            set_submitted_rows=lambda keys: self._set_submitted_rows(keys),
            definition_output_directory=lambda definition: definition_output_directory(
                self._document, definition
            ),
            job_output_directory=lambda job: self._job_output_directory(job),
            job_seed_name=lambda job: self._job_seed_name(job),
            job_file_cleanup=self._job_file_cleanup,
            run_for_job=lambda job_id: self._run_for_job(job_id),
            total_energy_for_job=lambda job_id: total_energy_for_job(self._document, job_id),
            max_force_for_job=lambda job_id: max_force_for_job(self._document, job_id),
        )

    def _build_convergence_cutoff_deps(self) -> ConvergenceCutoffDeps:
        """Build the cutoff controller's deps: shared convergence helpers.

        These helpers stay on ``MainWindow`` because non-migrating convergence
        code still calls them; the controller reaches them only through these
        late-bound wrappers.
        """

        return ConvergenceCutoffDeps(
            deserialize_convergence_config=deserialize_convergence_config,
            cutoff_key=lambda value: self._cutoff_key(value),
            max_stress_for_job=lambda job_id: max_stress_for_job(self._document, job_id),
            **self._sweep_convergence_window_deps(),
        )

    def _sweep_convergence_window_deps(self) -> dict[str, Callable[..., Any]]:
        """Return the dialogs and refreshes both convergence sweeps reach the window through."""

        return {
            "run_examiner": lambda *args, **kwargs: self._run_convergence_examiner(*args, **kwargs),
            "show_warning": lambda title, text: QMessageBox.warning(self, title, text),
            "confirm_choice": lambda *args, **kwargs: workflow_table_utils.confirm_choice(
                self, *args, **kwargs
            ),
            "refresh_after_default_adopted": lambda: self._refresh_after_convergence_default(),
            "select_simulation_cell_in_editor": lambda *args, **kwargs: (
                self._select_simulation_cell_in_editor(*args, **kwargs)
            ),
            "refresh_jobs_table": lambda: self._refresh_jobs_table(),
            "refresh_journal_view": lambda: self._refresh_journal_view(),
        }

    def _run_convergence_examiner(
        self,
        points: Mapping[float, SweepPointRunState],
        *,
        on_adopted: Callable[..., None],
        on_state_changed: Callable[[dict], None],
        **options: Any,
    ) -> dict[str, object]:
        """Open the examiner modally on ``points``; return its state when it closes."""

        dialog = ConvergenceExaminerDialog(points, parent=self, **options)
        dialog.default_adopted.connect(on_adopted)
        dialog.state_changed.connect(on_state_changed)
        dialog.exec()
        return dialog.state_snapshot()

    def _refresh_after_convergence_default(self) -> None:
        """Refresh the views an adopted convergence default changes."""

        self._refresh_project_defaults_views()
        self._refresh_convergence_overview()
        self._refresh_journal_view()
        self._refresh_navigator()

    def _build_convergence_kpoint_deps(self) -> ConvergenceKpointDeps:
        """Build the k-point controller's deps: shared convergence helpers.

        These helpers stay on ``MainWindow`` because non-migrating convergence
        code still calls them; the controller reaches them only through these
        late-bound wrappers. ``deserialize_convergence_config`` overlaps
        ``ConvergenceCutoffDeps`` but the two sweeps deliberately keep separate
        deps types. Cross-workflow job/definition readers live on the shared
        context, not here; ``max_stress_for_job`` stays because only the two
        convergence sweeps use it.
        """

        return ConvergenceKpointDeps(
            deserialize_convergence_config=deserialize_convergence_config,
            grid_key=lambda grid: self._grid_key(grid),
            max_stress_for_job=lambda job_id: max_stress_for_job(self._document, job_id),
            **self._sweep_convergence_window_deps(),
        )

    def _build_comparison_tables_deps(self) -> ComparisonTablesDeps:
        """Build the comparison-table controller's deps; see the dataclass."""

        return ComparisonTablesDeps(
            completed_jobs=lambda: self._pristine_relaxation_controller.completed_jobs(),
            choose_jobs=lambda jobs, default_label: run_comparison_job_dialog(
                jobs, default_label, parent=self
            ),
            show_warning=lambda title, text: QMessageBox.warning(self, title, text),
            refresh_navigator=lambda: self._refresh_navigator(),
            refresh_journal_view=lambda: self._refresh_journal_view(),
            set_navigator_label=lambda route, label: self._project_navigator.set_row_label(
                route, label
            ),
            active_route=lambda: self._project_navigator.active_route(),
            open_route=lambda route: self._set_active_route(route),
            shutting_down=lambda: self._shutting_down,
        )

    def _build_muon_relaxation_deps(self) -> MuonRelaxationDeps:
        """Build the muon-relaxation controller's deps; see the dataclass."""

        def _show_staged_sites() -> None:
            self._refresh_muon_generation_table()
            self._refresh_workflow_defaults(self._muon_relaxation_view)
            self._refresh_muon_relaxation_rows()
            self._refresh_journal_view()
            self._set_active_route("muon_relaxations")

        return MuonRelaxationDeps(
            handle_site_host_mismatch=(lambda mismatch: self._handle_site_host_mismatch(mismatch)),
            dialog_parent=lambda: self,
            show_staged_sites=_show_staged_sites,
        )

    def _build_site_generation_deps(self) -> SiteGenerationDeps:
        """Build the site-generation controller's deps; see the dataclass."""

        return SiteGenerationDeps(
            dialog_parent=lambda: self,
            refresh_generation_table=lambda: self._refresh_muon_generation_table(),
            refresh_journal_view=lambda: self._refresh_journal_view(),
            shutting_down=lambda: self._shutting_down,
        )

    def _build_imported_relaxations_deps(self) -> ImportedRelaxationsDeps:
        """Build the imported-relaxations controller's deps; see the dataclass."""

        def _set_import_enabled(enabled: bool) -> None:
            if self._import_relaxations_action is not None:
                self._import_relaxations_action.setEnabled(enabled)

        return ImportedRelaxationsDeps(
            dialog_parent=lambda: self,
            set_import_enabled=_set_import_enabled,
            clear_status=lambda: self.statusBar().clearMessage(),
            active_muon_relaxation_definition=lambda: (
                self._muon_relaxation_controller.active_definition()
            ),
            refresh_muon_relaxation_rows=lambda: self._refresh_muon_relaxation_rows(),
            refresh_jobs_table=lambda: self._refresh_jobs_table(),
            refresh_muon_sites_summary=lambda: self._refresh_muon_sites_summary(),
            show_kept_folders=lambda kept: self._show_kept_folders(kept),
        )

    def _build_analysis_workspaces_deps(self) -> AnalysisWorkspacesDeps:
        """Build the analysis-workspace controller's deps; see the dataclass."""

        return AnalysisWorkspacesDeps(
            dialog_parent=lambda: self,
            completed_sources=lambda: self._completed_muon_relaxation_sources(),
            refresh_analysis_page=lambda: self._refresh_relaxation_analysis_context(),
            refresh_navigator=lambda: self._refresh_navigator(),
            refresh_journal_view=lambda: self._refresh_journal_view(),
            refresh_muon_sites_summary=lambda: self._refresh_muon_sites_summary(),
            refresh_pinned_site_consumers=lambda: self._refresh_pinned_site_consumers(),
            active_route=lambda: self._project_navigator.active_route(),
            open_route=lambda route: self._set_active_route(route),
            shutting_down=lambda: self._shutting_down,
        )

    def _build_uep_deps(self) -> UepDeps:
        """Build the UEP controller's deps; see the dataclass."""

        return UepDeps(
            dialog_parent=lambda: self,
            refresh_workflow_defaults=lambda: self._refresh_workflow_defaults(self._uep_view),
            note_kpoint_adjustments=lambda expansion: self._note_kpoint_adjustments(expansion),
            refresh_journal_view=lambda: self._refresh_journal_view(),
            refresh_generation_table=lambda: self._refresh_muon_generation_table(),
            shutting_down=lambda: self._shutting_down,
        )

    def _build_hyperfine_coupling_deps(self) -> HyperfineCouplingDeps:
        """Build the hyperfine controller's deps: the host-mismatch dialogs."""

        return HyperfineCouplingDeps(
            handle_structure_host_mismatch=lambda mismatch: self._handle_structure_host_mismatch(
                mismatch
            ),
            handle_site_host_mismatch=lambda mismatch: self._handle_site_host_mismatch(mismatch),
        )

    def _build_quantum_corrections_deps(self) -> QuantumCorrectionsDeps:
        """Build the quantum-corrections controller's deps: its window dialogs."""

        return QuantumCorrectionsDeps(
            handle_structure_host_mismatch=lambda mismatch: self._handle_structure_host_mismatch(
                mismatch
            ),
            handle_site_host_mismatch=lambda mismatch: self._handle_site_host_mismatch(mismatch),
            refresh_site_label_views=self._refresh_views_after_quantum_site_rename,
            show_warning=lambda title, text: QMessageBox.warning(self, title, text),
            shutting_down=lambda: self._shutting_down,
        )

    def _refresh_views_after_quantum_site_rename(self) -> None:
        self._refresh_hyperfine_coupling_view()
        self._refresh_energy_barriers_view()

    def _build_energy_barriers_deps(self) -> EnergyBarriersDeps:
        """Build the energy-barriers (NEB) controller's deps: its window dialogs."""

        return EnergyBarriersDeps(
            handle_structure_host_mismatch=lambda mismatch: self._handle_structure_host_mismatch(
                mismatch
            ),
            handle_site_host_mismatch=lambda mismatch: self._handle_site_host_mismatch(mismatch),
            show_warning=lambda title, text: QMessageBox.warning(self, title, text),
            dialog_parent=lambda: self,
            clean_up_removed_jobs=lambda plan, **labels: self._clean_up_removed_jobs(
                plan, **labels
            ),
            report_cleanup_outcome=self._report_cleanup_outcome,
            paths_removed=self._on_energy_barrier_paths_removed,
        )

    def _on_energy_barrier_paths_removed(self) -> None:
        self._refresh_jobs_table()
        self._refresh_journal_view()

    def _run_for_job(self, job_id: UUID) -> CalculationRun | None:
        return self._job_execution.run_for_job(job_id)

    def _refresh_hyperfine_coupling_view(self) -> None:
        """Re-render the hyperfine-coupling view via its controller."""

        self._hyperfine_coupling_controller.refresh_view()

    def _refresh_quantum_corrections_view(self) -> None:
        """Re-render the quantum-corrections view via its controller.

        Thin wrapper (delegates to
        :meth:`QuantumCorrectionsController.refresh_view`) kept on ``MainWindow``
        because it has many callers across document/route changes, submission,
        configuration, rename, and running jobs again.
        """

        self._quantum_corrections_controller.refresh_view()

    def _refresh_energy_barriers_view(self) -> None:
        """Re-render the energy-barriers view via its controller.

        Thin wrapper (delegates to
        :meth:`EnergyBarriersController.refresh_view`) kept on ``MainWindow``
        because it has many callers across document/route changes, submission, path
        generation, source removal, and result configuration. The controller
        no-ops when the lazily-created view does not yet exist, exactly as the
        former inline body did.
        """

        self._energy_barriers_controller.refresh_view()

    def _job_output_directory(self, job: Job) -> Path | None:
        return job_folder(
            getattr(self, "_workflow_controllers", ()),
            self._document,
            job,
            quantum_seed_name=lambda definition, key: quantum_seed_name_for_key(
                self._document, definition, key
            ),
        )

    def _job_seed_name(self, job: Job) -> str:
        return job_seed_name(job, self._job_output_directory)

    def _on_load_structure_requested(self) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        if self._site_generation_controller.blocks_host_change():
            return

        file_path, _selected_filter = self._file_dialogs.get_open_file_name(
            title="Select Structure File",
            file_filter="Structure Files (*.cif *.CIF *.mcif *.MCIF *.cell *.CELL);;All Files (*)",
            context="load_structure",
        )
        if not file_path:
            return

        try:
            structure = self._project_service.import_structure_file(
                self._document,
                Path(file_path),
                make_active=True,
            )
        except (StructureLoadError, ValueError) as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return

        self._refresh_host_structure_view()
        self._refresh_sweep_simulation_cells()
        self._refresh_workflow_defaults(self._pristine_relaxation_view)
        self._pristine_relaxation_controller.refresh_view()
        self._refresh_convergence_overview()
        self._refresh_navigator()
        self._refresh_journal_view()
        self._set_active_route("host_structure")
        self.statusBar().showMessage(
            f"Loaded host structure '{structure.label}' ({len(structure.species)} atoms)"
        )

    def _on_muon_relaxation_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state["muon_relaxation_form"] = state
        self._refresh_muon_relaxation_rows()

    def _on_relaxation_analysis_current_workspace_changed(self, workspace_id: str) -> None:
        # Keep the project tree on the workspace whose tab is showing, but only
        # while the Analysis page is the one in view.
        if self._main_stack.currentWidget() is not self._relaxation_analysis_view:
            return
        route = self._relaxation_analysis_workspace_route(str(workspace_id))
        if self._project_navigator.active_route() != route:
            self._set_active_route(route)

    # --- Muon Sites summary (pinned analyses) -------------------------------

    @staticmethod
    def _muon_sites_summary_placeholder(charge_state: str) -> str:
        label = muon_charge_state_label(charge_state)
        return (
            f"No {label} analysis is pinned yet. Use 'Go to Analysis' to cluster the "
            f"relaxed {label} sites, then choose 'Pin to Summary' to bring the distinct "
            "sites here."
        )

    def _refresh_muon_sites_summary(self) -> None:
        summary_view = getattr(self, "_muon_sites_summary_view", None)
        if summary_view is None:
            return
        # The summary's two viewers draw only while the page shows, so a refresh
        # behind another page updates the panels without rendering.
        host = (
            self._project_service.active_host_structure(self._document)
            if self._document is not None
            else None
        )
        for charge_state in (POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE):
            summary_view.set_panel(
                charge_state,
                snapshot=validated_muon_sites_summary_snapshot(self._document, charge_state),
                host_structure=host,
                placeholder_message=self._muon_sites_summary_placeholder(charge_state),
            )

    def _refresh_pinned_site_consumers(self) -> None:
        # Quantum corrections and energy barriers list the pinned sites by
        # default, so a pin change re-renders them; the Magnetic Fields site
        # dialog reads the pins each time it opens.
        self._refresh_hyperfine_coupling_view()
        self._refresh_quantum_corrections_view()
        self._refresh_energy_barriers_view()

    def _on_convergence_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state["convergence_form"] = state
        self._convergence_controller.refresh_view()

    def _refresh_sweep_simulation_cells(self) -> None:
        self._convergence_controller.refresh_simulation_cell()
        self._kpoint_convergence_controller.refresh_simulation_cell()

    def _refresh_all_supercell_options(self) -> None:
        self._refresh_sweep_simulation_cells()
        self._refresh_workflow_supercell_options(self._pristine_relaxation_view)
        self._refresh_workflow_supercell_options(self._muon_relaxation_view)
        self._refresh_workflow_supercell_options(self._uep_view)

    def _open_supercell_editor(
        self,
        *,
        source_structure: Structure,
        workflow_stage: WorkflowStage | None,
        initial_matrix: list[list[int]] | None = None,
        select_mode: bool = False,
    ) -> SupercellEditorDialog | None:
        if self._document is None:
            return None
        dialog = SupercellEditorDialog(
            document=self._document,
            project_service=self._project_service,
            source_structure=source_structure,
            workflow_stage=workflow_stage,
            initial_matrix=initial_matrix,
            select_mode=select_mode,
            parent=self,
        )
        result = dialog.exec()
        self._refresh_all_supercell_options()
        self._refresh_navigator()
        self._persist_document_ui_state(
            failure_message="Supercells were updated, but autosave failed."
        )
        if select_mode and result != dialog.DialogCode.Accepted:
            return None
        return dialog

    def _on_host_supercell_editor_requested(self) -> None:
        self._edit_supercell(
            None,
            initial_matrix=self._project_service.supercell_editor_seed_matrix,
            apply=lambda _dialog: None,
            missing_source="Load a host structure before editing supercells.",
            select_mode=False,
        )

    def _edit_supercell(
        self,
        workflow_stage: WorkflowStage | None,
        *,
        initial_matrix: Callable[[ProjectDocument], Sequence[Sequence[int]] | None],
        apply: Callable[[SupercellEditorDialog], None],
        missing_source: str,
        missing_source_title: str = "No host structure",
        source_structure: Callable[[ProjectDocument], Structure | None] | None = None,
        select_mode: bool = True,
    ) -> None:
        """Open the Supercell Editor for one workflow and apply what it returns.

        Every entry point to the editor goes through here: it needs an open
        project and a source structure (the active host unless
        ``source_structure`` names another), then opens the editor, whose
        close refreshes the supercell options, the project tree and autosave.
        ``initial_matrix`` is read only once both guards pass, and ``apply``
        runs only when the editor returns (in select mode, when it is accepted).
        """

        document = self._document
        if document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        if source_structure is None:
            source = self._project_service.active_host_structure(document)
        else:
            source = source_structure(document)
        if source is None:
            QMessageBox.warning(self, missing_source_title, missing_source)
            return
        matrix = initial_matrix(document)
        dialog = self._open_supercell_editor(
            source_structure=source,
            workflow_stage=workflow_stage,
            initial_matrix=None if matrix is None else [list(row) for row in matrix],
            select_mode=select_mode,
        )
        if dialog is not None:
            apply(dialog)

    def _select_simulation_cell_in_editor(
        self,
        workflow_stage: WorkflowStage,
        *,
        initial_matrix: Callable[[], list[list[int]]],
        on_selected: Callable[[UUID], None],
    ) -> None:
        """Pick or create a convergence sweep's simulation cell in the editor.

        ``on_selected`` receives the id of the cell the editor returns with.
        """

        def apply(dialog: SupercellEditorDialog) -> None:
            structure_id = dialog.selected_structure_id()
            if structure_id is not None:
                on_selected(structure_id)

        self._edit_supercell(
            workflow_stage,
            initial_matrix=lambda _document: initial_matrix(),
            apply=apply,
            missing_source="Load a host structure before creating a saved supercell.",
        )

    def _edit_row_supercell(
        self,
        view: RowTableWidget,
        payload: object,
        workflow_stage: WorkflowStage,
        *,
        workflow_name: str,
        missing_source: str,
        missing_source_title: str = "No host structure",
        source_structure: Callable[[ProjectDocument], Structure | None] | None = None,
    ) -> None:
        """Choose one row's supercell in the editor and set it on the row."""

        if self._document is None:
            return
        try:
            row_id = int(payload)
        except (TypeError, ValueError):
            return

        def row_matrix(_document: ProjectDocument) -> SupercellMatrix:
            existing = view.row_supercell_matrix(row_id)
            return existing if existing is not None else IDENTITY_SUPERCELL

        def apply(dialog: SupercellEditorDialog) -> None:
            matrix = coerce_supercell_matrix(dialog.selected_matrix())
            if matrix is None:
                return
            view.set_row_supercell(row_id, matrix)
            self.statusBar().showMessage(f"Selected {workflow_name} simulation cell")

        self._edit_supercell(
            workflow_stage,
            initial_matrix=row_matrix,
            apply=apply,
            missing_source=missing_source,
            missing_source_title=missing_source_title,
            source_structure=source_structure,
        )

    @staticmethod
    def _grid_key(grid: tuple[int, int, int]) -> str:
        return f"{int(grid[0])}x{int(grid[1])}x{int(grid[2])}"

    def _on_kpoint_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state["kpoint_convergence_form"] = state
        self._kpoint_convergence_controller.refresh_view()

    def _on_remove_grids_from_kpoint_convergence(self, grids: list[tuple[int, int, int]]) -> None:
        if self._document is None:
            self._kpoint_convergence_view.remove_grids(grids)
            return

        self._kpoint_convergence_controller.remove_rows({self._grid_key(grid) for grid in grids})

        self._kpoint_convergence_view.remove_grids(grids)
        self._kpoint_convergence_controller.refresh_view()

    def _project_defaults_strips(self) -> tuple[ProjectDefaultsStrip, ...]:
        return (
            self._convergence_view.project_defaults_strip,
            self._kpoint_convergence_view.project_defaults_strip,
            self._pristine_relaxation_view.project_defaults_strip,
            self._uep_view.project_defaults_strip,
            self._muon_relaxation_view.project_defaults_strip,
        )

    def _refresh_convergence_project_defaults(self) -> None:
        """Push the project defaults to every strip and to the convergence sweeps."""

        project = self._document.project if self._document is not None else None
        summary = ProjectDefaultsSummary.from_project(project)
        for strip in self._project_defaults_strips():
            strip.set_defaults(summary)
            strip.setEnabled(project is not None)
        self._convergence_view.set_xc_functional_default(summary.xc_functional)
        self._kpoint_convergence_view.set_xc_functional_default(summary.xc_functional)
        self._kpoint_convergence_view.set_cutoff_default(summary.cutoff_energy)

    def _refresh_project_defaults_views(self) -> None:
        """Re-apply the project defaults to every workflow view."""

        self._refresh_convergence_project_defaults()
        self._refresh_workflow_defaults(self._pristine_relaxation_view)
        self._refresh_workflow_defaults(self._muon_relaxation_view)
        self._refresh_workflow_defaults(self._uep_view, supercells=False)

    def _on_project_xc_functional_change_requested(self, xc_functional: object) -> None:
        if self._document is None:
            return
        functional = str(xc_functional) if xc_functional else None
        if functional == self._document.project.recommended_xc_functional:
            return
        self._project_service.set_recommended_xc_functional(self._document, functional)
        self._finish_project_default_edit(
            f"Project default XC functional set to {functional}"
            if functional is not None
            else "Project default XC functional cleared"
        )

    def _on_project_cutoff_change_requested(self, cutoff_energy: object) -> None:
        """Apply a cutoff typed into the Project Defaults dialog (``None`` clears)."""

        if self._document is None:
            return
        # The pinned overview plot documents the adopted value. Drop it, or
        # typing that same number back in later would resurrect the plot for a
        # default that is now manual.
        store_convergence_overview_snapshot(self._document, CUTOFF_SNAPSHOT, None)
        if cutoff_energy is None:
            self._project_service.clear_recommended_cutoff(self._document)
            self._finish_project_default_edit("Project default cutoff cleared")
            return
        value = float(cutoff_energy)  # type: ignore[arg-type]
        host_id = self._document.project.host_structure_id
        self._project_service.set_recommended_cutoff(
            self._document,
            value,
            method=MANUAL_DEFAULT_METHOD,
            cell_id=str(host_id) if host_id else None,
        )
        self._finish_project_default_edit(f"Project default cutoff set to {value:.8g} eV")

    def _on_project_kpoint_grid_change_requested(self, kpoint_grid: object) -> None:
        """Apply a grid typed into the Project Defaults dialog (``None`` clears).

        A manual grid is taken to be for the host unit cell, which is recorded
        as its reference lattice so supercell jobs rescale it as usual.
        """

        if self._document is None:
            return
        store_convergence_overview_snapshot(self._document, KPOINT_SNAPSHOT, None)
        if kpoint_grid is None:
            self._project_service.clear_recommended_kpoint_grid(self._document)
            self._finish_project_default_edit("Project default k-point grid cleared")
            return
        grid = tuple(int(component) for component in kpoint_grid)  # type: ignore[union-attr]
        host = self._project_service.active_host_structure(self._document)
        self._project_service.set_recommended_kpoint_grid(
            self._document,
            grid,
            lattice_vectors=(
                [list(row) for row in host.lattice_vectors] if host is not None else None
            ),
            method=MANUAL_DEFAULT_METHOD,
            cell_id=str(host.id) if host is not None else None,
        )
        self._finish_project_default_edit(
            f"Project default k-point grid set to {grid[0]}x{grid[1]}x{grid[2]}"
        )

    def _finish_project_default_edit(self, message: str) -> None:
        self._refresh_project_defaults_views()
        self._refresh_convergence_overview()
        self._refresh_journal_view()
        self._persist_document_ui_state(
            failure_message="Could not persist the project defaults to disk"
        )
        self._refresh_navigator()
        self.statusBar().showMessage(message)

    def _workflow_source_structure(self, view) -> Structure | None:
        """Return the structure a relaxation or UEP view's rows are built on.

        UEP rows run on the selected UEP source structure; the relaxation
        workflows run on the active host.
        """

        if self._document is None:
            return None
        if view is self._uep_view:
            return self._uep_controller.selected_source_structure()
        return self._project_service.active_host_structure(self._document)

    def _refresh_workflow_defaults(self, view, *, supercells: bool = True) -> None:
        """Push the project defaults and source-structure context to a workflow view.

        ``view`` is the pristine-relaxation, muon-relaxation or UEP view. With
        ``supercells`` the view's supercell options are refreshed as well.
        """

        if self._document is None:
            view.set_host_lattice(None)
            view.set_project_defaults(
                cutoff_energy=None,
                kpoint_grid=None,
                kpoint_reference_lattice=None,
            )
            view.set_parameter_species_options(())
        else:
            structure = self._workflow_source_structure(view)
            view.set_parameter_species_options(structure_species_options(structure))
            view.set_host_lattice(
                None if structure is None else [list(row) for row in structure.lattice_vectors]
            )
            project = self._document.project
            view.set_project_defaults(
                cutoff_energy=project.recommended_cutoff_energy,
                kpoint_grid=project.recommended_kpoint_grid,
                kpoint_reference_lattice=project.recommended_kpoint_lattice,
                xc_functional=project.recommended_xc_functional,
            )
        if supercells:
            self._refresh_workflow_supercell_options(view)

    def _refresh_workflow_supercell_options(self, view) -> None:
        """Offer a workflow view the saved supercells of its source structure."""

        structure = self._workflow_source_structure(view)
        if self._document is None or structure is None:
            view.set_supercell_options([])
            return
        assets = self._project_service.list_workflow_supercell_assets(
            self._document,
            host_structure_id=structure.id,
        )
        view.set_supercell_options(
            [
                {"label": asset.label, "matrix": [list(row) for row in asset.matrix]}
                for asset in assets
            ]
        )

    def _on_pristine_relaxation_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state["pristine_relaxation_form"] = state
        self._pristine_relaxation_controller.refresh_view()

    @staticmethod
    def _removed_row_labels(payload: object) -> set[str]:
        if payload is None:
            return set()
        if isinstance(payload, str):
            values = [payload]
        elif isinstance(payload, (list, tuple, set)):
            values = list(payload)
        else:
            values = [payload]
        return {str(value).strip() for value in values if str(value).strip()}

    def _connect_workflow_row_actions(self, view, controller: WorkflowJobController) -> None:
        """Route a workflow table's row-menu Cancel / Inspect Failure signals.

        Each signal carries the selected row keys; the generic handlers below
        resolve them to jobs through ``controller.jobs_for_keys``. The row
        menu's job-folder entries ask ``controller.job_files_exports``.
        """

        view.cancel_requested.connect(partial(self._cancel_workflow_rows, controller))
        view.inspect_failed_requested.connect(partial(self._inspect_workflow_rows, controller))
        view.job_files_menu.set_provider(controller.job_files_exports)

    def _cancel_workflow_rows(self, controller, keys: object) -> None:
        """Cancel the jobs behind a workflow table's rows (row menu "Cancel")."""

        self._jobs.cancel_jobs(controller.jobs_for_keys(list(keys)))

    def _inspect_workflow_rows(self, controller, keys: object) -> None:
        """Show failure diagnostics for a workflow table's rows (row menu "Inspect Failure")."""

        self._jobs.inspect_failed_jobs(controller.jobs_for_keys(list(keys)))

    def _on_remove_pristine_relaxation_rows(self, payload: object) -> None:
        labels = self._removed_row_labels(payload)
        if self._document is None or not labels:
            return
        controller = self._pristine_relaxation_controller
        definitions = controller.definitions()
        key_matches = controller.row_key_matcher(labels)
        should_autosave = self._relaxation_removal_touches_project_files(definitions, key_matches)
        if not self._cleanup_relaxation_rows(controller.jobs_for_removed_rows(labels)):
            self._restore_pristine_relaxation_form_from_definition()
            return
        controller.remove_rows(labels)
        controller.refresh_view()
        if should_autosave:
            self._persist_document_ui_state(
                failure_message="Pristine relaxation rows were removed, but autosave failed."
            )

    def _on_remove_muon_relaxation_rows(self, payload: object) -> None:
        labels = self._removed_row_labels(payload)
        if self._document is None or not labels:
            return
        controller = self._muon_relaxation_controller
        definitions = controller.definitions()
        key_matches = controller.row_key_matcher(labels)
        removed_site_ids = controller.site_ids_for_removed_rows(definitions, labels)
        should_autosave = self._relaxation_removal_touches_project_files(definitions, key_matches)
        if not self._cleanup_relaxation_rows(controller.jobs_for_removed_rows(labels)):
            if controller.restore_form_from_definition():
                self._refresh_muon_relaxation_rows()
            return
        controller.remove_rows(labels)
        controller.remove_labels_from_form_state(labels)
        site_status_changed = controller.unstage_unlisted_sites(removed_site_ids)
        self._refresh_muon_relaxation_rows()
        if should_autosave or self._site_status_revert_touches_project_files(
            removed_site_ids, site_status_changed
        ):
            self._persist_document_ui_state(
                failure_message="Muon relaxation rows were removed, but autosave failed."
            )

    def _site_status_revert_touches_project_files(self, site_ids: set[UUID], changed: bool) -> bool:
        if not changed:
            return False
        return any(self._entity_file("sites", site_id).exists() for site_id in site_ids)

    def _relaxation_removal_touches_project_files(
        self,
        definitions: list,
        key_matches: Callable[[object], bool],
    ) -> bool:
        if self._document is None:
            return False

        for definition in definitions:
            config_file = self._entity_file("calculations", definition.id)
            config = definition.parameters.get("configuration", {})
            rows = config.get("rows", []) if isinstance(config, dict) else []
            if config_file.exists() and isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if any(key_matches(key) for key in stored_row_keys(row)):
                        return True

            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict):
                continue
            for key, raw_job_id in mapping.items():
                if not key_matches(key):
                    continue
                try:
                    job_id = UUID(str(raw_job_id))
                except (TypeError, ValueError):
                    continue
                job = self._document.jobs.get(job_id)
                if self._job_touches_project_files(job_id, job):
                    return True
        return False

    def _job_touches_project_files(self, job_id: UUID, job: Job | None = None) -> bool:
        if self._document is None:
            return False
        if self._entity_file("jobs", job_id).exists():
            return True
        for run in self._document.runs.values():
            if run.job_id != job_id:
                continue
            if self._entity_file("runs", run.id).exists():
                return True
            if run.result_id is not None and self._entity_file("results", run.result_id).exists():
                return True
        if job is not None:
            job_dir = self._job_output_directory(job)
            if job_dir is not None and job_dir.exists():
                return True
        return False

    def _entity_file(self, entity_type: str, entity_id: UUID) -> Path:
        if self._document is None:
            return Path("")
        return self._document.project_root / "entities" / entity_type / f"{entity_id}.json"

    def _cleanup_relaxation_rows(self, jobs: Sequence[Job]) -> bool:
        """Delete the jobs bound to removed relaxation rows, asking what to do with their files.

        Returns ``False`` when the user cancels a file prompt or a file
        operation fails, leaving the rows in place.
        """

        if self._document is None:
            return True
        plan = []
        for job in jobs:
            job_dir = self._job_output_directory(job)
            archive_root = (
                self._relaxation_archive_root_for_job(job, job_dir) if job_dir is not None else None
            )
            plan.append(self._job_file_cleanup.plan_item([job], archive_root=archive_root))
        outcome = self._clean_up_removed_jobs(
            plan,
            title="Remove relaxation job with files",
            describe=lambda item: (
                f"Relaxation job '{item.label}' has associated files.\n\n{item.directory}"
            ),
            keep_text="or Cancel to keep the relaxation row.",
        )
        removed = sum(len(item.jobs) for item in outcome.cleaned)
        if removed:
            self._report_cleanup_outcome(f"Removed {plural(removed, 'job')}", outcome)
        return outcome.completed

    def _clean_up_removed_jobs(
        self,
        plan: Sequence[JobCleanupItem],
        *,
        title: str,
        describe: Callable[[JobCleanupItem], str],
        keep_text: str,
    ) -> JobCleanupOutcome:
        """Run a row-removal clean-up plan, asking per item what to do with its files.

        Stops at the first file operation that fails and reports it. The caller says
        what happened in the status bar, where its own message would replace this one.
        """

        multi = len(plan) > 1
        outcome = self._job_file_cleanup.execute(
            plan,
            choose_disposition=lambda item: self._jobs.prompt_job_file_disposition(
                title=title,
                text=describe(item),
                keep_text=keep_text,
                include_apply_all=multi,
            ),
            stop_on_error=True,
        )
        for failure in outcome.failures:
            QMessageBox.warning(
                self,
                "File cleanup failed",
                f"Could not update files for job '{failure.item.label}':\n{failure.reason}",
            )
        return outcome

    def _report_cleanup_outcome(self, done: str, outcome: JobCleanupOutcome) -> None:
        """Put a clean-up's counts in the status bar and say why any folder stayed."""

        self.statusBar().showMessage(
            folder_outcome_status(
                done,
                deleted=len(outcome.deleted),
                archived=len(outcome.archived),
                kept=len(outcome.kept_folders),
            )
        )
        self._show_kept_folders(outcome.kept_folders)

    def _show_kept_folders(
        self, kept: Sequence[KeptFolder], *, title: str = "Folders kept"
    ) -> None:
        """Say why folders stayed, however many, with their names under Details."""

        if kept:
            message = kept_folders_message(kept)
            show_detailed_warning(
                self,
                title,
                message.summary,
                informative_text=message.instruction,
                details=message.details,
            )

    def _relaxation_archive_root_for_job(self, job: Job, job_dir: Path) -> Path | None:
        if self._document is None:
            return None

        for definition in self._document.calculations.values():
            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict) or str(job.id) not in {
                str(value) for value in mapping.values()
            }:
                continue
            base = definition_output_directory(self._document, definition)
            if base is None:
                continue
            if (
                definition.workflow_stage == WorkflowStage.MUON_RELAXATION
                and job_dir.parent.name
                in {
                    "positivemuon",
                    "muonium",
                }
            ):
                return job_dir.parent
            return base
        return job_dir.parent

    def _restore_pristine_relaxation_form_from_definition(self) -> None:
        definition = self._pristine_relaxation_controller.active_definition()
        if definition is None:
            return
        config = deserialize_pristine_relaxation_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return
        state = serialize_pristine_relaxation_config(config)
        for row in state.get("rows", []):
            if isinstance(row, dict) and "params" not in row:
                row["params"] = dict(row.get("parameter_values", {}))
        self._pristine_relaxation_view.set_form_state(state)
        if self._document is not None:
            self._document.ui_state["pristine_relaxation_form"] = state
        self._pristine_relaxation_controller.refresh_view()

    def _definition_for_pristine_relaxation_label(self, label: str):
        definition = self._pristine_relaxation_controller.active_definition()
        if definition is None:
            return None, None

        self._pristine_relaxation_controller.sync_mappings(definition)
        config = deserialize_pristine_relaxation_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return None, None
        labels = {row.label for row in config.rows}
        if label in labels:
            return definition, config
        return None, None

    def _refresh_muon_relaxation_rows(self) -> None:
        self._muon_relaxation_controller.refresh_view()
        self._refresh_relaxation_analysis_context()

    def _completed_muon_relaxation_sources(self) -> list[CompletedRelaxationAnalysisSource]:
        if self._document is None:
            return []
        host = self._project_service.active_host_structure(self._document)
        if host is None:
            return []
        controller = self._muon_relaxation_controller
        rows: list[MuonRelaxationRowJob] = []
        for definition in controller.definitions():
            if host.id not in definition.input_structure_ids:
                continue
            config = deserialize_muon_relaxation_config(
                definition.parameters.get("configuration", {})
            )
            if config is None:
                continue
            rows.extend(
                (
                    row.label,
                    row.charge_state,
                    row.site_id,
                    controller.job_for_label(definition, row.label, row.charge_state),
                )
                for row in config.rows
            )
        return completed_muon_relaxation_sources(
            self._document,
            rows,
            run_for_job=self._run_for_job,
            total_energy_for_job=partial(total_energy_for_job, self._document),
        )

    def _comparison_structure_options_for_relaxation_analysis(self) -> list[tuple[UUID, str, bool]]:
        if self._document is None:
            return []

        options: list[tuple[UUID, str, bool]] = []
        host = self._project_service.active_host_structure(self._document)
        if host is not None:
            options.append((host.id, f"Host structure - {host.label}", False))

        definition = self._pristine_relaxation_controller.active_definition()
        fix_all_cell_by_label: dict[str, bool] = {}
        if definition is not None:
            config = deserialize_pristine_relaxation_config(
                definition.parameters.get("configuration", {})
            )
            if config is not None:
                fix_all_cell_by_label = {row.label: bool(row.fix_all_cell) for row in config.rows}

        for job in self._pristine_relaxation_controller.completed_jobs():
            fixed_cell = fix_all_cell_by_label.get(job.job_label, True)
            label = f"Pristine relaxed - {job.job_label}"
            if not fixed_cell:
                label = f"{label} (free cell; unavailable)"
            options.append((job.structure_id, label, not fixed_cell))
        return options

    def _default_displacement_reference_id(
        self, comparison_options: list[tuple[UUID, str, bool]]
    ) -> UUID | None:
        """The newest completed fixed-cell pristine relaxation, else ``None`` (the host)."""

        usable = {
            structure_id for structure_id, _label, blocked in comparison_options if not blocked
        }
        return newest_completed_structure_id(
            self._document,
            [
                job
                for job in self._pristine_relaxation_controller.completed_jobs()
                if job.structure_id in usable
            ],
        )

    def _refresh_relaxation_analysis_context(
        self, *, force: bool = False, current_workspace_id: str | None = None
    ) -> None:
        """Rebuild the Relaxation Analysis page from the document, if it is showing.

        Every workspace tab holds a 3D viewer, so a hidden page is only marked
        stale; the route dispatch rebuilds a stale page as it shows it. ``force``
        rebuilds now, for that dispatch, opening ``current_workspace_id``'s tab.
        A project change defers the rebuild to one call at its end. With no
        project the page is always cleared, which builds no viewer.
        """

        self._relaxation_analysis_context_stale = True
        view = self._relaxation_analysis_view
        if (
            not force
            and self._document is not None
            and (
                self._deferring_relaxation_analysis_refresh
                or self._main_stack.currentWidget() is not view
            )
        ):
            return
        self._relaxation_analysis_context_stale = False
        if self._document is None:
            view.set_context(
                workspaces=[],
                sources=[],
                structures={},
                host_structure=None,
                comparison_options=[],
            )
            return
        comparison_options = self._comparison_structure_options_for_relaxation_analysis()
        view.set_context(
            workspaces=analysis_workspaces(self._document),
            sources=self._completed_muon_relaxation_sources(),
            structures=self._document.structures,
            host_structure=self._project_service.active_host_structure(self._document),
            comparison_options=comparison_options,
            current_workspace_id=current_workspace_id,
            default_comparison_id=self._default_displacement_reference_id(comparison_options),
        )

    def _refresh_stale_relaxation_analysis_context(
        self, current_workspace_id: str | None = None
    ) -> None:
        """Bring the Relaxation Analysis page up to date before it is shown."""

        if self._relaxation_analysis_context_stale and not (
            self._deferring_relaxation_analysis_refresh
        ):
            self._refresh_relaxation_analysis_context(
                force=True, current_workspace_id=current_workspace_id
            )

    def _on_pristine_relaxation_supercell_edit_requested(self, payload: object) -> None:
        self._edit_row_supercell(
            self._pristine_relaxation_view,
            payload,
            WorkflowStage.PRISTINE_RELAXATION,
            workflow_name="pristine-relaxation",
            missing_source=(
                "Load a host structure before configuring pristine-relaxation supercells."
            ),
        )

    def _on_muon_relaxation_supercell_edit_requested(self, payload: object) -> None:
        self._edit_row_supercell(
            self._muon_relaxation_view,
            payload,
            WorkflowStage.MUON_RELAXATION,
            workflow_name="muon-relaxation",
            missing_source="Load a host structure before configuring muon-relaxation supercells.",
        )

    def _note_kpoint_adjustments(self, expansion: AdapterExpansion) -> None:
        """Keep the rows whose project k-point grid was rescaled for the coming submission."""

        self._kpoint_adjustments = expansion.kpoint_adjustments

    def _set_submitted_rows(self, keys: Sequence[object]) -> None:
        """Word the rescaled-grid note for the rows being submitted; ``()`` ends the note."""

        submitted = {str(key) for key in keys}
        self._kpoint_status_note = kpoint_adjustment_status(
            [item for item in self._kpoint_adjustments if item.row_key in submitted]
        )
        self._kpoint_adjustments = ()

    def _with_kpoint_status_note(self, message: str) -> str:
        """Return a submission's status ``message`` with the rescaled-grid note, if any."""

        return f"{message}; {self._kpoint_status_note}" if self._kpoint_status_note else message

    def _on_pristine_relaxation_configured(self, config: PristineRelaxationConfig) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return

        host = self._project_service.active_host_structure(self._document)
        if host is None:
            QMessageBox.warning(
                self,
                "No host structure",
                "Load a host structure before generating pristine-relaxation jobs.",
            )
            return

        try:
            expansion = expand_pristine_relaxation_jobs(
                host_structure=host,
                config=config,
                project_cutoff_energy=self._document.project.recommended_cutoff_energy,
                project_kpoint_grid=self._document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=self._document.project.recommended_kpoint_lattice,
                project_xc_functional=self._document.project.recommended_xc_functional,
            )
        except (ValueError, StructureBridgeError) as exc:
            QMessageBox.warning(self, "Pristine relaxation setup invalid", str(exc))
            return
        self._note_kpoint_adjustments(expansion)

        output_root = self._document.project_root / "files" / "pristine_relaxation"
        self._pristine_relaxation_controller.store_configuration(
            serialize_pristine_relaxation_config(config),
            input_structure_id=host.id,
            output_directory=output_root,
            register=self._project_service.register_pristine_relaxation_definition,
        )

        self._pristine_relaxation_controller.refresh_view()
        self._refresh_journal_view()
        self.statusBar().showMessage(
            f"Configured {len(config.rows)} pristine-relaxation jobs; input files will be generated on submission"
        )

    def _on_muon_relaxation_configured(self, config: MuonRelaxationConfig) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        host = self._project_service.active_host_structure(self._document)
        if host is None:
            QMessageBox.warning(
                self,
                "No host structure",
                "Load a host structure before generating muon-relaxation jobs.",
            )
            return
        try:
            expansion = expand_muon_relaxation_jobs(
                host_structure=host,
                config=config,
                project_cutoff_energy=self._document.project.recommended_cutoff_energy,
                project_kpoint_grid=self._document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=self._document.project.recommended_kpoint_lattice,
                project_xc_functional=self._document.project.recommended_xc_functional,
            )
        except (ValueError, StructureBridgeError) as exc:
            QMessageBox.warning(self, "Muon relaxation setup invalid", str(exc))
            return
        self._note_kpoint_adjustments(expansion)
        output_root = self._document.project_root / "files" / "muon_relaxation"
        self._muon_relaxation_controller.store_configuration(
            serialize_muon_relaxation_config(config),
            input_structure_id=host.id,
            output_directory=output_root,
            register=self._project_service.register_muon_relaxation_definition,
        )
        self._refresh_muon_relaxation_rows()
        self._refresh_journal_view()
        self.statusBar().showMessage(
            f"Configured {len(config.rows)} muon-relaxation jobs; input files will be generated on submission"
        )

    def _on_hyperfine_coupling_source_label_renamed(self, key: str, label: str) -> None:
        if self._document is None or self._shutting_down:
            return
        if not self._hyperfine_coupling_controller.rename_source(key, label):
            return
        self._refresh_quantum_corrections_view()
        self._refresh_energy_barriers_view()
        self._persist_document_ui_state(
            failure_message="Hyperfine Coupling site label was updated, but autosave failed."
        )

    def _store_unedited_magnetic_field_forms(self) -> None:
        """Store the state of each Magnetic Fields form that has none, as its page shows it.

        A form never edited stores nothing, though its page shows a default
        muon site; the hyperfine hand-off reads the stored forms, so it must
        see that site too.
        """

        if self._document is None:
            return
        if not self._document.ui_state.get(self._MAGNETIC_FIELD_FORM_KEY):
            self._on_magnetic_field_form_state_changed(self._magnetic_field_view.collect_state())
        for workspace in self._magnetic_field_workspaces_state():
            workspace_id = str(workspace.get("id", "")).strip()
            widget = self._magnetic_field_workspace_widgets.get(workspace_id)
            if widget is not None and not workspace.get("form"):
                self._on_magnetic_field_workspace_form_state_changed(
                    workspace_id, widget.collect_state()
                )

    def _on_use_hyperfine_in_magnetic_fields(self, result_id: str) -> None:
        """Fill the Hyperfine coupling of every Magnetic Fields form showing the result's site.

        Nothing changes when no form shows the site. A form holding a different
        coupling is overwritten only after confirmation; a form whose contact
        shell cannot give a coupling is skipped and named.
        """

        if self._document is None or self._shutting_down:
            return
        result = self._hyperfine_coupling_controller.hyperfine_result(result_id)
        if result is None:
            return
        title = "Use in Magnetic Fields"
        self._store_unedited_magnetic_field_forms()
        try:
            targets = plan_magnetic_field_handoff(self._document, result)
        except HyperfineError as exc:
            QMessageBox.information(self, title, str(exc))
            return
        if not targets:
            QMessageBox.information(
                self,
                title,
                "No Magnetic Fields form shows this muon site. Select the site on the Magnetic "
                "Fields page or in a saved workspace first; nothing was changed.",
            )
            return
        overwritten = [target for target in targets if target.overwrites]
        if overwritten:
            listing = "\n".join(
                f"  - {target.title}: {target.existing_display} → {target.new_text} Å⁻³"
                for target in overwritten
            )
            if not workflow_table_utils.confirm_destructive(
                self,
                "Replace hyperfine coupling",
                "These Magnetic Fields forms already hold a different hyperfine coupling:",
                detail=f"{listing}\n\nReplace it with the computed value?",
                accept_label="Replace",
            ):
                return
        changed = apply_magnetic_field_handoff(self._document, targets)
        self._refresh_magnetic_field_view()
        self._persist_document_ui_state(
            failure_message="The hyperfine coupling was handed to Magnetic Fields, but autosave failed."
        )
        lines = [
            f"{target.title}: {target.new_text} Å⁻³"
            for target in targets
            if target.coupling is not None
        ]
        skipped = [f"{target.title}: {target.error}" for target in targets if target.error]
        ambiguous = [f"{target.title}: {target.warning}" for target in targets if target.warning]
        message = f"Filled the hyperfine coupling of {plural(changed, 'Magnetic Fields form')}."
        if lines:
            message += "\n\n" + "\n".join(lines)
        if ambiguous:
            message += "\n\nWarning:\n" + "\n".join(ambiguous)
        if skipped:
            message += "\n\nSkipped:\n" + "\n".join(skipped)
        QMessageBox.information(self, title, message)

    def _on_uep_supercell_edit_requested(self, payload: object) -> None:
        self._edit_row_supercell(
            self._uep_view,
            payload,
            WorkflowStage.UEP,
            workflow_name="UEP",
            missing_source=(
                "Select or generate a source structure before configuring UEP supercells."
            ),
            missing_source_title="No source structure",
            source_structure=lambda _document: self._uep_controller.selected_source_structure(),
        )

    def _refresh_muon_generation_table(self) -> None:
        if self._document is None:
            self._muon_generation_view.set_sites([])
            self._muon_generation_view.set_host_structure(None)
            self._muon_generation_view.set_uep_cluster_sources([])
            return

        host = (
            self._document.structures.get(self._document.project.host_structure_id)
            if self._document.project.host_structure_id is not None
            else None
        )
        self._muon_generation_view.set_host_structure(host)
        candidates = [
            site
            for site in self._document.sites.values()
            if site.status in {"generated", "candidate", "staged"}
        ]
        groups = generation_run_groups(self._document, candidates)
        by_id = {site.id: site for site in candidates}
        rows = [
            {
                "id": str(site.id),
                "label": site.label,
                "method": site.generation_method or "",
                "coordinate_warning": uep_site_coordinate_warning(site),
                "status": site.status,
                "fractional": site.initial_fractional_position,
                "cartesian": site.initial_cartesian_position,
                "source_structure_id": (
                    str(site.source_structure_id) if site.source_structure_id else ""
                ),
            }
            for group in groups
            for site in (by_id[site_id] for site_id in group.site_ids)
        ]
        self._muon_generation_view.set_sites(
            rows,
            groups=[
                {
                    "key": group.key,
                    "title": group.title,
                    "description": group.description,
                    "site_ids": [str(site_id) for site_id in group.site_ids],
                }
                for group in groups
            ],
        )
        self._muon_generation_view.set_uep_cluster_sources(
            uep_cluster_source_rows(self._document, self._uep_controller.active_job_ids())
        )
        self._refresh_navigator()

    def _refresh_host_structure_view(self, *, force: bool = False) -> None:
        if self._document is None:
            self._host_structure_view.set_structure(None)
            self._host_structure_view.set_promotable_hosts([], None)
            self._refresh_magnetic_field_view()
            self._muon_generation_view.set_active_host(None)
            return

        host = self._project_service.active_host_structure(self._document)
        promotable = self._project_service.list_promotable_host_structures(self._document)
        self._host_structure_view.set_promotable_hosts(promotable, host and host.id)
        source = self._project_service.host_structure_source(self._document, host, promotable)
        self._host_structure_view.set_structure(host, source=source, force=force)
        self._refresh_magnetic_field_view()
        self._muon_generation_view.set_active_host(host)

    def _on_active_host_change_requested(self, structure_id_str: str) -> None:
        if self._document is None:
            return
        if self._site_generation_controller.blocks_host_change():
            self._refresh_host_structure_view()
            return
        try:
            structure_id = UUID(structure_id_str)
        except (ValueError, AttributeError):
            return
        if not confirm_host_switch(self, self._project_service, self._document, structure_id):
            self._refresh_host_structure_view()
            return
        try:
            self._project_service.set_active_host_structure(self._document, structure_id)
        except ValueError:
            # Refused: put the selector back on the active host.
            self._refresh_host_structure_view()
            return
        self._refresh_host_structure_view()
        self._refresh_sweep_simulation_cells()
        self._refresh_muon_generation_table()
        self._refresh_muon_relaxation_rows()
        # Drop the summary's pins that are keyed to the old host.
        self._refresh_muon_sites_summary()
        self._persist_document_ui_state(
            failure_message="Active host changed, but autosave failed.",
        )

    def _handle_site_host_mismatch(self, mismatch: object) -> bool:
        """Show a dialog when sites were generated for a different host.

        Returns True if the user resolved the mismatch by switching to the
        source host and the caller should re-run the validation.  Returns
        False if the user cancelled.
        """
        from mufinder.services.project_service import SiteHostMismatch

        if not isinstance(mismatch, SiteHostMismatch):
            return False

        unique_source_ids = {offender[2] for offender in mismatch.offenders}
        unique_source_labels = {offender[3] for offender in mismatch.offenders}
        active_label = mismatch.active_host_label

        if len(unique_source_ids) == 1:
            source_id = next(iter(unique_source_ids))
            source_label = next(iter(unique_source_labels))
            msg = (
                f"The selected sites were generated for a different host structure "
                f"({source_label!r}) than the current active host ({active_label!r}).\n\n"
                f"Would you like to switch the active host to {source_label!r} "
                f"and proceed?"
            )
            result = workflow_table_utils.confirm_choice(
                self,
                "Host structure mismatch",
                msg,
                [("switch", "Switch host"), ("cancel", "Cancel")],
                default="cancel",
                escape="cancel",
            )
            if (
                result == "switch"
                and source_id is not None
                and self._document is not None
                and not self._site_generation_controller.blocks_host_change()
            ):
                try:
                    self._project_service.set_active_host_structure(self._document, source_id)
                    self._refresh_host_structure_view()
                    self._refresh_sweep_simulation_cells()
                    self._refresh_muon_generation_table()
                    self._refresh_muon_relaxation_rows()
                    return True
                except ValueError:
                    pass
            return False

        source_labels_str = ", ".join(repr(label) for label in sorted(unique_source_labels))
        QMessageBox.warning(
            self,
            "Host structure mismatch",
            f"The selected sites were generated for multiple different host structures "
            f"({source_labels_str}), which cannot all match the active host "
            f"({active_label!r}) simultaneously.\n\n"
            f"Please regenerate the sites for the current active host, or select only "
            f"sites that share a single source host.",
        )
        return False

    def _handle_structure_host_mismatch(self, mismatch: object) -> bool:
        """Show a dialog when structures resolve to a different host via their provenance chain."""
        from mufinder.services.project_service import StructureHostMismatch

        if not isinstance(mismatch, StructureHostMismatch):
            return False

        unique_source_ids = {offender[2] for offender in mismatch.offenders}
        unique_source_labels = {offender[3] for offender in mismatch.offenders}
        active_label = mismatch.active_host_label

        if len(unique_source_ids) == 1:
            source_id = next(iter(unique_source_ids))
            source_label = next(iter(unique_source_labels))
            msg = (
                f"The selected structures were relaxed in a supercell derived from a different host "
                f"({source_label!r}) than the current active host ({active_label!r}).\n\n"
                f"Would you like to switch the active host to {source_label!r} and proceed?"
            )
            result = workflow_table_utils.confirm_choice(
                self,
                "Host structure mismatch",
                msg,
                [("switch", "Switch host"), ("cancel", "Cancel")],
                default="cancel",
                escape="cancel",
            )
            if (
                result == "switch"
                and source_id is not None
                and self._document is not None
                and not self._site_generation_controller.blocks_host_change()
            ):
                try:
                    self._project_service.set_active_host_structure(self._document, source_id)
                    self._refresh_host_structure_view()
                    self._refresh_sweep_simulation_cells()
                    self._refresh_muon_generation_table()
                    self._refresh_muon_relaxation_rows()
                    return True
                except ValueError:
                    pass
            return False

        source_labels_str = ", ".join(repr(label) for label in sorted(unique_source_labels))
        QMessageBox.warning(
            self,
            "Host structure mismatch",
            f"The selected structures trace back to multiple different host structures "
            f"({source_labels_str}), which cannot all match the active host "
            f"({active_label!r}) simultaneously.\n\n"
            f"Please select only structures that share a single source host.",
        )
        return False

    def _refresh_magnetic_field_view(self) -> None:
        self._magnetic_field_view.set_document(self._document)
        if self._document is None:
            return
        self._magnetic_field_view.set_form_state(
            attach_magnetic_field_results(
                self._document,
                MAGNETIC_FIELD_PAGE_OWNER,
                self._document.ui_state.get(self._MAGNETIC_FIELD_FORM_KEY, {}),
            ),
            preserve_edits=True,
        )
        self._refresh_magnetic_field_workspace_views()

    def _on_host_magnetic_configuration_changed(self, payload: object) -> None:
        if self._document is None or not isinstance(payload, dict):
            return
        if self._project_service.set_host_magnetic_configuration(
            self._document,
            payload.get("atomic_spins", []),
            payload.get("propagation_vector", [0.0, 0.0, 0.0]),
        ):
            self._refresh_magnetic_field_view()
            self._refresh_navigator()

    def _on_magnetic_field_structure_changed(self, payload: object) -> None:
        if self._document is None or not isinstance(payload, dict):
            return
        self._document.project.magnetic_structure = payload
        host_id = self._document.project.host_structure_id
        host = self._document.structures.get(host_id) if host_id is not None else None
        if host is not None:
            host.properties["magnetic_structure"] = payload
        self._refresh_navigator()

    def _refresh_polarisation_view(self) -> None:
        self._polarisation_view.set_document(
            self._document,
            None
            if self._document is None
            else self._document.ui_state.get(self._POLARISATION_FORM_KEY, {}),
        )

    def _open_polarisation_for(self, owner: str) -> None:
        """Show the Polarisation page with ``owner``'s field results selected."""

        self._set_active_route("polarisation")
        self._polarisation_view.select_sources([owner])

    def _on_polarisation_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state[self._POLARISATION_FORM_KEY] = state

    def _on_polarisation_simulations_changed(self) -> None:
        if self._document is None or self._shutting_down:
            return
        self._refresh_navigator()
        self._persist_document_ui_state(
            failure_message="The polarisation simulation was kept, but autosave failed."
        )
        self._refresh_journal_view()

    def _on_magnetic_field_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state[self._MAGNETIC_FIELD_FORM_KEY] = store_magnetic_field_results(
            self._document, MAGNETIC_FIELD_PAGE_OWNER, state, journal_title="Magnetic Fields"
        )
        self._refresh_journal_view()

    def _normalize_magnetic_field_workspace_title(self, title: object) -> str:
        normalized = str(title).strip()
        return normalized or "Magnetic Field Workspace"

    def _replace_magnetic_field_workspace(
        self, workspace_id: str, updates: Mapping[str, object]
    ) -> bool:
        if self._document is None:
            return False
        workspaces = list(self._magnetic_field_workspaces_state())
        updated = False
        for index, workspace in enumerate(workspaces):
            if str(workspace.get("id", "")).strip() != workspace_id:
                continue
            replacement = dict(workspace)
            replacement.update(updates)
            workspaces[index] = replacement
            updated = True
            break
        if not updated:
            return False
        self._document.ui_state[self._MAGNETIC_FIELD_WORKSPACES_KEY] = workspaces
        return True

    def _on_magnetic_field_workspace_structure_changed(
        self, workspace_id: str, payload: object
    ) -> None:
        if self._document is None or not isinstance(payload, dict):
            return
        self._replace_magnetic_field_workspace(workspace_id, {"magnetic_structure": payload})

    def _on_magnetic_field_workspace_form_state_changed(
        self, workspace_id: str, state: dict
    ) -> None:
        if self._document is None or self._shutting_down:
            return
        workspace = next(
            (
                workspace
                for workspace in self._magnetic_field_workspaces_state()
                if str(workspace.get("id", "")).strip() == workspace_id
            ),
            None,
        )
        if workspace is None:
            return
        self._replace_magnetic_field_workspace(
            workspace_id,
            {
                "form": store_magnetic_field_results(
                    self._document,
                    workspace_id,
                    state,
                    journal_title=self._normalize_magnetic_field_workspace_title(
                        workspace.get("title", "")
                    ),
                )
            },
        )
        self._refresh_journal_view()

    def _on_magnetic_field_workspace_title_changed(self, workspace_id: str, title: str) -> None:
        if self._document is None or self._shutting_down:
            return
        normalized = self._normalize_magnetic_field_workspace_title(title)
        if not self._replace_magnetic_field_workspace(workspace_id, {"title": normalized}):
            return
        widget = self._magnetic_field_workspace_widgets.get(workspace_id)
        if widget is not None:
            widget.set_workspace_title(normalized, editable=True, deletable=True)
        self._refresh_navigator()
        self._set_active_route(self._magnetic_field_workspace_route(workspace_id))
        self._persist_document_ui_state(
            failure_message="Magnetic field workspace title was updated, but autosave failed."
        )

    def _on_magnetic_field_pin_requested(self, snapshot: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        default_title = self._normalize_magnetic_field_workspace_title(
            snapshot.get("title", "Magnetic Field Workspace")
        )
        title, accepted = QInputDialog.getText(
            self,
            "Save Magnetic Field Workspace",
            "Workspace name:",
            QLineEdit.EchoMode.Normal,
            default_title,
        )
        if not accepted:
            return
        normalized = self._normalize_magnetic_field_workspace_title(title)
        workspace_id = str(uuid4())
        magnetic_structure = snapshot.get("magnetic_structure")
        form_state = snapshot.get("form", {})
        workspace = {
            "id": workspace_id,
            "title": normalized,
            "magnetic_structure": (
                magnetic_structure if isinstance(magnetic_structure, dict) else None
            ),
            "form": store_magnetic_field_results(
                self._document, workspace_id, form_state if isinstance(form_state, dict) else {}
            ),
        }
        workspaces = list(self._magnetic_field_workspaces_state())
        workspaces.append(workspace)
        self._document.ui_state[self._MAGNETIC_FIELD_WORKSPACES_KEY] = workspaces
        self._sync_magnetic_field_workspace_views()
        self._refresh_navigator()
        self._set_active_route(self._magnetic_field_workspace_route(workspace_id))
        self._persist_document_ui_state(
            failure_message="Magnetic field workspace was saved, but autosave failed."
        )
        self.statusBar().showMessage(f"Saved magnetic field workspace '{normalized}'")

    def _delete_magnetic_field_workspace(self, workspace_id: str, *, confirm: bool = True) -> None:
        if self._document is None:
            return
        workspaces = list(self._magnetic_field_workspaces_state())
        target = next(
            (
                workspace
                for workspace in workspaces
                if str(workspace.get("id", "")).strip() == workspace_id
            ),
            None,
        )
        if target is None:
            return
        title = self._normalize_magnetic_field_workspace_title(
            target.get("title", "Magnetic Field Workspace")
        )
        if confirm:
            if not workflow_table_utils.confirm_destructive(
                self,
                "Delete magnetic field workspace",
                f"Delete magnetic field workspace '{title}'?",
                accept_label="Delete workspace",
            ):
                return
        self._document.ui_state[self._MAGNETIC_FIELD_WORKSPACES_KEY] = [
            workspace
            for workspace in workspaces
            if str(workspace.get("id", "")).strip() != workspace_id
        ]
        remove_magnetic_field_results(self._document, workspace_id)
        self._sync_magnetic_field_workspace_views()
        self._refresh_navigator()
        route = self._project_navigator.active_route()  # rebuilt pages are shown again
        gone = route is None or self._magnetic_field_workspace_id_from_route(route) == workspace_id
        self._set_active_route("magnetic_fields" if gone else route)
        self._persist_document_ui_state(
            failure_message="Magnetic field workspace was deleted, but autosave failed."
        )
        self.statusBar().showMessage(f"Deleted magnetic field workspace '{title}'")
