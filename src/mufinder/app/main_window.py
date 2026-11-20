"""Primary application window for MuFinder."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
import logging
from pathlib import Path
import sys
import time
from typing import Any, TypeVar
from uuid import UUID, uuid4

from PySide6.QtCore import (
    QByteArray,
    QEvent,
    QEventLoop,
    QSize,
    QSettings,
    QTimer,
    QUrl,
    Qt,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QDesktopServices,
    QGuiApplication,
    QKeySequence,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QFileDialog,
    QDockWidget,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder._version import current_version
from mufinder.branding import (
    DOCUMENTATION_URL,
    PRODUCT_NAME,
    PRODUCT_TAGLINE,
    REPOSITORY_URL,
    app_icon,
    logo_path,
)
from mufinder.cluster_setup import load_cluster_configs
from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import (
    MANUAL_DEFAULT_METHOD,
    CalculationRun,
    Job,
    MuonSite,
    Result,
    Structure,
)
from mufinder.app.cluster_setup_dialog import ClusterSetupDialog
from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.comparison_job_dialog import run_comparison_job_dialog
from mufinder.app.comparison_table_widget import ComparisonTablesPage
from mufinder.app.blocking_call import run_blocking_call
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
from mufinder.app.relaxation_import_dialog import (
    RelaxationImportDialog,
    choose_import_batch,
    start_relaxation_scan,
)
from mufinder.domain.text import (
    KeptFolder,
    folder_outcome_status,
    kept_folders_message,
    plain_error,
    plural,
)
from mufinder.services.relaxation_import import RelaxationFolderScan
from mufinder.services.relaxation_import_commit import IMPORT_ACTIONS, commit_relaxation_import
from mufinder.services.relaxation_import_removal import (
    detach_rows_from_definition,
    imported_relaxations_in_batch,
    plan_removal,
    remove_imported_relaxations,
)
from mufinder.app.relaxation_adapter import (
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    expand_muon_relaxation_jobs,
    expand_pristine_relaxation_jobs,
    muon_charge_state_label,
    muon_charge_state_prefix,
    muon_relaxation_calc_type,
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
    MuonRelaxationRow,
    PristineRelaxationConfig,
    SupercellMatrix,
)
from mufinder.app.relaxation_analysis_widget import (
    RelaxationAnalysisJobDialog,
    RelaxationAnalysisWidget,
    default_workspace_state,
)
from mufinder.app.relaxation_widget import PristineRelaxationWidget
from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget
from mufinder.app.muon_sites_summary_widget import MuonSitesSummaryWidget
from mufinder.app.uep_adapter import expand_uep_jobs
from mufinder.app.uep_types import UepConfig, UepSourceRow
from mufinder.app.uep_wizard import UepAnalysisDialog
from mufinder.app.uep_widget import UEPWidget
from mufinder.app.host_structure_widget import HostStructureWidget
from mufinder.app.icons import icon, workflow_stage_icon
from mufinder.app.job_dashboard import ClusterQueueStatus, JobDashboard, JobDashboardDeps
from mufinder.app.jobs_controller import JobsController, JobsControllerDeps
from mufinder.app.journal_widget import JournalWidget
from mufinder.app.magnetic_field_widget import MagneticFieldWidget
from mufinder.app.polarisation_widget import PolarisationWidget
from mufinder.app.muon_generation_widget import MuonGenerationWidget
from mufinder.app.new_project_dialog import NewProjectDialog
from mufinder.app.project_defaults_widget import ProjectDefaultsStrip, ProjectDefaultsSummary

# MainWindow uses these, and they stay importable from this module as public
# aliases for one minor release; see docs/compatibility_policy.rst.
from mufinder.app.muon_generation_support import MuonChargeStateDialog
from mufinder.app.muon_generation_support import MuonGenerationThread as _MuonGenerationThread
from mufinder.app.project_home_trail import (
    ConstructionSignWidget,  # noqa: F401 - re-export only
    ProjectHomeWidget,
    WorkflowDefinition,
)
from mufinder.app.hyperfine_coupling_widget import HyperfineCouplingWidget
from mufinder.app.recent_project_button import RecentProjectButton
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
    track_item_view_corners,
)
from mufinder.app.workflow_config_codecs import (
    deserialize_convergence_config,
    deserialize_muon_relaxation_config,
    deserialize_pristine_relaxation_config,
    deserialize_uep_config,
    normalized_parameter_values,
    normalized_supercell_matrix,
    serialize_muon_relaxation_config,
    serialize_pristine_relaxation_config,
    serialize_uep_config,
)
from mufinder.app.workflow_dialogs import (
    HyperfineCouplingConfigDialog,
    run_source_configure_dialog,
)
from mufinder.app.workflow_parameters import (
    structure_species_options,
)
from mufinder.app.workflow_submission import (
    WorkflowSubmissionCoordinator,
)
from mufinder.app.workflow_controllers.comparison_tables import (
    ComparisonTablesController,
    ComparisonTablesDeps,
    comparison_route,
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
    QuantumCorrectionsController,
    QuantumCorrectionsDeps,
    UepController,
    UepDeps,
    WorkflowControllerContext,
    WorkflowJobController,
    dedupe_jobs,
)
from mufinder.persistence.codec import to_jsonable, uuid_or_none
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
    ensure_job_mapping,
    relaxation_mapping_job_priority,
    sweep_definition_context_matches,
)
from mufinder.services.job_file_cleanup import (
    KEPT_AS_NOT_ITS_OWN,
    JobCleanupItem,
    JobCleanupOutcome,
    JobFileCleanup,
)
from mufinder.services.job_paths import (
    job_directory_label,
    definition_output_directory,
    job_output_directory,
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
    MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX,
    magnetic_field_workspace_route,
)
from mufinder.services.study_findings import record_finding, record_uep_minima
from mufinder.services.uep_grouping import uep_source_structure_id
from mufinder.services.muon_sites_summary import (
    analysis_workspace_created_record,
    build_muon_sites_summary_snapshot,
    muon_sites_summary_snapshots,
    relaxed_sites_grouped_record,
    store_muon_sites_summary_snapshot,
    validated_muon_sites_summary_snapshot,
)
from mufinder.services.perf_trace import PerfTrace, StageTiming
from mufinder.services.site_generation_service import (
    WORKER_GENERATION_METHODS,
    uep_generation_selection,
    uep_site_coordinate_warning,
)
from mufinder.services.project_service import (
    ProjectDocument,
    ProjectService,
    UnsavedChanges,
)
from mufinder.services.project_summary import (
    COMPLETION_CHECKS,
    PROJECT_HOME_STAGE_KEY,
    WORKFLOW_STAGES,
    job_stage_key,
    job_stage_name,
    magnetic_field_workspace_complete,
    relaxation_analysis_workspace_clustered,
    stage_group,
    stage_info,
)
from mufinder.services.source_candidates import (
    CompletedRelaxationAnalysisSource,
    MuonRelaxationRowJob,
    coerce_fractional_triplet,
    completed_muon_relaxation_sources,
    mixes_calculations,
    newest_completed_structure_id,
    quantum_seed_name_for_key,
    relaxation_sources_for_labels,
)
from mufinder.services.structure_export import relaxed_structure_export_for_job_id
from mufinder.science.hyperfine import HyperfineError
from mufinder.science.site_generation import SiteGenerationError
from mufinder.science.uep import (
    UepError,
    UepPotentialGrid,
    deserialize_potential_grid,
    serialize_potential_grid,
)
from mufinder.science.structure_io import StructureLoadError
from mufinder.workflow import StructureBridgeError


_LOG = logging.getLogger(__name__)

_RELAXATION_ANALYSIS_STATE_KEY = "relaxation_analysis_workspaces"
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


#: Why a batch's folder stays, for one folder and for several.
_KEPT_FOR_ANOTHER_IMPORT = ("another import uses it", "other imports use them")


class MainWindow(QMainWindow):
    """Main shell containing project navigation and job dashboard."""

    _MAIN_WINDOW_GEOMETRY_KEY = "ui/main_window_geometry"
    _MAIN_WINDOW_STATE_KEY = "ui/main_window_state"
    _LAST_DIALOG_DIR_KEY = "ui/last_dialog_dir"
    _LAST_DIALOG_DIR_BY_CONTEXT_PREFIX = "ui/last_dialog_dir_by_context"
    _RECENT_PROJECTS_KEY = "ui/recent_projects"
    _PROJECT_NAVIGATOR_WIDTH_KEY = "ui/project_navigator_width"
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
    _MAX_RECENT_PROJECTS = 10
    # The dock opens showing about this many rows.
    _JOB_DOCK_DEFAULT_ROWS = 5
    # Both the default and the minimum: it fits a workspace label such as "Positive
    # muon sites" at the deepest indent, with room for a scrollbar.
    _PROJECT_TREE_MIN_WIDTH = 220
    _PROJECT_TREE_MAX_SAVED_WIDTH = 4096
    _ACTIVE_CONVERGENCE_DEFINITION_KEY = ConvergenceCutoffController.active_key
    _ACTIVE_KPOINT_DEFINITION_KEY = ConvergenceKpointController.active_key
    _ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY = PristineRelaxationController.active_key
    _ACTIVE_MUON_RELAXATION_DEFINITION_KEY = MuonRelaxationController.active_key
    _ACTIVE_UEP_DEFINITION_KEY = UepController.active_key
    _UEP_ANALYSIS_STATES_KEY = "uep_analysis_states"
    _UEP_POTENTIAL_GRID_RESULT_TYPE = "uep_potential_grid"
    _MAGNETIC_FIELD_FORM_KEY = "magnetic_field_form"
    _POLARISATION_FORM_KEY = "polarisation_form"
    _MAGNETIC_FIELD_WORKSPACES_KEY = "magnetic_field_workspaces"
    _HOME_STAGE_KEYS = frozenset(stage.key for stage in WORKFLOW_STAGES)
    _MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX = MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX
    _RELAXATION_ANALYSIS_WORKSPACE_ROUTE_PREFIX = "muon_analysis/workspace/"
    _LAST_CLUSTER_SUBMISSION_CONFIG_KEY = "last_cluster_submission_config"
    _WORKFLOW_COMPLETE_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    _WORKFLOW_GOAL_ROLE = int(Qt.ItemDataRole.UserRole) + 2

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
            job_output_directory=self._job_output_directory,
            job_seed_name=self._job_seed_name,
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
        self._recent_projects_menu: QMenu | None = None
        self._home_recent_projects_layout: QVBoxLayout | None = None
        self._home_content_stack: QStackedWidget | None = None
        self._home_welcome_page: QWidget | None = None
        self._project_home_page: QWidget | None = None
        self._project_home: ProjectHomeWidget | None = None
        self._project_home_button: QToolButton | None = None
        self._magnetic_field_workspace_widgets: dict[str, MagneticFieldWidget] = {}
        self._uep_analysis_dialogs: dict[UUID, UepAnalysisDialog] = {}
        self._muon_generation_thread: _MuonGenerationThread | None = None
        # The document a running generation was prepared from; its sites are
        # applied only to that document.
        self._muon_generation_document: ProjectDocument | None = None
        self._muon_generation_failed = False

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

        self._project_tree = QTreeWidget()
        track_item_view_corners(self._project_tree)
        self._project_tree.setObjectName("workflowTree")
        self._project_tree.setHeaderLabel("Project")
        self._project_tree.setHeaderHidden(True)
        self._project_tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self._project_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._project_tree.customContextMenuRequested.connect(self._open_project_tree_context_menu)

        self._main_title = QLabel("Welcome to MuFinder")
        self._main_title.setObjectName("homeHeroTitle")
        self._main_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._main_body = QLabel(
            "Your compass for muon site discovery.\n"
            "Start by loading a structure, then follow guided convergence and muon-site steps."
        )
        self._main_body.setObjectName("homeHeroSubtitle")
        self._main_body.setWordWrap(True)

        self._main_stack = QStackedWidget()
        self._home_page = self._build_home_page()
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
        self._uep_view.configure_requested.connect(self._on_uep_configured)
        self._uep_view.form_state_changed.connect(self._on_uep_form_state_changed)
        self._uep_view.row_supercell_edit_requested.connect(self._on_uep_supercell_edit_requested)
        self._uep_view.source_structure_selection_requested.connect(
            self._on_uep_source_structure_selection_requested
        )
        self._uep_view.wizard_requested.connect(self._on_open_uep_analysis_requested)
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
        self._uep_view.submit_local_requested.connect(
            lambda values: self._uep_controller.submit(values, target="local")
        )
        self._uep_view.submit_cluster_requested.connect(
            lambda values: self._uep_controller.submit(values, target="cluster")
        )
        self._connect_workflow_row_actions(self._uep_view, self._uep_controller)
        self._uep_view.remove_requested.connect(self._on_remove_uep_rows)
        self._muon_generation_view = MuonGenerationWidget()
        self._muon_generation_view.generate_requested.connect(self._on_generate_requested)
        self._muon_generation_view.cancel_generation_requested.connect(
            self._on_cancel_generation_requested
        )
        self._muon_generation_view.cluster_positions_requested.connect(
            self._on_cluster_muon_positions_requested
        )
        self._muon_generation_view.delete_candidates_requested.connect(
            self._on_delete_muon_candidates_requested
        )
        self._muon_generation_view.create_jobs_requested.connect(
            self._on_create_muon_relaxation_jobs_requested
        )
        self._muon_generation_view.form_state_changed.connect(self._on_muon_form_state_changed)
        self._import_relaxations_action = None
        self._remove_imported_action = None
        self._relaxation_scan_host: Structure | None = None
        self._relaxation_scan_root: Path | None = None
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
        self._muon_relaxation_view.analysis_workspace_requested.connect(
            self._on_create_relaxation_analysis_workspace_requested
        )
        self._relaxation_analysis_view = RelaxationAnalysisWidget()
        # The page is rebuilt only while it is showing; see
        # _refresh_relaxation_analysis_context.
        self._relaxation_analysis_context_stale = False
        self._deferring_relaxation_analysis_refresh = False
        self._relaxation_analysis_view.set_relaxed_structure_export_provider(
            lambda job_id: relaxed_structure_export_for_job_id(self._document, job_id)
        )
        self._relaxation_analysis_view.state_changed.connect(
            self._on_relaxation_analysis_state_changed
        )
        self._relaxation_analysis_view.add_sites_requested.connect(
            self._on_relaxation_analysis_add_sites_requested
        )
        self._relaxation_analysis_view.pin_summary_requested.connect(
            self._on_muon_sites_pin_requested
        )
        self._relaxation_analysis_view.grouping_applied.connect(self._record_relaxed_sites_grouped)
        self._relaxation_analysis_view.delete_requested.connect(
            self._delete_relaxation_analysis_workspace
        )
        self._relaxation_analysis_view.current_workspace_changed.connect(
            self._on_relaxation_analysis_current_workspace_changed
        )
        self._muon_sites_summary_view = MuonSitesSummaryWidget()
        self._muon_sites_summary_view.open_analysis_requested.connect(
            self._on_open_muon_analysis_requested
        )
        self._muon_sites_summary_view.open_analysis_root_requested.connect(
            lambda: self._set_active_route("muon_analysis")
        )
        self._muon_sites_summary_view.focus_site_requested.connect(
            self._on_focus_muon_site_requested
        )
        self._muon_sites_summary_view.unpin_requested.connect(self._on_unpin_muon_sites_requested)
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
        self._hyperfine_coupling_view.configure_requested.connect(
            self._on_configure_hyperfine_coupling_source
        )
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
        self._journal_view.set_route_checker(
            lambda route: self._find_tree_item_by_route(route) is not None
        )
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
                can_check_now=lambda job: self._job_execution.can_check_now(job),
                staleness=lambda job, now: self._job_execution.staleness(job, now=now),
                can_continue_relaxation=lambda job: (
                    self._jobs.continue_relaxation_for_job(job) is not None
                ),
                show_status=lambda message: self.statusBar().showMessage(message),
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
        # obscured, or window minimized), and refresh + re-arm when it returns,
        # so we do not update an invisible table every second. Connected after
        # the first sync: hiding the dock until a project opens is no such pause.
        self._jobs_dock.visibilityChanged.connect(self._job_dashboard.set_visible_to_user)

    def _build_layout(self) -> None:
        central = QWidget()
        central_layout = QHBoxLayout(central)

        self._navigator_splitter = QSplitter(Qt.Orientation.Horizontal, central)
        self._navigator_splitter.setObjectName("projectNavigatorSplitter")
        self._navigator_splitter.setChildrenCollapsible(False)
        self._navigator_splitter.setHandleWidth(6)

        self._project_tree_panel = QFrame(self._navigator_splitter)
        self._project_tree_panel.setObjectName("projectTreePanel")
        self._project_tree_panel.setMinimumWidth(self._PROJECT_TREE_MIN_WIDTH)
        project_tree_layout = QVBoxLayout(self._project_tree_panel)
        project_tree_layout.setContentsMargins(0, 0, 0, 0)
        project_tree_layout.setSpacing(0)

        project_tree_header = QFrame(self._project_tree_panel)
        project_tree_header.setObjectName("projectTreePanelHeader")
        project_tree_header_layout = QHBoxLayout(project_tree_header)
        project_tree_header_layout.setContentsMargins(8, 6, 10, 6)
        project_tree_header_layout.setSpacing(6)

        self._project_home_button = QToolButton(project_tree_header)
        self._project_home_button.setObjectName("projectHomeButton")
        self._project_home_button.setIcon(icon("project_home"))
        self._project_home_button.setIconSize(QSize(22, 22))
        self._project_home_button.setToolTip("Show project home.")
        self._project_home_button.setEnabled(False)
        self._project_home_button.clicked.connect(self._show_project_home)
        project_tree_header_layout.addWidget(self._project_home_button)

        self._project_tree_title = QLabel("Project", project_tree_header)
        self._project_tree_title.setObjectName("projectTreePanelTitle")
        self._project_tree_title.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        project_tree_header_layout.addWidget(self._project_tree_title, 1)
        project_tree_layout.addWidget(project_tree_header)

        self._project_tree_body = QFrame(self._project_tree_panel)
        self._project_tree_body.setObjectName("projectTreePanelBody")
        project_tree_body_layout = QVBoxLayout(self._project_tree_body)
        project_tree_body_layout.setContentsMargins(1, 0, 1, 1)
        project_tree_body_layout.setSpacing(0)
        project_tree_body_layout.addWidget(self._project_tree, 1)
        project_tree_layout.addWidget(self._project_tree_body, 1)

        self._navigator_splitter.addWidget(self._project_tree_panel)
        self._navigator_splitter.addWidget(self._main_stack)
        self._navigator_splitter.handle(1).setAccessibleName("Resize project navigator")
        # Window resizes go to the pages; the navigator keeps its width.
        self._navigator_splitter.setStretchFactor(0, 0)
        self._navigator_splitter.setStretchFactor(1, 1)
        self._navigator_splitter.setSizes([self._saved_navigator_width(), self.width()])
        self._navigator_splitter.handle(1).installEventFilter(self)
        self._navigator_width_in_drag = self._project_tree_panel.width()
        self._dragged_navigator_width: int | None = None
        self._navigator_splitter.splitterMoved.connect(self._on_navigator_splitter_moved)
        central_layout.addWidget(self._navigator_splitter)
        self.setCentralWidget(central)

        jobs_dock = QDockWidget("Job Dashboard", self)
        jobs_dock.setObjectName("jobDashboardDock")
        self._jobs_dock = jobs_dock
        jobs_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        jobs_dock.setWidget(self._job_dashboard)
        self._job_dashboard.bind_delete_keys(jobs_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, jobs_dock)

    def _on_jobs_dock_toggled(self, shown: bool) -> None:
        # Only the user changes it with a project open: the title-bar button or the View
        # action. Hiding it for having no project is not the user's setting, and
        # minimising the window also unchecks the action.
        if self._document is not None and not self.isMinimized():
            self._jobs_dock_wanted = shown

    def _sync_project_panels(self) -> None:
        """Fit the window to whether a project is open.

        The navigator shows only with a project, and the Job Dashboard also needs
        ``View > Job Dashboard`` on. The project actions and the title follow.
        """

        has_project = self._document is not None
        self._project_tree_panel.setVisible(has_project)
        self._place_jobs_dock_once()
        self._jobs_dock.setVisible(has_project and self._jobs_dock_wanted)
        self._jobs_dock.toggleViewAction().setEnabled(has_project)
        self._save_project_action.setEnabled(has_project)
        self._import_relaxations_action.setEnabled(
            has_project and self._relaxation_scan_root is None
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
        """Put the dock where it was last time, closed or not, else at its default.

        This waits for the first time a project is on screen: a saved state would
        otherwise show a dock that stays hidden until a project opens.
        """

        if self._jobs_dock_placed or not self.isVisible() or self._document is None:
            return
        self._jobs_dock_placed = True
        saved_state = self._settings.value(self._MAIN_WINDOW_STATE_KEY, QByteArray())
        if (
            not isinstance(saved_state, QByteArray)
            or saved_state.isEmpty()
            or not self.restoreState(saved_state)
        ):
            # The dock has its real geometry once this show has been laid out.
            QTimer.singleShot(0, self._apply_default_jobs_dock_height)
        else:
            self._jobs_dock_wanted = not self._jobs_dock.isHidden()

    def _apply_default_jobs_dock_height(self) -> None:
        """Size the dock to show about ``_JOB_DOCK_DEFAULT_ROWS`` rows."""

        if self._jobs_dock.isHidden() or self._jobs_dock.isFloating():
            return
        table = self._job_dashboard.table
        chrome = self._jobs_dock.height() - table.viewport().height()
        rows_height = self._JOB_DOCK_DEFAULT_ROWS * table.verticalHeader().defaultSectionSize()
        self.resizeDocks([self._jobs_dock], [chrome + rows_height], Qt.Orientation.Vertical)

    def _saved_navigator_width(self) -> int:
        raw_value = self._settings.value(
            self._PROJECT_NAVIGATOR_WIDTH_KEY, self._PROJECT_TREE_MIN_WIDTH
        )
        try:
            width = int(raw_value)
        except (TypeError, ValueError):
            return self._PROJECT_TREE_MIN_WIDTH
        if not self._PROJECT_TREE_MIN_WIDTH <= width <= self._PROJECT_TREE_MAX_SAVED_WIDTH:
            return self._PROJECT_TREE_MIN_WIDTH
        return width

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API naming.
        if (
            event.type() == QEvent.Type.MouseButtonPress
            and watched is self._navigator_splitter.handle(1)
        ):
            self._navigator_width_in_drag = self._project_tree_panel.width()
        return super().eventFilter(watched, event)

    def _on_navigator_splitter_moved(self, _position: int, _index: int) -> None:
        # Only a drag lands here, so a window too narrow for the chosen width
        # squeezes the navigator for now without changing the width closeEvent saves.
        # The signal also fires for a drag the page minimum blocks; that one chose nothing.
        width = self._project_tree_panel.width()
        if width == self._navigator_width_in_drag:
            return
        self._navigator_width_in_drag = width
        self._dragged_navigator_width = width

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if getattr(self, "_shutting_down", False):
            super().closeEvent(event)
            return
        self._settings.setValue(self._MAIN_WINDOW_GEOMETRY_KEY, self.saveGeometry())
        if self._document is not None and self.isVisible():
            # Without a project the dock is hidden whatever the user chose, which
            # is not worth remembering.
            self._settings.setValue(self._MAIN_WINDOW_STATE_KEY, self.saveState())
        if self._dragged_navigator_width is not None:
            self._settings.setValue(
                self._PROJECT_NAVIGATOR_WIDTH_KEY, self._dragged_navigator_width
            )
        job_dashboard = getattr(self, "_job_dashboard", None)
        if job_dashboard is not None:
            job_dashboard.runtime_timer.stop()
        dashboard_flush_timer = getattr(self, "_dashboard_flush_timer", None)
        if dashboard_flush_timer is not None:
            dashboard_flush_timer.stop()
        job_check_timer = getattr(getattr(self, "_jobs", None), "check_timer", None)
        if job_check_timer is not None:
            job_check_timer.stop()
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
            self._close_open_uep_analysis_dialogs()
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
        muon_generation_thread = getattr(self, "_muon_generation_thread", None)
        if muon_generation_thread is not None:
            # The worker only computes points and never touches the document,
            # so closing cancels it and discards its sites. It notices at its
            # next progress report, which a run that rejects many trial
            # positions in a row can take a while to reach, so the event loop
            # runs behind a busy dialog meanwhile; a bare wait() would show
            # the window as not responding. The window modality prevents
            # re-entrant close clicks.
            muon_generation_thread.cancel()
            if not muon_generation_thread.isFinished():
                progress = QProgressDialog("Cancelling muon site generation…", None, 0, 0, self)
                progress.setWindowTitle("Closing")
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.show()
                loop = QEventLoop()
                muon_generation_thread.finished.connect(loop.quit)
                if not muon_generation_thread.isFinished():
                    loop.exec()
                progress.close()
            muon_generation_thread.wait()
            self._muon_generation_thread = None
            self._muon_generation_document = None
        self._shutdown_embedded_structure_viewers()
        local_runner = getattr(self, "_local_runner", None)
        if local_runner is not None:
            local_runner.shutdown()
        cluster_runner = getattr(self, "_cluster_runner", None)
        if cluster_runner is not None:
            cluster_runner.shutdown()
        super().closeEvent(event)

    def _answer_for_saving_on_close(self) -> str:
        """Return ``save``, ``tidy``, ``discard`` or ``cancel``, asking only if the user must choose."""

        assert self._document is not None
        self._flush_uep_analysis_dialogs()
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
        # Nothing to decide. "discard" here has nothing to discard, and writes nothing.
        return {UnsavedChanges.ROUTINE: "save", UnsavedChanges.TIDYING: "tidy"}.get(
            changes, "discard"
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

    def _flush_uep_analysis_dialogs(self) -> None:
        """Bring the document up to date with a regroup whose edit has not settled."""

        for dialog in tuple(self._uep_analysis_dialogs.values()):
            try:
                dialog.flush_pending_state()
            except RuntimeError:
                continue

    def _close_open_uep_analysis_dialogs(self) -> None:
        dialogs: list[QDialog] = []
        seen: set[int] = set()
        for dialog in tuple(self._uep_analysis_dialogs.values()):
            dialog_id = id(dialog)
            if dialog_id in seen:
                continue
            seen.add(dialog_id)
            dialogs.append(dialog)
        for dialog in self.findChildren(UepAnalysisDialog):
            dialog_id = id(dialog)
            if dialog_id in seen:
                continue
            seen.add(dialog_id)
            dialogs.append(dialog)

        for dialog in dialogs:
            try:
                dialog.close()
            except RuntimeError:
                continue
        self._uep_analysis_dialogs.clear()

    def _drop_uep_analysis_dialog(self, job_id: UUID) -> None:
        self._uep_analysis_dialogs.pop(job_id, None)

    def _make_home_card(self, title: str, body: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame(self)
        card.setObjectName("homeCard")
        layout = QVBoxLayout(card)

        title_label = QLabel(title, card)
        title_label.setObjectName("cardTitle")
        # A card beside a taller one has height to spare. The body takes it, so
        # every title keeps one height and the buttons line up at the bottom.
        title_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        body_label = QLabel(body, card)
        body_label.setObjectName("cardBody")
        body_label.setWordWrap(True)
        body_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        layout.addWidget(title_label)
        layout.addWidget(body_label)
        return card, layout

    @staticmethod
    def _fallback_device_pixel_ratio() -> float:
        app = QGuiApplication.instance()
        if app is None:
            return 1.0
        screen = app.primaryScreen()
        if screen is None:
            return 1.0
        return max(1.0, float(screen.devicePixelRatio()))

    def _home_logo_device_pixel_ratio(self) -> float:
        handle = self.windowHandle()
        if handle is not None and handle.screen() is not None:
            return max(1.0, float(handle.screen().devicePixelRatio()))
        return self._fallback_device_pixel_ratio()

    def _scaled_home_logo(self, logical_size: int = 92) -> QPixmap | None:
        logo_pixmap = QPixmap(logo_path())
        if logo_pixmap.isNull():
            return None

        dpr = self._home_logo_device_pixel_ratio()
        physical_size = max(1, int(round(logical_size * dpr)))
        logo_scaled = logo_pixmap.scaled(
            physical_size,
            physical_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        logo_scaled.setDevicePixelRatio(dpr)
        return logo_scaled

    def _build_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        hero_row = QHBoxLayout()
        hero_row.setSpacing(14)

        logo_label = QLabel(page)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        logo_pixmap = self._scaled_home_logo()
        if logo_pixmap is not None:
            logo_label.setPixmap(logo_pixmap)
        hero_row.addWidget(logo_label)

        hero_text = QVBoxLayout()
        hero_text.setSpacing(6)
        hero_text.addWidget(self._main_title)
        hero_text.addWidget(self._main_body)
        hero_row.addLayout(hero_text, 1)
        layout.addLayout(hero_row)

        self._home_content_stack = QStackedWidget(page)
        self._home_content_stack.setObjectName("homeContentStack")
        self._home_welcome_page = self._build_welcome_home_page()
        self._project_home_page = self._build_project_home_page()
        self._home_content_stack.addWidget(self._home_welcome_page)
        self._home_content_stack.addWidget(self._project_home_page)
        layout.addWidget(self._home_content_stack, 1)

        self._refresh_home_recent_projects()
        self._sync_project_home()
        self._sync_home_content_page()
        return page

    def _build_welcome_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        actions_grid = QGridLayout()
        actions_grid.setHorizontalSpacing(12)
        actions_grid.setVerticalSpacing(12)

        new_card, new_layout = self._make_home_card(
            "New Project",
            "Start a fresh MuFinder workspace with guided defaults.",
        )
        new_button = QPushButton("Create New Project", new_card)
        new_button.setIcon(icon("new_project"))
        new_button.clicked.connect(self._new_project)
        new_layout.addWidget(new_button)
        self._home_new_project_button = new_button

        open_card, open_layout = self._make_home_card(
            "Open Project",
            "Continue from an existing .mufinder folder.",
        )
        open_button = QPushButton("Open Existing Project", open_card)
        open_button.setIcon(icon("open_project"))
        open_button.clicked.connect(self._open_project)
        open_layout.addWidget(open_button)

        actions_grid.addWidget(new_card, 0, 0)
        actions_grid.addWidget(open_card, 0, 1)
        layout.addLayout(actions_grid)

        recent_card, recent_layout = self._make_home_card(
            "Recent Projects",
            "Jump back into your recent work.",
        )
        self._home_recent_projects_layout = QVBoxLayout()
        recent_layout.addLayout(self._home_recent_projects_layout)
        layout.addWidget(recent_card)
        layout.addStretch(1)
        return page

    def _build_project_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._project_home = ProjectHomeWidget(page)
        self._project_home.route_requested.connect(self._set_active_route)
        layout.addWidget(self._project_home, 1)
        return page

    def _refresh_home_recent_projects(self) -> None:
        if self._home_recent_projects_layout is None:
            return

        focused_row: int | None = None
        focused_path: str | None = None
        row = 0
        while self._home_recent_projects_layout.count():
            item = self._home_recent_projects_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                if widget.hasFocus() and isinstance(widget, RecentProjectButton):
                    focused_row = row
                    focused_path = widget.project_path
                widget.deleteLater()
            row += 1

        recent = self._recent_projects()[:5]
        if not recent:
            empty = QLabel("No recent projects yet.")
            empty.setObjectName("cardBody")
            self._home_recent_projects_layout.addWidget(empty)
            if focused_row is not None:
                self._home_new_project_button.setFocus(Qt.FocusReason.OtherFocusReason)
            return

        for project_path in recent:
            button = RecentProjectButton(project_path, self)
            button.clicked.connect(
                lambda _checked=False, project_path=project_path: self._open_recent_project(
                    project_path
                )
            )
            self._home_recent_projects_layout.addWidget(button)

        # Focus stays with its project; if that entry has gone, its neighbour takes over.
        if focused_row is not None:
            row = (
                recent.index(focused_path)
                if focused_path in recent
                else min(focused_row, len(recent) - 1)
            )
            successor = self._home_recent_projects_layout.itemAt(row).widget()
            # The layout shows it only on its next pass, and a hidden widget takes no focus.
            successor.show()
            successor.setFocus(Qt.FocusReason.OtherFocusReason)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        new_action = file_menu.addAction(icon("new_project"), "New Project…")
        new_action.setShortcuts(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_project)

        open_action = file_menu.addAction(icon("open_project"), "Open Project…")
        open_action.setShortcuts(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_project)

        self._recent_projects_menu = file_menu.addMenu("Open Recent")
        self._refresh_recent_projects_menu()

        self._save_project_action = file_menu.addAction(icon("save_project"), "Save Project")
        self._save_project_action.setShortcuts(QKeySequence.StandardKey.Save)
        self._save_project_action.triggered.connect(self._save_project)

        file_menu.addSeparator()

        self._import_relaxations_action = file_menu.addAction("Import CASTEP Relaxations…")
        self._import_relaxations_action.triggered.connect(self._on_import_castep_relaxations)

        self._remove_imported_action = file_menu.addAction("Remove Imported Relaxations…")
        self._remove_imported_action.triggered.connect(self._on_remove_imported_relaxations)

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
        dialog = NewProjectDialog(self._dialog_start_dir("new_project_parent"), parent=self)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        project_name = dialog.project_name()
        project_root = dialog.project_root()
        dialog.deleteLater()
        if not accepted or project_root is None:
            return
        self._remember_dialog_path(project_root.parent, "new_project_parent")

        # Persist any coalesced flush for the outgoing project before swapping.
        self._flush_dashboard()
        try:
            self._document = self._project_service.create_project(project_root, project_name)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Project error", str(exc))
            return

        self._record_recent_project(project_root)
        self._refresh_ui_after_project_change()
        self.statusBar().showMessage(f"Created project at {project_root}")
        self._focus_project_navigator()

    def _open_project(self) -> None:
        project_dir = self._get_existing_directory(
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
        if self._find_tree_item_by_route(route) is None:
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

        # Persist any coalesced autosave for the outgoing project before swapping.
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

        self._record_recent_project(project_root)
        self._remember_dialog_path(project_root, "open_project")
        # Shown first: the refresh below reports any local job it had to stop.
        self.statusBar().showMessage(f"Opened project {self._document.project.name}")
        self._refresh_ui_after_project_change()

        integrity_warnings = self._document.ui_state.get("_integrity_warnings", [])
        if integrity_warnings:
            QMessageBox.warning(
                self,
                "Integrity warnings",
                "Project loaded with integrity warnings:\n\n"
                + "\n".join(f"- {warning}" for warning in integrity_warnings),
            )
        self._focus_project_navigator()
        return True

    def _focus_project_navigator(self) -> None:
        """Give the navigator keyboard focus once a project has opened.

        With another window active, such as a dialog still open, this only sets
        what takes focus when the main window is active again.
        """

        self._project_tree.setFocus(Qt.FocusReason.OtherFocusReason)

    def _recent_projects(self) -> list[str]:
        raw_value = self._settings.value(self._RECENT_PROJECTS_KEY, [])
        if isinstance(raw_value, str):
            candidates = [raw_value]
        elif isinstance(raw_value, (list, tuple)):
            candidates = [str(entry) for entry in raw_value]
        else:
            candidates = []

        deduped: list[str] = []
        seen: set[str] = set()
        for entry in candidates:
            normalized = entry.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped[: self._MAX_RECENT_PROJECTS]

    def _set_recent_projects(self, project_paths: list[str]) -> None:
        cleaned = [path for path in project_paths if path.strip()][: self._MAX_RECENT_PROJECTS]
        self._settings.setValue(self._RECENT_PROJECTS_KEY, cleaned)

    def _record_recent_project(self, project_root: Path) -> None:
        normalized = str(project_root.expanduser())
        recent = [entry for entry in self._recent_projects() if entry != normalized]
        recent.insert(0, normalized)
        self._set_recent_projects(recent)
        self._refresh_recent_projects_menu()
        self._refresh_home_recent_projects()

    def _remove_recent_project(self, project_path: str) -> None:
        updated = [entry for entry in self._recent_projects() if entry != project_path]
        self._set_recent_projects(updated)
        self._refresh_recent_projects_menu()
        self._refresh_home_recent_projects()

    def _open_recent_project(self, project_path: str) -> None:
        project_root = Path(project_path).expanduser()
        opened = self._open_project_from_path(project_root, show_errors=False)
        if opened:
            return

        self._remove_recent_project(project_path)
        QMessageBox.warning(
            self,
            "Recent project unavailable",
            f"Could not open recent project at {project_root}. The entry has been removed from history.",
        )

    def _clear_recent_projects(self) -> None:
        self._set_recent_projects([])
        self._refresh_recent_projects_menu()
        self._refresh_home_recent_projects()

    def _refresh_recent_projects_menu(self) -> None:
        if self._recent_projects_menu is None:
            return

        menu = self._recent_projects_menu
        menu.clear()

        recent = self._recent_projects()
        if not recent:
            placeholder = menu.addAction("(No recent projects)")
            placeholder.setEnabled(False)
            return

        for project_path in recent:
            action = menu.addAction(project_path)
            action.triggered.connect(
                lambda _checked=False, project_path=project_path: self._open_recent_project(
                    project_path
                )
            )

        menu.addSeparator()
        clear_action = menu.addAction("Clear Recent Projects")
        clear_action.triggered.connect(self._clear_recent_projects)

    def _save_project(self) -> None:
        # The action is disabled with no project.
        if self._document is None:
            return

        self._flush_uep_analysis_dialogs()
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

    def _dialog_key(self, context: str | None = None) -> str:
        if context:
            return f"{self._LAST_DIALOG_DIR_BY_CONTEXT_PREFIX}/{context}"
        return self._LAST_DIALOG_DIR_KEY

    def _dialog_start_dir(self, context: str | None = None) -> str:
        raw_value = self._settings.value(self._dialog_key(context), "", str)
        if not raw_value and context:
            raw_value = self._settings.value(self._dialog_key(), "", str)
        if not raw_value:
            return str(Path.home())

        candidate = Path(raw_value).expanduser()
        if candidate.exists():
            return str(candidate if candidate.is_dir() else candidate.parent)

        parent = candidate.parent
        if parent.exists() and parent.is_dir():
            return str(parent)
        return str(Path.home())

    def _remember_dialog_path(self, selected_path: str | Path, context: str | None = None) -> None:
        directory = Path(selected_path).expanduser()
        if directory.exists() and directory.is_file():
            directory = directory.parent
        elif not directory.exists():
            directory = directory.parent

        if directory.exists() and directory.is_dir():
            stored = str(directory)
            self._settings.setValue(self._dialog_key(), stored)
            if context:
                self._settings.setValue(self._dialog_key(context), stored)

    def _get_existing_directory(self, title: str, context: str) -> str:
        selected = QFileDialog.getExistingDirectory(
            self,
            title,
            self._dialog_start_dir(context),
        )
        if selected:
            self._remember_dialog_path(selected, context)
        return selected

    def _get_open_file_name(self, title: str, file_filter: str, context: str) -> tuple[str, str]:
        file_path, selected_filter = QFileDialog.getOpenFileName(
            self,
            title,
            self._dialog_start_dir(context),
            file_filter,
        )
        if file_path:
            self._remember_dialog_path(file_path, context)
        return file_path, selected_filter

    def _get_save_file_name(self, title: str, file_filter: str, context: str) -> tuple[str, str]:
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            title,
            self._dialog_start_dir(context),
            file_filter,
        )
        if file_path:
            self._remember_dialog_path(file_path, context)
        return file_path, selected_filter

    def _refresh_ui_after_project_change(self) -> None:
        if self._document is None:
            return

        # Each project opens with its dismissed jobs hidden; the refresh below redraws.
        self._job_dashboard.hide_dismissed()

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
            with trace.stage("_populate_project_tree"):
                self._populate_project_tree(preserve_selection=False)
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
                self._refresh_uep_rows()
                self._refresh_uep_source_structures()
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
        self._main_title.setText(f"Project: {self._document.project.name}")
        self._main_body.setText(
            "Pick a stage on the chart to see where it stands; double-click it to open it."
        )
        self._project_tree_title.setText(f"Project: {self._document.project.name}")

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
        if not isinstance(route, str) or not route.startswith(
            self._MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX
        ):
            return None
        workspace_id = route[len(self._MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX) :].strip()
        return workspace_id or None

    def _relaxation_analysis_workspace_route(self, workspace_id: str) -> str:
        return f"{self._RELAXATION_ANALYSIS_WORKSPACE_ROUTE_PREFIX}{workspace_id}"

    def _relaxation_analysis_workspace_id_from_route(self, route: object) -> str | None:
        if not isinstance(route, str) or not route.startswith(
            self._RELAXATION_ANALYSIS_WORKSPACE_ROUTE_PREFIX
        ):
            return None
        workspace_id = route[len(self._RELAXATION_ANALYSIS_WORKSPACE_ROUTE_PREFIX) :].strip()
        return workspace_id or None

    def _workflow_definitions(self) -> tuple[WorkflowDefinition, ...]:
        def stage(key: str, children: tuple[WorkflowDefinition, ...] = ()) -> WorkflowDefinition:
            info = stage_info(key)
            return WorkflowDefinition(
                key=key,
                title=info.title,
                route=key,
                completion_goal=info.completion_goal,
                completion_check=COMPLETION_CHECKS[key],
                children=children,
            )

        def group(key: str, children: tuple[WorkflowDefinition, ...]) -> WorkflowDefinition:
            info = stage_group(key)
            return WorkflowDefinition(
                key=key,
                title=info.title,
                route=key,
                completion_goal=info.completion_goal,
                completion_check=COMPLETION_CHECKS[key],
                children=children,
            )

        magnetic_workspaces = tuple(
            WorkflowDefinition(
                key=f"magnetic_field_workspace_{str(workspace.get('id', '')).strip()}",
                title=str(workspace.get("title", "Magnetic Fields")).strip() or "Magnetic Fields",
                route=self._magnetic_field_workspace_route(str(workspace.get("id", "")).strip()),
                completion_goal="Define at least one magnetic component in this saved workspace.",
                completion_check=(
                    lambda document, workspace_id=str(workspace.get("id", "")).strip(): (
                        magnetic_field_workspace_complete(document, workspace_id)
                    )
                ),
            )
            for workspace in self._magnetic_field_workspaces_state()
            if str(workspace.get("id", "")).strip()
        )
        analysis_workspaces = tuple(
            WorkflowDefinition(
                key=f"muon_analysis_workspace_{workspace['id']}",
                title=str(workspace.get("name", "")).strip() or "Relaxation analysis",
                route=self._relaxation_analysis_workspace_route(str(workspace["id"])),
                completion_goal="Apply clustering to this analysis.",
                completion_check=(
                    lambda document, workspace_id=str(workspace["id"]): (
                        relaxation_analysis_workspace_clustered(document, workspace_id)
                    )
                ),
            )
            for workspace in self._relaxation_analysis_workspaces()
            if str(workspace.get("id", "")).strip()
        )

        return (
            stage("host_structure"),
            group("convergence", (stage("convergence_cutoff"), stage("convergence_kpoint"))),
            stage("pristine_relaxation"),
            stage("uep"),
            group(
                "muon_sites",
                (
                    stage("muon_candidates"),
                    stage("muon_relaxations"),
                    stage("muon_analysis", analysis_workspaces),
                ),
            ),
            stage("hyperfine_coupling"),
            stage("magnetic_fields", magnetic_workspaces),
            stage("polarisation"),
            stage("quantum_corrections"),
            stage("energy_barriers"),
            WorkflowDefinition(key="provenance_log", title="Journal", route="provenance_log"),
        )

    def _workflow_label(self, definition: WorkflowDefinition) -> str:
        return definition.title

    def _workflow_completion_checks_by_route(
        self,
        definitions: tuple[WorkflowDefinition, ...] | None = None,
    ) -> dict[str, Callable[[ProjectDocument], bool]]:
        checks: dict[str, Callable[[ProjectDocument], bool]] = {}
        for definition in self._workflow_definitions() if definitions is None else definitions:
            if definition.route and definition.completion_check is not None:
                checks[definition.route] = definition.completion_check
            checks.update(self._workflow_completion_checks_by_route(definition.children))
        return checks

    def _build_workflow_tree_item(self, definition: WorkflowDefinition) -> QTreeWidgetItem:
        item = QTreeWidgetItem([self._workflow_label(definition)])
        if definition.route:
            item.setData(0, Qt.ItemDataRole.UserRole, definition.route)

        if definition.completion_goal:
            item.setData(0, self._WORKFLOW_GOAL_ROLE, definition.completion_goal)
            item.setToolTip(0, f"Complete when: {definition.completion_goal}")
            is_complete = bool(
                definition.completion_check(self._document)
                if self._document is not None and definition.completion_check is not None
                else False
            )
            item.setData(0, self._WORKFLOW_COMPLETE_ROLE, is_complete)
            item.setIcon(0, workflow_stage_icon("completed" if is_complete else "incomplete"))

        for child in definition.children:
            item.addChild(self._build_workflow_tree_item(child))
        return item

    def _find_workflow_item_by_route(
        self,
        parent: QTreeWidgetItem,
        route: str,
    ) -> QTreeWidgetItem | None:
        if parent.data(0, Qt.ItemDataRole.UserRole) == route:
            return parent
        for index in range(parent.childCount()):
            child = parent.child(index)
            if child is None:
                continue
            found = self._find_workflow_item_by_route(child, route)
            if found is not None:
                return found
        return None

    def _find_tree_item_by_route(self, route: str) -> QTreeWidgetItem | None:
        for index in range(self._project_tree.topLevelItemCount()):
            item = self._project_tree.topLevelItem(index)
            if item is None:
                continue
            found = self._find_workflow_item_by_route(item, route)
            if found is not None:
                return found
        return None

    def _populate_project_tree(self, *, preserve_selection: bool = True) -> None:
        if self._document is None:
            return

        previous_route: str | None = None
        current_item = self._project_tree.currentItem() if preserve_selection else None
        if current_item is not None:
            raw_route = current_item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(raw_route, str) and raw_route.strip():
                previous_route = raw_route.strip()

        self._project_tree.clear()

        for definition in self._workflow_definitions():
            self._project_tree.addTopLevelItem(self._build_workflow_tree_item(definition))

        pristine_item = self._find_tree_item_by_route("pristine_relaxation")
        if pristine_item is not None:
            ordered_tables = sorted(
                self._document.comparison_tables.values(),
                key=lambda table: (table.created_at, table.label.casefold()),
            )
            for table in ordered_tables:
                child = QTreeWidgetItem([table.label or "Comparison Table"])
                child.setData(0, Qt.ItemDataRole.UserRole, comparison_route(table.id))
                pristine_item.addChild(child)
            pristine_item.setExpanded(True)

        analysis_item = self._find_tree_item_by_route("muon_analysis")
        if analysis_item is not None and analysis_item.childCount():
            analysis_item.setExpanded(True)

        for index in range(self._project_tree.topLevelItemCount()):
            item = self._project_tree.topLevelItem(index)
            if item is None:
                continue
            label = item.text(0)
            if label.startswith(("Convergence Testing", "Magnetic Fields", "Muon Sites")):
                item.setExpanded(True)
        selected_item = None
        if previous_route is not None:
            match = self._find_tree_item_by_route(previous_route)
            if match is not None:
                selected_item = match

        if selected_item is None:
            self._project_tree.clearSelection()
            # No row is the user's on the project home. Without this, focus arriving
            # picks the first row as current, and the first Down skips past it.
            self._project_tree.setCurrentItem(None)
            self._main_stack.setCurrentWidget(self._home_page)
            self._sync_home_content_page()
        else:
            self._project_tree.setCurrentItem(selected_item)
        self._refresh_workflow_stage_icons()
        self._sync_project_home()

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
        self._refresh_uep_rows()
        self._refresh_workflow_stage_icons()
        self._sync_project_home()

    def _persist_document_ui_state(self, *, failure_message: str) -> None:
        if self._document is None:
            return
        self._flush_uep_analysis_dialogs()
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

    def _on_tree_selection_changed(self) -> None:
        selected_items = self._project_tree.selectedItems()
        if not selected_items:
            return

        route = selected_items[0].data(0, Qt.ItemDataRole.UserRole)
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
        self._remember_project_home_stage(route)
        # Navigation does not mutate the document, so workflow completion state
        # cannot change here. Stage icons are kept current by tree construction
        # and by the mutation-driven refresh calls, so recomputing every check
        # on each tab switch (which scans jobs/results) is wasted work.
        self._sync_project_home()

    def _remember_project_home_stage(self, route: object) -> None:
        """Record the workflow stage a route belongs to, so the project home opens on it."""

        if self._document is None or not isinstance(route, str):
            return
        if route in self._HOME_STAGE_KEYS:
            stage_key: str | None = route
        elif (
            route == "muon_sites"
            or self._relaxation_analysis_workspace_id_from_route(route) is not None
        ):
            stage_key = "muon_analysis"
        elif self._magnetic_field_workspace_id_from_route(route) is not None:
            stage_key = "magnetic_fields"
        elif comparison_table_id_from_route(route) is not None:
            stage_key = "pristine_relaxation"
        else:
            stage_key = None
        if (
            stage_key is not None
            and self._document.ui_state.get(PROJECT_HOME_STAGE_KEY) != stage_key
        ):
            self._document.ui_state[PROJECT_HOME_STAGE_KEY] = stage_key

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
        item = self._find_tree_item_by_route(route)
        if item is None:
            return
        self._project_tree.setCurrentItem(item)
        self._on_tree_selection_changed()

    def _set_tree_item_label(self, route: str, label: str) -> None:
        """Relabel one route's project-tree item, leaving the selection and the page alone."""

        item = self._find_tree_item_by_route(route)
        if item is not None:
            item.setText(0, label)

    def _active_route(self) -> str | None:
        selected_items = self._project_tree.selectedItems()
        if not selected_items:
            return None
        route = selected_items[0].data(0, Qt.ItemDataRole.UserRole)
        if isinstance(route, str) and route.strip():
            return route.strip()
        return None

    def _show_project_home(self) -> None:
        self._project_tree.clearSelection()
        self._main_stack.setCurrentWidget(self._home_page)
        self._sync_home_content_page()
        self._refresh_workflow_stage_icons()
        self._sync_project_home()

    def _sync_home_content_page(self) -> None:
        if self._home_content_stack is None:
            return
        target = self._project_home_page if self._document is not None else self._home_welcome_page
        if target is not None:
            self._home_content_stack.setCurrentWidget(target)
        if self._project_home_button is not None:
            self._project_home_button.setEnabled(self._document is not None)

    def _sync_project_home(self) -> None:
        if self._project_home is None:
            return
        self._project_home.set_document(self._document)

    def _refresh_workflow_stage_icons(self) -> None:
        completion_checks = self._workflow_completion_checks_by_route()

        def visit(item: QTreeWidgetItem) -> None:
            has_goal = item.data(0, self._WORKFLOW_GOAL_ROLE) is not None
            if has_goal:
                route = item.data(0, Qt.ItemDataRole.UserRole)
                if (
                    self._document is not None
                    and isinstance(route, str)
                    and (completion_check := completion_checks.get(route)) is not None
                ):
                    is_complete = bool(completion_check(self._document))
                    item.setData(0, self._WORKFLOW_COMPLETE_ROLE, is_complete)
                else:
                    is_complete = bool(item.data(0, self._WORKFLOW_COMPLETE_ROLE))
                state = "completed" if is_complete else "incomplete"
                item.setIcon(0, workflow_stage_icon(state))

            for idx in range(item.childCount()):
                child = item.child(idx)
                if child is not None:
                    visit(child)

        for index in range(self._project_tree.topLevelItemCount()):
            item = self._project_tree.topLevelItem(index)
            if item is not None:
                visit(item)

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

    @staticmethod
    def _job_submission_sort_key(job: Job) -> float:
        for stamp in (job.submitted_at, job.started_at, job.completed_at):
            if stamp is not None:
                return stamp.timestamp()
        return 0.0

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
        """Repaint every workflow table that can show a just-reset job.

        Shared by the explicit per-row/dashboard Reset action and by the lazy
        reset a workflow row performs when it resubmits an ended job. ``_refresh_jobs_table``
        already re-renders the dashboard plus the convergence, pristine, muon,
        and UEP tables; the convergence controllers are refreshed explicitly
        because a reset also detaches a cutoff sweep point, and the two
        lazily-created views (energy barriers, quantum corrections) refresh
        through their own no-op-until-created wrappers. Every workflow now has an
        in-table reset action, so every workflow's table has to repaint.
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
            muon_relaxation_row_key=lambda row: self._muon_relaxation_row_key(row),
            refresh_jobs_table=lambda: self._refresh_jobs_table(),
            refresh_journal_view=lambda: self._refresh_journal_view(),
            refresh_hyperfine_coupling_view=lambda: self._refresh_hyperfine_coupling_view(),
            refresh_quantum_corrections_view=lambda: self._refresh_quantum_corrections_view(),
            refresh_energy_barriers_view=lambda: self._refresh_energy_barriers_view(),
            refresh_views_after_job_reset=lambda: self._refresh_views_after_job_reset(),
            request_dashboard_flush=lambda **kwargs: self._request_dashboard_flush(**kwargs),
            persist_document_ui_state=lambda **kwargs: self._persist_document_ui_state(**kwargs),
            with_kpoint_status_note=lambda message: self._with_kpoint_status_note(message),
            show_kept_folders=lambda *args, **kwargs: self._show_kept_folders(*args, **kwargs),
            detach_convergence_job=lambda job_id: self._detach_convergence_job(job_id),
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
        self._populate_project_tree()

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
            refresh_project_tree=lambda: self._populate_project_tree(),
            refresh_journal_view=lambda: self._refresh_journal_view(),
            set_tree_label=lambda route, label: self._set_tree_item_label(route, label),
            active_route=lambda: self._active_route(),
            open_route=lambda route: self._set_active_route(route),
            shutting_down=lambda: self._shutting_down,
        )

    def _build_muon_relaxation_deps(self) -> MuonRelaxationDeps:
        """Build the muon-relaxation controller's deps.

        These helpers stay on ``MainWindow`` because non-migrating code still
        calls them: the definition/config/mapping helpers are shared with row
        removal, staging, and dashboard sync; ``_muon_relaxation_row_key`` has
        other view-refresh callers. The
        controller reaches them only through these late-bound wrappers, and
        ``coerce_fractional_triplet`` comes from
        :mod:`mufinder.services.source_candidates`.
        """

        return MuonRelaxationDeps(
            deserialize_muon_relaxation_config=deserialize_muon_relaxation_config,
            serialize_muon_relaxation_config=serialize_muon_relaxation_config,
            sync_active_muon_relaxation_dashboard_jobs=(
                lambda definition: self._sync_active_muon_relaxation_dashboard_jobs(definition)
            ),
            muon_relaxation_form_config=lambda: self._muon_relaxation_form_config(),
            muon_relaxation_row_signature=(lambda row: self._muon_relaxation_row_signature(row)),
            muon_relaxation_row_key=lambda row: self._muon_relaxation_row_key(row),
            handle_site_host_mismatch=(lambda mismatch: self._handle_site_host_mismatch(mismatch)),
            muon_relaxation_definition_and_row_for_job=(
                lambda job: self._muon_relaxation_definition_and_row_for_job(job)
            ),
            coerce_fractional_triplet=coerce_fractional_triplet,
        )

    def _build_uep_deps(self) -> UepDeps:
        """Build the UEP controller's deps.

        These helpers stay on ``MainWindow`` because non-migrating code still
        calls them: the definition/config/mapping helpers are shared with the
        UEP configure path, row removal, the analysis wizard, and dashboard
        cluster-source sync; ``_uep_potential_grid_result_for_job`` still feeds
        the wizard's "already analysed" detection. The cross-workflow job/
        definition readers (``definition_output_directory``,
        ``job_output_directory``, ``job_seed_name``, ``run_for_job``) now live on
        the shared context. The controller reaches these deps only through
        late-bound wrappers.
        """

        return UepDeps(
            deserialize_uep_config=deserialize_uep_config,
            serialize_uep_config=serialize_uep_config,
            uep_row_signature=lambda row: self._uep_row_signature(row),
            uep_potential_grid_result_for_job=(
                lambda job_id: self._uep_potential_grid_result_for_job(job_id)
            ),
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

    def _results_for_run(self, run_id: UUID) -> list[Result]:
        if self._document is None:
            return []
        return [
            result
            for result in self._document.results.values()
            if result.calculation_run_id == run_id
        ]

    def _refresh_hyperfine_coupling_view(self) -> None:
        """Re-render the hyperfine-coupling view via its controller."""

        self._hyperfine_coupling_controller.refresh_view()

    def _refresh_quantum_corrections_view(self) -> None:
        """Re-render the quantum-corrections view via its controller.

        Thin wrapper (delegates to
        :meth:`QuantumCorrectionsController.refresh_view`) kept on ``MainWindow``
        because it has many callers across document/route changes, submission,
        configuration, rename, and reset.
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

    def _uep_potential_grid_result_for_job(self, job_id: UUID) -> Result | None:
        run = self._run_for_job(job_id)
        if run is None:
            return None
        for result in self._results_for_run(run.id):
            if result.result_type == self._UEP_POTENTIAL_GRID_RESULT_TYPE:
                return result
        return None

    def _load_uep_potential_grid_for_job(self, job_id: UUID) -> UepPotentialGrid | None:
        result = self._uep_potential_grid_result_for_job(job_id)
        if result is None:
            return None
        return deserialize_potential_grid(result.data)

    def _job_output_directory(self, job: Job) -> Path | None:
        return job_output_directory(
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
        if self._muon_generation_blocks_host_change():
            return

        file_path, _selected_filter = self._get_open_file_name(
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
        self._populate_project_tree()
        self._refresh_journal_view()
        self._set_active_route("host_structure")
        self.statusBar().showMessage(
            f"Loaded host structure '{structure.label}' ({len(structure.species)} atoms)"
        )

    def _on_generate_requested(self, method: str, parameters: dict) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        if self._muon_generation_thread is not None:
            QMessageBox.information(
                self, "Generation in progress", "Wait for the current generation task to finish."
            )
            return

        if method in WORKER_GENERATION_METHODS:
            self._start_muon_generation_thread(method, parameters)
            return

        try:
            created = self._project_service.generate_candidate_sites(
                self._document,
                method=method,
                parameters=parameters,
            )
        except (SiteGenerationError, ValueError) as exc:
            QMessageBox.warning(self, "Generation blocked", str(exc))
            return

        self._persist_document_ui_state(
            failure_message="Muon sites were generated, but autosave failed."
        )
        self._refresh_muon_generation_table()
        self._refresh_journal_view()
        self._finish_generation_status(method, parameters, created)

    def _start_muon_generation_thread(self, method: str, parameters: dict) -> None:
        if self._document is None:
            return
        try:
            request = self._project_service.prepare_candidate_sites(
                self._document, method=method, parameters=parameters
            )
        except (SiteGenerationError, ValueError) as exc:
            QMessageBox.warning(self, "Generation blocked", str(exc))
            return
        thread = _MuonGenerationThread(self._project_service, request, parent=self)
        self._muon_generation_thread = thread
        self._muon_generation_document = self._document
        self._muon_generation_failed = False
        thread.generated.connect(
            lambda outcome, worker=thread: self._on_threaded_generation_finished(worker, outcome)
        )
        thread.failed.connect(
            lambda message, current_method=method: self._on_threaded_generation_failed(
                current_method, message
            )
        )
        thread.cancelled.connect(self._on_threaded_generation_cancelled)
        random_mode = (
            str(parameters.get("selection_mode", parameters.get("mode", ""))).strip().lower()
        )
        thread.progress_changed.connect(
            lambda generated, expected, current_method=method, mode=random_mode: (
                self._on_generation_progress_changed(
                    current_method,
                    generated,
                    expected,
                    mode,
                )
            )
        )
        thread.finished.connect(self._on_muon_generation_thread_finished)
        thread.finished.connect(thread.deleteLater)
        method_label = "random" if method == "random" else "near-element"
        self._muon_generation_view.set_generation_busy(
            True, f"Preparing {method_label} generation…"
        )
        self.statusBar().showMessage(f"Generating {method_label} muon sites…")
        thread.start()

    def _on_cancel_generation_requested(self) -> None:
        thread = self._muon_generation_thread
        if thread is None or thread.cancel_requested:
            return
        if not workflow_table_utils.confirm_destructive(
            self,
            "Cancel generation",
            "Cancel the running muon site generation?",
            "The sites it has generated so far are discarded.",
            accept_label="Cancel generation",
            reject_label="Keep generating",
        ):
            return
        # The run may have finished while the question was open.
        if self._muon_generation_thread is thread:
            thread.cancel()
            self._muon_generation_view.set_generation_progress(0, 1, "Cancelling…")

    def _muon_generation_blocks_host_change(self) -> bool:
        """Refuse a host change while a generation computes sites for the current host."""

        if self._muon_generation_thread is None:
            return False
        QMessageBox.information(
            self,
            "Generation in progress",
            "Muon sites are being generated for the current host structure. Wait for the "
            "generation to finish, or cancel it, before changing the host.",
        )
        return True

    def _on_generation_progress_changed(
        self, method: str, generated: int, expected: int, mode: str
    ) -> None:
        if method == "random" and mode in {"separation", "distance", "spacing"}:
            message = "Generated %v sites"
        else:
            message = "Generated %v/%m sites"
        self._muon_generation_view.set_generation_progress(
            generated,
            expected,
            message,
        )

    def _on_threaded_generation_finished(
        self, thread: _MuonGenerationThread, outcome: object
    ) -> None:
        """Apply a finished worker's sites to the document, on the GUI thread."""

        if self._shutting_down:
            return
        document = self._muon_generation_document
        if document is None or document is not self._document:
            self._on_threaded_generation_discarded()
            return
        if thread.cancel_requested:
            self._on_threaded_generation_cancelled()
            return
        method = thread.method
        parameters = thread.parameters
        try:
            sites = self._project_service.apply_generated_sites(document, thread.request, outcome)
        except SiteGenerationError as exc:
            self._on_threaded_generation_failed(method, str(exc))
            return
        self._persist_document_ui_state(
            failure_message="Muon sites were generated, but autosave failed."
        )
        used_spacing: object | None = None
        if method == "random" and sites:
            used_spacing = sites[0].properties.get(
                "min_muon_muon",
                sites[0].properties.get("estimated_min_muon_muon"),
            )
        self._muon_generation_view.set_generation_progress(
            len(sites),
            max(len(sites), 1),
            f"Generated {plural(len(sites), 'site')}",
        )
        self._refresh_muon_generation_table()
        if used_spacing is not None:
            self._muon_generation_view.set_random_muon_muon_distance(float(used_spacing))
        self._refresh_journal_view()
        self._finish_generation_status(method, parameters, sites)

    def _on_threaded_generation_cancelled(self) -> None:
        if self._shutting_down:
            return
        if self._muon_generation_document is not self._document:
            self._on_threaded_generation_discarded()
            return
        self._muon_generation_view.set_generation_busy(False, "Generation cancelled")
        self.statusBar().showMessage("Muon site generation cancelled; no sites were added")

    def _on_threaded_generation_discarded(self) -> None:
        """Drop a run started in a project that is no longer open."""

        self._muon_generation_view.set_generation_busy(False, "Ready")
        self.statusBar().showMessage(
            "Discarded the muon site generation started in the previously open project"
        )

    def _on_threaded_generation_failed(self, method: str, message: str) -> None:
        self._muon_generation_failed = True
        if self._shutting_down:
            return
        if self._muon_generation_document is not self._document:
            self._on_threaded_generation_discarded()
            return
        self._muon_generation_view.set_generation_busy(False, "Generation failed")
        QMessageBox.warning(self, "Generation blocked", message)
        method_label = "Random" if method == "random" else "Near-element"
        self.statusBar().showMessage(f"{method_label} site generation failed")

    def _on_muon_generation_thread_finished(self) -> None:
        self._muon_generation_thread = None
        self._muon_generation_document = None
        if not self._muon_generation_failed:
            self._muon_generation_view.set_generation_controls_enabled(True)

    def _finish_generation_status(
        self, method: str, parameters: dict, created: list[MuonSite]
    ) -> None:
        requested = (
            self._project_service.requested_site_count(
                self._document, method=method, parameters=parameters
            )
            if self._document is not None
            else None
        )
        suffix = ""
        if requested is not None and len(created) < requested:
            suffix = f" (requested {requested}; constraints prevented full count)"
        if (
            method == "random"
            and created
            and created[0].properties.get("estimated_min_muon_muon") is not None
        ):
            estimated = float(created[0].properties["estimated_min_muon_muon"])
            used = float(created[0].properties.get("min_muon_muon", estimated))
            if abs(used - estimated) > 1e-9:
                suffix += f" (estimated spacing {estimated:.2f} A; used {used:.2f} A)"
            else:
                suffix += f" (estimated spacing {estimated:.2f} A)"
        self.statusBar().showMessage(f"Generated {plural(len(created), 'muon site')}{suffix}")

    def _on_delete_muon_candidates_requested(self, raw_site_ids: object) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return

        site_ids: list[UUID] = []
        raw_values = raw_site_ids if isinstance(raw_site_ids, list) else [raw_site_ids]
        for raw_value in raw_values:
            try:
                site_ids.append(UUID(str(raw_value)))
            except (TypeError, ValueError):
                continue
        if not site_ids:
            return

        removed = self._project_service.remove_candidate_sites(self._document, site_ids)
        if removed:
            self._persist_document_ui_state(
                failure_message="Muon sites were deleted, but autosave failed."
            )
        self._refresh_muon_generation_table()
        self._refresh_journal_view()
        self.statusBar().showMessage(f"Deleted {plural(len(removed), 'generated muon site')}")

    def _on_cluster_muon_positions_requested(self, payload: object) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        if not isinstance(payload, dict):
            return

        raw_site_ids = payload.get("site_ids", [])
        raw_values = raw_site_ids if isinstance(raw_site_ids, list) else [raw_site_ids]
        site_ids: list[UUID] = []
        for raw_value in raw_values:
            try:
                site_ids.append(UUID(str(raw_value)))
            except (TypeError, ValueError):
                continue
        anchor = payload.get("anchor_fractional", [])

        try:
            clustered = self._project_service.cluster_candidate_positions(
                self._document,
                site_ids,
                anchor_fractional=anchor,
            )
        except (SiteGenerationError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Clustering blocked", str(exc))
            return

        clustered_ids = [str(site.id) for site in clustered]
        self._persist_document_ui_state(
            failure_message="Muon positions were clustered, but autosave failed."
        )
        self._refresh_muon_generation_table()
        self._muon_generation_view.select_site_ids(clustered_ids)
        self._refresh_journal_view()
        self.statusBar().showMessage(f"Clustered {plural(len(clustered), 'muon position')}")

    def _on_create_muon_relaxation_jobs_requested(self, raw_site_ids: object) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        site_ids: list[UUID] = []
        raw_values = raw_site_ids if isinstance(raw_site_ids, list) else [raw_site_ids]
        for raw_value in raw_values:
            try:
                site_ids.append(UUID(str(raw_value)))
            except (TypeError, ValueError):
                continue
        if not site_ids:
            return
        mismatch = self._project_service.validate_sites_match_active_host(self._document, site_ids)
        if mismatch is not None:
            if not self._handle_site_host_mismatch(mismatch):
                return
            mismatch = self._project_service.validate_sites_match_active_host(
                self._document, site_ids
            )
            if mismatch is not None:
                return
        dialog = MuonChargeStateDialog(site_count=len(site_ids), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        charge_states = dialog.selected_charge_states()
        staged = self._project_service.stage_generated_sites(self._document, site_ids)
        if not staged:
            QMessageBox.information(
                self,
                "No generated sites",
                "Select one or more generated muon sites that have not already been staged.",
            )
            return
        rows = [
            {
                "id": str(site.id),
                "label": site.label,
                "fractional_position": list(site.initial_fractional_position),
                "method": site.generation_method or "",
            }
            for site in staged
        ]
        self._muon_relaxation_view.add_staged_sites(rows, charge_states=charge_states)
        self._persist_document_ui_state(
            failure_message="Muon sites were staged, but autosave failed."
        )
        self._refresh_muon_generation_table()
        self._refresh_workflow_defaults(self._muon_relaxation_view)
        self._refresh_muon_relaxation_rows()
        self._refresh_journal_view()
        self._set_active_route("muon_relaxations")
        charge_label = " and ".join(
            muon_charge_state_label(charge_state) for charge_state in charge_states
        )
        self.statusBar().showMessage(
            f"Staged {plural(len(staged), 'muon relaxation position')} for {charge_label}"
        )

    def _on_muon_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state["muon_generation_form"] = state

    def _on_muon_relaxation_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state["muon_relaxation_form"] = state
        self._refresh_muon_relaxation_rows()

    def _on_create_relaxation_analysis_workspace_requested(self, labels: object) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        host = self._project_service.active_host_structure(self._document)
        if host is None:
            QMessageBox.warning(
                self,
                "No host structure",
                "Load a host structure before creating an analysis workspace.",
            )
            return
        label_list = [str(label) for label in labels] if isinstance(labels, list) else [str(labels)]
        sources = relaxation_sources_for_labels(
            self._completed_muon_relaxation_sources(), label_list
        )
        if not sources:
            QMessageBox.information(
                self,
                "No completed relaxations",
                "Select completed, converged structural relaxation rows before creating an analysis workspace.",
            )
            return
        charge_states = {source.charge_state for source in sources}
        if len(charge_states) > 1:
            QMessageBox.warning(
                self,
                "Mixed charge states",
                "Create separate analysis workspaces for positive muon and muonium relaxation results.",
            )
            return

        if not self._accept_mixed_relaxation_calculations(sources):
            return

        charge_state = next(iter(charge_states))
        default_name = f"{muon_charge_state_label(charge_state)} analysis"
        name, accepted = QInputDialog.getText(
            self,
            "Create Analysis Workspace",
            "Workspace name",
            QLineEdit.EchoMode.Normal,
            default_name,
        )
        if not accepted:
            return

        workspace = default_workspace_state(
            name=name,
            charge_state=charge_state,
            host_structure_id=host.id,
            job_ids=[source.job_id for source in sources],
        )
        workspaces = self._relaxation_analysis_workspaces()
        workspaces.append(workspace)
        self._set_relaxation_analysis_workspaces(workspaces)
        self._project_service.append_provenance(
            self._document, analysis_workspace_created_record(workspace)
        )
        # The Journal now names the workspace, so save it with its record.
        self._persist_document_ui_state(
            failure_message="The analysis workspace was created, but autosave failed."
        )
        self._refresh_relaxation_analysis_context()
        self._populate_project_tree()
        self._refresh_journal_view()
        self._set_active_route(self._relaxation_analysis_workspace_route(str(workspace["id"])))
        self.statusBar().showMessage(f"Created analysis workspace '{workspace['name']}'")

    def _on_relaxation_analysis_state_changed(self, workspaces: list) -> None:
        if self._document is None or self._shutting_down:
            return
        cleaned = [dict(workspace) for workspace in workspaces if isinstance(workspace, dict)]
        previous_nodes = [
            (w.get("id"), w.get("name")) for w in self._relaxation_analysis_workspaces()
        ]
        self._set_relaxation_analysis_workspaces(cleaned)
        if [(w.get("id"), w.get("name")) for w in cleaned] != previous_nodes:
            # A rename changes the workspace's project-tree label.
            self._populate_project_tree()
        else:
            self._refresh_workflow_stage_icons()

    def _on_relaxation_analysis_current_workspace_changed(self, workspace_id: str) -> None:
        # Keep the project tree on the workspace whose tab is showing, but only
        # while the Analysis page is the one in view.
        if self._main_stack.currentWidget() is not self._relaxation_analysis_view:
            return
        route = self._relaxation_analysis_workspace_route(str(workspace_id))
        if self._active_route() != route:
            self._set_active_route(route)

    def _open_project_tree_context_menu(self, position) -> QMenu | None:
        item = self._project_tree.itemAt(position)
        if item is None:
            return None
        menu = self._project_tree_context_menu(item)
        if menu is not None:
            menu.exec(self._project_tree.viewport().mapToGlobal(position))
        return menu

    def _project_tree_context_menu(self, item: QTreeWidgetItem) -> QMenu | None:
        """Build the right-click menu for a project-tree node, if it has one.

        Only user-created workspace nodes (structural-relaxation analyses and
        saved magnetic field workspaces) can be deleted from the tree; workflow
        stages have no menu.
        """

        if self._document is None:
            return None
        route = item.data(0, Qt.ItemDataRole.UserRole)
        menu = QMenu(self._project_tree)
        if (analysis_id := self._relaxation_analysis_workspace_id_from_route(route)) is not None:
            action = menu.addAction("Delete Analysis…")
            action.triggered.connect(
                lambda _checked=False, wid=analysis_id: self._delete_relaxation_analysis_workspace(
                    wid
                )
            )
        elif (magnetic_id := self._magnetic_field_workspace_id_from_route(route)) is not None:
            action = menu.addAction("Delete Workspace…")
            action.triggered.connect(
                lambda _checked=False, wid=magnetic_id: self._delete_magnetic_field_workspace(wid)
            )
        else:
            menu.deleteLater()
            return None
        return menu

    def _delete_relaxation_analysis_workspace(self, workspace_id: str) -> None:
        """Delete one analysis workspace, keeping the relaxations it drew on.

        A preliminary analysis made before every relaxation finished goes stale
        once the full set is in; deleting it only drops the workspace and its
        clustering. A Muon Sites summary pin taken from it is removed as well,
        so the summary never keeps showing sites from an analysis that is gone.
        """

        if self._document is None:
            return
        workspaces = self._relaxation_analysis_workspaces()
        target = next(
            (
                workspace
                for workspace in workspaces
                if str(workspace.get("id", "")) == str(workspace_id)
            ),
            None,
        )
        if target is None:
            return
        name = str(target.get("name", "")).strip() or "Relaxation analysis"
        pinned_charge_states = [
            str(charge_state)
            for charge_state, snapshot in muon_sites_summary_snapshots(self._document).items()
            if isinstance(snapshot, dict)
            and str(snapshot.get("workspace_id", "")) == str(workspace_id)
        ]
        detail = (
            "Its clustering results are discarded. The structural relaxations it "
            "uses are kept and can be added to another analysis."
        )
        if pinned_charge_states:
            detail += " Its sites are pinned to the Muon Sites summary; that pin is removed too."
        if not workflow_table_utils.confirm_destructive(
            self,
            "Delete analysis",
            f"Delete analysis '{name}'?",
            detail,
            accept_label="Delete",
        ):
            return
        self._set_relaxation_analysis_workspaces(
            [workspace for workspace in workspaces if workspace is not target]
        )
        for charge_state in pinned_charge_states:
            store_muon_sites_summary_snapshot(self._document, charge_state, None)
        if self._active_route() == self._relaxation_analysis_workspace_route(str(workspace_id)):
            # Step up to the Analysis node first: rebuilding the tree without
            # the selected node would otherwise fall back to the home page.
            self._set_active_route("muon_analysis")
        self._refresh_relaxation_analysis_context()
        self._populate_project_tree()
        if pinned_charge_states:
            self._refresh_muon_sites_summary()
            self._refresh_pinned_site_consumers()
        self._persist_document_ui_state(
            failure_message="The analysis was deleted, but autosave failed."
        )
        self.statusBar().showMessage(f"Deleted analysis '{name}'")

    def _on_relaxation_analysis_add_sites_requested(self, workspace_id: str) -> None:
        if self._document is None:
            return
        workspace = next(
            (
                entry
                for entry in self._relaxation_analysis_workspaces()
                if str(entry.get("id", "")) == str(workspace_id)
            ),
            None,
        )
        if workspace is None:
            return
        charge_state = str(workspace.get("charge_state", POSITIVE_MUON_CHARGE_STATE))
        existing_ids = set()
        raw_job_ids = workspace.get("job_ids", [])
        if isinstance(raw_job_ids, list):
            for raw_job_id in raw_job_ids:
                try:
                    existing_ids.add(UUID(str(raw_job_id)))
                except (TypeError, ValueError):
                    continue
        sources = [
            source
            for source in self._completed_muon_relaxation_sources()
            if source.charge_state == charge_state
        ]
        if not sources:
            QMessageBox.information(
                self,
                "No matching relaxations",
                f"No completed {muon_charge_state_label(charge_state)} relaxation results are available to add.",
            )
            return
        dialog = RelaxationAnalysisJobDialog(
            title="Add Structure to Analysis",
            sources=sources,
            excluded_job_ids=existing_ids,
            reference_job_ids=existing_ids,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_job_ids()
        if not selected:
            return
        joined = [
            source
            for source in sources
            if source.job_id in existing_ids or source.job_id in set(selected)
        ]
        if not self._accept_mixed_relaxation_calculations(joined):
            return
        self._relaxation_analysis_view.add_job_ids_to_workspace(str(workspace_id), selected)
        self.statusBar().showMessage(
            f"Added {plural(len(selected), 'relaxation result')} to the analysis workspace"
        )

    def _accept_mixed_relaxation_calculations(
        self, sources: list[CompletedRelaxationAnalysisSource]
    ) -> bool:
        """Return whether to go on with relaxations from more than one calculation.

        Total energies are only comparable between relaxations in the same
        supercell with the same settings, so mixing them asks first, with
        ``Cancel`` the default.
        """

        if not mixes_calculations(sources):
            return True
        answer = workflow_table_utils.confirm_choice(
            self,
            "Different calculations",
            "These relaxations ran in different supercells or with different settings "
            "(cutoff, k-points, functional, Hubbard U or spin), so their energies are not comparable "
            "and the workspace's dE values would be meaningless.",
            (("add", "Use them anyway"), ("cancel", "Cancel")),
            default="cancel",
            escape="cancel",
            icon=QMessageBox.Icon.Warning,
        )
        return answer == "add"

    def _record_relaxed_sites_grouped(self, workspace_id: str) -> None:
        if self._document is None:
            return
        workspace = next(
            (
                entry
                for entry in self._relaxation_analysis_workspaces()
                if str(entry.get("id", "")) == str(workspace_id)
            ),
            None,
        )
        if workspace is None:
            return
        record = relaxed_sites_grouped_record(workspace, self._workspace_host(workspace))
        if record is None:
            return
        # A repeat adds no record, but the grouping it applied is still saved.
        record_finding(self._document, record)
        self._persist_document_ui_state(
            failure_message="The grouping was applied, but autosave failed."
        )
        self._refresh_journal_view()

    def _workspace_host(self, workspace: dict) -> Structure | None:
        """Return the host structure a relaxation analysis workspace was made for."""

        if self._document is None:
            return None
        host_id = uuid_or_none(workspace.get("host_structure_id"))
        return self._document.structures.get(host_id) if host_id is not None else None

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

    def _on_muon_sites_pin_requested(self, workspace_id: str) -> None:
        if self._document is None:
            return
        workspace = next(
            (
                entry
                for entry in self._relaxation_analysis_workspaces()
                if str(entry.get("id", "")) == str(workspace_id)
            ),
            None,
        )
        if workspace is None:
            return
        snapshot = build_muon_sites_summary_snapshot(workspace, self._workspace_host(workspace))
        if snapshot is None:
            QMessageBox.information(
                self,
                "Nothing to pin",
                "Run clustering on this workspace first — the Muon Sites summary shows the "
                "distinct clustered muon sites.",
            )
            return
        if str(snapshot.get("host_structure_id", "")) != (self._host_structure_id_text() or ""):
            # The sites are fractional coordinates in the workspace's host; pinning
            # while a different host is active would only ever validate to a hidden
            # panel, so reject it explicitly rather than storing a silent pin.
            QMessageBox.information(
                self,
                "Cannot pin",
                "This analysis was run against a host structure that is not the active "
                "host. Switch to that host before pinning it to the Muon Sites summary.",
            )
            return
        charge_state = str(snapshot.get("charge_state", POSITIVE_MUON_CHARGE_STATE))
        if charge_state not in (POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE):
            # The summary only has a panel per canonical charge state; storing a
            # snapshot under any other key would silently vanish (never rendered,
            # never unpinnable) despite a success message.
            QMessageBox.information(
                self,
                "Cannot pin",
                "This workspace's charge state is not supported by the Muon Sites summary.",
            )
            return
        store_muon_sites_summary_snapshot(self._document, charge_state, snapshot)
        self._refresh_muon_sites_summary()
        self._refresh_pinned_site_consumers()
        self._set_active_route("muon_sites")
        self._persist_document_ui_state(
            failure_message="The muon sites summary was pinned, but autosave failed."
        )
        self.statusBar().showMessage(
            f"Pinned {muon_charge_state_label(charge_state)} sites to the Muon Sites summary"
        )

    def _on_unpin_muon_sites_requested(self, charge_state: str) -> None:
        if self._document is None:
            return
        key = str(charge_state)
        if key not in muon_sites_summary_snapshots(self._document):
            return
        store_muon_sites_summary_snapshot(self._document, key, None)
        self._refresh_muon_sites_summary()
        self._refresh_pinned_site_consumers()
        self._persist_document_ui_state(
            failure_message="The muon sites summary was updated, but autosave failed."
        )
        self.statusBar().showMessage(
            f"Unpinned {muon_charge_state_label(key)} sites from the Muon Sites summary"
        )

    def _navigate_to_pinned_analysis(self, workspace_id: str, cluster_index: int | None) -> None:
        self._set_active_route("muon_analysis")
        resolved = self._relaxation_analysis_view.focus_workspace(
            str(workspace_id), cluster_index=cluster_index
        )
        if not resolved:
            self.statusBar().showMessage(
                "The pinned analysis has changed — opened the Analysis tab instead"
            )

    def _on_open_muon_analysis_requested(self, workspace_id: str) -> None:
        self._navigate_to_pinned_analysis(str(workspace_id), None)

    def _on_focus_muon_site_requested(self, workspace_id: str, cluster_index: int) -> None:
        self._navigate_to_pinned_analysis(str(workspace_id), int(cluster_index))

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
        self._populate_project_tree()
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
        self._populate_project_tree()
        self.statusBar().showMessage(message)

    def _workflow_source_structure(self, view) -> Structure | None:
        """Return the structure a relaxation or UEP view's rows are built on.

        UEP rows run on the selected UEP source structure; the relaxation
        workflows run on the active host.
        """

        if self._document is None:
            return None
        if view is self._uep_view:
            return self._selected_uep_source_structure()
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

    def _available_uep_source_structures(self) -> list[Structure]:
        if self._document is None:
            return []
        host = self._project_service.active_host_structure(self._document)
        if host is None:
            return []
        derived = [
            structure
            for structure in self._document.structures.values()
            if structure.origin == "pristine_relaxation"
            and structure.parent_structure_id == host.id
        ]
        derived.sort(key=lambda structure: (structure.label.casefold(), str(structure.id)))
        return [host, *derived]

    def _selected_uep_source_structure(self) -> Structure | None:
        if self._document is None:
            return None
        selected_id = None
        raw_state = self._document.ui_state.get("uep_form", {})
        if isinstance(raw_state, dict):
            raw_id = raw_state.get("selected_source_structure_id")
            if isinstance(raw_id, str):
                try:
                    selected_id = UUID(raw_id)
                except ValueError:
                    selected_id = None
        available = self._available_uep_source_structures()
        if selected_id is not None:
            structure = self._document.structures.get(selected_id)
            if structure is not None:
                return structure
        return available[0] if available else None

    def _refresh_uep_source_structures(self) -> None:
        if self._document is None:
            self._uep_view.set_source_structure_options(
                [], selected_id=None, summary="Create or open a project first."
            )
            self._uep_view.set_parameter_species_options(())
            return

        structures = self._available_uep_source_structures()
        selected = self._selected_uep_source_structure()
        selected_id = str(selected.id) if selected is not None else None
        if not structures:
            self._uep_view.set_source_structure_options(
                [],
                selected_id=None,
                summary="Load a host structure before configuring the UEP workflow.",
            )
            self._uep_view.set_parameter_species_options(())
            return

        summary = (
            f"Using source structure: {selected.label}"
            if selected is not None
            else "Choose the host structure or a completed pristine-relaxed structure as the UEP source."
        )
        self._uep_view.set_source_structure_options(
            [{"id": str(structure.id), "label": structure.label} for structure in structures],
            selected_id=selected_id,
            summary=summary,
        )
        self._uep_view.set_parameter_species_options(structure_species_options(selected))

    def _on_uep_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state["uep_form"] = state
        if getattr(self, "_handling_uep_form_state", False):
            return
        self._handling_uep_form_state = True
        try:
            self._refresh_workflow_defaults(self._uep_view)
        finally:
            self._handling_uep_form_state = False

    def _muon_relaxation_row_signature(self, row: MuonRelaxationRow) -> tuple[object, ...]:
        return (
            str(row.site_id),
            row.label,
            tuple(float(value) for value in row.initial_fractional_position),
            str(row.charge_state),
            normalized_supercell_matrix(row.supercell_matrix),
            normalized_parameter_values(row.parameter_values),
            tuple(str(name) for name in row.added_advanced),
            # Included deliberately. If the imported marker is ever dropped from
            # one side of the form/definition pair, the signatures stop matching
            # and the row visibly falls back to "ready" -- far better than the
            # row rendering correctly while silently becoming editable.
            bool(row.imported),
            str(row.import_batch_id) if row.import_batch_id is not None else None,
        )

    def _uep_row_signature(self, row: UepSourceRow) -> tuple[object, ...]:
        return (
            normalized_supercell_matrix(row.supercell_matrix),
            normalized_parameter_values(row.parameter_values),
            tuple(str(name) for name in row.added_advanced),
        )

    def _sync_active_muon_relaxation_dashboard_jobs(self, target_definition) -> None:
        if self._document is None:
            return
        config = deserialize_muon_relaxation_config(
            target_definition.parameters.get("configuration", {})
        )
        if config is None:
            return

        mapping = ensure_job_mapping(target_definition)
        for row in config.rows:
            key = self._muon_relaxation_row_key(row)
            existing = mapping.get(key)
            if (
                isinstance(existing, str)
                and self._muon_relaxation_controller.job_for_mapping_value(key, existing)
                is not None
            ):
                continue

            expected_calc_type = muon_relaxation_calc_type(row.charge_state)
            candidates = [
                job
                for job in self._document.jobs.values()
                if job.workflow_stage == WorkflowStage.MUON_RELAXATION
                and job.calc_type == expected_calc_type
                and job.label == row.label
                and job.status == JobStatus.ACTIVE
            ]
            if not candidates:
                continue

            job = max(
                candidates,
                key=lambda candidate: (
                    relaxation_mapping_job_priority(self._document, str(candidate.id)),
                    self._job_submission_sort_key(candidate),
                ),
            )
            mapping[key] = str(job.id)

    def _uep_source_structure_id_for_job(self, job_id: UUID) -> UUID | None:
        if self._document is None:
            return None
        return uep_source_structure_id(self._document, job_id)

    def _active_uep_job_ids(self) -> set[UUID]:
        definition = self._uep_controller.active_definition()
        if definition is None:
            return set()
        self._uep_controller.sync_mappings(definition)
        mapping = definition.parameters.get("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            return set()
        job_ids: set[UUID] = set()
        for raw_job_id in mapping.values():
            try:
                job_ids.add(UUID(str(raw_job_id)))
            except ValueError:
                continue
        return job_ids

    @staticmethod
    def _uep_classification_label(result: dict) -> str:
        classification = result.get("classification")
        if isinstance(classification, dict):
            return str(classification.get("label", ""))
        return str(classification or "")

    def _uep_cluster_source_rows(self) -> list[dict[str, object]]:
        if self._document is None:
            return []
        states = self._document.ui_state.get(self._UEP_ANALYSIS_STATES_KEY, {})
        if not isinstance(states, dict):
            return []

        active_job_ids = self._active_uep_job_ids()
        if not active_job_ids:
            return []

        rows: list[dict[str, object]] = []
        for raw_job_id, state in states.items():
            if not isinstance(state, dict):
                continue
            try:
                job_id = UUID(str(raw_job_id))
            except ValueError:
                continue
            if job_id not in active_job_ids:
                continue
            job = self._document.jobs.get(job_id)
            if job is None:
                continue
            source_job_label = job.label
            source_structure_id = self._uep_source_structure_id_for_job(job_id)
            batch_results = state.get("batch_results")
            batch_summary = state.get("batch_summary_data")
            if not isinstance(batch_results, list) or not isinstance(batch_summary, dict):
                continue
            clusters = batch_summary.get("clusters")
            if not isinstance(clusters, (list, tuple)):
                continue
            clustering_properties = {
                "clustering_method": batch_summary.get("clustering_method"),
                "threshold_ang": batch_summary.get("threshold_ang"),
                "energy_tolerance_ev": batch_summary.get("energy_tolerance_ev"),
            }
            for cluster in clusters:
                if not isinstance(cluster, dict):
                    continue
                try:
                    cluster_index = int(cluster.get("cluster_index"))
                    cluster_size = int(cluster.get("size", 0))
                    representative_index = int(cluster.get("representative_index"))
                except (TypeError, ValueError):
                    continue

                def _representative_row(result_index: int) -> dict[str, object] | None:
                    if not (0 <= result_index < len(batch_results)):
                        return None
                    result = batch_results[result_index]
                    if not isinstance(result, dict):
                        return None
                    # The group reports its representative in the asymmetric unit.
                    fractional = cluster.get(
                        "representative_fractional_position",
                        result.get("optimized_fractional_position"),
                    )
                    cartesian = cluster.get(
                        "representative_position_ang", result.get("optimized_position_ang")
                    )
                    if not isinstance(fractional, (list, tuple)) or len(fractional) != 3:
                        return None
                    if not isinstance(cartesian, (list, tuple)) or len(cartesian) != 3:
                        return None
                    classification = self._uep_classification_label(result)
                    properties = {
                        **clustering_properties,
                        "source_job_label": source_job_label,
                        "source_job_id": str(job_id),
                        "cluster_index": cluster_index,
                        "cluster_size": cluster_size,
                        "representative_index": representative_index,
                        "member_index": result_index,
                        "selection_kind": "cluster",
                        "classification": classification,
                        "classical_energy_ev": result.get("classical_energy_ev"),
                        "zero_point_energy_ev": result.get("zero_point_energy_ev"),
                        "total_energy_ev": result.get("total_energy_ev"),
                    }
                    return {
                        "source_key": f"{job_id}:cluster:{cluster_index}:cluster:{result_index}",
                        "source_job_id": str(job_id),
                        "source_job_label": source_job_label,
                        "source_structure_id": (
                            "" if source_structure_id is None else str(source_structure_id)
                        ),
                        "cluster_index": cluster_index,
                        "cluster_size": cluster_size,
                        "selection_kind": "cluster",
                        "classification": classification,
                        "classical_energy_ev": result.get("classical_energy_ev"),
                        "zero_point_energy_ev": result.get("zero_point_energy_ev"),
                        "total_energy_ev": result.get("total_energy_ev"),
                        "fractional_position": [float(value) for value in fractional],
                        "cartesian_position": [float(value) for value in cartesian],
                        "generation_method": "uep_cluster",
                        "label": f"UEP Site {cluster_index}",
                        "properties": properties,
                    }

                representative_row = _representative_row(representative_index)
                if representative_row is not None:
                    try:
                        rows.append(uep_generation_selection(self._document, representative_row))
                    except SiteGenerationError:
                        continue

        rows.sort(
            key=lambda row: (
                str(row.get("source_job_label", "")),
                int(row.get("cluster_index", 0)),
                str(row.get("source_key", "")),
            )
        )
        return rows

    def _refresh_uep_rows(self) -> None:
        # UEP row state building moved to ``UepController.build_states``; this thin
        # wrapper delegates to the controller's ``refresh_view`` (mirroring
        # ``_refresh_muon_relaxation_rows``) so its many call sites — and the test
        # that monkeypatches ``_refresh_uep_rows`` — keep working unchanged.
        self._uep_controller.refresh_view()

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
        """Route a workflow table's row-menu Cancel / Inspect Failure / Reset signals.

        Each signal carries the selected row keys; the generic handlers below
        resolve them to jobs through ``controller.jobs_for_keys``. The row
        menu's job-folder entries ask ``controller.job_files_exports``.
        """

        view.cancel_requested.connect(partial(self._cancel_workflow_rows, controller))
        view.inspect_failed_requested.connect(partial(self._inspect_workflow_rows, controller))
        view.reset_failed_requested.connect(partial(self._reset_workflow_rows, controller))
        view.job_files_menu.set_provider(controller.job_files_exports)

    def _cancel_workflow_rows(self, controller, keys: object) -> None:
        """Cancel the jobs behind a workflow table's rows (row menu "Cancel")."""

        self._jobs.cancel_jobs(controller.jobs_for_keys(list(keys)))

    def _inspect_workflow_rows(self, controller, keys: object) -> None:
        """Show failure diagnostics for a workflow table's rows (row menu "Inspect Failure")."""

        self._jobs.inspect_failed_jobs(controller.jobs_for_keys(list(keys)))

    def _reset_workflow_rows(self, controller, keys: object) -> None:
        """Reset the failed jobs behind a workflow table's rows (row menu "Reset Failed Jobs...")."""

        self._jobs.reset_failed_jobs(controller.jobs_for_keys(list(keys)))

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
        removed_site_ids = self._muon_relaxation_site_ids_for_removed_rows(definitions, labels)
        should_autosave = self._relaxation_removal_touches_project_files(definitions, key_matches)
        if not self._cleanup_relaxation_rows(controller.jobs_for_removed_rows(labels)):
            self._restore_muon_relaxation_form_from_definition()
            return
        controller.remove_rows(labels)
        self._remove_muon_relaxation_labels_from_ui_state(labels)
        site_status_changed = self._restore_removed_muon_relaxation_sites(removed_site_ids)
        self._refresh_muon_relaxation_rows()
        if should_autosave or self._site_status_revert_touches_project_files(
            removed_site_ids, site_status_changed
        ):
            self._persist_document_ui_state(
                failure_message="Muon relaxation rows were removed, but autosave failed."
            )

    def _muon_relaxation_site_ids_for_removed_rows(
        self,
        definitions: list,
        labels: set[str],
    ) -> set[UUID]:
        if self._document is None:
            return set()

        key_matches = self._muon_relaxation_controller.row_key_matcher(labels)
        site_ids: set[UUID] = set()

        def collect_from_rows(rows: object) -> None:
            if not isinstance(rows, list):
                return
            for row in rows:
                if not isinstance(row, dict):
                    continue
                label = str(row.get("label", "")).strip()
                charge_state = str(row.get("charge_state", "")).strip()
                keys = (
                    [f"{muon_charge_state_prefix(charge_state)}:{label}"]
                    if charge_state
                    else [label]
                )
                if not any(key_matches(key) for key in keys):
                    continue
                try:
                    site_ids.add(UUID(str(row.get("site_id"))))
                except (TypeError, ValueError):
                    continue

        state = self._document.ui_state.get("muon_relaxation_form", {})
        if isinstance(state, dict):
            collect_from_rows(state.get("rows", []))

        for definition in definitions:
            config = definition.parameters.get("configuration", {})
            if isinstance(config, dict):
                collect_from_rows(config.get("rows", []))

        return site_ids

    def _restore_removed_muon_relaxation_sites(self, site_ids: set[UUID]) -> bool:
        if self._document is None:
            return False

        changed = False
        for site_id in site_ids:
            site = self._document.sites.get(site_id)
            if (
                site is None
                or site.status != "staged"
                or self._site_has_muon_relaxation_rows(site_id)
            ):
                continue
            site.status = "generated"
            changed = True
        return changed

    def _site_has_muon_relaxation_rows(self, site_id: UUID) -> bool:
        if self._document is None:
            return False

        def rows_include_site(rows: object) -> bool:
            if not isinstance(rows, list):
                return False
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    if UUID(str(row.get("site_id"))) == site_id:
                        return True
                except (TypeError, ValueError):
                    continue
            return False

        state = self._document.ui_state.get("muon_relaxation_form", {})
        if isinstance(state, dict) and rows_include_site(state.get("rows", [])):
            return True

        for definition in self._muon_relaxation_controller.definitions():
            config = definition.parameters.get("configuration", {})
            if isinstance(config, dict) and rows_include_site(config.get("rows", [])):
                return True
        return False

    def _site_status_revert_touches_project_files(self, site_ids: set[UUID], changed: bool) -> bool:
        if not changed:
            return False
        return any(self._entity_file("sites", site_id).exists() for site_id in site_ids)

    def _remove_muon_relaxation_labels_from_ui_state(self, labels: set[str]) -> None:
        if self._document is None:
            return
        state = self._document.ui_state.get("muon_relaxation_form", {})
        if not isinstance(state, dict):
            return
        rows = state.get("rows", [])
        if not isinstance(rows, list):
            return
        key_matches = self._muon_relaxation_controller.row_key_matcher(labels)
        kept: list[object] = []
        for row in rows:
            if not isinstance(row, dict):
                kept.append(row)
                continue
            label = str(row.get("label", "")).strip()
            charge_state = str(row.get("charge_state", "")).strip()
            keys = (
                [f"{muon_charge_state_prefix(charge_state)}:{label}"] if charge_state else [label]
            )
            if any(key_matches(key) for key in keys):
                continue
            kept.append(row)
        updated = dict(state)
        updated["rows"] = kept
        self._document.ui_state["muon_relaxation_form"] = updated

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
                    label = str(row.get("label", "")).strip()
                    charge_state = str(row.get("charge_state", "")).strip()
                    keys = (
                        [f"{muon_charge_state_prefix(charge_state)}:{label}"]
                        if charge_state
                        else [label]
                    )
                    if any(key_matches(key) for key in keys):
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

    def _restore_muon_relaxation_form_from_definition(self) -> None:
        definition = self._muon_relaxation_controller.active_definition()
        if definition is None:
            return
        config = deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return
        state = serialize_muon_relaxation_config(config)
        for row in state.get("rows", []):
            if isinstance(row, dict) and "fractional_position" not in row:
                row["fractional_position"] = list(
                    row.get("initial_fractional_position", (0.0, 0.0, 0.0))
                )
        self._muon_relaxation_view.set_form_state(state)
        if self._document is not None:
            self._document.ui_state["muon_relaxation_form"] = state
        self._refresh_muon_relaxation_rows()

    def _on_remove_uep_rows(self, payload: object) -> None:
        labels = self._removed_row_labels(payload)
        if self._document is None or not labels:
            return
        self._uep_controller.remove_rows(labels)
        self._refresh_uep_rows()

    def _on_import_castep_relaxations(self) -> None:
        """Scan a folder for CASTEP relaxations and import the ones chosen from it."""

        document = self._document
        if document is None:
            QMessageBox.information(
                self,
                "No project open",
                "Open or create a project before importing CASTEP relaxations.",
            )
            return

        host = self._project_service.active_host_structure(document)
        if host is None:
            # Deliberately not offered as "import anyway": every imported
            # structure must hang off the active host, or its downstream jobs
            # fail host resolution at submission time.
            QMessageBox.information(
                self,
                "No host structure",
                "Import a host structure first. Imported relaxations are anchored to "
                "the project's active host so the sites they create can be used "
                "downstream.",
            )
            return

        directory = QFileDialog.getExistingDirectory(
            self, "Select a folder containing CASTEP relaxations"
        )
        if not directory:
            return

        self._begin_relaxation_scan(Path(directory), host)

    def _begin_relaxation_scan(self, root: Path, host: Structure) -> None:
        """Scan ``root`` on a worker thread, then review and commit."""

        self.statusBar().showMessage(f"Scanning {root.name} for CASTEP relaxations…")
        if self._import_relaxations_action is not None:
            self._import_relaxations_action.setEnabled(False)
        self._relaxation_scan_host = host
        self._relaxation_scan_root = root
        start_relaxation_scan(
            self,
            root,
            [list(vector) for vector in host.lattice_vectors],
            list(host.species),
            self._on_relaxation_scan_completed,
            self._on_relaxation_scan_failed,
            self._on_relaxation_scan_progress,
        )

    @Slot(int, int)
    def _on_relaxation_scan_progress(self, done: int, total: int) -> None:
        root = self._relaxation_scan_root
        name = root.name if root is not None else "folder"
        self.statusBar().showMessage(f"Scanning {name}: relaxation {done} of {total}…")

    @Slot(str)
    def _on_relaxation_scan_failed(self, message: str) -> None:
        self._finish_relaxation_scan()
        QMessageBox.warning(self, "Import failed", f"Could not scan the folder:\n{message}")

    @Slot(object)
    def _on_relaxation_scan_completed(self, scan: object) -> None:
        self._finish_relaxation_scan()
        document = self._document
        host = self._relaxation_scan_host
        if document is None or host is None or not isinstance(scan, RelaxationFolderScan):
            return

        if not scan.runs:
            if scan.unreadable:
                QMessageBox.information(self, "Could not read folder", scan.unreadable_note)
                return
            QMessageBox.information(
                self,
                "Nothing to import",
                "No CASTEP outputs (.castep files) were found in or under that folder.",
            )
            return

        dialog = RelaxationImportDialog(scan, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_runs()
        if not selected:
            return

        self._commit_relaxation_import(scan, selected, host)

    def _finish_relaxation_scan(self) -> None:
        self._relaxation_scan_root = None
        if self._import_relaxations_action is not None:
            self._import_relaxations_action.setEnabled(True)
        self.statusBar().clearMessage()

    def _commit_relaxation_import(
        self, scan: RelaxationFolderScan, runs: list, host: Structure
    ) -> None:
        """Build entities and files, attach the rows, and save."""

        document = self._document
        if document is None:
            return

        output_root = document.project_root / "files" / WorkflowStage.MUON_RELAXATION.value
        definition = self._muon_relaxation_controller.active_definition()
        existing_labels: list[str] = []
        if definition is not None:
            config = deserialize_muon_relaxation_config(
                definition.parameters.get("configuration", {})
            )
            if config is not None:
                existing_labels = [row.label for row in config.rows]

        try:
            outcome = commit_relaxation_import(
                document,
                scan=scan,
                runs=runs,
                host_structure=host,
                output_root=output_root,
                existing_labels=existing_labels,
            )
        except OSError as exc:
            QMessageBox.warning(
                self, "Import failed", f"Could not write the imported files: {plain_error(exc)}."
            )
            return

        if not self.merge_imported_relaxation_rows(outcome, output_directory=output_root):
            QMessageBox.warning(
                self,
                "Import incomplete",
                "The relaxations were imported but could not be attached to the relaxations table.",
            )
            return

        # Only once the merge has succeeded: append_provenance writes straight to
        # provenance.jsonl, so recording the import before the rows were attached
        # would leave a durable claim about a state the project never reached.
        if outcome.provenance is not None:
            self._project_service.append_provenance(document, outcome.provenance)

        self._refresh_muon_relaxation_rows()
        self._refresh_jobs_table()
        self._refresh_muon_sites_summary()
        self._persist_document_ui_state(
            failure_message="CASTEP relaxations were imported, but autosave failed.",
        )

        messages = [f"Imported {plural(len(outcome.imported), 'relaxation')}."]
        if outcome.skipped:
            messages.append(f"{plural(len(outcome.skipped), 'run was', 'runs were')} skipped.")
        row_warnings = [note for item in outcome.imported for note in item.warnings]
        if row_warnings:
            messages.append("")
            messages.extend(f"- {note}" for note in dict.fromkeys(row_warnings))
        QMessageBox.information(self, "Import complete", "\n".join(messages))
        self.statusBar().showMessage(
            f"Imported {plural(len(outcome.imported), 'CASTEP relaxation')}"
        )

    def _on_remove_imported_relaxations(self) -> None:
        """Remove one import batch, or explain why it cannot be removed.

        An imported relaxation owns the only copy of its results, so this is
        irreversible in a way that removing a natively-run relaxation is not:
        that one can be re-run. The reference check therefore refuses rather
        than warns, and the confirmation names what will be deleted.
        """

        document = self._document
        if document is None:
            QMessageBox.information(self, "No project open", "Open a project first.")
            return

        batches = self._imported_relaxation_batches()
        if not batches:
            QMessageBox.information(
                self,
                "Nothing imported",
                "This project contains no imported relaxations.",
            )
            return

        labels = [f"{label} ({plural(count, 'relaxation')})" for label, count, _batch_id in batches]
        chosen = choose_import_batch(self, labels)
        if chosen is None:
            return
        batch_id = batches[chosen][2]

        refs = imported_relaxations_in_batch(document, batch_id)
        directories = {
            ref.job_id: path
            for ref in refs
            if (path := self._job_output_directory(document.jobs[ref.job_id])) is not None
        }
        others = {
            job.id: path
            for job in document.jobs.values()
            if job.id not in directories and (path := self._job_output_directory(job)) is not None
        }
        plan = plan_removal(
            document, refs, job_directories=directories, other_job_directories=others
        )

        if plan.unsafe:
            self._warn_with_lines(
                "Cannot remove safely",
                f"MuFinder will not delete the folders of {plural(len(plan.unsafe), 'relaxation')}, "
                "so nothing was removed.",
                plan.unsafe,
            )
            return

        if plan.blockers:
            self._warn_with_lines(
                "Still in use",
                "These imported relaxations cannot be removed because other work uses them. "
                "Remove that first.",
                plan.blockers,
            )
            return

        if not workflow_table_utils.confirm_destructive(
            self,
            "Remove imported relaxations?",
            self._removal_question(plan),
            "MuFinder did not run these calculations and cannot recreate them.",
            accept_label=(f"Remove {plural(len(plan.targets), 'relaxation', 'relaxations')}"),
        ):
            return

        outcome = remove_imported_relaxations(document, plan, batch_id=batch_id)
        removed, errors = outcome.removed, outcome.errors
        if not removed:
            self._warn_with_lines("Cannot remove safely", "Nothing was removed.", errors)
            return
        # Only what actually went: a row whose job stayed keeps its binding.
        gone = [ref for ref in plan.targets if ref.job_id not in document.jobs]
        definition = self._muon_relaxation_controller.active_definition()
        if definition is not None:
            detach_rows_from_definition(definition, gone)
            self._muon_relaxation_view.set_form_state(
                self._muon_relaxation_form_state_without(gone)
            )
            document.ui_state["muon_relaxation_form"] = self._muon_relaxation_view.collect_state()

        # Appended once the document edits are done, because append_provenance
        # writes straight to provenance.jsonl: recording the removal before the
        # rows were detached would leave a durable claim about a state the
        # project never reached.
        if outcome.provenance is not None:
            self._project_service.append_provenance(document, outcome.provenance)

        self._refresh_muon_relaxation_rows()
        self._refresh_jobs_table()
        self._refresh_muon_sites_summary()
        self._persist_document_ui_state(
            failure_message="Imported relaxations were removed, but autosave failed.",
        )

        if errors:
            self._warn_with_lines("Removal issues", "Some files could not be deleted.", errors)
            self.statusBar().showMessage(
                f"Removed {plural(removed, 'imported relaxation')}; some files could not be deleted"
            )
            return
        kept = [
            KeptFolder(job_directory_label(document.project_root, path), because)
            for paths, because in (
                (outcome.kept_directories, _KEPT_FOR_ANOTHER_IMPORT),
                (outcome.misplaced_directories, KEPT_AS_NOT_ITS_OWN),
            )
            for path in paths
        ]
        self.statusBar().showMessage(
            folder_outcome_status(
                f"Removed {plural(removed, 'imported relaxation')}",
                deleted=len(outcome.deleted_directories),
                kept=len(kept),
            )
        )
        self._show_kept_folders(kept)

    def _warn_with_lines(self, title: str, summary: str, lines: Sequence[str]) -> None:
        """Warn with ``summary`` and one line per item, kept on screen however many there are."""

        if len(lines) == 1:
            QMessageBox.warning(self, title, f"{summary}\n\n- {lines[0]}")
            return
        show_detailed_warning(
            self,
            title,
            summary + "\n" + "\n".join(f"- {line}" for line in lines),
            informative_text="Open Details to see each one.",
        )

    @staticmethod
    def _removal_question(plan) -> str:
        """Return what removing a batch will do with its folders, in plain words."""

        count = plural(len(plan.targets), "imported relaxation")
        stays = (
            "another import uses the same folder. The folder is deleted when the last import "
            "using it is removed."
        )
        elsewhere = (
            "the location saved for {them} is not {their} own folder. Nothing there is touched."
        )
        single = len(plan.targets) == 1
        whose = "Its" if single else "Their"
        if not plan.directories and plan.kept_job_ids and not plan.misplaced_job_ids:
            return f"Remove {count} from the project? {whose} files stay, because {stays}"
        if not plan.directories and plan.misplaced_job_ids and not plan.kept_job_ids:
            why = (
                elsewhere.format(them="it", their="its")
                if single
                else (elsewhere.format(them="them", their="their"))
            )
            return f"Remove {count} from the project? {whose} files stay, because {why}"
        if plan.directories:
            directories = "its job directory" if single else "their job directories"
            question = (
                f"Permanently remove {count} and delete {directories}, including the "
                "archived original input files?"
            )
        else:
            question = f"Remove {count} from the project?"
        if plan.kept_job_ids:
            kept = plural(len(plan.kept_job_ids), "relaxation")
            question += f"\n\nThe files of {kept} stay, because {stays}"
        if plan.misplaced_job_ids:
            misplaced = plural(len(plan.misplaced_job_ids), "relaxation")
            why = (
                elsewhere.format(them="it", their="its")
                if len(plan.misplaced_job_ids) == 1
                else elsewhere.format(them="them", their="their")
            )
            question += f"\n\nThe files of {misplaced} stay, because {why}"
        return question

    def _muon_relaxation_form_state_without(self, targets) -> dict[str, object]:
        """Return the widget's form state with ``targets``' rows dropped."""

        state = self._muon_relaxation_view.collect_state()
        rows = state.get("rows", []) if isinstance(state, dict) else []
        removed = {str(ref.site_id) for ref in targets}
        return {"rows": [row for row in rows if str(row.get("site_id", "")) not in removed]}

    def _imported_relaxation_batches(self) -> list[tuple[str, int, UUID]]:
        """Return ``(label, count, batch_id)`` for every import batch, oldest first.

        A label names the source folder and when it was imported, since one folder
        can be imported more than once. A batch whose record is missing comes last.
        """

        document = self._document
        if document is None:
            return []
        counts: dict[str, int] = {}
        for site in document.sites.values():
            raw = site.properties.get("import_batch_id")
            if raw:
                counts[str(raw)] = counts.get(str(raw), 0) + 1

        labels: dict[str, str] = {}
        order: dict[str, float] = {}
        for record in document.provenance:
            if record.action not in IMPORT_ACTIONS:
                continue
            raw_batch = record.parameters.get("batch_id")
            source = record.parameters.get("source_folder")
            if raw_batch:
                folder = Path(str(source)).name if source else "imported batch"
                when = record.timestamp.astimezone()
                labels[str(raw_batch)] = f"{folder}, imported {when:%d %b %Y at %H:%M:%S}"
                order[str(raw_batch)] = when.timestamp()

        batches: list[tuple[float, str, int, UUID]] = []
        for raw_batch, count in counts.items():
            try:
                batch_id = UUID(raw_batch)
            except ValueError:
                continue
            batches.append(
                (
                    order.get(raw_batch, float("inf")),
                    labels.get(raw_batch, "imported batch"),
                    count,
                    batch_id,
                )
            )
        batches.sort(key=lambda item: item[:2])
        return [(label, count, batch_id) for _when, label, count, batch_id in batches]

    def merge_imported_relaxation_rows(
        self,
        outcome,
        *,
        output_directory: Path,
    ) -> bool:
        """Attach an import batch's rows to the muon-relaxation definition.

        The app-layer half of an import. ``services`` builds the entities and
        files but cannot finish the job: the relaxations table renders the
        *intersection* of the widget's form state and the active definition's
        rows, matched on a signature that includes normalised parameter values,
        so the rows have to be pushed through the live widget and read back.
        A services-only import would appear to succeed while every imported row
        rendered "ready" with no energy.

        The active definition is mutated **in place**. Re-registering it would
        arrive with an empty ``job_ids_by_label``, and the synchronisation pass
        that normally re-adopts those bindings only matches rows whose
        signatures already agree — which an import's never do — so every
        existing row would lose its job.

        Returns ``False`` when there is no document or no host structure to
        anchor a new definition to.
        """

        document = self._document
        if document is None:
            return False
        payloads = list(outcome.row_payloads)
        if not payloads:
            return True

        definition = self._muon_relaxation_controller.active_definition()
        if definition is None:
            host_id = document.project.host_structure_id
            if host_id is None:
                return False
            definition = self._project_service.register_muon_relaxation_definition(
                document,
                input_structure_id=host_id,
                parameters={"rows": []},
                output_directory=output_directory,
            )
            document.ui_state[self._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)

        configuration = definition.parameters.setdefault("configuration", {"rows": []})
        if not isinstance(configuration, dict):
            configuration = {"rows": []}
            definition.parameters["configuration"] = configuration
        rows = configuration.setdefault("rows", [])
        if not isinstance(rows, list):
            rows = []
            configuration["rows"] = rows

        mapping = ensure_job_mapping(definition)
        seed_mapping = definition.parameters.setdefault("job_seed_names_by_label", {})
        if not isinstance(seed_mapping, dict):
            seed_mapping = {}
            definition.parameters["job_seed_names_by_label"] = seed_mapping

        # Mirror first, then build the definition rows from what the widget gives
        # back. The row signature is computed over both sides, so anything the
        # widget normalises on ingest has to be what the definition stores too;
        # appending the raw payload here would leave the two describing the same
        # row differently, and a signature mismatch drops the imported row to a
        # fallback that only accepts jobs in an active status — rendering a
        # completed import as "ready" with no energy.
        normalized_rows = self._mirror_imported_rows_into_form(payloads)
        if normalized_rows is None:
            return False

        for item, payload in zip(outcome.imported, payloads, strict=False):
            row = normalized_rows.get(str(item.site_id))
            if row is None:
                # The widget did not take the row; adding it to the definition
                # anyway would create exactly the mismatch described above.
                continue
            rows.append(row)
            key = f"{muon_charge_state_prefix(item.charge_state)}:{item.label}"
            mapping[key] = str(item.job_id)
            seed_mapping[key] = item.seed_name
            if item.site_id not in definition.input_site_ids:
                definition.input_site_ids.append(item.site_id)

        return True

    def _mirror_imported_rows_into_form(
        self, payloads: list[dict[str, object]]
    ) -> dict[str, dict[str, object]] | None:
        """Add imported rows to the widget's form state and store the result.

        The widget's state shape is **not** the definition's configuration shape
        — it names the position ``fractional_position`` where the configuration
        says ``initial_fractional_position`` — so the payloads are translated
        rather than passed through. Feeding the configuration shape straight in
        parses without error and silently zeroes every position.

        Returns the widget's own normalised rows, in the *configuration* shape,
        keyed by site id — the caller stores these rather than the payloads it
        passed in, because the row signature is computed over them. Returns
        ``None`` when there is no document, so the caller can decline to touch
        the definition at all rather than leave it holding rows the form never
        received.
        """

        document = self._document
        if document is None:
            return None
        state = self._muon_relaxation_view.collect_state()
        rows = list(state.get("rows", [])) if isinstance(state, dict) else []
        for payload in payloads:
            entry = dict(payload)
            entry["fractional_position"] = list(
                entry.pop("initial_fractional_position", [0.0, 0.0, 0.0])
            )
            rows.append(entry)
        self._muon_relaxation_view.set_form_state({"rows": rows})
        normalized = self._muon_relaxation_view.collect_state()
        if not isinstance(normalized, dict):
            return None
        document.ui_state["muon_relaxation_form"] = normalized

        wanted = {str(payload.get("site_id", "")) for payload in payloads}
        by_site: dict[str, dict[str, object]] = {}
        for row in normalized.get("rows", []):
            if not isinstance(row, dict):
                continue
            site_id = str(row.get("site_id", ""))
            if site_id not in wanted:
                continue
            entry = dict(row)
            # Back into the configuration shape the definition stores.
            entry["initial_fractional_position"] = list(
                entry.pop("fractional_position", [0.0, 0.0, 0.0])
            )
            by_site[site_id] = entry
        return by_site

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

    def _muon_relaxation_form_config(self) -> MuonRelaxationConfig | None:
        if self._document is None:
            return None

        form_sources: list[dict] = []
        widget_state = self._muon_relaxation_view.collect_state()
        if isinstance(widget_state, dict):
            form_sources.append(widget_state)

        persisted = self._document.ui_state.get("muon_relaxation_form", {})
        if isinstance(persisted, dict):
            form_sources.append(persisted)

        for form_state in form_sources:
            config = deserialize_muon_relaxation_config(form_state)
            if config is not None and config.rows:
                return config
        return None

    @staticmethod
    def _muon_relaxation_row_key(row: MuonRelaxationRow) -> str:
        return f"{muon_charge_state_prefix(row.charge_state)}:{row.label}"

    def _muon_relaxation_definition_and_row_for_job(self, job: Job):
        if self._document is None:
            return None, None
        for definition in self._muon_relaxation_controller.definitions():
            config = deserialize_muon_relaxation_config(
                definition.parameters.get("configuration", {})
            )
            if config is None:
                continue
            for row in config.rows:
                candidate = self._muon_relaxation_controller.job_for_label(
                    definition, row.label, row.charge_state
                )
                if candidate is not None and candidate.id == job.id:
                    return definition, row
        return None, None

    def _refresh_muon_relaxation_rows(self) -> None:
        self._muon_relaxation_controller.refresh_view()
        self._refresh_relaxation_analysis_context()

    def _relaxation_analysis_workspaces(self) -> list[dict[str, object]]:
        if self._document is None:
            return []
        raw = self._document.ui_state.get(_RELAXATION_ANALYSIS_STATE_KEY, [])
        if not isinstance(raw, list):
            return []
        return [dict(workspace) for workspace in raw if isinstance(workspace, dict)]

    def _set_relaxation_analysis_workspaces(self, workspaces: list[dict[str, object]]) -> None:
        if self._document is None:
            return
        self._document.ui_state[_RELAXATION_ANALYSIS_STATE_KEY] = [
            dict(workspace) for workspace in workspaces
        ]

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
            workspaces=self._relaxation_analysis_workspaces(),
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
        definition = self._project_service.register_pristine_relaxation_definition(
            self._document,
            input_structure_id=host.id,
            parameters=serialize_pristine_relaxation_config(config),
            output_directory=output_root,
        )
        self._pristine_relaxation_controller.sync_mappings(definition)
        self._document.ui_state[self._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(
            definition.id
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
        definition = self._project_service.register_muon_relaxation_definition(
            self._document,
            input_structure_id=host.id,
            parameters=serialize_muon_relaxation_config(config),
            output_directory=output_root,
        )
        self._muon_relaxation_controller.sync_mappings(definition)
        self._document.ui_state[self._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
        self._refresh_muon_relaxation_rows()
        self._refresh_journal_view()
        self.statusBar().showMessage(
            f"Configured {len(config.rows)} muon-relaxation jobs; input files will be generated on submission"
        )

    def _on_configure_hyperfine_coupling_source(self, payload: object) -> None:
        controller = self._hyperfine_coupling_controller
        request = controller.configure_request(_ordered_string_list(payload))
        if request is None:
            return
        result = run_source_configure_dialog(
            partial(
                HyperfineCouplingConfigDialog,
                specs=controller.parameter_specs(),
                intro_text=request.intro_text,
                read_only=request.read_only,
                species_options=request.species_options,
                parent=self,
            ),
            request,
        )
        if result is not None:
            controller.apply_configuration(request.keys, result)

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

    def _on_uep_source_structure_selection_requested(self, payload: object) -> None:
        if self._document is None:
            return
        state = dict(self._document.ui_state.get("uep_form", {}))
        structure_id = str(payload).strip()
        if structure_id:
            state["selected_source_structure_id"] = structure_id
        else:
            state.pop("selected_source_structure_id", None)
        self._document.ui_state["uep_form"] = state
        self._refresh_uep_source_structures()
        self._refresh_workflow_defaults(self._uep_view)

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
            source_structure=lambda _document: self._selected_uep_source_structure(),
        )

    def _on_uep_configured(self, config: UepConfig) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        source_structure = self._selected_uep_source_structure()
        if source_structure is None:
            QMessageBox.warning(
                self,
                "No source structure",
                "Load a host structure or select a pristine-relaxed structure first.",
            )
            return
        try:
            expansion = expand_uep_jobs(
                host_structure=source_structure,
                config=config,
                project_cutoff_energy=self._document.project.recommended_cutoff_energy,
                project_kpoint_grid=self._document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=self._document.project.recommended_kpoint_lattice,
                project_xc_functional=self._document.project.recommended_xc_functional,
            )
        except (ValueError, StructureBridgeError) as exc:
            QMessageBox.warning(self, "UEP setup invalid", str(exc))
            return
        self._note_kpoint_adjustments(expansion)

        output_root = self._document.project_root / "files" / "uep"
        definition = self._project_service.register_uep_definition(
            self._document,
            input_structure_id=source_structure.id,
            parameters=serialize_uep_config(config),
            output_directory=output_root,
        )
        self._uep_controller.sync_mappings(definition)
        self._document.ui_state[self._ACTIVE_UEP_DEFINITION_KEY] = str(definition.id)
        self._refresh_uep_rows()
        self._refresh_journal_view()
        self.statusBar().showMessage(
            f"Configured {plural(len(config.rows), 'UEP source job')}; input files will be generated on submission"
        )

    def _on_open_uep_analysis_requested(self, label: object) -> None:
        if self._document is None:
            return
        row_label = str(label).strip()
        if not row_label:
            return

        definition = self._uep_controller.active_definition()
        if definition is None:
            QMessageBox.information(
                self, "No UEP workflow", "Configure and complete a UEP source job first."
            )
            return

        job = self._uep_controller.job_for_key(definition, row_label)
        if job is None or job.status != JobStatus.COMPLETED:
            QMessageBox.information(
                self,
                "UEP source incomplete",
                "Only completed UEP source jobs can be analysed.",
            )
            return

        output_dir = self._job_output_directory(job)
        if output_dir is None:
            QMessageBox.warning(
                self, "UEP files unavailable", "Could not locate the UEP job output directory."
            )
            return

        seed_name = self._job_seed_name(job)
        den_fmt_path = output_dir / f"{seed_name}.den_fmt"
        castep_path = output_dir / f"{seed_name}.castep"
        cell_path = output_dir / f"{seed_name}.cell"
        if not den_fmt_path.exists() or not castep_path.exists():
            QMessageBox.warning(
                self,
                "UEP files unavailable",
                "UEP analysis requires both `.den_fmt` and `.castep` files from the completed source job.",
            )
            return

        source_structure_id = (
            definition.input_structure_ids[0] if definition.input_structure_ids else None
        )
        if source_structure_id is None:
            QMessageBox.warning(
                self, "UEP source unavailable", "The source structure reference is missing."
            )
            return

        existing_dialog = self._uep_analysis_dialogs.get(job.id)
        if existing_dialog is not None:
            try:
                is_visible = getattr(existing_dialog, "isVisible", None)
                if callable(is_visible) and not is_visible():
                    self._drop_uep_analysis_dialog(job.id)
                else:
                    existing_dialog.show()
                    existing_dialog.raise_()
                    existing_dialog.activateWindow()
                    return
            except RuntimeError:
                self._drop_uep_analysis_dialog(job.id)

        try:
            persisted_states = self._document.ui_state.get(self._UEP_ANALYSIS_STATES_KEY, {})
            if not isinstance(persisted_states, dict):
                persisted_states = {}
            initial_state = persisted_states.get(str(job.id), {})
            if not isinstance(initial_state, dict):
                initial_state = {}
            initial_state = self._migrate_legacy_uep_state_potential_grid(job.id, initial_state)
            dialog = UepAnalysisDialog(
                source_job_label=job.label,
                source_structure_id=source_structure_id,
                source_structure=self._document.structures.get(source_structure_id),
                den_fmt_path=den_fmt_path,
                castep_path=castep_path,
                cell_path=cell_path if cell_path.exists() else None,
                initial_state=initial_state,
                parent=self,
            )
        except (UepError, OSError, ValueError) as exc:
            QMessageBox.warning(self, "UEP analysis unavailable", str(exc))
            return

        dialog.state_changed.connect(
            lambda state, job_id=job.id: self._on_uep_analysis_state_changed(job_id, state)
        )
        dialog.setModal(False)
        dialog.destroyed.connect(
            lambda *_args, job_id=job.id: self._drop_uep_analysis_dialog(job_id)
        )
        self._uep_analysis_dialogs[job.id] = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _migrate_legacy_uep_state_potential_grid(self, job_id: UUID, state: dict) -> dict:
        """Move a potential grid saved inside the UEP UI state into its own result.

        Older snapshots repeated the grid in ``ui_state`` on every change. The
        wizard no longer writes it there, so the next state change would drop
        the only copy for a project that never stored the grid result.
        """

        if self._document is None or "potential_grid" not in state:
            return state
        if self._uep_potential_grid_result_for_job(job_id) is None:
            self._on_uep_potential_grid_ready(job_id, state.get("potential_grid"))
            if self._uep_potential_grid_result_for_job(job_id) is None:
                return state
        migrated = {key: value for key, value in state.items() if key != "potential_grid"}
        states = self._document.ui_state.get(self._UEP_ANALYSIS_STATES_KEY, {})
        if isinstance(states, dict) and str(job_id) in states:
            self._document.ui_state[self._UEP_ANALYSIS_STATES_KEY] = {
                **states,
                str(job_id): migrated,
            }
        return migrated

    def _on_uep_potential_grid_ready(self, job_id: UUID, payload: object) -> None:
        if self._document is None:
            return
        grid = deserialize_potential_grid(payload)
        if grid is None:
            return
        run = self._run_for_job(job_id)
        if run is None:
            return

        serialized = serialize_potential_grid(grid)
        existing = self._uep_potential_grid_result_for_job(job_id)
        if existing is None:
            existing = Result(
                calculation_run_id=run.id,
                result_type=self._UEP_POTENTIAL_GRID_RESULT_TYPE,
                data=serialized,
            )
        else:
            existing.data = serialized
        self._document.results[existing.id] = existing
        self._queue_raise_visible_uep_analysis_dialog(job_id)

    @staticmethod
    def _uep_analysis_state_affects_muon_generation(previous: object, updated: dict) -> bool:
        if not isinstance(previous, dict):
            previous = {}
        return previous.get("batch_results") != updated.get("batch_results") or previous.get(
            "batch_summary_data"
        ) != updated.get("batch_summary_data")

    @staticmethod
    def _raise_existing_uep_analysis_dialog(dialog: object) -> None:
        try:
            is_visible = getattr(dialog, "isVisible", None)
            if callable(is_visible) and not is_visible():
                return
            show = getattr(dialog, "show", None)
            if callable(show):
                show()
            raise_window = getattr(dialog, "raise_", None)
            if callable(raise_window):
                raise_window()
            activate = getattr(dialog, "activateWindow", None)
            if callable(activate):
                activate()
        except RuntimeError:
            return

    def _queue_raise_visible_uep_analysis_dialog(self, job_id: UUID) -> None:
        dialog = self._uep_analysis_dialogs.get(job_id)
        if dialog is None:
            return
        QTimer.singleShot(0, lambda dialog=dialog: self._raise_existing_uep_analysis_dialog(dialog))

    def _on_uep_analysis_state_changed(self, job_id: UUID, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return

        serialized_state = to_jsonable(state)
        if not isinstance(serialized_state, dict):
            return

        states = self._document.ui_state.get(self._UEP_ANALYSIS_STATES_KEY, {})
        if not isinstance(states, dict):
            states = {}
        updated = dict(states)
        previous_state = updated.get(str(job_id), {})
        updated[str(job_id)] = serialized_state
        self._document.ui_state[self._UEP_ANALYSIS_STATES_KEY] = updated
        job = self._document.jobs.get(job_id)
        if job is not None and record_uep_minima(
            self._document, job, previous_state, serialized_state
        ):
            self._refresh_journal_view()
        if self._uep_analysis_state_affects_muon_generation(previous_state, serialized_state):
            QTimer.singleShot(0, self._refresh_muon_generation_table)
        self._queue_raise_visible_uep_analysis_dialog(job_id)

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
        self._muon_generation_view.set_uep_cluster_sources(self._uep_cluster_source_rows())
        self._refresh_workflow_stage_icons()
        self._sync_project_home()

    def _refresh_host_structure_view(self, *, force: bool = False) -> None:
        if self._document is None:
            self._host_structure_view.set_structure(None)
            self._host_structure_view.set_promotable_hosts([], None)
            self._refresh_magnetic_field_view()
            self._refresh_active_host_badges()
            return

        host_id = self._document.project.host_structure_id
        host: Structure | None = self._document.structures.get(host_id) if host_id else None
        promotable = self._project_service.list_promotable_host_structures(self._document)
        self._host_structure_view.set_promotable_hosts(promotable, host_id)
        self._host_structure_view.set_structure(host, force=force)
        self._refresh_magnetic_field_view()
        self._refresh_active_host_badges()

    def _refresh_active_host_badges(self) -> None:
        """Update all embedded active-host badges in workflow tabs."""
        host: Structure | None = None
        if self._document is not None:
            host_id = self._document.project.host_structure_id
            host = self._document.structures.get(host_id) if host_id else None
        self._muon_generation_view.set_active_host(host)
        self._muon_relaxation_view.set_active_host(host)

    def _on_active_host_change_requested(self, structure_id_str: str) -> None:
        if self._document is None:
            return
        if self._muon_generation_blocks_host_change():
            self._refresh_host_structure_view()
            return
        try:
            structure_id = UUID(structure_id_str)
        except (ValueError, AttributeError):
            return
        try:
            self._project_service.set_active_host_structure(self._document, structure_id)
        except ValueError:
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
                and not self._muon_generation_blocks_host_change()
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
                and not self._muon_generation_blocks_host_change()
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

        host_id = self._document.project.host_structure_id
        host = self._document.structures.get(host_id) if host_id is not None else None
        if host is None:
            return

        raw_spins = payload.get("atomic_spins", [])
        if isinstance(raw_spins, list):
            host.properties["atomic_spins"] = [float(value) for value in raw_spins]

        raw_propagation = payload.get("propagation_vector", [0.0, 0.0, 0.0])
        if isinstance(raw_propagation, list) and len(raw_propagation) == 3:
            propagation = [
                float(raw_propagation[0]),
                float(raw_propagation[1]),
                float(raw_propagation[2]),
            ]
            host.properties["propagation_vector"] = propagation
        else:
            propagation = [0.0, 0.0, 0.0]

        magnetic_structure = payload.get("magnetic_structure")
        if isinstance(magnetic_structure, dict):
            host.properties["magnetic_structure"] = magnetic_structure
            self._document.project.magnetic_structure = magnetic_structure
        else:
            host.properties.pop("magnetic_structure", None)
            if any(abs(value) > 1e-12 for value in propagation):
                self._document.project.magnetic_structure = {
                    "propagation_vector": propagation,
                    "moments": [],
                }
            else:
                self._document.project.magnetic_structure = None

        self._refresh_magnetic_field_view()
        self._refresh_workflow_stage_icons()

    def _on_magnetic_field_structure_changed(self, payload: object) -> None:
        if self._document is None or not isinstance(payload, dict):
            return
        self._document.project.magnetic_structure = payload
        host_id = self._document.project.host_structure_id
        host = self._document.structures.get(host_id) if host_id is not None else None
        if host is not None:
            host.properties["magnetic_structure"] = payload
        self._refresh_workflow_stage_icons()

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
        self._refresh_workflow_stage_icons()
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
        self._refresh_workflow_stage_icons()

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
        self._populate_project_tree()
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
        self._populate_project_tree()
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
        self._populate_project_tree()
        route = self._active_route()
        if route is None or self._magnetic_field_workspace_id_from_route(route) == workspace_id:
            self._set_active_route("magnetic_fields")
        self._persist_document_ui_state(
            failure_message="Magnetic field workspace was deleted, but autosave failed."
        )
        self.statusBar().showMessage(f"Deleted magnetic field workspace '{title}'")

    def _detach_convergence_job(self, job_id: UUID) -> None:
        if self._document is None:
            return
        for definition in self._convergence_controller.definitions():
            mapping = definition.parameters.get("job_ids_by_cutoff", {})
            if not isinstance(mapping, dict):
                continue
            stale_keys = [key for key, value in mapping.items() if str(value) == str(job_id)]
            for key in stale_keys:
                mapping.pop(key, None)
