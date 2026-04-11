"""Primary application window for MuFinder."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from math import isnan
from pathlib import Path
import re
import shutil
from uuid import UUID

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QActionGroup, QColor, QFont, QGuiApplication, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QFileDialog,
    QDockWidget,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.castep_setup import load_castep_setup
from mufinder.branding import app_icon, logo_path
from mufinder.cluster_setup import ClusterConfig, load_cluster_config, load_cluster_configs
from mufinder.castep_io.gui_metadata import normalize_gui_value
from mufinder.castep_io.schema import get_keyword_spec
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, ComparisonTable, Job, Result, Structure, utc_now
from mufinder.app.cluster_setup_dialog import ClusterSetupDialog
from mufinder.app.cluster_submission_dialog import ClusterSubmissionConfig, ClusterSubmissionDialog
from mufinder.app.comparison_job_dialog import ComparisonJobDialog, CompletedPristineJob
from mufinder.app.comparison_table_widget import ComparisonTableWidget
from mufinder.app.cluster_ui_utils import (
    format_cluster_connection_error,
    prompt_to_trust_host_key,
    show_detailed_warning,
)
from mufinder.app.castep_setup_dialog import CastepSetupDialog
from mufinder.app.convergence_adapter import convergence_job_name, expand_convergence_jobs
from mufinder.app.convergence_examiner import ConvergenceExaminerDialog
from mufinder.app.convergence_display import (
    format_convergence_criteria,
    format_convergence_method,
    format_cutoff_energy,
    format_kpoint_grid,
)
from mufinder.app.convergence_overview_widget import ConvergenceOverviewWidget
from mufinder.app.convergence_types import ConvergenceConfig, SweepPointRunState
from mufinder.app.convergence_widget import ConvergenceWidget
from mufinder.app.kpoint_convergence_widget import KPointConvergenceWidget
from mufinder.app.relaxation_adapter import expand_pristine_relaxation_jobs
from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow, RelaxationRunState
from mufinder.app.relaxation_widget import PristineRelaxationWidget
from mufinder.app.host_structure_widget import HostStructureWidget
from mufinder.app.icons import icon, workflow_stage_icon
from mufinder.app.muon_generation_widget import MuonGenerationWidget
from mufinder.app.supercell_dialog import SupercellPreviewDialog
from mufinder.app.submission_dialog import LocalSubmissionDialog
from mufinder.app.theme import (
    THEME_DARK,
    THEME_LIGHT,
    THEME_SYSTEM,
    apply_theme,
    body_font_family,
    current_palette,
    load_theme_mode,
    save_theme_mode,
    status_brush,
    use_square_top_table_corners,
)
from mufinder.castep_io.cell_parser import CellParseError, parse_cell_file
from mufinder.castep_io.output_parser import CastepResult
from mufinder.castep_io.types import BlockSection, ValueWithUnit
from mufinder.persistence.codec import to_jsonable
from mufinder.services.cluster_runner import ClusterJobRunner
from mufinder.services.local_runner import LocalJobRunner
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.ssh_client import SSHUnknownHostKeyError, save_trusted_host_key
from mufinder.science.kpoint_grids import lattice_parameters
from mufinder.science.site_generation import SiteGenerationError
from mufinder.science.structure_comparison import DftComparisonInput, compare_structures
from mufinder.science.structure_io import StructureLoadError
from mufinder.workflow import StructureBridgeError, structure_to_spec, write_job_directory


_ACTIVE_LOCAL_JOB_STATES = {
    JobStatus.SUBMITTING,
    JobStatus.QUEUED,
    JobStatus.RUNNING,
    JobStatus.RETRIEVING,
}
_VISIBLE_DASHBOARD_STATES = _ACTIVE_LOCAL_JOB_STATES | {JobStatus.STOPPED, JobStatus.FAILED}

# Backward-compatible alias used by legacy tests and call sites.
CutoffRunState = SweepPointRunState


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    """Describe a workflow node and its completion rule."""

    key: str
    title: str
    subtitle: str = ""
    route: str | None = None
    completion_goal: str | None = None
    completion_check: Callable[[ProjectDocument], bool] | None = None
    children: tuple["WorkflowDefinition", ...] = ()


class ExpeditionTrailWidget(QWidget):
    """Paint a compact workflow trail with completion markers."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stages: list[tuple[str, str]] = []
        self.setMinimumHeight(136)

    def set_stages(self, stages: list[tuple[str, str]]) -> None:
        self._stages = list(stages)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().paintEvent(event)
        if not self._stages:
            return

        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        width = max(1, self.width())
        margin = 20
        center_y = 44
        count = len(self._stages)
        usable = max(1, width - (margin * 2))
        step = usable / max(1, count - 1)
        compact = step < 105
        alternate = step >= 88
        label_width = int(max(62 if compact else 84, min(122, step - (8 if compact else 12) if count > 1 else 122)))
        compact_titles = {
            "Host Structure": "Host",
            "Convergence": "Conv.",
            "Muon Sites": "Muon",
            "Results": "Results",
        }

        if count > 1:
            painter.setPen(QPen(QColor(palette.border), 1.2, Qt.PenStyle.DotLine))
            painter.drawLine(margin, center_y, width - margin, center_y)

        for index, (title, state) in enumerate(self._stages):
            x = int(margin + (step * index if count > 1 else usable / 2))

            if state == "completed":
                painter.setPen(QPen(QColor(palette.border), 1.1))
                painter.setBrush(QColor(palette.seal_gold))
                painter.drawEllipse(x - 8, center_y - 8, 16, 16)
                painter.setPen(
                    QPen(
                        QColor(palette.bg_surface),
                        1.9,
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.drawLine(x - 3, center_y, x - 1, center_y + 3)
                painter.drawLine(x - 1, center_y + 3, x + 4, center_y - 3)
            elif state == "active":
                painter.setPen(QPen(QColor(palette.accent), 1.8))
                painter.setBrush(QColor(palette.bg_surface))
                painter.drawEllipse(x - 8, center_y - 8, 16, 16)
                painter.setPen(
                    QPen(
                        QColor(palette.accent),
                        1.8,
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                # Anchor symbol for active stage.
                painter.drawLine(x, center_y - 4, x, center_y + 4)
                painter.drawLine(x - 2, center_y - 3, x + 2, center_y - 3)
                painter.drawLine(x - 5, center_y + 3, x - 2, center_y)
                painter.drawLine(x + 5, center_y + 3, x + 2, center_y)
                painter.drawArc(x - 5, center_y - 1, 10, 10, 180 * 16, 180 * 16)
            else:
                painter.setPen(
                    QPen(
                        QColor(palette.text_muted),
                        1.4,
                        Qt.PenStyle.DotLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.setBrush(QColor(palette.bg_surface))
                painter.drawEllipse(x - 7, center_y - 7, 14, 14)

            label_font = QFont(body_font_family())
            label_font.setPointSize(9 if compact else 10)
            painter.setFont(label_font)
            painter.setPen(
                QColor(palette.text_primary if state in {"completed", "active"} else palette.text_muted)
            )

            label_y = center_y + 14
            if alternate:
                label_y = center_y + 14 if index % 2 == 0 else center_y - 48
            label_x = max(4, min(x - (label_width // 2), width - label_width - 4))
            label_text = compact_titles.get(title, title) if compact else title
            painter.drawText(
                label_x,
                label_y,
                label_width,
                36,
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                label_text,
            )

        painter.end()


class MainWindow(QMainWindow):
    """Main shell containing project navigation and job dashboard."""

    _LAST_DIALOG_DIR_KEY = "ui/last_dialog_dir"
    _LAST_DIALOG_DIR_BY_CONTEXT_PREFIX = "ui/last_dialog_dir_by_context"
    _RECENT_PROJECTS_KEY = "ui/recent_projects"
    _MAX_RECENT_PROJECTS = 10
    _JOB_LABEL_COLUMN = 0
    _JOB_WORKFLOW_COLUMN = 1
    _JOB_CALC_TYPE_COLUMN = 2
    _JOB_STATUS_COLUMN = 3
    _JOB_CLUSTER_COLUMN = 4
    _JOB_REMOTE_ID_COLUMN = 5
    _JOB_RUNTIME_COLUMN = 6
    _JOB_PROCESSES_COLUMN = 7
    _JOB_SUBMIT_COLUMN = 8
    _JOB_CANCEL_COLUMN = 9
    _JOB_INSPECT_COLUMN = 10
    _JOB_RESET_COLUMN = 11
    _ACTIVE_CONVERGENCE_DEFINITION_KEY = "active_convergence_cutoff_definition_id"
    _ACTIVE_KPOINT_DEFINITION_KEY = "active_convergence_kpoint_definition_id"
    _ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY = "active_pristine_relaxation_definition_id"
    _CONVERGENCE_EXAMINER_STATE_KEY = "convergence_examiner_state"
    _KPOINT_EXAMINER_STATE_KEY = "kpoint_examiner_state"
    _CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY = "convergence_overview_snapshots"
    _LAST_CLUSTER_SUBMISSION_CONFIG_KEY = "last_cluster_submission_config"
    _COMPARISON_ROUTE_PREFIX = "pristine_relaxation/comparison/"
    _WORKFLOW_COMPLETE_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    _WORKFLOW_GOAL_ROLE = int(Qt.ItemDataRole.UserRole) + 2

    def __init__(self) -> None:
        super().__init__()
        self._project_service = ProjectService()
        self._document: ProjectDocument | None = None
        self._settings = QSettings("MuFinder", "MuFinder")
        self._appearance_mode = load_theme_mode(self._settings)
        self._local_runner = LocalJobRunner(self)
        self._cluster_runner = ClusterJobRunner(self, settings=self._settings)
        self._active_run_by_job: dict[UUID, UUID] = {}
        self._recent_projects_menu: QMenu | None = None
        self._home_recent_projects_layout: QVBoxLayout | None = None
        self._expedition_trail: ExpeditionTrailWidget | None = None
        self._cluster_connected = False
        self._comparison_widgets: dict[UUID, ComparisonTableWidget] = {}

        app = QApplication.instance()
        if app is not None:
            apply_theme(app, self._appearance_mode)

        self.setWindowTitle("MuFinder")
        self.setWindowIcon(app_icon())
        self.resize(1280, 840)

        self._project_tree = QTreeWidget()
        self._project_tree.setObjectName("workflowTree")
        self._project_tree.setHeaderLabel("Project")
        self._project_tree.setHeaderHidden(True)
        self._project_tree.itemSelectionChanged.connect(self._on_tree_selection_changed)

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
        self._convergence_overview_view = ConvergenceOverviewWidget()
        self._convergence_overview_view.open_cutoff_requested.connect(
            lambda: self._set_active_route("convergence_cutoff")
        )
        self._convergence_overview_view.open_kpoint_requested.connect(
            lambda: self._set_active_route("convergence_kpoint")
        )
        self._convergence_view = ConvergenceWidget()
        self._convergence_view.configure_requested.connect(self._on_convergence_configured)
        self._convergence_view.examine_requested.connect(self._on_examine_convergence)
        self._convergence_view.form_state_changed.connect(self._on_convergence_form_state_changed)
        self._convergence_view.simulation_cell_create_requested.connect(
            self._on_convergence_simulation_cell_create_requested
        )
        self._convergence_view.simulation_cell_selection_requested.connect(
            self._on_convergence_simulation_cell_selection_requested
        )
        self._convergence_view.simulation_cell_use_unit_cell_requested.connect(
            self._on_convergence_simulation_cell_use_unit_cell_requested
        )
        self._convergence_view.submit_local_requested.connect(self._on_submit_local_from_convergence)
        self._convergence_view.submit_cluster_requested.connect(self._on_submit_cluster_from_convergence)
        self._convergence_view.cancel_requested.connect(self._on_cancel_from_convergence)
        self._convergence_view.inspect_failed_requested.connect(self._on_inspect_failed_from_convergence)
        self._convergence_view.reset_failed_requested.connect(self._on_reset_failed_from_convergence)
        self._convergence_view.remove_cutoff_requested.connect(self._on_remove_cutoffs_from_convergence)
        self._kpoint_convergence_view = KPointConvergenceWidget()
        self._kpoint_convergence_view.configure_requested.connect(self._on_kpoint_convergence_configured)
        self._kpoint_convergence_view.examine_requested.connect(self._on_examine_kpoint_convergence)
        self._kpoint_convergence_view.form_state_changed.connect(self._on_kpoint_form_state_changed)
        self._kpoint_convergence_view.simulation_cell_create_requested.connect(
            self._on_kpoint_simulation_cell_create_requested
        )
        self._kpoint_convergence_view.simulation_cell_selection_requested.connect(
            self._on_kpoint_simulation_cell_selection_requested
        )
        self._kpoint_convergence_view.simulation_cell_use_unit_cell_requested.connect(
            self._on_kpoint_simulation_cell_use_unit_cell_requested
        )
        self._kpoint_convergence_view.submit_local_requested.connect(
            self._on_submit_local_from_kpoint_convergence
        )
        self._kpoint_convergence_view.submit_cluster_requested.connect(
            self._on_submit_cluster_from_kpoint_convergence
        )
        self._kpoint_convergence_view.cancel_requested.connect(self._on_cancel_from_kpoint_convergence)
        self._kpoint_convergence_view.inspect_failed_requested.connect(
            self._on_inspect_failed_from_kpoint_convergence
        )
        self._kpoint_convergence_view.reset_failed_requested.connect(
            self._on_reset_failed_from_kpoint_convergence
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
        self._pristine_relaxation_view.submit_local_requested.connect(
            self._on_submit_local_from_pristine_relaxation
        )
        self._pristine_relaxation_view.submit_cluster_requested.connect(
            self._on_submit_cluster_from_pristine_relaxation
        )
        self._pristine_relaxation_view.generate_comparison_requested.connect(
            self._on_generate_comparison_table_requested
        )
        self._muon_generation_view = MuonGenerationWidget()
        self._muon_generation_view.generate_requested.connect(self._on_generate_requested)
        self._muon_generation_view.form_state_changed.connect(self._on_muon_form_state_changed)
        self._main_stack.addWidget(self._home_page)
        self._main_stack.addWidget(self._host_structure_view)
        self._main_stack.addWidget(self._convergence_overview_view)
        self._main_stack.addWidget(self._convergence_view)
        self._main_stack.addWidget(self._kpoint_convergence_view)
        self._main_stack.addWidget(self._pristine_relaxation_view)
        self._main_stack.addWidget(self._muon_generation_view)

        self._jobs_table = QTableWidget(0, 12)
        self._jobs_table.setHorizontalHeaderLabels(
            [
                "Label",
                "Workflow",
                "Calc Type",
                "Status",
                "Cluster",
                "Remote Job ID",
                "Runtime",
                "Processes",
                "Submit",
                "Cancel",
                "Inspect",
                "Reset",
            ]
        )
        use_square_top_table_corners(self._jobs_table)
        self._jobs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._jobs_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._jobs_table.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self._configure_jobs_table_columns()

        self._local_runner.job_started.connect(self._on_local_job_started)
        self._local_runner.job_finished.connect(self._on_local_job_finished)
        self._local_runner.job_failed.connect(self._on_local_job_failed)
        self._local_runner.job_cancelled.connect(self._on_local_job_cancelled)
        self._local_runner.status_changed.connect(self._on_local_job_status_changed)

        self._cluster_runner.job_started.connect(self._on_cluster_job_started)
        self._cluster_runner.job_finished.connect(self._on_cluster_job_finished)
        self._cluster_runner.job_failed.connect(self._on_cluster_job_failed)
        self._cluster_runner.job_cancelled.connect(self._on_cluster_job_cancelled)
        self._cluster_runner.status_changed.connect(self._on_cluster_job_status_changed)
        self._cluster_runner.connection_status_changed.connect(self._on_cluster_connection_status_changed)
        self._cluster_runner.queue_checked.connect(self._on_cluster_queue_checked)
        self._cluster_runner.poll_failed.connect(self._on_cluster_poll_failed)

        self._build_layout()
        self._build_menu()
        self.setStatusBar(QStatusBar(self))
        self._cluster_connection_label = QLabel("Cluster: disconnected", self)
        self._cluster_queue_checked_label = QLabel("Queue last checked: never", self)
        self._cluster_refresh_button = QPushButton("Refresh Queue", self)
        self._cluster_refresh_button.setEnabled(True)
        self._cluster_refresh_button.clicked.connect(self._on_refresh_cluster_queue)
        self.statusBar().addPermanentWidget(self._cluster_connection_label)
        self.statusBar().addPermanentWidget(self._cluster_queue_checked_label)
        self.statusBar().addPermanentWidget(self._cluster_refresh_button)
        self._refresh_cluster_connection_label_colour()
        self.statusBar().showMessage("Ready")

    def _build_layout(self) -> None:
        central = QWidget()
        central_layout = QHBoxLayout(central)

        self._project_tree_panel = QFrame(central)
        self._project_tree_panel.setObjectName("projectTreePanel")
        self._project_tree_panel.setMinimumWidth(280)
        project_tree_layout = QVBoxLayout(self._project_tree_panel)
        project_tree_layout.setContentsMargins(0, 0, 0, 0)
        project_tree_layout.setSpacing(0)

        self._project_tree_title = QLabel("Project", self._project_tree_panel)
        self._project_tree_title.setObjectName("projectTreePanelTitle")
        self._project_tree_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        project_tree_layout.addWidget(self._project_tree_title)

        self._project_tree_body = QFrame(self._project_tree_panel)
        self._project_tree_body.setObjectName("projectTreePanelBody")
        project_tree_body_layout = QVBoxLayout(self._project_tree_body)
        project_tree_body_layout.setContentsMargins(1, 0, 1, 1)
        project_tree_body_layout.setSpacing(0)
        project_tree_body_layout.addWidget(self._project_tree, 1)
        project_tree_layout.addWidget(self._project_tree_body, 1)

        central_layout.addWidget(self._project_tree_panel, 1)

        central_layout.addWidget(self._main_stack, 3)
        self.setCentralWidget(central)

        jobs_dock = QDockWidget("Job Dashboard", self)
        jobs_dock.setObjectName("jobDashboardDock")
        jobs_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        jobs_dock.setWidget(self._jobs_table)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, jobs_dock)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if self._document is not None and self.isVisible():
            answer = QMessageBox.question(
                self,
                "Save project?",
                "Save the current project before closing?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Yes:
                try:
                    self._project_service.save_project(self._document)
                except ValueError as exc:
                    QMessageBox.critical(self, "Save blocked", str(exc))
                    event.ignore()
                    return
        self._local_runner.shutdown()
        self._cluster_runner.shutdown()
        super().closeEvent(event)

    def _make_home_card(self, title: str, body: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame(self)
        card.setObjectName("homeCard")
        layout = QVBoxLayout(card)

        title_label = QLabel(title, card)
        title_label.setObjectName("cardTitle")
        body_label = QLabel(body, card)
        body_label.setObjectName("cardBody")
        body_label.setWordWrap(True)

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

        open_card, open_layout = self._make_home_card(
            "Open Project",
            "Continue from an existing .mufinder folder.",
        )
        open_button = QPushButton("Open Existing Project", open_card)
        open_button.setIcon(icon("open_project"))
        open_button.clicked.connect(self._open_project)
        open_layout.addWidget(open_button)

        demo_card, demo_layout = self._make_home_card(
            "Open Demo",
            "Use a demo job to explore the interface before running real calculations.",
        )
        demo_button = QPushButton("Add Demo Job", demo_card)
        demo_button.setIcon(icon("demo"))
        demo_button.clicked.connect(self._add_demo_job)
        demo_layout.addWidget(demo_button)

        actions_grid.addWidget(new_card, 0, 0)
        actions_grid.addWidget(open_card, 0, 1)
        actions_grid.addWidget(demo_card, 0, 2)
        layout.addLayout(actions_grid)

        lower_grid = QGridLayout()
        lower_grid.setHorizontalSpacing(12)
        lower_grid.setVerticalSpacing(12)

        recent_card, recent_layout = self._make_home_card(
            "Recent Projects",
            "Jump back into your recent work.",
        )
        self._home_recent_projects_layout = QVBoxLayout()
        recent_layout.addLayout(self._home_recent_projects_layout)
        lower_grid.addWidget(recent_card, 0, 0)

        guide_card, guide_layout = self._make_home_card(
            "Expedition Trail",
            "Track your journey through each workflow milestone.",
        )
        title = QLabel("Notebook Progress", guide_card)
        title.setObjectName("expeditionTrailTitle")
        guide_layout.addWidget(title)

        hint = QLabel(
            "Seals mark completed milestones, open rings mark pending milestones.",
            guide_card,
        )
        hint.setObjectName("expeditionTrailHint")
        hint.setWordWrap(True)
        guide_layout.addWidget(hint)

        self._expedition_trail = ExpeditionTrailWidget(guide_card)
        guide_layout.addWidget(self._expedition_trail)

        legend = QHBoxLayout()
        complete = QLabel("Completed", guide_card)
        complete.setObjectName("expeditionLegendComplete")
        pending = QLabel("Pending", guide_card)
        pending.setObjectName("expeditionLegendPending")
        legend.addWidget(complete)
        legend.addWidget(pending)
        legend.addStretch(1)
        guide_layout.addLayout(legend)
        lower_grid.addWidget(guide_card, 0, 1)

        layout.addLayout(lower_grid)
        self._refresh_home_recent_projects()
        self._sync_expedition_trail()
        layout.addStretch(1)
        return page

    def _refresh_home_recent_projects(self) -> None:
        if self._home_recent_projects_layout is None:
            return

        while self._home_recent_projects_layout.count():
            item = self._home_recent_projects_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        recent = self._recent_projects()[:5]
        if not recent:
            empty = QLabel("No recent projects yet.")
            empty.setObjectName("cardBody")
            self._home_recent_projects_layout.addWidget(empty)
            return

        for project_path in recent:
            button = QPushButton(project_path, self)
            button.setIcon(icon("open_project"))
            button.clicked.connect(
                lambda _checked=False, project_path=project_path: self._open_recent_project(project_path)
            )
            self._home_recent_projects_layout.addWidget(button)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        new_action = file_menu.addAction(icon("new_project"), "New Project")
        new_action.triggered.connect(self._new_project)

        open_action = file_menu.addAction(icon("open_project"), "Open Project")
        open_action.triggered.connect(self._open_project)

        self._recent_projects_menu = file_menu.addMenu("Open Recent")
        self._refresh_recent_projects_menu()

        save_action = file_menu.addAction(icon("save_project"), "Save Project")
        save_action.triggered.connect(self._save_project)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        setup_menu = self.menuBar().addMenu("Setup")
        castep_action = setup_menu.addAction(icon("settings"), "CASTEP...")
        castep_action.triggered.connect(self._open_castep_setup_dialog)
        cluster_action = setup_menu.addAction(icon("cluster"), "HPC Clusters...")
        cluster_action.triggered.connect(self._open_cluster_setup_dialog)

        view_menu = self.menuBar().addMenu("View")
        appearance_menu = view_menu.addMenu("Appearance")
        appearance_group = QActionGroup(self)
        appearance_group.setExclusive(True)

        light_action = appearance_menu.addAction(icon("appearance_light"), "Light")
        light_action.setCheckable(True)
        light_action.triggered.connect(lambda _checked=False: self._set_appearance_mode(THEME_LIGHT))
        appearance_group.addAction(light_action)

        dark_action = appearance_menu.addAction(icon("appearance_dark"), "Dark")
        dark_action.setCheckable(True)
        dark_action.triggered.connect(lambda _checked=False: self._set_appearance_mode(THEME_DARK))
        appearance_group.addAction(dark_action)

        system_action = appearance_menu.addAction(icon("appearance_system"), "System")
        system_action.setCheckable(True)
        system_action.triggered.connect(lambda _checked=False: self._set_appearance_mode(THEME_SYSTEM))
        appearance_group.addAction(system_action)

        if self._appearance_mode == THEME_DARK:
            dark_action.setChecked(True)
        elif self._appearance_mode == THEME_LIGHT:
            light_action.setChecked(True)
        else:
            system_action.setChecked(True)

    def _set_appearance_mode(self, mode: str) -> None:
        self._appearance_mode = mode
        save_theme_mode(self._settings, mode)
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, mode)
        self._refresh_cluster_connection_label_colour()
        self._refresh_workflow_stage_icons()
        self._sync_expedition_trail()

    def _new_project(self) -> None:
        project_name, ok = QInputDialog.getText(self, "New Project", "Project name:")
        if not ok or not project_name.strip():
            return

        base_dir = self._get_existing_directory(
            title="Select parent folder",
            context="new_project_parent",
        )
        if not base_dir:
            return

        project_root = Path(base_dir) / f"{project_name.strip()}.mufinder"
        if project_root.exists() and any(project_root.iterdir()):
            QMessageBox.warning(
                self,
                "Project exists",
                "Target project folder already exists and is not empty.",
            )
            return

        try:
            self._document = self._project_service.create_project(project_root, project_name.strip())
        except ValueError as exc:
            QMessageBox.critical(self, "Project error", str(exc))
            return

        self._record_recent_project(project_root)
        self._refresh_ui_after_project_change()
        self.statusBar().showMessage(f"Created project at {project_root}")

    def _open_project(self) -> None:
        project_dir = self._get_existing_directory(
            title="Select .mufinder project folder",
            context="open_project",
        )
        if not project_dir:
            return

        self._open_project_from_path(Path(project_dir))

    def _open_project_from_path(self, project_root: Path, *, show_errors: bool = True) -> bool:
        manifest = project_root / "project.json"
        if not manifest.exists():
            if show_errors:
                QMessageBox.critical(
                    self,
                    "Invalid project",
                    "Selected folder does not contain project.json.",
                )
            return False

        try:
            self._document = self._project_service.load_project(project_root)
        except Exception as exc:  # noqa: BLE001 - surface project-load failures to users.
            if show_errors:
                QMessageBox.critical(self, "Open failed", str(exc))
            return False

        self._record_recent_project(project_root)
        self._remember_dialog_path(project_root, "open_project")
        self._refresh_ui_after_project_change()

        integrity_warnings = self._document.ui_state.get("_integrity_warnings", [])
        if integrity_warnings:
            QMessageBox.warning(
                self,
                "Integrity warnings",
                "Project loaded with integrity warnings:\n\n"
                + "\n".join(f"- {warning}" for warning in integrity_warnings),
            )
        self.statusBar().showMessage(f"Opened project {self._document.project.name}")
        return True

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
                lambda _checked=False, project_path=project_path: self._open_recent_project(project_path)
            )

        menu.addSeparator()
        clear_action = menu.addAction("Clear Recent Projects")
        clear_action.triggered.connect(self._clear_recent_projects)

    def _save_project(self) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return

        try:
            self._project_service.save_project(self._document)
        except ValueError as exc:
            QMessageBox.critical(self, "Save blocked", str(exc))
            return
        self.statusBar().showMessage("Project saved")

    def _open_castep_setup_dialog(self) -> None:
        dialog = CastepSetupDialog(settings=self._settings, parent=self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        setup = dialog.setup()
        status_bits: list[str] = []
        if setup.executable_path:
            status_bits.append("executable configured")
        if setup.keywords_json_path:
            status_bits.append("keywords JSON configured")
        if setup.mpirun_path:
            status_bits.append("mpirun configured")
        detail = ", ".join(status_bits) if status_bits else "paths cleared"
        self.statusBar().showMessage(f"CASTEP setup updated ({detail})")

    def _open_cluster_setup_dialog(self) -> None:
        dialog = ClusterSetupDialog(settings=self._settings, parent=self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self.statusBar().showMessage("HPC cluster setup updated")

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

        self._cluster_runner.clear_tracked_jobs()
        self._sync_cluster_tracking_from_document()

        if self._document.project.recommended_cutoff_energy is not None:
            self._document.ui_state["default_cut_off_energy"] = float(
                self._document.project.recommended_cutoff_energy
            )
        if self._document.project.recommended_cutoff_method is not None:
            self._document.ui_state["default_cut_off_energy_method"] = (
                self._document.project.recommended_cutoff_method
            )
        if self._document.project.recommended_cutoff_energy_threshold is not None:
            self._document.ui_state["default_cut_off_energy_threshold"] = float(
                self._document.project.recommended_cutoff_energy_threshold
            )
        if self._document.project.recommended_cutoff_force_threshold is not None:
            self._document.ui_state["default_cut_off_force_threshold"] = float(
                self._document.project.recommended_cutoff_force_threshold
            )

        self._main_title.setText(f"Project: {self._document.project.name}")
        self._main_body.setText(
            "Continue your expedition from the journal on the left, or use quick actions below."
        )
        self._sync_comparison_table_views()
        self._populate_project_tree()
        self._refresh_jobs_table()
        self._refresh_host_structure_view()
        self._refresh_muon_generation_table()
        self._convergence_view.set_form_state(self._convergence_form_restore_state())
        self._refresh_convergence_simulation_cell()
        self._refresh_convergence_cutoff_rows()
        self._kpoint_convergence_view.set_form_state(
            self._document.ui_state.get("kpoint_convergence_form", {})
        )
        self._refresh_kpoint_convergence_simulation_cell()
        self._refresh_kpoint_convergence_rows()
        self._pristine_relaxation_view.set_form_state(
            self._document.ui_state.get("pristine_relaxation_form", {})
        )
        self._muon_generation_view.set_form_state(
            self._document.ui_state.get("muon_generation_form", {})
        )
        self._convergence_view.set_project_defaults(
            cutoff_energy=self._document.project.recommended_cutoff_energy,
            cutoff_method=self._document.project.recommended_cutoff_method,
            energy_threshold=self._document.project.recommended_cutoff_energy_threshold,
            force_threshold=self._document.project.recommended_cutoff_force_threshold,
        )
        self._kpoint_convergence_view.set_project_defaults(
            kpoint_grid=self._document.project.recommended_kpoint_grid,
            kpoint_method=self._document.project.recommended_kpoint_method,
            energy_threshold=self._document.project.recommended_kpoint_energy_threshold,
            force_threshold=self._document.project.recommended_kpoint_force_threshold,
        )
        self._kpoint_convergence_view.set_cutoff_default(
            self._document.project.recommended_cutoff_energy
        )
        self._refresh_pristine_relaxation_defaults()
        self._refresh_pristine_relaxation_rows()
        self._refresh_convergence_overview()
        self._main_stack.setCurrentWidget(self._home_page)
        self._sync_expedition_trail()

    @staticmethod
    def _overview_placeholder_message(title: str) -> str:
        return (
            f"No adopted {title.lower()} analysis is pinned yet.\n"
            "Open the analysis dialog in the child workflow and adopt a project default to pin it here."
        )

    @staticmethod
    def _snapshot_host_id(snapshot: Mapping[str, object]) -> str | None:
        raw_value = snapshot.get("host_structure_id")
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip()
        return None

    @staticmethod
    def _snapshot_simulation_cell_id(snapshot: Mapping[str, object]) -> str | None:
        raw_value = snapshot.get("simulation_cell_structure_id")
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip()
        return None

    def _host_structure_id_text(self) -> str | None:
        if self._document is None or self._document.project.host_structure_id is None:
            return None
        return str(self._document.project.host_structure_id)

    def _convergence_overview_snapshots_state(self) -> dict[str, object]:
        if self._document is None:
            return {}
        raw_value = self._document.ui_state.get(self._CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY, {})
        if isinstance(raw_value, dict):
            return raw_value
        return {}

    def _validated_cutoff_overview_snapshot(self) -> dict[str, object] | None:
        if self._document is None:
            return None

        snapshot = self._convergence_overview_snapshots_state().get("cutoff")
        if not isinstance(snapshot, dict):
            return None
        if self._snapshot_host_id(snapshot) != self._host_structure_id_text():
            return None

        try:
            snapshot_value = float(snapshot.get("default_value"))
        except (TypeError, ValueError):
            return None

        current_value = self._document.project.recommended_cutoff_energy
        if current_value is None or abs(snapshot_value - float(current_value)) > 1e-9:
            return None
        return snapshot

    def _validated_kpoint_overview_snapshot(self) -> dict[str, object] | None:
        if self._document is None:
            return None

        snapshot = self._convergence_overview_snapshots_state().get("kpoint")
        if not isinstance(snapshot, dict):
            return None
        if self._snapshot_host_id(snapshot) != self._host_structure_id_text():
            return None

        raw_value = snapshot.get("default_value")
        if not isinstance(raw_value, list) or len(raw_value) != 3:
            return None
        try:
            snapshot_grid = (int(raw_value[0]), int(raw_value[1]), int(raw_value[2]))
        except (TypeError, ValueError):
            return None

        current_grid = self._document.project.recommended_kpoint_grid
        if current_grid is None or snapshot_grid != current_grid:
            return None
        return snapshot

    def _cutoff_overview_summary_rows(self) -> list[tuple[str, str]]:
        if self._document is None:
            return [
                ("Project default", "Not set"),
                ("Method", "Not set"),
                ("Criteria", "Not set"),
            ]
        return [
            ("Project default", format_cutoff_energy(self._document.project.recommended_cutoff_energy)),
            ("Method", format_convergence_method(self._document.project.recommended_cutoff_method)),
            (
                "Criteria",
                format_convergence_criteria(
                    energy_threshold=self._document.project.recommended_cutoff_energy_threshold,
                    force_threshold=self._document.project.recommended_cutoff_force_threshold,
                ),
            ),
        ]

    def _kpoint_overview_summary_rows(self) -> list[tuple[str, str]]:
        if self._document is None:
            return [
                ("Project default", "Not set"),
                ("Cutoff default", "Not set"),
                ("Method", "Not set"),
                ("Criteria", "Not set"),
            ]
        return [
            ("Project default", format_kpoint_grid(self._document.project.recommended_kpoint_grid)),
            ("Cutoff default", format_cutoff_energy(self._document.project.recommended_cutoff_energy)),
            ("Method", format_convergence_method(self._document.project.recommended_kpoint_method)),
            (
                "Criteria",
                format_convergence_criteria(
                    energy_threshold=self._document.project.recommended_kpoint_energy_threshold,
                    force_threshold=self._document.project.recommended_kpoint_force_threshold,
                ),
            ),
        ]

    def _refresh_convergence_overview(self) -> None:
        cutoff_snapshot = self._validated_cutoff_overview_snapshot()
        kpoint_snapshot = self._validated_kpoint_overview_snapshot()
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

        definition = self._active_convergence_definition()
        if definition is None:
            return persisted

        configuration = definition.parameters.get("configuration", {})
        if not isinstance(configuration, dict):
            return persisted

        recovered = dict(configuration)

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

    @staticmethod
    def _workflow_host_complete(document: ProjectDocument) -> bool:
        return document.project.host_structure_id is not None

    @staticmethod
    def _workflow_cutoff_complete(document: ProjectDocument) -> bool:
        return document.project.recommended_cutoff_energy is not None

    @staticmethod
    def _workflow_kpoint_complete(document: ProjectDocument) -> bool:
        return document.project.recommended_kpoint_grid is not None

    @staticmethod
    def _workflow_muon_candidates_complete(document: ProjectDocument) -> bool:
        return bool(document.sites)

    @staticmethod
    def _workflow_results_complete(document: ProjectDocument) -> bool:
        return bool(document.results)

    @staticmethod
    def _workflow_convergence_complete(document: ProjectDocument) -> bool:
        return (
            document.project.recommended_cutoff_energy is not None
            and document.project.recommended_kpoint_grid is not None
        )

    @staticmethod
    def _workflow_pristine_relaxation_complete(document: ProjectDocument) -> bool:
        return bool(document.comparison_tables)

    def _workflow_definitions(self) -> tuple[WorkflowDefinition, ...]:
        cutoff = WorkflowDefinition(
            key="convergence_cutoff",
            title="Cutoff Energy",
            subtitle="Accuracy check",
            route="convergence_cutoff",
            completion_goal="Set a project default cutoff value.",
            completion_check=self._workflow_cutoff_complete,
        )
        kpoint = WorkflowDefinition(
            key="convergence_kpoint",
            title="K-Point Grid",
            subtitle="Sampling check",
            route="convergence_kpoint",
            completion_goal="Set a project default k-point grid.",
            completion_check=self._workflow_kpoint_complete,
        )
        convergence = WorkflowDefinition(
            key="convergence",
            title="Convergence Testing",
            subtitle="Find stable settings",
            route="convergence",
            completion_goal="Set both project default convergence values (cutoff and k-point).",
            completion_check=self._workflow_convergence_complete,
            children=(cutoff, kpoint),
        )
        return (
            WorkflowDefinition(
                key="host_structure",
                title="Host Structure",
                subtitle="Load and inspect crystal",
                route="host_structure",
                completion_goal="Load a host structure.",
                completion_check=self._workflow_host_complete,
            ),
            convergence,
            WorkflowDefinition(
                key="pristine_relaxation",
                title="Pristine Relaxation",
                subtitle="Optional cleanup",
                route="pristine_relaxation",
                completion_goal="Generate at least one comparison table.",
                completion_check=self._workflow_pristine_relaxation_complete,
            ),
            WorkflowDefinition(
                key="muon_sites",
                title="Muon Sites",
                subtitle="Place and refine muons",
                completion_goal="Generate at least one candidate muon site.",
                completion_check=self._workflow_muon_candidates_complete,
                children=(
                    WorkflowDefinition(
                        key="muon_candidates",
                        title="Candidates",
                        route="muon_candidates",
                        completion_goal="Generate at least one candidate muon site.",
                        completion_check=self._workflow_muon_candidates_complete,
                    ),
                    WorkflowDefinition(
                        key="muon_relaxations",
                        title="Relaxations",
                        route="muon_relaxations",
                    ),
                    WorkflowDefinition(
                        key="muon_confirmed",
                        title="Confirmed Sites",
                        route="muon_confirmed",
                    ),
                ),
            ),
            WorkflowDefinition(
                key="results",
                title="Results",
                subtitle="Review what MuFinder found",
                route="results",
                completion_goal="Capture at least one completed parsed result.",
                completion_check=self._workflow_results_complete,
            ),
            WorkflowDefinition(key="magnetic_fields", title="Magnetic Fields"),
            WorkflowDefinition(key="quantum_corrections", title="Quantum Corrections"),
            WorkflowDefinition(key="energy_barriers", title="Energy Barriers"),
            WorkflowDefinition(key="provenance_log", title="Provenance Log"),
        )

    def _workflow_label(self, definition: WorkflowDefinition) -> str:
        return definition.title

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

    def _populate_project_tree(self) -> None:
        if self._document is None:
            return

        previous_route: str | None = None
        current_item = self._project_tree.currentItem()
        if current_item is not None:
            raw_route = current_item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(raw_route, str) and raw_route.strip():
                previous_route = raw_route.strip()

        self._project_tree.clear()
        root = QTreeWidgetItem([self._document.project.name])

        for definition in self._workflow_definitions():
            root.addChild(self._build_workflow_tree_item(definition))

        pristine_item = self._find_workflow_item_by_route(root, "pristine_relaxation")
        if pristine_item is not None:
            ordered_tables = sorted(
                self._document.comparison_tables.values(),
                key=lambda table: (table.created_at, table.label.casefold()),
            )
            for table in ordered_tables:
                child = QTreeWidgetItem([table.label or "Comparison Table"])
                child.setData(0, Qt.ItemDataRole.UserRole, self._comparison_route(table.id))
                pristine_item.addChild(child)
            pristine_item.setExpanded(True)

        self._project_tree.addTopLevelItem(root)
        root.setExpanded(True)
        for index in range(root.childCount()):
            child = root.child(index)
            if child is None:
                continue
            label = child.text(0)
            if label.startswith("Convergence Testing") or label.startswith("Muon Sites"):
                child.setExpanded(True)
        selected_item = root
        if previous_route is not None:
            match = self._find_workflow_item_by_route(root, previous_route)
            if match is not None:
                selected_item = match

        self._project_tree.setCurrentItem(selected_item)
        self._refresh_workflow_stage_icons()
        self._sync_expedition_trail()

    def _refresh_jobs_table(self) -> None:
        self._jobs_table.setRowCount(0)
        if self._document is None:
            self._convergence_view.set_cutoff_runtime_states({})
            self._kpoint_convergence_view.set_runtime_states({})
            self._pristine_relaxation_view.set_runtime_states({})
            return

        for job in self._document.jobs.values():
            if job.status not in _VISIBLE_DASHBOARD_STATES:
                continue
            row = self._jobs_table.rowCount()
            self._jobs_table.insertRow(row)
            label_item = QTableWidgetItem(job.label)
            label_item.setData(Qt.ItemDataRole.UserRole, str(job.id))
            self._jobs_table.setItem(row, self._JOB_LABEL_COLUMN, label_item)
            self._jobs_table.setItem(row, self._JOB_WORKFLOW_COLUMN, QTableWidgetItem(job.workflow_stage.value))
            self._jobs_table.setItem(row, self._JOB_CALC_TYPE_COLUMN, QTableWidgetItem(job.calc_type))

            status_item = QTableWidgetItem(job.status.value)
            status_item.setForeground(self._status_brush(job.status.value))
            self._jobs_table.setItem(row, self._JOB_STATUS_COLUMN, status_item)

            cluster_name = "local" if job.execution_target == "local" else job.execution_target
            self._jobs_table.setItem(row, self._JOB_CLUSTER_COLUMN, QTableWidgetItem(cluster_name))
            self._jobs_table.setItem(
                row,
                self._JOB_REMOTE_ID_COLUMN,
                QTableWidgetItem(job.remote_job_id or ""),
            )
            self._jobs_table.setItem(row, self._JOB_RUNTIME_COLUMN, QTableWidgetItem(self._job_runtime_text(job)))
            self._jobs_table.setItem(
                row,
                self._JOB_PROCESSES_COLUMN,
                QTableWidgetItem("" if job.num_processes is None else str(job.num_processes)),
            )

            can_submit = job.status == JobStatus.DEFINED
            can_cancel = job.status in _ACTIVE_LOCAL_JOB_STATES
            can_inspect = job.status in {JobStatus.FAILED, JobStatus.STOPPED}
            can_reset = job.status in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}

            self._jobs_table.setCellWidget(
                row,
                self._JOB_SUBMIT_COLUMN,
                self._make_job_action_button(
                    "Submit",
                    enabled=can_submit,
                    callback=lambda _checked=False, job_id=job.id: self._on_submit_from_dashboard([job_id]),
                ),
            )
            self._jobs_table.setCellWidget(
                row,
                self._JOB_CANCEL_COLUMN,
                self._make_job_action_button(
                    "Cancel",
                    enabled=can_cancel,
                    callback=lambda _checked=False, job_id=job.id: self._on_cancel_from_dashboard([job_id]),
                ),
            )
            self._jobs_table.setCellWidget(
                row,
                self._JOB_INSPECT_COLUMN,
                self._make_job_action_button(
                    "Inspect",
                    enabled=can_inspect,
                    callback=lambda _checked=False, job_id=job.id: self._on_inspect_failed_from_dashboard([job_id]),
                ),
            )
            self._jobs_table.setCellWidget(
                row,
                self._JOB_RESET_COLUMN,
                self._make_job_action_button(
                    "Reset",
                    enabled=can_reset,
                    callback=lambda _checked=False, job_id=job.id: self._on_reset_failed_from_dashboard([job_id]),
                ),
            )

        self._configure_jobs_table_columns()

        self._refresh_convergence_cutoff_rows()
        self._refresh_kpoint_convergence_rows()
        self._refresh_pristine_relaxation_rows()
        self._sync_expedition_trail()

    def _configure_jobs_table_columns(self) -> None:
        """Stretch primary dashboard columns while keeping action columns compact."""

        header = self._jobs_table.horizontalHeader()
        stretch_columns = (
            self._JOB_LABEL_COLUMN,
            self._JOB_WORKFLOW_COLUMN,
            self._JOB_CALC_TYPE_COLUMN,
            self._JOB_STATUS_COLUMN,
            self._JOB_CLUSTER_COLUMN,
            self._JOB_REMOTE_ID_COLUMN,
        )
        contents_columns = (
            self._JOB_RUNTIME_COLUMN,
            self._JOB_PROCESSES_COLUMN,
            self._JOB_SUBMIT_COLUMN,
            self._JOB_CANCEL_COLUMN,
            self._JOB_INSPECT_COLUMN,
            self._JOB_RESET_COLUMN,
        )

        for column in stretch_columns:
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

        for column in contents_columns:
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            self._jobs_table.resizeColumnToContents(column)

    @staticmethod
    def _make_job_action_button(text: str, *, enabled: bool, callback) -> QPushButton:
        button = QPushButton(text)
        button.setEnabled(enabled)
        button.clicked.connect(callback)
        return button

    @staticmethod
    def _status_brush(status_value: str):
        return status_brush(status_value)

    @staticmethod
    def _job_runtime_text(job: Job) -> str:
        if job.started_at is None:
            return ""

        end_time = job.completed_at or utc_now()
        elapsed_seconds = max(0, int((end_time - job.started_at).total_seconds()))
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _add_demo_job(self) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return

        job = self._project_service.add_job(
            self._document,
            label="Muon relaxation demo",
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type="castep_geom_opt",
        )
        self._project_service.submit_job(job)
        self._project_service.mark_job_queued(job, remote_job_id="demo-queued")
        self._refresh_jobs_table()

    def _persist_document_ui_state(self, *, failure_message: str) -> None:
        if self._document is None:
            return
        try:
            self._project_service.save_project(self._document)
        except ValueError:
            self.statusBar().showMessage(failure_message)

    def _store_convergence_overview_snapshot(self, kind: str, snapshot: Mapping[str, object] | None) -> None:
        if self._document is None:
            return

        snapshots = dict(self._convergence_overview_snapshots_state())
        if snapshot is None:
            snapshots.pop(kind, None)
        else:
            serialized = to_jsonable(dict(snapshot))
            if isinstance(serialized, dict):
                snapshots[kind] = serialized
        self._document.ui_state[self._CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] = snapshots

    def _build_cutoff_overview_snapshot(
        self,
        *,
        cutoff_energy: float,
        method: str,
        energy_threshold: float,
        force_threshold: float,
        simulation_cell_structure_id: str | None,
    ) -> dict[str, object] | None:
        if self._document is None:
            return None
        host_structure_id = self._host_structure_id_text()
        if host_structure_id is None:
            return None

        state = self._document.ui_state.get(self._CONVERGENCE_EXAMINER_STATE_KEY, {})
        if not isinstance(state, Mapping):
            return None
        plot_data = state.get("plot_data")
        if not isinstance(plot_data, Mapping):
            return None

        return {
            "host_structure_id": host_structure_id,
            "simulation_cell_structure_id": simulation_cell_structure_id,
            "default_value": float(cutoff_energy),
            "method": method,
            "energy_threshold": float(energy_threshold),
            "force_threshold": float(force_threshold),
            "plot_data": dict(plot_data),
        }

    def _build_kpoint_overview_snapshot(
        self,
        *,
        grid: tuple[int, int, int],
        method: str,
        energy_threshold: float,
        force_threshold: float,
        simulation_cell_structure_id: str | None,
        labels: Mapping[float, str],
    ) -> dict[str, object] | None:
        if self._document is None:
            return None
        host_structure_id = self._host_structure_id_text()
        if host_structure_id is None:
            return None

        state = self._document.ui_state.get(self._KPOINT_EXAMINER_STATE_KEY, {})
        if not isinstance(state, Mapping):
            return None
        plot_data = state.get("plot_data")
        if not isinstance(plot_data, Mapping):
            return None

        serialized_labels = {str(float(key)): value for key, value in labels.items()}
        return {
            "host_structure_id": host_structure_id,
            "simulation_cell_structure_id": simulation_cell_structure_id,
            "default_value": [int(grid[0]), int(grid[1]), int(grid[2])],
            "method": method,
            "energy_threshold": float(energy_threshold),
            "force_threshold": float(force_threshold),
            "plot_data": dict(plot_data),
            "point_labels": serialized_labels,
        }

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
        elif route == "muon_candidates":
            target = self._muon_generation_view
        elif isinstance(route, str):
            table_id = self._comparison_table_id_from_route(route)
            if table_id is not None:
                target = self._comparison_widgets.get(table_id, self._pristine_relaxation_view)
        self._main_stack.setCurrentWidget(target)
        self._refresh_workflow_stage_icons()
        self._sync_expedition_trail()

    def _comparison_route(self, table_id: UUID) -> str:
        return f"{self._COMPARISON_ROUTE_PREFIX}{table_id}"

    def _comparison_table_id_from_route(self, route: object) -> UUID | None:
        if not isinstance(route, str):
            return None
        if not route.startswith(self._COMPARISON_ROUTE_PREFIX):
            return None
        raw_id = route[len(self._COMPARISON_ROUTE_PREFIX) :].strip()
        if not raw_id:
            return None
        try:
            return UUID(raw_id)
        except ValueError:
            return None

    def _sync_comparison_table_views(self) -> None:
        for widget in self._comparison_widgets.values():
            self._main_stack.removeWidget(widget)
            widget.deleteLater()
        self._comparison_widgets.clear()

        if self._document is None:
            return

        ordered_tables = sorted(
            self._document.comparison_tables.values(),
            key=lambda table: (table.created_at, table.label.casefold()),
        )
        for table in ordered_tables:
            widget = ComparisonTableWidget(table, parent=self)
            widget.delete_requested.connect(self._on_delete_comparison_table_requested)
            self._main_stack.addWidget(widget)
            self._comparison_widgets[table.id] = widget

    def _on_delete_comparison_table_requested(self, payload: object) -> None:
        if isinstance(payload, UUID):
            table_id = payload
        elif isinstance(payload, str):
            try:
                table_id = UUID(payload)
            except ValueError:
                return
        else:
            return
        # The comparison widget already asks for confirmation before emitting.
        self._delete_comparison_table(table_id, confirm=False)

    def _delete_comparison_table(self, table_id: UUID, *, confirm: bool = True) -> None:
        if self._document is None:
            return
        if table_id not in self._document.comparison_tables:
            return

        if confirm:
            answer = QMessageBox.question(
                self,
                "Delete comparison table",
                "Delete this comparison table?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self._project_service.remove_comparison_table(self._document, table_id)
        self._sync_comparison_table_views()
        self._populate_project_tree()

        route = self._active_route()
        if route is None or self._comparison_table_id_from_route(route) is not None:
            self._set_active_route("pristine_relaxation")

        self.statusBar().showMessage("Deleted comparison table")

    def _set_active_route(self, route: str) -> None:
        root = self._project_tree.topLevelItem(0)
        if root is None:
            return
        item = self._find_workflow_item_by_route(root, route)
        if item is None:
            return
        self._project_tree.setCurrentItem(item)
        self._on_tree_selection_changed()

    def _active_route(self) -> str | None:
        selected_items = self._project_tree.selectedItems()
        if not selected_items:
            return None
        route = selected_items[0].data(0, Qt.ItemDataRole.UserRole)
        if isinstance(route, str) and route.strip():
            return route.strip()
        return None

    def _expedition_stage_states(self) -> list[tuple[str, str]]:
        if self._document is None:
            return [
                ("Host Structure", "incomplete"),
                ("Convergence", "incomplete"),
                ("Muon Sites", "incomplete"),
                ("Results", "incomplete"),
            ]

        completion = {
            "host": self._workflow_host_complete(self._document),
            "convergence": self._workflow_convergence_complete(self._document),
            "muon": self._workflow_muon_candidates_complete(self._document),
            "results": self._workflow_results_complete(self._document),
        }
        stages = [
            ("host", "Host Structure"),
            ("convergence", "Convergence"),
            ("muon", "Muon Sites"),
            ("results", "Results"),
        ]
        resolved: list[tuple[str, str]] = []
        for key, label in stages:
            state = "completed" if completion[key] else "incomplete"
            resolved.append((label, state))
        return resolved

    def _sync_expedition_trail(self) -> None:
        if self._expedition_trail is None:
            return
        self._expedition_trail.set_stages(self._expedition_stage_states())

    def _refresh_workflow_stage_icons(self) -> None:
        root = self._project_tree.topLevelItem(0)
        if root is None:
            return

        def visit(item: QTreeWidgetItem) -> None:
            has_goal = item.data(0, self._WORKFLOW_GOAL_ROLE) is not None
            if has_goal:
                is_complete = bool(item.data(0, self._WORKFLOW_COMPLETE_ROLE))
                state = "completed" if is_complete else "incomplete"
                item.setIcon(0, workflow_stage_icon(state))

            for idx in range(item.childCount()):
                child = item.child(idx)
                if child is not None:
                    visit(child)

        visit(root)

    def _on_submit_local_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return
        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        self._submit_jobs_locally([job for job in jobs if job is not None])

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
            self._submit_jobs_locally(local_jobs)

        if cluster_jobs:
            preferred = None
            targets = {job.execution_target for job in cluster_jobs}
            if len(targets) == 1:
                preferred = next(iter(targets))
            self._submit_jobs_on_cluster(cluster_jobs, default_cluster_name=preferred)

    def _on_cancel_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return
        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        self._cancel_jobs([job for job in jobs if job is not None])

    def _on_inspect_failed_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return
        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        self._inspect_failed_jobs([job for job in jobs if job is not None])

    def _on_reset_failed_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return
        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        self._reset_failed_jobs([job for job in jobs if job is not None])

    def _jobs_for_convergence_cutoffs(self, cutoff_values: list[float]) -> list[Job]:
        if self._document is None:
            return []

        states = self._build_convergence_cutoff_states()
        jobs_by_label = {job.label: job for job in self._document.jobs.values()}
        selected: list[Job] = []
        seen: set[UUID] = set()
        for cutoff in cutoff_values:
            state = states.get(float(cutoff))
            if state is None:
                continue
            job = self._document.jobs.get(state.job_id) if state.job_id is not None else None
            if job is None and state.job_label is not None:
                job = jobs_by_label.get(state.job_label)
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    @staticmethod
    def _cutoff_key(value: float) -> str:
        return f"{float(value):.12g}"

    def _deserialize_convergence_config(self, payload: object) -> ConvergenceConfig | None:
        if not isinstance(payload, dict):
            return None

        sweep_parameter = str(payload.get("sweep_parameter", "cut_off_energy")).strip() or "cut_off_energy"
        if sweep_parameter not in {"cut_off_energy", "kpoint_mp_grid"}:
            return None

        def parse_sweep_value(value: object):
            if sweep_parameter == "cut_off_energy":
                return float(value)
            if isinstance(value, (list, tuple)) and len(value) == 3:
                return (int(value[0]), int(value[1]), int(value[2]))
            raise ValueError

        try:
            raw_sweep_values = payload.get("sweep_values")
            if sweep_parameter == "cut_off_energy":
                raw_cutoff_values = payload.get("cutoff_energies")
                if isinstance(raw_cutoff_values, list):
                    raw_sweep_values = raw_cutoff_values
            if not isinstance(raw_sweep_values, list):
                raw_sweep_values = payload.get("cutoff_energies", [])
            sweep_values = tuple(parse_sweep_value(value) for value in raw_sweep_values)
        except (TypeError, ValueError):
            return None

        shared_params = payload.get("shared_params", {})
        if not isinstance(shared_params, dict):
            return None

        shared_clean = {
            str(name): self._restore_castep_param_value(str(name), raw_value)
            for name, raw_value in shared_params.items()
        }

        overrides: dict[frozenset[object], dict[str, object]] = {}
        raw_overrides = payload.get("overrides", [])
        if isinstance(raw_overrides, list):
            for entry in raw_overrides:
                if not isinstance(entry, dict):
                    continue
                try:
                    raw_group = entry.get("cutoffs", entry.get("sweep_values", []))
                    cutoff_group = frozenset(parse_sweep_value(value) for value in raw_group)
                except (TypeError, ValueError):
                    continue
                params = entry.get("params", {})
                if not cutoff_group or not isinstance(params, dict):
                    continue
                overrides[cutoff_group] = {
                    str(name): self._restore_castep_param_value(str(name), raw_value)
                    for name, raw_value in params.items()
                }

        default_prefix = "kpoint_convergence" if sweep_parameter == "kpoint_mp_grid" else "cutoff_convergence"

        return ConvergenceConfig(
            sweep_parameter=sweep_parameter,
            sweep_values=sweep_values,
            shared_params=shared_clean,
            overrides=overrides,
            name_prefix=str(payload.get("name_prefix", default_prefix)).strip() or default_prefix,
        )

    def _restore_castep_param_value(self, name: str, value: object) -> object:
        spec = get_keyword_spec("param", name) or get_keyword_spec("cell", name)
        if spec is None:
            return value

        if isinstance(value, dict):
            if spec.kind == "block" and "rows" in value:
                rows = tuple(tuple(row) for row in value.get("rows", ()))
                return BlockSection(
                    name=str(value.get("name", spec.name)),
                    rows=rows,
                    unit=str(value["unit"]) if value.get("unit") is not None else None,
                )
            if spec.value_type in {"physical", "positive_float_with_optional_unit"} and {"value", "unit"} <= value.keys():
                return ValueWithUnit(value=value["value"], unit=str(value["unit"]))

        if isinstance(value, list) and spec.value_type in {"int_triplet", "real_vector"}:
            value = tuple(value)

        try:
            return normalize_gui_value(spec, value)
        except (TypeError, ValueError):
            return value

    def _convergence_definitions(self) -> list:
        if self._document is None:
            return []
        definitions = [
            definition
            for definition in self._document.calculations.values()
            if definition.workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF
        ]
        definitions.sort(key=lambda definition: definition.created_at, reverse=True)
        return definitions

    def _active_convergence_definition(self):
        if self._document is None:
            return None

        raw_definition_id = self._document.ui_state.get(self._ACTIVE_CONVERGENCE_DEFINITION_KEY)
        if isinstance(raw_definition_id, str):
            try:
                definition_id = UUID(raw_definition_id)
            except ValueError:
                definition_id = None
            if definition_id is not None:
                definition = self._document.calculations.get(definition_id)
                if definition is not None and definition.workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF:
                    return definition

        definitions = self._convergence_definitions()
        return definitions[0] if definitions else None

    def _definition_for_convergence_cutoff(self, cutoff: float):
        cutoff_value = float(cutoff)
        active_definition = self._active_convergence_definition()
        if active_definition is None:
            return None, None

        config = self._deserialize_convergence_config(active_definition.parameters.get("configuration", {}))
        if config is not None and cutoff_value in config.cutoff_energies:
            return active_definition, config
        return None, None

    def _job_for_definition_cutoff(self, definition, cutoff: float) -> Job | None:
        if self._document is None:
            return None

        mapping = definition.parameters.get("job_ids_by_cutoff", {})
        if isinstance(mapping, dict):
            raw_job_id = mapping.get(self._cutoff_key(cutoff))
            if isinstance(raw_job_id, str):
                try:
                    job_id = UUID(raw_job_id)
                except ValueError:
                    job_id = None
                if job_id is not None:
                    job = self._document.jobs.get(job_id)
                    if job is not None:
                        return job

        labels = definition.parameters.get("job_labels", [])
        configuration = self._deserialize_convergence_config(definition.parameters.get("configuration", {}))
        if isinstance(labels, list) and configuration is not None:
            try:
                index = list(configuration.cutoff_energies).index(float(cutoff))
            except ValueError:
                return None
            if index >= len(labels):
                return None
            label = str(labels[index])
            for job in reversed(tuple(self._document.jobs.values())):
                if job.label == label:
                    return job
        return None

    def _definition_output_directory(self, definition) -> Path | None:
        if self._document is None:
            return None
        raw_output_dir = definition.parameters.get("output_directory")
        if not isinstance(raw_output_dir, str) or not raw_output_dir.strip():
            return None
        base = Path(raw_output_dir)
        if not base.is_absolute():
            base = self._document.project_root / base
        return base

    def _materialize_convergence_jobs(self, cutoff_values: list[float]) -> list[Job]:
        if self._document is None:
            return []

        created: list[Job] = []
        grouped: dict[UUID, tuple[object, ConvergenceConfig, set[float]]] = {}
        for cutoff in cutoff_values:
            definition, config = self._definition_for_convergence_cutoff(cutoff)
            if definition is None or config is None:
                continue
            existing = self._job_for_definition_cutoff(definition, cutoff)
            if existing is not None:
                created.append(existing)
                continue
            record = grouped.get(definition.id)
            if record is None:
                grouped[definition.id] = (definition, config, {float(cutoff)})
            else:
                record[2].add(float(cutoff))

        for definition, config, cutoffs in grouped.values():
            if not definition.input_structure_ids:
                continue
            structure = self._document.structures.get(definition.input_structure_ids[0])
            output_root = self._definition_output_directory(definition)
            if structure is None or output_root is None:
                continue
            self._project_service.repair_supercell_asset_if_needed(
                self._document,
                structure_id=structure.id,
            )

            try:
                structure_spec = structure_to_spec(structure)
                all_jobs = expand_convergence_jobs(structure=structure_spec, config=config)
            except (ValueError, StructureBridgeError):
                continue

            desired_labels = {
                convergence_job_name(
                    config.name_prefix,
                    cutoff,
                    sweep_parameter="cut_off_energy",
                ): cutoff
                for cutoff in cutoffs
            }
            mapping = definition.parameters.setdefault("job_ids_by_cutoff", {})
            if not isinstance(mapping, dict):
                mapping = {}
                definition.parameters["job_ids_by_cutoff"] = mapping

            for job_spec in all_jobs:
                cutoff = desired_labels.get(job_spec.name)
                if cutoff is None:
                    continue
                write_job_directory(job_spec, output_root)
                job = self._project_service.add_job(
                    self._document,
                    label=job_spec.name,
                    workflow_stage=job_spec.workflow_stage,
                    calc_type="castep_convergence_cutoff",
                )
                mapping[self._cutoff_key(cutoff)] = str(job.id)
                created.append(job)

        deduped: list[Job] = []
        seen: set[UUID] = set()
        for job in created:
            if job.id in seen:
                continue
            seen.add(job.id)
            deduped.append(job)
        return deduped

    def _on_submit_local_from_convergence(self, cutoff_values: list[float]) -> None:
        seen_cutoffs: set[float] = set()
        ordered_cutoffs: list[float] = []
        for cutoff in cutoff_values:
            normalized = float(cutoff)
            if normalized in seen_cutoffs:
                continue
            seen_cutoffs.add(normalized)
            ordered_cutoffs.append(normalized)

        states = self._build_convergence_cutoff_states()
        jobs = self._jobs_for_convergence_cutoffs(ordered_cutoffs)
        ready_cutoffs = [
            cutoff
            for cutoff in ordered_cutoffs
            if (states.get(cutoff) or SweepPointRunState()).status == "ready"
        ]
        jobs.extend(self._materialize_convergence_jobs(ready_cutoffs))
        jobs = self._dedupe_jobs(jobs)
        if not jobs:
            QMessageBox.information(
                self,
                "No ready jobs",
                "Only configured cutoffs in the 'ready' state can be submitted.",
            )
            return
        self._submit_jobs_locally(jobs)

    def _on_submit_cluster_from_convergence(self, cutoff_values: list[float]) -> None:
        seen_cutoffs: set[float] = set()
        ordered_cutoffs: list[float] = []
        for cutoff in cutoff_values:
            normalized = float(cutoff)
            if normalized in seen_cutoffs:
                continue
            seen_cutoffs.add(normalized)
            ordered_cutoffs.append(normalized)

        states = self._build_convergence_cutoff_states()
        jobs = self._jobs_for_convergence_cutoffs(ordered_cutoffs)
        ready_cutoffs = [
            cutoff
            for cutoff in ordered_cutoffs
            if (states.get(cutoff) or SweepPointRunState()).status == "ready"
        ]
        jobs.extend(self._materialize_convergence_jobs(ready_cutoffs))
        jobs = self._dedupe_jobs(jobs)
        if not jobs:
            QMessageBox.information(
                self,
                "No ready jobs",
                "Only configured cutoffs in the 'ready' state can be submitted.",
            )
            return

        self._submit_jobs_on_cluster(jobs)

    def _on_examine_convergence(self) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return

        host_id = self._document.project.host_structure_id
        host = self._document.structures.get(host_id) if host_id is not None else None
        if host is None:
            QMessageBox.warning(
                self,
                "No host structure",
                "Load a host structure before examining convergence.",
            )
            return

        persisted_state = self._document.ui_state.get(self._CONVERGENCE_EXAMINER_STATE_KEY, {})
        if not isinstance(persisted_state, dict):
            persisted_state = {}

        dialog = ConvergenceExaminerDialog(
            self._build_convergence_cutoff_states(),
            num_atoms=len(host.species),
            initial_state=persisted_state,
            parent=self,
        )
        dialog.cutoff_adopted.connect(self._on_cutoff_adopted)
        dialog.state_changed.connect(self._on_convergence_examiner_state_changed)
        dialog.exec()
        self._on_convergence_examiner_state_changed(dialog.state_snapshot())

    def _on_convergence_examiner_state_changed(self, state: dict) -> None:
        if self._document is None:
            return

        serialized_state = to_jsonable(state)
        if not isinstance(serialized_state, dict):
            return

        self._document.ui_state[self._CONVERGENCE_EXAMINER_STATE_KEY] = serialized_state
        try:
            self._project_service.save_project(self._document)
        except ValueError:
            self.statusBar().showMessage("Could not persist convergence examiner state to disk")

    def _on_cutoff_adopted(
        self,
        cutoff_energy: float,
        method: str,
        energy_threshold: float,
        force_threshold: float,
    ) -> None:
        if self._document is None:
            return
        definition = self._active_convergence_definition()
        definition_cell_id = None
        if definition is not None and definition.input_structure_ids:
            definition_cell_id = str(definition.input_structure_ids[0])
        self._project_service.set_recommended_cutoff(
            self._document,
            cutoff_energy,
            method=method,
            energy_threshold=energy_threshold,
            force_threshold=force_threshold,
            cell_id=(
                definition_cell_id
                if definition_cell_id is not None
                else (str(self._document.project.host_structure_id) if self._document.project.host_structure_id else None)
            ),
        )
        self._document.ui_state["default_cut_off_energy"] = float(cutoff_energy)
        self._document.ui_state["default_cut_off_energy_method"] = method
        self._document.ui_state["default_cut_off_energy_threshold"] = float(energy_threshold)
        self._document.ui_state["default_cut_off_force_threshold"] = float(force_threshold)
        self._store_convergence_overview_snapshot(
            "cutoff",
            self._build_cutoff_overview_snapshot(
                cutoff_energy=float(cutoff_energy),
                method=method,
                energy_threshold=float(energy_threshold),
                force_threshold=float(force_threshold),
                simulation_cell_structure_id=definition_cell_id,
            ),
        )
        self._convergence_view.set_project_defaults(
            cutoff_energy=float(cutoff_energy),
            cutoff_method=method,
            energy_threshold=float(energy_threshold),
            force_threshold=float(force_threshold),
        )
        self._kpoint_convergence_view.set_cutoff_default(float(cutoff_energy))
        self._refresh_convergence_overview()
        self._persist_document_ui_state(
            failure_message="Could not persist convergence overview snapshot to disk"
        )
        self._populate_project_tree()
        self.statusBar().showMessage(f"Project default cutoff set to {float(cutoff_energy):.8g} eV")

    def _on_cancel_from_convergence(self, cutoff_values: list[float]) -> None:
        jobs = self._jobs_for_convergence_cutoffs(cutoff_values)
        self._cancel_jobs(jobs)

    def _on_inspect_failed_from_convergence(self, cutoff_values: list[float]) -> None:
        jobs = self._jobs_for_convergence_cutoffs(cutoff_values)
        self._inspect_failed_jobs(jobs)

    def _on_reset_failed_from_convergence(self, cutoff_values: list[float]) -> None:
        jobs = self._jobs_for_convergence_cutoffs(cutoff_values)
        self._reset_failed_jobs(jobs)

    def _on_remove_cutoffs_from_convergence(self, cutoff_values: list[float]) -> None:
        if not cutoff_values:
            return

        if self._document is None:
            self._convergence_view.remove_cutoffs(cutoff_values)
            return

        removed_cutoffs: list[float] = []
        shared_file_disposition: str | None = None
        multi = len(cutoff_values) > 1
        for cutoff in cutoff_values:
            removed, shared_file_disposition, aborted = self._remove_single_convergence_cutoff(
                float(cutoff),
                shared_file_disposition=shared_file_disposition,
                include_apply_all=multi,
            )
            if aborted:
                break
            if removed:
                removed_cutoffs.append(float(cutoff))

        if removed_cutoffs:
            self._convergence_view.remove_cutoffs(removed_cutoffs)
            self._refresh_jobs_table()
            self._refresh_convergence_cutoff_rows()
            self.statusBar().showMessage(
                f"Removed {len(removed_cutoffs)} convergence cutoff entr{('y' if len(removed_cutoffs) == 1 else 'ies')}"
            )

    def _remove_single_convergence_cutoff(
        self,
        cutoff: float,
        *,
        shared_file_disposition: str | None = None,
        include_apply_all: bool = False,
    ) -> tuple[bool, str | None, bool]:
        if self._document is None:
            return True, shared_file_disposition, False

        definition, config = self._definition_for_convergence_cutoff(cutoff)
        job = self._job_for_definition_cutoff(definition, cutoff) if definition is not None else None

        job_label = None
        if job is not None:
            job_label = job.label
        elif config is not None:
            job_label = convergence_job_name(
                config.name_prefix,
                cutoff,
                sweep_parameter="cut_off_energy",
            )

        output_root = self._definition_output_directory(definition) if definition is not None else None
        job_dir = output_root / job_label if output_root is not None and job_label else None

        if job_dir is not None and job_dir.exists():
            disposition = shared_file_disposition
            if disposition is None:
                disposition = self._prompt_cutoff_file_disposition(
                    cutoff,
                    job_dir,
                    include_apply_all=include_apply_all,
                )
            if disposition is None:
                return False, shared_file_disposition, True
            try:
                if disposition in {"delete", "delete_all"}:
                    self._delete_path(job_dir)
                else:
                    self._archive_cutoff_directory(job_dir, output_root)
            except OSError as exc:
                QMessageBox.warning(
                    self,
                    "File cleanup failed",
                    f"Could not update files for cutoff {cutoff:.8g} eV:\n{exc}",
                )
                return False, shared_file_disposition, False

            if disposition in {"delete_all", "archive_all"}:
                shared_file_disposition = disposition

        if job is not None:
            self._delete_job_records(job.id)

        if definition is not None:
            self._remove_cutoff_from_definition(definition, cutoff)

        return True, shared_file_disposition, False

    def _remove_cutoff_from_definition(self, definition, cutoff: float) -> None:
        if self._document is None:
            return

        configuration = definition.parameters.get("configuration")
        if not isinstance(configuration, dict):
            return

        cutoffs = configuration.get("cutoff_energies")
        if not isinstance(cutoffs, list):
            return

        remove_index: int | None = None
        updated_cutoffs: list[float] = []
        cutoff_to_remove = float(cutoff)
        for index, raw_value in enumerate(cutoffs):
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                updated_cutoffs.append(raw_value)
                continue
            if remove_index is None and value == cutoff_to_remove:
                remove_index = index
                continue
            updated_cutoffs.append(value)
        configuration["cutoff_energies"] = updated_cutoffs

        raw_sweep_values = configuration.get("sweep_values")
        if isinstance(raw_sweep_values, list):
            updated_sweep_values: list[object] = []
            for raw_value in raw_sweep_values:
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    updated_sweep_values.append(raw_value)
                    continue
                if value == cutoff_to_remove:
                    continue
                updated_sweep_values.append(value)
            configuration["sweep_values"] = updated_sweep_values

        raw_overrides = configuration.get("overrides", [])
        if isinstance(raw_overrides, list):
            updated_overrides: list[dict[str, object]] = []
            for entry in raw_overrides:
                if not isinstance(entry, dict):
                    continue
                group_key = "cutoffs" if "cutoffs" in entry else "sweep_values"
                raw_group = entry.get(group_key, [])
                if not isinstance(raw_group, list):
                    continue
                remaining: list[float] = []
                for raw_value in raw_group:
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    if value == cutoff_to_remove:
                        continue
                    remaining.append(value)
                if not remaining:
                    continue
                updated_entry = dict(entry)
                updated_entry[group_key] = remaining
                updated_overrides.append(updated_entry)
            configuration["overrides"] = updated_overrides

        mapping = definition.parameters.get("job_ids_by_cutoff", {})
        if isinstance(mapping, dict):
            mapping.pop(self._cutoff_key(cutoff), None)

        labels = definition.parameters.get("job_labels", [])
        if isinstance(labels, list) and remove_index is not None and remove_index < len(labels):
            labels.pop(remove_index)

        if not configuration.get("cutoff_energies"):
            self._document.calculations.pop(definition.id, None)
            if self._document.ui_state.get(self._ACTIVE_CONVERGENCE_DEFINITION_KEY) == str(definition.id):
                self._document.ui_state.pop(self._ACTIVE_CONVERGENCE_DEFINITION_KEY, None)

    @staticmethod
    def _delete_path(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
            return
        path.unlink()

    @staticmethod
    def _unique_destination(path: Path) -> Path:
        if not path.exists():
            return path
        suffix = 1
        while True:
            candidate = path.parent / f"{path.name}_{suffix}"
            if not candidate.exists():
                return candidate
            suffix += 1

    def _archive_cutoff_directory(self, source: Path, output_root: Path | None) -> Path:
        archive_root = (output_root if output_root is not None else source.parent) / "removed_from_workflow"
        archive_root.mkdir(parents=True, exist_ok=True)
        destination = self._unique_destination(archive_root / source.name)
        return source.rename(destination)

    def _prompt_cutoff_file_disposition(
        self,
        cutoff: float,
        job_dir: Path,
        *,
        include_apply_all: bool = False,
    ) -> str | None:
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("Remove cutoff with files")
        message.setText(f"Cutoff {cutoff:.8g} eV has associated files in {job_dir.name}.")
        message.setInformativeText(
            "Choose Yes to delete those files, No to move them into a separate 'removed_from_workflow' "
            "subfolder, or Cancel to keep the cutoff entry."
        )
        buttons = (
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel
        )
        if include_apply_all:
            buttons |= QMessageBox.StandardButton.YesToAll | QMessageBox.StandardButton.NoToAll
        message.setStandardButtons(buttons)
        message.setDefaultButton(QMessageBox.StandardButton.No)
        answer = message.exec()
        if answer == QMessageBox.StandardButton.Cancel:
            return None
        if answer == QMessageBox.StandardButton.NoToAll:
            return "archive_all"
        if answer == QMessageBox.StandardButton.YesToAll:
            return "delete_all"
        if answer == QMessageBox.StandardButton.Yes:
            return "delete"
        return "archive"

    def _delete_job_records(self, job_id: UUID) -> None:
        if self._document is None:
            return

        run_ids = [run.id for run in self._document.runs.values() if run.job_id == job_id]
        result_ids = [
            run.result_id
            for run in self._document.runs.values()
            if run.job_id == job_id and run.result_id is not None
        ]

        for result_id in result_ids:
            self._document.results.pop(result_id, None)
        for run_id in run_ids:
            self._document.runs.pop(run_id, None)

        self._active_run_by_job.pop(job_id, None)
        self._document.jobs.pop(job_id, None)

    def _submit_jobs_locally(self, jobs: list[Job]) -> None:
        if self._document is None or not jobs:
            return

        defined_jobs = [job for job in self._dedupe_jobs(jobs) if job.status == JobStatus.DEFINED]
        if not defined_jobs:
            QMessageBox.information(
                self,
                "No eligible jobs",
                "Selected jobs must be in the 'defined' state for local submission.",
            )
            return

        setup = load_castep_setup(self._settings)
        if not setup.executable_path:
            QMessageBox.warning(
                self,
                "CASTEP executable missing",
                "Configure a CASTEP executable in Setup -> CASTEP before submitting jobs.",
            )
            return

        dialog = LocalSubmissionDialog(
            job_labels=[job.label for job in defined_jobs],
            mpirun_configured=bool(setup.mpirun_path),
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        config = dialog.config()
        self._local_runner.max_concurrent = config.max_concurrent if config.parallel else 1

        missing_paths: list[str] = []
        submitted = 0
        for job in defined_jobs:
            output_dir = self._job_output_directory(job)
            if output_dir is None:
                missing_paths.append(job.label)
                continue

            try:
                self._project_service.submit_job(job)
            except ValueError:
                continue
            job.num_processes = max(1, int(config.num_cores))

            run = CalculationRun(
                job_id=job.id,
                run_index=self._next_run_index(job.id),
                status=JobStatus.SUBMITTING,
            )
            self._document.runs[run.id] = run
            self._active_run_by_job[job.id] = run.id
            submitted += 1

            self._local_runner.submit(
                job_id=job.id,
                job_dir=output_dir,
                seed_name=job.label,
                castep_executable=setup.executable_path,
                mpirun_path=setup.mpirun_path,
                num_cores=config.num_cores,
            )

        self._refresh_jobs_table()

        if missing_paths:
            QMessageBox.warning(
                self,
                "Missing job directories",
                "Could not locate generated input directories for:\n"
                + "\n".join(f"- {label}" for label in missing_paths),
            )

        if submitted:
            self.statusBar().showMessage(f"Submitted {submitted} local job(s)")

    def _submit_jobs_on_cluster(
        self,
        jobs: list[Job],
        *,
        default_cluster_name: str | None = None,
    ) -> None:
        if self._document is None or not jobs:
            return

        defined_jobs = [job for job in self._dedupe_jobs(jobs) if job.status == JobStatus.DEFINED]
        if not defined_jobs:
            QMessageBox.information(
                self,
                "No eligible jobs",
                "Selected jobs must be in the 'defined' state for cluster submission.",
            )
            return

        cluster_configs = load_cluster_configs(self._settings)
        if not cluster_configs:
            QMessageBox.warning(
                self,
                "No configured clusters",
                "Configure at least one cluster in Setup -> HPC Clusters before submitting.",
            )
            return

        workflow_type = self._cluster_workflow_directory_fragment(defined_jobs)
        last_submission = self._load_last_cluster_submission_config(workflow_type=workflow_type)
        initial_cluster_name = (
            default_cluster_name
            or (last_submission.cluster_name if last_submission is not None else None)
        )

        dialog = ClusterSubmissionDialog(
            job_labels=[job.label for job in defined_jobs],
            cluster_configs=cluster_configs,
            default_cluster_name=initial_cluster_name,
            initial_config=last_submission,
            project_name=self._document.project.name,
            workflow_type=workflow_type,
            connected_cluster_name=self._cluster_runner.connected_cluster_name,
            connected_username=self._cluster_runner.connected_username,
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        submission = dialog.config()
        self._persist_last_cluster_submission_config(submission, workflow_type=workflow_type)
        password = dialog.password().strip() or None
        selected_cluster = load_cluster_config(self._settings, submission.cluster_name)
        if selected_cluster is None:
            QMessageBox.warning(self, "Cluster not found", "Selected cluster is no longer configured.")
            return
        runtime_username = submission.username.strip()
        if runtime_username and runtime_username != selected_cluster.username:
            selected_cluster = replace(selected_cluster, username=runtime_username)

        invalid_labels = self._cluster_runner.invalid_remote_job_labels([job.label for job in defined_jobs])
        if invalid_labels:
            show_detailed_warning(
                self,
                "Unsafe job labels",
                (
                    "Cluster job labels may contain only letters, numbers, dot, underscore, and hyphen.\n\n"
                    "Rename the following jobs before cluster submission:\n"
                    + "\n".join(f"- {label}" for label in invalid_labels)
                ),
                informative_text="Remote paths and scheduler commands use these labels directly.",
            )
            return

        connected_name = (self._cluster_runner.connected_cluster_name or "").strip().casefold()
        connected_user = (self._cluster_runner.connected_username or "").strip()
        reusing_connection = (
            self._cluster_runner.is_connected
            and connected_name == selected_cluster.name.strip().casefold()
            and (not connected_user or connected_user == selected_cluster.username.strip())
        )

        try:
            self._connect_to_cluster_with_trust(selected_cluster, password=password)
        except Exception as exc:  # noqa: BLE001
            show_detailed_warning(
                self,
                "Cluster connection failed",
                format_cluster_connection_error(selected_cluster, exc),
                informative_text="Open Details for the full connection error and recovery guidance.",
            )
            return

        if not reusing_connection:
            QMessageBox.information(
                self,
                "Cluster connection successful",
                f"Connected to {selected_cluster.name} as {selected_cluster.username}.",
            )

        missing_paths: list[str] = []
        submitted = 0
        failed_submit: list[str] = []
        remote_target_by_job: dict[UUID, str] = {}
        existing_remote_targets: list[str] = []
        for index, job in enumerate(defined_jobs, start=1):
            self.statusBar().showMessage(
                f"Checking remote directories ({index}/{len(defined_jobs)}): {job.label}"
            )
            QApplication.processEvents()
            remote_target = self._cluster_runner.remote_job_directory(
                working_directory=submission.working_directory,
                seed_name=job.label,
            )
            remote_target_by_job[job.id] = remote_target
            try:
                if self._cluster_runner.remote_directory_exists(remote_target):
                    existing_remote_targets.append(remote_target)
            except Exception as exc:  # noqa: BLE001
                failed_submit.append(f"- {job.label}: Failed to check remote directory ({exc})")

        overwrite_mode: str | None = None
        for job in defined_jobs:
            self.statusBar().showMessage(
                f"Submitting cluster jobs ({submitted + 1}/{len(defined_jobs)}): {job.label}"
            )
            QApplication.processEvents()

            output_dir = self._job_output_directory(job)
            if output_dir is None:
                missing_paths.append(job.label)
                continue

            remote_target = remote_target_by_job.get(job.id)
            if remote_target and remote_target in existing_remote_targets:
                if overwrite_mode is None:
                    choice = self._prompt_remote_directory_overwrite(job.label, remote_target)
                    if choice == "yes_to_all":
                        confirmed = self._confirm_bulk_remote_delete(existing_remote_targets)
                        overwrite_mode = "yes_to_all" if confirmed else "no_to_all"
                        choice = overwrite_mode
                    elif choice == "no_to_all":
                        overwrite_mode = "no_to_all"
                else:
                    choice = overwrite_mode

                if choice in {"yes", "yes_to_all"}:
                    try:
                        self._cluster_runner.remove_remote_directory(remote_target)
                    except Exception as exc:  # noqa: BLE001
                        failed_submit.append(f"- {job.label}: Failed to delete remote directory ({exc})")
                        self._refresh_jobs_table()
                        QApplication.processEvents()
                        continue

            try:
                self._project_service.submit_job(job)
            except ValueError:
                self._refresh_jobs_table()
                QApplication.processEvents()
                continue
            job.num_processes = max(1, int(submission.num_processes))

            run = CalculationRun(
                job_id=job.id,
                run_index=self._next_run_index(job.id),
                status=JobStatus.SUBMITTING,
            )
            self._document.runs[run.id] = run
            self._active_run_by_job[job.id] = run.id

            job.execution_target = selected_cluster.name
            try:
                remote_job_id = self._cluster_runner.submit(
                    job_id=job.id,
                    local_job_dir=output_dir,
                    seed_name=job.label,
                    num_processes=submission.num_processes,
                    walltime=submission.walltime,
                    working_directory=submission.working_directory,
                    template_script_path=selected_cluster.template_script_path,
                    cores_per_node=selected_cluster.cores_per_node,
                    memory_gb=getattr(submission, "memory_gb", None),
                    queue_name=getattr(submission, "queue_name", ""),
                    comment=getattr(submission, "comment", ""),
                    castep_executable=selected_cluster.castep_executable,
                )
                job.remote_job_id = remote_job_id
                job.remote_directory = remote_target
                if job.status == JobStatus.SUBMITTING:
                    self._project_service.mark_job_queued(job, remote_job_id=remote_job_id)
                run.status = job.status
                submitted += 1
            except Exception as exc:  # noqa: BLE001
                failed_submit.append(f"- {job.label}: {exc}")
                try:
                    self._project_service.fail_job(job, str(exc))
                except ValueError:
                    pass
                run.status = JobStatus.FAILED
                run.completed_at = utc_now()
            self._refresh_jobs_table()
            QApplication.processEvents()

        self._refresh_jobs_table()

        if missing_paths:
            QMessageBox.warning(
                self,
                "Missing job directories",
                "Could not locate generated input directories for:\n"
                + "\n".join(f"- {label}" for label in missing_paths),
            )

        if failed_submit:
            show_detailed_warning(
                self,
                "Cluster submission failed",
                "Some jobs failed to submit:\n" + "\n".join(failed_submit),
                informative_text="Open Details to inspect the full submission failures.",
            )

        if submitted:
            self.statusBar().showMessage(
                f"Submitted {submitted} cluster job(s) on {selected_cluster.name}"
            )

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

        persisted_workflow_type = str(payload.get("workflow_type", "")).strip()
        same_workflow = (
            not workflow_type
            or not persisted_workflow_type
            or workflow_type == persisted_workflow_type
        )

        return ClusterSubmissionConfig(
            cluster_name=cluster_name,
            working_directory=str(payload.get("working_directory", "")).strip() if same_workflow else "",
            username=str(payload.get("username", "")).strip(),
            job_name_prefix=(
                str(payload.get("job_name_prefix", "convergence")).strip() if same_workflow else "convergence"
            )
            or "convergence",
            num_processes=num_processes,
            walltime=str(payload.get("walltime", "")).strip(),
            queue_name=str(payload.get("queue_name", "")).strip(),
            memory_gb=memory_gb,
            comment=str(payload.get("comment", "")).strip(),
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
            "job_name_prefix": config.job_name_prefix,
            "num_processes": int(config.num_processes),
            "walltime": config.walltime,
            "queue_name": str(getattr(config, "queue_name", "")),
            "memory_gb": getattr(config, "memory_gb", None),
            "comment": str(getattr(config, "comment", "")),
            "workflow_type": workflow_type,
        }

    def _prompt_remote_directory_overwrite(self, job_label: str, remote_dir: str) -> str:
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("Remote directory exists")
        message.setText(f"Remote directory already exists for {job_label}.")
        message.setInformativeText(
            f"Directory: {remote_dir}\n\nDelete it before submission?"
        )
        message.setStandardButtons(
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.YesToAll
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.NoToAll
        )
        message.setDefaultButton(QMessageBox.StandardButton.No)
        response = message.exec()
        if response == QMessageBox.StandardButton.Yes:
            return "yes"
        if response == QMessageBox.StandardButton.YesToAll:
            return "yes_to_all"
        if response == QMessageBox.StandardButton.NoToAll:
            return "no_to_all"
        return "no"

    def _confirm_bulk_remote_delete(self, remote_dirs: list[str]) -> bool:
        if not remote_dirs:
            return True

        details = "\n".join(f"- {path}" for path in sorted(set(remote_dirs)))
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("Confirm remote deletion")
        message.setText("Delete all existing remote job directories?")
        message.setInformativeText(
            "The following remote directories will be deleted before submission."
        )
        message.setDetailedText(details)
        message.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        message.setDefaultButton(QMessageBox.StandardButton.No)
        return message.exec() == QMessageBox.StandardButton.Yes

    @staticmethod
    def _sanitize_remote_path_component(value: str) -> str:
        cleaned = re.sub(r"\s+", "_", value.strip())
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
        return cleaned.strip("_") or "workflow"

    def _cluster_workflow_directory_fragment(self, jobs: list[Job]) -> str:
        workflow_values = {job.workflow_stage.value for job in jobs}
        if len(workflow_values) == 1:
            return self._sanitize_remote_path_component(next(iter(workflow_values)))
        return "mixed_workflows"

    def _cancel_jobs(self, jobs: list[Job]) -> None:
        if self._document is None:
            return

        cancelled = 0
        for job in jobs:
            if job.status not in _ACTIVE_LOCAL_JOB_STATES:
                continue
            if job.execution_target == "local":
                self._local_runner.cancel(job.id)
            else:
                try:
                    self._cluster_runner.cancel(job.id)
                except Exception as exc:  # noqa: BLE001
                    show_detailed_warning(
                        self,
                        "Cluster cancellation failed",
                        str(exc),
                        informative_text="Open Details to inspect the full cancellation error.",
                    )
                    continue
            cancelled += 1

        if cancelled:
            self.statusBar().showMessage(f"Cancellation requested for {cancelled} job(s)")

    def _inspect_failed_jobs(self, jobs: list[Job]) -> None:
        if self._document is None:
            return

        failed_jobs = [job for job in jobs if job.status in {JobStatus.FAILED, JobStatus.STOPPED}]
        if not failed_jobs:
            QMessageBox.information(
                self,
                "No failed jobs",
                "Select one or more failed or stopped jobs to inspect details.",
            )
            return

        sections = [self._failure_diagnostics_for_job(job) for job in failed_jobs]
        details = "\n\n".join(
            (
                "=" * 80,
                *sections,
            )
        )

        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Information)
        message.setWindowTitle("Failed job diagnostics")
        message.setText(f"Failure details for {len(failed_jobs)} job(s).")
        message.setInformativeText("Open details to inspect available error/log file content.")
        message.setDetailedText(details)
        message.exec()

    def _reset_failed_jobs(self, jobs: list[Job]) -> None:
        if self._document is None:
            return

        resettable_jobs = [
            job for job in jobs if job.status in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
        ]
        if not resettable_jobs:
            QMessageBox.information(
                self,
                "No resettable jobs",
                "Only jobs in the 'failed', 'stopped', or 'cancelled' state can be reset.",
            )
            return

        cleanup_choice = self._prompt_failed_job_cleanup(len(resettable_jobs))
        if cleanup_choice is None:
            return

        cleanup_errors: list[str] = []
        reset_count = 0
        for job in resettable_jobs:
            if cleanup_choice:
                output_dir = self._job_output_directory(job)
                if output_dir is not None and output_dir.exists() and output_dir.is_dir():
                    try:
                        self._clear_directory_contents(output_dir)
                    except OSError as exc:
                        cleanup_errors.append(f"- {job.label}: {exc}")

            try:
                self._project_service.reset_failed_job(job)
            except ValueError:
                continue

            if job.workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF:
                self._detach_convergence_job(job.id)

            self._active_run_by_job.pop(job.id, None)
            reset_count += 1

        self._refresh_jobs_table()

        if cleanup_errors:
            QMessageBox.warning(
                self,
                "Directory cleanup issues",
                "Some job directories could not be fully cleaned:\n" + "\n".join(cleanup_errors),
            )

        if reset_count:
            if cleanup_choice:
                self.statusBar().showMessage(f"Reset {reset_count} failed job(s) and cleaned directories")
            else:
                self.statusBar().showMessage(f"Reset {reset_count} failed job(s)")

    @staticmethod
    def _dedupe_jobs(jobs: list[Job]) -> list[Job]:
        deduped: list[Job] = []
        seen: set[UUID] = set()
        for job in jobs:
            if job.id in seen:
                continue
            seen.add(job.id)
            deduped.append(job)
        return deduped

    def _failure_diagnostics_for_job(self, job: Job) -> str:
        lines = [
            f"Job: {job.label}",
            f"Workflow: {job.workflow_stage.value}",
            f"Status: {job.status.value}",
        ]

        if job.error_message:
            lines.append(f"Recorded failure message: {job.error_message}")

        output_dir = self._job_output_directory(job)
        if output_dir is None:
            lines.append("Output directory: not found")
            return "\n".join(lines)

        lines.append(f"Output directory: {output_dir}")

        candidates = self._error_file_candidates(job, output_dir)
        if not candidates:
            lines.append("No .err files were found in the output directory.")
            return "\n".join(lines)

        for path in candidates:
            lines.append("")
            lines.append(f"[{path.name}]")
            lines.append(self._read_error_contexts(path))

        return "\n".join(lines)

    def _error_file_candidates(self, job: Job, output_dir: Path) -> list[Path]:
        ordered_names = (f"{job.label}.err",)

        discovered: list[Path] = []
        for name in ordered_names:
            candidate = output_dir / name
            if candidate.exists() and candidate.is_file():
                discovered.append(candidate)

        if discovered:
            return discovered

        fallback_patterns = ("*.err",)
        fallback: list[Path] = []
        for pattern in fallback_patterns:
            fallback.extend(sorted(path for path in output_dir.glob(pattern) if path.is_file()))
        return fallback

    @staticmethod
    def _read_error_contexts(path: Path, *, context_lines: int = 2, max_matches: int = 8) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"Could not read file: {exc}"

        if not lines:
            return "[File is empty]"

        marker_re = re.compile(r"error|failed|exception|traceback|warning", re.IGNORECASE)
        matched_indexes = [index for index, line in enumerate(lines) if marker_re.search(line)]
        if not matched_indexes:
            tail = lines[-min(len(lines), 20):]
            start_line = len(lines) - len(tail) + 1
            return "\n".join(
                f"{index}: {line}" for index, line in enumerate(tail, start=start_line)
            )

        excerpts: list[str] = []
        covered: set[int] = set()
        for match_index in matched_indexes[:max_matches]:
            start = max(0, match_index - context_lines)
            stop = min(len(lines), match_index + context_lines + 1)
            if any(index in covered for index in range(start, stop)):
                continue
            covered.update(range(start, stop))
            excerpt = "\n".join(
                f"{index + 1}: {lines[index]}" for index in range(start, stop)
            )
            excerpts.append(excerpt)

        return "\n\n...\n\n".join(excerpts)

    @staticmethod
    def _read_text_excerpt(path: Path, *, max_chars: int = 12000) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Could not read file: {exc}"

        stripped = text.strip()
        if not stripped:
            return "[File is empty]"
        if len(stripped) <= max_chars:
            return stripped
        return stripped[:max_chars] + "\n... [truncated]"

    def _prompt_failed_job_cleanup(self, job_count: int) -> bool | None:
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Question)
        message.setWindowTitle("Reset failed jobs")
        message.setText(f"Reset {job_count} failed job(s) to the 'defined' state?")
        message.setInformativeText(
            "Choose Yes to clean each job directory before reset, "
            "No to keep existing files, or Cancel to abort."
        )
        message.setStandardButtons(
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel
        )
        message.setDefaultButton(QMessageBox.StandardButton.No)
        answer = message.exec()
        if answer == QMessageBox.StandardButton.Cancel:
            return None
        return answer == QMessageBox.StandardButton.Yes

    @staticmethod
    def _clear_directory_contents(directory: Path) -> None:
        for child in directory.iterdir():
            if child.is_file() and child.suffix.lower() in {".cell", ".param"}:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def _next_run_index(self, job_id: UUID) -> int:
        if self._document is None:
            return 1
        indices = [run.run_index for run in self._document.runs.values() if run.job_id == job_id]
        if not indices:
            return 1
        return max(indices) + 1

    def _run_for_job(self, job_id: UUID) -> CalculationRun | None:
        if self._document is None:
            return None

        run_id = self._active_run_by_job.get(job_id)
        if run_id is not None:
            return self._document.runs.get(run_id)

        runs = [run for run in self._document.runs.values() if run.job_id == job_id]
        if not runs:
            return None
        return max(runs, key=lambda run: run.run_index)

    def _job_output_directory(self, job: Job) -> Path | None:
        if self._document is None:
            return None

        for definition in self._document.calculations.values():
            params = definition.parameters
            job_ids_by_cutoff = params.get("job_ids_by_cutoff", {})
            if isinstance(job_ids_by_cutoff, dict) and str(job.id) in {
                str(value) for value in job_ids_by_cutoff.values()
            }:
                base = self._definition_output_directory(definition)
                if base is not None:
                    return base / job.label

            labels = params.get("job_labels", [])
            if not isinstance(labels, list) or job.label not in labels:
                continue

            raw_output_dir = params.get("output_directory")
            if not isinstance(raw_output_dir, str) or not raw_output_dir.strip():
                continue
            base = Path(raw_output_dir)
            if not base.is_absolute():
                base = self._document.project_root / base
            candidate = base / job.label
            if candidate.exists() and candidate.is_dir():
                return candidate

        fallback_candidates = (
            self._document.project_root / "files" / job.workflow_stage.value / job.label,
            self._document.project_root / "files" / "convergence_cutoff" / job.label,
            self._document.project_root / "files" / job.label,
        )
        for fallback in fallback_candidates:
            if fallback.exists() and fallback.is_dir():
                return fallback
        return None

    def _output_files_for_job(self, job: Job) -> list[str]:
        if self._document is None:
            return []
        output_dir = self._job_output_directory(job)
        if output_dir is None:
            return []

        paths = [
            output_dir / f"{job.label}.cell",
            output_dir / f"{job.label}.param",
            output_dir / f"{job.label}.castep",
            output_dir / f"{job.label}-out.cell",
        ]
        files: list[str] = []
        for path in paths:
            if not path.exists():
                continue
            try:
                files.append(str(path.relative_to(self._document.project_root)))
            except ValueError:
                files.append(str(path))
        return files

    def _on_local_job_status_changed(self, job_id: UUID, status_value: str) -> None:
        if self._document is None:
            return
        job = self._document.jobs.get(job_id)
        if job is None:
            return

        if status_value == JobStatus.QUEUED.value and job.status == JobStatus.SUBMITTING:
            try:
                self._project_service.mark_job_queued(job, remote_job_id=None)
            except ValueError:
                return
            run = self._run_for_job(job_id)
            if run is not None:
                run.status = JobStatus.QUEUED
            self._refresh_jobs_table()

    def _on_local_job_started(self, job_id: UUID) -> None:
        if self._document is None:
            return
        job = self._document.jobs.get(job_id)
        if job is None:
            return

        if job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED}:
            try:
                self._project_service.start_job(job)
            except ValueError:
                pass

        run = self._run_for_job(job_id)
        if run is not None:
            run.status = JobStatus.RUNNING
            if run.started_at is None:
                run.started_at = utc_now()

        self._refresh_jobs_table()

    def _on_local_job_finished(self, job_id: UUID, parsed: CastepResult) -> None:
        if self._document is None:
            return
        job = self._document.jobs.get(job_id)
        if job is None:
            return

        if job.calc_type == "castep_pristine_relaxation" and parsed.geom_converged is False:
            self._on_local_job_failed(
                job_id,
                parsed.error_message
                or "Geometry optimization did not satisfy CASTEP convergence criteria.",
            )
            return

        run = self._run_for_job(job_id)
        if run is None:
            run = CalculationRun(
                job_id=job.id,
                run_index=self._next_run_index(job.id),
                status=JobStatus.RUNNING,
                started_at=utc_now(),
            )
            self._document.runs[run.id] = run

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

        derived_structure_ids: list[UUID] = []
        if job.calc_type == "castep_pristine_relaxation":
            output_dir = self._job_output_directory(job)
            out_cell_path = None if output_dir is None else output_dir / f"{job.label}-out.cell"
            if out_cell_path is None or not out_cell_path.exists():
                self._on_local_job_failed(
                    job_id,
                    f"Expected relaxed structure output was not found: {job.label}-out.cell",
                )
                return

            try:
                relaxed = parse_cell_file(out_cell_path)
            except CellParseError as exc:
                self._on_local_job_failed(job_id, f"Failed to parse relaxed CELL output: {exc}")
                return

            lattice_a, lattice_b, lattice_c = lattice_parameters(relaxed.lattice_vectors)[:3]
            result_payload["lattice_a"] = lattice_a
            result_payload["lattice_b"] = lattice_b
            result_payload["lattice_c"] = lattice_c

            definition = self._active_pristine_relaxation_definition()
            parent_structure_id = (
                definition.input_structure_ids[0]
                if definition is not None and definition.input_structure_ids
                else None
            )
            relaxed_structure = Structure(
                label=f"{job.label} relaxed",
                origin="pristine_relaxation",
                lattice_vectors=[list(row) for row in relaxed.lattice_vectors],
                species=list(relaxed.species),
                fractional_positions=[list(row) for row in relaxed.fractional_positions],
                parent_structure_id=parent_structure_id,
                properties={
                    "source_job_id": str(job.id),
                    "source_workflow_stage": WorkflowStage.PRISTINE_RELAXATION.value,
                },
            )
            self._project_service.add_structure(self._document, relaxed_structure)
            derived_structure_ids.append(relaxed_structure.id)

        result = Result(
            calculation_run_id=run.id,
            result_type=job.calc_type,
            data=result_payload,
            output_files=self._output_files_for_job(job),
            derived_structure_ids=derived_structure_ids,
        )
        self._document.results[result.id] = result

        run.result_id = result.id
        run.status = JobStatus.COMPLETED
        if run.started_at is None:
            run.started_at = utc_now()
        run.completed_at = utc_now()
        run.warnings = list(parsed.warnings)
        run.convergence_achieved = parsed.geom_converged

        if job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED}:
            try:
                self._project_service.start_job(job)
            except ValueError:
                pass
        if job.status in {JobStatus.RUNNING, JobStatus.RETRIEVING}:
            try:
                self._project_service.complete_job(job)
            except ValueError:
                pass

        self._active_run_by_job.pop(job_id, None)
        self._refresh_jobs_table()
        self.statusBar().showMessage(f"Completed local job {job.label}")

    def _on_local_job_failed(self, job_id: UUID, message: str) -> None:
        if self._document is None:
            return
        job = self._document.jobs.get(job_id)
        if job is None:
            return

        if job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING}:
            try:
                self._project_service.fail_job(job, message)
            except ValueError:
                pass

        run = self._run_for_job(job_id)
        if run is not None:
            if run.started_at is None:
                run.started_at = utc_now()
            run.completed_at = utc_now()
            run.status = JobStatus.FAILED
            run.warnings = [message]

        self._active_run_by_job.pop(job_id, None)
        self._refresh_jobs_table()
        self.statusBar().showMessage(f"Local job failed: {job.label}")

    def _on_local_job_cancelled(self, job_id: UUID) -> None:
        if self._document is None:
            return
        job = self._document.jobs.get(job_id)
        if job is None:
            return

        if job.status in _ACTIVE_LOCAL_JOB_STATES | {JobStatus.DEFINED}:
            try:
                self._project_service.cancel_job(job)
            except ValueError:
                pass

        run = self._run_for_job(job_id)
        if run is not None:
            run.completed_at = utc_now()
            run.status = JobStatus.CANCELLED

        self._active_run_by_job.pop(job_id, None)
        self._refresh_jobs_table()
        self.statusBar().showMessage(f"Cancelled local job {job.label}")

    def _on_cluster_job_status_changed(self, job_id: UUID, status_value: str) -> None:
        if self._document is None:
            return
        job = self._document.jobs.get(job_id)
        if job is None:
            return

        status_map = {
            JobStatus.QUEUED.value: JobStatus.QUEUED,
            JobStatus.RUNNING.value: JobStatus.RUNNING,
            JobStatus.RETRIEVING.value: JobStatus.RETRIEVING,
            JobStatus.STOPPED.value: JobStatus.STOPPED,
        }
        mapped = status_map.get(status_value)
        if mapped is None:
            return

        try:
            if mapped == JobStatus.QUEUED and job.status == JobStatus.SUBMITTING:
                self._project_service.mark_job_queued(job, remote_job_id=job.remote_job_id)
            elif mapped == JobStatus.RUNNING and job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED}:
                self._project_service.start_job(job)
            elif mapped == JobStatus.RETRIEVING and job.status == JobStatus.RUNNING:
                self._project_service.mark_job_retrieving(job)
            elif mapped == JobStatus.STOPPED and job.status in {JobStatus.RUNNING, JobStatus.RETRIEVING}:
                self._project_service.stop_job(job, "Job stopped before successful completion.")
        except ValueError:
            return

        run = self._run_for_job(job_id)
        if run is not None:
            run.status = mapped
            if mapped == JobStatus.RUNNING and run.started_at is None:
                run.started_at = utc_now()
            if mapped == JobStatus.STOPPED:
                run.completed_at = utc_now()

        self._refresh_jobs_table()

    def _on_cluster_job_started(self, job_id: UUID) -> None:
        if self._document is None:
            return
        job = self._document.jobs.get(job_id)
        if job is None:
            return

        if job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED}:
            try:
                self._project_service.start_job(job)
            except ValueError:
                pass

        run = self._run_for_job(job_id)
        if run is not None:
            run.status = JobStatus.RUNNING
            if run.started_at is None:
                run.started_at = utc_now()

        self._refresh_jobs_table()

    def _on_cluster_job_finished(self, job_id: UUID, parsed: CastepResult) -> None:
        self._on_local_job_finished(job_id, parsed)
        if self._document is None:
            return
        job = self._document.jobs.get(job_id)
        if job is not None:
            self.statusBar().showMessage(f"Completed cluster job {job.label}")

    def _on_cluster_job_failed(self, job_id: UUID, message: str) -> None:
        self._on_local_job_failed(job_id, message)
        if self._document is None:
            return
        job = self._document.jobs.get(job_id)
        if job is not None:
            self.statusBar().showMessage(f"Cluster job failed: {job.label}")

    def _on_cluster_job_cancelled(self, job_id: UUID) -> None:
        self._on_local_job_cancelled(job_id)
        if self._document is None:
            return
        job = self._document.jobs.get(job_id)
        if job is not None:
            self.statusBar().showMessage(f"Cancelled cluster job {job.label}")

    def _on_cluster_connection_status_changed(self, connected: bool) -> None:
        self._cluster_connected = bool(connected)
        if connected:
            cluster_name = self._cluster_runner.connected_cluster_name or "unknown"
            self._cluster_connection_label.setText(f"Cluster: connected ({cluster_name})")
            self._refresh_cluster_connection_label_colour()
            self._cluster_refresh_button.setEnabled(True)
            return

        self._cluster_connection_label.setText("Cluster: disconnected")
        self._refresh_cluster_connection_label_colour()
        self._cluster_queue_checked_label.setText("Queue last checked: disconnected")
        self._cluster_refresh_button.setEnabled(True)

    def _refresh_cluster_connection_label_colour(self) -> None:
        palette = current_palette()
        colour = palette.success if self._cluster_connected else palette.danger
        self._cluster_connection_label.setStyleSheet(f"color: {colour};")

    def _on_cluster_queue_checked(self, checked_at) -> None:
        if hasattr(checked_at, "strftime"):
            localized = checked_at
            if getattr(checked_at, "tzinfo", None) is not None and hasattr(checked_at, "astimezone"):
                try:
                    localized = checked_at.astimezone()
                except Exception:
                    localized = checked_at
            stamp = localized.strftime("%H:%M:%S")
        else:
            stamp = str(checked_at)
        self._cluster_queue_checked_label.setText(f"Queue last checked: {stamp}")

    def _on_cluster_poll_failed(self, message: str) -> None:
        show_detailed_warning(
            self,
            "Queue polling failed",
            (
                f"{message}\n\n"
                "Check VPN/network access, SSH trust, authentication, and scheduler command paths."
            ),
            informative_text="Open Details to inspect the full polling failure.",
        )

    def _connect_to_cluster_with_trust(self, config: ClusterConfig, *, password: str | None) -> None:
        try:
            self._cluster_runner.connect_to_cluster(config, password=password)
            return
        except SSHUnknownHostKeyError as exc:
            if not prompt_to_trust_host_key(self, config, exc):
                raise ConnectionError("Connection cancelled because the SSH host key was not trusted.") from exc
            save_trusted_host_key(
                self._settings,
                config,
                algorithm=exc.algorithm,
                key_base64=exc.key_base64,
            )
        self._cluster_runner.connect_to_cluster(config, password=password)

    def _on_refresh_cluster_queue(self) -> None:
        try:
            self._sync_cluster_tracking_from_document()
            if not self._cluster_runner.is_connected:
                self._refresh_queue_for_active_dashboard_clusters()
                return
            self._cluster_runner.refresh_queue()
        except Exception as exc:  # noqa: BLE001
            show_detailed_warning(
                self,
                "Queue refresh failed",
                str(exc),
                informative_text="Open Details to inspect the full queue refresh error.",
            )

    def _refresh_queue_for_active_dashboard_clusters(self) -> None:
        if self._document is None:
            self._cluster_runner.refresh_queue()
            return

        target_clusters = self._active_dashboard_cluster_targets()
        if not target_clusters:
            self._cluster_runner.refresh_queue()
            return

        configured = load_cluster_configs(self._settings)
        by_name = {config.name.strip().casefold(): config for config in configured}

        refreshed: list[str] = []
        failures: list[str] = []
        for target in target_clusters:
            config = by_name.get(target.casefold())
            if config is None:
                failures.append(
                    f"- {target}: cluster is not configured. Set it in Setup -> HPC Clusters."
                )
                continue
            try:
                self._connect_to_cluster_with_trust(config, password=None)
                self._cluster_runner.refresh_queue()
                refreshed.append(config.name)
            except Exception as exc:  # noqa: BLE001
                if config.auth_method == "password":
                    raw_password, accepted = QInputDialog.getText(
                        self,
                        f"Reconnect to {config.name}",
                        f"Enter password for {config.username}@{config.hostname}:",
                        QLineEdit.EchoMode.Password,
                    )
                    if not accepted or not raw_password.strip():
                        failures.append(
                            f"- {config.name}: Password required for reconnect. Enter it when prompted."
                        )
                        continue
                    try:
                        self._connect_to_cluster_with_trust(config, password=raw_password.strip())
                        self._cluster_runner.refresh_queue()
                        refreshed.append(config.name)
                        continue
                    except Exception as retry_exc:  # noqa: BLE001
                        failures.append(
                            f"- {config.name}: {format_cluster_connection_error(config, retry_exc)}"
                        )
                        continue
                failures.append(f"- {config.name}: {format_cluster_connection_error(config, exc)}")

        if refreshed:
            self.statusBar().showMessage(
                "Queue refreshed for clusters: " + ", ".join(sorted(set(refreshed)))
            )

        if failures:
            show_detailed_warning(
                self,
                "Queue refresh failed",
                "Some clusters could not be refreshed:\n" + "\n".join(failures),
                informative_text="Open Details to inspect the full queue refresh failures.",
            )

    def _active_dashboard_cluster_targets(self) -> list[str]:
        if self._document is None:
            return []

        targets: list[str] = []
        seen: set[str] = set()
        for job in self._document.jobs.values():
            if job.status not in _ACTIVE_LOCAL_JOB_STATES:
                continue
            target = job.execution_target.strip()
            if not target or target.casefold() == "local":
                continue
            key = target.casefold()
            if key in seen:
                continue
            seen.add(key)
            targets.append(target)
        return targets

    def _sync_cluster_tracking_from_document(self) -> None:
        if self._document is None:
            return

        for job in self._document.jobs.values():
            if job.status not in _ACTIVE_LOCAL_JOB_STATES:
                continue

            target = job.execution_target.strip()
            if not target or target.casefold() == "local":
                continue

            remote_job_id = (job.remote_job_id or "").strip()
            if not remote_job_id:
                continue

            remote_dir = (job.remote_directory or "").strip()
            if not remote_dir:
                remote_dir = self._infer_remote_job_directory(job, target)
                if remote_dir:
                    job.remote_directory = remote_dir
            if not remote_dir:
                continue

            output_dir = self._job_output_directory(job)
            if output_dir is None:
                continue

            self._cluster_runner.track_existing_job(
                job_id=job.id,
                remote_job_id=remote_job_id,
                remote_dir=remote_dir,
                seed_name=job.label,
                local_output_dir=output_dir,
                cluster_name=target,
                status=job.status,
            )

    def _infer_remote_job_directory(self, job: Job, cluster_name: str) -> str:
        config = load_cluster_config(self._settings, cluster_name)
        if config is None:
            return ""

        working_directory = config.default_working_directory.strip()
        if not working_directory:
            return ""

        try:
            return self._cluster_runner.remote_job_directory(
                working_directory=working_directory,
                seed_name=job.label,
            )
        except Exception:
            return ""

    def _on_convergence_configured(self, config: ConvergenceConfig) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return

        resolution = self._project_service.resolve_workflow_structure(
            self._document,
            WorkflowStage.CONVERGENCE_CUTOFF,
            normalize_state=True,
        )
        structure = resolution.structure
        if structure is None:
            QMessageBox.warning(
                self,
                "No host structure",
                "Load a host structure before generating convergence jobs.",
            )
            return

        try:
            structure_spec = structure_to_spec(structure)
            expand_convergence_jobs(structure=structure_spec, config=config)
        except (ValueError, StructureBridgeError) as exc:
            QMessageBox.warning(self, "Convergence setup invalid", str(exc))
            return

        output_root = self._document.project_root / "files" / "convergence_cutoff"

        self._project_service.register_convergence_cutoff_definition(
            self._document,
            input_structure_id=structure.id,
            parameters=self._serialize_convergence_config(config),
            output_directory=output_root,
            simulation_cell_mode=resolution.normalized_selection.mode,
            supercell_source_workflow=(
                resolution.source_workflow_stage.value
                if resolution.source_workflow_stage is not None
                else None
            ),
        )
        latest_definition = max(
            (
                definition
                for definition in self._document.calculations.values()
                if definition.workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF
            ),
            key=lambda definition: definition.created_at,
        )
        self._document.ui_state[self._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(latest_definition.id)

        self._refresh_jobs_table()
        self._refresh_convergence_cutoff_rows()
        self.statusBar().showMessage(
            f"Configured {len(config.cutoff_energies)} cutoff jobs; input files will be generated on submission"
        )

    def _on_load_structure_requested(self) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
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
        self._refresh_convergence_simulation_cell()
        self._refresh_kpoint_convergence_simulation_cell()
        self._refresh_pristine_relaxation_defaults()
        self._refresh_pristine_relaxation_rows()
        self._refresh_convergence_overview()
        self._populate_project_tree()
        self.statusBar().showMessage(
            f"Loaded host structure '{structure.label}' ({len(structure.species)} atoms)"
        )

    def _on_generate_requested(self, method: str, parameters: dict) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
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

        self._refresh_muon_generation_table()
        requested = int(parameters.get("count", 1)) if method != "manual" else 1
        suffix = ""
        if len(created) < requested:
            suffix = f" (requested {requested}; constraints prevented full count)"
        self.statusBar().showMessage(f"Generated {len(created)} candidate muon site(s){suffix}")

    def _on_muon_form_state_changed(self, state: dict) -> None:
        if self._document is None:
            return
        self._document.ui_state["muon_generation_form"] = state

    def _on_convergence_form_state_changed(self, state: dict) -> None:
        if self._document is None:
            return
        self._document.ui_state["convergence_form"] = state
        self._refresh_convergence_cutoff_rows()

    def _refresh_convergence_simulation_cell(self) -> None:
        if self._document is None:
            self._convergence_view.set_simulation_cell_options(
                summary="Load a host structure to choose a simulation cell.",
                assets=[],
                selected_structure_id=None,
                enabled=False,
            )
            return

        resolution = self._project_service.resolve_workflow_structure(
            self._document,
            WorkflowStage.CONVERGENCE_CUTOFF,
            normalize_state=True,
        )
        assets = [
            {
                "structure_id": str(asset.structure_id),
                "label": asset.label,
            }
            for asset in resolution.available_assets
        ]
        selected_structure_id = (
            str(resolution.normalized_selection.structure_id)
            if resolution.normalized_selection.structure_id is not None
            else None
        )
        self._convergence_view.set_simulation_cell_options(
            summary=resolution.status_message,
            assets=assets,
            selected_structure_id=selected_structure_id,
            enabled=resolution.structure is not None,
        )

    def _convergence_supercell_initial_matrix(self) -> list[list[int]]:
        if self._document is None:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        resolution = self._project_service.resolve_workflow_structure(
            self._document,
            WorkflowStage.CONVERGENCE_CUTOFF,
            normalize_state=True,
        )
        if resolution.selected_asset is not None:
            return [list(row) for row in resolution.selected_asset.matrix]
        return [list(row) for row in self._document.project.supercell_matrix]

    def _on_convergence_simulation_cell_create_requested(self) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return

        host = self._project_service.active_host_structure(self._document)
        if host is None:
            QMessageBox.warning(
                self,
                "No host structure",
                "Load a host structure before creating a saved supercell.",
            )
            return

        dialog = SupercellPreviewDialog(
            structure=host,
            atomic_spins=self._project_service.structure_atomic_spins(host),
            propagation_vector=self._project_service.structure_propagation_vector(host),
            initial_matrix=self._convergence_supercell_initial_matrix(),
            accept_button_text="Save Supercell",
            show_unit_cell_button=True,
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        if dialog.result_mode() == "unit_cell":
            self._project_service.clear_workflow_supercell_selection(
                self._document,
                WorkflowStage.CONVERGENCE_CUTOFF,
            )
            self._refresh_convergence_simulation_cell()
            self.statusBar().showMessage("Convergence workflow reset to the single unit cell")
            return

        saved = self._project_service.create_workflow_supercell_asset(
            self._document,
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            host_structure_id=host.id,
            matrix=dialog.matrix(),
            reuse_existing=False,
        )
        self._project_service.use_workflow_supercell_asset(
            self._document,
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            asset_structure_id=saved.id,
        )
        self._refresh_convergence_simulation_cell()
        self.statusBar().showMessage(f"Saved convergence supercell '{saved.label}'")

    def _on_convergence_simulation_cell_selection_requested(self, payload: object) -> None:
        if self._document is None:
            return

        structure_id = None
        if payload is not None:
            try:
                structure_id = UUID(str(payload))
            except ValueError:
                structure_id = None

        if structure_id is None:
            self._project_service.clear_workflow_supercell_selection(
                self._document,
                WorkflowStage.CONVERGENCE_CUTOFF,
            )
            self._refresh_convergence_simulation_cell()
            return

        try:
            selected = self._project_service.use_workflow_supercell_asset(
                self._document,
                workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
                asset_structure_id=structure_id,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Simulation cell unavailable", str(exc))
            self._refresh_convergence_simulation_cell()
            return

        self._refresh_convergence_simulation_cell()
        self.statusBar().showMessage(f"Selected convergence simulation cell '{selected.label}'")

    def _on_convergence_simulation_cell_use_unit_cell_requested(self) -> None:
        if self._document is None:
            return
        self._project_service.clear_workflow_supercell_selection(
            self._document,
            WorkflowStage.CONVERGENCE_CUTOFF,
        )
        self._refresh_convergence_simulation_cell()

    @staticmethod
    def _grid_key(grid: tuple[int, int, int]) -> str:
        return f"{int(grid[0])}x{int(grid[1])}x{int(grid[2])}"

    @staticmethod
    def _parse_grid_key(value: str) -> tuple[int, int, int] | None:
        parts = value.strip().split("x")
        if len(parts) != 3:
            return None
        try:
            grid = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            return None
        if grid[0] <= 0 or grid[1] <= 0 or grid[2] <= 0:
            return None
        return grid

    def _kpoint_definitions(self) -> list:
        if self._document is None:
            return []
        definitions = [
            definition
            for definition in self._document.calculations.values()
            if definition.workflow_stage == WorkflowStage.CONVERGENCE_KPOINT
        ]
        definitions.sort(key=lambda definition: definition.created_at, reverse=True)
        return definitions

    def _active_kpoint_definition(self):
        if self._document is None:
            return None

        raw_definition_id = self._document.ui_state.get(self._ACTIVE_KPOINT_DEFINITION_KEY)
        if isinstance(raw_definition_id, str):
            try:
                definition_id = UUID(raw_definition_id)
            except ValueError:
                definition_id = None
            if definition_id is not None:
                definition = self._document.calculations.get(definition_id)
                if definition is not None and definition.workflow_stage == WorkflowStage.CONVERGENCE_KPOINT:
                    return definition

        definitions = self._kpoint_definitions()
        return definitions[0] if definitions else None

    def _definition_for_kpoint_grid(self, grid: tuple[int, int, int]):
        active_definition = self._active_kpoint_definition()
        if active_definition is None:
            return None, None

        config = self._deserialize_convergence_config(active_definition.parameters.get("configuration", {}))
        if config is None or config.sweep_parameter != "kpoint_mp_grid":
            return None, None

        if grid in config.sweep_values:
            return active_definition, config
        return None, None

    def _job_for_definition_grid(self, definition, grid: tuple[int, int, int]) -> Job | None:
        if self._document is None:
            return None

        mapping = definition.parameters.get("job_ids_by_grid", {})
        if isinstance(mapping, dict):
            raw_job_id = mapping.get(self._grid_key(grid))
            if isinstance(raw_job_id, str):
                try:
                    job_id = UUID(raw_job_id)
                except ValueError:
                    job_id = None
                if job_id is not None:
                    job = self._document.jobs.get(job_id)
                    if job is not None:
                        return job

        label = convergence_job_name(
            definition.parameters.get("configuration", {}).get("name_prefix", "kpoint_convergence"),
            grid,
            sweep_parameter="kpoint_mp_grid",
        )
        for job in reversed(tuple(self._document.jobs.values())):
            if job.label == label:
                return job
        return None

    def _kpoint_form_snapshot(self) -> tuple[str, tuple[tuple[int, int, int], ...]]:
        if self._document is None:
            return "kpoint_convergence", ()

        persisted = self._document.ui_state.get("kpoint_convergence_form", {})
        if not isinstance(persisted, dict):
            return "kpoint_convergence", ()

        name_prefix = str(persisted.get("name_prefix", "kpoint_convergence")).strip() or "kpoint_convergence"
        raw_grids = persisted.get("kpoint_grids", [])
        grids: list[tuple[int, int, int]] = []
        if isinstance(raw_grids, list):
            for entry in raw_grids:
                if not isinstance(entry, list) or len(entry) != 3:
                    continue
                try:
                    grid = (int(entry[0]), int(entry[1]), int(entry[2]))
                except (TypeError, ValueError):
                    continue
                grids.append(grid)
        return name_prefix, tuple(dict.fromkeys(grids))

    def _refresh_kpoint_convergence_rows(self) -> None:
        if self._document is None:
            self._kpoint_convergence_view.set_runtime_states({})
            return
        self._kpoint_convergence_view.set_runtime_states(self._build_convergence_kpoint_states())

    def _build_convergence_kpoint_states(self) -> dict[tuple[int, int, int], SweepPointRunState]:
        if self._document is None:
            return {}

        states: dict[tuple[int, int, int], SweepPointRunState] = {}
        form_prefix, form_grids = self._kpoint_form_snapshot()
        for grid in form_grids:
            states[grid] = SweepPointRunState(
                status="ready",
                job_label=convergence_job_name(
                    form_prefix,
                    grid,
                    sweep_parameter="kpoint_mp_grid",
                ),
            )

        definition = self._active_kpoint_definition()
        if definition is None:
            return states

        config = self._deserialize_convergence_config(definition.parameters.get("configuration", {}))
        if config is None or config.sweep_parameter != "kpoint_mp_grid":
            return states

        config_grids = tuple(
            tuple(int(component) for component in value)
            for value in config.sweep_values
            if isinstance(value, tuple)
        )
        if form_grids and (form_prefix != config.name_prefix or form_grids != config_grids):
            return states

        for grid in config_grids:
            job = self._job_for_definition_grid(definition, grid)
            if job is None:
                states[grid] = SweepPointRunState(
                    status="ready",
                    job_label=convergence_job_name(
                        config.name_prefix,
                        grid,
                        sweep_parameter="kpoint_mp_grid",
                    ),
                )
                continue

            status_value = "ready" if job.status == JobStatus.DEFINED else job.status.value
            states[grid] = SweepPointRunState(
                status=status_value,
                total_energy=self._total_energy_for_job(job.id),
                max_force=self._max_force_for_job(job.id),
                job_id=job.id,
                job_label=job.label,
            )
        return states

    def _jobs_for_convergence_kpoint_grids(self, grids: list[tuple[int, int, int]]) -> list[Job]:
        if self._document is None:
            return []

        states = self._build_convergence_kpoint_states()
        jobs_by_label = {job.label: job for job in self._document.jobs.values()}
        selected: list[Job] = []
        seen: set[UUID] = set()
        for grid in grids:
            state = states.get(tuple(grid))
            if state is None:
                continue
            job = self._document.jobs.get(state.job_id) if state.job_id is not None else None
            if job is None and state.job_label is not None:
                job = jobs_by_label.get(state.job_label)
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    def _materialize_kpoint_jobs(self, grids: list[tuple[int, int, int]]) -> list[Job]:
        if self._document is None:
            return []

        created: list[Job] = []
        grouped: dict[UUID, tuple[object, ConvergenceConfig, set[tuple[int, int, int]]]] = {}
        for grid in grids:
            definition, config = self._definition_for_kpoint_grid(grid)
            if definition is None or config is None:
                continue
            existing = self._job_for_definition_grid(definition, grid)
            if existing is not None:
                created.append(existing)
                continue
            record = grouped.get(definition.id)
            if record is None:
                grouped[definition.id] = (definition, config, {tuple(grid)})
            else:
                record[2].add(tuple(grid))

        for definition, config, grouped_grids in grouped.values():
            if not definition.input_structure_ids:
                continue
            structure = self._document.structures.get(definition.input_structure_ids[0])
            output_root = self._definition_output_directory(definition)
            if structure is None or output_root is None:
                continue
            self._project_service.repair_supercell_asset_if_needed(
                self._document,
                structure_id=structure.id,
            )

            try:
                structure_spec = structure_to_spec(structure)
                all_jobs = expand_convergence_jobs(structure=structure_spec, config=config)
            except (ValueError, StructureBridgeError):
                continue

            desired_labels = {
                convergence_job_name(
                    config.name_prefix,
                    grid,
                    sweep_parameter="kpoint_mp_grid",
                ): grid
                for grid in grouped_grids
            }
            mapping = definition.parameters.setdefault("job_ids_by_grid", {})
            if not isinstance(mapping, dict):
                mapping = {}
                definition.parameters["job_ids_by_grid"] = mapping

            for job_spec in all_jobs:
                grid = desired_labels.get(job_spec.name)
                if grid is None:
                    continue
                write_job_directory(job_spec, output_root)
                job = self._project_service.add_job(
                    self._document,
                    label=job_spec.name,
                    workflow_stage=job_spec.workflow_stage,
                    calc_type="castep_convergence_kpoint",
                )
                mapping[self._grid_key(grid)] = str(job.id)
                created.append(job)

        deduped: list[Job] = []
        seen: set[UUID] = set()
        for job in created:
            if job.id in seen:
                continue
            seen.add(job.id)
            deduped.append(job)
        return deduped

    def _on_kpoint_convergence_configured(self, config: ConvergenceConfig) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return

        resolution = self._project_service.resolve_workflow_structure(
            self._document,
            WorkflowStage.CONVERGENCE_KPOINT,
            normalize_state=True,
        )
        structure = resolution.structure
        if structure is None:
            QMessageBox.warning(
                self,
                "No host structure",
                "Load a host structure before generating k-point convergence jobs.",
            )
            return

        try:
            structure_spec = structure_to_spec(structure)
            expand_convergence_jobs(structure=structure_spec, config=config)
        except (ValueError, StructureBridgeError) as exc:
            QMessageBox.warning(self, "Convergence setup invalid", str(exc))
            return

        output_root = self._document.project_root / "files" / "convergence_kpoint"

        self._project_service.register_convergence_kpoint_definition(
            self._document,
            input_structure_id=structure.id,
            parameters=self._serialize_convergence_config(config),
            output_directory=output_root,
            simulation_cell_mode=resolution.normalized_selection.mode,
            supercell_source_workflow=(
                resolution.source_workflow_stage.value
                if resolution.source_workflow_stage is not None
                else None
            ),
        )
        latest_definition = max(
            (
                definition
                for definition in self._document.calculations.values()
                if definition.workflow_stage == WorkflowStage.CONVERGENCE_KPOINT
            ),
            key=lambda definition: definition.created_at,
        )
        self._document.ui_state[self._ACTIVE_KPOINT_DEFINITION_KEY] = str(latest_definition.id)

        self._refresh_jobs_table()
        self._refresh_kpoint_convergence_rows()
        self.statusBar().showMessage(
            f"Configured {len(config.sweep_values)} k-point jobs; input files will be generated on submission"
        )

    def _on_kpoint_form_state_changed(self, state: dict) -> None:
        if self._document is None:
            return
        self._document.ui_state["kpoint_convergence_form"] = state
        self._refresh_kpoint_convergence_rows()

    def _refresh_kpoint_convergence_simulation_cell(self) -> None:
        if self._document is None:
            self._kpoint_convergence_view.set_simulation_cell_options(
                summary="Load a host structure to choose a simulation cell.",
                assets=[],
                selected_structure_id=None,
                enabled=False,
            )
            self._kpoint_convergence_view.set_generation_context(structure_id=None, lattice_vectors=None)
            return

        resolution = self._project_service.resolve_workflow_structure(
            self._document,
            WorkflowStage.CONVERGENCE_KPOINT,
            normalize_state=True,
        )
        assets = [
            {
                "structure_id": str(asset.structure_id),
                "label": asset.label,
            }
            for asset in resolution.available_assets
        ]
        selected_structure_id = (
            str(resolution.normalized_selection.structure_id)
            if resolution.normalized_selection.structure_id is not None
            else None
        )
        self._kpoint_convergence_view.set_simulation_cell_options(
            summary=resolution.status_message,
            assets=assets,
            selected_structure_id=selected_structure_id,
            enabled=resolution.structure is not None,
        )
        self._kpoint_convergence_view.set_generation_context(
            structure_id=(str(resolution.structure.id) if resolution.structure is not None else None),
            lattice_vectors=(resolution.structure.lattice_vectors if resolution.structure is not None else None),
        )

    def _kpoint_supercell_initial_matrix(self) -> list[list[int]]:
        if self._document is None:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        resolution = self._project_service.resolve_workflow_structure(
            self._document,
            WorkflowStage.CONVERGENCE_KPOINT,
            normalize_state=True,
        )
        if resolution.selected_asset is not None:
            return [list(row) for row in resolution.selected_asset.matrix]
        return [list(row) for row in self._document.project.supercell_matrix]

    def _on_kpoint_simulation_cell_create_requested(self) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return

        host = self._project_service.active_host_structure(self._document)
        if host is None:
            QMessageBox.warning(
                self,
                "No host structure",
                "Load a host structure before creating a saved supercell.",
            )
            return

        dialog = SupercellPreviewDialog(
            structure=host,
            atomic_spins=self._project_service.structure_atomic_spins(host),
            propagation_vector=self._project_service.structure_propagation_vector(host),
            initial_matrix=self._kpoint_supercell_initial_matrix(),
            accept_button_text="Save Supercell",
            show_unit_cell_button=True,
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        if dialog.result_mode() == "unit_cell":
            self._project_service.clear_workflow_supercell_selection(
                self._document,
                WorkflowStage.CONVERGENCE_KPOINT,
            )
            self._refresh_kpoint_convergence_simulation_cell()
            self.statusBar().showMessage("K-point workflow reset to the single unit cell")
            return

        saved = self._project_service.create_workflow_supercell_asset(
            self._document,
            workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
            host_structure_id=host.id,
            matrix=dialog.matrix(),
            reuse_existing=False,
        )
        self._project_service.use_workflow_supercell_asset(
            self._document,
            workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
            asset_structure_id=saved.id,
        )
        self._refresh_kpoint_convergence_simulation_cell()
        self.statusBar().showMessage(f"Saved k-point convergence supercell '{saved.label}'")

    def _on_kpoint_simulation_cell_selection_requested(self, payload: object) -> None:
        if self._document is None:
            return

        structure_id = None
        if payload is not None:
            try:
                structure_id = UUID(str(payload))
            except ValueError:
                structure_id = None

        if structure_id is None:
            self._project_service.clear_workflow_supercell_selection(
                self._document,
                WorkflowStage.CONVERGENCE_KPOINT,
            )
            self._refresh_kpoint_convergence_simulation_cell()
            return

        try:
            selected = self._project_service.use_workflow_supercell_asset(
                self._document,
                workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
                asset_structure_id=structure_id,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Simulation cell unavailable", str(exc))
            self._refresh_kpoint_convergence_simulation_cell()
            return

        self._refresh_kpoint_convergence_simulation_cell()
        self.statusBar().showMessage(f"Selected k-point simulation cell '{selected.label}'")

    def _on_kpoint_simulation_cell_use_unit_cell_requested(self) -> None:
        if self._document is None:
            return
        self._project_service.clear_workflow_supercell_selection(
            self._document,
            WorkflowStage.CONVERGENCE_KPOINT,
        )
        self._refresh_kpoint_convergence_simulation_cell()

    def _on_submit_local_from_kpoint_convergence(
        self,
        grids: list[tuple[int, int, int]],
    ) -> None:
        ordered_grids: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for grid in grids:
            normalized = (int(grid[0]), int(grid[1]), int(grid[2]))
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered_grids.append(normalized)

        states = self._build_convergence_kpoint_states()
        jobs = self._jobs_for_convergence_kpoint_grids(ordered_grids)
        ready_grids = [
            grid
            for grid in ordered_grids
            if (states.get(grid) or SweepPointRunState()).status == "ready"
        ]
        jobs.extend(self._materialize_kpoint_jobs(ready_grids))
        jobs = self._dedupe_jobs(jobs)
        if not jobs:
            QMessageBox.information(
                self,
                "No ready jobs",
                "Only configured grids in the 'ready' state can be submitted.",
            )
            return
        self._submit_jobs_locally(jobs)

    def _on_submit_cluster_from_kpoint_convergence(
        self,
        grids: list[tuple[int, int, int]],
    ) -> None:
        ordered_grids: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for grid in grids:
            normalized = (int(grid[0]), int(grid[1]), int(grid[2]))
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered_grids.append(normalized)

        states = self._build_convergence_kpoint_states()
        jobs = self._jobs_for_convergence_kpoint_grids(ordered_grids)
        ready_grids = [
            grid
            for grid in ordered_grids
            if (states.get(grid) or SweepPointRunState()).status == "ready"
        ]
        jobs.extend(self._materialize_kpoint_jobs(ready_grids))
        jobs = self._dedupe_jobs(jobs)
        if not jobs:
            QMessageBox.information(
                self,
                "No ready jobs",
                "Only configured grids in the 'ready' state can be submitted.",
            )
            return
        self._submit_jobs_on_cluster(jobs)

    def _on_cancel_from_kpoint_convergence(self, grids: list[tuple[int, int, int]]) -> None:
        jobs = self._jobs_for_convergence_kpoint_grids(grids)
        self._cancel_jobs(jobs)

    def _on_inspect_failed_from_kpoint_convergence(self, grids: list[tuple[int, int, int]]) -> None:
        jobs = self._jobs_for_convergence_kpoint_grids(grids)
        self._inspect_failed_jobs(jobs)

    def _on_reset_failed_from_kpoint_convergence(self, grids: list[tuple[int, int, int]]) -> None:
        jobs = self._jobs_for_convergence_kpoint_grids(grids)
        self._reset_failed_jobs(jobs)

    def _on_remove_grids_from_kpoint_convergence(self, grids: list[tuple[int, int, int]]) -> None:
        if self._document is None:
            self._kpoint_convergence_view.remove_grids(grids)
            return

        definition = self._active_kpoint_definition()
        if definition is not None:
            configuration = definition.parameters.get("configuration")
            if isinstance(configuration, dict):
                raw_values = configuration.get("sweep_values", [])
                updated_values: list[object] = []
                target_keys = {self._grid_key(grid) for grid in grids}
                for raw_value in raw_values:
                    if not isinstance(raw_value, list) or len(raw_value) != 3:
                        updated_values.append(raw_value)
                        continue
                    try:
                        key = self._grid_key((int(raw_value[0]), int(raw_value[1]), int(raw_value[2])))
                    except (TypeError, ValueError):
                        updated_values.append(raw_value)
                        continue
                    if key in target_keys:
                        continue
                    updated_values.append(raw_value)
                configuration["sweep_values"] = updated_values

                mapping = definition.parameters.get("job_ids_by_grid", {})
                if isinstance(mapping, dict):
                    for key in target_keys:
                        mapping.pop(key, None)

        self._kpoint_convergence_view.remove_grids(grids)
        self._refresh_kpoint_convergence_rows()

    def _on_examine_kpoint_convergence(self) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return

        host_id = self._document.project.host_structure_id
        host = self._document.structures.get(host_id) if host_id is not None else None
        if host is None:
            QMessageBox.warning(
                self,
                "No host structure",
                "Load a host structure before examining convergence.",
            )
            return

        states = self._build_convergence_kpoint_states()
        counts_map: dict[float, SweepPointRunState] = {}
        labels: dict[float, str] = {}
        for grid, state in states.items():
            count = float(int(grid[0]) * int(grid[1]) * int(grid[2]))
            counts_map[count] = state
            labels[count] = f"Grid: {grid[0]} x {grid[1]} x {grid[2]}"

        persisted_state = self._document.ui_state.get(self._KPOINT_EXAMINER_STATE_KEY, {})
        if not isinstance(persisted_state, dict):
            persisted_state = {}

        dialog = ConvergenceExaminerDialog(
            counts_map,
            num_atoms=len(host.species),
            sweep_kind="kpoint",
            point_labels=labels,
            initial_state=persisted_state,
            parent=self,
        )
        dialog.default_adopted.connect(lambda value, method, et, ft: self._on_kpoint_default_adopted(value, method, et, ft, labels))
        dialog.state_changed.connect(self._on_kpoint_examiner_state_changed)
        dialog.exec()
        self._on_kpoint_examiner_state_changed(dialog.state_snapshot())

    def _on_kpoint_examiner_state_changed(self, state: dict) -> None:
        if self._document is None:
            return

        serialized_state = to_jsonable(state)
        if not isinstance(serialized_state, dict):
            return

        self._document.ui_state[self._KPOINT_EXAMINER_STATE_KEY] = serialized_state
        try:
            self._project_service.save_project(self._document)
        except ValueError:
            self.statusBar().showMessage("Could not persist k-point examiner state to disk")

    def _on_kpoint_default_adopted(
        self,
        value: float,
        method: str,
        energy_threshold: float,
        force_threshold: float,
        labels: Mapping[float, str],
    ) -> None:
        if self._document is None:
            return

        label = labels.get(float(value), "")
        grid = None
        if label.startswith("Grid:"):
            payload = label.split(":", maxsplit=1)[1].strip().replace(" ", "")
            grid = self._parse_grid_key(payload.replace("x", "x"))

        if grid is None:
            QMessageBox.warning(
                self,
                "Adopt failed",
                "Could not resolve the recommended k-point grid from the plotted data.",
            )
            return

        active_definition = self._active_kpoint_definition()
        definition_cell_id = None
        if active_definition is not None and active_definition.input_structure_ids:
            definition_cell_id = str(active_definition.input_structure_ids[0])

        self._project_service.set_recommended_kpoint_grid(
            self._document,
            grid,
            method=method,
            energy_threshold=energy_threshold,
            force_threshold=force_threshold,
            cell_id=(
                definition_cell_id
                if definition_cell_id is not None
                else (str(self._document.project.host_structure_id) if self._document.project.host_structure_id else None)
            ),
        )
        self._kpoint_convergence_view.set_project_defaults(
            kpoint_grid=grid,
            kpoint_method=method,
            energy_threshold=energy_threshold,
            force_threshold=force_threshold,
        )
        self._store_convergence_overview_snapshot(
            "kpoint",
            self._build_kpoint_overview_snapshot(
                grid=grid,
                method=method,
                energy_threshold=float(energy_threshold),
                force_threshold=float(force_threshold),
                simulation_cell_structure_id=definition_cell_id,
                labels=labels,
            ),
        )
        self._refresh_convergence_overview()
        self._persist_document_ui_state(
            failure_message="Could not persist k-point convergence overview snapshot to disk"
        )
        self._populate_project_tree()
        self.statusBar().showMessage(
            f"Project default k-point grid set to {grid[0]}x{grid[1]}x{grid[2]}"
        )

    def _refresh_pristine_relaxation_defaults(self) -> None:
        if self._document is None:
            self._pristine_relaxation_view.set_host_lattice(None)
            self._pristine_relaxation_view.set_project_defaults(
                cutoff_energy=None,
                kpoint_grid=None,
                kpoint_reference_lattice=None,
            )
            self._pristine_relaxation_view.set_supercell_options([])
            return

        host = self._project_service.active_host_structure(self._document)
        self._pristine_relaxation_view.set_host_lattice(
            host.lattice_vectors if host is not None else None
        )
        self._pristine_relaxation_view.set_project_defaults(
            cutoff_energy=self._document.project.recommended_cutoff_energy,
            kpoint_grid=self._document.project.recommended_kpoint_grid,
            kpoint_reference_lattice=self._document.project.recommended_kpoint_lattice,
        )
        self._refresh_pristine_relaxation_supercell_options()

    def _refresh_pristine_relaxation_supercell_options(self) -> None:
        if self._document is None:
            self._pristine_relaxation_view.set_supercell_options([])
            return

        host = self._project_service.active_host_structure(self._document)
        if host is None:
            self._pristine_relaxation_view.set_supercell_options([])
            return

        assets = self._project_service.list_workflow_supercell_assets(
            self._document,
            host_structure_id=host.id,
        )
        self._pristine_relaxation_view.set_supercell_options(
            [
                {
                    "label": asset.label,
                    "matrix": [list(row) for row in asset.matrix],
                }
                for asset in assets
            ]
        )

    def _on_pristine_relaxation_form_state_changed(self, state: dict) -> None:
        if self._document is None:
            return
        self._document.ui_state["pristine_relaxation_form"] = state

    def _serialize_pristine_relaxation_config(self, config: PristineRelaxationConfig) -> dict[str, object]:
        return {
            "name_prefix": config.name_prefix,
            "rows": [
                {
                    "label": row.label,
                    "fix_all_cell": bool(row.fix_all_cell),
                    "supercell_matrix": (
                        [list(line) for line in row.supercell_matrix]
                        if row.supercell_matrix is not None
                        else None
                    ),
                    "parameter_values": dict(row.parameter_values),
                    "added_advanced": list(row.added_advanced),
                }
                for row in config.rows
            ],
        }

    def _deserialize_pristine_relaxation_config(
        self,
        payload: object,
    ) -> PristineRelaxationConfig | None:
        if not isinstance(payload, dict):
            return None

        rows_payload = payload.get("rows", [])
        if not isinstance(rows_payload, list):
            return None

        rows: list[PristineRelaxationRow] = []
        for entry in rows_payload:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label", "")).strip()
            if not label:
                continue

            matrix_payload = entry.get("supercell_matrix")
            matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None = None
            if isinstance(matrix_payload, list) and len(matrix_payload) == 3:
                try:
                    matrix = (
                        (
                            int(matrix_payload[0][0]),
                            int(matrix_payload[0][1]),
                            int(matrix_payload[0][2]),
                        ),
                        (
                            int(matrix_payload[1][0]),
                            int(matrix_payload[1][1]),
                            int(matrix_payload[1][2]),
                        ),
                        (
                            int(matrix_payload[2][0]),
                            int(matrix_payload[2][1]),
                            int(matrix_payload[2][2]),
                        ),
                    )
                except (TypeError, ValueError, IndexError):
                    matrix = None

            raw_params = entry.get("parameter_values", {})
            params = raw_params if isinstance(raw_params, dict) else {}
            restored_params = {
                str(name): self._restore_castep_param_value(str(name), raw_value)
                for name, raw_value in params.items()
            }

            raw_advanced = entry.get("added_advanced", [])
            added_advanced = (
                tuple(str(name) for name in raw_advanced if isinstance(name, str))
                if isinstance(raw_advanced, list)
                else ()
            )

            rows.append(
                PristineRelaxationRow(
                    label=label,
                    fix_all_cell=bool(entry.get("fix_all_cell", False)),
                    supercell_matrix=matrix,
                    parameter_values=restored_params,
                    added_advanced=added_advanced,
                )
            )

        return PristineRelaxationConfig(
            name_prefix=str(payload.get("name_prefix", "pristine_relax")).strip() or "pristine_relax",
            rows=tuple(rows),
        )

    def _pristine_relaxation_definitions(self) -> list:
        if self._document is None:
            return []
        definitions = [
            definition
            for definition in self._document.calculations.values()
            if definition.workflow_stage == WorkflowStage.PRISTINE_RELAXATION
        ]
        definitions.sort(key=lambda definition: definition.created_at, reverse=True)
        return definitions

    def _active_pristine_relaxation_definition(self):
        if self._document is None:
            return None

        raw_definition_id = self._document.ui_state.get(self._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY)
        if isinstance(raw_definition_id, str):
            try:
                definition_id = UUID(raw_definition_id)
            except ValueError:
                definition_id = None
            if definition_id is not None:
                definition = self._document.calculations.get(definition_id)
                if definition is not None and definition.workflow_stage == WorkflowStage.PRISTINE_RELAXATION:
                    return definition

        definitions = self._pristine_relaxation_definitions()
        return definitions[0] if definitions else None

    def _definition_for_pristine_relaxation_label(self, label: str):
        definition = self._active_pristine_relaxation_definition()
        if definition is None:
            return None, None

        config = self._deserialize_pristine_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return None, None
        labels = {row.label for row in config.rows}
        if label in labels:
            return definition, config
        return None, None

    def _job_for_pristine_relaxation_label(self, definition, label: str) -> Job | None:
        if self._document is None:
            return None

        mapping = definition.parameters.get("job_ids_by_label", {})
        if isinstance(mapping, dict):
            raw_job_id = mapping.get(label)
            if isinstance(raw_job_id, str):
                try:
                    job_id = UUID(raw_job_id)
                except ValueError:
                    job_id = None
                if job_id is not None:
                    job = self._document.jobs.get(job_id)
                    if job is not None:
                        return job
        return None

    def _build_pristine_relaxation_states(self) -> dict[str, RelaxationRunState]:
        states: dict[str, RelaxationRunState] = {}
        if self._document is None:
            return states

        definition = self._active_pristine_relaxation_definition()
        if definition is None:
            return states
        config = self._deserialize_pristine_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return states

        for row in config.rows:
            job = self._job_for_pristine_relaxation_label(definition, row.label)
            if job is None:
                states[row.label] = RelaxationRunState(status="ready")
                continue

            status_value = "ready" if job.status == JobStatus.DEFINED else job.status.value
            run = self._run_for_job(job.id)
            if status_value == JobStatus.COMPLETED.value and run is not None and run.convergence_achieved is False:
                status_value = "not converged"

            lattice_a = None
            lattice_b = None
            lattice_c = None
            if run is not None and run.result_id is not None:
                result = self._document.results.get(run.result_id)
                if result is not None:
                    lattice_a = result.data.get("lattice_a") if isinstance(result.data, dict) else None
                    lattice_b = result.data.get("lattice_b") if isinstance(result.data, dict) else None
                    lattice_c = result.data.get("lattice_c") if isinstance(result.data, dict) else None

            states[row.label] = RelaxationRunState(
                status=status_value,
                total_energy=self._total_energy_for_job(job.id),
                max_force=self._max_force_for_job(job.id),
                lattice_a=float(lattice_a) if isinstance(lattice_a, (int, float)) else None,
                lattice_b=float(lattice_b) if isinstance(lattice_b, (int, float)) else None,
                lattice_c=float(lattice_c) if isinstance(lattice_c, (int, float)) else None,
                job_id=job.id,
                job_label=job.label,
                geom_converged=(run.convergence_achieved if run is not None else None),
                error_message=job.error_message,
            )

        return states

    def _refresh_pristine_relaxation_rows(self) -> None:
        self._pristine_relaxation_view.set_runtime_states(self._build_pristine_relaxation_states())

    def _on_pristine_relaxation_supercell_edit_requested(self, payload: object) -> None:
        if self._document is None:
            return

        try:
            row_id = int(payload)
        except (TypeError, ValueError):
            return

        host = self._project_service.active_host_structure(self._document)
        if host is None:
            QMessageBox.warning(
                self,
                "No host structure",
                "Load a host structure before configuring pristine-relaxation supercells.",
            )
            return

        existing = self._pristine_relaxation_view.row_supercell_matrix(row_id)
        initial_matrix = (
            [list(line) for line in existing]
            if existing is not None
            else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

        dialog = SupercellPreviewDialog(
            structure=host,
            atomic_spins=self._project_service.structure_atomic_spins(host),
            propagation_vector=self._project_service.structure_propagation_vector(host),
            initial_matrix=initial_matrix,
            accept_button_text="Use Supercell",
            show_unit_cell_button=True,
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        if dialog.result_mode() == "unit_cell":
            self._pristine_relaxation_view.set_row_supercell(row_id, None)
            return

        matrix_values = [
            [int(row[0]), int(row[1]), int(row[2])]
            for row in dialog.matrix()
        ]
        saved = self._project_service.create_workflow_supercell_asset(
            self._document,
            workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
            host_structure_id=host.id,
            matrix=matrix_values,
            reuse_existing=True,
        )
        matrix = (
            (matrix_values[0][0], matrix_values[0][1], matrix_values[0][2]),
            (matrix_values[1][0], matrix_values[1][1], matrix_values[1][2]),
            (matrix_values[2][0], matrix_values[2][1], matrix_values[2][2]),
        )

        self._refresh_pristine_relaxation_supercell_options()
        self._pristine_relaxation_view.set_row_supercell(row_id, matrix)
        self.statusBar().showMessage(f"Saved pristine-relaxation simulation cell '{saved.label}'")

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
            expand_pristine_relaxation_jobs(
                host_structure=host,
                config=config,
                project_cutoff_energy=self._document.project.recommended_cutoff_energy,
                project_kpoint_grid=self._document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=self._document.project.recommended_kpoint_lattice,
            )
        except (ValueError, StructureBridgeError) as exc:
            QMessageBox.warning(self, "Pristine relaxation setup invalid", str(exc))
            return

        output_root = self._document.project_root / "files" / "pristine_relaxation"
        self._project_service.register_pristine_relaxation_definition(
            self._document,
            input_structure_id=host.id,
            parameters=self._serialize_pristine_relaxation_config(config),
            output_directory=output_root,
        )
        latest_definition = max(
            (
                definition
                for definition in self._document.calculations.values()
                if definition.workflow_stage == WorkflowStage.PRISTINE_RELAXATION
            ),
            key=lambda definition: definition.created_at,
        )
        self._document.ui_state[self._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(latest_definition.id)

        self._refresh_pristine_relaxation_rows()
        self.statusBar().showMessage(
            f"Configured {len(config.rows)} pristine-relaxation jobs; input files will be generated on submission"
        )

    def _completed_pristine_relaxation_jobs(self) -> list[CompletedPristineJob]:
        if self._document is None:
            return []

        definition = self._active_pristine_relaxation_definition()
        if definition is None:
            return []
        config = self._deserialize_pristine_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return []

        parameter_values_by_label = {
            row.label: dict(row.parameter_values)
            for row in config.rows
        }
        supercell_label_by_row: dict[str, str] = {}
        for row in config.rows:
            matrix = row.supercell_matrix
            if matrix is None:
                supercell_label_by_row[row.label] = "Unit cell"
            else:
                supercell_label_by_row[row.label] = "[" + "; ".join(
                    ",".join(str(int(value)) for value in vector) for vector in matrix
                ) + "]"

        completed: list[CompletedPristineJob] = []
        for row_label, state in sorted(self._build_pristine_relaxation_states().items()):
            if state.status != JobStatus.COMPLETED.value:
                continue
            if state.job_id is None:
                continue

            run = self._run_for_job(state.job_id)
            if run is None or run.result_id is None:
                continue
            result = self._document.results.get(run.result_id)
            if result is None or not result.derived_structure_ids:
                continue

            structure_id = result.derived_structure_ids[0]
            completed.append(
                CompletedPristineJob(
                    job_id=state.job_id,
                    job_label=row_label,
                    structure_id=structure_id,
                    supercell_label=supercell_label_by_row.get(row_label, "Unit cell"),
                    status=state.status,
                    total_energy=state.total_energy,
                    lattice_a=state.lattice_a,
                    lattice_b=state.lattice_b,
                    lattice_c=state.lattice_c,
                    parameter_values=parameter_values_by_label.get(row_label, {}),
                )
            )

        return completed

    @staticmethod
    def _format_parameter_value(value: object) -> str:
        if isinstance(value, ValueWithUnit):
            return f"{value.value} {value.unit}".strip()
        if isinstance(value, BlockSection):
            return f"%BLOCK {value.name} ({len(value.rows)} row(s))"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(entry) for entry in value)
        if isinstance(value, dict):
            parts = [f"{key}={entry}" for key, entry in value.items()]
            return ", ".join(parts)
        if value is None:
            return ""
        return str(value)

    def _build_parameter_difference_rows(
        self,
        selected_jobs: list[CompletedPristineJob],
    ) -> list[dict[str, object]]:
        if len(selected_jobs) < 2:
            return []

        names = sorted(
            {
                name
                for job in selected_jobs
                for name in job.parameter_values
            }
        )
        rows: list[dict[str, object]] = []
        for name in names:
            displays = [
                self._format_parameter_value(job.parameter_values.get(name))
                for job in selected_jobs
            ]
            if len(set(displays)) <= 1:
                continue
            rows.append(
                {
                    "name": name,
                    "values": displays,
                }
            )
        return rows

    def _serialize_comparison_result(
        self,
        result,
        *,
        parameter_differences: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "metric_definitions": [
                {
                    "key": definition.key,
                    "label": definition.label,
                    "unit": definition.unit,
                    "group": definition.group,
                    "meaning": definition.meaning,
                    "calculation": definition.calculation,
                    "interpretation": definition.interpretation,
                    "caveats": definition.caveats,
                    "physics": definition.physics,
                    "method": definition.method,
                    "equation": definition.equation,
                    "symbols": [
                        {"symbol": symbol, "meaning": meaning}
                        for symbol, meaning in definition.symbols
                    ],
                }
                for definition in result.definitions
            ],
            "experimental_values": {
                key: {
                    "value": value.value,
                    "display": value.display,
                    "warning": value.warning,
                }
                for key, value in result.experimental_values.items()
            },
            "dft_columns": [
                {
                    "label": column.label,
                    "structure_id": str(column.structure_id),
                    "warnings": list(column.warnings),
                    "supercell_matrix": [
                        [int(entry) for entry in row]
                        for row in column.normalization.supercell_matrix
                    ],
                    "values": {
                        key: {
                            "value": value.value,
                            "display": value.display,
                            "warning": value.warning,
                        }
                        for key, value in column.values.items()
                    },
                }
                for column in result.dft_columns
            ],
            "parameter_differences": parameter_differences,
            "warnings": list(result.warnings),
        }

    def _on_generate_comparison_table_requested(self) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return

        host = self._project_service.active_host_structure(self._document)
        if host is None:
            QMessageBox.warning(
                self,
                "No host structure",
                "Load a host structure before generating a comparison table.",
            )
            return

        completed_jobs = self._completed_pristine_relaxation_jobs()
        if not completed_jobs:
            QMessageBox.information(
                self,
                "No completed pristine relaxations",
                "Run and complete one or more pristine-relaxation jobs first.",
            )
            return

        default_label = f"Comparison {len(self._document.comparison_tables) + 1}"
        dialog = ComparisonJobDialog(jobs=completed_jobs, default_label=default_label, parent=self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        selected_job_ids = set(dialog.selected_job_ids())
        selected_jobs = [job for job in completed_jobs if job.job_id in selected_job_ids]
        if not selected_jobs:
            QMessageBox.warning(self, "No jobs selected", "Select at least one completed job.")
            return

        dft_inputs: list[DftComparisonInput] = []
        for selected_job in selected_jobs:
            structure = self._document.structures.get(selected_job.structure_id)
            if structure is None:
                continue
            dft_inputs.append(
                DftComparisonInput(
                    label=f"DFT ({selected_job.job_label})",
                    structure=structure,
                    structure_id=structure.id,
                )
            )

        if not dft_inputs:
            QMessageBox.warning(
                self,
                "Comparison unavailable",
                "The selected jobs did not provide relaxed structures for comparison.",
            )
            return

        try:
            comparison_result = compare_structures(host, dft_inputs)
        except ValueError as exc:
            QMessageBox.warning(self, "Comparison failed", str(exc))
            return

        parameter_differences = self._build_parameter_difference_rows(selected_jobs)

        table = ComparisonTable(
            label=dialog.table_label(),
            experimental_structure_id=host.id,
            dft_structure_ids=[item.structure_id for item in dft_inputs],
            dft_labels=[item.label for item in dft_inputs],
            result_data=self._serialize_comparison_result(
                comparison_result,
                parameter_differences=parameter_differences,
            ),
        )
        self._project_service.add_comparison_table(self._document, table)

        self._sync_comparison_table_views()
        self._populate_project_tree()
        self._set_active_route(self._comparison_route(table.id))
        self.statusBar().showMessage(f"Generated comparison table '{table.label}'")

    def _jobs_for_pristine_relaxation_labels(self, labels: list[str]) -> list[Job]:
        if self._document is None:
            return []
        definition = self._active_pristine_relaxation_definition()
        if definition is None:
            return []

        selected: list[Job] = []
        seen: set[UUID] = set()
        for label in labels:
            job = self._job_for_pristine_relaxation_label(definition, label)
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    def _materialize_pristine_relaxation_jobs(self, labels: list[str]) -> list[Job]:
        if self._document is None:
            return []

        definition = self._active_pristine_relaxation_definition()
        if definition is None:
            return []

        config = self._deserialize_pristine_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return []
        if not definition.input_structure_ids:
            return []

        host = self._document.structures.get(definition.input_structure_ids[0])
        output_root = self._definition_output_directory(definition)
        if host is None or output_root is None:
            return []

        try:
            expansion = expand_pristine_relaxation_jobs(
                host_structure=host,
                config=config,
                project_cutoff_energy=self._document.project.recommended_cutoff_energy,
                project_kpoint_grid=self._document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=self._document.project.recommended_kpoint_lattice,
            )
        except (ValueError, StructureBridgeError):
            return []

        mapping = definition.parameters.setdefault("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            mapping = {}
            definition.parameters["job_ids_by_label"] = mapping

        spec_by_label = {
            str(spec.provenance.get("label", "")): spec
            for spec in expansion.jobs
        }

        created: list[Job] = []
        for label in labels:
            existing_job = self._job_for_pristine_relaxation_label(definition, label)
            if existing_job is not None:
                created.append(existing_job)
                continue

            spec = spec_by_label.get(label)
            if spec is None:
                continue

            write_job_directory(spec, output_root)
            job = self._project_service.add_job(
                self._document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type="castep_pristine_relaxation",
            )
            mapping[label] = str(job.id)
            created.append(job)

        return self._dedupe_jobs(created)

    def _on_submit_local_from_pristine_relaxation(self, labels: list[str]) -> None:
        states = self._build_pristine_relaxation_states()
        ready_labels = [
            label
            for label in labels
            if (states.get(label) or RelaxationRunState()).status in {"ready", "defined", "not generated"}
        ]
        jobs = self._jobs_for_pristine_relaxation_labels(labels)
        jobs.extend(self._materialize_pristine_relaxation_jobs(ready_labels))
        jobs = self._dedupe_jobs(jobs)
        if not jobs:
            QMessageBox.information(
                self,
                "No ready jobs",
                "Only configured pristine-relaxation rows in the 'ready' state can be submitted.",
            )
            return
        self._submit_jobs_locally(jobs)

    def _on_submit_cluster_from_pristine_relaxation(self, labels: list[str]) -> None:
        states = self._build_pristine_relaxation_states()
        ready_labels = [
            label
            for label in labels
            if (states.get(label) or RelaxationRunState()).status in {"ready", "defined", "not generated"}
        ]
        jobs = self._jobs_for_pristine_relaxation_labels(labels)
        jobs.extend(self._materialize_pristine_relaxation_jobs(ready_labels))
        jobs = self._dedupe_jobs(jobs)
        if not jobs:
            QMessageBox.information(
                self,
                "No ready jobs",
                "Only configured pristine-relaxation rows in the 'ready' state can be submitted.",
            )
            return
        self._submit_jobs_on_cluster(jobs)

    @staticmethod
    def _serialize_convergence_config(config: ConvergenceConfig) -> dict:
        return {
            "name_prefix": config.name_prefix,
            "sweep_parameter": config.sweep_parameter,
            "sweep_values": [
                list(value) if isinstance(value, tuple) else value for value in config.sweep_values
            ],
            "cutoff_energies": list(config.cutoff_energies),
            "shared_params": config.shared_params,
            "overrides": [
                {
                    "cutoffs": [
                        list(value) if isinstance(value, tuple) else value for value in sorted(group)
                    ],
                    "params": payload,
                }
                for group, payload in config.overrides.items()
            ],
        }

    def _refresh_muon_generation_table(self) -> None:
        if self._document is None:
            self._muon_generation_view.set_sites([])
            return

        candidates = [site for site in self._document.sites.values() if site.status == "candidate"]
        candidates.sort(key=lambda item: item.label)
        rows = [
            {
                "label": site.label,
                "method": site.generation_method or "",
                "status": site.status,
                "fractional": site.fractional_position,
            }
            for site in candidates
        ]
        self._muon_generation_view.set_sites(rows)

    def _refresh_host_structure_view(self) -> None:
        if self._document is None:
            self._host_structure_view.set_structure(None)
            self._host_structure_view.set_project_supercell_matrix(None)
            return

        host_id = self._document.project.host_structure_id
        host: Structure | None = self._document.structures.get(host_id) if host_id else None
        self._host_structure_view.set_project_supercell_matrix(self._document.project.supercell_matrix)
        self._host_structure_view.set_structure(host)

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
            propagation = [float(raw_propagation[0]), float(raw_propagation[1]), float(raw_propagation[2])]
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

        raw_matrix = payload.get("supercell_matrix")
        if isinstance(raw_matrix, list) and len(raw_matrix) == 3 and all(
            isinstance(row, list) and len(row) == 3 for row in raw_matrix
        ):
            self._document.project.supercell_matrix = [
                [int(row[0]), int(row[1]), int(row[2])]
                for row in raw_matrix
            ]

    def _refresh_convergence_cutoff_rows(self) -> None:
        if self._document is None:
            self._convergence_view.set_cutoff_runtime_states({})
            return
        self._convergence_view.set_cutoff_runtime_states(
            self._build_convergence_cutoff_states()
        )

    def _convergence_form_snapshot(self) -> tuple[str, tuple[float, ...]]:
        if self._document is None:
            return "cutoff_convergence", ()

        form_sources: list[dict] = []
        widget_state = self._convergence_view.collect_state()
        if isinstance(widget_state, dict):
            form_sources.append(widget_state)

        persisted = self._document.ui_state.get("convergence_form", {})
        if isinstance(persisted, dict):
            form_sources.append(persisted)

        for form_state in form_sources:
            name_prefix = (
                str(form_state.get("name_prefix", "cutoff_convergence")).strip()
                or "cutoff_convergence"
            )

            cutoffs: list[float] = []
            raw_cutoffs = form_state.get("cutoff_energies", [])
            if isinstance(raw_cutoffs, list):
                for value in raw_cutoffs:
                    try:
                        cutoffs.append(float(value))
                    except (TypeError, ValueError):
                        continue

            unique = tuple(dict.fromkeys(cutoffs))
            if unique:
                return name_prefix, unique

        return "cutoff_convergence", ()

    def _detach_convergence_job(self, job_id: UUID) -> None:
        if self._document is None:
            return
        for definition in self._convergence_definitions():
            mapping = definition.parameters.get("job_ids_by_cutoff", {})
            if not isinstance(mapping, dict):
                continue
            stale_keys = [key for key, value in mapping.items() if str(value) == str(job_id)]
            for key in stale_keys:
                mapping.pop(key, None)

    def _build_convergence_cutoff_states(self) -> dict[float, SweepPointRunState]:
        if self._document is None:
            return {}

        states: dict[float, SweepPointRunState] = {}
        form_prefix, form_cutoffs = self._convergence_form_snapshot()
        for cutoff in form_cutoffs:
            states[cutoff] = SweepPointRunState(
                status="ready",
                job_label=convergence_job_name(
                    form_prefix,
                    cutoff,
                    sweep_parameter="cut_off_energy",
                ),
            )

        definition = self._active_convergence_definition()
        if definition is None:
            return states

        config = self._deserialize_convergence_config(definition.parameters.get("configuration", {}))
        if config is None:
            return states

        config_cutoffs = tuple(dict.fromkeys(float(value) for value in config.cutoff_energies))
        # If the editable convergence form has diverged from the active definition,
        # do not project old runtime results into the regenerated cutoff table.
        if form_cutoffs and (form_prefix != config.name_prefix or form_cutoffs != config_cutoffs):
            return states

        for cutoff in config.cutoff_energies:
            job = self._job_for_definition_cutoff(definition, cutoff)
            if job is None:
                states[cutoff] = SweepPointRunState(
                    status="ready",
                    job_label=convergence_job_name(
                        config.name_prefix,
                        cutoff,
                        sweep_parameter="cut_off_energy",
                    ),
                )
                continue

            status_value = "ready" if job.status == JobStatus.DEFINED else job.status.value

            states[cutoff] = SweepPointRunState(
                status=status_value,
                total_energy=self._total_energy_for_job(job.id),
                max_force=self._max_force_for_job(job.id),
                job_id=job.id,
                job_label=job.label,
            )

        return states

    def _total_energy_for_job(self, job_id: UUID) -> float | None:
        if self._document is None:
            return None

        runs = [run for run in self._document.runs.values() if run.job_id == job_id]
        if not runs:
            return None

        latest_run = max(runs, key=lambda run: run.run_index)
        if latest_run.result_id is None:
            return None
        result = self._document.results.get(latest_run.result_id)
        if result is None:
            return None
        return self._extract_total_energy(result.data)

    def _max_force_for_job(self, job_id: UUID) -> float | None:
        if self._document is None:
            return None

        runs = [run for run in self._document.runs.values() if run.job_id == job_id]
        if not runs:
            return None

        latest_run = max(runs, key=lambda run: run.run_index)
        if latest_run.result_id is None:
            return None
        result = self._document.results.get(latest_run.result_id)
        if result is None:
            return None
        return self._extract_max_force(result.data)

    @staticmethod
    def _extract_total_energy(payload: object) -> float | None:
        def parse_numeric(candidate: object) -> float | None:
            if isinstance(candidate, bool):
                return None
            if isinstance(candidate, (int, float)):
                numeric = float(candidate)
                if isnan(numeric):
                    return None
                return numeric
            if isinstance(candidate, dict):
                return parse_numeric(candidate.get("value"))
            return None

        if not isinstance(payload, dict):
            return None

        for key in ("total_energy", "total_energy_ev", "final_energy", "final_energy_ev", "energy", "energy_ev"):
            value = parse_numeric(payload.get(key))
            if value is not None:
                return value

        energies = payload.get("energies")
        if isinstance(energies, dict):
            for key in ("total", "total_energy", "final", "final_energy", "energy"):
                value = parse_numeric(energies.get(key))
                if value is not None:
                    return value

        return None

    @staticmethod
    def _extract_max_force(payload: object) -> float | None:
        def parse_numeric(candidate: object) -> float | None:
            if isinstance(candidate, bool):
                return None
            if isinstance(candidate, (int, float)):
                numeric = float(candidate)
                if isnan(numeric):
                    return None
                return numeric
            if isinstance(candidate, dict):
                return parse_numeric(candidate.get("value"))
            return None

        if not isinstance(payload, dict):
            return None

        for key in ("max_force", "maximum_force", "max_force_eva"):
            value = parse_numeric(payload.get(key))
            if value is not None:
                return value

        forces = payload.get("forces")
        if isinstance(forces, dict):
            for key in ("max", "max_force", "maximum"):
                value = parse_numeric(forces.get(key))
                if value is not None:
                    return value

        return None
