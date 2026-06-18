"""Primary application window for MuFinder."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from math import isnan
import os
from pathlib import Path
import posixpath
import re
import shlex
import shutil
import stat
import time
from typing import TypeVar
from uuid import UUID, uuid4

import numpy as np
from PySide6.QtCore import QEventLoop, QPointF, QRectF, QSize, QSettings, QThread, QTimer, Qt, Signal
from PySide6.QtGui import (
    QActionGroup,
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
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
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.castep_setup import load_castep_setup
from mufinder.branding import (
    app_icon,
    logo_path,
    project_home_color_map_path,
    project_home_map_path,
)
from mufinder.cluster_setup import ClusterConfig, load_cluster_config, load_cluster_configs
from mufinder.castep_io.gui_metadata import make_block_section, make_keyword_assignment, normalize_gui_value
from mufinder.castep_io.schema import KeywordSpec, canonical_keyword_name, get_keyword_spec
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, ComparisonTable, Job, MuonSite, Result, Structure, utc_now
from mufinder.app.cluster_setup_dialog import ClusterSetupDialog
from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.cluster_submission_dialog import ClusterSubmissionConfig, ClusterSubmissionDialog
from mufinder.app.comparison_job_dialog import ComparisonJobDialog, CompletedPristineJob
from mufinder.app.comparison_table_widget import ComparisonTableWidget
from mufinder.app.cluster_ui_utils import (
    format_cluster_connection_error,
    make_message_box_resizable,
    prompt_for_cluster_credentials,
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
from mufinder.app.energy_barriers_widget import (
    EnergyBarriersWidget,
    EnergyBarrierEndpointRow,
    EnergyBarrierNebResultRow,
    EnergyBarrierPreviewImageRow,
    EnergyBarrierResultRow,
    EnergyBarrierSourceRow,
)
from mufinder.app.kpoint_convergence_widget import KPointConvergenceWidget
from mufinder.app.relaxation_adapter import (
    MUON_RELAXATION_CALC_TYPES,
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    expand_muon_relaxation_jobs,
    expand_pristine_relaxation_jobs,
    is_muon_relaxation_calc_type,
    muon_charge_state_label,
    muon_charge_state_prefix,
    muon_relaxation_calc_type,
)
from mufinder.app.relaxation_types import (
    MuonRelaxationConfig,
    MuonRelaxationRow,
    PristineRelaxationConfig,
    PristineRelaxationRow,
    RelaxationRunState,
    SupercellMatrix,
)
from mufinder.app.relaxation_analysis_widget import (
    CompletedRelaxationAnalysisSource,
    RelaxationAnalysisJobDialog,
    RelaxationAnalysisWidget,
    default_workspace_state,
)
from mufinder.app.relaxation_widget import PristineRelaxationWidget
from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget
from mufinder.app.uep_adapter import expand_uep_jobs
from mufinder.app.uep_types import UepConfig, UepRunState, UepSourceRow
from mufinder.app.uep_wizard import UepAnalysisDialog
from mufinder.app.uep_widget import UEPWidget
from mufinder.app.active_host_badge import ActiveHostBadge
from mufinder.app.host_structure_widget import HostStructureWidget
from mufinder.app.icons import icon, workflow_stage_icon
from mufinder.app.journal_widget import JournalWidget
from mufinder.app.magnetic_field_widget import MagneticFieldWidget
from mufinder.app.muon_generation_widget import MuonGenerationWidget
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.quantum_analysis_dialog import QuantumPhononAnalysisDialog
from mufinder.app.quantum_corrections_widget import (
    QuantumCorrectionsWidget,
    QuantumResultRow,
    QuantumSourceRow,
)
from mufinder.app.supercell_dialog import SupercellEditorDialog
from mufinder.app.submission_dialog import LocalSubmissionDialog
from mufinder.app.theme import (
    THEME_DARK,
    THEME_LIGHT,
    THEME_SYSTEM,
    apply_theme,
    body_font_family,
    current_palette,
    heading_font_family,
    light_palette,
    load_theme_mode,
    save_theme_mode,
    style_section_subtitle,
    style_section_title,
    status_brush,
    use_square_top_table_corners,
)
from mufinder.app.workflow_dialogs import EnergyBarrierPathViewerDialog, QuantumCorrectionsConfigDialog
from mufinder.app.workflow_parameters import build_workflow_parameter_surface, kpoint_exclusive_hint, workflow_basic_parameter_names
from mufinder.app.workflow_submission import (
    WorkflowSubmissionCoordinator,
    sweep_point_state_is_ready,
    workflow_row_state_is_ready,
)
from mufinder.castep_io.cell_parser import CellParseError, parse_cell_file
from mufinder.castep_io.output_parser import CastepResult
from mufinder.castep_io.ts_parser import (
    TransitionStateImage,
    TransitionStateParseError,
    parse_transition_state_file,
)
from mufinder.castep_io.types import AtomicPosition, BlockSection, PositionBlock, ValueWithUnit
from mufinder.persistence.codec import to_jsonable
from mufinder.app.cluster_controller import ClusterController
from mufinder.services.job_execution_controller import (
    ClusterJobSubmission,
    JobExecutionController,
    PreparedClusterSubmission,
)
from mufinder.services.local_runner import LocalJobRunner
from mufinder.app.selection_snapshot import SelectionSnapshotPushButton, resolve_target_keys
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.ssh_client import (
    SSHAuthenticationError,
    SSHCredentials,
    SSHUnknownHostKeyError,
    save_trusted_host_key,
)
from mufinder.science.energy_barriers import (
    EnergyBarrierError,
    MuonPath,
    enumerate_distinct_muon_paths,
    prepare_transition_state_structures,
)
from mufinder.science.kpoint_grids import equivalent_grid_for_supercell, lattice_parameters
from mufinder.science.quantum_phonons import PhononParseError, muon_zpe, parse_phonon_path, w1_entanglement_witness
from mufinder.science.relaxation_analysis import folded_muon_position_from_structure
from mufinder.science.site_generation import SiteGenerationError, lattice_cart_to_frac, lattice_frac_to_cart
from mufinder.science.supercell import build_supercell_structure
from mufinder.science.uep import UepError, UepPotentialGrid, deserialize_potential_grid, serialize_potential_grid
from mufinder.science.structure_comparison import DftComparisonInput, compare_structures
from mufinder.science.structure_io import StructureLoadError
from mufinder.workflow import StructureBridgeError, structure_to_spec, write_job_directory
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    NUDGED_ELASTIC_BAND_SET_CALC_TYPE,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
    TRANSITION_STATE_SEARCH_SET_CALC_TYPE,
    NudgedElasticBandJobConfig,
    NudgedElasticBandSource,
    TransitionStateSearchJobConfig,
    TransitionStateSearchSource,
    expand_nudged_elastic_band_jobs,
    expand_transition_state_search_jobs,
    position_block_from_structure,
)
from mufinder.workflow.naming import job_name
from mufinder.workflow.quantum import (
    QUANTUM_SINGLEPOINT_CALC_TYPE,
    QUANTUM_PHONON_CALC_TYPE,
    QUANTUM_PHONON_SET_CALC_TYPE,
    QuantumPhononJobConfig,
    QuantumPhononSource,
    expand_quantum_singlepoint_jobs,
    expand_quantum_phonon_jobs,
    quantum_source_eligibility,
)
from mufinder.workflow.renderer import render_job
from mufinder.workflow.types import CellTemplate, InputTemplate, JobSpec, ParamTemplate


_ACTIVE_LOCAL_JOB_STATES = {
    JobStatus.SUBMITTING,
    JobStatus.QUEUED,
    JobStatus.RUNNING,
    JobStatus.RETRIEVING,
}
_RESUBMITTABLE_JOB_STATES = {JobStatus.FAILED, JobStatus.CANCELLED}
_VISIBLE_DASHBOARD_STATES = _ACTIVE_LOCAL_JOB_STATES | {JobStatus.STOPPED} | _RESUBMITTABLE_JOB_STATES
_RELAXATION_ANALYSIS_STATE_KEY = "relaxation_analysis_workspaces"
_WorkflowKey = TypeVar("_WorkflowKey")
_STRUCTURE_CONTROLLED_CELL_KEYWORDS = {"lattice_cart", "lattice_abc", "positions_frac", "positions_abs"}
_QUANTUM_CONFIG_KEYWORD_NAMES = workflow_basic_parameter_names()
_ENERGY_BARRIER_TSS_KEYWORD_NAMES = workflow_basic_parameter_names(
    "tssearch_lstqst_protocol",
    "tssearch_qst_max_iter",
    "tssearch_cg_max_iter",
    "tssearch_max_path_points",
)
_ENERGY_BARRIER_NEB_KEYWORD_NAMES = workflow_basic_parameter_names(
    "tssearch_max_path_points",
    "tssearch_neb_spring_constant",
    "tssearch_neb_tangent_mode",
    "tssearch_neb_method",
    "tssearch_neb_max_iter",
    "tssearch_neb_climbing",
    "tssearch_neb_normed",
)
_QUANTUM_GUI_TO_METHOD = {
    "FINITEDISPLACEMENT": "finite_displacement",
    "DFPT": "dfpt",
    "LINEARRESPONSE": "linear_response",
}
_QUANTUM_DEFAULT_ELEC_ENERGY_TOL = ValueWithUnit(1e-9, "eV")
_QUANTUM_METHOD_TO_GUI = {value: key for key, value in _QUANTUM_GUI_TO_METHOD.items()}
_QUANTUM_STAGE_SINGLEPOINT = "singlepoint"
_QUANTUM_STAGE_PHONON = "phonon"


def _structure_species_options(structure: Structure | None) -> tuple[str, ...]:
    if structure is None:
        return ()
    return tuple(dict.fromkeys(species for item in structure.species if (species := str(item).strip())))


def _merge_species_options(structures: list[Structure | None]) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for structure in structures:
        for species in _structure_species_options(structure):
            if species in seen:
                continue
            seen.add(species)
            values.append(species)
    return tuple(values)


def _format_energy(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g} eV"
    except (TypeError, ValueError):
        return ""


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


def _format_phonon_eigenvalues(value: object) -> str:
    if not isinstance(value, list):
        return ""
    entries: list[str] = []
    for mode_index, eigenvalue in enumerate(value, start=1):
        try:
            entries.append(f"{mode_index}:{float(eigenvalue):.1f}")
        except (TypeError, ValueError):
            continue
    return ", ".join(entries)


def _merge_serialized_sources(
    existing_sources: object,
    new_sources: list[dict[str, object]],
    *,
    key_field: str,
) -> list[dict[str, object]]:
    merged_by_key: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for source in tuple(existing_sources) if isinstance(existing_sources, list) else ():
        if not isinstance(source, dict):
            continue
        key = str(source.get(key_field, "")).strip()
        if not key:
            continue
        if key not in merged_by_key:
            order.append(key)
        merged_by_key[key] = dict(source)
    for source in new_sources:
        key = str(source.get(key_field, "")).strip()
        if not key:
            continue
        if key not in merged_by_key:
            order.append(key)
        merged_by_key[key] = dict(source)
    return [merged_by_key[key] for key in order]


def _replace_serialized_sources_for_pair(
    existing_sources: object,
    *,
    reactant_site_id: str,
    product_site_id: str,
    new_sources: list[dict[str, object]],
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for source in tuple(existing_sources) if isinstance(existing_sources, list) else ():
        if not isinstance(source, dict):
            continue
        if (
            str(source.get("reactant_site_id", "")).strip() == reactant_site_id
            and str(source.get("product_site_id", "")).strip() == product_site_id
        ):
            continue
        filtered.append(dict(source))
    return _merge_serialized_sources(filtered, new_sources, key_field="key")


_LEGACY_ENERGY_BARRIER_OBSOLETE_PARAMS: frozenset[str] = frozenset({"tssearch_lst_max_iter"})
_LEGACY_ENERGY_BARRIER_ZERO_DEFAULT_PARAMS: frozenset[str] = frozenset({
    "fine_cut_off_energy",
    "tssearch_force_tol",
    "tssearch_disp_tol",
    "tssearch_energy_tol",
})


def _strip_legacy_energy_barrier_params(params: dict[str, object]) -> dict[str, object]:
    """Drop params that were stored by the old TSS dialog but are now invalid.

    tssearch_lst_max_iter is obsolete.  The three tolerance keywords default to
    their GEOM_* counterparts in CASTEP; the old dialog exposed them with a
    cross-reference default that rendered as 0, so any stored value ≤ 0 is a
    legacy placeholder that should be omitted so CASTEP uses its own defaults.
    fine_cut_off_energy was also persisted as a zero-value placeholder when the
    Energy Barriers surface briefly exposed it as a regular physical editor row.
    """
    result: dict[str, object] = {}
    for name, value in params.items():
        if name in _LEGACY_ENERGY_BARRIER_OBSOLETE_PARAMS:
            continue
        if name in _LEGACY_ENERGY_BARRIER_ZERO_DEFAULT_PARAMS:
            numeric = value.value if isinstance(value, ValueWithUnit) else value
            try:
                if float(numeric) <= 0:
                    continue
            except (TypeError, ValueError):
                pass
        result[name] = value
    return result


def _input_template_from_keyword_values(values: Mapping[str, object]) -> InputTemplate:
    lattice_cart = None
    lattice_abc = None
    positions_frac = None
    positions_abs = None
    cell_scalars = []
    cell_blocks = []
    param_keywords = []
    param_blocks = []

    species_pot = None
    kpoint_mp_grid = None
    kpoint_mp_spacing = None
    kpoints_list = None
    symmetry_generate = None

    for raw_name, raw_value in values.items():
        spec = get_keyword_spec("param", raw_name) or get_keyword_spec("cell", raw_name)
        if spec is None:
            raise ValueError(f"Unsupported CASTEP keyword '{raw_name}' in workflow settings.")

        canonical_name = canonical_keyword_name(spec.file, spec.name)
        if spec.file == "cell" and canonical_name in _STRUCTURE_CONTROLLED_CELL_KEYWORDS:
            raise ValueError(f"{canonical_name} is controlled by the workflow structure and cannot be overridden.")

        if spec.kind == "block":
            block = make_block_section(spec, raw_value)
            if block is None:
                continue
            if spec.file == "cell":
                if canonical_name == "lattice_cart":
                    lattice_cart = block
                    lattice_abc = None
                    continue
                if canonical_name == "lattice_abc":
                    lattice_abc = block
                    lattice_cart = None
                    continue
                if canonical_name == "positions_frac":
                    positions_frac = block
                    positions_abs = None
                    continue
                if canonical_name == "positions_abs":
                    positions_abs = block
                    positions_frac = None
                    continue
                if canonical_name == "species_pot":
                    species_pot = tuple(block.rows)
                    continue
                if canonical_name == "kpoints_list":
                    kpoints_list = tuple(block.rows)
                    kpoint_mp_grid = None
                    kpoint_mp_spacing = None
                    continue
                cell_blocks.append(block)
                continue
            param_blocks.append(block)
            continue

        if spec.file == "cell":
            if canonical_name == "symmetry_generate":
                symmetry_generate = bool(raw_value)
                continue
            if canonical_name == "kpoint_mp_grid":
                assignment = make_keyword_assignment(spec, raw_value)
                if assignment is None:
                    continue
                triplet = tuple(int(component) for component in assignment.value)  # type: ignore[arg-type]
                if len(triplet) != 3:
                    raise ValueError("kpoint_mp_grid requires three integer values.")
                kpoint_mp_grid = (triplet[0], triplet[1], triplet[2])
                kpoint_mp_spacing = None
                kpoints_list = ()
                continue
            if canonical_name == "kpoint_mp_spacing":
                assignment = make_keyword_assignment(spec, raw_value)
                if assignment is None:
                    continue
                kpoint_mp_spacing = assignment.value  # type: ignore[assignment]
                kpoint_mp_grid = None
                kpoints_list = ()
                continue

        assignment = make_keyword_assignment(spec, raw_value)
        if assignment is None:
            continue
        if spec.file == "param":
            param_keywords.append(assignment)
        else:
            cell_scalars.append(assignment)

    cell_template = None
    if (
        lattice_cart is not None
        or lattice_abc is not None
        or positions_frac is not None
        or positions_abs is not None
        or species_pot is not None
        or kpoint_mp_grid is not None
        or kpoint_mp_spacing is not None
        or kpoints_list is not None
        or symmetry_generate is not None
        or cell_scalars
        or cell_blocks
    ):
        cell_template = CellTemplate(
            lattice_cart=lattice_cart,
            lattice_abc=lattice_abc,
            positions_frac=positions_frac,
            positions_abs=positions_abs,
            species_pot=species_pot,
            kpoint_mp_grid=kpoint_mp_grid,
            kpoint_mp_spacing=kpoint_mp_spacing,
            kpoints_list=kpoints_list,
            symmetry_generate=symmetry_generate,
            extra_scalars=tuple(cell_scalars),
            extra_blocks=tuple(cell_blocks),
        )

    param_template = (
        ParamTemplate(keywords=tuple(param_keywords), extra_blocks=tuple(param_blocks))
        if param_keywords or param_blocks
        else None
    )
    return InputTemplate(cell=cell_template, param=param_template)


def _quantum_method_from_gui_value(value: object) -> str:
    normalized = str(value).strip().upper()
    try:
        return _QUANTUM_GUI_TO_METHOD[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported phonon method '{value}'.") from exc


def _quantum_method_to_gui_value(value: object) -> str:
    normalized = str(value).strip().lower()
    return _QUANTUM_METHOD_TO_GUI.get(normalized, "FINITEDISPLACEMENT")


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


@dataclass(frozen=True, slots=True)
class ProjectTrailDestination:
    """Describe one clickable landmark on the project home trail."""

    title: str
    route: str
    theme: str
    completed: bool = False
    completion_goal: str | None = None
    placement: str = "trail"


@dataclass(frozen=True, slots=True)
class _QuantumSourceCandidate:
    """Internal source candidate for quantum-corrections job generation."""

    key: str
    site: MuonSite
    run: CalculationRun
    job: Job
    structure: Structure


@dataclass(frozen=True, slots=True)
class _QuantumStageState:
    """Current two-stage Quantum Corrections state for one source row."""

    stage: str
    status: str
    current_job: Job | None = None


@dataclass(frozen=True, slots=True)
class _EnergyBarrierSourceCandidate:
    """Internal path candidate for Energy Barriers job generation."""

    key: str
    reactant: _QuantumSourceCandidate
    product: _QuantumSourceCandidate
    path: MuonPath
    path_index: int


@dataclass(frozen=True, slots=True)
class _EnergyBarrierNebCandidate:
    """Endpoint path source eligible for NEB generation.

    NEB is seeded directly from the two relaxed endpoints; CASTEP interpolates
    the band between them, so no transition-state search is required.
    """

    key: str
    source: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _EnergyBarrierCompletedNebCandidate:
    """Completed NEB result eligible for preview."""

    key: str
    run: CalculationRun
    job: Job
    result: Result


class MuonChargeStateDialog(QDialog):
    """Choose which muon charge state tabs receive generated positions."""

    POSITIVE_ONLY = "positive"
    MUONIUM_ONLY = "muonium"
    BOTH = "both"

    def __init__(self, *, site_count: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("muonChargeStateDialog")
        self.setWindowTitle("Create Muon Relaxation Jobs")
        self.resize(480, 280)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._positive = QRadioButton("Positive muon", self)
        self._positive.setProperty("chargeStateOption", "true")
        self._positive.setChecked(True)
        self._muonium = QRadioButton("Muonium", self)
        self._muonium.setProperty("chargeStateOption", "true")
        self._both = QRadioButton("Both charge states", self)
        self._both.setProperty("chargeStateOption", "true")
        for index, button in enumerate((self._positive, self._muonium, self._both)):
            self._button_group.addButton(button, index)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(10)

        title = QLabel("Choose Charge State", self)
        style_section_title(title)
        layout.addWidget(title)

        noun = "position" if site_count == 1 else "positions"
        subtitle = QLabel(
            f"Create relaxation jobs for {site_count} selected initial muon {noun}.",
            self,
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(subtitle)

        card = QWidget(self)
        card.setObjectName("muonChargeStateCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(8)
        card_layout.addWidget(self._positive)
        card_layout.addWidget(self._muonium)
        card_layout.addWidget(self._both)
        layout.addWidget(card)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create Jobs")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_charge_states(self) -> tuple[str, ...]:
        if self._both.isChecked():
            return (POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE)
        if self._muonium.isChecked():
            return (MUONIUM_CHARGE_STATE,)
        return (POSITIVE_MUON_CHARGE_STATE,)


class _MuonGenerationThread(QThread):
    """Run site generation away from the Qt main thread."""

    generated = Signal(object)
    failed = Signal(str)
    progress_changed = Signal(int, int)

    def __init__(
        self,
        service: ProjectService,
        document: ProjectDocument,
        *,
        method: str,
        parameters: dict,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._document = document
        self._method = method
        self._parameters = dict(parameters)

    @property
    def method(self) -> str:
        return self._method

    @property
    def parameters(self) -> dict:
        return dict(self._parameters)

    def run(self) -> None:
        try:
            created = self._service.generate_candidate_sites(
                self._document,
                method=self._method,
                parameters=self._parameters,
                progress_callback=self.progress_changed.emit,
            )
        except (SiteGenerationError, ValueError) as exc:
            self.failed.emit(str(exc))
            return
        self.generated.emit(created)


class ProjectHomeTrailWidget(QWidget):
    """Paint the clickable project-home trail map."""

    route_requested = Signal(str)

    _SOURCE_WIDTH = 1536.0
    _SOURCE_HEIGHT = 1024.0
    _ARTWORK_SOURCE_REGION = QRectF(80.0, 44.0, 1421.0, 800.0)
    _SIGN_REGIONS = {
        "host_structure": QRectF(114.0, 340.0, 196.0, 46.0),
        "supercell_editor": QRectF(422.0, 315.0, 262.0, 46.0),
        "convergence_cutoff": QRectF(356.0, 503.0, 252.0, 46.0),
        "convergence_kpoint": QRectF(763.0, 311.0, 220.0, 46.0),
        "pristine_relaxation": QRectF(835.0, 502.0, 250.0, 46.0),
        "uep": QRectF(1078.0, 276.0, 164.0, 47.0),
        "muon_candidates": QRectF(1275.0, 382.0, 201.0, 47.0),
        "magnetic_fields": QRectF(1128.0, 586.0, 196.0, 47.0),
        "quantum_corrections": QRectF(711.0, 758.0, 225.0, 47.0),
        "energy_barriers": QRectF(224.0, 747.0, 245.0, 47.0),
    }
    _SIGN_TEXT_INSETS = {
        "host_structure": (12.0, 1.0, 14.0, 10.0),
        "supercell_editor": (14.0, 1.0, 16.0, 10.0),
        "convergence_cutoff": (12.0, 1.0, 14.0, 10.0),
        "convergence_kpoint": (12.0, 1.0, 14.0, 10.0),
        "pristine_relaxation": (14.0, 1.0, 16.0, 10.0),
        "uep": (14.0, 1.0, 16.0, 10.0),
        "muon_candidates": (8.0, 1.0, 10.0, 10.0),
        "magnetic_fields": (6.0, 1.0, 8.0, 10.0),
        "quantum_corrections": (6.0, 1.0, 8.0, 10.0),
        "energy_barriers": (14.0, 1.0, 16.0, 10.0),
    }
    _SIGN_TEXT_OFFSETS = {
        "host_structure": (0.0, 6.0),
        "supercell_editor": (-16.0, 3.0),
        "convergence_cutoff": (-20.0, 6.0),
        "convergence_kpoint": (-5.0, 5.0),
        "pristine_relaxation": (-12.0, 6.0),
        "muon_candidates": (9.0, -3.0),
        "magnetic_fields": (0.0, 12.0),
        "quantum_corrections": (12.0, 3.0),
    }
    _SIGN_HIGHLIGHT_OFFSETS = {
        "supercell_editor": (-14.0, 0.0),
        "convergence_cutoff": (-14.0, 0.0),
        "muon_candidates": (8.0, -5.0),
    }
    _SIGN_TEXT_MAX_SIZES = {
        "magnetic_fields": 28,
        "quantum_corrections": 28,
    }
    _LANDMARK_REGIONS = {
        "host_structure": QRectF(70.0, 100.0, 260.0, 230.0),
        "supercell_editor": QRectF(385.0, 30.0, 330.0, 275.0),
        "convergence_cutoff": QRectF(280.0, 370.0, 465.0, 165.0),
        "convergence_kpoint": QRectF(720.0, 70.0, 260.0, 220.0),
        "pristine_relaxation": QRectF(780.0, 360.0, 330.0, 150.0),
        "uep": QRectF(965.0, 35.0, 295.0, 210.0),
        "muon_candidates": QRectF(1288.0, 130.0, 225.0, 240.0),
        "magnetic_fields": QRectF(1070.0, 420.0, 290.0, 180.0),
        "quantum_corrections": QRectF(690.0, 590.0, 285.0, 200.0),
        "energy_barriers": QRectF(210.0, 585.0, 360.0, 195.0),
        "provenance_log": QRectF(1080.0, 675.0, 355.0, 170.0),
    }
    _CHECK_POINTS = {
        "host_structure": QPointF(294.0, 331.0),
        "supercell_editor": QPointF(660.0, 304.0),
        "convergence_cutoff": QPointF(580.0, 492.0),
        "convergence_kpoint": QPointF(982.0, 301.0),
        "pristine_relaxation": QPointF(1062.0, 492.0),
        "uep": QPointF(1223.0, 274.0),
        "muon_candidates": QPointF(1468.0, 364.0),
        "magnetic_fields": QPointF(1320.0, 578.0),
        "quantum_corrections": QPointF(913.0, 756.0),
        "energy_barriers": QPointF(450.0, 745.0),
    }
    _LOGO_COVER_REGION = QRectF(34.0, 780.0, 174.0, 190.0)
    # Banner height (logical px) at a typical window; signpost label sizes are
    # scaled relative to this so they stay proportionate as the map grows/shrinks.
    _SIGN_REF_BANNER_HEIGHT = 34.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._destinations: list[ProjectTrailDestination] = []
        self._hit_rects: dict[str, QRectF] = {}
        self._hovered_route: str | None = None
        self._logo_pixmap = QPixmap(logo_path())
        self._map_pixmap = QPixmap(project_home_map_path())
        self._color_map_pixmap = QPixmap(project_home_color_map_path())
        self.setObjectName("projectHomeTrail")
        self.setMinimumSize(720, 480)
        self.setMouseTracking(True)
        self.setAccessibleName("Project workflow home trail")

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return QSize(960, 640)

    def set_destinations(self, destinations: list[ProjectTrailDestination]) -> None:
        self._destinations = list(destinations)
        self.setAccessibleName(
            "Project workflow home trail: "
            + ", ".join(destination.title for destination in self._destinations)
        )
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if event.button() == Qt.MouseButton.LeftButton:
            point = event.position()
            for route, rect in self._destination_hit_rects().items():
                if rect.contains(point):
                    self.route_requested.emit(route)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        hovered_route = self._route_at(event.position())
        if hovered_route != self._hovered_route:
            self._hovered_route = hovered_route
            self.setCursor(Qt.CursorShape.PointingHandCursor if hovered_route else Qt.CursorShape.ArrowCursor)
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if self._hovered_route is not None:
            self._hovered_route = None
            self.unsetCursor()
            self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().paintEvent(event)
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        painter.fillRect(self.rect(), QColor(palette.bg_primary))
        map_rect = self._map_rect()
        if self._map_pixmap.isNull():
            self._paint_map_background(painter, palette, map_rect)
            self._paint_terrain(painter, palette, map_rect)
            self._paint_trail_path(painter, palette, map_rect)
            for destination in self._destinations:
                self._paint_destination(painter, palette, destination, map_rect)
        else:
            self._paint_map_artwork(painter, map_rect)
        self._hit_rects = self._destination_hit_rects(map_rect)

        self._paint_hover_region(painter, palette, map_rect)
        if not self._map_pixmap.isNull():
            self._paint_sign_titles(painter, palette, map_rect)
        self._paint_completion_seals(painter, palette, map_rect)

        painter.end()

    def _map_rect(self) -> QRectF:
        available = QRectF(self.rect()).adjusted(8.0, 8.0, -8.0, -8.0)
        if available.isEmpty():
            return QRectF()
        source_region = self._active_source_region()
        source_ratio = source_region.width() / source_region.height()
        available_ratio = available.width() / max(1.0, available.height())
        if available_ratio > source_ratio:
            height = available.height()
            width = height * source_ratio
        else:
            width = available.width()
            height = width / source_ratio
        return QRectF(
            available.center().x() - width / 2.0,
            available.center().y() - height / 2.0,
            width,
            height,
        )

    def _active_source_region(self) -> QRectF:
        if self._map_pixmap.isNull():
            return QRectF(0.0, 0.0, self._SOURCE_WIDTH, self._SOURCE_HEIGHT)
        return self._ARTWORK_SOURCE_REGION

    def _map_rect_from_source(self, source_rect: QRectF, map_rect: QRectF | None = None) -> QRectF:
        target = map_rect if map_rect is not None else self._map_rect()
        source_region = self._active_source_region()
        return QRectF(
            target.left() + target.width() * ((source_rect.left() - source_region.left()) / source_region.width()),
            target.top() + target.height() * ((source_rect.top() - source_region.top()) / source_region.height()),
            target.width() * (source_rect.width() / source_region.width()),
            target.height() * (source_rect.height() / source_region.height()),
        )

    def _map_point_from_source(self, point: QPointF, map_rect: QRectF | None = None) -> QPointF:
        target = map_rect if map_rect is not None else self._map_rect()
        source_region = self._active_source_region()
        return QPointF(
            target.left() + target.width() * ((point.x() - source_region.left()) / source_region.width()),
            target.top() + target.height() * ((point.y() - source_region.top()) / source_region.height()),
        )

    def _destination_hit_rects(self, map_rect: QRectF | None = None) -> dict[str, QRectF]:
        hit_rects: dict[str, QRectF] = {}
        for destination in self._destinations:
            region = self._LANDMARK_REGIONS.get(destination.route)
            if region is None:
                region = self._SIGN_REGIONS.get(destination.route)
            if region is not None:
                hit_rects[destination.route] = self._map_rect_from_source(region, map_rect).adjusted(
                    -8.0,
                    -8.0,
                    8.0,
                    8.0,
                )
                sign_region = self._SIGN_REGIONS.get(destination.route)
                if sign_region is not None:
                    hit_rects[destination.route] = hit_rects[destination.route].united(
                        self._map_rect_from_source(sign_region, map_rect).adjusted(-6.0, -6.0, 6.0, 6.0)
                    )
        return hit_rects

    def _route_at(self, point: QPointF) -> str | None:
        for route, rect in self._destination_hit_rects().items():
            if rect.contains(point):
                return route
        return None

    def _paint_map_background(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        if map_rect.isEmpty():
            return

        painter.setPen(QPen(QColor(palette.border).darker(135), 2.0))
        painter.setBrush(QColor(palette.bg_surface))
        painter.drawRoundedRect(map_rect, 18.0, 18.0)

        inner = map_rect.adjusted(10.0, 10.0, -10.0, -10.0)
        painter.setPen(QPen(QColor(palette.border), 1.0, Qt.PenStyle.DotLine))
        painter.setBrush(QColor(palette.bg_primary))
        painter.drawRoundedRect(inner, 12.0, 12.0)

        painter.setPen(QPen(QColor(palette.border), 1.0, Qt.PenStyle.DotLine))
        for index in range(4):
            y = map_rect.top() + map_rect.height() * (0.18 + index * 0.14)
            path = QPainterPath(QPointF(map_rect.left() + 24.0, y))
            path.cubicTo(
                QPointF(map_rect.left() + map_rect.width() * 0.28, y - 16.0),
                QPointF(map_rect.left() + map_rect.width() * 0.58, y + 18.0),
                QPointF(map_rect.right() - 24.0, y - 6.0),
            )
            painter.drawPath(path)

    def _source_rect(self, left: float, top: float, width: float, height: float, map_rect: QRectF) -> QRectF:
        return self._map_rect_from_source(QRectF(left, top, width, height), map_rect)

    def _paint_map_artwork(self, painter: QPainter, map_rect: QRectF) -> None:
        if map_rect.isEmpty():
            return
        painter.drawPixmap(map_rect, self._map_pixmap, self._active_source_region())

    def _paint_terrain(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        water = QColor("#2E7D8A" if palette.mode == THEME_LIGHT else "#2A6472")
        water.setAlpha(34 if palette.mode == THEME_LIGHT else 48)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(water)
        painter.drawEllipse(self._source_rect(-166.0, 430.0, 250.0, 190.0, map_rect))
        painter.drawEllipse(self._source_rect(1460.0, 430.0, 210.0, 212.0, map_rect))
        painter.drawEllipse(self._source_rect(740.0, 982.0, 430.0, 72.0, map_rect))

        wave_pen = QColor(palette.bg_surface)
        wave_pen.setAlpha(120)
        painter.setPen(QPen(wave_pen, 1.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for offset in (0.0, 20.0):
            wave = QPainterPath(self._map_point_from_source(QPointF(22.0, 486.0 + offset), map_rect))
            wave.cubicTo(
                self._map_point_from_source(QPointF(64.0, 462.0 + offset), map_rect),
                self._map_point_from_source(QPointF(96.0, 512.0 + offset), map_rect),
                self._map_point_from_source(QPointF(132.0, 488.0 + offset), map_rect),
            )
            painter.drawPath(wave)

        for source_rect in (
            QRectF(260.0, 72.0, 112.0, 136.0),
            QRectF(675.0, 92.0, 96.0, 86.0),
            QRectF(1298.0, 86.0, 118.0, 92.0),
            QRectF(280.0, 638.0, 250.0, 188.0),
            QRectF(1018.0, 508.0, 95.0, 76.0),
        ):
            painter.save()
            painter.setOpacity(0.28)
            self._draw_mountain_group(painter, palette, self._map_rect_from_source(source_rect, map_rect))
            painter.restore()

        for point in (
            QPointF(210.0, 410.0),
            QPointF(182.0, 556.0),
            QPointF(258.0, 585.0),
            QPointF(710.0, 862.0),
            QPointF(1260.0, 500.0),
            QPointF(1372.0, 532.0),
            QPointF(690.0, 230.0),
            QPointF(1186.0, 164.0),
        ):
            painter.save()
            painter.setOpacity(0.5)
            self._draw_tree(painter, palette, self._map_point_from_source(point, map_rect), map_rect.width() * 0.018)
            painter.restore()

    def _paint_trail_path(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        trail_points = [
            QPointF(300.0, 370.0),
            QPointF(378.0, 320.0),
            QPointF(520.0, 338.0),
            QPointF(748.0, 338.0),
            QPointF(842.0, 414.0),
            QPointF(860.0, 560.0),
            QPointF(650.0, 730.0),
            QPointF(480.0, 824.0),
            QPointF(735.0, 778.0),
            QPointF(978.0, 636.0),
            QPointF(1150.0, 628.0),
            QPointF(1230.0, 518.0),
            QPointF(1228.0, 378.0),
            QPointF(1335.0, 392.0),
        ]
        if len(trail_points) < 2:
            return

        path = QPainterPath(self._map_point_from_source(trail_points[0], map_rect))
        for start, end in zip(trail_points, trail_points[1:]):
            start_point = self._map_point_from_source(start, map_rect)
            end_point = self._map_point_from_source(end, map_rect)
            mid_x = (start_point.x() + end_point.x()) / 2.0
            path.cubicTo(QPointF(mid_x, start_point.y()), QPointF(mid_x, end_point.y()), end_point)

        painter.setPen(
            QPen(
                QColor(palette.text_muted).darker(130),
                max(2.0, map_rect.width() * 0.008),
                Qt.PenStyle.DotLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.drawPath(path)

    def _draw_mountain_group(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.1))
        painter.setBrush(QColor(palette.bg_surface_alt))
        for x_frac, peak_frac, width_frac in ((0.0, 0.18, 0.48), (0.28, 0.02, 0.52), (0.58, 0.24, 0.42)):
            left = rect.left() + rect.width() * x_frac
            width = rect.width() * width_frac
            peak = QPointF(left + width * 0.5, rect.top() + rect.height() * peak_frac)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(left, rect.bottom()),
                        peak,
                        QPointF(left + width, rect.bottom()),
                    ]
                )
            )
            painter.setPen(QPen(QColor(palette.bg_surface), 1.0))
            painter.drawLine(peak, QPointF(peak.x() - width * 0.12, peak.y() + rect.height() * 0.18))
            painter.drawLine(peak, QPointF(peak.x() + width * 0.14, peak.y() + rect.height() * 0.22))
            painter.setPen(QPen(QColor(palette.border), 1.1))

    def _draw_tree(self, painter: QPainter, palette, base: QPointF, size: float) -> None:
        painter.setPen(QPen(QColor(palette.border).darker(135), 1.0))
        painter.setBrush(QColor("#5D9A57" if palette.mode == THEME_LIGHT else "#3D7661"))
        for index in range(3):
            y = base.y() - size * (0.8 + index * 0.55)
            half_width = size * (0.75 - index * 0.12)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(base.x() - half_width, y + size * 0.7),
                        QPointF(base.x(), y),
                        QPointF(base.x() + half_width, y + size * 0.7),
                    ]
                )
            )
        painter.setPen(QPen(QColor(palette.border).darker(150), 1.4))
        painter.drawLine(base, QPointF(base.x(), base.y() - size * 0.7))

    def _paint_logo_overlay(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        cover_rect = self._map_rect_from_source(self._LOGO_COVER_REGION, map_rect)
        painter.setPen(QPen(QColor(palette.border).darker(135), 1.2))
        painter.setBrush(QColor("#D6AE6B"))
        painter.drawEllipse(cover_rect.adjusted(6.0, 10.0, -6.0, -10.0))

        if self._logo_pixmap.isNull():
            return
        logo_rect = cover_rect.adjusted(32.0, 40.0, -32.0, -40.0)
        painter.drawPixmap(logo_rect, self._logo_pixmap, QRectF(self._logo_pixmap.rect()))

    def _paint_hover_region(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        if self._hovered_route is None:
            return
        rect = self._hit_rects.get(self._hovered_route)
        if rect is None:
            return
        if not self._map_pixmap.isNull() and not self._color_map_pixmap.isNull() and not map_rect.isEmpty():
            self._paint_colored_destination_artwork(painter, map_rect, self._hovered_route)

        sign_source = self._SIGN_REGIONS.get(self._hovered_route)
        if sign_source is None:
            return
        rect = self._map_rect_from_source(sign_source, map_rect)
        highlight_offset = self._SIGN_HIGHLIGHT_OFFSETS.get(self._hovered_route)
        if highlight_offset is not None:
            scale = rect.height() / 46.0
            rect.translate(highlight_offset[0] * scale, highlight_offset[1] * scale)

        painter.save()
        clip_path = QPainterPath()
        clip_path.addRoundedRect(rect, 6.0, 6.0)
        painter.setClipPath(clip_path)

        glow = QColor(palette.seal_gold)
        glow.setAlpha(70)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawRoundedRect(rect, 6.0, 6.0)

        wash = QColor("#FFFFFF")
        wash.setAlpha(46)
        painter.setBrush(wash)
        painter.drawRoundedRect(rect.adjusted(3.0, 3.0, -3.0, -3.0), 4.5, 4.5)
        painter.restore()

    def _paint_colored_destination_artwork(self, painter: QPainter, map_rect: QRectF, route: str) -> None:
        reveal_path = QPainterPath()
        landmark_source = self._LANDMARK_REGIONS.get(route)
        if landmark_source is not None:
            landmark_rect = self._map_rect_from_source(landmark_source, map_rect).adjusted(-8.0, -8.0, 8.0, 8.0)
            radius = min(28.0, max(12.0, landmark_rect.height() * 0.18))
            reveal_path.addRoundedRect(landmark_rect, radius, radius)

        sign_source = self._SIGN_REGIONS.get(route)
        if sign_source is not None:
            sign_rect = self._map_rect_from_source(sign_source, map_rect).adjusted(-4.0, -4.0, 4.0, 4.0)
            reveal_path.addRoundedRect(sign_rect, 8.0, 8.0)

        if reveal_path.isEmpty():
            return

        painter.save()
        painter.setClipPath(reveal_path)
        painter.drawPixmap(map_rect, self._color_map_pixmap, self._active_source_region())
        painter.restore()

    def _paint_sign_titles(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        for destination in self._destinations:
            sign_source = self._SIGN_REGIONS.get(destination.route)
            if sign_source is None:
                continue
            sign_rect = self._map_rect_from_source(sign_source, map_rect)
            self._draw_sign_title(painter, palette, sign_rect, destination.title, destination.route)

    @staticmethod
    def _workflow_scroll_label_color() -> QColor:
        return QColor(light_palette().text_primary)

    @classmethod
    def _sign_title_point_size(cls, route: str, title: str, banner_height: float) -> float:
        """Point size for a signpost label, scaled to the rendered banner.

        A fixed point size made labels look generous on small windows but shrink
        to near-illegible on large or high-DPI displays. Tying the size to the
        banner height keeps the same proportion at every window size. The
        reference height matches the banner at a typical window, so the
        already-tuned default appearance is preserved; the result is floored for
        legibility and capped so it never overwhelms its banner.
        """

        base = float(cls._SIGN_TEXT_MAX_SIZES.get(route, 13 if len(title) <= 16 else 11))
        scale = banner_height / cls._SIGN_REF_BANNER_HEIGHT
        return max(11.0, min(base * scale, base * 2.0))

    def _draw_sign_title(self, painter: QPainter, palette, rect: QRectF, title: str, route: str) -> None:
        left, top, right, bottom = self._SIGN_TEXT_INSETS.get(route, (18.0, 5.0, 20.0, 8.0))
        text_rect = rect.adjusted(left, top, -right, -bottom)
        offset_x, offset_y = self._SIGN_TEXT_OFFSETS.get(route, (0.0, 0.0))
        offset_scale = rect.height() / 46.0
        text_rect.translate(offset_x * offset_scale, offset_y * offset_scale)

        font = QFont(heading_font_family())
        font.setBold(True)
        font.setPointSizeF(self._sign_title_point_size(route, title, rect.height()))
        while font.pointSizeF() > 8.0 and QFontMetricsF(font).horizontalAdvance(title) > text_rect.width() - 8.0:
            font.setPointSizeF(font.pointSizeF() - 1.0)

        painter.setFont(font)
        painter.setPen(self._workflow_scroll_label_color())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextSingleLine),
            title,
        )

    def _paint_completion_seals(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        completed_routes = {destination.route for destination in self._destinations if destination.completed}
        for route in completed_routes:
            source_point = self._CHECK_POINTS.get(route)
            if source_point is None:
                continue
            center = self._map_point_from_source(source_point, map_rect)
            self._draw_completion_seal(painter, palette, center)

    def _paint_destination(
        self,
        painter: QPainter,
        palette,
        destination: ProjectTrailDestination,
        map_rect: QRectF,
    ) -> None:
        landmark_source = self._LANDMARK_REGIONS.get(destination.route)
        sign_source = self._SIGN_REGIONS.get(destination.route)
        if landmark_source is None or sign_source is None:
            return

        landmark_rect = self._map_rect_from_source(landmark_source, map_rect)
        sign_rect = self._map_rect_from_source(sign_source, map_rect)

        self._draw_scaled_landmark(painter, palette, destination.theme, landmark_rect)
        self._draw_sign(painter, palette, sign_rect, destination.title)

    def _draw_scaled_landmark(self, painter: QPainter, palette, theme: str, rect: QRectF) -> None:
        if rect.isEmpty():
            return
        painter.save()
        painter.translate(rect.left(), rect.top())
        painter.scale(rect.width() / 68.0, rect.height() / 54.0)
        self._draw_landmark(painter, palette, theme, QRectF(0.0, 0.0, 68.0, 54.0))
        painter.restore()

    def _draw_sign(self, painter: QPainter, palette, rect: QRectF, title: str) -> None:
        post_x = rect.center().x() - 3.0
        painter.setPen(QPen(QColor(palette.border).darker(145), 1.0))
        painter.setBrush(QColor(palette.border).darker(115))
        painter.drawRoundedRect(QRectF(post_x, rect.bottom() - 2.0, 6.0, 28.0), 2.0, 2.0)

        painter.setPen(QPen(QColor(palette.border).darker(140), 1.1))
        painter.setBrush(QColor(palette.bg_surface))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        font = QFont(body_font_family())
        font.setPointSize(8 if len(title) > 16 else 9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(palette.text_primary))
        painter.drawText(
            rect.adjusted(7.0, 3.0, -7.0, -3.0),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap),
            title,
        )

    def _draw_completion_seal(self, painter: QPainter, palette, center: QPointF) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.1))
        painter.setBrush(QColor(palette.seal_gold))
        painter.drawEllipse(center, 14.0, 14.0)
        painter.setPen(QPen(QColor(palette.bg_surface), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(center.x() - 6.0, center.y()), QPointF(center.x() - 2.0, center.y() + 5.0))
        painter.drawLine(QPointF(center.x() - 2.0, center.y() + 5.0), QPointF(center.x() + 7.0, center.y() - 6.0))

    def _draw_landmark(self, painter: QPainter, palette, theme: str, rect: QRectF) -> None:
        if theme == "house":
            self._draw_house(painter, palette, rect)
        elif theme == "castle":
            self._draw_castle(painter, palette, rect)
        elif theme == "ocean":
            self._draw_ocean(painter, palette, rect)
        elif theme == "cube":
            self._draw_cube(painter, palette, rect)
        elif theme == "hammock":
            self._draw_hammock(painter, palette, rect)
        elif theme == "storm":
            self._draw_storm(painter, palette, rect)
        elif theme == "muon":
            self._draw_muon(painter, palette, rect)
        elif theme == "magnet":
            self._draw_magnet(painter, palette, rect)
        elif theme == "cat":
            self._draw_cat(painter, palette, rect)
        elif theme == "mountains":
            self._draw_mountains(painter, palette, rect)
        elif theme == "journal":
            self._draw_journal(painter, palette, rect)

    def _draw_house(self, painter: QPainter, palette, rect: QRectF) -> None:
        body = QRectF(rect.left() + 14, rect.top() + 24, 40, 26)
        roof = QPolygonF(
            [
                QPointF(body.left() - 4, body.top() + 2),
                QPointF(body.center().x(), rect.top() + 8),
                QPointF(body.right() + 4, body.top() + 2),
            ]
        )
        painter.setPen(QPen(QColor(palette.border), 1.2))
        painter.setBrush(QColor(palette.bg_surface))
        painter.drawRect(body)
        painter.setBrush(QColor(palette.accent))
        painter.drawPolygon(roof)
        painter.setBrush(QColor(palette.bg_surface_alt))
        painter.drawRect(QRectF(body.center().x() - 5, body.bottom() - 15, 10, 15))

    def _draw_castle(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.2))
        painter.setBrush(QColor(palette.bg_surface_alt))
        for x in (rect.left() + 10, rect.center().x() - 10, rect.right() - 30):
            painter.drawRect(QRectF(x, rect.top() + 14, 20, 36))
        painter.drawRect(QRectF(rect.left() + 16, rect.top() + 28, 36, 22))
        painter.setBrush(QColor(palette.seal_gold))
        for x in (rect.left() + 10, rect.center().x() - 10, rect.right() - 30):
            painter.drawRect(QRectF(x, rect.top() + 8, 20, 8))

    def _draw_ocean(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.accent), 2.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for index in range(4):
            y = rect.top() + 16 + index * 8
            path = QPainterPath(QPointF(rect.left() + 8, y))
            path.cubicTo(
                QPointF(rect.left() + 22, y - 10),
                QPointF(rect.left() + 34, y + 10),
                QPointF(rect.left() + 52, y),
            )
            painter.drawPath(path)

    def _draw_cube(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.1))
        colors = (palette.accent, palette.seal_gold, palette.bg_surface_alt)
        size = 12
        left = rect.left() + 14
        top = rect.top() + 10
        for row in range(3):
            for col in range(3):
                painter.setBrush(QColor(colors[(row + col) % len(colors)]))
                painter.drawRect(QRectF(left + col * size, top + row * size, size, size))

    def _draw_hammock(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.border).darker(140), 2.0))
        painter.drawLine(QPointF(rect.left() + 12, rect.bottom() - 6), QPointF(rect.left() + 20, rect.top() + 6))
        painter.drawLine(QPointF(rect.right() - 12, rect.bottom() - 6), QPointF(rect.right() - 20, rect.top() + 6))
        painter.setPen(QPen(QColor(palette.accent), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        path = QPainterPath(QPointF(rect.left() + 19, rect.top() + 17))
        path.cubicTo(
            QPointF(rect.left() + 28, rect.top() + 42),
            QPointF(rect.right() - 28, rect.top() + 42),
            QPointF(rect.right() - 19, rect.top() + 17),
        )
        painter.drawPath(path)

    def _draw_storm(self, painter: QPainter, palette, rect: QRectF) -> None:
        cloud = QColor(palette.bg_surface_alt)
        painter.setPen(QPen(QColor(palette.border), 1.2))
        painter.setBrush(cloud)
        painter.drawEllipse(QRectF(rect.left() + 10, rect.top() + 16, 26, 22))
        painter.drawEllipse(QRectF(rect.left() + 26, rect.top() + 9, 26, 28))
        painter.drawEllipse(QRectF(rect.left() + 38, rect.top() + 18, 20, 20))
        painter.setBrush(QColor(palette.seal_gold))
        bolt = QPolygonF(
            [
                QPointF(rect.center().x(), rect.top() + 30),
                QPointF(rect.center().x() - 8, rect.bottom()),
                QPointF(rect.center().x() + 2, rect.bottom() - 11),
                QPointF(rect.center().x() - 2, rect.bottom() - 11),
                QPointF(rect.center().x() + 8, rect.top() + 30),
            ]
        )
        painter.drawPolygon(bolt)

    def _draw_muon(self, painter: QPainter, palette, rect: QRectF) -> None:
        font = QFont(body_font_family())
        font.setPointSize(34)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(palette.accent))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "mu")

    def _draw_magnet(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.accent), 8.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(rect.left() + 14, rect.top() + 10, 40, 42), 0, 180 * 16)
        painter.setPen(QPen(QColor(palette.border), 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(rect.left() + 15, rect.top() + 30), QPointF(rect.left() + 15, rect.bottom() - 4))
        painter.drawLine(QPointF(rect.right() - 15, rect.top() + 30), QPointF(rect.right() - 15, rect.bottom() - 4))

    def _draw_cat(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.2))
        painter.setBrush(QColor(palette.bg_surface_alt))
        face = QRectF(rect.left() + 16, rect.top() + 16, 36, 30)
        painter.drawEllipse(face)
        ears = QPolygonF(
            [
                QPointF(face.left() + 4, face.top() + 7),
                QPointF(face.left() + 10, face.top() - 8),
                QPointF(face.left() + 17, face.top() + 6),
                QPointF(face.right() - 17, face.top() + 6),
                QPointF(face.right() - 10, face.top() - 8),
                QPointF(face.right() - 4, face.top() + 7),
            ]
        )
        painter.drawPolygon(ears)
        painter.setBrush(QColor(palette.text_primary))
        painter.drawEllipse(QPointF(face.left() + 13, face.top() + 14), 2.0, 2.0)
        painter.drawEllipse(QPointF(face.right() - 13, face.top() + 14), 2.0, 2.0)

    def _draw_mountains(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.2))
        painter.setBrush(QColor(palette.bg_surface_alt))
        painter.drawPolygon(
            QPolygonF(
                [
                    QPointF(rect.left() + 4, rect.bottom() - 4),
                    QPointF(rect.left() + 24, rect.top() + 8),
                    QPointF(rect.left() + 42, rect.bottom() - 4),
                ]
            )
        )
        painter.setBrush(QColor(palette.bg_surface))
        painter.drawPolygon(
            QPolygonF(
                [
                    QPointF(rect.left() + 28, rect.bottom() - 4),
                    QPointF(rect.left() + 48, rect.top() + 14),
                    QPointF(rect.right() - 4, rect.bottom() - 4),
                ]
            )
        )

    def _draw_journal(self, painter: QPainter, palette, rect: QRectF) -> None:
        book = QRectF(rect.left() + 16, rect.top() + 8, 36, 42)
        painter.setPen(QPen(QColor(palette.border), 1.2))
        painter.setBrush(QColor(palette.bg_surface))
        painter.drawRoundedRect(book, 4.0, 4.0)
        painter.setPen(QPen(QColor(palette.accent), 2.0))
        painter.drawLine(QPointF(book.left() + 8, book.top() + 7), QPointF(book.left() + 8, book.bottom() - 7))
        painter.setPen(QPen(QColor(palette.text_muted), 1.0))
        for index in range(3):
            y = book.top() + 14 + index * 8
            painter.drawLine(QPointF(book.left() + 15, y), QPointF(book.right() - 7, y))


class ConstructionSignWidget(QWidget):
    """Paint an in-progress workflow placeholder as a planted wooden sign."""

    _MESSAGE = "Under construction.\nCome back soon."

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 300)
        self.setAccessibleName(self._MESSAGE.replace("\n", " "))

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().paintEvent(event)
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        width = max(1, self.width())
        height = max(1, self.height())
        horizon = int(height * 0.72)
        painter.fillRect(self.rect(), QColor(palette.bg_primary))
        painter.fillRect(0, horizon, width, height - horizon, QColor(palette.bg_surface_alt))

        sign_width = int(min(560, max(260, width - 96)))
        sign_height = 144
        sign_x = int((width - sign_width) / 2)
        sign_y = int(max(36, min(horizon - sign_height - 58, height * 0.24)))
        post_top = sign_y + sign_height - 10
        post_bottom = min(height - 22, horizon + 104)

        post_width = 34
        post_x = int(width / 2 - post_width / 2)
        painter.setPen(QPen(QColor("#5C341E"), 1.2))
        painter.setBrush(QColor("#7A4728"))
        painter.drawRoundedRect(QRectF(post_x, post_top, post_width, post_bottom - post_top), 5, 5)
        painter.setPen(QPen(QColor("#A66A3A"), 1.0))
        painter.drawLine(post_x + 8, post_top + 8, post_x + 8, post_bottom - 8)

        plank_height = sign_height / 3
        for index, colour in enumerate(("#9B5F32", "#B3723D", "#8A512C")):
            plank_y = sign_y + index * plank_height
            painter.setPen(QPen(QColor("#5F371F"), 1.4))
            painter.setBrush(QColor(colour))
            painter.drawRoundedRect(QRectF(sign_x, plank_y, sign_width, plank_height + 2), 10, 10)
            painter.setPen(QPen(QColor("#C88B55"), 1.0))
            painter.drawLine(
                sign_x + 24,
                int(plank_y + plank_height * 0.46),
                sign_x + sign_width - 24,
                int(plank_y + plank_height * 0.46),
            )

        painter.setPen(QPen(QColor("#5F371F"), 1.2))
        painter.setBrush(QColor("#D2A169"))
        for nail_x in (sign_x + 26, sign_x + sign_width - 26):
            for nail_y in (sign_y + 22, sign_y + sign_height - 22):
                painter.drawEllipse(nail_x - 3, nail_y - 3, 6, 6)

        font = QFont(body_font_family(), 20)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#FFF8EA"), 1.0))
        painter.drawText(
            QRectF(sign_x + 28, sign_y + 18, sign_width - 56, sign_height - 36),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            self._MESSAGE,
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
    _PROJECT_TREE_MIN_WIDTH = 220
    _PROJECT_TREE_MAX_WIDTH = 300
    _ACTIVE_CONVERGENCE_DEFINITION_KEY = "active_convergence_cutoff_definition_id"
    _ACTIVE_KPOINT_DEFINITION_KEY = "active_convergence_kpoint_definition_id"
    _ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY = "active_pristine_relaxation_definition_id"
    _ACTIVE_MUON_RELAXATION_DEFINITION_KEY = "active_muon_relaxation_definition_id"
    _ACTIVE_UEP_DEFINITION_KEY = "active_uep_definition_id"
    _CONVERGENCE_EXAMINER_STATE_KEY = "convergence_examiner_state"
    _KPOINT_EXAMINER_STATE_KEY = "kpoint_examiner_state"
    _UEP_ANALYSIS_STATES_KEY = "uep_analysis_states"
    _UEP_POTENTIAL_GRID_RESULT_TYPE = "uep_potential_grid"
    _MAGNETIC_FIELD_FORM_KEY = "magnetic_field_form"
    _MAGNETIC_FIELD_WORKSPACES_KEY = "magnetic_field_workspaces"
    _MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX = "magnetic_fields/workspace/"
    _CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY = "convergence_overview_snapshots"
    _LAST_CLUSTER_SUBMISSION_CONFIG_KEY = "last_cluster_submission_config"
    _COMPARISON_ROUTE_PREFIX = "pristine_relaxation/comparison/"
    _WORKFLOW_COMPLETE_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    _WORKFLOW_GOAL_ROLE = int(Qt.ItemDataRole.UserRole) + 2

    def __init__(self) -> None:
        super().__init__()
        self._project_service = ProjectService()
        self._document: ProjectDocument | None = None
        # Tracks whether the job dashboard is actually visible to the user.
        # Driven by jobs_dock.visibilityChanged, which (unlike isHidden())
        # also reports False on window minimize / tabbed-dock obscuring.
        # Defaults True so an as-yet-unshown window still arms the timer.
        self._jobs_dock_visible = True
        self._settings = QSettings("MuFinder", "MuFinder")
        self._appearance_mode = load_theme_mode(self._settings)
        self._local_runner = LocalJobRunner(self)
        # Cluster SSH work runs on a dedicated worker thread via this facade,
        # which mirrors ClusterJobRunner's public surface (see cluster_controller).
        self._cluster_runner = ClusterController(self, settings=self._settings)
        self._active_run_by_job: dict[UUID, UUID] = {}
        self._job_execution = JobExecutionController(
            document_getter=lambda: self._document,
            project_service=self._project_service,
            local_runner=self._local_runner,
            cluster_runner=self._cluster_runner,
            job_output_directory=self._job_output_directory,
            job_seed_name=self._job_seed_name,
            active_run_by_job=self._active_run_by_job,
        )
        self._workflow_submission = WorkflowSubmissionCoordinator(dedupe_jobs=self._dedupe_jobs)
        self._job_runtime_timer = QTimer(self)
        self._job_runtime_timer.setInterval(1000)
        self._job_runtime_timer.timeout.connect(self._refresh_active_job_runtime_cells)
        # Job-lifecycle handlers (completion/status/cancel) coalesce their heavy
        # dashboard work — the full-document autosave AND the journal/jobs-table/
        # workflow-view rebuilds — through this timer. A burst of cluster jobs
        # finishing in one poll otherwise rebuilds every view and rewrites the
        # whole document to disk once per job, which freezes the GUI.
        self._dashboard_flush_pending = False
        self._dashboard_autosave_failure_message = ""
        self._dashboard_flush_timer = QTimer(self)
        self._dashboard_flush_timer.setSingleShot(True)
        self._dashboard_flush_timer.setInterval(300)
        self._dashboard_flush_timer.timeout.connect(self._flush_dashboard)
        self._cluster_elapsed_seconds_by_job: dict[UUID, int] = {}
        self._cluster_elapsed_checked_at: datetime | None = None
        self._shutting_down = False
        self._recent_projects_menu: QMenu | None = None
        self._home_recent_projects_layout: QVBoxLayout | None = None
        self._home_content_stack: QStackedWidget | None = None
        self._home_welcome_page: QWidget | None = None
        self._project_home_page: QWidget | None = None
        self._project_home_trail: ProjectHomeTrailWidget | None = None
        self._project_home_button: QToolButton | None = None
        self._cluster_connected = False
        self._comparison_widgets: dict[UUID, ComparisonTableWidget] = {}
        self._magnetic_field_workspace_widgets: dict[str, MagneticFieldWidget] = {}
        self._uep_analysis_dialogs: dict[UUID, UepAnalysisDialog] = {}
        self._quantum_analysis_dialogs: dict[UUID, QuantumPhononAnalysisDialog] = {}
        self._muon_generation_thread: _MuonGenerationThread | None = None
        self._muon_generation_failed = False

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
        self._host_structure_view.supercell_editor_requested.connect(self._on_host_supercell_editor_requested)
        self._host_structure_view.active_host_change_requested.connect(self._on_active_host_change_requested)
        self._supercell_editor_page = self._build_supercell_editor_page()
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
        self._pristine_relaxation_view.remove_requested.connect(
            self._on_remove_pristine_relaxation_rows
        )
        self._uep_view = UEPWidget()
        self._uep_view.configure_requested.connect(self._on_uep_configured)
        self._uep_view.form_state_changed.connect(self._on_uep_form_state_changed)
        self._uep_view.row_supercell_edit_requested.connect(self._on_uep_supercell_edit_requested)
        self._uep_view.source_structure_selection_requested.connect(
            self._on_uep_source_structure_selection_requested
        )
        self._uep_view.wizard_requested.connect(self._on_open_uep_analysis_requested)
        self._uep_view.submit_local_requested.connect(self._on_submit_local_from_uep)
        self._uep_view.submit_cluster_requested.connect(self._on_submit_cluster_from_uep)
        self._uep_view.cancel_requested.connect(self._on_cancel_from_uep)
        self._uep_view.inspect_failed_requested.connect(self._on_inspect_failed_from_uep)
        self._uep_view.reset_failed_requested.connect(self._on_reset_failed_from_uep)
        self._uep_view.remove_requested.connect(self._on_remove_uep_rows)
        self._muon_generation_view = MuonGenerationWidget()
        self._muon_generation_view.generate_requested.connect(self._on_generate_requested)
        self._muon_generation_view.cluster_positions_requested.connect(self._on_cluster_muon_positions_requested)
        self._muon_generation_view.delete_candidates_requested.connect(self._on_delete_muon_candidates_requested)
        self._muon_generation_view.create_jobs_requested.connect(self._on_create_muon_relaxation_jobs_requested)
        self._muon_generation_view.form_state_changed.connect(self._on_muon_form_state_changed)
        self._muon_relaxation_view = MuonRelaxationWidget()
        self._muon_relaxation_view.configure_requested.connect(self._on_muon_relaxation_configured)
        self._muon_relaxation_view.form_state_changed.connect(self._on_muon_relaxation_form_state_changed)
        self._muon_relaxation_view.row_supercell_edit_requested.connect(
            self._on_muon_relaxation_supercell_edit_requested
        )
        self._muon_relaxation_view.submit_local_requested.connect(self._on_submit_local_from_muon_relaxation)
        self._muon_relaxation_view.submit_cluster_requested.connect(self._on_submit_cluster_from_muon_relaxation)
        self._muon_relaxation_view.remove_requested.connect(self._on_remove_muon_relaxation_rows)
        self._muon_relaxation_view.analysis_workspace_requested.connect(
            self._on_create_relaxation_analysis_workspace_requested
        )
        self._relaxation_analysis_view = RelaxationAnalysisWidget()
        self._relaxation_analysis_view.state_changed.connect(self._on_relaxation_analysis_state_changed)
        self._relaxation_analysis_view.add_sites_requested.connect(self._on_relaxation_analysis_add_sites_requested)
        self._magnetic_field_view = MagneticFieldWidget()
        self._magnetic_field_view.magnetic_structure_changed.connect(
            self._on_magnetic_field_structure_changed
        )
        self._magnetic_field_view.form_state_changed.connect(self._on_magnetic_field_form_state_changed)
        self._magnetic_field_view.pin_requested.connect(self._on_magnetic_field_pin_requested)
        self._quantum_corrections_view = QuantumCorrectionsWidget()
        self._quantum_corrections_view.configure_requested.connect(self._on_configure_quantum_corrections_source)
        self._quantum_corrections_view.source_label_renamed.connect(
            self._on_quantum_corrections_source_label_renamed
        )
        self._quantum_corrections_view.submit_local_requested.connect(
            self._on_submit_local_from_quantum_corrections
        )
        self._quantum_corrections_view.submit_cluster_requested.connect(
            self._on_submit_cluster_from_quantum_corrections
        )
        self._quantum_corrections_view.reset_stage_requested.connect(
            self._on_reset_quantum_corrections_to_stage_one
        )
        self._quantum_corrections_view.analysis_requested.connect(
            self._on_launch_quantum_phonon_analysis
        )
        self._energy_barriers_view: EnergyBarriersWidget | None = None
        self._journal_view = JournalWidget()
        self._main_stack.addWidget(self._home_page)
        self._main_stack.addWidget(self._host_structure_view)
        self._main_stack.addWidget(self._supercell_editor_page)
        self._main_stack.addWidget(self._convergence_overview_view)
        self._main_stack.addWidget(self._convergence_view)
        self._main_stack.addWidget(self._kpoint_convergence_view)
        self._main_stack.addWidget(self._pristine_relaxation_view)
        self._main_stack.addWidget(self._uep_view)
        self._main_stack.addWidget(self._muon_generation_view)
        self._main_stack.addWidget(self._muon_relaxation_view)
        self._main_stack.addWidget(self._relaxation_analysis_view)
        self._main_stack.addWidget(self._magnetic_field_view)
        self._main_stack.addWidget(self._quantum_corrections_view)
        self._main_stack.addWidget(self._journal_view)

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
                "Resubmit",
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
        self._project_tree_panel.setMinimumWidth(self._PROJECT_TREE_MIN_WIDTH)
        self._project_tree_panel.setMaximumWidth(self._PROJECT_TREE_MAX_WIDTH)
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
        self._project_home_button.setToolTip("Show project home")
        self._project_home_button.setEnabled(False)
        self._project_home_button.clicked.connect(self._show_project_home)
        project_tree_header_layout.addWidget(self._project_home_button)

        self._project_tree_title = QLabel("Project", project_tree_header)
        self._project_tree_title.setObjectName("projectTreePanelTitle")
        self._project_tree_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        project_tree_header_layout.addWidget(self._project_tree_title, 1)
        project_tree_layout.addWidget(project_tree_header)

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
        # Pause the 1s runtime timer while the dock is not visible (closed,
        # obscured, or window minimized), and refresh + re-arm when it returns,
        # so we do not update an invisible table every second.
        jobs_dock.visibilityChanged.connect(self._on_jobs_dock_visibility_changed)
        self._jobs_delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, jobs_dock)
        self._jobs_delete_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._jobs_delete_shortcut.activated.connect(self._on_dashboard_delete_key_requested)
        self._jobs_backspace_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Backspace), jobs_dock)
        self._jobs_backspace_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._jobs_backspace_shortcut.activated.connect(self._on_dashboard_delete_key_requested)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, jobs_dock)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if getattr(self, "_shutting_down", False):
            super().closeEvent(event)
            return
        job_runtime_timer = getattr(self, "_job_runtime_timer", None)
        if job_runtime_timer is not None:
            job_runtime_timer.stop()
        dashboard_flush_timer = getattr(self, "_dashboard_flush_timer", None)
        if dashboard_flush_timer is not None:
            dashboard_flush_timer.stop()
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
                # Staying open: persist/refresh the coalesced flush that closeEvent
                # stopped the timer for, so it is not orphaned until the next job.
                self._flush_dashboard()
                self._sync_job_runtime_timer()
                event.ignore()
                return
            self._shutting_down = True
            self._close_open_uep_analysis_dialogs()
            if answer == QMessageBox.StandardButton.Yes:
                try:
                    self._project_service.save_project(self._document)
                except ValueError as exc:
                    self._shutting_down = False
                    QMessageBox.critical(self, "Save blocked", str(exc))
                    self._sync_job_runtime_timer()
                    event.ignore()
                    return
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
            muon_generation_thread.wait()
            self._muon_generation_thread = None
        self._shutdown_embedded_structure_viewers()
        local_runner = getattr(self, "_local_runner", None)
        if local_runner is not None:
            local_runner.shutdown()
        cluster_runner = getattr(self, "_cluster_runner", None)
        if cluster_runner is not None:
            cluster_runner.shutdown()
        super().closeEvent(event)

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

        self._home_content_stack = QStackedWidget(page)
        self._home_content_stack.setObjectName("homeContentStack")
        self._home_welcome_page = self._build_welcome_home_page()
        self._project_home_page = self._build_project_home_page()
        self._home_content_stack.addWidget(self._home_welcome_page)
        self._home_content_stack.addWidget(self._project_home_page)
        layout.addWidget(self._home_content_stack, 1)

        self._refresh_home_recent_projects()
        self._sync_project_home_trail()
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

        self._project_home_trail = ProjectHomeTrailWidget(page)
        self._project_home_trail.route_requested.connect(self._set_active_route)
        layout.addWidget(self._project_home_trail, 1)
        return page

    def _build_supercell_editor_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Supercell Editor")
        style_section_title(title)
        layout.addWidget(title)

        subtitle = QLabel("Manage reusable saved supercells for the active host structure.")
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(subtitle)

        self._supercell_editor_host_badge = ActiveHostBadge()
        layout.addWidget(self._supercell_editor_host_badge)

        open_button = QPushButton("Open Supercell Editor")
        open_button.clicked.connect(self._on_host_supercell_editor_requested)
        layout.addWidget(open_button)
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
        self._sync_project_home_trail()

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

        # Persist any coalesced flush for the outgoing project before swapping.
        self._flush_dashboard()
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

        # Persist any coalesced autosave for the outgoing project before swapping.
        self._flush_dashboard()
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

        # This writes the whole document, so any coalesced flush is subsumed.
        self._dashboard_flush_pending = False
        self._dashboard_flush_timer.stop()
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
            "Follow the workflow trail, or return here from the project home button."
        )
        self._project_tree_title.setText(f"Project: {self._document.project.name}")
        self._sync_comparison_table_views()
        self._sync_magnetic_field_workspace_views()
        self._populate_project_tree(preserve_selection=False)
        self._refresh_jobs_table()
        self._refresh_host_structure_view()
        self._refresh_magnetic_field_view()
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
        self._uep_view.set_form_state(self._document.ui_state.get("uep_form", {}))
        self._muon_generation_view.set_form_state(
            self._document.ui_state.get("muon_generation_form", {})
        )
        self._muon_relaxation_view.set_form_state(
            self._document.ui_state.get("muon_relaxation_form", {})
        )
        self._convergence_view.set_project_defaults(
            cutoff_energy=self._document.project.recommended_cutoff_energy,
            cutoff_method=self._document.project.recommended_cutoff_method,
            energy_threshold=self._document.project.recommended_cutoff_energy_threshold,
            force_threshold=self._document.project.recommended_cutoff_force_threshold,
            stress_threshold=self._document.project.recommended_cutoff_stress_threshold,
        )
        self._kpoint_convergence_view.set_project_defaults(
            kpoint_grid=self._document.project.recommended_kpoint_grid,
            kpoint_method=self._document.project.recommended_kpoint_method,
            energy_threshold=self._document.project.recommended_kpoint_energy_threshold,
            force_threshold=self._document.project.recommended_kpoint_force_threshold,
            stress_threshold=self._document.project.recommended_kpoint_stress_threshold,
        )
        self._kpoint_convergence_view.set_cutoff_default(
            self._document.project.recommended_cutoff_energy
        )
        self._refresh_pristine_relaxation_defaults()
        self._refresh_pristine_relaxation_rows()
        self._refresh_muon_relaxation_defaults()
        self._refresh_muon_relaxation_rows()
        self._refresh_relaxation_analysis_context()
        self._refresh_uep_defaults()
        self._refresh_uep_rows()
        self._refresh_uep_source_structures()
        self._refresh_uep_supercell_options()
        self._refresh_convergence_overview()
        self._refresh_journal_view()

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

        # Recompute point_spacings from the current host structure's lattice
        # vectors so that stale values persisted with an older formula are
        # always refreshed on load — even for projects adopted before the fix.
        host_id = self._document.project.host_structure_id
        host = self._document.structures.get(host_id) if host_id is not None else None
        if host is not None:
            try:
                snapshot = ConvergenceExaminerDialog.recompute_spacings_in_snapshot(
                    snapshot, host.lattice_vectors
                )
            except Exception:
                pass  # Safety: fall back to stored spacings on any unexpected error

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
                    stress_threshold=self._document.project.recommended_cutoff_stress_threshold,
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
                    stress_threshold=self._document.project.recommended_kpoint_stress_threshold,
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

    def _workflow_supercell_editor_complete(self, document: ProjectDocument) -> bool:
        return bool(
            self._project_service.list_workflow_supercell_assets(
                document,
                host_structure_id=document.project.host_structure_id,
            )
        )

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
    def _workflow_convergence_complete(document: ProjectDocument) -> bool:
        return (
            document.project.recommended_cutoff_energy is not None
            and document.project.recommended_kpoint_grid is not None
        )

    @staticmethod
    def _workflow_pristine_relaxation_complete(document: ProjectDocument) -> bool:
        return bool(document.comparison_tables)

    @staticmethod
    def _workflow_uep_complete(document: ProjectDocument) -> bool:
        return any(
            job.workflow_stage == WorkflowStage.UEP and job.status == JobStatus.COMPLETED
            for job in document.jobs.values()
        )

    @staticmethod
    def _workflow_quantum_corrections_complete(document: ProjectDocument) -> bool:
        return any(result.result_type == QUANTUM_PHONON_CALC_TYPE for result in document.results.values())

    @staticmethod
    def _workflow_energy_barriers_complete(document: ProjectDocument) -> bool:
        return any(result.result_type == TRANSITION_STATE_SEARCH_CALC_TYPE for result in document.results.values())

    @staticmethod
    def _active_muon_relaxation_definition_for_document(document: ProjectDocument):
        raw_definition_id = document.ui_state.get(MainWindow._ACTIVE_MUON_RELAXATION_DEFINITION_KEY)
        if isinstance(raw_definition_id, str):
            try:
                definition_id = UUID(raw_definition_id)
            except ValueError:
                definition_id = None
            if definition_id is not None:
                definition = document.calculations.get(definition_id)
                if (
                    definition is not None
                    and definition.workflow_stage == WorkflowStage.MUON_RELAXATION
                    and definition.calc_type == "castep_muon_relaxation"
                ):
                    return definition

        definitions = [
            definition
            for definition in document.calculations.values()
            if definition.workflow_stage == WorkflowStage.MUON_RELAXATION
            and definition.calc_type == "castep_muon_relaxation"
        ]
        definitions.sort(key=lambda definition: definition.created_at, reverse=True)
        return definitions[0] if definitions else None

    @staticmethod
    def _muon_relaxation_job_for_table_row(
        document: ProjectDocument,
        definition,
        label: str,
        charge_state: str,
    ) -> Job | None:
        keys = (f"{muon_charge_state_prefix(charge_state)}:{label}", label)
        mapping = definition.parameters.get("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            return None
        for key in keys:
            raw_job_id = mapping.get(key)
            if not isinstance(raw_job_id, str):
                continue
            try:
                job_id = UUID(raw_job_id)
            except ValueError:
                continue
            job = document.jobs.get(job_id)
            if job is not None and is_muon_relaxation_calc_type(job.calc_type):
                if charge_state and job.calc_type != muon_relaxation_calc_type(charge_state):
                    continue
                return job
        return None

    @staticmethod
    def _latest_run_for_document_job(document: ProjectDocument, job_id: UUID) -> CalculationRun | None:
        runs = [run for run in document.runs.values() if run.job_id == job_id]
        return max(runs, key=lambda run: run.run_index) if runs else None

    @staticmethod
    def _workflow_muon_relaxations_complete(document: ProjectDocument) -> bool:
        definition = MainWindow._active_muon_relaxation_definition_for_document(document)
        if definition is None:
            return False
        configuration = definition.parameters.get("configuration", {})
        rows = configuration.get("rows", []) if isinstance(configuration, dict) else []
        if not isinstance(rows, list):
            return False

        saw_row = False
        for entry in rows:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label", "")).strip()
            if not label:
                continue
            saw_row = True
            charge_state = str(entry.get("charge_state", "positive_muon"))
            job = MainWindow._muon_relaxation_job_for_table_row(
                document,
                definition,
                label,
                charge_state,
            )
            if job is None or job.status != JobStatus.COMPLETED:
                return False
            latest_run = MainWindow._latest_run_for_document_job(document, job.id)
            if latest_run is not None and latest_run.convergence_achieved is False:
                return False

        return saw_row

    @staticmethod
    def _workflow_relaxation_analysis_complete(document: ProjectDocument) -> bool:
        workspaces = document.ui_state.get(_RELAXATION_ANALYSIS_STATE_KEY)
        return isinstance(workspaces, list) and any(isinstance(workspace, dict) for workspace in workspaces)

    @staticmethod
    def _workflow_magnetic_fields_complete(document: ProjectDocument) -> bool:
        magnetic_structure = document.project.magnetic_structure
        if not isinstance(magnetic_structure, dict):
            return False
        components = magnetic_structure.get("components")
        if isinstance(components, list):
            return any(isinstance(component, dict) for component in components)
        moments = magnetic_structure.get("moments")
        return isinstance(moments, list) and bool(moments)

    def _magnetic_field_workspaces_state(self) -> list[dict[str, object]]:
        if self._document is None:
            return []
        workspaces = self._document.ui_state.get(self._MAGNETIC_FIELD_WORKSPACES_KEY, [])
        if not isinstance(workspaces, list):
            return []
        return [workspace for workspace in workspaces if isinstance(workspace, dict)]

    def _magnetic_field_workspace_route(self, workspace_id: str) -> str:
        return f"{self._MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX}{workspace_id}"

    def _magnetic_field_workspace_id_from_route(self, route: object) -> str | None:
        if not isinstance(route, str) or not route.startswith(self._MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX):
            return None
        workspace_id = route[len(self._MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX) :].strip()
        return workspace_id or None

    @staticmethod
    def _magnetic_workspace_has_structure(workspace: Mapping[str, object]) -> bool:
        magnetic_structure = workspace.get("magnetic_structure")
        if not isinstance(magnetic_structure, Mapping):
            return False
        components = magnetic_structure.get("components")
        if isinstance(components, list):
            return any(isinstance(component, dict) for component in components)
        moments = magnetic_structure.get("moments")
        return isinstance(moments, list) and bool(moments)

    def _workflow_magnetic_field_workspace_complete(self, workspace_id: str) -> bool:
        for workspace in self._magnetic_field_workspaces_state():
            if str(workspace.get("id", "")).strip() == workspace_id:
                return self._magnetic_workspace_has_structure(workspace)
        return False

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
        magnetic_workspaces = tuple(
            WorkflowDefinition(
                key=f"magnetic_field_workspace_{str(workspace.get('id', '')).strip()}",
                title=str(workspace.get("title", "Magnetic Fields")).strip() or "Magnetic Fields",
                route=self._magnetic_field_workspace_route(str(workspace.get("id", "")).strip()),
                completion_goal="Define at least one magnetic component in this saved workspace.",
                completion_check=(
                    lambda _document, workspace_id=str(workspace.get("id", "")).strip(): (
                        self._workflow_magnetic_field_workspace_complete(workspace_id)
                    )
                ),
            )
            for workspace in self._magnetic_field_workspaces_state()
            if str(workspace.get("id", "")).strip()
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
            WorkflowDefinition(
                key="supercell_editor",
                title="Supercell Editor",
                subtitle="Reusable cells",
                route="supercell_editor",
                completion_goal="Save reusable supercells for workflow setup.",
                completion_check=self._workflow_supercell_editor_complete,
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
                key="uep",
                title="UEP",
                subtitle="Electrostatic screening",
                route="uep",
                completion_goal="Complete at least one UEP source job.",
                completion_check=self._workflow_uep_complete,
            ),
            WorkflowDefinition(
                key="muon_sites",
                title="Muon Sites",
                subtitle="Place and refine muons",
                completion_goal="Generate at least one muon site.",
                completion_check=self._workflow_muon_candidates_complete,
                children=(
                    WorkflowDefinition(
                        key="muon_candidates",
                        title="Generation",
                        route="muon_candidates",
                        completion_goal="Generate at least one muon site.",
                        completion_check=self._workflow_muon_candidates_complete,
                    ),
                    WorkflowDefinition(
                        key="muon_relaxations",
                        title="Relaxations",
                        route="muon_relaxations",
                        completion_goal="All structural relaxations completed.",
                        completion_check=self._workflow_muon_relaxations_complete,
                    ),
                    WorkflowDefinition(
                        key="muon_analysis",
                        title="Analysis",
                        route="muon_analysis",
                        completion_goal="Create at least one structural-relaxation analysis workspace.",
                        completion_check=self._workflow_relaxation_analysis_complete,
                    ),
                    WorkflowDefinition(
                        key="muon_confirmed",
                        title="Confirmed Sites",
                        route="muon_confirmed",
                    ),
                ),
            ),
            WorkflowDefinition(
                key="magnetic_fields",
                title="Magnetic Fields",
                subtitle="Build magnetic model",
                route="magnetic_fields",
                completion_goal="Define at least one magnetic component.",
                completion_check=self._workflow_magnetic_fields_complete,
                children=magnetic_workspaces,
            ),
            WorkflowDefinition(
                key="quantum_corrections",
                title="Quantum Corrections",
                route="quantum_corrections",
                completion_goal="Complete at least one quantum phonon analysis.",
                completion_check=self._workflow_quantum_corrections_complete,
            ),
            WorkflowDefinition(
                key="energy_barriers",
                title="Energy Barriers",
                route="energy_barriers",
                completion_goal="Complete at least one transition-state search.",
                completion_check=self._workflow_energy_barriers_complete,
            ),
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
                child.setData(0, Qt.ItemDataRole.UserRole, self._comparison_route(table.id))
                pristine_item.addChild(child)
            pristine_item.setExpanded(True)

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
            self._main_stack.setCurrentWidget(self._home_page)
            self._sync_home_content_page()
        else:
            self._project_tree.setCurrentItem(selected_item)
        self._refresh_workflow_stage_icons()
        self._sync_project_home_trail()

    def _refresh_jobs_table(self) -> None:
        self._jobs_table.setRowCount(0)
        if self._document is None:
            self._job_runtime_timer.stop()
            self._cluster_elapsed_seconds_by_job = {}
            self._cluster_elapsed_checked_at = None
            self._convergence_view.set_cutoff_runtime_states({})
            self._kpoint_convergence_view.set_runtime_states({})
            self._pristine_relaxation_view.set_runtime_states({})
            self._muon_relaxation_view.set_runtime_states({})
            self._uep_view.set_runtime_states({})
            self._relaxation_analysis_view.set_context(
                workspaces=[],
                sources=[],
                structures={},
                host_structure=None,
                comparison_options=[],
            )
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

            can_resubmit = job.status in _RESUBMITTABLE_JOB_STATES
            can_cancel = job.status in _ACTIVE_LOCAL_JOB_STATES
            can_inspect = job.status in {JobStatus.FAILED, JobStatus.STOPPED}
            can_reset = job.status in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}

            resubmit_btn = self._make_job_action_button("Resubmit", job_id=job.id, enabled=can_resubmit)
            resubmit_btn.clicked.connect(
                lambda _checked=False, jid=job.id, btn=resubmit_btn: self._dispatch_dashboard_action(btn, jid, "resubmit")
            )
            self._jobs_table.setCellWidget(row, self._JOB_SUBMIT_COLUMN, resubmit_btn)

            cancel_btn = self._make_job_action_button("Cancel", job_id=job.id, enabled=can_cancel)
            cancel_btn.clicked.connect(
                lambda _checked=False, jid=job.id, btn=cancel_btn: self._dispatch_dashboard_action(btn, jid, "cancel")
            )
            self._jobs_table.setCellWidget(row, self._JOB_CANCEL_COLUMN, cancel_btn)

            inspect_btn = self._make_job_action_button("Inspect", job_id=job.id, enabled=can_inspect)
            inspect_btn.clicked.connect(
                lambda _checked=False, jid=job.id, btn=inspect_btn: self._dispatch_dashboard_action(btn, jid, "inspect")
            )
            self._jobs_table.setCellWidget(row, self._JOB_INSPECT_COLUMN, inspect_btn)

            reset_btn = self._make_job_action_button("Reset", job_id=job.id, enabled=can_reset)
            reset_btn.clicked.connect(
                lambda _checked=False, jid=job.id, btn=reset_btn: self._dispatch_dashboard_action(btn, jid, "reset")
            )
            self._jobs_table.setCellWidget(row, self._JOB_RESET_COLUMN, reset_btn)

        self._configure_jobs_table_columns()
        self._sync_job_runtime_timer()

        self._refresh_convergence_cutoff_rows()
        self._refresh_kpoint_convergence_rows()
        self._refresh_pristine_relaxation_rows()
        self._refresh_muon_relaxation_rows()
        self._refresh_uep_rows()
        self._refresh_workflow_stage_icons()
        self._sync_project_home_trail()

    def _refresh_active_job_runtime_cells(self) -> None:
        if self._document is None:
            self._job_runtime_timer.stop()
            return

        has_live_runtime = False
        for row in range(self._jobs_table.rowCount()):
            label_item = self._jobs_table.item(row, self._JOB_LABEL_COLUMN)
            runtime_item = self._jobs_table.item(row, self._JOB_RUNTIME_COLUMN)
            if label_item is None or runtime_item is None:
                continue
            raw_job_id = label_item.data(Qt.ItemDataRole.UserRole)
            try:
                job_id = UUID(str(raw_job_id))
            except (TypeError, ValueError, AttributeError):
                continue
            job = self._document.jobs.get(job_id)
            if job is None:
                continue
            if (
                job.status in {JobStatus.RUNNING, JobStatus.RETRIEVING}
                and (
                    job.started_at is not None
                    or (job.execution_target != "local" and job.id in self._cluster_elapsed_seconds_by_job)
                )
            ):
                has_live_runtime = True
            runtime_item.setText(self._job_runtime_text(job))

        if not has_live_runtime:
            self._job_runtime_timer.stop()

    def _sync_job_runtime_timer(self) -> None:
        if self._document is None:
            self._job_runtime_timer.stop()
            return
        has_live_runtime = any(
            job.status in {JobStatus.RUNNING, JobStatus.RETRIEVING}
            and (
                job.started_at is not None
                or (job.execution_target != "local" and job.id in self._cluster_elapsed_seconds_by_job)
            )
            for job in self._document.jobs.values()
            if job.status in _VISIBLE_DASHBOARD_STATES
        )
        # No point ticking once a second to update a table the user cannot see;
        # _on_jobs_dock_visibility_changed re-arms it when the dock returns.
        if has_live_runtime and self._jobs_dock_visible:
            if not self._job_runtime_timer.isActive():
                self._job_runtime_timer.start()
        else:
            self._job_runtime_timer.stop()

    def _on_jobs_dock_visibility_changed(self, visible: bool) -> None:
        self._jobs_dock_visible = bool(visible)
        # On becoming visible, bring the (possibly stale) runtime cells current
        # before re-arming. _sync_job_runtime_timer stops the timer when hidden.
        if self._jobs_dock_visible:
            self._refresh_active_job_runtime_cells()
        self._sync_job_runtime_timer()

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
    def _make_job_action_button(text: str, *, job_id: UUID, enabled: bool) -> SelectionSnapshotPushButton:
        button = SelectionSnapshotPushButton(text)
        button.selection_snapshot_key = job_id
        button.setEnabled(enabled)
        return button

    @staticmethod
    def _status_brush(status_value: str):
        return status_brush(status_value)

    @staticmethod
    def _format_runtime_seconds(elapsed_seconds: int) -> str:
        elapsed_seconds = max(0, int(elapsed_seconds))
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _parse_scheduler_elapsed_seconds(value: str) -> int | None:
        text = str(value).strip()
        if not text:
            return None
        day_count = 0
        if "-" in text:
            day_text, remainder = text.split("-", 1)
            try:
                day_count = int(day_text)
            except ValueError:
                return None
            text = remainder
        parts = text.split(":")
        try:
            numbers = [int(part) for part in parts]
        except ValueError:
            return None
        if len(numbers) == 2:
            hours = 0
            minutes, seconds = numbers
        elif len(numbers) == 3:
            hours, minutes, seconds = numbers
        else:
            return None
        return day_count * 86400 + hours * 3600 + minutes * 60 + seconds

    def _job_runtime_text(self, job: Job) -> str:
        if (
            job.execution_target != "local"
            and job.status in {JobStatus.RUNNING, JobStatus.RETRIEVING}
            and (base_elapsed := self._cluster_elapsed_seconds_by_job.get(job.id)) is not None
        ):
            checked_at = self._cluster_elapsed_checked_at
            if checked_at is None:
                return self._format_runtime_seconds(base_elapsed)
            delta_seconds = max(0, int((utc_now() - checked_at).total_seconds()))
            return self._format_runtime_seconds(base_elapsed + delta_seconds)
        if job.started_at is None:
            return ""
        end_time = job.completed_at or utc_now()
        elapsed_seconds = max(0, int((end_time - job.started_at).total_seconds()))
        return self._format_runtime_seconds(elapsed_seconds)

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

    def _request_dashboard_flush(self, *, failure_message: str) -> None:
        """Schedule a coalesced dashboard refresh + autosave for lifecycle handlers.

        Many cluster jobs can finish in a single queue poll; rebuilding every view
        and saving the whole document per job freezes the GUI. Instead, mark a
        flush pending and let it run once after the burst (via ``_flush_dashboard``,
        called from the queue-check handler and a short debounce timer).
        """

        if self._document is None:
            return
        self._dashboard_flush_pending = True
        self._dashboard_autosave_failure_message = failure_message
        self._dashboard_flush_timer.start()

    def _flush_dashboard(self) -> None:
        """Rebuild the dashboard views and persist the document, if a flush is due."""

        self._dashboard_flush_timer.stop()
        if not self._dashboard_flush_pending:
            return
        self._dashboard_flush_pending = False
        if self._document is None:
            return
        # _refresh_jobs_table already refreshes the convergence/kpoint/pristine/
        # muon-relaxation/uep rows, so only the views it does not cover are added.
        self._refresh_jobs_table()
        self._refresh_journal_view()
        self._refresh_host_structure_view()
        self._refresh_muon_generation_table()
        self._refresh_quantum_corrections_view()
        self._refresh_energy_barriers_view()
        self._persist_document_ui_state(failure_message=self._dashboard_autosave_failure_message)

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
        stress_threshold: float | None = None,
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

        snapshot: dict[str, object] = {
            "host_structure_id": host_structure_id,
            "simulation_cell_structure_id": simulation_cell_structure_id,
            "default_value": float(cutoff_energy),
            "method": method,
            "energy_threshold": float(energy_threshold),
            "force_threshold": float(force_threshold),
            "plot_data": dict(plot_data),
        }
        if stress_threshold is not None:
            snapshot["stress_threshold"] = float(stress_threshold)
        return snapshot

    def _build_kpoint_overview_snapshot(
        self,
        *,
        grid: tuple[int, int, int],
        method: str,
        energy_threshold: float,
        force_threshold: float,
        simulation_cell_structure_id: str | None,
        labels: Mapping[float, str],
        stress_threshold: float | None = None,
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
        snapshot: dict[str, object] = {
            "host_structure_id": host_structure_id,
            "simulation_cell_structure_id": simulation_cell_structure_id,
            "default_value": [int(grid[0]), int(grid[1]), int(grid[2])],
            "method": method,
            "energy_threshold": float(energy_threshold),
            "force_threshold": float(force_threshold),
            "plot_data": dict(plot_data),
            "point_labels": serialized_labels,
        }
        if stress_threshold is not None:
            snapshot["stress_threshold"] = float(stress_threshold)
        return snapshot

    def _on_tree_selection_changed(self) -> None:
        selected_items = self._project_tree.selectedItems()
        if not selected_items:
            return

        route = selected_items[0].data(0, Qt.ItemDataRole.UserRole)
        target = self._home_page
        if route == "host_structure":
            target = self._host_structure_view
        elif route == "supercell_editor":
            target = self._supercell_editor_page
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
            target = self._relaxation_analysis_view
        elif route == "magnetic_fields":
            target = self._magnetic_field_view
        elif isinstance(route, str) and self._magnetic_field_workspace_id_from_route(route) is not None:
            workspace_id = self._magnetic_field_workspace_id_from_route(route)
            if workspace_id is not None:
                target = self._magnetic_field_workspace_widgets.get(workspace_id, self._magnetic_field_view)
        elif route == "quantum_corrections":
            self._refresh_quantum_corrections_view()
            target = self._quantum_corrections_view
        elif route == "energy_barriers":
            target = self._ensure_energy_barriers_view()
            self._refresh_energy_barriers_view()
        elif route == "provenance_log":
            target = self._journal_view
        elif isinstance(route, str):
            table_id = self._comparison_table_id_from_route(route)
            if table_id is not None:
                target = self._comparison_widgets.get(table_id, self._pristine_relaxation_view)
        self._main_stack.setCurrentWidget(target)
        # Navigation does not mutate the document, so workflow completion state
        # cannot change here. Stage icons are kept current by tree construction
        # and by the mutation-driven refresh calls, so recomputing every check
        # on each tab switch (which scans jobs/results) is wasted work.
        self._sync_project_home_trail()

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
            widget.title_changed.connect(self._on_comparison_table_title_changed)
            self._main_stack.addWidget(widget)
            self._comparison_widgets[table.id] = widget

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
                magnetic_structure=magnetic_structure if isinstance(magnetic_structure, Mapping) else None,
                title=title,
                title_editable=True,
                deletable=True,
            )
            widget.set_form_state(form_state if isinstance(form_state, Mapping) else {})
            widget.magnetic_structure_changed.connect(
                lambda payload, workspace_id=workspace_id: self._on_magnetic_field_workspace_structure_changed(
                    workspace_id,
                    payload,
                )
            )
            widget.form_state_changed.connect(
                lambda state, workspace_id=workspace_id: self._on_magnetic_field_workspace_form_state_changed(
                    workspace_id,
                    state,
                )
            )
            widget.workspace_title_changed.connect(
                lambda title, workspace_id=workspace_id: self._on_magnetic_field_workspace_title_changed(
                    workspace_id,
                    title,
                )
            )
            widget.pin_requested.connect(self._on_magnetic_field_pin_requested)
            widget.delete_requested.connect(
                lambda workspace_id=workspace_id: self._delete_magnetic_field_workspace(workspace_id)
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
                magnetic_structure=magnetic_structure if isinstance(magnetic_structure, Mapping) else None,
                title=title,
                title_editable=True,
                deletable=True,
            )
            widget.set_form_state(form_state if isinstance(form_state, Mapping) else {})

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

    def _on_comparison_table_title_changed(self, payload: object, title: str) -> None:
        if self._document is None or self._shutting_down:
            return
        if isinstance(payload, UUID):
            table_id = payload
        elif isinstance(payload, str):
            try:
                table_id = UUID(payload)
            except ValueError:
                return
        else:
            return
        normalized = str(title).strip() or "Comparison Table"
        table = self._project_service.rename_comparison_table(self._document, table_id, normalized)
        if table is None:
            return
        widget = self._comparison_widgets.get(table_id)
        if widget is not None:
            widget.set_table_label(table.label)
        self._populate_project_tree()
        self._set_active_route(self._comparison_route(table_id))
        try:
            self._project_service.save_project(self._document)
        except ValueError:
            self.statusBar().showMessage("Comparison table title was updated, but autosave failed.")
            return
        self.statusBar().showMessage(f"Renamed comparison table to '{table.label}'")

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
        item = self._find_tree_item_by_route(route)
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

    def _show_project_home(self) -> None:
        self._project_tree.clearSelection()
        self._main_stack.setCurrentWidget(self._home_page)
        self._sync_home_content_page()
        self._refresh_workflow_stage_icons()
        self._sync_project_home_trail()

    def _sync_home_content_page(self) -> None:
        if self._home_content_stack is None:
            return
        target = self._project_home_page if self._document is not None else self._home_welcome_page
        if target is not None:
            self._home_content_stack.setCurrentWidget(target)
        if self._project_home_button is not None:
            self._project_home_button.setEnabled(self._document is not None)

    def _project_home_destinations(self) -> list[ProjectTrailDestination]:
        if self._document is None:
            return []

        return [
            ProjectTrailDestination(
                title="Host Structure",
                route="host_structure",
                theme="house",
                completed=self._workflow_host_complete(self._document),
                completion_goal="Load a host structure.",
            ),
            ProjectTrailDestination(
                title="Supercell Generation",
                route="supercell_editor",
                theme="castle",
                completed=self._workflow_supercell_editor_complete(self._document),
                completion_goal="Save reusable supercells for workflow setup.",
            ),
            ProjectTrailDestination(
                title="Cutoff Convergence",
                route="convergence_cutoff",
                theme="ocean",
                completed=self._workflow_cutoff_complete(self._document),
                completion_goal="Set a project default cutoff value.",
            ),
            ProjectTrailDestination(
                title="K-Point Grid",
                route="convergence_kpoint",
                theme="cube",
                completed=self._workflow_kpoint_complete(self._document),
                completion_goal="Set a project default k-point grid.",
            ),
            ProjectTrailDestination(
                title="Pristine Relaxation",
                route="pristine_relaxation",
                theme="hammock",
                completed=self._workflow_pristine_relaxation_complete(self._document),
                completion_goal="Generate at least one comparison table.",
            ),
            ProjectTrailDestination(
                title="UEP",
                route="uep",
                theme="storm",
                completed=self._workflow_uep_complete(self._document),
                completion_goal="Complete at least one UEP source job.",
            ),
            ProjectTrailDestination(
                title="Muon Sites",
                route="muon_candidates",
                theme="muon",
                completed=self._workflow_muon_candidates_complete(self._document),
                completion_goal="Generate at least one muon site.",
            ),
            ProjectTrailDestination(
                title="Magnetic Fields",
                route="magnetic_fields",
                theme="magnet",
                completed=self._workflow_magnetic_fields_complete(self._document),
                completion_goal="Define at least one magnetic component.",
            ),
            ProjectTrailDestination(
                title="Quantum Corrections",
                route="quantum_corrections",
                theme="cat",
                completed=self._workflow_quantum_corrections_complete(self._document),
                completion_goal="Complete at least one quantum phonon analysis.",
            ),
            ProjectTrailDestination(
                title="Energy Barriers",
                route="energy_barriers",
                theme="mountains",
                completed=self._workflow_energy_barriers_complete(self._document),
                completion_goal="Complete at least one transition-state search.",
            ),
            ProjectTrailDestination(
                title="Journal",
                route="provenance_log",
                theme="journal",
                placement="journal",
            ),
        ]

    def _sync_project_home_trail(self) -> None:
        if self._project_home_trail is None:
            return
        self._project_home_trail.set_destinations(self._project_home_destinations())

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

    def _on_resubmit_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return

        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        selected = [job for job in jobs if job is not None and job.status in _RESUBMITTABLE_JOB_STATES]
        if not selected:
            QMessageBox.information(
                self,
                "No resubmittable jobs",
                "Only failed or cancelled jobs can be resubmitted from the job dashboard.",
            )
            return

        local_jobs = [job for job in selected if job.execution_target == "local"]
        cluster_jobs = [job for job in selected if job.execution_target != "local"]

        if local_jobs:
            self._submit_jobs_locally(
                local_jobs,
                eligible_statuses=_RESUBMITTABLE_JOB_STATES,
                no_eligible_message=(
                    "Selected jobs must be in the 'failed' or 'cancelled' state for local resubmission."
                ),
                success_message=lambda count: f"Resubmitted {count} local job(s)",
            )

        if cluster_jobs:
            preferred = None
            targets = {job.execution_target for job in cluster_jobs}
            if len(targets) == 1:
                preferred = next(iter(targets))
            self._submit_jobs_on_cluster(
                cluster_jobs,
                default_cluster_name=preferred,
                eligible_statuses=_RESUBMITTABLE_JOB_STATES,
                no_eligible_message=(
                    "Selected jobs must be in the 'failed' or 'cancelled' state for cluster resubmission."
                ),
                reuse_existing_remote_files=True,
                success_message=lambda count, cluster: f"Resubmitted {count} cluster job(s) on {cluster}",
            )

    def _dispatch_dashboard_action(
        self, button: SelectionSnapshotPushButton, job_id: UUID, kind: str
    ) -> None:
        if self._document is None:
            return

        snapshot_keys: list[UUID] = []
        for row in button.selection_snapshot_rows:
            label_item = self._jobs_table.item(row, self._JOB_LABEL_COLUMN)
            if label_item is None:
                continue
            try:
                sid = UUID(str(label_item.data(Qt.ItemDataRole.UserRole)))
            except (TypeError, ValueError, AttributeError):
                continue
            snapshot_keys.append(sid)

        eligible_statuses = {
            "resubmit": _RESUBMITTABLE_JOB_STATES,
            "cancel": _ACTIVE_LOCAL_JOB_STATES,
            "inspect": {JobStatus.FAILED, JobStatus.STOPPED},
            "reset": {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED},
        }[kind]

        def eligibility(uid: UUID) -> bool:
            job = self._document.jobs.get(uid)  # type: ignore[union-attr]
            return job is not None and job.status in eligible_statuses

        titles = {
            "resubmit": "Resubmit jobs",
            "cancel": "Cancel jobs",
            "inspect": "Inspect jobs",
            "reset": "Reset jobs",
        }

        target_ids = resolve_target_keys(
            button,
            clicked_key=job_id,
            snapshot_keys=snapshot_keys,
            eligibility=eligibility,
            parent=self,
            title=titles[kind],
            message=None,
        )

        handlers = {
            "resubmit": self._on_resubmit_from_dashboard,
            "cancel": self._on_cancel_from_dashboard,
            "inspect": self._on_inspect_failed_from_dashboard,
            "reset": self._on_reset_failed_from_dashboard,
        }
        handlers[kind](list(target_ids))

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

    def _on_dashboard_delete_key_requested(self) -> None:
        jobs = self._selected_dashboard_jobs()
        if not jobs:
            return

        active_jobs = [job for job in jobs if job.status in _ACTIVE_LOCAL_JOB_STATES]
        resettable_jobs = [
            job for job in jobs if job.status in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
        ]
        deletable_jobs = [
            job
            for job in jobs
            if job.status not in _ACTIVE_LOCAL_JOB_STATES
            and job.status not in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
        ]

        if active_jobs:
            self._cancel_jobs(active_jobs)
        if resettable_jobs:
            self._reset_failed_jobs(resettable_jobs)
        if deletable_jobs:
            self._delete_dashboard_jobs(deletable_jobs)

    def _selected_dashboard_jobs(self) -> list[Job]:
        if self._document is None:
            return []

        selection_model = self._jobs_table.selectionModel()
        if selection_model is None:
            return []

        selected_rows = selection_model.selectedRows()
        if not selected_rows and self._jobs_table.currentRow() >= 0:
            selected_rows = [self._jobs_table.model().index(self._jobs_table.currentRow(), 0)]

        selected: list[Job] = []
        seen: set[UUID] = set()
        for index in selected_rows:
            label_item = self._jobs_table.item(index.row(), self._JOB_LABEL_COLUMN)
            if label_item is None:
                continue
            try:
                job_id = UUID(str(label_item.data(Qt.ItemDataRole.UserRole)))
            except (TypeError, ValueError, AttributeError):
                continue
            job = self._document.jobs.get(job_id)
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    def _delete_dashboard_jobs(self, jobs: list[Job]) -> None:
        if self._document is None:
            return

        deletable_jobs = [
            job
            for job in self._dedupe_jobs(jobs)
            if job.status not in _ACTIVE_LOCAL_JOB_STATES
            and job.status not in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
        ]
        if not deletable_jobs:
            return

        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("Delete jobs")
        message.setText(f"Delete {len(deletable_jobs)} selected job(s)?")
        message.setInformativeText("This removes the selected job records and any linked run/result records.")
        message.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        message.setDefaultButton(QMessageBox.StandardButton.No)
        if message.exec() != QMessageBox.StandardButton.Yes:
            return

        for job in deletable_jobs:
            self._delete_job_records(job.id)

        self._refresh_jobs_table()
        self._refresh_convergence_cutoff_rows()
        self._refresh_kpoint_convergence_rows()
        self.statusBar().showMessage(f"Deleted {len(deletable_jobs)} job(s)")

    def _jobs_for_convergence_cutoffs(self, cutoff_values: list[float]) -> list[Job]:
        if self._document is None:
            return []

        states = self._build_convergence_cutoff_states()
        selected: list[Job] = []
        seen: set[UUID] = set()
        for cutoff in cutoff_values:
            state = states.get(float(cutoff))
            if state is None:
                continue
            job = self._document.jobs.get(state.job_id) if state.job_id is not None else None
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

    @staticmethod
    def _configuration_mapping(definition) -> dict[str, object]:
        if definition is None:
            return {}
        config = definition.parameters.setdefault("configuration", {})
        if isinstance(config, dict):
            return config
        definition.parameters["configuration"] = {}
        return definition.parameters["configuration"]

    def _workflow_row_configuration(self, definition, key: str) -> tuple[dict[str, object], tuple[str, ...], dict[str, object]]:
        if definition is None:
            return {}, (), {}
        config = definition.parameters.get("configuration", {})
        if not isinstance(config, dict):
            return {}, (), {}
        raw_rows = config.get("row_settings", {})
        if not isinstance(raw_rows, dict):
            return {}, (), {}
        raw_entry = raw_rows.get(key, {})
        if not isinstance(raw_entry, dict):
            return {}, (), {}

        raw_params = raw_entry.get("parameter_values", raw_entry.get("params", {}))
        params = raw_params if isinstance(raw_params, dict) else {}
        restored_params = {
            str(name): self._restore_castep_param_value(str(name), raw_value)
            for name, raw_value in params.items()
        }
        raw_advanced = raw_entry.get("added_advanced", [])
        added_advanced = (
            tuple(str(name) for name in raw_advanced if isinstance(name, str))
            if isinstance(raw_advanced, list)
            else ()
        )
        extras = {
            str(name): value
            for name, value in raw_entry.items()
            if name not in {"parameter_values", "params", "added_advanced"}
        }
        return restored_params, added_advanced, extras

    def _project_convergence_parameter_values(
        self,
        *,
        target_lattice: list[list[float]] | None = None,
    ) -> dict[str, object]:
        if self._document is None:
            return {}
        values: dict[str, object] = {}
        cutoff = self._document.project.recommended_cutoff_energy
        if cutoff is not None:
            values["cut_off_energy"] = ValueWithUnit(value=float(cutoff), unit="eV")
        grid = self._document.project.recommended_kpoint_grid
        if grid is not None:
            resolved_grid = tuple(int(component) for component in grid)
            reference_lattice = self._document.project.recommended_kpoint_lattice
            if (
                target_lattice is not None
                and isinstance(reference_lattice, list)
                and len(reference_lattice) == 3
            ):
                resolved_grid = equivalent_grid_for_supercell(reference_lattice, resolved_grid, target_lattice)
            values["kpoint_mp_grid"] = resolved_grid
        return values

    def _workflow_row_configuration_with_project_defaults(
        self,
        definition,
        key: str,
        *,
        target_lattice: list[list[float]] | None = None,
    ) -> tuple[dict[str, object], tuple[str, ...], dict[str, object]]:
        params, advanced, extras = self._workflow_row_configuration(definition, key)
        values = self._project_convergence_parameter_values(target_lattice=target_lattice)
        values.update(params)
        return values, advanced, extras

    def _project_convergence_input_defaults(
        self,
        *,
        target_lattice: list[list[float]] | None = None,
    ) -> InputTemplate | None:
        values = self._project_convergence_parameter_values(target_lattice=target_lattice)
        if not values:
            return None
        return _input_template_from_keyword_values(values)

    @staticmethod
    def _parse_param_scalar_value(raw_value: str, spec: KeywordSpec | None) -> object:
        if spec is not None and spec.value_type == "text":
            return raw_value.strip()

        tokens = shlex.split(raw_value)
        if not tokens:
            return None

        if spec is not None and spec.value_type == "bool":
            token = tokens[0].strip().lower()
            if token in {"true", "t", ".true.", "yes", "1"}:
                return True
            if token in {"false", "f", ".false.", "no", "0"}:
                return False
            raise ValueError(f"Invalid boolean value '{tokens[0]}' for {spec.name.upper()}.")

        if spec is not None and spec.value_type == "int_triplet":
            if len(tokens) < 3:
                raise ValueError(f"{spec.name.upper()} requires three integer values.")
            return (int(tokens[0]), int(tokens[1]), int(tokens[2]))

        if spec is not None and spec.value_type in {"physical", "positive_float_with_optional_unit"}:
            if len(tokens) == 1:
                return float(tokens[0])
            if len(tokens) >= 2:
                return ValueWithUnit(value=float(tokens[0]), unit=tokens[1])

        if spec is not None and spec.value_type == "int":
            return int(tokens[0])

        if spec is not None and spec.value_type in {"float", "number"}:
            return float(tokens[0])

        if len(tokens) == 1:
            token = tokens[0]
            lowered = token.strip().lower()
            if lowered in {"true", "t", ".true.", "yes"}:
                return True
            if lowered in {"false", "f", ".false.", "no"}:
                return False
            try:
                return int(token)
            except ValueError:
                pass
            try:
                return float(token)
            except ValueError:
                return token

        values: list[object] = []
        for token in tokens:
            try:
                values.append(int(token))
                continue
            except ValueError:
                pass
            try:
                values.append(float(token))
                continue
            except ValueError:
                values.append(token)
        return tuple(values)

    @staticmethod
    def _parse_param_scalar_assignments(text: str) -> dict[str, object]:
        assignments: dict[str, object] = {}
        in_block = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for marker in ("!", "#"):
                if marker in line:
                    line = line.split(marker, maxsplit=1)[0].strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("%BLOCK "):
                in_block = True
                continue
            if upper.startswith("%ENDBLOCK "):
                in_block = False
                continue
            if in_block:
                continue
            if ":" in line:
                raw_name, raw_value = line.split(":", maxsplit=1)
            else:
                parts = line.split(maxsplit=1)
                if not parts:
                    continue
                raw_name = parts[0]
                raw_value = parts[1] if len(parts) > 1 else ""
            canonical_name = canonical_keyword_name("param", raw_name.strip())
            spec = get_keyword_spec("param", canonical_name)
            if not raw_value.strip():
                assignments[canonical_name] = None
                continue
            assignments[canonical_name] = MainWindow._parse_param_scalar_value(raw_value.strip(), spec)
        return assignments

    def _quantum_source_default_parameter_values(self, candidate: _QuantumSourceCandidate) -> dict[str, object]:
        values: dict[str, object] = {}
        output_dir = self._job_output_directory(candidate.job)
        if output_dir is None:
            return values

        seed_name = self._job_seed_name(candidate.job)
        cell_path = output_dir / f"{seed_name}.cell"
        if cell_path.is_file():
            try:
                parsed_cell = parse_cell_file(cell_path)
            except (CellParseError, OSError, ValueError):
                parsed_cell = None
            if parsed_cell is not None and parsed_cell.cell_file.kpoint_mp_grid is not None:
                values["kpoint_mp_grid"] = tuple(int(component) for component in parsed_cell.cell_file.kpoint_mp_grid)

        param_path = output_dir / f"{seed_name}.param"
        if param_path.is_file():
            try:
                assignments = self._parse_param_scalar_assignments(param_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                assignments = {}
            cutoff = assignments.get("cut_off_energy")
            if cutoff is not None:
                values["cut_off_energy"] = cutoff
        return values

    def _quantum_row_configuration_with_source_defaults(
        self,
        definition,
        key: str,
    ) -> tuple[dict[str, object], tuple[str, ...], dict[str, object]]:
        params, advanced, extras = self._workflow_row_configuration(definition, key)
        candidate = self._quantum_source_candidate_by_key(key)
        values = self._quantum_source_default_parameter_values(candidate) if candidate is not None else {}
        legacy_method = extras.get("phonon_method")
        if legacy_method is not None:
            values.setdefault("phonon_method", _quantum_method_to_gui_value(legacy_method))
        values.update(params)
        return values, advanced, extras

    @staticmethod
    def _quantum_num_farms_from_extras(extras: Mapping[str, object]) -> int:
        raw = extras.get("num_farms", 1)
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _quantum_phonon_method_from_extras(extras: Mapping[str, object]) -> str:
        raw = extras.get("phonon_method", "finite_displacement")
        normalized = str(raw).strip().lower()
        return normalized or "finite_displacement"

    def _workflow_row_configuration_exists(self, definition, key: str) -> bool:
        params, advanced, extras = self._workflow_row_configuration(definition, key)
        return bool(params) or bool(advanced) or any(str(value).strip() for value in extras.values())

    def _store_workflow_row_configuration(self, definition, key: str, payload: dict[str, object] | None) -> None:
        config = self._configuration_mapping(definition)
        rows = config.setdefault("row_settings", {})
        if not isinstance(rows, dict):
            rows = {}
            config["row_settings"] = rows
        if payload is None:
            rows.pop(key, None)
            if not rows:
                config.pop("row_settings", None)
            return
        rows[key] = payload

    def _quantum_parameter_specs(self) -> list[KeywordSpec]:
        specs = build_workflow_parameter_surface(
            workflow_stage="quantum_corrections",
            basic_names=_QUANTUM_CONFIG_KEYWORD_NAMES,
        )
        return [
            replace(spec, default_value=_QUANTUM_DEFAULT_ELEC_ENERGY_TOL)
            if spec.name == "elec_energy_tol"
            else spec
            for spec in specs
        ]

    def _energy_barrier_row_configuration_with_source_defaults(
        self,
        definition,
        key: str,
        *,
        target_lattice: list[list[float]] | None = None,
    ) -> tuple[dict[str, object], tuple[str, ...], dict[str, object]]:
        params, advanced, extras = self._workflow_row_configuration(definition, key)
        params = _strip_legacy_energy_barrier_params(params)
        values = self._project_convergence_parameter_values(target_lattice=target_lattice)
        candidate = self._energy_barrier_source_candidate_by_key(key)
        if candidate is not None:
            source_defaults = self._quantum_source_default_parameter_values(candidate.reactant)
            values.update(source_defaults)
        values.update(params)
        return values, advanced, extras

    def _energy_barrier_source_parameter_specs(self) -> list[KeywordSpec]:
        return build_workflow_parameter_surface(
            workflow_stage="energy_barriers",
            basic_names=_ENERGY_BARRIER_TSS_KEYWORD_NAMES,
        )

    def _energy_barrier_neb_parameter_specs(self) -> list[KeywordSpec]:
        return build_workflow_parameter_surface(
            workflow_stage="energy_barriers",
            basic_names=_ENERGY_BARRIER_NEB_KEYWORD_NAMES,
        )

    def _workflow_dialog_payload(
        self,
        dialog: ParameterDialog,
        *,
        extra_payload: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        collected = dict(dialog.collect_values())
        collected.pop("task", None)
        parameter_values = self._normalized_parameter_values(collected)
        payload = {
            "parameter_values": parameter_values if isinstance(parameter_values, dict) else {},
            "added_advanced": sorted(set(dialog.advanced_names())),
        }
        extras = {
            str(name): value
            for name, value in (extra_payload or {}).items()
        }
        payload.update(extras)
        if payload["parameter_values"] or payload["added_advanced"] or extras:
            return payload
        return None

    def _quantum_runtime_configuration(self, definition, key: str) -> tuple[dict[str, object], str]:
        params, _advanced, extras = self._quantum_row_configuration_with_source_defaults(definition, key)
        config = definition.parameters.get("configuration", {}) if definition is not None else {}
        legacy_method = "finite_displacement"
        legacy_pseudopotential = "unknown"
        if isinstance(config, dict):
            legacy_method = str(config.get("method", legacy_method)).strip() or legacy_method
            legacy_pseudopotential = (
                str(config.get("pseudopotential_family", legacy_pseudopotential)).strip() or legacy_pseudopotential
            )
        values = dict(params)
        values.setdefault("phonon_method", _quantum_method_to_gui_value(legacy_method))
        values.setdefault("elec_energy_tol", _QUANTUM_DEFAULT_ELEC_ENERGY_TOL)
        pseudopotential_family = str(extras.get("pseudopotential_family", legacy_pseudopotential)).strip() or "unknown"
        return values, pseudopotential_family

    def _energy_barrier_preview_structure(self, candidate: _EnergyBarrierSourceCandidate) -> Structure:
        if self._document is None:
            raise ValueError("No project document is loaded.")
        host_id = self._document.project.host_structure_id
        host = self._document.structures.get(host_id) if host_id is not None else None
        if host is None:
            raise ValueError("A pristine host structure is required for the path preview.")

        pristine_lattice = np.asarray(host.lattice_vectors, dtype=float)
        target_lattice = np.asarray(candidate.reactant.structure.lattice_vectors, dtype=float)
        if pristine_lattice.shape != (3, 3) or target_lattice.shape != (3, 3):
            raise ValueError("Energy barrier path preview requires 3x3 lattice vectors.")

        matrix = np.rint(target_lattice @ np.linalg.inv(pristine_lattice)).astype(int)
        if not np.allclose(matrix @ pristine_lattice, target_lattice, atol=1e-6):
            raise ValueError("Could not construct a pristine supercell matching the relaxed endpoint cell.")

        expanded_structure, _spins = build_supercell_structure(host, matrix.tolist())
        return replace(
            expanded_structure,
            label=f"{self._energy_barrier_source_label(candidate)} preview",
            origin="energy_barriers_path_preview",
            parent_structure_id=host.id,
        )

    def _ensure_energy_barriers_view(self) -> EnergyBarriersWidget:
        energy_barriers_view = self._energy_barriers_view
        if energy_barriers_view is not None:
            return energy_barriers_view

        energy_barriers_view = EnergyBarriersWidget()
        energy_barriers_view.endpoint_selection_changed.connect(
            self._on_energy_barrier_endpoint_selection_changed
        )
        energy_barriers_view.generate_paths_requested.connect(
            self._on_generate_energy_barrier_paths
        )
        energy_barriers_view.remove_sources_requested.connect(
            self._on_remove_energy_barrier_sources
        )
        energy_barriers_view.configure_source_requested.connect(self._on_configure_energy_barrier_source)
        energy_barriers_view.view_source_requested.connect(self._on_view_energy_barrier_source)
        energy_barriers_view.configure_result_requested.connect(self._on_configure_energy_barrier_result)
        energy_barriers_view.submit_neb_local_requested.connect(
            self._on_submit_local_neb_from_energy_barriers
        )
        energy_barriers_view.submit_neb_cluster_requested.connect(
            self._on_submit_cluster_neb_from_energy_barriers
        )
        energy_barriers_view.neb_result_selected.connect(
            self._on_energy_barrier_neb_result_selected
        )
        self._energy_barriers_view = energy_barriers_view
        self._main_stack.insertWidget(self._main_stack.indexOf(self._journal_view), energy_barriers_view)
        return energy_barriers_view

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

        self._sync_equivalent_convergence_cutoff_mappings(active_definition)
        config = self._deserialize_convergence_config(active_definition.parameters.get("configuration", {}))
        if config is not None and cutoff_value in config.cutoff_energies:
            return active_definition, config
        return None, None

    def _job_for_definition_cutoff(self, definition, cutoff: float) -> Job | None:
        if self._document is None:
            return None

        mapping = definition.parameters.get("job_ids_by_cutoff")
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
            return None

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

    def _convergence_cutoff_definition_matches(
        self,
        definition,
        *,
        input_structure_id: UUID,
        parameters: dict[str, object],
        output_directory: Path,
        simulation_cell_mode: str,
        supercell_source_workflow: str | None,
    ) -> bool:
        return (
            definition.input_structure_ids == [input_structure_id]
            and definition.parameters.get("configuration") == parameters
            and self._definition_output_directory(definition) == output_directory
            and str(definition.parameters.get("simulation_cell_mode", "unit_cell")) == simulation_cell_mode
            and definition.parameters.get("supercell_source_workflow") == supercell_source_workflow
        )

    def _convergence_cutoff_definition_context_matches(
        self,
        definition,
        *,
        input_structure_id: UUID,
        output_directory: Path,
        simulation_cell_mode: str,
        supercell_source_workflow: str | None,
    ) -> bool:
        return (
            definition.input_structure_ids == [input_structure_id]
            and self._definition_output_directory(definition) == output_directory
            and str(definition.parameters.get("simulation_cell_mode", "unit_cell")) == simulation_cell_mode
            and definition.parameters.get("supercell_source_workflow") == supercell_source_workflow
        )

    @staticmethod
    def _convergence_effective_params_for_cutoff(
        config: ConvergenceConfig,
        cutoff: float,
    ) -> dict[str, object] | None:
        cutoff_value = float(cutoff)
        if cutoff_value not in tuple(float(value) for value in config.cutoff_energies):
            return None

        params = dict(config.shared_params)
        for members, override in config.overrides.items():
            if cutoff_value in {float(value) for value in members}:
                params.update(override)
        return params

    def _convergence_cutoff_configs_match_for_value(
        self,
        target: ConvergenceConfig,
        source: ConvergenceConfig,
        cutoff: float,
    ) -> bool:
        if target.sweep_parameter != "cut_off_energy" or source.sweep_parameter != "cut_off_energy":
            return False
        if target.name_prefix != source.name_prefix:
            return False
        target_params = self._convergence_effective_params_for_cutoff(target, cutoff)
        source_params = self._convergence_effective_params_for_cutoff(source, cutoff)
        return target_params is not None and source_params is not None and target_params == source_params

    @staticmethod
    def _convergence_effective_params_for_grid(
        config: ConvergenceConfig,
        grid: tuple[int, int, int],
    ) -> dict[str, object] | None:
        normalized_grid = tuple(int(value) for value in grid)
        normalized_values = {
            tuple(int(component) for component in value)
            for value in config.sweep_values
            if isinstance(value, tuple) and len(value) == 3
        }
        if normalized_grid not in normalized_values:
            return None

        params = dict(config.shared_params)
        for members, override in config.overrides.items():
            normalized_members = {
                tuple(int(component) for component in value)
                for value in members
                if isinstance(value, tuple) and len(value) == 3
            }
            if normalized_grid in normalized_members:
                params.update(override)
        return params

    def _convergence_kpoint_configs_match_for_grid(
        self,
        target: ConvergenceConfig,
        source: ConvergenceConfig,
        grid: tuple[int, int, int],
    ) -> bool:
        if target.sweep_parameter != "kpoint_mp_grid" or source.sweep_parameter != "kpoint_mp_grid":
            return False
        if target.name_prefix != source.name_prefix:
            return False
        target_params = self._convergence_effective_params_for_grid(target, grid)
        source_params = self._convergence_effective_params_for_grid(source, grid)
        return target_params is not None and source_params is not None and target_params == source_params

    def _matching_convergence_cutoff_definition(
        self,
        *,
        input_structure_id: UUID,
        parameters: dict[str, object],
        output_directory: Path,
        simulation_cell_mode: str,
        supercell_source_workflow: str | None,
    ):
        if self._document is None:
            return None

        active = self._active_convergence_definition()
        definitions = self._convergence_definitions()
        ordered = [definition for definition in (active, *definitions) if definition is not None]
        seen: set[UUID] = set()
        for definition in ordered:
            if definition.id in seen:
                continue
            seen.add(definition.id)
            if self._convergence_cutoff_definition_matches(
                definition,
                input_structure_id=input_structure_id,
                parameters=parameters,
                output_directory=output_directory,
                simulation_cell_mode=simulation_cell_mode,
                supercell_source_workflow=supercell_source_workflow,
            ):
                return definition
        return None

    def _sync_equivalent_convergence_kpoint_mappings(self, target_definition) -> None:
        if self._document is None:
            return

        target_config = self._deserialize_convergence_config(target_definition.parameters.get("configuration", {}))
        if (
            target_config is None
            or target_config.sweep_parameter != "kpoint_mp_grid"
            or not target_definition.input_structure_ids
        ):
            return

        input_structure_id = target_definition.input_structure_ids[0]
        output_directory = self._definition_output_directory(target_definition)
        if output_directory is None:
            return

        simulation_cell_mode = str(target_definition.parameters.get("simulation_cell_mode", "unit_cell"))
        supercell_source_workflow = target_definition.parameters.get("supercell_source_workflow")

        candidates: dict[str, str] = {}
        for definition in self._kpoint_definitions():
            if not self._convergence_cutoff_definition_context_matches(
                definition,
                input_structure_id=input_structure_id,
                output_directory=output_directory,
                simulation_cell_mode=simulation_cell_mode,
                supercell_source_workflow=supercell_source_workflow,
            ):
                continue
            source_config = self._deserialize_convergence_config(definition.parameters.get("configuration", {}))
            if source_config is None or source_config.sweep_parameter != "kpoint_mp_grid":
                continue
            mapping = definition.parameters.get("job_ids_by_grid", {})
            if not isinstance(mapping, dict):
                continue
            for key, raw_job_id in mapping.items():
                if not isinstance(key, str) or not isinstance(raw_job_id, str):
                    continue
                grid = self._parse_grid_key(key)
                if grid is None:
                    continue
                if not self._convergence_kpoint_configs_match_for_grid(
                    target_config,
                    source_config,
                    grid,
                ):
                    continue
                normalized_key = self._grid_key(grid)
                existing = candidates.get(normalized_key)
                if (
                    existing is None
                    or self._convergence_mapping_job_priority(raw_job_id)
                    > self._convergence_mapping_job_priority(existing)
                ):
                    candidates[normalized_key] = raw_job_id

        if not candidates:
            return

        target_mapping = target_definition.parameters.setdefault("job_ids_by_grid", {})
        if not isinstance(target_mapping, dict):
            target_mapping = {}
            target_definition.parameters["job_ids_by_grid"] = target_mapping

        for key, raw_job_id in candidates.items():
            existing = target_mapping.get(key)
            if (
                not isinstance(existing, str)
                or self._convergence_mapping_job_priority(raw_job_id)
                > self._convergence_mapping_job_priority(existing)
            ):
                target_mapping[key] = raw_job_id

    def _sync_equivalent_convergence_cutoff_mappings(self, target_definition) -> None:
        """Merge mappings from duplicate cutoff definitions into the active one."""
        if self._document is None:
            return

        target_config = self._deserialize_convergence_config(target_definition.parameters.get("configuration", {}))
        if target_config is None or not target_definition.input_structure_ids:
            return

        input_structure_id = target_definition.input_structure_ids[0]
        output_directory = self._definition_output_directory(target_definition)
        if output_directory is None:
            return

        simulation_cell_mode = str(target_definition.parameters.get("simulation_cell_mode", "unit_cell"))
        supercell_source_workflow = target_definition.parameters.get("supercell_source_workflow")
        parameters = target_definition.parameters.get("configuration")
        if not isinstance(parameters, dict):
            return

        candidates: dict[str, str] = {}
        for definition in self._convergence_definitions():
            if not self._convergence_cutoff_definition_context_matches(
                definition,
                input_structure_id=input_structure_id,
                output_directory=output_directory,
                simulation_cell_mode=simulation_cell_mode,
                supercell_source_workflow=supercell_source_workflow,
            ):
                continue
            source_config = self._deserialize_convergence_config(definition.parameters.get("configuration", {}))
            if source_config is None:
                continue
            mapping = definition.parameters.get("job_ids_by_cutoff", {})
            if not isinstance(mapping, dict):
                continue
            for key, raw_job_id in mapping.items():
                if not isinstance(raw_job_id, str):
                    continue
                try:
                    cutoff = float(key)
                except (TypeError, ValueError):
                    continue
                if not self._convergence_cutoff_configs_match_for_value(
                    target_config,
                    source_config,
                    cutoff,
                ):
                    continue
                normalized_key = self._cutoff_key(cutoff)
                existing = candidates.get(normalized_key)
                if (
                    existing is None
                    or self._convergence_mapping_job_priority(raw_job_id)
                    > self._convergence_mapping_job_priority(existing)
                ):
                    candidates[normalized_key] = raw_job_id

        if not candidates:
            return

        target_mapping = target_definition.parameters.setdefault("job_ids_by_cutoff", {})
        if not isinstance(target_mapping, dict):
            target_mapping = {}
            target_definition.parameters["job_ids_by_cutoff"] = target_mapping

        for key, raw_job_id in candidates.items():
            existing = target_mapping.get(key)
            if (
                not isinstance(existing, str)
                or self._convergence_mapping_job_priority(raw_job_id)
                > self._convergence_mapping_job_priority(existing)
            ):
                target_mapping[key] = raw_job_id

    def _convergence_mapping_job_priority(self, raw_job_id: str) -> int:
        if self._document is None:
            return 0
        try:
            job_id = UUID(str(raw_job_id))
        except ValueError:
            return 0
        job = self._document.jobs.get(job_id)
        if job is None:
            return 0
        if job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING}:
            return 50
        if job.status == JobStatus.COMPLETED:
            return 40
        if job.status in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}:
            return 30
        if job.status == JobStatus.DEFINED:
            return 10
        return 0

    def _relaxation_mapping_job_priority(self, raw_job_id: str) -> int:
        if self._document is None:
            return 0
        try:
            job_id = UUID(str(raw_job_id))
        except ValueError:
            return 0
        job = self._document.jobs.get(job_id)
        if job is None:
            return 0
        if job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING}:
            return 50
        if job.status == JobStatus.COMPLETED:
            return 40
        if job.status == JobStatus.DEFINED:
            return 35
        if job.status in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}:
            return 30
        return 0

    @staticmethod
    def _job_submission_sort_key(job: Job) -> float:
        for stamp in (job.submitted_at, job.started_at, job.completed_at):
            if stamp is not None:
                return stamp.timestamp()
        return 0.0

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
        self._submit_workflow_selection(
            cutoff_values,
            normalize=float,
            build_states=self._build_convergence_cutoff_states,
            jobs_for_values=self._jobs_for_convergence_cutoffs,
            materialize_values=self._materialize_convergence_jobs,
            is_ready=sweep_point_state_is_ready,
            no_ready_message="Only configured cutoffs in the 'ready' state can be submitted.",
            submit=self._submit_jobs_locally,
        )

    def _on_submit_cluster_from_convergence(self, cutoff_values: list[float]) -> None:
        self._submit_workflow_selection(
            cutoff_values,
            normalize=float,
            build_states=self._build_convergence_cutoff_states,
            jobs_for_values=self._jobs_for_convergence_cutoffs,
            materialize_values=self._materialize_convergence_jobs,
            is_ready=sweep_point_state_is_ready,
            no_ready_message="Only configured cutoffs in the 'ready' state can be submitted.",
            submit=self._submit_jobs_on_cluster,
        )

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

    def _on_cutoff_adopted(
        self,
        cutoff_energy: float,
        method: str,
        energy_threshold: float,
        force_threshold: float,
        stress_threshold: float | None = None,
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
            stress_threshold=stress_threshold,
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
        if stress_threshold is None:
            self._document.ui_state.pop("default_cut_off_stress_threshold", None)
        else:
            self._document.ui_state["default_cut_off_stress_threshold"] = float(stress_threshold)
        self._store_convergence_overview_snapshot(
            "cutoff",
            self._build_cutoff_overview_snapshot(
                cutoff_energy=float(cutoff_energy),
                method=method,
                energy_threshold=float(energy_threshold),
                force_threshold=float(force_threshold),
                simulation_cell_structure_id=definition_cell_id,
                stress_threshold=float(stress_threshold) if stress_threshold is not None else None,
            ),
        )
        self._convergence_view.set_project_defaults(
            cutoff_energy=float(cutoff_energy),
            cutoff_method=method,
            energy_threshold=float(energy_threshold),
            force_threshold=float(force_threshold),
            stress_threshold=float(stress_threshold) if stress_threshold is not None else None,
        )
        self._kpoint_convergence_view.set_cutoff_default(float(cutoff_energy))
        self._refresh_convergence_overview()
        self._refresh_journal_view()
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
            self._remove_cutoff_from_equivalent_definitions(definition, cutoff)

        return True, shared_file_disposition, False

    def _remove_cutoff_from_equivalent_definitions(self, source_definition, cutoff: float) -> None:
        if self._document is None:
            return

        if not source_definition.input_structure_ids:
            self._remove_cutoff_from_definition(source_definition, cutoff)
            return

        input_structure_id = source_definition.input_structure_ids[0]
        output_directory = self._definition_output_directory(source_definition)
        if output_directory is None:
            self._remove_cutoff_from_definition(source_definition, cutoff)
            return

        simulation_cell_mode = str(source_definition.parameters.get("simulation_cell_mode", "unit_cell"))
        supercell_source_workflow = source_definition.parameters.get("supercell_source_workflow")
        for definition in self._convergence_definitions():
            if not self._convergence_cutoff_definition_context_matches(
                definition,
                input_structure_id=input_structure_id,
                output_directory=output_directory,
                simulation_cell_mode=simulation_cell_mode,
                supercell_source_workflow=supercell_source_workflow,
            ):
                continue
            self._remove_cutoff_from_definition(definition, cutoff)

    def _remove_cutoff_from_definition(self, definition, cutoff: float) -> None:
        if self._document is None:
            return

        configuration = definition.parameters.get("configuration")
        if not isinstance(configuration, dict):
            return

        cutoff_to_remove = float(cutoff)
        remove_index: int | None = None
        cutoffs = configuration.get("cutoff_energies")
        if isinstance(cutoffs, list):
            updated_cutoffs: list[object] = []
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
            for index, raw_value in enumerate(raw_sweep_values):
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    updated_sweep_values.append(raw_value)
                    continue
                if value == cutoff_to_remove:
                    if remove_index is None:
                        remove_index = index
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

        remaining_cutoffs = configuration.get("cutoff_energies", configuration.get("sweep_values"))
        if not remaining_cutoffs:
            self._document.calculations.pop(definition.id, None)
            if self._document.ui_state.get(self._ACTIVE_CONVERGENCE_DEFINITION_KEY) == str(definition.id):
                self._document.ui_state.pop(self._ACTIVE_CONVERGENCE_DEFINITION_KEY, None)

    _FILESYSTEM_RETRY_DELAYS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)

    @staticmethod
    def _force_writable_and_retry(func, path, exc_info) -> None:
        # shutil.rmtree onexc handler: Windows refuses to delete read-only files
        # (CASTEP often marks output read-only) and raises PermissionError. Clear
        # the read-only attribute and retry the failed primitive once. Anything
        # else — including a second failure — propagates so the outer retry loop
        # or the UI sees it.
        exc = exc_info[1] if isinstance(exc_info, tuple) else exc_info
        if not isinstance(exc, PermissionError):
            raise exc
        try:
            os.chmod(path, stat.S_IWRITE)
        except OSError:
            raise exc
        func(path)

    @classmethod
    def _retry_filesystem_op(cls, op) -> None:
        # OneDrive sync, antivirus scanners, and the Windows search indexer hold
        # transient handles that produce PermissionError / [WinError 5]. Retry
        # with short backoff (~2 s total) so the common case clears silently;
        # re-raise the last error otherwise so the existing UI dialog fires.
        for delay in cls._FILESYSTEM_RETRY_DELAYS:
            try:
                op()
                return
            except FileNotFoundError:
                return
            except PermissionError:
                time.sleep(delay)
        try:
            op()
        except FileNotFoundError:
            pass

    @classmethod
    def _delete_path(cls, path: Path) -> None:
        if path.is_dir():
            cls._retry_filesystem_op(
                lambda: shutil.rmtree(path, onexc=cls._force_writable_and_retry)
            )
            return
        cls._retry_filesystem_op(path.unlink)

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
        self._retry_filesystem_op(lambda: source.rename(destination))
        return destination

    def _prompt_cutoff_file_disposition(
        self,
        cutoff: float,
        job_dir: Path,
        *,
        include_apply_all: bool = False,
    ) -> str | None:
        return self._prompt_job_file_disposition(
            title="Remove cutoff with files",
            text=f"Cutoff {cutoff:.8g} eV has associated files in {job_dir.name}.",
            keep_text="or Cancel to keep the cutoff entry.",
            include_apply_all=include_apply_all,
        )

    def _prompt_job_file_disposition(
        self,
        *,
        title: str,
        text: str,
        keep_text: str,
        include_apply_all: bool = False,
    ) -> str | None:
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle(title)
        message.setText(text)
        message.setInformativeText(
            "Choose Yes to delete those files, No to move them into a separate 'removed_from_workflow' "
            f"subfolder, {keep_text}"
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
        result_ids = {
            result.id
            for result in self._document.results.values()
            if result.calculation_run_id in run_ids
        }
        result_ids.update(
            run.result_id
            for run in self._document.runs.values()
            if run.job_id == job_id and run.result_id is not None
        )

        run_id_set = set(run_ids)
        for site in self._document.sites.values():
            if site.relaxation_run_id not in run_id_set:
                continue
            site.relaxation_run_id = None
            site.relative_energy_eV = None
            if site.status == "relaxed":
                site.status = "staged"

        for result_id in result_ids:
            self._document.results.pop(result_id, None)
        for run_id in run_ids:
            self._document.runs.pop(run_id, None)

        self._active_run_by_job.pop(job_id, None)
        self._detach_job_references(job_id)
        self._document.jobs.pop(job_id, None)

    def _detach_job_references(self, job_id: UUID) -> None:
        if self._document is None:
            return
        target = str(job_id)
        for definition in self._document.calculations.values():
            for mapping_name in (
                "job_ids_by_cutoff",
                "job_ids_by_grid",
                "job_ids_by_label",
                "singlepoint_job_ids_by_label",
                "phonon_job_ids_by_label",
            ):
                mapping = definition.parameters.get(mapping_name, {})
                if not isinstance(mapping, dict):
                    continue
                stale_keys = [key for key, value in mapping.items() if str(value) == target]
                for key in stale_keys:
                    mapping.pop(key, None)

    def _quantum_definition_and_key_for_job(self, job: Job) -> tuple[object, str] | tuple[None, None]:
        for definition in self._quantum_corrections_definitions():
            for mapping_name in ("singlepoint_job_ids_by_label", "phonon_job_ids_by_label", "job_ids_by_label"):
                mapping = self._quantum_job_mapping(definition, mapping_name)
                for key, value in mapping.items():
                    if str(value) == str(job.id):
                        return definition, str(key)
        return None, None

    def _quantum_task_farming_for_job(self, job: Job) -> tuple[bool, int | None]:
        if job.calc_type != QUANTUM_PHONON_CALC_TYPE:
            return False, None
        definition, key = self._quantum_definition_and_key_for_job(job)
        if definition is None or key is None:
            return False, None
        _params, _advanced, extras = self._workflow_row_configuration(definition, key)
        num_farms = self._quantum_num_farms_from_extras(extras)
        return num_farms > 1, (num_farms if num_farms > 1 else None)

    def _quantum_continuation_remote_target(self, job: Job, *, cluster_name: str) -> str:
        if job.calc_type != QUANTUM_PHONON_CALC_TYPE:
            return ""
        definition, key = self._quantum_definition_and_key_for_job(job)
        if definition is None or key is None:
            return ""
        source_job = self._singlepoint_job_for_quantum_source_key(definition, key)
        if source_job is None or source_job.status != JobStatus.COMPLETED:
            return ""
        if source_job.execution_target.strip().casefold() != cluster_name.strip().casefold():
            return ""
        return (source_job.remote_directory or "").strip()

    def _submit_jobs_locally(
        self,
        jobs: list[Job],
        *,
        eligible_statuses: set[JobStatus] | None = None,
        no_eligible_message: str = "Selected jobs must be in the 'defined' state for local submission.",
        success_message: Callable[[int], str] | None = None,
    ) -> None:
        if self._document is None or not jobs:
            return

        allowed_statuses = eligible_statuses or {JobStatus.DEFINED}
        eligible_jobs = [job for job in self._dedupe_jobs(jobs) if job.status in allowed_statuses]
        if not eligible_jobs:
            QMessageBox.information(
                self,
                "No eligible jobs",
                no_eligible_message,
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
            job_labels=[job.label for job in eligible_jobs],
            mpirun_configured=bool(setup.mpirun_path),
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        config = dialog.config()

        outcome = self._job_execution.submit_local_jobs(
            eligible_jobs,
            num_cores=config.num_cores,
            max_concurrent=config.max_concurrent if config.parallel else 1,
            castep_executable=setup.executable_path,
            mpirun_path=setup.mpirun_path,
            eligible_statuses=allowed_statuses,
            task_farming_for_job=self._quantum_task_farming_for_job,
        )

        self._refresh_jobs_table()
        self._refresh_convergence_cutoff_rows()
        self._refresh_kpoint_convergence_rows()
        self._refresh_quantum_corrections_view()
        if outcome.journal_changed:
            self._refresh_journal_view()

        if outcome.missing_paths:
            QMessageBox.warning(
                self,
                "Missing job directories",
                "Could not locate generated input directories for:\n"
                + "\n".join(f"- {label}" for label in outcome.missing_paths),
            )

        if outcome.failed_messages:
            show_detailed_warning(
                self,
                "Local submission failed",
                "Some jobs failed to submit:\n" + "\n".join(outcome.failed_messages),
                informative_text="Open Details to inspect the full submission failures.",
            )

        if outcome.submitted_count:
            self._persist_document_ui_state(
                failure_message="Submitted local jobs, but autosave failed.",
            )
            message = (
                success_message(outcome.submitted_count)
                if success_message is not None
                else f"Submitted {outcome.submitted_count} local job(s)"
            )
            self.statusBar().showMessage(message)

    def _submit_jobs_on_cluster(
        self,
        jobs: list[Job],
        *,
        default_cluster_name: str | None = None,
        eligible_statuses: set[JobStatus] | None = None,
        no_eligible_message: str = "Selected jobs must be in the 'defined' state for cluster submission.",
        reuse_existing_remote_files: bool = False,
        success_message: Callable[[int, str], str] | None = None,
    ) -> None:
        if self._document is None or not jobs:
            return

        allowed_statuses = eligible_statuses or {JobStatus.DEFINED}
        eligible_jobs = [job for job in self._dedupe_jobs(jobs) if job.status in allowed_statuses]
        if not eligible_jobs:
            QMessageBox.information(
                self,
                "No eligible jobs",
                no_eligible_message,
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

        workflow_type = self._cluster_workflow_directory_fragment(eligible_jobs)
        last_submission = self._load_last_cluster_submission_config(workflow_type=workflow_type)
        initial_cluster_name = (
            default_cluster_name
            or (last_submission.cluster_name if last_submission is not None else None)
        )

        dialog = ClusterSubmissionDialog(
            job_labels=[job.label for job in eligible_jobs],
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
        if hasattr(dialog, "credentials"):
            dialog_credentials = dialog.credentials()
        else:
            dialog_credentials = SSHCredentials(target_secret=dialog.password().strip() or None)
        selected_cluster = load_cluster_config(self._settings, submission.cluster_name)
        if selected_cluster is None:
            QMessageBox.warning(self, "Cluster not found", "Selected cluster is no longer configured.")
            return
        runtime_username = submission.username.strip()
        if runtime_username and runtime_username != selected_cluster.username:
            selected_cluster = replace(selected_cluster, username=runtime_username)

        seed_name_by_job = {job.id: self._job_seed_name(job) for job in eligible_jobs}
        invalid_labels = self._cluster_runner.invalid_remote_job_labels(list(seed_name_by_job.values()))
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

        credentials = SSHCredentials() if reusing_connection else dialog_credentials
        if credentials is None:
            QMessageBox.warning(self, "Missing credentials", "Credentials are required for cluster submission.")
            return

        try:
            self._connect_to_cluster_with_trust(selected_cluster, credentials=credentials)
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
        submitted_jobs: list[Job] = []
        failed_submit: list[str] = []
        skipped_submit: list[str] = []

        # 1. Compute remote targets for every job (no SSH; GUI thread).
        remote_target_by_job: dict[UUID, str] = {}
        reuse_existing_files_by_job: dict[UUID, bool] = {}
        allow_existing_remote_directory_by_job: dict[UUID, bool] = {}
        targets_to_check: list[tuple[UUID, str]] = []
        for job in eligible_jobs:
            allow_existing_remote_directory = False
            remote_target = (job.remote_directory or "").strip() if reuse_existing_remote_files else ""
            if not remote_target:
                continuation_target = self._quantum_continuation_remote_target(job, cluster_name=selected_cluster.name)
                if continuation_target:
                    remote_target = continuation_target
                    allow_existing_remote_directory = True
            reuse_existing_files_for_job = bool(remote_target)
            if allow_existing_remote_directory:
                reuse_existing_files_for_job = False
            if not remote_target:
                remote_target = self._cluster_runner.remote_job_directory(
                    working_directory=submission.working_directory,
                    seed_name=seed_name_by_job.get(job.id, job.label),
                )
            remote_target_by_job[job.id] = remote_target
            reuse_existing_files_by_job[job.id] = reuse_existing_files_for_job
            allow_existing_remote_directory_by_job[job.id] = allow_existing_remote_directory
            if not reuse_existing_files_for_job and not allow_existing_remote_directory:
                targets_to_check.append((job.id, remote_target))

        # 2. Check existing remote directories on the worker thread.
        existing_remote_targets: list[str] = []
        if targets_to_check:
            check_result = self._await_remote_dir_check(targets_to_check)
            existing_remote_targets = list(check_result.get("existing", []))
            errors = check_result.get("errors", {})
            for job in eligible_jobs:
                error = errors.get(job.id)
                if error:
                    failed_submit.append(f"- {job.label}: Failed to check remote directory ({error})")

        # 3. Resolve overwrite decisions up front (GUI thread), so no modal prompt
        #    interrupts the submission loop running on the worker thread.
        decision_by_job: dict[UUID, str] = {}
        overwrite_mode: str | None = None
        for job in eligible_jobs:
            remote_target = remote_target_by_job.get(job.id)
            if not (remote_target and remote_target in existing_remote_targets):
                continue
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
                decision_by_job[job.id] = "delete"
            else:
                decision_by_job[job.id] = "skip"
                skipped_submit.append(
                    f"- {job.label}: Submission skipped because remote directory already exists ({remote_target})"
                )

        # 4. Prepare document state (GUI thread) and build the worker batch.
        prepared_by_job: dict[UUID, PreparedClusterSubmission] = {}
        submit_items: list[dict[str, object]] = []
        for job in eligible_jobs:
            if decision_by_job.get(job.id) == "skip":
                continue
            output_dir = self._job_output_directory(job)
            if output_dir is None:
                missing_paths.append(job.label)
                continue
            seed_name = seed_name_by_job.get(job.id, self._job_seed_name(job))
            remote_target = remote_target_by_job.get(job.id)
            task_farming_enabled, num_farms = self._quantum_task_farming_for_job(job)
            prepared = self._job_execution.prepare_cluster_submission(
                ClusterJobSubmission(
                    job=job,
                    output_dir=output_dir,
                    seed_name=seed_name,
                    remote_target=remote_target,
                    execution_target=selected_cluster.name,
                    num_processes=submission.num_processes,
                    walltime=submission.walltime,
                    working_directory=submission.working_directory,
                    template_script_path=selected_cluster.template_script_path,
                    cores_per_node=selected_cluster.cores_per_node,
                    memory_gb=getattr(submission, "memory_gb", None),
                    queue_name=getattr(submission, "queue_name", ""),
                    comment=getattr(submission, "comment", ""),
                    castep_executable=selected_cluster.castep_executable,
                    retrieve_den_fmt=job.calc_type == "castep_uep_source",
                    retrieve_out_cell=job.calc_type == "castep_pristine_relaxation"
                    or is_muon_relaxation_calc_type(job.calc_type),
                    retrieve_phonon=job.calc_type == QUANTUM_PHONON_CALC_TYPE,
                    retrieve_ts=job.calc_type in {TRANSITION_STATE_SEARCH_CALC_TYPE, NUDGED_ELASTIC_BAND_CALC_TYPE},
                    reuse_existing_files=reuse_existing_files_by_job.get(job.id, False),
                    allow_existing_remote_directory=allow_existing_remote_directory_by_job.get(job.id, False),
                    task_farming_enabled=task_farming_enabled,
                    num_farms=num_farms,
                )
            )
            if prepared is None:
                continue
            prepared_by_job[job.id] = prepared
            submit_items.append(
                {
                    "job_id": job.id,
                    "label": job.label,
                    "submit_kwargs": prepared.submit_kwargs,
                    "remote_dir_to_delete": (
                        remote_target if decision_by_job.get(job.id) == "delete" else None
                    ),
                }
            )

        # 5. Run the batch on the worker thread with a cancellable progress dialog.
        if submit_items:
            batch_submitted, batch_failed = self._run_cluster_submit_batch(submit_items, prepared_by_job)
            submitted = len(batch_submitted)
            submitted_jobs = batch_submitted
            failed_submit.extend(batch_failed)

        self._refresh_jobs_table()
        self._refresh_convergence_cutoff_rows()
        self._refresh_kpoint_convergence_rows()
        self._refresh_quantum_corrections_view()
        self._refresh_energy_barriers_view()

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

        if skipped_submit:
            QMessageBox.information(
                self,
                "Cluster submission skipped",
                "Some jobs were not submitted because their remote directories already exist:\n"
                + "\n".join(skipped_submit),
            )

        if submitted:
            if self._job_execution.record_submission(submitted_jobs, selected_cluster.name):
                self._refresh_journal_view()
            self._persist_document_ui_state(
                failure_message="Submitted cluster jobs, but autosave failed.",
            )
            message = (
                success_message(submitted, selected_cluster.name)
                if success_message is not None
                else f"Submitted {submitted} cluster job(s) on {selected_cluster.name}"
            )
            self.statusBar().showMessage(message)

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
        default_job_name_prefix = self._default_cluster_job_name_prefix(workflow_type)

        return ClusterSubmissionConfig(
            cluster_name=cluster_name,
            working_directory=str(payload.get("working_directory", "")).strip() if same_workflow else "",
            username=str(payload.get("username", "")).strip(),
            job_name_prefix=(
                str(payload.get("job_name_prefix", default_job_name_prefix)).strip()
                if same_workflow
                else default_job_name_prefix
            )
            or default_job_name_prefix,
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
        """

        captured: dict[str, object] = {}
        loop = QEventLoop()

        def on_signal(payload: object = None) -> None:
            captured["payload"] = payload
            loop.quit()

        progress = QProgressDialog(label, None, 0, 0, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        signal.connect(on_signal)
        progress.show()
        try:
            start()
            loop.exec()
        finally:
            signal.disconnect(on_signal)
            progress.close()
        return captured.get("payload")

    def _await_remote_dir_check(self, targets: list[tuple[UUID, str]]) -> dict[str, object]:
        payload = self._await_cluster_signal(
            start=lambda: self._cluster_runner.check_remote_dirs(targets),
            signal=self._cluster_runner.remote_dirs_checked,
            title="Checking remote directories",
            label="Checking existing remote job directories…",
        )
        return payload if isinstance(payload, dict) else {"existing": [], "errors": {}}

    def _await_cluster_connect(
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

        progress = QProgressDialog("Submitting cluster jobs…", "Cancel", 0, total, self)
        progress.setWindowTitle("Submitting to cluster")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def on_progress(index: int, count: int, label: str) -> None:
            progress.setLabelText(f"Submitting {label} ({index}/{count})")
            progress.setValue(index - 1)

        def on_one(job_id: object, remote_job_id: object, error: str) -> None:
            if not isinstance(job_id, UUID):
                return
            results_seen.add(job_id)
            prepared = prepared_by_job.get(job_id)
            if prepared is None:
                return
            if remote_job_id:
                outcome = self._job_execution.finalize_cluster_submission(prepared, str(remote_job_id))
            else:
                outcome = self._job_execution.fail_cluster_submission(prepared, error)
            if outcome.submitted_job is not None:
                submitted_jobs.append(outcome.submitted_job)
            if outcome.failed_message is not None:
                failed.append(outcome.failed_message)
            if outcome.journal_changed:
                self._refresh_journal_view()
            progress.setValue(progress.value() + 1)
            self._refresh_jobs_table()

        self._cluster_runner.submit_batch_progress.connect(on_progress)
        self._cluster_runner.job_submit_finished.connect(on_one)
        self._cluster_runner.submit_batch_finished.connect(loop.quit)
        progress.canceled.connect(self._cluster_runner.cancel_batch)
        progress.show()
        try:
            self._cluster_runner.submit_batch(submit_items)
            loop.exec()
        finally:
            self._cluster_runner.submit_batch_progress.disconnect(on_progress)
            self._cluster_runner.job_submit_finished.disconnect(on_one)
            self._cluster_runner.submit_batch_finished.disconnect(loop.quit)
            progress.close()

        # Jobs prepared but never submitted (cancelled mid-batch) are cancelled so
        # they do not linger in the ``submitting`` state.
        for job_id, prepared in prepared_by_job.items():
            if job_id not in results_seen:
                self._job_execution.cancel_job(prepared.job.id)
        return submitted_jobs, failed

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
        message = make_message_box_resizable(QMessageBox(self))
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

    @classmethod
    def _sanitize_remote_path_fragment(cls, value: str) -> str:
        parts = [
            cls._sanitize_remote_path_component(part)
            for part in value.strip().split("/")
            if part.strip()
        ]
        return "/".join(parts) or "workflow"

    @staticmethod
    def _default_cluster_job_name_prefix(workflow_type: str | None) -> str:
        normalized = (workflow_type or "").strip().strip("/")
        workflow_defaults = {
            "pristine_relaxation": "pristine_relax",
            "uep": "uep",
        }
        if normalized in workflow_defaults:
            return workflow_defaults[normalized]
        last_component = normalized.rsplit("/", 1)[-1]
        if last_component in {"positivemuon", "muonium"}:
            return last_component
        return "convergence"

    def _cluster_workflow_directory_fragment(self, jobs: list[Job]) -> str:
        muon_calc_types = {job.calc_type for job in jobs if is_muon_relaxation_calc_type(job.calc_type)}
        if muon_calc_types and len(muon_calc_types) == len({job.calc_type for job in jobs}):
            if len(muon_calc_types) == 1:
                calc_type = next(iter(muon_calc_types))
                prefix = (
                    "positivemuon"
                    if calc_type == MUON_RELAXATION_CALC_TYPES["positive_muon"]
                    else "muonium"
                )
                return self._sanitize_remote_path_fragment(f"muon_relaxation/{prefix}")
            return "muon_relaxation_mixed_charge_states"
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

        message = make_message_box_resizable(QMessageBox(self))
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
            cleaned_output_directory = False
            if cleanup_choice:
                output_dir = self._job_output_directory(job)
                if output_dir is not None and output_dir.exists() and output_dir.is_dir():
                    try:
                        self._clear_directory_contents(output_dir)
                        cleaned_output_directory = True
                    except OSError as exc:
                        cleanup_errors.append(f"- {job.label}: {exc}")

            try:
                self._project_service.reset_failed_job(job)
            except ValueError:
                continue

            if job.workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF:
                self._detach_convergence_job(job.id)

            self._active_run_by_job.pop(job.id, None)
            self._record_job_reset(job=job, cleaned_output_directory=cleaned_output_directory)
            reset_count += 1

        self._refresh_jobs_table()
        self._refresh_convergence_cutoff_rows()
        self._refresh_kpoint_convergence_rows()

        if cleanup_errors:
            QMessageBox.warning(
                self,
                "Directory cleanup issues",
                "Some job directories could not be fully cleaned:\n" + "\n".join(cleanup_errors),
            )

        if reset_count:
            self._persist_document_ui_state(
                failure_message="Reset job state was recorded, but autosave failed.",
            )
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

    def _submit_workflow_selection(
        self,
        values: list[_WorkflowKey],
        *,
        normalize: Callable[[object], _WorkflowKey],
        build_states: Callable[[], Mapping[_WorkflowKey, object]],
        jobs_for_values: Callable[[list[_WorkflowKey]], list[Job]],
        materialize_values: Callable[[list[_WorkflowKey]], list[Job]],
        is_ready: Callable[[Mapping[_WorkflowKey, object], _WorkflowKey], bool],
        no_ready_message: str,
        submit: Callable[[list[Job]], None],
    ) -> None:
        plan = self._workflow_submission.submission_plan(
            values,
            normalize=normalize,
            build_states=build_states,
            jobs_for_values=jobs_for_values,
            materialize_values=materialize_values,
            is_ready=is_ready,
        )
        if not plan.has_jobs:
            QMessageBox.information(self, "No ready jobs", no_ready_message)
            return
        submit(list(plan.jobs))

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
            lines.append("No .err or SLURM output files were found in the output directory.")
            return "\n".join(lines)

        for path in candidates:
            lines.append("")
            lines.append(f"[{path.name}]")
            lines.append(self._read_error_contexts(path))

        return "\n".join(lines)

    def _error_file_candidates(self, job: Job, output_dir: Path) -> list[Path]:
        discovered: list[Path] = []
        seed_names = list(dict.fromkeys([job.label, self._job_seed_name(job)]))
        for seed_name in seed_names:
            preferred = output_dir / f"{seed_name}.err"
            if preferred.exists() and preferred.is_file():
                discovered.append(preferred)

        remote_job_id = (job.remote_job_id or "").strip()
        if remote_job_id and re.fullmatch(r"[A-Za-z0-9_.-]+", remote_job_id):
            for slurm_output in sorted(output_dir.glob(f"*-{remote_job_id}.out")):
                if slurm_output.is_file():
                    discovered.append(slurm_output)

        numbered_patterns = [
            re.compile(rf"{re.escape(seed_name)}\.\d+\.err$")
            for seed_name in seed_names
        ]
        if not output_dir.exists() or not output_dir.is_dir():
            return list(dict.fromkeys(discovered))
        for candidate in sorted(output_dir.iterdir()):
            if candidate.is_file() and any(pattern.fullmatch(candidate.name) for pattern in numbered_patterns):
                discovered.append(candidate)

        if discovered:
            return list(dict.fromkeys(discovered))

        fallback_patterns = ("*.err", "slurm-*.out")
        if remote_job_id and re.fullmatch(r"[A-Za-z0-9_.-]+", remote_job_id):
            fallback_patterns = (*fallback_patterns, f"*-{remote_job_id}.out")
        fallback: list[Path] = []
        for pattern in fallback_patterns:
            fallback.extend(sorted(path for path in output_dir.glob(pattern) if path.is_file()))
        return list(dict.fromkeys(fallback))

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

    def _run_for_job(self, job_id: UUID) -> CalculationRun | None:
        return self._job_execution.run_for_job(job_id)

    def _results_for_run(self, run_id: UUID) -> list[Result]:
        if self._document is None:
            return []
        return [result for result in self._document.results.values() if result.calculation_run_id == run_id]

    @staticmethod
    def _charge_state_label_for_job(job: Job | None) -> str:
        if job is None:
            return ""
        if job.calc_type == muon_relaxation_calc_type(POSITIVE_MUON_CHARGE_STATE):
            return muon_charge_state_label(POSITIVE_MUON_CHARGE_STATE)
        if job.calc_type == muon_relaxation_calc_type(MUONIUM_CHARGE_STATE):
            return muon_charge_state_label(MUONIUM_CHARGE_STATE)
        return ""

    def _refresh_quantum_corrections_view(self) -> None:
        if self._document is None:
            self._quantum_corrections_view.set_sources([])
            self._quantum_corrections_view.set_results([])
            return

        source_rows: list[QuantumSourceRow] = []
        quantum_states = self._build_quantum_corrections_states()
        definition = self._active_quantum_corrections_definition()
        for candidate in self._eligible_quantum_source_candidates():
            state = quantum_states.get(candidate.key)
            stage_state = self._quantum_stage_state(definition, candidate)
            _params, _advanced, extras = self._workflow_row_configuration(definition, candidate.key)
            source_rows.append(
                QuantumSourceRow(
                    key=candidate.key,
                    label=candidate.site.label or candidate.job.label,
                    charge_state=self._charge_state_label_for_job(candidate.job),
                    relative_energy=_format_energy(candidate.site.relative_energy_eV),
                    displacements=str(3 * len(candidate.structure.species)),
                    source_job=candidate.job.label,
                    stage=stage_state.stage,
                    num_farms_display=(
                        str(self._quantum_num_farms_from_extras(extras))
                        if stage_state.stage == _QUANTUM_STAGE_PHONON
                        else ""
                    ),
                    status=getattr(state, "status", "ready") or "ready",
                    configured=self._workflow_row_configuration_exists(definition, candidate.key),
                    resettable=self._quantum_can_reset_to_stage_one(definition, candidate),
                )
            )

        result_rows: list[QuantumResultRow] = []
        for result in self._document.results.values():
            if result.result_type != QUANTUM_PHONON_CALC_TYPE:
                continue
            job_label = ""
            run = self._document.runs.get(result.calculation_run_id) if result.calculation_run_id is not None else None
            if run is not None and run.job_id is not None:
                job = self._document.jobs.get(run.job_id)
                if job is not None:
                    job_label = job.label
            eigenvalue_text = (
                _format_phonon_eigenvalues(result.data.get("phonon_mode_eigenvalues_cm"))
                if isinstance(result.data, dict)
                else ""
            )
            if not eigenvalue_text and isinstance(result.data, dict):
                modes = result.data.get("muon_zpe_modes", [])
                if isinstance(modes, list):
                    eigenvalue_text = ", ".join(
                        f"{mode.get('mode_index')}:{float(mode.get('frequency_cm', 0.0)):.1f}"
                        for mode in modes
                        if isinstance(mode, dict)
                    )
            result_rows.append(
                QuantumResultRow(
                    key=str(result.id),
                    label=job_label or "Quantum phonon",
                    zpe=_format_energy(result.data.get("muon_zpe_total_eV") if isinstance(result.data, dict) else None),
                    eigenvalues=eigenvalue_text,
                    witness=(
                        f"{float(result.data['w1_entanglement_witness']):.4g}"
                        if isinstance(result.data, dict) and "w1_entanglement_witness" in result.data
                        else ""
                    ),
                )
            )

        self._quantum_corrections_view.set_sources(source_rows)
        self._quantum_corrections_view.set_results(result_rows)

    def _refresh_energy_barriers_view(self) -> None:
        energy_barriers_view = self._energy_barriers_view
        if energy_barriers_view is None:
            return
        if self._document is None:
            energy_barriers_view.set_endpoint_options([])
            energy_barriers_view.set_sources([])
            energy_barriers_view.set_results([])
            energy_barriers_view.set_neb_results([])
            return

        endpoint_rows = self._energy_barrier_endpoint_rows()
        reactant_key, product_key = self._selected_energy_barrier_endpoint_keys()
        energy_barriers_view.set_endpoint_options(
            endpoint_rows,
            reactant_key=reactant_key,
            product_key=product_key,
            max_path_length_text=self._energy_barrier_max_path_length_text(),
        )

        source_rows: list[EnergyBarrierSourceRow] = []
        barrier_states = self._build_energy_barriers_states()
        source_definition = self._active_energy_barriers_definition()
        for candidate in self._eligible_energy_barrier_source_candidates():
            state = barrier_states.get(candidate.key)
            source_rows.append(
                EnergyBarrierSourceRow(
                    key=candidate.key,
                    reactant=candidate.reactant.site.label or candidate.reactant.job.label,
                    product=candidate.product.site.label or candidate.product.job.label,
                    charge_state=self._charge_state_label_for_job(candidate.reactant.job),
                    path_length=f"{candidate.path.length_ang:.4g} ang",
                    source_jobs=f"{candidate.reactant.job.label} -> {candidate.product.job.label}",
                    status=getattr(state, "status", "ready") or "ready",
                    configured=self._workflow_row_configuration_exists(source_definition, candidate.key),
                )
            )

        result_rows: list[EnergyBarrierResultRow] = []
        neb_states = self._build_energy_barrier_neb_states()
        neb_definition = self._active_energy_barrier_neb_definition()
        for candidate in self._energy_barrier_neb_source_candidates():
            # Endpoint path ready for NEB; the computed barrier appears in the NEB
            # results table once the job completes.
            label = str(candidate.source.get("label", "")).strip() or candidate.key
            result_rows.append(
                EnergyBarrierResultRow(
                    key=candidate.key,
                    label=label,
                    neb_status=getattr(neb_states.get(candidate.key), "status", "ready") or "ready",
                    configured=self._workflow_row_configuration_exists(neb_definition, candidate.key),
                )
            )

        energy_barriers_view.set_sources(source_rows)
        energy_barriers_view.set_results(result_rows)

        neb_result_rows: list[EnergyBarrierNebResultRow] = []
        for candidate in self._completed_energy_barrier_neb_candidates():
            result = candidate.result
            neb_result_rows.append(
                EnergyBarrierNebResultRow(
                    key=candidate.key,
                    label=candidate.job.label,
                    barrier_height=_format_energy(result.data.get("barrier_height_eV") if isinstance(result.data, dict) else None),
                    images=str(result.data.get("num_images", "")) if isinstance(result.data, dict) else "",
                    transition_state_file=(
                        str(result.data.get("transition_state_file", "")) if isinstance(result.data, dict) else ""
                    ),
                )
            )
        energy_barriers_view.set_neb_results(neb_result_rows)

    def _eligible_quantum_source_candidates(self) -> list[_QuantumSourceCandidate]:
        if self._document is None:
            return []
        candidates: list[_QuantumSourceCandidate] = []
        for site in self._document.sites.values():
            run = self._document.runs.get(site.relaxation_run_id) if site.relaxation_run_id is not None else None
            job = self._document.jobs.get(run.job_id) if run is not None and run.job_id is not None else None
            result = self._document.results.get(run.result_id) if run is not None and run.result_id is not None else None
            relaxed_structure = None
            if result is not None and result.derived_structure_ids:
                relaxed_structure = self._document.structures.get(result.derived_structure_ids[0])
            verdict = quantum_source_eligibility(
                site=site,
                run=run,
                job=job,
                relaxed_structure=relaxed_structure,
            )
            if not verdict.eligible or run is None or job is None or relaxed_structure is None:
                continue
            candidates.append(
                _QuantumSourceCandidate(
                    key=str(site.id),
                    site=site,
                    run=run,
                    job=job,
                    structure=relaxed_structure,
                )
            )
        return sorted(candidates, key=lambda item: (item.site.relative_energy_eV is None, item.site.relative_energy_eV or 0.0, item.site.label))

    def _quantum_source_candidate_by_key(self, key: str) -> _QuantumSourceCandidate | None:
        for candidate in self._eligible_quantum_source_candidates():
            if candidate.key == key:
                return candidate
        return None

    def _quantum_source_labels_by_key(self) -> dict[str, str]:
        labels_by_key: dict[str, str] = {}
        counts: dict[str, int] = {}
        for candidate in self._eligible_quantum_source_candidates():
            base = str(candidate.site.label or candidate.job.label or "site").strip() or "site"
            count = counts.get(base, 0) + 1
            counts[base] = count
            labels_by_key[candidate.key] = base if count == 1 else f"{base}_{count}"
        return labels_by_key

    def _quantum_source_label(self, candidate: _QuantumSourceCandidate) -> str:
        return self._quantum_source_labels_by_key().get(candidate.key, "site")

    def _energy_barrier_form_state(self) -> dict[str, object]:
        if self._document is None:
            return {}
        raw = self._document.ui_state.get("energy_barriers_form", {})
        return dict(raw) if isinstance(raw, dict) else {}

    def _energy_barrier_max_path_length_text(self) -> str:
        state = self._energy_barrier_form_state()
        raw = state.get("max_path_length_ang")
        if raw in (None, ""):
            return ""
        try:
            return f"{float(raw):.6g}"
        except (TypeError, ValueError):
            return ""

    def _selected_energy_barrier_endpoint_keys(self) -> tuple[str, str]:
        state = self._energy_barrier_form_state()
        reactant_key = str(state.get("selected_reactant_key", "")).strip()
        product_key = str(state.get("selected_product_key", "")).strip()
        if reactant_key and product_key:
            return reactant_key, product_key
        candidates = self._eligible_quantum_source_candidates()
        if not reactant_key and candidates:
            reactant_key = candidates[0].key
        if not product_key:
            for candidate in candidates:
                if candidate.key != reactant_key:
                    product_key = candidate.key
                    break
        return reactant_key, product_key

    def _energy_barrier_endpoint_rows(self) -> list[EnergyBarrierEndpointRow]:
        rows: list[EnergyBarrierEndpointRow] = []
        for candidate in self._eligible_quantum_source_candidates():
            rows.append(
                EnergyBarrierEndpointRow(
                    key=candidate.key,
                    label=candidate.site.label or candidate.job.label or candidate.structure.label,
                    charge_state=self._charge_state_label_for_job(candidate.job),
                    source_job=candidate.job.label,
                    relative_energy=_format_energy(candidate.site.relative_energy_eV) or "",
                )
            )
        return rows

    def _serialize_energy_barrier_source_candidates(
        self,
        candidates: list[_EnergyBarrierSourceCandidate],
    ) -> list[dict[str, object]]:
        serialized: list[dict[str, object]] = []
        for candidate in candidates:
            serialized.append(
                {
                    "key": candidate.key,
                    "label": self._energy_barrier_source_label(candidate),
                    "path_index": candidate.path_index,
                    "reactant_site_id": str(candidate.reactant.site.id),
                    "product_site_id": str(candidate.product.site.id),
                    "reactant_run_id": str(candidate.reactant.run.id),
                    "product_run_id": str(candidate.product.run.id),
                    "reactant_job_id": str(candidate.reactant.job.id),
                    "product_job_id": str(candidate.product.job.id),
                    "reactant_structure_id": str(candidate.reactant.structure.id),
                    "product_structure_id": str(candidate.product.structure.id),
                    "path_start_fractional": list(candidate.path.start_fractional),
                    "path_end_fractional": list(candidate.path.end_fractional),
                    "path_length_ang": candidate.path.length_ang,
                    "charge_state": self._charge_state_for_job(candidate.reactant.job),
                }
            )
        return serialized

    def _energy_barrier_generated_sources_from_state(self) -> list[dict[str, object]]:
        state = self._energy_barrier_form_state()
        raw_sources = state.get("generated_sources", [])
        return [dict(source) for source in raw_sources if isinstance(source, dict)] if isinstance(raw_sources, list) else []

    def _energy_barrier_generated_sources(self) -> list[dict[str, object]]:
        sources = self._energy_barrier_generated_sources_from_state()
        for definition in self._energy_barriers_definitions():
            config = definition.parameters.get("configuration", {})
            if not isinstance(config, dict):
                continue
            raw_sources = config.get("sources", [])
            if not isinstance(raw_sources, list):
                continue
            sources = _merge_serialized_sources(
                sources,
                [dict(source) for source in raw_sources if isinstance(source, dict)],
                key_field="key",
            )
        return sources

    def _enumerate_energy_barrier_source_candidates(
        self,
        reactant_key: str,
        product_key: str,
        *,
        max_distance_ang: float | None = None,
    ) -> list[_EnergyBarrierSourceCandidate]:
        if self._document is None:
            return []
        host_id = self._document.project.host_structure_id
        host = self._document.structures.get(host_id) if host_id is not None else None
        reactant = self._quantum_source_candidate_by_key(reactant_key)
        product = self._quantum_source_candidate_by_key(product_key)
        if host is None or reactant is None or product is None:
            return []
        if self._charge_state_for_job(reactant.job) != self._charge_state_for_job(product.job):
            return []
        if reactant.structure.parent_structure_id != host.id or product.structure.parent_structure_id != host.id:
            return []
        try:
            paths = enumerate_distinct_muon_paths(
                pristine_structure=host,
                reactant_structure=reactant.structure,
                product_structure=product.structure,
                max_distance_ang=max_distance_ang,
            )
        except EnergyBarrierError:
            return []
        return [
            _EnergyBarrierSourceCandidate(
                key=f"{reactant.site.id}:{product.site.id}:{path_index}",
                reactant=reactant,
                product=product,
                path=path,
                path_index=path_index,
            )
            for path_index, path in enumerate(paths, start=1)
        ]

    def _eligible_energy_barrier_source_candidates(self) -> list[_EnergyBarrierSourceCandidate]:
        if self._document is None:
            return []
        candidates: list[_EnergyBarrierSourceCandidate] = []
        for raw_source in self._energy_barrier_generated_sources():
            if not isinstance(raw_source, dict):
                continue
            reactant_site_id = self._uuid_from_source(raw_source, "reactant_site_id")
            product_site_id = self._uuid_from_source(raw_source, "product_site_id")
            reactant_run_id = self._uuid_from_source(raw_source, "reactant_run_id")
            product_run_id = self._uuid_from_source(raw_source, "product_run_id")
            reactant_job_id = self._uuid_from_source(raw_source, "reactant_job_id")
            product_job_id = self._uuid_from_source(raw_source, "product_job_id")
            reactant_structure_id = self._uuid_from_source(raw_source, "reactant_structure_id")
            product_structure_id = self._uuid_from_source(raw_source, "product_structure_id")
            if any(
                value is None
                for value in (
                    reactant_site_id,
                    product_site_id,
                    reactant_run_id,
                    product_run_id,
                    reactant_job_id,
                    product_job_id,
                    reactant_structure_id,
                    product_structure_id,
                )
            ):
                continue
            reactant_site = self._document.sites.get(reactant_site_id)
            product_site = self._document.sites.get(product_site_id)
            reactant_run = self._document.runs.get(reactant_run_id)
            product_run = self._document.runs.get(product_run_id)
            reactant_job = self._document.jobs.get(reactant_job_id)
            product_job = self._document.jobs.get(product_job_id)
            reactant_structure = self._document.structures.get(reactant_structure_id)
            product_structure = self._document.structures.get(product_structure_id)
            if any(
                value is None
                for value in (
                    reactant_site,
                    product_site,
                    reactant_run,
                    product_run,
                    reactant_job,
                    product_job,
                    reactant_structure,
                    product_structure,
                )
            ):
                continue
            try:
                path_index = int(raw_source.get("path_index", 1))
                path = MuonPath(
                    start_fractional=self._fractional_triplet_from_source(raw_source, "path_start_fractional"),
                    end_fractional=self._fractional_triplet_from_source(raw_source, "path_end_fractional"),
                    length_ang=float(raw_source.get("path_length_ang", 0.0)),
                )
            except (TypeError, ValueError):
                continue
            candidates.append(
                _EnergyBarrierSourceCandidate(
                    key=str(raw_source.get("key", "")).strip(),
                    reactant=_QuantumSourceCandidate(
                        key=str(reactant_site.id),
                        site=reactant_site,
                        run=reactant_run,
                        job=reactant_job,
                        structure=reactant_structure,
                    ),
                    product=_QuantumSourceCandidate(
                        key=str(product_site.id),
                        site=product_site,
                        run=product_run,
                        job=product_job,
                        structure=product_structure,
                    ),
                    path=path,
                    path_index=path_index,
                )
            )
        return candidates

    def _energy_barrier_source_candidate_by_key(self, key: str) -> _EnergyBarrierSourceCandidate | None:
        for candidate in self._eligible_energy_barrier_source_candidates():
            if candidate.key == key:
                return candidate
        return None

    def _energy_barrier_source_label(self, candidate: _EnergyBarrierSourceCandidate) -> str:
        reactant = candidate.reactant.site.label or candidate.reactant.job.label or "reactant"
        product = candidate.product.site.label or candidate.product.job.label or "product"
        return f"{reactant}_to_{product}_path{candidate.path_index}"

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

            if definition.workflow_stage == WorkflowStage.QUANTUM_CORRECTIONS:
                base = self._definition_output_directory(definition)
                if base is None:
                    continue
                for mapping_name in ("singlepoint_job_ids_by_label", "phonon_job_ids_by_label", "job_ids_by_label"):
                    mapping = params.get(mapping_name, {})
                    if not isinstance(mapping, dict):
                        continue
                    matching_key = next(
                        (
                            str(key)
                            for key, value in mapping.items()
                            if str(value) == str(job.id)
                        ),
                        "",
                    )
                    if matching_key:
                        return base / self._quantum_seed_name_for_key(definition, matching_key)

            job_ids_by_label = params.get("job_ids_by_label", {})
            if isinstance(job_ids_by_label, dict) and str(job.id) in {
                str(value) for value in job_ids_by_label.values()
            }:
                base = self._definition_output_directory(definition)
                if base is not None:
                    if definition.workflow_stage == WorkflowStage.MUON_RELAXATION and is_muon_relaxation_calc_type(job.calc_type):
                        matching_key = next(
                            (
                                str(key)
                                for key, value in job_ids_by_label.items()
                                if str(value) == str(job.id)
                            ),
                            "",
                        )
                        prefix = (
                            "positivemuon"
                            if job.calc_type == MUON_RELAXATION_CALC_TYPES["positive_muon"]
                            else "muonium"
                        )
                        seed_mapping = params.get("job_seed_names_by_label", {})
                        seed_name = job.label
                        if isinstance(seed_mapping, dict):
                            raw_seed = seed_mapping.get(matching_key)
                            if isinstance(raw_seed, str) and raw_seed.strip():
                                seed_name = raw_seed.strip()
                        if seed_name == job.label:
                            remote_seed = self._remote_directory_seed_name(job)
                            if remote_seed:
                                seed_name = remote_seed
                        if seed_name == job.label:
                            inferred_seed = job_name(prefix, job.label)
                            inferred_path = base / prefix / inferred_seed
                            if inferred_path.exists() and inferred_path.is_dir():
                                seed_name = inferred_seed
                        return base / prefix / seed_name
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

    def _job_seed_name(self, job: Job) -> str:
        remote_seed = self._remote_directory_seed_name(job)
        if remote_seed:
            return remote_seed
        output_dir = self._job_output_directory(job)
        if output_dir is not None:
            return output_dir.name
        return job.label

    @staticmethod
    def _remote_directory_seed_name(job: Job) -> str:
        remote_directory = (job.remote_directory or "").strip()
        if not remote_directory:
            return ""
        seed_name = posixpath.basename(posixpath.normpath(remote_directory)).strip()
        return seed_name if seed_name not in {"", ".", "/"} else ""

    def _output_files_for_job(self, job: Job) -> list[str]:
        if self._document is None:
            return []
        output_dir = self._job_output_directory(job)
        if output_dir is None:
            return []
        seed_name = self._job_seed_name(job)

        paths = [
            output_dir / f"{seed_name}.cell",
            output_dir / f"{seed_name}.param",
            output_dir / f"{seed_name}.castep",
            output_dir / f"{seed_name}_farm001.castep",
            output_dir / f"{seed_name}.den_fmt",
            output_dir / f"{seed_name}-out.cell",
            output_dir / f"{seed_name}.phonon",
            output_dir / f"{seed_name}.ts",
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

    def _write_quantum_job_directory(self, job_spec, output_root: Path, *, seed_name: str) -> Path:
        job_dir = output_root / seed_name
        job_dir.mkdir(parents=True, exist_ok=True)
        cell_text, param_text = render_job(job_spec)
        (job_dir / f"{seed_name}.cell").write_text(cell_text, encoding="utf-8", newline="\n")
        (job_dir / f"{seed_name}.param").write_text(param_text, encoding="utf-8", newline="\n")
        return job_dir

    def _record_job_reset(self, *, job: Job, cleaned_output_directory: bool) -> None:
        if self._document is None:
            return
        self._project_service.record_job_reset(
            self._document,
            job=job,
            cleaned_output_directory=cleaned_output_directory,
        )
        self._refresh_journal_view()

    def _on_local_job_status_changed(self, job_id: UUID, status_value: str) -> None:
        if self._document is None:
            return
        update = self._job_execution.mark_local_status_changed(job_id, status_value)
        if update.changed:
            self._refresh_jobs_table()
            self._refresh_convergence_cutoff_rows()
            self._refresh_kpoint_convergence_rows()
            self._refresh_quantum_corrections_view()
            self._refresh_energy_barriers_view()
            self._persist_document_ui_state(
                failure_message="Local job status changed, but autosave failed.",
            )

    def _on_local_job_started(self, job_id: UUID) -> None:
        if self._document is None:
            return
        update = self._job_execution.mark_started(job_id)
        if update.changed:
            self._refresh_jobs_table()
            self._persist_document_ui_state(
                failure_message="Local job start was recorded, but autosave failed.",
            )

    def _on_local_job_finished(self, job_id: UUID, parsed: CastepResult) -> None:
        if self._document is None:
            return
        job = self._document.jobs.get(job_id)
        if job is None:
            return

        if (
            job.calc_type == "castep_pristine_relaxation"
            or is_muon_relaxation_calc_type(job.calc_type)
        ) and parsed.geom_converged is False:
            self._on_local_job_failed(
                job_id,
                parsed.error_message
                or "Geometry optimization did not satisfy CASTEP convergence criteria.",
            )
            return

        run = self._job_execution.ensure_run_for_job(job, status=JobStatus.RUNNING, started=True)
        if run is None:
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

        if job.calc_type == "castep_uep_source":
            output_dir = self._job_output_directory(job)
            seed_name = self._job_seed_name(job)
            density_path = None if output_dir is None else output_dir / f"{seed_name}.den_fmt"
            if density_path is None or not density_path.exists():
                self._on_local_job_failed(
                    job_id,
                    f"Expected formatted density output was not found: {seed_name}.den_fmt",
                )
                return
            result_payload["density_available"] = True

        if job.calc_type == QUANTUM_PHONON_CALC_TYPE:
            output_dir = self._job_output_directory(job)
            seed_name = self._job_seed_name(job)
            phonon_path = None if output_dir is None else output_dir / f"{seed_name}.phonon"
            if phonon_path is None or not phonon_path.exists():
                self._on_local_job_failed(
                    job_id,
                    f"Expected CASTEP phonon output was not found: {seed_name}.phonon",
                )
                return
            try:
                phonon = parse_phonon_path(phonon_path)
                zpe = muon_zpe(phonon)
                result_payload["phonon_file"] = str(phonon_path)
                result_payload["gamma_only"] = len(phonon.qpoints) == 1 and phonon.qpoints[0].is_gamma
                result_payload["phonon_mode_eigenvalues_cm"] = [
                    float(value) for value in phonon.gamma_point().frequencies_cm
                ]
                result_payload["muon_zpe_total_eV"] = zpe.total_eV
                result_payload["muon_zpe_modes"] = [
                    {
                        "mode_index": mode.mode_index,
                        "frequency_cm": mode.frequency_cm,
                        "participation": mode.participation,
                        "zpe_eV": mode.zpe_eV,
                    }
                    for mode in zpe.modes
                ]
                result_payload["w1_entanglement_witness"] = w1_entanglement_witness(phonon)
                if zpe.warnings:
                    result_payload["phonon_warnings"] = list(zpe.warnings)
            except PhononParseError as exc:
                self._on_local_job_failed(job_id, f"Failed to parse CASTEP phonon output: {exc}")
                return

        if job.calc_type in {TRANSITION_STATE_SEARCH_CALC_TYPE, NUDGED_ELASTIC_BAND_CALC_TYPE}:
            output_dir = self._job_output_directory(job)
            seed_name = self._job_seed_name(job)
            ts_path = None if output_dir is None else output_dir / f"{seed_name}.ts"
            if ts_path is None or not ts_path.exists():
                self._on_local_job_failed(
                    job_id,
                    f"Expected CASTEP path output was not found: {seed_name}.ts",
                )
                return
            try:
                transition_state = parse_transition_state_file(ts_path)
                result_payload["transition_state_file"] = str(ts_path)
                result_payload["barrier_height_eV"] = transition_state.barrier_height_eV
                result_payload["num_images"] = len(transition_state.images)
                result_payload["image_labels"] = [image.label for image in transition_state.images]
                result_payload["reaction_coordinates"] = [image.reaction_coordinate for image in transition_state.images]
                result_payload["image_energies_eV"] = [image.energy for image in transition_state.images]
            except TransitionStateParseError as exc:
                self._on_local_job_failed(job_id, f"Failed to parse CASTEP path output: {exc}")
                return

        derived_structure_ids: list[UUID] = []
        if job.calc_type == "castep_pristine_relaxation" or is_muon_relaxation_calc_type(job.calc_type):
            output_dir = self._job_output_directory(job)
            seed_name = self._job_seed_name(job)
            out_cell_path = None if output_dir is None else output_dir / f"{seed_name}-out.cell"
            if out_cell_path is None or not out_cell_path.exists():
                self._on_local_job_failed(
                    job_id,
                    f"Expected relaxed structure output was not found: {seed_name}-out.cell",
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

            if job.calc_type == "castep_pristine_relaxation":
                definition = self._active_pristine_relaxation_definition()
                origin = "pristine_relaxation"
                stage = WorkflowStage.PRISTINE_RELAXATION
            else:
                definition, row = self._muon_relaxation_definition_and_row_for_job(job)
                if definition is None:
                    definition = self._active_muon_relaxation_definition()
                origin = "muon_relaxation"
                stage = WorkflowStage.MUON_RELAXATION
                final_supercell_position = self._muon_fractional_position_from_cell(relaxed)
                if final_supercell_position is None:
                    self._on_local_job_failed(
                        job_id,
                        f"Expected H:mu muon position was not found in {seed_name}-out.cell.",
                    )
                    return
                final_host_position = self._fold_supercell_fractional_to_host(
                    final_supercell_position,
                    row.supercell_matrix if row is not None else None,
                )
                result_payload["muon_final_fractional_position_supercell"] = list(final_supercell_position)
                result_payload["muon_final_fractional_position_host"] = list(final_host_position)
            parent_structure_id = definition.input_structure_ids[0] if definition is not None and definition.input_structure_ids else None
            relaxed_structure = Structure(
                label=f"{job.label} relaxed",
                origin=origin,
                lattice_vectors=[list(row) for row in relaxed.lattice_vectors],
                species=list(relaxed.species),
                fractional_positions=[list(row) for row in relaxed.fractional_positions],
                parent_structure_id=parent_structure_id,
                properties={
                    "source_job_id": str(job.id),
                    "source_workflow_stage": stage.value,
                },
            )
            self._project_service.add_structure(self._document, relaxed_structure)
            derived_structure_ids.append(relaxed_structure.id)
            if is_muon_relaxation_calc_type(job.calc_type):
                site = self._site_for_muon_relaxation_job(job)
                if site is not None:
                    site.status = "relaxed"
                    site.relaxation_run_id = run.id
                    site.relative_energy_eV = parsed.total_energy

        result = Result(
            calculation_run_id=run.id,
            result_type=job.calc_type,
            data=result_payload,
            output_files=self._output_files_for_job(job),
            derived_structure_ids=derived_structure_ids,
        )
        self._document.results[result.id] = result

        self._job_execution.complete_job_with_result(
            job,
            result=result,
            warnings=parsed.warnings,
            convergence_achieved=parsed.geom_converged,
        )
        self._request_dashboard_flush(
            failure_message="Completed job result was recorded, but autosave failed.",
        )
        self.statusBar().showMessage(f"Completed local job {job.label}")

    def _on_local_job_failed(self, job_id: UUID, message: str) -> None:
        if self._document is None:
            return
        update = self._job_execution.fail_job(job_id, message)
        if update.job is None:
            return
        self._request_dashboard_flush(
            failure_message="Failed job state was recorded, but autosave failed.",
        )
        self.statusBar().showMessage(f"Local job failed: {update.job.label}")

    def _on_local_job_cancelled(self, job_id: UUID) -> None:
        if self._document is None:
            return
        update = self._job_execution.cancel_job(job_id)
        if update.job is None:
            return
        self._request_dashboard_flush(
            failure_message="Cancelled job state was recorded, but autosave failed.",
        )
        self.statusBar().showMessage(f"Cancelled local job {update.job.label}")

    def _on_cluster_job_status_changed(self, job_id: UUID, status_value: str) -> None:
        if self._document is None:
            return
        update = self._job_execution.mark_cluster_status_changed(job_id, status_value)
        if not update.changed:
            return
        self._request_dashboard_flush(
            failure_message="Cluster job status changed, but autosave failed.",
        )

    def _on_cluster_job_started(self, job_id: UUID) -> None:
        if self._document is None:
            return
        update = self._job_execution.mark_started(job_id)
        if update.changed:
            self._request_dashboard_flush(
                failure_message="Cluster job start was recorded, but autosave failed.",
            )

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

        self._cluster_elapsed_seconds_by_job = {}
        self._cluster_elapsed_checked_at = None
        self._cluster_connection_label.setText("Cluster: disconnected")
        self._refresh_cluster_connection_label_colour()
        self._cluster_queue_checked_label.setText("Queue last checked: disconnected")
        self._cluster_refresh_button.setEnabled(True)

    def _refresh_cluster_connection_label_colour(self) -> None:
        palette = current_palette()
        colour = palette.success if self._cluster_connected else palette.danger
        self._cluster_connection_label.setStyleSheet(f"color: {colour};")

    def _on_cluster_queue_checked(self, checked_at) -> None:
        # A queue poll fires this once after any jobs it retrieved have been
        # applied, so flush the autosave those completions coalesced into a single
        # write here.
        self._flush_dashboard()
        elapsed_by_job_id = getattr(checked_at, "elapsed_by_job_id", None)
        if isinstance(elapsed_by_job_id, Mapping):
            parsed: dict[UUID, int] = {}
            for job_id, elapsed in elapsed_by_job_id.items():
                try:
                    normalized_job_id = job_id if isinstance(job_id, UUID) else UUID(str(job_id))
                except (TypeError, ValueError, AttributeError):
                    continue
                seconds = self._parse_scheduler_elapsed_seconds(str(elapsed))
                if seconds is not None:
                    parsed[normalized_job_id] = seconds
            self._cluster_elapsed_seconds_by_job = parsed
        stamp_source = getattr(checked_at, "checked_at", checked_at)
        self._cluster_elapsed_checked_at = stamp_source if isinstance(stamp_source, datetime) else None
        if hasattr(stamp_source, "strftime"):
            localized = stamp_source
            if getattr(stamp_source, "tzinfo", None) is not None and hasattr(stamp_source, "astimezone"):
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
        self._sync_job_runtime_timer()
        self._refresh_active_job_runtime_cells()

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

    def _connect_to_cluster_with_trust(self, config: ClusterConfig, *, credentials: SSHCredentials) -> None:
        trusted_endpoints: set[str] = set()
        while True:
            error = self._await_cluster_connect(config, credentials=credentials)
            if error is None:
                return
            if isinstance(error, SSHUnknownHostKeyError):
                if error.endpoint in trusted_endpoints:
                    raise error
                if not prompt_to_trust_host_key(self, config, error):
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
                self._connect_to_cluster_with_trust(config, credentials=SSHCredentials())
                self._cluster_runner.refresh_queue()
                refreshed.append(config.name)
            except Exception as exc:  # noqa: BLE001
                auth_method = config.auth_method.strip().lower()
                needs_credentials = auth_method == "password" or isinstance(exc, SSHAuthenticationError)
                if needs_credentials:
                    credentials = prompt_for_cluster_credentials(self, config)
                    if credentials is None:
                        failures.append(
                            f"- {config.name}: credentials required for reconnect. Enter them when prompted."
                        )
                        continue
                    try:
                        self._connect_to_cluster_with_trust(config, credentials=credentials)
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
                seed_name=self._job_seed_name(job),
                local_output_dir=output_dir,
                cluster_name=target,
                status=job.status,
                retrieve_den_fmt=job.calc_type == "castep_uep_source",
                retrieve_out_cell=job.calc_type == "castep_pristine_relaxation"
                or is_muon_relaxation_calc_type(job.calc_type),
                retrieve_phonon=job.calc_type == QUANTUM_PHONON_CALC_TYPE,
                retrieve_ts=job.calc_type in {TRANSITION_STATE_SEARCH_CALC_TYPE, NUDGED_ELASTIC_BAND_CALC_TYPE},
                task_farming_enabled=self._quantum_task_farming_for_job(job)[0],
                num_farms=self._quantum_task_farming_for_job(job)[1],
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
                seed_name=self._job_seed_name(job),
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
        serialized_config = self._serialize_convergence_config(config)
        simulation_cell_mode = resolution.normalized_selection.mode
        supercell_source_workflow = (
            resolution.source_workflow_stage.value
            if resolution.source_workflow_stage is not None
            else None
        )

        matching_definition = self._matching_convergence_cutoff_definition(
            input_structure_id=structure.id,
            parameters=serialized_config,
            output_directory=output_root,
            simulation_cell_mode=simulation_cell_mode,
            supercell_source_workflow=supercell_source_workflow,
        )
        if matching_definition is not None:
            self._sync_equivalent_convergence_cutoff_mappings(matching_definition)
            self._document.ui_state[self._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(matching_definition.id)
            self._refresh_jobs_table()
            self._refresh_convergence_cutoff_rows()
            self._refresh_journal_view()
            self.statusBar().showMessage(
                f"Using existing cutoff setup with {len(config.cutoff_energies)} cutoff job(s)"
            )
            return

        self._project_service.register_convergence_cutoff_definition(
            self._document,
            input_structure_id=structure.id,
            parameters=serialized_config,
            output_directory=output_root,
            simulation_cell_mode=simulation_cell_mode,
            supercell_source_workflow=supercell_source_workflow,
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
        self._refresh_journal_view()
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
            QMessageBox.information(self, "Generation in progress", "Wait for the current generation task to finish.")
            return

        if method in {"random", "near_element"}:
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

        self._refresh_muon_generation_table()
        self._refresh_journal_view()
        self._finish_generation_status(method, parameters, created)

    def _start_muon_generation_thread(self, method: str, parameters: dict) -> None:
        if self._document is None:
            return
        thread = _MuonGenerationThread(
            self._project_service,
            self._document,
            method=method,
            parameters=parameters,
            parent=self,
        )
        self._muon_generation_thread = thread
        self._muon_generation_failed = False
        thread.generated.connect(
            lambda created, current_method=method, params=dict(parameters): self._on_threaded_generation_finished(
                current_method,
                params,
                created,
            )
        )
        thread.failed.connect(lambda message, current_method=method: self._on_threaded_generation_failed(current_method, message))
        random_mode = str(parameters.get("selection_mode", parameters.get("mode", ""))).strip().lower()
        thread.progress_changed.connect(
            lambda generated, expected, current_method=method, mode=random_mode: self._on_generation_progress_changed(
                current_method,
                generated,
                expected,
                mode,
            )
        )
        thread.finished.connect(self._on_muon_generation_thread_finished)
        thread.finished.connect(thread.deleteLater)
        method_label = "random" if method == "random" else "near-element"
        self._muon_generation_view.set_generation_busy(True, f"Preparing {method_label} generation...")
        self.statusBar().showMessage(f"Generating {method_label} muon sites...")
        thread.start()

    def _on_generation_progress_changed(self, method: str, generated: int, expected: int, mode: str) -> None:
        if method == "random" and mode in {"separation", "distance", "spacing"}:
            message = "Generated %v sites"
        else:
            message = "Generated %v/%m sites"
        self._muon_generation_view.set_generation_progress(
            generated,
            expected,
            message,
        )

    def _on_threaded_generation_finished(self, method: str, parameters: dict, created: object) -> None:
        sites = list(created) if isinstance(created, list) else []
        used_spacing: object | None = None
        if method == "random" and sites:
            used_spacing = sites[0].properties.get(
                "min_muon_muon",
                sites[0].properties.get("estimated_min_muon_muon"),
            )
        self._muon_generation_view.set_generation_progress(
            len(sites),
            max(len(sites), 1),
            f"Generated {len(sites)} site(s)",
        )
        self._refresh_muon_generation_table()
        if used_spacing is not None:
            self._muon_generation_view.set_random_muon_muon_distance(float(used_spacing))
        self._refresh_journal_view()
        self._finish_generation_status(method, parameters, sites)

    def _on_threaded_generation_failed(self, method: str, message: str) -> None:
        self._muon_generation_failed = True
        self._muon_generation_view.set_generation_busy(False, "Generation failed")
        QMessageBox.warning(self, "Generation blocked", message)
        method_label = "Random" if method == "random" else "Near-element"
        self.statusBar().showMessage(f"{method_label} site generation failed")

    def _on_muon_generation_thread_finished(self) -> None:
        self._muon_generation_thread = None
        if not self._muon_generation_failed:
            self._muon_generation_view.set_generation_controls_enabled(True)

    def _finish_generation_status(self, method: str, parameters: dict, created: list[MuonSite]) -> None:
        random_mode = str(parameters.get("selection_mode", parameters.get("mode", ""))).strip().lower()
        if method == "random" and random_mode in {"separation", "distance", "spacing"}:
            requested = None
        elif method == "uep_clusters" and isinstance(parameters.get("selections"), list):
            requested = len(parameters.get("selections", []))
        elif method == "near_element" and self._document is not None:
            host = self._project_service.active_host_structure(self._document)
            target = str(parameters.get("element", "")).strip().lower()
            matching_atoms = (
                sum(1 for species in host.species if str(species).strip().lower() == target)
                if host is not None and target
                else 0
            )
            requested = matching_atoms * max(1, int(parameters.get("muons_per_atom", parameters.get("count", 1))))
        elif method != "manual":
            requested = int(parameters.get("count", 1))
        else:
            requested = 1
        suffix = ""
        if requested is not None and len(created) < requested:
            suffix = f" (requested {requested}; constraints prevented full count)"
        if method == "random" and created and created[0].properties.get("estimated_min_muon_muon") is not None:
            estimated = float(created[0].properties["estimated_min_muon_muon"])
            used = float(created[0].properties.get("min_muon_muon", estimated))
            if abs(used - estimated) > 1e-9:
                suffix += f" (estimated spacing {estimated:.2f} A; used {used:.2f} A)"
            else:
                suffix += f" (estimated spacing {estimated:.2f} A)"
        self.statusBar().showMessage(f"Generated {len(created)} muon site(s){suffix}")

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
        self.statusBar().showMessage(f"Deleted {len(removed)} generated muon site(s)")

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
        self._refresh_muon_generation_table()
        self._muon_generation_view.select_site_ids(clustered_ids)
        self._refresh_journal_view()
        self.statusBar().showMessage(f"Clustered {len(clustered)} muon position(s)")

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
            mismatch = self._project_service.validate_sites_match_active_host(self._document, site_ids)
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
                "fractional_position": list(site.fractional_position),
                "method": site.generation_method or "",
            }
            for site in staged
        ]
        self._muon_relaxation_view.add_staged_sites(rows, charge_states=charge_states)
        self._refresh_muon_generation_table()
        self._refresh_muon_relaxation_defaults()
        self._refresh_muon_relaxation_rows()
        self._refresh_journal_view()
        self._set_active_route("muon_relaxations")
        charge_label = " and ".join(muon_charge_state_label(charge_state) for charge_state in charge_states)
        self.statusBar().showMessage(f"Staged {len(staged)} muon relaxation position(s) for {charge_label}")

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
            QMessageBox.warning(self, "No host structure", "Load a host structure before creating an analysis workspace.")
            return
        label_list = [str(label) for label in labels] if isinstance(labels, list) else [str(labels)]
        sources = self._completed_muon_relaxation_sources_for_labels(label_list)
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
        self._refresh_relaxation_analysis_context()
        self._populate_project_tree()
        self._set_active_route("muon_analysis")
        self._relaxation_analysis_view.focus_workspace(str(workspace["id"]))
        self.statusBar().showMessage(f"Created analysis workspace '{workspace['name']}'")

    def _on_relaxation_analysis_state_changed(self, workspaces: list) -> None:
        if self._document is None or self._shutting_down:
            return
        cleaned = [dict(workspace) for workspace in workspaces if isinstance(workspace, dict)]
        self._set_relaxation_analysis_workspaces(cleaned)
        self._refresh_workflow_stage_icons()

    def _on_relaxation_analysis_add_sites_requested(self, workspace_id: str) -> None:
        if self._document is None:
            return
        workspace = next(
            (entry for entry in self._relaxation_analysis_workspaces() if str(entry.get("id", "")) == str(workspace_id)),
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
        self._relaxation_analysis_view.add_job_ids_to_workspace(str(workspace_id), selected)
        self._refresh_relaxation_analysis_context()
        self.statusBar().showMessage(f"Added {len(selected)} relaxation result(s) to the analysis workspace")

    def _on_convergence_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
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
            self._convergence_view.set_parameter_species_options(())
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
        self._convergence_view.set_parameter_species_options(
            _structure_species_options(resolution.structure)
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

    def _refresh_all_supercell_options(self) -> None:
        self._refresh_convergence_simulation_cell()
        self._refresh_kpoint_convergence_simulation_cell()
        self._refresh_pristine_relaxation_supercell_options()
        self._refresh_muon_relaxation_supercell_options()
        self._refresh_uep_supercell_options()

    def _open_supercell_editor(
        self,
        *,
        source_structure: Structure,
        workflow_stage: WorkflowStage,
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
        self._persist_document_ui_state(failure_message="Supercells were updated, but autosave failed.")
        if select_mode and result != dialog.DialogCode.Accepted:
            return None
        return dialog

    def _on_host_supercell_editor_requested(self) -> None:
        if self._document is None:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        host = self._project_service.active_host_structure(self._document)
        if host is None:
            QMessageBox.warning(self, "No host structure", "Load a host structure before editing supercells.")
            return
        dialog = self._open_supercell_editor(
            source_structure=host,
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            initial_matrix=self._document.project.supercell_matrix,
            select_mode=False,
        )
        if dialog is not None:
            self._host_structure_view.set_project_supercell_matrix(dialog.matrix())

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

        dialog = self._open_supercell_editor(
            source_structure=host,
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            initial_matrix=self._convergence_supercell_initial_matrix(),
            select_mode=True,
        )
        if dialog is None or dialog.selected_structure_id() is None:
            return

        selected = self._project_service.use_workflow_supercell_asset(
            self._document,
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            asset_structure_id=dialog.selected_structure_id(),
        )
        self._refresh_convergence_simulation_cell()
        self.statusBar().showMessage(f"Selected convergence simulation cell '{selected.label}'")

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

        self._sync_equivalent_convergence_kpoint_mappings(active_definition)
        config = self._deserialize_convergence_config(active_definition.parameters.get("configuration", {}))
        if config is None or config.sweep_parameter != "kpoint_mp_grid":
            return None, None

        if grid in config.sweep_values:
            return active_definition, config
        return None, None

    def _job_for_definition_grid(self, definition, grid: tuple[int, int, int]) -> Job | None:
        if self._document is None:
            return None

        mapping = definition.parameters.get("job_ids_by_grid")
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
            return None

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

        self._sync_equivalent_convergence_kpoint_mappings(definition)
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
                max_stress=self._max_stress_for_job(job.id),
                job_id=job.id,
                job_label=job.label,
            )
        return states

    def _jobs_for_convergence_kpoint_grids(self, grids: list[tuple[int, int, int]]) -> list[Job]:
        if self._document is None:
            return []

        states = self._build_convergence_kpoint_states()
        selected: list[Job] = []
        seen: set[UUID] = set()
        for grid in grids:
            state = states.get(tuple(grid))
            if state is None:
                continue
            job = self._document.jobs.get(state.job_id) if state.job_id is not None else None
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

        definition = self._project_service.register_convergence_kpoint_definition(
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
        self._sync_equivalent_convergence_kpoint_mappings(definition)
        self._document.ui_state[self._ACTIVE_KPOINT_DEFINITION_KEY] = str(definition.id)

        self._refresh_jobs_table()
        self._refresh_kpoint_convergence_rows()
        self._refresh_journal_view()
        self.statusBar().showMessage(
            f"Configured {len(config.sweep_values)} k-point jobs; input files will be generated on submission"
        )

    def _on_kpoint_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
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
            self._kpoint_convergence_view.set_parameter_species_options(())
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
        self._kpoint_convergence_view.set_parameter_species_options(
            _structure_species_options(resolution.structure)
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

        dialog = self._open_supercell_editor(
            source_structure=host,
            workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
            initial_matrix=self._kpoint_supercell_initial_matrix(),
            select_mode=True,
        )
        if dialog is None or dialog.selected_structure_id() is None:
            return

        selected = self._project_service.use_workflow_supercell_asset(
            self._document,
            workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
            asset_structure_id=dialog.selected_structure_id(),
        )
        self._refresh_kpoint_convergence_simulation_cell()
        self.statusBar().showMessage(f"Selected k-point simulation cell '{selected.label}'")

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
        self._submit_workflow_selection(
            grids,
            normalize=lambda grid: (int(grid[0]), int(grid[1]), int(grid[2])),
            build_states=self._build_convergence_kpoint_states,
            jobs_for_values=self._jobs_for_convergence_kpoint_grids,
            materialize_values=self._materialize_kpoint_jobs,
            is_ready=sweep_point_state_is_ready,
            no_ready_message="Only configured grids in the 'ready' state can be submitted.",
            submit=self._submit_jobs_locally,
        )

    def _on_submit_cluster_from_kpoint_convergence(
        self,
        grids: list[tuple[int, int, int]],
    ) -> None:
        self._submit_workflow_selection(
            grids,
            normalize=lambda grid: (int(grid[0]), int(grid[1]), int(grid[2])),
            build_states=self._build_convergence_kpoint_states,
            jobs_for_values=self._jobs_for_convergence_kpoint_grids,
            materialize_values=self._materialize_kpoint_jobs,
            is_ready=sweep_point_state_is_ready,
            no_ready_message="Only configured grids in the 'ready' state can be submitted.",
            submit=self._submit_jobs_on_cluster,
        )

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

        target_keys = {self._grid_key(grid) for grid in grids}
        for definition in self._kpoint_definitions():
            self._remove_kpoint_grids_from_definition(definition, target_keys)

        self._kpoint_convergence_view.remove_grids(grids)
        self._refresh_kpoint_convergence_rows()

    def _remove_kpoint_grids_from_definition(self, definition, target_keys: set[str]) -> None:
        if self._document is None or not target_keys:
            return
        configuration = definition.parameters.get("configuration")
        if isinstance(configuration, dict):
            raw_values = configuration.get("sweep_values", [])
            updated_values: list[object] = []
            if isinstance(raw_values, list):
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
            lattice_vectors=host.lattice_vectors,
            initial_state=persisted_state,
            parent=self,
        )
        dialog.default_adopted.connect(
            lambda value, method, et, ft, st: self._on_kpoint_default_adopted(value, method, et, ft, labels, st)
        )
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

    def _on_kpoint_default_adopted(
        self,
        value: float,
        method: str,
        energy_threshold: float,
        force_threshold: float,
        labels: Mapping[float, str],
        stress_threshold: float | None = None,
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
            stress_threshold=stress_threshold,
            cell_id=(
                definition_cell_id
                if definition_cell_id is not None
                else (str(self._document.project.host_structure_id) if self._document.project.host_structure_id else None)
            ),
        )
        if stress_threshold is None:
            self._document.ui_state.pop("default_kpoint_stress_threshold", None)
        else:
            self._document.ui_state["default_kpoint_stress_threshold"] = float(stress_threshold)
        self._kpoint_convergence_view.set_project_defaults(
            kpoint_grid=grid,
            kpoint_method=method,
            energy_threshold=energy_threshold,
            force_threshold=force_threshold,
            stress_threshold=float(stress_threshold) if stress_threshold is not None else None,
        )
        self._refresh_pristine_relaxation_defaults()
        self._refresh_muon_relaxation_defaults()
        self._refresh_uep_defaults()
        self._store_convergence_overview_snapshot(
            "kpoint",
            self._build_kpoint_overview_snapshot(
                grid=grid,
                method=method,
                energy_threshold=float(energy_threshold),
                force_threshold=float(force_threshold),
                simulation_cell_structure_id=definition_cell_id,
                labels=labels,
                stress_threshold=float(stress_threshold) if stress_threshold is not None else None,
            ),
        )
        self._refresh_convergence_overview()
        self._refresh_journal_view()
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
            self._pristine_relaxation_view.set_parameter_species_options(())
            return

        host = self._project_service.active_host_structure(self._document)
        self._pristine_relaxation_view.set_parameter_species_options(
            _structure_species_options(host)
        )
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
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state["pristine_relaxation_form"] = state
        self._refresh_pristine_relaxation_rows()

    def _refresh_muon_relaxation_defaults(self) -> None:
        if self._document is None:
            self._muon_relaxation_view.set_host_lattice(None)
            self._muon_relaxation_view.set_project_defaults(
                cutoff_energy=None,
                kpoint_grid=None,
                kpoint_reference_lattice=None,
            )
            self._muon_relaxation_view.set_supercell_options([])
            self._muon_relaxation_view.set_parameter_species_options(())
            return

        host = self._project_service.active_host_structure(self._document)
        self._muon_relaxation_view.set_parameter_species_options(
            _structure_species_options(host)
        )
        self._muon_relaxation_view.set_host_lattice(
            host.lattice_vectors if host is not None else None
        )
        self._muon_relaxation_view.set_project_defaults(
            cutoff_energy=self._document.project.recommended_cutoff_energy,
            kpoint_grid=self._document.project.recommended_kpoint_grid,
            kpoint_reference_lattice=self._document.project.recommended_kpoint_lattice,
        )
        self._refresh_muon_relaxation_supercell_options()

    def _refresh_muon_relaxation_supercell_options(self) -> None:
        if self._document is None:
            self._muon_relaxation_view.set_supercell_options([])
            return
        host = self._project_service.active_host_structure(self._document)
        if host is None:
            self._muon_relaxation_view.set_supercell_options([])
            return
        assets = self._project_service.list_workflow_supercell_assets(
            self._document,
            host_structure_id=host.id,
        )
        self._muon_relaxation_view.set_supercell_options(
            [
                {"label": asset.label, "matrix": [list(row) for row in asset.matrix]}
                for asset in assets
            ]
        )

    def _refresh_uep_defaults(self) -> None:
        if self._document is None:
            self._uep_view.set_project_defaults(
                cutoff_energy=None,
                kpoint_grid=None,
                kpoint_reference_lattice=None,
            )
            self._uep_view.set_host_lattice(None)
            self._uep_view.set_parameter_species_options(())
            return

        source_structure = self._selected_uep_source_structure()
        self._uep_view.set_parameter_species_options(_structure_species_options(source_structure))
        self._uep_view.set_project_defaults(
            cutoff_energy=self._document.project.recommended_cutoff_energy,
            kpoint_grid=self._document.project.recommended_kpoint_grid,
            kpoint_reference_lattice=self._document.project.recommended_kpoint_lattice,
        )
        self._uep_view.set_host_lattice(
            None if source_structure is None else [list(row) for row in source_structure.lattice_vectors]
        )

    def _available_uep_source_structures(self) -> list[Structure]:
        if self._document is None:
            return []
        host = self._project_service.active_host_structure(self._document)
        if host is None:
            return []
        derived = [
            structure
            for structure in self._document.structures.values()
            if structure.origin == "pristine_relaxation" and structure.parent_structure_id == host.id
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
            self._uep_view.set_source_structure_options([], selected_id=None, summary="Create or open a project first.")
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
        self._uep_view.set_parameter_species_options(_structure_species_options(selected))

    def _refresh_uep_supercell_options(self) -> None:
        if self._document is None:
            self._uep_view.set_supercell_options([])
            return
        source_structure = self._selected_uep_source_structure()
        if source_structure is None:
            self._uep_view.set_supercell_options([])
            return
        assets = self._project_service.list_workflow_supercell_assets(
            self._document,
            host_structure_id=source_structure.id,
        )
        self._uep_view.set_supercell_options(
            [
                {
                    "label": asset.label,
                    "matrix": [list(row) for row in asset.matrix],
                }
                for asset in assets
            ]
        )

    def _on_uep_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state["uep_form"] = state
        if getattr(self, "_handling_uep_form_state", False):
            return
        self._handling_uep_form_state = True
        try:
            self._refresh_uep_defaults()
            self._refresh_uep_supercell_options()
        finally:
            self._handling_uep_form_state = False

    def _serialize_uep_config(self, config: UepConfig) -> dict[str, object]:
        return {
            "name_prefix": "uep" if config.name_prefix == "uep_source" else config.name_prefix,
            "rows": [
                {
                    "label": row.label,
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

    def _deserialize_uep_config(self, payload: object) -> UepConfig | None:
        if not isinstance(payload, dict):
            return None
        rows_payload = payload.get("rows", [])
        if not isinstance(rows_payload, list):
            return None

        rows: list[UepSourceRow] = []
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
                        (int(matrix_payload[0][0]), int(matrix_payload[0][1]), int(matrix_payload[0][2])),
                        (int(matrix_payload[1][0]), int(matrix_payload[1][1]), int(matrix_payload[1][2])),
                        (int(matrix_payload[2][0]), int(matrix_payload[2][1]), int(matrix_payload[2][2])),
                    )
                except (TypeError, ValueError, IndexError):
                    matrix = None
            raw_params = entry.get("parameter_values", entry.get("params", {}))
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
                UepSourceRow(
                    label=label,
                    supercell_matrix=matrix,
                    parameter_values=restored_params,
                    added_advanced=added_advanced,
                )
            )

        return UepConfig(
            name_prefix=(
                "uep"
                if str(payload.get("name_prefix", "uep")).strip() == "uep_source"
                else str(payload.get("name_prefix", "uep")).strip() or "uep"
            ),
            rows=tuple(rows),
        )

    @staticmethod
    def _normalized_supercell_matrix(
        matrix: SupercellMatrix | None,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None:
        if matrix is None:
            return None
        return (
            (int(matrix[0][0]), int(matrix[0][1]), int(matrix[0][2])),
            (int(matrix[1][0]), int(matrix[1][1]), int(matrix[1][2])),
            (int(matrix[2][0]), int(matrix[2][1]), int(matrix[2][2])),
        )

    @staticmethod
    def _normalized_parameter_values(values: Mapping[str, object]) -> object:
        return to_jsonable({str(name): value for name, value in values.items()})

    def _pristine_relaxation_row_signature(self, row: PristineRelaxationRow) -> tuple[object, ...]:
        return (
            bool(row.fix_all_cell),
            self._normalized_supercell_matrix(row.supercell_matrix),
            self._normalized_parameter_values(row.parameter_values),
            tuple(str(name) for name in row.added_advanced),
        )

    def _muon_relaxation_row_signature(self, row: MuonRelaxationRow) -> tuple[object, ...]:
        return (
            str(row.site_id),
            row.label,
            tuple(float(value) for value in row.initial_fractional_position),
            str(row.charge_state),
            self._normalized_supercell_matrix(row.supercell_matrix),
            self._normalized_parameter_values(row.parameter_values),
            tuple(str(name) for name in row.added_advanced),
        )

    def _uep_row_signature(self, row: UepSourceRow) -> tuple[object, ...]:
        return (
            self._normalized_supercell_matrix(row.supercell_matrix),
            self._normalized_parameter_values(row.parameter_values),
            tuple(str(name) for name in row.added_advanced),
        )

    def _definition_workflow_context_matches(self, source_definition, target_definition) -> bool:
        return (
            source_definition.workflow_stage == target_definition.workflow_stage
            and source_definition.calc_type == target_definition.calc_type
            and source_definition.input_structure_ids == target_definition.input_structure_ids
            and self._definition_output_directory(source_definition)
            == self._definition_output_directory(target_definition)
        )

    @staticmethod
    def _ensure_label_mapping(definition) -> dict[str, object]:
        mapping = definition.parameters.setdefault("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            mapping = {}
            definition.parameters["job_ids_by_label"] = mapping
        return mapping

    def _sync_equivalent_pristine_relaxation_mappings(self, target_definition) -> None:
        if self._document is None:
            return
        target_config = self._deserialize_pristine_relaxation_config(
            target_definition.parameters.get("configuration", {})
        )
        if target_config is None:
            return

        target_signatures = {
            row.label: self._pristine_relaxation_row_signature(row)
            for row in target_config.rows
        }
        candidates: dict[str, str] = {}
        for definition in self._pristine_relaxation_definitions():
            if not self._definition_workflow_context_matches(definition, target_definition):
                continue
            source_config = self._deserialize_pristine_relaxation_config(
                definition.parameters.get("configuration", {})
            )
            if source_config is None:
                continue
            source_signatures = {
                row.label: self._pristine_relaxation_row_signature(row)
                for row in source_config.rows
            }
            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict):
                continue
            for label, raw_job_id in mapping.items():
                if not isinstance(label, str) or not isinstance(raw_job_id, str):
                    continue
                if target_signatures.get(label) != source_signatures.get(label):
                    continue
                existing = candidates.get(label)
                if (
                    existing is None
                    or self._relaxation_mapping_job_priority(raw_job_id)
                    > self._relaxation_mapping_job_priority(existing)
                ):
                    candidates[label] = raw_job_id

        if not candidates:
            return
        target_mapping = self._ensure_label_mapping(target_definition)
        for label, raw_job_id in candidates.items():
            existing = target_mapping.get(label)
            if (
                not isinstance(existing, str)
                or self._relaxation_mapping_job_priority(raw_job_id)
                > self._relaxation_mapping_job_priority(existing)
            ):
                target_mapping[label] = raw_job_id

    def _sync_equivalent_muon_relaxation_mappings(self, target_definition) -> None:
        if self._document is None:
            return
        target_config = self._deserialize_muon_relaxation_config(
            target_definition.parameters.get("configuration", {})
        )
        if target_config is None:
            return

        target_signatures = {
            f"{muon_charge_state_prefix(row.charge_state)}:{row.label}": self._muon_relaxation_row_signature(row)
            for row in target_config.rows
        }
        candidates: dict[str, str] = {}
        seed_candidates: dict[str, str] = {}
        for definition in self._muon_relaxation_definitions():
            if not self._definition_workflow_context_matches(definition, target_definition):
                continue
            source_config = self._deserialize_muon_relaxation_config(
                definition.parameters.get("configuration", {})
            )
            if source_config is None:
                continue
            source_signatures: dict[str, tuple[object, ...]] = {}
            legacy_key_by_row_label: dict[str, str] = {}
            for row in source_config.rows:
                key = f"{muon_charge_state_prefix(row.charge_state)}:{row.label}"
                source_signatures[key] = self._muon_relaxation_row_signature(row)
                legacy_key_by_row_label[row.label] = key

            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict):
                continue
            seed_mapping = definition.parameters.get("job_seed_names_by_label", {})
            for raw_key, raw_job_id in mapping.items():
                if not isinstance(raw_key, str) or not isinstance(raw_job_id, str):
                    continue
                key = raw_key if ":" in raw_key else legacy_key_by_row_label.get(raw_key, raw_key)
                if target_signatures.get(key) != source_signatures.get(key):
                    continue
                job = self._muon_relaxation_job_for_mapping_value(key, raw_job_id)
                if job is None or job.status == JobStatus.DEFINED:
                    continue
                existing = candidates.get(key)
                if (
                    existing is None
                    or self._relaxation_mapping_job_priority(raw_job_id)
                    > self._relaxation_mapping_job_priority(existing)
                ):
                    candidates[key] = raw_job_id
                    if isinstance(seed_mapping, dict):
                        raw_seed = seed_mapping.get(raw_key, seed_mapping.get(key))
                        if isinstance(raw_seed, str) and raw_seed.strip():
                            seed_candidates[key] = raw_seed.strip()

        if not candidates:
            return
        target_mapping = self._ensure_label_mapping(target_definition)
        target_seed_mapping = target_definition.parameters.setdefault("job_seed_names_by_label", {})
        if not isinstance(target_seed_mapping, dict):
            target_seed_mapping = {}
            target_definition.parameters["job_seed_names_by_label"] = target_seed_mapping
        for key, raw_job_id in candidates.items():
            existing = target_mapping.get(key)
            if (
                not isinstance(existing, str)
                or self._relaxation_mapping_job_priority(raw_job_id)
                > self._relaxation_mapping_job_priority(existing)
            ):
                target_mapping[key] = raw_job_id
                raw_seed = seed_candidates.get(key)
                if raw_seed:
                    target_seed_mapping[key] = raw_seed

    def _sync_active_muon_relaxation_dashboard_jobs(self, target_definition) -> None:
        if self._document is None:
            return
        config = self._deserialize_muon_relaxation_config(
            target_definition.parameters.get("configuration", {})
        )
        if config is None:
            return

        mapping = self._ensure_label_mapping(target_definition)
        for row in config.rows:
            key = self._muon_relaxation_row_key(row)
            existing = mapping.get(key)
            if isinstance(existing, str) and self._muon_relaxation_job_for_mapping_value(key, existing) is not None:
                continue

            expected_calc_type = muon_relaxation_calc_type(row.charge_state)
            candidates = [
                job
                for job in self._document.jobs.values()
                if job.workflow_stage == WorkflowStage.MUON_RELAXATION
                and job.calc_type == expected_calc_type
                and job.label == row.label
                and job.status in _ACTIVE_LOCAL_JOB_STATES
            ]
            if not candidates:
                continue

            job = max(
                candidates,
                key=lambda candidate: (
                    self._relaxation_mapping_job_priority(str(candidate.id)),
                    self._job_submission_sort_key(candidate),
                ),
            )
            mapping[key] = str(job.id)

    def _sync_equivalent_uep_mappings(self, target_definition) -> None:
        if self._document is None:
            return
        target_config = self._deserialize_uep_config(target_definition.parameters.get("configuration", {}))
        if target_config is None:
            return

        target_signatures = {
            row.label: self._uep_row_signature(row)
            for row in target_config.rows
        }
        candidates: dict[str, str] = {}
        for definition in self._uep_definitions():
            if not self._definition_workflow_context_matches(definition, target_definition):
                continue
            source_config = self._deserialize_uep_config(definition.parameters.get("configuration", {}))
            if source_config is None:
                continue
            source_signatures = {
                row.label: self._uep_row_signature(row)
                for row in source_config.rows
            }
            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict):
                continue
            for label, raw_job_id in mapping.items():
                if not isinstance(label, str) or not isinstance(raw_job_id, str):
                    continue
                if target_signatures.get(label) != source_signatures.get(label):
                    continue
                existing = candidates.get(label)
                if (
                    existing is None
                    or self._convergence_mapping_job_priority(raw_job_id)
                    > self._convergence_mapping_job_priority(existing)
                ):
                    candidates[label] = raw_job_id

        if not candidates:
            return
        target_mapping = self._ensure_label_mapping(target_definition)
        for label, raw_job_id in candidates.items():
            existing = target_mapping.get(label)
            if (
                not isinstance(existing, str)
                or self._convergence_mapping_job_priority(raw_job_id)
                > self._convergence_mapping_job_priority(existing)
            ):
                target_mapping[label] = raw_job_id

    def _uep_definitions(self) -> list:
        if self._document is None:
            return []
        definitions = [
            definition
            for definition in self._document.calculations.values()
            if definition.workflow_stage == WorkflowStage.UEP
        ]
        definitions.sort(key=lambda definition: definition.created_at, reverse=True)
        return definitions

    def _active_uep_definition(self):
        if self._document is None:
            return None
        raw_definition_id = self._document.ui_state.get(self._ACTIVE_UEP_DEFINITION_KEY)
        if isinstance(raw_definition_id, str):
            try:
                definition_id = UUID(raw_definition_id)
            except ValueError:
                definition_id = None
            if definition_id is not None:
                definition = self._document.calculations.get(definition_id)
                if definition is not None and definition.workflow_stage == WorkflowStage.UEP:
                    return definition
        definitions = self._uep_definitions()
        return definitions[0] if definitions else None

    def _job_for_uep_label(self, definition, label: str) -> Job | None:
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
                    return self._document.jobs.get(job_id)
        return None

    def _uep_source_structure_id_for_job(self, job_id: UUID) -> UUID | None:
        if self._document is None:
            return None
        for run in self._document.runs.values():
            if run.job_id != job_id or run.definition_id is None:
                continue
            definition = self._document.calculations.get(run.definition_id)
            if definition is not None and definition.input_structure_ids:
                return definition.input_structure_ids[0]
        for definition in self._uep_definitions():
            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict):
                continue
            if str(job_id) in {str(value) for value in mapping.values()} and definition.input_structure_ids:
                return definition.input_structure_ids[0]
        return None

    def _active_uep_job_ids(self) -> set[UUID]:
        definition = self._active_uep_definition()
        if definition is None:
            return set()
        self._sync_equivalent_uep_mappings(definition)
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
                "clustering_hier_t": batch_summary.get("clustering_hier_t"),
                "clustering_kmeans_k": batch_summary.get("clustering_kmeans_k"),
                "clustering_linkage_method": batch_summary.get("clustering_linkage_method"),
                "feature_mode": batch_summary.get("feature_mode"),
                "threshold_ang": batch_summary.get("threshold_ang"),
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
                    fractional = result.get("optimized_fractional_position")
                    cartesian = result.get("optimized_position_ang")
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
                        "source_structure_id": "" if source_structure_id is None else str(source_structure_id),
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
                        "label": f"UEP Cluster {cluster_index}",
                        "properties": properties,
                    }

                representative_row = _representative_row(representative_index)
                if representative_row is not None:
                    rows.append(representative_row)

        rows.sort(
            key=lambda row: (
                str(row.get("source_job_label", "")),
                int(row.get("cluster_index", 0)),
                str(row.get("source_key", "")),
            )
        )
        return rows

    def _build_uep_states(self) -> dict[str, UepRunState]:
        states: dict[str, UepRunState] = {}
        if self._document is None:
            return states
        definition = self._active_uep_definition()
        if definition is None:
            return states
        self._sync_equivalent_uep_mappings(definition)
        config = self._deserialize_uep_config(definition.parameters.get("configuration", {}))
        if config is None:
            return states

        for row in config.rows:
            job = self._job_for_uep_label(definition, row.label)
            if job is None:
                states[row.label] = UepRunState(status="ready")
                continue
            run = self._run_for_job(job.id)
            result = self._document.results.get(run.result_id) if run is not None and run.result_id is not None else None
            data = result.data if result is not None and isinstance(result.data, dict) else {}
            potential_grid_available = self._uep_potential_grid_result_for_job(job.id) is not None
            states[row.label] = UepRunState(
                status=("ready" if job.status == JobStatus.DEFINED else job.status.value),
                total_energy=float(data["total_energy"]) if isinstance(data.get("total_energy"), (int, float)) else None,
                potential_grid_available=potential_grid_available,
                job_id=job.id,
                job_label=job.label,
                error_message=job.error_message,
                density_available=bool(data.get("density_available")) if data.get("density_available") is not None else None,
            )
        return states

    def _refresh_uep_rows(self) -> None:
        self._uep_view.set_runtime_states(self._build_uep_states())

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

    def _on_remove_pristine_relaxation_rows(self, payload: object) -> None:
        labels = self._removed_row_labels(payload)
        if self._document is None or not labels:
            return
        definitions = self._pristine_relaxation_definitions()
        should_autosave = self._relaxation_removal_touches_project_files(
            definitions,
            lambda key: str(key) in labels,
        )
        if not self._cleanup_pristine_relaxation_rows(definitions, labels):
            self._restore_pristine_relaxation_form_from_definition()
            return
        for definition in definitions:
            self._remove_pristine_relaxation_labels_from_definition(definition, labels)
        self._refresh_pristine_relaxation_rows()
        if should_autosave:
            self._persist_document_ui_state(
                failure_message="Pristine relaxation rows were removed, but autosave failed."
            )

    def _remove_pristine_relaxation_labels_from_definition(self, definition, labels: set[str]) -> None:
        config = self._deserialize_pristine_relaxation_config(definition.parameters.get("configuration", {}))
        if config is not None:
            rows = tuple(row for row in config.rows if row.label not in labels)
            if len(rows) != len(config.rows):
                definition.parameters["configuration"] = self._serialize_pristine_relaxation_config(
                    replace(config, rows=rows)
                )

        mapping = definition.parameters.get("job_ids_by_label", {})
        if isinstance(mapping, dict):
            for label in labels:
                mapping.pop(label, None)

        self._remove_definition_if_no_rows(definition, self._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY)

    def _on_remove_muon_relaxation_rows(self, payload: object) -> None:
        labels = self._removed_row_labels(payload)
        if self._document is None or not labels:
            return
        definitions = self._muon_relaxation_definitions()
        key_matches = self._muon_relaxation_label_matcher(labels)
        removed_site_ids = self._muon_relaxation_site_ids_for_removed_rows(definitions, labels)
        should_autosave = self._relaxation_removal_touches_project_files(definitions, key_matches)
        if not self._cleanup_muon_relaxation_rows(definitions, labels):
            self._restore_muon_relaxation_form_from_definition()
            return
        for definition in definitions:
            self._remove_muon_relaxation_labels_from_definition(definition, labels)
        self._remove_muon_relaxation_labels_from_ui_state(labels)
        site_status_changed = self._restore_removed_muon_relaxation_sites(removed_site_ids)
        self._refresh_muon_relaxation_rows()
        if should_autosave or self._site_status_revert_touches_project_files(removed_site_ids, site_status_changed):
            self._persist_document_ui_state(
                failure_message="Muon relaxation rows were removed, but autosave failed."
            )

    def _cleanup_pristine_relaxation_rows(self, definitions: list, labels: set[str]) -> bool:
        return self._cleanup_relaxation_row_jobs(
            definitions,
            lambda key: str(key) in labels,
            title="Remove relaxation job with files",
            text_for_job=lambda job: f"Relaxation job '{job.label}' has associated files.",
            keep_text="or Cancel to keep the relaxation row.",
        )

    def _cleanup_muon_relaxation_rows(self, definitions: list, labels: set[str]) -> bool:
        return self._cleanup_relaxation_row_jobs(
            definitions,
            self._muon_relaxation_label_matcher(labels),
            title="Remove relaxation job with files",
            text_for_job=lambda job: f"Relaxation job '{job.label}' has associated files.",
            keep_text="or Cancel to keep the relaxation row.",
        )

    def _muon_relaxation_label_matcher(self, labels: set[str]) -> Callable[[object], bool]:
        unqualified_labels = {label for label in labels if ":" not in label}
        full_key_labels = {label.partition(":")[2] for label in labels if ":" in label}

        def matches(key: object) -> bool:
            if not isinstance(key, str):
                return False
            return (
                key in labels
                or (":" in key and key.partition(":")[2] in unqualified_labels)
                or (":" not in key and key in full_key_labels)
            )

        return matches

    def _muon_relaxation_site_ids_for_removed_rows(
        self,
        definitions: list,
        labels: set[str],
    ) -> set[UUID]:
        if self._document is None:
            return set()

        key_matches = self._muon_relaxation_label_matcher(labels)
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
            if site is None or site.status != "staged" or self._site_has_muon_relaxation_rows(site_id):
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

        for definition in self._muon_relaxation_definitions():
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
        key_matches = self._muon_relaxation_label_matcher(labels)
        kept: list[object] = []
        for row in rows:
            if not isinstance(row, dict):
                kept.append(row)
                continue
            label = str(row.get("label", "")).strip()
            charge_state = str(row.get("charge_state", "")).strip()
            keys = (
                [f"{muon_charge_state_prefix(charge_state)}:{label}"]
                if charge_state
                else [label]
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

    def _cleanup_relaxation_row_jobs(
        self,
        definitions: list,
        key_matches: Callable[[object], bool],
        *,
        title: str,
        text_for_job: Callable[[Job], str],
        keep_text: str,
    ) -> bool:
        if self._document is None:
            return True

        jobs: list[Job] = []
        seen: set[UUID] = set()
        for definition in definitions:
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
                if job is None or job.id in seen:
                    continue
                jobs.append(job)
                seen.add(job.id)

        shared_file_disposition: str | None = None
        multi = len(jobs) > 1
        for job in jobs:
            job_dir = self._job_output_directory(job)
            if job_dir is not None and job_dir.exists():
                disposition = shared_file_disposition
                if disposition is None:
                    disposition = self._prompt_job_file_disposition(
                        title=title,
                        text=f"{text_for_job(job)}\n\n{job_dir}",
                        keep_text=keep_text,
                        include_apply_all=multi,
                    )
                if disposition is None:
                    return False
                try:
                    if disposition in {"delete", "delete_all"}:
                        self._delete_path(job_dir)
                    else:
                        output_root = self._relaxation_archive_root_for_job(job, job_dir)
                        self._archive_cutoff_directory(job_dir, output_root)
                except OSError as exc:
                    QMessageBox.warning(
                        self,
                        "File cleanup failed",
                        f"Could not update files for job '{job.label}':\n{exc}",
                    )
                    return False

                if disposition in {"delete_all", "archive_all"}:
                    shared_file_disposition = disposition

            self._delete_job_records(job.id)

        return True

    def _relaxation_archive_root_for_job(self, job: Job, job_dir: Path) -> Path | None:
        if self._document is None:
            return None

        for definition in self._document.calculations.values():
            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict) or str(job.id) not in {str(value) for value in mapping.values()}:
                continue
            base = self._definition_output_directory(definition)
            if base is None:
                continue
            if definition.workflow_stage == WorkflowStage.MUON_RELAXATION and job_dir.parent.name in {
                "positivemuon",
                "muonium",
            }:
                return job_dir.parent
            return base
        return job_dir.parent

    def _restore_pristine_relaxation_form_from_definition(self) -> None:
        definition = self._active_pristine_relaxation_definition()
        if definition is None:
            return
        config = self._deserialize_pristine_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return
        state = self._serialize_pristine_relaxation_config(config)
        for row in state.get("rows", []):
            if isinstance(row, dict) and "params" not in row:
                row["params"] = dict(row.get("parameter_values", {}))
        self._pristine_relaxation_view.set_form_state(state)
        if self._document is not None:
            self._document.ui_state["pristine_relaxation_form"] = state
        self._refresh_pristine_relaxation_rows()

    def _restore_muon_relaxation_form_from_definition(self) -> None:
        definition = self._active_muon_relaxation_definition()
        if definition is None:
            return
        config = self._deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return
        state = self._serialize_muon_relaxation_config(config)
        for row in state.get("rows", []):
            if isinstance(row, dict) and "fractional_position" not in row:
                row["fractional_position"] = list(row.get("initial_fractional_position", (0.0, 0.0, 0.0)))
        self._muon_relaxation_view.set_form_state(state)
        if self._document is not None:
            self._document.ui_state["muon_relaxation_form"] = state
        self._refresh_muon_relaxation_rows()

    def _remove_muon_relaxation_labels_from_definition(self, definition, labels: set[str]) -> None:
        unqualified_labels = {label for label in labels if ":" not in label}
        full_key_labels = {label.partition(":")[2] for label in labels if ":" in label}

        def row_matches(row: MuonRelaxationRow) -> bool:
            key = f"{muon_charge_state_prefix(row.charge_state)}:{row.label}"
            return key in labels or row.label in unqualified_labels

        config = self._deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is not None:
            rows = tuple(row for row in config.rows if not row_matches(row))
            if len(rows) != len(config.rows):
                definition.parameters["configuration"] = self._serialize_muon_relaxation_config(
                    replace(config, rows=rows)
                )

        for mapping_name in ("job_ids_by_label", "job_seed_names_by_label"):
            mapping = definition.parameters.get(mapping_name, {})
            if not isinstance(mapping, dict):
                continue
            stale_keys = [
                key
                for key in mapping
                if isinstance(key, str)
                and (
                    key in labels
                    or (":" in key and key.partition(":")[2] in unqualified_labels)
                    or (":" not in key and key in full_key_labels)
                )
            ]
            for key in stale_keys:
                mapping.pop(key, None)

        self._remove_definition_if_no_rows(definition, self._ACTIVE_MUON_RELAXATION_DEFINITION_KEY)

    def _on_remove_uep_rows(self, payload: object) -> None:
        labels = self._removed_row_labels(payload)
        if self._document is None or not labels:
            return
        for definition in self._uep_definitions():
            self._remove_uep_labels_from_definition(definition, labels)
        self._refresh_uep_rows()

    def _remove_uep_labels_from_definition(self, definition, labels: set[str]) -> None:
        config = self._deserialize_uep_config(definition.parameters.get("configuration", {}))
        if config is not None:
            rows = tuple(row for row in config.rows if row.label not in labels)
            if len(rows) != len(config.rows):
                definition.parameters["configuration"] = self._serialize_uep_config(
                    replace(config, rows=rows)
                )

        mapping = definition.parameters.get("job_ids_by_label", {})
        if isinstance(mapping, dict):
            for label in labels:
                mapping.pop(label, None)

        self._remove_definition_if_no_rows(definition, self._ACTIVE_UEP_DEFINITION_KEY)

    def _remove_definition_if_no_rows(self, definition, active_key: str) -> None:
        if self._document is None:
            return
        configuration = definition.parameters.get("configuration")
        if not isinstance(configuration, dict):
            return
        rows = configuration.get("rows")
        if not isinstance(rows, list):
            return
        if rows:
            return
        self._document.calculations.pop(definition.id, None)
        if self._document.ui_state.get(active_key) == str(definition.id):
            self._document.ui_state.pop(active_key, None)

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

            raw_params = entry.get("parameter_values", entry.get("params", {}))
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

    def _serialize_muon_relaxation_config(self, config: MuonRelaxationConfig) -> dict[str, object]:
        return {
            "rows": [
                {
                    "site_id": str(row.site_id),
                    "label": row.label,
                    "initial_fractional_position": list(row.initial_fractional_position),
                    "charge_state": row.charge_state,
                    "supercell_matrix": (
                        [list(line) for line in row.supercell_matrix]
                        if row.supercell_matrix is not None
                        else None
                    ),
                    "parameter_values": dict(row.parameter_values),
                    "added_advanced": list(row.added_advanced),
                }
                for row in config.rows
            ]
        }

    def _deserialize_muon_relaxation_config(self, payload: object) -> MuonRelaxationConfig | None:
        if not isinstance(payload, dict):
            return None
        rows_payload = payload.get("rows", [])
        if not isinstance(rows_payload, list):
            return None
        rows: list[MuonRelaxationRow] = []
        for entry in rows_payload:
            if not isinstance(entry, dict):
                continue
            try:
                site_id = UUID(str(entry.get("site_id")))
            except (TypeError, ValueError):
                continue
            label = str(entry.get("label", "")).strip()
            if not label:
                continue
            raw_position = entry.get("initial_fractional_position", entry.get("fractional_position", []))
            if not isinstance(raw_position, (list, tuple)) or len(raw_position) != 3:
                continue
            matrix_payload = entry.get("supercell_matrix")
            matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None = None
            if isinstance(matrix_payload, list) and len(matrix_payload) == 3:
                try:
                    matrix = (
                        (int(matrix_payload[0][0]), int(matrix_payload[0][1]), int(matrix_payload[0][2])),
                        (int(matrix_payload[1][0]), int(matrix_payload[1][1]), int(matrix_payload[1][2])),
                        (int(matrix_payload[2][0]), int(matrix_payload[2][1]), int(matrix_payload[2][2])),
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
                MuonRelaxationRow(
                    site_id=site_id,
                    label=label,
                    initial_fractional_position=(
                        float(raw_position[0]),
                        float(raw_position[1]),
                        float(raw_position[2]),
                    ),
                    charge_state=str(entry.get("charge_state", "positive_muon")),
                    supercell_matrix=matrix,
                    parameter_values=restored_params,
                    added_advanced=added_advanced,
                )
            )
        return MuonRelaxationConfig(rows=tuple(rows))

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

        self._sync_equivalent_pristine_relaxation_mappings(definition)
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

    def _pristine_relaxation_form_config(self) -> PristineRelaxationConfig | None:
        if self._document is None:
            return None

        form_sources: list[dict] = []
        widget_state = self._pristine_relaxation_view.collect_state()
        if isinstance(widget_state, dict):
            form_sources.append(widget_state)

        persisted = self._document.ui_state.get("pristine_relaxation_form", {})
        if isinstance(persisted, dict):
            form_sources.append(persisted)

        for form_state in form_sources:
            config = self._deserialize_pristine_relaxation_config(form_state)
            if config is not None and config.rows:
                return config
        return None

    def _build_pristine_relaxation_states(self) -> dict[str, RelaxationRunState]:
        states: dict[str, RelaxationRunState] = {}
        if self._document is None:
            return states

        form_config = self._pristine_relaxation_form_config()
        if form_config is not None:
            for row in form_config.rows:
                states[row.label] = RelaxationRunState(status="ready")

        definition = self._active_pristine_relaxation_definition()
        if definition is None:
            return states
        self._sync_equivalent_pristine_relaxation_mappings(definition)
        config = self._deserialize_pristine_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return states

        if form_config is not None:
            definition_rows_by_label = {row.label: row for row in config.rows}
            rows_for_status = [
                definition_row
                for row in form_config.rows
                if (definition_row := definition_rows_by_label.get(row.label)) is not None
                and self._pristine_relaxation_row_signature(row)
                == self._pristine_relaxation_row_signature(definition_row)
            ]
        else:
            rows_for_status = list(config.rows)

        for row in rows_for_status:
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

    def _muon_relaxation_definitions(self) -> list:
        if self._document is None:
            return []
        definitions = [
            definition
            for definition in self._document.calculations.values()
            if definition.workflow_stage == WorkflowStage.MUON_RELAXATION
            and definition.calc_type == "castep_muon_relaxation"
        ]
        definitions.sort(key=lambda definition: definition.created_at, reverse=True)
        return definitions

    def _active_muon_relaxation_definition(self):
        if self._document is None:
            return None
        raw_definition_id = self._document.ui_state.get(self._ACTIVE_MUON_RELAXATION_DEFINITION_KEY)
        if isinstance(raw_definition_id, str):
            try:
                definition_id = UUID(raw_definition_id)
            except ValueError:
                definition_id = None
            if definition_id is not None:
                definition = self._document.calculations.get(definition_id)
                if definition is not None and definition.workflow_stage == WorkflowStage.MUON_RELAXATION:
                    return definition
        definitions = self._muon_relaxation_definitions()
        return definitions[0] if definitions else None

    def _muon_relaxation_job_for_mapping_value(self, key: str, raw_job_id: str) -> Job | None:
        if self._document is None:
            return None
        try:
            job_id = UUID(str(raw_job_id))
        except ValueError:
            return None
        job = self._document.jobs.get(job_id)
        if job is None or not is_muon_relaxation_calc_type(job.calc_type):
            return None
        prefix, separator, _label = str(key).partition(":")
        if not separator:
            return job
        if prefix == muon_charge_state_prefix(POSITIVE_MUON_CHARGE_STATE):
            expected = muon_relaxation_calc_type(POSITIVE_MUON_CHARGE_STATE)
        elif prefix == muon_charge_state_prefix(MUONIUM_CHARGE_STATE):
            expected = muon_relaxation_calc_type(MUONIUM_CHARGE_STATE)
        else:
            return None
        return job if job.calc_type == expected else None

    def _job_for_muon_relaxation_label(self, definition, label: str, charge_state: str | None = None) -> Job | None:
        if self._document is None:
            return None
        keys = [label]
        if charge_state:
            keys.insert(0, f"{muon_charge_state_prefix(charge_state)}:{label}")
        mapping = definition.parameters.get("job_ids_by_label", {})
        if isinstance(mapping, dict):
            for key in keys:
                raw_job_id = mapping.get(key)
                if isinstance(raw_job_id, str):
                    job = self._muon_relaxation_job_for_mapping_value(key, raw_job_id)
                    if job is not None:
                        return job
        return None

    @staticmethod
    def _muon_relaxation_selection_matches(row: MuonRelaxationRow, selection: str) -> bool:
        selection = str(selection)
        prefix, separator, label = selection.partition(":")
        if separator:
            return prefix == muon_charge_state_prefix(row.charge_state) and label == row.label
        return selection == row.label

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
            config = self._deserialize_muon_relaxation_config(form_state)
            if config is not None and config.rows:
                return config
        return None

    @staticmethod
    def _muon_relaxation_row_key(row: MuonRelaxationRow) -> str:
        return f"{muon_charge_state_prefix(row.charge_state)}:{row.label}"

    def _muon_relaxation_row_for_job(self, job: Job) -> MuonRelaxationRow | None:
        _definition, row = self._muon_relaxation_definition_and_row_for_job(job)
        return row

    def _muon_relaxation_definition_and_row_for_job(self, job: Job):
        if self._document is None:
            return None, None
        for definition in self._muon_relaxation_definitions():
            config = self._deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
            if config is None:
                continue
            for row in config.rows:
                candidate = self._job_for_muon_relaxation_label(definition, row.label, row.charge_state)
                if candidate is not None and candidate.id == job.id:
                    return definition, row
        return None, None

    def _site_for_muon_relaxation_job(self, job: Job) -> MuonSite | None:
        if self._document is None:
            return None
        row = self._muon_relaxation_row_for_job(job)
        if row is None:
            return None
        return self._document.sites.get(row.site_id)

    @staticmethod
    def _muon_fractional_position_from_cell(cell: object) -> tuple[float, float, float] | None:
        species = getattr(cell, "species", ())
        positions = getattr(cell, "fractional_positions", ())
        for atom_species, position in reversed(list(zip(species, positions, strict=False))):
            if str(atom_species).strip().casefold() != "h:mu":
                continue
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                return None
            return (float(position[0]), float(position[1]), float(position[2]))
        return None

    @staticmethod
    def _fold_supercell_fractional_to_host(
        position: tuple[float, float, float],
        matrix: SupercellMatrix | None,
    ) -> tuple[float, float, float]:
        if matrix is None:
            return tuple(float(component) % 1.0 for component in position)
        x, y, z = position
        host_position = (
            x * matrix[0][0] + y * matrix[1][0] + z * matrix[2][0],
            x * matrix[0][1] + y * matrix[1][1] + z * matrix[2][1],
            x * matrix[0][2] + y * matrix[1][2] + z * matrix[2][2],
        )
        return tuple(float(component) % 1.0 for component in host_position)

    @staticmethod
    def _coerce_fractional_triplet(value: object) -> tuple[float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None

    def _build_muon_relaxation_states(self) -> dict[str, RelaxationRunState]:
        states: dict[str, RelaxationRunState] = {}
        if self._document is None:
            return states
        form_config = self._muon_relaxation_form_config()
        if form_config is not None:
            for row in form_config.rows:
                states[self._muon_relaxation_row_key(row)] = RelaxationRunState(status="ready")

        definition = self._active_muon_relaxation_definition()
        if definition is None:
            return states
        self._sync_equivalent_muon_relaxation_mappings(definition)
        self._sync_active_muon_relaxation_dashboard_jobs(definition)
        config = self._deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return states

        if form_config is not None:
            definition_rows_by_key = {
                self._muon_relaxation_row_key(row): row
                for row in config.rows
            }
            rows_for_status = []
            for row in form_config.rows:
                key = self._muon_relaxation_row_key(row)
                definition_row = definition_rows_by_key.get(key)
                if definition_row is None:
                    continue
                if self._muon_relaxation_row_signature(row) == self._muon_relaxation_row_signature(definition_row):
                    rows_for_status.append(definition_row)
                    continue
                job = self._job_for_muon_relaxation_label(
                    definition,
                    definition_row.label,
                    definition_row.charge_state,
                )
                if job is not None and job.status in _ACTIVE_LOCAL_JOB_STATES:
                    rows_for_status.append(definition_row)
        else:
            rows_for_status = list(config.rows)

        for row in rows_for_status:
            state_key = self._muon_relaxation_row_key(row)
            job = self._job_for_muon_relaxation_label(definition, row.label, row.charge_state)
            if job is None:
                states[state_key] = RelaxationRunState(status="ready")
                continue
            status_value = job.status.value
            run = self._run_for_job(job.id)
            if status_value == JobStatus.COMPLETED.value and run is not None and run.convergence_achieved is False:
                status_value = "not converged"
            lattice_a = lattice_b = lattice_c = None
            final_fractional_position = None
            if run is not None and run.result_id is not None:
                result = self._document.results.get(run.result_id)
                if result is not None and isinstance(result.data, dict):
                    lattice_a = result.data.get("lattice_a")
                    lattice_b = result.data.get("lattice_b")
                    lattice_c = result.data.get("lattice_c")
                    final_fractional_position = self._coerce_fractional_triplet(
                        result.data.get("muon_final_fractional_position_host")
                    )
            states[state_key] = RelaxationRunState(
                status=status_value,
                total_energy=self._total_energy_for_job(job.id),
                max_force=self._max_force_for_job(job.id),
                lattice_a=float(lattice_a) if isinstance(lattice_a, (int, float)) else None,
                lattice_b=float(lattice_b) if isinstance(lattice_b, (int, float)) else None,
                lattice_c=float(lattice_c) if isinstance(lattice_c, (int, float)) else None,
                final_fractional_position=final_fractional_position,
                job_id=job.id,
                job_label=job.label,
                geom_converged=(run.convergence_achieved if run is not None else None),
                error_message=job.error_message,
            )
        return states

    def _refresh_muon_relaxation_rows(self) -> None:
        self._muon_relaxation_view.set_runtime_states(self._build_muon_relaxation_states())
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
        self._document.ui_state[_RELAXATION_ANALYSIS_STATE_KEY] = [dict(workspace) for workspace in workspaces]

    def _completed_muon_relaxation_sources(self) -> list[CompletedRelaxationAnalysisSource]:
        if self._document is None:
            return []
        host = self._project_service.active_host_structure(self._document)
        if host is None:
            return []

        sources: list[CompletedRelaxationAnalysisSource] = []
        seen: set[UUID] = set()
        for definition in self._muon_relaxation_definitions():
            if host.id not in definition.input_structure_ids:
                continue
            config = self._deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
            if config is None:
                continue
            for row in config.rows:
                job = self._job_for_muon_relaxation_label(definition, row.label, row.charge_state)
                if job is None or job.status != JobStatus.COMPLETED or job.id in seen:
                    continue
                run = self._run_for_job(job.id)
                if run is None or run.result_id is None or run.convergence_achieved is False:
                    continue
                result = self._document.results.get(run.result_id)
                if result is None or not result.derived_structure_ids:
                    continue
                structure_id = result.derived_structure_ids[0]
                structure = self._document.structures.get(structure_id)
                if structure is None:
                    continue
                fractional = None
                if isinstance(result.data, dict):
                    fractional = self._coerce_fractional_triplet(result.data.get("muon_final_fractional_position_host"))
                if fractional is None:
                    fractional = folded_muon_position_from_structure(structure)
                if fractional is None:
                    continue
                total_energy = self._total_energy_for_job(job.id)
                sources.append(
                    CompletedRelaxationAnalysisSource(
                        job_id=job.id,
                        job_label=row.label,
                        charge_state=row.charge_state,
                        structure_id=structure_id,
                        site_id=row.site_id,
                        folded_fractional_position=fractional,
                        total_energy_ev=total_energy,
                    )
                )
                seen.add(job.id)
        return sorted(
            sources,
            key=lambda source: (
                float("inf") if source.total_energy_ev is None else float(source.total_energy_ev),
                source.job_label.casefold(),
            ),
        )

    def _completed_muon_relaxation_sources_for_labels(
        self,
        labels: list[str],
    ) -> list[CompletedRelaxationAnalysisSource]:
        wanted = {str(label) for label in labels}
        selected: list[CompletedRelaxationAnalysisSource] = []
        for source in self._completed_muon_relaxation_sources():
            key = f"{muon_charge_state_prefix(source.charge_state)}:{source.job_label}"
            if key in wanted or source.job_label in wanted:
                selected.append(source)
        return selected

    def _comparison_structure_options_for_relaxation_analysis(self) -> list[tuple[UUID, str, bool]]:
        if self._document is None:
            return []

        options: list[tuple[UUID, str, bool]] = []
        host = self._project_service.active_host_structure(self._document)
        if host is not None:
            options.append((host.id, f"Host structure - {host.label}", False))

        definition = self._active_pristine_relaxation_definition()
        fix_all_cell_by_label: dict[str, bool] = {}
        if definition is not None:
            config = self._deserialize_pristine_relaxation_config(definition.parameters.get("configuration", {}))
            if config is not None:
                fix_all_cell_by_label = {row.label: bool(row.fix_all_cell) for row in config.rows}

        for job in self._completed_pristine_relaxation_jobs():
            fixed_cell = fix_all_cell_by_label.get(job.job_label, True)
            label = f"Pristine relaxed - {job.job_label}"
            if not fixed_cell:
                label = f"{label} (free cell; unavailable)"
            options.append((job.structure_id, label, not fixed_cell))
        return options

    def _refresh_relaxation_analysis_context(self) -> None:
        if self._document is None:
            self._relaxation_analysis_view.set_context(
                workspaces=[],
                sources=[],
                structures={},
                host_structure=None,
                comparison_options=[],
            )
            return
        self._relaxation_analysis_view.set_context(
            workspaces=self._relaxation_analysis_workspaces(),
            sources=self._completed_muon_relaxation_sources(),
            structures=self._document.structures,
            host_structure=self._project_service.active_host_structure(self._document),
            comparison_options=self._comparison_structure_options_for_relaxation_analysis(),
        )

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

        dialog = self._open_supercell_editor(
            source_structure=host,
            workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
            initial_matrix=initial_matrix,
            select_mode=True,
        )
        if dialog is None:
            return

        matrix_values = dialog.selected_matrix()
        if matrix_values is None:
            return
        matrix = (
            (matrix_values[0][0], matrix_values[0][1], matrix_values[0][2]),
            (matrix_values[1][0], matrix_values[1][1], matrix_values[1][2]),
            (matrix_values[2][0], matrix_values[2][1], matrix_values[2][2]),
        )

        self._refresh_pristine_relaxation_supercell_options()
        self._pristine_relaxation_view.set_row_supercell(row_id, matrix)
        self.statusBar().showMessage("Selected pristine-relaxation simulation cell")

    def _on_muon_relaxation_supercell_edit_requested(self, payload: object) -> None:
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
                "Load a host structure before configuring muon-relaxation supercells.",
            )
            return
        existing = self._muon_relaxation_view.row_supercell_matrix(row_id)
        initial_matrix = (
            [list(line) for line in existing]
            if existing is not None
            else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        dialog = self._open_supercell_editor(
            source_structure=host,
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            initial_matrix=initial_matrix,
            select_mode=True,
        )
        if dialog is None:
            return
        matrix_values = dialog.selected_matrix()
        if matrix_values is None:
            return
        matrix = (
            (matrix_values[0][0], matrix_values[0][1], matrix_values[0][2]),
            (matrix_values[1][0], matrix_values[1][1], matrix_values[1][2]),
            (matrix_values[2][0], matrix_values[2][1], matrix_values[2][2]),
        )
        self._refresh_muon_relaxation_supercell_options()
        self._muon_relaxation_view.set_row_supercell(row_id, matrix)
        self.statusBar().showMessage("Selected muon-relaxation simulation cell")

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
        definition = self._project_service.register_pristine_relaxation_definition(
            self._document,
            input_structure_id=host.id,
            parameters=self._serialize_pristine_relaxation_config(config),
            output_directory=output_root,
        )
        self._sync_equivalent_pristine_relaxation_mappings(definition)
        self._document.ui_state[self._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)

        self._refresh_pristine_relaxation_rows()
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
            expand_muon_relaxation_jobs(
                host_structure=host,
                config=config,
                project_cutoff_energy=self._document.project.recommended_cutoff_energy,
                project_kpoint_grid=self._document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=self._document.project.recommended_kpoint_lattice,
            )
        except (ValueError, StructureBridgeError) as exc:
            QMessageBox.warning(self, "Muon relaxation setup invalid", str(exc))
            return
        output_root = self._document.project_root / "files" / "muon_relaxation"
        definition = self._project_service.register_muon_relaxation_definition(
            self._document,
            input_structure_id=host.id,
            parameters=self._serialize_muon_relaxation_config(config),
            output_directory=output_root,
        )
        self._sync_equivalent_muon_relaxation_mappings(definition)
        self._document.ui_state[self._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
        self._refresh_muon_relaxation_rows()
        self._refresh_journal_view()
        self.statusBar().showMessage(
            f"Configured {len(config.rows)} muon-relaxation jobs; input files will be generated on submission"
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
        self._project_service.record_comparison_table_created(self._document, table=table)

        self._sync_comparison_table_views()
        self._populate_project_tree()
        self._set_active_route(self._comparison_route(table.id))
        self._refresh_journal_view()
        self.statusBar().showMessage(f"Generated comparison table '{table.label}'")

    def _jobs_for_pristine_relaxation_labels(self, labels: list[str]) -> list[Job]:
        if self._document is None:
            return []
        definition = self._active_pristine_relaxation_definition()
        if definition is None:
            return []
        self._sync_equivalent_pristine_relaxation_mappings(definition)

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
        self._sync_equivalent_pristine_relaxation_mappings(definition)

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
            spec = spec_by_label.get(label)
            existing_job = self._job_for_pristine_relaxation_label(definition, label)
            if existing_job is not None:
                if existing_job.status == JobStatus.DEFINED and spec is not None:
                    write_job_directory(spec, output_root)
                created.append(existing_job)
                continue

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

    def _jobs_for_muon_relaxation_labels(self, labels: list[str]) -> list[Job]:
        if self._document is None:
            return []
        definition = self._active_muon_relaxation_definition()
        if definition is None:
            return []
        self._sync_equivalent_muon_relaxation_mappings(definition)
        self._sync_active_muon_relaxation_dashboard_jobs(definition)
        config = self._deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return []
        selected: list[Job] = []
        seen: set[UUID] = set()
        wanted = {str(label) for label in labels}
        for row in config.rows:
            if not any(self._muon_relaxation_selection_matches(row, selection) for selection in wanted):
                continue
            job = self._job_for_muon_relaxation_label(definition, row.label, row.charge_state)
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    def _materialize_muon_relaxation_jobs(self, labels: list[str]) -> list[Job]:
        if self._document is None:
            return []
        definition = self._active_muon_relaxation_definition()
        if definition is None:
            return []
        self._sync_equivalent_muon_relaxation_mappings(definition)
        self._sync_active_muon_relaxation_dashboard_jobs(definition)
        config = self._deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None or not definition.input_structure_ids:
            return []
        host = self._document.structures.get(definition.input_structure_ids[0])
        output_root = self._definition_output_directory(definition)
        if host is None or output_root is None:
            return []
        try:
            expansion = expand_muon_relaxation_jobs(
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
        seed_mapping = definition.parameters.setdefault("job_seed_names_by_label", {})
        if not isinstance(seed_mapping, dict):
            seed_mapping = {}
            definition.parameters["job_seed_names_by_label"] = seed_mapping
        spec_by_key = {
            f"{muon_charge_state_prefix(str(spec.provenance.get('charge_state', '')))}:{spec.provenance.get('label', '')}": spec
            for spec in expansion.jobs
        }
        wanted = {str(label) for label in labels}
        rows_by_label = [
            row
            for row in config.rows
            if any(self._muon_relaxation_selection_matches(row, selection) for selection in wanted)
        ]
        created: list[Job] = []
        for row in rows_by_label:
            key = f"{muon_charge_state_prefix(row.charge_state)}:{row.label}"
            spec = spec_by_key.get(key)
            existing_job = self._job_for_muon_relaxation_label(definition, row.label, row.charge_state)
            if existing_job is not None:
                if existing_job.status == JobStatus.DEFINED and spec is not None:
                    seed_mapping[key] = spec.name
                    write_job_directory(spec, output_root / muon_charge_state_prefix(row.charge_state))
                created.append(existing_job)
                continue
            if spec is None:
                continue
            charge_root = output_root / muon_charge_state_prefix(row.charge_state)
            write_job_directory(spec, charge_root)
            job = self._project_service.add_job(
                self._document,
                label=row.label,
                workflow_stage=spec.workflow_stage,
                calc_type=muon_relaxation_calc_type(row.charge_state),
            )
            mapping[key] = str(job.id)
            seed_mapping[key] = spec.name
            created.append(job)
        jobs = self._dedupe_jobs(created)
        if jobs:
            self._persist_document_ui_state(
                failure_message="Muon relaxation jobs were generated, but autosave failed."
            )
        return jobs

    def _on_submit_local_from_pristine_relaxation(self, labels: list[str]) -> None:
        self._submit_workflow_selection(
            labels,
            normalize=str,
            build_states=self._build_pristine_relaxation_states,
            jobs_for_values=self._jobs_for_pristine_relaxation_labels,
            materialize_values=self._materialize_pristine_relaxation_jobs,
            is_ready=workflow_row_state_is_ready,
            no_ready_message="Only configured pristine-relaxation rows in the 'ready' state can be submitted.",
            submit=self._submit_jobs_locally,
        )

    def _on_submit_cluster_from_pristine_relaxation(self, labels: list[str]) -> None:
        self._submit_workflow_selection(
            labels,
            normalize=str,
            build_states=self._build_pristine_relaxation_states,
            jobs_for_values=self._jobs_for_pristine_relaxation_labels,
            materialize_values=self._materialize_pristine_relaxation_jobs,
            is_ready=workflow_row_state_is_ready,
            no_ready_message="Only configured pristine-relaxation rows in the 'ready' state can be submitted.",
            submit=self._submit_jobs_on_cluster,
        )

    @staticmethod
    def _muon_relaxation_label_is_ready(states: dict[str, RelaxationRunState], label: str) -> bool:
        ready_statuses = {"ready", "defined", "not generated"}
        state = states.get(label)
        if state is not None and state.status in ready_statuses:
            return True
        if ":" in str(label):
            return False
        suffix = f":{label}"
        return any(
            candidate.status in ready_statuses
            for key, candidate in states.items()
            if key.endswith(suffix)
        )

    def _on_submit_local_from_muon_relaxation(self, labels: list[str]) -> None:
        if not self._check_staged_muon_relaxation_host():
            return
        self._submit_workflow_selection(
            labels,
            normalize=str,
            build_states=self._build_muon_relaxation_states,
            jobs_for_values=self._jobs_for_muon_relaxation_labels,
            materialize_values=self._materialize_muon_relaxation_jobs,
            is_ready=self._muon_relaxation_label_is_ready,
            no_ready_message="Only configured muon-relaxation rows in the 'ready' state can be submitted.",
            submit=lambda jobs: self._submit_muon_relaxation_jobs_grouped(jobs, target="local"),
        )

    def _on_submit_cluster_from_muon_relaxation(self, labels: list[str]) -> None:
        if not self._check_staged_muon_relaxation_host():
            return
        self._submit_workflow_selection(
            labels,
            normalize=str,
            build_states=self._build_muon_relaxation_states,
            jobs_for_values=self._jobs_for_muon_relaxation_labels,
            materialize_values=self._materialize_muon_relaxation_jobs,
            is_ready=self._muon_relaxation_label_is_ready,
            no_ready_message="Only configured muon-relaxation rows in the 'ready' state can be submitted.",
            submit=lambda jobs: self._submit_muon_relaxation_jobs_grouped(jobs, target="cluster"),
        )

    def _submit_muon_relaxation_jobs_grouped(self, jobs: list[Job], *, target: str) -> None:
        grouped: dict[str, list[Job]] = {}
        for job in self._dedupe_jobs(jobs):
            grouped.setdefault(job.calc_type, []).append(job)
        for calc_type in (
            MUON_RELAXATION_CALC_TYPES["positive_muon"],
            MUON_RELAXATION_CALC_TYPES["muonium"],
        ):
            group = grouped.get(calc_type, [])
            if not group:
                continue
            if target == "local":
                self._submit_jobs_locally(group)
            else:
                self._submit_jobs_on_cluster(group)

    def _quantum_corrections_definitions(self) -> list[object]:
        if self._document is None:
            return []
        return [
            definition
            for definition in self._document.calculations.values()
            if definition.workflow_stage == WorkflowStage.QUANTUM_CORRECTIONS
            and definition.calc_type == QUANTUM_PHONON_SET_CALC_TYPE
        ]

    def _active_quantum_corrections_definition(self):
        definitions = self._quantum_corrections_definitions()
        return definitions[0] if definitions else None

    @staticmethod
    def _quantum_job_mapping(definition, key: str) -> dict[str, str]:
        if definition is None:
            return {}
        mapping = definition.parameters.get(key, {})
        return dict(mapping) if isinstance(mapping, dict) else {}

    def _quantum_seed_name_for_candidate(self, candidate: _QuantumSourceCandidate) -> str:
        return job_name("quantum", self._quantum_source_label(candidate))

    def _quantum_seed_name_for_key(self, definition, key: str) -> str:
        mapping = self._quantum_job_mapping(definition, "job_seed_names_by_label")
        raw_seed = mapping.get(key)
        if isinstance(raw_seed, str) and raw_seed.strip():
            return raw_seed.strip()
        candidate = self._quantum_source_candidate_by_key(key)
        if candidate is None:
            return "quantum"
        return self._quantum_seed_name_for_candidate(candidate)

    def _job_from_mapping_for_quantum_source_key(
        self,
        definition,
        *,
        mapping_name: str,
        key: str,
        calc_type: str,
    ) -> Job | None:
        if self._document is None or definition is None:
            return None
        mapping = self._quantum_job_mapping(definition, mapping_name)
        raw_job_id = mapping.get(key)
        if not isinstance(raw_job_id, str):
            return None
        try:
            job_id = UUID(raw_job_id)
        except ValueError:
            return None
        job = self._document.jobs.get(job_id)
        if job is None or job.calc_type != calc_type:
            return None
        return job

    def _singlepoint_job_for_quantum_source_key(self, definition, key: str) -> Job | None:
        return self._job_from_mapping_for_quantum_source_key(
            definition,
            mapping_name="singlepoint_job_ids_by_label",
            key=key,
            calc_type=QUANTUM_SINGLEPOINT_CALC_TYPE,
        )

    def _phonon_job_for_quantum_source_key(self, definition, key: str) -> Job | None:
        job = self._job_from_mapping_for_quantum_source_key(
            definition,
            mapping_name="phonon_job_ids_by_label",
            key=key,
            calc_type=QUANTUM_PHONON_CALC_TYPE,
        )
        if job is not None:
            return job
        return self._job_from_mapping_for_quantum_source_key(
            definition,
            mapping_name="job_ids_by_label",
            key=key,
            calc_type=QUANTUM_PHONON_CALC_TYPE,
        )

    def _quantum_stage_state(self, definition, candidate: _QuantumSourceCandidate) -> _QuantumStageState:
        phonon_job = self._phonon_job_for_quantum_source_key(definition, candidate.key)
        if phonon_job is not None:
            return _QuantumStageState(
                stage=_QUANTUM_STAGE_PHONON,
                status=phonon_job.status.value,
                current_job=phonon_job,
            )

        singlepoint_job = self._singlepoint_job_for_quantum_source_key(definition, candidate.key)
        if singlepoint_job is None:
            return _QuantumStageState(stage=_QUANTUM_STAGE_SINGLEPOINT, status="ready")
        if singlepoint_job.status == JobStatus.COMPLETED:
            return _QuantumStageState(stage=_QUANTUM_STAGE_PHONON, status="ready")
        return _QuantumStageState(
            stage=_QUANTUM_STAGE_SINGLEPOINT,
            status=singlepoint_job.status.value,
            current_job=singlepoint_job,
        )

    def _quantum_allows_farms_only_edit(self, definition, candidate: _QuantumSourceCandidate) -> bool:
        stage_state = self._quantum_stage_state(definition, candidate)
        return (
            stage_state.stage == _QUANTUM_STAGE_PHONON
            and stage_state.status == "ready"
            and stage_state.current_job is None
        )

    def _quantum_can_reset_to_stage_one(self, definition, candidate: _QuantumSourceCandidate) -> bool:
        singlepoint_job = self._singlepoint_job_for_quantum_source_key(definition, candidate.key)
        if singlepoint_job is None:
            return False
        phonon_job = self._phonon_job_for_quantum_source_key(definition, candidate.key)
        if phonon_job is not None:
            return phonon_job.status not in _ACTIVE_LOCAL_JOB_STATES
        return singlepoint_job.status == JobStatus.COMPLETED

    def _ensure_quantum_corrections_definition(self, source_keys: list[str]):
        if self._document is None:
            return None
        candidates = [
            candidate
            for key in source_keys
            if (candidate := self._quantum_source_candidate_by_key(key)) is not None
        ]
        if not candidates:
            return None
        definition = self._active_quantum_corrections_definition()
        if definition is not None:
            for candidate in candidates:
                if candidate.structure.id not in definition.input_structure_ids:
                    definition.input_structure_ids.append(candidate.structure.id)
                if candidate.site.id not in definition.input_site_ids:
                    definition.input_site_ids.append(candidate.site.id)
            config = definition.parameters.setdefault("configuration", {})
            if isinstance(config, dict):
                config.setdefault("method", "finite_displacement")
                config.setdefault("pseudopotential_family", "unknown")
                config["sources"] = self._serialize_quantum_sources(definition.input_site_ids)
            definition.parameters.setdefault("singlepoint_job_ids_by_label", {})
            definition.parameters.setdefault("phonon_job_ids_by_label", {})
            definition.parameters.setdefault("job_seed_names_by_label", {})
            return definition

        output_root = self._document.project_root / "files" / "quantum_corrections"
        definition = self._project_service.register_quantum_phonon_definition(
            self._document,
            input_structure_ids=list(dict.fromkeys(candidate.structure.id for candidate in candidates)),
            input_site_ids=list(dict.fromkeys(candidate.site.id for candidate in candidates)),
            parameters={
                "method": "finite_displacement",
                "pseudopotential_family": "unknown",
                "sources": self._serialize_quantum_sources([candidate.site.id for candidate in candidates]),
                "singlepoint_job_ids_by_label": {},
                "phonon_job_ids_by_label": {},
                "job_seed_names_by_label": {},
            },
            output_directory=output_root,
        )
        definition.parameters.setdefault("singlepoint_job_ids_by_label", {})
        definition.parameters.setdefault("phonon_job_ids_by_label", {})
        definition.parameters.setdefault("job_seed_names_by_label", {})
        return definition

    def _serialize_quantum_sources(self, site_ids: list[UUID]) -> list[dict[str, object]]:
        serialized: list[dict[str, object]] = []
        for raw_site_id in site_ids:
            candidate = self._quantum_source_candidate_by_key(str(raw_site_id))
            if candidate is None:
                continue
            serialized.append(
                {
                    "site_id": str(candidate.site.id),
                    "source_relaxation_run_id": str(candidate.run.id),
                    "source_relaxation_job_id": str(candidate.job.id),
                    "relaxed_structure_id": str(candidate.structure.id),
                    "label": self._quantum_source_label(candidate),
                    "charge_state": self._charge_state_for_job(candidate.job),
                }
            )
        return serialized

    @staticmethod
    def _charge_state_for_job(job: Job | None) -> str | None:
        if job is None:
            return None
        if job.calc_type == muon_relaxation_calc_type(POSITIVE_MUON_CHARGE_STATE):
            return POSITIVE_MUON_CHARGE_STATE
        if job.calc_type == muon_relaxation_calc_type(MUONIUM_CHARGE_STATE):
            return MUONIUM_CHARGE_STATE
        return None

    def _job_for_quantum_source_key(self, definition, key: str) -> Job | None:
        candidate = self._quantum_source_candidate_by_key(key)
        if candidate is None:
            return None
        return self._quantum_stage_state(definition, candidate).current_job

    def _quantum_job_by_label(self, label: str) -> Job | None:
        if self._document is None:
            return None
        for job in self._document.jobs.values():
            if job.calc_type in {QUANTUM_SINGLEPOINT_CALC_TYPE, QUANTUM_PHONON_CALC_TYPE} and job.label == label:
                return job
        return None

    def _build_quantum_corrections_states(self) -> dict[str, RelaxationRunState]:
        states: dict[str, RelaxationRunState] = {}
        definition = self._active_quantum_corrections_definition()
        for candidate in self._eligible_quantum_source_candidates():
            stage_state = self._quantum_stage_state(definition, candidate)
            job = stage_state.current_job
            if job is None:
                states[candidate.key] = RelaxationRunState(status=stage_state.status)
                continue
            error_message = job.error_message
            states[candidate.key] = RelaxationRunState(
                status=stage_state.status,
                job_id=job.id,
                job_label=job.label,
                error_message=error_message,
            )
        return states

    def _energy_barriers_definitions(self) -> list[object]:
        if self._document is None:
            return []
        return [
            definition
            for definition in self._document.calculations.values()
            if definition.workflow_stage == WorkflowStage.ENERGY_BARRIERS
            and definition.calc_type == TRANSITION_STATE_SEARCH_SET_CALC_TYPE
        ]

    def _energy_barrier_neb_definitions(self) -> list[object]:
        if self._document is None:
            return []
        return [
            definition
            for definition in self._document.calculations.values()
            if definition.workflow_stage == WorkflowStage.ENERGY_BARRIERS
            and definition.calc_type == NUDGED_ELASTIC_BAND_SET_CALC_TYPE
        ]

    def _active_energy_barriers_definition(self):
        definitions = self._energy_barriers_definitions()
        return definitions[0] if definitions else None

    def _active_energy_barrier_neb_definition(self):
        definitions = self._energy_barrier_neb_definitions()
        return definitions[0] if definitions else None

    def _ensure_energy_barriers_definition(self, source_keys: list[str]):
        if self._document is None:
            return None
        candidates = [
            candidate
            for key in source_keys
            if (candidate := self._energy_barrier_source_candidate_by_key(key)) is not None
        ]
        if not candidates:
            return None
        definition = self._active_energy_barriers_definition()
        if definition is not None:
            for candidate in candidates:
                for structure_id in (candidate.reactant.structure.id, candidate.product.structure.id):
                    if structure_id not in definition.input_structure_ids:
                        definition.input_structure_ids.append(structure_id)
                for site_id in (candidate.reactant.site.id, candidate.product.site.id):
                    if site_id not in definition.input_site_ids:
                        definition.input_site_ids.append(site_id)
            config = definition.parameters.setdefault("configuration", {})
            if isinstance(config, dict):
                config.setdefault("method", "LSTQST")
                config["sources"] = _merge_serialized_sources(
                    config.get("sources", []),
                    self._serialize_energy_barrier_sources(source_keys),
                    key_field="key",
                )
            return definition

        output_root = self._document.project_root / "files" / "energy_barriers"
        structure_ids = list(
            dict.fromkeys(
                structure_id
                for candidate in candidates
                for structure_id in (candidate.reactant.structure.id, candidate.product.structure.id)
            )
        )
        site_ids = list(
            dict.fromkeys(
                site_id
                for candidate in candidates
                for site_id in (candidate.reactant.site.id, candidate.product.site.id)
            )
        )
        return self._project_service.register_transition_state_search_definition(
            self._document,
            input_structure_ids=structure_ids,
            input_site_ids=site_ids,
            parameters={
                "method": "LSTQST",
                "sources": self._serialize_energy_barrier_sources(source_keys),
            },
            output_directory=output_root,
        )

    def _serialize_energy_barrier_sources(self, keys: list[str]) -> list[dict[str, object]]:
        return self._serialize_energy_barrier_source_candidates(
            [
                candidate
                for key in keys
                if (candidate := self._energy_barrier_source_candidate_by_key(key)) is not None
            ]
        )

    def _remove_energy_barrier_sources_from_ui_state(self, keys: set[str]) -> None:
        if self._document is None:
            return
        state = self._document.ui_state.get("energy_barriers_form", {})
        if not isinstance(state, dict):
            return
        raw_sources = state.get("generated_sources", [])
        if isinstance(raw_sources, list):
            state["generated_sources"] = [
                source
                for source in raw_sources
                if not isinstance(source, dict) or str(source.get("key", "")).strip() not in keys
            ]

    def _remove_energy_barrier_sources_from_definition(self, definition, keys: set[str]) -> None:
        config = definition.parameters.get("configuration", {})
        if isinstance(config, dict):
            raw_sources = config.get("sources", [])
            if isinstance(raw_sources, list):
                config["sources"] = [
                    source
                    for source in raw_sources
                    if not isinstance(source, dict) or str(source.get("key", "")).strip() not in keys
                ]
            row_settings = config.get("row_settings", {})
            if isinstance(row_settings, dict):
                for key in keys:
                    row_settings.pop(key, None)
                if not row_settings:
                    config.pop("row_settings", None)
        mapping = definition.parameters.get("job_ids_by_label", {})
        if isinstance(mapping, dict):
            for key in keys:
                mapping.pop(key, None)
        self._remove_energy_barrier_definition_if_no_sources(definition)

    def _remove_energy_barrier_neb_sources_from_definition(self, definition, keys: set[str]) -> None:
        config = definition.parameters.get("configuration", {})
        if isinstance(config, dict):
            raw_sources = config.get("sources", [])
            if isinstance(raw_sources, list):
                config["sources"] = [
                    source
                    for source in raw_sources
                    if not isinstance(source, dict) or str(source.get("key", "")).strip() not in keys
                ]
            row_settings = config.get("row_settings", {})
            if isinstance(row_settings, dict):
                for key in keys:
                    row_settings.pop(key, None)
                if not row_settings:
                    config.pop("row_settings", None)
        mapping = definition.parameters.get("job_ids_by_label", {})
        if isinstance(mapping, dict):
            for key in keys:
                mapping.pop(key, None)
        self._remove_energy_barrier_definition_if_no_sources(definition)

    def _remove_energy_barrier_definition_if_no_sources(self, definition) -> None:
        if self._document is None:
            return
        config = definition.parameters.get("configuration", {})
        if not isinstance(config, dict):
            return
        raw_sources = config.get("sources", [])
        if isinstance(raw_sources, list) and raw_sources:
            return
        self._document.calculations.pop(definition.id, None)

    def _cleanup_energy_barrier_source_jobs(self, keys: set[str]) -> bool:
        if self._document is None:
            return True
        targets: list[tuple[object, Job]] = []
        seen: set[UUID] = set()
        for definitions in (self._energy_barriers_definitions(), self._energy_barrier_neb_definitions()):
            for definition in definitions:
                mapping = definition.parameters.get("job_ids_by_label", {})
                if not isinstance(mapping, dict):
                    continue
                for key in keys:
                    raw_job_id = mapping.get(key)
                    if not isinstance(raw_job_id, str):
                        continue
                    try:
                        job_id = UUID(raw_job_id)
                    except ValueError:
                        continue
                    job = self._document.jobs.get(job_id)
                    if job is None or job.id in seen:
                        continue
                    targets.append((definition, job))
                    seen.add(job.id)

        shared_file_disposition: str | None = None
        multi = len(targets) > 1
        for definition, job in targets:
            job_dir = self._job_output_directory(job)
            if job_dir is not None and job_dir.exists():
                disposition = shared_file_disposition
                if disposition is None:
                    disposition = self._prompt_job_file_disposition(
                        title="Remove energy-barrier path with files",
                        text=f"Energy-barrier job '{job.label}' has associated files.\n\n{job_dir}",
                        keep_text="or Cancel to keep the generated path.",
                        include_apply_all=multi,
                    )
                if disposition is None:
                    return False
                try:
                    if disposition in {"delete", "delete_all"}:
                        self._delete_path(job_dir)
                    else:
                        self._archive_cutoff_directory(job_dir, self._definition_output_directory(definition))
                except OSError as exc:
                    QMessageBox.warning(
                        self,
                        "File cleanup failed",
                        f"Could not update files for job '{job.label}':\n{exc}",
                    )
                    return False
                if disposition in {"delete_all", "archive_all"}:
                    shared_file_disposition = disposition
            self._delete_job_records(job.id)
        return True

    def _energy_barrier_neb_source_candidates(self) -> list[_EnergyBarrierNebCandidate]:
        if self._document is None:
            return []
        candidates: list[_EnergyBarrierNebCandidate] = []
        for definition in self._energy_barriers_definitions():
            config = definition.parameters.get("configuration", {})
            if not isinstance(config, dict):
                continue
            raw_sources = config.get("sources", [])
            if not isinstance(raw_sources, list):
                continue
            source_by_key = {
                str(source.get("key", "")): source
                for source in raw_sources
                if isinstance(source, dict) and str(source.get("key", "")).strip()
            }
            for key, source in source_by_key.items():
                # NEB is seeded directly from the endpoint pair; CASTEP interpolates the band.
                candidates.append(_EnergyBarrierNebCandidate(key=key, source=source))
        return sorted(candidates, key=lambda item: item.key)

    def _energy_barrier_neb_source_candidate_by_key(self, key: str) -> _EnergyBarrierNebCandidate | None:
        for candidate in self._energy_barrier_neb_source_candidates():
            if candidate.key == key:
                return candidate
        return None

    def _completed_energy_barrier_neb_candidates(self) -> list[_EnergyBarrierCompletedNebCandidate]:
        if self._document is None:
            return []
        candidates: list[_EnergyBarrierCompletedNebCandidate] = []
        for result in self._document.results.values():
            if result.result_type != NUDGED_ELASTIC_BAND_CALC_TYPE:
                continue
            run = self._document.runs.get(result.calculation_run_id) if result.calculation_run_id is not None else None
            job = self._document.jobs.get(run.job_id) if run is not None and run.job_id is not None else None
            if run is None or job is None:
                continue
            candidates.append(
                _EnergyBarrierCompletedNebCandidate(
                    key=str(result.id),
                    run=run,
                    job=job,
                    result=result,
                )
            )
        return sorted(candidates, key=lambda item: (item.job.label, item.key))

    def _completed_energy_barrier_neb_candidate_by_key(self, key: str) -> _EnergyBarrierCompletedNebCandidate | None:
        for candidate in self._completed_energy_barrier_neb_candidates():
            if candidate.key == key:
                return candidate
        return None

    def _serialize_energy_barrier_neb_sources(self, keys: list[str]) -> list[dict[str, object]]:
        candidates_by_key = {candidate.key: candidate for candidate in self._energy_barrier_neb_source_candidates()}
        serialized: list[dict[str, object]] = []
        for key in keys:
            candidate = candidates_by_key.get(key)
            if candidate is None:
                continue
            serialized.append(dict(candidate.source))
        return serialized

    def _job_for_energy_barrier_source_key(self, definition, key: str) -> Job | None:
        if self._document is None or definition is None:
            return None
        mapping = definition.parameters.get("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            return None
        raw_job_id = mapping.get(key)
        if not isinstance(raw_job_id, str):
            return None
        try:
            job_id = UUID(raw_job_id)
        except ValueError:
            return None
        job = self._document.jobs.get(job_id)
        if job is None or job.calc_type != TRANSITION_STATE_SEARCH_CALC_TYPE:
            return None
        return job

    def _ensure_energy_barrier_neb_definition(self, source_keys: list[str]):
        if self._document is None:
            return None
        candidates = [
            candidate
            for key in source_keys
            if (candidate := self._energy_barrier_neb_source_candidate_by_key(key)) is not None
        ]
        if not candidates:
            return None
        definition = self._active_energy_barrier_neb_definition()
        if definition is not None:
            for candidate in candidates:
                for source_key in ("reactant_structure_id", "product_structure_id"):
                    structure_id = self._uuid_from_source(candidate.source, source_key)
                    if structure_id is not None and structure_id not in definition.input_structure_ids:
                        definition.input_structure_ids.append(structure_id)
                for source_key in ("reactant_site_id", "product_site_id"):
                    site_id = self._uuid_from_source(candidate.source, source_key)
                    if site_id is not None and site_id not in definition.input_site_ids:
                        definition.input_site_ids.append(site_id)
            config = definition.parameters.setdefault("configuration", {})
            if isinstance(config, dict):
                config.setdefault("method", "NEB")
                config["sources"] = _merge_serialized_sources(
                    config.get("sources", []),
                    self._serialize_energy_barrier_neb_sources(source_keys),
                    key_field="key",
                )
            return definition

        output_root = self._document.project_root / "files" / "energy_barriers_neb"
        structure_ids = list(
            dict.fromkeys(
                structure_id
                for candidate in candidates
                for structure_id in (
                    self._uuid_from_source(candidate.source, "reactant_structure_id"),
                    self._uuid_from_source(candidate.source, "product_structure_id"),
                )
                if structure_id is not None
            )
        )
        site_ids = list(
            dict.fromkeys(
                site_id
                for candidate in candidates
                for site_id in (
                    self._uuid_from_source(candidate.source, "reactant_site_id"),
                    self._uuid_from_source(candidate.source, "product_site_id"),
                )
                if site_id is not None
            )
        )
        return self._project_service.register_nudged_elastic_band_definition(
            self._document,
            input_structure_ids=structure_ids,
            input_site_ids=site_ids,
            parameters={
                "method": "NEB",
                "sources": self._serialize_energy_barrier_neb_sources(source_keys),
            },
            output_directory=output_root,
        )

    def _job_for_energy_barrier_neb_source_key(self, definition, key: str) -> Job | None:
        if self._document is None or definition is None:
            return None
        mapping = definition.parameters.get("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            return None
        raw_job_id = mapping.get(key)
        if not isinstance(raw_job_id, str):
            return None
        try:
            job_id = UUID(raw_job_id)
        except ValueError:
            return None
        job = self._document.jobs.get(job_id)
        if job is None or job.calc_type != NUDGED_ELASTIC_BAND_CALC_TYPE:
            return None
        return job

    def _build_energy_barriers_states(self) -> dict[str, RelaxationRunState]:
        states: dict[str, RelaxationRunState] = {}
        definition = self._active_energy_barriers_definition()
        for candidate in self._eligible_energy_barrier_source_candidates():
            job = self._job_for_energy_barrier_source_key(definition, candidate.key)
            if job is None:
                states[candidate.key] = RelaxationRunState(status="ready")
                continue
            states[candidate.key] = RelaxationRunState(
                status=job.status.value,
                job_id=job.id,
                job_label=job.label,
                error_message=job.error_message,
            )
        return states

    def _build_energy_barrier_neb_states(self) -> dict[str, RelaxationRunState]:
        states: dict[str, RelaxationRunState] = {}
        definition = self._active_energy_barrier_neb_definition()
        for candidate in self._energy_barrier_neb_source_candidates():
            job = self._job_for_energy_barrier_neb_source_key(definition, candidate.key)
            if job is None:
                states[candidate.key] = RelaxationRunState(status="ready")
                continue
            states[candidate.key] = RelaxationRunState(
                status=job.status.value,
                job_id=job.id,
                job_label=job.label,
                error_message=job.error_message,
            )
        return states

    def _jobs_for_energy_barrier_source_keys(self, keys: list[str]) -> list[Job]:
        if self._document is None:
            return []
        selected: list[Job] = []
        seen: set[UUID] = set()
        for definition in self._energy_barriers_definitions():
            for key in keys:
                job = self._job_for_energy_barrier_source_key(definition, key)
                if job is None or job.id in seen:
                    continue
                selected.append(job)
                seen.add(job.id)
        return selected

    def _jobs_for_energy_barrier_neb_source_keys(self, keys: list[str]) -> list[Job]:
        if self._document is None:
            return []
        selected: list[Job] = []
        seen: set[UUID] = set()
        for definition in self._energy_barrier_neb_definitions():
            for key in keys:
                job = self._job_for_energy_barrier_neb_source_key(definition, key)
                if job is None or job.id in seen:
                    continue
                selected.append(job)
                seen.add(job.id)
        return selected

    def _materialize_energy_barriers_jobs(self, keys: list[str]) -> list[Job]:
        if self._document is None:
            return []
        definition = self._ensure_energy_barriers_definition(keys)
        if definition is None:
            return []
        output_root = self._definition_output_directory(definition)
        host_id = self._document.project.host_structure_id
        host = self._document.structures.get(host_id) if host_id is not None else None
        if output_root is None or host is None:
            return []

        configs: list[TransitionStateSearchJobConfig] = []
        for key in keys:
            candidate = self._energy_barrier_source_candidate_by_key(key)
            if candidate is None:
                continue
            try:
                prepared = prepare_transition_state_structures(
                    pristine_structure=host,
                    reactant_structure=candidate.reactant.structure,
                    product_structure=candidate.product.structure,
                    reactant_anchor_fractional=candidate.path.start_fractional,
                    product_anchor_fractional=candidate.path.end_fractional,
                )
                reactant_spec = structure_to_spec(prepared.reactant)
                product_spec = structure_to_spec(prepared.product)
            except (EnergyBarrierError, StructureBridgeError):
                continue
            values, _advanced_names, _extras = self._energy_barrier_row_configuration_with_source_defaults(
                definition,
                key,
                target_lattice=candidate.reactant.structure.lattice_vectors,
            )
            values = {name: value for name, value in values.items() if name != "task"}
            try:
                overrides = _input_template_from_keyword_values(values)
            except ValueError:
                continue
            configs.append(
                TransitionStateSearchJobConfig(
                    source=TransitionStateSearchSource(
                        label=self._energy_barrier_source_label(candidate),
                        reactant_structure=reactant_spec,
                        product_positions_frac=position_block_from_structure(product_spec),
                        product_atomic_spins=product_spec.atomic_spins,
                        source_site_ids=(candidate.reactant.site.id, candidate.product.site.id),
                        source_relaxation_run_ids=(candidate.reactant.run.id, candidate.product.run.id),
                        source_relaxation_job_ids=(candidate.reactant.job.id, candidate.product.job.id),
                        path_start_fractional=candidate.path.start_fractional,
                        path_end_fractional=candidate.path.end_fractional,
                        path_length_ang=candidate.path.length_ang,
                        provenance={"source_key": key},
                    ),
                    cell_overrides=overrides.cell,
                    param_overrides=overrides.param,
                )
            )
        if not configs:
            return []
        try:
            specs = expand_transition_state_search_jobs(
                jobs=configs,
                name_prefix="barrier",
                global_defaults=self._project_convergence_input_defaults(),
            )
        except (ValueError, StructureBridgeError):
            return []

        mapping = definition.parameters.setdefault("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            mapping = {}
            definition.parameters["job_ids_by_label"] = mapping

        spec_by_key = {str(spec.provenance.get("source_key", "")): spec for spec in specs}
        created: list[Job] = []
        for key in keys:
            spec = spec_by_key.get(key)
            existing = self._job_for_energy_barrier_source_key(definition, key)
            if existing is not None:
                if existing.status == JobStatus.DEFINED and spec is not None:
                    write_job_directory(spec, output_root)
                created.append(existing)
                continue
            if spec is None:
                continue
            write_job_directory(spec, output_root)
            job = self._project_service.add_job(
                self._document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type=TRANSITION_STATE_SEARCH_CALC_TYPE,
            )
            mapping[key] = str(job.id)
            created.append(job)

        jobs = self._dedupe_jobs(created)
        if jobs:
            self._refresh_energy_barriers_view()
            self._persist_document_ui_state(
                failure_message="Energy Barriers jobs were generated, but autosave failed."
            )
        return jobs

    def _materialize_energy_barrier_neb_jobs(self, keys: list[str]) -> list[Job]:
        if self._document is None:
            return []
        definition = self._ensure_energy_barrier_neb_definition(keys)
        if definition is None:
            return []
        output_root = self._definition_output_directory(definition)
        host_id = self._document.project.host_structure_id
        host = self._document.structures.get(host_id) if host_id is not None else None
        if output_root is None or host is None:
            return []

        candidates_by_key = {candidate.key: candidate for candidate in self._energy_barrier_neb_source_candidates()}
        configs: list[NudgedElasticBandJobConfig] = []
        for key in keys:
            candidate = candidates_by_key.get(key)
            if candidate is None:
                continue
            reactant_id = self._uuid_from_source(candidate.source, "reactant_structure_id")
            product_id = self._uuid_from_source(candidate.source, "product_structure_id")
            reactant_site_id = self._uuid_from_source(candidate.source, "reactant_site_id")
            product_site_id = self._uuid_from_source(candidate.source, "product_site_id")
            reactant_run_id = self._uuid_from_source(candidate.source, "reactant_run_id")
            product_run_id = self._uuid_from_source(candidate.source, "product_run_id")
            reactant_job_id = self._uuid_from_source(candidate.source, "reactant_job_id")
            product_job_id = self._uuid_from_source(candidate.source, "product_job_id")
            reactant_structure = self._document.structures.get(reactant_id) if reactant_id is not None else None
            product_structure = self._document.structures.get(product_id) if product_id is not None else None
            if reactant_structure is None or product_structure is None:
                continue
            try:
                prepared = prepare_transition_state_structures(
                    pristine_structure=host,
                    reactant_structure=reactant_structure,
                    product_structure=product_structure,
                    reactant_anchor_fractional=self._fractional_triplet_from_source(candidate.source, "path_start_fractional"),
                    product_anchor_fractional=self._fractional_triplet_from_source(candidate.source, "path_end_fractional"),
                )
                reactant_spec = structure_to_spec(prepared.reactant)
                product_block = self._absolute_position_block_from_structure(prepared.product)
            except (EnergyBarrierError, StructureBridgeError, ValueError):
                continue
            values, _advanced_names, _extras = self._workflow_row_configuration_with_project_defaults(
                definition,
                key,
                target_lattice=reactant_structure.lattice_vectors,
            )
            values = _strip_legacy_energy_barrier_params(values)
            values = {name: value for name, value in values.items() if name != "task"}
            try:
                overrides = _input_template_from_keyword_values(values)
            except ValueError:
                continue
            configs.append(
                NudgedElasticBandJobConfig(
                    source=NudgedElasticBandSource(
                        label=f"{str(candidate.source.get('label', '')).strip() or candidate.key}_neb",
                        reactant_structure=reactant_spec,
                        product_positions_abs=product_block,
                        intermediate_positions_abs=None,
                        product_atomic_spins=reactant_spec.atomic_spins,
                        intermediate_atomic_spins=None,
                        source_site_ids=tuple(
                            value
                            for value in (reactant_site_id, product_site_id)
                            if value is not None
                        ),
                        source_relaxation_run_ids=tuple(
                            value
                            for value in (reactant_run_id, product_run_id)
                            if value is not None
                        ),
                        source_relaxation_job_ids=(reactant_job_id, product_job_id),
                        source_transition_state_run_id=None,
                        source_transition_state_job_id=None,
                        path_length_ang=self._float_from_source(candidate.source, "path_length_ang"),
                        provenance={"source_key": key},
                    ),
                    cell_overrides=overrides.cell,
                    param_overrides=overrides.param,
                )
            )
        if not configs:
            return []
        try:
            specs = expand_nudged_elastic_band_jobs(
                jobs=configs,
                name_prefix="barrier",
                global_defaults=self._project_convergence_input_defaults(),
            )
        except (ValueError, StructureBridgeError):
            return []

        mapping = definition.parameters.setdefault("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            mapping = {}
            definition.parameters["job_ids_by_label"] = mapping

        spec_by_key = {str(spec.provenance.get("source_key", "")): spec for spec in specs}
        created: list[Job] = []
        for key in keys:
            spec = spec_by_key.get(key)
            existing = self._job_for_energy_barrier_neb_source_key(definition, key)
            if existing is not None:
                if existing.status == JobStatus.DEFINED and spec is not None:
                    write_job_directory(spec, output_root)
                created.append(existing)
                continue
            if spec is None:
                continue
            write_job_directory(spec, output_root)
            job = self._project_service.add_job(
                self._document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
            )
            mapping[key] = str(job.id)
            created.append(job)

        jobs = self._dedupe_jobs(created)
        if jobs:
            self._refresh_energy_barriers_view()
            self._persist_document_ui_state(
                failure_message="Energy Barriers NEB jobs were generated, but autosave failed."
            )
        return jobs

    def _uuid_from_source(self, source: Mapping[str, object], key: str) -> UUID | None:
        raw = source.get(key)
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            return UUID(raw)
        except ValueError:
            return None

    def _fractional_triplet_from_source(self, source: Mapping[str, object], key: str) -> tuple[float, float, float]:
        raw = source.get(key)
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError(f"Expected three fractional coordinates for {key}.")
        return (float(raw[0]), float(raw[1]), float(raw[2]))

    def _float_from_source(self, source: Mapping[str, object], key: str) -> float | None:
        raw = source.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _transition_state_file_path(self, result: Result) -> Path | None:
        if not isinstance(result.data, dict):
            return None
        raw = result.data.get("transition_state_file")
        if not isinstance(raw, str) or not raw.strip():
            return None
        return Path(raw)

    def _structure_from_transition_state_image(self, job_label: str, image: TransitionStateImage) -> Structure:
        lattice = [list(vector) for vector in image.lattice_vectors]
        fractional_positions = [
            lattice_cart_to_frac(position, lattice)
            for position in image.cartesian_positions
        ]
        host_id = self._document.project.host_structure_id if self._document is not None else None
        return Structure(
            label=f"{job_label} {image.label}",
            origin="energy_barriers_neb_preview",
            lattice_vectors=lattice,
            species=list(image.species),
            fractional_positions=[list(position) for position in fractional_positions],
            parent_structure_id=host_id,
            properties={
                "pbc": [True, True, True],
                "source_image_label": image.label,
                "reaction_coordinate": image.reaction_coordinate,
                "energy_eV": image.energy,
            },
        )
    def _absolute_position_block_from_structure(self, structure: Structure) -> PositionBlock:
        rows = []
        for species, fractional in zip(structure.species, structure.fractional_positions, strict=True):
            coordinates = lattice_frac_to_cart(fractional, structure.lattice_vectors)
            rows.append(AtomicPosition(str(species), tuple(float(value) for value in coordinates)))
        return PositionBlock(positions=tuple(rows), unit="ang")

    def _on_configure_quantum_corrections_source(self, payload: object) -> None:
        if self._document is None:
            return
        keys = _ordered_string_list(payload)
        if not keys:
            return
        candidates = [
            candidate
            for key in keys
            if (candidate := self._quantum_source_candidate_by_key(key)) is not None
        ]
        if not candidates:
            return
        definition = self._active_quantum_corrections_definition()
        key_list = [candidate.key for candidate in candidates]
        values_by_key: list[dict[str, object]] = []
        advanced_names: set[str] = set()
        stage_states = []
        pseudopotential_families: list[str | None] = []
        farms_only_values: list[bool] = []
        for candidate in candidates:
            values, pseudopotential_family = self._quantum_runtime_configuration(definition, candidate.key)
            _params, row_advanced_names, extras = self._workflow_row_configuration(definition, candidate.key)
            values_by_key.append(values)
            advanced_names.update(row_advanced_names)
            stage_states.append(self._quantum_stage_state(definition, candidate))
            pseudopotential_families.append(pseudopotential_family)
            farms_only_values.append(self._quantum_allows_farms_only_edit(definition, candidate))
        common_values = dict(values_by_key[0])
        for row_values in values_by_key[1:]:
            common_values = {
                name: value for name, value in common_values.items() if name in row_values and row_values[name] == value
            }
        first_candidate = candidates[0]
        first_stage_state = stage_states[0]
        farms_only = len(set(farms_only_values)) == 1 and farms_only_values[0]
        fully_editable = all(
            stage_state.stage == _QUANTUM_STAGE_SINGLEPOINT and stage_state.status == "ready"
            for stage_state in stage_states
        )
        read_only = not fully_editable and not farms_only
        common_pseudopotential_family = (
            pseudopotential_families[0]
            if pseudopotential_families and all(value == pseudopotential_families[0] for value in pseudopotential_families)
            else ""
        )
        num_farms_values = [
            self._quantum_num_farms_from_extras(self._workflow_row_configuration(definition, candidate.key)[2])
            for candidate in candidates
        ]
        common_num_farms = (
            num_farms_values[0] if num_farms_values and all(value == num_farms_values[0] for value in num_farms_values) else 1
        )

        dialog = QuantumCorrectionsConfigDialog(
            specs=self._quantum_parameter_specs(),
            pseudopotential_family=common_pseudopotential_family or "unknown",
            num_farms=common_num_farms,
            intro_text=(
                f"Selected sources: {len(candidates)}\n"
                f"Current stage: {first_stage_state.stage if len(candidates) == 1 else 'mixed/selected'}\n"
                f"Displacements required for the phonon stage: {3 * len(first_candidate.structure.species)}"
            ),
            read_only=read_only,
            editable_param_names=() if farms_only else None,
            pseudopotential_editable=fully_editable,
            num_farms_editable=(fully_editable or farms_only),
            species_options=_merge_species_options([candidate.structure for candidate in candidates]),
            parent=self,
        )
        if read_only:
            dialog.setWindowTitle("Inspect Quantum Corrections Parameters")
        elif farms_only:
            dialog.setWindowTitle("Configure Quantum Corrections Stage 2 Farms")
        else:
            dialog.setWindowTitle("Configure Quantum Corrections Parameters")
        dialog.set_values(common_values, preferred_exclusive=kpoint_exclusive_hint(values_by_key))
        dialog.set_advanced_names(tuple(sorted(advanced_names)))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if read_only:
            return

        pseudopotential_family = dialog.pseudopotential_family()
        dialog_payload = self._workflow_dialog_payload(
            dialog,
            extra_payload={
                "pseudopotential_family": pseudopotential_family,
                "num_farms": dialog.num_farms(),
            },
        )
        definition = self._ensure_quantum_corrections_definition(key_list)
        if definition is None:
            return
        for key in key_list:
            self._store_workflow_row_configuration(definition, key, dialog_payload)
        self._refresh_quantum_corrections_view()
        self._persist_document_ui_state(
            failure_message="Quantum Corrections configuration was updated, but autosave failed."
        )

    def _on_quantum_corrections_source_label_renamed(self, key: str, label: str) -> None:
        if self._document is None or self._shutting_down:
            return
        candidate = self._quantum_source_candidate_by_key(key)
        if candidate is None:
            return
        normalized = str(label).strip()
        if not normalized or normalized == candidate.site.label:
            self._refresh_quantum_corrections_view()
            return
        candidate.site.label = normalized
        self._refresh_quantum_corrections_view()
        self._refresh_energy_barriers_view()
        self._persist_document_ui_state(
            failure_message="Quantum Corrections site label was updated, but autosave failed."
        )

    def _on_configure_energy_barrier_source(self, payload: object) -> None:
        if self._document is None:
            return
        keys = _ordered_string_list(payload)
        if not keys:
            return
        candidates = [
            candidate
            for key in keys
            if (candidate := self._energy_barrier_source_candidate_by_key(key)) is not None
        ]
        if not candidates:
            return
        definition = self._active_energy_barriers_definition()
        values_by_key: list[dict[str, object]] = []
        advanced_names: set[str] = set()
        for candidate in candidates:
            values, row_advanced_names, _extras = self._energy_barrier_row_configuration_with_source_defaults(
                definition,
                candidate.key,
                target_lattice=candidate.reactant.structure.lattice_vectors,
            )
            values_by_key.append(values)
            advanced_names.update(row_advanced_names)
        common_values = dict(values_by_key[0])
        for row_values in values_by_key[1:]:
            common_values = {
                name: value for name, value in common_values.items() if name in row_values and row_values[name] == value
            }
        read_only = any(not workflow_row_state_is_ready(self._build_energy_barriers_states(), candidate.key) for candidate in candidates)

        dialog = ParameterDialog(
            specs=self._energy_barrier_source_parameter_specs(),
            workflow_stage="energy_barriers",
            locked_values={
                "task": "TRANSITIONSTATESEARCH",
                "tssearch_method": "LSTQST",
            },
            intro_text=(
                f"Path: {self._energy_barrier_source_label(candidates[0])}"
                if len(candidates) == 1
                else f"Selected paths: {len(candidates)}"
            ),
            read_only=read_only,
            species_options=_merge_species_options([candidate.reactant.structure for candidate in candidates]),
            parent=self,
        )
        dialog.setWindowTitle(
            "Inspect Energy Barriers Parameters" if read_only else "Configure Energy Barriers Parameters"
        )
        dialog.set_values(common_values, preferred_exclusive=kpoint_exclusive_hint(values_by_key))
        dialog.set_advanced_names(tuple(sorted(advanced_names)))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if read_only:
            return

        row_payload = self._workflow_dialog_payload(dialog)
        definition = self._ensure_energy_barriers_definition([candidate.key for candidate in candidates])
        if definition is None:
            return
        for candidate in candidates:
            self._store_workflow_row_configuration(definition, candidate.key, row_payload)
        self._refresh_energy_barriers_view()
        self._persist_document_ui_state(
            failure_message="Energy Barriers configuration was updated, but autosave failed."
        )

    def _on_configure_energy_barrier_result(self, payload: object) -> None:
        if self._document is None:
            return
        keys = _ordered_string_list(payload)
        if not keys:
            return
        candidates = [
            candidate
            for key in keys
            if (candidate := self._energy_barrier_neb_source_candidate_by_key(key)) is not None
        ]
        if not candidates:
            return
        definition = self._active_energy_barrier_neb_definition()
        values_by_key: list[dict[str, object]] = []
        advanced_names: set[str] = set()
        reactant_structures: list[Structure | None] = []
        for candidate in candidates:
            reactant_structure = self._document.structures.get(self._uuid_from_source(candidate.source, "reactant_structure_id"))
            reactant_structures.append(reactant_structure)
            values, row_advanced_names, _extras = self._workflow_row_configuration_with_project_defaults(
                definition,
                candidate.key,
                target_lattice=reactant_structure.lattice_vectors if reactant_structure is not None else None,
            )
            values = _strip_legacy_energy_barrier_params(values)
            values_by_key.append(values)
            advanced_names.update(row_advanced_names)
        common_values = dict(values_by_key[0])
        for row_values in values_by_key[1:]:
            common_values = {
                name: value for name, value in common_values.items() if name in row_values and row_values[name] == value
            }
        read_only = any(
            not workflow_row_state_is_ready(self._build_energy_barrier_neb_states(), candidate.key)
            for candidate in candidates
        )

        dialog = ParameterDialog(
            specs=self._energy_barrier_neb_parameter_specs(),
            workflow_stage="energy_barriers",
            locked_values={
                "task": "TRANSITIONSTATESEARCH",
                "tssearch_method": "NEB",
            },
            intro_text=(
                f"Path: {str(candidates[0].source.get('label', '')).strip() or candidates[0].key}"
                if len(candidates) == 1
                else f"Selected paths: {len(candidates)}"
            ),
            read_only=read_only,
            species_options=_merge_species_options(reactant_structures),
            parent=self,
        )
        dialog.setWindowTitle(
            "Inspect NEB Parameters" if read_only else "Configure NEB Parameters"
        )
        dialog.set_values(common_values, preferred_exclusive=kpoint_exclusive_hint(values_by_key))
        dialog.set_advanced_names(tuple(sorted(advanced_names)))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if read_only:
            return

        row_payload = self._workflow_dialog_payload(dialog)
        definition = self._ensure_energy_barrier_neb_definition([candidate.key for candidate in candidates])
        if definition is None:
            return
        for candidate in candidates:
            self._store_workflow_row_configuration(definition, candidate.key, row_payload)
        self._refresh_energy_barriers_view()
        self._persist_document_ui_state(
            failure_message="Energy Barriers NEB configuration was updated, but autosave failed."
        )

    def _on_view_energy_barrier_source(self, key: str) -> None:
        candidate = self._energy_barrier_source_candidate_by_key(key)
        if candidate is None:
            return
        try:
            preview_structure = self._energy_barrier_preview_structure(candidate)
        except ValueError as exc:
            QMessageBox.warning(self, "Path preview unavailable", str(exc))
            return

        dialog = EnergyBarrierPathViewerDialog(
            structure=preview_structure,
            start_fractional=candidate.path.start_fractional,
            end_fractional=candidate.path.end_fractional,
            title=f"{self._energy_barrier_source_label(candidate)} path preview",
            parent=self,
        )
        dialog.exec()

    def _jobs_for_quantum_source_keys(self, keys: list[str]) -> list[Job]:
        if self._document is None:
            return []
        selected: list[Job] = []
        seen: set[UUID] = set()
        for definition in self._quantum_corrections_definitions():
            for key in keys:
                job = self._job_for_quantum_source_key(definition, key)
                if job is None or job.id in seen:
                    continue
                selected.append(job)
                seen.add(job.id)
        return selected

    def _materialize_quantum_corrections_jobs(self, keys: list[str]) -> list[Job]:
        if self._document is None:
            return []
        definition = self._ensure_quantum_corrections_definition(keys)
        if definition is None:
            return []
        output_root = self._definition_output_directory(definition)
        if output_root is None:
            return []

        singlepoint_configs: list[QuantumPhononJobConfig] = []
        phonon_configs: list[QuantumPhononJobConfig] = []
        for key in keys:
            candidate = self._quantum_source_candidate_by_key(key)
            if candidate is None:
                continue
            stage_state = self._quantum_stage_state(definition, candidate)
            if stage_state.status != "ready" and (
                stage_state.current_job is None or stage_state.current_job.status != JobStatus.DEFINED
            ):
                continue
            try:
                structure = structure_to_spec(candidate.structure)
            except StructureBridgeError:
                continue
            values, pseudopotential_family = self._quantum_runtime_configuration(definition, key)
            _params, _advanced, extras = self._workflow_row_configuration(definition, key)
            values = {name: value for name, value in values.items() if name != "task"}
            method_value = values.pop("phonon_method", "FINITEDISPLACEMENT")
            try:
                method = _quantum_method_from_gui_value(method_value)
                overrides = _input_template_from_keyword_values(values)
            except ValueError:
                continue
            config = QuantumPhononJobConfig(
                source=QuantumPhononSource(
                    label=self._quantum_source_label(candidate),
                    structure=structure,
                    source_site_id=candidate.site.id,
                    source_relaxation_run_id=candidate.run.id,
                    source_relaxation_job_id=candidate.job.id,
                    charge_state=self._charge_state_for_job(candidate.job),
                    relative_energy_eV=candidate.site.relative_energy_eV,
                ),
                method=method,
                pseudopotential_family=str(pseudopotential_family),
                num_farms=self._quantum_num_farms_from_extras(extras),
                cell_overrides=overrides.cell,
                param_overrides=overrides.param,
            )
            if stage_state.stage == _QUANTUM_STAGE_SINGLEPOINT:
                singlepoint_configs.append(config)
            else:
                phonon_configs.append(config)
        if not singlepoint_configs and not phonon_configs:
            return []

        singlepoint_specs: list[JobSpec] = []
        phonon_specs: list[JobSpec] = []
        try:
            if singlepoint_configs:
                singlepoint_specs = expand_quantum_singlepoint_jobs(
                    jobs=singlepoint_configs,
                    name_prefix="quantum_singlepoint",
                )
            if phonon_configs:
                phonon_specs = expand_quantum_phonon_jobs(
                    jobs=phonon_configs,
                    name_prefix="quantum_phonon",
                )
        except (ValueError, StructureBridgeError):
            return []

        singlepoint_mapping = definition.parameters.setdefault("singlepoint_job_ids_by_label", {})
        if not isinstance(singlepoint_mapping, dict):
            singlepoint_mapping = {}
            definition.parameters["singlepoint_job_ids_by_label"] = singlepoint_mapping
        phonon_mapping = definition.parameters.setdefault("phonon_job_ids_by_label", {})
        if not isinstance(phonon_mapping, dict):
            phonon_mapping = {}
            definition.parameters["phonon_job_ids_by_label"] = phonon_mapping
        seed_mapping = definition.parameters.setdefault("job_seed_names_by_label", {})
        if not isinstance(seed_mapping, dict):
            seed_mapping = {}
            definition.parameters["job_seed_names_by_label"] = seed_mapping
        definition.parameters["job_ids_by_label"] = phonon_mapping

        singlepoint_specs_by_key = {
            str(spec.provenance.get("source_site_id", "")): spec
            for spec in singlepoint_specs
        }
        phonon_specs_by_key = {
            str(spec.provenance.get("source_site_id", "")): spec
            for spec in phonon_specs
        }
        created: list[Job] = []
        for key in keys:
            candidate = self._quantum_source_candidate_by_key(key)
            if candidate is None:
                continue
            stage_state = self._quantum_stage_state(definition, candidate)
            seed_name = self._quantum_seed_name_for_key(definition, key)
            seed_mapping[key] = seed_name
            if stage_state.stage == _QUANTUM_STAGE_SINGLEPOINT:
                spec = singlepoint_specs_by_key.get(key)
                existing = self._singlepoint_job_for_quantum_source_key(definition, key)
                mapping = singlepoint_mapping
                calc_type = QUANTUM_SINGLEPOINT_CALC_TYPE
            else:
                spec = phonon_specs_by_key.get(key)
                existing = self._phonon_job_for_quantum_source_key(definition, key)
                mapping = phonon_mapping
                calc_type = QUANTUM_PHONON_CALC_TYPE
            if spec is None:
                continue
            if existing is not None:
                if existing.status == JobStatus.DEFINED:
                    existing.label = spec.name
                    self._write_quantum_job_directory(spec, output_root, seed_name=seed_name)
                created.append(existing)
                continue
            self._write_quantum_job_directory(spec, output_root, seed_name=seed_name)
            job = self._project_service.add_job(
                self._document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type=calc_type,
            )
            mapping[key] = str(job.id)
            created.append(job)

        jobs = self._dedupe_jobs(created)
        if jobs:
            self._refresh_quantum_corrections_view()
            self._persist_document_ui_state(
                failure_message="Quantum Corrections jobs were generated, but autosave failed."
            )
        return jobs

    def _on_submit_local_from_quantum_corrections(self, keys: list[str]) -> None:
        if not self._check_definition_host(self._active_quantum_corrections_definition()):
            return
        self._submit_workflow_selection(
            keys,
            normalize=str,
            build_states=self._build_quantum_corrections_states,
            jobs_for_values=self._jobs_for_quantum_source_keys,
            materialize_values=self._materialize_quantum_corrections_jobs,
            is_ready=workflow_row_state_is_ready,
            no_ready_message="Only eligible relaxed muon sites in the 'ready' state can be submitted.",
            submit=self._submit_jobs_locally,
        )

    def _on_submit_cluster_from_quantum_corrections(self, keys: list[str]) -> None:
        if not self._check_definition_host(self._active_quantum_corrections_definition()):
            return
        self._submit_workflow_selection(
            keys,
            normalize=str,
            build_states=self._build_quantum_corrections_states,
            jobs_for_values=self._jobs_for_quantum_source_keys,
            materialize_values=self._materialize_quantum_corrections_jobs,
            is_ready=workflow_row_state_is_ready,
            no_ready_message="Only eligible relaxed muon sites in the 'ready' state can be submitted.",
            submit=self._submit_jobs_on_cluster,
        )

    def _on_reset_quantum_corrections_to_stage_one(self, keys: list[str]) -> None:
        if self._document is None:
            return
        definition = self._active_quantum_corrections_definition()
        if definition is None:
            return

        resettable_keys: list[str] = []
        for key in keys:
            candidate = self._quantum_source_candidate_by_key(key)
            if candidate is None:
                continue
            if self._quantum_can_reset_to_stage_one(definition, candidate):
                resettable_keys.append(key)
        if not resettable_keys:
            QMessageBox.information(
                self,
                "No resettable rows",
                "Only progressed Quantum Corrections rows without an active job can be reset to stage 1.",
            )
            return

        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("Reset Quantum Corrections")
        message.setText(f"Reset {len(resettable_keys)} Quantum Corrections row(s) to stage 1?")
        message.setInformativeText(
            "This removes the staged Quantum Corrections job records and clears the shared output directory "
            "for each selected row so stage 1 can be rerun with updated parameters."
        )
        message.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        message.setDefaultButton(QMessageBox.StandardButton.No)
        if message.exec() != QMessageBox.StandardButton.Yes:
            return

        cleanup_errors: list[str] = []
        reset_count = 0
        seed_mapping = definition.parameters.setdefault("job_seed_names_by_label", {})
        if not isinstance(seed_mapping, dict):
            seed_mapping = {}
            definition.parameters["job_seed_names_by_label"] = seed_mapping
        for key in resettable_keys:
            candidate = self._quantum_source_candidate_by_key(key)
            if candidate is None:
                continue
            singlepoint_job = self._singlepoint_job_for_quantum_source_key(definition, key)
            phonon_job = self._phonon_job_for_quantum_source_key(definition, key)
            output_dir = self._job_output_directory(phonon_job or singlepoint_job) if (phonon_job or singlepoint_job) else None
            if output_dir is not None and output_dir.exists() and output_dir.is_dir():
                try:
                    self._delete_path(output_dir)
                except OSError as exc:
                    cleanup_errors.append(f"- {candidate.site.label or candidate.job.label}: {exc}")
                    continue
            for job in (phonon_job, singlepoint_job):
                if job is None:
                    continue
                self._delete_job_records(job.id)
            seed_mapping.pop(key, None)
            reset_count += 1

        self._refresh_jobs_table()
        self._refresh_quantum_corrections_view()

        if cleanup_errors:
            QMessageBox.warning(
                self,
                "Directory cleanup issues",
                "Some Quantum Corrections rows could not be reset cleanly:\n" + "\n".join(cleanup_errors),
            )

        if reset_count:
            self._persist_document_ui_state(
                failure_message="Quantum Corrections stage reset was recorded, but autosave failed.",
            )
            self.statusBar().showMessage(f"Reset {reset_count} Quantum Corrections row(s) to stage 1")

    def _on_launch_quantum_phonon_analysis(self, result_id: str) -> None:
        if self._document is None:
            return
        try:
            parsed_result_id = UUID(str(result_id))
        except ValueError:
            return
        result = self._document.results.get(parsed_result_id)
        if result is None or result.result_type != QUANTUM_PHONON_CALC_TYPE:
            return

        existing = self._quantum_analysis_dialogs.get(result.id)
        if existing is not None:
            if existing.isVisible():
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            self._quantum_analysis_dialogs.pop(result.id, None)

        phonon_file = result.data.get("phonon_file") if isinstance(result.data, dict) else None
        if not isinstance(phonon_file, str) or not phonon_file.strip():
            QMessageBox.warning(self, "Phonon analysis unavailable", "The completed result does not reference a phonon file.")
            return
        phonon_path = Path(phonon_file)
        if not phonon_path.is_absolute() and self._document is not None:
            phonon_path = self._document.project_root / phonon_path
        if not phonon_path.exists():
            QMessageBox.warning(self, "Phonon analysis unavailable", f"Could not find phonon file:\n{phonon_path}")
            return

        job_label = "Quantum phonon"
        run = self._document.runs.get(result.calculation_run_id) if result.calculation_run_id is not None else None
        job = self._document.jobs.get(run.job_id) if run is not None and run.job_id is not None else None
        if job is not None:
            job_label = job.label

        try:
            dialog = QuantumPhononAnalysisDialog(
                phonon_path=str(phonon_path),
                job_label=job_label,
                result_data=result.data if isinstance(result.data, dict) else {},
                parent=self,
            )
        except PhononParseError as exc:
            QMessageBox.warning(self, "Phonon analysis unavailable", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Phonon analysis unavailable", f"Could not open the analysis window: {exc}")
            return

        dialog.destroyed.connect(lambda *_args, result_id=result.id: self._quantum_analysis_dialogs.pop(result_id, None))
        self._quantum_analysis_dialogs[result.id] = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_energy_barrier_endpoint_selection_changed(self, reactant_key: str, product_key: str) -> None:
        if self._document is None:
            return
        state = self._energy_barrier_form_state()
        state["selected_reactant_key"] = reactant_key
        state["selected_product_key"] = product_key
        self._document.ui_state["energy_barriers_form"] = state
        self._refresh_energy_barriers_view()

    @staticmethod
    def _parse_energy_barrier_max_path_length(text: str) -> float | None:
        normalized = str(text).strip()
        if not normalized:
            return None
        value = float(normalized)
        if value <= 0.0:
            raise ValueError("Maximum path length must be greater than zero.")
        return value

    def _on_generate_energy_barrier_paths(
        self,
        reactant_key: str,
        product_key: str,
        max_path_length_text: str = "",
    ) -> None:
        if self._document is None:
            return
        if not reactant_key or not product_key:
            QMessageBox.information(
                self,
                "Select endpoints",
                "Choose initial and final relaxed structures before generating paths.",
            )
            return
        reactant = self._quantum_source_candidate_by_key(reactant_key)
        product = self._quantum_source_candidate_by_key(product_key)
        if reactant is None or product is None:
            return
        if self._charge_state_for_job(reactant.job) != self._charge_state_for_job(product.job):
            QMessageBox.information(
                self,
                "Charge-state mismatch",
                "Initial and final structures must use the same muon charge state.",
            )
            return
        try:
            max_distance_ang = self._parse_energy_barrier_max_path_length(max_path_length_text)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid path length", str(exc))
            return
        candidates = self._enumerate_energy_barrier_source_candidates(
            reactant_key,
            product_key,
            max_distance_ang=max_distance_ang,
        )
        state = self._energy_barrier_form_state()
        state["selected_reactant_key"] = reactant_key
        state["selected_product_key"] = product_key
        if max_distance_ang is None:
            state.pop("max_path_length_ang", None)
        else:
            state["max_path_length_ang"] = max_distance_ang
        state["generated_sources"] = _replace_serialized_sources_for_pair(
            self._energy_barrier_generated_sources_from_state(),
            reactant_site_id=str(reactant.site.id),
            product_site_id=str(product.site.id),
            new_sources=self._serialize_energy_barrier_source_candidates(candidates),
        )
        self._document.ui_state["energy_barriers_form"] = state
        self._refresh_energy_barriers_view()
        self._persist_document_ui_state(
            failure_message="Energy Barriers paths were generated, but autosave failed."
        )

    def _on_remove_energy_barrier_sources(self, payload: object) -> None:
        keys = self._removed_row_labels(payload)
        if self._document is None or not keys:
            return
        if not self._cleanup_energy_barrier_source_jobs(keys):
            self._refresh_energy_barriers_view()
            return
        self._remove_energy_barrier_sources_from_ui_state(keys)
        for definition in self._energy_barriers_definitions():
            self._remove_energy_barrier_sources_from_definition(definition, keys)
        for definition in self._energy_barrier_neb_definitions():
            self._remove_energy_barrier_neb_sources_from_definition(definition, keys)
        self._refresh_energy_barriers_view()
        self._refresh_jobs_table()
        self._refresh_journal_view()
        self.statusBar().showMessage(f"Deleted {len(keys)} energy-barrier path(s)")
        self._persist_document_ui_state(
            failure_message="Energy Barriers paths were removed, but autosave failed."
        )

    def _on_energy_barrier_neb_result_selected(self, key: str) -> None:
        energy_barriers_view = self._energy_barriers_view
        if energy_barriers_view is None:
            return
        candidate = self._completed_energy_barrier_neb_candidate_by_key(key)
        if candidate is None:
            energy_barriers_view.clear_neb_preview()
            return
        ts_path = self._transition_state_file_path(candidate.result)
        if ts_path is None or not ts_path.exists():
            energy_barriers_view.clear_neb_preview(title=f"{candidate.job.label} preview unavailable")
            return
        try:
            transition_state = parse_transition_state_file(ts_path)
        except TransitionStateParseError:
            energy_barriers_view.clear_neb_preview(title=f"{candidate.job.label} preview unavailable")
            return
        image_rows = [
            EnergyBarrierPreviewImageRow(
                label=image.label,
                reaction_coordinate=f"{image.reaction_coordinate:.6g}",
                energy=f"{image.energy:.6g} eV",
                reaction_coordinate_value=image.reaction_coordinate,
                energy_value=image.energy,
            )
            for image in transition_state.images
        ]
        structures = [
            self._structure_from_transition_state_image(candidate.job.label, image)
            for image in transition_state.images
        ]
        energy_barriers_view.set_neb_preview(
            title=f"{candidate.job.label} ({len(transition_state.images)} images)",
            image_rows=image_rows,
            structures=structures,
        )

    def _on_submit_local_neb_from_energy_barriers(self, keys: list[str]) -> None:
        if not self._check_definition_host(self._active_energy_barrier_neb_definition()):
            return
        self._submit_workflow_selection(
            keys,
            normalize=str,
            build_states=self._build_energy_barrier_neb_states,
            jobs_for_values=self._jobs_for_energy_barrier_neb_source_keys,
            materialize_values=self._materialize_energy_barrier_neb_jobs,
            is_ready=workflow_row_state_is_ready,
            no_ready_message="Only completed transition-state searches in the 'ready' state can be submitted as NEB jobs.",
            submit=self._submit_jobs_locally,
        )

    def _on_submit_cluster_neb_from_energy_barriers(self, keys: list[str]) -> None:
        if not self._check_definition_host(self._active_energy_barrier_neb_definition()):
            return
        self._submit_workflow_selection(
            keys,
            normalize=str,
            build_states=self._build_energy_barrier_neb_states,
            jobs_for_values=self._jobs_for_energy_barrier_neb_source_keys,
            materialize_values=self._materialize_energy_barrier_neb_jobs,
            is_ready=workflow_row_state_is_ready,
            no_ready_message="Only completed transition-state searches in the 'ready' state can be submitted as NEB jobs.",
            submit=self._submit_jobs_on_cluster,
        )

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
        self._refresh_uep_defaults()
        self._refresh_uep_supercell_options()

    def _on_uep_supercell_edit_requested(self, payload: object) -> None:
        if self._document is None:
            return
        try:
            row_id = int(payload)
        except (TypeError, ValueError):
            return

        source_structure = self._selected_uep_source_structure()
        if source_structure is None:
            QMessageBox.warning(
                self,
                "No source structure",
                "Select or generate a source structure before configuring UEP supercells.",
            )
            return

        existing = self._uep_view.row_supercell_matrix(row_id)
        initial_matrix = (
            [list(line) for line in existing]
            if existing is not None
            else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        dialog = self._open_supercell_editor(
            source_structure=source_structure,
            workflow_stage=WorkflowStage.UEP,
            initial_matrix=initial_matrix,
            select_mode=True,
        )
        if dialog is None:
            return

        matrix_values = dialog.selected_matrix()
        if matrix_values is None:
            return
        matrix = (
            (matrix_values[0][0], matrix_values[0][1], matrix_values[0][2]),
            (matrix_values[1][0], matrix_values[1][1], matrix_values[1][2]),
            (matrix_values[2][0], matrix_values[2][1], matrix_values[2][2]),
        )
        self._refresh_uep_supercell_options()
        self._uep_view.set_row_supercell(row_id, matrix)
        self.statusBar().showMessage("Selected UEP simulation cell")

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
            expand_uep_jobs(
                host_structure=source_structure,
                config=config,
                project_cutoff_energy=self._document.project.recommended_cutoff_energy,
                project_kpoint_grid=self._document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=self._document.project.recommended_kpoint_lattice,
            )
        except (ValueError, StructureBridgeError) as exc:
            QMessageBox.warning(self, "UEP setup invalid", str(exc))
            return

        output_root = self._document.project_root / "files" / "uep"
        definition = self._project_service.register_uep_definition(
            self._document,
            input_structure_id=source_structure.id,
            parameters=self._serialize_uep_config(config),
            output_directory=output_root,
        )
        self._sync_equivalent_uep_mappings(definition)
        self._document.ui_state[self._ACTIVE_UEP_DEFINITION_KEY] = str(definition.id)
        self._refresh_uep_rows()
        self._refresh_journal_view()
        self.statusBar().showMessage(
            f"Configured {len(config.rows)} UEP source job(s); input files will be generated on submission"
        )

    def _jobs_for_uep_labels(self, labels: list[str]) -> list[Job]:
        if self._document is None:
            return []
        definition = self._active_uep_definition()
        if definition is None:
            return []
        self._sync_equivalent_uep_mappings(definition)
        selected: list[Job] = []
        seen: set[UUID] = set()
        for label in labels:
            job = self._job_for_uep_label(definition, label)
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    def _materialize_uep_jobs(self, labels: list[str]) -> list[Job]:
        if self._document is None:
            return []
        definition = self._active_uep_definition()
        if definition is None or not definition.input_structure_ids:
            return []
        self._sync_equivalent_uep_mappings(definition)
        config = self._deserialize_uep_config(definition.parameters.get("configuration", {}))
        if config is None:
            return []
        source_structure = self._document.structures.get(definition.input_structure_ids[0])
        output_root = self._definition_output_directory(definition)
        if source_structure is None or output_root is None:
            return []

        try:
            expansion = expand_uep_jobs(
                host_structure=source_structure,
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

        spec_by_label = {str(spec.provenance.get("label", "")): spec for spec in expansion.jobs}
        created: list[Job] = []
        for label in labels:
            spec = spec_by_label.get(label)
            existing = self._job_for_uep_label(definition, label)
            if existing is not None:
                if existing.status == JobStatus.DEFINED and spec is not None:
                    write_job_directory(spec, output_root)
                created.append(existing)
                continue
            if spec is None:
                continue
            write_job_directory(spec, output_root)
            job = self._project_service.add_job(
                self._document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type="castep_uep_source",
            )
            mapping[label] = str(job.id)
            created.append(job)
        return self._dedupe_jobs(created)

    def _on_submit_local_from_uep(self, labels: list[str]) -> None:
        self._submit_workflow_selection(
            labels,
            normalize=str,
            build_states=self._build_uep_states,
            jobs_for_values=self._jobs_for_uep_labels,
            materialize_values=self._materialize_uep_jobs,
            is_ready=workflow_row_state_is_ready,
            no_ready_message="Only configured UEP rows in the 'ready' state can be submitted.",
            submit=self._submit_jobs_locally,
        )

    def _on_submit_cluster_from_uep(self, labels: list[str]) -> None:
        self._submit_workflow_selection(
            labels,
            normalize=str,
            build_states=self._build_uep_states,
            jobs_for_values=self._jobs_for_uep_labels,
            materialize_values=self._materialize_uep_jobs,
            is_ready=workflow_row_state_is_ready,
            no_ready_message="Only configured UEP rows in the 'ready' state can be submitted.",
            submit=self._submit_jobs_on_cluster,
        )

    def _on_cancel_from_uep(self, labels: list[str]) -> None:
        self._cancel_jobs(self._jobs_for_uep_labels(labels))

    def _on_inspect_failed_from_uep(self, labels: list[str]) -> None:
        self._inspect_failed_jobs(self._jobs_for_uep_labels(labels))

    def _on_reset_failed_from_uep(self, labels: list[str]) -> None:
        self._reset_failed_jobs(self._jobs_for_uep_labels(labels))

    def _on_open_uep_analysis_requested(self, label: object) -> None:
        if self._document is None:
            return
        row_label = str(label).strip()
        if not row_label:
            return

        definition = self._active_uep_definition()
        if definition is None:
            QMessageBox.information(self, "No UEP workflow", "Configure and complete a UEP source job first.")
            return

        job = self._job_for_uep_label(definition, row_label)
        if job is None or job.status != JobStatus.COMPLETED:
            QMessageBox.information(
                self,
                "UEP source incomplete",
                "Only completed UEP source jobs can be analysed.",
            )
            return

        output_dir = self._job_output_directory(job)
        if output_dir is None:
            QMessageBox.warning(self, "UEP files unavailable", "Could not locate the UEP job output directory.")
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

        source_structure_id = definition.input_structure_ids[0] if definition.input_structure_ids else None
        if source_structure_id is None:
            QMessageBox.warning(self, "UEP source unavailable", "The source structure reference is missing.")
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
            dialog = UepAnalysisDialog(
                source_job_label=job.label,
                source_structure_id=source_structure_id,
                source_structure=self._document.structures.get(source_structure_id),
                den_fmt_path=den_fmt_path,
                castep_path=castep_path,
                cell_path=cell_path if cell_path.exists() else None,
                initial_potential_grid=self._load_uep_potential_grid_for_job(job.id),
                initial_state=initial_state,
                parent=self,
            )
        except (UepError, OSError, ValueError) as exc:
            QMessageBox.warning(self, "UEP analysis unavailable", str(exc))
            return

        dialog.state_changed.connect(lambda state, job_id=job.id: self._on_uep_analysis_state_changed(job_id, state))
        dialog.potential_grid_ready.connect(
            lambda payload, job_id=job.id: self._on_uep_potential_grid_ready(job_id, payload)
        )
        dialog.setModal(False)
        dialog.destroyed.connect(lambda *_args, job_id=job.id: self._drop_uep_analysis_dialog(job_id))
        self._uep_analysis_dialogs[job.id] = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

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
        return (
            previous.get("batch_results") != updated.get("batch_results")
            or previous.get("batch_summary_data") != updated.get("batch_summary_data")
        )

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
        if self._uep_analysis_state_affects_muon_generation(previous_state, serialized_state):
            QTimer.singleShot(0, self._refresh_muon_generation_table)
        self._queue_raise_visible_uep_analysis_dialog(job_id)

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
            site for site in self._document.sites.values() if site.status in {"generated", "candidate", "staged"}
        ]
        candidates.sort(key=lambda item: self._natural_label_sort_key(item.label))
        rows = [
            {
                "id": str(site.id),
                "label": site.label,
                "method": site.generation_method or "",
                "status": site.status,
                "fractional": site.fractional_position,
                "cartesian": site.cartesian_position,
                "source_structure_id": str(site.source_structure_id) if site.source_structure_id else "",
            }
            for site in candidates
        ]
        self._muon_generation_view.set_sites(rows)
        self._muon_generation_view.set_uep_cluster_sources(self._uep_cluster_source_rows())
        self._refresh_workflow_stage_icons()
        self._sync_project_home_trail()

    @staticmethod
    def _natural_label_sort_key(label: str) -> tuple[tuple[int, object], ...]:
        parts: list[tuple[int, object]] = []
        for part in re.split(r"(\d+)", str(label)):
            if not part:
                continue
            if part.isdigit():
                parts.append((0, int(part)))
            else:
                parts.append((1, part.casefold()))
        return tuple(parts)

    def _refresh_host_structure_view(self) -> None:
        if self._document is None:
            self._host_structure_view.set_structure(None)
            self._host_structure_view.set_project_supercell_matrix(None)
            self._host_structure_view.set_promotable_hosts([], None)
            self._refresh_magnetic_field_view()
            self._refresh_active_host_badges()
            return

        host_id = self._document.project.host_structure_id
        host: Structure | None = self._document.structures.get(host_id) if host_id else None
        self._host_structure_view.set_project_supercell_matrix(self._document.project.supercell_matrix)
        promotable = self._project_service.list_promotable_host_structures(self._document)
        self._host_structure_view.set_promotable_hosts(promotable, host_id)
        self._host_structure_view.set_structure(host)
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
        self._supercell_editor_host_badge.set_active_host(host)

    def _on_active_host_change_requested(self, structure_id_str: str) -> None:
        if self._document is None:
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
        self._refresh_muon_generation_table()
        self._refresh_muon_relaxation_rows()
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
            result = QMessageBox.question(
                self,
                "Host structure mismatch",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if result == QMessageBox.StandardButton.Yes and source_id is not None and self._document is not None:
                try:
                    self._project_service.set_active_host_structure(self._document, source_id)
                    self._refresh_host_structure_view()
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
            result = QMessageBox.question(
                self,
                "Host structure mismatch",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if result == QMessageBox.StandardButton.Yes and source_id is not None and self._document is not None:
                try:
                    self._project_service.set_active_host_structure(self._document, source_id)
                    self._refresh_host_structure_view()
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

    def _check_staged_muon_relaxation_host(self) -> bool:
        """Validate all staged muon-relaxation sites match the active host.

        Returns True if the submission should proceed (no mismatch, or the
        user resolved the mismatch by switching hosts).  Returns False if
        the user cancelled or the mismatch could not be resolved.
        """
        if self._document is None:
            return True
        definition = self._active_muon_relaxation_definition()
        if definition is None or not definition.input_site_ids:
            return True
        mismatch = self._project_service.validate_sites_match_active_host(
            self._document, definition.input_site_ids
        )
        return mismatch is None or self._handle_site_host_mismatch(mismatch)

    def _check_definition_host(self, definition) -> bool:
        """Validate a QC / energy-barriers / NEB definition against the active host.

        Checks structures first (via parent_structure_id chain walk), then
        sites as a defence-in-depth layer.  Returns True if ok to proceed.
        """
        if self._document is None or definition is None:
            return True
        if definition.input_structure_ids:
            struct_mismatch = self._project_service.validate_structures_match_active_host(
                self._document, definition.input_structure_ids
            )
            if struct_mismatch is not None:
                return self._handle_structure_host_mismatch(struct_mismatch)
        if definition.input_site_ids:
            site_mismatch = self._project_service.validate_sites_match_active_host(
                self._document, definition.input_site_ids
            )
            if site_mismatch is not None:
                return self._handle_site_host_mismatch(site_mismatch)
        return True

    def _refresh_magnetic_field_view(self) -> None:
        self._magnetic_field_view.set_document(self._document)
        if self._document is None:
            return
        self._magnetic_field_view.set_form_state(
            self._document.ui_state.get(self._MAGNETIC_FIELD_FORM_KEY, {})
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

    def _on_magnetic_field_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state[self._MAGNETIC_FIELD_FORM_KEY] = state

    def _normalize_magnetic_field_workspace_title(self, title: object) -> str:
        normalized = str(title).strip()
        return normalized or "Magnetic Field Workspace"

    def _replace_magnetic_field_workspace(self, workspace_id: str, updates: Mapping[str, object]) -> bool:
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

    def _on_magnetic_field_workspace_structure_changed(self, workspace_id: str, payload: object) -> None:
        if self._document is None or not isinstance(payload, dict):
            return
        self._replace_magnetic_field_workspace(workspace_id, {"magnetic_structure": payload})
        self._refresh_workflow_stage_icons()

    def _on_magnetic_field_workspace_form_state_changed(self, workspace_id: str, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._replace_magnetic_field_workspace(workspace_id, {"form": state})

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
        default_title = self._normalize_magnetic_field_workspace_title(snapshot.get("title", "Magnetic Field Workspace"))
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
            "magnetic_structure": magnetic_structure if isinstance(magnetic_structure, dict) else None,
            "form": form_state if isinstance(form_state, dict) else {},
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
            (workspace for workspace in workspaces if str(workspace.get("id", "")).strip() == workspace_id),
            None,
        )
        if target is None:
            return
        title = self._normalize_magnetic_field_workspace_title(target.get("title", "Magnetic Field Workspace"))
        if confirm:
            answer = QMessageBox.question(
                self,
                "Delete magnetic field workspace",
                f"Delete magnetic field workspace '{title}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._document.ui_state[self._MAGNETIC_FIELD_WORKSPACES_KEY] = [
            workspace
            for workspace in workspaces
            if str(workspace.get("id", "")).strip() != workspace_id
        ]
        self._sync_magnetic_field_workspace_views()
        self._populate_project_tree()
        route = self._active_route()
        if route is None or self._magnetic_field_workspace_id_from_route(route) == workspace_id:
            self._set_active_route("magnetic_fields")
        self._persist_document_ui_state(
            failure_message="Magnetic field workspace was deleted, but autosave failed."
        )
        self.statusBar().showMessage(f"Deleted magnetic field workspace '{title}'")

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

        self._sync_equivalent_convergence_cutoff_mappings(definition)
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
                max_stress=self._max_stress_for_job(job.id),
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

    def _max_stress_for_job(self, job_id: UUID) -> float | None:
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
        return self._extract_max_stress(result.data)

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

    @staticmethod
    def _extract_max_stress(payload: object) -> float | None:
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

        for key in ("max_stress", "maximum_stress", "max_stress_gpa"):
            value = parse_numeric(payload.get(key))
            if value is not None:
                return value

        stress = payload.get("stress")
        if isinstance(stress, dict):
            for key in ("max", "max_stress", "maximum"):
                value = parse_numeric(stress.get(key))
                if value is not None:
                    return value

        return None
