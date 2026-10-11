"""Primary application window for MuFinder."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
import logging
import os
from pathlib import Path
import posixpath
import re
import shutil
import stat
import threading
import time
from typing import TypeVar
from uuid import UUID, uuid4

import numpy as np
from PySide6.QtCore import QByteArray, QEventLoop, QSize, QSettings, QTimer, Qt, Signal, Slot
from PySide6.QtGui import (
    QActionGroup,
    QBrush,
    QColor,
    QGuiApplication,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
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
)
from mufinder.cluster_setup import ClusterConfig, load_cluster_config, load_cluster_configs
from mufinder.castep_io.gui_metadata import make_block_section, make_keyword_assignment
from mufinder.castep_io.schema import KeywordSpec, canonical_keyword_name, get_keyword_spec
from mufinder.domain.constants import IMPORTED_JOB_TAG
from mufinder.domain.enums import ACTIVE_JOB_STATUSES, FailureReason, JobStatus, WorkflowStage
from mufinder.domain.models import (
    MANUAL_DEFAULT_METHOD,
    CalculationDefinition,
    CalculationRun,
    ComparisonTable,
    Job,
    MuonSite,
    Result,
    Structure,
    utc_now,
)
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
from mufinder.app.keyword_catalog_guard import KeywordCatalogGuard
from mufinder.app.convergence_adapter import convergence_job_name, expand_convergence_jobs
from mufinder.app.convergence_examiner import ConvergenceExaminerDialog
from mufinder.app.convergence_display import (
    format_convergence_criteria,
    format_convergence_method,
    format_cutoff_energy,
    format_kpoint_grid,
)
from mufinder.app.convergence_overview_widget import ConvergenceOverviewWidget
from mufinder.app.convergence_types import (
    ConvergenceConfig,
    SweepPointRunState,
    convergence_config_xc_functional,
)
from mufinder.app.convergence_widget import ConvergenceWidget
from mufinder.app.energy_barriers_widget import (
    EnergyBarriersWidget,
    EnergyBarrierEndpointRow,
    EnergyBarrierPreviewImageRow,
)
from mufinder.app.kpoint_convergence_widget import KPointConvergenceWidget
from mufinder.app.legacy_import_dialog import LegacyImportDialog, start_legacy_scan
from mufinder.services.legacy_import import LegacyFolderScan
from mufinder.services.legacy_import_commit import IMPORT_ACTION, commit_legacy_import
from mufinder.services.legacy_import_removal import (
    detach_rows_from_definition,
    imported_relaxations_in_batch,
    plan_removal,
    remove_imported_relaxations,
)
from mufinder.app.relaxation_adapter import (
    MUON_RELAXATION_CALC_TYPES,
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    canonical_charge_state,
    expand_muon_relaxation_jobs,
    expand_pristine_relaxation_jobs,
    is_muon_relaxation_calc_type,
    muon_charge_state_label,
    muon_charge_state_prefix,
    muon_relaxation_calc_type,
)
from mufinder.app.adapter_common import AdapterExpansion
from mufinder.app.relaxation_types import (
    MuonRelaxationConfig,
    MuonRelaxationRow,
    PristineRelaxationConfig,
    PristineRelaxationRow,
    RelaxationRunState,
)
from mufinder.app.relaxation_analysis_widget import (
    CompletedRelaxationAnalysisSource,
    RelaxationAnalysisJobDialog,
    RelaxationAnalysisWidget,
    default_workspace_state,
)
from mufinder.app.relaxation_widget import PristineRelaxationWidget
from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget
from mufinder.app.continue_relaxation_dialog import ContinueRelaxationDialog
from mufinder.app.resubmit_choice_dialog import (
    ResubmitCandidate,
    ResubmitChoiceDialog,
    ResubmitMode,
)
from mufinder.app.muon_sites_summary_widget import MuonSitesSummaryWidget
from mufinder.app.uep_adapter import expand_uep_jobs
from mufinder.app.uep_types import UepConfig, UepSourceRow
from mufinder.app.uep_wizard import UepAnalysisDialog
from mufinder.app.uep_widget import UEPWidget
from mufinder.app.active_host_badge import ActiveHostBadge
from mufinder.app.host_structure_widget import HostStructureWidget
from mufinder.app.icons import icon, workflow_stage_icon
from mufinder.app.journal_widget import JournalWidget
from mufinder.app.magnetic_field_widget import MagneticFieldWidget
from mufinder.app.muon_generation_widget import MuonGenerationWidget
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.project_defaults_widget import ProjectDefaultsStrip, ProjectDefaultsSummary
# MainWindow uses these, and they stay importable from this module as public
# aliases for one minor release; see docs/compatibility_policy.rst.
from mufinder.app.muon_generation_support import MuonChargeStateDialog
from mufinder.app.muon_generation_support import MuonGenerationThread as _MuonGenerationThread
from mufinder.app.project_home_trail import (
    ConstructionSignWidget,  # noqa: F401 - re-export only
    ProjectHomeTrailWidget,
    ProjectTrailDestination,
    WorkflowDefinition,
)
from mufinder.app.quantum_analysis_dialog import QuantumPhononAnalysisDialog
from mufinder.app.quantum_corrections_widget import (
    QuantumCorrectionsWidget,
)
from mufinder.app.screen_utils import clamp_to_screen, clamped_size
from mufinder.app.supercell_dialog import SupercellEditorDialog
from mufinder.app.submission_dialog import LocalSubmissionDialog
from mufinder.app.theme import (
    THEME_DARK,
    THEME_LIGHT,
    THEME_SYSTEM,
    apply_theme,
    current_palette,
    load_theme_mode,
    save_theme_mode,
    style_section_subtitle,
    style_section_title,
    status_brush,
    use_square_top_table_corners,
)
from mufinder.app.workflow_config_codecs import (
    deserialize_convergence_config,
    deserialize_muon_relaxation_config,
    deserialize_pristine_relaxation_config,
    deserialize_uep_config,
    normalized_supercell_matrix,
    serialize_convergence_config,
    serialize_muon_relaxation_config,
    serialize_pristine_relaxation_config,
    serialize_uep_config,
)
from mufinder.app.workflow_dialogs import (
    EnergyBarrierPathViewerDialog,
    NudgedElasticBandConfigDialog,
    QuantumCorrectionsConfigDialog,
)
from mufinder.app.workflow_parameters import (
    merge_batch_advanced,
    merge_batch_values,
    mixed_parameter_names,
    build_workflow_parameter_surface,
    kpoint_exclusive_hint,
    reorder_workflow_parameter_specs,
    workflow_basic_parameter_names,
)
from mufinder.app.workflow_submission import (
    WorkflowSubmissionCoordinator,
    effective_row_status,
    job_is_resubmittable_after_dismissal,
    workflow_row_state_is_ready,
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
    PristineRelaxationDeps,
    QuantumCorrectionsController,
    QuantumCorrectionsDeps,
    UepController,
    UepDeps,
    WorkflowControllerContext,
    WorkflowJobController,
    merge_controller_result,
    validate_controller_artifacts,
)
from mufinder.app.workflow_controllers.quantum_corrections import (
    QUANTUM_STAGE_PHONON,
    QUANTUM_STAGE_SINGLEPOINT,
)
from mufinder.castep_io.cell_parser import CellParseError, parse_cell_file
from mufinder.castep_io.output_parser import CastepResult
from mufinder.castep_io.param_parser import parse_param_assignments, parse_param_value, run_is_spin_polarized
from mufinder.castep_io.ts_parser import (
    TransitionStateImage,
    TransitionStateParseError,
    parse_transition_state_file,
)
from mufinder.castep_io.types import AtomicPosition, BlockSection, PositionBlock, ValueWithUnit
from mufinder.persistence.codec import to_jsonable
from mufinder.app.cluster_controller import ClusterController
from mufinder.services.cluster_runner import CheckpointProbeResult
from mufinder.services.job_execution_controller import (
    ClusterJobSubmission,
    JobExecutionController,
    PreparedClusterSubmission,
)
from mufinder.services.failure_diagnosis import (
    FailureDiagnosis,
    failure_reason_label,
    failure_reason_summary,
    find_failure_log_files,
    read_log_tail,
)
from mufinder.services.job_artifacts import ParsedJobArtifacts, RetrievalFlags, collect_job_artifacts
from mufinder.services.job_paths import (
    definition_output_directory,
    job_output_directory,
    job_output_files,
    job_seed_name,
)
from mufinder.services.job_results import max_force_for_job, max_stress_for_job, total_energy_for_job
from mufinder.services.job_monitoring import job_staleness
from mufinder.services.job_watchdog import StuckJobWatchdog
from mufinder.services.local_runner import LocalJobRunner
from mufinder.app import workflow_table_utils
from mufinder.app.selection_snapshot import SelectionSnapshotPushButton, resolve_target_keys
from mufinder.app.workflow_table_utils import SelectionFeedback, count_phrase, partial_count_suffix
from mufinder.app.workflow_parameters import restore_structured_value
from mufinder.services.perf_trace import PerfTrace
from mufinder.services.project_service import (
    CONTINUABLE_JOB_STATUSES,
    ProjectDocument,
    ProjectService,
    active_definition,
    job_for_mapping_key,
)
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
)
from mufinder.science.kpoint_grids import equivalent_grid_for_supercell, kpoint_count, most_uniform_grid
from mufinder.science.quantum_phonons import PhononParseError
from mufinder.science.relaxation_analysis import folded_muon_position_from_structure, grouped_relative_energies
from mufinder.science.site_generation import SiteGenerationError, lattice_cart_to_frac
from mufinder.science.supercell import build_supercell_structure
from mufinder.science.uep import UepError, UepPotentialGrid, deserialize_potential_grid, serialize_potential_grid
from mufinder.science.structure_comparison import DftComparisonInput, compare_structures
from mufinder.science.structure_io import StructureLoadError
from mufinder.workflow import StructureBridgeError, structure_to_spec
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
    TRANSITION_STATE_SEARCH_SET_CALC_TYPE,
)
from mufinder.workflow.naming import job_name
from mufinder.workflow.quantum import (
    QUANTUM_SINGLEPOINT_CALC_TYPE,
    QUANTUM_PHONON_CALC_TYPE,
    quantum_source_eligibility,
)
from mufinder.workflow.renderer import render_job
from mufinder.workflow.types import CellTemplate, InputTemplate, ParamTemplate


_LOG = logging.getLogger(__name__)

# Stopped jobs (walltime, orphaned) are resubmittable like failed ones: the
# submit script is regenerated, so a longer walltime takes effect.
_RESUBMITTABLE_JOB_STATES = {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
# Stage-1 singlepoint statuses from which a Quantum Corrections row may be rewound
# to stage 1 while no phonon job exists. ``COMPLETED`` is the "redo stage 1 with
# different parameters" case; the three terminal-recoverable statuses are the
# failure escape hatch (the same set ``reset_failed_job`` accepts, spelled out
# here because this path deletes job records rather than resetting them).
_QUANTUM_STAGE_ONE_RESETTABLE_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
)
_VISIBLE_DASHBOARD_STATES = ACTIVE_JOB_STATUSES | {JobStatus.STOPPED} | _RESUBMITTABLE_JOB_STATES
_RELAXATION_ANALYSIS_STATE_KEY = "relaxation_analysis_workspaces"
_WorkflowKey = TypeVar("_WorkflowKey")
_STRUCTURE_CONTROLLED_CELL_KEYWORDS = {"lattice_cart", "lattice_abc", "positions_frac", "positions_abs"}
_QUANTUM_CONFIG_KEYWORD_NAMES = workflow_basic_parameter_names()
# Climbing image is locked on (see the NEB Configure dialog's locked_values): a barrier
# calculation needs the top image driven onto the saddle, so it is not user-editable here.
_ENERGY_BARRIER_NEB_KEYWORD_NAMES = workflow_basic_parameter_names(
    "tssearch_max_path_points",
    "tssearch_neb_spring_constant",
    "tssearch_neb_tangent_mode",
    "tssearch_neb_method",
    "tssearch_neb_max_iter",
    "tssearch_neb_normed",
    "tssearch_force_tol",
)
# Default NEB image count (TSSEARCH_MAX_PATH_POINTS). Odd, as climbing-image NEB requires.
# 11 gives a finer band than the CASTEP minimum: a smoother barrier path and a lower residual
# force at the climbing image (7 images left a ~0.08 eV/Ang plateau just above the force tol).
_ENERGY_BARRIER_NEB_DEFAULT_IMAGES = 11
# Editable NEB force-convergence tolerance (TSSEARCH_FORCE_TOL). Default mirrors CASTEP's own
# default (0.05 eV/Ang via GEOM_FORCE_TOL). Exposed so users can loosen it for stubborn bands whose
# barrier is converged but whose max force component plateaus above 0.05. The keyword's canonical
# unit is H/BOHR; we surface it in eV/Ang (what the .castep output reports) for legibility.
_ENERGY_BARRIER_NEB_FORCE_TOL_UNIT = "eV/ang"
_ENERGY_BARRIER_NEB_DEFAULT_FORCE_TOL = ValueWithUnit(0.05, _ENERGY_BARRIER_NEB_FORCE_TOL_UNIT)
_QUANTUM_GUI_TO_METHOD = {
    "FINITEDISPLACEMENT": "finite_displacement",
    "DFPT": "dfpt",
    "LINEARRESPONSE": "linear_response",
}
_QUANTUM_DEFAULT_ELEC_ENERGY_TOL = ValueWithUnit(1e-9, "eV")
_QUANTUM_METHOD_TO_GUI = {value: key for key, value in _QUANTUM_GUI_TO_METHOD.items()}
# Stage identifiers now live in the quantum-corrections controller (single source
# of truth); aliased here so the still-inline stage helpers keep their names.
_QUANTUM_STAGE_SINGLEPOINT = QUANTUM_STAGE_SINGLEPOINT
_QUANTUM_STAGE_PHONON = QUANTUM_STAGE_PHONON


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


def job_is_imported(job: Job) -> bool:
    """Return whether ``job`` records a calculation performed outside MuFinder.

    Imported jobs must never be reset, resubmitted, or continued: MuFinder did
    not run them, and their directories hold the only copy of the imported
    results together with the archived originals.
    """

    return IMPORTED_JOB_TAG in (job.tags or ())


def _format_energy(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g} eV"
    except (TypeError, ValueError):
        return ""


def _format_relative_energy(value: float | None) -> str:
    """Format a site energy relative to its comparable minimum, as in the Muon Sites summary."""

    return "" if value is None else f"dE {float(value):.5f} eV"


def _neb_barrier_summaries(data: Mapping[str, object]) -> tuple[str, str, str]:
    """Format (forward barrier, reverse barrier, endpoint ΔE) from a NEB result payload.

    Both barriers and ΔE derive from the persisted per-image energies; results
    that predate ``image_energies_eV`` fall back to the stored forward barrier
    with blank reverse/ΔE columns.
    """

    energies_payload = data.get("image_energies_eV")
    energies: list[float] = []
    if isinstance(energies_payload, (list, tuple)):
        try:
            energies = [float(value) for value in energies_payload]
        except (TypeError, ValueError):
            energies = []
    if len(energies) >= 2:
        peak = max(energies)
        return (
            _format_energy(peak - energies[0]),
            _format_energy(peak - energies[-1]),
            _format_energy(energies[-1] - energies[0]),
        )
    return _format_energy(data.get("barrier_height_eV")), "", ""


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


class MainWindow(QMainWindow):
    """Main shell containing project navigation and job dashboard."""

    _MAIN_WINDOW_GEOMETRY_KEY = "ui/main_window_geometry"
    _LAST_DIALOG_DIR_KEY = "ui/last_dialog_dir"
    _LAST_DIALOG_DIR_BY_CONTEXT_PREFIX = "ui/last_dialog_dir_by_context"
    _RECENT_PROJECTS_KEY = "ui/recent_projects"
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
    _JOB_LABEL_COLUMN = 0
    _JOB_WORKFLOW_COLUMN = 1
    _JOB_CALC_TYPE_COLUMN = 2
    _JOB_STATUS_COLUMN = 3
    _JOB_LAST_CHECKED_COLUMN = 4
    _JOB_CLUSTER_COLUMN = 5
    _JOB_REMOTE_ID_COLUMN = 6
    _JOB_RUNTIME_COLUMN = 7
    _JOB_PROCESSES_COLUMN = 8
    _JOB_SUBMIT_COLUMN = 9
    _JOB_RETRY_RETRIEVAL_COLUMN = 10
    _JOB_CANCEL_COLUMN = 11
    _JOB_INSPECT_COLUMN = 12
    _JOB_RESET_COLUMN = 13
    _JOB_DISMISS_COLUMN = 14
    _PROJECT_TREE_MIN_WIDTH = 220
    _PROJECT_TREE_MAX_WIDTH = 300
    _ACTIVE_CONVERGENCE_DEFINITION_KEY = ConvergenceCutoffController.active_key
    _ACTIVE_KPOINT_DEFINITION_KEY = ConvergenceKpointController.active_key
    _ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY = PristineRelaxationController.active_key
    _ACTIVE_MUON_RELAXATION_DEFINITION_KEY = MuonRelaxationController.active_key
    _ACTIVE_UEP_DEFINITION_KEY = UepController.active_key
    _CONVERGENCE_EXAMINER_STATE_KEY = "convergence_examiner_state"
    _KPOINT_EXAMINER_STATE_KEY = "kpoint_examiner_state"
    _UEP_ANALYSIS_STATES_KEY = "uep_analysis_states"
    _UEP_POTENTIAL_GRID_RESULT_TYPE = "uep_potential_grid"
    _MAGNETIC_FIELD_FORM_KEY = "magnetic_field_form"
    _MAGNETIC_FIELD_WORKSPACES_KEY = "magnetic_field_workspaces"
    _MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX = "magnetic_fields/workspace/"
    _RELAXATION_ANALYSIS_WORKSPACE_ROUTE_PREFIX = "muon_analysis/workspace/"
    _CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY = "convergence_overview_snapshots"
    _MUON_SITES_SUMMARY_SNAPSHOTS_KEY = "muon_sites_summary_snapshots"
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
        # Monotonic timestamp (time.monotonic(), never wall-clock) of the last
        # completed flush, and of the deadline currently armed on
        # _dashboard_flush_timer. Both None until the first request. Used by
        # _request_dashboard_flush to enforce _DASHBOARD_MIN_FLUSH_INTERVAL_MS
        # without re-arming (and thereby postponing) an already-pending deadline.
        self._dashboard_last_flush_at: float | None = None
        self._dashboard_flush_deadline: float | None = None
        # Jobs stopped by the most recent orphaned-local-job reconciliation on
        # project open; surfaced as a status-bar message by _open_project_from_path.
        self._reconciled_orphaned_local_jobs: list[Job] = []
        self._dashboard_flush_timer = QTimer(self)
        self._dashboard_flush_timer.setSingleShot(True)
        self._dashboard_flush_timer.setInterval(self._DASHBOARD_FLUSH_DEBOUNCE_MS)
        self._dashboard_flush_timer.timeout.connect(self._flush_dashboard)
        # Active-job watchdog: every minute, look for jobs left active with no
        # runner tracking them (a dead local QProcess, an untracked cluster job)
        # and stop them so they cannot linger in the dashboard forever. Findings
        # are two-tick confirmed inside StuckJobWatchdog to ride out in-flight
        # runner signals. The timer runs unconditionally; the handler is cheap
        # when there are no active jobs.
        self._stuck_job_watchdog = StuckJobWatchdog()
        self._job_watchdog_timer = QTimer(self)
        self._job_watchdog_timer.setInterval(60_000)
        self._job_watchdog_timer.timeout.connect(self._on_job_watchdog_tick)
        self._job_watchdog_timer.start()
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
        self.resize(clamped_size(self, 1280, 840))
        saved_geometry = self._settings.value(self._MAIN_WINDOW_GEOMETRY_KEY, QByteArray())
        if isinstance(saved_geometry, QByteArray) and not saved_geometry.isEmpty() and self.restoreGeometry(saved_geometry):
            clamp_to_screen(self)

        self._project_tree = QTreeWidget()
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
            self._build_pristine_relaxation_deps(),
            self._pristine_relaxation_view,
        )
        self._workflow_controllers.append(self._pristine_relaxation_controller)
        self._pristine_relaxation_view.submit_local_requested.connect(
            lambda values: self._pristine_relaxation_controller.submit(values, target="local")
        )
        self._pristine_relaxation_view.submit_cluster_requested.connect(
            lambda values: self._pristine_relaxation_controller.submit(values, target="cluster")
        )
        self._pristine_relaxation_view.generate_comparison_requested.connect(
            self._on_generate_comparison_table_requested
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
        self._pristine_relaxation_view.continue_relaxation_requested.connect(
            self._on_continue_relaxation_requested_from_pristine_relaxation
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
        self._muon_generation_view.cluster_positions_requested.connect(self._on_cluster_muon_positions_requested)
        self._muon_generation_view.delete_candidates_requested.connect(self._on_delete_muon_candidates_requested)
        self._muon_generation_view.create_jobs_requested.connect(self._on_create_muon_relaxation_jobs_requested)
        self._muon_generation_view.form_state_changed.connect(self._on_muon_form_state_changed)
        self._import_legacy_action = None
        self._remove_imported_action = None
        self._legacy_scan_host: Structure | None = None
        self._legacy_scan_root: Path | None = None
        self._muon_relaxation_view = MuonRelaxationWidget()
        self._muon_relaxation_view.configure_requested.connect(self._on_muon_relaxation_configured)
        self._muon_relaxation_view.form_state_changed.connect(self._on_muon_relaxation_form_state_changed)
        for strip in self._project_defaults_strips():
            strip.xc_functional_change_requested.connect(
                self._on_project_xc_functional_change_requested
            )
            strip.cutoff_energy_change_requested.connect(
                self._on_project_cutoff_change_requested
            )
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
        self._muon_relaxation_view.continue_relaxation_requested.connect(
            self._on_continue_relaxation_requested_from_muon_relaxation
        )
        self._muon_relaxation_view.analysis_workspace_requested.connect(
            self._on_create_relaxation_analysis_workspace_requested
        )
        self._relaxation_analysis_view = RelaxationAnalysisWidget()
        self._relaxation_analysis_view.state_changed.connect(self._on_relaxation_analysis_state_changed)
        self._relaxation_analysis_view.add_sites_requested.connect(self._on_relaxation_analysis_add_sites_requested)
        self._relaxation_analysis_view.pin_summary_requested.connect(self._on_muon_sites_pin_requested)
        self._relaxation_analysis_view.delete_requested.connect(self._delete_relaxation_analysis_workspace)
        self._relaxation_analysis_view.current_workspace_changed.connect(
            self._on_relaxation_analysis_current_workspace_changed
        )
        self._muon_sites_summary_view = MuonSitesSummaryWidget()
        self._muon_sites_summary_view.open_analysis_requested.connect(self._on_open_muon_analysis_requested)
        self._muon_sites_summary_view.open_analysis_root_requested.connect(
            lambda: self._set_active_route("muon_analysis")
        )
        self._muon_sites_summary_view.focus_site_requested.connect(self._on_focus_muon_site_requested)
        self._muon_sites_summary_view.unpin_requested.connect(self._on_unpin_muon_sites_requested)
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
        # Quantum-corrections controller: the two-stage (singlepoint → phonon)
        # migration. Built here (after its view exists) and appended to the
        # registry so completed castep_quantum_singlepoint / castep_quantum_phonon
        # jobs route through its build_result (phonon → ZPE/eigenvalue/witness
        # payload; singlepoint → empty result). The pre_submit_check runs the
        # shared ``_check_definition_host`` guard; task farming stays in the shared
        # submit backend (not this controller). Its validate_artifacts fails a
        # phonon job whose ``.phonon`` is missing or unparseable.
        self._quantum_corrections_controller = QuantumCorrectionsController(
            self._build_workflow_controller_context(),
            self._build_quantum_corrections_deps(),
            self._quantum_corrections_view,
        )
        self._workflow_controllers.append(self._quantum_corrections_controller)
        self._quantum_corrections_view.submit_local_requested.connect(
            lambda values: self._quantum_corrections_controller.submit(
                values, target="local"
            )
        )
        self._quantum_corrections_view.submit_cluster_requested.connect(
            lambda values: self._quantum_corrections_controller.submit(
                values, target="cluster"
            )
        )
        self._quantum_corrections_view.reset_stage_requested.connect(
            self._on_reset_quantum_corrections_to_stage_one
        )
        # Quantum rows reset through "Reset to stage 1" instead of a failed-job reset.
        self._connect_workflow_row_actions(
            self._quantum_corrections_view, self._quantum_corrections_controller, reset=False
        )
        self._quantum_corrections_view.analysis_requested.connect(
            self._on_launch_quantum_phonon_analysis
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
        self._main_stack.addWidget(self._muon_sites_summary_view)
        self._main_stack.addWidget(self._magnetic_field_view)
        self._main_stack.addWidget(self._quantum_corrections_view)
        self._main_stack.addWidget(self._journal_view)

        self._jobs_table = QTableWidget(0, 15)
        self._jobs_table.setHorizontalHeaderLabels(
            [
                "Label",
                "Workflow",
                "Calc Type",
                "Status",
                "Last Checked",
                "Cluster",
                "Remote Job ID",
                "Runtime",
                "Processes",
                "Resubmit",
                "Retry retrieval",
                "Cancel",
                "Inspect",
                "Reset",
                "Dismiss",
            ]
        )
        use_square_top_table_corners(self._jobs_table)
        self._jobs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._jobs_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._jobs_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._jobs_table.customContextMenuRequested.connect(self._open_jobs_table_context_menu)
        # Shown only while two or more rows are selected, so it costs no height
        # otherwise; it is what tells the user a row button acts on them all.
        self._jobs_selection_hint = QLabel("")
        self._jobs_selection_hint.setObjectName("hintText")
        self._jobs_selection_hint.setContentsMargins(8, 2, 8, 2)
        self._jobs_selection_hint.hide()
        self._jobs_selection_feedback = SelectionFeedback(
            self._jobs_table,
            multi_row_hint=self._jobs_selection_hint,
            multi_row_hint_text=lambda count: (
                f"{count} selected \u00b7 row buttons apply to all selected"
            ),
        )
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
        self._keyword_catalog_guard = KeywordCatalogGuard(
            parent=self,
            settings=self._settings,
            open_setup=self._open_castep_setup_dialog,
        )
        self.statusBar().addPermanentWidget(self._keyword_catalog_guard.status_button)
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
        jobs_panel = QWidget(jobs_dock)
        jobs_panel_layout = QVBoxLayout(jobs_panel)
        jobs_panel_layout.setContentsMargins(0, 0, 0, 0)
        jobs_panel_layout.setSpacing(0)
        jobs_panel_layout.addWidget(self._jobs_selection_hint)
        jobs_panel_layout.addWidget(self._jobs_table, 1)
        jobs_dock.setWidget(jobs_panel)
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
        self._settings.setValue(self._MAIN_WINDOW_GEOMETRY_KEY, self.saveGeometry())
        job_runtime_timer = getattr(self, "_job_runtime_timer", None)
        if job_runtime_timer is not None:
            job_runtime_timer.stop()
        dashboard_flush_timer = getattr(self, "_dashboard_flush_timer", None)
        if dashboard_flush_timer is not None:
            dashboard_flush_timer.stop()
        job_watchdog_timer = getattr(self, "_job_watchdog_timer", None)
        if job_watchdog_timer is not None:
            job_watchdog_timer.stop()
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
            # Generation mutates the project document mid-run, so it cannot be
            # interrupted or abandoned — but a bare wait() froze the window
            # into "Not Responding" for the remainder of a long generation.
            # Pump the event loop behind a busy dialog instead; the window
            # modality prevents re-entrant close clicks.
            if not muon_generation_thread.isFinished():
                progress = QProgressDialog("Finishing muon site generation…", None, 0, 0, self)
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

        self._import_legacy_action = file_menu.addAction("Import Legacy Relaxations...")
        self._import_legacy_action.triggered.connect(self._on_import_legacy_relaxations)

        self._remove_imported_action = file_menu.addAction("Remove Imported Relaxations...")
        self._remove_imported_action.triggered.connect(self._on_remove_imported_relaxations)

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
        self._refresh_ui_after_project_change()

        integrity_warnings = self._document.ui_state.get("_integrity_warnings", [])
        if integrity_warnings:
            QMessageBox.warning(
                self,
                "Integrity warnings",
                "Project loaded with integrity warnings:\n\n"
                + "\n".join(f"- {warning}" for warning in integrity_warnings),
            )
        reconciled = self._reconciled_orphaned_local_jobs
        if reconciled:
            count = len(reconciled)
            self.statusBar().showMessage(
                f"{count} local job(s) were stopped because MuFinder closed "
                "while they were in progress."
            )
        else:
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
        self._dashboard_flush_deadline = None
        self._dashboard_last_flush_at = time.monotonic()
        try:
            self._project_service.save_project(self._document)
        except ValueError as exc:
            QMessageBox.critical(self, "Save blocked", str(exc))
            return
        self.statusBar().showMessage("Project saved")

    def _open_castep_setup_dialog(self) -> None:
        dialog = CastepSetupDialog(settings=self._settings, parent=self)
        accepted = dialog.exec() == dialog.DialogCode.Accepted
        # A keyword catalog generated or imported in the dialog is installed
        # immediately, whether or not the dialog is then accepted.
        self._keyword_catalog_guard.refresh()
        if not accepted:
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

    def prompt_for_keyword_catalog_if_needed(self) -> None:
        """Explain a missing CASTEP keyword catalog once the window is up."""

        self._keyword_catalog_guard.prompt_at_startup_if_needed()

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

        # Clear then re-register tracked cluster jobs. Both are queued onto the
        # cluster worker thread and return immediately (never blocking the GUI,
        # even mid-retrieval); FIFO ordering guarantees the clear runs before the
        # re-registrations, and the GUI-side tracked-id mirror is cleared eagerly
        # inside clear_tracked_jobs so the watchdog observes the drop at once.
        self._cluster_runner.clear_tracked_jobs()
        self._sync_cluster_tracking_from_document()

        # Reconcile local jobs that were still in progress when MuFinder closed.
        # This must run BEFORE the jobs table is rebuilt below so the user never
        # sees phantom "running" rows for local jobs with no live process.
        self._reconciled_orphaned_local_jobs = (
            self._project_service.reconcile_orphaned_local_jobs(self._document)
        )
        if self._reconciled_orphaned_local_jobs:
            self._request_dashboard_flush(
                failure_message=(
                    "Reconciled orphaned local jobs, but saving the project failed."
                )
            )

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
        self._convergence_controller.refresh_view()
        self._kpoint_convergence_view.set_form_state(
            self._document.ui_state.get("kpoint_convergence_form", {})
        )
        self._refresh_kpoint_convergence_simulation_cell()
        self._kpoint_convergence_controller.refresh_view()
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
        self._refresh_convergence_project_defaults()
        self._refresh_workflow_defaults(self._pristine_relaxation_view)
        self._pristine_relaxation_controller.refresh_view()
        self._refresh_workflow_defaults(self._muon_relaxation_view)
        self._refresh_muon_relaxation_rows()
        self._refresh_relaxation_analysis_context()
        self._refresh_workflow_defaults(self._uep_view, supercells=False)
        self._refresh_uep_rows()
        self._refresh_uep_source_structures()
        self._refresh_workflow_supercell_options(self._uep_view)
        self._refresh_convergence_overview()
        # Keep a visible summary current on project change without rebuilding its
        # two PyVista viewers off-screen when the page is not showing (route
        # dispatch and pin/unpin refresh it unconditionally when it is shown).
        self._refresh_muon_sites_summary(only_if_visible=True)
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

        # Recompute point_spacings so that stale values persisted with an older
        # formula are always refreshed on load — even for projects adopted
        # before the fix. Use the cell the sweep ran on (recorded with the
        # snapshot), which may be a saved supercell; older snapshots without it
        # fall back to the host.
        cell = self._kpoint_snapshot_cell(snapshot)
        if cell is not None:
            try:
                snapshot = ConvergenceExaminerDialog.recompute_spacings_in_snapshot(
                    snapshot, cell.lattice_vectors
                )
            except Exception:
                pass  # Safety: fall back to stored spacings on any unexpected error

        return snapshot

    def _kpoint_snapshot_cell(self, snapshot: Mapping[str, object]) -> Structure | None:
        """Return the structure a k-point overview snapshot's sweep ran on, else the host."""

        if self._document is None:
            return None
        raw_cell_id = snapshot.get("simulation_cell_structure_id")
        if isinstance(raw_cell_id, str) and raw_cell_id.strip():
            try:
                cell = self._document.structures.get(UUID(raw_cell_id.strip()))
            except ValueError:
                cell = None
            if cell is not None:
                return cell
        host_id = self._document.project.host_structure_id
        return self._document.structures.get(host_id) if host_id is not None else None

    def _cutoff_overview_summary_rows(self) -> list[tuple[str, str]]:
        if self._document is None:
            return [
                ("Project default", "Not set"),
                ("Functional", "Not recorded"),
                ("Method", "Not set"),
                ("Criteria", "Not set"),
            ]
        return [
            ("Project default", format_cutoff_energy(self._document.project.recommended_cutoff_energy)),
            (
                "Functional",
                self._document.project.recommended_cutoff_xc_functional or "Not recorded",
            ),
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
                ("Functional", "Not recorded"),
                ("Method", "Not set"),
                ("Criteria", "Not set"),
            ]
        return [
            ("Project default", format_kpoint_grid(self._document.project.recommended_kpoint_grid)),
            ("Cutoff default", format_cutoff_energy(self._document.project.recommended_cutoff_energy)),
            (
                "Functional",
                self._document.project.recommended_kpoint_xc_functional or "Not recorded",
            ),
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

        definition = self._convergence_controller.active_definition()
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
    def _muon_relaxation_job_for_table_row(
        document: ProjectDocument,
        definition,
        label: str,
        charge_state: str,
    ) -> Job | None:
        keys = (f"{muon_charge_state_prefix(charge_state)}:{label}", label)
        calc_type = muon_relaxation_calc_type(charge_state) if charge_state else None
        for key in keys:
            job = job_for_mapping_key(document, definition, key, calc_type=calc_type)
            if job is not None and is_muon_relaxation_calc_type(job.calc_type):
                return job
        return None

    @staticmethod
    def _latest_run_for_document_job(document: ProjectDocument, job_id: UUID) -> CalculationRun | None:
        runs = [run for run in document.runs.values() if run.job_id == job_id]
        return max(runs, key=lambda run: run.run_index) if runs else None

    @staticmethod
    def _workflow_muon_relaxations_complete(document: ProjectDocument) -> bool:
        definition = active_definition(
            document,
            MuonRelaxationController.stage,
            calc_type=MuonRelaxationController.definition_calc_type,
            active_key=MuonRelaxationController.active_key,
        )
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

    def _relaxation_analysis_workspace_route(self, workspace_id: str) -> str:
        return f"{self._RELAXATION_ANALYSIS_WORKSPACE_ROUTE_PREFIX}{workspace_id}"

    def _relaxation_analysis_workspace_id_from_route(self, route: object) -> str | None:
        if not isinstance(route, str) or not route.startswith(self._RELAXATION_ANALYSIS_WORKSPACE_ROUTE_PREFIX):
            return None
        workspace_id = route[len(self._RELAXATION_ANALYSIS_WORKSPACE_ROUTE_PREFIX) :].strip()
        return workspace_id or None

    @staticmethod
    def _relaxation_analysis_workspace_clustered(document: ProjectDocument, workspace_id: str) -> bool:
        workspaces = document.ui_state.get(_RELAXATION_ANALYSIS_STATE_KEY)
        if not isinstance(workspaces, list):
            return False
        for workspace in workspaces:
            if isinstance(workspace, dict) and str(workspace.get("id", "")) == workspace_id:
                clustering = workspace.get("clustering")
                return isinstance(clustering, dict) and clustering.get("result") is not None
        return False

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
        analysis_workspaces = tuple(
            WorkflowDefinition(
                key=f"muon_analysis_workspace_{workspace['id']}",
                title=str(workspace.get("name", "")).strip() or "Relaxation analysis",
                route=self._relaxation_analysis_workspace_route(str(workspace["id"])),
                completion_goal="Apply clustering to this analysis.",
                completion_check=(
                    lambda document, workspace_id=str(workspace["id"]): (
                        self._relaxation_analysis_workspace_clustered(document, workspace_id)
                    )
                ),
            )
            for workspace in self._relaxation_analysis_workspaces()
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
                route="muon_sites",
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
                        children=analysis_workspaces,
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
            self._main_stack.setCurrentWidget(self._home_page)
            self._sync_home_content_page()
        else:
            self._project_tree.setCurrentItem(selected_item)
        self._refresh_workflow_stage_icons()
        self._sync_project_home_trail()

    def _refresh_jobs_table(self) -> None:
        if self._document is None:
            self._jobs_table.setRowCount(0)
            self._job_runtime_timer.stop()
            self._cluster_elapsed_seconds_by_job = {}
            self._cluster_elapsed_checked_at = None
            self._convergence_view.set_runtime_states({})
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

        now = utc_now()
        desired_jobs = [
            job
            for job in self._document.jobs.values()
            if job.status in _VISIBLE_DASHBOARD_STATES and job.dismissed_at is None
        ]
        self._sync_jobs_table_rows(desired_jobs, now=now)

        self._configure_jobs_table_columns()
        self._sync_job_runtime_timer()

        self._convergence_controller.refresh_view()
        self._kpoint_convergence_controller.refresh_view()
        self._pristine_relaxation_controller.refresh_view()
        self._refresh_muon_relaxation_rows()
        self._refresh_uep_rows()
        self._refresh_workflow_stage_icons()
        self._sync_project_home_trail()

    def _job_button_specs(self, job: Job) -> tuple[tuple[int, str, str, bool], ...]:
        """Return ``(column, text, action_kind, enabled)`` for each action button.

        Single source of truth shared by row insertion and in-place updates so
        the button set, labels, dispatch kinds, and enable rules stay identical.
        """

        return (
            (self._JOB_SUBMIT_COLUMN, "Resubmit", "resubmit", job.status in _RESUBMITTABLE_JOB_STATES),
            (
                self._JOB_RETRY_RETRIEVAL_COLUMN,
                "Retry retrieval",
                "retry_retrieval",
                self._can_retry_retrieval(job),
            ),
            (self._JOB_CANCEL_COLUMN, "Cancel", "cancel", job.status in ACTIVE_JOB_STATUSES),
            (
                self._JOB_INSPECT_COLUMN,
                "Inspect",
                "inspect",
                job.status in {JobStatus.FAILED, JobStatus.STOPPED},
            ),
            (
                self._JOB_RESET_COLUMN,
                "Reset",
                "reset",
                job.status in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED},
            ),
            (
                self._JOB_DISMISS_COLUMN,
                "Dismiss",
                "dismiss",
                job.status in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED},
            ),
        )

    @staticmethod
    def _jobs_table_row_job_id(label_item) -> UUID | None:
        """Return the job id stashed on a label item's UserRole, or ``None``."""

        if label_item is None:
            return None
        try:
            return UUID(str(label_item.data(Qt.ItemDataRole.UserRole)))
        except (TypeError, ValueError, AttributeError):
            return None

    def _sync_jobs_table_rows(self, desired_jobs: list[Job], *, now) -> None:
        """Reconcile visible dashboard rows against ``desired_jobs`` by job id.

        Rows whose job disappeared are removed; brand-new jobs get a freshly
        built row inserted at the correct position; rows whose job id is
        unchanged are updated in place so their ``QTableWidgetItem`` and
        ``QPushButton`` cell widgets are never recreated (no widget churn, no
        stale button closures). A row is never repurposed for a different job
        id — mismatches are removed and re-inserted so each button's captured
        job id stays correct.
        """

        desired_ids = [job.id for job in desired_jobs]

        # Remove rows whose job is no longer desired. Iterate top-down so row
        # indexes stay valid as we delete.
        desired_id_set = set(desired_ids)
        self._jobs_table.setUpdatesEnabled(False)
        try:
            for row in range(self._jobs_table.rowCount() - 1, -1, -1):
                existing_id = self._jobs_table_row_job_id(
                    self._jobs_table.item(row, self._JOB_LABEL_COLUMN)
                )
                if existing_id is None or existing_id not in desired_id_set:
                    self._jobs_table.removeRow(row)
        finally:
            self._jobs_table.setUpdatesEnabled(True)

        # Walk the desired order; align each row to the expected job id,
        # inserting new rows and evicting mismatched rows as needed.
        for target_row, job in enumerate(desired_jobs):
            current_id = self._jobs_table_row_job_id(
                self._jobs_table.item(target_row, self._JOB_LABEL_COLUMN)
            )
            if current_id == job.id:
                self._update_jobs_table_row(target_row, job, now=now)
                continue

            # Either the slot is empty/beyond the current rows, or it holds a
            # different job id (order changed / new job). A row must never be
            # reused for a different job id, so insert a fresh row here. Any row
            # that previously held this job (further down) is left to be handled
            # when its own turn comes; but because a mismatched row can only be
            # a not-yet-visited desired job or removed above, inserting keeps
            # closures correct.
            self._jobs_table.setUpdatesEnabled(False)
            try:
                self._insert_jobs_table_row(target_row, job, now=now)
            finally:
                self._jobs_table.setUpdatesEnabled(True)

        # Trim any trailing rows left over (e.g. when the desired list shrank
        # but the removal pass above could not have caught reordered leftovers).
        if self._jobs_table.rowCount() > len(desired_jobs):
            self._jobs_table.setUpdatesEnabled(False)
            try:
                while self._jobs_table.rowCount() > len(desired_jobs):
                    self._jobs_table.removeRow(self._jobs_table.rowCount() - 1)
            finally:
                self._jobs_table.setUpdatesEnabled(True)

    def _insert_jobs_table_row(self, row: int, job: Job, *, now) -> None:
        """Insert a fully built dashboard row for ``job`` at ``row``.

        Creates every ``QTableWidgetItem`` and action ``QPushButton`` once. Each
        button captures ``job.id`` in its click closure, so this row must only
        ever be used for this job (callers guarantee that via id-keyed
        reconciliation).
        """

        self._jobs_table.insertRow(row)

        label_item = QTableWidgetItem(job.label)
        label_item.setData(Qt.ItemDataRole.UserRole, str(job.id))
        self._jobs_table.setItem(row, self._JOB_LABEL_COLUMN, label_item)
        self._jobs_table.setItem(row, self._JOB_WORKFLOW_COLUMN, QTableWidgetItem(job.workflow_stage.value))
        self._jobs_table.setItem(row, self._JOB_CALC_TYPE_COLUMN, QTableWidgetItem(job.calc_type))

        status_item = QTableWidgetItem()
        self._apply_job_status_cell(status_item, job)
        self._jobs_table.setItem(row, self._JOB_STATUS_COLUMN, status_item)

        last_checked_item = QTableWidgetItem()
        self._apply_last_checked_cell(last_checked_item, job, now=now)
        self._jobs_table.setItem(row, self._JOB_LAST_CHECKED_COLUMN, last_checked_item)

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

        for column, text, kind, enabled in self._job_button_specs(job):
            button = self._make_job_action_button(text, job_id=job.id, enabled=enabled)
            button.clicked.connect(
                lambda _checked=False, jid=job.id, btn=button, action=kind: self._dispatch_dashboard_action(
                    btn, jid, action
                )
            )
            self._jobs_table.setCellWidget(row, column, button)

    def _update_jobs_table_row(self, row: int, job: Job, *, now) -> None:
        """Refresh an existing row in place without recreating items or widgets.

        Updates cell texts, the status foreground brush, the last-checked cell,
        the runtime text, and each button's enabled state. The label item's
        job-id UserRole and every button widget (with its captured closure) are
        left untouched, so button clicks keep dispatching the correct job id.
        """

        label_item = self._jobs_table.item(row, self._JOB_LABEL_COLUMN)
        if label_item is not None:
            label_item.setText(job.label)

        self._set_row_item_text(row, self._JOB_WORKFLOW_COLUMN, job.workflow_stage.value)
        self._set_row_item_text(row, self._JOB_CALC_TYPE_COLUMN, job.calc_type)

        status_item = self._jobs_table.item(row, self._JOB_STATUS_COLUMN)
        if status_item is not None:
            self._apply_job_status_cell(status_item, job)

        last_checked_item = self._jobs_table.item(row, self._JOB_LAST_CHECKED_COLUMN)
        if last_checked_item is not None:
            self._apply_last_checked_cell(last_checked_item, job, now=now)

        cluster_name = "local" if job.execution_target == "local" else job.execution_target
        self._set_row_item_text(row, self._JOB_CLUSTER_COLUMN, cluster_name)
        self._set_row_item_text(row, self._JOB_REMOTE_ID_COLUMN, job.remote_job_id or "")
        self._set_row_item_text(row, self._JOB_RUNTIME_COLUMN, self._job_runtime_text(job))
        self._set_row_item_text(
            row,
            self._JOB_PROCESSES_COLUMN,
            "" if job.num_processes is None else str(job.num_processes),
        )

        for column, _text, _kind, enabled in self._job_button_specs(job):
            button = self._jobs_table.cellWidget(row, column)
            if button is not None:
                button.setEnabled(enabled)

    def _set_row_item_text(self, row: int, column: int, text: str) -> None:
        """Set a cell's text in place, or create the item if it is missing."""

        item = self._jobs_table.item(row, column)
        if item is None:
            self._jobs_table.setItem(row, column, QTableWidgetItem(text))
        else:
            item.setText(text)

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
            self._JOB_LAST_CHECKED_COLUMN,
            self._JOB_RUNTIME_COLUMN,
            self._JOB_PROCESSES_COLUMN,
            self._JOB_SUBMIT_COLUMN,
            self._JOB_RETRY_RETRIEVAL_COLUMN,
            self._JOB_CANCEL_COLUMN,
            self._JOB_INSPECT_COLUMN,
            self._JOB_RESET_COLUMN,
            self._JOB_DISMISS_COLUMN,
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

    @classmethod
    def _apply_job_status_cell(cls, item: QTableWidgetItem, job: Job) -> None:
        """Show the job status, with the failure cause for failed/stopped jobs.

        The cell reads e.g. ``stopped · walltime``; its tooltip explains the
        cause and repeats the recorded failure message, so the reason is
        visible without opening Inspect.
        """

        text = job.status.value
        tooltip = ""
        if job.status in {JobStatus.FAILED, JobStatus.STOPPED}:
            label = failure_reason_label(job.failure_reason)
            if label:
                text = f"{text} · {label}"
            tooltip = cls._failure_tooltip(job)
        if item.text() != text:
            item.setText(text)
        if item.toolTip() != tooltip:
            item.setToolTip(tooltip)
        item.setForeground(cls._status_brush(job.status.value))

    _WALLTIME_NEXT_STEP = "Resubmit to run it again, with a longer walltime if needed."
    _RELAXATION_WALLTIME_NEXT_STEP = (
        "Resubmit to continue from its checkpoint or start again, with a longer "
        "walltime if needed."
    )

    @classmethod
    def _failure_tooltip(cls, job: Job) -> str:
        summary = failure_reason_summary(job.failure_reason)
        message = (job.error_message or "").strip()
        if job.failure_reason is None and message:
            return f"{message}\n\nUse Inspect to see the job's logs."
        parts = [summary]
        if message and message != summary:
            parts.append(message)
        if job.failure_reason == FailureReason.WALLTIME:
            parts.append(
                cls._RELAXATION_WALLTIME_NEXT_STEP
                if cls._job_resumes_from_checkpoint(job)
                else cls._WALLTIME_NEXT_STEP
            )
        return "\n\n".join(parts)

    def _can_retry_retrieval(self, job: Job) -> bool:
        """Return True when the "Retry retrieval" action is eligible for a job.

        Eligible iff the job is failed/stopped, targets a cluster with both a
        remote job id and remote directory persisted, and the cluster runner is
        currently connected to that job's cluster.
        """

        if job.status not in {JobStatus.FAILED, JobStatus.STOPPED}:
            return False
        if not (job.remote_job_id or "").strip() or not (job.remote_directory or "").strip():
            return False
        return self._is_cluster_connected_for_job(job)

    def _is_cluster_connected_for_job(self, job: Job) -> bool:
        """Return True when the cluster runner is connected to this job's cluster.

        Mirrors the casefold comparison used by cluster resubmission
        (``_submit_jobs_on_cluster``). Local jobs are never "cluster connected".
        """

        target = (job.execution_target or "").strip()
        if not target or target.casefold() == "local":
            return False
        if not self._cluster_runner.is_connected:
            return False
        connected = (self._cluster_runner.connected_cluster_name or "").strip()
        return bool(connected) and connected.casefold() == target.casefold()

    @staticmethod
    def _format_relative_age(then, *, now) -> str:
        """Render a compact relative age such as ``just now`` / ``2 min ago``."""

        if then is None:
            return "—"
        stamp = then if then.tzinfo is not None else then.replace(tzinfo=UTC)
        seconds = max(0, int((now - stamp).total_seconds()))
        if seconds < 45:
            return "just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{max(1, minutes)} min ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours} hr ago"
        days = hours // 24
        return f"{days} day{'s' if days != 1 else ''} ago"

    def _apply_last_checked_cell(self, item, job: Job, *, now) -> None:
        """Fill one Last-checked cell: relative age, stale marker, and tooltip."""

        base_text = self._format_relative_age(job.last_checked_at, now=now)
        stale_reason = job_staleness(
            job,
            now=now,
            cluster_connected=self._is_cluster_connected_for_job(job),
        )
        if stale_reason is not None:
            item.setText(f"{base_text} — stale")
            item.setForeground(QBrush(QColor(current_palette().warning)))
        else:
            item.setText(base_text)

        tooltip_lines: list[str] = []
        if job.last_status_source:
            tooltip_lines.append(f"Source: {job.last_status_source}")
        if job.last_status_message:
            tooltip_lines.append(f"Message: {job.last_status_message}")
        if stale_reason is not None:
            tooltip_lines.append(f"Stale: {stale_reason}")
        if tooltip_lines:
            item.setToolTip("\n".join(tooltip_lines))

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
        with trace.stage("_refresh_quantum_corrections_view"):
            self._refresh_quantum_corrections_view()
        with trace.stage("_refresh_energy_barriers_view"):
            self._refresh_energy_barriers_view()
        with trace.stage("_persist_document_ui_state"):
            self._persist_document_ui_state(failure_message=self._dashboard_autosave_failure_message)
        trace.log_summary()

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
        elif (analysis_workspace_id := self._relaxation_analysis_workspace_id_from_route(route)) is not None:
            self._relaxation_analysis_view.focus_workspace(analysis_workspace_id)
            target = self._relaxation_analysis_view
        elif route == "muon_sites":
            self._refresh_muon_sites_summary()
            target = self._muon_sites_summary_view
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
                "Only failed, stopped, or cancelled jobs can be resubmitted from the job dashboard.",
            )
            return

        refusals = self._fresh_start_refusals(
            [job for job in selected if self._job_resumes_from_checkpoint(job)]
        )
        if refusals:
            # Neither start these again nor continue them: their directory is
            # not theirs alone, so starting again would delete another job's
            # files and continuing would run in (and overwrite) them.
            selected = [job for job in selected if job.id not in refusals]
            QMessageBox.warning(
                self,
                "Some jobs were not resubmitted",
                "These relaxations' folders are not theirs alone, so resubmitting them "
                "would delete or overwrite another job's files. They were not "
                "resubmitted:\n\n"
                + "\n".join(refusals.values())
                + "\n\nDismiss or reset them instead.",
            )
            if not selected:
                return

        fresh_start_job_ids = self._choose_relaxation_resubmit_mode(selected)
        if fresh_start_job_ids is None:
            return

        local_jobs = [job for job in selected if job.execution_target == "local"]
        cluster_jobs = [job for job in selected if job.execution_target != "local"]

        if local_jobs:
            self._submit_jobs_locally(
                local_jobs,
                fresh_start_job_ids=fresh_start_job_ids,
                eligible_statuses=_RESUBMITTABLE_JOB_STATES,
                no_eligible_message=(
                    "Selected jobs must be failed, stopped, or cancelled for local resubmission."
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
                    "Selected jobs must be failed, stopped, or cancelled for cluster resubmission."
                ),
                reuse_existing_remote_files=True,
                fresh_start_job_ids=fresh_start_job_ids,
                success_message=lambda count, cluster: f"Resubmitted {count} cluster job(s) on {cluster}",
            )

    @staticmethod
    def _job_resumes_from_checkpoint(job: Job) -> bool:
        """Return whether resubmitting ``job`` in place resumes from its checkpoint.

        Relaxation inputs always carry ``CONTINUATION : DEFAULT``, so CASTEP
        picks up a ``{seed}.check`` left in the run directory. Imported jobs are
        excluded: their directories hold the only copy of the imported results.
        """

        return job.workflow_stage in {
            WorkflowStage.PRISTINE_RELAXATION,
            WorkflowStage.MUON_RELAXATION,
        } and not job_is_imported(job)

    def _choose_relaxation_resubmit_mode(self, jobs: Sequence[Job]) -> set[UUID] | None:
        """Ask how resubmitted relaxations should treat their previous run.

        Returns the ids of the jobs to start again from a clean directory (empty
        when they continue, or when no relaxation is being resubmitted), or
        ``None`` when the user cancels.
        """

        relaxations = [job for job in jobs if self._job_resumes_from_checkpoint(job)]
        if not relaxations:
            return set()

        probes = self._checkpoint_probes(relaxations)
        dialog = ResubmitChoiceDialog(
            [
                ResubmitCandidate(
                    label=job.label,
                    status=job.status,
                    reason=job.failure_reason,
                    checkpoint=probes[job.id].status,
                )
                for job in relaxations
            ],
            other_job_count=len(jobs) - len(relaxations),
            parent=self,
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
        :meth:`_directory_ownership_refusals`). Reconfiguring a relaxation row
        can leave an old failed job and its replacement sharing one directory,
        which is the case this exists for.
        """

        return self._directory_ownership_refusals(jobs)

    def _directory_ownership_refusals(
        self,
        jobs: Sequence[Job],
        *,
        shared_with: Collection[UUID] = (),
        check_remote: bool = True,
    ) -> dict[UUID, str]:
        """Return, per job id, why deleting that job's directories would be unsafe.

        A job is refused when its local directory is not strictly inside
        ``<project>/files/<stage>/``, is not named after the job's seed, or is
        used by another job (the same directory, or one inside it); and, with
        ``check_remote``, when its remote directory is not named after the seed
        or another job on the same cluster uses it or a directory inside it.
        Jobs in ``shared_with`` may share a directory without causing a
        refusal: they are being cleaned together (a Quantum Corrections
        singlepoint and phonon pair share one seed directory by design).
        """

        if self._document is None or not jobs:
            return {}
        files_root = (self._document.project_root / "files").resolve()

        def local_directory(job: Job) -> Path | None:
            directory = self._job_output_directory(job)
            return directory.resolve() if directory is not None else None

        def remote_directory(job: Job) -> tuple[str, str] | None:
            target = (job.execution_target or "").strip().casefold()
            directory = (job.remote_directory or "").strip()
            if not target or target == "local" or not directory:
                return None
            return target, posixpath.normpath(directory)

        others = [
            (other.id, local_directory(other), remote_directory(other))
            for other in self._document.jobs.values()
        ]
        refusals: dict[UUID, str] = {}
        for job in jobs:
            seed = self._job_seed_name(job)
            local = local_directory(job)
            remote = remote_directory(job)
            reason = ""
            if local is not None:
                if not local.is_relative_to(files_root) or len(local.relative_to(files_root).parts) < 2:
                    reason = f"its output folder is outside the project's job folders ({local})"
                elif local.name != seed:
                    reason = f"its output folder is not named after the job ({local})"
                elif any(
                    other_id != job.id
                    and other_id not in shared_with
                    and other_local is not None
                    and other_local.is_relative_to(local)
                    for other_id, other_local, _other_remote in others
                ):
                    reason = f"another job uses the same output folder ({local})"
            if not reason and check_remote and remote is not None:
                cluster, path = remote
                if posixpath.basename(path) != seed:
                    reason = f"its cluster directory is not named after the job ({path})"
                elif any(
                    other_id != job.id
                    and other_id not in shared_with
                    and other_remote is not None
                    and other_remote[0] == cluster
                    and (other_remote[1] == path or other_remote[1].startswith(path.rstrip("/") + "/"))
                    for other_id, _other_local, other_remote in others
                ):
                    reason = f"another job uses the same cluster directory ({path})"
            if reason:
                refusals[job.id] = f"- {job.label}: {reason}"
        return refusals

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
            output_dir = self._job_output_directory(job)
            if output_dir is not None and output_dir.is_dir():
                try:
                    self._clear_directory_contents(output_dir)
                except OSError as exc:
                    errors.append(f"- {job.label}: could not clear its previous output ({exc})")
                    continue
            cleaned.append(job)
        return cleaned, errors

    def _on_retry_retrieval_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return

        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        eligible = [job for job in jobs if job is not None and self._can_retry_retrieval(job)]
        if not eligible:
            QMessageBox.information(
                self,
                "No jobs to retry",
                "Retry retrieval is available for failed or stopped cluster jobs that still "
                "have remote metadata while their cluster is connected.",
            )
            return

        retried = 0
        for job in self._dedupe_jobs(eligible):
            output_dir = self._job_output_directory(job)
            if output_dir is None:
                continue
            prepared = self._job_execution.prepare_retry_retrieval(job.id)
            if prepared is None:
                continue
            self._cluster_runner.retrieve_outputs_for(
                job.id,
                remote_job_id=(job.remote_job_id or "").strip(),
                remote_directory=(job.remote_directory or "").strip(),
                seed_name=self._job_seed_name(job),
                local_output_dir=output_dir,
                cluster_name=job.execution_target,
                **self._retrieval_flags_for_job(job).as_kwargs(),
            )
            retried += 1

        if retried:
            self._request_dashboard_flush(
                failure_message="Retry retrieval was requested, but autosave failed.",
            )
            self.statusBar().showMessage(f"Retrying retrieval for {retried} cluster job(s)")

    def _on_dismiss_from_dashboard(self, job_ids: list[UUID]) -> None:
        if self._document is None:
            return

        jobs = [self._document.jobs.get(job_id) for job_id in job_ids]
        dismissable = [
            job
            for job in jobs
            if job is not None
            and job.status in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
            and job.dismissed_at is None
        ]
        if not dismissable:
            QMessageBox.information(
                self,
                "No jobs to dismiss",
                "Only failed, stopped, or cancelled jobs can be dismissed from the dashboard.",
            )
            return

        dismissed = 0
        for job in self._dedupe_jobs(dismissable):
            try:
                self._project_service.dismiss_job(job)
            except ValueError:
                continue
            self._project_service.record_job_dismissed(self._document, job=job)
            dismissed += 1

        if dismissed:
            self._refresh_journal_view()
            self._request_dashboard_flush(
                failure_message="Jobs were dismissed, but autosave failed.",
            )
            self.statusBar().showMessage(f"Dismissed {dismissed} job(s) from the dashboard")

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

        def eligibility(uid: UUID) -> bool:
            job = self._document.jobs.get(uid)  # type: ignore[union-attr]
            return job is not None and self._dashboard_action_applies(kind, job)

        actions = {
            "resubmit": "Resubmit",
            "retry_retrieval": "Retry retrieval for",
            "cancel": "Cancel",
            "inspect": "Inspect",
            "reset": "Reset",
            "dismiss": "Dismiss",
        }

        # Cancel and Dismiss act immediately, so a single-row click is confirmed
        # here; on a multi-row selection the three-way prompt is the
        # confirmation. Reset always asks its own clean-up question, which has
        # a Cancel, so it needs nothing extra.
        clicked_job = self._document.jobs.get(job_id)
        clicked_label = clicked_job.label if clicked_job is not None else str(job_id)
        single_confirmations = {
            "cancel": lambda: workflow_table_utils.confirm_destructive(
                self,
                "Cancel job",
                f"Cancel job '{clicked_label}'?",
                "",
                "Cancel job",
                "Keep running",
            ),
            "dismiss": lambda: workflow_table_utils.confirm_destructive(
                self,
                "Dismiss job",
                f"Dismiss job '{clicked_label}' from the dashboard?",
                "The job stays in the project; its workflow row can still resubmit it.",
                "Dismiss job",
            ),
        }

        target_ids = resolve_target_keys(
            button,
            clicked_key=job_id,
            snapshot_keys=snapshot_keys,
            eligibility=eligibility,
            parent=self,
            action=actions[kind],
            noun=("job", "jobs"),
            confirm_unprompted=single_confirmations.get(kind),
        )
        if not target_ids:
            return

        self._dashboard_action_handler(kind)(list(target_ids))

    def _dashboard_action_applies(self, kind: str, job: Job) -> bool:
        """Return whether dashboard action ``kind`` can act on ``job``.

        Single source of truth for the row buttons' multi-row prompt and the
        right-click menu.
        """

        if kind == "retry_retrieval":
            # Retry needs remote metadata + a live connection, not just a status.
            return self._can_retry_retrieval(job)
        terminal = {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
        eligible_statuses = {
            "resubmit": _RESUBMITTABLE_JOB_STATES,
            "cancel": ACTIVE_JOB_STATUSES,
            "inspect": {JobStatus.FAILED, JobStatus.STOPPED},
            "reset": terminal,
            "dismiss": terminal,
        }
        return job.status in eligible_statuses[kind]

    def _dashboard_action_handler(self, kind: str) -> Callable[[list[UUID]], None]:
        # Looked up by name at call time so a replaced handler is honoured.
        names = {
            "resubmit": "_on_resubmit_from_dashboard",
            "retry_retrieval": "_on_retry_retrieval_from_dashboard",
            "cancel": "_on_cancel_from_dashboard",
            "inspect": "_on_inspect_failed_from_dashboard",
            "reset": "_on_reset_failed_from_dashboard",
            "dismiss": "_on_dismiss_from_dashboard",
        }
        return getattr(self, names[kind])

    #: Right-click menu entries for the Job Dashboard, in the row buttons' order.
    _DASHBOARD_MENU_ACTIONS = (
        ("Resubmit", "resubmit"),
        ("Retry Retrieval", "retry_retrieval"),
        ("Cancel", "cancel"),
        ("Inspect", "inspect"),
        ("Reset", "reset"),
        ("Dismiss", "dismiss"),
    )

    def _open_jobs_table_context_menu(self, position) -> QMenu | None:
        """Offer the dashboard actions for the selected jobs.

        A right-click on a row outside the selection selects it first, so the
        menu always describes what is highlighted. Each action is enabled when
        it applies to at least one selected job and acts on those jobs only,
        which is how the row buttons already treat a mixed selection; for a
        mixed selection the entry says how many ("Reset (2 of 3)").
        """

        if self._document is None:
            return None
        row = self._jobs_table.rowAt(position.y())
        if row < 0:
            return None
        selection_model = self._jobs_table.selectionModel()
        selected_rows = {index.row() for index in selection_model.selectedRows()}
        if row not in selected_rows:
            self._jobs_table.selectRow(row)
        jobs = self._selected_dashboard_jobs()
        if not jobs:
            return None

        menu = QMenu(self._jobs_table)
        for text, kind in self._DASHBOARD_MENU_ACTIONS:
            eligible_ids = [job.id for job in jobs if self._dashboard_action_applies(kind, job)]
            action = menu.addAction(text + partial_count_suffix(len(eligible_ids), len(jobs)))
            action.setEnabled(bool(eligible_ids))
            action.triggered.connect(
                lambda _checked=False, k=kind, ids=eligible_ids: self._run_dashboard_menu_action(k, ids)
            )
        menu.addSeparator()
        delete_action = menu.addAction("Delete...")
        delete_action.triggered.connect(
            lambda _checked=False: self._on_dashboard_delete_key_requested()
        )
        menu.exec(self._jobs_table.viewport().mapToGlobal(position))
        return menu

    def _run_dashboard_menu_action(self, kind: str, job_ids: list[UUID]) -> None:
        if not job_ids:
            return
        # Cancel and Dismiss act immediately, so they are confirmed here; Reset
        # asks its own clean-up question, which has a Cancel.
        phrase = count_phrase(len(job_ids), ("job", "jobs"))
        if kind == "cancel" and not workflow_table_utils.confirm_destructive(
            self, "Cancel jobs", f"Cancel {phrase}?", "", f"Cancel {phrase}", "Keep running"
        ):
            return
        if kind == "dismiss" and not workflow_table_utils.confirm_destructive(
            self,
            "Dismiss jobs",
            f"Dismiss {phrase} from the dashboard?",
            "Dismissed jobs stay in the project; their workflow rows can still resubmit them.",
            f"Dismiss {phrase}",
        ):
            return
        self._dashboard_action_handler(kind)(list(job_ids))

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
        """Apply Delete/Backspace to the selected dashboard jobs.

        What the key does depends on each job's state: running jobs are
        cancelled, failed/stopped/cancelled ones are reset, and the rest have
        their records deleted. One summary dialog states all of that before
        anything happens, so nothing is cancelled or deleted unannounced and a
        single Cancel leaves every selected job untouched.
        """

        jobs = self._selected_dashboard_jobs()
        if not jobs:
            return

        terminal_statuses = {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
        active_jobs = [job for job in jobs if job.status in ACTIVE_JOB_STATUSES]
        terminal_jobs = [job for job in jobs if job.status in terminal_statuses]
        resettable_jobs = [job for job in terminal_jobs if not job_is_imported(job)]
        deletable_jobs = [
            job
            for job in jobs
            if job.status not in ACTIVE_JOB_STATUSES and job.status not in terminal_statuses
        ]

        if not (active_jobs or resettable_jobs or deletable_jobs):
            # Only imported terminal jobs are selected: let the reset path
            # explain why they cannot be reset.
            self._reset_failed_jobs(terminal_jobs)
            return

        cleanup_choice = self._prompt_dashboard_bulk_action(
            cancel_count=len(active_jobs),
            reset_count=len(resettable_jobs),
            delete_count=len(deletable_jobs),
            skipped_imported_count=len(terminal_jobs) - len(resettable_jobs),
        )
        if cleanup_choice is None:
            return

        if active_jobs:
            self._cancel_jobs(active_jobs)
        if resettable_jobs:
            self._reset_failed_jobs(resettable_jobs, cleanup_choice=cleanup_choice)
        if deletable_jobs:
            self._delete_dashboard_jobs(deletable_jobs, confirmed=True)

    def _prompt_dashboard_bulk_action(
        self,
        *,
        cancel_count: int,
        reset_count: int,
        delete_count: int,
        skipped_imported_count: int = 0,
    ) -> bool | None:
        """Show the dashboard Delete summary; return the clean-up choice or ``None``.

        ``None`` means the user cancelled. Otherwise the bool says whether the
        output directories of the jobs being reset should be cleaned (always
        ``False`` when nothing is being reset, since the option is not shown).
        """

        lines: list[str] = []
        if cancel_count:
            lines.append(f"Cancel {count_phrase(cancel_count, ('running job', 'running jobs'))}")
        if reset_count:
            lines.append(f"Reset {count_phrase(reset_count, ('failed job', 'failed jobs'))} to ready")
        if delete_count:
            lines.append(
                f"Delete {count_phrase(delete_count, ('finished job record', 'finished job records'))} "
                "and linked results"
            )
        total = cancel_count + reset_count + delete_count

        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("Apply to selected jobs")
        message.setText(f"Apply to {count_phrase(total, ('selected job', 'selected jobs'))}?")
        informative = "\n".join(f"\u2022 {line}" for line in lines)
        if skipped_imported_count:
            informative += (
                f"\n\n{count_phrase(skipped_imported_count, ('imported job', 'imported jobs'))} "
                "will be left as is: imported jobs cannot be reset."
            )
        message.setInformativeText(informative)
        cleanup_box: QCheckBox | None = None
        if reset_count:
            cleanup_box = QCheckBox(
                "Also clean the output "
                + ("directory of the job being reset" if reset_count == 1 else "directories of the jobs being reset"),
                message,
            )
            message.setCheckBox(cleanup_box)
        apply_button = message.addButton("Apply", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = message.addButton(QMessageBox.StandardButton.Cancel)
        message.setDefaultButton(cancel_button)
        message.setEscapeButton(cancel_button)
        message.exec()
        if message.clickedButton() is not apply_button:
            return None
        return bool(cleanup_box is not None and cleanup_box.isChecked())

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

    def _delete_dashboard_jobs(self, jobs: list[Job], *, confirmed: bool = False) -> None:
        if self._document is None:
            return

        deletable_jobs = [
            job
            for job in self._dedupe_jobs(jobs)
            if job.status not in ACTIVE_JOB_STATUSES
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
        if not confirmed and message.exec() != QMessageBox.StandardButton.Yes:
            return

        for job in deletable_jobs:
            self._delete_job_records(job.id)

        self._refresh_jobs_table()
        self._convergence_controller.refresh_view()
        self._kpoint_convergence_controller.refresh_view()
        self.statusBar().showMessage(f"Deleted {len(deletable_jobs)} job(s)")

    @staticmethod
    def _cutoff_key(value: float) -> str:
        return f"{float(value):.12g}"

    def _deserialize_convergence_config(self, payload: object) -> ConvergenceConfig | None:
        return deserialize_convergence_config(payload)

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
            str(name): restore_structured_value(None, str(name), raw_value)
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
        xc_functional = self._document.project.recommended_xc_functional
        if xc_functional is not None:
            values["xc_functional"] = xc_functional
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

    def _energy_barrier_neb_row_configuration_with_source_defaults(
        self,
        definition,
        key: str,
        *,
        target_lattice: list[list[float]] | None = None,
        source_candidate: _EnergyBarrierSourceCandidate | None = None,
        source_defaults_cache: dict[UUID, dict[str, object]] | None = None,
    ) -> tuple[dict[str, object], tuple[str, ...], dict[str, object]]:
        # Precedence: project convergence defaults < reactant relaxation settings < user config.
        # Inheriting the reactant endpoint's kpoint grid and cut-off keeps the NEB band on the
        # same numerical footing as the relaxed structures it connects. (When the relaxation's
        # output .cell/.param are unavailable, the reactant defaults are simply empty and the
        # project convergence defaults stand in — graceful by design.)
        # Callers that loop over many keys may pass a pre-resolved source_candidate and a shared
        # source_defaults_cache to avoid rebuilding the candidate list and re-reading the
        # reactant's files per key.
        params, advanced, extras = self._workflow_row_configuration(definition, key)
        params = _strip_legacy_energy_barrier_params(params)
        values = self._project_convergence_parameter_values(target_lattice=target_lattice)
        candidate = source_candidate if source_candidate is not None else self._energy_barrier_source_candidate_by_key(key)
        if candidate is not None:
            reactant_job_id = candidate.reactant.job.id
            if source_defaults_cache is not None and reactant_job_id in source_defaults_cache:
                reactant_defaults = source_defaults_cache[reactant_job_id]
            else:
                reactant_defaults = self._quantum_source_default_parameter_values(candidate.reactant)
                if source_defaults_cache is not None:
                    source_defaults_cache[reactant_job_id] = reactant_defaults
            values.update(reactant_defaults)
        # Default to an odd image count, and round any user-supplied even count up: climbing-image
        # NEB requires an odd TSSEARCH_MAX_PATH_POINTS (CASTEP otherwise silently adds one).
        values.setdefault("tssearch_max_path_points", _ENERGY_BARRIER_NEB_DEFAULT_IMAGES)
        values.update(params)
        try:
            image_count = int(values.get("tssearch_max_path_points"))
        except (TypeError, ValueError):
            image_count = None
        if image_count is not None and image_count % 2 == 0:
            values["tssearch_max_path_points"] = image_count + 1
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
        """Delegate to :func:`mufinder.castep_io.param_parser.parse_param_value`.

        Kept as a thin wrapper because existing callers reach it here; the
        coercion itself belongs in ``castep_io`` beside the ``.param`` reader
        that shares it.
        """

        return parse_param_value(raw_value, spec)

    @staticmethod
    def _parse_param_scalar_assignments(text: str) -> dict[str, object]:
        """Delegate to :func:`mufinder.castep_io.param_parser.parse_param_assignments`."""

        return parse_param_assignments(text)

    def _quantum_source_default_parameter_values(self, candidate: _QuantumSourceCandidate) -> dict[str, object]:
        values: dict[str, object] = {}
        output_dir = self._job_output_directory(candidate.job)
        if output_dir is None:
            return values

        seed_name = self._job_seed_name(candidate.job)
        cell_path = output_dir / f"{seed_name}.cell"
        parsed_cell = None
        if cell_path.is_file():
            try:
                parsed_cell = parse_cell_file(cell_path)
            except (CellParseError, OSError, ValueError):
                parsed_cell = None
            if parsed_cell is not None and parsed_cell.cell_file.kpoint_mp_grid is not None:
                values["kpoint_mp_grid"] = tuple(int(component) for component in parsed_cell.cell_file.kpoint_mp_grid)

        param_path = output_dir / f"{seed_name}.param"
        assignments: dict[str, object] | None = None
        if param_path.is_file():
            try:
                assignments = self._parse_param_scalar_assignments(param_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                assignments = None
        if assignments is not None:
            cutoff = assignments.get("cut_off_energy")
            if cutoff is not None:
                values["cut_off_energy"] = cutoff
            # A follow-on calculation is only comparable with its source
            # relaxation when both use the same functional.
            xc_functional = assignments.get("xc_functional")
            if xc_functional is not None:
                values["xc_functional"] = xc_functional
            # ...and the same spin treatment.
            values["spin_polarized"] = run_is_spin_polarized(
                assignments,
                parsed_cell.atomic_spins if parsed_cell is not None else None,
            )
        elif self._charge_state_for_job(candidate.job) == MUONIUM_CHARGE_STATE:
            # Without the source's .param, fall back to the default a muonium
            # relaxation is written with (see relaxation_adapter). The value
            # also seeds the always-shown Spin Polarized checkbox, which would
            # otherwise save an unticked box over the workflow's own default.
            values["spin_polarized"] = True
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
    def _num_farms_from_extras(extras: Mapping[str, object]) -> int:
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

    def _energy_barrier_neb_parameter_specs(self) -> list[KeywordSpec]:
        specs = build_workflow_parameter_surface(
            workflow_stage="energy_barriers",
            basic_names=_ENERGY_BARRIER_NEB_KEYWORD_NAMES,
        )
        # Surface TSSEARCH_FORCE_TOL in eV/Ang (its catalog unit is H/BOHR) with CASTEP's own
        # default pre-filled, so the editor labels/edits it legibly and an untouched row reproduces
        # CASTEP's behaviour. The physical editor forces the value's unit to spec.unit, so both the
        # unit and the default_value must be overridden together.
        return [
            replace(
                spec,
                unit=_ENERGY_BARRIER_NEB_FORCE_TOL_UNIT,
                default_value=_ENERGY_BARRIER_NEB_DEFAULT_FORCE_TOL,
            )
            if spec.name == "tssearch_force_tol"
            else spec
            for spec in specs
        ]

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

    def _batch_row_payload(
        self,
        definition,
        key: str,
        payload: dict[str, object] | None,
        untouched_mixed: set[str],
    ) -> dict[str, object] | None:
        """Return the configuration to store for one row of a batch edit.

        ``payload`` is what the dialog produced for every row. Parameters that
        differed between the rows and were left alone are taken from the row's
        own stored configuration instead: a row that set one explicitly keeps
        it, and a row that did not keeps deriving it from its source defaults.
        """

        if not untouched_mixed:
            return payload
        row_params, row_advanced, _extras = self._workflow_row_configuration(definition, key)
        merged = dict(payload or {})
        collected = merged.get("parameter_values")
        merged["parameter_values"] = merge_batch_values(
            collected if isinstance(collected, dict) else {},
            self._normalized_parameter_values(dict(row_params)) or {},
            untouched_mixed,
        )
        merged["added_advanced"] = sorted(
            merge_batch_advanced(merged.get("added_advanced", ()), row_advanced, untouched_mixed)
        )
        has_extras = any(name not in {"parameter_values", "added_advanced"} for name in merged)
        if not (merged["parameter_values"] or merged["added_advanced"] or has_extras):
            return None
        return merged

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
        self._connect_workflow_row_actions(energy_barriers_view, self._energy_barriers_controller)
        energy_barriers_view.view_source_requested.connect(self._on_view_energy_barrier_source)
        energy_barriers_view.configure_result_requested.connect(self._on_configure_energy_barrier_result)
        # The controller already exists (built + registered in ``__init__`` even
        # though this view is created lazily); wire its submit here at view
        # creation, mirroring the six eager controllers' constructor-time wiring.
        energy_barriers_view.submit_neb_local_requested.connect(
            lambda values: self._energy_barriers_controller.submit(
                values, target="local"
            )
        )
        energy_barriers_view.submit_neb_cluster_requested.connect(
            lambda values: self._energy_barriers_controller.submit(
                values, target="cluster"
            )
        )
        energy_barriers_view.neb_result_selected.connect(
            self._on_energy_barrier_neb_result_selected
        )
        self._energy_barriers_view = energy_barriers_view
        self._main_stack.insertWidget(self._main_stack.indexOf(self._journal_view), energy_barriers_view)
        return energy_barriers_view

    def _definition_for_convergence_cutoff(self, cutoff: float):
        cutoff_value = float(cutoff)
        active_definition = self._convergence_controller.active_definition()
        if active_definition is None:
            return None, None

        self._sync_equivalent_convergence_cutoff_mappings(active_definition)
        config = self._deserialize_convergence_config(active_definition.parameters.get("configuration", {}))
        if config is not None and cutoff_value in config.cutoff_energies:
            return active_definition, config
        return None, None

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
            and definition_output_directory(self._document, definition) == output_directory
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
            and definition_output_directory(self._document, definition) == output_directory
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

        active = self._convergence_controller.active_definition()
        definitions = self._convergence_controller.definitions()
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
        output_directory = definition_output_directory(self._document, target_definition)
        if output_directory is None:
            return

        simulation_cell_mode = str(target_definition.parameters.get("simulation_cell_mode", "unit_cell"))
        supercell_source_workflow = target_definition.parameters.get("supercell_source_workflow")

        candidates: dict[str, str] = {}
        for definition in self._kpoint_convergence_controller.definitions():
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
        output_directory = definition_output_directory(self._document, target_definition)
        if output_directory is None:
            return

        simulation_cell_mode = str(target_definition.parameters.get("simulation_cell_mode", "unit_cell"))
        supercell_source_workflow = target_definition.parameters.get("supercell_source_workflow")
        parameters = target_definition.parameters.get("configuration")
        if not isinstance(parameters, dict):
            return

        candidates: dict[str, str] = {}
        for definition in self._convergence_controller.definitions():
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

        cell = self._convergence_sweep_cell(self._convergence_controller.active_definition()) or host
        dialog = ConvergenceExaminerDialog(
            self._convergence_controller.build_states(),
            num_atoms=len(cell.species),
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
        definition = self._convergence_controller.active_definition()
        definition_cell_id = None
        if definition is not None and definition.input_structure_ids:
            definition_cell_id = str(definition.input_structure_ids[0])
        xc_functional = self._convergence_definition_xc_functional(definition)
        if not self._confirm_adopted_xc_functional(xc_functional):
            return
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
            xc_functional=xc_functional,
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
        self._refresh_project_defaults_views()
        self._refresh_convergence_overview()
        self._refresh_journal_view()
        self._persist_document_ui_state(
            failure_message="Could not persist convergence overview snapshot to disk"
        )
        self._populate_project_tree()
        self.statusBar().showMessage(f"Project default cutoff set to {float(cutoff_energy):.8g} eV")

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
            self._convergence_controller.refresh_view()
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
        job = self._convergence_controller.job_for_cutoff(definition, cutoff) if definition is not None else None

        job_label = None
        if job is not None:
            job_label = job.label
        elif config is not None:
            job_label = convergence_job_name(
                config.name_prefix,
                cutoff,
                sweep_parameter="cut_off_energy",
            )

        output_root = definition_output_directory(self._document, definition) if definition is not None else None
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
        output_directory = definition_output_directory(self._document, source_definition)
        if output_directory is None:
            self._remove_cutoff_from_definition(source_definition, cutoff)
            return

        simulation_cell_mode = str(source_definition.parameters.get("simulation_cell_mode", "unit_cell"))
        supercell_source_workflow = source_definition.parameters.get("supercell_source_workflow")
        for definition in self._convergence_controller.definitions():
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

    def _run_blocking_file_op(self, op: Callable[[], object], *, description: str) -> object:
        """Run a blocking filesystem operation off the GUI thread.

        Preserves the caller's synchronous semantics (returns the result,
        re-raises the exception) while pumping the event loop, so the retry
        backoff against OneDrive/antivirus file locks and large recursive
        deletes cannot freeze the window. The window is disabled while the
        operation runs to prevent re-entrant user actions, and a busy dialog
        appears only when the operation takes noticeably long.
        """

        outcome: dict[str, object] = {}
        done = threading.Event()

        def _worker() -> None:
            try:
                outcome["result"] = op()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the GUI thread
                outcome["error"] = exc
            finally:
                done.set()

        worker = threading.Thread(target=_worker, name="mufinder-file-op", daemon=True)
        loop = QEventLoop()
        progress: QProgressDialog | None = None
        started = time.monotonic()
        poll = QTimer(self)
        poll.setInterval(25)

        def _tick() -> None:
            nonlocal progress
            if done.is_set():
                loop.quit()
                return
            if progress is None and time.monotonic() - started > 0.4:
                progress = QProgressDialog(description, None, 0, 0, self)
                progress.setWindowTitle("Working on files")
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.show()

        poll.timeout.connect(_tick)
        was_enabled = self.isEnabled()
        self.setEnabled(False)
        worker.start()
        poll.start()
        try:
            if not done.is_set():
                loop.exec()
        finally:
            poll.stop()
            poll.deleteLater()
            if progress is not None:
                progress.close()
            self.setEnabled(was_enabled)
            worker.join()
        error = outcome.get("error")
        if isinstance(error, BaseException):
            raise error
        return outcome.get("result")

    def _delete_path(self, path: Path) -> None:
        if path.is_dir():
            self._run_blocking_file_op(
                lambda: self._retry_filesystem_op(
                    lambda: shutil.rmtree(path, onexc=self._force_writable_and_retry)
                ),
                description=f"Deleting {path.name}…",
            )
            return
        self._run_blocking_file_op(
            lambda: self._retry_filesystem_op(path.unlink),
            description=f"Deleting {path.name}…",
        )

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
        self._run_blocking_file_op(
            lambda: self._retry_filesystem_op(lambda: source.rename(destination)),
            description=f"Archiving {source.name}…",
        )
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
            site.total_energy_eV = None
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
        for definition in self._quantum_corrections_controller.definitions():
            for mapping_name in ("singlepoint_job_ids_by_label", "phonon_job_ids_by_label", "job_ids_by_label"):
                mapping = self._quantum_job_mapping(definition, mapping_name)
                for key, value in mapping.items():
                    if str(value) == str(job.id):
                        return definition, str(key)
        return None, None

    def _task_farming_for_job(self, job: Job) -> tuple[bool, int | None]:
        """Return ``(farming_enabled, num_farms)`` for any farm-aware calc type.

        Both runners use this to decide which output files a finished job wrote:
        task farming renames every output to ``<seed>_farmNNN.<ext>``, so a job
        whose farm count is misreported here is read from the wrong files. The
        quantum-phonon and NEB workflows each store their farm count in the
        owning definition's per-row configuration extras; every other calc type
        is unfarmed.
        """

        if job.calc_type == QUANTUM_PHONON_CALC_TYPE:
            definition, key = self._quantum_definition_and_key_for_job(job)
        elif job.calc_type == NUDGED_ELASTIC_BAND_CALC_TYPE:
            definition, key = self._energy_barrier_neb_definition_and_key_for_job(job)
        else:
            return False, None
        if definition is None or key is None:
            return False, None
        _params, _advanced, extras = self._workflow_row_configuration(definition, key)
        num_farms = self._num_farms_from_extras(extras)
        return num_farms > 1, (num_farms if num_farms > 1 else None)

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

    def _energy_barrier_neb_definition_and_key_for_job(self, job: Job) -> tuple[object | None, str | None]:
        """Locate the NEB definition and source key that materialized ``job``."""

        for definition in self._energy_barriers_controller.definitions():
            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict):
                continue
            for key, value in mapping.items():
                if str(value) == str(job.id):
                    return definition, str(key)
        return None, None

    def _quantum_stage_one_target(self, job: Job) -> str | None:
        """Return where a quantum phonon job's stage-1 singlepoint ran.

        ``None`` when ``job`` is not a quantum phonon job or its completed
        singlepoint cannot be found; otherwise the singlepoint's
        ``execution_target`` (``"local"`` or a cluster name).
        """

        if job.calc_type != QUANTUM_PHONON_CALC_TYPE:
            return None
        definition, key = self._quantum_definition_and_key_for_job(job)
        if definition is None or key is None:
            return None
        source_job = self._singlepoint_job_for_quantum_source_key(definition, key)
        if source_job is None or source_job.status != JobStatus.COMPLETED:
            return None
        return (source_job.execution_target or "").strip() or "local"

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
        eligible_jobs = [job for job in self._dedupe_jobs(jobs) if job.status in allowed_statuses]
        if not eligible_jobs:
            QMessageBox.information(
                self,
                "No eligible jobs",
                no_eligible_message,
            )
            return

        # A quantum phonon job continues from its stage-1 checkpoint, which never
        # leaves the cluster that ran the singlepoint. The Quantum Corrections page
        # refuses these up front; this catches every other route (the dashboard).
        stranded = [
            (job, stage_one)
            for job in eligible_jobs
            if (stage_one := self._quantum_stage_one_target(job)) is not None
            and stage_one.casefold() != "local"
        ]
        if stranded:
            stranded_ids = {job.id for job, _stage_one in stranded}
            eligible_jobs = [job for job in eligible_jobs if job.id not in stranded_ids]
            QMessageBox.information(
                self,
                "Phonon stage must run where stage 1 ran",
                "These phonon jobs were not run on this computer because their stage-1 "
                "checkpoint (.check) is only on the cluster that ran the singlepoint:\n\n"
                + "\n".join(f"  - {job.label} (stage 1 ran on {stage_one})" for job, stage_one in stranded),
            )
            if not eligible_jobs:
                return

        setup = load_castep_setup(self._settings)
        if not setup.executable_path:
            QMessageBox.warning(
                self,
                "CASTEP executable missing",
                "Configure a CASTEP executable in Setup -> CASTEP before submitting jobs.",
            )
            return
        if not self._keyword_catalog_guard.confirm_submission():
            return

        dialog = LocalSubmissionDialog(
            job_labels=[job.label for job in eligible_jobs],
            mpirun_configured=bool(setup.mpirun_path),
            farm_counts=self._farm_counts_for_jobs(eligible_jobs),
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        config = dialog.config()

        fresh_jobs = [job for job in eligible_jobs if job.id in fresh_start_job_ids]
        if fresh_jobs:
            cleaned, cleanup_errors = self._clear_outputs_for_fresh_start(fresh_jobs)
            if cleanup_errors:
                not_cleaned = {job.id for job in fresh_jobs} - {job.id for job in cleaned}
                eligible_jobs = [job for job in eligible_jobs if job.id not in not_cleaned]
                show_detailed_warning(
                    self,
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
            retrieval_for_job=self._retrieval_flags_for_job,
        )

        self._refresh_jobs_table()
        self._convergence_controller.refresh_view()
        self._kpoint_convergence_controller.refresh_view()
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
        fresh_start_job_ids: Collection[UUID] = (),
    ) -> None:
        """Submit ``jobs`` to a cluster chosen in the cluster submission dialog.

        Jobs in ``fresh_start_job_ids`` start again: on their own cluster, their
        remote directory is deleted and their inputs uploaded afresh, without the
        overwrite prompt (the user already chose to discard the previous run),
        and their local output is cleared as Reset would clear it.
        """

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
        if not self._keyword_catalog_guard.confirm_submission():
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
            farm_counts=self._farm_counts_for_jobs(eligible_jobs),
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

        # A quantum phonon job must continue in the remote directory holding its
        # stage-1 checkpoint. Submitted anywhere else it would start cold in a
        # fresh directory and CASTEP would fail for want of the .check file, so
        # refuse it here, after the dialog, where a different cluster may have
        # been picked than the one the Quantum Corrections page preselected.
        stranded = [
            (job, stage_one)
            for job in eligible_jobs
            if (stage_one := self._quantum_stage_one_target(job)) is not None
            and not self._quantum_continuation_remote_target(job, cluster_name=selected_cluster.name)
        ]
        if stranded:
            stranded_ids = {job.id for job, _stage_one in stranded}
            eligible_jobs = [job for job in eligible_jobs if job.id not in stranded_ids]
            QMessageBox.information(
                self,
                "Phonon stage must run where stage 1 ran",
                f"These phonon jobs were not submitted to {selected_cluster.name} because "
                "their stage-1 checkpoint (.check) is not in a directory there:\n\n"
                + "\n".join(
                    f"  - {job.label} (stage 1 ran {'on this computer' if stage_one.casefold() == 'local' else 'on ' + stage_one})"
                    for job, stage_one in stranded
                ),
            )
            if not eligible_jobs:
                return

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
        fresh_delete_job_ids: set[UUID] = set()
        previous_remote_job_id_by_job: dict[UUID, str] = {}
        for job in eligible_jobs:
            fresh_target = (
                (job.remote_directory or "").strip()
                if job.id in fresh_start_job_ids
                and (job.execution_target or "").strip().casefold()
                == selected_cluster.name.strip().casefold()
                else ""
            )
            if fresh_target:
                # Start again in the job's own directory: delete it and upload the
                # inputs afresh, so no checkpoint or output of the previous run
                # survives for CONTINUATION : DEFAULT to pick up.
                remote_target_by_job[job.id] = fresh_target
                reuse_existing_files_by_job[job.id] = False
                allow_existing_remote_directory_by_job[job.id] = False
                fresh_delete_job_ids.add(job.id)
                # A stopped job may still be running (the watchdog stops jobs
                # this session lost track of without cancelling them), so the
                # worker deletes the directory only once the scheduler confirms
                # the previous run has finished.
                previous_remote_job_id_by_job[job.id] = (job.remote_job_id or "").strip()
                continue
            allow_existing_remote_directory = False
            # A DEFINED job that still carries a remote directory is a continuation.
            # Only ProjectService.prepare_job_continuation produces that combination
            # (see its docstring for the writer audit), and it preserves the directory
            # precisely so CASTEP resumes from the ``.check`` file already there. Reuse
            # the directory verbatim and upload fresh inputs into it.
            #
            # Apart from the start-again branch above, whose jobs are never DEFINED,
            # this branch is evaluated first and cannot steal either path below.
            # ``reuse_existing_remote_files`` is passed only by the dashboard resubmit
            # action, whose eligible statuses are {FAILED, STOPPED, CANCELLED}, so DEFINED jobs
            # never reach the loop on that call. The quantum branch needs a quantum
            # phonon calc type with a COMPLETED singlepoint source; a continued phonon
            # job would satisfy both, but its own preserved remote directory is the
            # source directory that branch would have returned, so the two agree.
            #
            # The cluster in this call is only *preselected* to the job's original
            # ``execution_target`` (see the Continue-relaxation caller): the
            # submission dialog still lets the user pick a different one. A
            # continuation must never carry ``job.remote_directory`` onto a
            # DIFFERENT cluster — that path does not exist there, so
            # ``allow_existing_remote_directory=True`` would submit a cold run into
            # a foreign directory string, silently rewrite ``execution_target``, and
            # orphan the checkpoint the continuation was meant to resume. Guarding on
            # a cluster-name match makes a same-cluster continuation behave exactly
            # as before and makes an off-cluster continuation fall through to the
            # normal path (recomputed target, normal existence check).
            continued_remote_target = (
                (job.remote_directory or "").strip()
                if job.status == JobStatus.DEFINED
                and (job.execution_target or "").strip().casefold()
                == selected_cluster.name.strip().casefold()
                else ""
            )
            if continued_remote_target:
                remote_target = continued_remote_target
                allow_existing_remote_directory = True
            else:
                remote_target = (
                    (job.remote_directory or "").strip()
                    if reuse_existing_remote_files and job.id not in fresh_start_job_ids
                    else ""
                )
                if not remote_target:
                    continuation_target = self._quantum_continuation_remote_target(
                        job, cluster_name=selected_cluster.name
                    )
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
            if job.id in fresh_delete_job_ids:
                decision_by_job[job.id] = "delete"
                continue
            # Directories we deliberately submit into are never candidates for the
            # overwrite prompt, whatever another job in the batch resolved to: this
            # loop matches by target string, so without the guard a second job whose
            # computed target collides with a preserved directory could mark it for
            # deletion.
            if allow_existing_remote_directory_by_job.get(job.id, False):
                continue
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
        # Directories another job in the batch is submitting into as-is are never
        # deleted, however that job's own overwrite prompt was answered: the
        # decision belongs to the colliding job, but the deletion would land on a
        # preserved continuation directory holding the run's ``.check`` file.
        protected_remote_targets = {
            remote_target_by_job[job_id]
            for job_id, allowed in allow_existing_remote_directory_by_job.items()
            if allowed and remote_target_by_job.get(job_id)
        }
        fresh_jobs = [
            job
            for job in eligible_jobs
            if job.id in fresh_start_job_ids and decision_by_job.get(job.id) != "skip"
        ]
        not_cleaned_job_ids: set[UUID] = set()
        if fresh_jobs:
            cleaned, cleanup_errors = self._clear_outputs_for_fresh_start(fresh_jobs)
            failed_submit.extend(cleanup_errors)
            not_cleaned_job_ids = {job.id for job in fresh_jobs} - {job.id for job in cleaned}
        prepared_by_job: dict[UUID, PreparedClusterSubmission] = {}
        submit_items: list[dict[str, object]] = []
        for job in eligible_jobs:
            if decision_by_job.get(job.id) == "skip" or job.id in not_cleaned_job_ids:
                continue
            if job.id in fresh_delete_job_ids and remote_target_by_job[job.id] in protected_remote_targets:
                # Its directory cannot be deleted, so submitting would resume
                # from the checkpoint the user chose to discard.
                failed_submit.append(
                    f"- {job.label}: another job in this submission is continuing in its "
                    f"directory ({remote_target_by_job[job.id]}), so it was not started again"
                )
                continue
            output_dir = self._job_output_directory(job)
            if output_dir is None:
                missing_paths.append(job.label)
                continue
            seed_name = seed_name_by_job.get(job.id, self._job_seed_name(job))
            remote_target = remote_target_by_job.get(job.id)
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
                    mem_per_cpu_gb=getattr(submission, "mem_per_cpu_gb", None),
                    queue_name=getattr(submission, "queue_name", ""),
                    comment=getattr(submission, "comment", ""),
                    castep_executable=selected_cluster.castep_executable,
                    reuse_existing_files=reuse_existing_files_by_job.get(job.id, False),
                    allow_existing_remote_directory=allow_existing_remote_directory_by_job.get(job.id, False),
                    retrieval=self._retrieval_flags_for_job(job),
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
                        remote_target
                        if decision_by_job.get(job.id) == "delete"
                        and remote_target not in protected_remote_targets
                        else None
                    ),
                    "delete_only_if_finished": previous_remote_job_id_by_job.get(job.id) or None,
                }
            )

        # 5. Run the batch on the worker thread with a cancellable progress dialog.
        if submit_items:
            batch_submitted, batch_failed = self._run_cluster_submit_batch(submit_items, prepared_by_job)
            submitted = len(batch_submitted)
            submitted_jobs = batch_submitted
            failed_submit.extend(batch_failed)

        self._refresh_jobs_table()
        self._convergence_controller.refresh_view()
        self._kpoint_convergence_controller.refresh_view()
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

        mem_per_cpu_raw = payload.get("mem_per_cpu_gb")
        mem_per_cpu_gb: float | None
        try:
            mem_per_cpu_gb = None if mem_per_cpu_raw in (None, "") else float(mem_per_cpu_raw)
        except (TypeError, ValueError):
            mem_per_cpu_gb = None
        if mem_per_cpu_gb is not None and mem_per_cpu_gb <= 0:
            mem_per_cpu_gb = None

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
            mem_per_cpu_gb=mem_per_cpu_gb,
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
            "mem_per_cpu_gb": getattr(config, "mem_per_cpu_gb", None),
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

        progress = QProgressDialog(label, None, 0, 0, self)
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
        loop wait ``_submit_jobs_on_cluster`` already uses mid-flow before
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
        timed_out = False

        progress = QProgressDialog("Submitting cluster jobs…", "Cancel", 0, total, self)
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
                f"{pending} pending job(s) were cancelled."
            )

        # Jobs prepared but never submitted (cancelled mid-batch or timed out)
        # are cancelled so they do not linger in the ``submitting`` state.
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
            if job.status not in ACTIVE_JOB_STATUSES:
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

        diagnoses = [self._job_failure_diagnosis(job) for job in failed_jobs]
        sections = [
            self._failure_diagnostics_for_job(job, diagnosis)
            for job, diagnosis in zip(failed_jobs, diagnoses, strict=True)
        ]
        details = "\n\n".join(
            (
                "=" * 80,
                *sections,
            )
        )

        message = make_message_box_resizable(QMessageBox(self))
        message.setIcon(QMessageBox.Icon.Information)
        message.setWindowTitle("Failed job diagnostics")
        message.setText(self._failure_headline(failed_jobs, diagnoses))
        message.setInformativeText("Open details to inspect available error/log file content.")
        message.setDetailedText(details)
        message.exec()

    def _reset_failed_jobs(self, jobs: list[Job], *, cleanup_choice: bool | None = None) -> None:
        """Reset terminal jobs back to ``defined``.

        ``cleanup_choice`` is for a caller that has already confirmed the reset
        and asked whether to clean the job directories (the dashboard Delete
        summary); left as ``None`` the user is asked here.
        """

        if self._document is None:
            return

        terminal_jobs = [
            job for job in jobs if job.status in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
        ]
        resettable_jobs = [job for job in terminal_jobs if not job_is_imported(job)]
        imported_count = len(terminal_jobs) - len(resettable_jobs)
        if imported_count:
            # Resetting an imported job would relaunch a calculation MuFinder
            # never ran, and its cleanup would delete the only copy of the
            # imported results and their archived originals.
            QMessageBox.information(
                self,
                "Imported jobs cannot be reset",
                f"{imported_count} imported job(s) were skipped. They record calculations "
                "run outside MuFinder, and their directories hold the only copy of those "
                "results.",
            )
        if not resettable_jobs:
            QMessageBox.information(
                self,
                "No resettable jobs",
                "Only jobs in the 'failed', 'stopped', or 'cancelled' state can be reset.",
            )
            return

        if cleanup_choice is None:
            cleanup_choice = self._prompt_failed_job_cleanup(len(resettable_jobs))
        if cleanup_choice is None:
            return

        reset_count, cleanup_errors = self._apply_failed_job_reset(
            resettable_jobs,
            cleanup_choice=cleanup_choice,
            detach_convergence=True,
        )

        self._refresh_views_after_job_reset()

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

    def _refresh_views_after_job_reset(self) -> None:
        """Repaint every workflow table that can show a just-reset job.

        Shared by the explicit per-row/dashboard Reset action and by the lazy
        reset a dismissed row performs on resubmit. ``_refresh_jobs_table``
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
        self._refresh_quantum_corrections_view()

    def _apply_failed_job_reset(
        self,
        jobs: Sequence[Job],
        *,
        cleanup_choice: bool,
        detach_convergence: bool,
    ) -> tuple[int, list[str]]:
        """Clean, reset, and record provenance for already-vetted terminal jobs.

        Shared core of the dashboard/table "Reset failed jobs" action and of the
        lazy reset the workflow-row submit cycle performs for dismissed jobs
        (:meth:`_prepare_jobs_for_resubmission`). Callers own the "nothing to
        reset" dialog, the cleanup prompt, the view refreshes, the autosave and
        the status-bar message; this helper only mutates job state and returns
        ``(reset_count, cleanup_errors)``.

        ``detach_convergence`` unbinds a reset cutoff-convergence job from its
        definition's ``job_ids_by_cutoff`` mapping. That is right for an explicit
        reset — the sweep point starts over from an empty row — but wrong for a
        resubmission, which must keep the row bound to the very job it is about
        to submit, or the row would project "ready" with no job and a second
        submit would mint a duplicate.
        """

        cleanup_errors: list[str] = []
        reset_count = 0
        # Never clean a directory another job still depends on: a Quantum
        # Corrections phonon job shares its seed directory, and the stage-1
        # checkpoint in it, with the completed singlepoint. Jobs reset together
        # may share one; anything else keeps its directory and is reported.
        cleanup_refusals = (
            self._directory_ownership_refusals(
                [job for job in jobs if not job_is_imported(job)],
                # Imported jobs are skipped below, never cleaned, so a folder
                # shared with one must not count as cleaned together.
                shared_with={job.id for job in jobs if not job_is_imported(job)},
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
            elif cleanup_choice:
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

            if detach_convergence and job.workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF:
                self._detach_convergence_job(job.id)

            self._active_run_by_job.pop(job.id, None)
            self._record_job_reset(job=job, cleaned_output_directory=cleaned_output_directory)
            reset_count += 1

        return reset_count, cleanup_errors

    def _prepare_jobs_for_resubmission(self, jobs: Sequence[Job]) -> bool:
        """Reset dismissed terminal jobs so a workflow row can resubmit them.

        Bound into every controller's
        :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`
        and called from the shared submit cycle just before dispatch. Dismissing
        a failed job from the jobs dashboard removes its Reset/Resubmit controls
        while its workflow row keeps pointing at it, so the row's own submit is
        the recovery path: it runs the same stale-output cleanup prompt the Reset
        action uses, then resets the jobs back to ``defined`` so they clear the
        ``{DEFINED}`` gate in the submit backends.

        Returns ``False`` only when the user cancels the cleanup prompt, which
        aborts the submission without an error dialog. Jobs that are *not*
        dismissed terminal jobs are never touched here.
        """

        if self._document is None:
            return False

        resettable_jobs = [
            job
            for job in jobs
            if job_is_resubmittable_after_dismissal(job) and not job_is_imported(job)
        ]
        if not resettable_jobs:
            return True

        cleanup_choice = self._prompt_failed_job_cleanup(len(resettable_jobs))
        if cleanup_choice is None:
            return False

        reset_count, cleanup_errors = self._apply_failed_job_reset(
            resettable_jobs,
            cleanup_choice=cleanup_choice,
            detach_convergence=False,
        )

        self._refresh_views_after_job_reset()

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
        return True

    def _continue_relaxation_checkpoint_probe(self, job: Job) -> CheckpointProbeResult:
        """Probe for ``{seed}.check`` in the directory a Continue would reuse.

        Local jobs get a plain :class:`~pathlib.Path` check. Cluster jobs are
        probed only when the cluster runner is currently connected to that
        job's cluster (:meth:`_is_cluster_connected_for_job`, the same check
        cluster resubmission uses) via the WP5 bridge
        (``ClusterController.check_remote_files``/``remote_files_checked``),
        awaited synchronously through :meth:`_await_remote_file_check` — the
        same nested-event-loop pattern ``_submit_jobs_on_cluster`` already uses
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
            seed_name = self._job_seed_name(job)
            if (job.execution_target or "").strip().casefold() == "local":
                output_dir = self._job_output_directory(job)
                if output_dir is None:
                    probes[job.id] = CheckpointProbeResult.unavailable("The job's output directory is unknown")
                elif (output_dir / f"{seed_name}.check").exists():
                    probes[job.id] = CheckpointProbeResult.found()
                else:
                    probes[job.id] = CheckpointProbeResult.absent()
                continue
            if not self._is_cluster_connected_for_job(job):
                probes[job.id] = CheckpointProbeResult.unavailable(f"Not connected to {job.execution_target}")
                continue
            remote_directory = (job.remote_directory or "").strip()
            if not remote_directory:
                probes[job.id] = CheckpointProbeResult.unavailable("No remote directory recorded for this job")
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
        and the identical seed → ``set_values``/``set_advanced_names`` →
        ``collect_values()`` sequence ``ContinueRelaxationDialog.__init__`` runs
        internally, so the two agree keyword-for-keyword on an unedited row.
        """

        specs = reorder_workflow_parameter_specs(list(seed.get("specs", ())))
        parameter_values = dict(seed.get("parameter_values", {}))
        added_advanced = tuple(seed.get("added_advanced", ()))
        reference = ParameterDialog(
            specs=specs,
            workflow_stage=str(seed.get("workflow_stage", "")),
            locked_values=seed.get("locked_values"),
            species_options=seed.get("species_options", ()),
        )
        try:
            reference.set_values(
                dict(parameter_values),
                preferred_exclusive=kpoint_exclusive_hint([parameter_values]),
            )
            reference.set_advanced_names(added_advanced)
            values = dict(reference.collect_values())
            values.pop("task", None)
            return values, reference.advanced_names()
        finally:
            reference.close()

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
        ``_submit_jobs_on_cluster`` with that cluster preselected, mirroring
        how ``_on_resubmit_from_dashboard`` picks ``default_cluster_name``).

        Deliberately bypasses ``controller.submit()``: that template method has
        no hook for preselecting a specific cluster (``submission_groups``
        supplies no per-group ``default_cluster_name``), and — for muon
        relaxation — its ``pre_submit_check`` host-mismatch validation would be
        redundant here: the row's site already matched the active host when
        this job was first submitted, and offers no sensible resolution for
        resubmitting that same site. Because ``continue_job`` already leaves
        the job ``DEFINED`` before either submit path runs,
        ``job_is_resubmittable_after_dismissal`` is false for it, so the
        dismissed-job cleanup prompt (``_prepare_jobs_for_resubmission``) is
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

        seed = view.continue_relaxation_dialog_seed(key)
        if seed is None:
            return

        checkpoint_probe = self._continue_relaxation_checkpoint_probe(job)

        def render_param_text(
            values: Mapping[str, object], advanced: tuple[str, ...]
        ) -> str:
            return controller.render_row_param_text(key, values, advanced)

        resolved_values, resolved_advanced = self._resolve_continue_relaxation_seed_values(seed)
        old_param_text = render_param_text(resolved_values, resolved_advanced)

        if (job.execution_target or "").strip().casefold() == "local":
            output_dir = self._job_output_directory(job)
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
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        parameter_values, added_advanced = dialog.edited_parameters()
        continued_job = controller.continue_job(key, parameter_values, added_advanced)
        if continued_job is None:
            # The row's parameters may already be mutated (update_row_parameters
            # ran before materialize failed), but nothing was submitted: warn
            # rather than fail silently, since the user just accepted a diff
            # that promised the new parameters would run.
            QMessageBox.warning(
                self,
                "Continue relaxation failed",
                "Could not regenerate this job's input files, so it was not "
                "resubmitted. The row's job was left in a recoverable state; "
                "try Continue relaxation again.",
            )
            return

        if (continued_job.execution_target or "").strip().casefold() == "local":
            self._submit_jobs_locally([continued_job])
        else:
            self._submit_jobs_on_cluster(
                [continued_job], default_cluster_name=continued_job.execution_target
            )

        self._refresh_views_after_job_reset()
        self._persist_document_ui_state(failure_message=autosave_failure_message)

    def _on_continue_relaxation_requested_from_pristine_relaxation(self, key: object) -> None:
        self._continue_relaxation_row(
            controller=self._pristine_relaxation_controller,
            view=self._pristine_relaxation_view,
            key=key,
            autosave_failure_message="Relaxation was continued, but autosave failed.",
        )

    def _on_continue_relaxation_requested_from_muon_relaxation(self, key: object) -> None:
        self._continue_relaxation_row(
            controller=self._muon_relaxation_controller,
            view=self._muon_relaxation_view,
            key=key,
            autosave_failure_message="Relaxation was continued, but autosave failed.",
        )

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
            submit_jobs_locally=lambda *args, **kwargs: self._submit_jobs_locally(*args, **kwargs),
            submit_jobs_on_cluster=lambda *args, **kwargs: self._submit_jobs_on_cluster(*args, **kwargs),
            prepare_jobs_for_resubmission=lambda jobs: self._prepare_jobs_for_resubmission(jobs),
            show_info=lambda title, text: QMessageBox.information(self, title, text),
            request_dashboard_flush=lambda **kwargs: self._request_dashboard_flush(**kwargs),
            persist_ui_state=lambda **kwargs: self._persist_document_ui_state(**kwargs),
            status_message=lambda message: self.statusBar().showMessage(message),
            definition_output_directory=lambda definition: definition_output_directory(self._document, definition),
            job_output_directory=lambda job: self._job_output_directory(job),
            job_seed_name=lambda job: self._job_seed_name(job),
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
            sync_equivalent_convergence_cutoff_mappings=(
                lambda definition: self._sync_equivalent_convergence_cutoff_mappings(definition)
            ),
            cutoff_key=lambda value: self._cutoff_key(value),
            max_stress_for_job=lambda job_id: max_stress_for_job(self._document, job_id),
        )

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
            sync_equivalent_convergence_kpoint_mappings=(
                lambda definition: self._sync_equivalent_convergence_kpoint_mappings(definition)
            ),
            grid_key=lambda grid: self._grid_key(grid),
            max_stress_for_job=lambda job_id: max_stress_for_job(self._document, job_id),
        )

    def _build_pristine_relaxation_deps(self) -> PristineRelaxationDeps:
        """Build the pristine-relaxation controller's deps.

        These helpers stay on ``MainWindow`` because non-migrating code still
        calls them. The relaxed ``-out.cell`` is parsed once on the runner worker
        thread and delivered via ``ParsedJobArtifacts``, so no parse dep is
        threaded here. Cross-workflow job/definition readers
        (``job_output_directory``, ``job_seed_name``, ``run_for_job``,
        ``definition_output_directory``, the energy/force readers) now live on
        the shared context. The controller reaches them only through late-bound
        wrappers.
        """

        return PristineRelaxationDeps(
            deserialize_pristine_relaxation_config=deserialize_pristine_relaxation_config,
            serialize_pristine_relaxation_config=serialize_pristine_relaxation_config,
            sync_equivalent_pristine_relaxation_mappings=(
                lambda definition: self._sync_equivalent_pristine_relaxation_mappings(definition)
            ),
            pristine_relaxation_form_config=lambda: self._pristine_relaxation_form_config(),
            pristine_relaxation_row_signature=(
                lambda row: self._pristine_relaxation_row_signature(row)
            ),
            expand_pristine_relaxation_jobs=(
                lambda definition, config: self._expand_pristine_relaxation_jobs_for_controller(
                    definition, config
                )
            ),
        )

    def _expand_pristine_relaxation_jobs_for_controller(
        self, definition: CalculationDefinition, config: PristineRelaxationConfig
    ) -> AdapterExpansion | None:
        """Look up the definition's host structure and expand its relaxation jobs.

        Returns ``None`` when the host structure is missing, mirroring the guard
        in the former ``_materialize_pristine_relaxation_jobs``. The
        ``ValueError`` / ``StructureBridgeError`` raised by
        :func:`expand_pristine_relaxation_jobs` propagate to the controller, which
        catches them (as the former handler did).
        """

        if self._document is None or not definition.input_structure_ids:
            return None
        host = self._document.structures.get(definition.input_structure_ids[0])
        if host is None:
            return None
        return expand_pristine_relaxation_jobs(
            host_structure=host,
            config=config,
            project_cutoff_energy=self._document.project.recommended_cutoff_energy,
            project_kpoint_grid=self._document.project.recommended_kpoint_grid,
            project_kpoint_reference_lattice=self._document.project.recommended_kpoint_lattice,
            project_xc_functional=self._document.project.recommended_xc_functional,
        )

    def _build_muon_relaxation_deps(self) -> MuonRelaxationDeps:
        """Build the muon-relaxation controller's deps.

        These helpers stay on ``MainWindow`` because non-migrating code still
        calls them: the definition/config/mapping helpers are shared with row
        removal, staging, and dashboard sync;
        ``_muon_fractional_position_from_cell`` locates the ``H:mu`` position
        for the controller's ``validate_artifacts`` and ``build_result``;
        ``_coerce_fractional_triplet`` and ``_muon_relaxation_row_key`` have
        other view-refresh callers. The controller reaches them only through
        these late-bound wrappers.
        """

        return MuonRelaxationDeps(
            deserialize_muon_relaxation_config=deserialize_muon_relaxation_config,
            serialize_muon_relaxation_config=serialize_muon_relaxation_config,
            sync_equivalent_muon_relaxation_mappings=(
                lambda definition: self._sync_equivalent_muon_relaxation_mappings(definition)
            ),
            sync_active_muon_relaxation_dashboard_jobs=(
                lambda definition: self._sync_active_muon_relaxation_dashboard_jobs(definition)
            ),
            muon_relaxation_form_config=lambda: self._muon_relaxation_form_config(),
            muon_relaxation_row_signature=(
                lambda row: self._muon_relaxation_row_signature(row)
            ),
            muon_relaxation_row_key=lambda row: self._muon_relaxation_row_key(row),
            handle_site_host_mismatch=(
                lambda mismatch: self._handle_site_host_mismatch(mismatch)
            ),
            muon_relaxation_definition_and_row_for_job=(
                lambda job: self._muon_relaxation_definition_and_row_for_job(job)
            ),
            muon_fractional_position_from_cell=(
                lambda cell: self._muon_fractional_position_from_cell(cell)
            ),
            coerce_fractional_triplet=(
                lambda value: self._coerce_fractional_triplet(value)
            ),
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
            sync_equivalent_uep_mappings=(
                lambda definition: self._sync_equivalent_uep_mappings(definition)
            ),
            uep_potential_grid_result_for_job=(
                lambda job_id: self._uep_potential_grid_result_for_job(job_id)
            ),
        )

    def _build_quantum_corrections_deps(self) -> QuantumCorrectionsDeps:
        """Build the quantum-corrections controller's deps.

        These helpers stay on ``MainWindow`` because non-migrating code still
        calls them: the definition/stage/config helpers are shared with the
        quantum configure path, source-label rename, and stage-one reset;
        ``_check_definition_host`` is shared with the NEB / energy-barriers
        workflow; the source-candidate and row-configuration helpers are shared
        across the quantum surface and the energy-barriers path; and
        ``_dedupe_jobs`` is shared across workflows (the cross-workflow job/
        definition readers now live on the shared context). The module-level
        ``_input_template_from_keyword_values`` / ``_format_energy``
        helpers and the ``_quantum_method_from_gui_value`` lookup are also reached
        through these late-bound wrappers (the controller must not import
        ``main_window``). The controller reaches them only through these callables.
        """

        return QuantumCorrectionsDeps(
            ensure_quantum_corrections_definition=(
                lambda keys: self._ensure_quantum_corrections_definition(keys)
            ),
            eligible_quantum_source_candidates=(
                lambda: self._eligible_quantum_source_candidates()
            ),
            quantum_source_relative_energies=(
                lambda: self._quantum_source_relative_energies()
            ),
            quantum_source_candidate_by_key=(
                lambda key: self._quantum_source_candidate_by_key(key)
            ),
            quantum_stage_state=(
                lambda definition, candidate: self._quantum_stage_state(
                    definition, candidate
                )
            ),
            build_quantum_corrections_states=(
                lambda: self._build_quantum_corrections_states()
            ),
            jobs_for_quantum_source_keys=(
                lambda keys: self._jobs_for_quantum_source_keys(keys)
            ),
            singlepoint_job_for_quantum_source_key=(
                lambda definition, key: self._singlepoint_job_for_quantum_source_key(
                    definition, key
                )
            ),
            phonon_job_for_quantum_source_key=(
                lambda definition, key: self._phonon_job_for_quantum_source_key(
                    definition, key
                )
            ),
            quantum_seed_name_for_key=(
                lambda definition, key: self._quantum_seed_name_for_key(definition, key)
            ),
            quantum_runtime_configuration=(
                lambda definition, key: self._quantum_runtime_configuration(
                    definition, key
                )
            ),
            workflow_row_configuration=(
                lambda definition, key: self._workflow_row_configuration(
                    definition, key
                )
            ),
            workflow_row_configuration_exists=(
                lambda definition, key: self._workflow_row_configuration_exists(
                    definition, key
                )
            ),
            quantum_num_farms_from_extras=(
                lambda extras: self._num_farms_from_extras(extras)
            ),
            quantum_source_label=(
                lambda candidate: self._quantum_source_label(candidate)
            ),
            quantum_method_from_gui_value=(
                lambda value: _quantum_method_from_gui_value(value)
            ),
            charge_state_for_job=lambda job: self._charge_state_for_job(job),
            charge_state_label_for_job=(
                lambda job: self._charge_state_label_for_job(job)
            ),
            quantum_can_reset_to_stage_one=(
                lambda definition, candidate: self._quantum_can_reset_to_stage_one(
                    definition, candidate
                )
            ),
            input_template_from_keyword_values=(
                lambda values: _input_template_from_keyword_values(values)
            ),
            write_quantum_job_directory=(
                lambda spec, output_root, *, seed_name: self._write_quantum_job_directory(
                    spec, output_root, seed_name=seed_name
                )
            ),
            dedupe_jobs=lambda jobs: self._dedupe_jobs(jobs),
            check_definition_host=(
                lambda definition: self._check_definition_host(definition)
            ),
            format_energy=lambda value: _format_energy(value),
            confirm=lambda title, text: (
                QMessageBox.question(
                    self,
                    title,
                    text,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                == QMessageBox.StandardButton.Yes
            ),
        )

    def _build_energy_barriers_deps(self) -> EnergyBarriersDeps:
        """Build the energy-barriers (NEB) controller's deps.

        These helpers stay on ``MainWindow`` because non-migrating code still
        calls them: the definition/source/candidate/config helpers are shared
        with the path-generation, source-removal, and result-configuration
        surfaces (which do NOT migrate); the host-mismatch handlers are shared
        across every host-guarded workflow; the endpoint-row / result-row / summary
        helpers back the view the controller renders; and the source-decoding and
        shared-workflow helpers are used across the app. The module-level
        ``_input_template_from_keyword_values`` / ``_neb_barrier_summaries`` and the
        ``muon_charge_state_label`` lookup are reached through these late-bound
        wrappers (the controller must not import ``main_window``). The controller
        reaches them only through these callables.
        """

        return EnergyBarriersDeps(
            build_energy_barrier_neb_states=(
                lambda: self._build_energy_barrier_neb_states()
            ),
            ensure_energy_barrier_neb_definition=(
                lambda keys: self._ensure_energy_barrier_neb_definition(keys)
            ),
            energy_barrier_neb_source_candidates=(
                lambda: self._energy_barrier_neb_source_candidates()
            ),
            eligible_energy_barrier_source_candidates=(
                lambda: self._eligible_energy_barrier_source_candidates()
            ),
            energy_barrier_neb_row_configuration_with_source_defaults=(
                lambda definition, key, **kwargs: (
                    self._energy_barrier_neb_row_configuration_with_source_defaults(
                        definition, key, **kwargs
                    )
                )
            ),
            num_farms_from_extras=(
                lambda extras: self._num_farms_from_extras(extras)
            ),
            energy_barrier_neb_row_extras=(
                lambda definition, key: self._workflow_row_configuration(definition, key)[2]
            ),
            handle_structure_host_mismatch=(
                lambda mismatch: self._handle_structure_host_mismatch(mismatch)
            ),
            handle_site_host_mismatch=(
                lambda mismatch: self._handle_site_host_mismatch(mismatch)
            ),
            uuid_from_source=(
                lambda source, key: self._uuid_from_source(source, key)
            ),
            float_from_source=(
                lambda source, key: self._float_from_source(source, key)
            ),
            fractional_triplet_from_source=(
                lambda source, key: self._fractional_triplet_from_source(source, key)
            ),
            fractional_position_block_from_structure=(
                lambda structure: self._fractional_position_block_from_structure(
                    structure
                )
            ),
            dedupe_jobs=lambda jobs: self._dedupe_jobs(jobs),
            project_convergence_input_defaults=(
                lambda: self._project_convergence_input_defaults()
            ),
            input_template_from_keyword_values=(
                lambda values: _input_template_from_keyword_values(values)
            ),
            workflow_row_configuration_exists=(
                lambda definition, key: self._workflow_row_configuration_exists(
                    definition, key
                )
            ),
            charge_state_label_for_job=(
                lambda job: self._charge_state_label_for_job(job)
            ),
            energy_barrier_endpoint_rows=(
                lambda: self._energy_barrier_endpoint_rows()
            ),
            selected_energy_barrier_endpoint_keys=(
                lambda: self._selected_energy_barrier_endpoint_keys()
            ),
            energy_barrier_max_path_length_text=(
                lambda: self._energy_barrier_max_path_length_text()
            ),
            completed_energy_barrier_neb_candidates=(
                lambda: self._completed_energy_barrier_neb_candidates()
            ),
            energy_barrier_neb_sources_by_job_id=(
                lambda: self._energy_barrier_neb_sources_by_job_id()
            ),
            energy_barrier_neb_result_path_label=(
                lambda candidate, source: self._energy_barrier_neb_result_path_label(
                    candidate, source
                )
            ),
            neb_barrier_summaries=lambda data: _neb_barrier_summaries(data),
            muon_charge_state_label=lambda value: muon_charge_state_label(value),
        )

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

    _FAILURE_HEADLINE_LIMIT = 8

    def _job_failure_diagnosis(self, job: Job) -> FailureDiagnosis:
        """Diagnose ``job`` from its logs, keeping the cause recorded at failure time.

        The fresh diagnosis supplies the evidence line, and covers jobs that
        failed before causes were recorded. A recorded cause the logs cannot
        show (an interrupted run, a rejected submission) is kept as is.
        """

        fresh = self._job_execution.diagnose_failure(job, job.error_message)
        if job.failure_reason is None or fresh.reason == job.failure_reason:
            return fresh
        return FailureDiagnosis(reason=job.failure_reason)

    @classmethod
    def _failure_headline(cls, jobs: list[Job], diagnoses: list[FailureDiagnosis]) -> str:
        lines = [
            f"{job.label}: {diagnosis.summary}"
            for job, diagnosis in zip(jobs[: cls._FAILURE_HEADLINE_LIMIT], diagnoses, strict=False)
        ]
        hidden = len(jobs) - len(lines)
        if hidden > 0:
            lines.append(f"…and {hidden} more job(s).")
        return "\n".join(lines)

    def _failure_diagnostics_for_job(self, job: Job, diagnosis: FailureDiagnosis | None = None) -> str:
        if diagnosis is None:
            diagnosis = self._job_failure_diagnosis(job)
        lines = [
            f"Job: {job.label}",
            f"Workflow: {job.workflow_stage.value}",
            f"Status: {job.status.value}",
            f"Cause: {diagnosis.summary}",
        ]
        if diagnosis.evidence:
            lines.append(f"Evidence ({diagnosis.evidence_source}): {diagnosis.evidence}")

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
        return find_failure_log_files(
            output_dir, [job.label, self._job_seed_name(job)], job.remote_job_id
        )

    _TAIL_READ_LIMIT_BYTES = 256 * 1024

    @classmethod
    def _read_bounded_tail(cls, path: Path, *, max_bytes: int | None = None) -> str:
        """Read at most the trailing ``max_bytes`` of ``path`` as text.

        Files at or under the limit are read in full, byte-for-byte identical
        to a plain ``read_text()`` call. Larger files are read from a tail
        offset only, so an inspect on a runaway multi-hundred-MB job output
        never loads the whole file into memory; a partial leading line (cut
        mid-line by the seek) is dropped and a truncation marker is prefixed,
        matching the "... [truncated]" convention used elsewhere in this file.
        """

        limit = cls._TAIL_READ_LIMIT_BYTES if max_bytes is None else max_bytes
        return read_log_tail(path, limit)

    @classmethod
    def _read_error_contexts(cls, path: Path, *, context_lines: int = 2, max_matches: int = 8) -> str:
        try:
            lines = cls._read_bounded_tail(path).splitlines()
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

    @classmethod
    def _read_text_excerpt(cls, path: Path, *, max_chars: int = 12000) -> str:
        try:
            text = cls._read_bounded_tail(path)
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
        """Delete a job directory's contents except its ``.cell``/``.param`` inputs.

        A job directory is named after the job's seed, so CASTEP's relaxed
        ``{seed}-out.cell`` is recognised by exact name and deleted with the
        rest of the run's output; every other ``.cell``/``.param`` is kept, so a
        seed that itself ends in ``-out`` keeps its input. A symlink is removed
        as a link, never followed into what it points at.
        """

        output_cell = f"{directory.name}-out.cell".casefold()
        for child in directory.iterdir():
            if (
                child.is_file()
                and child.suffix.lower() in {".cell", ".param"}
                and child.name.casefold() != output_cell
            ):
                continue
            if child.is_symlink():
                child.unlink()
            elif child.is_dir():
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
        """Re-render the quantum-corrections view via its controller.

        Thin wrapper (delegates to
        :meth:`QuantumCorrectionsController.refresh_view`) kept on ``MainWindow``
        because it has many callers across document/route changes, submission,
        configuration, rename, and stage reset.
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
        relative = self._quantum_source_relative_energies(candidates)

        def order(item: _QuantumSourceCandidate) -> tuple[object, ...]:
            energy = relative.get(item.key)
            charge_state = self._charge_state_for_job(item.job) or ""
            return (charge_state, energy is None, energy or 0.0, item.site.label)

        return sorted(candidates, key=order)

    def _quantum_source_relative_energies(
        self,
        candidates: list[_QuantumSourceCandidate] | None = None,
    ) -> dict[str, float | None]:
        """Return each relaxed source's energy relative to its comparable minimum.

        ``MuonSite.total_energy_eV`` holds the relaxation's absolute final
        total energy, which is only comparable between relaxations of the same
        cell in the same charge state. Sources are therefore grouped by charge
        state and cell composition (which separates different supercells), and
        each is reported relative to the lowest-energy source in its group.
        """

        if candidates is None:
            candidates = self._eligible_quantum_source_candidates()
        entries: dict[str, tuple[object, float | None]] = {}
        for candidate in candidates:
            composition = tuple(sorted(Counter(candidate.structure.species).items()))
            group = (self._charge_state_for_job(candidate.job), composition)
            entries[candidate.key] = (group, candidate.site.total_energy_eV)
        return grouped_relative_energies(entries)

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
        candidates = self._eligible_quantum_source_candidates()
        relative = self._quantum_source_relative_energies(candidates)
        for candidate in candidates:
            rows.append(
                EnergyBarrierEndpointRow(
                    key=candidate.key,
                    label=candidate.site.label or candidate.job.label or candidate.structure.label,
                    charge_state=self._charge_state_label_for_job(candidate.job),
                    source_job=candidate.job.label,
                    relative_energy=_format_relative_energy(relative.get(candidate.key)),
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
        return job_output_directory(self._document, job, quantum_seed_name=self._quantum_seed_name_for_key)

    def _job_seed_name(self, job: Job) -> str:
        return job_seed_name(job, self._job_output_directory)

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
            self._convergence_controller.refresh_view()
            self._kpoint_convergence_controller.refresh_view()
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
                self._job_output_directory(job),
                self._job_seed_name(job),
                want_out_cell=bool(flags.get("retrieve_out_cell", False)),
                want_phonon=bool(flags.get("retrieve_phonon", False)),
                want_ts=bool(flags.get("retrieve_ts", False)),
                num_farms=farm_count or 1,
            )

        run = self._job_execution.ensure_run_for_job(job, status=JobStatus.RUNNING, started=True)
        if run is None:
            return

        # The owning controller rejects a completion whose outputs cannot carry
        # a result (unconverged geometry, a missing, unparseable or stale
        # secondary output). It reads the carrier, never disk, and runs before
        # any result is built.
        failure = validate_controller_artifacts(
            self._workflow_controllers, job, parsed, run, artifacts
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
                self._workflow_controllers,
                job,
                parsed,
                run,
                artifacts,
                payload=result_payload,
                output_files=job_output_files(
                    self._document.project_root,
                    self._job_output_directory(job),
                    self._job_seed_name(job),
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

    def _on_cluster_job_finished(
        self,
        job_id: UUID,
        parsed: CastepResult,
        artifacts: ParsedJobArtifacts | None = None,
    ) -> None:
        self._on_local_job_finished(job_id, parsed, artifacts)
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
        # Stamp per-job monitor metadata (last_checked_at/source/message) from
        # the snapshot's observations; these land in the in-memory document
        # either way, so a deferred flush (below) is fine — nothing is lost by
        # not persisting them immediately.
        stamped = self._job_execution.mark_queue_checked(checked_at)
        # A queue poll fires this once after any jobs it retrieved have been
        # applied, so this also drains whatever those completions' own handlers
        # already marked pending. Routed through the debounced/rate-limited
        # scheduler rather than flushing synchronously: on a serial cluster
        # retrieval, queue polls land every few seconds, and flushing every
        # poll defeats the burst coalescing (this was the bypass that caused
        # 15 flushes in 80s in the field trace).
        if stamped or self._dashboard_flush_pending:
            self._request_dashboard_flush(
                failure_message="Queue check was recorded, but autosave failed.",
            )
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
            if job.status not in ACTIVE_JOB_STATUSES:
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

    def _on_job_watchdog_tick(self) -> None:
        """Stop active jobs that no runner is tracking (two-tick confirmed).

        Runs off a 60 s timer. Collects live introspection snapshots from the
        local runner and the cluster controller's tracked-id mirror, asks the
        watchdog for confirmed stuck jobs, stops each via
        ``JobExecutionController.stop_stuck_job``, and coalesces the dashboard
        refresh/autosave. Cheap when there are no active jobs.
        """

        if self._document is None:
            return

        active_jobs = [
            job for job in self._document.jobs.values() if job.status in ACTIVE_JOB_STATUSES
        ]
        if not active_jobs:
            # Nothing active: keep the two-tick state from accumulating stale
            # strikes and return early.
            self._stuck_job_watchdog.reset()
            return

        confirmed = self._stuck_job_watchdog.evaluate(
            active_jobs,
            local_active_ids=self._local_runner.active_job_ids(),
            local_pending_ids=self._local_runner.pending_job_ids(),
            cluster_tracked_ids=self._cluster_runner.tracked_active_job_ids(),
            cluster_connected=self._cluster_runner.is_connected,
            now=utc_now(),
            connected_cluster_name=self._cluster_runner.connected_cluster_name,
        )
        if not confirmed:
            return

        stopped = 0
        for job, message in confirmed:
            update = self._job_execution.stop_stuck_job(job.id, message)
            if update.changed:
                stopped += 1
                self._stuck_job_watchdog.forget(job.id)

        if stopped == 0:
            return

        self._request_dashboard_flush(
            failure_message="Jobs were stopped by the watchdog, but autosave failed."
        )
        self.statusBar().showMessage(
            f"{stopped} job(s) were stopped by the job watchdog because no runner "
            "was tracking them."
        )

    def _sync_cluster_tracking_from_document(self) -> None:
        if self._document is None:
            return

        for job in self._document.jobs.values():
            if job.status not in ACTIVE_JOB_STATUSES:
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
                **self._retrieval_flags_for_job(job).as_kwargs(),
            )

    def _retrieval_flags_for_job(self, job: Job) -> RetrievalFlags:
        """Return the retrieval flags and farm count both runners need for ``job``.

        Shared by local and cluster submission, cluster re-attachment and the
        dashboard "Retry retrieval" action, so every path collects the same
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
        }

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
            self._convergence_controller.refresh_view()
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
        self._convergence_controller.refresh_view()
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
        self._refresh_workflow_defaults(self._muon_relaxation_view)
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
        self._set_active_route(self._relaxation_analysis_workspace_route(str(workspace["id"])))
        self.statusBar().showMessage(f"Created analysis workspace '{workspace['name']}'")

    def _on_relaxation_analysis_state_changed(self, workspaces: list) -> None:
        if self._document is None or self._shutting_down:
            return
        cleaned = [dict(workspace) for workspace in workspaces if isinstance(workspace, dict)]
        previous_nodes = [(w.get("id"), w.get("name")) for w in self._relaxation_analysis_workspaces()]
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
            action = menu.addAction("Delete Analysis...")
            action.triggered.connect(
                lambda _checked=False, wid=analysis_id: self._delete_relaxation_analysis_workspace(wid)
            )
        elif (magnetic_id := self._magnetic_field_workspace_id_from_route(route)) is not None:
            action = menu.addAction("Delete Workspace...")
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
            (workspace for workspace in workspaces if str(workspace.get("id", "")) == str(workspace_id)),
            None,
        )
        if target is None:
            return
        name = str(target.get("name", "")).strip() or "Relaxation analysis"
        pinned_charge_states = [
            str(charge_state)
            for charge_state, snapshot in self._muon_sites_summary_snapshots_state().items()
            if isinstance(snapshot, dict) and str(snapshot.get("workspace_id", "")) == str(workspace_id)
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
            self._store_muon_sites_summary_snapshot(charge_state, None)
        if self._active_route() == self._relaxation_analysis_workspace_route(str(workspace_id)):
            # Step up to the Analysis node first: rebuilding the tree without
            # the selected node would otherwise fall back to the home page.
            self._set_active_route("muon_analysis")
        self._refresh_relaxation_analysis_context()
        self._populate_project_tree()
        if pinned_charge_states:
            self._refresh_muon_sites_summary(only_if_visible=True)
        self._persist_document_ui_state(
            failure_message="The analysis was deleted, but autosave failed."
        )
        self.statusBar().showMessage(f"Deleted analysis '{name}'")

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

    # --- Muon Sites summary (pinned analyses) -------------------------------

    def _muon_sites_summary_snapshots_state(self) -> dict[str, object]:
        if self._document is None:
            return {}
        raw = self._document.ui_state.get(self._MUON_SITES_SUMMARY_SNAPSHOTS_KEY, {})
        return raw if isinstance(raw, dict) else {}

    def _store_muon_sites_summary_snapshot(
        self, charge_state: str, snapshot: Mapping[str, object] | None
    ) -> None:
        if self._document is None:
            return
        snapshots = dict(self._muon_sites_summary_snapshots_state())
        if snapshot is None:
            snapshots.pop(charge_state, None)
        else:
            serialized = to_jsonable(dict(snapshot))
            if isinstance(serialized, dict):
                snapshots[charge_state] = serialized
        self._document.ui_state[self._MUON_SITES_SUMMARY_SNAPSHOTS_KEY] = snapshots

    def _validated_muon_sites_summary_snapshot(self, charge_state: str) -> dict[str, object] | None:
        if self._document is None:
            return None
        snapshot = self._muon_sites_summary_snapshots_state().get(charge_state)
        if not isinstance(snapshot, dict):
            return None
        # The snapshot is a decoupled copy; only show it while it still describes
        # the current host structure (mirrors the convergence overview).
        if str(snapshot.get("host_structure_id", "")) != (self._host_structure_id_text() or ""):
            return None
        sites = snapshot.get("sites")
        if not isinstance(sites, list) or not sites:
            return None
        return snapshot

    @staticmethod
    def _muon_site_wyckoff_labels(result: Mapping[str, object]) -> dict[int, str]:
        labels: dict[int, str] = {}
        raw = result.get("wyckoff_assignments")
        if not isinstance(raw, list):
            return labels
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            try:
                cluster_index = int(entry.get("cluster_index", 0))
            except (TypeError, ValueError):
                continue
            match = entry.get("match")
            if cluster_index > 0 and isinstance(match, Mapping):
                label = str(match.get("label", "")).strip()
                if label:
                    labels[cluster_index] = label
        return labels

    def _build_muon_sites_summary_snapshot(
        self, workspace: Mapping[str, object]
    ) -> dict[str, object] | None:
        # Key the snapshot to the workspace's own host, not the active project
        # host: the cluster positions are fractional coordinates in that host's
        # cell, so validation must hide the pin (rather than overlay stale sites
        # on a different host) whenever the active host is not the one the
        # analysis was run against.
        host_structure_id = str(workspace.get("host_structure_id", "")).strip()
        if not host_structure_id:
            return None
        clustering = workspace.get("clustering")
        result = clustering.get("result") if isinstance(clustering, Mapping) else None
        clusters = result.get("clusters") if isinstance(result, Mapping) else None
        if not isinstance(clusters, list) or not clusters:
            return None
        wyckoff_labels = self._muon_site_wyckoff_labels(result)
        sites: list[dict[str, object]] = []
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                continue
            try:
                cluster_index = int(cluster.get("cluster_index", 0))
            except (TypeError, ValueError):
                continue
            position = cluster.get("representative_fractional_position")
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                continue
            try:
                fractional = [float(position[0]), float(position[1]), float(position[2])]
            except (TypeError, ValueError):
                continue
            raw_energy = cluster.get("minimum_relative_energy_ev")
            try:
                energy: float | None = None if raw_energy is None else float(raw_energy)
            except (TypeError, ValueError):
                energy = None
            sites.append(
                {
                    "cluster_index": cluster_index,
                    "fractional_position": fractional,
                    "relative_energy_ev": energy,
                    "wyckoff_label": wyckoff_labels.get(cluster_index, ""),
                }
            )
        if not sites:
            return None
        # Rank distinct sites by relative energy (lowest first); sites without an
        # energy sort last so an unranked cluster never masquerades as the minimum.
        # cluster_index breaks ties so ordering stays deterministic even when
        # every site lacks an energy (otherwise the "rank 1" site would depend on
        # the persisted clusters-list order and shuffle across re-clustering).
        sites.sort(
            key=lambda site: (
                site["relative_energy_ev"] is None,
                site["relative_energy_ev"] if site["relative_energy_ev"] is not None else 0.0,
                site["cluster_index"],
            )
        )
        charge_state = canonical_charge_state(workspace.get("charge_state")) or POSITIVE_MUON_CHARGE_STATE
        return {
            "charge_state": charge_state,
            "workspace_id": str(workspace.get("id", "")),
            "workspace_name": str(workspace.get("name", "")).strip() or "Analysis workspace",
            "host_structure_id": host_structure_id,
            "sites": sites,
        }

    @staticmethod
    def _muon_sites_summary_placeholder(charge_state: str) -> str:
        label = muon_charge_state_label(charge_state)
        return (
            f"No {label} analysis is pinned yet. Use “Go to Analysis” to cluster the "
            f"relaxed {label} sites, then choose “Pin to Summary” to bring the distinct "
            "sites here."
        )

    def _refresh_muon_sites_summary(self, *, only_if_visible: bool = False) -> None:
        summary_view = getattr(self, "_muon_sites_summary_view", None)
        if summary_view is None:
            return
        # The summary drives two PyVista viewers, so host/project-change callers
        # pass only_if_visible=True: they keep a *visible* summary current (e.g.
        # dropping a pin whose host was switched away) without paying for an
        # off-screen rebuild when the page is not showing. Navigation and
        # pin/unpin call it unconditionally because they are about to show it.
        if only_if_visible and self._main_stack.currentWidget() is not summary_view:
            return
        host = (
            self._project_service.active_host_structure(self._document)
            if self._document is not None
            else None
        )
        for charge_state in (POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE):
            summary_view.set_panel(
                charge_state,
                snapshot=self._validated_muon_sites_summary_snapshot(charge_state),
                host_structure=host,
                placeholder_message=self._muon_sites_summary_placeholder(charge_state),
            )

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
        snapshot = self._build_muon_sites_summary_snapshot(workspace)
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
        self._store_muon_sites_summary_snapshot(charge_state, snapshot)
        self._refresh_muon_sites_summary()
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
        if key not in self._muon_sites_summary_snapshots_state():
            return
        self._store_muon_sites_summary_snapshot(key, None)
        self._refresh_muon_sites_summary()
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
                "The pinned analysis has changed — opened the Analysis tab instead."
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
        self._refresh_workflow_supercell_options(self._pristine_relaxation_view)
        self._refresh_workflow_supercell_options(self._muon_relaxation_view)
        self._refresh_workflow_supercell_options(self._uep_view)

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
        self._kpoint_convergence_controller.refresh_view()
        self._refresh_journal_view()
        self.statusBar().showMessage(
            f"Configured {len(config.sweep_values)} k-point jobs; input files will be generated on submission"
        )

    def _on_kpoint_form_state_changed(self, state: dict) -> None:
        if self._document is None or self._shutting_down:
            return
        self._document.ui_state["kpoint_convergence_form"] = state
        self._kpoint_convergence_controller.refresh_view()

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

    def _on_remove_grids_from_kpoint_convergence(self, grids: list[tuple[int, int, int]]) -> None:
        if self._document is None:
            self._kpoint_convergence_view.remove_grids(grids)
            return

        target_keys = {self._grid_key(grid) for grid in grids}
        for definition in self._kpoint_convergence_controller.definitions():
            self._remove_kpoint_grids_from_definition(definition, target_keys)

        self._kpoint_convergence_view.remove_grids(grids)
        self._kpoint_convergence_controller.refresh_view()

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

        cell = self._convergence_sweep_cell(self._kpoint_convergence_controller.active_definition()) or host
        states = self._kpoint_convergence_controller.build_states()
        counts_map, labels, notes = self._kpoint_examiner_points(states, cell.lattice_vectors)

        persisted_state = self._document.ui_state.get(self._KPOINT_EXAMINER_STATE_KEY, {})
        if not isinstance(persisted_state, dict):
            persisted_state = {}

        dialog = ConvergenceExaminerDialog(
            counts_map,
            num_atoms=len(cell.species),
            sweep_kind="kpoint",
            point_labels=labels,
            lattice_vectors=cell.lattice_vectors,
            initial_state=persisted_state,
            notes=notes,
            parent=self,
        )
        dialog.default_adopted.connect(
            lambda value, method, et, ft, st: self._on_kpoint_default_adopted(value, method, et, ft, labels, st)
        )
        dialog.state_changed.connect(self._on_kpoint_examiner_state_changed)
        dialog.exec()
        self._on_kpoint_examiner_state_changed(dialog.state_snapshot())

    def _convergence_sweep_cell(self, definition) -> Structure | None:
        """Return the cell a convergence sweep ran on: its definition's input structure.

        That is the unit cell or the saved supercell chosen under ``Simulation
        Cell``. Per-atom energies and k-point spacings must use this cell, not the
        host unit cell, or a supercell sweep is normalized by the wrong atom count
        and its grids are related to the wrong lattice.
        """

        if self._document is None or definition is None or not definition.input_structure_ids:
            return None
        return self._document.structures.get(definition.input_structure_ids[0])

    @staticmethod
    def _kpoint_examiner_points(
        states: Mapping[tuple[int, int, int], SweepPointRunState],
        lattice_vectors: Sequence[Sequence[float]],
    ) -> tuple[dict[float, SweepPointRunState], dict[float, str], list[str]]:
        """Key k-point sweep states by k-point count for the examiner's axis.

        Two different grids with the same count (only possible with hand-entered
        grids) cannot share a point on that axis. One is kept per count: a grid
        with a completed energy is preferred, then the most uniformly sampled one
        for the cell (see :func:`~mufinder.science.kpoint_grids.most_uniform_grid`).
        Each grid left out is described in the returned notes.
        """

        by_count: dict[int, list[tuple[int, int, int]]] = {}
        for grid in states:
            by_count.setdefault(kpoint_count(grid), []).append(tuple(int(n) for n in grid))  # type: ignore[arg-type]
        counts_map: dict[float, SweepPointRunState] = {}
        labels: dict[float, str] = {}
        notes: list[str] = []
        for count, grids in by_count.items():
            kept = grids[0]
            if len(grids) > 1:
                completed = [grid for grid in grids if states[grid].total_energy is not None]
                if len(completed) == 1:
                    kept = completed[0]
                    reason = "it is the only one with a completed result"
                else:
                    kept = most_uniform_grid(lattice_vectors, completed or grids)
                    reason = "it samples the cell most uniformly" + (
                        " of those with a completed result" if completed and len(completed) < len(grids) else ""
                    )
                excluded = ", ".join(format_kpoint_grid(grid) for grid in grids if grid != kept)
                notes.append(
                    f"{excluded} left out: same k-point count ({count}) as {format_kpoint_grid(kept)}, "
                    f"which is examined instead because {reason}. "
                    "Remove the extra grid from the sweep to examine it."
                )
            counts_map[float(count)] = states[kept]
            labels[float(count)] = f"Grid: {format_kpoint_grid(kept)}"
        return counts_map, labels, notes

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

        active_definition = self._kpoint_convergence_controller.active_definition()
        definition_cell_id = None
        if active_definition is not None and active_definition.input_structure_ids:
            definition_cell_id = str(active_definition.input_structure_ids[0])
        xc_functional = self._convergence_definition_xc_functional(active_definition)
        if not self._confirm_adopted_xc_functional(xc_functional):
            return
        # The grid is only meaningful together with the lattice it was converged
        # on: later workflows rescale it to their own cell with
        # equivalent_grid_for_supercell, which needs this reference lattice.
        reference_cell = self._convergence_sweep_cell(active_definition)
        if reference_cell is None and self._document.project.host_structure_id is not None:
            reference_cell = self._document.structures.get(self._document.project.host_structure_id)

        self._project_service.set_recommended_kpoint_grid(
            self._document,
            grid,
            lattice_vectors=(
                [list(row) for row in reference_cell.lattice_vectors] if reference_cell is not None else None
            ),
            method=method,
            energy_threshold=energy_threshold,
            force_threshold=force_threshold,
            stress_threshold=stress_threshold,
            cell_id=(
                definition_cell_id
                if definition_cell_id is not None
                else (str(self._document.project.host_structure_id) if self._document.project.host_structure_id else None)
            ),
            xc_functional=xc_functional,
        )
        if stress_threshold is None:
            self._document.ui_state.pop("default_kpoint_stress_threshold", None)
        else:
            self._document.ui_state["default_kpoint_stress_threshold"] = float(stress_threshold)
        self._refresh_project_defaults_views()
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

    def _convergence_definition_xc_functional(self, definition) -> str | None:
        """Return the one functional a convergence definition's sweep ran with."""

        if definition is None:
            return None
        config = self._deserialize_convergence_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return None
        return convergence_config_xc_functional(config)

    def _confirm_adopted_xc_functional(self, xc_functional: str | None) -> bool:
        """Confirm adopting a result converged with a non-default functional.

        Adopting a convergence result also makes the functional it was
        converged with the project default, so switching functional is
        confirmed rather than done silently.
        """

        if self._document is None or xc_functional is None:
            return True
        current = self._document.project.recommended_xc_functional
        if current is None or current == xc_functional:
            return True
        answer = QMessageBox.question(
            self,
            "Change project functional?",
            f"These convergence calculations used the {xc_functional} functional, "
            f"but the project default functional is {current}.\n\n"
            f"Adopting this result will also change the project default "
            f"functional to {xc_functional}. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

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
        ui_state = self._document.ui_state
        for key in (
            "default_cut_off_energy",
            "default_cut_off_energy_method",
            "default_cut_off_energy_threshold",
            "default_cut_off_force_threshold",
            "default_cut_off_stress_threshold",
        ):
            ui_state.pop(key, None)
        # The pinned overview plot documents the adopted value. Drop it, or
        # typing that same number back in later would resurrect the plot for a
        # default that is now manual.
        self._store_convergence_overview_snapshot("cutoff", None)
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
        ui_state["default_cut_off_energy"] = value
        ui_state["default_cut_off_energy_method"] = MANUAL_DEFAULT_METHOD
        self._finish_project_default_edit(f"Project default cutoff set to {value:.8g} eV")

    def _on_project_kpoint_grid_change_requested(self, kpoint_grid: object) -> None:
        """Apply a grid typed into the Project Defaults dialog (``None`` clears).

        A manual grid is taken to be for the host unit cell, which is recorded
        as its reference lattice so supercell jobs rescale it as usual.
        """

        if self._document is None:
            return
        self._document.ui_state.pop("default_kpoint_stress_threshold", None)
        self._store_convergence_overview_snapshot("kpoint", None)
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
            view.set_parameter_species_options(_structure_species_options(structure))
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
            [{"label": asset.label, "matrix": [list(row) for row in asset.matrix]} for asset in assets]
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

    def _serialize_uep_config(self, config: UepConfig) -> dict[str, object]:
        return serialize_uep_config(config)

    def _deserialize_uep_config(self, payload: object) -> UepConfig | None:
        return deserialize_uep_config(payload)

    @staticmethod
    def _normalized_parameter_values(values: Mapping[str, object]) -> object:
        return to_jsonable({str(name): value for name, value in values.items()})

    def _pristine_relaxation_row_signature(self, row: PristineRelaxationRow) -> tuple[object, ...]:
        return (
            bool(row.fix_all_cell),
            normalized_supercell_matrix(row.supercell_matrix),
            self._normalized_parameter_values(row.parameter_values),
            tuple(str(name) for name in row.added_advanced),
        )

    def _muon_relaxation_row_signature(self, row: MuonRelaxationRow) -> tuple[object, ...]:
        return (
            str(row.site_id),
            row.label,
            tuple(float(value) for value in row.initial_fractional_position),
            str(row.charge_state),
            normalized_supercell_matrix(row.supercell_matrix),
            self._normalized_parameter_values(row.parameter_values),
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
            self._normalized_parameter_values(row.parameter_values),
            tuple(str(name) for name in row.added_advanced),
        )

    def _definition_workflow_context_matches(self, source_definition, target_definition) -> bool:
        return (
            source_definition.workflow_stage == target_definition.workflow_stage
            and source_definition.calc_type == target_definition.calc_type
            and source_definition.input_structure_ids == target_definition.input_structure_ids
            and definition_output_directory(self._document, source_definition)
            == definition_output_directory(self._document, target_definition)
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
        for definition in self._pristine_relaxation_controller.definitions():
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
        for definition in self._muon_relaxation_controller.definitions():
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
                and job.status in ACTIVE_JOB_STATUSES
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
        for definition in self._uep_controller.definitions():
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

    def _uep_source_structure_id_for_job(self, job_id: UUID) -> UUID | None:
        if self._document is None:
            return None
        for run in self._document.runs.values():
            if run.job_id != job_id or run.definition_id is None:
                continue
            definition = self._document.calculations.get(run.definition_id)
            if definition is not None and definition.input_structure_ids:
                return definition.input_structure_ids[0]
        for definition in self._uep_controller.definitions():
            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict):
                continue
            if str(job_id) in {str(value) for value in mapping.values()} and definition.input_structure_ids:
                return definition.input_structure_ids[0]
        return None

    def _active_uep_job_ids(self) -> set[UUID]:
        definition = self._uep_controller.active_definition()
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

    def _connect_workflow_row_actions(
        self, view, controller: WorkflowJobController, *, reset: bool = True
    ) -> None:
        """Route a workflow table's row-menu Cancel / Inspect Failure / Reset signals.

        Each signal carries the selected row keys; the generic handlers below
        resolve them to jobs through ``controller.jobs_for_keys``. ``reset`` is
        ``False`` for a view without a ``reset_failed_requested`` signal.
        """

        view.cancel_requested.connect(partial(self._cancel_workflow_rows, controller))
        view.inspect_failed_requested.connect(partial(self._inspect_workflow_rows, controller))
        if reset:
            view.reset_failed_requested.connect(partial(self._reset_workflow_rows, controller))

    def _cancel_workflow_rows(self, controller, keys: object) -> None:
        """Cancel the jobs behind a workflow table's rows (row menu "Cancel")."""

        self._cancel_jobs(controller.jobs_for_keys(list(keys)))

    def _inspect_workflow_rows(self, controller, keys: object) -> None:
        """Show failure diagnostics for a workflow table's rows (row menu "Inspect Failure")."""

        self._inspect_failed_jobs(controller.jobs_for_keys(list(keys)))

    def _reset_workflow_rows(self, controller, keys: object) -> None:
        """Reset the failed jobs behind a workflow table's rows (row menu "Reset Failed Jobs...")."""

        self._reset_failed_jobs(controller.jobs_for_keys(list(keys)))

    def _on_remove_pristine_relaxation_rows(self, payload: object) -> None:
        labels = self._removed_row_labels(payload)
        if self._document is None or not labels:
            return
        definitions = self._pristine_relaxation_controller.definitions()
        should_autosave = self._relaxation_removal_touches_project_files(
            definitions,
            lambda key: str(key) in labels,
        )
        if not self._cleanup_pristine_relaxation_rows(definitions, labels):
            self._restore_pristine_relaxation_form_from_definition()
            return
        for definition in definitions:
            self._remove_pristine_relaxation_labels_from_definition(definition, labels)
        self._pristine_relaxation_controller.refresh_view()
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
        definitions = self._muon_relaxation_controller.definitions()
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
            base = definition_output_directory(self._document, definition)
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
        definition = self._pristine_relaxation_controller.active_definition()
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
        self._pristine_relaxation_controller.refresh_view()

    def _restore_muon_relaxation_form_from_definition(self) -> None:
        definition = self._muon_relaxation_controller.active_definition()
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
        for definition in self._uep_controller.definitions():
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
        return serialize_pristine_relaxation_config(config)

    def _deserialize_pristine_relaxation_config(self, payload: object) -> PristineRelaxationConfig | None:
        return deserialize_pristine_relaxation_config(payload)

    def _on_import_legacy_relaxations(self) -> None:
        """Scan a legacy MuFinder folder and import the relaxations chosen from it."""

        document = self._document
        if document is None:
            QMessageBox.information(
                self,
                "No project open",
                "Open or create a project before importing legacy relaxations.",
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
            self, "Select a legacy MuFinder working folder"
        )
        if not directory:
            return

        self._begin_legacy_scan(Path(directory), host)

    def _begin_legacy_scan(self, root: Path, host: Structure) -> None:
        """Scan ``root`` on a worker thread, then review and commit."""

        self.statusBar().showMessage(f"Scanning {root.name} for legacy relaxations...")
        if self._import_legacy_action is not None:
            self._import_legacy_action.setEnabled(False)
        self._legacy_scan_host = host
        self._legacy_scan_root = root
        start_legacy_scan(
            self,
            root,
            [list(vector) for vector in host.lattice_vectors],
            list(host.species),
            self._on_legacy_scan_completed,
            self._on_legacy_scan_failed,
            self._on_legacy_scan_progress,
        )

    @Slot(int, int)
    def _on_legacy_scan_progress(self, done: int, total: int) -> None:
        root = self._legacy_scan_root
        name = root.name if root is not None else "folder"
        self.statusBar().showMessage(f"Scanning {name}: relaxation {done} of {total}...")

    @Slot(str)
    def _on_legacy_scan_failed(self, message: str) -> None:
        self._finish_legacy_scan()
        QMessageBox.warning(self, "Import failed", f"Could not scan the folder:\n{message}")

    @Slot(object)
    def _on_legacy_scan_completed(self, scan: object) -> None:
        self._finish_legacy_scan()
        document = self._document
        host = self._legacy_scan_host
        if document is None or host is None or not isinstance(scan, LegacyFolderScan):
            return

        if not scan.runs:
            QMessageBox.information(
                self,
                "Nothing to import",
                "No muon relaxations were found in that folder.\n\n"
                + "\n".join(scan.warnings),
            )
            return

        dialog = LegacyImportDialog(scan, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_runs()
        if not selected:
            return

        self._commit_legacy_import(scan, selected, host)

    def _finish_legacy_scan(self) -> None:
        if self._import_legacy_action is not None:
            self._import_legacy_action.setEnabled(True)
        self.statusBar().clearMessage()

    def _commit_legacy_import(
        self, scan: LegacyFolderScan, runs: list, host: Structure
    ) -> None:
        """Build entities and files, attach the rows, and save."""

        document = self._document
        if document is None:
            return

        output_root = document.project_root / "files" / WorkflowStage.MUON_RELAXATION.value
        definition = self._muon_relaxation_controller.active_definition()
        existing_labels: list[str] = []
        if definition is not None:
            config = self._deserialize_muon_relaxation_config(
                definition.parameters.get("configuration", {})
            )
            if config is not None:
                existing_labels = [row.label for row in config.rows]

        try:
            outcome = commit_legacy_import(
                document,
                scan=scan,
                runs=runs,
                host_structure=host,
                output_root=output_root,
                existing_labels=existing_labels,
            )
        except OSError as exc:
            QMessageBox.warning(
                self, "Import failed", f"Could not write the imported files:\n{exc}"
            )
            return

        if not self.merge_imported_relaxation_rows(outcome, output_directory=output_root):
            QMessageBox.warning(
                self,
                "Import incomplete",
                "The relaxations were imported but could not be attached to the "
                "relaxations table.",
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
            failure_message="Legacy relaxations were imported, but autosave failed.",
        )

        messages = [f"Imported {len(outcome.imported)} relaxation(s)."]
        if outcome.skipped:
            messages.append(f"{len(outcome.skipped)} were skipped.")
        row_warnings = [note for item in outcome.imported for note in item.warnings]
        if row_warnings:
            messages.append("")
            messages.extend(f"- {note}" for note in dict.fromkeys(row_warnings))
        QMessageBox.information(self, "Import complete", "\n".join(messages))
        self.statusBar().showMessage(f"Imported {len(outcome.imported)} legacy relaxation(s)")

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

        labels = [
            f"{label} ({count} relaxation(s))" for label, count, _batch_id in batches
        ]
        choice, accepted = QInputDialog.getItem(
            self, "Remove Imported Relaxations", "Import batch", labels, 0, False
        )
        if not accepted:
            return
        batch_id = batches[labels.index(choice)][2]

        refs = imported_relaxations_in_batch(document, batch_id)
        directories = {
            ref.job_id: path
            for ref in refs
            if (path := self._job_output_directory(document.jobs[ref.job_id])) is not None
        }
        plan = plan_removal(document, refs, job_directories=directories)

        if plan.blockers:
            QMessageBox.warning(
                self,
                "Still in use",
                "These imported relaxations cannot be removed because they are still "
                "used by:\n\n"
                + "\n".join(f"- {item}" for item in plan.blockers)
                + "\n\nRemove those first.",
            )
            return

        confirmed = QMessageBox.question(
            self,
            "Remove imported relaxations?",
            f"Permanently remove {len(plan.targets)} imported relaxation(s) and delete "
            "their job directories, including the archived original input files?\n\n"
            "MuFinder did not run these calculations and cannot recreate them.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return

        outcome = remove_imported_relaxations(document, plan, batch_id=batch_id)
        removed, errors = outcome.removed, outcome.errors
        definition = self._muon_relaxation_controller.active_definition()
        if definition is not None:
            detach_rows_from_definition(definition, plan.targets)
            self._muon_relaxation_view.set_form_state(
                self._muon_relaxation_form_state_without(plan.targets)
            )
            document.ui_state["muon_relaxation_form"] = (
                self._muon_relaxation_view.collect_state()
            )

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
            QMessageBox.warning(self, "Removal issues", "\n".join(errors))
        self.statusBar().showMessage(f"Removed {removed} imported relaxation(s)")

    def _muon_relaxation_form_state_without(self, targets) -> dict[str, object]:
        """Return the widget's form state with ``targets``' rows dropped."""

        state = self._muon_relaxation_view.collect_state()
        rows = state.get("rows", []) if isinstance(state, dict) else []
        removed = {str(ref.site_id) for ref in targets}
        return {
            "rows": [row for row in rows if str(row.get("site_id", "")) not in removed]
        }

    def _imported_relaxation_batches(self) -> list[tuple[str, int, UUID]]:
        """Return ``(label, count, batch_id)`` for every import batch present."""

        document = self._document
        if document is None:
            return []
        counts: dict[str, int] = {}
        for site in document.sites.values():
            raw = site.properties.get("import_batch_id")
            if raw:
                counts[str(raw)] = counts.get(str(raw), 0) + 1

        labels: dict[str, str] = {}
        for record in document.provenance:
            if record.action != IMPORT_ACTION:
                continue
            raw_batch = record.parameters.get("batch_id")
            source = record.parameters.get("source_folder")
            if raw_batch:
                labels[str(raw_batch)] = Path(str(source)).name if source else "imported batch"

        batches: list[tuple[str, int, UUID]] = []
        for raw_batch, count in counts.items():
            try:
                batch_id = UUID(raw_batch)
            except ValueError:
                continue
            batches.append((labels.get(raw_batch, "imported batch"), count, batch_id))
        return sorted(batches, key=lambda item: item[0])

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

        mapping = self._ensure_label_mapping(definition)
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

    def _serialize_muon_relaxation_config(self, config: MuonRelaxationConfig) -> dict[str, object]:
        return serialize_muon_relaxation_config(config)

    def _deserialize_muon_relaxation_config(self, payload: object) -> MuonRelaxationConfig | None:
        return deserialize_muon_relaxation_config(payload)

    def _definition_for_pristine_relaxation_label(self, label: str):
        definition = self._pristine_relaxation_controller.active_definition()
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

    def _muon_relaxation_definition_and_row_for_job(self, job: Job):
        if self._document is None:
            return None, None
        for definition in self._muon_relaxation_controller.definitions():
            config = self._deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
            if config is None:
                continue
            for row in config.rows:
                candidate = self._muon_relaxation_controller.job_for_label(definition, row.label, row.charge_state)
                if candidate is not None and candidate.id == job.id:
                    return definition, row
        return None, None

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
    def _coerce_fractional_triplet(value: object) -> tuple[float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None

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
        self._document.ui_state[_RELAXATION_ANALYSIS_STATE_KEY] = [dict(workspace) for workspace in workspaces]

    def _completed_muon_relaxation_sources(self) -> list[CompletedRelaxationAnalysisSource]:
        if self._document is None:
            return []
        host = self._project_service.active_host_structure(self._document)
        if host is None:
            return []

        sources: list[CompletedRelaxationAnalysisSource] = []
        seen: set[UUID] = set()
        for definition in self._muon_relaxation_controller.definitions():
            if host.id not in definition.input_structure_ids:
                continue
            config = self._deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
            if config is None:
                continue
            for row in config.rows:
                job = self._muon_relaxation_controller.job_for_label(definition, row.label, row.charge_state)
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
                total_energy = total_energy_for_job(self._document, job.id)
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

        definition = self._pristine_relaxation_controller.active_definition()
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

        self._refresh_workflow_supercell_options(self._pristine_relaxation_view)
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
        self._refresh_workflow_supercell_options(self._muon_relaxation_view)
        self._muon_relaxation_view.set_row_supercell(row_id, matrix)
        self.statusBar().showMessage("Selected muon-relaxation simulation cell")

    def _notify_adapter_messages(self, expansion: AdapterExpansion) -> None:
        """Surface any adapter adjustment messages (e.g. k-point rescaling).

        Every workflow adapter reports adjustment notes on
        :attr:`AdapterExpansion.messages`; convergence has none, while
        relaxation/UEP report supercell k-point rescaling. Shown as an
        information dialog so the note is not overwritten by the trailing
        ``statusBar`` confirmation on the configure path.
        """

        messages = expansion.messages
        if not messages:
            return
        QMessageBox.information(
            self,
            "K-point grid adjusted",
            "\n".join(messages),
        )

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
        self._notify_adapter_messages(expansion)

        output_root = self._document.project_root / "files" / "pristine_relaxation"
        definition = self._project_service.register_pristine_relaxation_definition(
            self._document,
            input_structure_id=host.id,
            parameters=self._serialize_pristine_relaxation_config(config),
            output_directory=output_root,
        )
        self._sync_equivalent_pristine_relaxation_mappings(definition)
        self._document.ui_state[self._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)

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
        self._notify_adapter_messages(expansion)
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

        definition = self._pristine_relaxation_controller.active_definition()
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
        for row_label, state in sorted(self._pristine_relaxation_controller.build_states().items()):
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

    def _singlepoint_job_for_quantum_source_key(self, definition, key: str) -> Job | None:
        return self._quantum_corrections_controller.job_for_key(
            definition,
            key,
            mapping_name="singlepoint_job_ids_by_label",
            calc_type=QUANTUM_SINGLEPOINT_CALC_TYPE,
        )

    def _phonon_job_for_quantum_source_key(self, definition, key: str) -> Job | None:
        controller = self._quantum_corrections_controller
        job = controller.job_for_key(
            definition,
            key,
            mapping_name="phonon_job_ids_by_label",
            calc_type=QUANTUM_PHONON_CALC_TYPE,
        )
        if job is not None:
            return job
        return controller.job_for_key(definition, key, calc_type=QUANTUM_PHONON_CALC_TYPE)

    def _quantum_stage_state(self, definition, candidate: _QuantumSourceCandidate) -> _QuantumStageState:
        phonon_job = self._phonon_job_for_quantum_source_key(definition, candidate.key)
        if phonon_job is not None:
            return _QuantumStageState(
                stage=_QUANTUM_STAGE_PHONON,
                status=effective_row_status(phonon_job),
                current_job=phonon_job,
            )

        singlepoint_job = self._singlepoint_job_for_quantum_source_key(definition, candidate.key)
        if singlepoint_job is None:
            return _QuantumStageState(stage=_QUANTUM_STAGE_SINGLEPOINT, status="ready")
        if singlepoint_job.status == JobStatus.COMPLETED:
            return _QuantumStageState(stage=_QUANTUM_STAGE_PHONON, status="ready")
        return _QuantumStageState(
            stage=_QUANTUM_STAGE_SINGLEPOINT,
            status=effective_row_status(singlepoint_job),
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
        """Return whether a Quantum Corrections row can be rewound to stage 1.

        Quantum Corrections is the one workflow whose recovery is a *stage*
        reset rather than a job reset: the row stages a singlepoint job and then
        a phonon job into a shared output directory, so
        :meth:`_on_reset_quantum_corrections_to_stage_one` deletes both job
        records instead of calling ``reset_failed_job``. That makes this
        predicate the workflow's whole in-table escape hatch.

        A row qualifies once it has progressed past "nothing staged" and has no
        job still running: either the phonon job exists and is not active, or the
        row is still on stage 1 with a singlepoint job that has finished — which
        includes a *failed* (or stopped/cancelled) one. Without that last case a
        stage-1 failure with no phonon job left the row permanently stuck: it
        showed ``"failed"``, so its Submit stayed disabled, and Reset was
        refused too.
        """

        singlepoint_job = self._singlepoint_job_for_quantum_source_key(definition, candidate.key)
        if singlepoint_job is None:
            return False
        phonon_job = self._phonon_job_for_quantum_source_key(definition, candidate.key)
        if phonon_job is not None:
            return phonon_job.status not in ACTIVE_JOB_STATUSES
        return singlepoint_job.status in _QUANTUM_STAGE_ONE_RESETTABLE_STATUSES

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
        definition = self._quantum_corrections_controller.active_definition()
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
        definition = self._quantum_corrections_controller.active_definition()
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
        for definitions in (self._energy_barriers_definitions(), self._energy_barriers_controller.definitions()):
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
                        self._archive_cutoff_directory(job_dir, definition_output_directory(self._document, definition))
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
        # NEB is seeded directly from each generated path (the union of form-state and
        # committed definition sources); CASTEP interpolates the band between the endpoints.
        # No transition-state-search definition is required.
        candidates: list[_EnergyBarrierNebCandidate] = []
        for source in self._energy_barrier_generated_sources():
            key = str(source.get("key", "")).strip()
            if not key:
                continue
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

    def _energy_barrier_neb_sources_by_job_id(self) -> dict[UUID, dict[str, object]]:
        """Map each materialized NEB job back to its generated-path source metadata."""

        sources_by_key = {
            str(source.get("key", "")).strip(): source
            for source in self._energy_barrier_generated_sources()
        }
        by_job_id: dict[UUID, dict[str, object]] = {}
        for definition in self._energy_barriers_controller.definitions():
            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict):
                continue
            for key, raw_job_id in mapping.items():
                source = sources_by_key.get(str(key).strip())
                if source is None or not isinstance(raw_job_id, str):
                    continue
                try:
                    by_job_id[UUID(raw_job_id)] = source
                except ValueError:
                    continue
        return by_job_id

    def _energy_barrier_neb_result_path_label(
        self,
        candidate: _EnergyBarrierCompletedNebCandidate,
        source: Mapping[str, object] | None,
    ) -> str:
        """Describe a completed NEB result as ``initial → final``, falling back to the job label."""

        if source is not None and self._document is not None:
            labels: list[str] = []
            for field in ("reactant_site_id", "product_site_id"):
                site_id = self._uuid_from_source(source, field)
                site = self._document.sites.get(site_id) if site_id is not None else None
                labels.append((site.label or "").strip() if site is not None else "")
            if all(labels):
                return f"{labels[0]} → {labels[1]}"
        return candidate.job.label

    def _serialize_energy_barrier_neb_sources(self, keys: list[str]) -> list[dict[str, object]]:
        candidates_by_key = {candidate.key: candidate for candidate in self._energy_barrier_neb_source_candidates()}
        serialized: list[dict[str, object]] = []
        for key in keys:
            candidate = candidates_by_key.get(key)
            if candidate is None:
                continue
            serialized.append(dict(candidate.source))
        return serialized

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
        definition = self._energy_barriers_controller.active_definition()
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

    def _build_energy_barrier_neb_states(self) -> dict[str, RelaxationRunState]:
        states: dict[str, RelaxationRunState] = {}
        definition = self._energy_barriers_controller.active_definition()
        for candidate in self._energy_barrier_neb_source_candidates():
            job = self._energy_barriers_controller.job_for_neb_source_key(definition, candidate.key)
            if job is None:
                states[candidate.key] = RelaxationRunState(status="ready")
                continue
            states[candidate.key] = RelaxationRunState(
                status=effective_row_status(job),
                job_id=job.id,
                job_label=job.label,
                error_message=job.error_message,
            )
        return states

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
    def _fractional_position_block_from_structure(self, structure: Structure) -> PositionBlock:
        # Fractional coordinates so the NEB product/intermediate blocks use
        # POSITIONS_FRAC_PRODUCT, whose CASTEP reader accepts tagged species like the muon "H:mu".
        rows = [
            AtomicPosition(str(species), tuple(float(value) for value in fractional))
            for species, fractional in zip(structure.species, structure.fractional_positions, strict=True)
        ]
        return PositionBlock(positions=tuple(rows), unit="frac")

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
        definition = self._quantum_corrections_controller.active_definition()
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
            self._num_farms_from_extras(self._workflow_row_configuration(definition, candidate.key)[2])
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
        dialog.set_mixed_names(mixed_parameter_names(values_by_key))
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
        untouched = dialog.untouched_mixed_names()
        for key in key_list:
            self._store_workflow_row_configuration(
                definition,
                key,
                self._batch_row_payload(definition, key, dialog_payload, untouched),
            )
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

    def _on_configure_energy_barrier_result(self, payload: object) -> None:
        if self._document is None:
            return
        keys = _ordered_string_list(payload)
        if not keys:
            return
        neb_candidates_by_key = {c.key: c for c in self._energy_barrier_neb_source_candidates()}
        candidates = [c for key in keys if (c := neb_candidates_by_key.get(key)) is not None]
        if not candidates:
            return
        definition = self._energy_barriers_controller.active_definition()
        source_candidates_by_key = {c.key: c for c in self._eligible_energy_barrier_source_candidates()}
        source_defaults_cache: dict[UUID, dict[str, object]] = {}
        values_by_key: list[dict[str, object]] = []
        advanced_names: set[str] = set()
        reactant_structures: list[Structure | None] = []
        for candidate in candidates:
            reactant_structure = self._document.structures.get(self._uuid_from_source(candidate.source, "reactant_structure_id"))
            reactant_structures.append(reactant_structure)
            values, row_advanced_names, _extras = self._energy_barrier_neb_row_configuration_with_source_defaults(
                definition,
                candidate.key,
                target_lattice=reactant_structure.lattice_vectors if reactant_structure is not None else None,
                source_candidate=source_candidates_by_key.get(candidate.key),
                source_defaults_cache=source_defaults_cache,
            )
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

        # Show a farm count only when every selected path already agrees;
        # a mixed selection opens at 1. Accepting the dialog then writes the
        # shown value to every selected row, as batch configure does for the
        # parameter rows too.
        num_farms_values = [
            self._num_farms_from_extras(self._workflow_row_configuration(definition, candidate.key)[2])
            for candidate in candidates
        ]
        common_num_farms = (
            num_farms_values[0]
            if num_farms_values and all(value == num_farms_values[0] for value in num_farms_values)
            else 1
        )
        dialog = NudgedElasticBandConfigDialog(
            specs=self._energy_barrier_neb_parameter_specs(),
            num_farms=common_num_farms,
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
        dialog.set_mixed_names(mixed_parameter_names(values_by_key))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if read_only:
            return

        row_payload = self._workflow_dialog_payload(
            dialog,
            extra_payload={"num_farms": dialog.num_farms()},
        )
        definition = self._ensure_energy_barrier_neb_definition([candidate.key for candidate in candidates])
        if definition is None:
            return
        untouched = dialog.untouched_mixed_names()
        for candidate in candidates:
            self._store_workflow_row_configuration(
                definition,
                candidate.key,
                self._batch_row_payload(definition, candidate.key, row_payload, untouched),
            )
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
        for definition in self._quantum_corrections_controller.definitions():
            for key in keys:
                job = self._job_for_quantum_source_key(definition, key)
                if job is None or job.id in seen:
                    continue
                selected.append(job)
                seen.add(job.id)
        return selected

    def _on_reset_quantum_corrections_to_stage_one(self, keys: list[str]) -> None:
        if self._document is None:
            return
        definition = self._quantum_corrections_controller.active_definition()
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
                "Only Quantum Corrections rows that have staged a job, and whose job is no longer "
                "running, can be reset to stage 1. That includes rows whose stage-1 singlepoint "
                "failed, was stopped, or was cancelled.",
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
            owner = phonon_job or singlepoint_job
            output_dir = self._job_output_directory(owner) if owner is not None else None
            refusal = (
                self._directory_ownership_refusals(
                    [owner],
                    shared_with={job.id for job in (phonon_job, singlepoint_job) if job is not None},
                    check_remote=False,
                ).get(owner.id)
                if owner is not None
                else None
            )
            if refusal:
                # The whole directory is deleted below, so it must hold only
                # this row's two stages.
                cleanup_errors.append(f"{refusal}; the row was not reset")
                continue
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
        for definition in self._energy_barriers_controller.definitions():
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
        # Report each image's energy relative to the initial image: absolute
        # CASTEP energies are ~10⁵ eV, so differences vanish in the display.
        initial_energy = transition_state.images[0].energy
        image_rows = [
            EnergyBarrierPreviewImageRow(
                label=image.label,
                reaction_coordinate=f"{image.reaction_coordinate:.6g}",
                energy=f"{image.energy - initial_energy:+.4g} eV",
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
        self._refresh_workflow_supercell_options(self._uep_view)
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
        self._notify_adapter_messages(expansion)

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

    def _on_open_uep_analysis_requested(self, label: object) -> None:
        if self._document is None:
            return
        row_label = str(label).strip()
        if not row_label:
            return

        definition = self._uep_controller.active_definition()
        if definition is None:
            QMessageBox.information(self, "No UEP workflow", "Configure and complete a UEP source job first.")
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
            initial_state = self._migrate_legacy_uep_state_potential_grid(job.id, initial_state)
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
            self._document.ui_state[self._UEP_ANALYSIS_STATES_KEY] = {**states, str(job_id): migrated}
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
    def _serialize_convergence_config(config: ConvergenceConfig) -> dict[str, object]:
        return serialize_convergence_config(config)

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
        # Drop/redraw a visible summary whose pins are keyed to the old host.
        self._refresh_muon_sites_summary(only_if_visible=True)
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
