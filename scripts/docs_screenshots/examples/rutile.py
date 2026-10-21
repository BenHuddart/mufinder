"""Rutile TiO2: the project most getting-started and setup scenes open.

:func:`fixture_project` is rutile (P4_2/mnm, a = 4.594 Å, c = 2.959 Å, O at 4f
with x = 0.3048), a host small enough to read in every view, with ``PBE`` as
the project functional, a saved 2x2x2 supercell, configured but unsubmitted
cutoff and k-point sweeps, and candidate muon sites.
:func:`converged_project` adds adopted convergence defaults with illustrative
pinned plots, and :func:`dashboard_project` adds three finished cutoff jobs
for the Job Dashboard.
"""

from __future__ import annotations

# The render module configures Qt and VTK, so it is imported before them.
from ..render import (
    XC_FUNCTIONAL,
    Dialog,
    Page,
    Scene,
    Window,
    focus_default_button,
    frame_viewers,
    hide_catalog_notice,
    new_project,
    settle,
    when_modal,
)

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtWidgets import QApplication, QWidget

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectDocument, ProjectService

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow

RUTILE_A = 4.594
RUTILE_C = 2.959
RUTILE_X = 0.3048
SUPERCELL_2X2X2 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
CUTOFFS_EV = (300.0, 400.0, 500.0, 600.0, 700.0, 800.0)
KPOINT_MIN_SPACING = 0.07  # 1/Å, the Generate From Spacing... value


def rutile_host() -> Structure:
    x = RUTILE_X
    return Structure(
        label="TiO2 (rutile)",
        lattice_vectors=[[RUTILE_A, 0.0, 0.0], [0.0, RUTILE_A, 0.0], [0.0, 0.0, RUTILE_C]],
        species=["Ti", "Ti", "O", "O", "O", "O"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [x, x, 0.0],
            [1.0 - x, 1.0 - x, 0.0],
            [0.5 + x, 0.5 - x, 0.5],
            [0.5 - x, 0.5 + x, 0.5],
        ],
    )


def _configure_sweep(
    service: ProjectService,
    document: ProjectDocument,
    *,
    stage: WorkflowStage,
    form_state: dict[str, object],
) -> None:
    """Configure a convergence sweep as its page's configure action does.

    The page widget turns ``form_state`` into a configuration with the
    catalog's default parameters and the project functional, and the
    definition is registered for the unit cell, leaving every row ready to run.
    """

    from mufinder.app.convergence_widget import CutoffConvergenceWidget
    from mufinder.app.kpoint_convergence_widget import KPointConvergenceWidget
    from mufinder.app.workflow_config_codecs import serialize_convergence_config
    from mufinder.app.workflow_controllers.convergence_cutoff import ConvergenceCutoffController
    from mufinder.app.workflow_controllers.convergence_kpoint import ConvergenceKpointController

    cutoff = stage == WorkflowStage.CONVERGENCE_CUTOFF
    widget = CutoffConvergenceWidget() if cutoff else KPointConvergenceWidget()
    controller = ConvergenceCutoffController if cutoff else ConvergenceKpointController
    try:
        widget.set_xc_functional_default(document.project.recommended_xc_functional)
        widget.set_form_state(form_state)
        config = widget.build_config()
        if config is None:
            raise RuntimeError(f"The {stage.value} sweep has no valid configuration")
        resolution = service.resolve_workflow_structure(document, stage, normalize_state=True)
        register = (
            service.register_convergence_cutoff_definition
            if cutoff
            else service.register_convergence_kpoint_definition
        )
        definition = register(
            document,
            input_structure_id=resolution.structure.id,
            parameters=serialize_convergence_config(config),
            output_directory=document.project_root / "files" / stage.value,
            simulation_cell_mode=resolution.normalized_selection.mode,
        )
        document.ui_state[controller.form_state_key] = widget.collect_state()
        document.ui_state[controller.active_key] = str(definition.id)
    finally:
        widget.deleteLater()


def _build_fixture(directory: Path) -> tuple[ProjectService, ProjectDocument]:
    """The fixture project in memory; :func:`fixture_project` saves it."""

    from mufinder.science.kpoint_grids import generate_kpoint_grids

    host = rutile_host()
    service, document = new_project(directory, "Rutile TiO2", host)
    service.set_recommended_xc_functional(document, XC_FUNCTIONAL)

    service.create_workflow_supercell_asset(
        document,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=SUPERCELL_2X2X2,
        label="Rutile 2x2x2",
    )
    document.project.supercell_matrix = [list(row) for row in SUPERCELL_2X2X2]

    _configure_sweep(
        service,
        document,
        stage=WorkflowStage.CONVERGENCE_CUTOFF,
        form_state={"cutoff_energies": list(CUTOFFS_EV)},
    )
    grids = generate_kpoint_grids(host.lattice_vectors, KPOINT_MIN_SPACING)
    _configure_sweep(
        service,
        document,
        stage=WorkflowStage.CONVERGENCE_KPOINT,
        form_state={
            "kpoint_grids": [list(grid) for grid in grids],
            "generated_for_cell_id": str(host.id),
            "generated_for_lattice": host.lattice_vectors,
        },
    )

    # One muon bonded to each oxygen, the hydroxyl-like starts hydrogen takes
    # in rutile, plus a few random interstitial starts.
    service.generate_candidate_sites(
        document,
        method="near_element",
        parameters={"element": "O", "muons_per_atom": 1, "bond_distance": 1.0, "min_muon_atom": 0.8},
    )
    service.generate_candidate_sites(
        document,
        method="random",
        parameters={
            "selection_mode": "count",
            "count": 4,
            "min_muon_atom": 1.0,
            "max_attempts": 50000,
            "seed": "7",
            "include_symmetry_equivalents": False,
        },
    )
    return service, document


def fixture_project(directory: Path) -> Path:
    """The rutile project most scenes open; see the module docstring."""

    service, document = _build_fixture(directory)
    service.save_project(document)
    return document.project_root


# The Convergence Testing overview pins the examiner plot of each adopted
# default. Pinning one needs a finished sweep, so the overview scene adopts a
# cutoff and a grid with illustrative sweep data: smooth decays towards the
# converged values, not CASTEP results.

ADOPTED_CUTOFF_EV = 600.0
ADOPTED_GRID = (3, 3, 4)
ENERGY_THRESHOLD = 0.001  # eV per atom
FORCE_THRESHOLD = 0.05  # eV/Å
_CUTOFF_SWEEP = {  # cutoff: (energy above the 800 eV value, max force)
    300.0: (0.92, 0.41),
    400.0: (0.21, 0.18),
    500.0: (0.038, 0.072),
    600.0: (0.0042, 0.031),
    700.0: (0.0006, 0.012),
    800.0: (0.0, 0.010),
}
_KPOINT_SWEEP = {  # grid: (energy above the densest grid's value, max force)
    (2, 2, 3): (0.95, 0.12),
    (3, 3, 4): (0.0031, 0.008),
    (4, 4, 5): (0.0004, 0.004),
    (5, 5, 7): (0.0, 0.003),
}
_CONVERGED_ENERGY_EV = -4985.274


def converged_project(directory: Path) -> Path:
    """The fixture project with a cutoff and a k-point grid adopted and pinned."""

    from mufinder.science.kpoint_grids import format_kpoint_grid

    service, document = _build_fixture(directory)
    host = service.active_host_structure(document)
    host_id = str(host.id)
    criteria = {
        "method": "delta-threshold",
        "energy_threshold": ENERGY_THRESHOLD,
        "force_threshold": FORCE_THRESHOLD,
        "cell_id": host_id,
        "xc_functional": XC_FUNCTIONAL,
    }
    service.set_recommended_cutoff(document, ADOPTED_CUTOFF_EV, **criteria)
    service.set_recommended_kpoint_grid(
        document, ADOPTED_GRID, lattice_vectors=host.lattice_vectors, **criteria
    )

    def snapshot(default_value: object, sweep: dict[Any, tuple[float, float]], x: Callable[[Any], float]) -> dict:
        return {
            "host_structure_id": host_id,
            "simulation_cell_structure_id": host_id,
            "default_value": default_value,
            "method": "delta-threshold",
            "energy_threshold": ENERGY_THRESHOLD,
            "force_threshold": FORCE_THRESHOLD,
            "plot_data": {
                "cutoffs": [x(key) for key in sweep],
                "energies": [_CONVERGED_ENERGY_EV + energy for energy, _force in sweep.values()],
                "forces": [force for _energy, force in sweep.values()],
                "energy_threshold": ENERGY_THRESHOLD,
                "force_threshold": FORCE_THRESHOLD,
            },
        }

    kpoint = snapshot(list(ADOPTED_GRID), _KPOINT_SWEEP, lambda grid: float(np.prod(grid)))
    kpoint["point_labels"] = {str(float(np.prod(grid))): format_kpoint_grid(grid) for grid in _KPOINT_SWEEP}
    document.ui_state["convergence_overview_snapshots"] = {
        "cutoff": snapshot(ADOPTED_CUTOFF_EV, _CUTOFF_SWEEP, float),
        "kpoint": kpoint,
    }
    service.save_project(document)
    return document.project_root


def sweep_job_labels(document: ProjectDocument, stage: WorkflowStage) -> list[str]:
    """The job labels a submission of the ``stage`` sweep's rows would create."""

    from mufinder.app.convergence_adapter import expand_convergence_jobs
    from mufinder.app.workflow_config_codecs import deserialize_convergence_config
    from mufinder.workflow.bridge import structure_to_spec

    definition = next(item for item in document.calculations.values() if item.workflow_stage == stage)
    config = deserialize_convergence_config(definition.parameters["configuration"])
    structure = document.structures[definition.input_structure_ids[0]]
    expansion = expand_convergence_jobs(structure=structure_to_spec(structure), config=config)
    return [job.name for job in expansion.jobs]


def dashboard_project(directory: Path) -> Path:
    """The fixture project after a first cutoff run that needs attention.

    Three cutoff rows ran and ended failed, stopped and cancelled, the states
    that keep a job on the Job Dashboard.
    """

    from datetime import timedelta

    from mufinder.domain.enums import FailureReason, JobStatus
    from mufinder.domain.models import utc_now

    service, document = _build_fixture(directory)
    definition = next(
        definition
        for definition in document.calculations.values()
        if definition.workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF
    )
    outcomes = (  # cutoff, status, cause, message, minutes run
        (300.0, JobStatus.FAILED, FailureReason.CASTEP_ERROR, "SCF did not converge in 100 cycles", 7),
        (400.0, JobStatus.STOPPED, FailureReason.WALLTIME, "Stopped at the walltime limit", 60),
        (500.0, JobStatus.CANCELLED, None, None, 3),
    )
    labels = dict(zip(CUTOFFS_EV, sweep_job_labels(document, WorkflowStage.CONVERGENCE_CUTOFF), strict=True))
    mapping = definition.parameters.setdefault("job_ids_by_cutoff", {})
    finished = utc_now()
    for cutoff, status, reason, message, minutes in outcomes:
        job = service.add_job(
            document,
            label=labels[cutoff],
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        job.status = status
        job.failure_reason = reason
        job.error_message = message
        job.num_processes = 4
        job.submitted_at = job.started_at = finished - timedelta(minutes=minutes)
        job.completed_at = finished
        mapping[f"{cutoff:.12g}"] = str(job.id)
    service.save_project(document)
    return document.project_root


# --- Scene steps --------------------------------------------------------------


def open_supercell_editor(window: MainWindow) -> None:
    """Click ``Open Supercell Editor`` on the Supercell Editor page."""

    window._on_host_supercell_editor_requested()


def show_saved_supercell(dialog: QWidget) -> None:
    """Select the saved supercell, which loads its matrix and preview."""

    dialog._asset_table.selectRow(0)
    settle(QApplication.instance())
    frame_viewers(dialog)


def open_supercell_suggestions(window: MainWindow) -> None:
    """Click ``Suggest…`` in the Supercell Editor."""

    when_modal("SupercellEditorDialog", lambda editor: editor._suggest_button.click())
    open_supercell_editor(window)


def open_cutoff_parameters(window: MainWindow) -> None:
    """Click the ``Configure...`` button of the 500 eV cutoff row."""

    view = window._convergence_view
    view._table.cellWidget(CUTOFFS_EV.index(500.0), view._CONFIGURE_COLUMN).click()


def open_local_submission(window: MainWindow) -> None:
    """Open the dialog ``Run All Locally`` shows for the cutoff sweep.

    The run button first checks for a CASTEP executable, which the scratch
    settings do not have, so the dialog is opened with the sweep's job labels
    directly. It reports the computer's core count; an eight-core computer
    keeps the image the same on every build machine.
    """

    from mufinder.app import submission_dialog

    labels = sweep_job_labels(window._document, WorkflowStage.CONVERGENCE_CUTOFF)
    ideal_thread_count = submission_dialog.QThread.idealThreadCount
    submission_dialog.QThread.idealThreadCount = staticmethod(lambda: 8)
    try:
        dialog = submission_dialog.LocalSubmissionDialog(job_labels=labels, mpirun_configured=True, parent=window)
    finally:
        submission_dialog.QThread.idealThreadCount = ideal_thread_count
    dialog.exec()


def select_all_sites(page: QWidget) -> None:
    """Widen the preview and select every generated site, which marks them in it."""

    from PySide6.QtWidgets import QSplitter

    page.findChild(QSplitter).setSizes([520, 640])
    page._site_table.selectAll()
    settle(QApplication.instance())
    frame_viewers(page)


# --- Scenes -------------------------------------------------------------------

SCENES: dict[str, Scene] = {
    "project_home": Scene(fixture_project, None, Window()),
    "host_structure_page": Scene(
        fixture_project, "host_structure", Page("_host_structure_view", (1100, 760), prepare=frame_viewers)
    ),
    "supercell_editor": Scene(
        fixture_project, "supercell_editor", Dialog(open_supercell_editor, size=(1180, 760), prepare=show_saved_supercell)
    ),
    "supercell_suggestions": Scene(
        fixture_project, "supercell_editor", Dialog(open_supercell_suggestions, kind="SupercellSuggestionDialog")
    ),
    "convergence_overview": Scene(converged_project, "convergence", Page("_convergence_overview_view", (1100, 760))),
    "cutoff_convergence_page": Scene(fixture_project, "convergence_cutoff", Page("_convergence_view", (1100, 720))),
    "kpoint_convergence_page": Scene(
        fixture_project, "convergence_kpoint", Page("_kpoint_convergence_view", (1100, 760))
    ),
    "parameter_dialog": Scene(
        fixture_project, "convergence_cutoff", Dialog(open_cutoff_parameters, kind="ParameterDialog", size=(900, 520), prepare=hide_catalog_notice)
    ),
    "site_generation_page": Scene(
        fixture_project, "muon_candidates", Page("_muon_generation_view", (1160, 740), prepare=select_all_sites)
    ),
    "journal_page": Scene(fixture_project, "provenance_log", Page("_journal_view", (1100, 800))),
    "castep_setup_dialog": Scene(fixture_project, None, Dialog(lambda window: window._open_castep_setup_dialog())),
    "cluster_setup_dialog": Scene(fixture_project, None, Dialog(lambda window: window._open_cluster_setup_dialog())),
    "local_submission_dialog": Scene(
        fixture_project, "convergence_cutoff", Dialog(open_local_submission, prepare=focus_default_button)
    ),
    "job_dashboard": Scene(dashboard_project, "convergence_cutoff", Window(dashboard=True)),
}
