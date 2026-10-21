"""Render the documentation screenshots from the running application.

The screenshots are build outputs, not sources: ``make docs`` runs this script
first and Sphinx copies the images into the built site, so no image is
committed. Each screenshot is a :class:`Scene`: a project builder writes a
small saved project, the script opens it in a real ``MainWindow`` on one
navigator route, optionally drives the window, and grabs a :class:`Target`:
the whole window, one page widget, a dock, a modal dialog, or a non-modal
window.

Most scenes open :func:`fixture_project`, a rutile TiO2 project with a saved
2x2x2 supercell, configured but unsubmitted cutoff and k-point sweeps, and
candidate muon sites. :func:`converged_project` adds adopted convergence
defaults with illustrative pinned plots, and :func:`dashboard_project` adds
three finished cutoff jobs for the Job Dashboard. The Magnetic Fields scenes
open :func:`magnetic_project`, a one-ion cubic cell with a magnetic model. The
results and analysis scenes open :func:`results_project`, a silicon project
whose workflows have run through the application's own submission and
completion path with synthetic CASTEP outputs. No scene runs CASTEP.

The script needs no display. Qt draws on its ``offscreen`` platform and VTK
renders through EGL, so it runs on a plain CI runner. pyvistaqt's
``QtInteractor`` embeds a native render window, which the offscreen platform
cannot host; the 3D viewers are given :class:`OffscreenInteractor` instead,
which renders the same PyVista scene off screen and paints the frame.

Run it through the Makefile, or name the screenshots to render::

    make screenshots
    python scripts/docs_screenshots.py --list
    python scripts/docs_screenshots.py project_home journal_page
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING, Any
from uuid import UUID
import weakref

# Qt, VTK and the settings directories are configured before anything imports
# them. Settings and app data go to a throwaway directory so a local run never
# reads or writes the developer's own MuFinder configuration.
_SCRATCH = Path(tempfile.mkdtemp(prefix="mufinder-screenshots-"))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"
os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkEGLRenderWindow"
os.environ.setdefault("QT_SCALE_FACTOR", "2")
for _variable in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[_variable] = str(_SCRATCH / _variable.lower())

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDockWidget, QWidget  # noqa: E402

from mufinder.domain.enums import WorkflowStage  # noqa: E402
from mufinder.domain.models import Structure  # noqa: E402
from mufinder.science.magnetic_structure import (  # noqa: E402
    AtomSelector,
    MagneticStructureModel,
    build_collinear_component,
    build_helix_component,
)
from mufinder.services.project_service import ProjectDocument, ProjectService  # noqa: E402

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "_generated" / "screenshots"

# Logical window size; QT_SCALE_FACTOR doubles the saved pixels.
WINDOW_SIZE = (1440, 1120)
# How often, and how long, a scene waits for a modal dialog to open.
MODAL_POLL_MS = 50
MODAL_TIMEOUT_MS = 20_000


class OffscreenInteractor(QWidget):
    """A stand-in for ``pyvistaqt.QtInteractor`` that needs no native window.

    It owns an off-screen ``pyvista.Plotter`` and hands every plotter attribute
    the viewers use (``add_mesh``, ``camera``, ``iren``, ...) through to it.
    ``render`` draws the scene at the widget's current size and paints it.
    """

    _instances: weakref.WeakSet[OffscreenInteractor] = weakref.WeakSet()

    def __init__(self, parent: QWidget | None = None, **_kwargs: Any) -> None:
        super().__init__(parent)
        import pyvista as pv

        self._plotter = pv.Plotter(off_screen=True)
        self._frame: QImage | None = None
        OffscreenInteractor._instances.add(self)

    def __getattr__(self, name: str) -> Any:
        plotter = self.__dict__.get("_plotter")
        if plotter is None:
            raise AttributeError(name)
        return getattr(plotter, name)

    def render(self, *_args: Any) -> None:  # noqa: D401 - mirrors Plotter.render.
        """Draw the scene at the widget's size and repaint."""

        if self._plotter is None or not self.isVisible():
            return
        self._sync_window_size()
        pixels = np.ascontiguousarray(self._plotter.screenshot(return_img=True))
        height, width = pixels.shape[:2]
        image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGB888)
        self._frame = image.copy()
        self._frame.setDevicePixelRatio(self.devicePixelRatioF())
        self.update()

    def _sync_window_size(self) -> None:
        ratio = self.devicePixelRatioF()
        self._plotter.window_size = (max(int(self.width() * ratio), 1), max(int(self.height() * ratio), 1))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        # A camera fitted to the scene uses the render window's aspect ratio,
        # so the plotter follows the widget's size as a native window would.
        super().resizeEvent(event)
        if self._plotter is not None:
            self._sync_window_size()

    def close(self) -> bool:  # noqa: D401 - mirrors QWidget.close and Plotter.close.
        """Close the widget and release the off-screen render window."""

        plotter, self._plotter = self._plotter, None
        if plotter is not None:
            plotter.close()
        return super().close()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API naming.
        if self._frame is None:
            return
        painter = QPainter(self)
        painter.drawImage(self.rect(), self._frame)
        painter.end()

    @classmethod
    def render_all(cls) -> None:
        """Re-render every visible viewer at its laid-out size."""

        for interactor in list(cls._instances):
            interactor.render()


def install_offscreen_viewers() -> None:
    """Give MuFinder's 3D viewers :class:`OffscreenInteractor` in place of ``QtInteractor``."""

    import pyvista as pv

    from mufinder.app import atomic_structure_viewer

    atomic_structure_viewer._pyvista_modules = lambda: (pv, OffscreenInteractor)
    from mufinder.app import magnetic_field_widget

    magnetic_field_widget._pyvista_modules = atomic_structure_viewer._pyvista_modules


def settle(app: QApplication) -> None:
    """Let layouts, deferred renders and repaints finish before a grab."""

    for _ in range(3):
        app.processEvents()
        OffscreenInteractor.render_all()
    app.processEvents()


def save(widget: QWidget, path: Path) -> None:
    settle(QApplication.instance())
    if not widget.grab().save(str(path)):
        raise RuntimeError(f"Could not write {path}")
    print(f"wrote {path}")


# --- Projects -----------------------------------------------------------------
#
# A project builder writes one saved project under a directory and returns its
# root. Builders go through ``ProjectService`` and the workflow widgets'
# own form logic, the calls the application makes, so the saved project is one
# the application could have written.

ProjectBuilder = Callable[[Path], Path]


def _new_project(directory: Path, name: str, host: Structure) -> tuple[ProjectService, ProjectDocument]:
    service = ProjectService()
    root = directory / f"{name.lower().replace(' ', '-')}.mufinder"
    document = service.create_project(root, name)
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    return service, document


# The fixture project: rutile TiO2 (P4_2/mnm, a = 4.594 Å, c = 2.959 Å,
# O at 4f with x = 0.3048), a host small enough to read in every view.

RUTILE_A = 4.594
RUTILE_C = 2.959
RUTILE_X = 0.3048
XC_FUNCTIONAL = "PBE"
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
    service, document = _new_project(directory, "Rutile TiO2", host)
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


# The Magnetic Fields models are the worked examples in
# docs/user_guide/magnetic_fields.rst: one 1 μB ion in a simple cubic cell with
# a = 3 Å.


def simple_cubic_host() -> Structure:
    return Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def ferromagnet() -> MagneticStructureModel:
    return MagneticStructureModel(
        components=(
            build_collinear_component(
                label="Fe moment",
                selector=AtomSelector.all(),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, 1.0),
            ),
        ),
    )


def helix(q: float) -> MagneticStructureModel:
    return MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, q), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            build_helix_component(
                label="Fe helix",
                selector=AtomSelector.all(),
                q_coeff=(1.0, 0.0, 0.0),
                real_axis_vector=(1.0, 0.0, 0.0),
                imaginary_axis_vector=(0.0, 1.0, 0.0),
            ),
        ),
    )


def custom_site(fractional: tuple[float, float, float]) -> dict[str, object]:
    return {
        "source": "custom",
        "label": "Custom site",
        "fractional_position": list(fractional),
        "cartesian_position": [3.0 * value for value in fractional],
    }


def magnetic_project(name: str, model: MagneticStructureModel, form: dict[str, object]) -> ProjectBuilder:
    """A builder for the simple cubic host with ``model`` and a Magnetic Fields ``form``."""

    def build(directory: Path) -> Path:
        service, document = _new_project(directory, name, simple_cubic_host())
        document.project.magnetic_structure = model.to_json()
        document.ui_state["magnetic_field_form"] = form
        service.save_project(document)
        return document.project_root

    return build


# --- The results project --------------------------------------------------------
#
# The results pages need finished calculations. No scene runs CASTEP: the
# results project submits each workflow's jobs through the application's own
# controllers and ``JobExecutionController``, with a local runner whose
# ``submit`` writes the job's CASTEP outputs instead of starting a process, and
# then finishes each job through ``MainWindow``'s completion handler. Result
# dispatch, the stored results, derived structures and relaxed sites are
# therefore exactly what the application builds after a real run.
#
# The outputs come from the repository's synthetic fixtures where one fits: the
# UEP source job is tests/fixtures/uep/Si2.castep with its .den_fmt, and the
# pristine relaxation is tests/fixtures/geometry/Si2_bfgs.castep. No fixture
# holds a muon relaxation, a phonon run of a muoniated cell or a silicon NEB
# path, so those outputs are written here in CASTEP's file shapes with
# illustrative numbers (see SI_MUON_SITES, SI_MUON_MODES_CM and NEB_BARRIER_EV);
# the captions of the pages that show them say so.

FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
SI_PRIMITIVE = 2.6954645  # Å, half the cubic edge of tests/fixtures/uep/Si2.cell
SI_SUPERCELL = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

# Each candidate start relaxes to the nearest bond-centre (BC) or tetrahedral
# (T) interstitial, given in host fractional coordinates; BC is the lower site
# for a positive muon in silicon. The energies, and how far the muon pushes
# its nearest silicon neighbours outwards, are illustrative.
SI_MUON_SITES = (  # start, site kind, relaxed site
    ((0.14, 0.11, 0.13), "BC", (0.125, 0.125, 0.125)),
    ((0.61, 0.14, 0.12), "BC", (0.625, 0.125, 0.125)),
    ((0.13, 0.62, 0.14), "BC", (0.125, 0.625, 0.125)),
    ((0.48, 0.52, 0.50), "T", (0.5, 0.5, 0.5)),
    ((0.74, 0.77, 0.75), "T", (0.75, 0.75, 0.75)),
)
SI_SITE_ENERGY_EV = {"BC": -2741.2364, "T": -2740.9271}
SI_SITE_PUSH_ANG = {"BC": 0.42, "T": 0.03}
SI_NEIGHBOUR_RADIUS_ANG = 2.6
# The phonon of the lowest site: three muon modes at these Γ frequencies, each
# mixed with one silicon neighbour's motion so that the muon carries
# SI_MUON_PARTICIPATION of each (w1 = 3 * participation - 3 = -0.14), below a
# host band up to SI_HOST_TOP_CM. The numbers are illustrative.
SI_MUON_MODES_CM = (2742.0, 3226.0, 3226.0)
SI_MUON_PARTICIPATION = 1.0 - 0.14 / 3.0
SI_HOST_TOP_CM = 520.0
NEB_BARRIER_EV = 0.38  # illustrative height above the reactant
NEB_IMAGES = 5
UEP_ROW = "source_01"  # the UEP page's first row label
BOHR_ANG = 0.529177210903
HARTREE_EV = 27.211386245988


def silicon_host() -> Structure:
    """Diamond silicon's two-atom primitive cell, the cell of tests/fixtures/uep/Si2.cell."""

    a = SI_PRIMITIVE
    return Structure(
        label="Si (diamond)",
        lattice_vectors=[[a, a, 0.0], [a, 0.0, a], [0.0, a, a]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _cell_text(lattice: Any, species: Any, fractional: Any) -> str:
    """A minimal ``-out.cell``: the lattice and fractional positions CASTEP writes."""

    rows = "\n".join(f"   {' '.join(f'{value:17.10f}' for value in vector)}" for vector in lattice)
    atoms = "\n".join(
        f"   {symbol:<5} {' '.join(f'{float(value):17.10f}' for value in position)}"
        for symbol, position in zip(species, fractional, strict=True)
    )
    return (
        f"%BLOCK LATTICE_CART\n   ANG\n{rows}\n%ENDBLOCK LATTICE_CART\n\n"
        f"%BLOCK POSITIONS_FRAC\n{atoms}\n%ENDBLOCK POSITIONS_FRAC\n"
    )


def _final_configuration(castep: Path) -> tuple[list[list[float]], list[str], list[list[float]]]:
    """The lattice and atoms of a geometry optimisation's ``Final Configuration`` block."""

    import re

    lines = castep.read_text(encoding="utf-8").splitlines()
    start = max(index for index, line in enumerate(lines) if "Final Configuration" in line)
    lattice_row = next(index for index in range(start, len(lines)) if "Real Lattice" in lines[index]) + 1
    lattice = [[float(value) for value in lines[lattice_row + row].split()[:3]] for row in range(3)]
    atom = re.compile(r"^\s*x\s+([A-Za-z][\w:]*)\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\s+x\s*$")
    species: list[str] = []
    fractional: list[list[float]] = []
    for line in lines[lattice_row:]:
        match = atom.match(line)
        if match:
            species.append(match.group(1))
            fractional.append([float(match.group(index)) for index in (2, 3, 4)])
        elif species and "xxxx" in line:
            break
    return lattice, species, fractional


def _castep_text(*, energy: float, species: list[str], geometry: bool, wall_time: float) -> str:
    """A CASTEP output with the lines MuFinder reads: final energy, forces and timing."""

    forces = "\n".join(
        f" * {symbol:<6} {index:4d} {0.004 * np.sin(1.7 * index):12.5f} {0.003 * np.cos(2.3 * index):12.5f}"
        f" {-0.002 * np.sin(0.9 * index):12.5f} *"
        for index, symbol in enumerate(species, start=1)
    )
    converged = " BFGS: Geometry optimization completed successfully.\n\n" if geometry else ""
    return (
        " Run started: Mon, 12 Oct 2026 09:14:02 +0100\n\n"
        f" Final energy, E             =  {energy:.10f}     eV\n\n"
        " ************************** Forces **************************\n"
        " *               Cartesian components (eV/A)                *\n"
        f"{forces}\n"
        " ************************************************************\n\n"
        f"{converged}"
        f" Total time          =    {wall_time:.2f} s\n"
    )


def _job_cell(job_dir: Path, seed: str):
    from mufinder.castep_io.cell_parser import parse_cell_file

    return parse_cell_file(job_dir / f"{seed}.cell")


def _relaxed_muon_cell(cell: Any, kind: str, start: Any, site: Any) -> str:
    """The input supercell with its muon moved to ``site`` and its neighbours pushed out."""

    lattice = np.asarray(cell.lattice_vectors, dtype=float)
    fractional = np.asarray(cell.fractional_positions, dtype=float)
    muon = next(index for index, symbol in enumerate(cell.species) if symbol == "H:mu")
    repeats = np.rint(np.linalg.norm(lattice, axis=1) / np.linalg.norm(silicon_host().lattice_vectors, axis=1))
    shift = (np.asarray(site) - np.asarray(start) + 0.5) % 1.0 - 0.5
    fractional[muon] = fractional[muon] + shift / repeats
    for index in range(len(fractional)):
        if index == muon:
            continue
        separation = (fractional[index] - fractional[muon] + 0.5) % 1.0 - 0.5
        vector = separation @ lattice
        distance = float(np.linalg.norm(vector))
        if distance < SI_NEIGHBOUR_RADIUS_ANG:
            moved = vector * (1.0 + SI_SITE_PUSH_ANG[kind] / distance)
            fractional[index] = fractional[muon] + np.linalg.solve(lattice.T, moved)
    return _cell_text(lattice.tolist(), cell.species, (fractional % 1.0).tolist())


@contextmanager
def refuse_message_boxes(app: QApplication) -> Iterator[None]:
    """Fail the build, rather than hang it, if the application shows a message box."""

    from PySide6.QtWidgets import QMessageBox

    shown: list[str] = []

    def check() -> None:
        dialog = app.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            shown.append(f"{dialog.windowTitle()}: {dialog.text()}")
            dialog.reject()

    timer = QTimer()
    timer.timeout.connect(check)
    timer.start(MODAL_POLL_MS)
    try:
        yield
    finally:
        timer.stop()
    if shown:
        raise RuntimeError("The results project showed a message box: " + "; ".join(shown))


Outputs = Callable[[Any, Path, str], bool]


@contextmanager
def synthetic_runs(window: MainWindow, outputs: Outputs) -> Iterator[None]:
    """Run the jobs a workflow submits locally without CASTEP.

    Submission goes through ``JobExecutionController.submit_local_jobs`` with a
    runner whose ``submit`` calls ``outputs(job, job_dir, seed)`` to write the
    job's CASTEP outputs; a job for which it returns ``True`` is then started
    and finished through ``MainWindow``'s local-runner handlers, which parse
    the ``.castep`` and dispatch the result to the owning workflow. A job for
    which it returns ``False`` is left running.
    """

    from mufinder.castep_io.output_parser import parse_castep_output

    finished: list[tuple[Any, Path]] = []
    errors: list[BaseException] = []
    document = window._document

    def submit(*, job_id: Any, job_dir: Path, seed_name: str, **_options: object) -> None:
        try:
            done = outputs(document.jobs[job_id], Path(job_dir), seed_name)
        except Exception as exc:
            # The job execution service records a runner error as a failed
            # submission; a docs build must stop instead.
            errors.append(exc)
            raise
        if done:
            finished.append((job_id, Path(job_dir) / f"{seed_name}.castep"))

    def submit_jobs_locally(jobs: list[Any], **_options: object) -> None:
        window._job_execution.submit_local_jobs(
            jobs,
            num_cores=8,
            max_concurrent=1,
            castep_executable="castep.mpi",
            mpirun_path="mpirun",
            retrieval_for_job=window._retrieval_flags_for_job,
        )
        if errors:
            raise errors[0]
        while finished:
            job_id, castep = finished.pop(0)
            window._on_local_job_started(job_id)
            window._on_local_job_finished(job_id, parse_castep_output(castep))

    window._local_runner.submit = submit
    window._submit_jobs_locally = submit_jobs_locally
    try:
        yield
    finally:
        del window._local_runner.submit
        del window._submit_jobs_locally
        window._flush_dashboard()


def _pristine_outputs(*, running: bool) -> Outputs:
    """Si2_bfgs.castep and the relaxed cell of its final configuration; cut short while running."""

    def write(_job: Any, job_dir: Path, seed: str) -> bool:
        source = FIXTURES / "geometry" / "Si2_bfgs.castep"
        text = source.read_text(encoding="utf-8")
        if running:
            # The output as CASTEP has written it part-way through the optimisation.
            text = text[: text.index(" BFGS: finished iteration     5")]
        (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
        if not running:
            (job_dir / f"{seed}-out.cell").write_text(_cell_text(*_final_configuration(source)), encoding="utf-8")
        return not running

    return write


def _muon_outputs(window: MainWindow) -> Outputs:
    """A relaxed cell and an illustrative CASTEP output for each SI_MUON_SITES start."""

    def write(job: Any, job_dir: Path, seed: str) -> bool:
        definition, row = window._muon_relaxation_definition_and_row_for_job(job)
        index, (start, kind, site) = next(
            (index, entry)
            for index, entry in enumerate(SI_MUON_SITES)
            if np.allclose(entry[0], row.initial_fractional_position)
        )
        cell = _job_cell(job_dir, seed)
        (job_dir / f"{seed}-out.cell").write_text(_relaxed_muon_cell(cell, kind, start, site), encoding="utf-8")
        energy = SI_SITE_ENERGY_EV[kind] + 0.0007 * index
        text = _castep_text(energy=energy, species=list(cell.species), geometry=True, wall_time=1843.2 + 211.0 * index)
        (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
        return True

    return write


def _uep_outputs(_job: Any, job_dir: Path, seed: str) -> bool:
    """The synthetic UEP source fixture: Si2.castep, Si2.den_fmt and Si2.cell."""

    for suffix in (".castep", ".den_fmt", ".cell"):
        shutil.copyfile(FIXTURES / "uep" / f"Si2{suffix}", job_dir / f"{seed}{suffix}")
    return True


def _phonon_text(cell: Any) -> str:
    """A Gamma-only CASTEP ``.phonon`` file for ``cell`` with the SI_MUON_MODES_CM muon modes.

    The eigenvectors are orthonormal: each muon direction is rotated with the
    same direction of the first silicon atom, which gives a muon mode and a
    host partner; every other degree of freedom is a host mode of its own.
    """

    from mufinder.domain.constants import MUON_MASS_AMU

    species = list(cell.species)
    count = len(species)
    muon = species.index("H:mu")
    neighbour = next(index for index, symbol in enumerate(species) if symbol != "H:mu")
    mix = np.sqrt(SI_MUON_PARTICIPATION)
    other = np.sqrt(1.0 - SI_MUON_PARTICIPATION)
    modes: list[tuple[float, np.ndarray]] = []
    paired: set[tuple[int, int]] = set()
    for axis, frequency in enumerate(SI_MUON_MODES_CM):
        muon_mode = np.zeros((count, 3))
        muon_mode[muon, axis], muon_mode[neighbour, axis] = mix, other
        partner = np.zeros((count, 3))
        partner[muon, axis], partner[neighbour, axis] = -other, mix
        modes += [(frequency, muon_mode), (SI_HOST_TOP_CM * (0.9 + 0.02 * axis), partner)]
        paired |= {(muon, axis), (neighbour, axis)}
    host = [(atom, axis) for atom in range(count) for axis in range(3) if (atom, axis) not in paired]
    for number, (atom, axis) in enumerate(host):
        vector = np.zeros((count, 3))
        vector[atom, axis] = 1.0
        modes.append((SI_HOST_TOP_CM * np.sin(0.5 * np.pi * (number + 1) / (len(host) + 1)), vector))
    modes.sort(key=lambda mode: mode[0])

    lines = [
        " BEGIN header",
        f" Number of ions {count:9d}",
        f" Number of branches {3 * count:5d}",
        " Number of wavevectors  1",
        " Frequencies in         cm-1",
        " Unit cell vectors (A)",
        *(f" {row[0]:12.6f}{row[1]:12.6f}{row[2]:12.6f}" for row in np.asarray(cell.lattice_vectors, dtype=float)),
        " Fractional Co-ordinates",
    ]
    for index, (symbol, position) in enumerate(zip(species, cell.fractional_positions, strict=True), start=1):
        mass = MUON_MASS_AMU if symbol == "H:mu" else 28.0855
        lines.append(f" {index:5d} {position[0]:12.6f}{position[1]:12.6f}{position[2]:12.6f}   {symbol:<8}{mass:12.6f}")
    lines += [" END header", "     q-pt=    1    0.000000  0.000000  0.000000      1.0000000000"]
    lines += [f" {number:7d} {frequency:14.6f}" for number, (frequency, _vector) in enumerate(modes, start=1)]
    lines += ["                        Phonon Eigenvectors", "Mode Ion                X                                   Y                                   Z"]
    for number, (_frequency, vector) in enumerate(modes, start=1):
        for atom in range(count):
            lines.append(f" {number:3d} {atom + 1:3d}" + "".join(f" {value:15.12f} {0.0:15.12f}" for value in vector[atom]))
    return "\n".join(lines) + "\n"


def _phonon_outputs(job: Any, job_dir: Path, seed: str) -> bool:
    """The stage-2 phonon of a quantum-corrections row: its ``.phonon`` and ``.castep``."""

    cell = _job_cell(job_dir, seed)
    (job_dir / f"{seed}.phonon").write_text(_phonon_text(cell), encoding="utf-8")
    text = _castep_text(energy=SI_SITE_ENERGY_EV["BC"] - 0.0004, species=list(cell.species), geometry=False, wall_time=9126.4)
    (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
    return True


def _ts_text(lattice: np.ndarray, species: list[str], images: list[tuple[str, float, float, np.ndarray]]) -> str:
    """A CASTEP ``.ts`` file in atomic units: one REA, the TST images and one PRO block."""

    def number(value: float) -> str:
        mantissa, exponent = f"{value:.8E}".split("E")
        return f"{mantissa}E{int(exponent):+04d}".rjust(18)

    def row(values: Any) -> str:
        return "".join(f"   {number(float(value))}" for value in values)

    lines = [" TSTYPE TSConfirmation"]
    for tag, coordinate, energy_ev, cartesian in images:
        lines.append(f" {tag:<3} {1:4d}   {number(coordinate)}")
        energy = energy_ev / HARTREE_EV
        lines.append(" " * 9 + row((energy, energy)) + " " * 21 + "  <-- E")
        lines.extend(" " * 9 + row(vector / BOHR_ANG) + "  <-- h" for vector in lattice)
        lines.extend(
            f" {symbol:<8} {1:4d}" + row(position / BOHR_ANG) + "  <-- R"
            for symbol, position in zip(species, cartesian, strict=True)
        )
        lines.extend(f" {symbol:<8} {1:4d}" + row((0.0, 0.0, 0.0)) + "  <-- F" for symbol in species)
        lines.append("  ")
    return "\n".join(lines) + "\n"


def _neb_outputs(window: MainWindow) -> Outputs:
    """An illustrative NEB band through the job's reactant, intermediate and product cells.

    The images follow the input band, reactant to intermediate to product; the
    energy profile rises NEB_BARRIER_EV above the straight line between the
    two relaxed endpoint energies.
    """

    def write(job: Any, job_dir: Path, seed: str) -> bool:
        cell = _job_cell(job_dir, seed)
        lattice = np.asarray(cell.lattice_vectors, dtype=float)
        reactant = np.asarray(cell.fractional_positions, dtype=float)

        def block(name: str) -> np.ndarray:
            rows = next(item for item in cell.cell_file.blocks if item.name.lower() == name).rows
            positions = np.asarray([[float(value) for value in row[1:4]] for row in rows], dtype=float)
            return reactant + (positions - reactant + 0.5) % 1.0 - 0.5

        intermediate = block("positions_frac_intermediate")
        product = intermediate + (block("positions_frac_product") - intermediate + 0.5) % 1.0 - 0.5
        reactant_energy, product_energy = source_energies(window, job)
        images = []
        for image in range(NEB_IMAGES + 2):
            coordinate = image / (NEB_IMAGES + 1)
            tag = "REA" if image == 0 else "PRO" if image == NEB_IMAGES + 1 else "TST"
            energy = (
                reactant_energy
                + (product_energy - reactant_energy) * coordinate
                + NEB_BARRIER_EV * np.sin(np.pi * coordinate) ** 2
            )
            if coordinate <= 0.5:
                fractional = reactant + (intermediate - reactant) * 2.0 * coordinate
            else:
                fractional = intermediate + (product - intermediate) * (2.0 * coordinate - 1.0)
            images.append((tag, coordinate, energy, fractional @ lattice))
        (job_dir / f"{seed}.ts").write_text(_ts_text(lattice, list(cell.species), images), encoding="utf-8")
        text = _castep_text(energy=images[-1][2], species=list(cell.species), geometry=False, wall_time=15214.8)
        (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
        return True

    return write


def source_energies(window: MainWindow, job: Any) -> tuple[float, float]:
    """The relaxed energies of the NEB job's reactant and product sites."""

    controller = window._energy_barriers_controller
    _definition, key = controller.definition_and_key_for_job(job)
    reactant, product = str(key).split(":")[:2]
    sites = window._document.sites
    return sites[UUID(reactant)].total_energy_eV, sites[UUID(product)].total_energy_eV


def _silicon_document(directory: Path) -> tuple[ProjectService, ProjectDocument]:
    """The silicon host with ``PBE`` and a saved 2x2x2 supercell for the muon relaxations."""

    host = silicon_host()
    service, document = _new_project(directory, "Silicon", host)
    service.set_recommended_xc_functional(document, XC_FUNCTIONAL)
    service.create_workflow_supercell_asset(
        document,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        host_structure_id=host.id,
        matrix=SI_SUPERCELL,
        label="Si 2x2x2",
    )
    document.project.supercell_matrix = [list(row) for row in SI_SUPERCELL]
    return service, document


def silicon_project(directory: Path) -> Path:
    """The silicon host alone, for the legacy importer."""

    service, document = _silicon_document(directory)
    service.save_project(document)
    return document.project_root


def _build_results(directory: Path, *, running_pristine: bool = False) -> Path:
    """Build and save the results project; see the section comment above."""

    app = QApplication.instance()
    service, document = _silicon_document(directory)
    for start, _kind, _site in SI_MUON_SITES:
        service.generate_candidate_sites(
            document,
            method="manual",
            parameters={"coord_system": "fractional", "x": start[0], "y": start[1], "z": start[2]},
        )
    service.save_project(document)

    with opened(app, document.project_root) as window, refuse_message_boxes(app):
        document = window._document
        # Pristine relaxation: one variable-cell row, run as Run All Locally runs it.
        pristine = window._pristine_relaxation_view
        window._set_active_route("pristine_relaxation")
        pristine._add_row()
        with synthetic_runs(window, _pristine_outputs(running=running_pristine)):
            pristine._run_local_selected_rows()
        if running_pristine:
            window._save_project()
            return document.project_root
        # A comparison table of the relaxed cell against the host.
        when_modal("ComparisonJobDialog", include_every_job)
        pristine.generate_comparison_requested.emit()

        # UEP: one source row on the host, then a batch search and its clustering.
        window._set_active_route("uep")
        window._uep_view._add_row()
        with synthetic_runs(window, _uep_outputs):
            window._uep_view._run_local_selected_rows()
        run_uep_analysis(window)

        # Muon relaxations of every candidate site in the 2x2x2 supercell.
        site_ids = [str(site.id) for site in document.sites.values()]
        when_modal("MuonChargeStateDialog", lambda dialog: dialog.accept())
        window._on_create_muon_relaxation_jobs_requested(site_ids)
        relaxations = window._muon_relaxation_view
        for entry in list(relaxations._row_entries()):
            relaxations.set_row_supercell(entry.row_id, [list(row) for row in SI_SUPERCELL])
        with synthetic_runs(window, _muon_outputs(window)):
            relaxations._run_local_selected_rows()

        # An analysis workspace of every relaxation, clustered and pinned.
        window._set_active_route("muon_relaxations")
        relaxations._active_table().selectAll()
        when_modal("QInputDialog", lambda dialog: (dialog.setTextValue("Positive muon sites"), dialog.accept()))
        relaxations._create_analysis_button.click()
        workspace_id = str(window._relaxation_analysis_workspaces()[0]["id"])
        when_modal("RelaxationClusteringDialog", lambda dialog: dialog._apply_button.click())
        window._relaxation_analysis_view._open_clustering(workspace_id)
        window._on_muon_sites_pin_requested(workspace_id)

        # Quantum corrections: the phonon job of the lowest-energy site.
        window._set_active_route("quantum_corrections")
        quantum = window._quantum_corrections_controller
        candidates = sorted(quantum.candidates(), key=lambda candidate: candidate.site.total_energy_eV)
        with synthetic_runs(window, _phonon_outputs):
            quantum.submit([candidates[0].key], target="local")

        # Energy barriers: the BC-to-T paths, and an NEB band along the first.
        window._set_active_route("energy_barriers")
        barriers = window._energy_barriers_controller
        energies = {row.key: document.sites[UUID(row.key)].total_energy_eV for row in barriers.endpoint_rows()}
        reactant, product = min(energies, key=energies.get), max(energies, key=energies.get)
        barriers.select_endpoints(reactant, product)
        barriers.generate_paths(reactant, product)
        with synthetic_runs(window, _neb_outputs(window)):
            barriers.submit([barriers.candidates()[0].key], target="local")
        window._save_project()
        return document.project_root


def include_every_job(dialog: QDialog) -> None:
    """Tick every completed relaxation in ``Generate Comparison Table`` and click OK."""

    from PySide6.QtCore import Qt

    for row in range(dialog._table.rowCount()):
        dialog._table.item(row, dialog._CHECK_COLUMN).setCheckState(Qt.CheckState.Checked)
    dialog._accept_if_valid()


def run_uep_analysis(window: MainWindow) -> None:
    """Open the UEP analysis, run a batch search, its ZPE and clustering, as the buttons do.

    The batch search and ZPE run their worker threads' ``run`` in this thread,
    and the random starts are seeded so every build finds the same minima.
    """

    from functools import partial

    from mufinder.app import uep_wizard
    from mufinder.science.site_generation import generate_random_sites

    job = window._uep_controller.jobs_for_keys([UEP_ROW])[0]
    window._on_open_uep_analysis_requested(UEP_ROW)
    dialog = window._uep_analysis_dialogs[job.id]
    threads = (uep_wizard._BatchSearchThread, uep_wizard._BatchZpeThread)
    starts = [thread.start for thread in threads]
    for thread in threads:
        thread.start = lambda worker: (worker.run(), worker.finished.emit())
    uep_wizard.generate_random_sites = partial(generate_random_sites, seed=7)
    try:
        dialog._run_batch_search()
        dialog._run_batch_zpe()
        dialog._open_clustering_dialog()
    finally:
        for thread, start in zip(threads, starts, strict=True):
            thread.start = start
        uep_wizard.generate_random_sites = generate_random_sites
    settle(QApplication.instance())
    for child in (dialog._clustering_dialog, dialog):
        if child is not None:
            child.close()


def results_project(directory: Path) -> Path:
    """The silicon project with every results workflow finished; see the section comment."""

    return _build_results(directory)


def running_relaxation_project(directory: Path) -> Path:
    """The silicon project with its pristine relaxation part-way through."""

    return _build_results(directory, running_pristine=True)


# --- Scenes -------------------------------------------------------------------

WidgetStep = Callable[[QWidget], None]


@dataclass(frozen=True)
class Window:
    """Grab the whole main window, optionally with the Job Dashboard dock shown."""

    size: tuple[int, int] = WINDOW_SIZE
    dashboard: bool = False
    prepare: WidgetStep | None = None


@dataclass(frozen=True)
class Page:
    """Grab one page widget, laid out at ``size``.

    ``attribute`` names the ``MainWindow`` attribute holding the page, or is a
    callable returning it, for the pages kept per saved item (a comparison
    table, a saved Magnetic Fields workspace).
    """

    attribute: str | Callable[[MainWindow], QWidget]
    size: tuple[int, int] = (1100, 820)
    prepare: WidgetStep | None = None


@dataclass(frozen=True)
class Dock:
    """Grab the dock titled ``title``, laid out in a window of ``window_size``."""

    title: str
    window_size: tuple[int, int] = WINDOW_SIZE
    prepare: WidgetStep | None = None


@dataclass(frozen=True)
class Dialog:
    """Grab the modal dialog that ``open(window)`` opens.

    ``open`` may block in the dialog's event loop. The grab waits for the
    active modal widget to inherit the class named ``kind``, so a dialog
    opened from another dialog can be the target; every dialog still open
    afterwards is dismissed. Dialogs size themselves to fit the screen, and the offscreen
    screen is small, so ``size`` gives the dialog its desktop size first.
    """

    open: Callable[[MainWindow], object]
    kind: str = "QDialog"
    size: tuple[int, int] | None = None
    prepare: WidgetStep | None = None


@dataclass(frozen=True)
class Floating:
    """Grab the non-modal window that ``open(window)`` shows and returns.

    Some windows (the geometry progress, UEP analysis and clustering windows)
    stay open beside the main window instead of blocking it. ``size`` gives
    the window its desktop size before the grab; it is closed afterwards.
    """

    open: Callable[[MainWindow], QWidget]
    size: tuple[int, int] | None = None
    prepare: WidgetStep | None = None


Target = Window | Page | Dock | Dialog | Floating


@dataclass(frozen=True)
class Scene:
    """One screenshot: a project, the route to open, an optional step, and the grab.

    ``route`` is a navigator route (see ``MainWindow._workflow_definitions``);
    ``None`` leaves the Project Home showing. ``setup`` drives the window
    before the grab target is laid out.
    """

    project: ProjectBuilder
    route: str | None
    target: Target
    setup: Callable[[MainWindow], None] | None = None


@contextmanager
def opened(app: QApplication, project: Path, *, dashboard: bool = False) -> Iterator[MainWindow]:
    """A shown ``MainWindow`` with ``project`` open."""

    from mufinder.app.main_window import MainWindow

    window = MainWindow()
    window.resize(*WINDOW_SIZE)
    # Most scenes run no jobs, so the empty Job Dashboard would only take room
    # from the page.
    for dock in window.findChildren(QDockWidget):
        dock.setVisible(dashboard)
    # The status bar warns, in red, that no keyword catalog is installed and
    # no cluster is connected: true of the scratch settings, noise in the docs.
    window.statusBar().hide()
    window.show()
    if not window._open_project_from_path(project, show_errors=False):
        raise RuntimeError(f"Could not open the project {project}")
    settle(app)
    try:
        yield window
    finally:
        # A hidden window closes without asking to save the project.
        window.hide()
        window.close()
        window.deleteLater()
        app.processEvents()


def _fit_page(app: QApplication, window: MainWindow, page: QWidget, size: tuple[int, int]) -> None:
    """Resize ``window`` until ``page`` is laid out at ``size``."""

    for _ in range(3):
        settle(app)
        width = window.width() + size[0] - page.width()
        height = window.height() + size[1] - page.height()
        if (width, height) == (window.width(), window.height()):
            return
        window.resize(width, height)
    settle(app)


def _dismiss_modals(app: QApplication) -> None:
    for _ in range(5):
        dialog = app.activeModalWidget()
        if not isinstance(dialog, QDialog):
            return
        dialog.reject()
        app.processEvents()


def _active_modal(app: QApplication, kind: str) -> QDialog | None:
    dialog = app.activeModalWidget()
    return dialog if isinstance(dialog, QDialog) and dialog.inherits(kind) else None


def when_modal(kind: str, step: Callable[[QDialog], None], *, timeout_ms: int = MODAL_TIMEOUT_MS) -> None:
    """Run ``step`` on the active modal dialog once it inherits the class named ``kind``.

    After ``timeout_ms`` without one, every open dialog is dismissed so a
    blocking ``exec`` returns and the scene fails instead of hanging.
    """

    app = QApplication.instance()
    remaining = [timeout_ms // MODAL_POLL_MS]

    def poll() -> None:
        dialog = _active_modal(app, kind)
        if dialog is not None:
            step(dialog)
        elif remaining[0] > 0:
            remaining[0] -= 1
            QTimer.singleShot(MODAL_POLL_MS, poll)
        else:
            _dismiss_modals(app)

    QTimer.singleShot(0, poll)


def capture_modal(app: QApplication, window: MainWindow, target: Dialog, path: Path) -> None:
    """Open the dialog of ``target``, save it to ``path`` and dismiss it."""

    captured: list[str] = []

    def grab(dialog: QDialog) -> None:
        try:
            if target.size is not None:
                dialog.resize(*target.size)
            if target.prepare is not None:
                target.prepare(dialog)
            save(dialog, path)
            captured.append(str(path))
        finally:
            dialog.reject()
            # A dialog opened from another one leaves its parent open.
            QTimer.singleShot(0, lambda: _dismiss_modals(app))

    when_modal(target.kind, grab)
    target.open(window)
    _dismiss_modals(app)
    if not captured:
        raise RuntimeError(f"No dialog opened for {path.name}")


def render(app: QApplication, scene: Scene, path: Path) -> None:
    """Open ``scene``'s project, drive it, and save its grab target to ``path``."""

    directory = Path(tempfile.mkdtemp(prefix=f"{path.stem}-", dir=_SCRATCH))
    target = scene.target
    dashboard = isinstance(target, Dock) or (isinstance(target, Window) and target.dashboard)
    with opened(app, scene.project(directory), dashboard=dashboard) as window:
        if scene.route is None:
            window._show_project_home()
        else:
            window._set_active_route(scene.route)
        settle(app)
        if scene.setup is not None:
            scene.setup(window)
            settle(app)
        if isinstance(target, Dialog):
            capture_modal(app, window, target, path)
            return
        if isinstance(target, Floating):
            floating = target.open(window)
            try:
                if target.size is not None:
                    floating.resize(*target.size)
                settle(app)
                if target.prepare is not None:
                    target.prepare(floating)
                save(floating, path)
            finally:
                floating.close()
            return
        if isinstance(target, Window):
            window.resize(*target.size)
            widget: QWidget = window
        elif isinstance(target, Dock):
            window.resize(*target.window_size)
            docks = [dock for dock in window.findChildren(QDockWidget) if dock.windowTitle() == target.title]
            if not docks:
                raise RuntimeError(f"No dock titled {target.title!r}")
            widget = docks[0]
        else:
            attribute = target.attribute
            widget = getattr(window, attribute) if isinstance(attribute, str) else attribute(window)
            _fit_page(app, window, widget, target.size)
        settle(app)
        if target.prepare is not None:
            target.prepare(widget)
        save(widget, path)


# --- Scene steps --------------------------------------------------------------


def calculate(page: Any) -> None:
    """Click ``Calculate Dipolar Field`` and select the first result row."""

    page._calculate_button.click()
    if page._result_table.rowCount() == 0:
        # A screenshot of a failed calculation must not reach the docs.
        raise RuntimeError(f"The field calculation gave no result: {page._calculation_status.text()}")
    page._result_table.selectRow(0)


def show_magnetic_cell(dialog: QWidget) -> None:
    """One magnetic cell of the q = 1/4 helix is four host cells along c."""

    dialog._repeat_c.setValue(4)


def open_distribution_plot(window: MainWindow) -> None:
    page = window._magnetic_field_view
    calculate(page)
    page._open_distribution_button.click()


def frame_viewers(widget: QWidget) -> None:
    """Reset every 3D viewer in ``widget`` so it frames its structure at its laid-out size."""

    from mufinder.app.atomic_structure_viewer import AtomicStructureViewer

    for viewer in widget.findChildren(AtomicStructureViewer):
        if viewer.isVisible():
            viewer._reset_view()


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


def focus_default_button(dialog: QWidget) -> None:
    """Move focus off the first field, so no text cursor shows in the grab."""

    from PySide6.QtWidgets import QPushButton

    for button in dialog.findChildren(QPushButton):
        if button.isDefault() or button.text() == "OK":
            button.setFocus()
            return


def hide_catalog_notice(dialog: QWidget) -> None:
    """Hide the notice that no keyword catalog is installed, true only of the scratch settings."""

    notice = dialog.findChild(QWidget, "catalogFallbackNotice")
    if notice is not None:
        notice.hide()


def select_all_sites(page: QWidget) -> None:
    """Widen the preview and select every generated site, which marks them in it."""

    from PySide6.QtWidgets import QSplitter

    page.findChild(QSplitter).setSizes([520, 640])
    page._site_table.selectAll()
    settle(QApplication.instance())
    frame_viewers(page)


def current_page(window: MainWindow) -> QWidget:
    """The page the navigator shows, for pages kept per saved item."""

    return window._main_stack.currentWidget()


def open_analysis_workspace(window: MainWindow) -> None:
    """Open the results project's analysis workspace in the navigator."""

    workspace_id = str(window._relaxation_analysis_workspaces()[0]["id"])
    window._set_active_route(window._relaxation_analysis_workspace_route(workspace_id))


def open_comparison_table(window: MainWindow) -> None:
    """Open the results project's comparison table in the navigator."""

    table_id = next(iter(window._document.comparison_tables))
    window._set_active_route(window._comparison_route(table_id))


def show_running_progress(window: MainWindow) -> QWidget:
    """Inspect the running pristine relaxation, as its dashboard Inspect button does.

    Reopening a project ends its local jobs, since no local runner survives
    the application, so the scene puts the job back in the state the local
    runner keeps it in while CASTEP writes its ``.castep``.
    """

    from mufinder.domain.enums import JobStatus

    job = next(job for job in window._document.jobs.values() if job.workflow_stage == WorkflowStage.PRISTINE_RELAXATION)
    job.status = JobStatus.RUNNING
    castep = window._job_output_directory(job) / f"{window._job_seed_name(job)}.castep"
    monitor = window._job_progress
    monitor._local_output_paths = lambda job_id: (castep,) if job_id == job.id else ()
    monitor._run_in_background = lambda task: task()
    window._refresh_jobs_table()
    return window._show_geometry_progress(job)


def open_uep_clustering(window: MainWindow) -> QWidget:
    """Open the UEP analysis of the source job, then its saved batch search's clustering."""

    window._on_open_uep_analysis_requested(UEP_ROW)
    analysis = next(iter(window._uep_analysis_dialogs.values()))
    analysis._open_clustering_dialog()
    analysis.hide()
    return analysis._clustering_dialog


def show_neb_analysis(page: QWidget) -> None:
    """Show the NEB Analysis tab with the transition-state image in the inspector."""

    page._tabs.setCurrentIndex(1)
    settle(QApplication.instance())
    page._apply_preview_index((NEB_IMAGES + 1) // 2)
    settle(QApplication.instance())
    frame_viewers(page)


def save_magnetic_workspace(window: MainWindow) -> None:
    """Calculate the field, then click ``Save Workspace`` and name it."""

    page = window._magnetic_field_view
    calculate(page)
    when_modal("QInputDialog", lambda dialog: (dialog.setTextValue("Ferromagnet"), dialog.accept()))
    page._pin_button.click()


def _legacy_castep_text(*, energy: float, converged: bool) -> str:
    """A legacy run's CASTEP output: the header lines the importer reads and the result."""

    tail = " Geometry optimization completed successfully.\n\n" if converged else ""
    return (
        " Run started: Tue, 14 Mar 2023 10:02:41 +0000\n\n"
        " type of calculation                            : geometry optimization\n"
        " using functional                               : Perdew Burke Ernzerhof\n"
        " plane wave basis set cut-off                   :   400.0000   eV\n"
        " net charge of system                           :  1.000\n"
        " treating system as non-spin-polarized\n\n"
        "                       MP grid size for SCF calculation is  2  2  2\n\n"
        f" Final energy, E             =  {energy:.4f}     eV\n\n"
        f"{tail} Total time          =    2214.37 s\n"
    )


def _legacy_folder(root: Path) -> Path:
    """A legacy working folder of three silicon muon relaxations, in the documented layout.

    Hand-authored, with illustrative energies: see docs/legacy_import.rst,
    "Research data boundary". Site 3 stopped before its geometry converged.
    """

    from mufinder.science.supercell import build_supercell_structure

    supercell, _spins = build_supercell_structure(silicon_host(), SI_SUPERCELL)
    for index, ((start, kind, site), converged) in enumerate(zip(SI_MUON_SITES[2:], (True, True, False)), start=1):
        species = [*supercell.species, "H:mu"]
        initial = [*supercell.fractional_positions, [value / 2.0 for value in start]]
        relaxed = [*supercell.fractional_positions, [value / 2.0 for value in site]]
        stem = f"Si_{index}"
        for folder in ("input", "results", f"run/site{index}"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        (root / "input" / f"{stem}.cell").write_text(
            _cell_text(supercell.lattice_vectors, species, initial) + "\nFIX_ALL_CELL : TRUE\n", encoding="utf-8"
        )
        (root / "results" / f"{stem}-out.cell").write_text(
            _cell_text(supercell.lattice_vectors, species, relaxed), encoding="utf-8"
        )
        energy = SI_SITE_ENERGY_EV[kind] + 0.0011 * index
        (root / "results" / f"{stem}.castep").write_text(
            _legacy_castep_text(energy=energy, converged=converged), encoding="utf-8"
        )
        (root / f"run/site{index}" / f"{stem}.param").write_text(
            "task : geometryoptimisation\ncharge : +1\nxc_functional : PBE\ncut_off_energy : 400 eV\n"
            "write_cell_structure : true\ngeom_max_iter : 300\n",
            encoding="utf-8",
        )
    a = 2.0 * SI_PRIMITIVE
    (root / "Si.cif").write_text(
        f"data_Si\n_symmetry_space_group_name_H-M 'F d -3 m'\n_cell_length_a {a:.6f}\n_cell_length_b {a:.6f}\n"
        f"_cell_length_c {a:.6f}\n_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nSi 0 0 0\n",
        encoding="utf-8",
    )
    return root


def open_legacy_import(window: MainWindow) -> None:
    """Scan a synthetic legacy folder and open ``Import Legacy Relaxations`` on it.

    ``File > Import Legacy Relaxations...`` scans on a worker thread after a
    folder dialog; the scene runs the same scan in this thread. Its notice that
    no keyword catalog is installed, true only of the scratch settings, is
    dropped as the other scenes hide it.
    """

    from dataclasses import replace

    from mufinder.app.legacy_import_dialog import LegacyImportDialog
    from mufinder.services.legacy_import import scan_legacy_folder

    host = window._project_service.active_host_structure(window._document)
    root = _legacy_folder(Path(tempfile.mkdtemp(prefix="legacy-", dir=_SCRATCH)) / "si-muon-sites")
    scan = scan_legacy_folder(root, host_lattice=[list(row) for row in host.lattice_vectors], host_species=list(host.species))
    scan = replace(scan, warnings=tuple(warning for warning in scan.warnings if "keyword catalog" not in warning))
    LegacyImportDialog(scan, parent=window).exec()


# --- Screenshots --------------------------------------------------------------

SCREENSHOTS: dict[str, Scene] = {
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
    "magnetic_fields_page": Scene(
        magnetic_project(
            "Simple cubic ferromagnet",
            ferromagnet(),
            {"demag_shape": "sphere", "selected_muon_site": custom_site((0.5, 0.5, 0.0))},
        ),
        "magnetic_fields",
        Window(),
        setup=lambda window: calculate(window._magnetic_field_view),
    ),
    "magnetic_structure_builder": Scene(
        magnetic_project("Commensurate helix", helix(0.25), {"selected_muon_site": custom_site((0.5, 0.0, 0.0))}),
        "magnetic_fields",
        Dialog(
            lambda window: window._magnetic_field_view._edit_structure_button.click(),
            size=(1180, 900),
            prepare=show_magnetic_cell,
        ),
    ),
    "magnetic_field_distribution": Scene(
        magnetic_project(
            "Incommensurate helix",
            helix(0.2861),
            {"demag_shape": "none", "selected_muon_site": custom_site((0.5, 0.0, 0.0))},
        ),
        "magnetic_fields",
        Dialog(open_distribution_plot),
    ),
    "magnetic_field_workspace": Scene(
        magnetic_project(
            "Simple cubic ferromagnet",
            ferromagnet(),
            {"demag_shape": "sphere", "selected_muon_site": custom_site((0.5, 0.5, 0.0))},
        ),
        "magnetic_fields",
        Page(current_page, (1100, 860)),
        setup=save_magnetic_workspace,
    ),
    "pristine_relaxation_page": Scene(results_project, "pristine_relaxation", Page("_pristine_relaxation_view", (1100, 540))),
    "geometry_progress_dialog": Scene(
        running_relaxation_project, "pristine_relaxation", Floating(show_running_progress, size=(820, 640))
    ),
    "comparison_table": Scene(results_project, None, Page(current_page, (780, 820)), setup=open_comparison_table),
    "uep_page": Scene(results_project, "uep", Page("_uep_view", (1100, 540))),
    "uep_clustering_dialog": Scene(results_project, "uep", Floating(open_uep_clustering, size=(1180, 760), prepare=frame_viewers)),
    "muon_relaxations_page": Scene(results_project, "muon_relaxations", Page("_muon_relaxation_view", (1100, 540))),
    "relaxation_analysis_workspace": Scene(
        results_project, None, Page("_relaxation_analysis_view", (1100, 680), prepare=frame_viewers), setup=open_analysis_workspace
    ),
    "muon_sites_summary": Scene(results_project, "muon_sites", Page("_muon_sites_summary_view", (1100, 800), prepare=frame_viewers)),
    "quantum_corrections_page": Scene(
        results_project, "quantum_corrections", Page("_quantum_corrections_view", (1100, 620))
    ),
    "energy_barriers_page": Scene(
        results_project, "energy_barriers", Page("_energy_barriers_view", (1100, 740))
    ),
    "energy_barrier_inspection": Scene(
        results_project, "energy_barriers", Page("_energy_barriers_view", (1100, 820), prepare=show_neb_analysis)
    ),
    "legacy_import_dialog": Scene(silicon_project, None, Dialog(open_legacy_import, kind="LegacyImportDialog", prepare=focus_default_button)),
}


def _run(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="directory to write the PNG files to")
    parser.add_argument("--list", action="store_true", help="print the screenshot names and exit")
    parser.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help="screenshots to render (default: all; see --list)",
    )
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(SCREENSHOTS))
        return 0
    unknown = sorted(set(args.names) - set(SCREENSHOTS))
    if unknown:
        parser.error(f"unknown screenshots: {', '.join(unknown)}")

    app = QApplication(sys.argv[:1])
    app.setApplicationName("MuFinder")
    app.setOrganizationName("MuFinder")
    from mufinder.app.theme import THEME_LIGHT, apply_theme

    apply_theme(app, THEME_LIGHT)
    install_offscreen_viewers()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    for name in args.names or SCREENSHOTS:
        render(app, SCREENSHOTS[name], output / f"{name}.png")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    finally:
        shutil.rmtree(_SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
