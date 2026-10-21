"""Render the documentation screenshots from the running application.

The screenshots are build outputs, not sources: ``make docs`` runs this script
first and Sphinx copies the images into the built site, so no image is
committed. Each screenshot is a :class:`Scene`: a project builder writes a
small saved project, the script opens it in a real ``MainWindow`` on one
navigator route, optionally drives the window, and grabs a :class:`Target`:
the whole window, one page widget, a dock, or a modal dialog.

Most scenes open :func:`fixture_project`, a rutile TiO2 project with a saved
2x2x2 supercell, configured but unsubmitted cutoff and k-point sweeps, and
candidate muon sites. :func:`converged_project` adds adopted convergence
defaults with illustrative pinned plots, and :func:`dashboard_project` adds
three finished cutoff jobs for the Job Dashboard. The Magnetic Fields scenes
open :func:`magnetic_project`, a one-ion cubic cell with a magnetic model. No
scene runs CASTEP.

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
    """Grab one page widget, a ``MainWindow`` attribute, laid out at ``size``."""

    attribute: str
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


Target = Window | Page | Dock | Dialog


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
            widget = getattr(window, target.attribute)
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
