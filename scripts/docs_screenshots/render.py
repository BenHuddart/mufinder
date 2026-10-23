"""The headless rendering machinery every example module shares.

Importing this module configures Qt, VTK and the settings directories before
anything imports them, so every other module of the package that touches Qt
imports it first. Settings and app data go to :data:`_SCRATCH`, a throwaway
directory, so a local run never reads or writes the developer's own MuFinder
configuration.

Qt draws on its ``offscreen`` platform at twice the logical size and VTK
renders through EGL, so the renders need no display. pyvistaqt's
``QtInteractor`` embeds a native render window, which the offscreen platform
cannot host; the 3D viewers are given :class:`OffscreenInteractor` instead,
which renders the same PyVista scene off screen and paints the frame.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any
import weakref

# Qt, VTK and the settings directories are configured before anything imports
# them.
_SCRATCH = Path(tempfile.mkdtemp(prefix="mufinder-screenshots-"))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"
os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkEGLRenderWindow"
# The screenshots are rendered at SCALE_FACTOR times their logical size, and
# docs/conf.py (SCREENSHOT_SCALE) divides the saved pixel size by the same
# number to give each <img> its logical width and height; the two must match,
# so an inherited QT_SCALE_FACTOR is overridden rather than kept.
SCALE_FACTOR = 2
os.environ["QT_SCALE_FACTOR"] = str(SCALE_FACTOR)
for _variable in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[_variable] = str(_SCRATCH / _variable.lower())

import numpy as np  # noqa: E402
from PySide6.QtCore import QEvent, QRect, QTimer  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDockWidget, QGroupBox, QWidget  # noqa: E402

from mufinder.domain.models import Structure  # noqa: E402
from mufinder.services.project_service import ProjectDocument, ProjectService  # noqa: E402

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "_generated" / "screenshots"

# Logical window size; the saved pixels are SCALE_FACTOR times larger.
WINDOW_SIZE = (1440, 1120)
# How often, and how long, a scene waits for a modal dialog to open.
MODAL_POLL_MS = 50
MODAL_TIMEOUT_MS = 20_000
# The project functional of every example host.
XC_FUNCTIONAL = "PBE"


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
        self._plotter.window_size = (
            max(int(self.width() * ratio), 1),
            max(int(self.height() * ratio), 1),
        )

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


# Screenshots are lossless WebP: Qt's WebP writer is lossless at quality 100.
IMAGE_SUFFIX = ".webp"
_IMAGE_FORMAT = "WEBP"
_LOSSLESS_QUALITY = 100
# A grab smaller than this is a blank or collapsed widget, not a screenshot.
MIN_IMAGE_BYTES = 10_000


def save(widget: QWidget, path: Path, rect: QRect | None = None) -> None:
    """Grab ``widget`` (or its ``rect``) once everything has settled and write it to ``path`` as lossless WebP."""

    settle(QApplication.instance())
    pixmap = widget.grab() if rect is None else widget.grab(rect)
    if not pixmap.save(str(path), _IMAGE_FORMAT, _LOSSLESS_QUALITY):
        raise RuntimeError(f"Could not write {path}")
    if path.stat().st_size <= MIN_IMAGE_BYTES:
        raise RuntimeError(f"{path} is suspiciously small: the grab may be blank")
    print(f"wrote {path}")


# --- Projects -----------------------------------------------------------------
#
# A project builder writes one saved project under a directory and returns its
# root. Builders go through ``ProjectService`` and the workflow widgets' own
# form logic, the calls the application makes, so the saved project is one the
# application could have written. The example modules define the builders.

ProjectBuilder = Callable[[Path], Path]


def new_project(
    directory: Path, name: str, host: Structure
) -> tuple[ProjectService, ProjectDocument]:
    """A new project named ``name`` under ``directory`` with ``host`` as its host structure."""

    service = ProjectService()
    root = directory / f"{name.lower().replace(' ', '-')}.mufinder"
    document = service.create_project(root, name)
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    return service, document


def custom_site(host: Structure, fractional: tuple[float, float, float]) -> dict[str, object]:
    """The saved form of a custom muon site at ``fractional`` in ``host``, as ``Select Muon Site`` stores it."""

    lattice = np.asarray(host.lattice_vectors, dtype=float)
    return {
        "source": "custom",
        "label": "Custom site",
        "fractional_position": list(fractional),
        "cartesian_position": (np.asarray(fractional, dtype=float) @ lattice).tolist(),
    }


def magnetic_project(
    name: str, host: Structure, model: Any, form: dict[str, object]
) -> ProjectBuilder:
    """A builder for ``host`` with the magnetic ``model`` and a Magnetic Fields ``form``.

    The project is a host structure and a Magnetic Fields form: no CASTEP job
    is involved.
    """

    def build(directory: Path) -> Path:
        service, document = new_project(directory, name, host)
        document.project.magnetic_structure = model.to_json()
        document.ui_state["magnetic_field_form"] = form
        service.save_project(document)
        return document.project_root

    return build


@contextmanager
def refuse_message_boxes(app: QApplication, *, expected: tuple[str, ...] = ()) -> Iterator[None]:
    """Fail the build, rather than hang it, if the application shows a message box.

    A message box whose title is in ``expected``, a notice the build's own
    steps are known to raise, is acknowledged instead.
    """

    from PySide6.QtWidgets import QMessageBox

    shown: list[str] = []

    def check() -> None:
        dialog = app.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            if dialog.windowTitle() in expected:
                dialog.accept()
                return
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
        raise RuntimeError("The project build showed a message box: " + "; ".join(shown))


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


def include_every_job(dialog: QDialog) -> None:
    """Tick every completed relaxation in ``Generate Comparison Table`` and click OK."""

    from PySide6.QtCore import Qt

    for row in range(dialog._table.rowCount()):
        dialog._table.item(row, dialog._CHECK_COLUMN).setCheckState(Qt.CheckState.Checked)
    dialog._accept_if_valid()


def run_uep_analysis(window: MainWindow, row: str) -> None:
    """Open the UEP analysis of ``row``, run a batch search, its ZPE and clustering, as the buttons do.

    The batch search and ZPE run their worker threads' ``run`` in this thread,
    and the random starts are seeded so every build finds the same minima.
    """

    from functools import partial

    from mufinder.app import uep_wizard
    from mufinder.science.site_generation import generate_random_sites

    job = window._uep_controller.jobs_for_keys([row])[0]
    window._on_open_uep_analysis_requested(row)
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
    # Grab this part of the dialog instead of all of it: a child widget (a
    # panel, a card) or a rectangle of the dialog (a row of a layout).
    region: Callable[[QDialog], QWidget | QRect] | None = None


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
        # processEvents leaves deferred deletes for the event loop, which a
        # headless run only enters inside a modal dialog. Delete the window
        # now: every live widget slows the next window's style sheet.
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


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


def when_modal(
    kind: str, step: Callable[[QDialog], None], *, timeout_ms: int = MODAL_TIMEOUT_MS
) -> None:
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
            region = None if target.region is None else target.region(dialog)
            if isinstance(region, QRect):
                save(dialog, path, region)
            else:
                save(dialog if region is None else region, path)
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


# Each builder's saved project, built once per process.
_PROJECT_CACHE: dict[ProjectBuilder, Path] = {}


def project_copy(builder: ProjectBuilder, prefix: str) -> Path:
    """A fresh copy of the project ``builder`` writes, building it on first use.

    A builder runs once per process; every scene that opens its project gets
    its own copy, so a scene that changes or saves the project leaves the
    next scene's copy alone. A saved project stores its paths relative to its
    root, so a copy opens as the original does.
    """

    built = _PROJECT_CACHE.get(builder)
    if built is None:
        built = builder(scratch_directory("project-"))
        _PROJECT_CACHE[builder] = built
    copy = scratch_directory(prefix) / built.name
    shutil.copytree(built, copy)
    return copy


def render(app: QApplication, scene: Scene, path: Path) -> None:
    """Open a copy of ``scene``'s project, drive it, and save its grab target to ``path``."""

    project = project_copy(scene.project, f"{path.stem}-")
    target = scene.target
    dashboard = isinstance(target, Dock) or (isinstance(target, Window) and target.dashboard)
    with opened(app, project, dashboard=dashboard) as window:
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
            docks = [
                dock
                for dock in window.findChildren(QDockWidget)
                if dock.windowTitle() == target.title
            ]
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


# --- Shared scene steps ---------------------------------------------------------
#
# Steps more than one example uses. A step only one example needs lives in
# that example's module.


def select_all_sites(page: QWidget) -> None:
    """Widen the preview and select every generated site, which marks them in it."""

    from PySide6.QtWidgets import QSplitter

    page.findChild(QSplitter).setSizes([520, 640])
    page._site_table.selectAll()
    settle(QApplication.instance())
    frame_viewers(page)


def cluster_and_pin(window: MainWindow, charge_state: str, name: str) -> None:
    """Select every row of a charge state's tab, create a workspace, cluster it and pin it."""

    relaxations = window._muon_relaxation_view
    window._set_active_route("muon_relaxations")
    relaxations._charge_tabs.setCurrentWidget(relaxations._table_for_charge_state(charge_state))
    relaxations._active_table().selectAll()
    when_modal("QInputDialog", lambda dialog: (dialog.setTextValue(name), dialog.accept()))
    relaxations._create_analysis_button.click()
    workspace_id = str(window._relaxation_analysis_workspaces()[-1]["id"])
    when_modal("RelaxationClusteringDialog", lambda dialog: dialog._apply_button.click())
    window._relaxation_analysis_view._open_clustering(workspace_id)
    window._on_muon_sites_pin_requested(workspace_id)


def frame_viewers(widget: QWidget) -> None:
    """Reset every 3D viewer in ``widget`` so it frames its structure at its laid-out size."""

    from mufinder.app.atomic_structure_viewer import AtomicStructureViewer

    for viewer in widget.findChildren(AtomicStructureViewer):
        if viewer.isVisible():
            viewer._reset_view()


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


def view_along_a(widget: QWidget) -> None:
    """Frame every 3D viewer in ``widget`` looking down a, where a cell's sublattices stand apart."""

    from mufinder.app.atomic_structure_viewer import AtomicStructureViewer

    frame_viewers(widget)
    for viewer in widget.findChildren(AtomicStructureViewer):
        if viewer.isVisible():
            viewer._set_named_view("a")


def show_host_with_spins(page: QWidget) -> None:
    """Widen the atom table to show the ``Spin (hbar/2)`` column, then view the cell down a."""

    from PySide6.QtWidgets import QSplitter

    page.findChild(QSplitter).setSizes([470, 630])
    settle(QApplication.instance())
    view_along_a(page)


def calculate_field(page: Any) -> None:
    """Click ``Calculate Dipolar Field`` on a Magnetic Fields page and select the first result row."""

    page._calculate_button.click()
    if page._result_table.rowCount() == 0:
        # A screenshot of a failed calculation must not reach the docs.
        raise RuntimeError(
            f"The field calculation gave no result: {page._calculation_status.text()}"
        )
    page._result_table.selectRow(0)


def calculation_card(window: MainWindow) -> QWidget:
    """The Magnetic Fields page's ``Calculation`` card."""

    for group in window._magnetic_field_view.findChildren(QGroupBox):
        if group.title() == "Calculation":
            return group
    raise RuntimeError("The Magnetic Fields page has no Calculation card")


def show_every_result_row(page: QWidget) -> None:
    """Make the results table tall enough to list every row, so the grab does not scroll it."""

    table = page._result_table
    rows = sum(table.rowHeight(row) for row in range(table.rowCount()))
    table.setMinimumHeight(table.horizontalHeader().height() + rows + 2 * table.frameWidth() + 2)


def calculate_and_list(window: MainWindow) -> None:
    """Calculate the field and show every result row."""

    page = window._magnetic_field_view
    calculate_field(page)
    show_every_result_row(page)


def save_magnetic_workspace(name: str) -> Callable[[MainWindow], None]:
    """A scene step: calculate the field, then click ``Save Workspace`` and name it ``name``."""

    def save(window: MainWindow) -> None:
        page = window._magnetic_field_view
        calculate_field(page)
        when_modal("QInputDialog", lambda dialog: (dialog.setTextValue(name), dialog.accept()))
        page._pin_button.click()

    return save


def current_page(window: MainWindow) -> QWidget:
    """The page the navigator shows, for pages kept per saved item."""

    return window._main_stack.currentWidget()


def scratch_directory(prefix: str) -> Path:
    """A new directory under the scratch directory, removed when the run ends."""

    return Path(tempfile.mkdtemp(prefix=prefix, dir=_SCRATCH))


# A writer that recreates the scratch directory (Qt makes the settings path
# with its parents) is outlasted by this many attempts.
_REMOVE_ATTEMPTS = 5


def remove_scratch() -> None:
    """Delete the scratch directory, even while another thread still writes into it.

    A scene that runs out of time is cleaned up from the time limit's thread
    while the scene goes on writing its project. Each attempt renames the
    directory first, so the scene's later paths under it fail instead of
    refilling the tree being deleted.
    """

    for attempt in range(_REMOVE_ATTEMPTS):
        if not _SCRATCH.exists():
            return
        doomed = _SCRATCH.with_name(f"{_SCRATCH.name}-removing-{attempt}")
        try:
            _SCRATCH.rename(doomed)
        except OSError:
            doomed = _SCRATCH
        shutil.rmtree(doomed, ignore_errors=True)
