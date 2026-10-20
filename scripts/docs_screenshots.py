"""Render the user-guide screenshots from the running application.

The screenshots are build outputs, not sources: ``make docs`` runs this script
first and Sphinx copies the images into the built site, so no image is
committed. Each screenshot opens a small project in a real ``MainWindow``,
drives the page through its buttons and grabs the window or dialog.

The script needs no display. Qt draws on its ``offscreen`` platform and VTK
renders through EGL, so it runs on a plain CI runner. pyvistaqt's
``QtInteractor`` embeds a native render window, which the offscreen platform
cannot host; the 3D viewers are given :class:`OffscreenInteractor` instead,
which renders the same PyVista scene off screen and paints the frame.

Run it through the Makefile::

    make screenshots
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
from typing import Any
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

from mufinder.domain.models import Structure  # noqa: E402
from mufinder.science.magnetic_structure import (  # noqa: E402
    AtomSelector,
    MagneticStructureModel,
    build_collinear_component,
    build_helix_component,
)
from mufinder.services.project_service import ProjectService  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "_generated" / "screenshots"

# Logical window size; QT_SCALE_FACTOR doubles the saved pixels.
WINDOW_SIZE = (1440, 1120)


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
        ratio = self.devicePixelRatioF()
        width = max(int(self.width() * ratio), 1)
        height = max(int(self.height() * ratio), 1)
        self._plotter.window_size = (width, height)
        pixels = np.ascontiguousarray(self._plotter.screenshot(return_img=True))
        height, width = pixels.shape[:2]
        image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGB888)
        self._frame = image.copy()
        self._frame.setDevicePixelRatio(ratio)
        self.update()

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


def capture_modal(
    open_dialog: Callable[[], object],
    path: Path,
    *,
    size: tuple[int, int] | None = None,
    prepare: Callable[[QDialog], None] | None = None,
) -> None:
    """Open a modal dialog through ``open_dialog``, save it to ``path`` and dismiss it.

    Dialogs size themselves to fit the screen, and the offscreen screen is
    small, so ``size`` gives the dialog its desktop size first.
    """

    captured: list[str] = []

    def grab() -> None:
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, QDialog):
            QTimer.singleShot(50, grab)
            return
        try:
            if size is not None:
                dialog.resize(*size)
            if prepare is not None:
                prepare(dialog)
            save(dialog, path)
            captured.append(str(path))
        finally:
            dialog.reject()

    QTimer.singleShot(0, grab)
    open_dialog()
    if not captured:
        raise RuntimeError(f"No dialog opened for {path.name}")


# --- Projects -----------------------------------------------------------------
#
# The models are the worked examples in docs/magnetic_fields.rst: one 1 μB ion
# in a simple cubic cell with a = 3 Å.


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


@dataclass(frozen=True)
class Scene:
    """A saved project to open: its name, magnetic model and Magnetic Fields form."""

    name: str
    model: MagneticStructureModel
    form: dict[str, object]


def write_project(directory: Path, scene: Scene) -> Path:
    service = ProjectService()
    root = directory / f"{scene.name.lower().replace(' ', '-')}.mufinder"
    document = service.create_project(root, scene.name)
    host = simple_cubic_host()
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    document.project.magnetic_structure = scene.model.to_json()
    document.ui_state["magnetic_field_form"] = scene.form
    service.save_project(document)
    return root


@contextmanager
def opened(app: QApplication, scene: Scene) -> Iterator[Any]:
    """A shown ``MainWindow`` with ``scene`` open on the Magnetic Fields page."""

    from mufinder.app.main_window import MainWindow

    window = MainWindow()
    window.resize(*WINDOW_SIZE)
    # The projects run no jobs, so the empty Job Dashboard would only take
    # room from the page.
    for dock in window.findChildren(QDockWidget):
        dock.hide()
    window.show()
    if not window._open_project_from_path(write_project(_SCRATCH, scene), show_errors=False):
        raise RuntimeError(f"Could not open the {scene.name} project")
    window._set_active_route("magnetic_fields")
    settle(app)
    try:
        yield window
    finally:
        # A hidden window closes without asking to save the project.
        window.hide()
        window.close()
        window.deleteLater()
        app.processEvents()


def calculate(page: Any) -> None:
    """Click ``Calculate Dipolar Field`` and select the first result row."""

    page._calculate_button.click()
    if page._result_table.rowCount() == 0:
        # A screenshot of a failed calculation must not reach the docs.
        raise RuntimeError(f"The field calculation gave no result: {page._calculation_status.text()}")
    page._result_table.selectRow(0)


# --- Screenshots --------------------------------------------------------------


def magnetic_fields_page(app: QApplication, output: Path) -> None:
    scene = Scene(
        "Simple cubic ferromagnet",
        ferromagnet(),
        {"demag_shape": "sphere", "selected_muon_site": custom_site((0.5, 0.5, 0.0))},
    )
    with opened(app, scene) as window:
        calculate(window._magnetic_field_view)
        save(window, output / "magnetic_fields_page.png")


def magnetic_structure_builder(app: QApplication, output: Path) -> None:
    scene = Scene("Commensurate helix", helix(0.25), {"selected_muon_site": custom_site((0.5, 0.0, 0.0))})
    with opened(app, scene) as window:
        page = window._magnetic_field_view

        def show_magnetic_cell(dialog: QDialog) -> None:
            # One magnetic cell of the q = 1/4 helix is four host cells along c.
            dialog._repeat_c.setValue(4)

        capture_modal(
            page._edit_structure_button.click,
            output / "magnetic_structure_builder.png",
            size=(1180, 900),
            prepare=show_magnetic_cell,
        )


def magnetic_field_distribution(app: QApplication, output: Path) -> None:
    scene = Scene(
        "Incommensurate helix",
        helix(0.2861),
        {"demag_shape": "none", "selected_muon_site": custom_site((0.5, 0.0, 0.0))},
    )
    with opened(app, scene) as window:
        page = window._magnetic_field_view
        calculate(page)
        capture_modal(page._open_distribution_button.click, output / "magnetic_field_distribution.png")


SCREENSHOTS: dict[str, Callable[[QApplication, Path], None]] = {
    "magnetic_fields_page": magnetic_fields_page,
    "magnetic_structure_builder": magnetic_structure_builder,
    "magnetic_field_distribution": magnetic_field_distribution,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="directory to write the PNG files to")
    parser.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help=f"screenshots to render: {', '.join(SCREENSHOTS)} (default: all)",
    )
    args = parser.parse_args(argv)
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
    try:
        for name in args.names or SCREENSHOTS:
            SCREENSHOTS[name](app, output)
    finally:
        shutil.rmtree(_SCRATCH, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
