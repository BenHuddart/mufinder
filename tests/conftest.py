"""Shared pytest fixtures for MuFinder tests."""

from __future__ import annotations

import atexit
from collections.abc import Generator
import gc
import os
import sys
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QWidget

from mufinder.app.theme import cleanup_theme_runtime


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

_PYTEST_EXITSTATUS = 0


@pytest.fixture(scope="session")
def qapp() -> Generator[QApplication, None, None]:
    """Provide one QApplication instance for all UI tests."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setQuitOnLastWindowClosed(False)

    yield app

    cleanup_theme_runtime(app)
    for widget in list(app.topLevelWidgets()):
        try:
            widget.close()
        except (RuntimeError, SystemError):
            continue
    for widget in [w for w in list(app.allWidgets()) if w.parent() is None]:
        try:
            widget.deleteLater()
        except (RuntimeError, SystemError):
            continue
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, 0)
    app.processEvents()
    gc.collect()
    app.quit()
    app.processEvents()


@pytest.fixture(autouse=True)
def _cleanup_qt_state_per_test(request: pytest.FixtureRequest):
    """Ensure Qt widgets and filters from one test do not leak into the next."""

    yield

    if "qapp" not in request.fixturenames:
        return

    app = QApplication.instance()
    if app is None:
        return

    cleanup_theme_runtime(app)
    for widget in list(app.topLevelWidgets()):
        try:
            widget.close()
            widget.deleteLater()
        except (RuntimeError, SystemError):
            continue
    QCoreApplication.sendPostedEvents(None, 0)
    app.processEvents()
    gc.collect()


@pytest.fixture(autouse=True)
def _fake_pyvista_qt_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Replace crash-prone QtInteractor widgets with lightweight Qt fakes during tests."""

    if "qapp" not in request.fixturenames:
        return

    pv = pytest.importorskip("pyvista")

    from mufinder.app import atomic_structure_viewer as atomic_structure_viewer_module

    class _FakeRenderTimer:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    class _FakeProperty:
        def __init__(self) -> None:
            self.opacity = 1.0

        def SetOpacity(self, value: float) -> None:
            self.opacity = float(value)

    class _FakeActor:
        def __init__(self, mesh=None, kwargs: dict | None = None) -> None:
            self.mesh = mesh
            self.kwargs = {} if kwargs is None else dict(kwargs)
            self._property = _FakeProperty()

        def GetProperty(self):
            return self._property

    class _FakeRenderer:
        def AutomaticLightCreationOff(self) -> None:
            return

        def UseImageBasedLightingOff(self) -> None:
            return

        def UseSphericalHarmonicsOff(self) -> None:
            return

        def SetEnvironmentTexture(self, _texture) -> None:
            return

    class _FakeInteractor:
        def __init__(self) -> None:
            self.picker = None
            self.interactor = SimpleNamespace(GetShiftKey=lambda: 0)
            self._event_position = (0, 0)

        def add_observer(self, _event_name: str, _callback) -> None:
            return

        def get_event_position(self) -> tuple[int, int]:
            return self._event_position

        def get_poked_renderer(self):
            return "renderer"

    class _FakePlotter(QWidget):
        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.renderer = _FakeRenderer()
            self.camera = SimpleNamespace(
                position=(8.0, 8.0, 8.0),
                focal_point=(0.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
                parallel_scale=1.0,
            )
            self.iren = _FakeInteractor()
            self.render_timer = _FakeRenderTimer()
            self._actors: list[object] = []

        def setPalette(self, _palette) -> None:
            return

        def set_background(self, *_args, **_kwargs) -> None:
            return

        def enable_parallel_projection(self) -> None:
            return

        def enable_anti_aliasing(self, *_args, **_kwargs) -> None:
            return

        def disable_shadows(self) -> None:
            return

        def disable_eye_dome_lighting(self) -> None:
            return

        def remove_environment_texture(self) -> None:
            return

        def remove_all_lights(self) -> None:
            return

        def add_light(self, light) -> object:
            actor = _FakeActor(light)
            self._actors.append(actor)
            return actor

        def clear(self) -> None:
            self._actors.clear()

        def add_mesh(self, mesh, **kwargs):
            actor = _FakeActor(mesh, kwargs)
            self._actors.append(actor)
            return actor

        def remove_actor(self, actor, *, render: bool = True) -> None:
            del render
            if actor in self._actors:
                self._actors.remove(actor)

        def add_text(self, *args, **kwargs):
            actor = _FakeActor(args, kwargs)
            self._actors.append(actor)
            return actor

        def add_point_labels(self, *args, **kwargs):
            actor = _FakeActor(args, kwargs)
            self._actors.append(actor)
            return actor

        def enable_custom_trackball_style(self, **_kwargs) -> None:
            return

        def reset_camera_clipping_range(self) -> None:
            return

        def render(self) -> None:
            return

        def close(self) -> None:
            return

    class _FakePyVistaCanvas(QWidget):
        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._plotter = _FakePlotter(self)

        @property
        def plotter(self):
            return self._plotter

        def draw_idle(self) -> None:
            self._plotter.render()

    class _FakePyVistaTriadCanvas(QWidget):
        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._plotter = _FakePlotter(self)
            self._scene = None
            self._light_direction = None
            self.setFixedWidth(132)
            self.setFixedHeight(130)

        @property
        def plotter(self):
            return self._plotter

        def set_scene(self, **kwargs) -> None:
            self._scene = dict(kwargs)

        def set_light_direction(self, direction) -> None:
            self._light_direction = direction

        def draw_idle(self) -> None:
            self._plotter.render()

    monkeypatch.setattr(atomic_structure_viewer_module, "_pyvista_modules", lambda: (pv, object()))
    monkeypatch.setattr(atomic_structure_viewer_module, "_PyVistaCanvas", _FakePyVistaCanvas)
    monkeypatch.setattr(atomic_structure_viewer_module, "_PyVistaTriadCanvas", _FakePyVistaTriadCanvas)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Record final status and perform last Qt cleanup before interpreter shutdown."""

    global _PYTEST_EXITSTATUS
    _PYTEST_EXITSTATUS = int(exitstatus)

    app = QApplication.instance()
    if app is not None:
        cleanup_theme_runtime(app)
        for widget in list(app.topLevelWidgets()):
            try:
                widget.close()
                widget.deleteLater()
            except (RuntimeError, SystemError):
                continue
        QCoreApplication.sendPostedEvents(None, 0)
        app.processEvents()
        gc.collect()


@atexit.register
def _force_exit_before_extension_teardown() -> None:
    """Bypass crash-prone C-extension finalization after pytest completes."""

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_PYTEST_EXITSTATUS)
