"""Shared pytest fixtures for MuFinder tests."""

from __future__ import annotations

import atexit
from collections.abc import Generator, Iterator
import gc
import os
import sys
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QThread
from PySide6.QtWidgets import QApplication, QWidget

import mufinder.castep_io.schema as _schema_module
from mufinder.app.theme import cleanup_theme_runtime


# Process-wide CASTEP keyword-schema state. Several tests rebuild it (refresh_keyword_catalog,
# swapping _candidate_catalog_sources, or pointing the *_KEYWORDS_JSON env vars at a stub) and
# restore it manually after their assertions. A mid-test assertion failure, or an env var leaked
# by another test, would skip that restore and strand the reduced starter catalog globally,
# cascading failures into every later test that reads the schema. The autouse fixture below makes
# the isolation guaranteed: teardown always runs, so the canonical schema is always reinstated.
_SCHEMA_GLOBAL_NAMES = (
    "CELL_KEYWORDS",
    "PARAM_KEYWORDS",
    "SOURCE_KEYWORDS",
    "SCHEMA",
    "CELL_SERIALIZATION_ORDER",
    "PARAM_SERIALIZATION_ORDER",
    "SERIALIZATION_INDEX",
    "CATALOG_SOURCE_INFO",
    "CATALOG_ISSUE",
    "CATALOG_CASTEP_VERSION",
    "_candidate_catalog_sources",
)
_SCHEMA_ENV_VARS = (
    "MUFINDER_CASTEP_KEYWORDS_JSON",
    "CASTEP_KEYWORDS_JSON",
    "MUFINDER_CASTEP_KEYWORDS_JSON_SETUP",
)


@pytest.fixture(autouse=True)
def _isolate_castep_keyword_catalog() -> Iterator[None]:
    """Snapshot and restore the global CASTEP keyword schema around every test."""

    saved_globals = {
        name: getattr(_schema_module, name)
        for name in _SCHEMA_GLOBAL_NAMES
        if hasattr(_schema_module, name)
    }
    saved_env = {name: os.environ.get(name) for name in _SCHEMA_ENV_VARS}
    try:
        yield
    finally:
        for name, value in saved_globals.items():
            setattr(_schema_module, name, value)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(autouse=True)
def _isolate_keyword_catalog_side_effects(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep catalog installs and catalog prompts away from the real machine.

    The suite normally runs on the bundled starter catalog, the state in which
    the application warns before submitting and offers to install a catalog.
    Unpatched, that would open modal prompts in every submission test and let
    an install write into the developer's real application-data directory.
    Tests of the prompts themselves re-patch ``_show_choice``.
    """

    import mufinder.castep_setup as castep_setup_module
    from mufinder.app.keyword_catalog_guard import CHOICE_CONTINUE, KeywordCatalogGuard

    data_dir = tmp_path_factory.mktemp("mufinder-app-data")
    monkeypatch.setattr(castep_setup_module, "_app_data_directory", lambda: data_dir)
    monkeypatch.setattr(
        KeywordCatalogGuard,
        "_show_choice",
        lambda self, **_kwargs: (CHOICE_CONTINUE, False),
    )


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

_PYTEST_EXITSTATUS = 0


def _drain_qt_deferred_deletes(app: QApplication) -> None:
    """Force-dispatch Qt DeferredDelete events outside a running event loop."""

    QCoreApplication.sendPostedEvents(None, 0)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _stop_qt_threads(app: QApplication) -> None:
    """Ask any QThreads parented into the widget tree to stop before teardown."""

    seen: set[int] = set()
    threads: list[QThread] = []
    for owner in [app, *list(app.topLevelWidgets()), *list(app.allWidgets())]:
        try:
            children = owner.findChildren(QThread)
        except (RuntimeError, SystemError):
            continue
        for thread in children:
            marker = id(thread)
            if marker in seen:
                continue
            seen.add(marker)
            threads.append(thread)

    for thread in threads:
        try:
            if thread.isRunning():
                thread.quit()
                thread.wait(5000)
        except (RuntimeError, SystemError):
            continue


def _cleanup_qt_application(app: QApplication, *, quit_app: bool = False) -> None:
    """Close, delete, and flush leaked widgets from the shared QApplication."""

    cleanup_theme_runtime(app)
    _stop_qt_threads(app)
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
    _drain_qt_deferred_deletes(app)
    gc.collect()
    if quit_app:
        app.quit()
        app.processEvents()


@pytest.fixture(scope="session")
def qapp() -> Generator[QApplication, None, None]:
    """Provide one QApplication instance for all UI tests."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setQuitOnLastWindowClosed(False)

    yield app

    _cleanup_qt_application(app, quit_app=True)


@pytest.fixture(autouse=True)
def _cleanup_qt_state_per_test() -> Iterator[None]:
    """Ensure Qt widgets and filters from one test do not leak into the next."""

    yield

    app = QApplication.instance()
    if app is None:
        return

    _cleanup_qt_application(app)


@pytest.fixture(autouse=True)
def _isolate_qsettings(tmp_path) -> Iterator[None]:
    """Redirect QSettings to a per-test path so xdist workers cannot share state."""

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    yield


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
        _cleanup_qt_application(app)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark GUI tests for fast local selection and strict marker hygiene."""

    del config
    for item in items:
        marker_names = {marker.name for marker in item.iter_markers()}
        if marker_names & {"unit", "gui"}:
            continue
        fixturenames = set(getattr(item, "fixturenames", ()))
        path = getattr(item, "path", None)
        is_app_test = path is not None and "tests/app/" in path.as_posix()
        if "qapp" in fixturenames or is_app_test:
            item.add_marker(pytest.mark.gui)
        else:
            item.add_marker(pytest.mark.unit)


@atexit.register
def _force_exit_before_extension_teardown() -> None:
    """Bypass crash-prone C-extension finalization after pytest completes."""

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_PYTEST_EXITSTATUS)
