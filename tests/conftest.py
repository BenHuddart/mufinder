"""Shared pytest fixtures for MuFinder tests."""

from __future__ import annotations

import atexit
from collections.abc import Generator, Iterator
import gc
import os
import sys

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QThread
from PySide6.QtWidgets import QApplication

import mufinder.castep_io.schema as _schema_module
from mufinder.app.gui_garbage_collector import GuiGarbageCollector, install_gui_garbage_collector
from mufinder.app.theme import cleanup_theme_runtime
from tests.support.pyvista_fakes import FakePyVistaCanvas, FakePyVistaTriadCanvas


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


# The test module whose objects are currently frozen out of garbage collection (see
# ``_collect_garbage``); ``None`` means nothing is frozen.
_gc_frozen_module: object = None


def _collect_garbage(module: object = None) -> None:
    """Collect reference cycles left by a test, scanning only what is new since the last test.

    A full ``gc.collect()`` walks every tracked object: around 200k once Qt, VTK, SciPy and
    matplotlib are imported, and more as the session grows. Paid after each of ~4000 tests,
    that was close to half of the suite's run time. Instead, the survivors of each collection
    are frozen into the permanent generation, so the next collection only examines objects
    allocated since, which is what the test that just ran created.

    A frozen object is never collected, so a cycle that outlives one test and becomes garbage
    in a later one would be missed. Passing the test's module bounds that: the first teardown
    in each new module (and any call without a module) unfreezes everything and does a full
    collection.
    """

    global _gc_frozen_module
    if module is None or module != _gc_frozen_module:
        gc.unfreeze()
        _gc_frozen_module = None
    gc.collect()
    if module is not None:
        gc.freeze()
        _gc_frozen_module = module


def _cleanup_qt_application(
    app: QApplication,
    *,
    quit_app: bool = False,
    gc_module: object = None,
) -> None:
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
    _collect_garbage(gc_module)
    if quit_app:
        app.quit()
        app.processEvents()


_session_garbage_collector: GuiGarbageCollector | None = None


@pytest.fixture(scope="session")
def qapp() -> Generator[QApplication, None, None]:
    """Provide one QApplication instance for all UI tests.

    Like the application, the session frees reference cycles on the GUI thread
    only (``mufinder.app.gui_garbage_collector``), so a worker thread a test
    starts never frees a Qt object.
    """

    global _session_garbage_collector
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    _session_garbage_collector = install_gui_garbage_collector(app)

    yield app

    _session_garbage_collector.stop()
    _session_garbage_collector = None
    _cleanup_qt_application(app, quit_app=True)


@pytest.fixture
def without_session_garbage_collector(qapp: QApplication) -> Iterator[None]:
    """Stop the session's garbage collector for one test, which turns automatic collection back on."""

    assert _session_garbage_collector is not None
    _session_garbage_collector.stop()
    try:
        yield
    finally:
        _session_garbage_collector.start()


@pytest.fixture(autouse=True)
def _cleanup_qt_state_per_test(request: pytest.FixtureRequest) -> Iterator[None]:
    """Ensure Qt widgets and filters from one test do not leak into the next."""

    yield

    app = QApplication.instance()
    if app is None:
        return

    _cleanup_qt_application(app, gc_module=request.node.path)


@pytest.fixture
def main_window(qapp: QApplication):
    """A fresh ``MainWindow``.

    It needs no explicit close: the per-test Qt cleanup above closes and deletes
    every top-level widget once the test finishes.
    """

    from mufinder.app.main_window import MainWindow

    return MainWindow()


@pytest.fixture
def project_with_host(main_window, tmp_path):
    """``(window, service, document)`` for a project with the default host structure loaded.

    The host is :func:`tests.support.structures.host_structure`; reach it through
    ``document.project.host_structure_id``.
    """

    from tests.support.main_window import open_project_with_host

    service, document, _host = open_project_with_host(main_window, tmp_path)
    return main_window, service, document


@pytest.fixture(autouse=True)
def confirmed_destructive_actions(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """Answer every destructive-action confirmation with "go ahead", and record it.

    ``workflow_table_utils.confirm_destructive`` opens a modal dialog, which can
    never be right in a test: it would block until the per-test timeout. So it
    is replaced for every test with a stub that accepts. A test that cares
    requests this fixture to inspect the recorded prompts, or patches the
    function itself to decline.
    """

    from mufinder.app import workflow_table_utils

    prompts: list[dict[str, str]] = []

    def _accept(
        _parent,
        title,
        text,
        detail="",
        accept_label="Remove",
        reject_label="Cancel",
        *,
        detailed_text="",
    ) -> bool:
        prompt = {
            "title": title,
            "text": text,
            "detail": detail,
            "accept_label": accept_label,
            "reject_label": reject_label,
        }
        if detailed_text:
            prompt["detailed_text"] = detailed_text
        prompts.append(prompt)
        return True

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", _accept)
    return prompts


@pytest.fixture(autouse=True)
def answered_choices(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Answer every ``confirm_choice`` question with its default, and record it.

    Like :func:`confirm_destructive`, ``workflow_table_utils.confirm_choice``
    opens a modal dialog that would block a test. Each prompt is recorded with
    its ``choices``, ``default`` and ``escape`` keys; a test that needs another
    answer patches the function itself.
    """

    from mufinder.app import workflow_table_utils

    prompts: list[dict[str, object]] = []

    def _default(_parent, title, text, choices, *, default, escape, **kwargs) -> str:
        prompts.append(
            {
                "title": title,
                "text": text,
                "choices": list(choices),
                "default": default,
                "escape": escape,
                **kwargs,
            }
        )
        return default

    monkeypatch.setattr(workflow_table_utils, "confirm_choice", _default)
    return prompts


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

    monkeypatch.setattr(atomic_structure_viewer_module, "_pyvista_modules", lambda: (pv, object()))
    monkeypatch.setattr(atomic_structure_viewer_module, "_PyVistaCanvas", FakePyVistaCanvas)
    monkeypatch.setattr(
        atomic_structure_viewer_module, "_PyVistaTriadCanvas", FakePyVistaTriadCanvas
    )


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


@pytest.fixture
def real_muesr() -> None:
    """Skip a test unless the real muesr and muLFC engines import."""

    from mufinder.science.magnetic_fields import (
        MagneticFieldDependencyError,
        _load_muesr_engines,
    )

    try:
        _load_muesr_engines()
    except MagneticFieldDependencyError as exc:
        pytest.skip(str(exc))
