"""Entrypoint tests for mufinder.main."""

from __future__ import annotations

from types import ModuleType
from types import SimpleNamespace
import sys

import mufinder.main as main_module


class _FakeApp:
    def __init__(self, argv):
        self.argv = argv
        self.app_name = None
        self.org_name = None
        self.window_icon = None
        self.events_processed = False

    def setApplicationName(self, value):
        self.app_name = value

    def setOrganizationName(self, value):
        self.org_name = value

    def setWindowIcon(self, icon):
        self.window_icon = icon

    def processEvents(self):
        self.events_processed = True

    def exec(self):
        return 77


class _FakeWindow:
    def __init__(self):
        self.shown = False

    def show(self):
        self.shown = True


class _FakeSplash:
    def __init__(self):
        self.shown = False
        self.progress = []
        self.finished_with = None

    def show(self):
        self.shown = True

    def set_progress(self, value, message):
        self.progress.append((value, message))

    def finish(self, window):
        self.finished_with = window


def test_run_bootstraps_qt_and_returns_exit_code(monkeypatch):
    created = {}

    def fake_app(argv):
        app = _FakeApp(argv)
        created["app"] = app
        return app

    def fake_window_ctor():
        win = _FakeWindow()
        created["window"] = win
        return win

    monkeypatch.setattr(main_module, "QApplication", fake_app)
    monkeypatch.setattr(main_module, "_main_window_ctor", lambda: fake_window_ctor)
    monkeypatch.setattr(main_module, "_apply_saved_castep_environment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "_app_icon", lambda: "icon-object")
    monkeypatch.setattr(
        main_module,
        "_theme_helpers",
        lambda: (
            lambda *_args, **_kwargs: "light",
            lambda _settings: "system",
        ),
    )
    monkeypatch.setattr(main_module, "_splash_screen_ctor", lambda: _FakeSplash)

    code = main_module.run()

    assert code == 77
    assert created["app"].app_name == "MuFinder"
    assert created["app"].org_name == "MuFinder"
    assert created["app"].window_icon == "icon-object"
    assert created["window"].shown is True
    assert created["app"].events_processed is True


def test_run_applies_saved_castep_environment(monkeypatch):
    created = {}

    monkeypatch.setattr(main_module, "QApplication", lambda argv: _FakeApp(argv))
    monkeypatch.setattr(main_module, "_main_window_ctor", lambda: _FakeWindow)
    monkeypatch.setattr(main_module, "_app_icon", lambda: "icon-object")
    monkeypatch.setattr(
        main_module,
        "_theme_helpers",
        lambda: (
            lambda *_args, **_kwargs: "light",
            lambda _settings: "system",
        ),
    )
    monkeypatch.setattr(main_module, "_splash_screen_ctor", lambda: _FakeSplash)

    def fake_apply(settings):
        created["settings"] = settings

    monkeypatch.setattr(main_module, "_apply_saved_castep_environment", fake_apply)

    main_module.run()

    assert created["settings"] is not None


def test_run_sets_windows_app_id_before_qt_startup(monkeypatch):
    called = []

    monkeypatch.setattr(main_module, "_set_windows_app_id", lambda: called.append("app-id"))
    monkeypatch.setattr(main_module, "QApplication", lambda argv: _FakeApp(argv))
    monkeypatch.setattr(main_module, "_main_window_ctor", lambda: _FakeWindow)
    monkeypatch.setattr(main_module, "_apply_saved_castep_environment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "_app_icon", lambda: "icon-object")
    monkeypatch.setattr(
        main_module,
        "_theme_helpers",
        lambda: (
            lambda *_args, **_kwargs: "light",
            lambda _settings: "system",
        ),
    )
    monkeypatch.setattr(main_module, "_splash_screen_ctor", lambda: _FakeSplash)

    main_module.run()

    assert called == ["app-id"]


def test_set_windows_app_id_calls_shell32(monkeypatch):
    class _FakeShell32:
        def __init__(self):
            self.calls = []

        def SetCurrentProcessExplicitAppUserModelID(self, value):
            self.calls.append(value)

    fake_shell32 = _FakeShell32()
    monkeypatch.setattr(main_module.sys, "platform", "win32")
    monkeypatch.setattr(main_module.ctypes, "windll", SimpleNamespace(shell32=fake_shell32), raising=False)

    main_module._set_windows_app_id()

    assert fake_shell32.calls == ["io.github.benhuddart.mufinder"]


def test_set_windows_app_id_ignores_missing_windows_api(monkeypatch):
    monkeypatch.setattr(main_module.sys, "platform", "win32")
    monkeypatch.delattr(main_module.ctypes, "windll", raising=False)

    assert main_module._set_windows_app_id() is None


def test_helper_import_wrappers(monkeypatch):
    fake_main_window_module = ModuleType("mufinder.app.main_window")
    fake_main_window_module.MainWindow = object()
    fake_splash_module = ModuleType("mufinder.app.splash_screen")
    fake_splash_module.MuFinderSplashScreen = object()
    fake_theme_module = ModuleType("mufinder.app.theme")
    fake_theme_module.apply_theme = object()
    fake_theme_module.load_theme_mode = object()
    fake_branding_module = ModuleType("mufinder.branding")
    fake_branding_module.app_icon = lambda: "wrapped-icon"
    recorded = {}
    fake_castep_setup_module = ModuleType("mufinder.castep_setup")

    def fake_apply_saved(settings):
        recorded["settings"] = settings

    fake_castep_setup_module.apply_saved_castep_environment = fake_apply_saved

    monkeypatch.setitem(sys.modules, "mufinder.app.main_window", fake_main_window_module)
    monkeypatch.setitem(sys.modules, "mufinder.app.splash_screen", fake_splash_module)
    monkeypatch.setitem(sys.modules, "mufinder.app.theme", fake_theme_module)
    monkeypatch.setitem(sys.modules, "mufinder.branding", fake_branding_module)
    monkeypatch.setitem(sys.modules, "mufinder.castep_setup", fake_castep_setup_module)

    assert main_module._main_window_ctor() is fake_main_window_module.MainWindow
    assert main_module._splash_screen_ctor() is fake_splash_module.MuFinderSplashScreen
    assert main_module._theme_helpers() == (
        fake_theme_module.apply_theme,
        fake_theme_module.load_theme_mode,
    )
    assert main_module._app_icon() == "wrapped-icon"

    settings = object()
    main_module._apply_saved_castep_environment(settings)
    assert recorded["settings"] is settings
