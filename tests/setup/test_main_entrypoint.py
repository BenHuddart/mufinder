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
    monkeypatch.setattr(
        main_module, "_apply_saved_castep_environment", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(main_module, "_app_icon", lambda: "icon-object")
    monkeypatch.setattr(main_module, "_install_text_editing_shortcuts", lambda _app: None)
    monkeypatch.setattr(
        main_module,
        "_install_gui_garbage_collector",
        lambda app: created.setdefault("collected_by", app),
    )
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
    assert created["collected_by"] is created["app"]


def test_run_applies_saved_castep_environment(monkeypatch):
    created = {}

    monkeypatch.setattr(main_module, "QApplication", lambda argv: _FakeApp(argv))
    monkeypatch.setattr(main_module, "_main_window_ctor", lambda: _FakeWindow)
    monkeypatch.setattr(main_module, "_app_icon", lambda: "icon-object")
    monkeypatch.setattr(main_module, "_install_text_editing_shortcuts", lambda _app: None)
    monkeypatch.setattr(main_module, "_install_gui_garbage_collector", lambda _app: None)
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
    monkeypatch.setattr(
        main_module, "_apply_saved_castep_environment", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(main_module, "_app_icon", lambda: "icon-object")
    monkeypatch.setattr(main_module, "_install_text_editing_shortcuts", lambda _app: None)
    monkeypatch.setattr(main_module, "_install_gui_garbage_collector", lambda _app: None)
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
    monkeypatch.setattr(
        main_module.ctypes, "windll", SimpleNamespace(shell32=fake_shell32), raising=False
    )

    main_module._set_windows_app_id()

    assert fake_shell32.calls == ["io.github.benhuddart.mufinder"]


def test_set_windows_app_id_ignores_missing_windows_api(monkeypatch):
    monkeypatch.setattr(main_module.sys, "platform", "win32")
    monkeypatch.delattr(main_module.ctypes, "windll", raising=False)

    assert main_module._set_windows_app_id() is None


def _clear_platform_env(monkeypatch):
    monkeypatch.setattr(main_module.sys, "platform", "linux")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)


def test_force_x11_platform_sets_xcb_under_wayland(monkeypatch):
    _clear_platform_env(monkeypatch)
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")

    main_module._force_x11_platform_on_wayland()

    assert main_module.os.environ["QT_QPA_PLATFORM"] == "xcb"


def test_force_x11_platform_respects_existing_override(monkeypatch):
    _clear_platform_env(monkeypatch)
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("DISPLAY", ":0")

    main_module._force_x11_platform_on_wayland()

    assert main_module.os.environ["QT_QPA_PLATFORM"] == "wayland"


def test_force_x11_platform_skips_without_xwayland_display(monkeypatch):
    _clear_platform_env(monkeypatch)
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")

    main_module._force_x11_platform_on_wayland()

    assert "QT_QPA_PLATFORM" not in main_module.os.environ


def test_force_x11_platform_skips_on_non_wayland(monkeypatch):
    _clear_platform_env(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":0")

    main_module._force_x11_platform_on_wayland()

    assert "QT_QPA_PLATFORM" not in main_module.os.environ


def test_force_x11_platform_skips_off_linux(monkeypatch):
    _clear_platform_env(monkeypatch)
    monkeypatch.setattr(main_module.sys, "platform", "win32")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("DISPLAY", ":0")

    main_module._force_x11_platform_on_wayland()

    assert "QT_QPA_PLATFORM" not in main_module.os.environ


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
    fake_collector_module = ModuleType("mufinder.app.gui_garbage_collector")
    fake_collector_module.install_gui_garbage_collector = lambda app: recorded.setdefault(
        "collector_app", app
    )

    def fake_apply_saved(settings):
        recorded["settings"] = settings

    fake_castep_setup_module.apply_saved_castep_environment = fake_apply_saved

    monkeypatch.setitem(sys.modules, "mufinder.app.main_window", fake_main_window_module)
    monkeypatch.setitem(sys.modules, "mufinder.app.splash_screen", fake_splash_module)
    monkeypatch.setitem(sys.modules, "mufinder.app.theme", fake_theme_module)
    monkeypatch.setitem(sys.modules, "mufinder.branding", fake_branding_module)
    monkeypatch.setitem(sys.modules, "mufinder.castep_setup", fake_castep_setup_module)
    monkeypatch.setitem(sys.modules, "mufinder.app.gui_garbage_collector", fake_collector_module)

    assert main_module._main_window_ctor() is fake_main_window_module.MainWindow
    assert main_module._splash_screen_ctor() is fake_splash_module.MuFinderSplashScreen
    assert main_module._theme_helpers() == (
        fake_theme_module.apply_theme,
        fake_theme_module.load_theme_mode,
    )
    assert main_module._app_icon() == "wrapped-icon"
    app = object()
    main_module._install_gui_garbage_collector(app)
    assert recorded["collector_app"] is app

    settings = object()
    main_module._apply_saved_castep_environment(settings)
    assert recorded["settings"] is settings


def test_prompt_for_keyword_catalog_is_deferred_to_the_event_loop(monkeypatch):
    scheduled = []
    monkeypatch.setattr(
        main_module.QTimer,
        "singleShot",
        lambda delay, callback: scheduled.append((delay, callback)),
    )
    prompted = []
    window = SimpleNamespace(prompt_for_keyword_catalog_if_needed=lambda: prompted.append(True))

    main_module._prompt_for_keyword_catalog(window)

    # Nothing is shown while run() is still tearing the splash screen down.
    assert prompted == []
    assert [delay for delay, _callback in scheduled] == [0]
    scheduled[0][1]()
    assert prompted == [True]

    # A window without the hook (the test doubles above) is left alone.
    main_module._prompt_for_keyword_catalog(_FakeWindow())
    assert len(scheduled) == 1


# The splash-exposure helpers below are driven directly: run() reaches them only
# with a real Qt splash screen, whose exposure timing a test cannot control.


def test_process_events_falls_back_when_the_timed_overload_is_missing():
    class AppWithoutTimedOverload:
        def __init__(self):
            self.calls = []

        def processEvents(self, *args):
            self.calls.append(args)
            if args:
                raise TypeError("overload not available")

    app = AppWithoutTimedOverload()
    main_module._process_events(app, max_ms=10)
    main_module._process_events(app, max_ms=None)
    main_module._process_events(object(), max_ms=10)

    assert app.calls == [
        (main_module.QEventLoop.ProcessEventsFlag.AllEvents, 10),
        (),
        (),
    ]


def test_splash_exposure_tolerates_missing_and_deleted_windows():
    class ExposedHandle:
        def isExposed(self):
            return True

    class ExposedSplash:
        def windowHandle(self):
            return ExposedHandle()

    class InvisibleSplash:
        def windowHandle(self):
            return None

        def isVisible(self):
            return False

    class DeletedSplash:
        def windowHandle(self):
            class _Handle:
                def isExposed(self):
                    raise RuntimeError("deleted")

            return _Handle()

        def isVisible(self):
            raise RuntimeError("deleted")

    assert main_module._is_splash_exposed(ExposedSplash()) is True
    assert main_module._is_splash_exposed(InvisibleSplash()) is False
    assert main_module._is_splash_exposed(DeletedSplash()) is False


def test_wait_for_splash_exposed_pumps_events_until_the_splash_is_shown(monkeypatch):
    class FakeTimer:
        def __init__(self):
            self._elapsed = -10

        def start(self):
            self._elapsed = 0

        def elapsed(self):
            self._elapsed += 10
            return self._elapsed

    exposures = iter([False, False, True])
    process_calls = []

    monkeypatch.setattr(main_module, "QElapsedTimer", FakeTimer)
    monkeypatch.setattr(main_module.QThread, "msleep", lambda _ms: None)
    monkeypatch.setattr(main_module, "_is_splash_exposed", lambda _splash: next(exposures))
    monkeypatch.setattr(
        main_module,
        "_process_events",
        lambda _app, *, max_ms=None: process_calls.append(max_ms),
    )

    main_module._wait_for_splash_exposed(object(), object(), timeout_ms=40)

    assert process_calls == [25, 25, 10]
