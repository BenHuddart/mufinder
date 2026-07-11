"""Application entrypoint for MuFinder."""

from __future__ import annotations

import ctypes
import os
import sys

from PySide6.QtCore import QEventLoop, QElapsedTimer, QSettings, QThread
from PySide6.QtWidgets import QApplication


def _main_window_ctor():
    from mufinder.app.main_window import MainWindow

    return MainWindow


def _splash_screen_ctor():
    from mufinder.app.splash_screen import MuFinderSplashScreen

    return MuFinderSplashScreen


def _theme_helpers():
    from mufinder.app.theme import apply_theme, load_theme_mode

    return apply_theme, load_theme_mode


def _install_text_editing_shortcuts(app: QApplication) -> None:
    from mufinder.app.text_editing import install_text_editing_shortcuts

    install_text_editing_shortcuts(app)


def _app_icon():
    from mufinder.branding import app_icon

    return app_icon()


def _set_windows_app_id() -> None:
    """Set a stable Windows application ID for taskbar icon grouping."""

    if not sys.platform.startswith("win"):
        return

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("io.github.benhuddart.mufinder")
    except (AttributeError, OSError):
        return


def _force_x11_platform_on_wayland() -> None:
    """Prefer the X11 (xcb) Qt platform under Wayland sessions.

    MuFinder embeds VTK render windows for its 3D viewers, and VTK renders
    through X11/GLX. Under a native Qt Wayland window the widget's ``winId`` is
    not an X11 window, so VTK issues ``XConfigureWindow`` on an invalid id and
    the process aborts with a fatal ``BadWindow`` X error. Routing Qt through
    XWayland (``xcb``) gives VTK real X11 windows.

    Only applied on Linux, when the user has not pinned ``QT_QPA_PLATFORM``
    themselves and an XWayland ``DISPLAY`` is available to connect to.
    """

    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("QT_QPA_PLATFORM"):
        return
    is_wayland = os.environ.get("XDG_SESSION_TYPE") == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))
    if not is_wayland:
        return
    if not os.environ.get("DISPLAY"):
        return
    os.environ["QT_QPA_PLATFORM"] = "xcb"


def _apply_saved_castep_environment(settings: QSettings) -> None:
    from mufinder.castep_setup import apply_saved_castep_environment

    apply_saved_castep_environment(settings)


def _process_events(app: QApplication, *, max_ms: int | None = None) -> None:
    if not hasattr(app, "processEvents"):
        return
    if max_ms is None:
        app.processEvents()
        return
    try:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, max_ms)
    except TypeError:
        app.processEvents()


def _is_splash_exposed(splash: object) -> bool:
    window_handle = getattr(splash, "windowHandle", None)
    if callable(window_handle):
        handle = window_handle()
        if handle is not None:
            is_exposed = getattr(handle, "isExposed", None)
            if callable(is_exposed):
                try:
                    return bool(is_exposed())
                except RuntimeError:
                    pass
    is_visible = getattr(splash, "isVisible", None)
    if callable(is_visible):
        try:
            return bool(is_visible())
        except RuntimeError:
            return False
    return True


def _wait_for_splash_exposed(app: QApplication, splash: object, *, timeout_ms: int = 1200) -> None:
    wait_clock = QElapsedTimer()
    wait_clock.start()
    while wait_clock.elapsed() < timeout_ms and not _is_splash_exposed(splash):
        _process_events(app, max_ms=25)
        QThread.msleep(10)
    _process_events(app, max_ms=10)


def run() -> int:
    """Start the Qt application."""
    minimum_splash_ms = 1600

    _set_windows_app_id()
    _force_x11_platform_on_wayland()
    app = QApplication(sys.argv)
    app.setApplicationName("MuFinder")
    app.setOrganizationName("MuFinder")
    app.setWindowIcon(_app_icon())
    _install_text_editing_shortcuts(app)

    splash = _splash_screen_ctor()()
    splash.show()
    if hasattr(splash, "raise_"):
        splash.raise_()
    if hasattr(splash, "activateWindow"):
        splash.activateWindow()
    _wait_for_splash_exposed(app, splash)

    splash_clock = QElapsedTimer()
    splash_clock.start()

    splash.set_progress(18, "Loading appearance...")
    settings = QSettings("MuFinder", "MuFinder")
    apply_theme, load_theme_mode = _theme_helpers()
    apply_theme(app, load_theme_mode(settings))
    _process_events(app, max_ms=10)

    splash.set_progress(35, "Loading CASTEP configuration...")
    _apply_saved_castep_environment(settings)
    _process_events(app, max_ms=10)

    splash.set_progress(70, "Building main window...")
    window = _main_window_ctor()()
    _process_events(app, max_ms=10)

    splash.set_progress(95, "Finalising UI...")
    window.show()
    _process_events(app, max_ms=10)

    should_delay = app.__class__.__module__.startswith("PySide6.")
    while should_delay and splash_clock.elapsed() < minimum_splash_ms:
        _process_events(app, max_ms=25)
        QThread.msleep(25)

    splash.set_progress(100, "Ready")
    _process_events(app, max_ms=10)
    splash.finish(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
