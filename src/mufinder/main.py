"""Application entrypoint for MuFinder."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

from PySide6.QtCore import QEventLoop, QElapsedTimer, QSettings, QThread, QTimer
from PySide6.QtWidgets import QApplication

from mufinder.native_threads import cap_native_thread_pools


def _main_window_ctor():
    from mufinder.app.main_window import MainWindow

    return MainWindow


# The splash stays up at least this long, so it does not flash past on a fast start.
MINIMUM_SPLASH_MS = 800


def _splash_screen_ctor():
    from mufinder.app.splash_screen import MuFinderSplashScreen

    return MuFinderSplashScreen


def _theme_helpers():
    from mufinder.app.theme import apply_theme, load_theme_mode

    return apply_theme, load_theme_mode


def _install_text_editing_shortcuts(app: QApplication) -> None:
    from mufinder.app.text_editing import install_text_editing_shortcuts

    install_text_editing_shortcuts(app)


def _install_gui_garbage_collector(app: QApplication) -> None:
    from mufinder.app.gui_garbage_collector import install_gui_garbage_collector

    install_gui_garbage_collector(app)


def _app_icon():
    from mufinder.branding import app_icon

    return app_icon()


def _set_windows_app_id() -> None:
    """Set a stable Windows application ID for taskbar icon grouping."""

    if not sys.platform.startswith("win"):
        return

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "io.github.benhuddart.mufinder"
        )
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
    is_wayland = os.environ.get("XDG_SESSION_TYPE") == "wayland" or bool(
        os.environ.get("WAYLAND_DISPLAY")
    )
    if not is_wayland:
        return
    if not os.environ.get("DISPLAY"):
        return
    os.environ["QT_QPA_PLATFORM"] = "xcb"


def _apply_saved_castep_environment(settings: QSettings) -> None:
    from mufinder.castep_setup import apply_saved_castep_environment

    apply_saved_castep_environment(settings)


def _prompt_for_keyword_catalog(window: object) -> None:
    """Ask the main window to explain a missing CASTEP keyword catalog.

    Deferred to the first turn of the event loop so the prompt appears over a
    fully shown window rather than while the splash screen is closing.
    """

    prompt = getattr(window, "prompt_for_keyword_catalog_if_needed", None)
    if callable(prompt):
        QTimer.singleShot(0, prompt)


# The --page values: the fixed routes of the project navigator.
PAGE_ROUTES = (
    "host_structure",
    "supercell_editor",
    "convergence",
    "convergence_cutoff",
    "convergence_kpoint",
    "pristine_relaxation",
    "uep",
    "muon_sites",
    "muon_candidates",
    "muon_relaxations",
    "muon_analysis",
    "hyperfine_coupling",
    "magnetic_fields",
    "polarisation",
    "quantum_corrections",
    "energy_barriers",
    "provenance_log",
)


def _parse_arguments(arguments: list[str], *, before_qt: bool = False) -> argparse.Namespace:
    """Parse the command line left over once Qt has taken its own options.

    With ``before_qt`` the raw command line is checked before any GUI starts,
    so ``--help`` and usage errors work without a display. Qt's own options
    are still present then, so unknown options are tolerated and the parse
    after ``QApplication`` strips them is the one that counts.
    """

    parser = argparse.ArgumentParser(
        prog="mufinder",
        description="MuFinder: muon stopping-site workflows with CASTEP.",
    )
    parser.add_argument(
        "project",
        nargs="?",
        type=Path,
        help="a .mufinder project folder (or its project.json) to open at startup",
    )
    parser.add_argument(
        "--page",
        metavar="PAGE",
        help="workflow page to show once the project is open, by route key: "
        + ", ".join(PAGE_ROUTES),
    )
    if before_qt:
        options, _qt_options = parser.parse_known_args(arguments)
    else:
        options = parser.parse_args(arguments)
    if options.page is not None and options.project is None:
        parser.error("--page needs a project to open")
    return options


def _open_project_on_start(window: object, project: Path | None, page: str | None) -> None:
    """Open the command-line project once the event loop is running.

    Deferred like the keyword-catalog prompt, so the project loads (and any
    error dialog appears) over the shown main window rather than during startup.
    """

    if project is None:
        return

    def open_requested() -> None:
        if window.open_project(project) and page is not None:
            window.show_page(page)

    QTimer.singleShot(0, open_requested)


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

    # Before anything imports NumPy: the BLAS and OpenMP pools read their size once,
    # when they load. Setting variables is cheap, so the splash rule holds.
    cap_native_thread_pools()
    _parse_arguments(sys.argv[1:], before_qt=True)
    _set_windows_app_id()
    _force_x11_platform_on_wayland()
    app = QApplication(sys.argv)
    # QApplication strips Qt's own options (``-platform xcb`` ...) from arguments().
    options = _parse_arguments(app.arguments()[1:])
    # Reference cycles are freed on this thread only; see gui_garbage_collector.
    _install_gui_garbage_collector(app)
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

    splash.set_progress(18, "Loading appearance…")
    settings = QSettings("MuFinder", "MuFinder")
    apply_theme, load_theme_mode = _theme_helpers()
    apply_theme(app, load_theme_mode(settings))
    _process_events(app, max_ms=10)

    splash.set_progress(35, "Loading CASTEP configuration…")
    _apply_saved_castep_environment(settings)
    _process_events(app, max_ms=10)

    splash.set_progress(70, "Building main window…")
    window = _main_window_ctor()()
    _process_events(app, max_ms=10)

    splash.set_progress(95, "Finalising UI…")
    window.show()
    _process_events(app, max_ms=10)

    should_delay = app.__class__.__module__.startswith("PySide6.")
    while should_delay and splash_clock.elapsed() < MINIMUM_SPLASH_MS:
        _process_events(app, max_ms=25)
        QThread.msleep(25)

    splash.set_progress(100, "Ready")
    _process_events(app, max_ms=10)
    splash.finish(window)
    _open_project_on_start(window, options.project, options.page)
    _prompt_for_keyword_catalog(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
