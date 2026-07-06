"""Screen-aware sizing helpers for top-level windows and dialogs.

Several MuFinder windows were originally sized with fixed pixel dimensions
chosen against a large development monitor (for example ``resize(1280, 840)``
or ``resize(1280, 860)``). On a 13" MacBook (1440x900 physical, ~850px of
*usable* height once the menu bar and Dock are subtracted) those windows open
taller than the available screen, forcing the user to manually resize or move
them before any content is visible. This module centralizes the fix: callers
route their default ``resize()`` calls and restored-geometry calls through
these helpers instead of consulting fixed pixel constants directly.
"""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import QApplication, QWidget


def _available_geometry(widget: QWidget):
    """Return the best-known available screen geometry for ``widget``.

    Prefers the screen the widget is (or will be) shown on. Falls back to the
    application's primary screen, and finally to ``None`` when no screen is
    available at all (for example in a headless test environment), letting
    callers fall back to the requested size unchanged.
    """

    screen: QScreen | None = widget.screen()
    if screen is None:
        app = QApplication.instance()
        if app is not None:
            screen = app.primaryScreen()
    if screen is None:
        return None
    return screen.availableGeometry()


def clamped_size(widget: QWidget, width: int, height: int, fraction: float = 0.9) -> QSize:
    """Return a size no larger than ``fraction`` of the widget's available screen.

    ``width``/``height`` are the window's preferred default size. The result
    is ``min(width, avail.width() * fraction)`` by ``min(height, avail.height()
    * fraction)``, so a window that would otherwise open larger than the
    laptop's usable desktop area (menu bar, Dock, taskbar excluded) is shrunk
    to fit while leaving smaller requests untouched. If no screen geometry can
    be determined (headless tests), the requested size is returned unchanged.
    """

    geometry = _available_geometry(widget)
    if geometry is None:
        return QSize(width, height)
    clamped_width = min(width, int(geometry.width() * fraction))
    clamped_height = min(height, int(geometry.height() * fraction))
    return QSize(clamped_width, clamped_height)


def clamp_to_screen(widget: QWidget) -> None:
    """Ensure ``widget``'s current frame geometry fits fully on its screen.

    Intended to run immediately after a persisted geometry restore (e.g. a
    ``QSettings``-backed ``restoreGeometry``/``setGeometry`` call). Geometry
    saved while the window lived on a large external monitor can otherwise
    restore off-screen, or larger than the display, when the same settings are
    loaded on a smaller laptop panel. This shrinks the widget to fit inside
    its screen's available geometry (never below its minimum size) and then
    translates it back fully on-screen.

    No-ops if no screen geometry can be determined.
    """

    geometry = _available_geometry(widget)
    if geometry is None:
        return

    # setGeometry()/move() operate on the content rectangle, not the frame
    # rectangle, so window-manager decorations (title bar, borders) must be
    # measured and re-applied on both axes independently: a title bar adds
    # height only above the content rect, while side borders are typically
    # thin and symmetric. Mixing a single top-level margin scalar into both
    # axes (as if they were interchangeable) leaves the frame off by the
    # decoration size on whichever side it was not measured against.
    frame = widget.frameGeometry()
    content = widget.geometry()
    left_margin = content.left() - frame.left()
    top_margin = content.top() - frame.top()
    right_margin = frame.right() - content.right()
    bottom_margin = frame.bottom() - content.bottom()
    extra_width = left_margin + right_margin
    extra_height = top_margin + bottom_margin

    target_width = min(frame.width(), geometry.width()) - extra_width
    target_height = min(frame.height(), geometry.height()) - extra_height
    target_width = max(target_width, widget.minimumWidth())
    target_height = max(target_height, widget.minimumHeight())

    frame_width = target_width + extra_width
    frame_height = target_height + extra_height
    frame_x = frame.x()
    frame_y = frame.y()
    frame_x = max(geometry.left(), min(frame_x, geometry.right() - frame_width + 1))
    frame_y = max(geometry.top(), min(frame_y, geometry.bottom() - frame_height + 1))

    widget.setGeometry(frame_x + left_margin, frame_y + top_margin, target_width, target_height)
