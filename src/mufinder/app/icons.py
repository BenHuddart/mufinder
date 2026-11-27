"""MuFinder's painted icons.

Each glyph is a short ``QPainter`` routine drawn on a small design grid: 16
units for the 16 px action, menu and tool-button icons, 22 units for the 22 px
project-home and help buttons. Straight strokes are centred on half units, so
at the size a glyph is shown at they sit on pixel centres instead of being
smeared across a pixel boundary by resampling.

The glyphs are served through :class:`_GlyphIconEngine`, a ``QIconEngine``
that paints at the size and device pixel ratio Qt asks for and reads the
palette when it paints. An icon is therefore sharp on every screen scale, and
an icon created before a theme change shows the new theme's colours the next
time it is painted, with no refresh code in the widgets that hold it.
"""

from __future__ import annotations

from collections.abc import Callable

from shiboken6 import isValid

from PySide6.QtCore import QLibraryInfo, QPointF, QRect, QRectF, QSize, Qt, QVersionNumber
from PySide6.QtGui import (
    QColor,
    QIcon,
    QIconEngine,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QApplication, QStyle, QStyleOption

_ICON_MAP: dict[str, QStyle.StandardPixmap] = {
    "new_project": QStyle.StandardPixmap.SP_FileDialogNewFolder,
    "open_project": QStyle.StandardPixmap.SP_DialogOpenButton,
    "project_home": QStyle.StandardPixmap.SP_DirHomeIcon,
    "save_project": QStyle.StandardPixmap.SP_DialogSaveButton,
    "refresh": QStyle.StandardPixmap.SP_BrowserReload,
    "cluster": QStyle.StandardPixmap.SP_DriveNetIcon,
    "settings": QStyle.StandardPixmap.SP_FileDialogDetailedView,
    "appearance_light": QStyle.StandardPixmap.SP_DialogYesButton,
    "appearance_dark": QStyle.StandardPixmap.SP_DialogNoButton,
    "appearance_system": QStyle.StandardPixmap.SP_BrowserReload,
    "help": QStyle.StandardPixmap.SP_DialogHelpButton,
    "host_structure": QStyle.StandardPixmap.SP_DirIcon,
    "convergence": QStyle.StandardPixmap.SP_MediaPlay,
    "kpoint": QStyle.StandardPixmap.SP_ArrowRight,
    "muon": QStyle.StandardPixmap.SP_DialogApplyButton,
}

# The ink pen of the 16-unit glyphs.
_STROKE = 1.25

# Qt 6.8 changed ``QIconEngine.scaledPixmap`` to receive the logical size;
# earlier releases pass the size already multiplied by the scale.
_SCALED_PIXMAP_TAKES_LOGICAL_SIZE = QLibraryInfo.version() >= QVersionNumber(6, 8)

# How many renderings one engine keeps (sizes x modes x states).
_PIXMAP_CACHE_LIMIT = 16

# Clones Qt asked for when it detached a shared icon. Qt owns and deletes each
# one, but PySide does not take the Python object over, so it is kept alive
# here until Qt has deleted it; otherwise Python would free it under Qt.
_LIVE_CLONES: list[QIconEngine] = []

ColourKey = tuple[int, ...]
DrawGlyph = Callable[[QPainter, tuple[QColor, ...]], None]


class _GlyphIconEngine(QIconEngine):
    """Paint one glyph at the requested size and scale in the current colours.

    ``colours`` returns the colours to paint with and is called at paint time,
    so a theme change reaches every existing icon. Renderings are cached per
    physical size, scale, mode, state and colour set; a change of colours makes the
    old renderings unreachable and they are dropped.
    """

    def __init__(
        self,
        draw: DrawGlyph,
        grid: float,
        colours: Callable[[], tuple[QColor, ...]],
    ) -> None:
        super().__init__()
        self._draw = draw
        self._grid = grid
        self._colours = colours
        self._cache: dict[tuple[int, int, float, int, int, ColourKey], QPixmap] = {}

    def clone(self) -> QIconEngine:
        _LIVE_CLONES[:] = [engine for engine in _LIVE_CLONES if isValid(engine)]
        engine = _GlyphIconEngine(self._draw, self._grid, self._colours)
        _LIVE_CLONES.append(engine)
        return engine

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        return self._render(size, mode, state, 1.0)

    def scaledPixmap(
        self, size: QSize, mode: QIcon.Mode, state: QIcon.State, scale: float
    ) -> QPixmap:
        logical = size if _SCALED_PIXMAP_TAKES_LOGICAL_SIZE else size / max(scale, 1.0)
        return self._render(logical, mode, state, max(scale, 1.0))

    def paint(self, painter: QPainter, rect: QRect, mode: QIcon.Mode, state: QIcon.State) -> None:
        scale = painter.device().devicePixelRatioF() if painter.device() is not None else 1.0
        rendered = self._render(rect.size(), mode, state, max(scale, 1.0))
        painter.drawPixmap(rect, rendered)

    def _render(self, size: QSize, mode: QIcon.Mode, state: QIcon.State, scale: float) -> QPixmap:
        width = max(1, round(size.width() * scale))
        height = max(1, round(size.height() * scale))
        colours = self._colours()
        key = (
            width,
            height,
            scale,
            int(mode.value),
            int(state.value),
            tuple(colour.rgba() for colour in colours),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        side = min(width, height)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.translate((width - side) / 2.0, (height - side) / 2.0)
        painter.scale(side / self._grid, side / self._grid)
        self._draw(painter, colours)
        painter.end()

        if mode != QIcon.Mode.Normal:
            pixmap = _generated_mode_pixmap(mode, pixmap)
        pixmap.setDevicePixelRatio(scale)

        if len(self._cache) >= _PIXMAP_CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = pixmap
        return pixmap


# One icon per glyph, shared by every widget that shows it: the engine reads the
# colours when it paints, so a shared icon follows the theme, and its rendering
# cache serves every widget.
_SHARED_ICONS: dict[str, QIcon] = {}


def _generated_mode_pixmap(mode: QIcon.Mode, pixmap: QPixmap) -> QPixmap:
    """Derive the disabled, active or selected look the way ``QIcon(pixmap)`` does."""

    app = QApplication.instance()
    if app is None:
        return pixmap
    option = QStyleOption()
    option.palette = QApplication.palette()
    return app.style().generatedIconPixmap(mode, pixmap, option)


def _palette_colours() -> tuple[QColor, ...]:
    """Ink, accent, muted and paper colours from the application palette."""

    palette = QApplication.palette()
    return (
        palette.buttonText().color(),
        palette.highlight().color(),
        palette.mid().color(),
        palette.base().color(),
    )


def _stage_colours() -> tuple[QColor, ...]:
    """Edge, fill, complete and pending colours from the MuFinder theme palette."""

    from mufinder.app.theme import current_palette

    palette = current_palette()
    return (
        QColor(palette.border),
        QColor(palette.bg_surface),
        QColor(palette.seal_gold),
        QColor(palette.text_muted),
    )


def _muted_text_colours() -> tuple[QColor, ...]:
    """The theme's muted text colour as ink, accent and muted, on the palette's paper."""

    from mufinder.app.theme import current_palette

    muted = QColor(current_palette().text_muted)
    return (muted, muted, muted, QApplication.palette().base().color())


def _ink_pen(ink: QColor, width: float = _STROKE) -> QPen:
    return QPen(
        ink, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
    )


def icon(name: str) -> QIcon:
    """Return an icon for the requested semantic name.

    MuFinder's own painted glyph is preferred whenever one exists, so action and
    tool-button icons look identical on every platform and follow the theme. The
    platform standard icon is used only as a fallback for names that do not
    have a MuFinder glyph.
    """

    app = QApplication.instance()
    if app is None:
        return QIcon()

    if name in _GLYPHS:
        cached = _SHARED_ICONS.get(name)
        if cached is None:
            grid, draw = _GLYPHS[name]
            colours = _GLYPH_COLOURS.get(name, _palette_colours)
            cached = QIcon(_GlyphIconEngine(_glyph_drawer(draw), grid, colours))
            _SHARED_ICONS[name] = cached
        return QIcon(cached)

    standard = _ICON_MAP.get(name)
    if standard is None:
        return QIcon()
    return app.style().standardIcon(standard)


#: The seal of each stage state, by the names ``workflow_stage_icon`` accepts.
_STAGE_SEALS: dict[str, str] = {
    "completed": "stage_completed",
    "complete": "stage_completed",
    "in_progress": "stage_under_way",
}


def workflow_stage_icon(state: str) -> QIcon:
    """Return the seal of a workflow stage's state.

    ``"completed"`` (or ``"complete"``) is the gold seal with a tick,
    ``"in_progress"`` the half-filled seal of a stage under way, and anything
    else (``"incomplete"``, ``"not_started"``) the dotted ring of a stage not
    started.
    """

    key = _STAGE_SEALS.get(state.strip().lower(), "stage_pending")
    cached = _SHARED_ICONS.get(key)
    if cached is None:
        draw = {
            "stage_completed": _draw_stage_completed,
            "stage_under_way": _draw_stage_under_way,
            "stage_pending": _draw_stage_pending,
        }[key]
        cached = QIcon(_GlyphIconEngine(draw, 16.0, _stage_colours))
        _SHARED_ICONS[key] = cached
    return QIcon(cached)


def job_state_icon(word: str) -> QIcon:
    """Return the glyph for a job status word, drawn in that state's colour.

    The Job Dashboard's summary shows it beside each count. Each state has its
    own shape (a play triangle for ``running``, a square for ``stopped``), so the
    glyph tells the states apart without its colour. A word with no glyph of its
    own gets a dot.
    """

    if QApplication.instance() is None:
        return QIcon()
    key = f"job_state:{word}"
    cached = _SHARED_ICONS.get(key)
    if cached is None:
        draw = _JOB_STATE_GLYPHS.get(word, _draw_job_state_other)

        def colours() -> tuple[QColor, ...]:
            from mufinder.app.theme import status_color

            return (status_color(word),)

        cached = QIcon(_GlyphIconEngine(draw, 16.0, colours))
        _SHARED_ICONS[key] = cached
    return QIcon(cached)


def _draw_job_state_running(painter: QPainter, colours: tuple[QColor, ...]) -> None:
    (ink,) = colours
    painter.setPen(_ink_pen(ink, 1.0))
    painter.setBrush(ink)
    painter.drawPath(_path((5.0, 3.5), (12.5, 8.0), (5.0, 12.5), close=True))


def _draw_job_state_queued(painter: QPainter, colours: tuple[QColor, ...]) -> None:
    """An hourglass: waiting for its turn."""

    (ink,) = colours
    painter.setPen(_ink_pen(ink, 1.25))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(_path((4.5, 2.5), (11.5, 2.5), (4.5, 13.5), (11.5, 13.5), close=True))


def _draw_job_state_submitting(painter: QPainter, colours: tuple[QColor, ...]) -> None:
    """An arrow up from a line: being handed over."""

    (ink,) = colours
    painter.setPen(_ink_pen(ink, 1.5))
    _line(painter, 3.5, 13.0, 12.5, 13.0)
    _line(painter, 8.0, 10.5, 8.0, 3.0)
    painter.drawPath(_path((4.75, 6.25), (8.0, 3.0), (11.25, 6.25)))


def _draw_job_state_retrieving(painter: QPainter, colours: tuple[QColor, ...]) -> None:
    """An arrow down onto a line: output coming back."""

    (ink,) = colours
    painter.setPen(_ink_pen(ink, 1.5))
    _line(painter, 3.5, 13.0, 12.5, 13.0)
    _line(painter, 8.0, 2.5, 8.0, 10.0)
    painter.drawPath(_path((4.75, 6.75), (8.0, 10.0), (11.25, 6.75)))


def _draw_job_state_failed(painter: QPainter, colours: tuple[QColor, ...]) -> None:
    """A cross in a circle."""

    (ink,) = colours
    painter.setPen(_ink_pen(ink, 1.3))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QRectF(2.5, 2.5, 11.0, 11.0))
    _line(painter, 5.75, 5.75, 10.25, 10.25)
    _line(painter, 10.25, 5.75, 5.75, 10.25)


def _draw_job_state_stopped(painter: QPainter, colours: tuple[QColor, ...]) -> None:
    (ink,) = colours
    painter.setPen(_ink_pen(ink, 1.0))
    painter.setBrush(ink)
    painter.drawRoundedRect(QRectF(4.0, 4.0, 8.0, 8.0), 1.0, 1.0)


def _draw_job_state_cancelled(painter: QPainter, colours: tuple[QColor, ...]) -> None:
    """A circle struck through."""

    (ink,) = colours
    painter.setPen(_ink_pen(ink, 1.3))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QRectF(2.5, 2.5, 11.0, 11.0))
    _line(painter, 4.5, 11.5, 11.5, 4.5)


def _draw_job_state_other(painter: QPainter, colours: tuple[QColor, ...]) -> None:
    (ink,) = colours
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(ink)
    painter.drawEllipse(QPointF(8.0, 8.0), 2.5, 2.5)


_JOB_STATE_GLYPHS: dict[str, DrawGlyph] = {
    "running": _draw_job_state_running,
    "queued": _draw_job_state_queued,
    "submitting": _draw_job_state_submitting,
    "retrieving": _draw_job_state_retrieving,
    "failed": _draw_job_state_failed,
    "stopped": _draw_job_state_stopped,
    "cancelled": _draw_job_state_cancelled,
}


def _glyph_drawer(
    draw: Callable[[QPainter, QColor, QColor, QColor, QColor], None],
) -> DrawGlyph:
    def paint(painter: QPainter, colours: tuple[QColor, ...]) -> None:
        ink, accent, muted, paper = colours
        painter.setPen(_ink_pen(ink))
        draw(painter, ink, accent, muted, paper)

    return paint


def _draw_stage_completed(painter: QPainter, colours: tuple[QColor, ...]) -> None:
    edge, fill, complete, _pending = colours
    painter.setPen(QPen(edge, 1.1))
    painter.setBrush(complete)
    painter.drawEllipse(QRectF(2, 2, 12, 12))
    painter.setPen(QPen(fill, 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    seal = QPainterPath()
    seal.moveTo(5.0, 8.0)
    seal.lineTo(7.0, 10.2)
    seal.lineTo(11.0, 5.8)
    painter.drawPath(seal)


def _draw_stage_under_way(painter: QPainter, colours: tuple[QColor, ...]) -> None:
    """A gold ring, its left half filled: the seal of a stage that has started."""

    _edge, fill, complete, _pending = colours
    disc = QRectF(2.5, 2.5, 11, 11)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(fill)
    painter.drawEllipse(disc)
    painter.setBrush(complete)
    # Qt angles are sixteenths of a degree, anticlockwise from three o'clock.
    painter.drawPie(disc, 90 * 16, 180 * 16)
    painter.setPen(QPen(complete, 1.3))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(disc)


def _draw_stage_pending(painter: QPainter, colours: tuple[QColor, ...]) -> None:
    _edge, fill, _complete, pending = colours
    painter.setPen(QPen(pending, 1.3, Qt.PenStyle.DotLine, Qt.PenCapStyle.RoundCap))
    painter.setBrush(fill)
    painter.drawEllipse(QRectF(3, 3, 10, 10))


def _path(*points: tuple[float, float], close: bool = False) -> QPainterPath:
    path = QPainterPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    if close:
        path.closeSubpath()
    return path


def _line(painter: QPainter, x1: float, y1: float, x2: float, y2: float) -> None:
    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))


def _draw_new_project_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    painter.setBrush(paper)
    painter.drawPath(
        _path((3.5, 2.5), (9.5, 2.5), (12.5, 5.5), (12.5, 13.5), (3.5, 13.5), close=True)
    )
    _line(painter, 9.5, 2.5, 9.5, 5.5)
    _line(painter, 9.5, 5.5, 12.5, 5.5)
    painter.setPen(QPen(accent, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    _line(painter, 8.0, 4.5, 8.0, 10.0)
    _line(painter, 5.25, 7.25, 10.75, 7.25)


def _draw_open_project_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    painter.setBrush(muted.lighter(140))
    painter.drawRoundedRect(QRectF(2.5, 5.5, 11.0, 7.0), 1.75, 1.75)
    painter.drawRoundedRect(QRectF(3.5, 3.5, 5.0, 3.0), 1.25, 1.25)


def _draw_save_project_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    painter.setBrush(ink.lighter(175))
    painter.drawRoundedRect(QRectF(2.5, 2.5, 11.0, 11.0), 1.5, 1.5)
    painter.setBrush(paper)
    painter.drawRect(QRectF(4.5, 4.0, 6.0, 3.0))
    painter.setBrush(accent)
    painter.drawRect(QRectF(5.5, 9.0, 5.0, 3.0))


def _draw_project_home_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    """A house with an accent roof, on the 22-unit grid of the 22 px home button."""

    painter.setPen(_ink_pen(ink, 1.5))
    painter.setBrush(paper)
    painter.drawPath(
        _path(
            (2.5, 10.5),
            (11.0, 3.5),
            (19.5, 10.5),
            (17.0, 10.5),
            (17.0, 18.5),
            (5.0, 18.5),
            (5.0, 10.5),
            close=True,
        )
    )

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(accent)
    painter.drawPath(
        _path(
            (3.5, 10.0),
            (11.0, 4.0),
            (18.5, 10.0),
            (16.25, 10.0),
            (11.0, 5.75),
            (5.75, 10.0),
            close=True,
        )
    )

    painter.setPen(_ink_pen(ink, 1.35))
    painter.setBrush(ink.lighter(170))
    painter.drawRoundedRect(QRectF(8.75, 12.5, 4.5, 6.0), 1.0, 1.0)


def _draw_journal_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    """A bound notebook with an accent ribbon and two ruled lines."""

    painter.setBrush(paper)
    painter.drawRoundedRect(QRectF(3.5, 2.5, 10.0, 11.0), 1.25, 1.25)
    _line(painter, 5.5, 2.5, 5.5, 13.5)
    painter.setPen(_ink_pen(muted, 1.0))
    _line(painter, 7.5, 6.0, 11.5, 6.0)
    _line(painter, 7.5, 8.5, 11.5, 8.5)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(accent)
    painter.drawPath(
        _path((9.5, 2.5), (11.5, 2.5), (11.5, 5.0), (10.5, 4.25), (9.5, 5.0), close=True)
    )


def _draw_save_workspace_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    painter.setPen(_ink_pen(ink, 1.1))
    painter.setBrush(paper)
    painter.drawRoundedRect(QRectF(4.0, 2.5, 9.0, 10.5), 1.5, 1.5)
    painter.setBrush(ink.lighter(180))
    painter.drawRoundedRect(QRectF(2.75, 3.75, 9.0, 10.5), 1.5, 1.5)
    painter.setBrush(paper)
    painter.drawRect(QRectF(4.5, 5.0, 5.75, 2.5))
    painter.setBrush(accent)
    painter.drawRoundedRect(QRectF(5.5, 9.0, 5.0, 3.5), 0.75, 0.75)
    painter.setPen(QPen(paper, 1.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    _line(painter, 8.0, 9.75, 8.0, 11.75)
    _line(painter, 7.0, 10.75, 9.0, 10.75)


def _draw_pin_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    painter.setBrush(accent.lighter(150))
    painter.drawPath(
        _path(
            (5.5, 2.5), (11.0, 2.5), (10.0, 6.0), (13.0, 9.5), (9.5, 10.5), (5.5, 6.5), close=True
        )
    )
    painter.setPen(QPen(ink, 1.35, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    _line(painter, 6.5, 9.5, 3.5, 12.5)
    _line(painter, 8.0, 10.5, 10.0, 13.5)


def _draw_pop_out_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    """A window corner with an arrow leaving it: "open in a separate window"."""

    painter.setBrush(Qt.BrushStyle.NoBrush)
    frame = QPainterPath()
    frame.moveTo(7.0, 3.5)
    frame.lineTo(4.5, 3.5)
    frame.quadTo(3.5, 3.5, 3.5, 4.5)
    frame.lineTo(3.5, 11.5)
    frame.quadTo(3.5, 12.5, 4.5, 12.5)
    frame.lineTo(11.5, 12.5)
    frame.quadTo(12.5, 12.5, 12.5, 11.5)
    frame.lineTo(12.5, 9.0)
    painter.drawPath(frame)
    painter.setPen(_ink_pen(accent, 1.4))
    _line(painter, 7.5, 8.5, 12.5, 3.5)
    _line(painter, 9.0, 3.5, 12.5, 3.5)
    _line(painter, 12.5, 3.5, 12.5, 7.0)


def _draw_delete_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    painter.setPen(_ink_pen(ink, 1.15))
    painter.setBrush(paper)
    painter.drawRoundedRect(QRectF(4.5, 5.5, 7.0, 8.0), 1.15, 1.15)
    _line(painter, 3.5, 4.5, 12.5, 4.5)
    _line(painter, 6.5, 3.0, 9.5, 3.0)
    _line(painter, 7.0, 7.5, 7.0, 11.5)
    _line(painter, 9.0, 7.5, 9.0, 11.5)


def _draw_dismiss_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    """A cross: take this off the list."""

    painter.setPen(_ink_pen(ink, 1.5))
    _line(painter, 4.5, 4.5, 11.5, 11.5)
    _line(painter, 11.5, 4.5, 4.5, 11.5)


def _draw_stale_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    """A clock: the status shown was read a while ago."""

    painter.setPen(_ink_pen(ink, 1.15))
    painter.setBrush(paper)
    painter.drawEllipse(QRectF(3.0, 3.0, 10.0, 10.0))
    _line(painter, 8.0, 5.0, 8.0, 8.0)
    _line(painter, 8.0, 8.0, 10.25, 9.25)


def _draw_chevron_up_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    """A chevron pointing up: open upwards."""

    painter.setPen(_ink_pen(ink, 1.5))
    painter.drawPath(_path((4.0, 10.0), (8.0, 6.0), (12.0, 10.0)))


def _draw_chevron_down_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    """A chevron pointing down: fold away downwards."""

    painter.setPen(_ink_pen(ink, 1.5))
    painter.drawPath(_path((4.0, 6.0), (8.0, 10.0), (12.0, 6.0)))


def _draw_monitor_stand(painter: QPainter) -> None:
    _line(painter, 6.0, 12.5, 10.0, 12.5)
    _line(painter, 8.0, 10.5, 8.0, 14.0)


def _draw_cluster_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    painter.setPen(QPen(ink, 1.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    _line(painter, 4.5, 5.5, 8.0, 8.0)
    _line(painter, 11.5, 5.5, 8.0, 8.0)
    _line(painter, 8.0, 8.0, 8.0, 11.0)
    painter.setBrush(paper)
    painter.drawEllipse(QRectF(2.5, 2.5, 3.5, 3.5))
    painter.drawEllipse(QRectF(10.0, 2.5, 3.5, 3.5))
    painter.setBrush(accent)
    painter.drawEllipse(QRectF(6.25, 9.75, 3.5, 3.5))


def _radial_spokes(inner: float, outer: float, diagonal_inner: float, diagonal_outer: float):
    """Eight spokes around the grid centre (8, 8): four straight, four diagonal."""

    c = 8.0
    return (
        (c, c - outer, c, c - inner),
        (c, c + inner, c, c + outer),
        (c - outer, c, c - inner, c),
        (c + inner, c, c + outer, c),
        (c - diagonal_outer, c - diagonal_outer, c - diagonal_inner, c - diagonal_inner),
        (c + diagonal_inner, c + diagonal_inner, c + diagonal_outer, c + diagonal_outer),
        (c + diagonal_inner, c - diagonal_inner, c + diagonal_outer, c - diagonal_outer),
        (c - diagonal_outer, c + diagonal_outer, c - diagonal_inner, c + diagonal_inner),
    )


def _draw_settings_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    for x1, y1, x2, y2 in _radial_spokes(3.5, 6.0, 2.8, 4.2):
        _line(painter, x1, y1, x2, y2)
    painter.setBrush(accent.lighter(150))
    painter.drawEllipse(QPointF(8.0, 8.0), 3.5, 3.5)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QPointF(8.0, 8.0), 1.6, 1.6)


def _draw_appearance_light_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    painter.setBrush(accent.lighter(135))
    painter.drawEllipse(QPointF(8.0, 8.0), 3.5, 3.5)
    for x1, y1, x2, y2 in _radial_spokes(4.5, 6.5, 3.75, 5.15):
        _line(painter, x1, y1, x2, y2)


def _draw_appearance_dark_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    painter.setBrush(ink)
    painter.drawEllipse(QRectF(3.5, 2.5, 9.0, 9.0))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(paper)
    painter.drawEllipse(QRectF(6.5, 2.0, 7.0, 9.5))


def _draw_appearance_system_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    painter.setBrush(paper)
    painter.drawRoundedRect(QRectF(2.5, 2.5, 11.0, 8.0), 1.4, 1.4)
    _draw_monitor_stand(painter)
    painter.setBrush(accent.lighter(150))
    painter.drawRect(QRectF(4.0, 4.0, 4.0, 5.0))
    painter.setBrush(ink.lighter(185))
    painter.drawRect(QRectF(8.0, 4.0, 4.0, 5.0))


def _draw_help_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    """An accent question mark, on the 22-unit grid of the 22 px help buttons."""

    painter.setPen(_ink_pen(accent, 2.45))
    question = QPainterPath()
    question.moveTo(7.4, 7.4)
    question.cubicTo(7.8, 5.1, 9.5, 4.0, 11.3, 4.2)
    question.cubicTo(13.5, 4.4, 14.9, 5.8, 14.9, 7.7)
    question.cubicTo(14.9, 9.2, 13.9, 10.1, 12.6, 10.9)
    question.cubicTo(11.4, 11.7, 11.1, 12.4, 11.1, 13.7)
    painter.drawPath(question)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(accent)
    painter.drawEllipse(QRectF(9.55, 17.0, 3.0, 3.0))


def _draw_quiet_help_icon(
    painter: QPainter, ink: QColor, accent: QColor, muted: QColor, paper: QColor
) -> None:
    """A thin ringed question mark in the muted text colour, for help beside a control.

    Drawn on the 16-unit grid to sit at body-text height, quieter than the
    accent question mark of the 22 px dialog help buttons.
    """

    painter.setPen(_ink_pen(ink, 1.1))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QRectF(1.5, 1.5, 13.0, 13.0))
    question = QPainterPath()
    question.moveTo(6.0, 6.3)
    question.cubicTo(6.2, 5.0, 7.1, 4.4, 8.1, 4.5)
    question.cubicTo(9.3, 4.6, 10.1, 5.4, 10.0, 6.4)
    question.cubicTo(9.95, 7.3, 9.3, 7.7, 8.7, 8.1)
    question.cubicTo(8.2, 8.45, 8.0, 8.8, 8.0, 9.6)
    painter.setPen(_ink_pen(ink, 1.3))
    painter.drawPath(question)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(ink)
    painter.drawEllipse(QRectF(7.25, 11.0, 1.5, 1.5))


# name -> (design grid, draw routine). The grid is the size the glyph is drawn
# for: 16 for icons shown at 16 px, 22 for the 22 px home and help buttons.
_GLYPHS: dict[str, tuple[float, Callable[[QPainter, QColor, QColor, QColor, QColor], None]]] = {
    "new_project": (16.0, _draw_new_project_icon),
    "open_project": (16.0, _draw_open_project_icon),
    "project_home": (22.0, _draw_project_home_icon),
    "save_project": (16.0, _draw_save_project_icon),
    "journal": (16.0, _draw_journal_icon),
    "save_workspace": (16.0, _draw_save_workspace_icon),
    "stale": (16.0, _draw_stale_icon),
    "cluster": (16.0, _draw_cluster_icon),
    "settings": (16.0, _draw_settings_icon),
    "appearance_light": (16.0, _draw_appearance_light_icon),
    "appearance_dark": (16.0, _draw_appearance_dark_icon),
    "appearance_system": (16.0, _draw_appearance_system_icon),
    "help": (22.0, _draw_help_icon),
    "help_quiet": (16.0, _draw_quiet_help_icon),
    "pin": (16.0, _draw_pin_icon),
    "delete": (16.0, _draw_delete_icon),
    "dismiss": (16.0, _draw_dismiss_icon),
    "pop_out": (16.0, _draw_pop_out_icon),
    "chevron_up": (16.0, _draw_chevron_up_icon),
    "chevron_down": (16.0, _draw_chevron_down_icon),
}

#: Glyphs painted in colours other than the palette's ink, accent, muted and paper.
_GLYPH_COLOURS: dict[str, Callable[[], tuple[QColor, ...]]] = {
    "help_quiet": _muted_text_colours,
}

GLYPH_NAMES: tuple[str, ...] = tuple(_GLYPHS)
"""Names that resolve to a MuFinder glyph rather than a platform icon."""
