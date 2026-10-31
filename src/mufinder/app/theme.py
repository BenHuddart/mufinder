"""Centralized theme and palette helpers for MuFinder UI."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
import logging
from pathlib import Path
import sys
import tempfile
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QEvent, QMargins, QObject, QRect, QRectF, QSettings, Qt
from PySide6.QtGui import (
    QColor,
    QBrush,
    QFont,
    QFontDatabase,
    QGuiApplication,
    QIcon,
    QPainter,
    QPalette,
    QPen,
    QRegion,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QRadioButton,
    QScrollBar,
    QStyle,
    QStyleOptionButton,
    QTabBar,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QWidget,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.legend import Legend

THEME_LIGHT = "light"
THEME_DARK = "dark"
THEME_SYSTEM = "system"
THEME_MODE_KEY = "ui/theme_mode"

# Base Qt style shared across platforms. Fusion renders identically on macOS,
# Windows, and Linux, giving the stylesheet one predictable baseline to sit on
# instead of each OS's native style (which leaks through on unstyled controls).
_BASE_STYLE = "Fusion"

# Bundled families MuFinder ships so typography is identical on every OS rather
# than depending on whatever fonts the host happens to have installed.
_BUNDLED_BODY_FAMILY = "Inter"
_BUNDLED_HEADING_FAMILY = "Fraunces"


def _bundled_font_dir() -> Path:
    """Return the directory holding MuFinder's bundled font files."""

    return Path(str(files("mufinder.assets").joinpath("fonts")))


def ensure_fonts_loaded() -> None:
    """Register MuFinder's bundled fonts with the Qt font database.

    Idempotent and safe to call repeatedly; it is a no-op until a Qt
    application object exists (the font database requires one), and a no-op
    once the bundled fonts have been registered. Loading the fonts ourselves
    guarantees the Inter/Fraunces families are present on every platform, so
    the UI never silently falls back to host system fonts.

    The bundled files are always registered on first run rather than skipped
    when a host font happens to share the family name: that keeps the exact,
    pinned weights MuFinder ships in control of the look, and avoids enumerating
    the whole font database on every call from the hot ``*_font_family`` paths.
    """

    global _FONTS_LOADED

    if _FONTS_LOADED or QGuiApplication.instance() is None:
        return

    font_dir = _bundled_font_dir()
    font_paths = sorted(font_dir.glob("*.ttf")) if font_dir.is_dir() else []
    if not font_paths:
        # A missing fonts directory means a broken packaged build (the
        # installer dropped mufinder/assets/fonts): the UI silently falls
        # back to host fonts, so make the regression visible in the log.
        logging.getLogger(__name__).warning(
            "Bundled fonts not found at %s; falling back to system fonts. "
            "Packaged builds must ship mufinder/assets/fonts.",
            font_dir,
        )
        return
    for font_path in font_paths:
        QFontDatabase.addApplicationFont(str(font_path))
    _FONTS_LOADED = True


def _first_available_font(candidates: tuple[str, ...], fallback: str | None = None) -> str:
    available = {family.casefold(): family for family in QFontDatabase.families()}
    for candidate in candidates:
        matched = available.get(candidate.casefold())
        if matched is not None:
            return matched

    if fallback is not None:
        matched = available.get(fallback.casefold())
        if matched is not None:
            return matched

    app = QGuiApplication.instance()
    if app is not None and app.font().family().strip():
        return app.font().family().strip()

    families = QFontDatabase.families()
    if families:
        return families[0]

    return "Helvetica"


def body_font_family() -> str:
    """Return preferred body font family available on this system."""

    ensure_fonts_loaded()
    return _first_available_font(
        (_BUNDLED_BODY_FAMILY, "Helvetica Neue", "Arial"), fallback="Helvetica"
    )


def heading_font_family() -> str:
    """Return preferred heading font family available on this system."""

    ensure_fonts_loaded()
    return _first_available_font(
        (_BUNDLED_HEADING_FAMILY, "Playfair Display", "Libre Baskerville", "Georgia", "Serif"),
        fallback=body_font_family(),
    )


def adjust_font_size(font: QFont, delta: int, *, minimum: int = 1) -> None:
    """Adjust ``font``'s size by ``delta`` in whichever unit it is sized.

    The application default font is sized in pixels (see ``apply_theme``), so
    ``pointSize()`` returns -1 for fonts derived from it and naive
    ``setPointSize(pointSize() + delta)`` arithmetic silently stops applying
    (or worse, clamps to an absolute point size). This helper applies the
    delta to the pixel or point size, whichever the font actually carries.
    """

    pixel = font.pixelSize()
    if pixel > 0:
        font.setPixelSize(max(minimum, pixel + delta))
        return
    point = font.pointSize()
    if point > 0:
        font.setPointSize(max(minimum, point + delta))


@dataclass(frozen=True, slots=True)
class TypeScale:
    """Font sizes in pixels, smallest first."""

    caption: int = 11
    body: int = 13
    emphasis: int = 15
    section: int = 18
    page: int = 28


@dataclass(frozen=True, slots=True)
class SpacingScale:
    """Padding, margins and gaps in pixels, smallest first."""

    xs: int = 4
    sm: int = 8
    md: int = 12
    lg: int = 16
    xl: int = 24


@dataclass(frozen=True, slots=True)
class RadiusScale:
    """Corner radii in pixels. A square corner is ``0px``."""

    control: int = 6
    card: int = 10


@dataclass(frozen=True, slots=True)
class RoundShapeRadii:
    """Radii that draw a round shape: half its side, set by its size and not a scale step."""

    journal_stamp: int = 5
    slider_groove: int = 7
    radio_indicator: int = 9
    help_button: int = 12
    slider_handle: int = 13


TYPE_SCALE = TypeScale()
SPACING = SpacingScale()
RADIUS = RadiusScale()
ROUND_RADIUS = RoundShapeRadii()

# The room a combo box and a spin box keep for their arrow buttons (24 and 22 px wide).
_COMBO_ARROW_GUTTER = 28
_SPIN_ARROW_GUTTER = 30

# A standard push button's height in pixels, so its click target does not depend
# on the padding step. Larger text grows it past this.
_STANDARD_BUTTON_HEIGHT = 28
_FOCUS_RING_WIDTH = 2
_FOCUS_RING_NAME = "mufinderFocusRing"

# The general font each platform's desktop ships with, in pixels: 13 pt at
# 72 dpi on macOS, 9 pt at 96 dpi on Windows, and 11 pt at 96 dpi on GNOME, the
# largest of the common Linux defaults, so no default Linux desktop is scaled.
_DEFAULT_SYSTEM_TEXT_PX = {"darwin": 13.0, "win32": 12.0}
_DEFAULT_SYSTEM_TEXT_PX_ELSEWHERE = 44.0 / 3.0
_MAX_TEXT_SCALE = 2.0
_CURRENT_TYPE_SCALE = TYPE_SCALE


def text_scale() -> float:
    """Return how far the desktop's text size is above the usual default, from 1 to 2.

    A smaller desktop font leaves MuFinder at its designed sizes, so the type
    scale's smallest step stays readable and a default desktop renders the same
    on every platform.
    """

    if not isinstance(QGuiApplication.instance(), QGuiApplication):
        return 1.0
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    pixels = float(font.pixelSize())
    if pixels <= 0:
        screen = QGuiApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch() if screen is not None else 96.0
        pixels = font.pointSizeF() * dpi / 72.0
    default = _DEFAULT_SYSTEM_TEXT_PX.get(sys.platform, _DEFAULT_SYSTEM_TEXT_PX_ELSEWHERE)
    return max(1.0, min(_MAX_TEXT_SCALE, pixels / default))


def type_scale() -> TypeScale:
    """Return the type scale in use, fixed when a theme is applied (see ``text_scale``)."""

    return _CURRENT_TYPE_SCALE


def scaled_to_text(pixels: int) -> int:
    """Return a width set for the designed text size, at the type scale in use."""

    return round(pixels * type_scale().caption / TYPE_SCALE.caption)


def _scaled_type_scale(factor: float) -> TypeScale:
    return TypeScale(
        caption=round(TYPE_SCALE.caption * factor),
        body=round(TYPE_SCALE.body * factor),
        emphasis=round(TYPE_SCALE.emphasis * factor),
        section=round(TYPE_SCALE.section * factor),
        page=round(TYPE_SCALE.page * factor),
    )


def page_margins() -> QMargins:
    """Return the margins of a page or a dialog's content."""

    return QMargins(SPACING.md, SPACING.md, SPACING.md, SPACING.md)


def card_margins() -> QMargins:
    """Return the margins inside a card or a group of controls."""

    return QMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)


def form_spacing() -> int:
    """Return the gap between the rows and controls of a form."""

    return SPACING.sm


def _arrow_asset_path(direction: str, colour: str) -> str:
    """Return a temporary SVG asset path for a themed arrow glyph.

    The path is returned with forward slashes: it is interpolated into
    stylesheet ``url("…")`` values, where Qt's CSS parser treats a backslash
    as an escape character. Windows-native separators would not only break
    the image reference but corrupt parsing of every rule that follows it.
    """

    normalized = colour.lower().replace("#", "")
    asset_dir = Path(tempfile.gettempdir()) / "mufinder-theme-assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    filename = asset_dir / f"arrow-{direction}-{normalized}.svg"
    if not filename.exists():
        if direction == "down":
            points = "2,3 10,3 6,8"
        elif direction == "right":
            points = "3,2 3,10 8,6"
        elif direction == "left":
            points = "9,2 9,10 4,6"
        else:
            points = "2,8 10,8 6,3"
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
            f'<polygon points="{points}" fill="{colour}"/>'
            "</svg>"
        )
        filename.write_text(svg, encoding="utf-8")
    return filename.as_posix()


def arrow_icon(direction: str) -> QIcon:
    """Return a themed arrow ``QIcon`` for ``direction`` in the current palette.

    ``direction`` is one of ``"up"``, ``"down"``, ``"left"``, or ``"right"``.
    Use this instead of unicode glyphs such as ``◀``/``▶`` for stepper
    buttons: those glyphs fall outside the bundled Inter font's coverage and
    render from a mismatched platform fallback font (Segoe UI Symbol on
    Windows, Apple Symbols on macOS), producing inconsistent size/alignment
    across platforms.
    """

    colour = current_palette().text_primary
    return QIcon(_arrow_asset_path(direction, colour))


def _indicator_glyph_path(kind: str, colour: str) -> str:
    """Return a temporary SVG asset path for a checkbox/radio indicator glyph.

    ``kind`` is one of ``check`` (tick mark), ``dot`` (filled radio dot), or
    ``dash`` (indeterminate checkbox). Styling a Qt ``::indicator`` via the
    stylesheet replaces the native glyph, so we supply our own here to keep the
    indicators identical on every platform and tinted to the MuFinder palette.
    Returned with forward slashes for the same ``url("…")`` escaping reason as
    :func:`_arrow_asset_path`.
    """

    normalized = colour.lower().replace("#", "")
    asset_dir = Path(tempfile.gettempdir()) / "mufinder-theme-assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    filename = asset_dir / f"indicator-{kind}-{normalized}.svg"
    if not filename.exists():
        if kind == "check":
            body = (
                f'<path d="M4 8.5 L7 11.2 L12.4 5" fill="none" stroke="{colour}" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        elif kind == "dash":
            body = (
                f'<line x1="4.5" y1="8" x2="11.5" y2="8" stroke="{colour}" '
                'stroke-width="2" stroke-linecap="round"/>'
            )
        else:  # dot
            body = f'<circle cx="8" cy="8" r="4" fill="{colour}"/>'
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">'
            f"{body}</svg>"
        )
        filename.write_text(svg, encoding="utf-8")
    return filename.as_posix()


@dataclass(frozen=True, slots=True)
class MuFinderPalette:
    """Named UI colors for a single MuFinder appearance mode."""

    mode: str
    bg_primary: str
    bg_surface: str
    bg_surface_alt: str
    border: str
    text_primary: str
    text_muted: str
    accent: str
    accent_hover: str
    accent_pressed: str
    seal_gold: str
    success: str
    warning: str
    danger: str
    status_ready: str
    status_defined: str
    status_submitting: str
    status_queued: str
    status_running: str
    status_retrieving: str
    status_stopped: str
    status_completed: str
    status_failed: str
    status_cancelled: str


def light_palette() -> MuFinderPalette:
    """Return the default light-mode palette."""

    return MuFinderPalette(
        mode=THEME_LIGHT,
        bg_primary="#F5EDE0",
        bg_surface="#FDF6EC",
        bg_surface_alt="#EDE0CE",
        border="#C4AD8F",
        text_primary="#2C2418",
        text_muted="#7A6B59",
        accent="#E02020",
        accent_hover="#C61D1D",
        accent_pressed="#A71919",
        seal_gold="#B8860B",
        success="#146C43",
        warning="#A16207",
        danger="#B91C1C",
        status_ready="#0369A1",
        status_defined="#4B5563",
        status_submitting="#2563EB",
        status_queued="#A16207",
        status_running="#2563EB",
        status_retrieving="#7C3AED",
        status_stopped="#B45309",
        status_completed="#15803D",
        status_failed="#B91C1C",
        status_cancelled="#6B7280",
    )


def dark_palette() -> MuFinderPalette:
    """Return the default dark-mode palette."""

    return MuFinderPalette(
        mode=THEME_DARK,
        bg_primary="#0D1B2A",
        bg_surface="#1B2838",
        bg_surface_alt="#243447",
        border="#3D5068",
        text_primary="#F0E6D3",
        text_muted="#A09484",
        accent="#E8A838",
        accent_hover="#D49530",
        accent_pressed="#C08528",
        seal_gold="#D4A853",
        success="#34D399",
        warning="#F59E0B",
        danger="#F87171",
        status_ready="#7DD3FC",
        status_defined="#A5B4FC",
        status_submitting="#60A5FA",
        status_queued="#FBBF24",
        status_running="#60A5FA",
        status_retrieving="#C4B5FD",
        status_stopped="#FBBF24",
        status_completed="#4ADE80",
        status_failed="#F87171",
        status_cancelled="#9CA3AF",
    )


_CURRENT_PALETTE: MuFinderPalette = light_palette()
_TABLE_CORNER_STYLER: "_TableCornerStyler | None" = None
_FOCUS_RING_TRACKER: "_FocusRingTracker | None" = None
_BASE_STYLE_APP: "QApplication | None" = None
_FONTS_LOADED: bool = False


def _repolish_widget(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def refresh_widget_style(widget: QWidget) -> None:
    """Refresh stylesheet-dependent state after dynamic properties change."""

    _repolish_widget(widget)


class _TableCornerStyler(QObject):
    """Toggle item-view corner styling based on scrollbar/header visibility.

    The styler watches only the item views attached to it (and their vertical
    scroll bars), never the whole application: an application-wide Python event
    filter runs for every Qt event on every object and dominated the cost of
    building the main window. Views are attached where they are created, through
    ``track_item_view_corners`` (called by the shared table helpers), and
    ``attach_existing_tables`` sweeps up any view that predates the styler.
    """

    _TABLE_WATCH_EVENTS = {
        QEvent.Type.Show,
        QEvent.Type.Resize,
        QEvent.Type.LayoutRequest,
        QEvent.Type.StyleChange,
    }
    _BAR_WATCH_EVENTS = {
        QEvent.Type.Show,
        QEvent.Type.Hide,
        QEvent.Type.Resize,
    }
    _WATCH_EVENTS = _TABLE_WATCH_EVENTS | _BAR_WATCH_EVENTS

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)

    @staticmethod
    def _is_supported_item_view(widget: object) -> bool:
        try:
            return isinstance(widget, (QTableWidget, QTreeWidget))
        except (RuntimeError, SystemError):
            return False

    @staticmethod
    def _is_scrollbar(widget: object) -> bool:
        try:
            return isinstance(widget, QScrollBar)
        except (RuntimeError, SystemError):
            return False

    def attach_existing_tables(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        for widget in app.allWidgets():
            if self._is_supported_item_view(widget):
                self.attach_table(widget)

    def attach_table(self, table: QWidget) -> None:
        if bool(table.property("_cornerStylerAttached")):
            return

        table.setProperty("_cornerStylerAttached", True)
        table.installEventFilter(self)

        bar = table.verticalScrollBar()
        bar.installEventFilter(self)
        bar.rangeChanged.connect(self._on_scrollbar_range_changed)

        self._sync_table_corner_property(table)

    def _table_for_bar(self, bar: object) -> QWidget | None:
        # Walk up from the scroll bar to its owning item view rather than caching
        # the (bar -> table) pair in a dict. Caching Qt wrappers keeps them alive
        # past their C++ objects' lifetime, and releasing such a stale wrapper
        # during QWidget teardown segfaults shiboken (notably on Linux).
        try:
            widget = bar.parentWidget()
            while widget is not None:
                if self._is_supported_item_view(widget):
                    return widget
                widget = widget.parentWidget()
        except (RuntimeError, SystemError):
            return None
        return None

    def _on_scrollbar_range_changed(self, _minimum: int, _maximum: int) -> None:
        try:
            sender = self.sender()
            if not self._is_scrollbar(sender):
                return
            table = self._table_for_bar(sender)
            if table is None:
                return
            self._sync_table_corner_property(table)
        except (RuntimeError, SystemError):
            return

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # Only attached views and their scroll bars reach this filter, but they
        # still see every paint and mouse event, so reject on type first.
        event_type = event.type()
        if event_type not in self._WATCH_EVENTS:
            return False
        try:
            if self._is_supported_item_view(watched):
                if event_type in self._TABLE_WATCH_EVENTS:
                    self._sync_table_corner_property(watched)
            elif self._is_scrollbar(watched):
                if event_type in self._BAR_WATCH_EVENTS:
                    table = self._table_for_bar(watched)
                    if table is not None:
                        self._sync_table_corner_property(table)
        except (RuntimeError, SystemError):
            return False
        return False

    def _sync_table_corner_property(self, table: QWidget) -> None:
        has_scrollbar = self._has_vertical_scrollbar(table)
        scrollbar_value = "true" if has_scrollbar else "false"
        has_visible_vertical_header = self._has_visible_vertical_header(table)
        vertical_header_value = "true" if has_visible_vertical_header else "false"
        header = self._horizontal_header(table)

        if table.property("hasVerticalScrollbar") != scrollbar_value:
            table.setProperty("hasVerticalScrollbar", scrollbar_value)
            _repolish_widget(table)
            if header is not None:
                _repolish_widget(header)
            _repolish_widget(table.verticalScrollBar())

        if table.property("hasVisibleVerticalHeader") != vertical_header_value:
            table.setProperty("hasVisibleVerticalHeader", vertical_header_value)
            _repolish_widget(table)
            if header is not None:
                _repolish_widget(header)
            _repolish_widget(table.verticalScrollBar())

    def shutdown(self) -> None:
        """Detach event filters and signal hooks before app teardown."""

        app = QApplication.instance()
        if app is None:
            return

        for widget in app.allWidgets():
            try:
                if self._is_supported_item_view(widget):
                    widget.removeEventFilter(self)
                    widget.setProperty("_cornerStylerAttached", False)
                elif self._is_scrollbar(widget):
                    widget.removeEventFilter(self)
                    try:
                        widget.rangeChanged.disconnect(self._on_scrollbar_range_changed)
                    except (RuntimeError, SystemError, TypeError):
                        pass
            except (RuntimeError, SystemError):
                pass

    @staticmethod
    def _has_vertical_scrollbar(table: QWidget) -> bool:
        policy = table.verticalScrollBarPolicy()
        if policy == Qt.ScrollBarPolicy.ScrollBarAlwaysOff:
            return False
        if policy == Qt.ScrollBarPolicy.ScrollBarAlwaysOn:
            return True

        bar = table.verticalScrollBar()
        return bar.isVisible() or bar.maximum() > bar.minimum()

    @staticmethod
    def _has_visible_vertical_header(table: QWidget) -> bool:
        if isinstance(table, QTableWidget):
            return table.verticalHeader().isVisible()
        return False

    @staticmethod
    def _horizontal_header(table: QWidget):
        if isinstance(table, QTableWidget):
            return table.horizontalHeader()
        if isinstance(table, QTreeWidget):
            return table.header()
        return None


def _ensure_table_corner_styler(app: QApplication) -> None:
    global _TABLE_CORNER_STYLER

    if _TABLE_CORNER_STYLER is None or _TABLE_CORNER_STYLER.parent() is not app:
        _TABLE_CORNER_STYLER = _TableCornerStyler(app)

    _TABLE_CORNER_STYLER.attach_existing_tables()


# The control kinds that show the keyboard focus ring.
_FOCUS_RING_CONTROLS = (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QTabBar,
    QTextEdit,
)
# Events on a ringed control that can change where its ring belongs.
_FOCUS_RING_LAYOUT_EVENTS = {
    QEvent.Type.Resize,
    QEvent.Type.StyleChange,
    QEvent.Type.FontChange,
    QEvent.Type.LayoutDirectionChange,
}


class _FocusRing(QWidget):
    """The accent ring painted over the control that has keyboard focus.

    A child widget that the layout does not know about, drawn inside its control's
    bounds on top of the control's own border. It therefore cannot change a size,
    move a neighbour or be clipped by a table row, and no stylesheet, including a
    feature-local one, can override it. The colour is read at paint time, so the
    ring follows a theme change.
    """

    def __init__(self, control: QWidget, tracker: "_FocusRingTracker") -> None:
        super().__init__(control)
        self._tracker = tracker
        self.setObjectName(_FOCUS_RING_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # The ring is an event filter only while it exists, and only on its own
        # control, so this costs nothing on any other widget. A click usually lands
        # on a child: a view's viewport or a spin box's line edit.
        control.installEventFilter(self)
        for child in control.findChildren(
            QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly
        ):
            if child is not self:
                child.installEventFilter(self)
        if isinstance(control, QTabBar):
            control.currentChanged.connect(self._follow_control)
        self._follow_control()
        self.show()

    def _indicator_rect(self, control: QCheckBox | QRadioButton) -> QRect:
        element = (
            QStyle.SubElement.SE_RadioButtonIndicator
            if isinstance(control, QRadioButton)
            else QStyle.SubElement.SE_CheckBoxIndicator
        )
        option = QStyleOptionButton()
        option.initFrom(control)
        return control.style().subElementRect(element, option, control)

    def _follow_control(self, *_args: object) -> None:
        control = self.parentWidget()
        if isinstance(control, (QCheckBox, QRadioButton)):
            rect = self._indicator_rect(control)
        elif isinstance(control, QTabBar):
            rect = control.tabRect(control.currentIndex())
        else:
            rect = control.rect()
        if rect == self.geometry():
            return
        self.setGeometry(rect)
        self.raise_()
        if isinstance(control, QAbstractScrollArea):
            # Mask to the ring's band, so the ring does not overlap the whole of a
            # scrolling viewport and cost it Qt's scrolling optimisation.
            whole = QRect(0, 0, rect.width(), rect.height())
            band = _FOCUS_RING_WIDTH + 1  # The rounded corners reach a pixel further in.
            self.setMask(QRegion(whole) - QRegion(whole.adjusted(band, band, -band, -band)))

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonPress:
            # A click on the control that already has focus is not keyboard use.
            self._tracker.note_mouse_use()
            self.hide()
        elif event_type in _FOCUS_RING_LAYOUT_EVENTS:
            if watched is self.parentWidget():
                self._follow_control()
        elif event_type == QEvent.Type.Paint and isinstance(watched, QTabBar):
            # A tab bar has no signal for its tabs moving: a new tab, a longer
            # title, a scroll. It repaints for all of them.
            self._follow_control()
        return False

    def paintEvent(self, _event: object) -> None:
        control = self.parentWidget()
        palette = current_palette()
        inset = _FOCUS_RING_WIDTH / 2
        ring = QRectF(self.rect()).adjusted(inset, inset, -inset, -inset)
        round_shape = isinstance(control, QRadioButton)
        radius = ring.width() / 2 if round_shape else max(0.0, RADIUS.control - inset)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if isinstance(control, (QCheckBox, QRadioButton)):
            # A checked indicator is filled with the accent colour, so a surface-
            # coloured line keeps the ring distinct from the fill.
            gap = inset + 0.5
            inner = ring.adjusted(gap, gap, -gap, -gap)
            inner_radius = inner.width() / 2 if round_shape else radius
            painter.setPen(QPen(QColor(palette.bg_surface), 1))
            painter.drawRoundedRect(inner, inner_radius, inner_radius)
        painter.setPen(QPen(QColor(palette.accent), _FOCUS_RING_WIDTH))
        painter.drawRoundedRect(ring, radius, radius)
        painter.end()


def _remove_focus_ring(control: QWidget) -> None:
    for ring in control.findChildren(_FocusRing, options=Qt.FindChildOption.FindDirectChildrenOnly):
        ring.hide()
        ring.deleteLater()


class _FocusRingTracker(QObject):
    """Put a ``_FocusRing`` on a control when focus reaches it during keyboard use.

    Qt marks a window with ``WA_KeyboardFocusChange`` when focus moves by Tab or a
    shortcut and clears the mark on a mouse press and when the window is deactivated.
    The tracker remembers the last input kind across that second case, so focus that
    returns from a dialog or popup, or that a handler moves after a key press, still
    shows. Listening to ``focusChanged`` needs no application-wide Python event filter
    (see ``_TableCornerStyler``), and the tracker keeps no references to widgets: the
    ring lives on its control.
    """

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        self._keyboard_in_use = False
        app.focusChanged.connect(self._on_focus_changed)

    def note_mouse_use(self) -> None:
        self._keyboard_in_use = False

    def _on_focus_changed(self, old: QWidget | None, now: QWidget | None) -> None:
        try:
            if old is not None:
                _remove_focus_ring(old)
            if now is None:
                return
            window = now.window()
            if window.testAttribute(Qt.WidgetAttribute.WA_KeyboardFocusChange):
                self._keyboard_in_use = True
            elif QGuiApplication.mouseButtons() != Qt.MouseButton.NoButton:
                self._keyboard_in_use = False
            if (
                self._keyboard_in_use
                and isinstance(now, _FOCUS_RING_CONTROLS)
                # A combo box's open list, whose current row already shows.
                and window.windowType() != Qt.WindowType.Popup
            ):
                _FocusRing(now, self)
        except (RuntimeError, SystemError):
            return

    def shutdown(self, app: QApplication) -> None:
        try:
            app.focusChanged.disconnect(self._on_focus_changed)
            focused = app.focusWidget()
            if focused is not None:
                _remove_focus_ring(focused)
        except (RuntimeError, SystemError, TypeError):
            pass


def _ensure_focus_ring_tracker(app: QApplication) -> None:
    global _FOCUS_RING_TRACKER

    if _FOCUS_RING_TRACKER is None or _FOCUS_RING_TRACKER.parent() is not app:
        _FOCUS_RING_TRACKER = _FocusRingTracker(app)


def track_item_view_corners(view: QTableWidget | QTreeWidget) -> None:
    """Keep ``view``'s corner-styling properties in step with its scroll bar and header.

    Sets ``hasVerticalScrollbar`` and ``hasVisibleVerticalHeader`` on the view and
    updates them as it is shown, resized or restyled and as its scroll range
    changes. The shared table helpers call this; call it directly for a
    ``QTableWidget`` or ``QTreeWidget`` built without them. Before the theme is
    applied this does nothing, and ``apply_theme`` attaches the view instead.
    """

    styler = _TABLE_CORNER_STYLER
    if styler is not None:
        styler.attach_table(view)


def cleanup_theme_runtime(app: QApplication | None = None) -> None:
    """Detach long-lived Qt hooks so test/process shutdown is deterministic."""

    global _TABLE_CORNER_STYLER, _FOCUS_RING_TRACKER, _BASE_STYLE_APP, _FONTS_LOADED

    _BASE_STYLE_APP = None
    _FONTS_LOADED = False

    tracker = _FOCUS_RING_TRACKER
    if tracker is not None:
        _FOCUS_RING_TRACKER = None
        try:
            tracker_app = tracker.parent()
            if isinstance(tracker_app, QApplication):
                tracker.shutdown(tracker_app)
            tracker.deleteLater()
        except (RuntimeError, SystemError):
            pass

    styler = _TABLE_CORNER_STYLER
    if styler is None:
        return

    try:
        styler.shutdown()
    except (RuntimeError, SystemError):
        pass

    try:
        styler.deleteLater()
    except (RuntimeError, SystemError):
        pass

    _TABLE_CORNER_STYLER = None


def _system_prefers_dark() -> bool:
    app = QGuiApplication.instance()
    if app is None:
        return False

    hints = app.styleHints()
    color_scheme_attr = getattr(Qt, "ColorScheme", None)
    if color_scheme_attr is None or not hasattr(hints, "colorScheme"):
        return False

    return hints.colorScheme() == Qt.ColorScheme.Dark


def resolve_theme_mode(mode: str) -> str:
    """Normalize the requested mode to either light or dark."""

    requested = (mode or THEME_SYSTEM).strip().lower()
    if requested == THEME_DARK:
        return THEME_DARK
    if requested == THEME_LIGHT:
        return THEME_LIGHT
    return THEME_DARK if _system_prefers_dark() else THEME_LIGHT


def current_palette() -> MuFinderPalette:
    """Return the currently active palette."""

    return _CURRENT_PALETTE


def _status_map(palette: MuFinderPalette) -> dict[str, str]:
    return {
        "ready": palette.status_ready,
        "defined": palette.status_defined,
        "submitting": palette.status_submitting,
        "queued": palette.status_queued,
        "running": palette.status_running,
        "retrieving": palette.status_retrieving,
        "stopped": palette.status_stopped,
        "completed": palette.status_completed,
        "failed": palette.status_failed,
        "cancelled": palette.status_cancelled,
        "not generated": palette.status_cancelled,
    }


def status_color(status_value: str) -> QColor:
    """Return QColor for a workflow status label."""

    palette = current_palette()
    normalized = status_value.strip().lower()
    color = _status_map(palette).get(normalized, palette.text_primary)
    return QColor(color)


def status_brush(status_value: str) -> QBrush:
    """Return a text brush for a workflow status label."""

    return QBrush(status_color(status_value))


def load_theme_mode(settings: QSettings) -> str:
    """Read persisted appearance mode from settings."""

    value = settings.value(THEME_MODE_KEY, THEME_SYSTEM, str)
    normalized = str(value or THEME_SYSTEM).strip().lower()
    if normalized not in {THEME_LIGHT, THEME_DARK, THEME_SYSTEM}:
        return THEME_SYSTEM
    return normalized


def save_theme_mode(settings: QSettings, mode: str) -> None:
    """Persist appearance mode to settings."""

    normalized = (mode or THEME_SYSTEM).strip().lower()
    if normalized not in {THEME_LIGHT, THEME_DARK, THEME_SYSTEM}:
        normalized = THEME_SYSTEM
    settings.setValue(THEME_MODE_KEY, normalized)


def style_plot_axes(
    axis: Axes,
    *,
    facecolor: str | None = None,
    tick_color: str | None = None,
    tick_labelsize: float | None = None,
    grid_alpha: float = 0.35,
    grid_linewidth: float | None = None,
    grid: bool = True,
) -> None:
    """Paint a matplotlib axis in the current palette.

    The axis background defaults to ``bg_surface_alt`` and the ticks to
    ``text_primary``; axis labels and the title use ``text_primary``, spines and
    grid lines use ``border``, and ``grid=False`` leaves the grid off. Major and
    minor ticks are both styled, because a log axis labels its minor ticks. Label
    and title colours persist, so this may run before or after the text is set.
    """

    palette = current_palette()
    axis.set_facecolor(facecolor or palette.bg_surface_alt)
    tick_kwargs: dict[str, Any] = {"which": "both", "colors": tick_color or palette.text_primary}
    if tick_labelsize is not None:
        tick_kwargs["labelsize"] = tick_labelsize
    axis.tick_params(**tick_kwargs)
    axis.xaxis.label.set_color(palette.text_primary)
    axis.yaxis.label.set_color(palette.text_primary)
    axis.title.set_color(palette.text_primary)
    for spine in axis.spines.values():
        spine.set_color(palette.border)
    if not grid:
        axis.grid(False)
        return
    grid_kwargs: dict[str, Any] = {"color": palette.border, "alpha": grid_alpha}
    if grid_linewidth is not None:
        grid_kwargs["linewidth"] = grid_linewidth
    axis.grid(True, **grid_kwargs)


def plot_badge_box(*, pad: float = 0.18) -> dict[str, object]:
    """Return the ``bbox`` style of a badge or hover annotation drawn on a plot.

    A rounded box in the surface colour with a border, nearly opaque, so the
    text stays legible over lines and grid.
    """

    palette = current_palette()
    return {
        "boxstyle": f"round,pad={pad:g}",
        "fc": palette.bg_surface,
        "ec": palette.border,
        "alpha": 0.95,
    }


def style_plot_legend(legend: Legend | None, *, frame_alpha: float | None = None) -> None:
    """Paint a matplotlib legend frame and text in the current palette.

    Does nothing when ``legend`` is ``None``, so ``axis.get_legend()`` can be
    passed straight in.
    """

    if legend is None:
        return
    palette = current_palette()
    frame = legend.get_frame()
    frame.set_facecolor(palette.bg_surface)
    frame.set_edgecolor(palette.border)
    if frame_alpha is not None:
        frame.set_alpha(frame_alpha)
    for text in legend.get_texts():
        text.set_color(palette.text_primary)


@dataclass(frozen=True, slots=True)
class PlotSeriesColours:
    """Colours of plotted data, by role, for one theme.

    ``energy``, ``force``, ``stress`` and ``displacement`` give a quantity the
    same colour on every plot. ``threshold`` is a tolerance, criterion or cut
    line and ``reference`` a neutral marker line such as a recommended value or
    zero. ``total`` is the sum or average drawn over the individual series.
    ``fit`` is a fitted curve and its limit, ``converged`` the line where an
    energy or force criterion converged and ``stress_converged`` the stress one.
    ``categorical`` colours series that have no role (sites, clusters, one
    unnamed curve), in order: take them with ``categorical_colour``.
    ``label_on_tint`` is text drawn on a pale tint of a series colour, such as a
    number inside a graph node, so it is dark in both themes.

    Every role but ``label_on_tint``, and every categorical colour, keeps at
    least 3:1 contrast against the plot backgrounds (``bg_primary``,
    ``bg_surface`` and ``bg_surface_alt``), and roles drawn on the same plot stay
    far enough apart to read as different lines.
    """

    energy: str
    force: str
    stress: str
    displacement: str
    threshold: str
    reference: str
    total: str
    fit: str
    converged: str
    stress_converged: str
    label_on_tint: str
    categorical: tuple[str, ...]


def plot_series_colours() -> PlotSeriesColours:
    """Return the plot series palette of the current theme."""

    palette = current_palette()
    if palette.mode == THEME_DARK:
        return PlotSeriesColours(
            energy="#60A5FA",
            force="#F87171",
            stress="#FBBF24",
            displacement="#C4B5FD",
            threshold="#E879F9",
            reference=palette.text_muted,
            total=palette.text_primary,
            fit="#2DD4BF",
            converged="#A3E635",
            stress_converged="#5EEAD4",
            label_on_tint="#111111",
            categorical=(
                "#60A5FA",
                "#F87171",
                "#2DD4BF",
                "#FBBF24",
                "#C4B5FD",
                "#4ADE80",
                "#F472B6",
                "#7DD3FC",
                "#BEF264",
                "#E879F9",
                "#D6A77A",
                "#CBD5E1",
            ),
        )
    return PlotSeriesColours(
        energy="#1D4ED8",
        force="#B91C1C",
        stress="#B85C00",
        displacement="#7C3AED",
        threshold="#A21CAF",
        reference=palette.text_muted,
        total=palette.text_primary,
        fit="#0F766E",
        converged="#15803D",
        stress_converged="#0E7490",
        label_on_tint="#111111",
        categorical=(
            "#1D4ED8",
            "#B91C1C",
            "#0F766E",
            "#B45309",
            "#7C3AED",
            "#15803D",
            "#BE185D",
            "#0369A1",
            "#4D7C0F",
            "#A21CAF",
            "#78350F",
            "#475569",
        ),
    )


def categorical_colour(index: int) -> str:
    """Return the ``index``-th categorical series colour, repeating past the end of the list."""

    colours = plot_series_colours().categorical
    return colours[index % len(colours)]


def cluster_color_map(cluster_labels: Iterable[int]) -> dict[int, str]:
    """Return each cluster label's categorical colour, in ascending label order."""

    colours = plot_series_colours().categorical
    return {
        label: colours[index % len(colours)]
        for index, label in enumerate(sorted({int(label) for label in cluster_labels}))
    }


def cell_survey_species_tones() -> tuple[str, ...]:
    """Return the muted atom tones of the project home's cell survey, for the current theme."""

    if current_palette().mode == THEME_DARK:
        return ("#B8A898", "#8FAABB", "#A9BB8F", "#BB8FA3", "#9C9CBB", "#BBB38F")
    return ("#8C7B6B", "#5E7A8C", "#7A8C5E", "#8C5E73", "#6B6B8C", "#8C845E")


@dataclass(frozen=True, slots=True)
class StructureOverlayColours:
    """Fixed colours of what a workflow draws over a 3D structure, the same in both themes.

    ``muon`` is a muon site or the muon's phonon displacement, ``path_start``
    and ``path_end`` the endpoints of an energy-barrier path and ``path_arrow``
    the arrow between them, ``spin_up`` and ``spin_down`` the two spin
    directions, and ``line`` a line overlay given no colour of its own.
    """

    muon: str = "#E11D48"
    path_start: str = "#FF7A18"
    path_end: str = "#0F8AA8"
    path_arrow: str = "#2F6F4F"
    spin_up: str = "#1F78B4"
    spin_down: str = "#D7301F"
    line: str = "#15A3A3"


STRUCTURE_OVERLAY_COLOURS = StructureOverlayColours()

#: The comparison table's best-agreement value, fixed because a table item keeps
#: its colour when the theme changes.
BEST_AGREEMENT_COLOUR = "#15803D"


def style_section_title(label: QLabel) -> None:
    """Tag a label as a section title for theme-based styling."""

    label.setObjectName("sectionTitle")


def style_section_subtitle(label: QLabel) -> None:
    """Tag a label as a section subtitle for theme-based styling."""

    label.setObjectName("sectionSubtitle")


def style_hint_text(label: QLabel) -> None:
    """Tag a label as helper hint text for theme-based styling."""

    label.setObjectName("hintText")


def style_warning_text(label: QLabel) -> None:
    """Tag a label as warning text for theme-based styling."""

    label.setObjectName("warningText")


def make_empty_state_label(
    text: str = "",
    *,
    object_name: str = "overviewPlaceholder",
    alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter,
    parent: QWidget | None = None,
) -> QLabel:
    """Return a word-wrapped placeholder ``QLabel`` for an empty job/result view.

    Centralizes the near-identical "nothing here yet" placeholder pattern
    previously hand-copied in ``convergence_overview_widget.py``,
    ``muon_sites_summary_widget.py``, and ``journal_widget.py``. Callers keep
    their own ``object_name`` so existing stylesheet rules and tests that key
    off it continue to work unchanged.
    """

    label = QLabel(text, parent)
    label.setObjectName(object_name)
    label.setWordWrap(True)
    label.setAlignment(alignment)
    return label


def style_inline_group(widget: QWidget) -> None:
    """Tag a compact row/group widget as transparent inline UI chrome."""

    widget.setObjectName("mufinderInlineGroup")


def style_transparent_surface(widget: QWidget) -> None:
    """Let what lies behind a widget and everything inside it show through.

    For containers and canvases drawn over another surface. Labels, checkboxes
    and radio buttons need no call: the application stylesheet already makes
    them transparent wherever they sit.
    """

    widget.setStyleSheet("background: transparent;")


def style_option_card(card: QWidget) -> None:
    """Tag a frame-like widget as a selectable feature option card."""

    card.setObjectName("mufinderOptionCard")
    card.setProperty("selected", False)
    card.setProperty("unavailable", False)


def set_option_card_state(
    card: QWidget,
    *,
    selected: bool,
    unavailable: bool = False,
) -> None:
    """Update option-card state properties and refresh stylesheet polish."""

    card.setProperty("selected", bool(selected))
    card.setProperty("unavailable", bool(unavailable))
    refresh_widget_style(card)


def style_help_button(button: QToolButton) -> None:
    """Tag a small help/info tool button for shared feature-dialog styling."""

    button.setObjectName("mufinderHelpButton")


def style_tab_widget(tabs: QTabWidget) -> None:
    """Give a tab widget the rounded, accent-marked MuFinder tabs.

    The tabs keep their natural width (no stretching across the bar) and long
    titles are elided rather than clipped.
    """

    tabs.setObjectName("mufinderTabs")
    tabs.setDocumentMode(False)
    tabs.tabBar().setExpanding(False)
    tabs.setElideMode(Qt.TextElideMode.ElideRight)


def style_small_checked_tool_button(button: QToolButton) -> None:
    """Tag a small checkable tool button for shared feature-dialog styling."""

    button.setObjectName("mufinderCheckedToolButton")


def use_square_top_table_corners(table: QTableWidget | QTreeWidget) -> None:
    """Opt a table/tree item view into square top corners for titled container layouts."""

    table.setProperty("squareTopCorners", "true")
    track_item_view_corners(table)
    _repolish_widget(table)
    if isinstance(table, QTableWidget):
        _repolish_widget(table.horizontalHeader())
    elif isinstance(table, QTreeWidget):
        _repolish_widget(table.header())
    _repolish_widget(table.verticalScrollBar())


def use_tabbed_table_corners(table: QTableWidget | QTreeWidget) -> None:
    """Opt an item view into the square-corner treatment used inside tab panes."""

    table.setProperty("tabbedTable", "true")
    use_square_top_table_corners(table)


def _build_palette(palette: MuFinderPalette) -> QPalette:
    qpalette = QPalette()
    qpalette.setColor(QPalette.ColorRole.Window, QColor(palette.bg_primary))
    qpalette.setColor(QPalette.ColorRole.WindowText, QColor(palette.text_primary))
    qpalette.setColor(QPalette.ColorRole.Base, QColor(palette.bg_surface))
    qpalette.setColor(QPalette.ColorRole.AlternateBase, QColor(palette.bg_surface_alt))
    qpalette.setColor(QPalette.ColorRole.ToolTipBase, QColor(palette.bg_surface))
    qpalette.setColor(QPalette.ColorRole.ToolTipText, QColor(palette.text_primary))
    qpalette.setColor(QPalette.ColorRole.Text, QColor(palette.text_primary))
    qpalette.setColor(QPalette.ColorRole.Button, QColor(palette.bg_surface))
    qpalette.setColor(QPalette.ColorRole.ButtonText, QColor(palette.text_primary))
    qpalette.setColor(QPalette.ColorRole.BrightText, QColor("#FFFFFF"))
    qpalette.setColor(QPalette.ColorRole.Highlight, QColor(palette.accent))
    qpalette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    return qpalette


def _base_widget_styles(p: MuFinderPalette) -> str:
    return f"""
    QWidget {{
        background: {p.bg_primary};
        color: {p.text_primary};
    }}
    QWidget:disabled {{
        color: {p.text_muted};
    }}
    QLabel, QCheckBox, QRadioButton {{
        background: transparent;
    }}
    QMainWindow, QDialog {{
        background: {p.bg_primary};
    }}
    QGroupBox {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.card}px;
        margin-top: 10px;
        padding: {SPACING.sm}px;
        font-weight: 600;
        background: {p.bg_surface};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0px {SPACING.xs}px;
        color: {p.text_primary};
    }}
    """


def feature_dialog_stylesheet(
    *, include_scroll_area: bool = False, emphasised_radio_buttons: bool = False
) -> str:
    """Return the stylesheet a feature dialog sets on itself.

    ``include_scroll_area`` makes scroll areas transparent and borderless;
    ``emphasised_radio_buttons`` draws radio buttons semibold with room around
    them, for dialogs whose choices are radio-button option cards.
    """

    palette = current_palette()
    radio_buttons = (
        f"""
        QRadioButton {{
            font-weight: 600;
            spacing: 10px;
            padding-top: {SPACING.xs}px;
            padding-bottom: {SPACING.xs}px;
        }}
        """
        if emphasised_radio_buttons
        else ""
    )
    scroll_area = (
        """
        QScrollArea {
            background: transparent;
            border: none;
        }
        QScrollArea > QWidget > QWidget {
            background: transparent;
        }
        """
        if include_scroll_area
        else ""
    )
    return f"""
        QDialog {{
            background: {palette.bg_primary};
        }}
        QGroupBox {{
            background: {palette.bg_surface};
            border: 1px solid {palette.border};
            border-radius: {RADIUS.card}px;
            margin-top: 12px;
            padding-top: {SPACING.md}px;
            font-weight: 700;
            color: {palette.text_primary};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0px {SPACING.sm}px;
        }}
        QFrame#mufinderOptionCard {{
            background: {palette.bg_surface_alt};
            border: 1px solid {palette.border};
            border-radius: {RADIUS.card}px;
        }}
        QFrame#mufinderOptionCard[selected="true"] {{
            background: {palette.bg_surface};
            border: 1px solid {palette.accent};
        }}
        QFrame#mufinderOptionCard[unavailable="true"] {{
            background: {palette.bg_primary};
            border: 1px solid {palette.border};
        }}
        QWidget#mufinderInlineGroup,
        QWidget#mufinderInlineGroup QLabel {{
            background: transparent;
        }}
        QRadioButton {{
            background: transparent;
        }}
        QCheckBox {{
            background: transparent;
        }}
        QToolButton#mufinderCheckedToolButton {{
            background: {palette.bg_surface_alt};
            border: 1px solid {palette.border};
            border-radius: {RADIUS.control}px;
            padding: {SPACING.xs}px;
            min-width: 24px;
            min-height: 24px;
        }}
        QToolButton#mufinderCheckedToolButton:checked {{
            background: {palette.bg_surface};
            border: 1px solid {palette.accent};
        }}
        QToolButton#mufinderCheckedToolButton:hover {{
            border: 1px solid {palette.accent_hover};
        }}
        {scroll_area}
        {radio_buttons}
        """


def uep_clustering_toolbar_stylesheet() -> str:
    """Return the stylesheet of the clustering dendrogram's navigation toolbar."""

    palette = current_palette()
    t = type_scale()
    return f"""
        QToolBar#uepClusteringDendrogramToolbar {{
            border: none;
            background: transparent;
            spacing: 8px;
            padding: 0px;
        }}
        QPushButton#uepClusteringToolbarButton {{
            border: 1px solid {palette.border};
            border-radius: {RADIUS.control}px;
            padding: {SPACING.xs}px {SPACING.md}px;
            background: {palette.bg_surface};
            color: {palette.text_primary};
            font-size: {t.body}px;
            font-weight: 600;
        }}
        QPushButton#uepClusteringToolbarButton:hover {{
            border-color: {palette.accent};
            background: {palette.bg_surface_alt};
        }}
        QPushButton#uepClusteringToolbarButton:pressed {{
            border-color: {palette.accent_hover};
            background: {palette.bg_surface_alt};
        }}
        QPushButton#uepClusteringToolbarButton:checked,
        QPushButton#uepClusteringToolbarButton[active="true"] {{
            border-color: {palette.accent_pressed};
            background: {palette.accent};
            color: #FFFFFF;
        }}
        QPushButton#uepClusteringToolbarButton:checked:hover,
        QPushButton#uepClusteringToolbarButton[active="true"]:hover {{
            background: {palette.accent_hover};
        }}
        """


def transparent_spin_box_stylesheet() -> str:
    """Return the stylesheet of a spin box drawn straight on its card, arrows included."""

    return """
        QSpinBox {
            background: transparent;
        }
        QSpinBox::up-button,
        QSpinBox::down-button {
            background: transparent;
        }
        """


def magnetic_fields_stylesheet() -> str:
    """Return the stylesheet the Magnetic Fields page and its muon-site dialog set on themselves.

    Option rows, labels and stacked pages draw on their parent's background, and
    radio buttons and checkboxes get more room. It holds no colours, so it
    stays right when the theme changes.
    """

    return f"""
        QRadioButton, QCheckBox {{
            background: transparent;
            spacing: 10px;
            padding-top: {SPACING.xs}px;
            padding-bottom: {SPACING.xs}px;
        }}
        QWidget#magneticMuonSiteInlinePanel {{
            background: transparent;
        }}
        QStackedWidget, QStackedWidget > QWidget, QLabel {{
            background: transparent;
        }}
        """


def magnetic_structure_builder_stylesheet() -> str:
    """Return the stylesheet the Magnetic Structure Builder sets on itself.

    The viewer's Cell, Bonds and Spins checkboxes get larger indicators and
    tighter spacing than the page's checkboxes.
    """

    return f"""
        QWidget#atomicViewerDisplayControls QCheckBox {{
            background: transparent;
            spacing: 9px;
            padding-left: {SPACING.xs}px;
            padding-right: {SPACING.xs}px;
        }}
        QWidget#atomicViewerDisplayControls QCheckBox::indicator {{
            width: 18px;
            height: 18px;
        }}
        """


@dataclass(frozen=True, slots=True)
class TexturePreviewColors:
    """Colours of the magnetic texture preview for one canvas (light or dark).

    The preview colours each moment arrow by its out-of-plane component, from
    ``negative`` (pointing down, at full length) through ``negative_zero`` and
    ``positive_zero`` to ``positive``, or, for a flat texture, by its length
    from ``magnitude_low`` to ``magnitude_high``.
    """

    surround: str
    plot_background: str
    frame: str
    placeholder_text: str
    zoom_box: str
    negative_zero: str
    negative: str
    positive_zero: str
    positive: str
    magnitude_low: str
    magnitude_high: str


def texture_preview_colors(canvas_mode: str) -> TexturePreviewColors:
    """Return the texture preview colours for ``canvas_mode`` (``"light"`` or ``"dark"``).

    The canvas follows the mode the user picked in the preview, whatever the
    theme: its frame, text and highlight colours come from the matching theme
    palette. The area around the canvas follows the current theme.
    """

    surround = current_palette().bg_surface
    if canvas_mode == THEME_DARK:
        dark = dark_palette()
        return TexturePreviewColors(
            surround=surround,
            plot_background="#111827",
            frame=dark.border,
            placeholder_text=dark.text_muted,
            zoom_box="#FFFFFF",
            negative_zero="#22D3EE",
            negative="#8B5CF6",
            positive_zero="#34D399",
            positive="#FACC15",
            magnitude_low="#22D3EE",
            magnitude_high="#FACC15",
        )
    light = light_palette()
    return TexturePreviewColors(
        surround=surround,
        plot_background="#FFFDF8",
        frame=light.border,
        placeholder_text=light.text_muted,
        zoom_box=light.accent,
        negative_zero="#1F6F78",
        negative="#3F2D5C",
        positive_zero="#5D9A57",
        positive=light.seal_gold,
        magnitude_low="#1F6F78",
        magnitude_high=light.accent,
    )


def _input_control_styles(p: MuFinderPalette, arrow_up: str, arrow_down: str) -> str:
    return f"""
    QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.control}px;
        padding: {SPACING.xs}px {SPACING.sm}px;
        background: {p.bg_surface};
    }}
    /* A disabled input is drawn muted, so a field that is ignored reads as such. */
    QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QSpinBox:disabled,
    QDoubleSpinBox:disabled, QComboBox:disabled {{
        background: {p.bg_surface_alt};
        color: {p.text_muted};
    }}
    QComboBox {{
        padding-right: {_COMBO_ARROW_GUTTER}px;
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 24px;
        border: 0;
        border-left: 1px solid {p.border};
        border-top-right-radius: {RADIUS.control}px;
        border-bottom-right-radius: {RADIUS.control}px;
        background: {p.bg_surface_alt};
    }}
    QComboBox::drop-down:hover {{
        background: {p.bg_surface_alt};
        border-left-color: {p.accent};
    }}
    QComboBox::down-arrow {{
        image: url("{arrow_down}");
        width: 12px;
        height: 12px;
    }}
    QSpinBox, QDoubleSpinBox {{
        padding-right: {_SPIN_ARROW_GUTTER}px;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 22px;
        border: 0;
        border-left: 1px solid {p.border};
        border-bottom: 1px solid {p.border};
        border-top-right-radius: {RADIUS.control}px;
        background: {p.bg_surface_alt};
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 22px;
        border: 0;
        border-left: 1px solid {p.border};
        border-bottom-right-radius: {RADIUS.control}px;
        background: {p.bg_surface_alt};
    }}
    QSpinBox::up-button:hover,
    QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover,
    QDoubleSpinBox::down-button:hover {{
        background: {p.bg_surface_alt};
        border-left-color: {p.accent};
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url("{arrow_up}");
        width: 12px;
        height: 12px;
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url("{arrow_down}");
        width: 12px;
        height: 12px;
    }}
    QSlider#uepIsosurfaceSlider {{
        min-height: 34px;
    }}
    QSlider#uepIsosurfaceSlider::groove:horizontal {{
        height: 14px;
        border: 1px solid {p.border};
        border-radius: {ROUND_RADIUS.slider_groove}px;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 {p.bg_surface_alt},
            stop: 1 {p.bg_primary}
        );
    }}
    QSlider#uepIsosurfaceSlider::sub-page:horizontal {{
        border-radius: {ROUND_RADIUS.slider_groove}px;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 {p.accent},
            stop: 1 {p.seal_gold}
        );
    }}
    QSlider#uepIsosurfaceSlider::add-page:horizontal {{
        border-radius: {ROUND_RADIUS.slider_groove}px;
        background: {p.bg_surface_alt};
    }}
    QSlider#uepIsosurfaceSlider::handle:horizontal {{
        width: 26px;
        margin: -9px 0;
        border: 2px solid {p.bg_surface};
        border-radius: {ROUND_RADIUS.slider_handle}px;
        background: {p.seal_gold};
    }}
    QSlider#uepIsosurfaceSlider::handle:horizontal:hover {{
        background: {p.accent};
    }}
    QSlider#uepIsosurfaceSlider::handle:horizontal:pressed {{
        background: {p.accent_hover};
    }}
    QSlider#uepIsosurfaceSlider:disabled::groove:horizontal {{
        background: {p.bg_surface_alt};
    }}
    QSlider#uepIsosurfaceSlider:disabled::sub-page:horizontal {{
        background: {p.border};
    }}
    QSlider#uepIsosurfaceSlider:disabled::handle:horizontal {{
        background: {p.bg_surface};
        border-color: {p.border};
    }}
    """


def _item_view_styles(p: MuFinderPalette) -> str:
    return f"""
    QTreeWidget, QTableWidget, QListWidget {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.control}px;
        gridline-color: {p.border};
        background: {p.bg_surface};
        alternate-background-color: {p.bg_surface_alt};
    }}
    QTableWidget QTableCornerButton::section {{
        background: {p.bg_surface_alt};
        border: 0;
        border-right: 1px solid {p.border};
        border-bottom: 1px solid {p.border};
        border-top-left-radius: {RADIUS.control}px;
    }}
    QTableWidget[squareTopCorners="true"] {{
        border-top-left-radius: 0px;
        border-top-right-radius: 0px;
    }}
    QTreeWidget[squareTopCorners="true"] {{
        border-top-left-radius: 0px;
        border-top-right-radius: 0px;
    }}
    QTableWidget[hasVerticalScrollbar="true"] {{
        border-top-right-radius: 0px;
        border-bottom-right-radius: 0px;
    }}
    QTreeWidget[hasVerticalScrollbar="true"] {{
        border-top-right-radius: 0px;
        border-bottom-right-radius: 0px;
    }}
    QTableWidget[squareTopCorners="true"] QTableCornerButton::section {{
        border-top-left-radius: 0px;
    }}
    """


def _panel_container_styles(p: MuFinderPalette) -> str:
    return f"""
    QFrame#projectTreePanel {{
        border: none;
        background: transparent;
    }}
    QFrame#projectTreePanelHeader {{
        background: {p.bg_surface_alt};
        border: 1px solid {p.border};
        border-bottom: 1px solid {p.border};
        border-top-left-radius: {RADIUS.card}px;
        border-top-right-radius: {RADIUS.card}px;
    }}
    QLabel#projectTreePanelTitle {{
        font-weight: 600;
    }}
    QFrame#projectTreePanelBody {{
        border: 1px solid {p.border};
        border-top: 0px;
        border-bottom-left-radius: {RADIUS.card}px;
        border-bottom-right-radius: {RADIUS.card}px;
        background: {p.bg_surface};
    }}
    QFrame#projectDefaultsStrip {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.card}px;
        background: {p.bg_surface};
    }}
    QFrame#projectDefaultsSeparator {{
        border: none;
        background: {p.border};
    }}
    QLabel#projectDefaultsCaption {{
        color: {p.text_muted};
        font-weight: 600;
    }}
    QLabel#projectDefaultsName {{
        color: {p.text_muted};
    }}
    QLabel#projectDefaultsValue {{
        font-weight: 600;
    }}
    QLabel#projectDefaultsValue[valueState="unset"] {{
        color: {p.text_muted};
        font-weight: 400;
    }}
    QLabel#projectDefaultsValue[valueState="stale"] {{
        color: {p.warning};
    }}
    QToolButton#projectDefaultsButton {{
        border: 1px solid {p.border};
        background: transparent;
        color: {p.text_primary};
        padding: {SPACING.xs}px {SPACING.sm}px;
        border-radius: {RADIUS.control}px;
    }}
    QToolButton#projectDefaultsButton:hover {{
        background: {p.bg_surface_alt};
    }}
    QDockWidget#jobDashboardDock {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.card}px;
        background: {p.bg_surface};
    }}
    QDockWidget#jobDashboardDock::title {{
        background: {p.bg_surface_alt};
        border: 1px solid {p.border};
        border-bottom: 1px solid {p.border};
        border-top-left-radius: {RADIUS.card}px;
        border-top-right-radius: {RADIUS.card}px;
        padding: {SPACING.sm}px {SPACING.md}px;
        text-align: left;
        font-weight: 600;
    }}
    """


def _muon_generation_styles(p: MuFinderPalette) -> str:
    return f"""
    QGroupBox#muonGenerationSetupCard {{
        background: {p.bg_surface};
    }}
    QStackedWidget#muonGenerationParamStack {{
        background: {p.bg_surface};
        border: 0px;
    }}
    QWidget#muonGenerationRandomPage,
    QWidget#muonGenerationNearPage,
    QWidget#muonGenerationManualPage,
    QWidget#muonGenerationUepPage {{
        background: {p.bg_surface};
    }}
    QWidget#randomSeparationChoice,
    QWidget#randomCountChoice {{
        background: {p.bg_surface};
        border: 1px solid {p.border};
        border-radius: {RADIUS.control}px;
    }}
    QWidget#randomSeparationChoice QRadioButton,
    QWidget#randomCountChoice QRadioButton {{
        color: {p.text_muted};
        background: transparent;
    }}
    QWidget#randomSeparationChoice[active="true"],
    QWidget#randomCountChoice[active="true"] {{
        background: {p.bg_surface_alt};
        border: 1px solid {p.accent};
        border-radius: {RADIUS.control}px;
    }}
    QWidget#randomSeparationChoice[active="true"] QRadioButton,
    QWidget#randomCountChoice[active="true"] QRadioButton {{
        color: {p.text_primary};
        background: transparent;
    }}
    QWidget#randomSeparationChoice[active="true"] QLineEdit,
    QWidget#randomCountChoice[active="true"] QSpinBox {{
        border-color: {p.accent};
    }}
    QDialog#clusterPositionsDialog {{
        background: {p.bg_primary};
    }}
    QWidget#clusterPositionsCard {{
        background: {p.bg_surface};
        border: 1px solid {p.border};
        border-radius: {RADIUS.card}px;
    }}
    QWidget#clusterPositionsCard QLabel {{
        background: transparent;
    }}
    QProgressBar#muonGenerationProgress {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.control}px;
        background: {p.bg_surface_alt};
        color: {p.text_primary};
        text-align: center;
        min-height: 22px;
    }}
    QProgressBar#muonGenerationProgress::chunk {{
        background: {p.accent};
        border-radius: {RADIUS.control}px;
        margin: 1px;
    }}
    QProgressBar#muonGenerationProgress[generationState="complete"] {{
        color: #FFFFFF;
    }}
    QProgressBar#muonGenerationProgress[generationState="complete"]::chunk {{
        background: {p.success};
    }}
    """


def _magnetic_field_styles(p: MuFinderPalette) -> str:
    return f"""
    QLineEdit#nuclearDirectionInput {{
        background: {p.bg_surface};
        border: 1px solid {p.border};
        color: {p.text_primary};
        border-radius: {RADIUS.control}px;
        padding: {SPACING.xs}px {SPACING.sm}px;
    }}
    QLineEdit#nuclearDirectionInput:disabled {{
        background: {p.bg_surface};
        color: {p.text_muted};
    }}
    """


def _muon_relaxation_styles(p: MuFinderPalette) -> str:
    return f"""
    QTabWidget#mufinderTabs::pane,
    QTabWidget#muonRelaxationChargeTabs::pane,
    QTabWidget#analysisWorkspaceTabs::pane,
    QTabWidget#energyBarriersTabs::pane,
    QTabWidget#quantumAnalysisTabs::pane {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.control}px;
        top: -1px;
        background: {p.bg_surface};
    }}
    QTabWidget#mufinderTabs,
    QTabWidget#muonRelaxationChargeTabs,
    QTabWidget#analysisWorkspaceTabs,
    QTabWidget#energyBarriersTabs,
    QTabWidget#quantumAnalysisTabs {{
        background: transparent;
    }}
    QTabWidget#mufinderTabs > QStackedWidget > QWidget {{
        background: {p.bg_surface};
    }}
    QTabWidget#mufinderTabs QTabBar,
    QTabWidget#muonRelaxationChargeTabs QTabBar,
    QTabWidget#analysisWorkspaceTabs QTabBar,
    QTabWidget#energyBarriersTabs QTabBar,
    QTabWidget#quantumAnalysisTabs QTabBar {{
        background: transparent;
    }}
    QTabWidget#mufinderTabs QTabBar::tab,
    QTabWidget#muonRelaxationChargeTabs QTabBar::tab,
    QTabWidget#analysisWorkspaceTabs QTabBar::tab,
    QTabWidget#energyBarriersTabs QTabBar::tab,
    QTabWidget#quantumAnalysisTabs QTabBar::tab {{
        background: {p.bg_surface_alt};
        color: {p.text_muted};
        border: 1px solid {p.border};
        border-bottom: none;
        border-top-left-radius: {RADIUS.control}px;
        border-top-right-radius: {RADIUS.control}px;
        padding: {SPACING.sm}px {SPACING.lg}px;
        margin-right: 4px;
        font-weight: 700;
    }}
    QTabWidget#mufinderTabs QTabBar::tab:selected,
    QTabWidget#muonRelaxationChargeTabs QTabBar::tab:selected,
    QTabWidget#analysisWorkspaceTabs QTabBar::tab:selected,
    QTabWidget#energyBarriersTabs QTabBar::tab:selected {{
        background: {p.bg_surface};
        color: {p.text_primary};
        border-color: {p.accent};
    }}
    QTabWidget#quantumAnalysisTabs QTabBar::tab:selected {{
        background: {p.bg_primary};
        color: {p.text_primary};
        border-color: {p.accent};
    }}
    QTabWidget#mufinderTabs QTabBar::tab:hover:!selected,
    QTabWidget#muonRelaxationChargeTabs QTabBar::tab:hover:!selected,
    QTabWidget#analysisWorkspaceTabs QTabBar::tab:hover:!selected,
    QTabWidget#energyBarriersTabs QTabBar::tab:hover:!selected,
    QTabWidget#quantumAnalysisTabs QTabBar::tab:hover:!selected {{
        background: {p.bg_surface};
        color: {p.text_primary};
    }}
    QDialog#quantumPhononAnalysisDialog,
    QDialog#quantumAnalysisHelpDialog,
    QTabWidget#quantumAnalysisTabs,
    QTabWidget#quantumAnalysisTabs::pane,
    QWidget#quantumAnalysisTabPage,
    QWidget#quantumAnalysisHelpContent {{
        background: {p.bg_primary};
    }}
    QToolButton#quantumAnalysisHelpButton {{
        background: {p.bg_surface_alt};
        color: {p.text_primary};
        border: 1px solid {p.border};
        border-radius: {ROUND_RADIUS.help_button}px;
        min-width: 24px;
        min-height: 24px;
        font-weight: 800;
    }}
    QToolButton#quantumAnalysisHelpButton:hover {{
        background: {p.bg_surface};
        border-color: {p.accent};
    }}
    QDialog#muonChargeStateDialog {{
        background: {p.bg_primary};
    }}
    QWidget#muonChargeStateCard {{
        background: {p.bg_surface};
        border: 1px solid {p.border};
        border-radius: {RADIUS.card}px;
    }}
    QWidget#muonChargeStateCard QLabel {{
        background: transparent;
    }}
    QRadioButton[chargeStateOption="true"] {{
        background: transparent;
        padding: {SPACING.xs}px;
        color: {p.text_primary};
    }}
    QRadioButton[chargeStateOption="true"]:checked {{
        color: {p.accent};
        font-weight: 700;
    }}
    """


def _workflow_tree_styles(
    p: MuFinderPalette,
    branch_right: str,
    branch_down: str,
    branch_right_selected: str,
    branch_down_selected: str,
) -> str:
    return f"""
    QTreeWidget#workflowTree {{
        background: transparent;
        border: none;
        border-radius: 0px;
        alternate-background-color: transparent;
        show-decoration-selected: 1;
    }}
    QTreeWidget#workflowTree::item {{
        background: transparent;
        padding: {SPACING.xs}px {SPACING.xs}px;
        border-radius: 0px;
    }}
    QTreeWidget#workflowTree::item:hover {{
        background: {p.bg_surface_alt};
    }}
    QTreeWidget#workflowTree::item:selected {{
        background: {p.accent};
        color: #FFFFFF;
        border-radius: 0px;
    }}
    QTreeWidget#workflowTree::item:selected:active,
    QTreeWidget#workflowTree::item:selected:!active {{
        background: {p.accent};
        color: #FFFFFF;
        border-radius: 0px;
    }}
    QTreeWidget#workflowTree::branch:selected,
    QTreeWidget#workflowTree::branch:selected:active,
    QTreeWidget#workflowTree::branch:selected:!active {{
        background: {p.accent};
    }}
    QTreeWidget#workflowTree::branch:closed:has-children,
    QTreeWidget#workflowTree::branch:closed:has-children:has-siblings {{
        image: url("{branch_right}");
    }}
    QTreeWidget#workflowTree::branch:open:has-children,
    QTreeWidget#workflowTree::branch:open:has-children:has-siblings {{
        image: url("{branch_down}");
    }}
    QTreeWidget#workflowTree::branch:closed:selected:has-children,
    QTreeWidget#workflowTree::branch:closed:selected:has-children:has-siblings {{
        image: url("{branch_right_selected}");
    }}
    QTreeWidget#workflowTree::branch:open:selected:has-children,
    QTreeWidget#workflowTree::branch:open:selected:has-children:has-siblings {{
        image: url("{branch_down_selected}");
    }}
    """


def _scrollbar_styles(p: MuFinderPalette) -> str:
    return f"""
    QScrollBar:vertical {{
        background: transparent;
        width: 13px;
        margin: 4px 2px 4px 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {p.seal_gold};
        border: 1px solid {p.border};
        border-radius: {RADIUS.control}px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {p.accent};
        border-color: {p.accent_hover};
    }}
    QScrollBar::handle:vertical:pressed {{
        background: {p.accent_hover};
        border-color: {p.accent_pressed};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
        background: transparent;
        border: none;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: {p.bg_surface_alt};
        border-radius: {RADIUS.control}px;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 13px;
        margin: 2px 4px 2px 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {p.seal_gold};
        border: 1px solid {p.border};
        border-radius: {RADIUS.control}px;
        min-width: 28px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {p.accent};
        border-color: {p.accent_hover};
    }}
    QScrollBar::handle:horizontal:pressed {{
        background: {p.accent_hover};
        border-color: {p.accent_pressed};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
        background: transparent;
        border: none;
    }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: {p.bg_surface_alt};
        border-radius: {RADIUS.control}px;
    }}
    QTreeWidget QScrollBar:horizontal, QTableWidget QScrollBar:horizontal, QListWidget QScrollBar:horizontal {{
        margin-left: 8px;
        margin-right: 8px;
    }}
    """


def _header_styles(p: MuFinderPalette) -> str:
    return f"""
    QHeaderView::section {{
        background: {p.bg_surface_alt};
        border: 0;
        border-bottom: 1px solid {p.border};
        padding: {SPACING.xs}px;
        font-weight: 600;
    }}
    QHeaderView {{
        background: transparent;
    }}
    QTableWidget[hasVisibleVerticalHeader="false"] QHeaderView::section:horizontal:first {{
        border-top-left-radius: {RADIUS.control}px;
    }}
    QTreeWidget QHeaderView::section:horizontal:first {{
        border-top-left-radius: {RADIUS.control}px;
    }}
    QTableWidget[hasVisibleVerticalHeader="true"] QHeaderView::section:horizontal:first {{
        border-top-left-radius: 0px;
    }}
    QTableWidget[squareTopCorners="true"] QHeaderView::section:horizontal:first {{
        border-top-left-radius: 0px;
    }}
    QTreeWidget[squareTopCorners="true"] QHeaderView::section:horizontal:first {{
        border-top-left-radius: 0px;
    }}
    QTableWidget QHeaderView::section:horizontal:last {{
        border-top-right-radius: {RADIUS.control}px;
    }}
    QTreeWidget QHeaderView::section:horizontal:last {{
        border-top-right-radius: {RADIUS.control}px;
    }}
    QTableWidget[hasVerticalScrollbar="true"] QHeaderView::section:horizontal:last {{
        border-top-right-radius: 0px;
    }}
    QTreeWidget[hasVerticalScrollbar="true"] QHeaderView::section:horizontal:last {{
        border-top-right-radius: 0px;
    }}
    QTableWidget[squareTopCorners="true"] QHeaderView::section:horizontal:last {{
        border-top-right-radius: 0px;
    }}
    QTreeWidget[squareTopCorners="true"] QHeaderView::section:horizontal:last {{
        border-top-right-radius: 0px;
    }}
    """


def _button_styles(p: MuFinderPalette) -> str:
    # Standard dialog buttons (OK, Cancel, Yes, No, also inside QMessageBox)
    # stay plain like every other MuFinder button. Without this, a Linux
    # desktop's platform theme makes Fusion add an icon to each one.
    return f"""
    QDialogButtonBox {{
        dialogbuttonbox-buttons-have-icons: 0;
    }}
    /* The focus ring (_FocusRing) marks these controls, so Fusion's own faint
       focus frame is not drawn as well. Item views keep theirs: it marks the
       current cell or row inside the ring. */
    QPushButton, QCheckBox, QRadioButton, QComboBox {{
        outline: none;
    }}
    QToolButton#plotPopOutButton {{
        background: {p.bg_surface};
        border: 1px solid {p.border};
        border-radius: {RADIUS.control}px;
        padding: 0px;
        min-width: 20px;
        min-height: 20px;
    }}
    QToolButton#plotPopOutButton:hover {{
        border-color: {p.accent};
    }}
    QPushButton {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.control}px;
        padding: {SPACING.xs}px {SPACING.sm}px;
        min-height: {_STANDARD_BUTTON_HEIGHT - 2 * SPACING.xs - 2}px;
        background: {p.bg_surface};
    }}
    QPushButton:hover {{
        border-color: {p.accent};
        background: {p.bg_surface_alt};
    }}
    QPushButton:pressed {{
        background: {p.bg_surface_alt};
    }}
    QPushButton:disabled {{
        color: {p.text_muted};
    }}
    /* A row control that will act on the whole multi-row selection. The
       selected row behind it is filled with the accent colour, so the marker
       uses the text colour to stay visible, and the padding gives back the
       extra border pixel so the control (and its column) keeps its size. */
    QPushButton[multiRowTarget="true"],
    QToolButton[multiRowTarget="true"],
    QComboBox[multiRowTarget="true"] {{
        border: 2px solid {p.text_primary};
    }}
    QPushButton[multiRowTarget="true"] {{
        padding: {SPACING.xs - 1}px {SPACING.sm - 1}px;
    }}
    QComboBox[multiRowTarget="true"] {{
        padding: {SPACING.xs - 1}px {_COMBO_ARROW_GUTTER - 1}px {SPACING.xs - 1}px {SPACING.sm - 1}px;
    }}
    QCheckBox#rowCheckBox {{
        margin-left: 16px;
    }}
    QCheckBox#rowCheckBox[multiRowTarget="true"]::indicator {{
        border: 2px solid {p.text_primary};
    }}
    QPushButton#overviewActionButton {{
        border-color: {p.accent};
        background: {p.bg_surface};
        font-weight: 600;
        padding: {SPACING.xs}px {SPACING.sm}px;
        min-height: 0px;
    }}
    QPushButton#overviewActionButton:hover {{
        background: {p.bg_surface_alt};
    }}
    QToolButton#sectionToggleButton {{
        border: none;
        background: transparent;
        color: {p.text_primary};
        font-weight: 600;
        padding: {SPACING.xs}px {SPACING.xs}px;
    }}
    QToolButton#sectionToggleButton:hover {{
        color: {p.accent};
    }}
    QToolButton#sectionToggleButton:pressed {{
        color: {p.accent_hover};
    }}
    QToolButton#mufinderHelpButton {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.control}px;
        padding: {SPACING.xs}px;
        min-width: 24px;
        min-height: 24px;
        background: {p.bg_surface};
    }}
    QToolButton#mufinderHelpButton:hover {{
        border-color: {p.accent};
        background: {p.bg_surface_alt};
    }}
    QToolButton#mufinderHelpButton:pressed {{
        border-color: {p.accent_hover};
        background: {p.bg_surface_alt};
    }}
    QToolButton#mufinderCheckedToolButton {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.control}px;
        padding: {SPACING.xs}px;
        min-width: 24px;
        min-height: 24px;
        background: {p.bg_surface_alt};
    }}
    QToolButton#mufinderCheckedToolButton:checked {{
        border-color: {p.accent};
        background: {p.bg_surface};
    }}
    QToolButton#mufinderCheckedToolButton:hover {{
        border-color: {p.accent_hover};
    }}
    QToolButton#projectHomeButton {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.control}px;
        padding: {SPACING.xs}px;
        min-width: 24px;
        min-height: 24px;
        background: {p.bg_surface};
    }}
    QToolButton#projectHomeButton:hover {{
        border-color: {p.accent};
        background: {p.bg_surface_alt};
    }}
    QToolButton#projectHomeButton:pressed {{
        border-color: {p.accent_hover};
        background: {p.bg_surface_alt};
    }}
    QToolButton#projectHomeButton:disabled {{
        border-color: {p.border};
        background: {p.bg_surface_alt};
    }}
    """


def _label_styles(p: MuFinderPalette, heading_font: str) -> str:
    t = type_scale()
    return f"""
    QLabel#sectionTitle {{
        font-size: {t.section}px;
        font-weight: 600;
        font-family: "{heading_font}";
        color: {p.text_primary};
    }}
    QLabel#sectionSubtitle {{
        font-size: {t.body}px;
        color: {p.text_muted};
    }}
    QLabel#hintText {{
        font-size: {t.caption}px;
        color: {p.text_muted};
    }}
    QLabel#mixedValueMarker {{
        font-size: {t.caption}px;
        font-style: italic;
        color: {p.warning};
    }}
    QLabel#warningText {{
        color: {p.warning};
        font-weight: 600;
    }}
    QLabel#homeHeroTitle {{
        font-size: {t.page}px;
        font-weight: 700;
        font-family: "{heading_font}";
        color: {p.text_primary};
    }}
    QLabel#homeHeroSubtitle {{
        font-size: {t.emphasis}px;
        color: {p.text_muted};
    }}
    QLabel#cardTitle {{
        font-size: {t.emphasis}px;
        font-weight: 600;
        font-family: "{heading_font}";
    }}
    QLabel#cardBody {{
        font-size: {t.body}px;
        color: {p.text_muted};
    }}
    QLabel#overviewCardTitle {{
        font-size: {t.emphasis}px;
        font-weight: 700;
        font-family: "{heading_font}";
        color: {p.text_primary};
    }}
    QLabel#overviewSummaryValue {{
        color: {p.text_primary};
        font-size: {t.body}px;
    }}
    QLabel#overviewPlaceholder {{
        color: {p.text_muted};
        font-size: {t.body}px;
    }}
    QLabel#uepIsosurfacePanelTitle {{
        color: {p.text_primary};
        font-size: {t.emphasis}px;
        font-weight: 700;
        font-family: "{heading_font}";
    }}
    QLabel#uepIsosurfaceMinorLabel {{
        color: {p.text_muted};
        font-size: {t.caption}px;
        font-weight: 600;
    }}
    QLabel#uepIsosurfaceLegendTitle {{
        color: {p.text_primary};
        font-size: {t.emphasis}px;
        font-weight: 700;
        font-family: "{heading_font}";
    }}
    QLabel#uepIsosurfaceRangeValue {{
        color: {p.text_primary};
        font-size: {t.body}px;
        font-weight: 500;
    }}
    QLabel#uepIsosurfaceBound {{
        color: {p.text_muted};
        font-size: {t.caption}px;
        font-weight: 600;
    }}
    QLabel#uepIsosurfaceStatusLabel {{
        color: {p.text_muted};
        font-size: {t.body}px;
        font-weight: 500;
        line-height: 1.35em;
    }}
    QLabel#uepIsosurfaceStatusLabel[statusKind="busy"] {{
        color: {p.status_running};
    }}
    QLabel#uepIsosurfaceStatusLabel[statusKind="success"] {{
        color: {p.status_completed};
    }}
    QLabel#uepIsosurfaceStatusLabel[statusKind="exported"] {{
        color: {p.status_ready};
    }}
    QLabel#uepIsosurfaceStatusLabel[statusKind="error"] {{
        color: {p.status_failed};
    }}
    QLabel#expeditionTrailTitle {{
        font-size: {t.emphasis}px;
        font-weight: 700;
        font-family: "{heading_font}";
        color: {p.text_primary};
    }}
    QLabel#expeditionTrailHint {{
        font-size: {t.body}px;
        color: {p.text_muted};
    }}
    QLabel#expeditionLegendComplete {{
        color: {p.seal_gold};
        font-weight: 600;
    }}
    QLabel#expeditionLegendCurrent {{
        color: {p.accent};
        font-weight: 600;
    }}
    QLabel#expeditionLegendPending {{
        color: {p.text_muted};
        font-weight: 600;
    }}
    QLabel#journalDayHeader {{
        font-size: {t.section}px;
        font-weight: 700;
        font-family: "{heading_font}";
        color: {p.text_primary};
        padding-top: {SPACING.xs}px;
    }}
    QLabel#journalEntryTitle {{
        font-size: {t.body}px;
        font-weight: 700;
        font-family: "{heading_font}";
        color: {p.text_primary};
    }}
    QLabel#journalWorkflowLabel {{
        font-size: {t.caption}px;
        font-weight: 600;
        color: {p.text_muted};
    }}
    QLabel#journalChevron {{
        font-size: {t.caption}px;
        color: {p.text_muted};
    }}
    QLabel#journalEntryBody {{
        font-size: {t.body}px;
        color: {p.text_primary};
    }}
    QLabel#journalDetailTime {{
        font-size: {t.caption}px;
        color: {p.text_muted};
    }}
    QLabel#journalDetailLine {{
        font-size: {t.body}px;
        color: {p.text_primary};
    }}
    QLabel#journalEmptyState {{
        font-size: {t.body}px;
        color: {p.text_muted};
        padding: {SPACING.lg}px;
    }}
    QLabel#journalStamp {{
        min-width: 10px;
        max-width: 10px;
        min-height: 10px;
        max-height: 10px;
        border-radius: {ROUND_RADIUS.journal_stamp}px;
        background: {p.border};
    }}
    QLabel#journalStamp[stampKind="milestone"] {{
        background: {p.text_muted};
    }}
    QLabel#journalStamp[stampKind="success"] {{
        background: {p.seal_gold};
    }}
    QLabel#journalStamp[stampKind="warning"] {{
        background: {p.warning};
    }}
    QLabel#journalStamp[stampKind="danger"] {{
        background: {p.danger};
    }}
    """


def _card_and_menu_styles(p: MuFinderPalette) -> str:
    return f"""
    QFrame#homeCard {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.card}px;
        background: {p.bg_surface};
    }}
    QFrame#homeCard:hover {{
        border-color: {p.accent};
    }}
    QFrame#overviewCard {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.card}px;
        background: {p.bg_surface};
    }}
    QFrame#overviewPinnedZone {{
        border: 2px dashed {p.border};
        border-radius: {RADIUS.card}px;
        background: {p.bg_surface_alt};
    }}
    QFrame#muonSummaryViewerFrame {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.card}px;
        background: {p.bg_primary};
    }}
    QWidget#muonSummaryPanelBody {{
        background: transparent;
    }}
    QFrame#mufinderOptionCard {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.card}px;
        background: {p.bg_surface_alt};
    }}
    QFrame#mufinderOptionCard[selected="true"] {{
        border-color: {p.accent};
        background: {p.bg_surface};
    }}
    QFrame#mufinderOptionCard[unavailable="true"] {{
        border-color: {p.border};
        background: {p.bg_primary};
    }}
    QWidget#mufinderInlineGroup,
    QWidget#mufinderInlineGroup QLabel {{
        background: transparent;
    }}
    QFrame#analysisSliderPanel,
    QFrame#uepIsosurfaceInfoCard,
    QFrame#connectedToleranceScanPanel {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.card}px;
        background: {p.bg_surface};
    }}
    QFrame#uepIsosurfaceStatusCard {{
        border: 1px solid {p.border};
        border-radius: {RADIUS.card}px;
        background: {p.bg_surface};
    }}
    QFrame#uepIsosurfaceStatusCard[statusKind="busy"] {{
        border-color: {p.status_running};
    }}
    QFrame#uepIsosurfaceStatusCard[statusKind="success"] {{
        border-color: {p.status_completed};
    }}
    QFrame#uepIsosurfaceStatusCard[statusKind="exported"] {{
        border-color: {p.status_ready};
    }}
    QFrame#uepIsosurfaceStatusCard[statusKind="error"] {{
        border-color: {p.status_failed};
    }}
    QFrame#journalEntryCard {{
        border: 1px solid {p.border};
        border-left: 4px solid {p.border};
        border-radius: {RADIUS.card}px;
        background: {p.bg_surface};
    }}
    QWidget#journalCardHeader, QWidget#journalEntryDetails {{
        background: transparent;
    }}
    QFrame#journalEntryCard:focus {{
        border-top-color: {p.text_primary};
        border-right-color: {p.text_primary};
        border-bottom-color: {p.text_primary};
    }}
    QFrame#journalEntryCard[stampKind="milestone"] {{
        border-left-color: {p.border};
    }}
    QFrame#journalEntryCard[stampKind="success"] {{
        border-left-color: {p.seal_gold};
    }}
    QFrame#journalEntryCard[stampKind="warning"] {{
        border-left-color: {p.warning};
    }}
    QFrame#journalEntryCard[stampKind="danger"] {{
        border-left-color: {p.danger};
    }}
    QMenu::item:selected {{
        background: {p.accent};
        color: #FFFFFF;
    }}
    /* A disabled action is drawn muted and takes no hover highlight, so it
       never looks clickable; its tooltip says why it is unavailable. */
    QMenu::item:disabled, QMenu::item:disabled:selected {{
        background: transparent;
        color: {p.text_muted};
    }}
    """


def _indicator_styles(
    p: MuFinderPalette,
    check_glyph: str,
    check_muted_glyph: str,
    dash_glyph: str,
    dot_glyph: str,
) -> str:
    # The same treatment is applied to standalone QCheckBox/QRadioButton widgets
    # and to checkable item-view rows (QTreeView/QTableView/QListView indicators),
    # so a table's checkboxes match the form checkboxes instead of falling back
    # to the plain Fusion indicator.
    return f"""
    QCheckBox::indicator, QRadioButton::indicator,
    QTreeView::indicator, QTableView::indicator, QListView::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {p.border};
        background: {p.bg_surface};
    }}
    QCheckBox::indicator,
    QTreeView::indicator, QTableView::indicator, QListView::indicator {{
        border-radius: {RADIUS.control}px;
    }}
    QRadioButton::indicator {{
        border-radius: {ROUND_RADIUS.radio_indicator}px;
    }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border-color: {p.accent};
    }}
    QCheckBox::indicator:checked,
    QTreeView::indicator:checked, QTableView::indicator:checked, QListView::indicator:checked {{
        background: {p.accent};
        border-color: {p.accent};
        image: url("{check_glyph}");
    }}
    QCheckBox::indicator:checked:hover {{
        background: {p.accent_hover};
        border-color: {p.accent_hover};
    }}
    QCheckBox::indicator:indeterminate,
    QTreeView::indicator:indeterminate, QTableView::indicator:indeterminate, QListView::indicator:indeterminate {{
        background: {p.accent};
        border-color: {p.accent};
        image: url("{dash_glyph}");
    }}
    QRadioButton::indicator:checked {{
        border-color: {p.accent};
        image: url("{dot_glyph}");
    }}
    QRadioButton::indicator:checked:hover {{
        border-color: {p.accent_hover};
    }}
    QCheckBox::indicator:disabled, QRadioButton::indicator:disabled,
    QTreeView::indicator:disabled, QTableView::indicator:disabled, QListView::indicator:disabled {{
        background: {p.bg_surface_alt};
        border-color: {p.border};
    }}
    QCheckBox::indicator:checked:disabled,
    QTreeView::indicator:checked:disabled,
    QTableView::indicator:checked:disabled,
    QListView::indicator:checked:disabled {{
        image: url("{check_muted_glyph}");
    }}
    QCheckBox:disabled, QRadioButton:disabled {{
        color: {p.text_muted};
    }}
    """


def _build_stylesheet(p: MuFinderPalette) -> str:
    heading_font = heading_font_family()
    arrow_down = _arrow_asset_path("down", p.text_primary)
    arrow_up = _arrow_asset_path("up", p.text_primary)
    arrow_right = _arrow_asset_path("right", p.text_primary)
    selected_arrow_down = _arrow_asset_path("down", "#FFFFFF")
    selected_arrow_right = _arrow_asset_path("right", "#FFFFFF")
    check_glyph = _indicator_glyph_path("check", "#FFFFFF")
    check_muted_glyph = _indicator_glyph_path("check", p.text_muted)
    dash_glyph = _indicator_glyph_path("dash", "#FFFFFF")
    dot_glyph = _indicator_glyph_path("dot", p.accent)
    sections = (
        _base_widget_styles(p),
        _input_control_styles(p, arrow_up, arrow_down),
        _indicator_styles(p, check_glyph, check_muted_glyph, dash_glyph, dot_glyph),
        _item_view_styles(p),
        _panel_container_styles(p),
        _muon_generation_styles(p),
        _muon_relaxation_styles(p),
        _magnetic_field_styles(p),
        _workflow_tree_styles(
            p,
            branch_right=arrow_right,
            branch_down=arrow_down,
            branch_right_selected=selected_arrow_right,
            branch_down_selected=selected_arrow_down,
        ),
        _scrollbar_styles(p),
        _header_styles(p),
        _button_styles(p),
        _label_styles(p, heading_font),
        _card_and_menu_styles(p),
    )
    return "\n".join(sections)


def apply_base_style(app: QApplication) -> None:
    """Force the shared Fusion base style for cross-platform consistency.

    Without this Qt uses each platform's native style (macOS/Windows/Linux),
    which leaks through on controls the stylesheet does not fully override and
    makes the same UI look different per OS. Fusion is the same everywhere and
    is fully styleable, so MuFinder renders identically on every platform.

    Idempotent per application instance: re-applying the style would needlessly
    rebuild and re-polish every widget on each theme toggle, so we set it once.
    """

    global _BASE_STYLE_APP

    if _BASE_STYLE_APP is app:
        return
    app.setStyle(_BASE_STYLE)
    _BASE_STYLE_APP = app


def apply_theme(app: QApplication, mode: str) -> str:
    """Apply MuFinder look-and-feel and return resolved light/dark mode."""

    global _CURRENT_PALETTE, _CURRENT_TYPE_SCALE

    ensure_fonts_loaded()
    apply_base_style(app)

    resolved = resolve_theme_mode(mode)
    _CURRENT_PALETTE = dark_palette() if resolved == THEME_DARK else light_palette()
    _CURRENT_TYPE_SCALE = _scaled_type_scale(text_scale())

    app.setPalette(_build_palette(_CURRENT_PALETTE))
    app.setStyleSheet(_build_stylesheet(_CURRENT_PALETTE))
    _ensure_table_corner_styler(app)
    _ensure_focus_ring_tracker(app)

    font = QFont(body_font_family())
    # Pixel size, not point size: the stylesheet declares every label size in
    # px, and point-to-pixel conversion differs per platform (macOS logical
    # 72 dpi vs Windows 96 dpi), which made Windows body text ~33% larger
    # than every px-sized label. The desktop's text size reaches both through
    # type_scale().
    font.setPixelSize(type_scale().caption)
    app.setFont(font)
    return resolved
