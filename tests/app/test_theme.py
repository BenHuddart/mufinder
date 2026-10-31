"""Tests for centralized app theming helpers."""

from __future__ import annotations

import dataclasses
import itertools
import math
import re

from matplotlib.figure import Figure
import pytest

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QSettings, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProxyStyle,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app import theme as theme_module
from mufinder.app import workflow_table_utils


def test_theme_mode_roundtrip() -> None:
    settings = QSettings(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, "MuFinder", "ThemeRoundtrip"
    )
    settings.clear()

    theme_module.save_theme_mode(settings, theme_module.THEME_DARK)
    assert theme_module.load_theme_mode(settings) == theme_module.THEME_DARK

    theme_module.save_theme_mode(settings, "invalid")
    assert theme_module.load_theme_mode(settings) == theme_module.THEME_SYSTEM


def test_apply_theme_sets_active_palette(qapp) -> None:
    resolved = theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    assert resolved == theme_module.THEME_LIGHT
    palette = theme_module.current_palette()
    assert palette.mode == theme_module.THEME_LIGHT
    assert palette.accent == "#E02020"
    assert palette.seal_gold == "#B8860B"

    stylesheet = qapp.styleSheet()
    assert theme_module.heading_font_family() in stylesheet
    assert "QFrame#projectTreePanel" in stylesheet
    assert "QLabel#projectTreePanelTitle" in stylesheet
    assert "QFrame#projectTreePanelBody" in stylesheet
    assert "QDockWidget#jobDashboardDock::title" in stylesheet
    assert 'QTableWidget[squareTopCorners="true"]' in stylesheet
    assert 'QTableWidget[hasVerticalScrollbar="true"] {' in stylesheet
    assert "QTreeWidget#workflowTree" in stylesheet
    assert "show-decoration-selected: 1;" in stylesheet
    assert "QTreeWidget#workflowTree::item:selected" in stylesheet
    assert "QTreeWidget#workflowTree::branch:selected" in stylesheet
    assert "QTreeWidget#workflowTree::branch:closed:has-children" in stylesheet
    assert "QToolButton#sectionToggleButton" in stylesheet
    assert "QToolButton#mufinderHelpButton" in stylesheet
    assert "QToolButton#mufinderCheckedToolButton" in stylesheet
    assert "QFrame#mufinderOptionCard" in stylesheet
    assert "QWidget#mufinderInlineGroup" in stylesheet
    assert (
        'QTableWidget[hasVerticalScrollbar="true"] QHeaderView::section:horizontal:last'
        in stylesheet
    )
    assert (
        'QTableWidget[hasVisibleVerticalHeader="true"] QHeaderView::section:horizontal:first'
        in stylesheet
    )
    assert 'QTreeWidget[squareTopCorners="true"]' in stylesheet
    assert (
        'QTreeWidget[hasVerticalScrollbar="true"] QHeaderView::section:horizontal:last'
        in stylesheet
    )
    assert "QComboBox::drop-down" in stylesheet
    assert "QComboBox::down-arrow" in stylesheet
    assert "QSpinBox::up-button" in stylesheet
    assert "QSpinBox::up-arrow" in stylesheet
    assert "QDoubleSpinBox::up-button" in stylesheet
    assert "QDoubleSpinBox::up-arrow" in stylesheet
    assert "QGroupBox#muonGenerationSetupCard" in stylesheet
    assert 'QWidget#randomSeparationChoice[active="true"]' in stylesheet
    assert "QDialog#clusterPositionsDialog" in stylesheet
    assert "QWidget#clusterPositionsCard" in stylesheet
    assert "QFrame#connectedToleranceScanPanel" in stylesheet
    assert "QTabWidget#muonRelaxationChargeTabs" in stylesheet
    assert "QTabWidget#muonRelaxationChargeTabs QTabBar" in stylesheet
    assert "QTabWidget#analysisWorkspaceTabs" in stylesheet
    assert "QTabWidget#analysisWorkspaceTabs QTabBar" in stylesheet
    assert "QDialog#muonChargeStateDialog" in stylesheet
    assert "QWidget#muonChargeStateCard" in stylesheet
    assert "QProgressBar#muonGenerationProgress" in stylesheet
    assert 'QProgressBar#muonGenerationProgress[generationState="complete"]::chunk' in stylesheet
    assert qapp.font().family() == theme_module.body_font_family()
    # The default font must be sized in pixels, not points: stylesheet sizes
    # are all px, and pt-to-px conversion differs between macOS (72 dpi) and
    # Windows (96 dpi), which previously inflated Windows body text by ~33%.
    assert qapp.font().pixelSize() == 11
    assert qapp.font().pointSize() == -1


def test_adjust_font_size_handles_pixel_point_and_unsized_fonts(qapp) -> None:
    from PySide6.QtGui import QFont

    pixel_font = QFont()
    pixel_font.setPixelSize(11)
    theme_module.adjust_font_size(pixel_font, 2)
    assert pixel_font.pixelSize() == 13

    point_font = QFont()
    point_font.setPointSize(10)
    theme_module.adjust_font_size(point_font, -1)
    assert point_font.pointSize() == 9

    clamped = QFont()
    clamped.setPixelSize(10)
    theme_module.adjust_font_size(clamped, -5, minimum=9)
    assert clamped.pixelSize() == 9


def test_bundled_font_assets_exist() -> None:
    font_dir = theme_module._bundled_font_dir()

    assert font_dir.is_dir()
    expected = {
        "Inter-Regular.ttf",
        "Inter-Medium.ttf",
        "Inter-SemiBold.ttf",
        "Inter-Bold.ttf",
        "Fraunces-Regular.ttf",
        "Fraunces-SemiBold.ttf",
        "Fraunces-Bold.ttf",
    }
    present = {path.name for path in font_dir.glob("*.ttf")}
    assert expected <= present
    # OFL license text must ship alongside the fonts for redistribution.
    licenses = {path.name for path in font_dir.glob("*.txt")}
    assert {"Inter-OFL.txt", "Fraunces-OFL.txt"} <= licenses


def test_ensure_fonts_loaded_registers_bundled_families(qapp) -> None:
    theme_module.ensure_fonts_loaded()

    families = set(QFontDatabase.families())
    assert theme_module._BUNDLED_BODY_FAMILY in families
    assert theme_module._BUNDLED_HEADING_FAMILY in families
    # Idempotent: calling again is a no-op and keeps the families present.
    theme_module.ensure_fonts_loaded()
    assert theme_module._BUNDLED_HEADING_FAMILY in set(QFontDatabase.families())


def test_ensure_fonts_loaded_registers_bundled_files_even_when_family_present(
    qapp, monkeypatch
) -> None:
    # A host machine may already expose families named "Inter"/"Fraunces". The
    # loader must still register the pinned bundled files (so the shipped weights
    # drive the look) instead of deferring to whatever the host has installed.
    theme_module.ensure_fonts_loaded()
    assert theme_module._BUNDLED_BODY_FAMILY in set(QFontDatabase.families())

    theme_module._FONTS_LOADED = False
    registered: list[str] = []
    real_add = QFontDatabase.addApplicationFont

    def _spy(path: str) -> int:
        registered.append(path)
        return real_add(path)

    monkeypatch.setattr(QFontDatabase, "addApplicationFont", _spy)
    # If the loader ever enumerates the font DB on the hot path, fail loudly.
    monkeypatch.setattr(
        QFontDatabase,
        "families",
        staticmethod(
            lambda *a, **k: pytest.fail("ensure_fonts_loaded must not enumerate families")
        ),
    )

    theme_module.ensure_fonts_loaded()

    assert registered, "bundled font files must be registered regardless of host fonts"
    assert all(path.endswith(".ttf") for path in registered)
    assert theme_module._FONTS_LOADED is True

    # Subsequent calls short-circuit without touching the font database again.
    registered.clear()
    theme_module.ensure_fonts_loaded()
    assert registered == []


def test_apply_theme_uses_bundled_typography(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    assert theme_module.body_font_family() == theme_module._BUNDLED_BODY_FAMILY
    assert theme_module.heading_font_family() == theme_module._BUNDLED_HEADING_FAMILY
    assert qapp.font().family() == theme_module._BUNDLED_BODY_FAMILY


def test_apply_base_style_forces_fusion(qapp) -> None:
    previous_sheet = qapp.styleSheet()
    try:
        qapp.setStyleSheet("")  # drop the wrapper so the base style is observable
        theme_module._BASE_STYLE_APP = None
        theme_module.apply_base_style(qapp)

        assert qapp.style().metaObject().className() == "QFusionStyle"
        # Marked applied so a redundant second call is skipped (avoids re-polish).
        assert theme_module._BASE_STYLE_APP is qapp
        theme_module.apply_base_style(qapp)
        assert theme_module._BASE_STYLE_APP is qapp
    finally:
        qapp.setStyleSheet(previous_sheet)


def test_apply_theme_styles_checkbox_and_radio_indicators(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    stylesheet = qapp.styleSheet()
    accent = theme_module.current_palette().accent

    # Indicators are explicitly styled (so they render identically on every OS
    # via Fusion) and tinted with the brand accent rather than left Fusion-gray.
    assert "QCheckBox::indicator" in stylesheet
    assert "QRadioButton::indicator" in stylesheet
    assert "QCheckBox::indicator:checked" in stylesheet
    assert "QRadioButton::indicator:checked" in stylesheet
    assert "QCheckBox::indicator:indeterminate" in stylesheet
    assert accent in stylesheet  # accent fill/border on the checked indicators
    # Checkable item-view rows get the same branded indicators as form widgets.
    assert "QTreeView::indicator:checked" in stylesheet
    assert "QTableView::indicator:checked" in stylesheet
    # Disabled+checked uses a muted tick instead of the low-contrast white one.
    assert "QCheckBox::indicator:checked:disabled" in stylesheet


def test_indicator_glyph_assets_are_generated(qapp) -> None:
    from pathlib import Path

    for kind, colour in (("check", "#FFFFFF"), ("dash", "#FFFFFF"), ("dot", "#E02020")):
        path = Path(theme_module._indicator_glyph_path(kind, colour))
        assert path.exists()
        svg = path.read_text(encoding="utf-8")
        assert svg.startswith("<svg")
        assert colour.lower() in svg.lower()


def test_stylesheet_asset_urls_never_contain_backslashes(qapp) -> None:
    # Qt's CSS parser treats a backslash inside url("…") as an escape
    # character: on Windows a native path both breaks the image reference and
    # corrupts parsing of every rule after it (dropping e.g. the heading font
    # rules). Asset paths must therefore always be emitted with forward slashes.
    from pathlib import Path

    for direction in ("up", "down", "left", "right"):
        path = theme_module._arrow_asset_path(direction, "#2C2418")
        assert "\\" not in path
        assert Path(path).exists()

    for kind in ("check", "dash", "dot"):
        path = theme_module._indicator_glyph_path(kind, "#FFFFFF")
        assert "\\" not in path
        assert Path(path).exists()

    for palette in (theme_module.light_palette(), theme_module.dark_palette()):
        assert "\\" not in theme_module._build_stylesheet(palette)


def test_arrow_asset_supports_left_direction(qapp) -> None:
    from pathlib import Path

    path = Path(theme_module._arrow_asset_path("left", "#E02020"))
    assert path.exists()
    svg = path.read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "#e02020" in svg.lower()


def test_arrow_icon_returns_non_null_icon_for_each_direction(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    for direction in ("up", "down", "left", "right"):
        icon = theme_module.arrow_icon(direction)
        assert not icon.isNull()


def test_status_brush_uses_palette_tokens(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)

    brush = theme_module.status_brush("failed")
    colour = brush.color().name().lower()

    assert colour == theme_module.current_palette().status_failed.lower()


def test_lifecycle_status_colours_distinguish_ready_queued_and_running(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    light_palette = theme_module.current_palette()
    assert light_palette.status_ready == "#0369A1"
    assert light_palette.status_queued == "#A16207"
    assert light_palette.status_running == "#2563EB"
    assert theme_module.status_brush("ready").color().name().lower() == "#0369a1"
    assert theme_module.status_brush("queued").color().name().lower() == "#a16207"
    assert theme_module.status_brush("running").color().name().lower() == "#2563eb"

    theme_module.apply_theme(qapp, theme_module.THEME_DARK)

    dark_palette = theme_module.current_palette()
    assert dark_palette.status_ready == "#7DD3FC"
    assert dark_palette.status_queued == "#FBBF24"
    assert dark_palette.status_running == "#60A5FA"
    assert theme_module.status_brush("ready").color().name().lower() == "#7dd3fc"
    assert theme_module.status_brush("queued").color().name().lower() == "#fbbf24"
    assert theme_module.status_brush("running").color().name().lower() == "#60a5fa"


def test_feature_dialog_style_helpers_tag_reusable_widgets(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    inline = QWidget()
    card = QFrame()
    help_button = QToolButton()
    checked_button = QToolButton()

    theme_module.style_inline_group(inline)
    theme_module.style_option_card(card)
    theme_module.style_help_button(help_button)
    theme_module.style_small_checked_tool_button(checked_button)
    theme_module.set_option_card_state(card, selected=True, unavailable=True)

    assert inline.objectName() == "mufinderInlineGroup"
    assert card.objectName() == "mufinderOptionCard"
    assert card.property("selected") is True
    assert card.property("unavailable") is True
    assert help_button.objectName() == "mufinderHelpButton"
    assert checked_button.objectName() == "mufinderCheckedToolButton"


def test_feature_dialog_stylesheet_uses_shared_feature_selectors(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)

    stylesheet = theme_module.feature_dialog_stylesheet(include_scroll_area=True)

    assert "QFrame#mufinderOptionCard" in stylesheet
    assert "QWidget#mufinderInlineGroup" in stylesheet
    assert "QToolButton#mufinderCheckedToolButton" in stylesheet
    assert "QScrollArea" in stylesheet


def test_dark_palette_uses_night_expedition_accent(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)

    palette = theme_module.current_palette()

    assert palette.accent == "#E8A838"
    assert palette.seal_gold == "#D4A853"


def test_workflow_table_factory_tracks_vertical_scrollbar(qapp) -> None:
    from mufinder.app.workflow_table_utils import create_workflow_table

    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    table = create_workflow_table(("Name",), show_vertical_header=True)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.resize(320, 140)
    table.show()
    qapp.processEvents()

    assert table.property("hasVerticalScrollbar") == "false"
    assert table.property("hasVisibleVerticalHeader") == "true"

    table.setRowCount(80)
    qapp.processEvents()
    assert table.property("hasVerticalScrollbar") == "true"

    table.setRowCount(1)
    qapp.processEvents()
    assert table.property("hasVerticalScrollbar") == "false"


def test_tracked_table_follows_vertical_header_visibility(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    table = QTableWidget(0, 1)
    theme_module.track_item_view_corners(table)
    table.show()
    qapp.processEvents()

    assert table.property("hasVisibleVerticalHeader") == "true"

    table.verticalHeader().setVisible(False)
    qapp.processEvents()
    assert table.property("hasVisibleVerticalHeader") == "false"

    table.verticalHeader().setVisible(True)
    qapp.processEvents()
    assert table.property("hasVisibleVerticalHeader") == "true"


def test_apply_theme_attaches_tables_built_before_the_theme(qapp) -> None:
    theme_module.cleanup_theme_runtime(qapp)

    table = QTableWidget(0, 1)
    theme_module.track_item_view_corners(table)
    table.resize(320, 140)
    table.show()
    qapp.processEvents()
    assert table.property("hasVerticalScrollbar") is None

    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    assert table.property("hasVerticalScrollbar") == "false"
    assert table.property("hasVisibleVerticalHeader") == "true"

    table.setRowCount(80)
    qapp.processEvents()
    assert table.property("hasVerticalScrollbar") == "true"


def test_reapplied_theme_reattaches_tables_after_cleanup(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    table = QTableWidget(0, 1)
    theme_module.track_item_view_corners(table)
    table.resize(320, 140)
    table.show()
    qapp.processEvents()

    theme_module.cleanup_theme_runtime(qapp)
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    table.setRowCount(80)
    qapp.processEvents()
    assert table.property("hasVerticalScrollbar") == "true"


def test_untracked_table_is_not_watched_by_an_application_filter(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    table = QTableWidget(0, 1)
    table.show()
    qapp.processEvents()

    assert table.property("hasVerticalScrollbar") is None


def test_every_app_item_view_is_attached_to_the_corner_styler() -> None:
    """Every QTableWidget/QTreeWidget built in ``mufinder.app`` opts into corner tracking.

    No application-wide event filter discovers item views, so a module that builds
    one must attach it through ``track_item_view_corners`` or a helper that calls it.
    """

    import re
    from pathlib import Path

    app_dir = Path(theme_module.__file__).parent
    constructs_view = re.compile(r"\bQ(?:Table|Tree)Widget\(|class \w+\(Q(?:Table|Tree)Widget\)")
    attaches_view = re.compile(
        r"\b(?:track_item_view_corners|use_square_top_table_corners|use_tabbed_table_corners)\("
    )
    missing = sorted(
        path.name
        for path in app_dir.glob("*.py")
        if path.name != "theme.py"
        and constructs_view.search(text := path.read_text(encoding="utf-8"))
        and len(attaches_view.findall(text)) < len(constructs_view.findall(text))
    )

    assert missing == []


def test_use_square_top_table_corners_sets_property(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    table = QTableWidget(0, 1)
    theme_module.use_square_top_table_corners(table)

    assert table.property("squareTopCorners") == "true"


def test_use_tabbed_table_corners_sets_tabbed_and_square_properties(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    table = QTableWidget(0, 1)
    theme_module.use_tabbed_table_corners(table)

    assert table.property("tabbedTable") == "true"
    assert table.property("squareTopCorners") == "true"


def test_square_corner_helper_tracks_tree_vertical_scrollbar(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    tree = QTreeWidget()
    tree.setColumnCount(1)
    tree.setHeaderLabels(["Name"])
    theme_module.use_square_top_table_corners(tree)
    tree.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    tree.resize(320, 140)
    tree.show()
    qapp.processEvents()

    assert tree.property("hasVerticalScrollbar") == "false"
    assert tree.property("hasVisibleVerticalHeader") == "false"

    for index in range(80):
        tree.addTopLevelItem(QTreeWidgetItem([f"Row {index}"]))
    qapp.processEvents()
    assert tree.property("hasVerticalScrollbar") == "true"


def test_use_square_top_table_corners_supports_tree_widgets(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    tree = QTreeWidget()
    tree.setColumnCount(1)
    theme_module.use_square_top_table_corners(tree)

    assert tree.property("squareTopCorners") == "true"


def test_style_tab_widget_uses_the_shared_tab_rules(qapp) -> None:
    from PySide6.QtWidgets import QTabWidget

    tabs = QTabWidget()
    tabs.addTab(QWidget(), "A long tab title")
    theme_module.style_tab_widget(tabs)

    assert tabs.objectName() == "mufinderTabs"
    assert tabs.tabBar().expanding() is False
    assert tabs.elideMode() == Qt.TextElideMode.ElideRight
    stylesheet = theme_module._build_stylesheet(theme_module.light_palette())
    assert "QTabWidget#mufinderTabs QTabBar::tab:selected" in stylesheet
    assert "QTabWidget#mufinderTabs > QStackedWidget > QWidget" in stylesheet


def test_checkboxes_and_radio_buttons_take_their_parent_background() -> None:
    stylesheet = theme_module._build_stylesheet(theme_module.light_palette())

    assert "QLabel, QCheckBox, QRadioButton {\n        background: transparent;" in stylesheet


@pytest.mark.parametrize("palette_factory", ["light_palette", "dark_palette"])
def test_disabled_menu_items_are_muted_and_never_highlighted(palette_factory: str) -> None:
    palette = getattr(theme_module, palette_factory)()
    stylesheet = theme_module._build_stylesheet(palette)

    rule = stylesheet[stylesheet.index("QMenu::item:disabled, QMenu::item:disabled:selected") :]
    block = rule[: rule.index("}")]
    assert f"color: {palette.text_muted};" in block
    assert "background: transparent;" in block
    # Declared after the hover highlight, so it wins for a hovered disabled item.
    assert stylesheet.index("QMenu::item:selected {") < stylesheet.index("QMenu::item:disabled")


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_disabled_inputs_and_labels_are_drawn_muted(qapp, mode: str) -> None:
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QLabel, QLineEdit, QWidget

    theme_module.apply_theme(qapp, mode)
    try:
        muted = QColor(theme_module.current_palette().text_muted)
        surface_alt = QColor(theme_module.current_palette().bg_surface_alt)
        container = QWidget()
        label = QLabel("Normal (h k l)", container)
        edit = QLineEdit("1", container)
        container.setEnabled(False)
        for widget in (label, edit):
            widget.ensurePolished()

        disabled = QPalette.ColorGroup.Disabled
        assert label.palette().color(disabled, QPalette.ColorRole.WindowText) == muted
        assert edit.palette().color(disabled, QPalette.ColorRole.Text) == muted
        assert edit.palette().color(disabled, QPalette.ColorRole.Base) == surface_alt
        container.deleteLater()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_style_plot_axes_paints_axis_in_the_active_palette(qapp, mode: str) -> None:
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure

    theme_module.apply_theme(qapp, mode)
    palette = theme_module.current_palette()
    axis = Figure().add_subplot(111)
    axis.set_xlabel("x")

    theme_module.style_plot_axes(axis)
    axis.set_ylabel("y")
    axis.set_title("title")

    assert axis.get_facecolor() == to_rgba(palette.bg_surface_alt)
    assert axis.xaxis.label.get_color() == palette.text_primary
    assert axis.yaxis.label.get_color() == palette.text_primary
    assert axis.title.get_color() == palette.text_primary
    assert all(spine.get_edgecolor() == to_rgba(palette.border) for spine in axis.spines.values())
    tick = axis.xaxis.get_major_ticks()[0]
    assert tick.label1.get_color() == palette.text_primary
    gridline = tick.gridline
    assert gridline.get_visible()
    assert gridline.get_color() == palette.border
    assert gridline.get_alpha() == pytest.approx(0.35)


def test_style_plot_axes_accepts_per_plot_overrides(qapp) -> None:
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure

    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    palette = theme_module.current_palette()
    axis = Figure().add_subplot(111)

    theme_module.style_plot_axes(
        axis,
        facecolor=palette.bg_primary,
        tick_color=palette.text_muted,
        tick_labelsize=9,
        grid_alpha=0.45,
        grid_linewidth=0.7,
    )

    assert axis.get_facecolor() == to_rgba(palette.bg_primary)
    tick = axis.yaxis.get_major_ticks()[0]
    assert tick.label1.get_color() == palette.text_muted
    assert tick.label1.get_fontsize() == 9
    assert tick.gridline.get_alpha() == pytest.approx(0.45)
    assert tick.gridline.get_linewidth() == pytest.approx(0.7)


def test_style_plot_legend_paints_frame_and_text(qapp) -> None:
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure

    theme_module.apply_theme(qapp, theme_module.THEME_DARK)
    palette = theme_module.current_palette()
    axis = Figure().add_subplot(111)
    axis.plot([0, 1], [0, 1], label="line")
    legend = axis.legend(frameon=True)

    theme_module.style_plot_legend(legend, frame_alpha=0.95)

    frame = legend.get_frame()
    assert frame.get_facecolor()[:3] == to_rgba(palette.bg_surface)[:3]
    assert frame.get_edgecolor()[:3] == to_rgba(palette.border)[:3]
    assert frame.get_alpha() == pytest.approx(0.95)
    assert [text.get_color() for text in legend.get_texts()] == [palette.text_primary]
    theme_module.style_plot_legend(None)


class _DesktopIconsStyle(QProxyStyle):
    """Fusion as a Linux desktop's platform theme sets it: standard buttons get icons."""

    def __init__(self) -> None:
        super().__init__("Fusion")

    def styleHint(self, hint, option=None, widget=None, returnData=None) -> int:  # noqa: N802
        if hint == QStyle.StyleHint.SH_DialogButtonBox_ButtonsHaveIcons:
            return 1
        return super().styleHint(hint, option, widget, returnData)


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_apply_theme_keeps_dialog_standard_buttons_plain(qapp, mode) -> None:
    previous_sheet = qapp.styleSheet()
    previous_mode = theme_module.current_palette().mode
    theme_module.apply_theme(qapp, mode)
    # Swap in a desktop-like base style; Qt keeps the app stylesheet on top of it.
    qapp.setStyle(_DesktopIconsStyle())
    widgets = []
    try:
        dialog = QDialog()
        widgets.append(dialog)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        box = QMessageBox(QMessageBox.Icon.Question, "Remove", "Remove it?")
        widgets.append(box)
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        standard_buttons = [*buttons.buttons(), *box.buttons()]
        assert len(standard_buttons) == 4
        assert all(button.icon().isNull() for button in standard_buttons)
    finally:
        for widget in widgets:
            widget.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.setStyle("Fusion")  # the base style apply_theme installs
        theme_module.apply_theme(qapp, previous_mode)
        qapp.setStyleSheet(previous_sheet)


def test_feature_dialog_stylesheet_emphasises_radio_buttons_on_request(qapp) -> None:
    plain = theme_module.feature_dialog_stylesheet()
    emphasised = theme_module.feature_dialog_stylesheet(emphasised_radio_buttons=True)

    assert "font-weight: 600;" not in plain.split("QRadioButton {")[1].split("}")[0]
    assert "font-weight: 600;" in emphasised.split("QRadioButton {")[-1].split("}")[0]


def test_magnetic_field_stylesheets_hold_no_colours() -> None:
    """Set on a widget, a colour would stay behind when the theme changes."""

    for stylesheet in (
        theme_module.magnetic_fields_stylesheet(),
        theme_module.magnetic_structure_builder_stylesheet(),
    ):
        assert "#" not in stylesheet.replace("QWidget#", "")
        assert "color" not in stylesheet


def test_nuclear_direction_inputs_follow_the_theme(qapp) -> None:
    previous_sheet = qapp.styleSheet()
    try:
        for mode, palette in (
            (theme_module.THEME_LIGHT, theme_module.light_palette()),
            (theme_module.THEME_DARK, theme_module.dark_palette()),
        ):
            theme_module.apply_theme(qapp, mode)
            rule = qapp.styleSheet().split("QLineEdit#nuclearDirectionInput {")[1].split("}")[0]
            assert f"background: {palette.bg_surface};" in rule
            assert f"color: {palette.text_primary};" in rule
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
        qapp.setStyleSheet(previous_sheet)


def test_texture_preview_canvas_colours_follow_the_canvas_not_the_theme(qapp) -> None:
    light, dark = theme_module.light_palette(), theme_module.dark_palette()
    previous_sheet = qapp.styleSheet()
    try:
        for mode, palette in (
            (theme_module.THEME_LIGHT, light),
            (theme_module.THEME_DARK, dark),
        ):
            theme_module.apply_theme(qapp, mode)
            light_canvas = theme_module.texture_preview_colors("light")
            dark_canvas = theme_module.texture_preview_colors("dark")

            assert light_canvas.surround == dark_canvas.surround == palette.bg_surface
            assert light_canvas.plot_background == "#FFFDF8"
            assert (light_canvas.frame, light_canvas.placeholder_text) == (
                light.border,
                light.text_muted,
            )
            assert light_canvas.zoom_box == light_canvas.magnitude_high == light.accent
            assert light_canvas.positive == light.seal_gold
            assert dark_canvas.plot_background == "#111827"
            assert (dark_canvas.frame, dark_canvas.placeholder_text) == (
                dark.border,
                dark.text_muted,
            )
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
        qapp.setStyleSheet(previous_sheet)


def _contrast_ratio(foreground: str, background: str) -> float:
    def luminance(colour: str) -> float:
        channels = [int(colour.lstrip("#")[index : index + 2], 16) / 255 for index in (0, 2, 4)]
        linear = [
            value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
            for value in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _plot_series_roles(colours) -> dict[str, str]:
    return {
        role: getattr(colours, role)
        for role in (
            "energy",
            "force",
            "stress",
            "displacement",
            "threshold",
            "reference",
            "total",
            "fit",
            "converged",
            "stress_converged",
        )
    } | {f"categorical[{index}]": colour for index, colour in enumerate(colours.categorical)}


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_plot_series_colours_contrast_with_the_plot_backgrounds(qapp, mode) -> None:
    theme_module.apply_theme(qapp, mode)
    try:
        palette = theme_module.current_palette()
        roles = _plot_series_roles(theme_module.plot_series_colours())
        for role, colour in roles.items():
            for background in (palette.bg_primary, palette.bg_surface, palette.bg_surface_alt):
                ratio = _contrast_ratio(colour, background)
                assert ratio >= 3.0, f"{mode} {role} on {background}: {ratio:.2f}"
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def test_plot_series_colours_change_with_the_theme(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    light = theme_module.plot_series_colours()
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)
    try:
        dark = theme_module.plot_series_colours()
        assert dark.reference == theme_module.current_palette().text_muted
        assert dark.total == theme_module.current_palette().text_primary
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    assert light.energy != dark.energy
    assert set(light.categorical).isdisjoint(dark.categorical)
    assert light.label_on_tint == dark.label_on_tint
    for colours in (light, dark):
        assert len(set(colours.categorical)) == len(colours.categorical)


def test_categorical_colour_repeats_past_the_end_of_the_list(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    colours = theme_module.plot_series_colours().categorical

    assert theme_module.categorical_colour(0) == colours[0]
    assert theme_module.categorical_colour(len(colours) + 1) == colours[1]


def test_cluster_color_map_follows_label_order_whatever_the_input_order(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    colours = theme_module.plot_series_colours().categorical

    assert theme_module.cluster_color_map([7, 2, 7, 4]) == {
        2: colours[0],
        4: colours[1],
        7: colours[2],
    }
    assert theme_module.cluster_color_map([]) == {}


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_labels_and_checkboxes_in_cards_show_the_card_behind_them(qapp, mode) -> None:
    previous_sheet = qapp.styleSheet()
    theme_module.apply_theme(qapp, mode)
    try:
        dialog = QDialog()
        dialog.setStyleSheet(theme_module.feature_dialog_stylesheet())
        layout = QVBoxLayout(dialog)
        group = QGroupBox("Options", dialog)
        group_layout = QVBoxLayout(group)
        card = QFrame(group)
        theme_module.style_option_card(card)
        card_layout = QVBoxLayout(card)
        widgets = [QLabel("Label", group), QCheckBox("Check", group)]
        card_widgets = [QLabel("Card label", card), QCheckBox("Card check", card)]
        for widget in widgets:
            group_layout.addWidget(widget)
        for widget in card_widgets:
            card_layout.addWidget(widget)
        group_layout.addWidget(card)
        layout.addWidget(group)
        dialog.resize(420, 320)
        dialog.show()
        qapp.processEvents()

        palette = theme_module.current_palette()
        image = dialog.grab().toImage()
        for widget, expected in [(w, palette.bg_surface) for w in widgets] + [
            (w, palette.bg_surface_alt) for w in card_widgets
        ]:
            # The far end of the widget holds no text or indicator.
            spot = widget.mapTo(dialog, QPoint(widget.width() - 2, widget.height() // 2))
            colour = image.pixelColor(spot)
            assert colour.name().upper() == expected, f"{widget.text()} in {mode}"
        dialog.close()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
        qapp.setStyleSheet(previous_sheet)


def test_style_transparent_surface_lets_the_parent_show_through(qapp) -> None:
    previous_sheet = qapp.styleSheet()
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    try:
        parent = QFrame()
        parent.setStyleSheet("QFrame { background: #123456; }")
        parent.resize(120, 80)
        surface = QWidget(parent)
        surface.setGeometry(0, 0, 120, 80)
        theme_module.style_transparent_surface(surface)
        parent.show()
        qapp.processEvents()

        assert parent.grab().toImage().pixelColor(60, 40).name() == "#123456"
        parent.close()
    finally:
        qapp.setStyleSheet(previous_sheet)


def _colour_distance(first: str, second: str) -> float:
    """Return the CIE76 distance between two colours; below about 25 they read as one hue."""

    def lab(colour: str) -> tuple[float, float, float]:
        channels = [int(colour.lstrip("#")[index : index + 2], 16) / 255 for index in (0, 2, 4)]
        red, green, blue = (
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            for value in channels
        )
        scaled = (
            (0.4124 * red + 0.3576 * green + 0.1805 * blue) / 0.95047,
            0.2126 * red + 0.7152 * green + 0.0722 * blue,
            (0.0193 * red + 0.1192 * green + 0.9505 * blue) / 1.08883,
        )
        x, y, z = (
            value ** (1 / 3) if value > 0.008856 else 7.787 * value + 16 / 116 for value in scaled
        )
        return 116 * y - 16, 500 * (x - y), 200 * (y - z)

    return math.dist(lab(first), lab(second))


#: The roles drawn together on one plot: the two convergence panels, and a
#: geometry-progress criterion or scan against its tolerance line.
_ROLES_SHARING_A_PLOT = (
    ("energy", "fit", "converged", "reference"),
    ("force", "stress", "threshold", "converged", "stress_converged", "reference"),
    ("displacement", "reference"),
)


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_plot_series_roles_sharing_a_plot_read_as_different_colours(qapp, mode) -> None:
    theme_module.apply_theme(qapp, mode)
    try:
        colours = theme_module.plot_series_colours()
        for roles in _ROLES_SHARING_A_PLOT:
            for first, second in itertools.combinations(roles, 2):
                distance = _colour_distance(getattr(colours, first), getattr(colours, second))
                assert distance >= 30.0, f"{mode} {first} and {second}: {distance:.1f}"
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def test_style_plot_axes_styles_minor_tick_labels_on_a_log_axis(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)
    try:
        figure = Figure()
        axis = figure.add_subplot(111)
        theme_module.style_plot_axes(axis, tick_labelsize=8)
        axis.set_yscale("log")
        axis.plot([1, 2], [5e-4, 9e-4])
        figure.canvas.draw()

        minor_labels = [tick.label1 for tick in axis.yaxis.get_minor_ticks()]
        assert minor_labels
        for label in minor_labels:
            assert label.get_color() == theme_module.current_palette().text_primary
            assert label.get_fontsize() == 8
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def test_style_plot_axes_can_leave_the_grid_off(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    figure = Figure()
    axis = figure.add_subplot(111)

    theme_module.style_plot_axes(axis, facecolor="#123456", grid=False)
    figure.canvas.draw()

    assert not any(line.get_visible() for line in axis.get_xgridlines())
    assert axis.xaxis.label.get_color() == theme_module.current_palette().text_primary


def _stylesheet_rules(stylesheet: str) -> list[tuple[str, str]]:
    """Split a stylesheet into (selector, declarations) pairs, comments removed."""

    without_comments = re.sub(r"/\*.*?\*/", "", stylesheet, flags=re.DOTALL)
    return [
        (selector.strip(), body)
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", without_comments)
    ]


def _every_stylesheet() -> list[str]:
    return [
        theme_module._build_stylesheet(theme_module.light_palette()),
        theme_module.feature_dialog_stylesheet(
            include_scroll_area=True, emphasised_radio_buttons=True
        ),
        theme_module.uep_clustering_toolbar_stylesheet(),
        theme_module.magnetic_fields_stylesheet(),
        theme_module.magnetic_structure_builder_stylesheet(),
    ]


def _declared_pixels(property_pattern: str) -> list[tuple[str, int]]:
    """Return (selector, pixels) for each value of the matching properties in every stylesheet."""

    found = []
    for stylesheet in _every_stylesheet():
        for selector, body in _stylesheet_rules(stylesheet):
            for values in re.findall(rf"(?:^|[;\s])(?:{property_pattern})\s*:\s*([^;]+);", body):
                found.extend(
                    (selector, int(value)) for value in re.findall(r"(\d+)(?:px)?\b", values)
                )
    return found


def test_scales_hold_the_agreed_steps() -> None:
    assert dataclasses.astuple(theme_module.TYPE_SCALE) == (11, 13, 15, 18, 28)
    assert dataclasses.astuple(theme_module.SPACING) == (4, 8, 12, 16, 24)
    assert dataclasses.astuple(theme_module.RADIUS) == (6, 10)


def test_stylesheet_font_sizes_are_type_scale_steps(qapp, desktop_font) -> None:
    desktop_font(_system_font(pixels=12))
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    sizes = {pixels for _selector, pixels in _declared_pixels("font-size")}

    assert sizes == set(dataclasses.astuple(theme_module.TYPE_SCALE))


def test_stylesheet_radii_are_scale_steps_or_round_shapes(qapp) -> None:
    allowed = (
        {0}
        | set(dataclasses.astuple(theme_module.RADIUS))
        | set(dataclasses.astuple(theme_module.ROUND_RADIUS))
    )
    radii = {pixels for _selector, pixels in _declared_pixels(r"border(?:-\w+-\w+)?-radius")}

    assert radii <= allowed, sorted(radii - allowed)


def test_stylesheet_padding_is_on_the_spacing_scale(qapp) -> None:
    steps = {0} | set(dataclasses.astuple(theme_module.SPACING))
    gutters = {theme_module._COMBO_ARROW_GUTTER, theme_module._SPIN_ARROW_GUTTER}
    offenders = set()
    for selector, pixels in _declared_pixels(r"padding(?:-\w+)?"):
        # A multi-row marker's thicker border is given back by the padding, so
        # the control keeps its size.
        compensated = (
            {step - 1 for step in steps | gutters} if "multiRowTarget" in selector else set()
        )
        if pixels not in steps | gutters | compensated:
            offenders.add((selector, pixels))

    assert not offenders


def test_stylesheet_round_radii_are_half_the_shape(qapp) -> None:
    round_radius = theme_module.ROUND_RADIUS
    stylesheet = theme_module._build_stylesheet(theme_module.light_palette())
    rules = dict(_stylesheet_rules(stylesheet))

    # 16 px and a 1 px border each side.
    assert f"border-radius: {round_radius.radio_indicator}px" in rules["QRadioButton::indicator"]
    assert round_radius.radio_indicator == (16 + 2) // 2
    handle = rules["QSlider#uepIsosurfaceSlider::handle:horizontal"]
    assert "width: 26px" in handle
    assert f"border-radius: {round_radius.slider_handle}px" in handle
    assert round_radius.slider_handle == 26 // 2
    stamp = rules["QLabel#journalStamp"]
    assert "min-width: 10px" in stamp
    assert f"border-radius: {round_radius.journal_stamp}px" in stamp
    assert round_radius.journal_stamp == 10 // 2
    help_button = rules["QToolButton#quantumAnalysisHelpButton"]
    assert "min-width: 24px" in help_button
    assert f"border-radius: {round_radius.help_button}px" in help_button
    assert round_radius.help_button == 24 // 2
    groove = rules["QSlider#uepIsosurfaceSlider::groove:horizontal"]
    assert "height: 14px" in groove
    assert f"border-radius: {round_radius.slider_groove}px" in groove
    assert round_radius.slider_groove == 14 // 2


def test_layout_helpers_take_their_values_from_the_spacing_scale(qapp) -> None:
    steps = set(dataclasses.astuple(theme_module.SPACING))
    layout = QVBoxLayout()

    layout.setContentsMargins(theme_module.page_margins())
    margins = layout.contentsMargins()
    assert {margins.left(), margins.top(), margins.right(), margins.bottom()} <= steps

    layout.setContentsMargins(theme_module.card_margins())
    margins = layout.contentsMargins()
    assert {margins.left(), margins.top(), margins.right(), margins.bottom()} <= steps

    layout.setSpacing(theme_module.form_spacing())
    assert layout.spacing() in steps


def _system_font(*, points: float | None = None, pixels: int | None = None) -> QFont:
    font = QFont("Sans")
    if points is not None:
        font.setPointSizeF(points)
    if pixels is not None:
        font.setPixelSize(pixels)
    return font


@pytest.fixture
def desktop_font(qapp, monkeypatch):
    """Pretend the desktop's general font is the given one, on a platform with a known default."""

    monkeypatch.setattr(theme_module.sys, "platform", "linux")

    def set_font(font: QFont) -> None:
        monkeypatch.setattr(theme_module.QFontDatabase, "systemFont", lambda _kind: font)

    yield set_font
    monkeypatch.undo()
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def test_default_or_smaller_desktop_text_keeps_the_designed_sizes(qapp, desktop_font) -> None:
    # 11 pt at 96 dpi is the default; 9 pt must not push the smallest step below 11 px.
    desktop_font(_system_font(pixels=12))

    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    assert theme_module.text_scale() == 1.0
    assert theme_module.type_scale() == theme_module.TYPE_SCALE
    assert theme_module.type_scale().caption >= 11


def test_larger_desktop_text_scales_the_stylesheet_and_application_font(qapp, desktop_font) -> None:
    default_size = theme_module._DEFAULT_SYSTEM_TEXT_PX_ELSEWHERE
    desktop_font(_system_font(pixels=round(default_size * 1.5)))
    factor = round(default_size * 1.5) / default_size

    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    assert theme_module.text_scale() == pytest.approx(factor)
    assert qapp.font().pixelSize() == round(11 * factor)
    rules = dict(_stylesheet_rules(qapp.styleSheet()))
    assert f"font-size: {round(28 * factor)}px" in rules["QLabel#homeHeroTitle"]
    assert f"font-size: {round(11 * factor)}px" in rules["QLabel#hintText"]
    toolbar = theme_module.uep_clustering_toolbar_stylesheet()
    assert f"font-size: {round(13 * factor)}px" in toolbar


def test_desktop_text_size_is_read_in_points_as_well_as_pixels(qapp, desktop_font) -> None:
    screen = qapp.primaryScreen()
    default_points = (
        theme_module._DEFAULT_SYSTEM_TEXT_PX_ELSEWHERE * 72.0 / screen.logicalDotsPerInch()
    )
    desktop_font(_system_font(points=default_points * 1.25))

    assert theme_module.text_scale() == pytest.approx(1.25, abs=0.01)


def test_text_scale_is_capped_so_a_huge_desktop_font_stays_usable(qapp, desktop_font) -> None:
    desktop_font(_system_font(pixels=200))

    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    assert theme_module.text_scale() == 2.0
    assert theme_module.type_scale().page == 56


def test_desktop_font_change_applies_with_the_next_theme_only(qapp, desktop_font) -> None:
    desktop_font(_system_font(pixels=12))
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    desktop_font(_system_font(pixels=22))

    assert theme_module.type_scale() == theme_module.TYPE_SCALE
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    assert theme_module.type_scale().caption > theme_module.TYPE_SCALE.caption


def test_text_scale_is_one_without_a_gui_application(monkeypatch) -> None:
    monkeypatch.setattr(
        theme_module.QGuiApplication, "instance", lambda: QCoreApplication.instance()
    )
    monkeypatch.setattr(QCoreApplication, "instance", lambda: object())

    assert theme_module.text_scale() == 1.0


def test_scaled_to_text_follows_the_type_scale(qapp, desktop_font) -> None:
    desktop_font(_system_font(pixels=12))
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    assert theme_module.scaled_to_text(58) == 58

    desktop_font(_system_font(pixels=200))
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    assert theme_module.scaled_to_text(58) == 116


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_standard_button_is_28_px_high_under_both_themes(qapp, desktop_font, mode) -> None:
    desktop_font(_system_font(pixels=12))
    theme_module.apply_theme(qapp, mode)

    button = QPushButton("Configure…")
    marked = QPushButton("Configure…")
    marked.setProperty("multiRowTarget", True)

    assert button.sizeHint().height() == 28
    # The marker's thicker border must not change the height.
    assert marked.sizeHint().height() == 28


def test_standard_button_grows_with_larger_desktop_text(qapp, desktop_font) -> None:
    desktop_font(_system_font(pixels=29))
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    assert QPushButton("Configure…").sizeHint().height() > 28


def test_a_row_button_fits_a_table_row_with_a_pixel_to_spare(qapp, desktop_font) -> None:
    desktop_font(_system_font(pixels=12))
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    table = workflow_table_utils.create_workflow_table(("Action",))
    table.setRowCount(1)
    table.setCellWidget(0, 0, QPushButton("Dismiss"))
    table.show()
    qapp.processEvents()

    assert table.rowHeight(0) >= 29
    assert table.cellWidget(0, 0).height() == table.rowHeight(0) - 1
    table.close()


def _focusable_tool_button() -> QToolButton:
    button = QToolButton()
    button.setText("Home")
    button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    return button


def _combo_box() -> QComboBox:
    combo = QComboBox()
    combo.addItems(["One", "Two"])
    return combo


def _checked(button):
    button.setChecked(True)
    return button


def _tab_bar() -> QTabBar:
    bar = QTabBar()
    bar.addTab("First")
    bar.addTab("Second")
    return bar


def _table() -> QTableWidget:
    table = QTableWidget(2, 2)
    table.setTabKeyNavigation(False)
    table.setMinimumHeight(120)
    return table


def _tree() -> QTreeWidget:
    tree = QTreeWidget()
    tree.setTabKeyNavigation(False)
    QTreeWidgetItem(tree, ["row"])
    return tree


def _scroll_area() -> QScrollArea:
    area = QScrollArea()
    area.setWidget(QLabel("contents"))
    return area


def _list_widget() -> QListWidget:
    widget = QListWidget()
    widget.addItem("row")
    return widget


_FOCUS_RING_KINDS = {
    "push button": lambda: QPushButton("Run"),
    "tool button": _focusable_tool_button,
    "combo box": _combo_box,
    "line edit": lambda: QLineEdit("3.5"),
    "spin box": QSpinBox,
    "double spin box": QDoubleSpinBox,
    "checkbox": lambda: QCheckBox("Cell"),
    "checked checkbox": lambda: _checked(QCheckBox("Cell")),
    "radio button": lambda: QRadioButton("Cell"),
    "checked radio button": lambda: _checked(QRadioButton("Cell")),
    "tab bar": _tab_bar,
    "table": _table,
    "tree": _tree,
    "list": _list_widget,
    "text edit": lambda: QPlainTextEdit("notes"),
    "scroll area": lambda: _scroll_area(),
}


class _FocusScene:
    """A shown window with a leading button, the control under test and a neighbour."""

    def __init__(self, control: QWidget) -> None:
        self.window = QWidget()
        layout = QVBoxLayout(self.window)
        self.leading = QPushButton("Before")
        self.control = control
        self.trailing = QPushButton("After")
        for widget in (self.leading, control, self.trailing):
            layout.addWidget(widget)
        self.window.resize(360, 320)
        self.window.show()
        self.window.activateWindow()
        assert QTest.qWaitForWindowActive(self.window)
        self.leading.setFocus()
        QApplication.processEvents()

    def tab_to_control(self) -> None:
        QTest.keyClick(self.leading, Qt.Key.Key_Tab)
        QApplication.processEvents()
        assert self.control.hasFocus()

    def ring(self) -> QWidget | None:
        rings = [
            ring
            for ring in self.window.findChildren(QWidget, "mufinderFocusRing")
            if ring.isVisible()
        ]
        assert len(rings) <= 1
        return rings[0] if rings else None

    def close(self) -> None:
        self.window.close()
        self.window.deleteLater()
        QApplication.processEvents()


@pytest.fixture
def focus_scene(qapp):
    scenes: list[_FocusScene] = []

    def build(control: QWidget, mode: str = theme_module.THEME_LIGHT) -> _FocusScene:
        # A fresh tracker, so an earlier test's keyboard use does not carry over.
        theme_module.cleanup_theme_runtime(qapp)
        theme_module.apply_theme(qapp, mode)
        scenes.append(_FocusScene(control))
        return scenes[-1]

    yield build
    for scene in scenes:
        scene.close()
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
@pytest.mark.parametrize("kind", _FOCUS_RING_KINDS)
def test_keyboard_focus_draws_an_accent_ring_without_moving_anything(
    focus_scene, kind: str, mode: str
) -> None:
    scene = focus_scene(_FOCUS_RING_KINDS[kind](), mode)
    control = scene.control
    before = control.grab().toImage()
    geometry = [w.geometry() for w in (scene.leading, control, scene.trailing)]
    size_hint = control.sizeHint()

    scene.tab_to_control()

    image = control.grab().toImage()
    assert image != before
    ring = scene.ring()
    assert ring is not None
    assert control.rect().contains(ring.geometry())
    # The ring stays clear of the layout: no control moved or changed its size.
    assert [w.geometry() for w in (scene.leading, control, scene.trailing)] == geometry
    assert control.sizeHint() == size_hint
    # Two pixels of solid accent on the ring's left edge, at half height.
    ratio = control.devicePixelRatioF()
    box = ring.geometry()
    y = int((box.top() + box.height() // 2) * ratio)
    accent = QColor(theme_module.current_palette().accent)
    for x in (box.left(), box.left() + 1):
        colour = image.pixelColor(int(x * ratio), y)
        # A round indicator's edge is antialiased, so allow a small blend.
        assert abs(colour.redF() - accent.redF()) < 0.05
        assert abs(colour.greenF() - accent.greenF()) < 0.05
        assert abs(colour.blueF() - accent.blueF()) < 0.05


@pytest.mark.parametrize("kind", ["push button", "tool button", "combo box", "line edit"])
def test_mouse_focus_leaves_no_ring(focus_scene, kind: str) -> None:
    scene = focus_scene(_FOCUS_RING_KINDS[kind]())

    QTest.mouseClick(scene.control, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    if isinstance(scene.control, QComboBox):
        scene.control.hidePopup()
        QApplication.processEvents()

    assert scene.control.hasFocus()
    assert scene.ring() is None


@pytest.mark.parametrize("kind", ["push button", "combo box", "checkbox", "radio button"])
def test_ring_replaces_the_base_style_focus_tint(focus_scene, kind: str) -> None:
    scene = focus_scene(_FOCUS_RING_KINDS[kind]())
    before = scene.control.grab().toImage()

    scene.tab_to_control()
    ring = scene.ring()
    ring.hide()
    QApplication.processEvents()

    assert scene.control.grab().toImage() == before


@pytest.mark.parametrize(
    ("kind", "click_target"),
    [
        ("push button", lambda control: control),
        ("line edit", lambda control: control),
        ("table", lambda control: control.viewport()),
        ("tree", lambda control: control.viewport()),
        ("text edit", lambda control: control.viewport()),
        ("spin box", lambda control: control.lineEdit()),
    ],
)
def test_clicking_the_control_that_has_the_ring_removes_it(
    focus_scene, kind: str, click_target
) -> None:
    scene = focus_scene(_FOCUS_RING_KINDS[kind]())
    scene.tab_to_control()
    assert scene.ring() is not None

    QTest.mouseClick(click_target(scene.control), Qt.MouseButton.LeftButton)
    QApplication.processEvents()

    assert scene.control.hasFocus()
    assert scene.ring() is None


def test_focus_returning_from_a_dialog_keeps_the_ring(focus_scene) -> None:
    scene = focus_scene(QPushButton("Remove"))
    scene.tab_to_control()

    dialog = QDialog(scene.window)
    QVBoxLayout(dialog).addWidget(QPushButton("OK", dialog))
    dialog.show()
    dialog.activateWindow()
    assert QTest.qWaitForWindowActive(dialog)
    dialog.close()
    scene.window.activateWindow()
    assert QTest.qWaitForWindowActive(scene.window)
    QApplication.processEvents()

    assert scene.control.hasFocus()
    assert scene.ring() is not None


def test_a_mouse_click_ends_keyboard_use_across_later_focus_changes(focus_scene) -> None:
    scene = focus_scene(QPushButton("Run"))
    scene.tab_to_control()

    QTest.mouseClick(scene.trailing, Qt.MouseButton.LeftButton)
    scene.control.setFocus()
    QApplication.processEvents()

    assert scene.control.hasFocus()
    assert scene.ring() is None


def test_open_combo_list_gets_no_ring(focus_scene) -> None:
    scene = focus_scene(_combo_box())
    scene.tab_to_control()

    scene.control.showPopup()
    QApplication.processEvents()
    popup_rings = [
        ring
        for ring in scene.control.view().findChildren(QWidget, "mufinderFocusRing")
        if ring.isVisible()
    ]
    scene.control.hidePopup()
    QApplication.processEvents()

    assert popup_rings == []
    assert scene.ring() is not None


def test_ring_follows_tabs_that_move_without_a_new_current_tab(focus_scene) -> None:
    scene = focus_scene(_tab_bar())
    scene.tab_to_control()
    bar = scene.control

    bar.insertTab(0, "A new first tab with a long title")
    QApplication.processEvents()
    scene.window.repaint()
    QApplication.processEvents()

    assert scene.ring().geometry() == bar.tabRect(bar.currentIndex())


def test_ring_moves_with_focus_and_leaves_the_previous_control_unmarked(focus_scene) -> None:
    scene = focus_scene(QPushButton("Run"))
    unfocused = scene.trailing.grab().toImage()

    scene.tab_to_control()
    QTest.keyClick(scene.control, Qt.Key.Key_Tab)
    QApplication.processEvents()

    assert scene.trailing.hasFocus()
    assert scene.ring().parentWidget() is scene.trailing
    assert scene.trailing.grab().toImage() != unfocused

    QTest.keyClick(scene.trailing, Qt.Key.Key_Backtab)
    QApplication.processEvents()
    assert scene.ring().parentWidget() is scene.control
    assert scene.trailing.grab().toImage() == unfocused


def test_ring_follows_the_current_tab(focus_scene) -> None:
    scene = focus_scene(_tab_bar())
    scene.tab_to_control()
    assert scene.ring().geometry() == scene.control.tabRect(0)

    QTest.keyClick(scene.control, Qt.Key.Key_Right)
    QApplication.processEvents()

    assert scene.control.currentIndex() == 1
    assert scene.ring().geometry() == scene.control.tabRect(1)


def test_ring_follows_a_resized_control(focus_scene) -> None:
    scene = focus_scene(QPushButton("Run"))
    scene.tab_to_control()

    scene.window.resize(520, 360)
    QApplication.processEvents()

    assert scene.ring().geometry() == scene.control.rect()


def test_indicator_ring_surrounds_the_indicator_not_the_label(focus_scene) -> None:
    scene = focus_scene(QCheckBox("A long label beside the indicator"))
    scene.tab_to_control()

    ring = scene.ring()

    assert ring.width() == ring.height()
    assert ring.width() < scene.control.width() / 2


def test_radio_ring_is_a_whole_circle(focus_scene) -> None:
    scene = focus_scene(QRadioButton("Cell"))
    scene.tab_to_control()

    box = scene.ring().geometry()
    radius = (box.width() - theme_module._FOCUS_RING_WIDTH) / 2
    image = scene.control.grab().toImage()
    ratio = scene.control.devicePixelRatioF()
    accent = QColor(theme_module.current_palette().accent)
    offset = radius * math.cos(math.pi / 4)
    for sign_x, sign_y in itertools.product((-1, 1), repeat=2):
        colour = image.pixelColor(
            int((box.center().x() + 0.5 + sign_x * offset) * ratio),
            int((box.center().y() + 0.5 + sign_y * offset) * ratio),
        )
        assert abs(colour.redF() - accent.redF()) < 0.2
        assert abs(colour.greenF() - accent.greenF()) < 0.2


def _wcag_ratio(first: QColor, second: QColor) -> float:
    return _contrast_ratio(first.name(), second.name())


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_ring_stands_out_on_a_row_button_in_a_selected_row(focus_scene, mode: str) -> None:
    table = QTableWidget(3, 1)
    table.setTabKeyNavigation(False)
    table.horizontalHeader().setStretchLastSection(True)
    row_button = QPushButton("Configured...")
    table.setCellWidget(0, 0, row_button)
    scene = focus_scene(table, mode)
    table.setRowHeight(0, row_button.sizeHint().height() + 1)
    table.selectRow(0)
    table.setFocus()
    QTest.keyClick(table, Qt.Key.Key_Tab)
    QApplication.processEvents()
    assert scene.ring().parentWidget() is row_button

    image = row_button.grab().toImage()
    ratio = row_button.devicePixelRatioF()
    y = int(row_button.height() / 2 * ratio)
    keyline, accent = (image.pixelColor(int(x * ratio), y) for x in (0, 1))

    palette = theme_module.current_palette()
    assert accent == QColor(palette.accent)
    assert keyline == QColor(palette.bg_surface)
    # The keyline separates the ring from the selected row's accent fill.
    assert _wcag_ratio(keyline, QColor(palette.accent)) >= 3


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_ring_stands_out_on_a_checked_accent_filled_button(focus_scene, mode: str) -> None:
    button = QPushButton("Cluster")
    button.setObjectName("uepClusteringToolbarButton")
    button.setCheckable(True)
    button.setChecked(True)
    scene = focus_scene(button, mode)
    button.setStyleSheet(theme_module.uep_clustering_toolbar_stylesheet())
    QApplication.processEvents()
    scene.tab_to_control()

    image = button.grab().toImage()
    ratio = button.devicePixelRatioF()
    y = int(button.height() / 2 * ratio)
    ring, keyline, fill = (image.pixelColor(int(x * ratio), y) for x in (1, 2, 4))

    palette = theme_module.current_palette()
    assert fill == QColor(palette.accent)
    assert ring == fill
    assert _wcag_ratio(keyline, fill) >= 3


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_ring_stands_out_on_a_selected_row_at_the_view_edge(focus_scene, mode: str) -> None:
    table = QTableWidget(3, 1)
    table.setTabKeyNavigation(False)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    table.setMinimumHeight(120)
    scene = focus_scene(table, mode)
    table.selectRow(1)
    QApplication.processEvents()
    scene.tab_to_control()

    image = table.grab().toImage()
    ratio = table.devicePixelRatioF()
    y = int(table.rowViewportPosition(1) + table.viewport().y() + table.rowHeight(1) / 2)
    ring, keyline, fill = (image.pixelColor(int(x * ratio), int(y * ratio)) for x in (1, 2, 5))

    # Fusion shades a selected row, so its fill is only close to the accent colour.
    assert ring == QColor(theme_module.current_palette().accent)
    assert _wcag_ratio(ring, fill) < 1.2
    assert _wcag_ratio(keyline, fill) >= 3


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_scroll_bars_do_not_cover_the_ring(focus_scene, mode: str) -> None:
    table = QTableWidget(40, 12)
    table.setTabKeyNavigation(False)
    table.setFixedSize(240, 160)
    scene = focus_scene(table, mode)
    assert table.verticalScrollBar().isVisible()
    assert table.horizontalScrollBar().isVisible()
    scene.tab_to_control()

    accent = QColor(theme_module.current_palette().accent)
    image = table.grab().toImage()
    ratio = table.devicePixelRatioF()
    width, height = image.width(), image.height()
    middle_x, middle_y = width // 2, height // 2
    bar_x = int((table.verticalScrollBar().parentWidget().x() + 4) * ratio)
    bar_y = int((table.horizontalScrollBar().parentWidget().y() + 4) * ratio)
    for pixels in (
        [(width - 1, middle_y), (width - 2, middle_y)],
        [(middle_x, height - 1), (middle_x, height - 2)],
        [(bar_x, 0), (bar_x, 1)],
        [(0, bar_y), (1, bar_y)],
    ):
        for x, y in pixels:
            assert image.pixelColor(x, y) == accent

    # And when a bar appears after the ring: the bars are raised as they are laid out.
    table.setRowCount(1)
    QApplication.processEvents()
    table.setRowCount(40)
    QApplication.processEvents()
    image = table.grab().toImage()
    assert image.pixelColor(width - 1, middle_y) == accent
    assert image.pixelColor(width - 2, middle_y) == accent


def test_ring_fits_the_drawn_tab_not_its_margin(focus_scene) -> None:
    tabs = QTabWidget()
    tabs.addTab(QWidget(), "First")
    tabs.addTab(QWidget(), "Second")
    theme_module.style_tab_widget(tabs)
    scene = focus_scene(tabs)
    scene.leading.setFocus()
    QTest.keyClick(scene.leading, Qt.Key.Key_Tab)
    QApplication.processEvents()
    bar = tabs.tabBar()
    assert bar.hasFocus()

    assert scene.ring().geometry() == bar.tabRect(0).adjusted(0, 0, -theme_module._TAB_GAP, 0)


def test_item_view_contents_do_not_shift_on_keyboard_focus(focus_scene) -> None:
    table = _table()
    table.horizontalHeader().setStretchLastSection(True)
    scene = focus_scene(table)
    viewport = table.viewport().geometry()
    columns = [table.columnWidth(column) for column in range(table.columnCount())]

    scene.tab_to_control()

    assert table.viewport().geometry() == viewport
    assert [table.columnWidth(column) for column in range(table.columnCount())] == columns


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_ring_on_a_row_button_is_whole_inside_its_table_row(focus_scene, mode: str) -> None:
    table = QTableWidget(3, 1)
    table.setTabKeyNavigation(False)
    table.horizontalHeader().setStretchLastSection(True)
    row_button = QPushButton("Configured...")
    table.setCellWidget(1, 0, row_button)
    scene = focus_scene(table, mode)
    table.setRowHeight(1, row_button.sizeHint().height() + 1)
    QApplication.processEvents()
    geometry = row_button.geometry()

    table.setFocus()
    QTest.keyClick(table, Qt.Key.Key_Tab)
    QApplication.processEvents()
    assert row_button.hasFocus()

    assert row_button.geometry() == geometry
    assert table.rowHeight(1) == row_button.sizeHint().height() + 1
    ring = scene.ring()
    assert ring.parentWidget() is row_button
    assert ring.geometry() == row_button.rect()
    # The ring's top and bottom edges reach the screen: nothing in the row clips them.
    image = table.viewport().grab().toImage()
    ratio = table.viewport().devicePixelRatioF()
    accent = QColor(theme_module.current_palette().accent)
    x = int(geometry.center().x() * ratio)
    # A keyline at each edge, then two pixels of accent.
    for y in (geometry.top() + 1, geometry.top() + 2, geometry.bottom() - 2, geometry.bottom() - 1):
        assert image.pixelColor(x, int(y * ratio)) == accent


def test_ring_takes_the_new_accent_after_a_theme_change(qapp, focus_scene) -> None:
    scene = focus_scene(QPushButton("Run"))
    scene.tab_to_control()

    theme_module.apply_theme(qapp, theme_module.THEME_DARK)
    QApplication.processEvents()

    image = scene.control.grab().toImage()
    ratio = scene.control.devicePixelRatioF()
    y = int(scene.control.height() / 2 * ratio)
    assert image.pixelColor(0, y) == QColor(theme_module.dark_palette().accent)


def test_cleanup_stops_the_focus_ring_and_apply_theme_restores_it(qapp, focus_scene) -> None:
    scene = focus_scene(QPushButton("Run"))

    theme_module.cleanup_theme_runtime(qapp)
    scene.tab_to_control()
    assert scene.ring() is None

    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    scene.leading.setFocus()
    QApplication.processEvents()
    scene.tab_to_control()
    assert scene.ring() is not None


def test_cleanup_removes_a_ring_that_is_showing(qapp, focus_scene) -> None:
    scene = focus_scene(QPushButton("Run"))
    scene.tab_to_control()
    assert scene.ring() is not None

    theme_module.cleanup_theme_runtime(qapp)

    assert scene.ring() is None
