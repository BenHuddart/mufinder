"""Tests for centralized app theming helpers."""

from __future__ import annotations

import pytest

from PySide6.QtCore import QEvent, QSettings, Qt
from PySide6.QtGui import QColor, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QMessageBox,
    QProxyStyle,
    QStyle,
    QTableWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from mufinder.app import theme as theme_module


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
