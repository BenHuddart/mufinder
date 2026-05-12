"""Tests for centralized app theming helpers."""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QFrame, QTableWidget, QToolButton, QTreeWidget, QTreeWidgetItem, QWidget

from mufinder.app import theme as theme_module


def test_theme_mode_roundtrip() -> None:
    settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "MuFinder", "ThemeRoundtrip")
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
    assert 'QTableWidget[hasVerticalScrollbar="true"] QHeaderView::section:horizontal:last' in stylesheet
    assert 'QTableWidget[hasVisibleVerticalHeader="true"] QHeaderView::section:horizontal:first' in stylesheet
    assert 'QTreeWidget[squareTopCorners="true"]' in stylesheet
    assert 'QTreeWidget[hasVerticalScrollbar="true"] QHeaderView::section:horizontal:last' in stylesheet
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


def test_table_corner_property_tracks_vertical_scrollbar(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    table = QTableWidget(0, 1)
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


def test_table_corner_property_tracks_vertical_header_visibility(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    table = QTableWidget(0, 1)
    table.show()
    qapp.processEvents()

    assert table.property("hasVisibleVerticalHeader") == "true"

    table.verticalHeader().setVisible(False)
    qapp.processEvents()
    assert table.property("hasVisibleVerticalHeader") == "false"

    table.verticalHeader().setVisible(True)
    qapp.processEvents()
    assert table.property("hasVisibleVerticalHeader") == "true"


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


def test_tree_corner_property_tracks_vertical_scrollbar(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)

    tree = QTreeWidget()
    tree.setColumnCount(1)
    tree.setHeaderLabels(["Name"])
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
