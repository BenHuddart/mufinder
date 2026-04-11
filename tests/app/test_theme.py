"""Tests for centralized app theming helpers."""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QTableWidget

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
    assert 'QTableWidget[hasVerticalScrollbar="true"] QHeaderView::section:horizontal:last' in stylesheet
    assert 'QTableWidget[hasVisibleVerticalHeader="true"] QHeaderView::section:horizontal:first' in stylesheet
    assert "QComboBox::drop-down" in stylesheet
    assert "QComboBox::down-arrow" in stylesheet
    assert "QSpinBox::up-button" in stylesheet
    assert "QSpinBox::up-arrow" in stylesheet
    assert qapp.font().family() == theme_module.body_font_family()


def test_status_brush_uses_palette_tokens(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)

    brush = theme_module.status_brush("failed")
    colour = brush.color().name().lower()

    assert colour == theme_module.current_palette().status_failed.lower()


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
