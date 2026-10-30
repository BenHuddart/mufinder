"""Tests for comparison-table integration in main window tree/routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from mufinder.app import workflow_table_utils
from mufinder.app.workflow_controllers.comparison_tables import comparison_route
from mufinder.domain.models import ComparisonTable, Structure
from mufinder.services.project_service import ProjectService
from tests.support.main_window import open_project_with_host
from tests.support.qt import find_tree_item_by_route


def test_main_window_adds_and_deletes_comparison_subtab(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "comparison_ui.mufinder", "ComparisonUI")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    table = ComparisonTable(
        label="Comparison 1",
        experimental_structure_id=host.id,
        dft_structure_ids=[host.id],
        dft_labels=["DFT-1"],
        result_data={
            "metric_definitions": [],
            "experimental_values": {},
            "dft_columns": [],
            "parameter_differences": [],
            "warnings": [],
        },
    )
    service.add_comparison_table(doc, table)

    window._document = doc
    window._refresh_ui_after_project_change()

    route = comparison_route(table.id)
    item = find_tree_item_by_route(window._project_tree, route)
    assert item is not None

    window._set_active_route(route)
    assert window._main_stack.currentWidget() is window._comparison_tables_view
    table_widget = window._comparison_tables_view.table_widget(table.id)
    assert window._comparison_tables_view.currentWidget() is table_widget

    window._comparison_tables_controller.delete_table(table.id)
    assert table.id not in doc.comparison_tables


def test_widget_delete_request_removes_only_requested_table_without_extra_prompt(
    main_window, tmp_path: Path, confirmed_destructive_actions
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "comparison_ui_multi.mufinder", "ComparisonUIMulti")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    table_a = ComparisonTable(
        label="Comparison A",
        experimental_structure_id=host.id,
        dft_structure_ids=[host.id],
        dft_labels=["DFT-A"],
        result_data={
            "metric_definitions": [],
            "experimental_values": {},
            "dft_columns": [],
            "parameter_differences": [],
            "warnings": [],
        },
    )
    table_b = ComparisonTable(
        label="Comparison B",
        experimental_structure_id=host.id,
        dft_structure_ids=[host.id],
        dft_labels=["DFT-B"],
        result_data={
            "metric_definitions": [],
            "experimental_values": {},
            "dft_columns": [],
            "parameter_differences": [],
            "warnings": [],
        },
    )
    service.add_comparison_table(doc, table_a)
    service.add_comparison_table(doc, table_b)

    window._document = doc
    window._refresh_ui_after_project_change()

    # The widget asked before emitting; MainWindow must not ask a second time.
    window._comparison_tables_controller.delete_table(table_a.id)

    assert table_a.id not in doc.comparison_tables
    assert table_b.id in doc.comparison_tables
    assert confirmed_destructive_actions == []


def test_comparison_delete_button_confirms_once_through_the_shared_helper(
    main_window, tmp_path: Path, monkeypatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "comparison_delete.mufinder", "ComparisonDelete")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    table = ComparisonTable(
        label="Comparison 1",
        experimental_structure_id=host.id,
        dft_structure_ids=[host.id],
        dft_labels=["DFT-1"],
        result_data={
            "metric_definitions": [],
            "experimental_values": {},
            "dft_columns": [],
            "parameter_differences": [],
            "warnings": [],
        },
    )
    service.add_comparison_table(doc, table)
    window._document = doc
    window._refresh_ui_after_project_change()

    answers = [False, True]
    prompts: list[tuple[str, str, str]] = []

    def _confirm(_parent, title, text, detail="", accept_label="Remove", *_args, **_kwargs):
        prompts.append((title, text, accept_label))
        return answers.pop(0)

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", _confirm)
    window._comparison_tables_view.table_widget(table.id)._delete_button.click()
    assert table.id in doc.comparison_tables

    window._comparison_tables_view.table_widget(table.id)._delete_button.click()

    assert table.id not in doc.comparison_tables
    # One prompt per click: the window does not ask again after the widget.
    assert (
        prompts
        == [("Delete comparison table", "Delete this comparison table?", "Delete table")] * 2
    )


@pytest.fixture
def renaming_table(main_window, tmp_path: Path):
    """``(window, table, title_edit)``: a table on screen with an edited title holding focus."""

    window = main_window
    service, doc, host = open_project_with_host(window, tmp_path)
    table = ComparisonTable(
        label="Original",
        experimental_structure_id=host.id,
        dft_structure_ids=[host.id],
        dft_labels=["DFT-1"],
        result_data={
            "experimental_values": {},
            "dft_columns": [],
            "parameter_differences": [],
            "warnings": [],
        },
    )
    service.add_comparison_table(doc, table)
    window._refresh_ui_after_project_change()
    window._set_active_route(comparison_route(table.id))
    window.show()
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window)

    title_edit = window._comparison_tables_view.table_widget(table.id)._title_edit
    title_edit.setFocus()
    QApplication.processEvents()
    assert title_edit.hasFocus()
    title_edit.setText("Renamed")

    yield window, table, title_edit

    # Hidden, the window closes at clean-up without asking to save the project.
    window.hide()


def _assert_renamed_in_tree(window, table: ComparisonTable) -> None:
    assert table.label == "Renamed"
    item = find_tree_item_by_route(window._project_tree, comparison_route(table.id))
    assert item is not None
    assert item.text(0) == "Renamed"


def test_rename_caused_by_navigating_away_stays_on_the_new_page(renaming_table) -> None:
    window, table, _title_edit = renaming_table

    window._set_active_route("provenance_log")
    QApplication.processEvents()

    assert window._active_route() == "provenance_log"
    assert window._main_stack.currentWidget() is window._journal_view
    _assert_renamed_in_tree(window, table)


def test_rename_caused_by_going_home_leaves_nothing_selected(renaming_table) -> None:
    window, table, _title_edit = renaming_table

    window._show_project_home()
    QApplication.processEvents()

    assert window._active_route() is None
    assert window._main_stack.currentWidget() is window._home_page
    _assert_renamed_in_tree(window, table)


def test_rename_caused_by_clicking_the_tree_opens_the_clicked_page(renaming_table) -> None:
    window, table, _title_edit = renaming_table
    tree = window._project_tree
    item = find_tree_item_by_route(tree, "provenance_log")
    tree.scrollToItem(item)

    QTest.mouseClick(
        tree.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        tree.visualItemRect(item).center(),
    )
    QApplication.processEvents()

    assert window._active_route() == "provenance_log"
    assert window._main_stack.currentWidget() is window._journal_view
    _assert_renamed_in_tree(window, table)


def test_rename_with_enter_keeps_the_table_open(renaming_table) -> None:
    window, table, title_edit = renaming_table

    QTest.keyClick(title_edit, Qt.Key.Key_Return)
    QApplication.processEvents()

    assert window._active_route() == comparison_route(table.id)
    assert window._main_stack.currentWidget() is window._comparison_tables_view
    _assert_renamed_in_tree(window, table)
