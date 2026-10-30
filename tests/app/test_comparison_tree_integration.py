"""Tests for comparison-table integration in main window tree/routes."""

from __future__ import annotations

from pathlib import Path

from mufinder.app import workflow_table_utils
from mufinder.app.workflow_controllers.comparison_tables import comparison_route
from mufinder.domain.models import ComparisonTable, Structure
from mufinder.services.project_service import ProjectService
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
