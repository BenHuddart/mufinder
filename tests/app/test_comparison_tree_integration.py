"""Tests for comparison-table integration in main window tree/routes."""

from __future__ import annotations

from pathlib import Path

from mufinder.app.main_window import MainWindow
from mufinder.domain.models import ComparisonTable, Structure
from mufinder.services.project_service import ProjectService


def _find_item_by_route(parent, route: str):
    if parent.data(0, 0x0100) == route:
        return parent
    for index in range(parent.childCount()):
        child = parent.child(index)
        if child is None:
            continue
        found = _find_item_by_route(child, route)
        if found is not None:
            return found
    return None


def _find_tree_item_by_route(window: MainWindow, route: str):
    for index in range(window._project_tree.topLevelItemCount()):
        root = window._project_tree.topLevelItem(index)
        if root is None:
            continue
        item = _find_item_by_route(root, route)
        if item is not None:
            return item
    return None


def test_main_window_adds_and_deletes_comparison_subtab(qapp, tmp_path: Path) -> None:
    window = MainWindow()
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

    route = window._comparison_route(table.id)
    item = _find_tree_item_by_route(window, route)
    assert item is not None

    window._set_active_route(route)
    assert window._main_stack.currentWidget() is window._comparison_widgets[table.id]

    window._delete_comparison_table(table.id, confirm=False)
    assert table.id not in doc.comparison_tables

    window.close()


def test_widget_delete_request_removes_only_requested_table_without_extra_prompt(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    window = MainWindow()
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

    calls: list[tuple] = []

    def _unexpected_question(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("MainWindow should not ask for a second confirmation for widget-originated deletes")

    monkeypatch.setattr("mufinder.app.main_window.QMessageBox.question", _unexpected_question)

    window._on_delete_comparison_table_requested(table_a.id)

    assert table_a.id not in doc.comparison_tables
    assert table_b.id in doc.comparison_tables
    assert calls == []

    window.close()
