"""Tests for comparison-table integration in main window tree/routes."""

from __future__ import annotations

from pathlib import Path

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

    route = window._comparison_route(table.id)
    item = find_tree_item_by_route(window._project_tree, route)
    assert item is not None

    window._set_active_route(route)
    assert window._main_stack.currentWidget() is window._comparison_widgets[table.id]

    window._delete_comparison_table(table.id, confirm=False)
    assert table.id not in doc.comparison_tables



def test_widget_delete_request_removes_only_requested_table_without_extra_prompt(
    main_window, tmp_path: Path, monkeypatch
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

    calls: list[tuple] = []

    def _unexpected_question(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("MainWindow should not ask for a second confirmation for widget-originated deletes")

    monkeypatch.setattr("mufinder.app.main_window.QMessageBox.question", _unexpected_question)

    window._on_delete_comparison_table_requested(table_a.id)

    assert table_a.id not in doc.comparison_tables
    assert table_b.id in doc.comparison_tables
    assert calls == []

