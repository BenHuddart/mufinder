"""Tests for comparison table persistence through project save/load."""

from __future__ import annotations

from pathlib import Path

from mufinder.domain.models import ComparisonTable, Structure
from mufinder.services.project_service import ProjectService


def test_comparison_table_roundtrip(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "comparison_roundtrip.mufinder"
    doc = service.create_project(root, "ComparisonRoundtrip")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    table = ComparisonTable(
        label="Comparison A",
        experimental_structure_id=host.id,
        dft_structure_ids=[host.id],
        dft_labels=["DFT-1"],
        result_data={
            "metric_definitions": [{"key": "a", "label": "a", "unit": "Ang", "group": "Lattice"}],
            "experimental_values": {"a": {"display": "5"}},
            "dft_columns": [{"label": "DFT-1", "values": {"a": {"display": "5.1"}}}],
            "parameter_differences": [],
            "warnings": [],
        },
    )
    service.add_comparison_table(doc, table)

    service.save_project(doc)
    loaded = service.load_project(root)

    assert table.id in loaded.comparison_tables
    restored = loaded.comparison_tables[table.id]
    assert restored.label == "Comparison A"
    assert restored.experimental_structure_id == host.id
    assert restored.dft_labels == ["DFT-1"]


def test_deleted_comparison_table_stays_deleted_after_save_reload(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "comparison_delete_roundtrip.mufinder"
    doc = service.create_project(root, "ComparisonDeleteRoundtrip")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    table = ComparisonTable(
        label="Comparison A",
        experimental_structure_id=host.id,
        dft_structure_ids=[host.id],
        dft_labels=["DFT-1"],
        result_data={
            "metric_definitions": [{"key": "a", "label": "a", "unit": "Ang", "group": "Lattice"}],
            "experimental_values": {"a": {"display": "5"}},
            "dft_columns": [{"label": "DFT-1", "values": {"a": {"display": "5.1"}}}],
            "parameter_differences": [],
            "warnings": [],
        },
    )
    service.add_comparison_table(doc, table)
    service.save_project(doc)

    service.remove_comparison_table(doc, table.id)
    service.save_project(doc)

    loaded = service.load_project(root)

    assert table.id not in loaded.comparison_tables


def test_record_comparison_table_created_appends_provenance(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "comparison_log.mufinder"
    doc = service.create_project(root, "ComparisonLog")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    table = ComparisonTable(
        label="Comparison B",
        experimental_structure_id=host.id,
        dft_structure_ids=[host.id],
        dft_labels=["DFT-1"],
        result_data={},
    )

    service.add_comparison_table(doc, table)
    service.record_comparison_table_created(doc, table=table)

    assert doc.provenance[-1].action == "create_comparison_table"
    assert doc.provenance[-1].parameters["table_label"] == "Comparison B"
