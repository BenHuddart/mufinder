"""Tests for comparison table persistence through project save/load."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

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
    # Removing a table that is already gone is a no-op.
    service.remove_comparison_table(doc, table.id)
    service.save_project(doc)

    loaded = service.load_project(root)

    assert table.id not in loaded.comparison_tables


def test_renamed_comparison_table_roundtrips(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "comparison_rename_roundtrip.mufinder"
    doc = service.create_project(root, "ComparisonRenameRoundtrip")

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
        result_data={},
    )
    service.add_comparison_table(doc, table)

    renamed = service.rename_comparison_table(doc, table.id, "  Better title  ")
    service.save_project(doc)
    loaded = service.load_project(root)

    assert renamed is table
    assert loaded.comparison_tables[table.id].label == "Better title"


def test_rename_comparison_table_falls_back_to_the_default_label(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "comparison_rename_blank.mufinder", "ComparisonRenameBlank")
    table = ComparisonTable(label="Original")
    service.add_comparison_table(doc, table)

    service.rename_comparison_table(doc, table.id, "   ")

    assert table.label == "Comparison Table"
    assert service.rename_comparison_table(doc, uuid4(), "X") is None


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

    record = doc.provenance[-1]
    assert record.action == "create_comparison_table"
    assert record.parameters["table_label"] == "Comparison B"
    assert record.parameters["dft_structure_count"] == 1
    assert record.input_entity_ids == [host.id]
    assert table.id in record.output_entity_ids

    without_experiment = ComparisonTable(label="NoExp", dft_structure_ids=[host.id])
    service.record_comparison_table_created(doc, table=without_experiment)

    assert doc.provenance[-1].input_entity_ids == []
