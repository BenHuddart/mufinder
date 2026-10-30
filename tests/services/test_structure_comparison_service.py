"""Tests for building a saved comparison table from completed relaxations."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.castep_io.types import BlockSection, ValueWithUnit
from mufinder.domain.models import ComparisonTable, Structure
from mufinder.science.structure_comparison import DftComparisonInput
from mufinder.science.structure_metrics import metric_definitions
from mufinder.services.project_service import ProjectService
from mufinder.services.source_candidates import CompletedPristineJob
from mufinder.services.structure_comparison import (
    build_comparison_table,
    comparison_inputs,
    parameter_difference_rows,
)


def _cubic(label: str, a: float) -> Structure:
    return Structure(
        label=label,
        lattice_vectors=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def _job(label: str, structure: Structure, **parameter_values: object) -> CompletedPristineJob:
    return CompletedPristineJob(
        job_id=uuid4(),
        job_label=label,
        structure_id=structure.id,
        supercell_label="Unit cell",
        status="completed",
        total_energy=None,
        lattice_a=None,
        lattice_b=None,
        lattice_c=None,
        parameter_values=dict(parameter_values),
    )


def test_a_new_table_stores_values_only() -> None:
    host = _cubic("Host", 5.0)
    relaxed = _cubic("Relaxed", 5.1)
    job = _job("alpha", relaxed)

    table = build_comparison_table(
        label="Comparison 1",
        host=host,
        inputs=[
            DftComparisonInput(label="DFT (alpha)", structure=relaxed, structure_id=relaxed.id)
        ],
        jobs=[job],
    )

    assert table.label == "Comparison 1"
    assert table.experimental_structure_id == host.id
    assert table.dft_structure_ids == [relaxed.id]
    assert table.dft_labels == ["DFT (alpha)"]

    data = table.result_data
    assert set(data) == {
        "experimental_values",
        "dft_columns",
        "parameter_differences",
        "warnings",
    }
    keys = {definition.key for definition in metric_definitions()}
    assert set(data["experimental_values"]) == keys
    assert data["experimental_values"]["a"] == {"value": 5.0, "display": "5", "warning": None}
    column = data["dft_columns"][0]
    assert column["label"] == "DFT (alpha)"
    assert column["structure_id"] == str(relaxed.id)
    assert column["supercell_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert set(column["values"]) == keys
    assert column["values"]["a"]["display"] == "5.1 (+2.00 %)"
    # What is saved must survive the project's JSON files, and holds no help text.
    text = json.dumps(data)
    assert "tooltip" not in text and "meaning" not in text


def test_a_table_that_cannot_be_compared_raises() -> None:
    with pytest.raises(ValueError):
        build_comparison_table(label="Empty", host=_cubic("Host", 5.0), inputs=[], jobs=[])


def test_comparison_inputs_skip_a_job_whose_structure_is_gone(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "inputs.mufinder", "Inputs")
    kept = _cubic("Relaxed", 5.1)
    service.add_structure(document, kept)

    inputs = comparison_inputs(document, [_job("gone", _cubic("Gone", 5.2)), _job("kept", kept)])

    assert [item.label for item in inputs] == ["DFT (kept)"]
    assert inputs[0].structure is kept


def test_parameter_differences_need_two_jobs_and_list_only_what_differs() -> None:
    structure = _cubic("Relaxed", 5.1)
    same = _job("a", structure, xc_functional="PBE", cut_off_energy=600)
    other = _job("b", structure, xc_functional="PBE", cut_off_energy=800)

    assert parameter_difference_rows([same]) == []
    assert parameter_difference_rows([same, other]) == [
        {"name": "cut_off_energy", "values": ["600", "800"]}
    ]


@pytest.mark.parametrize(
    ("value", "display"),
    [
        (ValueWithUnit(600.0, "eV"), "600.0 eV"),
        (
            BlockSection(name="EXTERNAL_PRESSURE", rows=("0 0 0", "0 0", "0")),
            "%BLOCK EXTERNAL_PRESSURE (3 rows)",
        ),
        ([1, 2, 3], "1, 2, 3"),
        ({"x": 1, "y": 2}, "x=1, y=2"),
        (None, ""),
        (True, "True"),
    ],
)
def test_parameter_differences_format_each_kind_of_value(value: object, display: str) -> None:
    structure = _cubic("Relaxed", 5.1)
    rows = parameter_difference_rows(
        [_job("a", structure, keyword=value), _job("b", structure, keyword="other")]
    )

    assert rows == [{"name": "keyword", "values": [display, "other"]}]


def test_a_new_table_reloads_without_definitions(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "values_only.mufinder"
    document = service.create_project(root, "ValuesOnly")
    host = _cubic("Host", 5.0)
    relaxed = _cubic("Relaxed", 5.1)
    service.add_structure(document, host)
    service.add_structure(document, relaxed)
    table = build_comparison_table(
        label="Comparison 1",
        host=host,
        inputs=[DftComparisonInput(label="DFT (a)", structure=relaxed, structure_id=relaxed.id)],
        jobs=[_job("a", relaxed)],
    )
    service.add_comparison_table(document, table)
    service.save_project(document)

    stored = json.loads(
        (root / "entities" / "comparison_tables" / f"{table.id}.json").read_text(encoding="utf-8")
    )
    assert "metric_definitions" not in json.dumps(stored)

    loaded = service.load_project(root).comparison_tables[table.id]
    assert isinstance(loaded, ComparisonTable)
    assert loaded.result_data["dft_columns"][0]["values"]["a"]["display"] == "5.1 (+2.00 %)"
