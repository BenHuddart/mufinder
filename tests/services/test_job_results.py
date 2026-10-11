"""Tests for reading scalar results from a job's latest run."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.models import CalculationRun, Project, Result
from mufinder.services.job_results import (
    extract_max_force,
    extract_max_stress,
    extract_total_energy,
    latest_result_for_job,
    max_force_for_job,
    max_stress_for_job,
    parse_numeric,
    total_energy_for_job,
)
from mufinder.services.project_service import ProjectDocument


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (1, 1.0),
        (-2.5, -2.5),
        ({"value": 3.0}, 3.0),
        ({"value": {"value": 4}}, 4.0),
        (True, None),
        (float("nan"), None),
        ("1.0", None),
        ({"unit": "eV"}, None),
    ],
)
def test_parse_numeric(candidate: object, expected: float | None) -> None:
    assert parse_numeric(candidate) == expected


def test_extractors_read_direct_keys_before_nested_groups() -> None:
    assert extract_total_energy({"final_energy": -10.0, "energies": {"total": -1.0}}) == -10.0
    assert extract_total_energy({"total_energy": None, "energies": {"final": {"value": -2.0}}}) == -2.0
    assert extract_max_force({"max_force_eva": 0.05}) == 0.05
    assert extract_max_force({"forces": {"maximum": 0.1}}) == 0.1
    assert extract_max_stress({"max_stress": float("nan"), "stress": {"max": 0.2}}) == 0.2
    assert extract_max_stress({"stress": [[0.0]]}) is None
    assert extract_total_energy(["not", "a", "mapping"]) is None


def _document(tmp_path: Path) -> ProjectDocument:
    return ProjectDocument(project_root=tmp_path, project=Project())


def _add_run(document: ProjectDocument, job_id, run_index: int, data: dict | None) -> None:
    result_id = None
    if data is not None:
        result = Result(data=data)
        document.results[result.id] = result
        result_id = result.id
    run = CalculationRun(job_id=job_id, run_index=run_index, result_id=result_id)
    document.runs[run.id] = run


def test_readers_use_the_highest_index_run(tmp_path: Path) -> None:
    document = _document(tmp_path)
    job_id = uuid4()
    _add_run(document, job_id, 2, {"total_energy": -20.0, "max_force": 0.01, "max_stress": 0.3})
    _add_run(document, job_id, 1, {"total_energy": -10.0, "max_force": 0.9, "max_stress": 0.9})
    _add_run(document, uuid4(), 5, {"total_energy": -99.0})

    assert total_energy_for_job(document, job_id) == -20.0
    assert max_force_for_job(document, job_id) == 0.01
    assert max_stress_for_job(document, job_id) == 0.3


def test_readers_return_none_without_a_latest_result(tmp_path: Path) -> None:
    document = _document(tmp_path)
    job_id = uuid4()

    assert total_energy_for_job(None, job_id) is None
    assert latest_result_for_job(document, job_id) is None

    _add_run(document, job_id, 1, {"total_energy": -10.0})
    _add_run(document, job_id, 2, None)

    assert latest_result_for_job(document, job_id) is None
    assert total_energy_for_job(document, job_id) is None
