"""Tests for decoding persisted project payloads into domain records."""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from mufinder.domain.enums import FailureReason, JobStatus, WorkflowStage
from mufinder.domain.models import (
    CONVERGED_DEFAULT_METHOD,
    GenerationRun,
    GenerationRunMove,
    MANUAL_DEFAULT_METHOD,
    Job,
    MuonSite,
    ProvenanceRecord,
    Structure,
)
from mufinder.persistence import project_codec
from mufinder.persistence.codec import to_jsonable
from mufinder.services.project_document import ProjectDocument


def _job_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(uuid4()),
        "workflow_stage": WorkflowStage.CONVERGENCE_CUTOFF.value,
        "status": JobStatus.DEFINED.value,
    }
    payload.update(overrides)
    return payload


def test_job_monitor_metadata_defaults_when_absent_from_payload() -> None:
    # Projects saved before the monitor-metadata fields existed must load with
    # None defaults; the decoder tolerates the missing keys.
    job = project_codec.job_from_dict(_job_payload())

    assert job.last_checked_at is None
    assert job.last_status_source is None
    assert job.last_status_message is None
    assert job.dismissed_at is None


def test_failure_reason_loads_as_none_when_absent_or_unknown() -> None:
    base = _job_payload(status=JobStatus.FAILED.value)

    assert project_codec.job_from_dict(base).failure_reason is None
    assert (
        project_codec.job_from_dict({**base, "failure_reason": "cosmic_ray"}).failure_reason is None
    )
    assert (
        project_codec.job_from_dict({**base, "failure_reason": "out_of_memory"}).failure_reason
        == FailureReason.OUT_OF_MEMORY
    )


@pytest.mark.parametrize(
    ("raw", "expected"), [(4, 4), ("2", 2), (0, None), (-1, None), ("many", None), (None, None)]
)
def test_job_num_processes_keeps_only_positive_counts(raw: object, expected: int | None) -> None:
    assert project_codec.job_from_dict(_job_payload(num_processes=raw)).num_processes == expected


def test_job_round_trips_through_to_jsonable() -> None:
    job = Job(
        label="Roundtrip",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        status=JobStatus.FAILED,
        num_processes=8,
        failure_reason=FailureReason.WALLTIME,
    )

    decoded = project_codec.job_from_dict(to_jsonable(job))

    assert decoded == job


def test_site_from_dict_maps_legacy_candidate_status_and_energy_key() -> None:
    payload = {"id": str(uuid4()), "status": "candidate", "relative_energy_eV": -12.5}

    site = project_codec.site_from_dict(payload)

    assert site.status == "generated"
    assert site.total_energy_eV == pytest.approx(-12.5)


def test_site_from_dict_prefers_total_energy_key_and_defaults_blank_status() -> None:
    payload = {
        "id": str(uuid4()),
        "status": "",
        "total_energy_eV": -3.0,
        "relative_energy_eV": -9.0,
    }

    site = project_codec.site_from_dict(payload)

    assert site.status == "generated"
    assert site.total_energy_eV == pytest.approx(-3.0)


def test_relaxed_site_round_trips_both_positions_under_their_own_keys() -> None:
    site = MuonSite(
        label="Site 1",
        status="relaxed",
        initial_fractional_position=[0.1, 0.2, 0.3],
        initial_cartesian_position=[0.4, 0.8, 1.2],
        relaxed_fractional_position=[0.15, 0.25, 0.35],
    )

    payload = to_jsonable(site)

    assert payload["initial_fractional_position"] == [0.1, 0.2, 0.3]
    assert payload["initial_cartesian_position"] == [0.4, 0.8, 1.2]
    assert payload["relaxed_fractional_position"] == [0.15, 0.25, 0.35]
    assert "fractional_position" not in payload
    assert "cartesian_position" not in payload
    assert project_codec.site_from_dict(payload) == site


def test_site_round_trips_through_to_jsonable() -> None:
    site = MuonSite(
        label="Site 1",
        status="staged",
        initial_fractional_position=[0.1, 0.2, 0.3],
        total_energy_eV=-1.0,
    )

    assert project_codec.site_from_dict(to_jsonable(site)) == site


def test_project_from_dict_drops_malformed_recommended_defaults() -> None:
    payload = {
        "id": str(uuid4()),
        "name": "Defaults",
        "schema_version": 1,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        "modified_at": datetime(2026, 1, 2, tzinfo=UTC).isoformat(),
        "recommended_cutoff_energy": "not-a-number",
        "recommended_cutoff_method": "   ",
        "recommended_kpoint_grid": [4, "x", 4],
        "recommended_kpoint_lattice": [[1.0, 0.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0]],
        "recommended_xc_functional": " pbe ",
        "recommended_cutoff_xc_functional": "",
    }

    project = project_codec.project_from_dict(payload)

    assert project.recommended_cutoff_energy is None
    assert project.recommended_cutoff_method is None
    assert project.recommended_kpoint_grid is None
    assert project.recommended_kpoint_lattice is None
    assert project.recommended_xc_functional == "PBE"
    assert project.recommended_cutoff_xc_functional is None


@pytest.mark.parametrize(
    ("value", "expected"), [(" pbe ", "PBE"), ("LDA", "LDA"), ("  ", None), (None, None), (3, None)]
)
def test_normalized_xc_functional(value: object, expected: str | None) -> None:
    assert project_codec.normalized_xc_functional(value) == expected


def test_jobs_in_creation_order_backfills_legacy_jobs_from_provenance() -> None:
    first, second = uuid4(), uuid4()
    record = ProvenanceRecord(
        timestamp=datetime(2026, 3, 1, tzinfo=UTC),
        action="submit_jobs",
        output_entity_ids=[second, first],
    )
    payloads = [_job_payload(id=str(first)), _job_payload(id=str(second))]

    jobs = project_codec.jobs_in_creation_order(payloads, [record])

    # The batch record lists ``second`` first, so it was created first.
    assert list(jobs) == [second, first]
    assert jobs[second].created_at < jobs[first].created_at


def test_entity_decoders_cover_every_independently_loaded_document_mapping() -> None:
    mappings = {
        item.name
        for item in fields(ProjectDocument)
        if item.name not in {"project_root", "project", "jobs", "provenance", "ui_state"}
    }

    assert set(project_codec.ENTITY_DECODERS) == mappings


def test_structure_round_trips_through_entity_decoder() -> None:
    structure = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        parent_structure_id=uuid4(),
    )

    assert project_codec.ENTITY_DECODERS["structures"](to_jsonable(structure)) == structure


@pytest.mark.parametrize("stored", ["delta-threshold", "derivative", "asymptotic", "converged"])
def test_project_from_dict_reads_any_convergence_method_as_converged(stored: str) -> None:
    payload = {
        "id": str(uuid4()),
        "name": "Defaults",
        "schema_version": 6,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        "modified_at": datetime(2026, 1, 2, tzinfo=UTC).isoformat(),
        "recommended_cutoff_method": stored,
        "recommended_kpoint_method": stored,
    }

    project = project_codec.project_from_dict(payload)

    assert project.recommended_cutoff_method == CONVERGED_DEFAULT_METHOD
    assert project.recommended_kpoint_method == CONVERGED_DEFAULT_METHOD


def test_project_from_dict_keeps_a_manual_default() -> None:
    payload = {
        "id": str(uuid4()),
        "name": "Defaults",
        "schema_version": 6,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        "modified_at": datetime(2026, 1, 2, tzinfo=UTC).isoformat(),
        "recommended_cutoff_method": " manual ",
        "recommended_kpoint_method": "manual",
    }

    project = project_codec.project_from_dict(payload)

    assert project.recommended_cutoff_method == MANUAL_DEFAULT_METHOD
    assert project.recommended_kpoint_method == MANUAL_DEFAULT_METHOD


def test_a_generation_run_round_trips_and_reads_malformed_optional_values_as_none() -> None:
    run = GenerationRun(
        method="random",
        parameters={"count": 4},
        seed=7,
        uses_symmetry=True,
        symmetry_tolerance_ang=1e-3,
        space_group_symbol="P4_2/mnm",
        space_group_number=136,
        symmetry_operation_count=16,
        site_ids=[uuid4()],
        moves=[GenerationRunMove(anchor_fractional=[0.5, 0.5, 0.5], original_positions={})],
    )

    assert project_codec.generation_run_from_dict(to_jsonable(run)) == run

    sparse = project_codec.generation_run_from_dict(
        {"id": str(uuid4()), "seed": "not a seed", "min_muon_muon_ang": "x", "moves": "bad"}
    )
    assert (sparse.seed, sparse.min_muon_muon_ang, sparse.moves, sparse.method) == (
        None,
        None,
        [],
        "",
    )
