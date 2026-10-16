"""Decode persisted project JSON payloads into domain records.

Encoding is :func:`mufinder.persistence.codec.to_jsonable`: every project
entity is a dataclass that serializes field by field. The decoders here are its
inverse for each entity type. They also own the read-side repairs for projects
saved by older versions (renamed keys, legacy statuses, missing job creation
timestamps); each repair has an entry in the compatibility ledger
(``docs/compatibility_policy.rst``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from mufinder._version import current_version
from mufinder.domain.enums import FailureReason, JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    ComparisonTable,
    Job,
    MuonSite,
    Project,
    ProvenanceRecord,
    Result,
    Structure,
    utc_now,
)
from mufinder.persistence.codec import parse_datetime, parse_uuid

#: Status a freshly generated muon site loads with.
_GENERATED_SITE_STATUS = "generated"
#: Status older projects stored for freshly generated sites; loads as generated.
_LEGACY_CANDIDATE_SITE_STATUS = "candidate"


def _as_aware_utc(stamp: datetime) -> datetime:
    """Treat a naive timestamp as UTC so it can be compared with aware ones."""
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)


def normalized_xc_functional(value: object) -> str | None:
    """Return an uppercase XC functional name, or ``None`` when unset/blank."""

    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().upper()


def _failure_reason_from_payload(value: object) -> FailureReason | None:
    """Parse a persisted failure reason; unknown or missing values load as ``None``."""

    if not isinstance(value, str):
        return None
    try:
        return FailureReason(value)
    except ValueError:
        return None


def project_from_dict(payload: dict[str, Any]) -> Project:
    """Decode the project manifest; malformed recommended defaults load as ``None``."""

    raw_recommended_cutoff = payload.get("recommended_cutoff_energy")
    try:
        recommended_cutoff = float(raw_recommended_cutoff) if raw_recommended_cutoff is not None else None
    except (TypeError, ValueError):
        recommended_cutoff = None

    raw_recommended_method = payload.get("recommended_cutoff_method")
    recommended_method = None
    if isinstance(raw_recommended_method, str) and raw_recommended_method.strip():
        recommended_method = raw_recommended_method.strip()

    raw_energy_threshold = payload.get("recommended_cutoff_energy_threshold")
    try:
        recommended_energy_threshold = (
            float(raw_energy_threshold) if raw_energy_threshold is not None else None
        )
    except (TypeError, ValueError):
        recommended_energy_threshold = None

    raw_force_threshold = payload.get("recommended_cutoff_force_threshold")
    try:
        recommended_force_threshold = (
            float(raw_force_threshold) if raw_force_threshold is not None else None
        )
    except (TypeError, ValueError):
        recommended_force_threshold = None

    raw_stress_threshold = payload.get("recommended_cutoff_stress_threshold")
    try:
        recommended_stress_threshold = (
            float(raw_stress_threshold) if raw_stress_threshold is not None else None
        )
    except (TypeError, ValueError):
        recommended_stress_threshold = None

    raw_recommended_cutoff_cell_id = payload.get("recommended_cutoff_cell_id")
    recommended_cutoff_cell_id = None
    if isinstance(raw_recommended_cutoff_cell_id, str) and raw_recommended_cutoff_cell_id.strip():
        recommended_cutoff_cell_id = raw_recommended_cutoff_cell_id.strip()

    raw_recommended_kpoint_grid = payload.get("recommended_kpoint_grid")
    recommended_kpoint_grid: tuple[int, int, int] | None = None
    if isinstance(raw_recommended_kpoint_grid, list) and len(raw_recommended_kpoint_grid) == 3:
        try:
            recommended_kpoint_grid = (
                int(raw_recommended_kpoint_grid[0]),
                int(raw_recommended_kpoint_grid[1]),
                int(raw_recommended_kpoint_grid[2]),
            )
        except (TypeError, ValueError):
            recommended_kpoint_grid = None

    raw_recommended_kpoint_lattice = payload.get("recommended_kpoint_lattice")
    recommended_kpoint_lattice: list[list[float]] | None = None
    if isinstance(raw_recommended_kpoint_lattice, list) and len(raw_recommended_kpoint_lattice) == 3:
        parsed_lattice: list[list[float]] = []
        valid_lattice = True
        for row in raw_recommended_kpoint_lattice:
            if not isinstance(row, list) or len(row) != 3:
                valid_lattice = False
                break
            try:
                parsed_lattice.append([float(row[0]), float(row[1]), float(row[2])])
            except (TypeError, ValueError):
                valid_lattice = False
                break
        if valid_lattice:
            recommended_kpoint_lattice = parsed_lattice

    raw_recommended_kpoint_method = payload.get("recommended_kpoint_method")
    recommended_kpoint_method = None
    if isinstance(raw_recommended_kpoint_method, str) and raw_recommended_kpoint_method.strip():
        recommended_kpoint_method = raw_recommended_kpoint_method.strip()

    raw_kpoint_energy_threshold = payload.get("recommended_kpoint_energy_threshold")
    try:
        recommended_kpoint_energy_threshold = (
            float(raw_kpoint_energy_threshold) if raw_kpoint_energy_threshold is not None else None
        )
    except (TypeError, ValueError):
        recommended_kpoint_energy_threshold = None

    raw_kpoint_force_threshold = payload.get("recommended_kpoint_force_threshold")
    try:
        recommended_kpoint_force_threshold = (
            float(raw_kpoint_force_threshold) if raw_kpoint_force_threshold is not None else None
        )
    except (TypeError, ValueError):
        recommended_kpoint_force_threshold = None

    raw_kpoint_stress_threshold = payload.get("recommended_kpoint_stress_threshold")
    try:
        recommended_kpoint_stress_threshold = (
            float(raw_kpoint_stress_threshold) if raw_kpoint_stress_threshold is not None else None
        )
    except (TypeError, ValueError):
        recommended_kpoint_stress_threshold = None

    raw_recommended_kpoint_cell_id = payload.get("recommended_kpoint_cell_id")
    recommended_kpoint_cell_id = None
    if isinstance(raw_recommended_kpoint_cell_id, str) and raw_recommended_kpoint_cell_id.strip():
        recommended_kpoint_cell_id = raw_recommended_kpoint_cell_id.strip()

    return Project(
        recommended_cutoff_xc_functional=normalized_xc_functional(
            payload.get("recommended_cutoff_xc_functional")
        ),
        recommended_kpoint_xc_functional=normalized_xc_functional(
            payload.get("recommended_kpoint_xc_functional")
        ),
        recommended_xc_functional=normalized_xc_functional(
            payload.get("recommended_xc_functional")
        ),
        id=UUID(payload["id"]),
        name=payload["name"],
        schema_version=payload["schema_version"],
        app_version=payload.get("app_version", current_version()),
        created_at=parse_datetime(payload["created_at"]) or utc_now(),
        modified_at=parse_datetime(payload["modified_at"]) or utc_now(),
        host_structure_id=parse_uuid(payload.get("host_structure_id")),
        magnetic_structure=payload.get("magnetic_structure"),
        supercell_matrix=payload.get("supercell_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        recommended_cutoff_energy=recommended_cutoff,
        recommended_cutoff_method=recommended_method,
        recommended_cutoff_energy_threshold=recommended_energy_threshold,
        recommended_cutoff_force_threshold=recommended_force_threshold,
        recommended_cutoff_stress_threshold=recommended_stress_threshold,
        recommended_cutoff_cell_id=recommended_cutoff_cell_id,
        recommended_kpoint_grid=recommended_kpoint_grid,
        recommended_kpoint_lattice=recommended_kpoint_lattice,
        recommended_kpoint_method=recommended_kpoint_method,
        recommended_kpoint_energy_threshold=recommended_kpoint_energy_threshold,
        recommended_kpoint_force_threshold=recommended_kpoint_force_threshold,
        recommended_kpoint_stress_threshold=recommended_kpoint_stress_threshold,
        recommended_kpoint_cell_id=recommended_kpoint_cell_id,
        notes=payload.get("notes", ""),
    )


def structure_from_dict(payload: dict[str, Any]) -> Structure:
    """Decode one structure entity."""

    return Structure(
        id=UUID(payload["id"]),
        label=payload.get("label", ""),
        origin=payload.get("origin", "imported"),
        lattice_vectors=payload.get("lattice_vectors", []),
        species=payload.get("species", []),
        fractional_positions=payload.get("fractional_positions", []),
        source_file=payload.get("source_file"),
        parent_structure_id=parse_uuid(payload.get("parent_structure_id")),
        properties=payload.get("properties", {}),
    )


def calculation_from_dict(payload: dict[str, Any]) -> CalculationDefinition:
    """Decode one calculation definition entity."""

    return CalculationDefinition(
        id=UUID(payload["id"]),
        workflow_stage=WorkflowStage(payload["workflow_stage"]),
        calc_type=payload.get("calc_type", ""),
        parameters=payload.get("parameters", {}),
        input_structure_ids=[UUID(v) for v in payload.get("input_structure_ids", [])],
        input_site_ids=[UUID(v) for v in payload.get("input_site_ids", [])],
        custom_param_text=payload.get("custom_param_text"),
        created_at=parse_datetime(payload.get("created_at")) or utc_now(),
    )


def run_from_dict(payload: dict[str, Any]) -> CalculationRun:
    """Decode one calculation run entity."""

    return CalculationRun(
        id=UUID(payload["id"]),
        definition_id=parse_uuid(payload.get("definition_id")),
        job_id=parse_uuid(payload.get("job_id")),
        run_index=int(payload.get("run_index", 1)),
        status=JobStatus(payload.get("status", JobStatus.DEFINED.value)),
        started_at=parse_datetime(payload.get("started_at")),
        completed_at=parse_datetime(payload.get("completed_at")),
        result_id=parse_uuid(payload.get("result_id")),
        warnings=payload.get("warnings", []),
        convergence_achieved=payload.get("convergence_achieved"),
    )


def result_from_dict(payload: dict[str, Any]) -> Result:
    """Decode one result entity."""

    return Result(
        id=UUID(payload["id"]),
        calculation_run_id=parse_uuid(payload.get("calculation_run_id")),
        result_type=payload.get("result_type", ""),
        data=payload.get("data", {}),
        output_files=payload.get("output_files", []),
        derived_structure_ids=[UUID(v) for v in payload.get("derived_structure_ids", [])],
        created_at=parse_datetime(payload.get("created_at")) or utc_now(),
    )


def comparison_table_from_dict(payload: dict[str, Any]) -> ComparisonTable:
    """Decode one comparison table entity."""

    return ComparisonTable(
        id=UUID(payload["id"]),
        label=payload.get("label", ""),
        experimental_structure_id=parse_uuid(payload.get("experimental_structure_id")),
        dft_structure_ids=[UUID(v) for v in payload.get("dft_structure_ids", [])],
        dft_labels=[str(label) for label in payload.get("dft_labels", [])],
        result_data=payload.get("result_data", {}),
        created_at=parse_datetime(payload.get("created_at")) or utc_now(),
    )


def jobs_in_creation_order(
    payloads: list[dict[str, Any]],
    provenance: Sequence[ProvenanceRecord],
) -> dict[UUID, Job]:
    """Build the ``jobs`` mapping from job payloads, ordered by creation.

    Job files are read back sorted by filename, which is the job's random
    UUID, so the on-disk order carries no meaning. Sorting by
    ``(created_at, id)`` restores the order the jobs were created in, which
    is the order the dashboard shows them in during a session.

    Payloads saved before ``created_at`` existed get it backfilled by
    ``_legacy_job_created_at``; the next save persists the value.
    """

    jobs = [job_from_dict(payload) for payload in payloads]
    legacy_ids = {job.id for job, payload in zip(jobs, payloads) if payload.get("created_at") is None}
    if legacy_ids:
        first_mentioned = _job_first_mentions(legacy_ids, provenance)
        for job in jobs:
            if job.id in legacy_ids:
                job.created_at = _legacy_job_created_at(job, first_mentioned.get(job.id))
    jobs.sort(key=lambda job: (job.created_at, str(job.id)))
    return {job.id: job for job in jobs}


def _job_first_mentions(
    job_ids: set[UUID],
    provenance: Sequence[ProvenanceRecord],
) -> dict[UUID, datetime]:
    """Return when each job in ``job_ids`` first appears in provenance.

    The provenance log is append-only, so its records are chronological.
    A batch ``submit_jobs`` record lists its jobs in submission order under
    one timestamp; each job's position in that list is added as
    microseconds so the batch keeps its order instead of tying.
    """

    first_mentioned: dict[UUID, datetime] = {}
    for record in provenance:
        mentioned = [*record.output_entity_ids, *record.input_entity_ids]
        if record.job_id is not None:
            mentioned.append(record.job_id)
        for position, entity_id in enumerate(mentioned):
            if entity_id in job_ids and entity_id not in first_mentioned:
                first_mentioned[entity_id] = _as_aware_utc(record.timestamp) + timedelta(
                    microseconds=position
                )
    return first_mentioned


def _legacy_job_created_at(job: Job, first_mentioned: datetime | None) -> datetime:
    """Estimate ``created_at`` for a job saved before the field existed.

    Prefers the job's first provenance mention, then its earliest lifecycle
    timestamp. A job with neither (never submitted, no provenance) sorts
    after every dated job; ties fall back to the job id, which is stable.
    """

    if first_mentioned is not None:
        return first_mentioned
    lifecycle = [
        _as_aware_utc(stamp)
        for stamp in (job.submitted_at, job.started_at, job.completed_at, job.dismissed_at)
        if stamp is not None
    ]
    if lifecycle:
        return min(lifecycle)
    return job.created_at


def job_from_dict(payload: dict[str, Any]) -> Job:
    """Decode one job entity; a non-positive process count loads as ``None``."""

    raw_num_processes = payload.get("num_processes")
    num_processes = None
    if raw_num_processes is not None:
        try:
            parsed_num_processes = int(raw_num_processes)
        except (TypeError, ValueError):
            parsed_num_processes = 0
        if parsed_num_processes > 0:
            num_processes = parsed_num_processes
    return Job(
        id=UUID(payload["id"]),
        label=payload.get("label", ""),
        workflow_stage=WorkflowStage(payload["workflow_stage"]),
        calc_type=payload.get("calc_type", ""),
        status=JobStatus(payload["status"]),
        execution_target=payload.get("execution_target", "local"),
        created_at=_as_aware_utc(parse_datetime(payload.get("created_at")) or utc_now()),
        submitted_at=parse_datetime(payload.get("submitted_at")),
        started_at=parse_datetime(payload.get("started_at")),
        completed_at=parse_datetime(payload.get("completed_at")),
        remote_job_id=payload.get("remote_job_id"),
        remote_directory=payload.get("remote_directory"),
        num_processes=num_processes,
        tags=payload.get("tags", []),
        error_message=payload.get("error_message"),
        failure_reason=_failure_reason_from_payload(payload.get("failure_reason")),
        last_checked_at=parse_datetime(payload.get("last_checked_at")),
        last_status_source=payload.get("last_status_source"),
        last_status_message=payload.get("last_status_message"),
        dismissed_at=parse_datetime(payload.get("dismissed_at")),
    )


def site_from_dict(payload: dict[str, Any]) -> MuonSite:
    """Decode one muon site entity, applying the legacy status and energy-key repairs."""

    return MuonSite(
        id=UUID(payload["id"]),
        label=payload.get("label", ""),
        status=_site_status_from_payload(payload),
        initial_fractional_position=payload.get("initial_fractional_position", [0.0, 0.0, 0.0]),
        initial_cartesian_position=payload.get("initial_cartesian_position", [0.0, 0.0, 0.0]),
        relaxed_fractional_position=payload.get("relaxed_fractional_position"),
        generation_method=payload.get("generation_method"),
        source_generation_batch_id=parse_uuid(payload.get("source_generation_batch_id")),
        source_structure_id=parse_uuid(payload.get("source_structure_id")),
        relaxation_run_id=parse_uuid(payload.get("relaxation_run_id")),
        total_energy_eV=_site_total_energy_from_payload(payload),
        host_displacement_max=payload.get("host_displacement_max"),
        cluster_id=parse_uuid(payload.get("cluster_id")),
        symmetry_label=payload.get("symmetry_label"),
        properties=payload.get("properties", {}),
    )


def _site_total_energy_from_payload(payload: dict[str, Any]) -> float | None:
    """Read a site's relaxation total energy, accepting the pre-rename key.

    Projects saved before the field was renamed stored the same absolute
    total energy under ``relative_energy_eV``; see the compatibility ledger.
    """

    if "total_energy_eV" in payload:
        return payload["total_energy_eV"]
    return payload.get("relative_energy_eV")


def _site_status_from_payload(payload: dict[str, Any]) -> str:
    status = str(payload.get("status", _GENERATED_SITE_STATUS) or _GENERATED_SITE_STATUS)
    if status == _LEGACY_CANDIDATE_SITE_STATUS:
        return _GENERATED_SITE_STATUS
    return status


def provenance_from_dict(payload: dict[str, Any]) -> ProvenanceRecord:
    """Decode one provenance log record."""

    workflow_stage = payload.get("workflow_stage")
    return ProvenanceRecord(
        id=UUID(payload["id"]),
        timestamp=parse_datetime(payload.get("timestamp")) or utc_now(),
        action=payload.get("action", ""),
        workflow_stage=WorkflowStage(workflow_stage) if workflow_stage else None,
        actor=payload.get("actor", "system"),
        description=payload.get("description", ""),
        parameters=payload.get("parameters", {}),
        input_entity_ids=[UUID(v) for v in payload.get("input_entity_ids", [])],
        output_entity_ids=[UUID(v) for v in payload.get("output_entity_ids", [])],
        job_id=parse_uuid(payload.get("job_id")),
    )


#: Decoder for each entity directory whose files load independently, keyed by
#: directory name. Jobs are absent: :func:`jobs_in_creation_order` decodes them
#: together with the provenance log.
ENTITY_DECODERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "structures": structure_from_dict,
    "sites": site_from_dict,
    "calculations": calculation_from_dict,
    "runs": run_from_dict,
    "results": result_from_dict,
    "comparison_tables": comparison_table_from_dict,
}
