"""Core dataclasses for persisted MuFinder entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from mufinder.domain.enums import JobStatus, WorkflowStage


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


@dataclass(slots=True)
class Structure:
    """Crystallographic structure model."""

    id: UUID = field(default_factory=uuid4)
    label: str = ""
    origin: str = "imported"
    lattice_vectors: list[list[float]] = field(default_factory=list)
    species: list[str] = field(default_factory=list)
    fractional_positions: list[list[float]] = field(default_factory=list)
    source_file: str | None = None
    parent_structure_id: UUID | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CalculationDefinition:
    """A reusable definition of a scientific calculation."""

    id: UUID = field(default_factory=uuid4)
    workflow_stage: WorkflowStage = WorkflowStage.MUON_RELAXATION
    calc_type: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    input_structure_ids: list[UUID] = field(default_factory=list)
    input_site_ids: list[UUID] = field(default_factory=list)
    custom_param_text: str | None = None
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class CalculationRun:
    """A concrete execution attempt for one calculation definition."""

    id: UUID = field(default_factory=uuid4)
    definition_id: UUID | None = None
    job_id: UUID | None = None
    run_index: int = 1
    status: JobStatus = JobStatus.DEFINED
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result_id: UUID | None = None
    warnings: list[str] = field(default_factory=list)
    convergence_achieved: bool | None = None


@dataclass(slots=True)
class Result:
    """Structured result payload produced by a completed run."""

    id: UUID = field(default_factory=uuid4)
    calculation_run_id: UUID | None = None
    result_type: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    output_files: list[str] = field(default_factory=list)
    derived_structure_ids: list[UUID] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class ComparisonTable:
    """Persisted structure-comparison table for pristine-relaxation analysis."""

    id: UUID = field(default_factory=uuid4)
    label: str = ""
    experimental_structure_id: UUID | None = None
    dft_structure_ids: list[UUID] = field(default_factory=list)
    dft_labels: list[str] = field(default_factory=list)
    result_data: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class MuonSite:
    """Muon stopping-site representation across workflow stages."""

    id: UUID = field(default_factory=uuid4)
    label: str = ""
    status: str = "candidate"
    fractional_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    cartesian_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    generation_method: str | None = None
    source_generation_batch_id: UUID | None = None
    source_structure_id: UUID | None = None
    relaxation_run_id: UUID | None = None
    relative_energy_eV: float | None = None
    host_displacement_max: float | None = None
    cluster_id: UUID | None = None
    symmetry_label: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Job:
    """A submitted or planned execution unit."""

    id: UUID = field(default_factory=uuid4)
    label: str = ""
    workflow_stage: WorkflowStage = WorkflowStage.MUON_RELAXATION
    calc_type: str = ""
    status: JobStatus = JobStatus.DEFINED
    execution_target: str = "local"
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    remote_job_id: str | None = None
    remote_directory: str | None = None
    num_processes: int | None = None
    tags: list[str] = field(default_factory=list)
    error_message: str | None = None


@dataclass(slots=True)
class ProvenanceRecord:
    """Structured provenance event for reproducibility."""

    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=utc_now)
    action: str = ""
    workflow_stage: WorkflowStage | None = None
    actor: str = "system"
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    input_entity_ids: list[UUID] = field(default_factory=list)
    output_entity_ids: list[UUID] = field(default_factory=list)
    job_id: UUID | None = None


@dataclass(slots=True)
class Project:
    """Top-level project aggregate."""

    id: UUID = field(default_factory=uuid4)
    name: str = "Untitled Project"
    schema_version: int = 1
    app_version: str = "0.1.0"
    created_at: datetime = field(default_factory=utc_now)
    modified_at: datetime = field(default_factory=utc_now)
    host_structure_id: UUID | None = None
    magnetic_structure: dict[str, Any] | None = None
    supercell_matrix: list[list[int]] = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    recommended_cutoff_energy: float | None = None
    recommended_cutoff_method: str | None = None
    recommended_cutoff_energy_threshold: float | None = None
    recommended_cutoff_force_threshold: float | None = None
    recommended_cutoff_cell_id: str | None = None
    recommended_kpoint_grid: tuple[int, int, int] | None = None
    recommended_kpoint_lattice: list[list[float]] | None = None
    recommended_kpoint_method: str | None = None
    recommended_kpoint_energy_threshold: float | None = None
    recommended_kpoint_force_threshold: float | None = None
    recommended_kpoint_cell_id: str | None = None
    notes: str = ""
