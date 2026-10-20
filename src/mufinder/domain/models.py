"""Core dataclasses for persisted MuFinder entities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from mufinder._version import current_version
from mufinder.domain.enums import FailureReason, JobStatus, WorkflowStage


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
    """Muon stopping-site representation across workflow stages.

    ``initial_fractional_position`` / ``initial_cartesian_position`` hold the
    position the site was generated (or staged) at, in the host cell, and never
    change when the site is relaxed. ``relaxed_fractional_position`` holds the
    relaxed muon position folded into the host cell while the site is linked to
    a completed relaxation run (``relaxation_run_id``), and ``None`` otherwise.
    The run's result stays the record; the site field is a convenience copy kept
    in step with it. :meth:`current_fractional_position` answers "where is the
    muon now".
    """

    id: UUID = field(default_factory=uuid4)
    label: str = ""
    status: str = "generated"
    initial_fractional_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    initial_cartesian_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    relaxed_fractional_position: list[float] | None = None
    generation_method: str | None = None
    source_generation_batch_id: UUID | None = None
    source_structure_id: UUID | None = None
    relaxation_run_id: UUID | None = None
    total_energy_eV: float | None = None
    host_displacement_max: float | None = None
    cluster_id: UUID | None = None
    symmetry_label: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def current_fractional_position(self) -> list[float]:
        """Return the relaxed host-cell position when known, else the initial one."""

        if self.relaxed_fractional_position is not None:
            return self.relaxed_fractional_position
        return self.initial_fractional_position

    def record_relaxation(
        self,
        run_id: UUID,
        *,
        total_energy_eV: float | None,
        fractional_position: Iterable[float] | None,
    ) -> None:
        """Mark the site relaxed by ``run_id`` with its host-cell final position."""

        self.status = "relaxed"
        self.relaxation_run_id = run_id
        self.total_energy_eV = total_energy_eV
        self.relaxed_fractional_position = (
            [float(value) for value in fractional_position] if fractional_position is not None else None
        )

    def clear_relaxation(self) -> None:
        """Unlink the site from its relaxation run and drop the values copied from it."""

        self.relaxation_run_id = None
        self.total_energy_eV = None
        self.relaxed_fractional_position = None
        if self.status == "relaxed":
            self.status = "staged"

    # Attribute aliases for the names the initial position had before it was
    # renamed; see the compatibility ledger. Constructors take the new names.
    @property
    def fractional_position(self) -> list[float]:
        """Alias of :attr:`initial_fractional_position`."""

        return self.initial_fractional_position

    @fractional_position.setter
    def fractional_position(self, value: list[float]) -> None:
        self.initial_fractional_position = value

    @property
    def cartesian_position(self) -> list[float]:
        """Alias of :attr:`initial_cartesian_position`."""

        return self.initial_cartesian_position

    @cartesian_position.setter
    def cartesian_position(self, value: list[float]) -> None:
        self.initial_cartesian_position = value


@dataclass(slots=True)
class Job:
    """A submitted or planned execution unit."""

    id: UUID = field(default_factory=uuid4)
    label: str = ""
    workflow_stage: WorkflowStage = WorkflowStage.MUON_RELAXATION
    calc_type: str = ""
    status: JobStatus = JobStatus.DEFINED
    execution_target: str = "local"
    # When the job was added to the project. Unlike ``submitted_at`` it never
    # changes, so it defines the job's stable position in the dashboard.
    created_at: datetime = field(default_factory=utc_now)
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    remote_job_id: str | None = None
    remote_directory: str | None = None
    num_processes: int | None = None
    tags: list[str] = field(default_factory=list)
    error_message: str | None = None
    # Classified cause of a failed or stopped job; ``None`` when no specific
    # cause was identified (or the job has not failed).
    failure_reason: FailureReason | None = None
    last_checked_at: datetime | None = None
    last_status_source: str | None = None
    last_status_message: str | None = None
    dismissed_at: datetime | None = None


def next_job_created_at(existing_jobs: Iterable[Job]) -> datetime:
    """Return a creation timestamp later than every job in ``existing_jobs``.

    Jobs created in a tight loop (a batch submission) can read the same clock
    value, especially on Windows where the system clock is coarse. Bumping past
    the latest existing timestamp keeps creation order strictly increasing, so
    sorting by ``created_at`` reproduces it exactly after a reload.
    """

    now = utc_now()
    latest = max((job.created_at for job in existing_jobs), default=None)
    if latest is not None and now <= latest:
        return latest + timedelta(microseconds=1)
    return now


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


#: ``Project.recommended_*_method`` value for a default the user typed in
#: directly rather than adopting from a convergence analysis.
MANUAL_DEFAULT_METHOD = "manual"

#: Saved-project schema version this code writes. Older projects are upgraded
#: on load by ``mufinder.persistence.migrations``.
PROJECT_SCHEMA_VERSION = 5


@dataclass(slots=True)
class Project:
    """Top-level project aggregate."""

    id: UUID = field(default_factory=uuid4)
    name: str = "Untitled Project"
    schema_version: int = PROJECT_SCHEMA_VERSION
    app_version: str = field(default_factory=current_version)
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
    recommended_cutoff_stress_threshold: float | None = None
    recommended_cutoff_cell_id: str | None = None
    recommended_cutoff_xc_functional: str | None = None
    recommended_kpoint_grid: tuple[int, int, int] | None = None
    recommended_kpoint_lattice: list[list[float]] | None = None
    recommended_kpoint_method: str | None = None
    recommended_kpoint_energy_threshold: float | None = None
    recommended_kpoint_force_threshold: float | None = None
    recommended_kpoint_stress_threshold: float | None = None
    recommended_kpoint_cell_id: str | None = None
    recommended_kpoint_xc_functional: str | None = None
    recommended_xc_functional: str | None = None
    notes: str = ""
