"""Enumerations shared across MuFinder domain entities."""

from __future__ import annotations

from enum import StrEnum


class WorkflowStage(StrEnum):
    """Known scientific workflow stages."""

    CONVERGENCE_CUTOFF = "convergence_cutoff"
    CONVERGENCE_KPOINT = "convergence_kpoint"
    PRISTINE_RELAXATION = "pristine_relaxation"
    UEP = "uep"
    MUON_GENERATION = "muon_generation"
    MUON_RELAXATION = "muon_relaxation"
    SITE_ANALYSIS = "site_analysis"
    MAGNETIC_FIELD = "magnetic_field"
    HYPERFINE_COUPLING = "hyperfine_coupling"
    QUANTUM_CORRECTIONS = "quantum_corrections"
    ENERGY_BARRIERS = "energy_barriers"


class JobStatus(StrEnum):
    """Lifecycle states for project jobs."""

    DEFINED = "defined"
    SUBMITTING = "submitting"
    QUEUED = "queued"
    RUNNING = "running"
    RETRIEVING = "retrieving"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureReason(StrEnum):
    """Why a failed or stopped job did not complete.

    Recorded on ``Job.failure_reason`` when the job reaches a terminal
    failure state, from the scheduler state and the job's scheduler and CASTEP
    logs. ``None`` on the job means no specific cause was identified.
    """

    WALLTIME = "walltime"
    OUT_OF_MEMORY = "out_of_memory"
    NODE_FAILURE = "node_failure"
    CASTEP_ERROR = "castep_error"
    OUTPUT_UNAVAILABLE = "output_unavailable"
    INTERRUPTED = "interrupted"
    SUBMISSION = "submission"


#: Statuses of a job that is still active in the local execution pipeline
#: (submitted but not yet stopped, completed, failed, or cancelled). Used for
#: cancel-eligibility, dashboard visibility, and reattachment across the app.
ACTIVE_JOB_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.SUBMITTING,
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.RETRIEVING,
    }
)
