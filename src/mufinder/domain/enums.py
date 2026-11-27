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
    """The stored state of a project job.

    An ``ACTIVE`` job also stores its :class:`JobActivity` and an ``ENDED`` job
    its :class:`EndReason`; the other states store neither.
    """

    DEFINED = "defined"
    ACTIVE = "active"
    COMPLETED = "completed"
    #: The job did not finish: it failed, was stopped or was cancelled.
    ENDED = "ended"


class JobActivity(StrEnum):
    """What an active job was last seen doing.

    Kept for display only: it follows the latest evidence and has no
    transitions of its own.
    """

    SUBMITTING = "submitting"
    QUEUED = "queued"
    RUNNING = "running"
    RETRIEVING = "retrieving"


class EndReason(StrEnum):
    """Why an ended job did not finish.

    Recorded on ``Job.end_reason`` when the job ends, from the user's
    cancellation, the scheduler state and the job's scheduler and CASTEP logs.
    ``UNKNOWN`` means no specific cause was identified.
    """

    WALLTIME = "walltime"
    OUT_OF_MEMORY = "out_of_memory"
    NODE_FAILURE = "node_failure"
    CASTEP_ERROR = "castep_error"
    OUTPUT_UNAVAILABLE = "output_unavailable"
    INTERRUPTED = "interrupted"
    SUBMISSION = "submission"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


#: Import alias for :class:`EndReason`, kept for one minor release
#: (see ``docs/compatibility_policy.rst``).
FailureReason = EndReason
