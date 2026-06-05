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
