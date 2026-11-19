"""What a cluster scheduler says about a job.

Shared by the scheduler backends, which map their own state names onto it, and
by :mod:`mufinder.services.job_reconcile`, which decides a job's state from it.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["SchedulerState"]


class SchedulerState(StrEnum):
    """What the cluster's queue and accounting say about a job."""

    #: The queue command failed, so the scheduler was not heard from.
    UNAVAILABLE = "unavailable"
    #: The job has no scheduler job ID to ask about.
    NO_JOB_ID = "no_job_id"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    #: Neither the queue nor accounting lists the job.
    NOT_LISTED = "not_listed"
