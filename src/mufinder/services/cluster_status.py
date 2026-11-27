"""What the status bar says about the clusters the open project is using.

The cluster widgets only mean something while the project has cluster jobs a
queue refresh can still change: active ones, which the queue poll follows, and
ended ones whose output may still be on the cluster. This module derives
whether they show, what the connection text says and whether a queue refresh
has anything to poll, from the project's jobs and the connection state. The
main window only renders the result.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import Enum

from mufinder.domain.enums import JobStatus
from mufinder.domain.models import Job
from mufinder.services.job_monitoring import has_unretrieved_cluster_output, is_cluster_job

NO_CLUSTER_JOBS_TOOLTIP = (
    "There is nothing to check: the project has no active cluster jobs and no ended "
    "ones whose output may still be on a cluster."
)
NO_REMOTE_IDS_TOOLTIP = "There is nothing to check yet: no active cluster job has a scheduler job ID or a remote folder."
REFRESH_TOOLTIP = "Check the cluster queue now, reconnecting if the connection is closed."


class ClusterConnectionTone(Enum):
    """How the connection text should read at a glance."""

    NEUTRAL = "neutral"
    CONNECTED = "connected"
    LOST = "lost"


@dataclass(frozen=True)
class ClusterStatus:
    """The state of the status bar's cluster widgets."""

    visible: bool
    connection_text: str
    tone: ClusterConnectionTone
    can_poll: bool
    refresh_tooltip: str


def is_monitored_cluster_job(job: Job) -> bool:
    """Return whether queue checks follow ``job``: an active cluster job."""

    return is_cluster_job(job) and job.status == JobStatus.ACTIVE


def is_refreshable_cluster_job(job: Job) -> bool:
    """Return whether a queue refresh can change ``job``: active, or ended with output to fetch.

    An ended cluster job counts while
    :func:`~mufinder.services.job_monitoring.has_unretrieved_cluster_output`
    holds for it, dismissed or not: its output may still be on the cluster.
    """

    return is_monitored_cluster_job(job) or has_unretrieved_cluster_output(job)


def monitored_cluster_names(jobs: Iterable[Job]) -> list[str]:
    """Return the clusters with jobs the queue poll follows, first seen first."""

    return _cluster_names(job for job in jobs if is_monitored_cluster_job(job))


def refreshable_cluster_names(jobs: Iterable[Job]) -> list[str]:
    """Return the clusters with jobs a queue refresh can change, first seen first.

    These are the clusters ``Refresh Queue`` reconnects to when none is connected.
    """

    return _cluster_names(job for job in jobs if is_refreshable_cluster_job(job))


def _cluster_names(jobs: Iterable[Job]) -> list[str]:
    names: dict[str, str] = {}
    for job in jobs:
        target = job.execution_target.strip()
        names.setdefault(target.casefold(), target)
    return list(names.values())


def cluster_status(
    jobs: Iterable[Job],
    *,
    connected_cluster: str | None,
    lost_clusters: Collection[str] = (),
) -> ClusterStatus:
    """Return what the status bar shows for ``jobs``.

    The widgets show while the project has a job a refresh can change
    (:func:`is_refreshable_cluster_job`). ``connected_cluster`` is the cluster
    connected now, and ``lost_clusters`` the casefolded names of those whose
    connection dropped and has not come back. A lost cluster only reads as lost
    while it still has active jobs, since those are the jobs the dropped
    connection leaves unchecked; a cluster with only ended jobs reads as not
    connected.
    """

    refreshable = [job for job in jobs if is_refreshable_cluster_job(job)]
    if not refreshable:
        return ClusterStatus(
            visible=False,
            connection_text="",
            tone=ClusterConnectionTone.NEUTRAL,
            can_poll=False,
            refresh_tooltip=NO_CLUSTER_JOBS_TOOLTIP,
        )

    connected_key = (connected_cluster or "").strip().casefold()
    followed = {name.casefold() for name in monitored_cluster_names(refreshable)}
    connected: list[str] = []
    lost: list[str] = []
    not_connected: list[str] = []
    for name in refreshable_cluster_names(refreshable):
        key = name.casefold()
        if key == connected_key:
            connected.append(name)
        elif key in lost_clusters and key in followed:
            lost.append(name)
        else:
            not_connected.append(name)

    can_poll = any(
        (job.remote_job_id or "").strip() or (job.remote_directory or "").strip()
        for job in refreshable
    )
    if lost:
        tone = ClusterConnectionTone.LOST
    elif connected:
        tone = ClusterConnectionTone.CONNECTED
    else:
        tone = ClusterConnectionTone.NEUTRAL
    return ClusterStatus(
        visible=True,
        connection_text=_connection_text(connected, lost, not_connected),
        tone=tone,
        can_poll=can_poll,
        refresh_tooltip=REFRESH_TOOLTIP if can_poll else NO_REMOTE_IDS_TOOLTIP,
    )


def _connection_text(connected: list[str], lost: list[str], not_connected: list[str]) -> str:
    parts: list[str] = []
    if connected:
        parts.append(f"connected to {', '.join(connected)}")
    if lost:
        parts.append(f"connection to {', '.join(lost)} lost")
    if not_connected:
        parts.append(f"not connected to {', '.join(not_connected)}")
    text = "; ".join(parts)
    return text[:1].upper() + text[1:]
