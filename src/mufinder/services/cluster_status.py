"""What the status bar says about the clusters the open project is using.

The cluster widgets only mean something while the project has cluster jobs the
queue poll still follows. This module derives whether they show, what the
connection text says and whether a queue refresh has anything to poll, from the
project's jobs and the connection state. The main window only renders the result.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import Enum

from mufinder.domain.enums import ACTIVE_JOB_STATUSES
from mufinder.domain.models import Job
from mufinder.services.job_execution_controller import awaits_scheduler_reappearance
from mufinder.services.job_monitoring import is_cluster_job

NO_CLUSTER_JOBS_TOOLTIP = "There is nothing to poll: the project has no active cluster jobs."
NO_REMOTE_IDS_TOOLTIP = "There is nothing to poll yet: no active cluster job has a remote job ID."
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
    """Return whether queue polls follow ``job``: an active cluster job or a watched one.

    A job awaiting scheduler reappearance is polled like an active one.
    """

    return is_cluster_job(job) and (
        job.status in ACTIVE_JOB_STATUSES or awaits_scheduler_reappearance(job)
    )


def monitored_cluster_names(jobs: Iterable[Job]) -> list[str]:
    """Return the clusters with jobs the queue poll follows, first seen first."""

    names: dict[str, str] = {}
    for job in jobs:
        if is_monitored_cluster_job(job):
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

    ``connected_cluster`` is the cluster connected now, and ``lost_clusters`` the
    casefolded names of those whose connection dropped and has not come back. A
    lost cluster only reads as lost while it still has jobs the poll follows,
    since those are the jobs the dropped connection leaves unchecked.
    """

    monitored = [job for job in jobs if is_monitored_cluster_job(job)]
    if not monitored:
        return ClusterStatus(
            visible=False,
            connection_text="",
            tone=ClusterConnectionTone.NEUTRAL,
            can_poll=False,
            refresh_tooltip=NO_CLUSTER_JOBS_TOOLTIP,
        )

    connected_key = (connected_cluster or "").strip().casefold()
    connected: list[str] = []
    lost: list[str] = []
    not_connected: list[str] = []
    for name in monitored_cluster_names(monitored):
        key = name.casefold()
        if key == connected_key:
            connected.append(name)
        elif key in lost_clusters:
            lost.append(name)
        else:
            not_connected.append(name)

    can_poll = any((job.remote_job_id or "").strip() for job in monitored)
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
