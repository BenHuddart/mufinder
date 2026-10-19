"""Checks on the queues and walltimes a cluster submission asks for.

A cluster profile lists named queues (``ClusterQueue``), and a submission picks
one. The setup dialog and the submission dialog both validate queue names and
walltimes the same way, through the helpers here.

Layering: this lives in ``mufinder.services`` and imports only ``domain``; it
holds no widgets.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

from mufinder.domain.cluster import ClusterQueue

#: A queue (SLURM partition or addqueue queue) name. SLURM accepts a
#: comma-separated list of partitions, so commas are allowed; anything that
#: would need shell quoting in a ``#SBATCH`` line or an addqueue argument is not.
_QUEUE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+(?:,[A-Za-z0-9._-]+)*$")

#: SLURM time formats: ``M``, ``M:S``, ``H:M:S``, ``D-H``, ``D-H:M`` and
#: ``D-H:M:S``.
_WALLTIME_RE = re.compile(r"^(?:(?P<days>\d+)-)?(?P<fields>\d+(?::\d+){0,2})$")


def is_valid_queue_name(name: str) -> bool:
    """Return whether ``name`` can be written into a job script unquoted."""

    return bool(_QUEUE_NAME_RE.match(name.strip()))


def parse_walltime_seconds(walltime: str) -> int | None:
    """Return a SLURM-style walltime in seconds, or ``None`` if it is not one.

    Without a day part, one field is minutes, two are minutes and seconds and
    three are hours, minutes and seconds. With a day part (``D-...``), the
    fields are hours, hours and minutes, or hours, minutes and seconds.
    """

    match = _WALLTIME_RE.match(walltime.strip())
    if match is None:
        return None
    fields = [int(part) for part in match.group("fields").split(":")]
    days = match.group("days")
    if days is not None:
        hours, minutes, seconds = (fields + [0, 0])[:3]
        return ((int(days) * 24 + hours) * 60 + minutes) * 60 + seconds
    if len(fields) == 1:
        return fields[0] * 60
    if len(fields) == 2:
        return fields[0] * 60 + fields[1]
    hours, minutes, seconds = fields
    return (hours * 60 + minutes) * 60 + seconds


def walltime_issue(walltime: str, *, max_walltime: str = "", queue_name: str = "") -> str:
    """Return why ``walltime`` cannot be submitted, or ``""`` when it can.

    An empty walltime is not checked here; whether one is required depends on
    the scheduler. A walltime beyond the queue's ``max_walltime`` is refused,
    because the scheduler would reject the job or hold it forever.
    """

    text = walltime.strip()
    if not text:
        return ""
    seconds = parse_walltime_seconds(text)
    if seconds is None:
        return f"'{text}' is not a walltime. Use HH:MM:SS, or D-HH:MM:SS for days."
    limit_text = max_walltime.strip()
    limit = parse_walltime_seconds(limit_text) if limit_text else None
    if limit is not None and seconds > limit:
        queue = f"queue '{queue_name.strip()}'" if queue_name.strip() else "this queue"
        return f"The walltime {text} is longer than the {limit_text} maximum for {queue}."
    return ""


def queue_issues(queues: Sequence[ClusterQueue], *, profile_walltime: str = "") -> list[str]:
    """Return every reason the listed queues cannot be saved, in table order.

    A queue needs a name that :func:`is_valid_queue_name` accepts and that no
    other queue uses, readable walltimes, and positive cores per node and
    memory where it sets them. Its default walltime, or ``profile_walltime``
    when it sets none, must not exceed its maximum, because every submission
    on the queue would start above the limit.
    """

    issues: list[str] = []
    seen: set[str] = set()
    for position, queue in enumerate(queues, start=1):
        name = queue.name.strip()
        label = f"Queue '{name}'" if name else f"The queue in row {position}"
        if not name:
            issues.append(f"{label} has no name.")
        else:
            if not is_valid_queue_name(name):
                issues.append(
                    f"'{name}' is not a queue name. Use letters, digits, '.', '_' and '-', "
                    "with commas between several partitions."
                )
            if name in seen:
                issues.append(f"The queue '{name}' is listed twice.")
            seen.add(name)

        walltime = queue.default_walltime.strip()
        max_walltime = queue.max_walltime.strip()
        format_problems = [problem for problem in map(walltime_issue, (walltime, max_walltime)) if problem]
        issues.extend(f"{label}: {problem}" for problem in format_problems)
        effective_walltime = walltime or profile_walltime.strip()
        if max_walltime and not format_problems and not walltime_issue(effective_walltime):
            limit_problem = walltime_issue(effective_walltime, max_walltime=max_walltime, queue_name=name)
            if limit_problem and not walltime:
                limit_problem = (
                    f"{limit_problem} The queue uses the cluster's default walltime; "
                    "give it a default walltime of its own."
                )
            if limit_problem:
                issues.append(f"{label}: {limit_problem}")

        if queue.cores_per_node is not None and queue.cores_per_node <= 0:
            issues.append(f"{label}: cores per node must be a positive whole number.")
        if any(
            memory is not None and memory <= 0
            for memory in (queue.default_memory_gb, queue.default_mem_per_cpu_gb)
        ):
            issues.append(f"{label}: memory must be a positive number of GB.")
    return issues
