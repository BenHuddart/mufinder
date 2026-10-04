"""Classify why a failed or stopped job did not complete.

A job that fails on a cluster rarely says why in its own failure message: the
scheduler kills CASTEP at the walltime limit or on an out-of-memory event and
records the cause only in its own log (``slurm-<id>.out``), while CASTEP writes
its own errors to ``<seed>.err`` / ``<seed>.NNNN.err``. This module reads the
logs a job left in its local output directory (the cluster runner downloads
them before reporting a failure) and turns them into a
:class:`~mufinder.domain.enums.FailureReason` plus the log line that shows it.

Evidence is ranked: a scheduler kill (walltime, memory, node failure) explains
everything after it, so it wins over a CASTEP error file, which in turn wins
over a retrieval failure. Log files older than the job's submission are
ignored: a continuation reuses its output directory, and the previous run's
logs would otherwise be reported as this run's cause.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mufinder.domain.enums import FailureReason

#: Prefix of the cluster runner's message when output download or parsing fails.
RETRIEVAL_FAILURE_PREFIX = "Failed to retrieve or parse remote output"

#: Message for a scheduler-missing job that exhausts its retrieval attempts.
MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE = (
    "Job disappeared from the scheduler queue and its output could not be "
    "retrieved after 3 checks. The scheduler history may have expired, the "
    "queue status command may be filtered, or a different cluster may be "
    "connected. Retry retrieval or resubmit."
)

#: Message for a retry-retrieval request made while the job's cluster is offline.
RETRY_RETRIEVAL_NOT_CONNECTED_MESSAGE = "Cannot retry retrieval: not connected to the job's cluster."

#: Short label shown next to the job status in the dashboard. Kept to a word
#: or two so ``failed · <label>`` fits the Status column on a laptop screen;
#: the cell tooltip carries the full explanation.
FAILURE_REASON_LABELS: dict[FailureReason, str] = {
    FailureReason.WALLTIME: "walltime",
    FailureReason.OUT_OF_MEMORY: "memory",
    FailureReason.NODE_FAILURE: "node fail",
    FailureReason.CASTEP_ERROR: "CASTEP",
    FailureReason.OUTPUT_UNAVAILABLE: "no output",
    FailureReason.INTERRUPTED: "orphaned",
    FailureReason.SUBMISSION: "submit",
}

#: One-sentence explanation of each cause, for tooltips and the Inspect dialog.
FAILURE_REASON_SUMMARIES: dict[FailureReason, str] = {
    FailureReason.WALLTIME: "The scheduler stopped the job when it reached its walltime limit.",
    FailureReason.OUT_OF_MEMORY: "The job ran out of memory and the scheduler killed it.",
    FailureReason.NODE_FAILURE: "A compute node failed while the job was running.",
    FailureReason.CASTEP_ERROR: "CASTEP stopped with an error.",
    FailureReason.OUTPUT_UNAVAILABLE: "The job's output could not be retrieved.",
    FailureReason.INTERRUPTED: "MuFinder lost track of the job before it finished.",
    FailureReason.SUBMISSION: "The job could not be submitted.",
}

UNDIAGNOSED_SUMMARY = "No specific cause was found in the job's logs."

#: Bytes read from the end of each log. Scheduler kill messages are the last
#: thing a log receives, and a CASTEP error file is a few lines long.
LOG_TAIL_BYTES = 64 * 1024

# A file written within this window before submission still counts as fresh,
# allowing for coarse filesystem timestamps.
_FRESHNESS_SLACK = timedelta(seconds=2)

# Scheduler evidence in priority order. SLURM writes "*** JOB n ON node
# CANCELLED AT time DUE TO TIME LIMIT ***" and "Detected n oom-kill event(s)";
# its state names (TIMEOUT, OUT_OF_MEMORY, NODE_FAIL) appear in the runner's
# own messages and in sacct output. The PBS forms cover templates that wrap
# another scheduler.
_SCHEDULER_PATTERNS: tuple[tuple[FailureReason, re.Pattern[str]], ...] = (
    (FailureReason.WALLTIME, re.compile(r"DUE TO TIME LIMIT", re.IGNORECASE)),
    (FailureReason.WALLTIME, re.compile(r"\bTIMEOUT\b")),
    (FailureReason.WALLTIME, re.compile(r"walltime\s+\d+\s+exceeded\s+limit", re.IGNORECASE)),
    (FailureReason.WALLTIME, re.compile(r"time\s+limit\s+(?:exceeded|reached)", re.IGNORECASE)),
    (FailureReason.OUT_OF_MEMORY, re.compile(r"\boom[-_ ]kill", re.IGNORECASE)),
    (FailureReason.OUT_OF_MEMORY, re.compile(r"out-of-memory handler", re.IGNORECASE)),
    (FailureReason.OUT_OF_MEMORY, re.compile(r"exceeded\s+(?:job|step)\s+memory\s+limit", re.IGNORECASE)),
    (FailureReason.OUT_OF_MEMORY, re.compile(r"\bOUT_OF_MEMORY\b")),
    (FailureReason.OUT_OF_MEMORY, re.compile(r"job killed:\s+mem\b", re.IGNORECASE)),
    (FailureReason.NODE_FAILURE, re.compile(r"DUE TO NODE FAIL", re.IGNORECASE)),
    (FailureReason.NODE_FAILURE, re.compile(r"\bNODE_FAIL\b")),
)

_CASTEP_ERR_NOISE = re.compile(r"^current trace stack", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FailureDiagnosis:
    """The classified cause of a failure and the log line that shows it."""

    reason: FailureReason | None = None
    evidence: str = ""
    evidence_source: str = ""

    @property
    def summary(self) -> str:
        return failure_reason_summary(self.reason)


def failure_reason_label(reason: FailureReason | None) -> str:
    """Return the short dashboard label for ``reason`` (empty when unknown)."""

    return FAILURE_REASON_LABELS.get(reason, "") if reason is not None else ""


def failure_reason_summary(reason: FailureReason | None) -> str:
    """Return the one-sentence explanation of ``reason``."""

    if reason is None:
        return UNDIAGNOSED_SUMMARY
    return FAILURE_REASON_SUMMARIES[reason]


def diagnose_failure(
    message: str | None,
    log_files: Iterable[Path],
    *,
    not_before: datetime | None = None,
) -> FailureDiagnosis:
    """Classify a failure from its message and the job's log files.

    ``not_before`` is the job's submission time: log files last modified
    before it belong to an earlier run in the same directory and are skipped.
    Unreadable files are skipped too, so a diagnosis never raises.
    """

    logs = [(path, text) for path in log_files if (text := _fresh_log_text(path, not_before)) is not None]

    sources = [("failure message", message or ""), *((path.name, text) for path, text in logs)]
    scheduler = _scheduler_evidence(sources)
    if scheduler is not None:
        return scheduler

    for path, text in logs:
        if path.suffix != ".err":
            continue
        line = _first_castep_error_line(text)
        if line:
            return FailureDiagnosis(
                reason=FailureReason.CASTEP_ERROR, evidence=line, evidence_source=path.name
            )

    stripped = (message or "").strip()
    if stripped.startswith(RETRIEVAL_FAILURE_PREFIX) or stripped in {
        MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE,
        RETRY_RETRIEVAL_NOT_CONNECTED_MESSAGE,
    }:
        return FailureDiagnosis(
            reason=FailureReason.OUTPUT_UNAVAILABLE,
            evidence=stripped,
            evidence_source="failure message",
        )

    return FailureDiagnosis()


def find_failure_log_files(
    output_dir: Path,
    seed_names: Sequence[str],
    remote_job_id: str | None,
) -> list[Path]:
    """Return a job's scheduler and CASTEP error logs in ``output_dir``.

    Preferred files are ``<seed>.err``, the scheduler's ``*-<remote id>.out``
    and CASTEP's per-rank ``<seed>[_farmNNN].NNNN.err``. When none exist, any
    ``*.err`` or ``slurm-*.out`` file in the directory is returned instead.
    """

    seeds = list(dict.fromkeys(name for name in seed_names if name))
    discovered: list[Path] = []
    for seed_name in seeds:
        preferred = output_dir / f"{seed_name}.err"
        if preferred.is_file():
            discovered.append(preferred)

    remote_id = (remote_job_id or "").strip()
    safe_remote_id = bool(remote_id) and re.fullmatch(r"[A-Za-z0-9_.-]+", remote_id) is not None
    if safe_remote_id:
        discovered.extend(
            path for path in sorted(output_dir.glob(f"*-{remote_id}.out")) if path.is_file()
        )

    if not output_dir.is_dir():
        return list(dict.fromkeys(discovered))

    numbered_patterns = [re.compile(rf"{re.escape(seed)}(_farm\d+)?\.\d+\.err") for seed in seeds]
    for candidate in sorted(output_dir.iterdir()):
        if candidate.is_file() and any(pattern.fullmatch(candidate.name) for pattern in numbered_patterns):
            discovered.append(candidate)

    if discovered:
        return list(dict.fromkeys(discovered))

    fallback_patterns = ["*.err", "slurm-*.out"]
    if safe_remote_id:
        fallback_patterns.append(f"*-{remote_id}.out")
    fallback: list[Path] = []
    for pattern in fallback_patterns:
        fallback.extend(sorted(path for path in output_dir.glob(pattern) if path.is_file()))
    return list(dict.fromkeys(fallback))


def read_log_tail(path: Path, max_bytes: int = LOG_TAIL_BYTES) -> str:
    """Read at most the trailing ``max_bytes`` of ``path`` as text.

    Files at or under the limit are read in full, byte-for-byte identical to
    ``Path.read_text()``. Larger files are read from a tail offset only; the
    partial leading line is dropped and a truncation marker prefixed.
    """

    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        truncated = size > max_bytes
        handle.seek(size - max_bytes if truncated else 0)
        raw = handle.read()

    # ``read_text()`` applies universal-newline translation; replicate it.
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    if not truncated:
        return text

    _, _, remainder = text.partition("\n")
    return "[... truncated] ...\n" + remainder


def _fresh_log_text(path: Path, not_before: datetime | None) -> str | None:
    try:
        if not_before is not None:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            threshold = not_before if not_before.tzinfo is not None else not_before.replace(tzinfo=UTC)
            if modified < threshold - _FRESHNESS_SLACK:
                return None
        return read_log_tail(path)
    except OSError:
        return None


def _scheduler_evidence(sources: Sequence[tuple[str, str]]) -> FailureDiagnosis | None:
    """Return the highest-priority scheduler cause across ``(source, text)`` pairs."""

    split = [(source, [line.strip() for line in text.splitlines() if line.strip()]) for source, text in sources]
    for reason, pattern in _SCHEDULER_PATTERNS:
        for source, lines in split:
            for line in lines:
                if pattern.search(line):
                    return FailureDiagnosis(reason=reason, evidence=line, evidence_source=source)
    return None


def _first_castep_error_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not _CASTEP_ERR_NOISE.match(stripped):
            return stripped
    return ""
