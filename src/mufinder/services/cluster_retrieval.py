"""Look at the output of a cluster job the scheduler no longer runs.

The functions here download and classify one job's remote output over an open
:class:`~mufinder.services.ssh_client.SSHConnection`. They report what the
folder holds as a :class:`RetrievalResult` and decide nothing about the job:
:func:`~mufinder.services.job_reconcile.reconcile` does that.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import posixpath
import re

from mufinder.castep_io.output_parser import CastepResult, parse_castep_output
from mufinder.domain.enums import FailureReason
from mufinder.services.failure_diagnosis import diagnose_failure, find_failure_log_files
from mufinder.services.job_artifacts import ParsedJobArtifacts, RetrievalFlags
from mufinder.services.job_reconcile import OutputState
from mufinder.services.ssh_client import SSHConnection

REMOTE_STALE_OUTPUT_SLACK_SECONDS = 5.0
"""Slack allowed when ordering two remote mtimes against each other.

Both sides of the comparison live on the cluster's own filesystem, so no
allowance for the offset between this machine and the cluster is needed — only
for remote mtime granularity (1 s on ext3/ext4, 2 s on some network stores) and
the sub-second NTP drift between a login node writing the uploaded input and a
compute node writing the output. The staleness being detected is a whole run
old, so a few seconds of slack costs nothing.
"""

_SAFE_REMOTE_JOB_ID = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """What one look at a job's remote folder found.

    ``detail`` is CASTEP's error message for incomplete output, and the error
    for output that could not be parsed or a folder that could not be read.
    ``walltime_seen`` marks incomplete output whose scheduler log shows a
    walltime kill.
    """

    output: OutputState
    detail: str = ""
    walltime_seen: bool = False
    parsed: CastepResult | None = None
    artifacts: ParsedJobArtifacts | None = None


def retrieve_job_outputs(
    ssh: SSHConnection,
    *,
    remote_job_id: str,
    remote_dir: str,
    seed_name: str,
    local_output_dir: Path,
    retrieval: RetrievalFlags,
) -> RetrievalResult:
    """Download, parse and classify one job's remote output.

    The result is ``absent`` when the folder holds no ``.castep`` written by
    this submission, and ``unreachable`` when the folder could not be read.

    A continuation run reuses its remote directory, so the *previous* run's
    ``.castep`` and ``-out.cell`` are still sitting there and neither one's
    existence proves this run wrote it. Retrieval therefore orders both
    against the ``{seed}.param`` uploaded at submission — see
    :func:`submission_marker_mtime` and :func:`remote_output_predates_submission`.
    A stale primary ``.castep`` counts as no output (the alternative is completing
    the job on the previous run's success banner); a stale ``-out.cell`` is
    simply not downloaded, which leaves the local copy carrying its
    previous-retrieval mtime so the workflow controllers' local stale-output
    guard rejects it. Both comparisons fail open when the marker is
    unavailable.

    Task farming renames every output to ``<seed>_farmNNN.<ext>``. When the
    flags request more than one farm and ``<seed>_farm001.castep`` exists it
    is the primary output, and the transition-state band is read from
    ``<seed>_farm001.ts``.
    """

    def path(name: str) -> tuple[str, Path]:
        return posixpath.join(remote_dir, name), local_output_dir / name

    remote_param = posixpath.join(remote_dir, f"{seed_name}.param")
    remote_castep, local_castep = path(f"{seed_name}.castep")
    remote_farm_castep, local_farm_castep = path(f"{seed_name}_farm001.castep")
    remote_den_fmt, local_den_fmt = path(f"{seed_name}.den_fmt")
    remote_out_cell, local_out_cell = path(f"{seed_name}-out.cell")
    remote_phonon, local_phonon = path(f"{seed_name}.phonon")
    remote_magres, local_magres = path(f"{seed_name}.magres")
    # Task farming renames every output to <seed>_farmNNN.<ext>, so a farmed
    # NEB run writes no <seed>.ts at all. Every farm writes an identical,
    # complete band; farm 001 is the canonical copy. Download it under the
    # same name locally so collect_job_artifacts finds it where it looks.
    farm_count = retrieval.farm_count
    remote_ts, local_ts = path(f"{seed_name}_farm001.ts" if farm_count > 1 else f"{seed_name}.ts")

    try:
        submitted_at = submission_marker_mtime(ssh, remote_param)
        # Logs are (re)downloaded just below, so any local log older than
        # this moment is a previous run's leftover in a reused directory.
        logs_fetched_from = datetime.now(UTC)
        retrieve_error_files(
            ssh,
            remote_job_id=remote_job_id,
            remote_dir=remote_dir,
            seed_name=seed_name,
            local_output_dir=local_output_dir,
        )
        use_farm = farm_count > 1 and ssh.exists(remote_farm_castep)
        primary_remote = remote_farm_castep if use_farm else remote_castep
        if not ssh.exists(primary_remote) or remote_output_predates_submission(
            ssh, primary_remote, submitted_at
        ):
            # A missing .castep, or one older than this submission: the run
            # wrote nothing into the folder (it has not started, failed to
            # start, or died before CASTEP opened its log). A surviving
            # .castep is the PREVIOUS run's, success banner included, so
            # parsing it would complete this job on the old geometry under
            # parameters that were never applied.
            return RetrievalResult(OutputState.ABSENT)
        if use_farm:
            if ssh.exists(remote_castep):
                ssh.get(remote_castep, local_castep)
            ssh.get(remote_farm_castep, local_farm_castep)
            primary_output = local_farm_castep
        else:
            ssh.get(remote_castep, local_castep)
            primary_output = local_castep
        # A reused directory keeps the previous run's outputs, and downloading
        # one would restamp it with the retrieval time, defeating the local
        # staleness guard: outputs older than this submission are left behind.
        if (
            retrieval.retrieve_den_fmt
            and ssh.exists(remote_den_fmt)
            and not remote_output_predates_submission(ssh, remote_den_fmt, submitted_at)
        ):
            ssh.get(remote_den_fmt, local_den_fmt)
        if (
            retrieval.retrieve_out_cell
            and ssh.exists(remote_out_cell)
            and not remote_output_predates_submission(ssh, remote_out_cell, submitted_at)
        ):
            ssh.get(remote_out_cell, local_out_cell)
        if retrieval.retrieve_phonon and ssh.exists(remote_phonon):
            ssh.get(remote_phonon, local_phonon)
        if retrieval.retrieve_ts and ssh.exists(remote_ts):
            ssh.get(remote_ts, local_ts)
        if (
            retrieval.retrieve_magres
            and ssh.exists(remote_magres)
            and not remote_output_predates_submission(ssh, remote_magres, submitted_at)
        ):
            ssh.get(remote_magres, local_magres)
    except Exception as exc:  # noqa: BLE001 - reported as an unreachable folder
        return RetrievalResult(OutputState.UNREACHABLE, detail=str(exc))

    try:
        parsed = parse_castep_output(primary_output)
    except Exception as exc:  # noqa: BLE001 - reported as unreadable output
        return RetrievalResult(OutputState.UNREADABLE, detail=str(exc))

    if not parsed.completed:
        return RetrievalResult(
            OutputState.INCOMPLETE,
            detail=parsed.error_message or "",
            walltime_seen=_stopped_at_walltime(
                parsed,
                remote_job_id=remote_job_id,
                seed_name=seed_name,
                local_output_dir=local_output_dir,
                logs_fetched_from=logs_fetched_from,
            ),
            parsed=parsed,
        )

    # Parse the secondary outputs (relaxed cell / phonon / transition state)
    # here on the worker thread, exactly once, driven by the retrieval flags
    # this job tracked. The GUI handler decides failure from the carrier and
    # the owning controller consumes the pre-parsed data — neither re-reads
    # disk. Errors are captured on the carrier, never raised.
    artifacts = retrieval.collect(local_output_dir, seed_name)
    return RetrievalResult(OutputState.COMPLETE, parsed=parsed, artifacts=artifacts)


def submission_marker_mtime(ssh: SSHConnection, remote_param: str) -> float | None:
    """Return the remote mtime that marks when this submission happened.

    ``{seed}.param`` is uploaded fresh on every submission that does not
    set ``reuse_existing_files`` — including every continuation, and
    regardless of queue system (unlike ``run.sh``, which ``addqueue`` never
    receives). Its remote mtime is therefore a same-clock marker for "this
    run's inputs landed", which any output the run wrote must postdate.

    ``None`` means no ordering is available and every comparison against it
    fails open: the file is missing (a ``reuse_existing_files``
    resubmission never re-uploaded it), the SFTP server reported no mtime,
    or the connection object has no ``mtime`` primitive.
    """

    probe = getattr(ssh, "mtime", None)
    if probe is None:
        return None
    try:
        return probe(remote_param)
    except Exception:  # noqa: BLE001 - a stat failure must not fail retrieval
        return None


def remote_output_predates_submission(
    ssh: SSHConnection,
    remote_path: str,
    submitted_at: float | None,
) -> bool:
    """Return whether ``remote_path`` was last written before this submission.

    Both mtimes come from the cluster's own filesystem, so they are
    directly comparable up to ``REMOTE_STALE_OUTPUT_SLACK_SECONDS``. An
    unknown marker or an unreadable output mtime returns ``False`` — an
    output is only ever rejected on positive evidence that it is stale.
    """

    if submitted_at is None:
        return False
    probe = getattr(ssh, "mtime", None)
    if probe is None:
        return False
    try:
        modified = probe(remote_path)
    except Exception:  # noqa: BLE001 - see submission_marker_mtime
        return False
    if modified is None:
        return False
    return (submitted_at - modified) > REMOTE_STALE_OUTPUT_SLACK_SECONDS


def retrieve_error_files(
    ssh: SSHConnection,
    *,
    remote_job_id: str,
    remote_dir: str,
    seed_name: str,
    local_output_dir: Path,
) -> None:
    """Download the job's error and scheduler log files, best effort.

    A file that cannot be downloaded is skipped, and an unwritable local
    directory skips them all: these logs only feed failure diagnosis.
    """

    try:
        local_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    names = remote_error_file_names(
        ssh, remote_job_id=remote_job_id, remote_dir=remote_dir, seed_name=seed_name
    )
    for name in names:
        remote_path = posixpath.join(remote_dir, name)
        try:
            if ssh.exists(remote_path):
                ssh.get(remote_path, local_output_dir / name)
        except Exception:  # noqa: BLE001
            continue


def remote_error_file_names(
    ssh: SSHConnection,
    *,
    remote_job_id: str,
    remote_dir: str,
    seed_name: str,
) -> list[str]:
    """Return the error and scheduler log names to look for in ``remote_dir``.

    ``<seed>.err`` and ``slurm-<id>.out`` are always candidates; a directory
    listing adds CASTEP's per-rank ``<seed>[_farmNNN].NNNN.err`` files and any
    ``*-<id>.out`` scheduler log.
    """

    names: list[str] = [f"{seed_name}.err"]
    job_id = remote_job_id.strip()
    safe_job_id = bool(job_id) and _SAFE_REMOTE_JOB_ID.fullmatch(job_id) is not None
    if safe_job_id:
        names.append(f"slurm-{job_id}.out")
    if not hasattr(ssh, "listdir"):
        return names

    try:
        entries = ssh.listdir(remote_dir)
    except Exception:  # noqa: BLE001
        return names

    # CASTEP embeds the MPI rank in every error filename (io_parallel_filename),
    # and task farming additionally rewrites the seedname to <seed>_farmNNN
    # before stderr is opened — so a farmed run writes <seed>_farm002.0003.err
    # and never <seed>.0003.err. Accept the optional farm infix so failure
    # diagnostics come back for farmed NEB and phonon jobs too.
    numbered = sorted(
        str(name)
        for name in entries
        if re.fullmatch(rf"{re.escape(seed_name)}(_farm\d+)?\.\d+\.err", str(name))
    )
    for name in numbered:
        if name not in names:
            names.append(name)
    if safe_job_id:
        scheduler_outputs = sorted(
            str(name)
            for name in entries
            if re.fullmatch(rf".*-{re.escape(job_id)}\.out", str(name))
        )
        for name in scheduler_outputs:
            if name not in names:
                names.append(name)
    return names


def _stopped_at_walltime(
    parsed: CastepResult,
    *,
    remote_job_id: str,
    seed_name: str,
    local_output_dir: Path,
    logs_fetched_from: datetime | None = None,
) -> bool:
    """Return whether an incomplete run was killed at the walltime limit.

    A walltime kill leaves no trace in the ``.castep`` file: the scheduler
    records it in its own log, which :func:`retrieve_error_files` has already
    downloaded next to the output. Local logs older than
    ``logs_fetched_from`` were not part of that download — a continuation
    reuses its directory, so they can be an earlier run's — and are ignored.
    """

    log_files = find_failure_log_files(local_output_dir, [seed_name], remote_job_id)
    diagnosis = diagnose_failure(parsed.error_message, log_files, not_before=logs_fetched_from)
    return diagnosis.reason == FailureReason.WALLTIME
