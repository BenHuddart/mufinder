"""Tests for remote output retrieval of a finished cluster job."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import mufinder.services.cluster_retrieval as cluster_retrieval_module
from mufinder.castep_io.output_parser import CastepResult
from mufinder.services.cluster_retrieval import (
    REMOTE_STALE_OUTPUT_SLACK_SECONDS,
    RetrievalOutcome,
    RetrievalResult,
    queue_observation,
    remote_error_file_names,
    remote_output_predates_submission,
    retrieve_error_files,
    retrieve_job_outputs,
    submission_marker_mtime,
)
from mufinder.services.failure_diagnosis import RETRIEVAL_FAILURE_PREFIX
from mufinder.services.job_artifacts import RetrievalFlags
from tests.support.cluster_ssh import FakeSSH, OutputUnavailableSSH, TimestampedFakeSSH

_REMOTE_DIR = "/remote/job"


def _retrieve(
    ssh: FakeSSH,
    local_output_dir: Path,
    *,
    remote_job_id: str = "125",
    retrieval: RetrievalFlags = RetrievalFlags(),
    missing_from_queue: bool = False,
) -> RetrievalResult:
    return retrieve_job_outputs(
        ssh,
        remote_job_id=remote_job_id,
        remote_dir=_REMOTE_DIR,
        seed_name="seed",
        local_output_dir=local_output_dir,
        retrieval=retrieval,
        missing_from_queue=missing_from_queue,
    )


def _parse_as(monkeypatch: pytest.MonkeyPatch, result: CastepResult) -> list[Path]:
    """Make every parse return ``result``; return the list of parsed paths."""

    seen: list[Path] = []

    def _parse(path: Path) -> CastepResult:
        seen.append(Path(path))
        return result

    monkeypatch.setattr(cluster_retrieval_module, "parse_castep_output", _parse)
    return seen


_COMPLETED = CastepResult(completed=True, total_energy=-1.0)


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("missing_from_queue", "observation"),
    [
        (False, "output retrieved and parsed"),
        (True, "missing from scheduler queue; output retrieved and parsed"),
    ],
)
def test_completed_output_is_parsed_with_its_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_from_queue: bool, observation: str
) -> None:
    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    _parse_as(monkeypatch, _COMPLETED)

    result = _retrieve(
        ssh,
        tmp_path,
        retrieval=RetrievalFlags(retrieve_out_cell=True),
        missing_from_queue=missing_from_queue,
    )

    assert result.outcome is RetrievalOutcome.COMPLETED
    assert result.observation == observation
    assert result.parsed is _COMPLETED
    assert result.failure is None
    # The flags asked for a relaxed cell the run did not write.
    assert result.artifacts is not None and result.artifacts.cell_missing is True


def test_incomplete_output_fails_with_the_castep_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    _parse_as(monkeypatch, CastepResult(completed=False, error_message="not converged"))

    result = _retrieve(ssh, tmp_path, missing_from_queue=True)

    assert result.outcome is RetrievalOutcome.FAILED
    assert result.failure is not None
    assert result.failure.message == "not converged"
    assert result.failure.retry_eligible is False
    assert result.observation == "missing from scheduler queue; output incomplete; job failed"


def test_missing_job_cut_off_during_stress_is_not_completed(tmp_path: Path) -> None:
    # Regression: a job that vanished from the queue (walltime, OOM, or a
    # transient squeue omission) while CASTEP was computing the stress tensor
    # has "Final energy" in its output but no stress and no closing summary.
    # It must not reach the analysis as a completed run.
    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = (
        b" Run started: Fri, 02 Oct 2026 10:00:00 +0100\n"
        b" stress calculation                             : on\n"
        b"Final energy, E             =  -1363.109714972     eV\n"
        b"NB est. 0K energy (E-0.5TS)      =  -1363.109714972     eV\n"
        b" ******************************** Symmetrised Forces ********************************\n"
        b" * Si              1      0.00000      0.00000      0.00000 *\n"
        b" ************************************************************************************\n"
    )

    result = _retrieve(ssh, tmp_path, remote_job_id="121", missing_from_queue=True)

    assert result.outcome is RetrievalOutcome.FAILED
    assert result.failure is not None
    assert result.failure.retry_eligible is False


@pytest.mark.parametrize("missing_from_queue", [False, True])
def test_parse_exception_is_a_final_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_from_queue: bool
) -> None:
    """Output that downloaded but cannot be read gets no grace window."""

    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"castep output"

    def _raise(_path: Path) -> CastepResult:
        raise ValueError("parse boom")

    monkeypatch.setattr(cluster_retrieval_module, "parse_castep_output", _raise)

    result = _retrieve(ssh, tmp_path, missing_from_queue=missing_from_queue)

    assert result.outcome is RetrievalOutcome.FAILED
    assert result.failure is not None
    assert result.failure.message == f"{RETRIEVAL_FAILURE_PREFIX}: parse boom"
    assert result.failure.retry_eligible is False
    assert result.observation == queue_observation(
        "output parsing failed", missing_from_queue=missing_from_queue
    )


@pytest.mark.parametrize("missing_from_queue", [False, True])
def test_unavailable_output_is_retry_eligible_only_when_missing_from_queue(
    tmp_path: Path, missing_from_queue: bool
) -> None:
    ssh = OutputUnavailableSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"castep output"

    result = _retrieve(ssh, tmp_path, missing_from_queue=missing_from_queue)

    assert result.outcome is RetrievalOutcome.FAILED
    assert result.failure is not None
    assert result.failure.message.startswith(f"{RETRIEVAL_FAILURE_PREFIX}: SFTP download failed")
    assert result.failure.retry_eligible is missing_from_queue
    assert result.observation == queue_observation(
        "output retrieval failed", missing_from_queue=missing_from_queue
    )


def test_incomplete_walltime_run_is_stopped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A walltime kill leaves nothing in the .castep file; only the scheduler
    # log (downloaded during retrieval) records it.
    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"partial"
    ssh.remote_files["/remote/job/slurm-130.out"] = (
        b"slurmstepd: error: *** JOB 130 ON node01 CANCELLED AT 2026-10-04T10:00:00 "
        b"DUE TO TIME LIMIT ***\n"
    )
    _parse_as(monkeypatch, CastepResult(completed=False, error_message=None))

    result = _retrieve(ssh, tmp_path, remote_job_id="130")

    assert result.outcome is RetrievalOutcome.STOPPED
    assert result.failure is None
    assert result.observation == "output incomplete; stopped (walltime)"


def test_incomplete_without_scheduler_evidence_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A CASTEP error that merely mentions wall time is not a walltime stop."""

    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"partial"
    _parse_as(
        monkeypatch,
        CastepResult(completed=False, error_message="ERROR: wall clock estimate failed"),
    )

    result = _retrieve(ssh, tmp_path, remote_job_id="131")

    assert result.outcome is RetrievalOutcome.FAILED
    assert result.failure is not None
    assert result.failure.message == "ERROR: wall clock estimate failed"


def test_an_earlier_runs_scheduler_log_is_ignored_when_classifying(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A continuation reuses its directory; the previous run's walltime log is not this run's."""

    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"partial"
    stale_log = tmp_path / "slurm-200.out"
    stale_log.write_text(
        "slurmstepd: error: *** JOB 200 CANCELLED DUE TO TIME LIMIT ***\n", encoding="utf-8"
    )
    an_hour_ago = stale_log.stat().st_mtime - 3600
    os.utime(stale_log, (an_hour_ago, an_hour_ago))
    _parse_as(monkeypatch, CastepResult(completed=False, error_message=None))

    result = _retrieve(ssh, tmp_path, remote_job_id="201")

    assert result.outcome is RetrievalOutcome.FAILED


def test_incomplete_output_without_a_castep_error_has_a_default_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"partial"
    _parse_as(monkeypatch, CastepResult(completed=False, error_message=None))

    result = _retrieve(ssh, tmp_path)

    assert result.failure is not None
    assert result.failure.message == "CASTEP did not report successful completion."


# ---------------------------------------------------------------------------
# Which files are downloaded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "remote_name", "content"),
    [
        (RetrievalFlags(retrieve_den_fmt=True), "seed.den_fmt", b"density output"),
        (RetrievalFlags(retrieve_phonon=True), "seed.phonon", b"phonon output"),
        (RetrievalFlags(retrieve_ts=True), "seed.ts", b"ts output"),
        (RetrievalFlags(retrieve_out_cell=True), "seed-out.cell", b"relaxed cell"),
        (RetrievalFlags(retrieve_magres=True), "seed.magres", b"magres output"),
    ],
)
def test_requested_secondary_output_is_downloaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flags: RetrievalFlags,
    remote_name: str,
    content: bytes,
) -> None:
    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    ssh.remote_files[f"/remote/job/{remote_name}"] = content
    _parse_as(monkeypatch, _COMPLETED)

    _retrieve(ssh, tmp_path, retrieval=flags)

    assert (tmp_path / remote_name).read_bytes() == content


def test_secondary_outputs_are_skipped_when_not_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    ssh.remote_files["/remote/job/seed.den_fmt"] = b"density output"
    ssh.remote_files["/remote/job/seed-out.cell"] = b"relaxed cell"
    ssh.remote_files["/remote/job/seed.phonon"] = b"phonon output"
    ssh.remote_files["/remote/job/seed.ts"] = b"ts output"
    ssh.remote_files["/remote/job/seed.magres"] = b"magres output"
    _parse_as(monkeypatch, _COMPLETED)

    _retrieve(
        ssh, tmp_path, retrieval=RetrievalFlags(retrieve_den_fmt=False, retrieve_out_cell=False)
    )

    for name in ("seed.den_fmt", "seed-out.cell", "seed.phonon", "seed.ts", "seed.magres"):
        assert not (tmp_path / name).exists()


def test_task_farm_primary_output_is_parsed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"minimal output"
    ssh.remote_files["/remote/job/seed_farm001.castep"] = b"farmed output"
    ssh.remote_files["/remote/job/seed.phonon"] = b"phonon output"
    seen = _parse_as(monkeypatch, _COMPLETED)

    _retrieve(
        ssh,
        tmp_path,
        retrieval=RetrievalFlags(retrieve_phonon=True, task_farming_enabled=True, num_farms=4),
    )

    assert seen == [tmp_path / "seed_farm001.castep"]
    assert (tmp_path / "seed_farm001.castep").read_bytes() == b"farmed output"
    assert (tmp_path / "seed.castep").read_bytes() == b"minimal output"


def test_farmed_ts_file_is_downloaded_under_its_farm_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A farmed NEB run writes the band to seed_farm001.ts and no seed.ts at all,
    # so retrieval must ask for the farmed name or the job fails with a missing
    # .ts despite having completed successfully.
    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    ssh.remote_files["/remote/job/seed_farm001.castep"] = b"farmed output"
    ssh.remote_files["/remote/job/seed_farm001.ts"] = b"farmed ts output"
    _parse_as(monkeypatch, _COMPLETED)

    _retrieve(
        ssh,
        tmp_path,
        retrieval=RetrievalFlags(retrieve_ts=True, task_farming_enabled=True, num_farms=3),
    )

    assert (tmp_path / "seed_farm001.ts").read_bytes() == b"farmed ts output"
    assert not (tmp_path / "seed.ts").exists()


def test_numbered_error_files_are_downloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    ssh.remote_files["/remote/job/seed.0001.err"] = b"rank one"
    ssh.remote_files["/remote/job/seed.0002.err"] = b"rank two"
    _parse_as(monkeypatch, _COMPLETED)

    _retrieve(ssh, tmp_path)

    assert (tmp_path / "seed.0001.err").read_bytes() == b"rank one"
    assert (tmp_path / "seed.0002.err").read_bytes() == b"rank two"


def test_farmed_error_files_are_downloaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Task farming rewrites the seedname before stderr is opened, so error files
    # gain a _farmNNN infix on top of the usual MPI-rank suffix.
    ssh = FakeSSH()
    ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    ssh.remote_files["/remote/job/seed_farm001.0000.err"] = b"farm 1 stderr"
    ssh.remote_files["/remote/job/seed_farm002.0001.err"] = b"farm 2 stderr"
    _parse_as(monkeypatch, _COMPLETED)

    _retrieve(ssh, tmp_path, retrieval=RetrievalFlags(task_farming_enabled=True, num_farms=2))

    assert (tmp_path / "seed_farm001.0000.err").read_bytes() == b"farm 1 stderr"
    assert (tmp_path / "seed_farm002.0001.err").read_bytes() == b"farm 2 stderr"


def test_remote_error_file_names_list_logs_for_a_safe_job_id() -> None:
    ssh = FakeSSH()
    for name in (
        "seed.0002.err",
        "seed.0001.err",
        "seed_farm003.0000.err",
        "seed.other.err",
        "castep.mpi-703535.out",
        "other-1.out",
    ):
        ssh.remote_files[f"/remote/job/{name}"] = b""

    names = remote_error_file_names(
        ssh, remote_job_id=" 703535 ", remote_dir=_REMOTE_DIR, seed_name="seed"
    )

    assert names == [
        "seed.err",
        "slurm-703535.out",
        "seed.0001.err",
        "seed.0002.err",
        "seed_farm003.0000.err",
        "castep.mpi-703535.out",
    ]


def test_remote_error_file_names_skip_scheduler_logs_for_an_unsafe_job_id() -> None:
    ssh = FakeSSH()
    ssh.remote_files["/remote/job/x-1;rm.out"] = b""

    names = remote_error_file_names(
        ssh, remote_job_id="1;rm", remote_dir=_REMOTE_DIR, seed_name="seed"
    )

    assert names == ["seed.err"]


def test_remote_error_file_names_without_a_listing_are_the_fixed_names() -> None:
    class _NoListing:
        pass

    names = remote_error_file_names(
        _NoListing(), remote_job_id="42", remote_dir=_REMOTE_DIR, seed_name="seed"
    )

    assert names == ["seed.err", "slurm-42.out"]


def test_retrieve_error_files_skips_a_file_that_fails_to_download(tmp_path: Path) -> None:
    class _FlakySSH(FakeSSH):
        def get(self, remote_path: str, local_path: Path) -> None:
            if remote_path.endswith("seed.err"):
                raise OSError("permission denied")
            super().get(remote_path, local_path)

    ssh = _FlakySSH()
    ssh.remote_files["/remote/job/seed.err"] = b"unreadable"
    ssh.remote_files["/remote/job/slurm-7.out"] = b"scheduler log"
    local_dir = tmp_path / "created"

    retrieve_error_files(
        ssh, remote_job_id="7", remote_dir=_REMOTE_DIR, seed_name="seed", local_output_dir=local_dir
    )

    assert not (local_dir / "seed.err").exists()
    assert (local_dir / "slurm-7.out").read_bytes() == b"scheduler log"


# ---------------------------------------------------------------------------
# Continuation runs: ordering outputs against the submission marker
# ---------------------------------------------------------------------------


def _seed_continuation_files(
    ssh: TimestampedFakeSSH,
    *,
    submitted_at: float,
    castep_offset: float,
    out_cell_offset: float,
) -> None:
    """Seed a reused remote directory relative to the submission marker.

    ``{seed}.param`` is uploaded fresh at submission, so its remote mtime is
    the marker every output of this run must postdate. Offsets are relative to
    it, on the cluster's own clock.
    """

    ssh.remote_files["/remote/job/seed.param"] = b"param"
    ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    ssh.remote_files["/remote/job/seed-out.cell"] = b"relaxed cell"
    ssh.mtimes["/remote/job/seed.param"] = submitted_at
    ssh.mtimes["/remote/job/seed.castep"] = submitted_at + castep_offset
    ssh.mtimes["/remote/job/seed-out.cell"] = submitted_at + out_cell_offset


_CONTINUATION_FLAGS = RetrievalFlags(retrieve_out_cell=True)


def test_out_cell_predating_this_submission_is_not_downloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A continuation reuses its remote directory, so the previous run's
    ``-out.cell`` is still there. Downloading it would restamp the local copy
    with the retrieval time and defeat the local staleness guard, so an
    ``-out.cell`` older than this submission's ``.param`` is treated as
    absent."""

    ssh = TimestampedFakeSSH()
    _seed_continuation_files(
        ssh, submitted_at=1_700_000_000.0, castep_offset=120.0, out_cell_offset=-3600.0
    )
    _parse_as(monkeypatch, _COMPLETED)

    result = _retrieve(ssh, tmp_path, remote_job_id="900", retrieval=_CONTINUATION_FLAGS)

    assert ("get", "/remote/job/seed-out.cell") not in ssh.commands
    assert not (tmp_path / "seed-out.cell").exists()
    # The .castep is this run's, so the job still completes here; the relaxed
    # cell is reported missing exactly as it would be if the remote file did
    # not exist, which is what the workflow layer fails on.
    assert result.outcome is RetrievalOutcome.COMPLETED
    assert result.artifacts is not None and result.artifacts.cell_missing is True


def test_previously_downloaded_out_cell_is_left_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real continuation case: a local ``-out.cell`` fetched by the
    PREVIOUS retrieval is already on disk. Skipping the download leaves it
    carrying that earlier retrieval's mtime, which is what lets
    ``workflow_controllers.base.output_predates_run`` reject it for cluster jobs."""

    ssh = TimestampedFakeSSH()
    _seed_continuation_files(
        ssh, submitted_at=1_700_000_000.0, castep_offset=120.0, out_cell_offset=-3600.0
    )
    local_out_cell = tmp_path / "seed-out.cell"
    local_out_cell.write_bytes(b"previous run geometry")
    modified_before = local_out_cell.stat().st_mtime
    _parse_as(monkeypatch, _COMPLETED)

    _retrieve(ssh, tmp_path, remote_job_id="900", retrieval=_CONTINUATION_FLAGS)

    assert local_out_cell.read_bytes() == b"previous run geometry"
    assert local_out_cell.stat().st_mtime == modified_before


@pytest.mark.parametrize("out_cell_offset", [0.0, 60.0, 3600.0])
def test_out_cell_written_by_this_run_is_downloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, out_cell_offset: float
) -> None:
    """The other direction of the comparison: an ``-out.cell`` at or after the
    submission marker is this run's output and is downloaded normally."""

    ssh = TimestampedFakeSSH()
    _seed_continuation_files(
        ssh,
        submitted_at=1_700_000_000.0,
        castep_offset=out_cell_offset,
        out_cell_offset=out_cell_offset,
    )
    _parse_as(monkeypatch, _COMPLETED)

    _retrieve(ssh, tmp_path, remote_job_id="900", retrieval=_CONTINUATION_FLAGS)

    assert (tmp_path / "seed-out.cell").read_bytes() == b"relaxed cell"


@pytest.mark.parametrize("missing_from_queue", [False, True])
def test_castep_predating_this_submission_is_an_unavailable_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_from_queue: bool
) -> None:
    """The failed-to-start continuation: CASTEP never ran, so the ``.castep``
    left in the reused directory still ends with the PREVIOUS run's success
    banner. Parsing it would complete the job on the old geometry under
    parameters that were never applied, so retrieval aborts before the parse,
    and a job missing from the queue keeps its grace window."""

    ssh = TimestampedFakeSSH()
    _seed_continuation_files(
        ssh, submitted_at=1_700_000_000.0, castep_offset=-3600.0, out_cell_offset=-3600.0
    )
    seen = _parse_as(monkeypatch, _COMPLETED)

    result = _retrieve(
        ssh,
        tmp_path,
        remote_job_id="900",
        retrieval=_CONTINUATION_FLAGS,
        missing_from_queue=missing_from_queue,
    )

    assert seen == []
    assert result.outcome is RetrievalOutcome.FAILED
    assert result.failure is not None
    assert "predates this submission" in result.failure.message
    assert result.failure.retry_eligible is missing_from_queue


def test_retrieval_fails_open_without_remote_mtimes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No marker, no ordering: a connection that cannot report mtimes (or a
    ``reuse_existing_files`` resubmission that never re-uploaded the ``.param``)
    trusts whatever is in the directory."""

    _parse_as(monkeypatch, _COMPLETED)
    timestamped = TimestampedFakeSSH()
    plain = FakeSSH()
    for index, ssh in enumerate((timestamped, plain)):
        ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
        ssh.remote_files["/remote/job/seed-out.cell"] = b"relaxed cell"
        local_dir = tmp_path / str(index)
        local_dir.mkdir()

        result = _retrieve(ssh, local_dir, remote_job_id="900", retrieval=_CONTINUATION_FLAGS)

        assert result.outcome is RetrievalOutcome.COMPLETED
        assert (local_dir / "seed-out.cell").read_bytes() == b"relaxed cell"


@pytest.mark.parametrize(
    ("age_seconds", "predates"),
    [
        (0.0, False),
        (REMOTE_STALE_OUTPUT_SLACK_SECONDS, False),
        (REMOTE_STALE_OUTPUT_SLACK_SECONDS + 1.0, True),
        (-60.0, False),
    ],
)
def test_output_is_stale_only_beyond_the_slack(age_seconds: float, predates: bool) -> None:
    ssh = TimestampedFakeSSH()
    ssh.mtimes["/remote/job/seed.castep"] = 1_700_000_000.0 - age_seconds

    assert (
        remote_output_predates_submission(ssh, "/remote/job/seed.castep", 1_700_000_000.0)
        is predates
    )


def test_staleness_probes_fail_open_on_stat_errors() -> None:
    class _BrokenStat(FakeSSH):
        def mtime(self, remote_path: str) -> float | None:
            raise OSError(f"stat failed: {remote_path}")

    ssh = _BrokenStat()

    assert submission_marker_mtime(ssh, "/remote/job/seed.param") is None
    assert remote_output_predates_submission(ssh, "/remote/job/seed.castep", 1.0) is False
    assert remote_output_predates_submission(TimestampedFakeSSH(), "/remote/job/x", None) is False
    assert submission_marker_mtime(FakeSSH(), "/remote/job/seed.param") is None
