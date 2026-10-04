from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mufinder.domain.enums import FailureReason
from mufinder.services.failure_diagnosis import (
    FAILURE_REASON_LABELS,
    FAILURE_REASON_SUMMARIES,
    MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE,
    RETRIEVAL_FAILURE_PREFIX,
    UNDIAGNOSED_SUMMARY,
    FailureDiagnosis,
    diagnose_failure,
    failure_reason_label,
    failure_reason_summary,
    find_failure_log_files,
    read_log_tail,
)

_SLURM_TIMEOUT = (
    "slurmstepd: error: *** JOB 4242 ON node017 CANCELLED AT 2026-10-04T10:00:00 "
    "DUE TO TIME LIMIT ***"
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("log_text", "reason"),
    [
        (f"mpirun noise\n{_SLURM_TIMEOUT}\n", FailureReason.WALLTIME),
        ("=>> PBS: job killed: walltime 7210 exceeded limit 7200\n", FailureReason.WALLTIME),
        (
            "slurmstepd: error: Detected 1 oom-kill event(s) in StepId=4242.0. "
            "Some of your processes may have been killed by the cgroup out-of-memory handler.\n",
            FailureReason.OUT_OF_MEMORY,
        ),
        ("slurmstepd: error: Exceeded job memory limit\n", FailureReason.OUT_OF_MEMORY),
        (
            "slurmstepd: error: *** JOB 4242 ON node017 CANCELLED AT 2026-10-04T10:00:00 "
            "DUE TO NODE FAILURE ***\n",
            FailureReason.NODE_FAILURE,
        ),
    ],
)
def test_scheduler_log_evidence_is_classified(tmp_path: Path, log_text: str, reason: FailureReason) -> None:
    log = _write(tmp_path / "slurm-4242.out", log_text)

    diagnosis = diagnose_failure("CASTEP did not report successful completion.", [log])

    assert diagnosis.reason == reason
    assert diagnosis.evidence_source == "slurm-4242.out"
    assert diagnosis.evidence in log_text


def test_scheduler_state_in_message_is_classified() -> None:
    diagnosis = diagnose_failure("Scheduler reported the job as OUT_OF_MEMORY.", [])

    assert diagnosis.reason == FailureReason.OUT_OF_MEMORY
    assert diagnosis.evidence_source == "failure message"


def test_scheduler_kill_outranks_castep_error_file(tmp_path: Path) -> None:
    err = _write(tmp_path / "seed.0001.err", "Error in allocate: out of memory\n")
    log = _write(tmp_path / "slurm-1.out", _SLURM_TIMEOUT)

    diagnosis = diagnose_failure(None, [err, log])

    assert diagnosis.reason == FailureReason.WALLTIME


def test_castep_error_file_gives_first_meaningful_line(tmp_path: Path) -> None:
    err = _write(
        tmp_path / "seed.0001.err",
        "\n Current trace stack:\n Error in parameters: unrecognised keyword FOO\n",
    )

    diagnosis = diagnose_failure("CASTEP did not report successful completion.", [err])

    assert diagnosis == FailureDiagnosis(
        reason=FailureReason.CASTEP_ERROR,
        evidence="Error in parameters: unrecognised keyword FOO",
        evidence_source="seed.0001.err",
    )


def test_empty_castep_error_file_is_not_evidence(tmp_path: Path) -> None:
    err = _write(tmp_path / "seed.err", "\n\n")

    assert diagnose_failure("process exited with code 1", [err]).reason is None


@pytest.mark.parametrize(
    "message",
    [
        f"{RETRIEVAL_FAILURE_PREFIX}: no such file",
        MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE,
    ],
)
def test_retrieval_failures_are_output_unavailable(message: str) -> None:
    assert diagnose_failure(message, []).reason == FailureReason.OUTPUT_UNAVAILABLE


def test_retrieval_failure_after_walltime_kill_reports_walltime(tmp_path: Path) -> None:
    log = _write(tmp_path / "slurm-9.out", _SLURM_TIMEOUT)

    diagnosis = diagnose_failure(f"{RETRIEVAL_FAILURE_PREFIX}: seed.castep missing", [log])

    assert diagnosis.reason == FailureReason.WALLTIME


def test_unrecognised_failure_has_no_reason() -> None:
    diagnosis = diagnose_failure("Something odd happened.", [])

    assert diagnosis.reason is None
    assert diagnosis.summary == UNDIAGNOSED_SUMMARY


def test_logs_older_than_submission_are_ignored(tmp_path: Path) -> None:
    log = _write(tmp_path / "slurm-1.out", _SLURM_TIMEOUT)
    an_hour_ago = (datetime.now(tz=UTC) - timedelta(hours=1)).timestamp()
    os.utime(log, (an_hour_ago, an_hour_ago))

    stale = diagnose_failure(None, [log], not_before=datetime.now(tz=UTC))
    fresh = diagnose_failure(None, [log], not_before=datetime.now(tz=UTC) - timedelta(days=1))

    assert stale.reason is None
    assert fresh.reason == FailureReason.WALLTIME


def test_missing_log_file_is_skipped(tmp_path: Path) -> None:
    assert diagnose_failure(None, [tmp_path / "gone.err"]).reason is None


def test_every_reason_has_a_label_and_summary() -> None:
    for reason in FailureReason:
        assert FAILURE_REASON_LABELS[reason]
        assert FAILURE_REASON_SUMMARIES[reason]
        assert failure_reason_label(reason) == FAILURE_REASON_LABELS[reason]
        assert failure_reason_summary(reason) == FAILURE_REASON_SUMMARIES[reason]
    assert failure_reason_label(None) == ""
    assert failure_reason_summary(None) == UNDIAGNOSED_SUMMARY


def test_find_failure_log_files_prefers_job_specific_logs(tmp_path: Path) -> None:
    for name in ("seed.err", "slurm-77.out", "seed_farm002.0003.err", "seed.0001.err", "other.err"):
        _write(tmp_path / name, "x")

    found = find_failure_log_files(tmp_path, ["seed"], "77")

    assert [path.name for path in found] == [
        "seed.err",
        "slurm-77.out",
        "seed.0001.err",
        "seed_farm002.0003.err",
    ]


def test_find_failure_log_files_falls_back_to_any_log(tmp_path: Path) -> None:
    _write(tmp_path / "other.err", "x")
    _write(tmp_path / "slurm-5.out", "x")

    found = find_failure_log_files(tmp_path, ["seed"], None)

    assert [path.name for path in found] == ["other.err", "slurm-5.out"]
    assert find_failure_log_files(tmp_path / "missing", ["seed"], "5") == []


def test_read_log_tail_truncates_large_files(tmp_path: Path) -> None:
    path = _write(tmp_path / "big.out", "first line\n" + "x" * 50 + "\nlast line\n")

    assert read_log_tail(path) == path.read_text(encoding="utf-8")
    tail = read_log_tail(path, max_bytes=20)
    assert tail.startswith("[... truncated] ...\n")
    assert tail.endswith("last line\n")
    assert "first line" not in tail
