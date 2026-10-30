from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mufinder.domain.enums import FailureReason, JobStatus, WorkflowStage
from mufinder.domain.models import Job
from mufinder.services.failure_diagnosis import (
    EXCERPT_TAIL_BYTES,
    FAILURE_REASON_LABELS,
    FAILURE_REASON_SUMMARIES,
    MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE,
    RETRIEVAL_FAILURE_PREFIX,
    UNDIAGNOSED_SUMMARY,
    FailureDiagnosis,
    diagnose_failure,
    failed_job_report,
    failure_headline,
    failure_log_files,
    failure_reason_label,
    failure_reason_summary,
    find_failure_log_files,
    group_reports_by_headline,
    read_error_contexts,
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
def test_scheduler_log_evidence_is_classified(
    tmp_path: Path, log_text: str, reason: FailureReason
) -> None:
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
    assert tail.startswith("[… truncated] …\n")
    assert tail.endswith("last line\n")
    assert "first line" not in tail


def _job(label: str = "job_failed", *, message: str | None = None) -> Job:
    return Job(
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        status=JobStatus.FAILED,
        error_message=message,
    )


def test_headline_prefers_the_cause_then_the_recorded_message() -> None:
    walltime = FailureDiagnosis(reason=FailureReason.WALLTIME)
    message = "CASTEP exited with status 1: SCF cycles did not converge"

    assert failure_headline(walltime, message) == FAILURE_REASON_SUMMARIES[FailureReason.WALLTIME]
    assert failure_headline(FailureDiagnosis(), f"  {message}\n") == message
    assert failure_headline(FailureDiagnosis(), "  ") == UNDIAGNOSED_SUMMARY
    assert failure_headline(FailureDiagnosis(), None) == "No cause found in the logs."


def test_report_without_an_output_folder_says_so(tmp_path: Path) -> None:
    job = _job(message="CASTEP exited with status 1: SCF cycles did not converge")

    report = failed_job_report(job, FailureDiagnosis(), tmp_path / "gone", job.label)

    assert report.headline == "CASTEP exited with status 1: SCF cycles did not converge"
    assert report.notes == ("The job's output folder was not found, so there are no logs to show.",)
    assert report.log_excerpt == ""
    assert report.output_directory is None
    assert (report.job_id, report.label, report.status) == (job.id, job.label, JobStatus.FAILED)


def test_report_lists_the_message_the_evidence_and_the_log_excerpt(tmp_path: Path) -> None:
    job = _job(message="process failed")
    _write(tmp_path / "job_failed.err", "line 1\nfatal error\nline 3\n")
    diagnosis = FailureDiagnosis(
        reason=FailureReason.CASTEP_ERROR, evidence="line 1", evidence_source="job_failed.err"
    )

    report = failed_job_report(job, diagnosis, tmp_path, job.label)

    assert report.headline == "CASTEP stopped with an error."
    assert report.notes == ("Recorded message: process failed", "From job_failed.err: line 1")
    assert report.log_excerpt.startswith("[job_failed.err]\n")
    assert "2: fatal error" in report.log_excerpt
    assert report.output_directory == tmp_path


def test_report_does_not_repeat_the_headline_or_the_message(tmp_path: Path) -> None:
    message = f"{RETRIEVAL_FAILURE_PREFIX}: timeout"
    job = _job(message=message)
    diagnosis = FailureDiagnosis(
        reason=FailureReason.OUTPUT_UNAVAILABLE, evidence=message, evidence_source="failure message"
    )

    report = failed_job_report(job, diagnosis, tmp_path, job.label)

    assert report.notes == (f"Recorded message: {message}",)
    assert report.log_excerpt == "No .err or SLURM output files were found in the output folder."


def test_reports_group_by_headline_in_first_seen_order(tmp_path: Path) -> None:
    walltime = FailureDiagnosis(reason=FailureReason.WALLTIME)
    reports = [
        failed_job_report(_job("a"), walltime, None, "a"),
        failed_job_report(_job("b", message="exit 137"), FailureDiagnosis(), None, "b"),
        failed_job_report(_job("c"), walltime, None, "c"),
    ]

    groups = group_reports_by_headline(reports)

    assert [(headline, [r.label for r in members]) for headline, members in groups] == [
        (FAILURE_REASON_SUMMARIES[FailureReason.WALLTIME], ["a", "c"]),
        ("exit 137", ["b"]),
    ]


def test_failure_log_files_prefers_numbered_error_files(tmp_path: Path) -> None:
    job = _job()
    _write(tmp_path / "job_failed.0001.err", "rank 1 error\n")
    _write(tmp_path / "job_failed.0002.err", "rank 2 error\n")
    _write(tmp_path / "unrelated.err", "other error\n")

    assert [path.name for path in failure_log_files(job, tmp_path, job.label)] == [
        "job_failed.0001.err",
        "job_failed.0002.err",
    ]
    assert failure_log_files(job, tmp_path / "does_not_exist", job.label) == []


def test_failure_log_files_finds_the_scheduler_output(tmp_path: Path) -> None:
    job = _job()
    job.remote_job_id = "771300"
    _write(tmp_path / "slurm-771300.out", "running job\nslurm error: out of memory\n")

    (path,) = failure_log_files(job, tmp_path, job.label)

    assert path.name == "slurm-771300.out"
    assert "2: slurm error: out of memory" in read_error_contexts(path)


def test_read_error_contexts_handles_empty_missing_and_unmarked_files(tmp_path: Path) -> None:
    small = _write(tmp_path / "small.err", "first line\nerror here\nthird line\n")
    empty = _write(tmp_path / "empty.err", "")
    quiet = _write(tmp_path / "quiet.out", "".join(f"line {n}\n" for n in range(1, 31)))

    assert read_error_contexts(empty) == "[File is empty]"
    assert read_error_contexts(tmp_path / "missing.err").startswith("Could not read file")
    assert read_error_contexts(small) == "1: first line\n2: error here\n3: third line"
    assert read_error_contexts(quiet).splitlines()[0] == "11: line 11"


def test_the_excerpt_of_a_large_file_reads_only_its_tail(tmp_path: Path) -> None:
    path = tmp_path / "huge.out"
    first_line = "FIRST-LINE-SENTINEL " + ("x" * 200) + "\n"
    filler_line = "y" * 500 + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(first_line)
        for _ in range(EXCERPT_TAIL_BYTES // len(filler_line) + 10):
            handle.write(filler_line)
        handle.write("LAST-LINE-SENTINEL\n")
    assert path.stat().st_size > EXCERPT_TAIL_BYTES

    excerpt = read_error_contexts(path)
    assert "FIRST-LINE-SENTINEL" not in excerpt
    assert "LAST-LINE-SENTINEL" in excerpt


def test_evidence_taken_from_the_recorded_message_is_not_repeated(tmp_path: Path) -> None:
    message = f"slurmstepd: noise\n{_SLURM_TIMEOUT}"
    job = _job(message=message)
    diagnosis = diagnose_failure(message, [])

    report = failed_job_report(job, diagnosis, tmp_path, job.label)

    assert diagnosis.evidence_source == "failure message"
    assert report.notes == (f"Recorded message: {message}",)
