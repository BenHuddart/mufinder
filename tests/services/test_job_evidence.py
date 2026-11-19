"""Evidence gathered about a local job: the runner's process and the output folder."""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mufinder.services import job_evidence
from mufinder.services.job_evidence import local_job_evidence, local_output_evidence
from mufinder.services.job_reconcile import OutputState, ProcessState
from tests.support.jobs import make_job

COMPLETE_OUTPUT = Path(__file__).parents[1] / "fixtures" / "uep" / "Si2.castep"
NOW = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)


def _write_complete(folder: Path, name: str = "job.castep") -> Path:
    return Path(shutil.copy(COMPLETE_OUTPUT, folder / name))


def _write_incomplete(folder: Path, name: str = "job.castep") -> Path:
    text = COMPLETE_OUTPUT.read_text(encoding="utf-8", errors="replace")
    path = folder / name
    path.write_text(text[: text.index("Total time")], encoding="utf-8")
    return path


def test_complete_output_is_parsed(tmp_path: Path) -> None:
    _write_complete(tmp_path)

    state, detail, parsed = local_output_evidence(tmp_path, "job", not_before=None)

    assert (state, detail) == (OutputState.COMPLETE, "")
    assert parsed is not None and parsed.completed


def test_output_without_a_completion_summary_is_incomplete(tmp_path: Path) -> None:
    _write_incomplete(tmp_path)

    state, _detail, parsed = local_output_evidence(tmp_path, "job", not_before=None)

    assert state == OutputState.INCOMPLETE
    assert parsed is not None and not parsed.completed


@pytest.mark.parametrize("folder", [None, "missing", "empty"])
def test_no_output_file_is_absent(tmp_path: Path, folder: str | None) -> None:
    (tmp_path / "empty").mkdir()
    output_dir = None if folder is None else tmp_path / folder

    assert local_output_evidence(output_dir, "job", not_before=None) == (
        OutputState.ABSENT,
        "",
        None,
    )


def test_output_written_before_the_submission_is_absent(tmp_path: Path) -> None:
    path = _write_complete(tmp_path)
    submitted_at = datetime.now(UTC)
    old = (submitted_at - timedelta(hours=1)).timestamp()
    os.utime(path, (old, old))

    state, _detail, parsed = local_output_evidence(tmp_path, "job", not_before=submitted_at)

    assert (state, parsed) == (OutputState.ABSENT, None)


def test_output_written_after_a_naive_submission_stamp_counts(tmp_path: Path) -> None:
    _write_complete(tmp_path)
    submitted_at = (datetime.now(UTC) - timedelta(minutes=1)).replace(tzinfo=None)

    state, _detail, _parsed = local_output_evidence(tmp_path, "job", not_before=submitted_at)

    assert state == OutputState.COMPLETE


def test_output_that_cannot_be_parsed_is_unreadable(tmp_path: Path, monkeypatch) -> None:
    _write_complete(tmp_path)

    def fail(_path: Path):
        raise ValueError("bad header")

    monkeypatch.setattr(job_evidence, "parse_castep_output", fail)

    assert local_output_evidence(tmp_path, "job", not_before=None) == (
        OutputState.UNREADABLE,
        "bad header",
        None,
    )


def test_a_farmed_run_is_read_from_its_first_farm(tmp_path: Path) -> None:
    _write_complete(tmp_path, "job.castep")
    _write_incomplete(tmp_path, "job_farm001.castep")

    state, _detail, _parsed = local_output_evidence(tmp_path, "job", not_before=None)

    assert state == OutputState.INCOMPLETE


def test_a_job_the_runner_holds_is_not_read_from_disk(tmp_path: Path) -> None:
    _write_complete(tmp_path)
    running = make_job(status="running")
    queued = make_job(status="queued")
    held = {"pending_ids": {queued.id}, "active_ids": {running.id}}

    running_evidence = local_job_evidence(
        running, output_dir=tmp_path, seed_name="job", now=NOW, **held
    )
    queued_evidence = local_job_evidence(
        queued, output_dir=tmp_path, seed_name="job", now=NOW, **held
    )

    assert (running_evidence.process, running_evidence.output) == (
        ProcessState.RUNNING,
        OutputState.NOT_LOOKED,
    )
    assert (queued_evidence.process, queued_evidence.output) == (
        ProcessState.PENDING,
        OutputState.NOT_LOOKED,
    )


def test_a_job_with_no_process_carries_its_output(tmp_path: Path) -> None:
    _write_complete(tmp_path)
    job = make_job(status="running")

    evidence = local_job_evidence(
        job, pending_ids=(), active_ids=(), output_dir=tmp_path, seed_name="job", now=NOW
    )

    assert evidence.process == ProcessState.NONE
    assert evidence.output == OutputState.COMPLETE
    assert evidence.parsed is not None
    assert evidence.checked_at == NOW
