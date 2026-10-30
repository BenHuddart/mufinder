"""The failed-job dialog shows why jobs failed and offers what to do next."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from PySide6.QtWidgets import QLabel

from mufinder.app.failed_job_dialog import (
    JOB_LINES_PER_CAUSE,
    FailedJobAction,
    FailedJobDialog,
    failure_title,
)
from mufinder.domain.enums import JobStatus
from mufinder.services.failure_diagnosis import FailedJobReport


def _report(
    label: str = "Muon_relax_0008",
    *,
    headline: str = "CASTEP exited with status 1: SCF cycles did not converge",
    status: JobStatus = JobStatus.FAILED,
    notes: tuple[str, ...] = (),
    log_excerpt: str = "",
    output_directory: Path | None = None,
) -> FailedJobReport:
    return FailedJobReport(
        job_id=uuid4(),
        label=label,
        status=status,
        headline=headline,
        notes=notes,
        log_excerpt=log_excerpt,
        output_directory=output_directory,
    )


def _dialog(reports, *, can_resubmit=True, can_continue=False, opened=None) -> FailedJobDialog:
    return FailedJobDialog(
        reports,
        can_resubmit=can_resubmit,
        can_continue=can_continue,
        open_folder=(opened.append if opened is not None else lambda _path: None),
    )


def _texts(dialog: FailedJobDialog) -> list[str]:
    return [label.text() for label in dialog.findChildren(QLabel)]


def test_one_job_shows_its_headline_and_notes(qapp) -> None:
    dialog = _dialog([_report(notes=("The job's output folder was not found.",))])

    assert dialog.windowTitle() == "Failed job"
    assert _texts(dialog)[:3] == [
        "Muon_relax_0008 failed",
        "CASTEP exited with status 1: SCF cycles did not converge",
        "The job's output folder was not found.",
    ]
    assert dialog.log_toggle is None
    assert dialog.open_folder_button is None
    dialog.close()


@pytest.mark.parametrize(
    ("can_resubmit", "can_continue", "expected"),
    [
        (True, False, ["Resubmit…", "Close"]),
        (True, True, ["Resubmit…", "Continue relaxation…", "Close"]),
        (False, False, ["Close"]),
    ],
)
def test_buttons_follow_what_the_job_allows(qapp, can_resubmit, can_continue, expected) -> None:
    dialog = _dialog([_report()], can_resubmit=can_resubmit, can_continue=can_continue)

    buttons = [
        button
        for button in (dialog.resubmit_button, dialog.continue_button, dialog.close_button)
        if button is not None
    ]
    assert [button.text() for button in buttons] == expected
    assert all(button.accessibleName() for button in buttons)
    assert dialog.close_button.isDefault()
    dialog.close()


def test_open_folder_opens_the_existing_folder_and_keeps_the_dialog(qapp, tmp_path: Path) -> None:
    opened: list[Path] = []
    dialog = _dialog([_report(output_directory=tmp_path)], opened=opened)
    dialog.show()

    assert dialog.open_folder_button is not None
    dialog.open_folder_button.click()

    assert opened == [tmp_path]
    assert dialog.isVisible()
    assert dialog.action() is None
    dialog.close()


@pytest.mark.parametrize(
    ("button", "action"),
    [
        ("resubmit_button", FailedJobAction.RESUBMIT),
        ("continue_button", FailedJobAction.CONTINUE_RELAXATION),
    ],
)
def test_an_action_button_closes_with_that_action(qapp, button, action) -> None:
    dialog = _dialog([_report()], can_continue=True)

    getattr(dialog, button).click()

    assert dialog.result() == FailedJobDialog.DialogCode.Accepted
    assert dialog.action() == action


def test_close_answers_no_action(qapp) -> None:
    dialog = _dialog([_report()])

    dialog.close_button.click()

    assert dialog.result() == FailedJobDialog.DialogCode.Rejected
    assert dialog.action() is None


def test_log_excerpt_is_collapsed_until_asked_for(qapp) -> None:
    dialog = _dialog([_report(log_excerpt="[job.err]\n2: fatal error")])
    dialog.show()

    assert dialog.log_toggle is not None
    assert not dialog.log_view.isVisible()
    assert dialog.log_view.isReadOnly()
    assert dialog.log_view.toPlainText() == "[job.err]\n2: fatal error"

    dialog.log_toggle.click()
    assert dialog.log_view.isVisible()
    dialog.log_toggle.click()
    assert not dialog.log_view.isVisible()
    dialog.close()


def test_several_jobs_are_listed_one_per_line_under_their_cause(qapp) -> None:
    walltime = "The scheduler stopped the job when it reached its walltime limit."
    reports = [
        _report(
            "a",
            headline=walltime,
            status=JobStatus.STOPPED,
            notes=("The job's output folder was not found.",),
        ),
        _report("b", headline="exit 137", log_excerpt="[b.err]\n1: error"),
        _report("c", headline=walltime, status=JobStatus.STOPPED),
    ]
    dialog = _dialog(reports, can_continue=False)

    assert dialog.windowTitle() == "Failed jobs"
    assert _texts(dialog) == ["3 jobs failed or stopped", walltime, "• a\n• c", "exit 137", "• b"]
    assert dialog.open_folder_button is None
    assert dialog.log_view.toPlainText() == (
        "=== a ===\nThe job's output folder was not found.\n\n=== b ===\n[b.err]\n1: error"
    )
    dialog.close()


def test_a_long_cause_group_is_cut_with_a_count(qapp) -> None:
    reports = [_report(f"job{index}", headline="same") for index in range(JOB_LINES_PER_CAUSE + 3)]

    dialog = _dialog(reports)

    lines = _texts(dialog)[2].splitlines()
    assert len(lines) == JOB_LINES_PER_CAUSE + 1
    assert lines[-1] == "…and 3 more jobs"
    dialog.close()


def test_title_names_the_status() -> None:
    assert failure_title([_report(status=JobStatus.STOPPED)]) == "Muon_relax_0008 stopped"
    assert failure_title([_report(), _report()]) == "2 jobs failed"
    assert failure_title([_report(status=JobStatus.STOPPED)] * 2) == "2 jobs stopped"


def test_needs_a_report(qapp) -> None:
    with pytest.raises(ValueError, match="at least one report"):
        _dialog([])
