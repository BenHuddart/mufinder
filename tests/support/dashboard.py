"""Reach the Job Dashboard's action slots from a test."""

from __future__ import annotations

from mufinder.app import job_dashboard as job_dashboard_module
from mufinder.domain.enums import EndReason, WorkflowStage
from tests.support.jobs import set_job_state
from tests.support.row_menu import FakeMenu

#: The row's slots, by what they hold: run (Cancel or Resubmit…), look (Inspect…
#: or Check now) and dismiss (the ✕ button).
RUN_SLOT, LOOK_SLOT, DISMISS_SLOT = 0, 1, 2


def slot_button(window, row: int = 0, slot: int = RUN_SLOT):
    """Return the button in ``slot`` of the row's action cell, shown or not."""

    return window._job_dashboard.table.cellWidget(
        row, window._job_dashboard.ACTION_COLUMN
    ).row_controls[slot]


def run_button(window, row: int = 0):
    """Return the row's first slot: ``Cancel`` or ``Resubmit…``."""

    return slot_button(window, row, RUN_SLOT)


def look_button(window, row: int = 0):
    """Return the row's second slot: ``Inspect…`` or ``Check now``."""

    return slot_button(window, row, LOOK_SLOT)


def dismiss_button(window, row: int = 0):
    """Return the row's ``✕`` (Dismiss) button."""

    return slot_button(window, row, DISMISS_SLOT)


def row_actions(window, row: int = 0) -> tuple[str, str, str]:
    """Return what each of the row's three slots shows: its label, ``✕``, or ``""``."""

    cell = window._job_dashboard.table.cellWidget(row, window._job_dashboard.ACTION_COLUMN)
    shown = []
    for slot, button in enumerate(cell.row_controls):
        if button.isHidden():
            shown.append("")
        else:
            shown.append("✕" if slot == DISMISS_SLOT else button.text())
    return tuple(shown)


def job_row(window, job_id) -> int:
    """Return the dashboard row showing ``job_id``."""

    for row in range(window._job_dashboard.table.rowCount()):
        if window._job_dashboard.row_job_id(window._job_dashboard.table.item(row, 0)) == job_id:
            return row
    raise AssertionError("job row missing")


def open_context_menu(window, monkeypatch, row: int = 0) -> FakeMenu:
    """Right-click the row and return the menu it opened."""

    FakeMenu.instances = []
    monkeypatch.setattr(job_dashboard_module, "QMenu", FakeMenu)
    rect = window._job_dashboard.table.visualRect(window._job_dashboard.table.model().index(row, 0))
    window._job_dashboard.table.customContextMenuRequested.emit(rect.center())
    return FakeMenu.instances[-1]


def add_dashboard_job(service, doc, label, status):
    """Add a cutoff-convergence job in ``status`` to ``doc``."""

    job = service.add_job(
        doc,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    set_job_state(job, status)
    return job


def orphaned_cluster_job(service, doc, label="o0"):
    """Add a stopped cluster job whose output may still be in its remote folder."""

    job = add_dashboard_job(service, doc, label, "stopped")
    job.end_reason = EndReason.INTERRUPTED
    job.execution_target = "SCARF"
    job.remote_job_id = "4127731"
    job.remote_directory = f"/remote/{label}"
    return job


def give_jobs_rows(window, monkeypatch) -> None:
    """Treat every ended job as one whose workflow row still holds it.

    ``Resubmit…`` runs a job's row, so the dashboard offers it only to such a
    job. Dashboard tests that add bare jobs, with no workflow definition, use
    this to stand for a project whose rows hold them.
    """

    from mufinder.app.workflow_submission import RESUBMITTABLE_JOB_STATES

    monkeypatch.setattr(
        window._jobs, "can_resubmit", lambda job: job.status in RESUBMITTABLE_JOB_STATES
    )
