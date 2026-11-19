"""Reach the Job Dashboard's action cell from a test."""

from __future__ import annotations

from mufinder.app import job_dashboard as job_dashboard_module
from mufinder.domain.enums import EndReason, WorkflowStage
from tests.support.jobs import set_job_state
from tests.support.reset_menu import FakeMenu


def action_button(window, row: int = 0):
    """Return the row's one action button."""

    return window._job_dashboard.table.cellWidget(
        row, window._job_dashboard.ACTION_COLUMN
    ).row_controls[0]


def more_button(window, row: int = 0):
    """Return the row's ``⋯`` button."""

    return window._job_dashboard.table.cellWidget(
        row, window._job_dashboard.ACTION_COLUMN
    ).row_controls[1]


def job_row(window, job_id) -> int:
    """Return the dashboard row showing ``job_id``."""

    for row in range(window._job_dashboard.table.rowCount()):
        if window._job_dashboard.row_job_id(window._job_dashboard.table.item(row, 0)) == job_id:
            return row
    raise AssertionError("job row missing")


def open_more_menu(window, monkeypatch, row: int = 0) -> FakeMenu:
    """Click the row's ``⋯`` button and return the menu it opened."""

    FakeMenu.instances = []
    monkeypatch.setattr(job_dashboard_module, "QMenu", FakeMenu)
    more_button(window, row).click()
    return FakeMenu.instances[-1]


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
