"""Dashboard "Last checked" column + stale marker rendering (rec #2/#3).

The monitor metadata (``last_checked_at`` / ``last_status_source`` /
``last_status_message``) is stamped at runtime by other streams; this column
must render gracefully when it is all-None ("—") and show a relative age plus a
staleness marker when it is present.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from PySide6.QtCore import Qt

import mufinder.app.main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import JobStatus
from mufinder.services.project_service import ProjectService


def _add_job(service, doc, *, label: str):
    return service.add_job(
        doc,
        label=label,
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )


def _last_checked_item(window, job_id):
    for row in range(window._jobs_table.rowCount()):
        item = window._jobs_table.item(row, window._JOB_LABEL_COLUMN)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == str(job_id):
            return window._jobs_table.item(row, window._JOB_LAST_CHECKED_COLUMN)
    return None


def test_format_relative_age_branches(qapp) -> None:
    now = datetime.now(UTC)
    assert MainWindow._format_relative_age(None, now=now) == "—"
    assert MainWindow._format_relative_age(now, now=now) == "just now"
    assert MainWindow._format_relative_age(now - timedelta(minutes=3), now=now) == "3 min ago"
    assert MainWindow._format_relative_age(now - timedelta(hours=5), now=now) == "5 hr ago"
    assert MainWindow._format_relative_age(now - timedelta(days=1), now=now) == "1 day ago"
    assert MainWindow._format_relative_age(now - timedelta(days=3), now=now) == "3 days ago"
    # Naive stamps are treated as UTC (older-build compatibility).
    naive = (now - timedelta(minutes=2)).replace(tzinfo=None)
    assert MainWindow._format_relative_age(naive, now=now) == "2 min ago"


def test_last_checked_column_header_present(qapp) -> None:
    window = MainWindow()
    headers = [
        window._jobs_table.horizontalHeaderItem(i).text()
        for i in range(window._jobs_table.columnCount())
    ]
    assert "Last Checked" in headers
    window.close()


def test_last_checked_renders_dash_without_metadata(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "nometa.mufinder", "NoMeta")
    job = _add_job(service, doc, label="failed")
    job.status = JobStatus.FAILED  # terminal, no monitor metadata
    window._document = doc
    window._refresh_jobs_table()

    item = _last_checked_item(window, job.id)
    assert item is not None
    assert item.text() == "—"
    # No tooltip when there is nothing to show.
    assert item.toolTip() == ""
    window.close()


def test_last_checked_renders_relative_time_and_tooltip(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "meta.mufinder", "Meta")
    job = _add_job(service, doc, label="running")
    job.status = JobStatus.RUNNING
    job.execution_target = "scarf"
    job.last_checked_at = datetime.now(UTC) - timedelta(minutes=2)
    job.last_status_source = "poll"
    job.last_status_message = "queued on scarf"
    window._document = doc
    # Connected to the job's cluster so the row is NOT stale.
    window._cluster_runner._connected = True
    window._cluster_runner._connected_cluster_name = "scarf"
    window._refresh_jobs_table()

    item = _last_checked_item(window, job.id)
    assert item is not None
    assert "min ago" in item.text()
    assert "stale" not in item.text()
    tooltip = item.toolTip()
    assert "poll" in tooltip
    assert "queued on scarf" in tooltip
    window.close()


def test_last_checked_marks_disconnected_active_cluster_job_stale(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "stale.mufinder", "Stale")
    job = _add_job(service, doc, label="queued")
    job.status = JobStatus.QUEUED
    job.execution_target = "scarf"
    job.last_checked_at = datetime.now(UTC) - timedelta(minutes=1)
    window._document = doc
    # Cluster runner is NOT connected -> active cluster job is stale.
    window._cluster_runner._connected = False
    window._cluster_runner._connected_cluster_name = None
    window._refresh_jobs_table()

    item = _last_checked_item(window, job.id)
    assert item is not None
    assert item.text().endswith("— stale")
    assert "cluster disconnected" in item.toolTip()
    assert item.data(Qt.ItemDataRole.ForegroundRole) is not None

    # The row is updated in place: once the job is no longer stale its cell
    # drops the marker, the warning colour and the stale tooltip.
    job.status = JobStatus.FAILED
    window._refresh_jobs_table()

    assert _last_checked_item(window, job.id) is item
    assert item.text() == "1 min ago"
    assert item.data(Qt.ItemDataRole.ForegroundRole) is None
    assert item.toolTip() == ""
    window.close()
