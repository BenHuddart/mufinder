"""The Job Dashboard's Status tooltip and stale icon.

The monitor metadata (``last_checked_at`` / ``last_status_source`` /
``last_status_message``) is stamped at runtime by other streams; the tooltip must
read cleanly when it is all-None and carry the age, source and any stale reason
when it is present.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

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


def _status_item(window):
    return window._jobs_table.item(0, window._JOB_STATUS_COLUMN)


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


def test_status_tooltip_is_empty_without_metadata(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "nometa.mufinder", "NoMeta")
    job = _add_job(service, doc, label="cancelled")
    job.status = JobStatus.CANCELLED
    window._document = doc
    window._refresh_jobs_table()

    assert _status_item(window).toolTip() == ""
    assert _status_item(window).icon().isNull()
    window.close()


def test_status_tooltip_carries_the_last_check(qapp, tmp_path: Path) -> None:
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

    item = _status_item(window)
    assert item.text() == "running"
    assert item.toolTip() == "Last checked: 2 min ago\nSource: poll\nMessage: queued on scarf"
    assert item.icon().isNull()
    window.close()


def test_disconnected_active_cluster_job_gets_the_stale_icon(qapp, tmp_path: Path) -> None:
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

    item = _status_item(window)
    # The state reads as it always does: no amber text, no "stale" in the cell.
    assert item.text() == "queued"
    assert not item.icon().isNull()
    assert "Stale: cluster disconnected" in item.toolTip()

    # The row is updated in place: once the job is no longer stale its cell
    # drops the icon and the stale line.
    job.status = JobStatus.FAILED
    window._refresh_jobs_table()

    assert _status_item(window) is item
    assert item.icon().isNull()
    assert "Stale" not in item.toolTip()
    window.close()
