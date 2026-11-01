"""The Job Dashboard's geometry-optimisation progress in Status, and the progress dialog."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path


import mufinder.app.main_window as main_window_module
from mufinder.app.geometry_progress_dialog import (
    GeometryProgressDialog,
    progress_heading,
    progress_status_text,
    progress_tooltip,
)
from mufinder.app.main_window import MainWindow
from mufinder.app.theme import plot_series_colours
from mufinder.castep_io.geometry_progress import GeometryProgress, parse_geometry_progress
from mufinder.domain.enums import JobStatus
from tests.support.dashboard import job_row, open_more_menu
from mufinder.services.project_service import ProjectService

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "geometry"
BFGS = FIXTURES / "Si2_bfgs.castep"
LBFGS_CONTINUED = FIXTURES / "Si2_lbfgs.castep"


def test_progress_status_text_is_the_step_of_the_limit() -> None:
    progress = parse_geometry_progress(BFGS)
    running = replace(progress, iterations=progress.iterations[:7], converged=None)

    assert progress_status_text(running) == "step 6 of 30"
    assert progress_status_text(replace(running, max_iterations=None)) == "step 6"
    assert (
        progress_status_text(GeometryProgress(method="BFGS", max_iterations=30, iterations=()))
        == "starting"
    )
    assert progress_status_text(None) == ""


def test_progress_tooltip_lists_each_criterion_against_its_tolerance() -> None:
    progress = parse_geometry_progress(BFGS)
    running = replace(progress, iterations=progress.iterations[:7], converged=None)

    tooltip = progress_tooltip(running)

    assert "BFGS · iteration 6 of at most 30" in tooltip
    for name in ("dE/ion", "|F|max", "|dR|max", "Smax"):
        assert name in tooltip
    assert "1.05e-03" in tooltip and "1.00e-03 A" in tooltip
    assert tooltip.count("✓") == 3 and tooltip.count("✗") == 1
    assert progress_tooltip(None) == ""


def test_progress_heading_names_the_restart_and_outcome() -> None:
    assert progress_heading(parse_geometry_progress(LBFGS_CONTINUED)) == (
        "LBFGS · iteration 3 of at most 30 · restart 1 · converged"
    )


def test_progress_dialog_tabulates_the_latest_step_and_plots_each_criterion(qapp) -> None:
    progress = parse_geometry_progress(BFGS)
    dialog = GeometryProgressDialog("Si_site_1", None)
    try:
        assert dialog._table.rowCount() == 0
        assert "No geometry-optimisation progress" in dialog._heading.text()

        dialog.set_progress(progress)

        assert dialog.windowTitle() == "Geometry Progress - Si_site_1"
        assert [dialog._table.item(row, 0).text() for row in range(4)] == [
            "dE/ion",
            "|F|max",
            "|dR|max",
            "Smax",
        ]
        assert dialog._table.item(1, 3).text() == "Yes"
        axes = dialog._figure.axes
        assert [axis.get_title() for axis in axes] == [
            "dE/ion (eV)",
            "|F|max (eV/A)",
            "|dR|max (A)",
            "Smax (GPa)",
        ]
        value_line, tolerance_line = axes[1].lines
        assert list(value_line.get_xdata()) == list(range(8))
        assert set(tolerance_line.get_ydata()) == {0.02}
        assert axes[0].get_yscale() == "log"
        series = plot_series_colours()
        assert [axis.lines[0].get_color() for axis in axes] == [
            series.energy,
            series.force,
            series.displacement,
            series.stress,
        ]
        assert tolerance_line.get_color() == series.reference
    finally:
        dialog.close()


def test_progress_dialog_refresh_button_asks_for_a_new_read(qapp) -> None:
    dialog = GeometryProgressDialog("job", parse_geometry_progress(BFGS))
    requests: list[bool] = []
    dialog.refresh_requested.connect(lambda: requests.append(True))
    try:
        dialog._refresh_button.click()
        assert requests == [True]
    finally:
        dialog.close()


def _running_window(window: MainWindow, tmp_path: Path, *, stage, target: str = "local"):
    service = ProjectService()
    doc = service.create_project(tmp_path / "progress.mufinder", "Progress")
    job = service.add_job(doc, label="Si_site_1", workflow_stage=stage, calc_type="castep")
    job.status = JobStatus.RUNNING
    job.execution_target = target
    monitor = window._job_progress
    monitor._local_output_paths = lambda job_id: (BFGS,) if job_id == job.id else ()
    monitor._run_in_background = lambda task: task()
    window._document = doc
    window._refresh_jobs_table()
    return window, job


def _status_item(window, job):
    return window._jobs_table.item(job_row(window, job.id), window._JOB_STATUS_COLUMN)


def _more_menu_texts(window, job, monkeypatch) -> list[str]:
    menu = open_more_menu(window, monkeypatch, job_row(window, job.id))
    return [action.text for action in menu.actions]


def test_dashboard_shows_progress_of_a_running_relaxation(
    main_window, tmp_path: Path, monkeypatch
) -> None:
    window, job = _running_window(
        main_window, tmp_path, stage=main_window_module.WorkflowStage.MUON_RELAXATION
    )
    try:
        item = _status_item(window, job)

        assert item.text() == "running · step 7 of 30"
        assert "iteration 7 of at most 30" in item.toolTip()
        assert "Inspect…" in _more_menu_texts(window, job, monkeypatch)
        header = window._jobs_table.horizontalHeaderItem(window._JOB_STATUS_COLUMN)
        assert header.text() == "Status"
        assert "step" in header.toolTip()

        # A finished cluster job is retrieved before it leaves the dashboard.
        job.status = JobStatus.RETRIEVING
        window._refresh_jobs_table()
        assert item.text() == "retrieving"
        assert "Inspect…" not in _more_menu_texts(window, job, monkeypatch)
    finally:
        window._document = None


def test_dashboard_progress_tooltip_joins_the_last_check(main_window, tmp_path: Path) -> None:
    window, job = _running_window(
        main_window, tmp_path, stage=main_window_module.WorkflowStage.MUON_RELAXATION
    )
    try:
        job.last_status_source = "poll"
        job.last_status_message = "a < b"
        window._refresh_jobs_table()

        tooltip = _status_item(window, job).toolTip()

        assert "iteration 7 of at most 30" in tooltip
        assert "<br>Source: poll<br>Message: a &lt; b" in tooltip
    finally:
        window._document = None


def test_dashboard_progress_follows_a_cluster_answer(main_window, tmp_path: Path) -> None:
    window, job = _running_window(
        main_window,
        tmp_path,
        stage=main_window_module.WorkflowStage.PRISTINE_RELAXATION,
        target="scarf",
    )
    try:
        item = _status_item(window, job)
        assert item.text() == "running"

        window._job_progress.accept_remote_progress(
            {job.id: parse_geometry_progress(LBFGS_CONTINUED)}
        )

        assert item.text().startswith("running · step 3")
    finally:
        window._document = None


def test_inspect_opens_one_progress_dialog_per_running_relaxation(
    main_window, tmp_path: Path, monkeypatch
) -> None:
    window, job = _running_window(
        main_window, tmp_path, stage=main_window_module.WorkflowStage.MUON_RELAXATION
    )
    try:
        for _ in range(2):
            open_more_menu(window, monkeypatch, job_row(window, job.id)).action(
                "Inspect…"
            ).triggered.emit()

        dialogs = window.findChildren(GeometryProgressDialog)
        assert len(dialogs) == 1
        assert dialogs[0].isVisible()
        assert dialogs[0].progress == parse_geometry_progress(BFGS)

        progress = parse_geometry_progress(LBFGS_CONTINUED)
        window._on_job_progress_changed(job.id, progress)
        assert dialogs[0].progress == progress

        # Closing the project closes its progress dialogs.
        window._document = None
        window._refresh_jobs_table()
        assert not dialogs[0].isVisible()
    finally:
        window._document = None


def test_inspect_is_not_offered_for_a_running_job_without_geometry_progress(
    main_window, tmp_path: Path, monkeypatch
) -> None:
    window, job = _running_window(
        main_window, tmp_path, stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF
    )
    try:
        assert _status_item(window, job).text() == "running"
        assert "Inspect…" not in _more_menu_texts(window, job, monkeypatch)
        assert not window._dashboard_action_applies("inspect", job)
    finally:
        window._document = None


def test_inspect_on_a_mixed_selection_routes_failed_jobs_to_diagnostics(
    main_window, tmp_path: Path, monkeypatch
) -> None:
    window, running = _running_window(
        main_window, tmp_path, stage=main_window_module.WorkflowStage.MUON_RELAXATION
    )
    failed = ProjectService().add_job(
        window._document,
        label="failed",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep",
    )
    failed.status = JobStatus.FAILED
    diagnosed: list[list] = []
    monkeypatch.setattr(window, "_on_inspect_failed_from_dashboard", diagnosed.append)
    try:
        window._on_inspect_from_dashboard([running.id, failed.id])

        assert diagnosed == [[failed.id]]
        assert len(window.findChildren(GeometryProgressDialog)) == 1
    finally:
        window._document = None
