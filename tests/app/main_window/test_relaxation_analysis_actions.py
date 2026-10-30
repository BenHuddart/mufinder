"""Creating and extending relaxation analysis workspaces from the window.

A new workspace starts ungrouped; relaxations that join a grouped workspace
leave its grouping out of date rather than regrouping it; relaxations from
different calculations are only mixed after a warning; and the displacement
reference defaults to the newest pristine relaxation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from PySide6.QtWidgets import QDialog, QLabel

from mufinder.app import main_window as main_window_module
from mufinder.app import workflow_table_utils
from mufinder.app.main_window import MainWindow
from mufinder.app.relaxation_analysis_widget import grouping_out_of_date_note
from mufinder.domain.charge_states import POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import Job
from mufinder.science.relaxation_analysis import cluster_connected_components
from mufinder.services.analysis_grouping import (
    DEFAULT_GROUPING_SETTINGS,
    SiteGrouping,
    grouped_clustering_state,
)
from mufinder.services.project_service import ProjectService
from mufinder.services.source_candidates import (
    CompletedPristineJob,
    CompletedRelaxationAnalysisSource,
)
from tests.support.structures import host_structure


def _source(
    label: str,
    position: tuple[float, float, float],
    energy: float,
    *,
    calculation: tuple[object, ...] = ("Si2H", None, None),
) -> CompletedRelaxationAnalysisSource:
    return CompletedRelaxationAnalysisSource(
        job_id=uuid4(),
        job_label=label,
        charge_state=POSITIVE_MUON_CHARGE_STATE,
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=position,
        total_energy_ev=energy,
        calculation=calculation,
    )


@pytest.fixture
def window(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MainWindow:
    service = ProjectService()
    document = service.create_project(tmp_path / "analysis.mufinder", "Analysis Doc")
    host = host_structure()
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    service.save_project(document)
    window = MainWindow()
    assert window.open_project(document.project_root)
    monkeypatch.setattr(
        main_window_module.QInputDialog, "getText", lambda *args, **kwargs: ("Sites", True)
    )
    return window


def _create(window: MainWindow, sources: list[CompletedRelaxationAnalysisSource]) -> None:
    window._completed_muon_relaxation_sources = lambda: list(sources)  # type: ignore[method-assign]
    window._on_create_relaxation_analysis_workspace_requested(
        [source.job_label for source in sources]
    )


def test_a_new_workspace_starts_ungrouped(window: MainWindow) -> None:
    sources = [
        _source("A", (0.5, 0.25, 0.0), -10.0),
        _source("C", (0.25, 0.25, 0.25), -9.5),
    ]

    _create(window, sources)

    (workspace,) = window._relaxation_analysis_workspaces()
    assert workspace["clustering"]["result"] is None
    created = [
        record
        for record in window._document.provenance
        if record.action == "create_analysis_workspace"
    ]
    assert "distinct_site_count" not in created[-1].parameters


def test_relaxations_from_different_calculations_need_a_confirmation(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def answer(_parent, title, *_args, **kwargs) -> str:
        asked.append(title)
        assert kwargs["default"] == "cancel"
        return "cancel"

    monkeypatch.setattr(workflow_table_utils, "confirm_choice", answer)
    sources = [
        _source("A", (0.5, 0.25, 0.0), -10.0, calculation=("Si16H", None, None)),
        _source("B", (0.25, 0.25, 0.25), -20.0, calculation=("Si54H", None, None)),
    ]

    _create(window, sources)

    assert asked == ["Different calculations"]
    assert window._relaxation_analysis_workspaces() == []

    monkeypatch.setattr(workflow_table_utils, "confirm_choice", lambda *_a, **_k: "add")
    _create(window, sources)
    assert len(window._relaxation_analysis_workspaces()) == 1


def _note(window: MainWindow) -> QLabel:
    page = window._relaxation_analysis_view._workspace_tabs.currentWidget()
    return grouping_out_of_date_note(page)


def test_relaxations_that_join_a_grouped_workspace_leave_its_grouping_out_of_date(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = [_source("A", (0.5, 0.25, 0.0), -10.0), _source("C", (0.25, 0.25, 0.25), -9.0)]
    _create(window, first)
    workspace_id = str(window._relaxation_analysis_workspaces()[0]["id"])
    window._set_active_route(window._relaxation_analysis_workspace_route(workspace_id))
    view = window._relaxation_analysis_view
    host = window._project_service.active_host_structure(window._document)
    sites = [source.to_relaxed_site(host) for source in first]
    result = cluster_connected_components(sites, host, tolerance_ang=0.25)
    view._apply_clustering(
        workspace_id,
        grouped_clustering_state(
            DEFAULT_GROUPING_SETTINGS,
            SiteGrouping("connected", result.cluster_labels, result.clusters, ()),
            host,
        ),
    )
    assert _note(window).isHidden()
    joining = [_source("B", (0.0, 0.5, 0.25), -9.5)]
    every = [*first, *joining]
    window._completed_muon_relaxation_sources = lambda: list(every)  # type: ignore[method-assign]
    window._refresh_relaxation_analysis_context(force=True)
    records_before = len(window._document.provenance)

    class _Picked:
        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return QDialog.DialogCode.Accepted

        def selected_job_ids(self) -> list[UUID]:
            return [source.job_id for source in joining]

    monkeypatch.setattr(main_window_module, "RelaxationAnalysisJobDialog", _Picked)

    window._on_relaxation_analysis_add_sites_requested(workspace_id)

    (workspace,) = window._relaxation_analysis_workspaces()
    assert len(workspace["job_ids"]) == 3
    # Nothing regrouped: A and C keep their groups and B, between them in energy, has none.
    assert workspace["clustering"]["result"]["cluster_labels"] == [1, 0, 2]
    assert len(window._document.provenance) == records_before
    note = _note(window)
    assert not note.isHidden()
    assert note.text() == (
        "The grouping is out of date: it leaves out 1 relaxation added after it was "
        "applied. Choose 'Clustering…' and 'Apply Clustering' to group every relaxation."
    )

    # The out-of-date grouping survives a save and reload.
    window._project_service.save_project(window._document)
    reopened = MainWindow()
    assert reopened.open_project(window._document.project_root)
    reopened._completed_muon_relaxation_sources = lambda: list(every)  # type: ignore[method-assign]
    reopened._set_active_route(reopened._relaxation_analysis_workspace_route(workspace_id))
    assert not _note(reopened).isHidden()

    # Applying a grouping again includes every relaxation and clears the note.
    every_site = [source.to_relaxed_site(host) for source in every]
    regrouped = cluster_connected_components(every_site, host, tolerance_ang=0.25)
    reopened._relaxation_analysis_view._apply_clustering(
        workspace_id,
        grouped_clustering_state(
            DEFAULT_GROUPING_SETTINGS,
            SiteGrouping("connected", regrouped.cluster_labels, regrouped.clusters, ()),
            host,
        ),
    )
    assert _note(reopened).isHidden()


def test_the_displacement_reference_defaults_to_the_newest_pristine_relaxation(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = window._document
    host_id = document.project.host_structure_id

    def pristine(label: str, completed_at: datetime) -> CompletedPristineJob:
        job = Job(label=label, status=JobStatus.COMPLETED, completed_at=completed_at)
        document.jobs[job.id] = job
        return CompletedPristineJob(
            job_id=job.id,
            job_label=label,
            structure_id=uuid4(),
            supercell_label="Unit cell",
            status="completed",
            total_energy=None,
            lattice_a=None,
            lattice_b=None,
            lattice_c=None,
            parameter_values={},
        )

    older = pristine("older", datetime(2026, 1, 1, tzinfo=UTC))
    newer = pristine("newer", datetime(2026, 3, 1, tzinfo=UTC))
    free_cell = pristine("free", datetime(2026, 5, 1, tzinfo=UTC))
    monkeypatch.setattr(
        window._pristine_relaxation_controller,
        "completed_jobs",
        lambda: [older, newer, free_cell],
    )
    options = [
        (host_id, "Host structure", False),
        (older.structure_id, "Pristine relaxed - older", False),
        (newer.structure_id, "Pristine relaxed - newer", False),
        (free_cell.structure_id, "Pristine relaxed - free (free cell; unavailable)", True),
    ]

    assert window._default_displacement_reference_id(options) == newer.structure_id
    assert window._default_displacement_reference_id(options[:1]) is None
