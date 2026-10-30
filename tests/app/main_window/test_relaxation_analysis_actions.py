"""Creating and extending relaxation analysis workspaces from the window.

A new workspace is grouped as it is created, relaxations that join it regroup
it, relaxations from different calculations are only mixed after a warning,
and the displacement reference defaults to the newest pristine relaxation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from PySide6.QtWidgets import QDialog

from mufinder.app import main_window as main_window_module
from mufinder.app import workflow_table_utils
from mufinder.app.main_window import MainWindow
from mufinder.domain.charge_states import POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import Job
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
    calculation: tuple[object, ...] = ("same",),
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


def test_a_new_workspace_is_grouped_without_opening_the_dialog(window: MainWindow) -> None:
    sources = [
        _source("A", (0.5, 0.25, 0.0), -10.0),
        _source("B", (0.0, 0.5, 0.25), -10.0 + 1e-6),
        _source("C", (0.25, 0.25, 0.25), -9.5),
    ]

    _create(window, sources)

    (workspace,) = window._relaxation_analysis_workspaces()
    clustering = workspace["clustering"]
    assert clustering["method"] == "connected"
    assert clustering["connected_tolerance_ang"] == 0.25
    # A and B are images of one site under the bcc symmetry; C is another site.
    assert clustering["result"]["cluster_labels"] == [1, 1, 2]
    assert clustering["grouping"]["method"] == "connected"
    assert clustering["grouping"]["space_group_symbol"] == "Im-3m"
    created = [
        record
        for record in window._document.provenance
        if record.action == "create_analysis_workspace"
    ]
    assert created[-1].parameters["distinct_site_count"] == 2


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
        _source("A", (0.5, 0.25, 0.0), -10.0, calculation=("2x2x2",)),
        _source("B", (0.25, 0.25, 0.25), -20.0, calculation=("3x3x3",)),
    ]

    _create(window, sources)

    assert asked == ["Different calculations"]
    assert window._relaxation_analysis_workspaces() == []

    monkeypatch.setattr(workflow_table_utils, "confirm_choice", lambda *_a, **_k: "add")
    _create(window, sources)
    assert len(window._relaxation_analysis_workspaces()) == 1


def test_relaxations_that_join_a_workspace_regroup_it(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _source("A", (0.5, 0.25, 0.0), -10.0)
    _create(window, [first])
    workspace_id = str(window._relaxation_analysis_workspaces()[0]["id"])
    window._set_active_route(window._relaxation_analysis_workspace_route(workspace_id))
    joining = [_source("B", (0.0, 0.5, 0.25), -10.0), _source("C", (0.25, 0.25, 0.25), -9.0)]
    every = [first, *joining]
    window._completed_muon_relaxation_sources = lambda: list(every)  # type: ignore[method-assign]
    window._refresh_relaxation_analysis_context(force=True)

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
    assert workspace["clustering"]["result"]["cluster_labels"] == [1, 1, 2]


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
        window, "_completed_pristine_relaxation_jobs", lambda: [older, newer, free_cell]
    )
    options = [
        (host_id, "Host structure", False),
        (older.structure_id, "Pristine relaxed - older", False),
        (newer.structure_id, "Pristine relaxed - newer", False),
        (free_cell.structure_id, "Pristine relaxed - free (free cell; unavailable)", True),
    ]

    assert window._default_displacement_reference_id(options) == newer.structure_id
    assert window._default_displacement_reference_id(options[:1]) is None
