"""Tests for the comparison-table controller: generate, rename and delete."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from mufinder.app.workflow_controllers.comparison_tables import (
    ComparisonTablesController,
    ComparisonTablesDeps,
    comparison_route,
    comparison_table_id_from_route,
)
from mufinder.domain.models import ComparisonTable, Structure
from mufinder.services.project_service import ProjectService
from mufinder.services.source_candidates import CompletedPristineJob
from tests.support.controllers import make_context


class _FakePage:
    def __init__(self) -> None:
        self.tables: list[ComparisonTable] = []
        self.labels: list[tuple[UUID, str]] = []

    def set_tables(self, tables) -> None:
        self.tables = list(tables)

    def set_table_label(self, table_id: UUID, label: str) -> None:
        self.labels.append((table_id, label))


def _cubic(label: str, a: float) -> Structure:
    return Structure(
        label=label,
        lattice_vectors=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def _completed(structure_id: UUID, label: str = "alpha") -> CompletedPristineJob:
    return CompletedPristineJob(
        job_id=uuid4(),
        job_label=label,
        structure_id=structure_id,
        supercell_label="Unit cell",
        status="completed",
        total_energy=None,
        lattice_a=None,
        lattice_b=None,
        lattice_c=None,
        parameter_values={},
    )


@dataclass
class _Harness:
    """A controller on a real project, recording everything it asks the window for."""

    service: ProjectService
    document: object
    relaxed: Structure | None
    page: _FakePage = field(default_factory=_FakePage)
    jobs: list[CompletedPristineJob] = field(default_factory=list)
    choice: object = None
    infos: list[tuple[str, str]] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    current_route: object = "pristine_relaxation"
    closing: bool = False
    offered: list[tuple[list[CompletedPristineJob], str]] = field(default_factory=list)

    def controller(self) -> ComparisonTablesController:
        def choose(jobs, default_label):
            self.offered.append((jobs, default_label))
            return self.choice

        return ComparisonTablesController(
            make_context(
                document=self.document,
                project_service=self.service,
                infos=self.infos,
                statuses=self.statuses,
            ),
            ComparisonTablesDeps(
                completed_jobs=lambda: self.jobs,
                choose_jobs=choose,
                show_warning=lambda title, text: self.warnings.append((title, text)),
                refresh_project_tree=lambda: self.calls.append("tree"),
                refresh_journal_view=lambda: self.calls.append("journal"),
                active_route=lambda: self.current_route,
                open_route=self.routes.append,
                shutting_down=lambda: self.closing,
            ),
            self.page,
        )


@pytest.fixture
def harness(tmp_path: Path) -> _Harness:
    service = ProjectService()
    document = service.create_project(tmp_path / "tables.mufinder", "Tables")
    host = _cubic("Host", 5.0)
    relaxed = _cubic("Relaxed", 5.1)
    service.add_structure(document, host)
    service.add_structure(document, relaxed)
    document.project.host_structure_id = host.id
    return _Harness(service=service, document=document, relaxed=relaxed)


def _saved_table(harness: _Harness, label: str = "Comparison 1") -> ComparisonTable:
    table = ComparisonTable(label=label)
    harness.service.add_comparison_table(harness.document, table)
    return table


def test_route_round_trips_and_other_routes_name_no_table() -> None:
    table_id = uuid4()

    assert comparison_table_id_from_route(comparison_route(table_id)) == table_id
    assert comparison_table_id_from_route("pristine_relaxation") is None
    assert comparison_table_id_from_route("pristine_relaxation/comparison/") is None
    assert comparison_table_id_from_route("pristine_relaxation/comparison/not-a-uuid") is None
    assert comparison_table_id_from_route(None) is None


def test_generate_saves_opens_and_records_the_chosen_jobs(harness: _Harness) -> None:
    chosen = _completed(harness.relaxed.id)
    other = _completed(harness.relaxed.id, "beta")
    harness.jobs = [chosen, other]
    harness.choice = ([chosen.job_id], "My table")

    harness.controller().generate_table()

    assert harness.offered == [([chosen, other], "Comparison 1")]
    table = next(iter(harness.document.comparison_tables.values()))
    assert table.label == "My table"
    assert table.dft_labels == ["DFT (alpha)"]
    assert "metric_definitions" not in table.result_data
    assert harness.page.tables == [table]
    assert harness.calls == ["tree", "journal"]
    assert harness.routes == [comparison_route(table.id)]
    assert harness.statuses == ["Generated comparison table 'My table'"]
    assert harness.warnings == []
    assert [
        entry.action
        for entry in harness.document.provenance
        if entry.action == "create_comparison_table"
    ] == ["create_comparison_table"]


def test_generate_without_a_project_says_so(harness: _Harness) -> None:
    harness.document = None

    harness.controller().generate_table()

    assert harness.infos == [("No project", "Create or open a project first.")]


def test_generate_without_a_host_warns(harness: _Harness) -> None:
    harness.document.project.host_structure_id = None
    harness.document.structures.clear()

    harness.controller().generate_table()

    assert harness.warnings == [
        ("No host structure", "Load a host structure before generating a comparison table.")
    ]
    assert harness.offered == []


def test_generate_without_completed_jobs_says_so(harness: _Harness) -> None:
    harness.controller().generate_table()

    assert harness.infos == [
        (
            "No completed pristine relaxations",
            "Run and complete one or more pristine-relaxation jobs first.",
        )
    ]
    assert harness.offered == []


def test_cancelling_the_dialog_changes_nothing(harness: _Harness) -> None:
    harness.jobs = [_completed(harness.relaxed.id)]
    harness.choice = None

    harness.controller().generate_table()

    assert harness.document.comparison_tables == {}
    assert harness.routes == []


def test_generate_warns_when_the_selected_structures_are_gone(harness: _Harness) -> None:
    job = _completed(uuid4())
    harness.jobs = [job]
    harness.choice = ([job.job_id], "Label")

    harness.controller().generate_table()

    assert harness.warnings == [
        (
            "Comparison unavailable",
            "The selected jobs did not provide relaxed structures for comparison.",
        )
    ]
    assert harness.document.comparison_tables == {}


def test_generate_reports_a_failed_comparison(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _completed(harness.relaxed.id)
    harness.jobs = [job]
    harness.choice = ([job.job_id], "Label")

    def fail(**_kwargs):
        raise ValueError("cells differ")

    monkeypatch.setattr(
        "mufinder.app.workflow_controllers.comparison_tables.build_comparison_table", fail
    )

    harness.controller().generate_table()

    assert harness.warnings == [("Comparison failed", "cells differ")]
    assert harness.document.comparison_tables == {}


def test_rename_stores_the_title_keeps_the_table_open_and_saves(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _saved_table(harness)
    saves: list[object] = []
    monkeypatch.setattr(harness.service, "save_project", saves.append)

    harness.controller().rename_table(table.id, "  Renamed  ")

    assert table.label == "Renamed"
    assert harness.page.labels == [(table.id, "Renamed")]
    assert harness.calls == ["tree"]
    assert harness.routes == [comparison_route(table.id)]
    assert saves == [harness.document]
    assert harness.statuses == ["Renamed comparison table to 'Renamed'"]


def test_rename_to_nothing_falls_back_to_the_default_title(harness: _Harness) -> None:
    table = _saved_table(harness)

    harness.controller().rename_table(table.id, "   ")

    assert table.label == "Comparison Table"


def test_rename_reports_a_failed_autosave(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _saved_table(harness)

    def fail(_document):
        raise ValueError("read-only")

    monkeypatch.setattr(harness.service, "save_project", fail)

    harness.controller().rename_table(table.id, "Renamed")

    assert table.label == "Renamed"
    assert harness.statuses == ["Comparison table title was updated, but autosave failed"]


def test_rename_does_nothing_while_the_window_closes_or_for_an_unknown_table(
    harness: _Harness,
) -> None:
    table = _saved_table(harness)
    controller = harness.controller()

    controller.rename_table(uuid4(), "Unknown")
    harness.closing = True
    controller.rename_table(table.id, "Closing")

    assert table.label == "Comparison 1"
    assert harness.statuses == []


def test_delete_removes_the_table_and_returns_to_the_page(harness: _Harness) -> None:
    table = _saved_table(harness)
    kept = _saved_table(harness, "Kept")
    harness.current_route = comparison_route(table.id)

    harness.controller().delete_table(table.id)

    assert list(harness.document.comparison_tables) == [kept.id]
    assert harness.page.tables == [kept]
    assert harness.calls == ["tree"]
    assert harness.routes == ["pristine_relaxation"]
    assert harness.statuses == ["Deleted comparison table"]


def test_delete_leaves_the_route_alone_when_another_page_shows(harness: _Harness) -> None:
    table = _saved_table(harness)
    harness.current_route = "journal"

    harness.controller().delete_table(table.id)

    assert harness.routes == []


def test_delete_of_an_unknown_table_does_nothing(harness: _Harness) -> None:
    harness.controller().delete_table(uuid4())

    assert harness.statuses == []
    assert harness.calls == []


def test_sync_view_clears_the_page_without_a_project() -> None:
    page = _FakePage()
    page.tables = [SimpleNamespace()]
    controller = ComparisonTablesController(
        make_context(document=None),
        ComparisonTablesDeps(
            completed_jobs=list,
            choose_jobs=lambda _jobs, _label: None,
            show_warning=lambda _title, _text: None,
            refresh_project_tree=lambda: None,
            refresh_journal_view=lambda: None,
            active_route=lambda: None,
            open_route=lambda _route: None,
            shutting_down=lambda: False,
        ),
        page,
    )

    controller.sync_view()

    assert page.tables == []
