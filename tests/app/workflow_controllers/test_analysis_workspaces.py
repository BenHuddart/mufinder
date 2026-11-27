"""Tests for the analysis-workspace controller: workspaces and Muon Sites summary pins."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from PySide6.QtWidgets import QDialog

from mufinder.app import workflow_table_utils
from mufinder.app.workflow_controllers import analysis_workspaces as controller_module
from mufinder.app.workflow_controllers.analysis_workspaces import (
    AnalysisWorkspacesController,
    AnalysisWorkspacesDeps,
)
from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.routes import relaxation_analysis_workspace_route
from mufinder.persistence.json_store import ProjectStore
from mufinder.services.analysis_workspaces import analysis_workspaces
from mufinder.services.muon_sites_summary import (
    muon_sites_summary_snapshots,
    store_muon_sites_summary_snapshot,
)
from mufinder.services.project_service import ProjectService
from mufinder.services.source_candidates import CompletedRelaxationAnalysisSource
from tests.support.controllers import make_context
from tests.support.muon_sites import ClusterSpec, clustered_workspace, store_workspaces
from tests.support.structures import host_structure


class _FakeView:
    def __init__(self) -> None:
        self.added: list[tuple[str, list[UUID]]] = []
        self.focused: list[tuple[str, int | None]] = []
        self.resolves = True

    def add_job_ids_to_workspace(self, workspace_id: str, job_ids: list[UUID]) -> None:
        self.added.append((workspace_id, list(job_ids)))

    def focus_workspace(self, workspace_id: str, *, cluster_index: int | None = None) -> bool:
        self.focused.append((workspace_id, cluster_index))
        return self.resolves


def _source(
    label: str,
    *,
    charge_state: str = POSITIVE_MUON_CHARGE_STATE,
    calculation: tuple[object, ...] = ("Si2H", None, None),
) -> CompletedRelaxationAnalysisSource:
    return CompletedRelaxationAnalysisSource(
        job_id=uuid4(),
        job_label=label,
        charge_state=charge_state,
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=(0.25, 0.25, 0.25),
        total_energy_ev=-10.0,
        calculation=calculation,
    )


@dataclass
class _Harness:
    """A controller on a real project, recording everything it asks the window for."""

    service: ProjectService
    document: object
    host: object
    view: _FakeView = field(default_factory=_FakeView)
    sources: list[CompletedRelaxationAnalysisSource] = field(default_factory=list)
    infos: list[tuple[str, str]] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    persisted: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    current_route: object = "muon_analysis"
    closing: bool = False

    def controller(self) -> AnalysisWorkspacesController:
        return AnalysisWorkspacesController(
            make_context(
                document=self.document,
                project_service=self.service,
                infos=self.infos,
                statuses=self.statuses,
                persisted=self.persisted,
            ),
            AnalysisWorkspacesDeps(
                dialog_parent=lambda: None,  # type: ignore[arg-type,return-value]
                completed_sources=lambda: list(self.sources),
                refresh_analysis_page=lambda: self.calls.append("analysis"),
                refresh_navigator=lambda: self.calls.append("navigator"),
                refresh_journal_view=lambda: self.calls.append("journal"),
                refresh_muon_sites_summary=lambda: self.calls.append("summary"),
                refresh_pinned_site_consumers=lambda: self.calls.append("consumers"),
                active_route=lambda: self.current_route,
                open_route=self.routes.append,
                shutting_down=lambda: self.closing,
            ),
            self.view,  # type: ignore[arg-type]
        )


@pytest.fixture
def harness(tmp_path: Path) -> _Harness:
    service = ProjectService()
    document = service.create_project(tmp_path / "analysis.mufinder", "Analysis")
    host = host_structure()
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    return _Harness(service=service, document=document, host=host)


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        controller_module.QMessageBox,
        "warning",
        lambda _parent, title, text, *_a, **_k: shown.append((title, text)),
    )
    return shown


def _name(monkeypatch: pytest.MonkeyPatch, name: str, accepted: bool = True) -> list[str]:
    defaults: list[str] = []

    def get_text(_parent, _title, _label, _mode, default):
        defaults.append(default)
        return name, accepted

    monkeypatch.setattr(controller_module.QInputDialog, "getText", get_text)
    return defaults


def _workspace(host, *, charge_state: str = POSITIVE_MUON_CHARGE_STATE, name: str = "Analysis"):
    return clustered_workspace(
        host, [ClusterSpec(1, uuid4(), wyckoff_label="8c")], charge_state=charge_state, name=name
    )


# --- creating -----------------------------------------------------------------------


def test_create_stores_the_workspace_records_it_and_opens_it(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.sources = [_source("A"), _source("B"), _source("C")]
    defaults = _name(monkeypatch, "Sites")

    harness.controller().create_workspace(["A", "C"])

    (workspace,) = analysis_workspaces(harness.document)
    assert defaults == ["positive muon analysis"]
    assert workspace["name"] == "Sites"
    assert workspace["charge_state"] == POSITIVE_MUON_CHARGE_STATE
    assert workspace["host_structure_id"] == str(harness.host.id)
    assert workspace["job_ids"] == [str(harness.sources[0].job_id), str(harness.sources[2].job_id)]
    assert harness.document.provenance[-1].action == "create_analysis_workspace"
    assert harness.persisted == ["The analysis workspace was created, but autosave failed."]
    assert harness.calls == ["analysis", "navigator", "journal"]
    assert harness.routes == [relaxation_analysis_workspace_route(str(workspace["id"]))]
    assert harness.statuses == ["Created analysis workspace 'Sites'"]


def test_a_workspace_whose_record_cannot_be_written_is_still_saved_and_opened(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, warnings
) -> None:
    harness.sources = [_source("A")]
    _name(monkeypatch, "Sites")
    records_before = len(harness.document.provenance)

    def refuse(_store, _payload) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ProjectStore, "append_provenance", refuse)

    harness.controller().create_workspace(["A"])

    (workspace,) = analysis_workspaces(harness.document)
    assert len(harness.document.provenance) == records_before
    assert harness.persisted == ["The analysis workspace was created, but autosave failed."]
    assert harness.calls == ["analysis", "navigator", "journal"]
    assert harness.routes == [relaxation_analysis_workspace_route(str(workspace["id"]))]
    assert warnings == [
        (
            "Journal not updated",
            "The analysis workspace 'Sites' was created, but its Journal record could not "
            "be written.",
        )
    ]
    assert harness.statuses == ["Created analysis workspace 'Sites'"]


def test_create_takes_a_single_label(harness: _Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    harness.sources = [_source("A")]
    _name(monkeypatch, "Sites")

    harness.controller().create_workspace("A")

    assert len(analysis_workspaces(harness.document)) == 1


def test_create_needs_a_project(harness: _Harness) -> None:
    harness.document = None

    harness.controller().create_workspace(["A"])

    assert harness.infos == [("No project", "Create or open a project first.")]


def test_create_needs_a_host(harness: _Harness, warnings) -> None:
    harness.document.project.host_structure_id = None

    harness.controller().create_workspace(["A"])

    assert warnings == [
        ("No host structure", "Load a host structure before creating an analysis workspace.")
    ]


def test_create_needs_completed_relaxations(harness: _Harness) -> None:
    harness.sources = [_source("A")]

    harness.controller().create_workspace(["B"])

    assert harness.infos[0][0] == "No completed relaxations"
    assert analysis_workspaces(harness.document) == []


def test_create_refuses_mixed_charge_states(harness: _Harness, warnings) -> None:
    harness.sources = [_source("A"), _source("B", charge_state=MUONIUM_CHARGE_STATE)]

    harness.controller().create_workspace(["A", "B"])

    assert [title for title, _text in warnings] == ["Mixed charge states"]
    assert analysis_workspaces(harness.document) == []


def test_create_asks_before_mixing_calculations(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.sources = [
        _source("A", calculation=("Si16H", None, None)),
        _source("B", calculation=("Si54H", None, None)),
    ]
    asked: list[tuple[str, str]] = []

    def answer(_parent, title, *_args, **kwargs) -> str:
        asked.append((title, kwargs["default"]))
        return "cancel"

    monkeypatch.setattr(workflow_table_utils, "confirm_choice", answer)

    harness.controller().create_workspace(["A", "B"])

    assert asked == [("Different calculations", "cancel")]
    assert analysis_workspaces(harness.document) == []


def test_cancelling_the_name_creates_nothing(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.sources = [_source("A")]
    _name(monkeypatch, "Sites", accepted=False)

    harness.controller().create_workspace(["A"])

    assert analysis_workspaces(harness.document) == []
    assert harness.document.provenance == []
    assert harness.persisted == []


# --- the page's edits ---------------------------------------------------------------


def test_storing_a_rename_rebuilds_the_navigator(harness: _Harness) -> None:
    workspace = _workspace(harness.host)
    store_workspaces(harness.document, workspace)

    harness.controller().store_workspaces([{**workspace, "name": "Renamed"}, "junk"])

    assert [w["name"] for w in analysis_workspaces(harness.document)] == ["Renamed"]
    assert harness.calls == ["navigator"]


def test_storing_other_edits_also_refreshes_the_navigator(harness: _Harness) -> None:
    workspace = _workspace(harness.host)
    store_workspaces(harness.document, workspace)

    harness.controller().store_workspaces([{**workspace, "clustering": {}}])

    assert analysis_workspaces(harness.document)[0]["clustering"] == {}
    assert harness.calls == ["navigator"]


def test_storing_is_skipped_while_the_window_closes(harness: _Harness) -> None:
    workspace = _workspace(harness.host)
    store_workspaces(harness.document, workspace)
    harness.closing = True

    harness.controller().store_workspaces([])

    assert len(analysis_workspaces(harness.document)) == 1
    assert harness.calls == []


# --- deleting -----------------------------------------------------------------------


def test_delete_confirms_then_removes_the_workspace_and_its_pin(
    harness: _Harness, confirmed_destructive_actions
) -> None:
    stale = _workspace(harness.host, name="Preliminary")
    kept = _workspace(harness.host, name="Full")
    store_workspaces(harness.document, stale, kept)
    store_muon_sites_summary_snapshot(
        harness.document,
        POSITIVE_MUON_CHARGE_STATE,
        {"workspace_id": stale["id"], "sites": []},
    )
    harness.current_route = relaxation_analysis_workspace_route(str(stale["id"]))

    harness.controller().delete_workspace(str(stale["id"]))

    prompt = confirmed_destructive_actions[-1]
    assert prompt["text"] == "Delete analysis 'Preliminary'?"
    assert prompt["accept_label"] == "Delete"
    assert "that pin is removed too" in prompt["detail"]
    assert [w["id"] for w in analysis_workspaces(harness.document)] == [kept["id"]]
    assert muon_sites_summary_snapshots(harness.document) == {}
    assert harness.routes == ["muon_analysis"]
    assert harness.calls == ["analysis", "navigator", "summary", "consumers"]
    assert harness.persisted == ["The analysis was deleted, but autosave failed."]
    assert harness.statuses == ["Deleted analysis 'Preliminary'"]


def test_delete_of_an_unpinned_workspace_off_its_route(
    harness: _Harness, confirmed_destructive_actions
) -> None:
    workspace = _workspace(harness.host, name="  ")
    store_workspaces(harness.document, workspace)

    harness.controller().delete_workspace(str(workspace["id"]))

    assert confirmed_destructive_actions[-1]["text"] == "Delete analysis 'Relaxation analysis'?"
    assert "pin" not in confirmed_destructive_actions[-1]["detail"]
    assert harness.routes == []
    assert harness.calls == ["analysis", "navigator"]


def test_declining_the_delete_keeps_the_workspace(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(harness.host)
    store_workspaces(harness.document, workspace)
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)

    harness.controller().delete_workspace(str(workspace["id"]))

    assert len(analysis_workspaces(harness.document)) == 1
    assert harness.persisted == []


def test_deleting_an_unknown_workspace_asks_nothing(
    harness: _Harness, confirmed_destructive_actions
) -> None:
    harness.controller().delete_workspace("missing")

    assert confirmed_destructive_actions == []


# --- adding relaxations -------------------------------------------------------------


class _Picker:
    picked: list[UUID] = []
    accepted = True
    opened: list[dict] = []

    def __init__(self, **kwargs) -> None:
        _Picker.opened.append(kwargs)

    def exec(self) -> int:
        return QDialog.DialogCode.Accepted if _Picker.accepted else QDialog.DialogCode.Rejected

    def selected_job_ids(self) -> list[UUID]:
        return list(_Picker.picked)


@pytest.fixture
def picker(monkeypatch: pytest.MonkeyPatch) -> type[_Picker]:
    _Picker.picked, _Picker.accepted, _Picker.opened = [], True, []
    monkeypatch.setattr(controller_module, "RelaxationAnalysisJobDialog", _Picker)
    return _Picker


def test_adding_offers_relaxations_of_the_workspace_charge_state(harness: _Harness, picker) -> None:
    held, joining = _source("A"), _source("B")
    harness.sources = [held, joining, _source("M", charge_state=MUONIUM_CHARGE_STATE)]
    workspace = {**_workspace(harness.host), "job_ids": [str(held.job_id), "junk"]}
    store_workspaces(harness.document, workspace)
    picker.picked = [joining.job_id]

    harness.controller().add_relaxations(str(workspace["id"]))

    (opened,) = picker.opened
    assert [source.job_label for source in opened["sources"]] == ["A", "B"]
    assert opened["excluded_job_ids"] == {held.job_id}
    assert opened["reference_job_ids"] == {held.job_id}
    assert harness.view.added == [(str(workspace["id"]), [joining.job_id])]
    assert harness.statuses == ["Added 1 relaxation result to the analysis workspace"]


def test_adding_with_no_matching_relaxations_says_so(harness: _Harness, picker) -> None:
    harness.sources = [_source("M", charge_state=MUONIUM_CHARGE_STATE)]
    workspace = _workspace(harness.host)
    store_workspaces(harness.document, workspace)

    harness.controller().add_relaxations(str(workspace["id"]))

    assert harness.infos == [
        (
            "No matching relaxations",
            "No completed positive muon relaxation results are available to add.",
        )
    ]
    assert picker.opened == []


def test_adding_nothing_or_cancelling_changes_nothing(harness: _Harness, picker) -> None:
    harness.sources = [_source("A")]
    workspace = _workspace(harness.host)
    store_workspaces(harness.document, workspace)

    harness.controller().add_relaxations(str(workspace["id"]))
    picker.accepted = False
    picker.picked = [harness.sources[0].job_id]
    harness.controller().add_relaxations(str(workspace["id"]))
    harness.controller().add_relaxations("missing")

    assert harness.view.added == []


def test_adding_from_another_calculation_asks_first(
    harness: _Harness, picker, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = _source("A", calculation=("Si16H", None, None))
    joining = _source("B", calculation=("Si54H", None, None))
    harness.sources = [held, joining]
    workspace = {**_workspace(harness.host), "job_ids": [str(held.job_id)]}
    store_workspaces(harness.document, workspace)
    picker.picked = [joining.job_id]
    monkeypatch.setattr(workflow_table_utils, "confirm_choice", lambda *_a, **_k: "cancel")

    harness.controller().add_relaxations(str(workspace["id"]))
    assert harness.view.added == []

    monkeypatch.setattr(workflow_table_utils, "confirm_choice", lambda *_a, **_k: "add")
    harness.controller().add_relaxations(str(workspace["id"]))
    assert harness.view.added == [(str(workspace["id"]), [joining.job_id])]


# --- grouping -----------------------------------------------------------------------


def test_recording_a_grouping_saves_and_refreshes_the_journal(harness: _Harness) -> None:
    workspace = _workspace(harness.host)
    store_workspaces(harness.document, workspace)

    harness.controller().record_grouping(str(workspace["id"]))

    assert harness.document.provenance[-1].action == "sites_grouped"
    assert harness.persisted == ["The grouping was applied, but autosave failed."]
    assert harness.calls == ["journal"]


def test_a_workspace_without_a_grouping_is_not_saved(harness: _Harness) -> None:
    workspace = {**_workspace(harness.host), "clustering": {"result": None}}
    store_workspaces(harness.document, workspace)

    harness.controller().record_grouping(str(workspace["id"]))

    assert harness.persisted == []
    assert harness.calls == []


# --- pins ---------------------------------------------------------------------------


def test_pinning_stores_the_sites_and_opens_the_summary(harness: _Harness) -> None:
    workspace = _workspace(harness.host, charge_state=MUONIUM_CHARGE_STATE)
    store_workspaces(harness.document, workspace)

    harness.controller().pin_to_summary(str(workspace["id"]))

    assert list(muon_sites_summary_snapshots(harness.document)) == [MUONIUM_CHARGE_STATE]
    assert harness.calls == ["summary", "consumers"]
    assert harness.routes == ["muon_sites"]
    assert harness.persisted == ["The muon sites summary was pinned, but autosave failed."]
    assert harness.statuses == ["Pinned muonium sites to the Muon Sites summary"]


@pytest.mark.parametrize(
    ("change", "title"),
    [
        ({"clustering": {"result": None}}, "Nothing to pin"),
        ({"charge_state": "neutral"}, "Cannot pin"),
    ],
)
def test_a_refused_pin_says_why(harness: _Harness, change: dict, title: str) -> None:
    workspace = {**_workspace(harness.host), **change}
    store_workspaces(harness.document, workspace)

    harness.controller().pin_to_summary(str(workspace["id"]))

    assert [shown for shown, _text in harness.infos] == [title]
    assert muon_sites_summary_snapshots(harness.document) == {}
    assert harness.persisted == []


def test_a_pin_from_another_host_says_to_switch_host(harness: _Harness) -> None:
    other_host = host_structure()
    harness.service.add_structure(harness.document, other_host)
    workspace = _workspace(other_host)
    store_workspaces(harness.document, workspace)

    harness.controller().pin_to_summary(str(workspace["id"]))

    ((title, text),) = harness.infos
    assert title == "Cannot pin"
    assert "Switch to that host" in text


def test_unpinning_removes_the_pin_and_saves(harness: _Harness) -> None:
    store_muon_sites_summary_snapshot(
        harness.document, POSITIVE_MUON_CHARGE_STATE, {"workspace_id": "ws", "sites": []}
    )

    harness.controller().unpin(POSITIVE_MUON_CHARGE_STATE)
    harness.controller().unpin(POSITIVE_MUON_CHARGE_STATE)

    assert muon_sites_summary_snapshots(harness.document) == {}
    assert harness.calls == ["summary", "consumers"]
    assert harness.persisted == ["The muon sites summary was updated, but autosave failed."]
    assert harness.statuses == ["Unpinned positive muon sites from the Muon Sites summary"]


def test_opening_a_pinned_analysis_focuses_its_workspace(harness: _Harness) -> None:
    controller = harness.controller()

    controller.open_pinned_analysis("ws")
    controller.focus_pinned_site("ws", 2)

    assert harness.routes == ["muon_analysis", "muon_analysis"]
    assert harness.view.focused == [("ws", None), ("ws", 2)]
    assert harness.statuses == []


def test_a_pinned_analysis_that_changed_opens_the_analysis_tab(harness: _Harness) -> None:
    harness.view.resolves = False

    harness.controller().focus_pinned_site("ws", 1)

    assert harness.statuses == ["The pinned analysis has changed — opened the Analysis tab instead"]
