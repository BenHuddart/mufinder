"""Behaviour tests for the stored relaxation analysis workspaces and their summary pins."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.persistence.json_store import ProjectStore
from mufinder.services.analysis_workspaces import (
    PinRefusal,
    add_analysis_workspace,
    analysis_workspaces,
    delete_analysis_workspace,
    find_analysis_workspace,
    pin_analysis_workspace,
    pinned_charge_states,
    record_grouping_applied,
    set_analysis_workspaces,
    workspace_host,
)
from mufinder.services.muon_sites_summary import (
    MUON_SITES_SUMMARY_SNAPSHOTS_KEY,
    muon_sites_summary_snapshots,
    store_muon_sites_summary_snapshot,
    unpin_muon_sites,
    workspace_job_ids,
)
from mufinder.services.project_service import ProjectService
from mufinder.services.project_summary import RELAXATION_ANALYSIS_WORKSPACES_KEY
from tests.support.muon_sites import ClusterSpec, clustered_workspace, store_workspaces
from tests.support.structures import host_structure


@pytest.fixture
def project(tmp_path: Path):
    service = ProjectService()
    doc = service.create_project(tmp_path / "workspaces.mufinder", "Workspaces")
    host = host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    return service, doc, host


def _workspace(host, *, charge_state: str = POSITIVE_MUON_CHARGE_STATE, name: str = "Analysis"):
    return clustered_workspace(
        host,
        [ClusterSpec(1, uuid4(), (0.25, 0.25, 0.25), 0.0, "8c")],
        charge_state=charge_state,
        name=name,
    )


# --- the stored list --------------------------------------------------------------


def test_reading_skips_entries_that_are_not_workspaces_and_returns_copies(project) -> None:
    _service, doc, host = project
    workspace = _workspace(host)
    doc.ui_state[RELAXATION_ANALYSIS_WORKSPACES_KEY] = [workspace, "junk", 3]

    (read,) = analysis_workspaces(doc)
    read["name"] = "Changed"

    assert workspace["name"] == "Analysis"
    assert analysis_workspaces(None) == []
    doc.ui_state[RELAXATION_ANALYSIS_WORKSPACES_KEY] = {"not": "a list"}
    assert analysis_workspaces(doc) == []


def test_storing_copies_each_workspace(project) -> None:
    _service, doc, host = project
    workspace = _workspace(host)

    set_analysis_workspaces(doc, [workspace])
    workspace["name"] = "Changed"

    assert doc.ui_state[RELAXATION_ANALYSIS_WORKSPACES_KEY][0]["name"] == "Analysis"
    set_analysis_workspaces(None, [workspace])  # no project: nothing to store


def test_finding_a_workspace_by_id(project) -> None:
    _service, doc, host = project
    first, second = _workspace(host, name="First"), _workspace(host, name="Second")
    store_workspaces(doc, first, second)

    assert find_analysis_workspace(doc, second["id"])["name"] == "Second"
    assert find_analysis_workspace(doc, "missing") is None
    assert find_analysis_workspace(None, first["id"]) is None


def test_adding_a_workspace_records_it_in_the_journal(project) -> None:
    _service, doc, host = project
    workspace = _workspace(host, name="Sites")

    add_analysis_workspace(doc, workspace)

    assert [w["id"] for w in analysis_workspaces(doc)] == [workspace["id"]]
    record = doc.provenance[-1]
    assert record.action == "create_analysis_workspace"
    assert record.parameters["workspace_id"] == workspace["id"]
    assert record.parameters["workspace_name"] == "Sites"


def test_a_workspace_whose_record_cannot_be_written_is_kept_and_saved(
    project, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    service, doc, host = project
    workspace = _workspace(host, name="Sites")
    records_before = list(doc.provenance)

    def refuse(_store, _payload) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ProjectStore, "append_provenance", refuse)

    with caplog.at_level("ERROR"):
        assert add_analysis_workspace(doc, workspace) is False

    assert [w["id"] for w in analysis_workspaces(doc)] == [workspace["id"]]
    assert doc.provenance == records_before
    assert "disk full" in caplog.text
    # The workspace is a project edit like any other: the next save keeps it.
    monkeypatch.undo()
    service.save_project(doc)
    reloaded = service.load_project(doc.project_root)
    assert [w["id"] for w in analysis_workspaces(reloaded)] == [workspace["id"]]
    assert [record.action for record in reloaded.provenance] == [
        record.action for record in records_before
    ]


def test_adding_a_workspace_says_its_record_was_written(project) -> None:
    _service, doc, host = project

    assert add_analysis_workspace(doc, _workspace(host)) is True


def test_the_workspace_host_is_the_structure_it_was_made_for(project) -> None:
    _service, doc, host = project
    workspace = _workspace(host)

    assert workspace_host(doc, workspace) is host
    assert workspace_host(None, workspace) is None
    assert workspace_host(doc, {**workspace, "host_structure_id": "not an id"}) is None
    assert workspace_host(doc, {**workspace, "host_structure_id": str(uuid4())}) is None


def test_workspace_job_ids_skip_values_that_are_not_ids() -> None:
    job_id = uuid4()

    assert workspace_job_ids({"job_ids": [str(job_id), "junk", None]}) == [job_id]
    assert workspace_job_ids({"job_ids": str(job_id)}) == []
    assert workspace_job_ids({}) == []


# --- deleting ---------------------------------------------------------------------


def test_deleting_removes_only_that_workspace_and_its_pin(project) -> None:
    _service, doc, host = project
    stale = _workspace(host, name="Preliminary")
    other = _workspace(host, charge_state=MUONIUM_CHARGE_STATE, name="Muonium")
    store_workspaces(doc, stale, other)
    assert pin_analysis_workspace(doc, stale["id"]).charge_state == POSITIVE_MUON_CHARGE_STATE
    assert pin_analysis_workspace(doc, other["id"]).charge_state == MUONIUM_CHARGE_STATE
    assert pinned_charge_states(doc, stale["id"]) == [POSITIVE_MUON_CHARGE_STATE]

    unpinned = delete_analysis_workspace(doc, stale["id"])

    assert unpinned == [POSITIVE_MUON_CHARGE_STATE]
    assert [w["id"] for w in analysis_workspaces(doc)] == [other["id"]]
    assert list(muon_sites_summary_snapshots(doc)) == [MUONIUM_CHARGE_STATE]


def test_deleting_an_unpinned_workspace_unpins_nothing(project) -> None:
    _service, doc, host = project
    workspace = _workspace(host)
    store_workspaces(doc, workspace)

    assert delete_analysis_workspace(doc, workspace["id"]) == []
    assert analysis_workspaces(doc) == []


def test_deleting_an_unknown_workspace_changes_nothing(project) -> None:
    _service, doc, host = project
    workspace = _workspace(host)
    store_workspaces(doc, workspace)

    assert delete_analysis_workspace(doc, "missing") is None
    assert [w["id"] for w in analysis_workspaces(doc)] == [workspace["id"]]


# --- grouping ---------------------------------------------------------------------


def test_recording_a_grouping_adds_one_finding_a_day(project) -> None:
    _service, doc, host = project
    workspace = _workspace(host)
    store_workspaces(doc, workspace)

    assert record_grouping_applied(doc, workspace["id"])
    count = len(doc.provenance)
    assert doc.provenance[-1].action == "sites_grouped"

    # A repeat is still reported, so the caller saves it, but adds no record.
    assert record_grouping_applied(doc, workspace["id"])
    assert len(doc.provenance) == count


def test_a_workspace_without_a_grouping_records_nothing(project) -> None:
    _service, doc, host = project
    workspace = {**_workspace(host), "clustering": {"result": None}}
    store_workspaces(doc, workspace)

    assert not record_grouping_applied(doc, workspace["id"])
    assert not record_grouping_applied(doc, "missing")
    assert doc.provenance == []


# --- pinning ----------------------------------------------------------------------


def test_pinning_stores_the_snapshot_under_its_charge_state(project) -> None:
    _service, doc, host = project
    workspace = _workspace(host)
    store_workspaces(doc, workspace)

    result = pin_analysis_workspace(doc, workspace["id"])

    assert result.charge_state == POSITIVE_MUON_CHARGE_STATE
    assert result.refusal is None
    stored = doc.ui_state[MUON_SITES_SUMMARY_SNAPSHOTS_KEY][POSITIVE_MUON_CHARGE_STATE]
    assert stored["workspace_id"] == workspace["id"]
    assert stored["host_structure_id"] == str(host.id)


def test_pinning_an_unknown_workspace_returns_none(project) -> None:
    _service, doc, _host = project

    assert pin_analysis_workspace(doc, "missing") is None


def test_pinning_an_unclustered_workspace_is_refused(project) -> None:
    _service, doc, host = project
    workspace = {**_workspace(host), "clustering": {"result": None}}
    store_workspaces(doc, workspace)

    assert pin_analysis_workspace(doc, workspace["id"]).refusal is PinRefusal.NOT_CLUSTERED
    assert muon_sites_summary_snapshots(doc) == {}


def test_pinning_a_workspace_of_another_host_is_refused(project) -> None:
    service, doc, _host = project
    other_host = host_structure()
    service.add_structure(doc, other_host)
    workspace = _workspace(other_host)
    store_workspaces(doc, workspace)

    assert pin_analysis_workspace(doc, workspace["id"]).refusal is PinRefusal.OTHER_HOST
    assert muon_sites_summary_snapshots(doc) == {}


def test_pinning_a_charge_state_without_a_panel_is_refused(project) -> None:
    _service, doc, host = project
    workspace = _workspace(host, charge_state="neutral")
    store_workspaces(doc, workspace)

    result = pin_analysis_workspace(doc, workspace["id"])

    assert result.refusal is PinRefusal.UNSUPPORTED_CHARGE_STATE
    assert muon_sites_summary_snapshots(doc) == {}


def test_unpinning_reports_whether_there_was_a_pin(project) -> None:
    _service, doc, _host = project
    store_muon_sites_summary_snapshot(
        doc, POSITIVE_MUON_CHARGE_STATE, {"workspace_id": "ws", "sites": []}
    )

    assert unpin_muon_sites(doc, POSITIVE_MUON_CHARGE_STATE)
    assert muon_sites_summary_snapshots(doc) == {}
    assert not unpin_muon_sites(doc, POSITIVE_MUON_CHARGE_STATE)
