"""Behaviour tests for the pinned Muon Sites summary and the sources it selects."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.enums import JobStatus
from mufinder.services.muon_sites_summary import (
    ENERGY_BARRIERS_SCOPE,
    MUON_SITES_SUMMARY_SNAPSHOTS_KEY,
    NO_PINNED_SITES_NOTICE,
    QUANTUM_CORRECTIONS_SCOPE,
    build_muon_sites_summary_snapshot,
    downstream_site_scope,
    pinned_cluster_labels,
    pinned_muon_sites,
    pinned_source_selection,
    set_show_every_relaxed_site,
    show_every_relaxed_site,
    store_muon_sites_summary_snapshot,
    validated_muon_sites_summary_snapshot,
)
from mufinder.services.project_service import ProjectService
from mufinder.services.source_candidates import eligible_quantum_source_candidates
from tests.support.muon_sites import (
    MUONIUM_RELAXATION_CALC_TYPE,
    ClusterSpec,
    add_relaxed_muon_site,
    clustered_workspace,
    relaxation_job_id,
    store_workspaces,
)
from tests.support.structures import host_structure


@pytest.fixture
def project(tmp_path: Path):
    service = ProjectService()
    doc = service.create_project(tmp_path / "pins.mufinder", "Pins")
    host = host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    return service, doc, host


def _pin(doc, workspace) -> dict:
    snapshot = build_muon_sites_summary_snapshot(workspace)
    assert snapshot is not None
    store_muon_sites_summary_snapshot(doc, str(snapshot["charge_state"]), snapshot)
    return snapshot


def _two_site_project(doc, host):
    """Three relaxed copies: ``low_a`` and ``low_b`` relax to one site, ``high`` to another."""

    low_a = add_relaxed_muon_site(doc, label="low_a", total_energy=-100.0)
    low_b = add_relaxed_muon_site(doc, label="low_b", total_energy=-99.999)
    high = add_relaxed_muon_site(doc, label="high", total_energy=-99.5)
    workspace = clustered_workspace(
        host,
        [
            ClusterSpec(2, relaxation_job_id(doc, high), (0.5, 0.0, 0.0), 0.5, "4b"),
            ClusterSpec(1, relaxation_job_id(doc, low_a), (0.25, 0.25, 0.25), 0.0, "8c"),
        ],
    )
    store_workspaces(doc, workspace)
    return low_a, low_b, high, workspace


# --- snapshots ----------------------------------------------------------------


def test_snapshot_records_each_clusters_representative_job(project) -> None:
    _service, doc, host = project
    low_a, _low_b, high, workspace = _two_site_project(doc, host)

    snapshot = build_muon_sites_summary_snapshot(workspace)

    assert snapshot is not None
    assert [site["cluster_index"] for site in snapshot["sites"]] == [1, 2]
    assert [site["representative_job_id"] for site in snapshot["sites"]] == [
        str(relaxation_job_id(doc, low_a)),
        str(relaxation_job_id(doc, high)),
    ]
    assert [site["wyckoff_label"] for site in snapshot["sites"]] == ["8c", "4b"]


def test_snapshot_needs_a_host_and_clusters(project) -> None:
    _service, _doc, host = project
    workspace = clustered_workspace(host, [])
    assert build_muon_sites_summary_snapshot(workspace) is None
    workspace = clustered_workspace(host, [ClusterSpec(1, uuid4())])
    workspace["host_structure_id"] = ""
    assert build_muon_sites_summary_snapshot(workspace) is None


def test_validated_snapshot_hides_a_pin_for_another_host(project) -> None:
    _service, doc, host = project
    _low_a, _low_b, _high, workspace = _two_site_project(doc, host)
    _pin(doc, workspace)
    assert validated_muon_sites_summary_snapshot(doc, POSITIVE_MUON_CHARGE_STATE) is not None

    doc.project.host_structure_id = uuid4()

    assert validated_muon_sites_summary_snapshot(doc, POSITIVE_MUON_CHARGE_STATE) is None
    assert pinned_muon_sites(doc, POSITIVE_MUON_CHARGE_STATE) == []


def test_unpinning_removes_the_snapshot(project) -> None:
    _service, doc, host = project
    _low_a, _low_b, _high, workspace = _two_site_project(doc, host)
    _pin(doc, workspace)

    store_muon_sites_summary_snapshot(doc, POSITIVE_MUON_CHARGE_STATE, None)

    assert doc.ui_state[MUON_SITES_SUMMARY_SNAPSHOTS_KEY] == {}


# --- resolving pinned sites --------------------------------------------------


def test_pinned_sites_resolve_to_their_representative_relaxed_sites(project) -> None:
    _service, doc, host = project
    low_a, _low_b, high, workspace = _two_site_project(doc, host)
    _pin(doc, workspace)

    selection = pinned_source_selection(doc)

    assert selection.pinned_charge_states == (POSITIVE_MUON_CHARGE_STATE,)
    assert selection.keys() == [str(low_a.id), str(high.id)]
    assert selection.labels_by_key() == {
        str(low_a.id): "8c · cluster 1",
        str(high.id): "4b · cluster 2",
    }
    assert selection.unresolved == ()


def test_a_pinned_site_whose_relaxation_was_reset_is_unresolved(project) -> None:
    _service, doc, host = project
    low_a, _low_b, high, workspace = _two_site_project(doc, host)
    _pin(doc, workspace)
    doc.jobs[relaxation_job_id(doc, high)].status = JobStatus.DEFINED

    selection = pinned_source_selection(doc)

    assert selection.keys() == [str(low_a.id)]
    assert [site.cluster_index for site in selection.unresolved] == [2]
    scope = downstream_site_scope(doc, QUANTUM_CORRECTIONS_SCOPE)
    assert "1 pinned site is no longer an eligible relaxed site" in scope.notice


def test_a_legacy_snapshot_takes_job_ids_from_the_pinned_workspace(project) -> None:
    _service, doc, host = project
    low_a, _low_b, high, workspace = _two_site_project(doc, host)
    snapshot = _pin(doc, workspace)
    for site in snapshot["sites"]:
        del site["representative_job_id"]
    store_muon_sites_summary_snapshot(doc, POSITIVE_MUON_CHARGE_STATE, snapshot)

    assert pinned_source_selection(doc).keys() == [str(low_a.id), str(high.id)]


def test_a_legacy_snapshot_whose_workspace_is_gone_is_unresolved(project) -> None:
    _service, doc, host = project
    _low_a, _low_b, _high, workspace = _two_site_project(doc, host)
    snapshot = _pin(doc, workspace)
    for site in snapshot["sites"]:
        del site["representative_job_id"]
    store_muon_sites_summary_snapshot(doc, POSITIVE_MUON_CHARGE_STATE, snapshot)
    store_workspaces(doc)

    selection = pinned_source_selection(doc)

    assert selection.sources == ()
    assert len(selection.unresolved) == 2


def test_a_pin_only_resolves_sites_of_its_own_charge_state(project) -> None:
    _service, doc, host = project
    muonium = add_relaxed_muon_site(doc, label="mu0", charge_calc_type=MUONIUM_RELAXATION_CALC_TYPE)
    workspace = clustered_workspace(host, [ClusterSpec(1, relaxation_job_id(doc, muonium))])
    _pin(doc, workspace)  # Pinned as a positive-muon analysis.

    selection = pinned_source_selection(doc)

    assert selection.sources == ()
    assert len(selection.unresolved) == 1


# --- downstream scope --------------------------------------------------------


def test_downstream_scope_lists_pinned_sites_then_tracked_ones(project) -> None:
    _service, doc, host = project
    low_a, low_b, high, workspace = _two_site_project(doc, host)
    _pin(doc, workspace)
    candidates = eligible_quantum_source_candidates(doc)

    scope = downstream_site_scope(doc, QUANTUM_CORRECTIONS_SCOPE, candidates)

    assert not scope.show_every_site
    assert scope.notice == ""
    assert [c.key for c in scope.scoped_candidates(candidates)] == [str(low_a.id), str(high.id)]
    assert [c.key for c in scope.scoped_candidates(candidates, keep_keys=[str(low_b.id)])] == [
        str(low_a.id),
        str(high.id),
        str(low_b.id),
    ]
    assert scope.pinned_label(str(high.id)) == "4b · cluster 2"
    assert scope.pinned_label(str(low_b.id)) == ""


def test_show_every_relaxed_site_lists_every_eligible_site(project) -> None:
    _service, doc, host = project
    _two_site_project(doc, host)
    candidates = eligible_quantum_source_candidates(doc)
    set_show_every_relaxed_site(doc, QUANTUM_CORRECTIONS_SCOPE, True)

    scope = downstream_site_scope(doc, QUANTUM_CORRECTIONS_SCOPE, candidates)

    assert scope.show_every_site
    assert scope.scoped_candidates(candidates) == candidates
    assert scope.pinned_label(candidates[0].key) == ""
    # The choice is per workflow.
    assert not show_every_relaxed_site(doc, ENERGY_BARRIERS_SCOPE)


def test_no_pin_lists_nothing_and_says_so(project) -> None:
    _service, doc, host = project
    _two_site_project(doc, host)
    candidates = eligible_quantum_source_candidates(doc)

    scope = downstream_site_scope(doc, QUANTUM_CORRECTIONS_SCOPE, candidates)

    assert scope.scoped_candidates(candidates) == []
    assert scope.notice == NO_PINNED_SITES_NOTICE


def test_a_charge_state_without_a_pin_is_named_in_the_notice(project) -> None:
    _service, doc, host = project
    _low_a, _low_b, _high, workspace = _two_site_project(doc, host)
    add_relaxed_muon_site(doc, label="mu0", charge_calc_type=MUONIUM_RELAXATION_CALC_TYPE)
    _pin(doc, workspace)

    scope = downstream_site_scope(doc, ENERGY_BARRIERS_SCOPE)

    assert "No muonium sites are pinned" in scope.notice


def test_pinned_cluster_labels_key_sites_by_workspace_and_cluster(project) -> None:
    _service, doc, host = project
    _low_a, _low_b, _high, workspace = _two_site_project(doc, host)
    muonium = add_relaxed_muon_site(doc, label="mu0", charge_calc_type=MUONIUM_RELAXATION_CALC_TYPE)
    mu_workspace = clustered_workspace(
        host, [ClusterSpec(1, relaxation_job_id(doc, muonium))], charge_state=MUONIUM_CHARGE_STATE
    )
    _pin(doc, workspace)
    _pin(doc, mu_workspace)

    labels = pinned_cluster_labels(doc)

    assert list(labels) == [
        (workspace["id"], 1),
        (workspace["id"], 2),
        (mu_workspace["id"], 1),
    ]
    assert labels[(mu_workspace["id"], 1)].label == "Cluster 1"


def test_show_every_relaxed_site_survives_save_and_reload(project) -> None:
    service, doc, _host = project
    set_show_every_relaxed_site(doc, ENERGY_BARRIERS_SCOPE, True)

    service.save_project(doc)
    reloaded = service.load_project(doc.project_root)

    assert show_every_relaxed_site(reloaded, ENERGY_BARRIERS_SCOPE)
    assert not show_every_relaxed_site(reloaded, QUANTUM_CORRECTIONS_SCOPE)
