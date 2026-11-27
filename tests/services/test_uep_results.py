"""Tests for the UEP batch searches a project stores as results."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.uep_results import (
    UEP_ANALYSIS_STATES_KEY,
    UEP_BATCH_SEARCH_RESULT_TYPE,
    regroup_saved_uep_searches,
    store_uep_analysis_state,
    store_uep_batch_search,
    uep_analysis_state,
    uep_batch_search,
    uep_batch_searches,
    uep_cluster_source_rows,
)

from tests.support.uep_results import fake_optimization, store_search

_A = 4.695
_C = 3.180
_X = 0.306
_LATTICE = ((_A, 0.0, 0.0), (0.0, _A, 0.0), (0.0, 0.0, _C))
_IMAGES_4D = ((0.0, 0.5, 0.25), (0.0, 0.5, 0.75), (0.5, 0.0, 0.25), (0.5, 0.0, 0.75))


def _cof2() -> Structure:
    return Structure(
        label="CoF2",
        lattice_vectors=[list(row) for row in _LATTICE],
        species=["Co", "Co", "F", "F", "F", "F"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [_X, _X, 0.0],
            [1.0 - _X, 1.0 - _X, 0.0],
            [0.5 + _X, 0.5 - _X, 0.5],
            [0.5 - _X, 0.5 + _X, 0.5],
        ],
    )


def _noisy_results() -> list[dict[str, object]]:
    generator = np.random.default_rng(5)
    return [
        asdict(
            fake_optimization(
                tuple(
                    float(value)
                    for value in np.mod(
                        np.asarray(_IMAGES_4D[index % 4]) + generator.normal(0.0, 1e-7, 3), 1.0
                    )
                ),
                classical=-15.976111,
                lattice_vectors=_LATTICE,
            )
        )
        for index in range(12)
    ]


def _old_state(results: list[dict[str, object]]) -> dict[str, object]:
    """A schema 7 analysis state: a search and its three-cluster hierarchical split."""

    return {
        "current_tab": 1,
        "batch_count": 12,
        "batch_results": results,
        "batch_summary_data": {
            "threshold_ang": 0.3,
            "clustering_method": "hier",
            "clustering_hier_t": 0.3,
            "clustering_kmeans_k": None,
            "clustering_linkage_method": "single",
            "feature_mode": "soprano_defect_asymmetric_fpos",
            "similarities": [],
            "clusters": [
                {
                    "cluster_index": index,
                    "size": 4,
                    "representative_index": index * 4 - 4,
                    "member_indexes": list(range(index * 4 - 4, index * 4)),
                    "representative_position_ang": [0.0, 2.3475, 0.795],
                    "representative_fractional_position": [0.0, 0.5, 0.25],
                    "minimum_energy_ev": None,
                    "average_energy_ev": None,
                    "standard_deviation_ev": None,
                }
                for index in (1, 2, 3)
            ],
        },
        "batch_summary_text": "Requested random starts: 12\nRelaxed sites available for clustering: 12",
        "batch_selected_keys": ["cluster:3:site:11"],
        "batch_selected_row": 2,
        "clustering_window_state": {"method": "kmeans", "kmeans_k": 4, "is_open": True},
    }


def _project(tmp_path: Path) -> tuple[ProjectService, ProjectDocument, Job]:
    service = ProjectService()
    document = service.create_project(tmp_path / "uep.mufinder", "UEP")
    host = _cof2()
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    definition = service.register_uep_definition(
        document,
        input_structure_id=host.id,
        parameters={"rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}]},
        output_directory=document.project_root / "files" / "uep",
    )
    job = service.add_job(
        document, label="uep_source", workflow_stage=WorkflowStage.UEP, calc_type="uep"
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    service.save_project(document)
    return service, document, job


def _saved_at_schema_7(document: ProjectDocument, states: dict[str, object]) -> None:
    """Rewrite a saved project as schema 7 kept its UEP searches: in ``state.json``."""

    root = document.project_root
    manifest = json.loads((root / "project.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = 7
    (root / "project.json").write_text(json.dumps(manifest), encoding="utf-8")
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    state[UEP_ANALYSIS_STATES_KEY] = states
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _old_project(
    tmp_path: Path, state: dict[str, object]
) -> tuple[ProjectService, ProjectDocument, Job]:
    service, document, job = _project(tmp_path)
    _saved_at_schema_7(document, {str(job.id): state})
    return service, document, job


# --- Storing ------------------------------------------------------------------


def _search(results: list[object], *, log: str = "Search log") -> dict[str, object]:
    return {
        "settings": {"gw_factor": 5.0, "batch_count": 12, "unknown": 1},
        "results": results,
        "grouping": None,
        "log": log,
    }


def test_a_finished_search_is_stored_as_one_result_linked_to_the_jobs_run(tmp_path: Path) -> None:
    _service, document, job = _project(tmp_path)
    run = CalculationRun(job_id=job.id, run_index=1)
    document.runs[run.id] = run

    assert store_uep_batch_search(document, job, _search(_noisy_results()))

    (record,) = [
        result
        for result in document.results.values()
        if result.result_type == UEP_BATCH_SEARCH_RESULT_TYPE
    ]
    assert record.calculation_run_id == run.id
    assert record.data["source_job_id"] == str(job.id)
    # Only the settings a search records are kept.
    assert record.data["settings"] == {"gw_factor": 5.0, "batch_count": 12}
    assert len(record.data["results"]) == 12
    assert record.data["log"] == "Search log"


def test_storing_the_same_search_again_changes_nothing_and_a_new_one_replaces_it(
    tmp_path: Path,
) -> None:
    _service, document, job = _project(tmp_path)
    store_uep_batch_search(document, job, _search(_noisy_results()))
    (first,) = uep_batch_searches(document)

    assert not store_uep_batch_search(document, job, _search(_noisy_results()))
    assert store_uep_batch_search(document, job, _search(_noisy_results()[:3], log="Again"))

    (search,) = uep_batch_searches(document)
    assert search.result_id == first.result_id
    assert len(search.results) == 3
    assert search.log == "Again"


@pytest.mark.parametrize("discarded", [None, {"results": []}])
def test_a_discarded_search_removes_its_result(tmp_path: Path, discarded) -> None:
    _service, document, job = _project(tmp_path)
    store_uep_batch_search(document, job, _search(_noisy_results()))

    assert store_uep_batch_search(document, job, discarded)
    assert uep_batch_search(document, job.id) is None
    assert not store_uep_batch_search(document, job, discarded)


def test_a_stored_search_is_on_disk_after_the_next_save_and_reloads_with_its_groups(
    tmp_path: Path,
) -> None:
    service, document, job = _project(tmp_path)
    results = _noisy_results()
    store_uep_batch_search(document, job, _search(results))

    service.save_project(document)
    loaded = service.load_project(document.project_root)

    search = uep_batch_search(loaded, job.id)
    assert search is not None
    assert search.results == json.loads(json.dumps(results))
    # A search stored without groups is grouped when the project opens, as before.
    assert [cluster["size"] for cluster in search.grouping["clusters"]] == [12]
    assert UEP_ANALYSIS_STATES_KEY not in json.loads(
        (document.project_root / "state.json").read_text(encoding="utf-8")
    )


def test_removing_the_source_job_removes_its_search(tmp_path: Path) -> None:
    from mufinder.services.job_file_cleanup import JobFileCleanup

    _service, document, job = _project(tmp_path)
    run = CalculationRun(job_id=job.id, run_index=1)
    document.runs[run.id] = run
    store_uep_batch_search(document, job, _search(_noisy_results()))

    JobFileCleanup(
        document_getter=lambda: document,
        job_output_directory=lambda _job: None,
        job_seed_name=lambda job: job.label,
    ).delete_job_records(job.id)

    assert uep_batch_searches(document) == []


# --- Generation rows ----------------------------------------------------------


def test_the_generation_rows_come_from_the_stored_search(tmp_path: Path) -> None:
    _service, document, job = _project(tmp_path)
    store_uep_batch_search(document, job, _search(_noisy_results()))
    regroup_saved_uep_searches(document)

    (row,) = uep_cluster_source_rows(document, {job.id})

    assert row["source_job_id"] == str(job.id)
    assert row["cluster_index"] == 1
    assert row["cluster_size"] == 12
    assert row["classification"] == "minimum"
    assert row["classical_energy_ev"] == pytest.approx(-15.976111)
    assert row["fractional_position"] == pytest.approx([0.0, 0.5, 0.25], abs=1e-6)
    assert row["properties"]["clustering_method"] == "symmetry_distance"
    assert row["properties"]["uep_position_frame"] == "host"
    # The window's display state plays no part.
    document.ui_state[UEP_ANALYSIS_STATES_KEY] = {str(job.id): {"current_tab": 1}}
    assert uep_cluster_source_rows(document, {job.id}) == [row]


def test_the_generation_rows_leave_out_inactive_jobs_and_ungrouped_searches(
    tmp_path: Path,
) -> None:
    _service, document, job = _project(tmp_path)
    store_search(document, job.id, results=_noisy_results(), grouping=None)
    other = uuid4()
    store_search(document, other, results=_noisy_results(), grouping={"clusters": []})

    assert uep_cluster_source_rows(document, {job.id, other}) == []
    regroup_saved_uep_searches(document)
    assert uep_cluster_source_rows(document, set()) == []
    assert len(uep_cluster_source_rows(document, {job.id})) == 1


# --- Loading a schema 7 project -----------------------------------------------


def test_loading_a_schema_7_search_with_an_old_cluster_summary_regroups_it(tmp_path: Path) -> None:
    service, document, job = _old_project(tmp_path, _old_state(_noisy_results()))

    loaded = service.load_project(document.project_root)

    search = uep_batch_search(loaded, job.id)
    summary = search.grouping
    assert summary["clustering_method"] == "symmetry_distance"
    assert summary["threshold_ang"] == 0.25
    assert [cluster["size"] for cluster in summary["clusters"]] == [12]
    assert summary["clusters"][0]["representative_fractional_position"] == pytest.approx(
        [0.0, 0.5, 0.25]
    )
    assert search.settings == {"batch_count": 12}
    state = loaded.ui_state[UEP_ANALYSIS_STATES_KEY][str(job.id)]
    assert state == {"current_tab": 1, "batch_count": 12}
    assert search.log.splitlines() == [
        "Requested random starts: 12",
        "Grouping: 12 minima in 1 site, within 0.25 Å and 1 meV.",
        "This search was saved without zero-point energies. Run the search again to compute them.",
    ]


def test_the_regrouped_search_reaches_disk_at_the_next_save_and_then_stays(
    tmp_path: Path,
) -> None:
    service, document, job = _old_project(tmp_path, _old_state(_noisy_results()))

    loaded = service.load_project(document.project_root)
    service.save_project(loaded)

    again = service.load_project(document.project_root)
    assert not regroup_saved_uep_searches(again)
    assert uep_batch_search(again, job.id) == uep_batch_search(loaded, job.id)


def test_regrouping_on_load_adds_nothing_to_the_journal(tmp_path: Path) -> None:
    service, document, _job = _old_project(tmp_path, _old_state(_noisy_results()))
    before = len(document.provenance)

    loaded = service.load_project(document.project_root)

    assert len(loaded.provenance) == before


def test_sites_created_from_the_old_clusters_keep_what_they_recorded(tmp_path: Path) -> None:
    service, document, job = _project(tmp_path)
    recorded = {
        "source_job_id": str(job.id),
        "source_job_label": job.label,
        "cluster_index": 3,
        "cluster_size": 4,
        "representative_index": 8,
        "member_index": 8,
        "clustering_method": "hier",
        "clustering_hier_t": 0.3,
        "uep_source_key": f"{job.id}:cluster:3:cluster:8",
    }
    site = MuonSite(
        label="UEP Cluster 3",
        generation_method="uep_cluster",
        initial_fractional_position=[0.5, 0.0, 0.75],
        initial_cartesian_position=[2.3475, 0.0, 2.385],
        properties=dict(recorded),
    )
    document.sites[site.id] = site
    service.save_project(document)
    _saved_at_schema_7(document, {str(job.id): _old_state(_noisy_results())})

    loaded = service.load_project(document.project_root)

    assert loaded.sites[site.id].properties == recorded
    assert loaded.sites[site.id].initial_fractional_position == [0.5, 0.0, 0.75]


def test_a_search_an_older_build_saved_in_a_schema_8_project_replaces_the_stored_one(
    tmp_path: Path,
) -> None:
    service, document, job = _project(tmp_path)
    store_uep_batch_search(document, job, _search(_noisy_results()))
    service.save_project(document)
    (stored,) = uep_batch_searches(document)
    # An older build keeps schema 8 in the manifest and writes its search in the state.
    later = {"batch_results": _noisy_results()[:4], "batch_summary_text": "Later search"}
    root = document.project_root
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    state[UEP_ANALYSIS_STATES_KEY] = {str(job.id): later}
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")

    loaded = service.load_project(root)

    (search,) = uep_batch_searches(loaded)
    assert search.result_id == stored.result_id
    assert len(search.results) == 4
    assert loaded.ui_state[UEP_ANALYSIS_STATES_KEY][str(job.id)] == {}


def test_a_window_an_older_build_opened_and_closed_does_not_lose_the_stored_search(
    tmp_path: Path,
) -> None:
    """Measured on origin/main and v0.8.0: closing the window writes ``batch_results: []``."""

    service, document, job = _project(tmp_path)
    store_uep_batch_search(document, job, _search(_noisy_results()))
    service.save_project(document)
    (stored,) = uep_batch_searches(document)
    root = document.project_root
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    state[UEP_ANALYSIS_STATES_KEY] = {
        str(job.id): {"gw_factor": 5.0, "batch_results": [], "batch_summary_data": None}
    }
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")

    loaded = service.load_project(root)

    (search,) = uep_batch_searches(loaded)
    assert search.result_id == stored.result_id
    assert search.results == stored.results
    assert loaded.ui_state[UEP_ANALYSIS_STATES_KEY][str(job.id)] == {"gw_factor": 5.0}


def test_a_search_saved_with_no_grouping_is_grouped_on_load(tmp_path: Path) -> None:
    state = {"batch_results": _noisy_results(), "batch_summary_data": None}
    service, document, job = _old_project(tmp_path, state)

    loaded = service.load_project(document.project_root)

    summary = uep_batch_search(loaded, job.id).grouping
    assert [cluster["size"] for cluster in summary["clusters"]] == [12]


def test_a_search_that_was_never_grouped_and_cannot_be_survives_reopening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {"batch_results": _noisy_results(), "batch_summary_text": "Search log"}
    service, document, job = _old_project(tmp_path, state)

    def _fail(*_args, **_kwargs):
        raise ValueError("no symmetry today")

    monkeypatch.setattr("mufinder.services.uep_results.group_uep_minima", _fail)
    loaded = service.load_project(document.project_root)
    service.save_project(loaded)
    again = service.load_project(document.project_root)

    search = uep_batch_search(again, job.id)
    assert search is not None
    assert len(search.results) == 12
    assert search.grouping is None
    assert search.log == "Search log"


@pytest.mark.parametrize("method", ["hier", "kmeans", "legacy_cartesian_hier"])
def test_every_retired_method_is_regrouped(tmp_path: Path, method: str) -> None:
    state = _old_state(_noisy_results())
    state["batch_summary_data"]["clustering_method"] = method
    service, document, job = _old_project(tmp_path, state)

    loaded = service.load_project(document.project_root)

    assert uep_batch_search(loaded, job.id).grouping["clustering_method"] == "symmetry_distance"


def test_a_search_with_malformed_results_keeps_what_it_has(tmp_path: Path) -> None:
    state = _old_state([{"optimized_position_ang": "not a position"}])
    service, document, job = _old_project(tmp_path, state)

    loaded = service.load_project(document.project_root)

    assert uep_batch_search(loaded, job.id).grouping["clustering_method"] == "hier"


def test_a_search_whose_structure_is_gone_keeps_what_it_has(tmp_path: Path) -> None:
    document = ProjectService().create_project(tmp_path / "bare.mufinder", "Bare")
    retired = {"clustering_method": "hier", "clusters": []}
    store_search(document, uuid4(), results=_noisy_results(), grouping=retired)
    store_search(document, "not-a-job-id", results=_noisy_results(), grouping=retired)

    assert not regroup_saved_uep_searches(document)
    assert [search.grouping for search in uep_batch_searches(document)] == [retired, retired]


def test_a_project_without_searches_is_untouched(tmp_path: Path) -> None:
    document = ProjectService().create_project(tmp_path / "none.mufinder", "None")

    assert not regroup_saved_uep_searches(document)


def test_a_search_that_fails_in_any_way_does_not_stop_the_project_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, document, job = _old_project(tmp_path, _old_state(_noisy_results()))

    def _fail(*_args, **_kwargs):
        raise TypeError("not a kind of failure the grouping expects")

    monkeypatch.setattr("mufinder.services.uep_results.group_uep_minima", _fail)

    loaded = service.load_project(document.project_root)

    assert uep_batch_search(loaded, job.id).grouping["clustering_method"] == "hier"


def test_a_regrouped_search_opened_and_closed_is_no_change_of_the_users(tmp_path: Path) -> None:
    from mufinder.services.project_service import UnsavedChanges

    service, document, _job = _old_project(tmp_path, _old_state(_noisy_results()))

    loaded = service.load_project(document.project_root)
    service.settle_ui_state(loaded)

    assert service.unsaved_changes(loaded) == UnsavedChanges.TIDYING


def test_an_analysis_state_is_kept_in_json_form_beside_the_others(tmp_path: Path) -> None:
    document = ProjectService().create_project(tmp_path / "state.mufinder", "State")
    job_id, other_id = uuid4(), uuid4()
    document.ui_state[UEP_ANALYSIS_STATES_KEY] = {str(other_id): {"current_tab": 1}}

    assert uep_analysis_state(document, job_id) == {}
    assert store_uep_analysis_state(document, job_id, {"current_tab": 2, "anchor": (0.5, 0.5)})

    assert document.ui_state[UEP_ANALYSIS_STATES_KEY] == {
        str(other_id): {"current_tab": 1},
        str(job_id): {"current_tab": 2, "anchor": [0.5, 0.5]},
    }
    assert uep_analysis_state(document, job_id) == {"current_tab": 2, "anchor": [0.5, 0.5]}


def test_a_malformed_analysis_state_store_reads_as_empty_and_is_replaced(tmp_path: Path) -> None:
    document = ProjectService().create_project(tmp_path / "bad.mufinder", "Bad")
    job_id = uuid4()

    document.ui_state[UEP_ANALYSIS_STATES_KEY] = {str(job_id): "not a state"}
    assert uep_analysis_state(document, job_id) == {}
    document.ui_state[UEP_ANALYSIS_STATES_KEY] = ["not", "a", "mapping"]
    assert uep_analysis_state(document, job_id) == {}

    assert store_uep_analysis_state(document, job_id, {"current_tab": 0})
    assert document.ui_state[UEP_ANALYSIS_STATES_KEY] == {str(job_id): {"current_tab": 0}}

    # A state that is not a mapping is not kept.
    assert not store_uep_analysis_state(document, job_id, ["not", "a", "state"])  # type: ignore[arg-type]
    assert document.ui_state[UEP_ANALYSIS_STATES_KEY] == {str(job_id): {"current_tab": 0}}
