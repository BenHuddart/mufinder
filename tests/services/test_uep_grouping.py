"""Tests for regrouping the UEP batch searches a saved project holds."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Job, MuonSite, Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.study_findings import record_uep_minima
from mufinder.services.uep_grouping import regroup_saved_uep_searches, uep_source_structure_id

from tests.support.uep_results import fake_optimization

_STATES_KEY = "uep_analysis_states"
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
    """The state a search and its three-cluster hierarchical split were saved with."""

    return {
        "current_tab": 1,
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


def _project(
    tmp_path: Path, state: dict[str, object]
) -> tuple[ProjectService, ProjectDocument, Job]:
    service = ProjectService()
    document = service.create_project(tmp_path / "old.mufinder", "Old UEP")
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
    document.ui_state[_STATES_KEY] = {str(job.id): state}
    service.save_project(document)
    return service, document, job


def test_loading_a_project_with_an_old_cluster_summary_regroups_it(tmp_path: Path) -> None:
    service, document, job = _project(tmp_path, _old_state(_noisy_results()))

    loaded = service.load_project(document.project_root)

    state = loaded.ui_state[_STATES_KEY][str(job.id)]
    summary = state["batch_summary_data"]
    assert summary["clustering_method"] == "symmetry_distance"
    assert summary["threshold_ang"] == 0.25
    assert [cluster["size"] for cluster in summary["clusters"]] == [12]
    assert summary["clusters"][0]["representative_fractional_position"] == pytest.approx(
        [0.0, 0.5, 0.25]
    )
    assert "clustering_hier_t" not in summary
    assert "clustering_window_state" not in state
    assert "batch_selected_keys" not in state
    assert "batch_selected_row" not in state
    assert state["current_tab"] == 1
    assert state["batch_summary_text"].splitlines() == [
        "Requested random starts: 12",
        "Grouping: 12 minima in 1 site, within 0.25 Å and 1 meV.",
    ]


def test_the_regrouped_search_reaches_disk_at_the_next_save(tmp_path: Path) -> None:
    service, document, job = _project(tmp_path, _old_state(_noisy_results()))

    loaded = service.load_project(document.project_root)
    service.save_project(loaded)

    saved = json.loads((document.project_root / "state.json").read_text(encoding="utf-8"))
    summary = saved[_STATES_KEY][str(job.id)]["batch_summary_data"]
    assert summary["clustering_method"] == "symmetry_distance"
    # A current project is left as it is.
    again = service.load_project(document.project_root)
    assert not regroup_saved_uep_searches(again)
    assert again.ui_state[_STATES_KEY][str(job.id)] == saved[_STATES_KEY][str(job.id)]


def test_regrouping_on_load_adds_nothing_to_the_journal(tmp_path: Path) -> None:
    service, document, _job = _project(tmp_path, _old_state(_noisy_results()))
    before = len(document.provenance)

    loaded = service.load_project(document.project_root)

    assert len(loaded.provenance) == before


def test_sites_created_from_the_old_clusters_keep_what_they_recorded(tmp_path: Path) -> None:
    service, document, job = _project(tmp_path, _old_state(_noisy_results()))
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

    loaded = service.load_project(document.project_root)

    assert loaded.sites[site.id].properties == recorded
    assert loaded.sites[site.id].initial_fractional_position == [0.5, 0.0, 0.75]


def test_a_search_saved_with_no_grouping_is_grouped_on_load(tmp_path: Path) -> None:
    state = {"batch_results": _noisy_results(), "batch_summary_data": None}
    service, document, job = _project(tmp_path, state)

    loaded = service.load_project(document.project_root)

    summary = loaded.ui_state[_STATES_KEY][str(job.id)]["batch_summary_data"]
    assert [cluster["size"] for cluster in summary["clusters"]] == [12]


@pytest.mark.parametrize("method", ["hier", "kmeans", "legacy_cartesian_hier"])
def test_every_retired_method_is_regrouped(tmp_path: Path, method: str) -> None:
    state = _old_state(_noisy_results())
    state["batch_summary_data"]["clustering_method"] = method
    service, document, job = _project(tmp_path, state)

    loaded = service.load_project(document.project_root)

    summary = loaded.ui_state[_STATES_KEY][str(job.id)]["batch_summary_data"]
    assert summary["clustering_method"] == "symmetry_distance"


def test_a_search_with_malformed_results_keeps_what_it_has(tmp_path: Path) -> None:
    state = _old_state([{"optimized_position_ang": "not a position"}])
    service, document, job = _project(tmp_path, state)

    loaded = service.load_project(document.project_root)

    kept = loaded.ui_state[_STATES_KEY][str(job.id)]
    assert kept["batch_summary_data"]["clustering_method"] == "hier"
    assert "clustering_window_state" not in kept


def test_a_search_whose_structure_is_gone_keeps_what_it_has(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "bare.mufinder", "Bare")
    raw_job_id = str(uuid4())
    document.ui_state[_STATES_KEY] = {
        raw_job_id: {
            "batch_results": _noisy_results(),
            "batch_summary_data": {"clustering_method": "hier", "clusters": []},
        },
        "not-a-job-id": {
            "batch_results": _noisy_results(),
            "batch_summary_data": {"clustering_method": "hier", "clusters": []},
        },
        "not-a-state": "text",
    }

    assert not regroup_saved_uep_searches(document)
    assert document.ui_state[_STATES_KEY][raw_job_id]["batch_summary_data"] == {
        "clustering_method": "hier",
        "clusters": [],
    }


def test_a_project_without_uep_states_is_untouched(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "none.mufinder", "None")

    assert not regroup_saved_uep_searches(document)

    document.ui_state[_STATES_KEY] = "not a mapping"
    assert not regroup_saved_uep_searches(document)


def test_a_job_finds_its_source_structure_through_its_definition(tmp_path: Path) -> None:
    _service, document, job = _project(tmp_path, {})

    assert uep_source_structure_id(document, job.id) == document.project.host_structure_id
    assert uep_source_structure_id(document, uuid4()) is None


def test_a_regrouped_search_is_plain_data_that_every_reader_accepts(tmp_path: Path) -> None:
    service, document, job = _project(tmp_path, _old_state(_noisy_results()))

    loaded = service.load_project(document.project_root)

    state = loaded.ui_state[_STATES_KEY][str(job.id)]
    assert isinstance(state["batch_summary_data"]["clusters"], list)
    # Turning a camera in the window it opens is not a new finding.
    assert not record_uep_minima(loaded, loaded.jobs[job.id], state, {**state, "current_tab": 2})


def test_a_search_that_fails_in_any_way_does_not_stop_the_project_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, document, job = _project(tmp_path, _old_state(_noisy_results()))

    def _fail(*_args, **_kwargs):
        raise TypeError("not a kind of failure the grouping expects")

    monkeypatch.setattr("mufinder.services.uep_grouping.group_uep_minima", _fail)

    loaded = service.load_project(document.project_root)

    assert loaded.ui_state[_STATES_KEY][str(job.id)]["batch_summary_data"]["clustering_method"] == (
        "hier"
    )
