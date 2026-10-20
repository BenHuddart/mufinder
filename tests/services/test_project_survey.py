"""Tests for the cell-survey positions in :mod:`mufinder.services.project_survey`."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, MuonSite, Result
from mufinder.services.magnetic_field_results import MAGNETIC_FIELD_PAGE_OWNER, store_magnetic_field_results
from mufinder.services.project_service import ProjectService
from mufinder.services.project_survey import (
    SurveyMarkerKind,
    discovered_sites_layer,
    survey_layers,
)
from mufinder.workflow.energy_barriers import NUDGED_ELASTIC_BAND_CALC_TYPE, NUDGED_ELASTIC_BAND_SET_CALC_TYPE
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_PHONON_SET_CALC_TYPE
from tests.support.project_home import PEROVSKITE_POSITIONS, muon_project
from tests.support.structures import host_structure


def _positions(layer) -> list[tuple[float, float, float]]:
    return [tuple(round(value, 6) for value in marker.position) for marker in layer.markers]


def _analysis_workspace(host_id, *, workspace_id: str = "ws-1", name: str = "Mu+ 2x2x2") -> dict[str, object]:
    return {
        "id": workspace_id,
        "name": name,
        "charge_state": "positive_muon",
        "host_structure_id": str(host_id),
        "clustering": {
            "result": {
                "clusters": [
                    {
                        "cluster_index": 1,
                        "size": 3,
                        "representative_fractional_position": [0.5, 0.25, 0.25],
                        "minimum_relative_energy_ev": 0.0,
                    },
                    {
                        "cluster_index": 2,
                        "size": 1,
                        "representative_fractional_position": [1.2, 0.5, 0.75],
                        "minimum_relative_energy_ev": 0.18,
                    },
                ],
                "wyckoff_assignments": [{"cluster_index": 1, "match": {"label": "3c"}}],
            }
        },
    }


def test_nothing_is_surveyed_without_a_host(tmp_path: Path) -> None:
    document = ProjectService().create_project(tmp_path / "none.mufinder", "None")

    assert survey_layers(document, "muon_candidates") == ()
    assert discovered_sites_layer(document) is None


def test_stages_without_positions_have_no_layers(tmp_path: Path) -> None:
    project = muon_project(tmp_path)

    for stage in ("host_structure", "supercell_editor", "convergence_cutoff", "pristine_relaxation"):
        assert survey_layers(project.document, stage) == ()


def test_candidates_on_the_active_host_are_shown_and_others_counted(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    stray = MuonSite(label="old", initial_fractional_position=[0.1, 0.1, 0.1], source_structure_id=uuid4())
    project.document.sites[stray.id] = stray

    (layer,) = survey_layers(project.document, "muon_candidates")

    assert layer.title == "2 candidate sites"
    assert {marker.kind for marker in layer.markers} == {SurveyMarkerKind.CANDIDATE}
    assert sorted(_positions(layer)) == [(0.2, 0.5, 0.75), (0.5, 0.2, 0.2)]
    assert layer.note == "1 from a previous host not shown"


def test_relaxations_show_where_each_muon_started_and_ended(tmp_path: Path) -> None:
    project = muon_project(tmp_path)

    (layer,) = survey_layers(project.document, "muon_relaxations")

    assert layer.title == "Positive muon · 1 of 2 relaxed"
    relaxed = next(marker for marker in layer.markers if marker.kind == SurveyMarkerKind.RELAXED)
    assert relaxed.position == pytest.approx((0.5, 0.25, 0.25))
    assert relaxed.start == pytest.approx((0.5, 0.2, 0.2))
    pending = next(marker for marker in layer.markers if marker.kind == SurveyMarkerKind.CANDIDATE)
    assert pending.position == pytest.approx((0.2, 0.5, 0.75))
    assert pending.detail == "mu2 · running"


def test_uep_minima_come_through_cartesian_into_the_host_cell(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    document, host = project.document, project.host
    job = Job(label="uep-host", workflow_stage=WorkflowStage.UEP, calc_type="castep_uep_source", status=JobStatus.COMPLETED)
    other_job = Job(label="uep-other", workflow_stage=WorkflowStage.UEP, calc_type="castep_uep_source", status=JobStatus.COMPLETED)
    unrelated = host_structure(label="Other", lattice=6.0)
    document.structures[unrelated.id] = unrelated
    for owner, source in ((job, host.id), (other_job, unrelated.id)):
        document.jobs[owner.id] = owner
        definition = CalculationDefinition(
            workflow_stage=WorkflowStage.UEP,
            calc_type="castep_uep_source",
            input_structure_ids=[source],
            parameters={"job_ids_by_label": {owner.label: str(owner.id)}},
        )
        document.calculations[definition.id] = definition
    cluster = {"cluster_index": 1, "size": 4, "representative_position_ang": [6.0, 0.8, 0.8], "minimum_energy_ev": 0.5}
    document.ui_state["uep_analysis_states"] = {
        str(job.id): {"batch_summary_data": {"clusters": [cluster]}},
        str(other_job.id): {"batch_summary_data": {"clusters": [cluster]}},
    }

    (layer,) = survey_layers(document, "uep")

    assert layer.title == "UEP minima · uep-host"
    assert _positions(layer) == [(0.5, 0.2, 0.2)]
    assert layer.markers[0].kind == SurveyMarkerKind.UEP_MINIMUM
    assert layer.markers[0].weight == 4


def test_analysis_layers_label_sites_and_mark_the_pinned_workspace(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    document = project.document
    document.ui_state["relaxation_analysis_workspaces"] = [
        _analysis_workspace(project.host.id),
        _analysis_workspace(uuid4(), workspace_id="ws-old", name="Old host"),
    ]
    document.ui_state["muon_sites_summary_snapshots"] = {
        "positive_muon": {"workspace_id": "ws-1", "host_structure_id": str(project.host.id), "sites": []}
    }

    (layer,) = survey_layers(document, "muon_analysis")

    assert layer.title == "Mu+ 2x2x2 (pinned)"
    assert [marker.label for marker in layer.markers] == ["3c", "Site 2"]
    assert _positions(layer)[1] == (0.2, 0.5, 0.75)
    assert layer.markers[0].weight == 3


def test_discovered_sites_prefer_the_pinned_summary_then_the_most_refined_stage(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    document = project.document

    assert discovered_sites_layer(document).stage_key == "muon_relaxations"

    document.ui_state["relaxation_analysis_workspaces"] = [_analysis_workspace(project.host.id)]
    assert discovered_sites_layer(document).stage_key == "muon_analysis"

    document.ui_state["muon_sites_summary_snapshots"] = {
        "positive_muon": {
            "workspace_id": "ws-1",
            "host_structure_id": str(project.host.id),
            "sites": [{"cluster_index": 1, "fractional_position": [0.5, 0.25, 0.25], "wyckoff_label": "3c"}],
        }
    }
    pinned = discovered_sites_layer(document)
    assert pinned.title == "Pinned distinct sites"
    assert [marker.label for marker in pinned.markers] == ["3c"]


def test_barrier_paths_are_mapped_from_the_supercell_and_labelled_with_heights(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    document = project.document
    supercell = host_structure(
        label="2x2x2",
        lattice=8.0,
        species=("Sr",) * len(PEROVSKITE_POSITIONS),
        positions=PEROVSKITE_POSITIONS,
        parent_structure_id=project.host.id,
    )
    document.structures[supercell.id] = supercell
    key = f"{project.relaxed_site.id}:{project.running_site.id}:0"
    document.ui_state["energy_barriers_form"] = {
        "generated_sources": [
            {
                "key": key,
                "reactant_structure_id": str(supercell.id),
                "path_start_fractional": [0.25, 0.1, 0.1],
                "path_end_fractional": [0.1, 0.25, 0.375],
            }
        ]
    }
    job = Job(workflow_stage=WorkflowStage.ENERGY_BARRIERS, calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE, status=JobStatus.COMPLETED)
    document.jobs[job.id] = job
    result = Result(result_type=NUDGED_ELASTIC_BAND_CALC_TYPE, data={"barrier_height_eV": 0.42})
    run = CalculationRun(job_id=job.id, result_id=result.id, status=JobStatus.COMPLETED)
    result.calculation_run_id = run.id
    document.runs[run.id] = run
    document.results[result.id] = result
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=NUDGED_ELASTIC_BAND_SET_CALC_TYPE,
        parameters={"job_ids_by_label": {key: str(job.id)}},
    )
    document.calculations[definition.id] = definition

    (layer,) = survey_layers(document, "energy_barriers")

    (path,) = layer.paths
    assert path.start == pytest.approx((0.5, 0.2, 0.2))
    assert path.end == pytest.approx((0.2, 0.5, 0.75))
    assert path.label == "0.42 eV"
    assert layer.title == "1 diffusion path, 1 with barriers"
    assert len(layer.markers) == 2


def test_quantum_layer_places_zero_point_energies_at_the_relaxed_position(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    document = project.document
    job = Job(workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS, calc_type=QUANTUM_PHONON_CALC_TYPE, status=JobStatus.COMPLETED)
    document.jobs[job.id] = job
    result = Result(result_type=QUANTUM_PHONON_CALC_TYPE, data={"muon_zpe_total_eV": 0.712})
    run = CalculationRun(job_id=job.id, result_id=result.id, status=JobStatus.COMPLETED)
    result.calculation_run_id = run.id
    document.runs[run.id] = run
    document.results[result.id] = result
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_SET_CALC_TYPE,
        parameters={"phonon_job_ids_by_label": {str(project.relaxed_site.id): str(job.id)}},
    )
    document.calculations[definition.id] = definition

    (layer,) = survey_layers(document, "quantum_corrections")

    (marker,) = layer.markers
    assert marker.position == pytest.approx((0.5, 0.25, 0.25))
    assert marker.label == "ZPE 0.71 eV"


def test_magnetic_layer_shows_commensurate_field_magnitudes(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    project.document.ui_state["magnetic_field_workspaces"] = [
        {"id": "mag-1", "title": "AFM", "form": {}},
        {"id": "mag-2", "title": "Distribution", "form": {}},
    ]
    store_magnetic_field_results(
        project.document,
        "mag-1",
        {
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [{"site_label": "3c", "muon_position": [0.5, 0.25, 0.25], "magnitude_T": 0.2134}],
            }
        },
    )
    store_magnetic_field_results(
        project.document,
        "mag-2",
        {"magnetic_field_results": {"kind": "distribution", "results": [{"site_label": "3c", "fields_T": []}]}},
    )
    # The page's own results are not a saved workspace and add no layer.
    store_magnetic_field_results(
        project.document,
        MAGNETIC_FIELD_PAGE_OWNER,
        {
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [{"site_label": "3c", "muon_position": [0.5, 0.25, 0.25], "magnitude_T": 0.5}],
            }
        },
    )

    (layer,) = survey_layers(project.document, "magnetic_fields")

    assert layer.title == "AFM · local fields"
    assert layer.markers[0].label == "0.213 T"
    assert "estimate" not in layer.markers[0].detail


def test_magnetic_layer_marks_fields_at_uep_sites_as_estimates(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    entry = {"site_label": "3c", "muon_position": [0.5, 0.25, 0.25], "magnitude_T": 0.2134}
    project.document.ui_state["magnetic_field_workspaces"] = [
        {"id": "uep", "title": "UEP", "form": {}},
        # Saved before results carried their site source: the workspace's selected site supplies it.
        {"id": "legacy", "title": "Legacy", "form": {"selected_muon_site": {"source": "uep_clustering"}}},
    ]
    store_magnetic_field_results(
        project.document,
        "uep",
        {"magnetic_field_results": {"kind": "commensurate", "results": [entry], "site_source": "uep_clustering"}},
    )
    store_magnetic_field_results(
        project.document, "legacy", {"magnetic_field_results": {"kind": "commensurate", "results": [entry]}}
    )

    layers = survey_layers(project.document, "magnetic_fields")

    for layer in layers:
        assert layer.markers[0].label == "≈0.213 T"
        assert layer.markers[0].detail.endswith("unrelaxed UEP site estimate")
