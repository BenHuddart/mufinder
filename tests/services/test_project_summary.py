"""Tests for the workflow-stage catalogue and project summary in :mod:`mufinder.services.project_summary`."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job, Result
from mufinder.services.project_service import ProjectService
from mufinder.services.project_summary import (
    COMPLETION_CHECKS,
    PROJECT_HOME_STAGE_KEY,
    STAGE_DEPENDENCIES,
    STAGE_GROUPS,
    WORKFLOW_STAGES,
    DependencyKind,
    StageStatus,
    group_stage_keys,
    muon_relaxations_complete,
    structure_formula,
    summarize_project,
)
from mufinder.workflow.energy_barriers import NUDGED_ELASTIC_BAND_CALC_TYPE
from tests.support.project_home import muon_project, perovskite_host
from tests.support.structures import host_structure


def _empty_project(tmp_path: Path):
    return ProjectService().create_project(tmp_path / "empty.mufinder", "Empty")


def test_dependency_graph_names_known_stages_and_runs_forward() -> None:
    order = [stage.key for stage in WORKFLOW_STAGES]
    group_keys = {group.key for group in STAGE_GROUPS}

    def position(key: str) -> int:
        if key in group_keys:
            return max(order.index(member) for member in group_stage_keys(key))
        return order.index(key)

    for dependency in STAGE_DEPENDENCIES:
        assert dependency.target in order
        assert dependency.source in order or dependency.source in group_keys
        assert position(dependency.source) < position(dependency.target), dependency
    assert all(group_stage_keys(group) for group in group_keys)
    assert set(COMPLETION_CHECKS) == set(order) | group_keys


def test_downstream_stages_require_the_distinct_sites_from_analysis() -> None:
    required = {
        (dependency.source, dependency.target)
        for dependency in STAGE_DEPENDENCIES
        if dependency.kind == DependencyKind.REQUIRED
    }

    assert {("muon_analysis", target) for target in ("magnetic_fields", "quantum_corrections", "energy_barriers")} <= required
    uep_to_fields = next(d for d in STAGE_DEPENDENCIES if (d.source, d.target) == ("uep", "magnetic_fields"))
    assert uep_to_fields.kind == DependencyKind.OPTIONAL


def test_empty_project_starts_at_the_host_and_everything_else_needs_it(tmp_path: Path) -> None:
    summary = summarize_project(_empty_project(tmp_path))

    host = summary.stage("host_structure")
    assert host.status == StageStatus.NOT_STARTED
    assert host.facts == ("No host structure yet",)
    assert summary.current_stage == "host_structure"
    assert summary.completed_count == 0
    for stage in summary.stages[1:]:
        assert "Host Structure" in stage.unmet_requirements


def test_convergence_defaults_and_active_jobs_are_reported(tmp_path: Path) -> None:
    service = ProjectService()
    document = _empty_project(tmp_path)
    host = perovskite_host()
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    document.project.recommended_cutoff_energy = 700.0
    job = service.add_job(
        document,
        label="kpt",
        workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
        calc_type="castep_kpoint",
    )
    job.status = JobStatus.RUNNING

    summary = summarize_project(document)

    assert summary.stage("host_structure").facts == ("SrTiO3 · 5 atoms",)
    cutoff = summary.stage("convergence_cutoff")
    assert cutoff.status == StageStatus.COMPLETE
    assert cutoff.facts[0] == "Default cutoff 700 eV"
    kpoint = summary.stage("convergence_kpoint")
    assert kpoint.status == StageStatus.IN_PROGRESS
    assert kpoint.active_jobs == 1
    assert kpoint.facts == ("No default grid yet", "1 active")
    assert summary.active_jobs == 1
    assert summary.current_stage == "convergence_kpoint"


def test_relaxation_progress_is_counted_per_charge_state(tmp_path: Path) -> None:
    project = muon_project(tmp_path)

    summary = summarize_project(project.document)

    relaxations = summary.stage("muon_relaxations")
    assert relaxations.status == StageStatus.IN_PROGRESS
    assert relaxations.facts == ("Positive muon: 1 of 2 relaxed, 1 active",)
    assert relaxations.active_jobs == 1
    assert relaxations.unmet_requirements == ()
    assert summary.stage("muon_candidates").facts == ("2 candidate sites", "1 from UEP clusters")
    assert summary.stage("muon_analysis").unmet_requirements == ()
    assert summary.stage("quantum_corrections").unmet_requirements == ("Analysis",)
    assert summary.current_stage == "muon_relaxations"


def test_unmet_requirements_name_unstarted_required_stages_only(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    document = project.document
    document.sites.clear()
    document.jobs.clear()

    summary = summarize_project(document)

    assert summary.stage("muon_relaxations").unmet_requirements == ("Generation",)
    assert summary.stage("muon_candidates").unmet_requirements == ()
    assert summary.stage("host_structure").unmet_requirements == ()

    document.ui_state["relaxation_analysis_workspaces"] = [{"id": "w1"}]
    analysis = summarize_project(document).stage("muon_analysis")
    assert analysis.status == StageStatus.COMPLETE
    assert analysis.unmet_requirements == ()


def test_relaxations_complete_once_every_row_converged(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    assert muon_relaxations_complete(project.document) is False

    project.running_job.status = JobStatus.COMPLETED

    assert muon_relaxations_complete(project.document) is True
    assert summarize_project(project.document).stage("muon_relaxations").status == StageStatus.COMPLETE


def test_saved_stage_is_shown_first_and_an_unknown_one_is_ignored(tmp_path: Path) -> None:
    project = muon_project(tmp_path)

    project.document.ui_state[PROJECT_HOME_STAGE_KEY] = "uep"
    assert summarize_project(project.document).current_stage == "uep"

    project.document.ui_state[PROJECT_HOME_STAGE_KEY] = "provenance_log"
    assert summarize_project(project.document).current_stage == "muon_relaxations"


def test_barrier_facts_count_neb_results(tmp_path: Path) -> None:
    document = _empty_project(tmp_path)
    job = Job(workflow_stage=WorkflowStage.ENERGY_BARRIERS, calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE, status=JobStatus.COMPLETED)
    document.jobs[job.id] = job
    for height in (0.42, 0.18):
        result = Result(result_type=NUDGED_ELASTIC_BAND_CALC_TYPE, data={"barrier_height_eV": height})
        document.results[result.id] = result

    barriers = summarize_project(document).stage("energy_barriers")

    assert barriers.facts[0] == "2 barriers, lowest 0.18 eV"


def test_magnetic_workspace_without_components_is_in_progress(tmp_path: Path) -> None:
    document = _empty_project(tmp_path)
    document.ui_state["magnetic_field_workspaces"] = [{"id": str(uuid4()), "title": "AFM"}]

    magnetic = summarize_project(document).stage("magnetic_fields")

    assert magnetic.status == StageStatus.IN_PROGRESS
    assert magnetic.facts == ("1 saved workspace",)


def test_structure_formula_reduces_counts_and_skips_the_muon() -> None:
    assert structure_formula(perovskite_host()) == "SrTiO3"
    hematite = host_structure(
        species=("Fe",) * 4 + ("O",) * 6 + ("H:mu",),
        positions=[(index / 11.0, 0.0, 0.0) for index in range(11)],
    )
    assert structure_formula(hematite) == "Fe2O3"
