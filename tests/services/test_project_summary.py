"""Tests for the workflow-stage catalogue and project summary in :mod:`mufinder.services.project_summary`."""

from __future__ import annotations

from datetime import timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.enums import JobActivity, JobStatus, WorkflowStage
from mufinder.domain.models import Job, MuonSite, ProvenanceRecord, Result, utc_now
from mufinder.services.project_service import ProjectService
from mufinder.services.project_summary import (
    COMPLETION_CHECKS,
    OTHER_RESULT_STAGES,
    STAGE_GROUPS,
    STAGE_PHASES,
    WORKFLOW_STAGES,
    LastStage,
    ProjectSummary,
    StageStatus,
    StageSummary,
    energy_barriers_complete,
    group_stage_keys,
    group_status,
    job_stage_key,
    job_stage_name,
    job_stage_title,
    muon_relaxations_complete,
    stage_info,
    stage_phase,
    stage_statuses,
    structure_formula,
    summarize_project,
)
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
)
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE
from tests.support.project_home import muon_project, perovskite_host
from tests.support.structures import host_structure


def _empty_project(tmp_path: Path):
    return ProjectService().create_project(tmp_path / "empty.mufinder", "Empty")


def test_completion_checks_cover_every_stage_and_group() -> None:
    order = [stage.key for stage in WORKFLOW_STAGES]
    group_keys = {group.key for group in STAGE_GROUPS}

    assert all(group_stage_keys(group) for group in group_keys)
    assert set(COMPLETION_CHECKS) == set(order) | group_keys
    assert set(OTHER_RESULT_STAGES) <= set(order)


def test_phases_hold_every_stage_once_in_navigator_order() -> None:
    expanded = [
        stage_key
        for phase in STAGE_PHASES
        for member in phase.members
        for stage_key in (group_stage_keys(member) or (member,))
    ]
    top_level = list(
        dict.fromkeys(stage.group or stage.key for stage in WORKFLOW_STAGES),
    )

    assert expanded == [stage.key for stage in WORKFLOW_STAGES]
    assert [member for phase in STAGE_PHASES for member in phase.members] == top_level
    assert [(phase.key, phase.title) for phase in STAGE_PHASES] == [
        ("prepare", "Prepare"),
        ("find_sites", "Find sites"),
        ("interpret", "Interpret"),
    ]
    assert stage_phase("find_sites").members == ("uep", "muon_sites")
    with pytest.raises(KeyError):
        stage_phase("analysis")


def test_status_map_has_every_stage_and_group_and_matches_the_summary(tmp_path: Path) -> None:
    summary = summarize_project(muon_project(tmp_path).document)

    statuses = stage_statuses(summary)

    assert set(statuses) == {stage.key for stage in WORKFLOW_STAGES} | {
        group.key for group in STAGE_GROUPS
    }
    assert all(statuses[stage.key] == stage.status for stage in summary.stages)


def test_groups_start_not_started_in_an_empty_project(tmp_path: Path) -> None:
    summary = summarize_project(_empty_project(tmp_path))

    assert {group.key: group_status(summary, group.key) for group in STAGE_GROUPS} == {
        "convergence": StageStatus.NOT_STARTED,
        "muon_sites": StageStatus.NOT_STARTED,
    }
    with pytest.raises(KeyError):
        group_status(summary, "host_structure")


def test_convergence_group_is_in_progress_until_both_defaults_are_set(tmp_path: Path) -> None:
    document = _empty_project(tmp_path)
    document.project.recommended_cutoff_energy = 700.0

    assert group_status(summarize_project(document), "convergence") == StageStatus.IN_PROGRESS

    document.project.recommended_kpoint_grid = (4, 4, 4)

    assert group_status(summarize_project(document), "convergence") == StageStatus.COMPLETE


def test_group_status_is_recomputed_after_reopening_the_project(tmp_path: Path) -> None:
    service = ProjectService()
    document = _empty_project(tmp_path)
    document.project.recommended_cutoff_energy = 700.0
    document.project.recommended_kpoint_grid = (4, 4, 4)
    service.save_project(document)

    reopened = service.load_project(document.project_root)

    assert stage_statuses(summarize_project(reopened))["convergence"] == StageStatus.COMPLETE


def test_muon_sites_group_is_in_progress_once_a_stage_has_a_job(tmp_path: Path) -> None:
    document = _empty_project(tmp_path)
    ProjectService().add_job(
        document,
        label="mu1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="muon_relaxation",
    )

    summary = summarize_project(document)

    assert summary.stage("muon_candidates").status == StageStatus.NOT_STARTED
    assert group_status(summary, "muon_sites") == StageStatus.IN_PROGRESS


def test_muon_sites_group_is_under_way_while_relaxations_run(tmp_path: Path) -> None:
    document = muon_project(tmp_path).document
    summary = summarize_project(document)

    assert summary.stage("muon_candidates").status == StageStatus.COMPLETE
    assert summary.stage("muon_relaxations").status == StageStatus.IN_PROGRESS
    assert stage_statuses(summary)["muon_sites"] == StageStatus.IN_PROGRESS
    assert COMPLETION_CHECKS["muon_sites"](document) is False


def _summary_with(statuses: dict[str, StageStatus]) -> ProjectSummary:
    stages = tuple(
        StageSummary(info.key, info.title, statuses.get(info.key, StageStatus.NOT_STARTED), ())
        for info in WORKFLOW_STAGES
    )
    last = LastStage("host_structure", "Host Structure", None, "")
    return ProjectSummary(stages, "", last, "", "")


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        ((StageStatus.NOT_STARTED,) * 3, StageStatus.NOT_STARTED),
        (
            (StageStatus.COMPLETE, StageStatus.NOT_STARTED, StageStatus.NOT_STARTED),
            StageStatus.IN_PROGRESS,
        ),
        (
            (StageStatus.COMPLETE, StageStatus.COMPLETE, StageStatus.IN_PROGRESS),
            StageStatus.IN_PROGRESS,
        ),
        ((StageStatus.COMPLETE,) * 3, StageStatus.COMPLETE),
    ],
)
def test_a_group_is_complete_only_when_every_stage_in_it_is(
    members: tuple[StageStatus, ...], expected: StageStatus
) -> None:
    keys = group_stage_keys("muon_sites")
    summary = _summary_with(dict(zip(keys, members, strict=True)))

    assert group_status(summary, "muon_sites") == expected


def test_a_group_check_passes_only_when_every_stage_check_does(tmp_path: Path) -> None:
    document = _empty_project(tmp_path)
    document.project.recommended_cutoff_energy = 700.0

    assert COMPLETION_CHECKS["convergence_cutoff"](document)
    assert not COMPLETION_CHECKS["convergence"](document)

    document.project.recommended_kpoint_grid = (4, 4, 4)

    assert COMPLETION_CHECKS["convergence"](document)


def test_empty_project_left_off_at_the_host_with_no_date(tmp_path: Path) -> None:
    summary = summarize_project(_empty_project(tmp_path))

    host = summary.stage("host_structure")
    assert host.status == StageStatus.NOT_STARTED
    assert host.facts == ("No host structure yet",)
    assert summary.last_stage == LastStage(
        "host_structure", "Host Structure", None, "No host structure yet"
    )
    assert summary.host_line == "No host structure yet"
    assert summary.site_progress == "No candidate muon sites yet."
    assert summary.other_results == ""


def test_with_no_activity_the_first_required_stage_still_to_do_is_shown(tmp_path: Path) -> None:
    service = ProjectService()
    document = _empty_project(tmp_path)
    host = perovskite_host()
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    last = summarize_project(document).last_stage

    assert (last.key, last.timestamp) == ("convergence_cutoff", None)


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
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING

    summary = summarize_project(document)

    assert summary.stage("host_structure").facts == ("SrTiO3 · 5 atoms", "No saved supercells")
    cutoff = summary.stage("convergence_cutoff")
    assert cutoff.status == StageStatus.COMPLETE
    assert cutoff.facts[0] == "Default cutoff 700 eV"
    kpoint = summary.stage("convergence_kpoint")
    assert kpoint.status == StageStatus.IN_PROGRESS
    assert kpoint.facts == ("No default grid yet", "1 active")
    assert kpoint.fact_line == "No default grid yet; 1 active"
    assert summary.last_stage == LastStage(
        "convergence_kpoint", "K-Point Grid", job.created_at, "No default grid yet; 1 active"
    )


def test_host_line_names_the_space_group_and_the_adopted_defaults(tmp_path: Path) -> None:
    service = ProjectService()
    document = _empty_project(tmp_path)
    host = perovskite_host()
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    assert summarize_project(document).host_line == "SrTiO3 · Pm-3m · 5 atoms · not converged yet"

    document.project.recommended_cutoff_energy = 700.0
    assert summarize_project(document).host_line == (
        "SrTiO3 · Pm-3m · 5 atoms · cutoff 700 eV, no default k-point grid yet"
    )

    document.project.recommended_kpoint_grid = (4, 4, 4)
    assert summarize_project(document).host_line == (
        "SrTiO3 · Pm-3m · 5 atoms · cutoff 700 eV, k-point grid 4×4×4"
    )


def test_relaxation_progress_is_counted_per_charge_state(tmp_path: Path) -> None:
    project = muon_project(tmp_path)

    summary = summarize_project(project.document)

    relaxations = summary.stage("muon_relaxations")
    assert relaxations.status == StageStatus.IN_PROGRESS
    assert relaxations.facts == ("Positive muon: 1 of 2 relaxed, 1 active",)
    assert summary.stage("muon_candidates").facts == ("2 candidate sites", "1 from UEP clusters")
    assert summary.site_progress == (
        "2 candidate sites, 1 of 2 relaxed. No sites pinned to the Muon Sites summary yet."
    )


def test_last_stage_is_the_stage_of_the_newest_job(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    project.relaxed_job.completed_at = utc_now() + timedelta(minutes=5)

    last = summarize_project(project.document).last_stage

    assert last.key == "muon_relaxations"
    assert last.timestamp == project.relaxed_job.completed_at
    assert last.fact_line == "Positive muon: 1 of 2 relaxed, 1 active"


def test_a_newer_journal_record_moves_the_last_stage(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    later = utc_now() + timedelta(hours=1)
    project.document.provenance.append(
        ProvenanceRecord(timestamp=later, action="configure_uep", workflow_stage=WorkflowStage.UEP)
    )
    # A record of no workflow, such as a host import, is no stage's activity.
    project.document.provenance.append(
        ProvenanceRecord(timestamp=later + timedelta(hours=1), action="import_structure")
    )

    last = summarize_project(project.document).last_stage

    assert (last.key, last.title, last.timestamp) == ("uep", "UEP", later)


def test_a_polarisation_simulation_leaves_off_at_polarisation(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    project.document.provenance.append(
        ProvenanceRecord(
            timestamp=utc_now() + timedelta(hours=1),
            action="polarisation_simulated",
            workflow_stage=WorkflowStage.MAGNETIC_FIELD,
        )
    )

    assert summarize_project(project.document).last_stage.key == "polarisation"


def test_site_progress_leaves_out_relaxations_of_a_previous_host(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    document = project.document
    new_host = perovskite_host()
    ProjectService().add_structure(document, new_host)
    document.project.host_structure_id = new_host.id
    site = MuonSite(
        label="new",
        initial_fractional_position=[0.1, 0.1, 0.1],
        source_structure_id=new_host.id,
    )
    document.sites[site.id] = site

    assert summarize_project(document).site_progress == (
        "1 candidate site. No sites pinned to the Muon Sites summary yet."
    )


def test_a_naive_time_from_an_older_project_counts_as_utc(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    naive = (utc_now() + timedelta(hours=1)).replace(tzinfo=None)
    project.document.provenance.append(
        ProvenanceRecord(timestamp=naive, workflow_stage=WorkflowStage.SITE_ANALYSIS)
    )

    last = summarize_project(project.document).last_stage

    assert last.key == "muon_analysis"
    assert last.timestamp == naive.replace(tzinfo=timezone.utc)


def test_other_results_join_the_later_workflows_that_have_started(tmp_path: Path) -> None:
    document = _empty_project(tmp_path)
    assert summarize_project(document).other_results == ""

    barrier = Result(result_type=NUDGED_ELASTIC_BAND_CALC_TYPE, data={"barrier_height_eV": 0.3})
    phonon = Result(result_type=QUANTUM_PHONON_CALC_TYPE, data={})
    document.results[barrier.id] = barrier
    document.results[phonon.id] = phonon
    document.ui_state["magnetic_field_workspaces"] = [{"id": str(uuid4()), "title": "AFM"}]

    assert summarize_project(document).other_results == (
        "Magnetic Fields: 1 saved workspace · Quantum Corrections: 1 phonon result"
        " · Energy Barriers: 1 barrier, lowest 0.30 eV"
    )


def test_other_results_leave_out_a_workflow_with_nothing_to_show(tmp_path: Path) -> None:
    document = _empty_project(tmp_path)
    ProjectService().add_job(
        document,
        label="Path 1",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
    )

    summary = summarize_project(document)

    assert summary.stage("energy_barriers").status == StageStatus.IN_PROGRESS
    assert summary.other_results == ""


def test_relaxations_complete_once_every_row_converged(tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    assert muon_relaxations_complete(project.document) is False

    project.running_job.status = JobStatus.COMPLETED

    assert muon_relaxations_complete(project.document) is True
    assert (
        summarize_project(project.document).stage("muon_relaxations").status == StageStatus.COMPLETE
    )
    assert summarize_project(project.document).site_progress.startswith(
        "2 candidate sites, 2 of 2 relaxed."
    )


def test_a_retired_saved_home_stage_is_ignored(tmp_path: Path) -> None:
    project = muon_project(tmp_path)

    project.document.ui_state["project_home_stage"] = "uep"

    assert summarize_project(project.document).last_stage.key == "muon_relaxations"


def test_host_facts_report_the_active_hosts_saved_supercells(tmp_path: Path) -> None:
    service = ProjectService()
    document = _empty_project(tmp_path)
    host = perovskite_host()
    other_host = perovskite_host()
    service.add_structure(document, host)
    service.add_structure(document, other_host)
    document.project.host_structure_id = host.id

    def add_supercell(structure, scale: int) -> None:
        service.create_workflow_supercell_asset(
            document,
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            host_structure_id=structure.id,
            matrix=[[scale, 0, 0], [0, 1, 0], [0, 0, 1]],
            reuse_existing=False,
        )

    add_supercell(other_host, 2)
    assert summarize_project(document).stage("host_structure").facts[-1] == "No saved supercells"

    add_supercell(host, 2)
    facts = summarize_project(document).stage("host_structure").facts
    assert facts[-1].startswith("1 saved supercell: ")

    for scale in (3, 4, 5):
        add_supercell(host, scale)
    facts = summarize_project(document).stage("host_structure").facts
    assert facts[-1].startswith("4 saved supercells: ")
    assert facts[-1].endswith(" and 1 more")


def test_barrier_facts_count_neb_results(tmp_path: Path) -> None:
    document = _empty_project(tmp_path)
    job = Job(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
        status=JobStatus.COMPLETED,
    )
    document.jobs[job.id] = job
    for height in (0.42, 0.18):
        result = Result(
            result_type=NUDGED_ELASTIC_BAND_CALC_TYPE, data={"barrier_height_eV": height}
        )
        document.results[result.id] = result

    barriers = summarize_project(document).stage("energy_barriers")

    assert barriers.facts[0] == "2 barriers, lowest 0.18 eV"


def test_energy_barriers_complete_once_an_neb_result_exists(tmp_path: Path) -> None:
    document = _empty_project(tmp_path)
    assert energy_barriers_complete(document) is False
    assert summarize_project(document).stage("energy_barriers").facts == (
        "No barrier calculations yet",
    )

    result = Result(result_type=NUDGED_ELASTIC_BAND_CALC_TYPE, data={"barrier_height_eV": 0.3})
    document.results[result.id] = result

    assert energy_barriers_complete(document) is True
    assert summarize_project(document).stage("energy_barriers").status == StageStatus.COMPLETE


def test_energy_barriers_still_count_an_older_projects_search_result(tmp_path: Path) -> None:
    document = _empty_project(tmp_path)
    result = Result(result_type=TRANSITION_STATE_SEARCH_CALC_TYPE, data={})
    document.results[result.id] = result

    assert energy_barriers_complete(document) is True


def test_energy_barriers_goal_names_neb() -> None:
    goal = next(info.completion_goal for info in WORKFLOW_STAGES if info.key == "energy_barriers")

    assert goal == "Complete at least one NEB barrier calculation."


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


def test_polarisation_needs_fields_and_completes_with_a_simulation(tmp_path: Path) -> None:
    from mufinder.services.magnetic_field_results import (
        MUON_POLARISATION_RESULT_TYPE,
        SIMULATION_OWNER_PREFIX,
    )
    from mufinder.services.project_summary import polarisation_complete

    document = _empty_project(tmp_path)
    assert polarisation_complete(document) is False
    assert summarize_project(document).stage("polarisation").facts == (
        "No polarisation simulations yet",
    )

    # A bare polarisation result owned by a field page is not a simulation.
    bare = Result(result_type=MUON_POLARISATION_RESULT_TYPE, data={"owner": "magnetic_fields"})
    document.results[bare.id] = bare
    assert polarisation_complete(document) is False

    for name in ("a", "b"):
        result = Result(
            result_type=MUON_POLARISATION_RESULT_TYPE,
            data={"owner": f"{SIMULATION_OWNER_PREFIX}{name}"},
        )
        document.results[result.id] = result

    assert polarisation_complete(document) is True
    polarisation = summarize_project(document).stage("polarisation")
    assert polarisation.status == StageStatus.COMPLETE
    assert polarisation.facts == ("2 simulations",)


def test_job_stage_title_names_the_page_a_batch_of_jobs_came_from() -> None:
    assert job_stage_title([WorkflowStage.CONVERGENCE_CUTOFF] * 3) == "Cutoff Energy"
    assert job_stage_title([WorkflowStage.QUANTUM_CORRECTIONS]) == "Quantum Corrections"
    # "Relaxations" names the page only under Muon Sites.
    assert job_stage_title([WorkflowStage.MUON_RELAXATION]) == "Muon Relaxation"


def test_job_stage_title_is_none_for_a_mixed_empty_or_jobless_batch() -> None:
    assert job_stage_title([WorkflowStage.CONVERGENCE_CUTOFF, WorkflowStage.UEP]) is None
    assert job_stage_title([]) is None
    assert job_stage_title([WorkflowStage.SITE_ANALYSIS]) is None


def test_every_job_stage_has_a_navigator_page_and_name() -> None:
    for stage in WorkflowStage:
        assert stage_info(job_stage_key(stage)).title == job_stage_name(stage)

    assert job_stage_name(WorkflowStage.CONVERGENCE_CUTOFF) == "Cutoff Energy"
    assert job_stage_name(WorkflowStage.MUON_RELAXATION) == "Relaxations"
    assert job_stage_name(WorkflowStage.QUANTUM_CORRECTIONS) == "Quantum Corrections"
