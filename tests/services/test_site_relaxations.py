"""Recording a completed muon relaxation on its source site, per charge state."""

from __future__ import annotations

from pathlib import Path

from mufinder.domain.charge_states import (
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    muon_relaxation_calc_type,
)
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.site_relaxations import record_site_relaxation


def _relaxation(doc: ProjectDocument, charge_state: str) -> tuple[CalculationRun, Job]:
    job = Job(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=muon_relaxation_calc_type(charge_state),
        status=JobStatus.COMPLETED,
    )
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    doc.jobs[job.id] = job
    doc.runs[run.id] = run
    return run, job


def test_the_site_link_keeps_the_first_charge_state_and_follows_its_reruns(tmp_path: Path) -> None:
    doc = ProjectService().create_project(tmp_path / "p.mufinder", "P")
    site = MuonSite(label="O1", status="staged")
    doc.sites[site.id] = site

    positive_run, positive_job = _relaxation(doc, POSITIVE_MUON_CHARGE_STATE)
    assert record_site_relaxation(
        doc,
        site,
        positive_run,
        positive_job,
        total_energy_eV=-10.0,
        fractional_position=(0.1, 0.2, 0.3),
    )
    assert (site.status, site.relaxation_run_id, site.total_energy_eV) == (
        "relaxed",
        positive_run.id,
        -10.0,
    )

    # A muonium relaxation of the same start finishing later leaves the link alone.
    muonium_run, muonium_job = _relaxation(doc, MUONIUM_CHARGE_STATE)
    assert not record_site_relaxation(
        doc,
        site,
        muonium_run,
        muonium_job,
        total_energy_eV=-12.0,
        fractional_position=(0.4, 0.5, 0.6),
    )
    assert (site.relaxation_run_id, site.total_energy_eV) == (positive_run.id, -10.0)
    assert site.relaxed_fractional_position == [0.1, 0.2, 0.3]

    # A continued positive-muon relaxation advances it.
    rerun = CalculationRun(job_id=positive_job.id, run_index=2, status=JobStatus.COMPLETED)
    doc.runs[rerun.id] = rerun
    assert record_site_relaxation(
        doc, site, rerun, positive_job, total_energy_eV=-10.5, fractional_position=None
    )
    assert (site.relaxation_run_id, site.total_energy_eV) == (rerun.id, -10.5)


def test_the_site_link_moves_once_the_linked_relaxation_is_gone(tmp_path: Path) -> None:
    doc = ProjectService().create_project(tmp_path / "p.mufinder", "P")
    positive_run, positive_job = _relaxation(doc, POSITIVE_MUON_CHARGE_STATE)
    site = MuonSite(label="O1", status="relaxed", relaxation_run_id=positive_run.id)
    doc.sites[site.id] = site
    del doc.runs[positive_run.id]

    muonium_run, muonium_job = _relaxation(doc, MUONIUM_CHARGE_STATE)
    assert record_site_relaxation(
        doc, site, muonium_run, muonium_job, total_energy_eV=-12.0, fractional_position=None
    )
    assert site.relaxation_run_id == muonium_run.id


def test_a_reset_relaxation_keeps_its_charge_state_as_the_site_key_owner(tmp_path: Path) -> None:
    from mufinder.services.source_candidates import (
        SITE_KEY_CHARGE_STATE_PROPERTY,
        quantum_source_key,
    )

    doc = ProjectService().create_project(tmp_path / "p.mufinder", "P")
    site = MuonSite(label="O1", status="staged")
    doc.sites[site.id] = site
    positive_run, positive_job = _relaxation(doc, POSITIVE_MUON_CHARGE_STATE)
    record_site_relaxation(
        doc, site, positive_run, positive_job, total_energy_eV=-10.0, fractional_position=None
    )
    assert site.properties[SITE_KEY_CHARGE_STATE_PROPERTY] == POSITIVE_MUON_CHARGE_STATE

    # The positive-muon relaxation is reset; the muonium one then finishes.
    site.clear_relaxation()
    muonium_run, muonium_job = _relaxation(doc, MUONIUM_CHARGE_STATE)
    assert not record_site_relaxation(
        doc, site, muonium_run, muonium_job, total_energy_eV=-12.0, fractional_position=None
    )
    assert site.relaxation_run_id is None
    # The bare site-id key still names the positive muon, so nothing stored under
    # either key changes charge state.
    assert quantum_source_key(doc, site, muonium_job) == f"{site.id}:muonium"
    assert quantum_source_key(doc, site, positive_job) == str(site.id)

    rerun = CalculationRun(job_id=positive_job.id, run_index=2, status=JobStatus.COMPLETED)
    doc.runs[rerun.id] = rerun
    assert record_site_relaxation(
        doc, site, rerun, positive_job, total_energy_eV=-10.2, fractional_position=None
    )
    assert site.relaxation_run_id == rerun.id
