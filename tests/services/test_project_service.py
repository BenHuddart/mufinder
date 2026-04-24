"""Tests for project lifecycle, integrity, and generation orchestration."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from ase import Atoms
from ase.io import write as ase_write

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, MuonSite, Result, Structure
from mufinder.castep_io.types import ValueWithUnit
from mufinder.science.site_generation import SiteGenerationError
from mufinder.services.project_service import ProjectService


def _host_structure() -> Structure:
    return Structure(
        label="Host",
        origin="imported",
        lattice_vectors=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        species=["O", "Si", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )


def test_create_save_and_load_roundtrip(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "roundtrip.mufinder"

    doc = service.create_project(root, "Roundtrip")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.ui_state["muon_generation_form"] = {"method": "random", "random": {"count": 5}}

    generated = service.generate_candidate_sites(
        doc,
        method="random",
        parameters={
            "count": 5,
            "min_muon_muon": 0.8,
            "min_muon_atom": 0.8,
            "max_attempts": 30000,
            "seed": "13",
        },
    )
    assert len(generated) == 5
    assert {site.status for site in generated} == {"generated"}

    service.save_project(doc)

    loaded = service.load_project(root)
    assert loaded.project.name == "Roundtrip"
    assert len(loaded.structures) == 1
    assert len(loaded.sites) == 5
    assert len(loaded.provenance) >= 1
    assert loaded.ui_state["muon_generation_form"]["method"] == "random"


def test_manual_generation_cartesian_conversion(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "manual.mufinder", "Manual")
    host = _host_structure()
    service.add_structure(doc, host)

    sites = service.generate_candidate_sites(
        doc,
        method="manual",
        parameters={"coord_system": "cartesian", "x": 2.0, "y": 3.0, "z": 4.0},
    )
    site = sites[0]
    assert site.status == "generated"
    assert site.fractional_position == pytest.approx([0.2, 0.3, 0.4])
    assert site.cartesian_position == pytest.approx([2.0, 3.0, 4.0])


def test_random_generation_by_target_count_estimates_spacing(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "random_count.mufinder", "RandomCount")
    host = _host_structure()
    service.add_structure(doc, host)

    sites = service.generate_candidate_sites(
        doc,
        method="random",
        parameters={
            "selection_mode": "count",
            "count": 3,
            "min_muon_atom": 0.8,
            "max_attempts": 50000,
            "seed": "7",
            "include_symmetry_equivalents": False,
        },
    )

    assert len(sites) == 3
    assert sites[0].properties["selection_mode"] == "count"
    assert sites[0].properties["target_count"] == 3
    assert sites[0].properties["estimated_min_muon_muon"] > 0.0
    assert sites[0].properties["min_muon_muon"] > 0.0


def test_candidate_numbering_is_scoped_to_label_prefix(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "prefix_numbering.mufinder", "PrefixNumbering")
    host = _host_structure()
    service.add_structure(doc, host)
    for index in range(1, 5):
        doc.sites[uuid4()] = MuonSite(
            label=f"UEP Cluster {index}",
            status="generated",
            generation_method="uep_cluster",
        )

    random_sites = service.generate_candidate_sites(
        doc,
        method="random",
        parameters={
            "selection_mode": "count",
            "count": 2,
            "min_muon_atom": 0.8,
            "max_attempts": 50000,
            "seed": "7",
            "include_symmetry_equivalents": False,
        },
    )

    assert [site.label for site in random_sites] == ["Random Site 1", "Random Site 2"]

    more_random_sites = service.generate_candidate_sites(
        doc,
        method="random",
        parameters={
            "selection_mode": "count",
            "count": 1,
            "min_muon_atom": 0.8,
            "max_attempts": 50000,
            "seed": "11",
            "include_symmetry_equivalents": False,
        },
    )

    assert [site.label for site in more_random_sites] == ["Random Site 3"]


def test_random_generation_reports_project_service_progress(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "random_progress.mufinder", "RandomProgress")
    service.add_structure(doc, _host_structure())
    updates: list[tuple[int, int]] = []

    sites = service.generate_candidate_sites(
        doc,
        method="random",
        parameters={
            "selection_mode": "count",
            "count": 2,
            "min_muon_atom": 0.8,
            "max_attempts": 50000,
            "seed": "7",
            "include_symmetry_equivalents": False,
        },
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert len(sites) == 2
    assert updates[0] == (0, 2)
    assert updates[-1] == (2, 2)


def test_random_generation_by_separation_fills_until_attempt_limit(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "random_spacing.mufinder", "RandomSpacing")
    host = _host_structure()
    service.add_structure(doc, host)

    sites = service.generate_candidate_sites(
        doc,
        method="random",
        parameters={
            "selection_mode": "separation",
            "min_muon_muon": 2.0,
            "min_muon_atom": 0.8,
            "max_attempts": 1000,
            "seed": "11",
            "include_symmetry_equivalents": False,
        },
    )

    assert len(sites) > 0
    assert all(site.properties["selection_mode"] == "separation" for site in sites)
    assert all(site.properties["min_muon_muon"] == pytest.approx(2.0) for site in sites)


def test_attached_element_generation_uses_muons_per_atom(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "attached_element.mufinder", "AttachedElement")
    host = _host_structure()
    service.add_structure(doc, host)

    sites = service.generate_candidate_sites(
        doc,
        method="near_element",
        parameters={
            "element": "O",
            "muons_per_atom": 2,
            "bond_distance": 1.0,
            "min_muon_atom": 0.6,
        },
    )

    assert len(sites) == 4
    assert all(site.generation_method == "near_element" for site in sites)
    assert all(site.properties["strategy"] == "attached" for site in sites)
    assert all(site.properties["muons_per_atom"] == 2 for site in sites)
    assert all("min_muon_muon" not in site.properties for site in sites)


def test_uep_cluster_generation_creates_sites_and_rejects_duplicates(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_cluster_generation.mufinder", "UEPClusterGeneration")
    host = _host_structure()
    service.add_structure(doc, host)

    selection = {
        "source_key": "job-1:cluster:1:cluster:0",
        "label": "UEP Cluster 1",
        "generation_method": "uep_cluster",
        "source_structure_id": str(host.id),
        "fractional_position": [0.1, 0.2, 0.3],
        "cartesian_position": [1.0, 2.0, 3.0],
        "properties": {
            "source_job_label": "uep_row_1",
            "source_job_id": "job-1",
            "cluster_index": 1,
            "cluster_size": 2,
            "representative_index": 0,
            "member_index": 0,
            "selection_kind": "cluster",
            "classification": "minimum",
            "classical_energy_ev": -0.7,
            "zero_point_energy_ev": 0.2,
            "total_energy_ev": -0.5,
            "clustering_method": "hier",
        },
    }

    sites = service.generate_candidate_sites(doc, method="uep_clusters", parameters={"selections": [selection]})

    assert len(sites) == 1
    site = sites[0]
    assert site.generation_method == "uep_cluster"
    assert site.source_structure_id == host.id
    assert site.fractional_position == pytest.approx([0.1, 0.2, 0.3])
    assert site.properties["uep_source_key"] == "job-1:cluster:1:cluster:0"
    assert site.properties["classification"] == "minimum"

    with pytest.raises(SiteGenerationError, match="already present"):
        service.generate_candidate_sites(doc, method="uep_clusters", parameters={"selections": [selection]})


def test_uep_cluster_generation_creates_one_site_per_selection(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_cluster_multi_generation.mufinder", "UEPClusterMultiGeneration")
    host = _host_structure()
    service.add_structure(doc, host)

    selections = [
        {
            "source_key": f"job-1:cluster:{index}:cluster:{index - 1}",
            "label": f"UEP Cluster {index}",
            "generation_method": "uep_cluster",
            "source_structure_id": str(host.id),
            "fractional_position": [0.1 * index, 0.2, 0.3],
            "cartesian_position": [float(index), 2.0, 3.0],
            "properties": {
                "source_job_label": "uep_row_1",
                "source_job_id": "job-1",
                "cluster_index": index,
                "cluster_size": index + 1,
                "representative_index": index - 1,
                "member_index": index - 1,
                "selection_kind": "cluster",
                "classification": "minimum",
                "classical_energy_ev": -0.7 - index,
                "clustering_method": "hier",
            },
        }
        for index in (1, 2, 3)
    ]

    sites = service.generate_candidate_sites(doc, method="uep_clusters", parameters={"selections": selections})

    assert len(sites) == 3
    assert len(doc.sites) == 3
    assert [site.properties["uep_source_key"] for site in sites] == [
        "job-1:cluster:1:cluster:0",
        "job-1:cluster:2:cluster:1",
        "job-1:cluster:3:cluster:2",
    ]
    assert [site.label for site in sites] == ["UEP Cluster 1", "UEP Cluster 2", "UEP Cluster 3"]
    assert [site.fractional_position[0] for site in sites] == pytest.approx([0.1, 0.2, 0.3])


def test_remove_candidate_sites_only_deletes_generated_sites(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "remove_candidates.mufinder", "RemoveCandidates")
    candidate = MuonSite(label="Candidate", status="generated")
    relaxed = MuonSite(label="Relaxed", status="relaxed")
    doc.sites[candidate.id] = candidate
    doc.sites[relaxed.id] = relaxed

    removed = service.remove_candidate_sites(doc, [candidate.id, relaxed.id, uuid4()])

    assert [site.id for site in removed] == [candidate.id]
    assert candidate.id not in doc.sites
    assert relaxed.id in doc.sites
    assert doc.provenance[-1].action == "delete_candidates"
    assert doc.provenance[-1].input_entity_ids == [candidate.id]


def test_cluster_candidate_positions_updates_selected_sites_to_anchor_orbit(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_candidates.mufinder", "ClusterCandidates")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    site = MuonSite(
        label="Candidate",
        status="generated",
        fractional_position=[0.1, 0.2, 0.3],
        cartesian_position=[0.5, 1.0, 1.5],
    )
    doc.sites[site.id] = site

    updated = service.cluster_candidate_positions(
        doc,
        [site.id],
        anchor_fractional=[0.9, 0.8, 0.7],
    )

    assert updated == [site]
    assert site.fractional_position == pytest.approx([0.9, 0.8, 0.7])
    assert site.cartesian_position == pytest.approx([4.5, 4.0, 3.5])
    assert doc.provenance[-1].action == "cluster_candidate_positions"
    assert doc.provenance[-1].parameters["anchor_fractional"] == pytest.approx([0.9, 0.8, 0.7])


def test_cluster_candidate_positions_does_not_rewrite_staged_sites(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_staged_candidates.mufinder", "ClusterStagedCandidates")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    generated = MuonSite(
        label="Generated",
        status="generated",
        fractional_position=[0.1, 0.2, 0.3],
    )
    staged = MuonSite(
        label="Staged",
        status="staged",
        fractional_position=[0.2, 0.3, 0.4],
    )
    doc.sites[generated.id] = generated
    doc.sites[staged.id] = staged

    updated = service.cluster_candidate_positions(
        doc,
        [generated.id, staged.id],
        anchor_fractional=[0.9, 0.8, 0.7],
    )

    assert updated == [generated]
    assert generated.fractional_position == pytest.approx([0.9, 0.8, 0.7])
    assert staged.fractional_position == pytest.approx([0.2, 0.3, 0.4])
    assert doc.provenance[-1].parameters["site_labels"] == ["Generated"]


def test_stage_generated_sites_allows_already_staged_sites(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "stage_sites.mufinder", "StageSites")
    generated = MuonSite(label="Generated", status="generated")
    staged = MuonSite(label="Staged", status="staged")
    relaxed = MuonSite(label="Relaxed", status="relaxed")
    doc.sites[generated.id] = generated
    doc.sites[staged.id] = staged
    doc.sites[relaxed.id] = relaxed

    staged_sites = service.stage_generated_sites(doc, [generated.id, staged.id, relaxed.id])

    assert [site.id for site in staged_sites] == [generated.id, staged.id]
    assert generated.status == "staged"
    assert staged.status == "staged"
    assert relaxed.status == "relaxed"


def test_load_project_normalizes_legacy_candidate_status(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "legacy_candidate_status.mufinder"
    doc = service.create_project(root, "LegacyCandidateStatus")
    site = MuonSite(label="Legacy", status="candidate")
    doc.sites[site.id] = site
    service.save_project(doc)

    site_file = root / "entities" / "sites" / f"{site.id}.json"
    text = site_file.read_text(encoding="utf-8")
    site_file.write_text(text.replace('"status": "generated"', '"status": "candidate"'), encoding="utf-8")

    loaded = service.load_project(root)

    assert loaded.sites[site.id].status == "generated"


def test_integrity_validation_detects_missing_references(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "integrity.mufinder", "Integrity")

    ghost_site = MuonSite(label="Ghost", relaxation_run_id=None)
    doc.sites[ghost_site.id] = ghost_site

    run = CalculationRun()
    run.result_id = Result().id
    doc.runs[run.id] = run

    issues = service.validate_document(doc)
    assert any("references missing result" in issue for issue in issues)


def test_job_transition_rules(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "job_rules_test.mufinder", "Jobs")
    job = service.add_job(
        doc,
        label="Demo",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_geom_opt",
    )

    service.submit_job(job)
    assert job.status == JobStatus.SUBMITTING

    service.mark_job_queued(job, remote_job_id="123")
    service.start_job(job)
    service.mark_job_retrieving(job)
    service.complete_job(job)

    assert job.status == JobStatus.COMPLETED
    assert job.remote_job_id == "123"

    with pytest.raises(ValueError):
        service.start_job(job)


def test_reset_failed_job_sets_defined_state(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "job_reset_test.mufinder", "JobReset")
    job = service.add_job(
        doc,
        label="ResetMe",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )

    service.submit_job(job)
    job.remote_directory = "/remote/work/ResetMe"
    job.num_processes = 8
    service.start_job(job)
    service.fail_job(job, "failure")

    assert job.status == JobStatus.FAILED
    assert job.error_message == "failure"

    service.reset_failed_job(job)

    assert job.status == JobStatus.DEFINED
    assert job.error_message is None
    assert job.submitted_at is None
    assert job.started_at is None
    assert job.completed_at is None
    assert job.remote_directory is None
    assert job.num_processes is None


def test_reset_failed_job_allows_cancelled_jobs(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "job_reset_cancelled_test.mufinder", "JobResetCancelled")
    job = service.add_job(
        doc,
        label="ResetCancelled",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )

    service.submit_job(job)
    job.num_processes = 4
    service.cancel_job(job)

    assert job.status == JobStatus.CANCELLED

    service.reset_failed_job(job)

    assert job.status == JobStatus.DEFINED
    assert job.error_message is None
    assert job.submitted_at is None
    assert job.started_at is None
    assert job.completed_at is None
    assert job.num_processes is None


def test_stop_job_sets_stopped_state(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "job_stop_test.mufinder", "JobStop")
    job = service.add_job(
        doc,
        label="StopMe",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )

    service.submit_job(job)
    service.start_job(job)
    service.stop_job(job, "walltime reached")

    assert job.status == JobStatus.STOPPED
    assert job.error_message == "walltime reached"
    assert job.completed_at is not None


def test_reset_failed_job_allows_stopped_jobs(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "job_reset_stopped_test.mufinder", "JobResetStopped")
    job = service.add_job(
        doc,
        label="ResetStopped",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )

    service.submit_job(job)
    job.num_processes = 16
    service.start_job(job)
    service.stop_job(job, "timeout")

    service.reset_failed_job(job)

    assert job.status == JobStatus.DEFINED
    assert job.error_message is None
    assert job.submitted_at is None
    assert job.started_at is None
    assert job.completed_at is None
    assert job.num_processes is None


def test_record_job_submission_appends_one_provenance_milestone(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "job_submission_log.mufinder", "JobSubmissionLog")
    job = service.add_job(
        doc,
        label="Conv300",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )

    service.submit_job(job)
    service.record_job_submission(doc, jobs=[job], execution_target="local")

    assert doc.provenance[-1].action == "submit_jobs"
    assert doc.provenance[-1].parameters["job_labels"] == ["Conv300"]
    assert doc.provenance[-1].parameters["execution_target"] == "local"


def test_record_job_terminal_events_and_reset_append_provenance(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "job_terminal_log.mufinder", "JobTerminalLog")
    job = service.add_job(
        doc,
        label="Relax1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    result = Result(
        result_type=job.calc_type,
        data={"total_energy": -500.0, "max_force": 0.03, "warnings": ["minor"]},
        derived_structure_ids=[uuid4()],
    )

    service.record_job_terminal_event(doc, action="job_completed", job=job, result=result)
    service.record_job_terminal_event(doc, action="job_failed", job=job, error_message="Submission failed")
    service.record_job_reset(doc, job=job, cleaned_output_directory=True)

    assert [record.action for record in doc.provenance[-3:]] == ["job_completed", "job_failed", "job_reset"]
    assert doc.provenance[-3].parameters["result_total_energy"] == -500.0
    assert doc.provenance[-3].parameters["derived_structure_count"] == 1
    assert doc.provenance[-2].parameters["error_message"] == "Submission failed"
    assert doc.provenance[-1].parameters["cleaned_output_directory"] is True


def test_lifecycle_transitions_do_not_append_intermediate_provenance(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "job_transition_no_log.mufinder", "JobTransitionNoLog")
    job = service.add_job(
        doc,
        label="NoJournalNoise",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )

    service.submit_job(job)
    service.mark_job_queued(job, remote_job_id="123")
    service.start_job(job)
    service.mark_job_retrieving(job)

    assert doc.provenance == []


def test_job_num_processes_roundtrip(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "job_processes_roundtrip.mufinder"
    doc = service.create_project(root, "JobProcessesRoundtrip")
    job = service.add_job(
        doc,
        label="ProcJob",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.num_processes = 32
    job.remote_directory = "/remote/work/ProcJob"

    service.save_project(doc)
    loaded = service.load_project(root)

    loaded_job = loaded.jobs[job.id]
    assert loaded_job.num_processes == 32
    assert loaded_job.remote_directory == "/remote/work/ProcJob"


def test_import_structure_file_sets_active_host_and_copies_source(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "import_test.mufinder", "Import")

    source = tmp_path / "input.cif"
    atoms = Atoms(
        "SiO",
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        pbc=True,
    )
    ase_write(source, atoms)

    structure = service.import_structure_file(doc, source, make_active=True)

    assert doc.project.host_structure_id == structure.id
    assert structure.source_file is not None
    copied = doc.project_root / structure.source_file
    assert copied.exists()
    assert len(doc.provenance) == 1


def test_import_structure_file_deduplicates_copy_name(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "import_dup.mufinder", "ImportDup")

    source = tmp_path / "same_name.cif"
    atoms = Atoms(
        "Si",
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )
    ase_write(source, atoms)

    first = service.import_structure_file(doc, source)
    second = service.import_structure_file(doc, source)

    assert first.source_file != second.source_file
    assert (doc.project_root / first.source_file).exists()
    assert (doc.project_root / second.source_file).exists()


def test_workflow_supercell_asset_roundtrip_and_selection_state(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "workflow_supercell_roundtrip.mufinder"
    doc = service.create_project(root, "WorkflowSupercell")
    host = _host_structure()
    host.properties["atomic_spins"] = [1.0, 0.0, -1.0]
    host.properties["propagation_vector"] = [0.0, 0.0, 0.5]
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    service.use_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        asset_structure_id=asset.id,
    )
    service.save_project(doc)

    loaded = service.load_project(root)
    resolution = service.resolve_workflow_structure(
        loaded,
        WorkflowStage.CONVERGENCE_CUTOFF,
        normalize_state=True,
    )

    assert loaded.ui_state["workflow_supercells"]["convergence_cutoff"]["mode"] == "saved_supercell"
    assert loaded.ui_state["workflow_supercells"]["convergence_cutoff"]["structure_id"] == str(asset.id)
    assert resolution.structure is not None
    assert resolution.structure.id == asset.id
    assert resolution.selected_asset is not None
    assert resolution.selected_asset.matrix == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert resolution.source_workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF
    assert loaded.structures[asset.id].properties["supercell_asset"]["owner_workflow_stage"] == "convergence_cutoff"


def test_use_workflow_supercell_asset_clones_for_other_workflow(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "workflow_supercell_clone.mufinder", "WorkflowClone")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    source_asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        reuse_existing=False,
    )

    selected = service.use_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        asset_structure_id=source_asset.id,
    )
    repeated = service.use_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        asset_structure_id=source_asset.id,
    )

    assert selected.id != source_asset.id
    assert repeated.id == selected.id
    assert selected.parent_structure_id == host.id
    assert selected.properties["supercell_asset"]["owner_workflow_stage"] == "convergence_cutoff"
    assert selected.properties["supercell_asset"]["source_asset_structure_id"] == str(source_asset.id)


def test_resolve_workflow_structure_falls_back_to_unit_cell_for_mismatched_host(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "workflow_supercell_host_change.mufinder", "WorkflowHostChange")
    first_host = _host_structure()
    second_host = Structure(
        label="OtherHost",
        origin="imported",
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, first_host)
    service.add_structure(doc, second_host)
    doc.project.host_structure_id = first_host.id

    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=first_host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    service.use_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        asset_structure_id=asset.id,
    )

    doc.project.host_structure_id = second_host.id
    resolution = service.resolve_workflow_structure(
        doc,
        WorkflowStage.CONVERGENCE_CUTOFF,
        normalize_state=True,
    )

    assert resolution.structure is not None
    assert resolution.structure.id == second_host.id
    assert resolution.selected_asset is None
    assert resolution.normalized_selection.mode == "unit_cell"
    assert doc.ui_state["workflow_supercells"]["convergence_cutoff"]["mode"] == "unit_cell"
    assert resolution.available_assets == []


def test_workflow_supercell_helpers_handle_invalid_metadata_and_defaults(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "workflow_supercell_helpers.mufinder", "WorkflowHelpers")
    doc.ui_state["workflow_supercells"] = "bad-state"

    assert service._identity_supercell_matrix() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert service._coerce_supercell_matrix([[1, 0, 0]]) is None
    assert service._coerce_supercell_matrix([[1, 0, 0], [0, 1], [0, 0, 1]]) is None
    assert service._parse_optional_uuid("not-a-uuid") is None
    assert "[" in service._supercell_asset_label(
        WorkflowStage.CONVERGENCE_CUTOFF,
        [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
    )

    selection = service.get_workflow_supercell_selection(doc, WorkflowStage.CONVERGENCE_CUTOFF)
    assert selection.mode == "unit_cell"
    assert isinstance(doc.ui_state["workflow_supercells"], dict)
    assert service.list_workflow_supercell_assets(doc, host_structure_id=None) == []

    invalid_asset = Structure(
        label="InvalidAsset",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        parent_structure_id=uuid4(),
        properties={"supercell_asset": {"matrix": "bad", "owner_workflow_stage": "bad"}},
    )
    assert service._supercell_asset(invalid_asset) is None
    assert service._find_matching_workflow_supercell_asset(
        doc,
        host_structure_id=uuid4(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
    ) is None

    magnetic = Structure(
        label="Magnetic",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        properties={
            "atomic_spins": ["bad"],
            "magnetic_structure": {
                "propagation_vector": ["bad", 0.25, 0.5],
                "moments": [{"vector": [0.0, 0.0, 1.25]}],
            },
        },
    )
    assert service.structure_atomic_spins(magnetic) == pytest.approx([1.25])
    assert service.structure_propagation_vector(magnetic) == pytest.approx([0.0, 0.25, 0.5])

    stale = Structure(
        label="StaleSpins",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        properties={
            "atomic_spins": [0.0],
            "magnetic_structure": {
                "moments": [{"vector": [0.0, 0.0, 1.75]}],
            },
        },
    )
    assert service.structure_atomic_spins(stale) == pytest.approx([1.75])


def test_resolve_workflow_structure_repairs_legacy_zero_spin_supercell_asset(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "legacy_spin_repair.mufinder", "LegacySpinRepair")
    host = _host_structure()
    host.properties["atomic_spins"] = [2.0, -2.0, 1.5]
    host.properties["magnetic_structure"] = {
        "moments": [
            {"label": "O1", "vector": [0.0, 0.0, 2.0], "coordinate_system": "cartesian"},
            {"label": "Si2", "vector": [0.0, 0.0, -2.0], "coordinate_system": "cartesian"},
            {"label": "O3", "vector": [0.0, 0.0, 1.5], "coordinate_system": "cartesian"},
        ],
        "propagation_vector": [0.0, 0.0, 0.0],
    }
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    asset.properties["atomic_spins"] = [0.0] * len(asset.species)

    service.use_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        asset_structure_id=asset.id,
    )
    resolution = service.resolve_workflow_structure(
        doc,
        WorkflowStage.CONVERGENCE_CUTOFF,
        normalize_state=True,
    )

    assert resolution.structure is not None
    repaired_spins = resolution.structure.properties.get("atomic_spins")
    assert isinstance(repaired_spins, list)
    assert any(abs(float(value)) > 1e-9 for value in repaired_spins)


def test_import_mcif_sets_project_magnetic_structure(tmp_path: Path) -> None:
        service = ProjectService()
        root = tmp_path / "import_mcif.mufinder"
        doc = service.create_project(root, "ImportMCIF")

        source = tmp_path / "magnetic.mcif"
        source.write_text(
                """
data_test
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 6
_cell_length_b 6
_cell_length_c 6
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_cell_wave_vector_x 0.0
_cell_wave_vector_y 0.0
_cell_wave_vector_z 0.25
loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Fe1 Fe 0.0 0.0 0.0
loop_
    _atom_site_moment.label
    _atom_site_moment.crystalaxis_x
    _atom_site_moment.crystalaxis_y
    _atom_site_moment.crystalaxis_z
    Fe1 0.0 0.0 2.0
""".strip(),
                encoding="utf-8",
        )

        structure = service.import_structure_file(doc, source, make_active=True)
        assert structure.properties.get("magnetic_structure") is not None
        assert structure.properties.get("atomic_spins") == pytest.approx([2.0])
        assert structure.properties.get("propagation_vector") == pytest.approx([0.0, 0.0, 0.25])
        assert doc.project.magnetic_structure is not None
        assert doc.project.magnetic_structure["propagation_vector"] == pytest.approx([0.0, 0.0, 0.25])

        service.save_project(doc)
        loaded = service.load_project(root)
        assert loaded.project.magnetic_structure is not None
        assert loaded.project.magnetic_structure["moments"][0]["vector"] == pytest.approx([0.0, 0.0, 2.0])


def test_register_convergence_cutoff_definition_persists_snapshot(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "conv_snapshot.mufinder"
    doc = service.create_project(root, "ConvSnapshot")
    host = _host_structure()
    service.add_structure(doc, host)

    definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0, 360.0],
            "shared_params": {"task": "SINGLEPOINT", "xc_functional": "PBE"},
            "overrides": [{"cutoffs": [360.0], "params": {"max_scf_cycles": 120}}],
        },
        output_directory=root / "files" / "convergence_cutoff",
    )

    assert definition.id in doc.calculations
    assert doc.calculations[definition.id].workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF
    assert len(doc.provenance) >= 1
    assert doc.provenance[-1].action == "configure_convergence_cutoff"

    service.save_project(doc)
    loaded = service.load_project(root)
    loaded_definition = loaded.calculations[definition.id]
    assert loaded_definition.parameters["configuration"]["cutoff_energies"] == [300.0, 360.0]
    assert loaded_definition.parameters["job_ids_by_cutoff"] == {}


def test_set_recommended_cutoff_persists_to_project_manifest(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "recommended_cutoff.mufinder"
    doc = service.create_project(root, "RecommendedCutoff")

    service.set_recommended_cutoff(
        doc,
        550.0,
        method="asymptotic",
        energy_threshold=0.001,
        force_threshold=0.05,
        cell_id="11111111-1111-1111-1111-111111111111",
    )

    assert doc.project.recommended_cutoff_energy == pytest.approx(550.0)
    assert doc.project.recommended_cutoff_method == "asymptotic"
    assert doc.project.recommended_cutoff_energy_threshold == pytest.approx(0.001)
    assert doc.project.recommended_cutoff_force_threshold == pytest.approx(0.05)
    assert doc.project.recommended_cutoff_cell_id == "11111111-1111-1111-1111-111111111111"
    assert doc.provenance[-1].action == "set_recommended_cutoff"

    service.save_project(doc)
    loaded = service.load_project(root)
    assert loaded.project.recommended_cutoff_energy == pytest.approx(550.0)
    assert loaded.project.recommended_cutoff_method == "asymptotic"
    assert loaded.project.recommended_cutoff_energy_threshold == pytest.approx(0.001)
    assert loaded.project.recommended_cutoff_force_threshold == pytest.approx(0.05)
    assert loaded.project.recommended_cutoff_cell_id == "11111111-1111-1111-1111-111111111111"


def test_set_recommended_kpoint_grid_persists_to_project_manifest(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "recommended_kpoint.mufinder"
    doc = service.create_project(root, "RecommendedKPoint")

    lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]

    service.set_recommended_kpoint_grid(
        doc,
        (4, 4, 3),
        lattice_vectors=lattice,
        method="delta-threshold",
        energy_threshold=0.001,
        force_threshold=0.05,
        cell_id="22222222-2222-2222-2222-222222222222",
    )

    assert doc.project.recommended_kpoint_grid == (4, 4, 3)
    assert doc.project.recommended_kpoint_method == "delta-threshold"
    assert doc.project.recommended_kpoint_energy_threshold == pytest.approx(0.001)
    assert doc.project.recommended_kpoint_force_threshold == pytest.approx(0.05)
    assert doc.project.recommended_kpoint_cell_id == "22222222-2222-2222-2222-222222222222"
    assert doc.project.recommended_kpoint_lattice == lattice
    assert doc.provenance[-1].action == "set_recommended_kpoint_grid"

    service.save_project(doc)
    loaded = service.load_project(root)
    assert loaded.project.recommended_kpoint_grid == (4, 4, 3)
    assert loaded.project.recommended_kpoint_method == "delta-threshold"
    assert loaded.project.recommended_kpoint_energy_threshold == pytest.approx(0.001)
    assert loaded.project.recommended_kpoint_force_threshold == pytest.approx(0.05)
    assert loaded.project.recommended_kpoint_cell_id == "22222222-2222-2222-2222-222222222222"
    assert loaded.project.recommended_kpoint_lattice == lattice


def test_register_pristine_relaxation_definition_persists_configuration(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "pristine_relax_definition.mufinder"
    doc = service.create_project(root, "PristineRelaxDefinition")
    structure = service.import_structure_file(doc, Path("tests/golden/si_geom_opt.cell"), make_active=True)

    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=structure.id,
        parameters={
            "name_prefix": "pristine_relax",
            "rows": [
                {
                    "label": "row_a",
                    "fix_all_cell": True,
                    "supercell_matrix": None,
                    "parameter_values": {"task": "GEOMETRYOPTIMIZATION"},
                    "added_advanced": [],
                }
            ],
        },
        output_directory=root / "files" / "pristine_relaxation",
    )

    assert definition.id in doc.calculations
    assert doc.calculations[definition.id].workflow_stage == WorkflowStage.PRISTINE_RELAXATION
    assert doc.calculations[definition.id].parameters["configuration"]["rows"][0]["label"] == "row_a"
    assert doc.calculations[definition.id].parameters["job_ids_by_label"] == {}
    assert doc.provenance[-1].action == "configure_pristine_relaxation"

    service.save_project(doc)
    loaded = service.load_project(root)
    loaded_definition = loaded.calculations[definition.id]
    assert loaded_definition.parameters["configuration"]["rows"][0]["label"] == "row_a"
    assert loaded_definition.parameters["job_ids_by_label"] == {}


def test_save_project_serializes_ui_state_with_value_with_unit(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "ui_state_value_with_unit.mufinder"
    doc = service.create_project(root, "UIStateValueWithUnit")

    doc.ui_state["example"] = {"threshold": ValueWithUnit(value=1e-6, unit="eV")}

    service.save_project(doc)
    loaded = service.load_project(root)

    assert loaded.ui_state["example"]["threshold"] == {"unit": "eV", "value": 1e-6}


def test_save_project_roundtrips_convergence_overview_snapshots(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "overview_snapshots.mufinder"
    doc = service.create_project(root, "OverviewSnapshots")

    doc.ui_state["convergence_overview_snapshots"] = {
        "cutoff": {
            "host_structure_id": "host-1",
            "simulation_cell_structure_id": "cell-1",
            "default_value": 550.0,
            "method": "asymptotic",
            "energy_threshold": 0.001,
            "force_threshold": 0.05,
            "plot_data": {
                "cutoffs": [300.0, 400.0, 550.0],
                "energies": [-10.0, -10.1, -10.15],
                "forces": [0.30, 0.18, 0.05],
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
            },
        },
        "kpoint": {
            "host_structure_id": "host-1",
            "simulation_cell_structure_id": "cell-2",
            "default_value": [2, 2, 2],
            "method": "delta-threshold",
            "energy_threshold": 0.001,
            "force_threshold": 0.05,
            "point_labels": {"8.0": "Grid: 2 x 2 x 2"},
            "plot_data": {
                "cutoffs": [8.0],
                "energies": [-10.0],
                "forces": [0.05],
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
            },
        },
    }

    service.save_project(doc)
    loaded = service.load_project(root)

    assert loaded.ui_state["convergence_overview_snapshots"]["cutoff"]["default_value"] == pytest.approx(550.0)
    assert loaded.ui_state["convergence_overview_snapshots"]["kpoint"]["default_value"] == [2, 2, 2]
