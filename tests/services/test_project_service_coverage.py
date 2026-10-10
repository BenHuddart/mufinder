"""Additional coverage for ProjectService mutation helpers and guard branches.

These tests target functions and error/guard branches not exercised by
``tests/services/test_project_service.py``: comparison-table helpers, job
lifecycle transitions (success + invalid), calculation-definition registration
error paths, recommended cutoff/k-point persistence with thresholds, several
``generate_candidate_sites`` branches, document validation issues, and small
parse/coercion helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    ComparisonTable,
    MuonSite,
    Result,
    Structure,
)
from mufinder.science.site_generation import SiteGenerationError
from mufinder.services.project_service import (
    ProjectService,
    WorkflowSupercellSelection,
)


def _host_structure() -> Structure:
    return Structure(
        label="Host",
        origin="imported",
        lattice_vectors=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        species=["O", "Si", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )


def _doc_with_host(tmp_path: Path, name: str):
    service = ProjectService()
    doc = service.create_project(tmp_path / f"{name}.mufinder", name)
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    return service, doc, host


# --------------------------------------------------------------------------- #
# Comparison-table helpers
# --------------------------------------------------------------------------- #


def test_add_rename_remove_comparison_table(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "cmp")
    table = ComparisonTable(label="Original")
    service.add_comparison_table(doc, table)
    assert doc.comparison_tables[table.id] is table

    renamed = service.rename_comparison_table(doc, table.id, "  New Label  ")
    assert renamed is table
    assert table.label == "New Label"

    # Empty label collapses to the default fallback.
    service.rename_comparison_table(doc, table.id, "   ")
    assert table.label == "Comparison Table"

    # Unknown id returns None and does not raise.
    assert service.rename_comparison_table(doc, uuid4(), "X") is None

    service.remove_comparison_table(doc, table.id)
    assert table.id not in doc.comparison_tables
    # Removing again is a no-op.
    service.remove_comparison_table(doc, table.id)


def test_record_comparison_table_created_appends_provenance(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "cmp_prov")
    dft_a, dft_b = uuid4(), uuid4()
    table = ComparisonTable(
        label="Compared",
        experimental_structure_id=host.id,
        dft_structure_ids=[dft_a, dft_b],
    )
    before = len(doc.provenance)
    service.record_comparison_table_created(doc, table=table)
    assert len(doc.provenance) == before + 1
    record = doc.provenance[-1]
    assert record.action == "create_comparison_table"
    assert record.parameters["dft_structure_count"] == 2
    assert host.id in record.input_entity_ids
    assert table.id in record.output_entity_ids

    # No experimental structure -> empty input entity ids.
    table2 = ComparisonTable(label="NoExp", dft_structure_ids=[dft_a])
    service.record_comparison_table_created(doc, table=table2)
    assert doc.provenance[-1].input_entity_ids == []


# --------------------------------------------------------------------------- #
# Job lifecycle transitions
# --------------------------------------------------------------------------- #


def test_full_job_lifecycle_through_completion(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "job_life")
    job = service.add_job(
        doc,
        label="Run A",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    assert job.status == JobStatus.DEFINED
    assert doc.jobs[job.id] is job

    service.submit_job(job)
    assert job.status == JobStatus.SUBMITTING
    assert job.submitted_at is not None

    service.mark_job_queued(job, remote_job_id="42")
    assert job.status == JobStatus.QUEUED
    assert job.remote_job_id == "42"

    service.start_job(job)
    assert job.status == JobStatus.RUNNING
    assert job.started_at is not None

    service.mark_job_retrieving(job)
    assert job.status == JobStatus.RETRIEVING

    service.complete_job(job)
    assert job.status == JobStatus.COMPLETED
    assert job.completed_at is not None
    assert job.error_message is None


def test_mark_job_queued_without_remote_id_keeps_existing(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "job_queue")
    job = service.add_job(
        doc,
        label="Q",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    service.submit_job(job)
    service.mark_job_queued(job)
    assert job.status == JobStatus.QUEUED
    assert job.remote_job_id is None


def test_fail_and_cancel_jobs(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "job_fail")
    job = service.add_job(
        doc,
        label="F",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    service.submit_job(job)
    service.start_job(job)
    service.fail_job(job, "boom")
    assert job.status == JobStatus.FAILED
    assert job.error_message == "boom"

    job2 = service.add_job(
        doc,
        label="C",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    service.cancel_job(job2)
    assert job2.status == JobStatus.CANCELLED
    assert job2.completed_at is not None


def test_invalid_transition_raises(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "job_invalid")
    job = service.add_job(
        doc,
        label="X",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    # Cannot complete a DEFINED job.
    with pytest.raises(ValueError, match="Invalid job transition"):
        service.complete_job(job)
    # Cannot start a DEFINED job (must submit first).
    with pytest.raises(ValueError, match="Invalid job transition"):
        service.start_job(job)


def test_reset_failed_job_rejects_completed(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "job_reset_bad")
    job = service.add_job(
        doc,
        label="R",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.COMPLETED
    with pytest.raises(ValueError, match="failed, stopped, or cancelled"):
        service.reset_failed_job(job)


# --------------------------------------------------------------------------- #
# Calculation-definition registration error/guard branches
# --------------------------------------------------------------------------- #


def test_register_convergence_cutoff_requires_known_structure(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "cutoff_err")
    with pytest.raises(ValueError, match="must reference a known structure"):
        service.register_convergence_cutoff_definition(
            doc,
            input_structure_id=uuid4(),
            parameters={"cutoff_energies": [400.0]},
            output_directory=tmp_path / "out",
        )


def test_register_convergence_kpoint_uses_host_fallback_and_outside_root(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "kpoint_def")
    # output_directory outside project root -> stored as absolute string.
    outside = Path("/tmp/definitely/outside/root")
    definition = service.register_convergence_kpoint_definition(
        doc,
        host_structure_id=host.id,
        parameters={"sweep_values": [[2, 2, 2], [4, 4, 4]]},
        output_directory=outside,
        job_ids_by_grid={"2 2 2": "j1"},
        simulation_cell_mode="saved_supercell",
        supercell_source_workflow="convergence_cutoff",
    )
    assert definition.workflow_stage == WorkflowStage.CONVERGENCE_KPOINT
    assert definition.input_structure_ids == [host.id]
    assert definition.parameters["output_directory"] == str(outside)
    assert definition.parameters["job_ids_by_grid"] == {"2 2 2": "j1"}
    assert definition.parameters["simulation_cell_mode"] == "saved_supercell"


def test_register_uep_definition_persists(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "uep_def")
    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={"rows": [{"label": "a"}, {"label": "b"}]},
        output_directory=doc.project_root / "uep_out",
    )
    assert definition.calc_type == "castep_uep_source"
    assert definition.parameters["output_directory"] == "uep_out"
    assert doc.provenance[-1].action == "configure_uep"


def test_register_muon_relaxation_parses_site_ids(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "muon_def")
    site = MuonSite(label="S1", status="staged", source_structure_id=host.id)
    doc.sites[site.id] = site
    definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters={"rows": [{"site_id": str(site.id)}, {"site_id": "not-a-uuid"}, {}]},
        output_directory=doc.project_root / "mr_out",
    )
    assert definition.input_site_ids == [site.id]
    assert definition.calc_type == "castep_muon_relaxation"


def test_register_quantum_phonon_validates_structures_and_sites(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "qc_def")
    site = MuonSite(label="S", status="staged", source_structure_id=host.id)
    doc.sites[site.id] = site

    with pytest.raises(ValueError, match="At least one input structure"):
        service.register_quantum_phonon_definition(
            doc,
            input_structure_ids=[],
            input_site_ids=[site.id],
            parameters={},
            output_directory=doc.project_root / "q",
        )
    with pytest.raises(ValueError, match="must reference known structures"):
        service.register_quantum_phonon_definition(
            doc,
            input_structure_ids=[uuid4()],
            input_site_ids=[site.id],
            parameters={},
            output_directory=doc.project_root / "q",
        )
    with pytest.raises(ValueError, match="must reference known sites"):
        service.register_quantum_phonon_definition(
            doc,
            input_structure_ids=[host.id],
            input_site_ids=[uuid4()],
            parameters={},
            output_directory=doc.project_root / "q",
        )

    definition = service.register_quantum_phonon_definition(
        doc,
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
        parameters={
            "singlepoint_job_ids_by_label": {"a": "1"},
            "phonon_job_ids_by_label": {"a": "2"},
            "job_seed_names_by_label": {"a": "seed"},
        },
        output_directory=doc.project_root / "q",
    )
    assert definition.parameters["singlepoint_job_ids_by_label"] == {"a": "1"}
    assert definition.parameters["phonon_job_ids_by_label"] == {"a": "2"}
    assert definition.parameters["job_seed_names_by_label"] == {"a": "seed"}


def test_register_neb_definition_validates_inputs(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "neb_def")
    site = MuonSite(label="S", status="staged", source_structure_id=host.id)
    doc.sites[site.id] = site

    with pytest.raises(ValueError, match="At least one input structure"):
        service.register_nudged_elastic_band_definition(
            doc,
            input_structure_ids=[],
            input_site_ids=[site.id],
            parameters={},
            output_directory=doc.project_root / "t",
        )
    with pytest.raises(ValueError, match="must reference known sites"):
        service.register_nudged_elastic_band_definition(
            doc,
            input_structure_ids=[host.id],
            input_site_ids=[uuid4()],
            parameters={},
            output_directory=doc.project_root / "t",
        )

    neb = service.register_nudged_elastic_band_definition(
        doc,
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
        parameters={"rows": [{"a": 1}]},
        output_directory=doc.project_root / "t",
    )
    assert neb.calc_type == "castep_nudged_elastic_band_set"


# --------------------------------------------------------------------------- #
# Recommended cutoff / k-point with thresholds and methods
# --------------------------------------------------------------------------- #


def test_set_recommended_cutoff_with_method_and_thresholds(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "rec_cutoff")
    service.set_recommended_cutoff(
        doc,
        450.0,
        method="energy",
        energy_threshold=0.001,
        force_threshold=0.05,
        stress_threshold=0.1,
        cell_id=" cellA ",
    )
    project = doc.project
    assert project.recommended_cutoff_energy == 450.0
    assert project.recommended_cutoff_method == "energy"
    assert project.recommended_cutoff_energy_threshold == 0.001
    assert project.recommended_cutoff_force_threshold == 0.05
    assert project.recommended_cutoff_stress_threshold == 0.1
    assert project.recommended_cutoff_cell_id == "cellA"
    assert "using method 'energy'" in doc.provenance[-1].description


def test_set_recommended_cutoff_blank_method_and_cell(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "rec_cutoff_blank")
    service.set_recommended_cutoff(doc, 500.0, method="   ", cell_id="   ")
    assert doc.project.recommended_cutoff_method is None
    assert doc.project.recommended_cutoff_cell_id is None
    assert "using method" not in doc.provenance[-1].description


def test_set_recommended_kpoint_grid_with_lattice(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "rec_kpoint")
    lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    service.set_recommended_kpoint_grid(
        doc,
        (4, 4, 4),
        lattice_vectors=lattice,
        method="spacing",
        energy_threshold=0.002,
        cell_id="kcell",
    )
    project = doc.project
    assert project.recommended_kpoint_grid == (4, 4, 4)
    assert project.recommended_kpoint_lattice == lattice
    assert project.recommended_kpoint_method == "spacing"
    assert project.recommended_kpoint_energy_threshold == 0.002
    assert project.recommended_kpoint_cell_id == "kcell"


def test_set_recommended_kpoint_grid_rejects_bad_lattice(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "rec_kpoint_bad")
    service.set_recommended_kpoint_grid(
        doc,
        (2, 2, 2),
        lattice_vectors=[[1.0, 0.0, 0.0]],  # wrong number of rows
    )
    assert doc.project.recommended_kpoint_lattice is None


# --------------------------------------------------------------------------- #
# generate_candidate_sites branches
# --------------------------------------------------------------------------- #


def test_generate_manual_fractional_wraps(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "gen_manual")
    sites = service.generate_candidate_sites(
        doc,
        method="manual",
        parameters={"coord_system": "fractional", "x": 1.25, "y": -0.25, "z": 0.5},
    )
    assert sites[0].fractional_position == pytest.approx([0.25, 0.75, 0.5])
    assert sites[0].properties["coordinate_system"] == "fractional"


def test_generate_near_element_count_branch(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "gen_near")
    sites = service.generate_candidate_sites(
        doc,
        method="near_element",
        parameters={
            "element": "Si",
            "muons_per_atom": 2,
            "bond_distance": 1.0,
            "min_muon_atom": 0.8,
        },
    )
    assert len(sites) >= 1
    assert all(site.generation_method == "near_element" for site in sites)
    assert sites[0].properties["target_element"] == "Si"
    assert sites[0].properties["strategy"] == "attached"


def test_generate_unknown_method_raises(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "gen_unknown")
    with pytest.raises(SiteGenerationError, match="Unknown generation method"):
        service.generate_candidate_sites(doc, method="bogus", parameters={})


def test_generate_uep_clusters_empty_selection_raises(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "gen_uep_empty")
    with pytest.raises(SiteGenerationError, match="one or more UEP cluster rows"):
        service.generate_candidate_sites(
            doc, method="uep_clusters", parameters={"selections": []}
        )


def test_generate_uep_clusters_missing_coords_raises(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "gen_uep_bad")
    with pytest.raises(SiteGenerationError, match="missing valid coordinates"):
        service.generate_candidate_sites(
            doc,
            method="uep_clusters",
            parameters={"selections": [{"fractional_position": [0.1, 0.2]}]},
        )


def test_generate_uep_clusters_all_duplicates_raises(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "gen_uep_dup")
    # Seed an existing cluster site with a matching source_key.
    existing = MuonSite(
        label="UEP Cluster Site 1",
        status="generated",
        generation_method="uep_cluster",
        fractional_position=[0.1, 0.1, 0.1],
        source_structure_id=host.id,
        properties={"uep_source_key": "k1"},
    )
    doc.sites[existing.id] = existing
    with pytest.raises(SiteGenerationError, match="already present"):
        service.generate_candidate_sites(
            doc,
            method="uep_clusters",
            parameters={
                "selections": [
                    {
                        "fractional_position": [0.2, 0.2, 0.2],
                        "cartesian_position": [2.0, 2.0, 2.0],
                        "source_key": "k1",
                    }
                ]
            },
        )


def test_generate_uep_clusters_creates_member_site(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "gen_uep_member")
    sites = service.generate_candidate_sites(
        doc,
        method="uep_clusters",
        parameters={
            "selections": [
                {
                    "fractional_position": [0.3, 0.3, 0.3],
                    "cartesian_position": [3.0, 3.0, 3.0],
                    "source_key": "key-a",
                    "properties": {"selection_kind": "member"},
                }
            ]
        },
    )
    assert len(sites) == 1
    site = sites[0]
    assert site.generation_method == "uep_cluster_member"
    assert site.properties["uep_source_key"] == "key-a"
    assert site.fractional_position == pytest.approx([0.3, 0.3, 0.3])


# --------------------------------------------------------------------------- #
# Small helper branches
# --------------------------------------------------------------------------- #


def test_random_selection_mode_variants() -> None:
    assert ProjectService._random_selection_mode({"selection_mode": "distance"}) == "separation"
    assert ProjectService._random_selection_mode({"mode": "number"}) == "count"
    assert ProjectService._random_selection_mode({"count": 3}) == "count"
    assert ProjectService._random_selection_mode({}) == "separation"


def test_parse_optional_int_branches() -> None:
    service = ProjectService()
    assert service._parse_optional_int(None, field_name="seed") is None
    assert service._parse_optional_int("   ", field_name="seed") is None
    assert service._parse_optional_int("12", field_name="seed") == 12
    with pytest.raises(SiteGenerationError, match="seed must be an integer"):
        service._parse_optional_int("abc", field_name="seed")


def test_has_duplicate_uep_cluster_site_position_match(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "dup_pos")
    site = MuonSite(
        label="member",
        status="generated",
        generation_method="uep_cluster_member",
        fractional_position=[0.4, 0.4, 0.4],
        source_structure_id=host.id,
    )
    doc.sites[site.id] = site
    # Same position, no source key -> matched by coordinates.
    assert service._has_duplicate_uep_cluster_site(doc, "", [0.4, 0.4, 0.4]) is True
    # Different position -> not a duplicate.
    assert service._has_duplicate_uep_cluster_site(doc, "", [0.9, 0.1, 0.2]) is False


def test_clear_workflow_supercell_selection(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "clear_sel")
    service.set_workflow_supercell_selection(
        doc,
        WorkflowStage.CONVERGENCE_CUTOFF,
        WorkflowSupercellSelection(mode="saved_supercell", structure_id=uuid4()),
    )
    service.clear_workflow_supercell_selection(doc, WorkflowStage.CONVERGENCE_CUTOFF)
    selection = service.get_workflow_supercell_selection(doc, WorkflowStage.CONVERGENCE_CUTOFF)
    assert selection.mode == "unit_cell"


def test_create_workflow_supercell_asset_errors(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "create_err")
    with pytest.raises(ValueError, match="must reference a known structure"):
        service.create_workflow_supercell_asset(
            doc,
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            host_structure_id=uuid4(),
            matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
    with pytest.raises(ValueError, match="3x3 integer matrix"):
        service.create_workflow_supercell_asset(
            doc,
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            host_structure_id=host.id,
            matrix=[[2, 0, 0], [0, 1, 0]],
        )


def test_rename_and_update_and_use_asset_errors(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "asset_err")
    # Rename: not a supercell asset.
    with pytest.raises(ValueError, match="not a saved supercell asset"):
        service.rename_workflow_supercell_asset(doc, asset_structure_id=host.id, label="x")

    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    # Rename: empty name.
    with pytest.raises(ValueError, match="must not be empty"):
        service.rename_workflow_supercell_asset(doc, asset_structure_id=asset.id, label="  ")

    # update matrix: unknown structure.
    with pytest.raises(ValueError, match="does not exist"):
        service.update_workflow_supercell_asset_matrix(
            doc, asset_structure_id=uuid4(), matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
    # update matrix: not an asset.
    with pytest.raises(ValueError, match="not a saved supercell asset"):
        service.update_workflow_supercell_asset_matrix(
            doc, asset_structure_id=host.id, matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
    # update matrix: bad matrix.
    with pytest.raises(ValueError, match="3x3 integer matrix"):
        service.update_workflow_supercell_asset_matrix(
            doc, asset_structure_id=asset.id, matrix=[[2, 0, 0]]
        )

    # delete: not an asset.
    with pytest.raises(ValueError, match="not a saved supercell asset"):
        service.delete_workflow_supercell_asset(doc, asset_structure_id=host.id)

    # use: unknown structure.
    with pytest.raises(ValueError, match="does not exist"):
        service.use_workflow_supercell_asset(
            doc, workflow_stage=WorkflowStage.UEP, asset_structure_id=uuid4()
        )
    # use: not an asset.
    with pytest.raises(ValueError, match="not a saved supercell asset"):
        service.use_workflow_supercell_asset(
            doc, workflow_stage=WorkflowStage.UEP, asset_structure_id=host.id
        )


def test_use_asset_without_host_raises(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "no_host.mufinder", "NoHost")
    with pytest.raises(ValueError, match="Load a host structure"):
        service.use_workflow_supercell_asset(
            doc, workflow_stage=WorkflowStage.UEP, asset_structure_id=uuid4()
        )


def test_repair_supercell_asset_if_needed_guards(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "repair_guard")
    # Unknown structure.
    assert service.repair_supercell_asset_if_needed(doc, structure_id=uuid4()) is False
    # Known but not an asset.
    assert service.repair_supercell_asset_if_needed(doc, structure_id=host.id) is False


def test_supercell_asset_references_includes_active_selection(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "asset_refs")
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
    refs = service.supercell_asset_references(doc, asset_structure_id=asset.id)
    assert any("active selection" in ref for ref in refs)
    assert service.supercell_asset_is_frozen(doc, asset_structure_id=asset.id) is False


# --------------------------------------------------------------------------- #
# Document validation branches
# --------------------------------------------------------------------------- #


def test_validate_document_detects_many_issues(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "validate.mufinder", "Validate")

    missing_struct = uuid4()
    doc.project.host_structure_id = missing_struct

    # Structure with missing parent and a broken supercell asset.
    bad_asset = Structure(
        label="BadAsset",
        lattice_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        properties={
            "supercell_asset": {
                "matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                "owner_workflow_stage": WorkflowStage.CONVERGENCE_CUTOFF.value,
                "source_asset_structure_id": str(uuid4()),
            }
        },
    )
    doc.structures[bad_asset.id] = bad_asset

    calc = CalculationDefinition(
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
        input_structure_ids=[uuid4()],
    )
    doc.calculations[calc.id] = calc

    run = CalculationRun(definition_id=uuid4(), job_id=uuid4(), result_id=uuid4())
    doc.runs[run.id] = run

    result = Result(calculation_run_id=uuid4(), derived_structure_ids=[uuid4()])
    doc.results[result.id] = result

    site = MuonSite(label="S", source_structure_id=uuid4(), relaxation_run_id=uuid4())
    doc.sites[site.id] = site

    table = ComparisonTable(
        label="T",
        experimental_structure_id=uuid4(),
        dft_structure_ids=[uuid4()],
    )
    doc.comparison_tables[table.id] = table

    issues = service.validate_document(doc)
    joined = "\n".join(issues)
    assert "host_structure_id does not reference a known structure" in joined
    assert "missing parent_structure_id" in joined
    assert "references missing source asset" in joined
    assert "references missing structures" in joined
    assert "references missing calculation definition" in joined
    assert "references missing job" in joined
    assert "references missing result" in joined
    assert "references missing calculation run" in joined
    assert "references missing derived structures" in joined
    assert "references missing structure" in joined
    assert "references missing run" in joined
    assert "references missing experimental structure" in joined
    assert "references missing DFT structures" in joined

    # save_project surfaces integrity errors.
    with pytest.raises(ValueError, match="integrity check failed"):
        service.save_project(doc)


# --------------------------------------------------------------------------- #
# load_project parse fallbacks (_project_from_dict)
# --------------------------------------------------------------------------- #


def test_load_project_tolerates_bad_recommended_values(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "load_fallback")
    service.save_project(doc)

    manifest_path = doc.project_root / "project.json"
    payload = json.loads(manifest_path.read_text())
    payload["recommended_cutoff_energy"] = "not-a-number"
    payload["recommended_cutoff_energy_threshold"] = "bad"
    payload["recommended_cutoff_force_threshold"] = "bad"
    payload["recommended_cutoff_stress_threshold"] = "bad"
    payload["recommended_cutoff_method"] = "   "
    payload["recommended_cutoff_cell_id"] = "   "
    payload["recommended_kpoint_grid"] = ["x", "y", "z"]
    payload["recommended_kpoint_lattice"] = [[1.0, 0.0, 0.0], "bad", [0.0, 0.0, 1.0]]
    payload["recommended_kpoint_energy_threshold"] = "bad"
    payload["recommended_kpoint_force_threshold"] = "bad"
    payload["recommended_kpoint_stress_threshold"] = "bad"
    payload["recommended_kpoint_method"] = "   "
    payload["recommended_kpoint_cell_id"] = "   "
    manifest_path.write_text(json.dumps(payload))

    loaded = service.load_project(doc.project_root)
    project = loaded.project
    assert project.recommended_cutoff_energy is None
    assert project.recommended_cutoff_energy_threshold is None
    assert project.recommended_cutoff_method is None
    assert project.recommended_cutoff_cell_id is None
    assert project.recommended_kpoint_grid is None
    assert project.recommended_kpoint_lattice is None
    assert project.recommended_kpoint_method is None
    assert project.recommended_kpoint_cell_id is None


def test_load_project_parses_valid_recommended_values(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "load_valid")
    service.set_recommended_cutoff(doc, 400.0, method="energy", cell_id="c1")
    service.set_recommended_kpoint_grid(
        doc,
        (3, 3, 3),
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        method="spacing",
        cell_id="k1",
    )
    service.save_project(doc)

    loaded = service.load_project(doc.project_root)
    project = loaded.project
    assert project.recommended_cutoff_energy == 400.0
    assert project.recommended_cutoff_method == "energy"
    assert project.recommended_cutoff_cell_id == "c1"
    assert project.recommended_kpoint_grid == (3, 3, 3)
    assert project.recommended_kpoint_method == "spacing"
    assert project.recommended_kpoint_cell_id == "k1"


# --------------------------------------------------------------------------- #
# record_job_terminal_event with result payload
# --------------------------------------------------------------------------- #


def test_record_job_terminal_event_with_result(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "terminal_result")
    job = service.add_job(
        doc,
        label="Done",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    derived = uuid4()
    result = Result(
        result_type="relaxation",
        data={"total_energy": -123.0, "max_force": 0.01, "warnings": ["w1", "w2"]},
        derived_structure_ids=[derived],
    )
    before = len(doc.provenance)
    service.record_job_terminal_event(
        doc, action="job_completed", job=job, result=result
    )
    assert len(doc.provenance) == before + 1
    record = doc.provenance[-1]
    assert record.parameters["result_total_energy"] == -123.0
    assert record.parameters["result_max_force"] == 0.01
    assert record.parameters["warnings_count"] == 2
    assert record.parameters["derived_structure_count"] == 1
    assert job.id in record.output_entity_ids
    assert result.id in record.output_entity_ids
    assert derived in record.output_entity_ids


def test_record_job_terminal_event_with_error_only(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "terminal_error")
    job = service.add_job(
        doc,
        label="Failed",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    service.record_job_terminal_event(
        doc, action="job_failed", job=job, error_message="kaboom"
    )
    record = doc.provenance[-1]
    assert record.parameters["error_message"] == "kaboom"
    assert "failed" in record.description.lower()


def test_record_job_reset_appends_provenance(tmp_path: Path) -> None:
    service, doc, _ = _doc_with_host(tmp_path, "reset_prov")
    job = service.add_job(
        doc,
        label="ResetMe",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    service.record_job_reset(doc, job=job, cleaned_output_directory=True)
    record = doc.provenance[-1]
    assert record.action == "job_reset"
    assert record.parameters["cleaned_output_directory"] is True
    assert record.job_id == job.id


# --------------------------------------------------------------------------- #
# stage_generated_sites no-op branch
# --------------------------------------------------------------------------- #


def test_stage_generated_sites_skips_non_stageable(tmp_path: Path) -> None:
    service, doc, host = _doc_with_host(tmp_path, "stage_skip")
    relaxed = MuonSite(label="R", status="relaxed", source_structure_id=host.id)
    doc.sites[relaxed.id] = relaxed
    before = len(doc.provenance)
    staged = service.stage_generated_sites(doc, [relaxed.id, uuid4()])
    assert staged == []
    # No provenance appended when nothing staged.
    assert len(doc.provenance) == before


def test_cluster_candidate_positions_guards(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_guard.mufinder", "ClusterGuard")
    # No host.
    with pytest.raises(SiteGenerationError, match="Select a host structure"):
        service.cluster_candidate_positions(doc, [uuid4()], anchor_fractional=[0.0, 0.0, 0.0])

    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    # Empty site ids.
    with pytest.raises(SiteGenerationError, match="Select one or more"):
        service.cluster_candidate_positions(doc, [], anchor_fractional=[0.0, 0.0, 0.0])
    # Bad anchor length.
    with pytest.raises(SiteGenerationError, match="three-component"):
        service.cluster_candidate_positions(doc, [uuid4()], anchor_fractional=[0.0, 0.0])
    # No generated sites among the selection.
    with pytest.raises(SiteGenerationError, match="did not contain any generated"):
        service.cluster_candidate_positions(doc, [uuid4()], anchor_fractional=[0.1, 0.1, 0.1])
