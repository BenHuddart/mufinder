"""Tests for project lifecycle, integrity, and generation orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from uuid import UUID, uuid4

import pytest
from ase import Atoms
from ase.io import write as ase_write

from mufinder.domain.enums import FailureReason, JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    ComparisonTable,
    Job,
    PROJECT_SCHEMA_VERSION,
    MuonSite,
    Result,
    Structure,
    utc_now,
)
from mufinder.castep_io.types import ValueWithUnit
from mufinder.persistence.codec import to_jsonable
from mufinder.science.site_generation import SiteGenerationError
from mufinder.services.journal import build_journal
from mufinder.services.magnetic_field_results import (
    MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE,
    MAGNETIC_FIELD_PAGE_OWNER,
    MAGNETIC_FIELD_RESULT_TYPE,
    NUCLEAR_RELAXATION_RESULT_TYPE,
    attach_magnetic_field_results,
    magnetic_field_results,
)
from mufinder.services.project_service import (
    ProjectService,
    SiteHostMismatch,
    StructureHostMismatch,
    WorkflowSupercellSelection,
    active_definition,
    definitions_for_stage,
    job_for_mapping_key,
)
from tests.support.jobs import make_job
from tests.support.structures import silicon_oxide_host


def _document_with_host(tmp_path: Path, name: str = "project"):
    """Return ``(service, document, host)`` with an imported O-Si-O host active."""

    service = ProjectService()
    document = service.create_project(tmp_path / f"{name}.mufinder", name)
    host = silicon_oxide_host(origin="imported")
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    return service, document, host


def test_create_save_and_load_roundtrip(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "roundtrip.mufinder"

    doc = service.create_project(root, "Roundtrip")
    host = silicon_oxide_host()
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
    host = silicon_oxide_host()
    service.add_structure(doc, host)

    sites = service.generate_candidate_sites(
        doc,
        method="manual",
        parameters={"coord_system": "cartesian", "x": 2.0, "y": 3.0, "z": 4.0},
    )
    site = sites[0]
    assert site.status == "generated"
    assert site.initial_fractional_position == pytest.approx([0.2, 0.3, 0.4])
    assert site.initial_cartesian_position == pytest.approx([2.0, 3.0, 4.0])


def test_random_generation_by_target_count_estimates_spacing(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "random_count.mufinder", "RandomCount")
    host = silicon_oxide_host()
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
    host = silicon_oxide_host()
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
    service.add_structure(doc, silicon_oxide_host())
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
    host = silicon_oxide_host()
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
    host = silicon_oxide_host()
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
    assert all(site.properties["target_element"] == "O" for site in sites)
    assert all(site.properties["strategy"] == "attached" for site in sites)
    assert all(site.properties["muons_per_atom"] == 2 for site in sites)
    assert all("min_muon_muon" not in site.properties for site in sites)


def test_uep_cluster_generation_creates_sites_and_rejects_duplicates(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_cluster_generation.mufinder", "UEPClusterGeneration")
    host = silicon_oxide_host()
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
    assert site.initial_fractional_position == pytest.approx([0.1, 0.2, 0.3])
    assert site.properties["uep_source_key"] == "job-1:cluster:1:cluster:0"
    assert site.properties["classification"] == "minimum"

    with pytest.raises(SiteGenerationError, match="already present"):
        service.generate_candidate_sites(doc, method="uep_clusters", parameters={"selections": [selection]})


def test_uep_cluster_generation_creates_one_site_per_selection(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_cluster_multi_generation.mufinder", "UEPClusterMultiGeneration")
    host = silicon_oxide_host()
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
    assert [site.initial_fractional_position[0] for site in sites] == pytest.approx([0.1, 0.2, 0.3])


def test_manual_generation_wraps_fractional_coordinates(tmp_path: Path) -> None:
    service, doc, _host = _document_with_host(tmp_path)

    sites = service.generate_candidate_sites(
        doc,
        method="manual",
        parameters={"coord_system": "fractional", "x": 1.25, "y": -0.25, "z": 0.5},
    )

    assert sites[0].initial_fractional_position == pytest.approx([0.25, 0.75, 0.5])
    assert sites[0].properties["coordinate_system"] == "fractional"


@pytest.mark.parametrize(
    ("mode_parameters", "expected_mode"),
    [
        ({"selection_mode": "distance", "count": 1}, "separation"),
        ({"mode": "number", "count": 1}, "count"),
        ({"count": 1}, "count"),
        ({}, "separation"),
    ],
    ids=["distance-alias", "number-alias", "count-implies-count", "default-separation"],
)
def test_random_generation_resolves_selection_mode_aliases(
    tmp_path: Path, mode_parameters: dict[str, object], expected_mode: str
) -> None:
    service, doc, _host = _document_with_host(tmp_path)

    sites = service.generate_candidate_sites(
        doc,
        method="random",
        parameters={
            **mode_parameters,
            "min_muon_muon": 2.0,
            "min_muon_atom": 0.8,
            "max_attempts": 500,
            "seed": "12",
            "include_symmetry_equivalents": False,
        },
    )

    assert sites
    assert {site.properties["selection_mode"] for site in sites} == {expected_mode}


@pytest.mark.parametrize("seed", [None, "   "], ids=["absent", "blank"])
def test_random_generation_treats_a_blank_seed_as_unseeded(tmp_path: Path, seed: object) -> None:
    service, doc, _host = _document_with_host(tmp_path)

    sites = service.generate_candidate_sites(
        doc,
        method="random",
        parameters={"count": 1, "min_muon_atom": 0.8, "seed": seed, "include_symmetry_equivalents": False},
    )

    assert len(sites) == 1


def test_random_generation_rejects_a_non_integer_seed(tmp_path: Path) -> None:
    service, doc, _host = _document_with_host(tmp_path)

    with pytest.raises(SiteGenerationError, match="seed must be an integer"):
        service.generate_candidate_sites(doc, method="random", parameters={"count": 1, "seed": "abc"})


def test_generation_rejects_an_unknown_method(tmp_path: Path) -> None:
    service, doc, _host = _document_with_host(tmp_path)

    with pytest.raises(SiteGenerationError, match="Unknown generation method"):
        service.generate_candidate_sites(doc, method="bogus", parameters={})


@pytest.mark.parametrize(
    ("selections", "message"),
    [
        ([], "one or more UEP cluster rows"),
        ([{"fractional_position": [0.1, 0.2]}], "missing valid coordinates"),
    ],
    ids=["no-selection", "missing-coordinates"],
)
def test_uep_cluster_generation_rejects_unusable_selections(
    tmp_path: Path, selections: list[object], message: str
) -> None:
    service, doc, _host = _document_with_host(tmp_path)

    with pytest.raises(SiteGenerationError, match=message):
        service.generate_candidate_sites(doc, method="uep_clusters", parameters={"selections": selections})


def test_uep_cluster_generation_creates_member_sites(tmp_path: Path) -> None:
    service, doc, host = _document_with_host(tmp_path)

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
    assert site.label == "UEP Cluster Member 1"
    assert site.source_structure_id == host.id
    assert site.properties["uep_source_key"] == "key-a"
    assert site.initial_fractional_position == pytest.approx([0.3, 0.3, 0.3])


def test_uep_cluster_generation_treats_a_site_at_the_same_position_as_a_duplicate(
    tmp_path: Path,
) -> None:
    service, doc, host = _document_with_host(tmp_path)
    existing = MuonSite(
        label="member",
        status="generated",
        generation_method="uep_cluster_member",
        initial_fractional_position=[0.4, 0.4, 0.4],
        source_structure_id=host.id,
    )
    doc.sites[existing.id] = existing

    def _selection(source_key: str, fractional: list[float]) -> dict[str, object]:
        return {"source_key": source_key, "fractional_position": fractional, "cartesian_position": [4.0, 4.0, 4.0]}

    # A new source key at an occupied position (modulo the lattice) is still a duplicate.
    with pytest.raises(SiteGenerationError, match="already present"):
        service.generate_candidate_sites(
            doc, method="uep_clusters", parameters={"selections": [_selection("new-key", [1.4, 0.4, -0.6])]}
        )

    sites = service.generate_candidate_sites(
        doc, method="uep_clusters", parameters={"selections": [_selection("new-key", [0.9, 0.1, 0.2])]}
    )
    assert [site.properties["uep_source_key"] for site in sites] == ["new-key"]


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
        initial_fractional_position=[0.1, 0.2, 0.3],
        initial_cartesian_position=[0.5, 1.0, 1.5],
    )
    doc.sites[site.id] = site

    updated = service.cluster_candidate_positions(
        doc,
        [site.id],
        anchor_fractional=[0.9, 0.8, 0.7],
    )

    assert updated == [site]
    assert site.initial_fractional_position == pytest.approx([0.9, 0.8, 0.7])
    assert site.initial_cartesian_position == pytest.approx([4.5, 4.0, 3.5])
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
        initial_fractional_position=[0.1, 0.2, 0.3],
    )
    staged = MuonSite(
        label="Staged",
        status="staged",
        initial_fractional_position=[0.2, 0.3, 0.4],
    )
    doc.sites[generated.id] = generated
    doc.sites[staged.id] = staged

    updated = service.cluster_candidate_positions(
        doc,
        [generated.id, staged.id],
        anchor_fractional=[0.9, 0.8, 0.7],
    )

    assert updated == [generated]
    assert generated.initial_fractional_position == pytest.approx([0.9, 0.8, 0.7])
    assert staged.initial_fractional_position == pytest.approx([0.2, 0.3, 0.4])
    assert doc.provenance[-1].parameters["site_labels"] == ["Generated"]


def test_cluster_candidate_positions_rejects_unusable_requests(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_guard.mufinder", "ClusterGuard")
    with pytest.raises(SiteGenerationError, match="Select a host structure"):
        service.cluster_candidate_positions(doc, [uuid4()], anchor_fractional=[0.0, 0.0, 0.0])

    host = silicon_oxide_host()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    with pytest.raises(SiteGenerationError, match="Select one or more"):
        service.cluster_candidate_positions(doc, [], anchor_fractional=[0.0, 0.0, 0.0])
    with pytest.raises(SiteGenerationError, match="three-component"):
        service.cluster_candidate_positions(doc, [uuid4()], anchor_fractional=[0.0, 0.0])
    with pytest.raises(SiteGenerationError, match="did not contain any generated"):
        service.cluster_candidate_positions(doc, [uuid4()], anchor_fractional=[0.1, 0.1, 0.1])


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


def test_stage_generated_sites_without_stageable_sites_changes_nothing(tmp_path: Path) -> None:
    service, doc, host = _document_with_host(tmp_path)
    relaxed = MuonSite(label="R", status="relaxed", source_structure_id=host.id)
    doc.sites[relaxed.id] = relaxed
    before = len(doc.provenance)

    staged = service.stage_generated_sites(doc, [relaxed.id, uuid4()])

    assert staged == []
    assert relaxed.status == "relaxed"
    assert len(doc.provenance) == before


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


def test_load_project_reads_site_total_energy_saved_under_legacy_key(tmp_path: Path) -> None:
    """A site saved before the rename stored its total energy as ``relative_energy_eV``."""

    service = ProjectService()
    root = tmp_path / "legacy_site_energy.mufinder"
    doc = service.create_project(root, "LegacySiteEnergy")
    site = MuonSite(label="Legacy", status="relaxed", total_energy_eV=-1234.5)
    doc.sites[site.id] = site
    service.save_project(doc)

    site_file = root / "entities" / "sites" / f"{site.id}.json"
    text = site_file.read_text(encoding="utf-8")
    assert '"total_energy_eV": -1234.5' in text
    assert "relative_energy_eV" not in text
    site_file.write_text(text.replace('"total_energy_eV"', '"relative_energy_eV"'), encoding="utf-8")

    loaded = service.load_project(root)

    assert loaded.sites[site.id].total_energy_eV == -1234.5


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


def test_integrity_validation_reports_every_dangling_reference_and_blocks_save(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "validate.mufinder", "Validate")
    doc.project.host_structure_id = uuid4()

    broken_asset = Structure(
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
    doc.structures[broken_asset.id] = broken_asset
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
    table = ComparisonTable(label="T", experimental_structure_id=uuid4(), dft_structure_ids=[uuid4()])
    doc.comparison_tables[table.id] = table

    joined = "\n".join(service.validate_document(doc))

    for fragment in (
        "host_structure_id does not reference a known structure",
        "missing parent_structure_id",
        "references missing source asset",
        "references missing structures",
        "references missing calculation definition",
        "references missing job",
        "references missing result",
        "references missing calculation run",
        "references missing derived structures",
        "references missing structure",
        "references missing run",
        "references missing experimental structure",
        "references missing DFT structures",
    ):
        assert fragment in joined
    with pytest.raises(ValueError, match="integrity check failed"):
        service.save_project(doc)


def test_job_transition_rules(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "job_rules_test.mufinder", "Jobs")
    job = service.add_job(
        doc,
        label="Demo",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_geom_opt",
    )

    assert doc.jobs[job.id] is job
    assert job.status == JobStatus.DEFINED

    service.submit_job(job)
    assert job.status == JobStatus.SUBMITTING
    assert job.submitted_at is not None

    service.mark_job_queued(job, remote_job_id="123")
    assert job.status == JobStatus.QUEUED
    service.start_job(job)
    assert job.started_at is not None
    service.mark_job_retrieving(job)
    assert job.status == JobStatus.RETRIEVING
    service.complete_job(job)

    assert job.status == JobStatus.COMPLETED
    assert job.remote_job_id == "123"
    assert job.completed_at is not None
    assert job.error_message is None

    with pytest.raises(ValueError):
        service.start_job(job)


def test_mark_job_queued_without_a_remote_id_keeps_it_unset(tmp_path: Path) -> None:
    service, doc, _host = _document_with_host(tmp_path)
    job = service.add_job(doc, label="Q", workflow_stage=WorkflowStage.UEP, calc_type="castep_uep_source")

    service.submit_job(job)
    service.mark_job_queued(job)

    assert job.status == JobStatus.QUEUED
    assert job.remote_job_id is None


def test_fail_job_records_the_error_and_cancel_job_the_completion_time(tmp_path: Path) -> None:
    service, doc, _host = _document_with_host(tmp_path)
    failed = service.add_job(doc, label="F", workflow_stage=WorkflowStage.UEP, calc_type="castep_uep_source")
    cancelled = service.add_job(doc, label="C", workflow_stage=WorkflowStage.UEP, calc_type="castep_uep_source")

    service.submit_job(failed)
    service.start_job(failed)
    service.fail_job(failed, "boom")
    service.cancel_job(cancelled)

    assert failed.status == JobStatus.FAILED
    assert failed.error_message == "boom"
    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.completed_at is not None


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
    job.remote_job_id = "12345"
    job.remote_directory = "/remote/work/Conv300"
    job.num_processes = 16
    service.record_job_submission(doc, jobs=[job], execution_target="local")

    assert doc.provenance[-1].action == "submit_jobs"
    assert doc.provenance[-1].parameters["job_labels"] == ["Conv300"]
    assert doc.provenance[-1].parameters["execution_target"] == "local"
    assert doc.provenance[-1].parameters["jobs"] == [
        {
            "job_id": str(job.id),
            "label": "Conv300",
            "status": "submitting",
            "execution_target": "local",
            "remote_job_id": "12345",
            "remote_directory": "/remote/work/Conv300",
            "num_processes": 16,
            "submitted_at": job.submitted_at.isoformat() if job.submitted_at else None,
        }
    ]


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

    completed, failed, reset = doc.provenance[-3:]
    assert [record.action for record in (completed, failed, reset)] == ["job_completed", "job_failed", "job_reset"]
    assert completed.parameters["result_total_energy"] == -500.0
    assert completed.parameters["result_max_force"] == 0.03
    assert completed.parameters["warnings_count"] == 1
    assert completed.parameters["derived_structure_count"] == 1
    assert {job.id, result.id, *result.derived_structure_ids} <= set(completed.output_entity_ids)
    assert failed.parameters["error_message"] == "Submission failed"
    assert "failed" in failed.description.lower()
    assert reset.parameters["cleaned_output_directory"] is True
    assert reset.job_id == job.id


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


def test_job_monitor_metadata_and_dismissal_roundtrip(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "job_monitor_roundtrip.mufinder"
    doc = service.create_project(root, "JobMonitorRoundtrip")
    job = service.add_job(
        doc,
        label="MonitorJob",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    checked_at = datetime(2026, 7, 4, 12, 30, tzinfo=UTC)
    dismissed_at = datetime(2026, 7, 4, 13, 0, tzinfo=UTC)
    job.last_checked_at = checked_at
    job.last_status_source = "scheduler_poll"
    job.last_status_message = "scheduler reports RUNNING"
    job.dismissed_at = dismissed_at

    service.save_project(doc)
    loaded = service.load_project(root)

    loaded_job = loaded.jobs[job.id]
    assert loaded_job.last_checked_at == checked_at
    assert loaded_job.last_status_source == "scheduler_poll"
    assert loaded_job.last_status_message == "scheduler reports RUNNING"
    assert loaded_job.dismissed_at == dismissed_at


def test_reset_failed_job_clears_monitor_metadata_and_dismissal() -> None:
    service = ProjectService()
    job = Job(
        label="ResetMonitor",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        status=JobStatus.FAILED,
    )
    job.last_checked_at = utc_now()
    job.last_status_source = "watchdog"
    job.last_status_message = "stopped by the job watchdog"
    job.dismissed_at = utc_now()

    service.reset_failed_job(job)

    assert job.last_checked_at is None
    assert job.last_status_source is None
    assert job.last_status_message is None
    assert job.dismissed_at is None


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
    host = silicon_oxide_host()
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
    host = silicon_oxide_host()
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


def test_workflow_supercell_asset_rename_update_and_delete(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "workflow_supercell_edit.mufinder", "WorkflowEdit")
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        label="Initial cell",
        reuse_existing=False,
    )

    renamed = service.rename_workflow_supercell_asset(
        doc,
        asset_structure_id=asset.id,
        label="Production 2x1x1",
    )
    assert renamed.label == "Production 2x1x1"

    updated = service.update_workflow_supercell_asset_matrix(
        doc,
        asset_structure_id=asset.id,
        matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        label="Production 2x2x1",
    )
    assert updated.id == asset.id
    assert updated.label == "Production 2x2x1"
    assert len(updated.species) == len(host.species) * 4
    assert updated.properties["supercell_asset"]["matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 1]]

    service.use_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        asset_structure_id=asset.id,
    )
    removed = service.delete_workflow_supercell_asset(doc, asset_structure_id=asset.id)

    assert removed.id == asset.id
    assert asset.id not in doc.structures
    assert doc.ui_state["workflow_supercells"]["pristine_relaxation"]["mode"] == "unit_cell"


def test_workflow_supercell_asset_delete_and_matrix_update_block_frozen_definitions(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "workflow_supercell_frozen.mufinder", "WorkflowFrozen")
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        input_structure_ids=[asset.id],
    )
    doc.calculations[definition.id] = definition

    assert service.supercell_asset_is_frozen(doc, asset_structure_id=asset.id)
    assert service.supercell_asset_references(doc, asset_structure_id=asset.id) == [
        "convergence_cutoff calculation 'castep_convergence_cutoff'"
    ]
    with pytest.raises(ValueError, match="configured calculations"):
        service.update_workflow_supercell_asset_matrix(
            doc,
            asset_structure_id=asset.id,
            matrix=[[3, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
    with pytest.raises(ValueError, match="configured calculations"):
        service.delete_workflow_supercell_asset(doc, asset_structure_id=asset.id)


def test_clear_workflow_supercell_selection_restores_the_unit_cell(tmp_path: Path) -> None:
    service, doc, _host = _document_with_host(tmp_path)
    service.set_workflow_supercell_selection(
        doc,
        WorkflowStage.CONVERGENCE_CUTOFF,
        WorkflowSupercellSelection(mode="saved_supercell", structure_id=uuid4()),
    )

    service.clear_workflow_supercell_selection(doc, WorkflowStage.CONVERGENCE_CUTOFF)

    assert service.get_workflow_supercell_selection(doc, WorkflowStage.CONVERGENCE_CUTOFF).mode == "unit_cell"


def test_workflow_supercell_asset_operations_reject_invalid_targets(tmp_path: Path) -> None:
    service, doc, host = _document_with_host(tmp_path)
    matrix = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]

    with pytest.raises(ValueError, match="must reference a known structure"):
        service.create_workflow_supercell_asset(
            doc, workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF, host_structure_id=uuid4(), matrix=matrix
        )
    with pytest.raises(ValueError, match="3x3 integer matrix"):
        service.create_workflow_supercell_asset(
            doc, workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF, host_structure_id=host.id, matrix=matrix[:2]
        )
    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=matrix,
        reuse_existing=False,
    )

    with pytest.raises(ValueError, match="not a saved supercell asset"):
        service.rename_workflow_supercell_asset(doc, asset_structure_id=host.id, label="x")
    with pytest.raises(ValueError, match="must not be empty"):
        service.rename_workflow_supercell_asset(doc, asset_structure_id=asset.id, label="  ")
    with pytest.raises(ValueError, match="does not exist"):
        service.update_workflow_supercell_asset_matrix(doc, asset_structure_id=uuid4(), matrix=matrix)
    with pytest.raises(ValueError, match="not a saved supercell asset"):
        service.update_workflow_supercell_asset_matrix(doc, asset_structure_id=host.id, matrix=matrix)
    with pytest.raises(ValueError, match="3x3 integer matrix"):
        service.update_workflow_supercell_asset_matrix(doc, asset_structure_id=asset.id, matrix=[[2, 0, 0]])
    with pytest.raises(ValueError, match="not a saved supercell asset"):
        service.delete_workflow_supercell_asset(doc, asset_structure_id=host.id)
    with pytest.raises(ValueError, match="does not exist"):
        service.use_workflow_supercell_asset(doc, workflow_stage=WorkflowStage.UEP, asset_structure_id=uuid4())
    with pytest.raises(ValueError, match="not a saved supercell asset"):
        service.use_workflow_supercell_asset(doc, workflow_stage=WorkflowStage.UEP, asset_structure_id=host.id)
    # Only known saved assets are ever repaired.
    assert service.repair_supercell_asset_if_needed(doc, structure_id=uuid4()) is False
    assert service.repair_supercell_asset_if_needed(doc, structure_id=host.id) is False


def test_use_workflow_supercell_asset_requires_a_host(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "no_host.mufinder", "NoHost")

    with pytest.raises(ValueError, match="Load a host structure"):
        service.use_workflow_supercell_asset(doc, workflow_stage=WorkflowStage.UEP, asset_structure_id=uuid4())


def test_supercell_asset_references_include_an_active_selection_without_freezing_it(
    tmp_path: Path,
) -> None:
    service, doc, host = _document_with_host(tmp_path)
    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    service.use_workflow_supercell_asset(
        doc, workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF, asset_structure_id=asset.id
    )

    references = service.supercell_asset_references(doc, asset_structure_id=asset.id)

    assert any("active selection" in reference for reference in references)
    assert service.supercell_asset_is_frozen(doc, asset_structure_id=asset.id) is False


def test_resolve_workflow_structure_falls_back_to_unit_cell_for_mismatched_host(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "workflow_supercell_host_change.mufinder", "WorkflowHostChange")
    first_host = silicon_oxide_host()
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

    selection = service.get_workflow_supercell_selection(doc, WorkflowStage.CONVERGENCE_CUTOFF)
    assert selection.mode == "unit_cell"
    assert isinstance(doc.ui_state["workflow_supercells"], dict)
    assert service.list_workflow_supercell_assets(doc, host_structure_id=None) == []

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
    host = silicon_oxide_host()
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
        assert len(doc.project.magnetic_structure["propagation_vectors_rlu"]) == 1
        assert doc.project.magnetic_structure["propagation_vectors_rlu"][0] == pytest.approx([0.0, 0.0, 0.25])
        assert doc.project.magnetic_structure["moments"][0]["atom_index"] == 0

        service.save_project(doc)
        loaded = service.load_project(root)
        assert loaded.project.magnetic_structure is not None
        assert loaded.project.magnetic_structure["moments"][0]["vector"] == pytest.approx([0.0, 0.0, 2.0])


def test_register_convergence_cutoff_definition_persists_snapshot(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "conv_snapshot.mufinder"
    doc = service.create_project(root, "ConvSnapshot")
    host = silicon_oxide_host()
    service.add_structure(doc, host)

    definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "sweep_values": [300.0, 360.0],
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
    assert loaded_definition.parameters["configuration"]["sweep_values"] == [300.0, 360.0]
    assert loaded_definition.parameters["job_ids_by_cutoff"] == {}


def test_register_convergence_cutoff_definition_requires_a_known_structure(tmp_path: Path) -> None:
    service, doc, _host = _document_with_host(tmp_path)

    with pytest.raises(ValueError, match="must reference a known structure"):
        service.register_convergence_cutoff_definition(
            doc,
            input_structure_id=uuid4(),
            parameters={"sweep_values": [400.0]},
            output_directory=tmp_path / "out",
        )


_SCHEMA_1_FIXTURE = Path(__file__).parents[1] / "fixtures" / "projects" / "schema_v1_cutoff_sweep"
_SCHEMA_1_CUTOFF_ID = UUID("2a16e0b8-9427-4964-8009-503aa12bd7c1")
_SCHEMA_1_KPOINT_ID = UUID("382dc08c-6750-40f1-9967-8758b292d57e")


def _copy_schema_1_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "schema_v1.mufinder"
    shutil.copytree(_SCHEMA_1_FIXTURE, root)
    return root


def test_schema_1_project_loads_with_cutoff_energies_migrated_to_sweep_values(tmp_path: Path) -> None:
    root = _copy_schema_1_fixture(tmp_path)

    loaded = ProjectService().load_project(root)

    assert loaded.project.schema_version == PROJECT_SCHEMA_VERSION
    cutoff = loaded.calculations[_SCHEMA_1_CUTOFF_ID].parameters["configuration"]
    assert cutoff["sweep_values"] == [300.0, 400.0, 500.0]
    assert "cutoff_energies" not in cutoff
    assert cutoff["overrides"] == [{"cutoffs": [500.0], "params": {"max_scf_cycles": 120}}]
    kpoint = loaded.calculations[_SCHEMA_1_KPOINT_ID].parameters["configuration"]
    assert kpoint["sweep_values"] == [[2, 2, 2], [4, 4, 4]]
    assert "cutoff_energies" not in kpoint
    # Provenance is history and the form state is not part of the migration.
    assert loaded.provenance[0].parameters["configuration"]["cutoff_energies"] == [300.0, 400.0, 500.0]
    assert loaded.ui_state["convergence_form"]["cutoff_energies"] == [300.0, 400.0, 500.0]
    assert "Prepared a cutoff convergence sweep with 3 cutoff value(s)." in [
        entry.body for day in build_journal("Schema 1", loaded.provenance).days for entry in day.entries
    ]


def test_schema_1_project_is_persisted_at_the_current_schema_by_the_next_save(tmp_path: Path) -> None:
    root = _copy_schema_1_fixture(tmp_path)
    service = ProjectService()

    service.save_project(service.load_project(root))

    assert json.loads((root / "project.json").read_text())["schema_version"] == PROJECT_SCHEMA_VERSION
    for definition_id in (_SCHEMA_1_CUTOFF_ID, _SCHEMA_1_KPOINT_ID):
        stored = json.loads((root / "entities" / "calculations" / f"{definition_id}.json").read_text())
        assert "cutoff_energies" not in stored["parameters"]["configuration"]
    assert (root / "provenance.jsonl").read_bytes() == (_SCHEMA_1_FIXTURE / "provenance.jsonl").read_bytes()

    reloaded = ProjectService().load_project(root)
    assert reloaded.project.schema_version == PROJECT_SCHEMA_VERSION
    assert reloaded.calculations[_SCHEMA_1_CUTOFF_ID].parameters["configuration"]["sweep_values"] == [
        300.0,
        400.0,
        500.0,
    ]


_SCHEMA_2_FIXTURE = Path(__file__).parents[1] / "fixtures" / "projects" / "schema_v2_relaxed_site"
_SCHEMA_2_RELAXED_SITE_ID = UUID("0d5c6e1a-8a3b-4f0e-9a7c-3c2b1d4e5f60")
_SCHEMA_2_STAGED_SITE_ID = UUID("7e2a9b4c-5d6f-4a1b-8c3d-9e0f1a2b3c4d")


def _copy_schema_2_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "schema_v2.mufinder"
    shutil.copytree(_SCHEMA_2_FIXTURE, root)
    return root


def test_schema_2_project_loads_with_initial_and_backfilled_relaxed_site_positions(tmp_path: Path) -> None:
    root = _copy_schema_2_fixture(tmp_path)

    loaded = ProjectService().load_project(root)

    assert loaded.project.schema_version == PROJECT_SCHEMA_VERSION
    relaxed = loaded.sites[_SCHEMA_2_RELAXED_SITE_ID]
    assert relaxed.status == "relaxed"
    assert relaxed.initial_fractional_position == pytest.approx([0.1, 0.2, 0.3])
    assert relaxed.initial_cartesian_position == pytest.approx([0.4, 0.8, 1.2])
    # Backfilled from the linked run's result.
    assert relaxed.relaxed_fractional_position == pytest.approx([0.15, 0.25, 0.35])
    assert relaxed.current_fractional_position() == pytest.approx([0.15, 0.25, 0.35])
    staged = loaded.sites[_SCHEMA_2_STAGED_SITE_ID]
    assert staged.initial_fractional_position == pytest.approx([0.5, 0.5, 0.5])
    assert staged.relaxed_fractional_position is None
    assert staged.current_fractional_position() == pytest.approx([0.5, 0.5, 0.5])


def test_schema_2_project_sites_are_rewritten_with_the_new_keys_by_the_next_save(tmp_path: Path) -> None:
    root = _copy_schema_2_fixture(tmp_path)
    service = ProjectService()

    service.save_project(service.load_project(root))

    assert json.loads((root / "project.json").read_text())["schema_version"] == PROJECT_SCHEMA_VERSION
    stored = json.loads((root / "entities" / "sites" / f"{_SCHEMA_2_RELAXED_SITE_ID}.json").read_text())
    assert "fractional_position" not in stored
    assert "cartesian_position" not in stored
    assert stored["initial_fractional_position"] == [0.1, 0.2, 0.3]
    assert stored["relaxed_fractional_position"] == [0.15, 0.25, 0.35]

    reloaded = ProjectService().load_project(root)
    assert reloaded.project.schema_version == PROJECT_SCHEMA_VERSION
    assert reloaded.sites[_SCHEMA_2_RELAXED_SITE_ID].relaxed_fractional_position == pytest.approx(
        [0.15, 0.25, 0.35]
    )
    assert reloaded.sites[_SCHEMA_2_STAGED_SITE_ID].initial_fractional_position == pytest.approx([0.5, 0.5, 0.5])


_SCHEMA_3_FIXTURE = Path(__file__).parents[1] / "fixtures" / "projects" / "schema_v3_magnetic_results"
_SCHEMA_3_WORKSPACE_ID = "d3b07384-d9a0-4c9b-8f1e-2a3b4c5d6e7f"


def _copy_schema_3_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "schema_v3.mufinder"
    shutil.copytree(_SCHEMA_3_FIXTURE, root)
    return root


def test_schema_3_project_loads_with_magnetic_field_results_as_result_records(tmp_path: Path) -> None:
    root = _copy_schema_3_fixture(tmp_path)

    loaded = ProjectService().load_project(root)

    assert loaded.project.schema_version == PROJECT_SCHEMA_VERSION
    page = magnetic_field_results(loaded, MAGNETIC_FIELD_PAGE_OWNER)
    assert [record.result_type for record in page] == [MAGNETIC_FIELD_RESULT_TYPE]
    assert page[0].calculation_run_id is None
    assert page[0].data["results"][0]["total_T"] == [0.1, 0.2, 0.3]
    workspace = {record.result_type for record in magnetic_field_results(loaded, _SCHEMA_3_WORKSPACE_ID)}
    assert workspace == {MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE, NUCLEAR_RELAXATION_RESULT_TYPE}
    assert loaded.ui_state["magnetic_field_form"] == {"histogram_bins": 512}
    assert loaded.ui_state["magnetic_field_workspaces"][0]["form"] == {"histogram_bins": 256}
    assert attach_magnetic_field_results(
        loaded, _SCHEMA_3_WORKSPACE_ID, loaded.ui_state["magnetic_field_workspaces"][0]["form"]
    )["nuclear_relaxation_results"] == {"mode": "zero_field_powder", "sigma_us_inv": 0.4}


def test_schema_3_project_is_persisted_with_result_records_by_the_next_save(tmp_path: Path) -> None:
    root = _copy_schema_3_fixture(tmp_path)
    service = ProjectService()
    loaded = service.load_project(root)
    record_ids = {record.id for record in magnetic_field_results(loaded)}

    service.save_project(loaded)

    assert json.loads((root / "project.json").read_text())["schema_version"] == PROJECT_SCHEMA_VERSION
    state = json.loads((root / "state.json").read_text())
    assert "magnetic_field_results" not in state["magnetic_field_form"]
    assert "dipolar_tensor_results" not in state["magnetic_field_workspaces"][0]["form"]
    assert {path.stem for path in (root / "entities" / "results").glob("*.json")} == {
        str(record_id) for record_id in record_ids
    }

    reloaded = ProjectService().load_project(root)
    assert {record.id for record in magnetic_field_results(reloaded)} == record_ids
    assert reloaded.ui_state["magnetic_field_form"] == {"histogram_bins": 512}


_SCHEMA_4_FIXTURE = Path(__file__).parents[1] / "fixtures" / "projects" / "schema_v4_dipolar_tensor"
_SCHEMA_4_TENSOR_ID = "5b6c7d8e-9f0a-4b1c-8d2e-3f4a5b6c7d8e"
# The fixture stores muesr's raw face-centre tensor, in Å^-3, for a 3 Å simple cubic lattice.
_SCHEMA_4_TENSOR_T_PER_MUB = [[0.14885, 0.0, 0.0], [0.0, 0.14885, 0.0], [0.0, 0.0, -0.29770]]


def _schema_4_tensor(root_or_project) -> list[list[float]]:
    if isinstance(root_or_project, Path):
        record = json.loads((root_or_project / "entities" / "results" / f"{_SCHEMA_4_TENSOR_ID}.json").read_text())
        return record["data"]["results"][0]["dipolar_tensor_T_per_muB"]
    (record,) = magnetic_field_results(root_or_project, MAGNETIC_FIELD_PAGE_OWNER)
    assert record.result_type == MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE
    return record.data["results"][0]["dipolar_tensor_T_per_muB"]


def test_schema_4_project_loads_with_dipolar_tensors_in_tesla_per_mu_b(tmp_path: Path) -> None:
    root = tmp_path / "schema_v4.mufinder"
    shutil.copytree(_SCHEMA_4_FIXTURE, root)

    loaded = ProjectService().load_project(root)

    assert loaded.project.schema_version == PROJECT_SCHEMA_VERSION
    assert _schema_4_tensor(loaded) == [pytest.approx(row, abs=1e-4) for row in _SCHEMA_4_TENSOR_T_PER_MUB]


def test_schema_4_project_tensor_is_converted_once_across_save_and_reload(tmp_path: Path) -> None:
    root = tmp_path / "schema_v4.mufinder"
    shutil.copytree(_SCHEMA_4_FIXTURE, root)
    service = ProjectService()

    service.save_project(service.load_project(root))

    assert json.loads((root / "project.json").read_text())["schema_version"] == PROJECT_SCHEMA_VERSION
    stored = _schema_4_tensor(root)
    assert stored == [pytest.approx(row, abs=1e-4) for row in _SCHEMA_4_TENSOR_T_PER_MUB]
    assert _schema_4_tensor(ProjectService().load_project(root)) == stored


def test_register_convergence_kpoint_definition_records_the_simulation_cell(tmp_path: Path) -> None:
    service, doc, host = _document_with_host(tmp_path)
    # An output directory outside the project root is stored as an absolute path.
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


def test_register_uep_definition_stores_a_project_relative_output_directory(tmp_path: Path) -> None:
    service, doc, host = _document_with_host(tmp_path)

    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={"rows": [{"label": "a"}, {"label": "b"}]},
        output_directory=doc.project_root / "uep_out",
    )

    assert definition.calc_type == "castep_uep_source"
    assert definition.parameters["output_directory"] == "uep_out"
    assert doc.provenance[-1].action == "configure_uep"


def test_register_muon_relaxation_definition_keeps_only_valid_site_ids(tmp_path: Path) -> None:
    service, doc, host = _document_with_host(tmp_path)
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


def _registration_snapshot(document, definition, placeholders: dict[str, str]) -> dict:
    """Return the definition and its provenance record as JSON-ready data.

    Generated ids and timestamps are dropped, and the ids a test knows are
    replaced by ``placeholders`` so the snapshot is stable across runs.
    """

    record = document.provenance[-1]
    payload = {
        "definition": {
            key: value
            for key, value in to_jsonable(definition).items()
            if key not in {"id", "created_at"}
        },
        "provenance": {
            key: value
            for key, value in to_jsonable(record).items()
            if key not in {"id", "timestamp"}
        },
    }
    text = json.dumps(payload)
    for identifier, name in {str(definition.id): "<definition>", **placeholders}.items():
        text = text.replace(identifier, name)
    return json.loads(text)


def test_register_definitions_persist_their_parameters_and_provenance_unchanged(
    tmp_path: Path,
) -> None:
    """Characterise the stored definition and provenance shape of every workflow."""

    service, doc, host = _document_with_host(tmp_path)
    site = MuonSite(label="S1", status="relaxed", source_structure_id=host.id)
    doc.sites[site.id] = site
    job = make_job("j", service=service, document=doc)
    outside = Path("/tmp/definitely/outside/root")
    placeholders = {str(host.id): "<host>", str(site.id): "<site>", str(job.id): "<job>"}
    snapshots: dict[str, dict] = {}

    definition = service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={"sweep_values": [300.0, 360.0], "shift": ValueWithUnit(0.5, "eV")},
        output_directory=doc.project_root / "files" / "cutoff",
        job_ids_by_cutoff={"300": str(job.id)},
        simulation_cell_mode="saved_supercell",
        supercell_source_workflow="muon_relaxation",
    )
    snapshots["cutoff"] = _registration_snapshot(doc, definition, placeholders)
    # Filling the definition's job map later must not rewrite the recorded event.
    definition.parameters["job_ids_by_cutoff"]["360"] = "later"
    assert doc.provenance[-1].parameters["job_ids_by_cutoff"] == {"300": str(job.id)}

    definition = service.register_convergence_kpoint_definition(
        doc,
        input_structure_id=host.id,
        parameters={"sweep_values": [[2, 2, 2], [4, 4, 4], [6, 6, 6]]},
        output_directory=outside,
    )
    snapshots["kpoint"] = _registration_snapshot(doc, definition, placeholders)

    definition = service.register_pristine_relaxation_definition(
        doc,
        host_structure_id=host.id,
        parameters={"rows": [{"label": "a"}]},
        output_directory=doc.project_root / "pristine",
        job_ids_by_label={"a": str(job.id)},
    )
    snapshots["pristine"] = _registration_snapshot(doc, definition, placeholders)

    definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters={"rows": [{"site_id": str(site.id)}, {"site_id": "bad"}, "x"]},
        output_directory=doc.project_root / "muon",
    )
    snapshots["muon"] = _registration_snapshot(doc, definition, placeholders)

    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={"rows": [{"label": "a"}, {"label": "b"}]},
        output_directory=outside,
        job_ids_by_label={"a": str(job.id)},
    )
    snapshots["uep"] = _registration_snapshot(doc, definition, placeholders)

    definition = service.register_quantum_phonon_definition(
        doc,
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
        parameters={
            "singlepoint_job_ids_by_label": {"S1": str(job.id)},
            "phonon_job_ids_by_label": ["not", "a", "mapping"],
        },
        output_directory=doc.project_root / "quantum",
        job_ids_by_label={"S1": str(job.id)},
    )
    snapshots["quantum"] = _registration_snapshot(doc, definition, placeholders)

    definition = service.register_nudged_elastic_band_definition(
        doc,
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
        parameters={"method": "NEB"},
        output_directory=doc.project_root / "neb",
    )
    snapshots["neb"] = _registration_snapshot(doc, definition, placeholders)

    assert snapshots == _REGISTRATION_SNAPSHOTS
    # Key order is part of the stored shape: compare the serialised text too.
    assert json.dumps(snapshots) == json.dumps(_REGISTRATION_SNAPSHOTS)
    assert len(doc.calculations) == 7
    assert len(doc.provenance) == 7


_REGISTRATION_SNAPSHOTS: dict[str, dict] = {
    "cutoff": {
        "definition": {
            "workflow_stage": "convergence_cutoff",
            "calc_type": "castep_convergence_cutoff",
            "parameters": {
                "configuration": {
                    "sweep_values": [300.0, 360.0],
                    "shift": {"value": 0.5, "unit": "eV"},
                },
                "output_directory": "files/cutoff",
                "job_ids_by_cutoff": {"300": "<job>"},
                "simulation_cell_mode": "saved_supercell",
                "supercell_source_workflow": "muon_relaxation",
            },
            "input_structure_ids": ["<host>"],
            "input_site_ids": [],
            "custom_param_text": None,
        },
        "provenance": {
            "action": "configure_convergence_cutoff",
            "workflow_stage": "convergence_cutoff",
            "actor": "user",
            "description": "Configured cutoff convergence with 2 cutoff value(s)",
            "parameters": {
                "calculation_definition_id": "<definition>",
                "configuration": {
                    "sweep_values": [300.0, 360.0],
                    "shift": {"value": 0.5, "unit": "eV"},
                },
                "output_directory": "files/cutoff",
                "job_ids_by_cutoff": {"300": "<job>"},
                "simulation_cell_mode": "saved_supercell",
                "supercell_source_workflow": "muon_relaxation",
            },
            "input_entity_ids": ["<host>"],
            "output_entity_ids": ["<definition>"],
            "job_id": None,
        },
    },
    "kpoint": {
        "definition": {
            "workflow_stage": "convergence_kpoint",
            "calc_type": "castep_convergence_kpoint",
            "parameters": {
                "configuration": {"sweep_values": [[2, 2, 2], [4, 4, 4], [6, 6, 6]]},
                "output_directory": "/tmp/definitely/outside/root",
                "job_ids_by_grid": {},
                "simulation_cell_mode": "unit_cell",
                "supercell_source_workflow": None,
            },
            "input_structure_ids": ["<host>"],
            "input_site_ids": [],
            "custom_param_text": None,
        },
        "provenance": {
            "action": "configure_convergence_kpoint",
            "workflow_stage": "convergence_kpoint",
            "actor": "user",
            "description": "Configured k-point convergence with 3 grid value(s)",
            "parameters": {
                "calculation_definition_id": "<definition>",
                "configuration": {"sweep_values": [[2, 2, 2], [4, 4, 4], [6, 6, 6]]},
                "output_directory": "/tmp/definitely/outside/root",
                "job_ids_by_grid": {},
                "simulation_cell_mode": "unit_cell",
                "supercell_source_workflow": None,
            },
            "input_entity_ids": ["<host>"],
            "output_entity_ids": ["<definition>"],
            "job_id": None,
        },
    },
    "pristine": {
        "definition": {
            "workflow_stage": "pristine_relaxation",
            "calc_type": "castep_pristine_relaxation",
            "parameters": {
                "configuration": {"rows": [{"label": "a"}]},
                "output_directory": "pristine",
                "job_ids_by_label": {"a": "<job>"},
            },
            "input_structure_ids": ["<host>"],
            "input_site_ids": [],
            "custom_param_text": None,
        },
        "provenance": {
            "action": "configure_pristine_relaxation",
            "workflow_stage": "pristine_relaxation",
            "actor": "user",
            "description": "Configured pristine relaxation with 1 job row(s)",
            "parameters": {
                "calculation_definition_id": "<definition>",
                "configuration": {"rows": [{"label": "a"}]},
                "output_directory": "pristine",
                "job_ids_by_label": {"a": "<job>"},
            },
            "input_entity_ids": ["<host>"],
            "output_entity_ids": ["<definition>"],
            "job_id": None,
        },
    },
    "muon": {
        "definition": {
            "workflow_stage": "muon_relaxation",
            "calc_type": "castep_muon_relaxation",
            "parameters": {
                "configuration": {"rows": [{"site_id": "<site>"}, {"site_id": "bad"}, "x"]},
                "output_directory": "muon",
                "job_ids_by_label": {},
            },
            "input_structure_ids": ["<host>"],
            "input_site_ids": ["<site>"],
            "custom_param_text": None,
        },
        "provenance": {
            "action": "configure_muon_relaxation",
            "workflow_stage": "muon_relaxation",
            "actor": "user",
            "description": "Configured muon relaxation with 3 job row(s)",
            "parameters": {
                "calculation_definition_id": "<definition>",
                "configuration": {"rows": [{"site_id": "<site>"}, {"site_id": "bad"}, "x"]},
                "output_directory": "muon",
                "job_ids_by_label": {},
            },
            "input_entity_ids": ["<host>", "<site>"],
            "output_entity_ids": ["<definition>"],
            "job_id": None,
        },
    },
    "uep": {
        "definition": {
            "workflow_stage": "uep",
            "calc_type": "castep_uep_source",
            "parameters": {
                "configuration": {"rows": [{"label": "a"}, {"label": "b"}]},
                "output_directory": "/tmp/definitely/outside/root",
                "job_ids_by_label": {"a": "<job>"},
            },
            "input_structure_ids": ["<host>"],
            "input_site_ids": [],
            "custom_param_text": None,
        },
        "provenance": {
            "action": "configure_uep",
            "workflow_stage": "uep",
            "actor": "user",
            "description": "Configured UEP source workflow with 2 job row(s)",
            "parameters": {
                "calculation_definition_id": "<definition>",
                "configuration": {"rows": [{"label": "a"}, {"label": "b"}]},
                "output_directory": "/tmp/definitely/outside/root",
                "job_ids_by_label": {"a": "<job>"},
            },
            "input_entity_ids": ["<host>"],
            "output_entity_ids": ["<definition>"],
            "job_id": None,
        },
    },
    "quantum": {
        "definition": {
            "workflow_stage": "quantum_corrections",
            "calc_type": "castep_quantum_phonon_set",
            "parameters": {
                "configuration": {
                    "singlepoint_job_ids_by_label": {"S1": "<job>"},
                    "phonon_job_ids_by_label": ["not", "a", "mapping"],
                },
                "output_directory": "quantum",
                "job_ids_by_label": {"S1": "<job>"},
                "singlepoint_job_ids_by_label": {"S1": "<job>"},
                "phonon_job_ids_by_label": {},
                "job_seed_names_by_label": {},
            },
            "input_structure_ids": ["<host>"],
            "input_site_ids": ["<site>"],
            "custom_param_text": None,
        },
        "provenance": {
            "action": "configure_quantum_corrections",
            "workflow_stage": "quantum_corrections",
            "actor": "user",
            "description": "Configured Quantum Corrections with 1 relaxed site(s)",
            "parameters": {
                "calculation_definition_id": "<definition>",
                "configuration": {
                    "singlepoint_job_ids_by_label": {"S1": "<job>"},
                    "phonon_job_ids_by_label": ["not", "a", "mapping"],
                },
                "output_directory": "quantum",
                "job_ids_by_label": {"S1": "<job>"},
            },
            "input_entity_ids": ["<host>", "<site>"],
            "output_entity_ids": ["<definition>"],
            "job_id": None,
        },
    },
    "neb": {
        "definition": {
            "workflow_stage": "energy_barriers",
            "calc_type": "castep_nudged_elastic_band_set",
            "parameters": {
                "configuration": {"method": "NEB"},
                "output_directory": "neb",
                "job_ids_by_label": {},
            },
            "input_structure_ids": ["<host>"],
            "input_site_ids": ["<site>"],
            "custom_param_text": None,
        },
        "provenance": {
            "action": "configure_energy_barriers",
            "workflow_stage": "energy_barriers",
            "actor": "user",
            "description": "Configured Energy Barriers NEB with 1 relaxed site(s)",
            "parameters": {
                "calculation_definition_id": "<definition>",
                "configuration": {"method": "NEB"},
                "output_directory": "neb",
                "job_ids_by_label": {},
            },
            "input_entity_ids": ["<host>", "<site>"],
            "output_entity_ids": ["<definition>"],
            "job_id": None,
        },
    },
}


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


def test_set_recommended_cutoff_records_stress_threshold_and_trims_text(tmp_path: Path) -> None:
    service, doc, _host = _document_with_host(tmp_path)

    service.set_recommended_cutoff(doc, 450.0, method="energy", stress_threshold=0.1, cell_id=" cellA ")

    assert doc.project.recommended_cutoff_stress_threshold == 0.1
    assert doc.project.recommended_cutoff_cell_id == "cellA"
    assert "using method 'energy'" in doc.provenance[-1].description

    service.set_recommended_cutoff(doc, 500.0, method="   ", cell_id="   ")

    assert doc.project.recommended_cutoff_method is None
    assert doc.project.recommended_cutoff_cell_id is None
    assert "using method" not in doc.provenance[-1].description


def test_set_recommended_kpoint_grid_ignores_a_malformed_lattice(tmp_path: Path) -> None:
    service, doc, _host = _document_with_host(tmp_path)

    service.set_recommended_kpoint_grid(doc, (2, 2, 2), lattice_vectors=[[1.0, 0.0, 0.0]])

    assert doc.project.recommended_kpoint_grid == (2, 2, 2)
    assert doc.project.recommended_kpoint_lattice is None


def test_load_project_drops_unparseable_recommended_values(tmp_path: Path) -> None:
    service, doc, _host = _document_with_host(tmp_path)
    service.save_project(doc)
    manifest_path = doc.project_root / "project.json"
    payload = json.loads(manifest_path.read_text())
    for key in (
        "recommended_cutoff_energy",
        "recommended_cutoff_energy_threshold",
        "recommended_cutoff_force_threshold",
        "recommended_cutoff_stress_threshold",
        "recommended_kpoint_energy_threshold",
        "recommended_kpoint_force_threshold",
        "recommended_kpoint_stress_threshold",
    ):
        payload[key] = "not-a-number"
    for key in (
        "recommended_cutoff_method",
        "recommended_cutoff_cell_id",
        "recommended_kpoint_method",
        "recommended_kpoint_cell_id",
    ):
        payload[key] = "   "
    payload["recommended_kpoint_grid"] = ["x", "y", "z"]
    payload["recommended_kpoint_lattice"] = [[1.0, 0.0, 0.0], "bad", [0.0, 0.0, 1.0]]
    manifest_path.write_text(json.dumps(payload))

    project = service.load_project(doc.project_root).project

    assert project.recommended_cutoff_energy is None
    assert project.recommended_cutoff_energy_threshold is None
    assert project.recommended_cutoff_method is None
    assert project.recommended_cutoff_cell_id is None
    assert project.recommended_kpoint_grid is None
    assert project.recommended_kpoint_lattice is None
    assert project.recommended_kpoint_method is None
    assert project.recommended_kpoint_cell_id is None


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


def test_register_quantum_phonon_definition_persists_configuration(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "quantum_definition.mufinder"
    doc = service.create_project(root, "QuantumDefinition")
    structure = service.import_structure_file(doc, Path("tests/golden/si_geom_opt.cell"), make_active=True)
    site = MuonSite(label="siteA", status="relaxed", source_structure_id=structure.id)
    doc.sites[site.id] = site

    definition = service.register_quantum_phonon_definition(
        doc,
        input_structure_ids=[structure.id],
        input_site_ids=[site.id],
        parameters={
            "method": "finite_displacement",
            "pseudopotential_family": "unknown",
            "sources": [{"site_id": str(site.id), "label": "siteA"}],
            "singlepoint_job_ids_by_label": {str(site.id): "sp-job"},
            "phonon_job_ids_by_label": {str(site.id): "ph-job"},
            "job_seed_names_by_label": {str(site.id): "quantum_siteA"},
        },
        output_directory=root / "files" / "quantum_corrections",
    )

    assert definition.id in doc.calculations
    assert doc.calculations[definition.id].workflow_stage == WorkflowStage.QUANTUM_CORRECTIONS
    assert doc.calculations[definition.id].input_site_ids == [site.id]
    assert doc.calculations[definition.id].parameters["configuration"]["method"] == "finite_displacement"
    assert doc.calculations[definition.id].parameters["job_ids_by_label"] == {}
    assert doc.calculations[definition.id].parameters["singlepoint_job_ids_by_label"] == {str(site.id): "sp-job"}
    assert doc.calculations[definition.id].parameters["phonon_job_ids_by_label"] == {str(site.id): "ph-job"}
    assert doc.calculations[definition.id].parameters["job_seed_names_by_label"] == {str(site.id): "quantum_siteA"}
    assert doc.provenance[-1].action == "configure_quantum_corrections"

    service.save_project(doc)
    loaded = service.load_project(root)
    loaded_definition = loaded.calculations[definition.id]
    assert loaded_definition.input_site_ids == [site.id]
    assert loaded_definition.parameters["configuration"]["sources"][0]["label"] == "siteA"
    assert loaded_definition.parameters["phonon_job_ids_by_label"] == {str(site.id): "ph-job"}


def test_register_nudged_elastic_band_definition_persists_configuration(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "energy_barriers_neb_definition.mufinder"
    doc = service.create_project(root, "EnergyBarriersNebDefinition")
    structure = service.import_structure_file(doc, Path("tests/golden/si_geom_opt.cell"), make_active=True)
    site = MuonSite(label="siteA", status="relaxed", source_structure_id=structure.id)
    doc.sites[site.id] = site

    definition = service.register_nudged_elastic_band_definition(
        doc,
        input_structure_ids=[structure.id],
        input_site_ids=[site.id],
        parameters={
            "method": "NEB",
            "sources": [{"site_id": str(site.id), "label": "siteA_to_siteB_path1_neb"}],
        },
        output_directory=root / "files" / "energy_barriers_neb",
    )

    assert definition.id in doc.calculations
    assert doc.calculations[definition.id].workflow_stage == WorkflowStage.ENERGY_BARRIERS
    assert doc.calculations[definition.id].calc_type == "castep_nudged_elastic_band_set"
    assert doc.calculations[definition.id].parameters["configuration"]["method"] == "NEB"
    assert doc.calculations[definition.id].parameters["job_ids_by_label"] == {}
    assert doc.provenance[-1].action == "configure_energy_barriers"

    service.save_project(doc)
    loaded = service.load_project(root)
    loaded_definition = loaded.calculations[definition.id]
    assert loaded_definition.parameters["configuration"]["sources"][0]["label"] == "siteA_to_siteB_path1_neb"


@pytest.mark.parametrize(
    "register",
    ["register_quantum_phonon_definition", "register_nudged_elastic_band_definition"],
)
def test_multi_input_definitions_validate_structures_and_sites(tmp_path: Path, register: str) -> None:
    service, doc, host = _document_with_host(tmp_path)
    site = MuonSite(label="S", status="staged", source_structure_id=host.id)
    doc.sites[site.id] = site
    register_definition = getattr(service, register)

    def _register(structure_ids, site_ids):
        return register_definition(
            doc,
            input_structure_ids=structure_ids,
            input_site_ids=site_ids,
            parameters={},
            output_directory=doc.project_root / "out",
        )

    with pytest.raises(ValueError, match="At least one input structure"):
        _register([], [site.id])
    with pytest.raises(ValueError, match="must reference known structures"):
        _register([uuid4()], [site.id])
    with pytest.raises(ValueError, match="must reference known sites"):
        _register([host.id], [uuid4()])


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


# ---------------------------------------------------------------------------
# Active host structure switching
# ---------------------------------------------------------------------------


def _make_doc(tmp_path: Path):
    service = ProjectService()
    doc = service.create_project(tmp_path / "proj.mufinder", "Test")
    return service, doc


def _supercell_structure(parent_id, stage: WorkflowStage = WorkflowStage.MUON_RELAXATION) -> Structure:
    return Structure(
        label="Supercell",
        origin="supercell",
        lattice_vectors=[[20.0, 0, 0], [0, 20.0, 0], [0, 0, 20.0]],
        species=["O", "Si", "O"],
        fractional_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=parent_id,
        properties={
            "supercell_asset": {
                "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                "owner_workflow_stage": stage.value,
                "source_asset_structure_id": None,
            }
        },
    )


def _completed_result(structure_id, convergence_achieved: bool | None = True) -> tuple[CalculationRun, Result]:
    run = CalculationRun(
        status=JobStatus.COMPLETED,
        convergence_achieved=convergence_achieved,
    )
    result = Result(
        calculation_run_id=run.id,
        result_type="castep_pristine_relaxation",
        derived_structure_ids=[structure_id],
    )
    run.result_id = result.id
    return run, result


# --- list_promotable_host_structures ---

def test_promotable_imported_structure_included(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)

    promotable = service.list_promotable_host_structures(doc)

    assert len(promotable) == 1
    assert promotable[0].structure_id == host.id
    assert promotable[0].kind == "imported"
    assert "Original" in promotable[0].label


def test_promotable_supercell_excluded(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    supercell = _supercell_structure(host.id)
    doc.structures[supercell.id] = supercell

    promotable = service.list_promotable_host_structures(doc)

    structure_ids = {h.structure_id for h in promotable}
    assert supercell.id not in structure_ids
    assert host.id in structure_ids


def test_promotable_completed_pristine_relaxation_included(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    relaxed = Structure(
        label="Fe2O3 relaxed",
        origin="pristine_relaxation",
        lattice_vectors=[[10.1, 0, 0], [0, 10.1, 0], [0, 0, 10.1]],
        species=["O", "Si", "O"],
        fractional_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )
    doc.structures[relaxed.id] = relaxed
    run, result = _completed_result(relaxed.id)
    doc.runs[run.id] = run
    doc.results[result.id] = result

    promotable = service.list_promotable_host_structures(doc)

    structure_ids = {h.structure_id for h in promotable}
    assert relaxed.id in structure_ids
    assert any(h.kind == "relaxed" for h in promotable)


def test_promotable_failed_relaxation_excluded(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    relaxed = Structure(
        label="Relaxed",
        origin="pristine_relaxation",
        lattice_vectors=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
        species=["O"],
        fractional_positions=[[0, 0, 0]],
    )
    doc.structures[relaxed.id] = relaxed
    run = CalculationRun(status=JobStatus.FAILED)
    result = Result(calculation_run_id=run.id, derived_structure_ids=[relaxed.id])
    doc.runs[run.id] = run
    doc.results[result.id] = result

    promotable = service.list_promotable_host_structures(doc)

    structure_ids = {h.structure_id for h in promotable}
    assert relaxed.id not in structure_ids


def test_promotable_non_converged_relaxation_excluded(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    relaxed = Structure(
        label="Relaxed NC",
        origin="pristine_relaxation",
        lattice_vectors=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
        species=["O"],
        fractional_positions=[[0, 0, 0]],
    )
    doc.structures[relaxed.id] = relaxed
    run, result = _completed_result(relaxed.id, convergence_achieved=False)
    doc.runs[run.id] = run
    doc.results[result.id] = result

    promotable = service.list_promotable_host_structures(doc)

    structure_ids = {h.structure_id for h in promotable}
    assert relaxed.id not in structure_ids


def test_promotable_muon_relaxation_origin_excluded(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    muoniated = Structure(
        label="Muoniated",
        origin="muon_relaxation",
        lattice_vectors=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
        species=["O", "H"],
        fractional_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    doc.structures[muoniated.id] = muoniated
    run, result = _completed_result(muoniated.id)
    doc.runs[run.id] = run
    doc.results[result.id] = result

    promotable = service.list_promotable_host_structures(doc)

    structure_ids = {h.structure_id for h in promotable}
    assert muoniated.id not in structure_ids


def test_promotable_order_imported_then_relaxed(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    relaxed = Structure(
        label="Relaxed",
        origin="pristine_relaxation",
        lattice_vectors=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
        species=["O"],
        fractional_positions=[[0, 0, 0]],
    )
    doc.structures[relaxed.id] = relaxed
    run, result = _completed_result(relaxed.id)
    doc.runs[run.id] = run
    doc.results[result.id] = result

    promotable = service.list_promotable_host_structures(doc)

    assert len(promotable) == 2
    assert promotable[0].kind == "imported"
    assert promotable[1].kind == "relaxed"


def test_promotable_label_includes_job_label(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    job = Job(label="Fe2O3 pristine #1", workflow_stage=WorkflowStage.PRISTINE_RELAXATION, calc_type="castep_pristine_relaxation")
    doc.jobs[job.id] = job
    relaxed = Structure(
        label="Relaxed",
        origin="pristine_relaxation",
        lattice_vectors=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
        species=["O"],
        fractional_positions=[[0, 0, 0]],
        properties={"source_job_id": str(job.id)},
    )
    doc.structures[relaxed.id] = relaxed
    run, result = _completed_result(relaxed.id)
    doc.runs[run.id] = run
    doc.results[result.id] = result

    promotable = service.list_promotable_host_structures(doc)

    relaxed_entry = next(h for h in promotable if h.kind == "relaxed")
    assert "Fe2O3 pristine #1" in relaxed_entry.label


# --- set_active_host_structure ---

def test_set_active_host_switches_structure(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    original = silicon_oxide_host()
    service.add_structure(doc, original)
    relaxed = Structure(
        label="Relaxed host",
        origin="pristine_relaxation",
        lattice_vectors=[[11.0, 0, 0], [0, 11.0, 0], [0, 0, 11.0]],
        species=["O"],
        fractional_positions=[[0, 0, 0]],
    )
    doc.structures[relaxed.id] = relaxed
    run, result = _completed_result(relaxed.id)
    doc.runs[run.id] = run
    doc.results[result.id] = result

    service.set_active_host_structure(doc, relaxed.id)

    assert doc.project.host_structure_id == relaxed.id


def test_set_active_host_no_op_when_already_active(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    initial_provenance_count = len(doc.provenance)

    service.set_active_host_structure(doc, host.id)

    assert doc.project.host_structure_id == host.id
    assert len(doc.provenance) == initial_provenance_count  # no new provenance record


def test_set_active_host_rejects_non_promotable(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    supercell = _supercell_structure(host.id)
    doc.structures[supercell.id] = supercell

    with pytest.raises(ValueError, match="not eligible"):
        service.set_active_host_structure(doc, supercell.id)


def test_set_active_host_appends_provenance(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    original = silicon_oxide_host()
    service.add_structure(doc, original)
    relaxed = Structure(
        label="Relaxed",
        origin="pristine_relaxation",
        lattice_vectors=[[11.0, 0, 0], [0, 11.0, 0], [0, 0, 11.0]],
        species=["O"],
        fractional_positions=[[0, 0, 0]],
    )
    doc.structures[relaxed.id] = relaxed
    run, result = _completed_result(relaxed.id)
    doc.runs[run.id] = run
    doc.results[result.id] = result
    before = len(doc.provenance)

    service.set_active_host_structure(doc, relaxed.id)

    assert len(doc.provenance) == before + 1
    record = doc.provenance[-1]
    assert record.action == "set_active_host_structure"
    assert str(relaxed.id) in record.parameters.get("new_host_structure_id", "")


def test_set_active_host_rebinds_magnetic_structure(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    original = Structure(
        label="Magnetic host",
        origin="imported",
        lattice_vectors=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
        species=["Fe"],
        fractional_positions=[[0, 0, 0]],
        properties={"magnetic_structure": {"moments": [{"vector": [0, 0, 2.0]}]}},
    )
    service.add_structure(doc, original)
    doc.project.magnetic_structure = original.properties["magnetic_structure"]

    relaxed = Structure(
        label="Relaxed",
        origin="pristine_relaxation",
        lattice_vectors=[[10.1, 0, 0], [0, 10.1, 0], [0, 0, 10.1]],
        species=["Fe"],
        fractional_positions=[[0, 0, 0]],
    )
    doc.structures[relaxed.id] = relaxed
    run, result = _completed_result(relaxed.id)
    doc.runs[run.id] = run
    doc.results[result.id] = result

    service.set_active_host_structure(doc, relaxed.id)

    assert doc.project.magnetic_structure is None


def test_set_active_host_preserves_magnetic_if_new_host_has_it(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    original = silicon_oxide_host()
    service.add_structure(doc, original)
    magnetic_data = {"moments": [{"vector": [0, 0, 2.0]}]}
    relaxed = Structure(
        label="Magnetic relaxed",
        origin="pristine_relaxation",
        lattice_vectors=[[10.1, 0, 0], [0, 10.1, 0], [0, 0, 10.1]],
        species=["Fe"],
        fractional_positions=[[0, 0, 0]],
        properties={"magnetic_structure": magnetic_data},
    )
    doc.structures[relaxed.id] = relaxed
    run, result = _completed_result(relaxed.id)
    doc.runs[run.id] = run
    doc.results[result.id] = result

    service.set_active_host_structure(doc, relaxed.id)

    assert doc.project.magnetic_structure == magnetic_data


# --- resolve_underlying_host ---

def test_resolve_host_for_imported_structure(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)

    assert service.resolve_underlying_host(doc, host.id) == host.id


def test_resolve_host_for_pristine_relaxation(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    relaxed = Structure(
        label="Relaxed",
        origin="pristine_relaxation",
        lattice_vectors=[[10.1, 0, 0], [0, 10.1, 0], [0, 0, 10.1]],
        species=["O"],
        fractional_positions=[[0, 0, 0]],
    )
    doc.structures[relaxed.id] = relaxed

    assert service.resolve_underlying_host(doc, relaxed.id) == relaxed.id


def test_resolve_host_through_supercell(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    supercell = _supercell_structure(host.id)
    doc.structures[supercell.id] = supercell

    assert service.resolve_underlying_host(doc, supercell.id) == host.id


def test_resolve_host_through_muon_relaxation_output(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    supercell = _supercell_structure(host.id)
    doc.structures[supercell.id] = supercell
    muoniated = Structure(
        label="Muoniated",
        origin="muon_relaxation",
        lattice_vectors=[[20.0, 0, 0], [0, 20.0, 0], [0, 0, 20.0]],
        species=["O", "H"],
        fractional_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        parent_structure_id=supercell.id,
    )
    doc.structures[muoniated.id] = muoniated

    assert service.resolve_underlying_host(doc, muoniated.id) == host.id


def test_resolve_host_broken_chain_returns_none(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    orphan_parent_id = uuid4()
    muoniated = Structure(
        label="Orphan",
        origin="muon_relaxation",
        lattice_vectors=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
        species=["O"],
        fractional_positions=[[0, 0, 0]],
        parent_structure_id=orphan_parent_id,
    )
    doc.structures[muoniated.id] = muoniated

    assert service.resolve_underlying_host(doc, muoniated.id) is None


def test_resolve_host_cycle_returns_none(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    a = Structure(label="A", origin="muon_relaxation", lattice_vectors=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]], species=["O"], fractional_positions=[[0, 0, 0]])
    b = Structure(label="B", origin="muon_relaxation", lattice_vectors=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]], species=["O"], fractional_positions=[[0, 0, 0]])
    a.parent_structure_id = b.id
    b.parent_structure_id = a.id
    doc.structures[a.id] = a
    doc.structures[b.id] = b

    assert service.resolve_underlying_host(doc, a.id) is None


def test_resolve_host_unknown_structure_returns_none(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)

    assert service.resolve_underlying_host(doc, uuid4()) is None


# --- validate_sites_match_active_host ---

def test_validate_sites_no_mismatch(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    site = MuonSite(label="S1", status="generated", source_structure_id=host.id)
    doc.sites[site.id] = site

    assert service.validate_sites_match_active_host(doc, [site.id]) is None


def test_validate_sites_mismatch_detected(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    other_host = Structure(label="Other", origin="imported", lattice_vectors=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]], species=["C"], fractional_positions=[[0, 0, 0]])
    doc.structures[other_host.id] = other_host
    site = MuonSite(label="S1", status="generated", source_structure_id=other_host.id)
    doc.sites[site.id] = site

    mismatch = service.validate_sites_match_active_host(doc, [site.id])

    assert isinstance(mismatch, SiteHostMismatch)
    assert mismatch.active_host_id == host.id
    assert len(mismatch.offenders) == 1
    offender_site_id, _, offender_source_id, _ = mismatch.offenders[0]
    assert offender_site_id == site.id
    assert offender_source_id == other_host.id


def test_validate_sites_skips_unknown_site_ids(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)

    assert service.validate_sites_match_active_host(doc, [uuid4()]) is None


def test_validate_sites_no_active_host_returns_none(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    site = MuonSite(label="S1", status="generated", source_structure_id=uuid4())
    doc.sites[site.id] = site

    assert service.validate_sites_match_active_host(doc, [site.id]) is None


# --- validate_structures_match_active_host ---

def test_validate_structures_no_mismatch(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host = silicon_oxide_host()
    service.add_structure(doc, host)
    supercell = _supercell_structure(host.id)
    doc.structures[supercell.id] = supercell
    muoniated = Structure(
        label="Muoniated",
        origin="muon_relaxation",
        lattice_vectors=[[20.0, 0, 0], [0, 20.0, 0], [0, 0, 20.0]],
        species=["O", "H"],
        fractional_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        parent_structure_id=supercell.id,
    )
    doc.structures[muoniated.id] = muoniated

    assert service.validate_structures_match_active_host(doc, [muoniated.id]) is None


def test_validate_structures_mismatch_detected(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)
    host_a = silicon_oxide_host()
    service.add_structure(doc, host_a)
    host_b = Structure(
        label="Host B",
        origin="imported",
        lattice_vectors=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        species=["C"],
        fractional_positions=[[0, 0, 0]],
    )
    doc.structures[host_b.id] = host_b
    supercell_b = _supercell_structure(host_b.id)
    doc.structures[supercell_b.id] = supercell_b
    muoniated_b = Structure(
        label="Muoniated B",
        origin="muon_relaxation",
        lattice_vectors=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
        species=["C", "H"],
        fractional_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        parent_structure_id=supercell_b.id,
    )
    doc.structures[muoniated_b.id] = muoniated_b

    mismatch = service.validate_structures_match_active_host(doc, [muoniated_b.id])

    assert isinstance(mismatch, StructureHostMismatch)
    assert mismatch.active_host_id == host_a.id
    assert len(mismatch.offenders) == 1
    offender_struct_id, _, resolved_host_id, _ = mismatch.offenders[0]
    assert offender_struct_id == muoniated_b.id
    assert resolved_host_id == host_b.id


def test_validate_structures_no_active_host_returns_none(tmp_path: Path) -> None:
    service, doc = _make_doc(tmp_path)

    assert service.validate_structures_match_active_host(doc, [uuid4()]) is None


def test_recommended_xc_functional_follows_adopted_convergence_and_persists(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "recommended_xc.mufinder"
    doc = service.create_project(root, "RecommendedXC")
    assert doc.project.recommended_xc_functional is None

    # Adopting a convergence result makes its functional the project default.
    service.set_recommended_cutoff(doc, 550.0, xc_functional="pbe")
    assert doc.project.recommended_cutoff_xc_functional == "PBE"
    assert doc.project.recommended_xc_functional == "PBE"
    assert doc.provenance[-1].parameters["recommended_cutoff_xc_functional"] == "PBE"

    # A sweep with no single functional records nothing and leaves the default alone.
    service.set_recommended_kpoint_grid(doc, (4, 4, 4), xc_functional=None)
    assert doc.project.recommended_kpoint_xc_functional is None
    assert doc.project.recommended_xc_functional == "PBE"

    service.set_recommended_kpoint_grid(doc, (4, 4, 4), xc_functional="LDA")
    assert doc.project.recommended_kpoint_xc_functional == "LDA"
    assert doc.project.recommended_xc_functional == "LDA"

    # Editing the default directly keeps what each value was converged with.
    service.set_recommended_xc_functional(doc, " pbesol ")
    assert doc.project.recommended_xc_functional == "PBESOL"
    assert doc.project.recommended_cutoff_xc_functional == "PBE"
    assert doc.project.recommended_kpoint_xc_functional == "LDA"
    assert doc.provenance[-1].action == "set_recommended_xc_functional"
    assert doc.provenance[-1].parameters == {"recommended_xc_functional": "PBESOL"}

    service.save_project(doc)
    loaded = service.load_project(root)
    assert loaded.project.recommended_xc_functional == "PBESOL"
    assert loaded.project.recommended_cutoff_xc_functional == "PBE"
    assert loaded.project.recommended_kpoint_xc_functional == "LDA"

    service.set_recommended_xc_functional(doc, None)
    assert doc.project.recommended_xc_functional is None
    assert "Cleared" in doc.provenance[-1].description


def test_clear_recommended_defaults_removes_value_and_metadata(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "clear_defaults.mufinder"
    doc = service.create_project(root, "ClearDefaults")
    service.set_recommended_cutoff(
        doc, 550.0, method="asymptotic", energy_threshold=0.001, cell_id="c", xc_functional="PBE"
    )
    service.set_recommended_kpoint_grid(
        doc,
        (4, 4, 4),
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        method="delta-threshold",
        xc_functional="PBE",
    )

    service.clear_recommended_cutoff(doc)
    assert doc.provenance[-1].action == "clear_recommended_cutoff"
    service.clear_recommended_kpoint_grid(doc)
    assert doc.provenance[-1].action == "clear_recommended_kpoint_grid"

    project = doc.project
    assert project.recommended_cutoff_energy is None
    assert project.recommended_cutoff_method is None
    assert project.recommended_cutoff_energy_threshold is None
    assert project.recommended_cutoff_cell_id is None
    assert project.recommended_cutoff_xc_functional is None
    assert project.recommended_kpoint_grid is None
    assert project.recommended_kpoint_lattice is None
    assert project.recommended_kpoint_method is None
    assert project.recommended_kpoint_xc_functional is None
    # The functional is its own default and survives.
    assert project.recommended_xc_functional == "PBE"

    service.save_project(doc)
    loaded = service.load_project(root)
    assert loaded.project.recommended_cutoff_energy is None
    assert loaded.project.recommended_kpoint_grid is None


def test_failure_reason_is_recorded_cleared_and_persisted(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "failure_reason.mufinder"
    doc = service.create_project(root, "FailureReason")
    job = service.add_job(
        doc,
        label="Timeout",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    service.submit_job(job)
    service.start_job(job)
    service.stop_job(job, "walltime", reason=FailureReason.WALLTIME)
    service.record_job_terminal_event(doc, action="job_stopped", job=job, error_message="walltime")

    assert doc.provenance[-1].parameters["failure_reason"] == "walltime"
    service.save_project(doc)
    assert service.load_project(root).jobs[job.id].failure_reason == FailureReason.WALLTIME

    service.submit_job(job)
    assert job.failure_reason is None
    service.start_job(job)
    service.fail_job(job, "oom", reason=FailureReason.OUT_OF_MEMORY)
    service.reset_failed_job(job)
    assert job.failure_reason is None


def _definition_at(
    document, stage: WorkflowStage, *, calc_type: str = "", day: int = 1
) -> CalculationDefinition:
    definition = CalculationDefinition(
        workflow_stage=stage,
        calc_type=calc_type,
        created_at=datetime(2025, 1, day, tzinfo=UTC),
    )
    document.calculations[definition.id] = definition
    return definition


def test_definitions_for_stage_lists_matching_definitions_newest_first(tmp_path: Path) -> None:
    doc = ProjectService().create_project(tmp_path / "defs.mufinder", "Defs")
    older = _definition_at(doc, WorkflowStage.UEP, day=1)
    newer = _definition_at(doc, WorkflowStage.UEP, day=3)
    _definition_at(doc, WorkflowStage.PRISTINE_RELAXATION, day=5)

    assert definitions_for_stage(doc, WorkflowStage.UEP) == [newer, older]
    assert definitions_for_stage(doc, WorkflowStage.CONVERGENCE_CUTOFF) == []


def test_definitions_for_stage_filters_on_calc_type(tmp_path: Path) -> None:
    doc = ProjectService().create_project(tmp_path / "defs.mufinder", "Defs")
    wanted = _definition_at(doc, WorkflowStage.ENERGY_BARRIERS, calc_type="neb_set", day=1)
    _definition_at(doc, WorkflowStage.ENERGY_BARRIERS, calc_type="ts_set", day=2)

    assert definitions_for_stage(doc, WorkflowStage.ENERGY_BARRIERS, calc_type="neb_set") == [wanted]
    assert len(definitions_for_stage(doc, WorkflowStage.ENERGY_BARRIERS)) == 2


def test_active_definition_prefers_the_ui_state_selection(tmp_path: Path) -> None:
    doc = ProjectService().create_project(tmp_path / "active.mufinder", "Active")
    older = _definition_at(doc, WorkflowStage.UEP, day=1)
    _definition_at(doc, WorkflowStage.UEP, day=2)
    doc.ui_state["active_uep"] = str(older.id)

    assert active_definition(doc, WorkflowStage.UEP, active_key="active_uep") is older


@pytest.mark.parametrize(
    "selection", ["not-a-uuid", "00000000-0000-4000-8000-000000000000", 7, "other-stage", "other-calc-type"]
)
def test_active_definition_falls_back_to_the_newest_match(tmp_path: Path, selection: object) -> None:
    doc = ProjectService().create_project(tmp_path / "active.mufinder", "Active")
    _definition_at(doc, WorkflowStage.MUON_RELAXATION, calc_type="muon_set", day=1)
    newest = _definition_at(doc, WorkflowStage.MUON_RELAXATION, calc_type="muon_set", day=2)
    other_stage = _definition_at(doc, WorkflowStage.UEP, day=9)
    other_calc_type = _definition_at(doc, WorkflowStage.MUON_RELAXATION, calc_type="other", day=9)
    doc.ui_state["active_muon"] = {
        "other-stage": str(other_stage.id),
        "other-calc-type": str(other_calc_type.id),
    }.get(selection, selection)

    resolved = active_definition(
        doc, WorkflowStage.MUON_RELAXATION, calc_type="muon_set", active_key="active_muon"
    )

    assert resolved is newest


def test_active_definition_without_a_key_or_match(tmp_path: Path) -> None:
    doc = ProjectService().create_project(tmp_path / "active.mufinder", "Active")
    first = _definition_at(doc, WorkflowStage.QUANTUM_CORRECTIONS, calc_type="phonon_set", day=1)
    doc.ui_state["ignored"] = str(uuid4())

    assert active_definition(doc, WorkflowStage.QUANTUM_CORRECTIONS, calc_type="phonon_set") is first
    assert active_definition(doc, WorkflowStage.UEP, active_key="ignored") is None


def test_job_for_mapping_key_resolves_the_mapped_job(tmp_path: Path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "jobs.mufinder", "Jobs")
    job = service.add_job(doc, label="a", workflow_stage=WorkflowStage.UEP, calc_type="castep_uep_source")
    definition = _definition_at(doc, WorkflowStage.UEP)
    definition.parameters["job_ids_by_label"] = {"a": str(job.id)}
    definition.parameters["job_ids_by_cutoff"] = {"300": str(job.id)}

    assert job_for_mapping_key(doc, definition, "a") is job
    assert job_for_mapping_key(doc, definition, "300", mapping_name="job_ids_by_cutoff") is job
    assert job_for_mapping_key(doc, definition, "a", calc_type="castep_uep_source") is job


@pytest.mark.parametrize(
    "case",
    ["no-definition", "no-mapping", "not-a-mapping", "unmapped", "non-string", "bad-uuid", "missing-job", "calc-type"],
)
def test_job_for_mapping_key_returns_none_when_unresolvable(tmp_path: Path, case: str) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "jobs.mufinder", "Jobs")
    job = service.add_job(doc, label="a", workflow_stage=WorkflowStage.UEP, calc_type="castep_uep_source")
    definition = _definition_at(doc, WorkflowStage.UEP)
    mapping: object = {
        "no-mapping": None,
        "not-a-mapping": [str(job.id)],
        "unmapped": {"b": str(job.id)},
        "non-string": {"a": 42},
        "bad-uuid": {"a": "not-a-uuid"},
        "missing-job": {"a": str(uuid4())},
    }.get(case, {"a": str(job.id)})
    if mapping is not None:
        definition.parameters["job_ids_by_label"] = mapping

    resolved = job_for_mapping_key(
        doc,
        None if case == "no-definition" else definition,
        "a",
        calc_type="castep_other" if case == "calc-type" else None,
    )

    assert resolved is None
