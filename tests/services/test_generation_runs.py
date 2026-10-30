"""Tests for the plain-language descriptions of generation runs and their groups of starts."""

from __future__ import annotations

from pathlib import Path

from mufinder.domain.models import GenerationRun, MuonSite, Project
from mufinder.services import generation_runs, site_generation_service
from mufinder.services.project_document import ProjectDocument
from tests.support.symmetry_hosts import cof2_host


def _random_run(**changes: object) -> GenerationRun:
    fields: dict[str, object] = {
        "method": "random",
        "parameters": {
            "selection_mode": "count",
            "count": 6,
            "min_muon_muon": 1.7912,
            "estimated_min_muon_muon": 1.99,
            "min_muon_atom": 1.0,
            "max_attempts": 20000,
            "include_symmetry_equivalents": True,
        },
        "seed": 418273,
        "uses_symmetry": True,
        "symmetry_tolerance_ang": 1e-3,
        "space_group_symbol": "P4_2/mnm",
        "space_group_number": 136,
        "symmetry_operation_count": 16,
        "site_ids": [MuonSite().id for _ in range(6)],
        "distinct_site_count": 6,
        "attempts_used": 314,
        "retries": 1,
    }
    fields.update(changes)
    return GenerationRun(**fields)  # type: ignore[arg-type]


def test_a_random_run_summary_states_its_parameters_in_one_line() -> None:
    assert generation_runs.run_summary_line(_random_run()) == (
        "Random, 6 starts, ≥ 1.79 Å apart, ≥ 1.0 Å from atoms, P4₂/mnm (16 ops)"
    )


def test_the_summary_says_when_symmetry_was_not_used_or_not_detected() -> None:
    off = _random_run(uses_symmetry=False)
    p1 = _random_run(space_group_symbol=None, space_group_number=None, symmetry_operation_count=1)

    assert generation_runs.run_summary_line(off).endswith(", symmetry not used")
    assert generation_runs.run_summary_line(p1).endswith(", no symmetry detected (P1)")


def test_the_journal_line_adds_the_seed() -> None:
    assert generation_runs.run_journal_line(_random_run()) == (
        "Random, 6 starts, ≥ 1.79 Å apart, ≥ 1.0 Å from atoms, P4₂/mnm (16 ops), seed 418273."
    )


def test_the_methods_sentence_quotes_every_recorded_input() -> None:
    sentence = generation_runs.run_methods_sentence(_random_run(), host_label="CoF2")

    assert sentence == (
        "MuFinder generated 6 initial muon positions at random in CoF2, each at least 1.79 Å "
        "from the other starts and their symmetry images and at least 1.0 Å from every host "
        "atom. The separation was estimated as 1.99 Å from the requested 6 starts and reduced to "
        "1.79 Å after 1 retry so that they fitted. 314 trial positions were drawn (limit 20000). "
        "Symmetry images were taken from the P4₂/mnm space group (16 operations) detected by "
        "spglib at a tolerance of 0.001 Å. The random seed was 418273."
    )


def test_attached_manual_and_uep_runs_are_described() -> None:
    attached = GenerationRun(
        method="near_element",
        parameters={
            "element": "F",
            "muons_per_atom": 1,
            "bond_distance": 1.0,
            "min_muon_atom": 0.8,
        },
        uses_symmetry=True,
        symmetry_tolerance_ang=1e-3,
        space_group_symbol="P4_2/mnm",
        symmetry_operation_count=16,
        site_ids=[MuonSite().id],
    )
    manual = GenerationRun(
        method="manual",
        parameters={"coord_system": "fractional", "coordinates": [0.25, 0.25, 0.5]},
        site_ids=[MuonSite().id],
    )
    uep = GenerationRun(
        method="uep_clusters",
        parameters={
            "source_jobs": [{"id": "1", "label": "UEP search"}],
            "grouping": {"clustering_method": "kmeans", "clustering_kmeans_k": 4},
        },
        site_ids=[MuonSite().id, MuonSite().id],
    )

    assert generation_runs.run_summary_line(attached) == (
        "Attached to F, 1 start, 1 per atom, 1.0 Å from F, ≥ 0.8 Å from atoms, P4₂/mnm (16 ops)"
    )
    assert generation_runs.run_summary_line(manual) == "Custom coordinates, 1 start, fractional"
    assert generation_runs.run_summary_line(uep) == "UEP clusters, 2 starts, from 'UEP search'"
    assert "1.0 Å from F" in generation_runs.run_methods_sentence(attached)
    assert generation_runs.run_methods_sentence(manual, host_label="CoF2") == (
        "The initial muon position in CoF2 was entered by hand at fractional coordinates "
        "(0.25, 0.25, 0.5)."
    )
    assert "grouped by k-means clustering (k = 4)" in generation_runs.run_methods_sentence(uep)


def _document(tmp_path: Path) -> ProjectDocument:
    host = cof2_host()
    document = ProjectDocument(
        project_root=tmp_path, project=Project(name="Runs", host_structure_id=host.id)
    )
    document.structures[host.id] = host
    return document


def test_starts_are_grouped_by_run_after_the_earlier_starts(tmp_path: Path) -> None:
    document = _document(tmp_path)
    earlier = MuonSite(
        label="Random Site 9",
        generation_method="random",
        properties={"min_muon_muon": 0.8, "min_muon_atom": 1.0, "selection_mode": "separation"},
    )
    document.sites[earlier.id] = earlier
    attached = site_generation_service.generate_candidate_sites(
        document,
        method="near_element",
        parameters={"element": "F", "muons_per_atom": 2, "bond_distance": 1.0},
    )
    manual = site_generation_service.generate_candidate_sites(
        document, method="manual", parameters={"x": 0.1, "y": 0.2, "z": 0.3}
    )

    groups = generation_runs.generation_run_groups(document, document.sites.values())

    assert [group.title for group in groups] == [
        "Earlier starts: Random, 1 start, ≥ 0.8 Å apart, ≥ 1.0 Å from atoms",
        generation_runs.run_summary_line(attached.run),
        "Custom coordinates, 1 start, fractional",
    ]
    assert [group.run_id for group in groups] == [None, attached.run.id, manual.run.id]
    assert groups[0].site_ids == (earlier.id,)
    assert groups[0].description.endswith("The seed and the host symmetry were not recorded.")
    assert groups[1].site_ids == tuple(site.id for site in attached.sites)
    assert "in CoF2" in groups[1].description


def test_earlier_starts_with_different_inputs_get_their_own_groups(tmp_path: Path) -> None:
    document = _document(tmp_path)
    for label, element in (("Attached-F Site 1", "F"), ("Attached-Co Site 1", "Co")):
        site = MuonSite(
            label=label,
            generation_method="near_element",
            properties={"target_element": element, "muons_per_atom": 1, "bond_distance": 1.0},
        )
        document.sites[site.id] = site

    groups = generation_runs.generation_run_groups(document, document.sites.values())

    assert sorted(group.title for group in groups) == [
        "Earlier starts: Attached to Co, 1 start, 1 per atom, 1.0 Å from Co",
        "Earlier starts: Attached to F, 1 start, 1 per atom, 1.0 Å from F",
    ]
