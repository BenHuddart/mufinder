"""Tests for candidate-site generation outcomes on a project document.

``tests/services/test_project_service.py`` covers each generation method
through ``ProjectService``; these tests cover the module contract: the
document changes in memory and the provenance comes back to the caller.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.domain.models import MuonSite, Project, Structure
from mufinder.science.site_generation import SiteGenerationError
from mufinder.services import site_generation_service
from mufinder.services.project_document import ProjectDocument
from tests.support.structures import silicon_oxide_host
from tests.support.symmetry_hosts import cof2_host


def _document(tmp_path: Path, host: Structure | None = None) -> ProjectDocument:
    host = host or silicon_oxide_host()
    document = ProjectDocument(
        project_root=tmp_path, project=Project(name="Sites", host_structure_id=host.id)
    )
    document.structures[host.id] = host
    return document


def test_generation_adds_sites_and_returns_provenance_without_recording_it(tmp_path: Path) -> None:
    document = _document(tmp_path)

    outcome = site_generation_service.generate_candidate_sites(
        document, method="manual", parameters={"x": 0.1, "y": 0.2, "z": 0.3}
    )

    assert [site.id for site in outcome.sites] == list(document.sites)
    assert outcome.sites[0].label == "Manual Site 1"
    assert outcome.sites[0].status == site_generation_service.GENERATED_SITE_STATUS
    assert [record.action for record in outcome.provenance] == ["generate_candidates"]
    assert outcome.provenance[0].output_entity_ids == [outcome.sites[0].id]
    assert document.provenance == []


def test_generation_short_of_the_requested_count_adds_a_partial_record(tmp_path: Path) -> None:
    document = _document(tmp_path)
    present = {
        "source_key": "job-1:cluster:0",
        "fractional_position": [0.1, 0.2, 0.3],
        "cartesian_position": [1, 2, 3],
    }
    new = {
        "source_key": "job-1:cluster:1",
        "fractional_position": [0.6, 0.2, 0.3],
        "cartesian_position": [6, 2, 3],
    }
    site_generation_service.generate_candidate_sites(
        document, method="uep_clusters", parameters={"selections": [present]}
    )

    outcome = site_generation_service.generate_candidate_sites(
        document, method="uep_clusters", parameters={"selections": [present, new]}
    )

    assert [site.properties["uep_source_key"] for site in outcome.sites] == ["job-1:cluster:1"]
    assert outcome.sites[0].label == "UEP Cluster Site 2"
    assert [record.action for record in outcome.provenance] == [
        "generate_candidates",
        "generation_partial",
    ]
    partial = outcome.provenance[1]
    assert partial.parameters == {"requested": 2, "generated": 1, "attempts": 1}
    assert (
        partial.description
        == "Requested 2 sites but generated 1. Some selected UEP rows were already present."
    )


def test_attached_generation_asks_for_starts_on_one_atom_per_equivalent_set(
    tmp_path: Path,
) -> None:
    document = _document(tmp_path, cof2_host())
    parameters = {"element": "F", "muons_per_atom": 2, "bond_distance": 1.0, "min_muon_atom": 0.8}

    requested = site_generation_service.requested_site_count(
        document, method="near_element", parameters=parameters
    )
    outcome = site_generation_service.generate_candidate_sites(
        document, method="near_element", parameters=parameters
    )

    assert requested == 2
    assert len(outcome.sites) == 2
    assert [record.action for record in outcome.provenance] == ["generate_candidates"]


def test_generation_records_a_warning_when_host_symmetry_is_not_detected(
    tmp_path: Path,
) -> None:
    document = _document(tmp_path, cof2_host(cobalt_label="Xx"))

    outcome = site_generation_service.generate_candidate_sites(
        document,
        method="random",
        parameters={
            "selection_mode": "separation",
            "min_muon_muon": 1.5,
            "min_muon_atom": 1.0,
            "max_attempts": 200,
            "seed": 1,
            "include_symmetry_equivalents": True,
        },
    )

    assert [record.action for record in outcome.provenance] == [
        "generate_candidates",
        "symmetry_unavailable",
    ]
    warning = outcome.provenance[1]
    assert "'Xx' name no element" in warning.description
    assert "(P1)" in warning.description
    assert warning.actor == "system"
    assert warning.output_entity_ids == [site.id for site in outcome.sites]


def test_generation_without_symmetry_records_no_symmetry_warning(tmp_path: Path) -> None:
    document = _document(tmp_path, cof2_host(cobalt_label="Xx"))

    outcome = site_generation_service.generate_candidate_sites(
        document,
        method="random",
        parameters={
            "selection_mode": "separation",
            "min_muon_muon": 1.5,
            "min_muon_atom": 1.0,
            "max_attempts": 200,
            "seed": 1,
            "include_symmetry_equivalents": False,
        },
    )

    assert [record.action for record in outcome.provenance] == ["generate_candidates"]


def test_random_count_generation_shrinks_the_separation_until_sites_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _document(tmp_path)
    real_generate = site_generation_service.generate_random_sites
    separations: list[float] = []

    def fail_twice(**kwargs: object):
        separations.append(float(kwargs["min_muon_muon"]))
        if len(separations) < 3:
            raise SiteGenerationError("too tight")
        return real_generate(**kwargs)

    monkeypatch.setattr(site_generation_service, "generate_random_sites", fail_twice)

    outcome = site_generation_service.generate_candidate_sites(
        document, method="random", parameters={"selection_mode": "count", "count": 2, "seed": 1}
    )

    assert separations[1] == pytest.approx(separations[0] * 0.9)
    assert separations[2] == pytest.approx(separations[0] * 0.81)
    assert outcome.sites[0].properties["min_muon_muon"] == pytest.approx(separations[2])
    assert outcome.sites[0].properties["estimated_min_muon_muon"] == pytest.approx(separations[0])


def test_random_count_generation_raises_the_last_error_after_every_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _document(tmp_path)
    calls: list[int] = []

    def always_fail(**_kwargs: object):
        calls.append(1)
        raise SiteGenerationError(f"attempt {len(calls)}")

    monkeypatch.setattr(site_generation_service, "generate_random_sites", always_fail)

    with pytest.raises(SiteGenerationError, match="attempt 6"):
        site_generation_service.generate_candidate_sites(
            document, method="random", parameters={"selection_mode": "count", "count": 2}
        )
    assert document.sites == {}


def test_cluster_returns_moved_sites_and_one_record(tmp_path: Path) -> None:
    document = _document(tmp_path)
    site = MuonSite(
        label="Random Site 1", status="generated", initial_fractional_position=[0.9, 0.1, 0.1]
    )
    document.sites[site.id] = site

    outcome = site_generation_service.cluster_candidate_positions(
        document, [site.id], anchor_fractional=[0.1, 0.1, 0.1]
    )

    assert outcome.sites == [site]
    assert [record.action for record in outcome.provenance] == ["cluster_candidate_positions"]
    assert document.provenance == []


def test_remove_and_stage_return_no_provenance_when_nothing_matches(tmp_path: Path) -> None:
    document = _document(tmp_path)
    relaxed = MuonSite(label="Relaxed", status="relaxed")
    document.sites[relaxed.id] = relaxed

    removed = site_generation_service.remove_candidate_sites(document, [relaxed.id])
    staged = site_generation_service.stage_generated_sites(document, [relaxed.id])

    assert removed.sites == [] and removed.provenance == []
    assert staged.sites == [] and staged.provenance == []
    assert document.sites == {relaxed.id: relaxed}
    assert relaxed.status == "relaxed"


def test_remove_and_stage_accept_the_legacy_candidate_status(tmp_path: Path) -> None:
    document = _document(tmp_path)
    to_stage = MuonSite(label="Old 1", status="candidate")
    to_remove = MuonSite(label="Old 2", status="candidate")
    document.sites.update({to_stage.id: to_stage, to_remove.id: to_remove})

    staged = site_generation_service.stage_generated_sites(document, [to_stage.id])
    removed = site_generation_service.remove_candidate_sites(document, [to_remove.id])

    assert to_stage.status == "staged"
    assert [record.action for record in staged.provenance] == ["stage_muon_relaxation_sites"]
    assert removed.sites == [to_remove]
    assert [record.action for record in removed.provenance] == ["delete_candidates"]
    assert list(document.sites) == [to_stage.id]


_RANDOM_REQUEST = {"selection_mode": "count", "count": 2, "seed": 1, "min_muon_atom": 0.8}


def test_a_worker_request_computes_sites_without_touching_the_document(tmp_path: Path) -> None:
    document = _document(tmp_path)
    existing = MuonSite(label="Random Site 4", status="generated")
    document.sites[existing.id] = existing
    request = site_generation_service.prepare_candidate_sites(
        document, method="random", parameters=_RANDOM_REQUEST
    )

    outcome = site_generation_service.compute_candidate_sites(request)

    assert list(document.sites) == [existing.id]
    assert [site.label for site in outcome.sites] == ["Random Site 5", "Random Site 6"]
    assert [record.action for record in outcome.provenance] == ["generate_candidates"]

    site_generation_service.apply_candidate_sites(document, request, outcome)

    assert list(document.sites) == [existing.id, *(site.id for site in outcome.sites)]
    assert document.provenance == []


def test_a_worker_request_reads_a_copy_of_the_host(tmp_path: Path) -> None:
    document = _document(tmp_path)
    host = document.structures[document.project.host_structure_id]
    request = site_generation_service.prepare_candidate_sites(
        document, method="near_element", parameters={"element": "O"}
    )

    host.fractional_positions[0][0] = 0.99
    host.species.append("Si")

    assert request.host.positions[0][0] != 0.99
    assert len(request.host.species) == len(host.species) - 1
    assert request.host_structure_id == host.id


def test_cancelling_a_worker_request_discards_its_sites(tmp_path: Path) -> None:
    document = _document(tmp_path)
    request = site_generation_service.prepare_candidate_sites(
        document, method="random", parameters=_RANDOM_REQUEST
    )
    reports: list[tuple[int, int]] = []

    with pytest.raises(site_generation_service.SiteGenerationCancelled):
        site_generation_service.compute_candidate_sites(
            request,
            progress_callback=lambda generated, expected: reports.append((generated, expected)),
            should_cancel=lambda: bool(reports),
        )

    # The first report went through; the next one found the run cancelled.
    assert reports == [(0, 2)]
    assert document.sites == {}


def test_cancelling_after_the_last_report_still_discards_the_sites(tmp_path: Path) -> None:
    document = _document(tmp_path)
    request = site_generation_service.prepare_candidate_sites(
        document, method="random", parameters=_RANDOM_REQUEST
    )
    reports: list[tuple[int, int]] = []
    finished = [False]

    def report(generated: int, expected: int) -> None:
        reports.append((generated, expected))
        finished[0] = generated >= expected

    with pytest.raises(site_generation_service.SiteGenerationCancelled):
        site_generation_service.compute_candidate_sites(
            request,
            progress_callback=report,
            # Cancelled only once every site has been reported.
            should_cancel=lambda: finished[0] and len(reports) > 2,
        )
    assert reports[-1] == (2, 2)


def test_applying_a_worker_request_after_a_host_change_adds_nothing(tmp_path: Path) -> None:
    document = _document(tmp_path)
    request = site_generation_service.prepare_candidate_sites(
        document, method="random", parameters=_RANDOM_REQUEST
    )
    outcome = site_generation_service.compute_candidate_sites(request)
    other = cof2_host()
    document.structures[other.id] = other
    document.project.host_structure_id = other.id

    with pytest.raises(SiteGenerationError, match="active host structure changed"):
        site_generation_service.apply_candidate_sites(document, request, outcome)
    assert document.sites == {}


@pytest.mark.parametrize("method", ["manual", "uep_clusters", "bogus"])
def test_only_random_and_attached_generation_prepare_a_worker_request(
    tmp_path: Path, method: str
) -> None:
    with pytest.raises(SiteGenerationError, match="cannot run on a worker thread"):
        site_generation_service.prepare_candidate_sites(
            _document(tmp_path), method=method, parameters={}
        )


def test_applying_a_worker_request_after_an_edit_to_the_host_adds_nothing(
    tmp_path: Path,
) -> None:
    document = _document(tmp_path)
    request = site_generation_service.prepare_candidate_sites(
        document, method="random", parameters=_RANDOM_REQUEST
    )
    outcome = site_generation_service.compute_candidate_sites(request)
    # The same host object takes a new geometry, as a refreshed relaxed host does.
    host = document.structures[document.project.host_structure_id]
    host.fractional_positions[0] = [0.4, 0.4, 0.4]

    with pytest.raises(SiteGenerationError, match="active host structure changed"):
        site_generation_service.apply_candidate_sites(document, request, outcome)
    assert document.sites == {}


def test_applying_a_worker_request_renumbers_labels_taken_meanwhile(tmp_path: Path) -> None:
    document = _document(tmp_path)
    request = site_generation_service.prepare_candidate_sites(
        document, method="random", parameters=_RANDOM_REQUEST
    )
    outcome = site_generation_service.compute_candidate_sites(request)
    assert [site.label for site in outcome.sites] == ["Random Site 1", "Random Site 2"]
    taken = MuonSite(label="Random Site 1", status="generated")
    document.sites[taken.id] = taken

    site_generation_service.apply_candidate_sites(document, request, outcome)

    assert sorted(site.label for site in document.sites.values()) == [
        "Random Site 1",
        "Random Site 2",
        "Random Site 3",
    ]
