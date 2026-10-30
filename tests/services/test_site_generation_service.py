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
from mufinder.services.project_service import ProjectService
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


_COUNT_REQUEST = {
    "selection_mode": "count",
    "count": 4,
    "min_muon_atom": 1.0,
    "max_attempts": 50000,
    "seed": 418273,
    "include_symmetry_equivalents": True,
}


def test_a_random_run_records_its_inputs_seed_space_group_and_outcome(tmp_path: Path) -> None:
    document = _document(tmp_path, cof2_host())

    outcome = site_generation_service.generate_candidate_sites(
        document, method="random", parameters=_COUNT_REQUEST
    )

    run = outcome.run
    assert run is not None and document.generation_runs == {run.id: run}
    assert run.method == "random"
    assert run.seed == 418273
    assert run.parameters["selection_mode"] == "count"
    assert run.parameters["count"] == 4
    assert run.parameters["min_muon_atom"] == 1.0
    assert run.parameters["max_attempts"] == 50000
    assert run.parameters["include_symmetry_equivalents"] is True
    assert run.parameters["min_muon_muon"] == pytest.approx(
        outcome.sites[0].properties["min_muon_muon"]
    )
    assert run.parameters["estimated_min_muon_muon"] == pytest.approx(
        outcome.sites[0].properties["estimated_min_muon_muon"]
    )
    assert run.uses_symmetry is True
    assert (run.space_group_symbol, run.space_group_number) == ("P4_2/mnm", 136)
    assert run.symmetry_operation_count == 16
    assert run.symmetry_tolerance_ang == 1e-3
    assert run.site_ids == [site.id for site in outcome.sites]
    assert run.requested_count == 4
    assert run.distinct_site_count == 4
    assert run.min_muon_muon_ang >= run.parameters["min_muon_muon"] - 1e-9
    assert run.attempts_used >= 4
    assert run.retries == 0
    assert run.source_structure_id == document.project.host_structure_id
    assert {site.source_generation_batch_id for site in outcome.sites} == {run.id}


def test_a_random_run_given_no_seed_draws_one_and_records_it(tmp_path: Path) -> None:
    document = _document(tmp_path, cof2_host())
    parameters = {key: value for key, value in _COUNT_REQUEST.items() if key != "seed"}

    outcome = site_generation_service.generate_candidate_sites(
        document, method="random", parameters=parameters
    )

    assert outcome.run is not None
    assert isinstance(outcome.run.seed, int) and 1 <= outcome.run.seed <= 999_999
    assert outcome.provenance[0].parameters["seed"] == outcome.run.seed
    assert "seed" not in parameters


def test_the_same_seed_reproduces_the_same_starts(tmp_path: Path) -> None:
    unseeded = {key: value for key, value in _COUNT_REQUEST.items() if key != "seed"}
    first = site_generation_service.generate_candidate_sites(
        _document(tmp_path / "a", cof2_host()), method="random", parameters=unseeded
    )
    assert first.run is not None
    again = site_generation_service.generate_candidate_sites(
        _document(tmp_path / "b", cof2_host()),
        method="random",
        parameters={**unseeded, "seed": first.run.seed},
    )

    assert [site.initial_fractional_position for site in again.sites] == [
        site.initial_fractional_position for site in first.sites
    ]


def test_a_random_separation_run_records_the_separation_and_no_retries(tmp_path: Path) -> None:
    document = _document(tmp_path, cof2_host())

    outcome = site_generation_service.generate_candidate_sites(
        document,
        method="random",
        parameters={
            "selection_mode": "separation",
            "min_muon_muon": 1.5,
            "min_muon_atom": 1.0,
            "max_attempts": 300,
            "seed": 7,
            "include_symmetry_equivalents": False,
        },
    )

    run = outcome.run
    assert run is not None
    assert run.parameters["min_muon_muon"] == 1.5
    assert "count" not in run.parameters
    assert run.requested_count is None
    assert run.retries is None
    assert run.attempts_used == 300
    assert run.uses_symmetry is False
    assert run.space_group_symbol == "P4_2/mnm"


def test_a_count_run_records_how_many_times_the_separation_was_reduced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _document(tmp_path)
    real_generate = site_generation_service.generate_random_sites
    calls: list[int] = []

    def fail_twice(**kwargs: object):
        calls.append(1)
        if len(calls) < 3:
            raise SiteGenerationError("too tight")
        return real_generate(**kwargs)

    monkeypatch.setattr(site_generation_service, "generate_random_sites", fail_twice)

    outcome = site_generation_service.generate_candidate_sites(
        document, method="random", parameters={"selection_mode": "count", "count": 2, "seed": 1}
    )

    assert outcome.run is not None and outcome.run.retries == 2


def test_an_attached_run_records_its_element_distances_and_neighbour_cutoff(
    tmp_path: Path,
) -> None:
    document = _document(tmp_path, cof2_host())

    outcome = site_generation_service.generate_candidate_sites(
        document,
        method="near_element",
        parameters={
            "element": "F",
            "muons_per_atom": 2,
            "bond_distance": 1.0,
            "min_muon_atom": 0.8,
        },
    )

    run = outcome.run
    assert run is not None
    assert run.parameters == {
        "element": "F",
        "muons_per_atom": 2,
        "bond_distance": 1.0,
        "min_muon_atom": 0.8,
        "neighbour_cutoff_ang": 2.6,
    }
    assert run.seed is None
    assert run.uses_symmetry is True
    assert run.space_group_symbol == "P4_2/mnm"
    assert run.retries is None
    assert len(run.site_ids) == 2


def test_a_manual_run_records_the_coordinate_system(tmp_path: Path) -> None:
    document = _document(tmp_path)

    outcome = site_generation_service.generate_candidate_sites(
        document,
        method="manual",
        parameters={"coord_system": "fractional", "x": 0.1, "y": 0.2, "z": 0.3},
    )

    run = outcome.run
    assert run is not None
    assert run.parameters == {"coord_system": "fractional", "coordinates": [0.1, 0.2, 0.3]}
    assert run.seed is None
    assert run.uses_symmetry is False
    assert run.attempts_used is None
    assert outcome.sites[0].source_generation_batch_id == run.id


def test_a_uep_run_records_the_source_job_and_its_grouping(tmp_path: Path) -> None:
    document = _document(tmp_path)
    selection = {
        "source_key": "job-1:cluster:0",
        "fractional_position": [0.1, 0.2, 0.3],
        "cartesian_position": [1, 2, 3],
        "properties": {
            "source_job_id": "job-1",
            "source_job_label": "UEP search",
            "clustering_method": "hier",
            "clustering_hier_t": 0.3,
            "clustering_kmeans_k": None,
        },
    }

    outcome = site_generation_service.generate_candidate_sites(
        document, method="uep_clusters", parameters={"selections": [selection]}
    )

    assert outcome.run is not None
    assert outcome.run.parameters == {
        "source_jobs": [{"id": "job-1", "label": "UEP search"}],
        "grouping": {"clustering_method": "hier", "clustering_hier_t": 0.3},
    }


def test_the_generation_record_carries_the_run(tmp_path: Path) -> None:
    document = _document(tmp_path, cof2_host())

    outcome = site_generation_service.generate_candidate_sites(
        document, method="random", parameters=_COUNT_REQUEST
    )

    assert outcome.run is not None
    parameters = outcome.provenance[0].parameters
    assert parameters["generation_run_id"] == str(outcome.run.id)
    assert parameters["generation_run"]["seed"] == 418273
    assert parameters["generation_run"]["space_group_symbol"] == "P4_2/mnm"
    assert parameters["generation_run"]["parameters"]["count"] == 4


def test_a_worker_request_records_its_run_when_applied(tmp_path: Path) -> None:
    document = _document(tmp_path)
    request = site_generation_service.prepare_candidate_sites(
        document, method="random", parameters={"selection_mode": "count", "count": 2}
    )
    outcome = site_generation_service.compute_candidate_sites(request)

    assert document.generation_runs == {}
    site_generation_service.apply_candidate_sites(document, request, outcome)

    assert outcome.run is not None
    assert document.generation_runs == {outcome.run.id: outcome.run}
    assert outcome.run.seed == request.parameters["seed"]


def test_cluster_positions_keeps_the_original_starts_and_the_anchor(tmp_path: Path) -> None:
    document = _document(tmp_path, cof2_host())
    outcome = site_generation_service.generate_candidate_sites(
        document, method="random", parameters=_COUNT_REQUEST
    )
    assert outcome.run is not None
    originals = {str(site.id): list(site.initial_fractional_position) for site in outcome.sites[:2]}

    clustered = site_generation_service.cluster_candidate_positions(
        document, [site.id for site in outcome.sites[:2]], anchor_fractional=[1.25, 0.25, 0.25]
    )

    run = document.generation_runs[outcome.run.id]
    assert len(run.moves) == 1
    assert run.moves[0].anchor_fractional == [0.25, 0.25, 0.25]
    assert run.moves[0].original_positions == originals
    record = clustered.provenance[0]
    assert record.parameters["generation_run_ids"] == [str(run.id)]
    assert record.parameters["original_fractional_positions"] == {
        site.label: originals[str(site.id)] for site in outcome.sites[:2]
    }


def test_generation_runs_survive_a_save_and_reload(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "runs.mufinder", "Runs")
    service.add_structure(document, cof2_host())
    created = service.generate_candidate_sites(document, method="random", parameters=_COUNT_REQUEST)
    service.cluster_candidate_positions(
        document, [created[0].id], anchor_fractional=[0.25, 0.25, 0.25]
    )
    service.save_project(document)

    reloaded = service.load_project(document.project_root)

    assert reloaded.generation_runs == document.generation_runs
    run = next(iter(reloaded.generation_runs.values()))
    assert reloaded.sites[created[0].id].source_generation_batch_id == run.id
    assert list(run.moves[0].original_positions) == [str(created[0].id)]


def test_a_count_run_counts_the_whole_limit_of_every_try_that_fell_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _document(tmp_path)
    real_generate = site_generation_service.generate_random_sites
    calls: list[int] = []

    def fail_once(**kwargs: object):
        calls.append(1)
        if len(calls) < 2:
            raise SiteGenerationError("too tight")
        return real_generate(**kwargs)

    monkeypatch.setattr(site_generation_service, "generate_random_sites", fail_once)

    outcome = site_generation_service.generate_candidate_sites(
        document,
        method="random",
        parameters={"selection_mode": "count", "count": 2, "seed": 1, "max_attempts": 500},
    )

    assert outcome.run is not None and outcome.run.retries == 1
    assert 500 < outcome.run.attempts_used <= 1000
