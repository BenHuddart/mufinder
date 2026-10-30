"""Muon site generation in the main window: candidates, random generation and the charge-state prompt."""

from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.domain.models import MuonSite, Structure
from mufinder.science.site_generation import SiteGenerationError
from mufinder.services.project_service import ProjectService
from tests.support.symmetry_hosts import cof2_host


def test_generate_requested_handles_errors(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "gen_ui.mufinder", "GenUI")

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append("warning"),
    )

    def boom(*args, **kwargs):
        raise SiteGenerationError("boom")

    monkeypatch.setattr(window._project_service, "generate_candidate_sites", boom)
    window._on_generate_requested(
        "manual", {"coord_system": "fractional", "x": 0.0, "y": 0.0, "z": 0.0}
    )

    assert warnings == ["warning"]
    window.close()


def test_delete_muon_candidates_requested_removes_selected_rows(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(
        tmp_path / "delete_candidate_ui.mufinder", "DeleteCandidateUI"
    )
    candidate = MuonSite(label="Candidate", status="generated", generation_method="manual")
    relaxed = MuonSite(label="Relaxed", status="relaxed", generation_method="manual")
    window._document.sites[candidate.id] = candidate
    window._document.sites[relaxed.id] = relaxed
    service.save_project(window._document)
    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project

    window._on_delete_muon_candidates_requested([str(candidate.id), str(relaxed.id), "not-a-uuid"])

    assert save_calls == ["save"]
    assert candidate.id not in window._document.sites
    assert relaxed.id in window._document.sites
    reloaded = service.load_project(window._document.project_root)
    assert candidate.id not in reloaded.sites
    assert relaxed.id in reloaded.sites
    assert window._muon_generation_view._site_table.rowCount() == 0
    window.close()


def test_cluster_muon_positions_requested_updates_rows(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(
        tmp_path / "cluster_candidate_ui.mufinder", "ClusterCandidateUI"
    )
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(window._document, host)
    site = MuonSite(
        label="Candidate",
        status="generated",
        generation_method="manual",
        initial_fractional_position=[0.1, 0.2, 0.3],
        initial_cartesian_position=[0.5, 1.0, 1.5],
    )
    window._document.sites[site.id] = site

    window._on_cluster_muon_positions_requested(
        {
            "site_ids": [str(site.id)],
            "anchor_fractional": [0.9, 0.8, 0.7],
        }
    )

    assert site.initial_fractional_position == pytest.approx([0.9, 0.8, 0.7])
    # Row 0 heads the earlier starts the site belongs to.
    assert window._muon_generation_view._site_table.item(1, 3).text() == "0.9000"
    assert window._muon_generation_view._selected_site_ids() == [str(site.id)]
    window.close()


def test_muon_generation_table_uses_natural_candidate_label_order(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(
        tmp_path / "candidate_order_ui.mufinder", "CandidateOrderUI"
    )
    for label in ("Random Site 1", "Random Site 10", "Random Site 2"):
        site = MuonSite(label=label, status="generated", generation_method="random")
        window._document.sites[site.id] = site

    window._refresh_muon_generation_table()

    table = window._muon_generation_view._site_table
    assert [table.item(row, 0).text() for row in range(table.rowCount())] == [
        "Earlier starts: Random, 3 starts",
        "Random Site 1",
        "Random Site 2",
        "Random Site 10",
    ]
    window.close()


def test_random_generation_runs_in_worker_and_updates_estimated_distance(
    main_window, qapp, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(
        tmp_path / "async_random_ui.mufinder", "AsyncRandomUI"
    )
    host = Structure(
        label="Host",
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(window._document, host)
    window._muon_generation_view._random_by_count.setChecked(True)
    window._muon_generation_view._random_count.setValue(1)

    window._on_generate_requested(
        "random",
        {
            "selection_mode": "count",
            "count": 1,
            "min_muon_atom": 0.8,
            "max_attempts": 5000,
            "seed": "3",
            "include_symmetry_equivalents": False,
        },
    )

    assert window._muon_generation_thread is not None
    assert window._muon_generation_view._generation_progress.maximum() == 0

    for _ in range(2000):
        qapp.processEvents()
        if window._muon_generation_thread is None:
            break

    assert window._muon_generation_thread is None
    assert len(window._document.sites) == 1
    site = next(iter(window._document.sites.values()))
    assert site.properties["estimated_min_muon_muon"] > 0.0
    assert window._muon_generation_view._random_min_muon_muon.value() == pytest.approx(
        site.properties["estimated_min_muon_muon"]
    )
    assert window._muon_generation_view._generation_progress.value() == 1
    assert window._muon_generation_view._generation_progress.maximum() == 1
    assert window._muon_generation_view._generation_progress.format() == "Generated 1 site"
    assert window._muon_generation_view._generate_btn.isEnabled()
    window.close()


def test_random_generation_progress_text_omits_predicted_total_for_separation_mode(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(
        tmp_path / "random_progress_text.mufinder", "RandomProgressText"
    )

    window._on_generation_progress_changed("random", 6, 13, "separation")

    assert window._muon_generation_view._generation_progress.value() == 6
    assert window._muon_generation_view._generation_progress.maximum() == 13
    assert window._muon_generation_view._generation_progress.format() == "Generated %v sites"

    window._on_generation_progress_changed("random", 6, 13, "count")

    assert window._muon_generation_view._generation_progress.format() == "Generated %v/%m sites"
    window.close()


def test_muon_charge_state_dialog_defaults_and_returns_both_states(qapp) -> None:
    dialog = main_window_module.MuonChargeStateDialog(site_count=2)

    assert dialog.objectName() == "muonChargeStateDialog"
    assert dialog.windowTitle() == "Create Muon Relaxation Jobs"
    assert dialog.selected_charge_states() == ("positive_muon",)

    dialog._both.setChecked(True)

    assert dialog.selected_charge_states() == ("positive_muon", "muonium")


def test_main_window_generate_requested_without_project_shows_info(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._on_generate_requested("random", {"count": 1})

    assert infos == ["No project"]
    window.close()


_HELD_RANDOM_PARAMETERS = {
    "selection_mode": "count",
    "count": 2,
    "min_muon_atom": 0.8,
    "max_attempts": 5000,
    "seed": "3",
    "include_symmetry_equivalents": False,
}


def _window_with_cubic_host(window, tmp_path: Path, name: str) -> Structure:
    service = ProjectService()
    window._document = service.create_project(tmp_path / f"{name}.mufinder", name)
    host = Structure(
        label="Host",
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(window._document, host)
    return host


@pytest.fixture
def held_worker(monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, threading.Event]:
    """Hold each generation worker once it has computed its sites, until released.

    Returns ``(computed, release)``: ``computed`` is set when the worker holds,
    and setting ``release`` lets it hand its sites back.
    """

    computed = threading.Event()
    release = threading.Event()
    original = ProjectService.compute_candidate_sites

    def held(request, **kwargs):
        outcome = original(request, **kwargs)
        computed.set()
        release.wait(10)
        return outcome

    monkeypatch.setattr(ProjectService, "compute_candidate_sites", staticmethod(held))
    yield computed, release
    release.set()


def _finish_worker(qapp, window) -> None:
    for _ in range(2000):
        qapp.processEvents()
        if window._muon_generation_thread is None:
            return
        time.sleep(0.005)
    raise AssertionError("the generation worker did not finish")


def test_generation_adds_no_site_to_the_document_until_the_worker_has_finished(
    main_window, qapp, tmp_path: Path, held_worker
) -> None:
    window = main_window
    _window_with_cubic_host(window, tmp_path, "HeldGeneration")
    computed, release = held_worker

    window._on_generate_requested("random", dict(_HELD_RANDOM_PARAMETERS))
    assert computed.wait(10)
    qapp.processEvents()

    # The worker holds its finished sites; the document has none of them.
    assert window._document.sites == {}
    assert not any(record.action == "generate_candidates" for record in window._document.provenance)

    release.set()
    _finish_worker(qapp, window)

    assert len(window._document.sites) == 2
    assert [record.action for record in window._document.provenance][-1] == ("generate_candidates")
    window.close()


def test_cancelling_generation_discards_its_sites(
    main_window, qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held_worker
) -> None:
    window = main_window
    _window_with_cubic_host(window, tmp_path, "CancelGeneration")
    computed, release = held_worker
    view = window._muon_generation_view
    questions: list[str] = []

    def confirm(_parent, title, *_args, **_kwargs) -> bool:
        questions.append(title)
        return True

    monkeypatch.setattr(main_window_module.workflow_table_utils, "confirm_destructive", confirm)

    window._on_generate_requested("random", dict(_HELD_RANDOM_PARAMETERS))
    assert computed.wait(10)
    assert view._cancel_generation_btn.isVisibleTo(view)
    assert not view._generate_btn.isEnabled()

    view._cancel_generation_btn.click()
    release.set()
    _finish_worker(qapp, window)

    assert questions == ["Cancel generation"]
    assert window._document.sites == {}
    assert view._generation_progress.format() == "Generation cancelled"
    assert view._generate_btn.isEnabled()
    assert not view._cancel_generation_btn.isVisibleTo(view)
    window.close()


def test_declining_to_cancel_generation_keeps_it_running(
    main_window, qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held_worker
) -> None:
    window = main_window
    _window_with_cubic_host(window, tmp_path, "KeepGeneration")
    computed, release = held_worker
    monkeypatch.setattr(
        main_window_module.workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False
    )

    window._on_generate_requested("random", dict(_HELD_RANDOM_PARAMETERS))
    assert computed.wait(10)
    window._muon_generation_view._cancel_generation_btn.click()
    release.set()
    _finish_worker(qapp, window)

    assert len(window._document.sites) == 2
    window.close()


def test_a_host_change_is_refused_while_generation_runs(
    main_window, qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held_worker
) -> None:
    window = main_window
    host = _window_with_cubic_host(window, tmp_path, "HostDuringGeneration")
    other = Structure(
        label="Other",
        lattice_vectors=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    window._document.structures[other.id] = other
    computed, release = held_worker
    refusals: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, *_args, **_kwargs: refusals.append(title),
    )

    window._on_generate_requested("random", dict(_HELD_RANDOM_PARAMETERS))
    assert computed.wait(10)
    window._on_active_host_change_requested(str(other.id))

    assert refusals == ["Generation in progress"]
    assert window._document.project.host_structure_id == host.id

    release.set()
    _finish_worker(qapp, window)
    assert len(window._document.sites) == 2

    # Once the run has finished, the host can change.
    window._on_active_host_change_requested(str(other.id))
    assert window._document.project.host_structure_id == other.id
    window.close()


def test_closing_the_window_cancels_a_running_generation(
    main_window, qapp, tmp_path: Path, held_worker
) -> None:
    window = main_window
    _window_with_cubic_host(window, tmp_path, "CloseDuringGeneration")
    computed, release = held_worker

    window._on_generate_requested("random", dict(_HELD_RANDOM_PARAMETERS))
    assert computed.wait(10)
    thread = window._muon_generation_thread
    document = window._document
    cancels: list[bool] = []
    original_cancel = thread.cancel
    thread.cancel = lambda: (cancels.append(True), original_cancel())
    release.set()
    window.close()

    assert cancels == [True]
    assert window._muon_generation_thread is None
    qapp.processEvents()
    assert document.sites == {}


def test_a_run_started_in_a_closed_project_is_discarded(
    main_window, qapp, tmp_path: Path, held_worker
) -> None:
    window = main_window
    _window_with_cubic_host(window, tmp_path, "FirstProject")
    first = window._document
    computed, release = held_worker

    window._on_generate_requested("random", dict(_HELD_RANDOM_PARAMETERS))
    assert computed.wait(10)
    second = ProjectService().create_project(tmp_path / "second.mufinder", "SecondProject")
    window._document = second
    release.set()
    _finish_worker(qapp, window)

    assert first.sites == {}
    assert second.sites == {}
    assert window._muon_generation_view._generation_progress.format() == "Ready"
    window.close()


def _counting_saves(window) -> list[str]:
    save_calls: list[str] = []
    original_save = window._project_service.save_project

    def _save_project(document) -> None:
        save_calls.append("save")
        original_save(document)

    window._project_service.save_project = _save_project
    return save_calls


def test_generation_saves_the_project_and_lists_the_run_under_its_header(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "saved_run.mufinder", "SavedRun")
    service.add_structure(window._document, cof2_host())
    save_calls = _counting_saves(window)

    window._on_generate_requested(
        "manual", {"coord_system": "fractional", "x": 0.25, "y": 0.25, "z": 0.5}
    )

    assert save_calls == ["save"]
    reloaded = service.load_project(window._document.project_root)
    assert len(reloaded.generation_runs) == 1
    assert window._muon_generation_view.run_header_titles() == [
        "Custom coordinates, 1 start, fractional"
    ]
    window.close()


def test_cluster_positions_saves_the_project_with_the_original_starts(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "cluster_run.mufinder", "ClusterRun")
    service.add_structure(window._document, cof2_host())
    created = service.generate_candidate_sites(
        window._document,
        method="random",
        parameters={"selection_mode": "count", "count": 2, "min_muon_atom": 1.0, "seed": 5},
    )
    original = list(created[0].initial_fractional_position)
    save_calls = _counting_saves(window)

    window._on_cluster_muon_positions_requested(
        {"site_ids": [str(created[0].id)], "anchor_fractional": [0.5, 0.5, 0.5]}
    )

    assert save_calls == ["save"]
    reloaded = service.load_project(window._document.project_root)
    run = next(iter(reloaded.generation_runs.values()))
    assert run.moves[0].anchor_fractional == [0.5, 0.5, 0.5]
    assert run.moves[0].original_positions == {str(created[0].id): original}
    assert window._document.provenance[-1].action == "cluster_candidate_positions"
    window.close()
