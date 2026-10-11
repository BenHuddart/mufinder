"""Muon site generation in the main window: candidates, random generation and the charge-state prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.domain.models import MuonSite, Structure
from mufinder.science.site_generation import SiteGenerationError
from mufinder.services.project_service import ProjectService


def test_generate_requested_handles_errors(main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "gen_ui.mufinder", "GenUI")

    warnings: list[str] = []
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: warnings.append("warning"))

    def boom(*args, **kwargs):
        raise SiteGenerationError("boom")

    monkeypatch.setattr(window._project_service, "generate_candidate_sites", boom)
    window._on_generate_requested("manual", {"coord_system": "fractional", "x": 0.0, "y": 0.0, "z": 0.0})

    assert warnings == ["warning"]
    window.close()


def test_delete_muon_candidates_requested_removes_selected_rows(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "delete_candidate_ui.mufinder", "DeleteCandidateUI")
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
    window._document = service.create_project(tmp_path / "cluster_candidate_ui.mufinder", "ClusterCandidateUI")
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
        fractional_position=[0.1, 0.2, 0.3],
        cartesian_position=[0.5, 1.0, 1.5],
    )
    window._document.sites[site.id] = site

    window._on_cluster_muon_positions_requested(
        {
            "site_ids": [str(site.id)],
            "anchor_fractional": [0.9, 0.8, 0.7],
        }
    )

    assert site.fractional_position == pytest.approx([0.9, 0.8, 0.7])
    assert window._muon_generation_view._site_table.item(0, 3).text() == "0.9000"
    assert window._muon_generation_view._selected_site_ids() == [str(site.id)]
    window.close()


def test_muon_generation_table_uses_natural_candidate_label_order(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "candidate_order_ui.mufinder", "CandidateOrderUI")
    for label in ("Random Site 1", "Random Site 10", "Random Site 2"):
        site = MuonSite(label=label, status="generated", generation_method="random")
        window._document.sites[site.id] = site

    window._refresh_muon_generation_table()

    table = window._muon_generation_view._site_table
    assert [table.item(row, 0).text() for row in range(table.rowCount())] == [
        "Random Site 1",
        "Random Site 2",
        "Random Site 10",
    ]
    window.close()


def test_random_generation_runs_in_worker_and_updates_estimated_distance(main_window, qapp, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "async_random_ui.mufinder", "AsyncRandomUI")
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
    assert window._muon_generation_view._generation_progress.format() == "Generated 1 site(s)"
    assert window._muon_generation_view._generate_btn.isEnabled()
    window.close()


def test_random_generation_progress_text_omits_predicted_total_for_separation_mode(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "random_progress_text.mufinder", "RandomProgressText")

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
