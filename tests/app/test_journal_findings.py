"""The window records findings where results are stored, and the Journal shows them."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QInputDialog

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Job
from mufinder.services.journal import build_journal
from tests.support.main_window import window_with_host


@pytest.fixture
def window_document(qapp, tmp_path: Path):
    window, _service, document, _host = window_with_host(tmp_path)
    try:
        yield window, document
    finally:
        window.close()


def _findings(document, action: str) -> list:
    return [record for record in document.provenance if record.action == action]


def _field_form(tesla: float) -> dict:
    return {
        "selected_muon_site": {"label": "Mu 1"},
        "magnetic_field_results": {
            "kind": "commensurate",
            "results": [
                {
                    "site_label": "Mu 1",
                    "muon_position": [0.0, 0.5, 0.0],
                    "dipolar_T": [0.0, 0.0, tesla],
                    "total_T": [0.0, 0.0, tesla],
                    "magnitude_T": tesla,
                }
            ],
        },
    }


def test_a_grouped_uep_search_reaches_the_journal_through_its_state(window_document) -> None:
    window, document = window_document
    job = Job(label="uep_source", workflow_stage=WorkflowStage.UEP, calc_type="uep")
    document.jobs[job.id] = job
    state = {
        "batch_results": [{"start_position_ang": [0.0, 0.0, 0.0], "total_energy_ev": None}],
        "batch_summary_data": {
            "clustering_method": "legacy_cartesian_hier",
            "threshold_ang": 0.2,
            "clusters": [
                {
                    "cluster_index": 1,
                    "size": 1,
                    "member_indexes": [0],
                    "representative_fractional_position": [0.0, 0.5, 0.0],
                    "minimum_energy_ev": -1.0,
                }
            ],
        },
    }

    window._on_uep_analysis_state_changed(job.id, state)
    window._on_uep_analysis_state_changed(job.id, {**state, "current_tab": 2})

    assert len(_findings(document, "uep_minima_found")) == 1
    entries = build_journal("Study", document.provenance).days[-1].entries
    assert entries[-1].body == "UEP: 1 distinct minimum; lowest at (0.000, 0.500, 0.000)"


def test_a_field_on_the_page_and_a_workspace_is_recorded_under_its_title(
    window_document, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, document = window_document
    monkeypatch.setattr(QInputDialog, "getText", lambda *_args, **_kwargs: ("Saved", True))
    window._on_magnetic_field_form_state_changed(_field_form(0.42))
    window._on_magnetic_field_form_state_changed(_field_form(0.42))
    window._on_magnetic_field_pin_requested(
        {"title": "Saved", "form": _field_form(0.42), "magnetic_structure": None}
    )
    workspace_id = window._magnetic_field_workspaces_state()[0]["id"]

    window._on_magnetic_field_workspace_form_state_changed(workspace_id, _field_form(0.5))

    records = _findings(document, "field_calculated")
    assert [record.parameters["owner_title"] for record in records] == ["Magnetic Fields", "Saved"]


def test_saving_results_as_a_workspace_is_not_a_new_calculation(
    window_document, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, document = window_document
    monkeypatch.setattr(QInputDialog, "getText", lambda *_args, **_kwargs: ("Saved", True))
    window._on_magnetic_field_form_state_changed(_field_form(0.42))

    window._on_magnetic_field_pin_requested(
        {"title": "Saved", "form": _field_form(0.42), "magnetic_structure": None}
    )

    assert len(_findings(document, "field_calculated")) == 1
