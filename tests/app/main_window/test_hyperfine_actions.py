"""Hyperfine Coupling actions in the main window: handing a coupling to Magnetic Fields."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.app import workflow_table_utils
from mufinder.castep_io.magres import hyperfine_tensor_for_species, parse_magres_hyperfine
from mufinder.domain.models import Result, Structure
from mufinder.services.hyperfine_results import contact_shell_inputs, hyperfine_result_payload
from mufinder.workflow.hyperfine import HYPERFINE_CALC_TYPE
from tests.support.main_window import open_project_with_host

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "hyperfine"


def _hyperfine_result(document, relaxation_job_id: str) -> Result:
    relaxed = Structure(
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Fe", "Fe", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    tensor = hyperfine_tensor_for_species(
        parse_magres_hyperfine(FIXTURES / "fe_tetrahedral_muon.magres"), "H:mu"
    )
    result = Result(
        result_type=HYPERFINE_CALC_TYPE,
        data=hyperfine_result_payload(
            tensor,
            pseudo_spin_density_angstrom3=None,
            shell=contact_shell_inputs(relaxed, (2.2, 2.0, 0.0)),
            source={
                "source_site_id": None,
                "source_relaxation_job_id": relaxation_job_id,
                "relaxed_structure_id": str(relaxed.id),
                "charge_state": "positive_muon",
            },
        ),
    )
    document.results[result.id] = result
    return result


def _selected_site(relaxation_job_id: str) -> dict:
    return {
        "source": "relaxation_clustering",
        "label": "Analysis cluster 1",
        "fractional_position": [0.25, 0.25, 0.25],
        "properties": {"cluster": {"representative_job_id": relaxation_job_id}},
    }


def test_use_in_magnetic_fields_confirms_before_replacing_a_coupling(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _service, document, _host = open_project_with_host(window, tmp_path)
    relaxation_job_id = str(uuid4())
    result = _hyperfine_result(document, relaxation_job_id)
    document.ui_state["magnetic_field_form"] = {
        "selected_muon_site": _selected_site(relaxation_job_id),
        "hyperfine_coupling": "-0.5",
        "hyperfine_coupling_unit": "T_per_muB",
    }
    prompts: list[tuple] = []
    answers = [False, True]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt, **kwargs: prompts.append((*prompt, kwargs)) or answers.pop(0),
    )
    messages: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, _title, text: messages.append(text),
    )

    window._on_use_hyperfine_in_magnetic_fields(str(result.id))

    assert len(prompts) == 1
    assert document.ui_state["magnetic_field_form"]["hyperfine_coupling"] == "-0.5"
    assert messages == []

    window._on_use_hyperfine_in_magnetic_fields(str(result.id))

    form = document.ui_state["magnetic_field_form"]
    assert form["hyperfine_coupling_unit"] == "angstrom^-3"
    assert float(form["hyperfine_coupling"]) < 0
    assert messages and messages[-1].startswith(
        "Filled the hyperfine coupling of 1 Magnetic Fields form(s)."
    )
    assert (
        window._magnetic_field_view.collect_state()["hyperfine_coupling"]
        == (form["hyperfine_coupling"])
    )


def test_use_in_magnetic_fields_without_a_matching_form_says_so(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _service, document, _host = open_project_with_host(window, tmp_path)
    result = _hyperfine_result(document, str(uuid4()))
    document.ui_state["magnetic_field_form"] = {"selected_muon_site": None}
    messages: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, _title, text: messages.append(text),
    )

    window._on_use_hyperfine_in_magnetic_fields(str(result.id))

    assert len(messages) == 1
    assert messages[0].startswith("No Magnetic Fields form shows this muon site.")
    assert "hyperfine_coupling" not in document.ui_state["magnetic_field_form"]


def test_use_in_magnetic_fields_sees_the_site_an_unedited_form_shows(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _service, document, _host = open_project_with_host(window, tmp_path)
    relaxation_job_id = str(uuid4())
    result = _hyperfine_result(document, relaxation_job_id)
    # A form never edited stores nothing, though its page shows a default site.
    document.ui_state.pop("magnetic_field_form", None)
    shown = {
        **window._magnetic_field_view.collect_state(),
        "selected_muon_site": _selected_site(relaxation_job_id),
    }
    monkeypatch.setattr(window._magnetic_field_view, "collect_state", lambda: dict(shown))
    prompts: list[tuple] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt, **kwargs: prompts.append(prompt) or True,
    )
    messages: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, _title, text: messages.append(text),
    )

    window._on_use_hyperfine_in_magnetic_fields(str(result.id))

    form = document.ui_state["magnetic_field_form"]
    assert float(form["hyperfine_coupling"]) < 0
    # The widget's default 0.0 coupling is empty, not a value to confirm replacing.
    assert prompts == []
    assert messages and messages[-1].startswith("Filled the hyperfine coupling of 1")
