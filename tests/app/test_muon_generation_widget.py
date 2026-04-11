"""Behavior tests for muon generation form widget."""

from __future__ import annotations

from mufinder.app.muon_generation_widget import MuonGenerationWidget


def test_widget_state_roundtrip_and_table_render(qapp) -> None:
    widget = MuonGenerationWidget()

    state = {
        "method": "near_element",
        "random": {
            "count": 30,
            "min_muon_muon": 0.9,
            "min_muon_atom": 1.1,
            "max_attempts": 25000,
            "seed": "5",
        },
        "near_element": {
            "element": "O",
            "count": 12,
            "shell_min": 0.8,
            "shell_max": 1.5,
            "min_muon_muon": 0.7,
            "min_muon_atom": 0.8,
            "max_attempts": 30000,
            "seed": "13",
        },
        "manual": {
            "coord_system": "cartesian",
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
        },
    }
    widget.set_form_state(state)
    collected = widget.collect_state()

    assert collected["method"] == "near_element"
    assert collected["near_element"]["shell_max"] == 1.5
    assert collected["manual"]["coord_system"] == "cartesian"

    emitted: list[tuple[str, dict]] = []
    widget.generate_requested.connect(lambda method, payload: emitted.append((method, payload)))
    widget._emit_generate()

    assert emitted[0][0] == "near_element"
    assert emitted[0][1]["element"] == "O"

    widget.set_sites(
        [
            {
                "label": "Site A",
                "method": "random",
                "status": "candidate",
                "fractional": [0.1, 0.2, 0.3],
            }
        ]
    )
    assert widget._site_table.rowCount() == 1
    assert widget._site_table.item(0, 0).text() == "Site A"


def test_widget_collect_method_parameters_for_all_modes(qapp) -> None:
    widget = MuonGenerationWidget()

    widget._method_combo.setCurrentText("random")
    random_params = widget._collect_method_parameters("random")
    assert "count" in random_params
    assert "max_attempts" in random_params

    widget._method_combo.setCurrentText("near_element")
    widget._near_element.setText("Si")
    near_params = widget._collect_method_parameters("near_element")
    assert near_params["element"] == "Si"
    assert near_params["shell_min"] < near_params["shell_max"]

    widget._method_combo.setCurrentText("manual")
    widget._manual_coord_system.setCurrentText("fractional")
    widget._manual_x.setValue(0.2)
    manual_params = widget._collect_method_parameters("manual")
    assert manual_params["coord_system"] == "fractional"
    assert manual_params["x"] == 0.2
