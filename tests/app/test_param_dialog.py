"""Tests for standalone CASTEP parameter dialog behavior."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QLabel, QPushButton

import mufinder.app.param_dialog as param_dialog_module
from mufinder.app.param_dialog import ParameterDialog
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import ValueWithUnit


def _dialog_specs() -> list[KeywordSpec]:
    return [
        KeywordSpec(
            name="task",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("SINGLEPOINT", "GEOMETRYOPTIMIZATION"),
            widget_hint="dropdown",
            category="workflow",
            basic_workflows=("convergence_cutoff", "convergence_kpoint"),
            default_value="SINGLEPOINT",
        ),
        KeywordSpec(
            name="xc_functional",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("PBE", "LDA"),
            widget_hint="dropdown",
            category="electronic",
            basic_workflows=("convergence_cutoff", "convergence_kpoint"),
            default_value="PBE",
        ),
        KeywordSpec(
            name="basis_precision",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("FINE", "PRECISE"),
            widget_hint="dropdown",
            category="electronic",
            advanced_workflows=("convergence_cutoff", "convergence_kpoint"),
            mutually_exclusive_with=("cut_off_energy",),
            default_value="FINE",
        ),
        KeywordSpec(
            name="cut_off_energy",
            file="param",
            kind="scalar",
            value_type="positive_float_with_optional_unit",
            unit="eV",
            widget_hint="unit_float",
            basic_workflows=("convergence_cutoff",),
            advanced_workflows=("convergence_kpoint",),
            locked_in_workflows=("convergence_cutoff",),
            mutually_exclusive_with=("basis_precision",),
            default_value=340.0,
        ),
        KeywordSpec(
            name="geom_method",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("BFGS", "LBFGS"),
            widget_hint="dropdown",
            category="geometry",
            basic_workflows=("convergence_cutoff", "convergence_kpoint"),
            allowed_tasks=("GEOMETRYOPTIMIZATION",),
            description="Geometry optimizer used for structure relaxation.",
            default_value="BFGS",
        ),
    ]


def test_parameter_dialog_infers_task_and_hides_locked_value(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_cutoff",
        locked_values={"cut_off_energy": "Variable (set by sweep)"},
    )

    values = dialog.collect_values()

    if dialog._task_editor is not None:
        assert dialog._task_editor.value() == "SINGLEPOINT"
    else:
        assert dialog._task_display.text() == "SINGLEPOINT"
    assert values["task"] == "SINGLEPOINT"
    assert "cut_off_energy" not in values


def test_parameter_dialog_blocks_conflict_with_locked_keyword(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_cutoff",
        locked_values={"cut_off_energy": "Variable (set by sweep)"},
    )

    dialog.set_advanced_names({"basis_precision"})

    assert "basis_precision" not in dialog._rows


def test_parameter_dialog_exclusive_choice_collects_one_value(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
    )
    dialog.set_advanced_names({"basis_precision", "cut_off_energy"})
    dialog.set_values(
        {
            "basis_precision": "PRECISE",
            "cut_off_energy": ValueWithUnit(420.0, "eV"),
        }
    )

    values = dialog.collect_values()
    assert "basis_precision" not in values
    assert values["cut_off_energy"].value == 420.0

    dialog._rows["basis_precision"].radio.setChecked(True)  # type: ignore[union-attr]
    switched = dialog.collect_values()
    assert switched["basis_precision"] == "PRECISE"
    assert "cut_off_energy" not in switched


def test_parameter_dialog_set_advanced_names_keeps_active_advanced_values_visible(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
    )
    dialog.set_values(
        {
            "basis_precision": "FINE",
            "cut_off_energy": ValueWithUnit(450.0, "eV"),
        }
    )

    dialog.set_advanced_names(())

    assert "cut_off_energy" in dialog._rows
    assert dialog._rows["cut_off_energy"].radio.isChecked()  # type: ignore[union-attr]
    values = dialog.collect_values()
    assert "basis_precision" not in values
    assert values["cut_off_energy"].value == 450.0


def test_parameter_dialog_always_shows_spin_polarized_unchecked_by_default(qapp) -> None:
    specs = [
        *_dialog_specs(),
        KeywordSpec(
            name="spin_polarized",
            file="param",
            kind="scalar",
            value_type="bool",
            widget_hint="checkbox",
            advanced_workflows=("convergence_kpoint",),
            default_value=False,
        ),
    ]
    dialog = ParameterDialog(
        specs=specs,
        workflow_stage="convergence_kpoint",
    )

    assert "spin_polarized" in dialog._rows
    assert dialog._rows["spin_polarized"].editor is not None
    assert dialog._rows["spin_polarized"].editor.value() is False
    assert dialog.collect_values()["spin_polarized"] is False


def test_parameter_dialog_roundtrip_values_and_advanced_rows(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
    )
    dialog.set_advanced_names({"basis_precision"})
    dialog.set_values({"xc_functional": "LDA", "basis_precision": "PRECISE"})

    collected = dialog.collect_values()

    assert collected["xc_functional"] == "LDA"
    assert collected["basis_precision"] == "PRECISE"
    assert set(dialog.advanced_names()) == {"basis_precision"}


def test_parameter_dialog_rebuild_does_not_duplicate_headers(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_cutoff",
        locked_values={"cut_off_energy": "Variable (set by sweep)"},
    )

    dialog.set_values({"xc_functional": "LDA"})
    dialog.set_advanced_names({"basis_precision"})
    dialog.set_values({"xc_functional": "PBE", "max_scf_cycles": 80})

    assert len(dialog._table_widget.findChildren(QLabel, "parameterHeader")) == 1
    assert len(dialog._table_widget.findChildren(QLabel, "valueHeader")) == 1
    assert len(dialog._table_widget.findChildren(QLabel, "resetHeader")) == 1
    assert len(dialog._table_widget.findChildren(QLabel, "infoHeader")) == 1


def test_parameter_dialog_includes_reset_button_per_visible_parameter(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
    )

    reset_buttons = [
        button for button in dialog._table_widget.findChildren(QPushButton) if button.text() == "Reset"
    ]

    assert len(reset_buttons) == len(dialog._rows)


def test_parameter_dialog_reset_restores_parameter_default(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
    )

    dialog.set_values({"xc_functional": "LDA"})
    assert dialog.collect_values()["xc_functional"] == "LDA"

    dialog._reset_parameter("xc_functional")

    assert dialog.collect_values()["xc_functional"] == "PBE"


def test_parameter_dialog_search_filters_by_description_and_aliases(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
    )
    dialog.set_advanced_names({"basis_precision"})
    assert dialog._task_editor is not None
    dialog._task_editor.set_value("GEOMETRYOPTIMIZATION")
    dialog._on_task_changed()
    dialog._search_input.setText("geometry optimizer")

    assert set(dialog._rows) == {"geom_method"}


def test_parameter_dialog_updates_task_visibility_and_prunes_incompatible_values(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
    )
    dialog.set_values({"geom_method": "LBFGS"})
    assert "geom_method" not in dialog._rows

    assert dialog._task_editor is not None
    dialog._task_editor.set_value("GEOMETRYOPTIMIZATION")
    dialog._on_task_changed()
    assert "geom_method" in dialog._rows

    dialog.set_values({"geom_method": "LBFGS"})
    dialog._task_editor.set_value("SINGLEPOINT")
    dialog._on_task_changed()
    assert "geom_method" not in dialog.collect_values()


def test_parameter_dialog_shows_catalog_fallback_notice(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        param_dialog_module,
        "keyword_catalog_source_info",
        lambda: type("CatalogSource", (), {"path": "/tmp/discovered/castep_keywords.json", "origin": "auto_discovery"})(),
    )

    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
    )

    assert dialog.findChild(QLabel, "catalogFallbackNotice") is not None


def test_parameter_dialog_ignores_malformed_real_vector_values_in_set_values(qapp) -> None:
    specs = [
        KeywordSpec(
            name="task",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("GEOMETRYOPTIMIZATION",),
            widget_hint="dropdown",
            basic_workflows=("pristine_relaxation",),
            default_value="GEOMETRYOPTIMIZATION",
        ),
        KeywordSpec(
            name="quantization_axis",
            file="cell",
            kind="scalar",
            value_type="real_vector",
            widget_hint="real_vector",
            basic_workflows=("pristine_relaxation",),
        ),
    ]
    dialog = ParameterDialog(
        specs=specs,
        workflow_stage="pristine_relaxation",
    )

    dialog.set_values({"quantization_axis": "D"})

    assert dialog.collect_values()["quantization_axis"] == (0.0, 0.0, 0.0)
