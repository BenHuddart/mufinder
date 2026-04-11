"""Tests for metadata-driven CASTEP parameter form behavior."""

from __future__ import annotations

from mufinder.app.param_form import CastepParamForm
from mufinder.castep_io.schema import KeywordSpec


def _form_specs() -> list[KeywordSpec]:
    return [
        KeywordSpec(
            name="task",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("SINGLEPOINT", "GEOMETRYOPTIMIZATION"),
            widget_hint="dropdown",
            category="workflow",
            basic_workflows=("convergence_cutoff",),
            default_value="SINGLEPOINT",
        ),
        KeywordSpec(
            name="basis_precision",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("FINE", "PRECISE"),
            widget_hint="dropdown",
            category="electronic",
            advanced_workflows=("convergence_cutoff",),
            default_value="FINE",
        ),
        KeywordSpec(
            name="spin_polarized",
            file="param",
            kind="scalar",
            value_type="bool",
            widget_hint="checkbox",
            category="electronic",
            basic_workflows=("convergence_cutoff",),
            mutually_exclusive_with=("spin_polarised",),
            default_value=False,
        ),
        KeywordSpec(
            name="spin_polarised",
            file="param",
            kind="scalar",
            value_type="bool",
            widget_hint="checkbox",
            category="electronic",
            basic_workflows=("convergence_cutoff",),
            mutually_exclusive_with=("spin_polarized",),
            default_value=False,
        ),
    ]


def test_param_form_collect_and_set_values(qapp) -> None:
    form = CastepParamForm(
        specs=_form_specs(),
        workflow_stage="convergence_cutoff",
        include_advanced=False,
    )

    values = form.collect_values()
    assert values["task"] == "SINGLEPOINT"
    assert "basis_precision" not in values

    form.set_advanced_visible(True)
    values_with_advanced = form.collect_values()
    assert values_with_advanced["basis_precision"] == "FINE"

    form.set_values({"task": "GEOMETRYOPTIMIZATION", "basis_precision": "PRECISE"})
    updated = form.collect_values()
    assert updated["task"] == "GEOMETRYOPTIMIZATION"
    assert updated["basis_precision"] == "PRECISE"


def test_param_form_advanced_toggle_controls_visibility(qapp) -> None:
    form = CastepParamForm(
        specs=_form_specs(),
        workflow_stage="convergence_cutoff",
        include_advanced=False,
    )

    assert form._rows["basis_precision"].container.isHidden() is True
    form.set_advanced_visible(True)
    assert form._rows["basis_precision"].container.isHidden() is False


def test_param_form_validate_conflicting_booleans(qapp) -> None:
    form = CastepParamForm(
        specs=_form_specs(),
        workflow_stage="convergence_cutoff",
        include_advanced=True,
    )

    form.set_values({"spin_polarized": True, "spin_polarised": True})
    errors = form.validate()
    assert any("conflicts with" in error for error in errors)
