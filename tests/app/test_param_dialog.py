"""Tests for standalone CASTEP parameter dialog behavior."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QStyle,
    QWidget,
)

import mufinder.app.param_dialog as param_dialog_module
from mufinder.app.param_widgets import HubbardUParamEditor
from mufinder.app.param_dialog import ParameterDialog
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import BlockSection, ValueWithUnit
from tests.support.keyword_catalogs import catalog_entry, refresh_from_catalog
from tests.support.qt import (
    buttons_left_to_right,
    open_and_take_focus,
    press_key,
    selected_text,
)


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


def test_parameter_dialog_uses_locked_task_value_when_overridden(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_cutoff",
        locked_values={"task": "PHONON"},
    )

    if dialog._task_editor is not None:
        assert dialog._task_editor.value() == "PHONON"
    else:
        assert dialog._task_display.text() == "PHONON"
    assert dialog.collect_values()["task"] == "PHONON"


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


def test_parameter_dialog_exclusive_radios_are_mutually_exclusive(qapp) -> None:
    # Regression: each row's radio is parented to its own label container,
    # so Qt's per-parent auto-exclusivity does not span rows. The dialog
    # registers every radio in a shared QButtonGroup so checking one
    # always unchecks the others in the same exclusive_group.
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

    bp_radio = dialog._rows["basis_precision"].radio
    ce_radio = dialog._rows["cut_off_energy"].radio
    assert bp_radio is not None and ce_radio is not None

    # Whichever was selected at construction stays selected; the other is off.
    assert bp_radio.isChecked() ^ ce_radio.isChecked()

    # Checking the inactive one must auto-uncheck the active one.
    bp_radio.setChecked(True)
    assert bp_radio.isChecked() is True
    assert ce_radio.isChecked() is False

    ce_radio.setChecked(True)
    assert ce_radio.isChecked() is True
    assert bp_radio.isChecked() is False


def test_parameter_dialog_preferred_exclusive_overrides_default_preference(qapp) -> None:
    # Regression: when both halves of a mutually exclusive pair are pre-filled
    # (for UX toggle convenience), preferred_exclusive must select the named
    # member even when _PREFERRED_EXCLUSIVE_NAMES would otherwise pick the other.
    # Here basis_precision is NOT in _PREFERRED_EXCLUSIVE_NAMES but cut_off_energy
    # IS. With preferred_exclusive="basis_precision", basis_precision should win.
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
    )
    dialog.set_advanced_names({"basis_precision", "cut_off_energy"})
    dialog.set_values(
        {
            "basis_precision": "PRECISE",
            "cut_off_energy": ValueWithUnit(420.0, "eV"),
        },
        preferred_exclusive="basis_precision",
    )

    assert dialog._rows["basis_precision"].radio.isChecked()  # type: ignore[union-attr]
    assert not dialog._rows["cut_off_energy"].radio.isChecked()  # type: ignore[union-attr]
    values = dialog.collect_values()
    assert values["basis_precision"] == "PRECISE"
    assert "cut_off_energy" not in values


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


@pytest.mark.parametrize("preferred", ["basis_precision", "cut_off_energy"])
def test_parameter_dialog_constructor_seed_matches_the_setters(qapp, preferred) -> None:
    seed = {
        "xc_functional": "LDA",
        "basis_precision": "PRECISE",
        "cut_off_energy": ValueWithUnit(420.0, "eV"),
    }
    seeded = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
        values=dict(seed),
        preferred_exclusive=preferred,
        advanced_names={"basis_precision", "cut_off_energy"},
        mixed_names={"xc_functional"},
    )
    stepwise = ParameterDialog(specs=_dialog_specs(), workflow_stage="convergence_kpoint")
    stepwise.set_values(dict(seed), preferred_exclusive=preferred)
    stepwise.set_advanced_names({"basis_precision", "cut_off_energy"})
    stepwise.set_mixed_names({"xc_functional"})

    assert seeded.collect_values() == stepwise.collect_values()
    assert seeded.advanced_names() == stepwise.advanced_names()
    assert seeded.untouched_mixed_names() == stepwise.untouched_mixed_names() == {"xc_functional"}
    assert list(seeded._rows) == list(stepwise._rows)
    assert seeded._rows[preferred].radio.isChecked()  # type: ignore[union-attr]
    assert seeded._mixed_markers["xc_functional"].isVisibleTo(seeded)


def test_parameter_dialog_uses_rich_hubbard_u_editor_when_advanced(qapp) -> None:
    specs = [
        *_dialog_specs(),
        KeywordSpec(
            name="hubbard_u",
            file="cell",
            kind="block",
            value_type="hubbard_u_block",
            widget_hint="hubbard_u",
            advanced_workflows=("convergence_kpoint",),
        ),
    ]
    dialog = ParameterDialog(
        specs=specs,
        workflow_stage="convergence_kpoint",
        species_options=("La", "Mn", "O"),
    )
    value = BlockSection(name="hubbard_u", rows=(("Ni", "d:", 2.4),), unit="eV")

    dialog.set_advanced_names({"hubbard_u"})
    dialog.set_values({"hubbard_u": value})

    assert isinstance(dialog._rows["hubbard_u"].editor, HubbardUParamEditor)
    assert dialog._rows["hubbard_u"].editor._species_options == ("La", "Mn", "O")
    assert dialog.collect_values()["hubbard_u"] == value
    assert set(dialog.advanced_names()) == {"hubbard_u"}


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
        button
        for button in dialog._table_widget.findChildren(QPushButton)
        if button.text() == "Reset"
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


def test_parameter_dialog_read_only_disables_editing_controls(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
        read_only=True,
    )
    dialog.set_advanced_names({"basis_precision", "cut_off_energy"})
    dialog.set_values({"xc_functional": "LDA", "cut_off_energy": ValueWithUnit(420.0, "eV")})

    assert dialog._read_only is True
    assert dialog._rows["xc_functional"].editor is not None
    assert not dialog._rows["xc_functional"].editor.isEnabled()
    assert dialog._rows["cut_off_energy"].radio is not None
    assert not dialog._rows["cut_off_energy"].radio.isEnabled()
    assert not dialog._add_advanced_button.isEnabled()
    reset_buttons = [
        button
        for button in dialog._table_widget.findChildren(QPushButton)
        if button.text() == "Reset"
    ]
    assert reset_buttons
    assert all(not button.isEnabled() for button in reset_buttons)
    assert dialog.collect_values()["xc_functional"] == "LDA"


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

    assert dialog._displayed_names() == ["geom_method"]
    # Filtered rows are hidden, not dropped from the dialog.
    assert {"geom_method", "xc_functional", "basis_precision"} <= set(dialog._rows)
    assert dialog._rows["xc_functional"].editor.isHidden()  # type: ignore[union-attr]
    assert not dialog._rows["geom_method"].editor.isHidden()  # type: ignore[union-attr]

    dialog._search_input.setText("")

    assert dialog._displayed_names() == list(dialog._rows)
    assert not dialog._rows["xc_functional"].editor.isHidden()  # type: ignore[union-attr]


def test_parameter_dialog_search_keeps_hidden_values_and_validation(qapp, monkeypatch) -> None:
    # Regression: pressing OK with a search active used to collect only the
    # rows on screen, so the caller stored a configuration missing every
    # hidden parameter.
    seen: list[dict] = []
    monkeypatch.setattr(
        param_dialog_module,
        "validate_gui_values",
        lambda values, _specs, task: seen.append(dict(values)) or ["checked"],
    )
    dialog = ParameterDialog(specs=_dialog_specs(), workflow_stage="convergence_kpoint")
    dialog.set_values({"xc_functional": "LDA"})
    collected = dialog.collect_values()
    errors = dialog.validate()

    dialog._search_input.setText("zzzz-no-match")

    assert dialog._displayed_names() == []
    assert dialog.collect_values() == collected
    assert collected["xc_functional"] == "LDA"
    assert dialog.validate() == errors
    assert seen[-1] == seen[0] == collected


def test_parameter_dialog_search_matching_one_exclusive_member_keeps_the_group(qapp) -> None:
    dialog = ParameterDialog(specs=_dialog_specs(), workflow_stage="convergence_kpoint")
    dialog.set_advanced_names({"basis_precision", "cut_off_energy"})
    dialog.set_values({"cut_off_energy": ValueWithUnit(420.0, "eV")})
    dialog._rows["basis_precision"].radio.setChecked(True)  # type: ignore[union-attr]
    groups = dict(dialog._exclusive_groups)
    collected = dialog.collect_values()
    assert collected["basis_precision"] == "FINE"
    assert "cut_off_energy" not in collected

    dialog._search_input.setText("cut_off")

    assert dialog._displayed_names() == ["cut_off_energy"]
    assert dialog._exclusive_groups == groups
    assert dialog._in_inactive_exclusive_choice("cut_off_energy")
    assert not dialog._rows["cut_off_energy"].radio.isChecked()  # type: ignore[union-attr]
    assert dialog.collect_values() == collected

    dialog._search_input.setText("")

    assert dialog._rows["basis_precision"].radio.isChecked()  # type: ignore[union-attr]
    assert dialog.collect_values() == collected


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


def test_parameter_dialog_notice_only_appears_on_the_limited_keyword_set(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _status(*, is_starter: bool):
        return SimpleNamespace(is_starter=is_starter)

    monkeypatch.setattr(
        param_dialog_module, "keyword_catalog_status", lambda: _status(is_starter=True)
    )
    dialog = ParameterDialog(specs=_dialog_specs(), workflow_stage="convergence_kpoint")
    notice = dialog.findChild(QLabel, "catalogFallbackNotice")
    assert notice is not None
    assert "some CASTEP parameters are not offered here" in notice.text()
    assert "Setup > CASTEP" in notice.text()

    # A full catalog, however it was found, needs no notice.
    monkeypatch.setattr(
        param_dialog_module, "keyword_catalog_status", lambda: _status(is_starter=False)
    )
    dialog = ParameterDialog(specs=_dialog_specs(), workflow_stage="convergence_kpoint")
    assert dialog.findChild(QLabel, "catalogFallbackNotice") is None


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


def _invalid_when_incomplete_specs() -> list[KeywordSpec]:
    """Specs where ``xc_functional`` is active but requires a missing keyword.

    The ``requires`` constraint yields a validation error that survives
    ``collect_values`` (unlike, e.g., a negative spinbox value that the editor
    would clamp), giving a deterministic way to drive the accept-blocking path.
    ``max_scf_cycles`` is a real CASTEP keyword so that, once added, the values
    also pass the per-keyword validators (a fictional companion would trip an
    "unsupported keyword" error and never validate clean).
    """

    return [
        KeywordSpec(
            name="task",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("SINGLEPOINT",),
            widget_hint="dropdown",
            category="workflow",
            basic_workflows=("convergence_kpoint",),
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
            basic_workflows=("convergence_kpoint",),
            default_value="PBE",
            requires=("max_scf_cycles",),
        ),
        KeywordSpec(
            name="max_scf_cycles",
            file="param",
            kind="scalar",
            value_type="positive_int",
            widget_hint="int",
            category="electronic",
            advanced_workflows=("convergence_kpoint",),
            default_value=30,
        ),
    ]


def _spy_on_accept(dialog: ParameterDialog) -> dict[str, bool]:
    """Replace ``dialog.accept`` with a recorder that never closes the dialog.

    ``dialog.result()`` reads 0 both before and after a *blocked* accept, so it
    is a weak "stayed open" signal. Spying on ``accept`` gives an unambiguous
    assertion: it fires only when validation passes.
    """

    calls = {"accepted": False}

    def _record() -> None:
        calls["accepted"] = True

    dialog.accept = _record  # type: ignore[method-assign]
    return calls


def test_parameter_dialog_ok_blocks_accept_when_invalid(qapp, monkeypatch) -> None:
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        param_dialog_module.QMessageBox,
        "warning",
        lambda _parent, title, text, *args, **kwargs: warnings.append((title, text)),
    )

    dialog = ParameterDialog(
        specs=_invalid_when_incomplete_specs(),
        workflow_stage="convergence_kpoint",
    )
    # xc_functional is active by default and requires max_scf_cycles, which is
    # an advanced row not currently added -> validation must report an error.
    assert dialog.validate()
    accepted = _spy_on_accept(dialog)

    dialog._accept_if_valid()

    assert accepted["accepted"] is False, "invalid parameters must not accept the dialog"
    assert warnings, "a warning must surface the validation errors"
    assert warnings[0][0] == "Invalid parameters"
    assert "max_scf_cycles" in warnings[0][1]
    # Values are preserved so the user can correct them in place.
    assert dialog.collect_values()["xc_functional"] == "PBE"


def test_parameter_dialog_ok_accepts_when_valid(qapp, monkeypatch) -> None:
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        param_dialog_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    dialog = ParameterDialog(
        specs=_invalid_when_incomplete_specs(),
        workflow_stage="convergence_kpoint",
    )
    # Adding the required companion keyword makes it active -> no errors.
    dialog.set_advanced_names({"max_scf_cycles"})
    assert dialog.validate() == []
    accepted = _spy_on_accept(dialog)

    dialog._accept_if_valid()

    assert accepted["accepted"] is True
    assert warnings == []


def test_parameter_dialog_read_only_closes_without_validation(qapp, monkeypatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        ParameterDialog,
        "validate",
        lambda self: called.append("validated") or [],
    )

    dialog = ParameterDialog(
        specs=_invalid_when_incomplete_specs(),
        workflow_stage="convergence_kpoint",
        read_only=True,
    )

    button_box = dialog.findChild(QDialogButtonBox)
    assert button_box is not None
    assert button_box.button(QDialogButtonBox.StandardButton.Close) is not None
    assert button_box.button(QDialogButtonBox.StandardButton.Ok) is None
    # Read-only has no OK button and no accept path, so nothing validates.
    assert called == []


def test_quantum_corrections_dialog_inherits_accept_blocking(qapp, monkeypatch) -> None:
    from mufinder.app.workflow_dialogs import QuantumCorrectionsConfigDialog

    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        param_dialog_module.QMessageBox,
        "warning",
        lambda _parent, title, text, *args, **kwargs: warnings.append((title, text)),
    )

    specs = [
        KeywordSpec(
            name="task",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("SINGLEPOINT", "PHONON"),
            widget_hint="dropdown",
            category="workflow",
            basic_workflows=("quantum_corrections",),
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
            basic_workflows=("quantum_corrections",),
            default_value="PBE",
            requires=("max_scf_cycles",),
        ),
        KeywordSpec(
            name="max_scf_cycles",
            file="param",
            kind="scalar",
            value_type="positive_int",
            widget_hint="int",
            category="electronic",
            advanced_workflows=("quantum_corrections",),
            default_value=30,
        ),
    ]

    dialog = QuantumCorrectionsConfigDialog(specs=specs, num_farms=4)
    assert dialog.validate()

    accepted = {"accepted": False}
    dialog.accept = lambda: accepted.__setitem__("accepted", True)  # type: ignore[method-assign]

    dialog._accept_if_valid()

    assert accepted["accepted"] is False
    assert warnings and warnings[0][0] == "Invalid parameters"
    # The farm count does not flow through validate_gui_values and remains
    # untouched by the blocked accept.
    assert dialog.num_farms() == 4


def _neb_dialog_specs() -> list[KeywordSpec]:
    return [
        KeywordSpec(
            name="task",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("TRANSITIONSTATESEARCH",),
            widget_hint="dropdown",
            category="workflow",
            basic_workflows=("energy_barriers",),
            default_value="TRANSITIONSTATESEARCH",
        ),
        KeywordSpec(
            name="tssearch_max_path_points",
            file="param",
            kind="scalar",
            value_type="positive_int",
            widget_hint="int",
            category="workflow",
            basic_workflows=("energy_barriers",),
            default_value=11,
        ),
    ]


def test_nudged_elastic_band_dialog_reports_farm_split(qapp) -> None:
    from mufinder.app.workflow_dialogs import NudgedElasticBandConfigDialog

    dialog = NudgedElasticBandConfigDialog(specs=_neb_dialog_specs(), num_farms=3)

    assert dialog.num_farms() == 3
    # 11 images over 3 farms is CASTEP's 4/4/3 block split.
    assert "4/4/3" in dialog._farm_hint.text()
    assert "multiple of 3" in dialog._farm_hint.text()
    dialog.close()


def test_nudged_elastic_band_dialog_hint_tracks_image_count(qapp) -> None:
    from mufinder.app.workflow_dialogs import NudgedElasticBandConfigDialog

    dialog = NudgedElasticBandConfigDialog(specs=_neb_dialog_specs(), num_farms=4)
    dialog.set_values({"tssearch_max_path_points": 8})

    # The hint is derived from the dialog's current image count, not the default.
    assert "2/2/2/2" in dialog._farm_hint.text()
    assert "100% balance" in dialog._farm_hint.text()
    dialog.close()


def test_nudged_elastic_band_dialog_farming_off_by_default(qapp) -> None:
    from mufinder.app.workflow_dialogs import NudgedElasticBandConfigDialog

    dialog = NudgedElasticBandConfigDialog(specs=_neb_dialog_specs())

    assert dialog.num_farms() == 1
    assert "disabled" in dialog._farm_hint.text().lower()
    dialog.close()


def test_nudged_elastic_band_dialog_read_only_disables_farm_spin(qapp) -> None:
    from mufinder.app.workflow_dialogs import NudgedElasticBandConfigDialog

    dialog = NudgedElasticBandConfigDialog(
        specs=_neb_dialog_specs(),
        num_farms=2,
        read_only=True,
    )

    assert dialog._num_farms_spin.isEnabled() is False
    dialog.close()


def test_nudged_elastic_band_dialog_blocks_num_farms_keyword_row(qapp) -> None:
    """NUM_FARMS is not offered as a parameter row: it has a dedicated control.

    The same value is read back from the row's extras to pick the expected
    output filenames and to validate MPI ranks at submission, so a second
    keyword-row copy could win in the rendered .param while those checks used
    the spin box -- CASTEP farming one way, MuFinder assuming another."""

    from mufinder.app.workflow_dialogs import NudgedElasticBandConfigDialog

    specs = [
        *_neb_dialog_specs(),
        KeywordSpec(
            name="num_farms",
            file="param",
            kind="scalar",
            value_type="positive_int",
            widget_hint="int",
            category="parallelism",
            advanced_workflows=("energy_barriers",),
            default_value=1,
        ),
    ]
    dialog = NudgedElasticBandConfigDialog(specs=specs, num_farms=2)

    assert "num_farms" not in dialog.collect_values()
    assert "num_farms" not in {spec.name for spec in dialog._available_advanced_specs()}
    # The dedicated control still carries the value.
    assert dialog.num_farms() == 2
    dialog.close()


# ---------------------------------------------------------------------------
# Mixed values: one dialog editing several rows
# ---------------------------------------------------------------------------


def _mixed_dialog(*mixed: str) -> ParameterDialog:
    dialog = ParameterDialog(specs=_dialog_specs(), workflow_stage="convergence_kpoint")
    dialog.set_advanced_names({"basis_precision", "cut_off_energy"})
    dialog.set_values({"cut_off_energy": ValueWithUnit(420.0, "eV")})
    dialog.set_mixed_names(mixed)
    return dialog


def test_parameter_dialog_without_mixed_names_shows_no_markers(qapp) -> None:
    dialog = ParameterDialog(specs=_dialog_specs(), workflow_stage="convergence_kpoint")
    dialog.set_values({"xc_functional": "LDA"})

    assert dialog.untouched_mixed_names() == set()
    assert dialog._mixed_markers == {}
    assert dialog._mixed_notice.isHidden()
    assert dialog.collect_values()["xc_functional"] == "LDA"


def test_parameter_dialog_mixed_parameter_is_marked_and_left_out_until_edited(qapp) -> None:
    dialog = _mixed_dialog("xc_functional")

    assert dialog.untouched_mixed_names() == {"xc_functional"}
    assert not dialog._mixed_notice.isHidden()
    assert not dialog._mixed_markers["xc_functional"].isHidden()
    # Every other row carries a marker label too, but hidden.
    assert dialog._mixed_markers["cut_off_energy"].isHidden()
    collected = dialog.collect_values()
    assert "xc_functional" not in collected
    assert collected["cut_off_energy"].value == 420.0

    dialog._rows["xc_functional"].editor.set_value("LDA")  # type: ignore[union-attr]

    assert dialog.untouched_mixed_names() == set()
    assert dialog._mixed_markers["xc_functional"].isHidden()
    assert dialog.collect_values()["xc_functional"] == "LDA"


def test_parameter_dialog_reset_on_a_mixed_parameter_applies_the_default_to_every_row(qapp) -> None:
    dialog = _mixed_dialog("xc_functional")

    # The editor already shows the default, so resetting emits no change; it
    # must still count as the user choosing that value.
    dialog._reset_parameter("xc_functional")

    assert dialog.untouched_mixed_names() == set()
    assert dialog.collect_values()["xc_functional"] == "PBE"


def test_parameter_dialog_mixed_marker_survives_a_row_rebuild(qapp) -> None:
    dialog = _mixed_dialog("xc_functional")

    dialog._search_input.setText("xc")
    dialog._search_input.setText("")

    assert dialog.untouched_mixed_names() == {"xc_functional"}
    assert not dialog._mixed_markers["xc_functional"].isHidden()


def test_parameter_dialog_mixed_member_keeps_its_whole_exclusive_group_untouched(qapp) -> None:
    # The rows may differ in which member of the pair they use, so neither
    # member is collected and both are reported, to be restored per row.
    dialog = _mixed_dialog("basis_precision")

    assert dialog.untouched_mixed_names() == {"basis_precision", "cut_off_energy"}
    assert not dialog._mixed_markers["cut_off_energy"].isHidden()
    collected = dialog.collect_values()
    assert "basis_precision" not in collected
    assert "cut_off_energy" not in collected


def test_parameter_dialog_choosing_an_exclusive_member_applies_the_group_to_every_row(qapp) -> None:
    dialog = _mixed_dialog("basis_precision")

    dialog._rows["basis_precision"].radio.setChecked(True)  # type: ignore[union-attr]

    assert dialog.untouched_mixed_names() == set()
    collected = dialog.collect_values()
    assert collected["basis_precision"] == "FINE"
    assert "cut_off_energy" not in collected


def test_parameter_dialog_editing_one_exclusive_member_clears_the_group(qapp) -> None:
    dialog = _mixed_dialog("basis_precision")

    dialog._rows["cut_off_energy"].editor.set_value(ValueWithUnit(500.0, "eV"))  # type: ignore[union-attr]

    assert dialog.untouched_mixed_names() == set()
    assert dialog.collect_values()["cut_off_energy"].value == 500.0


def test_parameter_dialog_removing_a_mixed_advanced_row_removes_it_from_every_row(qapp) -> None:
    dialog = _mixed_dialog("basis_precision")

    dialog._remove_advanced("basis_precision")

    assert dialog.untouched_mixed_names() == set()
    assert "basis_precision" not in dialog.advanced_names()


def test_parameter_dialog_validation_ignores_untouched_mixed_parameters(qapp, monkeypatch) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(
        param_dialog_module,
        "validate_gui_values",
        lambda values, _specs, task: seen.append(dict(values)) or [],
    )
    dialog = _mixed_dialog("xc_functional")

    assert dialog.validate() == []
    assert "xc_functional" not in seen[0]


def test_parameter_dialog_resetting_the_inactive_member_of_a_mixed_pair_keeps_it_mixed(
    qapp,
) -> None:
    dialog = _mixed_dialog("basis_precision")
    inactive = next(
        name
        for name in ("basis_precision", "cut_off_energy")
        if dialog._in_inactive_exclusive_choice(name)
    )

    dialog._reset_parameter(inactive)

    assert dialog.untouched_mixed_names() == {"basis_precision", "cut_off_energy"}
    collected = dialog.collect_values()
    assert "basis_precision" not in collected
    assert "cut_off_energy" not in collected


def test_parameter_dialog_resetting_the_active_member_of_a_mixed_pair_applies_it(qapp) -> None:
    dialog = _mixed_dialog("basis_precision")
    active = next(
        name
        for name in ("basis_precision", "cut_off_energy")
        if not dialog._in_inactive_exclusive_choice(name)
    )

    dialog._reset_parameter(active)

    assert dialog.untouched_mixed_names() == set()
    assert active in dialog.collect_values()


# --- Add Parameter picker ------------------------------------------------------


def _picker_spec(name: str, label: str, category: str, **extra) -> KeywordSpec:
    return KeywordSpec(
        name=name,
        file="param",
        kind="scalar",
        value_type="int",
        widget_hint="spinbox_int",
        label=label,
        description=f"{label} description.",
        category=category,
        advanced_workflows=("convergence_kpoint",),
        default_value=1,
        **extra,
    )


def _picker_rows(picker) -> list[tuple[str, list[str]]]:
    tree = picker._tree
    rows: list[tuple[str, list[str]]] = []
    for group_index in range(tree.topLevelItemCount()):
        group = tree.topLevelItem(group_index)
        rows.append((group.text(0), [group.child(i).text(0) for i in range(group.childCount())]))
    return rows


def _picker_dialog() -> ParameterDialog:
    specs = [
        *_dialog_specs(),
        _picker_spec("zeta_count", "Zeta Count", "electronic"),
        _picker_spec("alpha_count", "Alpha Count", "electronic"),
        _picker_spec("output_level", "Output Level", "io"),
        _picker_spec("kpoint_extra", "Extra Grid", "kpoints"),
        _picker_spec("farm_count", "Farm Count", "general", user_visible=False),
    ]
    return ParameterDialog(specs=specs, workflow_stage="convergence_kpoint")


def test_add_parameter_groups_by_category_and_sorts_by_name(qapp) -> None:
    dialog = _picker_dialog()
    picker = param_dialog_module._AdvancedSelectionDialog(
        dialog._available_advanced_specs(), dialog
    )

    # A keyword without a label is listed by its name, and one without a
    # category under General.
    assert _picker_rows(picker) == [
        ("Electronic", ["Alpha Count", "BASIS_PRECISION", "Zeta Count"]),
        ("General", ["CUT_OFF_ENERGY"]),
        ("Input and output", ["Output Level"]),
        ("K-points", ["Extra Grid"]),
    ]
    assert picker._tree.columnCount() == 2
    assert picker._tree.headerItem().text(0) == "Parameter"
    assert picker._tree.headerItem().text(1) == "Description"
    first = picker._tree.topLevelItem(0).child(0)
    assert first.text(1) == "Alpha Count description."
    assert picker._tree.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert picker._tree.wordWrap()


def test_add_parameter_never_offers_hidden_keywords(qapp) -> None:
    dialog = _picker_dialog()

    names = {spec.name for spec in dialog._available_advanced_specs()}

    assert "farm_count" not in names
    assert {"zeta_count", "alpha_count", "output_level", "kpoint_extra"} <= names


def test_hidden_keyword_with_a_stored_value_still_shows_as_a_row(qapp) -> None:
    dialog = _picker_dialog()
    dialog.set_values({"farm_count": 4})
    dialog.set_advanced_names(())

    assert "farm_count" in dialog._rows
    assert dialog.collect_values()["farm_count"] == 4


def test_add_parameter_search_hides_empty_groups_and_returns_checked_names(qapp) -> None:
    dialog = _picker_dialog()
    picker = param_dialog_module._AdvancedSelectionDialog(
        dialog._available_advanced_specs(), dialog
    )

    picker._search.setText("output")

    tree = picker._tree
    visible_groups = [
        tree.topLevelItem(i).text(0)
        for i in range(tree.topLevelItemCount())
        if not tree.topLevelItem(i).isHidden()
    ]
    assert visible_groups == ["Input and output"]

    picker._search.setText("")
    tree.topLevelItem(0).child(2).setCheckState(0, Qt.CheckState.Checked)
    tree.topLevelItem(2).child(0).setCheckState(0, Qt.CheckState.Checked)

    assert picker.selected_names() == ("zeta_count", "output_level")


def test_add_parameter_rows_fit_the_check_box_and_the_wrapped_description(qapp) -> None:
    long_text = " ".join(["A long description that wraps."] * 8)
    spec = KeywordSpec(
        name="wordy",
        file="param",
        kind="scalar",
        value_type="int",
        label="Wordy",
        description=long_text,
        category="general",
    )
    picker = param_dialog_module._AdvancedSelectionDialog([spec])
    picker.resize(480, 400)
    picker.show()
    qapp.processEvents()

    tree = picker._tree
    item = tree.topLevelItem(0).child(0)
    row_height = tree.visualItemRect(item).height()
    line_height = tree.fontMetrics().height()
    indicator = tree.style().pixelMetric(QStyle.PixelMetric.PM_IndicatorHeight, None, tree)
    assert row_height >= indicator
    assert row_height >= 2 * line_height
    picker.close()


@pytest.mark.parametrize(
    "workflow",
    ["muon_relaxation", "quantum_corrections", "energy_barriers", "hyperfine_coupling"],
)
def test_add_parameter_from_the_catalog_offers_no_internal_keywords(qapp, workflow) -> None:
    from mufinder.castep_io.gui_metadata import get_gui_keyword_specs_for_files

    specs = get_gui_keyword_specs_for_files(
        files=("param", "cell"), workflow_stage=workflow, include_advanced=True
    )
    dialog = ParameterDialog(specs=specs, workflow_stage=workflow)

    offered = {spec.name for spec in dialog._available_advanced_specs()}

    assert offered
    assert offered.isdisjoint(
        {"unit_cell", "num_farms", "magres_task", "devel_code", "tssearch_lst_max_iter"}
    )


def _retired_keyword_dialog(tmp_path, monkeypatch) -> ParameterDialog:
    from mufinder.castep_io.gui_metadata import get_gui_keyword_specs_for_files

    refresh_from_catalog(
        tmp_path,
        monkeypatch,
        cell={"atomic_init": catalog_entry("Block", level="Dummy")},
        param={
            "task": catalog_entry("String"),
            "ppd_size_x": catalog_entry(
                "Integer",
                "This is an OBSOLETE form; NLXC_PPD_SIZE_X replaces it.",
                level="Intermediate",
            ),
            "nlxc_ppd_size_x": catalog_entry("Integer", level="Intermediate"),
            "nlxc_exchange_fraction": catalog_entry(
                "Real", "The keyword has been removed.", level="Expert"
            ),
        },
    )
    specs = get_gui_keyword_specs_for_files(
        files=("param", "cell"), workflow_stage="muon_relaxation", include_advanced=True
    )
    return ParameterDialog(specs=specs, workflow_stage="muon_relaxation")


def test_add_parameter_leaves_out_obsolete_removed_and_dummy_level_keywords(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog = _retired_keyword_dialog(tmp_path, monkeypatch)

    offered = {spec.name for spec in dialog._available_advanced_specs()}

    assert "nlxc_ppd_size_x" in offered
    assert offered.isdisjoint({"ppd_size_x", "nlxc_exchange_fraction", "atomic_init"})


def test_obsolete_keyword_with_a_stored_value_still_shows_as_a_row(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog = _retired_keyword_dialog(tmp_path, monkeypatch)
    dialog.set_values({"ppd_size_x": 2})
    dialog.set_advanced_names(())

    assert "ppd_size_x" in dialog._rows
    assert dialog.collect_values()["ppd_size_x"] == 2


def test_removed_keyword_with_a_stored_value_still_shows_as_a_row(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog = _retired_keyword_dialog(tmp_path, monkeypatch)
    dialog.set_values({"nlxc_exchange_fraction": 0.25})
    dialog.set_advanced_names(())

    assert "nlxc_exchange_fraction" in dialog._rows
    assert dialog.collect_values()["nlxc_exchange_fraction"] == 0.25


# CASTEP's catalog types each vector-valued cell keyword in one of three ways:
# as an integer vector, as a real vector, or as a string (the Monkhorst-Pack
# offsets, which also take fractions or a keyword). The docstrings are stand-ins.
_VECTOR_KEYWORD_CATALOG = {
    "kpoint_mp_grid": catalog_entry("Integer Vector"),
    "kpoint_mp_offset": catalog_entry("String", "Default value: Determined from another keyword"),
    "supercell_kpoint_mp_grid": catalog_entry("Integer Vector"),
    "supercell_kpoints_mp_offset": catalog_entry(
        "String", "Offset of the supercell grid.  Default value: (0, 0, 0)."
    ),
    "quantization_axis": catalog_entry("Real Vector", "Default value: along c vector"),
}


def _vector_keyword_dialog(tmp_path, monkeypatch) -> ParameterDialog:
    from mufinder.castep_io.gui_metadata import get_gui_keyword_specs_for_files

    refresh_from_catalog(
        tmp_path,
        monkeypatch,
        cell=_VECTOR_KEYWORD_CATALOG,
        param={"task": catalog_entry("String")},
    )
    specs = get_gui_keyword_specs_for_files(
        files=("param", "cell"), workflow_stage="muon_relaxation", include_advanced=True
    )
    return ParameterDialog(specs=specs, workflow_stage="muon_relaxation")


def _add_through_add_parameter(dialog: ParameterDialog, monkeypatch, names) -> None:
    """Press Add Parameter and tick ``names`` in the picker."""

    picker = param_dialog_module._AdvancedSelectionDialog
    monkeypatch.setattr(picker, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(picker, "selected_names", lambda self: tuple(names))
    dialog._add_advanced_button.click()


def _press_ok(dialog: ParameterDialog, monkeypatch) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    monkeypatch.setattr(
        param_dialog_module.QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    accepted = _spy_on_accept(dialog)
    buttons = dialog.findChild(QDialogButtonBox)
    buttons.button(QDialogButtonBox.StandardButton.Ok).click()
    return accepted["accepted"], warnings


@pytest.mark.parametrize("name", sorted(_VECTOR_KEYWORD_CATALOG))
def test_vector_keyword_added_and_accepted_untouched_passes_validation(
    qapp, tmp_path, monkeypatch, name: str
) -> None:
    dialog = _vector_keyword_dialog(tmp_path, monkeypatch)
    offered = {spec.name for spec in dialog._available_advanced_specs()}
    if name not in offered:
        pytest.skip(f"{name} is a standing row in this workflow, not offered by Add Parameter")

    _add_through_add_parameter(dialog, monkeypatch, [name])
    assert name in dialog._rows

    accepted, warnings = _press_ok(dialog, monkeypatch)

    assert warnings == []
    assert accepted is True
    dialog.close()


def test_every_keyword_add_parameter_offers_starts_at_a_value_validation_accepts(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog = _vector_keyword_dialog(tmp_path, monkeypatch)
    offered = sorted(spec.name for spec in dialog._available_advanced_specs())
    assert {"kpoint_mp_offset", "supercell_kpoints_mp_offset", "quantization_axis"} <= set(offered)
    dialog.close()

    for name in offered:
        dialog = _vector_keyword_dialog(tmp_path, monkeypatch)
        _add_through_add_parameter(dialog, monkeypatch, [name])
        assert dialog.validate() == [], name
        dialog.close()


def test_offsets_added_untouched_start_at_zero_or_blank(qapp, tmp_path, monkeypatch) -> None:
    dialog = _vector_keyword_dialog(tmp_path, monkeypatch)
    _add_through_add_parameter(
        dialog, monkeypatch, ["kpoint_mp_offset", "supercell_kpoints_mp_offset"]
    )

    values = dialog.collect_values()

    # A real-vector offset starts at the origin; a text offset whose catalog
    # default is a tuple starts blank, so an untouched row writes nothing.
    assert values["kpoint_mp_offset"] == (0.0, 0.0, 0.0)
    assert "supercell_kpoints_mp_offset" not in values
    dialog.close()


def test_parameter_dialog_search_matches_the_group_heading(qapp) -> None:
    dialog = ParameterDialog(specs=_dialog_specs(), workflow_stage="convergence_kpoint")
    assert dialog._task_editor is not None
    dialog._task_editor.set_value("GEOMETRYOPTIMIZATION")
    dialog._on_task_changed()

    dialog._search_input.setText("geometry optimisation")

    assert dialog._displayed_names() == ["geom_method"]


def _symmetry_flag_spec() -> KeywordSpec:
    return KeywordSpec(
        name="symmetry_generate",
        file="cell",
        kind="flag",
        value_type="bool",
        widget_hint="flag",
        advanced_workflows=("pristine_relaxation",),
        default_value=True,
    )


def test_parameter_dialog_collects_an_unchecked_flag_as_false(qapp) -> None:
    dialog = ParameterDialog(
        specs=[_symmetry_flag_spec()],
        workflow_stage="pristine_relaxation",
        advanced_names={"symmetry_generate"},
    )
    assert dialog.collect_values()["symmetry_generate"] is True

    dialog._rows["symmetry_generate"].editor.set_value(False)

    assert dialog.collect_values()["symmetry_generate"] is False
    assert dialog.validate() == []


def test_parameter_dialog_leaves_out_a_flag_that_has_no_row(qapp) -> None:
    dialog = ParameterDialog(
        specs=[_symmetry_flag_spec()],
        workflow_stage="pristine_relaxation",
    )

    assert "symmetry_generate" not in dialog.collect_values()


def test_parameter_dialog_keeps_the_row_of_a_stored_disabled_flag(qapp) -> None:
    # Without its row the value would not be collected, so accepting the dialog
    # would hand the row back to the workflow default.
    dialog = ParameterDialog(
        specs=[_symmetry_flag_spec()],
        workflow_stage="pristine_relaxation",
        values={"symmetry_generate": False},
        advanced_names=(),
    )

    assert dialog._rows["symmetry_generate"].editor.value() is False
    assert dialog.collect_values()["symmetry_generate"] is False
    assert dialog.advanced_names() == ("symmetry_generate",)


def test_parameter_dialog_removing_a_disabled_flag_returns_it_to_the_default(qapp) -> None:
    dialog = ParameterDialog(
        specs=[_symmetry_flag_spec()],
        workflow_stage="pristine_relaxation",
        values={"symmetry_generate": False},
        advanced_names={"symmetry_generate"},
    )

    dialog._remove_advanced("symmetry_generate")

    assert "symmetry_generate" not in dialog.collect_values()
    assert dialog.advanced_names() == ()


NEB_LOCKED_VALUES = ("tssearch_method", "tssearch_neb_climbing")
NEB_EXPERT_KEYWORDS = (
    "tssearch_neb_spring_constant",
    "tssearch_neb_tangent_mode",
    "tssearch_neb_method",
    "tssearch_neb_max_iter",
    "tssearch_neb_normed",
)


def _neb_dialog(**kwargs):
    from mufinder.app.workflow_controllers.energy_barriers import EnergyBarriersController
    from mufinder.app.workflow_dialogs import NudgedElasticBandConfigDialog

    return NudgedElasticBandConfigDialog(specs=EnergyBarriersController.parameter_specs(), **kwargs)


def _table_rows(dialog) -> list[str]:
    """Return what the dialog's table shows, top to bottom, by the first column's text."""

    layout = dialog._table_layout
    rows: dict[int, str] = {}
    for index in range(layout.count()):
        row, column, _row_span, _column_span = layout.getItemPosition(index)
        widget = layout.itemAt(index).widget()
        if column != 0 or row == 0:
            continue
        label = widget if isinstance(widget, QLabel) else widget.findChild(QLabel)
        rows[row] = label.text()
    return [rows[row] for row in sorted(rows)]


def test_neb_dialog_default_rows_in_order(qapp) -> None:
    dialog = _neb_dialog(num_farms=2)

    assert _table_rows(dialog) == [
        "Basis Precision",
        "Cutoff Energy",
        "SCF Energy Tolerance",
        "K-Point MP Grid",
        "K-Point MP Spacing",
        "Images",
        "",  # The note on what the image count means.
        "Force tolerance (eV/Å)",
        "Task farms",
        "XC Functional",
        "Spin Polarized",
        "TS Search Method",
        "Climbing image",
    ]
    assert dialog.num_farms() == 2
    assert dialog._num_farms_spin.accessibleName() == "Task farms"
    dialog.close()


def test_neb_dialog_says_what_the_image_count_means(qapp) -> None:
    dialog = _neb_dialog()

    note = next(row.value for row in dialog._workflow_rows if row.after.endswith("path_points"))
    assert "two more" in note.text()
    assert "rounded up" in note.text()
    dialog.close()


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("tssearch_max_path_points", "Images"),
        ("tssearch_force_tol", "Force tolerance (eV/Å)"),
        ("tssearch_neb_spring_constant", "Spring constant (eV/Å²)"),
        ("tssearch_neb_tangent_mode", "Tangent"),
        ("tssearch_neb_method", "NEB method"),
        ("tssearch_neb_max_iter", "Max iterations"),
        ("tssearch_neb_normed", "Normed"),
        ("tssearch_neb_climbing", "Climbing image"),
    ],
)
def test_neb_keywords_have_plain_labels_on_any_catalog(
    name: str, label: str, tmp_path, monkeypatch
) -> None:
    from mufinder.castep_io.schema import get_keyword_spec

    starter = get_keyword_spec("param", name)
    assert starter is not None and starter.label == label

    refresh_from_catalog(tmp_path, monkeypatch, cell={}, param={name: catalog_entry("Real")})
    full = get_keyword_spec("param", name)
    assert full is not None and full.label == label


def test_neb_dialog_units_are_in_the_labels_not_beside_the_editors(qapp) -> None:
    dialog = _neb_dialog(values={"tssearch_neb_spring_constant": ValueWithUnit(5.0, "eV/ang**2")})

    for name in ("tssearch_force_tol", "tssearch_neb_spring_constant"):
        editor = dialog._rows[name].editor
        assert [label.text() for label in editor.findChildren(QLabel)] == []
    # The value keeps its CASTEP unit, and other keywords still show theirs.
    values = dialog.collect_values()
    assert values["tssearch_force_tol"] == ValueWithUnit(0.05, "eV/ang")
    assert values["tssearch_neb_spring_constant"] == ValueWithUnit(5.0, "eV/ang**2")
    cutoff = dialog._rows["cut_off_energy"].editor
    assert [label.text() for label in cutoff.findChildren(QLabel)] == ["eV"]
    dialog.close()


def test_keyword_whose_unit_differs_from_its_label_keeps_the_unit_beside_the_editor(qapp) -> None:
    from mufinder.app.param_widgets import PhysicalParamEditor
    from mufinder.castep_io.gui_metadata import keyword_unit_in_label

    spec = KeywordSpec(
        name="tssearch_force_tol",
        file="param",
        kind="scalar",
        value_type="physical",
        unit="ha/bohr",
        label="Force tolerance (eV/Å)",
    )

    assert keyword_unit_in_label(spec) is False
    editor = PhysicalParamEditor(spec)
    assert [label.text() for label in editor.findChildren(QLabel)] == ["ha/bohr"]
    editor.deleteLater()


def test_neb_dialog_keeps_expert_keywords_behind_add_parameter(qapp) -> None:
    dialog = _neb_dialog()

    available = {spec.name for spec in dialog._available_advanced_specs()}
    assert set(NEB_EXPERT_KEYWORDS) <= available
    # An untouched dialog writes none of them, so CASTEP's defaults apply.
    assert set(NEB_EXPERT_KEYWORDS).isdisjoint(dialog.collect_values())
    assert set(NEB_EXPERT_KEYWORDS).isdisjoint(dialog._visible_names())
    dialog.close()


def test_neb_dialog_shows_a_spring_constant_configured_earlier(qapp) -> None:
    stored = ValueWithUnit(3.5, "eV/ang**2")
    dialog = _neb_dialog(values={"tssearch_neb_spring_constant": stored})

    assert "tssearch_neb_spring_constant" in dialog._visible_names()
    assert dialog.advanced_names() == ("tssearch_neb_spring_constant",)
    assert dialog.collect_values()["tssearch_neb_spring_constant"] == stored
    assert _table_rows(dialog).index("Spring constant (eV/Å²)") > _table_rows(dialog).index(
        "Task farms"
    )
    dialog.close()


def test_neb_dialog_locks_the_workflow_rows(qapp) -> None:
    dialog = _neb_dialog(
        values={"tssearch_method": "LSTQST"},
    )

    for name in NEB_LOCKED_VALUES:
        assert dialog._rows[name].locked is True
        assert dialog._rows[name].editor is None
    assert dialog._rows["tssearch_method"].cells[1].text() == "NEB"
    assert dialog._rows["tssearch_neb_climbing"].cells[1].text() == "TRUE"
    assert dialog._task == "TRANSITIONSTATESEARCH"
    dialog.close()


def test_neb_dialog_task_farm_row_survives_a_row_rebuild(qapp) -> None:
    dialog = _neb_dialog(num_farms=3, values={"tssearch_max_path_points": 11})
    spin = dialog._num_farms_spin
    spin.setValue(4)

    dialog.set_advanced_names(["tssearch_neb_max_iter"])

    assert dialog._num_farms_spin is spin
    assert dialog.num_farms() == 4
    assert "Max iterations" in _table_rows(dialog)
    assert _table_rows(dialog).count("Task farms") == 1
    assert "11 images over 4 farms" in dialog._farm_hint.text()
    dialog.close()


def test_neb_dialog_tab_order_reaches_task_farms_after_the_force_tolerance(qapp) -> None:
    dialog = _neb_dialog()

    # The screen's Tab order is set once the dialog is the active window.
    open_and_take_focus(dialog)
    tab_stops = []
    widget = dialog.nextInFocusChain()
    while widget is not dialog:
        if (
            widget.focusPolicy() & Qt.FocusPolicy.TabFocus
            and widget.isVisible()
            and widget.isEnabled()
        ):
            tab_stops.append(widget)
        widget = widget.nextInFocusChain()

    info_button = dialog._rows["tssearch_force_tol"].cells[3]
    spin = dialog._num_farms_spin
    assert tab_stops.index(info_button) < tab_stops.index(spin)
    following = tab_stops[tab_stops.index(info_button) + 1 :]
    assert following[0] is spin
    dialog.close()


def test_neb_dialog_search_filters_the_rows_without_a_keyword(qapp) -> None:
    dialog = _neb_dialog()
    dialog.show()
    farms_row, note_row = dialog._workflow_rows[1], dialog._workflow_rows[0]

    dialog._search_input.setText("farm")
    assert farms_row.value.isVisible() is True
    assert note_row.value.isVisible() is False
    assert dialog._rows["tssearch_force_tol"].cells[0].isVisible() is False

    dialog._search_input.setText("force")
    assert farms_row.value.isVisible() is False
    assert dialog._rows["tssearch_force_tol"].cells[0].isVisible() is True

    # The note follows the Images row it explains.
    dialog._search_input.setText("images")
    assert note_row.value.isVisible() is True
    assert dialog._rows["tssearch_max_path_points"].cells[0].isVisible() is True

    dialog._search_input.clear()
    assert farms_row.value.isVisible() is True
    assert note_row.value.isVisible() is True
    dialog.close()


def _tall_parameter_dialog(*, read_only: bool) -> ParameterDialog:
    """A dialog too short for its rows, shown and active."""

    specs = _dialog_specs()
    xc_functional = next(spec for spec in specs if spec.name == "xc_functional")
    dialog = ParameterDialog(
        specs=[
            *specs,
            *(replace(xc_functional, name=f"xc_copy_{index}") for index in range(20)),
        ],
        workflow_stage="convergence_kpoint",
        read_only=read_only,
    )
    dialog.resize(dialog.width(), 320)
    dialog.show()
    dialog.activateWindow()
    assert QTest.qWaitForWindowActive(dialog)
    QApplication.processEvents()
    assert dialog._scroll_area.verticalScrollBar().maximum() > 0
    return dialog


def test_tab_reaches_every_row_in_view_without_stopping_on_the_scroll_area(
    fresh_focus_tracker,
) -> None:
    dialog = _tall_parameter_dialog(read_only=False)
    area = dialog._scroll_area
    viewport = area.viewport()
    dialog._search_input.setFocus()
    area.verticalScrollBar().setValue(area.verticalScrollBar().maximum())

    stops_in_area = 0
    furthest = 0
    for _ in range(80):
        QTest.keyClick(dialog.windowHandle(), Qt.Key.Key_Tab)
        QApplication.processEvents()
        focused = QApplication.focusWidget()
        assert focused is not area
        if focused is dialog._search_input:
            break
        if area.isAncestorOf(focused):
            stops_in_area += 1
            furthest = max(furthest, area.verticalScrollBar().value())
            top_left = focused.mapTo(viewport, QPoint(0, 0))
            assert viewport.rect().contains(focused.rect().translated(top_left)), focused

    assert focused is dialog._search_input
    assert stops_in_area >= 20
    assert furthest == area.verticalScrollBar().maximum()
    dialog.close()


def test_read_only_dialog_keeps_the_stop_its_disabled_rows_need_and_covers_no_label(
    fresh_focus_tracker,
) -> None:
    dialog = _tall_parameter_dialog(read_only=True)
    area = dialog._scroll_area
    dialog._search_input.setFocus()
    pixels_before = area.viewport().grab().toImage()

    QTest.keyClick(dialog.windowHandle(), Qt.Key.Key_Tab)
    QApplication.processEvents()

    assert area.hasFocus()
    assert area.viewport().grab().toImage() == pixels_before
    QTest.keyClick(area, Qt.Key.Key_PageDown)
    assert area.verticalScrollBar().value() > 0
    dialog.close()


def _controls_in(widget: QWidget) -> list[QWidget]:
    """The shown, enabled controls in ``widget`` that Tab can stop on, in no set order."""

    if widget.focusPolicy() & Qt.FocusPolicy.TabFocus:
        return [widget] if widget.isVisible() and widget.isEnabled() else []
    return [
        control
        for child in widget.children()
        if isinstance(child, QWidget)
        for control in _controls_in(child)
    ]


def _reading_order(dialog: ParameterDialog) -> list[QWidget]:
    """What Tab should reach after the search box, by where each control is on screen.

    The table's controls go row by row, left to right in each row, then come
    ``+ Add Parameter`` and the dialog's buttons as the style lays them out.
    """

    table = dialog._table_layout

    def place(control: QWidget) -> tuple[int, int]:
        cell = control
        while cell.parentWidget() is not dialog._table_widget:
            cell = cell.parentWidget()
        row = table.getItemPosition(table.indexOf(cell))[0]
        return row, control.mapTo(dialog._table_widget, QPoint(0, 0)).x()

    assert dialog._add_advanced_button.isEnabled()
    return [
        *sorted(_controls_in(dialog._table_widget), key=place),
        dialog._add_advanced_button,
        *buttons_left_to_right(dialog.findChild(QDialogButtonBox)),
    ]


def _assert_tab_follows_the_screen(dialog: ParameterDialog) -> None:
    expected = _reading_order(dialog)
    dialog._search_input.setFocus()

    forward = [press_key(Qt.Key.Key_Tab) for _ in expected]
    backward = [press_key(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier) for _ in expected]

    assert forward == expected
    assert backward == [*reversed(expected[:-1]), dialog._search_input]


def test_tab_goes_from_the_search_box_to_the_first_row(qapp) -> None:
    dialog = _tall_parameter_dialog(read_only=False)
    first_row = dialog._rows[dialog._displayed_names()[0]]
    dialog._search_input.setFocus()

    focused = press_key(Qt.Key.Key_Tab)

    assert first_row.editor is not None
    assert first_row.editor.isAncestorOf(focused)
    dialog.close()


def test_tab_follows_the_rows_then_add_parameter_ok_and_cancel(qapp) -> None:
    dialog = _tall_parameter_dialog(read_only=False)
    last_row = dialog._rows[dialog._displayed_names()[-1]]

    expected = _reading_order(dialog)

    assert expected[-4] is last_row.cells[3]
    _assert_tab_follows_the_screen(dialog)
    dialog.close()


def test_tab_follows_the_screen_after_a_parameter_is_added(qapp) -> None:
    dialog = _tall_parameter_dialog(read_only=False)

    dialog.set_advanced_names({"basis_precision"})

    assert "basis_precision" in dialog._displayed_names()
    _assert_tab_follows_the_screen(dialog)
    dialog.close()


def test_tab_follows_the_screen_after_a_parameter_is_removed(qapp) -> None:
    dialog = _tall_parameter_dialog(read_only=False)
    dialog.set_advanced_names({"basis_precision"})

    dialog.set_advanced_names(())

    assert "basis_precision" not in dialog._rows
    _assert_tab_follows_the_screen(dialog)
    dialog.close()


def test_tab_follows_the_screen_after_the_search_filters_the_rows(qapp) -> None:
    dialog = _tall_parameter_dialog(read_only=False)
    dialog._search_input.setFocus()

    QTest.keyClicks(dialog._search_input, "copy_1")
    QApplication.processEvents()

    shown = dialog._displayed_names()
    assert 1 < len(shown) < len(dialog._rows)
    _assert_tab_follows_the_screen(dialog)
    dialog.close()


def test_tab_from_a_newly_chosen_exclusive_option_reaches_its_value_after_a_restore(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
        advanced_names=["basis_precision", "cut_off_energy"],
    )
    dialog.show()
    dialog.activateWindow()
    assert QTest.qWaitForWindowActive(dialog)
    chosen = dialog._rows["basis_precision"].radio
    other = dialog._rows["cut_off_energy"].radio
    chosen.setFocus()
    # Qt takes Tab focus from the option that is not chosen while its partner has focus.
    assert not other.focusPolicy() & Qt.FocusPolicy.TabFocus
    dialog.showMinimized()
    QApplication.processEvents()
    dialog.showNormal()
    dialog.activateWindow()
    assert QTest.qWaitForWindowActive(dialog)
    chosen.setFocus()

    assert press_key(Qt.Key.Key_Down) is other
    focused = press_key(Qt.Key.Key_Tab)

    assert dialog._rows["cut_off_energy"].editor.isAncestorOf(focused)
    dialog.close()


# Both options of every either-or pair hold a value, so the dialog can open on either.
_EITHER_OR_VALUES = {
    "basis_precision": "FINE",
    "cut_off_energy": ValueWithUnit(500.0, "eV"),
    "kpoint_mp_grid": (4, 4, 4),
    "kpoint_mp_spacing": 0.05,
}


def _either_or_dialog(kind: str, chosen: str) -> ParameterDialog:
    """A dialog of ``kind`` that opens with ``chosen`` as the choice of its pair."""

    from mufinder.app.workflow_controllers.quantum_corrections import (
        QuantumCorrectionsController,
    )
    from mufinder.app.workflow_dialogs import QuantumCorrectionsConfigDialog

    if kind == "parameter":
        values = {name: _EITHER_OR_VALUES[name] for name in ("basis_precision", "cut_off_energy")}
        return ParameterDialog(
            specs=_dialog_specs(),
            workflow_stage="convergence_kpoint",
            values=values,
            preferred_exclusive=chosen,
            advanced_names=list(values),
        )
    if kind == "energy_barriers":
        return _neb_dialog(values=dict(_EITHER_OR_VALUES), preferred_exclusive=chosen)
    return QuantumCorrectionsConfigDialog(
        specs=QuantumCorrectionsController.parameter_specs(),
        values=dict(_EITHER_OR_VALUES),
        preferred_exclusive=chosen,
    )


_EITHER_OR_OPENINGS = [
    *(("parameter", chosen) for chosen in ("basis_precision", "cut_off_energy")),
    *(
        (kind, chosen)
        for kind in ("energy_barriers", "quantum_corrections")
        for chosen in _EITHER_OR_VALUES
    ),
]


def _tab_round(backward: bool = False) -> list[QWidget]:
    """Press Tab, or Shift+Tab, until focus comes back; return each control it stopped on.

    The last stop is the control it started from.
    """

    modifiers = Qt.KeyboardModifier.ShiftModifier if backward else Qt.KeyboardModifier.NoModifier
    start = QApplication.focusWidget()
    stops: list[QWidget] = []
    for _ in range(200):
        focused = press_key(Qt.Key.Key_Tab, modifiers)
        stops.append(focused)
        if focused is start:
            return stops
    raise AssertionError("Tab never came back to where it started")


def _assert_tab_stops_on_the_chosen_options(
    dialog: ParameterDialog, *, backward: bool = False
) -> None:
    """Tab round the dialog stops on the chosen option of each pair, then on its value."""

    radios = {name: row.radio for name, row in dialog._rows.items() if row.radio is not None}
    assert radios

    stops = _tab_round(backward)

    for name, radio in radios.items():
        if not radio.isChecked():
            assert radio not in stops, f"Tab stopped on {name}, which is not chosen"
            continue
        assert radio in stops, f"Tab never stopped on {name}, which is chosen"
        if not backward:
            after = stops[(stops.index(radio) + 1) % len(stops)]
            assert dialog._rows[name].editor.isAncestorOf(after), name


@pytest.mark.parametrize("backward", [False, True], ids=["tab", "shift_tab"])
@pytest.mark.parametrize(("kind", "chosen"), _EITHER_OR_OPENINGS)
def test_tab_stops_on_the_chosen_option_of_each_pair_from_the_moment_the_dialog_opens(
    qapp, kind: str, chosen: str, backward: bool
) -> None:
    dialog = _either_or_dialog(kind, chosen)
    assert dialog._rows[chosen].radio.isChecked()

    open_and_take_focus(dialog)

    _assert_tab_stops_on_the_chosen_options(dialog, backward=backward)
    dialog.close()


@pytest.mark.parametrize("while_open", [False, True], ids=["before_opening", "while_open"])
@pytest.mark.parametrize(("kind", "chosen"), _EITHER_OR_OPENINGS)
def test_tab_stops_on_the_chosen_option_after_new_values_change_the_choice(
    qapp, kind: str, chosen: str, while_open: bool
) -> None:
    group = next(
        group
        for group in (
            ("basis_precision", "cut_off_energy"),
            ("kpoint_mp_grid", "kpoint_mp_spacing"),
        )
        if chosen in group
    )
    first = next(name for name in group if name != chosen)
    dialog = _either_or_dialog(kind, first)
    if while_open:
        open_and_take_focus(dialog)

    dialog.set_values(dict(_EITHER_OR_VALUES), preferred_exclusive=chosen)
    if not while_open:
        open_and_take_focus(dialog)

    assert dialog._rows[chosen].radio.isChecked()
    _assert_tab_stops_on_the_chosen_options(dialog)
    dialog.close()


@pytest.mark.parametrize("kind", ["parameter", "energy_barriers", "quantum_corrections"])
def test_tab_stops_on_an_option_chosen_without_the_keyboard(qapp, kind: str) -> None:
    dialog = _either_or_dialog(kind, "basis_precision")
    open_and_take_focus(dialog)

    dialog._rows["cut_off_energy"].radio.click()

    assert dialog._rows["cut_off_energy"].radio.isChecked()
    assert dialog._rows["cut_off_energy"].radio is not QApplication.focusWidget()
    _assert_tab_stops_on_the_chosen_options(dialog)
    dialog.close()


def test_tab_reaches_the_shown_option_while_the_search_hides_the_chosen_one(qapp) -> None:
    dialog = _either_or_dialog("parameter", "basis_precision")
    open_and_take_focus(dialog)
    shown = dialog._rows["cut_off_energy"].radio
    dialog._search_input.setFocus()

    QTest.keyClicks(dialog._search_input, "cut_off")
    QApplication.processEvents()

    assert not dialog._rows["basis_precision"].radio.isVisible()
    assert shown.isVisible()
    stops = _tab_round()
    assert shown in stops
    for _ in range(stops.index(shown) + 1):
        press_key(Qt.Key.Key_Tab)
    assert press_key(Qt.Key.Key_Space) is shown
    assert shown.isChecked()

    dialog._search_input.clear()
    _assert_tab_stops_on_the_chosen_options(dialog)
    dialog.close()


def test_tab_reaches_the_enabled_option_when_the_chosen_one_cannot_be_changed(qapp) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(),
        workflow_stage="convergence_kpoint",
        values={"basis_precision": "FINE", "cut_off_energy": ValueWithUnit(500.0, "eV")},
        preferred_exclusive="basis_precision",
        advanced_names=["basis_precision", "cut_off_energy"],
        editable_names={"cut_off_energy"},
    )
    open_and_take_focus(dialog)
    chosen = dialog._rows["basis_precision"].radio
    enabled = dialog._rows["cut_off_energy"].radio
    assert chosen.isChecked() and not chosen.isEnabled()

    assert enabled in _tab_round()
    dialog.close()


def test_tab_stops_on_the_chosen_option_after_values_load_while_the_dialog_is_disabled(
    qapp,
) -> None:
    dialog = _either_or_dialog("parameter", "basis_precision")
    open_and_take_focus(dialog)

    dialog.setEnabled(False)
    dialog.set_values(dict(_EITHER_OR_VALUES), preferred_exclusive="cut_off_energy")
    dialog.setEnabled(True)
    dialog._search_input.setFocus()

    _assert_tab_stops_on_the_chosen_options(dialog)
    dialog.close()


@pytest.mark.parametrize("kind", ["parameter", "energy_barriers", "quantum_corrections"])
def test_arrow_keys_still_move_the_choice_and_tab_follows_it(qapp, kind: str) -> None:
    dialog = _either_or_dialog(kind, "cut_off_energy")
    open_and_take_focus(dialog)
    chosen = dialog._rows["cut_off_energy"].radio
    other = dialog._rows["basis_precision"].radio
    start = QApplication.focusWidget()
    while QApplication.focusWidget() is not chosen:
        assert press_key(Qt.Key.Key_Tab) is not start

    assert press_key(Qt.Key.Key_Up) is other
    assert other.isChecked()
    assert dialog._rows["basis_precision"].editor.isAncestorOf(press_key(Qt.Key.Key_Tab))
    _assert_tab_stops_on_the_chosen_options(dialog)
    dialog.close()


@pytest.mark.parametrize(
    ("options", "opening"),
    [
        ({}, "task"),
        ({"intro_text": "Cutoff convergence rows."}, "task"),
        ({"read_only": True}, "search"),
    ],
)
def test_parameter_dialog_opens_on_the_task_or_the_search_box_with_nothing_selected(
    qapp, options: dict[str, object], opening: str
) -> None:
    dialog = ParameterDialog(specs=_dialog_specs(), workflow_stage="convergence_kpoint", **options)
    task_choice = dialog._task_editor.findChild(QComboBox)

    focused = open_and_take_focus(dialog)

    assert focused is (task_choice if opening == "task" else dialog._search_input)
    assert selected_text(focused) == ""
    dialog.close()


def test_neb_dialog_opens_on_its_locked_task_with_nothing_selected(qapp) -> None:
    dialog = _neb_dialog()

    focused = open_and_take_focus(dialog)

    assert focused is dialog._task_display
    assert selected_text(focused) == ""
    dialog.close()


@pytest.mark.parametrize("added", [("basis_precision",), ("basis_precision", "cut_off_energy")])
def test_removing_a_parameter_with_the_keyboard_returns_focus_to_the_task(
    qapp, added: tuple[str, ...]
) -> None:
    dialog = ParameterDialog(
        specs=_dialog_specs(), workflow_stage="convergence_kpoint", advanced_names=added
    )
    task_choice = dialog._task_editor.findChild(QComboBox)
    buttons = dialog.findChild(QDialogButtonBox)
    open_and_take_focus(dialog)
    assert dialog._add_advanced_button.isEnabled() is (len(added) == 1)
    dialog._rows[added[-1]].cells[4].setFocus()

    focused = press_key(Qt.Key.Key_Space)

    assert added[-1] not in dialog._rows
    assert focused is task_choice
    assert not buttons.isAncestorOf(focused)
    press_key(Qt.Key.Key_Space)
    task_choice.hidePopup()
    assert dialog.isVisible()
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.close()


def test_a_rebuild_while_the_dialog_is_inactive_returns_its_focus_to_the_task(qapp) -> None:
    added = ("basis_precision", "cut_off_energy")
    dialog = ParameterDialog(
        specs=_dialog_specs(), workflow_stage="convergence_kpoint", advanced_names=added
    )
    task_choice = dialog._task_editor.findChild(QComboBox)
    open_and_take_focus(dialog)
    dialog._rows["cut_off_energy"].cells[4].setFocus()
    other = QWidget()
    open_and_take_focus(other)

    dialog.set_advanced_names(added[:1])
    other.close()
    focused = open_and_take_focus(dialog)

    assert "cut_off_energy" not in dialog._rows
    assert focused is task_choice
    dialog.close()
