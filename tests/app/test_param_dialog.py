"""Tests for standalone CASTEP parameter dialog behavior."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDialogButtonBox, QLabel, QPushButton

import mufinder.app.param_dialog as param_dialog_module
from mufinder.app.param_widgets import HubbardUParamEditor
from mufinder.app.param_dialog import ParameterDialog
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import BlockSection, ValueWithUnit


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
        button for button in dialog._table_widget.findChildren(QPushButton) if button.text() == "Reset"
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

    dialog = QuantumCorrectionsConfigDialog(
        specs=specs,
        pseudopotential_family="norm_conserving",
        num_farms=4,
    )
    assert dialog.validate()

    accepted = {"accepted": False}
    dialog.accept = lambda: accepted.__setitem__("accepted", True)  # type: ignore[method-assign]

    dialog._accept_if_valid()

    assert accepted["accepted"] is False
    assert warnings and warnings[0][0] == "Invalid parameters"
    # The extra Quantum Corrections controls do not flow through
    # validate_gui_values and remain untouched by the blocked accept.
    assert dialog.pseudopotential_family() == "norm_conserving"
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
