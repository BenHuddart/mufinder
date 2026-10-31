"""Tests for the Magnetic Structure Builder component, atom-phase and preset dialogs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QGridLayout, QMessageBox

import mufinder.app.magnetic_component_dialogs as magnetic_component_dialogs_module
from mufinder.app import workflow_table_utils
from mufinder.app.magnetic_component_dialogs import (
    AtomPhaseDialog,
    MagneticComponentDialog,
    MagneticPresetDialog,
    component_moment_label,
    selector_label,
)
from mufinder.domain.models import Structure
from mufinder.science.magnetic_structure import (
    AtomSelector,
    ComplexVector,
    FourierTerm,
    MagneticComponent,
    MagneticStructureModel,
    build_collinear_component,
    build_conical_component,
)
from tests.support.magnetic_field_sites import feo_host_structure
from tests.support.magnetic_fields import (
    coo_structure,
    select_combo_data,
    set_direction_inputs_text,
    two_row_helix_model,
)


def test_component_dialog_loads_existing_component_for_editing(qapp) -> None:
    component = MagneticComponent(
        component_id="helix",
        label="Existing helix",
        kind="helix",
        selector=AtomSelector.species("Co"),
        terms=(
            FourierTerm(
                q_coeff=(1.0, 0.0, 0.0),
                coefficient=ComplexVector(real=(0.0, 0.0, 2.0), imaginary=(2.0, 0.0, 0.0)),
            ),
        ),
        phase_turns=0.25,
    )

    dialog = MagneticComponentDialog(coo_structure(), component=component, q_vector=(0.0, 0.0, 0.5))

    assert dialog.windowTitle() == "Edit Magnetic Component"
    assert dialog._label_edit.text() == "Existing helix"
    assert dialog._kind_combo.currentText() == "Helix"
    assert dialog._species_radio.isChecked()
    assert dialog._species_combo.currentText() == "Co"
    assert dialog._q_z.text() == "0.5"
    assert dialog._phase.text() == "0.25"
    assert dialog.component((1.0, 0.0, 0.0)).component_id == "helix"
    dialog.close()


def _phased_cobalt_component(phases: dict[int, float]) -> MagneticComponent:
    return replace(
        build_collinear_component(
            label="AFM", selector=AtomSelector.all(), q_coeff=(0.0, 0.0, 0.0), moment=(0, 0, 1)
        ),
        atom_phase_turns=phases,
    )


def _select_species_target(dialog: MagneticComponentDialog, species: str) -> None:
    dialog._species_combo.setCurrentIndex(dialog._species_combo.findText(species))
    dialog._species_radio.setChecked(True)


def test_component_dialog_starts_without_atom_phases(qapp) -> None:
    dialog = MagneticComponentDialog(coo_structure())

    assert dialog._atom_phase_summary.text() == "None"
    assert dialog._atom_phase_button.text() == "Per-Atom Phases…"
    assert dialog.component((0.0, 0.0, 0.0)).atom_phase_turns == {}
    dialog.close()


def test_component_dialog_atom_phases_follow_the_target(qapp) -> None:
    dialog = MagneticComponentDialog(
        coo_structure(),
        component=_phased_cobalt_component({1: 0.5, 2: 0.25}),
        q_vector=(0.0, 0.0, 0.0),
    )
    assert dialog._atom_phase_summary.text() == "atom 2 +0.5 turns, atom 3 +0.25 turns"

    _select_species_target(dialog, "Co")
    assert dialog._atom_phase_summary.text() == "atom 3 +0.25 turns"
    assert dialog.component((0.0, 0.0, 0.0)).atom_phase_turns == {2: 0.25}

    # Phases of atoms the target drops come back when the target covers them again.
    dialog._all_atoms_radio.setChecked(True)
    assert dialog.component((0.0, 0.0, 0.0)).atom_phase_turns == {1: 0.5, 2: 0.25}
    dialog.close()


@pytest.mark.parametrize("accepted", [True, False])
def test_component_dialog_edits_atom_phases_in_the_phase_dialog(
    qapp, monkeypatch, accepted
) -> None:
    opened: list[tuple[tuple[int, ...], dict[int, float]]] = []

    class _PhaseDialog:
        def __init__(self, _structure, atom_indices, phases, *, parent=None):
            opened.append((tuple(atom_indices), dict(phases)))

        def exec(self):
            return QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected

        def atom_phase_turns(self):
            return {0: 0.5}

    monkeypatch.setattr(magnetic_component_dialogs_module, "AtomPhaseDialog", _PhaseDialog)
    dialog = MagneticComponentDialog(
        coo_structure(),
        component=_phased_cobalt_component({1: 0.5, 2: 0.25}),
        q_vector=(0.0, 0.0, 0.0),
    )
    _select_species_target(dialog, "Co")

    dialog._atom_phase_button.click()

    assert opened == [((0, 2), {2: 0.25})]
    if accepted:
        # The cobalt phases are replaced; the oxygen phase outside the target is kept.
        assert dialog._atom_phase_summary.text() == "atom 1 +0.5 turns"
        dialog._all_atoms_radio.setChecked(True)
        assert dialog.component((0.0, 0.0, 0.0)).atom_phase_turns == {0: 0.5, 1: 0.5}
    else:
        assert dialog._atom_phase_summary.text() == "atom 3 +0.25 turns"
    dialog.close()


def test_atom_phase_dialog_lists_the_target_atoms_and_reads_phases(qapp) -> None:
    dialog = AtomPhaseDialog(coo_structure(), (0, 2), {2: 0.25})
    table = dialog._table

    assert table.rowCount() == 2
    assert [table.item(row, 0).text() for row in range(2)] == ["1", "3"]
    assert table.item(1, 1).text() == "Co"
    assert table.item(1, 2).text() == "0.25, 0.25, 0.25"
    assert table.item(0, 3).text() == ""
    assert table.item(1, 3).text() == "0.25"
    assert not table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable
    assert (table.currentRow(), table.currentColumn()) == (0, 3)
    assert dialog.atom_phase_turns() == {2: 0.25}

    table.item(0, 3).setText("-0.5")
    table.item(1, 3).setText("0")

    assert dialog.atom_phase_turns() == {0: -0.5}
    dialog.close()


@pytest.mark.parametrize(
    ("text", "message"),
    [("half", "must be a decimal number"), ("11", "between -10 and 10"), ("nan", "between")],
)
def test_atom_phase_dialog_refuses_an_invalid_phase(qapp, monkeypatch, text, message) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, body: warnings.append(body),
    )
    dialog = AtomPhaseDialog(coo_structure(), (0, 2), {})
    dialog._table.item(1, 3).setText(text)

    dialog.accept()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "atom 3" in warnings[-1]
    assert message in warnings[-1]
    dialog.close()


@pytest.mark.parametrize("confirmed", [False, True])
def test_atom_phase_dialog_clear_asks_first(qapp, monkeypatch, confirmed) -> None:
    asked: list[str] = []

    def _confirm(_parent, title, *_args, **_kwargs):
        asked.append(title)
        return confirmed

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", _confirm)
    dialog = AtomPhaseDialog(coo_structure(), (0, 2), {0: 0.5, 2: 0.25})

    dialog._clear_button.click()

    assert asked == ["Clear Per-Atom Phases"]
    assert dialog.atom_phase_turns() == ({} if confirmed else {0: 0.5, 2: 0.25})
    dialog.close()


def test_component_dialog_adapts_atom_target_options(qapp) -> None:
    dialog = MagneticComponentDialog(coo_structure(), selected_indices=(0, 2))

    assert dialog._all_atoms_radio.isEnabled()
    assert dialog._species_radio.isEnabled()
    assert dialog._selected_radio.isEnabled()
    assert dialog._component_info_button.objectName() == "mufinderHelpButton"
    assert not dialog._component_info_button.icon().isNull()

    assert dialog.selector() == AtomSelector.indices([0, 2])
    dialog.close()


def test_component_dialog_target_cards_use_compact_grid(qapp) -> None:
    dialog = MagneticComponentDialog(coo_structure(), selected_indices=(0, 2))

    target_layout = dialog._all_atoms_radio.parentWidget().parentWidget().layout()

    assert isinstance(target_layout, QGridLayout)
    assert target_layout.itemAtPosition(0, 0).widget() is dialog._all_atoms_radio.parentWidget()
    assert target_layout.itemAtPosition(0, 1).widget() is dialog._species_radio.parentWidget()
    assert target_layout.itemAtPosition(1, 0).widget() is dialog._equivalent_radio.parentWidget()
    assert target_layout.itemAtPosition(1, 1).widget() is dialog._selected_radio.parentWidget()
    dialog.close()


def test_component_dialog_disables_selection_target_without_selection(qapp) -> None:
    dialog = MagneticComponentDialog(coo_structure(), selected_indices=())

    assert not dialog._selected_radio.isEnabled()
    assert dialog.selector().kind in {"all", "species"}
    dialog.close()


def test_equivalent_site_labels_include_species_position_and_group_size() -> None:
    structure = Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe", "Fe"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    label = selector_label(AtomSelector.symmetry_equivalent(0), structure)

    assert label == "Fe site: atom 1 at (0, 0, 0); 2 equivalent atoms (1, 2)"


def test_component_dialog_presets_create_expected_components(qapp) -> None:
    dialog = MagneticComponentDialog(coo_structure(), selected_indices=(0,))
    dialog._selected_radio.setChecked(True)
    dialog._q_z.setText("0.5")
    dialog._amplitude.setText("2.5")
    set_direction_inputs_text(dialog._primary_direction, (0.0, 0.0, 2.0))

    dialog._kind_combo.setCurrentText("Collinear")
    collinear = dialog.component((1.0, 0.0, 0.0))
    assert collinear.kind == "collinear"
    assert collinear.selector == AtomSelector.indices([0])
    np.testing.assert_allclose(collinear.terms[0].coefficient.to_array(), [0.0, 0.0, 2.5])

    dialog._kind_combo.setCurrentText("Spin-density wave")
    sdw = dialog.component((1.0, 0.0, 0.0))
    assert sdw.kind == "spin_density_wave"
    np.testing.assert_allclose(sdw.terms[0].coefficient.to_array(), [0.0, 0.0, 2.5])

    dialog._kind_combo.setCurrentText("Helix")
    set_direction_inputs_text(dialog._primary_direction, (1.0, 0.0, 0.0))
    helix = dialog.component((1.0, 0.0, 0.0))
    assert helix.kind == "helix"
    np.testing.assert_allclose(helix.terms[0].coefficient.to_array(), [2.5, 2.5j, 0.0])

    dialog._kind_combo.setCurrentText("Cycloid")
    cycloid = dialog.component((1.0, 0.0, 0.0))
    assert cycloid.kind == "cycloid"
    np.testing.assert_allclose(cycloid.terms[0].coefficient.to_array(), [2.5, 0.0, 2.5j])

    dialog._kind_combo.setCurrentText("Conical")
    dialog._cone_angle.setText("60")
    conical = dialog.component((1.0, 0.0, 0.0))
    assert conical.kind == "conical"
    assert len(conical.terms) == 2
    np.testing.assert_allclose(
        conical.terms[0].coefficient.to_array(), [0.0, 0.0, 1.25], atol=1e-12
    )
    np.testing.assert_allclose(
        conical.terms[1].coefficient.to_array(),
        [2.5 * np.sqrt(3.0) / 2.0, 2.5j * np.sqrt(3.0) / 2.0, 0.0],
        atol=1e-12,
    )
    dialog.close()


def test_component_dialog_loads_signed_conical_static_component(qapp) -> None:
    component = build_conical_component(
        label="Signed cone",
        selector=AtomSelector.species("Co"),
        q_coeff=(1.0, 0.0, 0.0),
        total_moment=2.0,
        real_axis_direction=(1.0, 0.0, 0.0),
        imaginary_axis_direction=(0.0, 1.0, 0.0),
        static_axis_direction=(0.0, 0.0, 1.0),
        cone_angle_degrees=120.0,
        component_id="signed-cone",
    )

    dialog = MagneticComponentDialog(
        coo_structure(), component=component, q_vector=(0.0, 0.0, 0.25)
    )

    assert float(dialog._cone_angle.text()) == pytest.approx(120.0)
    round_trip = dialog.component((1.0, 0.0, 0.0))
    assert round_trip.terms[0].coefficient.real[2] < 0.0
    dialog.close()


def test_component_moment_label_summarises_conical_components(qapp) -> None:
    dialog = MagneticComponentDialog(coo_structure(), selected_indices=(0,))
    dialog._selected_radio.setChecked(True)
    dialog._kind_combo.setCurrentText("Conical")
    dialog._q_z.setText("0.5")
    dialog._amplitude.setText("2.5")
    dialog._cone_angle.setText("60")
    set_direction_inputs_text(dialog._primary_direction, (1.0, 0.0, 0.0))

    label = component_moment_label(dialog.component((1.0, 0.0, 0.0)))

    assert label == "Conical α≈60°\nstatic 1.25 μB\nrotating 2.17 μB"
    dialog.close()


def test_component_dialog_decimal_inputs_validate_ranges(qapp) -> None:
    dialog = MagneticComponentDialog(coo_structure(), selected_indices=(0,))

    dialog._amplitude.setText("2.25")
    dialog._q_x.setText("-0.125")
    dialog._q_y.setText("0.0")
    dialog._q_z.setText("0.5")
    dialog._phase.setText("0.25")

    assert dialog.q_vector() == (-0.125, 0.0, 0.5)
    component = dialog.component((1.0, 0.0, 0.0))
    assert component.phase_turns == 0.25

    dialog._amplitude.setText("1001")
    try:
        dialog.component((1.0, 0.0, 0.0))
    except ValueError as exc:
        assert "amplitude" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected out-of-range amplitude to fail")
    dialog.close()


def test_component_dialog_constrains_rotating_directions_from_q_and_handedness(qapp) -> None:
    dialog = MagneticComponentDialog(coo_structure(), selected_indices=(0,))
    dialog._kind_combo.setCurrentText("Helix")
    dialog._q_z.setText("0.5")
    set_direction_inputs_text(dialog._primary_direction, (1.0, 1.0, 1.0))

    helix = dialog.component((1.0, 0.0, 0.0))
    real = np.asarray(helix.terms[0].coefficient.real, dtype=float)
    imag = np.asarray(helix.terms[0].coefficient.imaginary, dtype=float)

    assert real[2] == pytest.approx(0.0)
    assert imag[2] == pytest.approx(0.0)
    assert float(real @ imag) == pytest.approx(0.0)

    dialog._handedness_combo.setCurrentIndex(dialog._handedness_combo.findData(-1))
    left_handed = dialog.component((1.0, 0.0, 0.0))
    np.testing.assert_allclose(
        left_handed.terms[0].coefficient.real, helix.terms[0].coefficient.real
    )
    np.testing.assert_allclose(left_handed.terms[0].coefficient.imaginary, -imag)

    dialog._kind_combo.setCurrentText("Conical")
    conical = dialog.component((1.0, 0.0, 0.0))
    np.testing.assert_allclose(conical.terms[0].coefficient.real[:2], [0.0, 0.0], atol=1e-12)
    dialog.close()


def test_component_dialog_rejects_zero_q_for_rotating_components(qapp) -> None:
    dialog = MagneticComponentDialog(coo_structure(), selected_indices=(0,))
    dialog._kind_combo.setCurrentText("Helix")

    try:
        dialog.component((1.0, 0.0, 0.0))
    except ValueError as exc:
        assert "q must be non-zero" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected zero-q helix to fail")
    dialog.close()


def test_component_dialog_help_documents_current_preset_parameters(qapp) -> None:
    dialog = MagneticComponentDialog(coo_structure(), selected_indices=(0,))
    dialog._selected_radio.setChecked(True)
    dialog._amplitude.setText("2.5")
    dialog._q_z.setText("0.5")
    dialog._phase.setText("0.25")
    set_direction_inputs_text(dialog._primary_direction, (0.0, 0.0, 1.0))
    set_direction_inputs_text(dialog._secondary_direction, (1.0, 0.0, 0.0))

    collinear = dialog._component_help_html()
    assert "Collinear" in collinear
    assert "<b>q</b> = (0.0, 0.0, 0.5)" in collinear
    assert "φ = <b>0.25</b> turns" in collinear
    assert "2.5 μ<sub>B</sub> <b>u</b><sub>1</sub>" in collinear
    assert "<b>u</b><sub>1</sub> = (0, 0, 1)" in collinear
    assert "Selected atoms: 1" in collinear
    assert "Cartesian x, y, z components" in collinear
    assert "Cartesian moment vector" in collinear

    dialog._kind_combo.setCurrentText("Helix")
    helix = dialog._component_help_html()
    assert "Helix" in helix
    assert "<b>u</b><sub>1</sub> + i <b>u</b><sub>2</sub>" in helix
    assert "perpendicular to" in helix
    assert "selected handedness" in helix

    dialog._kind_combo.setCurrentText("Conical")
    conical = dialog._component_help_html()
    assert "Conical" in conical
    assert "q=0" in conical
    assert "cone axis" in conical
    dialog.close()


def test_component_dialog_lists_the_q_basis_and_picks_a_stored_row(qapp) -> None:
    dialog = MagneticComponentDialog(feo_host_structure(), model=two_row_helix_model())
    combo = dialog._q_source_combo
    table = dialog._q_basis_table

    assert [combo.itemText(index) for index in range(combo.count())] == [
        "Enter a vector (r.l.u.)",
        "q1 = (0, 0, 0.25)",
        "q2 = (0.5, 0, 0)",
        "Combination of basis rows",
    ]
    assert [table.item(row, 2).text() for row in range(3)] == ["Fe helix", "Unused", "Unused"]
    assert not table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    assert dialog.q_basis_coefficients() is None
    assert not dialog._q_x.isReadOnly()
    assert dialog._q_coeff_group.isHidden()

    select_combo_data(combo, 1)

    assert dialog.q_basis_coefficients() == (0.0, 1.0, 0.0)
    assert dialog.q_vector() == (0.5, 0.0, 0.0)
    assert dialog._q_x.isReadOnly()
    assert "q = q2 = (0.5, 0, 0)" in dialog._q_expression.text()

    select_combo_data(combo, MagneticComponentDialog._Q_SOURCE_VECTOR)
    assert not dialog._q_x.isReadOnly()
    dialog.close()


def test_component_dialog_combines_basis_rows(qapp) -> None:
    dialog = MagneticComponentDialog(feo_host_structure(), model=two_row_helix_model())
    select_combo_data(dialog._q_source_combo, MagneticComponentDialog._Q_SOURCE_COMBINATION)

    assert not dialog._q_coeff_group.isHidden()
    assert not dialog._q_coeff_inputs[2].isEnabled()
    dialog._q_coeff_inputs[0].setText("1")
    dialog._q_coeff_inputs[1].setText("-1")

    assert dialog.q_basis_coefficients() == (1.0, -1.0, 0.0)
    assert dialog.q_vector() == (-0.5, 0.0, 0.25)
    assert "q = q1 − q2 = (-0.5, 0, 0.25)" in dialog._q_expression.text()
    dialog.close()


def test_component_dialog_rejects_an_empty_basis_combination(qapp, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    dialog = MagneticComponentDialog(feo_host_structure(), model=two_row_helix_model())
    select_combo_data(dialog._q_source_combo, MagneticComponentDialog._Q_SOURCE_COMBINATION)

    dialog.accept()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "at least one q-basis row" in warnings[-1]
    assert "at least one q-basis row" in dialog._q_expression.text()
    dialog.close()


def test_component_dialog_without_stored_rows_only_offers_vector_entry(qapp) -> None:
    dialog = MagneticComponentDialog(feo_host_structure(), model=MagneticStructureModel())

    assert dialog._q_source_combo.count() == 1
    assert dialog.q_basis_coefficients() is None
    assert [dialog._q_basis_table.item(row, 2).text() for row in range(3)] == ["Unused"] * 3
    dialog.close()


@pytest.mark.parametrize(
    ("q_coeff", "expected_source", "expected_coefficients"),
    [
        ((0.0, 1.0, 0.0), 1, (0.0, 1.0, 0.0)),
        ((1.0, 1.0, 0.0), MagneticComponentDialog._Q_SOURCE_COMBINATION, (1.0, 1.0, 0.0)),
        (None, MagneticComponentDialog._Q_SOURCE_VECTOR, None),
    ],
)
def test_component_dialog_reopens_a_component_on_its_basis_rows(
    qapp, q_coeff, expected_source, expected_coefficients
) -> None:
    model = two_row_helix_model()
    component = model.components[0]

    dialog = MagneticComponentDialog(
        feo_host_structure(),
        component=component,
        q_vector=(0.0, 0.0, 0.25),
        model=model,
        q_coeff=q_coeff,
    )

    assert dialog._q_source_combo.currentData() == expected_source
    assert dialog.q_basis_coefficients() == expected_coefficients
    dialog.close()


def test_magnetic_preset_dialog_defaults_to_a_helix_along_c_star(qapp) -> None:
    dialog = MagneticPresetDialog(feo_host_structure(), model=MagneticStructureModel())

    assert dialog._kind_combo.currentData() == "helix"
    assert dialog._propagation_combo.currentData() == "c*"
    assert dialog._reference_combo.currentData() == "a"
    assert dialog._label_edit.text() == "Helix along c*"
    assert dialog.q_vector() == (0.0, 0.0, 0.25)
    assert dialog.q_basis_coefficients() is None
    assert dialog._cone_angle.isHidden()
    assert "perpendicular to q = (0, 0, 0.25)" in dialog._summary.text()

    component = dialog.component((1.0, 0.0, 0.0))
    assert component.kind == "helix"
    assert component.label == "Helix along c*"
    assert component.selector == AtomSelector.species("Fe")
    assert np.allclose(component.terms[0].coefficient.real, (1.0, 0.0, 0.0))
    assert np.allclose(component.terms[0].coefficient.imaginary, (0.0, 1.0, 0.0))
    dialog.close()


def test_magnetic_preset_dialog_builds_a_conical_component_on_a_stored_row(qapp) -> None:
    dialog = MagneticPresetDialog(feo_host_structure(), model=two_row_helix_model())
    select_combo_data(dialog._kind_combo, "conical")
    select_combo_data(dialog._propagation_combo, 1)
    dialog._cone_angle.setText("30")
    dialog._moment.setText("2")

    assert not dialog._cone_angle.isHidden()
    assert dialog._wavevector_group.isHidden()
    assert dialog.q_basis_coefficients() == (0.0, 1.0, 0.0)
    assert dialog.q_vector() == (0.5, 0.0, 0.0)
    assert dialog._reference_combo.currentData() == "b"
    assert dialog._label_edit.text() == "Conical along q2"

    component = dialog.component((0.0, 1.0, 0.0))
    assert component.kind == "conical"
    assert np.allclose(
        component.terms[0].coefficient.real, (2.0 * np.cos(np.deg2rad(30.0)), 0.0, 0.0)
    )
    dialog.close()


def test_magnetic_preset_dialog_keeps_an_edited_label(qapp) -> None:
    dialog = MagneticPresetDialog(feo_host_structure())
    dialog._label_edit.setText("My helix")
    dialog._label_edit.textEdited.emit("My helix")
    select_combo_data(dialog._kind_combo, "cycloid")

    assert dialog._label_edit.text() == "My helix"
    assert dialog._reference_row_label.text() == "Rotation plane holds q and"
    dialog.close()


def test_magnetic_preset_dialog_rejects_a_reference_axis_parallel_to_q(qapp, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    dialog = MagneticPresetDialog(feo_host_structure())
    select_combo_data(dialog._reference_combo, "c")

    assert "Cannot build this preset" in dialog._summary.text()
    dialog.accept()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "parallel to q" in warnings[-1]
    dialog.close()


def test_magnetic_preset_dialog_ignores_a_hidden_cone_angle(qapp) -> None:
    dialog = MagneticPresetDialog(feo_host_structure())
    select_combo_data(dialog._kind_combo, "conical")
    dialog._cone_angle.setText("")
    assert "Cannot build this preset" in dialog._summary.text()

    select_combo_data(dialog._kind_combo, "helix")

    assert dialog._cone_angle.isHidden()
    assert "Cannot build this preset" not in dialog._summary.text()
    assert dialog.component((1.0, 0.0, 0.0)).kind == "helix"
    dialog.close()


def test_magnetic_preset_dialog_takes_the_wavevector_in_inverse_angstrom(qapp) -> None:
    dialog = MagneticPresetDialog(feo_host_structure())
    assert dialog._wavevector_unit.currentData() == "rlu"
    assert dialog._wavevector_row_label.text() == "Wavevector δ (r.l.u.)"

    select_combo_data(dialog._wavevector_unit, "inverse_angstrom")

    # 0.25 r.l.u. along c* of the 3 Å cube is 0.25 x 2π / 3 Å⁻¹.
    assert dialog._wavevector_row_label.text() == "Wavevector k (Å⁻¹)"
    assert float(dialog._wavevector.text()) == pytest.approx(0.25 * 2.0 * np.pi / 3.0, rel=1e-5)
    assert dialog.q_vector() == pytest.approx((0.0, 0.0, 0.25), abs=1e-6)

    dialog._wavevector.setText("0.5")
    select_combo_data(dialog._propagation_combo, "a*")
    assert dialog.q_vector() == pytest.approx((0.5 * 3.0 / (2.0 * np.pi), 0.0, 0.0))
    assert "q = (0.238732, 0, 0)" in dialog._summary.text()

    select_combo_data(dialog._wavevector_unit, "rlu")
    assert float(dialog._wavevector.text()) == pytest.approx(0.238732, rel=1e-5)
    dialog.close()


def test_magnetic_preset_dialog_unit_switch_keeps_q_exactly(qapp) -> None:
    dialog = MagneticPresetDialog(feo_host_structure())

    select_combo_data(dialog._wavevector_unit, "inverse_angstrom")
    assert dialog.q_vector() == pytest.approx((0.0, 0.0, 0.25), abs=1e-14)

    select_combo_data(dialog._wavevector_unit, "rlu")
    assert dialog._wavevector.text() == "0.25"
    assert dialog.q_vector() == pytest.approx((0.0, 0.0, 0.25), abs=1e-14)
    dialog.close()


@pytest.mark.parametrize("delta", [-0.5, 4.0, -4.0, 0.137])
def test_magnetic_preset_dialog_unit_switch_keeps_every_valid_wavevector(qapp, delta) -> None:
    dialog = MagneticPresetDialog(feo_host_structure())
    dialog._wavevector.setText(f"{delta:g}")

    select_combo_data(dialog._wavevector_unit, "inverse_angstrom")
    assert "Cannot build" not in dialog._summary.text()
    assert dialog.q_vector() == pytest.approx((0.0, 0.0, delta), abs=1e-14)

    select_combo_data(dialog._wavevector_unit, "rlu")
    assert "Cannot build" not in dialog._summary.text()
    assert dialog.q_vector() == pytest.approx((0.0, 0.0, delta), abs=1e-14)
    dialog.close()


def test_magnetic_preset_dialog_k_range_follows_the_axis(qapp) -> None:
    # A 3 x 3 x 6 Å cell: 4 r.l.u. is 4 x 2π / 3 = 8.37758 Å⁻¹ along a* and half that along c*.
    structure = feo_host_structure(lattice=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 6.0]])
    dialog = MagneticPresetDialog(structure)
    select_combo_data(dialog._wavevector_unit, "inverse_angstrom")

    dialog._wavevector.setText("-5")
    assert "between -4.18879 and 4.18879" in dialog._summary.text()

    select_combo_data(dialog._propagation_combo, "a*")
    assert "Cannot build" not in dialog._summary.text()
    assert dialog.q_vector() == pytest.approx((-5.0 * 3.0 / (2.0 * np.pi), 0.0, 0.0))
    dialog._wavevector.setText("9")
    assert "between -8.37758 and 8.37758" in dialog._summary.text()
    dialog.close()


def test_magnetic_preset_dialog_refuses_a_non_finite_wavevector(qapp) -> None:
    dialog = MagneticPresetDialog(feo_host_structure())

    dialog._wavevector.setText("nan")

    assert "between -4 and 4" in dialog._summary.text()
    dialog.close()


def test_magnetic_preset_dialog_refuses_a_zero_inverse_angstrom_wavevector(qapp) -> None:
    dialog = MagneticPresetDialog(feo_host_structure())
    select_combo_data(dialog._wavevector_unit, "inverse_angstrom")

    dialog._wavevector.setText("0")

    assert "non-zero number of inverse Angstrom" in dialog._summary.text()
    dialog.close()


def test_magnetic_preset_dialog_keeps_the_reference_axis_while_the_wavevector_changes(qapp) -> None:
    dialog = MagneticPresetDialog(feo_host_structure())
    select_combo_data(dialog._reference_combo, "b")

    dialog._wavevector.setText("0.3")
    assert dialog._reference_combo.currentData() == "b"

    select_combo_data(dialog._propagation_combo, "a*")
    assert dialog._reference_combo.currentData() == "b"
    select_combo_data(dialog._reference_combo, "c")
    select_combo_data(dialog._propagation_combo, "b*")
    assert dialog._reference_combo.currentData() == "a"
    dialog.close()


def test_magnetic_preset_dialog_takes_a_typed_vector_on_an_empty_basis(qapp) -> None:
    dialog = MagneticPresetDialog(feo_host_structure(), model=MagneticStructureModel())
    assert dialog._propagation_combo.findText("Enter a vector (r.l.u.)") == 3
    assert dialog._typed_q_group.isHidden()

    select_combo_data(dialog._propagation_combo, "vector")
    assert not dialog._typed_q_group.isHidden()
    assert dialog._wavevector_group.isHidden()
    # Nothing typed yet: the dialog says why it cannot build, and refuses.
    assert dialog._summary.text().startswith("Cannot build this preset")

    set_direction_inputs_text(dialog._typed_q, (0.0147, 0.0147, 0.0147))

    assert dialog.q_basis_coefficients() is None
    assert dialog.q_vector() == (0.0147, 0.0147, 0.0147)
    assert dialog._label_edit.text() == "Helix along (0.0147, 0.0147, 0.0147)"
    assert "q = (0.0147, 0.0147, 0.0147)" in dialog._summary.text()
    component = dialog.component((1.0, 0.0, 0.0))
    assert component.kind == "helix"
    assert component.terms[0].q_coeff == (1.0, 0.0, 0.0)
    dialog.close()


def test_magnetic_preset_dialog_refuses_a_zero_or_invalid_typed_vector(qapp, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        magnetic_component_dialogs_module.QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    dialog = MagneticPresetDialog(feo_host_structure())
    select_combo_data(dialog._propagation_combo, "vector")

    dialog.accept()
    assert len(warnings) == 1

    dialog._typed_q[1].setText("")
    dialog.accept()
    assert len(warnings) == 2
    assert dialog._label_edit.text() == "Helix along a typed q"
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.close()


def test_magnetic_preset_dialog_typed_vector_resets_the_reference_only_when_it_turns(qapp) -> None:
    dialog = MagneticPresetDialog(feo_host_structure())
    select_combo_data(dialog._propagation_combo, "vector")

    def enter(values: tuple[float, float, float]) -> None:
        set_direction_inputs_text(dialog._typed_q, values)
        dialog._typed_q[0].editingFinished.emit()

    enter((0.1, 0.0, 0.0))
    assert dialog._reference_combo.currentData() == "b"

    select_combo_data(dialog._reference_combo, "c")
    enter((0.2, 0.0, 0.0))
    assert dialog._reference_combo.currentData() == "c"

    # A rescale passes through other directions while it is typed, which do not count.
    enter((0.1, 0.1, 0.1))
    select_combo_data(dialog._reference_combo, "c")
    dialog._typed_q[0].setText("0.2")
    assert dialog._reference_combo.currentData() == "c"
    enter((0.2, 0.2, 0.2))
    assert dialog._reference_combo.currentData() == "c"

    enter((0.0, 0.0, 0.2))
    assert dialog._reference_combo.currentData() == "a"
    dialog.close()


def test_magnetic_preset_dialog_typed_vector_ignores_the_wavevector_unit(qapp) -> None:
    dialog = MagneticPresetDialog(feo_host_structure())
    select_combo_data(dialog._wavevector_unit, "inverse_angstrom")
    select_combo_data(dialog._propagation_combo, "vector")
    set_direction_inputs_text(dialog._typed_q, (0.25, 0.0, 0.0))

    assert dialog.q_vector() == (0.25, 0.0, 0.0)
    assert dialog._wavevector_row_label.isHidden()
    dialog.close()


# --- Contact-shell tie warnings ---------------------------------------------------
#
# B2 FeMn with Fe up and Mn down: at the tetrahedral site (1/2, 1/4, 0) two Fe
# and two Mn tie at 1.603 Å, so two Contact neighbours split them.
