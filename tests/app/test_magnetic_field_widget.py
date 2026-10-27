"""Tests for magnetic-field workflow UI helpers."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from uuid import uuid4

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QItemSelectionModel, QPointF, Qt
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtWidgets import QAbstractItemView, QDialog, QGridLayout, QScrollArea
from PySide6.QtTest import QTest

import mufinder.app.magnetic_field_widget as magnetic_field_widget_module
import mufinder.science.magnetic_fields as magnetic_fields_module
from mufinder.services.muon_sites_summary import MAGNETIC_FIELDS_SCOPE, set_show_every_relaxed_site
from mufinder.app import workflow_table_utils
from mufinder.app import theme as theme_module
from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.magnetic_field_widget import (
    DipolarTensorDialog,
    DipolarTensorResultsDialog,
    AtomPhaseDialog,
    MagneticComponentDialog,
    MagneticFieldCalculationSettingsDialog,
    MagneticFieldDistributionDialog,
    MagneticFieldWidget,
    MagneticPresetDialog,
    MagneticStructureBuilderDialog,
    MagneticTextureDialog,
    MagneticTexturePreview,
    MuonSiteSelectionDialog,
    NuclearRelaxationDialog,
    _evaluate_texture_preview_grid,
    _component_moment_label,
    _selector_label,
)
from mufinder.domain.models import CalculationRun, Job, Result, Structure
from mufinder.science.magnetic_structure import (
    AtomSelector,
    ComplexVector,
    FourierTerm,
    MagneticComponent,
    MagneticStructureModel,
    MagneticTexture,
    MagneticTextureTerm,
    build_collinear_component,
    build_conical_component,
    build_helix_component,
    texture_diagnostics,
)
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    attach_magnetic_field_results,
    store_magnetic_field_results,
)
from mufinder.science.demagnetization import PlatePhoto, demagnetizing_tensor
from mufinder.services.project_service import ProjectService
from tests.support.codata import BOHR_MAGNETON_J_PER_T, MU0_OVER_4PI_T_M_PER_A
from tests.support.structures import host_structure, relaxed_muon_supercell


_host_structure = partial(
    host_structure,
    label="FeO",
    lattice=3.0,
    species=("Fe", "O"),
    properties={"atomic_spins": [1.0, -0.5], "propagation_vector": [0.0, 0.0, 0.0]},
)


def _multi_species_structure() -> Structure:
    return Structure(
        label="CoO",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Co", "O", "Co"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )


def _component(component_id: str, label: str, atom_index: int) -> MagneticComponent:
    return MagneticComponent(
        component_id=component_id,
        label=label,
        kind="test",
        selector=AtomSelector.indices([atom_index]),
        terms=(
            FourierTerm(
                q_coeff=(0.0, 0.0, 0.0),
                coefficient=ComplexVector.from_vector([0.0, 0.0, 1.0]),
            ),
        ),
    )


def _set_direction(inputs, values: tuple[float, float, float]) -> None:
    for edit, value in zip(inputs, values, strict=True):
        edit.setText(f"{value:g}")


def _set_site_source(widget, source: str) -> None:
    index = widget._site_source_combo.findData(source)
    assert index >= 0
    widget._site_source_combo.setCurrentIndex(index)


def _custom_muon_site(
    position: tuple[float, float, float],
) -> magnetic_field_widget_module.MagneticMuonSiteOption:
    return magnetic_field_widget_module.MagneticMuonSiteOption(
        source="custom",
        label="Custom site",
        fractional_position=position,
        cartesian_position=tuple(3.0 * value for value in position),
    )


def test_atomic_structure_viewer_accepts_vector_moments(qapp) -> None:
    viewer = AtomicStructureViewer()
    viewer.set_structure(_host_structure())

    viewer.set_moment_vectors(np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))

    assert viewer._spins is not None
    assert viewer._spins.shape == (2, 3)
    viewer.close()


def test_magnetic_structure_builder_imports_legacy_host_spins(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic.mufinder", "Magnetic")
    service.add_structure(document, _host_structure())

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)

    dialog._build_from_host_spins()

    assert dialog._component_table.rowCount() == 2
    assert dialog._component_table.columnCount() == 6
    assert dialog._preview_status.text() == "Preview: full magnetic structure (1x1x1)"
    assert dialog._classification.text() == "Structure type: commensurate"
    assert dialog._viewer._spins is not None
    assert dialog._viewer._spins.shape == (2, 3)
    assert dialog._viewer._show_cell.objectName() == "magneticBuilderViewerCheckBox"
    assert "spacing: 9px" in dialog._viewer._show_cell.styleSheet()
    assert len(dialog.model().components) == 2
    dialog.close()


def test_magnetic_structure_builder_delete_and_backspace_remove_components(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-delete.mufinder", "Magnetic Delete")
    service.add_structure(document, _host_structure())

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    model = MagneticStructureModel(
        components=(
            _component("one", "One", 0),
            _component("two", "Two", 1),
        )
    )
    dialog._set_model(model)

    dialog._component_table.selectRow(0)
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
    dialog._component_table.eventFilter(dialog._component_table.viewport(), event)
    assert dialog._component_table.rowCount() == 1

    dialog._component_table.selectRow(0)
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
    dialog._component_table.eventFilter(dialog._component_table.viewport(), event)
    assert dialog._component_table.rowCount() == 0
    dialog.close()


def test_magnetic_structure_builder_label_column_is_editable(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-label.mufinder", "Magnetic Label")
    service.add_structure(document, _host_structure())

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    model = MagneticStructureModel(components=(_component("one", "One", 0),))
    dialog._set_model(model)

    label_item = dialog._component_table.item(0, 1)
    type_item = dialog._component_table.item(0, 2)
    assert label_item.flags() & Qt.ItemFlag.ItemIsEditable
    assert not type_item.flags() & Qt.ItemFlag.ItemIsEditable

    label_item.setText("Renamed")

    assert dialog.model().components[0].label == "Renamed"
    assert dialog._component_table.item(0, 1).text() == "Renamed"
    dialog.close()


def _builder_with_components(
    tmp_path, *components: MagneticComponent
) -> MagneticStructureBuilderDialog:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-builder.mufinder", "Magnetic Builder")
    service.add_structure(document, _host_structure())
    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    dialog._set_model(MagneticStructureModel(components=components))
    return dialog


def _builder_labels(dialog: MagneticStructureBuilderDialog) -> list[str]:
    return [
        dialog._component_table.item(row, 1).text()
        for row in range(dialog._component_table.rowCount())
    ]


def _select_builder_rows(dialog: MagneticStructureBuilderDialog, *rows: int) -> None:
    table = dialog._component_table
    table.clearSelection()
    for row in rows:
        table.selectionModel().select(
            table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )


def test_magnetic_structure_builder_enabled_column_toggles_components(qapp, tmp_path) -> None:
    dialog = _builder_with_components(
        tmp_path, _component("one", "One", 0), _component("two", "Two", 1)
    )
    table = dialog._component_table
    enabled_item = table.item(1, 0)

    assert table.horizontalHeaderItem(0).text() == "Enabled"
    assert enabled_item.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert not enabled_item.flags() & Qt.ItemFlag.ItemIsEditable
    assert enabled_item.checkState() == Qt.CheckState.Checked
    assert dialog._viewer._spins is not None
    assert np.count_nonzero(np.linalg.norm(dialog._viewer._spins, axis=1)) == 2

    enabled_item.setCheckState(Qt.CheckState.Unchecked)

    assert [component.enabled for component in dialog.model().components] == [True, False]
    assert table.rowCount() == 2
    assert "Disabled" in enabled_item.toolTip()
    assert table.item(1, 1).foreground().color() == QColor(
        theme_module.current_palette().text_muted
    )
    assert dialog._preview_status.text() == "Preview: full magnetic structure (1x1x1)"
    assert np.count_nonzero(np.linalg.norm(dialog._viewer._spins, axis=1)) == 1

    table.selectRow(1)
    assert dialog._preview_status.text() == "Preview: the selected components are disabled"
    assert dialog._viewer._spins is None
    assert dialog._classification.text() == "Structure type: commensurate"

    table.clearSelection()
    table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert dialog._preview_status.text() == "Preview: every component is disabled"
    assert dialog._classification.text() == "Structure type: -"
    assert dialog._viewer._spins is None

    table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    assert [component.enabled for component in dialog.model().components] == [False, True]
    assert not table.item(1, 1).data(Qt.ItemDataRole.ForegroundRole)
    assert dialog._preview_status.text() == "Preview: full magnetic structure (1x1x1)"
    dialog.close()


def test_magnetic_structure_builder_loads_disabled_components_unchecked(qapp, tmp_path) -> None:
    disabled = replace(_component("two", "Two", 1), enabled=False)
    dialog = _builder_with_components(tmp_path, _component("one", "One", 0), disabled)

    assert dialog._component_table.item(0, 0).checkState() == Qt.CheckState.Checked
    assert dialog._component_table.item(1, 0).checkState() == Qt.CheckState.Unchecked
    dialog.close()


def test_magnetic_structure_builder_duplicates_selected_components(qapp, tmp_path) -> None:
    dialog = _builder_with_components(
        tmp_path,
        _component("one", "One", 0),
        replace(_component("two", "Two", 1), enabled=False),
    )
    assert not dialog._duplicate_button.isEnabled()

    _select_builder_rows(dialog, 0, 1)
    assert dialog._duplicate_button.isEnabled()
    dialog._duplicate_button.click()

    components = dialog.model().components
    assert _builder_labels(dialog) == ["One", "One (copy)", "Two", "Two (copy)"]
    assert components[1].component_id not in {"one", "two"}
    assert components[1].terms == components[0].terms
    assert [component.enabled for component in components] == [True, True, False, False]
    assert dialog._component_table.item(3, 0).checkState() == Qt.CheckState.Unchecked
    assert dialog._selected_component_ids() == (
        components[1].component_id,
        components[3].component_id,
    )
    assert dialog._preview_status.text() == "Preview: selected components (1x1x1)"
    dialog.close()


def test_magnetic_structure_builder_moves_selected_components(qapp, tmp_path) -> None:
    dialog = _builder_with_components(
        tmp_path,
        _component("one", "One", 0),
        _component("two", "Two", 1),
        _component("three", "Three", 0),
    )
    assert not dialog._move_up_button.isEnabled()
    assert not dialog._move_down_button.isEnabled()

    _select_builder_rows(dialog, 0)
    assert not dialog._move_up_button.isEnabled()
    assert dialog._move_down_button.isEnabled()

    dialog._move_down_button.click()
    assert _builder_labels(dialog) == ["Two", "One", "Three"]
    assert dialog._selected_component_ids() == ("one",)
    assert dialog._move_up_button.isEnabled()

    dialog._move_down_button.click()
    assert _builder_labels(dialog) == ["Two", "Three", "One"]
    assert not dialog._move_down_button.isEnabled()

    _select_builder_rows(dialog, 1, 2)
    dialog._move_up_button.click()
    assert _builder_labels(dialog) == ["Three", "One", "Two"]
    assert dialog._selected_component_ids() == ("three", "one")
    assert not dialog._move_up_button.isEnabled()
    assert dialog._move_down_button.isEnabled()

    _select_builder_rows(dialog, 0, 2)
    assert dialog._move_up_button.isEnabled()
    dialog._move_up_button.click()
    assert _builder_labels(dialog) == ["Three", "Two", "One"]
    dialog.close()


def test_magnetic_structure_builder_remove_asks_for_confirmation(
    qapp, tmp_path, monkeypatch, confirmed_destructive_actions
) -> None:
    dialog = _builder_with_components(
        tmp_path, _component("one", "One", 0), _component("two", "Two", 1)
    )

    _select_builder_rows(dialog, 0, 1)
    dialog._remove_button.click()
    assert confirmed_destructive_actions[-1]["text"] == "Remove 2 components?"
    assert dialog._component_table.rowCount() == 0

    dialog._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    monkeypatch.setattr(
        workflow_table_utils, "confirm_destructive", lambda *_args, **_kwargs: False
    )
    dialog._component_table.selectRow(0)
    for key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
        event = QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
        dialog._component_table.eventFilter(dialog._component_table.viewport(), event)
    dialog._remove_button.click()
    assert _builder_labels(dialog) == ["One"]
    dialog.close()


def test_magnetic_structure_builder_confirms_replacing_components_from_host_spins(
    qapp, tmp_path, monkeypatch, confirmed_destructive_actions
) -> None:
    dialog = _builder_with_components(tmp_path)
    dialog._build_from_host_spins()
    assert not confirmed_destructive_actions
    assert dialog._component_table.rowCount() == 2

    dialog._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    monkeypatch.setattr(
        workflow_table_utils, "confirm_destructive", lambda *_args, **_kwargs: False
    )
    dialog._build_from_host_spins()
    assert _builder_labels(dialog) == ["One"]
    dialog.close()


def test_magnetic_structure_builder_host_spin_prompt_names_component_count(
    qapp, tmp_path, confirmed_destructive_actions
) -> None:
    dialog = _builder_with_components(tmp_path, _component("one", "One", 0))

    dialog._build_from_host_spins()

    assert confirmed_destructive_actions[-1]["title"] == "Replace magnetic structure"
    assert "Replace the 1 component" in confirmed_destructive_actions[-1]["text"]
    assert confirmed_destructive_actions[-1]["accept_label"] == "Replace Components"
    assert dialog._component_table.rowCount() == 2
    dialog.close()


def test_magnetic_field_widget_reports_and_persists_disabled_components(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-disabled.mufinder", "Magnetic Disabled")
    service.add_structure(document, _host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)
    payloads: list[dict] = []
    widget.magnetic_structure_changed.connect(payloads.append)

    widget._set_model(
        MagneticStructureModel(
            components=(
                _component("one", "One", 0),
                replace(_component("two", "Two", 1), enabled=False),
            )
        )
    )
    assert widget._structure_summary.text() == "Magnetic structure: 2 components (1 disabled)"
    assert widget._calculate_button.isEnabled()
    restored = MagneticStructureModel.from_json(payloads[-1])
    assert [component.enabled for component in restored.components] == [True, False]

    summary = widget._component_summary_table
    assert not summary.isHidden()
    assert summary.rowCount() == 1
    assert [
        summary.horizontalHeaderItem(column).text() for column in range(summary.columnCount())
    ] == [
        "Component",
        "Type",
        "Apply to",
        "q",
        "Moment",
    ]
    assert summary.item(0, 0).text() == "One"
    assert summary.item(0, 2).text() == "Selected atoms: 1"

    widget._set_model(
        MagneticStructureModel(components=(replace(_component("one", "One", 0), enabled=False),))
    )
    assert widget._structure_summary.text() == "Magnetic structure: 1 component (1 disabled)"
    assert summary.isHidden()
    assert summary.rowCount() == 0
    assert widget._classification.text() == "Structure type: -"
    assert not widget._calculate_button.isEnabled()
    widget._selected_muon_site = _custom_muon_site((0.25, 0.5, 0.75))
    widget._calculate_fields()
    assert (
        widget._calculation_status.text()
        == "Results: enable at least one magnetic-structure component first."
    )
    widget.close()


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

    dialog = MagneticComponentDialog(
        _multi_species_structure(), component=component, q_vector=(0.0, 0.0, 0.5)
    )

    assert dialog.windowTitle() == "Edit Magnetic Component"
    assert dialog._label_edit.text() == "Existing helix"
    assert dialog._kind_combo.currentText() == "Helix"
    assert dialog._species_radio.isChecked()
    assert dialog._species_combo.currentText() == "Co"
    assert dialog._q_z.text() == "0.5"
    assert dialog._phase.text() == "0.25"
    assert dialog.component((1.0, 0.0, 0.0)).component_id == "helix"
    dialog.close()


def test_magnetic_structure_builder_edits_selected_component(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-edit.mufinder", "Magnetic Edit")
    service.add_structure(document, _host_structure())

    original = MagneticComponent(
        component_id="one",
        label="One",
        kind="collinear",
        selector=AtomSelector.indices([0]),
        terms=(
            FourierTerm(
                q_coeff=(1.0, 0.0, 0.0),
                coefficient=ComplexVector.from_vector([0.0, 0.0, 1.0]),
            ),
        ),
    )
    replacement = MagneticComponent(
        component_id="one",
        label="Edited",
        kind="spin_density_wave",
        selector=AtomSelector.indices([0]),
        terms=(
            FourierTerm(
                q_coeff=(1.0, 0.0, 0.0),
                coefficient=ComplexVector.from_vector([0.0, 1.0, 0.0]),
            ),
        ),
    )

    class _Dialog:
        def __init__(self, *args, **kwargs):
            assert kwargs["component"] == original
            assert kwargs["q_vector"] == (0.0, 0.0, 0.5)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def q_basis_coefficients(self):
            return None

        def q_vector(self):
            return (0.25, 0.0, 0.0)

        def component(self, q_coeff):
            assert q_coeff == (1.0, 0.0, 0.0)
            return replacement

    monkeypatch.setattr(magnetic_field_widget_module, "MagneticComponentDialog", _Dialog)

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    dialog._set_model(
        MagneticStructureModel(
            q_basis_rlu=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), components=(original,)
        )
    )
    dialog._component_table.selectRow(0)

    dialog._edit_selected_component()

    assert dialog.model().components == (replacement,)
    assert dialog.model().q_basis_rlu[0] == (0.25, 0.0, 0.0)
    assert dialog._component_table.item(0, 1).text() == "Edited"
    dialog.close()


def test_magnetic_texture_dialog_builds_hexagonal_texture_component(qapp) -> None:
    dialog = MagneticTextureDialog(_multi_species_structure(), selected_indices=(0, 2))
    dialog._selected_radio.setChecked(True)
    dialog._label_edit.setText("Hex texture")
    dialog._amplitude.setText("3.0")
    _set_direction(dialog._q1, (0.1, 0.0, 0.0))
    _set_direction(dialog._core_direction, (0.0, 0.0, 1.0))

    q_basis = dialog.q_basis_vectors()
    component = dialog.component(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)), q_basis)

    assert component.kind == "texture"
    assert component.label == "Hex texture"
    assert component.selector == AtomSelector.indices([0, 2])
    assert len(component.terms) == 3
    assert component.metadata["type"] == "texture"
    assert component.metadata["texture_mode"] == "hexagonal_skyrmion"
    assert "term(s)" in dialog._diagnostics.text()
    assert dialog._preview._moments.shape[:2] == (161, 161)
    assert isinstance(dialog._controls_scroll, QScrollArea)
    assert dialog._custom_group.isHidden()
    assert dialog._q2_row.isHidden()
    assert dialog._q3_row.isHidden()
    assert not dialog._generated_q_hint.isHidden()
    assert not dialog._skyrmion_group.isHidden()
    assert dialog._square_background_row.isHidden()
    assert dialog._preview._canvas_mode == "light"
    dialog._preview_dark_button.click()
    assert dialog._preview._canvas_mode == "dark"
    assert dialog._preview_dark_button.isChecked()
    dialog._preview_light_button.click()
    assert dialog._preview._canvas_mode == "light"
    assert dialog._magnitude_info_button.objectName() == "mufinderHelpButton"
    assert not dialog._magnitude_info_button.icon().isNull()
    assert dialog._magnitude_info_button.toolTip() == "Explain magnitude normalization and pruning"
    dialog.close()


def test_magnetic_texture_dialog_round_trips_preset_builder_state(qapp) -> None:
    structure = _multi_species_structure()
    dialog = MagneticTextureDialog(structure, selected_indices=(0, 2))
    dialog._mode_buttons["square_skyrmion"].setChecked(True)
    dialog._label_edit.setText("Square texture")
    dialog._selected_radio.setChecked(True)
    dialog._amplitude.setText("2.5")
    _set_direction(dialog._q1, (0.12, 0.0, 0.0))
    _set_direction(dialog._core_direction, (0.0, 0.0, 1.0))
    dialog._skyrmion_type.setCurrentText("Neel")
    dialog._helicity.setCurrentIndex(dialog._helicity.findData(-1))
    dialog._cone_angle.setText("120")
    dialog._background.setText("0.25")
    dialog._normalise_check.setChecked(True)
    dialog._grid_u.setValue(8)
    dialog._grid_v.setValue(10)
    dialog._normalise_threshold.setText("0.002")
    dialog._prune_mode.setCurrentText("Keep strongest N")
    dialog._prune_top_n.setValue(6)
    q_basis = dialog.q_basis_vectors()
    component = dialog.component(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)), q_basis)
    dialog.close()

    edit = MagneticTextureDialog(structure, component=component)

    assert edit.mode() == "square_skyrmion"
    assert edit._label_edit.text() == "Square texture"
    assert edit._selected_radio.isChecked()
    assert edit._amplitude.text() == "2.5"
    assert edit._q1[0].text() == "0.12"
    assert edit._core_direction[2].text() == "1"
    assert edit._skyrmion_type.currentText() == "Neel"
    assert edit._helicity.currentData() == -1
    assert edit._cone_angle.text() == "120"
    assert edit._background.text() == "0.25"
    assert edit._normalise_check.isChecked()
    assert edit._grid_u.value() == 8
    assert edit._grid_v.value() == 10
    assert edit._normalise_threshold.text() == "0.002"
    assert edit._prune_mode.currentText() == "Keep strongest N"
    assert edit._prune_top_n.value() == 6
    edited = edit.component(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        edit.q_basis_vectors(),
    )
    assert edited.to_json() == component.to_json()
    edit.close()


def test_magnetic_texture_dialog_legacy_preset_opens_as_exact_custom_terms(qapp) -> None:
    component = MagneticComponent(
        component_id="legacy-texture",
        label="Legacy texture",
        kind="texture",
        selector=AtomSelector.all(),
        terms=(
            FourierTerm((0.0, 1.0, 0.0), ComplexVector.from_vector([0.0, 1.0, 0.0])),
            FourierTerm(
                (1.0, 0.0, 0.0), ComplexVector(real=(0.0, 0.0, 1.0), imaginary=(0.5, 0.0, 0.0))
            ),
        ),
        metadata={
            "type": "texture",
            "texture_mode": "hexagonal_skyrmion",
            "q_basis_rlu": [[0.0, 0.2, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "normalization": {"enabled": True, "grid_resolution": [8, 8, 1], "threshold": 0.001},
            "pruning": {"mode": "Keep strongest N", "top_n": 1, "threshold": 0.001},
        },
    )

    dialog = MagneticTextureDialog(_multi_species_structure(), component=component)

    assert dialog.mode() == "custom"
    assert not dialog._custom_group.isHidden()
    assert dialog._custom_table.rowCount() == 2
    assert not dialog._normalise_check.isChecked()
    assert dialog._prune_mode.currentText() == "No pruning"
    assert dialog._custom_table.item(0, 0).text() == "0"
    assert dialog._custom_table.item(0, 1).text() == "1"
    assert dialog._custom_table.item(1, 6).text() == "0.5"
    round_trip = dialog.component(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)), dialog.q_basis_vectors()
    )
    assert round_trip.terms == component.terms
    dialog.close()


def test_magnetic_texture_dialog_shows_custom_controls_only_for_custom_mode(qapp) -> None:
    dialog = MagneticTextureDialog(_multi_species_structure())

    assert dialog._custom_group.isHidden()
    assert dialog._q2_row.isHidden()

    dialog._mode_buttons["custom"].setChecked(True)

    assert not dialog._custom_group.isHidden()
    assert not dialog._q2_row.isHidden()
    assert not dialog._q3_row.isHidden()
    assert dialog._generated_q_hint.isHidden()
    assert dialog._skyrmion_group.isHidden()

    dialog._mode_buttons["square_skyrmion"].setChecked(True)

    assert dialog._custom_group.isHidden()
    assert not dialog._skyrmion_group.isHidden()
    assert not dialog._square_background_row.isHidden()
    dialog.close()


def test_magnetic_texture_preview_prefers_out_of_plane_background(qapp) -> None:
    preview = MagneticTexturePreview()
    moments = np.zeros((4, 4, 3), dtype=float)
    moments[..., 0] = 1.0
    moments[..., 2] = np.linspace(-1.0, 1.0, 16).reshape(4, 4)

    preview.set_moments(moments)

    values, mode = preview._background_values()
    assert mode == "out_of_plane"
    assert values.shape == (4, 4)
    assert preview._display_moments().shape == (4, 4, 3)

    moments[..., 2] = 0.0
    preview.set_moments(moments)

    _, mode = preview._background_values()
    assert mode == "magnitude"
    preview.close()


def test_magnetic_texture_preview_supports_zoom_pan_and_box_view(qapp) -> None:
    preview = MagneticTexturePreview()
    preview.resize(320, 300)
    moments = np.zeros((100, 100, 3), dtype=float)
    moments[..., 0] = 1.0

    preview.set_moments(moments, initial_view_fraction=0.5)

    assert preview._view_fraction == pytest.approx(0.5)
    assert 48 <= preview._display_moments().shape[0] <= 52

    preview._zoom_at(preview._plot_rect().center(), 0.5)
    assert preview._view_fraction == pytest.approx(0.25)

    original_center = preview._view_center.copy()
    preview._pan_start_center = preview._view_center.copy()
    preview._pan_by_pixels(QPointF(30.0, 0.0))
    assert preview._view_center[0] < original_center[0]

    plot_rect = preview._plot_rect()
    preview._zoom_to_box(
        QPointF(
            plot_rect.left() + 0.25 * plot_rect.width(), plot_rect.top() + 0.25 * plot_rect.height()
        ),
        QPointF(
            plot_rect.left() + 0.75 * plot_rect.width(), plot_rect.top() + 0.75 * plot_rect.height()
        ),
    )
    assert preview._view_fraction == pytest.approx(0.125, rel=0.05)
    preview.close()


def test_magnetic_texture_dialog_renders_wider_grid_than_initial_view(qapp) -> None:
    dialog = MagneticTextureDialog(_multi_species_structure())

    assert dialog._preview._moments.shape[:2] == (
        dialog._PREVIEW_RESOLUTION,
        dialog._PREVIEW_RESOLUTION,
    )
    assert dialog._preview._view_fraction == pytest.approx(
        dialog._PREVIEW_INITIAL_REPEATS / dialog._PREVIEW_RENDER_REPEATS
    )
    assert dialog._preview._display_moments().shape[0] < dialog._preview._moments.shape[0]
    dialog.close()


def test_magnetic_texture_dialog_magnitude_pruning_help_uses_current_values(qapp) -> None:
    dialog = MagneticTextureDialog(_multi_species_structure())
    dialog._grid_u.setValue(16)
    dialog._grid_v.setValue(24)
    dialog._grid_w.setValue(2)
    dialog._normalise_threshold.setText("0.0005")
    dialog._prune_mode.setCurrentText("Keep strongest N")
    dialog._prune_top_n.setValue(12)

    html = dialog._magnitude_pruning_help_html()

    assert "mag2spectra" in html
    assert "16 × 24 × 2" in html
    assert "0.0005" in html
    assert "Keep strongest N" in html
    assert "12 largest" in html
    dialog.close()


def test_magnetic_texture_preview_uses_hexagonal_q_basis() -> None:
    texture = MagneticTexture((MagneticTextureTerm.from_vector((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),))

    square = _evaluate_texture_preview_grid(
        texture, mode="square_skyrmion", resolution=3, repeats=1.0
    )
    hexagonal = _evaluate_texture_preview_grid(
        texture, mode="hexagonal_skyrmion", resolution=3, repeats=1.0
    )

    assert np.isclose(square[2, 1, 2], 1.0)
    assert abs(hexagonal[2, 1, 2]) < 1e-12


def test_magnetic_texture_preview_q_basis_follows_core_direction_handedness() -> None:
    texture = MagneticTexture((MagneticTextureTerm.from_vector((0.0, 1.0, 0.0), (0.0, 0.0, 1.0j)),))

    core_down = _evaluate_texture_preview_grid(
        texture,
        mode="hexagonal_skyrmion",
        core_direction=(0.0, 0.0, -1.0),
        resolution=3,
        repeats=1.0,
    )
    core_up = _evaluate_texture_preview_grid(
        texture,
        mode="hexagonal_skyrmion",
        core_direction=(0.0, 0.0, 1.0),
        resolution=3,
        repeats=1.0,
    )

    assert core_down[1, 2, 2] < 0.0
    assert core_up[1, 2, 2] > 0.0
    assert np.isclose(core_down[1, 2, 2], -core_up[1, 2, 2])


def test_magnetic_texture_dialog_custom_table_delete_and_backspace(qapp) -> None:
    dialog = MagneticTextureDialog(_multi_species_structure())
    dialog._mode_buttons["custom"].setChecked(True)
    row_count = dialog._custom_table.rowCount()
    dialog._add_custom_term_row()
    assert dialog._custom_table.rowCount() == row_count + 1

    dialog._custom_table.selectRow(0)
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
    dialog._custom_table.eventFilter(dialog._custom_table.viewport(), event)
    assert dialog._custom_table.rowCount() == row_count

    dialog._custom_table.selectRow(0)
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
    dialog._custom_table.eventFilter(dialog._custom_table.viewport(), event)
    assert dialog._custom_table.rowCount() == max(0, row_count - 1)
    dialog.close()


def test_magnetic_texture_dialog_normalization_and_pruning_update_diagnostics(qapp) -> None:
    dialog = MagneticTextureDialog(_multi_species_structure())
    dialog._mode_buttons["square_skyrmion"].setChecked(True)
    dialog._normalise_check.setChecked(True)
    dialog._grid_u.setValue(8)
    dialog._grid_v.setValue(8)
    dialog._prune_mode.setCurrentText("Keep strongest N")
    dialog._prune_top_n.setValue(2)

    texture = dialog.texture()

    assert len(texture.terms) == 2
    assert "2 term(s)" in dialog._diagnostics.text()
    dialog.close()


def test_magnetic_texture_dialog_reports_topological_charge_per_magnetic_cell(qapp) -> None:
    dialog = MagneticTextureDialog(_multi_species_structure())

    assert "Topological charge per magnetic cell: -1." in dialog._diagnostics.text()
    _set_direction(dialog._core_direction, (0.0, 0.0, 1.0))
    assert "Topological charge per magnetic cell: 1." in dialog._diagnostics.text()
    dialog._helicity.setCurrentIndex(dialog._helicity.findData(-1))
    assert "Topological charge per magnetic cell: 1." in dialog._diagnostics.text()

    _set_direction(dialog._core_direction, (0.0, 0.0, -1.0))
    dialog._mode_buttons["square_skyrmion"].setChecked(True)
    assert (
        "Topological charge per magnetic cell: undefined (the moment magnitude vanishes"
        in dialog._diagnostics.text()
    )
    dialog._background.setText("-0.3")
    assert "Topological charge per magnetic cell: -1." in dialog._diagnostics.text()
    dialog.close()


def test_magnetic_texture_dialog_magnitude_map_toggles_the_preview(qapp) -> None:
    dialog = MagneticTextureDialog(_multi_species_structure())
    dialog._amplitude.setText("0.385")
    dialog.show()
    qapp.processEvents()

    assert dialog._preview_stack.currentWidget() is dialog._preview
    assert dialog._magnitude_mesh is None
    dialog._magnitude_map_check.setChecked(True)
    qapp.processEvents()

    assert dialog._preview_stack.currentWidget() is dialog._magnitude_frame
    assert "moment magnitude |m|" in dialog._preview_legend.text()
    assert not dialog._preview_dark_button.isVisible()
    assert dialog._magnitude_mesh is not None
    assert dialog._magnitude_mesh.get_array().size == 32 * 32
    cell_axes = dialog._magnitude_figure.axes[0]
    colorbar_axes = dialog._magnitude_figure.axes[1]
    assert colorbar_axes.get_ylabel() == "|m| (μB)"
    for size in ((860, 560), (1000, 1100), (1500, 700)):
        dialog.resize(*size)
        qapp.processEvents()
        dialog._magnitude_canvas.draw()
        renderer = dialog._magnitude_canvas.get_renderer()
        cell = cell_axes.get_window_extent(renderer)
        colorbar = colorbar_axes.get_window_extent(renderer)
        assert cell.width == pytest.approx(cell.height, abs=1.0), size
        assert colorbar.height == pytest.approx(cell.height, abs=1.0), size
        assert colorbar.y0 == pytest.approx(cell.y0, abs=1.0), size
        decorations = colorbar_axes.get_tightbbox(renderer)
        assert decorations.x1 <= dialog._magnitude_figure.bbox.x1, size

    dialog._grid_u.setValue(16)
    assert dialog._magnitude_mesh.get_array().size == 16 * 32

    dialog._magnitude_map_check.setChecked(False)
    qapp.processEvents()
    assert dialog._preview_stack.currentWidget() is dialog._preview
    assert "out-of-plane moment" in dialog._preview_legend.text()
    assert dialog._preview_dark_button.isVisible()
    dialog.close()


def test_magnetic_texture_dialog_reuses_the_charge_while_the_texture_is_unchanged(
    qapp, monkeypatch
) -> None:
    dialog = MagneticTextureDialog(_multi_species_structure())
    calls = []
    original = magnetic_field_widget_module.topological_charge_magnetic_unit_cell

    def counting(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        magnetic_field_widget_module, "topological_charge_magnetic_unit_cell", counting
    )
    dialog._label_edit.setText("Renamed texture")
    dialog._magnitude_map_check.setChecked(True)
    assert calls == []
    assert "Topological charge per magnetic cell: -1." in dialog._diagnostics.text()

    _set_direction(dialog._core_direction, (0.0, 0.0, 1.0))
    assert len(calls) == 1
    assert "Topological charge per magnetic cell: 1." in dialog._diagnostics.text()
    dialog.close()


def test_magnetic_texture_dialog_unticking_the_map_brings_a_popped_out_map_back(qapp) -> None:
    dialog = MagneticTextureDialog(_multi_species_structure())
    dialog.show()
    dialog._magnitude_map_check.setChecked(True)
    dialog._magnitude_frame.pop_out()
    qapp.processEvents()
    assert dialog._magnitude_frame.is_popped_out()

    dialog._magnitude_map_check.setChecked(False)
    qapp.processEvents()

    assert not dialog._magnitude_frame.is_popped_out()
    assert dialog._preview_stack.currentWidget() is dialog._preview
    dialog.close()


def test_magnetic_texture_dialog_magnitude_map_of_normalized_texture_matches_reported_std(
    qapp,
) -> None:
    dialog = MagneticTextureDialog(_multi_species_structure())
    dialog._normalise_check.setChecked(True)
    dialog._magnitude_map_check.setChecked(True)

    magnitudes = np.asarray(dialog._magnitude_mesh.get_array(), dtype=float)
    reported = texture_diagnostics(dialog.texture(), dialog._grid_resolution())

    assert float(np.std(magnitudes)) == pytest.approx(reported.magnitude_std, rel=1e-9)
    assert float(np.ptp(magnitudes)) < 0.1 * float(np.mean(magnitudes))
    assert f"std {reported.magnitude_std:.4g}" in dialog._diagnostics.text()
    dialog.close()


def test_magnetic_structure_builder_adds_texture_as_single_grouped_row(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-texture.mufinder", "Magnetic Texture")
    service.add_structure(document, _host_structure())

    class _Dialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def q_basis_vectors(self):
            return ((0.25, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 0.0))

        def component(self, transform, q_basis):
            assert transform == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
            return MagneticComponent(
                component_id="texture",
                label="Texture",
                kind="texture",
                selector=AtomSelector.all(),
                terms=(
                    FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([1.0, 0.0, 0.0])),
                    FourierTerm((0.0, 1.0, 0.0), ComplexVector.from_vector([0.0, 1.0, 0.0])),
                ),
                metadata={"type": "texture", "texture_mode": "custom"},
            )

    monkeypatch.setattr(magnetic_field_widget_module, "MagneticTextureDialog", _Dialog)

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    dialog._set_model(MagneticStructureModel())
    dialog._add_texture()

    assert len(dialog.model().components) == 1
    assert dialog.model().components[0].kind == "texture"
    assert dialog.model().q_basis_rlu[0] == (0.25, 0.0, 0.0)
    assert dialog.model().q_basis_rlu[1] == (0.0, 0.25, 0.0)
    assert dialog._component_table.rowCount() == 1
    assert "Fourier term" in dialog._component_table.item(0, 5).text()
    dialog.close()


def test_magnetic_structure_builder_edits_texture_component(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-texture-edit.mufinder", "Magnetic Texture Edit"
    )
    service.add_structure(document, _host_structure())
    original = MagneticComponent(
        component_id="texture",
        label="Texture",
        kind="texture",
        selector=AtomSelector.all(),
        terms=(FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([1.0, 0.0, 0.0])),),
        metadata={"type": "texture", "texture_mode": "custom"},
    )
    replacement = replace(original, label="Edited texture")

    class _Dialog:
        def __init__(self, *args, **kwargs):
            assert kwargs["component"] == original

        def exec(self):
            return QDialog.DialogCode.Accepted

        def q_basis_vectors(self):
            return ((0.0, 0.25, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        def component(self, transform, q_basis):
            assert transform == ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            return replacement

    monkeypatch.setattr(magnetic_field_widget_module, "MagneticTextureDialog", _Dialog)

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    dialog._set_model(
        MagneticStructureModel(
            q_basis_rlu=((0.25, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), components=(original,)
        )
    )
    dialog._component_table.selectRow(0)

    dialog._edit_selected_component()

    assert dialog.model().components[0].label == "Edited texture"
    assert dialog.model().q_basis_rlu[0] == (0.0, 0.25, 0.0)
    dialog.close()


@pytest.mark.parametrize("texture", [False, True])
def test_magnetic_structure_builder_edit_keeps_disabled_state(
    qapp, tmp_path, monkeypatch, texture
) -> None:
    original = replace(_component("one", "One", 0), enabled=False)
    if texture:
        original = replace(
            original, kind="texture", metadata={"type": "texture", "texture_mode": "custom"}
        )

    class _Dialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def q_basis_coefficients(self):
            return None

        def q_vector(self):
            return (0.0, 0.0, 0.0)

        def q_basis_vectors(self):
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        def component(self, *_args):
            # The real editors always build an enabled component.
            return replace(original, label="Edited", enabled=True)

    dialog_name = "MagneticTextureDialog" if texture else "MagneticComponentDialog"
    monkeypatch.setattr(magnetic_field_widget_module, dialog_name, _Dialog)
    dialog = _builder_with_components(tmp_path, original)
    dialog._component_table.selectRow(0)

    dialog._edit_selected_component()

    assert dialog.model().components[0].label == "Edited"
    assert dialog.model().components[0].enabled is False
    assert dialog._component_table.item(0, 0).checkState() == Qt.CheckState.Unchecked
    dialog.close()


def test_magnetic_structure_builder_texture_edit_keeps_atom_phases_on_atoms_still_selected(
    qapp, tmp_path, monkeypatch
) -> None:
    original = replace(
        _component("one", "One", 0),
        selector=AtomSelector.all(),
        atom_phase_turns={0: 0.1, 1: 0.5},
        kind="texture",
        metadata={"type": "texture", "texture_mode": "custom"},
    )

    class _Dialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def q_basis_coefficients(self):
            return None

        def q_vector(self):
            return (0.0, 0.0, 0.0)

        def q_basis_vectors(self):
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        def component(self, *_args):
            # The texture editor rebuilds the component without atom phases, here on atom 2 only.
            return replace(
                original, selector=AtomSelector.indices([1]), atom_phase_turns={}, label="Edited"
            )

    monkeypatch.setattr(magnetic_field_widget_module, "MagneticTextureDialog", _Dialog)
    dialog = _builder_with_components(tmp_path, original)
    dialog._component_table.selectRow(0)

    dialog._edit_selected_component()

    assert dialog.model().components[0].label == "Edited"
    assert dialog.model().components[0].atom_phase_turns == {1: 0.5}
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
    dialog = MagneticComponentDialog(_multi_species_structure())

    assert dialog._atom_phase_summary.text() == "None"
    assert dialog._atom_phase_button.text() == "Per-Atom Phases..."
    assert dialog.component((0.0, 0.0, 0.0)).atom_phase_turns == {}
    dialog.close()


def test_component_dialog_atom_phases_follow_the_target(qapp) -> None:
    dialog = MagneticComponentDialog(
        _multi_species_structure(),
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

    monkeypatch.setattr(magnetic_field_widget_module, "AtomPhaseDialog", _PhaseDialog)
    dialog = MagneticComponentDialog(
        _multi_species_structure(),
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
    dialog = AtomPhaseDialog(_multi_species_structure(), (0, 2), {2: 0.25})
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
        magnetic_field_widget_module.QMessageBox,
        "warning",
        lambda _parent, _title, body: warnings.append(body),
    )
    dialog = AtomPhaseDialog(_multi_species_structure(), (0, 2), {})
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
    dialog = AtomPhaseDialog(_multi_species_structure(), (0, 2), {0: 0.5, 2: 0.25})

    dialog._clear_button.click()

    assert asked == ["Clear Per-Atom Phases"]
    assert dialog.atom_phase_turns() == ({} if confirmed else {0: 0.5, 2: 0.25})
    dialog.close()


def test_magnetic_structure_builder_shutdowns_viewer_once(qapp) -> None:
    dialog = MagneticStructureBuilderDialog()
    calls = {"shutdown": 0}

    def _shutdown() -> None:
        calls["shutdown"] += 1

    dialog._viewer.shutdown = _shutdown  # type: ignore[method-assign]

    dialog.done(0)
    dialog.done(0)

    assert calls["shutdown"] == 1
    assert dialog._viewer_shutdown is True


def test_component_dialog_adapts_atom_target_options(qapp) -> None:
    dialog = MagneticComponentDialog(_multi_species_structure(), selected_indices=(0, 2))

    assert dialog._all_atoms_radio.isEnabled()
    assert dialog._species_radio.isEnabled()
    assert dialog._selected_radio.isEnabled()
    assert dialog._component_info_button.objectName() == "mufinderHelpButton"
    assert not dialog._component_info_button.icon().isNull()

    assert dialog.selector() == AtomSelector.indices([0, 2])
    dialog.close()


def test_component_dialog_target_cards_use_compact_grid(qapp) -> None:
    dialog = MagneticComponentDialog(_multi_species_structure(), selected_indices=(0, 2))

    target_layout = dialog._all_atoms_radio.parentWidget().parentWidget().layout()

    assert isinstance(target_layout, QGridLayout)
    assert target_layout.itemAtPosition(0, 0).widget() is dialog._all_atoms_radio.parentWidget()
    assert target_layout.itemAtPosition(0, 1).widget() is dialog._species_radio.parentWidget()
    assert target_layout.itemAtPosition(1, 0).widget() is dialog._equivalent_radio.parentWidget()
    assert target_layout.itemAtPosition(1, 1).widget() is dialog._selected_radio.parentWidget()
    dialog.close()


def test_component_dialog_disables_selection_target_without_selection(qapp) -> None:
    dialog = MagneticComponentDialog(_multi_species_structure(), selected_indices=())

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

    label = _selector_label(AtomSelector.symmetry_equivalent(0), structure)

    assert label == "Fe site: atom 1 at (0, 0, 0); 2 equivalent atoms (1, 2)"


def test_component_dialog_presets_create_expected_components(qapp) -> None:
    dialog = MagneticComponentDialog(_multi_species_structure(), selected_indices=(0,))
    dialog._selected_radio.setChecked(True)
    dialog._q_z.setText("0.5")
    dialog._amplitude.setText("2.5")
    _set_direction(dialog._primary_direction, (0.0, 0.0, 2.0))

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
    _set_direction(dialog._primary_direction, (1.0, 0.0, 0.0))
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
        _multi_species_structure(), component=component, q_vector=(0.0, 0.0, 0.25)
    )

    assert float(dialog._cone_angle.text()) == pytest.approx(120.0)
    round_trip = dialog.component((1.0, 0.0, 0.0))
    assert round_trip.terms[0].coefficient.real[2] < 0.0
    dialog.close()


def test_component_moment_label_summarises_conical_components(qapp) -> None:
    dialog = MagneticComponentDialog(_multi_species_structure(), selected_indices=(0,))
    dialog._selected_radio.setChecked(True)
    dialog._kind_combo.setCurrentText("Conical")
    dialog._q_z.setText("0.5")
    dialog._amplitude.setText("2.5")
    dialog._cone_angle.setText("60")
    _set_direction(dialog._primary_direction, (1.0, 0.0, 0.0))

    label = _component_moment_label(dialog.component((1.0, 0.0, 0.0)))

    assert label == "Conical α≈60°\nstatic 1.25 μB\nrotating 2.17 μB"
    dialog.close()


def test_component_dialog_decimal_inputs_validate_ranges(qapp) -> None:
    dialog = MagneticComponentDialog(_multi_species_structure(), selected_indices=(0,))

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
    dialog = MagneticComponentDialog(_multi_species_structure(), selected_indices=(0,))
    dialog._kind_combo.setCurrentText("Helix")
    dialog._q_z.setText("0.5")
    _set_direction(dialog._primary_direction, (1.0, 1.0, 1.0))

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
    dialog = MagneticComponentDialog(_multi_species_structure(), selected_indices=(0,))
    dialog._kind_combo.setCurrentText("Helix")

    try:
        dialog.component((1.0, 0.0, 0.0))
    except ValueError as exc:
        assert "q must be non-zero" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected zero-q helix to fail")
    dialog.close()


def test_component_dialog_help_documents_current_preset_parameters(qapp) -> None:
    dialog = MagneticComponentDialog(_multi_species_structure(), selected_indices=(0,))
    dialog._selected_radio.setChecked(True)
    dialog._amplitude.setText("2.5")
    dialog._q_z.setText("0.5")
    dialog._phase.setText("0.25")
    _set_direction(dialog._primary_direction, (0.0, 0.0, 1.0))
    _set_direction(dialog._secondary_direction, (1.0, 0.0, 0.0))

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


def test_magnetic_structure_builder_zero_q_component_does_not_consume_q_basis(
    qapp, tmp_path
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-zero-q.mufinder", "Magnetic Zero Q")
    service.add_structure(document, _host_structure())

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    full_basis = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))
    dialog._set_model(MagneticStructureModel(q_basis_rlu=full_basis))

    q_basis, q_coeff = dialog._q_basis_for((0.0, 0.0, 0.0))

    assert q_basis == full_basis
    assert q_coeff == (0.0, 0.0, 0.0)
    dialog.close()


def test_magnetic_structure_builder_finite_q_after_zero_q_component_uses_free_basis_row(
    qapp, tmp_path
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-fm-then-helix.mufinder", "Magnetic FM Then Helix"
    )
    service.add_structure(document, _host_structure())

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    stale_full_basis = ((0.25, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 0.25))
    zero_q_component = MagneticComponent(
        component_id="fm",
        label="FM",
        kind="collinear",
        selector=AtomSelector.all(),
        terms=(FourierTerm((0.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 1.0])),),
    )
    dialog._set_model(
        MagneticStructureModel(q_basis_rlu=stale_full_basis, components=(zero_q_component,))
    )

    q_basis, q_coeff = dialog._q_basis_for((0.0, 0.0, 0.5))

    assert q_basis[0] == (0.0, 0.0, 0.5)
    assert q_coeff == (1.0, 0.0, 0.0)
    dialog.close()


def _q_component(
    component_id: str, label: str, q_coeff, kind: str = "spin_density_wave", **kwargs
) -> MagneticComponent:
    return MagneticComponent(
        component_id=component_id,
        label=label,
        kind=kind,
        selector=AtomSelector.indices([0]),
        terms=(FourierTerm(q_coeff, ComplexVector.from_vector([0.0, 0.0, 1.0])),),
        **kwargs,
    )


def _builder_with_q_basis(
    tmp_path, q_basis, *components: MagneticComponent
) -> MagneticStructureBuilderDialog:
    dialog = _builder_with_components(tmp_path)
    dialog._set_model(MagneticStructureModel(q_basis_rlu=q_basis, components=components))
    return dialog


def _q_basis_texts(dialog: MagneticStructureBuilderDialog, column: int) -> list[str]:
    return [dialog._q_basis_table.item(row, column).text() for row in range(3)]


def test_magnetic_structure_builder_q_basis_panel_shows_rows_and_their_users(
    qapp, tmp_path
) -> None:
    texture = _q_component(
        "texture", "Skyrmion", (1.0, -1.0, 0.0), kind="texture", metadata={"type": "texture"}
    )
    dialog = _builder_with_q_basis(
        tmp_path,
        ((0.25, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 0.0)),
        _q_component("sdw", "SDW", (1.0, 0.0, 0.0)),
        replace(texture, enabled=False),
    )
    table = dialog._q_basis_table

    assert [table.horizontalHeaderItem(column).text() for column in range(3)] == [
        "Row",
        "q (r.l.u.)",
        "Used by",
    ]
    assert _q_basis_texts(dialog, 0) == ["q1", "q2", "q3"]
    assert _q_basis_texts(dialog, 1) == ["0.25, 0, 0", "0, 0.25, 0", "0, 0, 0"]
    assert _q_basis_texts(dialog, 2) == [
        "SDW, Skyrmion (disabled)",
        "Skyrmion (disabled)",
        "Unused",
    ]
    assert table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable == Qt.ItemFlag.NoItemFlags
    assert "texture" in table.item(0, 1).toolTip()
    assert table.item(2, 1).flags() & Qt.ItemFlag.ItemIsEditable
    assert dialog._component_table.item(0, 4).text() == "q1 = (0.25, 0, 0)"
    assert dialog._component_table.item(1, 4).text() == "q1 − q2 = (0.25, -0.25, 0)"
    dialog.close()


def test_magnetic_structure_builder_q_basis_panel_highlights_selected_components_rows(
    qapp, tmp_path
) -> None:
    dialog = _builder_with_q_basis(
        tmp_path,
        ((0.25, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 0.0)),
        _q_component("one", "One", (1.0, 0.0, 0.0)),
        _q_component("two", "Two", (0.0, 1.0, 0.0)),
    )
    table = dialog._q_basis_table

    assert not any(table.item(row, 0).font().bold() for row in range(3))
    _select_builder_rows(dialog, 1)
    assert [table.item(row, 0).font().bold() for row in range(3)] == [False, True, False]
    assert table.item(1, 0).toolTip() == "Used by the selected component"
    _select_builder_rows(dialog, 0, 1)
    assert [table.item(row, 1).font().bold() for row in range(3)] == [True, True, False]
    assert dialog.model().q_basis_rlu[0] == (0.25, 0.0, 0.0)
    dialog.close()


def test_magnetic_structure_builder_q_basis_users_follow_renames_and_enable_toggles(
    qapp, tmp_path
) -> None:
    dialog = _builder_with_q_basis(
        tmp_path,
        ((0.25, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        _q_component("one", "One", (1.0, 0.0, 0.0)),
        _q_component("two", "Two", (1.0, 0.0, 0.0)),
    )
    table = dialog._component_table

    table.item(0, 1).setText("Renamed")
    assert _q_basis_texts(dialog, 2)[0] == "Renamed, Two"

    table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    assert _q_basis_texts(dialog, 2)[0] == "Renamed, Two (disabled)"

    table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    assert _q_basis_texts(dialog, 2)[0] == "Renamed, Two"
    assert dialog.model().q_basis_rlu[0] == (0.25, 0.0, 0.0)
    dialog.close()


def test_magnetic_structure_builder_q_basis_edit_moves_every_component_on_the_row(
    qapp, tmp_path
) -> None:
    dialog = _builder_with_q_basis(
        tmp_path,
        ((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        _q_component("one", "One", (1.0, 0.0, 0.0)),
        _q_component("two", "Two", (1.0, 0.0, 0.0)),
    )
    _select_builder_rows(dialog, 0)

    dialog._q_basis_table.item(0, 1).setText("(0 0 0.25)")

    assert dialog.model().q_basis_rlu[0] == (0.0, 0.0, 0.25)
    assert _q_basis_texts(dialog, 1)[0] == "0, 0, 0.25"
    assert [dialog._component_table.item(row, 4).text() for row in range(2)] == [
        "q1 = (0, 0, 0.25)"
    ] * 2
    assert dialog._selected_component_ids() == ("one",)
    assert dialog._q_basis_table.item(0, 0).font().bold()
    assert dialog._viewer._spins is not None
    restored = MagneticStructureModel.from_json(dialog.model().to_json())
    assert restored.q_basis_rlu[0] == (0.0, 0.0, 0.25)
    dialog.close()


@pytest.mark.parametrize(
    ("row", "text", "message"),
    [
        (0, "0.1, 0", "three numbers"),
        (0, "a, b, c", "not three numbers"),
        (0, "0, 0, 0", "cannot be zero"),
        (0, "0, 0.25, 0", "q2 already holds"),
        (1, "0.1, 0.1, 0", "keep its direction"),
    ],
)
def test_magnetic_structure_builder_q_basis_edit_rejects_invalid_vectors(
    qapp, tmp_path, monkeypatch, row: int, text: str, message: str
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        magnetic_field_widget_module.QMessageBox,
        "warning",
        lambda _parent, _title, body: warnings.append(body),
    )
    dialog = _builder_with_q_basis(
        tmp_path,
        ((0.0, 0.0, 0.5), (0.0, 0.25, 0.0), (0.0, 0.0, 0.0)),
        _q_component("one", "One", (1.0, 0.0, 0.0)),
        _q_component("helix", "Helix", (0.0, 1.0, 0.0), kind="helix"),
    )
    before = dialog.model()

    dialog._q_basis_table.item(row, 1).setText(text)

    assert len(warnings) == 1 and message in warnings[0]
    assert dialog.model() == before
    assert _q_basis_texts(dialog, 1) == ["0, 0, 0.5", "0, 0.25, 0", "0, 0, 0"]
    dialog.close()


@pytest.mark.parametrize("key", [Qt.Key.Key_Return, Qt.Key.Key_Enter])
def test_magnetic_structure_builder_enter_in_preview_repeats_keeps_dialog_open(
    qapp, tmp_path, key
) -> None:
    dialog = _builder_with_components(tmp_path, _component("one", "One", 0))
    finished: list[int] = []
    dialog.finished.connect(finished.append)
    dialog.show()
    qapp.processEvents()
    dialog._repeat_c.setFocus()
    dialog._repeat_c.selectAll()

    QTest.keyClicks(dialog._repeat_c, "10")
    QTest.keyClick(dialog._repeat_c, key)
    qapp.processEvents()

    assert finished == []
    assert dialog.isVisible()
    assert dialog._repeat_c.value() == 10
    assert dialog._preview_status.text() == "Preview: full magnetic structure (1x1x10)"
    dialog.close()


def test_magnetic_structure_builder_supercell_preview_propagates_moments(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-supercell.mufinder", "Magnetic Supercell"
    )
    host = Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="afm",
                label="AFM",
                kind="spin_density_wave",
                selector=AtomSelector.all(),
                terms=(FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 2.0])),),
            ),
        ),
    )
    dialog._repeat_c.setValue(2)
    dialog._set_model(model)

    assert dialog._viewer._spins is not None
    assert dialog._viewer._spins.shape == (2, 3)
    assert sorted(round(value, 6) for value in dialog._viewer._spins[:, 2]) == [-2.0, 2.0]
    assert len(dialog._viewer._structure.species) == 2
    dialog.close()


def _add_uep_cluster(document) -> None:
    """Give ``document`` one UEP batch cluster at (0.1, 0.2, 0.3)."""

    job_id = uuid4()
    document.jobs[job_id] = Job(id=job_id, label="UEP batch")
    document.ui_state["uep_analysis_states"] = {
        str(job_id): {
            "batch_results": [
                {
                    "optimized_fractional_position": [0.1, 0.2, 0.3],
                    "optimized_position_ang": [0.3, 0.6, 0.9],
                }
            ],
            "batch_summary_data": {
                "clusters": [
                    {
                        "cluster_index": 2,
                        "size": 4,
                        "representative_index": 0,
                    }
                ]
            },
        }
    }


def _uep_cluster_document(tmp_path):
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-uep-sites.mufinder", "Magnetic UEP Sites"
    )
    service.add_structure(document, _host_structure())
    _add_uep_cluster(document)
    return document


def test_muon_site_selection_dialog_lists_uep_cluster_muon_sites(qapp, tmp_path) -> None:
    document = _uep_cluster_document(tmp_path)

    dialog = MuonSiteSelectionDialog(document, _host_structure())

    assert dialog._site_table.rowCount() == 1
    selected = dialog.selected_muon_site()
    assert selected is not None
    assert selected.source == "uep_clustering"
    assert selected.label == "UEP cluster 2"
    assert selected.source_label == "UEP batch"
    assert selected.cluster_label == "Cluster 2"
    assert selected.fractional_position == (0.1, 0.2, 0.3)
    assert not dialog._distortion_checkbox.isEnabled()
    dialog.close()


_RELAXED_SITE = (0.5, 0.0, 0.0)


def _relaxation_cluster_document(tmp_path, relaxed_structure: Structure):
    """A project whose relaxation-analysis workspace has one cluster represented by ``relaxed_structure``."""

    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-relax-sites.mufinder", "Magnetic Relax Sites"
    )
    service.add_structure(document, _host_structure())
    document.structures[relaxed_structure.id] = relaxed_structure
    job_id = uuid4()
    result = Result(derived_structure_ids=[relaxed_structure.id])
    run = CalculationRun(job_id=job_id, result_id=result.id)
    document.results[result.id] = result
    document.runs[run.id] = run
    document.ui_state["relaxation_analysis_workspaces"] = [
        {
            "name": "Positive muon relaxations",
            "clustering": {
                "result": {
                    "clusters": [
                        {
                            "cluster_index": 1,
                            "size": 3,
                            "representative_job_id": str(job_id),
                            "representative_fractional_position": list(_RELAXED_SITE),
                            "representative_cartesian_position_ang": [1.5, 0.0, 0.0],
                        }
                    ]
                }
            },
        }
    ]
    return document


def _displaced_iron_supercell() -> Structure:
    """The FeO host relaxed around the muon, with the Fe one cell along +x pushed 0.1 Å away."""

    return relaxed_muon_supercell(
        _host_structure(), (2, 2, 2), _RELAXED_SITE, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)}
    )


def test_muon_site_selection_dialog_lists_relaxation_clusters_with_distortion_option(
    qapp, tmp_path
) -> None:
    relaxed_structure = _displaced_iron_supercell()
    document = _relaxation_cluster_document(tmp_path, relaxed_structure)
    set_show_every_relaxed_site(document, MAGNETIC_FIELDS_SCOPE, True)

    dialog = MuonSiteSelectionDialog(document, _host_structure())
    _set_site_source(dialog, "relaxation_clustering")

    assert dialog._site_table.rowCount() == 1
    selected = dialog.selected_muon_site()
    assert selected is not None
    assert selected.source == "relaxation_clustering"
    assert selected.label == "Positive muon relaxations cluster 1"
    assert selected.distortion_structure_id == relaxed_structure.id
    assert selected.include_distortion is False
    assert dialog._distortion_checkbox.isEnabled()

    dialog._distortion_checkbox.setChecked(True)
    assert dialog.selected_muon_site().include_distortion is True
    dialog.close()


def test_muon_site_selection_dialog_restores_the_distortion_choice(qapp, tmp_path) -> None:
    document = _relaxation_cluster_document(tmp_path, _displaced_iron_supercell())
    set_show_every_relaxed_site(document, MAGNETIC_FIELDS_SCOPE, True)
    first = MuonSiteSelectionDialog(document, _host_structure())
    _set_site_source(first, "relaxation_clustering")
    first._distortion_checkbox.setChecked(True)
    chosen = first.selected_muon_site()
    first.close()

    reopened = MuonSiteSelectionDialog(document, _host_structure(), chosen)

    assert reopened._distortion_checkbox.isChecked()
    assert reopened.selected_muon_site().include_distortion is True
    _set_site_source(reopened, "custom")
    assert not reopened._distortion_checkbox.isEnabled()
    assert not reopened._distortion_checkbox.isChecked()
    assert reopened.selected_muon_site().include_distortion is False
    reopened.close()


def test_muon_site_selection_dialog_custom_muon_site(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-custom-site.mufinder", "Magnetic Custom Site"
    )
    service.add_structure(document, _host_structure())

    dialog = MuonSiteSelectionDialog(document, _host_structure())
    _set_site_source(dialog, "custom")
    dialog._custom_x.setText("0.25")
    dialog._custom_y.setText("0.5")
    dialog._custom_z.setText("0.75")

    selected = dialog.selected_muon_site()

    assert selected is not None
    assert selected.source == "custom"
    assert selected.fractional_position == (0.25, 0.5, 0.75)
    assert selected.cartesian_position == (0.75, 1.5, 2.25)
    dialog.close()


def test_magnetic_field_widget_calculates_commensurate_fields(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-calc.mufinder", "Magnetic Calc")
    service.add_structure(document, _host_structure())

    calls: list[dict[str, object]] = []

    class _Result:
        site_label = "site 1"
        total = (0.1, 0.2, 0.3)
        magnitude = float(np.linalg.norm(total))
        distortion = None

    def _calculate(structure, model, muon_sites, *, options, distorted_structures=None):
        calls.append(
            {
                "structure": structure,
                "model": model,
                "muon_sites": muon_sites,
                "options": options,
                "distorted_structures": distorted_structures,
            }
        )
        return (_Result(),)

    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(magnetic_field_widget_module, "calculate_commensurate_fields", _calculate)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.25, 0.5, 0.75))
    widget._refresh_site_summary()
    widget._none_radio.setChecked(True)

    widget._calculate_fields()

    assert calls
    assert calls[0]["muon_sites"] == [(0.25, 0.5, 0.75)]
    assert calls[0]["distorted_structures"] == [None]
    assert calls[0]["options"].demag_shape == "none"
    assert calls[0]["options"].supercell == (100, 100, 100)
    assert widget._result_table.rowCount() == 1
    assert widget._result_table.item(0, 1).text() == "0.1"
    assert widget._result_table.item(0, 5).text() == "field"
    assert widget.collect_state()["magnetic_field_results"]["kind"] == "commensurate"
    assert widget.collect_state()["magnetic_field_results"]["results"][0][
        "total_T"
    ] == pytest.approx([0.1, 0.2, 0.3])
    assert "commensurate" in widget._calculation_status.text()
    widget.close()


def _fake_muesr_without_lattice_field(monkeypatch) -> None:
    """Run the real field helpers with a muesr stand-in that returns zero lattice fields."""

    class _Sample:
        def __init__(self) -> None:
            self.muons: list[object] = []

        def add_muon(self, position) -> None:
            self.muons.append(position)

        def new_mm(self) -> None:
            self.mm = type("_MM", (), {})()

    def _build_sample(_structure, muons):
        sample = _Sample()
        for muon in muons:
            sample.add_muon(muon)
        return sample

    def _locfield(sample, *_args, **_kwargs):
        return [type("_Field", (), {"Dipolar": np.zeros(3)})() for _ in sample.muons]

    monkeypatch.setattr(magnetic_fields_module, "_build_muesr_sample", _build_sample)
    monkeypatch.setattr(
        magnetic_fields_module, "_load_muesr_engines", lambda: (_locfield, lambda *_args: 1.0, None)
    )


def _distortion_widget(document, *, include_distortion: bool) -> MagneticFieldWidget:
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("fe", "Fe", 0),)))
    (site,) = magnetic_field_widget_module._relaxation_cluster_muon_site_options(
        document, _host_structure()
    )
    widget._selected_muon_site = replace(site, include_distortion=include_distortion)
    widget._refresh_site_summary()
    return widget


def test_magnetic_field_widget_applies_the_muon_induced_distortion_correction(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    document = _relaxation_cluster_document(tmp_path, _displaced_iron_supercell())
    widget = _distortion_widget(document, include_distortion=True)

    assert "corrected for muon-induced displacements" in widget._site_summary.text()
    widget._calculate_fields()

    # Only the moved Fe (z moment of 1 μB, 1.5 Å -> 1.6 Å along x from the muon) changes the field.
    expected_bz = (
        -MU0_OVER_4PI_T_M_PER_A * BOHR_MAGNETON_J_PER_T / 1e-30 * (1.0 / 1.6**3 - 1.0 / 1.5**3)
    )
    assert "with muon-induced distortion corrections" in widget._calculation_status.text()
    assert widget._result_table.item(0, 5).text() == "field (distortion-corrected)"
    state = widget.collect_state()
    assert state["selected_muon_site"]["include_distortion"] is True
    payload = state["magnetic_field_results"]["results"][0]
    assert payload["distortion_T"] == pytest.approx([0.0, 0.0, expected_bz], abs=1e-12)
    assert payload["total_T"] == pytest.approx([0.0, 0.0, expected_bz], abs=1e-12)

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(state)
    assert restored.selected_muon_site().include_distortion is True
    assert restored._result_table.item(0, 5).text() == "field (distortion-corrected)"
    restored.close()
    widget.close()


def test_magnetic_field_widget_corrects_an_incommensurate_distribution(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    document = _relaxation_cluster_document(tmp_path, _displaced_iron_supercell())
    widget = _distortion_widget(document, include_distortion=True)
    helix = MagneticComponent(
        component_id="fe",
        label="Fe",
        kind="helix",
        selector=AtomSelector.indices([0]),
        terms=(
            FourierTerm(
                q_coeff=(1.0, 0.0, 0.0),
                coefficient=ComplexVector(real=(1.0, 0.0, 0.0), imaginary=(0.0, 1.0, 0.0)),
            ),
        ),
    )
    widget._set_model(
        MagneticStructureModel(
            q_basis_rlu=((0.12345, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            components=(helix,),
        )
    )

    widget._calculate_fields()

    # The zero lattice field leaves only the moved Fe's dipole change, turning with the helix.
    change = MU0_OVER_4PI_T_M_PER_A * BOHR_MAGNETON_J_PER_T / 1e-30 * (1.5**-3 - 1.6**-3)
    assert widget._calculation_status.text().startswith(
        "Results: sampled 1 incommensurate field distribution(s) with muon-induced distortion corrections"
    )
    assert widget._result_table.item(0, 5).text() == "distribution mean (distortion-corrected)"
    state = widget.collect_state()
    (payload,) = state["magnetic_field_results"]["results"]
    assert change < payload["terms"]["dipolar"]["rms_T"] < 2.0 * change
    assert payload["calculation_parameters"]["distortion_corrected"] is True

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(state)
    assert restored._result_table.item(0, 5).text() == "distribution mean (distortion-corrected)"
    restored.close()
    widget.close()


def test_magnetic_field_widget_leaves_the_field_uncorrected_when_not_requested(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    document = _relaxation_cluster_document(tmp_path, _displaced_iron_supercell())
    widget = _distortion_widget(document, include_distortion=False)

    widget._calculate_fields()

    assert widget._result_table.item(0, 5).text() == "field"
    assert "distortion_T" not in widget.collect_state()["magnetic_field_results"]["results"][0]
    widget.close()


def test_magnetic_field_widget_reports_a_missing_relaxed_structure(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    relaxed = _displaced_iron_supercell()
    document = _relaxation_cluster_document(tmp_path, relaxed)
    widget = _distortion_widget(document, include_distortion=True)
    del document.structures[relaxed.id]

    widget._calculate_fields()

    assert (
        "relaxed structure for the muon-induced distortion correction is no longer available"
        in (widget._calculation_status.text())
    )
    assert widget._result_table.rowCount() == 0
    widget.close()


def test_muon_site_selection_dialog_labels_uep_sites_as_unrelaxed_estimates(qapp, tmp_path) -> None:
    document = _relaxation_cluster_document(tmp_path, _displaced_iron_supercell())
    _add_uep_cluster(document)
    dialog = MuonSiteSelectionDialog(document, _host_structure())

    _set_site_source(dialog, "uep_clustering")
    assert dialog._site_source_combo.currentText() == "UEP clustering (unrelaxed estimate)"
    assert not dialog._estimate_note.isHidden()
    assert "estimates" in dialog._estimate_note.text()

    _set_site_source(dialog, "relaxation_clustering")
    assert dialog._estimate_note.isHidden()
    dialog.close()


def test_muon_site_selection_dialog_offers_relaxation_clusters_first_once_they_exist(
    qapp, tmp_path
) -> None:
    uep_only = MuonSiteSelectionDialog(_uep_cluster_document(tmp_path / "uep"), _host_structure())
    assert uep_only._site_source_combo.currentData() == "uep_clustering"
    uep_only.close()

    document = _relaxation_cluster_document(tmp_path, _displaced_iron_supercell())
    _add_uep_cluster(document)
    # With nothing pinned the relaxation list is empty, so the UEP sites stay first.
    unpinned = MuonSiteSelectionDialog(document, _host_structure())
    assert unpinned._site_source_combo.currentData() == "uep_clustering"
    unpinned.close()

    set_show_every_relaxed_site(document, MAGNETIC_FIELDS_SCOPE, True)
    both = MuonSiteSelectionDialog(document, _host_structure())
    assert both._site_source_combo.currentData() == "relaxation_clustering"
    assert both.selected_muon_site().source == "relaxation_clustering"
    both.close()


def test_magnetic_field_widget_defaults_to_a_relaxation_cluster_once_one_exists(
    qapp, tmp_path
) -> None:
    uep_widget = MagneticFieldWidget()
    uep_widget.set_document(_uep_cluster_document(tmp_path / "uep"))
    assert uep_widget.selected_muon_site().source == "uep_clustering"
    assert "unrelaxed" in uep_widget._site_summary.text()
    uep_widget.close()

    document = _relaxation_cluster_document(tmp_path, _displaced_iron_supercell())
    _add_uep_cluster(document)
    widget = MagneticFieldWidget()
    widget.set_document(document)
    assert widget.selected_muon_site().source == "relaxation_clustering"
    assert "unrelaxed" not in widget._site_summary.text()
    widget.close()


def test_magnetic_field_widget_marks_fields_at_a_uep_site_as_estimates_across_a_reload(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    service = ProjectService()
    document = _uep_cluster_document(tmp_path)
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("fe", "Fe", 0),)))

    widget._calculate_fields()

    assert widget._result_table.item(0, 5).text() == "field, unrelaxed estimate"
    assert "unrelaxed UEP site" in widget._calculation_status.text()
    state = widget.collect_state()
    assert state["magnetic_field_results"]["site_source"] == "uep_clustering"
    document.ui_state["magnetic_field_form"] = store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, state
    )
    service.save_project(document)
    widget.close()

    reloaded = service.load_project(document.project_root)
    restored = MagneticFieldWidget()
    restored.set_document(reloaded)
    restored.set_form_state(
        attach_magnetic_field_results(
            reloaded, MAGNETIC_FIELD_PAGE_OWNER, reloaded.ui_state["magnetic_field_form"]
        )
    )
    assert restored._result_table.item(0, 5).text() == "field, unrelaxed estimate"
    assert "unrelaxed UEP site" in restored._calculation_status.text()
    restored.close()


def test_magnetic_field_widget_does_not_mark_fields_at_a_relaxation_cluster(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    document = _relaxation_cluster_document(tmp_path, _displaced_iron_supercell())
    widget = _distortion_widget(document, include_distortion=False)

    widget._calculate_fields()

    assert widget._result_table.item(0, 5).text() == "field"
    assert "estimate" not in widget._calculation_status.text()
    assert (
        widget.collect_state()["magnetic_field_results"]["site_source"] == "relaxation_clustering"
    )
    widget.close()


def _site_from(source: str) -> magnetic_field_widget_module.MagneticMuonSiteOption:
    return replace(_custom_muon_site((0.25, 0.25, 0.25)), source=source, label=f"{source} site")


def _tensor_dialog(site, results_state=()) -> DipolarTensorDialog:
    tensor = {
        "site_label": "site 1",
        "muon_position": [0.25, 0.25, 0.25],
        "dipolar_tensor_T_per_muB": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
    }
    return DipolarTensorDialog(
        _host_structure(),
        MagneticStructureModel(components=(_component("one", "One", 0),)),
        site,
        include_equivalent_sites=False,
        results_state=results_state,
        calculate_callback=lambda _dialog: (tensor,),
    )


def _nuclear_dialog(site, results_state=None) -> NuclearRelaxationDialog:
    return NuclearRelaxationDialog(
        muon_site=site,
        distortion_available=False,
        results_state=results_state,
        calculate_callback=lambda _dialog: {
            "site_label": "site 1",
            "mode": "zf_polycrystal",
            "sigma": 0.2,
        },
    )


@pytest.mark.parametrize(
    ("source", "estimate"),
    [("uep_clustering", True), ("relaxation_clustering", False), ("custom", False)],
)
def test_tensor_and_nuclear_dialogs_mark_results_at_a_uep_site_as_estimates(
    qapp, source, estimate
) -> None:
    tensor_dialog = _tensor_dialog(_site_from(source))
    tensor_dialog._calculate()

    assert tensor_dialog.results_state()[0]["site_source"] == source
    assert (not tensor_dialog._estimate_note.isHidden()) is estimate
    title = tensor_dialog._result_content.findChildren(magnetic_field_widget_module.QGroupBox)[
        0
    ].title()
    assert title == ("site 1 (unrelaxed estimate)" if estimate else "site 1")
    assert ("unrelaxed UEP site" in tensor_dialog._status.text()) is estimate
    tensor_dialog.close()

    nuclear_dialog = _nuclear_dialog(_site_from(source))
    nuclear_dialog._calculate()

    assert nuclear_dialog.results_state()["site_source"] == source
    assert (not nuclear_dialog._estimate_note.isHidden()) is estimate
    assert ("unrelaxed UEP site" in nuclear_dialog._status.text()) is estimate
    nuclear_dialog.close()


def test_tensor_and_nuclear_estimates_survive_a_reload_whatever_site_is_selected(
    qapp, tmp_path
) -> None:
    service = ProjectService()
    document = _uep_cluster_document(tmp_path)
    widget = MagneticFieldWidget()
    widget.set_document(document)
    site = widget.selected_muon_site()
    assert site.source == "uep_clustering"
    tensor_dialog = _tensor_dialog(site)
    tensor_dialog._calculate()
    nuclear_dialog = _nuclear_dialog(site)
    nuclear_dialog._calculate()
    widget._dipolar_tensor_results_state = tensor_dialog.results_state()
    widget._nuclear_relaxation_results_state = nuclear_dialog.results_state()
    document.ui_state["magnetic_field_form"] = store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, widget.collect_state()
    )
    service.save_project(document)
    for closed in (tensor_dialog, nuclear_dialog, widget):
        closed.close()

    reloaded = service.load_project(document.project_root)
    restored = MagneticFieldWidget()
    restored.set_document(reloaded)
    restored.set_form_state(
        attach_magnetic_field_results(
            reloaded, MAGNETIC_FIELD_PAGE_OWNER, reloaded.ui_state["magnetic_field_form"]
        )
    )
    # The results keep their own source after another site is chosen.
    other_site = _site_from("custom")
    restored_tensor = _tensor_dialog(other_site, restored._dipolar_tensor_results_state)
    restored_nuclear = _nuclear_dialog(other_site, restored._nuclear_relaxation_results_state)
    assert not restored_tensor._estimate_note.isHidden()
    assert not restored_nuclear._estimate_note.isHidden()
    for closed in (restored_tensor, restored_nuclear, restored):
        closed.close()


def test_tensor_and_nuclear_results_without_a_source_show_no_estimate(qapp) -> None:
    uep_site = _site_from("uep_clustering")
    tensor_dialog = _tensor_dialog(
        uep_site,
        [
            {
                "site_label": "site 1",
                "dipolar_tensor_T_per_muB": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
            }
        ],
    )
    nuclear_dialog = _nuclear_dialog(uep_site, {"mode": "zf_polycrystal", "sigma": 0.2})

    assert tensor_dialog._estimate_note.isHidden()
    assert nuclear_dialog._estimate_note.isHidden()
    tensor_dialog.close()
    nuclear_dialog.close()


def test_magnetic_field_widget_enables_nuclear_relaxation_without_magnetic_model(
    qapp, tmp_path
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "nuclear-no-model.mufinder", "Nuclear No Model")
    service.add_structure(
        document,
        Structure(
            label="Si",
            lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            species=["Si"],
            fractional_positions=[[0.0, 0.0, 0.0]],
        ),
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)

    assert not widget._calculate_button.isEnabled()
    assert widget._nuclear_relaxation_button.isEnabled()
    widget.close()


def test_nuclear_relaxation_dialog_controls_follow_mode_and_distortion_state(qapp) -> None:
    dialog = NuclearRelaxationDialog(
        muon_site=_custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
    )

    assert not dialog._distortion.isEnabled()
    assert dialog._quadrupolar.isEnabled()
    assert dialog._direction_x.objectName() == "nuclearDirectionInput"
    assert "QLineEdit#nuclearDirectionInput:disabled" in dialog.styleSheet()
    dialog._regime_combo.setCurrentIndex(dialog._regime_combo.findData("tf"))

    assert not dialog._quadrupolar.isEnabled()
    assert not dialog._quadrupolar.isChecked()
    assert dialog.options().mode == "tf_polycrystal"
    dialog.close()


def test_nuclear_relaxation_dialog_filters_elements(qapp) -> None:
    calls: list[object] = []
    dialog = NuclearRelaxationDialog(
        muon_site=_custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
        elements=("Fe", "O"),
        calculate_callback=lambda d: calls.append(d.options()) or {"mode": "zf_polycrystal"},
    )
    checks = dialog._element_checks

    assert list(checks) == ["Fe", "O"]
    assert all(check.isChecked() for check in checks.values())
    assert dialog.options().elements is None

    checks["Fe"].setChecked(False)
    assert dialog.options().elements == ("O",)
    assert dialog.options_state()["elements"] == ["O"]

    checks["O"].setChecked(False)
    dialog._calculate_button.click()
    assert not calls
    assert dialog._status.text() == "Tick at least one element to include in the sum."

    checks["O"].setChecked(True)
    dialog._calculate_button.click()
    assert calls[-1].elements == ("O",)
    dialog.close()


def test_nuclear_relaxation_dialog_records_the_element_filter_with_the_result(qapp) -> None:
    dialog = NuclearRelaxationDialog(
        muon_site=_custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
        elements=("Fe", "O"),
        calculate_callback=lambda _d: {"mode": "zf_polycrystal", "sigma": 0.2},
    )

    def _elements_row() -> str:
        table = dialog._summary_table
        row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == "Elements")
        return table.item(row, 1).text()

    dialog._calculate_button.click()
    assert "elements" not in dialog.results_state()
    assert _elements_row() == "All"

    dialog._element_checks["O"].setChecked(False)
    dialog._calculate_button.click()
    assert dialog.results_state()["elements"] == ["Fe"]
    assert _elements_row() == "Fe"
    dialog.close()


def test_nuclear_relaxation_dialog_without_elements_keeps_a_saved_filter(qapp) -> None:
    dialog = NuclearRelaxationDialog(
        muon_site=_custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
        options_state={"mode": "zf_polycrystal", "elements": ["Fe"]},
    )

    assert dialog.options().elements == ("Fe",)
    assert dialog.options_state()["elements"] == ["Fe"]
    dialog.close()


def test_nuclear_relaxation_dialog_restores_the_element_filter(qapp) -> None:
    restored = NuclearRelaxationDialog(
        muon_site=_custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
        elements=("Fe", "O"),
        options_state={"mode": "zf_polycrystal", "elements": ["Fe"]},
    )
    every = NuclearRelaxationDialog(
        muon_site=_custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
        elements=("Fe", "O"),
        options_state={"mode": "zf_polycrystal", "elements": None},
    )

    assert restored._element_checks["Fe"].isChecked()
    assert not restored._element_checks["O"].isChecked()
    assert restored.options().elements == ("Fe",)
    assert every.options().elements is None
    restored.close()
    every.close()


def test_magnetic_field_widget_opens_nuclear_relaxation_dialog(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "nuclear-dialog.mufinder", "Nuclear Dialog")
    service.add_structure(document, _host_structure())

    calls: list[dict[str, object]] = []

    class _Dialog:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def options_state(self):
            return {"mode": "zf_polycrystal", "direction": [0.0, 0.0, 1.0]}

        def results_state(self):
            return {"site_label": "site 1", "sigma": 0.2}

    monkeypatch.setattr(magnetic_field_widget_module, "NuclearRelaxationDialog", _Dialog)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._selected_muon_site = _custom_muon_site((0.25, 0.5, 0.75))

    widget._open_nuclear_relaxation_dialog()

    assert calls
    assert calls[0]["muon_site"].fractional_position == (0.25, 0.5, 0.75)
    assert calls[0]["distortion_available"] is False
    assert calls[0]["elements"] == ("Fe", "O")
    assert widget.collect_state()["nuclear_relaxation_results"]["sigma"] == 0.2
    widget.close()


def test_magnetic_field_widget_calculates_nuclear_relaxation_from_dialog(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "nuclear-calc.mufinder", "Nuclear Calc")
    service.add_structure(document, _host_structure())
    calls: list[dict[str, object]] = []

    class _Result:
        def to_json(self):
            return {"site_label": "Custom site", "sigma": 0.123, "mode": "zf_polycrystal"}

    class _Dialog:
        def options(self):
            return magnetic_field_widget_module.NuclearRelaxationOptions(mode="zf_polycrystal")

    def _calculate(structure, muon_position, *, site_label, options):
        calls.append(
            {
                "structure": structure,
                "muon_position": muon_position,
                "site_label": site_label,
                "options": options,
            }
        )
        return _Result()

    monkeypatch.setattr(magnetic_field_widget_module, "calculate_nuclear_relaxation", _calculate)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._selected_muon_site = _custom_muon_site((0.25, 0.5, 0.75))

    result = widget._calculate_nuclear_relaxation_from_dialog(_Dialog())

    assert result.to_json()["sigma"] == 0.123
    assert calls[0]["muon_position"] == (0.25, 0.5, 0.75)
    assert calls[0]["site_label"] == "Custom site"
    widget.close()


def test_magnetic_field_widget_passes_calculation_cell_counts(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-cells.mufinder", "Magnetic Cells")
    service.add_structure(document, _host_structure())

    calls: list[dict[str, object]] = []

    def _calculate(structure, model, muon_sites, *, options, distorted_structures=None):
        calls.append(
            {"structure": structure, "model": model, "muon_sites": muon_sites, "options": options}
        )
        return ()

    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(magnetic_field_widget_module, "calculate_commensurate_fields", _calculate)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.0, 0.0, 0.0))
    widget._calc_cells_a.setValue(24)
    widget._calc_cells_b.setValue(36)
    widget._calc_cells_c.setValue(48)

    widget._calculate_fields()

    assert calls
    assert calls[0]["options"].supercell == (24, 36, 48)
    assert calls[0]["options"].nangles == 32_768
    assert calls[0]["options"].histogram_bins == 512
    widget.close()


def test_magnetic_field_widget_defaults_demag_shape_to_none(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-default-demag.mufinder", "Magnetic Default Demag"
    )
    service.add_structure(document, _host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)

    assert widget._demag_shape() == "none"

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state({"demag_shape": "unknown"})

    assert restored._demag_shape() == "none"
    widget.close()
    restored.close()


class _DemagPreview(magnetic_field_widget_module.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.samples = []

    def set_sample(self, sample, lattice_vectors=None):
        self.samples.append((sample, lattice_vectors))

    def shutdown(self):
        pass


_HEXAGONAL_LATTICE = [[3.0, 0.0, 0.0], [-1.5, 1.5 * np.sqrt(3.0), 0.0], [0.0, 0.0, 5.0]]


def _demag_dialog(monkeypatch, sample=None, lattice_vectors=None):
    monkeypatch.setattr(magnetic_field_widget_module, "_DemagnetizingPyVistaCanvas", _DemagPreview)
    return magnetic_field_widget_module.DemagnetizingFieldAdvancedDialog(
        sample or magnetic_field_widget_module.DemagnetizingSample(shape="ellipsoid"),
        lattice_vectors=lattice_vectors,
    )


def _set_indices(inputs, values) -> None:
    for edit, value in zip(inputs, values, strict=True):
        edit.setText(str(value))


def test_demagnetizing_advanced_dialog_collects_prism_values(qapp, monkeypatch) -> None:
    monkeypatch.setattr(magnetic_field_widget_module, "_DemagnetizingPyVistaCanvas", _DemagPreview)

    dialog = magnetic_field_widget_module.DemagnetizingFieldAdvancedDialog(
        magnetic_field_widget_module.DemagnetizingSample(shape="ellipsoid")
    )
    dialog._shape.setCurrentIndex(dialog._shape.findData("rectangular_prism"))
    dialog._prism_a.setValue(2.0)
    dialog._prism_b.setValue(3.0)
    dialog._prism_c.setValue(4.0)

    sample = dialog.selected_sample()

    assert sample.shape == "rectangular_prism"
    assert sample.prism_lengths == pytest.approx((2.0, 3.0, 4.0))
    assert "N =" in dialog._tensor_label.text()
    # The prism tensor is exact, so no Fourier settings or trace are shown.
    assert "Tr N" not in dialog._tensor_label.text()
    assert not hasattr(dialog, "_fourier_grid")
    dialog.close()


def _plate_sample(**fields):
    x = (np.arange(40) + 0.5) / 40 * 2.0 - 1.0
    outline = x[:, None] ** 2 + x[None, :] ** 2 <= 1.0
    return magnetic_field_widget_module.DemagnetizingSample.image_plate(
        outline, (5.0, 5.0), 0.2, fourier_grid=32, padding=2.0, **fields
    )


def test_demagnetizing_advanced_dialog_needs_an_outline_for_a_plate(qapp, monkeypatch) -> None:
    dialog = _demag_dialog(monkeypatch)
    dialog._shape.setCurrentIndex(dialog._shape.findData("image_plate"))

    assert dialog._stack.currentIndex() == 2
    assert "No outline yet" in dialog._plate_summary.text()
    assert dialog._photo_button.text() == "From photo..."
    assert "no outline" in dialog._tensor_label.text()
    assert not dialog._use_button.isEnabled()
    dialog.close()


def test_demagnetizing_advanced_dialog_takes_the_plate_from_the_photo_dialog(
    qapp, monkeypatch
) -> None:
    plate = _plate_sample(photo=PlatePhoto(path="/photos/crystal.png"))

    class _PhotoDialog:
        def __init__(self, sample, parent=None):
            assert sample.shape == "image_plate"

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selected_sample(self):
            return plate

    monkeypatch.setattr(magnetic_field_widget_module, "SamplePhotoDialog", _PhotoDialog)
    dialog = _demag_dialog(monkeypatch, lattice_vectors=np.eye(3).tolist())
    dialog._shape.setCurrentIndex(dialog._shape.findData("image_plate"))

    dialog._open_photo_dialog()

    assert dialog._use_button.isEnabled()
    assert "Plate from photo 5 x 5 x 0.2 (crystal.png)" in dialog._plate_summary.text()
    assert dialog._photo_button.text() == "Edit outline..."
    assert "Tr N = 1.00000" in dialog._tensor_label.text()
    assert "FFT box" in dialog._tensor_label.text()
    assert dialog._preview.samples[-1][0].shape == "image_plate"
    # The orientation is applied to the plate like any shape.
    dialog._orientation_mode.setCurrentIndex(dialog._orientation_mode.findData("directions"))
    _set_indices(dialog._normal_inputs, (1, 0, 0))
    _set_indices(dialog._in_plane_inputs, (0, 1, 0))
    selected = dialog.selected_sample()
    assert selected.plate_mask == plate.plate_mask
    assert selected.photo == plate.photo
    assert selected.normal_hkl == (1.0, 0.0, 0.0)
    assert demagnetizing_tensor(selected)[0, 0] > 0.8
    dialog.close()


def test_demagnetizing_advanced_dialog_keeps_a_cancelled_photo_dialog_out(
    qapp, monkeypatch
) -> None:
    class _PhotoDialog:
        def __init__(self, sample, parent=None):
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(magnetic_field_widget_module, "SamplePhotoDialog", _PhotoDialog)
    plate = _plate_sample()
    dialog = _demag_dialog(monkeypatch, sample=plate)

    dialog._open_photo_dialog()

    assert dialog.selected_sample() == plate
    dialog.close()


def test_demagnetizing_advanced_dialog_offers_voxel_grid_only_for_a_voxel_seed(
    qapp, monkeypatch
) -> None:
    voxels = magnetic_field_widget_module.DemagnetizingSample.voxel_grid(
        np.ones((6, 4, 2), dtype=bool), 0.5
    )
    dialog = _demag_dialog(monkeypatch, sample=voxels)

    assert dialog._shape.currentData() == "voxel_grid"
    assert "Voxel grid 6 x 4 x 2 voxels" in dialog._voxel_summary.text()
    assert "Python API" in dialog._voxel_summary.text()
    assert dialog.selected_sample() == voxels
    dialog.close()

    plain = _demag_dialog(monkeypatch)
    assert plain._shape.findData("voxel_grid") == -1
    plain.close()


def test_demagnetizing_advanced_dialog_orients_sample_by_crystal_directions(
    qapp, monkeypatch
) -> None:
    dialog = _demag_dialog(monkeypatch, lattice_vectors=np.eye(3).tolist())
    dialog._ellipsoid_c.setValue(3.0)
    assert dialog.selected_sample().rotation is None
    assert "Cartesian x, y and z" in dialog._orientation_hint.text()
    # Aligned ignores the indices, so their rows (labels and fields) are greyed out.
    assert not any(row.isEnabled() for row in dialog._index_rows)
    assert not dialog._normal_inputs[0].isEnabled()

    dialog._orientation_mode.setCurrentIndex(dialog._orientation_mode.findData("directions"))
    assert all(row.isEnabled() for row in dialog._index_rows)
    assert dialog._in_plane_inputs[2].isEnabled()
    _set_indices(dialog._normal_inputs, (1, 1, 0))
    _set_indices(dialog._in_plane_inputs, (0, 0, 1))

    sample = dialog.selected_sample()
    assert sample.normal_hkl == (1.0, 1.0, 0.0)
    assert sample.in_plane_uvw == (0.0, 0.0, 1.0)
    np.testing.assert_allclose(
        sample.rotation_matrix()[:, 2], np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    )
    assert "normal to (h k l)" in dialog._orientation_hint.text()
    assert "°" not in dialog._orientation_hint.text()
    tensor = demagnetizing_tensor(sample)
    assert f"{tensor[0, 1]:.5f}" in dialog._tensor_label.text()
    assert tensor[0, 1] != pytest.approx(0.0)
    assert dialog._use_button.isEnabled()
    preview_sample, preview_lattice = dialog._preview.samples[-1]
    assert preview_sample == sample
    assert preview_lattice == np.eye(3).tolist()
    dialog.close()


def test_demagnetizing_advanced_dialog_reports_angle_of_non_orthogonal_directions(
    qapp, monkeypatch
) -> None:
    dialog = _demag_dialog(monkeypatch, lattice_vectors=_HEXAGONAL_LATTICE)
    dialog._orientation_mode.setCurrentIndex(dialog._orientation_mode.findData("directions"))
    _set_indices(dialog._normal_inputs, (1, 0, 0))
    _set_indices(dialog._in_plane_inputs, (1, 0, 0))

    assert "30.00°" in dialog._orientation_hint.text()
    assert dialog._use_button.isEnabled()
    assert dialog.selected_sample().rotation is not None
    dialog.close()


@pytest.mark.parametrize(
    ("normal", "in_plane", "message"),
    [
        ((1, 1, 0), (2, 2, 0), "parallel"),
        ((0, 0, 0), (1, 0, 0), "non-zero"),
        ((0, "", 1), (1, 0, 0), "three numbers"),
    ],
    ids=["parallel", "zero-normal", "blank-index"],
)
def test_demagnetizing_advanced_dialog_blocks_unusable_directions(
    qapp, monkeypatch, normal, in_plane, message
) -> None:
    dialog = _demag_dialog(monkeypatch, lattice_vectors=np.eye(3).tolist())
    dialog._orientation_mode.setCurrentIndex(dialog._orientation_mode.findData("directions"))
    _set_indices(dialog._normal_inputs, normal)
    _set_indices(dialog._in_plane_inputs, in_plane)

    assert message in dialog._orientation_hint.text()
    assert "Tensor unavailable" in dialog._tensor_label.text()
    assert not dialog._use_button.isEnabled()

    _set_indices(dialog._normal_inputs, (0, 0, 1))
    _set_indices(dialog._in_plane_inputs, (1, 0, 0))
    assert dialog._use_button.isEnabled()
    dialog.close()


def test_demagnetizing_advanced_dialog_without_host_offers_only_aligned(qapp, monkeypatch) -> None:
    dialog = _demag_dialog(monkeypatch)

    assert not dialog._orientation_mode.isEnabled()
    assert dialog._orientation_mode.currentData() == "aligned"
    assert "needs a host structure" in dialog._orientation_hint.text()
    assert not dialog.selected_sample().is_oriented
    dialog.close()


def test_demagnetizing_advanced_dialog_without_host_keeps_saved_directions(
    qapp, monkeypatch
) -> None:
    seed = magnetic_field_widget_module.DemagnetizingSample.oriented(
        np.eye(3), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), shape="ellipsoid"
    )
    dialog = _demag_dialog(monkeypatch, sample=seed)
    dialog._ellipsoid_c.setValue(3.0)

    assert not dialog._orientation_mode.isEnabled()
    assert dialog._orientation_mode.currentData() == "directions"
    assert not any(row.isEnabled() for row in dialog._index_rows)
    assert "kept" in dialog._orientation_hint.text()
    sample = dialog.selected_sample()
    assert sample.semi_axes == (1.0, 1.0, 3.0)
    assert (sample.rotation, sample.normal_hkl, sample.in_plane_uvw) == (
        seed.rotation,
        seed.normal_hkl,
        seed.in_plane_uvw,
    )
    assert dialog._use_button.isEnabled()
    dialog.close()


def test_demagnetizing_advanced_dialog_keeps_a_rotation_saved_without_directions(
    qapp, monkeypatch
) -> None:
    rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    seed = magnetic_field_widget_module.DemagnetizingSample(
        shape="ellipsoid", semi_axes=(1.0, 2.0, 3.0), rotation=rotation
    )
    dialog = _demag_dialog(monkeypatch, sample=seed, lattice_vectors=np.eye(3).tolist())

    assert dialog._orientation_mode.currentData() == "saved"
    assert "saved without crystal directions" in dialog._orientation_hint.text()
    assert not any(row.isEnabled() for row in dialog._index_rows)
    assert dialog.selected_sample() == seed
    expected = demagnetizing_tensor(seed)
    assert expected[0, 0] != pytest.approx(expected[1, 1])
    assert f"[[{expected[0, 0]:.5f}, {expected[0, 1]:.5f}" in dialog._tensor_label.text()

    dialog._orientation_mode.setCurrentIndex(dialog._orientation_mode.findData("aligned"))
    assert dialog.selected_sample().rotation is None
    dialog.close()


def test_demagnetizing_advanced_dialog_reuses_principal_tensor_while_orienting(
    qapp, monkeypatch
) -> None:
    calls = []
    original = magnetic_field_widget_module.principal_demagnetizing_tensor

    def _counting(sample):
        calls.append(sample)
        return original(sample)

    monkeypatch.setattr(magnetic_field_widget_module, "principal_demagnetizing_tensor", _counting)
    dialog = _demag_dialog(monkeypatch, lattice_vectors=np.eye(3).tolist())
    dialog._orientation_mode.setCurrentIndex(dialog._orientation_mode.findData("directions"))
    _set_indices(dialog._normal_inputs, (1, 1, 0))
    _set_indices(dialog._in_plane_inputs, (0, 0, 1))

    assert len(calls) == 1
    dialog._ellipsoid_c.setValue(2.0)
    assert len(calls) == 2
    dialog.close()


def test_demagnetizing_advanced_dialog_restores_saved_directions(qapp, monkeypatch) -> None:
    seed = magnetic_field_widget_module.DemagnetizingSample.oriented(
        _HEXAGONAL_LATTICE, (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), shape="rectangular_prism"
    )
    dialog = _demag_dialog(monkeypatch, sample=seed, lattice_vectors=_HEXAGONAL_LATTICE)

    assert dialog._orientation_mode.currentData() == "directions"
    assert [edit.text() for edit in dialog._in_plane_inputs] == ["1", "1", "0"]
    assert dialog.selected_sample() == seed
    dialog.close()


def test_demagnetizing_pyvista_preview_rotates_shape_and_follows_crystal_axes(
    qapp, monkeypatch
) -> None:
    class _FakeMesh:
        def __init__(self, points=None):
            self.points = points

        def transform(self, matrix, inplace=False):
            return ("transformed", np.asarray(matrix))

    class _FakePv:
        @staticmethod
        def Cube(*, bounds):
            return _FakeMesh()

        @staticmethod
        def Arrow(**kwargs):
            return ("arrow", kwargs)

        @staticmethod
        def PolyData(points, faces):
            return _FakeMesh(points)

    class _FakePlotter(magnetic_field_widget_module.QWidget):
        def __init__(self, parent=None, **kwargs):
            super().__init__(parent)
            self.meshes = []

        def add_mesh(self, mesh, **kwargs):
            self.meshes.append(mesh)

        def set_background(self, *args, **kwargs):
            pass

        enable_parallel_projection = clear = add_point_labels = set_background
        reset_camera = render = set_background

    monkeypatch.setattr(
        magnetic_field_widget_module, "_pyvista_modules", lambda: (_FakePv, _FakePlotter)
    )
    canvas = magnetic_field_widget_module._DemagnetizingPyVistaCanvas()
    ellipsoid = magnetic_field_widget_module.DemagnetizingSample.oriented(
        np.eye(3), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), shape="ellipsoid", semi_axes=(1.0, 1.0, 3.0)
    )

    canvas.set_sample(ellipsoid, _HEXAGONAL_LATTICE)

    points = canvas._plotter.meshes[0].points
    # The long sample c axis now lies along Cartesian x.
    assert np.max(np.abs(points[:, 0])) == pytest.approx(3.0)
    assert np.max(np.abs(points[:, 2])) <= 1.0 + 1e-12
    b_arrow = canvas._plotter.meshes[2][1]
    np.testing.assert_allclose(b_arrow["direction"], (-0.5, np.sqrt(3.0) / 2.0, 0.0))

    prism = replace(ellipsoid, shape="rectangular_prism")
    canvas.set_sample(prism)
    transformed = canvas._plotter.meshes[-5]
    np.testing.assert_allclose(transformed[1][:3, :3], prism.rotation_matrix())
    canvas.close()


def test_demagnetizing_pyvista_axis_triad_uses_active_theme(qapp, monkeypatch) -> None:
    class _FakeMesh:
        pass

    class _FakePv:
        @staticmethod
        def Cube(*, bounds):
            return ("cube", bounds)

        @staticmethod
        def Arrow(**kwargs):
            return ("arrow", kwargs)

        @staticmethod
        def PolyData(points, faces):
            return ("polydata", points, faces)

    class _FakePlotter(magnetic_field_widget_module.QWidget):
        def __init__(self, parent=None, **kwargs):
            super().__init__(parent)
            self.backgrounds = []
            self.meshes = []
            self.labels = []
            self.clear_calls = 0
            self.render_calls = 0

        def set_background(self, color):
            self.backgrounds.append(color)

        def enable_parallel_projection(self):
            pass

        def clear(self):
            self.clear_calls += 1

        def add_mesh(self, mesh, **kwargs):
            self.meshes.append((mesh, kwargs))
            return _FakeMesh()

        def add_point_labels(self, points, labels, **kwargs):
            self.labels.append((tuple(labels), kwargs))

        def reset_camera(self):
            pass

        def render(self):
            self.render_calls += 1

    monkeypatch.setattr(
        magnetic_field_widget_module, "_pyvista_modules", lambda: (_FakePv, _FakePlotter)
    )
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)
    try:
        canvas = magnetic_field_widget_module._DemagnetizingPyVistaCanvas()
        sample = magnetic_field_widget_module.DemagnetizingSample(
            shape="rectangular_prism",
            prism_lengths=(2.0, 3.0, 4.0),
        )

        canvas.set_sample(sample)
        palette = magnetic_field_widget_module.current_palette()

        assert canvas._plotter.backgrounds[-1] == palette.bg_primary
        assert [entry[1]["color"] for entry in canvas._plotter.meshes[1:4]] == [
            palette.accent,
            palette.success,
            palette.status_submitting,
        ]
        assert [entry[1]["text_color"] for entry in canvas._plotter.labels[:3]] == [
            palette.accent,
            palette.success,
            palette.status_submitting,
        ]

        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
        canvas.changeEvent(QEvent(QEvent.Type.ApplicationPaletteChange))
        assert (
            canvas._plotter.backgrounds[-1]
            == magnetic_field_widget_module.current_palette().bg_primary
        )
        assert canvas._plotter.clear_calls >= 2
        clear_calls = canvas._plotter.clear_calls
        canvas.changeEvent(QEvent(QEvent.Type.StyleChange))
        assert canvas._plotter.clear_calls == clear_calls
        canvas.close()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def test_magnetic_field_widget_opens_advanced_demag_dialog(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "advanced-demag.mufinder", "Advanced Demag")
    service.add_structure(document, _host_structure())

    class _Dialog:
        def __init__(self, sample, parent=None, *, lattice_vectors=None):
            assert sample.shape == "ellipsoid"
            assert lattice_vectors == _host_structure().lattice_vectors

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selected_sample(self):
            return magnetic_field_widget_module.DemagnetizingSample(
                shape="ellipsoid",
                semi_axes=(1.0, 2.0, 3.0),
            )

    monkeypatch.setattr(magnetic_field_widget_module, "DemagnetizingFieldAdvancedDialog", _Dialog)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    emitted: list[dict] = []
    widget.form_state_changed.connect(emitted.append)

    widget._open_advanced_demag_dialog()

    options = widget._field_options()
    assert widget._demag_shape() == "ellipsoid"
    assert options.demag_sample is not None
    assert options.demag_sample.semi_axes == pytest.approx((1.0, 2.0, 3.0))
    assert emitted[-1]["demag_shape"] == "ellipsoid"
    assert emitted[-1]["demag_sample"]["semi_axes"] == [1.0, 2.0, 3.0]
    widget.close()


def test_demagnetizing_preview_draws_the_voxels_of_a_plate(qapp, monkeypatch) -> None:
    pyvista = pytest.importorskip("pyvista")

    class _FakePlotter(magnetic_field_widget_module.QWidget):
        def __init__(self, parent=None, **kwargs):
            super().__init__(parent)
            self.meshes = []

        def add_mesh(self, mesh, **kwargs):
            self.meshes.append(mesh)

        def set_background(self, *args, **kwargs):
            pass

        enable_parallel_projection = clear = add_point_labels = set_background
        reset_camera = render = set_background

    monkeypatch.setattr(
        magnetic_field_widget_module, "_pyvista_modules", lambda: (pyvista, _FakePlotter)
    )
    canvas = magnetic_field_widget_module._DemagnetizingPyVistaCanvas()
    plate = _plate_sample(thickness_voxels=4)

    canvas.set_sample(plate)

    surface = canvas._plotter.meshes[0]
    xmin, xmax, ymin, ymax, zmin, zmax = surface.bounds
    assert (xmax - xmin, ymax - ymin) == pytest.approx((5.0, 5.0), rel=1e-6)
    assert zmax - zmin == pytest.approx(0.2)
    assert xmin == pytest.approx(-2.5)
    # Orienting the plate rotates the cached surface rather than rebuilding it.
    built = []
    build = canvas._build_voxel_mesh
    monkeypatch.setattr(canvas, "_build_voxel_mesh", lambda s: built.append(s) or build(s))
    canvas.set_sample(replace(plate, rotation=tuple(map(tuple, np.eye(3)[[1, 2, 0]]))))
    assert built == []
    # A different shape setting builds a new one.
    canvas.set_sample(replace(plate, smoothing=0.5))
    assert len(built) == 1
    # A plate without an outline draws only the three axis arrows and M.
    canvas._plotter.meshes.clear()
    canvas.set_sample(magnetic_field_widget_module.DemagnetizingSample(shape="image_plate"))
    assert len(canvas._plotter.meshes) == 4
    canvas.close()


def test_magnetic_field_widget_restores_a_photographed_plate(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "plate-demag.mufinder", "Plate Demag")
    service.add_structure(document, _host_structure())
    plate = _plate_sample(photo=PlatePhoto(path="/p/disc.png"))

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state({"demag_shape": "image_plate", "demag_sample": plate.to_json()})

    options = widget._field_options()
    assert widget._demag_shape() == "image_plate"
    assert options.demag_sample == plate
    assert widget._advanced_demag_summary.text() == "Plate from photo 5 x 5 x 0.2 (disc.png)"
    assert widget.collect_state()["demag_sample"] == plate.to_json()
    widget.close()


def test_magnetic_field_widget_restores_advanced_demag_state(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "advanced-demag-restore.mufinder", "Advanced Demag Restore"
    )
    service.add_structure(document, _host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(
        {
            "demag_shape": "rectangular_prism",
            "demag_sample": {
                "shape": "rectangular_prism",
                "prism_lengths": [2.0, 4.0, 6.0],
                "fourier_grid": 24,
                "padding": 3.0,
            },
        }
    )

    options = widget._field_options()

    assert widget._demag_shape() == "rectangular_prism"
    assert options.demag_sample is not None
    assert options.demag_sample.prism_lengths == pytest.approx((2.0, 4.0, 6.0))
    assert "Rectangular prism" in widget._advanced_demag_summary.text()
    widget.close()


def test_magnetic_field_widget_persists_and_resolves_demag_orientation(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "advanced-demag-orientation.mufinder", "Advanced Demag Orientation"
    )
    service.add_structure(document, _host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(
        {
            "demag_shape": "ellipsoid",
            "demag_sample": {
                "shape": "ellipsoid",
                "semi_axes": [1.0, 1.0, 3.0],
                "normal_hkl": [1.0, 1.0, 0.0],
                "in_plane_uvw": [0.0, 0.0, 1.0],
            },
        }
    )

    options = widget._field_options()

    assert options.demag_sample is not None
    np.testing.assert_allclose(
        options.demag_sample.rotation_matrix()[:, 2], np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    )
    assert "normal (1 1 0), in-plane [0 0 1]" in widget._advanced_demag_summary.text()

    widget.set_form_state(
        {
            "demag_shape": "ellipsoid",
            "demag_sample": {
                "shape": "ellipsoid",
                "rotation": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            },
        }
    )
    assert widget._advanced_demag_summary.text().endswith("; rotated sample axes")
    widget.set_form_state(
        {
            "demag_shape": "ellipsoid",
            "demag_sample": {
                "shape": "ellipsoid",
                "normal_hkl": [1.0, 1.0, 0.0],
                "in_plane_uvw": [0.0, 0.0, 1.0],
            },
        }
    )
    state = widget.collect_state()
    assert state["demag_sample"]["normal_hkl"] == [1.0, 1.0, 0.0]
    assert state["demag_sample"]["in_plane_uvw"] == [0.0, 0.0, 1.0]
    widget.close()


def test_magnetic_field_widget_measured_magnetisation_reaches_the_options_and_round_trips(
    qapp, tmp_path
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "measured-magnetisation.mufinder", "Measured Magnetisation"
    )
    service.add_structure(document, _host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    emitted: list[dict] = []
    widget.form_state_changed.connect(emitted.append)

    assert widget._field_options().measured_magnetization is None
    assert not any(edit.isEnabled() for edit in widget._measured_magnetization_inputs)
    assert widget._measured_magnetization_unit.text().startswith("μB per FeO")

    widget._measured_magnetization_check.setChecked(True)
    x, y, z = widget._measured_magnetization_inputs
    x.setText("0.5")
    y.setText("")
    z.setText("-1.25")

    assert all(edit.isEnabled() for edit in widget._measured_magnetization_inputs)
    assert widget._field_options().measured_magnetization == (0.5, 0.0, -1.25)
    assert emitted[-1]["use_measured_magnetization"] is True
    assert emitted[-1]["measured_magnetization"] == [0.5, 0.0, -1.25]

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(widget.collect_state())

    assert restored._measured_magnetization_check.isChecked()
    assert restored._field_options().measured_magnetization == (0.5, 0.0, -1.25)

    # Unchecking keeps the values for later but falls back to the model's moment.
    restored._measured_magnetization_check.setChecked(False)
    assert restored._field_options().measured_magnetization is None
    assert restored.collect_state()["measured_magnetization"] == [0.5, 0.0, -1.25]

    restored.set_form_state({})
    assert not restored._measured_magnetization_check.isChecked()
    assert restored.collect_state()["measured_magnetization"] == [0.0, 0.0, 0.0]
    widget.close()
    restored.close()


def test_magnetic_field_widget_applied_field_reaches_the_calculation_and_round_trips(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "applied-field.mufinder", "Applied Field")
    service.add_structure(document, _host_structure())
    seen_options: list[object] = []

    def fake_fields(_host, _model, positions, *, options, **_kwargs):
        seen_options.append(options)
        return (
            magnetic_fields_module.MagneticFieldResult(
                site_label="muon 1",
                muon_position=tuple(positions[0]),
                dipolar=(0.1, 0.0, 0.0),
                lorentz=(0.0, 0.0, 0.0),
                contact=(0.0, 0.0, 0.0),
                demag=(0.0, 0.0, 0.0),
                applied=options.applied_field,
            ),
        )

    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(magnetic_field_widget_module, "calculate_commensurate_fields", fake_fields)
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.25, 0.5, 0.75))
    emitted: list[dict] = []
    widget.form_state_changed.connect(emitted.append)

    assert widget._field_options().applied_field == (0.0, 0.0, 0.0)
    assert all(edit.isEnabled() for edit in widget._applied_field_inputs)
    x, y, z = widget._applied_field_inputs
    x.setText("")
    y.setText("0.01")
    z.setText("-0.25")
    assert emitted[-1]["applied_field_T"] == [0.0, 0.01, -0.25]
    assert not widget._polarisation_button.isEnabled()

    widget._calculate_fields()

    assert seen_options[-1].applied_field == (0.0, 0.01, -0.25)
    assert widget._polarisation_button.isEnabled()
    requested: list[bool] = []
    widget.open_polarisation_requested.connect(lambda: requested.append(True))
    widget._polarisation_button.click()
    assert requested == [True]
    assert _breakdown_rows(widget)[4][:4] == ["Applied", "0", "0.01", "-0.25"]

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(widget.collect_state())
    assert restored._field_options().applied_field == (0.0, 0.01, -0.25)
    assert _breakdown_rows(restored)[4][:4] == ["Applied", "0", "0.01", "-0.25"]

    # Changing the field invalidates the shown results.
    restored._applied_field_inputs[2].setText("0.5")
    assert restored.collect_state()["magnetic_field_results"] is None
    assert not restored._polarisation_button.isEnabled()

    restored.set_form_state({"applied_field_T": ["a", 1.0, 2.0]})
    assert restored._field_options().applied_field == (0.0, 0.0, 0.0)
    restored.set_form_state({})
    assert restored.collect_state()["applied_field_T"] == [0.0, 0.0, 0.0]
    widget.close()
    restored.close()


def test_magnetic_field_widget_ignores_a_malformed_saved_magnetisation(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "bad-magnetisation.mufinder", "Bad Magnetisation")
    service.add_structure(document, _host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(
        {
            "demag_shape": "sphere",
            "use_measured_magnetization": True,
            "measured_magnetization": ["a", 1.0],
        }
    )

    assert widget._field_options().measured_magnetization == (0.0, 0.0, 0.0)
    widget.close()


def test_magnetic_field_widget_reports_an_unfinished_measured_magnetisation(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "unfinished-magnetisation.mufinder", "Unfinished Magnetisation"
    )
    service.add_structure(document, _host_structure())
    calls: list[object] = []
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_commensurate_fields",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.0, 0.0, 0.0))
    widget._measured_magnetization_check.setChecked(True)
    widget._measured_magnetization_inputs[0].setText("-")

    widget._calculate_fields()

    assert calls == []
    assert (
        "measured magnetisation components must be decimal numbers"
        in widget._calculation_status.text()
    )
    widget.close()


def test_magnetic_field_widget_empty_form_state_resets_previous_project_values(
    qapp, tmp_path
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-empty-state.mufinder", "Magnetic Empty State"
    )
    service.add_structure(document, _host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(
        {
            "demag_shape": "sphere",
            "supercell": [24, 36, 48],
            "include_equivalent_sites": True,
            "incommensurate_samples": 8192,
            "histogram_bins": 1024,
            "hyperfine_coupling": "2.5",
            "hyperfine_coupling_unit": "T_per_muB",
            "selected_muon_site": {
                "source": "custom",
                "label": "Previous site",
                "fractional_position": [0.25, 0.5, 0.75],
            },
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [
                    {
                        "site_label": "site 1",
                        "total_T": [0.1, 0.2, 0.3],
                        "magnitude_T": 0.374,
                    }
                ],
            },
            "dipolar_tensor_selector": {"mode": "species", "species": "Fe"},
            "dipolar_tensor_results": [{"site_label": "site 1"}],
            "use_commensurate_approximants": True,
            "approximant_tolerance": 0.004,
            "approximant_max_period": 20,
        }
    )

    widget.set_form_state({})

    assert widget.collect_state()["use_commensurate_approximants"] is False
    assert widget._field_options().approximant_tolerance == 1e-3
    assert widget._field_options().approximant_max_period == 12

    assert widget._demag_shape() == "none"
    assert widget._field_options().supercell == (100, 100, 100)
    assert widget._include_equivalent_sites.isChecked() is False
    assert widget._lookup_angles.value() == 32_768
    assert widget._phase_samples.value() == 200_000
    assert widget._field_options().phase_sampling == "auto"
    assert widget._histogram_bins.value() == 512
    assert widget._hyperfine_coupling.text() == "0.0"
    assert widget._current_hyperfine_unit() == "angstrom^-3"
    assert widget.selected_muon_site() is not None
    assert widget.selected_muon_site().label == "Custom site"
    assert widget._result_table.rowCount() == 0
    assert widget.collect_state()["magnetic_field_results"] is None
    assert widget.collect_state()["dipolar_tensor_selector"] == {"mode": "assigned"}
    assert widget.collect_state()["dipolar_tensor_results"] == []
    widget.close()


def test_magnetic_field_calculation_settings_dialog_collects_values(qapp) -> None:
    dialog = MagneticFieldCalculationSettingsDialog(
        supercell=(12, 24, 36),
        nangles=4096,
        phase_sample_count=200_000,
        phase_sampling="auto",
        histogram_bins=512,
    )

    dialog._calc_cells_a.setValue(18)
    dialog._lookup_angles.setValue(8192)
    dialog._phase_samples.setValue(250_000)
    dialog._phase_sampling.setCurrentIndex(dialog._phase_sampling.findData("montecarlo"))
    dialog._histogram_bins.setValue(1024)
    dialog._use_approximants.setChecked(True)
    dialog._approximant_tolerance.setValue(0.002)
    dialog._approximant_max_period.setValue(8)

    assert dialog.values() == {
        "supercell": (18, 24, 36),
        "nangles": 8192,
        "phase_sample_count": 250_000,
        "phase_sampling": "montecarlo",
        "histogram_bins": 1024,
        "use_approximants": True,
        "approximant_tolerance": 0.002,
        "approximant_max_period": 8,
    }
    assert (
        dialog._phase_samples.maximum()
        == magnetic_field_widget_module.MAX_INCOMMENSURATE_PHASE_SAMPLES
    )
    dialog.close()


def test_magnetic_field_widget_opens_calculation_settings_dialog(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-settings-dialog.mufinder", "Magnetic Settings Dialog"
    )
    service.add_structure(document, _host_structure())

    class _Dialog:
        def __init__(
            self,
            *,
            supercell,
            nangles,
            phase_sample_count,
            histogram_bins,
            phase_sampling,
            use_approximants,
            approximant_tolerance,
            approximant_max_period,
            parent=None,
        ):
            assert supercell == (100, 100, 100)
            assert nangles == 32_768
            assert phase_sample_count == 200_000
            assert histogram_bins == 512
            assert phase_sampling == "auto"
            assert use_approximants is False
            assert approximant_tolerance == 1e-3
            assert approximant_max_period == 12

        def exec(self):
            return QDialog.DialogCode.Accepted

        def values(self):
            return {
                "supercell": (24, 36, 48),
                "nangles": 8192,
                "phase_sample_count": 250_000,
                "phase_sampling": "grid",
                "histogram_bins": 1024,
                "use_approximants": True,
                "approximant_tolerance": 0.002,
                "approximant_max_period": 8,
            }

    monkeypatch.setattr(
        magnetic_field_widget_module, "MagneticFieldCalculationSettingsDialog", _Dialog
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    emitted: list[dict] = []
    widget.form_state_changed.connect(emitted.append)

    widget._open_calculation_settings_dialog()

    assert widget._field_options().supercell == (24, 36, 48)
    assert widget._field_options().nangles == 8192
    assert widget._field_options().phase_sample_count == 250_000
    assert widget._field_options().phase_sampling == "grid"
    assert widget._field_options().histogram_bins == 1024
    assert widget._field_options().approximant_tolerance == 0.002
    assert widget._field_options().approximant_max_period == 8
    assert "cells 24 x 36 x 48" in widget._calculation_settings_summary.text()
    assert "commensurate approximants within 0.002 r.l.u., period up to 8" in (
        widget._calculation_settings_summary.text()
    )
    assert emitted[-1]["supercell"] == [24, 36, 48]
    assert emitted[-1]["use_commensurate_approximants"] is True
    assert emitted[-1]["approximant_tolerance"] == 0.002
    assert emitted[-1]["approximant_max_period"] == 8
    widget.close()


def test_dipolar_tensor_dialog_defaults_to_assigned_magnetic_atoms(qapp) -> None:
    dialog = DipolarTensorDialog(
        _host_structure(),
        MagneticStructureModel(components=(_component("one", "One", 0),)),
        _custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=True,
    )

    assert dialog._assigned_radio.isChecked()
    assert dialog.selected_ion_indices() == (0,)
    assert dialog.selector_state()["mode"] == "assigned"
    assert dialog.selector_state()["include_equivalent_sites"] is True
    assert dialog._calculate_button.isEnabled()
    dialog.close()


def test_dipolar_tensor_dialog_requires_explicit_ions_without_assigned_atoms(qapp) -> None:
    dialog = DipolarTensorDialog(
        _host_structure(),
        MagneticStructureModel(),
        _custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=False,
    )

    assert not dialog._assigned_radio.isEnabled()
    assert not dialog._calculate_button.isEnabled()

    dialog._species_radio.setChecked(True)

    assert dialog.selected_ion_indices() == (0,)
    assert dialog.selector_state()["selector"] == AtomSelector.species("Fe").to_json()
    assert dialog._calculate_button.isEnabled()
    dialog.close()


def test_dipolar_tensor_dialog_atom_indices_count_from_one(qapp) -> None:
    dialog = DipolarTensorDialog(
        _host_structure(),
        MagneticStructureModel(),
        _custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=False,
    )

    dialog._indices_radio.setChecked(True)
    dialog._indices_edit.setText("2")

    # The box counts from 1; the stored selector keeps zero-based indices.
    assert dialog.selected_ion_indices() == (1,)
    assert dialog.selector_state()["selector"] == AtomSelector.indices([1]).to_json()
    assert dialog._calculate_button.isEnabled()

    dialog._indices_edit.setText("0")
    assert not dialog._calculate_button.isEnabled()
    # Closing on invalid input keeps the selection the dialog opened with.
    assert dialog.selector_state()["selector"] == AtomSelector.indices([]).to_json()
    dialog.close()

    restored = DipolarTensorDialog(
        _host_structure(),
        MagneticStructureModel(),
        _custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=False,
        selector_state={"mode": "selector", "selector": AtomSelector.indices([0]).to_json()},
    )
    assert restored._indices_radio.isChecked()
    assert restored._indices_edit.text() == "1"
    assert restored.selected_ion_indices() == (0,)
    restored._indices_edit.setText("x")
    assert restored.selector_state()["selector"] == AtomSelector.indices([0]).to_json()
    restored.close()


def test_dipolar_tensor_dialog_shows_and_updates_recent_results(qapp) -> None:
    class _Result:
        site_label = "site 2"
        muon_position = (0.5, 0.5, 0.5)
        tensor_T_per_muB = ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))

        def to_json(self):
            return {
                "site_label": self.site_label,
                "muon_position": list(self.muon_position),
                "dipolar_tensor_T_per_muB": np.asarray(self.tensor_T_per_muB, dtype=float).tolist(),
            }

    calls: list[tuple[int, ...]] = []

    def _calculate(dialog):
        calls.append(dialog.selected_ion_indices())
        return (_Result(),)

    dialog = DipolarTensorDialog(
        _host_structure(),
        MagneticStructureModel(components=(_component("one", "One", 0),)),
        _custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=False,
        supercell=(12, 24, 36),
        results_state=[
            {
                "site_label": "site 1",
                "muon_position": [0.25, 0.25, 0.25],
                "dipolar_tensor_T_per_muB": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }
        ],
        calculate_callback=_calculate,
    )

    assert "12 x 24 x 36" in dialog._settings_summary.text()
    assert dialog._result_tables[0].item(0, 0).text() == "1"

    dialog._calculate()

    assert calls == [(0,)]
    assert dialog._result_tables[0].item(0, 0).text() == "2"
    assert dialog.results_state()[0]["site_label"] == "site 2"
    dialog.close()


def test_dipolar_tensor_results_dialog_displays_matrix(qapp) -> None:
    class _Result:
        site_label = "site 1"
        muon_position = (0.25, 0.25, 0.25)
        tensor_T_per_muB = ((1.0, 0.1, 0.2), (0.1, 2.0, 0.3), (0.2, 0.3, 3.0))

    dialog = DipolarTensorResultsDialog((_Result(),))

    assert len(dialog._matrix_tables) == 1
    assert dialog._matrix_tables[0].item(1, 1).text() == "2"

    assert [dialog._unit_combo.itemData(index) for index in range(dialog._unit_combo.count())] == [
        "angstrom^-3",
        "T_per_muB",
        "MHz_per_muB",
    ]
    assert dialog._unit_combo.currentData() == "T_per_muB"
    dialog._unit_combo.setCurrentIndex(dialog._unit_combo.findData("MHz_per_muB"))

    assert float(dialog._matrix_tables[0].item(1, 1).text()) == pytest.approx(
        2.0 * magnetic_field_widget_module.MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    )
    dialog.close()


def test_dipolar_tensor_results_dialog_converts_to_angstrom_units(qapp) -> None:
    class _Result:
        site_label = "site 1"
        muon_position = (0.25, 0.25, 0.25)
        tensor_T_per_muB = ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    dialog = DipolarTensorResultsDialog((_Result(),))
    dialog._unit_combo.setCurrentIndex(dialog._unit_combo.findData("angstrom^-3"))

    assert float(dialog._matrix_tables[0].item(0, 0).text()) == pytest.approx(
        1.0 / magnetic_field_widget_module.CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB
    )
    dialog.close()


def test_magnetic_field_widget_opens_dipolar_tensor_dialog(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-tensor-dialog.mufinder", "Magnetic Tensor Dialog"
    )
    service.add_structure(document, _host_structure())

    opened: list[object] = []
    calculated: list[dict[str, object]] = []

    class _Dialog:
        def __init__(
            self,
            structure,
            model,
            muon_site,
            *,
            include_equivalent_sites,
            selector_state,
            supercell,
            results_state,
            calculate_callback,
            parent=None,
        ):
            self._calculate_callback = calculate_callback
            self._results_state = list(results_state)
            opened.append(
                {
                    "structure": structure,
                    "model": model,
                    "muon_site": muon_site,
                    "include_equivalent_sites": include_equivalent_sites,
                    "selector_state": selector_state,
                    "supercell": supercell,
                    "results_state": results_state,
                }
            )

        def exec(self):
            self._results_state = [
                {
                    "site_label": result.site_label,
                    "muon_position": list(result.muon_position),
                    "dipolar_tensor_T_per_muB": np.asarray(
                        result.tensor_T_per_muB, dtype=float
                    ).tolist(),
                }
                for result in self._calculate_callback(self)
            ]
            return QDialog.DialogCode.Rejected

        def selector_state(self):
            return {
                "mode": "selector",
                "selector": AtomSelector.indices([0]).to_json(),
                "include_equivalent_sites": True,
            }

        def selected_ion_indices(self):
            return (0,)

        def include_equivalent_sites(self):
            return True

        def results_state(self):
            return self._results_state

    class _Result:
        site_label = "site 1"
        muon_position = (0.25, 0.25, 0.25)
        tensor_T_per_muB = np.eye(3)

    def _calculate(structure, muon_sites, *, magnetic_ion_indices, options):
        calculated.append(
            {
                "structure": structure,
                "muon_sites": tuple(muon_sites),
                "magnetic_ion_indices": tuple(magnetic_ion_indices),
                "options": options,
            }
        )
        return (_Result(),)

    monkeypatch.setattr(magnetic_field_widget_module, "DipolarTensorDialog", _Dialog)
    monkeypatch.setattr(
        magnetic_field_widget_module, "calculate_dipolar_tensors_for_atoms", _calculate
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.25, 0.25, 0.25))
    widget._include_equivalent_sites.setChecked(False)

    widget._open_dipolar_tensor_dialog()

    assert opened
    assert opened[0]["include_equivalent_sites"] is False
    assert opened[0]["supercell"] == (100, 100, 100)
    assert calculated[0]["muon_sites"] == ((0.25, 0.25, 0.25),)
    assert calculated[0]["magnetic_ion_indices"] == (0,)
    assert calculated[0]["options"].find_equivalent_sites is True
    assert calculated[0]["options"].demag_shape == "none"
    assert calculated[0]["options"].contact_coupling is None
    assert (
        widget.collect_state()["dipolar_tensor_selector"]["selector"]
        == AtomSelector.indices([0]).to_json()
    )
    assert widget.collect_state()["dipolar_tensor_results"][0]["site_label"] == "site 1"
    widget.close()


def test_magnetic_field_widget_passes_incommensurate_sampling_controls(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-sampling.mufinder", "Magnetic Sampling")
    service.add_structure(document, _host_structure())

    calls: list[dict[str, object]] = []

    def _calculate(structure, model, muon_sites, *, options, distorted_structures=None):
        calls.append(
            {"structure": structure, "model": model, "muon_sites": muon_sites, "options": options}
        )
        return ()

    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "incommensurate",
    )
    monkeypatch.setattr(
        magnetic_field_widget_module, "calculate_incommensurate_field_distributions", _calculate
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.0, 0.0, 0.0))
    widget._phase_samples.setValue(250_000)
    widget._phase_sampling_method = "montecarlo"
    widget._histogram_bins.setValue(1024)

    widget._calculate_fields()

    assert calls
    assert calls[0]["options"].nangles == 32_768
    assert calls[0]["options"].phase_sample_count == 250_000
    assert calls[0]["options"].phase_sampling == "montecarlo"
    assert calls[0]["options"].histogram_bins == 1024
    widget.close()


def test_magnetic_field_widget_converts_hyperfine_coupling_units(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-hyperfine.mufinder", "Magnetic Hyperfine"
    )
    service.add_structure(document, _host_structure())

    calls: list[dict[str, object]] = []

    def _calculate(structure, model, muon_sites, *, options, distorted_structures=None):
        calls.append(
            {"structure": structure, "model": model, "muon_sites": muon_sites, "options": options}
        )
        return ()

    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(magnetic_field_widget_module, "calculate_commensurate_fields", _calculate)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.0, 0.0, 0.0))
    widget._hyperfine_coupling.setText("12.5")
    widget._hyperfine_units.setCurrentIndex(widget._hyperfine_units.findData("T_per_muB"))

    widget._calculate_fields()

    assert calls
    assert calls[0]["options"].contact_coupling == pytest.approx(
        magnetic_field_widget_module.contact_coupling_to_angstrom_minus3(12.5, "T_per_muB")
    )
    assert [
        widget._hyperfine_units.itemData(index) for index in range(widget._hyperfine_units.count())
    ] == [
        "angstrom^-3",
        "T_per_muB",
        "MHz_per_muB",
    ]
    widget.close()


def test_magnetic_field_widget_passes_equivalent_site_option(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-equivalent-sites.mufinder", "Magnetic Equivalent Sites"
    )
    service.add_structure(document, _host_structure())

    calls: list[dict[str, object]] = []

    def _calculate(structure, model, muon_sites, *, options, distorted_structures=None):
        calls.append(
            {"structure": structure, "model": model, "muon_sites": muon_sites, "options": options}
        )
        return ()

    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(magnetic_field_widget_module, "calculate_commensurate_fields", _calculate)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.25, 0.25, 0.25))
    widget._include_equivalent_sites.setChecked(True)

    widget._calculate_fields()

    assert calls
    assert calls[0]["options"].find_equivalent_sites is True
    widget.close()


def test_magnetic_field_widget_collects_and_restores_form_state(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-form-state.mufinder", "Magnetic Form State"
    )
    service.add_structure(document, _host_structure())

    distortion_id = uuid4()
    selected_site = magnetic_field_widget_module.MagneticMuonSiteOption(
        source="relaxation_cluster",
        label="Relaxed site",
        fractional_position=(0.125, 0.25, 0.375),
        cartesian_position=(0.375, 0.75, 1.125),
        cluster_label="cluster 2",
        source_label="Pristine relaxation",
        distortion_structure_id=distortion_id,
        properties={"energy_eV": -12.5},
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._selected_muon_site = selected_site
    widget._plate_radio.setChecked(True)
    widget._calc_cells_a.setValue(24)
    widget._calc_cells_b.setValue(36)
    widget._calc_cells_c.setValue(48)
    widget._include_equivalent_sites.setChecked(True)
    widget._phase_samples.setValue(250_000)
    widget._phase_sampling_method = "grid"
    widget._histogram_bins.setValue(1024)
    widget._hyperfine_coupling.setText("2.5")
    widget._hyperfine_units.setCurrentIndex(widget._hyperfine_units.findData("MHz_per_muB"))
    widget._dipolar_tensor_selector_state = {
        "mode": "selector",
        "selector": AtomSelector.species("Fe").to_json(),
        "include_equivalent_sites": True,
    }
    widget._dipolar_tensor_results_state = [
        {
            "site_label": "site 1",
            "muon_position": [0.125, 0.25, 0.375],
            "dipolar_tensor_T_per_muB": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        }
    ]
    widget._field_results_state = {
        "kind": "commensurate",
        "results": [
            {
                "site_label": "site 1",
                "muon_position": [0.125, 0.25, 0.375],
                "total_T": [0.1, 0.2, 0.3],
                "magnitude_T": 0.37416573867739417,
            }
        ],
    }
    widget._nuclear_relaxation_options_state = {
        "mode": "zf_single_crystal",
        "direction": [0.0, 1.0, 0.0],
        "include_quadrupolar": True,
        "radius": 12.0,
        "convergence_rtol": 1e-5,
        "use_distortion_correction": True,
    }
    widget._nuclear_relaxation_results_state = {
        "site_label": "Relaxed site",
        "sigma": 0.25,
        "sigma_squared": 0.0625,
        "mode": "zf_single_crystal",
    }

    state = widget.collect_state()

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(state)

    assert restored._demag_shape() == "infinite_plate"
    assert restored._field_options().supercell == (24, 36, 48)
    assert restored._field_options().find_equivalent_sites is True
    assert restored._field_options().nangles == 32_768
    assert restored._field_options().phase_sample_count == 250_000
    assert restored._field_options().phase_sampling == "grid"
    assert restored._field_options().histogram_bins == 1024
    assert restored.collect_state()["hyperfine_coupling"] == "2.5"
    assert restored.collect_state()["hyperfine_coupling_unit"] == "MHz_per_muB"
    assert restored.collect_state()["dipolar_tensor_selector"] == {
        "mode": "selector",
        "selector": AtomSelector.species("Fe").to_json(),
        "include_equivalent_sites": True,
    }
    assert restored.collect_state()["dipolar_tensor_results"][0]["site_label"] == "site 1"
    assert restored.collect_state()["nuclear_relaxation_options"]["mode"] == "zf_single_crystal"
    assert restored.collect_state()["nuclear_relaxation_results"]["sigma"] == pytest.approx(0.25)
    assert restored.collect_state()["magnetic_field_results"]["kind"] == "commensurate"
    assert restored._result_table.rowCount() == 1
    assert restored._result_table.item(0, 1).text() == "0.1"
    assert restored._field_options().contact_coupling == pytest.approx(
        magnetic_field_widget_module.contact_coupling_to_angstrom_minus3(2.5, "MHz_per_muB")
    )
    assert restored.selected_muon_site() == selected_site
    widget.close()
    restored.close()


def test_magnetic_field_widget_restores_distribution_results(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-restore-dist.mufinder", "Magnetic Restore Dist"
    )
    service.add_structure(document, _host_structure())

    state = {
        "magnetic_field_results": {
            "kind": "distribution",
            "results": [
                {
                    "site_label": "site 1",
                    "muon_position": [0.0, 0.0, 0.0],
                    "fields_T": [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]],
                    "histogram_centres_T": [1.0, 3.0],
                    "histogram_counts": [1.0, 1.0],
                    "calculation_parameters": {"nangles": 4096},
                }
            ],
        }
    }

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(state)

    assert widget._result_table.rowCount() == 1
    assert widget._result_table.item(0, 5).text() == "distribution mean"
    assert widget._open_distribution_button.isEnabled()
    assert widget._selected_distribution_results()[0].site_label == "site 1"
    widget.close()


def test_magnetic_field_widget_scrolls_a_long_component_summary(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-long-summary.mufinder", "Magnetic Long Summary"
    )
    service.add_structure(document, _host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)
    summary = widget._component_summary_table

    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    short_height = summary.maximumHeight()
    assert summary.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded

    limit = MagneticFieldWidget._SUMMARY_MAX_VISIBLE_ROWS
    components = tuple(
        _component(f"c{index}", f"C{index}", index % 2) for index in range(limit + 4)
    )
    widget._set_model(MagneticStructureModel(components=components))
    assert summary.rowCount() == limit + 4
    capped_height = summary.maximumHeight()
    assert short_height < capped_height
    assert capped_height < summary.horizontalHeader().sizeHint().height() + sum(
        summary.rowHeight(row) for row in range(summary.rowCount())
    )
    widget.close()


def test_magnetic_field_widget_places_demagnetizing_controls_in_the_calculation_group(qapp) -> None:
    widget = MagneticFieldWidget()

    for control in (
        widget._sphere_radio,
        widget._plate_radio,
        widget._none_radio,
        widget._advanced_demag_radio,
        widget._advanced_demag_button,
        widget._advanced_demag_summary,
    ):
        assert control.parentWidget().title() == "Calculation"
    assert widget._advanced_demag_summary.isHidden()
    widget._advanced_demag_radio.setChecked(True)
    assert not widget._advanced_demag_summary.isHidden()
    widget._sphere_radio.setChecked(True)
    assert widget._advanced_demag_summary.isHidden()
    widget.close()


def _breakdown_rows(widget: MagneticFieldWidget) -> list[list[str]]:
    table = widget._breakdown_table
    return [
        [table.item(row, column).text() for column in range(table.columnCount())]
        for row in range(table.rowCount())
    ]


def test_magnetic_field_widget_breaks_a_commensurate_field_into_its_terms(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-breakdown.mufinder", "Magnetic Breakdown"
    )
    service.add_structure(document, _host_structure())
    result = magnetic_fields_module.MagneticFieldResult(
        site_label="muon 1",
        muon_position=(0.25, 0.5, 0.75),
        dipolar=(0.1, 0.0, 0.0),
        lorentz=(0.0, 0.2, 0.0),
        contact=(0.0, 0.0, 0.3),
        demag=(0.0, -0.4, 0.0),
        applied=(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_commensurate_fields",
        lambda *_args, **_kwargs: (result,),
    )
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.25, 0.5, 0.75))
    assert widget._breakdown_table.isHidden()

    widget._calculate_fields()

    expected = [
        ["Dipolar", "0.1", "0", "0", "0.1"],
        ["Lorentz", "0", "0.2", "0", "0.2"],
        ["Contact", "0", "0", "0.3", "0.3"],
        ["Demagnetizing", "0", "-0.4", "0", "0.4"],
        ["Applied", "0", "0", "0", "0"],
        ["Total", "0.1", "-0.2", "0.3", f"{np.linalg.norm([0.1, -0.2, 0.3]):.6g}"],
    ]
    assert not widget._breakdown_table.isHidden()
    assert _breakdown_rows(widget) == expected
    assert widget._breakdown_hint.isHidden()

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(widget.collect_state())
    assert _breakdown_rows(restored) == expected
    restored.close()

    widget._clear_results(clear_state=True)
    assert widget._breakdown_table.isHidden()
    assert widget._breakdown_table.rowCount() == 0
    widget.close()


def test_magnetic_field_widget_shows_numerically_zero_field_components_as_zero(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-zero-display.mufinder", "Magnetic Zero Display"
    )
    service.add_structure(document, _host_structure())
    result = magnetic_fields_module.MagneticFieldResult(
        site_label="muon 1",
        muon_position=(0.5, 0.5, 0.0),
        dipolar=(1.28162e-18, -5.32134e-18, -0.297712),
        lorentz=(0.0, 0.0, 0.143831),
        contact=(0.0, 0.0, 0.0),
        demag=(-0.0, -0.0, 0.0),
        applied=(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_commensurate_fields",
        lambda *_args, **_kwargs: (result,),
    )
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.5, 0.5, 0.0))

    widget._calculate_fields()

    expected_result_row = ["muon 1", "0", "0", "-0.153881", "0.153881", "field"]
    assert [
        widget._result_table.item(0, column).text() for column in range(6)
    ] == expected_result_row
    expected_breakdown = [
        ["Dipolar", "0", "0", "-0.297712", "0.297712"],
        ["Lorentz", "0", "0", "0.143831", "0.143831"],
        ["Contact", "0", "0", "0", "0"],
        ["Demagnetizing", "0", "0", "0", "0"],
        ["Applied", "0", "0", "0", "0"],
        ["Total", "0", "0", "-0.153881", "0.153881"],
    ]
    assert _breakdown_rows(widget) == expected_breakdown

    stored = widget.collect_state()["magnetic_field_results"]["results"][0]
    assert stored["total_T"][:2] == [1.28162e-18, -5.32134e-18]
    assert stored["dipolar_T"][:2] == [1.28162e-18, -5.32134e-18]

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(widget.collect_state())
    assert [
        restored._result_table.item(0, column).text() for column in range(6)
    ] == expected_result_row
    assert _breakdown_rows(restored) == expected_breakdown
    restored.close()
    widget.close()


def test_magnetic_field_widget_explains_results_without_a_term_breakdown(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-no-breakdown.mufinder", "Magnetic No Breakdown"
    )
    service.add_structure(document, _host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)

    widget.set_form_state(
        {
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [
                    {"site_label": "site 1", "total_T": [0.0, 0.0, 1.0], "magnitude_T": 1.0}
                ],
            }
        }
    )
    assert widget._breakdown_table.isHidden()
    assert (
        widget._breakdown_hint.text()
        == "This saved result has no term breakdown; recalculate to see it."
    )

    widget.set_form_state(
        {
            "magnetic_field_results": {
                "kind": "distribution",
                "results": [
                    {
                        "site_label": "site 1",
                        "muon_position": [0.0, 0.0, 0.0],
                        "fields_T": [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]],
                        "histogram_centres_T": [1.0, 3.0],
                        "histogram_counts": [1.0, 1.0],
                    }
                ],
            }
        }
    )
    assert widget._breakdown_table.isHidden()
    assert (
        widget._breakdown_hint.text()
        == "This saved distribution has no term breakdown; recalculate to see it."
    )
    widget.close()


def test_magnetic_field_widget_emits_form_state_for_setting_changes(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-form-emits.mufinder", "Magnetic Form Emits"
    )
    service.add_structure(document, _host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    emitted: list[dict] = []
    widget.form_state_changed.connect(emitted.append)

    widget._none_radio.setChecked(True)
    widget._histogram_bins.setValue(768)
    widget._hyperfine_coupling.setText("1.25")

    assert emitted
    assert emitted[-1]["demag_shape"] == "none"
    assert emitted[-1]["histogram_bins"] == 768
    assert emitted[-1]["hyperfine_coupling"] == "1.25"
    widget.close()


def test_magnetic_field_widget_emits_pin_title_and_delete_signals(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-workspace-signals.mufinder", "Magnetic Workspace"
    )
    service.add_structure(document, _host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document, title="Pinned helix", title_editable=True, deletable=True)
    pinned: list[dict] = []
    renamed: list[str] = []
    deleted: list[bool] = []
    widget.pin_requested.connect(pinned.append)
    widget.workspace_title_changed.connect(renamed.append)
    widget.delete_requested.connect(lambda: deleted.append(True))

    widget._phase_samples.setValue(250_000)
    widget._pin_button.click()
    widget._title_edit.setText("Renamed helix")
    widget._on_title_edit_finished()
    widget._delete_workspace_button.click()

    assert not widget._title_edit.isHidden()
    assert not widget._delete_workspace_button.isHidden()
    assert pinned[0]["title"] == "Pinned helix"
    assert pinned[0]["form"]["incommensurate_samples"] == 250_000
    assert renamed == ["Renamed helix"]
    assert deleted == [True]
    widget.close()


def test_magnetic_field_widget_calculates_incommensurate_distribution(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-dist.mufinder", "Magnetic Dist")
    service.add_structure(document, _host_structure())

    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)

    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "incommensurate",
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: (_Distribution(),),
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.0, 0.0, 0.0))

    widget._calculate_fields()

    assert widget._result_table.rowCount() == 1
    assert widget._result_table.item(0, 3).text() == "2"
    assert widget._result_table.item(0, 4).text() == "2"
    assert widget._result_table.item(0, 5).text() == "distribution mean"
    assert widget._open_distribution_button.isEnabled()
    assert widget.collect_state()["magnetic_field_results"]["kind"] == "distribution"
    assert widget.collect_state()["magnetic_field_results"]["results"][0]["fields_T"] == [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 3.0],
    ]
    widget.close()


def _breakdown_headers(widget: MagneticFieldWidget) -> list[str]:
    table = widget._breakdown_table
    return [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]


def test_magnetic_field_widget_breaks_a_distribution_into_term_means_and_rms(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-dist-breakdown.mufinder", "Magnetic Dist Breakdown"
    )
    service.add_structure(document, _host_structure())
    stats = magnetic_fields_module.FieldTermStatistics
    distribution = magnetic_fields_module.MagneticFieldDistributionResult(
        site_label="site 1",
        muon_position=(0.0, 0.0, 0.0),
        fields_T=np.asarray([[0.2, 0.0, 0.1], [-0.2, 0.0, 0.1]], dtype=float),
        histogram_centres_T=np.asarray([0.2236], dtype=float),
        histogram_counts=np.asarray([2.0], dtype=float),
        convergence=(
            magnetic_fields_module.DistributionConvergenceCheck(
                setting="histogram_bins",
                label="Histogram bins",
                converged=True,
                metric=2.0,
                message="2 samples per occupied histogram bin.",
            ),
        ),
        terms={
            "dipolar": stats(mean_T=(1e-18, 0.0, 0.0), rms_T=0.2),
            "lorentz": stats(mean_T=(0.0, 0.0, 0.0), rms_T=0.0),
            "contact": stats(mean_T=(0.0, 0.0, 0.05), rms_T=0.05),
            "demag": stats(mean_T=(0.0, 0.0, 0.0), rms_T=0.0),
            "applied": stats(mean_T=(0.0, 0.0, 0.05), rms_T=0.05),
        },
        total_rms_T=float(np.hypot(0.2, 0.1)),
        applied_T=(0.0, 0.0, 0.05),
        demag_T=(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "incommensurate",
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: (distribution,),
    )
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.0, 0.0, 0.0))

    widget._calculate_fields()

    expected = [
        ["Dipolar", "0", "0", "0", "0.2"],
        ["Lorentz", "0", "0", "0", "0"],
        ["Contact", "0", "0", "0.05", "0.05"],
        ["Demagnetizing", "0", "0", "0", "0"],
        ["Applied", "0", "0", "0.05", "0.05"],
        ["Total", "0", "0", "0.1", "0.223607"],
    ]
    assert not widget._breakdown_table.isHidden()
    assert _breakdown_headers(widget) == [
        "Term",
        "Mean Bx (T)",
        "Mean By (T)",
        "Mean Bz (T)",
        "RMS |B| (T)",
    ]
    assert _breakdown_rows(widget) == expected
    assert widget._breakdown_hint.text().splitlines() == [
        "Convergence checks:",
        "Histogram bins: OK. 2 samples per occupied histogram bin.",
    ]

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(widget.collect_state())
    assert _breakdown_rows(restored) == expected
    restored.close()

    widget.set_form_state(
        {
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [
                    {
                        "site_label": "site 1",
                        "total_T": [0.0, 0.0, 1.0],
                        "dipolar_T": [0.0, 0.0, 1.0],
                    }
                ],
            }
        }
    )
    assert _breakdown_headers(widget)[1] == "Bx (T)"

    corrupt = widget.collect_state()
    corrupt["magnetic_field_results"] = restored_state = {
        "kind": "distribution",
        "results": [dict(distribution.to_json())],
    }
    restored_state["results"][0]["terms"]["dipolar"] = {"mean_T": [0.0, 0.0, 0.0], "rms_T": "n/a"}
    widget.set_form_state(corrupt)
    widget._result_table.selectRow(0)
    assert widget._breakdown_table.isHidden()
    assert widget._breakdown_hint.text().endswith(
        "This saved distribution has no term breakdown; recalculate to see it."
    )
    widget.close()


_APPROXIMANT = magnetic_fields_module.CommensurateApproximant(
    q_rlu=((0.3334, 0.0, 0.0),),
    approximant_rlu=((1.0 / 3.0, 0.0, 0.0),),
    denominators=((3, 1, 1),),
    periods=(3, 1, 1),
    max_deviation_rlu=6.7e-5,
)


def _approximant_field(label: str, bz: float) -> magnetic_fields_module.MagneticFieldResult:
    zero = (0.0, 0.0, 0.0)
    return magnetic_fields_module.MagneticFieldResult(
        site_label=label,
        muon_position=zero,
        dipolar=(0.0, 0.0, bz),
        lorentz=zero,
        contact=zero,
        demag=zero,
        applied=zero,
    )


def _incommensurate_widget(document, monkeypatch, *, approximant) -> MagneticFieldWidget:
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "incommensurate",
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "find_commensurate_approximant",
        lambda *_args, **_kwargs: approximant,
    )
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.0, 0.0, 0.0))
    return widget


def test_magnetic_field_widget_lists_commensurate_approximant_fields(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-approximant.mufinder", "Magnetic Approximant"
    )
    service.add_structure(document, _host_structure())
    calls: list[object] = []

    def _approximant_fields(*_args, options, distorted_structures=None):
        calls.append(options)
        return (
            _approximant_field("site 1 + (0, 0, 0)", 1.0),
            _approximant_field("site 1 + (1, 0, 0)", -0.5),
        )

    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_commensurate_approximant_fields",
        _approximant_fields,
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: pytest.fail("an approximant should not sample a distribution"),
    )
    widget = _incommensurate_widget(document, monkeypatch, approximant=_APPROXIMANT)
    widget._use_approximants = True
    widget._approximant_tolerance = 2e-3
    widget._approximant_max_period = 8

    widget._calculate_fields()

    assert calls[0].approximant_tolerance == 2e-3
    assert calls[0].approximant_max_period == 8
    assert [widget._result_table.item(row, 0).text() for row in range(2)] == [
        "site 1 + (0, 0, 0)",
        "site 1 + (1, 0, 0)",
    ]
    assert widget._result_table.item(1, 5).text() == "field"
    assert widget._calculation_status.text() == (
        "Results: calculated 2 field value(s) of the commensurate approximant q = (1/3, 0, 0) with a "
        "3 x 1 x 1 magnetic supercell (q moved by at most 6.7e-05 r.l.u.)."
    )
    state = widget.collect_state()["magnetic_field_results"]
    assert state["kind"] == "commensurate"
    assert state["results"][1]["dipolar_T"] == [0.0, 0.0, -0.5]
    assert not widget._breakdown_table.isHidden()
    widget.close()


def test_magnetic_field_widget_suggests_an_available_approximant(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-suggest.mufinder", "Magnetic Suggest")
    service.add_structure(document, _host_structure())
    flagged = magnetic_fields_module.DistributionConvergenceCheck(
        setting="nangles",
        label="Lookup angles per q",
        converged=False,
        metric=0.2,
        message="Halving the lookup angles to 8 shifts the |B| distribution by 20%.",
    )

    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        convergence = (flagged,)

    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_commensurate_approximant_fields",
        lambda *_args, **_kwargs: pytest.fail("approximants are turned off"),
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: (_Distribution(),),
    )
    widget = _incommensurate_widget(document, monkeypatch, approximant=_APPROXIMANT)

    widget._calculate_fields()

    status = widget._calculation_status.text()
    assert status.startswith(
        "Results: sampled 1 incommensurate field distribution(s); "
        "1 convergence check(s) need attention, select a row for advice."
    )
    assert "within 6.7e-05 r.l.u. of the commensurate approximant q = (1/3, 0, 0)" in status
    assert status.endswith(
        "turn on commensurate approximants in Calculation Settings to list its finite fields."
    )
    widget.close()


def test_magnetic_field_widget_reports_a_converged_distribution_without_advice(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-converged.mufinder", "Magnetic Converged"
    )
    service.add_structure(document, _host_structure())

    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        convergence = ()

    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: (_Distribution(),),
    )
    widget = _incommensurate_widget(document, monkeypatch, approximant=None)
    widget._use_approximants = True

    widget._calculate_fields()

    assert (
        widget._calculation_status.text()
        == "Results: sampled 1 incommensurate field distribution(s)."
    )
    widget.close()


def test_magnetic_field_widget_shows_distribution_convergence_checks(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-convergence.mufinder", "Magnetic Convergence"
    )
    service.add_structure(document, _host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)

    widget.set_form_state(
        {
            "magnetic_field_results": {
                "kind": "distribution",
                "results": [
                    {
                        "site_label": "site 1",
                        "fields_T": [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]],
                        "convergence": [
                            {
                                "setting": "nangles",
                                "label": "Lookup angles per q",
                                "converged": True,
                                "message": "Halving the lookup angles to 16384 shifts the |B| distribution by 0.05%.",
                            },
                            {
                                "setting": "histogram_bins",
                                "label": "Histogram bins",
                                "converged": False,
                                "message": "8 samples per occupied histogram bin (fewer than 20).",
                            },
                            "not a check",
                        ],
                    }
                ],
            }
        }
    )

    hint = widget._breakdown_hint.text().splitlines()
    assert hint[:3] == [
        "Convergence checks:",
        "Lookup angles per q: OK. Halving the lookup angles to 16384 shifts the |B| distribution by 0.05%.",
        "Histogram bins: Needs attention. 8 samples per occupied histogram bin (fewer than 20).",
    ]
    assert hint[3] == "This saved distribution has no term breakdown; recalculate to see it."
    widget.close()


def test_magnetic_field_widget_saves_and_restores_approximant_settings(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-approximant-state.mufinder", "Approximant State"
    )
    service.add_structure(document, _host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)

    widget.set_form_state(
        {
            "use_commensurate_approximants": True,
            "approximant_tolerance": 0.004,
            "approximant_max_period": 20,
        }
    )

    state = widget.collect_state()
    assert state["use_commensurate_approximants"] is True
    assert state["approximant_tolerance"] == 0.004
    assert state["approximant_max_period"] == 20
    assert "commensurate approximants within 0.004 r.l.u., period up to 20." in (
        widget._calculation_settings_summary.text()
    )

    widget.set_form_state({"approximant_tolerance": "wide", "approximant_max_period": 500})
    assert widget._use_approximants is False
    assert widget._field_options().approximant_tolerance == 1e-3
    assert widget._field_options().approximant_max_period == 12

    widget.set_form_state({"approximant_tolerance": -1.0, "approximant_max_period": "long"})
    assert widget._field_options().approximant_tolerance == 1e-3
    assert widget._field_options().approximant_max_period == 12
    assert widget._calculation_settings_summary.text().endswith("512 histogram bins.")
    widget.close()


def test_magnetic_field_widget_opens_incommensurate_distribution_plot(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-dist-plot.mufinder", "Magnetic Dist Plot"
    )
    service.add_structure(document, _host_structure())

    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        histogram_centres_T = np.asarray([1.0, 3.0], dtype=float)
        histogram_counts = np.asarray([1, 1], dtype=float)

    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "incommensurate",
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: (_Distribution(),),
    )

    opened: list[object] = []

    class _Dialog:
        def __init__(self, results, parent=None, *, estimate=False):
            opened.append(tuple(results))

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(magnetic_field_widget_module, "MagneticFieldDistributionDialog", _Dialog)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.0, 0.0, 0.0))

    widget._calculate_fields()
    widget._open_distribution_plot()

    assert opened == [(widget._distribution_results[0],)]
    widget.close()


def test_magnetic_field_widget_opens_selected_distribution_plots_together(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-multi-dist-plot.mufinder", "Magnetic Multi Dist Plot"
    )
    service.add_structure(document, _host_structure())

    class _Distribution:
        def __init__(self, label: str, z_values: list[float]) -> None:
            self.site_label = label
            self.fields_T = np.asarray([[0.0, 0.0, value] for value in z_values], dtype=float)
            self.histogram_centres_T = np.asarray(z_values, dtype=float)
            self.histogram_counts = np.ones(len(z_values), dtype=float)

    distributions = (
        _Distribution("site 1", [1.0, 2.0]),
        _Distribution("site 2", [1.5, 2.5]),
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "incommensurate",
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: distributions,
    )

    opened: list[tuple[object, ...]] = []

    class _Dialog:
        def __init__(self, results, parent=None, *, estimate=False):
            opened.append(tuple(results))

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(magnetic_field_widget_module, "MagneticFieldDistributionDialog", _Dialog)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.0, 0.0, 0.0))
    widget._calculate_fields()
    widget._result_table.selectRow(0)
    widget._result_table.selectionModel().select(
        widget._result_table.model().index(1, 0),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )

    widget._open_distribution_plot()

    assert widget._result_table.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert opened == [distributions]
    widget.close()


def test_magnetic_field_widget_plots_a_restored_distribution(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-restored-plot.mufinder", "Magnetic Restored Plot"
    )
    service.add_structure(document, _host_structure())
    opened: list[tuple[object, ...]] = []

    class _Dialog:
        def __init__(self, results, parent=None, *, estimate=False):
            opened.append(tuple(results))

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(magnetic_field_widget_module, "MagneticFieldDistributionDialog", _Dialog)
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(
        {
            "magnetic_field_results": {
                "kind": "distribution",
                "results": [
                    {
                        "site_label": "site 1",
                        "muon_position": [0.0, 0.0, 0.0],
                        "fields_T": [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]],
                        "histogram_centres_T": [1.0, 3.0],
                        "histogram_counts": [4.0, 2.0],
                    },
                    {"site_label": "malformed", "fields_T": [[1.0, 2.0]]},
                ],
            }
        }
    )
    widget._result_table.selectRow(0)

    widget._open_distribution_plot()

    (restored,) = opened[0]
    monkeypatch.undo()
    series = MagneticFieldDistributionDialog(restored)._distribution_series()
    assert [label for *_values, label in series] == ["site 1"]
    np.testing.assert_allclose(series[0][0], [1.0, 3.0])
    np.testing.assert_allclose(series[0][2], [1.0, 3.0])
    assert widget._result_table.rowCount() == 2
    widget.close()


def test_magnetic_field_distribution_dialog_plots_histogram(qapp) -> None:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        histogram_centres_T = np.asarray([1.0, 3.0], dtype=float)
        histogram_counts = np.asarray([4, 2], dtype=float)

    dialog = MagneticFieldDistributionDialog(_Distribution())

    axis = dialog._figure.axes[0]
    assert axis.get_xlabel() == r"$B$ (T)"
    assert axis.get_ylabel() == r"$p(B)$"
    assert len(axis.lines) == 1
    assert axis.lines[0].get_ydata().max() == pytest.approx(1.0)
    assert axis.get_ylim()[0] == pytest.approx(0.0)
    dialog.close()


def test_magnetic_field_distribution_dialog_marks_an_unrelaxed_estimate(qapp) -> None:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        histogram_centres_T = np.asarray([1.0, 3.0], dtype=float)
        histogram_counts = np.asarray([4, 2], dtype=float)

    estimate = MagneticFieldDistributionDialog(_Distribution(), estimate=True)
    assert estimate._figure.axes[0].get_title() == "site 1 (unrelaxed estimate)"
    assert "# Estimate: UEP sites are unrelaxed" in estimate._export_text()
    estimate.close()

    relaxed = MagneticFieldDistributionDialog(_Distribution())
    assert relaxed._figure.axes[0].get_title() == "site 1"
    assert "# Estimate" not in relaxed._export_text()
    relaxed.close()


def test_magnetic_field_distribution_zero_padding_uses_bin_edges(qapp) -> None:
    x_values, y_values = MagneticFieldDistributionDialog._probability_steps_with_zero_baseline(
        np.asarray([1.0, 2.0], dtype=float),
        np.asarray([1.0, 0.5], dtype=float),
        0.0,
        3.0,
    )

    assert x_values == pytest.approx([0.0, 0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.0])
    assert y_values == pytest.approx([0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0])


def test_magnetic_field_distribution_dialog_overlays_multiple_histograms(qapp) -> None:
    class _Distribution:
        def __init__(self, label: str, centres: list[float], counts: list[float]) -> None:
            self.site_label = label
            self.fields_T = np.asarray([[0.0, 0.0, value] for value in centres], dtype=float)
            self.histogram_centres_T = np.asarray(centres, dtype=float)
            self.histogram_counts = np.asarray(counts, dtype=float)

    dialog = MagneticFieldDistributionDialog(
        (
            _Distribution("site 1", [1.0, 2.0], [4.0, 2.0]),
            _Distribution("site 2", [1.5, 2.5], [1.0, 3.0]),
        )
    )

    axis = dialog._figure.axes[0]
    legend = axis.get_legend()
    assert axis.get_title() == "2 field distributions"
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["site 1", "site 2", "sum"]
    distribution_lines = [
        line for line in axis.lines if line.get_label() in {"site 1", "site 2", "sum"}
    ]
    assert len(distribution_lines) == 3
    assert axis.get_ylim()[1] > 1.0
    dialog.close()


def test_magnetic_field_distribution_sum_uses_common_bins_for_delta(qapp) -> None:
    class _Distribution:
        def __init__(self, label: str, centres: list[float], counts: list[float]) -> None:
            self.site_label = label
            self.fields_T = np.asarray([[0.0, 0.0, value] for value in centres], dtype=float)
            self.histogram_centres_T = np.asarray(centres, dtype=float)
            self.histogram_counts = np.asarray(counts, dtype=float)

    delta_centre = 0.0719
    dialog = MagneticFieldDistributionDialog(
        (
            _Distribution("delta", [delta_centre], [5.0]),
            _Distribution("broad", [0.07, 0.08, 0.09], [1.0, 0.5, 0.25]),
        )
    )

    axis = dialog._figure.axes[0]
    delta_line = next(line for line in axis.lines if line.get_label() == "delta")
    sum_line = next(line for line in axis.lines if line.get_label() == "sum")
    delta_x = np.asarray(delta_line.get_xdata(), dtype=float)
    delta_y = np.asarray(delta_line.get_ydata(), dtype=float)
    sum_x = np.asarray(sum_line.get_xdata(), dtype=float)
    sum_y = np.asarray(sum_line.get_ydata(), dtype=float)
    delta_support = delta_y > 0.5

    assert sum_x == pytest.approx(delta_x)
    assert np.any(delta_support)
    assert float(np.max(sum_y[delta_support])) >= 1.0
    dialog.close()


def test_magnetic_field_distribution_dialog_exports_dat_file(qapp, tmp_path, monkeypatch) -> None:
    class _Distribution:
        site_label = "site 1"
        muon_position = (0.25, 0.25, 0.25)
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        histogram_centres_T = np.asarray([1.0, 3.0], dtype=float)
        histogram_counts = np.asarray([4, 2], dtype=float)
        calculation_parameters = {"supercell": [12, 12, 12], "nangles": 4096}

    export_path = tmp_path / "distribution.dat"
    monkeypatch.setattr(
        magnetic_field_widget_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(export_path), "DAT files (*.dat)"),
    )

    dialog = MagneticFieldDistributionDialog(_Distribution())
    dialog._export_data()

    text = export_path.read_text(encoding="utf-8")
    assert "# MuFinder field distribution export" in text
    assert "#   nangles: 4096" in text
    assert "# Columns: B_T p_site_1" in text
    assert "1" in text
    dialog.close()


def test_magnetic_field_distribution_dialog_expands_near_delta_axis(qapp) -> None:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray(
            [
                [0.0, 0.0, 0.1907378471],
                [0.0, 0.0, 0.1907378471001],
                [0.0, 0.0, 0.1907378470999],
            ],
            dtype=float,
        )
        histogram_centres_T = np.asarray(
            [0.1907378470999, 0.1907378471, 0.1907378471001], dtype=float
        )
        histogram_counts = np.asarray([1, 3, 1], dtype=float)

    dialog = MagneticFieldDistributionDialog(_Distribution())

    axis = dialog._figure.axes[0]
    x_low, x_high = axis.get_xlim()
    assert x_high - x_low > 0.001
    assert axis.xaxis.get_major_formatter().get_useOffset() is False
    dialog.close()


def test_magnetic_field_widget_summarises_model_and_emits_changes(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-summary.mufinder", "Magnetic Summary")
    service.add_structure(document, _host_structure())

    widget = MagneticFieldWidget()
    emitted: list[object] = []
    widget.magnetic_structure_changed.connect(emitted.append)
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))

    assert widget._structure_summary.text() == "Magnetic structure: 1 component"
    assert widget._classification.text() == "Structure type: commensurate"
    assert emitted
    assert isinstance(emitted[-1], dict)
    widget.close()


def _two_row_helix_model() -> MagneticStructureModel:
    helix = build_helix_component(
        label="Fe helix",
        selector=AtomSelector.species("Fe"),
        q_coeff=(1.0, 0.0, 0.0),
        real_axis_vector=(1.0, 0.0, 0.0),
        imaginary_axis_vector=(0.0, 1.0, 0.0),
    )
    return MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.25), (0.5, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(helix,),
    )


def _select_combo_data(combo, data) -> None:
    index = combo.findData(data)
    assert index >= 0
    combo.setCurrentIndex(index)


def test_component_dialog_lists_the_q_basis_and_picks_a_stored_row(qapp) -> None:
    dialog = MagneticComponentDialog(_host_structure(), model=_two_row_helix_model())
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

    _select_combo_data(combo, 1)

    assert dialog.q_basis_coefficients() == (0.0, 1.0, 0.0)
    assert dialog.q_vector() == (0.5, 0.0, 0.0)
    assert dialog._q_x.isReadOnly()
    assert "q = q2 = (0.5, 0, 0)" in dialog._q_expression.text()

    _select_combo_data(combo, MagneticComponentDialog._Q_SOURCE_VECTOR)
    assert not dialog._q_x.isReadOnly()
    dialog.close()


def test_component_dialog_combines_basis_rows(qapp) -> None:
    dialog = MagneticComponentDialog(_host_structure(), model=_two_row_helix_model())
    _select_combo_data(dialog._q_source_combo, MagneticComponentDialog._Q_SOURCE_COMBINATION)

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
        magnetic_field_widget_module.QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    dialog = MagneticComponentDialog(_host_structure(), model=_two_row_helix_model())
    _select_combo_data(dialog._q_source_combo, MagneticComponentDialog._Q_SOURCE_COMBINATION)

    dialog.accept()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "at least one q-basis row" in warnings[-1]
    assert "at least one q-basis row" in dialog._q_expression.text()
    dialog.close()


def test_component_dialog_without_stored_rows_only_offers_vector_entry(qapp) -> None:
    dialog = MagneticComponentDialog(_host_structure(), model=MagneticStructureModel())

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
    model = _two_row_helix_model()
    component = model.components[0]

    dialog = MagneticComponentDialog(
        _host_structure(),
        component=component,
        q_vector=(0.0, 0.0, 0.25),
        model=model,
        q_coeff=q_coeff,
    )

    assert dialog._q_source_combo.currentData() == expected_source
    assert dialog.q_basis_coefficients() == expected_coefficients
    dialog.close()


def test_magnetic_structure_builder_adds_a_component_on_a_basis_combination(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog = _builder_with_components(tmp_path)
    dialog._set_model(_two_row_helix_model())

    def _accept_combination(component_dialog):
        _select_combo_data(
            component_dialog._q_source_combo, MagneticComponentDialog._Q_SOURCE_COMBINATION
        )
        component_dialog._q_coeff_inputs[0].setText("1")
        component_dialog._q_coeff_inputs[1].setText("1")
        component_dialog._label_edit.setText("Sum")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(MagneticComponentDialog, "exec", _accept_combination)
    dialog._add_component()

    added = dialog.model().components[-1]
    assert added.label == "Sum"
    assert added.terms[0].q_coeff == (1.0, 1.0, 0.0)
    assert dialog.model().q_basis_rlu == _two_row_helix_model().q_basis_rlu
    assert dialog._q_basis_table.item(0, 2).text() == "Fe helix, Sum"
    assert dialog._q_basis_table.item(1, 2).text() == "Sum"
    assert dialog._q_basis_table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    dialog.close()


def _checklist_states(dialog: MagneticStructureBuilderDialog) -> dict[str, str]:
    return {
        key: label.property("checklistState") for key, label in dialog._checklist_labels.items()
    }


def test_magnetic_structure_builder_checklist_ticks_off_from_model_state(qapp, tmp_path) -> None:
    dialog = _builder_with_components(tmp_path)

    assert _checklist_states(dialog) == {
        "components": "todo",
        "q_basis": "todo",
        "preview": "todo",
        "ready": "todo",
    }
    assert "Add Preset" in dialog._checklist_labels["components"].text()

    dialog._set_model(_two_row_helix_model())
    assert _checklist_states(dialog) == {
        "components": "done",
        "q_basis": "done",
        "preview": "todo",
        "ready": "todo",
    }
    assert "1×1×4" in dialog._checklist_labels["preview"].text()

    dialog._repeat_c.setValue(4)
    assert _checklist_states(dialog) == {
        "components": "done",
        "q_basis": "done",
        "preview": "done",
        "ready": "done",
    }
    dialog.close()


def test_magnetic_structure_builder_checklist_flags_problems(qapp, tmp_path) -> None:
    model = _two_row_helix_model()
    incommensurate = replace(
        model, q_basis_rlu=((0.0, 0.0, 0.123456789), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    )
    zero = build_collinear_component(
        label="Zero", selector=AtomSelector.all(), q_coeff=(0.0, 0.0, 0.0), moment=(0.0, 0.0, 0.0)
    )
    dialog = _builder_with_components(tmp_path)

    dialog._set_model(incommensurate)
    assert _checklist_states(dialog)["preview"] == "problem"
    assert "incommensurate along c" in dialog._checklist_labels["preview"].text()
    dialog._repeat_c.setValue(3)
    assert _checklist_states(dialog)["preview"] == "done"

    dialog._set_model(replace(incommensurate, components=(*incommensurate.components, zero)))
    assert _checklist_states(dialog)["components"] == "problem"
    assert "Zero moment: &#x27;Zero&#x27;" in dialog._checklist_labels["components"].text()

    dialog._set_model(incommensurate)
    dialog._component_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert _checklist_states(dialog)["components"] == "problem"
    assert "Every component is disabled" in dialog._checklist_labels["components"].text()
    assert _checklist_states(dialog)["ready"] == "todo"
    dialog.close()


def test_magnetic_preset_dialog_defaults_to_a_helix_along_c_star(qapp) -> None:
    dialog = MagneticPresetDialog(_host_structure(), model=MagneticStructureModel())

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
    dialog = MagneticPresetDialog(_host_structure(), model=_two_row_helix_model())
    _select_combo_data(dialog._kind_combo, "conical")
    _select_combo_data(dialog._propagation_combo, 1)
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
    dialog = MagneticPresetDialog(_host_structure())
    dialog._label_edit.setText("My helix")
    dialog._label_edit.textEdited.emit("My helix")
    _select_combo_data(dialog._kind_combo, "cycloid")

    assert dialog._label_edit.text() == "My helix"
    assert dialog._reference_row_label.text() == "Rotation plane holds q and"
    dialog.close()


def test_magnetic_preset_dialog_rejects_a_reference_axis_parallel_to_q(qapp, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        magnetic_field_widget_module.QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    dialog = MagneticPresetDialog(_host_structure())
    _select_combo_data(dialog._reference_combo, "c")

    assert "Cannot build this preset" in dialog._summary.text()
    dialog.accept()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "parallel to q" in warnings[-1]
    dialog.close()


def test_magnetic_structure_builder_adds_a_preset_component(qapp, tmp_path, monkeypatch) -> None:
    dialog = _builder_with_components(tmp_path)
    dialog._set_model(_two_row_helix_model())

    def _accept_spin_density_wave(preset_dialog):
        _select_combo_data(preset_dialog._kind_combo, "spin_density_wave")
        _select_combo_data(preset_dialog._propagation_combo, "a*")
        preset_dialog._wavevector.setText("0.2")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(MagneticPresetDialog, "exec", _accept_spin_density_wave)
    assert dialog._add_preset_button.isEnabled()
    dialog._add_preset_button.click()

    added = dialog.model().components[-1]
    assert added.kind == "spin_density_wave"
    assert added.label == "Spin-density wave along a*"
    # q2 holds a vector no component uses, so the new q takes that row.
    assert added.terms[0].q_coeff == (0.0, 1.0, 0.0)
    assert dialog.model().q_basis_rlu[1] == (0.2, 0.0, 0.0)
    assert np.allclose(added.terms[0].coefficient.real, (0.0, 1.0, 0.0))
    dialog.close()


def test_magnetic_preset_dialog_ignores_a_hidden_cone_angle(qapp) -> None:
    dialog = MagneticPresetDialog(_host_structure())
    _select_combo_data(dialog._kind_combo, "conical")
    dialog._cone_angle.setText("")
    assert "Cannot build this preset" in dialog._summary.text()

    _select_combo_data(dialog._kind_combo, "helix")

    assert dialog._cone_angle.isHidden()
    assert "Cannot build this preset" not in dialog._summary.text()
    assert dialog.component((1.0, 0.0, 0.0)).kind == "helix"
    dialog.close()


def test_magnetic_preset_dialog_takes_the_wavevector_in_inverse_angstrom(qapp) -> None:
    dialog = MagneticPresetDialog(_host_structure())
    assert dialog._wavevector_unit.currentData() == "rlu"
    assert dialog._wavevector_row_label.text() == "Wavevector δ (r.l.u.)"

    _select_combo_data(dialog._wavevector_unit, "inverse_angstrom")

    # 0.25 r.l.u. along c* of the 3 Å cube is 0.25 x 2π / 3 Å⁻¹.
    assert dialog._wavevector_row_label.text() == "Wavevector k (Å⁻¹)"
    assert float(dialog._wavevector.text()) == pytest.approx(0.25 * 2.0 * np.pi / 3.0, rel=1e-5)
    assert dialog.q_vector() == pytest.approx((0.0, 0.0, 0.25), abs=1e-6)

    dialog._wavevector.setText("0.5")
    _select_combo_data(dialog._propagation_combo, "a*")
    assert dialog.q_vector() == pytest.approx((0.5 * 3.0 / (2.0 * np.pi), 0.0, 0.0))
    assert "q = (0.238732, 0, 0)" in dialog._summary.text()

    _select_combo_data(dialog._wavevector_unit, "rlu")
    assert float(dialog._wavevector.text()) == pytest.approx(0.238732, rel=1e-5)
    dialog.close()


def test_magnetic_preset_dialog_unit_switch_keeps_q_exactly(qapp) -> None:
    dialog = MagneticPresetDialog(_host_structure())

    _select_combo_data(dialog._wavevector_unit, "inverse_angstrom")
    assert dialog.q_vector() == pytest.approx((0.0, 0.0, 0.25), abs=1e-14)

    _select_combo_data(dialog._wavevector_unit, "rlu")
    assert dialog._wavevector.text() == "0.25"
    assert dialog.q_vector() == pytest.approx((0.0, 0.0, 0.25), abs=1e-14)
    dialog.close()


@pytest.mark.parametrize("delta", [-0.5, 4.0, -4.0, 0.137])
def test_magnetic_preset_dialog_unit_switch_keeps_every_valid_wavevector(qapp, delta) -> None:
    dialog = MagneticPresetDialog(_host_structure())
    dialog._wavevector.setText(f"{delta:g}")

    _select_combo_data(dialog._wavevector_unit, "inverse_angstrom")
    assert "Cannot build" not in dialog._summary.text()
    assert dialog.q_vector() == pytest.approx((0.0, 0.0, delta), abs=1e-14)

    _select_combo_data(dialog._wavevector_unit, "rlu")
    assert "Cannot build" not in dialog._summary.text()
    assert dialog.q_vector() == pytest.approx((0.0, 0.0, delta), abs=1e-14)
    dialog.close()


def test_magnetic_preset_dialog_k_range_follows_the_axis(qapp) -> None:
    # A 3 x 3 x 6 Å cell: 4 r.l.u. is 4 x 2π / 3 = 8.37758 Å⁻¹ along a* and half that along c*.
    structure = _host_structure(lattice=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 6.0]])
    dialog = MagneticPresetDialog(structure)
    _select_combo_data(dialog._wavevector_unit, "inverse_angstrom")

    dialog._wavevector.setText("-5")
    assert "between -4.18879 and 4.18879" in dialog._summary.text()

    _select_combo_data(dialog._propagation_combo, "a*")
    assert "Cannot build" not in dialog._summary.text()
    assert dialog.q_vector() == pytest.approx((-5.0 * 3.0 / (2.0 * np.pi), 0.0, 0.0))
    dialog._wavevector.setText("9")
    assert "between -8.37758 and 8.37758" in dialog._summary.text()
    dialog.close()


def test_magnetic_preset_dialog_refuses_a_non_finite_wavevector(qapp) -> None:
    dialog = MagneticPresetDialog(_host_structure())

    dialog._wavevector.setText("nan")

    assert "between -4 and 4" in dialog._summary.text()
    dialog.close()


def test_magnetic_preset_dialog_refuses_a_zero_inverse_angstrom_wavevector(qapp) -> None:
    dialog = MagneticPresetDialog(_host_structure())
    _select_combo_data(dialog._wavevector_unit, "inverse_angstrom")

    dialog._wavevector.setText("0")

    assert "non-zero number of inverse Angstrom" in dialog._summary.text()
    dialog.close()


def test_magnetic_preset_dialog_keeps_the_reference_axis_while_the_wavevector_changes(qapp) -> None:
    dialog = MagneticPresetDialog(_host_structure())
    _select_combo_data(dialog._reference_combo, "b")

    dialog._wavevector.setText("0.3")
    assert dialog._reference_combo.currentData() == "b"

    _select_combo_data(dialog._propagation_combo, "a*")
    assert dialog._reference_combo.currentData() == "b"
    _select_combo_data(dialog._reference_combo, "c")
    _select_combo_data(dialog._propagation_combo, "b*")
    assert dialog._reference_combo.currentData() == "a"
    dialog.close()
