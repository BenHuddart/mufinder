"""Tests for magnetic-field workflow UI helpers."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QItemSelectionModel, QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QAbstractItemView, QDialog, QGridLayout, QScrollArea

import mufinder.app.magnetic_field_widget as magnetic_field_widget_module
from mufinder.app import theme as theme_module
from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.magnetic_field_widget import (
    DipolarTensorDialog,
    DipolarTensorResultsDialog,
    MagneticComponentDialog,
    MagneticFieldCalculationSettingsDialog,
    MagneticFieldDistributionDialog,
    MagneticFieldWidget,
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
    build_conical_component,
)
from mufinder.services.project_service import ProjectService


def _host_structure() -> Structure:
    return Structure(
        label="FeO",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
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


def _custom_muon_site(position: tuple[float, float, float]) -> magnetic_field_widget_module.MagneticMuonSiteOption:
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
    assert dialog._component_table.columnCount() == 5
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

    label_item = dialog._component_table.item(0, 0)
    type_item = dialog._component_table.item(0, 1)
    assert label_item.flags() & Qt.ItemFlag.ItemIsEditable
    assert not type_item.flags() & Qt.ItemFlag.ItemIsEditable

    label_item.setText("Renamed")

    assert dialog.model().components[0].label == "Renamed"
    assert dialog._component_table.item(0, 0).text() == "Renamed"
    dialog.close()


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

    dialog = MagneticComponentDialog(_multi_species_structure(), component=component, q_vector=(0.0, 0.0, 0.5))

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

        def q_vector(self):
            return (0.25, 0.0, 0.0)

        def component(self, q_coeff):
            assert q_coeff == (1.0, 0.0, 0.0)
            return replacement

    monkeypatch.setattr(magnetic_field_widget_module, "MagneticComponentDialog", _Dialog)

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    dialog._set_model(MagneticStructureModel(q_basis_rlu=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), components=(original,)))
    dialog._component_table.selectRow(0)

    dialog._edit_selected_component()

    assert dialog.model().components == (replacement,)
    assert dialog.model().q_basis_rlu[0] == (0.25, 0.0, 0.0)
    assert dialog._component_table.item(0, 0).text() == "Edited"
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
            FourierTerm((1.0, 0.0, 0.0), ComplexVector(real=(0.0, 0.0, 1.0), imaginary=(0.5, 0.0, 0.0))),
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
    round_trip = dialog.component(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)), dialog.q_basis_vectors())
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
        QPointF(plot_rect.left() + 0.25 * plot_rect.width(), plot_rect.top() + 0.25 * plot_rect.height()),
        QPointF(plot_rect.left() + 0.75 * plot_rect.width(), plot_rect.top() + 0.75 * plot_rect.height()),
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

    square = _evaluate_texture_preview_grid(texture, mode="square_skyrmion", resolution=3, repeats=1.0)
    hexagonal = _evaluate_texture_preview_grid(texture, mode="hexagonal_skyrmion", resolution=3, repeats=1.0)

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


def test_magnetic_structure_builder_adds_texture_as_single_grouped_row(qapp, tmp_path, monkeypatch) -> None:
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
    assert "Fourier term" in dialog._component_table.item(0, 4).text()
    dialog.close()


def test_magnetic_structure_builder_edits_texture_component(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-texture-edit.mufinder", "Magnetic Texture Edit")
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
    dialog._set_model(MagneticStructureModel(q_basis_rlu=((0.25, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), components=(original,)))
    dialog._component_table.selectRow(0)

    dialog._edit_selected_component()

    assert dialog.model().components[0].label == "Edited texture"
    assert dialog.model().q_basis_rlu[0] == (0.0, 0.25, 0.0)
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
    np.testing.assert_allclose(conical.terms[0].coefficient.to_array(), [0.0, 0.0, 1.25], atol=1e-12)
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

    dialog = MagneticComponentDialog(_multi_species_structure(), component=component, q_vector=(0.0, 0.0, 0.25))

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
    np.testing.assert_allclose(left_handed.terms[0].coefficient.real, helix.terms[0].coefficient.real)
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


def test_magnetic_structure_builder_zero_q_component_does_not_consume_q_basis(qapp, tmp_path) -> None:
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


def test_magnetic_structure_builder_finite_q_after_zero_q_component_uses_free_basis_row(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-fm-then-helix.mufinder", "Magnetic FM Then Helix")
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
    dialog._set_model(MagneticStructureModel(q_basis_rlu=stale_full_basis, components=(zero_q_component,)))

    q_basis, q_coeff = dialog._q_basis_for((0.0, 0.0, 0.5))

    assert q_basis[0] == (0.0, 0.0, 0.5)
    assert q_coeff == (1.0, 0.0, 0.0)
    dialog.close()


def test_magnetic_structure_builder_supercell_preview_propagates_moments(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-supercell.mufinder", "Magnetic Supercell")
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


def test_muon_site_selection_dialog_lists_uep_cluster_muon_sites(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-uep-sites.mufinder", "Magnetic UEP Sites")
    service.add_structure(document, _host_structure())
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


def test_muon_site_selection_dialog_lists_relaxation_clusters_with_distortion_option(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-relax-sites.mufinder", "Magnetic Relax Sites")
    service.add_structure(document, _host_structure())
    relaxed_structure = Structure(
        label="Relaxed muon structure",
        lattice_vectors=_host_structure().lattice_vectors,
        species=["Fe", "O", "H"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.2, 0.3, 0.4]],
    )
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
                            "representative_fractional_position": [0.2, 0.3, 0.4],
                            "representative_cartesian_position_ang": [0.6, 0.9, 1.2],
                        }
                    ]
                }
            },
        }
    ]

    dialog = MuonSiteSelectionDialog(document, _host_structure())
    _set_site_source(dialog, "relaxation_clustering")

    assert dialog._site_table.rowCount() == 1
    selected = dialog.selected_muon_site()
    assert selected is not None
    assert selected.source == "relaxation_clustering"
    assert selected.label == "Positive muon relaxations cluster 1"
    assert selected.distortion_structure_id == relaxed_structure.id
    assert dialog._distortion_checkbox.isEnabled()
    dialog.close()


def test_muon_site_selection_dialog_custom_muon_site(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-custom-site.mufinder", "Magnetic Custom Site")
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

    def _calculate(structure, model, muon_sites, *, options):
        calls.append({"structure": structure, "model": model, "muon_sites": muon_sites, "options": options})
        return (_Result(),)

    monkeypatch.setattr(magnetic_field_widget_module, "classify_magnetic_model", lambda *_args, **_kwargs: "commensurate")
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
    assert calls[0]["options"].demag_shape == "none"
    assert calls[0]["options"].supercell == (100, 100, 100)
    assert widget._result_table.rowCount() == 1
    assert widget._result_table.item(0, 1).text() == "0.1"
    assert widget._result_table.item(0, 5).text() == "field"
    assert widget.collect_state()["magnetic_field_results"]["kind"] == "commensurate"
    assert widget.collect_state()["magnetic_field_results"]["results"][0]["total_T"] == pytest.approx([0.1, 0.2, 0.3])
    assert "commensurate" in widget._calculation_status.text()
    widget.close()


def test_magnetic_field_widget_enables_nuclear_relaxation_without_magnetic_model(qapp, tmp_path) -> None:
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
    assert widget.collect_state()["nuclear_relaxation_results"]["sigma"] == 0.2
    widget.close()


def test_magnetic_field_widget_calculates_nuclear_relaxation_from_dialog(qapp, tmp_path, monkeypatch) -> None:
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
        calls.append({"structure": structure, "muon_position": muon_position, "site_label": site_label, "options": options})
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

    def _calculate(structure, model, muon_sites, *, options):
        calls.append({"structure": structure, "model": model, "muon_sites": muon_sites, "options": options})
        return ()

    monkeypatch.setattr(magnetic_field_widget_module, "classify_magnetic_model", lambda *_args, **_kwargs: "commensurate")
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
    assert calls[0]["options"].nangles == 4096
    assert calls[0]["options"].histogram_bins == 512
    widget.close()


def test_magnetic_field_widget_defaults_demag_shape_to_none(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-default-demag.mufinder", "Magnetic Default Demag")
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


def test_demagnetizing_advanced_dialog_collects_prism_values(qapp, monkeypatch) -> None:
    class _Preview(magnetic_field_widget_module.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.samples = []

        def set_sample(self, sample):
            self.samples.append(sample)

        def shutdown(self):
            pass

    monkeypatch.setattr(magnetic_field_widget_module, "_DemagnetizingPyVistaCanvas", _Preview)

    dialog = magnetic_field_widget_module.DemagnetizingFieldAdvancedDialog(
        magnetic_field_widget_module.DemagnetizingSample(shape="ellipsoid")
    )
    dialog._shape.setCurrentIndex(dialog._shape.findData("rectangular_prism"))
    dialog._prism_a.setValue(2.0)
    dialog._prism_b.setValue(3.0)
    dialog._prism_c.setValue(4.0)
    dialog._fourier_grid.setValue(24)
    dialog._padding.setValue(3.0)

    sample = dialog.selected_sample()

    assert sample.shape == "rectangular_prism"
    assert sample.prism_lengths == pytest.approx((2.0, 3.0, 4.0))
    assert sample.fourier_grid == 24
    assert sample.padding == pytest.approx(3.0)
    assert "N =" in dialog._tensor_label.text()
    dialog.close()


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
        def __init__(self, parent=None):
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

    monkeypatch.setattr(magnetic_field_widget_module, "_pyvista_modules", lambda: (_FakePv, _FakePlotter))
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
        assert canvas._plotter.backgrounds[-1] == magnetic_field_widget_module.current_palette().bg_primary
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
        def __init__(self, sample, parent=None):
            assert sample.shape == "ellipsoid"

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


def test_magnetic_field_widget_restores_advanced_demag_state(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "advanced-demag-restore.mufinder", "Advanced Demag Restore")
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


def test_magnetic_field_widget_empty_form_state_resets_previous_project_values(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-empty-state.mufinder", "Magnetic Empty State")
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
        }
    )

    widget.set_form_state({})

    assert widget._demag_shape() == "none"
    assert widget._field_options().supercell == (100, 100, 100)
    assert widget._include_equivalent_sites.isChecked() is False
    assert widget._lookup_angles.value() == 4096
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

    assert dialog.values() == {
        "supercell": (18, 24, 36),
        "nangles": 8192,
        "phase_sample_count": 250_000,
        "phase_sampling": "montecarlo",
        "histogram_bins": 1024,
    }
    assert dialog._phase_samples.maximum() == magnetic_field_widget_module.MAX_INCOMMENSURATE_PHASE_SAMPLES
    dialog.close()


def test_magnetic_field_widget_opens_calculation_settings_dialog(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-settings-dialog.mufinder", "Magnetic Settings Dialog")
    service.add_structure(document, _host_structure())

    class _Dialog:
        def __init__(self, *, supercell, nangles, phase_sample_count, histogram_bins, phase_sampling, parent=None):
            assert supercell == (100, 100, 100)
            assert nangles == 4096
            assert phase_sample_count == 200_000
            assert histogram_bins == 512
            assert phase_sampling == "auto"

        def exec(self):
            return QDialog.DialogCode.Accepted

        def values(self):
            return {
                "supercell": (24, 36, 48),
                "nangles": 8192,
                "phase_sample_count": 250_000,
                "phase_sampling": "grid",
                "histogram_bins": 1024,
            }

    monkeypatch.setattr(magnetic_field_widget_module, "MagneticFieldCalculationSettingsDialog", _Dialog)

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
    assert "cells 24 x 36 x 48" in widget._calculation_settings_summary.text()
    assert emitted[-1]["supercell"] == [24, 36, 48]
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
    document = service.create_project(tmp_path / "magnetic-tensor-dialog.mufinder", "Magnetic Tensor Dialog")
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
                    "dipolar_tensor_T_per_muB": np.asarray(result.tensor_T_per_muB, dtype=float).tolist(),
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
    monkeypatch.setattr(magnetic_field_widget_module, "calculate_dipolar_tensors_for_atoms", _calculate)

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
    assert widget.collect_state()["dipolar_tensor_selector"]["selector"] == AtomSelector.indices([0]).to_json()
    assert widget.collect_state()["dipolar_tensor_results"][0]["site_label"] == "site 1"
    widget.close()


def test_magnetic_field_widget_passes_incommensurate_sampling_controls(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-sampling.mufinder", "Magnetic Sampling")
    service.add_structure(document, _host_structure())

    calls: list[dict[str, object]] = []

    def _calculate(structure, model, muon_sites, *, options):
        calls.append({"structure": structure, "model": model, "muon_sites": muon_sites, "options": options})
        return ()

    monkeypatch.setattr(magnetic_field_widget_module, "classify_magnetic_model", lambda *_args, **_kwargs: "incommensurate")
    monkeypatch.setattr(magnetic_field_widget_module, "calculate_incommensurate_field_distributions", _calculate)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._set_model(MagneticStructureModel(components=(_component("one", "One", 0),)))
    widget._selected_muon_site = _custom_muon_site((0.0, 0.0, 0.0))
    widget._phase_samples.setValue(250_000)
    widget._phase_sampling_method = "montecarlo"
    widget._histogram_bins.setValue(1024)

    widget._calculate_fields()

    assert calls
    assert calls[0]["options"].nangles == 4096
    assert calls[0]["options"].phase_sample_count == 250_000
    assert calls[0]["options"].phase_sampling == "montecarlo"
    assert calls[0]["options"].histogram_bins == 1024
    widget.close()


def test_magnetic_field_widget_converts_hyperfine_coupling_units(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-hyperfine.mufinder", "Magnetic Hyperfine")
    service.add_structure(document, _host_structure())

    calls: list[dict[str, object]] = []

    def _calculate(structure, model, muon_sites, *, options):
        calls.append({"structure": structure, "model": model, "muon_sites": muon_sites, "options": options})
        return ()

    monkeypatch.setattr(magnetic_field_widget_module, "classify_magnetic_model", lambda *_args, **_kwargs: "commensurate")
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
    assert [widget._hyperfine_units.itemData(index) for index in range(widget._hyperfine_units.count())] == [
        "angstrom^-3",
        "T_per_muB",
        "MHz_per_muB",
    ]
    widget.close()


def test_magnetic_field_widget_passes_equivalent_site_option(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-equivalent-sites.mufinder", "Magnetic Equivalent Sites")
    service.add_structure(document, _host_structure())

    calls: list[dict[str, object]] = []

    def _calculate(structure, model, muon_sites, *, options):
        calls.append({"structure": structure, "model": model, "muon_sites": muon_sites, "options": options})
        return ()

    monkeypatch.setattr(magnetic_field_widget_module, "classify_magnetic_model", lambda *_args, **_kwargs: "commensurate")
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
    document = service.create_project(tmp_path / "magnetic-form-state.mufinder", "Magnetic Form State")
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
    assert restored._field_options().nangles == 4096
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
    document = service.create_project(tmp_path / "magnetic-restore-dist.mufinder", "Magnetic Restore Dist")
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
    assert widget._selected_distribution_results()[0]["site_label"] == "site 1"
    widget.close()


def test_magnetic_field_widget_emits_form_state_for_setting_changes(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-form-emits.mufinder", "Magnetic Form Emits")
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
    document = service.create_project(tmp_path / "magnetic-workspace-signals.mufinder", "Magnetic Workspace")
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


def test_magnetic_field_widget_calculates_incommensurate_distribution(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-dist.mufinder", "Magnetic Dist")
    service.add_structure(document, _host_structure())

    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)

    monkeypatch.setattr(magnetic_field_widget_module, "classify_magnetic_model", lambda *_args, **_kwargs: "incommensurate")
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
    assert widget.collect_state()["magnetic_field_results"]["results"][0]["fields_T"] == [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]]
    widget.close()


def test_magnetic_field_widget_opens_incommensurate_distribution_plot(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-dist-plot.mufinder", "Magnetic Dist Plot")
    service.add_structure(document, _host_structure())

    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        histogram_centres_T = np.asarray([1.0, 3.0], dtype=float)
        histogram_counts = np.asarray([1, 1], dtype=float)

    monkeypatch.setattr(magnetic_field_widget_module, "classify_magnetic_model", lambda *_args, **_kwargs: "incommensurate")
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: (_Distribution(),),
    )

    opened: list[object] = []

    class _Dialog:
        def __init__(self, results, parent=None):
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


def test_magnetic_field_widget_opens_selected_distribution_plots_together(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-multi-dist-plot.mufinder", "Magnetic Multi Dist Plot")
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
    monkeypatch.setattr(magnetic_field_widget_module, "classify_magnetic_model", lambda *_args, **_kwargs: "incommensurate")
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: distributions,
    )

    opened: list[tuple[object, ...]] = []

    class _Dialog:
        def __init__(self, results, parent=None):
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
    distribution_lines = [line for line in axis.lines if line.get_label() in {"site 1", "site 2", "sum"}]
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
        histogram_centres_T = np.asarray([0.1907378470999, 0.1907378471, 0.1907378471001], dtype=float)
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
