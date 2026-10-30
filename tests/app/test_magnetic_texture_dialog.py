"""Tests for the magnetic texture dialog and its preview canvas."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtWidgets import QDialog, QMessageBox, QScrollArea

import mufinder.app.magnetic_texture_dialog as magnetic_texture_dialog_module
from mufinder.app import theme as theme_module
from mufinder.app.magnetic_texture_dialog import MagneticTextureDialog, MagneticTexturePreview
from mufinder.science.magnetic_structure import (
    AtomSelector,
    ComplexVector,
    FourierTerm,
    MagneticComponent,
    texture_diagnostics,
)
from tests.support.magnetic_fields import coo_structure, set_direction_inputs_text


def test_magnetic_texture_dialog_refuses_a_zero_core_direction(qapp, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    dialog = MagneticTextureDialog(coo_structure())
    set_direction_inputs_text(dialog._q1, (0.1, 0.0, 0.0))
    set_direction_inputs_text(dialog._core_direction, (0.0, 0.0, 0.0))

    dialog._refresh_preview()
    dialog.accept()

    message = "core direction must be a non-zero direction"
    assert dialog._diagnostics.text() == f"Texture preview unavailable: {message}"
    assert warnings == [message]
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.close()


def test_magnetic_texture_dialog_builds_hexagonal_texture_component(qapp) -> None:
    dialog = MagneticTextureDialog(coo_structure(), selected_indices=(0, 2))
    dialog._selected_radio.setChecked(True)
    dialog._label_edit.setText("Hex texture")
    dialog._amplitude.setText("3.0")
    set_direction_inputs_text(dialog._q1, (0.1, 0.0, 0.0))
    set_direction_inputs_text(dialog._core_direction, (0.0, 0.0, 1.0))

    q_basis = dialog.q_basis_vectors()
    component = dialog.component(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)), q_basis)

    assert component.kind == "texture"
    assert component.label == "Hex texture"
    assert component.selector == AtomSelector.indices([0, 2])
    assert len(component.terms) == 3
    assert component.metadata["type"] == "texture"
    assert component.metadata["texture_mode"] == "hexagonal_skyrmion"
    assert "terms" in dialog._diagnostics.text()
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
    assert dialog._magnitude_info_button.toolTip() == "Explain magnitude normalization and pruning."
    dialog.close()


def test_magnetic_texture_dialog_round_trips_preset_builder_state(qapp) -> None:
    structure = coo_structure()
    dialog = MagneticTextureDialog(structure, selected_indices=(0, 2))
    dialog._mode_buttons["square_skyrmion"].setChecked(True)
    dialog._label_edit.setText("Square texture")
    dialog._selected_radio.setChecked(True)
    dialog._amplitude.setText("2.5")
    set_direction_inputs_text(dialog._q1, (0.12, 0.0, 0.0))
    set_direction_inputs_text(dialog._core_direction, (0.0, 0.0, 1.0))
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

    dialog = MagneticTextureDialog(coo_structure(), component=component)

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
    dialog = MagneticTextureDialog(coo_structure())

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
    dialog = MagneticTextureDialog(coo_structure())

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
    dialog = MagneticTextureDialog(coo_structure())
    dialog._grid_u.setValue(16)
    dialog._grid_v.setValue(24)
    dialog._grid_w.setValue(2)
    dialog._normalise_threshold.setText("0.0005")
    dialog._prune_mode.setCurrentText("Keep strongest N")
    dialog._prune_top_n.setValue(12)

    html = dialog._magnitude_pruning_help_html()

    assert "inverse magnitude field" in html
    assert "16 × 24 × 2" in html
    assert "0.0005" in html
    assert "Keep strongest N" in html
    assert "12 largest" in html
    dialog.close()


def test_magnetic_texture_dialog_custom_table_delete_and_backspace(qapp) -> None:
    dialog = MagneticTextureDialog(coo_structure())
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
    dialog = MagneticTextureDialog(coo_structure())
    dialog._mode_buttons["square_skyrmion"].setChecked(True)
    dialog._normalise_check.setChecked(True)
    dialog._grid_u.setValue(8)
    dialog._grid_v.setValue(8)
    dialog._prune_mode.setCurrentText("Keep strongest N")
    dialog._prune_top_n.setValue(2)

    texture = dialog.texture()

    assert len(texture.terms) == 2
    assert "2 terms" in dialog._diagnostics.text()
    dialog.close()


def test_magnetic_texture_dialog_reports_topological_charge_per_magnetic_cell(qapp) -> None:
    dialog = MagneticTextureDialog(coo_structure())

    assert "Topological charge per magnetic cell: -1." in dialog._diagnostics.text()
    set_direction_inputs_text(dialog._core_direction, (0.0, 0.0, 1.0))
    assert "Topological charge per magnetic cell: 1." in dialog._diagnostics.text()
    dialog._helicity.setCurrentIndex(dialog._helicity.findData(-1))
    assert "Topological charge per magnetic cell: 1." in dialog._diagnostics.text()

    set_direction_inputs_text(dialog._core_direction, (0.0, 0.0, -1.0))
    dialog._mode_buttons["square_skyrmion"].setChecked(True)
    assert (
        "Topological charge per magnetic cell: undefined (the moment magnitude vanishes"
        in dialog._diagnostics.text()
    )
    dialog._background.setText("-0.3")
    assert "Topological charge per magnetic cell: -1." in dialog._diagnostics.text()
    dialog.close()


def test_magnetic_texture_dialog_magnitude_map_toggles_the_preview(qapp) -> None:
    dialog = MagneticTextureDialog(coo_structure())
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
    dialog = MagneticTextureDialog(coo_structure())
    calls = []
    original = magnetic_texture_dialog_module.topological_charge_magnetic_unit_cell

    def counting(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        magnetic_texture_dialog_module, "topological_charge_magnetic_unit_cell", counting
    )
    dialog._label_edit.setText("Renamed texture")
    dialog._magnitude_map_check.setChecked(True)
    assert calls == []
    assert "Topological charge per magnetic cell: -1." in dialog._diagnostics.text()

    set_direction_inputs_text(dialog._core_direction, (0.0, 0.0, 1.0))
    assert len(calls) == 1
    assert "Topological charge per magnetic cell: 1." in dialog._diagnostics.text()
    dialog.close()


def test_magnetic_texture_dialog_unticking_the_map_brings_a_popped_out_map_back(qapp) -> None:
    dialog = MagneticTextureDialog(coo_structure())
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
    dialog = MagneticTextureDialog(coo_structure())
    dialog._normalise_check.setChecked(True)
    dialog._magnitude_map_check.setChecked(True)

    magnitudes = np.asarray(dialog._magnitude_mesh.get_array(), dtype=float)
    reported = texture_diagnostics(dialog.texture(), dialog._grid_resolution())

    assert float(np.std(magnitudes)) == pytest.approx(reported.magnitude_std, rel=1e-9)
    assert float(np.ptp(magnitudes)) < 0.1 * float(np.mean(magnitudes))
    assert f"std {reported.magnitude_std:.4g}" in dialog._diagnostics.text()
    dialog.close()


@pytest.mark.parametrize("canvas", ["light", "dark"])
def test_magnetic_texture_preview_paints_its_canvas_colours(qapp, canvas) -> None:
    preview = MagneticTexturePreview()
    preview.resize(300, 260)
    preview.set_canvas_mode(canvas)
    colors = theme_module.texture_preview_colors(canvas)

    image = preview.grab().toImage()
    rect = preview._plot_rect()

    assert image.pixelColor(2, 2) == QColor(colors.surround)
    assert image.pixelColor(int(rect.left()) + 20, int(rect.top()) + 20) == QColor(
        colors.plot_background
    )
    preview.close()
