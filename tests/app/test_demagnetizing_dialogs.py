"""Tests for the demagnetizing-field sample dialog and its 3D preview."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QWidget

import mufinder.app.demagnetizing_dialogs as demagnetizing_dialogs_module
from mufinder.app.sample_photo_dialog import PREVIEW_DEBOUNCE_MS
from mufinder.app import theme as theme_module
from mufinder.app.theme import current_palette
from mufinder.science.demagnetization import DemagnetizingSample, PlatePhoto, demagnetizing_tensor
from tests.support.magnetic_fields import photo_plate_sample


class _DemagPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.samples = []

    def set_sample(self, sample, lattice_vectors=None):
        self.samples.append((sample, lattice_vectors))

    def shutdown(self):
        pass


_HEXAGONAL_LATTICE = [[3.0, 0.0, 0.0], [-1.5, 1.5 * np.sqrt(3.0), 0.0], [0.0, 0.0, 5.0]]


def _shown_preview(dialog):
    """The sample and lattice the 3D preview shows once the debounced redraw has run."""

    for _ in range(100):
        if not dialog._preview_timer.isActive():
            break
        QTest.qWait(20)
    assert not dialog._preview_timer.isActive()
    return dialog._preview.samples[-1]


def _demag_dialog(monkeypatch, sample=None, lattice_vectors=None):
    monkeypatch.setattr(demagnetizing_dialogs_module, "_DemagnetizingPyVistaCanvas", _DemagPreview)
    return demagnetizing_dialogs_module.DemagnetizingFieldAdvancedDialog(
        sample or DemagnetizingSample(shape="ellipsoid"),
        lattice_vectors=lattice_vectors,
    )


def _set_indices(inputs, values) -> None:
    for edit, value in zip(inputs, values, strict=True):
        edit.setText(str(value))


def test_demagnetizing_advanced_dialog_collects_prism_values(qapp, monkeypatch) -> None:
    monkeypatch.setattr(demagnetizing_dialogs_module, "_DemagnetizingPyVistaCanvas", _DemagPreview)

    dialog = demagnetizing_dialogs_module.DemagnetizingFieldAdvancedDialog(
        DemagnetizingSample(shape="ellipsoid")
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


def test_demagnetizing_advanced_dialog_needs_an_outline_for_a_plate(qapp, monkeypatch) -> None:
    dialog = _demag_dialog(monkeypatch)
    dialog._shape.setCurrentIndex(dialog._shape.findData("image_plate"))

    assert dialog._stack.currentIndex() == 2
    assert "No outline yet" in dialog._plate_summary.text()
    assert dialog._photo_button.text() == "From photo…"
    assert "no outline" in dialog._tensor_label.text()
    assert not dialog._use_button.isEnabled()
    dialog.close()


def test_demagnetizing_advanced_dialog_takes_the_plate_from_the_photo_dialog(
    qapp, monkeypatch
) -> None:
    plate = photo_plate_sample(photo=PlatePhoto(path="/photos/crystal.png"))

    class _PhotoDialog:
        def __init__(self, sample, parent=None):
            assert sample.shape == "image_plate"

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selected_sample(self):
            return plate

    monkeypatch.setattr(demagnetizing_dialogs_module, "SamplePhotoDialog", _PhotoDialog)
    dialog = _demag_dialog(monkeypatch, lattice_vectors=np.eye(3).tolist())
    dialog._shape.setCurrentIndex(dialog._shape.findData("image_plate"))

    dialog._open_photo_dialog()

    assert dialog._use_button.isEnabled()
    assert "Plate from photo 5 x 5 x 0.2 (crystal.png)" in dialog._plate_summary.text()
    assert dialog._photo_button.text() == "Edit outline…"
    assert "Tr N = 1.00000" in dialog._tensor_label.text()
    assert "FFT box" in dialog._tensor_label.text()
    assert _shown_preview(dialog)[0].shape == "image_plate"
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

    monkeypatch.setattr(demagnetizing_dialogs_module, "SamplePhotoDialog", _PhotoDialog)
    plate = photo_plate_sample()
    dialog = _demag_dialog(monkeypatch, sample=plate)

    dialog._open_photo_dialog()

    assert dialog.selected_sample() == plate
    dialog.close()


def test_demagnetizing_advanced_dialog_offers_voxel_grid_only_for_a_voxel_seed(
    qapp, monkeypatch
) -> None:
    voxels = DemagnetizingSample.voxel_grid(np.ones((6, 4, 2), dtype=bool), 0.5)
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
    preview_sample, preview_lattice = _shown_preview(dialog)
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
    seed = DemagnetizingSample.oriented(
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
    seed = DemagnetizingSample(shape="ellipsoid", semi_axes=(1.0, 2.0, 3.0), rotation=rotation)
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
    original = demagnetizing_dialogs_module.principal_demagnetizing_tensor

    def _counting(sample):
        calls.append(sample)
        return original(sample)

    monkeypatch.setattr(demagnetizing_dialogs_module, "principal_demagnetizing_tensor", _counting)
    dialog = _demag_dialog(monkeypatch, lattice_vectors=np.eye(3).tolist())
    dialog._orientation_mode.setCurrentIndex(dialog._orientation_mode.findData("directions"))
    _set_indices(dialog._normal_inputs, (1, 1, 0))
    _set_indices(dialog._in_plane_inputs, (0, 0, 1))

    assert len(calls) == 1
    dialog._ellipsoid_c.setValue(2.0)
    assert len(calls) == 2
    dialog.close()


def test_demagnetizing_advanced_dialog_restores_saved_directions(qapp, monkeypatch) -> None:
    seed = DemagnetizingSample.oriented(
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

    class _FakePlotter(QWidget):
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
        demagnetizing_dialogs_module, "_pyvista_modules", lambda: (_FakePv, _FakePlotter)
    )
    canvas = demagnetizing_dialogs_module._DemagnetizingPyVistaCanvas()
    ellipsoid = DemagnetizingSample.oriented(
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

    class _FakePlotter(QWidget):
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
        demagnetizing_dialogs_module, "_pyvista_modules", lambda: (_FakePv, _FakePlotter)
    )
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)
    try:
        canvas = demagnetizing_dialogs_module._DemagnetizingPyVistaCanvas()
        sample = DemagnetizingSample(
            shape="rectangular_prism",
            prism_lengths=(2.0, 3.0, 4.0),
        )

        canvas.set_sample(sample)
        palette = current_palette()

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
        assert canvas._plotter.backgrounds[-1] == current_palette().bg_primary
        assert canvas._plotter.clear_calls >= 2
        clear_calls = canvas._plotter.clear_calls
        canvas.changeEvent(QEvent(QEvent.Type.StyleChange))
        assert canvas._plotter.clear_calls == clear_calls
        canvas.close()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def test_demagnetizing_preview_draws_the_voxels_of_a_plate(qapp, monkeypatch) -> None:
    pyvista = pytest.importorskip("pyvista")

    class _FakePlotter(QWidget):
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
        demagnetizing_dialogs_module, "_pyvista_modules", lambda: (pyvista, _FakePlotter)
    )
    canvas = demagnetizing_dialogs_module._DemagnetizingPyVistaCanvas()
    plate = photo_plate_sample(thickness_voxels=4)

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
    canvas.set_sample(DemagnetizingSample(shape="image_plate"))
    assert len(canvas._plotter.meshes) == 4
    canvas.close()


def test_demagnetizing_dialog_redraws_its_preview_once_edits_pause(qapp, monkeypatch) -> None:
    dialog = _demag_dialog(monkeypatch)
    # The dialog opens with its preview drawn.
    assert len(dialog._preview.samples) == 1

    for value in (1.5, 2.0, 2.5):
        dialog._ellipsoid_c.setValue(value)

    # The tensor follows every edit; the preview redraws once, with the last shape.
    assert dialog.selected_sample().semi_axes[2] == pytest.approx(2.5)
    assert "N = " in dialog._tensor_label.text()
    assert len(dialog._preview.samples) == 1
    sample, _lattice = _shown_preview(dialog)
    assert len(dialog._preview.samples) == 2
    assert sample.semi_axes[2] == pytest.approx(2.5)
    dialog.close()


def test_demagnetizing_dialog_draws_no_preview_after_it_is_accepted(qapp, monkeypatch) -> None:
    dialog = _demag_dialog(monkeypatch)
    dialog._ellipsoid_c.setValue(2.0)

    dialog.accept()
    QTest.qWait(2 * PREVIEW_DEBOUNCE_MS)

    assert len(dialog._preview.samples) == 1
    dialog.close()
