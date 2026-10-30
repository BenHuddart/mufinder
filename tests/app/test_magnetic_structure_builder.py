"""Tests for the Magnetic Structure Builder dialog."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QItemSelectionModel, Qt
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QMessageBox

import mufinder.app.magnetic_structure_builder as magnetic_structure_builder_module
from mufinder.app import theme as theme_module
from mufinder.app import workflow_table_utils
from mufinder.app.magnetic_component_dialogs import MagneticComponentDialog, MagneticPresetDialog
from mufinder.app.magnetic_structure_builder import MagneticStructureBuilderDialog
from mufinder.domain.models import Structure
from mufinder.science.magnetic_structure import (
    AtomSelector,
    ComplexVector,
    FourierTerm,
    MagneticComponent,
    MagneticStructureError,
    MagneticStructureModel,
    build_collinear_component,
)
from mufinder.services.project_service import ProjectService
from tests.support.magnetic_field_sites import feo_host_structure
from tests.support.magnetic_fields import select_combo_data, two_row_helix_model, z_moment_component


def test_magnetic_structure_builder_imports_legacy_host_spins(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic.mufinder", "Magnetic")
    service.add_structure(document, feo_host_structure())

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)

    dialog._build_from_host_spins()

    assert dialog._component_table.rowCount() == 2
    assert dialog._component_table.columnCount() == 6
    assert dialog._preview_status.text() == "Preview: full magnetic structure (1x1x1)"
    assert dialog._classification.text() == "Structure type: commensurate"
    assert dialog._viewer._spins is not None
    assert dialog._viewer._spins.shape == (2, 3)
    # The builder styles the viewer's checkboxes from its own stylesheet, not the viewer's.
    assert dialog.styleSheet() == theme_module.magnetic_structure_builder_stylesheet()
    assert dialog._viewer._show_cell.styleSheet() == ""
    assert len(dialog.model().components) == 2
    dialog.close()


def test_magnetic_structure_builder_preview_reports_an_invalid_model(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-invalid.mufinder", "Magnetic")
    service.add_structure(document, feo_host_structure())
    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)

    def _invalid(*_args, **_kwargs):
        raise MagneticStructureError("atom selector contains an out-of-range index")

    monkeypatch.setattr(magnetic_structure_builder_module, "moments_in_supercell", _invalid)
    dialog._build_from_host_spins()

    assert dialog._preview_status.text() == (
        "Preview unavailable: atom selector contains an out-of-range index"
    )
    assert dialog._classification.text() == "Structure type: -"
    dialog.close()


def test_magnetic_structure_builder_delete_and_backspace_remove_components(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-delete.mufinder", "Magnetic Delete")
    service.add_structure(document, feo_host_structure())

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    model = MagneticStructureModel(
        components=(
            z_moment_component("one", "One", 0),
            z_moment_component("two", "Two", 1),
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
    service.add_structure(document, feo_host_structure())

    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    model = MagneticStructureModel(components=(z_moment_component("one", "One", 0),))
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
    service.add_structure(document, feo_host_structure())
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
        tmp_path, z_moment_component("one", "One", 0), z_moment_component("two", "Two", 1)
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
    disabled = replace(z_moment_component("two", "Two", 1), enabled=False)
    dialog = _builder_with_components(tmp_path, z_moment_component("one", "One", 0), disabled)

    assert dialog._component_table.item(0, 0).checkState() == Qt.CheckState.Checked
    assert dialog._component_table.item(1, 0).checkState() == Qt.CheckState.Unchecked
    dialog.close()


def test_magnetic_structure_builder_duplicates_selected_components(qapp, tmp_path) -> None:
    dialog = _builder_with_components(
        tmp_path,
        z_moment_component("one", "One", 0),
        replace(z_moment_component("two", "Two", 1), enabled=False),
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
        z_moment_component("one", "One", 0),
        z_moment_component("two", "Two", 1),
        z_moment_component("three", "Three", 0),
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
        tmp_path, z_moment_component("one", "One", 0), z_moment_component("two", "Two", 1)
    )

    _select_builder_rows(dialog, 0, 1)
    dialog._remove_button.click()
    assert confirmed_destructive_actions[-1]["text"] == "Remove 2 components?"
    assert dialog._component_table.rowCount() == 0

    dialog._set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
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

    dialog._set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    monkeypatch.setattr(
        workflow_table_utils, "confirm_destructive", lambda *_args, **_kwargs: False
    )
    dialog._build_from_host_spins()
    assert _builder_labels(dialog) == ["One"]
    dialog.close()


def test_magnetic_structure_builder_host_spin_prompt_names_component_count(
    qapp, tmp_path, confirmed_destructive_actions
) -> None:
    dialog = _builder_with_components(tmp_path, z_moment_component("one", "One", 0))

    dialog._build_from_host_spins()

    assert confirmed_destructive_actions[-1]["title"] == "Replace magnetic structure"
    assert "Replace the 1 component" in confirmed_destructive_actions[-1]["text"]
    assert confirmed_destructive_actions[-1]["accept_label"] == "Replace Components"
    assert dialog._component_table.rowCount() == 2
    dialog.close()


def test_magnetic_structure_builder_edits_selected_component(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-edit.mufinder", "Magnetic Edit")
    service.add_structure(document, feo_host_structure())

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

    monkeypatch.setattr(magnetic_structure_builder_module, "MagneticComponentDialog", _Dialog)

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


def test_magnetic_structure_builder_adds_texture_as_single_grouped_row(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-texture.mufinder", "Magnetic Texture")
    service.add_structure(document, feo_host_structure())

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

    monkeypatch.setattr(magnetic_structure_builder_module, "MagneticTextureDialog", _Dialog)

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
    service.add_structure(document, feo_host_structure())
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

    monkeypatch.setattr(magnetic_structure_builder_module, "MagneticTextureDialog", _Dialog)

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
    original = replace(z_moment_component("one", "One", 0), enabled=False)
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
    monkeypatch.setattr(magnetic_structure_builder_module, dialog_name, _Dialog)
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
        z_moment_component("one", "One", 0),
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

    monkeypatch.setattr(magnetic_structure_builder_module, "MagneticTextureDialog", _Dialog)
    dialog = _builder_with_components(tmp_path, original)
    dialog._component_table.selectRow(0)

    dialog._edit_selected_component()

    assert dialog.model().components[0].label == "Edited"
    assert dialog.model().components[0].atom_phase_turns == {1: 0.5}
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
    assert dialog.viewers_shut_down() is True


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
    assert table.item(1, 0).toolTip() == "Used by the selected component."
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
        QMessageBox,
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
    dialog = _builder_with_components(tmp_path, z_moment_component("one", "One", 0))
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


def test_magnetic_structure_builder_adds_a_component_on_a_basis_combination(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog = _builder_with_components(tmp_path)
    dialog._set_model(two_row_helix_model())

    def _accept_combination(component_dialog):
        select_combo_data(
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
    assert dialog.model().q_basis_rlu == two_row_helix_model().q_basis_rlu
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

    dialog._set_model(two_row_helix_model())
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
    model = two_row_helix_model()
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


def test_magnetic_structure_builder_adds_a_preset_component(qapp, tmp_path, monkeypatch) -> None:
    dialog = _builder_with_components(tmp_path)
    dialog._set_model(two_row_helix_model())

    def _accept_spin_density_wave(preset_dialog):
        select_combo_data(preset_dialog._kind_combo, "spin_density_wave")
        select_combo_data(preset_dialog._propagation_combo, "a*")
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
