"""Tests for the Supercell Editor dialog UI and generation behavior."""

from __future__ import annotations

from collections import UserList
from functools import partial

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QHeaderView, QPushButton

from mufinder.app import theme as theme_module
from mufinder.app import workflow_table_utils
from mufinder.app.supercell_dialog import SupercellEditorDialog
from mufinder.app.supercell_suggestion_dialog import SupercellSuggestionDialog
from mufinder.app.workflow_table_utils import confirm_destructive as _real_confirm_destructive
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Structure
from mufinder.science.supercell import build_supercell_structure
from mufinder.science.supercell_suggestion import SupercellCandidate, SupercellSuggestion
from mufinder.services.project_service import ProjectService
from tests.support.qt import answer_next_message_box, assert_tab_leaves_table
from tests.support.structures import si_pair_host


_host_structure = partial(si_pair_host, label="Host", properties={"pbc": [True, True, True]})


def _project_with_saved_cells(tmp_path) -> dict[str, object]:
    """Editor arguments for a project with the saved cells ``Three`` and ``Two``."""

    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_open.mufinder", "EditorOpen")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    for label, scale in (("Three", 3), ("Two", 2)):
        service.create_workflow_supercell_asset(
            doc,
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            host_structure_id=host.id,
            matrix=[[scale, 0, 0], [0, scale, 0], [0, 0, scale]],
            label=label,
        )
    return {"document": doc, "project_service": service, "source_structure": host}


def test_supercell_editor_saves_named_asset_and_selects_it(qapp, tmp_path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_save.mufinder", "EditorSave")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    dialog = SupercellEditorDialog(
        document=doc,
        project_service=service,
        source_structure=host,
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        initial_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        select_mode=True,
    )
    dialog._name_edit.setText("Relaxation 2x1x1")

    saved = dialog._save_as_asset()
    assert saved is not None
    dialog._use_selected()

    assert dialog.result() == dialog.DialogCode.Accepted
    assert dialog.selected_structure_id() == saved.id
    assert dialog.selected_matrix() == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert doc.structures[saved.id].label == "Relaxation 2x1x1"
    dialog.close()


def test_supercell_editor_asset_table_gives_the_spare_width_to_the_name(qapp, tmp_path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_columns.mufinder", "EditorColumns")
    host = _host_structure()
    service.add_structure(doc, host)

    dialog = SupercellEditorDialog(
        document=doc,
        project_service=service,
        source_structure=host,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
    )

    header = dialog._asset_table.horizontalHeader()
    assert header.sectionResizeMode(dialog._ID_COLUMN) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(dialog._NAME_COLUMN) == QHeaderView.ResizeMode.Stretch
    for column in (dialog._MATRIX_COLUMN, dialog._ATOMS_COLUMN, dialog._USAGE_COLUMN):
        assert header.sectionResizeMode(column) == QHeaderView.ResizeMode.ResizeToContents
    dialog.close()


_LONG_NAME = "CoF2 production 3x3x3 for muon relaxation"


def test_supercell_editor_shows_a_long_name_whole_at_the_default_size(qapp, tmp_path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_long_name.mufinder", "EditorLongName")
    host = _host_structure()
    service.add_structure(doc, host)
    service.create_workflow_supercell_asset(
        doc,
        workflow_stage=None,
        host_structure_id=host.id,
        matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        label=_LONG_NAME,
    )

    dialog = SupercellEditorDialog(document=doc, project_service=service, source_structure=host)
    dialog.show()
    qapp.processEvents()

    table = dialog._asset_table
    name_item = table.item(0, dialog._NAME_COLUMN)
    assert name_item.toolTip() == _LONG_NAME
    assert table.columnWidth(dialog._NAME_COLUMN) > table.fontMetrics().horizontalAdvance(
        _LONG_NAME
    )
    dialog.close()


def test_supercell_editor_opens_on_the_saved_cell_the_seed_names(qapp, tmp_path) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)

    dialog = SupercellEditorDialog(**dialog_args, initial_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    dialog.show()
    qapp.processEvents()

    assert dialog.focusWidget() is dialog._name_edit
    assert not dialog._name_edit.hasSelectedText()
    table = dialog._asset_table
    assert table.item(table.currentRow(), dialog._NAME_COLUMN).text() == "Three"
    assert table.selectedItems()[0].row() == table.currentRow()
    assert dialog._name_edit.text() == "Three"
    assert dialog._save_button.isEnabled()
    assert dialog._delete_button.isEnabled()
    dialog.close()


def test_supercell_editor_opens_on_no_cell_when_the_seed_names_none(qapp, tmp_path) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)

    dialog = SupercellEditorDialog(**dialog_args, initial_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 1]])
    dialog.show()
    qapp.processEvents()

    table = dialog._asset_table
    assert table.currentItem() is None
    assert table.selectedItems() == []
    assert dialog.focusWidget() is dialog._name_edit
    assert dialog._name_edit.text() == ""
    assert dialog.matrix() == [[4, 0, 0], [0, 4, 0], [0, 0, 1]]
    assert not dialog._save_button.isEnabled()
    assert "Save As" in dialog._save_button.toolTip()
    assert dialog._save_as_button.isEnabled()
    assert not dialog._delete_button.isEnabled()
    dialog.close()


def test_supercell_editor_save_is_enabled_only_with_a_selected_cell(qapp, tmp_path) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    dialog = SupercellEditorDialog(**dialog_args)
    assert not dialog._save_button.isEnabled()

    dialog._asset_table.selectRow(0)
    assert dialog._save_button.isEnabled()
    assert dialog._save_button.toolTip() == ""

    dialog._new_button.click()
    assert not dialog._save_button.isEnabled()
    assert dialog._asset_table.currentItem() is None
    dialog.close()


def test_supercell_editor_save_as_selects_the_cell_it_adds(qapp, tmp_path) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    dialog = SupercellEditorDialog(**dialog_args, initial_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 1]])
    dialog._name_edit.setText("Four")

    dialog._save_as_button.click()

    table = dialog._asset_table
    assert table.item(table.currentRow(), dialog._NAME_COLUMN).text() == "Four"
    assert dialog._save_button.isEnabled()
    dialog.close()


def test_supercell_editor_return_in_a_field_presses_no_button(qapp, tmp_path) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    dialog = SupercellEditorDialog(**dialog_args, initial_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 1]])
    dialog.show()
    qapp.processEvents()
    dialog._name_edit.setText("Work in progress")

    QTest.keyClick(dialog._name_edit, Qt.Key.Key_Return)
    QTest.keyClick(dialog._simple_nx, Qt.Key.Key_Return)

    assert dialog.isVisible()
    assert dialog._name_edit.text() == "Work in progress"
    assert dialog.matrix() == [[4, 0, 0], [0, 4, 0], [0, 0, 1]]
    dialog.close()


def test_supercell_editor_return_on_a_focused_button_presses_it(qapp, tmp_path) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    dialog = SupercellEditorDialog(**dialog_args, initial_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 1]])
    dialog.show()
    qapp.processEvents()

    dialog._new_button.setFocus()
    QTest.keyClick(dialog._new_button, Qt.Key.Key_Return)
    assert dialog.matrix() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    dialog._simple_nx.setValue(4)
    dialog._new_button.setFocus()
    QTest.keyClick(dialog._new_button, Qt.Key.Key_Enter)
    assert dialog.matrix() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    dialog._close_button.setFocus()
    QTest.keyClick(dialog._close_button, Qt.Key.Key_Return)
    assert not dialog.isVisible()
    dialog.close()


def test_supercell_editor_holding_return_presses_a_button_once(qapp, tmp_path) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    dialog = SupercellEditorDialog(**dialog_args, initial_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 1]])
    dialog.show()
    qapp.processEvents()
    dialog._save_as_button.setFocus()
    rows = dialog._asset_table.rowCount()

    for auto_repeat in (False, True, True):
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Return,
            Qt.KeyboardModifier.NoModifier,
            autorep=auto_repeat,
        )
        QApplication.sendEvent(dialog._save_as_button, event)

    assert dialog._asset_table.rowCount() == rows + 1
    dialog.close()


def test_supercell_editor_save_as_with_an_empty_name_loads_the_name(qapp, tmp_path) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    dialog = SupercellEditorDialog(**dialog_args, initial_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 1]])
    assert dialog._name_edit.text() == ""

    dialog._save_as_button.click()

    assert dialog._name_edit.text() == "4x4x1 supercell"
    table = dialog._asset_table
    assert table.item(table.currentRow(), dialog._NAME_COLUMN).text() == "4x4x1 supercell"
    dialog.close()


def test_supercell_editor_save_with_an_empty_name_loads_the_name(qapp, tmp_path) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    dialog = SupercellEditorDialog(**dialog_args, initial_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    dialog._name_edit.setText("")

    dialog._save_button.click()

    assert dialog._name_edit.text() == "3x3x3 supercell"
    dialog.close()


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
def test_supercell_editor_scaling_boxes_fit_two_digits_under_the_theme(
    qapp, tmp_path, mode
) -> None:
    theme_module.apply_theme(qapp, mode)
    try:
        dialog = _make_editor(tmp_path)
        dialog.show()
        qapp.processEvents()

        # A line edit keeps a 2 px margin on each side of its text.
        for box in (dialog._simple_nx, dialog._simple_ny, dialog._simple_nz):
            edit = box.lineEdit()
            widest = max(edit.fontMetrics().horizontalAdvance(str(n)) for n in range(10, 65))
            assert edit.contentsRect().width() >= widest + 4
        dialog.close()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def test_supercell_editor_scaling_row_has_narrow_named_boxes(qapp, tmp_path) -> None:
    dialog = _make_editor(tmp_path)
    dialog.show()
    qapp.processEvents()

    boxes = (dialog._simple_nx, dialog._simple_ny, dialog._simple_nz)
    assert [box.accessibleName() for box in boxes] == ["Nx", "Ny", "Nz"]
    for box in boxes:
        assert box.width() < box.sizeHint().width()
        assert box.toolTip().endswith(".")
    # The toggle sits beneath the boxes, lined up with them.
    assert dialog._advanced_toggle.y() >= boxes[0].geometry().bottom()
    assert dialog._advanced_toggle.x() == boxes[0].x()
    dialog.close()


def test_supercell_editor_opens_on_the_cell_the_workflow_uses_among_equal_matrices(
    qapp, tmp_path
) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    doc, service = dialog_args["document"], dialog_args["project_service"]
    assets = {
        asset.label: asset
        for asset in service.list_workflow_supercell_assets(
            doc, host_structure_id=dialog_args["source_structure"].id
        )
    }
    service.create_workflow_supercell_asset(
        doc,
        workflow_stage=None,
        host_structure_id=dialog_args["source_structure"].id,
        matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        label="Newer three",
        reuse_existing=False,
    )
    service.use_workflow_supercell_asset(
        doc, workflow_stage=WorkflowStage.UEP, asset_structure_id=assets["Three"].structure_id
    )

    dialog = SupercellEditorDialog(
        **dialog_args,
        workflow_stage=WorkflowStage.UEP,
        initial_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    )
    assert dialog._name_edit.text() == "Three"
    dialog.close()

    dialog = SupercellEditorDialog(**dialog_args, initial_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    assert dialog._name_edit.text() == "Newer three"
    dialog.close()


def test_supercell_editor_use_selected_waits_for_a_changed_matrix_to_be_saved(
    qapp, tmp_path
) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    dialog = SupercellEditorDialog(
        **dialog_args, initial_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]], select_mode=True
    )
    assert dialog._use_button.isEnabled()

    dialog._simple_nx.setValue(4)
    assert not dialog._use_button.isEnabled()
    assert dialog._use_button.toolTip().endswith(".")

    dialog._simple_nx.setValue(3)
    assert dialog._use_button.isEnabled()
    dialog.close()


def _editor_on_cell_three(tmp_path, monkeypatch, *, stages, workflow_stage=None):
    """An editor on the saved cell ``Three``, which each of ``stages`` has selected."""

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", _real_confirm_destructive)
    dialog_args = _project_with_saved_cells(tmp_path)
    doc, service = dialog_args["document"], dialog_args["project_service"]
    asset = service.latest_workflow_supercell_asset(
        doc,
        host_structure_id=dialog_args["source_structure"].id,
        matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    )
    for stage in stages:
        service.use_workflow_supercell_asset(
            doc, workflow_stage=stage, asset_structure_id=asset.structure_id
        )
    dialog = SupercellEditorDialog(
        **dialog_args,
        workflow_stage=workflow_stage,
        initial_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    )
    return dialog, doc.structures[asset.structure_id]


def _stored_matrix(structure) -> list[list[int]]:
    return structure.properties["supercell_asset"]["matrix"]


_CUTOFF_AND_KPOINT = (WorkflowStage.CONVERGENCE_CUTOFF, WorkflowStage.CONVERGENCE_KPOINT)


def test_supercell_editor_asks_before_changing_a_cell_other_workflows_select(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog, cell = _editor_on_cell_three(tmp_path, monkeypatch, stages=_CUTOFF_AND_KPOINT)
    dialog._simple_nx.setValue(4)

    seen = answer_next_message_box(click="Change supercell")
    dialog._save_button.click()

    assert sorted(seen["buttons"]) == ["Cancel", "Change supercell"]
    assert seen["default"] == "Cancel"
    assert seen["escape"] == "Cancel"
    assert "Cutoff Convergence and K-Point Convergence" in seen["text"]
    assert _stored_matrix(cell) == [[4, 0, 0], [0, 3, 0], [0, 0, 3]]
    dialog.close()


def test_supercell_editor_cancelling_the_question_keeps_the_cell_and_the_edit(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog, cell = _editor_on_cell_three(tmp_path, monkeypatch, stages=_CUTOFF_AND_KPOINT)
    dialog._simple_nx.setValue(4)

    answer_next_message_box(click="Cancel")
    dialog._save_button.click()

    assert _stored_matrix(cell) == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    assert dialog.matrix() == [[4, 0, 0], [0, 3, 0], [0, 0, 3]]
    assert dialog._current_asset_id == cell.id
    dialog.close()


def test_supercell_editor_rejects_an_invalid_matrix_before_asking(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog, cell = _editor_on_cell_three(tmp_path, monkeypatch, stages=_CUTOFF_AND_KPOINT)
    dialog._advanced_toggle.setChecked(True)
    dialog._advanced_cells[0][0].setValue(0)
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda *_args, **_kwargs: pytest.fail("no question for a matrix that cannot be saved"),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        "mufinder.app.supercell_dialog.QMessageBox.warning",
        lambda _parent, title, _text: warnings.append(title),
    )
    dialog._save_button.click()

    assert warnings == ["Supercell unavailable"]
    assert _stored_matrix(cell) == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    dialog.close()


def test_supercell_editor_names_only_the_other_workflows(qapp, tmp_path, monkeypatch) -> None:
    dialog, _cell = _editor_on_cell_three(
        tmp_path,
        monkeypatch,
        stages=_CUTOFF_AND_KPOINT,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
    )
    dialog._simple_nx.setValue(4)

    seen = answer_next_message_box(click="Change supercell")
    dialog._save_button.click()

    assert "K-Point Convergence" in seen["text"]
    assert "Cutoff Convergence" not in seen["text"]
    dialog.close()


@pytest.mark.parametrize(
    ("stages", "workflow_stage"),
    [
        ((), None),
        ((WorkflowStage.CONVERGENCE_CUTOFF,), WorkflowStage.CONVERGENCE_CUTOFF),
    ],
)
def test_supercell_editor_saves_a_cell_only_this_workflow_uses_without_asking(
    qapp, tmp_path, monkeypatch, stages, workflow_stage
) -> None:
    dialog, cell = _editor_on_cell_three(
        tmp_path, monkeypatch, stages=stages, workflow_stage=workflow_stage
    )
    dialog._simple_nx.setValue(4)
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda *_args, **_kwargs: pytest.fail("no question expected"),
    )

    dialog._save_button.click()

    assert _stored_matrix(cell) == [[4, 0, 0], [0, 3, 0], [0, 0, 3]]
    dialog.close()


def test_supercell_editor_renaming_a_shared_cell_asks_nothing(qapp, tmp_path, monkeypatch) -> None:
    dialog, cell = _editor_on_cell_three(tmp_path, monkeypatch, stages=_CUTOFF_AND_KPOINT)
    dialog._name_edit.setText("Renamed")
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda *_args, **_kwargs: pytest.fail("no question expected"),
    )

    dialog._save_button.click()

    assert cell.label == "Renamed"
    dialog.close()


def _freeze(document, structure) -> None:
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        input_structure_ids=[structure.id],
    )
    document.calculations[definition.id] = definition


def test_supercell_editor_frozen_save_names_a_distinct_copy_in_the_status_line(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog, cell = _editor_on_cell_three(tmp_path, monkeypatch, stages=_CUTOFF_AND_KPOINT)
    _freeze(dialog._document, cell)
    dialog._simple_nx.setValue(4)
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda *_args, **_kwargs: pytest.fail("a copy changes no workflow"),
    )

    dialog._save_button.click()

    assert cell.label == "Three"
    assert _stored_matrix(cell) == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    assert dialog._name_edit.text() == "Three copy"
    assert dialog._current_asset_id != cell.id
    status = dialog._status_label.text()
    assert "Saved copy 'Three copy'" in status
    assert "'Three'" in status
    assert not status.endswith(".")
    table = dialog._asset_table
    assert table.item(table.currentRow(), dialog._NAME_COLUMN).text() == "Three copy"
    dialog.close()


def test_supercell_editor_second_frozen_copy_gets_its_own_name(qapp, tmp_path, monkeypatch) -> None:
    dialog, cell = _editor_on_cell_three(tmp_path, monkeypatch, stages=())
    _freeze(dialog._document, cell)

    names = []
    for scale in (4, 5):
        dialog._refresh_assets(select_id=cell.id)
        dialog._name_edit.setText(cell.label)
        dialog._apply_matrix([[scale, 0, 0], [0, 3, 0], [0, 0, 3]])
        dialog._save_button.click()
        names.append(dialog._name_edit.text())

    assert names == ["Three copy", "Three copy 2"]
    dialog.close()


def test_supercell_editor_frozen_copy_keeps_a_name_the_user_typed(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog, cell = _editor_on_cell_three(tmp_path, monkeypatch, stages=())
    _freeze(dialog._document, cell)
    dialog._simple_nx.setValue(4)
    dialog._name_edit.setText("Future cell")

    dialog._save_button.click()

    assert dialog._name_edit.text() == "Future cell"
    assert "Saved copy 'Future cell'" in dialog._status_label.text()
    dialog.close()


def test_supercell_editor_save_tooltip_no_longer_warns_about_sharing(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog, _cell = _editor_on_cell_three(tmp_path, monkeypatch, stages=_CUTOFF_AND_KPOINT)

    assert dialog._save_button.toolTip() == ""
    dialog.close()


def test_supercell_editor_automatic_name_follows_the_matrix(qapp, tmp_path) -> None:
    dialog = _make_editor(tmp_path)
    dialog._new_button.click()
    assert dialog._name_edit.text() == "1x1x1 supercell"

    dialog._simple_nx.setValue(3)
    assert dialog._name_edit.text() == "3x1x1 supercell"

    dialog._name_edit.setText("My cell")
    dialog._simple_nx.setValue(2)
    assert dialog._name_edit.text() == "My cell"
    dialog.close()


def test_supercell_editor_keeps_a_saved_name_that_reads_like_an_automatic_one(
    qapp, tmp_path
) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    service, doc = dialog_args["project_service"], dialog_args["document"]
    service.create_workflow_supercell_asset(
        doc,
        workflow_stage=None,
        host_structure_id=dialog_args["source_structure"].id,
        matrix=[[5, 0, 0], [0, 1, 0], [0, 0, 1]],
        label="1x1x1 supercell",
    )
    dialog = SupercellEditorDialog(**dialog_args)
    row = next(
        row
        for row in range(dialog._asset_table.rowCount())
        if dialog._asset_table.item(row, dialog._NAME_COLUMN).text() == "1x1x1 supercell"
    )

    dialog._asset_table.selectRow(row)

    assert dialog._name_edit.text() == "1x1x1 supercell"
    assert dialog.matrix() == [[5, 0, 0], [0, 1, 0], [0, 0, 1]]
    dialog.close()


def test_supercell_editor_saves_from_the_host_page_without_an_owner(qapp, tmp_path) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    dialog = SupercellEditorDialog(**dialog_args, initial_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 1]])

    saved = dialog._save_as_asset()

    assert saved.label == "4x4x1 supercell"
    assert saved.properties["supercell_asset"]["owner_workflow_stage"] is None
    dialog.close()


def test_supercell_editor_labels_matrices_through_the_shared_label(qapp, tmp_path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_labels.mufinder", "EditorLabels")
    host = _host_structure()
    service.add_structure(doc, host)
    for label, matrix in (
        ("Diagonal", [[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        ("Skewed", [[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
    ):
        service.create_workflow_supercell_asset(
            doc,
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            host_structure_id=host.id,
            matrix=matrix,
            label=label,
            reuse_existing=False,
        )

    dialog = SupercellEditorDialog(document=doc, project_service=service, source_structure=host)

    matrix_column = {
        dialog._asset_table.item(row, dialog._NAME_COLUMN).text(): dialog._asset_table.item(
            row, dialog._MATRIX_COLUMN
        ).text()
        for row in range(dialog._asset_table.rowCount())
    }
    assert matrix_column == {"Diagonal": "2x2x2", "Skewed": "[1 1 0; 0 1 0; 0 0 1]"}
    assert dialog._facts_card.facts()[0][0] == ("Matrix", "1x1x1")

    dialog._new_button.click()

    assert dialog._name_edit.text() == "1x1x1 supercell"
    dialog.close()


def test_supercell_editor_preview_propagates_the_host_spins_along_q(qapp, tmp_path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_modulated.mufinder", "EditorModulated")
    host = _host_structure(
        properties={
            "pbc": [True, True, True],
            "atomic_spins": [1.0, -1.0],
            "propagation_vector": [0.0, 0.0, 0.5],
        }
    )
    service.add_structure(doc, host)
    matrix = [[1, 1, 0], [0, 1, 0], [0, 0, 2]]
    _expanded, expected = build_supercell_structure(
        host, matrix, atomic_spins=[1.0, -1.0], propagation_vector=[0.0, 0.0, 0.5]
    )
    _expanded, unmodulated = build_supercell_structure(
        host, matrix, atomic_spins=[1.0, -1.0], propagation_vector=[0.0, 0.0, 0.0]
    )
    assert list(expected) != list(unmodulated)

    dialog = SupercellEditorDialog(
        document=doc, project_service=service, source_structure=host, initial_matrix=matrix
    )

    assert dialog._advanced_toggle.isChecked()
    assert dialog._viewer._spins.tolist() == list(expected)
    dialog.close()


def test_supercell_editor_matrix_edit_saves_copy_for_frozen_asset(qapp, tmp_path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_frozen.mufinder", "EditorFrozen")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    asset = service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        label="Frozen 2x1x1",
        reuse_existing=False,
    )
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        input_structure_ids=[asset.id],
    )
    doc.calculations[definition.id] = definition

    dialog = SupercellEditorDialog(
        document=doc,
        project_service=service,
        source_structure=host,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
    )
    dialog._refresh_assets(select_id=asset.id)
    dialog._simple_nx.setValue(3)
    dialog._name_edit.setText("Future 3x1x1")

    saved = dialog._save_current()

    assert saved is not None
    assert saved.id != asset.id
    assert doc.structures[asset.id].properties["supercell_asset"]["matrix"] == [
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    assert saved.properties["supercell_asset"]["matrix"] == [[3, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert saved.properties["supercell_asset"]["source_asset_structure_id"] == str(asset.id)
    dialog.close()


def test_supercell_editor_shutdowns_viewer_on_reject(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_reject.mufinder", "EditorReject")
    host = _host_structure()
    service.add_structure(doc, host)

    dialog = SupercellEditorDialog(
        document=doc,
        project_service=service,
        source_structure=host,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
    )
    calls: list[bool] = []
    monkeypatch.setattr(dialog._viewer, "shutdown", lambda: calls.append(True))

    dialog.reject()

    assert calls == [True]
    dialog.close()


def _make_editor(
    tmp_path,
    lattice_vectors: list[list[float]] | None = None,
) -> SupercellEditorDialog:
    service = ProjectService()
    doc = service.create_project(tmp_path / "suggest.mufinder", "SuggestTest")
    host = Structure(
        label="Host",
        lattice_vectors=lattice_vectors or [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"pbc": [True, True, True]},
    )
    service.add_structure(doc, host)
    return SupercellEditorDialog(
        document=doc,
        project_service=service,
        source_structure=host,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
    )


def test_supercell_editor_has_suggest_button(qapp, tmp_path) -> None:
    dialog = _make_editor(tmp_path)
    assert hasattr(dialog, "_suggest_button")
    assert dialog._suggest_button.text() == "Suggest…"
    dialog.close()


def test_supercell_editor_suggest_writes_scaling_to_matrix(qapp, tmp_path, monkeypatch) -> None:
    dialog = _make_editor(tmp_path)

    suggestion = SupercellSuggestion(
        candidates=(
            SupercellCandidate(
                scaling=(3, 3, 3), atom_count=54, min_image_distance=12.0, satisfies_criterion=True
            ),
        ),
        recommended_index=0,
        primitive_atom_count=2,
        distance_target=10.0,
        display_threshold=8.0,
        max_atoms=200,
        primitive_exceeds_budget=False,
    )

    def fake_exec(self):
        self._chosen_scaling = self._suggestion.candidates[
            self._suggestion.recommended_index
        ].scaling
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SupercellSuggestionDialog, "exec", fake_exec)

    from unittest.mock import patch

    with patch("mufinder.app.supercell_dialog.suggest_supercell", return_value=suggestion):
        dialog._suggest_supercell()

    assert dialog.matrix() == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    dialog.close()


def test_suggest_and_the_facts_card_give_a_sheared_host_the_same_image_distance(
    qapp, tmp_path, monkeypatch
) -> None:
    # -2 A1 + A2 = (2, 4, 0) Å is the unit cell's nearest image, beyond its adjacent cells.
    sheared = [[20.0, 0.0, 0.0], [42.0, 4.0, 0.0], [0.0, 0.0, 20.0]]
    dialog = _make_editor(tmp_path, lattice_vectors=sheared)
    shown: dict[str, str] = {}

    def choose_recommended(self):
        # The recommended row is preselected; Use This Supercell takes it.
        row = self._table.selectedItems()[0].row()
        shown["scaling"] = self._table.item(row, self._SCALING_COLUMN).text()
        shown["distance"] = self._table.item(row, self._DISTANCE_COLUMN).text()
        use = next(b for b in self.findChildren(QPushButton) if b.text() == "Use This Supercell")
        use.click()
        if self.chosen_scaling() is None:
            return QDialog.DialogCode.Rejected
        return QDialog.DialogCode.Accepted

    warnings: list[str] = []
    monkeypatch.setattr(SupercellSuggestionDialog, "exec", choose_recommended)
    monkeypatch.setattr(
        "mufinder.app.supercell_dialog.QMessageBox.warning",
        lambda _parent, _title, text, *_a: warnings.append(text),
    )

    dialog._suggest_button.click()

    assert warnings == []
    assert dialog.matrix() == [[1, 0, 0], [0, 3, 0], [0, 0, 1]]
    assert shown["scaling"].startswith("1x3x1")
    assert shown["distance"] == "13.42 Å ✓"
    assert dict(dialog._facts_card.facts()[0])["Image distance"] == "13.42 Å"
    dialog.close()


def test_supercell_editor_spinbox_range_allows_n_20(qapp, tmp_path, monkeypatch) -> None:
    # Regression: spinbox range was (1,12) so setValue(20) would silently clamp to 12.
    # _generate_preview is patched out to avoid rendering a 20³-cell structure in CI.
    dialog = _make_editor(tmp_path)
    monkeypatch.setattr(dialog, "_generate_preview", lambda: None)
    dialog._apply_matrix([[20, 0, 0], [0, 20, 0], [0, 0, 20]])
    assert dialog.matrix() == [[20, 0, 0], [0, 20, 0], [0, 0, 20]]
    dialog.close()


def test_supercell_editor_spinbox_range_allows_n_64(qapp, tmp_path, monkeypatch) -> None:
    # Regression: spinbox range must accommodate suggestions up to n=64.
    # _generate_preview is patched out to avoid rendering a 64³-cell structure in CI,
    # which causes a fatal VTK abort (524,288 atoms exceeds the offscreen renderer limit).
    dialog = _make_editor(tmp_path)
    monkeypatch.setattr(dialog, "_generate_preview", lambda: None)
    dialog._apply_matrix([[64, 0, 0], [0, 64, 0], [0, 0, 64]])
    assert dialog.matrix() == [[64, 0, 0], [0, 64, 0], [0, 0, 64]]
    dialog.close()


def _editor_with_saved_supercell(tmp_path, monkeypatch):
    # The autouse stub accepts every confirmation; these tests answer the real box.
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", _real_confirm_destructive)
    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_delete.mufinder", "EditorDelete")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    dialog = SupercellEditorDialog(
        document=doc,
        project_service=service,
        source_structure=host,
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        initial_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    dialog._name_edit.setText("Delete me")
    saved = dialog._save_as_asset()
    assert saved is not None
    return dialog, doc, saved


def test_supercell_delete_confirmation_defaults_to_cancel(qapp, tmp_path, monkeypatch) -> None:
    dialog, doc, saved = _editor_with_saved_supercell(tmp_path, monkeypatch)

    seen = answer_next_message_box()
    dialog._delete_current()

    assert sorted(seen["buttons"]) == ["Cancel", "Delete supercell"]
    assert seen["default"] == "Cancel"
    assert seen["escape"] == "Cancel"
    assert seen["text"] == "Delete saved supercell 'Delete me'?"
    assert saved.id in doc.structures
    dialog.close()


def test_supercell_delete_removes_the_asset_once_confirmed(qapp, tmp_path, monkeypatch) -> None:
    dialog, doc, saved = _editor_with_saved_supercell(tmp_path, monkeypatch)

    seen = answer_next_message_box(click="Delete supercell")
    dialog._delete_current()

    assert "missing" not in seen
    assert saved.id not in doc.structures
    dialog.close()


def _count_editor_previews(monkeypatch) -> list[list[list[int]]]:
    """Record the matrix of every Supercell Editor preview build."""

    previews: list[list[list[int]]] = []
    original = SupercellEditorDialog._generate_preview

    def counted(self) -> None:
        previews.append(self.matrix())
        original(self)

    monkeypatch.setattr(SupercellEditorDialog, "_generate_preview", counted)
    return previews


def test_supercell_editor_opens_with_one_preview(qapp, tmp_path, monkeypatch) -> None:
    previews = _count_editor_previews(monkeypatch)
    service = ProjectService()
    doc = service.create_project(tmp_path / "open_preview.mufinder", "OpenPreview")
    host = _host_structure()
    service.add_structure(doc, host)

    dialog = SupercellEditorDialog(
        document=doc,
        project_service=service,
        source_structure=host,
        initial_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    )

    assert previews == [[[3, 0, 0], [0, 3, 0], [0, 0, 3]]]
    dialog.close()


def test_supercell_editor_apply_matrix_previews_once(qapp, tmp_path, monkeypatch) -> None:
    dialog = _make_editor(tmp_path)
    previews = _count_editor_previews(monkeypatch)

    dialog._apply_matrix([[2, 1, 0], [0, 2, 0], [0, 0, 2]])

    assert previews == [[[2, 1, 0], [0, 2, 0], [0, 0, 2]]]
    assert dialog._advanced_toggle.isChecked()
    assert not dialog._advanced_group.isHidden()

    previews.clear()
    dialog._apply_matrix([[2, 0, 0], [0, 3, 0], [0, 0, 1]])

    assert previews == [[[2, 0, 0], [0, 3, 0], [0, 0, 1]]]
    assert not dialog._advanced_toggle.isChecked()
    assert dialog._advanced_group.isHidden()
    dialog.close()


def test_supercell_editor_selecting_saved_row_previews_once(qapp, tmp_path, monkeypatch) -> None:
    dialog, _doc, saved = _editor_with_saved_supercell(tmp_path, monkeypatch)
    dialog._new_button.click()
    previews = _count_editor_previews(monkeypatch)

    dialog._asset_table.selectRow(0)

    assert dialog._current_asset_id == saved.id
    assert previews == [[[2, 0, 0], [0, 1, 0], [0, 0, 1]]]
    dialog.close()


def test_supercell_editor_new_asset_previews_once(qapp, tmp_path, monkeypatch) -> None:
    dialog, _doc, _saved = _editor_with_saved_supercell(tmp_path, monkeypatch)
    previews = _count_editor_previews(monkeypatch)

    dialog._new_button.click()

    assert previews == [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    dialog.close()


def test_supercell_editor_delete_previews_once(qapp, tmp_path, monkeypatch) -> None:
    dialog, doc, saved = _editor_with_saved_supercell(tmp_path, monkeypatch)
    previews = _count_editor_previews(monkeypatch)
    answer_next_message_box(click="Delete supercell")

    dialog._delete_button.click()

    assert saved.id not in doc.structures
    assert previews == [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    dialog.close()


def test_supercell_editor_use_suggestion_previews_once(qapp, tmp_path, monkeypatch) -> None:
    dialog = _make_editor(tmp_path)
    previews = _count_editor_previews(monkeypatch)
    suggestion = SupercellSuggestion(
        candidates=(
            SupercellCandidate(
                scaling=(2, 3, 2), atom_count=24, min_image_distance=8.0, satisfies_criterion=True
            ),
        ),
        recommended_index=0,
        primitive_atom_count=2,
        distance_target=8.0,
        display_threshold=8.0,
        max_atoms=200,
        primitive_exceeds_budget=False,
    )

    def fake_exec(self):
        self._chosen_scaling = self._suggestion.candidates[0].scaling
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SupercellSuggestionDialog, "exec", fake_exec)
    monkeypatch.setattr(
        "mufinder.app.supercell_dialog.suggest_supercell", lambda *_a, **_k: suggestion
    )

    dialog._suggest_button.click()

    assert previews == [[[2, 0, 0], [0, 3, 0], [0, 0, 2]]]
    dialog.close()


def test_supercell_editor_hand_edit_still_previews_each_change(qapp, tmp_path, monkeypatch) -> None:
    dialog = _make_editor(tmp_path)
    previews = _count_editor_previews(monkeypatch)

    dialog._simple_nx.setValue(2)
    dialog._simple_ny.setValue(3)

    assert previews == [
        [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[2, 0, 0], [0, 3, 0], [0, 0, 1]],
    ]

    previews.clear()
    dialog._advanced_toggle.setChecked(True)
    dialog._advanced_cells[0][1].setValue(1)

    # Ticking carries the scaling into the matrix, so the first preview is the same cell.
    assert previews == [
        [[2, 0, 0], [0, 3, 0], [0, 0, 1]],
        [[2, 1, 0], [0, 3, 0], [0, 0, 1]],
    ]
    dialog.close()


def test_tab_leaves_the_saved_supercells_table(qapp, tmp_path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_tab.mufinder", "EditorTab")
    host = _host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    dialog = SupercellEditorDialog(
        document=doc,
        project_service=service,
        source_structure=host,
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        initial_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        select_mode=True,
    )
    try:
        assert_tab_leaves_table(dialog._asset_table)
    finally:
        dialog.close()


# --- The host, the facts card and the status line -------------------------------------------


def test_supercell_editor_subtitle_names_the_host_as_its_selector_does(qapp, tmp_path) -> None:
    dialog = SupercellEditorDialog(**_project_with_saved_cells(tmp_path))

    assert dialog._subtitle.text() == "Supercells of Original — Host"
    dialog.close()


def test_supercell_editor_subtitle_shows_the_host_name_as_plain_text(qapp, tmp_path) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_markup.mufinder", "EditorMarkup")
    host = _host_structure(label="Fe <b>trial</b>")

    dialog = SupercellEditorDialog(document=doc, project_service=service, source_structure=host)

    assert dialog._subtitle.text() == "Supercells of Fe <b>trial</b>"
    assert dialog._subtitle.textFormat() == Qt.TextFormat.PlainText
    dialog.close()


def test_supercell_editor_opens_a_malformed_matrix_on_the_unit_cell(qapp, tmp_path) -> None:
    dialog = SupercellEditorDialog(
        **_project_with_saved_cells(tmp_path),
        initial_matrix=[[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0]],
    )

    assert dialog.matrix() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    dialog.close()


def test_supercell_editor_opens_a_matrix_given_as_any_sequence(qapp, tmp_path) -> None:
    dialog = SupercellEditorDialog(
        **_project_with_saved_cells(tmp_path),
        initial_matrix=UserList([UserList([2, 0, 0]), (0, 2, 0), [0, 0, 2]]),
    )

    assert dialog.matrix() == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    dialog.close()


def test_supercell_editor_subtitle_names_a_host_outside_the_selector_by_its_label(
    qapp, tmp_path
) -> None:
    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_unlisted.mufinder", "EditorUnlisted")
    host = _host_structure(label="Loose cell")

    dialog = SupercellEditorDialog(document=doc, project_service=service, source_structure=host)

    assert dialog._subtitle.text() == "Supercells of Loose cell"
    dialog.close()


def test_supercell_editor_facts_card_states_the_previewed_cell(qapp, tmp_path) -> None:
    dialog = SupercellEditorDialog(
        **_project_with_saved_cells(tmp_path), initial_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 3]]
    )

    # The 4 Å host doubled along a and b and tripled along c: the nearest image is 8 Å away.
    assert dialog._facts_card.facts() == [
        [
            ("Matrix", "2x2x3"),
            ("Atoms", "24"),
            ("Image distance", "8.00 Å"),
            ("a", "8.0000 Å"),
            ("b", "8.0000 Å"),
            ("c", "12.0000 Å"),
        ]
    ]
    assert dialog._facts_card.hairlines() == []
    assert dialog._facts_card.accessibleName() == "Previewed supercell"
    assert dialog._status_label.text() == ""

    dialog._simple_nz.setValue(1)

    assert dict(dialog._facts_card.facts()[0])["Image distance"] == "4.00 Å"
    dialog.close()


def test_supercell_editor_failed_generation_keeps_the_matrix_fact_and_says_why(
    qapp, tmp_path
) -> None:
    dialog = SupercellEditorDialog(**_project_with_saved_cells(tmp_path))
    dialog._advanced_toggle.setChecked(True)

    dialog._advanced_cells[2][2].setValue(0)

    assert dialog._facts_card.facts() == [[("Matrix", "[1 0 0; 0 1 0; 0 0 0]")]]
    assert dialog._status_label.text().startswith("Supercell generation failed: ")
    dialog.close()


def test_supercell_editor_status_line_sits_beside_close_and_clears_on_the_next_edit(
    qapp, tmp_path
) -> None:
    dialog = SupercellEditorDialog(
        **_project_with_saved_cells(tmp_path), initial_matrix=[[2, 1, 0], [0, 1, 0], [0, 0, 1]]
    )
    dialog._name_edit.setText("Sheared")

    dialog._save_as_button.click()

    assert dialog._status_label.text() == "Saved 'Sheared' with 4 atoms"
    bottom_row = dialog.layout().itemAt(dialog.layout().count() - 1).layout()
    assert bottom_row.indexOf(dialog._status_label) == 0
    assert bottom_row.indexOf(dialog._close_button) >= 0

    dialog._advanced_cells[0][1].setValue(0)

    assert dialog._status_label.text() == ""
    dialog.close()


# --- Buttons and the Delete key ------------------------------------------------------------


def test_supercell_editor_new_and_delete_sit_under_the_saved_list(qapp, tmp_path) -> None:
    dialog = SupercellEditorDialog(**_project_with_saved_cells(tmp_path))
    list_panel = dialog._asset_table.parentWidget()

    for button in (dialog._new_button, dialog._delete_button):
        assert button.parentWidget() is list_panel
    for button in (dialog._save_button, dialog._save_as_button, dialog._suggest_button):
        assert button.parentWidget() is not list_panel
        assert button.parentWidget().title() == "Supercell"
    dialog.close()


@pytest.mark.parametrize("key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace])
def test_supercell_editor_delete_keys_on_the_saved_list_delete_after_confirming(
    qapp, tmp_path, monkeypatch, key
) -> None:
    dialog, doc, saved = _editor_with_saved_supercell(tmp_path, monkeypatch)
    dialog.show()
    dialog.activateWindow()
    assert QTest.qWaitForWindowActive(dialog)
    dialog._asset_table.setFocus()
    qapp.processEvents()
    assert dialog._asset_table.hasFocus()

    seen = answer_next_message_box(click="Delete supercell")
    try:
        QTest.keyClick(dialog._asset_table, key)
    finally:
        # A key the shortcut missed must not leave the responder for a later test's box.
        seen["_timer"].stop()

    assert seen["text"] == "Delete saved supercell 'Delete me'?"
    assert seen["default"] == "Cancel"
    assert saved.id not in doc.structures
    assert dialog._status_label.text() == "Deleted 'Delete me'"
    dialog.close()


@pytest.mark.parametrize("key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace])
def test_supercell_editor_delete_keys_leave_a_cell_configured_jobs_use(
    qapp, tmp_path, monkeypatch, key
) -> None:
    first, doc, saved = _editor_with_saved_supercell(tmp_path, monkeypatch)
    first.close()
    _freeze(doc, saved)
    dialog = SupercellEditorDialog(
        document=doc,
        project_service=first._project_service,
        source_structure=first._source_structure,
        initial_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert dialog._current_asset_id == saved.id
    assert not dialog._delete_button.isEnabled()
    dialog.show()
    dialog.activateWindow()
    assert QTest.qWaitForWindowActive(dialog)
    dialog._asset_table.setFocus()
    qapp.processEvents()
    assert dialog._asset_table.hasFocus()
    asked: list[object] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda *args, **_kwargs: asked.append(args) or False,
    )

    QTest.keyClick(dialog._asset_table, key)

    assert asked == []
    assert saved.id in doc.structures
    dialog.close()


def test_supercell_editor_backspace_in_the_name_edits_the_name(qapp, tmp_path, monkeypatch) -> None:
    dialog, doc, saved = _editor_with_saved_supercell(tmp_path, monkeypatch)
    dialog.show()
    dialog._name_edit.setFocus()
    dialog._name_edit.end(False)
    qapp.processEvents()
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda *_args, **_kwargs: pytest.fail("Backspace in a field deletes nothing"),
    )

    QTest.keyClick(dialog._name_edit, Qt.Key.Key_Backspace)

    assert dialog._name_edit.text() == "Delete m"
    assert saved.id in doc.structures
    dialog.close()


# --- Usage ---------------------------------------------------------------------------------


_ROWS_NOT_COUNTED = (
    "Rows on the Pristine Relaxation, Muon Relaxations and UEP pages pick their cell by its"
    " matrix and are not counted here."
)


def _usage(dialog: SupercellEditorDialog, name: str) -> tuple[str, str]:
    table = dialog._asset_table
    row = next(
        row
        for row in range(table.rowCount())
        if table.item(row, dialog._NAME_COLUMN).text() == name
    )
    item = table.item(row, dialog._USAGE_COLUMN)
    return item.text(), item.toolTip()


def test_supercell_editor_usage_names_the_workflows(qapp, tmp_path, monkeypatch) -> None:
    dialog, cell = _editor_on_cell_three(
        tmp_path, monkeypatch, stages=(WorkflowStage.CONVERGENCE_KPOINT,)
    )

    assert _usage(dialog, "Three") == (
        "Selected by K-Point Convergence",
        "K-Point Convergence has it selected. " + _ROWS_NOT_COUNTED,
    )
    assert _usage(dialog, "Two") == (
        "Not selected",
        "No workflow's simulation cell is this supercell and no configured job uses it. "
        + _ROWS_NOT_COUNTED,
    )
    dialog.close()


def test_supercell_editor_usage_counts_several_workflows_and_names_them_in_the_tooltip(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog, cell = _editor_on_cell_three(tmp_path, monkeypatch, stages=_CUTOFF_AND_KPOINT)

    assert _usage(dialog, "Three") == (
        "Selected by 2 workflows",
        "Cutoff Convergence and K-Point Convergence have it selected. " + _ROWS_NOT_COUNTED,
    )
    dialog.close()


def test_supercell_editor_usage_puts_configured_jobs_first(qapp, tmp_path, monkeypatch) -> None:
    first, cell = _editor_on_cell_three(
        tmp_path, monkeypatch, stages=(WorkflowStage.CONVERGENCE_KPOINT,)
    )
    first.close()
    _freeze(first._document, cell)

    dialog = SupercellEditorDialog(
        document=first._document,
        project_service=first._project_service,
        source_structure=first._source_structure,
    )

    assert _usage(dialog, "Three") == (
        "Jobs in Cutoff Convergence",
        "Configured jobs of Cutoff Convergence use this supercell, so a changed matrix is"
        " saved as a copy. K-Point Convergence has it selected. " + _ROWS_NOT_COUNTED,
    )
    dialog.close()


# --- The propagation vector ----------------------------------------------------------------


def _editor_with_propagation(tmp_path, k_vector, matrix) -> SupercellEditorDialog:
    service = ProjectService()
    doc = service.create_project(tmp_path / "editor_k.mufinder", "EditorK")
    host = _host_structure(
        properties={
            "pbc": [True, True, True],
            "atomic_spins": [1.0, -1.0],
            "propagation_vector": list(k_vector),
        }
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    return SupercellEditorDialog(
        document=doc, project_service=service, source_structure=host, initial_matrix=matrix
    )


def test_supercell_editor_warns_when_the_cell_does_not_fit_the_propagation_vector(
    qapp, tmp_path
) -> None:
    dialog = _editor_with_propagation(tmp_path, (0.5, 0.0, 0.0), [[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    dialog.show()
    qapp.processEvents()
    assert not dialog._propagation_row.isVisible()

    dialog._simple_nx.setValue(3)

    assert dialog._propagation_row.isVisible()
    assert dialog._propagation_warning.text() == (
        "This supercell does not fit the propagation vector k = (0.5, 0, 0)."
    )
    assert dialog._propagation_warning.objectName() == "warningText"
    help_icon = dialog._propagation_help
    assert help_icon.accessibleName() == "About fitting the propagation vector"
    assert "cos(2π k·R)" in help_icon.help_text()
    assert "3x1x1" in help_icon.help_text()

    dialog._simple_nx.setValue(4)

    assert not dialog._propagation_row.isVisible()
    dialog.close()


def test_supercell_editor_checks_a_general_matrix_against_the_propagation_vector(
    qapp, tmp_path
) -> None:
    fitting = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]
    dialog = _editor_with_propagation(tmp_path, (0.5, 0.5, 0.0), fitting)
    dialog.show()
    qapp.processEvents()
    assert dialog._advanced_toggle.isChecked()
    assert not dialog._propagation_row.isVisible()

    dialog._advanced_cells[1][0].setValue(0)

    assert dialog._propagation_row.isVisible()
    dialog.close()


def test_supercell_editor_never_warns_without_a_propagation_vector(qapp, tmp_path) -> None:
    dialog = _editor_with_propagation(tmp_path, (0.0, 0.0, 0.0), [[3, 0, 0], [0, 1, 0], [0, 0, 1]])
    dialog.show()
    qapp.processEvents()

    assert not dialog._propagation_row.isVisible()
    dialog.close()


def test_supercell_editor_saves_a_cell_that_does_not_fit_the_propagation_vector(
    qapp, tmp_path
) -> None:
    dialog = _editor_with_propagation(tmp_path, (0.5, 0.0, 0.0), [[3, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert not dialog._propagation_row.isHidden()
    assert dialog._save_as_button.isEnabled()

    dialog._save_as_button.click()

    saved = dialog._document.structures[dialog._current_asset_id]
    assert saved.properties["supercell_asset"]["matrix"] == [[3, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert dialog._status_label.text() == f"Saved '{saved.label}' with 6 atoms"
    assert not dialog._propagation_row.isHidden()
    dialog.close()


# --- Wording, the general matrix and the saved list --------------------------------------


def test_supercell_editor_offers_a_general_matrix(qapp, tmp_path) -> None:
    dialog = _make_editor(tmp_path)

    assert dialog._advanced_toggle.text() == "General 3×3 matrix"
    dialog.close()


def _scaling(dialog: SupercellEditorDialog) -> list[int]:
    return [box.value() for box in (dialog._simple_nx, dialog._simple_ny, dialog._simple_nz)]


def _scaling_enabled(dialog: SupercellEditorDialog) -> list[bool]:
    return [box.isEnabled() for box in (dialog._simple_nx, dialog._simple_ny, dialog._simple_nz)]


def _general_editor(tmp_path, monkeypatch, matrix):
    dialog = SupercellEditorDialog(**_project_with_saved_cells(tmp_path), initial_matrix=matrix)
    return dialog, _count_editor_previews(monkeypatch)


def test_ticking_the_general_matrix_keeps_the_cell_and_disables_the_scaling(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog, previews = _general_editor(tmp_path, monkeypatch, [[2, 0, 0], [0, 1, 0], [0, 0, 3]])
    # A stale matrix in the hidden boxes must not reach the preview.
    dialog._advanced_cells[0][1].blockSignals(True)
    dialog._advanced_cells[0][1].setValue(5)
    dialog._advanced_cells[0][1].blockSignals(False)
    atoms = dict(dialog._facts_card.facts()[0])["Atoms"]

    dialog._advanced_toggle.setChecked(True)

    assert previews == [[[2, 0, 0], [0, 1, 0], [0, 0, 3]]]
    assert dict(dialog._facts_card.facts()[0])["Atoms"] == atoms
    assert _scaling_enabled(dialog) == [False, False, False]
    for box in (dialog._simple_nx, dialog._simple_ny, dialog._simple_nz):
        assert box.toolTip() == "The general 3×3 matrix sets the cell while it is ticked."
    assert [dialog._simple_nx.accessibleName(), dialog._simple_nz.accessibleName()] == [
        "Nx",
        "Nz",
    ]
    dialog.close()


def test_a_plain_scaling_in_the_general_matrix_shows_in_the_disabled_boxes(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog, previews = _general_editor(tmp_path, monkeypatch, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    dialog._advanced_toggle.setChecked(True)
    previews.clear()

    dialog._advanced_cells[2][2].setValue(4)

    assert _scaling(dialog) == [2, 2, 4]
    assert previews == [[[2, 0, 0], [0, 2, 0], [0, 0, 4]]]

    dialog._advanced_cells[0][1].setValue(1)

    assert _scaling(dialog) == [2, 2, 4]
    assert _scaling_enabled(dialog) == [False, False, False]
    assert dialog.matrix() == [[2, 1, 0], [0, 2, 0], [0, 0, 4]]
    dialog.close()


def test_disabled_scaling_boxes_never_drive_the_cell(qapp, tmp_path, monkeypatch) -> None:
    dialog, previews = _general_editor(tmp_path, monkeypatch, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    previews.clear()

    dialog._simple_nx.setValue(3)

    assert previews == []
    assert dialog.matrix() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    dialog.close()


def test_unticking_from_a_plain_scaling_keeps_the_cell(qapp, tmp_path, monkeypatch) -> None:
    dialog, previews = _general_editor(tmp_path, monkeypatch, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    dialog._advanced_toggle.setChecked(True)
    dialog._advanced_cells[0][0].setValue(3)
    previews.clear()

    dialog._advanced_toggle.setChecked(False)

    assert previews == [[[3, 0, 0], [0, 2, 0], [0, 0, 2]]]
    assert _scaling_enabled(dialog) == [True, True, True]
    assert dialog._simple_nx.toolTip() == "Nx: repeats of the unit cell along a."
    dialog.close()


def test_unticking_from_a_general_matrix_returns_to_the_scaling(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog, previews = _general_editor(tmp_path, monkeypatch, [[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    dialog._new_button.click()
    dialog._simple_nx.setValue(2)
    dialog._simple_ny.setValue(2)
    dialog._advanced_toggle.setChecked(True)
    dialog._advanced_cells[0][1].setValue(1)
    assert dialog._name_edit.text() == "[2 1 0; 0 2 0; 0 0 1] supercell"
    previews.clear()

    dialog._advanced_toggle.setChecked(False)

    assert previews == [[[2, 0, 0], [0, 2, 0], [0, 0, 1]]]
    assert dialog._name_edit.text() == "2x2x1 supercell"
    dialog.close()


_SHEARED = [[2, 1, 0], [0, 2, 0], [0, 0, 1]]


def _sheared_then_unticked(tmp_path, monkeypatch):
    """An editor whose general matrix ``_SHEARED`` was entered and then unticked."""

    dialog, previews = _general_editor(tmp_path, monkeypatch, [[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    dialog._new_button.click()
    dialog._simple_nx.setValue(2)
    dialog._simple_ny.setValue(2)
    dialog._advanced_toggle.setChecked(True)
    dialog._advanced_cells[0][1].setValue(1)
    dialog._advanced_toggle.setChecked(False)
    assert dialog.matrix() == [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    previews.clear()
    return dialog, previews


def test_ticking_again_restores_an_unticked_general_matrix(qapp, tmp_path, monkeypatch) -> None:
    dialog, previews = _sheared_then_unticked(tmp_path, monkeypatch)

    dialog._advanced_toggle.setChecked(True)

    assert dialog.matrix() == _SHEARED
    assert previews == [_SHEARED]
    assert _scaling_enabled(dialog) == [False, False, False]
    assert _scaling(dialog) == [2, 2, 1]
    assert dict(dialog._facts_card.facts()[0])["Matrix"] == "[2 1 0; 0 2 0; 0 0 1]"
    dialog.close()


def test_a_scaling_changed_while_unticked_keeps_the_general_matrix(
    qapp, tmp_path, monkeypatch
) -> None:
    dialog, _previews = _sheared_then_unticked(tmp_path, monkeypatch)
    dialog._simple_nz.setValue(3)

    dialog._advanced_toggle.setChecked(True)

    assert dialog.matrix() == _SHEARED

    dialog._advanced_toggle.setChecked(False)

    assert dialog.matrix() == [[2, 0, 0], [0, 2, 0], [0, 0, 3]]
    dialog.close()


def test_a_plain_scaling_is_not_remembered(qapp, tmp_path, monkeypatch) -> None:
    dialog, _previews = _general_editor(tmp_path, monkeypatch, [[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    dialog._advanced_toggle.setChecked(True)
    dialog._advanced_cells[2][2].setValue(3)
    dialog._advanced_toggle.setChecked(False)
    dialog._simple_nx.setValue(4)

    dialog._advanced_toggle.setChecked(True)

    assert dialog.matrix() == [[4, 0, 0], [0, 2, 0], [0, 0, 3]]
    dialog.close()


@pytest.mark.parametrize("load", ["new", "saved"])
def test_loading_a_cell_forgets_the_unticked_general_matrix(
    qapp, tmp_path, monkeypatch, load
) -> None:
    dialog, _previews = _sheared_then_unticked(tmp_path, monkeypatch)
    if load == "new":
        dialog._new_button.click()
        expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        table = dialog._asset_table
        row = next(
            row
            for row in range(table.rowCount())
            if table.item(row, dialog._NAME_COLUMN).text() == "Three"
        )
        table.selectRow(row)
        expected = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]

    dialog._advanced_toggle.setChecked(True)

    assert dialog.matrix() == expected
    dialog.close()


def test_a_remembered_singular_matrix_comes_back_as_it_was(qapp, tmp_path, monkeypatch) -> None:
    dialog, _previews = _general_editor(tmp_path, monkeypatch, [[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    dialog._new_button.click()
    dialog._advanced_toggle.setChecked(True)
    dialog._advanced_cells[2][2].setValue(0)
    assert dialog._status_label.text().startswith("Supercell generation failed: ")
    dialog._advanced_toggle.setChecked(False)
    assert dialog._status_label.text() == ""

    dialog._advanced_toggle.setChecked(True)

    assert dialog.matrix() == [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    assert dialog._facts_card.facts() == [[("Matrix", "[1 0 0; 0 1 0; 0 0 0]")]]
    assert dialog._status_label.text().startswith("Supercell generation failed: ")
    dialog.close()


def test_selecting_saved_cells_sets_the_matrix_mode(qapp, tmp_path, monkeypatch) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    dialog_args["project_service"].create_workflow_supercell_asset(
        dialog_args["document"],
        workflow_stage=None,
        host_structure_id=dialog_args["source_structure"].id,
        matrix=[[1, 1, 0], [0, 1, 0], [0, 0, 1]],
        label="Sheared",
    )
    dialog = SupercellEditorDialog(**dialog_args)
    previews = _count_editor_previews(monkeypatch)
    table = dialog._asset_table
    rows = {table.item(row, dialog._NAME_COLUMN).text(): row for row in range(table.rowCount())}

    table.selectRow(rows["Sheared"])

    assert dialog._advanced_toggle.isChecked()
    assert _scaling_enabled(dialog) == [False, False, False]
    assert len(previews) == 1

    table.selectRow(rows["Two"])

    assert not dialog._advanced_toggle.isChecked()
    assert _scaling_enabled(dialog) == [True, True, True]
    assert _scaling(dialog) == [2, 2, 2]
    assert len(previews) == 2
    dialog.close()


def test_a_saved_general_cell_sets_the_scaling_from_its_diagonal(qapp, tmp_path) -> None:
    dialog_args = _project_with_saved_cells(tmp_path)
    dialog_args["project_service"].create_workflow_supercell_asset(
        dialog_args["document"],
        workflow_stage=None,
        host_structure_id=dialog_args["source_structure"].id,
        matrix=[[1, 1, 0], [-1, 1, 0], [0, 0, 2]],
        label="Rotated",
    )
    dialog = SupercellEditorDialog(**dialog_args, initial_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    table = dialog._asset_table
    rows = {table.item(row, dialog._NAME_COLUMN).text(): row for row in range(table.rowCount())}

    table.selectRow(rows["Rotated"])
    dialog._advanced_toggle.setChecked(False)

    assert dialog.matrix() == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
    dialog.close()


def test_supercell_editor_status_line_shows_a_name_as_plain_text(qapp, tmp_path) -> None:
    dialog = SupercellEditorDialog(**_project_with_saved_cells(tmp_path))
    dialog._name_edit.setText("<b>bold</b>")

    dialog._save_as_button.click()

    assert dialog._status_label.textFormat() == Qt.TextFormat.PlainText
    assert dialog._status_label.text().startswith("Saved '<b>bold</b>'")
    dialog.close()


def test_supercell_editor_opens_a_diagonal_with_a_negative_entry_as_a_general_matrix(
    qapp, tmp_path
) -> None:
    mirrored = [[-1, 0, 0], [0, -1, 0], [0, 0, 2]]
    dialog = SupercellEditorDialog(**_project_with_saved_cells(tmp_path), initial_matrix=mirrored)

    assert dialog._advanced_toggle.isChecked()
    assert dialog.matrix() == mirrored
    assert dialog._facts_card.facts()[0][0] == ("Matrix", "[-1 0 0; 0 -1 0; 0 0 2]")
    dialog.close()


def test_supercell_editor_saved_list_columns_reach_the_right_edge(qapp, tmp_path) -> None:
    dialog = SupercellEditorDialog(**_project_with_saved_cells(tmp_path))
    dialog.show()
    qapp.processEvents()

    table = dialog._asset_table
    columns = sum(
        table.columnWidth(column)
        for column in range(table.columnCount())
        if not table.isColumnHidden(column)
    )
    assert table.verticalScrollBar().isVisible() is False
    assert columns == table.viewport().width()
    dialog.close()
