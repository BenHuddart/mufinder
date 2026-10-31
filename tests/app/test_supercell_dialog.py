"""Tests for the Supercell Editor dialog UI and generation behavior."""

from __future__ import annotations

from functools import partial

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QHeaderView

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
from tests.support.qt import answer_next_message_box
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
    assert dialog._status_label.text().startswith("Previewing 1x1x1 → ")

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


def _make_editor(tmp_path) -> SupercellEditorDialog:
    service = ProjectService()
    doc = service.create_project(tmp_path / "suggest.mufinder", "SuggestTest")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
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

    assert len(previews) == 2
    assert previews[-1] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    dialog.close()
