"""Tests for the Supercell Editor dialog UI and generation behavior."""

from __future__ import annotations

from functools import partial

from PySide6.QtWidgets import QDialog, QHeaderView

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

    saved = dialog._save_current()
    assert saved is not None
    dialog._use_selected()

    assert dialog.result() == dialog.DialogCode.Accepted
    assert dialog.selected_structure_id() == saved.id
    assert dialog.selected_matrix() == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert doc.structures[saved.id].label == "Relaxation 2x1x1"
    dialog.close()


def test_supercell_editor_asset_table_visible_columns_stretch(qapp, tmp_path) -> None:
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
    for column in (
        dialog._NAME_COLUMN,
        dialog._MATRIX_COLUMN,
        dialog._ATOMS_COLUMN,
        dialog._USAGE_COLUMN,
    ):
        assert header.sectionResizeMode(column) == QHeaderView.ResizeMode.Stretch
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
    saved = dialog._save_current()
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
