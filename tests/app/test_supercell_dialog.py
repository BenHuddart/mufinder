"""Tests for the supercell preview dialog UI and generation behavior."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QHeaderView

from mufinder.app.supercell_dialog import SupercellEditorDialog, SupercellPreviewDialog
from mufinder.app.supercell_suggestion_dialog import SupercellSuggestionDialog
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Structure
from mufinder.science.supercell_suggestion import SupercellCandidate, SupercellSuggestion
from mufinder.services.project_service import ProjectService


def _host_structure() -> Structure:
    return Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"pbc": [True, True, True]},
    )


def test_supercell_dialog_simple_mode_matrix(qapp) -> None:
    dialog = SupercellPreviewDialog(
        structure=_host_structure(),
        atomic_spins=[1.0, -1.0],
        propagation_vector=[0.0, 0.0, 0.0],
        initial_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 3]],
    )

    assert not dialog._advanced_toggle.isChecked()
    assert dialog.matrix() == [[2, 0, 0], [0, 1, 0], [0, 0, 3]]
    assert dialog._viewer._atoms is not None
    assert len(dialog._viewer._atoms) == 12

    dialog.close()


def test_supercell_dialog_advanced_mode_matrix(qapp) -> None:
    dialog = SupercellPreviewDialog(
        structure=_host_structure(),
        atomic_spins=[1.0, -1.0],
        propagation_vector=[0.0, 0.0, 0.5],
        initial_matrix=[[1, 1, 0], [0, 1, 0], [0, 0, 1]],
    )

    assert dialog._advanced_toggle.isChecked()
    assert "background: transparent" in dialog._advanced_toggle.styleSheet()
    assert dialog.matrix() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]

    dialog._advanced_cells[0][0].setValue(2)
    dialog._generate_preview()

    assert dialog._viewer._atoms is not None
    assert len(dialog._viewer._atoms) >= 2

    dialog.close()


def test_supercell_dialog_supports_unit_cell_action(qapp) -> None:
    dialog = SupercellPreviewDialog(
        structure=_host_structure(),
        atomic_spins=[1.0, -1.0],
        propagation_vector=[0.0, 0.0, 0.0],
        accept_button_text="Save Supercell",
        show_unit_cell_button=True,
    )

    dialog._accept_unit_cell()

    assert dialog.result() == dialog.DialogCode.Accepted
    assert dialog.result_mode() == "unit_cell"
    dialog.close()


def test_supercell_dialog_shutdowns_viewer_on_reject(qapp, monkeypatch) -> None:
    dialog = SupercellPreviewDialog(
        structure=_host_structure(),
        atomic_spins=[1.0, -1.0],
        propagation_vector=[0.0, 0.0, 0.0],
    )
    calls: list[bool] = []
    monkeypatch.setattr(dialog._viewer, "shutdown", lambda: calls.append(True))

    dialog.reject()

    assert calls == [True]
    dialog.close()


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
    assert doc.structures[asset.id].properties["supercell_asset"]["matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
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
            SupercellCandidate(scaling=(3, 3, 3), atom_count=54, min_image_distance=12.0, satisfies_criterion=True),
        ),
        recommended_index=0,
        primitive_atom_count=2,
        distance_target=10.0,
        display_threshold=8.0,
        max_atoms=200,
        primitive_exceeds_budget=False,
    )

    def fake_exec(self):
        self._chosen_scaling = self._suggestion.candidates[self._suggestion.recommended_index].scaling
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
