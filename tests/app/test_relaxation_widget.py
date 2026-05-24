"""Tests for pristine-relaxation widget supercell controls and layout sizing."""

from __future__ import annotations

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QHeaderView, QMessageBox, QPushButton, QToolButton

from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.relaxation_widget import PristineRelaxationWidget
from mufinder.app.relaxation_types import RelaxationRunState
from mufinder.castep_io.types import ValueWithUnit


def test_relaxation_widget_supercell_dropdown_lists_project_cells(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget.set_supercell_options(
        [
            {
                "label": "convergence_cutoff 2x2x1",
                "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            }
        ]
    )
    widget._add_row()

    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    assert isinstance(combo, QComboBox)
    labels = [combo.itemText(index) for index in range(combo.count())]
    assert labels[0] == "Unit cell"
    assert "convergence_cutoff 2x2x1" in labels
    assert labels[-1] == "Manage Supercells..."


def test_relaxation_widget_queued_row_locks_controls_and_parameters(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    label = widget._label_at_row(0)

    widget.set_runtime_states({label: RelaxationRunState(status="queued")})

    label_item = widget._table.item(0, widget._LABEL_COLUMN)
    assert label_item is not None
    assert not bool(label_item.flags() & Qt.ItemFlag.ItemIsEditable)
    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    assert isinstance(combo, QComboBox)
    assert not combo.isEnabled()
    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)
    assert configure_button.text() == "Configured..."
    remove_button = widget._table.cellWidget(0, widget._REMOVE_COLUMN)
    assert isinstance(remove_button, QToolButton)
    assert remove_button.isEnabled()
    row_id = widget._row_id_at(0)
    assert row_id is not None
    assert widget._row_parameters_locked((row_id,)) is True



def test_relaxation_widget_cancelled_row_keeps_parameters_editable(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    label = widget._label_at_row(0)

    widget.set_runtime_states({label: RelaxationRunState(status="cancelled")})

    label_item = widget._table.item(0, widget._LABEL_COLUMN)
    assert label_item is not None
    assert bool(label_item.flags() & Qt.ItemFlag.ItemIsEditable)
    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    assert isinstance(combo, QComboBox)
    assert combo.isEnabled()
    row_id = widget._row_id_at(0)
    assert row_id is not None
    assert widget._row_parameters_locked((row_id,)) is False


def test_relaxation_widget_supercell_dropdown_create_emits_signal(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget.set_supercell_options(
        [
            {
                "label": "convergence_cutoff 2x2x1",
                "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            }
        ]
    )
    widget._add_row()

    row_id = widget._row_id_at(0)
    assert row_id is not None

    requested: list[int] = []
    widget.row_supercell_edit_requested.connect(lambda payload: requested.append(int(payload)))

    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    assert isinstance(combo, QComboBox)
    combo.setCurrentIndex(combo.count() - 1)

    assert requested == [row_id]



def test_relaxation_widget_supercell_dropdown_selection_sets_matrix(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget.set_supercell_options(
        [
            {
                "label": "convergence_cutoff 2x2x1",
                "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            }
        ]
    )
    widget._add_row()

    row_id = widget._row_id_at(0)
    assert row_id is not None

    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    assert isinstance(combo, QComboBox)
    index = combo.findText("convergence_cutoff 2x2x1")
    assert index >= 0
    combo.setCurrentIndex(index)

    assert widget.row_supercell_matrix(row_id) == ((2, 0, 0), (0, 2, 0), (0, 0, 1))


def test_relaxation_widget_supercell_selection_can_apply_to_selected_rows(
    qapp,
    monkeypatch,
) -> None:
    widget = PristineRelaxationWidget()
    widget.set_supercell_options(
        [
            {
                "label": "convergence_cutoff 2x2x1",
                "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            }
        ]
    )
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)
    assert first_id is not None
    assert second_id is not None

    questions: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda _parent, title, *_args: questions.append(title) or QMessageBox.StandardButton.Yes,
    )

    selection = widget._table.selectionModel()
    assert selection is not None
    for row in (0, 1):
        selection.select(
            widget._table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    assert isinstance(combo, QComboBox)
    combo.setCurrentIndex(combo.findText("convergence_cutoff 2x2x1"))

    expected = ((2, 0, 0), (0, 2, 0), (0, 0, 1))
    assert questions == ["Apply supercell to selected rows?"]
    assert widget.row_supercell_matrix(first_id) == expected
    assert widget.row_supercell_matrix(second_id) == expected


def test_relaxation_widget_supercell_selection_can_stay_on_current_row(
    qapp,
    monkeypatch,
) -> None:
    widget = PristineRelaxationWidget()
    widget.set_supercell_options(
        [
            {
                "label": "convergence_cutoff 2x2x1",
                "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            }
        ]
    )
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)
    assert first_id is not None
    assert second_id is not None

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args: QMessageBox.StandardButton.No,
    )

    selection = widget._table.selectionModel()
    assert selection is not None
    for row in (0, 1):
        selection.select(
            widget._table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    assert isinstance(combo, QComboBox)
    combo.setCurrentIndex(combo.findText("convergence_cutoff 2x2x1"))

    assert widget.row_supercell_matrix(first_id) == ((2, 0, 0), (0, 2, 0), (0, 0, 1))
    assert widget.row_supercell_matrix(second_id) is None



def test_relaxation_widget_table_uses_stretch_columns(qapp) -> None:
    widget = PristineRelaxationWidget()
    header = widget._table.horizontalHeader()

    assert header.sectionResizeMode(widget._LABEL_COLUMN) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(widget._SUPERCELL_COLUMN) == QHeaderView.ResizeMode.Stretch


def test_relaxation_widget_omits_separate_kpoint_column(qapp) -> None:
    widget = PristineRelaxationWidget()

    headers = [widget._table.horizontalHeaderItem(index).text() for index in range(widget._table.columnCount())]

    assert "K-Point Grid" not in headers


def test_relaxation_widget_defaults_fix_cell_to_relax(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()

    combo = widget._table.cellWidget(0, widget._FIX_CELL_COLUMN)
    assert isinstance(combo, QComboBox)
    assert combo.currentData() is False


def test_relaxation_widget_delete_shortcut_removes_selected_jobs(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    removed: list[list[str]] = []
    widget.remove_requested.connect(lambda labels: removed.append(list(labels)))
    removed_label = widget._label_at_row(0)
    remaining_label = widget._label_at_row(1)

    widget._table.selectRow(0)
    widget._table_delete_shortcut.activated.emit()

    assert removed == [[removed_label]]
    assert widget._table.rowCount() == 1
    assert widget._label_at_row(0) == remaining_label


def test_relaxation_widget_backspace_shortcut_removes_selected_jobs(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    removed: list[list[str]] = []
    widget.remove_requested.connect(lambda labels: removed.append(list(labels)))
    removed_label = widget._label_at_row(0)

    widget._table.selectRow(0)
    widget._table_backspace_shortcut.activated.emit()

    assert removed == [[removed_label]]
    assert widget._table.rowCount() == 0


def test_relaxation_widget_delete_key_removes_selected_job_from_child_focus(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    removed: list[list[str]] = []
    widget.remove_requested.connect(lambda labels: removed.append(list(labels)))
    removed_label = widget._label_at_row(0)
    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)

    widget.show()
    widget._table.selectRow(0)
    configure_button.setFocus()
    qapp.processEvents()
    QTest.keyClick(configure_button, Qt.Key.Key_Delete)

    assert removed == [[removed_label]]
    assert widget._table.rowCount() == 1


def test_relaxation_widget_dialog_locks_write_cell_structure(qapp) -> None:
    widget = PristineRelaxationWidget()
    dialog = ParameterDialog(
        specs=widget._specs,
        workflow_stage="pristine_relaxation",
        locked_values={
            "task": "GEOMETRYOPTIMIZATION",
            "fix_all_cell": "Variable (set by row control)",
            "write_cell_structure": "TRUE (required for relaxed-cell outputs)",
        },
    )

    assert "write_cell_structure" in dialog._rows
    assert dialog._rows["write_cell_structure"].locked is True


def test_relaxation_widget_parameter_subset_matches_convergence_plus_geometry(qapp) -> None:
    widget = PristineRelaxationWidget()
    dialog = ParameterDialog(specs=widget._specs, workflow_stage="pristine_relaxation")

    names = set(dialog._visible_names())

    assert "cut_off_energy" in names
    assert "kpoint_mp_grid" in names
    assert "spin_polarized" in names
    assert "fix_all_cell" not in names
    assert "md_delta_t" not in names
    assert {"geom_energy_tol", "geom_force_tol", "geom_max_iter", "geom_stress_tol"} <= names


def test_relaxation_widget_defaults_continuation_to_default(qapp) -> None:
    widget = PristineRelaxationWidget()
    dialog = ParameterDialog(specs=widget._specs, workflow_stage="pristine_relaxation")
    dialog.set_values(widget._default_params)

    assert widget._default_params["continuation"] == "DEFAULT"
    assert dialog.collect_values()["continuation"] == "DEFAULT"

    dialog._rows["continuation"].editor.set_value("CUSTOM")
    dialog._reset_parameter("continuation")

    assert dialog.collect_values()["continuation"] == "DEFAULT"


def test_relaxation_widget_defaults_geometry_max_iterations_to_300(qapp) -> None:
    widget = PristineRelaxationWidget()
    dialog = ParameterDialog(specs=widget._specs, workflow_stage="pristine_relaxation")
    dialog.set_values(widget._default_params)

    assert widget._default_params["geom_max_iter"] == 300
    assert dialog.collect_values()["geom_max_iter"] == 300

    dialog._rows["geom_max_iter"].editor.set_value(30)
    dialog._reset_parameter("geom_max_iter")

    assert dialog.collect_values()["geom_max_iter"] == 300


def test_relaxation_widget_positions_kpoint_below_basis_controls(qapp) -> None:
    widget = PristineRelaxationWidget()
    dialog = ParameterDialog(specs=widget._specs, workflow_stage="pristine_relaxation")

    names = dialog._visible_names()

    assert "basis_precision" in names
    assert "cut_off_energy" in names
    assert "elec_energy_tol" in names
    assert "kpoint_mp_grid" in names
    assert names.index("basis_precision") < names.index("cut_off_energy") < names.index("elec_energy_tol") < names.index("kpoint_mp_grid")


def test_relaxation_widget_uses_project_default_kpoint_in_configure_values(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget.set_host_lattice([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    widget.set_project_defaults(
        cutoff_energy=600.0,
        kpoint_grid=(4, 4, 4),
        kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )
    widget._add_row()

    row_id = widget._row_id_at(0)
    assert row_id is not None

    params = widget._common_params_for_rows((row_id,))

    assert params["kpoint_mp_grid"] == (4, 4, 4)


def test_relaxation_widget_uses_exact_project_kpoint_for_unit_cell_when_reference_lattice_differs(
    qapp,
) -> None:
    widget = PristineRelaxationWidget()
    widget.set_host_lattice([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    widget.set_project_defaults(
        cutoff_energy=600.0,
        kpoint_grid=(3, 3, 3),
        kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
    )
    widget._add_row()

    row_id = widget._row_id_at(0)
    assert row_id is not None

    params = widget._common_params_for_rows((row_id,))

    assert params["kpoint_mp_grid"] == (3, 3, 3)


def test_relaxation_widget_uses_exact_project_kpoint_for_anisotropic_unit_cell_without_reference(
    qapp,
) -> None:
    widget = PristineRelaxationWidget()
    widget.set_host_lattice([[3.0, 0.0, 0.0], [0.0, 3.2, 0.0], [0.0, 0.0, 5.0]])
    widget.set_project_defaults(
        cutoff_energy=600.0,
        kpoint_grid=(3, 3, 3),
        kpoint_reference_lattice=None,
    )
    widget._add_row()

    row_id = widget._row_id_at(0)
    assert row_id is not None

    params = widget._common_params_for_rows((row_id,))

    assert params["kpoint_mp_grid"] == (3, 3, 3)


def test_relaxation_widget_applies_project_defaults_to_existing_rows(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    row_id = widget._row_id_at(0)
    assert row_id is not None

    widget.set_host_lattice([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    widget.set_project_defaults(
        cutoff_energy=650.0,
        kpoint_grid=(5, 5, 5),
        kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    config = widget._build_config_or_warn()

    assert config is not None
    assert config.rows[0].parameter_values == {}
    effective = widget._effective_params_for_row(row_id)
    assert effective["cut_off_energy"] == ValueWithUnit(650.0, "eV")
    assert effective["kpoint_mp_grid"] == (5, 5, 5)


def test_relaxation_widget_updates_effective_defaults_when_project_defaults_change(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    row_id = widget._row_id_at(0)
    assert row_id is not None
    widget.set_host_lattice([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])

    widget.set_project_defaults(
        cutoff_energy=500.0,
        kpoint_grid=(4, 4, 4),
        kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )
    first = widget._effective_params_for_row(row_id)
    assert first["cut_off_energy"] == ValueWithUnit(500.0, "eV")
    assert first["kpoint_mp_grid"] == (4, 4, 4)

    widget.set_project_defaults(
        cutoff_energy=700.0,
        kpoint_grid=(6, 6, 6),
        kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )
    second = widget._effective_params_for_row(row_id)
    assert second["cut_off_energy"] == ValueWithUnit(700.0, "eV")
    assert second["kpoint_mp_grid"] == (6, 6, 6)


def test_relaxation_widget_keeps_explicit_cutoff_override_when_project_defaults_change(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget.set_host_lattice([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])

    row_id = widget._row_id_at(0)
    assert row_id is not None
    widget._row_parameter_values[row_id] = {"cut_off_energy": ValueWithUnit(350.0, "eV")}

    widget.set_project_defaults(
        cutoff_energy=800.0,
        kpoint_grid=(4, 4, 4),
        kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    config = widget._build_config_or_warn()
    assert config is not None
    values = config.rows[0].parameter_values
    assert values["cut_off_energy"] == ValueWithUnit(350.0, "eV")

    effective = widget._effective_params_for_row(row_id)
    assert effective["cut_off_energy"] == ValueWithUnit(350.0, "eV")
    assert effective["kpoint_mp_grid"] == (4, 4, 4)
