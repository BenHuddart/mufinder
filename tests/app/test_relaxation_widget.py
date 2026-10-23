"""Tests for pristine-relaxation widget supercell controls and layout sizing."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QDialog, QHeaderView, QMessageBox, QPushButton, QToolButton

from mufinder.app import workflow_table_utils
from mufinder.app import selection_snapshot as selection_snapshot_module
from mufinder.app.selection_snapshot import SelectionScope
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.relaxation_widget import PristineRelaxationWidget
from mufinder.app.workflow_parameters import kpoint_exclusive_hint
from mufinder.app.relaxation_types import RelaxationRunState
from mufinder.app.selection_snapshot import SelectionSnapshotPushButton, SelectionSnapshotToolButton
from mufinder.castep_io.types import ValueWithUnit
from tests.support.reset_menu import install_fake_menu, open_reset_menu


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
        selection_snapshot_module,
        "ask_selection_scope",
        lambda _parent, title, *_args: questions.append(title) or SelectionScope.ALL_SELECTED,
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
    # Simulate the press snapshot (real interaction captures it via mousePressEvent).
    combo.selection_snapshot_rows = (0, 1)
    combo.setCurrentIndex(combo.findText("convergence_cutoff 2x2x1"))

    expected = ((2, 0, 0), (0, 2, 0), (0, 0, 1))
    assert questions == ["Apply to selected rows?"]
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
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *_args: SelectionScope.CLICKED_ONLY,
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
    # Simulate the press snapshot.
    combo.selection_snapshot_rows = (0, 1)
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

    headers = [
        widget._table.horizontalHeaderItem(index).text()
        for index in range(widget._table.columnCount())
    ]

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
    widget._delete_shortcut.activated.emit()

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
    widget._backspace_shortcut.activated.emit()

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
    assert (
        names.index("basis_precision")
        < names.index("cut_off_energy")
        < names.index("elec_energy_tol")
        < names.index("kpoint_mp_grid")
    )


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


def test_relaxation_widget_pre_fills_spacing_equivalent_of_project_default_grid(qapp) -> None:
    # When the project has adopted a (4,4,4) grid on a cubic cell, the
    # configure-parameters dialog should pre-fill the kpoint_mp_spacing field
    # with the equivalent value (1/(a*(n-1)) for cubic) so the user can
    # toggle to Use spacing and see a sensible starting value, mirroring how
    # the basis_precision / cut_off_energy radio pair works.
    a = 4.026
    widget = PristineRelaxationWidget()
    widget.set_host_lattice([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
    widget.set_project_defaults(
        cutoff_energy=600.0,
        kpoint_grid=(4, 4, 4),
        kpoint_reference_lattice=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
    )
    widget._add_row()
    row_id = widget._row_id_at(0)
    assert row_id is not None

    # Dialog seeding pre-fills both fields so the radio toggle has a
    # sensible value to show when the user switches modes.
    seed = widget._common_params_for_rows((row_id,))
    assert seed["kpoint_mp_grid"] == (4, 4, 4)
    spacing = seed.get("kpoint_mp_spacing")
    assert spacing is not None
    expected_value = 1.0 / (a * 3.0)  # cubic spacing for (4,4,4)
    assert spacing.value == pytest.approx(expected_value)
    assert spacing.unit == "1/ang"

    # The row's effective values (used at validation/persistence time)
    # carry only ONE of the pair so the CASTEP mutual-exclusion check
    # doesn't fire when the row has not been explicitly configured yet.
    effective = widget._effective_params_for_row(row_id)
    assert effective["kpoint_mp_grid"] == (4, 4, 4)
    assert "kpoint_mp_spacing" not in effective


def test_relaxation_widget_does_not_pre_fill_spacing_when_no_project_grid(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget.set_host_lattice([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    widget.set_project_defaults(
        cutoff_energy=None,
        kpoint_grid=None,
        kpoint_reference_lattice=None,
    )
    widget._add_row()
    row_id = widget._row_id_at(0)
    assert row_id is not None

    seed = widget._common_params_for_rows((row_id,))
    assert "kpoint_mp_spacing" not in seed
    effective = widget._effective_params_for_row(row_id)
    assert "kpoint_mp_spacing" not in effective


def test_relaxation_widget_does_not_pre_fill_spacing_for_gamma_only(qapp) -> None:
    # Gamma-only (1,1,1) has no meaningful equivalent spacing, so the
    # spacing field stays unpopulated and the user can fill it themselves
    # if they want to toggle modes.
    widget = PristineRelaxationWidget()
    widget.set_host_lattice([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    widget.set_project_defaults(
        cutoff_energy=None,
        kpoint_grid=(1, 1, 1),
        kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )
    widget._add_row()
    row_id = widget._row_id_at(0)
    assert row_id is not None

    seed = widget._common_params_for_rows((row_id,))
    assert "kpoint_mp_spacing" not in seed
    effective = widget._effective_params_for_row(row_id)
    assert "kpoint_mp_spacing" not in effective


def test_relaxation_widget_set_project_defaults_keeps_cutoff_over_basis_precision(qapp) -> None:
    # cut_off_energy already has a schema default (340 eV) that wins over the
    # mutually-exclusive basis_precision default at construction time (see
    # PristineRelaxationWidget._default_shared_params /
    # _drop_default_conflicts). set_project_defaults() must preserve that
    # invariant when it overwrites cut_off_energy with the project's
    # determined value, not reintroduce a conflicting basis_precision -
    # see https://github.com/BenHuddart/mufinder/issues/24, where the
    # sibling KPointConvergenceWidget had exactly this defect because its
    # basic-visible default set never included cut_off_energy at
    # construction time.
    widget = PristineRelaxationWidget()
    assert "basis_precision" not in widget._default_params
    assert widget._default_params["cut_off_energy"] == ValueWithUnit(340.0, "eV")

    widget.set_project_defaults(
        cutoff_energy=800.0,
        kpoint_grid=None,
        kpoint_reference_lattice=None,
    )

    assert widget._default_params["cut_off_energy"] == ValueWithUnit(800.0, "eV")
    assert "basis_precision" not in widget._default_params


def test_relaxation_widget_unconfigured_row_builds_valid_config_after_cutoff_set(qapp) -> None:
    # End-to-end safety net for the user flow reported in issue #24: setting
    # a project cutoff energy and then running an unconfigured row must not
    # raise a "basis_precision conflicts with cut_off_energy" validation
    # error.
    widget = PristineRelaxationWidget()
    widget.set_project_defaults(
        cutoff_energy=800.0,
        kpoint_grid=None,
        kpoint_reference_lattice=None,
    )
    widget._add_row()

    config = widget._build_config_or_warn()

    assert config is not None
    assert config.rows[0].label == widget._label_at_row(0)


def test_relaxation_widget_inspect_dialog_shows_spacing_when_spacing_is_active(qapp) -> None:
    # Regression: when a row was configured with kpoint_mp_spacing, the
    # inspect/configure dialog must show spacing as the active exclusive
    # choice.  Before the fix the pre-fill of the inactive grid half caused
    # _default_selection_for_group to prefer kpoint_mp_grid (it was in
    # _PREFERRED_EXCLUSIVE_NAMES), making the dialog misleadingly show grid
    # as the active selection even though the files used spacing.
    a = 4.026
    widget = PristineRelaxationWidget()
    widget.set_host_lattice([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
    widget.set_project_defaults(
        cutoff_energy=600.0,
        kpoint_grid=(4, 4, 4),
        kpoint_reference_lattice=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
    )
    widget._add_row()
    row_id = widget._row_id_at(0)
    assert row_id is not None

    # Simulate the user having configured the row with spacing.
    widget._row_parameter_values[row_id] = {
        "kpoint_mp_spacing": ValueWithUnit(0.07, "1/ang"),
    }

    # Reproduce what the inspect/configure dialog open path does.
    common = widget._common_params_for_rows((row_id,))
    # The pre-fill adds kpoint_mp_grid as well; both are present.
    assert "kpoint_mp_spacing" in common
    assert "kpoint_mp_grid" in common

    kpoint_hint = kpoint_exclusive_hint(widget._effective_params_for_row(r) for r in (row_id,))

    dialog = ParameterDialog(
        specs=widget._specs,
        workflow_stage="pristine_relaxation",
        locked_values={"task": "GEOMETRYOPTIMIZATION"},
    )
    dialog.set_values(common, preferred_exclusive=kpoint_hint)

    # The dialog must select spacing, not grid.
    spacing_row = dialog._rows.get("kpoint_mp_spacing")
    grid_row = dialog._rows.get("kpoint_mp_grid")
    assert spacing_row is not None and spacing_row.radio is not None
    assert grid_row is not None and grid_row.radio is not None
    assert spacing_row.radio.isChecked(), "spacing radio should be active"
    assert not grid_row.radio.isChecked(), "grid radio should not be active"

    # collect_values must return spacing but not grid.
    values = dialog.collect_values()
    assert "kpoint_mp_spacing" in values
    assert "kpoint_mp_grid" not in values


def test_relaxation_widget_effective_params_never_emit_both_kpoint_keywords(qapp) -> None:
    # Regression: _effective_params_for_row is fed through validate_gui_values
    # at submit time, and the CASTEP validator flags kpoint_mp_grid and
    # kpoint_mp_spacing as mutually exclusive. Even with the project's
    # spacing-equivalent pre-fill enabled for dialog UX, the row's effective
    # values must never carry both simultaneously.
    a = 4.026
    widget = PristineRelaxationWidget()
    widget.set_host_lattice([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
    widget.set_project_defaults(
        cutoff_energy=600.0,
        kpoint_grid=(4, 4, 4),
        kpoint_reference_lattice=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
    )
    widget._add_row()
    row_id = widget._row_id_at(0)

    effective = widget._effective_params_for_row(row_id)
    assert not ("kpoint_mp_grid" in effective and "kpoint_mp_spacing" in effective)

    # When the user has explicitly chosen spacing via the dialog, only
    # spacing remains in effective_values — never grid alongside it.
    widget._row_parameter_values[row_id] = {
        "kpoint_mp_spacing": ValueWithUnit(0.07, "1/ang"),
    }
    effective = widget._effective_params_for_row(row_id)
    assert "kpoint_mp_spacing" in effective
    assert "kpoint_mp_grid" not in effective


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


def test_relaxation_widget_keeps_explicit_cutoff_override_when_project_defaults_change(
    qapp,
) -> None:
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


# ---------------------------------------------------------------------------
# Multi-row snapshot tests (Windows selection-collapse simulation)
# ---------------------------------------------------------------------------


def _select_rows(widget: PristineRelaxationWidget, rows: list[int]) -> None:
    selection = widget._table.selectionModel()
    assert selection is not None
    selection.clearSelection()
    for row in rows:
        selection.select(
            widget._table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )


def test_relaxation_supercell_uses_snapshot_when_selection_collapsed(qapp, monkeypatch):
    """Windows: selection collapses to one row before the slot fires.
    The handler must apply to the snapshot rows, not the live selection."""
    widget = PristineRelaxationWidget()
    widget.set_supercell_options([{"label": "2x2x1", "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]}])
    widget._add_row()
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)
    third_id = widget._row_id_at(2)

    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )

    _select_rows(widget, [0, 1, 2])
    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    assert isinstance(combo, QComboBox)
    # Simulate the press snapshot (before Windows collapses the selection)
    combo.selection_snapshot_rows = (0, 1, 2)
    # Simulate Windows collapsing the selection to just row 0 BEFORE slot fires
    _select_rows(widget, [0])

    combo.setCurrentIndex(combo.findText("2x2x1"))

    expected = ((2, 0, 0), (0, 2, 0), (0, 0, 1))
    assert widget.row_supercell_matrix(first_id) == expected
    assert widget.row_supercell_matrix(second_id) == expected
    assert widget.row_supercell_matrix(third_id) == expected


def test_relaxation_supercell_single_row_no_prompt(qapp, monkeypatch):
    widget = PristineRelaxationWidget()
    widget.set_supercell_options([{"label": "2x2x1", "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]}])
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)

    prompts: list = []
    monkeypatch.setattr(
        selection_snapshot_module, "ask_selection_scope", lambda *a, **kw: prompts.append(1)
    )

    _select_rows(widget, [0])
    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    assert isinstance(combo, QComboBox)
    combo.selection_snapshot_rows = (0,)
    combo.setCurrentIndex(combo.findText("2x2x1"))

    assert prompts == []
    assert widget.row_supercell_matrix(first_id) == ((2, 0, 0), (0, 2, 0), (0, 0, 1))
    assert widget.row_supercell_matrix(second_id) is None


def test_relaxation_supercell_click_unselected_row_no_prompt(qapp, monkeypatch):
    """Clicking a button in a row that is NOT in the selection → single-row only."""
    widget = PristineRelaxationWidget()
    widget.set_supercell_options([{"label": "2x2x1", "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]}])
    widget._add_row()
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)
    third_id = widget._row_id_at(2)

    prompts: list = []
    monkeypatch.setattr(
        selection_snapshot_module, "ask_selection_scope", lambda *a, **kw: prompts.append(1)
    )

    _select_rows(widget, [0, 1])  # rows 0, 1 selected
    combo = widget._table.cellWidget(2, widget._SUPERCELL_COLUMN)  # click row 2 (unselected)
    assert isinstance(combo, QComboBox)
    combo.selection_snapshot_rows = (0, 1)  # snapshot captured rows 0,1
    combo.setCurrentIndex(combo.findText("2x2x1"))

    assert prompts == []
    assert widget.row_supercell_matrix(third_id) == ((2, 0, 0), (0, 2, 0), (0, 0, 1))
    assert widget.row_supercell_matrix(first_id) is None
    assert widget.row_supercell_matrix(second_id) is None


def test_relaxation_configure_button_applies_to_snapshot_rows(qapp, monkeypatch):
    """Configure button: prompt then open dialog for all snapshot rows."""
    opened_for: list = []

    def _fake_dialog(row_ids, _widget=None):
        opened_for.extend(row_ids)

    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)
    third_id = widget._row_id_at(2)

    monkeypatch.setattr(widget, "_open_parameter_dialog", _fake_dialog)
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )

    btn = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(btn, SelectionSnapshotPushButton)
    btn.selection_snapshot_rows = (0, 1, 2)

    btn.clicked.emit()

    assert set(opened_for) == {first_id, second_id, third_id}


def test_relaxation_configure_button_single_row_no_prompt(qapp, monkeypatch):
    opened_for: list = []
    monkeypatch.setattr(
        PristineRelaxationWidget,
        "_open_parameter_dialog",
        lambda self, row_ids: opened_for.extend(row_ids),
    )
    prompts: list = []
    monkeypatch.setattr(
        selection_snapshot_module, "ask_selection_scope", lambda *a, **kw: prompts.append(1)
    )

    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)

    btn = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(btn, SelectionSnapshotPushButton)
    btn.selection_snapshot_rows = (0,)

    btn.clicked.emit()

    assert prompts == []
    assert first_id in opened_for


def test_relaxation_remove_button_applies_to_snapshot_rows(qapp, monkeypatch):
    removed: list = []
    monkeypatch.setattr(
        PristineRelaxationWidget,
        "_remove_rows_by_id",
        lambda self, row_ids: removed.extend(row_ids),
    )
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )

    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)

    btn = widget._table.cellWidget(0, widget._REMOVE_COLUMN)
    assert isinstance(btn, SelectionSnapshotToolButton)
    btn.selection_snapshot_rows = (0, 1)

    btn.clicked.emit()

    assert set(removed) == {first_id, second_id}


def _select_relaxation_rows(widget: PristineRelaxationWidget, rows: tuple[int, ...]) -> None:
    selection = widget._table.selectionModel()
    assert selection is not None
    selection.clearSelection()
    for row in rows:
        selection.select(
            widget._table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )


def test_relaxation_batch_configure_button_tracks_selection(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()

    assert widget._batch_configure_button.text() == "Configure Selected Parameters..."
    # Nothing selected on construction → disabled.
    widget._refresh_batch_button_state()
    assert widget._batch_configure_button.isEnabled() is False

    _select_relaxation_rows(widget, (0, 1))
    widget._refresh_batch_button_state()
    assert widget._batch_configure_button.isEnabled() is True

    widget._table.clearSelection()
    widget._refresh_batch_button_state()
    assert widget._batch_configure_button.isEnabled() is False


def test_relaxation_batch_configure_applies_to_all_selected_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    third_id = widget._row_id_at(2)
    unselected_id = widget._row_id_at(1)
    assert first_id is not None and third_id is not None and unselected_id is not None

    monkeypatch.setattr(
        ParameterDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        ParameterDialog,
        "collect_values",
        lambda _self: {"task": "GEOMETRYOPTIMIZATION", "max_scf_cycles": 77},
    )
    monkeypatch.setattr(
        ParameterDialog,
        "advanced_names",
        lambda _self: ("basis_precision",),
    )

    _select_relaxation_rows(widget, (0, 2))
    widget._configure_selected_rows()

    assert widget._row_parameter_values[first_id]["max_scf_cycles"] == 77
    assert widget._row_parameter_values[third_id]["max_scf_cycles"] == 77
    # Row 1 was not selected, so it must remain untouched.
    assert widget._row_parameter_values.get(unselected_id, {}) == {}


def test_relaxation_batch_configure_empty_selection_shows_info(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()

    infos: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )
    opened: list[int] = []
    monkeypatch.setattr(
        PristineRelaxationWidget,
        "_open_parameter_dialog",
        lambda _self, _ids: opened.append(1),
    )

    widget._table.clearSelection()
    widget._configure_selected_rows()

    assert infos == ["No rows selected"]
    assert opened == []


def test_relaxation_batch_configure_uses_selection_snapshot_not_collapsed(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The per-row Configure button snapshots the selection so a platform
    # focus-collapse to the clicked row does not shrink a multi-row batch.
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)
    assert first_id is not None and second_id is not None

    captured: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        PristineRelaxationWidget,
        "_open_parameter_dialog",
        lambda _self, ids: captured.append(ids),
    )
    # A multi-row snapshot prompts (via selection_snapshot.resolve_target_keys)
    # to confirm applying to all selected rows.

    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *_args, **_kwargs: SelectionScope.ALL_SELECTED,
    )

    _select_relaxation_rows(widget, (0, 1))
    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, SelectionSnapshotPushButton)
    # Simulate the press-time snapshot capturing both rows.
    configure_button.selection_snapshot_rows = (0, 1)
    configure_button.clicked.emit()

    assert captured, "per-row Configure did not open the dialog"
    assert set(captured[0]) == {first_id, second_id}


def test_relaxation_widget_context_menu_emits_reset_for_failed_row(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    label = widget._label_at_row(0)
    widget.set_runtime_states({label: RelaxationRunState(status="failed")})

    payloads: list[object] = []
    widget.reset_failed_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    open_reset_menu(widget._table, row=0)

    assert payloads == [[label]]


def test_relaxation_widget_context_menu_disables_reset_for_running_row(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    label = widget._label_at_row(0)
    widget.set_runtime_states({label: RelaxationRunState(status="running")})

    payloads: list[object] = []
    widget.reset_failed_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(widget._table, row=0)

    assert menu is not None and menu.reset_action().enabled is False
    assert payloads == []


def test_relaxation_widget_continue_action_defaults_disabled_when_unwired(
    qapp, monkeypatch
) -> None:
    # An unwired widget (e.g. a stray fixture in another test) never offers an
    # action it cannot fulfil, whatever the row's displayed status is.
    widget = PristineRelaxationWidget()
    widget._add_row()
    label = widget._label_at_row(0)
    widget.set_runtime_states({label: RelaxationRunState(status="completed")})

    install_fake_menu(monkeypatch)
    menu = open_reset_menu(widget._table, row=0)

    assert menu is not None
    action = menu.action("Continue relaxation…")
    assert action is not None
    assert action.enabled is False


def test_relaxation_widget_continue_action_uses_injected_predicate(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    label = widget._label_at_row(0)
    widget.set_runtime_states({label: RelaxationRunState(status="completed")})

    seen_keys: list[object] = []

    def predicate(key: object) -> bool:
        seen_keys.append(key)
        return key == label

    widget.set_continuable_predicate(predicate)

    install_fake_menu(monkeypatch)
    menu = open_reset_menu(widget._table, row=0)

    assert menu is not None
    action = menu.action("Continue relaxation…")
    assert action is not None
    assert action.enabled is True
    assert seen_keys == [label]


def test_relaxation_widget_continue_action_disabled_when_multiple_rows_selected(
    qapp, monkeypatch
) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    widget.set_continuable_predicate(lambda _key: True)
    for row in range(2):
        label = widget._label_at_row(row)
        widget.set_runtime_states({label: RelaxationRunState(status="completed")})

    install_fake_menu(monkeypatch)
    widget._table.selectAll()
    menu = open_reset_menu(widget._table, row=0)

    assert menu is not None
    action = menu.action("Continue relaxation…")
    assert action is not None
    assert action.enabled is False


def test_relaxation_widget_continue_action_emits_row_label(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    label = widget._label_at_row(0)
    widget.set_continuable_predicate(lambda _key: True)

    payloads: list[object] = []
    widget.continue_relaxation_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(widget._table, row=0)
    action = menu.action("Continue relaxation…")
    assert action is not None
    action.triggered.emit()

    assert payloads == [label]


def test_relaxation_widget_continue_dialog_seed_mirrors_row_parameter_dialog(qapp) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    label = widget._label_at_row(0)
    row_id = widget._row_id_at(0)
    widget._row_parameter_values[row_id] = {"geom_max_iter": 250}
    widget._row_added_advanced[row_id] = {"geom_max_iter"}

    seed = widget.continue_relaxation_dialog_seed(label)

    assert seed is not None
    assert seed["workflow_stage"] == "pristine_relaxation"
    assert seed["specs"] == widget._specs
    assert seed["parameter_values"]["geom_max_iter"] == 250
    assert seed["added_advanced"] == ("geom_max_iter",)
    assert seed["locked_values"]["task"] == "GEOMETRYOPTIMIZATION"
    assert seed["species_options"] == widget._parameter_species_options

    assert widget.continue_relaxation_dialog_seed("does-not-exist") is None


def test_relaxation_widget_uses_project_xc_functional_for_unconfigured_rows(qapp) -> None:
    widget = PristineRelaxationWidget()
    schema_default = widget._default_params["xc_functional"]
    widget._add_row()
    row_id = widget._row_id_at(0)
    assert row_id is not None

    widget.set_project_defaults(
        cutoff_energy=None,
        kpoint_grid=None,
        kpoint_reference_lattice=None,
        xc_functional="PBE",
    )
    assert widget._effective_params_for_row(row_id)["xc_functional"] == "PBE"

    # A row configured with its own functional keeps it when the default changes.
    widget._row_parameter_values[row_id] = {"xc_functional": "PBESOL"}
    widget.set_project_defaults(
        cutoff_energy=None,
        kpoint_grid=None,
        kpoint_reference_lattice=None,
        xc_functional="LDA",
    )
    assert widget._effective_params_for_row(row_id)["xc_functional"] == "PBESOL"

    widget._row_parameter_values.pop(row_id)
    widget.set_project_defaults(cutoff_energy=None, kpoint_grid=None, kpoint_reference_lattice=None)
    assert widget._effective_params_for_row(row_id)["xc_functional"] == schema_default


def test_relaxation_fix_cell_applies_to_selected_rows(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    for _ in range(3):
        widget._add_row()
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )
    states: list[dict] = []
    widget.form_state_changed.connect(states.append)

    _select_relaxation_rows(widget, (0, 1))
    combo = widget._table.cellWidget(0, widget._FIX_CELL_COLUMN)
    assert isinstance(combo, QComboBox)
    combo.selection_snapshot_rows = (0, 1)
    combo.setCurrentIndex(combo.findData(True))

    assert widget._fix_all_cell_at_row(0) is True
    assert widget._fix_all_cell_at_row(1) is True
    assert widget._fix_all_cell_at_row(2) is False
    # One state emission for the whole batch, not one per propagated row.
    assert len(states) == 1


def test_relaxation_fix_cell_this_row_only(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.CLICKED_ONLY,
    )

    _select_relaxation_rows(widget, (0, 1))
    combo = widget._table.cellWidget(0, widget._FIX_CELL_COLUMN)
    combo.selection_snapshot_rows = (0, 1)
    combo.setCurrentIndex(combo.findData(True))

    assert widget._fix_all_cell_at_row(0) is True
    assert widget._fix_all_cell_at_row(1) is False


def test_relaxation_fix_cell_cancel_restores_previous_value(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    monkeypatch.setattr(
        selection_snapshot_module, "ask_selection_scope", lambda *a, **kw: SelectionScope.CANCEL
    )
    states: list[dict] = []
    widget.form_state_changed.connect(states.append)

    _select_relaxation_rows(widget, (0, 1))
    combo = widget._table.cellWidget(0, widget._FIX_CELL_COLUMN)
    # Emulate the press: the combo records the value the user started from.
    combo.pre_interaction_index = combo.currentIndex()
    combo.selection_snapshot_rows = (0, 1)
    combo.setCurrentIndex(combo.findData(True))

    assert widget._fix_all_cell_at_row(0) is False
    assert widget._fix_all_cell_at_row(1) is False
    assert states == []


def test_relaxation_fix_cell_skips_locked_rows(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    for _ in range(3):
        widget._add_row()
    locked_label = widget._label_at_row(2)
    widget.set_runtime_states({locked_label: RelaxationRunState(status="running")})
    bodies: list[str] = []
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda _parent, _title, body, *a: bodies.append(body) or SelectionScope.ALL_SELECTED,
    )

    _select_relaxation_rows(widget, (0, 1, 2))
    combo = widget._table.cellWidget(0, widget._FIX_CELL_COLUMN)
    combo.selection_snapshot_rows = (0, 1, 2)
    combo.setCurrentIndex(combo.findData(True))

    assert bodies == [
        "Set this cell constraint on 2 of 3 selected rows? 1 is not eligible and will be skipped."
    ]
    assert widget._fix_all_cell_at_row(0) is True
    assert widget._fix_all_cell_at_row(1) is True
    assert widget._fix_all_cell_at_row(2) is False


def test_relaxation_supercell_cancel_restores_combo_and_changes_nothing(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    widget.set_supercell_options([{"label": "2x2x1", "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]}])
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)
    monkeypatch.setattr(
        selection_snapshot_module, "ask_selection_scope", lambda *a, **kw: SelectionScope.CANCEL
    )

    _select_relaxation_rows(widget, (0, 1))
    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    combo.selection_snapshot_rows = (0, 1)
    combo.setCurrentIndex(combo.findText("2x2x1"))

    assert widget.row_supercell_matrix(first_id) is None
    assert widget.row_supercell_matrix(second_id) is None
    assert combo.currentText() == "Unit cell"


def test_relaxation_remove_button_cancel_removes_nothing(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    monkeypatch.setattr(
        selection_snapshot_module, "ask_selection_scope", lambda *a, **kw: SelectionScope.CANCEL
    )

    btn = widget._table.cellWidget(0, widget._REMOVE_COLUMN)
    btn.selection_snapshot_rows = (0, 1)
    btn.clicked.emit()

    assert widget._table.rowCount() == 2


def test_relaxation_delete_shortcut_confirms_and_declining_removes_nothing(
    qapp, monkeypatch, confirmed_destructive_actions
) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    widget.set_runtime_states({widget._label_at_row(1): RelaxationRunState(status="completed")})
    removed: list[list[str]] = []
    widget.remove_requested.connect(lambda labels: removed.append(list(labels)))
    _select_relaxation_rows(widget, (0, 1))

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)
    widget._backspace_shortcut.activated.emit()
    assert removed == []
    assert widget._table.rowCount() == 2

    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: confirmed_destructive_actions.append(prompt) or True,
    )
    widget._delete_shortcut.activated.emit()
    assert widget._table.rowCount() == 0
    assert confirmed_destructive_actions == [
        ("Remove rows", "Remove 2 rows?", "1 row already has a submitted job.", "Remove 2 rows")
    ]


def test_relaxation_remove_button_single_row_is_confirmed(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    answers = [False, True]
    monkeypatch.setattr(
        workflow_table_utils, "confirm_destructive", lambda *_a, **_k: answers.pop(0)
    )

    btn = widget._table.cellWidget(0, widget._REMOVE_COLUMN)
    btn.selection_snapshot_rows = (0,)
    btn.clicked.emit()
    assert widget._table.rowCount() == 2

    btn.selection_snapshot_rows = (0,)
    btn.clicked.emit()
    assert widget._table.rowCount() == 1
    assert answers == []


def test_relaxation_remove_button_multi_row_prompt_is_the_only_confirmation(
    qapp, monkeypatch, confirmed_destructive_actions
) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *a, **kw: SelectionScope.ALL_SELECTED,
    )

    btn = widget._table.cellWidget(0, widget._REMOVE_COLUMN)
    btn.selection_snapshot_rows = (0, 1)
    btn.clicked.emit()

    assert widget._table.rowCount() == 0
    assert confirmed_destructive_actions == []


def test_relaxation_row_menu_has_shared_core_then_continue(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(widget._table)

    assert [action.text for action in menu.actions] == [
        "Run Local",
        "Run on Cluster",
        "Cancel",
        "Inspect Failure",
        "Reset Failed Jobs...",
        "Configure...",
        "Remove",
        "Continue relaxation…",
    ]
    assert menu.separators_before == [7]


def test_relaxation_row_menu_cancel_and_inspect_emit_row_labels(qapp, monkeypatch) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    labels = [widget._label_at_row(0), widget._label_at_row(1)]
    cancelled: list[object] = []
    inspected: list[object] = []
    widget.cancel_requested.connect(cancelled.append)
    widget.inspect_failed_requested.connect(inspected.append)
    install_fake_menu(monkeypatch)

    widget.set_runtime_states({label: RelaxationRunState(status="running") for label in labels})
    widget._table.selectAll()
    open_reset_menu(widget._table).action("Cancel").triggered.emit()

    widget.set_runtime_states({label: RelaxationRunState(status="failed") for label in labels})
    menu = open_reset_menu(widget._table)
    menu.action("Inspect Failure").triggered.emit()

    assert cancelled == [labels]
    assert inspected == [labels]


def test_relaxation_run_buttons_and_remove_selected_follow_the_selection(
    qapp, confirmed_destructive_actions
) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()

    assert widget._run_local_button.text() == "Run All Locally"
    assert widget._run_cluster_button.text() == "Run All on Cluster"
    assert widget._selection_count_label.text() == ""
    assert not widget._remove_selected_button.isEnabled()

    _select_relaxation_rows(widget, (0, 1))
    assert widget._run_local_button.text() == "Run Selected Locally"
    assert widget._selection_count_label.text() == "2 selected"
    prop = workflow_table_utils.MULTI_ROW_TARGET_PROPERTY
    assert bool(widget._table.cellWidget(0, widget._SUPERCELL_COLUMN).property(prop)) is True
    assert bool(widget._table.cellWidget(0, widget._FIX_CELL_COLUMN).property(prop)) is True

    widget._remove_selected_button.click()
    assert widget._table.rowCount() == 0
    assert [prompt["text"] for prompt in confirmed_destructive_actions] == ["Remove 2 rows?"]


def test_relaxation_row_menu_acts_on_the_eligible_rows_of_a_mixed_selection(
    qapp, monkeypatch, confirmed_destructive_actions
) -> None:
    widget = PristineRelaxationWidget()
    for _ in range(3):
        widget._add_row()
    ready, running, failed = (widget._label_at_row(row) for row in range(3))
    widget.set_runtime_states(
        {
            running: RelaxationRunState(status="running"),
            failed: RelaxationRunState(status="failed"),
        }
    )
    reset: list[object] = []
    cancelled: list[object] = []
    widget.reset_failed_requested.connect(reset.append)
    widget.cancel_requested.connect(cancelled.append)
    install_fake_menu(monkeypatch)
    widget._table.selectAll()

    menu = open_reset_menu(widget._table)

    assert [action.text for action in menu.actions][:5] == [
        "Run Local (1 of 3)",
        "Run on Cluster (1 of 3)",
        "Cancel (1 of 3)",
        "Inspect Failure (1 of 3)",
        "Reset Failed Jobs... (1 of 3)",
    ]
    menu.action("Cancel (1 of 3)").triggered.emit()

    # The fake menu's exec() already clicked the enabled reset entry.
    assert reset == [[failed]]
    assert cancelled == [[running]]


def test_relaxation_batch_configure_keeps_each_rows_own_differing_values(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = PristineRelaxationWidget()
    widget._add_row()
    widget._add_row()
    first_id, second_id = widget._row_id_at(0), widget._row_id_at(1)
    assert "max_scf_cycles" in widget._specs_by_name
    widget._row_parameter_values[first_id] = {"max_scf_cycles": 80}
    widget._row_parameter_values[second_id] = {"max_scf_cycles": 120}
    other_xc = "PBE" if widget._default_params.get("xc_functional") != "PBE" else "LDA"
    mixed: list[set[str]] = []

    def _fake_exec(dialog) -> int:
        mixed.append(dialog.untouched_mixed_names())
        dialog._rows["xc_functional"].editor.set_value(other_xc)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ParameterDialog, "exec", _fake_exec)

    widget._open_parameter_dialog((first_id, second_id))

    assert mixed == [{"max_scf_cycles"}]
    first, second = (widget._row_parameter_values[row_id] for row_id in (first_id, second_id))
    assert (first["max_scf_cycles"], second["max_scf_cycles"]) == (80, 120)
    assert first["xc_functional"] == second["xc_functional"] == other_xc


def test_relaxation_row_explicit_value_wins_over_an_exclusive_default(qapp) -> None:
    # A row that chose basis_precision drops the project cut-off default it is
    # mutually exclusive with, so the row validates and runs on its own choice.
    widget = PristineRelaxationWidget()
    widget.set_project_defaults(
        cutoff_energy=800.0, kpoint_grid=None, kpoint_reference_lattice=None
    )
    widget._add_row()
    row_id = widget._row_id_at(0)
    widget._row_parameter_values[row_id] = {"basis_precision": "FINE"}

    effective = widget._effective_params_for_row(row_id)

    assert effective["basis_precision"] == "FINE"
    assert "cut_off_energy" not in effective
    assert widget._build_config_or_warn() is not None
