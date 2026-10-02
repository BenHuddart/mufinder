"""Tests for k-point convergence configuration widget."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPushButton

import mufinder.app.kpoint_convergence_widget as kpoint_widget_module
from mufinder.app import workflow_table_utils
from mufinder.app.convergence_types import SweepPointRunState
from mufinder.app.kpoint_convergence_widget import KPointConvergenceWidget, KPointGridDialog, KPointSpacingDialog
from mufinder.castep_io.types import ValueWithUnit
from tests.support.reset_menu import install_fake_menu, open_reset_menu


def _table_grids(widget: KPointConvergenceWidget) -> list[tuple[int, int, int]]:
    values: list[tuple[int, int, int]] = []
    for row in range(widget._table.rowCount()):
        item = widget._table.item(row, widget._GRID_COLUMN)
        if item is None:
            continue
        values.append(widget._parse_grid(item.text()))
    return values


def test_kpoint_widget_generates_grids_from_spacing(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = KPointConvergenceWidget()
    widget.set_generation_context(
        structure_id="11111111-1111-1111-1111-111111111111",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]],
    )

    monkeypatch.setattr(
        kpoint_widget_module.KPointSpacingDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(kpoint_widget_module.KPointSpacingDialog, "spacing", lambda _self: 0.1)
    monkeypatch.setattr(kpoint_widget_module.KPointSpacingDialog, "replace_existing", lambda _self: True)
    monkeypatch.setattr(
        kpoint_widget_module.KPointSpacingDialog,
        "defaults_state",
        lambda _self: {"spacing": 0.1, "replace_existing": True},
    )

    widget._open_spacing_dialog()

    assert _table_grids(widget) == [(1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 2, 2)]
    state = widget.collect_state()
    assert state["generated_for_cell_id"] == "11111111-1111-1111-1111-111111111111"


def test_kpoint_widget_build_config_returns_kpoint_sweep(qapp) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget._add_grid((3, 3, 2))

    config = widget.build_config()

    assert config is not None
    assert config.sweep_parameter == "kpoint_mp_grid"
    assert config.sweep_values == ((2, 2, 2), (3, 3, 2))


def test_kpoint_widget_defaults_include_generate_symmetry(qapp) -> None:
    widget = KPointConvergenceWidget()

    assert widget._default_params["task"] == "SINGLEPOINT"
    assert widget._default_params["xc_functional"] == "LDA"
    assert widget._default_params["basis_precision"] == "FINE"
    assert widget._default_params["symmetry_generate"] is True


def test_kpoint_widget_warns_when_cell_changes_after_generation(qapp) -> None:
    widget = KPointConvergenceWidget()
    widget.set_generation_context(
        structure_id="a",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )
    widget._generation_lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    widget.set_generation_context(
        structure_id="b",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    assert widget._warning_label.isHidden() is False


def test_kpoint_widget_updates_runtime_and_examine_enablement(qapp) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget._add_grid((3, 3, 2))

    widget.set_runtime_states(
        {
            (2, 2, 2): SweepPointRunState(status="completed", total_energy=-10.0, max_force=0.2),
            (3, 3, 2): SweepPointRunState(status="completed", total_energy=-10.01, max_force=0.19),
        }
    )

    assert widget._examine_button.isEnabled() is True



def test_kpoint_spacing_dialog_round_trip_state(qapp) -> None:
    dialog = KPointSpacingDialog({"spacing": 0.25, "replace_existing": True})

    assert dialog.spacing() == pytest.approx(0.25)
    assert dialog.replace_existing() is True
    assert dialog.defaults_state() == {"spacing": pytest.approx(0.25), "replace_existing": True}


def test_kpoint_spacing_default_is_dense_sampling(qapp) -> None:
    dialog = KPointSpacingDialog()
    widget = KPointConvergenceWidget()

    assert dialog.spacing() == pytest.approx(0.02)
    assert widget.collect_state()["range"]["spacing"] == pytest.approx(0.02)

    widget.set_form_state({})

    assert widget.collect_state()["range"]["spacing"] == pytest.approx(0.02)


def test_kpoint_grid_dialog_enforces_integer_triplet(qapp) -> None:
    dialog = KPointGridDialog((2, 3, 4))

    assert dialog.grid() == (2, 3, 4)
    ok_button = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_button is not None
    assert ok_button.isEnabled() is True

    dialog._nx.setText("")
    assert ok_button.isEnabled() is False

    dialog._nx.setText("5")
    assert ok_button.isEnabled() is True
    assert dialog.grid() == (5, 3, 4)


def test_kpoint_widget_open_add_grid_dialog_adds_row_on_accept(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()

    monkeypatch.setattr(kpoint_widget_module.KPointGridDialog, "exec", lambda _self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(kpoint_widget_module.KPointGridDialog, "grid", lambda _self: (4, 4, 2))

    widget._open_add_grid_dialog()

    assert _table_grids(widget) == [(4, 4, 2)]


def test_kpoint_widget_open_add_grid_dialog_ignores_cancel(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()

    monkeypatch.setattr(kpoint_widget_module.KPointGridDialog, "exec", lambda _self: QDialog.DialogCode.Rejected)

    widget._open_add_grid_dialog()

    assert _table_grids(widget) == []


def test_kpoint_widget_applies_project_defaults_to_row_defaults(qapp) -> None:
    widget = KPointConvergenceWidget()
    schema_default = widget._default_params["xc_functional"]

    widget.set_xc_functional_default("PBE")
    widget.set_cutoff_default(450.0)

    assert widget._default_params["xc_functional"] == "PBE"
    widget.set_xc_functional_default(None)
    assert widget._default_params["xc_functional"] == schema_default
    assert widget._default_params["cut_off_energy"] == ValueWithUnit(450.0, "eV")
    assert "basis_precision" not in widget._default_params


def test_kpoint_widget_cutoff_default_clears_conflicting_basis_precision(qapp) -> None:
    widget = KPointConvergenceWidget()
    assert widget._default_params["basis_precision"] == "FINE"

    widget.set_cutoff_default(340.0)

    assert widget._default_params["cut_off_energy"] == ValueWithUnit(340.0, "eV")
    assert "basis_precision" not in widget._default_params


def test_kpoint_widget_unconfigured_grid_builds_valid_config_after_cutoff_set(qapp) -> None:
    # Regression test for https://github.com/BenHuddart/mufinder/issues/24:
    # an unconfigured grid row must not inherit both cut_off_energy and the
    # mutually-exclusive basis_precision schema default once a project
    # cutoff energy has been determined.
    widget = KPointConvergenceWidget()
    widget.set_cutoff_default(800.0)
    widget._add_grid((2, 2, 2))

    config = widget.build_config()

    assert config is not None
    assert config.shared_params["cut_off_energy"] == ValueWithUnit(800.0, "eV")
    assert "basis_precision" not in config.shared_params


def test_kpoint_widget_set_cutoff_default_none_leaves_basis_precision(qapp) -> None:
    widget = KPointConvergenceWidget()

    widget.set_cutoff_default(340.0)
    assert "basis_precision" not in widget._default_params

    widget.set_cutoff_default(None)

    assert "cut_off_energy" not in widget._default_params
    # basis_precision was dropped once cut_off_energy was set and is not
    # restored automatically when the cutoff default is cleared again; this
    # matches the widget's existing "explicit override wins" contract.
    assert "basis_precision" not in widget._default_params


def test_kpoint_widget_simulation_cell_combo_emits_expected_signals(qapp) -> None:
    widget = KPointConvergenceWidget()

    selected: list[str] = []
    used_unit_cell: list[bool] = []
    widget.simulation_cell_selection_requested.connect(lambda value: selected.append(str(value)))
    widget.simulation_cell_use_unit_cell_requested.connect(lambda: used_unit_cell.append(True))

    widget.set_simulation_cell_options(
        summary="ready",
        assets=[{"label": "2x2x2", "structure_id": "abc"}],
        selected_structure_id="abc",
        enabled=True,
    )
    widget._simulation_cell_combo.setCurrentIndex(0)
    widget._simulation_cell_combo.setCurrentIndex(1)

    assert used_unit_cell
    assert selected == ["abc"]


def test_kpoint_widget_open_spacing_dialog_requires_generation_context(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()

    warnings: list[str] = []
    monkeypatch.setattr(
        kpoint_widget_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    widget._open_spacing_dialog()

    assert warnings == ["No simulation cell"]


def test_kpoint_widget_open_spacing_dialog_handles_reject_and_generation_error(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()
    widget.set_generation_context(
        structure_id="host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    monkeypatch.setattr(
        kpoint_widget_module.KPointSpacingDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Rejected,
    )
    monkeypatch.setattr(
        kpoint_widget_module.KPointSpacingDialog,
        "defaults_state",
        lambda _self: {"spacing": 0.2, "replace_existing": True},
    )

    widget._open_spacing_dialog()
    assert widget._range_state == {"spacing": 0.2, "replace_existing": True}

    monkeypatch.setattr(
        kpoint_widget_module.KPointSpacingDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(kpoint_widget_module.KPointSpacingDialog, "spacing", lambda _self: 0.2)
    monkeypatch.setattr(kpoint_widget_module.KPointSpacingDialog, "replace_existing", lambda _self: True)
    monkeypatch.setattr(
        kpoint_widget_module,
        "generate_kpoint_grids",
        lambda _lattice, _spacing: (_ for _ in ()).throw(ValueError("bad spacing")),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        kpoint_widget_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    widget._open_spacing_dialog()

    assert warnings == ["Grid generation failed"]


def test_kpoint_widget_open_spacing_dialog_merges_existing_grids(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((1, 1, 1))
    widget.set_generation_context(
        structure_id="host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    monkeypatch.setattr(
        kpoint_widget_module.KPointSpacingDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(kpoint_widget_module.KPointSpacingDialog, "spacing", lambda _self: 0.15)
    monkeypatch.setattr(kpoint_widget_module.KPointSpacingDialog, "replace_existing", lambda _self: False)
    monkeypatch.setattr(
        kpoint_widget_module.KPointSpacingDialog,
        "defaults_state",
        lambda _self: {"spacing": 0.15, "replace_existing": False},
    )
    monkeypatch.setattr(
        kpoint_widget_module,
        "generate_kpoint_grids",
        lambda _lattice, _spacing: ((1, 1, 1), (2, 2, 2)),
    )

    widget._open_spacing_dialog()

    assert _table_grids(widget) == [(1, 1, 1), (2, 2, 2)]


def test_kpoint_widget_configure_selected_grids_handles_empty_selection(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()

    infos: list[str] = []
    monkeypatch.setattr(
        kpoint_widget_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    widget._configure_selected_grids()

    assert infos == ["No grids selected"]


def test_kpoint_widget_configure_button_reopens_after_first_use(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for issue #23: the per-row Configure button must stay
    clickable after being used once, regardless of whether the dialog was
    accepted or dismissed."""
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))

    exec_calls: list[int] = []

    def fake_exec(_self) -> int:
        exec_calls.append(1)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(kpoint_widget_module.ParameterDialog, "exec", fake_exec)

    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)

    configure_button.click()
    assert len(exec_calls) == 1

    configure_button.click()
    assert len(exec_calls) == 2, "Configure button stopped opening the dialog after first use"


def test_kpoint_widget_open_grid_parameter_dialog_applies_to_all_selected(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget._add_grid((3, 3, 2))

    monkeypatch.setattr(
        kpoint_widget_module.ParameterDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        kpoint_widget_module.ParameterDialog,
        "collect_values",
        lambda _self: {"task": "SINGLEPOINT", "xc_functional": "PBE", "max_scf_cycles": 75},
    )
    monkeypatch.setattr(
        kpoint_widget_module.ParameterDialog,
        "advanced_names",
        lambda _self: ("basis_precision",),
    )

    widget._open_grid_parameter_dialog(((2, 2, 2), (3, 3, 2)))

    assert widget._parameter_values[(2, 2, 2)]["xc_functional"] == "PBE"
    assert widget._parameter_values[(3, 3, 2)]["xc_functional"] == "PBE"
    assert widget._added_advanced[(2, 2, 2)] == {"basis_precision"}


def test_kpoint_widget_queued_grid_parameter_dialog_is_read_only(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget._parameter_values[(2, 2, 2)] = {"max_scf_cycles": 50}
    widget.set_runtime_states({(2, 2, 2): SweepPointRunState(status="queued")})

    observed: dict[str, object] = {}

    def fake_exec(dialog) -> int:
        observed["read_only"] = dialog._read_only
        observed["title"] = dialog.windowTitle()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(kpoint_widget_module.ParameterDialog, "exec", fake_exec)
    monkeypatch.setattr(
        kpoint_widget_module.ParameterDialog,
        "collect_values",
        lambda _self: {"task": "SINGLEPOINT", "max_scf_cycles": 99},
    )

    widget._open_grid_parameter_dialog(((2, 2, 2),))

    assert observed == {"read_only": True, "title": "Inspect K-Point Parameters"}
    assert widget._parameter_values[(2, 2, 2)]["max_scf_cycles"] == 50
    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)
    assert configure_button.text() == "Configured..."


def test_kpoint_widget_cancelled_grid_parameters_are_editable(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget._parameter_values[(2, 2, 2)] = {"max_scf_cycles": 50}
    widget.set_runtime_states({(2, 2, 2): SweepPointRunState(status="cancelled")})

    observed: dict[str, object] = {}

    def fake_exec(dialog) -> int:
        observed["read_only"] = dialog._read_only
        observed["title"] = dialog.windowTitle()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(kpoint_widget_module.ParameterDialog, "exec", fake_exec)
    monkeypatch.setattr(
        kpoint_widget_module.ParameterDialog,
        "collect_values",
        lambda _self: {"task": "SINGLEPOINT", "max_scf_cycles": 99},
    )

    widget._open_grid_parameter_dialog(((2, 2, 2),))

    assert observed == {"read_only": False, "title": "Configure K-Point Parameters"}
    assert widget._parameter_values[(2, 2, 2)]["max_scf_cycles"] == 99


def test_kpoint_widget_parameter_dialog_prefers_cutoff_and_shows_spin_polarized(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget.set_cutoff_default(340.0)

    observed: dict[str, object] = {}

    def fake_exec(dialog) -> int:
        observed["rows"] = set(dialog._rows)
        observed["cutoff_checked"] = bool(dialog._rows["cut_off_energy"].radio.isChecked())
        observed["spin_default"] = (
            dialog._rows["spin_polarized"].editor.value()  # type: ignore[union-attr]
            if dialog._rows["spin_polarized"].editor is not None
            else None
        )
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(kpoint_widget_module.ParameterDialog, "exec", fake_exec)

    widget._open_grid_parameter_dialog(((2, 2, 2),))

    rows = observed["rows"]
    assert isinstance(rows, set)
    assert "basis_precision" in rows
    assert "cut_off_energy" in rows
    assert "spin_polarized" in rows
    assert "symmetry_generate" in rows
    assert observed["cutoff_checked"] is True
    assert observed["spin_default"] is False


def test_kpoint_widget_parameter_dialog_places_kpoint_below_basis_controls(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget._added_advanced[(2, 2, 2)] = {"cut_off_energy"}

    observed: dict[str, object] = {}

    def fake_exec(dialog) -> int:
        names = dialog._visible_names()
        observed["names"] = names
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(kpoint_widget_module.ParameterDialog, "exec", fake_exec)

    widget._open_grid_parameter_dialog(((2, 2, 2),))

    names = observed.get("names")
    assert isinstance(names, list)
    assert "basis_precision" in names
    assert "cut_off_energy" in names
    assert "elec_energy_tol" in names
    assert "kpoint_mp_grid" in names
    assert names.index("basis_precision") < names.index("cut_off_energy") < names.index("elec_energy_tol") < names.index("kpoint_mp_grid")


def test_kpoint_widget_remove_grids_prunes_configs(qapp) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget._add_grid((3, 3, 2))
    widget._set_grid_parameter_values(
        (2, 2, 2),
        {"task": "SINGLEPOINT", "xc_functional": "PBE"},
        set(),
    )

    widget.remove_grids([(2, 2, 2)])

    assert (2, 2, 2) not in widget._parameter_values
    assert _table_grids(widget) == [(3, 3, 2)]


def test_kpoint_widget_delete_shortcut_requests_selected_grid_removal(qapp) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget._add_grid((3, 3, 2))
    requested: list[list[tuple[int, int, int]]] = []
    widget.remove_grid_requested.connect(lambda grids: requested.append(list(grids)))

    widget._table.selectRow(1)
    widget._delete_shortcut.activated.emit()

    assert requested == [[(3, 3, 2)]]


def test_kpoint_widget_backspace_shortcut_requests_current_grid_removal(qapp) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    requested: list[list[tuple[int, int, int]]] = []
    widget.remove_grid_requested.connect(lambda grids: requested.append(list(grids)))

    widget._table.setCurrentCell(0, widget._GRID_COLUMN)
    widget._backspace_shortcut.activated.emit()

    assert requested == [[(2, 2, 2)]]


def test_kpoint_widget_delete_key_requests_selected_grid_from_child_focus(qapp) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget._add_grid((3, 3, 2))
    requested: list[list[tuple[int, int, int]]] = []
    widget.remove_grid_requested.connect(lambda grids: requested.append(list(grids)))
    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)

    widget.show()
    widget._table.selectRow(0)
    configure_button.setFocus()
    qapp.processEvents()
    QTest.keyClick(configure_button, Qt.Key.Key_Delete)

    assert requested == [[(2, 2, 2)]]


def test_kpoint_widget_build_config_warns_for_invalid_grids_and_duplicates(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()
    warnings: list[str] = []
    monkeypatch.setattr(
        kpoint_widget_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    assert widget.build_config() is None

    widget._add_grid((2, 2, 2))
    widget._add_grid((2, 2, 2))
    assert widget.build_config() is None

    widget._table.item(0, widget._GRID_COLUMN).setText("bad")
    assert widget.build_config() is None

    assert warnings == ["No grids", "Duplicate grids", "Invalid grid list"]


def test_kpoint_widget_build_config_warns_for_invalid_parameter_values(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))

    warnings: list[str] = []
    monkeypatch.setattr(
        kpoint_widget_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )
    monkeypatch.setattr(
        kpoint_widget_module,
        "validate_gui_values",
        lambda *_args, **_kwargs: ["bad value"],
    )

    assert widget.build_config() is None
    assert warnings == ["Invalid parameters"]


def test_kpoint_widget_emit_and_submit_actions(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))

    configured: list[object] = []
    local_submits: list[list[tuple[int, int, int]]] = []
    cluster_submits: list[list[tuple[int, int, int]]] = []
    widget.configure_requested.connect(lambda payload: configured.append(payload))
    widget.submit_local_requested.connect(lambda payload: local_submits.append(list(payload)))
    widget.submit_cluster_requested.connect(lambda payload: cluster_submits.append(list(payload)))

    widget._emit_config()
    widget._run_local_selected_grids()
    widget._run_cluster_selected_grids()

    assert configured
    assert local_submits == [[(2, 2, 2)]]
    assert cluster_submits == [[(2, 2, 2)]]


def test_kpoint_widget_run_with_nothing_selected_runs_every_row(qapp) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget._add_grid((3, 3, 3))
    widget._table.clearSelection()
    local: list[object] = []
    cluster: list[object] = []
    widget.submit_local_requested.connect(local.append)
    widget.submit_cluster_requested.connect(cluster.append)

    assert widget._run_local_button.text() == "Run All Locally"
    assert widget._run_cluster_button.text() == "Run All on Cluster"
    widget._run_local_selected_grids()
    widget._run_cluster_selected_grids()

    assert local == [[(2, 2, 2), (3, 3, 3)]]
    assert cluster == [[(2, 2, 2), (3, 3, 3)]]

    widget._table.selectRow(1)
    assert widget._run_local_button.text() == "Run Selected Locally"
    assert widget._run_cluster_button.text() == "Run Selected on Cluster"
    widget._run_local_selected_grids()
    assert local[-1] == [(3, 3, 3)]


def test_kpoint_widget_set_form_state_restores_structured_values(qapp) -> None:
    widget = KPointConvergenceWidget()
    state = {
        "name_prefix": "kconv",
        "range": {"spacing": 0.2, "replace_existing": True},
        "kpoint_grids": [[2, 2, 2]],
        "kpoint_parameter_rows": [
            {
                "grid": [2, 2, 2],
                "params": {
                    "task": "SINGLEPOINT",
                    "cut_off_energy": {"value": 520.0, "unit": "eV"},
                    "kpoint_mp_grid": [2, 2, 2],
                    "lattice_cart": {"name": "lattice_cart", "rows": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                },
                "added_advanced": ["max_scf_cycles"],
            }
        ],
        "generated_for_cell_id": "host",
        "generated_for_lattice": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }

    widget.set_form_state(state)

    params = widget._parameter_values[(2, 2, 2)]
    assert isinstance(params["cut_off_energy"], ValueWithUnit)
    assert params["kpoint_mp_grid"] == (2, 2, 2)
    assert "lattice_cart" not in params
    assert widget._added_advanced[(2, 2, 2)] == {"max_scf_cycles"}


def test_kpoint_widget_context_menu_emits_reset_for_failed_row(qapp, monkeypatch) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget.set_runtime_states({(2, 2, 2): SweepPointRunState(status="failed")})

    payloads: list[object] = []
    widget.reset_failed_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    open_reset_menu(widget._table, row=0)

    assert payloads == [[(2, 2, 2)]]


def test_kpoint_widget_context_menu_disables_reset_for_completed_row(qapp, monkeypatch) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget.set_runtime_states({(2, 2, 2): SweepPointRunState(status="completed", total_energy=-1.0)})

    payloads: list[object] = []
    widget.reset_failed_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(widget._table, row=0)

    assert menu is not None and menu.reset_action().enabled is False
    assert payloads == []


def test_kpoint_widget_delete_shortcut_declined_requests_nothing(qapp, monkeypatch) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget._add_grid((3, 3, 2))
    requested: list[object] = []
    widget.remove_grid_requested.connect(requested.append)
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)

    widget._table.selectRow(1)
    widget._backspace_shortcut.activated.emit()

    assert requested == []


def test_kpoint_widget_removal_is_confirmed_once_per_request(
    qapp, confirmed_destructive_actions
) -> None:
    widget = KPointConvergenceWidget()
    widget._add_grid((2, 2, 2))
    widget._add_grid((3, 3, 2))
    requested: list[object] = []
    widget.remove_grid_requested.connect(lambda grids: requested.append(list(grids)))

    widget._request_grid_removals((0, 1))
    widget._request_grid_removals((0,), confirmed=True)

    assert requested == [[(2, 2, 2), (3, 3, 2)], [(2, 2, 2)]]
    assert [prompt["text"] for prompt in confirmed_destructive_actions] == ["Remove 2 rows?"]
