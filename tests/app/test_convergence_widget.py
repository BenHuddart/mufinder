"""Tests for cutoff convergence configuration widget."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QItemSelectionModel, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QMessageBox, QPushButton, QToolButton

import mufinder.app.convergence_widget as convergence_widget_module
from mufinder.app.convergence_types import ConvergenceConfig, CutoffRunState
from mufinder.app.convergence_widget import ConvergenceWidget, CutoffRangeDialog


def _table_values(widget: ConvergenceWidget) -> list[float]:
    values: list[float] = []
    for row in range(widget._cutoff_table.rowCount()):
        item = widget._cutoff_table.item(row, widget._CUTOFF_COLUMN)
        if item is not None:
            values.append(float(item.text()))
    return values


def test_cutoff_range_dialog_validates_and_returns_values(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = CutoffRangeDialog({"start": 300.0, "stop": 360.0, "step": 30.0})
    warnings: list[str] = []
    monkeypatch.setattr(
        convergence_widget_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    monkeypatch.setattr(dialog._step, "value", lambda: 0.0)
    dialog._accept_if_valid()
    assert warnings[-1] == "Invalid range"

    monkeypatch.setattr(dialog._step, "value", lambda: 30.0)
    dialog._start.setValue(400.0)
    dialog._stop.setValue(300.0)
    dialog._accept_if_valid()
    assert warnings[-1] == "Invalid range"

    dialog._start.setValue(300.0)
    dialog._stop.setValue(360.0)
    dialog._accept_if_valid()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.values() == (300.0, 330.0, 360.0)


def test_convergence_widget_range_defaults_match_cutoff_policy(qapp) -> None:
    widget = ConvergenceWidget()
    dialog = CutoffRangeDialog(widget._range_state)

    assert widget._range_state == {
        "start": 300.0,
        "stop": 1400.0,
        "step": 100.0,
        "replace_existing": False,
    }
    assert dialog.defaults_state() == widget._range_state


def test_convergence_widget_generates_range_from_dialog(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff()
    widget._cutoff_table.item(0, widget._CUTOFF_COLUMN).setText("280")

    monkeypatch.setattr(
        convergence_widget_module.CutoffRangeDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        convergence_widget_module.CutoffRangeDialog,
        "values",
        lambda _self: (300.0, 330.0, 360.0),
    )
    monkeypatch.setattr(
        convergence_widget_module.CutoffRangeDialog,
        "replace_existing",
        lambda _self: False,
    )
    monkeypatch.setattr(
        convergence_widget_module.CutoffRangeDialog,
        "defaults_state",
        lambda _self: {"start": 300.0, "stop": 360.0, "step": 30.0, "replace_existing": False},
    )

    widget._open_range_dialog()

    assert _table_values(widget) == [280.0, 300.0, 330.0, 360.0]


def test_convergence_widget_uses_cutoff_standard_defaults(qapp) -> None:
    widget = ConvergenceWidget()

    assert widget._name_prefix.text() == "cutoff_convergence"
    assert widget._default_params["task"] == "SINGLEPOINT"
    assert widget._default_params["xc_functional"] == "LDA"
    assert widget._default_params["elec_energy_tol"].value == 1e-6
    assert widget._default_params["max_scf_cycles"] == 100
    assert widget._default_params["metals_method"] == "DM"
    assert widget._default_params["spin_polarized"] is False
    assert widget._default_params["symmetry_generate"] is True
    assert "continuation" not in widget._default_params



def test_convergence_widget_state_roundtrip_uses_per_cutoff_rows(qapp) -> None:
    widget = ConvergenceWidget()
    state = {
        "name_prefix": "conv",
        "range": {"start": 280.0, "stop": 340.0, "step": 20.0, "replace_existing": True},
        "cutoff_energies": [280.0, 300.0, 320.0],
        "cutoff_parameter_rows": [
            {
                "cutoff": 320.0,
                "params": {"task": "SINGLEPOINT", "max_scf_cycles": 120},
                "added_advanced": ["basis_precision"],
            }
        ],
    }

    widget.set_form_state(state)
    collected = widget.collect_state()

    assert collected["mode"] == "manual"
    assert collected["name_prefix"] == "conv"
    assert collected["range"]["replace_existing"] is True
    assert collected["cutoff_energies"] == [280.0, 300.0, 320.0]
    assert collected["cutoff_parameter_rows"] == [
        {
            "cutoff": 320.0,
            "params": {"task": "SINGLEPOINT", "max_scf_cycles": 120},
            "added_advanced": ["basis_precision"],
        }
    ]


def test_convergence_widget_simulation_cell_selector_emits_selection_signals(qapp) -> None:
    widget = ConvergenceWidget()
    selected: list[str] = []
    reset_requested: list[bool] = []
    widget.simulation_cell_selection_requested.connect(lambda payload: selected.append(str(payload)))
    widget.simulation_cell_use_unit_cell_requested.connect(lambda: reset_requested.append(True))

    widget.set_simulation_cell_options(
        summary="Using saved supercell: convergence_cutoff 2x2x1",
        assets=[
            {
                "structure_id": "11111111-1111-1111-1111-111111111111",
                "label": "convergence_cutoff 2x2x1",
            }
        ],
        selected_structure_id=None,
        enabled=True,
    )

    widget._simulation_cell_combo.setCurrentIndex(1)
    widget._simulation_cell_combo.setCurrentIndex(0)

    assert selected == ["11111111-1111-1111-1111-111111111111"]
    assert reset_requested == [True]


def test_convergence_widget_restores_cutoffs_from_parameter_rows_when_list_missing(qapp) -> None:
    widget = ConvergenceWidget()
    state = {
        "name_prefix": "conv",
        "cutoff_energies": [],
        "cutoff_parameter_rows": [
            {
                "cutoff": 300.0,
                "params": {"task": "SINGLEPOINT", "max_scf_cycles": 80},
                "added_advanced": [],
            },
            {
                "cutoff": 360.0,
                "params": {"task": "SINGLEPOINT", "max_scf_cycles": 90},
                "added_advanced": [],
            },
        ],
    }

    widget.set_form_state(state)

    assert _table_values(widget) == [300.0, 360.0]
    assert widget._cutoff_parameter_values[300.0]["max_scf_cycles"] == 80
    assert widget._cutoff_parameter_values[360.0]["max_scf_cycles"] == 90


def test_convergence_widget_restores_legacy_shared_and_override_state(qapp) -> None:
    widget = ConvergenceWidget()
    state = {
        "name_prefix": "conv",
        "cutoff_energies": [280.0, 300.0, 320.0],
        "shared_params": {"task": "SINGLEPOINT", "max_scf_cycles": 70},
        "shared_added_advanced": ["basis_precision"],
        "overrides": [
            {
                "cutoffs": [320.0],
                "params": {"max_scf_cycles": 120},
            }
        ],
    }

    widget.set_form_state(state)

    assert widget._effective_params_for_cutoff(280.0)["max_scf_cycles"] == 70
    assert widget._effective_params_for_cutoff(320.0)["max_scf_cycles"] == 120
    assert widget._cutoff_added_advanced[320.0] == {"basis_precision", "max_scf_cycles"}


def test_convergence_widget_single_cutoff_parameter_dialog_updates_row(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)

    monkeypatch.setattr(
        convergence_widget_module.ParameterDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        convergence_widget_module.ParameterDialog,
        "collect_values",
        lambda _self: {"task": "SINGLEPOINT", "max_scf_cycles": 99},
    )
    monkeypatch.setattr(
        convergence_widget_module.ParameterDialog,
        "advanced_names",
        lambda _self: ("basis_precision",),
    )

    widget._open_cutoff_parameter_dialog((300.0,))

    assert widget._cutoff_parameter_values[300.0]["max_scf_cycles"] == 99
    assert widget._cutoff_added_advanced[300.0] == {"basis_precision"}
    configure_button = widget._cutoff_table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)
    assert configure_button.text() == "Configured..."


def test_convergence_widget_queued_cutoff_parameter_dialog_is_read_only(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._cutoff_parameter_values[300.0] = {"max_scf_cycles": 50}
    widget.set_cutoff_runtime_states({300.0: CutoffRunState(status="queued")})

    observed: dict[str, object] = {}

    def fake_exec(dialog) -> int:
        observed["read_only"] = dialog._read_only
        observed["title"] = dialog.windowTitle()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(convergence_widget_module.ParameterDialog, "exec", fake_exec)
    monkeypatch.setattr(
        convergence_widget_module.ParameterDialog,
        "collect_values",
        lambda _self: {"task": "SINGLEPOINT", "max_scf_cycles": 99},
    )

    widget._open_cutoff_parameter_dialog((300.0,))

    assert observed == {"read_only": True, "title": "Inspect Parameters: 300 eV"}
    assert widget._cutoff_parameter_values[300.0]["max_scf_cycles"] == 50
    configure_button = widget._cutoff_table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)
    assert configure_button.text() == "Configured..."


def test_convergence_widget_cancelled_cutoff_parameters_are_editable(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._cutoff_parameter_values[300.0] = {"max_scf_cycles": 50}
    widget.set_cutoff_runtime_states({300.0: CutoffRunState(status="cancelled")})

    observed: dict[str, object] = {}

    def fake_exec(dialog) -> int:
        observed["read_only"] = dialog._read_only
        observed["title"] = dialog.windowTitle()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(convergence_widget_module.ParameterDialog, "exec", fake_exec)
    monkeypatch.setattr(
        convergence_widget_module.ParameterDialog,
        "collect_values",
        lambda _self: {"task": "SINGLEPOINT", "max_scf_cycles": 99},
    )

    widget._open_cutoff_parameter_dialog((300.0,))

    assert observed == {"read_only": False, "title": "Configure Parameters: 300 eV"}
    assert widget._cutoff_parameter_values[300.0]["max_scf_cycles"] == 99


def test_convergence_widget_batch_configure_selected_cutoffs(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._add_cutoff(360.0)
    monkeypatch.setattr(widget, "_selected_cutoffs", lambda: (300.0, 360.0))

    intros: list[str | None] = []
    original_init = convergence_widget_module.ParameterDialog.__init__

    def fake_init(self, *args, **kwargs):
        intros.append(kwargs.get("intro_text"))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(convergence_widget_module.ParameterDialog, "__init__", fake_init)
    monkeypatch.setattr(
        convergence_widget_module.ParameterDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        convergence_widget_module.ParameterDialog,
        "collect_values",
        lambda _self: {"task": "SINGLEPOINT", "xc_functional": "PBE"},
    )
    monkeypatch.setattr(
        convergence_widget_module.ParameterDialog,
        "advanced_names",
        lambda _self: (),
    )

    widget._configure_selected_cutoffs()

    assert intros[-1] == "Included cutoffs: 300 eV, 360 eV"
    assert widget._cutoff_parameter_values[300.0]["xc_functional"] == "PBE"
    assert widget._cutoff_parameter_values[360.0]["xc_functional"] == "PBE"


def test_convergence_widget_parameter_dialog_places_kpoint_below_basis_controls(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._cutoff_added_advanced[300.0] = {"kpoint_mp_grid"}

    observed: dict[str, object] = {}

    def fake_exec(dialog) -> int:
        names = dialog._visible_names()
        observed["names"] = names
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(convergence_widget_module.ParameterDialog, "exec", fake_exec)

    widget._open_cutoff_parameter_dialog((300.0,))

    names = observed.get("names")
    assert isinstance(names, list)
    assert "cut_off_energy" in names
    assert "elec_energy_tol" in names
    assert "kpoint_mp_grid" in names
    assert names.index("cut_off_energy") < names.index("elec_energy_tol") < names.index("kpoint_mp_grid")


def test_convergence_widget_add_cutoff_fallback_on_invalid_previous(qapp) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff()
    widget._cutoff_table.item(0, widget._CUTOFF_COLUMN).setText("not-a-number")

    widget._add_cutoff()

    assert widget._cutoff_table.item(1, widget._CUTOFF_COLUMN).text() == "300"


def test_convergence_widget_updates_runtime_columns(qapp) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._add_cutoff(360.0)

    widget.set_cutoff_runtime_states(
        {
            300.0: CutoffRunState(status="running"),
            360.0: CutoffRunState(status="completed", total_energy=-1234.567, max_force=0.0123),
        }
    )

    assert widget._cutoff_table.item(0, widget._STATUS_COLUMN).text() == "running"
    assert (
        widget._cutoff_table.item(0, widget._STATUS_COLUMN).foreground().color().name()
        == convergence_widget_module.status_brush("running").color().name()
    )
    assert widget._cutoff_table.item(0, widget._TOTAL_ENERGY_COLUMN).text() == ""
    assert widget._cutoff_table.item(0, widget._MAX_FORCE_COLUMN).text() == ""
    assert widget._cutoff_table.item(1, widget._STATUS_COLUMN).text() == "completed"
    assert (
        widget._cutoff_table.item(1, widget._STATUS_COLUMN).foreground().color().name()
        == convergence_widget_module.status_brush("completed").color().name()
    )
    assert widget._cutoff_table.item(1, widget._TOTAL_ENERGY_COLUMN).text() == "-1234.567"
    assert widget._cutoff_table.item(1, widget._MAX_FORCE_COLUMN).text() == "0.0123"


def test_convergence_widget_ready_rows_lock_cutoff_editing(qapp) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)

    cutoff_item = widget._cutoff_table.item(0, widget._CUTOFF_COLUMN)
    assert cutoff_item is not None
    assert bool(cutoff_item.flags() & Qt.ItemFlag.ItemIsEditable)

    # 'ready' state alone should not lock direct editing; lock when queued or later
    widget.set_cutoff_runtime_states({300.0: CutoffRunState(status="ready")})
    assert bool(cutoff_item.flags() & Qt.ItemFlag.ItemIsEditable)

    widget.set_cutoff_runtime_states({300.0: CutoffRunState(status="queued")})
    assert not bool(cutoff_item.flags() & Qt.ItemFlag.ItemIsEditable)


def test_convergence_widget_runtime_columns_use_higher_precision(qapp) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)

    total_energy = -1234.56789012345
    max_force = 0.012345678901234
    widget.set_cutoff_runtime_states(
        {
            300.0: CutoffRunState(status="completed", total_energy=total_energy, max_force=max_force),
        }
    )

    assert widget._cutoff_table.item(0, widget._TOTAL_ENERGY_COLUMN).text() == f"{total_energy:.12g}"
    assert widget._cutoff_table.item(0, widget._MAX_FORCE_COLUMN).text() == f"{max_force:.12g}"


def test_convergence_widget_remove_cutoff_warns_for_completed_jobs(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget.set_cutoff_runtime_states({300.0: CutoffRunState(status="completed")})

    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    monkeypatch.setattr(
        convergence_widget_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: answers.pop(0),
    )

    widget._request_cutoff_removal(0)
    assert widget._cutoff_table.rowCount() == 1

    requested: list[object] = []
    widget.remove_cutoff_requested.connect(lambda payload: requested.append(payload))
    widget._request_cutoff_removal(0)
    assert requested == [[300.0]]
    assert widget._cutoff_table.rowCount() == 1


def test_convergence_widget_remove_cutoff_running_emits_cancel_request(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget.set_cutoff_runtime_states({300.0: CutoffRunState(status="running")})

    cancelled: list[object] = []
    widget.cancel_requested.connect(lambda payload: cancelled.append(payload))
    monkeypatch.setattr(
        convergence_widget_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    widget._request_cutoff_removal(0)

    assert cancelled == [[300.0]]
    assert widget._cutoff_table.rowCount() == 1


def test_convergence_widget_remove_button_emits_removal_request(qapp) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)

    requested: list[object] = []
    widget.remove_cutoff_requested.connect(lambda payload: requested.append(payload))

    remove_button = widget._cutoff_table.cellWidget(0, widget._REMOVE_COLUMN)
    assert isinstance(remove_button, QToolButton)

    widget._remove_cutoff_button_clicked(remove_button)

    assert requested == [[300.0]]
    assert widget._cutoff_table.rowCount() == 1


def test_convergence_widget_run_button_emits_config_and_submit_for_current_row(qapp) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._cutoff_table.setCurrentCell(0, widget._CUTOFF_COLUMN)

    configured: list[ConvergenceConfig] = []
    submitted: list[object] = []
    widget.configure_requested.connect(lambda payload: configured.append(payload))
    widget.submit_local_requested.connect(lambda payload: submitted.append(payload))

    widget._run_local_selected_cutoffs()

    assert len(configured) == 1
    assert configured[0].cutoff_energies == (300.0,)
    assert submitted == [[300.0]]


def test_convergence_widget_cluster_run_emits_signal_for_current_row(qapp) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._cutoff_table.setCurrentCell(0, widget._CUTOFF_COLUMN)

    configured: list[ConvergenceConfig] = []
    submitted: list[object] = []
    widget.configure_requested.connect(lambda payload: configured.append(payload))
    widget.submit_cluster_requested.connect(lambda payload: submitted.append(payload))

    widget._run_cluster_selected_cutoffs()

    assert len(configured) == 1
    assert configured[0].cutoff_energies == (300.0,)
    assert submitted == [[300.0]]


def test_convergence_widget_selected_or_current_cutoffs_prefers_selection_and_current(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._add_cutoff(360.0)

    widget._cutoff_table.selectRow(1)
    assert widget._selected_or_current_cutoffs() == (360.0,)

    widget._cutoff_table.clearSelection()
    widget._cutoff_table.setCurrentCell(0, widget._CUTOFF_COLUMN)
    assert widget._selected_or_current_cutoffs() == (300.0,)

    single = ConvergenceWidget()
    single._add_cutoff(420.0)
    monkeypatch.setattr(single._cutoff_table, "currentRow", lambda: -1)
    assert single._selected_or_current_cutoffs() == (420.0,)


def test_convergence_widget_remove_selected_or_current_cutoffs_uses_selection(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._add_cutoff(360.0)
    selection_model = widget._cutoff_table.selectionModel()
    assert selection_model is not None
    selection_model.select(
        widget._cutoff_table.model().index(0, widget._CUTOFF_COLUMN),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )
    selection_model.select(
        widget._cutoff_table.model().index(1, widget._CUTOFF_COLUMN),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )

    requested_rows: list[tuple[int, ...]] = []
    monkeypatch.setattr(widget, "_request_cutoff_removals", lambda rows: requested_rows.append(tuple(rows)))

    widget._remove_selected_or_current_cutoffs()

    assert requested_rows == [(0, 1)]


def test_convergence_widget_delete_shortcut_requests_selected_cutoff_removal(qapp) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._add_cutoff(360.0)
    requested: list[list[float]] = []
    widget.remove_cutoff_requested.connect(lambda cutoffs: requested.append(list(cutoffs)))

    widget._cutoff_table.selectRow(1)
    widget._delete_shortcut.activated.emit()

    assert requested == [[360.0]]


def test_convergence_widget_backspace_shortcut_requests_current_cutoff_removal(qapp) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    requested: list[list[float]] = []
    widget.remove_cutoff_requested.connect(lambda cutoffs: requested.append(list(cutoffs)))

    widget._cutoff_table.setCurrentCell(0, widget._CUTOFF_COLUMN)
    widget._backspace_shortcut.activated.emit()

    assert requested == [[300.0]]


def test_convergence_widget_delete_key_requests_selected_cutoff_from_child_focus(qapp) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._add_cutoff(360.0)
    requested: list[list[float]] = []
    widget.remove_cutoff_requested.connect(lambda cutoffs: requested.append(list(cutoffs)))
    configure_button = widget._cutoff_table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)

    widget.show()
    widget._cutoff_table.selectRow(0)
    configure_button.setFocus()
    qapp.processEvents()
    QTest.keyClick(configure_button, Qt.Key.Key_Delete)

    assert requested == [[300.0]]


def test_convergence_widget_batch_confirm_uses_yes_to_all_for_progressed_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._add_cutoff(360.0)
    widget.set_cutoff_runtime_states(
        {
            300.0: CutoffRunState(status="completed"),
            360.0: CutoffRunState(status="failed"),
        }
    )

    monkeypatch.setattr(
        widget,
        "_ask_batch_cutoff_question",
        lambda **_kwargs: QMessageBox.StandardButton.YesToAll,
    )

    approved, aborted = widget._confirm_cutoff_removals((0, 1))

    assert approved == [300.0, 360.0]
    assert aborted is False


def test_convergence_widget_emit_config_groups_shared_and_override_values(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._add_cutoff(360.0)
    widget._set_cutoff_parameter_values(
        300.0,
        {"task": "SINGLEPOINT", "xc_functional": "PBE", "max_scf_cycles": 80},
        set(),
    )
    widget._set_cutoff_parameter_values(
        360.0,
        {"task": "SINGLEPOINT", "xc_functional": "PBE", "max_scf_cycles": 120},
        set(),
    )

    emitted = []
    widget.configure_requested.connect(lambda payload: emitted.append(payload))
    widget._emit_config()

    assert len(emitted) == 1
    config = emitted[0]
    assert config.shared_params["task"] == "SINGLEPOINT"
    assert config.shared_params["xc_functional"] == "PBE"
    assert config.shared_params.get("max_scf_cycles") is None
    assert config.overrides == {
        frozenset({300.0}): {"max_scf_cycles": 80},
        frozenset({360.0}): {"max_scf_cycles": 120},
    }


def test_convergence_widget_emit_config_warning_paths(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    warnings: list[str] = []
    monkeypatch.setattr(
        convergence_widget_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    widget._add_cutoff()
    widget._cutoff_table.item(0, widget._CUTOFF_COLUMN).setText("abc")
    widget._emit_config()
    assert warnings[-1] == "Invalid cutoff list"

    widget._cutoff_table.setRowCount(0)
    widget._emit_config()
    assert warnings[-1] == "No cutoffs"

    widget._add_cutoff()
    widget._add_cutoff()
    widget._cutoff_table.item(0, widget._CUTOFF_COLUMN).setText("300")
    widget._cutoff_table.item(1, widget._CUTOFF_COLUMN).setText("300")
    widget._emit_config()
    assert warnings[-1] == "Duplicate cutoffs"

    widget._cutoff_table.item(1, widget._CUTOFF_COLUMN).setText("360")
    widget._set_cutoff_parameter_values(
        300.0,
        {"task": "SINGLEPOINT", "metals_method": "NOTREAL"},
        set(),
    )
    widget._emit_config()
    assert warnings[-1] == "Invalid parameters"


def test_convergence_widget_context_menu_emits_submit_for_ready_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget.set_cutoff_runtime_states({300.0: CutoffRunState(status="ready")})

    submitted: list[object] = []
    widget.submit_local_requested.connect(lambda payload: submitted.append(payload))

    class _FakeSignal:
        def __init__(self) -> None:
            self._callbacks = []

        def connect(self, callback) -> None:
            self._callbacks.append(callback)

        def emit(self) -> None:
            for callback in self._callbacks:
                callback(False)

    class _FakeAction:
        def __init__(self, text: str) -> None:
            self.text = text
            self.enabled = True
            self.triggered = _FakeSignal()

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = enabled

    class _FakeMenu:
        def __init__(self, *_args, **_kwargs) -> None:
            self.actions: list[_FakeAction] = []

        def addAction(self, text: str) -> _FakeAction:
            action = _FakeAction(text)
            self.actions.append(action)
            return action

        def exec(self, *_args, **_kwargs) -> None:
            for action in self.actions:
                if action.text == "Submit Local Calculation" and action.enabled:
                    action.triggered.emit()
                    return

    monkeypatch.setattr(convergence_widget_module, "QMenu", _FakeMenu)

    widget._open_cutoff_context_menu(QPoint(10, 10))

    assert submitted == [[300.0]]


def test_convergence_widget_context_menu_emits_inspect_for_failed_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget.set_cutoff_runtime_states({300.0: CutoffRunState(status="failed")})

    inspected: list[object] = []
    widget.inspect_failed_requested.connect(lambda payload: inspected.append(payload))

    class _FakeSignal:
        def __init__(self) -> None:
            self._callbacks = []

        def connect(self, callback) -> None:
            self._callbacks.append(callback)

        def emit(self) -> None:
            for callback in self._callbacks:
                callback(False)

    class _FakeAction:
        def __init__(self, text: str) -> None:
            self.text = text
            self.enabled = True
            self.triggered = _FakeSignal()

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = enabled

    class _FakeMenu:
        def __init__(self, *_args, **_kwargs) -> None:
            self.actions: list[_FakeAction] = []

        def addAction(self, text: str) -> _FakeAction:
            action = _FakeAction(text)
            self.actions.append(action)
            return action

        def exec(self, *_args, **_kwargs) -> None:
            for action in self.actions:
                if action.text == "Inspect Failure Files" and action.enabled:
                    action.triggered.emit()
                    return

    monkeypatch.setattr(convergence_widget_module, "QMenu", _FakeMenu)

    widget._open_cutoff_context_menu(QPoint(10, 10))

    assert inspected == [[300.0]]


def test_convergence_widget_context_menu_emits_reset_for_failed_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget.set_cutoff_runtime_states({300.0: CutoffRunState(status="failed")})

    reset_payloads: list[object] = []
    widget.reset_failed_requested.connect(lambda payload: reset_payloads.append(payload))

    class _FakeSignal:
        def __init__(self) -> None:
            self._callbacks = []

        def connect(self, callback) -> None:
            self._callbacks.append(callback)

        def emit(self) -> None:
            for callback in self._callbacks:
                callback(False)

    class _FakeAction:
        def __init__(self, text: str) -> None:
            self.text = text
            self.enabled = True
            self.triggered = _FakeSignal()

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = enabled

    class _FakeMenu:
        def __init__(self, *_args, **_kwargs) -> None:
            self.actions: list[_FakeAction] = []

        def addAction(self, text: str) -> _FakeAction:
            action = _FakeAction(text)
            self.actions.append(action)
            return action

        def exec(self, *_args, **_kwargs) -> None:
            for action in self.actions:
                if action.text == "Reset Failed Jobs..." and action.enabled:
                    action.triggered.emit()
                    return

    monkeypatch.setattr(convergence_widget_module, "QMenu", _FakeMenu)

    widget._open_cutoff_context_menu(QPoint(10, 10))

    assert reset_payloads == [[300.0]]


def test_convergence_widget_context_menu_emits_reset_for_cancelled_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget.set_cutoff_runtime_states({300.0: CutoffRunState(status="cancelled")})

    reset_payloads: list[object] = []
    widget.reset_failed_requested.connect(lambda payload: reset_payloads.append(payload))

    class _FakeSignal:
        def __init__(self) -> None:
            self._callbacks = []

        def connect(self, callback) -> None:
            self._callbacks.append(callback)

        def emit(self) -> None:
            for callback in self._callbacks:
                callback(False)

    class _FakeAction:
        def __init__(self, text: str) -> None:
            self.text = text
            self.enabled = True
            self.triggered = _FakeSignal()

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = enabled

    class _FakeMenu:
        def __init__(self, *_args, **_kwargs) -> None:
            self.actions: list[_FakeAction] = []

        def addAction(self, text: str) -> _FakeAction:
            action = _FakeAction(text)
            self.actions.append(action)
            return action

        def exec(self, *_args, **_kwargs) -> None:
            for action in self.actions:
                if action.text == "Reset Failed Jobs..." and action.enabled:
                    action.triggered.emit()
                    return

    monkeypatch.setattr(convergence_widget_module, "QMenu", _FakeMenu)

    widget._open_cutoff_context_menu(QPoint(10, 10))

    assert reset_payloads == [[300.0]]


def test_convergence_widget_enables_examine_button_for_completed_results(qapp) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._add_cutoff(400.0)

    widget.set_cutoff_runtime_states(
        {
            300.0: CutoffRunState(status="completed", total_energy=-10.0, max_force=0.2),
            400.0: CutoffRunState(status="completed", total_energy=-10.01, max_force=0.18),
        }
    )

    assert widget._examine_button.isEnabled() is True


def test_convergence_widget_disables_examine_button_without_complete_metrics(qapp) -> None:
    widget = ConvergenceWidget()
    widget._add_cutoff(300.0)
    widget._add_cutoff(400.0)

    widget.set_cutoff_runtime_states(
        {
            300.0: CutoffRunState(status="completed", total_energy=-10.0, max_force=0.2),
            400.0: CutoffRunState(status="completed", total_energy=-10.01, max_force=None),
        }
    )

    assert widget._examine_button.isEnabled() is False


def test_convergence_widget_examine_button_emits_signal(qapp) -> None:
    widget = ConvergenceWidget()
    emitted: list[bool] = []
    widget.examine_requested.connect(lambda: emitted.append(True))

    widget._examine_button.setEnabled(True)
    widget._examine_button.click()

    assert emitted == [True]


def test_convergence_widget_displays_project_defaults(qapp) -> None:
    widget = ConvergenceWidget()

    widget.set_project_defaults(
        cutoff_energy=550.0,
        cutoff_method="asymptotic",
        energy_threshold=0.001,
        force_threshold=0.05,
    )

    assert widget._project_default_cutoff.text() == "550 eV"
    assert widget._project_default_method.text() == "Asymptotic Extrapolation"
    assert widget._project_default_criteria.text() == "dE <= 0.001 eV/atom, dF <= 0.05 eV/Ang"
