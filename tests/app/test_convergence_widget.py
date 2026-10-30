"""Tests for cutoff convergence configuration widget."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QPushButton, QToolButton

import mufinder.app.convergence_widget as convergence_widget_module
from tests.support.reset_menu import install_fake_menu, open_reset_menu
from mufinder.app import workflow_table_utils
from mufinder.app.convergence_types import ConvergenceConfig, SweepPointRunState
from mufinder.app.convergence_widget import CutoffConvergenceWidget, CutoffRangeDialog
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.theme import status_brush


def _table_values(widget: CutoffConvergenceWidget) -> list[float]:
    values: list[float] = []
    for row in range(widget._table.rowCount()):
        item = widget._table.item(row, widget._KEY_COLUMN)
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
    widget = CutoffConvergenceWidget()
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
    widget = CutoffConvergenceWidget()
    widget._add_row()
    widget._table.item(0, widget._KEY_COLUMN).setText("280")

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

    widget._open_generator_dialog()

    assert _table_values(widget) == [280.0, 300.0, 330.0, 360.0]


def test_convergence_widget_uses_cutoff_standard_defaults(qapp) -> None:
    widget = CutoffConvergenceWidget()

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
    widget = CutoffConvergenceWidget()
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
    widget = CutoffConvergenceWidget()
    selected: list[str] = []
    reset_requested: list[bool] = []
    widget.simulation_cell_selection_requested.connect(
        lambda payload: selected.append(str(payload))
    )
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
    widget = CutoffConvergenceWidget()
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
    assert widget._parameter_values[300.0]["max_scf_cycles"] == 80
    assert widget._parameter_values[360.0]["max_scf_cycles"] == 90


def test_convergence_widget_restores_legacy_shared_and_override_state(qapp) -> None:
    widget = CutoffConvergenceWidget()
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

    assert widget._effective_params_for_key(280.0)["max_scf_cycles"] == 70
    assert widget._effective_params_for_key(320.0)["max_scf_cycles"] == 120
    assert widget._added_advanced[320.0] == {"basis_precision", "max_scf_cycles"}


def test_convergence_widget_configure_button_reopens_after_first_use(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for issue #23: the per-row Configure button must stay
    clickable after being used once, regardless of whether the dialog was
    accepted or dismissed."""
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)

    exec_calls: list[int] = []

    def fake_exec(_self) -> int:
        exec_calls.append(1)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(ParameterDialog, "exec", fake_exec)

    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)

    configure_button.click()
    assert len(exec_calls) == 1

    configure_button.click()
    assert len(exec_calls) == 2, "Configure button stopped opening the dialog after first use"


def test_convergence_widget_single_cutoff_parameter_dialog_updates_row(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)

    monkeypatch.setattr(
        ParameterDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        ParameterDialog,
        "collect_values",
        lambda _self: {"task": "SINGLEPOINT", "max_scf_cycles": 99},
    )
    monkeypatch.setattr(
        ParameterDialog,
        "advanced_names",
        lambda _self: ("basis_precision",),
    )

    widget._open_parameter_dialog((300.0,))

    assert widget._parameter_values[300.0]["max_scf_cycles"] == 99
    assert widget._added_advanced[300.0] == {"basis_precision"}
    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)
    assert configure_button.text() == "Configured…"


def test_convergence_widget_queued_cutoff_parameter_dialog_is_read_only(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._parameter_values[300.0] = {"max_scf_cycles": 50}
    widget.set_runtime_states({300.0: SweepPointRunState(status="queued")})

    observed: dict[str, object] = {}

    def fake_exec(dialog) -> int:
        observed["read_only"] = dialog._read_only
        observed["title"] = dialog.windowTitle()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ParameterDialog, "exec", fake_exec)
    monkeypatch.setattr(
        ParameterDialog,
        "collect_values",
        lambda _self: {"task": "SINGLEPOINT", "max_scf_cycles": 99},
    )

    widget._open_parameter_dialog((300.0,))

    assert observed == {"read_only": True, "title": "Inspect Parameters: 300 eV"}
    assert widget._parameter_values[300.0]["max_scf_cycles"] == 50
    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)
    assert configure_button.text() == "Configured…"


def test_convergence_widget_cancelled_cutoff_parameters_are_editable(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._parameter_values[300.0] = {"max_scf_cycles": 50}
    widget.set_runtime_states({300.0: SweepPointRunState(status="cancelled")})

    observed: dict[str, object] = {}

    def fake_exec(dialog) -> int:
        observed["read_only"] = dialog._read_only
        observed["title"] = dialog.windowTitle()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ParameterDialog, "exec", fake_exec)
    monkeypatch.setattr(
        ParameterDialog,
        "collect_values",
        lambda _self: {"task": "SINGLEPOINT", "max_scf_cycles": 99},
    )

    widget._open_parameter_dialog((300.0,))

    assert observed == {"read_only": False, "title": "Configure Parameters: 300 eV"}
    assert widget._parameter_values[300.0]["max_scf_cycles"] == 99


def test_convergence_widget_batch_configure_selected_cutoffs(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(360.0)
    monkeypatch.setattr(widget, "_selected_keys", lambda: (300.0, 360.0))

    intros: list[str | None] = []
    original_init = ParameterDialog.__init__

    def fake_init(self, *args, **kwargs):
        intros.append(kwargs.get("intro_text"))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(ParameterDialog, "__init__", fake_init)
    monkeypatch.setattr(
        ParameterDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        ParameterDialog,
        "collect_values",
        lambda _self: {"task": "SINGLEPOINT", "xc_functional": "PBE"},
    )
    monkeypatch.setattr(
        ParameterDialog,
        "advanced_names",
        lambda _self: (),
    )

    widget._configure_selected()

    assert intros[-1] == "Included cutoffs: 300 eV, 360 eV"
    assert widget._parameter_values[300.0]["xc_functional"] == "PBE"
    assert widget._parameter_values[360.0]["xc_functional"] == "PBE"


def test_convergence_widget_parameter_dialog_places_kpoint_below_basis_controls(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._added_advanced[300.0] = {"kpoint_mp_grid"}

    observed: dict[str, object] = {}

    def fake_exec(dialog) -> int:
        names = dialog._visible_names()
        observed["names"] = names
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(ParameterDialog, "exec", fake_exec)

    widget._open_parameter_dialog((300.0,))

    names = observed.get("names")
    assert isinstance(names, list)
    assert "cut_off_energy" in names
    assert "elec_energy_tol" in names
    assert "kpoint_mp_grid" in names
    assert (
        names.index("cut_off_energy")
        < names.index("elec_energy_tol")
        < names.index("kpoint_mp_grid")
    )


def test_convergence_widget_add_cutoff_fallback_on_invalid_previous(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row()
    widget._table.item(0, widget._KEY_COLUMN).setText("not-a-number")

    widget._add_row()

    assert widget._table.item(1, widget._KEY_COLUMN).text() == "300"


def test_convergence_widget_updates_runtime_columns(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(360.0)

    widget.set_runtime_states(
        {
            300.0: SweepPointRunState(status="running"),
            360.0: SweepPointRunState(status="completed", total_energy=-1234.567, max_force=0.0123),
        }
    )

    assert widget._table.item(0, widget._STATUS_COLUMN).text() == "running"
    assert (
        widget._table.item(0, widget._STATUS_COLUMN).foreground().color().name()
        == status_brush("running").color().name()
    )
    assert widget._table.item(0, widget._TOTAL_ENERGY_COLUMN).text() == ""
    assert widget._table.item(0, widget._MAX_FORCE_COLUMN).text() == ""
    assert widget._table.item(1, widget._STATUS_COLUMN).text() == "completed"
    assert (
        widget._table.item(1, widget._STATUS_COLUMN).foreground().color().name()
        == status_brush("completed").color().name()
    )
    assert widget._table.item(1, widget._TOTAL_ENERGY_COLUMN).text() == "-1234.567"
    assert widget._table.item(1, widget._MAX_FORCE_COLUMN).text() == "0.0123"


def test_convergence_widget_ready_rows_lock_cutoff_editing(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)

    cutoff_item = widget._table.item(0, widget._KEY_COLUMN)
    assert cutoff_item is not None
    assert bool(cutoff_item.flags() & Qt.ItemFlag.ItemIsEditable)

    # 'ready' state alone should not lock direct editing; lock when queued or later
    widget.set_runtime_states({300.0: SweepPointRunState(status="ready")})
    assert bool(cutoff_item.flags() & Qt.ItemFlag.ItemIsEditable)

    widget.set_runtime_states({300.0: SweepPointRunState(status="queued")})
    assert not bool(cutoff_item.flags() & Qt.ItemFlag.ItemIsEditable)


def test_convergence_widget_runtime_columns_use_higher_precision(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)

    total_energy = -1234.56789012345
    max_force = 0.012345678901234
    widget.set_runtime_states(
        {
            300.0: SweepPointRunState(
                status="completed", total_energy=total_energy, max_force=max_force
            ),
        }
    )

    assert widget._table.item(0, widget._TOTAL_ENERGY_COLUMN).text() == f"{total_energy:.12g}"
    assert widget._table.item(0, widget._MAX_FORCE_COLUMN).text() == f"{max_force:.12g}"


def test_convergence_widget_remove_cutoff_confirms_and_reports_submitted_jobs(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget.set_runtime_states({300.0: SweepPointRunState(status="completed")})
    requested: list[object] = []
    widget.remove_cutoff_requested.connect(lambda payload: requested.append(payload))
    answers = [False, True]
    prompts: list[tuple[str, str, str, str]] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or answers.pop(0),
    )

    widget._request_removals((0,))
    assert requested == []

    widget._request_removals((0,))
    assert requested == [[300.0]]
    assert prompts[1] == (
        "Remove cutoff",
        "Remove 1 cutoff?",
        "1 cutoff already has a submitted job.",
        "Remove 1 cutoff",
    )


def test_convergence_widget_remove_cutoff_running_emits_cancel_request(
    qapp, confirmed_destructive_actions
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(360.0)
    widget._add_row(420.0)
    widget.set_runtime_states(
        {
            300.0: SweepPointRunState(status="running"),
            360.0: SweepPointRunState(status="completed"),
            420.0: SweepPointRunState(status="queued"),
        }
    )

    cancelled: list[object] = []
    widget.cancel_requested.connect(lambda payload: cancelled.append(payload))

    widget._request_removals((0, 1, 2))

    assert cancelled == [[300.0, 420.0]]
    assert confirmed_destructive_actions[0]["text"] == "Remove 3 cutoffs?"
    assert confirmed_destructive_actions[0]["detail"] == (
        "2 jobs are queued or running and will be cancelled. 1 cutoff already has a submitted job."
    )


def test_convergence_widget_declined_removal_cancels_nothing(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget.set_runtime_states({300.0: SweepPointRunState(status="running")})
    emitted: list[object] = []
    widget.cancel_requested.connect(emitted.append)
    widget.remove_cutoff_requested.connect(emitted.append)
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)

    widget._request_removals((0,))

    assert emitted == []


def test_convergence_widget_remove_button_emits_removal_request(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)

    requested: list[object] = []
    widget.remove_cutoff_requested.connect(lambda payload: requested.append(payload))

    remove_button = widget._table.cellWidget(0, widget._REMOVE_COLUMN)
    assert isinstance(remove_button, QToolButton)

    widget._remove_button_clicked(remove_button, 300.0)

    assert requested == [[300.0]]
    assert widget._table.rowCount() == 1


def test_convergence_widget_run_button_emits_config_and_submit_for_current_row(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._table.setCurrentCell(0, widget._KEY_COLUMN)

    configured: list[ConvergenceConfig] = []
    submitted: list[object] = []
    widget.configure_requested.connect(lambda payload: configured.append(payload))
    widget.submit_local_requested.connect(lambda payload: submitted.append(payload))

    widget._run_local_selected()

    assert len(configured) == 1
    assert configured[0].sweep_values == (300.0,)
    assert submitted == [[300.0]]


def test_convergence_widget_cluster_run_emits_signal_for_current_row(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._table.setCurrentCell(0, widget._KEY_COLUMN)

    configured: list[ConvergenceConfig] = []
    submitted: list[object] = []
    widget.configure_requested.connect(lambda payload: configured.append(payload))
    widget.submit_cluster_requested.connect(lambda payload: submitted.append(payload))

    widget._run_cluster_selected()

    assert len(configured) == 1
    assert configured[0].sweep_values == (300.0,)
    assert submitted == [[300.0]]


def test_convergence_widget_run_targets_selection_else_every_row(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(360.0)

    widget._table.selectRow(1)
    assert widget._selected_or_all_keys() == (360.0,)

    # With nothing selected the run buttons read "Run All ...": the current
    # cell is not a hidden target.
    widget._table.clearSelection()
    widget._table.setCurrentCell(0, widget._KEY_COLUMN)
    widget._table.clearSelection()
    assert widget._selected_or_all_keys() == (300.0, 360.0)


def test_convergence_widget_remove_selected_or_current_cutoffs_uses_selection(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(360.0)
    selection_model = widget._table.selectionModel()
    assert selection_model is not None
    selection_model.select(
        widget._table.model().index(0, widget._KEY_COLUMN),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )
    selection_model.select(
        widget._table.model().index(1, widget._KEY_COLUMN),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )

    requested_rows: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        widget, "_request_removals", lambda rows: requested_rows.append(tuple(rows))
    )

    widget._remove_selected_or_current()

    assert requested_rows == [(0, 1)]


def test_convergence_widget_delete_shortcut_requests_selected_cutoff_removal(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(360.0)
    requested: list[list[float]] = []
    widget.remove_cutoff_requested.connect(lambda cutoffs: requested.append(list(cutoffs)))

    widget._table.selectRow(1)
    widget._delete_shortcut.activated.emit()

    assert requested == [[360.0]]


def test_convergence_widget_backspace_shortcut_requests_current_cutoff_removal(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    requested: list[list[float]] = []
    widget.remove_cutoff_requested.connect(lambda cutoffs: requested.append(list(cutoffs)))

    widget._table.setCurrentCell(0, widget._KEY_COLUMN)
    widget._backspace_shortcut.activated.emit()

    assert requested == [[300.0]]


def test_convergence_widget_delete_key_requests_selected_cutoff_from_child_focus(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(360.0)
    requested: list[list[float]] = []
    widget.remove_cutoff_requested.connect(lambda cutoffs: requested.append(list(cutoffs)))
    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)

    widget.show()
    widget._table.selectRow(0)
    configure_button.setFocus()
    qapp.processEvents()
    QTest.keyClick(configure_button, Qt.Key.Key_Delete)

    assert requested == [[300.0]]


def test_convergence_widget_already_confirmed_removal_does_not_ask_again(
    qapp, confirmed_destructive_actions
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(360.0)
    requested: list[object] = []
    widget.remove_cutoff_requested.connect(lambda payload: requested.append(payload))

    widget._request_removals((0, 1), confirmed=True)

    assert requested == [[300.0, 360.0]]
    assert confirmed_destructive_actions == []


def test_convergence_widget_emit_config_groups_shared_and_override_values(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(360.0)
    widget._set_parameter_values(
        300.0,
        {"task": "SINGLEPOINT", "xc_functional": "PBE", "max_scf_cycles": 80},
        set(),
    )
    widget._set_parameter_values(
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
    widget = CutoffConvergenceWidget()
    warnings: list[str] = []
    monkeypatch.setattr(
        convergence_widget_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    widget._add_row()
    widget._table.item(0, widget._KEY_COLUMN).setText("abc")
    widget._emit_config()
    assert warnings[-1] == "Invalid cutoff list"

    widget._table.setRowCount(0)
    widget._emit_config()
    assert warnings[-1] == "No cutoffs"

    widget._add_row()
    widget._add_row()
    widget._table.item(0, widget._KEY_COLUMN).setText("300")
    widget._table.item(1, widget._KEY_COLUMN).setText("300")
    widget._emit_config()
    assert warnings[-1] == "Duplicate cutoffs"

    widget._table.item(1, widget._KEY_COLUMN).setText("360")
    widget._set_parameter_values(
        300.0,
        {"task": "SINGLEPOINT", "metals_method": "NOTREAL"},
        set(),
    )
    widget._emit_config()
    assert warnings[-1] == "Invalid parameters"


def test_convergence_widget_context_menu_emits_submit_for_ready_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget.set_runtime_states({300.0: SweepPointRunState(status="ready")})
    emitted: list[object] = []
    widget.submit_local_requested.connect(lambda payload: emitted.append(payload))
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(widget._table)
    action = menu.action("Run Local")
    assert action is not None and action.enabled
    action.triggered.emit()

    assert emitted == [[300.0]]


def test_convergence_widget_context_menu_submit_enabled_for_not_generated_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget.set_runtime_states({300.0: SweepPointRunState(status="not generated")})
    emitted: list[object] = []
    widget.submit_cluster_requested.connect(lambda payload: emitted.append(payload))
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(widget._table)
    action = menu.action("Run on Cluster")
    assert action is not None and action.enabled
    action.triggered.emit()

    assert emitted == [[300.0]]


def test_convergence_widget_context_menu_emits_inspect_for_failed_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget.set_runtime_states({300.0: SweepPointRunState(status="failed")})
    emitted: list[object] = []
    widget.inspect_failed_requested.connect(lambda payload: emitted.append(payload))
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(widget._table)
    action = menu.action("Inspect Failure")
    assert action is not None and action.enabled
    action.triggered.emit()

    assert emitted == [[300.0]]


def test_convergence_widget_context_menu_emits_reset_for_failed_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget.set_runtime_states({300.0: SweepPointRunState(status="failed")})
    emitted: list[object] = []
    widget.reset_failed_requested.connect(lambda payload: emitted.append(payload))
    install_fake_menu(monkeypatch)

    # The fake menu's exec() clicks an enabled reset action, as a user would.
    open_reset_menu(widget._table)

    assert emitted == [[300.0]]


def test_convergence_widget_context_menu_emits_reset_for_cancelled_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget.set_runtime_states({300.0: SweepPointRunState(status="cancelled")})
    emitted: list[object] = []
    widget.reset_failed_requested.connect(lambda payload: emitted.append(payload))
    install_fake_menu(monkeypatch)

    # The fake menu's exec() clicks an enabled reset action, as a user would.
    open_reset_menu(widget._table)

    assert emitted == [[300.0]]


def test_convergence_widget_enables_examine_button_for_completed_results(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(400.0)

    widget.set_runtime_states(
        {
            300.0: SweepPointRunState(status="completed", total_energy=-10.0, max_force=0.2),
            400.0: SweepPointRunState(status="completed", total_energy=-10.01, max_force=0.18),
        }
    )

    assert widget._examine_button.isEnabled() is True


def test_convergence_widget_disables_examine_button_without_complete_metrics(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(400.0)

    widget.set_runtime_states(
        {
            300.0: SweepPointRunState(status="completed", total_energy=-10.0, max_force=0.2),
            400.0: SweepPointRunState(status="completed", total_energy=-10.01, max_force=None),
        }
    )

    assert widget._examine_button.isEnabled() is False


def test_convergence_widget_examine_button_emits_signal(qapp) -> None:
    widget = CutoffConvergenceWidget()
    emitted: list[bool] = []
    widget.examine_requested.connect(lambda: emitted.append(True))

    widget._examine_button.setEnabled(True)
    widget._examine_button.click()

    assert emitted == [True]


def test_convergence_widget_seeds_rows_with_project_xc_functional(qapp) -> None:
    widget = CutoffConvergenceWidget()
    schema_default = widget._default_params["xc_functional"]
    widget._add_row(400.0)

    widget.set_xc_functional_default("PBE")

    assert widget._effective_params_for_key(400.0)["xc_functional"] == "PBE"
    config = widget.build_config()
    assert config is not None
    assert config.shared_params["xc_functional"] == "PBE"

    # Clearing the project default restores the schema default.
    widget.set_xc_functional_default(None)
    assert widget._effective_params_for_key(400.0)["xc_functional"] == schema_default


def test_convergence_widget_project_xc_functional_keeps_configured_rows(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(400.0)
    configured = dict(widget._default_params)
    configured["xc_functional"] = "PBESOL"
    widget._set_parameter_values(400.0, configured, set())

    widget.set_xc_functional_default("PBE")

    assert widget._effective_params_for_key(400.0)["xc_functional"] == "PBESOL"


def test_convergence_widget_remove_button_scope_prompt_does_not_cover_running_jobs(
    qapp, monkeypatch: pytest.MonkeyPatch, confirmed_destructive_actions
) -> None:
    from mufinder.app import selection_snapshot
    from mufinder.app.selection_snapshot import SelectionScope

    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(360.0)
    monkeypatch.setattr(
        selection_snapshot, "ask_selection_scope", lambda *a, **kw: SelectionScope.ALL_SELECTED
    )
    cancelled: list[object] = []
    widget.cancel_requested.connect(cancelled.append)

    # No running rows: the three-way prompt is the only confirmation.
    button = widget._table.cellWidget(0, widget._REMOVE_COLUMN)
    button.selection_snapshot_rows = (0, 1)
    button.clicked.emit()
    assert confirmed_destructive_actions == []

    # A running row: the prompt said nothing about cancelling, so ask again.
    widget.set_runtime_states({300.0: SweepPointRunState(status="running")})
    button = widget._table.cellWidget(0, widget._REMOVE_COLUMN)
    button.selection_snapshot_rows = (0, 1)
    button.clicked.emit()
    assert [prompt["detail"] for prompt in confirmed_destructive_actions] == [
        "1 job is queued or running and will be cancelled."
    ]
    assert cancelled == [[300.0]]


def test_convergence_widget_context_cancel_is_confirmed(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(360.0)
    widget.set_runtime_states(
        {300.0: SweepPointRunState(status="running"), 360.0: SweepPointRunState(status="queued")}
    )
    cancelled: list[object] = []
    widget.cancel_requested.connect(cancelled.append)
    prompts: list[tuple[str, ...]] = []
    answers = [False, True]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or answers.pop(0),
    )
    install_fake_menu(monkeypatch)
    widget._table.selectAll()

    open_reset_menu(widget._table).action("Cancel").triggered.emit()
    assert cancelled == []

    open_reset_menu(widget._table).action("Cancel").triggered.emit()
    assert cancelled == [[300.0, 360.0]]
    assert prompts[0] == (
        "Cancel calculations",
        "Cancel 2 calculations?",
        "",
        "Cancel 2 calculations",
        "Keep running",
    )


def test_convergence_widget_context_menu_has_the_shared_order(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(widget._table)

    assert [action.text for action in menu.actions] == [
        "Run Local",
        "Run on Cluster",
        "Cancel",
        "Inspect Failure",
        "Reset Failed Jobs…",
        "Configure…",
        "Remove",
        "Open Containing Folder",
        "Export Inputs/Outputs…",
    ]
    enabled = {action.text for action in menu.actions if action.enabled}
    assert enabled == {"Run Local", "Run on Cluster", "Configure…", "Remove"}


def _two_cutoffs_with_different_scf_cycles() -> CutoffConvergenceWidget:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    widget._add_row(360.0)
    for cutoff, cycles in ((300.0, 80), (360.0, 120)):
        widget._set_parameter_values(
            cutoff, {**widget._default_params, "max_scf_cycles": cycles}, set()
        )
    return widget


def _accept_after(monkeypatch: pytest.MonkeyPatch, edit) -> list[set[str]]:
    """Make the parameter dialog apply ``edit`` and accept, recording its mixed names."""

    mixed: list[set[str]] = []

    def _fake_exec(dialog) -> int:
        mixed.append(dialog.untouched_mixed_names())
        edit(dialog)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ParameterDialog, "exec", _fake_exec)
    return mixed


def test_convergence_batch_configure_keeps_differing_values_when_another_field_is_edited(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = _two_cutoffs_with_different_scf_cycles()
    other_xc = "PBE" if widget._default_params["xc_functional"] != "PBE" else "LDA"
    mixed = _accept_after(
        monkeypatch,
        lambda dialog: dialog._rows["xc_functional"].editor.set_value(other_xc),
    )

    widget._open_parameter_dialog((300.0, 360.0))

    assert mixed == [{"max_scf_cycles"}]
    # The edited field went to both cutoffs; the mixed one was left on each.
    assert widget._effective_params_for_key(300.0)["xc_functional"] == other_xc
    assert widget._effective_params_for_key(360.0)["xc_functional"] == other_xc
    assert widget._effective_params_for_key(300.0)["max_scf_cycles"] == 80
    assert widget._effective_params_for_key(360.0)["max_scf_cycles"] == 120


def test_convergence_batch_configure_writes_an_edited_mixed_value_to_every_cutoff(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = _two_cutoffs_with_different_scf_cycles()
    _accept_after(
        monkeypatch,
        lambda dialog: dialog._rows["max_scf_cycles"].editor.set_value(150),
    )

    widget._open_parameter_dialog((300.0, 360.0))

    assert widget._effective_params_for_key(300.0)["max_scf_cycles"] == 150
    assert widget._effective_params_for_key(360.0)["max_scf_cycles"] == 150


def test_convergence_batch_configure_accepted_unchanged_changes_nothing(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = _two_cutoffs_with_different_scf_cycles()
    before = {c: widget._effective_params_for_key(c) for c in (300.0, 360.0)}
    _accept_after(monkeypatch, lambda dialog: None)

    widget._open_parameter_dialog((300.0, 360.0))

    assert {c: widget._effective_params_for_key(c) for c in (300.0, 360.0)} == before


def test_convergence_widget_remove_button_is_named_after_its_cutoff(qapp) -> None:
    widget = CutoffConvergenceWidget()
    widget._add_row(300.0)
    key_item = widget._table.item(0, widget._KEY_COLUMN)

    assert widget._table.accessibleName() == "Cutoff Sweep"
    assert (
        widget._table.cellWidget(0, widget._REMOVE_COLUMN).accessibleName()
        == f"Remove Cutoff {key_item.text()}"
    )
    assert (
        widget._table.cellWidget(0, widget._CONFIGURE_COLUMN).accessibleName()
        == f"Configure Cutoff {key_item.text()}"
    )


def test_convergence_widget_disabled_symmetry_reopens_unchecked(qapp) -> None:
    widget = CutoffConvergenceWidget()
    dialog = ParameterDialog(
        specs=widget._specs,
        workflow_stage="convergence_cutoff",
        values=dict(widget._default_params),
    )
    dialog._rows["symmetry_generate"].editor.set_value(False)
    collected = dialog.collect_values()
    assert collected["symmetry_generate"] is False

    reopened = ParameterDialog(
        specs=widget._specs,
        workflow_stage="convergence_cutoff",
        values=collected,
    )

    assert reopened._rows["symmetry_generate"].editor.value() is False


def test_convergence_widget_untouched_dialog_still_means_the_defaults(qapp) -> None:
    # Calculate Stress is an unchecked flag by default, so a dialog accepted
    # unchanged must collect the False the defaults hold.
    widget = CutoffConvergenceWidget()
    dialog = ParameterDialog(
        specs=widget._specs,
        workflow_stage="convergence_cutoff",
        values=dict(widget._default_params),
    )

    assert widget._is_default_parameter_set(widget._sanitize(dialog.collect_values()))
