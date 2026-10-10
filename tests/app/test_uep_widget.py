"""Tests for the UEP workflow widget."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QDialog, QLabel, QPushButton, QToolButton

from mufinder.app import workflow_table_utils
from mufinder.app import selection_snapshot as selection_snapshot_module
from mufinder.app.selection_snapshot import SelectionScope
from mufinder.app import uep_widget as uep_widget_module
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.uep_types import UepRunState
from mufinder.app.uep_widget import UEPWidget
from mufinder.castep_io.types import ValueWithUnit
from tests.support.reset_menu import install_fake_menu, open_reset_menu


def test_uep_widget_source_structure_selector_emits_selection_signal(qapp) -> None:
    widget = UEPWidget()
    selected: list[str] = []
    widget.source_structure_selection_requested.connect(lambda payload: selected.append(str(payload)))

    widget.set_source_structure_options(
        [
            {"id": "host", "label": "Host structure"},
            {"id": "pristine_relaxed", "label": "Best pristine relaxation"},
        ],
        selected_id="host",
    )
    widget._source_structure_combo.setCurrentIndex(1)

    assert selected[-1] == "pristine_relaxed"


def test_uep_widget_programmatic_defaults_and_source_updates_do_not_emit_form_state(qapp) -> None:
    widget = UEPWidget()
    emitted: list[dict[str, object]] = []
    widget.form_state_changed.connect(emitted.append)

    widget.set_project_defaults(
        cutoff_energy=500.0,
        kpoint_grid=(2, 2, 2),
        kpoint_reference_lattice=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        xc_functional="PBE",
    )
    widget.set_source_structure_options(
        [
            {"id": "host", "label": "Host structure"},
            {"id": "relaxed", "label": "Relaxed structure"},
        ],
        selected_id="relaxed",
        summary="Using source structure: Relaxed structure",
    )

    assert emitted == []
    assert widget.selected_source_structure_id() == "relaxed"
    assert widget._default_params["cut_off_energy"] == ValueWithUnit(500.0, "eV")
    assert widget._default_params["xc_functional"] == "PBE"

    widget.set_project_defaults(cutoff_energy=None, kpoint_grid=None)
    assert "xc_functional" not in widget._default_params


def test_uep_widget_supercell_dropdown_lists_project_cells(qapp) -> None:
    widget = UEPWidget()
    widget.set_supercell_options(
        [
            {
                "label": "pristine_relaxation 2x2x1",
                "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            }
        ]
    )
    widget._add_row()

    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    assert isinstance(combo, QComboBox)
    labels = [combo.itemText(index) for index in range(combo.count())]
    assert labels[0] == "Unit cell"
    assert "pristine_relaxation 2x2x1" in labels
    assert labels[-1] == "Manage Supercells..."


def test_uep_widget_supercell_combo_shows_a_restored_custom_matrix(qapp) -> None:
    widget = UEPWidget()
    widget.set_supercell_options([{"label": "2x2x1", "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]}])
    widget.set_form_state(
        {
            "name_prefix": "uep",
            "rows": [{"label": "source_01", "supercell_matrix": [[1, 1, 0], [0, 1, 0], [0, 0, 2]]}],
        }
    )

    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    row_id = widget._row_id_at(0)
    assert [combo.itemText(index) for index in range(combo.count())] == [
        "Unit cell",
        "2x2x1",
        "Custom 3x3",
        "Manage Supercells...",
    ]
    assert combo.currentText() == "Custom 3x3"
    assert widget.row_supercell_matrix(row_id) == ((1, 1, 0), (0, 1, 0), (0, 0, 2))


def test_uep_widget_manage_supercells_entry_requests_the_manager(qapp) -> None:
    widget = UEPWidget()
    widget.set_supercell_options([{"label": "2x2x1", "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]}])
    widget._add_row()
    row_id = widget._row_id_at(0)
    requested: list[int] = []
    widget.row_supercell_edit_requested.connect(lambda payload: requested.append(int(payload)))
    states: list[dict] = []
    widget.form_state_changed.connect(states.append)

    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    combo.setCurrentIndex(combo.findText("Manage Supercells..."))

    assert requested == [row_id]
    assert widget.row_supercell_matrix(row_id) is None
    assert combo.currentText() == "Unit cell"
    assert states == []

    combo.setCurrentIndex(combo.findText("2x2x1"))
    assert widget.row_supercell_matrix(row_id) == ((2, 0, 0), (0, 2, 0), (0, 0, 1))
    assert len(states) == 1


def test_uep_widget_dialog_locks_write_formatted_density(qapp) -> None:
    widget = UEPWidget()
    dialog = ParameterDialog(
        specs=widget._specs,
        workflow_stage="uep",
        locked_values={
            "task": "SINGLEPOINT",
            "write_formatted_density": "TRUE (required for UEP source analysis)",
        },
    )

    assert "write_formatted_density" in dialog._rows
    assert dialog._rows["write_formatted_density"].locked is True


def test_uep_widget_defaults_symmetry_generate(qapp) -> None:
    widget = UEPWidget()
    dialog = ParameterDialog(specs=widget._specs, workflow_stage="uep")
    dialog.set_values(widget._default_params)

    assert widget._default_params["symmetry_generate"] is True
    assert dialog.collect_values()["symmetry_generate"] is True

    dialog._rows["symmetry_generate"].editor.set_value(False)
    dialog._reset_parameter("symmetry_generate")

    assert dialog.collect_values()["symmetry_generate"] is True


def test_uep_widget_uses_uep_name_prefix_by_default(qapp) -> None:
    widget = UEPWidget()

    assert widget._name_prefix.text() == "uep"


def _iter_layout_widgets(layout):
    for index in range(layout.count()):
        item = layout.itemAt(index)
        widget = item.widget()
        if widget is not None:
            yield widget
            nested_layout = widget.layout()
            if nested_layout is not None:
                yield from _iter_layout_widgets(nested_layout)
        nested = item.layout()
        if nested is not None:
            yield from _iter_layout_widgets(nested)


def test_uep_widget_shows_name_prefix_field(qapp) -> None:
    widget = UEPWidget()

    # DL-1: the name-prefix field was previously created and wired up but never
    # added to any layout, so users could not see or edit it.
    assert widget._name_prefix.parentWidget() is not None
    assert widget.isAncestorOf(widget._name_prefix)

    layout_widgets = list(_iter_layout_widgets(widget.layout()))
    assert widget._name_prefix in layout_widgets

    labels = [w.text() for w in layout_widgets if isinstance(w, QLabel)]
    assert "Job name prefix" in labels


def test_uep_widget_uses_source_labels_by_default(qapp) -> None:
    widget = UEPWidget()
    widget._add_row()

    label_item = widget._table.item(0, widget._LABEL_COLUMN)
    assert label_item is not None
    assert label_item.text() == "source_01"


def test_uep_widget_normalizes_legacy_uep_source_prefix_in_form_state(qapp) -> None:
    widget = UEPWidget()

    widget.set_form_state({"name_prefix": "uep_source", "rows": []})

    assert widget._name_prefix.text() == "uep"


def test_uep_widget_enables_wizard_only_for_completed_density_sources(qapp) -> None:
    widget = UEPWidget()
    widget._add_row()
    row_id = widget._row_id_at(0)
    assert row_id is not None

    wizard_button = widget._table.cellWidget(0, widget._WIZARD_COLUMN)
    assert isinstance(wizard_button, QPushButton)
    assert wizard_button.isEnabled() is False

    label_item = widget._table.item(0, widget._LABEL_COLUMN)
    assert label_item is not None
    widget.set_runtime_states(
        {
            label_item.text(): UepRunState(
                status="completed",
                total_energy=-10.5,
                potential_grid_available=True,
                density_available=True,
            )
        }
    )

    assert wizard_button.isEnabled() is True
    potential_item = widget._table.item(0, widget._POTENTIAL_GRID_COLUMN)
    assert potential_item is not None
    assert potential_item.text() == "Saved"


def test_uep_widget_configure_row_uses_current_parameter_dialog_api(qapp, monkeypatch) -> None:
    widget = UEPWidget()
    widget._add_row()
    row_id = widget._row_id_at(0)
    assert row_id is not None

    calls: list[tuple[str, Any]] = []
    collected_values = {"cut_off_energy": 450}

    class DummyDialog:
        class DialogCode:
            Accepted = 1

        def __init__(
            self,
            *,
            specs: list[Any],
            workflow_stage: str,
                locked_values: Mapping[str, object],
                intro_text: str | None = None,
                species_options: tuple[str, ...] = (),
                parent: object | None = None,
            ) -> None:
                del specs, workflow_stage, locked_values, intro_text, species_options, parent

        def set_values(
            self, values: Mapping[str, object], *, preferred_exclusive: str | None = None
        ) -> None:
            calls.append(("set_values", dict(values)))

        def set_advanced_names(self, names: set[str]) -> None:
            calls.append(("set_advanced_names", tuple(sorted(names))))

        def exec(self) -> int:
            return self.DialogCode.Accepted

        def collect_values(self) -> dict[str, object]:
            return dict(collected_values)

        def advanced_names(self) -> tuple[str, ...]:
            return ("spin_polarized",)

    monkeypatch.setattr("mufinder.app.uep_widget.ParameterDialog", DummyDialog)
    widget._row_added_advanced[row_id] = {"spin_polarized"}

    widget._configure_row(row_id)

    assert ("set_advanced_names", ("spin_polarized",)) in calls
    assert widget._row_parameter_values[row_id]["task"] == "SINGLEPOINT"
    assert widget._row_parameter_values[row_id]["write_formatted_density"] is True
    assert widget._row_added_advanced[row_id] == {"spin_polarized"}


def test_uep_widget_reopens_dialog_with_saved_row_values(qapp, monkeypatch) -> None:
    widget = UEPWidget()
    widget._add_row()
    row_id = widget._row_id_at(0)
    assert row_id is not None

    seen_values: list[dict[str, object]] = []
    dialog_payloads = iter(
        (
            {"cut_off_energy": 500.0, "kpoint_mp_grid": (2, 2, 2)},
            {"cut_off_energy": 500.0, "kpoint_mp_grid": (2, 2, 2)},
        )
    )

    class DummyDialog:
        class DialogCode:
            Accepted = 1

        def __init__(
            self,
            *,
            specs: list[Any],
            workflow_stage: str,
                locked_values: Mapping[str, object],
                intro_text: str | None = None,
                species_options: tuple[str, ...] = (),
                parent: object | None = None,
            ) -> None:
                del specs, workflow_stage, locked_values, intro_text, species_options, parent
                self._payload = next(dialog_payloads)

        def set_values(
            self, values: Mapping[str, object], *, preferred_exclusive: str | None = None
        ) -> None:
            seen_values.append(dict(values))

        def set_advanced_names(self, names: set[str]) -> None:
            del names

        def exec(self) -> int:
            return self.DialogCode.Accepted

        def collect_values(self) -> dict[str, object]:
            return dict(self._payload)

        def advanced_names(self) -> tuple[str, ...]:
            return ()

    monkeypatch.setattr("mufinder.app.uep_widget.ParameterDialog", DummyDialog)

    widget._configure_row(row_id)
    widget._configure_row(row_id)

    assert seen_values[0]["task"] == "SINGLEPOINT"
    assert seen_values[0]["write_formatted_density"] is True
    assert seen_values[1]["cut_off_energy"] == ValueWithUnit(500.0, "eV")
    assert seen_values[1]["kpoint_mp_grid"] == (2, 2, 2)


def test_uep_widget_marks_row_configure_button_as_configured_when_customized(qapp) -> None:
    widget = UEPWidget()
    widget._add_row()

    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)
    assert configure_button.text() == "Configure..."

    row_id = widget._row_id_at(0)
    assert row_id is not None
    widget._row_parameter_values[row_id] = {"cut_off_energy": ValueWithUnit(500.0, "eV")}
    widget._apply_row_state(row_id)

    assert configure_button.text() == "Configured..."


def test_uep_widget_queued_row_locks_controls_and_parameters(qapp) -> None:
    widget = UEPWidget()
    widget._add_row()
    row_id = widget._row_id_at(0)
    assert row_id is not None
    label = widget._table.item(0, widget._LABEL_COLUMN).text()

    widget.set_runtime_states({label: UepRunState(status="queued")})

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
    assert widget._runtime_controls_locked(widget._runtime_state_for_row(row_id).status)


def test_uep_widget_delete_shortcut_removes_selected_job(qapp) -> None:
    widget = UEPWidget()
    widget._add_row()
    widget._add_row()
    removed: list[list[str]] = []
    widget.remove_requested.connect(lambda labels: removed.append(list(labels)))
    removed_label = widget._table.item(0, widget._LABEL_COLUMN).text()

    widget._table.selectRow(0)
    widget._delete_shortcut.activated.emit()

    assert removed == [[removed_label]]
    assert widget._table.rowCount() == 1


def test_uep_widget_delete_key_removes_selected_job_from_child_focus(qapp) -> None:
    widget = UEPWidget()
    widget._add_row()
    widget._add_row()
    removed: list[list[str]] = []
    widget.remove_requested.connect(lambda labels: removed.append(list(labels)))
    removed_label = widget._table.item(0, widget._LABEL_COLUMN).text()
    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)

    widget.show()
    widget._table.selectRow(0)
    configure_button.setFocus()
    qapp.processEvents()
    QTest.keyClick(configure_button, Qt.Key.Key_Delete)

    assert removed == [[removed_label]]
    assert widget._table.rowCount() == 1


def test_uep_widget_cancelled_row_keeps_parameters_editable(qapp) -> None:
    widget = UEPWidget()
    widget._add_row()
    row_id = widget._row_id_at(0)
    assert row_id is not None
    label = widget._table.item(0, widget._LABEL_COLUMN).text()

    widget.set_runtime_states({label: UepRunState(status="cancelled")})

    assert not widget._runtime_controls_locked(widget._runtime_state_for_row(row_id).status)
    label_item = widget._table.item(0, widget._LABEL_COLUMN)
    assert label_item is not None
    assert bool(label_item.flags() & Qt.ItemFlag.ItemIsEditable)
    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    assert isinstance(combo, QComboBox)
    assert combo.isEnabled()


def test_uep_widget_run_cluster_submits_selected_rows_only(qapp) -> None:
    widget = UEPWidget()
    widget._add_row()
    widget._add_row()

    emitted: list[list[str]] = []
    widget.submit_cluster_requested.connect(lambda labels: emitted.append(list(labels)))

    widget._table.selectRow(1)
    widget._emit_submit_cluster()

    assert emitted == [["source_02"]]


def test_uep_widget_restores_structured_parameter_values_from_form_state(qapp) -> None:
    widget = UEPWidget()

    widget.set_form_state(
        {
            "name_prefix": "uep_source",
            "rows": [
                {
                    "label": "source_01",
                    "supercell_matrix": None,
                    "parameter_values": {
                        "cut_off_energy": {"value": 500.0, "unit": "eV"},
                        "kpoint_mp_grid": [2, 2, 2],
                    },
                    "added_advanced": [],
                }
            ],
        }
    )

    row_id = widget._row_id_at(0)
    assert row_id is not None
    assert widget._row_parameter_values[row_id]["cut_off_energy"] == ValueWithUnit(500.0, "eV")
    assert widget._row_parameter_values[row_id]["kpoint_mp_grid"] == (2, 2, 2)


def _select_uep_rows(widget: UEPWidget, rows: tuple[int, ...]) -> None:
    selection = widget._table.selectionModel()
    assert selection is not None
    selection.clearSelection()
    for row in rows:
        selection.select(
            widget._table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )


def test_uep_batch_configure_button_tracks_selection(qapp) -> None:
    widget = UEPWidget()
    widget._add_row()
    widget._add_row()

    assert widget._batch_configure_button.text() == "Configure Selected Parameters..."
    widget._refresh_batch_button_state()
    assert widget._batch_configure_button.isEnabled() is False

    _select_uep_rows(widget, (0, 1))
    widget._refresh_batch_button_state()
    assert widget._batch_configure_button.isEnabled() is True

    widget._table.clearSelection()
    widget._refresh_batch_button_state()
    assert widget._batch_configure_button.isEnabled() is False


def test_uep_batch_configure_applies_to_all_selected_rows(qapp, monkeypatch) -> None:
    widget = UEPWidget()
    widget._add_row()
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    third_id = widget._row_id_at(2)
    unselected_id = widget._row_id_at(1)
    assert first_id is not None and third_id is not None and unselected_id is not None

    monkeypatch.setattr(
        uep_widget_module.ParameterDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        uep_widget_module.ParameterDialog,
        "collect_values",
        lambda _self: {"cut_off_energy": 555.0},
    )
    monkeypatch.setattr(
        uep_widget_module.ParameterDialog,
        "advanced_names",
        lambda _self: (),
    )

    _select_uep_rows(widget, (0, 2))
    widget._on_batch_configure_clicked()

    # The collected value is round-tripped through the widget's structured
    # sanitizer, which restores the eV unit on cut_off_energy.
    expected_cutoff = ValueWithUnit(555.0, "eV")
    assert widget._row_parameter_values[first_id]["cut_off_energy"] == expected_cutoff
    assert widget._row_parameter_values[third_id]["cut_off_energy"] == expected_cutoff
    # UEP always pins the source-job task/density flags on batch apply.
    assert widget._row_parameter_values[first_id]["task"] == "SINGLEPOINT"
    assert widget._row_parameter_values[first_id]["write_formatted_density"] is True
    # Row 1 was not selected → no custom parameters applied.
    assert "cut_off_energy" not in widget._row_parameter_values.get(unselected_id, {})


def test_uep_batch_configure_empty_selection_is_noop(qapp, monkeypatch) -> None:
    widget = UEPWidget()
    widget._add_row()

    opened: list[int] = []
    monkeypatch.setattr(
        uep_widget_module.ParameterDialog,
        "exec",
        lambda _self: opened.append(1) or QDialog.DialogCode.Accepted,
    )

    widget._table.clearSelection()
    widget._on_batch_configure_clicked()

    assert opened == []


def test_uep_configure_uses_selection_snapshot_not_collapsed(qapp, monkeypatch) -> None:
    # The per-row Configure button snapshots the selection so a platform
    # focus-collapse to the clicked row does not shrink a multi-row batch.
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    widget = UEPWidget()
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)
    assert first_id is not None and second_id is not None

    captured: list[tuple[int, ...]] = []
    # Multi-row snapshots route to _configure_rows (single-row → _configure_row).
    monkeypatch.setattr(
        UEPWidget,
        "_configure_rows",
        lambda _self, ids: captured.append(tuple(ids)),
    )
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *_args, **_kwargs: SelectionScope.ALL_SELECTED,
    )

    _select_uep_rows(widget, (0, 1))
    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, SelectionSnapshotPushButton)
    configure_button.selection_snapshot_rows = (0, 1)
    configure_button.clicked.emit()

    assert captured, "per-row Configure did not open the multi-row dialog"
    assert set(captured[0]) == {first_id, second_id}


def test_uep_batch_configure_does_not_swallow_delete_shortcut(qapp) -> None:
    # The batch-configure button must not consume the widget-scoped
    # Delete/Backspace shortcuts (WidgetWithChildrenShortcut) that remove rows.
    widget = UEPWidget()
    widget._add_row()
    widget._add_row()
    removed: list[list[str]] = []
    widget.remove_requested.connect(lambda labels: removed.append(list(labels)))

    widget._table.selectRow(0)
    widget._delete_shortcut.activated.emit()
    assert removed and widget._table.rowCount() == 1

    widget._table.selectRow(0)
    widget._backspace_shortcut.activated.emit()
    assert widget._table.rowCount() == 0


def test_uep_widget_context_menu_emits_reset_for_failed_row(qapp, monkeypatch) -> None:
    widget = UEPWidget()
    widget._add_row()
    label_item = widget._table.item(0, widget._LABEL_COLUMN)
    assert label_item is not None
    label = label_item.text()
    widget.set_runtime_states({label: UepRunState(status="failed")})

    payloads: list[object] = []
    widget.reset_failed_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    open_reset_menu(widget._table, row=0)

    assert payloads == [[label]]


def test_uep_widget_context_menu_disables_reset_for_ready_row(qapp, monkeypatch) -> None:
    widget = UEPWidget()
    widget._add_row()

    payloads: list[object] = []
    widget.reset_failed_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(widget._table, row=0)

    assert menu is not None and menu.reset_action().enabled is False
    assert payloads == []


def test_uep_wizard_button_opens_clicked_row_without_prompt(qapp, monkeypatch) -> None:
    # The wizard runs on one row at a time, so a multi-row selection neither
    # prompts nor changes which row it opens for.
    from PySide6.QtWidgets import QMessageBox

    widget = UEPWidget()
    widget._add_row()
    widget._add_row()
    prompts: list[object] = []
    monkeypatch.setattr(selection_snapshot_module, "ask_selection_scope", lambda *a, **kw: prompts.append(a))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: prompts.append(a))
    requested: list[str] = []
    widget.wizard_requested.connect(requested.append)

    _select_uep_rows(widget, (0, 1))
    wizard_button = widget._table.cellWidget(1, widget._WIZARD_COLUMN)
    wizard_button.selection_snapshot_rows = (0, 1)
    wizard_button.clicked.emit()

    assert prompts == []
    assert requested == [widget._table.item(1, widget._LABEL_COLUMN).text()]
    assert wizard_button.selection_snapshot_rows == ()


def test_uep_supercell_cancel_restores_combo_and_changes_nothing(qapp, monkeypatch) -> None:
    widget = UEPWidget()
    widget.set_supercell_options(
        [{"label": "2x2x1", "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]}]
    )
    widget._add_row()
    widget._add_row()
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)
    monkeypatch.setattr(
        selection_snapshot_module, "ask_selection_scope", lambda *a, **kw: SelectionScope.CANCEL
    )
    states: list[dict] = []
    widget.form_state_changed.connect(states.append)

    _select_uep_rows(widget, (0, 1))
    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    combo.selection_snapshot_rows = (0, 1)
    combo.setCurrentIndex(combo.findText("2x2x1"))

    assert widget.row_supercell_matrix(first_id) is None
    assert widget.row_supercell_matrix(second_id) is None
    assert combo.currentText() == "Unit cell"
    assert states == []


def test_uep_delete_shortcut_declined_removes_nothing(qapp, monkeypatch) -> None:
    widget = UEPWidget()
    widget._add_row()
    widget._add_row()
    removed: list[object] = []
    widget.remove_requested.connect(removed.append)
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)

    widget._table.selectRow(0)
    widget._backspace_shortcut.activated.emit()

    assert removed == []
    assert widget._table.rowCount() == 2


def test_uep_remove_button_single_row_is_confirmed(qapp, confirmed_destructive_actions) -> None:
    widget = UEPWidget()
    widget._add_row()
    widget._add_row()

    btn = widget._table.cellWidget(1, widget._REMOVE_COLUMN)
    btn.selection_snapshot_rows = (1,)
    btn.clicked.emit()

    assert widget._table.rowCount() == 1
    assert [prompt["text"] for prompt in confirmed_destructive_actions] == ["Remove 1 row?"]


def test_uep_wizard_button_is_not_marked_as_a_multi_row_target(qapp) -> None:
    widget = UEPWidget()
    widget._add_row()
    widget._add_row()
    _select_uep_rows(widget, (0, 1))
    prop = workflow_table_utils.MULTI_ROW_TARGET_PROPERTY

    assert bool(widget._table.cellWidget(0, widget._CONFIGURE_COLUMN).property(prop)) is True
    assert bool(widget._table.cellWidget(0, widget._WIZARD_COLUMN).property(prop)) is False
    assert widget._selection_count_label.text() == "2 selected"
    assert widget._run_cluster_button.text() == "Run Selected on Cluster"


def test_uep_row_menu_uses_the_shared_core(qapp, monkeypatch) -> None:
    from tests.support.reset_menu import install_fake_menu, open_reset_menu

    widget = UEPWidget()
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
    ]


def test_uep_batch_configure_keeps_each_rows_own_differing_values(qapp, monkeypatch) -> None:
    from PySide6.QtWidgets import QDialog

    widget = UEPWidget()
    widget._add_row()
    widget._add_row()
    first_id, second_id = widget._row_id_at(0), widget._row_id_at(1)
    assert "max_scf_cycles" in widget._specs_by_name
    for row_id, cycles in ((first_id, 80), (second_id, 120)):
        widget._row_parameter_values[row_id] = {
            **widget._effective_params_for_row(row_id),
            "max_scf_cycles": cycles,
        }
    mixed: list[set[str]] = []

    def _fake_exec(dialog) -> int:
        mixed.append(dialog.untouched_mixed_names())
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(uep_widget_module.ParameterDialog, "exec", _fake_exec)

    widget._configure_rows((first_id, second_id))

    assert mixed == [{"max_scf_cycles"}]
    assert widget._row_parameter_values[first_id]["max_scf_cycles"] == 80
    assert widget._row_parameter_values[second_id]["max_scf_cycles"] == 120
