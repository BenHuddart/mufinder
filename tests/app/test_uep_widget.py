"""Tests for the UEP workflow widget."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QPushButton, QToolButton

from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.uep_types import UepRunState
from mufinder.app.uep_widget import UEPWidget
from mufinder.castep_io.types import ValueWithUnit


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
    assert widget._project_default_cutoff.text() == "500 eV"


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

        def set_values(self, values: Mapping[str, object]) -> None:
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

        def set_values(self, values: Mapping[str, object]) -> None:
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
