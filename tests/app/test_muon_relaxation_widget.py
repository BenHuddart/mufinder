"""Behavior tests for the muon-relaxation widget."""

from __future__ import annotations

from uuid import uuid4

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QDialog, QMessageBox, QPushButton, QToolButton

from mufinder.app import workflow_table_utils
from mufinder.app import selection_snapshot as selection_snapshot_module
from mufinder.app.selection_snapshot import SelectionScope
from mufinder.app import muon_relaxation_widget as muon_relaxation_widget_module
from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.relaxation_types import RelaxationRunState
from mufinder.castep_io.gui_metadata import keyword_visible_in_workflow
from mufinder.castep_io.types import ValueWithUnit
from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE
from tests.support.reset_menu import install_fake_menu, open_reset_menu


def test_muon_relaxation_widget_stages_sites_with_positive_muon_default(qapp) -> None:
    widget = MuonRelaxationWidget()
    site_id = uuid4()

    widget.add_staged_sites(
        [
            {
                "id": str(site_id),
                "label": "Random Site 1",
                "fractional_position": [0.1, 0.2, 0.3],
            }
        ]
    )

    assert widget._table.rowCount() == 1
    assert widget._charge_tabs.count() == 2
    assert widget._charge_tabs.tabText(0) == "Positive muon"
    assert widget._charge_tabs.tabText(1) == "Muonium"
    assert widget._table.item(0, 0).text() == "Random Site 1"
    assert widget._table.horizontalHeaderItem(widget._INITIAL_FX_COLUMN).text() == "fX"
    assert widget._table.horizontalHeaderItem(widget._FINAL_FX_COLUMN).text() == "fX"
    assert "Charge" not in [
        widget._table.horizontalHeaderItem(column).text()
        for column in range(widget._table.columnCount())
        if widget._table.horizontalHeaderItem(column) is not None
    ]
    assert widget._table.horizontalHeader()._groups == (
        (widget._INITIAL_FX_COLUMN, widget._INITIAL_FZ_COLUMN, "Initial"),
        (widget._FINAL_FX_COLUMN, widget._FINAL_FZ_COLUMN, "Final"),
    )
    assert widget._table.item(0, widget._INITIAL_FX_COLUMN).text() == "0.100"
    assert widget._tables_by_charge_state["muonium"].rowCount() == 0

    config = widget._build_config_or_warn()
    assert config is not None
    assert config.rows[0].site_id == site_id
    assert config.rows[0].charge_state == "positive_muon"


def test_muon_relaxation_widget_tabbed_tables_use_square_corner_theme(qapp) -> None:
    widget = MuonRelaxationWidget()

    for table in widget._tables_by_charge_state.values():
        assert table.property("tabbedTable") == "true"
        assert table.property("squareTopCorners") == "true"


def test_muon_relaxation_widget_exposes_geometry_optimization_parameters(qapp) -> None:
    widget = MuonRelaxationWidget()

    visible_basic = {
        spec.name
        for spec in widget._specs
        if keyword_visible_in_workflow(
            spec,
            workflow_stage="muon_relaxation",
            include_advanced=False,
        )
    }

    assert {
        "geom_max_iter",
        "geom_energy_tol",
        "geom_force_tol",
        "geom_disp_tol",
        "geom_stress_tol",
        "geom_frequency_est",
    } <= visible_basic


def test_muon_relaxation_widget_defaults_continuation_to_default(qapp) -> None:
    widget = MuonRelaxationWidget()
    dialog = ParameterDialog(specs=widget._specs, workflow_stage="muon_relaxation")
    dialog.set_values(widget._default_params)

    assert widget._default_params["continuation"] == "DEFAULT"
    assert dialog.collect_values()["continuation"] == "DEFAULT"

    dialog._rows["continuation"].editor.set_value("CUSTOM")
    dialog._reset_parameter("continuation")

    assert dialog.collect_values()["continuation"] == "DEFAULT"


def test_muon_relaxation_widget_does_not_default_symmetry_generate(qapp) -> None:
    widget = MuonRelaxationWidget()
    dialog = ParameterDialog(specs=widget._specs, workflow_stage="muon_relaxation")
    dialog.set_values(widget._default_params)

    assert "symmetry_generate" not in widget._default_params
    assert "symmetry_generate" not in dialog.collect_values()


def test_muon_relaxation_widget_defaults_geometry_max_iterations_to_300(qapp) -> None:
    widget = MuonRelaxationWidget()
    dialog = ParameterDialog(specs=widget._specs, workflow_stage="muon_relaxation")
    dialog.set_values(widget._default_params)

    assert widget._default_params["geom_max_iter"] == 300
    assert dialog.collect_values()["geom_max_iter"] == 300

    dialog._rows["geom_max_iter"].editor.set_value(30)
    dialog._reset_parameter("geom_max_iter")

    assert dialog.collect_values()["geom_max_iter"] == 300


def test_muon_relaxation_widget_can_stage_both_charge_states(qapp) -> None:
    widget = MuonRelaxationWidget()
    site_id = uuid4()

    widget.add_staged_sites(
        [
            {
                "id": str(site_id),
                "label": "Shared Site",
                "fractional_position": [0.1, 0.2, 0.3],
            }
        ],
        charge_states=("positive_muon", "muonium"),
    )

    positive = widget._tables_by_charge_state["positive_muon"]
    muonium = widget._tables_by_charge_state["muonium"]
    assert positive.rowCount() == 1
    assert muonium.rowCount() == 1
    assert positive.item(0, widget._LABEL_COLUMN).text() == "Shared Site"
    assert muonium.item(0, widget._LABEL_COLUMN).text() == "Shared Site"

    config = widget._build_config_or_warn()
    assert config is not None
    assert [row.charge_state for row in config.rows] == ["positive_muon", "muonium"]


def test_muon_relaxation_widget_emits_analysis_workspace_from_selected_rows(qapp) -> None:
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [
            {
                "id": str(uuid4()),
                "label": "Site A",
                "fractional_position": [0.1, 0.2, 0.3],
            },
            {
                "id": str(uuid4()),
                "label": "Site B",
                "fractional_position": [0.4, 0.5, 0.6],
            },
        ]
    )
    captured: list[list[str]] = []
    widget.analysis_workspace_requested.connect(lambda labels: captured.append(labels))

    widget._table.selectRow(1)
    widget._create_analysis_workspace_from_selection()

    assert captured == [["positivemuon:Site B"]]


def test_muon_relaxation_widget_queued_row_locks_controls_and_parameters(qapp) -> None:
    widget = MuonRelaxationWidget()
    site_id = uuid4()
    widget.add_staged_sites(
        [
            {
                "id": str(site_id),
                "label": "Queued Site",
                "fractional_position": [0.1, 0.2, 0.3],
            }
        ]
    )
    widget.set_runtime_states({"positivemuon:Queued Site": RelaxationRunState(status="queued")})

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


def test_muon_relaxation_widget_cancelled_row_keeps_parameters_editable(qapp) -> None:
    widget = MuonRelaxationWidget()
    site_id = uuid4()
    widget.add_staged_sites(
        [
            {
                "id": str(site_id),
                "label": "Cancelled Site",
                "fractional_position": [0.1, 0.2, 0.3],
                "charge_state": "positive_muon",
            }
        ]
    )

    widget.set_runtime_states({"positivemuon:Cancelled Site": RelaxationRunState(status="cancelled")})

    label_item = widget._table.item(0, widget._LABEL_COLUMN)
    assert label_item is not None
    assert bool(label_item.flags() & Qt.ItemFlag.ItemIsEditable)
    combo = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    assert isinstance(combo, QComboBox)
    assert combo.isEnabled()
    row_id = widget._row_id_at(0)
    assert row_id is not None
    assert widget._row_parameters_locked((row_id,)) is False


def test_muon_relaxation_widget_state_roundtrip_preserves_muonium(qapp) -> None:
    site_id = uuid4()
    widget = MuonRelaxationWidget()
    widget.set_form_state(
        {
            "rows": [
                {
                    "site_id": str(site_id),
                    "label": "Manual Site 1",
                    "fractional_position": [0.4, 0.5, 0.6],
                    "charge_state": "muonium",
                }
            ]
        }
    )

    state = widget.collect_state()

    assert state["rows"][0]["site_id"] == str(site_id)
    assert state["rows"][0]["charge_state"] == "muonium"
    assert state["rows"][0]["fractional_position"] == [0.4, 0.5, 0.6]


def test_muon_relaxation_widget_allows_same_site_to_be_staged_multiple_times(qapp) -> None:
    site_id = uuid4()
    widget = MuonRelaxationWidget()

    row = {
        "id": str(site_id),
        "label": "Random Site 1",
        "fractional_position": [0.1, 0.2, 0.3],
    }
    widget.add_staged_sites([row])
    widget.add_staged_sites([row])

    assert widget._table.rowCount() == 2
    assert widget._table.item(0, 0).text() == "Random Site 1"
    assert widget._table.item(1, 0).text() == "Random Site 1 2"

    config = widget._build_config_or_warn()
    assert config is not None
    assert [entry.site_id for entry in config.rows] == [site_id, site_id]


def test_muon_relaxation_widget_delete_shortcut_removes_selected_jobs(qapp) -> None:
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [
            {"id": str(uuid4()), "label": "Site A", "fractional_position": [0.1, 0.2, 0.3]},
            {"id": str(uuid4()), "label": "Site B", "fractional_position": [0.4, 0.5, 0.6]},
        ]
    )
    removed: list[list[str]] = []
    widget.remove_requested.connect(lambda labels: removed.append(list(labels)))

    widget._table.selectRow(0)
    widget._delete_shortcut.activated.emit()

    assert removed == [["positivemuon:Site A"]]
    assert widget._table.rowCount() == 1
    assert widget._label_at_row(0, table=widget._table) == "Site B"


def test_muon_relaxation_widget_delete_shortcut_removes_current_job_without_selection(qapp) -> None:
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [
            {"id": str(uuid4()), "label": "Site A", "fractional_position": [0.1, 0.2, 0.3]},
            {"id": str(uuid4()), "label": "Site B", "fractional_position": [0.4, 0.5, 0.6]},
        ]
    )
    removed: list[list[str]] = []
    widget.remove_requested.connect(lambda labels: removed.append(list(labels)))

    widget._table.setCurrentCell(1, widget._LABEL_COLUMN)
    widget._table.clearSelection()
    widget._delete_shortcut.activated.emit()

    assert removed == [["positivemuon:Site B"]]
    assert widget._table.rowCount() == 1
    assert widget._label_at_row(0, table=widget._table) == "Site A"


def test_muon_relaxation_widget_delete_shortcuts_include_cell_widgets(qapp) -> None:
    widget = MuonRelaxationWidget()

    assert widget._delete_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    assert widget._backspace_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut


def test_muon_relaxation_widget_delete_key_removes_selected_row_from_child_focus(qapp) -> None:
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [
            {"id": str(uuid4()), "label": "Site A", "fractional_position": [0.1, 0.2, 0.3]},
            {"id": str(uuid4()), "label": "Site B", "fractional_position": [0.4, 0.5, 0.6]},
        ]
    )
    removed: list[list[str]] = []
    widget.remove_requested.connect(lambda labels: removed.append(list(labels)))
    configure_button = widget._table.cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, QPushButton)

    widget.show()
    widget._table.selectRow(0)
    configure_button.setFocus()
    qapp.processEvents()
    QTest.keyClick(configure_button, Qt.Key.Key_Delete)

    assert removed == [["positivemuon:Site A"]]
    assert widget._table.rowCount() == 1
    assert widget._label_at_row(0, table=widget._table) == "Site B"


def test_muon_relaxation_widget_backspace_shortcut_removes_selected_jobs_from_active_charge_tab(
    qapp,
) -> None:
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [
            {"id": str(uuid4()), "label": "Shared Site", "fractional_position": [0.1, 0.2, 0.3]},
        ],
        charge_states=("positive_muon", "muonium"),
    )
    removed: list[list[str]] = []
    widget.remove_requested.connect(lambda labels: removed.append(list(labels)))
    muonium_table = widget._tables_by_charge_state["muonium"]

    widget._charge_tabs.setCurrentWidget(muonium_table)
    muonium_table.selectRow(0)
    widget._backspace_shortcut.activated.emit()

    assert removed == [["muonium:Shared Site"]]
    assert muonium_table.rowCount() == 0
    assert widget._tables_by_charge_state["positive_muon"].rowCount() == 1


def test_muon_relaxation_widget_project_cutoff_removes_basis_precision_default(qapp) -> None:
    site_id = uuid4()
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [
            {
                "id": str(site_id),
                "label": "Random Site 1",
                "fractional_position": [0.1, 0.2, 0.3],
            }
        ]
    )

    widget.set_project_defaults(cutoff_energy=650.0, kpoint_grid=None, kpoint_reference_lattice=None)

    effective = widget._effective_params_for_row(1)
    assert effective["cut_off_energy"] == ValueWithUnit(650.0, "eV")
    assert "basis_precision" not in effective
    assert widget._build_config_or_warn() is not None


def test_muon_relaxation_widget_explicit_basis_precision_removes_implicit_cutoff(qapp) -> None:
    site_id = uuid4()
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [
            {
                "id": str(site_id),
                "label": "Random Site 1",
                "fractional_position": [0.1, 0.2, 0.3],
            }
        ]
    )
    widget.set_project_defaults(cutoff_energy=650.0, kpoint_grid=None, kpoint_reference_lattice=None)
    widget._row_parameter_values[1] = {"basis_precision": "FINE"}

    effective = widget._effective_params_for_row(1)
    assert effective["basis_precision"] == "FINE"
    assert "cut_off_energy" not in effective
    assert widget._build_config_or_warn() is not None


def test_muon_relaxation_configure_dialog_uses_radio_group_for_basis_control(qapp) -> None:
    widget = MuonRelaxationWidget()

    dialog = ParameterDialog(specs=widget._specs, workflow_stage="muon_relaxation")
    dialog.set_values(widget._default_params)

    assert "basis_precision" in dialog._rows
    assert "cut_off_energy" in dialog._rows
    assert dialog._rows["basis_precision"].radio is not None
    assert dialog._rows["cut_off_energy"].radio is not None
    assert "cut_off_energy" not in {spec.name for spec in dialog._available_advanced_specs()}


def test_muon_relaxation_widget_runtime_states_are_charge_aware(qapp) -> None:
    site_id = uuid4()
    widget = MuonRelaxationWidget()
    rows = [
        {
            "site_id": str(site_id),
            "label": "Shared Site",
            "fractional_position": [0.1, 0.2, 0.3],
            "charge_state": "positive_muon",
        },
        {
            "site_id": str(site_id),
            "label": "Shared Site",
            "fractional_position": [0.1, 0.2, 0.3],
            "charge_state": "muonium",
        },
    ]
    widget.set_form_state({"rows": rows})

    widget.set_runtime_states(
        {
            "positivemuon:Shared Site": RelaxationRunState(
                status="completed",
                total_energy=-12.3456789,
                final_fractional_position=(0.7, 0.8, 0.9),
            ),
            "muonium:Shared Site": RelaxationRunState(
                status="running",
                final_fractional_position=(0.4, 0.5, 0.6),
            ),
        }
    )

    assert widget._table.horizontalHeaderItem(widget._TOTAL_ENERGY_COLUMN).text() == "Total Energy\n(eV)"
    assert widget._table.item(0, widget._STATUS_COLUMN).text() == "completed"
    assert widget._table.item(0, widget._FINAL_FX_COLUMN).text() == "0.700"
    assert widget._table.item(0, widget._FINAL_FY_COLUMN).text() == "0.800"
    assert widget._table.item(0, widget._FINAL_FZ_COLUMN).text() == "0.900"
    assert widget._table.item(0, widget._TOTAL_ENERGY_COLUMN).text() == "-12.345679"
    muonium_table = widget._tables_by_charge_state["muonium"]
    assert muonium_table.item(0, widget._STATUS_COLUMN).text() == "running"
    assert muonium_table.item(0, widget._FINAL_FX_COLUMN).text() == "0.400"


def test_muon_relaxation_widget_submits_charge_qualified_selection(qapp) -> None:
    site_id = uuid4()
    widget = MuonRelaxationWidget()
    widget.set_form_state(
        {
            "rows": [
                {
                    "site_id": str(site_id),
                    "label": "Shared Site",
                    "fractional_position": [0.1, 0.2, 0.3],
                    "charge_state": "positive_muon",
                },
                {
                    "site_id": str(site_id),
                    "label": "Shared Site",
                    "fractional_position": [0.1, 0.2, 0.3],
                    "charge_state": "muonium",
                },
            ]
        }
    )
    submitted: list[list[str]] = []
    widget.submit_local_requested.connect(lambda payload: submitted.append(list(payload)))

    widget._charge_tabs.setCurrentWidget(widget._tables_by_charge_state["muonium"])
    widget._tables_by_charge_state["muonium"].selectRow(0)
    widget._run_local_selected_rows()

    assert submitted == [["muonium:Shared Site"]]


def test_muon_relaxation_widget_submits_current_charge_tab_without_selection(qapp) -> None:
    site_id = uuid4()
    widget = MuonRelaxationWidget()
    widget.set_form_state(
        {
            "rows": [
                {
                    "site_id": str(site_id),
                    "label": "Positive Only",
                    "fractional_position": [0.1, 0.2, 0.3],
                    "charge_state": "positive_muon",
                },
                {
                    "site_id": str(site_id),
                    "label": "Muonium Only",
                    "fractional_position": [0.1, 0.2, 0.3],
                    "charge_state": "muonium",
                },
            ]
        }
    )
    submitted: list[list[str]] = []
    widget.submit_local_requested.connect(lambda payload: submitted.append(list(payload)))

    widget._charge_tabs.setCurrentWidget(widget._tables_by_charge_state["muonium"])
    widget._run_local_selected_rows()

    assert submitted == [["muonium:Muonium Only"]]


def test_muon_relaxation_widget_ignores_hidden_charge_tab_selection_for_submission(qapp) -> None:
    site_id = uuid4()
    widget = MuonRelaxationWidget()
    widget.set_form_state(
        {
            "rows": [
                {
                    "site_id": str(site_id),
                    "label": "Shared Site",
                    "fractional_position": [0.1, 0.2, 0.3],
                    "charge_state": "positive_muon",
                },
                {
                    "site_id": str(site_id),
                    "label": "Shared Site",
                    "fractional_position": [0.1, 0.2, 0.3],
                    "charge_state": "muonium",
                },
            ]
        }
    )
    submitted: list[list[str]] = []
    widget.submit_local_requested.connect(lambda payload: submitted.append(list(payload)))

    widget._charge_tabs.setCurrentWidget(widget._tables_by_charge_state["positive_muon"])
    widget._tables_by_charge_state["muonium"].selectRow(0)
    widget._run_local_selected_rows()

    assert submitted == [["positivemuon:Shared Site"]]


def test_muon_relaxation_widget_locks_supercell_after_queue(qapp) -> None:
    widget = MuonRelaxationWidget()
    rows = [
        {
            "site_id": str(uuid4()),
            "label": "Queued Site",
            "fractional_position": [0.1, 0.2, 0.3],
            "charge_state": "positive_muon",
        },
        {
            "site_id": str(uuid4()),
            "label": "Ready Site",
            "fractional_position": [0.4, 0.5, 0.6],
            "charge_state": "positive_muon",
        },
    ]
    widget.set_form_state({"rows": rows})

    widget.set_runtime_states(
        {
            "positivemuon:Queued Site": RelaxationRunState(status="queued"),
            "positivemuon:Ready Site": RelaxationRunState(status="ready"),
        }
    )

    queued_supercell = widget._table.cellWidget(0, widget._SUPERCELL_COLUMN)
    ready_supercell = widget._table.cellWidget(1, widget._SUPERCELL_COLUMN)
    assert isinstance(queued_supercell, QComboBox)
    assert isinstance(ready_supercell, QComboBox)
    assert not queued_supercell.isEnabled()
    assert ready_supercell.isEnabled()

    widget.set_runtime_states({"positivemuon:Queued Site": RelaxationRunState(status="ready")})

    assert queued_supercell.isEnabled()


def test_muon_relaxation_widget_supercell_selection_can_apply_to_selected_rows(
    qapp,
    monkeypatch,
) -> None:
    widget = MuonRelaxationWidget()
    widget.set_supercell_options(
        [
            {
                "label": "Shared 2x2x1",
                "matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            }
        ]
    )
    widget.add_staged_sites(
        [
            {"id": str(uuid4()), "label": "Site A", "fractional_position": [0.1, 0.2, 0.3]},
            {"id": str(uuid4()), "label": "Site B", "fractional_position": [0.4, 0.5, 0.6]},
        ]
    )
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)
    assert first_id is not None
    assert second_id is not None

    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *_args: SelectionScope.ALL_SELECTED,
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
    # Simulate the press snapshot (real interaction captures this via mousePressEvent).
    combo.selection_snapshot_rows = (0, 1)
    combo.setCurrentIndex(combo.findText("Shared 2x2x1"))

    expected = ((2, 0, 0), (0, 2, 0), (0, 0, 1))
    assert widget.row_supercell_matrix(first_id) == expected
    assert widget.row_supercell_matrix(second_id) == expected


def _stage_muon_rows(widget: MuonRelaxationWidget, count: int) -> None:
    widget.add_staged_sites(
        [
            {
                "id": str(uuid4()),
                "label": f"Site {index}",
                "fractional_position": [0.1 * index, 0.2, 0.3],
            }
            for index in range(count)
        ]
    )


def _select_muon_rows(widget: MuonRelaxationWidget, rows: tuple[int, ...]) -> None:
    table = widget._active_table()
    selection = table.selectionModel()
    assert selection is not None
    selection.clearSelection()
    for row in rows:
        selection.select(
            table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )


def test_muon_relaxation_batch_configure_button_tracks_selection(qapp) -> None:
    widget = MuonRelaxationWidget()
    _stage_muon_rows(widget, 2)

    assert widget._batch_configure_button.text() == "Configure Selected Parameters..."
    widget._refresh_batch_button_state()
    assert widget._batch_configure_button.isEnabled() is False

    _select_muon_rows(widget, (0, 1))
    widget._refresh_batch_button_state()
    assert widget._batch_configure_button.isEnabled() is True

    widget._active_table().clearSelection()
    widget._refresh_batch_button_state()
    assert widget._batch_configure_button.isEnabled() is False


def test_muon_relaxation_batch_configure_applies_to_all_selected_rows(
    qapp, monkeypatch
) -> None:
    widget = MuonRelaxationWidget()
    _stage_muon_rows(widget, 3)
    first_id = widget._row_id_at(0)
    third_id = widget._row_id_at(2)
    unselected_id = widget._row_id_at(1)
    assert first_id is not None and third_id is not None and unselected_id is not None

    monkeypatch.setattr(
        muon_relaxation_widget_module.ParameterDialog,
        "exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        muon_relaxation_widget_module.ParameterDialog,
        "collect_values",
        lambda _self: {"task": "GEOMETRYOPTIMIZATION", "max_scf_cycles": 88},
    )
    monkeypatch.setattr(
        muon_relaxation_widget_module.ParameterDialog,
        "advanced_names",
        lambda _self: (),
    )

    _select_muon_rows(widget, (0, 2))
    widget._configure_selected_rows()

    assert widget._row_parameter_values[first_id]["max_scf_cycles"] == 88
    assert widget._row_parameter_values[third_id]["max_scf_cycles"] == 88
    assert widget._row_parameter_values.get(unselected_id, {}) == {}


def test_muon_relaxation_batch_configure_empty_selection_shows_info(
    qapp, monkeypatch
) -> None:
    widget = MuonRelaxationWidget()
    _stage_muon_rows(widget, 1)

    infos: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )
    opened: list[int] = []
    monkeypatch.setattr(
        MuonRelaxationWidget,
        "_open_row_parameter_dialog",
        lambda _self, _ids: opened.append(1),
    )

    widget._active_table().clearSelection()
    widget._configure_selected_rows()

    assert infos == ["No rows selected"]
    assert opened == []


def test_muon_relaxation_configure_uses_selection_snapshot_not_collapsed(
    qapp, monkeypatch
) -> None:
    # The per-row Configure button snapshots the selection so a platform
    # focus-collapse to the clicked row does not shrink a multi-row batch.
    from PySide6.QtWidgets import QPushButton as _QPushButton

    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    widget = MuonRelaxationWidget()
    _stage_muon_rows(widget, 2)
    first_id = widget._row_id_at(0)
    second_id = widget._row_id_at(1)
    assert first_id is not None and second_id is not None

    captured: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        MuonRelaxationWidget,
        "_open_row_parameter_dialog",
        lambda _self, ids: captured.append(ids),
    )
    monkeypatch.setattr(
        selection_snapshot_module,
        "ask_selection_scope",
        lambda *_args, **_kwargs: SelectionScope.ALL_SELECTED,
    )

    _select_muon_rows(widget, (0, 1))
    configure_button = widget._active_table().cellWidget(0, widget._CONFIGURE_COLUMN)
    assert isinstance(configure_button, SelectionSnapshotPushButton)
    assert isinstance(configure_button, _QPushButton)
    configure_button.selection_snapshot_rows = (0, 1)
    configure_button.clicked.emit()

    assert captured, "per-row Configure did not open the dialog"
    assert set(captured[0]) == {first_id, second_id}


def test_muon_relaxation_batch_configure_does_not_swallow_delete_shortcut(qapp) -> None:
    # Adding the batch-configure button must not consume the widget-scoped
    # Delete/Backspace shortcuts (WidgetWithChildrenShortcut) that remove rows.
    widget = MuonRelaxationWidget()
    _stage_muon_rows(widget, 2)
    removed: list[list[str]] = []
    widget.remove_requested.connect(lambda labels: removed.append(list(labels)))

    _select_muon_rows(widget, (0,))
    widget._delete_shortcut.activated.emit()
    assert removed and widget._table.rowCount() == 1

    _select_muon_rows(widget, (0,))
    widget._backspace_shortcut.activated.emit()
    assert widget._table.rowCount() == 0


def test_muon_relaxation_widget_context_menu_emits_reset_for_failed_row(qapp, monkeypatch) -> None:
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [
            {
                "id": str(uuid4()),
                "label": "Failed Site",
                "fractional_position": [0.1, 0.2, 0.3],
            }
        ]
    )
    widget.set_runtime_states({"positivemuon:Failed Site": RelaxationRunState(status="failed")})

    payloads: list[object] = []
    widget.reset_failed_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    open_reset_menu(widget._table, row=0)

    # The payload is the runtime submission key, not the bare label, so
    # MuonRelaxationController.jobs_for_keys can resolve the right job.
    assert payloads == [["positivemuon:Failed Site"]]


def test_muon_relaxation_widget_context_menu_targets_the_clicked_charge_state_table(
    qapp, monkeypatch
) -> None:
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [
            {
                "id": str(uuid4()),
                "label": "Mu Site",
                "fractional_position": [0.1, 0.2, 0.3],
            }
        ],
        charge_states=(MUONIUM_CHARGE_STATE,),
    )
    widget.set_runtime_states({"muonium:Mu Site": RelaxationRunState(status="failed")})
    muonium_table = widget._tables_by_charge_state[MUONIUM_CHARGE_STATE]
    assert muonium_table.rowCount() == 1
    # The positive-muon tab stays current, so a menu that read _active_table()
    # instead of its own table would resolve the wrong (empty) table.
    assert widget._active_table() is not muonium_table

    payloads: list[object] = []
    widget.reset_failed_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    open_reset_menu(muonium_table, row=0)

    assert payloads == [["muonium:Mu Site"]]


def test_muon_relaxation_widget_context_menu_disables_reset_for_queued_row(qapp, monkeypatch) -> None:
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [
            {
                "id": str(uuid4()),
                "label": "Queued Site",
                "fractional_position": [0.1, 0.2, 0.3],
            }
        ]
    )
    widget.set_runtime_states({"positivemuon:Queued Site": RelaxationRunState(status="queued")})

    payloads: list[object] = []
    widget.reset_failed_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(widget._table, row=0)

    assert menu is not None and menu.reset_action().enabled is False
    assert payloads == []


def test_muon_relaxation_widget_continue_action_defaults_disabled_when_unwired(
    qapp, monkeypatch
) -> None:
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [{"id": str(uuid4()), "label": "Site 1", "fractional_position": [0.1, 0.2, 0.3]}]
    )
    widget.set_runtime_states({"positivemuon:Site 1": RelaxationRunState(status="completed")})

    install_fake_menu(monkeypatch)
    menu = open_reset_menu(widget._table, row=0)

    assert menu is not None
    action = menu.action("Continue relaxation…")
    assert action is not None
    assert action.enabled is False


def test_muon_relaxation_widget_continue_action_uses_injected_predicate_with_charge_key(
    qapp, monkeypatch
) -> None:
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [{"id": str(uuid4()), "label": "Site 1", "fractional_position": [0.1, 0.2, 0.3]}]
    )
    widget.set_runtime_states({"positivemuon:Site 1": RelaxationRunState(status="completed")})

    seen_keys: list[object] = []

    def predicate(key: object) -> bool:
        seen_keys.append(key)
        return key == "positivemuon:Site 1"

    widget.set_continuable_predicate(predicate)

    install_fake_menu(monkeypatch)
    menu = open_reset_menu(widget._table, row=0)

    assert menu is not None
    action = menu.action("Continue relaxation…")
    assert action is not None
    assert action.enabled is True
    assert seen_keys == ["positivemuon:Site 1"]


def test_muon_relaxation_widget_continue_action_emits_charge_qualified_key(
    qapp, monkeypatch
) -> None:
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [{"id": str(uuid4()), "label": "Mu Site", "fractional_position": [0.1, 0.2, 0.3]}],
        charge_states=(MUONIUM_CHARGE_STATE,),
    )
    widget.set_continuable_predicate(lambda _key: True)
    muonium_table = widget._tables_by_charge_state[MUONIUM_CHARGE_STATE]

    payloads: list[object] = []
    widget.continue_relaxation_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(muonium_table, row=0)
    action = menu.action("Continue relaxation…")
    assert action is not None
    action.triggered.emit()

    assert payloads == ["muonium:Mu Site"]


def test_muon_relaxation_widget_continue_dialog_seed_includes_charge_locked_value(
    qapp,
) -> None:
    widget = MuonRelaxationWidget()
    widget.add_staged_sites(
        [{"id": str(uuid4()), "label": "Site 1", "fractional_position": [0.1, 0.2, 0.3]}]
    )
    row_id = widget._row_id_at(0)
    widget._row_parameter_values[row_id] = {"geom_max_iter": 250}
    widget._row_added_advanced[row_id] = {"geom_max_iter"}

    seed = widget.continue_relaxation_dialog_seed("positivemuon:Site 1")

    assert seed is not None
    assert seed["workflow_stage"] == "muon_relaxation"
    assert seed["specs"] == widget._specs
    assert seed["parameter_values"]["geom_max_iter"] == 250
    assert seed["added_advanced"] == ("geom_max_iter",)
    assert seed["locked_values"]["task"] == "GEOMETRYOPTIMIZATION"
    assert seed["locked_values"]["charge"] == "+1 (positive muon)"
    assert seed["species_options"] == widget._parameter_species_options

    assert widget.continue_relaxation_dialog_seed("positivemuon:does-not-exist") is None


def test_muon_relaxation_delete_shortcut_declined_removes_nothing(qapp, monkeypatch) -> None:
    widget = MuonRelaxationWidget()
    _stage_muon_rows(widget, 2)
    removed: list[object] = []
    widget.remove_requested.connect(removed.append)
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)

    widget._table.selectRow(0)
    widget._delete_shortcut.activated.emit()

    assert removed == []
    assert widget._table.rowCount() == 2


def test_muon_relaxation_remove_button_single_row_is_confirmed(
    qapp, confirmed_destructive_actions
) -> None:
    widget = MuonRelaxationWidget()
    _stage_muon_rows(widget, 2)

    btn = widget._table.cellWidget(0, widget._REMOVE_COLUMN)
    btn.selection_snapshot_rows = (0,)
    btn.clicked.emit()

    assert widget._table.rowCount() == 1
    assert [prompt["text"] for prompt in confirmed_destructive_actions] == ["Remove 1 row?"]
