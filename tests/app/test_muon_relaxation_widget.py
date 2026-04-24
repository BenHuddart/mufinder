"""Behavior tests for the muon-relaxation widget."""

from __future__ import annotations

from uuid import uuid4

from PySide6.QtWidgets import QComboBox, QHeaderView

from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.relaxation_types import RelaxationRunState
from mufinder.castep_io.types import ValueWithUnit


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
    assert widget._table.item(0, 0).text() == "Random Site 1"
    assert widget._table.horizontalHeaderItem(widget._INITIAL_FX_COLUMN).text() == "fX"
    assert widget._table.horizontalHeaderItem(widget._FINAL_FX_COLUMN).text() == "fX"
    assert widget._table.horizontalHeaderItem(widget._CHARGE_COLUMN).text() == "Charge"
    assert (
        widget._table.horizontalHeader().sectionResizeMode(widget._CHARGE_COLUMN)
        == QHeaderView.ResizeMode.Fixed
    )
    assert widget._table.columnWidth(widget._CHARGE_COLUMN) >= (
        widget._table.fontMetrics().horizontalAdvance("positive muon") + 72
    )
    assert widget._table.horizontalHeader()._groups == (
        (widget._INITIAL_FX_COLUMN, widget._INITIAL_FZ_COLUMN, "Initial"),
        (widget._FINAL_FX_COLUMN, widget._FINAL_FZ_COLUMN, "Final"),
    )
    assert widget._table.item(0, widget._INITIAL_FX_COLUMN).text() == "0.100"
    charge_combo = widget._table.cellWidget(0, 2)
    assert isinstance(charge_combo, QComboBox)
    assert charge_combo.currentData() == "positive_muon"

    config = widget._build_config_or_warn()
    assert config is not None
    assert config.rows[0].site_id == site_id
    assert config.rows[0].charge_state == "positive_muon"


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
    assert widget._table.item(1, widget._STATUS_COLUMN).text() == "running"
    assert widget._table.item(1, widget._FINAL_FX_COLUMN).text() == "0.400"


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

    widget._table.selectRow(1)
    widget._run_local_selected_rows()

    assert submitted == [["muonium:Shared Site"]]


def test_muon_relaxation_widget_locks_supercell_and_charge_after_queue(qapp) -> None:
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
    queued_charge = widget._table.cellWidget(0, widget._CHARGE_COLUMN)
    ready_supercell = widget._table.cellWidget(1, widget._SUPERCELL_COLUMN)
    ready_charge = widget._table.cellWidget(1, widget._CHARGE_COLUMN)
    assert isinstance(queued_supercell, QComboBox)
    assert isinstance(queued_charge, QComboBox)
    assert isinstance(ready_supercell, QComboBox)
    assert isinstance(ready_charge, QComboBox)
    assert not queued_supercell.isEnabled()
    assert not queued_charge.isEnabled()
    assert ready_supercell.isEnabled()
    assert ready_charge.isEnabled()

    widget.set_runtime_states({"positivemuon:Queued Site": RelaxationRunState(status="ready")})

    assert queued_supercell.isEnabled()
    assert queued_charge.isEnabled()
