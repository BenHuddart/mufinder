"""Tests for the Polarisation page."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence

from mufinder.app import polarisation_widget as module
from mufinder.app import workflow_table_utils
from mufinder.app.polarisation_widget import PolarisationWidget, export_text
from mufinder.science.polarisation_simulation import PolarisationSettings
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    UEP_SITE_SOURCE,
    store_magnetic_field_results,
)
from mufinder.services.polarisation_simulations import polarisation_simulations
from mufinder.services.project_service import ProjectService
from mufinder.services.project_summary import MAGNETIC_FIELD_WORKSPACES_KEY
from tests.support.controllers import muon_host_document
from tests.support.field_results import field_result


def _store_fields(document, owner: str, *results, site_source: str | None = None) -> None:
    value: dict[str, object] = {
        "kind": "commensurate",
        "results": [result.to_json() for result in results],
    }
    if site_source:
        value["site_source"] = site_source
    store_magnetic_field_results(document, owner, {"magnetic_field_results": value})


@pytest.fixture
def document(tmp_path: Path):
    document, _host = muon_host_document(ProjectService(), tmp_path)
    _store_fields(
        document,
        MAGNETIC_FIELD_PAGE_OWNER,
        field_result((0.0, 0.0, 0.01), position=(0.25, 0.25, 0.25)),
    )
    document.ui_state[MAGNETIC_FIELD_WORKSPACES_KEY] = [
        {"id": "ws-lf", "title": "In field"},
        {"id": "ws-uep", "title": "UEP site"},
    ]
    _store_fields(
        document,
        "ws-lf",
        field_result((0.01, 0.0, 0.05), applied=(0.0, 0.0, 0.05), position=(0.5, 0.5, 0.5)),
    )
    _store_fields(
        document,
        "ws-uep",
        field_result((0.02, 0.0, 0.0), label="uep 1", position=(0.1, 0.2, 0.3)),
        site_source=UEP_SITE_SOURCE,
    )
    return document


@pytest.fixture
def page(qapp, document):
    widget = PolarisationWidget()
    widget.set_document(document)
    # A short grid keeps each simulation quick.
    widget._end_time.setValue(4.0)
    widget._time_points.setValue(401)
    yield widget
    widget.close()


def _checked_owners(page: PolarisationWidget) -> list[str]:
    return [source.owner for source in page._checked_sources()]


def _set_checked(page: PolarisationWidget, owners: set[str]) -> None:
    table = page._sources_table
    for row in range(table.rowCount()):
        item = table.item(row, module._USE_COLUMN)
        item.setCheckState(
            Qt.CheckState.Checked
            if item.data(module._OWNER_ROLE) in owners
            else Qt.CheckState.Unchecked
        )


def test_page_lists_each_stored_field_result_and_checks_the_first(page) -> None:
    table = page._sources_table

    assert table.rowCount() == 3
    labels = [table.item(row, module._SOURCE_COLUMN).text() for row in range(3)]
    assert labels == [
        "Magnetic Fields · 1 field at site 1",
        "In field · 1 field at site 1",
        "UEP site · 1 field at uep 1 (unrelaxed estimate)",
    ]
    assert table.item(1, module._APPLIED_COLUMN).text() == "0, 0, 0.05"
    assert _checked_owners(page) == [MAGNETIC_FIELD_PAGE_OWNER]
    assert not page._open_fields_button.isVisibleTo(page)


def test_page_without_field_results_points_to_magnetic_fields(qapp, tmp_path) -> None:
    document, _host = muon_host_document(ProjectService(), tmp_path)
    page = PolarisationWidget()
    page.set_document(document)
    requested: list[bool] = []
    page.open_magnetic_fields_requested.connect(lambda: requested.append(True))

    assert page._sources_table.rowCount() == 0
    assert "No field results yet" in page._sources_hint.text()
    assert page._open_fields_button.isVisibleTo(page)
    assert not page._add_button.isEnabled()
    page._open_fields_button.click()
    assert requested == [True]
    page.close()


def test_geometry_and_sample_enable_the_controls_they_use(page) -> None:
    assert page._spin.isEnabled()
    assert not page._detector.isEnabled()
    page._detector_along_spin.setChecked(False)
    assert page._detector.isEnabled()

    page._geometry.setCurrentIndex(page._geometry.findData("longitudinal"))
    assert not page._spin.isEnabled()
    assert not page._detector.isEnabled()

    assert not page._sphere_points.isEnabled()
    page._sample.setCurrentIndex(page._sample.findData(True))
    assert page._sphere_points.isEnabled()

    page._sigma_from_widths.setChecked(True)
    assert not page._sigma.isEnabled()
    assert not page._dynamics_method.isEnabled()
    page._collision_rate.setValue(1.0)
    assert page._dynamics_method.isEnabled()


@pytest.mark.parametrize(
    ("geometry", "polycrystal", "detector_along_spin", "allowed"),
    [
        ("zero_field", True, True, True),
        ("longitudinal", True, True, True),
        ("zero_field", False, True, False),
        ("zero_field", True, False, False),
        ("transverse", True, True, False),
    ],
)
def test_dynamics_are_offered_only_where_symmetry_allows_them(
    page, geometry: str, polycrystal: bool, detector_along_spin: bool, allowed: bool
) -> None:
    page._collision_rate.setValue(2.0)
    page._geometry.setCurrentIndex(page._geometry.findData(geometry))
    page._sample.setCurrentIndex(page._sample.findData(polycrystal))
    page._detector_along_spin.setChecked(detector_along_spin)

    assert page._collision_rate.isEnabled() is allowed
    assert page._dynamics_method.isEnabled() is allowed
    assert page.settings().collision_rate_per_us == (2.0 if allowed else 0.0)
    if not allowed:
        assert "polycrystal" in page._collision_rate.toolTip()


def test_an_applied_field_can_be_added_only_to_zero_field_results(page) -> None:
    assert page._added_field.isEnabled()
    assert "can be added" in page._applied_summary.text()

    _set_checked(page, {"ws-lf"})
    assert not page._added_field.isEnabled()
    assert page._applied_summary.text() == "From the field results: (0, 0, 0.05) T."
    assert page.settings().added_applied_field_T is None

    _set_checked(page, {MAGNETIC_FIELD_PAGE_OWNER, "ws-lf"})
    assert "different applied fields" in page._applied_summary.text()
    _set_checked(page, set())
    assert page._applied_summary.text() == "Select field results."


def test_adding_a_simulation_stores_shows_and_plots_it(page, document) -> None:
    changed: list[bool] = []
    page.simulations_changed.connect(lambda: changed.append(True))
    _set_checked(page, {MAGNETIC_FIELD_PAGE_OWNER, "ws-uep"})
    page._title_edit.setText("ZF pair")
    page._lambda_z.setValue(0.1)

    page._add_button.click()

    (record,) = polarisation_simulations(document)
    assert record.title == "ZF pair"
    assert record.settings.lambda_z_per_us == pytest.approx(0.1)
    assert [source.owner for source in record.sources] == [MAGNETIC_FIELD_PAGE_OWNER, "ws-uep"]
    assert changed == [True]
    table = page._simulations_table
    assert [table.item(0, column).text() for column in range(5)] == [
        "ZF pair",
        "Zero field",
        "Single crystal",
        "Magnetic Fields, UEP site",
        "Current, estimate",
    ]
    assert page.selected_simulation_id() == record.simulation_id
    assert page._estimate_note.isVisibleTo(page)
    time_axis, spectrum_axis = page._figure.axes[:2]
    # Two site traces and their average.
    assert len(time_axis.lines) == 3
    np.testing.assert_allclose(
        time_axis.lines[-1].get_ydata(), record.simulation.static.polarisation
    )
    assert spectrum_axis.lines
    assert page._recalculate_button.isEnabled() and page._export_button.isEnabled()


def test_a_failed_simulation_reports_why_and_stores_nothing(page, document) -> None:
    page._geometry.setCurrentIndex(page._geometry.findData("longitudinal"))
    page._added_field.set_value((0.0, 0.0, 0.0))

    page._add_button.click()

    assert page._status.text().startswith("Simulation failed:")
    assert "applied field" in page._status.text()
    assert polarisation_simulations(document) == []


def test_invalid_settings_are_reported_before_running(page, document) -> None:
    page._spin.set_value((0.0, 0.0, 0.0))

    page._add_button.click()

    assert page._status.text().startswith("Invalid settings:")
    assert polarisation_simulations(document) == []


def test_selecting_a_simulation_loads_its_settings_and_sources(page) -> None:
    page._title_edit.setText("first")
    page._add_button.click()
    first = page.selected_simulation_id()
    _set_checked(page, {"ws-lf"})
    page._geometry.setCurrentIndex(page._geometry.findData("transverse"))
    page._title_edit.setText("second")
    page._add_button.click()

    page._select_simulation(first)

    assert page._title_edit.text() == "first"
    assert page._geometry.currentData() == "zero_field"
    assert _checked_owners(page) == [MAGNETIC_FIELD_PAGE_OWNER]


def test_recalculate_replaces_the_selected_simulation(page, document) -> None:
    page._sample.setCurrentIndex(page._sample.findData(True))
    page._add_button.click()
    simulation_id = page.selected_simulation_id()

    page._collision_rate.setValue(2.0)
    page._recalculate_button.click()

    (record,) = polarisation_simulations(document)
    assert record.simulation_id == simulation_id
    assert record.simulation.dynamic is not None
    assert record.simulation.dynamic.collision_rate_per_us == pytest.approx(2.0)
    assert page._figure.axes[0].lines[-1].get_label() == "dynamic, ν = 2 μs⁻¹"


def test_recalculated_fields_mark_a_simulation_stale(page, document) -> None:
    page._add_button.click()

    _store_fields(document, MAGNETIC_FIELD_PAGE_OWNER, field_result((0.0, 0.0, 0.02)))
    page.refresh()

    assert page._simulations_table.item(0, 4).text() == "Stale"
    assert page._stale_note.isVisibleTo(page)
    assert "Magnetic Fields" in page._stale_note.text()
    page._recalculate_button.click()
    assert page._simulations_table.item(0, 4).text() == "Current"
    assert not page._stale_note.isVisibleTo(page)


def test_remove_asks_first_and_binds_delete_and_backspace(
    page, document, monkeypatch, confirmed_destructive_actions
) -> None:
    page._add_button.click()
    page._add_button.click()
    keys = {shortcut.key() for shortcut in page._delete_shortcuts}
    assert keys == {
        QKeySequence(QKeySequence.StandardKey.Delete),
        QKeySequence(Qt.Key.Key_Backspace),
    }

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)
    page._remove_button.click()
    assert len(polarisation_simulations(document)) == 2

    monkeypatch.undo()
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda *args, **kwargs: confirmed_destructive_actions.append({"text": args[2]}) or True,
    )
    page._delete_shortcuts[0].activated.emit()
    assert len(polarisation_simulations(document)) == 1
    page._select_simulation(polarisation_simulations(document)[0].simulation_id)
    page._delete_shortcuts[1].activated.emit()
    assert polarisation_simulations(document) == []
    assert confirmed_destructive_actions[-1]["text"].startswith("Remove the simulation")
    assert not page._remove_button.isEnabled()


def test_form_state_round_trips_settings_sources_and_selection(page, document) -> None:
    _set_checked(page, {"ws-lf"})
    page._geometry.setCurrentIndex(page._geometry.findData("longitudinal"))
    page._sample.setCurrentIndex(page._sample.findData(True))
    page._sphere_points.setValue(120)
    page._apodisation_sigma.setText("3")
    page._title_edit.setText("LF powder")
    page._add_button.click()
    state = page.collect_state()

    restored = PolarisationWidget()
    restored.set_document(document)
    restored.set_form_state(state)

    assert restored.selected_simulation_id() == page.selected_simulation_id()
    assert restored.settings() == page.settings()
    assert restored.settings().apodisation_sigma_us == pytest.approx(3.0)
    assert _checked_owners(restored) == ["ws-lf"]
    assert restored._title_edit.text() == "LF powder"
    restored.set_form_state({"settings": {"time_points": "lots"}, "sources": [{"bad": 1}]})
    assert restored.settings().time_points == PolarisationSettings().time_points
    assert _checked_owners(restored) == []
    restored.close()


def test_select_sources_checks_exactly_those_owners(page) -> None:
    emitted: list[dict] = []
    page.form_state_changed.connect(emitted.append)

    page.select_sources(["ws-lf"])

    assert _checked_owners(page) == ["ws-lf"]
    assert emitted[-1]["sources"][0]["owner"] == "ws-lf"


def test_export_writes_the_traces_and_spectrum(page, tmp_path, monkeypatch) -> None:
    _set_checked(page, {MAGNETIC_FIELD_PAGE_OWNER, "ws-uep"})
    page._sample.setCurrentIndex(page._sample.findData(True))
    page._collision_rate.setValue(0.5)
    page._title_edit.setText("ZF pair")
    page._add_button.click()
    target = tmp_path / "out.dat"
    monkeypatch.setattr(module, "ask_save_path", lambda *_args, **_kwargs: target)
    saved: list[Path] = []
    monkeypatch.setattr(module, "report_export_saved", lambda _parent, path: saved.append(path))

    page._export_button.click()

    text = target.read_text(encoding="utf-8")
    assert saved == [target]
    assert "# Simulation: ZF pair" in text
    assert "# Estimate:" in text
    assert "# Time domain. Columns: t_us P_site_1 P_uep_1 P_average P_dynamic" in text
    time_block, spectrum_block = text.split("\n\n")
    time_rows = [line for line in time_block.splitlines() if not line.startswith("#")]
    assert len(time_rows) == 401
    assert len(time_rows[0].split()) == 5
    assert spectrum_block.startswith("# Spectrum. Columns: f_MHz B_T amplitude_per_MHz")
    record = polarisation_simulations(page._document)[0]
    assert text == export_text(record)


def test_an_unsaved_form_keeps_the_first_field_result_ticked(page) -> None:
    page.set_form_state({})

    assert _checked_owners(page) == [MAGNETIC_FIELD_PAGE_OWNER]
