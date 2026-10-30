"""Tests for the Polarisation page.

The tests reach the page's controls by the names a user sees (accessible
names, button texts and form-row labels) through ``find_control``.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
)

from mufinder.app import polarisation_widget as module
from mufinder.app import workflow_table_utils
from mufinder.app.blocking_call import BlockingCallCancelled
from mufinder.app.decimal_input import VectorInput
from mufinder.app.polarisation_widget import PolarisationWidget, export_text
from mufinder.app.theme import plot_series_colours
from mufinder.science.polarisation import PolarisationError
from mufinder.science.polarisation_simulation import PolarisationSettings
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    UEP_SITE_SOURCE,
    UNRELAXED_RESULTS_NOTE,
    store_magnetic_field_results,
)
from mufinder.services.polarisation_simulations import polarisation_simulations
from mufinder.services.project_service import ProjectService
from mufinder.services.project_summary import MAGNETIC_FIELD_WORKSPACES_KEY
from tests.support.controllers import muon_host_document
from tests.support.field_results import field_result
from tests.support.qt import find_control

# The title each source's row starts with.
_SOURCE_TITLES = {
    MAGNETIC_FIELD_PAGE_OWNER: "Magnetic Fields",
    "ws-lf": "In field",
    "ws-uep": "UEP site",
}


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
    find_control(widget, "End Time", QDoubleSpinBox).setValue(4.0)
    find_control(widget, "Points", QSpinBox).setValue(401)
    yield widget
    widget.close()


def _button(page: PolarisationWidget, text: str) -> QPushButton:
    return find_control(page, text, QPushButton)


def _combo(page: PolarisationWidget, name: str) -> QComboBox:
    return find_control(page, name, QComboBox)


def _choose(page: PolarisationWidget, name: str, data: object) -> None:
    combo = _combo(page, name)
    combo.setCurrentIndex(combo.findData(data))


def _check(page: PolarisationWidget, text: str) -> QCheckBox:
    return find_control(page, text, QCheckBox)


def _rate(page: PolarisationWidget, name: str) -> QDoubleSpinBox:
    return find_control(page, name, QDoubleSpinBox)


def _edit(page: PolarisationWidget, name: str) -> QLineEdit:
    return find_control(page, name, QLineEdit)


def _vector(page: PolarisationWidget, name: str) -> VectorInput:
    box = find_control(page, f"{name} x", QLineEdit).parentWidget()
    assert isinstance(box, VectorInput)
    return box


def _table(page: PolarisationWidget, name: str) -> QTableWidget:
    return find_control(page, name, QTableWidget)


def _column(table: QTableWidget, header: str) -> int:
    headers = [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
    return headers.index(header)


def _note(page: PolarisationWidget, start: str) -> str | None:
    """The text of the shown label that starts with ``start``, or ``None``."""

    texts = [
        label.text()
        for label in page.findChildren(QLabel)
        if label.isVisibleTo(page) and label.text().startswith(start)
    ]
    assert len(texts) <= 1
    return texts[0] if texts else None


def _figure(page: PolarisationWidget):
    return page.findChild(FigureCanvasQTAgg).figure


def _checked_owners(page: PolarisationWidget) -> list[str]:
    return [source["owner"] for source in page.collect_state()["sources"]]


def _set_checked(page: PolarisationWidget, owners: set[str]) -> None:
    table = _table(page, "Field Results")
    use, title = _column(table, "Use"), _column(table, "Field result")
    titles = {_SOURCE_TITLES[owner] for owner in owners}
    for row in range(table.rowCount()):
        checked = table.item(row, title).text().split(" · ")[0] in titles
        table.item(row, use).setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )


def _select(page: PolarisationWidget, title: str) -> None:
    table = _table(page, "Simulations")
    (row,) = (row for row in range(table.rowCount()) if table.item(row, 0).text() == title)
    table.selectRow(row)


def test_page_lists_each_stored_field_result_and_checks_the_first(page) -> None:
    table = _table(page, "Field Results")

    assert table.rowCount() == 3
    labels = [table.item(row, _column(table, "Field result")).text() for row in range(3)]
    assert labels == [
        "Magnetic Fields · 1 field at site 1",
        "In field · 1 field at site 1",
        "UEP site · 1 field at uep 1 (unrelaxed estimate)",
    ]
    assert table.item(1, _column(table, "Applied field (T)")).text() == "0, 0, 0.05"
    assert _checked_owners(page) == [MAGNETIC_FIELD_PAGE_OWNER]
    assert not _button(page, "Go to Magnetic Fields →").isVisibleTo(page)


def test_page_without_field_results_points_to_magnetic_fields(qapp, tmp_path) -> None:
    document, _host = muon_host_document(ProjectService(), tmp_path)
    page = PolarisationWidget()
    page.set_document(document)
    requested: list[bool] = []
    page.open_magnetic_fields_requested.connect(lambda: requested.append(True))

    assert _table(page, "Field Results").rowCount() == 0
    assert _note(page, "No field results yet")
    assert _button(page, "Go to Magnetic Fields →").isVisibleTo(page)
    assert not _button(page, "Add Simulation").isEnabled()
    _button(page, "Go to Magnetic Fields →").click()
    assert requested == [True]
    page.close()


def test_geometry_and_sample_enable_the_controls_they_use(page) -> None:
    assert _vector(page, "Initial Polarisation").isEnabled()
    assert not _vector(page, "Detector").isEnabled()
    _check(page, "Along the spin").setChecked(False)
    assert _vector(page, "Detector").isEnabled()

    _choose(page, "Geometry", "longitudinal")
    assert not _vector(page, "Initial Polarisation").isEnabled()
    assert not _vector(page, "Detector").isEnabled()

    assert not find_control(page, "Sphere points", QSpinBox).isEnabled()
    _choose(page, "Sample", True)
    assert find_control(page, "Sphere points", QSpinBox).isEnabled()

    _check(page, "From nuclear widths").setChecked(True)
    assert not _rate(page, "σ").isEnabled()
    assert not _combo(page, "Dynamics Method").isEnabled()
    _rate(page, "Collision Rate ν").setValue(1.0)
    assert _combo(page, "Dynamics Method").isEnabled()


@pytest.mark.parametrize(
    ("geometry", "polycrystal", "detector", "allowed"),
    [
        ("zero_field", True, None, True),
        ("longitudinal", True, None, True),
        ("zero_field", False, None, False),
        ("zero_field", True, (1.0, 0.0, 0.0), False),
        ("zero_field", True, (0.0, 0.0, 2.0), True),
        ("transverse", True, None, False),
    ],
)
def test_dynamics_are_offered_only_where_symmetry_allows_them(
    page, geometry: str, polycrystal: bool, detector, allowed: bool
) -> None:
    _rate(page, "Collision Rate ν").setValue(2.0)
    _choose(page, "Geometry", geometry)
    _choose(page, "Sample", polycrystal)
    if detector is not None:
        _check(page, "Along the spin").setChecked(False)
        _vector(page, "Detector").set_value(detector)

    assert _rate(page, "Collision Rate ν").isEnabled() is allowed
    assert _combo(page, "Dynamics Method").isEnabled() is allowed
    assert page.settings().collision_rate_per_us == (2.0 if allowed else 0.0)
    if not allowed:
        assert "polycrystal" in _rate(page, "Collision Rate ν").toolTip()


def test_an_applied_field_can_be_added_only_to_zero_field_results(page) -> None:
    assert _vector(page, "Add Applied Field (T)").isEnabled()
    assert "can be added" in find_control(page, "Applied field", QLabel).text()

    _set_checked(page, {"ws-lf"})
    assert not _vector(page, "Add Applied Field (T)").isEnabled()
    assert (
        find_control(page, "Applied field", QLabel).text()
        == "From the field results: (0, 0, 0.05) T."
    )
    assert page.settings().added_applied_field_T is None

    _set_checked(page, {MAGNETIC_FIELD_PAGE_OWNER, "ws-lf"})
    assert "different applied fields" in find_control(page, "Applied field", QLabel).text()
    _set_checked(page, set())
    assert find_control(page, "Applied field", QLabel).text() == "Select field results."


def test_adding_a_simulation_stores_shows_and_plots_it(page, document) -> None:
    changed: list[bool] = []
    page.simulations_changed.connect(lambda: changed.append(True))
    _set_checked(page, {MAGNETIC_FIELD_PAGE_OWNER, "ws-uep"})
    _edit(page, "Name").setText("ZF pair")
    _rate(page, "λz").setValue(0.1)

    _button(page, "Add Simulation").click()

    (record,) = polarisation_simulations(document)
    assert record.title == "ZF pair"
    assert record.settings.lambda_z_per_us == pytest.approx(0.1)
    assert [source.owner for source in record.sources] == [MAGNETIC_FIELD_PAGE_OWNER, "ws-uep"]
    assert changed == [True]
    table = _table(page, "Simulations")
    assert [table.item(0, column).text() for column in range(5)] == [
        "ZF pair",
        "Zero field",
        "Single crystal",
        "Magnetic Fields, UEP site",
        "Current, estimate",
    ]
    assert page.selected_simulation_id() == record.simulation_id
    assert _note(page, UNRELAXED_RESULTS_NOTE)
    time_axis, spectrum_axis = _figure(page).axes[:2]
    # Two site traces and their average.
    assert len(time_axis.lines) == 3
    np.testing.assert_allclose(
        time_axis.lines[-1].get_ydata(), record.simulation.static.polarisation
    )
    series = plot_series_colours()
    assert [line.get_color() for line in time_axis.lines] == [
        series.categorical[0],
        series.categorical[1],
        series.total,
    ]
    assert spectrum_axis.lines
    assert _button(page, "Recalculate").isEnabled() and _button(page, "Export Data…").isEnabled()


def test_a_failed_simulation_reports_why_and_stores_nothing(page, document) -> None:
    _choose(page, "Geometry", "longitudinal")
    _vector(page, "Add Applied Field (T)").set_value((0.0, 0.0, 0.0))

    _button(page, "Add Simulation").click()

    failure = _note(page, "Simulation failed:")
    assert failure is not None and "applied field" in failure
    assert polarisation_simulations(document) == []


def test_invalid_settings_are_reported_before_running(page, document) -> None:
    _vector(page, "Initial Polarisation").set_value((0.0, 0.0, 0.0))

    _button(page, "Add Simulation").click()

    assert _note(page, "Invalid settings:")
    assert polarisation_simulations(document) == []


def test_selecting_a_simulation_loads_its_settings_and_sources(page) -> None:
    _edit(page, "Name").setText("first")
    _button(page, "Add Simulation").click()
    first = page.selected_simulation_id()
    _set_checked(page, {"ws-lf"})
    _choose(page, "Geometry", "transverse")
    _edit(page, "Name").setText("second")
    _button(page, "Add Simulation").click()

    _select(page, "first")

    assert page.selected_simulation_id() == first
    assert _edit(page, "Name").text() == "first"
    assert _combo(page, "Geometry").currentData() == "zero_field"
    assert _checked_owners(page) == [MAGNETIC_FIELD_PAGE_OWNER]


def test_recalculate_replaces_the_selected_simulation(page, document) -> None:
    _choose(page, "Sample", True)
    _button(page, "Add Simulation").click()
    simulation_id = page.selected_simulation_id()

    _rate(page, "Collision Rate ν").setValue(2.0)
    _button(page, "Recalculate").click()

    (record,) = polarisation_simulations(document)
    assert record.simulation_id == simulation_id
    assert record.simulation.dynamic is not None
    assert record.simulation.dynamic.collision_rate_per_us == pytest.approx(2.0)
    assert _figure(page).axes[0].lines[-1].get_label() == "dynamic, ν = 2 μs⁻¹"


def test_recalculated_fields_mark_a_simulation_stale(page, document) -> None:
    _button(page, "Add Simulation").click()

    _store_fields(document, MAGNETIC_FIELD_PAGE_OWNER, field_result((0.0, 0.0, 0.02)))
    page.refresh()

    assert _table(page, "Simulations").item(0, 4).text() == "Stale"
    stale = _note(page, "Stale:")
    assert stale is not None and "Magnetic Fields" in stale
    _button(page, "Recalculate").click()
    assert _table(page, "Simulations").item(0, 4).text() == "Current"
    assert _note(page, "Stale:") is None


def test_remove_asks_first_and_binds_delete_and_backspace(
    page, document, monkeypatch, confirmed_destructive_actions
) -> None:
    _button(page, "Add Simulation").click()
    _button(page, "Add Simulation").click()
    shortcuts = _table(page, "Simulations").findChildren(QShortcut)
    keys = {shortcut.key() for shortcut in shortcuts}
    assert keys == {
        QKeySequence(QKeySequence.StandardKey.Delete),
        QKeySequence(Qt.Key.Key_Backspace),
    }

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)
    _button(page, "Delete…").click()
    assert len(polarisation_simulations(document)) == 2

    monkeypatch.undo()
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda *args, **kwargs: confirmed_destructive_actions.append({"text": args[2]}) or True,
    )
    shortcuts[0].activated.emit()
    assert len(polarisation_simulations(document)) == 1
    _select(page, polarisation_simulations(document)[0].title)
    shortcuts[1].activated.emit()
    assert polarisation_simulations(document) == []
    assert confirmed_destructive_actions[-1]["text"].startswith("Delete the simulation")
    assert not _button(page, "Delete…").isEnabled()


def test_form_state_round_trips_settings_sources_and_selection(page, document) -> None:
    _set_checked(page, {"ws-lf"})
    _choose(page, "Geometry", "longitudinal")
    _choose(page, "Sample", True)
    find_control(page, "Sphere points", QSpinBox).setValue(120)
    _edit(page, "Apodisation σ").setText("3")
    _edit(page, "Name").setText("LF powder")
    _button(page, "Add Simulation").click()
    state = page.collect_state()

    restored = PolarisationWidget()
    restored.set_document(document)
    restored.set_form_state(state)

    assert restored.selected_simulation_id() == page.selected_simulation_id()
    assert restored.settings() == page.settings()
    assert restored.settings().apodisation_sigma_us == pytest.approx(3.0)
    assert _checked_owners(restored) == ["ws-lf"]
    assert _edit(restored, "Name").text() == "LF powder"
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


def test_export_writes_the_traces_and_spectrum(page, document, tmp_path, monkeypatch) -> None:
    _set_checked(page, {MAGNETIC_FIELD_PAGE_OWNER, "ws-uep"})
    _choose(page, "Sample", True)
    _rate(page, "Collision Rate ν").setValue(0.5)
    _edit(page, "Name").setText("ZF pair")
    _button(page, "Add Simulation").click()
    target = tmp_path / "out.dat"
    monkeypatch.setattr(module, "ask_save_path", lambda *_args, **_kwargs: target)
    saved: list[Path] = []
    monkeypatch.setattr(module, "report_export_saved", lambda _parent, path: saved.append(path))

    _button(page, "Export Data…").click()

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
    record = polarisation_simulations(document)[0]
    assert text == export_text(record)


def test_an_unsaved_form_keeps_the_first_field_result_ticked(page) -> None:
    page.set_form_state({})

    assert _checked_owners(page) == [MAGNETIC_FIELD_PAGE_OWNER]


def test_greyed_out_directions_do_not_block_a_field_geometry(page, document) -> None:
    _set_checked(page, {"ws-lf"})
    _vector(page, "Initial Polarisation").set_value((0.0, 0.0, 0.0))
    _choose(page, "Geometry", "longitudinal")

    _button(page, "Add Simulation").click()

    assert len(polarisation_simulations(document)) == 1
    _choose(page, "Geometry", "zero_field")
    with pytest.raises(PolarisationError, match="polarisation_direction"):
        page.settings()


def test_an_invalid_box_keeps_the_last_valid_settings_in_the_form(page) -> None:
    _rate(page, "End Time").setValue(7.0)
    _edit(page, "Apodisation σ").setText("3")
    assert page.collect_state()["settings"]["apodisation_sigma_us"] == 3.0

    _edit(page, "Apodisation σ").setText("0")

    state = page.collect_state()
    assert state["settings"]["end_time_us"] == 7.0
    assert state["settings"]["apodisation_sigma_us"] == 3.0


def test_unnamed_simulations_take_the_first_unused_number(page, document) -> None:
    for _ in range(3):
        _edit(page, "Name").setText("")
        _button(page, "Add Simulation").click()
    first = next(r for r in polarisation_simulations(document) if r.title == "Simulation 1")
    _select(page, first.title)
    _button(page, "Delete…").click()

    _edit(page, "Name").setText("")
    _button(page, "Add Simulation").click()

    titles = sorted(record.title for record in polarisation_simulations(document))
    assert titles == ["Simulation 1", "Simulation 2", "Simulation 3"]


def test_export_columns_stay_distinct_when_labels_collide() -> None:
    assert module._unique_column_labels(["site 1", "site-1", "?", "site_1"]) == [
        "site_1",
        "site_1_2",
        "site3",
        "site_1_3",
    ]


def _count_decodes(monkeypatch) -> list[bool]:
    """Record each simulation whose traces are decoded."""

    from mufinder.services import polarisation_simulations as services

    decoded: list[bool] = []
    real = services.PolarisationSimulation.from_json
    monkeypatch.setattr(
        services.PolarisationSimulation,
        "from_json",
        staticmethod(lambda payload, settings: decoded.append(True) or real(payload, settings)),
    )
    return decoded


def _count_draws(page: PolarisationWidget, monkeypatch) -> list[bool]:
    """Record each time the plot is drawn afresh."""

    figure = _figure(page)
    drawn: list[bool] = []
    real = figure.clear
    monkeypatch.setattr(figure, "clear", lambda *a, **k: drawn.append(True) or real(*a, **k))
    return drawn


def test_the_table_lists_simulations_without_decoding_their_traces(page, monkeypatch) -> None:
    _button(page, "Add Simulation").click()
    _button(page, "Add Simulation").click()
    decoded = _count_decodes(monkeypatch)

    page.refresh()

    # The selected simulation was decoded when it was added, and nothing else is.
    assert decoded == []


def test_showing_the_page_twice_decodes_and_draws_the_record_once(
    qapp, page, document, monkeypatch
) -> None:
    _button(page, "Add Simulation").click()
    state = page.collect_state()
    shown = PolarisationWidget()
    decoded = _count_decodes(monkeypatch)
    drawn = _count_draws(shown, monkeypatch)

    # Opening the project shows its simulations with the saved form and its selection.
    shown.set_document(document, state)
    assert shown.selected_simulation_id() == state["selected_simulation"]
    assert decoded == [True] and drawn == [True]
    # Showing the page re-reads the project.
    shown.refresh()
    shown.refresh()

    assert decoded == [True] and drawn == [True]
    shown.close()


def test_a_changed_record_or_theme_is_decoded_and_drawn_again(page, document, monkeypatch) -> None:
    from mufinder.app import theme

    _choose(page, "Sample", True)
    _button(page, "Add Simulation").click()
    decoded = _count_decodes(monkeypatch)
    drawn = _count_draws(page, monkeypatch)

    _rate(page, "Collision Rate ν").setValue(2.0)
    _button(page, "Recalculate").click()

    assert decoded == [True] and drawn == [True]
    assert _figure(page).axes[0].lines[-1].get_label() == "dynamic, ν = 2 μs⁻¹"
    other = (
        theme.light_palette() if module.current_palette().mode == "dark" else theme.dark_palette()
    )
    monkeypatch.setattr(module, "current_palette", lambda: other)
    page.refresh()
    assert decoded == [True] and drawn == [True, True]


def test_cancel_leaves_the_project_alone_while_the_worker_runs(page, document, monkeypatch) -> None:
    before = dict(document.results)
    started: list = []

    def _cancelled(_parent, op, **_kwargs):
        started.append(op)
        raise BlockingCallCancelled("cancelled")

    monkeypatch.setattr(module, "run_blocking_call", _cancelled)
    _button(page, "Add Simulation").click()

    assert _note(page, "Simulation cancelled.")
    assert document.results == before
    # The abandoned worker carries on with the project gone from under it.
    (worker,) = started
    document.results.clear()
    document.structures.clear()
    record = worker()
    assert record.simulation.static.polarisation.size == 401
    assert document.results == {} and document.structures == {}


def test_a_greyed_out_collision_rate_is_kept_in_the_saved_form(page, document) -> None:
    _choose(page, "Sample", True)
    _rate(page, "Collision Rate ν").setValue(2.0)
    _choose(page, "Sample", False)
    assert page.settings().collision_rate_per_us == 0.0

    state = page.collect_state()

    assert state["settings"]["collision_rate_per_us"] == 2.0
    restored = PolarisationWidget()
    restored.set_document(document)
    restored.set_form_state(state)
    assert _rate(restored, "Collision Rate ν").value() == 2.0
    assert restored.settings().collision_rate_per_us == 0.0
    _choose(restored, "Sample", True)
    assert restored.settings().collision_rate_per_us == 2.0
    restored.close()


def test_each_vector_component_is_named_after_its_row(page) -> None:
    names = [edit.accessibleName() for edit in _vector(page, "Initial Polarisation").edits]

    assert names == ["Initial Polarisation x", "Initial Polarisation y", "Initial Polarisation z"]
