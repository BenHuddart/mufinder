"""Tests for the project home chart, field notes and cell survey widgets."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest

from mufinder.app import theme as theme_module
from mufinder.app.cell_survey_widget import FALLBACK_NOTE, CellSurveyWidget
from mufinder.app.project_home_trail import (
    FieldNotesPanel,
    ProjectHomeChartWidget,
    ProjectHomeWidget,
)
from mufinder.services.project_summary import WORKFLOW_STAGES, summarize_project
from mufinder.services.project_survey import (
    SurveyLayer,
    SurveyMarker,
    SurveyMarkerKind,
    survey_layers,
)
from tests.support.project_home import muon_project


def _chart(tmp_path: Path) -> ProjectHomeChartWidget:
    chart = ProjectHomeChartWidget()
    chart.resize(960, 360)
    chart.set_summary(summarize_project(muon_project(tmp_path).document))
    return chart


def _point(chart: ProjectHomeChartWidget, key: str) -> QPoint:
    return chart.node_center(key).toPoint()


def test_chart_places_every_stage() -> None:
    assert set(ProjectHomeChartWidget.NODE_POSITIONS) == {stage.key for stage in WORKFLOW_STAGES}


def test_chart_opens_on_the_current_stage_and_selects_on_click(qapp, tmp_path: Path) -> None:
    chart = _chart(tmp_path)
    selected: list[str] = []
    chart.stage_selected.connect(selected.append)

    assert chart.selected_stage() == "muon_relaxations"
    QTest.mouseClick(chart, Qt.MouseButton.LeftButton, pos=_point(chart, "uep"))

    assert chart.selected_stage() == "uep"
    assert selected == ["uep"]
    assert "UEP" in chart.accessibleName()


def test_chart_opens_a_stage_on_double_click_and_return(qapp, tmp_path: Path) -> None:
    chart = _chart(tmp_path)
    routes: list[str] = []
    chart.route_requested.connect(routes.append)

    QTest.mouseDClick(chart, Qt.MouseButton.LeftButton, pos=_point(chart, "energy_barriers"))
    chart.setFocus()
    QTest.keyClick(chart, Qt.Key.Key_Left)
    QTest.keyClick(chart, Qt.Key.Key_Return)

    assert routes == ["energy_barriers", "quantum_corrections"]


def test_chart_ignores_clicks_on_empty_map(qapp, tmp_path: Path) -> None:
    chart = _chart(tmp_path)

    QTest.mouseClick(chart, Qt.MouseButton.LeftButton, pos=QPoint(3, 3))

    assert chart.selected_stage() == "muon_relaxations"


def test_chart_renders_in_both_themes(qapp, tmp_path: Path) -> None:
    chart = _chart(tmp_path)
    try:
        for mode in (theme_module.THEME_LIGHT, theme_module.THEME_DARK):
            theme_module.apply_theme(qapp, mode)
            pixmap = QPixmap(chart.size())
            chart.render(pixmap)
            assert not pixmap.toImage().isNull()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def test_field_notes_show_progress_needs_and_open_the_stage(qapp, tmp_path: Path) -> None:
    summary = summarize_project(muon_project(tmp_path).document)
    panel = FieldNotesPanel()
    routes: list[str] = []
    panel.route_requested.connect(routes.append)

    panel.set_stage(summary.stage("quantum_corrections"))
    assert panel._title.text() == "Quantum Corrections"
    assert panel._needs.text() == "Needs: Analysis"
    assert not panel._needs.isHidden()

    panel.set_stage(summary.stage("muon_relaxations"))
    assert panel._status.text() == "In progress · 1 job active"
    assert panel._facts.text() == "Positive muon: 1 of 2 relaxed, 1 active"
    assert panel._needs.isHidden()
    panel._open_button.click()
    assert routes == ["muon_relaxations"]


def test_home_widget_follows_the_selected_stage_in_the_survey(qapp, tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    home = ProjectHomeWidget()
    home.set_document(project.document)
    home.refresh()

    assert home._progress.text() == "2 of 12 stages complete · 1 job active"
    assert home.survey.current_layer().title == "Positive muon · 1 of 2 relaxed"

    home.chart.select_stage("convergence_cutoff")
    assert home.field_notes.stage().key == "convergence_cutoff"
    assert home.survey.canvas.faded is True
    assert home.survey.note_label.text() == FALLBACK_NOTE

    routes: list[str] = []
    home.route_requested.connect(routes.append)
    home._journal_button.click()
    assert routes == ["provenance_log"]


def test_home_widget_defers_work_until_shown(qapp, tmp_path: Path) -> None:
    home = ProjectHomeWidget()

    home.set_document(muon_project(tmp_path).document)

    assert home.chart.summary() is None
    home.show()
    assert home.chart.summary() is not None
    home.close()


def _site_layer(title: str, position: tuple[float, float, float]) -> SurveyLayer:
    marker = SurveyMarker(position, SurveyMarkerKind.SITE, "A", f"{title} site")
    return SurveyLayer("muon_analysis", title, (marker,))


def test_survey_steps_through_result_sets_and_wraps(qapp, tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    survey = CellSurveyWidget()
    layers = (_site_layer("First", (0.5, 0.25, 0.25)), _site_layer("Second", (0.25, 0.5, 0.75)))

    survey.set_survey(project.host, layers, None)
    assert survey.layer_label.text() == "First (1 of 2)"
    assert not survey.next_button.isHidden()

    survey.next_button.click()
    assert survey.layer_label.text() == "Second (2 of 2)"
    survey.next_button.click()
    assert survey.current_index() == 0
    survey.previous_button.click()
    assert survey.current_index() == 1

    survey.set_survey(project.host, layers, None)
    assert survey.current_index() == 1

    survey.set_survey(project.host, layers[:1], None)
    assert survey.next_button.isHidden()
    assert survey.layer_label.text() == "First"


def test_survey_projection_toggle_and_marker_tooltips(qapp, tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    survey = CellSurveyWidget()
    survey.resize(360, 320)
    survey.set_survey(project.host, survey_layers(project.document, "muon_relaxations"), None)

    survey.axis_buttons["a"].click()
    assert survey.axis() == "a"
    assert survey.axis_buttons["a"].isChecked()

    pixmap = QPixmap(survey.size())
    survey.render(pixmap)
    details = [text for _center, _radius, text in survey.canvas._hits]
    assert "mu1 · +0.000 eV" in details
    assert "Sr" in details
    center = next(
        center for center, _radius, text in survey.canvas._hits if text == "mu1 · +0.000 eV"
    )
    assert survey.canvas.hit_text_at(center) == "mu1 · +0.000 eV"


def test_survey_without_host_or_positions_says_so(qapp, tmp_path: Path) -> None:
    survey = CellSurveyWidget()

    survey.set_survey(None, (), None)
    assert survey.note_label.isHidden()

    survey.set_survey(muon_project(tmp_path).host, (), None)
    assert survey.note_label.text() == "No muon positions found yet."
    pixmap = QPixmap(320, 280)
    survey.render(pixmap)
    assert not pixmap.toImage().isNull()


def test_short_window_scrolls_instead_of_squeezing_the_page(qapp, tmp_path: Path) -> None:
    home = ProjectHomeWidget()
    home.set_document(muon_project(tmp_path).document)
    home.resize(1000, 360)
    home.show()
    qapp.processEvents()

    assert home.chart.height() >= home.chart.minimumHeight()
    assert home.survey.canvas.height() >= home.survey.canvas.minimumHeight()
    assert home._scroll.verticalScrollBar().maximum() > 0
    home.close()


def test_chart_labels_stay_inside_the_chart_at_its_minimum_width(qapp, tmp_path: Path) -> None:
    chart = _chart(tmp_path)
    chart.resize(chart.minimumSize())

    for key in ProjectHomeChartWidget.NODE_POSITIONS:
        rect = chart._label_rect(key, chart.node_center(key))
        assert rect.left() >= 0.0
        assert rect.right() <= chart.width()
