"""Tests for the project home page and its cell survey."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from mufinder.app import atomic_structure_viewer as viewer_module
from mufinder.app import cell_survey_widget as cell_survey_module
from mufinder.app import theme as theme_module
from mufinder.app.atomic_structure_viewer import AtomicStructureViewer, MUON_VIEWER_RADIUS_ANG
from mufinder.app.cell_survey_widget import (
    HIGHLIGHT_RADIUS_ANG,
    NO_HOST_TEXT,
    VIEW_MINIMUM_HEIGHT,
    VIEWER_UNAVAILABLE_TEXT,
    CellSurveyWidget,
)
from mufinder.app.main_window import MainWindow
from mufinder.app.project_home import ProjectHomeWidget
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import MuonSite, ProvenanceRecord, Result, utc_now
from mufinder.services.journal import short_day_label
from mufinder.services.project_service import ProjectService
from mufinder.services.project_survey import discovered_sites_layer
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE
from tests.support.project_home import muon_project, pin_muon_sites
from tests.support.pyvista_fakes import count_created_interactors, count_scene_renders
from tests.support.qt import pump_until

_POSITIVE_SITES = [
    {
        "cluster_index": 2,
        "fractional_position": [0.5, 0.25, 0.25],
        "wyckoff_label": "3c",
        "relative_energy_ev": 0.0,
    },
    {"cluster_index": 1, "fractional_position": [0.25, 0.5, 0.75], "relative_energy_ev": 0.184},
]
_MUONIUM_SITES = [
    {"cluster_index": 3, "fractional_position": [0.1, 0.2, 0.3], "relative_energy_ev": 0.0},
]


def _pinned_project(tmp_path: Path):
    project = muon_project(tmp_path)
    pin_muon_sites(project.document, project.host.id, "positive_muon", _POSITIVE_SITES)
    pin_muon_sites(project.document, project.host.id, "muonium", _MUONIUM_SITES)
    return project


def _home(document) -> ProjectHomeWidget:
    home = ProjectHomeWidget()
    home.resize(1100, 760)
    home.set_document(document)
    home.refresh()
    return home


def _cells(home: ProjectHomeWidget) -> list[list[str]]:
    table = home.sites_table
    return [
        [table.item(row, column).text() for column in range(table.columnCount())]
        for row in range(table.rowCount())
    ]


def _shown_home(qapp, document) -> ProjectHomeWidget:
    """A shown home page whose survey has created its viewer (the deferred step, run now)."""

    home = _home(document)
    home.show()
    qapp.processEvents()
    home.survey.create_viewer()
    return home


def _markers(survey: CellSurveyWidget) -> list[tuple[tuple[float, ...], str, float]]:
    """Each point overlay the survey's viewer draws: its position, colour and radius."""

    markers = []
    for spec in survey.viewer._point_overlay_specs:
        assert len(spec.positions) == 1
        colour = "#{:02X}{:02X}{:02X}".format(*(round(value * 255) for value in spec.colors[0]))
        markers.append((tuple(spec.positions[0]), colour, float(spec.radii[0])))
    return markers


def _positions(survey: CellSurveyWidget) -> list[tuple[float, ...]]:
    return [position for position, _colour, _radius in _markers(survey)]


def test_header_and_left_off_describe_an_empty_project(qapp, tmp_path: Path) -> None:
    document = ProjectService().create_project(tmp_path / "empty.mufinder", "Empty")
    home = _home(document)

    assert home.name_label.text() == "Empty"
    assert home.host_label.text() == "No host structure yet"
    assert home.stage_title.text() == "Host Structure"
    assert home.stage_date.text() == ""
    assert home.stage_facts.text() == "No host structure yet"
    assert home.open_stage_button.text() == "Open Host Structure"
    assert home.sites_table.isHidden()
    assert home.sites_progress.text() == "No candidate muon sites yet."
    assert home.other_results_card.isHidden()
    assert home.activity_card.isHidden()


def test_pinned_sites_fill_the_table_in_both_charge_states(qapp, tmp_path: Path) -> None:
    home = _home(_pinned_project(tmp_path).document)

    assert [home.sites_table.horizontalHeaderItem(c).text() for c in range(4)] == [
        "Site",
        "Charge state",
        "Wyckoff",
        "dE (eV)",
    ]
    assert _cells(home) == [
        ["Site 1", "Positive muon", "3c", "0.000"],
        ["Site 2", "Positive muon", "—", "0.184"],
        ["Site 1", "Muonium", "—", "0.000"],
    ]
    assert not home.sites_table.isHidden()
    assert home.sites_progress.isHidden()
    assert home.survey.current_layer().title == "Pinned muon sites"
    assert len(home.survey.current_layer().markers) == 3


def test_with_nothing_pinned_a_progress_sentence_replaces_the_table(qapp, tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    home = _home(project.document)

    assert home.sites_table.isHidden()
    assert home.sites_progress.text() == (
        "2 candidate sites, 1 of 2 relaxed. No sites pinned to the Muon Sites summary yet."
    )
    assert home.survey.current_layer() == discovered_sites_layer(project.document)
    assert home.survey.current_layer().title == "Positive muon · 1 of 2 relaxed"


def test_selecting_a_site_row_highlights_its_marker(qapp, tmp_path: Path) -> None:
    home = _shown_home(qapp, _pinned_project(tmp_path).document)
    muon = theme_module.STRUCTURE_OVERLAY_COLOURS.muon
    halo = theme_module.STRUCTURE_OVERLAY_COLOURS.highlight

    rect = home.sites_table.visualRect(home.sites_table.model().index(2, 0))
    QTest.mouseClick(home.sites_table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert home.survey.highlighted_marker() == 2
    assert [(colour, radius) for _position, colour, radius in _markers(home.survey)] == [
        (muon, MUON_VIEWER_RADIUS_ANG),
        (muon, MUON_VIEWER_RADIUS_ANG),
        (muon, MUON_VIEWER_RADIUS_ANG),
        (halo, HIGHLIGHT_RADIUS_ANG),
    ]
    # The halo sits on the selected site, whose own sphere keeps its size.
    assert _positions(home.survey)[3] == _positions(home.survey)[2]

    home.sites_table.setFocus()
    QTest.keyClick(home.sites_table, Qt.Key.Key_Up)
    assert home.survey.highlighted_marker() == 1
    assert [radius for _position, _colour, radius in _markers(home.survey)] == [
        MUON_VIEWER_RADIUS_ANG,
        MUON_VIEWER_RADIUS_ANG,
        MUON_VIEWER_RADIUS_ANG,
        HIGHLIGHT_RADIUS_ANG,
    ]
    assert _positions(home.survey)[3] == _positions(home.survey)[1]

    home.sites_table.clearSelection()
    assert home.survey.highlighted_marker() is None
    assert {(colour, radius) for _position, colour, radius in _markers(home.survey)} == {
        (muon, MUON_VIEWER_RADIUS_ANG)
    }
    home.close()


def test_survey_highlight_ignores_a_marker_the_layer_lacks(qapp, tmp_path: Path) -> None:
    home = _shown_home(qapp, muon_project(tmp_path).document)
    survey = home.survey
    layer = survey.current_layer()

    survey.highlight_marker(0)
    assert survey.highlighted_marker() == 0
    survey.highlight_marker(len(layer.markers))
    assert survey.highlighted_marker() is None
    assert HIGHLIGHT_RADIUS_ANG not in [radius for _p, _c, radius in _markers(survey)]
    home.close()


def test_a_refresh_keeps_the_selected_site_not_its_row(qapp, tmp_path: Path) -> None:
    project = _pinned_project(tmp_path)
    home = _home(project.document)
    home.show()
    qapp.processEvents()
    home.sites_table.selectRow(2)
    assert _cells(home)[2][1] == "Muonium"

    pin_muon_sites(project.document, project.host.id, "positive_muon", _POSITIVE_SITES[:1])
    home.refresh()

    assert home.sites_table.selectionModel().selectedRows()[0].row() == 1
    assert _cells(home)[1][1] == "Muonium"
    assert home.survey.highlighted_marker() == 1

    pin_muon_sites(project.document, project.host.id, "muonium", [])
    home.refresh()

    assert home.sites_table.selectionModel().selectedRows() == []
    assert home.survey.highlighted_marker() is None
    home.close()


def test_return_and_double_click_on_a_site_open_the_muon_sites_summary(
    qapp, tmp_path: Path
) -> None:
    home = _home(_pinned_project(tmp_path).document)
    home.show()
    qapp.processEvents()
    routes: list[str] = []
    home.route_requested.connect(routes.append)

    home.sites_table.setFocus()
    home.sites_table.selectRow(0)
    QTest.keyClick(home.sites_table, Qt.Key.Key_Return)
    QTest.keyClick(home.sites_table, Qt.Key.Key_Enter)
    rect = home.sites_table.visualRect(home.sites_table.model().index(1, 1))
    QTest.mouseClick(home.sites_table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    QTest.mouseDClick(home.sites_table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())

    assert routes == ["muon_sites", "muon_sites", "muon_sites"]
    home.close()


def test_open_buttons_ask_for_the_stage_left_off_at_and_the_journal(qapp, tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    later = utc_now() + timedelta(hours=1)
    project.document.provenance.append(
        ProvenanceRecord(
            timestamp=later,
            action="set_recommended_cutoff",
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            parameters={"recommended_cutoff_energy": 600.0},
        )
    )
    home = _home(project.document)
    routes: list[str] = []
    home.route_requested.connect(routes.append)

    assert home.stage_title.text() == "Cutoff Energy"
    assert home.stage_date.text() == short_day_label(later)
    assert home.open_stage_button.text() == "Open Cutoff Energy"
    home.open_stage_button.click()
    home.journal_button.click()

    assert routes == ["convergence_cutoff", "provenance_log"]
    (date, title) = home.activity_rows[0]
    assert (date.text(), title.text()) == (short_day_label(later), "Recommended cutoff adopted")
    assert home.activity_rows[1][0].isHidden()
    assert not home.activity_card.isHidden()


def test_other_results_line_shows_the_later_workflows(qapp, tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    phonon = Result(result_type=QUANTUM_PHONON_CALC_TYPE, data={})
    project.document.results[phonon.id] = phonon

    home = _home(project.document)

    assert not home.other_results_card.isHidden()
    assert home.other_results_label.text() == "Quantum Corrections: 1 phonon result"


def test_switching_projects_resets_every_section(qapp, tmp_path: Path) -> None:
    first = _pinned_project(tmp_path / "first")
    home = _home(first.document)
    home.show()
    home.sites_table.selectRow(1)
    home.resize(1100, 300)
    qapp.processEvents()
    home._scroll.verticalScrollBar().setValue(home._scroll.verticalScrollBar().maximum())
    assert home.survey.highlighted_marker() == 1

    # The same document refreshed keeps the selection.
    home.refresh()
    assert home.survey.highlighted_marker() == 1

    second = _pinned_project(tmp_path / "second")
    second.document.project.name = "Second"
    home.set_document(second.document)
    qapp.processEvents()

    assert home.name_label.text() == "Second"
    assert home.sites_table.selectionModel().selectedRows() == []
    assert home.survey.highlighted_marker() is None
    assert home._scroll.verticalScrollBar().value() == 0

    empty = ProjectService().create_project(tmp_path / "empty.mufinder", "Empty")
    home.set_document(empty)
    qapp.processEvents()
    assert home.sites_table.rowCount() == 0
    assert home.stage_title.text() == "Host Structure"
    assert home.survey.current_layer() is None
    home.close()


def test_a_project_with_the_retired_home_stage_key_opens_with_its_home_filled(
    main_window: MainWindow, tmp_path: Path
) -> None:
    project = muon_project(tmp_path)
    site = MuonSite(
        label="mu3",
        initial_fractional_position=[0.25, 0.25, 0.25],
        source_structure_id=project.host.id,
    )
    project.document.sites[site.id] = site
    project.document.ui_state["project_home_stage"] = "uep"
    ProjectService().save_project(project.document)

    main_window.show()
    assert main_window.open_project(project.document.project_root)
    QApplication.processEvents()

    home = main_window._home_page.project_home
    assert (
        main_window._home_page.content_stack.currentWidget() is main_window._home_page.project_page
    )
    assert main_window._document.sites[site.id].generation_method is None
    assert home.stage_title.text() == "Relaxations"
    assert home.sites_progress.text().startswith("3 candidate sites, 1 of 2 relaxed.")
    # The survey's viewer comes after the page's first paint, inside the main window,
    # whose close shuts it down with every other structure viewer.
    assert pump_until(lambda: home.survey.viewer is not None)
    assert main_window.isAncestorOf(home.survey.viewer)
    main_window.close()


def test_home_widget_defers_work_until_shown(qapp, tmp_path: Path) -> None:
    home = ProjectHomeWidget()

    home.set_document(muon_project(tmp_path).document)

    assert home.summary() is None
    home.show()
    assert home.summary() is not None
    home.close()


def test_home_renders_in_both_themes(qapp, tmp_path: Path) -> None:
    home = _shown_home(qapp, _pinned_project(tmp_path).document)
    home.sites_table.selectRow(0)
    try:
        for mode in (theme_module.THEME_LIGHT, theme_module.THEME_DARK):
            theme_module.apply_theme(qapp, mode)
            qapp.processEvents()
            pixmap = QPixmap(home.size())
            home.render(pixmap)
            assert not pixmap.toImage().isNull()
            # The halo is one fixed colour in both themes.
            assert _markers(home.survey)[-1][1] == theme_module.STRUCTURE_OVERLAY_COLOURS.highlight
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
        home.close()


def test_pinned_sites_reach_the_viewer_at_their_cartesian_positions(qapp, tmp_path: Path) -> None:
    home = _shown_home(qapp, _pinned_project(tmp_path).document)

    # The 4 Å cubic host: the table's order, positive muon sites then muonium.
    assert _positions(home.survey) == [
        pytest.approx((2.0, 1.0, 1.0)),
        pytest.approx((1.0, 2.0, 3.0)),
        pytest.approx((0.4, 0.8, 1.2)),
    ]
    assert {colour for _position, colour, _radius in _markers(home.survey)} == {
        theme_module.STRUCTURE_OVERLAY_COLOURS.muon
    }
    assert home.survey.viewer.has_structure()
    assert home.survey.layer_label.text() == "Pinned muon sites"
    home.close()


def test_with_nothing_pinned_the_discovered_positions_reach_the_viewer(
    qapp, tmp_path: Path
) -> None:
    project = muon_project(tmp_path)
    home = _shown_home(qapp, project.document)

    layer = discovered_sites_layer(project.document)
    assert _positions(home.survey) == [
        pytest.approx(tuple(4.0 * value for value in marker.position)) for marker in layer.markers
    ]
    # mu1 relaxed to (0.5, 0.25, 0.25); mu2 is still at its candidate position.
    assert _positions(home.survey) == [
        pytest.approx((2.0, 1.0, 1.0)),
        pytest.approx((0.8, 2.0, 3.0)),
    ]
    assert home.survey.layer_label.text() == "Positive muon · 1 of 2 relaxed"
    home.close()


def test_without_a_host_the_survey_says_so_and_creates_no_viewer(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = count_created_interactors(monkeypatch)
    document = ProjectService().create_project(tmp_path / "empty.mufinder", "Empty")
    home = _home(document)
    home.show()
    for _ in range(3):
        qapp.processEvents()

    assert home.survey.create_viewer() is None
    assert home.survey.viewer is None
    assert home.findChildren(AtomicStructureViewer) == []
    assert created == []
    assert home.survey.placeholder_label.text() == NO_HOST_TEXT
    assert not home.survey.placeholder_label.isHidden()
    assert home.survey.layer_label.isHidden()
    assert home.survey.note_label.isHidden()
    home.close()


def test_the_viewer_is_created_after_the_page_first_paints(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = count_created_interactors(monkeypatch)
    home = _home(_pinned_project(tmp_path).document)
    home.show()

    # Shown but not yet painted: no viewer and no render window.
    assert home.survey.viewer is None
    assert created == []

    viewer = home.survey.create_viewer()

    assert viewer is home.survey.viewer
    assert home.survey.isAncestorOf(viewer)
    assert len(created) == 2  # the scene and the axes triad
    assert viewer._interaction_hint.isHidden()
    assert viewer._selection_label.isHidden()
    assert viewer.accessibleName() == "Cell Survey"
    assert home.survey.placeholder_label.isHidden()
    # A second call keeps the viewer it has.
    assert home.survey.create_viewer() is viewer
    assert len(created) == 2
    home.close()


def test_the_first_paint_schedules_the_viewer(qapp, tmp_path: Path) -> None:
    home = _home(_pinned_project(tmp_path).document)
    home.show()

    assert pump_until(lambda: home.survey.viewer is not None)
    assert len(_markers(home.survey)) == 3
    home.close()


def test_a_hidden_survey_creates_no_viewer(qapp, tmp_path: Path) -> None:
    home = _home(_pinned_project(tmp_path).document)

    assert home.survey.create_viewer() is None
    assert home.survey.viewer is None
    home.deleteLater()


def test_a_refresh_of_the_same_project_keeps_the_viewer_and_another_reloads_it(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renders = count_scene_renders(monkeypatch)
    first = _pinned_project(tmp_path / "first")
    home = _shown_home(qapp, first.document)
    viewer = home.survey.viewer
    loaded: list[object] = []
    original = viewer.set_structure
    monkeypatch.setattr(
        viewer, "set_structure", lambda structure: (loaded.append(structure), original(structure))
    )
    home.sites_table.selectRow(1)
    qapp.processEvents()
    rebuilds = renders[viewer]

    # The dashboard flush re-sends the same project.
    home.set_document(first.document)
    home.refresh()
    qapp.processEvents()

    assert loaded == []
    assert renders[viewer] == rebuilds
    assert home.survey.highlighted_marker() == 1

    # Spins edited in place on the same host are redrawn with the view kept.
    viewer._set_named_view("a")
    camera = viewer._camera_state
    spins = [0.0] * len(first.host.species)
    first.host.properties["atomic_spins"] = spins
    home.refresh()
    spins[1] = 2.0
    home.refresh()
    assert list(viewer._spins) == spins
    assert viewer._camera_state == camera
    # Three sites and the halo around the one still selected.
    assert len(_markers(home.survey)) == 4
    assert home.survey.highlighted_marker() == 1
    loaded.clear()
    home.refresh()
    assert loaded == []

    # A newly pinned site on the same host changes only the markers.
    pin_muon_sites(first.document, first.host.id, "muonium", [])
    home.refresh()
    assert loaded == []
    # Two sites left, and the halo around the one still selected.
    assert len(_markers(home.survey)) == 3

    second = _pinned_project(tmp_path / "second")
    home.set_document(second.document)
    qapp.processEvents()

    assert loaded == [second.host]
    assert home.survey.viewer is viewer
    assert home.survey.highlighted_marker() is None
    assert HIGHLIGHT_RADIUS_ANG not in [radius for _p, _c, radius in _markers(home.survey)]

    empty = ProjectService().create_project(tmp_path / "empty.mufinder", "Empty")
    home.set_document(empty)
    qapp.processEvents()
    assert not viewer.isVisible()
    assert home.survey.placeholder_label.text() == NO_HOST_TEXT

    home.set_document(second.document)
    qapp.processEvents()
    assert not viewer.isHidden()
    assert home.survey.placeholder_label.isHidden()
    home.close()


def _no_pyvista_backend(*_args, **_kwargs) -> AtomicStructureViewer:
    raise RuntimeError("AtomicStructureViewer requires PyVista and pyvistaqt.")


def _no_opengl(_parent, **_kwargs):
    raise RuntimeError("Could not create an OpenGL context")


def test_the_card_keeps_its_height_when_the_viewer_arrives(qapp, tmp_path: Path) -> None:
    home = _home(_pinned_project(tmp_path).document)
    home.show()
    qapp.processEvents()
    height = home.survey.height()

    viewer = home.survey.create_viewer()
    # A viewer that would like more height than the page leaves the card.
    viewer.setMinimumHeight(home.survey.view_area.height() // 2)
    assert viewer.sizeHint().height() > 0
    qapp.processEvents()

    assert home.survey.view_area.sizeHint().height() == VIEW_MINIMUM_HEIGHT
    assert home.survey.height() == height
    assert home.survey.viewer.parentWidget().height() == home.survey.view_area.height()
    home.close()


@pytest.mark.parametrize("failure", ["construction", "first show", "unreported first show"])
def test_a_viewer_that_cannot_be_created_leaves_the_page_working(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog, failure: str
) -> None:
    import pyvista as pv

    count_created_interactors(monkeypatch)
    if failure == "construction":
        monkeypatch.setattr(cell_survey_module, "AtomicStructureViewer", _no_pyvista_backend)
    elif failure == "first show":
        monkeypatch.setattr(viewer_module, "_pyvista_modules", lambda: (pv, _no_opengl))
    else:
        # The show event failed without the error reaching show()'s caller.
        monkeypatch.setattr(AtomicStructureViewer, "_ensure_backend", lambda _viewer: None)
    home = _home(_pinned_project(tmp_path).document)
    home.show()

    with caplog.at_level("WARNING", logger="mufinder.app.cell_survey_widget"):
        assert home.survey.create_viewer() is None
    qapp.processEvents()

    assert "3D view could not be created" in caplog.text
    assert home.survey.viewer is None
    assert home.findChildren(AtomicStructureViewer) == []
    assert home.survey.placeholder_label.text() == VIEWER_UNAVAILABLE_TEXT
    assert not home.survey.placeholder_label.isHidden()
    assert home.survey.layer_label.text() == "Pinned muon sites"
    # No second attempt, and the rest of the page still works.
    assert home.survey.create_viewer() is None
    home.sites_table.selectRow(2)
    assert home.survey.highlighted_marker() == 2
    home.refresh()
    assert home.survey.placeholder_label.text() == VIEWER_UNAVAILABLE_TEXT
    pixmap = QPixmap(home.size())
    home.render(pixmap)
    assert not pixmap.toImage().isNull()
    home.close()


def test_project_home_scroll_area_is_not_a_tab_stop(qapp) -> None:
    home = ProjectHomeWidget()

    assert not home._scroll.focusPolicy() & Qt.FocusPolicy.TabFocus

    home.deleteLater()


def test_a_host_without_atoms_is_treated_as_no_host(qapp, tmp_path: Path) -> None:
    project = muon_project(tmp_path)
    project.host.species = []
    project.host.fractional_positions = []
    home = _home(project.document)
    home.show()
    qapp.processEvents()

    assert home.survey.create_viewer() is None
    assert home.survey.placeholder_label.text() == NO_HOST_TEXT
    assert home.survey.layer_label.isHidden()
    home.close()


def test_a_refresh_compares_spins_held_as_an_array(qapp, tmp_path: Path) -> None:
    import numpy as np

    project = muon_project(tmp_path)
    count = len(project.host.species)
    project.host.properties["atomic_spins"] = np.ones(count)
    home = _shown_home(qapp, project.document)
    viewer = home.survey.viewer

    project.host.properties["atomic_spins"] = np.ones(count)
    home.refresh()
    project.host.properties["atomic_spins"] = np.full(count, 2.0)
    home.refresh()

    assert list(viewer._spins) == [2.0] * count
    home.close()
