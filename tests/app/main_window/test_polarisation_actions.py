"""MainWindow wiring of the Polarisation page."""

from __future__ import annotations

from pathlib import Path

from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    store_magnetic_field_results,
)
from mufinder.services.polarisation_simulations import polarisation_simulations
from mufinder.services.project_summary import COMPLETION_CHECKS
from tests.support.field_results import field_result
from tests.support.main_window import select_route, window_with_host


def _field_form(*totals: tuple[float, float, float]) -> dict[str, object]:
    results = [
        field_result(total, label=f"site {index}", position=(0.25, 0.25, 0.25)).to_json()
        for index, total in enumerate(totals, start=1)
    ]
    return {"magnetic_field_results": {"kind": "commensurate", "results": results}}


def _window_with_fields(tmp_path: Path):
    window, service, document, host = window_with_host(tmp_path)
    document.ui_state["magnetic_field_form"] = store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, _field_form((0.0, 0.0, 0.01))
    )
    document.ui_state["magnetic_field_workspaces"] = [
        {
            "id": "ws-1",
            "title": "Second site",
            "form": store_magnetic_field_results(document, "ws-1", _field_form((0.02, 0.0, 0.0))),
        }
    ]
    window._refresh_ui_after_project_change()
    return window, service, document


def test_polarisation_route_shows_the_page_with_the_projects_field_results(
    qapp, tmp_path: Path
) -> None:
    window, _service, _document = _window_with_fields(tmp_path)

    select_route(window, "polarisation")

    assert window._main_stack.currentWidget() is window._polarisation_view
    assert window._polarisation_view._sources_table.rowCount() == 2
    window.close()


def test_polarisation_buttons_open_the_page_with_their_fields_selected(
    qapp, tmp_path: Path
) -> None:
    window, _service, _document = _window_with_fields(tmp_path)
    page_button = window._magnetic_field_view._polarisation_button
    assert page_button.isEnabled()

    window._magnetic_field_workspace_widgets["ws-1"]._polarisation_button.click()

    assert window._main_stack.currentWidget() is window._polarisation_view
    assert [s.owner for s in window._polarisation_view._checked_sources()] == ["ws-1"]

    page_button.click()
    assert [s.owner for s in window._polarisation_view._checked_sources()] == [
        MAGNETIC_FIELD_PAGE_OWNER
    ]
    window.close()


def test_a_simulation_is_saved_marks_the_stage_and_its_form_reloads(qapp, tmp_path: Path) -> None:
    from mufinder.app.main_window import MainWindow

    window, service, document = _window_with_fields(tmp_path)
    select_route(window, "polarisation")
    page = window._polarisation_view
    page._time_points.setValue(201)
    page._title_edit.setText("ZF")

    page._add_button.click()

    assert COMPLETION_CHECKS["polarisation"](document)
    reloaded = service.load_project(document.project_root)
    (record,) = polarisation_simulations(reloaded)
    assert record.title == "ZF"
    assert reloaded.ui_state["polarisation_form"]["selected_simulation"] == record.simulation_id

    reopened = MainWindow()
    reopened._document = reloaded
    reopened._refresh_ui_after_project_change()
    assert reopened._polarisation_view.selected_simulation_id() == record.simulation_id
    assert reopened._polarisation_view.settings().time_points == 201
    window.close()
    reopened.close()
