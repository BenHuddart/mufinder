"""MainWindow wiring of the Polarisation page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QSpinBox, QTableWidget

from mufinder.app.magnetic_field_widget import MagneticFieldWidget
from mufinder.app.polarisation_widget import PolarisationWidget
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    store_magnetic_field_results,
)
from mufinder.services.polarisation_simulations import polarisation_simulations
from mufinder.services.project_summary import COMPLETION_CHECKS
from tests.support.field_results import field_result
from tests.support.main_window import select_route, window_with_host
from tests.support.qt import find_control


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


def _polarisation_page(window) -> PolarisationWidget:
    (page,) = window.findChildren(PolarisationWidget)
    return page


def _polarisation_button(window, title: str) -> QPushButton:
    """The ``Polarisation →`` button of the Magnetic Fields page or workspace called ``title``."""

    (page,) = (
        page
        for page in window.findChildren(MagneticFieldWidget)
        if any(label.text() == title for label in page.findChildren(QLabel))
    )
    return find_control(page, "Polarisation →", QPushButton)


def _checked_owners(page: PolarisationWidget) -> list[str]:
    return [source["owner"] for source in page.collect_state()["sources"]]


def test_polarisation_route_shows_the_page_with_the_projects_field_results(
    qapp, tmp_path: Path
) -> None:
    window, _service, _document = _window_with_fields(tmp_path)

    select_route(window, "polarisation")

    page = _polarisation_page(window)
    assert page.isVisibleTo(window)
    assert find_control(page, "Field Results", QTableWidget).rowCount() == 2
    window.close()


def test_polarisation_buttons_open_the_page_with_their_fields_selected(
    qapp, tmp_path: Path
) -> None:
    window, _service, _document = _window_with_fields(tmp_path)
    page = _polarisation_page(window)
    page_button = _polarisation_button(window, "Magnetic Fields")
    assert page_button.isEnabled()

    _polarisation_button(window, "Second site").click()

    assert page.isVisibleTo(window)
    assert _checked_owners(page) == ["ws-1"]

    page_button.click()
    assert _checked_owners(page) == [MAGNETIC_FIELD_PAGE_OWNER]
    window.close()


def test_a_simulation_is_saved_marks_the_stage_and_its_form_reloads(qapp, tmp_path: Path) -> None:
    from mufinder.app.main_window import MainWindow

    window, service, document = _window_with_fields(tmp_path)
    select_route(window, "polarisation")
    page = _polarisation_page(window)
    find_control(page, "Points", QSpinBox).setValue(201)
    find_control(page, "Name", QLineEdit).setText("ZF")

    find_control(page, "Add Simulation", QPushButton).click()

    assert COMPLETION_CHECKS["polarisation"](document)
    reloaded = service.load_project(document.project_root)
    (record,) = polarisation_simulations(reloaded)
    assert record.title == "ZF"
    assert reloaded.ui_state["polarisation_form"]["selected_simulation"] == record.simulation_id

    reopened = MainWindow()
    reopened._document = reloaded
    reopened._refresh_ui_after_project_change()
    assert _polarisation_page(reopened).selected_simulation_id() == record.simulation_id
    assert _polarisation_page(reopened).settings().time_points == 201
    window.close()
    reopened.close()
