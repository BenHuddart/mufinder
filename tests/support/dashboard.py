"""Reach the Job Dashboard's action cell from a test."""

from __future__ import annotations

from mufinder.app import main_window as main_window_module
from tests.support.reset_menu import FakeMenu


def action_button(window, row: int = 0):
    """Return the row's one action button."""

    return window._jobs_table.cellWidget(row, window._JOB_ACTION_COLUMN).row_controls[0]


def more_button(window, row: int = 0):
    """Return the row's ``⋯`` button."""

    return window._jobs_table.cellWidget(row, window._JOB_ACTION_COLUMN).row_controls[1]


def job_row(window, job_id) -> int:
    """Return the dashboard row showing ``job_id``."""

    for row in range(window._jobs_table.rowCount()):
        if window._jobs_table_row_job_id(window._jobs_table.item(row, 0)) == job_id:
            return row
    raise AssertionError("job row missing")


def open_more_menu(window, monkeypatch, row: int = 0) -> FakeMenu:
    """Click the row's ``⋯`` button and return the menu it opened."""

    FakeMenu.instances = []
    monkeypatch.setattr(main_window_module, "QMenu", FakeMenu)
    more_button(window, row).click()
    return FakeMenu.instances[-1]


def open_context_menu(window, monkeypatch, row: int = 0) -> FakeMenu:
    """Right-click the row and return the menu it opened."""

    FakeMenu.instances = []
    monkeypatch.setattr(main_window_module, "QMenu", FakeMenu)
    rect = window._jobs_table.visualRect(window._jobs_table.model().index(row, 0))
    window._jobs_table.customContextMenuRequested.emit(rect.center())
    return FakeMenu.instances[-1]
