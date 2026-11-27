"""The home page: the welcome page with no project open, the project home with one."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QPushButton, QWidget

import mufinder.app.home_page as home_page_module
from mufinder.app.home_page import HomePage
from mufinder.app.recent_project_button import RecentProjectButton
from mufinder.app.recent_projects import RecentProjects
from mufinder.services.project_service import ProjectService


@pytest.fixture
def recent(qapp) -> RecentProjects:
    return RecentProjects(QSettings("MuFinder", "MuFinder"))


@pytest.fixture
def page(recent: RecentProjects) -> HomePage:
    home = HomePage(recent)
    recent.changed.connect(home.refresh_recent_projects)
    return home


def _recent_rows(page: HomePage) -> list[QWidget]:
    layout = page.recent_projects_layout
    return [layout.itemAt(row).widget() for row in range(layout.count())]


def _button(page: HomePage, text: str) -> QPushButton:
    return next(button for button in page.findChildren(QPushButton) if button.text() == text)


def test_the_welcome_page_shows_its_three_cards_until_a_project_opens(page: HomePage) -> None:
    titles = [label.text() for label in page.welcome_page.findChildren(QLabel, "cardTitle")]

    assert titles == ["New Project", "Open Project", "Recent Projects"]
    assert page.content_stack.currentWidget() is page.welcome_page
    assert page.findChild(QLabel, "homeHeroTitle").text() == "Welcome to MuFinder"


def test_a_project_shows_the_project_home_without_the_logo_or_title(
    page: HomePage, tmp_path: Path
) -> None:
    document = ProjectService().create_project(tmp_path / "CoF2.mufinder", "CoF2")

    page.set_document(document)
    page.show_project_page(True)
    page.resize(900, 700)
    page.show()
    QApplication.processEvents()

    assert page.content_stack.currentWidget() is page.project_page
    assert page.project_home.name_label.text() == "CoF2"
    assert page.project_home.summary() is not None
    for name in ("homeHeroTitle", "homeHeroSubtitle", "homeLogo"):
        label = page.findChild(QLabel, name)
        assert label.isVisible() is False
        assert page.welcome_page.isAncestorOf(label)

    page.set_project_name("Renamed")
    assert page.project_home.name_label.text() == "Renamed"

    page.show_project_page(False)
    QApplication.processEvents()
    assert page.findChild(QLabel, "homeHeroTitle").isVisible() is True
    page.close()


def test_the_cards_ask_for_a_new_or_an_existing_project(page: HomePage) -> None:
    asked: list[str] = []
    page.new_requested.connect(lambda: asked.append("new"))
    page.open_requested.connect(lambda: asked.append("open"))

    _button(page, "Create New Project").click()
    _button(page, "Open Existing Project").click()

    assert asked == ["new", "open"]


def test_a_recent_entry_asks_for_its_project(
    page: HomePage, recent: RecentProjects, tmp_path: Path
) -> None:
    path = str(tmp_path / "CoF2.mufinder")
    recent.set_paths([path])
    requested: list[str] = []
    page.recent_requested.connect(requested.append)

    page.welcome_page.findChildren(RecentProjectButton)[0].click()

    assert requested == [path]


def test_a_stage_opened_on_the_project_home_is_asked_for(page: HomePage) -> None:
    requested: list[str] = []
    page.route_requested.connect(requested.append)

    page.project_home.route_requested.emit("uep")

    assert requested == ["uep"]


def test_recent_entries_list_at_most_five_projects_and_rebuild(
    page: HomePage, recent: RecentProjects, tmp_path: Path
) -> None:
    recent.set_paths([str(tmp_path / f"P{index}.mufinder") for index in range(7)])

    assert [button.accessibleName().rsplit(" ", 1)[1] for button in _recent_rows(page)] == [
        f"P{index}" for index in range(5)
    ]

    recent.clear()
    rows = _recent_rows(page)
    assert len(rows) == 1
    assert rows[0].text() == "No recent projects yet."


def test_card_titles_share_a_height_when_one_body_wraps_further(page: HomePage) -> None:
    # A short card beside a taller one takes on the extra height; none of it
    # may go to the title.
    host = QWidget()
    grid = QGridLayout(host)
    short, _short_layout = page.make_card("Short", "One line.")
    tall, _tall_layout = page.make_card("Tall", "Many words that wrap. " * 12)
    grid.addWidget(short, 0, 0)
    grid.addWidget(tall, 0, 1)
    host.resize(420, 400)
    host.show()
    QApplication.processEvents()

    short_title = short.findChild(QLabel, "cardTitle")
    tall_title = tall.findChild(QLabel, "cardTitle")

    assert short.height() == tall.height()
    assert short_title.height() == tall_title.height() == short_title.sizeHint().height()
    host.close()


def test_scaled_logo_is_none_for_a_missing_image(
    page: HomePage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(home_page_module, "logo_path", lambda: "/tmp/definitely-missing-logo.png")

    assert page.scaled_logo() is None


def test_scaled_logo_uses_the_device_pixel_ratio(
    page: HomePage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logo_file = tmp_path / "logo.png"
    assert QPixmap(64, 64).save(str(logo_file))
    monkeypatch.setattr(home_page_module, "logo_path", lambda: str(logo_file))
    monkeypatch.setattr(page, "logo_device_pixel_ratio", lambda: 2.0)

    scaled = page.scaled_logo()

    assert scaled is not None
    assert not scaled.isNull()
    assert scaled.devicePixelRatio() == 2.0
    assert scaled.width() == 184
    assert scaled.height() == 184
