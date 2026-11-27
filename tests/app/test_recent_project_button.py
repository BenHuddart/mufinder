"""A recent-project entry: the project's name over its elided parent folder."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSizePolicy, QVBoxLayout, QWidget

from mufinder.app.eliding_label import ElidingLabel
from mufinder.app.recent_project_button import RecentProjectButton


def _line(button: RecentProjectButton, name: str) -> ElidingLabel:
    return button.findChild(ElidingLabel, name)


def _in_window(button: RecentProjectButton, width: int = 300) -> QWidget:
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.addWidget(button)
    holder.resize(width, 100)
    holder.show()
    QApplication.processEvents()
    return holder


@pytest.mark.parametrize(
    ("path", "name", "folder"),
    [
        ("/data/studies/CoF2.mufinder", "CoF2", "/data/studies"),
        ("/data/studies/CoF2.mufinder/", "CoF2", "/data/studies"),
        ("/data/studies/plain-folder", "plain-folder", "/data/studies"),
        ("/data/studies/.mufinder", ".mufinder", "/data/studies"),
        ("/data/My.mufinder.old.mufinder", "My.mufinder.old", "/data"),
    ],
)
def test_name_is_the_folder_without_its_suffix(qapp, path: str, name: str, folder: str) -> None:
    button = RecentProjectButton(path)

    assert _line(button, "recentProjectName").text() == name
    assert _line(button, "recentProjectFolder").text() == folder
    assert button.toolTip() == path
    assert button.accessibleName() == f"Open recent project {name}"


def test_a_relative_or_home_path_still_names_a_real_folder(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    relative = RecentProjectButton("rel.mufinder")
    home = RecentProjectButton("~/studies/rel.mufinder")

    assert _line(relative, "recentProjectFolder").text() == str(tmp_path)
    assert _line(home, "recentProjectFolder").text() == str(tmp_path / "home" / "studies")
    assert home.toolTip() == "~/studies/rel.mufinder"


def test_a_wrapped_eliding_label_displays_its_whole_text(qapp) -> None:
    label = ElidingLabel("Words " * 100)
    label.setWordWrap(True)
    label.resize(100, 400)

    assert label.displayed_text() == label.text()


def test_folder_line_elides_in_the_middle_and_keeps_both_ends(qapp) -> None:
    folder = "/start-of-the-path/" + "deep/" * 60 + "end-of-the-path"
    button = RecentProjectButton(folder + "/CoF2.mufinder")
    holder = _in_window(button)

    shown = _line(button, "recentProjectFolder").displayed_text()

    assert shown != folder
    assert "…" in shown
    assert shown.startswith("/start")
    assert shown.endswith("path")
    holder.close()


def test_a_long_name_elides_at_its_end(qapp) -> None:
    button = RecentProjectButton("/data/" + "n" * 200 + ".mufinder")
    holder = _in_window(button)

    assert _line(button, "recentProjectName").displayed_text().endswith("…")
    holder.close()


def test_width_never_comes_from_the_path(qapp) -> None:
    short = RecentProjectButton("/d/a.mufinder")
    long = RecentProjectButton("/" + "d" * 300 + "/" + "a" * 200 + ".mufinder")

    assert long.minimumSizeHint().width() == short.minimumSizeHint().width()
    assert long.sizeHint().width() == short.sizeHint().width()
    # Not the ``Minimum`` a push button has by default, which reads the size hint as a floor.
    assert long.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Preferred


def test_both_lines_fit_inside_the_button(qapp) -> None:
    button = RecentProjectButton("/data/studies/CoF2.mufinder")
    holder = _in_window(button)

    assert button.height() == button.minimumSizeHint().height()
    for label in (
        button.findChild(QWidget, "recentProjectName"),
        button.findChild(QWidget, "recentProjectFolder"),
    ):
        assert label.height() >= label.fontMetrics().height()
        assert label.geometry().bottom() < button.height()
    holder.close()


def test_a_click_on_a_line_reaches_the_button(qapp) -> None:
    button = RecentProjectButton("/data/studies/CoF2.mufinder")
    holder = _in_window(button)
    clicks: list[bool] = []
    button.clicked.connect(lambda *_args: clicks.append(True))
    name_line = button.findChild(QWidget, "recentProjectName")

    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=name_line.geometry().center())
    assert clicks == [True]
    assert name_line.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    holder.close()
