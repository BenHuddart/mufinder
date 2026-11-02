"""The New Project dialog: what it creates, and why it may refuse to."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import QFileDialog
import pytest

from mufinder.app.new_project_dialog import NewProjectDialog

pytestmark = pytest.mark.usefixtures("qapp")


def _dialog(location: Path | str, name: str = "") -> NewProjectDialog:
    dialog = NewProjectDialog(str(location))
    dialog._name_edit.setText(name)
    return dialog


def test_the_folder_it_will_create_shows_as_you_type(tmp_path: Path) -> None:
    dialog = _dialog(tmp_path)

    dialog._name_edit.setText("Co")
    dialog._name_edit.setText("CoF2")

    assert dialog._target_label.text() == str(tmp_path / "CoF2.mufinder")
    assert dialog.project_root() == tmp_path / "CoF2.mufinder"
    assert dialog._create_button.isEnabled()
    assert dialog._reason_label.text() == ""


def test_the_name_is_trimmed_and_a_typed_suffix_is_not_doubled(tmp_path: Path) -> None:
    dialog = _dialog(tmp_path, "  CoF2.mufinder ")

    assert dialog.project_name() == "CoF2"
    assert dialog.project_root() == tmp_path / "CoF2.mufinder"


@pytest.mark.parametrize("name", ["", "   ", ".mufinder"])
def test_an_empty_name_is_refused_with_the_reason(tmp_path: Path, name: str) -> None:
    dialog = _dialog(tmp_path, name)

    assert not dialog._create_button.isEnabled()
    assert dialog._reason_label.text() == "Enter a project name."
    assert dialog._target_label.text() == ""


@pytest.mark.parametrize("name", ["a/b", "a\\b", ".."])
def test_a_name_that_would_leave_the_location_is_refused(tmp_path: Path, name: str) -> None:
    dialog = _dialog(tmp_path, name)

    assert not dialog._create_button.isEnabled()
    assert "slash" in dialog._reason_label.text()


def test_a_folder_that_is_not_empty_is_refused(tmp_path: Path) -> None:
    (tmp_path / "CoF2.mufinder").mkdir()
    (tmp_path / "CoF2.mufinder" / "project.json").write_text("{}")

    dialog = _dialog(tmp_path, "CoF2")

    assert not dialog._create_button.isEnabled()
    assert dialog._reason_label.text() == "CoF2.mufinder already exists there and is not empty."

    dialog._name_edit.setText("MnSi")
    assert dialog._create_button.isEnabled()


def test_an_empty_folder_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "CoF2.mufinder").mkdir()

    assert _dialog(tmp_path, "CoF2")._create_button.isEnabled()


def test_a_file_in_the_way_is_refused(tmp_path: Path) -> None:
    (tmp_path / "CoF2.mufinder").write_text("")

    dialog = _dialog(tmp_path, "CoF2")

    assert not dialog._create_button.isEnabled()
    assert dialog._reason_label.text() == "CoF2.mufinder already exists there and is not a folder."


@pytest.mark.parametrize(
    ("location", "reason"),
    [
        ("", "Choose a location."),
        ("relative/place", "The location must be a full path."),
        ("~nosuchuserxyz/place", "The location must be a full path."),
    ],
)
def test_an_unusable_location_is_refused(location: str, reason: str) -> None:
    dialog = _dialog(location, "CoF2")

    assert not dialog._create_button.isEnabled()
    assert dialog._reason_label.text() == reason


def test_a_location_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    dialog = _dialog(tmp_path / "missing", "CoF2")

    assert not dialog._create_button.isEnabled()
    assert dialog._reason_label.text() == "The location is not an existing folder."


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs file modes")
def test_an_unreadable_target_folder_is_refused_with_the_reason(tmp_path: Path) -> None:
    target = tmp_path / "CoF2.mufinder"
    target.mkdir()
    target.chmod(0o000)
    try:
        dialog = _dialog(tmp_path, "CoF2")
        assert not dialog._create_button.isEnabled()
        assert "Permission denied" in dialog._reason_label.text()
    finally:
        target.chmod(0o755)


def test_create_checks_again_if_the_folder_filled_since_the_last_edit(tmp_path: Path) -> None:
    dialog = _dialog(tmp_path, "CoF2")
    assert dialog._create_button.isEnabled()
    (tmp_path / "CoF2.mufinder").mkdir()
    (tmp_path / "CoF2.mufinder" / "project.json").write_text("{}")

    dialog.accept()

    assert dialog.result() != dialog.DialogCode.Accepted
    assert not dialog._create_button.isEnabled()


def test_browse_fills_the_location_and_cancelling_it_keeps_the_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = _dialog(tmp_path, "CoF2")
    chosen = tmp_path / "elsewhere"
    chosen.mkdir()
    starts: list[str] = []

    def _pick(_parent, _title, start) -> str:
        starts.append(start)
        return str(chosen)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _pick)
    dialog._browse_button.click()
    assert starts == [str(tmp_path)]
    assert dialog.project_root() == chosen / "CoF2.mufinder"

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a: "")
    dialog._browse_button.click()
    assert dialog.project_root() == chosen / "CoF2.mufinder"


def test_a_long_location_does_not_widen_the_dialog(tmp_path: Path) -> None:
    short = _dialog(tmp_path, "CoF2").minimumSizeHint().width()
    long = _dialog("/" + "very-long-folder/" * 40, "CoF2")

    assert long._target_label.minimumSizeHint().width() == 0
    assert long.minimumSizeHint().width() <= short
