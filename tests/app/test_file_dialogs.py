"""File dialogs that open in the folder last chosen for the same purpose."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QWidget

import mufinder.app.file_dialogs as file_dialogs_module
from mufinder.app.file_dialogs import RememberedFileDialogs


@pytest.fixture
def dialogs(qapp) -> RememberedFileDialogs:
    return RememberedFileDialogs(QSettings("MuFinder", "MuFinder"), QWidget())


def test_with_nothing_remembered_a_dialog_opens_at_home(dialogs: RememberedFileDialogs) -> None:
    assert dialogs.start_dir("open_project") == str(Path.home())
    assert dialogs.start_dir() == str(Path.home())


def test_a_context_opens_where_its_last_choice_was(
    dialogs: RememberedFileDialogs, tmp_path: Path
) -> None:
    projects = tmp_path / "projects"
    structures = tmp_path / "structures"
    projects.mkdir()
    structures.mkdir()

    dialogs.remember(projects, "open_project")
    dialogs.remember(structures / "host.cif", "load_structure")

    assert dialogs.start_dir("open_project") == str(projects)
    assert dialogs.start_dir("load_structure") == str(structures)
    # A context with no choice of its own takes the last folder of any dialog.
    assert dialogs.start_dir("new_project_parent") == str(structures)


def test_the_settings_keys_are_the_ones_older_versions_wrote(
    dialogs: RememberedFileDialogs, tmp_path: Path
) -> None:
    settings = QSettings("MuFinder", "MuFinder")
    settings.setValue("ui/last_dialog_dir_by_context/open_project", str(tmp_path))

    assert dialogs.start_dir("open_project") == str(tmp_path)

    dialogs.remember(tmp_path, "load_structure")
    assert settings.value("ui/last_dialog_dir") == str(tmp_path)
    assert settings.value("ui/last_dialog_dir_by_context/load_structure") == str(tmp_path)


def test_a_remembered_folder_that_has_gone_falls_back_to_its_parent(
    dialogs: RememberedFileDialogs, tmp_path: Path
) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    dialogs.remember(gone, "open_project")
    gone.rmdir()

    assert dialogs.start_dir("open_project") == str(tmp_path)


def test_a_path_with_no_folder_left_is_not_remembered(
    dialogs: RememberedFileDialogs, tmp_path: Path
) -> None:
    dialogs.remember(tmp_path / "missing" / "deeper" / "file.cif", "load_structure")

    assert dialogs.start_dir("load_structure") == str(Path.home())


@pytest.mark.parametrize(
    ("name", "ask"),
    [
        ("getOpenFileName", lambda dialogs: dialogs.get_open_file_name("Open", "*", "files")),
        ("getSaveFileName", lambda dialogs: dialogs.get_save_file_name("Save", "*", "files")),
    ],
)
def test_a_file_dialog_starts_in_its_folder_and_remembers_the_choice(
    dialogs: RememberedFileDialogs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    ask,
) -> None:
    dialogs.remember(tmp_path, "files")
    chosen = tmp_path / "sub" / "out.cif"
    chosen.parent.mkdir()
    starts: list[str] = []

    def answer(_parent, _title, start, _filter):
        starts.append(start)
        return str(chosen), "*"

    monkeypatch.setattr(file_dialogs_module.QFileDialog, name, answer)

    assert ask(dialogs) == (str(chosen), "*")
    assert starts == [str(tmp_path)]
    assert dialogs.start_dir("files") == str(chosen.parent)


def test_a_cancelled_folder_dialog_remembers_nothing(
    dialogs: RememberedFileDialogs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialogs.remember(tmp_path, "open_project")
    monkeypatch.setattr(file_dialogs_module.QFileDialog, "getExistingDirectory", lambda *args: "")

    assert dialogs.get_existing_directory("Open", "open_project") == ""
    assert dialogs.start_dir("open_project") == str(tmp_path)
