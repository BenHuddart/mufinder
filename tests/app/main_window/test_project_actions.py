"""Project actions in the main window: new, open, recent, save, close, journal and file helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from ase import Atoms
from ase.io import write as ase_write

import mufinder.app.main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.castep_setup import CastepSetup
from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectService
from tests.support.qt import find_item_by_route
from tests.support.main_window import select_route


def test_new_project_success(main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window = main_window

    monkeypatch.setattr(main_window_module.QInputDialog, "getText", lambda *args, **kwargs: ("MyProj", True))
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )

    window._new_project()

    assert window._document is not None
    assert window._document.project.name == "MyProj"
    assert (tmp_path / "MyProj.mufinder" / "project.json").exists()
    window.close()


def test_new_project_existing_non_empty_warns(main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "Exists.mufinder"
    root.mkdir()
    (root / "dummy.txt").write_text("x", encoding="utf-8")

    window = main_window
    calls: list[str] = []

    monkeypatch.setattr(main_window_module.QInputDialog, "getText", lambda *args, **kwargs: ("Exists", True))
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: calls.append("warn"))

    window._new_project()

    assert calls == ["warn"]
    window.close()


def test_open_invalid_project_shows_error(main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "not_project"
    folder.mkdir()

    window = main_window
    errors: list[str] = []
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(folder),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *args, **kwargs: errors.append("critical"))

    window._open_project()
    assert errors == ["critical"]
    window.close()


def test_main_window_recent_projects_tracks_latest_first(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeSettings:
        def __init__(self, *args, **kwargs) -> None:
            self._values: dict[str, object] = {}

        def value(self, key: str, default=None, value_type=None):  # noqa: ANN001 - Qt compatibility.
            return self._values.get(key, default)

        def setValue(self, key: str, value) -> None:  # noqa: N802, ANN001 - Qt API compatibility.
            self._values[key] = value

    monkeypatch.setattr(main_window_module, "QSettings", FakeSettings)

    service = ProjectService()
    first = service.create_project(tmp_path / "first_recent.mufinder", "FirstRecent")
    second = service.create_project(tmp_path / "second_recent.mufinder", "SecondRecent")

    window = MainWindow()

    assert window._open_project_from_path(first.project_root)
    assert window._open_project_from_path(second.project_root)

    recents = window._recent_projects()
    assert recents[:2] == [str(second.project_root), str(first.project_root)]

    menu_actions = [action.text() for action in window._recent_projects_menu.actions() if action.text()]
    assert str(second.project_root) in menu_actions
    assert str(first.project_root) in menu_actions
    assert "Clear Recent Projects" in menu_actions
    window.close()


def test_open_recent_project_removes_missing_entry(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeSettings:
        def __init__(self, *args, **kwargs) -> None:
            self._values: dict[str, object] = {}

        def value(self, key: str, default=None, value_type=None):  # noqa: ANN001 - Qt compatibility.
            return self._values.get(key, default)

        def setValue(self, key: str, value) -> None:  # noqa: N802, ANN001 - Qt API compatibility.
            self._values[key] = value

    monkeypatch.setattr(main_window_module, "QSettings", FakeSettings)

    window = MainWindow()
    missing_path = str(tmp_path / "missing_recent.mufinder")
    window._set_recent_projects([missing_path])
    window._refresh_recent_projects_menu()

    warnings: list[str] = []
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args, **kwargs: warnings.append("warn"))

    window._open_recent_project(missing_path)

    assert warnings == ["warn"]
    assert window._recent_projects() == []
    window.close()


def test_file_dialogs_use_scoped_last_selected_directory(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeSettings:
        def __init__(self, *args, **kwargs) -> None:
            self._values: dict[str, str] = {}

        def value(self, key: str, default: str = "", value_type: type[str] | None = None) -> str:
            value = self._values.get(key, default)
            return str(value) if value_type is str else value

        def setValue(self, key: str, value: str) -> None:  # noqa: N802 - Qt API naming.
            self._values[key] = value

    monkeypatch.setattr(main_window_module, "QSettings", FakeSettings)

    project_root = tmp_path / "dialog_reuse.mufinder"
    ProjectService().create_project(project_root, "DialogReuse")

    window = MainWindow()

    chosen_dirs: list[str] = []

    def fake_get_existing_directory(*args, **kwargs) -> str:
        start_dir = args[2] if len(args) > 2 else kwargs.get("dir", "")
        chosen_dirs.append(start_dir)
        if len(chosen_dirs) == 1:
            return str(project_root)
        return ""

    source_dir = tmp_path / "imports"
    source_dir.mkdir()
    source = source_dir / "host.cif"
    atoms = Atoms(
        "Si",
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )
    ase_write(source, atoms)

    structure_dialog_dirs: list[str] = []

    def fake_get_open_file_name(*args, **kwargs) -> tuple[str, str]:
        start_dir = args[2] if len(args) > 2 else kwargs.get("dir", "")
        structure_dialog_dirs.append(start_dir)
        return (str(source), "Structure Files")

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        fake_get_existing_directory,
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        fake_get_open_file_name,
    )

    window._open_project()
    window._on_load_structure_requested()
    window._open_project()

    assert structure_dialog_dirs == [str(project_root)]
    assert len(chosen_dirs) == 2
    assert chosen_dirs[1] == str(project_root)
    window.close()


def test_save_without_document_shows_info(main_window, monkeypatch: pytest.MonkeyPatch) -> None:
    window = main_window
    infos: list[str] = []
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *args, **kwargs: infos.append("info"))

    window._save_project()

    assert infos == ["info"]
    window.close()


def test_add_demo_job_requires_project(main_window, monkeypatch: pytest.MonkeyPatch) -> None:
    window = main_window
    infos: list[str] = []
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *args, **kwargs: infos.append("info"))
    window._add_demo_job()
    assert infos == ["info"]
    window.close()


def test_open_castep_setup_dialog_updates_status_bar(main_window, monkeypatch: pytest.MonkeyPatch) -> None:
    window = main_window

    class FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, **kwargs) -> None:
            self._setup = type(
                "Setup",
                (),
                {
                    "executable_path": "/opt/castep/bin/castep",
                    "keywords_json_path": "/tmp/castep_keywords.json",
                    "mpirun_path": "",
                },
            )()

        def exec(self):
            return 1

        def setup(self):
            return self._setup

    monkeypatch.setattr(main_window_module, "CastepSetupDialog", FakeDialog)

    window._open_castep_setup_dialog()

    assert "CASTEP setup updated" in window.statusBar().currentMessage()
    window.close()


def test_journal_write_does_not_change_active_workflow_view_on_submission(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "journal_route_preserved.mufinder", "JournalRoutePreserved")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    output_dir = doc.project_root / "files" / "convergence_cutoff"
    (output_dir / job.label).mkdir(parents=True, exist_ok=True)
    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "sweep_values": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=output_dir,
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )
    window._document = doc
    window._refresh_ui_after_project_change()

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/usr/bin/castep", mpirun_path=""),
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(num_cores=1, parallel=False, max_concurrent=1)

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(window._local_runner, "submit", lambda **kwargs: None)

    select_route(window, "host_structure")
    assert window._main_stack.currentWidget() is window._host_structure_view

    window._submit_jobs_locally([job])

    assert window._main_stack.currentWidget() is window._host_structure_view
    assert window._active_route() == "host_structure"
    window.close()


def test_main_window_project_tree_uses_journal_route_and_view(main_window, tmp_path: Path) -> None:
    window = main_window
    doc = ProjectService().create_project(tmp_path / "journal_route.mufinder", "JournalRoute")
    window._document = doc
    window._refresh_ui_after_project_change()

    journal_item = None
    for index in range(window._project_tree.topLevelItemCount()):
        item = window._project_tree.topLevelItem(index)
        if item is not None:
            journal_item = find_item_by_route(item, "provenance_log")
        if journal_item is not None:
            break
    assert journal_item is not None
    assert journal_item.text(0) == "Journal"

    select_route(window, "provenance_log")

    assert window._main_stack.currentWidget() is window._journal_view
    window.close()


def test_main_window_journal_populates_from_imported_provenance(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "journal_populated.mufinder", "JournalPopulated")
    window._document = doc

    source = tmp_path / "host.cif"
    atoms = Atoms(
        "Si",
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )
    ase_write(source, atoms)
    service.import_structure_file(doc, source, make_active=True)

    window._refresh_ui_after_project_change()
    select_route(window, "provenance_log")

    texts = [label.text() for label in window._journal_view.findChildren(main_window_module.QLabel)]
    expected_day = doc.provenance[-1].timestamp.astimezone().strftime("%A %d %B %Y")
    assert expected_day in texts
    assert "Host structure imported" in texts
    window.close()


def test_main_window_close_prompts_to_save_visible_project(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "close_prompt.mufinder", "ClosePrompt")

    saves: list[bool] = []
    shutdowns: list[bool] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: main_window_module.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(window._project_service, "save_project", lambda _doc: saves.append(True))
    monkeypatch.setattr(window._local_runner, "shutdown", lambda: shutdowns.append(True))

    window.show()
    window.close()

    assert saves == [True]
    assert shutdowns == [True]


def test_main_window_close_cancel_keeps_window_open(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    window._document = service.create_project(tmp_path / "close_cancel.mufinder", "CloseCancel")

    shutdowns: list[bool] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: main_window_module.QMessageBox.StandardButton.Cancel,
    )
    monkeypatch.setattr(window._local_runner, "shutdown", lambda: shutdowns.append(True))

    window.show()
    window.close()

    assert shutdowns == []
    assert window.isVisible() is True
    window._document = None
    window.close()


def test_main_window_file_helpers_cover_file_prompt_and_unique_archive(main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeMessageBox:
        Icon = SimpleNamespace(Warning=1)
        StandardButton = SimpleNamespace(Yes=1, No=2, Cancel=3, YesToAll=4, NoToAll=5)
        _answer = 1

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def setIcon(self, *_args, **_kwargs) -> None:
            pass

        def setWindowTitle(self, *_args, **_kwargs) -> None:
            pass

        def setText(self, *_args, **_kwargs) -> None:
            pass

        def setInformativeText(self, *_args, **_kwargs) -> None:
            pass

        def setStandardButtons(self, *_args, **_kwargs) -> None:
            pass

        def setDefaultButton(self, *_args, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return _FakeMessageBox._answer

    window = main_window
    monkeypatch.setattr(main_window_module, "QMessageBox", _FakeMessageBox)

    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.Yes
    assert window._prompt_cutoff_file_disposition(300.0, tmp_path / "job") == "delete"
    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.No
    assert window._prompt_cutoff_file_disposition(300.0, tmp_path / "job") == "archive"
    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.Cancel
    assert window._prompt_cutoff_file_disposition(300.0, tmp_path / "job") is None
    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.Yes
    assert window._prompt_cutoff_file_disposition(300.0, tmp_path / "job", include_apply_all=True) == "delete"
    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.YesToAll
    assert window._prompt_cutoff_file_disposition(300.0, tmp_path / "job", include_apply_all=True) == "delete_all"
    _FakeMessageBox._answer = _FakeMessageBox.StandardButton.NoToAll
    assert window._prompt_cutoff_file_disposition(300.0, tmp_path / "job", include_apply_all=True) == "archive_all"

    file_path = tmp_path / "orphan.txt"
    file_path.write_text("x", encoding="utf-8")
    window._delete_path(file_path)
    assert not file_path.exists()

    output_root = tmp_path / "files"
    source = output_root / "job_a"
    archive_root = output_root / "removed_from_workflow"
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "job_a").mkdir()
    source.mkdir(parents=True, exist_ok=True)

    archived = window._archive_cutoff_directory(source, output_root)

    assert archived.name == "job_a_1"
    assert archived.exists()
    assert not source.exists()
    window.close()
