"""Tests for the shared export save dialogs and the relaxed-structure export flows."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QMenu

from mufinder.app import export_dialogs, workflow_table_utils
from mufinder.app.export_dialogs import (
    CASTEP_CELL_FORMAT,
    CIF_FORMAT,
    EXPORT_JOB_FILES_TEXT,
    NO_JOB_REASON,
    OPEN_CONTAINING_FOLDER_TEXT,
    STRUCTURE_FORMATS,
    JobFilesMenu,
    add_format_actions,
    ask_save_path,
    export_relaxed_structure,
    export_job_files_to_folder,
    export_relaxed_structures_to_folder,
    open_containing_folder,
)
from mufinder.domain.models import Structure
from mufinder.science.structure_io import load_structure_file
from mufinder.services.job_file_export import NO_FOLDER_REASON, JobFilesExport, job_files_export
from mufinder.services.structure_export import NO_STRUCTURE_REASON, RelaxedStructureExport
from tests.support.jobs import make_job


def _export(stem: str = "mu1_relaxed", directory: Path | None = None) -> RelaxedStructureExport:
    return RelaxedStructureExport(
        structure=Structure(
            label="Mu1 relaxed",
            lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            species=["Fe", "H:mu"],
            fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        ),
        default_stem=stem,
        header_lines=("Relaxed structure exported by MuFinder",),
        default_directory=directory,
    )


class _Messages:
    def __init__(self, monkeypatch) -> None:
        self.information: list[tuple[str, str]] = []
        self.warning: list[tuple[str, str]] = []
        monkeypatch.setattr(
            export_dialogs.QMessageBox,
            "information",
            lambda _parent, title, text: self.information.append((title, text)),
        )
        monkeypatch.setattr(
            export_dialogs.QMessageBox,
            "warning",
            lambda _parent, title, text: self.warning.append((title, text)),
        )


def _patch_save_dialog(monkeypatch, path: str, selected_filter: str, seen: list | None = None):
    def fake(_parent, _title, suggested, filters):
        if seen is not None:
            seen.append((suggested, filters))
        return path, selected_filter

    monkeypatch.setattr(export_dialogs.QFileDialog, "getSaveFileName", fake)


@pytest.mark.parametrize(
    ("chosen", "selected", "expected"),
    [
        ("out.cif", CIF_FORMAT.filter, "out.cif"),
        ("out.CELL", CIF_FORMAT.filter, "out.CELL"),
        ("out", CASTEP_CELL_FORMAT.filter, "out.cell"),
        ("out.txt", CIF_FORMAT.filter, "out.txt.cif"),
        ("out", "unknown", "out.cif"),
    ],
)
def test_ask_save_path_enforces_an_offered_extension(
    qapp, monkeypatch, tmp_path: Path, chosen: str, selected: str, expected: str
) -> None:
    _patch_save_dialog(monkeypatch, str(tmp_path / chosen), selected)

    path = ask_save_path(None, "Export", "x.cif", STRUCTURE_FORMATS)

    assert path == tmp_path / expected


def test_ask_save_path_suggests_the_name_in_the_directory(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    seen: list = []
    _patch_save_dialog(monkeypatch, "", "", seen)

    assert ask_save_path(None, "Export", "x.cif", STRUCTURE_FORMATS, directory=tmp_path) is None
    assert seen == [(str(tmp_path / "x.cif"), "CIF Files (*.cif);;CASTEP Cell Files (*.cell)")]


@pytest.mark.parametrize("fmt", [CIF_FORMAT, CASTEP_CELL_FORMAT])
def test_export_relaxed_structure_suggests_and_writes_the_chosen_format(
    qapp, monkeypatch, tmp_path: Path, fmt
) -> None:
    messages = _Messages(monkeypatch)
    seen: list = []
    _patch_save_dialog(monkeypatch, str(tmp_path / "picked"), fmt.filter, seen)

    written = export_relaxed_structure(None, _export(directory=tmp_path), fmt)

    assert written == tmp_path / f"picked{fmt.suffix}"
    assert seen == [(str(tmp_path / f"mu1_relaxed{fmt.suffix}"), fmt.filter)]
    marker = "#" if fmt is CIF_FORMAT else "!"
    assert written.read_text(encoding="utf-8").startswith(
        f"{marker} Relaxed structure exported by MuFinder"
    )
    expected_muon = "H" if fmt is CIF_FORMAT else "H:mu"
    assert load_structure_file(written).species == ["Fe", expected_muon]
    assert messages.information == [("Export complete", f"Saved to:\n{written}")]


def test_export_relaxed_structure_cancelled_writes_nothing(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    messages = _Messages(monkeypatch)
    _patch_save_dialog(monkeypatch, "", "")

    assert export_relaxed_structure(None, _export(), CIF_FORMAT) is None
    assert messages.information == []
    assert list(tmp_path.iterdir()) == []


def test_export_relaxed_structure_without_a_structure_explains_why(qapp, monkeypatch) -> None:
    messages = _Messages(monkeypatch)
    _patch_save_dialog(monkeypatch, "never", "")

    unavailable = RelaxedStructureExport(None, NO_STRUCTURE_REASON)
    assert export_relaxed_structure(None, unavailable, CIF_FORMAT) is None
    assert messages.information == [("Nothing to export", NO_STRUCTURE_REASON)]


def test_export_relaxed_structure_reports_a_write_failure(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    messages = _Messages(monkeypatch)
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("", encoding="utf-8")
    _patch_save_dialog(monkeypatch, str(blocker / "out.cif"), CIF_FORMAT.filter)

    assert export_relaxed_structure(None, _export(), CIF_FORMAT) is None
    assert [title for title, _text in messages.warning] == ["Export failed"]


def test_add_format_actions_offers_each_format_in_order(qapp) -> None:
    menu = QMenu()
    chosen: list[object] = []

    add_format_actions(menu, chosen.append)
    actions = menu.actions()
    actions[1].trigger()

    assert [action.text() for action in actions] == ["CIF (.cif)…", "CASTEP cell (.cell)…"]
    assert chosen == [CASTEP_CELL_FORMAT]


def _patch_folder(monkeypatch, folder: str, seen: list | None = None) -> None:
    def fake(_parent, title, *_args, **_kwargs):
        if seen is not None:
            seen.append(title)
        return folder

    monkeypatch.setattr(export_dialogs.QFileDialog, "getExistingDirectory", fake)


def test_folder_export_writes_one_file_per_structure_with_unique_names(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    messages = _Messages(monkeypatch)
    titles: list[str] = []
    _patch_folder(monkeypatch, str(tmp_path), titles)

    written = export_relaxed_structures_to_folder(
        None,
        [
            _export("mu1_relaxed"),
            RelaxedStructureExport(None, NO_STRUCTURE_REASON),
            _export("mu1_relaxed"),
            _export("mu2_relaxed"),
        ],
        CIF_FORMAT,
    )

    assert titles == ["Choose a folder for the relaxed structures (CIF)"]
    assert written == [
        tmp_path / "mu1_relaxed.cif",
        tmp_path / "mu1_relaxed_2.cif",
        tmp_path / "mu2_relaxed.cif",
    ]
    assert all("H   Mu1" in path.read_text(encoding="utf-8") for path in written)
    assert messages.information == [("Export complete", f"Saved 3 files to:\n{tmp_path}")]


def test_folder_export_writes_cell_files_when_chosen(qapp, monkeypatch, tmp_path: Path) -> None:
    _Messages(monkeypatch)
    _patch_folder(monkeypatch, str(tmp_path))

    written = export_relaxed_structures_to_folder(
        None, [_export("mu1_relaxed")], CASTEP_CELL_FORMAT
    )

    assert written == [tmp_path / "mu1_relaxed.cell"]
    assert "H:mu 0.25 0.25 0.25" in written[0].read_text(encoding="utf-8")


def test_folder_export_confirms_before_replacing_files(qapp, monkeypatch, tmp_path: Path) -> None:
    _Messages(monkeypatch)
    _patch_folder(monkeypatch, str(tmp_path))
    existing = tmp_path / "mu1_relaxed.cell"
    existing.write_text("keep", encoding="utf-8")
    asked: list[str] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, _title, text, *_args: asked.append(text) or False,
    )

    written = export_relaxed_structures_to_folder(
        None, [_export("mu1_relaxed")], CASTEP_CELL_FORMAT
    )

    assert written == []
    assert asked == [f"Replace 1 file in {tmp_path}?"]
    assert existing.read_text(encoding="utf-8") == "keep"


def test_folder_export_cancelled_at_folder_choice_writes_nothing(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    messages = _Messages(monkeypatch)
    _patch_folder(monkeypatch, "")

    assert export_relaxed_structures_to_folder(None, [_export()], CIF_FORMAT) == []
    assert list(tmp_path.iterdir()) == []
    assert messages.information == []


def test_folder_export_with_nothing_available_says_so(qapp, monkeypatch) -> None:
    messages = _Messages(monkeypatch)

    unavailable = [RelaxedStructureExport(None, NO_STRUCTURE_REASON)]
    assert export_relaxed_structures_to_folder(None, unavailable, CIF_FORMAT) == []
    assert [title for title, _text in messages.information] == ["Nothing to export"]


def test_ask_save_path_replaces_another_export_suffix_instead_of_doubling(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    _patch_save_dialog(monkeypatch, str(tmp_path / "site.cif"), CASTEP_CELL_FORMAT.filter)

    assert ask_save_path(None, "Export", "x.cell", (CASTEP_CELL_FORMAT,)) == tmp_path / "site.cell"


@pytest.mark.parametrize("accept", [True, False])
def test_ask_save_path_confirms_replacing_a_file_its_added_suffix_points_at(
    qapp, monkeypatch, tmp_path: Path, accept: bool
) -> None:
    existing = tmp_path / "mu1.cif"
    existing.write_text("keep", encoding="utf-8")
    _patch_save_dialog(monkeypatch, str(tmp_path / "mu1"), CIF_FORMAT.filter)
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, title, text, *_args: asked.append((title, text)) or accept,
    )

    path = ask_save_path(None, "Export", "x.cif", (CIF_FORMAT,))

    assert asked == [("Replace file", "Replace mu1.cif?")]
    assert path == (existing if accept else None)


def test_ask_save_path_leaves_overwrite_to_the_dialog_when_the_name_is_complete(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "mu1.cif").write_text("keep", encoding="utf-8")
    _patch_save_dialog(monkeypatch, str(tmp_path / "mu1.cif"), CIF_FORMAT.filter)
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda *_args: pytest.fail("the native dialog already asked"),
    )

    assert ask_save_path(None, "Export", "x.cif", (CIF_FORMAT,)) == tmp_path / "mu1.cif"


def _job_files(tmp_path: Path, seed: str, *names: str) -> JobFilesExport:
    directory = tmp_path / "project" / seed
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text(name, encoding="utf-8")
    return job_files_export(make_job(seed), directory, seed)


def test_job_files_export_copies_each_job_into_its_seed_folder(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    messages = _Messages(monkeypatch)
    destination = tmp_path / "out"
    titles: list[str] = []
    _patch_folder(monkeypatch, str(destination), titles)

    written = export_job_files_to_folder(
        None,
        [
            _job_files(tmp_path, "Si", "Si.cell", "Si.castep", "Si.check"),
            JobFilesExport(make_job("remote"), "remote", None, (), NO_FOLDER_REASON),
        ],
    )

    assert titles == ["Choose a folder for the job files"]
    assert written == [destination / "Si" / "Si.castep", destination / "Si" / "Si.cell"]
    assert not (destination / "Si" / "Si.check").exists()
    assert messages.information == [("Export complete", f"Saved 2 files to:\n{destination}")]


def test_job_files_export_confirms_before_replacing_and_keeps_files_when_declined(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    _Messages(monkeypatch)
    destination = tmp_path / "out"
    (destination / "Si").mkdir(parents=True)
    (destination / "Si" / "Si.cell").write_text("keep", encoding="utf-8")
    _patch_folder(monkeypatch, str(destination))
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, _title, text, detail, *_args: asked.append((text, detail)) or False,
    )

    written = export_job_files_to_folder(None, [_job_files(tmp_path, "Si", "Si.cell")])

    assert written == []
    assert asked == [(f"Replace 1 file in {destination}?", "Si/Si.cell")]
    assert (destination / "Si" / "Si.cell").read_text(encoding="utf-8") == "keep"


def test_job_files_export_reports_files_it_could_not_copy(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    messages = _Messages(monkeypatch)
    destination = tmp_path / "out"
    _patch_folder(monkeypatch, str(destination))
    export = _job_files(tmp_path, "Si", "Si.cell", "Si.castep")
    (export.directory / "Si.castep").unlink()

    written = export_job_files_to_folder(None, [export])

    assert written == [destination / "Si" / "Si.cell"]
    assert messages.information == []
    [(title, text)] = messages.warning
    assert title == "Export incomplete"
    assert text.startswith(f"Saved 1 file to:\n{destination}\n\nCould not write:\nSi/Si.castep: ")


def test_job_files_export_with_nothing_available_says_why(qapp, monkeypatch) -> None:
    messages = _Messages(monkeypatch)

    unavailable = [JobFilesExport(make_job("remote"), "remote", None, (), NO_FOLDER_REASON)]

    assert export_job_files_to_folder(None, unavailable) == []
    assert messages.information == [("Nothing to export", NO_FOLDER_REASON)]


def test_job_files_export_cancelled_at_folder_choice_copies_nothing(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    messages = _Messages(monkeypatch)
    _patch_folder(monkeypatch, "")

    assert export_job_files_to_folder(None, [_job_files(tmp_path, "Si", "Si.cell")]) == []
    assert messages.information == []


def test_open_containing_folder_opens_the_directory(qapp, monkeypatch, tmp_path: Path) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        export_dialogs.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile())
    )

    open_containing_folder(tmp_path)

    assert opened == [str(tmp_path)]


def _menu_actions(provider) -> dict[str, object]:
    menu = JobFilesMenu()
    menu.set_provider(provider)
    return {action.text: action for action in menu.row_actions(None)}


def test_job_files_menu_unwired_offers_nothing() -> None:
    actions = {action.text: action for action in JobFilesMenu().row_actions(None)}

    assert list(actions) == [OPEN_CONTAINING_FOLDER_TEXT, EXPORT_JOB_FILES_TEXT]
    assert actions[OPEN_CONTAINING_FOLDER_TEXT].separator_before
    for action in actions.values():
        assert not action.is_enabled(["row1"])
        assert action.tooltip_for(["row1"]) == NO_JOB_REASON


def test_job_files_menu_opens_one_jobs_folder_and_exports_any_with_files(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    with_files = _job_files(tmp_path, "Si", "Si.cell")
    no_folder = JobFilesExport(make_job("remote"), "remote", None, (), NO_FOLDER_REASON)
    by_key = {"si": with_files, "remote": no_folder}
    actions = _menu_actions(lambda keys: [by_key[key] for key in keys])
    opened: list[str] = []
    monkeypatch.setattr(
        export_dialogs.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile())
    )
    exported: list[list[JobFilesExport]] = []
    monkeypatch.setattr(
        export_dialogs,
        "export_job_files_to_folder",
        lambda _parent, exports: exported.append(list(exports)),
    )

    folder = actions[OPEN_CONTAINING_FOLDER_TEXT]
    assert folder.is_enabled(["si"])
    assert folder.tooltip_for(["si"]) == f"Open {with_files.directory} in the file manager."
    assert not folder.is_enabled(["remote"])
    assert folder.tooltip_for(["remote"]) == NO_FOLDER_REASON
    assert not folder.is_enabled(["si", "remote"])
    assert folder.tooltip_for(["si", "remote"]) == "Select one row to open its job folder."
    folder.trigger(["si"])
    assert opened == [str(with_files.directory)]

    files = actions[EXPORT_JOB_FILES_TEXT]
    assert files.is_enabled(["si", "remote"])
    assert files.tooltip_for(["si", "remote"]).startswith("Copy the CASTEP input and output")
    assert not files.is_enabled(["remote"])
    assert files.tooltip_for(["remote"]) == NO_FOLDER_REASON
    files.trigger(["si", "remote"])
    assert exported == [[with_files, no_folder]]


def test_job_files_export_refuses_to_copy_files_onto_themselves(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    messages = _Messages(monkeypatch)
    export = _job_files(tmp_path, "Si", "Si.cell")
    _patch_folder(monkeypatch, str(export.directory.parent))

    assert export_job_files_to_folder(None, [export]) == []
    assert [title for title, _text in messages.warning] == ["Choose another folder"]
    assert (export.directory / "Si.cell").read_text(encoding="utf-8") == "Si.cell"


def test_job_files_export_replace_prompt_lists_at_most_ten_files(
    qapp, monkeypatch, tmp_path: Path
) -> None:
    _Messages(monkeypatch)
    names = [f"Si.{index:04d}.err" for index in range(12)]
    export = _job_files(tmp_path, "Si", *names)
    destination = tmp_path / "out"
    (destination / "Si").mkdir(parents=True)
    for name in names:
        (destination / "Si" / name).write_text("old", encoding="utf-8")
    _patch_folder(monkeypatch, str(destination))
    details: list[str] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, _title, _text, detail, *_args: details.append(detail) or False,
    )

    export_job_files_to_folder(None, [export])

    lines = details[0].splitlines()
    assert len(lines) == 11
    assert lines[-1] == "… and 2 more"


def test_job_files_menu_resolves_rows_once_per_menu(qapp) -> None:
    calls: list[tuple[object, ...]] = []
    menu = JobFilesMenu()
    menu.set_provider(lambda keys: calls.append(tuple(keys)) or ())

    for action in menu.row_actions(None):
        action.is_enabled(["row1"])
        action.tooltip_for(["row1"])
    assert calls == [("row1",)]

    qapp.processEvents()
    menu.row_actions(None)[0].is_enabled(["row1"])
    assert calls == [("row1",), ("row1",)]
