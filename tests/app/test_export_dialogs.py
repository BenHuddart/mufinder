"""Tests for the shared export save dialogs and the relaxed-structure export flows."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QMenu

from mufinder.app import export_dialogs, workflow_table_utils
from mufinder.app.export_dialogs import (
    CASTEP_CELL_FORMAT,
    CIF_FORMAT,
    STRUCTURE_FORMATS,
    add_format_actions,
    ask_save_path,
    export_relaxed_structure,
    export_relaxed_structures_to_folder,
)
from mufinder.domain.models import Structure
from mufinder.science.structure_io import load_structure_file
from mufinder.services.structure_export import NO_STRUCTURE_REASON, RelaxedStructureExport


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
