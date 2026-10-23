"""Save dialogs and result messages for exporting files from a results view.

Every export action asks for its destination through :func:`ask_save_path`,
which starts from a suggested name, offers one filter per format and makes sure
the chosen path carries one of those formats' extensions, and then reports the
outcome through :func:`report_export_saved` or :func:`report_export_failed`.
:func:`export_relaxed_structure` and :func:`export_relaxed_structures_to_folder`
are the relaxed-structure flows built on them; the file itself is written by
:func:`mufinder.science.structure_io.export_structure`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox, QWidget

from mufinder.app import workflow_table_utils
from mufinder.science.structure_io import export_structure
from mufinder.services.structure_export import RelaxedStructureExport


@dataclass(frozen=True, slots=True)
class ExportFormat:
    """One file format a save dialog offers: ``"CIF Files"`` with suffix ``".cif"``."""

    description: str
    suffix: str

    @property
    def filter(self) -> str:
        return f"{self.description} (*{self.suffix})"


CIF_FORMAT = ExportFormat("CIF Files", ".cif")
CASTEP_CELL_FORMAT = ExportFormat("CASTEP Cell Files", ".cell")
#: The formats a relaxed structure exports to, the default first.
STRUCTURE_FORMATS = (CIF_FORMAT, CASTEP_CELL_FORMAT)


def ask_save_path(
    parent: QWidget,
    title: str,
    default_name: str,
    formats: Sequence[ExportFormat],
    *,
    directory: Path | None = None,
) -> Path | None:
    """Ask where to save a file in one of ``formats``; ``None`` when cancelled.

    ``default_name`` is suggested inside ``directory`` when one is given. A
    chosen name without one of the formats' suffixes gets the suffix of the
    filter selected in the dialog (the first format when none matches).
    """

    suggested = str(directory / default_name) if directory is not None else default_name
    file_path, selected_filter = QFileDialog.getSaveFileName(
        parent, title, suggested, ";;".join(fmt.filter for fmt in formats)
    )
    if not file_path:
        return None
    path = Path(file_path)
    if path.suffix.lower() in {fmt.suffix for fmt in formats}:
        return path
    chosen = next((fmt for fmt in formats if fmt.filter == selected_filter), formats[0])
    return path.with_name(path.name + chosen.suffix)


def report_export_saved(parent: QWidget, path: Path, warnings: Sequence[str] = ()) -> None:
    """Tell the user where the export was saved, with any warnings it raised."""

    if warnings:
        QMessageBox.warning(
            parent,
            "Export saved with warnings",
            f"Saved to:\n{path}\n\n" + "\n".join(warnings),
        )
    else:
        QMessageBox.information(parent, "Export complete", f"Saved to:\n{path}")


def report_export_failed(parent: QWidget, error: Exception) -> None:
    """Tell the user an export could not be written."""

    QMessageBox.warning(parent, "Export failed", str(error))


def export_relaxed_structure(parent: QWidget, export: RelaxedStructureExport) -> Path | None:
    """Ask where to save one relaxed structure as CIF or CASTEP cell, write it and report.

    Returns the written path, or ``None`` when there was nothing to export, the
    dialog was cancelled or writing failed.
    """

    if export.structure is None:
        QMessageBox.information(
            parent, "Nothing to export", export.unavailable_reason or "No relaxed structure."
        )
        return None
    path = ask_save_path(
        parent,
        "Export Relaxed Structure",
        export.default_stem + STRUCTURE_FORMATS[0].suffix,
        STRUCTURE_FORMATS,
        directory=export.default_directory,
    )
    if path is None:
        return None
    try:
        export_structure(export.structure, path, header_lines=export.header_lines)
    except (OSError, ValueError) as exc:
        report_export_failed(parent, exc)
        return None
    report_export_saved(parent, path)
    return path


def export_relaxed_structures_to_folder(
    parent: QWidget, exports: Sequence[RelaxedStructureExport]
) -> list[Path]:
    """Write every exportable structure in ``exports`` to a chosen folder, one file each.

    Asks for the format, then the folder. Files are named from each export's
    ``default_stem``, with ``_2``, ``_3``… added where two would share a name.
    Writing over existing files is confirmed first. Returns the written paths.
    """

    available = [export for export in exports if export.structure is not None]
    if not available:
        QMessageBox.information(
            parent, "Nothing to export", "No relaxed structures are available to export."
        )
        return []
    labels = [fmt.filter for fmt in STRUCTURE_FORMATS]
    label, accepted = QInputDialog.getItem(
        parent, "Export Relaxed Structures", "File format:", labels, 0, False
    )
    if not accepted:
        return []
    chosen = STRUCTURE_FORMATS[labels.index(label)] if label in labels else STRUCTURE_FORMATS[0]
    folder = QFileDialog.getExistingDirectory(
        parent,
        "Choose a folder for the relaxed structures",
        str(available[0].default_directory or ""),
    )
    if not folder:
        return []
    targets = _unique_targets(Path(folder), available, chosen.suffix)
    existing = [path for path in targets if path.exists()]
    if existing and not workflow_table_utils.confirm_destructive(
        parent,
        "Replace files",
        f"Replace {workflow_table_utils.count_phrase(len(existing), ('file', 'files'))} "
        f"in {folder}?",
        "\n".join(path.name for path in existing),
        "Replace",
    ):
        return []

    written: list[Path] = []
    failures: list[str] = []
    for export, path in zip(available, targets, strict=True):
        assert export.structure is not None
        try:
            export_structure(export.structure, path, header_lines=export.header_lines)
        except (OSError, ValueError) as exc:
            failures.append(f"{path.name}: {exc}")
        else:
            written.append(path)
    if failures:
        QMessageBox.warning(
            parent,
            "Export incomplete",
            f"Saved {workflow_table_utils.count_phrase(len(written), ('file', 'files'))} to:\n"
            f"{folder}\n\nCould not write:\n" + "\n".join(failures),
        )
    else:
        QMessageBox.information(
            parent,
            "Export complete",
            f"Saved {workflow_table_utils.count_phrase(len(written), ('file', 'files'))} to:\n"
            f"{folder}",
        )
    return written


def _unique_targets(
    folder: Path, exports: Sequence[RelaxedStructureExport], suffix: str
) -> list[Path]:
    used: set[str] = set()
    targets: list[Path] = []
    for export in exports:
        stem = export.default_stem or "relaxed_structure"
        candidate, counter = stem, 1
        while candidate.lower() in used:
            counter += 1
            candidate = f"{stem}_{counter}"
        used.add(candidate.lower())
        targets.append(folder / f"{candidate}{suffix}")
    return targets


__all__ = [
    "CASTEP_CELL_FORMAT",
    "CIF_FORMAT",
    "STRUCTURE_FORMATS",
    "ExportFormat",
    "ask_save_path",
    "export_relaxed_structure",
    "export_relaxed_structures_to_folder",
    "report_export_failed",
    "report_export_saved",
]
