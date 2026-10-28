"""Save dialogs and result messages for exporting files from a results view.

Every export action asks for its destination through :func:`ask_save_path`,
which starts from a suggested name, offers one filter per format and makes sure
the chosen path carries one of those formats' extensions, and then reports the
outcome through :func:`report_export_saved` or :func:`report_export_failed`.
:func:`export_relaxed_structure` and :func:`export_relaxed_structures_to_folder`
are the relaxed-structure flows built on them, each for one format the user
picked from a menu (:data:`STRUCTURE_FORMATS`, :func:`add_format_actions`); the
file itself is written by :func:`mufinder.science.structure_io.export_structure`.
:func:`export_job_files_to_folder` copies jobs' CASTEP files into a chosen
folder, and :func:`open_containing_folder` shows a folder in the file manager;
:class:`JobFilesMenu` offers both from every workflow table's row menu.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox, QWidget

from mufinder.app import workflow_table_utils
from mufinder.science.structure_io import export_structure
from mufinder.services.job_file_export import (
    JobFilesExport,
    copies_onto_source,
    copy_job_files,
    plan_job_file_copy,
)
from mufinder.services.structure_export import RelaxedStructureExport


@dataclass(frozen=True, slots=True)
class ExportFormat:
    """One export file format: ``"CIF"``, filter ``"CIF Files"``, suffix ``".cif"``."""

    name: str
    description: str
    suffix: str

    @property
    def filter(self) -> str:
        return f"{self.description} (*{self.suffix})"

    @property
    def menu_text(self) -> str:
        """The menu entry that exports in this format: ``"CIF (.cif)…"``."""

        return f"{self.name} ({self.suffix})…"


CIF_FORMAT = ExportFormat("CIF", "CIF Files", ".cif")
CASTEP_CELL_FORMAT = ExportFormat("CASTEP cell", "CASTEP Cell Files", ".cell")
#: The formats a relaxed structure exports to, in menu order.
STRUCTURE_FORMATS = (CIF_FORMAT, CASTEP_CELL_FORMAT)
_KNOWN_SUFFIXES = frozenset(fmt.suffix for fmt in STRUCTURE_FORMATS)
#: Warning title when an exported muon site sits on or very near a host atom.
MUON_HOST_OVERLAP_TITLE = "Muon sites coincide with host atoms"


def add_format_actions(
    menu: QMenu,
    trigger: Callable[[ExportFormat], None],
    formats: Sequence[ExportFormat] = STRUCTURE_FORMATS,
) -> None:
    """Add one entry per format to ``menu`` (``"CIF (.cif)…"``) that calls ``trigger`` with it."""

    for fmt in formats:
        action = menu.addAction(fmt.menu_text)
        action.triggered.connect(lambda _checked=False, chosen=fmt: trigger(chosen))


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
    filter selected in the dialog (the first format when none matches); a name
    ending in another known export suffix (``site.cif`` while saving CASTEP
    cell) has that suffix replaced rather than doubled. When the suffix is
    changed after the dialog closed, the dialog's own overwrite question was
    about a different file, so replacing an existing file is confirmed here.
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
    if path.suffix.lower() in _KNOWN_SUFFIXES:
        path = path.with_suffix(chosen.suffix)
    else:
        path = path.with_name(path.name + chosen.suffix)
    if path.exists() and not workflow_table_utils.confirm_destructive(
        parent,
        "Replace file",
        f"Replace {path.name}?",
        f"A file with this name already exists in {path.parent}.",
        "Replace",
    ):
        return None
    return path


def report_export_saved(
    parent: QWidget,
    path: Path,
    warnings: Sequence[str] = (),
    *,
    warning_title: str = "Export saved with warnings",
) -> None:
    """Tell the user where the export was saved, with any warnings it raised under ``warning_title``."""

    if warnings:
        QMessageBox.warning(
            parent,
            warning_title,
            f"Saved to:\n{path}\n\n" + "\n".join(warnings),
        )
    else:
        QMessageBox.information(parent, "Export complete", f"Saved to:\n{path}")


def report_export_failed(parent: QWidget, error: Exception) -> None:
    """Tell the user an export could not be written."""

    QMessageBox.warning(parent, "Export failed", str(error))


def export_relaxed_structure(
    parent: QWidget, export: RelaxedStructureExport, fmt: ExportFormat
) -> Path | None:
    """Ask where to save one relaxed structure in ``fmt``, write it and report.

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
        f"Export Relaxed Structure as {fmt.name}",
        export.default_stem + fmt.suffix,
        (fmt,),
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
    parent: QWidget, exports: Sequence[RelaxedStructureExport], fmt: ExportFormat
) -> list[Path]:
    """Write every exportable structure in ``exports`` to a chosen folder in ``fmt``, one file each.

    Asks for the folder. Files are named from each export's
    ``default_stem``, with ``_2``, ``_3``… added where two would share a name.
    Writing over existing files is confirmed first. Returns the written paths.
    """

    available = [export for export in exports if export.structure is not None]
    if not available:
        QMessageBox.information(
            parent, "Nothing to export", "No relaxed structures are available to export."
        )
        return []
    folder = QFileDialog.getExistingDirectory(
        parent,
        f"Choose a folder for the relaxed structures ({fmt.name})",
        str(available[0].default_directory or ""),
    )
    if not folder:
        return []
    targets = _unique_targets(Path(folder), available, fmt.suffix)
    if not _confirm_replacing(parent, Path(folder), targets):
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
    _report_folder_export(parent, Path(folder), len(written), failures)
    return written


def export_job_files_to_folder(parent: QWidget, exports: Sequence[JobFilesExport]) -> list[Path]:
    """Copy the CASTEP files of every job in ``exports`` into a chosen folder.

    Each job's files go into a sub-folder named after its seed (see
    :func:`~mufinder.services.job_file_export.plan_job_file_copy`). Writing
    over existing files is confirmed first. Returns the written paths.
    """

    available = [export for export in exports if export.files]
    if not available:
        reasons = {export.unavailable_reason for export in exports if export.unavailable_reason}
        QMessageBox.information(
            parent,
            "Nothing to export",
            "\n".join(sorted(reasons)) or "No job files are available to export.",
        )
        return []
    folder = QFileDialog.getExistingDirectory(parent, "Choose a folder for the job files")
    if not folder:
        return []
    pairs = plan_job_file_copy(available, Path(folder))
    if copies_onto_source(pairs):
        QMessageBox.warning(
            parent,
            "Choose another folder",
            f"{folder} already holds these job folders, so the files would be copied onto "
            "themselves. Choose a folder outside the project's job folders.",
        )
        return []
    if not _confirm_replacing(parent, Path(folder), [target for _source, target in pairs]):
        return []
    copied, failures = copy_job_files(pairs)
    _report_folder_export(parent, Path(folder), len(copied), failures)
    return copied


#: Resolves a workflow table's selected row keys to their jobs' files.
JobFilesProvider = Callable[[Sequence[object]], Sequence[JobFilesExport]]
OPEN_CONTAINING_FOLDER_TEXT = "Open Containing Folder"
EXPORT_JOB_FILES_TEXT = "Export Inputs/Outputs…"
#: Tooltip when the selected rows have no job yet.
NO_JOB_REASON = "This row has no job yet; run it first."


class JobFilesMenu:
    """The job-folder entries of a workflow table's row menu.

    A table adds :meth:`row_actions` to its menu when it is built; the
    controller wiring then calls :meth:`set_provider` with how row keys resolve
    to jobs' files. The provider is asked fresh each time the menu opens.
    Unwired, no row has files.
    """

    def __init__(self) -> None:
        self._source: JobFilesProvider = lambda _keys: ()
        # One menu asks about the same keys several times (enabled state and
        # tooltip of each entry); answers are kept until control returns to
        # the event loop, so each menu resolves its rows once.
        self._cache: dict[tuple[object, ...], Sequence[JobFilesExport]] = {}

    def set_provider(self, provider: JobFilesProvider) -> None:
        self._source = provider
        self._cache.clear()

    def _provider(self, keys: Sequence[object]) -> Sequence[JobFilesExport]:
        cache_key = tuple(keys)
        if cache_key not in self._cache:
            if not self._cache:
                QTimer.singleShot(0, self._cache.clear)
            self._cache[cache_key] = self._source(keys)
        return self._cache[cache_key]

    def row_actions(self, parent: QWidget) -> tuple[workflow_table_utils.RowMenuAction, ...]:
        """Return Open Containing Folder (one row) and Export Inputs/Outputs… (any rows)."""

        return (
            workflow_table_utils.RowMenuAction(
                text=OPEN_CONTAINING_FOLDER_TEXT,
                is_enabled=lambda keys: self._single_directory(keys) is not None,
                trigger=lambda keys: self._open_folder(keys),
                separator_before=True,
                tooltip_for=self._folder_tooltip,
            ),
            workflow_table_utils.RowMenuAction(
                text=EXPORT_JOB_FILES_TEXT,
                is_enabled=lambda keys: any(export.files for export in self._provider(keys)),
                trigger=lambda keys: export_job_files_to_folder(parent, self._provider(keys)),
                tooltip_for=self._files_tooltip,
            ),
        )

    def _single_directory(self, keys: Sequence[object]) -> Path | None:
        if len(keys) != 1:
            return None
        exports = self._provider(keys)
        return exports[0].directory if exports else None

    def _open_folder(self, keys: Sequence[object]) -> None:
        directory = self._single_directory(keys)
        if directory is not None:
            open_containing_folder(directory)

    def _folder_tooltip(self, keys: Sequence[object]) -> str:
        if len(keys) > 1:
            return "Select one row to open its job folder."
        exports = self._provider(keys)
        if not exports:
            return NO_JOB_REASON
        if exports[0].directory is None:
            return exports[0].unavailable_reason or ""
        return f"Open {exports[0].directory} in the file manager."

    def _files_tooltip(self, keys: Sequence[object]) -> str:
        exports = self._provider(keys)
        if any(export.files for export in exports):
            return (
                "Copy the CASTEP input and output files (not checkpoints) into a folder you "
                "choose, one sub-folder per job."
            )
        if not exports:
            return NO_JOB_REASON
        return "\n".join(sorted({export.unavailable_reason or "" for export in exports}))


def open_containing_folder(directory: Path) -> None:
    """Show ``directory`` in the desktop's file manager."""

    QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))


def _confirm_replacing(parent: QWidget, folder: Path, targets: Sequence[Path]) -> bool:
    """Confirm replacing any of ``targets`` that exist; ``True`` when none do."""

    existing = [path for path in targets if path.exists()]
    if not existing:
        return True
    return workflow_table_utils.confirm_destructive(
        parent,
        "Replace files",
        f"Replace {workflow_table_utils.count_phrase(len(existing), ('file', 'files'))} "
        f"in {folder}?",
        _listed(str(path.relative_to(folder)) for path in existing),
        "Replace",
    )


#: Most file names a confirmation lists before summarising the rest.
_MAX_LISTED_FILES = 10


def _listed(names: Iterable[str]) -> str:
    names = list(names)
    shown = names[:_MAX_LISTED_FILES]
    if len(names) > len(shown):
        shown.append(f"… and {len(names) - len(shown)} more")
    return "\n".join(shown)


def _report_folder_export(
    parent: QWidget, folder: Path, written: int, failures: Sequence[str]
) -> None:
    saved = f"Saved {workflow_table_utils.count_phrase(written, ('file', 'files'))} to:\n{folder}"
    if failures:
        QMessageBox.warning(
            parent, "Export incomplete", f"{saved}\n\nCould not write:\n" + "\n".join(failures)
        )
    else:
        QMessageBox.information(parent, "Export complete", saved)


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
    "EXPORT_JOB_FILES_TEXT",
    "MUON_HOST_OVERLAP_TITLE",
    "NO_JOB_REASON",
    "OPEN_CONTAINING_FOLDER_TEXT",
    "STRUCTURE_FORMATS",
    "ExportFormat",
    "JobFilesMenu",
    "JobFilesProvider",
    "add_format_actions",
    "ask_save_path",
    "export_job_files_to_folder",
    "export_relaxed_structure",
    "export_relaxed_structures_to_folder",
    "open_containing_folder",
    "report_export_failed",
    "report_export_saved",
]
