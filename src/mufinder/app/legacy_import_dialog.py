"""Review dialog for a scanned legacy MuFinder folder.

Shows every relaxation the scan found, why any of them cannot be imported, and
what the scan had to resolve on the user's behalf — then lets the user choose
which to bring in.

The dialog is deliberately a *review* step rather than a confirmation. An import
writes entities and files that are then read-only, and it resolves genuine
ambiguities along the way: a stale local input tree disagreeing with the output,
a charge state that could not be determined, a muon labelled with a bare ``H``.
Those resolutions are defensible but they are the user's science, so they are
surfaced before anything is committed rather than reported afterwards.

Scanning happens on a worker thread (see
:class:`~mufinder.app.legacy_import_dialog.LegacyScanWorker`): a folder can hold
dozens of multi-megabyte outputs, and parsing them on the GUI thread would
freeze the window.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from mufinder.app.theme import style_section_subtitle, style_section_title
from mufinder.domain.charge_states import muon_charge_state_label
from mufinder.services.legacy_import import DetectedRun, LegacyFolderScan, scan_legacy_folder


class LegacyScanWorker(QObject):
    """Scan a legacy folder off the GUI thread."""

    completed = Signal(object)
    failed = Signal(str)
    #: ``(done, total)`` as each run is examined, so a folder of large outputs
    #: reports progress instead of sitting on a static message.
    progressed = Signal(int, int)

    def __init__(
        self,
        root: Path,
        host_lattice: list[list[float]] | None,
        host_species: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._root = root
        self._host_lattice = host_lattice
        self._host_species = host_species

    @Slot()
    def run(self) -> None:
        try:
            scan = scan_legacy_folder(
                self._root,
                host_lattice=self._host_lattice,
                host_species=self._host_species,
                progress=self.progressed.emit,
            )
        except Exception as exc:  # noqa: BLE001 - reported to the user, never raised across threads
            self.failed.emit(str(exc))
            return
        self.completed.emit(scan)


def start_legacy_scan(
    parent: QObject,
    root: Path,
    host_lattice: list[list[float]] | None,
    host_species: list[str] | None,
    on_completed,
    on_failed,
    on_progress=None,
) -> QThread:
    """Run :func:`scan_legacy_folder` on a worker thread.

    Returns the started thread so the caller can keep it alive. Receivers must be
    bound methods or slots on a GUI-thread object, so the signals deliver as
    queued connections.
    """

    worker = LegacyScanWorker(root, host_lattice, host_species)
    thread = QThread(parent)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.completed.connect(on_completed)
    worker.failed.connect(on_failed)
    if on_progress is not None:
        worker.progressed.connect(on_progress)
    worker.completed.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.start()
    return thread


def _status_text(run: DetectedRun) -> str:
    if not run.importable:
        return "cannot import"
    if run.converged is True:
        return "converged"
    if run.converged is False:
        return "not converged"
    return "completed"


def run_summary_lines(scan: LegacyFolderScan) -> list[str]:
    """Return the folder-level notes worth showing above the table."""

    lines = list(scan.warnings)
    conflicts = sum(len(run.conflicts) for run in scan.runs)
    if conflicts:
        lines.append(
            f"{conflicts} parameter disagreement(s) between the run outputs and the "
            "local input files. The value the calculation actually used will be "
            "imported; see the Notes column."
        )
    undetermined = [run.site_index for run in scan.runs if run.charge_state is None]
    if undetermined:
        lines.append(
            "Charge state could not be determined for site(s) "
            f"{', '.join(str(index) for index in undetermined)}; they will be imported "
            "as positive muons."
        )
    return lines


def run_notes(run: DetectedRun) -> str:
    """Return the per-run notes shown in the table and its tooltip."""

    notes = [conflict.describe() for conflict in run.conflicts]
    notes.extend(run.warnings)
    notes.extend(run.blockers)
    return " ".join(notes)


class LegacyImportDialog(QDialog):
    """Choose which detected legacy relaxations to import."""

    _SELECT_COLUMN = 0
    _SITE_COLUMN = 1
    _CHARGE_COLUMN = 2
    _STATUS_COLUMN = 3
    _SUPERCELL_COLUMN = 4
    _ENERGY_COLUMN = 5
    _NOTES_COLUMN = 6

    def __init__(self, scan: LegacyFolderScan, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Legacy Relaxations")
        self.resize(940, 560)
        self._scan = scan

        layout = QVBoxLayout(self)

        title = QLabel("Import Legacy Relaxations")
        style_section_title(title)
        layout.addWidget(title)

        subtitle = QLabel(
            f"Found {len(scan.runs)} relaxation(s) in {scan.root.name}. "
            "Imported relaxations are read-only: they record calculations run "
            "outside MuFinder."
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(subtitle)

        notes = run_summary_lines(scan)
        if notes:
            summary = QTextEdit()
            summary.setReadOnly(True)
            summary.setPlainText("\n".join(f"- {line}" for line in notes))
            summary.setMaximumHeight(96)
            layout.addWidget(summary)

        self._table = QTableWidget(0, 7, self)
        self._table.setHorizontalHeaderLabels(
            ["Import", "Site", "Charge", "Status", "Supercell", "Energy (eV)", "Notes"]
        )
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self._NOTES_COLUMN, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, 1)

        buttons_row = QHBoxLayout()
        self._select_converged = QPushButton("Select converged")
        self._select_converged.clicked.connect(self._on_select_converged)
        self._select_none = QPushButton("Select none")
        self._select_none.clicked.connect(self._on_select_none)
        buttons_row.addWidget(self._select_converged)
        buttons_row.addWidget(self._select_none)
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._populate()
        self._refresh_ok_state()

    def _populate(self) -> None:
        self._table.setRowCount(len(self._scan.runs))
        for row, run in enumerate(self._scan.runs):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            importable = run.importable
            if not importable:
                # A run that cannot be imported is shown rather than hidden, so
                # the user can see why it is absent instead of wondering.
                check.setFlags(Qt.ItemFlag.NoItemFlags)
            check.setCheckState(
                Qt.CheckState.Checked
                if (importable and run.is_converged)
                else Qt.CheckState.Unchecked
            )
            self._table.setItem(row, self._SELECT_COLUMN, check)

            self._table.setItem(row, self._SITE_COLUMN, QTableWidgetItem(str(run.site_index)))
            charge = (
                muon_charge_state_label(run.charge_state) if run.charge_state else "undetermined"
            )
            self._table.setItem(row, self._CHARGE_COLUMN, QTableWidgetItem(charge))
            self._table.setItem(row, self._STATUS_COLUMN, QTableWidgetItem(_status_text(run)))

            matrix = run.supercell_matrix
            supercell = (
                "unit cell"
                if matrix is None
                else f"{matrix[0][0]}x{matrix[1][1]}x{matrix[2][2]}"
                if _is_diagonal(matrix)
                else "custom"
            )
            self._table.setItem(row, self._SUPERCELL_COLUMN, QTableWidgetItem(supercell))
            energy = "" if run.total_energy is None else f"{run.total_energy:.4f}"
            self._table.setItem(row, self._ENERGY_COLUMN, QTableWidgetItem(energy))

            note_text = run_notes(run)
            note_item = QTableWidgetItem(note_text)
            note_item.setToolTip(note_text)
            self._table.setItem(row, self._NOTES_COLUMN, note_item)

        self._table.itemChanged.connect(self._on_item_changed)
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(
            self._NOTES_COLUMN, QHeaderView.ResizeMode.Stretch
        )

    @Slot()
    def _on_select_converged(self) -> None:
        self._set_all(lambda run: run.importable and run.is_converged)

    @Slot()
    def _on_select_none(self) -> None:
        self._set_all(lambda _run: False)

    def _set_all(self, predicate) -> None:
        for row, run in enumerate(self._scan.runs):
            item = self._table.item(row, self._SELECT_COLUMN)
            if item is None or not run.importable:
                continue
            item.setCheckState(
                Qt.CheckState.Checked if predicate(run) else Qt.CheckState.Unchecked
            )

    def _on_item_changed(self, _item: QTableWidgetItem) -> None:
        self._refresh_ok_state()

    def _refresh_ok_state(self) -> None:
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(bool(self.selected_runs()))

    def selected_runs(self) -> list[DetectedRun]:
        """Return the runs the user ticked."""

        selected: list[DetectedRun] = []
        for row, run in enumerate(self._scan.runs):
            item = self._table.item(row, self._SELECT_COLUMN)
            if item is None or not run.importable:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(run)
        return selected


def _is_diagonal(matrix) -> bool:
    return all(
        matrix[i][j] == 0
        for i in range(3)
        for j in range(3)
        if i != j
    )


__all__ = [
    "LegacyImportDialog",
    "LegacyScanWorker",
    "run_notes",
    "run_summary_lines",
    "start_legacy_scan",
]
