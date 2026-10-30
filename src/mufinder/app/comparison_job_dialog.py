"""Dialog for selecting completed pristine-relaxation jobs for comparison."""

from __future__ import annotations

from uuid import UUID

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTableWidgetItem,
    QVBoxLayout,
)

from mufinder.app.workflow_table_utils import create_workflow_table
from mufinder.services.source_candidates import CompletedPristineJob


class ComparisonJobDialog(QDialog):
    """Allow users to select one or more completed pristine jobs for comparison."""

    _CHECK_COLUMN = 0
    _LABEL_COLUMN = 1
    _SUPERCELL_COLUMN = 2
    _STATUS_COLUMN = 3
    _ENERGY_COLUMN = 4
    _A_COLUMN = 5
    _B_COLUMN = 6
    _C_COLUMN = 7

    def __init__(
        self,
        *,
        jobs: list[CompletedPristineJob],
        default_label: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate Comparison Table")
        self.resize(980, 460)

        self._jobs = list(jobs)
        self._label_input = QLineEdit(default_label)
        # Preserves the original Qt defaults this table relied on: default
        # (non-row) selection behavior and standard edit triggers (only the
        # checkbox column is actually interacted with, but other cells were
        # always incidentally editable). Columns are sized by
        # resizeColumnsToContents() in _populate_rows.
        self._table = create_workflow_table(
            (
                "Include",
                "Label",
                "Supercell",
                "Status",
                "Total Energy (eV)",
                "a (Ang)",
                "b (Ang)",
                "c (Ang)",
            ),
            selection_behavior=QAbstractItemView.SelectionBehavior.SelectItems,
            editable=True,
            default_resize_mode=QHeaderView.ResizeMode.Interactive,
            stretch_last_section=False,
            accessible_name="Comparison Jobs",
        )

        self._build_layout()
        self._populate_rows()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Select completed pristine-relaxation jobs to compare against the experimental host structure."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        label_row = QHBoxLayout()
        label_caption = QLabel("Comparison table label")
        label_caption.setBuddy(self._label_input)
        label_row.addWidget(label_caption)
        label_row.addWidget(self._label_input, 1)
        layout.addLayout(label_row)

        layout.addWidget(self._table, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_rows(self) -> None:
        self._table.setRowCount(0)
        for job in self._jobs:
            row = self._table.rowCount()
            self._table.insertRow(row)

            include = QTableWidgetItem("")
            include.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
            )
            include.setCheckState(Qt.CheckState.Unchecked)
            include.setData(Qt.ItemDataRole.UserRole, str(job.job_id))
            self._table.setItem(row, self._CHECK_COLUMN, include)

            self._table.setItem(row, self._LABEL_COLUMN, QTableWidgetItem(job.job_label))
            self._table.setItem(row, self._SUPERCELL_COLUMN, QTableWidgetItem(job.supercell_label))
            self._table.setItem(row, self._STATUS_COLUMN, QTableWidgetItem(job.status))
            self._table.setItem(
                row,
                self._ENERGY_COLUMN,
                QTableWidgetItem(
                    "" if job.total_energy is None else f"{float(job.total_energy):.8g}"
                ),
            )
            self._table.setItem(
                row,
                self._A_COLUMN,
                QTableWidgetItem("" if job.lattice_a is None else f"{float(job.lattice_a):.8g}"),
            )
            self._table.setItem(
                row,
                self._B_COLUMN,
                QTableWidgetItem("" if job.lattice_b is None else f"{float(job.lattice_b):.8g}"),
            )
            self._table.setItem(
                row,
                self._C_COLUMN,
                QTableWidgetItem("" if job.lattice_c is None else f"{float(job.lattice_c):.8g}"),
            )

        self._table.resizeColumnsToContents()

    def _accept_if_valid(self) -> None:
        if not self.selected_job_ids():
            QMessageBox.warning(self, "No jobs selected", "Select at least one completed job.")
            return
        if not self.table_label():
            QMessageBox.warning(
                self, "Missing label", "Provide a non-empty comparison table label."
            )
            return
        self.accept()

    def table_label(self) -> str:
        return self._label_input.text().strip()

    def selected_job_ids(self) -> list[UUID]:
        selected: list[UUID] = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self._CHECK_COLUMN)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            raw_job_id = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(raw_job_id, str):
                continue
            try:
                selected.append(UUID(raw_job_id))
            except ValueError:
                continue
        return selected
