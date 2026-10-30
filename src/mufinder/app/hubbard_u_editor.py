"""Hubbard U table helpers and dialog."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.theme import style_section_title, track_item_view_corners
from mufinder.castep_io.types import BlockSection
from mufinder.domain.text import plural

ORBITALS: tuple[str, ...] = ("s", "p", "d", "f")
HUBBARD_UNITS: tuple[str, ...] = ("eV", "Ha")


@dataclass(frozen=True, slots=True)
class HubbardUEntry:
    """One target row in a CASTEP HUBBARD_U block."""

    species: str
    orbital: str
    value: float


def summarize_hubbard_u(value: BlockSection | None) -> str:
    """Return a compact summary for a HUBBARD_U block."""

    if value is None:
        return "Not set"
    try:
        entries = hubbard_entries_from_block(value)
    except ValueError:
        return plural(len(value.rows), "row")
    unit = value.unit or "eV"
    if not entries:
        return "Not set"
    entry_noun = "entry" if len(entries) == 1 else "entries"
    return f"{len(entries)} {entry_noun} ({unit})"


def hubbard_block_from_entries(
    entries: tuple[HubbardUEntry, ...], *, unit: str = "eV"
) -> BlockSection | None:
    """Build a normalized HUBBARD_U block from dialog entries."""

    rows: list[tuple[object, ...]] = []
    for entry in entries:
        species = entry.species.strip()
        if not species:
            continue
        orbital = entry.orbital.strip().lower()
        value = float(entry.value)
        if orbital not in ORBITALS or not isfinite(value):
            continue
        rows.append((species, f"{orbital}:", value))

    if not rows:
        return None
    return BlockSection(name="hubbard_u", rows=tuple(rows), unit=unit.strip() or "eV")


def hubbard_entries_from_block(value: BlockSection | None) -> tuple[HubbardUEntry, ...]:
    """Parse a HUBBARD_U block into rich table rows."""

    if value is None:
        return ()
    entries: list[HubbardUEntry] = []
    for row_index, row in enumerate(value.rows, start=1):
        if not row:
            continue
        species = str(row[0]).strip()
        if not species:
            raise ValueError(f"HUBBARD_U row {row_index} is missing a species.")

        token_index = 1
        if len(row) > 1 and ":" not in str(row[1]):
            _parse_atom_index(row[1], row_index=row_index)
            token_index = 2

        orbitals = _parse_orbital_values(row[token_index:], row_index=row_index)
        for orbital, orbital_value in orbitals.items():
            entries.append(HubbardUEntry(species=species, orbital=orbital, value=orbital_value))
    return tuple(entries)


def _parse_atom_index(value: object, *, row_index: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"HUBBARD_U row {row_index} has an invalid atom index '{value}'.") from exc
    if parsed <= 0:
        raise ValueError(f"HUBBARD_U row {row_index} atom index must be greater than zero.")
    return parsed


def _parse_orbital_values(tokens: tuple[object, ...], *, row_index: int) -> dict[str, float]:
    orbitals: dict[str, float] = {}
    index = 0
    while index < len(tokens):
        token = str(tokens[index]).strip()
        if ":" not in token:
            raise ValueError(f"HUBBARD_U row {row_index} expected an orbital token such as 'd:'.")
        orbital, tail = token.split(":", maxsplit=1)
        orbital = orbital.strip().lower()
        if orbital not in ORBITALS:
            raise ValueError(f"HUBBARD_U row {row_index} has unsupported orbital '{orbital}'.")
        if tail.strip():
            raw_value = tail.strip()
        else:
            index += 1
            if index >= len(tokens):
                raise ValueError(
                    f"HUBBARD_U row {row_index} is missing a value for orbital '{orbital}'."
                )
            raw_value = str(tokens[index]).strip()
        try:
            orbitals[orbital] = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"HUBBARD_U row {row_index} has an invalid U value '{raw_value}'."
            ) from exc
        index += 1
    if not orbitals:
        raise ValueError(f"HUBBARD_U row {row_index} must define at least one orbital U value.")
    return orbitals


class HubbardUDialog(QDialog):
    """Structured editor for CASTEP HUBBARD_U blocks."""

    _SPECIES_COLUMN = 0
    _ORBITAL_COLUMN = 1
    _VALUE_COLUMN = 2

    def __init__(
        self,
        *,
        initial_value: BlockSection | None = None,
        species_options: Sequence[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hubbard U")
        self.resize(720, 420)
        self._species_options = _normalize_species_options(species_options)

        root = QVBoxLayout(self)

        title = QLabel("Hubbard U", self)
        style_section_title(title)
        root.addWidget(title)

        summary = QLabel(
            "Choose a species from the structure, select the orbital, then enter the Hubbard U in the chosen units.",
            self,
        )
        summary.setWordWrap(True)
        root.addWidget(summary)

        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Units", self))
        self._unit_combo = QComboBox(self)
        self._unit_combo.addItems(HUBBARD_UNITS)
        unit_row.addWidget(self._unit_combo)
        unit_row.addStretch(1)
        root.addLayout(unit_row)

        self._table = QTableWidget(self)
        track_item_view_corners(self._table)
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(("Species", "Orbital", "U"))
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        root.addWidget(self._table, 1)

        controls = QHBoxLayout()
        self._add_button = QPushButton("Add Row", self)
        self._duplicate_button = QPushButton("Duplicate Selected", self)
        self._remove_button = QPushButton("Remove Selected", self)
        self._add_button.clicked.connect(self._add_blank_row)
        self._duplicate_button.clicked.connect(self._duplicate_selected)
        self._remove_button.clicked.connect(self._remove_selected)
        controls.addWidget(self._add_button)
        controls.addWidget(self._duplicate_button)
        controls.addWidget(self._remove_button)
        controls.addStretch(1)
        root.addLayout(controls)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        if initial_value is not None and initial_value.unit:
            self._unit_combo.setCurrentText(initial_value.unit)
        for entry in hubbard_entries_from_block(initial_value):
            self._append_entry(entry)
        if self._table.rowCount() == 0:
            self._add_blank_row()
        self._refresh_controls()

    def block_value(self) -> BlockSection | None:
        return hubbard_block_from_entries(self.entries(), unit=self._unit_combo.currentText())

    def entries(self) -> tuple[HubbardUEntry, ...]:
        entries: list[HubbardUEntry] = []
        for row in range(self._table.rowCount()):
            entries.append(self._entry_for_row(row, require_value=True))
        return tuple(entries)

    def _add_blank_row(self) -> None:
        default_species = self._species_options[0] if self._species_options else ""
        self._append_entry(HubbardUEntry(species=default_species, orbital="d", value=float("nan")))
        self._refresh_controls()

    def _append_entry(self, entry: HubbardUEntry) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        species = QComboBox(self._table)
        species.setEditable(not self._species_options)
        species.addItems(self._species_options)
        if entry.species and species.findText(entry.species) < 0:
            species.addItem(entry.species)
        if entry.species:
            species.setCurrentText(entry.species)
        self._table.setCellWidget(row, self._SPECIES_COLUMN, species)

        orbital = QComboBox(self._table)
        orbital.addItems(ORBITALS)
        if entry.orbital:
            orbital.setCurrentText(entry.orbital.strip().lower())
        self._table.setCellWidget(row, self._ORBITAL_COLUMN, orbital)

        editor = QLineEdit(self._table)
        editor.setPlaceholderText("e.g. 4.0")
        validator = QDoubleValidator(-1_000_000_000.0, 1_000_000_000.0, 8, editor)
        validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        editor.setValidator(validator)
        if isfinite(float(entry.value)):
            editor.setText(f"{entry.value:.12g}")
        self._table.setCellWidget(row, self._VALUE_COLUMN, editor)
        self._table.resizeColumnsToContents()

    def _duplicate_selected(self) -> None:
        rows = sorted({index.row() for index in self._table.selectedIndexes()})
        for row in rows:
            self._append_entry(self._entry_for_row(row, require_value=False))
        self._refresh_controls()

    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self._table.selectedIndexes()}, reverse=True)
        for row in rows:
            self._table.removeRow(row)
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        self._remove_button.setEnabled(self._table.rowCount() > 0)
        self._duplicate_button.setEnabled(self._table.rowCount() > 0)

    def _accept_if_valid(self) -> None:
        try:
            entries = self.entries()
            seen_targets: set[tuple[str, str]] = set()
            for index, entry in enumerate(entries, start=1):
                if not entry.species.strip():
                    raise ValueError(f"Row {index} requires a species.")
                if entry.orbital not in ORBITALS:
                    raise ValueError(f"Row {index} has an unsupported orbital.")
                target_key = (entry.species.strip().casefold(), entry.orbital)
                if target_key in seen_targets:
                    raise ValueError(f"Row {index} duplicates an existing species/orbital target.")
                seen_targets.add(target_key)
            block = hubbard_block_from_entries(entries, unit=self._unit_combo.currentText())
            if block is None:
                raise ValueError("Add at least one Hubbard U row.")
            hubbard_entries_from_block(block)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Hubbard U table", str(exc))
            return
        self.accept()

    def _entry_for_row(self, row: int, *, require_value: bool) -> HubbardUEntry:
        species_widget = self._table.cellWidget(row, self._SPECIES_COLUMN)
        species = (
            species_widget.currentText().strip() if isinstance(species_widget, QComboBox) else ""
        )

        orbital_widget = self._table.cellWidget(row, self._ORBITAL_COLUMN)
        orbital = (
            orbital_widget.currentText().strip().lower()
            if isinstance(orbital_widget, QComboBox)
            else ""
        )

        value_widget = self._table.cellWidget(row, self._VALUE_COLUMN)
        text = value_widget.text().strip() if isinstance(value_widget, QLineEdit) else ""
        if not text:
            if require_value:
                raise ValueError(f"Row {row + 1} requires a Hubbard U value.")
            value = float("nan")
        else:
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(f"Row {row + 1} has an invalid Hubbard U value '{text}'.") from exc
        return HubbardUEntry(species=species, orbital=orbital, value=value)


def _normalize_species_options(species_options: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for option in species_options:
        species = str(option).strip()
        if not species or species in seen:
            continue
        normalized.append(species)
        seen.add(species)
    return tuple(normalized)
