"""Muon candidate-site generation panel."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.theme import style_section_subtitle, style_section_title


class MuonGenerationWidget(QWidget):
    """Form-based UI for candidate muon-site generation."""

    generate_requested = Signal(str, dict)
    form_state_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["random", "near_element", "manual"])

        self._param_stack = QStackedWidget()

        self._random_count = QSpinBox()
        self._random_count.setRange(1, 2000)
        self._random_count.setValue(24)

        self._random_min_muon_muon = DecimalLineEdit()
        self._random_min_muon_muon.setRange(0.1, 10.0)
        self._random_min_muon_muon.setSingleStep(0.1)
        self._random_min_muon_muon.setValue(0.8)

        self._random_min_muon_atom = DecimalLineEdit()
        self._random_min_muon_atom.setRange(0.1, 10.0)
        self._random_min_muon_atom.setSingleStep(0.1)
        self._random_min_muon_atom.setValue(1.0)

        self._random_max_attempts = QSpinBox()
        self._random_max_attempts.setRange(100, 1_000_000)
        self._random_max_attempts.setSingleStep(100)
        self._random_max_attempts.setValue(20000)

        self._random_seed = QLineEdit("")
        self._random_seed.setPlaceholderText("Optional integer")

        self._near_element = QLineEdit("O")
        self._near_count = QSpinBox()
        self._near_count.setRange(1, 2000)
        self._near_count.setValue(12)

        self._near_shell_min = DecimalLineEdit()
        self._near_shell_min.setRange(0.1, 10.0)
        self._near_shell_min.setSingleStep(0.1)
        self._near_shell_min.setValue(0.9)

        self._near_shell_max = DecimalLineEdit()
        self._near_shell_max.setRange(0.1, 10.0)
        self._near_shell_max.setSingleStep(0.1)
        self._near_shell_max.setValue(1.4)

        self._near_min_muon_muon = DecimalLineEdit()
        self._near_min_muon_muon.setRange(0.1, 10.0)
        self._near_min_muon_muon.setSingleStep(0.1)
        self._near_min_muon_muon.setValue(0.8)

        self._near_min_muon_atom = DecimalLineEdit()
        self._near_min_muon_atom.setRange(0.1, 10.0)
        self._near_min_muon_atom.setSingleStep(0.1)
        self._near_min_muon_atom.setValue(1.0)

        self._near_max_attempts = QSpinBox()
        self._near_max_attempts.setRange(100, 1_000_000)
        self._near_max_attempts.setSingleStep(100)
        self._near_max_attempts.setValue(30000)

        self._near_seed = QLineEdit("")
        self._near_seed.setPlaceholderText("Optional integer")

        self._manual_coord_system = QComboBox()
        self._manual_coord_system.addItems(["fractional", "cartesian"])
        self._manual_x = DecimalLineEdit()
        self._manual_y = DecimalLineEdit()
        self._manual_z = DecimalLineEdit()
        for control in (self._manual_x, self._manual_y, self._manual_z):
            control.setRange(-10000.0, 10000.0)
            control.setDecimals(6)

        self._site_table = QTableWidget(0, 6)
        self._site_table.setHorizontalHeaderLabels(
            ["Label", "Method", "Status", "fX", "fY", "fZ"]
        )

        self._build_layout()
        self._connect_signals()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel("Muon Site Generation")
        style_section_title(header)
        layout.addWidget(header)

        subtitle = QLabel("Create candidate muon positions using guided generation methods.")
        style_section_subtitle(subtitle)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        form_box = QGroupBox("Generation Setup")
        form_layout = QFormLayout(form_box)
        form_layout.addRow("Method", self._method_combo)
        form_layout.addRow("Parameters", self._param_stack)

        random_page = QWidget()
        random_layout = QFormLayout(random_page)
        random_layout.addRow("Site count", self._random_count)
        random_layout.addRow("Min muon-muon distance", self._random_min_muon_muon)
        random_layout.addRow("Min muon-atom distance", self._random_min_muon_atom)
        random_layout.addRow("Max attempts", self._random_max_attempts)
        random_layout.addRow("Seed", self._random_seed)

        near_page = QWidget()
        near_layout = QFormLayout(near_page)
        near_layout.addRow("Element", self._near_element)
        near_layout.addRow("Site count", self._near_count)
        near_layout.addRow("Shell min (A)", self._near_shell_min)
        near_layout.addRow("Shell max (A)", self._near_shell_max)
        near_layout.addRow("Min muon-muon distance", self._near_min_muon_muon)
        near_layout.addRow("Min muon-atom distance", self._near_min_muon_atom)
        near_layout.addRow("Max attempts", self._near_max_attempts)
        near_layout.addRow("Seed", self._near_seed)

        manual_page = QWidget()
        manual_layout = QFormLayout(manual_page)
        manual_layout.addRow("Coordinate system", self._manual_coord_system)
        manual_layout.addRow("X", self._manual_x)
        manual_layout.addRow("Y", self._manual_y)
        manual_layout.addRow("Z", self._manual_z)

        self._param_stack.addWidget(random_page)
        self._param_stack.addWidget(near_page)
        self._param_stack.addWidget(manual_page)

        action_row = QHBoxLayout()
        generate_btn = QPushButton("Generate Candidates")
        generate_btn.clicked.connect(self._emit_generate)
        action_row.addWidget(generate_btn)
        action_row.addStretch(1)

        layout.addWidget(form_box)
        layout.addLayout(action_row)
        layout.addWidget(self._site_table)

    def _connect_signals(self) -> None:
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._method_combo.currentIndexChanged.connect(self._emit_state)

        self._random_count.valueChanged.connect(self._emit_state)
        self._random_min_muon_muon.valueChanged.connect(self._emit_state)
        self._random_min_muon_atom.valueChanged.connect(self._emit_state)
        self._random_max_attempts.valueChanged.connect(self._emit_state)
        self._random_seed.textChanged.connect(self._emit_state)

        self._near_element.textChanged.connect(self._emit_state)
        self._near_count.valueChanged.connect(self._emit_state)
        self._near_shell_min.valueChanged.connect(self._emit_state)
        self._near_shell_max.valueChanged.connect(self._emit_state)
        self._near_min_muon_muon.valueChanged.connect(self._emit_state)
        self._near_min_muon_atom.valueChanged.connect(self._emit_state)
        self._near_max_attempts.valueChanged.connect(self._emit_state)
        self._near_seed.textChanged.connect(self._emit_state)

        self._manual_coord_system.currentIndexChanged.connect(self._emit_state)
        self._manual_x.valueChanged.connect(self._emit_state)
        self._manual_y.valueChanged.connect(self._emit_state)
        self._manual_z.valueChanged.connect(self._emit_state)

    def _on_method_changed(self) -> None:
        self._param_stack.setCurrentIndex(self._method_combo.currentIndex())

    def _emit_generate(self) -> None:
        method = self._method_combo.currentText()
        payload = self._collect_method_parameters(method)
        self.generate_requested.emit(method, payload)

    def _collect_method_parameters(self, method: str) -> dict[str, Any]:
        if method == "random":
            return {
                "count": self._random_count.value(),
                "min_muon_muon": self._random_min_muon_muon.value(),
                "min_muon_atom": self._random_min_muon_atom.value(),
                "max_attempts": self._random_max_attempts.value(),
                "seed": self._random_seed.text().strip(),
            }
        if method == "near_element":
            return {
                "element": self._near_element.text().strip(),
                "count": self._near_count.value(),
                "shell_min": self._near_shell_min.value(),
                "shell_max": self._near_shell_max.value(),
                "min_muon_muon": self._near_min_muon_muon.value(),
                "min_muon_atom": self._near_min_muon_atom.value(),
                "max_attempts": self._near_max_attempts.value(),
                "seed": self._near_seed.text().strip(),
            }
        return {
            "coord_system": self._manual_coord_system.currentText(),
            "x": self._manual_x.value(),
            "y": self._manual_y.value(),
            "z": self._manual_z.value(),
        }

    def _emit_state(self) -> None:
        self.form_state_changed.emit(self.collect_state())

    def collect_state(self) -> dict[str, Any]:
        """Collect all form values for project persistence."""
        return {
            "method": self._method_combo.currentText(),
            "random": {
                "count": self._random_count.value(),
                "min_muon_muon": self._random_min_muon_muon.value(),
                "min_muon_atom": self._random_min_muon_atom.value(),
                "max_attempts": self._random_max_attempts.value(),
                "seed": self._random_seed.text(),
            },
            "near_element": {
                "element": self._near_element.text(),
                "count": self._near_count.value(),
                "shell_min": self._near_shell_min.value(),
                "shell_max": self._near_shell_max.value(),
                "min_muon_muon": self._near_min_muon_muon.value(),
                "min_muon_atom": self._near_min_muon_atom.value(),
                "max_attempts": self._near_max_attempts.value(),
                "seed": self._near_seed.text(),
            },
            "manual": {
                "coord_system": self._manual_coord_system.currentText(),
                "x": self._manual_x.value(),
                "y": self._manual_y.value(),
                "z": self._manual_z.value(),
            },
        }

    def set_form_state(self, state: dict[str, Any]) -> None:
        """Apply saved form values when a project is loaded."""
        if not state:
            return

        random_state = state.get("random", {})
        near_state = state.get("near_element", {})
        manual_state = state.get("manual", {})

        self._random_count.setValue(int(random_state.get("count", self._random_count.value())))
        self._random_min_muon_muon.setValue(
            float(random_state.get("min_muon_muon", self._random_min_muon_muon.value()))
        )
        self._random_min_muon_atom.setValue(
            float(random_state.get("min_muon_atom", self._random_min_muon_atom.value()))
        )
        self._random_max_attempts.setValue(
            int(random_state.get("max_attempts", self._random_max_attempts.value()))
        )
        self._random_seed.setText(str(random_state.get("seed", self._random_seed.text())))

        self._near_element.setText(str(near_state.get("element", self._near_element.text())))
        self._near_count.setValue(int(near_state.get("count", self._near_count.value())))
        self._near_shell_min.setValue(float(near_state.get("shell_min", self._near_shell_min.value())))
        self._near_shell_max.setValue(float(near_state.get("shell_max", self._near_shell_max.value())))
        self._near_min_muon_muon.setValue(
            float(near_state.get("min_muon_muon", self._near_min_muon_muon.value()))
        )
        self._near_min_muon_atom.setValue(
            float(near_state.get("min_muon_atom", self._near_min_muon_atom.value()))
        )
        self._near_max_attempts.setValue(int(near_state.get("max_attempts", self._near_max_attempts.value())))
        self._near_seed.setText(str(near_state.get("seed", self._near_seed.text())))

        self._manual_coord_system.setCurrentText(
            str(manual_state.get("coord_system", self._manual_coord_system.currentText()))
        )
        self._manual_x.setValue(float(manual_state.get("x", self._manual_x.value())))
        self._manual_y.setValue(float(manual_state.get("y", self._manual_y.value())))
        self._manual_z.setValue(float(manual_state.get("z", self._manual_z.value())))

        method = str(state.get("method", "random"))
        self._method_combo.setCurrentText(method)
        self._on_method_changed()

    def set_sites(self, rows: list[dict[str, Any]]) -> None:
        """Render candidate sites in the table."""
        self._site_table.setRowCount(0)
        for row_data in rows:
            row = self._site_table.rowCount()
            self._site_table.insertRow(row)
            self._site_table.setItem(row, 0, QTableWidgetItem(str(row_data.get("label", ""))))
            self._site_table.setItem(row, 1, QTableWidgetItem(str(row_data.get("method", ""))))
            self._site_table.setItem(row, 2, QTableWidgetItem(str(row_data.get("status", ""))))
            frac = row_data.get("fractional", [0.0, 0.0, 0.0])
            self._site_table.setItem(row, 3, QTableWidgetItem(f"{float(frac[0]):.4f}"))
            self._site_table.setItem(row, 4, QTableWidgetItem(f"{float(frac[1]):.4f}"))
            self._site_table.setItem(row, 5, QTableWidgetItem(f"{float(frac[2]):.4f}"))
