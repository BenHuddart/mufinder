"""Validated decimal text input for non-integer numeric fields."""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QLineEdit, QWidget


class DecimalLineEdit(QLineEdit):
    """Line edit with float validation and a QDoubleSpinBox-like API subset."""

    valueChanged = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._minimum = -1_000_000_000.0
        self._maximum = 1_000_000_000.0
        self._decimals = 8
        self._single_step = 0.1
        self._last_value = 0.0
        self._validator = QDoubleValidator(self._minimum, self._maximum, self._decimals, self)
        self._validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self.setValidator(self._validator)
        self.textChanged.connect(self._on_text_changed)
        self.editingFinished.connect(self._snap_to_range)
        self.setValue(0.0)

    def setRange(self, minimum: float, maximum: float) -> None:  # noqa: N802 - Qt API compatibility.
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._validator.setBottom(self._minimum)
        self._validator.setTop(self._maximum)
        self.setValue(self.value())

    def setDecimals(self, decimals: int) -> None:  # noqa: N802 - Qt API compatibility.
        self._decimals = max(0, int(decimals))
        self._validator.setDecimals(self._decimals)
        self.setValue(self.value())

    def setSingleStep(self, step: float) -> None:  # noqa: N802 - Qt API compatibility.
        self._single_step = float(step)

    def setValue(self, value: float) -> None:  # noqa: N802 - Qt API compatibility.
        numeric = self._clamp(float(value))
        text = self._format_value(numeric)
        if self.text() == text:
            self._last_value = numeric
            return
        self.setText(text)
        self._last_value = numeric

    def value(self) -> float:
        parsed = self._parse(self.text())
        if parsed is None:
            return float(self._last_value)
        return self._clamp(parsed)

    def _on_text_changed(self, _text: str) -> None:
        parsed = self._parse(self.text())
        if parsed is None:
            return
        value = self._clamp(parsed)
        tolerance = 10 ** (-(self._decimals + 2))
        if math.isclose(value, self._last_value, rel_tol=0.0, abs_tol=tolerance):
            return
        self._last_value = value
        self.valueChanged.emit(value)

    def _snap_to_range(self) -> None:
        """Show the clamped value when an out-of-range number is left in the field.

        ``value()`` already clamps, so without this the field could display, say,
        ``0.0001`` while every reader used the minimum ``0.001``. Like a
        ``QDoubleSpinBox``, the text is corrected when editing finishes. Blank or
        unparsable text is left alone: some callers treat a blank field as unset.
        """

        parsed = self._parse(self.text())
        if parsed is None or self._clamp(parsed) == parsed:
            return
        self.setValue(parsed)

    def _parse(self, text: str) -> float | None:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None

    def _clamp(self, value: float) -> float:
        return max(self._minimum, min(self._maximum, value))

    def _format_value(self, value: float) -> str:
        if self._decimals <= 0:
            return str(int(round(value)))
        formatted = f"{value:.{self._decimals}f}".rstrip("0").rstrip(".")
        if formatted in {"", "-"}:
            return "0"
        return formatted
