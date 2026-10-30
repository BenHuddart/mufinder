"""Validated decimal text input for non-integer numeric fields."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QWidget

from mufinder.app.theme import style_inline_group

_AXES = ("x", "y", "z")


class DecimalLineEdit(QLineEdit):
    """Line edit with float validation and a QDoubleSpinBox-like API subset.

    The keyword arguments set it up in one call. ``text`` is shown exactly as
    given (``"1.0"`` stays ``"1.0"``, ``""`` stays blank) rather than
    reformatted, for forms that read the typed text. ``standard_notation``
    refuses exponents such as ``1e-3``; by default they are accepted.
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        text: str | None = None,
        minimum: float = -1_000_000_000.0,
        maximum: float = 1_000_000_000.0,
        decimals: int = 8,
        standard_notation: bool = False,
        placeholder: str = "",
    ) -> None:
        super().__init__(parent)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._decimals = max(0, int(decimals))
        self._single_step = 0.1
        self._last_value = 0.0
        self._validator = QDoubleValidator(self._minimum, self._maximum, self._decimals, self)
        self._validator.setNotation(
            QDoubleValidator.Notation.StandardNotation
            if standard_notation
            else QDoubleValidator.Notation.ScientificNotation
        )
        self.setValidator(self._validator)
        self.textChanged.connect(self._on_text_changed)
        self.editingFinished.connect(self._snap_to_range)
        if placeholder:
            self.setPlaceholderText(placeholder)
        if text is None:
            self.setValue(0.0)
        else:
            self.setText(text)

    def minimum(self) -> float:
        return self._minimum

    def maximum(self) -> float:
        return self._maximum

    def setRange(self, minimum: float, maximum: float, *, keep_text: bool = False) -> None:  # noqa: N802 - Qt API compatibility.
        """Accept values from ``minimum`` to ``maximum``.

        The shown value is clamped into the new range, as a ``QDoubleSpinBox``
        does, unless ``keep_text`` leaves the text as typed.
        """

        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._validator.setBottom(self._minimum)
        self._validator.setTop(self._maximum)
        if not keep_text:
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


class VectorInput(QWidget):
    """Three decimal boxes for a vector's x, y and z components; an empty box counts as zero.

    ``name`` names each box for screen readers as ``"<name> x"`` and so on.
    ``tooltip`` is either a text, shown on each box as ``"<tooltip> (x)"``, or
    a function of the axis that returns each box's tooltip. Each box starts
    with ``values``, or with ``text`` exactly as given when it is set. The
    boxes keep their own range (``minimum``, ``maximum``, ``decimals``) and
    width; ``stretch`` adds a trailing stretch so the boxes stay at the left
    of a wider row.
    """

    #: Emitted when the text of any box changes.
    textChanged = Signal()

    def __init__(
        self,
        values: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        name: str,
        tooltip: str | Callable[[str], str] = "",
        text: str | None = None,
        minimum: float = -1_000_000_000.0,
        maximum: float = 1_000_000_000.0,
        decimals: int = 8,
        box_width: int = 64,
        stretch: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        style_inline_group(self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        edits = []
        for axis in _AXES:
            edit = DecimalLineEdit(
                self, text="", minimum=minimum, maximum=maximum, decimals=decimals
            )
            edit.setMaximumWidth(box_width)
            if callable(tooltip):
                edit.setToolTip(tooltip(axis))
            else:
                edit.setToolTip(f"{tooltip} ({axis})" if tooltip else axis)
            edit.setAccessibleName(f"{name} {axis}")
            edit.textChanged.connect(self._on_box_text_changed)
            layout.addWidget(edit)
            edits.append(edit)
        if stretch:
            layout.addStretch(1)
        x, y, z = edits
        self._edits = (x, y, z)
        if text is None:
            self.set_value(values)
        else:
            for edit in self._edits:
                edit.setText(text)

    def _on_box_text_changed(self, _text: str) -> None:
        self.textChanged.emit()

    @property
    def edits(self) -> tuple[DecimalLineEdit, DecimalLineEdit, DecimalLineEdit]:
        return self._edits

    def value(self, quantity: str) -> tuple[float, float, float]:
        """The vector; raises ``ValueError`` naming ``quantity`` for a component that is not a number."""

        x, y, z = (_vector_component(edit.text(), quantity) for edit in self._edits)
        return (x, y, z)

    def value_or_zero(self) -> tuple[float, float, float]:
        """The vector, with each component that is not a number read as zero."""

        components = []
        for edit in self._edits:
            try:
                components.append(_vector_component(edit.text(), "vector"))
            except ValueError:
                components.append(0.0)
        x, y, z = components
        return (x, y, z)

    def set_value(self, values: object) -> None:
        """Show the three components of ``values``, or zeros when it is not three finite numbers."""

        components: tuple[float, ...] = (0.0, 0.0, 0.0)
        if (
            isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
            and len(values) == 3
        ):
            try:
                parsed = tuple(float(value) for value in values)
            except (TypeError, ValueError):
                parsed = components
            if all(math.isfinite(value) for value in parsed):
                components = parsed
        for edit, value in zip(self._edits, components, strict=True):
            edit.setText(f"{value:g}")


def _vector_component(text: str, quantity: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    try:
        value = float(stripped)
    except ValueError as exc:
        raise ValueError(f"{quantity} components must be decimal numbers") from exc
    if not math.isfinite(value):
        raise ValueError(f"{quantity} components must be finite")
    return value
