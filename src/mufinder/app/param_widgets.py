"""Reusable typed widgets for CASTEP parameter entry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from mufinder.app.block_dialog import BlockParameterDialog, summarize_block_value
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.hubbard_u_editor import HubbardUDialog, summarize_hubbard_u
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import BlockSection, ValueWithUnit


class ParamEditorBase(QWidget):
    """Abstract editor widget for one CASTEP keyword."""

    value_changed = Signal()

    def __init__(self, spec: KeywordSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec = spec

    @property
    def spec(self) -> KeywordSpec:
        return self._spec

    def value(self) -> object:
        raise NotImplementedError

    def set_value(self, value: object) -> None:
        raise NotImplementedError


class EnumParamEditor(ParamEditorBase):
    def __init__(self, spec: KeywordSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._combo = QComboBox()
        self._combo.addItems(spec.allowed_values)
        self._combo.currentIndexChanged.connect(lambda *_: self.value_changed.emit())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._combo)

    def value(self) -> object:
        return self._combo.currentText()

    def set_value(self, value: object) -> None:
        self._combo.setCurrentText(str(value))


class BoolParamEditor(ParamEditorBase):
    def __init__(self, spec: KeywordSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._checkbox = QCheckBox()
        self._checkbox.stateChanged.connect(lambda *_: self.value_changed.emit())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def value(self) -> object:
        return self._checkbox.isChecked()

    def set_value(self, value: object) -> None:
        self._checkbox.setChecked(bool(value))


class IntParamEditor(ParamEditorBase):
    def __init__(self, spec: KeywordSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spin = QSpinBox()
        self._spin.setRange(
            int(spec.min_value if spec.min_value is not None else -1_000_000_000),
            int(spec.max_value if spec.max_value is not None else 1_000_000_000),
        )
        if spec.step is not None:
            self._spin.setSingleStep(int(spec.step))
        self._spin.valueChanged.connect(lambda *_: self.value_changed.emit())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._spin)

    def value(self) -> object:
        return self._spin.value()

    def set_value(self, value: object) -> None:
        self._spin.setValue(int(value))


class FloatParamEditor(ParamEditorBase):
    def __init__(self, spec: KeywordSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._spin = DecimalLineEdit()
        self._spin.setDecimals(12)
        self._spin.setRange(
            float(spec.min_value if spec.min_value is not None else -1_000_000_000.0),
            float(spec.max_value if spec.max_value is not None else 1_000_000_000.0),
        )
        if spec.step is not None:
            self._spin.setSingleStep(float(spec.step))
        self._spin.valueChanged.connect(lambda *_: self.value_changed.emit())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._spin)

    def value(self) -> object:
        return self._spin.value()

    def set_value(self, value: object) -> None:
        self._spin.setValue(float(value))


class PhysicalParamEditor(FloatParamEditor):
    def __init__(self, spec: KeywordSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        if spec.unit:
            self.layout().addWidget(QLabel(spec.unit))
            self._unit_edit: QLineEdit | None = None
        else:
            self._unit_edit = QLineEdit()
            self._unit_edit.setPlaceholderText("unit (optional)")
            self._unit_edit.textChanged.connect(lambda *_: self.value_changed.emit())
            self.layout().addWidget(self._unit_edit)

    def value(self) -> object:
        if self.spec.unit:
            return ValueWithUnit(self._spin.value(), self.spec.unit)
        if self._unit_edit is not None and self._unit_edit.text().strip():
            return ValueWithUnit(self._spin.value(), self._unit_edit.text().strip())
        return self._spin.value()

    def set_value(self, value: object) -> None:
        if isinstance(value, ValueWithUnit):
            self._spin.setValue(float(value.value))
            if self._unit_edit is not None:
                self._unit_edit.setText(value.unit)
            return
        self._spin.setValue(float(value))


class IntTripletParamEditor(ParamEditorBase):
    def __init__(self, spec: KeywordSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        minimum = int(spec.min_value if spec.min_value is not None else 1)
        maximum = int(spec.max_value if spec.max_value is not None else 512)
        step = int(spec.step if spec.step is not None else 1)

        self._x = QSpinBox()
        self._y = QSpinBox()
        self._z = QSpinBox()
        for control in (self._x, self._y, self._z):
            control.setRange(minimum, maximum)
            control.setSingleStep(step)
            control.valueChanged.connect(lambda *_: self.value_changed.emit())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._x)
        layout.addWidget(QLabel("x"))
        layout.addWidget(self._y)
        layout.addWidget(QLabel("x"))
        layout.addWidget(self._z)

    def value(self) -> object:
        return (self._x.value(), self._y.value(), self._z.value())

    def set_value(self, value: object) -> None:
        x, y, z = value  # type: ignore[misc]
        self._x.setValue(int(x))
        self._y.setValue(int(y))
        self._z.setValue(int(z))


class RealVectorParamEditor(ParamEditorBase):
    def __init__(self, spec: KeywordSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        minimum = float(spec.min_value if spec.min_value is not None else -1_000_000_000.0)
        maximum = float(spec.max_value if spec.max_value is not None else 1_000_000_000.0)
        step = float(spec.step if spec.step is not None else 0.1)
        length = int(spec.vector_length if spec.vector_length is not None else 3)

        self._controls: list[DecimalLineEdit] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for index in range(length):
            control = DecimalLineEdit()
            control.setDecimals(8)
            control.setRange(minimum, maximum)
            control.setSingleStep(step)
            control.valueChanged.connect(lambda *_: self.value_changed.emit())
            self._controls.append(control)
            layout.addWidget(control)
            if index < length - 1:
                layout.addWidget(QLabel("x"))

    def value(self) -> object:
        return tuple(control.value() for control in self._controls)

    def set_value(self, value: object) -> None:
        for control, component in zip(self._controls, value):  # type: ignore[arg-type]
            control.setValue(float(component))


class TextParamEditor(ParamEditorBase):
    def __init__(self, spec: KeywordSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._text = QLineEdit()
        self._text.textChanged.connect(lambda *_: self.value_changed.emit())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text)

    def value(self) -> object:
        return self._text.text().strip()

    def set_value(self, value: object) -> None:
        self._text.setText(str(value))


class BlockParamEditor(ParamEditorBase):
    def __init__(self, spec: KeywordSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec, parent)
        self._value: BlockSection | None = None

        self._summary = QLabel("Not set")
        self._edit_button = QPushButton("Edit…")
        self._clear_button = QPushButton("Clear")
        self._edit_button.clicked.connect(self._open_dialog)
        self._clear_button.clicked.connect(self._clear_value)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._summary, 1)
        layout.addWidget(self._edit_button)
        layout.addWidget(self._clear_button)

    def value(self) -> object:
        return self._value

    def set_value(self, value: object) -> None:
        if value is None:
            self._value = None
        elif isinstance(value, BlockSection):
            self._value = value
        else:
            raise TypeError(f"{self.spec.name} block editor expects a BlockSection or None.")
        self._refresh_summary()

    def _open_dialog(self) -> None:
        dialog = BlockParameterDialog(self.spec, initial_value=self._value, parent=self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self._value = dialog.block_value()
        self._refresh_summary()
        self.value_changed.emit()

    def _clear_value(self) -> None:
        self._value = None
        self._refresh_summary()
        self.value_changed.emit()

    def _refresh_summary(self) -> None:
        self._summary.setText(summarize_block_value(self.spec, self._value))


class HubbardUParamEditor(ParamEditorBase):
    def __init__(
        self,
        spec: KeywordSpec,
        parent: QWidget | None = None,
        *,
        species_options: Sequence[str] = (),
    ) -> None:
        super().__init__(spec, parent)
        self._value: BlockSection | None = None
        self._species_options = tuple(species_options)

        self._summary = QLabel("Not set")
        self._edit_button = QPushButton("Edit…")
        self._clear_button = QPushButton("Clear")
        self._edit_button.clicked.connect(self._open_dialog)
        self._clear_button.clicked.connect(self._clear_value)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._summary, 1)
        layout.addWidget(self._edit_button)
        layout.addWidget(self._clear_button)

    def value(self) -> object:
        return self._value

    def set_value(self, value: object) -> None:
        if value is None:
            self._value = None
        elif isinstance(value, BlockSection):
            self._value = value
        else:
            raise TypeError(f"{self.spec.name} editor expects a BlockSection or None.")
        self._refresh_summary()

    def set_species_options(self, species_options: Sequence[str]) -> None:
        self._species_options = tuple(species_options)

    def _open_dialog(self) -> None:
        dialog = HubbardUDialog(
            initial_value=self._value,
            species_options=self._species_options,
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self._value = dialog.block_value()
        self._refresh_summary()
        self.value_changed.emit()

    def _clear_value(self) -> None:
        self._value = None
        self._refresh_summary()
        self.value_changed.emit()

    def _refresh_summary(self) -> None:
        self._summary.setText(summarize_hubbard_u(self._value))


def _fallback_widget_hint(spec: KeywordSpec) -> str:
    if spec.kind == "flag":
        return "flag"
    if spec.kind == "block":
        return "block"
    if spec.value_type == "enum":
        return "dropdown"
    if spec.value_type == "bool":
        return "checkbox"
    if spec.value_type in {"int", "positive_int", "non_negative_int"}:
        return "spinbox_int"
    if spec.value_type == "int_triplet":
        return "int_triplet"
    if spec.value_type == "real_vector":
        return "real_vector"
    if spec.value_type in {"float", "number"}:
        return "spinbox_float"
    if spec.value_type in {"physical", "positive_float_with_optional_unit"}:
        return "physical"
    return "text"


def create_param_editor(
    spec: KeywordSpec,
    parent: QWidget | None = None,
    *,
    species_options: Sequence[str] = (),
) -> ParamEditorBase:
    """Create a typed editor widget for the provided keyword metadata."""

    hint = spec.widget_hint or _fallback_widget_hint(spec)
    if hint == "hubbard_u":
        editor: ParamEditorBase = HubbardUParamEditor(spec, parent, species_options=species_options)
        if spec.default_value is not None:
            try:
                editor.set_value(spec.default_value)
            except (TypeError, ValueError):
                pass
        return editor

    factories: dict[str, Callable[[KeywordSpec, QWidget | None], ParamEditorBase]] = {
        "dropdown": EnumParamEditor,
        "checkbox": BoolParamEditor,
        "flag": BoolParamEditor,
        "spinbox_int": IntParamEditor,
        "spinbox_float": FloatParamEditor,
        "unit_float": PhysicalParamEditor,
        "physical": PhysicalParamEditor,
        "int_triplet": IntTripletParamEditor,
        "real_vector": RealVectorParamEditor,
        "block": BlockParamEditor,
        "text": TextParamEditor,
    }

    factory = factories.get(hint, TextParamEditor)
    editor = factory(spec, parent)
    if spec.default_value is not None:
        try:
            editor.set_value(spec.default_value)
        except (TypeError, ValueError):
            # Runtime CASTEP metadata can occasionally contain mismatched
            # defaults (for example string placeholders for vector fields).
            # Keep the editor usable by leaving that field unset.
            pass
    return editor
