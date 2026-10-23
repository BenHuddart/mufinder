"""Shared project-defaults strip shown in the header of every workflow tab.

One compact, uniform summary of the three project-wide calculation defaults
(XC functional, cutoff energy, k-point grid) replaces the per-widget
"Project Defaults" group boxes. How each convergence default was established
(method, criteria, the functional it was converged with) lives in the tooltips
and in :class:`ProjectDefaultsDialog`, which is also where the defaults can be
edited by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.castep_io.schema import get_keyword_spec
from mufinder.domain.models import MANUAL_DEFAULT_METHOD, Project

from mufinder.app.convergence_display import (
    format_convergence_criteria,
    format_convergence_method,
    format_cutoff_energy,
    format_kpoint_grid,
)
from mufinder.app.theme import refresh_widget_style, style_hint_text, style_warning_text

_NOT_SET = "Not set"


@dataclass(frozen=True, slots=True)
class ProjectDefaultsSummary:
    """Display snapshot of the project-wide calculation defaults."""

    xc_functional: str | None = None
    cutoff_energy: float | None = None
    cutoff_method: str | None = None
    cutoff_energy_threshold: float | None = None
    cutoff_force_threshold: float | None = None
    cutoff_stress_threshold: float | None = None
    cutoff_xc_functional: str | None = None
    kpoint_grid: tuple[int, int, int] | None = None
    kpoint_method: str | None = None
    kpoint_energy_threshold: float | None = None
    kpoint_force_threshold: float | None = None
    kpoint_stress_threshold: float | None = None
    kpoint_xc_functional: str | None = None

    @classmethod
    def from_project(cls, project: Project | None) -> ProjectDefaultsSummary:
        if project is None:
            return cls()
        return cls(
            xc_functional=project.recommended_xc_functional,
            cutoff_energy=project.recommended_cutoff_energy,
            cutoff_method=project.recommended_cutoff_method,
            cutoff_energy_threshold=project.recommended_cutoff_energy_threshold,
            cutoff_force_threshold=project.recommended_cutoff_force_threshold,
            cutoff_stress_threshold=project.recommended_cutoff_stress_threshold,
            cutoff_xc_functional=project.recommended_cutoff_xc_functional,
            kpoint_grid=project.recommended_kpoint_grid,
            kpoint_method=project.recommended_kpoint_method,
            kpoint_energy_threshold=project.recommended_kpoint_energy_threshold,
            kpoint_force_threshold=project.recommended_kpoint_force_threshold,
            kpoint_stress_threshold=project.recommended_kpoint_stress_threshold,
            kpoint_xc_functional=project.recommended_kpoint_xc_functional,
        )

    def with_xc_functional(self, xc_functional: str | None) -> ProjectDefaultsSummary:
        return replace(self, xc_functional=xc_functional)

    @property
    def cutoff_is_stale(self) -> bool:
        """True when the cutoff was converged with a different functional."""

        return self.cutoff_energy is not None and _functionals_differ(
            self.xc_functional, self.cutoff_xc_functional
        )

    @property
    def kpoint_is_stale(self) -> bool:
        """True when the k-point grid was converged with a different functional."""

        return self.kpoint_grid is not None and _functionals_differ(
            self.xc_functional, self.kpoint_xc_functional
        )

    def cutoff_detail(self) -> str:
        if self.cutoff_energy is None:
            return "Adopt a value from the cutoff convergence analysis to set this."
        return _convergence_detail(
            method=self.cutoff_method,
            energy_threshold=self.cutoff_energy_threshold,
            force_threshold=self.cutoff_force_threshold,
            stress_threshold=self.cutoff_stress_threshold,
            xc_functional=self.cutoff_xc_functional,
        )

    def kpoint_detail(self) -> str:
        if self.kpoint_grid is None:
            return "Adopt a grid from the k-point convergence analysis to set this."
        return _convergence_detail(
            method=self.kpoint_method,
            energy_threshold=self.kpoint_energy_threshold,
            force_threshold=self.kpoint_force_threshold,
            stress_threshold=self.kpoint_stress_threshold,
            xc_functional=self.kpoint_xc_functional,
        )

    def stale_warning(self, converged_with: str | None) -> str:
        return (
            f"Converged with {converged_with}, but the project functional is "
            f"{self.xc_functional}. Re-run the convergence test to confirm this value."
        )


def _functionals_differ(project_functional: str | None, converged_with: str | None) -> bool:
    if not project_functional or not converged_with:
        return False
    return project_functional.strip().upper() != converged_with.strip().upper()


def _convergence_detail(
    *,
    method: str | None,
    energy_threshold: float | None,
    force_threshold: float | None,
    stress_threshold: float | None,
    xc_functional: str | None,
) -> str:
    if method == MANUAL_DEFAULT_METHOD:
        return f"{format_convergence_method(method)}, not from a convergence test."
    parts: list[str] = []
    if method is not None:
        parts.append(format_convergence_method(method))
    if not (energy_threshold is None and force_threshold is None and stress_threshold is None):
        parts.append(
            format_convergence_criteria(
                energy_threshold=energy_threshold,
                force_threshold=force_threshold,
                stress_threshold=stress_threshold,
            )
        )
    parts.append(f"converged with {xc_functional}" if xc_functional else "functional not recorded")
    return " \N{MIDDLE DOT} ".join(parts)


def xc_functional_choices() -> tuple[str, ...]:
    """Return the catalogued ``XC_FUNCTIONAL`` values offered as project defaults."""

    spec = get_keyword_spec("param", "xc_functional")
    if spec is None:
        return ()
    return tuple(str(value) for value in spec.allowed_values)


def parse_cutoff_energy_text(text: str) -> float | None:
    """Parse a cutoff energy in eV; blank means "not set"."""

    stripped = text.strip()
    if stripped.lower().endswith("ev"):
        stripped = stripped[:-2].strip()
    if not stripped:
        return None
    try:
        value = float(stripped)
    except ValueError:
        raise ValueError("Cutoff energy must be a number in eV.") from None
    if not value > 0 or value == float("inf"):
        raise ValueError("Cutoff energy must be a positive number in eV.")
    return value


def parse_kpoint_grid_text(text: str) -> tuple[int, int, int] | None:
    """Parse a Monkhorst-Pack grid such as ``4 4 4`` or ``4x4x4``; blank means "not set"."""

    if not text.strip():
        return None
    parts = [part for part in re.split(r"[\sx,\u00d7]+", text.strip().lower()) if part]
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError("K-point grid must be three positive integers, e.g. 4 4 4.")
    grid = (int(parts[0]), int(parts[1]), int(parts[2]))
    if min(grid) <= 0:
        raise ValueError("K-point grid must be three positive integers, e.g. 4 4 4.")
    return grid


class ProjectDefaultsDialog(QDialog):
    """Details of the project defaults, each of which can be edited by hand."""

    def __init__(self, summary: ProjectDefaultsSummary, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project Defaults")
        self._summary = summary

        self._xc_combo = QComboBox()
        self._xc_combo.addItem(_NOT_SET, None)
        choices = list(xc_functional_choices())
        if summary.xc_functional and summary.xc_functional not in choices:
            choices.insert(0, summary.xc_functional)
        for choice in choices:
            self._xc_combo.addItem(choice, choice)
        index = self._xc_combo.findData(summary.xc_functional)
        self._xc_combo.setCurrentIndex(max(index, 0))

        xc_hint = QLabel(
            "Applied to new calculations in every workflow. Rows you have "
            "configured explicitly keep their own functional."
        )
        xc_hint.setWordWrap(True)
        style_hint_text(xc_hint)

        self._cutoff_value = QLineEdit()
        self._cutoff_value.setPlaceholderText(_NOT_SET)
        if summary.cutoff_energy is not None:
            self._cutoff_value.setText(f"{float(summary.cutoff_energy):.8g}")
        cutoff_row = QHBoxLayout()
        cutoff_row.addWidget(self._cutoff_value, 1)
        cutoff_row.addWidget(QLabel("eV"))
        self._cutoff_detail = QLabel()
        self._cutoff_warning = QLabel()
        self._kpoint_value = QLineEdit()
        self._kpoint_value.setPlaceholderText(_NOT_SET)
        if summary.kpoint_grid is not None:
            self._kpoint_value.setText(format_kpoint_grid(summary.kpoint_grid))
        self._kpoint_detail = QLabel()
        self._kpoint_warning = QLabel()
        self._error = QLabel()
        self._error.setWordWrap(True)
        style_warning_text(self._error)
        self._error.hide()
        for detail in (self._cutoff_detail, self._kpoint_detail):
            detail.setWordWrap(True)
            style_hint_text(detail)
        for warning in (self._cutoff_warning, self._kpoint_warning):
            warning.setWordWrap(True)
            style_warning_text(warning)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.addWidget(QLabel("XC functional"), 0, 0)
        grid.addWidget(self._xc_combo, 0, 1)
        grid.addWidget(xc_hint, 1, 1)
        grid.setRowMinimumHeight(2, 8)
        grid.addWidget(QLabel("Cutoff energy"), 3, 0)
        grid.addLayout(cutoff_row, 3, 1)
        grid.addWidget(self._cutoff_detail, 4, 1)
        grid.addWidget(self._cutoff_warning, 5, 1)
        grid.setRowMinimumHeight(6, 8)
        grid.addWidget(QLabel("K-point grid"), 7, 0)
        grid.addWidget(self._kpoint_value, 7, 1)
        grid.addWidget(self._kpoint_detail, 8, 1)
        grid.addWidget(self._kpoint_warning, 9, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(12)
        layout.addLayout(grid)
        layout.addWidget(self._error)
        layout.addStretch(1)
        layout.addWidget(buttons)
        self.setMinimumWidth(460)

        self._xc_combo.currentIndexChanged.connect(self._refresh_details)
        self._cutoff_value.textChanged.connect(self._refresh_details)
        self._kpoint_value.textChanged.connect(self._refresh_details)
        self._refresh_details()

    def xc_functional(self) -> str | None:
        """Return the functional currently selected in the dialog."""

        value = self._xc_combo.currentData()
        return str(value) if value else None

    def cutoff_energy(self) -> float | None:
        """Return the entered cutoff in eV (``None`` when blank).

        Raises ``ValueError`` when the text is not a positive number.
        """

        return parse_cutoff_energy_text(self._cutoff_value.text())

    def kpoint_grid(self) -> tuple[int, int, int] | None:
        """Return the entered grid (``None`` when blank).

        Raises ``ValueError`` when the text is not three positive integers.
        """

        return parse_kpoint_grid_text(self._kpoint_value.text())

    def edited_summary(self) -> ProjectDefaultsSummary:
        """Return the summary as it would stand if the dialog were accepted.

        A cutoff or grid that differs from the stored one is a manual entry, so
        it loses the method, criteria and functional of the old value.
        """

        summary = self._summary.with_xc_functional(self.xc_functional())
        cutoff = self.cutoff_energy()
        if cutoff != summary.cutoff_energy:
            summary = replace(
                summary,
                cutoff_energy=cutoff,
                cutoff_method=None if cutoff is None else MANUAL_DEFAULT_METHOD,
                cutoff_energy_threshold=None,
                cutoff_force_threshold=None,
                cutoff_stress_threshold=None,
                cutoff_xc_functional=None,
            )
        grid = self.kpoint_grid()
        if grid != summary.kpoint_grid:
            summary = replace(
                summary,
                kpoint_grid=grid,
                kpoint_method=None if grid is None else MANUAL_DEFAULT_METHOD,
                kpoint_energy_threshold=None,
                kpoint_force_threshold=None,
                kpoint_stress_threshold=None,
                kpoint_xc_functional=None,
            )
        return summary

    def _accept_if_valid(self) -> None:
        try:
            self.edited_summary()
        except ValueError as exc:
            self._error.setText(str(exc))
            self._error.show()
            return
        self.accept()

    def _refresh_details(self) -> None:
        self._error.hide()
        try:
            summary = self.edited_summary()
        except ValueError:
            # Mid-edit text; the details catch up once it parses again.
            return
        self._cutoff_detail.setText(summary.cutoff_detail())
        self._cutoff_warning.setText(summary.stale_warning(summary.cutoff_xc_functional))
        self._cutoff_warning.setVisible(summary.cutoff_is_stale)
        self._kpoint_detail.setText(summary.kpoint_detail())
        self._kpoint_warning.setText(summary.stale_warning(summary.kpoint_xc_functional))
        self._kpoint_warning.setVisible(summary.kpoint_is_stale)


class ProjectDefaultsStrip(QFrame):
    """Single-line summary of the project defaults for a workflow tab header."""

    #: Each is emitted with the newly entered value; ``None`` clears the default.
    xc_functional_change_requested = Signal(object)
    cutoff_energy_change_requested = Signal(object)
    kpoint_grid_change_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectDefaultsStrip")
        self._summary = ProjectDefaultsSummary()

        caption = QLabel("Project defaults")
        caption.setObjectName("projectDefaultsCaption")

        self._xc_value = QLabel()
        self._cutoff_value = QLabel()
        self._kpoint_value = QLabel()

        self._details_button = QToolButton()
        self._details_button.setObjectName("projectDefaultsButton")
        self._details_button.setText("Edit\N{HORIZONTAL ELLIPSIS}")
        self._details_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._details_button.setToolTip(
            "Show how the project defaults were established, or change them."
        )
        self._details_button.clicked.connect(self._open_details)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 5, 5, 5)
        layout.setSpacing(10)
        layout.addWidget(caption)
        for name, value in (
            ("XC", self._xc_value),
            ("Cutoff", self._cutoff_value),
            ("K-points", self._kpoint_value),
        ):
            layout.addWidget(self._separator())
            name_label = QLabel(name)
            name_label.setObjectName("projectDefaultsName")
            value.setObjectName("projectDefaultsValue")
            segment = QHBoxLayout()
            segment.setSpacing(6)
            segment.addWidget(name_label)
            segment.addWidget(value)
            layout.addLayout(segment)
        layout.addWidget(self._details_button)

        self.set_defaults(self._summary)

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setObjectName("projectDefaultsSeparator")
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFixedWidth(1)
        return line

    def summary(self) -> ProjectDefaultsSummary:
        return self._summary

    def set_defaults(self, summary: ProjectDefaultsSummary) -> None:
        """Display ``summary``; values converged with another functional are flagged."""

        self._summary = summary
        self._xc_value.setText(summary.xc_functional or _NOT_SET)
        self._xc_value.setToolTip(
            "Exchange-correlation functional applied by default to new calculations."
        )
        self._set_value(self._xc_value, unset=summary.xc_functional is None, stale=False)

        self._show_convergence_value(
            self._cutoff_value,
            text=format_cutoff_energy(summary.cutoff_energy),
            unset=summary.cutoff_energy is None,
            stale=summary.cutoff_is_stale,
            converged_with=summary.cutoff_xc_functional,
            detail=summary.cutoff_detail(),
        )
        self._show_convergence_value(
            self._kpoint_value,
            text=format_kpoint_grid(summary.kpoint_grid),
            unset=summary.kpoint_grid is None,
            stale=summary.kpoint_is_stale,
            converged_with=summary.kpoint_xc_functional,
            detail=summary.kpoint_detail(),
        )

    def _show_convergence_value(
        self,
        label: QLabel,
        *,
        text: str,
        unset: bool,
        stale: bool,
        converged_with: str | None,
        detail: str,
    ) -> None:
        if stale:
            label.setText(f"{text} ({converged_with})")
            label.setToolTip(f"{self._summary.stale_warning(converged_with)}\n{detail}")
        else:
            label.setText(text)
            label.setToolTip(detail)
        self._set_value(label, unset=unset, stale=stale)

    @staticmethod
    def _set_value(label: QLabel, *, unset: bool, stale: bool) -> None:
        state = "stale" if stale else ("unset" if unset else "set")
        if label.property("valueState") != state:
            label.setProperty("valueState", state)
            refresh_widget_style(label)

    def _open_details(self) -> None:
        dialog = ProjectDefaultsDialog(self._summary, self.window())
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        current = self._summary
        xc_functional = dialog.xc_functional()
        cutoff_energy = dialog.cutoff_energy()
        kpoint_grid = dialog.kpoint_grid()
        if xc_functional != current.xc_functional:
            self.xc_functional_change_requested.emit(xc_functional)
        if cutoff_energy != current.cutoff_energy:
            self.cutoff_energy_change_requested.emit(cutoff_energy)
        if kpoint_grid != current.kpoint_grid:
            self.kpoint_grid_change_requested.emit(kpoint_grid)


__all__ = [
    "ProjectDefaultsDialog",
    "ProjectDefaultsStrip",
    "ProjectDefaultsSummary",
    "parse_cutoff_energy_text",
    "parse_kpoint_grid_text",
    "xc_functional_choices",
]
