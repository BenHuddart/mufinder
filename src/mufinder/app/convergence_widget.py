"""Cutoff-convergence configuration panel with workflow-aware CASTEP settings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.convergence_types import CUTOFF_KEY_CODEC
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.sweep_convergence_widget import SweepConvergenceWidget
from mufinder.castep_io.gui_metadata import infer_task

_DEFAULT_CUTOFF_RANGE: Mapping[str, object] = {
    "start": 300.0,
    "stop": 1400.0,
    "step": 100.0,
    "replace_existing": False,
}
_FIRST_CUTOFF = 300.0
_CUTOFF_STEP = 20.0


class CutoffRangeDialog(QDialog):
    """Compact dialog for generating multiple cutoff rows from a numeric range."""

    def __init__(self, defaults: Mapping[str, object] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate Cutoffs From Range")

        defaults = defaults or {}
        start = float(defaults.get("start", 300.0))
        stop = float(defaults.get("stop", 1400.0))
        step = float(defaults.get("step", 100.0))
        replace_existing = bool(defaults.get("replace_existing", False))

        self._start = DecimalLineEdit(self)
        self._start.setRange(1.0, 10_000.0)
        self._start.setValue(start)

        self._stop = DecimalLineEdit(self)
        self._stop.setRange(1.0, 10_000.0)
        self._stop.setValue(stop)

        self._step = DecimalLineEdit(self)
        self._step.setRange(1.0, 5000.0)
        self._step.setValue(step)

        self._replace_existing = QCheckBox("Replace existing cutoff list", self)
        self._replace_existing.setChecked(replace_existing)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("Start (eV)", self._start)
        form.addRow("Stop (eV)", self._stop)
        form.addRow("Step (eV)", self._step)
        layout.addLayout(form)
        layout.addWidget(self._replace_existing)
        layout.addWidget(self._buttons)

        self._buttons.accepted.connect(self._accept_if_valid)
        self._buttons.rejected.connect(self.reject)

    def values(self) -> tuple[float, ...]:
        """Return the generated cutoff list implied by the current controls."""
        start = self._start.value()
        stop = self._stop.value()
        step = self._step.value()

        values: list[float] = []
        current = start
        guard = 0
        while current <= stop + (step * 1e-6):
            values.append(round(current, 8))
            current += step
            guard += 1
            if guard > 5000:
                raise ValueError("Generated range exceeded 5000 rows.")
        return tuple(values)

    def replace_existing(self) -> bool:
        """Return whether generated cutoffs should replace the current list."""
        return self._replace_existing.isChecked()

    def defaults_state(self) -> dict[str, object]:
        """Persistable dialog state for later reopenings."""
        return {
            "start": self._start.value(),
            "stop": self._stop.value(),
            "step": self._step.value(),
            "replace_existing": self._replace_existing.isChecked(),
        }

    def _accept_if_valid(self) -> None:
        if self._step.value() <= 0:
            QMessageBox.warning(self, "Invalid range", "Step must be positive.")
            return
        if self._stop.value() < self._start.value():
            QMessageBox.warning(self, "Invalid range", "Stop must be greater than or equal to start.")
            return
        try:
            self.values()
        except ValueError as exc:
            QMessageBox.warning(self, "Range too large", str(exc))
            return
        self.accept()


class CutoffConvergenceWidget(SweepConvergenceWidget):
    """UI for configuring cutoff convergence sweeps and per-cutoff parameters."""

    remove_cutoff_requested = Signal(object)

    _CODEC = CUTOFF_KEY_CODEC
    _WORKFLOW_STAGE = "convergence_cutoff"
    _SWEEP_PARAMETER = "cut_off_energy"
    _KEYS_STATE_FIELD = "cutoff_energies"
    _ROWS_STATE_FIELD = "cutoff_parameter_rows"
    _ROW_KEY_FIELD = "cutoff"
    _NAME_PREFIX_DEFAULT = "cutoff_convergence"
    _DEFAULT_RANGE_STATE = _DEFAULT_CUTOFF_RANGE
    _REMOVE_SIGNAL = "remove_cutoff_requested"

    _TITLE = "Cutoff Convergence"
    _SUBTITLE = "Check whether cutoff settings are accurate enough before production runs."
    _SWEEP_GROUP_TITLE = "Cutoff Sweep"
    _KEY_COLUMN_LABEL = "Cutoff Energy (eV)"
    _ADD_BUTTON_TEXT = "Add Cutoff"
    _GENERATE_BUTTON_TEXT = "Generate From Range..."
    _EXAMINE_BUTTON_TEXT = "Examine Cutoff Convergence"
    _EXAMINE_READY_TOOLTIP = "Analyse completed cutoff energy and force trends."
    _EXAMINE_WAITING_TOOLTIP = (
        "Need at least two completed cutoffs with both total energy and max-force results."
    )
    _LOCKED_KEY_TOOLTIP = (
        "Cutoff value is locked once this row has been queued for execution or has entered "
        "execution history."
    )
    _REMOVE_ROW_TOOLTIP = "Remove this cutoff"
    _CONFIGURE_NOUN = "cutoff"
    _REMOVAL_NOUN = ("cutoff", "cutoffs")
    # Removing a cutoff whose job is queued or running cancels the job first.
    _CANCEL_ACTIVE_ON_REMOVE = True
    _DIALOG_SUBJECT = "Cutoff Parameters"
    _DIALOG_INTRO_PREFIX = "Included cutoffs: "
    _INVALID_PARAMETERS_PREFIX = "Cutoff"
    _NO_SELECTION_MESSAGE = ("No cutoffs selected", "Select one or more cutoff rows first.")
    _INVALID_KEYS_MESSAGE = ("Invalid cutoff list", "Cutoff values must be numeric.")
    _NO_KEYS_MESSAGE = ("No cutoffs", "Add one or more cutoff energies first.")
    _DUPLICATE_KEYS_MESSAGE = ("Duplicate cutoffs", "Cutoff energies must be unique.")

    # -- Sweep hooks -------------------------------------------------------------

    def next_key(self, previous: float | None) -> float:
        return _FIRST_CUTOFF if previous is None else previous + _CUTOFF_STEP

    def generate_keys(self) -> tuple[tuple[float, ...], bool] | None:
        dialog = CutoffRangeDialog(self._range_state, self)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        self._range_state = dialog.defaults_state()
        if not accepted:
            return None
        return dialog.values(), dialog.replace_existing()

    def remove_cutoffs(self, cutoffs: Sequence[float]) -> None:
        """Remove the rows for ``cutoffs`` and forget their parameters."""

        self.remove_keys(cutoffs)

    # -- Cutoff-only behaviour ---------------------------------------------------

    def _dialog_subject(self, keys: tuple[Any, ...]) -> str:
        if len(keys) == 1:
            return f"Parameters: {self._CODEC.describe(keys[0])}"
        return self._DIALOG_SUBJECT

    def _is_default_parameter_set(self, values: Mapping[str, Any]) -> bool:
        # A task-only mapping is what the legacy-restore path used to write for
        # a default row; the parameter dialog cannot produce one. Reading it
        # back as "defaults" repairs rows saved that way, so they match the
        # definition their jobs ran under again.
        task_only = {"task": self._default_params.get("task", infer_task(self._WORKFLOW_STAGE))}
        return values in (self._default_params, task_only)

    def _restore_legacy_parameters(self, state: Mapping[str, Any], keys: tuple[Any, ...]) -> None:
        # Only a legacy state (or a definition's configuration) carries
        # shared_params/overrides. A current state whose rows all use the
        # defaults has no cutoff_parameter_rows and no shared_params, so it
        # never reaches here: reading it as legacy would pin every row to an
        # explicit task-only parameter set, so the rows would stop matching the
        # definition their jobs ran under and be submitted again.
        shared = state.get("shared_params", {})
        shared_clean = (
            self._sanitize({str(name): raw for name, raw in shared.items()})
            if isinstance(shared, dict)
            else dict(self._default_params)
        )
        if "task" not in shared_clean:
            shared_clean["task"] = infer_task(self._WORKFLOW_STAGE)
        shared_advanced = {
            str(name) for name in state.get("shared_added_advanced", []) if isinstance(name, str)
        }

        overrides_by_cutoff: dict[float, dict[str, object]] = {}
        override_advanced: dict[float, set[str]] = {}
        for entry in state.get("overrides", []):
            if not isinstance(entry, dict):
                continue
            params = self._sanitize({str(name): raw for name, raw in dict(entry.get("params", {})).items()})
            advanced = {name for name in params if name in self._specs_by_name and params[name] is not None}
            for raw_cutoff in entry.get("cutoffs", []):
                cutoff = self._CODEC.coerce(raw_cutoff)
                if cutoff is None:
                    continue
                overrides_by_cutoff[cutoff] = dict(params)
                override_advanced[cutoff] = set(advanced)

        for cutoff in keys:
            merged = dict(shared_clean)
            merged.update(overrides_by_cutoff.get(cutoff, {}))
            advanced_names = set(shared_advanced).union(override_advanced.get(cutoff, set()))
            self._set_parameter_values(cutoff, merged, advanced_names)


# Backward-compatible export used by existing call sites.
ConvergenceWidget = CutoffConvergenceWidget
