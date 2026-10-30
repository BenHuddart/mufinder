"""K-point convergence configuration panel with workflow-aware CASTEP settings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.convergence_types import KPOINT_GRID_CODEC, KPointGrid
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.sweep_convergence_widget import SweepConvergenceWidget
from mufinder.castep_io.types import ValueWithUnit
from mufinder.science.kpoint_grids import generate_kpoint_grids, lattice_matches

_DEFAULT_KPOINT_SPACING = 0.04
_FIRST_GRID: KPointGrid = (1, 1, 1)


class KPointSpacingDialog(QDialog):
    """Dialog for generating Monkhorst-Pack grids from a spacing target."""

    def __init__(
        self, defaults: Mapping[str, object] | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate K-Point Grids")

        defaults = defaults or {}
        spacing = float(defaults.get("spacing", _DEFAULT_KPOINT_SPACING))
        replace_existing = bool(defaults.get("replace_existing", False))

        self._spacing = DecimalLineEdit(self)
        self._spacing.setDecimals(4)
        self._spacing.setRange(0.001, 10.0)
        self._spacing.setSingleStep(0.01)
        self._spacing.setValue(spacing)
        self._spacing.setToolTip(
            "Minimum KPOINT_MP_SPACING for the sweep, in 1/Ang. The generator "
            "walks CASTEP's spacing curve and emits every distinct grid from "
            "(1, 1, 1) up to the grid CASTEP would produce at this spacing."
        )

        self._replace_existing = QCheckBox("Replace existing grid list", self)
        self._replace_existing.setChecked(replace_existing)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("Minimum spacing (1/Ang)", self._spacing)
        layout.addLayout(form)
        layout.addWidget(self._replace_existing)
        layout.addWidget(self._buttons)

        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

    def spacing(self) -> float:
        return float(self._spacing.value())

    def replace_existing(self) -> bool:
        return self._replace_existing.isChecked()

    def defaults_state(self) -> dict[str, object]:
        return {
            "spacing": float(self._spacing.value()),
            "replace_existing": self._replace_existing.isChecked(),
        }


class KPointGridDialog(QDialog):
    """Small helper dialog for entering one Monkhorst-Pack grid triplet."""

    def __init__(self, grid: KPointGrid = _FIRST_GRID, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add K-Point Grid")

        validator = QIntValidator(1, 9999, self)

        self._nx = QLineEdit(str(int(grid[0])), self)
        self._ny = QLineEdit(str(int(grid[1])), self)
        self._nz = QLineEdit(str(int(grid[2])), self)
        for field in (self._nx, self._ny, self._nz):
            field.setValidator(validator)
            field.setMaxLength(4)
            field.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Enter Monkhorst-Pack grid integers", self))

        row = QHBoxLayout()
        row.addWidget(self._nx)
        row.addWidget(QLabel("x", self))
        row.addWidget(self._ny)
        row.addWidget(QLabel("x", self))
        row.addWidget(self._nz)
        layout.addLayout(row)
        layout.addWidget(self._buttons)

        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._nx.textChanged.connect(self._refresh_ok_state)
        self._ny.textChanged.connect(self._refresh_ok_state)
        self._nz.textChanged.connect(self._refresh_ok_state)

        self._refresh_ok_state()

    def _refresh_ok_state(self) -> None:
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is None:
            return
        ok_button.setEnabled(
            bool(self._nx.hasAcceptableInput())
            and bool(self._ny.hasAcceptableInput())
            and bool(self._nz.hasAcceptableInput())
        )

    def grid(self) -> KPointGrid:
        return (int(self._nx.text()), int(self._ny.text()), int(self._nz.text()))


class KPointConvergenceWidget(SweepConvergenceWidget):
    """UI for configuring k-point convergence sweeps and per-grid parameters."""

    remove_grid_requested = Signal(object)

    _CODEC = KPOINT_GRID_CODEC
    _WORKFLOW_STAGE = "convergence_kpoint"
    _SWEEP_PARAMETER = "kpoint_mp_grid"
    _KEYS_STATE_FIELD = "kpoint_grids"
    _ROWS_STATE_FIELD = "kpoint_parameter_rows"
    _ROW_KEY_FIELD = "grid"
    _NAME_PREFIX_DEFAULT = "kpoint_convergence"
    _DEFAULT_RANGE_STATE = {"spacing": _DEFAULT_KPOINT_SPACING, "replace_existing": False}
    _REMOVE_SIGNAL = "remove_grid_requested"
    _KEY_COLUMN_WIDTH = 128

    _TITLE = "K-Point Convergence"
    _SUBTITLE = "Refine k-point sampling so energies and forces are stable."
    _SWEEP_GROUP_TITLE = "K-Point Grid Sweep"
    _KEY_COLUMN_LABEL = "K-Point Grid"
    _ADD_BUTTON_TEXT = "Add Grid"
    _GENERATE_BUTTON_TEXT = "Generate From Spacing…"
    _EXAMINE_BUTTON_TEXT = "Examine K-Point Convergence"
    _EXAMINE_READY_TOOLTIP = "Analyse completed k-point energy and force trends."
    _EXAMINE_WAITING_TOOLTIP = (
        "Need at least two completed grids with both total energy and max-force results."
    )
    _LOCKED_KEY_TOOLTIP = (
        "K-point grid is locked once this row has been queued for execution or has entered "
        "execution history."
    )
    _REMOVE_ROW_TOOLTIP = "Remove this k-point grid"
    _ROW_NOUN = "K-Point Grid"
    _CONFIGURE_NOUN = "grid"
    _REMOVAL_NOUN = ("grid", "grids")
    _DIALOG_SUBJECT = "K-Point Parameters"
    _DIALOG_INTRO_PREFIX = "Included grids: "
    _INVALID_PARAMETERS_PREFIX = "Grid"
    _NO_SELECTION_MESSAGE = ("No grids selected", "Select one or more k-point rows first.")
    _INVALID_KEYS_MESSAGE = ("Invalid grid list", "K-point grids must be integer triplets.")
    _NO_KEYS_MESSAGE = ("No grids", "Add one or more k-point grids first.")
    _DUPLICATE_KEYS_MESSAGE = ("Duplicate grids", "K-point grids must be unique.")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # The cell the listed grids were generated for, and the cell the
        # workflow uses now; a mismatch shows the regenerate warning.
        self._generation_cell_id: str | None = None
        self._generation_lattice: list[list[float]] | None = None
        self._current_cell_id: str | None = None
        self._current_lattice: list[list[float]] | None = None

    # -- Sweep hooks -------------------------------------------------------------

    def next_key(self, previous: KPointGrid | None) -> KPointGrid:
        return _FIRST_GRID if previous is None else previous

    def prompt_new_key(self) -> KPointGrid | None:
        dialog = KPointGridDialog(grid=self.next_key(self._last_key()), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.grid()

    def generate_keys(self) -> tuple[tuple[KPointGrid, ...], bool] | None:
        if self._current_lattice is None:
            QMessageBox.warning(
                self, "No simulation cell", "Select or create a simulation cell first."
            )
            return None

        dialog = KPointSpacingDialog(self._range_state, self)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        self._range_state = dialog.defaults_state()
        if not accepted:
            return None
        try:
            generated = generate_kpoint_grids(self._current_lattice, dialog.spacing())
        except ValueError as exc:
            QMessageBox.warning(self, "Grid generation failed", str(exc))
            return None

        self._generation_cell_id = self._current_cell_id
        self._generation_lattice = [list(row) for row in self._current_lattice]
        self._refresh_generation_warning()
        return generated, dialog.replace_existing()

    def remove_grids(self, grids: Sequence[KPointGrid]) -> None:
        """Remove the rows for ``grids`` and forget their parameters."""

        self.remove_keys(grids)

    # -- Project state pushed in -------------------------------------------------

    def set_cutoff_default(self, cutoff_energy: float | None) -> None:
        if cutoff_energy is None:
            self._default_params.pop("cut_off_energy", None)
        else:
            self._default_params["cut_off_energy"] = ValueWithUnit(float(cutoff_energy), "eV")
            self._drop_shared_default_conflicts("cut_off_energy")

    def _drop_shared_default_conflicts(self, explicit_name: str) -> None:
        """Drop shared defaults that are mutually exclusive with ``explicit_name``.

        Once a value is explicitly determined for a keyword (e.g. the
        project's cutoff energy convergence result), any other shared
        default that CASTEP treats as mutually exclusive with it (e.g. the
        ``basis_precision`` schema default) must be removed. Otherwise every
        unconfigured grid row inherits both values and fails validation with
        a "conflicts with" error until the user manually opens and confirms
        the row's Configure dialog.
        """
        spec = self._specs_by_name.get(explicit_name)
        if spec is None:
            return
        for conflict in spec.mutually_exclusive_with:
            self._default_params.pop(conflict, None)

    def set_generation_context(
        self,
        *,
        structure_id: str | None,
        lattice_vectors: list[list[float]] | None,
    ) -> None:
        self._current_cell_id = structure_id
        self._current_lattice = lattice_vectors
        self._refresh_generation_warning()

    def _refresh_generation_warning(self) -> None:
        if self._generation_lattice is None or self._current_lattice is None:
            self._warning_label.hide()
            return
        if lattice_matches(self._generation_lattice, self._current_lattice):
            self._warning_label.hide()
            return

        self._warning_label.setText(
            "K-point grids were generated for a different simulation cell. Regenerate grids before running new jobs."
        )
        self._warning_label.show()

    # -- Form state --------------------------------------------------------------

    def _extra_state(self) -> dict[str, Any]:
        return {
            "generated_for_cell_id": self._generation_cell_id,
            "generated_for_lattice": self._generation_lattice,
        }

    def _restore_extra_state(self, state: Mapping[str, Any]) -> None:
        generated_cell = state.get("generated_for_cell_id")
        self._generation_cell_id = str(generated_cell) if isinstance(generated_cell, str) else None

        generated_lattice = state.get("generated_for_lattice")
        if isinstance(generated_lattice, list) and len(generated_lattice) == 3:
            self._generation_lattice = [
                list(row) for row in generated_lattice if isinstance(row, list)
            ]
        else:
            self._generation_lattice = None
        self._refresh_generation_warning()
