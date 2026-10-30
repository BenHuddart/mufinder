"""Dialog presenting suggested supercell sizes for muon stopping-site calculations."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from mufinder.app.theme import style_warning_text
from mufinder.app.workflow_table_utils import (
    create_workflow_table,
    set_header_tooltips,
    set_workflow_table_item,
)
from mufinder.domain.text import plural
from mufinder.science.supercell_suggestion import SupercellSuggestion


def _scaling_label(scaling: tuple[int, int, int]) -> str:
    return f"{scaling[0]} × {scaling[1]} × {scaling[2]}"


def _distance_label(distance: float, reaches_target: bool) -> str:
    tick = " ✓" if reaches_target else ""
    return f"{distance:.2f} Å{tick}"


def _intro_text(suggestion: SupercellSuggestion) -> str:
    return (
        "Diagonal supercells whose periodic muon images are at least "
        f"{suggestion.display_threshold:.4g} Å apart. The recommended cell is the cheapest one "
        f"that reaches {suggestion.distance_target:.4g} Å (✓) with fewer than "
        f"{plural(suggestion.max_atoms, 'atom')}."
    )


def _status_text(suggestion: SupercellSuggestion) -> str:
    """Return the warning to show under the table, or an empty string when there is none."""
    target = f"{suggestion.distance_target:.4g} Å"
    budget = plural(suggestion.max_atoms, "atom")
    recommended = suggestion.candidates[suggestion.recommended_index]
    if suggestion.primitive_exceeds_budget:
        misses_target = "" if recommended.satisfies_criterion else f" It does not reach {target}."
        return (
            f"The primitive cell already has {plural(suggestion.primitive_atom_count, 'atom')}, "
            f"so no search under {budget} was possible. This calculation will be expensive."
            f"{misses_target}"
        )
    if not recommended.satisfies_criterion:
        return (
            f"No cell under {budget} reaches {target}. "
            "The recommended cell is the best one under that limit."
        )
    if recommended.atom_count >= suggestion.max_atoms:
        return (
            f"No cell under {budget} reaches {target}. The recommended cell is the smallest "
            f"that does, and its {plural(recommended.atom_count, 'atom')} may be expensive."
        )
    return ""


class SupercellSuggestionDialog(QDialog):
    """Present Pareto-optimal candidate supercells and let the user confirm a choice."""

    _SCALING_COLUMN = 0
    _ATOMS_COLUMN = 1
    _DISTANCE_COLUMN = 2

    def __init__(self, *, suggestion: SupercellSuggestion, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Suggest Supercell")
        self.resize(520, 300)

        self._suggestion = suggestion
        self._chosen_scaling: tuple[int, int, int] | None = None
        self._table = None

        self._build_layout()

    def _build_layout(self) -> None:
        s = self._suggestion
        layout = QVBoxLayout(self)

        # With the primitive cell over budget nothing was searched, so the intro would not
        # describe the single row; the status line says what happened instead.
        if not s.primitive_exceeds_budget:
            intro = QLabel(_intro_text(s))
            intro.setWordWrap(True)
            layout.addWidget(intro)

        self._build_candidate_table(layout)

        status = _status_text(s)
        if status:
            status_label = QLabel(status)
            status_label.setWordWrap(True)
            style_warning_text(status_label)
            layout.addWidget(status_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Use This Supercell")
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_candidate_table(self, layout: QVBoxLayout) -> None:
        s = self._suggestion
        self._table = create_workflow_table(
            ("Scaling", "Atoms", "Min image (Å)"),
            resize_to_contents_columns=(self._ATOMS_COLUMN,),
            accessible_name="Supercell Candidates",
        )
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setRowCount(len(s.candidates))
        set_header_tooltips(
            self._table,
            {
                "Min image (Å)": (
                    "The shortest distance from the muon to one of its periodic images. "
                    f"A tick marks a cell that reaches {s.distance_target:.4g} Å."
                )
            },
        )

        for row, cand in enumerate(s.candidates):
            scaling = _scaling_label(cand.scaling)
            if row == s.recommended_index:
                scaling = f"{scaling} (Recommended)"
            set_workflow_table_item(self._table, row, self._SCALING_COLUMN, scaling)
            set_workflow_table_item(self._table, row, self._ATOMS_COLUMN, str(cand.atom_count))
            set_workflow_table_item(
                self._table,
                row,
                self._DISTANCE_COLUMN,
                _distance_label(cand.min_image_distance, cand.satisfies_criterion),
            )

        self._table.selectRow(s.recommended_index)
        layout.addWidget(self._table, 1)

    def _accept_if_valid(self) -> None:
        if self._table is not None:
            selected = self._table.selectedItems()
            if not selected:
                QMessageBox.warning(self, "No selection", "Select a candidate supercell.")
                return
            self._chosen_scaling = self._suggestion.candidates[selected[0].row()].scaling
        self.accept()

    def chosen_scaling(self) -> tuple[int, int, int] | None:
        """Return the confirmed diagonal scaling, or None if the dialog was not accepted."""
        return self._chosen_scaling

    def chosen_matrix(self) -> list[list[int]] | None:
        """Return the confirmed diagonal matrix, or None if the dialog was not accepted."""
        if self._chosen_scaling is None:
            return None
        n1, n2, n3 = self._chosen_scaling
        return [[n1, 0, 0], [0, n2, 0], [0, 0, n3]]
