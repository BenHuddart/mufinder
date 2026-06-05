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
from mufinder.app.workflow_table_utils import create_workflow_table, set_workflow_table_item
from mufinder.science.supercell_suggestion import SupercellSuggestion


def _scaling_label(scaling: tuple[int, int, int]) -> str:
    return f"{scaling[0]} × {scaling[1]} × {scaling[2]}"


def _distance_label(distance: float) -> str:
    return f"{distance:.2f} Å"


def _criterion_label(satisfies: bool) -> str:
    return "Yes" if satisfies else "No"


class SupercellSuggestionDialog(QDialog):
    """Present Pareto-optimal candidate supercells and let the user confirm a choice."""

    _TAG_COLUMN = 0
    _SCALING_COLUMN = 1
    _ATOMS_COLUMN = 2
    _DISTANCE_COLUMN = 3
    _CRITERION_COLUMN = 4

    def __init__(self, *, suggestion: SupercellSuggestion, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Suggest Supercell")
        self.resize(660, 420)

        self._suggestion = suggestion
        self._chosen_scaling: tuple[int, int, int] | None = None
        self._table = None

        self._build_layout()

    def _build_layout(self) -> None:
        s = self._suggestion
        layout = QVBoxLayout(self)

        intro = QLabel(
            "The muon is modelled as a charged interstitial defect in a periodic supercell. "
            "If the cell is too small the muon spuriously interacts with its own periodic "
            "images — through both Coulomb and lattice-strain coupling — biasing stopping-site "
            "energetics. "
            f"The standard rule of thumb is a minimum image distance of at least "
            f"{s.distance_target:.4g} Å, "
            "where the distance is the shortest lattice translation over all "
            "(n₁, n₂, n₃) ∈ {-1, 0, 1}³ "
            "excluding the origin — not simply the shortest cell edge. "
            f"All cells with minimum image distance ≥ {s.display_threshold:.4g} Å are shown "
            "below; each row strictly improves on cheaper options."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._build_candidate_table(layout)

        if s.primitive_exceeds_budget:
            warning = QLabel(
                f"The primitive cell already contains {s.primitive_atom_count} "
                f"atoms (≥ {s.max_atoms}); this calculation will be expensive. "
                "No scaling search was performed."
            )
            warning.setWordWrap(True)
            style_warning_text(warning)
            layout.addWidget(warning)
        elif not s.candidates[s.recommended_index].satisfies_criterion:
            warning = QLabel(
                f"No cell under {s.max_atoms} atoms meets the "
                f"≥ {s.distance_target:.4g} Å criterion. "
                "The highlighted option is the best available within budget."
            )
            warning.setWordWrap(True)
            style_warning_text(warning)
            layout.addWidget(warning)
        elif s.candidates[s.recommended_index].atom_count >= s.max_atoms:
            warning = QLabel(
                f"The ≥ {s.distance_target:.4g} Å criterion cell has "
                f"{s.candidates[s.recommended_index].atom_count} atoms "
                f"(≥ {s.max_atoms}) and may be expensive to compute."
            )
            warning.setWordWrap(True)
            style_warning_text(warning)
            layout.addWidget(warning)

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
            ("", "Scaling", "Atoms", "Min image (Å)", f"≥ {s.distance_target:.4g} Å"),
            resize_to_contents_columns=(0, 2, 4),
        )
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setRowCount(len(s.candidates))

        for row, cand in enumerate(s.candidates):
            tag = "(Recommended)" if row == s.recommended_index else ""
            set_workflow_table_item(self._table, row, self._TAG_COLUMN, tag)
            set_workflow_table_item(self._table, row, self._SCALING_COLUMN, _scaling_label(cand.scaling))
            set_workflow_table_item(self._table, row, self._ATOMS_COLUMN, str(cand.atom_count))
            set_workflow_table_item(self._table, row, self._DISTANCE_COLUMN, _distance_label(cand.min_image_distance))
            set_workflow_table_item(self._table, row, self._CRITERION_COLUMN, _criterion_label(cand.satisfies_criterion))

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
