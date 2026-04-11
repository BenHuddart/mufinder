"""Tests for the supercell preview dialog UI and generation behavior."""

from __future__ import annotations

from mufinder.app.supercell_dialog import SupercellPreviewDialog
from mufinder.domain.models import Structure


def _host_structure() -> Structure:
    return Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"pbc": [True, True, True]},
    )


def test_supercell_dialog_simple_mode_matrix(qapp) -> None:
    dialog = SupercellPreviewDialog(
        structure=_host_structure(),
        atomic_spins=[1.0, -1.0],
        propagation_vector=[0.0, 0.0, 0.0],
        initial_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 3]],
    )

    assert not dialog._advanced_toggle.isChecked()
    assert dialog.matrix() == [[2, 0, 0], [0, 1, 0], [0, 0, 3]]
    assert dialog._viewer._atoms is not None
    assert len(dialog._viewer._atoms) == 12

    dialog.close()


def test_supercell_dialog_advanced_mode_matrix(qapp) -> None:
    dialog = SupercellPreviewDialog(
        structure=_host_structure(),
        atomic_spins=[1.0, -1.0],
        propagation_vector=[0.0, 0.0, 0.5],
        initial_matrix=[[1, 1, 0], [0, 1, 0], [0, 0, 1]],
    )

    assert dialog._advanced_toggle.isChecked()
    assert "background: transparent" in dialog._advanced_toggle.styleSheet()
    assert dialog.matrix() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]

    dialog._advanced_cells[0][0].setValue(2)
    dialog._generate_preview()

    assert dialog._viewer._atoms is not None
    assert len(dialog._viewer._atoms) >= 2

    dialog.close()


def test_supercell_dialog_supports_unit_cell_action(qapp) -> None:
    dialog = SupercellPreviewDialog(
        structure=_host_structure(),
        atomic_spins=[1.0, -1.0],
        propagation_vector=[0.0, 0.0, 0.0],
        accept_button_text="Save Supercell",
        show_unit_cell_button=True,
    )

    dialog._accept_unit_cell()

    assert dialog.result() == dialog.DialogCode.Accepted
    assert dialog.result_mode() == "unit_cell"
    dialog.close()
