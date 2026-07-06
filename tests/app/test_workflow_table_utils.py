"""Regression tests for shared workflow table/button helpers.

The button tests assert the exact rendered text/tooltip for every wording
variant in use so that adopting the helper across widgets cannot silently
drift the visible text. The table tests pin down ``create_workflow_table``'s
default chrome (the DL-4 house style) plus every keyword-only option added
for the raw-``QTableWidget`` conversion sites, so future edits cannot
silently change the shared defaults out from under existing callers.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QAbstractItemView, QFrame, QHeaderView, QPushButton

from mufinder.app.workflow_table_utils import create_workflow_table, style_workflow_configure_button


@pytest.fixture()
def button(qapp) -> QPushButton:
    return QPushButton()


def test_create_workflow_table_default_chrome(qapp) -> None:
    table = create_workflow_table(("Alpha", "Beta"))
    assert table.frameShape() == QFrame.Shape.NoFrame
    assert table.wordWrap() is False
    assert table.alternatingRowColors() is True
    assert table.selectionBehavior() == QAbstractItemView.SelectionBehavior.SelectRows
    assert table.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert table.verticalHeader().isHidden() is True
    assert table.horizontalHeader().stretchLastSection() is True
    assert table.horizontalHeader().sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert table.property("squareTopCorners") == "true"
    assert table.property("tabbedTable") is None


def test_create_workflow_table_resize_to_contents_columns(qapp) -> None:
    table = create_workflow_table(("A", "B", "C"), resize_to_contents_columns=(1,))
    header = table.horizontalHeader()
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Stretch


def test_create_workflow_table_selection_overrides(qapp) -> None:
    table = create_workflow_table(
        ("A",),
        selection_mode=QAbstractItemView.SelectionMode.SingleSelection,
        selection_behavior=QAbstractItemView.SelectionBehavior.SelectItems,
    )
    assert table.selectionMode() == QAbstractItemView.SelectionMode.SingleSelection
    assert table.selectionBehavior() == QAbstractItemView.SelectionBehavior.SelectItems


def test_create_workflow_table_editable(qapp) -> None:
    table = create_workflow_table(("A",), editable=True)
    assert table.editTriggers() != QAbstractItemView.EditTrigger.NoEditTriggers


def test_create_workflow_table_show_vertical_header(qapp) -> None:
    table = create_workflow_table(("A",), show_vertical_header=True)
    assert table.verticalHeader().isHidden() is False


def test_create_workflow_table_tabbed_corners(qapp) -> None:
    table = create_workflow_table(("A",), corner_style="tabbed")
    assert table.property("squareTopCorners") == "true"
    assert table.property("tabbedTable") == "true"


def test_create_workflow_table_stretch_last_section_and_resize_mode(qapp) -> None:
    table = create_workflow_table(
        ("A", "B"),
        stretch_last_section=False,
        default_resize_mode=QHeaderView.ResizeMode.Interactive,
    )
    header = table.horizontalHeader()
    assert header.stretchLastSection() is False
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.Interactive


def test_configure_button_energy_barriers_defaults(button: QPushButton) -> None:
    # Adopter baseline: no noun/used → "for this row." with no "used".
    style_workflow_configure_button(
        button, workflow_label="Energy Barriers", configured=False, locked=False
    )
    assert button.text() == "Configure..."
    assert button.toolTip() == "Configure Energy Barriers parameters for this row."

    style_workflow_configure_button(
        button, workflow_label="Energy Barriers", configured=True, locked=False
    )
    assert button.text() == "Configured..."
    assert button.toolTip() == "Edit custom Energy Barriers parameters for this row."

    style_workflow_configure_button(
        button, workflow_label="Energy Barriers", configured=False, locked=True
    )
    assert button.text() == "Configured..."
    assert button.toolTip() == "Inspect the Energy Barriers parameters for this row."


def test_configure_button_quantum_corrections_defaults(button: QPushButton) -> None:
    style_workflow_configure_button(
        button, workflow_label="Quantum Corrections", configured=False, locked=True
    )
    assert button.text() == "Configured..."
    assert button.toolTip() == "Inspect the Quantum Corrections parameters for this row."


def test_configure_button_relaxation_wording(button: QPushButton) -> None:
    # Relaxation / muon relaxation: label "CASTEP", noun "row", "used" phrasing.
    kwargs = {"workflow_label": "CASTEP", "noun": "row", "used": True}

    style_workflow_configure_button(button, configured=False, locked=False, **kwargs)
    assert button.text() == "Configure..."
    assert button.toolTip() == "Configure CASTEP parameters for this row."

    style_workflow_configure_button(button, configured=True, locked=False, **kwargs)
    assert button.text() == "Configured..."
    assert button.toolTip() == "Edit custom CASTEP parameters for this row."

    style_workflow_configure_button(button, configured=False, locked=True, **kwargs)
    assert button.text() == "Configured..."
    assert button.toolTip() == "Inspect the CASTEP parameters used for this row."


def test_configure_button_uep_wording(button: QPushButton) -> None:
    kwargs = {"workflow_label": "CASTEP", "noun": "UEP source job", "used": True}

    style_workflow_configure_button(button, configured=False, locked=False, **kwargs)
    assert button.text() == "Configure..."
    assert button.toolTip() == "Configure CASTEP parameters for this UEP source job."

    style_workflow_configure_button(button, configured=True, locked=False, **kwargs)
    assert button.text() == "Configured..."
    assert button.toolTip() == "Edit custom CASTEP parameters for this UEP source job."

    style_workflow_configure_button(button, configured=False, locked=True, **kwargs)
    assert button.text() == "Configured..."
    assert button.toolTip() == "Inspect the CASTEP parameters used for this UEP source job."


def test_configure_button_locked_takes_priority_over_configured(button: QPushButton) -> None:
    style_workflow_configure_button(
        button, workflow_label="CASTEP", configured=True, locked=True, noun="row", used=True
    )
    assert button.text() == "Configured..."
    # Locked wins: inspect phrasing, not the edit phrasing.
    assert button.toolTip() == "Inspect the CASTEP parameters used for this row."
