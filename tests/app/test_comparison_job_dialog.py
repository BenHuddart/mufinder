"""Tests for pristine-relaxation comparison job selection dialog."""

from __future__ import annotations

from uuid import uuid4

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from mufinder.app.comparison_job_dialog import ComparisonJobDialog, CompletedPristineJob


def _job(label: str) -> CompletedPristineJob:
    return CompletedPristineJob(
        job_id=uuid4(),
        job_label=label,
        structure_id=uuid4(),
        supercell_label="Unit cell",
        status="completed",
        total_energy=-10.0,
        lattice_a=5.0,
        lattice_b=5.0,
        lattice_c=5.0,
        parameter_values={"cut_off_energy": 600},
    )


def test_comparison_job_dialog_collects_checked_job_ids(qapp) -> None:
    jobs = [_job("job-a"), _job("job-b")]
    dialog = ComparisonJobDialog(jobs=jobs, default_label="Comparison 1")

    first = dialog._table.item(0, dialog._CHECK_COLUMN)
    assert first is not None
    first.setCheckState(Qt.CheckState.Checked)

    selected = dialog.selected_job_ids()

    assert selected == [jobs[0].job_id]
    assert dialog.table_label() == "Comparison 1"


def test_comparison_job_dialog_trims_table_label(qapp) -> None:
    jobs = [_job("job-a")]
    dialog = ComparisonJobDialog(jobs=jobs, default_label="  Demo  ")

    assert dialog.table_label() == "Demo"


def test_comparison_job_dialog_label_field_has_its_visible_label_as_buddy(qapp) -> None:
    dialog = ComparisonJobDialog(jobs=[_job("job-a")], default_label="Comparison 1")

    buddies = {
        label.buddy(): label.text() for label in dialog.findChildren(QLabel) if label.buddy()
    }

    assert buddies[dialog._label_input] == "Comparison table label"
    assert dialog._table.accessibleName() == "Comparison Jobs"
