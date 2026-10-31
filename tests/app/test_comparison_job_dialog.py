"""Tests for pristine-relaxation comparison job selection dialog."""

from __future__ import annotations

from uuid import uuid4

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QMessageBox

from mufinder.app.comparison_job_dialog import (
    ComparisonJobDialog,
    CompletedPristineJob,
    run_comparison_job_dialog,
)


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


def test_comparison_job_dialog_starts_with_every_job_ticked(qapp) -> None:
    jobs = [_job("job-a"), _job("job-b")]
    dialog = ComparisonJobDialog(jobs=jobs, default_label="Comparison 1")

    assert dialog.selected_job_ids() == [jobs[0].job_id, jobs[1].job_id]
    assert dialog.table_label() == "Comparison 1"


def test_comparison_job_dialog_collects_only_the_jobs_left_ticked(qapp) -> None:
    jobs = [_job("job-a"), _job("job-b")]
    dialog = ComparisonJobDialog(jobs=jobs, default_label="Comparison 1")

    dialog._table.item(1, dialog._CHECK_COLUMN).setCheckState(Qt.CheckState.Unchecked)

    assert dialog.selected_job_ids() == [jobs[0].job_id]


def test_comparison_job_dialog_refuses_when_every_job_is_unticked(qapp, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda _parent, title, _text: warnings.append(title)
    )
    dialog = ComparisonJobDialog(jobs=[_job("job-a")], default_label="Comparison 1")
    dialog._table.item(0, dialog._CHECK_COLUMN).setCheckState(Qt.CheckState.Unchecked)

    dialog._accept_if_valid()

    assert warnings == ["No jobs selected"]
    assert dialog.result() != QDialog.DialogCode.Accepted


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


def test_run_comparison_job_dialog_returns_the_choice_or_none_when_cancelled(
    qapp, monkeypatch
) -> None:
    jobs = [_job("job-a"), _job("job-b")]

    def accept(dialog) -> int:
        dialog._table.item(1, dialog._CHECK_COLUMN).setCheckState(Qt.CheckState.Checked)
        dialog._table.item(0, dialog._CHECK_COLUMN).setCheckState(Qt.CheckState.Unchecked)
        dialog._label_input.setText("  Chosen  ")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ComparisonJobDialog, "exec", accept)
    assert run_comparison_job_dialog(jobs, "Comparison 1") == ([jobs[1].job_id], "Chosen")

    monkeypatch.setattr(ComparisonJobDialog, "exec", lambda _dialog: QDialog.DialogCode.Rejected)
    assert run_comparison_job_dialog(jobs, "Comparison 1") is None
