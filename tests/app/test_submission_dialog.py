"""Tests for local submission dialog runtime options."""

from __future__ import annotations

from PySide6.QtWidgets import QDialogButtonBox

import mufinder.app.submission_dialog as submission_dialog_module
from mufinder.app.submission_dialog import LocalSubmissionDialog


def test_submission_dialog_single_job_defaults_to_sequential(
    qapp, monkeypatch
) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 8)

    dialog = LocalSubmissionDialog(job_labels=["conv_ecut_300"], mpirun_configured=True)
    config = dialog.config()

    assert config.num_cores == 1
    assert config.parallel is False
    assert config.max_concurrent == 1
    assert dialog._parallel_group.isVisible() is False
    dialog.close()


def test_submission_dialog_requires_single_core_when_mpirun_missing(
    qapp, monkeypatch
) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 16)

    dialog = LocalSubmissionDialog(
        job_labels=["conv_ecut_300", "conv_ecut_360"],
        mpirun_configured=False,
    )

    assert dialog._core_spin.minimum() == 1
    assert dialog._core_spin.maximum() == 1
    assert "mpirun" in dialog._core_hint.text().lower()
    dialog.close()


def test_submission_dialog_parallel_mode_reports_oversubscription(
    qapp, monkeypatch
) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 4)

    dialog = LocalSubmissionDialog(
        job_labels=["a", "b", "c"],
        mpirun_configured=True,
    )

    dialog._parallel_radio.setChecked(True)
    dialog._core_spin.setValue(3)
    dialog._max_concurrent_spin.setValue(2)
    config = dialog.config()

    assert config.parallel is True
    assert config.max_concurrent == 2
    assert config.num_cores == 3
    assert "oversubscription" in dialog._oversubscription_warning.text().lower()
    dialog.close()


def test_submission_dialog_default_height_avoids_squashed_layout(
    qapp, monkeypatch
) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 8)

    dialog = LocalSubmissionDialog(
        job_labels=["a", "b", "c"],
        mpirun_configured=True,
    )

    assert dialog.size().height() >= 420
    dialog.close()


def test_submission_dialog_blocks_when_farms_exceed_cores(qapp, monkeypatch) -> None:
    # CASTEP calls comms_abort() when farms outnumber MPI ranks, so this is a
    # certain failure rather than a slow run: OK must be unavailable.
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 16)

    dialog = LocalSubmissionDialog(
        job_labels=["barrier_siteA_to_siteB_neb"],
        mpirun_configured=True,
        farm_counts={"barrier_siteA_to_siteB_neb": 4},
    )
    dialog._core_spin.setValue(2)

    ok_button = dialog._button_box.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_button.isEnabled() is False
    assert "at least 4" in dialog._oversubscription_warning.text()

    dialog._core_spin.setValue(4)
    assert ok_button.isEnabled() is True
    dialog.close()


def test_submission_dialog_warns_when_farms_do_not_divide_cores(qapp, monkeypatch) -> None:
    # Ranks beyond floor(ranks / farms) * farms are assigned to no farm at all.
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 16)

    dialog = LocalSubmissionDialog(
        job_labels=["barrier_siteA_to_siteB_neb"],
        mpirun_configured=True,
        farm_counts={"barrier_siteA_to_siteB_neb": 3},
    )
    dialog._core_spin.setValue(8)

    ok_button = dialog._button_box.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_button.isEnabled() is True
    assert "idle" in dialog._oversubscription_warning.text().lower()
    dialog.close()


def test_submission_dialog_blocks_farmed_jobs_without_mpirun(qapp, monkeypatch) -> None:
    # Without mpirun the core count is pinned to 1, so any farmed job would abort.
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 16)

    dialog = LocalSubmissionDialog(
        job_labels=["barrier_siteA_to_siteB_neb"],
        mpirun_configured=False,
        farm_counts={"barrier_siteA_to_siteB_neb": 2},
    )

    ok_button = dialog._button_box.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_button.isEnabled() is False
    assert "mpirun" in dialog._oversubscription_warning.text().lower()
    dialog.close()


def test_submission_dialog_ignores_unfarmed_jobs(qapp, monkeypatch) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 8)

    dialog = LocalSubmissionDialog(
        job_labels=["relax_siteA"],
        mpirun_configured=True,
        farm_counts={"relax_siteA": 1},
    )

    ok_button = dialog._button_box.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_button.isEnabled() is True
    assert dialog._oversubscription_warning.text() == ""
    dialog.close()
