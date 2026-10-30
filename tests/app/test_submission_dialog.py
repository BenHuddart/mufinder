"""Tests for local submission dialog runtime options."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDialogButtonBox

import mufinder.app.submission_dialog as submission_dialog_module
from mufinder.app.submission_dialog import LocalSubmissionDialog


@pytest.fixture(autouse=True)
def _unknown_physical_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    # The test machine's topology must not leak into the default; tests that
    # need a physical count set one.
    monkeypatch.setattr(submission_dialog_module, "physical_core_count", lambda: None)


def _run_button(dialog: LocalSubmissionDialog):
    return dialog._button_box.button(QDialogButtonBox.StandardButton.Ok)


def test_submission_dialog_first_time_offers_every_logical_core(qapp, monkeypatch) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 8)

    dialog = LocalSubmissionDialog(job_labels=["conv_ecut_300"], mpirun_configured=True)
    config = dialog.config()

    assert config.num_cores == 8
    assert config.parallel is False
    assert config.max_concurrent == 1
    assert dialog._jobs_at_once_spin.isVisibleTo(dialog) is False
    assert dialog._cores_line.text() == "Uses 8 of 8 cores"
    assert _run_button(dialog).text() == "Run 1 job"
    dialog.close()


def test_submission_dialog_first_time_offers_every_physical_core(qapp, monkeypatch) -> None:
    # OpenMPI refuses more ranks than physical cores, so two threads per core
    # must not double the default.
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 20)
    monkeypatch.setattr(submission_dialog_module, "physical_core_count", lambda: 14)

    dialog = LocalSubmissionDialog(job_labels=["a", "b"], mpirun_configured=True)

    assert dialog.config().num_cores == 14
    assert dialog._core_spin.maximum() == 20
    assert dialog._cores_line.text() == "Uses 14 of 20 cores"
    dialog.close()


def test_submission_dialog_offers_the_remembered_core_count(qapp, monkeypatch) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 8)

    dialog = LocalSubmissionDialog(job_labels=["a", "b"], mpirun_configured=True, initial_cores=3)

    assert dialog.config().num_cores == 3
    dialog.close()


def test_submission_dialog_clamps_a_remembered_count_above_this_computer(qapp, monkeypatch) -> None:
    # A project opened on a smaller computer than the one it last ran on.
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 4)

    dialog = LocalSubmissionDialog(job_labels=["a"], mpirun_configured=True, initial_cores=16)

    assert dialog.config().num_cores == 4
    dialog.close()


def test_submission_dialog_requires_single_core_when_mpirun_missing(qapp, monkeypatch) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 16)

    dialog = LocalSubmissionDialog(
        job_labels=["conv_ecut_300", "conv_ecut_360"],
        mpirun_configured=False,
        initial_cores=8,
    )

    assert dialog._core_spin.minimum() == 1
    assert dialog._core_spin.maximum() == 1
    assert dialog._mpirun_hint is not None
    assert "mpirun" in dialog._mpirun_hint.text()
    dialog.close()


def test_submission_dialog_jobs_at_once_maps_to_the_config(qapp, monkeypatch) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 20)

    dialog = LocalSubmissionDialog(job_labels=["a", "b", "c"], mpirun_configured=True)

    assert dialog._jobs_at_once_spin.isVisibleTo(dialog) is True
    assert dialog._jobs_at_once_spin.maximum() == 3
    config = dialog.config()
    assert (config.parallel, config.max_concurrent) == (False, 1)

    dialog._core_spin.setValue(4)
    dialog._jobs_at_once_spin.setValue(3)
    config = dialog.config()

    assert config.num_cores == 4
    assert (config.parallel, config.max_concurrent) == (True, 3)
    assert dialog._cores_line.text() == "Uses 12 of 20 cores"
    assert dialog._cores_line.objectName() == "hintText"
    dialog.close()


def test_submission_dialog_cores_line_warns_above_the_logical_count(qapp, monkeypatch) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 4)

    dialog = LocalSubmissionDialog(job_labels=["a", "b", "c"], mpirun_configured=True)
    dialog._core_spin.setValue(3)
    dialog._jobs_at_once_spin.setValue(2)

    assert dialog._cores_line.text().startswith("Uses 6 of 4 cores.")
    assert dialog._cores_line.objectName() == "warningText"
    assert _run_button(dialog).isEnabled() is True

    dialog._jobs_at_once_spin.setValue(1)
    assert dialog._cores_line.text() == "Uses 3 of 4 cores"
    assert dialog._cores_line.objectName() == "hintText"
    dialog.close()


def test_submission_dialog_heading_names_the_stage_and_count(qapp, monkeypatch) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 8)
    labels = [f"cutoff_{value}eV" for value in range(300, 900, 100)]

    dialog = LocalSubmissionDialog(
        job_labels=labels, mpirun_configured=True, stage_title="Cutoff Energy"
    )

    assert dialog._heading.text() == "Run 6 Cutoff Energy jobs"
    assert dialog._heading.toolTip().splitlines() == labels
    assert dialog._heading.accessibleDescription() == ", ".join(labels)
    assert _run_button(dialog).text() == "Run 6 jobs"
    dialog.close()


def test_submission_dialog_heading_keeps_many_labels_out_of_the_text(qapp, monkeypatch) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 8)
    labels = [f"relax_{index:03d}" for index in range(100)]

    dialog = LocalSubmissionDialog(job_labels=labels, mpirun_configured=True)

    assert dialog._heading.text() == "Run 100 jobs"
    tooltip = dialog._heading.toolTip().splitlines()
    assert tooltip[:20] == labels[:20]
    assert tooltip[-1] == "and 80 more"
    assert len(tooltip) == 21
    dialog.close()


def test_submission_dialog_blocks_when_farms_exceed_cores(qapp, monkeypatch) -> None:
    # CASTEP calls comms_abort() when farms outnumber MPI ranks, so this is a
    # certain failure rather than a slow run: the run button must be unavailable.
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 16)

    dialog = LocalSubmissionDialog(
        job_labels=["barrier_siteA_to_siteB_neb"],
        mpirun_configured=True,
        farm_counts={"barrier_siteA_to_siteB_neb": 4},
    )
    dialog._core_spin.setValue(2)

    run_button = _run_button(dialog)
    assert run_button.isEnabled() is False
    assert "at least 4" in dialog._farm_warning.text()

    dialog._core_spin.setValue(4)
    assert run_button.isEnabled() is True
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

    assert _run_button(dialog).isEnabled() is True
    assert "idle" in dialog._farm_warning.text().lower()
    dialog.close()


def test_submission_dialog_blocks_farmed_jobs_without_mpirun(qapp, monkeypatch) -> None:
    # Without mpirun the core count is pinned to 1, so any farmed job would abort.
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 16)

    dialog = LocalSubmissionDialog(
        job_labels=["barrier_siteA_to_siteB_neb"],
        mpirun_configured=False,
        farm_counts={"barrier_siteA_to_siteB_neb": 2},
    )

    assert _run_button(dialog).isEnabled() is False
    assert "mpirun" in dialog._farm_warning.text().lower()
    dialog.close()


def test_submission_dialog_ignores_unfarmed_jobs(qapp, monkeypatch) -> None:
    monkeypatch.setattr(submission_dialog_module.QThread, "idealThreadCount", lambda: 8)

    dialog = LocalSubmissionDialog(
        job_labels=["relax_siteA"],
        mpirun_configured=True,
        farm_counts={"relax_siteA": 1},
    )

    assert _run_button(dialog).isEnabled() is True
    assert dialog._farm_warning.text() == ""
    assert dialog._farm_warning.isVisibleTo(dialog) is False
    dialog.close()
