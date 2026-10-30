"""Quantum Corrections actions in the main window: configuring rows and resetting their jobs."""

from __future__ import annotations

from pathlib import Path

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.services.project_service import ProjectService
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE
from tests.support.main_window import stage_quantum_row


def _failed_phonon_row(main_window, tmp_path: Path, *, legacy: bool):
    """Stage a row whose phonon job failed, with its seed directory on disk.

    ``legacy`` adds the completed singlepoint an older MuFinder version ran in
    the same directory first.
    """

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_reset.mufinder", "QuantumReset")
    window._document = doc
    definition, candidate, singlepoint = stage_quantum_row(
        window, service, doc, singlepoint_status=JobStatus.COMPLETED if legacy else None
    )
    definition.parameters["job_seed_names_by_label"] = {candidate.key: "quantum_siteA"}
    phonon = service.add_job(
        doc,
        label="quantum_phonon_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
    )
    phonon.status = JobStatus.FAILED
    phonon.execution_target = "archer2"
    phonon.remote_directory = "/work/quantum_siteA"
    definition.parameters["phonon_job_ids_by_label"] = {candidate.key: str(phonon.id)}
    directory = doc.project_root / "files" / "quantum_corrections" / "quantum_siteA"
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in (".cell", ".param", ".castep", ".check"):
        (directory / f"quantum_siteA{suffix}").write_text("data\n", encoding="utf-8")
    window._refresh_quantum_corrections_view()
    return window, doc, definition, candidate, singlepoint, phonon, directory


def _answer_cleanup(monkeypatch: pytest.MonkeyPatch, answer: bool | None) -> list[int]:
    """Answer the shared reset's clean-up prompt and record how many jobs it named."""

    asked: list[int] = []

    def prompt(_self, job_count: int) -> bool | None:
        asked.append(job_count)
        return answer

    monkeypatch.setattr(main_window_module.MainWindow, "_prompt_failed_job_cleanup", prompt)
    return asked


@pytest.mark.parametrize("legacy", [False, True])
def test_quantum_row_reset_with_cleanup_reruns_as_a_fresh_job(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    window, doc, _definition, candidate, singlepoint, phonon, directory = _failed_phonon_row(
        main_window, tmp_path, legacy=legacy
    )
    asked = _answer_cleanup(monkeypatch, True)

    window._quantum_corrections_view.reset_failed_requested.emit([candidate.key])

    assert asked == [1]
    assert phonon.status == JobStatus.DEFINED
    # The remote directory is forgotten, so a cluster resubmission goes to a fresh target.
    assert phonon.remote_directory is None
    assert sorted(path.name for path in directory.iterdir()) == [
        "quantum_siteA.cell",
        "quantum_siteA.param",
    ]
    if singlepoint is not None:
        assert doc.jobs[singlepoint.id].status == JobStatus.COMPLETED
    controller = window._quantum_corrections_controller
    assert controller.build_states()[candidate.key].status == "defined"
    request = controller.configure_request([candidate.key])
    assert request is not None and request.read_only is False
    window.close()


def test_quantum_row_reset_without_cleanup_keeps_the_checkpoint(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, _definition, candidate, _singlepoint, phonon, directory = _failed_phonon_row(
        main_window, tmp_path, legacy=False
    )
    _answer_cleanup(monkeypatch, False)

    window._quantum_corrections_view.reset_failed_requested.emit([candidate.key])

    assert phonon.status == JobStatus.DEFINED
    assert (directory / "quantum_siteA.check").read_text(encoding="utf-8") == "data\n"
    window.close()


def test_quantum_row_reset_is_not_applied_when_cancelled(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, _definition, candidate, _singlepoint, phonon, directory = _failed_phonon_row(
        main_window, tmp_path, legacy=True
    )
    asked = _answer_cleanup(monkeypatch, None)

    window._quantum_corrections_view.reset_failed_requested.emit([candidate.key])

    assert asked == [1]
    assert phonon.status == JobStatus.FAILED
    assert phonon.remote_directory == "/work/quantum_siteA"
    assert (directory / "quantum_siteA.check").exists()
    window.close()


def test_quantum_reset_button_reaches_the_shared_reset(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, _definition, _candidate, _singlepoint, phonon, _directory = _failed_phonon_row(
        main_window, tmp_path, legacy=False
    )
    asked = _answer_cleanup(monkeypatch, False)
    view = window._quantum_corrections_view

    view._sources_table.selectRow(0)
    assert view._reset_failed_button.isEnabled() is True
    view._reset_failed_button.click()

    assert asked == [1]
    assert phonon.status == JobStatus.DEFINED
    window.close()


class _FakeConfigDialog:
    """Records how a source Configure dialog was opened and accepts it."""

    def __init__(self, opened: list[dict[str, object]], *, num_farms: int, **kwargs) -> None:
        self._opened = opened
        self._num_farms = num_farms
        opened.append({"num_farms": num_farms, **kwargs})

    def setWindowTitle(self, title: str) -> None:  # noqa: N802 - Qt API naming.
        self._opened[-1]["title"] = title

    def exec(self):
        from PySide6.QtWidgets import QDialog

        return QDialog.DialogCode.Accepted

    def deleteLater(self) -> None:  # noqa: N802 - Qt API naming.
        self._opened[-1]["deleted"] = True

    def collect_values(self) -> dict[str, object]:
        return {}

    def advanced_names(self) -> tuple[str, ...]:
        return ()

    def untouched_mixed_names(self) -> set[str]:
        return set()

    def num_farms(self) -> int:
        return 4


def test_configuring_a_ready_row_edits_every_parameter(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.workflow_config_codecs import workflow_row_configuration

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_farms.mufinder", "QuantumFarms")
    window._document = doc
    definition, candidate, _singlepoint = stage_quantum_row(window, service, doc)
    opened: list[dict[str, object]] = []
    monkeypatch.setattr(
        main_window_module,
        "QuantumCorrectionsConfigDialog",
        lambda **kwargs: _FakeConfigDialog(opened, **kwargs),
    )

    window._quantum_corrections_view.configure_requested.emit([candidate.key])

    assert len(opened) == 1
    assert opened[0]["title"] == "Configure Quantum Corrections Parameters"
    assert opened[0]["read_only"] is False
    assert not {"editable_param_names", "num_farms_editable", "pseudopotential_editable"} & set(
        opened[0]
    )
    assert workflow_row_configuration(definition, candidate.key)[2] == {"num_farms": 4}
    assert "pseudopotential_family" not in opened[0]
    # Seeded through the constructor and deleted once it closes.
    assert {"values", "advanced_names", "mixed_names"} <= set(opened[0])
    assert opened[0]["deleted"] is True


def test_configuring_a_failed_row_is_read_only_until_it_is_reset(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, _definition, candidate, _singlepoint, _phonon, _directory = _failed_phonon_row(
        main_window, tmp_path, legacy=True
    )
    opened: list[dict[str, object]] = []
    monkeypatch.setattr(
        main_window_module,
        "QuantumCorrectionsConfigDialog",
        lambda **kwargs: _FakeConfigDialog(opened, **kwargs),
    )

    window._quantum_corrections_view.configure_requested.emit([candidate.key])
    assert opened[-1]["read_only"] is True
    assert "use Reset Failed Jobs… to edit" in str(opened[-1]["intro_text"])

    _answer_cleanup(monkeypatch, False)
    window._quantum_corrections_view.reset_failed_requested.emit([candidate.key])
    window._quantum_corrections_view.configure_requested.emit([candidate.key])

    assert opened[-1]["read_only"] is False
    window.close()
