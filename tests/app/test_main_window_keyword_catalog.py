"""Main-window wiring for the CASTEP keyword-catalog indicator and warnings."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.castep_setup import CastepSetup
from mufinder.domain.enums import JobStatus
from mufinder.services.project_service import ProjectService


def _window_with_job(tmp_path: Path):
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "catalog.mufinder", "Catalog")
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc
    return window, job


def test_status_bar_shows_the_catalog_indicator(qapp) -> None:
    window = MainWindow()

    button = window._keyword_catalog_guard.status_button
    assert button.parent() is not None
    assert button in window.statusBar().children()
    assert button.text().endswith("CASTEP keywords: limited built-in set")
    window.close()


def test_closing_castep_setup_refreshes_the_indicator_even_when_cancelled(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A catalog installed in the dialog is live even if the dialog is cancelled."""

    window = MainWindow()
    refreshed: list[bool] = []
    monkeypatch.setattr(window._keyword_catalog_guard, "refresh", lambda: refreshed.append(True))

    class _CancelledDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, *args, **kwargs) -> None:
            pass

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(main_window_module, "CastepSetupDialog", _CancelledDialog)
    window.statusBar().showMessage("before")

    window._keyword_catalog_guard.status_button.click()

    assert refreshed == [True]
    assert window.statusBar().currentMessage() == "before"
    window.close()


def test_startup_prompt_delegates_to_the_guard(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    window = MainWindow()
    prompted: list[bool] = []
    monkeypatch.setattr(window._keyword_catalog_guard, "prompt_at_startup_if_needed", lambda: prompted.append(True))

    window.prompt_for_keyword_catalog_if_needed()

    assert prompted == [True]
    window.close()


@pytest.mark.parametrize("allowed", [False, True])
def test_local_submission_respects_the_catalog_warning(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allowed: bool,
) -> None:
    window, job = _window_with_job(tmp_path)
    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/usr/bin/castep"),
    )
    monkeypatch.setattr(window._keyword_catalog_guard, "confirm_submission", lambda: allowed)
    dialogs: list[bool] = []

    class _Dialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            dialogs.append(True)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _Dialog)

    window._submit_jobs_locally([job])

    # Declining the warning stops before the submission dialog; accepting it
    # carries on exactly as before.
    assert dialogs == ([True] if allowed else [])
    assert job.status == JobStatus.DEFINED
    window.close()


@pytest.mark.parametrize("allowed", [False, True])
def test_cluster_submission_respects_the_catalog_warning(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allowed: bool,
) -> None:
    window, job = _window_with_job(tmp_path)
    monkeypatch.setattr(window._keyword_catalog_guard, "confirm_submission", lambda: allowed)
    dialogs: list[bool] = []

    class _Dialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            dialogs.append(True)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _Dialog)

    window._submit_jobs_on_cluster([job])

    assert dialogs == ([True] if allowed else [])
    assert job.status == JobStatus.DEFINED
    window.close()
