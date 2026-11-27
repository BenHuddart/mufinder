"""A Run stores the page's configuration only when it differs from the active one.

Pressing Run on this computer or Run on a cluster on a row page first stores
the page's configuration. A configuration equal to the active definition's, in
its persisted form and for the same input structure and output folder,
registers no new definition and writes no "Configured …" Journal record, so a
saved project keeps no unsaved changes when the dialog is then cancelled. A
changed configuration is registered and recorded as before. Each project is
configured on the page, saved and reopened as File > Open leaves it first.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.app.main_window import MainWindow
from mufinder.services.project_service import ProjectService, UnsavedChanges
from tests.support.main_window import (
    fake_local_submission,
    open_project_with_host,
    stage_muon_relaxation_row,
)
from tests.support.submission_snapshot import ProjectSnapshot, cancel_submission, reopen_saved

_SUPERCELL = ((2, 0, 0), (0, 1, 0), (0, 0, 1))


def _row_page(view_name: str) -> Callable[[MainWindow, Path], object]:
    """Open a project with a host and add one row to the page ``view_name`` names."""

    def build(window: MainWindow, tmp_path: Path) -> object:
        open_project_with_host(window, tmp_path)
        view = getattr(window, view_name)
        view._add_row()
        return view

    return build


def _kpoint_page(window: MainWindow, tmp_path: Path) -> object:
    open_project_with_host(window, tmp_path)
    view = window._kpoint_convergence_view
    view._add_row((2, 2, 2))
    return view


def _muon_page(window: MainWindow, tmp_path: Path) -> object:
    service = ProjectService()
    document = service.create_project(tmp_path / "muon.mufinder", "Muon")
    stage_muon_relaxation_row(window, service, document)
    window._document = document
    window._refresh_ui_after_project_change()
    return window._muon_relaxation_view


def _change_supercell(window: MainWindow, view) -> None:
    document = window._document
    host = document.structures[document.project.host_structure_id]
    view.set_host_lattice(host.lattice_vectors)
    view.set_row_supercell(view._row_id_at(0), _SUPERCELL)


def _add_grid(_window: MainWindow, view) -> None:
    view._add_row((3, 3, 3))


#: Each page with a configure-on-Run: how to open it with one row, how to
#: change its configuration, the Journal action of a stored configuration and
#: the UI-state key naming the page's active definition.
_PAGES = {
    "pristine relaxation": (
        _row_page("_pristine_relaxation_view"),
        _change_supercell,
        "configure_pristine_relaxation",
        "active_pristine_relaxation_definition_id",
    ),
    "muon relaxation": (
        _muon_page,
        _change_supercell,
        "configure_muon_relaxation",
        "active_muon_relaxation_definition_id",
    ),
    "uep": (
        _row_page("_uep_view"),
        _change_supercell,
        "configure_uep",
        "active_uep_definition_id",
    ),
    "k-point convergence": (
        _kpoint_page,
        _add_grid,
        "configure_convergence_kpoint",
        "active_convergence_kpoint_definition_id",
    ),
    "cutoff convergence": (
        _row_page("_convergence_view"),
        lambda _window, view: view._add_row(),
        "configure_convergence_cutoff",
        "active_convergence_cutoff_definition_id",
    ),
}

#: The Run button each submission target is pressed with.
_RUN_BUTTONS = {"local": "_run_local_button", "alpha": "_run_cluster_button"}


@pytest.fixture
def configured_page(main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request):
    """``(window, view, page)``: the page configured by one cancelled Run, then saved and reopened."""

    page = request.param
    window = main_window
    view = _PAGES[page][0](window, tmp_path)
    cancel_submission(window, monkeypatch, "local")
    view._run_local_button.click()
    assert window._document.calculations, "the first Run stores the page's configuration"
    reopen_saved(window, window._document)
    return window, view, page


@pytest.mark.parametrize("target", list(_RUN_BUTTONS))
@pytest.mark.parametrize("configured_page", list(_PAGES), indirect=True)
def test_a_cancelled_run_of_an_unchanged_page_stores_no_configuration(
    configured_page, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    window, view, _page = configured_page
    before = ProjectSnapshot.take(window)
    opened = cancel_submission(window, monkeypatch, target)

    getattr(view, _RUN_BUTTONS[target]).click()

    assert opened == ["local" if target == "local" else "cluster"]
    after = ProjectSnapshot.take(window)
    assert after.definitions == before.definitions
    assert before.new_journal_records(after) == ()
    assert after.journal_on_disk == before.journal_on_disk
    assert after.unsaved is before.unsaved is UnsavedChanges.NONE


@pytest.mark.parametrize("configured_page", list(_PAGES), indirect=True)
def test_a_run_of_a_changed_page_stores_its_configuration_and_records_it_once_accepted(
    configured_page, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, view, page = configured_page
    _build, change, action, active_key = _PAGES[page]
    change(window, view)
    before = ProjectSnapshot.take(window)
    cancel_submission(window, monkeypatch, "local")

    view._run_local_button.click()

    after = ProjectSnapshot.take(window)
    (new_definition,) = after.definitions - before.definitions
    # Every page holds the record until a Run is accepted
    # (test_*_planned_jobs.py); the Run here is cancelled.
    assert [record[0] for record in before.new_journal_records(after)] == []
    assert window._document.ui_state[active_key] == str(new_definition)

    fake_local_submission(window, monkeypatch)
    view._run_local_button.click()

    accepted = ProjectSnapshot.take(window)
    assert [record[0] for record in after.new_journal_records(accepted)][:1] == [action]


@pytest.mark.parametrize("configured_page", ["muon relaxation"], indirect=True)
def test_a_run_registers_again_when_the_active_definition_names_a_site_no_row_has(
    configured_page, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row removed after its Run leaves the definition naming its site; the next Run renames them."""

    window, view, _page = configured_page
    active = window._muon_relaxation_controller.active_definition()
    row_sites = list(active.input_site_ids)
    active.input_site_ids.append(uuid4())
    before = ProjectSnapshot.take(window)
    cancel_submission(window, monkeypatch, "local")

    view._run_local_button.click()

    after = ProjectSnapshot.take(window)
    (new_definition,) = after.definitions - before.definitions
    assert window._document.calculations[new_definition].input_site_ids == row_sites
