"""A dashboard refresh that changes nothing a structure viewer shows rebuilds no viewer scene.

The window refreshes its pages whenever the Job Dashboard flushes (a job
changes state, a queue poll comes back). These tests open a project with each
page or window that holds a structure viewer populated, show it, let it settle
and then run two flushes that change nothing, counting full scene rebuilds per
viewer. A modal dialog is measured while it is open, since a flush can run
from its event loop.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog, QMessageBox, QPushButton, QWidget

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.magnetic_structure_builder import MagneticStructureBuilderDialog
from mufinder.app.main_window import MainWindow
from mufinder.app.relaxation_analysis_widget import (
    DisplacementAnalysisDialog,
    RelaxationClusteringDialog,
    default_workspace_state,
)
from mufinder.app.supercell_dialog import SupercellEditorDialog
from mufinder.domain.charge_states import POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.enums import JobStatus
from mufinder.services.project_service import ProjectService
from tests.support.energy_barriers import (
    NEB_TS_FIXTURE,
    add_band,
    seed_energy_barriers_document,
)
from tests.support.main_window import flush_dashboard, select_route
from tests.support.project_home import muon_project, pin_muon_sites
from tests.support.pyvista_fakes import count_scene_renders
from tests.support.qt import pump_until
from tests.support.quantum_corrections import (
    QUANTUM_PHONON_TEXT,
    bind_phonon,
    seed_quantum_document,
    store_phonon_result,
)
from tests.support.uep_sources import completed_uep_job_with_grid

_PINNED_SITES = [
    {"cluster_index": 1, "fractional_position": [0.5, 0.25, 0.25], "relative_energy_ev": 0.0},
    {"cluster_index": 2, "fractional_position": [0.25, 0.5, 0.75], "relative_energy_ev": 0.18},
]


@pytest.fixture
def renders(monkeypatch: pytest.MonkeyPatch) -> Counter:
    return count_scene_renders(monkeypatch)


@pytest.fixture(autouse=True)
def _no_message_boxes(monkeypatch: pytest.MonkeyPatch):
    """Fail on a message box, which would otherwise wait for a click.

    The box is refused at once and recorded, and the test fails at teardown, so
    a box opened from a Qt slot (whose exception Qt only prints) still fails it.
    """

    refused: list[object] = []

    def _refuse(*args, **_kwargs):
        refused.append(args[1:3])
        raise AssertionError(f"unexpected message box: {args[1:3]}")

    for name in ("information", "warning", "critical", "question", "exec"):
        monkeypatch.setattr(QMessageBox, name, _refuse)
    yield
    assert refused == []


@pytest.fixture
def window(qapp):
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    qapp.processEvents()
    yield window
    window.close()


def _open(qapp, window: MainWindow, document) -> None:
    ProjectService().save_project(document)
    assert window.open_project(document.project_root)
    _settle(qapp)


def _open_muon_project(qapp, window: MainWindow, tmp_path: Path) -> None:
    """A host with two muon sites, one relaxed, pinned sites and an analysis workspace."""

    project = muon_project(tmp_path)
    document = project.document
    # The relaxation rows carry their start positions, so the relaxed site is an analysis source.
    for row, site in zip(
        project.definition.parameters["configuration"]["rows"],
        (project.relaxed_site, project.running_site),
        strict=True,
    ):
        row["initial_fractional_position"] = list(site.initial_fractional_position)
    pin_muon_sites(document, project.host.id, POSITIVE_MUON_CHARGE_STATE, _PINNED_SITES)
    document.ui_state["relaxation_analysis_workspaces"] = [
        default_workspace_state(
            name="Analysis",
            charge_state=POSITIVE_MUON_CHARGE_STATE,
            host_structure_id=project.host.id,
            job_ids=[project.relaxed_job.id],
        )
    ]
    _open(qapp, window, document)


def _settle(qapp) -> None:
    for _ in range(3):
        qapp.processEvents()


def _shown_viewers(*roots: QWidget) -> list[AtomicStructureViewer]:
    return [
        viewer
        for root in roots
        for viewer in root.findChildren(AtomicStructureViewer)
        if viewer.isVisible()
    ]


def _visible_viewers(*roots: QWidget) -> list[AtomicStructureViewer]:
    viewers = _shown_viewers(*roots)
    assert viewers
    assert all(viewer.has_structure() for viewer in viewers)
    return viewers


def _rebuilds_over_two_unchanged_refreshes(
    qapp, window: MainWindow, renders: Counter, *roots: QWidget
) -> Counter:
    """Count scene rebuilds over two flushes that change nothing, of any viewer.

    The viewers shown stay the same ones, so a refresh that replaced a viewer
    with a new one (and drew that) is caught as well as one that redrew it.
    """

    roots = roots or (window,)

    def _drawn() -> bool:
        shown = _shown_viewers(*roots)
        return bool(shown) and all(v.has_structure() and renders[v] for v in shown)

    # Wait, bounded, for each viewer to be created, shown and drawn once, so the
    # counter sees it; a viewer can be created from a timer its page's first paint starts.
    assert pump_until(_drawn, timeout=5.0)
    viewers = _visible_viewers(*roots)
    renders.clear()
    for _ in range(2):
        flush_dashboard(window)
        _settle(qapp)
        assert _visible_viewers(*roots) == viewers
    return +renders


@pytest.mark.parametrize(
    ("route", "page"),
    [
        ("project_home", "_home_page"),
        ("host_structure", "_host_structure_view"),
        ("muon_candidates", "_muon_generation_view"),
        ("muon_sites", "_muon_sites_summary_view"),
        ("muon_analysis", "_relaxation_analysis_view"),
    ],
)
def test_an_unchanged_refresh_rebuilds_no_viewer_on_a_page(
    qapp, tmp_path: Path, window: MainWindow, renders: Counter, route: str, page: str
) -> None:
    _open_muon_project(qapp, window, tmp_path)
    assert window.show_page(route)
    _settle(qapp)
    shown = getattr(window, page)
    assert window._main_stack.currentWidget() is shown

    assert _rebuilds_over_two_unchanged_refreshes(qapp, window, renders, shown) == Counter()


@pytest.mark.parametrize("band", [False, True], ids=["hop", "finished-band"])
def test_an_unchanged_refresh_rebuilds_no_energy_barrier_path_viewer(
    qapp, tmp_path: Path, window: MainWindow, renders: Counter, band: bool
) -> None:
    service = ProjectService()
    document, definition, _source, _output = seed_energy_barriers_document(service, tmp_path)
    if band:
        ts_path = tmp_path / "barrier.ts"
        ts_path.write_text(NEB_TS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
        job, _result = add_band(service, document, ts_path)
        definition.parameters["job_ids_by_label"]["source-key"] = str(job.id)
    _open(qapp, window, document)
    select_route(window, "energy_barriers")
    _settle(qapp)
    view = window._energy_barriers_view
    view._sources_table.selectRow(0)
    _settle(qapp)
    assert view._preview_viewer is not None
    assert view._band_controls.isVisible() is band

    assert _rebuilds_over_two_unchanged_refreshes(qapp, window, renders) == Counter()


@pytest.mark.parametrize("tab", ["_potential_tab", "_batch_tab"])
def test_an_unchanged_refresh_rebuilds_no_viewer_in_the_uep_analysis_window(
    qapp,
    tmp_path: Path,
    window: MainWindow,
    renders: Counter,
    monkeypatch: pytest.MonkeyPatch,
    tab: str,
) -> None:
    from mufinder.app import uep_wizard

    def _start_inline(thread) -> None:
        thread.run()
        thread.finished.emit()

    # The potential's grid and mesh are built before the measurement starts.
    monkeypatch.setattr(uep_wizard._IsosurfaceGridThread, "start", _start_inline)
    monkeypatch.setattr(uep_wizard._IsosurfaceMeshThread, "start", _start_inline)
    document, job, _run, _grid = completed_uep_job_with_grid(window, tmp_path)
    _open(qapp, window, document)
    window._uep_controller.open_analysis("row_1")
    dialog = window._uep_controller.analysis_dialogs[job.id]
    dialog.show()
    dialog._tabs.setCurrentWidget(getattr(dialog, tab))
    _settle(qapp)
    if tab == "_potential_tab":
        # The potential scene holds an isosurface, not only the host.
        dialog._isosurface_level_input.setValue(1.0)
        dialog._isosurface_level_input.editingFinished.emit()
        _settle(qapp)
        assert dialog._isosurface_layers[0].mesh is not None
    try:
        assert _rebuilds_over_two_unchanged_refreshes(qapp, window, renders, dialog) == Counter()
    finally:
        dialog.close()


def test_an_unchanged_refresh_rebuilds_no_viewer_in_the_quantum_analysis_window(
    qapp, tmp_path: Path, window: MainWindow, renders: Counter
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    job = bind_phonon(service, document, definition, key, status=JobStatus.COMPLETED)
    phonon_path = output_root / "site.phonon"
    phonon_path.parent.mkdir(parents=True, exist_ok=True)
    phonon_path.write_text(QUANTUM_PHONON_TEXT, encoding="utf-8")
    result = store_phonon_result(document, job, {"phonon_file": str(phonon_path)})
    _open(qapp, window, document)
    select_route(window, "quantum_corrections")
    window._quantum_corrections_controller.open_analysis(str(result.id))
    dialog = window._quantum_corrections_controller.analysis_dialog(result.id)
    assert dialog is not None
    _settle(qapp)
    try:
        assert _rebuilds_over_two_unchanged_refreshes(qapp, window, renders, dialog) == Counter()
    finally:
        dialog.close()


def _measure_while_open(
    monkeypatch: pytest.MonkeyPatch, dialog_class: type[QDialog], measure: Callable[[QDialog], None]
) -> list[QDialog]:
    """Run ``measure`` from inside each ``dialog_class.exec`` event loop, then reject the dialog."""

    opened: list[QDialog] = []
    original_exec = dialog_class.exec

    def _measure_then_reject(dialog: QDialog) -> None:
        try:
            measure(dialog)
        finally:
            dialog.reject()

    def _exec(dialog: QDialog) -> int:
        opened.append(dialog)
        QTimer.singleShot(0, dialog, lambda: _measure_then_reject(dialog))
        return original_exec(dialog)

    monkeypatch.setattr(dialog_class, "exec", _exec)
    return opened


def test_an_unchanged_refresh_rebuilds_no_viewer_in_the_supercell_editor(
    qapp, tmp_path: Path, window: MainWindow, renders: Counter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _open_muon_project(qapp, window, tmp_path)
    select_route(window, "host_structure")
    measured: list[Counter] = []
    opened = _measure_while_open(
        monkeypatch,
        SupercellEditorDialog,
        lambda dialog: measured.append(
            _rebuilds_over_two_unchanged_refreshes(qapp, window, renders, dialog)
        ),
    )

    window._host_structure_view._supercell_button.click()

    assert len(opened) == 1
    assert measured == [Counter()]


def test_an_unchanged_refresh_rebuilds_no_viewer_in_the_magnetic_structure_builder(
    qapp, tmp_path: Path, window: MainWindow, renders: Counter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _open_muon_project(qapp, window, tmp_path)
    select_route(window, "magnetic_fields")
    _settle(qapp)
    measured: list[Counter] = []
    opened = _measure_while_open(
        monkeypatch,
        MagneticStructureBuilderDialog,
        lambda dialog: measured.append(
            _rebuilds_over_two_unchanged_refreshes(qapp, window, renders, dialog)
        ),
    )

    window._magnetic_field_view._edit_structure_button.click()

    assert len(opened) == 1
    assert measured == [Counter()]


@pytest.mark.parametrize(
    ("dialog_class", "button_text"),
    [
        (RelaxationClusteringDialog, "Clustering…"),
        (DisplacementAnalysisDialog, "Analyse Displacements"),
    ],
    ids=["clustering", "displacements"],
)
def test_an_unchanged_refresh_rebuilds_no_viewer_in_a_relaxation_analysis_dialog(
    qapp,
    tmp_path: Path,
    window: MainWindow,
    renders: Counter,
    monkeypatch: pytest.MonkeyPatch,
    dialog_class: type[QDialog],
    button_text: str,
) -> None:
    _open_muon_project(qapp, window, tmp_path)
    select_route(window, "muon_analysis")
    _settle(qapp)
    view = window._relaxation_analysis_view
    measured: list[Counter] = []
    opened = _measure_while_open(
        monkeypatch,
        dialog_class,
        lambda dialog: measured.append(
            _rebuilds_over_two_unchanged_refreshes(qapp, window, renders, dialog)
        ),
    )

    workspace = view._workspace_tabs.currentWidget()
    button = next(
        button for button in workspace.findChildren(QPushButton) if button.text() == button_text
    )
    button.click()

    assert len(opened) == 1
    assert measured == [Counter()]


def test_a_refresh_after_the_host_moved_rebuilds_the_host_viewer_once(
    qapp, tmp_path: Path, window: MainWindow, renders: Counter
) -> None:
    _open_muon_project(qapp, window, tmp_path)
    select_route(window, "host_structure")
    _settle(qapp)
    viewer = window._host_structure_view._viewer
    renders.clear()

    host = window._document.structures[window._document.project.host_structure_id]
    host.fractional_positions[0][0] = 0.01
    flush_dashboard(window)
    _settle(qapp)

    assert renders[viewer] == 1
