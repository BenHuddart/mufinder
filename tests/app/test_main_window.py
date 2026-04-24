"""Smoke tests for main window wiring and route changes."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDockWidget, QHeaderView, QLabel, QFrame, QTreeWidgetItem

import mufinder.app.main_window as main_window_module
import mufinder.castep_io.schema as schema_module
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import CalculationDefinition, ComparisonTable
from mufinder.services.project_service import ProjectService


def _find_item_by_route(parent: QTreeWidgetItem, route: str) -> QTreeWidgetItem | None:
    if parent.data(0, 0x0100) == route:  # Qt.UserRole
        return parent
    for index in range(parent.childCount()):
        child = parent.child(index)
        if child is None:
            continue
        found = _find_item_by_route(child, route)
        if found is not None:
            return found
    return None


def _find_item_by_prefix(parent: QTreeWidgetItem, prefix: str) -> QTreeWidgetItem | None:
    if parent.text(0).startswith(prefix):
        return parent
    for index in range(parent.childCount()):
        child = parent.child(index)
        if child is None:
            continue
        found = _find_item_by_prefix(child, prefix)
        if found is not None:
            return found
    return None


def test_main_window_constructs_and_routes_to_muon_panel(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_doc.mufinder", "UI Doc")
    window._document = doc
    window._refresh_ui_after_project_change()

    # Inject a candidates node selection and verify stacked panel switch.
    item = QTreeWidgetItem(["Candidates"])
    item.setData(0, 0x0100, "muon_candidates")  # Qt.UserRole
    window._project_tree.clearSelection()
    window._project_tree.addTopLevelItem(item)
    window._project_tree.setCurrentItem(item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._muon_generation_view

    # Persist form state callback should mutate project UI state.
    window._on_muon_form_state_changed({"method": "random"})
    assert doc.ui_state["muon_generation_form"]["method"] == "random"

    window.close()


def test_main_window_constructs_with_bundled_castep_catalog_fallback(qapp, tmp_path: Path) -> None:
    original_candidate_sources = schema_module._candidate_catalog_sources

    try:
        schema_module._candidate_catalog_sources = lambda: (
            schema_module.CatalogSourceInfo(
                path=tmp_path / "missing.json",
                origin="test_missing",
            ),
        )
        source = schema_module.refresh_keyword_catalog(allow_fallback=True)

        assert source is not None
        assert source.origin == "bundled_starter_catalog"

        window = MainWindow()
        assert window._convergence_view is not None
        assert window._kpoint_convergence_view is not None
        window.close()
    finally:
        schema_module._candidate_catalog_sources = original_candidate_sources
        schema_module.refresh_keyword_catalog(allow_fallback=True)


def test_main_window_routes_to_convergence_panel(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_conv_doc.mufinder", "UI Conv")
    window._document = doc
    window._refresh_ui_after_project_change()

    item = QTreeWidgetItem(["Cutoff Energy"])
    item.setData(0, 0x0100, "convergence_cutoff")  # Qt.UserRole
    window._project_tree.clearSelection()
    window._project_tree.addTopLevelItem(item)
    window._project_tree.setCurrentItem(item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._convergence_view
    window.close()


def test_main_window_routes_to_convergence_overview_panel(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_conv_overview_doc.mufinder", "UI Conv Overview")
    window._document = doc
    window._refresh_ui_after_project_change()

    item = QTreeWidgetItem(["Convergence Testing"])
    item.setData(0, 0x0100, "convergence")  # Qt.UserRole
    window._project_tree.clearSelection()
    window._project_tree.addTopLevelItem(item)
    window._project_tree.setCurrentItem(item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._convergence_overview_view
    window.close()


def test_main_window_routes_to_pristine_relaxation_panel(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_pristine_doc.mufinder", "UI Pristine")
    window._document = doc
    window._refresh_ui_after_project_change()

    item = QTreeWidgetItem(["Pristine Relaxation"])
    item.setData(0, 0x0100, "pristine_relaxation")  # Qt.UserRole
    window._project_tree.clearSelection()
    window._project_tree.addTopLevelItem(item)
    window._project_tree.setCurrentItem(item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._pristine_relaxation_view
    window.close()


def test_main_window_routes_to_muon_relaxation_panel(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_muon_relax_doc.mufinder", "UI Muon Relax")
    window._document = doc
    window._refresh_ui_after_project_change()

    item = QTreeWidgetItem(["Relaxations"])
    item.setData(0, 0x0100, "muon_relaxations")  # Qt.UserRole
    window._project_tree.clearSelection()
    window._project_tree.addTopLevelItem(item)
    window._project_tree.setCurrentItem(item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._muon_relaxation_view
    window.close()


def test_main_window_jobs_table_includes_cluster_columns(qapp) -> None:
    window = MainWindow()

    headers = [
        window._jobs_table.horizontalHeaderItem(index).text()
        for index in range(window._jobs_table.columnCount())
    ]

    assert "Cluster" in headers
    assert "Remote Job ID" in headers
    assert "Runtime" in headers
    assert "Processes" in headers
    assert window._jobs_table.columnCount() == 12
    assert window._jobs_table.property("squareTopCorners") == "true"
    header = window._jobs_table.horizontalHeader()
    assert header.sectionResizeMode(window._JOB_LABEL_COLUMN) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(window._JOB_REMOTE_ID_COLUMN) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(window._JOB_RUNTIME_COLUMN) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(window._JOB_PROCESSES_COLUMN) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(window._JOB_SUBMIT_COLUMN) == QHeaderView.ResizeMode.ResizeToContents
    dock = window.findChild(QDockWidget, "jobDashboardDock")
    assert dock is not None
    window.close()


def test_main_window_project_tree_uses_custom_panel_header(qapp) -> None:
    window = MainWindow()

    panel = window.findChild(QFrame, "projectTreePanel")
    body = window.findChild(QFrame, "projectTreePanelBody")
    title = window.findChild(QLabel, "projectTreePanelTitle")

    assert panel is not None
    assert body is not None
    assert title is not None
    assert title.text() == "Project"
    assert window._project_tree.isHeaderHidden() is True

    window.close()


def test_main_window_workflow_completion_roles_track_project_defaults(
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_progress_doc.mufinder", "UI Progress")
    window._document = doc
    window._refresh_ui_after_project_change()

    root = window._project_tree.topLevelItem(0)
    assert root is not None

    host_item = _find_item_by_route(root, "host_structure")
    cutoff_item = _find_item_by_route(root, "convergence_cutoff")
    kpoint_item = _find_item_by_route(root, "convergence_kpoint")
    pristine_item = _find_item_by_route(root, "pristine_relaxation")
    convergence_item = _find_item_by_prefix(root, "Convergence Testing")

    assert host_item is not None
    assert cutoff_item is not None
    assert kpoint_item is not None
    assert pristine_item is not None
    assert convergence_item is not None

    assert host_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is False
    assert cutoff_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is False
    assert kpoint_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is False
    assert pristine_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is False
    assert convergence_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is False

    doc.project.host_structure_id = uuid4()
    doc.project.recommended_cutoff_energy = 500.0
    doc.project.recommended_kpoint_grid = (2, 2, 2)
    doc.comparison_tables[uuid4()] = ComparisonTable(label="Comparison 1")
    window._populate_project_tree()

    root = window._project_tree.topLevelItem(0)
    assert root is not None
    host_item = _find_item_by_route(root, "host_structure")
    cutoff_item = _find_item_by_route(root, "convergence_cutoff")
    kpoint_item = _find_item_by_route(root, "convergence_kpoint")
    pristine_item = _find_item_by_route(root, "pristine_relaxation")
    convergence_item = _find_item_by_prefix(root, "Convergence Testing")

    assert host_item is not None
    assert cutoff_item is not None
    assert kpoint_item is not None
    assert pristine_item is not None
    assert convergence_item is not None

    assert host_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is True
    assert cutoff_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is True
    assert kpoint_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is True
    assert pristine_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is True
    assert convergence_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is True

    assert "Load a host structure" in str(host_item.data(0, window._WORKFLOW_GOAL_ROLE))
    assert "project default cutoff" in str(cutoff_item.data(0, window._WORKFLOW_GOAL_ROLE)).lower()
    assert "project default k-point" in str(kpoint_item.data(0, window._WORKFLOW_GOAL_ROLE)).lower()
    assert "comparison table" in str(pristine_item.data(0, window._WORKFLOW_GOAL_ROLE)).lower()

    window.close()


def test_main_window_muon_relaxation_workflow_complete_when_table_jobs_done(
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_muon_relax_progress.mufinder", "Muon Relax Progress")

    first_job = service.add_job(
        doc,
        label="Relax A",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.MUON_RELAXATION_CALC_TYPES["positive_muon"],
    )
    first_job.status = JobStatus.COMPLETED
    second_job = service.add_job(
        doc,
        label="Relax B",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.MUON_RELAXATION_CALC_TYPES["positive_muon"],
    )
    second_job.status = JobStatus.QUEUED
    extra_job = service.add_job(
        doc,
        label="Historical Relax",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type=main_window_module.MUON_RELAXATION_CALC_TYPES["positive_muon"],
    )
    extra_job.status = JobStatus.RUNNING
    definition = CalculationDefinition(
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={
            "configuration": {
                "rows": [
                    {
                        "site_id": str(uuid4()),
                        "label": "Relax A",
                        "initial_fractional_position": [0.1, 0.2, 0.3],
                        "charge_state": "positive_muon",
                    },
                    {
                        "site_id": str(uuid4()),
                        "label": "Relax B",
                        "initial_fractional_position": [0.4, 0.5, 0.6],
                        "charge_state": "positive_muon",
                    },
                ]
            },
            "job_ids_by_label": {
                "positivemuon:Relax A": str(first_job.id),
                "positivemuon:Relax B": str(second_job.id),
            },
        },
    )
    doc.calculations[definition.id] = definition
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    window._document = doc
    window._refresh_ui_after_project_change()

    root = window._project_tree.topLevelItem(0)
    assert root is not None
    relaxation_item = _find_item_by_route(root, "muon_relaxations")

    assert relaxation_item is not None
    assert relaxation_item.toolTip(0) == "Complete when: All structural relaxations completed."
    assert relaxation_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is False

    second_job.status = JobStatus.COMPLETED
    window._populate_project_tree()

    root = window._project_tree.topLevelItem(0)
    assert root is not None
    relaxation_item = _find_item_by_route(root, "muon_relaxations")
    assert relaxation_item is not None
    assert relaxation_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is True

    window.close()


def test_scaled_home_logo_returns_none_for_missing_image(monkeypatch, qapp) -> None:
    monkeypatch.setattr(main_window_module, "logo_path", lambda: "/tmp/definitely-missing-logo.png")

    window = MainWindow()
    assert window._scaled_home_logo() is None
    window.close()


def test_scaled_home_logo_uses_device_pixel_ratio(monkeypatch, tmp_path: Path, qapp) -> None:
    logo_file = tmp_path / "logo.png"
    pixmap = QPixmap(64, 64)
    assert pixmap.save(str(logo_file))

    monkeypatch.setattr(main_window_module, "logo_path", lambda: str(logo_file))

    window = MainWindow()
    monkeypatch.setattr(window, "_home_logo_device_pixel_ratio", lambda: 2.0)

    scaled = window._scaled_home_logo()
    assert scaled is not None
    assert not scaled.isNull()
    assert scaled.devicePixelRatio() == 2.0
    assert scaled.width() == 184
    assert scaled.height() == 184

    window.close()
