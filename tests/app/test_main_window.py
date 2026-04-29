"""Smoke tests for main window wiring and route changes."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDockWidget, QFrame, QHeaderView, QLabel, QMessageBox, QToolButton, QTreeWidgetItem

import mufinder.app.magnetic_field_widget as magnetic_field_widget_module
import mufinder.app.main_window as main_window_module
import mufinder.castep_io.schema as schema_module
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, ComparisonTable, Structure
from mufinder.science.magnetic_structure import AtomSelector, MagneticStructureModel, build_collinear_component
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


def _find_tree_item_by_route(window: MainWindow, route: str) -> QTreeWidgetItem | None:
    for index in range(window._project_tree.topLevelItemCount()):
        item = window._project_tree.topLevelItem(index)
        if item is None:
            continue
        found = _find_item_by_route(item, route)
        if found is not None:
            return found
    return None


def _find_tree_item_by_prefix(window: MainWindow, prefix: str) -> QTreeWidgetItem | None:
    for index in range(window._project_tree.topLevelItemCount()):
        item = window._project_tree.topLevelItem(index)
        if item is None:
            continue
        found = _find_item_by_prefix(item, prefix)
        if found is not None:
            return found
    return None


def _magnetic_model(label: str) -> MagneticStructureModel:
    return MagneticStructureModel(
        components=(
            build_collinear_component(
                label=label,
                selector=AtomSelector.indices([0]),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, 1.0),
                component_id=label.casefold().replace(" ", "_"),
            ),
        )
    )


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


def test_main_window_routes_to_magnetic_fields_panel_and_persists_model(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_magnetic_doc.mufinder", "UI Magnetic")
    host = Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    window._document = doc
    window._refresh_ui_after_project_change()

    item = QTreeWidgetItem(["Magnetic Fields"])
    item.setData(0, 0x0100, "magnetic_fields")  # Qt.UserRole
    window._project_tree.clearSelection()
    window._project_tree.addTopLevelItem(item)
    window._project_tree.setCurrentItem(item)
    window._on_tree_selection_changed()

    payload = {"schema_version": 1, "q_basis_rlu": [[0.0, 0.0, 0.0]] * 3, "components": [{}]}
    window._on_magnetic_field_structure_changed(payload)

    assert window._main_stack.currentWidget() is window._magnetic_field_view
    assert doc.project.magnetic_structure == payload
    assert host.properties["magnetic_structure"] == payload
    window.close()


def test_main_window_restores_magnetic_field_form_state(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_magnetic_form_doc.mufinder", "UI Magnetic Form")
    host = Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.ui_state["magnetic_field_form"] = {
        "demag_shape": "none",
        "supercell": [18, 20, 22],
        "include_equivalent_sites": True,
        "incommensurate_samples": 8192,
        "histogram_bins": 1024,
        "selected_muon_site": {
            "source": "custom",
            "label": "Custom site",
            "fractional_position": [0.25, 0.5, 0.75],
            "cartesian_position": [0.75, 1.5, 2.25],
        },
    }

    window._document = doc
    window._refresh_ui_after_project_change()

    view = window._magnetic_field_view
    assert view._demag_shape() == "none"
    assert view._field_options().supercell == (18, 20, 22)
    assert view._field_options().find_equivalent_sites is True
    assert view._field_options().nangles == 8192
    assert view._field_options().histogram_bins == 1024
    assert view.selected_muon_site().fractional_position == (0.25, 0.5, 0.75)

    view._histogram_bins.setValue(768)

    assert doc.ui_state["magnetic_field_form"]["histogram_bins"] == 768

    service.save_project(doc)
    reloaded = service.load_project(doc.project_root)
    reloaded_window = MainWindow()
    reloaded_window._document = reloaded
    reloaded_window._refresh_ui_after_project_change()
    reloaded_view = reloaded_window._magnetic_field_view

    assert reloaded_view._demag_shape() == "none"
    assert reloaded_view._field_options().supercell == (18, 20, 22)
    assert reloaded_view._field_options().find_equivalent_sites is True
    assert reloaded_view._field_options().nangles == 8192
    assert reloaded_view._field_options().histogram_bins == 768
    assert reloaded_view.selected_muon_site().fractional_position == (0.25, 0.5, 0.75)

    window.close()
    reloaded_window.close()


def test_main_window_persists_magnetic_field_calculation_results(qapp, tmp_path: Path, monkeypatch) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_magnetic_results_doc.mufinder", "UI Magnetic Results")
    host = Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)

    class _Result:
        site_label = "site 1"
        muon_position = (0.25, 0.25, 0.25)
        total = (0.1, 0.2, 0.3)
        magnitude = float(np.linalg.norm(total))

        def to_json(self):
            return {
                "site_label": self.site_label,
                "muon_position": list(self.muon_position),
                "total_T": list(self.total),
                "magnitude_T": self.magnitude,
            }

    monkeypatch.setattr(magnetic_field_widget_module, "classify_magnetic_model", lambda *_args, **_kwargs: "commensurate")
    monkeypatch.setattr(magnetic_field_widget_module, "calculate_commensurate_fields", lambda *_args, **_kwargs: (_Result(),))

    window._document = doc
    window._refresh_ui_after_project_change()
    window._magnetic_field_view._set_model(_magnetic_model("Main field"))
    window._magnetic_field_view._selected_muon_site = window._magnetic_field_view._default_muon_site()
    window._magnetic_field_view._calculate_fields()

    assert doc.ui_state["magnetic_field_form"]["magnetic_field_results"]["kind"] == "commensurate"

    service.save_project(doc)
    reloaded = service.load_project(doc.project_root)
    reloaded_window = MainWindow()
    reloaded_window._document = reloaded
    reloaded_window._refresh_ui_after_project_change()

    assert reloaded_window._magnetic_field_view._result_table.rowCount() == 1
    assert reloaded_window._magnetic_field_view._result_table.item(0, 1).text() == "0.1"

    window.close()
    reloaded_window.close()


def test_main_window_pins_restores_renames_and_deletes_magnetic_field_workspace(
    qapp,
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_magnetic_workspace_doc.mufinder", "UI Magnetic Workspace")
    host = Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    window._document = doc
    window._refresh_ui_after_project_change()
    window._set_active_route("magnetic_fields")
    window._magnetic_field_view._set_model(_magnetic_model("Main helix"))
    window._magnetic_field_view._none_radio.setChecked(True)
    window._magnetic_field_view._phase_samples.setValue(8192)
    main_structure = dict(doc.project.magnetic_structure)
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("Pinned helix", True),
    )

    window._magnetic_field_view._pin_button.click()

    workspaces = doc.ui_state["magnetic_field_workspaces"]
    assert len(workspaces) == 1
    workspace_id = workspaces[0]["id"]
    route = window._magnetic_field_workspace_route(workspace_id)
    workspace_item = _find_tree_item_by_route(window, route)
    workspace_widget = window._magnetic_field_workspace_widgets[workspace_id]
    assert workspace_item is not None
    assert workspace_item.text(0) == "Pinned helix"
    assert window._main_stack.currentWidget() is workspace_widget
    assert workspace_widget.workspace_title() == "Pinned helix"
    assert workspace_widget._field_options().nangles == 8192

    workspace_widget._set_model(_magnetic_model("Copied helix"))

    assert doc.project.magnetic_structure == main_structure
    assert doc.ui_state["magnetic_field_workspaces"][0]["magnetic_structure"]["components"][0]["label"] == (
        "Copied helix"
    )

    workspace_widget._title_edit.setText("Renamed helix")
    workspace_widget._on_title_edit_finished()

    assert doc.ui_state["magnetic_field_workspaces"][0]["title"] == "Renamed helix"
    workspace_item = _find_tree_item_by_route(window, route)
    assert workspace_item is not None
    assert workspace_item.text(0) == "Renamed helix"

    service.save_project(doc)
    reloaded = service.load_project(doc.project_root)
    reloaded_window = MainWindow()
    reloaded_window._document = reloaded
    reloaded_window._refresh_ui_after_project_change()
    reloaded_workspace = reloaded.ui_state["magnetic_field_workspaces"][0]
    reloaded_workspace_widget = reloaded_window._magnetic_field_workspace_widgets[workspace_id]
    assert reloaded_workspace["title"] == "Renamed helix"
    assert reloaded_workspace_widget.workspace_title() == "Renamed helix"
    assert reloaded.project.magnetic_structure == main_structure

    reloaded_window._set_active_route(route)
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    reloaded_workspace_widget._delete_workspace_button.click()

    assert reloaded.ui_state["magnetic_field_workspaces"] == []
    assert reloaded_window._main_stack.currentWidget() is reloaded_window._magnetic_field_view
    window.close()
    reloaded_window.close()


def test_main_window_routes_to_construction_placeholders(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_placeholders_doc.mufinder", "UI Placeholders")
    window._document = doc
    window._refresh_ui_after_project_change()

    quantum_item = _find_tree_item_by_route(window, "quantum_corrections")
    assert quantum_item is not None
    window._project_tree.setCurrentItem(quantum_item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._quantum_corrections_placeholder
    assert window._quantum_corrections_placeholder.accessibleName() == (
        "Under construction. Come back soon."
    )

    barriers_item = _find_tree_item_by_route(window, "energy_barriers")
    assert barriers_item is not None
    window._project_tree.setCurrentItem(barriers_item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._energy_barriers_placeholder
    assert window._energy_barriers_placeholder.accessibleName() == (
        "Under construction. Come back soon."
    )
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
    header = window.findChild(QFrame, "projectTreePanelHeader")
    body = window.findChild(QFrame, "projectTreePanelBody")
    title = window.findChild(QLabel, "projectTreePanelTitle")
    home_button = window.findChild(QToolButton, "projectHomeButton")

    assert panel is not None
    assert header is not None
    assert body is not None
    assert title is not None
    assert home_button is not None
    assert title.text() == "Project"
    assert home_button.isEnabled() is False
    assert panel.minimumWidth() == 220
    assert panel.maximumWidth() == 300
    assert window._project_tree.isHeaderHidden() is True

    window.close()


def test_main_window_project_tree_header_shows_project_name_without_root_item(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    doc = ProjectService().create_project(tmp_path / "ui_flat_tree_doc.mufinder", "CoF2")
    window._document = doc
    window._refresh_ui_after_project_change()

    title = window.findChild(QLabel, "projectTreePanelTitle")
    assert title is not None
    assert title.text() == "Project: CoF2"
    assert window._project_tree.topLevelItem(0).text(0) == "Host Structure"
    assert _find_tree_item_by_route(window, "host_structure") is not None
    assert window._main_stack.currentWidget() is window._home_page
    assert window._home_content_stack.currentWidget() is window._project_home_page
    assert window._project_home_button.isEnabled() is True

    window.close()


def test_project_home_trail_destinations_include_journal(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    doc = ProjectService().create_project(tmp_path / "project_home_trail_doc.mufinder", "Trail")
    window._document = doc
    window._refresh_ui_after_project_change()

    destinations = window._project_home_destinations()

    assert [destination.title for destination in destinations] == [
        "Host Structure",
        "Supercell Generation",
        "Cutoff Convergence",
        "K-Point Grid",
        "Pristine Relaxation",
        "UEP",
        "Muon Sites",
        "Magnetic Fields",
        "Quantum Corrections",
        "Energy Barriers",
        "Journal",
    ]
    assert destinations[-1].route == "provenance_log"
    assert destinations[-1].placement == "journal"
    assert window.findChildren(main_window_module.ProjectHomeTrailWidget)
    assert not hasattr(main_window_module, "ExpeditionTrailWidget")
    assert window._project_home_trail._map_pixmap.isNull() is False
    assert window._project_home_trail._color_map_pixmap.isNull() is False

    pixmap = QPixmap(900, 520)
    window._project_home_trail.resize(900, 520)
    assert window._project_home_trail._map_rect().width() > 850
    assert "provenance_log" not in main_window_module.ProjectHomeTrailWidget._SIGN_REGIONS
    host_target = window._project_home_trail._map_rect_from_source(
        main_window_module.ProjectHomeTrailWidget._LANDMARK_REGIONS["host_structure"]
    ).center()
    assert window._project_home_trail._route_at(host_target) == "host_structure"
    journal_target = window._project_home_trail._map_rect_from_source(
        main_window_module.ProjectHomeTrailWidget._LANDMARK_REGIONS["provenance_log"]
    ).center()
    assert window._project_home_trail._route_at(journal_target) == "provenance_log"
    window._project_home_trail.render(pixmap)
    assert not pixmap.isNull()

    window._project_home_trail._hovered_route = "host_structure"
    window._project_home_trail.render(pixmap)
    assert not pixmap.isNull()

    window.close()


def test_project_home_button_returns_to_project_home(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    doc = ProjectService().create_project(tmp_path / "project_home_button_doc.mufinder", "Home")
    window._document = doc
    window._refresh_ui_after_project_change()

    window._set_active_route("host_structure")
    assert window._main_stack.currentWidget() is window._host_structure_view

    window._project_home_button.click()

    assert window._main_stack.currentWidget() is window._home_page
    assert window._home_content_stack.currentWidget() is window._project_home_page
    assert window._active_route() is None

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

    host_item = _find_tree_item_by_route(window, "host_structure")
    cutoff_item = _find_tree_item_by_route(window, "convergence_cutoff")
    kpoint_item = _find_tree_item_by_route(window, "convergence_kpoint")
    pristine_item = _find_tree_item_by_route(window, "pristine_relaxation")
    convergence_item = _find_tree_item_by_prefix(window, "Convergence Testing")

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

    host_item = _find_tree_item_by_route(window, "host_structure")
    cutoff_item = _find_tree_item_by_route(window, "convergence_cutoff")
    kpoint_item = _find_tree_item_by_route(window, "convergence_kpoint")
    pristine_item = _find_tree_item_by_route(window, "pristine_relaxation")
    convergence_item = _find_tree_item_by_prefix(window, "Convergence Testing")

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


def test_main_window_supercell_editor_complete_when_active_host_has_saved_asset(
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_supercell_progress.mufinder", "UI Supercell Progress")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    window._document = doc
    window._refresh_ui_after_project_change()

    supercell_item = _find_tree_item_by_route(window, "supercell_editor")
    assert supercell_item is not None
    assert supercell_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is False
    assert "Save reusable supercells" in str(supercell_item.data(0, window._WORKFLOW_GOAL_ROLE))

    other_host = Structure(
        label="Other host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, other_host)
    service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=other_host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    window._populate_project_tree()

    supercell_item = _find_tree_item_by_route(window, "supercell_editor")
    assert supercell_item is not None
    assert supercell_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is False

    service.create_workflow_supercell_asset(
        doc,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        host_structure_id=host.id,
        matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reuse_existing=False,
    )
    window._populate_project_tree()

    supercell_item = _find_tree_item_by_route(window, "supercell_editor")
    assert supercell_item is not None
    assert supercell_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is True

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

    relaxation_item = _find_tree_item_by_route(window, "muon_relaxations")

    assert relaxation_item is not None
    assert relaxation_item.toolTip(0) == "Complete when: All structural relaxations completed."
    assert relaxation_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is False

    second_job.status = JobStatus.COMPLETED
    window._populate_project_tree()

    relaxation_item = _find_tree_item_by_route(window, "muon_relaxations")
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
