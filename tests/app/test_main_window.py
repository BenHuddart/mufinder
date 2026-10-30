"""Smoke tests for main window wiring and route changes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDockWidget,
    QFrame,
    QHeaderView,
    QLabel,
    QTabWidget,
    QToolButton,
    QTreeWidgetItem,
)

import mufinder.app.magnetic_field_widget as magnetic_field_widget_module
import mufinder.services.magnetic_field_calculation as magnetic_field_calculation_module
import mufinder.app.main_window as main_window_module
import mufinder.app.workflow_controllers.energy_barriers as energy_barriers_controller_module
import mufinder.services.job_artifacts as job_artifacts_module
import mufinder.castep_io.schema as schema_module
from mufinder.app.main_window import MainWindow
from tests.support.muon_sites import add_relaxed_muon_site
from mufinder.services.muon_sites_summary import (
    ENERGY_BARRIERS_SCOPE,
    QUANTUM_CORRECTIONS_SCOPE,
    set_show_every_relaxed_site,
)
from mufinder.app.workflow_dialogs import QuantumCorrectionsConfigDialog
from mufinder.castep_io.types import ValueWithUnit
from mufinder.castep_io.ts_parser import TransitionStateImage, TransitionStatePath
from mufinder.app.param_dialog import ParameterDialog
from mufinder.domain.charge_states import MUON_RELAXATION_CALC_TYPES
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    ComparisonTable,
    Job,
    MuonSite,
    Result,
    Structure,
)
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
)
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    build_collinear_component,
)
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    MAGNETIC_FIELD_RESULT_TYPE,
    magnetic_field_results,
)
from mufinder.services.project_service import ProjectService
from mufinder.services.project_summary import StageStatus, summarize_project
from mufinder.services.source_candidates import (
    energy_barrier_generated_sources,
    quantum_source_relative_energies,
)
from tests.support.qt import find_tree_item_by_route


PHONON_TEXT = """ BEGIN header
 Number of ions         2
 Number of branches     6
 Number of wavevectors  1
 Frequencies in         cm-1
 Unit cell vectors (A)
    1.000000    0.000000    0.000000
    0.000000    1.000000    0.000000
    0.000000    0.000000    1.000000
 Fractional Co-ordinates
     1     0.000000    0.000000    0.000000   N        14.006740
     2     0.500000    0.500000    0.500000   H:mu      0.113429
 END header
     q-pt=    1    0.000000  0.000000  0.000000      1.0000000000
       1     100.000000
       2     400.000000
       3     900.000000
       4    1600.000000
       5    2500.000000
       6    3600.000000
                        Phonon Eigenvectors
Mode Ion                X                                   Y                                   Z
   1   1  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   1   2  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   2   1 -0.336650000000  0.000000000000      0.019803000000  0.000000000000      0.138621000000  0.000000000000
   2   2 -0.693103000000  0.000000000000      0.376256000000  0.000000000000      0.495074000000  0.000000000000
   3   1 -0.080432200000  0.000000000000     -0.350117000000  0.000000000000      0.230257000000  0.000000000000
   3   2  0.189252000000  0.000000000000     -0.619801000000  0.000000000000      0.630841000000  0.000000000000
   4   1 -0.501570000000  0.000000000000     -0.324961000000  0.000000000000     -0.339089000000  0.000000000000
   4   2  0.536891000000  0.000000000000      0.459183000000  0.000000000000      0.169545000000  0.000000000000
   5   1  0.583042000000  0.000000000000     -0.144467000000  0.000000000000     -0.701203000000  0.000000000000
   5   2 -0.145760000000  0.000000000000      0.057786700000  0.000000000000      0.350601000000  0.000000000000
   6   1 -0.349315000000  0.000000000000      0.764127000000  0.000000000000     -0.392979000000  0.000000000000
   6   2  0.087328800000  0.000000000000     -0.305651000000  0.000000000000      0.196490000000  0.000000000000
"""


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


def _find_tree_item_by_prefix(window: MainWindow, prefix: str) -> QTreeWidgetItem | None:
    for index in range(window._project_tree.topLevelItemCount()):
        item = window._project_tree.topLevelItem(index)
        if item is None:
            continue
        found = _find_item_by_prefix(item, prefix)
        if found is not None:
            return found
    return None


def _add_transition_state_search_set_definition(
    doc,
    *,
    input_structure_ids: list,
    input_site_ids: list,
    parameters: dict,
) -> CalculationDefinition:
    """Store an Energy Barriers source definition as a saved project holds it."""

    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type="castep_transition_state_search_set",
        parameters={
            "configuration": parameters,
            "output_directory": "files/energy_barriers",
            "job_ids_by_label": {},
        },
        input_structure_ids=list(input_structure_ids),
        input_site_ids=list(input_site_ids),
    )
    doc.calculations[definition.id] = definition
    return definition


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


def test_main_window_routes_to_magnetic_fields_panel_and_persists_model(
    qapp, tmp_path: Path
) -> None:
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
    assert view.field_options().supercell == (18, 20, 22)
    assert view.field_options().find_equivalent_sites is True
    assert view.field_options().phase_sample_count == 8192
    assert view.field_options().histogram_bins == 1024
    assert view.selected_muon_site().fractional_position == (0.25, 0.5, 0.75)

    view.set_calculation_settings(replace(view.calculation_settings(), histogram_bins=768))

    assert doc.ui_state["magnetic_field_form"]["histogram_bins"] == 768

    service.save_project(doc)
    reloaded = service.load_project(doc.project_root)
    reloaded_window = MainWindow()
    reloaded_window._document = reloaded
    reloaded_window._refresh_ui_after_project_change()
    reloaded_view = reloaded_window._magnetic_field_view

    assert reloaded_view._demag_shape() == "none"
    assert reloaded_view.field_options().supercell == (18, 20, 22)
    assert reloaded_view.field_options().find_equivalent_sites is True
    assert reloaded_view.field_options().phase_sample_count == 8192
    assert reloaded_view.field_options().histogram_bins == 768
    assert reloaded_view.selected_muon_site().fractional_position == (0.25, 0.5, 0.75)

    window.close()
    reloaded_window.close()


def test_main_window_persists_magnetic_field_calculation_results(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(
        tmp_path / "ui_magnetic_results_doc.mufinder", "UI Magnetic Results"
    )
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
        distortion = None

        def to_json(self):
            return {
                "site_label": self.site_label,
                "muon_position": list(self.muon_position),
                "total_T": list(self.total),
                "magnitude_T": self.magnitude,
            }

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_commensurate_fields",
        lambda *_args, **_kwargs: (_Result(),),
    )

    window._document = doc
    window._refresh_ui_after_project_change()
    window._magnetic_field_view.set_model(_magnetic_model("Main field"))
    window._magnetic_field_view._selected_muon_site = (
        magnetic_field_widget_module.MagneticMuonSiteOption(
            source="custom",
            label="Custom site",
            fractional_position=(0.25, 0.25, 0.25),
            cartesian_position=(0.75, 0.75, 0.75),
        )
    )
    window._magnetic_field_view.calculate_fields()

    assert "magnetic_field_results" not in doc.ui_state["magnetic_field_form"]
    (record,) = magnetic_field_results(doc)
    assert record.result_type == MAGNETIC_FIELD_RESULT_TYPE
    assert record.calculation_run_id is None
    assert record.data["owner"] == MAGNETIC_FIELD_PAGE_OWNER
    assert record.data["kind"] == "commensurate"
    assert record.data["results"][0]["total_T"] == [0.1, 0.2, 0.3]

    service.save_project(doc)
    reloaded = service.load_project(doc.project_root)
    assert [result.id for result in magnetic_field_results(reloaded)] == [record.id]
    reloaded_window = MainWindow()
    reloaded_window._document = reloaded
    reloaded_window._refresh_ui_after_project_change()

    assert reloaded_window._magnetic_field_view._result_table.rowCount() == 1
    assert reloaded_window._magnetic_field_view._result_table.item(0, 1).text() == "0.1"
    assert "magnetic_field_results" not in reloaded.ui_state["magnetic_field_form"]

    # A form change that invalidates the results removes the record.
    reloaded_view = reloaded_window._magnetic_field_view
    settings = reloaded_view.calculation_settings()
    reloaded_view.set_calculation_settings(
        replace(settings, histogram_bins=settings.histogram_bins + 1)
    )

    assert magnetic_field_results(reloaded) == []

    window.close()
    reloaded_window.close()


def test_main_window_magnetic_field_results_do_not_leak_into_the_next_project(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    first = service.create_project(tmp_path / "first.mufinder", "First")
    second = service.create_project(tmp_path / "second.mufinder", "Second")
    # The second project has saved page settings but no result records.
    second.ui_state["magnetic_field_form"] = {"histogram_bins": 640}

    window._document = first
    window._refresh_ui_after_project_change()
    view = window._magnetic_field_view
    view.set_form_state(
        {
            "histogram_bins": 512,
            "dipolar_tensor_results": [{"site_label": "site 1"}],
            "nuclear_relaxation_results": {"mode": "zero_field_powder", "sigma_us_inv": 0.4},
        }
    )
    view._emit_form_state_changed()
    assert len(magnetic_field_results(first)) == 2

    window._document = second
    window._refresh_ui_after_project_change()
    view._emit_form_state_changed()

    state = view.collect_state()
    assert state["dipolar_tensor_results"] == []
    assert state["nuclear_relaxation_results"] == {}
    assert magnetic_field_results(second) == []
    window.close()


def test_main_window_pinned_magnetic_field_workspace_owns_a_copy_of_the_page_results(
    qapp,
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(
        tmp_path / "ui_magnetic_pin_results_doc.mufinder", "UI Magnetic Pin Results"
    )
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
    view = window._magnetic_field_view
    view.set_form_state(
        {
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [
                    {"site_label": "site 1", "total_T": [0.1, 0.2, 0.3], "magnitude_T": 0.374}
                ],
            },
            "nuclear_relaxation_results": {"mode": "zero_field_powder", "sigma_us_inv": 0.4},
        }
    )
    view._emit_form_state_changed()
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("Pinned results", True),
    )

    view._pin_button.click()

    workspace_id = doc.ui_state["magnetic_field_workspaces"][0]["id"]
    assert "magnetic_field_results" not in doc.ui_state["magnetic_field_workspaces"][0]["form"]
    page_records = magnetic_field_results(doc, MAGNETIC_FIELD_PAGE_OWNER)
    workspace_records = magnetic_field_results(doc, workspace_id)
    assert len(page_records) == 2
    assert {record.result_type for record in workspace_records} == {
        record.result_type for record in page_records
    }
    assert not {record.id for record in workspace_records} & {record.id for record in page_records}
    assert window._magnetic_field_workspace_widgets[workspace_id]._result_table.rowCount() == 1

    service.save_project(doc)
    reloaded = service.load_project(doc.project_root)
    reloaded_window = MainWindow()
    reloaded_window._document = reloaded
    reloaded_window._refresh_ui_after_project_change()
    reloaded_widget = reloaded_window._magnetic_field_workspace_widgets[workspace_id]
    assert reloaded_widget._result_table.rowCount() == 1
    assert reloaded_widget.collect_state()["nuclear_relaxation_results"]["sigma_us_inv"] == 0.4

    reloaded_widget._delete_workspace_button.click()

    assert magnetic_field_results(reloaded, workspace_id) == []
    assert len(magnetic_field_results(reloaded, MAGNETIC_FIELD_PAGE_OWNER)) == 2
    window.close()
    reloaded_window.close()


def test_main_window_pins_restores_renames_and_deletes_magnetic_field_workspace(
    qapp,
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(
        tmp_path / "ui_magnetic_workspace_doc.mufinder", "UI Magnetic Workspace"
    )
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
    window._magnetic_field_view.set_model(_magnetic_model("Main helix"))
    window._magnetic_field_view._none_radio.setChecked(True)
    window._magnetic_field_view.set_calculation_settings(
        replace(window._magnetic_field_view.calculation_settings(), phase_sample_count=8192)
    )
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
    workspace_item = find_tree_item_by_route(window._project_tree, route)
    workspace_widget = window._magnetic_field_workspace_widgets[workspace_id]
    assert workspace_item is not None
    assert workspace_item.text(0) == "Pinned helix"
    assert window._main_stack.currentWidget() is workspace_widget
    assert workspace_widget.workspace_title() == "Pinned helix"
    assert workspace_widget.field_options().phase_sample_count == 8192

    workspace_widget.set_model(_magnetic_model("Copied helix"))

    assert doc.project.magnetic_structure == main_structure
    assert doc.ui_state["magnetic_field_workspaces"][0]["magnetic_structure"]["components"][0][
        "label"
    ] == ("Copied helix")

    workspace_widget._title_edit.setText("Renamed helix")
    workspace_widget._on_title_edit_finished()

    assert doc.ui_state["magnetic_field_workspaces"][0]["title"] == "Renamed helix"
    workspace_item = find_tree_item_by_route(window._project_tree, route)
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
    reloaded_workspace_widget._delete_workspace_button.click()

    assert reloaded.ui_state["magnetic_field_workspaces"] == []
    assert reloaded_window._main_stack.currentWidget() is reloaded_window._magnetic_field_view
    window.close()
    reloaded_window.close()


def test_main_window_routes_to_quantum_view_and_construction_placeholders(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_placeholders_doc.mufinder", "UI Placeholders")
    window._document = doc
    window._refresh_ui_after_project_change()

    quantum_item = find_tree_item_by_route(window._project_tree, "quantum_corrections")
    assert quantum_item is not None
    window._project_tree.setCurrentItem(quantum_item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._quantum_corrections_view
    assert window._quantum_corrections_view.accessibleName() == "Quantum Corrections workflow"

    barriers_item = find_tree_item_by_route(window._project_tree, "energy_barriers")
    assert barriers_item is not None
    window._project_tree.setCurrentItem(barriers_item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._energy_barriers_view
    assert window._energy_barriers_view.accessibleName() == "Energy Barriers workflow"
    tabs = window._energy_barriers_view.findChild(QTabWidget, "energyBarriersTabs")
    assert tabs is not None
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Paths",
        "NEB Analysis",
    ]
    window.close()


def test_main_window_launches_quantum_phonon_analysis_for_result(
    qapp,
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_analysis_doc.mufinder", "Quantum Analysis")
    window._document = doc

    phonon_path = doc.project_root / "files" / "quantum_corrections" / "site.phonon"
    phonon_path.parent.mkdir(parents=True, exist_ok=True)
    phonon_path.write_text("phonon", encoding="utf-8")

    job = Job(
        label="quantum_phonon_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
        status=JobStatus.COMPLETED,
    )
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    result = Result(
        calculation_run_id=run.id,
        result_type=QUANTUM_PHONON_CALC_TYPE,
        data={"phonon_file": str(phonon_path), "muon_zpe_total_eV": 0.1},
    )
    run.result_id = result.id
    doc.jobs[job.id] = job
    doc.runs[run.id] = run
    doc.results[result.id] = result

    constructed: list[dict[str, object]] = []

    class _Signal:
        def connect(self, _callback) -> None:
            return None

    class _FakeDialog:
        destroyed = _Signal()

        def __init__(self, **kwargs) -> None:
            constructed.append(kwargs)
            self.shown = False

        def isVisible(self) -> bool:
            return self.shown

        def show(self) -> None:
            self.shown = True

        def raise_(self) -> None:
            return None

        def activateWindow(self) -> None:
            return None

    monkeypatch.setattr(main_window_module, "QuantumPhononAnalysisDialog", _FakeDialog)

    window._on_launch_quantum_phonon_analysis(str(result.id))

    assert len(constructed) == 1
    assert constructed[0]["phonon_path"] == str(phonon_path)
    assert constructed[0]["job_label"] == "quantum_phonon_siteA"
    assert constructed[0]["result_data"] == result.data
    assert window._quantum_analysis_dialogs[result.id].shown is True

    window.close()


def test_main_window_quantum_completed_results_show_muon_modes(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_results_doc.mufinder", "Quantum Results")
    window._document = doc

    phonon_path = doc.project_root / "files" / "quantum_corrections" / "site.phonon"
    phonon_path.parent.mkdir(parents=True, exist_ok=True)
    phonon_path.write_text(PHONON_TEXT, encoding="utf-8")

    job = Job(
        label="quantum_phonon_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
        status=JobStatus.COMPLETED,
    )
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    result = Result(
        calculation_run_id=run.id,
        result_type=QUANTUM_PHONON_CALC_TYPE,
        data={
            "phonon_file": str(phonon_path),
            "muon_zpe_total_eV": 0.1,
            "w1_entanglement_witness": -0.25,
            "w1_definition": "participation_minus_3",
            "muon_zpe_modes": [
                {"mode_index": 6, "frequency_cm": 3600.0, "participation": 0.9, "width_A": 0.12},
                {"mode_index": 4, "frequency_cm": 1600.0, "participation": 0.9, "width_A": 0.18},
                {"mode_index": 5, "frequency_cm": 2500.0, "participation": 0.95, "width_A": 0.15},
            ],
            "phonon_mode_eigenvalues_cm": [100.0, 400.0, 900.0, 1600.0, 2500.0, 3600.0],
        },
    )
    run.result_id = result.id
    doc.jobs[job.id] = job
    doc.runs[run.id] = run
    doc.results[result.id] = result

    window._refresh_quantum_corrections_view()

    results_table = window._quantum_corrections_view._results_table
    assert [results_table.item(0, column).text() for column in range(5)] == [
        "quantum_phonon_siteA",
        "0.100",
        "1600.0, 2500.0, 3600.0",
        "0.18, 0.15, 0.12",
        "-0.25",
    ]

    window.close()


def test_main_window_close_tolerates_missing_relaxation_analysis_view(qapp) -> None:
    window = MainWindow()

    delattr(window, "_relaxation_analysis_view")

    window.close()

    assert window._shutting_down is True


def test_main_window_materializes_one_quantum_phonon_job_from_relaxed_sources(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_doc.mufinder", "Quantum")
    window._document = doc
    doc.project.recommended_cutoff_energy = 620.0
    doc.project.recommended_kpoint_grid = (2, 2, 1)

    relaxed_structure = Structure(
        label="site relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_structure.id] = relaxed_structure

    source_job = Job(
        label="mu-site",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[source_job.id] = source_job
    run = CalculationRun(
        job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True
    )
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
    doc.sites[site.id] = site
    source_dir = doc.project_root / "files" / WorkflowStage.MUON_RELAXATION.value / source_job.label
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / f"{source_job.label}.cell").write_text(
        "\n".join(
            [
                "%BLOCK LATTICE_CART",
                "ang",
                "5.0 0.0 0.0",
                "0.0 5.0 0.0",
                "0.0 0.0 5.0",
                "%ENDBLOCK LATTICE_CART",
                "",
                "%BLOCK POSITIONS_FRAC",
                "Si 0.0 0.0 0.0",
                "H:mu 0.25 0.25 0.25",
                "%ENDBLOCK POSITIONS_FRAC",
                "",
                "KPOINT_MP_GRID 4 1 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (source_dir / f"{source_job.label}.param").write_text(
        "TASK : GEOMETRYOPTIMIZATION\nCUT_OFF_ENERGY : 700 eV\n",
        encoding="utf-8",
    )

    jobs = window._quantum_corrections_controller.materialize([str(site.id)])

    assert len(jobs) == 1
    assert jobs[0].calc_type == QUANTUM_PHONON_CALC_TYPE
    assert jobs[0].label == "quantum_phonon_siteA"
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    assert output_dir.name == "quantum_siteA"
    cell_text = (output_dir / "quantum_siteA.cell").read_text(encoding="utf-8")
    param_text = (output_dir / "quantum_siteA.param").read_text(encoding="utf-8")
    assert "PHONON_KPOINT_MP_GRID 1 1 1" in cell_text
    assert "%BLOCK SPECIES_MASS" in cell_text
    assert "%BLOCK SPECIES_GAMMA" in cell_text
    assert "KPOINT_MP_GRID 4 1 1" in cell_text
    assert "KPOINT_MP_GRID 2 2 1" not in cell_text
    assert "TASK : PHONON" in param_text
    assert "PHONON_METHOD : FINITEDISPLACEMENT" in param_text
    assert "CONTINUATION : DEFAULT" in param_text
    assert "CUT_OFF_ENERGY : 700 eV" in param_text
    assert "CUT_OFF_ENERGY : 620" not in param_text
    assert "ELEC_ENERGY_TOL : 1e-09 eV" in param_text
    assert "NUM_FARMS" not in param_text


def test_main_window_materializes_a_configured_quantum_phonon_job_and_shows_its_farms(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_farms_doc.mufinder", "QuantumFarms")
    window._document = doc

    relaxed_structure = Structure(
        label="site relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_structure.id] = relaxed_structure

    source_job = Job(
        label="mu-site",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[source_job.id] = source_job
    run = CalculationRun(
        job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True
    )
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
    doc.sites[site.id] = site
    source_dir = doc.project_root / "files" / WorkflowStage.MUON_RELAXATION.value / source_job.label
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / f"{source_job.label}.cell").write_text(
        "%BLOCK LATTICE_CART\nang\n5.0 0.0 0.0\n0.0 5.0 0.0\n0.0 0.0 5.0\n%ENDBLOCK LATTICE_CART\n\n"
        "%BLOCK POSITIONS_FRAC\nSi 0.0 0.0 0.0\nH:mu 0.25 0.25 0.25\n%ENDBLOCK POSITIONS_FRAC\n\n"
        "KPOINT_MP_GRID 4 1 1\n",
        encoding="utf-8",
    )
    (source_dir / f"{source_job.label}.param").write_text(
        "TASK : GEOMETRYOPTIMIZATION\nCUT_OFF_ENERGY : 700 eV\n",
        encoding="utf-8",
    )

    definition = window._quantum_corrections_controller.ensure_definition([str(site.id)])
    assert definition is not None
    definition.parameters["configuration"]["row_settings"] = {
        str(site.id): {
            "parameter_values": {
                "phonon_method": "DFPT",
                "cut_off_energy": {"value": 650.0, "unit": "eV"},
                "kpoint_mp_grid": [2, 2, 2],
            },
            "pseudopotential_family": "norm_conserving",
            "num_farms": 4,
        }
    }

    window._refresh_quantum_corrections_view()
    assert window._quantum_corrections_view._sources_table.item(0, 3).text() == "6"
    assert window._quantum_corrections_view._sources_table.item(0, 5).text() == "4"
    assert window._quantum_corrections_view._sources_table.item(0, 6).text() == "ready"

    phonon_jobs = window._quantum_corrections_controller.materialize([str(site.id)])
    assert window._quantum_corrections_view._sources_table.item(0, 6).text() == "defined"

    assert len(phonon_jobs) == 1
    assert phonon_jobs[0].calc_type == QUANTUM_PHONON_CALC_TYPE
    assert phonon_jobs[0].label == "quantum_phonon_siteA"
    output_dir = window._job_output_directory(phonon_jobs[0])
    assert output_dir is not None
    assert output_dir.name == "quantum_siteA"
    cell_text = (output_dir / "quantum_siteA.cell").read_text(encoding="utf-8")
    param_text = (output_dir / "quantum_siteA.param").read_text(encoding="utf-8")
    assert "PHONON_KPOINT_MP_GRID 1 1 1" in cell_text
    assert "KPOINT_MP_GRID 2 2 2" in cell_text
    # A row an older version configured for DFPT runs as finite displacement.
    assert "PHONON_METHOD : FINITEDISPLACEMENT" in param_text
    assert "CONTINUATION : DEFAULT" in param_text
    assert "NUM_FARMS : 4" in param_text
    assert "TASK : PHONON" in param_text


def test_main_window_applies_quantum_row_configuration_to_materialized_job(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_row_config_doc.mufinder", "QuantumRowConfig")
    window._document = doc

    relaxed_structure = Structure(
        label="site relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_structure.id] = relaxed_structure

    source_job = Job(
        label="mu-site",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[source_job.id] = source_job
    run = CalculationRun(
        job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True
    )
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
    doc.sites[site.id] = site
    source_dir = doc.project_root / "files" / WorkflowStage.MUON_RELAXATION.value / source_job.label
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / f"{source_job.label}.cell").write_text(
        "\n".join(
            [
                "%BLOCK LATTICE_CART",
                "ang",
                "5.0 0.0 0.0",
                "0.0 5.0 0.0",
                "0.0 0.0 5.0",
                "%ENDBLOCK LATTICE_CART",
                "",
                "%BLOCK POSITIONS_FRAC",
                "Si 0.0 0.0 0.0",
                "H:mu 0.25 0.25 0.25",
                "%ENDBLOCK POSITIONS_FRAC",
                "",
                "KPOINT_MP_GRID 4 4 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (source_dir / f"{source_job.label}.param").write_text(
        "TASK : GEOMETRYOPTIMIZATION\nCUT_OFF_ENERGY : 710 eV\n",
        encoding="utf-8",
    )

    definition = window._quantum_corrections_controller.ensure_definition([str(site.id)])
    assert definition is not None
    definition.parameters["configuration"]["row_settings"] = {
        str(site.id): {
            "parameter_values": {
                "phonon_method": "DFPT",
                "cut_off_energy": {"value": 650.0, "unit": "eV"},
                "kpoint_mp_grid": [2, 2, 2],
            },
            "pseudopotential_family": "norm_conserving",
            "num_farms": 3,
        }
    }

    window._refresh_quantum_corrections_view()
    configure_button = window._quantum_corrections_view._sources_table.cellWidget(0, 7)
    assert configure_button.text() == "Configured…"

    jobs = window._quantum_corrections_controller.materialize([str(site.id)])

    assert len(jobs) == 1
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    cell_text = (output_dir / "quantum_siteA.cell").read_text(encoding="utf-8")
    param_text = (output_dir / "quantum_siteA.param").read_text(encoding="utf-8")
    # A row an older version configured for DFPT runs as finite displacement.
    assert "PHONON_METHOD : FINITEDISPLACEMENT" in param_text
    assert "CUT_OFF_ENERGY : 650" in param_text
    assert "CUT_OFF_ENERGY : 710" not in param_text
    assert "KPOINT_MP_GRID 2 2 2" in cell_text
    assert "KPOINT_MP_GRID 4 4 1" not in cell_text
    assert "TASK : PHONON" in param_text
    assert "NUM_FARMS : 3" in param_text


def test_main_window_quantum_corrections_hides_fine_cutoff_from_default_rows(qapp) -> None:
    window = MainWindow()

    dialog = QuantumCorrectionsConfigDialog(
        specs=window._quantum_corrections_controller.parameter_specs()
    )

    assert "fine_cut_off_energy" not in dialog._visible_names()
    assert "fine_cut_off_energy" in {spec.name for spec in dialog._available_advanced_specs()}
    assert dialog._visible_names()[:4] == [
        "basis_precision",
        "cut_off_energy",
        "elec_energy_tol",
        "kpoint_mp_grid",
    ]
    assert dialog.collect_values()["elec_energy_tol"] == ValueWithUnit(1e-09, "eV")

    dialog.close()
    window.close()


def test_main_window_energy_barriers_hides_fine_cutoff_from_default_rows(qapp) -> None:
    window = MainWindow()

    # Mirror the locked values the real NEB configure handler passes, including the
    # locked climbing flag (a non-string locked value must not crash dialog construction).
    neb_dialog = ParameterDialog(
        specs=window._energy_barriers_controller.parameter_specs(),
        workflow_stage="energy_barriers",
        locked_values={
            "task": "TRANSITIONSTATESEARCH",
            "tssearch_method": "NEB",
            "tssearch_neb_climbing": "TRUE",
        },
    )

    assert "fine_cut_off_energy" not in neb_dialog._visible_names()
    assert "fine_cut_off_energy" in {spec.name for spec in neb_dialog._available_advanced_specs()}
    assert neb_dialog._visible_names()[:4] == [
        "basis_precision",
        "cut_off_energy",
        "elec_energy_tol",
        "kpoint_mp_grid",
    ]
    # Climbing is locked on and shown as a fixed row.
    assert "tssearch_neb_climbing" in neb_dialog._visible_names()

    neb_dialog.close()
    window.close()


def test_main_window_quantum_corrections_uses_numeric_suffixes_for_duplicate_site_labels(
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "quantum_duplicate_labels_doc.mufinder", "Quantum Duplicates"
    )
    window._document = doc

    for index, muon_x in enumerate((0.25, 0.75), start=1):
        relaxed_structure = Structure(
            label=f"site relaxed {index}",
            origin="muon_relaxation",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si", "H:mu"],
            fractional_positions=[[0.0, 0.0, 0.0], [muon_x, 0.25, 0.25]],
            properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
        )
        doc.structures[relaxed_structure.id] = relaxed_structure

        source_job = Job(
            label=f"mu-site-{index}",
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type="castep_positivemuon_relaxation",
            status=JobStatus.COMPLETED,
        )
        doc.jobs[source_job.id] = source_job
        run = CalculationRun(
            job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True
        )
        result = Result(
            calculation_run_id=run.id,
            result_type=source_job.calc_type,
            data={"total_energy": -1.0 - index},
            derived_structure_ids=[relaxed_structure.id],
        )
        run.result_id = result.id
        doc.runs[run.id] = run
        doc.results[result.id] = result
        site = MuonSite(
            label="siteA", status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0 - index
        )
        doc.sites[site.id] = site

        source_dir = (
            doc.project_root / "files" / WorkflowStage.MUON_RELAXATION.value / source_job.label
        )
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / f"{source_job.label}.cell").write_text(
            "\n".join(
                [
                    "%BLOCK LATTICE_CART",
                    "ang",
                    "5.0 0.0 0.0",
                    "0.0 5.0 0.0",
                    "0.0 0.0 5.0",
                    "%ENDBLOCK LATTICE_CART",
                    "",
                    "%BLOCK POSITIONS_FRAC",
                    "Si 0.0 0.0 0.0",
                    f"H:mu {muon_x} 0.25 0.25",
                    "%ENDBLOCK POSITIONS_FRAC",
                    "",
                    "KPOINT_MP_GRID 2 2 1",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (source_dir / f"{source_job.label}.param").write_text(
            "TASK : GEOMETRYOPTIMIZATION\nCUT_OFF_ENERGY : 650 eV\n",
            encoding="utf-8",
        )

    labels = [
        window._quantum_corrections_controller.source_label(candidate)
        for candidate in window._quantum_corrections_controller.candidates()
    ]

    assert labels == ["siteA", "siteA_2"]

    jobs = window._quantum_corrections_controller.materialize(
        [candidate.key for candidate in window._quantum_corrections_controller.candidates()]
    )

    assert [job.label for job in jobs] == ["quantum_phonon_siteA", "quantum_phonon_siteA_2"]

    window.close()


def test_main_window_materialize_quantum_corrections_repairs_stale_suffixed_job_labels(
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "quantum_stale_label_doc.mufinder", "Quantum Stale Label"
    )
    window._document = doc

    relaxed_structure = Structure(
        label="site relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_structure.id] = relaxed_structure

    source_job = Job(
        label="mu-site",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[source_job.id] = source_job
    run = CalculationRun(
        job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True
    )
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
    doc.sites[site.id] = site

    definition = window._quantum_corrections_controller.ensure_definition([str(site.id)])
    assert definition is not None
    stale_job = service.add_job(
        doc,
        label=f"quantum_phonon_siteA_{str(site.id)[:8]}",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
    )
    definition.parameters["phonon_job_ids_by_label"] = {str(site.id): str(stale_job.id)}

    jobs = window._quantum_corrections_controller.materialize([str(site.id)])

    assert [job.id for job in jobs] == [stale_job.id]
    assert stale_job.label == "quantum_phonon_siteA"
    assert definition.parameters["phonon_job_ids_by_label"][str(site.id)] == str(stale_job.id)
    output_dir = window._job_output_directory(stale_job)
    assert output_dir is not None
    assert output_dir.name == "quantum_siteA"

    window.close()


def test_main_window_quantum_corrections_allows_editing_eligible_site_name(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_rename_doc.mufinder", "Quantum Rename")
    window._document = doc
    set_show_every_relaxed_site(doc, QUANTUM_CORRECTIONS_SCOPE, True)

    relaxed_structure = Structure(
        label="site relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_structure.id] = relaxed_structure

    source_job = Job(
        label="mu-site",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[source_job.id] = source_job
    run = CalculationRun(
        job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True
    )
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
    doc.sites[site.id] = site

    window._refresh_quantum_corrections_view()
    item = window._quantum_corrections_view._sources_table.item(0, 0)

    assert item is not None
    item.setText("Renamed Site")

    assert doc.sites[site.id].label == "Renamed Site"
    assert window._quantum_corrections_view._sources_table.item(0, 0).text() == "Renamed Site"

    window.close()


def test_main_window_applies_energy_barrier_row_configuration_and_opens_path_preview(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_row_config_doc.mufinder", "BarrierRowConfig")
    window._document = doc

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    relaxed_a = Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={
            "source_workflow_stage": WorkflowStage.MUON_RELAXATION.value,
            "atomic_spins": [1.0, -1.0, 0.0],
        },
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={
            "source_workflow_stage": WorkflowStage.MUON_RELAXATION.value,
            "atomic_spins": [1.0, -1.0, 0.0],
        },
    )
    doc.structures[relaxed_a.id] = relaxed_a
    doc.structures[relaxed_b.id] = relaxed_b

    job_a = Job(
        label="mu-site-a",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    job_b = Job(
        label="mu-site-b",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[job_a.id] = job_a
    doc.jobs[job_b.id] = job_b
    run_a = CalculationRun(job_id=job_a.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    run_b = CalculationRun(job_id=job_b.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result_a = Result(
        calculation_run_id=run_a.id,
        result_type=job_a.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_a.id],
    )
    result_b = Result(
        calculation_run_id=run_b.id,
        result_type=job_b.calc_type,
        data={"total_energy": -0.9},
        derived_structure_ids=[relaxed_b.id],
    )
    run_a.result_id = result_a.id
    run_b.result_id = result_b.id
    doc.runs[run_a.id] = run_a
    doc.runs[run_b.id] = run_b
    doc.results[result_a.id] = result_a
    doc.results[result_b.id] = result_b
    site_a = MuonSite(
        label="siteA", status="relaxed", relaxation_run_id=run_a.id, total_energy_eV=-1.0
    )
    site_b = MuonSite(
        label="siteB", status="relaxed", relaxation_run_id=run_b.id, total_energy_eV=-0.9
    )
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._energy_barriers_controller.generate_paths(str(site_a.id), str(site_b.id))
    candidate = window._energy_barriers_controller.candidates()[0]
    neb_definition = window._energy_barriers_controller.ensure_definition([candidate.key])
    assert neb_definition is not None
    neb_definition.parameters["configuration"]["row_settings"] = {
        candidate.key: {
            "parameter_values": {
                "kpoint_mp_grid": [2, 1, 1],
                "tssearch_neb_max_iter": 18,
            }
        }
    }

    window._ensure_energy_barriers_view()
    window._refresh_energy_barriers_view()
    configure_button = window._energy_barriers_view._sources_table.cellWidget(0, 6)
    assert configure_button.text() == "Configured…"

    jobs = window._energy_barriers_controller.materialize([candidate.key])

    assert len(jobs) == 1
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    cell_text = (output_dir / f"{output_dir.name}.cell").read_text(encoding="utf-8")
    param_text = (output_dir / f"{output_dir.name}.param").read_text(encoding="utf-8")
    assert "KPOINT_MP_GRID 2 1 1" in cell_text
    assert "TSSEARCH_NEB_MAX_ITER : 18" in param_text

    opened: dict[str, object] = {}

    class _FakeDialog:
        def __init__(self, *, structure, start_fractional, end_fractional, title, parent=None):
            opened["structure"] = structure
            opened["start_fractional"] = tuple(start_fractional)
            opened["end_fractional"] = tuple(end_fractional)
            opened["title"] = title

        def exec(self):
            opened["shown"] = True
            return 0

    monkeypatch.setattr(main_window_module, "EnergyBarrierPathViewerDialog", _FakeDialog)

    window._on_view_energy_barrier_source(candidate.key)

    assert opened["shown"] is True
    assert opened["structure"].origin == "energy_barriers_path_preview"
    assert opened["start_fractional"] == candidate.path.start_fractional
    assert opened["end_fractional"] == candidate.path.end_fractional


def test_main_window_energy_barrier_paths_persist_when_endpoint_selection_changes(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_persist_doc.mufinder", "BarrierPersist")
    window._document = doc

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    relaxed_a = Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_a.id] = relaxed_a
    doc.structures[relaxed_b.id] = relaxed_b

    job_a = Job(
        label="mu-site-a",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    job_b = Job(
        label="mu-site-b",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[job_a.id] = job_a
    doc.jobs[job_b.id] = job_b
    run_a = CalculationRun(job_id=job_a.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    run_b = CalculationRun(job_id=job_b.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result_a = Result(
        calculation_run_id=run_a.id,
        result_type=job_a.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_a.id],
    )
    result_b = Result(
        calculation_run_id=run_b.id,
        result_type=job_b.calc_type,
        data={"total_energy": -0.9},
        derived_structure_ids=[relaxed_b.id],
    )
    run_a.result_id = result_a.id
    run_b.result_id = result_b.id
    doc.runs[run_a.id] = run_a
    doc.runs[run_b.id] = run_b
    doc.results[result_a.id] = result_a
    doc.results[result_b.id] = result_b
    site_a = MuonSite(
        label="siteA", status="relaxed", relaxation_run_id=run_a.id, total_energy_eV=-1.0
    )
    site_b = MuonSite(
        label="siteB", status="relaxed", relaxation_run_id=run_b.id, total_energy_eV=-0.9
    )
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._energy_barriers_controller.generate_paths(str(site_a.id), str(site_b.id), "")
    generated_before = window._energy_barriers_controller.candidates()

    assert generated_before

    window._energy_barriers_controller.select_endpoints(str(site_b.id), str(site_a.id))
    generated_after = window._energy_barriers_controller.candidates()

    assert [candidate.key for candidate in generated_after] == [
        candidate.key for candidate in generated_before
    ]
    window.close()


def test_main_window_energy_barrier_defaults_apply_equivalent_density_kpoints_for_supercell(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "barrier_supercell_kpoints_doc.mufinder", "BarrierSupercellKPoints"
    )
    window._document = doc

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id
    doc.project.recommended_kpoint_grid = (4, 4, 4)
    doc.project.recommended_kpoint_lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    supercell_species = ["Si"] * 8 + ["H:mu"]
    supercell_positions_a = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.125, 0.125, 0.125],
    ]
    supercell_positions_b = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.375, 0.125, 0.125],
    ]
    relaxed_a = Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        species=supercell_species,
        fractional_positions=supercell_positions_a,
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        species=supercell_species,
        fractional_positions=supercell_positions_b,
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_a.id] = relaxed_a
    doc.structures[relaxed_b.id] = relaxed_b

    job_a = Job(
        label="mu-site-a",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    job_b = Job(
        label="mu-site-b",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[job_a.id] = job_a
    doc.jobs[job_b.id] = job_b
    run_a = CalculationRun(job_id=job_a.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    run_b = CalculationRun(job_id=job_b.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result_a = Result(
        calculation_run_id=run_a.id,
        result_type=job_a.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_a.id],
    )
    result_b = Result(
        calculation_run_id=run_b.id,
        result_type=job_b.calc_type,
        data={"total_energy": -0.9},
        derived_structure_ids=[relaxed_b.id],
    )
    run_a.result_id = result_a.id
    run_b.result_id = result_b.id
    doc.runs[run_a.id] = run_a
    doc.runs[run_b.id] = run_b
    doc.results[result_a.id] = result_a
    doc.results[result_b.id] = result_b
    site_a = MuonSite(
        label="siteA", status="relaxed", relaxation_run_id=run_a.id, total_energy_eV=-1.0
    )
    site_b = MuonSite(
        label="siteB", status="relaxed", relaxation_run_id=run_b.id, total_energy_eV=-0.9
    )
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._energy_barriers_controller.generate_paths(str(site_a.id), str(site_b.id))
    candidate = window._energy_barriers_controller.candidates()[0]

    jobs = window._energy_barriers_controller.materialize([candidate.key])

    assert len(jobs) == 1
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    cell_text = (output_dir / f"{output_dir.name}.cell").read_text(encoding="utf-8")
    assert "KPOINT_MP_GRID 2 2 2" in cell_text
    window.close()


def test_main_window_energy_barriers_view_loads_with_existing_state(qapp, tmp_path: Path) -> None:
    """Navigating to energy barriers with pre-existing generated sources and max_path_length works."""
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "barrier_existing_state.mufinder", "BarrierExistingState"
    )
    window._document = doc

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    relaxed_a = Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_a.id] = relaxed_a
    doc.structures[relaxed_b.id] = relaxed_b

    job_a = Job(
        label="mu-site-a",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    job_b = Job(
        label="mu-site-b",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[job_a.id] = job_a
    doc.jobs[job_b.id] = job_b
    run_a = CalculationRun(job_id=job_a.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    run_b = CalculationRun(job_id=job_b.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result_a = Result(
        calculation_run_id=run_a.id,
        result_type=job_a.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_a.id],
    )
    result_b = Result(
        calculation_run_id=run_b.id,
        result_type=job_b.calc_type,
        data={"total_energy": -0.9},
        derived_structure_ids=[relaxed_b.id],
    )
    run_a.result_id = result_a.id
    run_b.result_id = result_b.id
    doc.runs[run_a.id] = run_a
    doc.runs[run_b.id] = run_b
    doc.results[result_a.id] = result_a
    doc.results[result_b.id] = result_b
    site_a = MuonSite(
        label="siteA", status="relaxed", relaxation_run_id=run_a.id, total_energy_eV=-1.0
    )
    site_b = MuonSite(
        label="siteB", status="relaxed", relaxation_run_id=run_b.id, total_energy_eV=-0.9
    )
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    # Generate paths first so state has generated_sources and max_path_length_ang.
    window._energy_barriers_controller.generate_paths(str(site_a.id), str(site_b.id), "4.0")
    assert doc.ui_state.get("energy_barriers_form", {}).get("generated_sources")

    # Simulate re-opening the project: create a fresh window that only has the document.
    window2 = MainWindow()
    window2._document = doc

    # Navigate to energy barriers — this is what happens when the user opens the section.
    window2._ensure_energy_barriers_view()
    window2._refresh_energy_barriers_view()  # Must not raise

    # The view should show the generated paths.
    candidates = window2._energy_barriers_controller.candidates()
    assert len(candidates) > 0

    # NEB configure dialog values must be gettable without error.
    definition = window2._energy_barriers_controller.active_definition()
    for candidate in candidates:
        values, _advanced, _extras = window2._energy_barriers_controller.row_configuration(
            definition,
            candidate.key,
            target_lattice=candidate.reactant.structure.lattice_vectors,
        )
        assert isinstance(values, dict)

    window.close()
    window2.close()


def test_main_window_energy_barrier_strips_legacy_tss_params_from_stored_config(
    qapp, tmp_path: Path
) -> None:
    """Legacy projects with zero-value tolerance params and tssearch_lst_max_iter open cleanly."""
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_legacy_doc.mufinder", "BarrierLegacy")
    window._document = doc

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    relaxed_a = Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_a.id] = relaxed_a
    doc.structures[relaxed_b.id] = relaxed_b

    job_a = Job(
        label="mu-site-a",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    job_b = Job(
        label="mu-site-b",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[job_a.id] = job_a
    doc.jobs[job_b.id] = job_b
    run_a = CalculationRun(job_id=job_a.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    run_b = CalculationRun(job_id=job_b.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result_a = Result(
        calculation_run_id=run_a.id,
        result_type=job_a.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_a.id],
    )
    result_b = Result(
        calculation_run_id=run_b.id,
        result_type=job_b.calc_type,
        data={"total_energy": -0.9},
        derived_structure_ids=[relaxed_b.id],
    )
    run_a.result_id = result_a.id
    run_b.result_id = result_b.id
    doc.runs[run_a.id] = run_a
    doc.runs[run_b.id] = run_b
    doc.results[result_a.id] = result_a
    doc.results[result_b.id] = result_b
    site_a = MuonSite(
        label="siteA", status="relaxed", relaxation_run_id=run_a.id, total_energy_eV=-1.0
    )
    site_b = MuonSite(
        label="siteB", status="relaxed", relaxation_run_id=run_b.id, total_energy_eV=-0.9
    )
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._energy_barriers_controller.generate_paths(str(site_a.id), str(site_b.id))
    candidate = window._energy_barriers_controller.candidates()[0]

    # Simulate legacy stored config as it would have been written by the old TSS dialog:
    # tssearch_force/disp/energy_tol stored as 0 with cross-reference unit strings,
    # and tssearch_lst_max_iter present as an obsolete keyword.
    definition = window._energy_barriers_controller.ensure_definition([candidate.key])
    assert definition is not None
    definition.parameters["configuration"]["row_settings"] = {
        candidate.key: {
            "parameter_values": {
                "cut_off_energy": {"value": 700.0, "unit": "eV"},
                "fine_cut_off_energy": {"value": 0.0, "unit": "FINE_GMAX"},
                "tssearch_lst_max_iter": 20,
                "tssearch_force_tol": {"value": 0.0, "unit": "GEOM_FORCE_TOL"},
                "tssearch_disp_tol": {"value": 0.0, "unit": "GEOM_DISP_TOL"},
                "tssearch_energy_tol": {"value": 0.0, "unit": "GEOM_ENERGY_TOL"},
            }
        }
    }

    jobs = window._energy_barriers_controller.materialize([candidate.key])

    assert len(jobs) == 1
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    param_text = (output_dir / f"{output_dir.name}.param").read_text(encoding="utf-8")
    # Legacy zero-value tolerances and obsolete keyword must not appear in the output.
    assert "FINE_CUT_OFF_ENERGY" not in param_text
    assert "TSSEARCH_FORCE_TOL" not in param_text
    assert "TSSEARCH_DISP_TOL" not in param_text
    assert "TSSEARCH_ENERGY_TOL" not in param_text
    assert "TSSEARCH_LST_MAX_ITER" not in param_text
    # Valid user-configured value must still be written.
    assert "CUT_OFF_ENERGY : 700" in param_text
    window.close()


def test_main_window_neb_inherits_kpoints_from_reactant_relaxation_cell_file(
    qapp, tmp_path: Path
) -> None:
    """NEB jobs inherit the k-point grid from the reactant relaxation's output cell file."""
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "barrier_cell_kpoints_doc.mufinder", "BarrierCellKPoints"
    )
    window._document = doc

    doc.project.recommended_kpoint_grid = (4, 4, 4)
    # Deliberately omit recommended_kpoint_lattice so the project-default scaling path cannot work.

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    supercell_species = ["Si"] * 8 + ["H:mu"]
    supercell_positions_a = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.125, 0.125, 0.125],
    ]
    supercell_positions_b = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.375, 0.125, 0.125],
    ]
    relaxed_a = Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        species=supercell_species,
        fractional_positions=supercell_positions_a,
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        species=supercell_species,
        fractional_positions=supercell_positions_b,
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_a.id] = relaxed_a
    doc.structures[relaxed_b.id] = relaxed_b

    job_a = Job(
        label="mu-site-a",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    job_b = Job(
        label="mu-site-b",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[job_a.id] = job_a
    doc.jobs[job_b.id] = job_b
    run_a = CalculationRun(job_id=job_a.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    run_b = CalculationRun(job_id=job_b.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result_a = Result(
        calculation_run_id=run_a.id,
        result_type=job_a.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_a.id],
    )
    result_b = Result(
        calculation_run_id=run_b.id,
        result_type=job_b.calc_type,
        data={"total_energy": -0.9},
        derived_structure_ids=[relaxed_b.id],
    )
    run_a.result_id = result_a.id
    run_b.result_id = result_b.id
    doc.runs[run_a.id] = run_a
    doc.runs[run_b.id] = run_b
    doc.results[result_a.id] = result_a
    doc.results[result_b.id] = result_b
    site_a = MuonSite(
        label="siteA", status="relaxed", relaxation_run_id=run_a.id, total_energy_eV=-1.0
    )
    site_b = MuonSite(
        label="siteB", status="relaxed", relaxation_run_id=run_b.id, total_energy_eV=-0.9
    )
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    # Write the reactant job's output cell file with the supercell k-point grid.
    source_dir_a = doc.project_root / "files" / WorkflowStage.MUON_RELAXATION.value / job_a.label
    source_dir_a.mkdir(parents=True, exist_ok=True)
    (source_dir_a / f"{job_a.label}.cell").write_text(
        "\n".join(
            [
                "%BLOCK LATTICE_CART",
                "ang",
                "8.0 0.0 0.0",
                "0.0 8.0 0.0",
                "0.0 0.0 8.0",
                "%ENDBLOCK LATTICE_CART",
                "%BLOCK POSITIONS_FRAC",
                "Si 0.0 0.0 0.0",
                "H:mu 0.125 0.125 0.125",
                "%ENDBLOCK POSITIONS_FRAC",
                "KPOINT_MP_GRID 2 2 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    window._energy_barriers_controller.generate_paths(str(site_a.id), str(site_b.id))
    candidate = window._energy_barriers_controller.candidates()[0]
    jobs = window._energy_barriers_controller.materialize([candidate.key])

    assert len(jobs) == 1
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    cell_text = (output_dir / f"{output_dir.name}.cell").read_text(encoding="utf-8")
    # Must use supercell grid from reactant job's cell file, not the raw project default (4 4 4).
    assert "KPOINT_MP_GRID 2 2 2" in cell_text
    assert "KPOINT_MP_GRID 4 4 4" not in cell_text
    window.close()


def test_main_window_generates_same_site_energy_barrier_paths(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_same_site_doc.mufinder", "BarrierSameSite")
    window._document = doc

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    relaxed = Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed.id] = relaxed

    job = Job(
        label="mu-site-a",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[job.id] = job
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result = Result(
        calculation_run_id=run.id,
        result_type=job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
    doc.sites[site.id] = site

    window._energy_barriers_controller.generate_paths(str(site.id), str(site.id), "")
    generated = window._energy_barriers_controller.candidates()

    assert generated
    assert all(candidate.path.length_ang > 0.0 for candidate in generated)
    assert all(candidate.reactant.site.id == site.id for candidate in generated)
    assert all(candidate.product.site.id == site.id for candidate in generated)
    window.close()


def test_main_window_generates_energy_barrier_paths_with_user_max_length(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_max_length_doc.mufinder", "BarrierMaxLength")
    window._document = doc

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    relaxed_a = Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_a.id] = relaxed_a
    doc.structures[relaxed_b.id] = relaxed_b

    job_a = Job(
        label="mu-site-a",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    job_b = Job(
        label="mu-site-b",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[job_a.id] = job_a
    doc.jobs[job_b.id] = job_b
    run_a = CalculationRun(job_id=job_a.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    run_b = CalculationRun(job_id=job_b.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result_a = Result(
        calculation_run_id=run_a.id,
        result_type=job_a.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_a.id],
    )
    result_b = Result(
        calculation_run_id=run_b.id,
        result_type=job_b.calc_type,
        data={"total_energy": -0.9},
        derived_structure_ids=[relaxed_b.id],
    )
    run_a.result_id = result_a.id
    run_b.result_id = result_b.id
    doc.runs[run_a.id] = run_a
    doc.runs[run_b.id] = run_b
    doc.results[result_a.id] = result_a
    doc.results[result_b.id] = result_b
    site_a = MuonSite(
        label="siteA", status="relaxed", relaxation_run_id=run_a.id, total_energy_eV=-1.0
    )
    site_b = MuonSite(
        label="siteB", status="relaxed", relaxation_run_id=run_b.id, total_energy_eV=-0.9
    )
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._energy_barriers_controller.generate_paths(str(site_a.id), str(site_b.id), "1.9")

    assert window._energy_barriers_controller.candidates() == []
    assert window._document.ui_state["energy_barriers_form"]["max_path_length_ang"] == 1.9
    window.close()


def test_main_window_removes_energy_barrier_generated_path(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_remove_doc.mufinder", "BarrierRemove")
    window._document = doc

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    relaxed_a = Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_a.id] = relaxed_a
    doc.structures[relaxed_b.id] = relaxed_b

    job_a = Job(
        label="mu-site-a",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    job_b = Job(
        label="mu-site-b",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[job_a.id] = job_a
    doc.jobs[job_b.id] = job_b
    run_a = CalculationRun(job_id=job_a.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    run_b = CalculationRun(job_id=job_b.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result_a = Result(
        calculation_run_id=run_a.id,
        result_type=job_a.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_a.id],
    )
    result_b = Result(
        calculation_run_id=run_b.id,
        result_type=job_b.calc_type,
        data={"total_energy": -0.9},
        derived_structure_ids=[relaxed_b.id],
    )
    run_a.result_id = result_a.id
    run_b.result_id = result_b.id
    doc.runs[run_a.id] = run_a
    doc.runs[run_b.id] = run_b
    doc.results[result_a.id] = result_a
    doc.results[result_b.id] = result_b
    site_a = MuonSite(
        label="siteA", status="relaxed", relaxation_run_id=run_a.id, total_energy_eV=-1.0
    )
    site_b = MuonSite(
        label="siteB", status="relaxed", relaxation_run_id=run_b.id, total_energy_eV=-0.9
    )
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._energy_barriers_controller.generate_paths(str(site_a.id), str(site_b.id), "")
    candidate = window._energy_barriers_controller.candidates()[0]
    definition = window._energy_barriers_controller.ensure_definition([candidate.key])

    assert definition is not None

    window._on_remove_energy_barrier_sources([candidate.key])

    assert candidate.key not in {
        str(source.get("key", "")).strip()
        for source in energy_barrier_generated_sources(window._document)
    }
    assert candidate.key not in {
        str(source.get("key", "")).strip()
        for source in definition.parameters["configuration"].get("sources", [])
        if isinstance(source, dict)
    }
    window.close()


def test_main_window_records_energy_barrier_result_from_ts_file(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_result_doc.mufinder", "BarrierResult")
    window._document = doc

    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type="castep_transition_state_search_set",
        parameters={
            "output_directory": "files/energy_barriers",
            "job_ids_by_label": {},
        },
    )
    doc.calculations[definition.id] = definition

    job = service.add_job(
        doc,
        label="barrier_siteA_to_siteB_path1",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=TRANSITION_STATE_SEARCH_CALC_TYPE,
    )
    definition.parameters["job_ids_by_label"] = {"source-key": str(job.id)}
    output_dir = window._job_output_directory(job)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{output_dir.name}.ts").write_text("placeholder", encoding="utf-8")

    def _fake_parse(_path):
        return TransitionStatePath(
            images=(
                TransitionStateImage(
                    label="REA",
                    iteration=1,
                    reaction_coordinate=0.0,
                    energy=-1.0,
                    lattice_vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    species=("Si",),
                    cartesian_positions=((0.0, 0.0, 0.0),),
                    forces=((0.0, 0.0, 0.0),),
                ),
                TransitionStateImage(
                    label="TST",
                    iteration=1,
                    reaction_coordinate=0.5,
                    energy=-0.8,
                    lattice_vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    species=("Si",),
                    cartesian_positions=((0.0, 0.0, 0.0),),
                    forces=((0.0, 0.0, 0.0),),
                ),
                TransitionStateImage(
                    label="PRO",
                    iteration=1,
                    reaction_coordinate=1.0,
                    energy=-0.9,
                    lattice_vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    species=("Si",),
                    cartesian_positions=((0.0, 0.0, 0.0),),
                    forces=((0.0, 0.0, 0.0),),
                ),
            )
        )

    # The ``.ts`` is parsed once by the off-thread collector
    # (mufinder.services.job_artifacts) and carried to both the controller's
    # validate_artifacts and build_result, so patch the collector's parse helper.
    monkeypatch.setattr(job_artifacts_module, "parse_transition_state_file", _fake_parse)

    window._on_local_job_finished(
        job.id, main_window_module.CastepResult(completed=True, total_energy=-1.0)
    )

    stored = next(iter(doc.results.values()))
    assert stored.result_type == TRANSITION_STATE_SEARCH_CALC_TYPE
    assert stored.data["barrier_height_eV"] == 0.19999999999999996
    assert stored.data["num_images"] == 3

    window.close()


def test_main_window_materializes_neb_jobs_directly_from_endpoints(qapp, tmp_path: Path) -> None:
    # The simplified workflow: NEB is seeded straight from the two relaxed endpoints with no
    # transition-state search. CASTEP's NEB driver still requires an INTERMEDIATE structure, so the
    # app synthesises the minimum-image midpoint and writes both a product and an intermediate block.
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_neb_endpoints.mufinder", "BarrierNebEndpoints")
    window._document = doc

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    relaxed_a = Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"atomic_spins": [1.0, -1.0, 0.0]},
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"atomic_spins": [1.0, -1.0, 0.0]},
    )
    doc.structures[relaxed_a.id] = relaxed_a
    doc.structures[relaxed_b.id] = relaxed_b

    site_a = MuonSite(label="siteA", status="relaxed")
    site_b = MuonSite(label="siteB", status="relaxed")
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    # Only the endpoint path source exists: no TS job, no job_ids_by_label, no TS result.
    _add_transition_state_search_set_definition(
        doc,
        input_structure_ids=[relaxed_a.id, relaxed_b.id],
        input_site_ids=[site_a.id, site_b.id],
        parameters={
            "method": "NEB",
            "sources": [
                {
                    "key": "source-key",
                    "label": "siteA_to_siteB_path1",
                    "reactant_site_id": str(site_a.id),
                    "product_site_id": str(site_b.id),
                    "reactant_run_id": str(uuid4()),
                    "product_run_id": str(uuid4()),
                    "reactant_job_id": str(uuid4()),
                    "product_job_id": str(uuid4()),
                    "reactant_structure_id": str(relaxed_a.id),
                    "product_structure_id": str(relaxed_b.id),
                    "path_start_fractional": [0.25, 0.25, 0.25],
                    "path_end_fractional": [0.75, 0.25, 0.25],
                    "path_length_ang": 2.5,
                }
            ],
        },
    )

    jobs = window._energy_barriers_controller.materialize(["source-key"])

    assert len(jobs) == 1
    assert jobs[0].calc_type == NUDGED_ELASTIC_BAND_CALC_TYPE
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    cell_text = (output_dir / f"{output_dir.name}.cell").read_text(encoding="utf-8")
    param_text = (output_dir / f"{output_dir.name}.param").read_text(encoding="utf-8")
    assert "%BLOCK POSITIONS_FRAC_PRODUCT" in cell_text
    # CASTEP NEB requires the intermediate; the app supplies an IDPP-seeded central image. We don't
    # assert its exact (optimiser-determined) coordinates, only that the block exists and carries
    # the muon as a tagged species with three fractional coordinates.
    assert "%BLOCK POSITIONS_FRAC_INTERMEDIATE" in cell_text
    intermediate_section = cell_text.split("%BLOCK POSITIONS_FRAC_INTERMEDIATE", 1)[1]
    intermediate_section = intermediate_section.split("%ENDBLOCK POSITIONS_FRAC_INTERMEDIATE", 1)[0]
    muon_line = next(
        line for line in intermediate_section.splitlines() if line.strip().startswith("H:mu")
    )
    muon_coords = [float(value) for value in muon_line.split()[1:4]]
    assert len(muon_coords) == 3
    assert "TSSEARCH_METHOD : NEB" in param_text
    # Climbing image is locked on, and the image count defaults to an odd value (11).
    assert "TSSEARCH_NEB_CLIMBING : TRUE" in param_text
    assert "TSSEARCH_MAX_PATH_POINTS : 11" in param_text

    window.close()


def test_main_window_applies_neb_row_configuration_to_materialized_job(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "barrier_neb_row_config_doc.mufinder", "BarrierNebRowConfig"
    )
    window._document = doc
    doc.project.recommended_cutoff_energy = 680.0

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    relaxed_a = Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"atomic_spins": [1.0, -1.0, 0.0]},
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"atomic_spins": [1.0, -1.0, 0.0]},
    )
    doc.structures[relaxed_a.id] = relaxed_a
    doc.structures[relaxed_b.id] = relaxed_b

    site_a = MuonSite(label="siteA", status="relaxed")
    site_b = MuonSite(label="siteB", status="relaxed")
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    definition = _add_transition_state_search_set_definition(
        doc,
        input_structure_ids=[relaxed_a.id, relaxed_b.id],
        input_site_ids=[site_a.id, site_b.id],
        parameters={
            "method": "LSTQST",
            "sources": [
                {
                    "key": "source-key",
                    "label": "siteA_to_siteB_path1",
                    "reactant_site_id": str(site_a.id),
                    "product_site_id": str(site_b.id),
                    "reactant_run_id": str(uuid4()),
                    "product_run_id": str(uuid4()),
                    "reactant_job_id": str(uuid4()),
                    "product_job_id": str(uuid4()),
                    "reactant_structure_id": str(relaxed_a.id),
                    "product_structure_id": str(relaxed_b.id),
                    "path_start_fractional": [0.25, 0.25, 0.25],
                    "path_end_fractional": [0.75, 0.25, 0.25],
                    "path_length_ang": 2.5,
                }
            ],
        },
    )

    ts_job = service.add_job(
        doc,
        label="barrier_siteA_to_siteB_path1",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=TRANSITION_STATE_SEARCH_CALC_TYPE,
    )
    definition.parameters["job_ids_by_label"] = {"source-key": str(ts_job.id)}
    ts_run = CalculationRun(job_id=ts_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    ts_path = doc.project_root / "files" / "energy_barriers" / ts_job.label / f"{ts_job.label}.ts"
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    ts_path.write_text("placeholder", encoding="utf-8")
    ts_result = Result(
        calculation_run_id=ts_run.id,
        result_type=TRANSITION_STATE_SEARCH_CALC_TYPE,
        data={
            "transition_state_file": str(ts_path),
            "barrier_height_eV": 0.2,
            "num_images": 3,
        },
    )
    ts_run.result_id = ts_result.id
    doc.runs[ts_run.id] = ts_run
    doc.results[ts_result.id] = ts_result

    monkeypatch.setattr(
        energy_barriers_controller_module,
        "parse_transition_state_file",
        lambda _path: TransitionStatePath(
            images=(
                TransitionStateImage(
                    label="REA",
                    iteration=1,
                    reaction_coordinate=0.0,
                    energy=-1.0,
                    lattice_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
                    species=("Si", "Si", "H:mu"),
                    cartesian_positions=((0.0, 0.0, 0.0), (2.6, 2.5, 2.5), (1.25, 1.25, 1.25)),
                    forces=((0.0, 0.0, 0.0),) * 3,
                ),
                TransitionStateImage(
                    label="TST",
                    iteration=1,
                    reaction_coordinate=0.5,
                    energy=-0.8,
                    lattice_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
                    species=("Si", "Si", "H:mu"),
                    cartesian_positions=((0.0, 0.0, 0.0), (2.6, 2.5, 2.5), (2.5, 1.25, 1.25)),
                    forces=((0.0, 0.0, 0.0),) * 3,
                ),
                TransitionStateImage(
                    label="PRO",
                    iteration=1,
                    reaction_coordinate=1.0,
                    energy=-0.9,
                    lattice_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
                    species=("Si", "Si", "H:mu"),
                    cartesian_positions=((0.0, 0.0, 0.0), (2.6, 2.5, 2.5), (3.75, 1.25, 1.25)),
                    forces=((0.0, 0.0, 0.0),) * 3,
                ),
            )
        ),
    )

    neb_definition = window._energy_barriers_controller.ensure_definition(["source-key"])
    assert neb_definition is not None
    neb_definition.parameters["configuration"]["row_settings"] = {
        "source-key": {
            "parameter_values": {
                "kpoint_mp_grid": [1, 1, 2],
                "tssearch_neb_max_iter": 24,
                "tssearch_max_path_points": 8,
                "tssearch_force_tol": {"value": 0.15, "unit": "eV/ang"},
            }
        }
    }

    jobs = window._energy_barriers_controller.materialize(["source-key"])

    assert len(jobs) == 1
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    cell_text = (output_dir / f"{output_dir.name}.cell").read_text(encoding="utf-8")
    param_text = (output_dir / f"{output_dir.name}.param").read_text(encoding="utf-8")
    assert "KPOINT_MP_GRID 1 1 2" in cell_text
    assert "CUT_OFF_ENERGY : 680" in param_text
    assert "TSSEARCH_NEB_MAX_ITER : 24" in param_text
    # An even configured image count is rounded up to the next odd value for climbing NEB.
    assert "TSSEARCH_MAX_PATH_POINTS : 9" in param_text
    # A user-set force tolerance is written through in eV/Ang.
    assert "TSSEARCH_FORCE_TOL : 0.15 eV/ang" in param_text


def test_energy_barrier_neb_specs_expose_force_tol_editable_with_castep_default(qapp) -> None:
    # TSSEARCH_FORCE_TOL is an editable NEB row defaulting to CASTEP's own value (0.05), surfaced in
    # eV/Ang rather than its H/BOHR catalog unit so the editor labels it the way the .castep reports.
    window = MainWindow()
    try:
        spec = next(
            (
                s
                for s in window._energy_barriers_controller.parameter_specs()
                if s.name == "tssearch_force_tol"
            ),
            None,
        )
        assert spec is not None, "tssearch_force_tol should be an exposed NEB parameter"
        assert spec.unit == "eV/ang"
        assert spec.value_type == "physical"
        assert spec.default_value == ValueWithUnit(0.05, "eV/ang")
    finally:
        window.close()


def test_main_window_records_neb_result_and_loads_preview(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_neb_preview_doc.mufinder", "BarrierNebPreview")
    window._document = doc

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type="castep_nudged_elastic_band_set",
        parameters={
            "output_directory": "files/energy_barriers_neb",
            "job_ids_by_label": {},
        },
    )
    doc.calculations[definition.id] = definition

    job = service.add_job(
        doc,
        label="barrier_siteA_to_siteB_path1_neb",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
    )
    definition.parameters["job_ids_by_label"] = {"source-key": str(job.id)}
    output_dir = window._job_output_directory(job)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{output_dir.name}.ts").write_text("placeholder", encoding="utf-8")

    def _fake_parse(_path):
        return TransitionStatePath(
            images=(
                TransitionStateImage(
                    label="REA",
                    iteration=1,
                    reaction_coordinate=0.0,
                    energy=-1.0,
                    lattice_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
                    species=("Si", "Si", "H:mu"),
                    cartesian_positions=((0.0, 0.0, 0.0), (2.5, 2.5, 2.5), (1.25, 1.25, 1.25)),
                    forces=((0.0, 0.0, 0.0),) * 3,
                ),
                TransitionStateImage(
                    label="TST",
                    iteration=1,
                    reaction_coordinate=0.5,
                    energy=-0.8,
                    lattice_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
                    species=("Si", "Si", "H:mu"),
                    cartesian_positions=((0.0, 0.0, 0.0), (2.5, 2.5, 2.5), (2.5, 1.25, 1.25)),
                    forces=((0.0, 0.0, 0.0),) * 3,
                ),
                TransitionStateImage(
                    label="PRO",
                    iteration=1,
                    reaction_coordinate=1.0,
                    energy=-0.9,
                    lattice_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
                    species=("Si", "Si", "H:mu"),
                    cartesian_positions=((0.0, 0.0, 0.0), (2.5, 2.5, 2.5), (3.75, 1.25, 1.25)),
                    forces=((0.0, 0.0, 0.0),) * 3,
                ),
            )
        )

    # Completion parses the ``.ts`` once via the off-thread collector
    # (mufinder.services.job_artifacts). The later result-preview selection
    # (``EnergyBarriersController.show_neb_preview``) is a separate on-demand
    # parse in the controller module, so patch both.
    monkeypatch.setattr(job_artifacts_module, "parse_transition_state_file", _fake_parse)
    monkeypatch.setattr(
        energy_barriers_controller_module, "parse_transition_state_file", _fake_parse
    )

    window._on_local_job_finished(
        job.id, main_window_module.CastepResult(completed=True, total_energy=-1.0)
    )

    stored = next(iter(doc.results.values()))
    assert stored.result_type == NUDGED_ELASTIC_BAND_CALC_TYPE
    assert stored.data["barrier_height_eV"] == 0.19999999999999996
    assert stored.data["num_images"] == 3

    window._ensure_energy_barriers_view()
    window._refresh_energy_barriers_view()
    assert window._energy_barriers_view._neb_results_table.rowCount() == 1

    window._energy_barriers_controller.show_neb_preview(str(stored.id))

    assert window._energy_barriers_view._profile_plot.point_count() == 3
    assert window._energy_barriers_view._preview_position_label.text() == "Image 1 / 3"
    results_table = window._energy_barriers_view._neb_results_table
    assert results_table.item(0, 3).text() == "0.2 eV"
    assert results_table.item(0, 4).text() == "0.1 eV"
    assert results_table.item(0, 5).text() == "0.1 eV"
    window._energy_barriers_view._preview_next_button.click()
    assert "TST" in window._energy_barriers_view._preview_summary.text()

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
    assert "Progress" in headers
    assert window._jobs_table.columnCount() == 16
    assert window._jobs_table.property("squareTopCorners") == "true"
    header = window._jobs_table.horizontalHeader()
    assert header.sectionResizeMode(window._JOB_LABEL_COLUMN) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(window._JOB_REMOTE_ID_COLUMN) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(window._JOB_RUNTIME_COLUMN) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(window._JOB_PROCESSES_COLUMN) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(window._JOB_PROGRESS_COLUMN) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(window._JOB_SUBMIT_COLUMN) == QHeaderView.ResizeMode.Fixed
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
    assert panel.minimumWidth() == 190
    assert window._project_tree.isHeaderHidden() is True

    window.close()


def test_main_window_project_tree_header_shows_project_name_without_root_item(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()

    doc = ProjectService().create_project(tmp_path / "ui_flat_tree_doc.mufinder", "CoF2")
    window._document = doc
    window._refresh_ui_after_project_change()

    title = window.findChild(QLabel, "projectTreePanelTitle")
    assert title is not None
    assert title.text() == "Project: CoF2"
    assert window._project_tree.topLevelItem(0).text(0) == "Host Structure"
    assert find_tree_item_by_route(window._project_tree, "host_structure") is not None
    assert window._main_stack.currentWidget() is window._home_page
    assert window._home_content_stack.currentWidget() is window._project_home_page
    assert window._project_home_button.isEnabled() is True

    window.close()


def test_project_home_shows_the_workflow_chart_and_opens_the_journal(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    doc = ProjectService().create_project(tmp_path / "project_home_chart_doc.mufinder", "Chart")
    window._document = doc
    window._refresh_ui_after_project_change()
    window._show_project_home()
    window._project_home.refresh()

    summary = window._project_home.chart.summary()
    assert summary is not None
    assert [stage.key for stage in summary.stages] == [
        "host_structure",
        "convergence_cutoff",
        "convergence_kpoint",
        "pristine_relaxation",
        "uep",
        "muon_candidates",
        "muon_relaxations",
        "muon_analysis",
        "hyperfine_coupling",
        "magnetic_fields",
        "polarisation",
        "quantum_corrections",
        "energy_barriers",
    ]
    assert not hasattr(main_window_module, "ProjectHomeTrailWidget")
    assert window._project_home.chart.selected_stage() == "host_structure"

    pixmap = QPixmap(960, 620)
    window._project_home.resize(960, 620)
    window._project_home.render(pixmap)
    assert not pixmap.isNull()

    window._project_home._journal_button.click()
    assert window._main_stack.currentWidget() is window._journal_view

    window.close()


def test_project_home_field_notes_open_the_stage_and_remember_it(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    doc = ProjectService().create_project(tmp_path / "project_home_notes_doc.mufinder", "Notes")
    window._document = doc
    window._refresh_ui_after_project_change()
    window._show_project_home()
    window._project_home.refresh()

    window._project_home.chart.select_stage("uep")
    assert window._project_home.field_notes.stage().key == "uep"
    window._project_home.field_notes._open_button.click()

    assert window._main_stack.currentWidget() is window._uep_view
    assert doc.ui_state["project_home_stage"] == "uep"

    window._project_home_button.click()
    window._project_home.refresh()
    assert window._project_home.chart.summary().current_stage == "uep"

    window._set_active_route("provenance_log")
    assert doc.ui_state["project_home_stage"] == "uep"

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

    host_item = find_tree_item_by_route(window._project_tree, "host_structure")
    cutoff_item = find_tree_item_by_route(window._project_tree, "convergence_cutoff")
    kpoint_item = find_tree_item_by_route(window._project_tree, "convergence_kpoint")
    pristine_item = find_tree_item_by_route(window._project_tree, "pristine_relaxation")
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

    host_item = find_tree_item_by_route(window._project_tree, "host_structure")
    cutoff_item = find_tree_item_by_route(window._project_tree, "convergence_cutoff")
    kpoint_item = find_tree_item_by_route(window._project_tree, "convergence_kpoint")
    pristine_item = find_tree_item_by_route(window._project_tree, "pristine_relaxation")
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


def test_main_window_refreshes_uep_and_muon_site_completion_from_live_document(
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_uep_muon_progress_doc.mufinder", "UEP Progress")
    window._document = doc
    window._refresh_ui_after_project_change()

    uep_item = find_tree_item_by_route(window._project_tree, "uep")
    muon_item = find_tree_item_by_route(window._project_tree, "muon_candidates")

    assert uep_item is not None
    assert muon_item is not None
    assert uep_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is False
    assert muon_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is False

    doc.jobs[uuid4()] = Job(
        label="uep_PBE",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
        status=JobStatus.COMPLETED,
    )
    doc.sites[uuid4()] = MuonSite(
        label="UEP Cluster 1",
        generation_method="uep_cluster",
        status="staged",
    )

    window._refresh_jobs_table()
    window._refresh_muon_generation_table()

    uep_item = find_tree_item_by_route(window._project_tree, "uep")
    muon_item = find_tree_item_by_route(window._project_tree, "muon_candidates")
    assert uep_item is not None
    assert muon_item is not None
    assert uep_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is True
    assert muon_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is True

    summary = summarize_project(doc)
    assert summary.stage("uep").status == StageStatus.COMPLETE
    assert summary.stage("muon_candidates").status == StageStatus.COMPLETE

    window.close()


def test_main_window_muon_relaxation_workflow_complete_when_table_jobs_done(
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "ui_muon_relax_progress.mufinder", "Muon Relax Progress"
    )

    first_job = service.add_job(
        doc,
        label="Relax A",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type=MUON_RELAXATION_CALC_TYPES["positive_muon"],
    )
    first_job.status = JobStatus.COMPLETED
    second_job = service.add_job(
        doc,
        label="Relax B",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type=MUON_RELAXATION_CALC_TYPES["positive_muon"],
    )
    second_job.status = JobStatus.QUEUED
    extra_job = service.add_job(
        doc,
        label="Historical Relax",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type=MUON_RELAXATION_CALC_TYPES["positive_muon"],
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

    relaxation_item = find_tree_item_by_route(window._project_tree, "muon_relaxations")

    assert relaxation_item is not None
    assert relaxation_item.toolTip(0) == "Complete when: All structural relaxations completed."
    assert relaxation_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is False

    second_job.status = JobStatus.COMPLETED
    window._populate_project_tree()

    relaxation_item = find_tree_item_by_route(window._project_tree, "muon_relaxations")
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


def test_main_window_persists_and_restores_geometry_via_qsettings(qapp) -> None:
    window = MainWindow()
    window.show()
    # Move to a distinctive position, but keep the window's current (already
    # screen-fitting) size, so the post-restore clamp_to_screen() call in
    # __init__ (SC-3/SC-4) is a no-op here and does not itself relocate the
    # window -- this test is about the QSettings save/restore wiring, not
    # about clamp_to_screen, which has its own dedicated coverage in
    # test_screen_utils.py.
    window.move(37, 41)
    qapp.processEvents()

    # closeEvent saves geometry under _MAIN_WINDOW_GEOMETRY_KEY without a
    # document loaded (no save-project prompt), exercising the persistence
    # path added alongside the small-screen sizing fixes.
    window.close()

    settings = window._settings
    saved = settings.value(window._MAIN_WINDOW_GEOMETRY_KEY)
    assert saved is not None

    # _isolate_qsettings redirects QSettings to a shared per-test tmp_path, so
    # a second MainWindow constructed in the same test reads back the same
    # store and restores the geometry saved above.
    reloaded_window = MainWindow()
    try:
        avail = reloaded_window.screen().availableGeometry()
        if (
            window.frameGeometry().width() <= avail.width()
            and window.frameGeometry().height() <= avail.height()
        ):
            assert reloaded_window.pos().x() == window.pos().x()
            assert reloaded_window.pos().y() == window.pos().y()
        else:
            # The window itself does not fit this screen (e.g. a small
            # offscreen-platform test display): clamp_to_screen legitimately
            # repositions it, so just confirm restoreGeometry actually ran by
            # checking the restored frame stays fully on-screen.
            frame = reloaded_window.frameGeometry()
            assert avail.left() <= frame.left()
            assert avail.top() <= frame.top()
    finally:
        reloaded_window.close()


def test_main_window_task_farming_dispatch_covers_neb_jobs(qapp, tmp_path: Path) -> None:
    """``_task_farming_for_job`` must resolve NEB jobs, not just quantum phonon ones.

    Both runners use this to decide whether to read ``<seed>_farm001.*`` instead
    of ``<seed>.*``. If a NEB job reports "unfarmed" here, the runner parses the
    master ``.castep`` stub -- which carries neither the band nor the failure --
    and the job's real outcome is lost.
    """

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "farm_dispatch.mufinder", "FarmDispatch")
    window._document = doc

    host = Structure(
        label="host",
        origin="import",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    doc.structures[host.id] = host

    job = service.add_job(
        doc,
        label="barrier_siteA_to_siteB_neb",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
    )
    definition = service.register_nudged_elastic_band_definition(
        doc,
        input_structure_ids=[host.id],
        input_site_ids=[],
        parameters={
            "method": "NEB",
            "row_settings": {
                "source-key": {
                    "parameter_values": {},
                    "added_advanced": [],
                    "num_farms": 3,
                }
            },
        },
        output_directory=doc.project_root / "files" / "energy_barriers_neb",
        job_ids_by_label={"source-key": str(job.id)},
    )
    assert definition is not None

    assert window._task_farming_for_job(job) == (True, 3)
    assert window._farm_counts_for_jobs([job]) == {"barrier_siteA_to_siteB_neb": 3}
    window.close()


def test_main_window_task_farming_reports_off_for_unfarmed_neb_jobs(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "farm_off.mufinder", "FarmOff")
    window._document = doc

    host = Structure(
        label="host",
        origin="import",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    doc.structures[host.id] = host

    job = service.add_job(
        doc,
        label="barrier_siteA_to_siteB_neb",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
    )
    service.register_nudged_elastic_band_definition(
        doc,
        input_structure_ids=[host.id],
        input_site_ids=[],
        parameters={"method": "NEB"},
        output_directory=doc.project_root / "files" / "energy_barriers_neb",
        job_ids_by_label={"source-key": str(job.id)},
    )

    assert window._task_farming_for_job(job) == (False, None)
    assert window._farm_counts_for_jobs([job]) == {}
    window.close()


def test_main_window_neb_submit_passes_farm_settings_to_local_runner(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    """The NEB row's farm count reaches the runner, which needs it to know that
    the finished job's outputs carry CASTEP's ``_farmNNN`` seedname suffix.

    ``_task_farming_for_job`` is consulted at five separate call sites; this
    pins the local submit path end to end so a missed site cannot regress
    silently into "job completed but no .ts found"."""

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "farm_submit.mufinder", "FarmSubmit")
    window._document = doc

    host = Structure(
        label="host",
        origin="import",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    doc.structures[host.id] = host

    job = service.add_job(
        doc,
        label="barrier_siteA_to_siteB_neb",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
    )
    service.register_nudged_elastic_band_definition(
        doc,
        input_structure_ids=[host.id],
        input_site_ids=[],
        parameters={
            "method": "NEB",
            "row_settings": {
                "source-key": {
                    "parameter_values": {},
                    "added_advanced": [],
                    "num_farms": 4,
                }
            },
        },
        output_directory=doc.project_root / "files" / "energy_barriers_neb",
        job_ids_by_label={"source-key": str(job.id)},
    )
    output_dir = doc.project_root / "files" / "energy_barriers_neb" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)

    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._local_runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs),
    )

    window._job_execution.submit_local_jobs(
        [job],
        num_cores=8,
        max_concurrent=1,
        castep_executable="/bin/castep",
        mpirun_path="/bin/mpirun",
        eligible_statuses={JobStatus.DEFINED},
        task_farming_for_job=window._task_farming_for_job,
        retrieve_flags_for_job=window._cluster_retrieve_flags,
    )

    assert len(submit_calls) == 1
    assert submit_calls[0]["task_farming_enabled"] is True
    assert submit_calls[0]["num_farms"] == 4
    assert submit_calls[0]["retrieve_ts"] is True
    window.close()


def test_main_window_neb_configure_is_read_only_once_a_job_exists(qapp, tmp_path: Path) -> None:
    """A row whose NEB job is FAILED is not "ready", so its configuration --
    including the farm count -- cannot be edited.

    This is what keeps the farm count in the row extras (which decides the
    filenames the runner looks for) in step with the NUM_FARMS already written
    into the job's .param (which decides what CASTEP does). ``materialize``
    rewrites the job directory only while the job is DEFINED, so if a FAILED row
    could be reconfigured the two would diverge and a successful run would be
    reported as a missing .ts."""

    from mufinder.app.workflow_submission import READY_WORKFLOW_ROW_STATUSES

    assert JobStatus.FAILED.value not in READY_WORKFLOW_ROW_STATUSES
    assert JobStatus.CANCELLED.value not in READY_WORKFLOW_ROW_STATUSES
    assert JobStatus.COMPLETED.value not in READY_WORKFLOW_ROW_STATUSES
    assert JobStatus.DEFINED.value in READY_WORKFLOW_ROW_STATUSES


def test_main_window_fallback_collection_reads_farmed_ts(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-carrier fallback path must resolve the farm count too.

    ``_on_local_job_finished`` collects synchronously when no artifacts carrier
    is supplied (reconciliation and retry paths). Collecting at the default of
    one farm there would look for <seed>.ts, which a task-farmed run never
    writes -- failing a job that actually succeeded."""

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "farmed_fallback.mufinder", "FarmedFallback")
    window._document = doc

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="barrier_siteA_to_siteB_farmed_neb",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
    )
    service.register_nudged_elastic_band_definition(
        doc,
        input_structure_ids=[host.id],
        input_site_ids=[],
        parameters={
            "method": "NEB",
            "row_settings": {
                "source-key": {
                    "parameter_values": {},
                    "added_advanced": [],
                    "num_farms": 3,
                }
            },
        },
        output_directory=doc.project_root / "files" / "energy_barriers_neb",
        job_ids_by_label={"source-key": str(job.id)},
    )

    output_dir = window._job_output_directory(job)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exactly what a farmed CASTEP run leaves behind: the band under the farm
    # seedname, and no unfarmed .ts at all.
    (output_dir / f"{output_dir.name}_farm001.ts").write_text("placeholder", encoding="utf-8")
    assert not (output_dir / f"{output_dir.name}.ts").exists()

    def _fake_parse(_path):
        return TransitionStatePath(
            images=(
                TransitionStateImage(
                    label="REA",
                    iteration=1,
                    reaction_coordinate=0.0,
                    energy=-1.0,
                    lattice_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
                    species=("Si", "Si"),
                    cartesian_positions=((0.0, 0.0, 0.0), (2.5, 2.5, 2.5)),
                    forces=((0.0, 0.0, 0.0),) * 2,
                ),
                TransitionStateImage(
                    label="TST",
                    iteration=1,
                    reaction_coordinate=0.5,
                    energy=-0.8,
                    lattice_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
                    species=("Si", "Si"),
                    cartesian_positions=((0.0, 0.0, 0.0), (2.5, 2.5, 2.5)),
                    forces=((0.0, 0.0, 0.0),) * 2,
                ),
            )
        )

    monkeypatch.setattr(job_artifacts_module, "parse_transition_state_file", _fake_parse)

    window._on_local_job_finished(
        job.id, main_window_module.CastepResult(completed=True, total_energy=-1.0)
    )

    assert job.status is not JobStatus.FAILED
    stored = next(iter(doc.results.values()))
    assert stored.result_type == NUDGED_ELASTIC_BAND_CALC_TYPE
    assert stored.data["num_images"] == 2
    # The recorded trajectory path is the farmed file the collector actually read.
    assert stored.data["transition_state_file"].endswith("_farm001.ts")
    window.close()


def test_main_window_missing_farmed_ts_names_the_farmed_file(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing-output failure names the file the collector looked for.

    Reporting <seed>.ts for a farmed job would send the user hunting for a file
    CASTEP never writes."""

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "farmed_missing.mufinder", "FarmedMissing")
    window._document = doc

    host = Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="barrier_siteA_to_siteB_farmed_neb",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
    )
    service.register_nudged_elastic_band_definition(
        doc,
        input_structure_ids=[host.id],
        input_site_ids=[],
        parameters={
            "method": "NEB",
            "row_settings": {
                "source-key": {
                    "parameter_values": {},
                    "added_advanced": [],
                    "num_farms": 2,
                }
            },
        },
        output_directory=doc.project_root / "files" / "energy_barriers_neb",
        job_ids_by_label={"source-key": str(job.id)},
    )
    output_dir = window._job_output_directory(job)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    monkeypatch.setattr(
        window,
        "_on_local_job_failed",
        lambda _job_id, message: failures.append(message),
    )

    window._on_local_job_finished(
        job.id, main_window_module.CastepResult(completed=True, total_energy=-1.0)
    )

    assert len(failures) == 1
    assert "_farm001.ts" in failures[0]
    window.close()


def test_main_window_quantum_sources_show_energy_relative_to_comparable_minimum(
    qapp, tmp_path: Path
) -> None:
    """The dE column offsets each site by the lowest site with the same charge state and cell.

    ``MuonSite.total_energy_eV`` stores the absolute total energy, so two
    positive-muon sites in the same cell must read 0 and their difference,
    while a muonium site and a positive-muon site in a larger cell are each the
    zero of their own group rather than being compared across charge or cell.
    """

    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_relative_energy.mufinder", "Quantum dE")
    window._document = doc
    set_show_every_relaxed_site(doc, QUANTUM_CORRECTIONS_SCOPE, True)
    set_show_every_relaxed_site(doc, ENERGY_BARRIERS_SCOPE, True)
    positive = "castep_positivemuon_relaxation"
    low = add_relaxed_muon_site(
        doc, label="low", charge_calc_type=positive, species=["Si", "H:mu"], total_energy=-100.0
    )
    high = add_relaxed_muon_site(
        doc, label="high", charge_calc_type=positive, species=["Si", "H:mu"], total_energy=-99.75
    )
    muonium = add_relaxed_muon_site(
        doc,
        label="mu0",
        charge_calc_type="castep_muonium_relaxation",
        species=["Si", "H:mu"],
        total_energy=-250.0,
    )
    supercell = add_relaxed_muon_site(
        doc,
        label="big",
        charge_calc_type=positive,
        species=["Si", "Si", "H:mu"],
        total_energy=-300.0,
    )

    relative = quantum_source_relative_energies(window._quantum_corrections_controller.candidates())

    assert relative[str(low.id)] == pytest.approx(0.0)
    assert relative[str(high.id)] == pytest.approx(0.25)
    assert relative[str(muonium.id)] == pytest.approx(0.0)
    assert relative[str(supercell.id)] == pytest.approx(0.0)

    window._refresh_quantum_corrections_view()
    table = window._quantum_corrections_view._sources_table
    assert table.horizontalHeaderItem(2).text() == "dE (eV)"
    shown = {
        table.item(row, 0).text(): table.item(row, 2).text() for row in range(table.rowCount())
    }
    assert shown == {"low": "0.00000", "high": "0.25000", "mu0": "0.00000", "big": "0.00000"}

    endpoint_labels = {
        row.label: row.relative_energy for row in window._energy_barriers_controller.endpoint_rows()
    }
    assert endpoint_labels["high"] == "dE 0.25000 eV"
    assert endpoint_labels["low"] == "dE 0.00000 eV"

    window.close()


def test_main_window_configures_neb_paths_through_the_dialog(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.workflow_config_codecs import workflow_row_configuration

    window = MainWindow()
    doc = ProjectService().create_project(tmp_path / "neb_configure.mufinder", "NEBConfigure")
    window._document = doc
    positive = "castep_positivemuon_relaxation"
    sites = [
        add_relaxed_muon_site(
            doc, label=label, charge_calc_type=positive, species=["Si", "H:mu"], total_energy=-1.0
        )
        for label in ("siteA", "siteB")
    ]
    source: dict[str, object] = {"key": "path-1", "label": "siteA_to_siteB_path1", "path_index": 1}
    for side, site in zip(("reactant", "product"), sites, strict=True):
        run = doc.runs[site.relaxation_run_id]
        source[f"{side}_site_id"] = str(site.id)
        source[f"{side}_run_id"] = str(run.id)
        source[f"{side}_job_id"] = str(run.job_id)
        source[f"{side}_structure_id"] = str(doc.results[run.result_id].derived_structure_ids[0])
    source.update(
        path_start_fractional=[0.25, 0.25, 0.25],
        path_end_fractional=[0.75, 0.25, 0.25],
        path_length_ang=2.5,
    )
    doc.ui_state["energy_barriers_form"] = {"generated_sources": [source]}

    opened: list[dict[str, object]] = []

    class _FakeNebDialog:
        def __init__(self, **kwargs) -> None:
            opened.append(dict(kwargs))

        def setWindowTitle(self, title: str) -> None:  # noqa: N802 - Qt API naming.
            opened[-1]["title"] = title

        def exec(self):
            return QDialog.DialogCode.Accepted

        def deleteLater(self) -> None:  # noqa: N802 - Qt API naming.
            opened[-1]["deleted"] = True

        def collect_values(self) -> dict[str, object]:
            return {"tssearch_max_path_points": 13}

        def advanced_names(self) -> tuple[str, ...]:
            return ()

        def untouched_mixed_names(self) -> set[str]:
            return set()

        def num_farms(self) -> int:
            return 4

    monkeypatch.setattr(main_window_module, "NudgedElasticBandConfigDialog", _FakeNebDialog)

    window._on_configure_energy_barrier_result(["path-1"])

    assert len(opened) == 1
    assert opened[0]["title"] == "Configure NEB Parameters"
    assert opened[0]["intro_text"] == "Path: siteA_to_siteB_path1"
    assert opened[0]["read_only"] is False
    assert opened[0]["values"]["tssearch_max_path_points"] == 11
    definition = window._energy_barriers_controller.active_definition()
    params, _advanced, extras = workflow_row_configuration(definition, "path-1")
    assert params == {"tssearch_max_path_points": 13}
    assert extras == {"num_farms": 4}
    assert opened[0]["deleted"] is True
    window.close()
