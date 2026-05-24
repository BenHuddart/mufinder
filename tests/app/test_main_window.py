"""Smoke tests for main window wiring and route changes."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDockWidget, QFrame, QHeaderView, QLabel, QMessageBox, QTabWidget, QToolButton, QTreeWidgetItem

import mufinder.app.magnetic_field_widget as magnetic_field_widget_module
import mufinder.app.main_window as main_window_module
from mufinder.app import theme as theme_module
import mufinder.castep_io.schema as schema_module
from mufinder.app.main_window import MainWindow
from mufinder.app.workflow_dialogs import QuantumCorrectionsConfigDialog
from mufinder.castep_io.types import ValueWithUnit
from mufinder.castep_io.ts_parser import TransitionStateImage, TransitionStatePath
from mufinder.app.param_dialog import ParameterDialog
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, ComparisonTable, Job, MuonSite, Result, Structure
from mufinder.workflow.energy_barriers import NUDGED_ELASTIC_BAND_CALC_TYPE, TRANSITION_STATE_SEARCH_CALC_TYPE
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_SINGLEPOINT_CALC_TYPE
from mufinder.science.magnetic_structure import AtomSelector, MagneticStructureModel, build_collinear_component
from mufinder.services.project_service import ProjectService


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
    assert view._field_options().phase_sample_count == 8192
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
    assert reloaded_view._field_options().phase_sample_count == 8192
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
    assert workspace_widget._field_options().phase_sample_count == 8192

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


def test_main_window_routes_to_quantum_view_and_construction_placeholders(qapp, tmp_path: Path) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_placeholders_doc.mufinder", "UI Placeholders")
    window._document = doc
    window._refresh_ui_after_project_change()

    quantum_item = _find_tree_item_by_route(window, "quantum_corrections")
    assert quantum_item is not None
    window._project_tree.setCurrentItem(quantum_item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._quantum_corrections_view
    assert window._quantum_corrections_view.accessibleName() == "Quantum Corrections workflow"

    barriers_item = _find_tree_item_by_route(window, "energy_barriers")
    assert barriers_item is not None
    window._project_tree.setCurrentItem(barriers_item)
    window._on_tree_selection_changed()

    assert window._main_stack.currentWidget() is window._energy_barriers_view
    assert window._energy_barriers_view.accessibleName() == "Energy Barriers workflow"
    tabs = window._energy_barriers_view.findChild(QTabWidget, "energyBarriersTabs")
    assert tabs is not None
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Paths",
        "Transition States",
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


def test_main_window_quantum_completed_results_show_phonon_eigenvalues(qapp, tmp_path: Path) -> None:
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
            "phonon_mode_eigenvalues_cm": [100.0, 400.0, 900.0, 1600.0, 2500.0, 3600.0],
        },
    )
    run.result_id = result.id
    doc.jobs[job.id] = job
    doc.runs[run.id] = run
    doc.results[result.id] = result

    window._refresh_quantum_corrections_view()

    assert window._quantum_corrections_view._results_table.item(0, 0).text() == "quantum_phonon_siteA"
    assert window._quantum_corrections_view._results_table.item(0, 2).text() == (
        "1:100.0, 2:400.0, 3:900.0, 4:1600.0, 5:2500.0, 6:3600.0"
    )
    assert window._quantum_corrections_view._results_table.item(0, 3).text() == "-0.25"

    window.close()


def test_main_window_close_tolerates_missing_relaxation_analysis_view(qapp) -> None:
    window = MainWindow()

    delattr(window, "_relaxation_analysis_view")

    window.close()

    assert window._shutting_down is True


def test_main_window_materializes_quantum_singlepoint_jobs_from_relaxed_sources(qapp, tmp_path: Path) -> None:
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
    run = CalculationRun(job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, relative_energy_eV=-1.0)
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

    jobs = window._materialize_quantum_corrections_jobs([str(site.id)])

    assert len(jobs) == 1
    assert jobs[0].calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE
    assert jobs[0].label == "quantum_singlepoint_siteA"
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    assert output_dir.name == "quantum_siteA"
    cell_text = (output_dir / "quantum_siteA.cell").read_text(encoding="utf-8")
    param_text = (output_dir / "quantum_siteA.param").read_text(encoding="utf-8")
    assert "PHONON_KPOINT_MP_GRID 1 1 1" not in cell_text
    assert "KPOINT_MP_GRID 4 1 1" in cell_text
    assert "KPOINT_MP_GRID 2 2 1" not in cell_text
    assert "TASK : SINGLEPOINT" in param_text
    assert "CUT_OFF_ENERGY : 700 eV" in param_text
    assert "CUT_OFF_ENERGY : 620" not in param_text
    assert "ELEC_ENERGY_TOL : 1e-09 eV" in param_text
    assert "PHONON_METHOD" not in param_text


def test_main_window_materializes_quantum_phonon_continuation_after_singlepoint(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_stage2_doc.mufinder", "QuantumStage2")
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
    run = CalculationRun(job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, relative_energy_eV=-1.0)
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

    definition = window._ensure_quantum_corrections_definition([str(site.id)])
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

    singlepoint_jobs = window._materialize_quantum_corrections_jobs([str(site.id)])
    assert len(singlepoint_jobs) == 1
    singlepoint_job = singlepoint_jobs[0]
    assert singlepoint_job.calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE
    singlepoint_job.status = JobStatus.COMPLETED
    window._refresh_quantum_corrections_view()
    assert window._quantum_corrections_view._sources_table.item(0, 3).text() == "6"
    assert window._quantum_corrections_view._sources_table.item(0, 5).text() == "phonon"
    assert window._quantum_corrections_view._sources_table.item(0, 6).text() == "4"
    assert window._quantum_corrections_view._sources_table.item(0, 7).text() == "ready"

    phonon_jobs = window._materialize_quantum_corrections_jobs([str(site.id)])

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
    assert "PHONON_METHOD : DFPT" in param_text
    assert "CONTINUATION : DEFAULT" in param_text
    assert "NUM_FARMS : 4" in param_text
    assert "TASK : PHONON" in param_text


def test_main_window_applies_quantum_row_configuration_to_materialized_job(qapp, tmp_path: Path) -> None:
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
    run = CalculationRun(job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, relative_energy_eV=-1.0)
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

    definition = window._ensure_quantum_corrections_definition([str(site.id)])
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
    configure_button = window._quantum_corrections_view._sources_table.cellWidget(0, 8)
    assert configure_button.text() == "Configured..."

    jobs = window._materialize_quantum_corrections_jobs([str(site.id)])

    assert len(jobs) == 1
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    cell_text = (output_dir / "quantum_siteA.cell").read_text(encoding="utf-8")
    param_text = (output_dir / "quantum_siteA.param").read_text(encoding="utf-8")
    assert "PHONON_METHOD" not in param_text
    assert "CUT_OFF_ENERGY : 650" in param_text
    assert "CUT_OFF_ENERGY : 710" not in param_text
    assert "KPOINT_MP_GRID 2 2 2" in cell_text
    assert "KPOINT_MP_GRID 4 4 1" not in cell_text
    assert "TASK : SINGLEPOINT" in param_text


def test_main_window_quantum_corrections_hides_fine_cutoff_from_default_rows(qapp) -> None:
    window = MainWindow()

    dialog = QuantumCorrectionsConfigDialog(specs=window._quantum_parameter_specs())

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

    source_dialog = ParameterDialog(
        specs=window._energy_barrier_source_parameter_specs(),
        workflow_stage="energy_barriers",
        locked_values={"task": "TRANSITIONSTATESEARCH", "tssearch_method": "LSTQST"},
    )
    neb_dialog = ParameterDialog(
        specs=window._energy_barrier_neb_parameter_specs(),
        workflow_stage="energy_barriers",
        locked_values={"task": "TRANSITIONSTATESEARCH", "tssearch_method": "NEB"},
    )

    assert "fine_cut_off_energy" not in source_dialog._visible_names()
    assert "fine_cut_off_energy" not in neb_dialog._visible_names()
    assert "fine_cut_off_energy" in {spec.name for spec in source_dialog._available_advanced_specs()}
    assert "fine_cut_off_energy" in {spec.name for spec in neb_dialog._available_advanced_specs()}
    assert source_dialog._visible_names()[:4] == [
        "basis_precision",
        "cut_off_energy",
        "elec_energy_tol",
        "kpoint_mp_grid",
    ]
    assert neb_dialog._visible_names()[:4] == [
        "basis_precision",
        "cut_off_energy",
        "elec_energy_tol",
        "kpoint_mp_grid",
    ]

    source_dialog.close()
    neb_dialog.close()
    window.close()


def test_main_window_quantum_corrections_uses_numeric_suffixes_for_duplicate_site_labels(
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_duplicate_labels_doc.mufinder", "Quantum Duplicates")
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
        run = CalculationRun(job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
        result = Result(
            calculation_run_id=run.id,
            result_type=source_job.calc_type,
            data={"total_energy": -1.0 - index},
            derived_structure_ids=[relaxed_structure.id],
        )
        run.result_id = result.id
        doc.runs[run.id] = run
        doc.results[result.id] = result
        site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, relative_energy_eV=-1.0 - index)
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

    labels = [window._quantum_source_label(candidate) for candidate in window._eligible_quantum_source_candidates()]

    assert labels == ["siteA", "siteA_2"]

    jobs = window._materialize_quantum_corrections_jobs(
        [candidate.key for candidate in window._eligible_quantum_source_candidates()]
    )

    assert [job.label for job in jobs] == ["quantum_singlepoint_siteA", "quantum_singlepoint_siteA_2"]

    window.close()


def test_main_window_materialize_quantum_corrections_repairs_stale_suffixed_job_labels(
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_stale_label_doc.mufinder", "Quantum Stale Label")
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
    run = CalculationRun(job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, relative_energy_eV=-1.0)
    doc.sites[site.id] = site

    definition = window._ensure_quantum_corrections_definition([str(site.id)])
    assert definition is not None
    stale_job = service.add_job(
        doc,
        label=f"quantum_singlepoint_siteA_{str(site.id)[:8]}",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_SINGLEPOINT_CALC_TYPE,
    )
    definition.parameters["singlepoint_job_ids_by_label"] = {str(site.id): str(stale_job.id)}

    jobs = window._materialize_quantum_corrections_jobs([str(site.id)])

    assert [job.id for job in jobs] == [stale_job.id]
    assert stale_job.label == "quantum_singlepoint_siteA"
    assert definition.parameters["singlepoint_job_ids_by_label"][str(site.id)] == str(stale_job.id)
    output_dir = window._job_output_directory(stale_job)
    assert output_dir is not None
    assert output_dir.name == "quantum_siteA"

    window.close()


def test_main_window_quantum_corrections_allows_editing_eligible_site_name(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_rename_doc.mufinder", "Quantum Rename")
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
    run = CalculationRun(job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, relative_energy_eV=-1.0)
    doc.sites[site.id] = site

    window._refresh_quantum_corrections_view()
    item = window._quantum_corrections_view._sources_table.item(0, 0)

    assert item is not None
    item.setText("Renamed Site")

    assert doc.sites[site.id].label == "Renamed Site"
    assert window._quantum_corrections_view._sources_table.item(0, 0).text() == "Renamed Site"

    window.close()


def test_main_window_materializes_energy_barrier_jobs_from_relaxed_sources(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_doc.mufinder", "Barriers")
    window._document = doc
    doc.project.recommended_cutoff_energy = 700.0
    doc.project.recommended_kpoint_grid = (3, 1, 1)

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
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value, "atomic_spins": [1.0, -1.0, 0.0]},
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value, "atomic_spins": [1.0, -1.0, 0.0]},
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
    site_a = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run_a.id, relative_energy_eV=-1.0)
    site_b = MuonSite(label="siteB", status="relaxed", relaxation_run_id=run_b.id, relative_energy_eV=-0.9)
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._ensure_energy_barriers_view()
    window._refresh_energy_barriers_view()
    assert window._energy_barriers_view._sources_table.rowCount() == 0

    window._on_generate_energy_barrier_paths(str(site_a.id), str(site_b.id))
    candidates = window._eligible_energy_barrier_source_candidates()

    assert candidates
    jobs = window._materialize_energy_barriers_jobs([candidates[0].key])

    assert len(jobs) == 1
    assert jobs[0].calc_type == TRANSITION_STATE_SEARCH_CALC_TYPE
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    cell_text = (output_dir / f"{output_dir.name}.cell").read_text(encoding="utf-8")
    param_text = (output_dir / f"{output_dir.name}.param").read_text(encoding="utf-8")
    assert "%BLOCK POSITIONS_FRAC_PRODUCT" in cell_text
    assert "KPOINT_MP_GRID 3 1 1" in cell_text
    assert "TASK : TRANSITIONSTATESEARCH" in param_text
    assert "CUT_OFF_ENERGY : 700" in param_text
    assert "TSSEARCH_METHOD : LSTQST" in param_text


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
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value, "atomic_spins": [1.0, -1.0, 0.0]},
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value, "atomic_spins": [1.0, -1.0, 0.0]},
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
    site_a = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run_a.id, relative_energy_eV=-1.0)
    site_b = MuonSite(label="siteB", status="relaxed", relaxation_run_id=run_b.id, relative_energy_eV=-0.9)
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._on_generate_energy_barrier_paths(str(site_a.id), str(site_b.id))
    candidate = window._eligible_energy_barrier_source_candidates()[0]
    definition = window._ensure_energy_barriers_definition([candidate.key])
    assert definition is not None
    definition.parameters["configuration"]["row_settings"] = {
        candidate.key: {
            "parameter_values": {
                "kpoint_mp_grid": [2, 1, 1],
                "tssearch_qst_max_iter": 12,
            }
        }
    }

    window._ensure_energy_barriers_view()
    window._refresh_energy_barriers_view()
    configure_button = window._energy_barriers_view._sources_table.cellWidget(0, 6)
    assert configure_button.text() == "Configured..."

    jobs = window._materialize_energy_barriers_jobs([candidate.key])

    assert len(jobs) == 1
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    cell_text = (output_dir / f"{output_dir.name}.cell").read_text(encoding="utf-8")
    param_text = (output_dir / f"{output_dir.name}.param").read_text(encoding="utf-8")
    assert "KPOINT_MP_GRID 2 1 1" in cell_text
    assert "TSSEARCH_QST_MAX_ITER : 12" in param_text

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


def test_main_window_applies_energy_barrier_batch_configuration(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_batch_config_doc.mufinder", "BarrierBatchConfig")
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
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value, "atomic_spins": [1.0, -1.0, 0.0]},
    )
    relaxed_b = Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value, "atomic_spins": [1.0, -1.0, 0.0]},
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
    site_a = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run_a.id, relative_energy_eV=-1.0)
    site_b = MuonSite(label="siteB", status="relaxed", relaxation_run_id=run_b.id, relative_energy_eV=-0.9)
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._on_generate_energy_barrier_paths(str(site_a.id), str(site_b.id))
    candidates = window._eligible_energy_barrier_source_candidates()[:2]
    assert len(candidates) == 2

    monkeypatch.setattr(
        main_window_module.ParameterDialog,
        "exec",
        lambda _self: main_window_module.QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        main_window_module.ParameterDialog,
        "collect_values",
        lambda _self: {"task": "TRANSITIONSTATESEARCH", "tssearch_qst_max_iter": 14},
    )
    monkeypatch.setattr(main_window_module.ParameterDialog, "advanced_names", lambda _self: ())

    window._on_configure_energy_barrier_source([candidate.key for candidate in candidates])

    definition = window._active_energy_barriers_definition()
    assert definition is not None
    row_settings = definition.parameters["configuration"]["row_settings"]
    for candidate in candidates:
        assert row_settings[candidate.key]["parameter_values"]["tssearch_qst_max_iter"] == 14
    window.close()


def test_main_window_energy_barrier_paths_persist_when_endpoint_selection_changes(qapp, tmp_path: Path) -> None:
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
    site_a = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run_a.id, relative_energy_eV=-1.0)
    site_b = MuonSite(label="siteB", status="relaxed", relaxation_run_id=run_b.id, relative_energy_eV=-0.9)
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._on_generate_energy_barrier_paths(str(site_a.id), str(site_b.id), "")
    generated_before = window._eligible_energy_barrier_source_candidates()

    assert generated_before

    window._on_energy_barrier_endpoint_selection_changed(str(site_b.id), str(site_a.id))
    generated_after = window._eligible_energy_barrier_source_candidates()

    assert [candidate.key for candidate in generated_after] == [candidate.key for candidate in generated_before]
    window.close()


def test_main_window_energy_barrier_defaults_apply_equivalent_density_kpoints_for_supercell(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_supercell_kpoints_doc.mufinder", "BarrierSupercellKPoints")
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
    site_a = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run_a.id, relative_energy_eV=-1.0)
    site_b = MuonSite(label="siteB", status="relaxed", relaxation_run_id=run_b.id, relative_energy_eV=-0.9)
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._on_generate_energy_barrier_paths(str(site_a.id), str(site_b.id))
    candidate = window._eligible_energy_barrier_source_candidates()[0]

    jobs = window._materialize_energy_barriers_jobs([candidate.key])

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
    doc = service.create_project(tmp_path / "barrier_existing_state.mufinder", "BarrierExistingState")
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
    site_a = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run_a.id, relative_energy_eV=-1.0)
    site_b = MuonSite(label="siteB", status="relaxed", relaxation_run_id=run_b.id, relative_energy_eV=-0.9)
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    # Generate paths first so state has generated_sources and max_path_length_ang.
    window._on_generate_energy_barrier_paths(str(site_a.id), str(site_b.id), "4.0")
    assert doc.ui_state.get("energy_barriers_form", {}).get("generated_sources")

    # Simulate re-opening the project: create a fresh window that only has the document.
    window2 = MainWindow()
    window2._document = doc

    # Navigate to energy barriers — this is what happens when the user opens the section.
    view = window2._ensure_energy_barriers_view()
    window2._refresh_energy_barriers_view()  # Must not raise

    # The view should show the generated paths.
    candidates = window2._eligible_energy_barrier_source_candidates()
    assert len(candidates) > 0

    # Configure dialog values must be gettable without error.
    definition = window2._active_energy_barriers_definition()
    for candidate in candidates:
        values, _advanced, _extras = window2._energy_barrier_row_configuration_with_source_defaults(
            definition,
            candidate.key,
            target_lattice=candidate.reactant.structure.lattice_vectors,
        )
        assert isinstance(values, dict)

    window.close()
    window2.close()


def test_main_window_energy_barrier_strips_legacy_tss_params_from_stored_config(qapp, tmp_path: Path) -> None:
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
    site_a = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run_a.id, relative_energy_eV=-1.0)
    site_b = MuonSite(label="siteB", status="relaxed", relaxation_run_id=run_b.id, relative_energy_eV=-0.9)
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._on_generate_energy_barrier_paths(str(site_a.id), str(site_b.id))
    candidate = window._eligible_energy_barrier_source_candidates()[0]

    # Simulate legacy stored config as it would have been written by the old TSS dialog:
    # tssearch_force/disp/energy_tol stored as 0 with cross-reference unit strings,
    # and tssearch_lst_max_iter present as an obsolete keyword.
    definition = window._ensure_energy_barriers_definition([candidate.key])
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

    jobs = window._materialize_energy_barriers_jobs([candidate.key])

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


def test_main_window_energy_barrier_defaults_read_kpoints_from_reactant_job_cell_file(qapp, tmp_path: Path) -> None:
    """Equivalent k-point density is taken from the reactant relaxation job's output cell file."""
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_cell_kpoints_doc.mufinder", "BarrierCellKPoints")
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
        [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5],
        [0.125, 0.125, 0.125],
    ]
    supercell_positions_b = [
        [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5],
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
    site_a = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run_a.id, relative_energy_eV=-1.0)
    site_b = MuonSite(label="siteB", status="relaxed", relaxation_run_id=run_b.id, relative_energy_eV=-0.9)
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    # Write the reactant job's output cell file with the supercell k-point grid.
    source_dir_a = doc.project_root / "files" / WorkflowStage.MUON_RELAXATION.value / job_a.label
    source_dir_a.mkdir(parents=True, exist_ok=True)
    (source_dir_a / f"{job_a.label}.cell").write_text(
        "\n".join([
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
        ]),
        encoding="utf-8",
    )

    window._on_generate_energy_barrier_paths(str(site_a.id), str(site_b.id))
    candidate = window._eligible_energy_barrier_source_candidates()[0]
    jobs = window._materialize_energy_barriers_jobs([candidate.key])

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
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, relative_energy_eV=-1.0)
    doc.sites[site.id] = site

    window._on_generate_energy_barrier_paths(str(site.id), str(site.id), "")
    generated = window._eligible_energy_barrier_source_candidates()

    assert generated
    assert all(candidate.path.length_ang > 0.0 for candidate in generated)
    assert all(candidate.reactant.site.id == site.id for candidate in generated)
    assert all(candidate.product.site.id == site.id for candidate in generated)
    window.close()


def test_main_window_generates_energy_barrier_paths_with_user_max_length(qapp, tmp_path: Path) -> None:
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
    site_a = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run_a.id, relative_energy_eV=-1.0)
    site_b = MuonSite(label="siteB", status="relaxed", relaxation_run_id=run_b.id, relative_energy_eV=-0.9)
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._on_generate_energy_barrier_paths(str(site_a.id), str(site_b.id), "1.9")

    assert window._eligible_energy_barrier_source_candidates() == []
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
    site_a = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run_a.id, relative_energy_eV=-1.0)
    site_b = MuonSite(label="siteB", status="relaxed", relaxation_run_id=run_b.id, relative_energy_eV=-0.9)
    doc.sites[site_a.id] = site_a
    doc.sites[site_b.id] = site_b

    window._on_generate_energy_barrier_paths(str(site_a.id), str(site_b.id), "")
    candidate = window._eligible_energy_barrier_source_candidates()[0]
    definition = window._ensure_energy_barriers_definition([candidate.key])

    assert definition is not None

    window._on_remove_energy_barrier_sources([candidate.key])

    assert candidate.key not in {
        str(source.get("key", "")).strip()
        for source in window._energy_barrier_generated_sources()
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

    monkeypatch.setattr(
        main_window_module,
        "parse_transition_state_file",
        lambda _path: TransitionStatePath(
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
        ),
    )

    window._on_local_job_finished(job.id, main_window_module.CastepResult(completed=True, total_energy=-1.0))

    stored = next(iter(doc.results.values()))
    assert stored.result_type == TRANSITION_STATE_SEARCH_CALC_TYPE
    assert stored.data["barrier_height_eV"] == 0.19999999999999996
    assert stored.data["num_images"] == 3

    window.close()


def test_main_window_materializes_neb_jobs_from_completed_transition_state_result(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_neb_doc.mufinder", "BarrierNeb")
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

    definition = service.register_transition_state_search_definition(
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
        output_directory=doc.project_root / "files" / "energy_barriers",
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
        main_window_module,
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

    jobs = window._materialize_energy_barrier_neb_jobs(["source-key"])

    assert len(jobs) == 1
    assert jobs[0].calc_type == NUDGED_ELASTIC_BAND_CALC_TYPE
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    cell_text = (output_dir / f"{output_dir.name}.cell").read_text(encoding="utf-8")
    param_text = (output_dir / f"{output_dir.name}.param").read_text(encoding="utf-8")
    assert "%BLOCK POSITIONS_ABS_PRODUCT" in cell_text
    assert "%BLOCK POSITIONS_ABS_INTERMEDIATE" in cell_text
    assert "TSSEARCH_METHOD : NEB" in param_text

    window.close()


def test_main_window_applies_neb_row_configuration_to_materialized_job(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "barrier_neb_row_config_doc.mufinder", "BarrierNebRowConfig")
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

    definition = service.register_transition_state_search_definition(
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
        output_directory=doc.project_root / "files" / "energy_barriers",
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
        main_window_module,
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

    neb_definition = window._ensure_energy_barrier_neb_definition(["source-key"])
    assert neb_definition is not None
    neb_definition.parameters["configuration"]["row_settings"] = {
        "source-key": {
            "parameter_values": {
                "kpoint_mp_grid": [1, 1, 2],
                "tssearch_neb_max_iter": 24,
            }
        }
    }

    window._ensure_energy_barriers_view()
    window._refresh_energy_barriers_view()
    configure_button = window._energy_barriers_view._results_table.cellWidget(0, 5)
    assert configure_button.text() == "Configured..."

    jobs = window._materialize_energy_barrier_neb_jobs(["source-key"])

    assert len(jobs) == 1
    output_dir = window._job_output_directory(jobs[0])
    assert output_dir is not None
    cell_text = (output_dir / f"{output_dir.name}.cell").read_text(encoding="utf-8")
    param_text = (output_dir / f"{output_dir.name}.param").read_text(encoding="utf-8")
    assert "KPOINT_MP_GRID 1 1 2" in cell_text
    assert "CUT_OFF_ENERGY : 680" in param_text
    assert "TSSEARCH_NEB_MAX_ITER : 24" in param_text


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

    monkeypatch.setattr(
        main_window_module,
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
        ),
    )

    window._on_local_job_finished(job.id, main_window_module.CastepResult(completed=True, total_energy=-1.0))

    stored = next(iter(doc.results.values()))
    assert stored.result_type == NUDGED_ELASTIC_BAND_CALC_TYPE
    assert stored.data["barrier_height_eV"] == 0.19999999999999996
    assert stored.data["num_images"] == 3

    window._ensure_energy_barriers_view()
    window._refresh_energy_barriers_view()
    assert window._energy_barriers_view._neb_results_table.rowCount() == 1

    window._on_energy_barrier_neb_result_selected(str(stored.id))

    assert window._energy_barriers_view._profile_plot.point_count() == 3
    assert window._energy_barriers_view._preview_slider.maximum() == 2
    assert window._energy_barriers_view._preview_slider_start_label.text() == "0"
    assert window._energy_barriers_view._preview_slider_end_label.text() == "1"
    window._energy_barriers_view._preview_slider.setValue(1)
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


def test_project_home_trail_scroll_labels_keep_dark_ink_in_dark_mode(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)
    try:
        dark_palette = theme_module.current_palette()
        label_color = main_window_module.ProjectHomeTrailWidget._workflow_scroll_label_color()

        assert label_color.name().lower() == theme_module.light_palette().text_primary.lower()
        assert label_color.name().lower() != dark_palette.text_primary.lower()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


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


def test_main_window_refreshes_uep_and_muon_site_completion_from_live_document(
    qapp,
    tmp_path: Path,
) -> None:
    window = MainWindow()

    service = ProjectService()
    doc = service.create_project(tmp_path / "ui_uep_muon_progress_doc.mufinder", "UEP Progress")
    window._document = doc
    window._refresh_ui_after_project_change()

    uep_item = _find_tree_item_by_route(window, "uep")
    muon_item = _find_tree_item_by_route(window, "muon_candidates")

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

    uep_item = _find_tree_item_by_route(window, "uep")
    muon_item = _find_tree_item_by_route(window, "muon_candidates")
    assert uep_item is not None
    assert muon_item is not None
    assert uep_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is True
    assert muon_item.data(0, window._WORKFLOW_COMPLETE_ROLE) is True

    destinations = {destination.route: destination for destination in window._project_home_destinations()}
    assert destinations["uep"].completed is True
    assert destinations["muon_candidates"].completed is True

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
