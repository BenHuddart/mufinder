"""Tests for the Magnetic Fields dipolar-tensor and nuclear-width dialogs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtWidgets import QGroupBox

from mufinder.app.blocking_call import BlockingCallCancelled
from mufinder.app.magnetic_analysis_dialogs import DipolarTensorDialog, NuclearRelaxationDialog
from mufinder.app.magnetic_field_widget import MagneticFieldWidget
from mufinder.domain.constants import MUON_GAMMA_OVER_2PI_MHZ_T
from mufinder.science.magnetic_fields import (
    CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB,
    MagneticFieldDependencyError,
    MagneticFieldError,
)
from mufinder.science.magnetic_structure import AtomSelector, MagneticStructureModel
from mufinder.science.nuclear_relaxation import NuclearRelaxationError
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    attach_magnetic_field_results,
    store_magnetic_field_results,
)
from mufinder.services.magnetic_field_sites import MagneticMuonSiteOption
from mufinder.services.project_service import ProjectService
from tests.support.magnetic_field_sites import feo_host_structure, uep_cluster_document
from tests.support.magnetic_fields import custom_muon_site, z_moment_component


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (
            MagneticFieldDependencyError("muesr is required"),
            "Result unavailable: muesr is required",
        ),
        (MagneticFieldError("no ions"), "Calculation failed: no ions"),
        (TypeError("bad sample"), "Calculation failed: bad sample"),
    ],
)
def test_dipolar_tensor_dialog_reports_a_failed_calculation(
    qapp, error: Exception, status: str
) -> None:
    def _calculate(_dialog):
        raise error

    dialog = DipolarTensorDialog(
        feo_host_structure(),
        MagneticStructureModel(components=(z_moment_component("one", "One", 0),)),
        custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=False,
        calculate_callback=_calculate,
    )

    dialog._calculate()

    assert dialog._status.text() == status
    assert dialog.results_state() == []
    dialog.close()


def test_nuclear_relaxation_dialog_reports_a_failed_calculation(qapp) -> None:
    def _calculate(_dialog):
        raise NuclearRelaxationError("No isotope data available for element 'Xx'")

    dialog = NuclearRelaxationDialog(
        muon_site=custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
        calculate_callback=_calculate,
    )

    dialog._calculate()

    assert dialog._status.text() == (
        "Calculation failed: No isotope data available for element 'Xx'"
    )
    assert dialog.results_state() == {}
    dialog.close()


def _site_from(source: str) -> MagneticMuonSiteOption:
    return replace(custom_muon_site((0.25, 0.25, 0.25)), source=source, label=f"{source} site")


def _tensor_dialog(site, results_state=()) -> DipolarTensorDialog:
    tensor = {
        "site_label": "site 1",
        "muon_position": [0.25, 0.25, 0.25],
        "dipolar_tensor_T_per_muB": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
    }
    return DipolarTensorDialog(
        feo_host_structure(),
        MagneticStructureModel(components=(z_moment_component("one", "One", 0),)),
        site,
        include_equivalent_sites=False,
        results_state=results_state,
        calculate_callback=lambda _dialog: (tensor,),
    )


def _nuclear_dialog(site, results_state=None) -> NuclearRelaxationDialog:
    return NuclearRelaxationDialog(
        muon_site=site,
        distortion_available=False,
        results_state=results_state,
        calculate_callback=lambda _dialog: {
            "site_label": "site 1",
            "mode": "zf_polycrystal",
            "sigma": 0.2,
        },
    )


@pytest.mark.parametrize(
    ("source", "estimate"),
    [("uep_clustering", True), ("relaxation_clustering", False), ("custom", False)],
)
def test_tensor_and_nuclear_dialogs_mark_results_at_a_uep_site_as_estimates(
    qapp, source, estimate
) -> None:
    tensor_dialog = _tensor_dialog(_site_from(source))
    tensor_dialog._calculate()

    assert tensor_dialog.results_state()[0]["site_source"] == source
    assert (not tensor_dialog._estimate_note.isHidden()) is estimate
    title = tensor_dialog._result_content.findChildren(QGroupBox)[0].title()
    assert title == ("site 1 (unrelaxed estimate)" if estimate else "site 1")
    assert ("unrelaxed UEP site" in tensor_dialog._status.text()) is estimate
    tensor_dialog.close()

    nuclear_dialog = _nuclear_dialog(_site_from(source))
    nuclear_dialog._calculate()

    assert nuclear_dialog.results_state()["site_source"] == source
    assert (not nuclear_dialog._estimate_note.isHidden()) is estimate
    assert ("unrelaxed UEP site" in nuclear_dialog._status.text()) is estimate
    nuclear_dialog.close()


def test_tensor_and_nuclear_estimates_survive_a_reload_whatever_site_is_selected(
    qapp, tmp_path
) -> None:
    service = ProjectService()
    document = uep_cluster_document(tmp_path)
    widget = MagneticFieldWidget()
    widget.set_document(document)
    site = widget.selected_muon_site()
    assert site.source == "uep_clustering"
    tensor_dialog = _tensor_dialog(site)
    tensor_dialog._calculate()
    nuclear_dialog = _nuclear_dialog(site)
    nuclear_dialog._calculate()
    widget._dipolar_tensor_results_state = tensor_dialog.results_state()
    widget._nuclear_relaxation_results_state = nuclear_dialog.results_state()
    document.ui_state["magnetic_field_form"] = store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, widget.collect_state()
    )
    service.save_project(document)
    for closed in (tensor_dialog, nuclear_dialog, widget):
        closed.close()

    reloaded = service.load_project(document.project_root)
    restored = MagneticFieldWidget()
    restored.set_document(reloaded)
    restored.set_form_state(
        attach_magnetic_field_results(
            reloaded, MAGNETIC_FIELD_PAGE_OWNER, reloaded.ui_state["magnetic_field_form"]
        )
    )
    # The results keep their own source after another site is chosen.
    other_site = _site_from("custom")
    restored_tensor = _tensor_dialog(other_site, restored._dipolar_tensor_results_state)
    restored_nuclear = _nuclear_dialog(other_site, restored._nuclear_relaxation_results_state)
    assert not restored_tensor._estimate_note.isHidden()
    assert not restored_nuclear._estimate_note.isHidden()
    for closed in (restored_tensor, restored_nuclear, restored):
        closed.close()


def test_tensor_and_nuclear_results_without_a_source_show_no_estimate(qapp) -> None:
    uep_site = _site_from("uep_clustering")
    tensor_dialog = _tensor_dialog(
        uep_site,
        [
            {
                "site_label": "site 1",
                "dipolar_tensor_T_per_muB": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
            }
        ],
    )
    nuclear_dialog = _nuclear_dialog(uep_site, {"mode": "zf_polycrystal", "sigma": 0.2})

    assert tensor_dialog._estimate_note.isHidden()
    assert nuclear_dialog._estimate_note.isHidden()
    tensor_dialog.close()
    nuclear_dialog.close()


def test_nuclear_relaxation_dialog_controls_follow_mode_and_distortion_state(qapp) -> None:
    dialog = NuclearRelaxationDialog(
        muon_site=custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
    )

    assert not dialog._distortion.isEnabled()
    assert dialog._quadrupolar.isEnabled()
    assert dialog._direction_x.objectName() == "nuclearDirectionInput"
    # The app stylesheet styles these inputs (see test_theme), so they follow theme changes.
    assert dialog.styleSheet() == ""
    dialog._regime_combo.setCurrentIndex(dialog._regime_combo.findData("tf"))

    assert not dialog._quadrupolar.isEnabled()
    assert not dialog._quadrupolar.isChecked()
    assert dialog.options().mode == "tf_polycrystal"
    dialog.close()


def test_nuclear_relaxation_dialog_filters_elements(qapp) -> None:
    calls: list[object] = []
    dialog = NuclearRelaxationDialog(
        muon_site=custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
        elements=("Fe", "O"),
        calculate_callback=lambda d: calls.append(d.options()) or {"mode": "zf_polycrystal"},
    )
    checks = dialog._element_checks

    assert list(checks) == ["Fe", "O"]
    assert all(check.isChecked() for check in checks.values())
    assert dialog.options().elements is None

    checks["Fe"].setChecked(False)
    assert dialog.options().elements == ("O",)
    assert dialog.options_state()["elements"] == ["O"]

    checks["O"].setChecked(False)
    dialog._calculate_button.click()
    assert not calls
    assert dialog._status.text() == "Tick at least one element to include in the sum."

    checks["O"].setChecked(True)
    dialog._calculate_button.click()
    assert calls[-1].elements == ("O",)
    dialog.close()


def test_nuclear_relaxation_dialog_records_the_element_filter_with_the_result(qapp) -> None:
    dialog = NuclearRelaxationDialog(
        muon_site=custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
        elements=("Fe", "O"),
        calculate_callback=lambda _d: {"mode": "zf_polycrystal", "sigma": 0.2},
    )

    def _elements_row() -> str:
        table = dialog._summary_table
        row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == "Elements")
        return table.item(row, 1).text()

    dialog._calculate_button.click()
    assert "elements" not in dialog.results_state()
    assert _elements_row() == "All"

    dialog._element_checks["O"].setChecked(False)
    dialog._calculate_button.click()
    assert dialog.results_state()["elements"] == ["Fe"]
    assert _elements_row() == "Fe"
    dialog.close()


def test_nuclear_relaxation_dialog_without_elements_keeps_a_saved_filter(qapp) -> None:
    dialog = NuclearRelaxationDialog(
        muon_site=custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
        options_state={"mode": "zf_polycrystal", "elements": ["Fe"]},
    )

    assert dialog.options().elements == ("Fe",)
    assert dialog.options_state()["elements"] == ["Fe"]
    dialog.close()


def test_nuclear_relaxation_dialog_restores_the_element_filter(qapp) -> None:
    restored = NuclearRelaxationDialog(
        muon_site=custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
        elements=("Fe", "O"),
        options_state={"mode": "zf_polycrystal", "elements": ["Fe"]},
    )
    every = NuclearRelaxationDialog(
        muon_site=custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
        elements=("Fe", "O"),
        options_state={"mode": "zf_polycrystal", "elements": None},
    )

    assert restored._element_checks["Fe"].isChecked()
    assert not restored._element_checks["O"].isChecked()
    assert restored.options().elements == ("Fe",)
    assert every.options().elements is None
    restored.close()
    every.close()


def test_dipolar_tensor_dialog_defaults_to_assigned_magnetic_atoms(qapp) -> None:
    dialog = DipolarTensorDialog(
        feo_host_structure(),
        MagneticStructureModel(components=(z_moment_component("one", "One", 0),)),
        custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=True,
    )

    assert dialog._assigned_radio.isChecked()
    assert dialog.selected_ion_indices() == (0,)
    assert dialog.selector_state()["mode"] == "assigned"
    assert dialog.selector_state()["include_equivalent_sites"] is True
    assert dialog._calculate_button.isEnabled()
    dialog.close()


def test_dipolar_tensor_dialog_requires_explicit_ions_without_assigned_atoms(qapp) -> None:
    dialog = DipolarTensorDialog(
        feo_host_structure(),
        MagneticStructureModel(),
        custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=False,
    )

    assert not dialog._assigned_radio.isEnabled()
    assert not dialog._calculate_button.isEnabled()

    dialog._species_radio.setChecked(True)

    assert dialog.selected_ion_indices() == (0,)
    assert dialog.selector_state()["selector"] == AtomSelector.species("Fe").to_json()
    assert dialog._calculate_button.isEnabled()
    dialog.close()


def test_dipolar_tensor_dialog_atom_indices_count_from_one(qapp) -> None:
    dialog = DipolarTensorDialog(
        feo_host_structure(),
        MagneticStructureModel(),
        custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=False,
    )

    dialog._indices_radio.setChecked(True)
    dialog._indices_edit.setText("2")

    # The box counts from 1; the stored selector keeps zero-based indices.
    assert dialog.selected_ion_indices() == (1,)
    assert dialog.selector_state()["selector"] == AtomSelector.indices([1]).to_json()
    assert dialog._calculate_button.isEnabled()

    dialog._indices_edit.setText("0")
    assert not dialog._calculate_button.isEnabled()
    # Closing on invalid input keeps the selection the dialog opened with.
    assert dialog.selector_state()["selector"] == AtomSelector.indices([]).to_json()
    dialog.close()

    restored = DipolarTensorDialog(
        feo_host_structure(),
        MagneticStructureModel(),
        custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=False,
        selector_state={"mode": "selector", "selector": AtomSelector.indices([0]).to_json()},
    )
    assert restored._indices_radio.isChecked()
    assert restored._indices_edit.text() == "1"
    assert restored.selected_ion_indices() == (0,)
    restored._indices_edit.setText("x")
    assert restored.selector_state()["selector"] == AtomSelector.indices([0]).to_json()
    restored.close()


def test_dipolar_tensor_dialog_shows_and_updates_recent_results(qapp) -> None:
    class _Result:
        site_label = "site 2"
        muon_position = (0.5, 0.5, 0.5)
        tensor_T_per_muB = ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))

        def to_json(self):
            return {
                "site_label": self.site_label,
                "muon_position": list(self.muon_position),
                "dipolar_tensor_T_per_muB": np.asarray(self.tensor_T_per_muB, dtype=float).tolist(),
            }

    calls: list[tuple[int, ...]] = []

    def _calculate(dialog):
        calls.append(dialog.selected_ion_indices())
        return (_Result(),)

    dialog = DipolarTensorDialog(
        feo_host_structure(),
        MagneticStructureModel(components=(z_moment_component("one", "One", 0),)),
        custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=False,
        supercell=(12, 24, 36),
        results_state=[
            {
                "site_label": "site 1",
                "muon_position": [0.25, 0.25, 0.25],
                "dipolar_tensor_T_per_muB": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }
        ],
        calculate_callback=_calculate,
    )

    assert "12 x 24 x 36" in dialog._settings_summary.text()
    assert dialog._result_tables[0].item(0, 0).text() == "1"

    dialog._calculate()

    assert calls == [(0,)]
    assert dialog._result_tables[0].item(0, 0).text() == "2"
    assert dialog.results_state()[0]["site_label"] == "site 2"
    dialog.close()


def _dipolar_dialog_with_result(tensor) -> DipolarTensorDialog:
    return DipolarTensorDialog(
        feo_host_structure(),
        MagneticStructureModel(components=(z_moment_component("one", "One", 0),)),
        custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=False,
        results_state=[
            {
                "site_label": "site 1",
                "muon_position": [0.25, 0.25, 0.25],
                "dipolar_tensor_T_per_muB": tensor,
            }
        ],
    )


def test_dipolar_tensor_dialog_converts_results_to_megahertz(qapp) -> None:
    dialog = _dipolar_dialog_with_result([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])
    combo = dialog._result_unit_combo

    assert dialog._result_tables[0].item(1, 1).text() == "2"
    assert [combo.itemData(index) for index in range(combo.count())] == [
        "angstrom^-3",
        "T_per_muB",
        "MHz_per_muB",
    ]
    assert combo.currentData() == "T_per_muB"
    combo.setCurrentIndex(combo.findData("MHz_per_muB"))

    assert float(dialog._result_tables[0].item(1, 1).text()) == pytest.approx(
        2.0 * MUON_GAMMA_OVER_2PI_MHZ_T
    )
    dialog.close()


def test_dipolar_tensor_dialog_converts_results_to_angstrom_units(qapp) -> None:
    dialog = _dipolar_dialog_with_result([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    combo = dialog._result_unit_combo
    combo.setCurrentIndex(combo.findData("angstrom^-3"))

    assert float(dialog._result_tables[0].item(0, 0).text()) == pytest.approx(
        1.0 / CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB
    )
    dialog.close()


def test_dipolar_tensor_dialog_shows_zeros_for_a_malformed_stored_tensor(qapp) -> None:
    dialog = _dipolar_dialog_with_result([[1.0, 2.0]])

    assert [dialog._result_tables[0].item(0, column).text() for column in range(3)] == ["0"] * 3
    dialog.close()


def _cancelled(_dialog):
    raise BlockingCallCancelled("Calculating…")


def test_cancelling_a_tensor_calculation_keeps_the_shown_tensors(qapp) -> None:
    dialog = _tensor_dialog(custom_muon_site((0.25, 0.25, 0.25)))
    dialog._calculate()
    shown = dialog.results_state()
    dialog._calculate_callback = _cancelled

    dialog._calculate()

    assert dialog._status.text() == "Calculation cancelled; the earlier results are kept"
    assert dialog.results_state() == shown
    dialog.close()


def test_cancelling_a_nuclear_calculation_keeps_the_shown_widths(qapp) -> None:
    dialog = _nuclear_dialog(custom_muon_site((0.25, 0.25, 0.25)))
    dialog._calculate()
    shown = dialog.results_state()
    dialog._calculate_callback = _cancelled

    dialog._calculate()

    assert dialog._status.text() == "Calculation cancelled; the earlier results are kept"
    assert dialog.results_state() == shown
    dialog.close()


def test_cancelling_a_first_calculation_says_only_that_it_was_cancelled(qapp) -> None:
    dialog = NuclearRelaxationDialog(
        muon_site=custom_muon_site((0.25, 0.25, 0.25)),
        distortion_available=False,
        calculate_callback=_cancelled,
    )

    dialog._calculate()

    assert dialog._status.text() == "Calculation cancelled"
    assert dialog.results_state() == {}
    dialog.close()
