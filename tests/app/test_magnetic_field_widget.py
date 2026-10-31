"""Tests for the Magnetic Fields page and its calculation settings dialog."""

from __future__ import annotations

from dataclasses import replace
import threading
import time
from uuid import uuid4

import numpy as np
import pytest
from PySide6.QtCore import QItemSelectionModel, Qt, QTimer
from PySide6.QtGui import QDoubleValidator
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractItemView, QApplication, QDialog, QProgressDialog

import mufinder.app.magnetic_field_widget as magnetic_field_widget_module
import mufinder.services.magnetic_field_calculation as magnetic_field_calculation_module
import mufinder.science.magnetic_fields as magnetic_fields_module
from mufinder.app.magnetic_analysis_dialogs import NuclearRelaxationDialog
from mufinder.app.magnetic_component_dialogs import MagneticComponentDialog
from mufinder.app.magnetic_field_distribution_dialog import MagneticFieldDistributionDialog
from mufinder.app.magnetic_field_widget import (
    MagneticFieldCalculationSettingsDialog,
    MagneticFieldWidget,
)
from mufinder.app.magnetic_site_dialog import MuonSiteSelectionDialog
from mufinder.app.magnetic_texture_dialog import MagneticTextureDialog
from mufinder.domain.models import Structure
from mufinder.science.demagnetization import DemagnetizingSample, PlatePhoto
from mufinder.science.magnetic_fields import MagneticFieldError, contact_coupling_to_angstrom_minus3
from mufinder.science.magnetic_structure import (
    AtomSelector,
    ComplexVector,
    FourierTerm,
    MagneticComponent,
    MagneticStructureModel,
    build_collinear_component,
    magnetic_model_from_host_spins,
)
from mufinder.science.nuclear_relaxation import NuclearRelaxationOptions
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    attach_magnetic_field_results,
    store_magnetic_field_results,
)
from mufinder.services.magnetic_field_settings import (
    MAX_INCOMMENSURATE_PHASE_SAMPLES,
    MagneticFieldCalculationSettings,
)
from mufinder.services.magnetic_field_sites import (
    MagneticMuonSiteOption,
    relaxation_cluster_muon_site_options,
)
from mufinder.services.project_service import ProjectService
from tests.support.codata import BOHR_MAGNETON_J_PER_T, MU0_OVER_4PI_T_M_PER_A
from tests.support.magnetic_field_sites import (
    add_uep_cluster,
    displaced_iron_supercell,
    feo_host_structure,
    relaxation_cluster_document,
    uep_cluster_document,
)
from tests.support.magnetic_fields import (
    coo_structure,
    custom_muon_site,
    photo_plate_sample,
    z_moment_component,
)
from tests.support.structures import host_structure


def test_magnetic_field_widget_reports_and_persists_disabled_components(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-disabled.mufinder", "Magnetic Disabled")
    service.add_structure(document, feo_host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    payloads: list[dict] = []
    widget.magnetic_structure_changed.connect(payloads.append)

    widget.set_model(
        MagneticStructureModel(
            components=(
                z_moment_component("one", "One", 0),
                replace(z_moment_component("two", "Two", 1), enabled=False),
            )
        )
    )
    assert widget._structure_summary.text() == "Magnetic structure: 2 components (1 disabled)"
    assert widget._calculate_button.isEnabled()
    restored = MagneticStructureModel.from_json(payloads[-1])
    assert [component.enabled for component in restored.components] == [True, False]

    summary = widget._component_summary_table
    assert not summary.isHidden()
    assert summary.rowCount() == 1
    assert [
        summary.horizontalHeaderItem(column).text() for column in range(summary.columnCount())
    ] == [
        "Component",
        "Type",
        "Apply to",
        "q",
        "Moment",
    ]
    assert summary.item(0, 0).text() == "One"
    assert summary.item(0, 2).text() == "Selected atoms: 1"

    widget.set_model(
        MagneticStructureModel(
            components=(replace(z_moment_component("one", "One", 0), enabled=False),)
        )
    )
    assert widget._structure_summary.text() == "Magnetic structure: 1 component (1 disabled)"
    assert summary.isHidden()
    assert summary.rowCount() == 0
    assert widget._classification.text() == "Structure type: -"
    assert not widget._calculate_button.isEnabled()
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    widget.calculate_fields()
    assert (
        widget._calculation_status.text()
        == "Results: enable at least one magnetic-structure component first."
    )
    widget.close()


def _accepted(edit) -> tuple[str, float, float, int, bool]:
    """An input's text, accepted range, decimals and whether it takes exponents."""

    validator = edit.validator()
    scientific = validator.notation() == QDoubleValidator.Notation.ScientificNotation
    return (edit.text(), validator.bottom(), validator.top(), validator.decimals(), scientific)


def test_decimal_inputs_accept_the_same_text_as_before(qapp) -> None:
    structure = coo_structure()
    component = MagneticComponentDialog(structure, selected_indices=(0,))
    texture = MagneticTextureDialog(structure)
    nuclear = NuclearRelaxationDialog(muon_site=None, distortion_available=False)
    site = MuonSiteSelectionDialog(None, feo_host_structure())
    page = MagneticFieldWidget()

    assert _accepted(component._amplitude) == ("1.0", 0.0, 1000.0, 8, False)
    assert _accepted(component._cone_angle) == ("45.0", 0.0, 180.0, 8, False)
    assert _accepted(component._q_x) == ("0.0", -4.0, 4.0, 8, False)
    assert _accepted(component._phase) == ("0.0", -10.0, 10.0, 8, False)
    assert _accepted(component._primary_direction[2]) == ("1", -100.0, 100.0, 8, False)
    assert _accepted(texture._q1[0]) == ("0.05", -10.0, 10.0, 8, False)
    assert _accepted(texture._normalise_threshold) == ("0.001", 1e-8, 1.0, 8, False)
    assert _accepted(nuclear._direction_z) == ("1.0", -1e6, 1e6, 12, True)
    assert _accepted(nuclear._radius) == ("", 0.0, 1e6, 12, True)
    assert _accepted(nuclear._rtol) == ("1e-6", 0.0, 1e6, 12, True)
    assert _accepted(site._custom_x) == ("0.0", 0.0, 1.0, 8, True)
    assert _accepted(page._applied_field_input.edits[0]) == ("0.0", -1000.0, 1000.0, 6, True)
    assert _accepted(page._measured_magnetization_input.edits[1]) == (
        "0.0",
        -1000.0,
        1000.0,
        6,
        True,
    )
    assert _accepted(page._hyperfine_coupling) == ("0.0", -1e6, 1e6, 8, True)
    for widget in (component, texture, nuclear, site, page):
        widget.close()


def test_magnetic_field_widget_opens_a_host_only_project_with_no_muon_site(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-host-only.mufinder", "Host Only")
    service.add_structure(document, feo_host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))

    assert widget.selected_muon_site() is None
    assert widget._site_summary.text() == "Choose a muon site."
    assert widget._select_site_button.isEnabled()
    for button in (
        widget._calculate_button,
        widget._dipolar_tensor_button,
        widget._nuclear_relaxation_button,
    ):
        assert not button.isEnabled()
        assert button.toolTip() == "Choose a muon site first."

    widget.set_form_state(
        {
            "selected_muon_site": {
                "source": "custom",
                "label": "Custom site",
                "fractional_position": [0.25, 0.5, 0.75],
            }
        }
    )

    for button in (
        widget._calculate_button,
        widget._dipolar_tensor_button,
        widget._nuclear_relaxation_button,
    ):
        assert button.isEnabled()
        assert button.toolTip() == ""

    widget.set_model(
        MagneticStructureModel(
            components=(replace(z_moment_component("one", "One", 0), enabled=False),)
        )
    )

    assert not widget._calculate_button.isEnabled()
    assert widget._calculate_button.toolTip() == (
        "Enable at least one magnetic-structure component first."
    )
    widget.close()


def test_magnetic_field_widget_restores_a_saved_site_on_an_atom_and_refuses_it(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-on-atom.mufinder", "On Atom")
    service.add_structure(document, feo_host_structure())
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_commensurate_fields",
        lambda *_args, **_kwargs: pytest.fail("a site on an atom must not be calculated"),
    )
    reason = "Custom site is 0.00 Å from Fe (atom 1); a muon site must be at least 0.5 Å from every atom."

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget.set_form_state(
        {
            "selected_muon_site": {
                "source": "custom",
                "label": "Custom site",
                "fractional_position": [0.0, 0.0, 0.0],
            }
        }
    )

    assert widget.selected_muon_site().fractional_position == (0.0, 0.0, 0.0)
    assert widget._site_summary.text().startswith(
        "Muon site: Custom site from Custom at (0.0000, 0.0000, 0.0000)."
    )
    assert widget._site_summary.text().endswith(reason)
    for button in (
        widget._calculate_button,
        widget._dipolar_tensor_button,
        widget._nuclear_relaxation_button,
    ):
        assert not button.isEnabled()
        assert button.toolTip() == reason

    widget.calculate_fields()

    assert widget._calculation_status.text() == f"Results: {reason}"
    assert widget._result_table.rowCount() == 0
    widget.close()


def test_magnetic_field_widget_never_shows_a_non_finite_field(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-nan.mufinder", "NaN Field")
    service.add_structure(document, feo_host_structure())

    class _Result:
        site_label = "Mu1"
        total = (0.1, 0.0, 0.0)
        magnitude = 0.1
        distortion = None

    class _NanResult(_Result):
        total = (float("nan"), 0.0, 0.0)
        magnitude = float("nan")

    returned: list[object] = [_Result()]
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_commensurate_fields",
        lambda *_args, **_kwargs: tuple(returned),
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    widget._sync_enabled_state()
    saved: list[dict] = []
    widget.form_state_changed.connect(saved.append)
    widget.calculate_fields()
    assert saved[-1]["magnetic_field_results"] is not None

    returned[:] = [_NanResult()]
    widget.calculate_fields()

    assert widget._result_table.rowCount() == 0
    assert widget.collect_state()["magnetic_field_results"] is None
    # The cleared results are saved, so the earlier ones do not come back on reopening.
    assert saved[-1]["magnetic_field_results"] is None
    assert widget._calculation_status.text() == (
        "Calculation failed: the field at Mu1 is not a finite number (NaN or infinity), "
        "so it is not shown"
    )
    widget.close()


def test_magnetic_field_widget_shows_an_unclassifiable_model(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-unclassified.mufinder", "Unclassified")
    service.add_structure(document, feo_host_structure())

    def _unclassifiable(*_args, **_kwargs):
        raise MagneticFieldError("magnetic structure has no non-zero Fourier components")

    monkeypatch.setattr(magnetic_field_widget_module, "classify_magnetic_model", _unclassifiable)
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))

    assert widget._classification.text() == "Structure type: unavailable"
    widget.close()


def test_magnetic_field_widget_reports_an_untyped_backend_error(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-backend.mufinder", "Backend Error")
    service.add_structure(document, feo_host_structure())

    def _backend_failure(*_args, **_kwargs):
        raise RuntimeError("Symbolic FC not defined!")

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module, "calculate_commensurate_fields", _backend_failure
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    widget._sync_enabled_state()
    widget.calculate_fields()

    assert widget._calculation_status.text() == "Calculation failed: Symbolic FC not defined!"
    assert widget._result_table.rowCount() == 0
    widget.close()


def test_magnetic_field_widget_never_restores_a_non_finite_field(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-saved-nan.mufinder", "Saved NaN")
    service.add_structure(document, feo_host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(
        {
            "selected_muon_site": {
                "source": "custom",
                "label": "Custom site",
                "fractional_position": [0.25, 0.5, 0.75],
            },
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [
                    {
                        "site_label": "Custom site",
                        "total_T": [float("nan"), 0.0, 0.0],
                        "magnitude_T": float("nan"),
                    }
                ],
            },
        }
    )

    assert widget._result_table.rowCount() == 0
    assert widget._calculation_status.text() == (
        "Results: the saved field at Custom site is not a finite number (NaN or infinity), "
        "so it is not shown."
    )
    assert not widget._polarisation_button.isEnabled()
    # The saved results are kept as they were: nothing is discarded without asking.
    assert widget.collect_state()["magnetic_field_results"]["results"][0]["site_label"] == (
        "Custom site"
    )
    widget.close()


def test_magnetic_field_widget_calculates_commensurate_fields(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-calc.mufinder", "Magnetic Calc")
    service.add_structure(document, feo_host_structure())

    calls: list[dict[str, object]] = []

    class _Result:
        site_label = "site 1"
        total = (0.1, 0.2, 0.3)
        magnitude = float(np.linalg.norm(total))
        distortion = None

    def _calculate(structure, model, muon_sites, *, options, distorted_structures=None):
        calls.append(
            {
                "structure": structure,
                "model": model,
                "muon_sites": muon_sites,
                "options": options,
                "distorted_structures": distorted_structures,
            }
        )
        return (_Result(),)

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module, "calculate_commensurate_fields", _calculate
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    widget._refresh_site_summary()
    widget._none_radio.setChecked(True)

    widget.calculate_fields()

    assert calls
    assert calls[0]["muon_sites"] == [(0.25, 0.5, 0.75)]
    assert calls[0]["distorted_structures"] == [None]
    assert calls[0]["options"].demag_shape == "none"
    assert calls[0]["options"].supercell == (100, 100, 100)
    assert widget._result_table.rowCount() == 1
    assert widget._result_table.item(0, 1).text() == "0.1"
    assert widget._result_table.item(0, 5).text() == "field"
    assert widget.collect_state()["magnetic_field_results"]["kind"] == "commensurate"
    assert widget.collect_state()["magnetic_field_results"]["results"][0][
        "total_T"
    ] == pytest.approx([0.1, 0.2, 0.3])
    assert "commensurate" in widget._calculation_status.text()
    widget.close()


def _fake_muesr_without_lattice_field(monkeypatch) -> None:
    """Run the real field helpers with a muesr stand-in that returns zero lattice fields."""

    class _Sample:
        def __init__(self) -> None:
            self.muons: list[object] = []

        def add_muon(self, position) -> None:
            self.muons.append(position)

        def new_mm(self) -> None:
            self.mm = type("_MM", (), {})()

    def _build_sample(_structure, muons):
        sample = _Sample()
        for muon in muons:
            sample.add_muon(muon)
        return sample

    def _locfield(sample, *_args, **_kwargs):
        return [type("_Field", (), {"Dipolar": np.zeros(3)})() for _ in sample.muons]

    monkeypatch.setattr(magnetic_fields_module, "_build_muesr_sample", _build_sample)
    monkeypatch.setattr(
        magnetic_fields_module, "_load_muesr_engines", lambda: (_locfield, lambda *_args: 1.0, None)
    )


def _distortion_widget(document, *, include_distortion: bool) -> MagneticFieldWidget:
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("fe", "Fe", 0),)))
    (site,) = relaxation_cluster_muon_site_options(document, feo_host_structure())
    widget._selected_muon_site = replace(site, include_distortion=include_distortion)
    widget._refresh_site_summary()
    return widget


def test_magnetic_field_widget_applies_the_muon_induced_distortion_correction(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())
    widget = _distortion_widget(document, include_distortion=True)

    assert "corrected for muon-induced displacements" in widget._site_summary.text()
    widget.calculate_fields()

    # Only the moved Fe (z moment of 1 μB, 1.5 Å -> 1.6 Å along x from the muon) changes the field.
    expected_bz = (
        -MU0_OVER_4PI_T_M_PER_A * BOHR_MAGNETON_J_PER_T / 1e-30 * (1.0 / 1.6**3 - 1.0 / 1.5**3)
    )
    assert "with muon-induced distortion corrections" in widget._calculation_status.text()
    assert widget._result_table.item(0, 5).text() == "field (distortion-corrected)"
    state = widget.collect_state()
    assert state["selected_muon_site"]["include_distortion"] is True
    payload = state["magnetic_field_results"]["results"][0]
    assert payload["distortion_T"] == pytest.approx([0.0, 0.0, expected_bz], abs=1e-12)
    assert payload["total_T"] == pytest.approx([0.0, 0.0, expected_bz], abs=1e-12)

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(state)
    assert restored.selected_muon_site().include_distortion is True
    assert restored._result_table.item(0, 5).text() == "field (distortion-corrected)"
    restored.close()
    widget.close()


def test_magnetic_field_widget_corrects_an_incommensurate_distribution(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())
    widget = _distortion_widget(document, include_distortion=True)
    helix = MagneticComponent(
        component_id="fe",
        label="Fe",
        kind="helix",
        selector=AtomSelector.indices([0]),
        terms=(
            FourierTerm(
                q_coeff=(1.0, 0.0, 0.0),
                coefficient=ComplexVector(real=(1.0, 0.0, 0.0), imaginary=(0.0, 1.0, 0.0)),
            ),
        ),
    )
    widget.set_model(
        MagneticStructureModel(
            q_basis_rlu=((0.12345, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            components=(helix,),
        )
    )

    widget.calculate_fields()

    # The zero lattice field leaves only the moved Fe's dipole change, turning with the helix.
    change = MU0_OVER_4PI_T_M_PER_A * BOHR_MAGNETON_J_PER_T / 1e-30 * (1.5**-3 - 1.6**-3)
    assert widget._calculation_status.text().startswith(
        "Results: sampled 1 incommensurate field distribution with muon-induced distortion corrections"
    )
    assert widget._result_table.item(0, 5).text() == "distribution mean (distortion-corrected)"
    state = widget.collect_state()
    (payload,) = state["magnetic_field_results"]["results"]
    assert change < payload["terms"]["dipolar"]["rms_T"] < 2.0 * change
    assert payload["calculation_parameters"]["distortion_corrected"] is True

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(state)
    assert restored._result_table.item(0, 5).text() == "distribution mean (distortion-corrected)"
    restored.close()
    widget.close()


def test_magnetic_field_widget_leaves_the_field_uncorrected_when_not_requested(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())
    widget = _distortion_widget(document, include_distortion=False)

    widget.calculate_fields()

    assert widget._result_table.item(0, 5).text() == "field"
    assert "distortion_T" not in widget.collect_state()["magnetic_field_results"]["results"][0]
    widget.close()


def test_magnetic_field_widget_reports_a_missing_relaxed_structure(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    relaxed = displaced_iron_supercell()
    document = relaxation_cluster_document(tmp_path, relaxed)
    widget = _distortion_widget(document, include_distortion=True)
    del document.structures[relaxed.id]

    widget.calculate_fields()

    assert (
        "relaxed structure for the muon-induced distortion correction is no longer available"
        in (widget._calculation_status.text())
    )
    assert widget._result_table.rowCount() == 0
    widget.close()


def test_magnetic_field_widget_defaults_to_a_relaxation_cluster_once_one_exists(
    qapp, tmp_path
) -> None:
    uep_widget = MagneticFieldWidget()
    uep_widget.set_document(uep_cluster_document(tmp_path / "uep"))
    assert uep_widget.selected_muon_site().source == "uep_clustering"
    assert "unrelaxed" in uep_widget._site_summary.text()
    uep_widget.close()

    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())
    add_uep_cluster(document)
    widget = MagneticFieldWidget()
    widget.set_document(document)
    assert widget.selected_muon_site().source == "relaxation_clustering"
    assert "unrelaxed" not in widget._site_summary.text()
    widget.close()


def test_magnetic_field_widget_marks_fields_at_a_uep_site_as_estimates_across_a_reload(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    service = ProjectService()
    document = uep_cluster_document(tmp_path)
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("fe", "Fe", 0),)))

    widget.calculate_fields()

    assert widget._result_table.item(0, 5).text() == "field, unrelaxed estimate"
    assert "unrelaxed UEP site" in widget._calculation_status.text()
    state = widget.collect_state()
    assert state["magnetic_field_results"]["site_source"] == "uep_clustering"
    document.ui_state["magnetic_field_form"] = store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, state
    )
    service.save_project(document)
    widget.close()

    reloaded = service.load_project(document.project_root)
    restored = MagneticFieldWidget()
    restored.set_document(reloaded)
    restored.set_form_state(
        attach_magnetic_field_results(
            reloaded, MAGNETIC_FIELD_PAGE_OWNER, reloaded.ui_state["magnetic_field_form"]
        )
    )
    assert restored._result_table.item(0, 5).text() == "field, unrelaxed estimate"
    assert "unrelaxed UEP site" in restored._calculation_status.text()
    restored.close()


def test_magnetic_field_widget_does_not_mark_fields_at_a_relaxation_cluster(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())
    widget = _distortion_widget(document, include_distortion=False)

    widget.calculate_fields()

    assert widget._result_table.item(0, 5).text() == "field"
    assert "estimate" not in widget._calculation_status.text()
    assert (
        widget.collect_state()["magnetic_field_results"]["site_source"] == "relaxation_clustering"
    )
    widget.close()


def test_magnetic_field_widget_enables_nuclear_relaxation_without_magnetic_model(
    qapp, tmp_path
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "nuclear-no-model.mufinder", "Nuclear No Model")
    service.add_structure(
        document,
        Structure(
            label="Si",
            lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            species=["Si"],
            fractional_positions=[[0.0, 0.0, 0.0]],
        ),
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    widget._sync_enabled_state()

    assert not widget._calculate_button.isEnabled()
    assert widget._nuclear_relaxation_button.isEnabled()
    widget.close()


def test_magnetic_field_widget_opens_nuclear_relaxation_dialog(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "nuclear-dialog.mufinder", "Nuclear Dialog")
    service.add_structure(document, feo_host_structure())

    calls: list[dict[str, object]] = []

    class _Dialog:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def options_state(self):
            return {"mode": "zf_polycrystal", "direction": [0.0, 0.0, 1.0]}

        def results_state(self):
            return {"site_label": "site 1", "sigma": 0.2}

    monkeypatch.setattr(magnetic_field_widget_module, "NuclearRelaxationDialog", _Dialog)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))

    widget._open_nuclear_relaxation_dialog()

    assert calls
    assert calls[0]["muon_site"].fractional_position == (0.25, 0.5, 0.75)
    assert calls[0]["distortion_available"] is False
    assert calls[0]["elements"] == ("Fe", "O")
    assert widget.collect_state()["nuclear_relaxation_results"]["sigma"] == 0.2
    widget.close()


def test_magnetic_field_widget_calculates_nuclear_relaxation_from_dialog(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "nuclear-calc.mufinder", "Nuclear Calc")
    service.add_structure(document, feo_host_structure())
    calls: list[dict[str, object]] = []

    class _Result:
        def to_json(self):
            return {"site_label": "Custom site", "sigma": 0.123, "mode": "zf_polycrystal"}

    class _Dialog:
        def options(self):
            return NuclearRelaxationOptions(mode="zf_polycrystal")

    def _calculate(structure, muon_position, *, site_label, options):
        calls.append(
            {
                "structure": structure,
                "muon_position": muon_position,
                "site_label": site_label,
                "options": options,
            }
        )
        return _Result()

    monkeypatch.setattr(
        magnetic_field_calculation_module, "calculate_nuclear_relaxation", _calculate
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))

    result = widget._calculate_nuclear_relaxation_from_dialog(_Dialog())

    assert result.to_json()["sigma"] == 0.123
    assert calls[0]["muon_position"] == (0.25, 0.5, 0.75)
    assert calls[0]["site_label"] == "Custom site"
    widget.close()


def test_magnetic_field_widget_passes_calculation_cell_counts(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-cells.mufinder", "Magnetic Cells")
    service.add_structure(document, feo_host_structure())

    calls: list[dict[str, object]] = []

    def _calculate(structure, model, muon_sites, *, options, distorted_structures=None):
        calls.append(
            {"structure": structure, "model": model, "muon_sites": muon_sites, "options": options}
        )
        return ()

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module, "calculate_commensurate_fields", _calculate
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    widget.set_calculation_settings(replace(widget.calculation_settings(), supercell=(24, 36, 48)))

    widget.calculate_fields()

    assert calls
    assert calls[0]["options"].supercell == (24, 36, 48)
    assert calls[0]["options"].nangles == 32_768
    assert calls[0]["options"].histogram_bins == 512
    widget.close()


def test_magnetic_field_widget_defaults_demag_shape_to_none(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-default-demag.mufinder", "Magnetic Default Demag"
    )
    service.add_structure(document, feo_host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)

    assert widget._demag_shape() == "none"

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state({"demag_shape": "unknown"})

    assert restored._demag_shape() == "none"
    widget.close()
    restored.close()


def test_magnetic_field_widget_opens_advanced_demag_dialog(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "advanced-demag.mufinder", "Advanced Demag")
    service.add_structure(document, feo_host_structure())

    class _Dialog:
        def __init__(self, sample, parent=None, *, lattice_vectors=None):
            assert sample.shape == "ellipsoid"
            assert lattice_vectors == feo_host_structure().lattice_vectors

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selected_sample(self):
            return DemagnetizingSample(
                shape="ellipsoid",
                semi_axes=(1.0, 2.0, 3.0),
            )

    monkeypatch.setattr(magnetic_field_widget_module, "DemagnetizingFieldAdvancedDialog", _Dialog)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    emitted: list[dict] = []
    widget.form_state_changed.connect(emitted.append)

    widget._open_advanced_demag_dialog()

    options = widget.field_options()
    assert widget._demag_shape() == "ellipsoid"
    assert options.demag_sample is not None
    assert options.demag_sample.semi_axes == pytest.approx((1.0, 2.0, 3.0))
    assert emitted[-1]["demag_shape"] == "ellipsoid"
    assert emitted[-1]["demag_sample"]["semi_axes"] == [1.0, 2.0, 3.0]
    widget.close()


def test_magnetic_field_widget_restores_a_photographed_plate(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "plate-demag.mufinder", "Plate Demag")
    service.add_structure(document, feo_host_structure())
    plate = photo_plate_sample(photo=PlatePhoto(path="/p/disc.png"))

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state({"demag_shape": "image_plate", "demag_sample": plate.to_json()})

    options = widget.field_options()
    assert widget._demag_shape() == "image_plate"
    assert options.demag_sample == plate
    assert widget._advanced_demag_summary.text() == "Plate from photo 5 x 5 x 0.2 (disc.png)"
    assert widget.collect_state()["demag_sample"] == plate.to_json()
    widget.close()


def test_magnetic_field_widget_restores_advanced_demag_state(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "advanced-demag-restore.mufinder", "Advanced Demag Restore"
    )
    service.add_structure(document, feo_host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(
        {
            "demag_shape": "rectangular_prism",
            "demag_sample": {
                "shape": "rectangular_prism",
                "prism_lengths": [2.0, 4.0, 6.0],
                "fourier_grid": 24,
                "padding": 3.0,
            },
        }
    )

    options = widget.field_options()

    assert widget._demag_shape() == "rectangular_prism"
    assert options.demag_sample is not None
    assert options.demag_sample.prism_lengths == pytest.approx((2.0, 4.0, 6.0))
    assert "Rectangular prism" in widget._advanced_demag_summary.text()
    widget.close()


def test_magnetic_field_widget_persists_and_resolves_demag_orientation(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "advanced-demag-orientation.mufinder", "Advanced Demag Orientation"
    )
    service.add_structure(document, feo_host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(
        {
            "demag_shape": "ellipsoid",
            "demag_sample": {
                "shape": "ellipsoid",
                "semi_axes": [1.0, 1.0, 3.0],
                "normal_hkl": [1.0, 1.0, 0.0],
                "in_plane_uvw": [0.0, 0.0, 1.0],
            },
        }
    )

    options = widget.field_options()

    assert options.demag_sample is not None
    np.testing.assert_allclose(
        options.demag_sample.rotation_matrix()[:, 2], np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    )
    assert "normal (1 1 0), in-plane [0 0 1]" in widget._advanced_demag_summary.text()

    widget.set_form_state(
        {
            "demag_shape": "ellipsoid",
            "demag_sample": {
                "shape": "ellipsoid",
                "rotation": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            },
        }
    )
    assert widget._advanced_demag_summary.text().endswith("; rotated sample axes")
    widget.set_form_state(
        {
            "demag_shape": "ellipsoid",
            "demag_sample": {
                "shape": "ellipsoid",
                "normal_hkl": [1.0, 1.0, 0.0],
                "in_plane_uvw": [0.0, 0.0, 1.0],
            },
        }
    )
    state = widget.collect_state()
    assert state["demag_sample"]["normal_hkl"] == [1.0, 1.0, 0.0]
    assert state["demag_sample"]["in_plane_uvw"] == [0.0, 0.0, 1.0]
    widget.close()


def test_magnetic_field_widget_measured_magnetisation_reaches_the_options_and_round_trips(
    qapp, tmp_path
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "measured-magnetisation.mufinder", "Measured Magnetisation"
    )
    service.add_structure(document, feo_host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    emitted: list[dict] = []
    widget.form_state_changed.connect(emitted.append)

    assert widget.field_options().measured_magnetization is None
    assert not any(edit.isEnabled() for edit in widget._measured_magnetization_input.edits)
    assert widget._measured_magnetization_unit.text().startswith("μB per FeO")

    widget._measured_magnetization_check.setChecked(True)
    x, y, z = widget._measured_magnetization_input.edits
    x.setText("0.5")
    y.setText("")
    z.setText("-1.25")

    assert all(edit.isEnabled() for edit in widget._measured_magnetization_input.edits)
    assert widget.field_options().measured_magnetization == (0.5, 0.0, -1.25)
    assert emitted[-1]["use_measured_magnetization"] is True
    assert emitted[-1]["measured_magnetization"] == [0.5, 0.0, -1.25]

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(widget.collect_state())

    assert restored._measured_magnetization_check.isChecked()
    assert restored.field_options().measured_magnetization == (0.5, 0.0, -1.25)

    # Unchecking keeps the values for later but falls back to the model's moment.
    restored._measured_magnetization_check.setChecked(False)
    assert restored.field_options().measured_magnetization is None
    assert restored.collect_state()["measured_magnetization"] == [0.5, 0.0, -1.25]

    restored.set_form_state({})
    assert not restored._measured_magnetization_check.isChecked()
    assert restored.collect_state()["measured_magnetization"] == [0.0, 0.0, 0.0]
    widget.close()
    restored.close()


def test_magnetic_field_widget_applied_field_reaches_the_calculation_and_round_trips(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "applied-field.mufinder", "Applied Field")
    service.add_structure(document, feo_host_structure())
    seen_options: list[object] = []

    def fake_fields(_host, _model, positions, *, options, **_kwargs):
        seen_options.append(options)
        return (
            magnetic_fields_module.MagneticFieldResult(
                site_label="muon 1",
                muon_position=tuple(positions[0]),
                dipolar=(0.1, 0.0, 0.0),
                lorentz=(0.0, 0.0, 0.0),
                contact=(0.0, 0.0, 0.0),
                demag=(0.0, 0.0, 0.0),
                applied=options.applied_field,
            ),
        )

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module, "calculate_commensurate_fields", fake_fields
    )
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    emitted: list[dict] = []
    widget.form_state_changed.connect(emitted.append)

    assert widget.field_options().applied_field == (0.0, 0.0, 0.0)
    assert all(edit.isEnabled() for edit in widget._applied_field_input.edits)
    x, y, z = widget._applied_field_input.edits
    x.setText("")
    y.setText("0.01")
    z.setText("-0.25")
    assert emitted[-1]["applied_field_T"] == [0.0, 0.01, -0.25]
    assert not widget._polarisation_button.isEnabled()

    widget.calculate_fields()

    assert seen_options[-1].applied_field == (0.0, 0.01, -0.25)
    assert widget._polarisation_button.isEnabled()
    requested: list[bool] = []
    widget.open_polarisation_requested.connect(lambda: requested.append(True))
    widget._polarisation_button.click()
    assert requested == [True]
    assert _breakdown_rows(widget)[4][:4] == ["Applied", "0", "0.01", "-0.25"]

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(widget.collect_state())
    assert restored.field_options().applied_field == (0.0, 0.01, -0.25)
    assert _breakdown_rows(restored)[4][:4] == ["Applied", "0", "0.01", "-0.25"]

    # Changing the field invalidates the shown results.
    restored._applied_field_input.edits[2].setText("0.5")
    assert restored.collect_state()["magnetic_field_results"] is None
    assert not restored._polarisation_button.isEnabled()

    restored.set_form_state({"applied_field_T": ["a", 1.0, 2.0]})
    assert restored.field_options().applied_field == (0.0, 0.0, 0.0)
    restored.set_form_state({})
    assert restored.collect_state()["applied_field_T"] == [0.0, 0.0, 0.0]
    widget.close()
    restored.close()


def test_magnetic_field_widget_ignores_a_malformed_saved_magnetisation(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "bad-magnetisation.mufinder", "Bad Magnetisation")
    service.add_structure(document, feo_host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(
        {
            "demag_shape": "sphere",
            "use_measured_magnetization": True,
            "measured_magnetization": ["a", 1.0],
        }
    )

    assert widget.field_options().measured_magnetization == (0.0, 0.0, 0.0)
    widget.close()


def test_magnetic_field_widget_reports_an_unfinished_measured_magnetisation(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "unfinished-magnetisation.mufinder", "Unfinished Magnetisation"
    )
    service.add_structure(document, feo_host_structure())
    calls: list[object] = []
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_commensurate_fields",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    widget._measured_magnetization_check.setChecked(True)
    widget._measured_magnetization_input.edits[0].setText("-")

    widget.calculate_fields()

    assert calls == []
    assert (
        "measured magnetisation components must be decimal numbers"
        in widget._calculation_status.text()
    )
    widget.close()


def test_magnetic_field_widget_empty_form_state_resets_previous_project_values(
    qapp, tmp_path
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-empty-state.mufinder", "Magnetic Empty State"
    )
    service.add_structure(document, feo_host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(
        {
            "demag_shape": "sphere",
            "supercell": [24, 36, 48],
            "include_equivalent_sites": True,
            "incommensurate_samples": 8192,
            "histogram_bins": 1024,
            "hyperfine_coupling": "2.5",
            "hyperfine_coupling_unit": "T_per_muB",
            "selected_muon_site": {
                "source": "custom",
                "label": "Previous site",
                "fractional_position": [0.25, 0.5, 0.75],
            },
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [
                    {
                        "site_label": "site 1",
                        "total_T": [0.1, 0.2, 0.3],
                        "magnitude_T": 0.374,
                    }
                ],
            },
            "dipolar_tensor_selector": {"mode": "species", "species": "Fe"},
            "dipolar_tensor_results": [{"site_label": "site 1"}],
            "use_commensurate_approximants": True,
            "approximant_tolerance": 0.004,
            "approximant_max_period": 20,
        }
    )

    widget.set_form_state({})

    assert widget.collect_state()["use_commensurate_approximants"] is False
    assert widget.field_options().approximant_tolerance == 1e-3
    assert widget.field_options().approximant_max_period == 12

    assert widget._demag_shape() == "none"
    assert widget.field_options().supercell == (100, 100, 100)
    assert widget._include_equivalent_sites.isChecked() is False
    assert widget.calculation_settings() == MagneticFieldCalculationSettings()
    assert widget.field_options().nangles == 32_768
    assert widget.field_options().phase_sample_count == 200_000
    assert widget.field_options().phase_sampling == "auto"
    assert widget.field_options().histogram_bins == 512
    assert widget._hyperfine_coupling.text() == "0.0"
    assert widget._current_hyperfine_unit() == "angstrom^-3"
    assert widget.selected_muon_site() is None
    assert widget._result_table.rowCount() == 0
    assert widget.collect_state()["magnetic_field_results"] is None
    assert widget.collect_state()["dipolar_tensor_selector"] == {"mode": "assigned"}
    assert widget.collect_state()["dipolar_tensor_results"] == []
    widget.close()


def test_magnetic_field_calculation_settings_dialog_collects_values(qapp) -> None:
    dialog = MagneticFieldCalculationSettingsDialog(
        MagneticFieldCalculationSettings(supercell=(12, 24, 36), nangles=4096)
    )

    assert dialog._calc_cells_b.value() == 24
    assert dialog._lookup_angles.value() == 4096
    dialog._calc_cells_a.setValue(18)
    dialog._lookup_angles.setValue(8192)
    dialog._phase_samples.setValue(250_000)
    dialog._phase_sampling.setCurrentIndex(dialog._phase_sampling.findData("montecarlo"))
    dialog._histogram_bins.setValue(1024)
    dialog._use_approximants.setChecked(True)
    dialog._approximant_tolerance.setValue(0.002)
    dialog._approximant_max_period.setValue(8)
    dialog._contact_nnn.setValue(4)
    dialog._contact_rcont.setValue(3.5)

    assert dialog.settings() == MagneticFieldCalculationSettings(
        supercell=(18, 24, 36),
        nangles=8192,
        phase_sample_count=250_000,
        phase_sampling="montecarlo",
        histogram_bins=1024,
        use_approximants=True,
        approximant_tolerance=0.002,
        approximant_max_period=8,
        contact_nnn=4,
        contact_rcont=3.5,
    )
    assert dialog._phase_samples.maximum() == MAX_INCOMMENSURATE_PHASE_SAMPLES
    dialog.close()


def test_magnetic_field_widget_opens_calculation_settings_dialog(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-settings-dialog.mufinder", "Magnetic Settings Dialog"
    )
    service.add_structure(document, feo_host_structure())
    chosen = MagneticFieldCalculationSettings(
        supercell=(24, 36, 48),
        nangles=8192,
        phase_sample_count=250_000,
        phase_sampling="grid",
        histogram_bins=1024,
        use_approximants=True,
        approximant_tolerance=0.002,
        approximant_max_period=8,
        contact_nnn=4,
        contact_rcont=3.5,
    )

    class _Dialog:
        def __init__(self, settings, *, contact_tie_check=None, parent=None):
            # A fresh page opens the dialog on the default settings.
            assert settings == MagneticFieldCalculationSettings()
            # The page's host has no hyperfine coupling, so there is no shell to check.
            assert contact_tie_check is None

        def exec(self):
            return QDialog.DialogCode.Accepted

        def settings(self):
            return chosen

    monkeypatch.setattr(
        magnetic_field_widget_module, "MagneticFieldCalculationSettingsDialog", _Dialog
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    emitted: list[dict] = []
    widget.form_state_changed.connect(emitted.append)

    widget._open_calculation_settings_dialog()

    assert widget.calculation_settings() == chosen
    assert widget.field_options().supercell == (24, 36, 48)
    assert widget.field_options().nangles == 8192
    assert widget.field_options().phase_sample_count == 250_000
    assert widget.field_options().phase_sampling == "grid"
    assert widget.field_options().histogram_bins == 1024
    assert widget.field_options().approximant_tolerance == 0.002
    assert widget.field_options().approximant_max_period == 8
    assert widget.field_options().nnn == 4
    assert widget.field_options().rcont == 3.5
    assert "cells 24 x 36 x 48" in widget._calculation_settings_summary.text()
    assert "commensurate approximants within 0.002 r.l.u., period up to 8" in (
        widget._calculation_settings_summary.text()
    )
    assert emitted[-1]["supercell"] == [24, 36, 48]
    assert emitted[-1]["use_commensurate_approximants"] is True
    assert emitted[-1]["approximant_tolerance"] == 0.002
    assert emitted[-1]["approximant_max_period"] == 8
    assert emitted[-1]["contact_nnn"] == 4
    assert emitted[-1]["contact_rcont"] == 3.5
    widget.close()


def test_calculation_settings_dialog_defaults_the_contact_shell_to_muesr(qapp) -> None:
    dialog = MagneticFieldCalculationSettingsDialog()

    assert dialog.settings().contact_nnn == 2
    assert dialog.settings().contact_rcont == 10.0
    dialog.close()


def test_calculation_settings_dialog_refuses_an_empty_contact_shell(qapp) -> None:
    dialog = MagneticFieldCalculationSettingsDialog()

    dialog._contact_nnn.setValue(0)
    dialog._contact_rcont.setValue(0.0)
    dialog._contact_rcont.setValue(-2.0)

    assert dialog.settings().contact_nnn == 1
    assert dialog.settings().contact_rcont > 0.0
    assert dialog._contact_nnn.minimum() == 1
    assert dialog._contact_rcont.minimum() > 0.0
    dialog.close()


def test_settings_summary_names_the_contact_shell_only_with_a_coupling(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "contact-summary.mufinder", "Contact Summary")
    service.add_structure(document, feo_host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state({"contact_nnn": 1, "contact_rcont": 2.5})

    assert "contact shell" not in widget._calculation_settings_summary.text()

    widget._hyperfine_coupling.setText("0.5")

    assert "contact shell of 1 nearest magnetic ion within 2.5 Å" in (
        widget._calculation_settings_summary.text()
    )
    # The validator lets a half-typed number through; it is no coupling yet.
    widget._hyperfine_coupling.clear()
    QTest.keyClicks(widget._hyperfine_coupling, "-")
    assert widget._hyperfine_coupling.text() == "-"
    assert "contact shell" not in widget._calculation_settings_summary.text()
    widget.close()


@pytest.mark.parametrize(
    "stored",
    [{}, {"contact_nnn": 0, "contact_rcont": -1.0}, {"contact_nnn": "x", "contact_rcont": None}],
    ids=["missing", "out-of-range", "malformed"],
)
def test_form_state_without_a_valid_contact_shell_uses_the_defaults(qapp, tmp_path, stored) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "contact-defaults.mufinder", "Contact Defaults")
    service.add_structure(document, feo_host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state({"contact_nnn": 5, "contact_rcont": 4.0})
    widget.set_form_state(stored)

    assert widget.field_options().nnn == 2
    assert widget.field_options().rcont == 10.0
    widget.close()


def test_form_state_rounds_the_contact_radius_to_what_the_dialog_shows(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "contact-rounding.mufinder", "Contact Rounding")
    service.add_structure(document, feo_host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)

    widget.set_form_state({"contact_rcont": 3.456})

    assert widget.field_options().rcont == 3.46
    widget.close()


def test_contact_shell_survives_save_and_reopen_with_its_result(
    qapp, tmp_path, monkeypatch
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    service = ProjectService()
    document = uep_cluster_document(tmp_path)
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("fe", "Fe", 0),)))
    widget.set_form_state({**widget.collect_state(), "contact_nnn": 4, "contact_rcont": 3.5})
    widget._hyperfine_coupling.setText("0.5")

    widget.calculate_fields()

    state = widget.collect_state()
    document.ui_state["magnetic_field_form"] = store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, state
    )
    service.save_project(document)
    widget.close()

    reloaded = service.load_project(document.project_root)
    restored = MagneticFieldWidget()
    restored.set_document(reloaded)
    restored.set_form_state(
        attach_magnetic_field_results(
            reloaded, MAGNETIC_FIELD_PAGE_OWNER, reloaded.ui_state["magnetic_field_form"]
        )
    )

    assert restored.field_options().nnn == 4
    assert restored.field_options().rcont == 3.5
    (result,) = restored.collect_state()["magnetic_field_results"]["results"]
    assert result["calculation_parameters"]["nnn"] == 4
    assert result["calculation_parameters"]["rcont"] == 3.5
    assert result["calculation_parameters"]["contact_coupling"] == pytest.approx(0.5)
    restored.close()


def test_magnetic_field_widget_opens_dipolar_tensor_dialog(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-tensor-dialog.mufinder", "Magnetic Tensor Dialog"
    )
    service.add_structure(document, feo_host_structure())

    opened: list[object] = []
    calculated: list[dict[str, object]] = []

    class _Dialog:
        def __init__(
            self,
            structure,
            model,
            muon_site,
            *,
            include_equivalent_sites,
            selector_state,
            supercell,
            results_state,
            calculate_callback,
            parent=None,
        ):
            self._calculate_callback = calculate_callback
            self._results_state = list(results_state)
            opened.append(
                {
                    "structure": structure,
                    "model": model,
                    "muon_site": muon_site,
                    "include_equivalent_sites": include_equivalent_sites,
                    "selector_state": selector_state,
                    "supercell": supercell,
                    "results_state": results_state,
                }
            )

        def exec(self):
            self._results_state = [
                {
                    "site_label": result.site_label,
                    "muon_position": list(result.muon_position),
                    "dipolar_tensor_T_per_muB": np.asarray(
                        result.tensor_T_per_muB, dtype=float
                    ).tolist(),
                }
                for result in self._calculate_callback(self)
            ]
            return QDialog.DialogCode.Rejected

        def selector_state(self):
            return {
                "mode": "selector",
                "selector": AtomSelector.indices([0]).to_json(),
                "include_equivalent_sites": True,
            }

        def selected_ion_indices(self):
            return (0,)

        def include_equivalent_sites(self):
            return True

        def results_state(self):
            return self._results_state

    class _Result:
        site_label = "site 1"
        muon_position = (0.25, 0.25, 0.25)
        tensor_T_per_muB = np.eye(3)

    def _calculate(structure, muon_sites, *, magnetic_ion_indices, options):
        calculated.append(
            {
                "structure": structure,
                "muon_sites": tuple(muon_sites),
                "magnetic_ion_indices": tuple(magnetic_ion_indices),
                "options": options,
            }
        )
        return (_Result(),)

    monkeypatch.setattr(magnetic_field_widget_module, "DipolarTensorDialog", _Dialog)
    monkeypatch.setattr(
        magnetic_field_calculation_module, "calculate_dipolar_tensors_for_atoms", _calculate
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.25, 0.25))
    widget._include_equivalent_sites.setChecked(False)

    widget._open_dipolar_tensor_dialog()

    assert opened
    assert opened[0]["include_equivalent_sites"] is False
    assert opened[0]["supercell"] == (100, 100, 100)
    assert calculated[0]["muon_sites"] == ((0.25, 0.25, 0.25),)
    assert calculated[0]["magnetic_ion_indices"] == (0,)
    assert calculated[0]["options"].find_equivalent_sites is True
    assert calculated[0]["options"].demag_shape == "none"
    assert calculated[0]["options"].contact_coupling is None
    assert (
        widget.collect_state()["dipolar_tensor_selector"]["selector"]
        == AtomSelector.indices([0]).to_json()
    )
    assert widget.collect_state()["dipolar_tensor_results"][0]["site_label"] == "site 1"
    widget.close()


def test_magnetic_field_widget_passes_incommensurate_sampling_controls(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-sampling.mufinder", "Magnetic Sampling")
    service.add_structure(document, feo_host_structure())

    calls: list[dict[str, object]] = []

    def _calculate(structure, model, muon_sites, *, options, distorted_structures=None):
        calls.append(
            {"structure": structure, "model": model, "muon_sites": muon_sites, "options": options}
        )
        return ()

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "incommensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_incommensurate_field_distributions",
        _calculate,
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    widget.set_calculation_settings(
        replace(
            widget.calculation_settings(),
            phase_sample_count=250_000,
            phase_sampling="montecarlo",
            histogram_bins=1024,
        )
    )

    widget.calculate_fields()

    assert calls
    assert calls[0]["options"].nangles == 32_768
    assert calls[0]["options"].phase_sample_count == 250_000
    assert calls[0]["options"].phase_sampling == "montecarlo"
    assert calls[0]["options"].histogram_bins == 1024
    widget.close()


def test_magnetic_field_widget_converts_hyperfine_coupling_units(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-hyperfine.mufinder", "Magnetic Hyperfine"
    )
    service.add_structure(document, feo_host_structure())

    calls: list[dict[str, object]] = []

    def _calculate(structure, model, muon_sites, *, options, distorted_structures=None):
        calls.append(
            {"structure": structure, "model": model, "muon_sites": muon_sites, "options": options}
        )
        return ()

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module, "calculate_commensurate_fields", _calculate
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    widget._hyperfine_coupling.setText("12.5")
    widget._hyperfine_units.setCurrentIndex(widget._hyperfine_units.findData("T_per_muB"))

    widget.calculate_fields()

    assert calls
    assert calls[0]["options"].contact_coupling == pytest.approx(
        contact_coupling_to_angstrom_minus3(12.5, "T_per_muB")
    )
    assert [
        widget._hyperfine_units.itemData(index) for index in range(widget._hyperfine_units.count())
    ] == [
        "angstrom^-3",
        "T_per_muB",
        "MHz_per_muB",
    ]
    widget.close()


def test_magnetic_field_widget_passes_equivalent_site_option(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-equivalent-sites.mufinder", "Magnetic Equivalent Sites"
    )
    service.add_structure(document, feo_host_structure())

    calls: list[dict[str, object]] = []

    def _calculate(structure, model, muon_sites, *, options, distorted_structures=None):
        calls.append(
            {"structure": structure, "model": model, "muon_sites": muon_sites, "options": options}
        )
        return ()

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module, "calculate_commensurate_fields", _calculate
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.25, 0.25))
    widget._include_equivalent_sites.setChecked(True)

    widget.calculate_fields()

    assert calls
    assert calls[0]["options"].find_equivalent_sites is True
    widget.close()


def test_magnetic_field_widget_collects_and_restores_form_state(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-form-state.mufinder", "Magnetic Form State"
    )
    service.add_structure(document, feo_host_structure())

    distortion_id = uuid4()
    selected_site = MagneticMuonSiteOption(
        source="relaxation_cluster",
        label="Relaxed site",
        fractional_position=(0.125, 0.25, 0.375),
        cartesian_position=(0.375, 0.75, 1.125),
        cluster_label="cluster 2",
        source_label="Pristine relaxation",
        distortion_structure_id=distortion_id,
        properties={"energy_eV": -12.5},
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget._selected_muon_site = selected_site
    widget._plate_radio.setChecked(True)
    widget.set_calculation_settings(
        replace(
            widget.calculation_settings(),
            supercell=(24, 36, 48),
            phase_sample_count=250_000,
            phase_sampling="grid",
            histogram_bins=1024,
        )
    )
    widget._include_equivalent_sites.setChecked(True)
    widget._hyperfine_coupling.setText("2.5")
    widget._hyperfine_units.setCurrentIndex(widget._hyperfine_units.findData("MHz_per_muB"))
    widget._dipolar_tensor_selector_state = {
        "mode": "selector",
        "selector": AtomSelector.species("Fe").to_json(),
        "include_equivalent_sites": True,
    }
    widget._dipolar_tensor_results_state = [
        {
            "site_label": "site 1",
            "muon_position": [0.125, 0.25, 0.375],
            "dipolar_tensor_T_per_muB": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        }
    ]
    widget._field_results_state = {
        "kind": "commensurate",
        "results": [
            {
                "site_label": "site 1",
                "muon_position": [0.125, 0.25, 0.375],
                "total_T": [0.1, 0.2, 0.3],
                "magnitude_T": 0.37416573867739417,
            }
        ],
    }
    widget._nuclear_relaxation_options_state = {
        "mode": "zf_single_crystal",
        "direction": [0.0, 1.0, 0.0],
        "include_quadrupolar": True,
        "radius": 12.0,
        "convergence_rtol": 1e-5,
        "use_distortion_correction": True,
    }
    widget._nuclear_relaxation_results_state = {
        "site_label": "Relaxed site",
        "sigma": 0.25,
        "sigma_squared": 0.0625,
        "mode": "zf_single_crystal",
    }

    state = widget.collect_state()

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(state)

    assert restored._demag_shape() == "infinite_plate"
    assert restored.field_options().supercell == (24, 36, 48)
    assert restored.field_options().find_equivalent_sites is True
    assert restored.field_options().nangles == 32_768
    assert restored.field_options().phase_sample_count == 250_000
    assert restored.field_options().phase_sampling == "grid"
    assert restored.field_options().histogram_bins == 1024
    assert restored.collect_state()["hyperfine_coupling"] == "2.5"
    assert restored.collect_state()["hyperfine_coupling_unit"] == "MHz_per_muB"
    assert restored.collect_state()["dipolar_tensor_selector"] == {
        "mode": "selector",
        "selector": AtomSelector.species("Fe").to_json(),
        "include_equivalent_sites": True,
    }
    assert restored.collect_state()["dipolar_tensor_results"][0]["site_label"] == "site 1"
    assert restored.collect_state()["nuclear_relaxation_options"]["mode"] == "zf_single_crystal"
    assert restored.collect_state()["nuclear_relaxation_results"]["sigma"] == pytest.approx(0.25)
    assert restored.collect_state()["magnetic_field_results"]["kind"] == "commensurate"
    assert restored._result_table.rowCount() == 1
    assert restored._result_table.item(0, 1).text() == "0.1"
    assert restored.field_options().contact_coupling == pytest.approx(
        contact_coupling_to_angstrom_minus3(2.5, "MHz_per_muB")
    )
    assert restored.selected_muon_site() == selected_site
    widget.close()
    restored.close()


def test_magnetic_field_widget_restores_distribution_results(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-restore-dist.mufinder", "Magnetic Restore Dist"
    )
    service.add_structure(document, feo_host_structure())

    state = {
        "magnetic_field_results": {
            "kind": "distribution",
            "results": [
                {
                    "site_label": "site 1",
                    "muon_position": [0.0, 0.0, 0.0],
                    "fields_T": [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]],
                    "histogram_centres_T": [1.0, 3.0],
                    "histogram_counts": [1.0, 1.0],
                    "calculation_parameters": {"nangles": 4096},
                }
            ],
        }
    }

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(state)

    assert widget._result_table.rowCount() == 1
    assert widget._result_table.item(0, 5).text() == "distribution mean"
    assert widget._open_distribution_button.isEnabled()
    assert widget._selected_distribution_results()[0].site_label == "site 1"
    widget.close()


def test_magnetic_field_widget_scrolls_a_long_component_summary(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-long-summary.mufinder", "Magnetic Long Summary"
    )
    service.add_structure(document, feo_host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)
    summary = widget._component_summary_table

    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    short_height = summary.maximumHeight()
    assert summary.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded

    limit = MagneticFieldWidget._SUMMARY_MAX_VISIBLE_ROWS
    components = tuple(
        z_moment_component(f"c{index}", f"C{index}", index % 2) for index in range(limit + 4)
    )
    widget.set_model(MagneticStructureModel(components=components))
    assert summary.rowCount() == limit + 4
    capped_height = summary.maximumHeight()
    assert short_height < capped_height
    assert capped_height < summary.horizontalHeader().sizeHint().height() + sum(
        summary.rowHeight(row) for row in range(summary.rowCount())
    )
    widget.close()


def test_magnetic_field_widget_places_demagnetizing_controls_in_the_calculation_group(qapp) -> None:
    widget = MagneticFieldWidget()

    for control in (
        widget._sphere_radio,
        widget._plate_radio,
        widget._none_radio,
        widget._advanced_demag_radio,
        widget._advanced_demag_button,
        widget._advanced_demag_summary,
    ):
        assert control.parentWidget().title() == "Calculation"
    assert widget._advanced_demag_summary.isHidden()
    widget._advanced_demag_radio.setChecked(True)
    assert not widget._advanced_demag_summary.isHidden()
    widget._sphere_radio.setChecked(True)
    assert widget._advanced_demag_summary.isHidden()
    widget.close()


def _breakdown_rows(widget: MagneticFieldWidget) -> list[list[str]]:
    table = widget._breakdown_table
    return [
        [table.item(row, column).text() for column in range(table.columnCount())]
        for row in range(table.rowCount())
    ]


def test_magnetic_field_widget_breaks_a_commensurate_field_into_its_terms(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-breakdown.mufinder", "Magnetic Breakdown"
    )
    service.add_structure(document, feo_host_structure())
    result = magnetic_fields_module.MagneticFieldResult(
        site_label="muon 1",
        muon_position=(0.25, 0.5, 0.75),
        dipolar=(0.1, 0.0, 0.0),
        lorentz=(0.0, 0.2, 0.0),
        contact=(0.0, 0.0, 0.3),
        demag=(0.0, -0.4, 0.0),
        applied=(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_commensurate_fields",
        lambda *_args, **_kwargs: (result,),
    )
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    assert widget._breakdown_table.isHidden()

    widget.calculate_fields()

    expected = [
        ["Dipolar", "0.1", "0", "0", "0.1"],
        ["Lorentz", "0", "0.2", "0", "0.2"],
        ["Contact", "0", "0", "0.3", "0.3"],
        ["Demagnetizing", "0", "-0.4", "0", "0.4"],
        ["Applied", "0", "0", "0", "0"],
        ["Total", "0.1", "-0.2", "0.3", f"{np.linalg.norm([0.1, -0.2, 0.3]):.6g}"],
    ]
    assert not widget._breakdown_table.isHidden()
    assert _breakdown_rows(widget) == expected
    assert widget._breakdown_hint.isHidden()

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(widget.collect_state())
    assert _breakdown_rows(restored) == expected
    restored.close()

    widget._clear_results(clear_state=True)
    assert widget._breakdown_table.isHidden()
    assert widget._breakdown_table.rowCount() == 0
    widget.close()


def test_magnetic_field_widget_shows_numerically_zero_field_components_as_zero(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-zero-display.mufinder", "Magnetic Zero Display"
    )
    service.add_structure(document, feo_host_structure())
    result = magnetic_fields_module.MagneticFieldResult(
        site_label="muon 1",
        muon_position=(0.5, 0.5, 0.0),
        dipolar=(1.28162e-18, -5.32134e-18, -0.297712),
        lorentz=(0.0, 0.0, 0.143831),
        contact=(0.0, 0.0, 0.0),
        demag=(-0.0, -0.0, 0.0),
        applied=(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "commensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_commensurate_fields",
        lambda *_args, **_kwargs: (result,),
    )
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.5, 0.5, 0.0))

    widget.calculate_fields()

    expected_result_row = ["muon 1", "0", "0", "-0.153881", "0.153881", "field"]
    assert [
        widget._result_table.item(0, column).text() for column in range(6)
    ] == expected_result_row
    expected_breakdown = [
        ["Dipolar", "0", "0", "-0.297712", "0.297712"],
        ["Lorentz", "0", "0", "0.143831", "0.143831"],
        ["Contact", "0", "0", "0", "0"],
        ["Demagnetizing", "0", "0", "0", "0"],
        ["Applied", "0", "0", "0", "0"],
        ["Total", "0", "0", "-0.153881", "0.153881"],
    ]
    assert _breakdown_rows(widget) == expected_breakdown

    stored = widget.collect_state()["magnetic_field_results"]["results"][0]
    assert stored["total_T"][:2] == [1.28162e-18, -5.32134e-18]
    assert stored["dipolar_T"][:2] == [1.28162e-18, -5.32134e-18]

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(widget.collect_state())
    assert [
        restored._result_table.item(0, column).text() for column in range(6)
    ] == expected_result_row
    assert _breakdown_rows(restored) == expected_breakdown
    restored.close()
    widget.close()


def test_magnetic_field_widget_explains_results_without_a_term_breakdown(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-no-breakdown.mufinder", "Magnetic No Breakdown"
    )
    service.add_structure(document, feo_host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)

    widget.set_form_state(
        {
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [
                    {"site_label": "site 1", "total_T": [0.0, 0.0, 1.0], "magnitude_T": 1.0}
                ],
            }
        }
    )
    assert widget._breakdown_table.isHidden()
    assert (
        widget._breakdown_hint.text()
        == "This saved result has no term breakdown; recalculate to see it."
    )

    widget.set_form_state(
        {
            "magnetic_field_results": {
                "kind": "distribution",
                "results": [
                    {
                        "site_label": "site 1",
                        "muon_position": [0.0, 0.0, 0.0],
                        "fields_T": [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]],
                        "histogram_centres_T": [1.0, 3.0],
                        "histogram_counts": [1.0, 1.0],
                    }
                ],
            }
        }
    )
    assert widget._breakdown_table.isHidden()
    assert (
        widget._breakdown_hint.text()
        == "This saved distribution has no term breakdown; recalculate to see it."
    )
    widget.close()


def test_magnetic_field_widget_emits_form_state_for_setting_changes(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-form-emits.mufinder", "Magnetic Form Emits"
    )
    service.add_structure(document, feo_host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    emitted: list[dict] = []
    widget.form_state_changed.connect(emitted.append)

    widget._none_radio.setChecked(True)
    widget.set_calculation_settings(replace(widget.calculation_settings(), histogram_bins=768))
    widget._hyperfine_coupling.setText("1.25")

    assert emitted
    assert emitted[-1]["demag_shape"] == "none"
    assert emitted[-1]["histogram_bins"] == 768
    assert emitted[-1]["hyperfine_coupling"] == "1.25"
    widget.close()


def test_magnetic_field_widget_emits_pin_title_and_delete_signals(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-workspace-signals.mufinder", "Magnetic Workspace"
    )
    service.add_structure(document, feo_host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document, title="Pinned helix", title_editable=True, deletable=True)
    pinned: list[dict] = []
    renamed: list[str] = []
    deleted: list[bool] = []
    widget.pin_requested.connect(pinned.append)
    widget.workspace_title_changed.connect(renamed.append)
    widget.delete_requested.connect(lambda: deleted.append(True))

    widget.set_calculation_settings(
        replace(widget.calculation_settings(), phase_sample_count=250_000)
    )
    widget._pin_button.click()
    widget._title_edit.setText("Renamed helix")
    widget._on_title_edit_finished()
    widget._delete_workspace_button.click()

    assert not widget._title_edit.isHidden()
    assert not widget._delete_workspace_button.isHidden()
    assert pinned[0]["title"] == "Pinned helix"
    assert pinned[0]["form"]["incommensurate_samples"] == 250_000
    assert renamed == ["Renamed helix"]
    assert deleted == [True]
    widget.close()


def test_magnetic_field_widget_calculates_incommensurate_distribution(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-dist.mufinder", "Magnetic Dist")
    service.add_structure(document, feo_host_structure())

    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "incommensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: (_Distribution(),),
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))

    widget.calculate_fields()

    assert widget._result_table.rowCount() == 1
    assert widget._result_table.item(0, 3).text() == "2"
    assert widget._result_table.item(0, 4).text() == "2"
    assert widget._result_table.item(0, 5).text() == "distribution mean"
    assert widget._open_distribution_button.isEnabled()
    assert widget.collect_state()["magnetic_field_results"]["kind"] == "distribution"
    assert widget.collect_state()["magnetic_field_results"]["results"][0]["fields_T"] == [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 3.0],
    ]
    widget.close()


def _watch_result_row_resizes(widget: MagneticFieldWidget) -> list[None]:
    """Make new result rows too tall until resized, and return a list that grows per resize."""

    table = widget._result_table
    table.verticalHeader().setDefaultSectionSize(200)
    resizes: list[None] = []
    resize_rows = table.resizeRowsToContents

    def counted_resize() -> None:
        resizes.append(None)
        resize_rows()

    table.resizeRowsToContents = counted_resize
    return resizes


def _assert_one_resize_fits_every_row(
    widget: MagneticFieldWidget, resizes: list[None], row_count: int
) -> None:
    table = widget._result_table
    assert table.rowCount() == row_count
    assert len(resizes) == 1
    heights = [table.rowHeight(row) for row in range(row_count)]
    assert max(heights) < 200
    table.resizeRowsToContents()
    assert [table.rowHeight(row) for row in range(row_count)] == heights


@pytest.mark.parametrize("row_count", [1, 5])
@pytest.mark.parametrize(
    ("classification", "science_call", "result_attributes"),
    [
        pytest.param(
            "commensurate",
            "calculate_commensurate_fields",
            {"total": (0.1, 0.2, 0.3), "magnitude": 0.37, "distortion": None},
            id="commensurate",
        ),
        pytest.param(
            "incommensurate",
            "calculate_incommensurate_field_distributions",
            {"fields_T": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)},
            id="distribution",
        ),
    ],
)
def test_magnetic_field_widget_resizes_calculated_result_rows_once(
    qapp, tmp_path, monkeypatch, classification, science_call, result_attributes, row_count
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-rows.mufinder", "Magnetic Rows")
    service.add_structure(document, feo_host_structure())

    result_type = type("_Result", (), {"site_label": "site 1", **result_attributes})
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: classification,
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        science_call,
        lambda *_args, **_kwargs: tuple(result_type() for _ in range(row_count)),
    )

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    resizes = _watch_result_row_resizes(widget)

    widget.calculate_fields()

    _assert_one_resize_fits_every_row(widget, resizes, row_count)
    assert widget._result_table.selectedItems()[0].row() == 0
    widget.close()


@pytest.mark.parametrize("row_count", [1, 5])
@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        pytest.param(
            "commensurate",
            {"site_label": "site 1", "total_T": [0.1, 0.2, 0.3], "magnitude_T": 0.37},
            id="commensurate",
        ),
        pytest.param(
            "distribution",
            {
                "site_label": "site 1",
                "muon_position": [0.0, 0.0, 0.0],
                "fields_T": [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]],
                "histogram_centres_T": [1.0, 3.0],
                "histogram_counts": [1.0, 1.0],
                "calculation_parameters": {"nangles": 4096},
            },
            id="distribution",
        ),
    ],
)
def test_magnetic_field_widget_resizes_restored_result_rows_once(
    qapp, tmp_path, kind, payload, row_count
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-rows.mufinder", "Magnetic Rows")
    service.add_structure(document, feo_host_structure())

    widget = MagneticFieldWidget()
    widget.set_document(document)
    resizes = _watch_result_row_resizes(widget)

    widget.set_form_state(
        {"magnetic_field_results": {"kind": kind, "results": [payload] * row_count}}
    )

    _assert_one_resize_fits_every_row(widget, resizes, row_count)
    assert widget._result_table.selectedItems()[0].row() == 0
    widget.close()


def _breakdown_headers(widget: MagneticFieldWidget) -> list[str]:
    table = widget._breakdown_table
    return [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]


def test_magnetic_field_widget_breaks_a_distribution_into_term_means_and_rms(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-dist-breakdown.mufinder", "Magnetic Dist Breakdown"
    )
    service.add_structure(document, feo_host_structure())
    stats = magnetic_fields_module.FieldTermStatistics
    distribution = magnetic_fields_module.MagneticFieldDistributionResult(
        site_label="site 1",
        muon_position=(0.0, 0.0, 0.0),
        fields_T=np.asarray([[0.2, 0.0, 0.1], [-0.2, 0.0, 0.1]], dtype=float),
        histogram_centres_T=np.asarray([0.2236], dtype=float),
        histogram_counts=np.asarray([2.0], dtype=float),
        convergence=(
            magnetic_fields_module.DistributionConvergenceCheck(
                setting="histogram_bins",
                label="Histogram bins",
                converged=True,
                metric=2.0,
                message="2 samples per occupied histogram bin.",
            ),
        ),
        terms={
            "dipolar": stats(mean_T=(1e-18, 0.0, 0.0), rms_T=0.2),
            "lorentz": stats(mean_T=(0.0, 0.0, 0.0), rms_T=0.0),
            "contact": stats(mean_T=(0.0, 0.0, 0.05), rms_T=0.05),
            "demag": stats(mean_T=(0.0, 0.0, 0.0), rms_T=0.0),
            "applied": stats(mean_T=(0.0, 0.0, 0.05), rms_T=0.05),
        },
        total_rms_T=float(np.hypot(0.2, 0.1)),
        applied_T=(0.0, 0.0, 0.05),
        demag_T=(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "incommensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: (distribution,),
    )
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))

    widget.calculate_fields()

    expected = [
        ["Dipolar", "0", "0", "0", "0.2"],
        ["Lorentz", "0", "0", "0", "0"],
        ["Contact", "0", "0", "0.05", "0.05"],
        ["Demagnetizing", "0", "0", "0", "0"],
        ["Applied", "0", "0", "0.05", "0.05"],
        ["Total", "0", "0", "0.1", "0.223607"],
    ]
    assert not widget._breakdown_table.isHidden()
    assert _breakdown_headers(widget) == [
        "Term",
        "Mean Bx (T)",
        "Mean By (T)",
        "Mean Bz (T)",
        "RMS |B| (T)",
    ]
    assert _breakdown_rows(widget) == expected
    assert widget._breakdown_hint.text().splitlines() == [
        "Convergence checks:",
        "Histogram bins: OK. 2 samples per occupied histogram bin.",
    ]

    restored = MagneticFieldWidget()
    restored.set_document(document)
    restored.set_form_state(widget.collect_state())
    assert _breakdown_rows(restored) == expected
    restored.close()

    widget.set_form_state(
        {
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [
                    {
                        "site_label": "site 1",
                        "total_T": [0.0, 0.0, 1.0],
                        "dipolar_T": [0.0, 0.0, 1.0],
                    }
                ],
            }
        }
    )
    assert _breakdown_headers(widget)[1] == "Bx (T)"

    corrupt = widget.collect_state()
    corrupt["magnetic_field_results"] = restored_state = {
        "kind": "distribution",
        "results": [dict(distribution.to_json())],
    }
    restored_state["results"][0]["terms"]["dipolar"] = {"mean_T": [0.0, 0.0, 0.0], "rms_T": "n/a"}
    widget.set_form_state(corrupt)
    widget._result_table.selectRow(0)
    assert widget._breakdown_table.isHidden()
    assert widget._breakdown_hint.text().endswith(
        "This saved distribution has no term breakdown; recalculate to see it."
    )
    widget.close()


def _restored_breakdown(qapp, tmp_path, kind: str, payload: dict) -> list[list[str]]:
    service = ProjectService()
    document = service.create_project(tmp_path / f"merged-{kind}.mufinder", "Merged Breakdown")
    service.add_structure(document, feo_host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state({"magnetic_field_results": {"kind": kind, "results": [payload]}})
    widget._result_table.selectRow(0)
    rows = _breakdown_rows(widget)
    widget.close()
    return rows


@pytest.mark.parametrize("modulated", [True, False], ids=["modulated", "uniform"])
def test_magnetic_field_widget_merges_dipolar_and_lorentz_for_a_modulated_field(
    qapp, tmp_path, modulated
) -> None:
    result = magnetic_fields_module.MagneticFieldResult(
        site_label="muon 1",
        muon_position=(0.5, 0.5, 0.5),
        dipolar=(0.0, 0.0, -0.3),
        lorentz=(0.0, 0.0, 0.1),
        contact=(0.0, 0.0, 0.0),
        demag=(0.0, 0.0, 0.0),
        applied=(0.0, 0.0, 0.0),
        calculation_parameters={"modulated": modulated},
    )

    rows = _restored_breakdown(qapp, tmp_path, "commensurate", result.to_json())

    labels = [row[0] for row in rows]
    if modulated:
        assert rows[0] == ["Dipolar (incl. Lorentz)", "0", "0", "-0.2", "0.2"]
        assert "Lorentz" not in labels
    else:
        assert labels[:2] == ["Dipolar", "Lorentz"]
    assert rows[-1] == ["Total", "0", "0", "-0.2", "0.2"]


def test_magnetic_field_widget_leaves_out_a_merged_row_without_its_lorentz_term(
    qapp, tmp_path
) -> None:
    payload = {
        "site_label": "muon 1",
        "dipolar_T": [0.0, 0.0, -0.3],
        "contact_T": [0.0, 0.0, 0.1],
        "total_T": [0.0, 0.0, -0.2],
        "calculation_parameters": {"modulated": True},
    }

    rows = _restored_breakdown(qapp, tmp_path, "commensurate", payload)

    assert [row[0] for row in rows] == ["Contact", "Total"]


@pytest.mark.parametrize("modulated", [True, False], ids=["modulated", "unflagged"])
def test_magnetic_field_widget_merges_distribution_dipolar_and_lorentz_rows(
    qapp, tmp_path, modulated
) -> None:
    stats = magnetic_fields_module.FieldTermStatistics
    distribution = magnetic_fields_module.MagneticFieldDistributionResult(
        site_label="site 1",
        muon_position=(0.5, 0.5, 0.5),
        fields_T=np.asarray([[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]], dtype=float),
        histogram_centres_T=np.asarray([0.2], dtype=float),
        histogram_counts=np.asarray([2.0], dtype=float),
        calculation_parameters={"modulated": modulated},
        terms={
            "dipolar": stats(mean_T=(0.0, 0.0, 0.0), rms_T=0.3),
            "lorentz": stats(mean_T=(0.0, 0.0, 0.0), rms_T=0.1),
            "contact": stats(mean_T=(0.0, 0.0, 0.0), rms_T=0.0),
            "demag": stats(mean_T=(0.0, 0.0, 0.0), rms_T=0.0),
            "applied": stats(mean_T=(0.0, 0.0, 0.0), rms_T=0.0),
        },
        total_rms_T=0.2,
        applied_T=(0.0, 0.0, 0.0),
        demag_T=(0.0, 0.0, 0.0),
        dipolar_lorentz=stats(mean_T=(0.0, 0.0, 0.0), rms_T=0.2),
    )

    rows = _restored_breakdown(qapp, tmp_path, "distribution", distribution.to_json())

    if modulated:
        assert rows[0] == ["Dipolar (incl. Lorentz)", "0", "0", "0", "0.2"]
        assert [row[0] for row in rows][1:] == [
            "Contact",
            "Demagnetizing",
            "Applied",
            "Total",
        ]
    else:
        assert [row[0] for row in rows][:2] == ["Dipolar", "Lorentz"]


_APPROXIMANT = magnetic_fields_module.CommensurateApproximant(
    q_rlu=((0.3334, 0.0, 0.0),),
    approximant_rlu=((1.0 / 3.0, 0.0, 0.0),),
    denominators=((3, 1, 1),),
    periods=(3, 1, 1),
    max_deviation_rlu=6.7e-5,
)


def _approximant_field(label: str, bz: float) -> magnetic_fields_module.MagneticFieldResult:
    zero = (0.0, 0.0, 0.0)
    return magnetic_fields_module.MagneticFieldResult(
        site_label=label,
        muon_position=zero,
        dipolar=(0.0, 0.0, bz),
        lorentz=zero,
        contact=zero,
        demag=zero,
        applied=zero,
    )


def _incommensurate_widget(document, monkeypatch, *, approximant) -> MagneticFieldWidget:
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "incommensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "find_commensurate_approximant",
        lambda *_args, **_kwargs: approximant,
    )
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    return widget


def test_magnetic_field_widget_lists_commensurate_approximant_fields(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-approximant.mufinder", "Magnetic Approximant"
    )
    service.add_structure(document, feo_host_structure())
    calls: list[object] = []

    def _approximant_fields(*_args, options, distorted_structures=None):
        calls.append(options)
        return (
            _approximant_field("site 1 + (0, 0, 0)", 1.0),
            _approximant_field("site 1 + (1, 0, 0)", -0.5),
        )

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_commensurate_approximant_fields",
        _approximant_fields,
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: pytest.fail("an approximant should not sample a distribution"),
    )
    widget = _incommensurate_widget(document, monkeypatch, approximant=_APPROXIMANT)
    widget.set_calculation_settings(
        replace(
            widget.calculation_settings(),
            use_approximants=True,
            approximant_tolerance=2e-3,
            approximant_max_period=8,
        )
    )

    widget.calculate_fields()

    assert calls[0].approximant_tolerance == 2e-3
    assert calls[0].approximant_max_period == 8
    assert [widget._result_table.item(row, 0).text() for row in range(2)] == [
        "site 1 + (0, 0, 0)",
        "site 1 + (1, 0, 0)",
    ]
    assert widget._result_table.item(1, 5).text() == "field"
    assert widget._calculation_status.text() == (
        "Results: calculated 2 field values of the commensurate approximant q = (1/3, 0, 0) with a "
        "3 x 1 x 1 magnetic supercell (q moved by at most 6.7e-05 r.l.u.)."
    )
    state = widget.collect_state()["magnetic_field_results"]
    assert state["kind"] == "commensurate"
    assert state["results"][1]["dipolar_T"] == [0.0, 0.0, -0.5]
    assert not widget._breakdown_table.isHidden()
    widget.close()


def test_magnetic_field_widget_suggests_an_available_approximant(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-suggest.mufinder", "Magnetic Suggest")
    service.add_structure(document, feo_host_structure())
    flagged = magnetic_fields_module.DistributionConvergenceCheck(
        setting="nangles",
        label="Lookup angles per q",
        converged=False,
        metric=0.2,
        message="Halving the lookup angles to 8 shifts the |B| distribution by 20%.",
    )

    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        convergence = (flagged,)

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_commensurate_approximant_fields",
        lambda *_args, **_kwargs: pytest.fail("approximants are turned off"),
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: (_Distribution(),),
    )
    widget = _incommensurate_widget(document, monkeypatch, approximant=_APPROXIMANT)

    widget.calculate_fields()

    status = widget._calculation_status.text()
    assert status.startswith(
        "Results: sampled 1 incommensurate field distribution; "
        "1 convergence check needs attention, select a row for advice."
    )
    assert "within 6.7e-05 r.l.u. of the commensurate approximant q = (1/3, 0, 0)" in status
    assert status.endswith(
        "turn on commensurate approximants in Calculation Settings to list its finite fields."
    )
    widget.close()


def test_magnetic_field_widget_reports_a_converged_distribution_without_advice(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-converged.mufinder", "Magnetic Converged"
    )
    service.add_structure(document, feo_host_structure())

    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        convergence = ()

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: (_Distribution(),),
    )
    widget = _incommensurate_widget(document, monkeypatch, approximant=None)
    widget.set_calculation_settings(replace(widget.calculation_settings(), use_approximants=True))

    widget.calculate_fields()

    assert (
        widget._calculation_status.text() == "Results: sampled 1 incommensurate field distribution."
    )
    widget.close()


def test_magnetic_field_widget_shows_distribution_convergence_checks(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-convergence.mufinder", "Magnetic Convergence"
    )
    service.add_structure(document, feo_host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)

    widget.set_form_state(
        {
            "magnetic_field_results": {
                "kind": "distribution",
                "results": [
                    {
                        "site_label": "site 1",
                        "fields_T": [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]],
                        "convergence": [
                            {
                                "setting": "nangles",
                                "label": "Lookup angles per q",
                                "converged": True,
                                "message": "Halving the lookup angles to 16384 shifts the |B| distribution by 0.05%.",
                            },
                            {
                                "setting": "histogram_bins",
                                "label": "Histogram bins",
                                "converged": False,
                                "message": "8 samples per occupied histogram bin (fewer than 20).",
                            },
                            "not a check",
                        ],
                    }
                ],
            }
        }
    )

    hint = widget._breakdown_hint.text().splitlines()
    assert hint[:3] == [
        "Convergence checks:",
        "Lookup angles per q: OK. Halving the lookup angles to 16384 shifts the |B| distribution by 0.05%.",
        "Histogram bins: Needs attention. 8 samples per occupied histogram bin (fewer than 20).",
    ]
    assert hint[3] == "This saved distribution has no term breakdown; recalculate to see it."
    widget.close()


def test_magnetic_field_widget_saves_and_restores_approximant_settings(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-approximant-state.mufinder", "Approximant State"
    )
    service.add_structure(document, feo_host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)

    widget.set_form_state(
        {
            "use_commensurate_approximants": True,
            "approximant_tolerance": 0.004,
            "approximant_max_period": 20,
        }
    )

    state = widget.collect_state()
    assert state["use_commensurate_approximants"] is True
    assert state["approximant_tolerance"] == 0.004
    assert state["approximant_max_period"] == 20
    assert "commensurate approximants within 0.004 r.l.u., period up to 20." in (
        widget._calculation_settings_summary.text()
    )

    widget.set_form_state({"approximant_tolerance": "wide", "approximant_max_period": 500})
    assert widget.calculation_settings().use_approximants is False
    assert widget.field_options().approximant_tolerance == 1e-3
    assert widget.field_options().approximant_max_period == 12

    widget.set_form_state({"approximant_tolerance": -1.0, "approximant_max_period": "long"})
    assert widget.field_options().approximant_tolerance == 1e-3
    assert widget.field_options().approximant_max_period == 12
    assert widget._calculation_settings_summary.text().endswith("512 histogram bins.")
    widget.close()


def test_magnetic_field_widget_opens_incommensurate_distribution_plot(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-dist-plot.mufinder", "Magnetic Dist Plot"
    )
    service.add_structure(document, feo_host_structure())

    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        histogram_centres_T = np.asarray([1.0, 3.0], dtype=float)
        histogram_counts = np.asarray([1, 1], dtype=float)

    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "incommensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: (_Distribution(),),
    )

    opened: list[object] = []

    class _Dialog:
        def __init__(self, results, parent=None, *, estimate=False):
            opened.append(tuple(results))

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(magnetic_field_widget_module, "MagneticFieldDistributionDialog", _Dialog)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))

    widget.calculate_fields()
    widget._open_distribution_plot()

    assert opened == [(widget._distribution_results[0],)]
    widget.close()


def test_magnetic_field_widget_opens_selected_distribution_plots_together(
    qapp, tmp_path, monkeypatch
) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-multi-dist-plot.mufinder", "Magnetic Multi Dist Plot"
    )
    service.add_structure(document, feo_host_structure())

    class _Distribution:
        def __init__(self, label: str, z_values: list[float]) -> None:
            self.site_label = label
            self.fields_T = np.asarray([[0.0, 0.0, value] for value in z_values], dtype=float)
            self.histogram_centres_T = np.asarray(z_values, dtype=float)
            self.histogram_counts = np.ones(len(z_values), dtype=float)

    distributions = (
        _Distribution("site 1", [1.0, 2.0]),
        _Distribution("site 2", [1.5, 2.5]),
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "classify_magnetic_model",
        lambda *_args, **_kwargs: "incommensurate",
    )
    monkeypatch.setattr(
        magnetic_field_calculation_module,
        "calculate_incommensurate_field_distributions",
        lambda *_args, **_kwargs: distributions,
    )

    opened: list[tuple[object, ...]] = []

    class _Dialog:
        def __init__(self, results, parent=None, *, estimate=False):
            opened.append(tuple(results))

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(magnetic_field_widget_module, "MagneticFieldDistributionDialog", _Dialog)

    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    widget.calculate_fields()
    widget._result_table.selectRow(0)
    widget._result_table.selectionModel().select(
        widget._result_table.model().index(1, 0),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )

    widget._open_distribution_plot()

    assert widget._result_table.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert opened == [distributions]
    widget.close()


def test_magnetic_field_widget_plots_a_restored_distribution(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-restored-plot.mufinder", "Magnetic Restored Plot"
    )
    service.add_structure(document, feo_host_structure())
    opened: list[tuple[object, ...]] = []

    class _Dialog:
        def __init__(self, results, parent=None, *, estimate=False):
            opened.append(tuple(results))

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(magnetic_field_widget_module, "MagneticFieldDistributionDialog", _Dialog)
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_form_state(
        {
            "magnetic_field_results": {
                "kind": "distribution",
                "results": [
                    {
                        "site_label": "site 1",
                        "muon_position": [0.0, 0.0, 0.0],
                        "fields_T": [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]],
                        "histogram_centres_T": [1.0, 3.0],
                        "histogram_counts": [4.0, 2.0],
                    },
                    {"site_label": "malformed", "fields_T": [[1.0, 2.0]]},
                ],
            }
        }
    )
    widget._result_table.selectRow(0)

    widget._open_distribution_plot()

    (restored,) = opened[0]
    monkeypatch.undo()
    series = MagneticFieldDistributionDialog(restored)._distribution_series()
    assert [label for *_values, label in series] == ["site 1"]
    np.testing.assert_allclose(series[0][0], [1.0, 3.0])
    np.testing.assert_allclose(series[0][2], [1.0, 3.0])
    assert widget._result_table.rowCount() == 2
    widget.close()


def test_magnetic_field_widget_summarises_model_and_emits_changes(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-summary.mufinder", "Magnetic Summary")
    service.add_structure(document, feo_host_structure())

    widget = MagneticFieldWidget()
    emitted: list[object] = []
    widget.magnetic_structure_changed.connect(emitted.append)
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))

    assert widget._structure_summary.text() == "Magnetic structure: 1 component"
    assert widget._classification.text() == "Structure type: commensurate"
    assert emitted
    assert isinstance(emitted[-1], dict)
    widget.close()


def _b2_document(tmp_path):
    service = ProjectService()
    document = service.create_project(tmp_path / "b2-tie.mufinder", "B2 tie")
    service.add_structure(
        document,
        host_structure(
            label="B2",
            lattice=2.868,
            species=("Fe", "Mn"),
            positions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
        ),
    )
    return document


def _b2_widget(tmp_path, *, coupling: str = "-0.5") -> MagneticFieldWidget:
    widget = MagneticFieldWidget()
    widget.set_document(_b2_document(tmp_path))
    widget.set_model(
        MagneticStructureModel(
            components=(
                build_collinear_component(
                    label="Fe",
                    selector=AtomSelector.species("Fe"),
                    q_coeff=(0, 0, 0),
                    moment=(0.0, 0.0, 2.0),
                ),
                build_collinear_component(
                    label="Mn",
                    selector=AtomSelector.species("Mn"),
                    q_coeff=(0, 0, 0),
                    moment=(0.0, 0.0, -2.0),
                ),
            )
        )
    )
    widget.set_form_state(
        {
            "selected_muon_site": {
                "source": "custom",
                "label": "Tetrahedral",
                "fractional_position": [0.5, 0.25, 0.0],
            },
            "hyperfine_coupling": coupling,
            "hyperfine_coupling_unit": "T_per_muB",
        }
    )
    return widget


def test_calculation_settings_dialog_warns_while_the_contact_shell_splits_a_tie(qapp) -> None:
    calls: list[tuple[int, float]] = []

    def _check(nnn: int, rcont: float) -> tuple[str, ...]:
        calls.append((nnn, rcont))
        return ("First site splits a tie.", "Second site too.") if nnn < 4 else ()

    dialog = MagneticFieldCalculationSettingsDialog(
        contact_tie_check=_check,
    )
    warning = dialog._contact_tie_warning

    assert not warning.isHidden()
    assert warning.text() == "First site splits a tie. The same holds at 1 more site."
    dialog._contact_nnn.setValue(4)
    assert warning.isHidden()
    dialog._contact_rcont.setValue(3.5)
    dialog._contact_nnn.setValue(3)
    assert not warning.isHidden()
    assert calls[-1] == (3, 3.5)
    # Warning only: the values still go through.
    assert dialog.settings().contact_nnn == 3
    dialog.close()


def test_calculation_settings_dialog_without_a_tie_check_shows_no_warning(qapp) -> None:
    dialog = MagneticFieldCalculationSettingsDialog()

    assert dialog._contact_tie_warning.isHidden()
    dialog.close()


def test_the_page_checks_the_selected_site_contact_shell_only_with_a_coupling(
    qapp, tmp_path
) -> None:
    widget = _b2_widget(tmp_path)

    check = widget._contact_tie_checker()

    (warning,) = check(2, 10.0)
    assert "keeps 2 of the 4 magnetic ions" in warning
    assert "(2 × Fe 1, 2 × Mn 2)" in warning
    assert check(4, 10.0) == ()
    widget._include_equivalent_sites.setChecked(True)
    assert len(widget._contact_tie_checker()(2, 10.0)) == 12
    widget._hyperfine_coupling.setText("")
    assert widget._contact_tie_checker() is None
    widget.close()


def test_the_settings_dialog_opened_from_the_page_warns_of_its_site(
    qapp, tmp_path, monkeypatch
) -> None:
    shown: list[str] = []

    def _exec(dialog) -> int:
        shown.append(
            "" if dialog._contact_tie_warning.isHidden() else dialog._contact_tie_warning.text()
        )
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(MagneticFieldCalculationSettingsDialog, "exec", _exec)
    widget = _b2_widget(tmp_path)

    widget._open_calculation_settings_dialog()

    (text,) = shown
    assert text.startswith("Contact neighbours (nnn) = 2 keeps 2 of the 4 magnetic ions")
    widget.close()


@pytest.mark.parametrize(("nnn", "warned"), [(2, True), (4, False)])
def test_the_breakdown_warns_of_a_result_whose_contact_shell_splits_a_tie(
    qapp, tmp_path, monkeypatch, nnn, warned
) -> None:
    _fake_muesr_without_lattice_field(monkeypatch)
    widget = _b2_widget(tmp_path)
    widget.set_form_state({**widget.collect_state(), "contact_nnn": nnn})

    widget.calculate_fields()
    widget._result_table.selectRow(0)

    warning = widget._breakdown_contact_warning
    assert warning.isHidden() is not warned
    if warned:
        assert warning.text().startswith(
            "Contact shell: Contact neighbours (nnn) = 2 keeps 2 of the 4 magnetic ions"
        )
    widget._clear_results(clear_state=True)
    assert warning.isHidden()
    widget.close()


def test_the_breakdown_shows_a_saved_tie_warning_without_its_terms(qapp, tmp_path) -> None:
    widget = _b2_widget(tmp_path)
    state = widget.collect_state()
    state["magnetic_field_results"] = {
        "kind": "commensurate",
        "site_source": "custom",
        "results": [
            {
                "site_label": "site 1",
                "muon_position": [0.5, 0.25, 0.0],
                "total_T": [0.0, 0.0, -1.0],
                "magnitude_T": 1.0,
                "calculation_parameters": {"contact_shell_warnings": ["A tie."]},
            }
        ],
    }
    widget.set_form_state(state)
    widget._result_table.selectRow(0)

    assert "recalculate" in widget._breakdown_hint.text()
    assert not widget._breakdown_contact_warning.isHidden()
    assert widget._breakdown_contact_warning.text() == "Contact shell: A tie."
    widget.close()


def test_the_contact_settings_check_ties_when_a_typed_value_is_finished(qapp) -> None:
    calls: list[tuple[int, float]] = []
    dialog = MagneticFieldCalculationSettingsDialog(
        contact_tie_check=lambda nnn, rcont: calls.append((nnn, rcont)) or (),
    )
    calls.clear()

    dialog._contact_rcont.lineEdit().selectAll()
    QTest.keyClicks(dialog._contact_rcont, "25.5")

    assert calls == []
    QTest.keyClick(dialog._contact_rcont, Qt.Key.Key_Return)
    assert calls == [(2, 25.5)]
    dialog.close()


class _StubField:
    def __init__(self, label: str, bz: float) -> None:
        self.site_label = label
        self.muon_position = (0.25, 0.5, 0.75)
        self.total = (0.0, 0.0, bz)
        self.magnitude = abs(bz)
        self.distortion = None

    def to_json(self) -> dict[str, object]:
        return {
            "site_label": self.site_label,
            "muon_position": list(self.muon_position),
            "total_T": list(self.total),
            "magnitude_T": self.magnitude,
        }


def _stub_calculation(label: str, bz: float):
    return magnetic_field_calculation_module.MagneticFieldCalculation(
        magnetic_field_calculation_module.COMMENSURATE_RESULTS,
        (_StubField(label, bz),),
        f"Results: calculated {label}.",
    )


def _ready_widget(tmp_path) -> MagneticFieldWidget:
    service = ProjectService()
    document = service.create_project(tmp_path / "off-thread.mufinder", "Off Thread")
    service.add_structure(document, feo_host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(MagneticStructureModel(components=(z_moment_component("one", "One", 0),)))
    widget._selected_muon_site = custom_muon_site((0.25, 0.5, 0.75))
    widget._include_equivalent_sites.setChecked(False)
    widget._sync_enabled_state()
    return widget


def _busy_dialog() -> QProgressDialog | None:
    for candidate in QApplication.topLevelWidgets():
        if isinstance(candidate, QProgressDialog) and candidate.isVisible():
            return candidate
    return None


def test_the_page_keeps_its_event_loop_running_during_a_slow_calculation(
    qapp, tmp_path, monkeypatch
) -> None:
    widget = _ready_widget(tmp_path)
    threads: list[threading.Thread] = []
    during: list[tuple[bool, str]] = []

    def _slow(_request):
        threads.append(threading.current_thread())
        time.sleep(0.3)
        return _stub_calculation("the slow field", 0.5)

    monkeypatch.setattr(
        magnetic_field_widget_module, "run_prepared_magnetic_field_calculation", _slow
    )
    ticks: list[float] = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
    QTimer.singleShot(
        100,
        lambda: during.append(
            (widget._calculate_button.isEnabled(), widget._calculation_status.text())
        ),
    )
    timer.start()
    widget.calculate_fields()
    timer.stop()

    assert threads and threads[0] is not threading.main_thread()
    # The GUI thread kept serving timers while the worker calculated.
    assert len(ticks) >= 10
    assert during == [(False, "Results: calculating the fields at 1 muon site…")]
    assert widget._calculate_button.isEnabled()
    assert widget._calculation_status.text() == "Results: calculated the slow field."
    assert widget._result_table.item(0, 0).text() == "the slow field"
    widget.close()


def test_cancelling_a_calculation_keeps_the_earlier_results_and_saves_nothing(
    qapp, tmp_path, monkeypatch
) -> None:
    widget = _ready_widget(tmp_path)
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "run_prepared_magnetic_field_calculation",
        lambda _request: _stub_calculation("the first field", 0.25),
    )
    widget.calculate_fields()
    saved = widget.collect_state()["magnetic_field_results"]
    emitted: list[object] = []
    widget.form_state_changed.connect(lambda *args: emitted.append(args))

    release = threading.Event()
    seen: list[str] = []

    def _stuck(_request):
        release.wait(10)
        return _stub_calculation("the abandoned field", 9.0)

    def _cancel() -> None:
        dialog = _busy_dialog()
        if dialog is None:
            QTimer.singleShot(20, _cancel)
            return
        seen.append(dialog.labelText())
        dialog.cancel()

    monkeypatch.setattr(
        magnetic_field_widget_module, "run_prepared_magnetic_field_calculation", _stuck
    )
    QTimer.singleShot(0, _cancel)
    try:
        widget.calculate_fields()
    finally:
        release.set()

    assert seen == ["Calculating the fields at 1 muon site…"]
    assert widget._calculation_status.text() == (
        "Results: calculation cancelled; the earlier results are kept"
    )
    assert widget._result_table.rowCount() == 1
    assert widget._result_table.item(0, 0).text() == "the first field"
    assert widget.collect_state()["magnetic_field_results"] == saved
    assert emitted == []
    assert widget._calculate_button.isEnabled()
    widget.close()


def test_the_dialogs_calculate_off_the_gui_thread(qapp, tmp_path, monkeypatch) -> None:
    widget = _ready_widget(tmp_path)
    threads: list[threading.Thread] = []
    descriptions: list[str] = []
    real_run = magnetic_field_widget_module.run_blocking_call

    def _recording_run(parent, op, *, title, description, cancellable):
        descriptions.append(description)
        assert cancellable
        return real_run(parent, op, title=title, description=description, cancellable=True)

    def _tensors(_request):
        threads.append(threading.current_thread())
        return ("tensor",)

    def _widths(_request):
        threads.append(threading.current_thread())
        return "widths"

    monkeypatch.setattr(magnetic_field_widget_module, "run_blocking_call", _recording_run)
    monkeypatch.setattr(
        magnetic_field_widget_module, "run_prepared_dipolar_tensor_calculation", _tensors
    )
    monkeypatch.setattr(
        magnetic_field_widget_module, "run_prepared_nuclear_relaxation_calculation", _widths
    )

    class _TensorDialog:
        def selected_ion_indices(self):
            return [0]

        def include_equivalent_sites(self):
            return False

    class _NuclearDialog:
        def options(self):
            return NuclearRelaxationOptions()

    assert widget._calculate_dipolar_tensors_from_dialog(_TensorDialog()) == ("tensor",)
    assert widget._calculate_nuclear_relaxation_from_dialog(_NuclearDialog()) == "widths"
    assert len(threads) == 2
    assert all(thread is not threading.main_thread() for thread in threads)
    assert descriptions == [
        "Calculating the dipolar tensors at 1 muon site…",
        "Calculating the nuclear relaxation…",
    ]
    widget.close()


def test_a_calculation_that_ends_after_the_page_reloaded_is_not_shown(
    qapp, tmp_path, monkeypatch
) -> None:
    widget = _ready_widget(tmp_path)
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "run_prepared_magnetic_field_calculation",
        lambda _request: _stub_calculation("the first field", 0.25),
    )
    widget.calculate_fields()
    saved = widget.collect_state()

    def _reload() -> None:
        # As the project does when a job finishes while the worker runs.
        widget.set_document(widget._document)
        widget.set_form_state(saved)

    monkeypatch.setattr(
        magnetic_field_widget_module,
        "run_prepared_magnetic_field_calculation",
        lambda _request: _stub_calculation("the stale field", 9.0),
    )
    monkeypatch.setattr(
        magnetic_field_widget_module,
        "run_blocking_call",
        lambda _parent, op, **_kwargs: (_reload(), op())[1],
    )
    emitted: list[object] = []
    widget.form_state_changed.connect(lambda *args: emitted.append(args))
    widget.calculate_fields()

    assert widget._calculation_status.text() == (
        "Results: the page was reloaded during the calculation; calculate again"
    )
    assert widget._result_table.item(0, 0).text() == "the first field"
    assert emitted == []
    widget.close()


def test_magnetic_field_widget_shows_host_spin_components_as_moment(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "magnetic-moment.mufinder", "Magnetic Moment")
    service.add_structure(document, feo_host_structure())
    widget = MagneticFieldWidget()
    widget.set_document(document)
    widget.set_model(magnetic_model_from_host_spins(feo_host_structure()))

    summary = widget._component_summary_table
    assert summary.rowCount() == 2
    assert summary.item(0, 1).text() == "Moment"
    widget.close()
