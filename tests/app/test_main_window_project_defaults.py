"""Main-window behaviour for the project default XC functional."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QMessageBox

import mufinder.app.main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.domain.models import Job, Structure
from mufinder.services.project_service import ProjectDocument, ProjectService


def _window_with_cutoff_sweep(
    tmp_path: Path,
    *,
    shared_params: dict[str, object],
    overrides: list[dict[str, object]] | None = None,
) -> tuple[MainWindow, ProjectDocument]:
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "xc_defaults.mufinder", "XCDefaults")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    definition = service.register_convergence_cutoff_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "sweep_parameter": "cut_off_energy",
            "sweep_values": [300.0, 400.0],
            "cutoff_energies": [300.0, 400.0],
            "shared_params": shared_params,
            "overrides": overrides or [],
        },
        output_directory=doc.project_root / "files" / "convergence_cutoff",
    )
    doc.ui_state[window._ACTIVE_CONVERGENCE_DEFINITION_KEY] = str(definition.id)
    window._document = doc
    window._refresh_ui_after_project_change()
    return window, doc


def test_adopting_cutoff_records_the_sweep_functional_as_project_default(
    qapp, tmp_path: Path
) -> None:
    window, doc = _window_with_cutoff_sweep(
        tmp_path, shared_params={"task": "SINGLEPOINT", "xc_functional": "PBE"}
    )
    for strip in window._project_defaults_strips():
        assert strip.isEnabled()
        assert strip.summary().xc_functional is None

    window._on_cutoff_adopted(400.0, "delta-threshold", 0.001, 0.05)

    assert doc.project.recommended_cutoff_xc_functional == "PBE"
    assert doc.project.recommended_xc_functional == "PBE"
    for strip in window._project_defaults_strips():
        assert strip._xc_value.text() == "PBE"
        assert strip._cutoff_value.text() == "400 eV"
    # Every workflow now defaults new rows to the project functional.
    assert window._kpoint_convergence_view._default_params["xc_functional"] == "PBE"
    assert window._pristine_relaxation_view._default_params["xc_functional"] == "PBE"
    assert window._muon_relaxation_view._default_params["xc_functional"] == "PBE"
    assert window._uep_view._default_params["xc_functional"] == "PBE"
    assert window._project_convergence_parameter_values()["xc_functional"] == "PBE"
    assert ("Functional", "PBE") in window._cutoff_overview_summary_rows()
    window.close()


def test_adopting_a_mixed_functional_sweep_leaves_the_project_functional_alone(
    qapp, tmp_path: Path
) -> None:
    window, doc = _window_with_cutoff_sweep(
        tmp_path,
        shared_params={"task": "SINGLEPOINT", "xc_functional": "PBE"},
        overrides=[{"cutoffs": [400.0], "params": {"xc_functional": "LDA"}}],
    )
    assert window._convergence_definition_xc_functional(
        window._active_convergence_definition()
    ) is None

    window._on_cutoff_adopted(400.0, "delta-threshold", 0.001, 0.05)

    assert doc.project.recommended_cutoff_energy == pytest.approx(400.0)
    assert doc.project.recommended_cutoff_xc_functional is None
    assert doc.project.recommended_xc_functional is None
    assert ("Functional", "Not recorded") in window._cutoff_overview_summary_rows()
    window.close()


def test_adopting_with_a_different_functional_asks_before_switching(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, doc = _window_with_cutoff_sweep(
        tmp_path, shared_params={"task": "SINGLEPOINT", "xc_functional": "PBE"}
    )
    window._on_project_xc_functional_change_requested("LDA")
    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    prompts: list[str] = []

    def fake_question(_parent, title, text, *args, **kwargs):
        prompts.append(f"{title}: {text}")
        return answers.pop(0)

    monkeypatch.setattr(main_window_module.QMessageBox, "question", fake_question)

    window._on_cutoff_adopted(400.0, "delta-threshold", 0.001, 0.05)
    assert doc.project.recommended_cutoff_energy is None
    assert doc.project.recommended_xc_functional == "LDA"
    assert "PBE" in prompts[0] and "LDA" in prompts[0]

    window._on_cutoff_adopted(400.0, "delta-threshold", 0.001, 0.05)
    assert doc.project.recommended_cutoff_energy == pytest.approx(400.0)
    assert doc.project.recommended_xc_functional == "PBE"
    assert len(prompts) == 2
    window.close()


def test_changing_the_project_functional_flags_stale_defaults_and_persists(
    qapp, tmp_path: Path
) -> None:
    window, doc = _window_with_cutoff_sweep(
        tmp_path, shared_params={"task": "SINGLEPOINT", "xc_functional": "PBE"}
    )
    window._on_cutoff_adopted(400.0, "delta-threshold", 0.001, 0.05)
    provenance_count = len(doc.provenance)

    # The strip on any workflow tab drives the same project-wide change.
    window._uep_view.project_defaults_strip.xc_functional_change_requested.emit("LDA")

    assert doc.project.recommended_xc_functional == "LDA"
    assert doc.project.recommended_cutoff_xc_functional == "PBE"
    assert doc.provenance[-1].action == "set_recommended_xc_functional"
    for strip in window._project_defaults_strips():
        assert strip._cutoff_value.text() == "400 eV (PBE)"
        assert strip._cutoff_value.property("valueState") == "stale"
    assert window._convergence_view._default_params["xc_functional"] == "LDA"
    assert window._pristine_relaxation_view._default_params["xc_functional"] == "LDA"
    assert "LDA" in window.statusBar().currentMessage()
    loaded = ProjectService().load_project(doc.project_root)
    assert loaded.project.recommended_xc_functional == "LDA"

    # Re-selecting the current functional is a no-op.
    window._on_project_xc_functional_change_requested("LDA")
    assert len(doc.provenance) == provenance_count + 1

    window._on_project_xc_functional_change_requested(None)
    assert doc.project.recommended_xc_functional is None
    assert "cleared" in window.statusBar().currentMessage()
    assert "xc_functional" not in window._project_convergence_parameter_values()
    window.close()


def test_project_defaults_strips_are_disabled_without_a_project(qapp) -> None:
    window = MainWindow()

    window._on_project_xc_functional_change_requested("PBE")
    window._refresh_project_defaults_views()

    for strip in window._project_defaults_strips():
        assert not strip.isEnabled()
        assert strip._xc_value.text() == "Not set"
    assert ("Functional", "Not recorded") in window._cutoff_overview_summary_rows()
    assert ("Functional", "Not recorded") in window._kpoint_overview_summary_rows()
    window.close()


def test_follow_on_rows_inherit_the_source_relaxation_functional(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    job = Job(label="relax_site_1")
    output_dir = tmp_path / job.label
    output_dir.mkdir()
    (output_dir / f"{job.label}.param").write_text(
        "TASK : GEOMETRYOPTIMIZATION\nXC_FUNCTIONAL : PBESOL\nCUT_OFF_ENERGY : 650 eV\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(window, "_job_output_directory", lambda _job: output_dir)
    candidate = SimpleNamespace(job=job)

    values = window._quantum_source_default_parameter_values(candidate)

    assert values["xc_functional"] == "PBESOL"
    assert values["cut_off_energy"].value == pytest.approx(650.0)
    window.close()


_SPIN_SOURCE_CELL = (
    "%BLOCK LATTICE_CART\n2.87 0 0\n0 2.87 0\n0 0 2.87\n%ENDBLOCK LATTICE_CART\n"
    "%BLOCK POSITIONS_FRAC\nFe 0 0 0{spin}\nFe 0.5 0.5 0.5{spin}\n%ENDBLOCK POSITIONS_FRAC\n"
)


@pytest.mark.parametrize(
    ("param_lines", "cell_spin", "expected"),
    [
        pytest.param("SPIN_POLARISED : TRUE\n", "", True, id="explicit-english-spelling"),
        pytest.param("SPIN_POLARIZED : FALSE\n", " SPIN=2", False, id="explicit-false-wins"),
        pytest.param("", " SPIN=2", True, id="implied-by-atomic-spins"),
        pytest.param("SPIN : -2\n", "", True, id="implied-by-net-spin"),
        pytest.param("", "", False, id="not-spin-polarised"),
    ],
)
def test_follow_on_rows_inherit_the_source_relaxation_spin_treatment(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    param_lines: str,
    cell_spin: str,
    expected: bool,
) -> None:
    window = MainWindow()
    job = Job(label="relax_site_1")
    output_dir = tmp_path / job.label
    output_dir.mkdir()
    (output_dir / f"{job.label}.param").write_text(
        "TASK : GEOMETRYOPTIMIZATION\n" + param_lines, encoding="utf-8"
    )
    (output_dir / f"{job.label}.cell").write_text(
        _SPIN_SOURCE_CELL.format(spin=cell_spin), encoding="utf-8"
    )
    monkeypatch.setattr(window, "_job_output_directory", lambda _job: output_dir)

    values = window._quantum_source_default_parameter_values(SimpleNamespace(job=job))

    assert values["spin_polarized"] is expected
    window.close()


@pytest.mark.parametrize(
    ("charge_state", "expected"),
    [
        (main_window_module.MUONIUM_CHARGE_STATE, {"spin_polarized": True}),
        (main_window_module.POSITIVE_MUON_CHARGE_STATE, {}),
    ],
)
def test_unreadable_source_param_defaults_only_muonium_to_spin_polarised(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    charge_state: str,
    expected: dict[str, object],
) -> None:
    window = MainWindow()
    job = Job(
        label="relax_site_1",
        calc_type=main_window_module.muon_relaxation_calc_type(charge_state),
    )
    output_dir = tmp_path / job.label
    output_dir.mkdir()
    monkeypatch.setattr(window, "_job_output_directory", lambda _job: output_dir)

    values = window._quantum_source_default_parameter_values(SimpleNamespace(job=job))

    assert values == expected
    window.close()


def test_manually_edited_cutoff_and_grid_replace_adopted_defaults(qapp, tmp_path: Path) -> None:
    window, doc = _window_with_cutoff_sweep(
        tmp_path, shared_params={"task": "SINGLEPOINT", "xc_functional": "PBE"}
    )
    window._on_cutoff_adopted(400.0, "delta-threshold", 0.001, 0.05)
    # Pinned overview plots as left behind by earlier adoptions.
    doc.ui_state[window._CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] = {
        "cutoff": {
            "host_structure_id": str(doc.project.host_structure_id),
            "default_value": 400.0,
        },
        "kpoint": {
            "host_structure_id": str(doc.project.host_structure_id),
            "default_value": [6, 6, 4],
        },
    }
    assert window._validated_cutoff_overview_snapshot() is not None
    strip = window._pristine_relaxation_view.project_defaults_strip

    strip.cutoff_energy_change_requested.emit(850.0)

    project = doc.project
    assert project.recommended_cutoff_energy == pytest.approx(850.0)
    assert project.recommended_cutoff_method == "manual"
    assert project.recommended_cutoff_energy_threshold is None
    assert project.recommended_cutoff_xc_functional is None
    assert project.recommended_cutoff_cell_id == str(project.host_structure_id)
    # Editing a value by hand never changes the project functional.
    assert project.recommended_xc_functional == "PBE"
    assert doc.ui_state["default_cut_off_energy"] == pytest.approx(850.0)
    assert doc.ui_state["default_cut_off_energy_method"] == "manual"
    assert "default_cut_off_energy_threshold" not in doc.ui_state
    assert strip._cutoff_value.text() == "850 eV"
    assert "Set manually" in strip._cutoff_value.toolTip()
    assert ("Method", "Set manually") in window._cutoff_overview_summary_rows()
    assert window._kpoint_convergence_view._default_params["cut_off_energy"].value == (
        pytest.approx(850.0)
    )
    assert window._project_convergence_parameter_values()["cut_off_energy"].value == (
        pytest.approx(850.0)
    )
    # The pinned overview plot belonged to the adopted value, so it no longer applies.
    assert window._validated_cutoff_overview_snapshot() is None
    assert "850" in window.statusBar().currentMessage()

    # Typing the adopted number back in is still a manual default: the plot
    # that documented the adopted value must not come back.
    strip.cutoff_energy_change_requested.emit(400.0)
    assert project.recommended_cutoff_method == "manual"
    assert window._validated_cutoff_overview_snapshot() is None
    assert "cutoff" not in doc.ui_state[window._CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY]
    strip.cutoff_energy_change_requested.emit(850.0)

    strip.kpoint_grid_change_requested.emit((6, 6, 4))

    assert project.recommended_kpoint_grid == (6, 6, 4)
    assert project.recommended_kpoint_method == "manual"
    assert project.recommended_kpoint_lattice == [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    assert window._project_convergence_parameter_values()["kpoint_mp_grid"] == (6, 6, 4)
    assert window._uep_view._default_params["kpoint_mp_grid"] == (6, 6, 4)
    assert "kpoint" not in doc.ui_state[window._CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY]

    loaded = ProjectService().load_project(doc.project_root)
    assert loaded.project.recommended_cutoff_method == "manual"
    assert loaded.project.recommended_kpoint_grid == (6, 6, 4)
    window.close()


def test_clearing_cutoff_and_grid_from_the_dialog(qapp, tmp_path: Path) -> None:
    window, doc = _window_with_cutoff_sweep(
        tmp_path, shared_params={"task": "SINGLEPOINT", "xc_functional": "PBE"}
    )
    window._on_project_cutoff_change_requested(700.0)
    window._on_project_kpoint_grid_change_requested((4, 4, 4))

    window._on_project_cutoff_change_requested(None)
    window._on_project_kpoint_grid_change_requested(None)

    assert doc.project.recommended_cutoff_energy is None
    assert doc.project.recommended_kpoint_grid is None
    assert "default_cut_off_energy" not in doc.ui_state
    for strip in window._project_defaults_strips():
        assert strip._cutoff_value.text() == "Not set"
        assert strip._kpoint_value.text() == "Not set"
    values = window._project_convergence_parameter_values()
    assert "cut_off_energy" not in values and "kpoint_mp_grid" not in values
    assert "cut_off_energy" not in window._pristine_relaxation_view._default_params
    assert "cleared" in window.statusBar().currentMessage()
    window.close()


def test_project_default_edits_are_ignored_without_a_project(qapp) -> None:
    window = MainWindow()

    window._on_project_cutoff_change_requested(700.0)
    window._on_project_kpoint_grid_change_requested((4, 4, 4))

    assert window._document is None
    window.close()
