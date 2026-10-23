"""Tests for the shared project-defaults strip and its details dialog."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDialog

from mufinder.app import project_defaults_widget as module
from mufinder.app.convergence_types import ConvergenceConfig, convergence_config_xc_functional
from mufinder.app.project_defaults_widget import (
    ProjectDefaultsDialog,
    ProjectDefaultsStrip,
    ProjectDefaultsSummary,
    xc_functional_choices,
)
from mufinder.domain.models import Project


def _summary(**overrides: object) -> ProjectDefaultsSummary:
    values: dict[str, object] = {
        "xc_functional": "PBE",
        "cutoff_energy": 700.0,
        "cutoff_method": "delta-threshold",
        "cutoff_energy_threshold": 0.001,
        "cutoff_force_threshold": 0.05,
        "cutoff_xc_functional": "PBE",
        "kpoint_grid": (4, 4, 4),
        "kpoint_method": "delta-threshold",
        "kpoint_energy_threshold": 0.001,
        "kpoint_force_threshold": 0.05,
        "kpoint_xc_functional": "PBE",
    }
    values.update(overrides)
    return ProjectDefaultsSummary(**values)  # type: ignore[arg-type]


def test_summary_from_project_copies_every_default() -> None:
    project = Project(
        recommended_xc_functional="PBE",
        recommended_cutoff_energy=700.0,
        recommended_cutoff_method="asymptotic",
        recommended_cutoff_stress_threshold=0.1,
        recommended_cutoff_xc_functional="PBE",
        recommended_kpoint_grid=(3, 3, 2),
        recommended_kpoint_xc_functional="LDA",
    )

    summary = ProjectDefaultsSummary.from_project(project)

    assert summary.xc_functional == "PBE"
    assert summary.cutoff_energy == pytest.approx(700.0)
    assert summary.cutoff_method == "asymptotic"
    assert summary.cutoff_stress_threshold == pytest.approx(0.1)
    assert summary.kpoint_grid == (3, 3, 2)
    assert summary.kpoint_xc_functional == "LDA"
    assert ProjectDefaultsSummary.from_project(None) == ProjectDefaultsSummary()


def test_summary_flags_defaults_converged_with_another_functional() -> None:
    summary = _summary(kpoint_xc_functional="LDA")

    assert not summary.cutoff_is_stale
    assert summary.kpoint_is_stale
    assert "converged with LDA" in summary.kpoint_detail()
    assert "LDA" in summary.stale_warning(summary.kpoint_xc_functional)


def test_summary_is_not_stale_without_both_functionals_or_a_value() -> None:
    # No recorded converged-with functional (legacy project): nothing to compare.
    assert not _summary(cutoff_xc_functional=None).cutoff_is_stale
    assert "functional not recorded" in _summary(cutoff_xc_functional=None).cutoff_detail()
    # No project functional chosen yet.
    assert not _summary(xc_functional=None).kpoint_is_stale
    # No default adopted at all.
    unset = _summary(kpoint_grid=None, kpoint_xc_functional="LDA")
    assert not unset.kpoint_is_stale
    assert "Adopt a grid" in unset.kpoint_detail()
    assert "Adopt a value" in _summary(cutoff_energy=None).cutoff_detail()


def test_strip_shows_values_and_flags_stale_defaults(qapp) -> None:
    strip = ProjectDefaultsStrip()

    assert strip._xc_value.text() == "Not set"
    assert strip._cutoff_value.property("valueState") == "unset"

    strip.set_defaults(_summary(kpoint_xc_functional="LDA", cutoff_stress_threshold=0.1))

    assert strip._xc_value.text() == "PBE"
    assert strip._cutoff_value.text() == "700 eV"
    assert strip._cutoff_value.property("valueState") == "set"
    assert "dS <= 0.1 GPa" in strip._cutoff_value.toolTip()
    assert strip._kpoint_value.text() == "4 x 4 x 4 (LDA)"
    assert strip._kpoint_value.property("valueState") == "stale"
    assert "Re-run the convergence test" in strip._kpoint_value.toolTip()


def test_dialog_offers_catalogued_functionals_and_updates_warnings(qapp) -> None:
    dialog = ProjectDefaultsDialog(_summary())
    choices = xc_functional_choices()

    assert "PBE" in choices
    assert dialog._xc_combo.count() == len(choices) + 1
    assert dialog.xc_functional() == "PBE"
    assert dialog._cutoff_warning.isHidden()

    dialog._xc_combo.setCurrentIndex(dialog._xc_combo.findData("LDA"))

    assert dialog.xc_functional() == "LDA"
    assert not dialog._cutoff_warning.isHidden()
    assert not dialog._kpoint_warning.isHidden()
    assert "project functional is LDA" in dialog._cutoff_warning.text()

    dialog._xc_combo.setCurrentIndex(0)
    assert dialog.xc_functional() is None
    assert dialog._cutoff_warning.isHidden()


def test_dialog_keeps_an_uncatalogued_current_functional(qapp) -> None:
    dialog = ProjectDefaultsDialog(_summary(xc_functional="LIBXC_CUSTOM"))

    assert dialog.xc_functional() == "LIBXC_CUSTOM"


@pytest.mark.parametrize(
    ("result", "selected", "expected"),
    [
        (QDialog.DialogCode.Accepted, "LDA", ["LDA"]),
        (QDialog.DialogCode.Accepted, None, [None]),
        (QDialog.DialogCode.Accepted, "PBE", []),
        (QDialog.DialogCode.Rejected, "LDA", []),
    ],
)
def test_strip_requests_functional_change_only_when_accepted_and_changed(
    qapp, monkeypatch, result, selected, expected
) -> None:
    strip = ProjectDefaultsStrip()
    strip.set_defaults(_summary())
    requested: list[object] = []
    strip.xc_functional_change_requested.connect(requested.append)
    monkeypatch.setattr(module.ProjectDefaultsDialog, "exec", lambda self: result)
    monkeypatch.setattr(module.ProjectDefaultsDialog, "xc_functional", lambda self: selected)

    strip._details_button.click()

    assert requested == expected


def test_convergence_config_xc_functional_requires_a_single_functional() -> None:
    shared = ConvergenceConfig(sweep_values=(300.0, 400.0), shared_params={"xc_functional": "pbe"})
    assert convergence_config_xc_functional(shared) == "PBE"

    mixed = ConvergenceConfig(
        sweep_values=(300.0, 400.0),
        shared_params={"xc_functional": "PBE"},
        overrides={frozenset({400.0}): {"xc_functional": "LDA"}},
    )
    assert convergence_config_xc_functional(mixed) is None

    # Every point overridden to the same functional is still unambiguous.
    uniform = ConvergenceConfig(
        sweep_values=(300.0, 400.0),
        overrides={frozenset({300.0, 400.0}): {"xc_functional": "PBESOL"}},
    )
    assert convergence_config_xc_functional(uniform) == "PBESOL"

    assert convergence_config_xc_functional(ConvergenceConfig(sweep_values=(300.0,))) is None
    assert convergence_config_xc_functional(ConvergenceConfig()) is None


def test_parse_cutoff_energy_text_accepts_numbers_and_blank() -> None:
    assert module.parse_cutoff_energy_text("  ") is None
    assert module.parse_cutoff_energy_text("650") == pytest.approx(650.0)
    assert module.parse_cutoff_energy_text("7e2 eV") == pytest.approx(700.0)
    for bad in ("abc", "0", "-5", "nan", "inf"):
        with pytest.raises(ValueError):
            module.parse_cutoff_energy_text(bad)


def test_parse_kpoint_grid_text_accepts_common_separators_and_blank() -> None:
    assert module.parse_kpoint_grid_text("") is None
    assert module.parse_kpoint_grid_text("4 4 4") == (4, 4, 4)
    assert module.parse_kpoint_grid_text("4 x 3 x 2") == (4, 3, 2)
    assert module.parse_kpoint_grid_text("6x6x4") == (6, 6, 4)
    assert module.parse_kpoint_grid_text("2, 2, 1") == (2, 2, 1)
    for bad in ("4 4", "4 4 4 4", "4 0 4", "a b c", "4.5 4 4", "-4 4 4"):
        with pytest.raises(ValueError):
            module.parse_kpoint_grid_text(bad)


def test_dialog_prefills_values_and_marks_edits_as_manual(qapp) -> None:
    dialog = ProjectDefaultsDialog(_summary())

    assert dialog.cutoff_energy() == pytest.approx(700.0)
    assert dialog.kpoint_grid() == (4, 4, 4)
    assert dialog.edited_summary() == _summary()
    assert "Delta-Threshold" in dialog._cutoff_detail.text()

    dialog._cutoff_value.setText("850")
    dialog._kpoint_value.setText("6 6 4")

    edited = dialog.edited_summary()
    assert edited.cutoff_energy == pytest.approx(850.0)
    assert edited.cutoff_method == "manual"
    assert edited.cutoff_energy_threshold is None
    assert edited.cutoff_xc_functional is None
    assert edited.kpoint_grid == (6, 6, 4)
    assert edited.kpoint_method == "manual"
    assert dialog._cutoff_detail.text() == "Set manually, not from a convergence test."
    assert dialog._kpoint_detail.text() == "Set manually, not from a convergence test."

    dialog._cutoff_value.clear()
    assert dialog.edited_summary().cutoff_energy is None
    assert dialog.edited_summary().cutoff_method is None
    assert "Adopt a value" in dialog._cutoff_detail.text()


def test_dialog_manual_edit_clears_a_stale_warning(qapp) -> None:
    dialog = ProjectDefaultsDialog(_summary(kpoint_xc_functional="LDA"))
    assert not dialog._kpoint_warning.isHidden()

    dialog._kpoint_value.setText("5 5 5")

    assert dialog._kpoint_warning.isHidden()


def test_dialog_rejects_invalid_values_with_an_inline_error(qapp) -> None:
    dialog = ProjectDefaultsDialog(_summary())
    dialog._kpoint_value.setText("4 4")

    dialog._accept_if_valid()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert not dialog._error.isHidden()
    assert "three positive integers" in dialog._error.text()

    dialog._kpoint_value.setText("4 4 2")
    assert dialog._error.isHidden()
    dialog._accept_if_valid()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_strip_requests_cutoff_and_grid_changes(qapp, monkeypatch) -> None:
    strip = ProjectDefaultsStrip()
    strip.set_defaults(_summary())
    cutoffs: list[object] = []
    grids: list[object] = []
    functionals: list[object] = []
    strip.cutoff_energy_change_requested.connect(cutoffs.append)
    strip.kpoint_grid_change_requested.connect(grids.append)
    strip.xc_functional_change_requested.connect(functionals.append)

    def fake_exec(dialog: ProjectDefaultsDialog) -> QDialog.DialogCode:
        dialog._cutoff_value.setText("800")
        dialog._kpoint_value.clear()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(module.ProjectDefaultsDialog, "exec", fake_exec)

    strip._details_button.click()

    assert cutoffs == [pytest.approx(800.0)]
    assert grids == [None]
    assert functionals == []
