"""Tests for the completed phonon analysis dialog."""

from __future__ import annotations

from pathlib import Path
from math import pi

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGroupBox, QLabel, QTabWidget, QToolButton

import mufinder.app.quantum_analysis_dialog as quantum_analysis_dialog_module
from mufinder.app.math_label import MathLabel
from mufinder.app.quantum_analysis_dialog import QuantumAnalysisHelpDialog, QuantumPhononAnalysisDialog
from mufinder.app.screen_utils import clamped_size
from mufinder.app.theme import current_palette


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


def test_quantum_analysis_dialog_tables_use_splitters_not_width_locks(qapp, tmp_path: Path) -> None:
    """Tables are capped, not width-locked, so a QSplitter governs sizing.

    Guards the SC-5 small-screen fix: narrowing the dialog should let a
    QSplitter hand space to the plot/viewer instead of a rigid table width
    starving them.
    """
    from PySide6.QtWidgets import QSplitter

    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(PHONON_TEXT, encoding="utf-8")
    dialog = QuantumPhononAnalysisDialog(phonon_path=str(phonon_path), job_label="quantum_phonon_splitters")

    phonon_splitter = dialog._phonon_mode_table.parentWidget()
    assert isinstance(phonon_splitter, QSplitter)
    assert dialog._phonon_mode_table.minimumWidth() == 320
    assert dialog._phonon_mode_table.minimumWidth() < dialog._phonon_mode_table.maximumWidth()

    single_particle_splitter = dialog._single_particle_table.parentWidget()
    assert isinstance(single_particle_splitter, QSplitter)
    assert dialog._single_particle_table.minimumWidth() == 320

    # Correlation-tab matrices are capped (setMaximumSize), never locked
    # (setFixedSize), so they can shrink and scroll on a narrow dialog.
    assert (
        dialog._covariance_table.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert dialog._covariance_table.maximumWidth() < 16_777_215
    assert dialog._covariance_table.maximumHeight() < 16_777_215

    dialog.close()


def test_quantum_phonon_analysis_dialog_solves_and_calculates_pair_matrices(qapp, tmp_path: Path) -> None:
    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(PHONON_TEXT, encoding="utf-8")

    dialog = QuantumPhononAnalysisDialog(
        phonon_path=str(phonon_path),
        job_label="quantum_phonon_site",
    )

    assert len(dialog.single_particle_results) == 3
    assert dialog._single_particle_table.rowCount() == 3
    assert dialog._single_particle_table.horizontalHeaderItem(0).text() == "Mode no."
    assert dialog._single_particle_table.horizontalHeaderItem(2).text() == "Frequency (cm⁻¹)"
    assert dialog._single_particle_table.horizontalHeaderItem(3).text() == "ZPE (eV)"
    assert dialog._single_particle_table.horizontalHeaderItem(4).text() == "Width (Å)"
    assert "cm" not in dialog._single_particle_table.item(0, 2).text()
    assert dialog._covariance_table.item(0, 0) is not None
    assert dialog._correlation_table.item(0, 0) is not None
    assert dialog._w2_label.text().startswith("w₂ = ")
    assert dialog._phonon_mode_table.rowCount() == 6
    assert dialog._phonon_mode_table.horizontalHeaderItem(0).text() == "Mode"
    assert dialog._phonon_mode_table.horizontalHeaderItem(1).text() == "Frequency (cm⁻¹)"
    assert dialog._phonon_mode_table.horizontalHeaderItem(2).text() == "Muon participation"
    assert dialog._phonon_mode_table.horizontalHeaderItem(3).text() == "Muon direction"
    assert dialog._phonon_mode_table.columnCount() == 4
    dominant_participation_item = dialog._phonon_mode_table.item(1, 2)
    ordinary_participation_item = dialog._phonon_mode_table.item(0, 2)
    assert dominant_participation_item is not None
    assert ordinary_participation_item is not None
    assert dominant_participation_item.foreground().color() == QColor(current_palette().success)
    assert dominant_participation_item.foreground().color() != ordinary_participation_item.foreground().color()
    assert dialog._selected_phonon_mode_row() == 1
    assert dialog._phonon_mode_table.selectionMode() == dialog._phonon_mode_table.SelectionMode.SingleSelection
    # The default 1320x760 size is routed through clamped_size, which shrinks
    # it to fit the test environment's (possibly small, e.g. offscreen-
    # platform) screen; compare against the same clamp rather than a raw
    # literal so this assertion holds on any screen size.
    assert dialog.width() >= clamped_size(dialog, 1320, 760).width()
    assert dialog._phonon_amplitude_spin.value() == 1.0
    assert dialog._phonon_cycle_spin.value() == 2.0
    assert not dialog._phonon_show_vectors_checkbox.isChecked()
    assert dialog._phonon_animation_timer.interval() == 20
    assert dialog._phonon_play_button.text() == "Play animation"
    assert dialog._phonon_animation_status_label.text() == "Animation paused"
    assert dialog._phonon_play_button.styleSheet() == ""
    assert dialog._phonon_play_button.minimumWidth() >= dialog._phonon_play_button.fontMetrics().horizontalAdvance("Pause animation")
    assert dialog._phonon_play_button.minimumHeight() == dialog._phonon_play_button.maximumHeight()
    assert dialog._phonon_play_button.height() == dialog._phonon_play_button.maximumHeight()
    assert dialog._phonon_viewer.selected_atom_indices() == (1,)
    assert dialog._phonon_viewer._line_overlay_specs == []
    assert dialog._phonon_mode_table.width() <= 620
    assert dialog._phonon_mode_table.minimumWidth() == 320
    assert dialog._phonon_mode_table.maximumWidth() <= 620
    assert dialog._phonon_mode_table.horizontalHeader().sectionResizeMode(3) == dialog._phonon_mode_table.horizontalHeader().ResizeMode.Stretch

    tabs = dialog.findChild(QTabWidget, "quantumAnalysisTabs")
    assert tabs is not None
    assert [tabs.tabText(index) for index in range(tabs.count())] == ["Phonon Modes", "Adiabatic Solver", "Muon Coupling"]
    assert tabs.widget(0).objectName() == "quantumAnalysisTabPage"
    assert tabs.widget(1).objectName() == "quantumAnalysisTabPage"
    assert tabs.widget(2).objectName() == "quantumAnalysisTabPage"
    assert dialog._single_particle_table.minimumWidth() == 320
    assert dialog._single_particle_table.item(0, 0).text() == "2"
    direction_item = dialog._single_particle_table.item(0, 1)
    assert direction_item.text().startswith("[")
    assert direction_item.text().endswith("]")
    assert " " in direction_item.text()
    assert "Mode " not in direction_item.text()
    assert "." in direction_item.text()
    assert "[1 0 0]" not in direction_item.text()
    assert direction_item.toolTip() == ""
    assert dialog._single_particle_table.horizontalHeader().sectionResizeMode(1) == dialog._single_particle_table.horizontalHeader().ResizeMode.Stretch
    assert dialog._covariance_table.width() > 0
    assert dialog._correlation_table.width() > 0
    covariance_box = next(box for box in dialog.findChildren(QGroupBox) if box.title() == "Covariance")
    correlation_box = next(box for box in dialog.findChildren(QGroupBox) if box.title() == "Correlation")
    covariance_layout = covariance_box.layout()
    correlation_layout = correlation_box.layout()
    assert covariance_layout.spacing() == 10
    assert correlation_layout.spacing() == 10
    assert covariance_layout.itemAt(1).alignment() & Qt.AlignmentFlag.AlignHCenter
    assert covariance_layout.itemAt(3).alignment() & Qt.AlignmentFlag.AlignHCenter
    assert correlation_layout.itemAt(1).alignment() & Qt.AlignmentFlag.AlignHCenter
    assert correlation_layout.itemAt(3).alignment() & Qt.AlignmentFlag.AlignHCenter

    help_buttons = dialog.findChildren(QToolButton, "quantumAnalysisHelpButton")
    assert len(help_buttons) == 2
    rendered_math = [label.math() for label in dialog.findChildren(MathLabel)]
    assert r"C_{ab}\;(\AA^2)" in rendered_math
    assert r"\rho_{ab}" in rendered_math

    hint_texts = [label.text() for label in dialog.findChildren(QLabel) if label.objectName() == "hintText"]
    assert any("embedded viewer" in text for text in hint_texts)
    assert any("confinement" in text for text in hint_texts)
    assert any("zero-point motion" in text for text in hint_texts)

    dialog.close()


def test_quantum_analysis_plot_follows_selected_direction_rows(qapp, tmp_path: Path) -> None:
    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(PHONON_TEXT, encoding="utf-8")
    dialog = QuantumPhononAnalysisDialog(phonon_path=str(phonon_path), job_label="quantum_phonon_site")

    assert len(dialog._figure.axes[0].lines) == 9

    dialog._single_particle_table.clearSelection()
    dialog._single_particle_table.item(0, 0).setSelected(True)
    dialog._single_particle_table.item(0, 1).setSelected(True)
    dialog._single_particle_table.item(2, 0).setSelected(True)
    dialog._single_particle_table.item(2, 1).setSelected(True)
    qapp.processEvents()

    assert dialog._selected_single_particle_indices() == [0, 2]
    assert len(dialog._figure.axes[0].lines) == 6

    dialog.close()


def test_quantum_analysis_phonon_mode_viewer_animates_and_resets(qapp, tmp_path: Path, monkeypatch) -> None:
    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(PHONON_TEXT, encoding="utf-8")
    dialog = QuantumPhononAnalysisDialog(phonon_path=str(phonon_path), job_label="quantum_phonon_site")

    equilibrium = np.asarray(dialog._phonon_structure.fractional_positions, dtype=float)
    current = np.asarray(dialog._phonon_viewer._structure.fractional_positions, dtype=float)
    assert np.allclose(current, equilibrium)
    dialog._phonon_show_vectors_checkbox.setChecked(True)
    dialog._phonon_viewer._camera_state = dialog._phonon_viewer._camera_state.__class__(23.0, 61.0, -6.0)
    dialog._phonon_viewer._axis.set_from_state(dialog._phonon_viewer._camera_state)
    dialog._phonon_viewer._view = (23.0, 61.0)
    dialog._phonon_viewer._view_roll = -6.0

    dialog._phonon_play_button.click()
    assert dialog._phonon_animation_timer.isActive()
    assert dialog._phonon_play_button.text() == "Pause animation"
    assert dialog._phonon_animation_status_label.text() == "Animation running"
    assert current_palette().success in dialog._phonon_play_button.styleSheet()
    dialog._phonon_last_tick_seconds = 10.0
    monkeypatch.setattr(quantum_analysis_dialog_module, "perf_counter", lambda: 10.5)

    dialog._advance_phonon_animation()
    assert np.isclose(dialog._phonon_phase_radians, pi / 2.0)
    displaced = np.asarray(dialog._phonon_viewer._structure.fractional_positions, dtype=float)
    assert not np.allclose(displaced, equilibrium)
    assert dialog._phonon_viewer._camera_state == dialog._phonon_viewer._camera_state.__class__(23.0, 61.0, -6.0)

    dialog._phonon_mode_table.selectRow(3)
    qapp.processEvents()
    assert dialog._selected_phonon_mode_row() == 3
    assert dialog._phonon_animation_timer.isActive()

    dialog._phonon_show_vectors_checkbox.setChecked(False)
    assert dialog._phonon_viewer._line_overlay_specs == []

    dialog._phonon_play_button.click()
    assert not dialog._phonon_animation_timer.isActive()
    assert dialog._phonon_play_button.text() == "Play animation"
    assert dialog._phonon_animation_status_label.text() == "Animation paused"
    paused = np.asarray(dialog._phonon_viewer._structure.fractional_positions, dtype=float)
    assert np.allclose(paused, equilibrium)

    dialog._phonon_phase_radians = 1.23
    dialog._phonon_reset_button.click()
    assert dialog._phonon_phase_radians == 0.0
    reset = np.asarray(dialog._phonon_viewer._structure.fractional_positions, dtype=float)
    assert np.allclose(reset, equilibrium)

    dialog.close()


def test_quantum_analysis_help_dialog_renders_equations(qapp) -> None:
    dialog = QuantumAnalysisHelpDialog(
        {
            "title": "Equation help",
            "paragraphs": ("Plain-language context.",),
            "equations": (r"E_0=\frac{1}{2}\hbar\omega", r"\sigma_\mu=\sqrt{\left<q^2\right>}"),
            "symbols": ((r"E_0", "Ground-state energy."),),
        }
    )

    assert dialog.windowTitle() == "Equation help"
    assert len(dialog.findChildren(MathLabel)) >= 3
    assert any(label.math() == r"E_0=\frac{1}{2}\hbar\omega" for label in dialog.findChildren(MathLabel))

    dialog.close()


def test_quantum_analysis_dialog_has_no_mass_banner_for_a_correct_run(qapp, tmp_path: Path) -> None:
    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(PHONON_TEXT, encoding="utf-8")

    dialog = QuantumPhononAnalysisDialog(
        phonon_path=str(phonon_path), job_label="quantum_phonon_site"
    )

    assert dialog.muon_mass_warning is None
    assert dialog._muon_mass_warning_label is None
    dialog.close()


def test_quantum_analysis_dialog_banners_a_hydrogen_mass_muon(qapp, tmp_path: Path) -> None:
    """Every tab of this dialog reads plausibly for a run done at the wrong mass,
    so the mismatch is stated up front instead of being left to be inferred."""

    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(
        PHONON_TEXT.replace("H:mu      0.113429", "H:mu      1.007940"), encoding="utf-8"
    )

    dialog = QuantumPhononAnalysisDialog(
        phonon_path=str(phonon_path), job_label="quantum_phonon_site"
    )

    assert dialog.muon_mass_warning is not None
    banner = dialog._muon_mass_warning_label
    assert banner is not None
    assert "not the muon mass" in banner.text()
    assert banner.wordWrap() is True
    assert banner.objectName() == "warningText"
    dialog.close()
