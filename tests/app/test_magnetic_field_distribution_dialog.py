"""Tests for the Magnetic Fields field-distribution plot and export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mufinder.app.magnetic_field_distribution_dialog as magnetic_field_distribution_dialog_module
from mufinder.app import export_dialogs
from mufinder.app.magnetic_field_distribution_dialog import MagneticFieldDistributionDialog
from mufinder.app.theme import plot_series_colours


def test_magnetic_field_distribution_dialog_plots_histogram(qapp) -> None:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        histogram_centres_T = np.asarray([1.0, 3.0], dtype=float)
        histogram_counts = np.asarray([4, 2], dtype=float)

    dialog = MagneticFieldDistributionDialog(_Distribution())

    axis = dialog._figure.axes[0]
    assert axis.get_xlabel() == r"$B$ (T)"
    assert axis.get_ylabel() == r"$p(B)$"
    assert len(axis.lines) == 1
    assert axis.lines[0].get_ydata().max() == pytest.approx(1.0)
    assert axis.get_ylim()[0] == pytest.approx(0.0)
    dialog.close()


def test_magnetic_field_distribution_dialog_marks_an_unrelaxed_estimate(qapp) -> None:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        histogram_centres_T = np.asarray([1.0, 3.0], dtype=float)
        histogram_counts = np.asarray([4, 2], dtype=float)

    estimate = MagneticFieldDistributionDialog(_Distribution(), estimate=True)
    assert estimate._figure.axes[0].get_title() == "site 1 (unrelaxed estimate)"
    assert "# Estimate: UEP sites are unrelaxed" in estimate._export_text()
    estimate.close()

    relaxed = MagneticFieldDistributionDialog(_Distribution())
    assert relaxed._figure.axes[0].get_title() == "site 1"
    assert "# Estimate" not in relaxed._export_text()
    relaxed.close()


def test_magnetic_field_distribution_dialog_overlays_multiple_histograms(qapp) -> None:
    class _Distribution:
        def __init__(self, label: str, centres: list[float], counts: list[float]) -> None:
            self.site_label = label
            self.fields_T = np.asarray([[0.0, 0.0, value] for value in centres], dtype=float)
            self.histogram_centres_T = np.asarray(centres, dtype=float)
            self.histogram_counts = np.asarray(counts, dtype=float)

    dialog = MagneticFieldDistributionDialog(
        (
            _Distribution("site 1", [1.0, 2.0], [4.0, 2.0]),
            _Distribution("site 2", [1.5, 2.5], [1.0, 3.0]),
        )
    )

    axis = dialog._figure.axes[0]
    legend = axis.get_legend()
    assert axis.get_title() == "2 field distributions"
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["site 1", "site 2", "sum"]
    distribution_lines = [
        line for line in axis.lines if line.get_label() in {"site 1", "site 2", "sum"}
    ]
    assert len(distribution_lines) == 3
    assert axis.get_ylim()[1] > 1.0
    series = plot_series_colours()
    assert [line.get_color() for line in distribution_lines] == [
        series.categorical[0],
        series.categorical[1],
        series.total,
    ]
    dialog.close()


def test_magnetic_field_distribution_sum_uses_common_bins_for_delta(qapp) -> None:
    class _Distribution:
        def __init__(self, label: str, centres: list[float], counts: list[float]) -> None:
            self.site_label = label
            self.fields_T = np.asarray([[0.0, 0.0, value] for value in centres], dtype=float)
            self.histogram_centres_T = np.asarray(centres, dtype=float)
            self.histogram_counts = np.asarray(counts, dtype=float)

    delta_centre = 0.0719
    dialog = MagneticFieldDistributionDialog(
        (
            _Distribution("delta", [delta_centre], [5.0]),
            _Distribution("broad", [0.07, 0.08, 0.09], [1.0, 0.5, 0.25]),
        )
    )

    axis = dialog._figure.axes[0]
    delta_line = next(line for line in axis.lines if line.get_label() == "delta")
    sum_line = next(line for line in axis.lines if line.get_label() == "sum")
    delta_x = np.asarray(delta_line.get_xdata(), dtype=float)
    delta_y = np.asarray(delta_line.get_ydata(), dtype=float)
    sum_x = np.asarray(sum_line.get_xdata(), dtype=float)
    sum_y = np.asarray(sum_line.get_ydata(), dtype=float)
    delta_support = delta_y > 0.5

    assert sum_x == pytest.approx(delta_x)
    assert np.any(delta_support)
    assert float(np.max(sum_y[delta_support])) >= 1.0
    dialog.close()


def test_magnetic_field_distribution_dialog_exports_dat_file(qapp, tmp_path, monkeypatch) -> None:
    class _Distribution:
        site_label = "site 1"
        muon_position = (0.25, 0.25, 0.25)
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        histogram_centres_T = np.asarray([1.0, 3.0], dtype=float)
        histogram_counts = np.asarray([4, 2], dtype=float)
        calculation_parameters = {"supercell": [12, 12, 12], "nangles": 4096}

    export_path = tmp_path / "distribution.dat"
    offered: list[str] = []

    def _get_save_file_name(_parent, _title, suggested, filters):
        offered.append(filters)
        assert suggested == "field_distribution.dat"
        # A name without a suffix takes the chosen filter's.
        return str(tmp_path / "distribution"), "DAT files (*.dat)"

    monkeypatch.setattr(export_dialogs.QFileDialog, "getSaveFileName", _get_save_file_name)
    saved: list[Path] = []
    monkeypatch.setattr(
        magnetic_field_distribution_dialog_module,
        "report_export_saved",
        lambda _parent, path: saved.append(path),
    )

    dialog = MagneticFieldDistributionDialog(_Distribution())
    dialog._export_data()

    assert offered == ["DAT files (*.dat);;Text files (*.txt)"]
    assert saved == [export_path]
    text = export_path.read_text(encoding="utf-8")
    assert "# MuFinder field distribution export" in text
    assert "#   nangles: 4096" in text
    assert "# Columns: B_T p_site_1" in text
    assert "1" in text
    dialog.close()


def test_magnetic_field_distribution_export_reports_a_failed_write(
    qapp, tmp_path, monkeypatch
) -> None:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)

    missing_folder = tmp_path / "missing" / "distribution.dat"
    monkeypatch.setattr(
        magnetic_field_distribution_dialog_module,
        "ask_save_path",
        lambda *_args, **_kwargs: missing_folder,
    )
    failures: list[Exception] = []
    monkeypatch.setattr(
        magnetic_field_distribution_dialog_module,
        "report_export_failed",
        lambda _parent, error: failures.append(error),
    )
    monkeypatch.setattr(
        magnetic_field_distribution_dialog_module,
        "report_export_saved",
        lambda *_args: pytest.fail("a failed export is not reported as saved"),
    )

    dialog = MagneticFieldDistributionDialog(_Distribution())
    dialog._export_data()

    assert len(failures) == 1
    assert isinstance(failures[0], OSError)
    dialog.close()


def test_magnetic_field_distribution_dialog_expands_near_delta_axis(qapp) -> None:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray(
            [
                [0.0, 0.0, 0.1907378471],
                [0.0, 0.0, 0.1907378471001],
                [0.0, 0.0, 0.1907378470999],
            ],
            dtype=float,
        )
        histogram_centres_T = np.asarray(
            [0.1907378470999, 0.1907378471, 0.1907378471001], dtype=float
        )
        histogram_counts = np.asarray([1, 3, 1], dtype=float)

    dialog = MagneticFieldDistributionDialog(_Distribution())

    axis = dialog._figure.axes[0]
    x_low, x_high = axis.get_xlim()
    assert x_high - x_low > 0.001
    assert axis.xaxis.get_major_formatter().get_useOffset() is False
    dialog.close()


def test_magnetic_field_distribution_dialog_reports_a_non_finite_sample(qapp) -> None:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, np.inf]], dtype=float)
        histogram_centres_T = np.asarray([], dtype=float)
        histogram_counts = np.asarray([], dtype=float)

    dialog = MagneticFieldDistributionDialog(_Distribution())

    axis = dialog._figure.axes[0]
    assert [text.get_text() for text in axis.texts] == [
        "Distribution unavailable: field distribution contains non-finite magnitudes."
    ]
    assert len(axis.lines) == 0
    assert not dialog._export_button.isEnabled()
    dialog.close()


def test_magnetic_field_distribution_dialog_plots_a_single_field_as_a_narrow_spike(qapp) -> None:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 0.2]] * 50, dtype=float)
        histogram_centres_T = np.asarray([], dtype=float)
        histogram_counts = np.asarray([], dtype=float)

    dialog = MagneticFieldDistributionDialog(_Distribution())

    (line,) = dialog._figure.axes[0].lines
    raised = np.asarray(line.get_xdata())[np.asarray(line.get_ydata()) > 0.0]
    assert raised.max() - raised.min() < 1e-6
    assert dialog._export_button.isEnabled()
    dialog.close()
