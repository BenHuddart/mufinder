"""Tests for the Magnetic Fields field-distribution plot and export."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mufinder.app.magnetic_field_distribution_dialog as magnetic_field_distribution_dialog_module
from mufinder.app import export_dialogs
from mufinder.app.magnetic_field_distribution_dialog import MagneticFieldDistributionDialog
from mufinder.app.theme import plot_series_colours
from mufinder.services.magnetic_field_results import OLDER_METHOD_RESULTS_NOTE
from mufinder.services.tabular_export import (
    ExportProvenance,
    TableColumn,
    TableExport,
    TableFormat,
    TableSection,
    render_table_export,
)

WRITTEN_AT = datetime(2026, 11, 23, 10, 15, tzinfo=UTC)


def _export_text(dialog: MagneticFieldDistributionDialog) -> str:
    return render_table_export(dialog._table_export(), TableFormat.DAT, written_at=WRITTEN_AT)


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
    assert "# Estimate: UEP sites are unrelaxed" in _export_text(estimate)
    estimate.close()

    relaxed = MagneticFieldDistributionDialog(_Distribution())
    assert relaxed._figure.axes[0].get_title() == "site 1"
    assert "# Estimate" not in _export_text(relaxed)
    relaxed.close()


@pytest.mark.parametrize(
    ("estimate", "title"),
    [(False, "site 1 (older method)"), (True, "site 1 (unrelaxed estimate, older method)")],
)
def test_magnetic_field_distribution_dialog_marks_an_older_method(qapp, estimate, title) -> None:
    result = SimpleNamespace(
        site_label="site 1",
        fields_T=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float),
        histogram_centres_T=np.asarray([1.0, 3.0], dtype=float),
        histogram_counts=np.asarray([1.0, 1.0], dtype=float),
    )

    dialog = MagneticFieldDistributionDialog(result, estimate=estimate, older_method=True)

    assert dialog.windowTitle() == f"Field Distribution - {title}"
    assert dialog._figure.axes[0].get_title() == title
    assert f"# Older method: {OLDER_METHOD_RESULTS_NOTE}" in _export_text(dialog).splitlines()
    dialog.close()

    plain = MagneticFieldDistributionDialog(result)
    assert plain._figure.axes[0].get_title() == "site 1"
    assert "# Older method" not in _export_text(plain)
    plain.close()


def test_a_long_distribution_title_wraps_inside_a_narrow_dialog(qapp) -> None:
    results = [
        SimpleNamespace(
            site_label=f"site 1 equiv {index}",
            fields_T=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float),
            histogram_centres_T=np.asarray([1.0, 3.0], dtype=float),
            histogram_counts=np.asarray([1.0, 1.0], dtype=float),
        )
        for index in range(4)
    ]
    dialog = MagneticFieldDistributionDialog(results, estimate=True, older_method=True)
    dialog.resize(dialog.minimumSizeHint())
    dialog.show()
    qapp.processEvents()

    dialog._canvas.draw()

    axis = dialog._figure.axes[0]
    extent = axis.title.get_window_extent(dialog._canvas.get_renderer())
    figure_box = dialog._figure.bbox
    assert axis.title.get_text() == "4 field distributions (unrelaxed estimate, older method)"
    assert extent.x0 >= figure_box.x0
    assert extent.x1 <= figure_box.x1
    assert extent.y1 <= figure_box.y1
    assert extent.y0 >= axis.get_window_extent().y1
    dialog.close()


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
    assert "# Units: T -" in text
    assert "1" in text
    dialog.close()


def test_field_distribution_export_header_names_the_project_and_source_jobs(qapp) -> None:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
        histogram_centres_T = np.asarray([1.0, 3.0], dtype=float)
        histogram_counts = np.asarray([4, 2], dtype=float)

    dialog = MagneticFieldDistributionDialog(
        _Distribution(),
        provenance=ExportProvenance("MnSi", ("mu_relax_01 (id 1234)",)),
    )

    lines = _export_text(dialog).splitlines()

    assert lines[:5] == [
        "# MuFinder field distribution export",
        "# Project: MnSi",
        "# Source jobs: mu_relax_01 (id 1234)",
        "# Exported: 2026-11-23T10:15:00+00:00",
        "# Distributions: 1",
    ]
    data = np.loadtxt(lines, encoding="utf-8")
    assert data.shape[1] == 2
    assert np.all(np.isfinite(data))
    dialog.close()


def test_field_distribution_export_into_a_missing_folder_creates_it_and_saves(
    qapp, tmp_path, monkeypatch
) -> None:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)

    target = tmp_path / "new" / "deeper" / "distribution.dat"
    monkeypatch.setattr(
        export_dialogs.QFileDialog,
        "getSaveFileName",
        lambda *_args: (str(target), "DAT files (*.dat)"),
    )
    shown: list[tuple[str, str]] = []
    for kind in ("information", "warning"):
        monkeypatch.setattr(
            export_dialogs.QMessageBox,
            kind,
            lambda _parent, title, text: shown.append((title, text)),
        )

    dialog = MagneticFieldDistributionDialog(_Distribution())
    dialog._export_data()

    assert shown == [("Export complete", f"Saved to:\n{target}")]
    assert "# MuFinder field distribution export" in target.read_text(encoding="utf-8")
    dialog.close()


def test_field_distribution_export_reports_a_table_that_cannot_be_written(
    qapp, tmp_path, monkeypatch
) -> None:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)

    monkeypatch.setattr(
        magnetic_field_distribution_dialog_module,
        "ask_save_path",
        lambda *_args, **_kwargs: tmp_path / "distribution.dat",
    )
    failures: list[Exception] = []
    monkeypatch.setattr(
        export_dialogs,
        "report_export_failed",
        lambda _parent, error, _path: failures.append(error),
    )
    dialog = MagneticFieldDistributionDialog(_Distribution())
    ragged = TableExport(
        "T", (TableSection((TableColumn("a", [1.0, 2.0]), TableColumn("b", [1.0]))),)
    )
    monkeypatch.setattr(dialog, "_table_export", lambda: ragged)

    dialog._export_data()

    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert not (tmp_path / "distribution.dat").exists()
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


def _failure_text(qapp, monkeypatch, target: Path, raise_error=None) -> str:
    class _Distribution:
        site_label = "site 1"
        fields_T = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)

    monkeypatch.setattr(
        magnetic_field_distribution_dialog_module,
        "ask_save_path",
        lambda *_args, **_kwargs: target,
    )
    if raise_error is not None:

        def _raise(*_args, **_kwargs):
            raise raise_error

        monkeypatch.setattr(magnetic_field_distribution_dialog_module, "write_table_export", _raise)
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        export_dialogs.QMessageBox,
        "warning",
        lambda _parent, title, text: warnings.append((title, text)),
    )
    dialog = MagneticFieldDistributionDialog(_Distribution())
    dialog._export_data()
    dialog.close()
    assert [title for title, _text in warnings] == ["Export failed"]
    return warnings[0][1]


def test_field_distribution_export_words_a_denied_write_plainly(
    qapp, tmp_path, monkeypatch
) -> None:
    target = tmp_path / "distribution.dat"

    text = _failure_text(
        qapp, monkeypatch, target, PermissionError(13, "Permission denied", str(target))
    )

    assert text == f"Could not save 'distribution.dat' in {tmp_path}: permission denied."
    assert "[Errno" not in text


def test_field_distribution_export_under_a_file_says_it_is_not_a_folder(
    qapp, tmp_path, monkeypatch
) -> None:
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("", encoding="utf-8")

    text = _failure_text(qapp, monkeypatch, blocker / "distribution.dat")

    assert text == f"Could not save 'distribution.dat': {blocker} is a file, not a folder."
    assert "[Errno" not in text


def test_field_distribution_export_words_a_table_error_plainly(qapp, tmp_path, monkeypatch) -> None:
    error = ValueError("the columns of a table section differ in length (a: 2, b: 1).")

    text = _failure_text(qapp, monkeypatch, tmp_path / "distribution.dat", error)

    assert text == (
        "Could not save 'distribution.dat': the columns of a table section differ in length "
        "(a: 2, b: 1)."
    )
    assert "[Errno" not in text
