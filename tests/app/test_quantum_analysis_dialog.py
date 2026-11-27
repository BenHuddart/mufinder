"""Tests for the completed phonon analysis dialog."""

from __future__ import annotations

from pathlib import Path
from math import pi

import numpy as np
import pytest

from tests.support.phonon_fixtures import phonon_calculation_text
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QLabel,
    QScrollArea,
    QStyle,
    QTableWidget,
    QTabWidget,
    QToolButton,
    QWidget,
)

import mufinder.app.quantum_analysis_dialog as quantum_analysis_dialog_module
from mufinder.app.math_label import MathLabel
from mufinder.app.quantum_analysis_dialog import (
    MuonCouplingPairDialog,
    QuantumAnalysisHelpDialog,
    QuantumPhononAnalysisDialog,
)
from mufinder.app.screen_utils import clamped_size
from mufinder.app.theme import SPACING, current_palette
from mufinder.app.workflow_table_utils import selected_table_rows
from mufinder.science.quantum_phonons import (
    correlation_matrix,
    muon_nearest_atoms,
    covariance_matrix,
    w2_entanglement_witness,
)
from tests.support.qt import (
    assert_tab_leaves_table,
    click_header_section,
    click_table_row,
)


ALPHA_N2_PHONON = Path(__file__).parents[1] / "fixtures" / "alpha_n2" / "aN2_mu_phonon.phonon"
GDRU2SI2_SITE2_PHONON = (
    Path(__file__).parents[1] / "fixtures" / "gdru2si2" / "GdRu2Si2_site2.phonon"
)
GDRU2SI2_SITE1_PHONON = GDRU2SI2_SITE2_PHONON.with_name("GdRu2Si2_site1.phonon")
ALPHA_N2_MUON = 65


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


def _open_dialog(tmp_path: Path, job_label: str = "quantum_phonon_site"):
    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(PHONON_TEXT, encoding="utf-8")
    return QuantumPhononAnalysisDialog(phonon_path=str(phonon_path), site_label=job_label)


def _show_tab(dialog: QuantumPhononAnalysisDialog, title: str) -> QTabWidget:
    """Switch the analysis window to ``title``, as a click on its tab does."""

    tabs = dialog.findChild(QTabWidget, "quantumAnalysisTabs")
    titles = [tabs.tabText(index) for index in range(tabs.count())]
    tabs.setCurrentIndex(titles.index(title))
    return tabs


def _show_every_tab(dialog: QuantumPhononAnalysisDialog) -> None:
    _show_tab(dialog, "Muon Coupling")
    _show_tab(dialog, "Phonon Modes")


def test_the_analysis_window_has_the_phonon_modes_and_muon_coupling_tabs(
    qapp, tmp_path: Path
) -> None:
    dialog = _open_dialog(tmp_path)
    tabs = dialog.findChild(QTabWidget, "quantumAnalysisTabs")

    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Phonon Modes",
        "Muon Coupling",
    ]

    dialog.close()


def test_opening_the_analysis_window_builds_only_the_shown_tab(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    pair_matrices: list[tuple[int, int]] = []
    real_covariance = quantum_analysis_dialog_module.covariance_matrix
    monkeypatch.setattr(
        quantum_analysis_dialog_module,
        "covariance_matrix",
        lambda calculation, a, b: (
            pair_matrices.append((a, b)) or real_covariance(calculation, a, b)
        ),
    )

    dialog = _open_dialog(tmp_path)
    tabs = dialog.findChild(QTabWidget, "quantumAnalysisTabs")

    assert tabs.currentIndex() == 0
    assert pair_matrices == []
    assert tabs.widget(1).findChildren(QWidget) == []

    _show_tab(dialog, "Muon Coupling")
    assert len(pair_matrices) == 1
    assert dialog._neighbour_table.rowCount() == 1

    # Coming back to a tab shows what was built; nothing is built twice.
    neighbour_table = dialog._neighbour_table
    _show_tab(dialog, "Phonon Modes")
    _show_tab(dialog, "Muon Coupling")
    assert dialog._neighbour_table is neighbour_table
    assert len(pair_matrices) == 1

    dialog.close()


def test_a_tab_that_fails_to_compute_says_so_and_leaves_the_window_working(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    def failing_covariance(_calculation, _a, _b):
        raise ValueError("the pair has no covariance")

    monkeypatch.setattr(quantum_analysis_dialog_module, "covariance_matrix", failing_covariance)
    dialog = _open_dialog(tmp_path)

    tabs = _show_tab(dialog, "Muon Coupling")

    page = tabs.widget(1)
    messages = [label.text() for label in page.findChildren(QLabel)]
    assert messages == ["Could not compute this tab: the pair has no covariance"]
    # The Phonon Modes tab still follows its table after the failure.
    _show_tab(dialog, "Phonon Modes")
    dialog._phonon_mode_table.selectRow(0)
    assert dialog._selected_phonon_mode_row() == 0
    assert dialog._current_phonon_mode_row == 0
    _show_tab(dialog, "Muon Coupling")
    assert [label.text() for label in page.findChildren(QLabel)] == messages

    dialog.close()


def test_opening_the_analysis_window_refreshes_the_phonon_viewer_once(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    refreshes: list[int | None] = []
    real_refresh = QuantumPhononAnalysisDialog._refresh_phonon_mode_view

    def counting_refresh(self) -> None:
        refreshes.append(self._selected_phonon_mode_row())
        real_refresh(self)

    monkeypatch.setattr(QuantumPhononAnalysisDialog, "_refresh_phonon_mode_view", counting_refresh)
    cleared: list[object] = []
    real_overlays = quantum_analysis_dialog_module.AtomicStructureViewer.set_line_overlays
    monkeypatch.setattr(
        quantum_analysis_dialog_module.AtomicStructureViewer,
        "set_line_overlays",
        lambda viewer, overlays: cleared.append(overlays) or real_overlays(viewer, overlays),
    )

    dialog = _open_dialog(tmp_path)

    # One refresh, for the muon-dominated default row, and no redundant clear of
    # vectors that were never drawn (each costs the viewer a render).
    assert refreshes == [1]
    assert cleared == []

    dialog._phonon_show_vectors_checkbox.setChecked(True)
    dialog._phonon_show_vectors_checkbox.setChecked(False)
    assert len(cleared) == 2
    assert cleared[-1] is None
    assert dialog._phonon_viewer._line_overlay_specs == []

    dialog.close()


def test_quantum_analysis_dialog_tables_use_splitters_not_width_locks(qapp, tmp_path: Path) -> None:
    """Tables are capped, not width-locked, so a QSplitter governs sizing.

    Guards the SC-5 small-screen fix: narrowing the dialog should let a
    QSplitter hand space to the viewer instead of a rigid table width
    starving them.
    """
    from PySide6.QtWidgets import QSplitter

    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(PHONON_TEXT, encoding="utf-8")
    dialog = QuantumPhononAnalysisDialog(
        phonon_path=str(phonon_path), site_label="quantum_phonon_splitters"
    )
    _show_every_tab(dialog)

    phonon_splitter = dialog._phonon_mode_table.parentWidget()
    assert isinstance(phonon_splitter, QSplitter)
    assert dialog._phonon_mode_table.minimumWidth() == 320
    assert dialog._phonon_mode_table.minimumWidth() < dialog._phonon_mode_table.maximumWidth()

    dialog.close()


def _open_alpha_n2_coupling() -> QuantumPhononAnalysisDialog:
    dialog = QuantumPhononAnalysisDialog(
        phonon_path=str(ALPHA_N2_PHONON), site_label="alpha-N2 site"
    )
    _show_tab(dialog, "Muon Coupling")
    return dialog


def _table_rows(table) -> list[tuple[str, ...]]:
    return [
        tuple(table.item(row, column).text() for column in range(table.columnCount()))
        for row in range(table.rowCount())
    ]


def test_muon_coupling_lists_the_nearest_atoms_with_their_witness(qapp) -> None:
    dialog = _open_alpha_n2_coupling()
    calculation = dialog.calculation

    table = dialog._neighbour_table
    assert [table.horizontalHeaderItem(column).text() for column in range(4)] == [
        "Atom",
        "Species",
        "Distance (Å)",
        "|c|",
    ]
    rows = _table_rows(table)
    assert len(rows) == 12
    assert rows[:3] == [
        ("57", "N", "1.268", "0.310"),
        ("1", "N", "1.288", "0.298"),
        ("29", "N", "2.367", "0.104"),
    ]
    assert [float(row[2]) for row in rows] == sorted(float(row[2]) for row in rows)
    for atom, _species, _distance, shown in rows:
        expected = -w2_entanglement_witness(calculation, ALPHA_N2_MUON, int(atom))
        assert float(shown) == pytest.approx(expected, rel=5e-3)

    dialog.close()


def test_selecting_a_row_shows_that_atoms_tensors_to_three_figures(qapp) -> None:
    dialog = _open_alpha_n2_coupling()
    tensors = dialog._neighbour_tensors
    table = dialog._neighbour_table

    # The nearest atom is selected from the start.
    assert selected_table_rows(table) == (0,)
    assert dialog._selected_atom_label.text() == "Muon and atom 57: N"

    table.selectRow(1)

    assert dialog._selected_atom_label.text() == "Muon and atom 1: N"
    covariance = covariance_matrix(dialog.calculation, ALPHA_N2_MUON, 1)
    correlation = correlation_matrix(dialog.calculation, ALPHA_N2_MUON, 1)
    for shown, matrix in (
        (tensors.covariance_table, covariance),
        (tensors.correlation_table, correlation),
    ):
        for row, column in np.ndindex(3, 3):
            text = shown.item(row, column).text()
            assert len(text.split("e")[0].replace("-", "").replace(".", "").lstrip("0")) <= 3
            assert float(text) == pytest.approx(matrix[row, column], rel=5e-3, abs=1e-9)

    dialog.close()


def test_the_tensors_fit_without_scrollbars(qapp) -> None:
    dialog = _open_alpha_n2_coupling()
    tensors = dialog._neighbour_tensors
    dialog.resize(1180, 560)
    dialog.show()
    qapp.processEvents()

    for table in (tensors.covariance_table, tensors.correlation_table):
        assert table.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert table.viewport().width() >= sum(table.columnWidth(column) for column in range(3))
        assert table.viewport().height() >= sum(table.rowHeight(row) for row in range(3))
        # Each column shows its whole text.
        for column in range(3):
            assert table.columnWidth(column) >= table.sizeHintForColumn(column)
    # Moving between atoms leaves the tensors where they are.
    sizes = {
        (tensors.covariance_table.sizeHint().width(), tensors.correlation_table.sizeHint().width())
    }
    for row in range(dialog._neighbour_table.rowCount()):
        dialog._neighbour_table.selectRow(row)
        qapp.processEvents()
        sizes.add(
            (
                tensors.covariance_table.sizeHint().width(),
                tensors.correlation_table.sizeHint().width(),
            )
        )
    assert len(sizes) == 1
    # The default list needs no scrolling at the window's default size either.
    dialog.resize(clamped_size(dialog, 1320, 760))
    qapp.processEvents()
    table = dialog._neighbour_table
    assert table.verticalScrollBar().maximum() == 0

    dialog.close()


def test_changing_atoms_shown_lengthens_the_list_and_keeps_the_selection(qapp) -> None:
    dialog = _open_alpha_n2_coupling()
    table = dialog._neighbour_table
    assert dialog._nearest_count_spin.value() == 12
    assert dialog._nearest_count_spin.maximum() == 64

    # A half-typed count changes nothing until it is finished.
    assert not dialog._nearest_count_spin.keyboardTracking()

    table.selectRow(5)
    selected_atom = table.item(5, 0).text()

    dialog._nearest_count_spin.setValue(64)
    assert table.rowCount() == 64
    assert table.item(selected_table_rows(table)[0], 0).text() == selected_atom
    # Atom 17 is 24th by distance, and more correlated than any of the twelve from the 9th on.
    atom_17 = next(row for row in _table_rows(table) if row[0] == "17")
    assert atom_17[3] == "0.0317"
    assert _table_rows(table).index(atom_17) == 23

    # An atom that drops off the list gives way to the nearest.
    dialog._nearest_count_spin.setValue(3)
    assert table.rowCount() == 3
    assert selected_table_rows(table) == (0,)
    assert dialog._selected_atom_label.text() == "Muon and atom 57: N"

    dialog.close()


def _open_shown_coupling(phonon_path: Path) -> QuantumPhononAnalysisDialog:
    dialog = QuantumPhononAnalysisDialog(phonon_path=str(phonon_path), site_label="site")
    _show_tab(dialog, "Muon Coupling")
    dialog.resize(1180, 560)
    dialog.show()
    QApplication.processEvents()
    return dialog


def _atoms(table) -> list[str]:
    return [table.item(row, 0).text() for row in range(table.rowCount())]


def _selected_atom(table) -> str:
    return table.item(selected_table_rows(table)[0], 0).text()


def test_the_coupling_table_is_ordered_by_distance_at_first(qapp) -> None:
    dialog = _open_shown_coupling(GDRU2SI2_SITE2_PHONON)
    header = dialog._neighbour_table.horizontalHeader()

    assert header.isSortIndicatorShown()
    assert header.sortIndicatorSection() == 2
    assert header.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
    # The shells of Ru 17 and 19, Si 8 and 12, Ru 21 and 27 and Ru 23 and 25 in index order.
    assert _atoms(dialog._neighbour_table) == [
        "17", "19", "4", "33", "1", "8", "12", "21", "27", "3", "23", "25",
    ]  # fmt: skip

    dialog.close()


def test_clicking_a_header_sorts_the_listed_atoms_and_keeps_the_selection(qapp) -> None:
    dialog = _open_shown_coupling(GDRU2SI2_SITE2_PHONON)
    table = dialog._neighbour_table
    before = sorted(_table_rows(table))
    click_table_row(table, 4)
    assert _selected_atom(table) == "1"
    shown_tensors = dialog._neighbour_tensors.correlation_table.item(0, 0).text()

    # |c| orders strongest first: Si 4 couples most, though it is third by distance.
    click_header_section(table, 3)
    assert table.horizontalHeader().sortIndicatorSection() == 3
    assert table.horizontalHeader().sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
    assert _atoms(table) == ["4", "17", "19", "1", "27", "3", "21", "12", "8", "33", "25", "23"]
    correlations = [float(row[3]) for row in _table_rows(table)]
    assert correlations == sorted(correlations, reverse=True)
    # Atom 1 stays selected, and its tensors stay shown.
    assert _selected_atom(table) == "1"
    assert dialog._selected_atom_label.text() == "Muon and atom 1: Si"
    assert dialog._neighbour_tensors.correlation_table.item(0, 0).text() == shown_tensors
    # Only the order changes, never a value.
    assert sorted(_table_rows(table)) == before

    # A second click reverses the order.
    click_header_section(table, 3)
    assert table.horizontalHeader().sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
    assert _atoms(table) == ["23", "25", "33", "8", "12", "21", "3", "27", "1", "19", "17", "4"]
    assert _selected_atom(table) == "1"
    assert sorted(_table_rows(table)) == before

    dialog.close()


def test_sorting_a_column_keeps_ties_in_index_order_both_ways(qapp) -> None:
    dialog = _open_shown_coupling(GDRU2SI2_SITE2_PHONON)
    table = dialog._neighbour_table

    # Distance (Å): the 3.212 and 3.221 Å shells keep index order when reversed.
    click_header_section(table, 2)
    assert table.horizontalHeader().sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
    assert _atoms(table) == ["23", "25", "3", "21", "27", "8", "12", "1", "33", "4", "17", "19"]

    # Species: alphabetical, each species in index order, in both directions.
    click_header_section(table, 1)
    assert _atoms(table) == ["33", "17", "19", "21", "23", "25", "27", "1", "3", "4", "8", "12"]
    click_header_section(table, 1)
    assert _atoms(table) == ["1", "3", "4", "8", "12", "17", "19", "21", "23", "25", "27", "33"]

    # Atom orders by number, not by text: 4 before 12 before 33.
    click_header_section(table, 0)
    assert _atoms(table) == ["1", "3", "4", "8", "12", "17", "19", "21", "23", "25", "27", "33"]

    dialog.close()


@pytest.mark.parametrize(
    "phonon_path", [ALPHA_N2_PHONON, GDRU2SI2_SITE1_PHONON, GDRU2SI2_SITE2_PHONON]
)
def test_distance_order_is_the_nearest_atom_order_for_every_count(qapp, phonon_path) -> None:
    # α-N2 atoms 23, 52 and 10 are one 0.001 Å shell; at 46 atoms shown, 52 is cut
    # off and 23 and 10, 0.0012 Å apart, must still keep index order.
    dialog = _open_shown_coupling(phonon_path)
    table = dialog._neighbour_table
    spin = dialog._nearest_count_spin

    for count in range(1, spin.maximum() + 1):
        spin.setValue(count)
        nearest = muon_nearest_atoms(dialog.calculation, count)
        assert _atoms(table) == [str(neighbour.atom_index) for neighbour in nearest], count

    # Reversed, the shells formed over every atom go farthest first, each in index order.
    click_header_section(table, 2)
    assert table.horizontalHeader().sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
    for count in range(1, spin.maximum() + 1):
        spin.setValue(count)
        shells: dict[int, list[str]] = {}
        for neighbour in muon_nearest_atoms(dialog.calculation, count):
            shells.setdefault(neighbour.distance_shell, []).append(str(neighbour.atom_index))
        expected = [atom for shell in sorted(shells, reverse=True) for atom in shells[shell]]
        assert _atoms(table) == expected, count

    dialog.close()


@pytest.mark.parametrize("phonon_path", [ALPHA_N2_PHONON, GDRU2SI2_SITE2_PHONON])
@pytest.mark.parametrize("size", ["minimum", "default"])
def test_the_sort_arrow_never_covers_a_header_label(qapp, phonon_path, size) -> None:
    dialog = _open_shown_coupling(phonon_path)
    if size == "minimum":
        dialog.resize(dialog.minimumSizeHint())
    else:
        dialog.resize(clamped_size(dialog, 1320, 760))
    QApplication.processEvents()
    table = dialog._neighbour_table
    header = table.horizontalHeader()
    # The theme draws header labels bold, padded by SPACING.xs on each side.
    font = header.font()
    font.setBold(True)
    metrics = QFontMetrics(font)
    # Qt's common style draws the arrow half the header's height wide, inset by
    # twice the header margin from the section's right edge.
    margin = header.style().pixelMetric(QStyle.PixelMetric.PM_HeaderMargin, None, header)
    arrow = header.height() // 2 + 2 * margin

    for column in range(table.columnCount()):
        header.setSortIndicator(column, Qt.SortOrder.AscendingOrder)
        QApplication.processEvents()
        for shown in range(table.columnCount()):
            label = table.horizontalHeaderItem(shown).text()
            needed = metrics.horizontalAdvance(label) + 2 * SPACING.xs
            if shown == column:
                needed += arrow
            assert header.sectionSize(shown) >= needed, (column, label)
        # Every column is shown whole, inside the table.
        assert header.length() <= table.viewport().width()

    dialog.close()


@pytest.mark.parametrize(("column", "clicks"), [(2, 0), (2, 1), (3, 1), (3, 2), (1, 1), (0, 1)])
def test_tab_leaves_the_sorted_table_and_the_current_row_follows_the_atom(
    qapp, column: int, clicks: int
) -> None:
    dialog = _open_shown_coupling(GDRU2SI2_SITE2_PHONON)
    table = dialog._neighbour_table
    click_table_row(table, 4)
    for _click in range(clicks):
        click_header_section(table, column)

    # Refilling in the new order leaves the current cell on the selected atom.
    assert _selected_atom(table) == "1"
    assert table.currentRow() == selected_table_rows(table)[0]
    assert_tab_leaves_table(table)

    dialog.close()


def test_changing_atoms_shown_keeps_the_sort(qapp) -> None:
    dialog = _open_shown_coupling(ALPHA_N2_PHONON)
    table = dialog._neighbour_table
    click_header_section(table, 3)
    click_table_row(table, 2)
    assert _selected_atom(table) == "29"

    dialog._nearest_count_spin.setValue(24)

    # Still strongest first, now with atom 17, 24th by distance, ninth by |c|.
    assert table.horizontalHeader().sortIndicatorSection() == 3
    assert table.rowCount() == 24
    correlations = [float(row[3]) for row in _table_rows(table)]
    assert correlations == sorted(correlations, reverse=True)
    assert _atoms(table)[8] == "17"
    assert _selected_atom(table) == "29"
    # The count still picks the nearest atoms; sorting only reorders them.
    assert sorted(_atoms(table), key=int) == sorted(
        (str(neighbour.atom_index) for neighbour in dialog._nearest), key=int
    )
    assert max(float(row[2]) for row in _table_rows(table)) == pytest.approx(4.704, abs=1e-3)

    # A selected atom that drops off the list gives way to the first row.
    dialog._nearest_count_spin.setValue(2)
    assert _atoms(table) == ["57", "1"]
    click_table_row(table, 1)
    dialog._nearest_count_spin.setValue(1)
    assert _atoms(table) == ["57"]
    assert dialog._selected_atom_label.text() == "Muon and atom 57: N"

    dialog.close()


def test_a_cell_with_few_atoms_lists_them_all(qapp, tmp_path: Path) -> None:
    dialog = _open_dialog(tmp_path)
    _show_tab(dialog, "Muon Coupling")

    assert dialog._nearest_count_spin.value() == 1
    assert dialog._nearest_count_spin.maximum() == 1
    assert _table_rows(dialog._neighbour_table)[0][:2] == ("1", "N")

    dialog.close()


def test_other_pair_shows_any_two_atoms(qapp, monkeypatch) -> None:
    dialog = _open_alpha_n2_coupling()
    dialog._neighbour_table.selectRow(2)
    opened: list[MuonCouplingPairDialog] = []
    monkeypatch.setattr(MuonCouplingPairDialog, "exec", lambda pair: opened.append(pair) or 0)

    assert dialog._other_pair_button.text() == "Other pair…"
    dialog._other_pair_button.click()

    pair = opened[0]
    # It starts on the muon and the selected atom, and follows each picker.
    assert pair._atom_a_combo.currentData() == ALPHA_N2_MUON
    assert pair._atom_b_combo.currentData() == 29
    assert pair._correlation_label.text() == "|c| = 0.104"
    assert pair._atom_a_combo.accessibleName() == "Atom A"
    assert pair._atom_b_combo.accessibleName() == "Atom B"

    pair._atom_a_combo.setCurrentIndex(pair._atom_a_combo.findData(1))
    pair._atom_b_combo.setCurrentIndex(pair._atom_b_combo.findData(2))

    expected = -w2_entanglement_witness(dialog.calculation, 1, 2)
    assert float(pair._correlation_label.text().removeprefix("|c| = ")) == pytest.approx(
        expected, rel=5e-3
    )
    assert float(pair._tensors.correlation_table.item(0, 0).text()) == pytest.approx(
        correlation_matrix(dialog.calculation, 1, 2)[0, 0], rel=5e-3
    )

    pair.close()
    dialog.close()


def test_the_muon_coupling_controls_are_named(qapp) -> None:
    dialog = _open_alpha_n2_coupling()

    assert dialog._neighbour_table.accessibleName() == "Muon's nearest atoms"
    assert dialog._nearest_count_spin.accessibleName() == "Atoms shown"
    tensors = dialog._neighbour_tensors
    assert tensors.covariance_table.accessibleName() == "Covariance tensor"
    assert tensors.correlation_table.accessibleName() == "Correlation tensor"
    assert (
        dialog.findChildren(QToolButton, "quantumAnalysisHelpButton")[0].accessibleName()
        == "Muon's nearest atoms help"
    )

    dialog.close()


def test_quantum_phonon_analysis_dialog_lists_modes_and_calculates_pair_matrices(
    qapp, tmp_path: Path
) -> None:
    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(PHONON_TEXT, encoding="utf-8")

    dialog = QuantumPhononAnalysisDialog(
        phonon_path=str(phonon_path),
        site_label="quantum_phonon_site",
    )
    _show_every_tab(dialog)

    assert dialog._neighbour_tensors.covariance_table.item(0, 0) is not None
    assert dialog._neighbour_tensors.correlation_table.item(0, 0) is not None
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
    assert (
        dominant_participation_item.foreground().color()
        != ordinary_participation_item.foreground().color()
    )
    assert dialog._selected_phonon_mode_row() == 1
    assert (
        dialog._phonon_mode_table.selectionMode()
        == dialog._phonon_mode_table.SelectionMode.SingleSelection
    )
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
    assert (
        dialog._phonon_play_button.minimumWidth()
        >= dialog._phonon_play_button.fontMetrics().horizontalAdvance("Pause animation")
    )
    assert dialog._phonon_play_button.minimumHeight() == dialog._phonon_play_button.maximumHeight()
    assert dialog._phonon_play_button.height() == dialog._phonon_play_button.maximumHeight()
    assert dialog._phonon_viewer.selected_atom_indices() == (1,)
    assert dialog._phonon_viewer._line_overlay_specs == []
    assert dialog._phonon_mode_table.width() <= 620
    assert dialog._phonon_mode_table.minimumWidth() == 320
    assert dialog._phonon_mode_table.maximumWidth() <= 620
    assert (
        dialog._phonon_mode_table.horizontalHeader().sectionResizeMode(3)
        == dialog._phonon_mode_table.horizontalHeader().ResizeMode.Stretch
    )

    tabs = dialog.findChild(QTabWidget, "quantumAnalysisTabs")
    assert tabs is not None
    assert tabs.widget(0).objectName() == "quantumAnalysisTabPage"
    assert tabs.widget(1).objectName() == "quantumAnalysisTabPage"
    direction_item = dialog._phonon_mode_table.item(1, 3)
    assert direction_item.text().startswith("[")
    assert direction_item.text().endswith("]")
    assert " " in direction_item.text()
    assert "Mode " not in direction_item.text()
    assert "." in direction_item.text()
    assert "[1 0 0]" not in direction_item.text()
    assert direction_item.toolTip() == ""
    covariance_box = next(
        box for box in dialog.findChildren(QGroupBox) if box.title() == "Covariance"
    )
    correlation_box = next(
        box for box in dialog.findChildren(QGroupBox) if box.title() == "Correlation"
    )
    covariance_layout = covariance_box.layout()
    correlation_layout = correlation_box.layout()
    assert covariance_layout.spacing() == SPACING.sm
    assert correlation_layout.spacing() == SPACING.sm
    assert covariance_layout.itemAt(0).alignment() & Qt.AlignmentFlag.AlignHCenter
    assert covariance_layout.itemAt(1).alignment() & Qt.AlignmentFlag.AlignHCenter
    assert correlation_layout.itemAt(0).alignment() & Qt.AlignmentFlag.AlignHCenter
    assert correlation_layout.itemAt(1).alignment() & Qt.AlignmentFlag.AlignHCenter

    help_buttons = dialog.findChildren(QToolButton, "quantumAnalysisHelpButton")
    assert len(help_buttons) == 1
    rendered_math = [label.math() for label in dialog.findChildren(MathLabel)]
    assert r"C_{ab}\;(\AA^2)" in rendered_math
    assert r"\rho_{ab}" in rendered_math

    hint_texts = [
        label.text() for label in dialog.findChildren(QLabel) if label.objectName() == "hintText"
    ]
    assert any("embedded viewer" in text for text in hint_texts)
    assert any("zero-point motion" in text for text in hint_texts)

    dialog.close()


def test_quantum_analysis_phonon_mode_viewer_animates_and_resets(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(PHONON_TEXT, encoding="utf-8")
    dialog = QuantumPhononAnalysisDialog(
        phonon_path=str(phonon_path), site_label="quantum_phonon_site"
    )

    equilibrium = np.asarray(dialog._phonon_structure.fractional_positions, dtype=float)
    current = np.asarray(dialog._phonon_viewer._structure.fractional_positions, dtype=float)
    assert np.allclose(current, equilibrium)
    dialog._phonon_show_vectors_checkbox.setChecked(True)
    dialog._phonon_viewer._camera_state = dialog._phonon_viewer._camera_state.__class__(
        23.0, 61.0, -6.0
    )
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
    assert dialog._phonon_viewer._camera_state == dialog._phonon_viewer._camera_state.__class__(
        23.0, 61.0, -6.0
    )

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


def _open_help_dialog_from_tab(
    qapp, dialog, monkeypatch, tab_index: int
) -> QuantumAnalysisHelpDialog:
    tabs = dialog.findChild(QTabWidget, "quantumAnalysisTabs")
    tabs.setCurrentIndex(tab_index)
    tab = tabs.widget(tab_index)
    opened: list[QuantumAnalysisHelpDialog] = []

    def record(help_dialog: QuantumAnalysisHelpDialog) -> int:
        opened.append(help_dialog)
        return 0

    monkeypatch.setattr(QuantumAnalysisHelpDialog, "exec", record)
    tab.findChild(QToolButton, "quantumAnalysisHelpButton").click()
    assert len(opened) == 1
    return opened[0]


def test_muon_coupling_help_describes_w2_as_harmonic_witness_estimate(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(PHONON_TEXT, encoding="utf-8")
    dialog = QuantumPhononAnalysisDialog(
        phonon_path=str(phonon_path), site_label="quantum_phonon_site"
    )

    help_dialog = _open_help_dialog_from_tab(qapp, dialog, monkeypatch, 1)

    text = " ".join(label.text() for label in help_dialog.findChildren(QLabel))
    assert "harmonic (phonon-covariance) estimate" in text
    assert "0.30(4)" in text
    assert "every stable mode" in text

    help_dialog.close()
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
    assert any(
        label.math() == r"E_0=\frac{1}{2}\hbar\omega" for label in dialog.findChildren(MathLabel)
    )

    dialog.close()


def test_quantum_analysis_dialog_has_no_mass_banner_for_a_correct_run(qapp, tmp_path: Path) -> None:
    phonon_path = tmp_path / "site.phonon"
    phonon_path.write_text(PHONON_TEXT, encoding="utf-8")

    dialog = QuantumPhononAnalysisDialog(
        phonon_path=str(phonon_path), site_label="quantum_phonon_site"
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
        phonon_path=str(phonon_path), site_label="quantum_phonon_site"
    )

    assert dialog.muon_mass_warning is not None
    banner = dialog._muon_mass_warning_label
    assert banner is not None
    assert "not the muon mass" in banner.text()
    assert banner.wordWrap() is True
    assert banner.objectName() == "warningText"
    dialog.close()


def test_quantum_analysis_help_is_a_tab_stop_only_when_it_overflows(qapp) -> None:
    dialog = QuantumAnalysisHelpDialog({"title": "Help", "paragraphs": ("Context.",)})
    dialog.show()
    QApplication.processEvents()
    scroll = dialog.findChild(QScrollArea)

    assert not scroll.focusPolicy() & Qt.FocusPolicy.TabFocus

    dialog.resize(dialog.width(), 80)
    scroll.widget().setMinimumHeight(1000)
    QApplication.processEvents()

    assert scroll.focusPolicy() & Qt.FocusPolicy.TabFocus

    dialog.close()
    dialog.deleteLater()


def test_tab_leaves_every_table_in_the_analysis_window(qapp, tmp_path: Path) -> None:
    """The phonon modes, the nearest atoms and the two coupling tensors."""

    dialog = _open_dialog(tmp_path)
    try:
        _show_every_tab(dialog)
        tables = dialog.findChildren(QTableWidget)
        assert sorted(table.accessibleName() for table in tables) == sorted(
            ("", "Muon's nearest atoms", "Covariance tensor", "Correlation tensor")
        )
        for table in tables:
            assert_tab_leaves_table(table)
    finally:
        dialog.close()


def test_a_run_whose_muon_block_cannot_be_built_still_opens_and_says_why(
    qapp, tmp_path: Path
) -> None:
    """Halved eigenvectors are not a complete set: the principal line reads a dash and why."""

    calculation, _matrix = _incomplete_toy()
    phonon_path = tmp_path / "halved.phonon"
    phonon_path.write_text(phonon_calculation_text(calculation), encoding="utf-8")

    dialog = QuantumPhononAnalysisDialog(phonon_path=str(phonon_path), site_label="site")

    assert dialog.principal_zpe_text() == "Muon ZPE along the principal axes of its well: –"
    assert "complete orthonormal set" in dialog._principal_zpe_label.toolTip()
    dialog.close()


def _incomplete_toy():
    from dataclasses import replace

    import numpy as np

    from mufinder.domain.constants import MUON_MASS_AMU
    from tests.support.phonon_fixtures import (
        phonon_calculation_from_dynamical_matrix,
        uncoupled_muon_dynamical_matrix,
    )

    matrix = uncoupled_muon_dynamical_matrix(
        np.diag([2000.0, 3000.0, 4000.0]) ** 2, (300.0,), muon_atom=1
    )
    calculation = phonon_calculation_from_dynamical_matrix(
        matrix,
        lattice_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
        species=("Si", "H:mu"),
        fractional_positions=((0.0, 0.0, 0.0), (0.25, 0.25, 0.25)),
        masses_amu=(28.0855, MUON_MASS_AMU),
    )
    gamma = calculation.gamma_point()
    return replace(
        calculation, qpoints=(replace(gamma, eigenvectors=gamma.eigenvectors * 0.5),)
    ), matrix
