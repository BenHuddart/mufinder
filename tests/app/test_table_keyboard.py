"""Tab leaves every table: one stop on the way round the window, however many rows.

Every test sends real key presses with ``QTest`` to whatever has focus in a shown,
active window, and lets go of Shift before anything reads the modifiers.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import re
from uuid import uuid4

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QInputMethodEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.host_structure_widget import HostStructureWidget
from mufinder.app.hubbard_u_editor import HubbardUDialog
from mufinder.app.relaxation_widget import PristineRelaxationWidget
from mufinder.app.table_keyboard import TabStopTable
from mufinder.app.workflow_table_utils import create_workflow_table
from mufinder.science.magnetic_structure import MagneticStructureModel
from mufinder.services.project_service import ProjectService
from tests.support.magnetic_field_sites import feo_host_structure
from tests.support.magnetic_fields import (
    coo_structure,
    custom_muon_site,
    two_row_helix_model,
    z_moment_component,
)
from tests.support.qt import assert_tab_leaves_table, press_key
from tests.support.structures import magnetic_silicon_oxide_host


def _tab() -> QWidget | None:
    return press_key(Qt.Key.Key_Tab)


def _shift_tab() -> QWidget | None:
    return press_key(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)


def _show(top: QWidget) -> None:
    top.show()
    top.activateWindow()
    QTest.qWaitForWindowActive(top)
    QApplication.processEvents()


def _focus(widget: QWidget) -> None:
    widget.setFocus(Qt.FocusReason.OtherFocusReason)
    QApplication.processEvents()
    assert QApplication.focusWidget() is widget


def _in_table(table: QTableWidget, widget: QWidget | None) -> bool:
    return widget is table or (widget is not None and table.isAncestorOf(widget))


class _Form(QWidget):
    """A line edit, a table and a line edit, in that order on the Tab path."""

    def __init__(self, table: QTableWidget) -> None:
        super().__init__()
        self.before = QLineEdit(self)
        self.before.setAccessibleName("Before")
        self.table = table
        table.setParent(self)
        self.after = QLineEdit(self)
        self.after.setAccessibleName("After")
        layout = QVBoxLayout(self)
        for widget in (self.before, table, self.after):
            layout.addWidget(widget)
        # The window's Tab path follows creation order; put the table between the two.
        QWidget.setTabOrder(self.before, table)
        QWidget.setTabOrder(table, self.after)


def _plain_rows(table: QTableWidget, rows: int = 5) -> None:
    table.setRowCount(rows)
    for row in range(rows):
        for column in range(table.columnCount()):
            table.setItem(row, column, QTableWidgetItem(f"{row}.{column}"))
    table.setCurrentCell(0, 0)


@pytest.fixture
def plain_form(qapp):
    table = create_workflow_table(("A", "B", "C"), editable=True, accessible_name="Plain")
    _plain_rows(table)
    form = _Form(table)
    _show(form)
    yield form
    form.close()


def test_tab_leaves_a_table_of_plain_rows_for_the_next_control(plain_form) -> None:
    _focus(plain_form.table)

    assert _tab() is plain_form.after
    assert _tab() is plain_form.before
    assert _tab() is plain_form.table


def test_shift_tab_leaves_a_table_of_plain_rows_for_the_control_before(plain_form) -> None:
    _focus(plain_form.table)

    assert _shift_tab() is plain_form.before
    assert _shift_tab() is plain_form.after
    assert _shift_tab() is plain_form.table


def test_arrow_keys_move_between_cells_and_keep_focus_on_the_table(plain_form) -> None:
    table = plain_form.table
    table.setCurrentCell(1, 1)
    _focus(table)

    assert press_key(Qt.Key.Key_Down) is table
    assert (table.currentRow(), table.currentColumn()) == (2, 1)
    assert press_key(Qt.Key.Key_Right) is table
    assert (table.currentRow(), table.currentColumn()) == (2, 2)
    assert press_key(Qt.Key.Key_Up) is table
    assert press_key(Qt.Key.Key_Left) is table
    assert (table.currentRow(), table.currentColumn()) == (1, 1)


def test_tab_in_a_cell_editor_commits_the_edit_and_edits_the_next_cell(plain_form) -> None:
    table = plain_form.table
    table.setCurrentCell(2, 0)
    _focus(table)
    press_key(Qt.Key.Key_F2)
    editor = QApplication.focusWidget()
    assert isinstance(editor, QLineEdit) and table.isAncestorOf(editor)
    editor.selectAll()
    QTest.keyClicks(editor, "new")

    _tab()

    assert table.item(2, 0).text() == "new"
    assert (table.currentRow(), table.currentColumn()) == (2, 1)
    assert table.state() == QTableWidget.State.EditingState
    # Escape ends that edit, and Tab then leaves the table.
    press_key(Qt.Key.Key_Escape)
    assert QApplication.focusWidget() is table
    assert _tab() is plain_form.after


@pytest.fixture
def controls_form(qapp):
    """Rows of a combo and a button in cells, beside a cell of text."""

    table = create_workflow_table(("Name", "Choice", "Action"), accessible_name="Controls")
    table.setRowCount(4)
    for row in range(4):
        table.setItem(row, 0, QTableWidgetItem(f"row {row}"))
        combo = QComboBox()
        combo.addItems(["one", "two"])
        combo.setAccessibleName(f"Choice {row}")
        table.setCellWidget(row, 1, combo)
        button = QPushButton(f"Act {row}")
        table.setCellWidget(row, 2, button)
    table.setCurrentCell(2, 0)
    form = _Form(table)
    _show(form)
    yield form
    form.close()


def test_tab_goes_through_the_current_rows_controls_and_out(controls_form) -> None:
    table = controls_form.table
    _focus(table)

    assert _tab() is table.cellWidget(2, 1)
    assert _tab() is table.cellWidget(2, 2)
    assert _tab() is controls_form.after
    assert _tab() is controls_form.before
    # Tab from elsewhere arrives at the table, never at a control in a row.
    assert _tab() is table


def test_shift_tab_comes_back_through_the_row_to_the_table_and_out(controls_form) -> None:
    table = controls_form.table
    _focus(controls_form.after)

    assert _shift_tab() is table
    assert _tab() is table.cellWidget(2, 1)
    assert _tab() is table.cellWidget(2, 2)
    assert _shift_tab() is table.cellWidget(2, 1)
    assert _shift_tab() is table
    assert _shift_tab() is controls_form.before


def test_tab_skips_a_disabled_or_hidden_control_in_the_row(controls_form) -> None:
    table = controls_form.table
    table.cellWidget(2, 1).setEnabled(False)
    _focus(table)

    assert _tab() is table.cellWidget(2, 2)
    table.cellWidget(2, 2).hide()
    table.cellWidget(2, 1).setEnabled(True)
    _focus(table)
    assert _tab() is table.cellWidget(2, 1)
    assert _tab() is controls_form.after


def test_tab_from_a_clicked_control_on_another_row_leaves_from_that_row(controls_form) -> None:
    table = controls_form.table
    button = table.cellWidget(0, 2)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert QApplication.focusWidget() is button

    assert _tab() is controls_form.after
    assert _shift_tab() is table
    assert table.currentRow() == 0


def test_a_control_disabled_while_focused_passes_focus_along_its_row(controls_form) -> None:
    table = controls_form.table
    _focus(table)
    assert _tab() is table.cellWidget(2, 1)

    table.cellWidget(2, 1).setEnabled(False)
    QApplication.processEvents()

    assert QApplication.focusWidget() is table.cellWidget(2, 2)


def test_the_last_control_in_a_row_disabled_while_focused_returns_focus_to_the_table(
    controls_form,
) -> None:
    """Focus stays where the user was working, not on the control after the table."""

    table = controls_form.table
    table.cellWidget(2, 1).setEnabled(False)
    _focus(table)
    assert _tab() is table.cellWidget(2, 2)

    table.cellWidget(2, 2).setEnabled(False)
    QApplication.processEvents()

    assert QApplication.focusWidget() is table


def test_tab_from_a_control_in_a_column_hidden_while_focused_returns_to_the_table(
    controls_form,
) -> None:
    """Qt leaves focus on a control whose column it hides; Tab then goes to the table."""

    table = controls_form.table
    _focus(table)
    assert _tab() is table.cellWidget(2, 1)
    table.setColumnHidden(1, True)
    QApplication.processEvents()

    assert _tab() is table
    assert _tab() is table.cellWidget(2, 2)


def test_tab_from_the_table_skips_an_open_cell_editor(plain_form) -> None:
    table = plain_form.table
    table.setCurrentCell(1, 1)
    _focus(table)
    press_key(Qt.Key.Key_F2)
    assert table.state() == QTableWidget.State.EditingState
    _focus(table)

    assert _tab() is plain_form.after


def test_tab_leaves_the_host_structure_atoms_table(qapp) -> None:
    """The issue's case: 40 presses used to walk the cells and never leave."""

    page = HostStructureWidget()
    page.set_structure(magnetic_silicon_oxide_host())
    table = page._atom_table
    assert table.rowCount() >= 2
    _show(page)
    try:
        _focus(table)
        after = _tab()
        assert not _in_table(table, after)
        _focus(table)
        assert not _in_table(table, _shift_tab())
        # Round the page and back to the table.
        for _ in range(40):
            if _tab() is table:
                break
        assert QApplication.focusWidget() is table
    finally:
        page.close()


def test_tab_commits_a_spin_edit_on_the_atoms_table(qapp) -> None:
    page = HostStructureWidget()
    page.set_structure(magnetic_silicon_oxide_host())
    table = page._atom_table
    spin_column = table.columnCount() - 1
    _show(page)
    try:
        table.setCurrentCell(0, spin_column)
        _focus(table)
        press_key(Qt.Key.Key_F2)
        editor = QApplication.focusWidget()
        assert isinstance(editor, QLineEdit)
        editor.selectAll()
        QTest.keyClicks(editor, "0.5")
        _tab()
        assert float(table.item(0, spin_column).text()) == 0.5
    finally:
        page.close()


def test_tab_leaves_the_hubbard_u_table_for_add_row(qapp) -> None:
    dialog = HubbardUDialog(species_options=("Fe", "O"))
    dialog._add_blank_row()
    table = dialog._table
    table.setCurrentCell(1, 0)
    _show(dialog)
    try:
        _focus(table)
        assert _tab() is table.cellWidget(1, 0)
        assert _tab() is table.cellWidget(1, 1)
        assert _tab() is table.cellWidget(1, 2)
        assert _tab() is dialog._add_button
        assert _shift_tab() is table
        assert not _in_table(table, _shift_tab())
    finally:
        dialog.close()


def test_the_empty_input_method_event_a_cell_line_edit_leaves_behind_keeps_focus(qapp) -> None:
    """Focus that left the U field stays on the table when the field's input method resets.

    The reset reaches the table a moment after Shift+Tab, as an input method event with
    no text; Qt's view would take it for typing and move focus back into the field.
    """

    dialog = HubbardUDialog(species_options=("Fe", "O"))
    table = dialog._table
    table.setCurrentCell(0, 2)
    _show(dialog)
    try:
        _focus(table)
        QApplication.sendEvent(table, QInputMethodEvent())
        QApplication.processEvents()
        assert QApplication.focusWidget() is table
    finally:
        dialog.close()


def test_text_from_an_input_method_still_starts_editing_a_cell(plain_form) -> None:
    table = plain_form.table
    table.setCurrentCell(1, 1)
    _focus(table)
    event = QInputMethodEvent()
    event.setCommitString("é")
    QApplication.sendEvent(table, event)
    QApplication.processEvents()

    assert table.state() == QTableWidget.State.EditingState


def test_tab_leaves_a_workflow_row_table_after_the_current_rows_controls(qapp) -> None:
    page = PristineRelaxationWidget()
    page._add_row()
    page._add_row()
    table = page._table
    table.setCurrentCell(1, 0)
    _show(page)
    try:
        _focus(table)
        stops = [_tab() for _ in range(table.columnCount() + 1)]
        inside = [stop for stop in stops if _in_table(table, stop)]
        assert all(cell_row(table, stop) == 1 for stop in inside)
        assert not _in_table(table, stops[len(inside)])
        # Every control Tab reached on main: the row's Remove button has Tab focus only.
        assert [stop.accessibleName() for stop in inside] == [
            "Supercell for relax_2",
            "Fix cell for relax_2",
            "Configure relax_2",
            "Remove relax_2",
        ]
    finally:
        page.close()


def test_shift_tab_lands_on_the_table_after_a_click_that_starts_a_delayed_edit(qapp) -> None:
    """A second click on a selected cell arms Qt's delayed edit of the current cell.

    Tab and Shift+Tab move the current cell to the row's last control before the delay
    runs out; the delayed edit must not then pull focus into that control.
    """

    page = PristineRelaxationWidget()
    page._add_row()
    page._add_row()
    table = page._table
    _show(page)
    delay = QApplication.doubleClickInterval() + 100
    try:
        for _attempt in range(3):
            label = table.visualRect(table.model().index(1, 0)).center()
            QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=label)
            QTest.qWait(delay)
            QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=label)
            while _in_table(table, _tab()):
                pass
            assert _shift_tab() is table
            QTest.qWait(delay)
            assert QApplication.focusWidget() is table
    finally:
        page.close()


def test_a_click_on_a_cell_that_cannot_be_selected_makes_its_row_current(qapp) -> None:
    """Polarisation's Field Results rows hold items that cannot be selected."""

    table = create_workflow_table(("Name", "Weight"), accessible_name="Weights")
    table.setRowCount(2)
    for row in range(2):
        item = QTableWidgetItem(f"row {row}")
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        table.setItem(row, 0, item)
        table.setCellWidget(row, 1, QDoubleSpinBox())
    form = _Form(table)
    _show(form)
    try:
        _focus(form.before)
        cell = table.visualRect(table.model().index(1, 0)).center()
        QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=cell)
        QApplication.processEvents()

        assert table.currentRow() == 1
        assert _tab() is table.cellWidget(1, 1)
        assert _tab() is form.after
    finally:
        form.close()


def test_the_mouse_wheel_still_gives_a_spin_box_in_a_cell_focus(controls_form) -> None:
    table = controls_form.table
    spin = QDoubleSpinBox()
    table.setCellWidget(2, 1, spin)
    _focus(controls_form.before)
    centre = QPointF(spin.rect().center())
    wheel = QWheelEvent(
        centre,
        QPointF(spin.mapToGlobal(spin.rect().center())),
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(spin, wheel)
    QApplication.processEvents()

    assert QApplication.focusWidget() is spin
    # Still off the window's Tab path: Shift+Tab from after the table lands on it.
    _focus(controls_form.after)
    assert _shift_tab() is table


def cell_row(table: QTableWidget, widget: QWidget) -> int | None:
    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            cell = table.cellWidget(row, column)
            if cell is not None and (cell is widget or cell.isAncestorOf(widget)):
                return row
    return None


def test_no_app_module_builds_a_table_that_walks_its_cells_on_tab() -> None:
    """Every table comes from ``TabStopTable``; a plain ``QTableWidget`` traps Tab."""

    app_dir = Path(__file__).resolve().parents[2] / "src" / "mufinder" / "app"
    plain_table = re.compile(r"\bQTable(?:Widget|View)\(|class \w+\(QTable(?:Widget|View)\)")
    offenders = sorted(
        path.name
        for path in app_dir.rglob("*.py")
        if path.name != "table_keyboard.py" and plain_table.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == []


def _host_structure_page():
    page = HostStructureWidget()
    page.set_structure(magnetic_silicon_oxide_host())
    return page


def _page(module: str, name: str, *, rows: int = 0):
    def build():
        page = getattr(importlib.import_module(f"mufinder.app.{module}"), name)()
        for _ in range(rows):
            page._add_row()
        return page

    return build


def _muon_relaxation_page():
    from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget

    page = MuonRelaxationWidget()
    page.add_staged_sites(
        [
            {"id": str(uuid4()), "label": f"Site {index}", "fractional_position": [0.1, 0.2, 0.3]}
            for index in (1, 2)
        ]
    )
    return page


def _hubbard_u_dialog():
    dialog = HubbardUDialog(species_options=("Fe", "O"))
    dialog._add_blank_row()
    return dialog


def _block_dialog():
    from mufinder.app.block_dialog import BlockParameterDialog
    from mufinder.castep_io.schema import get_keyword_spec

    return BlockParameterDialog(get_keyword_spec("cell", "lattice_cart"))


def _cluster_setup_dialog():
    from mufinder.app.cluster_setup_dialog import ClusterSetupDialog

    class _Settings:
        def __init__(self) -> None:
            self._values: dict[str, object] = {}

        def value(self, key, default=None, value_type=None):  # noqa: ANN001
            value = self._values.get(key, default)
            return str(value) if value_type is str else value

        def setValue(self, key, value) -> None:  # noqa: N802, ANN001
            self._values[key] = value

    return ClusterSetupDialog(settings=_Settings())


def _geometry_progress_dialog():
    from mufinder.app.geometry_progress_dialog import GeometryProgressDialog

    return GeometryProgressDialog("Si_site_1", None)


def _nuclear_relaxation_dialog():
    from mufinder.app.magnetic_analysis_dialogs import NuclearRelaxationDialog

    return NuclearRelaxationDialog(
        muon_site=custom_muon_site((0.25, 0.25, 0.25)), distortion_available=False
    )


def _dipolar_tensor_dialog():
    from mufinder.app.magnetic_analysis_dialogs import DipolarTensorDialog

    tensor = {
        "site_label": "site 1",
        "muon_position": [0.25, 0.25, 0.25],
        "dipolar_tensor_T_per_muB": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
    }
    dialog = DipolarTensorDialog(
        feo_host_structure(),
        MagneticStructureModel(components=(z_moment_component("one", "One", 0),)),
        custom_muon_site((0.25, 0.25, 0.25)),
        include_equivalent_sites=False,
        calculate_callback=lambda _dialog: (tensor,),
    )
    dialog._calculate()
    return dialog


def _magnetic_structure_builder(tmp_path):
    from mufinder.app.magnetic_structure_builder import MagneticStructureBuilderDialog

    service = ProjectService()
    document = service.create_project(tmp_path / "builder.mufinder", "Builder")
    service.add_structure(document, feo_host_structure())
    dialog = MagneticStructureBuilderDialog()
    dialog.set_document(document)
    dialog._set_model(two_row_helix_model())
    return dialog


def _magnetic_texture_dialog():
    from mufinder.app.magnetic_texture_dialog import MagneticTextureDialog

    dialog = MagneticTextureDialog(coo_structure())
    dialog._mode_buttons["custom"].setChecked(True)
    return dialog


def _magnetic_component_dialog():
    from mufinder.app.magnetic_component_dialogs import MagneticComponentDialog

    return MagneticComponentDialog(coo_structure())


def _atom_phase_dialog():
    from mufinder.app.magnetic_component_dialogs import AtomPhaseDialog

    return AtomPhaseDialog(coo_structure(), (0, 2), {2: 0.25})


# Every table a page or dialog builds, by owner and the tables' accessible names. A
# table built somewhere new comes from ``TabStopTable`` (the source check above) and
# belongs in this list, or, where its owner needs builders private to the owner's
# test module, in a test there that calls ``assert_tab_leaves_table``.
TABLES = {
    "host structure": (_host_structure_page, ("Atoms",)),
    "muon generation": (
        _page("muon_generation_widget", "MuonGenerationWidget"),
        ("UEP sites", "Candidate Sites"),
    ),
    "cut-off convergence": (
        _page("convergence_widget", "CutoffConvergenceWidget"),
        ("Cutoff Sweep",),
    ),
    "k-point convergence": (
        _page("kpoint_convergence_widget", "KPointConvergenceWidget"),
        ("K-Point Grid Sweep",),
    ),
    "muon sites": (
        _page("muon_sites_summary_widget", "MuonSitesSummaryWidget"),
        ("Muon Sites", "Muon Sites"),
    ),
    "energy barriers": (_page("energy_barriers_widget", "EnergyBarriersWidget"), ("Paths",)),
    "pristine relaxation": (
        _page("relaxation_widget", "PristineRelaxationWidget", rows=2),
        ("Relaxation Jobs",),
    ),
    "muon relaxation": (
        _muon_relaxation_page,
        ("Muonium Relaxation Jobs", "Positive muon Relaxation Jobs"),
    ),
    "uep": (_page("uep_widget", "UEPWidget", rows=2), ("UEP Source Jobs",)),
    "quantum corrections": (
        _page("quantum_corrections_widget", "QuantumCorrectionsWidget"),
        ("Quantum Corrections Sites",),
    ),
    "hyperfine coupling": (
        _page("hyperfine_coupling_widget", "HyperfineCouplingWidget"),
        ("Hyperfine Jobs", "Completed Results"),
    ),
    "polarisation": (
        _page("polarisation_widget", "PolarisationWidget"),
        ("Simulations", "Field Results"),
    ),
    "magnetic fields": (
        _page("magnetic_field_widget", "MagneticFieldWidget"),
        ("Magnetic Structure Components", "Field Results", "Field Breakdown"),
    ),
    "hubbard u": (_hubbard_u_dialog, ("",)),
    "block parameter": (_block_dialog, ("",)),
    "cluster setup": (_cluster_setup_dialog, ("",)),
    "geometry progress": (_geometry_progress_dialog, ("",)),
    "nuclear relaxation": (_nuclear_relaxation_dialog, ("Element Contributions", "Results")),
    "dipolar tensor": (_dipolar_tensor_dialog, ("",)),
    "magnetic structure builder": (_magnetic_structure_builder, ("", "")),
    "magnetic texture": (_magnetic_texture_dialog, ("",)),
    "magnetic component": (_magnetic_component_dialog, ("",)),
    "atom phase": (_atom_phase_dialog, ("Per-Atom Phases",)),
}


@pytest.mark.parametrize("owner", sorted(TABLES))
def test_tab_leaves_every_table(qapp, tmp_path, monkeypatch, owner: str) -> None:
    build, names = TABLES[owner]
    had_tab_focus: dict[QWidget, list[QWidget]] = {}
    set_cell_widget = TabStopTable.setCellWidget

    def _recording(self, row, column, widget):
        had_tab_focus[widget] = _tab_focus_controls(widget)
        set_cell_widget(self, row, column, widget)

    monkeypatch.setattr(TabStopTable, "setCellWidget", _recording)
    top = build(tmp_path) if "tmp_path" in inspect.signature(build).parameters else build()
    try:
        tables = top.findChildren(QTableWidget)
        assert sorted(table.accessibleName() for table in tables) == sorted(names)
        for table in tables:
            stops = assert_tab_leaves_table(table)
            # Tab visits exactly the current row's controls that had Tab focus when
            # they were put in their cells: everything Tab reached on main.
            assert stops == _reachable_tab_controls(table, table.currentRow(), had_tab_focus)
    finally:
        top.close()


def _tab_focus_controls(widget: QWidget) -> list[QWidget]:
    if widget.focusPolicy().value & Qt.FocusPolicy.TabFocus.value:
        return [widget]
    return [
        control
        for child in widget.children()
        if isinstance(child, QWidget)
        for control in _tab_focus_controls(child)
    ]


def _reachable_tab_controls(table, row: int, had_tab_focus: dict) -> list[QWidget]:
    header = table.horizontalHeader()
    controls = []
    for visual in range(table.columnCount()):
        column = header.logicalIndex(visual)
        cell = table.cellWidget(row, column)
        if table.isColumnHidden(column) or cell is None:
            continue
        controls.extend(
            control
            for control in had_tab_focus.get(cell, [])
            if control.isEnabled() and control.isVisibleTo(table)
        )
    return controls
