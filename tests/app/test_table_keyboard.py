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
    QDialog,
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
from mufinder.app.table_keyboard import TabStopTable, keep_own_tab_order
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
from tests.support.qt import assert_tab_leaves_table, buttons_left_to_right, press_key
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


def _two_columns() -> tuple[QWidget, list[QWidget]]:
    """A window of two columns whose controls are made in reverse screen order.

    Left column: edits ``a`` over ``b``; right column: buttons ``c`` over ``d``.
    """

    from PySide6.QtWidgets import QHBoxLayout

    window = QWidget()
    left, right = QWidget(window), QWidget(window)
    d, c = QPushButton("d", right), QPushButton("c", right)
    b, a = QLineEdit(left), QLineEdit(left)
    for column, controls in ((left, (a, b)), (right, (c, d))):
        column_layout = QVBoxLayout(column)
        for control in controls:
            column_layout.addWidget(control)
    row = QHBoxLayout(window)
    row.addWidget(left)
    row.addWidget(right)
    return window, [a, b, c, d]


def test_screen_tab_order_finishes_a_panel_before_the_next(qapp) -> None:
    from mufinder.app.table_keyboard import screen_tab_order

    window, (a, b, c, d) = _two_columns()

    assert screen_tab_order(window) == [a, b, c, d]


def test_set_screen_tab_order_chains_tab_by_the_screen_and_keeps_the_opening_control(
    qapp,
) -> None:
    from mufinder.app.table_keyboard import set_screen_tab_order

    window, (a, b, c, d) = _two_columns()
    set_screen_tab_order(window)
    _show(window)
    try:
        # Qt opens on the control made first; the order does not move it.
        assert QApplication.focusWidget() is d
        assert [_tab() for _ in range(4)] == [a, b, c, d]
        assert [_shift_tab() for _ in range(4)] == [c, b, a, d]
    finally:
        window.close()


def test_set_screen_tab_order_puts_a_part_of_a_window_before_the_control_after_it(qapp) -> None:
    from PySide6.QtWidgets import QHBoxLayout

    from mufinder.app.table_keyboard import set_screen_tab_order

    window = QWidget()
    before = QLineEdit(window)
    panel = QWidget(window)
    after = QLineEdit(window)
    # Made last, so Qt's own order has them after ``after``.
    second, first = QPushButton("second", panel), QPushButton("first", panel)
    panel_layout = QHBoxLayout(panel)
    panel_layout.addWidget(first)
    panel_layout.addWidget(second)
    layout = QVBoxLayout(window)
    for widget in (before, panel, after):
        layout.addWidget(widget)
    set_screen_tab_order(panel, followed_by=after)
    _show(window)
    try:
        before.setFocus()
        assert [_tab() for _ in range(4)] == [first, second, after, before]
    finally:
        window.close()


def test_screen_tab_order_keeps_a_marked_panel_and_a_button_box_in_their_own_order(qapp) -> None:
    from PySide6.QtWidgets import QDialogButtonBox

    from mufinder.app.table_keyboard import screen_tab_order

    window = QWidget()
    block = QWidget(window)
    upper, lower = QLineEdit(block), QLineEdit(block)
    block_layout = QVBoxLayout(block)
    block_layout.addWidget(upper)
    block_layout.addWidget(lower)
    hidden = QLineEdit(window)
    hidden.hide()
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok
        | QDialogButtonBox.StandardButton.Cancel
        | QDialogButtonBox.StandardButton.Help,
        window,
    )
    layout = QVBoxLayout(window)
    for widget in (block, hidden, buttons):
        layout.addWidget(widget)
    QWidget.setTabOrder(lower, upper)
    keep_own_tab_order(block, "Lower first, by hand.")
    window.show()
    try:
        order = screen_tab_order(window)

        assert order == [lower, upper, *buttons_left_to_right(buttons)]
    finally:
        window.close()


def test_set_screen_tab_order_never_walks_the_rest_of_the_window(qapp, monkeypatch) -> None:
    """It reads only the part it orders: walking a whole window's focus chain wraps
    every widget in it, which has crashed a main window holding 3D viewers."""

    from PySide6.QtWidgets import QHBoxLayout

    from mufinder.app.table_keyboard import set_screen_tab_order

    window = QWidget()
    layout = QVBoxLayout(window)
    others = [QLineEdit(window) for _ in range(50)]
    panel = QWidget(window)
    after = QLineEdit(window)
    second, first = QPushButton("second", panel), QPushButton("first", panel)
    panel_layout = QHBoxLayout(panel)
    panel_layout.addWidget(first)
    panel_layout.addWidget(second)
    for widget in (*others, panel, after):
        layout.addWidget(widget)
    steps = []
    for name in ("nextInFocusChain", "previousInFocusChain"):
        original = getattr(QWidget, name)

        def counted(self, _original=original):
            steps.append(self)
            return _original(self)

        monkeypatch.setattr(QWidget, name, counted)

    set_screen_tab_order(panel, followed_by=after)

    assert len(steps) < 5
    monkeypatch.undo()
    chain, widget = [], window.nextInFocusChain()
    while widget is not window:
        chain.append(widget)
        widget = widget.nextInFocusChain()
    focusable = [widget for widget in chain if widget in (*others, first, second, after)]
    assert focusable == [*others, first, second, after]


def test_a_checkable_group_box_comes_before_the_controls_in_it(qapp) -> None:
    from PySide6.QtWidgets import QGroupBox

    from mufinder.app.table_keyboard import set_screen_tab_order

    window = QWidget()
    group = QGroupBox("Group", window)
    group.setCheckable(True)
    inside = QLineEdit(group)
    QVBoxLayout(group).addWidget(inside)
    above = QLineEdit(window)
    layout = QVBoxLayout(window)
    layout.addWidget(above)
    layout.addWidget(group)
    set_screen_tab_order(window)
    _show(window)
    try:
        above.setFocus()
        assert [_tab() for _ in range(3)] == [group, inside, above]
    finally:
        window.close()


def test_a_part_with_one_stop_still_goes_before_the_control_after_it(qapp) -> None:
    from mufinder.app.table_keyboard import set_screen_tab_order

    window = QWidget()
    before = QLineEdit(window)
    panel = QWidget(window)
    after = QLineEdit(window)
    only = QPushButton("only", panel)
    QVBoxLayout(panel).addWidget(only)
    layout = QVBoxLayout(window)
    for widget in (before, panel, after):
        layout.addWidget(widget)
    set_screen_tab_order(panel, followed_by=after)
    _show(window)
    try:
        before.setFocus()
        assert [_tab() for _ in range(3)] == [only, after, before]
    finally:
        window.close()


def test_a_button_box_keeps_its_own_order_right_to_left(qapp) -> None:
    from PySide6.QtWidgets import QDialogButtonBox

    from mufinder.app.table_keyboard import screen_tab_order

    window = QWidget()
    window.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, window
    )
    QVBoxLayout(window).addWidget(buttons)
    window.show()
    try:
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        own = [ok, cancel] if ok.nextInFocusChain() is cancel else [cancel, ok]
        # Right to left, the box's own order is not the order left to right on screen.
        assert own != buttons_left_to_right(buttons)

        assert screen_tab_order(window) == own
    finally:
        window.close()


def test_a_tab_widget_is_one_stop_and_keeps_its_own_way_through_its_pages(qapp) -> None:
    from PySide6.QtWidgets import QTabWidget

    from mufinder.app.table_keyboard import set_screen_tab_order

    window = QWidget()
    above = QLineEdit(window)
    tabs = QTabWidget(window)
    pages = [QWidget(), QWidget()]
    fields = [QLineEdit(page) for page in pages]
    for index, page in enumerate(pages):
        QVBoxLayout(page).addWidget(fields[index])
        tabs.addTab(page, f"Page {index + 1}")
    below = QLineEdit(window)
    layout = QVBoxLayout(window)
    for widget in (above, tabs, below):
        layout.addWidget(widget)
    set_screen_tab_order(window)
    _show(window)
    try:
        above.setFocus()
        assert [_tab() for _ in range(4)] == [tabs.tabBar(), fields[0], below, above]
        tabs.setCurrentIndex(1)
        above.setFocus()
        assert [_tab() for _ in range(4)] == [tabs.tabBar(), fields[1], below, above]
    finally:
        window.close()


def test_set_screen_tab_order_keeps_the_opening_control_when_the_first_is_disabled(qapp) -> None:
    from mufinder.app.table_keyboard import set_screen_tab_order

    window, (a, b, c, d) = _two_columns()
    d.setEnabled(False)
    set_screen_tab_order(window)
    _show(window)
    try:
        assert QApplication.focusWidget() is c
    finally:
        window.close()


# ------------------------------------------------------------- the central hook


class _HookedDialog(QDialog):
    """A dialog the hook orders: its class is defined in a MuFinder module."""


_HookedDialog.__module__ = "mufinder.test_hook"  # not mufinder.app: not a registry dialog

# Made B, A, C: from A, Qt's creation order Tabs to C, then B.
_CREATION_FROM_A = ["C", "B", "A"]
_SCREEN_FROM_A = ["B", "C", "A"]


def _hooked_dialog(*, mark: str | None = None, mark_panel: bool = False):
    """A dialog whose line edits A, B, C read top to bottom but were made B, A, C."""

    dialog = _HookedDialog()
    if mark is not None:
        keep_own_tab_order(dialog, mark)
    panel = QWidget(dialog)
    b, a, c = QLineEdit(panel), QLineEdit(panel), QLineEdit(panel)
    column = QVBoxLayout(panel)
    for edit, name in ((a, "A"), (b, "B"), (c, "C")):
        edit.setAccessibleName(name)
        column.addWidget(edit)
    QVBoxLayout(dialog).addWidget(panel)
    if mark_panel:
        keep_own_tab_order(panel, "Made in the order Tab takes.")
    return dialog, a, b


def _tab_names(start: QWidget) -> list[str]:
    start.setFocus()
    QApplication.processEvents()
    return [press_key(Qt.Key.Key_Tab).accessibleName() for _ in range(3)]


def test_a_dialog_is_chained_in_screen_order_when_it_is_shown(qapp) -> None:
    dialog, a, _b = _hooked_dialog()
    _show(dialog)
    try:
        assert _tab_names(a) == _SCREEN_FROM_A
    finally:
        dialog.close()


def test_a_control_a_shown_dialog_reveals_takes_its_place_on_screen(qapp) -> None:
    dialog, a, b = _hooked_dialog()
    b.hide()
    _show(dialog)
    try:
        b.show()
        QApplication.processEvents()

        assert _tab_names(a) == _SCREEN_FROM_A
    finally:
        dialog.close()


def test_tab_pressed_before_the_event_loop_turns_follows_a_reveal(qapp) -> None:
    dialog, a, b = _hooked_dialog()
    b.hide()
    _show(dialog)
    a.setFocus()
    QApplication.processEvents()
    try:
        b.show()
        # The first Tab is sent before the event loop turns, and read before it does.
        QTest.keyClick(a, Qt.Key.Key_Tab)
        first = QApplication.focusWidget().accessibleName()

        assert [first, *(press_key(Qt.Key.Key_Tab).accessibleName() for _ in range(2))] == (
            _SCREEN_FROM_A
        )
    finally:
        dialog.close()


def test_a_dialog_shown_again_is_ordered_for_what_changed_while_it_was_hidden(qapp) -> None:
    dialog, a, b = _hooked_dialog()
    _show(dialog)
    dialog.hide()
    # Moved to the end of its column while the dialog is hidden: A, C, B on screen.
    column = b.parentWidget().layout()
    column.removeWidget(b)
    column.addWidget(b)
    other = _HookedDialog()  # another window active in between, as in the application
    _show(other)
    other.close()
    _show(dialog)
    try:
        assert _tab_names(a) == ["C", "B", "A"]
    finally:
        dialog.close()


def test_a_marked_dialog_keeps_the_order_its_controls_were_made_in(qapp) -> None:
    dialog, a, _b = _hooked_dialog(mark="Its order is set by hand.")
    _show(dialog)
    try:
        assert _tab_names(a) == _CREATION_FROM_A
    finally:
        dialog.close()


def test_a_marked_panel_keeps_its_own_order_inside_a_hooked_dialog(qapp) -> None:
    dialog, a, _b = _hooked_dialog(mark_panel=True)
    _show(dialog)
    try:
        assert _tab_names(a) == _CREATION_FROM_A
    finally:
        dialog.close()


def test_qt_dialogs_are_left_alone(qapp) -> None:
    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox()
    before = box.nextInFocusChain()
    _show(box)
    try:
        assert box.nextInFocusChain() is before
    finally:
        box.close()


def _chain_names(start: QWidget) -> list[str]:
    """The next three named controls after ``start`` in the focus chain, as Tab takes them."""

    names, widget = [], start.nextInFocusChain()
    while widget is not start:
        if widget.accessibleName():
            names.append(widget.accessibleName())
        widget = widget.nextInFocusChain()
    return [*names, start.accessibleName()][:3]


def test_a_dialog_shown_without_activation_is_ordered_once_it_is_activated(qapp) -> None:
    dialog, a, _b = _hooked_dialog()
    # The offscreen platform the tests run on activates a shown window later,
    # through the event loop; a platform that activates it at once orders it then.
    dialog.show()
    try:
        if QApplication.activeWindow() is not dialog:
            assert _chain_names(a) == _CREATION_FROM_A

        dialog.activateWindow()
        assert QTest.qWaitForWindowActive(dialog)

        assert _chain_names(a) == _SCREEN_FROM_A
    finally:
        dialog.close()


_FRESH_APPLICATION_SCRIPT = """
from PySide6.QtCore import QObject, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QVBoxLayout

installed_on = []
original = QObject.installEventFilter


def recording(self, watcher):
    installed_on.append(self)
    return original(self, watcher)


QObject.installEventFilter = recording
app = QApplication([])
from mufinder.app.table_keyboard import install_screen_tab_order

install_screen_tab_order(app)


class Dialog(QDialog):
    pass


Dialog.__module__ = "mufinder.fresh"
dialog = Dialog()
b, a, c = QLineEdit(dialog), QLineEdit(dialog), QLineEdit(dialog)
layout = QVBoxLayout(dialog)
for edit in (a, b, c):
    layout.addWidget(edit)
dialog.show()
dialog.activateWindow()
assert QTest.qWaitForWindowActive(dialog)
app.processEvents()
a.setFocus()
app.processEvents()
QTest.keyClick(a, Qt.Key.Key_Tab)
assert app.focusWidget() is b, "the dialog is not in screen order"
assert installed_on == [], "the hook installed an event filter"
print("ok")
"""


def test_the_hook_installs_no_event_filter_on_the_application() -> None:
    """Nothing of the hook runs for every event in the application.

    It runs in a fresh process, so the test session's own hook plays no part.
    """

    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _FRESH_APPLICATION_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_ordering_a_window_walks_only_a_bounded_part_of_its_focus_chain(qapp, monkeypatch) -> None:
    """Finding where a window opens never walks its whole focus chain."""

    from PySide6.QtWidgets import QLabel

    from mufinder.app.table_keyboard import set_screen_tab_order

    window = QWidget()
    layout = QVBoxLayout(window)
    for index in range(400):
        layout.addWidget(QLabel(f"label {index}", window))
    b, a = QLineEdit(window), QLineEdit(window)
    for edit, name in ((a, "A"), (b, "B")):
        edit.setAccessibleName(name)
        layout.addWidget(edit)
    window.resize(300, 9000)
    window.show()
    steps = []
    original = QWidget.nextInFocusChain

    def counted(self):
        steps.append(self)
        return original(self)

    try:
        with monkeypatch.context() as patched:
            patched.setattr(QWidget, "nextInFocusChain", counted)
            set_screen_tab_order(window)

        # The chain holds over 400 widgets before the first stop; the walk gives up at 200.
        assert len(steps) <= 201
        assert _chain_names(a)[:1] == ["B"]
    finally:
        window.close()


def test_a_reading_scroll_area_keeps_its_place_when_its_content_starts_to_overflow(qapp) -> None:
    from PySide6.QtWidgets import QDialogButtonBox, QLabel

    from mufinder.app.theme import reading_scroll_area

    dialog = _HookedDialog()
    search = QLineEdit(dialog)
    search.setAccessibleName("Search")
    area = reading_scroll_area(dialog)
    area.setAccessibleName("Reading")
    content = QWidget()
    column = QVBoxLayout(content)
    for index in range(20):
        column.addWidget(QLabel(f"line {index}", content))
    area.setWidget(content)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
    layout = QVBoxLayout(dialog)
    for widget in (search, area, buttons):
        layout.addWidget(widget)
    dialog.resize(400, 1200)
    _show(dialog)
    try:
        assert not area.focusPolicy() & Qt.FocusPolicy.TabFocus
        dialog.resize(400, 200)
        QApplication.processEvents()
        assert area.focusPolicy() & Qt.FocusPolicy.TabFocus
        search.setFocus()
        QApplication.processEvents()

        assert press_key(Qt.Key.Key_Tab) is area
    finally:
        dialog.close()


def test_a_marked_panel_keeps_its_order_when_other_widgets_sit_between_in_the_chain(qapp) -> None:
    dialog = _HookedDialog()
    panel = QWidget(dialog)
    keep_own_tab_order(panel, "Set by hand.")
    sibling = QPushButton("Sibling", dialog)
    sibling.setAccessibleName("Sibling")
    a, b = QLineEdit(panel), QLineEdit(panel)
    a.setAccessibleName("A")
    b.setAccessibleName("B")
    QWidget.setTabOrder(b, a)
    column = QVBoxLayout(panel)
    column.addWidget(a)
    column.addWidget(b)
    layout = QVBoxLayout(dialog)
    layout.addWidget(panel)
    layout.addWidget(sibling)
    _show(dialog)
    try:
        b.setFocus()
        QApplication.processEvents()

        assert press_key(Qt.Key.Key_Tab) is a
    finally:
        dialog.close()


# ------------------------------------------------------------- reading the layouts


def _named(parent: QWidget, *names: str) -> list[QLineEdit]:
    """Line edits made in the order ``names`` gives, each named by its name."""

    edits = []
    for name in names:
        edit = QLineEdit(parent)
        edit.setAccessibleName(name)
        edits.append(edit)
    return edits


def _tab_names_from(start: QWidget, presses: int) -> list[str]:
    _focus(start)
    return [press_key(Qt.Key.Key_Tab).accessibleName() for _ in range(presses)]


def _ordered_and_shown(window: QWidget) -> None:
    from mufinder.app.table_keyboard import set_screen_tab_order

    window.resize(400, 300)
    set_screen_tab_order(window)
    _show(window)


def test_a_box_is_read_in_its_direction(qapp) -> None:
    from PySide6.QtWidgets import QBoxLayout, QHBoxLayout

    window = QWidget()
    b, a, c, d = _named(window, "B", "A", "C", "D")
    column = QVBoxLayout(window)
    forward = QHBoxLayout()
    forward.addWidget(a)
    forward.addWidget(b)
    backward = QHBoxLayout()
    backward.setDirection(QBoxLayout.Direction.RightToLeft)
    backward.addWidget(d)  # first item, so last on screen: C is to its left
    backward.addWidget(c)
    column.addLayout(forward)
    column.addLayout(backward)
    _ordered_and_shown(window)
    try:
        assert c.x() < d.x()
        assert _tab_names_from(a, 4) == ["B", "C", "D", "A"]
    finally:
        window.close()


def test_a_grid_is_read_by_row_then_column_with_a_spanning_cell(qapp) -> None:
    from PySide6.QtWidgets import QGridLayout

    window = QWidget()
    c, b, a, d = _named(window, "C", "B", "A", "D")
    grid = QGridLayout(window)
    grid.addWidget(c, 1, 0, 1, 2)
    grid.addWidget(b, 0, 1)
    grid.addWidget(a, 0, 0)
    grid.addWidget(d, 2, 1)
    _ordered_and_shown(window)
    try:
        assert _tab_names_from(a, 4) == ["B", "C", "D", "A"]
    finally:
        window.close()


def test_a_form_is_read_by_row_label_side_first(qapp) -> None:
    from PySide6.QtWidgets import QCheckBox, QFormLayout

    window = QWidget()
    option = QCheckBox("Option", window)
    option.setAccessibleName("Option")
    first, second = _named(window, "First", "Second")
    form = QFormLayout(window)
    form.addRow("First", first)
    form.addRow(option, second)
    _ordered_and_shown(window)
    try:
        assert _tab_names_from(first, 3) == ["Option", "Second", "First"]
    finally:
        window.close()


def test_a_splitter_is_read_pane_by_pane(qapp) -> None:
    from PySide6.QtWidgets import QSplitter

    window = QWidget()
    splitter = QSplitter(window)
    right_pane, left_pane = QWidget(), QWidget()
    right_edits = _named(right_pane, "R1", "R2")
    left_edits = _named(left_pane, "L2", "L1")[::-1]
    for pane, edits in ((right_pane, right_edits), (left_pane, left_edits)):
        column = QVBoxLayout(pane)
        for edit in edits:
            column.addWidget(edit)
    splitter.addWidget(right_pane)
    splitter.insertWidget(0, left_pane)
    QVBoxLayout(window).addWidget(splitter)
    _ordered_and_shown(window)
    try:
        assert _tab_names_from(left_edits[0], 4) == ["L2", "R1", "R2", "L1"]
    finally:
        window.close()


def test_a_hidden_control_keeps_its_place_and_is_reached_once_shown(qapp) -> None:
    from PySide6.QtWidgets import QHBoxLayout

    window = QWidget()
    c, b, a = _named(window, "C", "B", "A")
    row = QHBoxLayout()
    row.addWidget(b)
    row.addWidget(c)
    column = QVBoxLayout(window)
    column.addWidget(a)
    column.addLayout(row)
    b.hide()
    _ordered_and_shown(window)
    try:
        assert _tab_names_from(a, 2) == ["C", "A"]
        b.show()
        QApplication.processEvents()

        assert _tab_names_from(a, 3) == ["B", "C", "A"]
    finally:
        window.close()


def test_a_child_no_layout_places_is_read_by_where_it_is(qapp) -> None:
    from PySide6.QtWidgets import QToolButton

    window = QWidget()
    second, first = _named(window, "Second", "First")
    corner = QToolButton(window)  # placed by hand above the laid-out fields
    corner.setAccessibleName("Corner")
    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 40, 0, 0)
    layout.addWidget(first)
    layout.addWidget(second)
    corner.move(0, 0)
    _ordered_and_shown(window)
    try:
        assert _tab_names_from(first, 3) == ["Second", "Corner", "First"]
    finally:
        window.close()


def test_a_button_box_is_read_in_the_order_it_lays_its_buttons_out(qapp) -> None:
    from PySide6.QtWidgets import QDialogButtonBox

    window = QWidget()
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, window
    )
    field = _named(window, "Field")[0]
    layout = QVBoxLayout(window)
    layout.addWidget(field)
    layout.addWidget(buttons)
    _ordered_and_shown(window)
    try:
        _focus(field)
        assert [_tab() for _ in range(3)] == [*buttons_left_to_right(buttons), field]
    finally:
        window.close()


def test_a_control_placed_by_hand_between_two_rows_is_read_between_them(qapp) -> None:
    from PySide6.QtWidgets import QToolButton

    from mufinder.app.table_keyboard import set_screen_tab_order

    window = QWidget()
    lower, upper = _named(window, "Lower", "Upper")
    between = QToolButton(window)
    between.setAccessibleName("Between")
    layout = QVBoxLayout(window)
    layout.setSpacing(60)
    layout.addWidget(upper)
    layout.addWidget(lower)
    _ordered_and_shown(window)
    try:
        between.move(0, (upper.geometry().bottom() + lower.geometry().top()) // 2 - 5)
        between.resize(20, 10)
        set_screen_tab_order(window)

        assert _tab_names_from(upper, 3) == ["Between", "Lower", "Upper"]
    finally:
        window.close()


def test_a_window_does_not_open_on_a_hidden_control(qapp) -> None:
    window = QWidget()
    hidden, a, b = _named(window, "Hidden", "A", "B")
    layout = QVBoxLayout(window)
    for edit in (hidden, a, b):
        layout.addWidget(edit)
    hidden.hide()
    _ordered_and_shown(window)
    try:
        assert QApplication.focusWidget() is a
    finally:
        window.close()


def test_a_marked_window_is_not_reordered_by_a_direct_call(qapp) -> None:
    from mufinder.app.table_keyboard import set_screen_tab_order

    window = QWidget()
    b, a, c = _named(window, "B", "A", "C")
    layout = QVBoxLayout(window)
    for edit in (a, b, c):
        layout.addWidget(edit)
    keep_own_tab_order(window, "Set by hand.")
    set_screen_tab_order(window)
    _show(window)
    try:
        assert _chain_names(a)[:1] == ["C"]  # still the order they were made in
    finally:
        window.close()


def test_the_mark_needs_a_reason(qapp) -> None:
    with pytest.raises(ValueError, match="reason"):
        keep_own_tab_order(QWidget(), "")


def test_a_box_in_a_widget_laid_out_right_to_left_is_read_as_drawn(qapp) -> None:
    from PySide6.QtWidgets import QHBoxLayout

    window = QWidget()
    window.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    a, b, c = _named(window, "A", "B", "C")
    row = QHBoxLayout(window)
    for edit in (a, b, c):
        row.addWidget(edit)
    _ordered_and_shown(window)
    try:
        assert c.x() < b.x() < a.x()
        assert _tab_names_from(c, 3) == ["B", "A", "C"]
    finally:
        window.close()


def test_the_dipolar_tensor_results_take_their_place_when_calculated_while_open(qapp) -> None:
    from mufinder.app.magnetic_analysis_dialogs import DipolarTensorDialog
    from tests.support.keyboard import screen_order_faults, tab_walk
    from tests.support.qt import open_and_take_focus

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
    open_and_take_focus(dialog)
    try:
        calculate = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Calculate")
        calculate.click()
        QApplication.processEvents()
        calculate.setFocus()
        QApplication.processEvents()

        walk = tab_walk(dialog)

        assert any(isinstance(stop, TabStopTable) for stop in walk.stops)
        assert screen_order_faults(walk.stops, dialog) == []
    finally:
        dialog.close()
