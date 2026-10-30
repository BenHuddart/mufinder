"""Every visible control on every workflow page has a name a screen reader can announce.

The census opens each page of a project with the shared host structure, with
one row on every page that lists rows (a staged job, a stored field result, a
source or a result), and counts the visible
controls a screen reader could only announce by their kind ("button",
"edit", "table"). A control counts as unnamed when it is a button whose text
is empty or one character, a line edit, combo box, spin box, item view or
slider, and it has no accessible name and no ``QLabel`` buddy (a form-layout
row label is a buddy). Line edits inside a combo or spin box, and table
headers, are not counted.

``UNNAMED_CONTROL_BUDGET`` is a ratchet: a page may have at most its budget
of unnamed controls, a page that is not listed (every new page) has none, and
a budget above a page's count fails until it is lowered.
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
import shiboken6
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QWidget,
)

from mufinder.app.energy_barriers_widget import EnergyBarrierNebResultRow, EnergyBarrierSourceRow
from mufinder.app.hyperfine_coupling_widget import HyperfineResultRow, HyperfineSourceRow
from mufinder.app.quantum_corrections_widget import QuantumResultRow, QuantumSourceRow
from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    store_magnetic_field_results,
)
from tests.support.field_results import field_result

PAGE_ROUTES = (
    "host_structure",
    "supercell_editor",
    "convergence",
    "convergence_cutoff",
    "convergence_kpoint",
    "pristine_relaxation",
    "uep",
    "muon_candidates",
    "muon_relaxations",
    "muon_analysis",
    "muon_sites",
    "magnetic_fields",
    "polarisation",
    "hyperfine_coupling",
    "quantum_corrections",
    "energy_barriers",
    "provenance_log",
)

# Unnamed visible controls each page may still have. Lower an entry when a
# page's count falls; never raise one, and never add a page.
UNNAMED_CONTROL_BUDGET: dict[str, int] = {}

_COUNTED_KINDS = (
    QAbstractButton,
    QLineEdit,
    QComboBox,
    QAbstractSpinBox,
    QAbstractItemView,
    QSlider,
)


def _counted_controls(page: QWidget) -> Iterator[QWidget]:
    for widget in page.findChildren(QWidget):
        if not isinstance(widget, _COUNTED_KINDS) or not widget.isVisible():
            continue
        if isinstance(widget, QLineEdit) and isinstance(
            widget.parent(), (QComboBox, QAbstractSpinBox)
        ):
            continue
        if type(widget).__name__ == "QHeaderView":
            continue
        if type(widget).__name__ == "QAbstractButton" and isinstance(
            widget.parent(), QAbstractItemView
        ):
            continue
        yield widget


def _address(widget: QWidget) -> int:
    """The C++ object's address: wrapper ids can be reused once a wrapper is collected."""

    return shiboken6.getCppPointer(widget)[0]


def _is_unnamed(widget: QWidget, buddies: set[int]) -> bool:
    text = widget.text().strip() if isinstance(widget, QAbstractButton) else ""
    return len(text) <= 1 and not widget.accessibleName() and _address(widget) not in buddies


def _describe(widget: QWidget) -> str:
    text = widget.text() if isinstance(widget, QAbstractButton) else ""
    return f"{type(widget).__name__}(text={text!r}, tooltip={widget.toolTip()[:60]!r})"


def _stage_rows(window, document) -> None:
    """Give every page that stages job rows one row, so its row controls are counted."""

    window._convergence_view._add_row(300.0)
    window._kpoint_convergence_view._add_row((2, 2, 2))
    window._pristine_relaxation_view._add_row()
    window._uep_view._add_row()
    window._muon_relaxation_view.add_staged_sites(
        [{"id": str(uuid4()), "label": "Manual Site 1", "fractional_position": [0.1, 0.2, 0.3]}],
        charge_states=(POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE),
    )
    # Polarisation lists the stored field results when it is shown.
    store_magnetic_field_results(
        document,
        MAGNETIC_FIELD_PAGE_OWNER,
        {
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [field_result((0.0, 0.0, 0.01)).to_json()],
            }
        },
    )


def _show_hyperfine_rows(window) -> None:
    view = window._hyperfine_coupling_view
    view.set_sources([HyperfineSourceRow("mu1", "mu1", "Positive muon", "relax-mu1", "ready")])
    view.set_results(
        [HyperfineResultRow("mu1", "mu1", "Positive muon", "1.0", "0.1", "0.0", "0.1", "0.0")]
    )


def _show_quantum_rows(window) -> None:
    view = window._quantum_corrections_view
    view.set_sources(
        [QuantumSourceRow("mu1", "mu1", "Positive muon", "0.00", "9", "relax-mu1", "1", "ready")]
    )
    view.set_results([QuantumResultRow("mu1", "mu1", "0.58", "1, 2, 3", "0.1", "0.0", "completed")])


def _show_energy_barrier_rows(window) -> None:
    view = window._energy_barriers_view
    view.set_sources(
        [EnergyBarrierSourceRow("mu1-mu2", "mu1", "mu2", "Positive muon", "1.0", "ready")]
    )
    view.set_neb_results(
        [
            EnergyBarrierNebResultRow(
                "neb-1",
                "mu1 to mu2",
                "Positive muon",
                "1.0",
                "0.2",
                "0.2",
                "0.0",
                "5",
                "completed",
                "",
            )
        ]
    )


# The source pages rebuild their rows from the project when they are shown, so
# their rows are put in after the page is showing.
_SHOW_ROWS = {
    "hyperfine_coupling": _show_hyperfine_rows,
    "quantum_corrections": _show_quantum_rows,
    "energy_barriers": _show_energy_barrier_rows,
}


def _census(window, qapp, routes=PAGE_ROUTES, show_rows=None) -> dict[str, list[str]]:
    show_rows = show_rows or {}
    window.show()
    qapp.processEvents()
    unnamed: dict[str, list[str]] = {}
    try:
        for route in routes:
            assert window._find_tree_item_by_route(route) is not None, route
            window._set_active_route(route)
            if route in show_rows:
                show_rows[route](window)
            qapp.processEvents()
            page = window._main_stack.currentWidget()
            buddies = {
                _address(label.buddy()) for label in window.findChildren(QLabel) if label.buddy()
            }
            unnamed[route] = [
                _describe(widget)
                for widget in _counted_controls(page)
                if _is_unnamed(widget, buddies)
            ]
    finally:
        # A shown window with a project asks to save on close; the census
        # changes nothing worth saving.
        window.hide()
    return unnamed


@pytest.fixture
def unnamed_controls_by_page(project_with_host, qapp) -> dict[str, list[str]]:
    window, _service, document = project_with_host
    _stage_rows(window, document)
    return _census(window, qapp, show_rows=_SHOW_ROWS)


def test_no_page_has_more_unnamed_controls_than_its_budget(unnamed_controls_by_page) -> None:
    over = {
        route: controls
        for route, controls in unnamed_controls_by_page.items()
        if len(controls) > UNNAMED_CONTROL_BUDGET.get(route, 0)
    }

    assert not over, (
        "Name these controls where they are built (setAccessibleName, or setBuddy on the "
        f"visible label): {over}"
    )


def test_unnamed_control_budget_is_not_stale(unnamed_controls_by_page) -> None:
    stale = {
        route: (budget, len(unnamed_controls_by_page.get(route, ())))
        for route, budget in UNNAMED_CONTROL_BUDGET.items()
        if len(unnamed_controls_by_page.get(route, ())) < budget
    }

    assert not stale, f"Lower these budgets to the page's count (budget, count): {stale}"


def test_census_counts_an_unnamed_control_until_it_is_named_or_labelled(
    project_with_host, qapp
) -> None:
    window, _service, _document = project_with_host
    page = window._host_structure_view
    edit = QLineEdit(page)
    page.layout().addWidget(edit)
    button = QPushButton("x", page)
    page.layout().addWidget(button)

    assert len(_census(window, qapp, ("host_structure",))["host_structure"]) == 2

    edit.setAccessibleName("Example Value")
    label = QLabel("Example action", page)
    page.layout().addWidget(label)
    label.setBuddy(button)

    assert _census(window, qapp, ("host_structure",)) == {"host_structure": []}


def test_census_counts_the_row_controls_of_every_page_that_lists_rows(
    unnamed_controls_by_page, project_with_host
) -> None:
    window, _service, _document = project_with_host
    tables = {
        "convergence_cutoff": window._convergence_view._table,
        "convergence_kpoint": window._kpoint_convergence_view._table,
        "pristine_relaxation": window._pristine_relaxation_view._table,
        "uep": window._uep_view._table,
        "muon_relaxations": window._muon_relaxation_view._table,
        "polarisation": window._polarisation_view._sources_table,
        "hyperfine_coupling": window._hyperfine_coupling_view._results_table,
        "quantum_corrections": window._quantum_corrections_view._results_table,
        "energy_barriers": window._energy_barriers_view._sources_table,
    }

    assert set(unnamed_controls_by_page) == set(PAGE_ROUTES)
    assert [route for route, table in tables.items() if table.rowCount() == 0] == []
    view = window._muon_relaxation_view
    assert view._table.cellWidget(0, view._REMOVE_COLUMN).accessibleName() == (
        "Remove Manual Site 1 (positive muon)"
    )
