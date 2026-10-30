"""Every visible control on every workflow page has a name a screen reader can announce.

The census opens each page of a project with the shared host structure, with
one job row staged on every page that stages rows, and counts the visible
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

from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE

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


def _is_unnamed(widget: QWidget, buddies: set[int]) -> bool:
    text = widget.text().strip() if isinstance(widget, QAbstractButton) else ""
    return len(text) <= 1 and not widget.accessibleName() and id(widget) not in buddies


def _describe(widget: QWidget) -> str:
    text = widget.text() if isinstance(widget, QAbstractButton) else ""
    return f"{type(widget).__name__}(text={text!r}, tooltip={widget.toolTip()[:60]!r})"


def _stage_one_row_per_page(window) -> None:
    """Give every page that stages job rows one row, so its row controls are counted."""

    window._convergence_view._add_row(300.0)
    window._kpoint_convergence_view._add_row((2, 2, 2))
    window._pristine_relaxation_view._add_row()
    window._uep_view._add_row()
    window._muon_relaxation_view.add_staged_sites(
        [{"id": str(uuid4()), "label": "Manual Site 1", "fractional_position": [0.1, 0.2, 0.3]}],
        charge_states=(POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE),
    )


def _census(window, qapp, routes=PAGE_ROUTES) -> dict[str, list[str]]:
    window.show()
    qapp.processEvents()
    unnamed: dict[str, list[str]] = {}
    try:
        for route in routes:
            assert window._find_tree_item_by_route(route) is not None, route
            window._set_active_route(route)
            qapp.processEvents()
            page = window._main_stack.currentWidget()
            buddies = {id(label.buddy()) for label in window.findChildren(QLabel) if label.buddy()}
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
    window, _service, _document = project_with_host
    _stage_one_row_per_page(window)
    return _census(window, qapp)


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
