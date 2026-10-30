"""The project navigator's splitter: default width, dragging and the remembered width."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFrame, QTreeWidgetItem

from mufinder.app.main_window import MainWindow
from mufinder.app.relaxation_analysis_widget import default_workspace_state
from mufinder.domain.charge_states import POSITIVE_MUON_CHARGE_STATE
from tests.support.main_window import window_with_host

_WINDOW_SIZE = (1280, 840)


def _shown(window: MainWindow, size: tuple[int, int] = _WINDOW_SIZE) -> MainWindow:
    window.show()
    window.resize(*size)
    QApplication.processEvents()
    return window


def _save_navigator_width(value: object) -> None:
    QSettings("MuFinder", "MuFinder").setValue(MainWindow._PROJECT_NAVIGATOR_WIDTH_KEY, value)


def _saved_navigator_width() -> object:
    return QSettings("MuFinder", "MuFinder").value(MainWindow._PROJECT_NAVIGATOR_WIDTH_KEY)


def _drag_navigator_edge(window: MainWindow, delta: int) -> None:
    handle = window._navigator_splitter.handle(1)
    start = handle.rect().center()
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(handle, start + QPoint(delta, 0))
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=start + QPoint(delta, 0))
    QApplication.processEvents()


def _text_fits_beside_a_scrollbar(window: MainWindow, item: QTreeWidgetItem) -> bool:
    tree = window._project_tree
    index = tree.indexFromItem(item)
    scrollbar = (
        0 if tree.verticalScrollBar().isVisible() else tree.verticalScrollBar().sizeHint().width()
    )
    available = tree.viewport().width() - scrollbar
    return tree.visualRect(index).x() + tree.sizeHintForIndex(index).width() <= available


def _all_items(item: QTreeWidgetItem) -> list[QTreeWidgetItem]:
    items = [item]
    for row in range(item.childCount()):
        items.extend(_all_items(item.child(row)))
    return items


def test_default_width_gives_the_pages_the_rest_of_the_window(qapp) -> None:
    window = _shown(MainWindow())
    try:
        assert window.findChild(QFrame, "projectTreePanel").width() == 220
        assert window._navigator_splitter.sizes() == [220, window._main_stack.width()]
        assert window._main_stack.width() > window._main_stack.minimumSizeHint().width()
        assert window._navigator_splitter.handle(1).accessibleName() == "Resize project navigator"
    finally:
        window.close()


def test_default_width_fits_the_deepest_label_without_eliding(qapp, tmp_path: Path) -> None:
    window, _service, _doc, host = window_with_host(tmp_path)
    try:
        window._set_relaxation_analysis_workspaces(
            [
                default_workspace_state(
                    name="Positive muon sites",
                    charge_state=POSITIVE_MUON_CHARGE_STATE,
                    host_structure_id=host.id,
                    job_ids=[uuid4()],
                )
            ]
        )
        window._refresh_relaxation_analysis_context()
        window._populate_project_tree()
        _shown(window)
        window._project_tree.expandAll()
        QApplication.processEvents()

        deepest = window._find_tree_item_by_route("muon_analysis").child(0)
        assert deepest.text(0) == "Positive muon sites"
        assert window._project_tree_panel.width() == 220
        assert _text_fits_beside_a_scrollbar(window, deepest)

        # The smallest width still shows every stage label in full; only a named
        # workspace at the deepest indent may elide there.
        _drag_navigator_edge(window, -1000)
        assert window._project_tree_panel.width() == window._PROJECT_TREE_MIN_WIDTH
        tree = window._project_tree
        for row in range(tree.topLevelItemCount()):
            for item in _all_items(tree.topLevelItem(row)):
                if item is not deepest:
                    assert _text_fits_beside_a_scrollbar(window, item), item.text(0)
    finally:
        window.close()


def test_splitter_restores_a_saved_width(qapp) -> None:
    _save_navigator_width(260)

    window = _shown(MainWindow())
    try:
        assert window._project_tree_panel.width() == 260
    finally:
        window.close()


def test_dragging_saves_the_width_for_the_next_window(qapp) -> None:
    window = _shown(MainWindow())
    try:
        _drag_navigator_edge(window, 45)
        assert window._project_tree_panel.width() == 265
    finally:
        window.close()

    assert int(_saved_navigator_width()) == 265
    reopened = _shown(MainWindow())
    try:
        assert reopened._project_tree_panel.width() == 265
    finally:
        reopened.close()


def test_navigator_cannot_be_dragged_past_its_minimum_or_over_the_page_minimum(qapp) -> None:
    window = _shown(MainWindow())
    try:
        _drag_navigator_edge(window, -1000)
        assert window._project_tree_panel.width() == window._PROJECT_TREE_MIN_WIDTH

        _drag_navigator_edge(window, 2000)
        assert window._main_stack.width() == window._main_stack.minimumSizeHint().width()
        assert window._project_tree_panel.width() > window._PROJECT_TREE_DEFAULT_WIDTH
    finally:
        window.close()


def test_resizing_the_window_changes_the_pages_and_keeps_the_navigator_width(qapp) -> None:
    window = _shown(MainWindow())
    try:
        stack_width = window._main_stack.width()

        window.resize(_WINDOW_SIZE[0] + 200, _WINDOW_SIZE[1])
        QApplication.processEvents()

        assert window._project_tree_panel.width() == 220
        assert window._main_stack.width() == stack_width + 200
    finally:
        window.close()


def test_a_window_too_narrow_for_the_width_keeps_the_saved_choice(qapp) -> None:
    _save_navigator_width(3000)

    window = _shown(MainWindow())
    try:
        assert window._project_tree_panel.width() < 3000
        assert window._main_stack.width() == window._main_stack.minimumSizeHint().width()
    finally:
        window.close()

    assert int(_saved_navigator_width()) == 3000


@pytest.mark.parametrize(
    "saved",
    ["not a width", "", "0", "-40", "189", "99999999999"],
)
def test_an_unusable_saved_width_falls_back_to_the_default(qapp, saved: str) -> None:
    _save_navigator_width(saved)

    window = _shown(MainWindow())
    try:
        assert window._project_tree_panel.width() == 220
    finally:
        window.close()
