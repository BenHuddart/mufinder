"""Qt event-loop and widget-tree helpers shared by the app tests."""

from __future__ import annotations

from collections.abc import Callable
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem


def pump_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> bool:
    """Pump the GUI event loop until ``predicate`` holds or ``timeout`` seconds elapse.

    Work that runs on a worker thread reports back through queued signals, which
    are only delivered while the event loop spins. Returns the final value of
    ``predicate`` so a test can ``assert pump_until(...)``.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        QApplication.processEvents()
        time.sleep(0.005)
    return predicate()


def find_item_by_route(item: QTreeWidgetItem, route: str) -> QTreeWidgetItem | None:
    """Depth-first search of ``item`` and its descendants for the one whose user-role data is ``route``."""

    if item.data(0, Qt.ItemDataRole.UserRole) == route:
        return item
    for index in range(item.childCount()):
        child = item.child(index)
        if child is None:
            continue
        found = find_item_by_route(child, route)
        if found is not None:
            return found
    return None


def find_tree_item_by_route(tree: QTreeWidget, route: str) -> QTreeWidgetItem | None:
    """Find the item anywhere in ``tree`` whose user-role data is ``route``."""

    for index in range(tree.topLevelItemCount()):
        top_level = tree.topLevelItem(index)
        if top_level is None:
            continue
        found = find_item_by_route(top_level, route)
        if found is not None:
            return found
    return None
