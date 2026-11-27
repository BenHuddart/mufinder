"""Test double for the shared workflow-row right-click menu.

Every workflow table opens its row menu through
``mufinder.app.workflow_table_utils.open_row_context_menu``, which calls
``QMenu.exec`` — a modal, blocking call no headless test can drive. These helpers
monkeypatch the ``QMenu`` symbol that helper resolves and record the actions it
adds, so a widget test can assert what the user would be able to do from a
right-click and trigger an entry as the user would.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint

from mufinder.app import workflow_table_utils


class FakeSignal:
    """Minimal stand-in for a ``QAction.triggered`` signal."""

    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self) -> None:
        for callback in self._callbacks:
            callback(False)


class FakeAction:
    def __init__(self, text: str) -> None:
        self.text = text
        self.enabled = True
        self.tooltip = ""
        self.triggered = FakeSignal()

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def setToolTip(self, tooltip: str) -> None:
        self.tooltip = tooltip


class FakeMenu:
    """Records the actions added to it; ``exec`` does nothing."""

    instances: list["FakeMenu"] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.actions: list[FakeAction] = []
        #: Index of the action each separator was added just before.
        self.separators_before: list[int] = []
        #: Submenus by title, each with the entry it adds to this menu.
        self.submenus: dict[str, FakeMenu] = {}
        self._menu_action: FakeAction | None = None
        FakeMenu.instances.append(self)

    def addAction(self, text: str) -> FakeAction:
        action = FakeAction(text)
        self.actions.append(action)
        return action

    def addSeparator(self) -> None:
        self.separators_before.append(len(self.actions))
        return None

    def setToolTipsVisible(self, _visible: bool) -> None:
        return None

    def deleteLater(self) -> None:
        return None

    def addMenu(self, title: str) -> FakeMenu:
        submenu = FakeMenu.__new__(FakeMenu)
        submenu.actions = []
        submenu.separators_before = []
        submenu.submenus = {}
        submenu._menu_action = self.addAction(title)
        self.submenus[title] = submenu
        return submenu

    def menuAction(self) -> FakeAction | None:
        return self._menu_action

    def exec(self, *_args, **_kwargs) -> None:
        return None

    def texts(self) -> list[str]:
        """Return the text of every entry, in menu order."""

        return [action.text for action in self.actions]

    def action(self, text: str) -> FakeAction | None:
        """Return the action with this exact text, or ``None``."""

        for action in self.actions:
            if action.text == text:
                return action
        return None


def install_fake_menu(monkeypatch) -> type[FakeMenu]:
    """Patch the ``QMenu`` the shared row-menu helper uses."""

    FakeMenu.instances = []
    monkeypatch.setattr(workflow_table_utils, "QMenu", FakeMenu)
    return FakeMenu


def open_row_menu(table, *, row: int = 0) -> FakeMenu | None:
    """Raise ``table``'s row context menu over ``row`` and return the menu.

    The real handler maps a viewport position to a row via ``rowAt``, so the
    click point is taken from the row's visual rect rather than a fixed offset —
    row heights differ per widget.
    """

    rect = table.visualRect(table.model().index(row, 0))
    table.customContextMenuRequested.emit(QPoint(rect.center().x(), rect.center().y()))
    return FakeMenu.instances[-1] if FakeMenu.instances else None


#: The row statuses of a job that ended without completing.
ENDED_ROW_STATUSES = ("failed", "stopped", "cancelled")


def menu_texts(menu: FakeMenu) -> list[str]:
    """Return every entry's text, submenu entries included."""

    texts = menu.texts()
    for submenu in menu.submenus.values():
        texts.extend(menu_texts(submenu))
    return texts


def assert_offers_no_reset(menu: FakeMenu | None) -> None:
    """Assert that no entry of ``menu`` resets a job."""

    assert menu is not None
    assert not [text for text in menu_texts(menu) if "reset" in text.casefold()]


def trigger(menu: FakeMenu, text: str) -> None:
    """Click the enabled entry ``text`` of ``menu``, as a user would."""

    action = menu.action(text)
    assert action is not None, f"{text!r} not in {menu.texts()}"
    assert action.enabled, f"{text!r} is disabled"
    action.triggered.emit()
