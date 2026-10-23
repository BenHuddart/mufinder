"""Test double for the shared workflow-row "Reset Failed Jobs..." menu.

Every workflow table opens its reset menu through
``mufinder.app.workflow_table_utils.open_reset_failed_context_menu``, which calls
``QMenu.exec`` — a modal, blocking call no headless test can drive. These helpers
monkeypatch the ``QMenu`` symbol that helper resolves and either trigger the
action or report its enabled state, so a widget test can assert what the user
would actually be able to do from a right-click.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint

from mufinder.app import workflow_table_utils
from mufinder.app.workflow_table_utils import RESET_FAILED_ACTION_TEXT


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
    """Records the actions added to it and fires the enabled reset action.

    ``exec`` deliberately triggers only an *enabled* reset action, mirroring what
    a user can click, so a test that expects no emission does not have to know
    whether the action was absent or merely greyed out.
    """

    instances: list["FakeMenu"] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.actions: list[FakeAction] = []
        #: Index of the action each separator was added just before.
        self.separators_before: list[int] = []
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

    def exec(self, *_args, **_kwargs) -> None:
        for action in self.actions:
            if action.text.startswith(RESET_FAILED_ACTION_TEXT) and action.enabled:
                action.triggered.emit()
                return

    def reset_action(self) -> FakeAction | None:
        """Return the reset action, whose text gains " (n of m)" for a mixed selection."""

        for action in self.actions:
            if action.text.startswith(RESET_FAILED_ACTION_TEXT):
                return action
        return None

    def action(self, text: str) -> FakeAction | None:
        """Return the action with this exact text, or ``None``.

        Generalizes :meth:`reset_action` for tests exercising an
        ``additional_actions`` entry (e.g. "Continue relaxation…") alongside
        the shared reset action.
        """

        for action in self.actions:
            if action.text == text:
                return action
        return None


def install_fake_menu(monkeypatch) -> type[FakeMenu]:
    """Patch the ``QMenu`` the shared reset-menu helper uses."""

    FakeMenu.instances = []
    monkeypatch.setattr(workflow_table_utils, "QMenu", FakeMenu)
    return FakeMenu


def open_reset_menu(table, *, row: int = 0) -> FakeMenu | None:
    """Raise ``table``'s reset context menu over ``row`` and return the menu.

    The real handler maps a viewport position to a row via ``rowAt``, so the
    click point is taken from the row's visual rect rather than a fixed offset —
    row heights differ per widget.
    """

    rect = table.visualRect(table.model().index(row, 0))
    table.customContextMenuRequested.emit(QPoint(rect.center().x(), rect.center().y()))
    return FakeMenu.instances[-1] if FakeMenu.instances else None
