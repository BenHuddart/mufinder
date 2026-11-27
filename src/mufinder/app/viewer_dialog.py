"""Dialogs that show a 3D structure viewer and release it when they finish.

A structure viewer holds a VTK render window, which has to be shut down before
the dialog's widgets are destroyed. :class:`StructureViewerDialog` does that
once, when the dialog finishes (``done``: accept, reject, Escape) or is closed,
for every viewer registered with :meth:`StructureViewerDialog.hold_viewer`.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QDialog, QWidget


class _ShutsDown(Protocol):
    def shutdown(self) -> None: ...


ViewerT = TypeVar("ViewerT", bound=_ShutsDown)


class StructureViewerDialog(QDialog):
    """A dialog that shuts its structure viewers down once, when it finishes or closes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._held_viewers: list[_ShutsDown] = []
        self._viewers_shut_down = False

    def hold_viewer(self, viewer: ViewerT) -> ViewerT:
        """Shut ``viewer`` down with the dialog; returns ``viewer``."""

        self._held_viewers.append(viewer)
        return viewer

    def viewers_shut_down(self) -> bool:
        """Whether the held viewers have been shut down."""

        return self._viewers_shut_down

    def shutdown_viewers(self) -> None:
        """Shut every held viewer down; later calls do nothing.

        A viewer whose Qt side is already gone raises ``RuntimeError``; it has
        nothing left to release, so that is ignored. Any other error is raised
        once every other viewer has been shut down.
        """

        if self._viewers_shut_down:
            return
        self._viewers_shut_down = True
        failure: Exception | None = None
        for viewer in self._held_viewers:
            try:
                viewer.shutdown()
            except RuntimeError:
                pass
            except Exception as exc:  # noqa: BLE001 - re-raised below, after the other viewers.
                failure = failure or exc
        if failure is not None:
            raise failure

    def done(self, result: int) -> None:  # noqa: D401 - Qt API override.
        """Finish the dialog after releasing viewer resources."""

        self.shutdown_viewers()
        super().done(result)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API naming.
        self.shutdown_viewers()
        super().closeEvent(event)
