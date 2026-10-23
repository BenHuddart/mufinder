"""Run Python's cyclic garbage collector on the GUI thread only.

Automatic collection runs in whichever thread happens to allocate when a
threshold is passed. When that is a worker thread (a blocking call, a
progress poller, a ``QThread`` worker) and the cycle it frees holds a PySide
object, ``QObject::~QObject`` runs there: it takes one of Qt's internal object
locks and then waits for the GIL to call back into Python. Meanwhile the GUI
thread, holding the GIL, can be deleting widgets that need the same lock, and
both threads stop for good. Qt shares a small pool of those locks between all
objects, so unrelated objects collide.

:class:`GuiGarbageCollector` turns automatic collection off and runs the
collection itself from a GUI-thread timer once the collector's own thresholds
are passed, so cycles are only ever freed on the GUI thread. Reference counting
still frees every other object at once, in any thread. Worker threads must not
call ``gc.collect()``.
"""

from __future__ import annotations

import gc

from PySide6.QtCore import QObject, QTimer

#: How often the GUI thread checks whether a collection is due, in milliseconds.
COLLECT_INTERVAL_MS = 1000


class GuiGarbageCollector(QObject):
    """Collect reference cycles from a GUI-thread timer instead of automatically."""

    def __init__(self, parent: QObject | None = None, *, interval_ms: int = COLLECT_INTERVAL_MS) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.collect_if_due)
        self._was_enabled = False

    def is_running(self) -> bool:
        return self._timer.isActive()

    def start(self) -> None:
        """Turn automatic collection off and start collecting from the timer."""

        if self.is_running():
            return
        self._was_enabled = gc.isenabled()
        gc.disable()
        self._timer.start()

    def stop(self) -> None:
        """Stop the timer and turn automatic collection back on if it was on."""

        if not self.is_running():
            return
        self._timer.stop()
        if self._was_enabled:
            gc.enable()

    def collect_if_due(self) -> int | None:
        """Collect the oldest generation whose threshold is passed; return it, or ``None``."""

        counts = gc.get_count()
        thresholds = gc.get_threshold()
        for generation in reversed(range(min(len(counts), len(thresholds)))):
            if thresholds[generation] and counts[generation] > thresholds[generation]:
                gc.collect(generation)
                return generation
        return None


def install_gui_garbage_collector(app: QObject) -> GuiGarbageCollector:
    """Start a :class:`GuiGarbageCollector` owned by ``app``, the running ``QApplication``."""

    collector = GuiGarbageCollector(app)
    collector.start()
    return collector
