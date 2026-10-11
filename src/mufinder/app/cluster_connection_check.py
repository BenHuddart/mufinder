"""Asynchronous SSH connection checks with host-key trust prompts.

Establishing an SSH session with paramiko can block for tens of seconds
(20 s connect + 30 s banner + 30 s auth timeouts, doubled when a jump host
is configured), so it must never run on the GUI thread: a single "Test
Connection" click against an unreachable host used to freeze the window
for over a minute. :class:`ClusterConnectionCheck` runs each connection
attempt on a worker ``QThread`` behind a cancellable progress dialog and
surfaces unknown-host-key errors back to the GUI thread, where the user is
prompted to trust the key before the attempt is retried.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QProgressDialog, QWidget

from mufinder.app.cluster_ui_utils import prompt_to_trust_host_key
from mufinder.domain.cluster import ClusterConfig
from mufinder.services.ssh_client import (
    SSHConnection,
    SSHCredentials,
    SSHUnknownHostKeyError,
    save_trusted_host_key,
)

# Strong references to in-flight attempts. Two reasons this must exist:
# PySide6 signal connections do not keep the Python worker object alive (it
# would be garbage-collected before run() fires), and cancelled attempts
# cannot interrupt a paramiko connect mid-flight, so their threads must stay
# referenced until they exit on their own — Qt aborts the process if a
# running QThread is torn down.
_LIVE_ATTEMPTS: set[tuple[QThread, "_ConnectionAttempt"]] = set()


class _ConnectionAttempt(QObject):
    """One SSH connection attempt, executed on a worker thread."""

    succeeded = Signal(int)
    failed = Signal(int, object)

    def __init__(
        self,
        serial: int,
        connection_factory: Callable[..., object],
        config: ClusterConfig,
        credentials: SSHCredentials,
        settings,
    ) -> None:
        super().__init__()
        self._serial = serial
        self._connection_factory = connection_factory
        self._config = config
        self._credentials = credentials
        self._settings = settings

    @Slot()
    def run(self) -> None:
        connection = self._connection_factory(self._config, settings=self._settings)
        try:
            if self._config.jump_host_enabled:
                connection.connect(credentials=self._credentials)
            else:
                connection.connect(password=self._credentials.target_secret)
        except Exception as exc:  # noqa: BLE001 - marshalled to the GUI thread
            self.failed.emit(self._serial, exc)
        else:
            self.succeeded.emit(self._serial)
        finally:
            try:
                if connection.is_connected:
                    connection.disconnect()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass


class ClusterConnectionCheck(QObject):
    """Check an SSH connection off the GUI thread, with trust prompts.

    Emits exactly one of ``succeeded``, ``failed(exception)``, or
    ``cancelled`` (user pressed Cancel on the progress dialog). Unknown host
    keys are handled internally: the user is prompted once per endpoint, the
    trusted key is persisted, and the connection attempt is retried.

    ``connection_factory`` and ``trust_prompt`` exist so callers pass their
    own module-level ``SSHConnection`` / ``prompt_to_trust_host_key``
    references, keeping those seams monkeypatchable per calling module.
    """

    succeeded = Signal()
    failed = Signal(object)
    cancelled = Signal()

    def __init__(
        self,
        parent: QWidget,
        config: ClusterConfig,
        credentials: SSHCredentials,
        settings,
        *,
        title: str = "Testing connection",
        connection_factory: Callable[..., object] = SSHConnection,
        trust_prompt: Callable[..., bool] = prompt_to_trust_host_key,
    ) -> None:
        super().__init__(parent)
        self._parent_widget = parent
        self._config = config
        self._credentials = credentials
        self._settings = settings
        self._title = title
        self._connection_factory = connection_factory
        self._trust_prompt = trust_prompt
        self._trusted_endpoints: set[str] = set()
        self._active = False
        self._attempt_serial = 0
        self._progress: QProgressDialog | None = None

    def start(self) -> None:
        if self._active:
            return
        self._active = True
        host = self._config.name or self._config.hostname
        progress = QProgressDialog(
            f"Connecting to {host}...", "Cancel", 0, 0, self._parent_widget
        )
        progress.setWindowTitle(self._title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.canceled.connect(self._on_cancel_requested)
        self._progress = progress
        progress.show()
        self._start_attempt()

    def _start_attempt(self) -> None:
        self._attempt_serial += 1
        worker = _ConnectionAttempt(
            self._attempt_serial,
            self._connection_factory,
            self._config,
            self._credentials,
            self._settings,
        )
        # The thread is deliberately parentless: if the user cancels (or the
        # owning dialog closes) while paramiko is mid-connect, the attempt is
        # abandoned and must outlive its owner until the connect times out.
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Bound-method receivers live on the GUI thread, so these deliver as
        # queued connections; never connect these signals to bare lambdas
        # (they would run on the worker thread).
        worker.succeeded.connect(self._on_attempt_succeeded)
        worker.failed.connect(self._on_attempt_failed)
        worker.succeeded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        entry = (thread, worker)
        _LIVE_ATTEMPTS.add(entry)
        thread.finished.connect(lambda: _LIVE_ATTEMPTS.discard(entry))
        thread.start()

    def _is_current(self, serial: int) -> bool:
        return self._active and serial == self._attempt_serial

    @Slot(int)
    def _on_attempt_succeeded(self, serial: int) -> None:
        if not self._is_current(serial):
            return
        self._finish()
        self.succeeded.emit()

    @Slot(int, object)
    def _on_attempt_failed(self, serial: int, exc: object) -> None:
        if not self._is_current(serial):
            return
        if (
            isinstance(exc, SSHUnknownHostKeyError)
            and exc.endpoint not in self._trusted_endpoints
        ):
            # Hide the busy dialog while the trust prompt is up so the two
            # modal dialogs do not fight for attention.
            if self._progress is not None:
                self._progress.hide()
            if self._trust_prompt(self._parent_widget, self._config, exc):
                save_trusted_host_key(
                    self._settings,
                    self._config,
                    algorithm=exc.algorithm,
                    key_base64=exc.key_base64,
                    endpoint=exc.endpoint,
                )
                self._trusted_endpoints.add(exc.endpoint)
                if self._progress is not None:
                    self._progress.show()
                self._start_attempt()
                return
            exc = ConnectionError(
                "Connection cancelled because the SSH host key was not trusted."
            )
        self._finish()
        self.failed.emit(exc)

    def _on_cancel_requested(self) -> None:
        if not self._active:
            return
        self._finish()
        self.cancelled.emit()

    def _finish(self) -> None:
        self._active = False
        if self._progress is not None:
            self._progress.canceled.disconnect(self._on_cancel_requested)
            self._progress.close()
            self._progress.deleteLater()
            self._progress = None
