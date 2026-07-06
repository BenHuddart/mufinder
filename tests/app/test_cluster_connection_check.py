"""Tests for the asynchronous cluster connection check."""

from __future__ import annotations

import time
from typing import Callable

from PySide6.QtWidgets import QApplication, QWidget

from mufinder.app.cluster_connection_check import ClusterConnectionCheck
from mufinder.cluster_setup import ClusterConfig
from mufinder.services.ssh_client import SSHCredentials, SSHUnknownHostKeyError


def _process_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        QApplication.processEvents()
        time.sleep(0.005)
    return predicate()


class _FakeSettings:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def value(self, key: str, default=None, value_type=None):  # noqa: ANN001
        value = self._values.get(key, default)
        if value_type is str:
            return str(value)
        return value

    def setValue(self, key: str, value) -> None:  # noqa: N802, ANN001
        self._values[key] = value


def _config(**overrides) -> ClusterConfig:
    values = {"name": "alpha", "hostname": "alpha.example", "username": "alice"}
    values.update(overrides)
    return ClusterConfig(**values)


class _Recorder:
    def __init__(self, check: ClusterConnectionCheck) -> None:
        self.succeeded = 0
        self.failures: list[object] = []
        self.cancelled = 0
        check.succeeded.connect(self._on_succeeded)
        check.failed.connect(self.failures.append)
        check.cancelled.connect(self._on_cancelled)

    def _on_succeeded(self) -> None:
        self.succeeded += 1

    def _on_cancelled(self) -> None:
        self.cancelled += 1

    @property
    def settled(self) -> bool:
        return bool(self.succeeded or self.failures or self.cancelled)


def _make_connection_class(connect_behaviour):
    class _Connection:
        def __init__(self, _config, *, settings=None) -> None:
            _ = settings
            self.connected = False

        @property
        def is_connected(self) -> bool:
            return self.connected

        def connect(self, password: str | None = None, **kwargs) -> None:
            connect_behaviour(self, password=password, **kwargs)

        def disconnect(self) -> None:
            self.connected = False

    return _Connection

def test_connection_check_success(qapp) -> None:
    parent = QWidget()

    def _behaviour(conn, **_kwargs) -> None:
        conn.connected = True

    check = ClusterConnectionCheck(
        parent,
        _config(),
        SSHCredentials(target_secret="secret"),
        _FakeSettings(),
        connection_factory=_make_connection_class(_behaviour),
    )
    recorder = _Recorder(check)
    check.start()

    assert _process_until(lambda: recorder.settled)
    assert recorder.succeeded == 1
    assert recorder.failures == []
    assert recorder.cancelled == 0
    parent.close()


def test_connection_check_reports_failure(qapp) -> None:
    parent = QWidget()

    def _behaviour(_conn, **_kwargs) -> None:
        raise ConnectionError("host unreachable")

    check = ClusterConnectionCheck(
        parent,
        _config(),
        SSHCredentials(target_secret="secret"),
        _FakeSettings(),
        connection_factory=_make_connection_class(_behaviour),
    )
    recorder = _Recorder(check)
    check.start()

    assert _process_until(lambda: recorder.settled)
    assert recorder.succeeded == 0
    assert len(recorder.failures) == 1
    assert str(recorder.failures[0]) == "host unreachable"
    parent.close()


def test_connection_check_retries_after_trusting_host_key(qapp) -> None:
    parent = QWidget()
    settings = _FakeSettings()
    attempts = {"count": 0}
    prompts: list[str] = []

    def _behaviour(conn, **_kwargs) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise SSHUnknownHostKeyError(
                hostname="alpha.example",
                port=22,
                algorithm="ssh-ed25519",
                fingerprint_sha256="SHA256:test",
                key_base64="dGVzdA==",
            )
        conn.connected = True

    check = ClusterConnectionCheck(
        parent,
        _config(),
        SSHCredentials(target_secret="secret"),
        settings,
        connection_factory=_make_connection_class(_behaviour),
        trust_prompt=lambda _parent, _config_, exc: prompts.append(exc.endpoint) or True,
    )
    recorder = _Recorder(check)
    check.start()

    assert _process_until(lambda: recorder.settled)
    assert recorder.succeeded == 1
    assert attempts["count"] == 2
    assert prompts == ["target"]
    assert (
        settings.value("trusted_host_keys/alpha/alpha.example/algorithm", "", str)
        == "ssh-ed25519"
    )
    parent.close()


def test_connection_check_fails_when_trust_declined(qapp) -> None:
    parent = QWidget()
    attempts = {"count": 0}

    def _behaviour(_conn, **_kwargs) -> None:
        attempts["count"] += 1
        raise SSHUnknownHostKeyError(
            hostname="alpha.example",
            port=22,
            algorithm="ssh-ed25519",
            fingerprint_sha256="SHA256:test",
            key_base64="dGVzdA==",
        )

    check = ClusterConnectionCheck(
        parent,
        _config(),
        SSHCredentials(target_secret="secret"),
        _FakeSettings(),
        connection_factory=_make_connection_class(_behaviour),
        trust_prompt=lambda *_args: False,
    )
    recorder = _Recorder(check)
    check.start()

    assert _process_until(lambda: recorder.settled)
    assert recorder.succeeded == 0
    assert attempts["count"] == 1
    assert len(recorder.failures) == 1
    assert isinstance(recorder.failures[0], ConnectionError)
    assert "host key was not trusted" in str(recorder.failures[0])
    parent.close()


def test_connection_check_cancel_abandons_attempt(qapp) -> None:
    parent = QWidget()
    release = {"flag": False}

    def _behaviour(conn, **_kwargs) -> None:
        # Simulate a slow connect the GUI cannot interrupt: poll until the
        # test releases it after cancelling.
        deadline = time.monotonic() + 5.0
        while not release["flag"] and time.monotonic() < deadline:
            time.sleep(0.01)
        conn.connected = True

    check = ClusterConnectionCheck(
        parent,
        _config(),
        SSHCredentials(target_secret="secret"),
        _FakeSettings(),
        connection_factory=_make_connection_class(_behaviour),
    )
    recorder = _Recorder(check)
    check.start()

    # Cancel while the worker is still blocked in connect(). Programmatic
    # cancel() does not emit canceled, so emit it as the button click would.
    assert check._progress is not None
    check._progress.canceled.emit()
    assert _process_until(lambda: recorder.cancelled == 1)

    # Release the worker; its late success must not fire stale callbacks.
    release["flag"] = True
    _process_until(lambda: False, timeout=0.3)
    assert recorder.succeeded == 0
    assert recorder.failures == []
    assert recorder.cancelled == 1
    parent.close()
