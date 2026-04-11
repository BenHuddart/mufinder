"""Tests for shared cluster UI helper functions."""

from __future__ import annotations

import mufinder.app.cluster_ui_utils as ui_module
from mufinder.cluster_setup import ClusterConfig
from mufinder.services.ssh_client import (
    SSHAuthenticationError,
    SSHHostKeyMismatchError,
    SSHTransportError,
    SSHUnknownHostKeyError,
)


def _config(*, auth_method: str = "key") -> ClusterConfig:
    return ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        port=22,
        username="alice",
        auth_method=auth_method,
    )


def test_show_detailed_warning_uses_simple_warning_for_short_message(monkeypatch) -> None:
    captured: list[tuple[object, str, str]] = []
    monkeypatch.setattr(
        ui_module.QMessageBox,
        "warning",
        lambda parent, title, text: captured.append((parent, title, text)),
    )

    ui_module.show_detailed_warning(None, "Short", "one-line summary")

    assert captured == [(None, "Short", "one-line summary")]


def test_show_detailed_warning_uses_expandable_dialog_for_long_message(monkeypatch) -> None:
    events: dict[str, object] = {}

    class _FakeBox:
        class Icon:
            Warning = "warning"

        def __init__(self, parent) -> None:
            events["parent"] = parent

        def setIcon(self, icon) -> None:  # noqa: N802 - Qt API
            events["icon"] = icon

        def setWindowTitle(self, title: str) -> None:  # noqa: N802 - Qt API
            events["title"] = title

        def setText(self, text: str) -> None:  # noqa: N802 - Qt API
            events["summary"] = text

        def setInformativeText(self, text: str) -> None:  # noqa: N802 - Qt API
            events["info"] = text

        def setDetailedText(self, text: str) -> None:  # noqa: N802 - Qt API
            events["details"] = text

        def exec(self) -> int:
            events["executed"] = True
            return 0

    monkeypatch.setattr(ui_module, "QMessageBox", _FakeBox)

    long_text = "first line\nsecond line\n" + ("x" * 260)
    ui_module.show_detailed_warning(
        parent="parent",
        title="Long",
        message=long_text,
        informative_text="Open details for full traceback.",
    )

    assert events["title"] == "Long"
    assert events["summary"] == "first line"
    assert events["info"] == "Open details for full traceback."
    assert str(events["details"]).startswith("first line")
    assert events["executed"] is True


def test_show_detailed_warning_truncates_long_single_line_summary(monkeypatch) -> None:
    events: dict[str, object] = {}

    class _FakeBox:
        class Icon:
            Warning = "warning"

        def __init__(self, _parent) -> None:
            pass

        def setIcon(self, _icon) -> None:  # noqa: N802 - Qt API
            return

        def setWindowTitle(self, _title: str) -> None:  # noqa: N802 - Qt API
            return

        def setText(self, text: str) -> None:  # noqa: N802 - Qt API
            events["summary"] = text

        def setInformativeText(self, _text: str) -> None:  # noqa: N802 - Qt API
            return

        def setDetailedText(self, _text: str) -> None:  # noqa: N802 - Qt API
            return

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(ui_module, "QMessageBox", _FakeBox)

    ui_module.show_detailed_warning(None, "Long", "x" * 260)

    summary = str(events["summary"])
    assert summary.endswith("...")
    assert len(summary) == 240


def test_format_cluster_connection_error_for_all_structured_errors() -> None:
    unknown = ui_module.format_cluster_connection_error(
        _config(),
        SSHUnknownHostKeyError(
            hostname="glamdring.physics.ox.ac.uk",
            port=22,
            algorithm="ssh-ed25519",
            fingerprint_sha256="SHA256:new",
            key_base64="AAA",
        ),
    )
    mismatch = ui_module.format_cluster_connection_error(
        _config(),
        SSHHostKeyMismatchError(
            hostname="glamdring.physics.ox.ac.uk",
            port=22,
            expected_algorithm="ssh-ed25519",
            expected_fingerprint_sha256="SHA256:old",
            presented_algorithm="ssh-ed25519",
            presented_fingerprint_sha256="SHA256:new",
        ),
    )
    auth_password = ui_module.format_cluster_connection_error(
        _config(auth_method="password"),
        SSHAuthenticationError("bad password"),
    )
    auth_key = ui_module.format_cluster_connection_error(
        _config(auth_method="key"),
        SSHAuthenticationError("bad key"),
    )
    transport = ui_module.format_cluster_connection_error(
        _config(),
        SSHTransportError("network down"),
    )
    fallback = ui_module.format_cluster_connection_error(_config(), RuntimeError(""))

    assert "not trusted yet" in unknown
    assert "does not match the key previously trusted" in mismatch
    assert "username/password pair" in auth_password
    assert "SSH key path" in auth_key
    assert "Could not establish an SSH session" in transport
    assert fallback == "Unknown cluster connection failure."


def test_prompt_to_trust_host_key_returns_boolean_from_dialog(monkeypatch) -> None:
    responses = [1, 2]
    seen_details: list[str] = []

    class _FakePrompt:
        class Icon:
            Warning = "warning"

        class StandardButton:
            Yes = 1
            No = 2

        def __init__(self, _parent) -> None:
            pass

        def setIcon(self, _icon) -> None:  # noqa: N802
            return

        def setWindowTitle(self, _title: str) -> None:  # noqa: N802
            return

        def setText(self, _text: str) -> None:  # noqa: N802
            return

        def setInformativeText(self, _text: str) -> None:  # noqa: N802
            return

        def setDetailedText(self, text: str) -> None:  # noqa: N802
            seen_details.append(text)

        def setStandardButtons(self, _buttons) -> None:  # noqa: N802
            return

        def setDefaultButton(self, _button) -> None:  # noqa: N802
            return

        def exec(self):
            return responses.pop(0)

    monkeypatch.setattr(ui_module, "QMessageBox", _FakePrompt)
    exc = SSHUnknownHostKeyError(
        hostname="glamdring.physics.ox.ac.uk",
        port=22,
        algorithm="ssh-ed25519",
        fingerprint_sha256="SHA256:test",
        key_base64="AAA",
    )

    assert ui_module.prompt_to_trust_host_key(None, _config(), exc) is True
    assert ui_module.prompt_to_trust_host_key(None, _config(), exc) is False
    assert "Fingerprint: SHA256:test" in seen_details[0]
