"""Tests for shared cluster UI helper functions."""

from __future__ import annotations

import mufinder.app.cluster_ui_utils as ui_module
from mufinder.domain.cluster import ClusterConfig
import mufinder.services.keychain as keychain_module
from mufinder.services.keychain import KeychainCredentialError, KeychainCredentialIssue
from mufinder.services.ssh_client import (
    SSHCredentials,
    SSHAuthenticationError,
    SSHHostKeyMismatchError,
    SSHProtocolBannerError,
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


def _jump_config() -> ClusterConfig:
    return ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="glam-user",
        auth_method="password",
        jump_host_enabled=True,
        jump_hostname="bastion.physics.ox.ac.uk",
        jump_username="phys-user",
        jump_auth_method="password",
    )


def _keychain_config() -> ClusterConfig:
    return ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="glam-user",
        auth_method="password",
        jump_host_enabled=True,
        jump_hostname="bastion.physics.ox.ac.uk",
        jump_username="phys-user",
        jump_auth_method="password",
        use_macos_keychain_passwords=True,
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

        def setSizeGripEnabled(self, enabled: bool) -> None:  # noqa: N802 - Qt API
            events["size_grip"] = enabled

        def setWindowFlag(self, flag, enabled: bool) -> None:  # noqa: N802 - Qt API
            events["window_flag"] = (flag, enabled)

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
    assert events["size_grip"] is True
    assert events["window_flag"] == (ui_module.Qt.WindowType.MSWindowsFixedSizeDialogHint, False)
    assert events["executed"] is True


def test_show_detailed_warning_truncates_long_single_line_summary(monkeypatch) -> None:
    events: dict[str, object] = {}

    class _FakeBox:
        class Icon:
            Warning = "warning"

        def __init__(self, _parent) -> None:
            pass

        def setSizeGripEnabled(self, _enabled: bool) -> None:  # noqa: N802 - Qt API
            return

        def setWindowFlag(self, _flag, _enabled: bool) -> None:  # noqa: N802 - Qt API
            return

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
    banner = ui_module.format_cluster_connection_error(
        _config(),
        SSHProtocolBannerError(
            hostname="glamdring.physics.ox.ac.uk",
            port=22,
            details="Error reading SSH protocol banner",
        ),
    )
    fallback = ui_module.format_cluster_connection_error(_config(), RuntimeError(""))

    assert "not trusted yet" in unknown
    assert "does not match the key previously trusted" in mismatch
    assert "username/password pair" in auth_password
    assert "SSH key path" in auth_key
    assert "Could not establish an SSH session" in transport
    assert "not answering as an SSH server" in banner
    assert "hostname and SSH port" in banner
    assert fallback == "Unknown cluster connection failure."


def test_prompt_for_cluster_credentials_collects_jump_then_target(monkeypatch) -> None:
    prompts: list[str] = []
    responses = [("phys-secret", True), ("glam-secret", True)]

    def _fake_get_text(_parent, title, prompt, _mode):
        prompts.append(f"{title}: {prompt}")
        return responses.pop(0)

    monkeypatch.setattr(ui_module.QInputDialog, "getText", _fake_get_text)

    credentials = ui_module.prompt_for_cluster_credentials(None, _jump_config())

    assert credentials is not None
    assert credentials.jump_secret == "phys-secret"
    assert credentials.target_secret == "glam-secret"
    assert "Jump host password" in prompts[0]
    assert "Physics/bastion password" in prompts[0]
    assert "Redwood password" in prompts[1]


def test_prompt_for_cluster_credentials_can_skip_optional_key_prompt(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        ui_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: calls.append("prompt") or ("", False),
    )

    credentials = ui_module.prompt_for_cluster_credentials(
        None,
        _config(auth_method="key"),
        prompt_optional_key=False,
    )

    assert credentials is not None
    assert credentials.target_secret is None
    assert calls == []


def test_prompt_for_cluster_credentials_uses_keychain_without_prompting(monkeypatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr(
        ui_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: prompts.append("prompt") or ("", False),
    )
    monkeypatch.setattr(
        ui_module,
        "is_glamdring_keychain_supported",
        lambda _config: True,
    )
    monkeypatch.setattr(
        ui_module,
        "load_keychain_credentials",
        lambda _config: SSHCredentials(target_secret="glam-secret", jump_secret="physics-secret"),
    )

    credentials = ui_module.prompt_for_cluster_credentials(None, _keychain_config())

    assert credentials == SSHCredentials(target_secret="glam-secret", jump_secret="physics-secret")
    assert prompts == []


def test_prompt_for_cluster_credentials_reports_missing_keychain_entry(monkeypatch) -> None:
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(keychain_module.sys, "platform", "darwin")

    def _raise_missing(_config):
        raise KeychainCredentialError(
            [
                KeychainCredentialIssue(
                    endpoint="jump",
                    service="bastion",
                    account="phys-user",
                    command="security add-generic-password -a phys-user -s bastion -w",
                )
            ]
        )

    monkeypatch.setattr(ui_module, "is_glamdring_keychain_supported", lambda _config: True)
    monkeypatch.setattr(ui_module, "load_keychain_credentials", _raise_missing)
    monkeypatch.setattr(
        ui_module,
        "show_detailed_warning",
        lambda _parent, title, message, **_kwargs: warnings.append((title, message)),
    )

    credentials = ui_module.prompt_for_cluster_credentials(None, _keychain_config())

    assert credentials is None
    assert warnings[0][0] == "Missing Keychain passwords"
    assert "security add-generic-password -a phys-user -s bastion -w" in warnings[0][1]


def test_prompt_for_cluster_credentials_reports_missing_linux_keyring_entry(monkeypatch) -> None:
    warnings: list[tuple[str, str, str]] = []
    command = "secret-tool store --label='MuFinder bastion (phys-user)' service bastion account phys-user"
    monkeypatch.setattr(keychain_module.sys, "platform", "linux")

    def _raise_missing(_config):
        raise KeychainCredentialError(
            [KeychainCredentialIssue(endpoint="jump", service="bastion", account="phys-user", command=command)]
        )

    monkeypatch.setattr(ui_module, "load_keychain_credentials", _raise_missing)
    monkeypatch.setattr(
        ui_module,
        "show_detailed_warning",
        lambda _parent, title, message, **kwargs: warnings.append((title, message, kwargs["informative_text"])),
    )

    credentials = ui_module.prompt_for_cluster_credentials(None, _keychain_config())

    assert credentials is None
    title, message, informative = warnings[0]
    assert title == "Missing keyring passwords"
    assert message.startswith("MuFinder could not read the required system keyring password entries.")
    assert command in message
    assert "sudo apt install libsecret-tools" in message
    assert informative == "Open Details for the exact system keyring setup commands."


def test_prompt_to_trust_host_key_returns_boolean_from_dialog(monkeypatch) -> None:
    responses = [1, 2]
    seen_details: list[str] = []
    size_grips: list[bool] = []
    flags: list[tuple[object, bool]] = []

    class _FakePrompt:
        class Icon:
            Warning = "warning"

        class StandardButton:
            Yes = 1
            No = 2

        def __init__(self, _parent) -> None:
            pass

        def setSizeGripEnabled(self, enabled: bool) -> None:  # noqa: N802
            size_grips.append(enabled)

        def setWindowFlag(self, flag, enabled: bool) -> None:  # noqa: N802
            flags.append((flag, enabled))

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
    assert size_grips == [True, True]
    assert flags == [
        (ui_module.Qt.WindowType.MSWindowsFixedSizeDialogHint, False),
        (ui_module.Qt.WindowType.MSWindowsFixedSizeDialogHint, False),
    ]
