"""Tests for macOS Keychain credential lookup."""

from __future__ import annotations

import subprocess

import pytest

from mufinder.cluster_setup import ClusterConfig
from mufinder.services.keychain import (
    KeychainCredentialError,
    is_glamdring_keychain_supported,
    keychain_setup_command,
    keychain_setup_help,
    load_keychain_credentials,
)


def _config() -> ClusterConfig:
    return ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="glam-user",
        auth_method="password",
        jump_host_enabled=True,
        jump_hostname="bastion.physics.ox.ac.uk",
        jump_username="physics-user",
        jump_auth_method="password",
    )


def test_keychain_support_is_macos_glamdring_password_only() -> None:
    assert is_glamdring_keychain_supported(_config(), platform="darwin") is True
    assert is_glamdring_keychain_supported(_config(), platform="linux") is False
    non_glamdring = ClusterConfig(name="Other", hostname="other")
    key_auth = ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="glam-user",
        auth_method="key",
        jump_host_enabled=True,
        jump_hostname="bastion.physics.ox.ac.uk",
        jump_username="physics-user",
        jump_auth_method="password",
    )

    assert is_glamdring_keychain_supported(non_glamdring, platform="darwin") is False
    assert is_glamdring_keychain_supported(
        key_auth,
        platform="darwin",
    ) is False


def test_load_keychain_credentials_returns_both_secrets() -> None:
    calls: list[list[str]] = []

    def _runner(args, **_kwargs):
        calls.append(args)
        service = args[args.index("-s") + 1]
        value = "physics-secret\n" if service == "bastion" else "glam-secret\n"
        return subprocess.CompletedProcess(args, 0, stdout=value, stderr="")

    credentials = load_keychain_credentials(_config(), runner=_runner)

    assert credentials.jump_secret == "physics-secret"
    assert credentials.target_secret == "glam-secret"
    assert calls == [
        ["/usr/bin/security", "find-generic-password", "-a", "physics-user", "-s", "bastion", "-w"],
        ["/usr/bin/security", "find-generic-password", "-a", "glam-user", "-s", "glamdring", "-w"],
    ]


def test_load_keychain_credentials_reports_missing_entries_without_secret_leak() -> None:
    def _runner(args, **_kwargs):
        service = args[args.index("-s") + 1]
        if service == "bastion":
            return subprocess.CompletedProcess(args, 44, stdout="", stderr="not found")
        return subprocess.CompletedProcess(args, 0, stdout="glam-secret\n", stderr="")

    with pytest.raises(KeychainCredentialError) as exc_info:
        load_keychain_credentials(_config(), runner=_runner)

    issue = exc_info.value.issues[0]
    assert issue.endpoint == "jump"
    assert issue.service == "bastion"
    assert issue.account == "physics-user"
    assert issue.command == "security add-generic-password -a physics-user -s bastion -w"
    assert "glam-secret" not in str(exc_info.value)


def test_load_keychain_credentials_handles_security_command_launch_failure() -> None:
    def _runner(_args, **_kwargs):
        raise OSError("security unavailable")

    with pytest.raises(KeychainCredentialError) as exc_info:
        load_keychain_credentials(_config(), runner=_runner)

    assert len(exc_info.value.issues) == 2
    assert all(issue.detail == "security unavailable" for issue in exc_info.value.issues)


def test_keychain_setup_help_uses_glamdring_tools_service_names() -> None:
    help_text = keychain_setup_help(_config())

    assert keychain_setup_command(endpoint="jump", account="physics-user") in help_text
    assert keychain_setup_command(endpoint="target", account="glam-user") in help_text
    assert "-s bastion" in help_text
    assert "-s glamdring" in help_text
