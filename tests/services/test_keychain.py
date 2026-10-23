"""Tests for macOS Keychain and Linux Secret Service credential lookup."""

from __future__ import annotations

import subprocess

import pytest

from mufinder.domain.cluster import ClusterConfig
from mufinder.services.keychain import (
    KeychainCredentialError,
    credential_store,
    credential_store_for_display,
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


def test_keychain_support_covers_macos_and_linux_glamdring_password_profiles() -> None:
    assert is_glamdring_keychain_supported(_config(), platform="darwin") is True
    assert is_glamdring_keychain_supported(_config(), platform="linux") is True
    assert is_glamdring_keychain_supported(_config(), platform="win32") is False
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

    for platform in ("darwin", "linux"):
        assert is_glamdring_keychain_supported(non_glamdring, platform=platform) is False
        assert is_glamdring_keychain_supported(key_auth, platform=platform) is False


def test_credential_store_names_the_platform_store() -> None:
    assert credential_store(platform="darwin").name == "macOS Keychain"
    assert credential_store(platform="linux").name == "system keyring"
    assert credential_store(platform="win32") is None
    assert credential_store_for_display(platform="win32").name == "system keyring"
    assert credential_store(platform="linux").sentence_name == "System keyring"
    assert credential_store(platform="darwin").sentence_name == "macOS Keychain"


def test_load_keychain_credentials_returns_both_secrets_on_macos() -> None:
    calls: list[list[str]] = []
    timeouts: list[object] = []

    def _runner(args, **kwargs):
        calls.append(args)
        timeouts.append(kwargs.get("timeout"))
        service = args[args.index("-s") + 1]
        value = "physics-secret\n" if service == "bastion" else "glam-secret\n"
        return subprocess.CompletedProcess(args, 0, stdout=value, stderr="")

    credentials = load_keychain_credentials(_config(), runner=_runner, platform="darwin")

    assert credentials.jump_secret == "physics-secret"
    assert credentials.target_secret == "glam-secret"
    assert calls == [
        ["/usr/bin/security", "find-generic-password", "-a", "physics-user", "-s", "bastion", "-w"],
        ["/usr/bin/security", "find-generic-password", "-a", "glam-user", "-s", "glamdring", "-w"],
    ]
    assert all(isinstance(timeout, (int, float)) and timeout > 0 for timeout in timeouts)


def test_load_keychain_credentials_returns_both_secrets_on_linux() -> None:
    calls: list[list[str]] = []

    def _runner(args, **_kwargs):
        calls.append(args)
        service = args[args.index("service") + 1]
        value = "physics-secret" if service == "bastion" else "glam-secret"
        return subprocess.CompletedProcess(args, 0, stdout=value, stderr="")

    credentials = load_keychain_credentials(_config(), runner=_runner, platform="linux")

    assert credentials.jump_secret == "physics-secret"
    assert credentials.target_secret == "glam-secret"
    assert calls == [
        ["/usr/bin/secret-tool", "lookup", "service", "bastion", "account", "physics-user"],
        ["/usr/bin/secret-tool", "lookup", "service", "glamdring", "account", "glam-user"],
    ]


def test_load_keychain_credentials_reports_missing_entries_without_secret_leak() -> None:
    def _runner(args, **_kwargs):
        service = args[args.index("-s") + 1]
        if service == "bastion":
            return subprocess.CompletedProcess(args, 44, stdout="", stderr="not found")
        return subprocess.CompletedProcess(args, 0, stdout="glam-secret\n", stderr="")

    with pytest.raises(KeychainCredentialError) as exc_info:
        load_keychain_credentials(_config(), runner=_runner, platform="darwin")

    issue = exc_info.value.issues[0]
    assert issue.endpoint == "jump"
    assert issue.service == "bastion"
    assert issue.account == "physics-user"
    assert issue.command == "security add-generic-password -a physics-user -s bastion -w"
    assert issue.detail == "not found"
    assert "macOS Keychain" in str(exc_info.value)
    assert "glam-secret" not in str(exc_info.value)


def test_load_keychain_credentials_reports_missing_linux_entry_with_store_command() -> None:
    def _runner(args, **_kwargs):
        # secret-tool exits 1 with no output when no item matches.
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="")

    with pytest.raises(KeychainCredentialError) as exc_info:
        load_keychain_credentials(_config(), runner=_runner, platform="linux")

    assert [issue.service for issue in exc_info.value.issues] == ["bastion", "glamdring"]
    issue = exc_info.value.issues[0]
    assert issue.command == (
        "secret-tool store --label='MuFinder bastion (physics-user)' service bastion account physics-user"
    )
    assert issue.detail == "Lookup exited with status 1."
    assert "system keyring" in str(exc_info.value)


def test_load_keychain_credentials_never_reports_stdout_of_failed_lookup() -> None:
    def _runner(args, **_kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="leaked-secret", stderr="")

    with pytest.raises(KeychainCredentialError) as exc_info:
        load_keychain_credentials(_config(), runner=_runner, platform="linux")

    assert all("leaked-secret" not in issue.detail for issue in exc_info.value.issues)


def test_load_keychain_credentials_rejects_empty_password() -> None:
    def _runner(args, **_kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="\n", stderr="")

    with pytest.raises(KeychainCredentialError) as exc_info:
        load_keychain_credentials(_config(), runner=_runner, platform="linux")

    assert all(
        issue.detail == "The entry exists but its password is empty."
        for issue in exc_info.value.issues
    )


def test_load_keychain_credentials_handles_security_command_launch_failure() -> None:
    def _runner(_args, **_kwargs):
        raise OSError("security unavailable")

    with pytest.raises(KeychainCredentialError) as exc_info:
        load_keychain_credentials(_config(), runner=_runner, platform="darwin")

    assert len(exc_info.value.issues) == 2
    assert all(issue.detail == "security unavailable" for issue in exc_info.value.issues)


def test_load_keychain_credentials_handles_missing_secret_tool() -> None:
    def _runner(_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "/usr/bin/secret-tool")

    with pytest.raises(KeychainCredentialError) as exc_info:
        load_keychain_credentials(_config(), runner=_runner, platform="linux")

    assert len(exc_info.value.issues) == 2
    assert all("/usr/bin/secret-tool" in issue.detail for issue in exc_info.value.issues)


def test_load_keychain_credentials_reports_lookup_timeout() -> None:
    def _runner(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    with pytest.raises(KeychainCredentialError) as exc_info:
        load_keychain_credentials(_config(), runner=_runner, platform="linux")

    assert all("timed out" in issue.detail for issue in exc_info.value.issues)


def test_load_keychain_credentials_requires_usernames() -> None:
    calls: list[list[str]] = []
    config = ClusterConfig(name="Redwood", hostname="glamdring.physics.ox.ac.uk")

    with pytest.raises(KeychainCredentialError) as exc_info:
        load_keychain_credentials(
            config, runner=lambda args, **_kw: calls.append(args), platform="linux"
        )

    assert calls == []
    assert all(issue.detail == "Missing username." for issue in exc_info.value.issues)


def test_keychain_setup_help_uses_glamdring_tools_service_names() -> None:
    help_text = keychain_setup_help(_config(), platform="darwin")

    assert (
        keychain_setup_command(endpoint="jump", account="physics-user", platform="darwin")
        in help_text
    )
    assert (
        keychain_setup_command(endpoint="target", account="glam-user", platform="darwin")
        in help_text
    )
    assert "-s bastion" in help_text
    assert "-s glamdring" in help_text


def test_keychain_setup_help_on_linux_uses_secret_tool() -> None:
    help_text = keychain_setup_help(_config(), platform="linux")

    assert "sudo apt install libsecret-tools" in help_text
    assert (
        "secret-tool store --label='MuFinder bastion (physics-user)' service bastion account physics-user"
        in help_text
    )
    assert (
        "secret-tool store --label='MuFinder glamdring (glam-user)' service glamdring account glam-user"
        in help_text
    )
    assert "security add-generic-password" not in help_text


def test_keychain_setup_command_quotes_unsafe_account_names() -> None:
    command = keychain_setup_command(endpoint="target", account="a b;rm", platform="linux")

    assert (
        command
        == "secret-tool store --label='MuFinder glamdring (a b;rm)' service glamdring account 'a b;rm'"
    )
