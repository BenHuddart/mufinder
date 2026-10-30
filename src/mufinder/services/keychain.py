"""Read-only system keyring credential lookup for supported cluster profiles.

On macOS the passwords come from the login Keychain through ``security``; on
Linux they come from the Secret Service keyring (GNOME Keyring, KWallet)
through ``secret-tool``. MuFinder only reads existing entries and never stores
passwords itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import shlex
import subprocess
import sys
from typing import Callable

from mufinder.domain.cluster import ClusterConfig
from mufinder.services.ssh_client import SSHCredentials

_GLAMDRING_HOSTNAME = "glamdring.physics.ox.ac.uk"
_BASTION_HOSTNAME = "bastion.physics.ox.ac.uk"
_BASTION_SERVICE = "bastion"
_GLAMDRING_SERVICE = "glamdring"
_SECURITY_BIN = "/usr/bin/security"
_SECRET_TOOL_BIN = "/usr/bin/secret-tool"
# Long enough for the user to answer a keyring unlock or access prompt.
_LOOKUP_TIMEOUT_SECONDS = 120


@dataclass(frozen=True, slots=True)
class CredentialStore:
    """The platform password store MuFinder reads cluster passwords from."""

    name: str
    sentence_name: str
    short_name: str


_MACOS_KEYCHAIN = CredentialStore(
    name="macOS Keychain", sentence_name="macOS Keychain", short_name="Keychain"
)
_SECRET_SERVICE_KEYRING = CredentialStore(
    name="system keyring", sentence_name="System keyring", short_name="keyring"
)


@dataclass(frozen=True, slots=True)
class KeychainCredentialIssue:
    """One missing or inaccessible keyring credential."""

    endpoint: str
    service: str
    account: str
    command: str
    detail: str = ""


class KeychainCredentialError(ConnectionError):
    """Raised when configured keyring credentials cannot be resolved."""

    def __init__(
        self, issues: list[KeychainCredentialIssue], *, store_name: str = _MACOS_KEYCHAIN.name
    ) -> None:
        self.issues = list(issues)
        missing = ", ".join(f"{issue.service}/{issue.account}" for issue in self.issues)
        noun = "credential" if len(self.issues) == 1 else "credentials"
        super().__init__(f"Missing {store_name} {noun}: {missing}")


SubprocessRun = Callable[..., subprocess.CompletedProcess[str]]


def credential_store(*, platform: str | None = None) -> CredentialStore | None:
    """Return the password store MuFinder can read on this platform, if any."""

    resolved_platform = sys.platform if platform is None else platform
    if resolved_platform == "darwin":
        return _MACOS_KEYCHAIN
    if resolved_platform.startswith("linux"):
        return _SECRET_SERVICE_KEYRING
    return None


def credential_store_for_display(*, platform: str | None = None) -> CredentialStore:
    """Return the store to name in user-facing text, with a generic fallback."""

    return credential_store(platform=platform) or _SECRET_SERVICE_KEYRING


def is_glamdring_keychain_supported(config: ClusterConfig, *, platform: str | None = None) -> bool:
    """Return whether MuFinder can read this cluster's passwords from a system keyring."""

    return (
        credential_store(platform=platform) is not None
        and config.hostname.strip().casefold() == _GLAMDRING_HOSTNAME
        and config.jump_host_enabled
        and config.jump_hostname.strip().casefold() == _BASTION_HOSTNAME
        and config.auth_method.strip().casefold() == "password"
        and config.jump_auth_method.strip().casefold() == "password"
    )


def keychain_setup_command(*, endpoint: str, account: str, platform: str | None = None) -> str:
    """Return the user-facing command to create one required keyring entry."""

    service = _service_for_endpoint(endpoint)
    if credential_store(platform=platform) is _SECRET_SERVICE_KEYRING:
        label = shlex.quote(f"MuFinder {service} ({account})")
        return f"secret-tool store --label={label} service {service} account {shlex.quote(account)}"
    return f"security add-generic-password -a {shlex.quote(account)} -s {service} -w"


def keychain_setup_help(config: ClusterConfig, *, platform: str | None = None) -> str:
    """Return setup guidance for the required Redwood/Glamdring keyring entries."""

    commands = (
        f"{keychain_setup_command(endpoint='jump', account=config.jump_username.strip(), platform=platform)}\n"
        f"{keychain_setup_command(endpoint='target', account=config.username.strip(), platform=platform)}"
    )
    if credential_store(platform=platform) is _SECRET_SERVICE_KEYRING:
        return (
            "MuFinder reads existing system keyring (Secret Service) entries and does not store "
            "passwords itself. The keyring must be unlocked in your desktop session.\n\n"
            "Install secret-tool if needed (Ubuntu: sudo apt install libsecret-tools), then create "
            "the required entries with the commands below. Each command prompts for the password.\n\n"
            f"{commands}"
        )
    return (
        "MuFinder reads existing macOS Keychain entries and does not store passwords itself.\n\n"
        "Create the required entries with:\n\n"
        f"{commands}"
    )


def load_keychain_credentials(
    config: ClusterConfig,
    *,
    runner: SubprocessRun = subprocess.run,
    platform: str | None = None,
) -> SSHCredentials:
    """Load Physics/bastion and Glamdring passwords from the platform keyring."""

    store_name = credential_store_for_display(platform=platform).name
    issues: list[KeychainCredentialIssue] = []
    jump_secret = _find_password(
        endpoint="jump",
        account=config.jump_username.strip(),
        platform=platform,
        runner=runner,
        issues=issues,
    )
    target_secret = _find_password(
        endpoint="target",
        account=config.username.strip(),
        platform=platform,
        runner=runner,
        issues=issues,
    )
    if issues:
        raise KeychainCredentialError(issues, store_name=store_name)
    return SSHCredentials(target_secret=target_secret, jump_secret=jump_secret)


def _service_for_endpoint(endpoint: str) -> str:
    return _BASTION_SERVICE if endpoint == "jump" else _GLAMDRING_SERVICE


def _lookup_command(*, service: str, account: str, platform: str | None) -> list[str] | None:
    store = credential_store(platform=platform)
    if store is _MACOS_KEYCHAIN:
        return [_SECURITY_BIN, "find-generic-password", "-a", account, "-s", service, "-w"]
    if store is _SECRET_SERVICE_KEYRING:
        return [_SECRET_TOOL_BIN, "lookup", "service", service, "account", account]
    return None


def _find_password(
    *,
    endpoint: str,
    account: str,
    platform: str | None,
    runner: SubprocessRun,
    issues: list[KeychainCredentialIssue],
) -> str | None:
    service = _service_for_endpoint(endpoint)
    command = keychain_setup_command(endpoint=endpoint, account=account, platform=platform)

    def _issue(detail: str) -> None:
        issues.append(
            KeychainCredentialIssue(
                endpoint=endpoint,
                service=service,
                account=account,
                command=command,
                detail=detail,
            )
        )

    if not account:
        _issue("Missing username.")
        return None
    lookup = _lookup_command(service=service, account=account, platform=platform)
    if lookup is None:
        _issue("No supported system keyring on this platform.")
        return None

    try:
        result = runner(
            lookup,
            capture_output=True,
            text=True,
            check=False,
            timeout=_LOOKUP_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        _issue(str(exc))
        return None
    except subprocess.TimeoutExpired:
        _issue(f"Keyring lookup timed out after {_LOOKUP_TIMEOUT_SECONDS} seconds.")
        return None
    if result.returncode == 0:
        secret = str(result.stdout).replace("\n", "").strip()
        if secret:
            return secret
        _issue("The entry exists but its password is empty.")
        return None

    # Only stderr is reported: stdout is where a lookup prints the secret.
    _issue(str(result.stderr or "").strip() or f"Lookup exited with status {result.returncode}.")
    return None
