"""Read-only macOS Keychain credential lookup for supported cluster profiles."""

from __future__ import annotations

from dataclasses import dataclass
import shlex
import subprocess
import sys
from typing import Callable

from mufinder.cluster_setup import ClusterConfig
from mufinder.services.ssh_client import SSHCredentials

_GLAMDRING_HOSTNAME = "glamdring.physics.ox.ac.uk"
_BASTION_HOSTNAME = "bastion.physics.ox.ac.uk"
_BASTION_SERVICE = "bastion"
_GLAMDRING_SERVICE = "glamdring"


@dataclass(frozen=True, slots=True)
class KeychainCredentialIssue:
    """One missing or inaccessible Keychain credential."""

    endpoint: str
    service: str
    account: str
    command: str
    detail: str = ""


class KeychainCredentialError(ConnectionError):
    """Raised when configured Keychain credentials cannot be resolved."""

    def __init__(self, issues: list[KeychainCredentialIssue]) -> None:
        self.issues = list(issues)
        missing = ", ".join(f"{issue.service}/{issue.account}" for issue in self.issues)
        super().__init__(f"Missing macOS Keychain credential(s): {missing}")


SubprocessRun = Callable[..., subprocess.CompletedProcess[str]]


def is_macos() -> bool:
    """Return whether the current platform is macOS."""

    return sys.platform == "darwin"


def is_glamdring_keychain_supported(config: ClusterConfig, *, platform: str | None = None) -> bool:
    """Return whether MuFinder can use macOS Keychain for this cluster."""

    resolved_platform = sys.platform if platform is None else platform
    return (
        resolved_platform == "darwin"
        and config.hostname.strip().casefold() == _GLAMDRING_HOSTNAME
        and config.jump_host_enabled
        and config.jump_hostname.strip().casefold() == _BASTION_HOSTNAME
        and config.auth_method.strip().casefold() == "password"
        and config.jump_auth_method.strip().casefold() == "password"
    )


def keychain_setup_command(*, endpoint: str, account: str) -> str:
    """Return the user-facing command to create one required Keychain entry."""

    service = _BASTION_SERVICE if endpoint == "jump" else _GLAMDRING_SERVICE
    return f"security add-generic-password -a {shlex.quote(account)} -s {service} -w"


def keychain_setup_help(config: ClusterConfig) -> str:
    """Return setup guidance for the required Redwood/Glamdring Keychain entries."""

    return (
        "MuFinder reads existing macOS Keychain entries and does not store passwords itself.\n\n"
        "Create the required entries with:\n\n"
        f"{keychain_setup_command(endpoint='jump', account=config.jump_username.strip())}\n"
        f"{keychain_setup_command(endpoint='target', account=config.username.strip())}"
    )


def load_keychain_credentials(
    config: ClusterConfig,
    *,
    runner: SubprocessRun = subprocess.run,
) -> SSHCredentials:
    """Load Physics/bastion and Glamdring passwords from macOS Keychain."""

    issues: list[KeychainCredentialIssue] = []
    jump_secret = _find_password(
        endpoint="jump",
        service=_BASTION_SERVICE,
        account=config.jump_username.strip(),
        runner=runner,
        issues=issues,
    )
    target_secret = _find_password(
        endpoint="target",
        service=_GLAMDRING_SERVICE,
        account=config.username.strip(),
        runner=runner,
        issues=issues,
    )
    if issues:
        raise KeychainCredentialError(issues)
    return SSHCredentials(target_secret=target_secret, jump_secret=jump_secret)


def _find_password(
    *,
    endpoint: str,
    service: str,
    account: str,
    runner: SubprocessRun,
    issues: list[KeychainCredentialIssue],
) -> str | None:
    command = keychain_setup_command(endpoint=endpoint, account=account)
    if not account:
        issues.append(
            KeychainCredentialIssue(
                endpoint=endpoint,
                service=service,
                account=account,
                command=command,
                detail="Missing username.",
            )
        )
        return None

    try:
        result = runner(
            ["security", "find-generic-password", "-a", account, "-s", service, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        issues.append(
            KeychainCredentialIssue(
                endpoint=endpoint,
                service=service,
                account=account,
                command=command,
                detail=str(exc),
            )
        )
        return None
    if result.returncode == 0:
        secret = str(result.stdout).replace("\n", "").strip()
        if secret:
            return secret

    issues.append(
        KeychainCredentialIssue(
            endpoint=endpoint,
            service=service,
            account=account,
            command=command,
            detail=str(result.stderr or result.stdout or "").strip(),
        )
    )
    return None
