"""Shared cluster UI helpers for warnings and SSH trust prompts."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QInputDialog, QLineEdit, QMessageBox

from mufinder.cluster_setup import ClusterConfig
from mufinder.services.keychain import (
    KeychainCredentialError,
    is_glamdring_keychain_supported,
    keychain_setup_help,
    load_keychain_credentials,
)
from mufinder.services.ssh_client import (
    SSHCredentials,
    SSHAuthenticationError,
    SSHHostKeyMismatchError,
    SSHProtocolBannerError,
    SSHTransportError,
    SSHUnknownHostKeyError,
)


def make_message_box_resizable(box: QMessageBox) -> QMessageBox:
    """Allow a QMessageBox with details to be resized by the user."""

    box.setSizeGripEnabled(True)
    box.setWindowFlag(Qt.WindowType.MSWindowsFixedSizeDialogHint, False)
    return box


def show_detailed_warning(
    parent,
    title: str,
    message: str,
    *,
    informative_text: str = "Open Details to inspect the full error message.",
) -> None:
    """Show a compact warning with expandable details for long messages."""

    text = message.strip()
    lines = [line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()]
    summary = lines[0] if lines else text
    if len(summary) > 240:
        summary = f"{summary[:237]}..."

    if len(lines) <= 1 and len(text) <= 240:
        QMessageBox.warning(parent, title, summary)
        return

    box = make_message_box_resizable(QMessageBox(parent))
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(summary)
    box.setInformativeText(informative_text)
    box.setDetailedText(text)
    box.exec()


def endpoint_label(config: ClusterConfig, endpoint: str) -> str:
    """Return a compact user-facing label for a cluster SSH endpoint."""

    if endpoint == "jump":
        return f"jump host {config.jump_hostname}:{config.jump_port}"
    return f"{config.hostname}:{config.port}"


def prompt_for_cluster_credentials(
    parent,
    config: ClusterConfig,
    *,
    target_secret: str | None = None,
    prompt_optional_key: bool = True,
) -> SSHCredentials | None:
    """Resolve cluster credentials, prompting only when Keychain mode is disabled."""

    return resolve_cluster_credentials(
        parent,
        config,
        target_secret=target_secret,
        prompt_optional_key=prompt_optional_key,
    )


def resolve_cluster_credentials(
    parent,
    config: ClusterConfig,
    *,
    target_secret: str | None = None,
    prompt_optional_key: bool = True,
    manual_credentials: SSHCredentials | None = None,
) -> SSHCredentials | None:
    """Resolve credentials from macOS Keychain or manual prompts/fields."""

    if config.use_macos_keychain_passwords and is_glamdring_keychain_supported(config):
        try:
            return load_keychain_credentials(config)
        except KeychainCredentialError as exc:
            show_keychain_credentials_missing(parent, config, exc)
            return None

    if manual_credentials is not None:
        return manual_credentials

    return _prompt_for_manual_cluster_credentials(
        parent,
        config,
        target_secret=target_secret,
        prompt_optional_key=prompt_optional_key,
    )


def show_keychain_credentials_missing(parent, config: ClusterConfig, exc: KeychainCredentialError) -> None:
    """Show setup guidance for missing macOS Keychain credentials."""

    lines = ["MuFinder could not read the required macOS Keychain password entries."]
    for issue in exc.issues:
        label = "Physics/bastion" if issue.endpoint == "jump" else "Glamdring"
        lines.append(f"- {label}: {issue.command}")
    lines.append("")
    lines.append(keychain_setup_help(config))
    show_detailed_warning(
        parent,
        "Missing Keychain passwords",
        "\n".join(lines),
        informative_text="Open Details for the exact macOS Keychain setup commands.",
    )


def _prompt_for_manual_cluster_credentials(
    parent,
    config: ClusterConfig,
    *,
    target_secret: str | None = None,
    prompt_optional_key: bool = True,
) -> SSHCredentials | None:
    """Prompt for target and jump-host secrets, returning None when required input is cancelled."""

    jump_secret: str | None = None
    if config.jump_host_enabled:
        jump_ok, jump_secret = _prompt_for_endpoint_secret(
            parent,
            config,
            endpoint="jump",
            prompt_optional_key=prompt_optional_key,
        )
        if not jump_ok:
            return None

    resolved_target = target_secret.strip() if target_secret is not None else None
    if resolved_target == "":
        resolved_target = None
    if resolved_target is None:
        target_ok, resolved_target = _prompt_for_endpoint_secret(
            parent,
            config,
            endpoint="target",
            prompt_optional_key=prompt_optional_key,
        )
        if not target_ok:
            return None

    return SSHCredentials(target_secret=resolved_target, jump_secret=jump_secret)


def _prompt_for_endpoint_secret(
    parent,
    config: ClusterConfig,
    *,
    endpoint: str,
    prompt_optional_key: bool,
) -> tuple[bool, str | None]:
    if endpoint == "jump":
        auth_method = config.jump_auth_method.strip().lower()
        username = config.jump_username.strip()
        hostname = config.jump_hostname.strip()
        title_prefix = "Jump host"
        system_name = "Physics/bastion"
    else:
        auth_method = config.auth_method.strip().lower()
        username = config.username.strip()
        hostname = config.hostname.strip()
        title_prefix = config.name
        system_name = config.name

    if auth_method == "password":
        raw_secret, accepted = QInputDialog.getText(
            parent,
            f"{title_prefix} password",
            f"Enter {system_name} password for {username}@{hostname}:",
            QLineEdit.EchoMode.Password,
        )
        secret = raw_secret.strip()
        return (bool(accepted and secret), secret or None)

    if not prompt_optional_key:
        return True, None

    raw_passphrase, accepted = QInputDialog.getText(
        parent,
        f"{title_prefix} key passphrase",
        f"Enter key passphrase for {username}@{hostname} if needed:",
        QLineEdit.EchoMode.Password,
    )
    if not accepted or not raw_passphrase.strip():
        return True, None
    return True, raw_passphrase.strip()


def format_cluster_connection_error(config: ClusterConfig, exc: Exception) -> str:
    """Return a user-facing cluster connection error with clear next steps."""

    if isinstance(exc, SSHUnknownHostKeyError):
        endpoint = endpoint_label(config, exc.endpoint)
        return (
            f"The SSH host key for {endpoint} is not trusted yet.\n\n"
            f"Fingerprint: {exc.fingerprint_sha256}\n"
            f"Algorithm: {exc.algorithm}\n\n"
            "Review the fingerprint carefully and trust it only if it matches the cluster documentation "
            "or a fingerprint from your system administrator."
        )

    if isinstance(exc, SSHHostKeyMismatchError):
        endpoint = endpoint_label(config, exc.endpoint)
        return (
            f"The SSH host key for {endpoint} does not match the key previously trusted.\n\n"
            f"Stored: {exc.expected_fingerprint_sha256} ({exc.expected_algorithm})\n"
            f"Presented: {exc.presented_fingerprint_sha256} ({exc.presented_algorithm})\n\n"
            "This can indicate a real server change or a security problem. Verify the new fingerprint with "
            "the cluster administrators before trusting it."
        )

    if isinstance(exc, SSHAuthenticationError):
        auth_hint = (
            "Check the username/password pair in Setup -> HPC Clusters."
            if config.auth_method == "password"
            else "Check the username, SSH key path, and key passphrase in Setup -> HPC Clusters."
        )
        return f"Authentication failed for {config.username}@{config.hostname}.\n\n{exc}\n\n{auth_hint}"

    if isinstance(exc, SSHProtocolBannerError):
        return (
            f"{exc}\n\n"
            "This usually means the configured host and port are reachable, but that endpoint is not "
            "answering as an SSH server. Check the cluster hostname and SSH port in Setup -> HPC Clusters, "
            "confirm VPN/network access, and try again in case the login node closed the connection under load."
        )

    if isinstance(exc, SSHTransportError):
        return (
            f"Could not establish an SSH session to {config.hostname}:{config.port}.\n\n"
            f"{exc}\n\n"
            "Check VPN/network access, hostname, port, proxy settings, and cluster SSH availability."
        )

    return str(exc).strip() or "Unknown cluster connection failure."


def prompt_to_trust_host_key(parent, config: ClusterConfig, exc: SSHUnknownHostKeyError) -> bool:
    """Prompt the user to trust an unknown SSH host key."""

    endpoint = endpoint_label(config, exc.endpoint)
    message = make_message_box_resizable(QMessageBox(parent))
    message.setIcon(QMessageBox.Icon.Warning)
    message.setWindowTitle("Trust SSH host key?")
    message.setText(f"Trust the SSH host key for {config.name} ({endpoint})?")
    message.setInformativeText(
        "Only continue if this fingerprint matches the cluster documentation or a trusted source."
    )
    message.setDetailedText(
        "\n".join(
            [
                f"Cluster: {config.name}",
                f"Endpoint: {exc.endpoint}",
                f"Host: {exc.hostname}",
                f"Port: {exc.port}",
                f"Algorithm: {exc.algorithm}",
                f"Fingerprint: {exc.fingerprint_sha256}",
            ]
        )
    )
    message.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    message.setDefaultButton(QMessageBox.StandardButton.No)
    return message.exec() == QMessageBox.StandardButton.Yes
