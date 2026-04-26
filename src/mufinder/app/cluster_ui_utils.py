"""Shared cluster UI helpers for warnings and SSH trust prompts."""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from mufinder.cluster_setup import ClusterConfig
from mufinder.services.ssh_client import (
    SSHAuthenticationError,
    SSHHostKeyMismatchError,
    SSHProtocolBannerError,
    SSHTransportError,
    SSHUnknownHostKeyError,
)


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

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(summary)
    box.setInformativeText(informative_text)
    box.setDetailedText(text)
    box.exec()


def format_cluster_connection_error(config: ClusterConfig, exc: Exception) -> str:
    """Return a user-facing cluster connection error with clear next steps."""

    if isinstance(exc, SSHUnknownHostKeyError):
        return (
            f"The SSH host key for {config.hostname}:{config.port} is not trusted yet.\n\n"
            f"Fingerprint: {exc.fingerprint_sha256}\n"
            f"Algorithm: {exc.algorithm}\n\n"
            "Review the fingerprint carefully and trust it only if it matches the cluster documentation "
            "or a fingerprint from your system administrator."
        )

    if isinstance(exc, SSHHostKeyMismatchError):
        return (
            f"The SSH host key for {config.hostname}:{config.port} does not match the key previously trusted.\n\n"
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

    message = QMessageBox(parent)
    message.setIcon(QMessageBox.Icon.Warning)
    message.setWindowTitle("Trust SSH host key?")
    message.setText(f"Trust the SSH host key for {config.name} ({config.hostname}:{config.port})?")
    message.setInformativeText(
        "Only continue if this fingerprint matches the cluster documentation or a trusted source."
    )
    message.setDetailedText(
        "\n".join(
            [
                f"Cluster: {config.name}",
                f"Host: {config.hostname}",
                f"Port: {config.port}",
                f"Algorithm: {exc.algorithm}",
                f"Fingerprint: {exc.fingerprint_sha256}",
            ]
        )
    )
    message.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    message.setDefaultButton(QMessageBox.StandardButton.No)
    return message.exec() == QMessageBox.StandardButton.Yes
