"""Paramiko-backed SSH and SFTP convenience wrapper."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import logging
import importlib
from pathlib import Path
import posixpath
import shlex
from typing import Any

from mufinder.cluster_setup import ClusterConfig


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrustedHostKeyRecord:
    """Persisted trusted host-key material for one endpoint."""

    algorithm: str
    key_base64: str


class SSHAuthenticationError(ConnectionError):
    """Raised when SSH authentication fails."""


class SSHTransportError(ConnectionError):
    """Raised when the SSH transport or session cannot be established."""


class SSHProtocolBannerError(SSHTransportError):
    """Raised when the remote endpoint does not present an SSH banner."""

    def __init__(self, *, hostname: str, port: int, details: str) -> None:
        self.hostname = hostname
        self.port = int(port)
        self.details = details.strip()
        message = (
            f"The server at {hostname}:{port} accepted the connection but did not send "
            "a valid SSH protocol banner."
        )
        if self.details:
            message = f"{message}\n\nTechnical detail: {self.details}"
        super().__init__(message)


class SSHUnknownHostKeyError(ConnectionError):
    """Raised when the remote host key is not yet trusted."""

    def __init__(
        self,
        *,
        hostname: str,
        port: int,
        algorithm: str,
        fingerprint_sha256: str,
        key_base64: str,
    ) -> None:
        self.hostname = hostname
        self.port = int(port)
        self.algorithm = algorithm
        self.fingerprint_sha256 = fingerprint_sha256
        self.key_base64 = key_base64
        super().__init__(f"SSH host key for {hostname}:{port} is not trusted.")


class SSHHostKeyMismatchError(ConnectionError):
    """Raised when a stored host key does not match the presented one."""

    def __init__(
        self,
        *,
        hostname: str,
        port: int,
        expected_algorithm: str,
        expected_fingerprint_sha256: str,
        presented_algorithm: str,
        presented_fingerprint_sha256: str,
    ) -> None:
        self.hostname = hostname
        self.port = int(port)
        self.expected_algorithm = expected_algorithm
        self.expected_fingerprint_sha256 = expected_fingerprint_sha256
        self.presented_algorithm = presented_algorithm
        self.presented_fingerprint_sha256 = presented_fingerprint_sha256
        super().__init__(f"SSH host key mismatch for {hostname}:{port}.")


def trusted_host_settings_prefix(config: ClusterConfig) -> str:
    """Return the QSettings prefix used for one trusted host entry."""

    endpoint = trusted_host_lookup_name(config)
    return f"trusted_host_keys/{config.name}/{endpoint}"


def trusted_host_lookup_name(config: ClusterConfig) -> str:
    """Return the host-key lookup name used by OpenSSH/Paramiko."""

    hostname = config.hostname.strip()
    port = int(config.port)
    return hostname if port == 22 else f"[{hostname}]:{port}"


def load_trusted_host_key(settings, config: ClusterConfig) -> TrustedHostKeyRecord | None:
    """Load one trusted host key from user-scoped settings."""

    if settings is None:
        return None
    prefix = trusted_host_settings_prefix(config)
    algorithm = str(settings.value(f"{prefix}/algorithm", "", str) or "").strip()
    key_base64 = str(settings.value(f"{prefix}/key_base64", "", str) or "").strip()
    if not algorithm or not key_base64:
        return None
    return TrustedHostKeyRecord(algorithm=algorithm, key_base64=key_base64)


def save_trusted_host_key(settings, config: ClusterConfig, *, algorithm: str, key_base64: str) -> None:
    """Persist one trusted host key in user-scoped settings."""

    if settings is None:
        return
    prefix = trusted_host_settings_prefix(config)
    settings.setValue(f"{prefix}/algorithm", algorithm.strip())
    settings.setValue(f"{prefix}/key_base64", key_base64.strip())


def _is_protocol_banner_error(exc: BaseException) -> bool:
    detail = _format_exception_detail(exc).casefold()
    return "error reading ssh protocol banner" in detail or "ssh protocol banner" in detail


def _format_exception_detail(exc: BaseException) -> str:
    text = str(exc).strip()
    if text:
        return text
    return exc.__class__.__name__


class SSHConnection:
    """Manage one SSH transport and SFTP session for a cluster."""

    def __init__(self, config: ClusterConfig, *, settings=None) -> None:
        self._config = config
        self._settings = settings
        self._client = None
        self._sftp = None

    @property
    def is_connected(self) -> bool:
        if self._client is None:
            return False
        transport = self._client.get_transport()
        return bool(transport and transport.is_active())

    def connect(self, password: str | None = None) -> None:
        """Open SSH and SFTP connections."""

        paramiko = self._import_paramiko()
        ssh_exception_type = getattr(paramiko, "SSHException", RuntimeError)

        if self.is_connected:
            return

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        self._load_app_trusted_host_key(paramiko, client)
        client.set_missing_host_key_policy(_MissingHostKeyPolicy(port=int(self._config.port)))

        hostname = self._config.hostname.strip()
        if not hostname:
            raise ConnectionError("Cluster hostname is required.")

        base_kwargs: dict[str, Any] = {
            "hostname": hostname,
            "port": int(self._config.port),
            "username": self._config.username.strip(),
            "timeout": 20,
            "banner_timeout": 30,
            "auth_timeout": 30,
        }

        key_path = self._config.key_path.strip()

        try:
            if self._config.auth_method == "password":
                if not password:
                    raise SSHAuthenticationError("Password is required for password authentication.")
                client.connect(
                    **base_kwargs,
                    password=password,
                    look_for_keys=False,
                    allow_agent=False,
                )
            else:
                key_kwargs: dict[str, Any] = {
                    "look_for_keys": True,
                    "allow_agent": True,
                }
                if key_path:
                    key_kwargs["key_filename"] = str(Path(key_path).expanduser())
                if password:
                    key_kwargs["passphrase"] = password
                try:
                    client.connect(**base_kwargs, **key_kwargs)
                except paramiko.AuthenticationException as key_error:
                    if not password:
                        raise SSHAuthenticationError(
                            f"SSH key authentication failed: {key_error}"
                        ) from key_error
                    client.connect(
                        **base_kwargs,
                        password=password,
                        look_for_keys=False,
                        allow_agent=False,
                    )
        except SSHUnknownHostKeyError:
            raise
        except paramiko.BadHostKeyException as exc:
            raise SSHHostKeyMismatchError(
                hostname=hostname,
                port=int(self._config.port),
                expected_algorithm=exc.expected_key.get_name(),
                expected_fingerprint_sha256=_host_key_fingerprint_sha256(exc.expected_key),
                presented_algorithm=exc.key.get_name(),
                presented_fingerprint_sha256=_host_key_fingerprint_sha256(exc.key),
            ) from exc
        except paramiko.AuthenticationException as exc:
            raise SSHAuthenticationError(f"SSH authentication failed: {exc}") from exc
        except ssh_exception_type as exc:
            if _is_protocol_banner_error(exc):
                raise SSHProtocolBannerError(
                    hostname=hostname,
                    port=int(self._config.port),
                    details=str(exc),
                ) from exc
            raise SSHTransportError(f"SSH transport negotiation failed: {exc}") from exc
        except ConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001
            if _is_protocol_banner_error(exc):
                raise SSHProtocolBannerError(
                    hostname=hostname,
                    port=int(self._config.port),
                    details=str(exc),
                ) from exc
            raise SSHTransportError(f"SSH connection failed: {_format_exception_detail(exc)}") from exc

        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(60)

        try:
            sftp = client.open_sftp()
        except Exception as exc:  # noqa: BLE001
            client.close()
            raise ConnectionError(f"Failed to open SFTP session: {exc}") from exc

        self._client = client
        self._sftp = sftp
        _LOG.debug("Connected to cluster %s", self._config.name)

    def reconnect(self, password: str | None = None) -> None:
        """Reconnect SSH transport."""

        self.disconnect()
        self.connect(password=password)

    def disconnect(self) -> None:
        """Close SSH and SFTP resources."""

        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:  # noqa: BLE001
                pass
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
        self._client = None
        self._sftp = None
        _LOG.debug("Disconnected from cluster %s", self._config.name)

    def stat(self, remote_path: str):
        """Return stat info for remote path, or None if it does not exist."""

        sftp = self._require_sftp()
        try:
            return sftp.stat(remote_path)
        except IOError:
            return None

    def exists(self, remote_path: str) -> bool:
        return self.stat(remote_path) is not None

    def is_dir(self, remote_path: str) -> bool:
        attributes = self.stat(remote_path)
        if attributes is None:
            return False
        return bool(attributes.st_mode & 0o040000)

    def mkdir(self, remote_path: str, exist_ok: bool = True) -> None:
        """Create remote directory recursively."""

        sftp = self._require_sftp()
        normalized = posixpath.normpath(remote_path)
        if normalized in {"", ".", "/"}:
            return

        segments = normalized.split("/")
        current = "/" if normalized.startswith("/") else ""
        for segment in segments:
            if not segment:
                continue
            current = posixpath.join(current, segment) if current else segment
            try:
                sftp.stat(current)
            except IOError:
                try:
                    sftp.mkdir(current)
                except IOError:
                    if not exist_ok:
                        raise

    def listdir(self, remote_path: str) -> list[str]:
        sftp = self._require_sftp()
        return list(sftp.listdir(remote_path))

    def read_text(self, remote_path: str) -> str:
        sftp = self._require_sftp()
        with sftp.open(remote_path, "rb") as handle:
            data = handle.read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)

    def put(self, local_path: Path, remote_path: str) -> None:
        sftp = self._require_sftp()
        source = Path(local_path)
        if not source.exists():
            raise ConnectionError(f"Local path does not exist: {source}")
        self.mkdir(posixpath.dirname(remote_path) or ".", exist_ok=True)
        sftp.put(str(source), remote_path)

    def get(self, remote_path: str, local_path: Path) -> None:
        sftp = self._require_sftp()
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, str(destination))

    def chmod(self, remote_path: str, mode: int) -> None:
        sftp = self._require_sftp()
        sftp.chmod(remote_path, mode)

    def remove(self, remote_path: str) -> None:
        sftp = self._require_sftp()
        sftp.remove(remote_path)

    def rmtree(self, remote_path: str) -> None:
        sftp = self._require_sftp()
        normalized = posixpath.normpath(remote_path)
        if normalized in {"", ".", "/"}:
            return
        if self.stat(normalized) is None:
            return
        self._rmtree_via_sftp(sftp, normalized)

    def exec_command(self, command: str, timeout: int = 30) -> tuple[int, str, str]:
        """Execute a remote shell command."""

        client = self._require_client()
        _LOG.debug("SSH exec on %s: %s", self._config.name, command)
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        if stdin is not None:
            stdin.close()
        exit_code = int(stdout.channel.recv_exit_status())
        out_text = stdout.read().decode("utf-8", errors="replace")
        err_text = stderr.read().decode("utf-8", errors="replace")
        return exit_code, out_text, err_text

    def resolve_command_path(self, command_name: str, configured_path: str = "") -> str:
        """Resolve a remote command path in the active non-interactive shell."""

        configured = configured_path.strip()
        if configured:
            return configured

        name = command_name.strip()
        if not name:
            return ""

        probe = f"command -v {shlex.quote(name)}"
        exit_code, stdout, _stderr = self.exec_command(probe, timeout=10)
        if exit_code == 0 and stdout.strip():
            resolved = stdout.strip().splitlines()[0].strip()
            if resolved:
                return resolved
        return name

    def _require_client(self):
        if not self.is_connected or self._client is None:
            raise ConnectionError("SSH connection is not active.")
        return self._client

    def _require_sftp(self):
        if not self.is_connected or self._sftp is None:
            raise ConnectionError("SFTP session is not active.")
        return self._sftp

    def _load_app_trusted_host_key(self, paramiko, client) -> None:
        trusted = load_trusted_host_key(self._settings, self._config)
        if trusted is None:
            return

        host_keys = client.get_host_keys()
        lookup = trusted_host_lookup_name(self._config)
        if host_keys.lookup(lookup) is not None:
            return
        host_keys.add(lookup, trusted.algorithm, _decode_host_key(paramiko, trusted))

    def _rmtree_via_sftp(self, sftp, remote_path: str) -> None:
        if self.is_dir(remote_path):
            for entry in self.listdir(remote_path):
                child = posixpath.join(remote_path, entry)
                self._rmtree_via_sftp(sftp, child)
            sftp.rmdir(remote_path)
            return
        sftp.remove(remote_path)

    @staticmethod
    def _import_paramiko():
        try:
            paramiko = importlib.import_module("paramiko")
        except Exception as exc:  # noqa: BLE001
            raise ConnectionError("paramiko is required for SSH support.") from exc
        return paramiko


class _MissingHostKeyPolicy:
    """Raise a structured error for unknown host keys."""

    def __init__(self, *, port: int) -> None:
        self._port = int(port)

    def missing_host_key(self, _client, hostname: str, key) -> None:
        raise SSHUnknownHostKeyError(
            hostname=hostname,
            port=self._port,
            algorithm=key.get_name(),
            fingerprint_sha256=_host_key_fingerprint_sha256(key),
            key_base64=key.get_base64(),
        )


def _host_key_fingerprint_sha256(key) -> str:
    raw = key.asbytes() if hasattr(key, "asbytes") else base64.b64decode(key.get_base64())
    encoded = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii").rstrip("=")
    return f"SHA256:{encoded}"


def _decode_host_key(paramiko, trusted: TrustedHostKeyRecord):
    raw = base64.b64decode(trusted.key_base64.encode("ascii"))
    key_type = trusted.algorithm.strip()
    if key_type == "ssh-rsa":
        return paramiko.RSAKey(data=raw)
    if key_type == "ssh-ed25519":
        return paramiko.Ed25519Key(data=raw)
    if key_type == "ssh-dss":
        return paramiko.DSSKey(data=raw)
    if key_type.startswith("ecdsa-sha2-"):
        return paramiko.ECDSAKey(data=raw)
    raise SSHTransportError(f"Unsupported trusted host key algorithm: {key_type}")
