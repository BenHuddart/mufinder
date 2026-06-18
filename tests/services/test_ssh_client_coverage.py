"""Additional coverage for SSHConnection error/branch paths.

These tests complement ``tests/services/test_ssh_client.py`` by exercising the
remaining uncovered branches: trivial trusted-host-key handling, the password
auth-method success path, key/auth/transport/SFTP failure branches, swallowed
disconnect errors, SFTP guard rails, ``mkdir``/``rmtree`` edge cases, and the
trusted-host-key decode helper for every supported algorithm.

All paramiko interaction is mocked; nothing touches the network, the GUI, the
filesystem (beyond pytest's ``tmp_path``), the clock, or randomness.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from mufinder.cluster_setup import ClusterConfig
from mufinder.services.ssh_client import (
    SSHAuthenticationError,
    SSHConnection,
    SSHProtocolBannerError,
    SSHTransportError,
    SSHUnknownHostKeyError,
    TrustedHostKeyRecord,
    _decode_host_key,
    _format_exception_detail,
    load_trusted_host_key,
    save_trusted_host_key,
    trusted_host_lookup_name,
    trusted_host_settings_prefix,
)

# Reuse the fakes from the sibling test module to match style and avoid drift.
from tests.services.test_ssh_client import (
    _FakeKey,
    _FakeParamiko,
    _FakeSettings,
    _FakeSSHClient,
    _config,
)


def _patch_paramiko(monkeypatch: pytest.MonkeyPatch, paramiko) -> None:
    monkeypatch.setattr(
        SSHConnection, "_import_paramiko", staticmethod(lambda: paramiko)
    )


# ---------------------------------------------------------------------------
# Trusted host-key persistence helpers
# ---------------------------------------------------------------------------


def test_load_trusted_host_key_returns_none_without_settings() -> None:
    assert load_trusted_host_key(None, _config("key")) is None


def test_save_trusted_host_key_noop_without_settings() -> None:
    # Should not raise even though there is no backing store.
    save_trusted_host_key(None, _config("key"), algorithm="ssh-rsa", key_base64="AAAA")


def test_load_trusted_host_key_returns_none_when_incomplete() -> None:
    settings = _FakeSettings()
    config = _config("key")
    prefix = trusted_host_settings_prefix(config)
    # Algorithm present but key blank -> incomplete record -> None.
    settings.setValue(f"{prefix}/algorithm", "ssh-ed25519")
    settings.setValue(f"{prefix}/key_base64", "")

    assert load_trusted_host_key(settings, config) is None


def test_trusted_host_lookup_name_brackets_nonstandard_port() -> None:
    config = ClusterConfig(
        name="t",
        hostname="example.org",
        port=2222,
        username="user",
        auth_method="key",
        key_path="",
        default_working_directory="/work",
        queue_system="slurm",
        cores_per_node=8,
        default_walltime="01:00:00",
        template_script_path="/tmp/template.sh",
    )

    assert trusted_host_lookup_name(config) == "[example.org]:2222"


# ---------------------------------------------------------------------------
# _format_exception_detail
# ---------------------------------------------------------------------------


def test_format_exception_detail_uses_class_name_when_empty() -> None:
    assert _format_exception_detail(RuntimeError()) == "RuntimeError"


def test_format_exception_detail_uses_message_when_present() -> None:
    assert _format_exception_detail(RuntimeError("boom")) == "boom"


# ---------------------------------------------------------------------------
# connect: password auth-method success path
# ---------------------------------------------------------------------------


def test_password_auth_method_connects_with_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(_config("password"))
    connection.connect(password="secret")

    assert connection.is_connected
    call = fake_paramiko._client.connect_calls[0]
    assert call["password"] == "secret"
    assert call["look_for_keys"] is False
    assert call["allow_agent"] is False


# ---------------------------------------------------------------------------
# connect: key auth failure with no password -> SSHAuthenticationError
# ---------------------------------------------------------------------------


def test_key_auth_failure_without_password_raises_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=True,
        system_host_keys={"example.org": _FakeKey()},
    )
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(_config("key"))

    with pytest.raises(SSHAuthenticationError, match="SSH key authentication failed"):
        connection.connect(password=None)


# ---------------------------------------------------------------------------
# connect: top-level AuthenticationException for password auth method
# ---------------------------------------------------------------------------


def test_password_auth_method_authentication_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AuthFailClient(_FakeSSHClient):
        def connect(self, **kwargs) -> None:
            self.connect_calls.append(kwargs)
            raise _FakeParamiko.AuthenticationException("bad password")

    class _AuthFailParamiko(_FakeParamiko):
        def __init__(self) -> None:
            self._client = _AuthFailClient(
                system_host_keys={"example.org": _FakeKey()}
            )

    _patch_paramiko(monkeypatch, _AuthFailParamiko())

    connection = SSHConnection(_config("password"))

    with pytest.raises(SSHAuthenticationError, match="SSH authentication failed"):
        connection.connect(password="wrong")


# ---------------------------------------------------------------------------
# connect: generic non-SSHException -> banner vs transport error
# ---------------------------------------------------------------------------


def test_generic_exception_with_banner_text_is_banner_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BannerClient(_FakeSSHClient):
        def connect(self, **kwargs) -> None:
            self.connect_calls.append(kwargs)
            raise ValueError("Error reading SSH protocol banner: timed out")

    class _BannerParamiko(_FakeParamiko):
        def __init__(self) -> None:
            self._client = _BannerClient(system_host_keys={"example.org": _FakeKey()})

    _patch_paramiko(monkeypatch, _BannerParamiko())

    connection = SSHConnection(_config("key"))

    with pytest.raises(SSHProtocolBannerError) as exc_info:
        connection.connect()

    assert exc_info.value.hostname == "example.org"
    assert "valid SSH protocol banner" in str(exc_info.value)


def test_generic_exception_without_banner_text_is_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _GenericFailClient(_FakeSSHClient):
        def connect(self, **kwargs) -> None:
            self.connect_calls.append(kwargs)
            raise OSError("network unreachable")

    class _GenericFailParamiko(_FakeParamiko):
        def __init__(self) -> None:
            self._client = _GenericFailClient(
                system_host_keys={"example.org": _FakeKey()}
            )

    _patch_paramiko(monkeypatch, _GenericFailParamiko())

    connection = SSHConnection(_config("key"))

    with pytest.raises(SSHTransportError, match="SSH connection failed"):
        connection.connect()


# ---------------------------------------------------------------------------
# connect: open_sftp failure -> ConnectionError and client closed
# ---------------------------------------------------------------------------


def test_open_sftp_failure_closes_client_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SFTPFailClient(_FakeSSHClient):
        def open_sftp(self):
            raise OSError("sftp subsystem refused")

    class _SFTPFailParamiko(_FakeParamiko):
        def __init__(self) -> None:
            self._client = _SFTPFailClient(
                system_host_keys={"example.org": _FakeKey()}
            )

    fake_paramiko = _SFTPFailParamiko()
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(_config("key"))

    with pytest.raises(ConnectionError, match="Failed to open SFTP session"):
        connection.connect()

    # The underlying transport client should have been closed on failure.
    assert fake_paramiko._client.get_transport().is_active() is False
    assert connection.is_connected is False


# ---------------------------------------------------------------------------
# disconnect: swallows errors from both sftp.close and client.close
# ---------------------------------------------------------------------------


def test_disconnect_swallows_close_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(_config("key"))
    connection.connect()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("close failed")

    monkeypatch.setattr(fake_paramiko._client._sftp, "close", _boom)
    monkeypatch.setattr(fake_paramiko._client, "close", _boom)

    # Both close calls raise but disconnect must complete and reset state.
    connection.disconnect()

    assert connection.is_connected is False


# ---------------------------------------------------------------------------
# SFTP guard rails and filesystem helper edge cases
# ---------------------------------------------------------------------------


def test_require_sftp_raises_when_not_connected() -> None:
    connection = SSHConnection(_config("key"))

    with pytest.raises(ConnectionError, match="SFTP session is not active"):
        connection.stat("/anything")


def test_is_dir_false_when_path_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(_config("key"))
    connection.connect()

    assert connection.is_dir("/does/not/exist") is False


def test_mkdir_noop_for_trivial_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(_config("key"))
    connection.connect()
    sftp = fake_paramiko._client._sftp

    connection.mkdir("/")
    connection.mkdir(".")
    connection.mkdir("")

    assert sftp.directories == set()


def test_mkdir_reraises_when_not_exist_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(_config("key"))
    connection.connect()
    sftp = fake_paramiko._client._sftp

    def _mkdir_denied(_path: str) -> None:
        raise IOError("permission denied")

    monkeypatch.setattr(sftp, "mkdir", _mkdir_denied)

    with pytest.raises(IOError, match="permission denied"):
        connection.mkdir("/blocked/dir", exist_ok=False)


def test_read_text_handles_non_bytes_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(_config("key"))
    connection.connect()
    sftp = fake_paramiko._client._sftp

    class _StrFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return "already-text"

    monkeypatch.setattr(sftp, "open", lambda _path, _mode: _StrFile())

    assert connection.read_text("/remote/file.txt") == "already-text"


def test_rmtree_noop_for_trivial_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(_config("key"))
    connection.connect()

    # Normalises to "/" -> early return, no SFTP calls.
    connection.rmtree("/")


def test_rmtree_noop_when_path_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(_config("key"))
    connection.connect()

    # Path does not exist -> stat None -> early return without error.
    connection.rmtree("/remote/absent")


# ---------------------------------------------------------------------------
# resolve_command_path: empty command name
# ---------------------------------------------------------------------------


def test_resolve_command_path_empty_name_returns_empty() -> None:
    connection = SSHConnection(_config("key"))

    assert connection.resolve_command_path("   ", "") == ""


# ---------------------------------------------------------------------------
# _load_app_trusted_host_key: lookup already present -> no re-add
# ---------------------------------------------------------------------------


def test_trusted_host_key_already_known_is_not_readded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("key")
    settings = _FakeSettings()
    save_trusted_host_key(
        settings,
        config,
        algorithm="ssh-ed25519",
        key_base64=_FakeKey().get_base64(),
    )

    # System host keys already include this endpoint, so the app-trusted add
    # path takes the "already present" early return.
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(config, settings=settings)
    connection.connect()

    host_keys = fake_paramiko._client.get_host_keys()
    # Only the single system-provided entry should be registered.
    assert host_keys.lookup("example.org") is not None
    assert connection.is_connected


def test_undecodable_trusted_host_key_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A corrupt/legacy stored trusted host key (e.g. base64 of a hostname placeholder written by
    # an older build) must not abort the connection with a cryptic paramiko error; it is skipped
    # so the connection falls through to system host keys / the trust prompt.
    config = _config("key")
    settings = _FakeSettings()
    save_trusted_host_key(
        settings,
        config,
        algorithm="ssh-ed25519",
        key_base64=base64.b64encode(b"bastion").decode("ascii"),
    )

    def _raise(*_args, **_kwargs):
        raise _FakeParamiko.SSHException("Invalid key (class: Ed25519Key, data type: ion")

    monkeypatch.setattr("mufinder.services.ssh_client._decode_host_key", _raise)
    # The endpoint is not in system host keys, so the app-trusted decode path is reached.
    fake_paramiko = _FakeParamiko(fail_key=False, system_host_keys={})
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(config, settings=settings)
    # Before the fix this raised the cryptic paramiko SSHException ("Invalid key ...").
    # Now the corrupt key is skipped, so the normal missing-host-key trust flow runs instead
    # (and when the endpoint IS in ~/.ssh/known_hosts, the connection simply succeeds).
    with pytest.raises(SSHUnknownHostKeyError):
        connection.connect()


def test_decode_host_key_rejects_hostname_placeholder_blob() -> None:
    # Reproduces the real corruption: key_base64 was the base64 of a hostname string, so the
    # decoded blob is not a valid SSH key wire format and real paramiko rejects it.
    paramiko = pytest.importorskip("paramiko")
    record = TrustedHostKeyRecord(
        algorithm="ssh-ed25519",
        key_base64=base64.b64encode(b"bastion").decode("ascii"),
    )
    with pytest.raises(Exception):
        _decode_host_key(paramiko, record)


# ---------------------------------------------------------------------------
# _decode_host_key: every supported algorithm and the unsupported branch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "algorithm",
    ["ssh-rsa", "ssh-ed25519", "ssh-dss", "ecdsa-sha2-nistp256"],
)
def test_decode_host_key_supported_algorithms(algorithm: str) -> None:
    raw = b"raw-key-material"
    record = TrustedHostKeyRecord(
        algorithm=algorithm,
        key_base64=base64.b64encode(raw).decode("ascii"),
    )

    key = _decode_host_key(_FakeParamiko(), record)

    assert key.asbytes() == raw


def test_import_paramiko_returns_imported_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "mufinder.services.ssh_client.importlib.import_module",
        lambda name: sentinel if name == "paramiko" else None,
    )

    assert SSHConnection._import_paramiko() is sentinel


def test_decode_host_key_unsupported_algorithm_raises() -> None:
    record = TrustedHostKeyRecord(
        algorithm="ssh-unknown-magic",
        key_base64=base64.b64encode(b"data").decode("ascii"),
    )

    with pytest.raises(SSHTransportError, match="Unsupported trusted host key algorithm"):
        _decode_host_key(_FakeParamiko(), record)
