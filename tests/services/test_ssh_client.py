"""Tests for SSHConnection wrapper."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from mufinder.domain.cluster import ClusterConfig
from mufinder.services.ssh_client import (
    SSHCredentials,
    SSHAuthenticationError,
    SSHConnection,
    SSHHostKeyMismatchError,
    SSHProtocolBannerError,
    SSHTransportError,
    SSHUnknownHostKeyError,
    TrustedHostKeyRecord,
    _decode_host_key,
    load_trusted_host_key,
    save_trusted_host_key,
    trusted_host_lookup_name,
    trusted_host_settings_prefix,
)


class _FakeTransport:
    def __init__(self) -> None:
        self.active = True
        self.keepalive = None
        self.opened_channels: list[tuple[str, tuple[str, int], tuple[str, int], object]] = []

    def is_active(self) -> bool:
        return self.active

    def set_keepalive(self, interval: int) -> None:
        self.keepalive = interval

    def open_channel(self, kind: str, destination: tuple[str, int], source: tuple[str, int]):
        channel = _FakeDirectTcpIpChannel(kind, destination, source)
        self.opened_channels.append((kind, destination, source, channel))
        return channel


class _FakeDirectTcpIpChannel:
    def __init__(self, kind: str, destination: tuple[str, int], source: tuple[str, int]) -> None:
        self.kind = kind
        self.destination = destination
        self.source = source
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeChannel:
    def recv_exit_status(self) -> int:
        return 0


class _FakeStream:
    def __init__(self, payload: bytes = b"") -> None:
        self._payload = payload
        self.channel = _FakeChannel()

    def read(self) -> bytes:
        return self._payload


class _FakeStdin:
    def close(self) -> None:
        return


class _FakeSFTPFile:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._position = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def seek(self, offset: int) -> None:
        self._position = offset

    def read(self) -> bytes:
        return self._data[self._position :]


class _FakeSFTP:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = set()
        # Symlink path -> target path.
        self.links: dict[str, str] = {}
        self.chmod_calls: list[tuple[str, int]] = []

    def close(self) -> None:
        return

    def stat(self, path: str):
        if path in self.links:
            return self.stat(self.links[path])
        return self.lstat(path)

    def lstat(self, path: str):
        if path in self.links:
            return type("Attr", (), {"st_mode": 0o120777})()
        if path in self.directories:
            return type("Attr", (), {"st_mode": 0o040755})()
        if path in self.files:
            return type("Attr", (), {"st_mode": 0o100644})()
        raise IOError("not found")

    def mkdir(self, path: str) -> None:
        self.directories.add(path)

    def listdir(self, path: str) -> list[str]:
        path = self.links.get(path, path)
        prefix = path.rstrip("/") + "/"
        entries: list[str] = []
        for entry_path in (*self.files, *self.links):
            if entry_path.startswith(prefix):
                entries.append(entry_path[len(prefix) :].split("/")[0])
        return sorted(set(entries))

    def open(self, path: str, _mode: str):
        return _FakeSFTPFile(self.files.get(path, b""))

    def put(self, local_path: str, remote_path: str) -> None:
        self.files[remote_path] = Path(local_path).read_bytes()

    def get(self, remote_path: str, local_path: str) -> None:
        Path(local_path).write_bytes(self.files.get(remote_path, b""))

    def chmod(self, path: str, mode: int) -> None:
        self.chmod_calls.append((path, mode))

    def remove(self, path: str) -> None:
        if self.links.pop(path, None) is None:
            self.files.pop(path, None)

    def rmdir(self, path: str) -> None:
        self.directories.discard(path)


class _FakeHostKeys:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, object]] = {}

    def add(self, hostname: str, keytype: str, key) -> None:
        self._entries.setdefault(hostname, {})[keytype] = key

    def lookup(self, hostname: str):
        return self._entries.get(hostname)


class _FakeKey:
    def __init__(self, keytype: str = "ssh-ed25519", raw: bytes = b"server-key") -> None:
        self._keytype = keytype
        self._raw = raw

    def get_name(self) -> str:
        return self._keytype

    def get_base64(self) -> str:
        return base64.b64encode(self._raw).decode("ascii")

    def asbytes(self) -> bytes:
        return self._raw


class _FakeSSHClient:
    def __init__(self, fail_key: bool = False, system_host_keys: dict[str, object] | None = None) -> None:
        self._transport = _FakeTransport()
        self._sftp = _FakeSFTP()
        self.fail_key = fail_key
        self.connect_calls: list[dict] = []
        self._policy = None
        self._system_host_keys = system_host_keys or {}
        self._host_keys = _FakeHostKeys()
        self._server_key = _FakeKey()

    def load_system_host_keys(self) -> None:
        for hostname, key in self._system_host_keys.items():
            self._host_keys.add(hostname, key.get_name(), key)

    def get_host_keys(self):
        return self._host_keys

    def set_missing_host_key_policy(self, policy) -> None:
        self._policy = policy

    def connect(self, **kwargs) -> None:
        self.connect_calls.append(kwargs)
        hostname = kwargs["hostname"]
        port = int(kwargs["port"])
        lookup = hostname if port == 22 else f"[{hostname}]:{port}"
        known = self._host_keys.lookup(lookup)
        if known is None:
            self._policy.missing_host_key(self, hostname, self._server_key)
        expected = known.get(self._server_key.get_name()) if known is not None else None
        if expected is not None and expected.get_base64() != self._server_key.get_base64():
            raise _FakeParamiko.BadHostKeyException(hostname, self._server_key, expected)
        if self.fail_key and kwargs.get("password") is None:
            raise _FakeParamiko.AuthenticationException("key auth failed")

    def open_sftp(self):
        return self._sftp

    def get_transport(self):
        return self._transport

    def close(self) -> None:
        self._transport.active = False

    def exec_command(self, _command: str, timeout: int = 30):
        _ = timeout
        return _FakeStdin(), _FakeStream(b"ok"), _FakeStream(b"")


class _FakeParamiko:
    def __init__(self, fail_key: bool = False, system_host_keys: dict[str, object] | None = None) -> None:
        self._client = _FakeSSHClient(fail_key=fail_key, system_host_keys=system_host_keys)

    def SSHClient(self):  # noqa: N802
        return self._client

    class AuthenticationException(Exception):
        pass

    class SSHException(Exception):
        pass

    class BadHostKeyException(Exception):
        def __init__(self, hostname: str, key, expected_key) -> None:
            super().__init__(hostname)
            self.hostname = hostname
            self.key = key
            self.expected_key = expected_key

    class RSAKey:
        def __init__(self, *, data: bytes) -> None:
            self._key = _FakeKey("ssh-rsa", raw=data)

        def get_name(self) -> str:
            return self._key.get_name()

        def get_base64(self) -> str:
            return self._key.get_base64()

        def asbytes(self) -> bytes:
            return self._key.asbytes()

    class Ed25519Key:
        def __init__(self, *, data: bytes) -> None:
            self._key = _FakeKey("ssh-ed25519", raw=data)

        def get_name(self) -> str:
            return self._key.get_name()

        def get_base64(self) -> str:
            return self._key.get_base64()

        def asbytes(self) -> bytes:
            return self._key.asbytes()

    class DSSKey:
        def __init__(self, *, data: bytes) -> None:
            self._key = _FakeKey("ssh-dss", raw=data)

        def get_name(self) -> str:
            return self._key.get_name()

        def get_base64(self) -> str:
            return self._key.get_base64()

        def asbytes(self) -> bytes:
            return self._key.asbytes()

    class ECDSAKey:
        def __init__(self, *, data: bytes) -> None:
            self._key = _FakeKey("ecdsa-sha2-nistp256", raw=data)

        def get_name(self) -> str:
            return self._key.get_name()

        def get_base64(self) -> str:
            return self._key.get_base64()

        def asbytes(self) -> bytes:
            return self._key.asbytes()


class _FakeSettings:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def value(self, key: str, default=None, value_type=None):  # noqa: ANN001
        value = self._values.get(key, default)
        if value_type is str:
            return str(value)
        return value

    def setValue(self, key: str, value) -> None:  # noqa: ANN001, N802
        self._values[key] = value


def _config(auth_method: str = "key") -> ClusterConfig:
    return ClusterConfig(
        name="test",
        hostname="example.org",
        port=22,
        username="user",
        auth_method=auth_method,
        key_path="/tmp/id_rsa",
        default_working_directory="/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/tmp/template.sh",
    )


def _jump_config() -> ClusterConfig:
    return ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        port=22,
        username="glam-user",
        auth_method="password",
        jump_host_enabled=True,
        jump_hostname="bastion.physics.ox.ac.uk",
        jump_port=22,
        jump_username="phys-user",
        jump_auth_method="password",
        default_working_directory="/mnt/redwoodfs/glam-user",
        queue_system="ADDQUEUE",
        cores_per_node=32,
        default_walltime="",
    )


def _patch_paramiko(monkeypatch: pytest.MonkeyPatch, paramiko) -> None:
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: paramiko))


def _connected(monkeypatch: pytest.MonkeyPatch) -> tuple[SSHConnection, _FakeParamiko]:
    """Return a key-authenticated connection to a host the system already trusts."""

    fake_paramiko = _FakeParamiko(fail_key=False, system_host_keys={"example.org": _FakeKey()})
    _patch_paramiko(monkeypatch, fake_paramiko)
    connection = SSHConnection(_config("key"))
    connection.connect()
    return connection, fake_paramiko


def _failing_connect_paramiko(error: BaseException) -> _FakeParamiko:
    """Return a fake paramiko whose client raises ``error`` from ``connect``."""

    class _FailingClient(_FakeSSHClient):
        def connect(self, **kwargs) -> None:
            self.connect_calls.append(kwargs)
            raise error

    fake_paramiko = _FakeParamiko()
    fake_paramiko._client = _FailingClient(system_host_keys={"example.org": _FakeKey()})
    return fake_paramiko


def test_ssh_connection_connects_with_key_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))
    connection.connect()

    assert connection.is_connected
    assert fake_paramiko._client.get_transport().keepalive == 60
    assert fake_paramiko._client.connect_calls[0]["banner_timeout"] == 30
    assert fake_paramiko._client.connect_calls[0]["auth_timeout"] == 30


def test_ssh_connection_connects_via_jump_host(monkeypatch: pytest.MonkeyPatch) -> None:
    jump_client = _FakeSSHClient(system_host_keys={"bastion.physics.ox.ac.uk": _FakeKey()})
    target_client = _FakeSSHClient(system_host_keys={"glamdring.physics.ox.ac.uk": _FakeKey()})

    class _JumpParamiko(_FakeParamiko):
        def __init__(self) -> None:
            self._clients = [jump_client, target_client]
            self._client = target_client

        def SSHClient(self):  # noqa: N802
            return self._clients.pop(0)

    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: _JumpParamiko()))

    connection = SSHConnection(_jump_config())
    connection.connect(credentials=SSHCredentials(target_secret="glam-secret", jump_secret="phys-secret"))

    assert connection.is_connected
    assert jump_client.connect_calls[0]["hostname"] == "bastion.physics.ox.ac.uk"
    assert jump_client.connect_calls[0]["username"] == "phys-user"
    assert jump_client.connect_calls[0]["password"] == "phys-secret"
    assert target_client.connect_calls[0]["hostname"] == "glamdring.physics.ox.ac.uk"
    assert target_client.connect_calls[0]["username"] == "glam-user"
    assert target_client.connect_calls[0]["password"] == "glam-secret"
    channel = jump_client.get_transport().opened_channels[0][3]
    assert jump_client.get_transport().opened_channels[0][:3] == (
        "direct-tcpip",
        ("glamdring.physics.ox.ac.uk", 22),
        ("127.0.0.1", 0),
    )
    assert target_client.connect_calls[0]["sock"] is channel


def test_ssh_connection_jump_host_unknown_key_raises_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(fail_key=False)
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_jump_config())

    with pytest.raises(SSHUnknownHostKeyError) as exc_info:
        connection.connect(credentials=SSHCredentials(target_secret="glam-secret", jump_secret="phys-secret"))

    assert exc_info.value.endpoint == "jump"
    assert exc_info.value.hostname == "bastion.physics.ox.ac.uk"


def test_ssh_connection_jump_host_password_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"bastion.physics.ox.ac.uk": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_jump_config())

    with pytest.raises(SSHAuthenticationError, match="jump host authentication"):
        connection.connect(credentials=SSHCredentials(target_secret="glam-secret"))


def test_ssh_connection_uses_saved_jump_host_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jump_client = _FakeSSHClient()
    target_client = _FakeSSHClient(system_host_keys={"glamdring.physics.ox.ac.uk": _FakeKey()})

    class _JumpParamiko(_FakeParamiko):
        def __init__(self) -> None:
            self._clients = [jump_client, target_client]
            self._client = target_client

        def SSHClient(self):  # noqa: N802
            return self._clients.pop(0)

    settings = _FakeSettings()
    config = _jump_config()
    save_trusted_host_key(
        settings,
        config,
        algorithm="ssh-ed25519",
        key_base64=_FakeKey().get_base64(),
        endpoint="jump",
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: _JumpParamiko()))

    connection = SSHConnection(config, settings=settings)
    connection.connect(credentials=SSHCredentials(target_secret="glam-secret", jump_secret="phys-secret"))

    loaded = load_trusted_host_key(settings, config, endpoint="jump")
    assert loaded is not None
    assert jump_client.connect_calls[0]["hostname"] == "bastion.physics.ox.ac.uk"


def test_ssh_connection_falls_back_to_password(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=True,
        system_host_keys={"example.org": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))
    connection.connect(password="secret")

    assert connection.is_connected
    assert len(fake_paramiko._client.connect_calls) == 2
    assert fake_paramiko._client.connect_calls[-1].get("password") == "secret"


def test_ssh_connection_password_auth_requires_password(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("password"))

    with pytest.raises(SSHAuthenticationError, match="Password is required"):
        connection.connect(password=None)


def test_ssh_connection_password_auth_sends_only_the_password(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(fail_key=False, system_host_keys={"example.org": _FakeKey()})
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(_config("password"))
    connection.connect(password="secret")

    assert connection.is_connected
    call = fake_paramiko._client.connect_calls[0]
    assert call["password"] == "secret"
    assert call["look_for_keys"] is False
    assert call["allow_agent"] is False


def test_ssh_connection_key_auth_failure_without_password_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_paramiko(
        monkeypatch,
        _FakeParamiko(fail_key=True, system_host_keys={"example.org": _FakeKey()}),
    )

    with pytest.raises(SSHAuthenticationError, match="SSH key authentication failed"):
        SSHConnection(_config("key")).connect(password=None)


def test_ssh_connection_password_auth_failure_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_paramiko(
        monkeypatch,
        _failing_connect_paramiko(_FakeParamiko.AuthenticationException("bad password")),
    )

    with pytest.raises(SSHAuthenticationError, match="SSH authentication failed"):
        SSHConnection(_config("password")).connect(password="wrong")


def test_ssh_connection_file_operations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))
    connection.connect()

    local_file = tmp_path / "local.txt"
    local_file.write_text("hello", encoding="utf-8")

    connection.mkdir("/remote/work/dir")
    connection.put(local_file, "/remote/work/dir/local.txt")
    assert connection.exists("/remote/work/dir/local.txt")
    assert connection.is_file("/remote/work/dir/local.txt") is True
    assert connection.is_file("/remote/work/dir") is False
    assert connection.is_dir("/remote/work/dir") is True

    text = connection.read_text("/remote/work/dir/local.txt")
    assert text == "hello"
    assert connection.read_bytes("/remote/work/dir/local.txt") == b"hello"
    assert connection.read_bytes("/remote/work/dir/local.txt", 2) == b"llo"

    out_file = tmp_path / "out.txt"
    connection.get("/remote/work/dir/local.txt", out_file)
    assert out_file.read_text(encoding="utf-8") == "hello"

    connection.chmod("/remote/work/dir/local.txt", 0o755)
    assert fake_paramiko._client._sftp.chmod_calls[-1] == ("/remote/work/dir/local.txt", 0o755)

    connection.remove("/remote/work/dir/local.txt")
    assert not connection.exists("/remote/work/dir/local.txt")


def test_ssh_connection_file_operations_require_an_sftp_session() -> None:
    connection = SSHConnection(_config("key"))

    with pytest.raises(ConnectionError, match="SFTP session is not active"):
        connection.stat("/anything")


def test_ssh_connection_missing_path_is_neither_file_nor_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, _fake_paramiko = _connected(monkeypatch)

    assert connection.is_dir("/does/not/exist") is False
    assert connection.is_file("/does/not/exist") is False


def test_ssh_connection_mtime_reports_the_remote_modification_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, fake_paramiko = _connected(monkeypatch)
    monkeypatch.setattr(
        fake_paramiko._client._sftp,
        "stat",
        lambda _path: type("Attr", (), {"st_mode": 0o100644, "st_mtime": 1_700_000_000})(),
    )

    assert connection.mtime("/remote/job/seed.param") == 1_700_000_000.0


@pytest.mark.parametrize(
    "attributes",
    [
        None,
        type("NoMtime", (), {"st_mode": 0o100644})(),
        type("BadMtime", (), {"st_mode": 0o100644, "st_mtime": "not-a-time"})(),
    ],
    ids=["missing-path", "no-mtime-attribute", "unparseable-mtime"],
)
def test_ssh_connection_mtime_is_none_when_the_server_reports_nothing_usable(
    monkeypatch: pytest.MonkeyPatch, attributes: object
) -> None:
    """Callers order two remote mtimes against each other, so "unknown" must be
    distinguishable from an ordering and never coerced into one."""

    connection, fake_paramiko = _connected(monkeypatch)
    if attributes is not None:
        monkeypatch.setattr(fake_paramiko._client._sftp, "stat", lambda _path: attributes)

    assert connection.mtime("/remote/job/seed.param") is None


def test_ssh_connection_mkdir_ignores_root_and_empty_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    connection, fake_paramiko = _connected(monkeypatch)

    connection.mkdir("/")
    connection.mkdir(".")
    connection.mkdir("")

    assert fake_paramiko._client._sftp.directories == set()


def test_ssh_connection_mkdir_reraises_when_not_exist_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    connection, fake_paramiko = _connected(monkeypatch)

    def _mkdir_denied(_path: str) -> None:
        raise IOError("permission denied")

    monkeypatch.setattr(fake_paramiko._client._sftp, "mkdir", _mkdir_denied)

    with pytest.raises(IOError, match="permission denied"):
        connection.mkdir("/blocked/dir", exist_ok=False)


def test_ssh_connection_read_text_accepts_a_text_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    connection, fake_paramiko = _connected(monkeypatch)

    class _TextFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return "already-text"

    monkeypatch.setattr(fake_paramiko._client._sftp, "open", lambda _path, _mode: _TextFile())

    assert connection.read_text("/remote/file.txt") == "already-text"


def test_ssh_connection_exec_command(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))
    connection.connect()

    code, stdout, stderr = connection.exec_command("echo ok")

    assert code == 0
    assert stdout == "ok"
    assert stderr == ""


def test_ssh_connection_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))
    connection.connect()
    connection.disconnect()

    assert not connection.is_connected


def test_ssh_connection_disconnect_swallows_close_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    connection, fake_paramiko = _connected(monkeypatch)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("close failed")

    monkeypatch.setattr(fake_paramiko._client._sftp, "close", _boom)
    monkeypatch.setattr(fake_paramiko._client, "close", _boom)

    connection.disconnect()

    assert connection.is_connected is False


def test_ssh_connection_requires_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_paramiko = _FakeParamiko(fail_key=False)
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    config = _config("key")
    config = ClusterConfig(
        name=config.name,
        hostname="",
        port=config.port,
        username=config.username,
        auth_method=config.auth_method,
        key_path=config.key_path,
        default_working_directory=config.default_working_directory,
        queue_system=config.queue_system,
        cores_per_node=config.cores_per_node,
        default_walltime=config.default_walltime,
        template_script_path=config.template_script_path,
    )
    connection = SSHConnection(config)

    with pytest.raises(ConnectionError, match="hostname"):
        connection.connect()


def test_ssh_connection_connects_again_after_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FreshParamiko:
        AuthenticationException = _FakeParamiko.AuthenticationException
        SSHException = _FakeParamiko.SSHException
        BadHostKeyException = _FakeParamiko.BadHostKeyException
        RSAKey = _FakeParamiko.RSAKey
        Ed25519Key = _FakeParamiko.Ed25519Key
        DSSKey = _FakeParamiko.DSSKey
        ECDSAKey = _FakeParamiko.ECDSAKey

        def SSHClient(self):  # noqa: N802
            return _FakeSSHClient(
                fail_key=False,
                system_host_keys={"example.org": _FakeKey()},
            )

    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: _FreshParamiko()))

    connection = SSHConnection(_config("key"))
    connection.connect()
    connection.disconnect()

    assert not connection.is_connected

    connection.connect()

    assert connection.is_connected


def test_ssh_connection_put_requires_existing_local_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))
    connection.connect()

    with pytest.raises(ConnectionError, match="Local path does not exist"):
        connection.put(tmp_path / "missing.txt", "/remote/missing.txt")


def test_ssh_connection_rmtree_uses_recursive_sftp_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))
    connection.connect()
    sftp = fake_paramiko._client._sftp
    sftp.directories.update({"/remote/path", "/remote/path/dir"})
    sftp.files["/remote/path/dir/file.txt"] = b"hello"

    connection.rmtree("/remote/path")

    assert "/remote/path" not in sftp.directories
    assert "/remote/path/dir" not in sftp.directories
    assert "/remote/path/dir/file.txt" not in sftp.files


def test_ssh_connection_rmtree_removes_symlinks_without_following_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))
    connection.connect()
    sftp = fake_paramiko._client._sftp
    sftp.directories.update({"/remote/job", "/shared/pseudos"})
    sftp.files["/remote/job/seed.castep"] = b"output"
    sftp.files["/shared/pseudos/H.usp"] = b"pseudopotential"
    sftp.links["/remote/job/pseudos"] = "/shared/pseudos"

    connection.rmtree("/remote/job")

    assert "/remote/job" not in sftp.directories
    assert "/remote/job/pseudos" not in sftp.links
    assert "/remote/job/seed.castep" not in sftp.files
    # The linked directory and its contents survive.
    assert "/shared/pseudos" in sftp.directories
    assert sftp.files["/shared/pseudos/H.usp"] == b"pseudopotential"


def test_ssh_connection_rmtree_tolerates_entries_that_vanish_mid_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))
    connection.connect()
    sftp = fake_paramiko._client._sftp
    sftp.directories.add("/remote/job")
    sftp.files["/remote/job/seed.castep"] = b"output"
    sftp.files["/remote/job/seed.check"] = b"checkpoint"
    real_lstat = sftp.lstat

    def _lstat(path: str):
        if path == "/remote/job/seed.check":
            sftp.files.pop(path, None)  # removed by something else just now
            raise FileNotFoundError(path)
        return real_lstat(path)

    sftp.lstat = _lstat

    connection.rmtree("/remote/job")

    assert "/remote/job" not in sftp.directories
    assert sftp.files == {}


@pytest.mark.parametrize("remote_path", ["/", "/remote/absent"], ids=["root", "missing"])
def test_ssh_connection_rmtree_leaves_root_and_missing_paths_alone(
    monkeypatch: pytest.MonkeyPatch, remote_path: str
) -> None:
    connection, fake_paramiko = _connected(monkeypatch)
    sftp = fake_paramiko._client._sftp
    sftp.directories.add("/remote")

    connection.rmtree(remote_path)

    assert sftp.directories == {"/remote"}


def test_ssh_connection_requires_active_connection_for_exec() -> None:
    connection = SSHConnection(_config("key"))

    with pytest.raises(ConnectionError, match="not active"):
        connection.exec_command("echo test")


def test_ssh_connection_unknown_host_key_raises_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(fail_key=False)
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))

    with pytest.raises(SSHUnknownHostKeyError) as exc_info:
        connection.connect()

    assert exc_info.value.algorithm == "ssh-ed25519"
    assert exc_info.value.fingerprint_sha256.startswith("SHA256:")


def test_ssh_connection_uses_saved_trusted_host_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(fail_key=False)
    settings = _FakeSettings()
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    config = _config("key")
    save_trusted_host_key(
        settings,
        config,
        algorithm="ssh-ed25519",
        key_base64=_FakeKey().get_base64(),
    )

    connection = SSHConnection(config, settings=settings)
    connection.connect()

    loaded = load_trusted_host_key(settings, config)
    assert loaded is not None
    assert loaded.algorithm == "ssh-ed25519"
    assert connection.is_connected


def test_ssh_connection_host_key_mismatch_raises_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(fail_key=False)
    settings = _FakeSettings()
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    config = _config("key")
    save_trusted_host_key(
        settings,
        config,
        algorithm="ssh-ed25519",
        key_base64=_FakeKey(raw=b"old-key").get_base64(),
    )

    connection = SSHConnection(config, settings=settings)

    with pytest.raises(SSHHostKeyMismatchError) as exc_info:
        connection.connect()

    assert exc_info.value.expected_fingerprint_sha256.startswith("SHA256:")
    assert exc_info.value.presented_fingerprint_sha256.startswith("SHA256:")


def test_trusted_host_key_storage_is_inert_without_settings_or_a_complete_record() -> None:
    config = _config("key")
    # Neither call needs a backing store.
    save_trusted_host_key(None, config, algorithm="ssh-rsa", key_base64="AAAA")
    assert load_trusted_host_key(None, config) is None

    settings = _FakeSettings()
    prefix = trusted_host_settings_prefix(config)
    settings.setValue(f"{prefix}/algorithm", "ssh-ed25519")
    settings.setValue(f"{prefix}/key_base64", "")

    assert load_trusted_host_key(settings, config) is None


def test_trusted_host_lookup_name_brackets_a_nonstandard_port() -> None:
    config = ClusterConfig(name="t", hostname="example.org", port=2222, username="user")

    assert trusted_host_lookup_name(config) == "[example.org]:2222"


@pytest.mark.parametrize("algorithm", ["ssh-rsa", "ssh-ed25519", "ssh-dss", "ecdsa-sha2-nistp256"])
def test_ssh_connection_registers_a_saved_trusted_host_key_of_each_algorithm(
    monkeypatch: pytest.MonkeyPatch, algorithm: str
) -> None:
    config = _config("key")
    settings = _FakeSettings()
    # The fake server presents an ed25519 key with this material, so the saved
    # ed25519 key matches it and the other algorithms sit alongside it.
    raw = _FakeKey().asbytes()
    save_trusted_host_key(
        settings,
        config,
        algorithm=algorithm,
        key_base64=base64.b64encode(raw).decode("ascii"),
    )
    fake_paramiko = _FakeParamiko(fail_key=False)
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(config, settings=settings)
    connection.connect()

    registered = fake_paramiko._client.get_host_keys().lookup("example.org")
    assert registered is not None
    assert registered[algorithm].asbytes() == raw


def test_ssh_connection_prefers_system_host_keys_over_a_saved_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("key")
    settings = _FakeSettings()
    save_trusted_host_key(
        settings,
        config,
        algorithm="ssh-rsa",
        key_base64=base64.b64encode(b"saved-key").decode("ascii"),
    )
    fake_paramiko = _FakeParamiko(fail_key=False, system_host_keys={"example.org": _FakeKey()})
    _patch_paramiko(monkeypatch, fake_paramiko)

    connection = SSHConnection(config, settings=settings)
    connection.connect()

    # The endpoint is already known, so the saved key is not added alongside it.
    assert set(fake_paramiko._client.get_host_keys().lookup("example.org")) == {"ssh-ed25519"}
    assert connection.is_connected


def test_ssh_connection_skips_an_undecodable_trusted_host_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A corrupt or legacy stored key must not abort the connection with a cryptic
    # decode error; it is skipped, so the connection falls through to system host
    # keys or the trust prompt, which re-saves a correct key.
    config = _config("key")
    settings = _FakeSettings()
    save_trusted_host_key(
        settings,
        config,
        algorithm="ssh-unknown-magic",
        key_base64=base64.b64encode(b"data").decode("ascii"),
    )
    _patch_paramiko(monkeypatch, _FakeParamiko(fail_key=False, system_host_keys={}))

    with pytest.raises(SSHUnknownHostKeyError):
        SSHConnection(config, settings=settings).connect()

    assert "Unsupported trusted host key algorithm: ssh-unknown-magic" in caplog.text


def test_decode_host_key_rejects_hostname_placeholder_blob() -> None:
    # Drives the decoder directly: only real paramiko rejects this blob, and the
    # connection tests run against a fake. The blob reproduces the real corruption,
    # where key_base64 held the base64 of a hostname instead of SSH key wire data.
    paramiko = pytest.importorskip("paramiko")
    record = TrustedHostKeyRecord(
        algorithm="ssh-ed25519",
        key_base64=base64.b64encode(b"bastion").decode("ascii"),
    )
    with pytest.raises(Exception):
        _decode_host_key(paramiko, record)


def test_ssh_connection_connect_returns_early_if_already_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))
    connection.connect()
    call_count = len(fake_paramiko._client.connect_calls)

    connection.connect()

    assert len(fake_paramiko._client.connect_calls) == call_count


def test_ssh_connection_import_failure_raises_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mufinder.services.ssh_client.importlib.import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("missing")),
    )

    with pytest.raises(ConnectionError, match="paramiko is required"):
        SSHConnection._import_paramiko()


def test_ssh_connection_import_returns_the_paramiko_module(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "mufinder.services.ssh_client.importlib.import_module",
        lambda name: sentinel if name == "paramiko" else None,
    )

    assert SSHConnection._import_paramiko() is sentinel


def test_ssh_connection_protocol_banner_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BannerClient(_FakeSSHClient):
        def connect(self, **kwargs) -> None:
            self.connect_calls.append(kwargs)
            raise _FakeParamiko.SSHException("Error reading SSH protocol banner")

    class _BannerParamiko(_FakeParamiko):
        def __init__(self) -> None:
            self._client = _BannerClient(system_host_keys={"example.org": _FakeKey()})

    fake_paramiko = _BannerParamiko()
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))

    with pytest.raises(SSHProtocolBannerError) as exc_info:
        connection.connect()

    assert exc_info.value.hostname == "example.org"
    assert exc_info.value.port == 22
    assert "valid SSH protocol banner" in str(exc_info.value)


def test_ssh_connection_generic_ssh_exception_is_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingClient(_FakeSSHClient):
        def connect(self, **kwargs) -> None:
            self.connect_calls.append(kwargs)
            raise _FakeParamiko.SSHException("kex failed")

    class _FailingParamiko(_FakeParamiko):
        def __init__(self) -> None:
            self._client = _FailingClient(system_host_keys={"example.org": _FakeKey()})

    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: _FailingParamiko()))

    connection = SSHConnection(_config("key"))

    with pytest.raises(SSHTransportError, match="transport negotiation failed"):
        connection.connect()


def test_ssh_connection_banner_text_in_a_non_ssh_error_is_a_banner_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_paramiko(
        monkeypatch,
        _failing_connect_paramiko(ValueError("Error reading SSH protocol banner: timed out")),
    )

    with pytest.raises(SSHProtocolBannerError) as exc_info:
        SSHConnection(_config("key")).connect()

    assert exc_info.value.hostname == "example.org"
    assert "valid SSH protocol banner" in str(exc_info.value)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (OSError("network unreachable"), "SSH connection failed: network unreachable"),
        (OSError(), "SSH connection failed: OSError"),
    ],
    ids=["with-detail", "without-detail"],
)
def test_ssh_connection_other_errors_are_transport_errors(
    monkeypatch: pytest.MonkeyPatch, error: OSError, message: str
) -> None:
    _patch_paramiko(monkeypatch, _failing_connect_paramiko(error))

    with pytest.raises(SSHTransportError) as exc_info:
        SSHConnection(_config("key")).connect()

    assert str(exc_info.value) == message


def test_ssh_connection_sftp_failure_closes_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SFTPFailClient(_FakeSSHClient):
        def open_sftp(self):
            raise OSError("sftp subsystem refused")

    fake_paramiko = _FakeParamiko()
    fake_paramiko._client = _SFTPFailClient(system_host_keys={"example.org": _FakeKey()})
    _patch_paramiko(monkeypatch, fake_paramiko)
    connection = SSHConnection(_config("key"))

    with pytest.raises(ConnectionError, match="Failed to open SFTP session"):
        connection.connect()

    assert fake_paramiko._client.get_transport().is_active() is False
    assert connection.is_connected is False


def test_ssh_connection_key_auth_forwards_passphrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paramiko = _FakeParamiko(
        fail_key=False,
        system_host_keys={"example.org": _FakeKey()},
    )
    monkeypatch.setattr(SSHConnection, "_import_paramiko", staticmethod(lambda: fake_paramiko))

    connection = SSHConnection(_config("key"))
    connection.connect(password="passphrase")

    assert len(fake_paramiko._client.connect_calls) == 1
    assert fake_paramiko._client.connect_calls[0].get("passphrase") == "passphrase"


def test_ssh_connection_resolve_command_path_prefers_configured() -> None:
    connection = SSHConnection(_config("key"))

    assert connection.resolve_command_path("sbatch", "/custom/sbatch") == "/custom/sbatch"
    assert connection.resolve_command_path("   ", "") == ""


def test_ssh_connection_resolve_command_path_probes_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = SSHConnection(_config("key"))

    monkeypatch.setattr(
        connection,
        "exec_command",
        lambda command, timeout=10: (0, "/usr/bin/sbatch\n", "") if "command -v" in command else (1, "", ""),
    )
    assert connection.resolve_command_path("sbatch") == "/usr/bin/sbatch"

    monkeypatch.setattr(connection, "exec_command", lambda command, timeout=10: (1, "", "missing"))
    assert connection.resolve_command_path("squeue") == "squeue"
