"""In-memory stand-ins for ``SSHConnection`` shared by the cluster tests."""

from __future__ import annotations

from pathlib import Path


class FakeSSH:
    """An SSH connection whose remote filesystem is the ``remote_files`` dict.

    ``commands`` records every mutating call and download in order. A download
    of a path that is not in ``remote_files`` writes ``b"ok"``. There is no
    ``mtime`` attribute, standing in for an SFTP server or connection object
    that cannot report one.
    """

    def __init__(self) -> None:
        self.is_connected = True
        self.commands: list[tuple[str, object]] = []
        self.remote_files: dict[str, bytes] = {}

    def mkdir(self, path: str, exist_ok: bool = True) -> None:
        self.commands.append(("mkdir", (path, exist_ok)))

    def put(self, local_path: Path, remote_path: str) -> None:
        self.commands.append(("put", remote_path))
        self.remote_files[remote_path] = Path(local_path).read_bytes()

    def chmod(self, remote_path: str, mode: int) -> None:
        self.commands.append(("chmod", (remote_path, mode)))

    def get(self, remote_path: str, local_path: Path) -> None:
        self.commands.append(("get", remote_path))
        local_path.write_bytes(self.remote_files.get(remote_path, b"ok"))

    def exists(self, remote_path: str) -> bool:
        return remote_path in self.remote_files

    def listdir(self, remote_path: str) -> list[str]:
        prefix = remote_path.rstrip("/") + "/"
        entries: list[str] = []
        for path in self.remote_files:
            if path.startswith(prefix):
                entries.append(path[len(prefix):].split("/", 1)[0])
        return sorted(set(entries))

    def disconnect(self) -> None:
        self.is_connected = False

    def connect(self, password: str | None = None) -> None:
        _ = password
        self.is_connected = True


class TimestampedFakeSSH(FakeSSH):
    """A fake whose remote paths can carry explicit remote mtimes.

    Only paths seeded in ``mtimes`` report one; every other path returns
    ``None``, so a test that does not opt in keeps the fail-open behaviour of
    the plain :class:`FakeSSH`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mtimes: dict[str, float] = {}

    def mtime(self, remote_path: str) -> float | None:
        return self.mtimes.get(remote_path)


class OutputUnavailableSSH(FakeSSH):
    """A fake whose downloads fail, simulating unavailable remote output."""

    def get(self, remote_path: str, local_path: Path) -> None:
        raise OSError(f"SFTP download failed: {remote_path}")
