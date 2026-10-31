"""Low-level filesystem-backed project storage."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import stat
from pathlib import Path
from typing import Any
from uuid import uuid4


_DARWIN_SF_DATALESS = 0x40000000


class ProjectStore:
    """Reads and writes MuFinder project metadata and entity files."""

    ENTITY_DIRS = (
        "structures",
        "sites",
        "clusters",
        "calculations",
        "runs",
        "jobs",
        "results",
        "comparison_tables",
        "generation_runs",
    )

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    @property
    def entities_root(self) -> Path:
        return self.project_root / "entities"

    @property
    def project_manifest_path(self) -> Path:
        return self.project_root / "project.json"

    @property
    def state_path(self) -> Path:
        return self.project_root / "state.json"

    @property
    def provenance_path(self) -> Path:
        return self.project_root / "provenance.jsonl"

    def initialize_layout(self) -> None:
        """Create canonical project folder layout if missing."""
        self.project_root.mkdir(parents=True, exist_ok=True)
        (self.project_root / "imports").mkdir(exist_ok=True)
        (self.project_root / "files").mkdir(exist_ok=True)
        (self.project_root / "exports").mkdir(exist_ok=True)
        self.entities_root.mkdir(exist_ok=True)

        for entity_dir in self.ENTITY_DIRS:
            (self.entities_root / entity_dir).mkdir(exist_ok=True)

    @staticmethod
    def canonical_json(payload: dict[str, Any]) -> str:
        """Return the canonical on-disk serialization for a JSON payload.

        This is the single source of truth for how MuFinder serializes JSON to
        disk. :meth:`write_json` writes exactly this string, and callers that
        need a content hash of what *would* be written (e.g. the dirty-tracking
        cache in ``ProjectService``) hash exactly this string, so the two can
        never drift.
        """
        return json.dumps(payload, indent=2, sort_keys=True)

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """Write JSON payload with stable formatting, atomically.

        The document goes to a sibling temporary file, is synced to disk and is
        moved into place, and the folder is synced to record the move, so a full
        disk or a crash never leaves a truncated file where a good one was. The
        temporary name does not end in ``.json``, so one
        left by a killed process is never read or pruned as an entity.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # The real file, so a symbolic link stays a link rather than being replaced.
        destination = Path(os.path.realpath(path))
        existing_mode = self._writable_file_mode(destination)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(self.canonical_json(payload))
                stream.flush()
                os.fsync(stream.fileno())
            if existing_mode is not None:
                temporary.chmod(existing_mode)
            os.replace(temporary, destination)
        except BaseException:
            # The original failure is the one to report, not a failed clean-up.
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise
        self._sync_folder(destination.parent)

    @staticmethod
    def _sync_folder(folder: Path) -> None:
        """Record a completed move on disk, where the platform allows it.

        The file is already complete and in place, so a folder that cannot be
        synced (Windows cannot open one, some mounts refuse) does not fail the write.
        """
        with contextlib.suppress(OSError):
            descriptor = os.open(folder, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    @staticmethod
    def _writable_file_mode(path: Path) -> int | None:
        """Return the permission bits of ``path``, or ``None`` if there is no such file.

        Renaming over a file needs only a writable folder, so a read-only file is
        refused here to keep it from being replaced.
        """
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except FileNotFoundError:
            return None
        if not os.access(path, os.W_OK):
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
        return mode

    @staticmethod
    def _is_dataless_file(path: Path) -> bool:
        try:
            flags = path.stat().st_flags  # type: ignore[attr-defined]
            return bool(flags & _DARWIN_SF_DATALESS)
        except (OSError, AttributeError):
            return False

    def read_json(self, path: Path) -> dict[str, Any]:
        """Read JSON payload from disk."""
        if self._is_dataless_file(path):
            raise TimeoutError(
                "Project file is stored as a cloud-only placeholder and is not available locally: "
                f"{path}"
            )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except TimeoutError as exc:
            raise TimeoutError(f"Timed out reading project file: {path}") from exc

    def write_entity(self, entity_type: str, entity_id: str, payload: dict[str, Any]) -> Path:
        """Persist an entity in its type-specific directory."""
        out_path = self.entities_root / entity_type / f"{entity_id}.json"
        self.write_json(out_path, payload)
        return out_path

    def prune_entities(self, entity_type: str, keep_ids: set[str]) -> set[str]:
        """Remove stale entity JSON files that are no longer present in memory.

        Returns the ids in ``keep_ids`` whose files are on disk, read from the
        same directory listing, so a caller need not stat each file again.
        """
        entity_dir = self.entities_root / entity_type
        if not entity_dir.exists():
            return set()

        present: set[str] = set()
        # ``os.scandir`` names the files without building a ``Path`` per entry,
        # which this listing of every entity file on each autosave would pay.
        with os.scandir(entity_dir) as entries:
            for entry in entries:
                # Every ``*.json`` name counts, dotfiles included (as ``Path.glob``
                # and ``load_entities`` match them): a stray macOS ``._*.json``
                # resource fork is pruned rather than later parsed as an entity.
                if not entry.name.endswith(".json"):
                    continue
                stem = entry.name[: -len(".json")]
                if stem in keep_ids:
                    present.add(stem)
                else:
                    os.unlink(entry.path)
        return present

    def load_entities(self, entity_type: str) -> list[dict[str, Any]]:
        """Load all entities of a type."""
        entity_dir = self.entities_root / entity_type
        if not entity_dir.exists():
            return []

        entities: list[dict[str, Any]] = []
        for entity_path in sorted(entity_dir.glob("*.json")):
            entities.append(self.read_json(entity_path))
        return entities

    def append_provenance(self, payload: dict[str, Any]) -> None:
        """Append one provenance record in JSONL format."""
        self.provenance_path.parent.mkdir(parents=True, exist_ok=True)
        with self.provenance_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

    def load_provenance(self) -> list[dict[str, Any]]:
        """Load all parseable provenance records from JSONL."""
        if not self.provenance_path.exists():
            return []

        rows: list[dict[str, Any]] = []
        for line in self.provenance_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows
