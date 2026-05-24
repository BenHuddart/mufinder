"""Low-level filesystem-backed project storage."""

from __future__ import annotations

import json
from pathlib import Path
import stat
from typing import Any


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

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """Write JSON payload with stable formatting."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

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

    def prune_entities(self, entity_type: str, keep_ids: set[str]) -> None:
        """Remove stale entity JSON files that are no longer present in memory."""
        entity_dir = self.entities_root / entity_type
        if not entity_dir.exists():
            return

        keep_filenames = {f"{entity_id}.json" for entity_id in keep_ids}
        for entity_path in entity_dir.glob("*.json"):
            if entity_path.name not in keep_filenames:
                entity_path.unlink()

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
