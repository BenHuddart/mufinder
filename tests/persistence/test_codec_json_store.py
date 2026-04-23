"""Tests for persistence codec and store helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mufinder.persistence.codec import parse_datetime, parse_uuid, to_jsonable
from mufinder.persistence.json_store import ProjectStore


@dataclass
class _Mini:
    value: int


def test_codec_helpers_cover_branching() -> None:
    now = datetime.now(UTC)
    payload = {
        "dt": now,
        "tuple": (1, 2),
        "dc": _Mini(3),
        "path": Path("abc"),
    }
    encoded = to_jsonable(payload)

    assert encoded["tuple"] == [1, 2]
    assert encoded["dc"]["value"] == 3
    assert parse_datetime(encoded["dt"]) == now
    assert parse_datetime(None) is None
    assert parse_uuid(None) is None


def test_project_store_entity_and_provenance_io(tmp_path: Path) -> None:
    root = tmp_path / "store.mufinder"
    store = ProjectStore(root)
    store.initialize_layout()

    assert not (root / ".autosave").exists()

    store.write_json(store.project_manifest_path, {"id": "abc"})
    assert store.read_json(store.project_manifest_path)["id"] == "abc"

    store.write_entity("structures", "s1", {"id": "s1", "kind": "structure"})
    items = store.load_entities("structures")
    assert items[0]["id"] == "s1"

    # Unknown directories should return an empty list.
    assert store.load_entities("missing_type") == []

    store.append_provenance({"event": 1})
    store.append_provenance({"event": 2})
    # Corrupt one line and ensure loader skips it.
    store.provenance_path.write_text(store.provenance_path.read_text(encoding="utf-8") + "{bad\n", encoding="utf-8")
    events = store.load_provenance()
    assert [row["event"] for row in events] == [1, 2]
