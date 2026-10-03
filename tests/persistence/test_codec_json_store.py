"""Tests for persistence codec and store helpers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

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


def test_to_jsonable_fast_path_keeps_enum_and_container_subclass_conversion() -> None:
    class _Level(IntEnum):
        HIGH = 2

    class _Kind(StrEnum):
        MESH = "mesh"

    class _Row(NamedTuple):
        x: float
        y: float

    class _Mapping(OrderedDict):
        pass

    ident = uuid4()
    payload = {
        "floats": [[0.5, 1.5, 2.5], (3.0, 4.0, 5.0)],
        "enums": [_Level.HIGH, _Kind.MESH],
        "row": _Row(1.0, 2.0),
        "mapping": _Mapping([(1, ident)]),
        "nested": {"dc": [_Mini(7)], "none": None, "flag": True},
    }

    encoded = to_jsonable(payload)

    assert encoded["floats"] == [[0.5, 1.5, 2.5], [3.0, 4.0, 5.0]]
    assert type(encoded["floats"][1]) is list
    assert encoded["enums"] == [2, "mesh"]
    assert type(encoded["enums"][0]) is int and type(encoded["enums"][1]) is str
    assert encoded["row"] == [1.0, 2.0]
    assert encoded["mapping"] == {"1": str(ident)}
    assert encoded["nested"] == {"dc": [{"value": 7}], "none": None, "flag": True}


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


def test_project_store_read_json_reports_dataless_cloud_placeholder(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "store.mufinder"
    store = ProjectStore(root)
    store.initialize_layout()

    path = store.project_manifest_path
    path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(ProjectStore, "_is_dataless_file", staticmethod(lambda _path: True))

    try:
        store.read_json(path)
    except TimeoutError as exc:
        assert "cloud-only placeholder" in str(exc)
        assert str(path) in str(exc)
    else:
        raise AssertionError("Expected read_json to raise for dataless files")


def test_project_store_read_json_wraps_timeouts_with_path(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "store.mufinder"
    store = ProjectStore(root)
    store.initialize_layout()

    path = store.project_manifest_path
    path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(ProjectStore, "_is_dataless_file", staticmethod(lambda _path: False))
    monkeypatch.setattr(Path, "read_text", lambda self, encoding="utf-8": (_ for _ in ()).throw(TimeoutError("boom")))

    try:
        store.read_json(path)
    except TimeoutError as exc:
        assert str(path) in str(exc)
    else:
        raise AssertionError("Expected read_json to wrap TimeoutError")
