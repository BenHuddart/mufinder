"""Tests for persistence codec and store helpers."""

from __future__ import annotations

import dataclasses
import inspect
import json
import types
import typing
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, IntEnum, StrEnum
from pathlib import Path
from typing import Any, NamedTuple
from uuid import UUID, uuid4

import pytest

from mufinder.domain import cluster, models

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


def _sample_value(annotation: Any) -> Any:
    """A non-default value of ``annotation``, so every field reaches the encoder."""

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return _sample_value(next(arg for arg in args if arg is not type(None)))
    if origin is typing.Literal:
        return args[0]
    if origin is list or annotation is list:
        return [_sample_value(args[0]) if args else 1]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return (_sample_value(args[0]),)
        return tuple(_sample_value(arg) for arg in args)
    if origin is dict or annotation is dict:
        return {"key": _sample_value(args[1]) if args else 1}
    if annotation is Any or annotation is object:
        return {"nested": [1.5, "text", None, (True, 2)]}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return next(iter(annotation))
    if dataclasses.is_dataclass(annotation):
        return _sample_instance(annotation)
    samples: dict[Any, Any] = {
        bool: True,
        int: 2,
        float: 0.5,
        str: "text",
        UUID: uuid4(),
        datetime: datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        Path: Path("files/run"),
    }
    return samples[annotation]


def _sample_instance(cls: type) -> Any:
    hints = typing.get_type_hints(cls)
    return cls(
        **{
            field.name: _sample_value(hints[field.name])
            for field in dataclasses.fields(cls)
            if field.init
        }
    )


_DOMAIN_DATACLASSES = [
    obj
    for module in (models, cluster)
    for obj in vars(module).values()
    if inspect.isclass(obj) and dataclasses.is_dataclass(obj) and obj.__module__ == module.__name__
]


@pytest.mark.parametrize("cls", _DOMAIN_DATACLASSES, ids=lambda cls: cls.__name__)
def test_to_jsonable_walks_domain_dataclasses_into_the_same_json_as_asdict(cls: type) -> None:
    # The encoder reads dataclass fields directly rather than through
    # ``dataclasses.asdict`` (which deep-copies the tree first); the JSON written
    # to disk must not change for any persisted domain type.
    instance = _sample_instance(cls)

    encoded = json.dumps(to_jsonable(instance), sort_keys=True)

    assert encoded == json.dumps(to_jsonable(dataclasses.asdict(instance)), sort_keys=True)


def test_domain_dataclass_list_covers_the_persisted_entity_types() -> None:
    names = {cls.__name__ for cls in _DOMAIN_DATACLASSES}

    assert {
        "Structure",
        "MuonSite",
        "CalculationDefinition",
        "CalculationRun",
        "Result",
        "ComparisonTable",
        "Job",
        "ProvenanceRecord",
        "Project",
    } <= names


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
    store.provenance_path.write_text(
        store.provenance_path.read_text(encoding="utf-8") + "{bad\n", encoding="utf-8"
    )
    events = store.load_provenance()
    assert [row["event"] for row in events] == [1, 2]


def test_project_store_read_json_reports_dataless_cloud_placeholder(
    tmp_path: Path, monkeypatch
) -> None:
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
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding="utf-8": (_ for _ in ()).throw(TimeoutError("boom")),
    )

    try:
        store.read_json(path)
    except TimeoutError as exc:
        assert str(path) in str(exc)
    else:
        raise AssertionError("Expected read_json to wrap TimeoutError")


def test_prune_entities_removes_stale_files_and_reports_the_kept_ones_on_disk(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "prune.mufinder")
    store.initialize_layout()
    store.write_entity("jobs", "kept", {"id": "kept"})
    store.write_entity("jobs", "stale", {"id": "stale"})
    jobs_dir = store.entities_root / "jobs"
    (jobs_dir / "._kept.json").write_text("resource fork", encoding="utf-8")
    (jobs_dir / "notes.txt").write_text("not an entity", encoding="utf-8")

    present = store.prune_entities("jobs", {"kept", "missing"})

    assert present == {"kept"}
    assert sorted(path.name for path in jobs_dir.iterdir()) == ["kept.json", "notes.txt"]
    assert store.load_entities("jobs") == [{"id": "kept"}]
    assert store.prune_entities("absent_type", {"kept"}) == set()
