"""Tests for persistence codec and store helpers."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import stat
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

from mufinder.persistence.codec import (
    float_triplet,
    float_triplet_or_none,
    int_or_none,
    parse_datetime,
    parse_uuid,
    to_jsonable,
    uuid_or_none,
)
from mufinder.persistence import json_store
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


def _project_file_with_content(store: ProjectStore) -> tuple[Path, bytes]:
    store.initialize_layout()
    path = store.project_manifest_path
    store.write_json(path, {"id": "earlier", "name": "Earlier save"})
    return path, path.read_bytes()


def _leftover_temporary_files(path: Path) -> list[str]:
    return sorted(entry.name for entry in path.parent.iterdir() if entry.name != path.name)


def test_write_failing_part_way_keeps_the_earlier_file(tmp_path: Path) -> None:
    resource = pytest.importorskip("resource")
    store = ProjectStore(tmp_path / "store.mufinder")
    path, earlier = _project_file_with_content(store)
    before = _leftover_temporary_files(path)

    # A real short write: the kernel takes the first bytes, then refuses the rest,
    # the way a disk filling up part-way through the document does.
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (16, hard))
    try:
        with pytest.raises(OSError):
            store.write_json(path, {"id": "later", "padding": "x" * 100_000})
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))

    assert path.read_bytes() == earlier
    assert _leftover_temporary_files(path) == before


@pytest.mark.parametrize("failing_call", ["fsync", "replace"])
def test_failed_sync_or_replace_keeps_the_file_and_removes_the_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_call: str
) -> None:
    store = ProjectStore(tmp_path / "store.mufinder")
    path, earlier = _project_file_with_content(store)
    before = _leftover_temporary_files(path)

    def _fail(*_args, **_kwargs) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(json_store.os, failing_call, _fail)

    with pytest.raises(OSError, match="No space left"):
        store.write_json(path, {"id": "later"})

    assert path.read_bytes() == earlier
    assert _leftover_temporary_files(path) == before


def test_temporary_file_is_synced_before_it_replaces_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ProjectStore(tmp_path / "store.mufinder")
    path, _earlier = _project_file_with_content(store)
    events: list[str] = []
    real_fsync = json_store.os.fsync
    real_replace = json_store.os.replace

    def _fsync(descriptor: int) -> None:
        events.append("fsync")
        real_fsync(descriptor)

    def _replace(source, destination) -> None:
        assert Path(source).parent == path.parent
        assert Path(destination) == path
        events.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(json_store.os, "fsync", _fsync)
    monkeypatch.setattr(json_store.os, "replace", _replace)

    store.write_json(path, {"id": "later"})

    assert events == ["fsync", "replace"]
    assert path.read_text(encoding="utf-8") == ProjectStore.canonical_json({"id": "later"})


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_write_json_keeps_the_mode_of_an_existing_file_and_honours_umask_for_a_new_one(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "store.mufinder")
    path, _earlier = _project_file_with_content(store)
    path.chmod(0o640)

    store.write_json(path, {"id": "later"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o640

    previous_umask = os.umask(0o027)
    try:
        fresh = path.with_name("fresh.json")
        store.write_json(fresh, {"id": "fresh"})
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
def test_write_json_refuses_a_read_only_file_or_folder(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "store.mufinder")
    path, earlier = _project_file_with_content(store)

    path.chmod(0o444)
    with pytest.raises(PermissionError):
        store.write_json(path, {"id": "later"})
    assert path.read_bytes() == earlier
    path.chmod(0o644)

    path.parent.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            store.write_json(path, {"id": "later"})
    finally:
        path.parent.chmod(0o755)
    assert path.read_bytes() == earlier


@pytest.mark.skipif(os.name != "posix", reason="symbolic links need privileges elsewhere")
def test_write_json_writes_through_a_symbolic_link(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "store.mufinder")
    store.initialize_layout()
    target = tmp_path / "elsewhere" / "project.json"
    target.parent.mkdir()
    target.write_text("{}", encoding="utf-8")
    store.project_manifest_path.symlink_to(target)

    store.write_json(store.project_manifest_path, {"id": "later"})

    assert store.project_manifest_path.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {"id": "later"}


def test_temporary_file_left_by_a_crash_is_not_read_as_an_entity(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "store.mufinder")
    store.initialize_layout()
    store.write_entity("structures", "s1", {"id": "s1"})
    # What a process killed between creating and replacing the file leaves behind.
    leftover = store.entities_root / "structures" / ".s1.json.0123abcd.tmp"
    leftover.write_text('{"id": "s1", "cut', encoding="utf-8")

    assert store.load_entities("structures") == [{"id": "s1"}]
    assert store.prune_entities("structures", {"s1"}) == {"s1"}


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


def test_lenient_readers_give_none_for_values_that_do_not_convert() -> None:
    identifier = uuid4()

    assert uuid_or_none(str(identifier)) == identifier
    assert uuid_or_none(identifier) == identifier
    for value in (None, "", "not-a-uuid", 0, False):
        assert uuid_or_none(value) is None
    assert int_or_none("3") == 3
    assert int_or_none(2.7) == 2
    for value in (None, "three", [1]):
        assert int_or_none(value) is None


def test_float_triplet_reads_three_numbers_and_raises_otherwise() -> None:
    assert float_triplet([1, "2", 3.5]) == (1.0, 2.0, 3.5)
    assert float_triplet((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="three-vector"):
        float_triplet([1.0, 2.0])
    with pytest.raises(ValueError):
        float_triplet(["x", 0.0, 0.0])
    with pytest.raises(TypeError):
        float_triplet(None)


def test_float_triplet_or_none_reads_only_a_list_or_tuple_of_three_numbers() -> None:
    assert float_triplet_or_none([1, "2", 3.5]) == (1.0, 2.0, 3.5)
    assert float_triplet_or_none((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    for value in (None, [1.0, 2.0], ["x", 0.0, 0.0], "abc", {0: 1.0, 1: 2.0, 2: 3.0}):
        assert float_triplet_or_none(value) is None
