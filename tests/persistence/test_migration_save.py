"""Durable redo plans survive interrupted writes and reject invalid journals."""

import copy
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.models import PROJECT_SCHEMA_VERSION
from mufinder.persistence.json_store import ProjectStore
from mufinder.persistence.migration_save import (
    JOURNAL_NAME,
    JOURNAL_VERSION,
    begin_migration_save,
    recover_migration_save,
)


@pytest.fixture
def migration_plan(tmp_path: Path):
    store = ProjectStore(tmp_path / "project.mufinder")
    store.initialize_layout()
    manifest = {"id": str(uuid4()), "schema_version": 4}
    store.write_json(store.project_manifest_path, manifest)
    entity_id = str(uuid4())
    plan = {
        "journal_version": JOURNAL_VERSION,
        "manifest": {**manifest, "schema_version": PROJECT_SCHEMA_VERSION},
        "entities": {"results": {entity_id: {"id": entity_id, "data": {"converted": 1}}}},
        "keep_ids": {"results": [entity_id]},
        "state": {"migrated": True},
    }
    return store, plan


def test_real_short_journal_write_leaves_original_metadata_untouched(migration_plan) -> None:
    resource = pytest.importorskip("resource")
    store, plan = migration_plan
    before = {
        p.relative_to(store.project_root): p.read_bytes()
        for p in store.project_root.rglob("*")
        if p.is_file()
    }
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (16, hard))
    try:
        with pytest.raises(OSError):
            begin_migration_save(store, plan)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
    assert {
        p.relative_to(store.project_root): p.read_bytes()
        for p in store.project_root.rglob("*")
        if p.is_file()
    } == before
    assert not recover_migration_save(store)


def test_completed_replay_prunes_and_can_be_repeated(migration_plan) -> None:
    store, plan = migration_plan
    stale_id = str(uuid4())
    stale = store.write_entity("results", stale_id, {"id": stale_id})
    unrelated = store.project_root / "files" / "output.castep"
    unrelated.write_text("keep")
    for _ in range(2):
        begin_migration_save(store, plan)
        assert recover_migration_save(store)
        assert not recover_migration_save(store)
        assert store.load_entities("results") == list(plan["entities"]["results"].values())
        assert store.read_json(store.state_path) == plan["state"]
        assert store.read_json(store.project_manifest_path) == plan["manifest"]
    assert not stale.exists()
    assert unrelated.read_text() == "keep"


def test_pending_journal_cannot_be_overwritten(migration_plan) -> None:
    store, plan = migration_plan
    begin_migration_save(store, plan)
    earlier = (store.project_root / JOURNAL_NAME).read_bytes()
    with pytest.raises(ValueError, match="Recover the pending"):
        begin_migration_save(store, plan)
    assert (store.project_root / JOURNAL_NAME).read_bytes() == earlier


def test_replay_syncs_new_entity_directories_before_removing_the_journal(
    migration_plan, monkeypatch
) -> None:
    store, plan = migration_plan
    (store.entities_root / "results").rmdir()
    begin_migration_save(store, plan)
    synced = []
    sync_folder, unlink = ProjectStore._sync_folder, Path.unlink

    def sync(folder):
        synced.append(folder)
        sync_folder(folder)

    def remove(path, *args, **kwargs):
        if path.name == JOURNAL_NAME:
            assert store.entities_root / "results" in synced
            assert store.entities_root in synced
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(ProjectStore, "_sync_folder", staticmethod(sync))
    monkeypatch.setattr(Path, "unlink", remove)
    assert recover_migration_save(store)


@pytest.mark.parametrize(
    "damage",
    [
        "format",
        "missing_state",
        "manifest",
        "schema",
        "type",
        "ids",
        "payload",
        "state",
        "project",
        "json",
    ],
)
def test_invalid_journal_refuses_recovery_before_changing_metadata(migration_plan, damage) -> None:
    store, original = migration_plan
    plan = copy.deepcopy(original)
    entity_id = plan["keep_ids"]["results"][0]
    if damage == "format":
        plan["journal_version"] = 99
    elif damage == "missing_state":
        plan.pop("state")
    elif damage == "manifest":
        plan["manifest"] = []
    elif damage == "schema":
        plan["manifest"]["schema_version"] = PROJECT_SCHEMA_VERSION + 1
    elif damage == "type":
        plan["keep_ids"]["../files"] = []
        plan["entities"]["../files"] = {}
    elif damage == "ids":
        plan["keep_ids"]["results"] = ["../../project"]
    elif damage == "payload":
        plan["entities"]["results"][entity_id]["id"] = str(uuid4())
    elif damage == "state":
        plan["state"] = []
    elif damage == "project":
        plan["manifest"]["id"] = str(uuid4())
    path = store.project_root / JOURNAL_NAME
    store.write_json(path, plan)
    if damage == "json":
        path.write_text("{")
    before = store.project_manifest_path.read_bytes()
    journal = path.read_bytes()
    with pytest.raises(ValueError, match="Cannot recover migration save"):
        recover_migration_save(store)
    assert store.project_manifest_path.read_bytes() == before
    assert path.read_bytes() == journal
    assert store.load_entities("results") == []
    assert not store.state_path.exists()
