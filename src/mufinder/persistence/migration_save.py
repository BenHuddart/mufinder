"""Replayable metadata writes for the first save after a schema migration."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from mufinder.domain.models import PROJECT_SCHEMA_VERSION
from mufinder.persistence.json_store import ProjectStore


JOURNAL_NAME = ".migration-save.json"
JOURNAL_VERSION = 1


def _validate(plan: Any) -> None:
    """Validate the whole plan before any writes or pruning can take place."""
    if (
        not isinstance(plan, dict)
        or set(plan) != {"journal_version", "manifest", "entities", "keep_ids", "state"}
        or type(plan.get("journal_version")) is not int
        or plan["journal_version"] != JOURNAL_VERSION
    ):
        raise ValueError("unsupported migration journal format")
    manifest = plan.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("missing target manifest")
    version = manifest.get("schema_version")
    if type(version) is not int or not 1 <= version <= PROJECT_SCHEMA_VERSION:
        raise ValueError("unsupported target schema version")
    UUID(manifest["id"])
    writes, keep = plan.get("entities"), plan.get("keep_ids")
    if not isinstance(writes, dict) or not isinstance(keep, dict) or writes.keys() != keep.keys():
        raise ValueError("entity writes and live IDs must name the same entity types")
    for entity_type, ids in keep.items():
        if entity_type not in ProjectStore.ENTITY_DIRS or not isinstance(ids, list):
            raise ValueError("unknown entity type or invalid live IDs")
        for entity_id in ids:
            if not isinstance(entity_id, str) or str(UUID(entity_id)) != entity_id:
                raise ValueError("entity IDs must be canonical UUIDs")
        payloads = writes[entity_type]
        if not isinstance(payloads, dict) or not payloads.keys() <= set(ids):
            raise ValueError("entity writes must be included in live IDs")
        for entity_id, payload in payloads.items():
            if not isinstance(payload, dict) or payload.get("id") != entity_id:
                raise ValueError("entity payload does not match its ID")
    if plan.get("state") is not None and not isinstance(plan["state"], dict):
        raise ValueError("invalid UI state")


def begin_migration_save(store: ProjectStore, plan: dict[str, Any]) -> None:
    """Commit a complete redo plan before modifying any canonical metadata."""
    _validate(plan)
    if (store.project_root / JOURNAL_NAME).exists():
        raise ValueError("Recover the pending migration save before starting another")
    store.write_json(store.project_root / JOURNAL_NAME, plan)


def save_migration_snapshot(store: ProjectStore, plan: dict[str, Any]) -> None:
    """Commit and apply a validated plan without reading its payloads back."""
    begin_migration_save(store, plan)
    _replay(store, plan)


def recover_migration_save(store: ProjectStore) -> bool:
    """Finish a pending snapshot without running migrations; retain it on failure.

    Returns whether recovery took place. A caller with cached file hashes must
    invalidate them after recovery, before making further save decisions.
    """
    path = store.project_root / JOURNAL_NAME
    if not path.exists():
        return False
    try:
        plan = store.read_json(path)
        _validate(plan)
        manifest = store.read_json(store.project_manifest_path)
        if manifest.get("id") != plan["manifest"]["id"]:
            raise ValueError("journal belongs to another project")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"Cannot recover migration save from {path}: {exc}. "
            "Restore a complete project backup; keep the journal for recovery."
        ) from exc
    _replay(store, plan)
    return True


def _replay(store: ProjectStore, plan: dict[str, Any]) -> None:
    """Apply an already-validated snapshot; leave its journal on failure."""
    try:
        for entity_type, payloads in plan["entities"].items():
            for entity_id, payload in payloads.items():
                store.write_entity(entity_type, entity_id, payload)
            store.prune_entities(entity_type, set(plan["keep_ids"][entity_type]))
            store._sync_folder(store.entities_root / entity_type)
        if plan["state"] is not None:
            store.write_json(store.state_path, plan["state"])
        # A migration can create an entity directory (for example results
        # extracted from UI state). Persist that directory's parent entry too.
        store._sync_folder(store.entities_root)
        store.write_json(store.project_manifest_path, plan["manifest"])
        (store.project_root / JOURNAL_NAME).unlink()
        store._sync_folder(store.project_root)
    except OSError as exc:
        raise OSError(
            f"Cannot finish migration save in {store.project_root}: {exc}. "
            "Resolve the filesystem error and reopen or retry saving; keep the migration journal."
        ) from exc
