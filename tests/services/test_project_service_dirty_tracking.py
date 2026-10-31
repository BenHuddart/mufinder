"""Tests for content-hash dirty tracking in ProjectService.save_project.

``save_project`` skips rewriting entity JSON files whose canonical serialized
payload is unchanged since the last save/load, guarding each skip with the
entity directory's listing so externally-deleted files are always regenerated. The
cache is per-project-root and reset on load/create. See ProjectService's class
docstring and ``docs/architecture.rst`` (persistence).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Result
from mufinder.persistence import json_store
from mufinder.persistence.json_store import ProjectStore
from mufinder.persistence.codec import to_jsonable
from mufinder.services.project_service import ProjectService
from tests.support.structures import silicon_oxide_host


def _populated_document(service: ProjectService, root: Path):
    """Create a project with a structure, a job, and a result already saved."""
    doc = service.create_project(root, "Dirty")
    service.add_structure(doc, silicon_oxide_host())
    job = service.add_job(
        doc,
        label="Relax",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    result = Result(
        result_type="relaxation",
        data={"total_energy": -1.5, "max_force": 0.01, "warnings": ["w"]},
        output_files=["out.castep"],
    )
    doc.results[result.id] = result
    service.save_project(doc)
    return doc, job, result


def _entity_dir(root: Path, entity_type: str) -> Path:
    return root / "entities" / entity_type


def _all_entity_mtimes(root: Path) -> dict[Path, float]:
    entities_root = root / "entities"
    return {path: path.stat().st_mtime_ns for path in entities_root.rglob("*.json")}


def test_second_save_without_mutation_writes_no_entity_files(tmp_path: Path, monkeypatch) -> None:
    service = ProjectService()
    root = tmp_path / "noop.mufinder"
    doc, _job, _result = _populated_document(service, root)

    written: list[tuple[str, str]] = []
    original = ProjectStore.write_entity

    def _tracking_write_entity(self, entity_type, entity_id, payload):
        written.append((entity_type, entity_id))
        return original(self, entity_type, entity_id, payload)

    monkeypatch.setattr(ProjectStore, "write_entity", _tracking_write_entity)

    service.save_project(doc)

    assert written == [], f"expected no entity writes, got {written}"


def test_second_save_leaves_entity_file_mtimes_untouched(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "mtimes.mufinder"
    doc, _job, _result = _populated_document(service, root)

    before = _all_entity_mtimes(root)
    assert before  # sanity: files exist

    service.save_project(doc)

    after = _all_entity_mtimes(root)
    assert after == before


def test_mutating_one_job_rewrites_only_that_job(tmp_path: Path, monkeypatch) -> None:
    service = ProjectService()
    root = tmp_path / "onejob.mufinder"
    doc, job, _result = _populated_document(service, root)

    job.status = JobStatus.COMPLETED  # mutate a single entity

    written: list[tuple[str, str]] = []
    original = ProjectStore.write_entity

    def _tracking_write_entity(self, entity_type, entity_id, payload):
        written.append((entity_type, entity_id))
        return original(self, entity_type, entity_id, payload)

    monkeypatch.setattr(ProjectStore, "write_entity", _tracking_write_entity)

    service.save_project(doc)

    assert written == [("jobs", str(job.id))], f"unexpected writes: {written}"


def test_externally_deleted_entity_file_is_rewritten(tmp_path: Path, monkeypatch) -> None:
    service = ProjectService()
    root = tmp_path / "deleted.mufinder"
    doc, _job, result = _populated_document(service, root)

    result_path = _entity_dir(root, "results") / f"{result.id}.json"
    assert result_path.exists()
    result_path.unlink()  # external deletion, no in-memory mutation

    written: list[tuple[str, str]] = []
    original = ProjectStore.write_entity

    def _tracking_write_entity(self, entity_type, entity_id, payload):
        written.append((entity_type, entity_id))
        return original(self, entity_type, entity_id, payload)

    monkeypatch.setattr(ProjectStore, "write_entity", _tracking_write_entity)

    service.save_project(doc)

    assert ("results", str(result.id)) in written
    assert result_path.exists()


def test_unchanged_save_takes_file_existence_from_the_directory_listing(
    tmp_path: Path, monkeypatch
) -> None:
    service = ProjectService()
    root = tmp_path / "listing.mufinder"
    doc, _job, _result = _populated_document(service, root)

    entities_root = root / "entities"
    stat_checks: list[Path] = []
    original_exists = Path.exists

    def _tracking_exists(self: Path, *args, **kwargs) -> bool:
        if self.suffix == ".json" and entities_root in self.parents:
            stat_checks.append(self)
        return original_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", _tracking_exists)

    service.save_project(doc)

    assert stat_checks == []


def test_switching_projects_writes_everything_for_new_project(tmp_path: Path, monkeypatch) -> None:
    """A save of project B on a service warmed only by project A writes all of B.

    The cache is keyed per project root, so project A's populated cache must
    not let any of project B's entities be skipped on B's first save.
    """
    service = ProjectService()
    root_a = tmp_path / "a.mufinder"
    root_b = tmp_path / "b.mufinder"

    # Warm the service's cache with project A only.
    _populated_document(service, root_a)

    # Build project B on disk with a *separate* service so B's entities are
    # never seeded into the first service's cache. B's payloads are otherwise
    # unrelated to A's, so a save of B on the A-warmed service must write every
    # B entity: a cache hit could only occur under the root_b key, which is
    # empty here.
    builder = ProjectService()
    doc_b, job_b, result_b = _populated_document(builder, root_b)

    written: list[tuple[str, str]] = []
    original = ProjectStore.write_entity

    def _tracking_write_entity(self, entity_type, entity_id, payload):
        written.append((entity_type, entity_id))
        return original(self, entity_type, entity_id, payload)

    monkeypatch.setattr(ProjectStore, "write_entity", _tracking_write_entity)

    service.save_project(doc_b)

    # The A-warmed service has no cache entries for root_b, so every B entity
    # (structure, job, result) is written.
    assert ("structures", str(next(iter(doc_b.structures)))) in written
    assert ("jobs", str(job_b.id)) in written
    assert ("results", str(result_b.id)) in written


def test_reload_then_first_save_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    """Seeding the cache on load makes the first post-open save a no-op."""
    service = ProjectService()
    root = tmp_path / "seed.mufinder"
    doc, _job, _result = _populated_document(service, root)

    reloaded = service.load_project(root)

    written: list[tuple[str, str]] = []
    original = ProjectStore.write_entity

    def _tracking_write_entity(self, entity_type, entity_id, payload):
        written.append((entity_type, entity_id))
        return original(self, entity_type, entity_id, payload)

    monkeypatch.setattr(ProjectStore, "write_entity", _tracking_write_entity)

    service.save_project(reloaded)

    assert written == [], f"first save after reload should write nothing, got {written}"


def test_tojsonable_roundtrips_to_exact_loaded_payload(tmp_path: Path) -> None:
    """Cache seeding relies on ``to_jsonable(entity)`` reproducing the on-disk payload.

    If this ever regresses, the seeded hash would not match the file the store
    would write, and the first post-load save would either rewrite everything
    (harmless) or skip a stale file. This asserts exactness for representative
    entities: a structure with nested properties, a job, and a result carrying
    a data payload.
    """
    service = ProjectService()
    root = tmp_path / "roundtrip.mufinder"

    doc = service.create_project(root, "RoundTrip")
    structure = silicon_oxide_host()
    structure.properties = {
        "magnetic_structure": {"moments": [{"vector": [0.0, 0.0, 1.0]}]},
        "atomic_spins": [1.0, 0.0, -1.0],
    }
    service.add_structure(doc, structure)
    job = service.add_job(
        doc,
        label="Relax",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    job.status = JobStatus.COMPLETED
    result = Result(
        result_type="relaxation",
        data={
            "total_energy": -1.5,
            "max_force": 0.01,
            "warnings": ["a", "b"],
            "nested": {"k": [1, 2, 3]},
        },
        output_files=["out.castep"],
    )
    doc.results[result.id] = result
    service.save_project(doc)

    loaded = service.load_project(root)

    cases = [
        ("structures", loaded.structures[structure.id]),
        ("jobs", loaded.jobs[job.id]),
        ("results", loaded.results[result.id]),
    ]
    for entity_type, entity in cases:
        entity_id = str(entity.id)
        on_disk = (root / "entities" / entity_type / f"{entity_id}.json").read_text(
            encoding="utf-8"
        )
        regenerated = ProjectStore.canonical_json(to_jsonable(entity))
        assert regenerated == on_disk, f"{entity_type} {entity_id} did not round-trip exactly"


def test_ui_state_hash_skip(tmp_path: Path, monkeypatch) -> None:
    """state.json is skipped when unchanged and rewritten when mutated or deleted."""
    service = ProjectService()
    root = tmp_path / "uistate.mufinder"
    doc, _job, _result = _populated_document(service, root)
    doc.ui_state["some_form"] = {"a": 1}
    service.save_project(doc)

    state_writes: list[Path] = []
    original = ProjectStore.write_json

    def _tracking_write_json(self, path, payload):
        if Path(path).name == "state.json":
            state_writes.append(Path(path))
        return original(self, path, payload)

    monkeypatch.setattr(ProjectStore, "write_json", _tracking_write_json)

    # No mutation -> no state write.
    service.save_project(doc)
    assert state_writes == []

    # Mutate ui_state -> state rewritten.
    doc.ui_state["some_form"] = {"a": 2}
    service.save_project(doc)
    assert len(state_writes) == 1

    # External deletion forces a rewrite even without mutation.
    state_writes.clear()
    (root / "state.json").unlink()
    service.save_project(doc)
    assert len(state_writes) == 1


def test_deleting_entity_prunes_file_and_cache(tmp_path: Path) -> None:
    """Removing an entity from memory prunes its file and drops its cache entry."""
    service = ProjectService()
    root = tmp_path / "prune.mufinder"
    doc, job, _result = _populated_document(service, root)

    job_path = _entity_dir(root, "jobs") / f"{job.id}.json"
    assert job_path.exists()

    doc.jobs.pop(job.id)
    service.save_project(doc)

    assert not job_path.exists()
    # Cache entry for the removed job should be gone (per class docstring choice).
    root_key = ProjectService._resolved_root(root)
    assert (root_key, "jobs", str(job.id)) not in service._content_hashes


def test_failed_save_leaves_the_earlier_project_loadable_and_a_later_save_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ProjectService()
    root = tmp_path / "failed.mufinder"
    doc, _job, _result = _populated_document(service, root)
    doc.ui_state["some_form"] = {"a": 1}
    service.save_project(doc)
    state_before = (root / "state.json").read_bytes()
    manifest_before = (root / "project.json").read_bytes()

    def _no_space(_descriptor: int) -> None:
        raise OSError(28, "No space left on device")

    doc.ui_state["some_form"] = {"a": 2}
    with monkeypatch.context() as failing:
        failing.setattr(json_store.os, "fsync", _no_space)
        with pytest.raises(OSError, match="No space left"):
            service.save_project(doc)

    assert (root / "project.json").read_bytes() == manifest_before
    assert (root / "state.json").read_bytes() == state_before
    assert ProjectService().load_project(root).ui_state["some_form"] == {"a": 1}

    # The failed write was not recorded as saved, so the next save writes it.
    service.save_project(doc)
    assert ProjectService().load_project(root).ui_state["some_form"] == {"a": 2}
