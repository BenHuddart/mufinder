"""Jobs reload in the order they were created, not in job-file (UUID) order."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from mufinder.domain import models
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job, next_job_created_at
from mufinder.persistence.json_store import ProjectStore
from mufinder.services.project_service import ProjectService

_FROZEN = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _descending_ids(count: int) -> list[UUID]:
    # Job files load sorted by filename, so descending ids make the on-disk
    # order the exact reverse of creation order.
    return [UUID(int=1000 - index) for index in range(count)]


def _legacy_payload(job_id: UUID, label: str, **timestamps: datetime) -> dict:
    """A job file as written before ``created_at`` existed."""

    payload = {
        "id": str(job_id),
        "label": label,
        "workflow_stage": WorkflowStage.MUON_RELAXATION.value,
        "status": JobStatus.QUEUED.value,
    }
    payload.update({key: value.isoformat() for key, value in timestamps.items()})
    return payload


def _write_legacy_project(root: Path, payloads: list[dict], provenance: list[dict]) -> None:
    ProjectService().save_project(ProjectService().create_project(root, "Legacy"))
    store = ProjectStore(root)
    for payload in payloads:
        store.write_entity("jobs", payload["id"], payload)
    for record in provenance:
        store.append_provenance(record)


def _provenance(timestamp: datetime, action: str, **fields) -> dict:
    return {
        "id": str(UUID(int=int(timestamp.timestamp()))),
        "timestamp": timestamp.isoformat(),
        "action": action,
        **fields,
    }


def test_next_job_created_at_strictly_increases_when_the_clock_ties(monkeypatch) -> None:
    monkeypatch.setattr(models, "utc_now", lambda: _FROZEN)
    jobs: list[Job] = []
    for _ in range(5):
        jobs.append(Job(created_at=next_job_created_at(jobs)))

    stamps = [job.created_at for job in jobs]
    assert stamps[0] == _FROZEN
    assert stamps == sorted(set(stamps))


def test_add_job_stamps_increasing_creation_times_under_a_coarse_clock(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(models, "utc_now", lambda: _FROZEN)
    service = ProjectService()
    document = service.create_project(tmp_path / "coarse.mufinder", "Coarse")

    created = [
        service.add_job(
            document,
            label=f"job-{i}",
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type="geom",
        )
        for i in range(4)
    ]

    stamps = [job.created_at for job in created]
    assert stamps == sorted(set(stamps))


def test_jobs_reload_in_creation_order_not_file_order(tmp_path: Path, monkeypatch) -> None:
    # A frozen clock forces every job onto the same instant, so the order can
    # only survive the reload through the strictly increasing bump.
    monkeypatch.setattr(models, "utc_now", lambda: _FROZEN)
    service = ProjectService()
    root = tmp_path / "order.mufinder"
    document = service.create_project(root, "Order")
    for index, job_id in enumerate(_descending_ids(6)):
        job = Job(
            id=job_id, label=f"job-{index}", created_at=next_job_created_at(document.jobs.values())
        )
        document.jobs[job.id] = job
    created_order = [job.label for job in document.jobs.values()]
    service.save_project(document)

    reloaded = service.load_project(root)

    assert [job.label for job in reloaded.jobs.values()] == created_order


def test_resubmitting_or_resetting_a_job_does_not_move_it(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "stable.mufinder"
    document = service.create_project(root, "Stable")
    for index, job_id in enumerate(_descending_ids(3)):
        job = Job(
            id=job_id, label=f"job-{index}", created_at=next_job_created_at(document.jobs.values())
        )
        document.jobs[job.id] = job
    first = next(iter(document.jobs.values()))
    first.status = JobStatus.FAILED
    service.reset_failed_job(first)
    service.submit_job(first)
    service.save_project(document)

    reloaded = service.load_project(root)

    assert [job.label for job in reloaded.jobs.values()] == ["job-0", "job-1", "job-2"]


def test_legacy_batch_keeps_its_submission_order_from_provenance(tmp_path: Path) -> None:
    root = tmp_path / "legacy-batch.mufinder"
    ids = _descending_ids(4)
    batch_at = _FROZEN
    _write_legacy_project(
        root,
        [_legacy_payload(job_id, f"job-{index}") for index, job_id in enumerate(ids)],
        [
            # One batch record lists every job under a single timestamp.
            _provenance(batch_at, "submit_jobs", output_entity_ids=[str(job_id) for job_id in ids]),
            # A later per-job record must not move job-0 behind the others.
            _provenance(
                batch_at + timedelta(hours=1),
                "job_reset",
                output_entity_ids=[str(ids[0])],
                job_id=str(ids[0]),
            ),
        ],
    )

    reloaded = ProjectService().load_project(root)

    assert [job.label for job in reloaded.jobs.values()] == ["job-0", "job-1", "job-2", "job-3"]


def test_legacy_jobs_without_provenance_fall_back_to_lifecycle_timestamps(tmp_path: Path) -> None:
    root = tmp_path / "legacy-lifecycle.mufinder"
    ids = _descending_ids(3)
    _write_legacy_project(
        root,
        [
            _legacy_payload(ids[0], "never-submitted"),
            _legacy_payload(ids[1], "second", submitted_at=_FROZEN + timedelta(minutes=5)),
            _legacy_payload(
                ids[2],
                "first",
                submitted_at=_FROZEN + timedelta(minutes=9),
                completed_at=_FROZEN,
            ),
        ],
        [],
    )

    reloaded = ProjectService().load_project(root)

    assert [job.label for job in reloaded.jobs.values()] == ["first", "second", "never-submitted"]


def test_backfilled_creation_time_is_persisted_on_the_next_save(tmp_path: Path) -> None:
    root = tmp_path / "legacy-persist.mufinder"
    ids = _descending_ids(3)
    _write_legacy_project(
        root,
        [_legacy_payload(job_id, f"job-{index}") for index, job_id in enumerate(ids)],
        [_provenance(_FROZEN, "submit_jobs", output_entity_ids=[str(job_id) for job_id in ids])],
    )
    service = ProjectService()
    service.save_project(service.load_project(root))

    store = ProjectStore(root)
    assert all("created_at" in payload for payload in store.load_entities("jobs"))
    # With the provenance gone, the persisted values alone keep the order.
    store.provenance_path.unlink()
    reloaded = ProjectService().load_project(root)
    assert [job.label for job in reloaded.jobs.values()] == ["job-0", "job-1", "job-2"]


def test_naive_creation_timestamps_load_as_utc(tmp_path: Path) -> None:
    root = tmp_path / "naive.mufinder"
    ids = _descending_ids(2)
    naive = _legacy_payload(ids[0], "naive")
    naive["created_at"] = _FROZEN.replace(tzinfo=None).isoformat()
    aware = _legacy_payload(ids[1], "aware")
    aware["created_at"] = (_FROZEN + timedelta(seconds=1)).isoformat()
    _write_legacy_project(root, [naive, aware], [])

    reloaded = ProjectService().load_project(root)

    assert [job.label for job in reloaded.jobs.values()] == ["naive", "aware"]
    assert reloaded.jobs[ids[0]].created_at == _FROZEN


@pytest.mark.parametrize("count", [0, 1])
def test_next_job_created_at_uses_the_clock_when_it_is_ahead(count: int) -> None:
    earlier = [Job(created_at=_FROZEN - timedelta(days=1)) for _ in range(count)]

    assert next_job_created_at(earlier) > _FROZEN
