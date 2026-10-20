"""Following a running job's ``.castep`` and reporting its geometry progress."""

from __future__ import annotations

from pathlib import Path
import threading
from uuid import uuid4

import pytest

from mufinder.castep_io.geometry_progress import parse_geometry_progress, parse_geometry_progress_text
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job
from mufinder.services.job_progress import (
    MIN_ON_DEMAND_REFRESH_SECONDS,
    GeometryProgressReader,
    JobProgressMonitor,
    castep_output_names,
    job_reports_geometry_progress,
    remote_castep_candidates,
)
from tests.support.cluster_ssh import GrowingOutputSSH

BFGS = Path(__file__).resolve().parents[1] / "fixtures" / "geometry" / "Si2_bfgs.castep"
LBFGS_CONTINUED = BFGS.with_name("Si2_lbfgs.castep")


def _job(stage: WorkflowStage = WorkflowStage.MUON_RELAXATION, *, status=JobStatus.RUNNING, target="local") -> Job:
    return Job(label="job", workflow_stage=stage, status=status, execution_target=target)


def test_geometry_progress_is_reported_for_relaxation_stages_only() -> None:
    assert job_reports_geometry_progress(_job(WorkflowStage.MUON_RELAXATION))
    assert job_reports_geometry_progress(_job(WorkflowStage.PRISTINE_RELAXATION))
    assert not job_reports_geometry_progress(_job(WorkflowStage.CONVERGENCE_CUTOFF))
    assert not job_reports_geometry_progress(_job(WorkflowStage.ENERGY_BARRIERS))


def test_castep_output_names_prefer_the_first_farm() -> None:
    assert castep_output_names("seed", 1) == ("seed.castep",)
    assert castep_output_names("seed", 4) == ("seed_farm001.castep", "seed.castep")
    assert remote_castep_candidates("/work/seed", "seed", 1) == ("/work/seed/seed.castep",)


def test_reader_follows_a_growing_file_reading_each_byte_once() -> None:
    data = BFGS.read_bytes()
    reader = GeometryProgressReader()
    offsets: list[int] = []

    def read_from(offset: int) -> bytes:
        offsets.append(offset)
        return data[offset:size]

    # Grow the file in uneven chunks that split lines and tables.
    for size in (*range(0, len(data), 7919), len(data)):
        progress = reader.read("job", size=size, read_from=read_from)
        complete = data[:size].rpartition(b"\n")[0].decode("utf-8")
        assert progress == parse_geometry_progress_text(complete)

    assert offsets == sorted(set(offsets))
    assert progress == parse_geometry_progress(BFGS)


def test_reader_starts_again_when_the_file_is_replaced() -> None:
    reader = GeometryProgressReader()
    bfgs = BFGS.read_bytes()
    full = LBFGS_CONTINUED.read_bytes()
    lbfgs = full[: full.index(b"Run started", full.index(b"Run started") + 1)]
    assert len(lbfgs) < len(bfgs)

    reader.read("job", size=len(bfgs), read_from=lambda offset: bfgs[offset:])
    progress = reader.read("job", size=len(lbfgs), read_from=lambda offset: lbfgs[offset:])

    assert progress is not None and progress.method == "LBFGS"


def test_reader_starts_again_when_a_task_farm_file_appears(tmp_path: Path) -> None:
    farmed = tmp_path / "seed_farm001.castep"
    master = tmp_path / "seed.castep"
    full = LBFGS_CONTINUED.read_bytes()
    master.write_bytes(full[: full.index(b"Run started", full.index(b"Run started") + 1)])
    reader = GeometryProgressReader()
    assert reader.read_local("job", (farmed, master)).method == "LBFGS"

    # The farm file is larger than the master offset, so resuming from that
    # offset would splice the two files together.
    farmed.write_bytes(BFGS.read_bytes())
    assert farmed.stat().st_size > master.stat().st_size

    assert reader.read_local("job", (farmed, master)) == parse_geometry_progress(BFGS)


def test_reader_does_not_hold_its_lock_while_reading() -> None:
    reader = GeometryProgressReader()
    data = BFGS.read_bytes()
    reading = threading.Event()
    release = threading.Event()

    def slow_read_from(offset: int) -> bytes:
        reading.set()
        release.wait(5)
        return data[offset:]

    worker = threading.Thread(target=lambda: reader.read("job", size=len(data), read_from=slow_read_from))
    worker.start()
    assert reading.wait(5)
    forgetting = threading.Thread(target=lambda: reader.forget_all_except(()))
    forgetting.start()
    forgetting.join(1)
    try:
        assert not forgetting.is_alive(), "forgetting waited for the read to finish"
    finally:
        release.set()
        worker.join(5)

    # The read finished after its cursor was forgotten, so nothing was kept:
    # the next read starts from the beginning.
    offsets: list[int] = []
    reader.read("job", size=len(data), read_from=lambda offset: offsets.append(offset) or data[offset:])
    assert offsets == [0]


def test_reader_reports_nothing_for_a_missing_file(tmp_path: Path) -> None:
    reader = GeometryProgressReader()

    assert reader.read_local("job", (tmp_path / "seed.castep",)) is None


def test_reader_reads_the_first_local_candidate_that_exists(tmp_path: Path) -> None:
    farmed = tmp_path / "seed_farm001.castep"
    master = tmp_path / "seed.castep"
    master.write_bytes(LBFGS_CONTINUED.read_bytes())
    reader = GeometryProgressReader()

    assert reader.read_local("job", (farmed, master)).method == "LBFGS"

    farmed.write_bytes(BFGS.read_bytes())
    assert GeometryProgressReader().read_local("job", (farmed, master)).method == "BFGS"


def test_reader_fetches_only_appended_bytes_over_sftp() -> None:
    ssh = GrowingOutputSSH()
    path = "/work/seed/seed.castep"
    data = BFGS.read_bytes()
    half = data.index(b"BFGS: finished iteration     4")
    ssh.remote_files[path] = data[:half]
    reader = GeometryProgressReader()

    first = reader.read_remote("job", ssh, ("/work/seed/seed_farm001.castep", path))
    ssh.remote_files[path] = data
    second = reader.read_remote("job", ssh, (path,))

    assert first.latest.iteration == 3
    assert second == parse_geometry_progress(BFGS)
    assert ssh.reads == [(path, 0), (path, half)]


class _Recorder:
    def __init__(self) -> None:
        self.remote_requests: list[list] = []
        self.changes: list[tuple] = []
        self.now = 1000.0


@pytest.fixture
def monitor(qapp, tmp_path: Path):
    recorder = _Recorder()
    paths: dict = {}
    monitor = JobProgressMonitor(
        local_output_paths=lambda job_id: paths.get(job_id, ()),
        request_remote_progress=recorder.remote_requests.append,
        run_in_background=lambda task: task(),
        clock=lambda: recorder.now,
    )
    monitor.progress_changed.connect(lambda job_id, progress: recorder.changes.append((job_id, progress)))
    yield monitor, recorder, paths
    monitor.deleteLater()


def test_monitor_follows_only_running_geometry_optimisations(monitor) -> None:
    monitor, recorder, paths = monitor
    relaxing = _job()
    paths[relaxing.id] = (BFGS,)
    queued = _job(status=JobStatus.QUEUED)
    sweep = _job(WorkflowStage.CONVERGENCE_CUTOFF)

    monitor.follow([relaxing, queued, sweep])

    assert monitor.followed_job_ids() == {relaxing.id}
    assert recorder.changes == [(relaxing.id, parse_geometry_progress(BFGS))]
    assert monitor.progress_for(relaxing.id).latest.iteration == 7


def test_monitor_asks_the_cluster_for_cluster_jobs_and_takes_its_answer(monitor) -> None:
    monitor, recorder, _paths = monitor
    remote = _job(target="scarf")
    progress = parse_geometry_progress(LBFGS_CONTINUED)

    monitor.follow([remote])
    monitor.refresh([remote.id], force=True)  # still in flight: not asked twice
    monitor.accept_remote_progress({remote.id: progress})

    assert recorder.remote_requests == [[remote.id]]
    assert recorder.changes == [(remote.id, progress)]


def test_monitor_rate_limits_on_demand_reads(monitor) -> None:
    monitor, recorder, _paths = monitor
    remote = _job(target="scarf")
    monitor.follow([remote])
    monitor.accept_remote_progress({remote.id: None})

    monitor.refresh([remote.id])
    recorder.now += MIN_ON_DEMAND_REFRESH_SECONDS + 1
    monitor.refresh([remote.id])

    assert recorder.remote_requests == [[remote.id], [remote.id]]


def test_monitor_keeps_the_last_progress_when_a_read_fails(monitor) -> None:
    monitor, recorder, _paths = monitor
    remote = _job(target="scarf")
    progress = parse_geometry_progress(BFGS)
    monitor.follow([remote])
    monitor.accept_remote_progress({remote.id: progress})

    monitor.refresh(force=True)
    monitor.accept_remote_progress({remote.id: None})

    assert monitor.progress_for(remote.id) == progress
    assert recorder.changes == [(remote.id, progress)]


def test_monitor_clears_progress_once_a_job_stops_running(monitor) -> None:
    monitor, recorder, paths = monitor
    job = _job()
    paths[job.id] = (BFGS,)
    monitor.follow([job])

    job.status = JobStatus.COMPLETED
    monitor.follow([job])

    assert monitor.progress_for(job.id) is None
    assert recorder.changes[-1] == (job.id, None)
    # A late answer for a job no longer followed is dropped.
    monitor.accept_remote_progress({job.id: parse_geometry_progress(BFGS)})
    assert monitor.progress_for(job.id) is None


def test_monitor_refreshes_periodically_only_while_enabled_with_jobs(monitor) -> None:
    monitor, _recorder, paths = monitor
    job = _job()
    paths[job.id] = (BFGS,)

    monitor.set_enabled(True)
    assert not monitor._timer.isActive()
    monitor.follow([job])
    assert monitor._timer.isActive()
    monitor.set_enabled(False)
    assert not monitor._timer.isActive()


def test_monitor_ignores_unknown_job_ids(monitor) -> None:
    monitor, recorder, _paths = monitor

    monitor.refresh([uuid4()], force=True)

    assert recorder.remote_requests == []
    assert recorder.changes == []
