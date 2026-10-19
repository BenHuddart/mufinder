"""Tests for remote cluster runner service."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

import mufinder.services.cluster_retrieval as cluster_retrieval_module
import mufinder.services.cluster_runner as cluster_runner_module
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import JobStatus
from mufinder.services.cluster_retrieval import (
    RetrievalFailure,
    RetrievalOutcome,
    RetrievalResult,
)
from mufinder.services.cluster_runner import ClusterJobRunner
from mufinder.services.job_artifacts import ParsedJobArtifacts, RetrievalFlags
from mufinder.services.scheduler import SchedulerJobInfo
from mufinder.services.ssh_client import SSHCredentials
from tests.support.cluster_ssh import FakeSSH, OutputUnavailableSSH


class _FakeScheduler:
    def __init__(self) -> None:
        self._states: dict[str, str] = {}
        self._elapsed: dict[str, str] = {}
        #: Accounting states for jobs the queue does not list (``sacct``).
        self._history: dict[str, str] = {}
        self.history_queries: list[list[str]] = []
        self.submit_requests: list[object] = []

    def submit(self, _ssh, request) -> str:
        self.submit_requests.append(request)
        return "12345"

    def cancel(self, _ssh, _job_id: str) -> None:
        return

    def status(self, _ssh, job_ids: list[str]):
        return {
            job_id: SchedulerJobInfo(
                job_id=job_id,
                state=self._states.get(job_id, "PD"),
                elapsed_runtime=self._elapsed.get(job_id, ""),
                name="job",
            )
            for job_id in job_ids
            if self._states.get(job_id, "PD") != "MISSING"
        }

    def history(self, _ssh, job_ids: list[str]):
        self.history_queries.append(list(job_ids))
        return {
            job_id: SchedulerJobInfo(job_id=job_id, state=self._history[job_id], name="job")
            for job_id in job_ids
            if job_id in self._history
        }

    def map_to_job_status(self, state: str) -> JobStatus:
        mapping = {
            "PD": JobStatus.QUEUED,
            "S": JobStatus.QUEUED,
            "R": JobStatus.RUNNING,
            "CD": JobStatus.COMPLETED,
            "F": JobStatus.FAILED,
            "OOM": JobStatus.FAILED,
            "TO": JobStatus.STOPPED,
            "CA": JobStatus.CANCELLED,
        }
        return mapping[state]

    def state_name(self, state: str) -> str:
        return {"F": "FAILED", "OOM": "OUT_OF_MEMORY", "TO": "TIMEOUT"}.get(state, state)


class _RaisingScheduler(_FakeScheduler):
    def status(self, _ssh, _job_ids: list[str]):
        raise RuntimeError("connection dropped")


def _prepare_local_job(tmp_path: Path, seed: str) -> Path:
    job_dir = tmp_path / seed
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{seed}.cell").write_text("cell", encoding="utf-8")
    (job_dir / f"{seed}.param").write_text("param", encoding="utf-8")
    return job_dir


def _template_file(tmp_path: Path) -> Path:
    template = tmp_path / "template.sh"
    template.write_text("#!/bin/sh\n#SBATCH -J $job_name\n$castep_command\n", encoding="utf-8")
    return template


def test_cluster_runner_polls_queue_once_per_minute(qapp) -> None:
    runner = ClusterJobRunner()

    assert runner._poll_timer.interval() == 60_000


def test_cluster_runner_submit_uploads_files_and_emits_queued(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _FakeScheduler()

    job_id = uuid4()
    seed = "conv_ecut_300"
    job_dir = _prepare_local_job(tmp_path, seed)
    template = _template_file(tmp_path)

    statuses: list[tuple[object, object]] = []
    runner.status_changed.connect(lambda jid, status: statuses.append((jid, status)))

    remote_job_id = runner.submit(
        job_id=job_id,
        local_job_dir=job_dir,
        seed_name=seed,
        num_processes=8,
        walltime="01:00:00",
        working_directory="/remote/work",
        template_script_path=str(template),
        cores_per_node=32,
    )

    assert remote_job_id == "12345"
    assert statuses[-1] == (job_id, JobStatus.QUEUED.value)
    assert any(cmd[0] == "put" and str(cmd[1]).endswith(".cell") for cmd in runner._ssh.commands)
    assert any(cmd[0] == "put" and str(cmd[1]).endswith(".param") for cmd in runner._ssh.commands)
    uploaded_runner = next(
        remote_path for action, remote_path in runner._ssh.commands if action == "put" and str(remote_path).endswith("run.sh")
    )
    assert b"\r\n" not in runner._ssh.remote_files[uploaded_runner]


def test_cluster_runner_submit_can_reuse_existing_remote_files(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler

    job_id = uuid4()
    seed = "conv_ecut_300"
    job_dir = _prepare_local_job(tmp_path, seed)
    template = _template_file(tmp_path)

    remote_job_id = runner.submit(
        job_id=job_id,
        local_job_dir=job_dir,
        seed_name=seed,
        num_processes=8,
        walltime="01:00:00",
        working_directory="/remote/work",
        template_script_path=str(template),
        cores_per_node=32,
        remote_directory="/remote/work/existing_continuation",
        reuse_existing_files=True,
    )

    assert remote_job_id == "12345"
    remote_script = "/remote/work/existing_continuation/run.sh"
    # Remote inputs are preserved, but the submit script is always regenerated so
    # the scheduler resources reflect this submission rather than the stale one.
    assert not any(
        action == "put" and str(payload).endswith((".cell", ".param"))
        for action, payload in runner._ssh.commands
    )
    assert ("put", remote_script) in runner._ssh.commands
    assert ("chmod", (remote_script, 0o755)) in runner._ssh.commands
    assert ("mkdir", ("/remote/work/existing_continuation", True)) in runner._ssh.commands
    request = scheduler.submit_requests[0]
    assert request.working_dir == "/remote/work/existing_continuation"
    assert request.script_path == remote_script


def test_cluster_runner_poll_moves_running_and_completed(qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler

    job_id = uuid4()
    seed = "conv_ecut_360"
    job_dir = _prepare_local_job(tmp_path, seed)
    template = _template_file(tmp_path)

    runner.submit(
        job_id=job_id,
        local_job_dir=job_dir,
        seed_name=seed,
        num_processes=8,
        walltime="01:00:00",
        working_directory="/remote/work",
        template_script_path=str(template),
        cores_per_node=32,
    )

    scheduler._states["12345"] = "R"
    status_events: list[tuple[object, object]] = []
    finished: list[tuple[object, object]] = []
    runner.status_changed.connect(lambda jid, status: status_events.append((jid, status)))
    runner.job_finished.connect(lambda jid, parsed: finished.append((jid, parsed)))

    monkeypatch.setattr(
        cluster_retrieval_module,
        "parse_castep_output",
        lambda _path: CastepResult(completed=True, total_energy=-1.0),
    )

    # First poll: running
    runner._poll_queue()
    assert (job_id, JobStatus.RUNNING.value) in status_events

    # Second poll: completion triggers retrieval and finish
    scheduler._states["12345"] = "CD"
    runner._poll_queue()
    assert any(status == JobStatus.RETRIEVING.value for _jid, status in status_events)
    assert any(status == JobStatus.COMPLETED.value for _jid, status in status_events)
    assert len(finished) == 1


def test_cluster_runner_poll_marks_timeout_as_stopped(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler

    job_id = uuid4()
    seed = "conv_ecut_420"
    job_dir = _prepare_local_job(tmp_path, seed)
    template = _template_file(tmp_path)

    runner.submit(
        job_id=job_id,
        local_job_dir=job_dir,
        seed_name=seed,
        num_processes=8,
        walltime="01:00:00",
        working_directory="/remote/work",
        template_script_path=str(template),
        cores_per_node=32,
    )

    scheduler._states["12345"] = "TO"
    status_events: list[tuple[object, object]] = []
    runner.status_changed.connect(lambda jid, status: status_events.append((jid, status)))

    runner._poll_queue()

    assert (job_id, JobStatus.STOPPED.value) in status_events


def test_cluster_runner_cancel_emits_cancelled(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _FakeScheduler()

    job_id = uuid4()
    seed = "conv_ecut_480"
    job_dir = _prepare_local_job(tmp_path, seed)
    template = _template_file(tmp_path)

    runner.submit(
        job_id=job_id,
        local_job_dir=job_dir,
        seed_name=seed,
        num_processes=8,
        walltime="01:00:00",
        working_directory="/remote/work",
        template_script_path=str(template),
        cores_per_node=32,
    )

    cancelled: list[object] = []
    statuses: list[tuple[object, object]] = []
    runner.job_cancelled.connect(lambda jid: cancelled.append(jid))
    runner.status_changed.connect(lambda jid, status: statuses.append((jid, status)))

    runner.cancel(job_id)

    assert cancelled == [job_id]
    assert statuses[-1] == (job_id, JobStatus.CANCELLED.value)


def test_cluster_runner_connect_disconnect_emits_status(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ClusterJobRunner()

    class _SSHFactory:
        def __init__(self, _config, *, settings=None) -> None:
            _ = settings
            self.is_connected = False

        def connect(self, password: str | None = None) -> None:
            _ = password
            self.is_connected = True

        def disconnect(self) -> None:
            self.is_connected = False

    monkeypatch.setattr(cluster_runner_module, "SSHConnection", _SSHFactory)
    monkeypatch.setattr(cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler())

    config = cluster_runner_module.ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="password",
        key_path="",
        default_working_directory="/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/run.sh",
    )

    events: list[bool] = []
    runner.connection_status_changed.connect(lambda connected: events.append(connected))

    runner.connect_to_cluster(config, password="secret")
    assert runner.is_connected is True
    assert events[-1] is True

    runner.disconnect()
    assert runner.is_connected is False
    assert events[-1] is False


def test_cluster_runner_poll_handles_scheduler_failure(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _RaisingScheduler()
    runner._config = cluster_runner_module.ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
    )

    job_id = uuid4()
    runner._tracked_jobs[job_id] = cluster_runner_module._TrackedClusterJob(
        job_id=job_id,
        remote_job_id="123",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )

    events: list[bool] = []
    failures: list[str] = []
    runner.connection_status_changed.connect(lambda connected: events.append(connected))
    runner.poll_failed.connect(lambda message: failures.append(message))

    runner._poll_queue()

    assert events[-1] is False
    assert failures[-1].startswith("Queue polling failed for alpha:")
    assert runner.is_connected is False
    assert runner._scheduler is None
    assert runner._config is None


def test_cluster_runner_poll_handles_cancelled_and_failed_states(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler

    cancel_job = uuid4()
    fail_job = uuid4()
    runner._tracked_jobs[cancel_job] = cluster_runner_module._TrackedClusterJob(
        job_id=cancel_job,
        remote_job_id="101",
        remote_dir="/remote/job_a",
        seed_name="a",
        local_output_dir=tmp_path,
    )
    runner._tracked_jobs[fail_job] = cluster_runner_module._TrackedClusterJob(
        job_id=fail_job,
        remote_job_id="102",
        remote_dir="/remote/job_b",
        seed_name="b",
        local_output_dir=tmp_path,
    )
    scheduler._states["101"] = "CA"
    scheduler._states["102"] = "F"

    cancelled: list[object] = []
    failed: list[tuple[object, str]] = []
    runner.job_cancelled.connect(lambda jid: cancelled.append(jid))
    runner.job_failed.connect(lambda jid, message: failed.append((jid, message)))

    runner._poll_queue()

    assert cancelled == [cancel_job]
    assert failed[0][0] == fail_job


def test_cluster_runner_poll_failed_downloads_numbered_error_files(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    ssh = FakeSSH()
    runner._ssh = ssh
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler

    job_id = uuid4()
    runner._tracked_jobs[job_id] = cluster_runner_module._TrackedClusterJob(
        job_id=job_id,
        remote_job_id="102",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )
    scheduler._states["102"] = "F"
    ssh.remote_files["/remote/job/seed.0001.err"] = b"numbered error"
    ssh.remote_files["/remote/job/seed.other.err"] = b"not selected"

    runner._poll_queue()

    assert (tmp_path / "seed.0001.err").read_bytes() == b"numbered error"
    assert not (tmp_path / "seed.other.err").exists()


def test_cluster_runner_poll_failed_downloads_slurm_output(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    ssh = FakeSSH()
    runner._ssh = ssh
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler

    job_id = uuid4()
    runner._tracked_jobs[job_id] = cluster_runner_module._TrackedClusterJob(
        job_id=job_id,
        remote_job_id="771300",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )
    scheduler._states["771300"] = "F"
    ssh.remote_files["/remote/job/slurm-771300.out"] = b"slurm failure details"

    runner._poll_queue()

    assert (tmp_path / "slurm-771300.out").read_bytes() == b"slurm failure details"


def test_cluster_runner_poll_failed_downloads_scheduler_job_output(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    ssh = FakeSSH()
    runner._ssh = ssh
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler

    job_id = uuid4()
    runner._tracked_jobs[job_id] = cluster_runner_module._TrackedClusterJob(
        job_id=job_id,
        remote_job_id="703535",
        remote_dir="/remote/job",
        seed_name="positivemuon_Random_Site_20",
        local_output_dir=tmp_path,
    )
    scheduler._states["703535"] = "F"
    ssh.remote_files["/remote/job/castep.mpi-703535.out"] = b"scheduler failure details"

    runner._poll_queue()

    assert (tmp_path / "castep.mpi-703535.out").read_bytes() == b"scheduler failure details"


def test_cluster_runner_addqueue_submission_uploads_the_staleness_marker(
    qapp, tmp_path: Path
) -> None:
    """``addqueue`` clusters get no ``run.sh``, so the submit script cannot be
    the staleness marker. ``{seed}.param`` is uploaded on every non-reuse
    submission whatever the queue system, which is why it is the marker."""

    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _FakeScheduler()
    runner._config = cluster_runner_module.ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="user",
        auth_method="key",
        key_path="",
        default_working_directory="/mnt/redwoodfs/user",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
    )
    seed = "seed"
    job_dir = _prepare_local_job(tmp_path, seed)

    runner.submit(
        job_id=uuid4(),
        local_job_dir=job_dir,
        seed_name=seed,
        num_processes=8,
        walltime="",
        working_directory="/remote/work",
        template_script_path="",
        cores_per_node=32,
    )

    uploads = [str(command[1]) for command in runner._ssh.commands if command[0] == "put"]
    assert "/remote/work/seed/seed.param" in uploads
    assert all(not upload.endswith("run.sh") for upload in uploads)


def test_cluster_runner_poll_names_the_scheduler_failure_state(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler
    job_id = uuid4()
    runner._tracked_jobs[job_id] = cluster_runner_module._TrackedClusterJob(
        job_id=job_id,
        remote_job_id="140",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )
    scheduler._states["140"] = "OOM"

    failures: list[str] = []
    runner.job_failed.connect(lambda _jid, message: failures.append(message))

    runner._poll_queue()

    assert failures == ["Scheduler reported the job as OUT_OF_MEMORY."]


def test_cluster_runner_poll_timeout_downloads_scheduler_log(qapp, tmp_path: Path) -> None:
    """A TIMEOUT stop downloads the scheduler log so the stop can be diagnosed."""

    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler
    job_id = uuid4()
    runner._tracked_jobs[job_id] = cluster_runner_module._TrackedClusterJob(
        job_id=job_id,
        remote_job_id="150",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )
    runner._ssh.remote_files["/remote/job/slurm-150.out"] = b"DUE TO TIME LIMIT"
    scheduler._states["150"] = "TO"

    statuses: list[str] = []
    runner.status_changed.connect(lambda _jid, status: statuses.append(status))

    runner._poll_queue()

    assert statuses == [JobStatus.STOPPED.value]
    assert (tmp_path / "slurm-150.out").read_bytes() == b"DUE TO TIME LIMIT"


def test_cluster_runner_connect_to_same_config_is_noop(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ClusterJobRunner()

    config = cluster_runner_module.ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/run.sh",
    )

    class _SSHFactory:
        def __init__(self, _config, *, settings=None) -> None:
            _ = settings
            self.is_connected = False

        def connect(self, password: str | None = None) -> None:
            _ = password
            self.is_connected = True

        def disconnect(self) -> None:
            self.is_connected = False

    monkeypatch.setattr(cluster_runner_module, "SSHConnection", _SSHFactory)
    monkeypatch.setattr(cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler())

    runner.connect_to_cluster(config)
    first_ssh = runner._ssh
    runner.connect_to_cluster(config)

    assert runner._ssh is first_ssh


def test_cluster_runner_submit_validation_errors(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _FakeScheduler()

    missing_dir = tmp_path / "missing"
    with pytest.raises(ConnectionError, match="does not exist"):
        runner.submit(
            job_id=uuid4(),
            local_job_dir=missing_dir,
            seed_name="seed",
            num_processes=8,
            walltime="01:00:00",
            working_directory="/remote/work",
            template_script_path=str(_template_file(tmp_path)),
            cores_per_node=32,
        )

    job_dir = _prepare_local_job(tmp_path, "seed2")
    with pytest.raises(ConnectionError, match="Remote working directory"):
        runner.submit(
            job_id=uuid4(),
            local_job_dir=job_dir,
            seed_name="seed2",
            num_processes=8,
            walltime="01:00:00",
            working_directory="",
            template_script_path=str(_template_file(tmp_path)),
            cores_per_node=32,
        )

    broken_seed = "seed3"
    broken_dir = tmp_path / broken_seed
    broken_dir.mkdir(parents=True, exist_ok=True)
    (broken_dir / f"{broken_seed}.cell").write_text("cell", encoding="utf-8")
    with pytest.raises(ConnectionError, match="Missing local input file"):
        runner.submit(
            job_id=uuid4(),
            local_job_dir=broken_dir,
            seed_name=broken_seed,
            num_processes=8,
            walltime="01:00:00",
            working_directory="/remote/work",
            template_script_path=str(_template_file(tmp_path)),
            cores_per_node=32,
        )

    valid_dir = _prepare_local_job(tmp_path, "seed4")
    with pytest.raises(ConnectionError, match="Remote job labels may contain only"):
        runner.submit(
            job_id=uuid4(),
            local_job_dir=valid_dir,
            seed_name="seed bad",
            num_processes=8,
            walltime="01:00:00",
            working_directory="/remote/work",
            template_script_path=str(_template_file(tmp_path)),
            cores_per_node=32,
        )


def test_cluster_runner_poll_no_connection_and_no_active_jobs(qapp) -> None:
    runner = ClusterJobRunner()

    # No connection branch
    runner._poll_queue()

    # Connected but nothing active branch
    runner._ssh = FakeSSH()
    runner._scheduler = _FakeScheduler()
    checked: list[object] = []
    runner.queue_checked.connect(lambda value: checked.append(value))
    runner._poll_queue()

    assert len(checked) == 1
    assert hasattr(checked[0], "checked_at")


def test_cluster_runner_queue_snapshot_includes_elapsed_runtime(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler

    job_id = uuid4()
    entry = cluster_runner_module._TrackedClusterJob(
        job_id=job_id,
        remote_job_id="200",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
        status=JobStatus.RUNNING,
    )
    runner._tracked_jobs[job_id] = entry
    scheduler._states["200"] = "R"
    scheduler._elapsed["200"] = "00:12:34"

    checked: list[object] = []
    runner.queue_checked.connect(lambda value: checked.append(value))
    runner._poll_queue()

    assert checked
    assert checked[-1].elapsed_by_job_id == {job_id: "00:12:34"}


def test_cluster_runner_poll_queued_state_and_same_state(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler

    job_id = uuid4()
    entry = cluster_runner_module._TrackedClusterJob(
        job_id=job_id,
        remote_job_id="200",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
        status=JobStatus.SUBMITTING,
    )
    runner._tracked_jobs[job_id] = entry
    scheduler._states["200"] = "PD"

    statuses: list[str] = []
    runner.status_changed.connect(lambda _jid, status: statuses.append(status))
    runner._poll_queue()

    assert statuses[-1] == JobStatus.QUEUED.value

    # Same-state branch should not emit anything new
    statuses_before = list(statuses)
    runner._poll_queue()
    assert statuses == statuses_before


def test_cluster_runner_helper_guards(qapp) -> None:
    runner = ClusterJobRunner()

    with pytest.raises(ConnectionError, match="not active"):
        runner._require_ssh()

    runner._ssh = FakeSSH()
    with pytest.raises(ConnectionError, match="not configured"):
        runner._require_scheduler()


def test_cluster_runner_cancel_ignores_terminal_job(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _FakeScheduler()

    job_id = uuid4()
    runner._tracked_jobs[job_id] = cluster_runner_module._TrackedClusterJob(
        job_id=job_id,
        remote_job_id="301",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
        terminal=True,
    )

    runner.cancel(job_id)


def test_cluster_runner_refresh_queue_calls_poll(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = ClusterJobRunner()
    calls: list[str] = []
    monkeypatch.setattr(runner, "_poll_queue", lambda: calls.append("poll"))

    runner.refresh_queue()

    assert calls == ["poll"]


def test_cluster_runner_submit_uses_builtin_scarf_template_when_unset(
    qapp,
    tmp_path: Path,
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _FakeScheduler()
    runner._config = cluster_runner_module.ClusterConfig(
        name="SCARF",
        hostname="ui1.scarf.rl.ac.uk",
        username="user",
        auth_method="key",
        key_path="",
        default_working_directory="/work4/isis/user",
        queue_system="SLURM",
        cores_per_node=24,
        default_walltime="12:00:00",
        template_script_path="builtin:scarf-slurm",
    )

    job_id = uuid4()
    seed = "conv_ecut_520"
    job_dir = _prepare_local_job(tmp_path, seed)

    runner.submit(
        job_id=job_id,
        local_job_dir=job_dir,
        seed_name=seed,
        num_processes=24,
        walltime="12:00:00",
        working_directory="/work4/isis/user/project/convergence_cutoff",
        template_script_path="",
        cores_per_node=24,
    )

    remote_script = "/work4/isis/user/project/convergence_cutoff/conv_ecut_520/run.sh"
    content = runner._ssh.remote_files[remote_script].decode("utf-8")
    assert "#SBATCH -p scarf" in content
    assert "srun -n 24 castep.mpi conv_ecut_520" in content


def test_cluster_runner_connected_properties_and_remote_directory_helpers(qapp) -> None:
    runner = ClusterJobRunner()

    assert runner.connected_cluster_name is None
    assert runner.connected_username is None

    class _SSHWithIsDir(FakeSSH):
        def __init__(self) -> None:
            super().__init__()
            self._dirs: set[str] = set()

        def mkdir(self, path: str, exist_ok: bool = True) -> None:
            _ = exist_ok
            self._dirs.add(path)

        def is_dir(self, path: str) -> bool:
            return path in self._dirs

    runner._ssh = _SSHWithIsDir()
    runner._config = cluster_runner_module.ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="",
        default_working_directory="/work/alice",
        queue_system="slurm",
        cores_per_node=24,
        default_walltime="01:00:00",
        template_script_path="builtin:slurm-basic",
    )

    assert runner.connected_cluster_name == "alpha"
    assert runner.connected_username == "alice"
    assert (
        runner.remote_job_directory(working_directory="/remote/work", seed_name="job_a")
        == "/remote/work/job_a"
    )

    runner._ssh.mkdir("/remote/work/job_a", exist_ok=True)
    assert runner.remote_directory_exists("/remote/work/job_a") is True


def test_cluster_runner_remote_file_exists_delegates_to_ssh_is_file(qapp) -> None:
    runner = ClusterJobRunner()

    class _SSHWithIsFile(FakeSSH):
        def __init__(self) -> None:
            super().__init__()
            self._files: set[str] = set()

        def touch(self, path: str) -> None:
            self._files.add(path)

        def is_file(self, path: str) -> bool:
            return path in self._files

    ssh = _SSHWithIsFile()
    ssh.touch("/remote/work/job_a/job_a.check")
    runner._ssh = ssh

    assert runner.remote_file_exists("/remote/work/job_a/job_a.check") is True
    assert runner.remote_file_exists("/remote/work/job_a/missing.check") is False


def test_cluster_runner_remote_file_exists_requires_active_connection(qapp) -> None:
    runner = ClusterJobRunner()

    with pytest.raises(ConnectionError):
        runner.remote_file_exists("/remote/work/job_a/job_a.check")


def test_cluster_runner_remote_file_exists_propagates_ssh_errors(qapp) -> None:
    runner = ClusterJobRunner()

    class _SSHRaisingOnIsFile(FakeSSH):
        def is_file(self, path: str) -> bool:
            raise ConnectionError("stat failed")

    runner._ssh = _SSHRaisingOnIsFile()

    with pytest.raises(ConnectionError, match="stat failed"):
        runner.remote_file_exists("/remote/work/job_a/job_a.check")


def test_cluster_runner_remove_remote_directory_delegates_to_ssh(qapp) -> None:
    runner = ClusterJobRunner()

    class _SSHWithRmtree(FakeSSH):
        def __init__(self) -> None:
            super().__init__()
            self.removed: list[str] = []

        def rmtree(self, path: str) -> None:
            self.removed.append(path)

    ssh = _SSHWithRmtree()
    runner._ssh = ssh

    runner.remove_remote_directory("/remote/work/job_a")

    assert ssh.removed == ["/remote/work/job_a"]


def test_cluster_runner_reconnect_reuses_cached_password_for_same_cluster(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = ClusterJobRunner()

    class _SSHFactory:
        passwords: list[str | None] = []

        def __init__(self, _config, *, settings=None) -> None:
            _ = settings
            self.is_connected = False

        def connect(self, password: str | None = None) -> None:
            self.__class__.passwords.append(password)
            self.is_connected = True

        def disconnect(self) -> None:
            self.is_connected = False

    monkeypatch.setattr(cluster_runner_module, "SSHConnection", _SSHFactory)
    monkeypatch.setattr(cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler())

    config = cluster_runner_module.ClusterConfig(
        name="redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="alice",
        auth_method="password",
        key_path="",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
    )

    runner.connect_to_cluster(config, password="secret")
    runner._ssh.is_connected = False
    runner.connect_to_cluster(config, password=None)

    assert _SSHFactory.passwords == ["secret", "secret"]


def test_cluster_runner_reconnect_reuses_cached_key_passphrase(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = ClusterJobRunner()

    class _SSHFactory:
        passwords: list[str | None] = []

        def __init__(self, _config, *, settings=None) -> None:
            _ = settings
            self.is_connected = False

        def connect(self, password: str | None = None) -> None:
            self.__class__.passwords.append(password)
            self.is_connected = True

        def disconnect(self) -> None:
            self.is_connected = False

    monkeypatch.setattr(cluster_runner_module, "SSHConnection", _SSHFactory)
    monkeypatch.setattr(cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler())

    config = cluster_runner_module.ClusterConfig(
        name="redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="alice",
        auth_method="key",
        key_path="/Users/alice/.ssh/id_ed25519",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
    )

    runner.connect_to_cluster(config, password="secret-passphrase")
    runner._ssh.is_connected = False
    runner.connect_to_cluster(config, password=None)

    assert _SSHFactory.passwords == ["secret-passphrase", "secret-passphrase"]


class _RecordingJumpSSH:
    """SSH double recording which secrets each connection attempt received."""

    attempts: list[tuple[str | None, str | None]] = []

    def __init__(self, _config, *, settings=None) -> None:
        _ = settings
        self.is_connected = False

    def connect(
        self,
        password: str | None = None,
        *,
        credentials: SSHCredentials | None = None,
    ) -> None:
        if credentials is None:
            self.__class__.attempts.append((password, None))
        else:
            self.__class__.attempts.append((credentials.target_secret, credentials.jump_secret))
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False


def _jump_host_redwood_config(**overrides) -> ClusterConfig:
    fields = dict(
        name="redwood",
        hostname="glamdring.physics.ox.ac.uk",
        port=22,
        username="alice",
        auth_method="password",
        jump_host_enabled=True,
        jump_hostname="gateway.physics.ox.ac.uk",
        jump_port=22,
        jump_username="alice",
        jump_auth_method="password",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
    )
    fields.update(overrides)
    return ClusterConfig(**fields)


@pytest.mark.parametrize(
    "changed",
    [
        {"hostname": "evil.example.org"},
        {"port": 2222},
        {"jump_hostname": "other-gateway.example.org"},
        {"jump_port": 2200},
    ],
    ids=["hostname", "port", "jump_hostname", "jump_port"],
)
def test_cluster_runner_reconnect_does_not_reuse_cached_credentials_after_endpoint_change(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    changed: dict[str, object],
) -> None:
    runner = ClusterJobRunner()
    _RecordingJumpSSH.attempts = []
    monkeypatch.setattr(cluster_runner_module, "SSHConnection", _RecordingJumpSSH)
    monkeypatch.setattr(cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler())

    original = _jump_host_redwood_config()
    runner.connect_to_cluster(
        original,
        credentials=SSHCredentials(target_secret="target-pw", jump_secret="jump-pw"),
    )
    runner._ssh.is_connected = False
    runner.connect_to_cluster(_jump_host_redwood_config(**changed), credentials=SSHCredentials())

    assert _RecordingJumpSSH.attempts == [("target-pw", "jump-pw"), (None, None)]


def test_cluster_runner_reconnect_does_not_reuse_direct_password_after_enabling_jump_host(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = ClusterJobRunner()
    _RecordingJumpSSH.attempts = []
    monkeypatch.setattr(cluster_runner_module, "SSHConnection", _RecordingJumpSSH)
    monkeypatch.setattr(cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler())

    # The jump fields stay saved while the jump host is switched off, so only
    # ``jump_host_enabled`` changes between the two connections.
    runner.connect_to_cluster(_jump_host_redwood_config(jump_host_enabled=False), password="target-pw")
    runner._ssh.is_connected = False
    runner.connect_to_cluster(_jump_host_redwood_config(), credentials=SSHCredentials())

    assert _RecordingJumpSSH.attempts == [("target-pw", None), (None, None)]


def test_cluster_runner_reconnect_reuses_cached_credentials_for_unchanged_endpoints(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = ClusterJobRunner()
    _RecordingJumpSSH.attempts = []
    monkeypatch.setattr(cluster_runner_module, "SSHConnection", _RecordingJumpSSH)
    monkeypatch.setattr(cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler())

    runner.connect_to_cluster(
        _jump_host_redwood_config(),
        credentials=SSHCredentials(target_secret="target-pw", jump_secret="jump-pw"),
    )
    runner._ssh.is_connected = False
    # A field outside the credential target (walltime) and host-name case do not
    # invalidate the cache.
    runner.connect_to_cluster(
        _jump_host_redwood_config(
            hostname="Glamdring.Physics.Ox.Ac.Uk",
            default_walltime="02:00:00",
        ),
        credentials=SSHCredentials(),
    )

    assert _RecordingJumpSSH.attempts == [("target-pw", "jump-pw"), ("target-pw", "jump-pw")]


def test_cluster_runner_submit_addqueue_forwards_runtime_options(
    qapp,
    tmp_path: Path,
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler
    runner._config = cluster_runner_module.ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="user",
        auth_method="key",
        key_path="",
        default_working_directory="/mnt/redwoodfs/user",
        queue_system="ADDQUEUE",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
        castep_executable="/mnt/redwoodfs/muon/castep.mpi",
        default_queue_name="redwood",
        default_memory_gb=7.5,
    )

    job_id = uuid4()
    seed = "conv_ecut_540"
    job_dir = _prepare_local_job(tmp_path, seed)

    runner.submit(
        job_id=job_id,
        local_job_dir=job_dir,
        seed_name=seed,
        num_processes=40,
        walltime="",
        working_directory="/mnt/redwoodfs/user/project/convergence_cutoff",
        template_script_path="",
        cores_per_node=32,
        memory_gb=9.0,
        queue_name="redwood-priority",
        comment="test comment",
    )

    request = scheduler.submit_requests[0]
    assert request.script_path == ""
    assert request.queue_name == "redwood-priority"
    assert request.memory_gb == 9.0
    assert request.comment == "test comment"
    assert request.castep_executable == "/mnt/redwoodfs/muon/castep.mpi"
    assert all(not str(cmd[1]).endswith("run.sh") for cmd in runner._ssh.commands if cmd[0] == "put")


def test_cluster_runner_track_existing_job_registers_active_entry(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()

    job_id = uuid4()
    output_dir = tmp_path / "tracked"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner.track_existing_job(
        job_id=job_id,
        remote_job_id="661111",
        remote_dir="/remote/work/tracked",
        seed_name="tracked",
        local_output_dir=output_dir,
        cluster_name="Redwood",
        status=JobStatus.RUNNING,
    )

    tracked = runner._tracked_jobs[job_id]
    assert tracked.remote_job_id == "661111"
    assert tracked.remote_dir == "/remote/work/tracked"
    assert tracked.cluster_name == "Redwood"
    assert tracked.terminal is False


def test_cluster_runner_track_existing_job_starts_polling_when_connected(
    qapp, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._config = cluster_runner_module.ClusterConfig(
        name="Redwood",
        hostname="redwood.example",
        username="alice",
        auth_method="key",
        key_path="",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=32,
        default_walltime="01:00:00",
        template_script_path="builtin:slurm-basic",
    )

    output_dir = tmp_path / "tracked_connected"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner.track_existing_job(
        job_id=uuid4(),
        remote_job_id="661112",
        remote_dir="/remote/work/tracked_connected",
        seed_name="tracked_connected",
        local_output_dir=output_dir,
        cluster_name="Redwood",
        status=JobStatus.RUNNING,
    )

    assert runner._poll_timer.isActive() is True
    runner._poll_timer.stop()


def test_cluster_runner_connect_starts_polling_for_existing_active_jobs(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    fake_ssh = FakeSSH()
    fake_scheduler = _FakeScheduler()

    monkeypatch.setattr(cluster_runner_module, "SSHConnection", lambda config, settings=None: fake_ssh)
    monkeypatch.setattr(cluster_runner_module, "create_scheduler", lambda *args, **kwargs: fake_scheduler)

    output_dir = tmp_path / "tracked_preconnect"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner.track_existing_job(
        job_id=uuid4(),
        remote_job_id="661113",
        remote_dir="/remote/work/tracked_preconnect",
        seed_name="tracked_preconnect",
        local_output_dir=output_dir,
        cluster_name="Redwood",
        status=JobStatus.RUNNING,
    )

    config = cluster_runner_module.ClusterConfig(
        name="Redwood",
        hostname="redwood.example",
        username="alice",
        auth_method="key",
        key_path="",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=32,
        default_walltime="01:00:00",
        template_script_path="builtin:slurm-basic",
    )

    runner.connect_to_cluster(config)

    assert runner._poll_timer.isActive() is True
    runner._poll_timer.stop()


def test_cluster_runner_poll_only_updates_jobs_for_connected_cluster(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler
    runner._config = cluster_runner_module.ClusterConfig(
        name="Redwood",
        hostname="redwood.example",
        username="alice",
        auth_method="key",
        key_path="",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=32,
        default_walltime="01:00:00",
        template_script_path="builtin:slurm-basic",
    )

    tracked_redwood = uuid4()
    tracked_scarf = uuid4()
    runner._tracked_jobs[tracked_redwood] = cluster_runner_module._TrackedClusterJob(
        job_id=tracked_redwood,
        remote_job_id="661200",
        remote_dir="/remote/work/redwood_job",
        seed_name="redwood_job",
        local_output_dir=tmp_path,
        cluster_name="Redwood",
        status=JobStatus.QUEUED,
    )
    runner._tracked_jobs[tracked_scarf] = cluster_runner_module._TrackedClusterJob(
        job_id=tracked_scarf,
        remote_job_id="771300",
        remote_dir="/remote/work/scarf_job",
        seed_name="scarf_job",
        local_output_dir=tmp_path,
        cluster_name="SCARF",
        status=JobStatus.QUEUED,
    )

    scheduler._states["661200"] = "R"
    scheduler._states["771300"] = "R"

    status_events: list[tuple[object, object]] = []
    runner.status_changed.connect(lambda jid, status: status_events.append((jid, status)))

    runner._poll_queue()

    assert (tracked_redwood, JobStatus.RUNNING.value) in status_events
    assert all(event_job_id != tracked_scarf for event_job_id, _status in status_events)
    assert runner._tracked_jobs[tracked_redwood].status == JobStatus.RUNNING
    assert runner._tracked_jobs[tracked_scarf].status == JobStatus.QUEUED


# ---------------------------------------------------------------------------
# Retrieval signal flow: each RetrievalResult becomes one signal sequence.
# ---------------------------------------------------------------------------


def _runner_with_canned_retrieval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, result: RetrievalResult
) -> tuple[ClusterJobRunner, object, list[dict[str, object]]]:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _FakeScheduler()
    entry = cluster_runner_module._TrackedClusterJob(
        job_id=uuid4(),
        remote_job_id="120",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
        retrieval=RetrievalFlags(retrieve_out_cell=True),
    )
    calls: list[dict[str, object]] = []

    def _retrieve(_ssh, **fields):
        calls.append(fields)
        return result

    monkeypatch.setattr(cluster_runner_module, "retrieve_job_outputs", _retrieve)
    return runner, entry, calls


def _record_signals(runner: ClusterJobRunner) -> dict[str, list[object]]:
    events: dict[str, list[object]] = {"status": [], "failed": [], "finished": []}
    runner.status_changed.connect(lambda _jid, status: events["status"].append(status))
    runner.job_failed.connect(lambda _jid, message: events["failed"].append(message))
    runner.job_finished.connect(
        lambda _jid, parsed, artifacts: events["finished"].append((parsed, artifacts))
    )
    return events


def test_cluster_runner_retrieval_passes_the_tracked_fields(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = RetrievalResult(RetrievalOutcome.STOPPED, "stopped")
    runner, entry, calls = _runner_with_canned_retrieval(monkeypatch, tmp_path, result)

    runner._retrieve_results(entry, missing_from_queue=True)

    assert calls == [
        {
            "remote_job_id": "120",
            "remote_dir": "/remote/job",
            "seed_name": "seed",
            "local_output_dir": tmp_path,
            "retrieval": RetrievalFlags(retrieve_out_cell=True),
            "missing_from_queue": True,
        }
    ]


def test_cluster_runner_completed_retrieval_emits_finished_then_completed(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parsed = CastepResult(completed=True, total_energy=-1.0)
    artifacts = ParsedJobArtifacts()
    result = RetrievalResult(RetrievalOutcome.COMPLETED, "output retrieved and parsed", parsed, artifacts)
    runner, entry, _calls = _runner_with_canned_retrieval(monkeypatch, tmp_path, result)
    events = _record_signals(runner)

    observation = runner._retrieve_results(entry)

    assert observation == "output retrieved and parsed"
    assert events["status"] == [JobStatus.RETRIEVING.value, JobStatus.COMPLETED.value]
    assert events["finished"] == [(parsed, artifacts)]
    assert events["failed"] == []
    assert entry.terminal is True


def test_cluster_runner_stopped_retrieval_emits_stopped_without_failure(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = RetrievalResult(RetrievalOutcome.STOPPED, "output incomplete; stopped (walltime)")
    runner, entry, _calls = _runner_with_canned_retrieval(monkeypatch, tmp_path, result)
    events = _record_signals(runner)

    observation = runner._retrieve_results(entry)

    assert observation == "output incomplete; stopped (walltime)"
    assert events["status"] == [JobStatus.RETRIEVING.value, JobStatus.STOPPED.value]
    assert events["failed"] == []
    assert events["finished"] == []
    assert entry.terminal is True


def test_cluster_runner_final_retrieval_failure_fails_immediately(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = RetrievalResult(
        RetrievalOutcome.FAILED,
        "missing from scheduler queue; output parsing failed",
        failure=RetrievalFailure("Failed to retrieve or parse remote output: boom"),
    )
    runner, entry, _calls = _runner_with_canned_retrieval(monkeypatch, tmp_path, result)
    events = _record_signals(runner)

    observation = runner._retrieve_results(entry, missing_from_queue=True)

    assert observation == "missing from scheduler queue; output parsing failed"
    assert events["failed"] == ["Failed to retrieve or parse remote output: boom"]
    assert events["status"] == [JobStatus.RETRIEVING.value, JobStatus.FAILED.value]
    assert entry.terminal is True
    assert entry.missing_retrieval_attempts == 0


def test_cluster_runner_retry_eligible_failure_counts_attempts_then_fails(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = RetrievalResult(
        RetrievalOutcome.FAILED,
        "missing from scheduler queue; output retrieval failed",
        failure=RetrievalFailure("Failed to retrieve or parse remote output: gone", retry_eligible=True),
    )
    runner, entry, _calls = _runner_with_canned_retrieval(monkeypatch, tmp_path, result)
    events = _record_signals(runner)
    limit = cluster_runner_module.MISSING_ENTRY_RETRIEVAL_ATTEMPTS

    observations = [runner._retrieve_results(entry, missing_from_queue=True) for _ in range(limit)]

    assert observations[:-1] == [
        f"missing from scheduler queue; waiting for output (attempt {attempt} of {limit})"
        for attempt in range(1, limit)
    ]
    assert observations[-1] == (
        f"missing from scheduler queue; output could not be retrieved after {limit} checks"
    )
    assert events["failed"] == [cluster_runner_module.MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE]
    assert events["status"] == [JobStatus.RETRIEVING.value] * limit + [JobStatus.FAILED.value]
    assert entry.missing_retrieval_attempts == limit
    assert entry.terminal is True


# ---------------------------------------------------------------------------
# Missing-scheduler-entry grace window (job-lifecycle hardening, rec #5) and
# per-job poll observations (rec #2).
# ---------------------------------------------------------------------------


def _submitted_runner(
    tmp_path: Path, seed: str, *, ssh: FakeSSH | None = None
) -> tuple[ClusterJobRunner, _FakeScheduler, object]:
    runner = ClusterJobRunner()
    runner._ssh = ssh or FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler
    job_id = uuid4()
    job_dir = _prepare_local_job(tmp_path, seed)
    template = _template_file(tmp_path)
    runner.submit(
        job_id=job_id,
        local_job_dir=job_dir,
        seed_name=seed,
        num_processes=8,
        walltime="01:00:00",
        working_directory="/remote/work",
        template_script_path=str(template),
        cores_per_node=32,
    )
    return runner, scheduler, job_id


def test_cluster_runner_missing_entry_defers_failure_within_grace_window(
    qapp, tmp_path: Path
) -> None:
    runner, scheduler, job_id = _submitted_runner(
        tmp_path, "grace_seed", ssh=OutputUnavailableSSH()
    )
    scheduler._states["12345"] = "MISSING"

    failed: list[tuple[object, str]] = []
    snapshots: list[object] = []
    runner.job_failed.connect(lambda jid, message: failed.append((jid, message)))
    runner.queue_checked.connect(lambda snapshot: snapshots.append(snapshot))

    # Attempts 1 and 2: output unavailable, but the job is NOT failed yet.
    runner._poll_queue()
    assert failed == []
    assert runner._tracked_jobs[job_id].terminal is False
    assert runner._tracked_jobs[job_id].missing_retrieval_attempts == 1
    assert snapshots[-1].observations_by_job_id[job_id] == (
        "missing from scheduler queue; waiting for output (attempt 1 of 3)"
    )

    runner._poll_queue()
    assert failed == []
    assert runner._tracked_jobs[job_id].terminal is False
    assert snapshots[-1].observations_by_job_id[job_id] == (
        "missing from scheduler queue; waiting for output (attempt 2 of 3)"
    )

    # Third failed attempt: the diagnostic failure is emitted verbatim.
    status_events: list[tuple[object, object]] = []
    runner.status_changed.connect(lambda jid, status: status_events.append((jid, status)))
    runner._poll_queue()
    assert len(failed) == 1
    assert failed[0][0] == job_id
    assert failed[0][1] == (
        "Job disappeared from the scheduler queue and its output could not be "
        "retrieved after 3 checks. The scheduler history may have expired, the "
        "queue status command may be filtered, or a different cluster may be "
        "connected. Retry retrieval or resubmit."
    )
    assert runner._tracked_jobs[job_id].terminal is True
    assert (job_id, JobStatus.FAILED.value) in status_events
    assert snapshots[-1].observations_by_job_id[job_id] == (
        "missing from scheduler queue; output could not be retrieved after 3 checks"
    )


def test_cluster_runner_missing_entry_reappearance_resets_counter(
    qapp, tmp_path: Path
) -> None:
    runner, scheduler, job_id = _submitted_runner(
        tmp_path, "reappear_seed", ssh=OutputUnavailableSSH()
    )
    snapshots: list[object] = []
    runner.queue_checked.connect(lambda snapshot: snapshots.append(snapshot))

    scheduler._states["12345"] = "MISSING"
    runner._poll_queue()
    assert runner._tracked_jobs[job_id].missing_retrieval_attempts == 1

    # The job reappears in the scheduler queue: the counter resets.
    scheduler._states["12345"] = "R"
    runner._poll_queue()
    assert runner._tracked_jobs[job_id].missing_retrieval_attempts == 0
    assert snapshots[-1].observations_by_job_id[job_id] == "scheduler reports RUNNING"

    # Vanishing again starts a fresh grace window.
    scheduler._states["12345"] = "MISSING"
    runner._poll_queue()
    assert runner._tracked_jobs[job_id].missing_retrieval_attempts == 1
    assert snapshots[-1].observations_by_job_id[job_id] == (
        "missing from scheduler queue; waiting for output (attempt 1 of 3)"
    )
    assert runner._tracked_jobs[job_id].terminal is False


def test_cluster_runner_missing_entry_parse_failure_fails_immediately(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner, scheduler, job_id = _submitted_runner(tmp_path, "parse_fail_seed")
    scheduler._states["12345"] = "MISSING"

    def _raise(_path: Path) -> CastepResult:
        raise ValueError("bad output")

    monkeypatch.setattr(cluster_retrieval_module, "parse_castep_output", _raise)

    failed: list[tuple[object, str]] = []
    snapshots: list[object] = []
    runner.job_failed.connect(lambda jid, message: failed.append((jid, message)))
    runner.queue_checked.connect(lambda snapshot: snapshots.append(snapshot))

    runner._poll_queue()

    # Downloaded-but-unreadable output is a real failure: no grace window.
    assert failed == [(job_id, "Failed to retrieve or parse remote output: bad output")]
    assert runner._tracked_jobs[job_id].terminal is True
    assert snapshots[-1].observations_by_job_id[job_id] == (
        "missing from scheduler queue; output parsing failed"
    )


def test_cluster_runner_missing_entry_successful_retrieval_completes(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner, scheduler, job_id = _submitted_runner(tmp_path, "late_success_seed")
    scheduler._states["12345"] = "MISSING"
    monkeypatch.setattr(
        cluster_retrieval_module,
        "parse_castep_output",
        lambda _path: CastepResult(completed=True, total_energy=-1.0),
    )

    finished: list[tuple[object, object]] = []
    snapshots: list[object] = []
    runner.job_finished.connect(lambda jid, parsed: finished.append((jid, parsed)))
    runner.queue_checked.connect(lambda snapshot: snapshots.append(snapshot))

    runner._poll_queue()

    assert len(finished) == 1
    assert finished[0][0] == job_id
    assert runner._tracked_jobs[job_id].terminal is True
    assert snapshots[-1].observations_by_job_id[job_id] == (
        "missing from scheduler queue; output retrieved and parsed"
    )


def test_cluster_runner_snapshot_observations_and_utc_checked_at(
    qapp, tmp_path: Path
) -> None:
    runner, scheduler, job_id = _submitted_runner(tmp_path, "observed_seed")
    scheduler._states["12345"] = "R"

    snapshots: list[object] = []
    runner.queue_checked.connect(lambda snapshot: snapshots.append(snapshot))

    runner._poll_queue()
    assert snapshots[-1].checked_at.tzinfo is not None
    assert snapshots[-1].observations_by_job_id[job_id] == "scheduler reports RUNNING"

    # Unchanged scheduler state emits no status event but is still observed.
    runner._poll_queue()
    assert snapshots[-1].observations_by_job_id[job_id] == "scheduler reports RUNNING"

    scheduler._states["12345"] = "PD"
    runner._poll_queue()
    assert snapshots[-1].observations_by_job_id[job_id] == "scheduler reports QUEUED"


def test_cluster_runner_poll_failure_leaves_tracked_jobs_untouched(
    qapp, tmp_path: Path
) -> None:
    # Rec #3: a poll failure disconnects and emits poll_failed, but must not
    # mutate tracked job state or emit any job lifecycle signal — staleness is
    # derived by the dashboard from last_checked_at plus the disconnect.
    runner, _scheduler, job_id = _submitted_runner(tmp_path, "poll_fail_seed")
    runner._scheduler = _RaisingScheduler()

    job_events: list[tuple[str, object]] = []
    poll_failures: list[str] = []
    runner.job_failed.connect(lambda jid, message: job_events.append(("failed", jid)))
    runner.job_finished.connect(lambda jid, parsed: job_events.append(("finished", jid)))
    runner.job_cancelled.connect(lambda jid: job_events.append(("cancelled", jid)))
    runner.status_changed.connect(lambda jid, status: job_events.append(("status", jid)))
    runner.poll_failed.connect(poll_failures.append)

    runner._poll_queue()

    assert len(poll_failures) == 1
    assert job_events == []
    entry = runner._tracked_jobs[job_id]
    assert entry.terminal is False
    assert entry.status == JobStatus.QUEUED
    assert entry.missing_retrieval_attempts == 0


def _submit_scarf_job(
    tmp_path: Path,
    *,
    seed: str,
    **submit_overrides: object,
) -> str:
    """Submit one SLURM job through the runner and return the uploaded run.sh."""

    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _FakeScheduler()
    runner._config = cluster_runner_module.ClusterConfig(
        name="SCARF",
        hostname="ui1.scarf.rl.ac.uk",
        username="user",
        auth_method="key",
        key_path="",
        default_working_directory="/work4/isis/user",
        queue_system="SLURM",
        cores_per_node=24,
        default_walltime="12:00:00",
        template_script_path="builtin:scarf-slurm",
    )

    submit_kwargs: dict[str, object] = {
        "job_id": uuid4(),
        "local_job_dir": _prepare_local_job(tmp_path, seed),
        "seed_name": seed,
        "num_processes": 144,
        "walltime": "12:00:00",
        "working_directory": "/work4/isis/user/project/muon_relaxation",
        "template_script_path": "",
        "cores_per_node": 24,
    }
    submit_kwargs.update(submit_overrides)
    runner.submit(**submit_kwargs)

    remote_script = f"/work4/isis/user/project/muon_relaxation/{seed}/run.sh"
    return runner._ssh.remote_files[remote_script].decode("utf-8")


def test_cluster_runner_submit_emits_mem_per_cpu_directive(qapp, tmp_path: Path) -> None:
    content = _submit_scarf_job(tmp_path, seed="mu_relax_1", mem_per_cpu_gb=6.0)

    assert "#SBATCH --mem-per-cpu=6G" in content
    assert "srun -n 144 castep.mpi mu_relax_1" in content


def test_cluster_runner_submit_omits_mem_per_cpu_directive_when_unset(
    qapp,
    tmp_path: Path,
) -> None:
    with_default = _submit_scarf_job(tmp_path, seed="mu_relax_2")
    explicit_none = _submit_scarf_job(tmp_path, seed="mu_relax_3", mem_per_cpu_gb=None)

    assert "--mem" not in with_default
    assert with_default == explicit_none.replace("mu_relax_3", "mu_relax_2")


def test_cluster_runner_submit_writes_the_chosen_partition(qapp, tmp_path: Path) -> None:
    runner = _slurm_runner()
    seed = "mu_relax_queue"

    runner.submit(
        job_id=uuid4(),
        local_job_dir=_prepare_local_job(tmp_path, seed),
        seed_name=seed,
        num_processes=48,
        walltime="12:00:00",
        working_directory="/remote/work",
        template_script_path="",
        cores_per_node=24,
        queue_name="daily",
    )

    content = runner._ssh.remote_files[f"/remote/work/{seed}/run.sh"].decode("utf-8")
    assert "#SBATCH -p daily" in content


def test_cluster_runner_submit_keeps_a_template_partition(qapp, tmp_path: Path) -> None:
    content = _submit_scarf_job(
        tmp_path,
        seed="mu_relax_scarf_queue",
        template_script_path="builtin:scarf-slurm",
        queue_name="devel",
    )

    assert "#SBATCH -p scarf" in content
    assert "devel" not in content


def _slurm_runner() -> ClusterJobRunner:
    """Return a runner wired to a SLURM cluster profile and one persistent fake SSH."""

    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _FakeScheduler()
    runner._config = cluster_runner_module.ClusterConfig(
        name="Test SLURM",
        hostname="cluster.example.invalid",
        username="user",
        auth_method="key",
        key_path="",
        default_working_directory="/remote/work",
        queue_system="SLURM",
        cores_per_node=24,
        default_walltime="12:00:00",
        template_script_path="",
    )
    return runner


def test_cluster_runner_resubmit_regenerates_script_with_new_walltime(
    qapp,
    tmp_path: Path,
) -> None:
    runner = _slurm_runner()
    seed = "mu_relax_walltime"
    job_dir = _prepare_local_job(tmp_path, seed)
    remote_dir = f"/remote/work/{seed}"
    remote_script = f"{remote_dir}/run.sh"

    submit_kwargs: dict[str, object] = {
        "local_job_dir": job_dir,
        "seed_name": seed,
        "num_processes": 48,
        "working_directory": "/remote/work",
        "template_script_path": "",
        "cores_per_node": 24,
        "remote_directory": remote_dir,
    }

    runner.submit(job_id=uuid4(), walltime="120:00:00", **submit_kwargs)
    assert "#SBATCH -t 120:00:00" in runner._ssh.remote_files[remote_script].decode("utf-8")

    runner.submit(
        job_id=uuid4(),
        walltime="168:00:00",
        reuse_existing_files=True,
        **submit_kwargs,
    )

    content = runner._ssh.remote_files[remote_script].decode("utf-8")
    assert "#SBATCH -t 168:00:00" in content
    assert "120:00:00" not in content


def test_cluster_runner_resubmit_applies_mem_per_cpu_directive(qapp, tmp_path: Path) -> None:
    runner = _slurm_runner()
    seed = "mu_relax_memory"
    job_dir = _prepare_local_job(tmp_path, seed)
    remote_dir = f"/remote/work/{seed}"
    remote_script = f"{remote_dir}/run.sh"

    submit_kwargs: dict[str, object] = {
        "local_job_dir": job_dir,
        "seed_name": seed,
        "num_processes": 48,
        "walltime": "12:00:00",
        "working_directory": "/remote/work",
        "template_script_path": "",
        "cores_per_node": 24,
        "remote_directory": remote_dir,
    }

    runner.submit(job_id=uuid4(), **submit_kwargs)
    assert "--mem" not in runner._ssh.remote_files[remote_script].decode("utf-8")

    runner.submit(
        job_id=uuid4(),
        reuse_existing_files=True,
        mem_per_cpu_gb=8.0,
        **submit_kwargs,
    )

    assert "#SBATCH --mem-per-cpu=8G" in runner._ssh.remote_files[remote_script].decode("utf-8")
    input_uploads = [
        payload
        for action, payload in runner._ssh.commands
        if action == "put" and str(payload).endswith((".cell", ".param"))
    ]
    assert sorted(input_uploads) == [f"{remote_dir}/{seed}.cell", f"{remote_dir}/{seed}.param"]


def test_checkpoint_probe_result_found_absent_unavailable() -> None:
    found = cluster_runner_module.CheckpointProbeResult.found()
    assert found.status is cluster_runner_module.CheckpointProbeStatus.FOUND
    assert found.reason == ""

    absent = cluster_runner_module.CheckpointProbeResult.absent()
    assert absent.status is cluster_runner_module.CheckpointProbeStatus.ABSENT
    assert absent.reason == ""

    unavailable = cluster_runner_module.CheckpointProbeResult.unavailable("not connected")
    assert unavailable.status is cluster_runner_module.CheckpointProbeStatus.UNAVAILABLE
    assert unavailable.reason == "not connected"


def test_checkpoint_probe_result_is_frozen() -> None:
    result = cluster_runner_module.CheckpointProbeResult.found()
    with pytest.raises(AttributeError):
        result.status = cluster_runner_module.CheckpointProbeStatus.ABSENT  # type: ignore[misc]


def test_scheduler_job_is_active_reads_the_scheduler_state(qapp) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler
    scheduler._states.update({"1": "R", "2": "PD", "3": "TO", "4": "MISSING"})
    scheduler.is_terminal = lambda state: state in {"CD", "F", "CA", "TO"}

    assert runner.scheduler_job_is_active("1") is True
    assert runner.scheduler_job_is_active("2") is True
    assert runner.scheduler_job_is_active("3") is False
    # No longer listed by the scheduler: finished.
    assert runner.scheduler_job_is_active("4") is False


def test_scheduler_job_is_active_reads_accounting_for_a_job_the_queue_hides(qapp) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler
    scheduler._states.update({"5": "MISSING", "6": "MISSING"})
    scheduler._history.update({"5": "PD", "6": "CD"})
    scheduler.is_terminal = lambda state: state in {"CD", "F", "CA", "TO"}

    # Pending in accounting: its directory must not be treated as finished.
    assert runner.scheduler_job_is_active("5") is True
    assert runner.scheduler_job_is_active("6") is False


def test_scheduler_job_is_active_propagates_an_accounting_failure(qapp) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler
    scheduler._states["7"] = "MISSING"

    def _broken_history(_ssh, _job_ids):
        raise RuntimeError("sacct: Problem talking to the database")

    scheduler.history = _broken_history

    # Unknown state: the caller about to delete the job's directory must refuse.
    with pytest.raises(RuntimeError, match="Problem talking to the database"):
        runner.scheduler_job_is_active("7")




# ---------------------------------------------------------------------------
# Long-queued jobs the queue command hides (issue #183): accounting fallback,
# watching a job failed for a missing scheduler entry, and the start signal.
# ---------------------------------------------------------------------------


def _record_lifecycle(runner: ClusterJobRunner) -> list[tuple[str, object]]:
    """Record the runner's lifecycle signals in emission order."""

    events: list[tuple[str, object]] = []
    runner.status_changed.connect(lambda _jid, status: events.append(("status", status)))
    runner.job_started.connect(lambda _jid: events.append(("started", None)))
    runner.job_failed.connect(lambda _jid, message: events.append(("failed", message)))
    runner.job_reappeared.connect(lambda _jid, status: events.append(("reappeared", status)))
    return events


def test_cluster_runner_pending_job_hidden_from_queue_is_never_failed(qapp, tmp_path: Path) -> None:
    runner, scheduler, job_id = _submitted_runner(
        tmp_path, "long_queue_seed", ssh=OutputUnavailableSSH()
    )
    events = _record_lifecycle(runner)
    snapshots: list[object] = []
    runner.queue_checked.connect(lambda snapshot: snapshots.append(snapshot))
    scheduler._states["12345"] = "MISSING"
    scheduler._history["12345"] = "PD"

    limit = cluster_runner_module.MISSING_ENTRY_RETRIEVAL_ATTEMPTS
    for _ in range(limit + 2):
        runner._poll_queue()

    # Accounting reports the job pending, so no retrieval is attempted and the
    # grace window never runs out, however long the job waits.
    assert events == []
    assert scheduler.history_queries == [["12345"]] * (limit + 2)
    tracked = runner._tracked_jobs[job_id]
    assert tracked.terminal is False
    assert tracked.missing_retrieval_attempts == 0
    assert not any(action == "get" for action, _path in runner._ssh.commands)
    assert snapshots[-1].observations_by_job_id[job_id] == (
        "missing from scheduler queue; scheduler accounting reports QUEUED"
    )

    # The job starts: accounting reports it running.
    scheduler._history["12345"] = "R"
    runner._poll_queue()

    assert events == [("status", JobStatus.RUNNING.value), ("started", None)]
    assert tracked.status == JobStatus.RUNNING


def test_cluster_runner_queue_lookup_skips_accounting_when_every_job_is_listed(
    qapp, tmp_path: Path
) -> None:
    runner, scheduler, _job_id = _submitted_runner(tmp_path, "listed_seed")

    runner._poll_queue()

    assert scheduler.history_queries == []


def test_cluster_runner_queued_to_running_emits_job_started_once(qapp, tmp_path: Path) -> None:
    runner, scheduler, _job_id = _submitted_runner(tmp_path, "start_seed")
    events = _record_lifecycle(runner)

    runner._poll_queue()
    scheduler._states["12345"] = "R"
    runner._poll_queue()
    runner._poll_queue()

    assert events == [("status", JobStatus.RUNNING.value), ("started", None)]


def test_cluster_runner_hidden_job_failed_after_grace_window_recovers_when_it_runs(
    qapp, tmp_path: Path
) -> None:
    runner, scheduler, job_id = _submitted_runner(
        tmp_path, "recover_seed", ssh=OutputUnavailableSSH()
    )
    events = _record_lifecycle(runner)
    snapshots: list[object] = []
    runner.queue_checked.connect(lambda snapshot: snapshots.append(snapshot))
    # Neither the queue nor accounting lists the pending job.
    scheduler._states["12345"] = "MISSING"

    limit = cluster_runner_module.MISSING_ENTRY_RETRIEVAL_ATTEMPTS
    for _ in range(limit):
        runner._poll_queue()

    tracked = runner._tracked_jobs[job_id]
    assert ("failed", cluster_runner_module.MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE) in events
    assert tracked.terminal is True
    assert tracked.watch_for_recovery is True

    # Still missing: the failed job stays watched without another retrieval.
    events.clear()
    downloads = len(runner._ssh.commands)
    runner._poll_queue()
    runner._poll_queue()
    assert events == []
    assert len(runner._ssh.commands) == downloads
    assert snapshots[-1].observations_by_job_id[job_id] == (
        "missing from scheduler queue; not in scheduler accounting; watching for the job to reappear"
    )

    # SLURM starts the job: the runner reports it once and ends the watch.
    scheduler._states["12345"] = "R"
    runner._poll_queue()
    runner._poll_queue()

    assert events == [("reappeared", JobStatus.RUNNING.value)]
    assert tracked.watch_for_recovery is False
    assert tracked.terminal is True
    assert snapshots[-2].observations_by_job_id[job_id] == (
        "scheduler reports RUNNING again; tracking resumed"
    )
    assert job_id not in snapshots[-1].observations_by_job_id


@pytest.mark.parametrize(
    ("history_state", "reappeared"),
    [("PD", JobStatus.QUEUED.value), ("CD", JobStatus.COMPLETED.value), ("CA", None)],
)
def test_cluster_runner_watched_job_follows_accounting(
    qapp, tmp_path: Path, history_state: str, reappeared: str | None
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler
    job_id = uuid4()
    runner.track_existing_job(
        job_id=job_id,
        remote_job_id="777",
        remote_dir="/remote/work/watched",
        seed_name="watched",
        local_output_dir=tmp_path,
        cluster_name="",
        status=JobStatus.FAILED,
        watch_for_recovery=True,
    )
    events = _record_lifecycle(runner)
    snapshots: list[object] = []
    runner.queue_checked.connect(lambda snapshot: snapshots.append(snapshot))
    scheduler._states["777"] = "MISSING"
    scheduler._history["777"] = history_state

    runner._poll_queue()

    expected = [] if reappeared is None else [("reappeared", reappeared)]
    assert events == expected
    assert runner._tracked_jobs[job_id].watch_for_recovery is False
    if reappeared is None:
        # Accounting confirms the job ended another way: it stays failed.
        assert snapshots[-1].observations_by_job_id[job_id] == "scheduler reports CA"


def test_cluster_runner_accounted_completion_with_unavailable_output_fails_without_watch(
    qapp, tmp_path: Path
) -> None:
    runner, scheduler, job_id = _submitted_runner(
        tmp_path, "accounted_seed", ssh=OutputUnavailableSSH()
    )
    events = _record_lifecycle(runner)
    scheduler._states["12345"] = "MISSING"
    scheduler._history["12345"] = "CD"

    limit = cluster_runner_module.MISSING_ENTRY_RETRIEVAL_ATTEMPTS
    for _ in range(limit):
        runner._poll_queue()

    # Accounting says the job completed, so the failure is the retrieval error
    # itself, retryable with Retry retrieval; the job is not watched.
    failures = [payload for kind, payload in events if kind == "failed"]
    assert len(failures) == 1
    assert str(failures[0]).startswith("Failed to retrieve or parse remote output")
    assert runner._tracked_jobs[job_id].watch_for_recovery is False


def test_cluster_runner_accounting_error_falls_back_to_the_grace_window(
    qapp, tmp_path: Path
) -> None:
    runner, scheduler, job_id = _submitted_runner(
        tmp_path, "sacct_error_seed", ssh=OutputUnavailableSSH()
    )

    def _broken_history(_ssh, _job_ids):
        raise RuntimeError("sacct timed out")

    scheduler.history = _broken_history
    scheduler._states["12345"] = "MISSING"
    disconnected: list[bool] = []
    runner.connection_status_changed.connect(disconnected.append)

    runner._poll_queue()

    assert disconnected == []
    assert runner._tracked_jobs[job_id].missing_retrieval_attempts == 1


def test_cluster_runner_track_existing_job_watches_only_failed_jobs(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _FakeScheduler()
    failed_id, running_id = uuid4(), uuid4()
    for job_id, status in ((failed_id, JobStatus.FAILED), (running_id, JobStatus.RUNNING)):
        runner.track_existing_job(
            job_id=job_id,
            remote_job_id=str(job_id.int % 1000),
            remote_dir="/remote/work/job",
            seed_name="job",
            local_output_dir=tmp_path,
            cluster_name="",
            status=status,
            watch_for_recovery=True,
        )

    assert runner._tracked_jobs[failed_id].watch_for_recovery is True
    # An active job is polled normally; the flag only applies to failed jobs.
    assert runner._tracked_jobs[running_id].watch_for_recovery is False
    assert runner._poll_timer.isActive() is True


def test_cluster_runner_suspended_running_job_stays_running(qapp, tmp_path: Path) -> None:
    runner, scheduler, job_id = _submitted_runner(tmp_path, "suspend_seed")
    events = _record_lifecycle(runner)
    snapshots: list[object] = []
    runner.queue_checked.connect(lambda snapshot: snapshots.append(snapshot))

    for state in ("R", "S", "R"):
        scheduler._states["12345"] = state
        runner._poll_queue()
        if state == "S":
            assert snapshots[-1].observations_by_job_id[job_id] == "scheduler reports QUEUED"

    # Resuming after a suspension is not a second start.
    assert events == [("status", JobStatus.RUNNING.value), ("started", None)]
    assert runner._tracked_jobs[job_id].status == JobStatus.RUNNING


def test_cluster_runner_stop_watching_drops_the_job_from_polls(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler
    watched_id, other_id = uuid4(), uuid4()
    for job_id, remote_id, status in (
        (watched_id, "801", JobStatus.FAILED),
        (other_id, "802", JobStatus.QUEUED),
    ):
        runner.track_existing_job(
            job_id=job_id,
            remote_job_id=remote_id,
            remote_dir="/remote/work/job",
            seed_name="job",
            local_output_dir=tmp_path,
            cluster_name="",
            status=status,
            watch_for_recovery=True,
        )
    scheduler._states["801"] = "MISSING"
    queried: list[list[str]] = []
    original_status = scheduler.status
    scheduler.status = lambda ssh, ids: queried.append(list(ids)) or original_status(ssh, ids)
    snapshots: list[object] = []
    runner.queue_checked.connect(lambda snapshot: snapshots.append(snapshot))

    runner._poll_queue()
    assert snapshots[-1].watched_job_ids == frozenset({watched_id})

    # The user reset the job: the GUI ends its watch, and polls skip it.
    runner.stop_watching([watched_id, other_id, uuid4()])
    runner._poll_queue()

    assert queried[-1] == ["802"]
    assert snapshots[-1].watched_job_ids == frozenset()
    assert runner._tracked_jobs[other_id].terminal is False
