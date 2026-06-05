"""Tests for remote cluster runner service."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

import mufinder.services.cluster_runner as cluster_runner_module
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus
from mufinder.services.cluster_runner import ClusterJobRunner
from mufinder.services.scheduler import SchedulerJobInfo


class _FakeSSH:
    def __init__(self) -> None:
        self.is_connected = True
        self.commands: list[tuple[str, object]] = []
        self.remote_files: dict[str, bytes] = {}

    def mkdir(self, path: str, exist_ok: bool = True) -> None:
        self.commands.append(("mkdir", (path, exist_ok)))

    def put(self, local_path: Path, remote_path: str) -> None:
        self.commands.append(("put", remote_path))
        self.remote_files[remote_path] = Path(local_path).read_bytes()

    def chmod(self, remote_path: str, mode: int) -> None:
        self.commands.append(("chmod", (remote_path, mode)))

    def get(self, remote_path: str, local_path: Path) -> None:
        self.commands.append(("get", remote_path))
        local_path.write_bytes(self.remote_files.get(remote_path, b"ok"))

    def exists(self, remote_path: str) -> bool:
        return remote_path in self.remote_files

    def listdir(self, remote_path: str) -> list[str]:
        prefix = remote_path.rstrip("/") + "/"
        entries: list[str] = []
        for path in self.remote_files:
            if path.startswith(prefix):
                entries.append(path[len(prefix):].split("/", 1)[0])
        return sorted(set(entries))

    def disconnect(self) -> None:
        self.is_connected = False

    def connect(self, password: str | None = None) -> None:
        _ = password
        self.is_connected = True


class _FakeScheduler:
    def __init__(self) -> None:
        self._states: dict[str, str] = {}
        self._elapsed: dict[str, str] = {}
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

    def map_to_job_status(self, state: str) -> JobStatus:
        mapping = {
            "PD": JobStatus.QUEUED,
            "R": JobStatus.RUNNING,
            "CD": JobStatus.COMPLETED,
            "F": JobStatus.FAILED,
            "TO": JobStatus.STOPPED,
            "CA": JobStatus.CANCELLED,
        }
        return mapping[state]


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
    runner._ssh = _FakeSSH()
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
    runner._ssh = _FakeSSH()
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
    assert runner._ssh.commands == [("mkdir", ("/remote/work/existing_continuation", True))]
    request = scheduler.submit_requests[0]
    assert request.working_dir == "/remote/work/existing_continuation"
    assert request.script_path == "/remote/work/existing_continuation/run.sh"


def test_cluster_runner_poll_moves_running_and_completed(qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
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
        cluster_runner_module,
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
    runner._ssh = _FakeSSH()
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
    runner._ssh = _FakeSSH()
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
    runner._ssh = _FakeSSH()
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
    runner._ssh = _FakeSSH()
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
    ssh = _FakeSSH()
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
    ssh = _FakeSSH()
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
    ssh = _FakeSSH()
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


def test_cluster_runner_retrieve_incomplete_output_marks_failed(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
    runner._scheduler = _FakeScheduler()

    entry = cluster_runner_module._TrackedClusterJob(
        job_id=uuid4(),
        remote_job_id="120",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )

    monkeypatch.setattr(
        cluster_runner_module,
        "parse_castep_output",
        lambda _path: CastepResult(completed=False, error_message="not converged"),
    )

    failures: list[str] = []
    statuses: list[str] = []
    runner.job_failed.connect(lambda _jid, message: failures.append(message))
    runner.status_changed.connect(lambda _jid, status: statuses.append(status))

    runner._retrieve_results(entry)

    assert failures == ["not converged"]
    assert statuses[-1] == JobStatus.FAILED.value


def test_cluster_runner_retrieve_downloads_numbered_error_files(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    ssh = _FakeSSH()
    runner._ssh = ssh
    runner._scheduler = _FakeScheduler()

    entry = cluster_runner_module._TrackedClusterJob(
        job_id=uuid4(),
        remote_job_id="125",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )
    ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    ssh.remote_files["/remote/job/seed.0001.err"] = b"rank one"
    ssh.remote_files["/remote/job/seed.0002.err"] = b"rank two"

    monkeypatch.setattr(
        cluster_runner_module,
        "parse_castep_output",
        lambda _path: CastepResult(completed=True, total_energy=-1.0),
    )

    runner._retrieve_results(entry)

    assert (tmp_path / "seed.0001.err").read_bytes() == b"rank one"
    assert (tmp_path / "seed.0002.err").read_bytes() == b"rank two"


def test_cluster_runner_retrieve_downloads_optional_formatted_density(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
    runner._scheduler = _FakeScheduler()

    entry = cluster_runner_module._TrackedClusterJob(
        job_id=uuid4(),
        remote_job_id="125",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
        retrieve_den_fmt=True,
    )
    runner._ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    runner._ssh.remote_files["/remote/job/seed.den_fmt"] = b"density output"

    monkeypatch.setattr(
        cluster_runner_module,
        "parse_castep_output",
        lambda _path: CastepResult(completed=True, total_energy=-1.0),
    )

    runner._retrieve_results(entry)

    assert (tmp_path / "seed.den_fmt").read_bytes() == b"density output"


def test_cluster_runner_retrieve_downloads_optional_phonon(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
    runner._scheduler = _FakeScheduler()

    entry = cluster_runner_module._TrackedClusterJob(
        job_id=uuid4(),
        remote_job_id="125",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
        retrieve_phonon=True,
    )
    runner._ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    runner._ssh.remote_files["/remote/job/seed.phonon"] = b"phonon output"

    monkeypatch.setattr(
        cluster_runner_module,
        "parse_castep_output",
        lambda _path: CastepResult(completed=True, total_energy=-1.0),
    )

    runner._retrieve_results(entry)

    assert (tmp_path / "seed.phonon").read_bytes() == b"phonon output"


def test_cluster_runner_retrieve_uses_task_farm_primary_output(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
    runner._scheduler = _FakeScheduler()

    entry = cluster_runner_module._TrackedClusterJob(
        job_id=uuid4(),
        remote_job_id="125",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
        retrieve_phonon=True,
        task_farming_enabled=True,
        num_farms=4,
    )
    runner._ssh.remote_files["/remote/job/seed.castep"] = b"minimal output"
    runner._ssh.remote_files["/remote/job/seed_farm001.castep"] = b"farmed output"
    runner._ssh.remote_files["/remote/job/seed.phonon"] = b"phonon output"

    seen_paths: list[Path] = []
    monkeypatch.setattr(
        cluster_runner_module,
        "parse_castep_output",
        lambda path: (seen_paths.append(Path(path)), CastepResult(completed=True, total_energy=-1.0))[1],
    )

    runner._retrieve_results(entry)

    assert seen_paths == [tmp_path / "seed_farm001.castep"]
    assert (tmp_path / "seed_farm001.castep").read_bytes() == b"farmed output"


def test_cluster_runner_retrieve_downloads_optional_ts_file(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
    runner._scheduler = _FakeScheduler()

    entry = cluster_runner_module._TrackedClusterJob(
        job_id=uuid4(),
        remote_job_id="125",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
        retrieve_ts=True,
    )
    runner._ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    runner._ssh.remote_files["/remote/job/seed.ts"] = b"ts output"

    monkeypatch.setattr(
        cluster_runner_module,
        "parse_castep_output",
        lambda _path: CastepResult(completed=True, total_energy=-1.0),
    )

    runner._retrieve_results(entry)

    assert (tmp_path / "seed.ts").read_bytes() == b"ts output"


def test_cluster_runner_retrieve_skips_optional_files_when_not_requested(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
    runner._scheduler = _FakeScheduler()

    entry = cluster_runner_module._TrackedClusterJob(
        job_id=uuid4(),
        remote_job_id="126",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
        retrieve_den_fmt=False,
        retrieve_out_cell=False,
    )
    runner._ssh.remote_files["/remote/job/seed.castep"] = b"castep output"
    runner._ssh.remote_files["/remote/job/seed.den_fmt"] = b"density output"
    runner._ssh.remote_files["/remote/job/seed-out.cell"] = b"relaxed cell"
    runner._ssh.remote_files["/remote/job/seed.phonon"] = b"phonon output"

    monkeypatch.setattr(
        cluster_runner_module,
        "parse_castep_output",
        lambda _path: CastepResult(completed=True, total_energy=-1.0),
    )

    runner._retrieve_results(entry)

    assert not (tmp_path / "seed.den_fmt").exists()
    assert not (tmp_path / "seed-out.cell").exists()
    assert not (tmp_path / "seed.phonon").exists()


def test_cluster_runner_retrieve_incomplete_walltime_marks_stopped(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
    runner._scheduler = _FakeScheduler()

    entry = cluster_runner_module._TrackedClusterJob(
        job_id=uuid4(),
        remote_job_id="130",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )

    monkeypatch.setattr(
        cluster_runner_module,
        "parse_castep_output",
        lambda _path: CastepResult(completed=False, error_message="wall time limit"),
    )

    statuses: list[str] = []
    runner.status_changed.connect(lambda _jid, status: statuses.append(status))

    runner._retrieve_results(entry)

    assert statuses[-1] == JobStatus.STOPPED.value


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
    runner._ssh = _FakeSSH()
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
    runner._ssh = _FakeSSH()
    runner._scheduler = _FakeScheduler()
    checked: list[object] = []
    runner.queue_checked.connect(lambda value: checked.append(value))
    runner._poll_queue()

    assert len(checked) == 1
    assert hasattr(checked[0], "checked_at")


def test_cluster_runner_queue_snapshot_includes_elapsed_runtime(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
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
    runner._ssh = _FakeSSH()
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
    scheduler._states["200"] = "PD"

    statuses: list[str] = []
    runner.status_changed.connect(lambda _jid, status: statuses.append(status))
    runner._poll_queue()

    assert statuses[-1] == JobStatus.QUEUED.value

    # Same-state branch should not emit anything new
    statuses_before = list(statuses)
    runner._poll_queue()
    assert statuses == statuses_before


def test_cluster_runner_retrieve_parse_exception_marks_failed(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
    runner._scheduler = _FakeScheduler()

    entry = cluster_runner_module._TrackedClusterJob(
        job_id=uuid4(),
        remote_job_id="130",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )

    monkeypatch.setattr(
        cluster_runner_module,
        "parse_castep_output",
        lambda _path: (_ for _ in ()).throw(ValueError("parse boom")),
    )

    failures: list[str] = []
    runner.job_failed.connect(lambda _jid, message: failures.append(message))
    runner._retrieve_results(entry)

    assert "Failed to retrieve or parse remote output" in failures[0]


def test_cluster_runner_helper_guards(qapp) -> None:
    runner = ClusterJobRunner()

    assert runner._looks_like_walltime(CastepResult(completed=False, error_message=None)) is False

    with pytest.raises(ConnectionError, match="not active"):
        runner._require_ssh()

    runner._ssh = _FakeSSH()
    with pytest.raises(ConnectionError, match="not configured"):
        runner._require_scheduler()


def test_cluster_runner_cancel_ignores_terminal_job(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
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
    runner._ssh = _FakeSSH()
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

    class _SSHWithIsDir(_FakeSSH):
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


def test_cluster_runner_remove_remote_directory_delegates_to_ssh(qapp) -> None:
    runner = ClusterJobRunner()

    class _SSHWithRmtree(_FakeSSH):
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


def test_cluster_runner_submit_addqueue_forwards_runtime_options(
    qapp,
    tmp_path: Path,
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = _FakeSSH()
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
    runner._ssh = _FakeSSH()
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
    fake_ssh = _FakeSSH()
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
    runner._ssh = _FakeSSH()
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
