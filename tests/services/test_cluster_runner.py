"""Tests for remote cluster runner service."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

import mufinder.services.cluster_runner as cluster_runner_module
from mufinder.castep_io.geometry_progress import parse_geometry_progress
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.cluster import ClusterConfig
from mufinder.services.cluster_retrieval import RetrievalResult
from mufinder.services.cluster_runner import CheckTarget, ClusterJobRunner, QueueCheckSnapshot
from mufinder.services.job_reconcile import JobEvidence, OutputState, SchedulerState
from mufinder.services.job_artifacts import ParsedJobArtifacts, RetrievalFlags
from mufinder.services.scheduler import SchedulerJobInfo
from mufinder.services.ssh_client import SSHCredentials
from mufinder.services.job_progress import REMOTE_READ_CHUNK_BYTES
from tests.support.cluster_ssh import FakeSSH, GrowingOutputSSH, OutputUnavailableSSH
from tests.support.qt import pump_until


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

    def map_to_scheduler_state(self, state: str) -> SchedulerState:
        mapping = {
            "PD": SchedulerState.PENDING,
            "S": SchedulerState.PENDING,
            "R": SchedulerState.RUNNING,
            "CD": SchedulerState.COMPLETED,
            "F": SchedulerState.FAILED,
            "OOM": SchedulerState.FAILED,
            "TO": SchedulerState.TIMED_OUT,
            "CA": SchedulerState.CANCELLED,
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


def test_cluster_runner_submit_uploads_files_and_returns_the_scheduler_id(
    qapp, tmp_path: Path
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    runner._scheduler = _FakeScheduler()

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
    )

    assert remote_job_id == "12345"
    assert any(cmd[0] == "put" and str(cmd[1]).endswith(".cell") for cmd in runner._ssh.commands)
    assert any(cmd[0] == "put" and str(cmd[1]).endswith(".param") for cmd in runner._ssh.commands)
    uploaded_runner = next(
        remote_path
        for action, remote_path in runner._ssh.commands
        if action == "put" and str(remote_path).endswith("run.sh")
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
    monkeypatch.setattr(
        cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler()
    )

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
    monkeypatch.setattr(
        cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler()
    )

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


def test_cluster_runner_helper_guards(qapp) -> None:
    runner = ClusterJobRunner()

    with pytest.raises(ConnectionError, match="not active"):
        runner._require_ssh()

    runner._ssh = FakeSSH()
    with pytest.raises(ConnectionError, match="not configured"):
        runner._require_scheduler()


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
    monkeypatch.setattr(
        cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler()
    )

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
    monkeypatch.setattr(
        cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler()
    )

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
    monkeypatch.setattr(
        cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler()
    )

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
    monkeypatch.setattr(
        cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler()
    )

    # The jump fields stay saved while the jump host is switched off, so only
    # ``jump_host_enabled`` changes between the two connections.
    runner.connect_to_cluster(
        _jump_host_redwood_config(jump_host_enabled=False), password="target-pw"
    )
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
    monkeypatch.setattr(
        cluster_runner_module, "create_scheduler", lambda _queue, **_kwargs: _FakeScheduler()
    )

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
        processes_per_node=20,
    )

    request = scheduler.submit_requests[0]
    assert request.processes_per_node == 20
    assert request.script_path == ""
    assert request.queue_name == "redwood-priority"
    assert request.memory_gb == 9.0
    assert request.comment == "test comment"
    assert request.castep_executable == "/mnt/redwoodfs/muon/castep.mpi"
    assert all(
        not str(cmd[1]).endswith("run.sh") for cmd in runner._ssh.commands if cmd[0] == "put"
    )


def test_cluster_runner_refuses_an_impossible_placement_before_uploading(
    qapp,
    tmp_path: Path,
) -> None:
    runner = ClusterJobRunner()
    runner._ssh = FakeSSH()
    scheduler = _FakeScheduler()
    runner._scheduler = scheduler
    runner._config = cluster_runner_module.ClusterConfig(
        name="Redwood", hostname="glamdring.physics.ox.ac.uk", queue_system="ADDQUEUE"
    )
    seed = "conv_ecut_540"

    with pytest.raises(ValueError, match="does not divide 96 cores"):
        runner.submit(
            job_id=uuid4(),
            local_job_dir=_prepare_local_job(tmp_path, seed),
            seed_name=seed,
            num_processes=96,
            walltime="",
            working_directory="/mnt/redwoodfs/user/project",
            template_script_path="",
            cores_per_node=32,
            processes_per_node=64,
        )

    assert runner._ssh.commands == []
    assert scheduler.submit_requests == []


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


@pytest.mark.parametrize("queue_system", ["slurm", "addqueue"])
def test_connect_passes_accounting_path_to_scheduler(qapp, monkeypatch, queue_system) -> None:
    ssh = FakeSSH()
    commands = []

    def remote_command(command, timeout=30):
        commands.append(command)
        if command.startswith("/opt/slurm/squeue "):
            return 0, "", ""
        assert command.startswith("/opt/slurm/sacct ")
        return 0, "123|FAILED|00:00:01|job\n", ""

    ssh.exec_command = remote_command
    monkeypatch.setattr(cluster_runner_module, "SSHConnection", lambda *args, **kwargs: ssh)
    runner = ClusterJobRunner()
    runner.connect_to_cluster(
        ClusterConfig(
            name="custom",
            hostname="cluster.example",
            username="alice",
            queue_system=queue_system,
            queue_check_command_path="/opt/slurm/squeue",
            accounting_command_path="/opt/slurm/sacct",
        )
    )
    assert runner.scheduler_job_is_active("123") is False
    assert len(commands) == 2
    runner.disconnect()


# --- check_jobs: what one pass reports about each target ----------------------

_REMOTE_DIR = "/remote/work/job"


def _target(remote_job_id: str = "100", *, tmp_path: Path, **fields) -> CheckTarget:
    fields.setdefault("job_id", uuid4())
    fields.setdefault("remote_dir", _REMOTE_DIR)
    fields.setdefault("seed_name", "job")
    fields.setdefault("local_output_dir", tmp_path)
    return CheckTarget(remote_job_id=remote_job_id, **fields)


def _checking_runner(scheduler: _FakeScheduler | None = None, ssh: FakeSSH | None = None):
    """Return a connected runner with the lists its signals fill."""

    runner = ClusterJobRunner()
    runner._ssh = ssh or FakeSSH()
    runner._scheduler = scheduler or _FakeScheduler()
    checked: list[tuple[object, JobEvidence]] = []
    snapshots: list[QueueCheckSnapshot] = []
    failures: list[str] = []
    runner.job_checked.connect(lambda job_id, evidence: checked.append((job_id, evidence)))
    runner.queue_checked.connect(snapshots.append)
    runner.poll_failed.connect(failures.append)
    return runner, checked, snapshots, failures


def _retrieve_as(monkeypatch: pytest.MonkeyPatch, result: RetrievalResult) -> list[dict]:
    """Make every look at a remote folder return ``result``; return the calls."""

    calls: list[dict] = []

    def _retrieve(_ssh, **kwargs) -> RetrievalResult:
        calls.append(kwargs)
        return result

    monkeypatch.setattr(cluster_runner_module, "retrieve_job_outputs", _retrieve)
    return calls


@pytest.mark.parametrize(
    ("queue_state", "state", "detail"),
    [
        ("PD", SchedulerState.PENDING, "PD"),
        ("S", SchedulerState.PENDING, "S"),
        ("R", SchedulerState.RUNNING, "R"),
        ("CA", SchedulerState.CANCELLED, "CA"),
    ],
)
def test_check_reports_a_listed_job_without_reading_its_folder(
    qapp, tmp_path: Path, monkeypatch, queue_state: str, state: SchedulerState, detail: str
) -> None:
    scheduler = _FakeScheduler()
    scheduler._states["100"] = queue_state
    runner, checked, snapshots, _failures = _checking_runner(scheduler)
    looks = _retrieve_as(monkeypatch, RetrievalResult(OutputState.COMPLETE))
    target = _target(tmp_path=tmp_path)

    runner.check_jobs([target])

    assert looks == []
    assert [job_id for job_id, _ in checked] == [target.job_id]
    evidence = checked[0][1]
    assert (evidence.scheduler, evidence.scheduler_detail) == (state, detail)
    assert (evidence.output, evidence.from_accounting) == (OutputState.NOT_LOOKED, False)
    assert (evidence.remote_job_id, evidence.remote_directory) == ("100", _REMOTE_DIR)
    assert evidence.checked_at.utcoffset() is not None
    assert snapshots[0].job_ids == (target.job_id,)


@pytest.mark.parametrize(
    ("queue_state", "state", "detail"),
    [
        ("CD", SchedulerState.COMPLETED, "CD"),
        ("TO", SchedulerState.TIMED_OUT, "TIMEOUT"),
        ("OOM", SchedulerState.FAILED, "OUT_OF_MEMORY"),
    ],
)
def test_check_reads_the_folder_of_a_job_the_scheduler_reports_finished(
    qapp, tmp_path: Path, monkeypatch, queue_state: str, state: SchedulerState, detail: str
) -> None:
    scheduler = _FakeScheduler()
    scheduler._states["100"] = queue_state
    runner, checked, _snapshots, _failures = _checking_runner(scheduler)
    parsed = CastepResult(completed=True, total_energy=-1.0)
    artifacts = ParsedJobArtifacts()
    flags = RetrievalFlags(retrieve_out_cell=True)
    looks = _retrieve_as(
        monkeypatch, RetrievalResult(OutputState.COMPLETE, parsed=parsed, artifacts=artifacts)
    )
    target = _target(tmp_path=tmp_path, retrieval=flags)

    runner.check_jobs([target])

    assert looks == [
        {
            "remote_job_id": "100",
            "remote_dir": _REMOTE_DIR,
            "seed_name": "job",
            "local_output_dir": tmp_path,
            "retrieval": flags,
        }
    ]
    # Reported once before the download, so the job reads as retrieving, and
    # once with what the folder held.
    first, second = (evidence for _job_id, evidence in checked)
    assert (first.scheduler, first.output) == (state, OutputState.NOT_LOOKED)
    assert (second.scheduler, second.scheduler_detail) == (state, detail)
    assert second.output is OutputState.COMPLETE
    assert second.parsed is parsed and second.artifacts is artifacts


def test_check_carries_incomplete_output_and_the_walltime_mark(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    scheduler = _FakeScheduler()
    scheduler._states["100"] = "CD"
    runner, checked, _snapshots, _failures = _checking_runner(scheduler)
    _retrieve_as(
        monkeypatch,
        RetrievalResult(OutputState.INCOMPLETE, detail="SCF failed", walltime_seen=True),
    )

    runner.check_jobs([_target(tmp_path=tmp_path)])

    evidence = checked[-1][1]
    assert (evidence.output, evidence.output_detail, evidence.walltime_seen) == (
        OutputState.INCOMPLETE,
        "SCF failed",
        True,
    )


def test_check_reads_accounting_for_a_job_the_queue_does_not_list(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    # A pending job in a partition the queue command hides: accounting still
    # knows it, so it is reported as pending and its folder is left alone.
    scheduler = _FakeScheduler()
    scheduler._states["100"] = "MISSING"
    scheduler._history["100"] = "PD"
    runner, checked, _snapshots, _failures = _checking_runner(scheduler)
    looks = _retrieve_as(monkeypatch, RetrievalResult(OutputState.ABSENT))

    runner.check_jobs([_target(tmp_path=tmp_path)])

    assert scheduler.history_queries == [["100"]]
    assert looks == []
    evidence = checked[0][1]
    assert (evidence.scheduler, evidence.from_accounting) == (SchedulerState.PENDING, True)


def test_check_skips_accounting_when_the_queue_lists_every_job(qapp, tmp_path: Path) -> None:
    scheduler = _FakeScheduler()
    scheduler._states["100"] = "R"
    runner, _checked, _snapshots, _failures = _checking_runner(scheduler)

    runner.check_jobs([_target(tmp_path=tmp_path)])

    assert scheduler.history_queries == []


def test_check_reports_a_job_listed_nowhere_with_what_its_folder_holds(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    scheduler = _FakeScheduler()
    scheduler._states["100"] = "MISSING"
    runner, checked, _snapshots, _failures = _checking_runner(scheduler)
    _retrieve_as(monkeypatch, RetrievalResult(OutputState.ABSENT))

    runner.check_jobs([_target(tmp_path=tmp_path)])

    # No report ahead of the look: nothing says the job finished.
    assert len(checked) == 1
    evidence = checked[0][1]
    assert (evidence.scheduler, evidence.from_accounting) == (SchedulerState.NOT_LISTED, False)
    assert evidence.output is OutputState.ABSENT


def test_check_treats_a_failed_accounting_lookup_as_not_listed(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    class _BrokenAccounting(_FakeScheduler):
        def history(self, _ssh, _job_ids):
            raise RuntimeError("sacct: command not found")

    scheduler = _BrokenAccounting()
    scheduler._states["100"] = "MISSING"
    runner, checked, snapshots, failures = _checking_runner(scheduler)
    _retrieve_as(monkeypatch, RetrievalResult(OutputState.COMPLETE))

    runner.check_jobs([_target(tmp_path=tmp_path)])

    assert checked[-1][1].scheduler is SchedulerState.NOT_LISTED
    assert checked[-1][1].output is OutputState.COMPLETE
    assert failures == [] and len(snapshots) == 1
    assert runner.is_connected


def test_check_reads_the_folder_of_a_job_with_no_scheduler_id_only_on_request(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    class _NoQueries(_FakeScheduler):
        def status(self, _ssh, job_ids):
            raise AssertionError(f"queue asked about {job_ids}")

    runner, checked, _snapshots, _failures = _checking_runner(_NoQueries())
    looks = _retrieve_as(monkeypatch, RetrievalResult(OutputState.INCOMPLETE))

    runner.check_jobs([_target("", tmp_path=tmp_path)])
    assert looks == []
    assert (checked[0][1].scheduler, checked[0][1].output) == (
        SchedulerState.NO_JOB_ID,
        OutputState.NOT_LOOKED,
    )

    runner.check_jobs([_target("", tmp_path=tmp_path, on_request=True)])

    assert len(looks) == 1
    evidence = checked[1][1]
    assert (evidence.scheduler, evidence.output) == (
        SchedulerState.NO_JOB_ID,
        OutputState.INCOMPLETE,
    )
    assert evidence.remote_job_id == ""


def test_check_reads_a_cancelled_jobs_folder_only_when_the_user_asked(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    scheduler = _FakeScheduler()
    scheduler._states["100"] = "CA"
    runner, checked, _snapshots, _failures = _checking_runner(scheduler)
    looks = _retrieve_as(monkeypatch, RetrievalResult(OutputState.COMPLETE))

    runner.check_jobs([_target(tmp_path=tmp_path)])
    runner.check_jobs([_target(tmp_path=tmp_path, on_request=True)])

    assert len(looks) == 1
    assert [evidence.output for _job_id, evidence in checked] == [
        OutputState.NOT_LOOKED,
        OutputState.COMPLETE,
    ]
    assert [evidence.on_request for _job_id, evidence in checked] == [False, True]


def test_check_reports_a_folder_that_raises_as_unreachable(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    scheduler = _FakeScheduler()
    scheduler._states["100"] = "MISSING"
    scheduler._states["101"] = "R"
    runner, checked, snapshots, _failures = _checking_runner(scheduler)

    def _raise(_ssh, **_kwargs):
        raise OSError("channel closed")

    monkeypatch.setattr(cluster_runner_module, "retrieve_job_outputs", _raise)

    runner.check_jobs([_target(tmp_path=tmp_path), _target("101", tmp_path=tmp_path)])

    # One job's folder does not end the pass for the others.
    first, second = (evidence for _job_id, evidence in checked)
    assert (first.output, first.output_detail) == (OutputState.UNREACHABLE, "channel closed")
    assert second.scheduler is SchedulerState.RUNNING
    assert len(snapshots) == 1


def test_check_downloads_output_and_logs_of_a_failed_job(qapp, tmp_path: Path) -> None:
    ssh = FakeSSH()
    ssh.remote_files[f"{_REMOTE_DIR}/job.castep"] = b"partial"
    ssh.remote_files[f"{_REMOTE_DIR}/job.0001.err"] = b"Error in parameters"
    ssh.remote_files[f"{_REMOTE_DIR}/slurm-100.out"] = b"oom-kill"
    scheduler = _FakeScheduler()
    scheduler._states["100"] = "F"
    runner, checked, _snapshots, _failures = _checking_runner(scheduler, ssh)

    runner.check_jobs([_target(tmp_path=tmp_path)])

    assert (tmp_path / "job.0001.err").read_bytes() == b"Error in parameters"
    assert (tmp_path / "slurm-100.out").read_bytes() == b"oom-kill"
    evidence = checked[-1][1]
    assert (evidence.scheduler, evidence.scheduler_detail) == (SchedulerState.FAILED, "FAILED")
    assert evidence.output is OutputState.INCOMPLETE


def test_check_reports_no_output_when_the_download_fails(qapp, tmp_path: Path) -> None:
    ssh = OutputUnavailableSSH()
    ssh.remote_files[f"{_REMOTE_DIR}/job.castep"] = b"castep output"
    scheduler = _FakeScheduler()
    scheduler._states["100"] = "CD"
    runner, checked, _snapshots, _failures = _checking_runner(scheduler, ssh)

    runner.check_jobs([_target(tmp_path=tmp_path)])

    assert checked[-1][1].output is OutputState.UNREACHABLE


def test_check_snapshot_carries_elapsed_runtimes_and_the_jobs_checked(qapp, tmp_path: Path) -> None:
    scheduler = _FakeScheduler()
    scheduler._states.update({"100": "R", "101": "PD"})
    scheduler._elapsed["100"] = "01:02:03"
    runner, _checked, snapshots, _failures = _checking_runner(scheduler)
    running = _target(tmp_path=tmp_path)
    queued = _target("101", tmp_path=tmp_path)

    runner.check_jobs([running, queued])

    assert snapshots[0].elapsed_by_job_id == {running.job_id: "01:02:03"}
    assert snapshots[0].job_ids == (running.job_id, queued.job_id)
    assert snapshots[0].checked_at.utcoffset() is not None


def test_check_with_no_targets_still_reports_the_pass(qapp) -> None:
    runner, checked, snapshots, _failures = _checking_runner()

    runner.check_jobs([])

    assert checked == []
    assert snapshots[0].job_ids == () and snapshots[0].elapsed_by_job_id == {}


def test_check_skips_targets_of_another_cluster(qapp, tmp_path: Path) -> None:
    scheduler = _FakeScheduler()
    scheduler._states.update({"100": "R", "200": "R"})
    runner, checked, snapshots, _failures = _checking_runner(scheduler)
    runner._config = ClusterConfig(name="alpha", hostname="alpha.example")
    here = _target(tmp_path=tmp_path, cluster_name="Alpha")
    elsewhere = _target("200", tmp_path=tmp_path, cluster_name="beta")

    runner.check_jobs([here, elsewhere])

    assert [job_id for job_id, _ in checked] == [here.job_id]
    assert snapshots[0].job_ids == (here.job_id,)


def test_check_whose_queue_command_fails_disconnects_and_ends_no_job(qapp, tmp_path: Path) -> None:
    runner, checked, snapshots, failures = _checking_runner(_RaisingScheduler())
    runner._config = ClusterConfig(name="alpha", hostname="alpha.example")
    target = _target(tmp_path=tmp_path)

    runner.check_jobs([target])

    assert [(job_id, evidence.scheduler) for job_id, evidence in checked] == [
        (target.job_id, SchedulerState.UNAVAILABLE)
    ]
    assert checked[0][1].scheduler_detail == "connection dropped"
    assert failures == ["Queue polling failed for alpha: connection dropped"]
    assert snapshots == []
    assert not runner.is_connected


def test_check_reports_a_transport_that_died(qapp, tmp_path: Path) -> None:
    runner, checked, snapshots, failures = _checking_runner()
    runner._config = ClusterConfig(name="alpha", hostname="alpha.example")
    runner._ssh.is_connected = False
    connection_events: list[bool] = []
    runner.connection_status_changed.connect(connection_events.append)

    runner.check_jobs([_target(tmp_path=tmp_path)])

    assert failures == ["Queue polling failed for alpha: the connection was lost."]
    assert connection_events == [False]
    assert checked == [] and snapshots == []


def test_check_closes_a_dead_connection_without_a_warning_when_nothing_was_asked(qapp) -> None:
    runner, _checked, _snapshots, failures = _checking_runner()
    runner._ssh.is_connected = False

    runner.check_jobs([])

    assert failures == []
    assert runner._ssh is None


def test_check_is_silent_when_never_connected(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    events: list[object] = []
    runner.job_checked.connect(lambda *args: events.append(args))
    runner.queue_checked.connect(events.append)
    runner.poll_failed.connect(events.append)
    runner.connection_status_changed.connect(events.append)

    runner.check_jobs([_target(tmp_path=tmp_path)])

    assert events == []


# --- cancel and geometry progress ----------------------------------------------


def test_cancel_asks_the_scheduler_and_reports_the_job(qapp) -> None:
    cancelled_remote: list[str] = []

    class _Recording(_FakeScheduler):
        def cancel(self, _ssh, job_id: str) -> None:
            cancelled_remote.append(job_id)

    runner, _checked, _snapshots, _failures = _checking_runner(_Recording())
    cancelled: list[object] = []
    runner.job_cancelled.connect(cancelled.append)
    job_id = uuid4()

    runner.cancel(job_id, " 100 ")

    assert cancelled_remote == ["100"]
    assert cancelled == [job_id]


def test_cancel_without_a_scheduler_id_reports_nothing(qapp) -> None:
    runner, _checked, _snapshots, _failures = _checking_runner()
    cancelled: list[object] = []
    runner.job_cancelled.connect(cancelled.append)

    runner.cancel(uuid4())

    assert cancelled == []


def test_cancel_scheduler_job_reports_no_job(qapp) -> None:
    cancelled_remote: list[str] = []

    class _Recording(_FakeScheduler):
        def cancel(self, _ssh, job_id: str) -> None:
            cancelled_remote.append(job_id)

    runner, _checked, _snapshots, _failures = _checking_runner(_Recording())
    cancelled: list[object] = []
    runner.job_cancelled.connect(cancelled.append)

    runner.cancel_scheduler_job(" 100 ")

    assert cancelled_remote == ["100"]
    assert cancelled == []
    with pytest.raises(ConnectionError):
        ClusterJobRunner().cancel_scheduler_job("100")


def test_cancel_requires_a_connection(qapp) -> None:
    with pytest.raises(ConnectionError):
        ClusterJobRunner().cancel(uuid4(), "100")


def test_reads_geometry_progress_of_its_targets(qapp, tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "geometry" / "Si2_bfgs.castep"
    ssh = GrowingOutputSSH()
    ssh.remote_files["/remote/farmed/seed_farm001.castep"] = fixture.read_bytes()
    runner, _checked, _snapshots, _failures = _checking_runner(ssh=ssh)
    runner._config = ClusterConfig(name="alpha", hostname="alpha.example")
    results: list[dict] = []
    runner.geometry_progress_read.connect(results.append)
    farmed = _target(
        tmp_path=tmp_path,
        remote_dir="/remote/farmed",
        seed_name="seed",
        cluster_name="alpha",
        retrieval=RetrievalFlags(task_farming_enabled=True, num_farms=2),
    )
    missing = _target(tmp_path=tmp_path, remote_dir="/remote/missing", seed_name="seed")
    elsewhere = _target(tmp_path=tmp_path, cluster_name="beta")

    runner.read_geometry_progress([farmed, missing, elsewhere])

    # One job is read per turn of the event loop, and each reports when it is done.
    assert pump_until(lambda: len(results) == 3)
    assert results == [
        {farmed.job_id: parse_geometry_progress(fixture)},
        {missing.job_id: None},
        {elsewhere.job_id: None},
    ]
    assert ssh.reads == [("/remote/farmed/seed_farm001.castep", 0)]


def _runner_following_a_large_log(tmp_path: Path):
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "geometry" / "Si2_bfgs.castep"
    runner = ClusterJobRunner()
    ssh = GrowingOutputSSH()
    ssh.remote_files["/remote/job/seed.castep"] = (
        b" filler line of CASTEP output\n" * 200_000 + fixture.read_bytes()
    )
    assert len(ssh.remote_files["/remote/job/seed.castep"]) > 2 * REMOTE_READ_CHUNK_BYTES
    runner._ssh = ssh
    target = CheckTarget(
        job_id=uuid4(),
        remote_job_id="1",
        remote_dir="/remote/job",
        seed_name="seed",
        local_output_dir=tmp_path,
    )
    results: list[dict] = []
    runner.geometry_progress_read.connect(results.append)
    return runner, ssh, target, results, parse_geometry_progress(fixture)


def test_cluster_runner_reads_a_large_log_in_bounded_turns(qapp, tmp_path: Path) -> None:
    runner, ssh, target, results, expected = _runner_following_a_large_log(tmp_path)

    runner.read_geometry_progress([target])

    # The first call read one chunk and reported nothing: the log is not read yet.
    assert ssh.reads == [("/remote/job/seed.castep", 0)]
    assert results == []
    assert pump_until(lambda: results)
    assert results == [{target.job_id: expected}]
    assert ssh.reads == [
        ("/remote/job/seed.castep", offset)
        for offset in range(
            0, len(ssh.remote_files["/remote/job/seed.castep"]), REMOTE_READ_CHUNK_BYTES
        )
    ]


def test_cluster_runner_stops_reading_a_large_log_when_progress_is_forgotten(
    qapp, tmp_path: Path
) -> None:
    runner, ssh, target, results, _expected = _runner_following_a_large_log(tmp_path)

    runner.read_geometry_progress([target])
    runner.forget_progress()

    assert pump_until(lambda: results)
    assert results == [{target.job_id: None}]
    assert len(ssh.reads) == 1


def test_cluster_runner_stops_reading_a_large_log_after_a_disconnect(qapp, tmp_path: Path) -> None:
    runner, ssh, target, results, expected = _runner_following_a_large_log(tmp_path)

    runner.read_geometry_progress([target])
    ssh.is_connected = False
    assert pump_until(lambda: results)
    assert results == [{target.job_id: None}]
    assert len(ssh.reads) == 1

    # After a reconnect the next refresh carries on from where the read stopped.
    ssh.is_connected = True
    runner.read_geometry_progress([target])
    assert pump_until(lambda: len(results) == 2)
    assert results[1] == {target.job_id: expected}
    assert ssh.reads[1] == ("/remote/job/seed.castep", REMOTE_READ_CHUNK_BYTES)


def test_cluster_runner_reports_no_progress_when_a_read_fails_part_way(
    qapp, tmp_path: Path
) -> None:
    runner, ssh, target, results, _expected = _runner_following_a_large_log(tmp_path)
    read_bytes = ssh.read_bytes

    def failing_second_read(remote_path, offset=0, max_bytes=None):
        if offset:
            raise OSError("SFTP read failed")
        return read_bytes(remote_path, offset, max_bytes)

    ssh.read_bytes = failing_second_read

    runner.read_geometry_progress([target])

    assert pump_until(lambda: results)
    assert results == [{target.job_id: None}]


def test_reports_no_geometry_progress_while_disconnected(qapp, tmp_path: Path) -> None:
    runner = ClusterJobRunner()
    reads: list[dict] = []
    runner.geometry_progress_read.connect(reads.append)
    target = _target(tmp_path=tmp_path)

    runner.read_geometry_progress([target])

    assert reads == [{target.job_id: None}]
