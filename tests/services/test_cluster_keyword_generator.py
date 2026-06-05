"""Tests for cluster-backed CASTEP keyword generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mufinder.services.cluster_keyword_generator as generator_module
from mufinder.cluster_setup import ClusterConfig
from mufinder.domain.enums import JobStatus
from mufinder.services.cluster_keyword_generator import (
    generate_cluster_castep_keywords_json,
    validate_cluster_keyword_config,
)


class _DirectSSH:
    def __init__(self, _config, *, settings=None) -> None:
        _ = settings
        self.connected = False
        self.commands: list[str] = []

    def connect(self, password: str | None = None) -> None:
        _ = password
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def exec_command(self, command: str, timeout: int = 30):
        _ = timeout
        self.commands.append(command)
        if "--version" in command:
            return 0, "CASTEP version: 24.1", ""
        if "-help all" in command:
            return 0, "TASK\nSPIN\n", ""
        if "-help TASK" in command:
            return 0, "Type: Integer\nLevel: Basic\n\nTask help text\n\nCELL keywords:\n\nNone found\n", ""
        if "-help SPIN" in command:
            return 0, "Type: Boolean\nLevel: Expert\n\nSpin help text\n\nPARAMETERS keywords:\n\nNone found\n", ""
        return 1, "", "unexpected command"


class _SchedulerSSH:
    def __init__(self, _config, *, settings=None) -> None:
        _ = settings
        self.connected = False
        self.commands: list[tuple[str, object]] = []
        self.remote_files: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.probe_exit_code = 0
        self.probe_stdout = "3.10.12\n"
        self.probe_stderr = ""

    def connect(self, password: str | None = None) -> None:
        _ = password
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def mkdir(self, path: str, exist_ok: bool = True) -> None:
        self.commands.append(("mkdir", (path, exist_ok)))

    def put(self, local_path: Path, remote_path: str) -> None:
        self.commands.append(("put", remote_path))
        self.remote_files[remote_path] = Path(local_path).read_bytes()

    def chmod(self, remote_path: str, mode: int) -> None:
        self.commands.append(("chmod", (remote_path, mode)))

    def exists(self, remote_path: str) -> bool:
        return remote_path in self.remote_files

    def get(self, remote_path: str, local_path: Path) -> None:
        self.commands.append(("get", remote_path))
        local_path.write_bytes(self.remote_files[remote_path])

    def read_text(self, remote_path: str) -> str:
        return self.remote_files.get(remote_path, b"").decode("utf-8")

    def rmtree(self, remote_path: str) -> None:
        self.deleted.append(remote_path)

    def exec_command(self, command: str, timeout: int = 30):
        _ = timeout
        self.commands.append(("exec", command))
        if "version_info" in command:
            return self.probe_exit_code, self.probe_stdout, self.probe_stderr
        return 0, "", ""


class _FakeScheduler:
    def __init__(self) -> None:
        self.submit_requests: list[object] = []
        self._status_calls = 0

    def submit(self, ssh: _SchedulerSSH, request) -> str:
        self.submit_requests.append(request)
        remote_json = f"{request.working_dir}/castep_keywords.json"
        ssh.remote_files[remote_json] = json.dumps(
            {
                "cell": {},
                "param": {"task": {"keyword": "task", "option_type": "Integer", "level": "Basic", "docstring": "Task help"}},
                "types": ["Integer"],
                "levels": ["Basic"],
                "castep_version": 24.1,
            }
        ).encode("utf-8")
        return "99123"

    def status(self, _ssh, _job_ids: list[str]):
        self._status_calls += 1
        if self._status_calls == 1:
            return {"99123": type("Info", (), {"job_id": "99123", "state": "R"})()}
        return {}

    def cancel(self, _ssh, _job_id: str) -> None:
        return

    def map_to_job_status(self, state: str) -> JobStatus:
        return {"PD": JobStatus.QUEUED, "R": JobStatus.RUNNING, "CD": JobStatus.COMPLETED}[state]


class _NoOutputScheduler(_FakeScheduler):
    def submit(self, ssh: _SchedulerSSH, request) -> str:
        self.submit_requests.append(request)
        remote_log = f"{request.working_dir}/keyword_generation.log"
        noisy_progress = "".join(f"Processed: [{'#' * ((i % 20) + 1):<20}] {i:>3d}%\n" for i in range(1, 60))
        ssh.remote_files[remote_log] = (
            noisy_progress
            + "Traceback (most recent call last):\n"
            + "TypeError: unsupported operand type(s) for |: 'type' and 'type'\n"
        ).encode("utf-8")
        return "99124"

    def status(self, _ssh, _job_ids: list[str]):
        return {}


def test_validate_cluster_keyword_config_requires_scheduler_fields() -> None:
    config = ClusterConfig(
        name="SCARF",
        hostname="ui1.scarf.rl.ac.uk",
        username="alice",
        auth_method="key",
        default_working_directory="",
        queue_system="SLURM",
        template_script_path="builtin:scarf-slurm",
        keyword_execution_mode="scheduler",
        keyword_shell_init="",
        keyword_python_command="",
    )

    issues = validate_cluster_keyword_config(config)

    assert "Default working directory is required for scheduler keyword generation." in issues
    assert "Keyword Python command is required for scheduler keyword generation." in issues
    assert any("Keyword shell init is required" in issue for issue in issues)


def test_generate_cluster_castep_keywords_json_direct_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generator_module, "SSHConnection", _DirectSSH)

    config = ClusterConfig(
        name="MyCluster",
        hostname="example.cluster.ac.uk",
        username="alice",
        auth_method="key",
        default_working_directory="/work/alice",
        queue_system="SLURM",
        castep_executable="castep.mpi",
        keyword_execution_mode="direct",
        keyword_shell_init="module load CASTEP",
    )

    output_path = generate_cluster_castep_keywords_json(
        config=config,
        output_path=tmp_path / "castep_keywords.json",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["castep_version"] == 24.1
    assert "task" in payload["param"]
    assert "spin" in payload["cell"]


def test_generate_cluster_castep_keywords_json_addqueue_uses_scheduler_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_scheduler = _FakeScheduler()
    ssh_instances: list[_SchedulerSSH] = []

    def make_ssh(config, *, settings=None) -> _SchedulerSSH:
        ssh = _SchedulerSSH(config, settings=settings)
        ssh_instances.append(ssh)
        return ssh

    monkeypatch.setattr(generator_module, "SSHConnection", make_ssh)
    monkeypatch.setattr(generator_module, "create_scheduler", lambda *_args, **_kwargs: fake_scheduler)
    monkeypatch.setattr(generator_module.time, "sleep", lambda _seconds: None)

    config = ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="alice",
        auth_method="key",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="ADDQUEUE",
        default_walltime="",
        castep_executable="/mnt/redwoodfs/muon/castep.mpi",
        keyword_execution_mode="direct",
        keyword_python_command="python3",
    )

    output_path = generate_cluster_castep_keywords_json(
        config=config,
        output_path=tmp_path / "scheduled_redwood_keywords.json",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["castep_version"] == 24.1
    assert fake_scheduler.submit_requests
    request = fake_scheduler.submit_requests[0]
    assert request.castep_executable == "/bin/bash"
    assert request.seed_name.endswith("run_keywords.sh")
    assert request.walltime == ""
    uploaded_runner = next(
        remote_path for action, remote_path in ssh_instances[0].commands if action == "put" and str(remote_path).endswith("run_keywords.sh")
    )
    runner_text = ssh_instances[0].remote_files[uploaded_runner].decode("utf-8")
    assert "export MUFINDER_CASTEP_LAUNCHER='mpirun -np 1'" in runner_text
    assert ssh_instances[0].deleted


def test_generate_cluster_castep_keywords_json_scheduler_mode_downloads_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_scheduler = _FakeScheduler()
    ssh_instances: list[_SchedulerSSH] = []

    def make_ssh(config, *, settings=None) -> _SchedulerSSH:
        ssh = _SchedulerSSH(config, settings=settings)
        ssh_instances.append(ssh)
        return ssh

    monkeypatch.setattr(generator_module, "SSHConnection", make_ssh)
    monkeypatch.setattr(generator_module, "create_scheduler", lambda *_args, **_kwargs: fake_scheduler)
    monkeypatch.setattr(generator_module.time, "sleep", lambda _seconds: None)

    config = ClusterConfig(
        name="SCARF",
        hostname="ui1.scarf.rl.ac.uk",
        username="alice",
        auth_method="key",
        default_working_directory="/work4/isis/alice",
        queue_system="SLURM",
        template_script_path="builtin:scarf-slurm",
        castep_executable="castep.mpi",
        keyword_execution_mode="scheduler",
        keyword_shell_init="module load CASTEP",
        keyword_python_command="python3",
    )

    output_path = generate_cluster_castep_keywords_json(
        config=config,
        output_path=tmp_path / "scheduled_keywords.json",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["castep_version"] == 24.1
    assert fake_scheduler.submit_requests
    assert ssh_instances
    ssh = ssh_instances[0]
    uploaded_runner = next(
        remote_path for action, remote_path in ssh.commands if action == "put" and str(remote_path).endswith("run_keywords.sh")
    )
    uploaded_generator = next(
        remote_path for action, remote_path in ssh.commands if action == "put" and str(remote_path).endswith("generate_keywords.py")
    )
    assert b"\r\n" not in ssh.remote_files[uploaded_runner]
    assert b"\r\n" not in ssh.remote_files[uploaded_generator]
    assert "module load CASTEP" in ssh.remote_files[uploaded_runner].decode("utf-8")
    generator_text = ssh.remote_files[uploaded_generator].decode("utf-8")
    assert "from __future__ import annotations" in generator_text
    assert "def detect_version(executable):" in generator_text
    assert "->" not in generator_text
    assert ssh.deleted


def test_generate_cluster_castep_keywords_json_scheduler_mode_fails_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ssh_instances: list[_SchedulerSSH] = []

    def make_ssh(config, *, settings=None) -> _SchedulerSSH:
        ssh = _SchedulerSSH(config, settings=settings)
        ssh.probe_exit_code = 1
        ssh.probe_stdout = ""
        ssh.probe_stderr = "python3: command not found"
        ssh_instances.append(ssh)
        return ssh

    monkeypatch.setattr(generator_module, "SSHConnection", make_ssh)

    config = ClusterConfig(
        name="SCARF",
        hostname="ui1.scarf.rl.ac.uk",
        username="alice",
        auth_method="key",
        default_working_directory="/work4/isis/alice",
        queue_system="SLURM",
        template_script_path="builtin:scarf-slurm",
        castep_executable="castep.mpi",
        keyword_execution_mode="scheduler",
        keyword_shell_init="module load CASTEP",
        keyword_python_command="python3",
    )

    with pytest.raises(RuntimeError, match="Keyword Python preflight failed"):
        generate_cluster_castep_keywords_json(
            config=config,
            output_path=tmp_path / "scheduled_keywords.json",
        )


def test_scheduler_mode_missing_json_surfaces_compatibility_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = _NoOutputScheduler()

    monkeypatch.setattr(generator_module, "SSHConnection", _SchedulerSSH)
    monkeypatch.setattr(generator_module, "create_scheduler", lambda *_args, **_kwargs: scheduler)
    monkeypatch.setattr(generator_module.time, "sleep", lambda _seconds: None)

    config = ClusterConfig(
        name="SCARF",
        hostname="ui1.scarf.rl.ac.uk",
        username="alice",
        auth_method="key",
        default_working_directory="/work4/isis/alice",
        queue_system="SLURM",
        template_script_path="builtin:scarf-slurm",
        castep_executable="castep.mpi",
        keyword_execution_mode="scheduler",
        keyword_shell_init="module load CASTEP",
        keyword_python_command="python3",
    )

    with pytest.raises(RuntimeError) as exc_info:
        generate_cluster_castep_keywords_json(
            config=config,
            output_path=tmp_path / "scheduled_keywords.json",
        )

    message = str(exc_info.value)
    assert "Compatibility hint:" in message
    assert "keyword_python_command" in message
    assert "TypeError: unsupported operand type(s) for |: 'type' and 'type'" in message
    assert "Processed:" not in message
