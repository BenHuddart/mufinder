"""Additional coverage for cluster keyword generator branches and error paths.

These tests target currently-uncovered code paths in
``mufinder.services.cluster_keyword_generator``:

* ``validate_cluster_keyword_config`` missing-field branches and the
  direct-mode (no-issues) path.
* ``generate_cluster_castep_keywords_json`` input validation guards
  (validation issues, non-``.json`` output path).
* The scheduler status polling loop branches (QUEUED, RUNNING, CANCELLED)
  and the cancel-during-loop path that cancels the remote job.
* The direct-mode cancel-after-payload path.
* ``_build_remote_keyword_payload`` empty-options error.
* ``_ensure_scheduler_python_compatible`` unparseable / too-old version
  errors.
* The ``ClusterKeywordGenerationWorker`` Qt object (run success / cancelled
  / failed, cancel flag, progress forwarding) without a running event loop.

All tests are deterministic: no network, no real SSH, no sleeps, no GUI
windows, no randomness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mufinder.services.cluster_keyword_generator as generator_module
from mufinder.cluster_setup import ClusterConfig
from mufinder.domain.enums import JobStatus
from mufinder.services.cluster_keyword_generator import (
    ClusterKeywordGenerationCancelled,
    ClusterKeywordGenerationError,
    ClusterKeywordGenerationWorker,
    generate_cluster_castep_keywords_json,
    validate_cluster_keyword_config,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _DirectSSH:
    """Minimal SSH double for direct-mode keyword generation."""

    def __init__(self, _config, *, settings=None) -> None:
        _ = settings
        self.connected = False
        self.disconnected = False

    def connect(self, password: str | None = None) -> None:
        _ = password
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def exec_command(self, command: str, timeout: int = 30):
        _ = timeout
        if "--version" in command:
            return 0, "CASTEP version: 24.1", ""
        if "-help all" in command:
            return 0, "TASK\n", ""
        if "-help TASK" in command:
            return (
                0,
                "Type: Integer\nLevel: Basic\n\nTask help text\n\nCELL keywords:\n\nNone found\n",
                "",
            )
        return 1, "", "unexpected command"


class _SchedulerSSH:
    """SSH double supporting the scheduler keyword-generation workflow."""

    def __init__(self, _config, *, settings=None) -> None:
        _ = settings
        self.connected = False
        self.disconnected = False
        self.commands: list[tuple[str, object]] = []
        self.remote_files: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.probe_exit_code = 0
        self.probe_stdout = "3.11.7\n"
        self.probe_stderr = ""

    def connect(self, password: str | None = None) -> None:
        _ = password
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

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


class _Info:
    def __init__(self, state: str) -> None:
        self.job_id = "42"
        self.state = state


class _StateSequenceScheduler:
    """Scheduler whose status() walks a predefined sequence of states.

    The mapping from raw state -> JobStatus is the identity on JobStatus
    values, so tests can drive specific loop branches directly.
    """

    def __init__(self, states: list[JobStatus], *, write_json: bool = True) -> None:
        self._states = list(states)
        self._write_json = write_json
        self.submit_requests: list[object] = []
        self.cancelled_jobs: list[str] = []

    def submit(self, ssh: _SchedulerSSH, request) -> str:
        self.submit_requests.append(request)
        if self._write_json:
            remote_json = f"{request.working_dir}/castep_keywords.json"
            ssh.remote_files[remote_json] = json.dumps(
                {
                    "cell": {},
                    "param": {"task": {"keyword": "task"}},
                    "types": ["Integer"],
                    "levels": ["Basic"],
                    "castep_version": 24.1,
                }
            ).encode("utf-8")
        return "42"

    def status(self, _ssh, _job_ids: list[str]):
        if self._states:
            state = self._states.pop(0)
        else:
            state = JobStatus.COMPLETED
        return {"42": _Info(state)}

    def cancel(self, _ssh, job_id: str) -> None:
        self.cancelled_jobs.append(job_id)

    def map_to_job_status(self, state) -> JobStatus:
        return state


def _scheduler_config(**overrides) -> ClusterConfig:
    base = dict(
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
    base.update(overrides)
    return ClusterConfig(**base)


# ---------------------------------------------------------------------------
# validate_cluster_keyword_config branches
# ---------------------------------------------------------------------------


def test_validate_reports_missing_core_fields() -> None:
    config = ClusterConfig(
        name="Blank",
        hostname="",
        username="",
        auth_method="key",
        castep_executable="",
        keyword_execution_mode="direct",
    )

    issues = validate_cluster_keyword_config(config)

    assert "Cluster hostname is required." in issues
    assert "Cluster username is required." in issues
    assert "CASTEP executable is required." in issues


def test_validate_direct_mode_complete_config_has_no_issues() -> None:
    config = ClusterConfig(
        name="Direct",
        hostname="host",
        username="alice",
        auth_method="key",
        castep_executable="castep.mpi",
        keyword_execution_mode="direct",
    )

    assert validate_cluster_keyword_config(config) == []


def test_validate_scheduler_module_executable_without_shell_init() -> None:
    # Module-provided executable (no "/") requires keyword_shell_init.
    config = _scheduler_config(
        default_working_directory="/work/alice",
        castep_executable="castep.mpi",
        keyword_shell_init="",
        keyword_python_command="python3",
    )

    issues = validate_cluster_keyword_config(config)

    assert any("Keyword shell init is required" in issue for issue in issues)


# ---------------------------------------------------------------------------
# generate_cluster_castep_keywords_json input guards
# ---------------------------------------------------------------------------


def test_generate_raises_on_validation_issues(tmp_path: Path) -> None:
    config = ClusterConfig(
        name="Blank",
        hostname="",
        username="",
        auth_method="key",
        castep_executable="",
        keyword_execution_mode="direct",
    )

    with pytest.raises(ClusterKeywordGenerationError) as exc_info:
        generate_cluster_castep_keywords_json(
            config=config,
            output_path=tmp_path / "keywords.json",
        )

    message = str(exc_info.value)
    assert "Cluster hostname is required." in message
    assert message.startswith("- ")


def test_generate_rejects_non_json_output_path(tmp_path: Path) -> None:
    config = ClusterConfig(
        name="Direct",
        hostname="host",
        username="alice",
        auth_method="key",
        castep_executable="castep.mpi",
        keyword_execution_mode="direct",
    )

    with pytest.raises(ClusterKeywordGenerationError, match="must end with .json"):
        generate_cluster_castep_keywords_json(
            config=config,
            output_path=tmp_path / "keywords.txt",
        )


def test_generate_direct_mode_disconnects_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instances: list[_DirectSSH] = []

    def make_ssh(config, *, settings=None) -> _DirectSSH:
        ssh = _DirectSSH(config, settings=settings)
        instances.append(ssh)
        return ssh

    monkeypatch.setattr(generator_module, "SSHConnection", make_ssh)

    config = ClusterConfig(
        name="Direct",
        hostname="host",
        username="alice",
        auth_method="key",
        default_working_directory="/work/alice",
        queue_system="SLURM",
        castep_executable="/opt/castep/castep.mpi",
        keyword_execution_mode="direct",
    )

    updates: list[tuple[str, int | None]] = []
    output_path = generate_cluster_castep_keywords_json(
        config=config,
        output_path=tmp_path / "keywords.json",
        progress=lambda msg, pct: updates.append((msg, pct)),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    # The TASK doc reports "CELL keywords: None found" -> classified as a param.
    assert "task" in payload["param"]
    assert instances and instances[0].disconnected is True
    # Progress callback was used (the 100% write step at minimum).
    assert any(pct == 100 for _msg, pct in updates)


# ---------------------------------------------------------------------------
# _build_remote_keyword_payload error path
# ---------------------------------------------------------------------------


def test_build_remote_keyword_payload_raises_without_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ClusterConfig(
        name="Direct",
        hostname="host",
        username="alice",
        auth_method="key",
        castep_executable="castep.mpi",
        keyword_execution_mode="direct",
    )

    monkeypatch.setattr(
        generator_module,
        "_run_remote_checked",
        lambda *_args, **_kwargs: "no recognisable keywords here",
    )

    with pytest.raises(ClusterKeywordGenerationError, match="did not contain any keywords"):
        generator_module._build_remote_keyword_payload(object(), config)


# ---------------------------------------------------------------------------
# direct-mode cancellation
# ---------------------------------------------------------------------------


def test_generate_direct_mode_cancel_before_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generator_module, "SSHConnection", _DirectSSH)

    config = ClusterConfig(
        name="Direct",
        hostname="host",
        username="alice",
        auth_method="key",
        castep_executable="/opt/castep/castep.mpi",
        keyword_execution_mode="direct",
    )

    with pytest.raises(ClusterKeywordGenerationCancelled):
        generate_cluster_castep_keywords_json(
            config=config,
            output_path=tmp_path / "keywords.json",
            cancel_check=lambda: True,
        )


# ---------------------------------------------------------------------------
# scheduler status-loop branches
# ---------------------------------------------------------------------------


def _patch_scheduler(monkeypatch: pytest.MonkeyPatch, scheduler) -> list[_SchedulerSSH]:
    instances: list[_SchedulerSSH] = []

    def make_ssh(config, *, settings=None) -> _SchedulerSSH:
        ssh = _SchedulerSSH(config, settings=settings)
        instances.append(ssh)
        return ssh

    monkeypatch.setattr(generator_module, "SSHConnection", make_ssh)
    monkeypatch.setattr(generator_module, "create_scheduler", lambda *_a, **_k: scheduler)
    monkeypatch.setattr(generator_module.time, "sleep", lambda _s: None)
    return instances


def test_scheduler_loop_walks_queued_running_then_completed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = _StateSequenceScheduler(
        [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED]
    )
    _patch_scheduler(monkeypatch, scheduler)

    updates: list[tuple[str, int | None]] = []
    output_path = generate_cluster_castep_keywords_json(
        config=_scheduler_config(),
        output_path=tmp_path / "keywords.json",
        progress=lambda msg, pct: updates.append((msg, pct)),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "task" in payload["param"]
    messages = [msg for msg, _ in updates]
    assert any("queued" in msg for msg in messages)
    assert any("running" in msg for msg in messages)


def test_scheduler_loop_cancelled_state_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = _StateSequenceScheduler([JobStatus.CANCELLED], write_json=False)
    _patch_scheduler(monkeypatch, scheduler)

    with pytest.raises(ClusterKeywordGenerationCancelled, match="was cancelled"):
        generate_cluster_castep_keywords_json(
            config=_scheduler_config(),
            output_path=tmp_path / "keywords.json",
        )


def test_scheduler_loop_failed_state_raises_with_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = _StateSequenceScheduler([JobStatus.FAILED], write_json=False)
    instances = _patch_scheduler(monkeypatch, scheduler)

    # Pre-seed a log so the failure message carries an excerpt. The log path is
    # derived from the remote directory created during submission, so patch the
    # scheduler to drop a log next to the runner.
    original_submit = scheduler.submit

    def submit_with_log(ssh, request):
        job_id = original_submit(ssh, request)
        ssh.remote_files[f"{request.working_dir}/keyword_generation.log"] = (
            b"Traceback (most recent call last):\nRuntimeError: boom\n"
        )
        return job_id

    scheduler.submit = submit_with_log  # type: ignore[assignment]

    with pytest.raises(ClusterKeywordGenerationError) as exc_info:
        generate_cluster_castep_keywords_json(
            config=_scheduler_config(),
            output_path=tmp_path / "keywords.json",
        )

    message = str(exc_info.value)
    assert "failed on SCARF" in message
    assert "RuntimeError: boom" in message
    assert instances and instances[0].deleted  # cleanup ran in finally


def test_scheduler_cancel_during_loop_cancels_remote_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = _StateSequenceScheduler(
        [JobStatus.RUNNING, JobStatus.RUNNING], write_json=False
    )
    instances = _patch_scheduler(monkeypatch, scheduler)

    # Cancel only after the job has been submitted (a real job id exists), so
    # the cancellation handler invokes scheduler.cancel.
    calls = {"n": 0}

    def cancel_check() -> bool:
        calls["n"] += 1
        # Allow submission + first status check, then request cancellation.
        return calls["n"] > 2

    with pytest.raises(ClusterKeywordGenerationCancelled):
        generate_cluster_castep_keywords_json(
            config=_scheduler_config(),
            output_path=tmp_path / "keywords.json",
            cancel_check=cancel_check,
        )

    assert scheduler.cancelled_jobs == ["42"]
    assert instances and instances[0].deleted


def test_scheduler_status_none_treated_as_completed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _EmptyStatusScheduler(_StateSequenceScheduler):
        def status(self, _ssh, _job_ids):
            return {}  # info is None -> mapped to COMPLETED

    scheduler = _EmptyStatusScheduler([])
    _patch_scheduler(monkeypatch, scheduler)

    output_path = generate_cluster_castep_keywords_json(
        config=_scheduler_config(),
        output_path=tmp_path / "keywords.json",
    )

    assert json.loads(output_path.read_text(encoding="utf-8"))["castep_version"] == 24.1


# ---------------------------------------------------------------------------
# _ensure_scheduler_python_compatible error branches
# ---------------------------------------------------------------------------


def test_python_preflight_unparseable_version_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _scheduler_config()
    ssh = _SchedulerSSH(config)
    ssh.probe_exit_code = 0
    ssh.probe_stdout = "not a version string"
    ssh.probe_stderr = ""

    with pytest.raises(ClusterKeywordGenerationError, match="could not parse a version"):
        generator_module._ensure_scheduler_python_compatible(ssh, config)


def test_python_preflight_too_old_version_raises() -> None:
    config = _scheduler_config()
    ssh = _SchedulerSSH(config)
    ssh.probe_exit_code = 0
    ssh.probe_stdout = "2.7.18"
    ssh.probe_stderr = ""

    with pytest.raises(ClusterKeywordGenerationError, match="requires Python >= 3.6"):
        generator_module._ensure_scheduler_python_compatible(ssh, config)


def test_python_preflight_accepts_supported_version() -> None:
    config = _scheduler_config()
    ssh = _SchedulerSSH(config)
    ssh.probe_stdout = "3.11.7\n"

    # Should not raise.
    generator_module._ensure_scheduler_python_compatible(ssh, config)


# ---------------------------------------------------------------------------
# ClusterKeywordGenerationWorker (no event loop required)
# ---------------------------------------------------------------------------


def test_worker_run_success_emits_finished(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "result.json"
    output_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        generator_module,
        "generate_cluster_castep_keywords_json",
        lambda **_kwargs: output_file,
    )

    config = ClusterConfig(
        name="W", hostname="h", username="u", auth_method="key",
        castep_executable="castep.mpi", keyword_execution_mode="direct",
    )
    worker = ClusterKeywordGenerationWorker(config=config, output_path=output_file)

    finished: list[str] = []
    failed: list[str] = []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)

    worker.run()

    assert finished == [str(output_file)]
    assert failed == []


def test_worker_run_cancelled_emits_cancelled_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(**_kwargs):
        raise ClusterKeywordGenerationCancelled("stop")

    monkeypatch.setattr(generator_module, "generate_cluster_castep_keywords_json", boom)

    config = ClusterConfig(
        name="W", hostname="h", username="u", auth_method="key",
        castep_executable="castep.mpi", keyword_execution_mode="direct",
    )
    worker = ClusterKeywordGenerationWorker(config=config, output_path="x.json")
    worker.cancel()
    assert worker._cancel_requested is True

    cancelled: list[bool] = []
    failed: list[str] = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.failed.connect(failed.append)

    worker.run()

    assert cancelled == [True]
    assert failed == []


def test_worker_run_failure_emits_failed_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(**_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(generator_module, "generate_cluster_castep_keywords_json", boom)

    config = ClusterConfig(
        name="W", hostname="h", username="u", auth_method="key",
        castep_executable="castep.mpi", keyword_execution_mode="direct",
    )
    worker = ClusterKeywordGenerationWorker(config=config, output_path="x.json")

    failed: list[str] = []
    worker.failed.connect(failed.append)

    worker.run()

    assert failed == ["network down"]


def test_worker_on_progress_forwards_to_signal() -> None:
    config = ClusterConfig(
        name="W", hostname="h", username="u", auth_method="key",
        castep_executable="castep.mpi", keyword_execution_mode="direct",
    )
    worker = ClusterKeywordGenerationWorker(config=config, output_path="x.json")

    updates: list[tuple[str, object]] = []
    worker.progress_changed.connect(lambda msg, pct: updates.append((msg, pct)))

    worker._on_progress("working", 55)
    worker._on_progress("done", None)

    assert updates == [("working", 55), ("done", None)]
