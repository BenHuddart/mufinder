"""Tests for cluster-backed CASTEP keyword generation.

The generator runs every ``castep -help`` call from a shell loop on the cluster
and parses the marked-up output locally. Most tests drive it through an SSH
double that answers the loop the way a cluster would; two run the generated
shell for real against a stand-in ``castep`` script, because the quoting and the
marker format are only proven by an actual ``bash``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest

import mufinder.services.cluster_keyword_generator as generator_module
import mufinder.services.cluster_keyword_scripts as scripts_module
from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import JobStatus
from mufinder.services.cluster_keyword_generator import (
    ClusterKeywordGenerationCancelled,
    ClusterKeywordGenerationError,
    ClusterKeywordGenerationWorker,
    generate_cluster_castep_keywords_json,
    validate_cluster_keyword_config,
)
from tests.support.keyword_catalogs import (
    complete_keyword_names,
    help_all_text,
    help_keyword_text,
)


_ALL_NAMES = [name.upper() for names in complete_keyword_names().values() for name in names]
_LOOP_NAMES_RE = re.compile(r"^set -- (?P<names>[A-Z_\d ]+)$", re.MULTILINE)


def _marked_help(names: list[str], *, failing: frozenset[str] = frozenset()) -> str:
    """Return what the remote help loop prints for ``names``."""

    chunks = []
    for name in names:
        if name in failing:
            body, status = "mpirun: could not start castep.mpi", 1
        else:
            body, status = help_keyword_text(name), 0
        chunks.append(f"@@MUFINDER_HELP_BEGIN {name}@@\n{body}\n@@MUFINDER_HELP_END {name} {status}@@\n")
    return "".join(chunks)


def _scheduler_dump(names: list[str], *, complete: bool = True, version: str = "26.11") -> bytes:
    text = (
        f"@@MUFINDER_VERSION_BEGIN@@\n CASTEP version: {version} \n\n@@MUFINDER_VERSION_END@@\n"
        + _marked_help(names)
        + ("@@MUFINDER_DUMP_COMPLETE@@\n" if complete else "")
    )
    return text.encode("utf-8")


class _DirectSSH:
    """SSH double that answers the direct-mode commands like a cluster."""

    failing: frozenset[str] = frozenset()
    help_all: str = help_all_text()

    def __init__(self, _config, *, settings=None) -> None:
        _ = settings
        self.connected = False
        self.disconnected = False
        self.commands: list[str] = []
        self.timeouts: list[int] = []
        _DirectSSH.last = self

    def connect(self, password: str | None = None) -> None:
        _ = password
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def exec_command(self, command: str, timeout: int = 30):
        self.commands.append(command)
        self.timeouts.append(timeout)
        loop = _LOOP_NAMES_RE.search(command)
        if loop is not None:
            return 0, _marked_help(loop.group("names").split(), failing=self.failing), ""
        if "--version" in command:
            return 0, " CASTEP version: 24.10 \n", ""
        if "-help all" in command:
            return 0, self.help_all, ""
        return 1, "", "unexpected command"


class _SchedulerSSH:
    def __init__(self, _config, *, settings=None) -> None:
        _ = settings
        self.connected = False
        self.disconnected = False
        self.commands: list[tuple[str, object]] = []
        self.remote_files: dict[str, bytes] = {}
        self.deleted: list[str] = []
        _SchedulerSSH.last = self

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

    def read_text(self, remote_path: str) -> str:
        return self.remote_files.get(remote_path, b"").decode("utf-8")

    def listdir(self, remote_path: str) -> list[str]:
        prefix = remote_path.rstrip("/") + "/"
        return [path[len(prefix) :] for path in self.remote_files if path.startswith(prefix)]

    def rmtree(self, remote_path: str) -> None:
        self.deleted.append(remote_path)

    def exec_command(self, command: str, timeout: int = 30):
        _ = timeout
        self.commands.append(("exec", command))
        if command.startswith("grep -c "):
            return self.grep_result
        return 0, "", ""

    #: What counting the dump's end markers returns while the job runs.
    grep_result: tuple[int, str, str] = (0, "137\n", "")


class _Info:
    def __init__(self, state: JobStatus) -> None:
        self.job_id = "42"
        self.state = state


class _Scheduler:
    """Scheduler double whose ``status`` walks a list of job states."""

    def __init__(
        self,
        states: list[JobStatus] | None = None,
        *,
        dump: bytes | None = None,
        log: bytes | None = None,
    ) -> None:
        self._states = list(states or [])
        self._dump = dump
        self._log = log
        self.submit_requests: list[object] = []
        self.cancelled_jobs: list[str] = []
        self.status_calls = 0

    def submit(self, ssh: _SchedulerSSH, request) -> str:
        self.submit_requests.append(request)
        if self._dump is not None:
            ssh.remote_files[f"{request.working_dir}/keyword_help.txt"] = self._dump
        if self._log is not None:
            ssh.remote_files[f"{request.working_dir}/keyword_generation.log"] = self._log
        return "42"

    def status(self, _ssh, _job_ids: list[str]):
        self.status_calls += 1
        if not self._states:
            # A finished job drops out of the queue listing altogether.
            return {}
        return {"42": _Info(self._states.pop(0))}

    def cancel(self, _ssh, job_id: str) -> None:
        self.cancelled_jobs.append(job_id)

    def map_to_job_status(self, state: JobStatus) -> JobStatus:
        return state


def _direct_config(**overrides) -> ClusterConfig:
    base = dict(
        name="Redwood",
        hostname="redwood.physics.ox.ac.uk",
        username="alice",
        auth_method="password",
        castep_executable="castep.mpi",
        keyword_execution_mode="direct",
    )
    base.update(overrides)
    return ClusterConfig(**base)


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
    )
    base.update(overrides)
    return ClusterConfig(**base)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(generator_module.time, "sleep", slept.append)
    return slept


def _use_scheduler(monkeypatch: pytest.MonkeyPatch, scheduler: _Scheduler) -> None:
    monkeypatch.setattr(generator_module, "SSHConnection", _SchedulerSSH)
    monkeypatch.setattr(generator_module, "create_scheduler", lambda *_args, **_kwargs: scheduler)


# --- configuration validation -------------------------------------------------


def test_validate_reports_missing_core_fields() -> None:
    issues = validate_cluster_keyword_config(ClusterConfig(name="Empty", hostname="", username="", castep_executable=""))

    assert "Cluster hostname is required." in issues
    assert "Cluster username is required." in issues
    assert "CASTEP executable is required." in issues


def test_validate_direct_mode_complete_config_has_no_issues() -> None:
    assert validate_cluster_keyword_config(_direct_config()) == []


def test_validate_scheduler_mode_needs_only_a_working_directory() -> None:
    """No Python interpreter and no separate shell init: the job template supplies the setup."""

    assert validate_cluster_keyword_config(_scheduler_config(keyword_shell_init="")) == []
    assert validate_cluster_keyword_config(_scheduler_config(default_working_directory="")) == [
        "Default working directory is required for scheduler keyword generation."
    ]
    assert not hasattr(ClusterConfig, "keyword_python_command")
    # A cluster that has never chosen a mode generates through the scheduler.
    assert ClusterConfig(name="New", hostname="h").keyword_execution_mode == "scheduler"


def test_generate_raises_on_validation_issues(tmp_path: Path) -> None:
    with pytest.raises(ClusterKeywordGenerationError, match="CASTEP executable is required"):
        generate_cluster_castep_keywords_json(
            config=_direct_config(castep_executable=""),
            output_path=tmp_path / "castep_keywords.json",
        )


def test_generate_rejects_non_json_output_path(tmp_path: Path) -> None:
    with pytest.raises(ClusterKeywordGenerationError, match="must end with .json"):
        generate_cluster_castep_keywords_json(config=_direct_config(), output_path=tmp_path / "keywords.txt")


# --- direct mode --------------------------------------------------------------


def test_direct_mode_batches_help_calls_into_a_few_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generator_module, "SSHConnection", _DirectSSH)
    monkeypatch.setattr(_DirectSSH, "failing", frozenset())
    output_path = tmp_path / "nested" / "castep_keywords.json"
    progress: list[tuple[str, int | None]] = []

    written = generate_cluster_castep_keywords_json(
        config=_direct_config(),
        output_path=output_path,
        password="secret",
        progress=lambda message, percent: progress.append((message, percent)),
    )

    ssh = _DirectSSH.last
    names = complete_keyword_names()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == output_path.resolve()
    assert set(payload["cell"]) == set(names["cell"])
    assert set(payload["param"]) == set(names["param"])
    assert payload["param"]["task"] == {
        "keyword": "task",
        "option_type": "String",
        "level": "Basic",
        "docstring": "Help for task.",
    }
    # The version is kept as text: a float would turn 24.10 into 24.1.
    assert payload["castep_version"] == "24.10"
    assert payload["types"] == ["String"]
    assert payload["levels"] == ["Basic"]

    # 211 keywords: one listing, one version query, and five batches of 50,
    # where the per-keyword implementation needed 213 round trips.
    loop_commands = [command for command in ssh.commands if _LOOP_NAMES_RE.search(command)]
    assert len(_ALL_NAMES) == 211
    assert len(loop_commands) == 5
    assert len(ssh.commands) == 7
    assert all(command.startswith("/bin/bash -lc ") for command in ssh.commands)
    assert ssh.disconnected is True
    assert progress[0] == ("Connecting to Redwood...", 0)
    assert ("Generating CASTEP keywords... 211 of 211", 99) in progress
    assert progress[-1] == ("Writing CASTEP keywords to castep_keywords.json...", 100)
    # No temporary file is left beside the catalog.
    assert [path.name for path in output_path.parent.iterdir()] == ["castep_keywords.json"]


def test_direct_mode_applies_launcher_and_shell_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator_module, "SSHConnection", _DirectSSH)
    monkeypatch.setattr(_DirectSSH, "failing", frozenset())
    monkeypatch.setattr(generator_module, "_effective_keyword_mode", lambda _config: "direct")

    generate_cluster_castep_keywords_json(
        config=_direct_config(queue_system="addqueue", keyword_shell_init="module load CASTEP"),
        output_path=tmp_path / "castep_keywords.json",
    )

    loop_command = next(command for command in _DirectSSH.last.commands if _LOOP_NAMES_RE.search(command))
    assert "module load CASTEP" in loop_command
    assert 'mpirun -np 1 castep.mpi -help "$keyword" > "$mufinder_scratch/$keyword.out" 2>&1 || status=$?' in loop_command


def test_direct_mode_refuses_to_write_an_incomplete_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that lost most of its -help calls must not replace a good catalog."""

    monkeypatch.setattr(generator_module, "SSHConnection", _DirectSSH)
    monkeypatch.setattr(_DirectSSH, "failing", frozenset(name for name in _ALL_NAMES if "FILLER" in name))
    output_path = tmp_path / "castep_keywords.json"
    output_path.write_text("existing catalog", encoding="utf-8")

    with pytest.raises(ClusterKeywordGenerationError) as excinfo:
        generate_cluster_castep_keywords_json(config=_direct_config(), output_path=output_path)

    message = str(excinfo.value)
    assert "CASTEP on Redwood described only 6 of 211 keywords" in message
    assert "First failing query:\ncastep -help CELL_FILLER_0:\nmpirun: could not start castep.mpi" in message
    assert output_path.read_text(encoding="utf-8") == "existing catalog"
    assert _DirectSSH.last.disconnected is True


def test_direct_mode_tolerates_a_few_failed_help_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator_module, "SSHConnection", _DirectSSH)
    monkeypatch.setattr(_DirectSSH, "failing", frozenset({"CELL_FILLER_3", "PARAM_FILLER_7"}))
    output_path = tmp_path / "castep_keywords.json"

    generate_cluster_castep_keywords_json(config=_direct_config(), output_path=output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "cell_filler_3" not in payload["cell"]
    assert "param_filler_7" not in payload["param"]
    assert len(payload["cell"]) + len(payload["param"]) == 209


def test_direct_mode_raises_when_castep_lists_no_keywords(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator_module, "SSHConnection", _DirectSSH)
    monkeypatch.setattr(_DirectSSH, "help_all", "no keywords here\n")

    with pytest.raises(ClusterKeywordGenerationError, match="did not contain any keywords"):
        generate_cluster_castep_keywords_json(config=_direct_config(), output_path=tmp_path / "castep_keywords.json")


def test_direct_mode_surfaces_remote_command_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenSSH(_DirectSSH):
        def exec_command(self, command: str, timeout: int = 30):
            return 127, "", "castep.mpi: command not found"

    monkeypatch.setattr(generator_module, "SSHConnection", _BrokenSSH)

    with pytest.raises(ClusterKeywordGenerationError, match="castep.mpi: command not found"):
        generate_cluster_castep_keywords_json(config=_direct_config(), output_path=tmp_path / "castep_keywords.json")


def test_direct_mode_cancel_between_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator_module, "SSHConnection", _DirectSSH)
    monkeypatch.setattr(_DirectSSH, "failing", frozenset())
    output_path = tmp_path / "castep_keywords.json"

    def cancel_after_two_batches() -> bool:
        return sum(1 for command in _DirectSSH.last.commands if _LOOP_NAMES_RE.search(command)) >= 2

    with pytest.raises(ClusterKeywordGenerationCancelled):
        generate_cluster_castep_keywords_json(
            config=_direct_config(),
            output_path=output_path,
            cancel_check=cancel_after_two_batches,
        )

    assert not output_path.exists()
    assert _DirectSSH.last.disconnected is True


def test_jump_host_connection_uses_full_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    class _JumpSSH(_DirectSSH):
        def connect(self, password: str | None = None, credentials=None) -> None:
            seen.append(credentials)

    monkeypatch.setattr(generator_module, "SSHConnection", _JumpSSH)
    monkeypatch.setattr(_DirectSSH, "failing", frozenset())

    generate_cluster_castep_keywords_json(
        config=_direct_config(jump_host_enabled=True),
        output_path=tmp_path / "castep_keywords.json",
        password="secret",
    )

    assert len(seen) == 1 and seen[0].target_secret == "secret"


# --- scheduler mode -----------------------------------------------------------


def test_scheduler_mode_parses_the_remote_dump_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
) -> None:
    scheduler = _Scheduler([JobStatus.QUEUED, JobStatus.RUNNING], dump=_scheduler_dump(_ALL_NAMES))
    _use_scheduler(monkeypatch, scheduler)
    output_path = tmp_path / "castep_keywords.json"
    progress: list[str] = []

    generate_cluster_castep_keywords_json(
        config=_scheduler_config(),
        output_path=output_path,
        progress=lambda message, _percent: progress.append(message),
    )

    ssh = _SchedulerSSH.last
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["cell"]) + len(payload["param"]) == 211
    assert payload["castep_version"] == "26.11"

    request = scheduler.submit_requests[0]
    assert request.script_path.endswith("/run_keywords.sh")
    assert request.working_dir.startswith("/work4/isis/alice/.mufinder-keywords-")
    runner = ssh.remote_files[request.script_path].decode("utf-8")
    # The job is modelled on the cluster's own job script: its partition, its
    # environment setup and its launcher, with a short walltime of its own.
    assert runner.startswith(
        "#!/bin/bash\n\n#SBATCH -J mufinder-keywords\n#SBATCH -n 1\n#SBATCH -p scarf\n#SBATCH -t 01:00:00\n\n"
        "module load CASTEP\n"
    )
    assert "mufinder_try 'a single job step' srun -n 1 bash mufinder_help_loop.sh castep.mpi; then" in runner
    assert "mufinder_try 'the launcher for every query' env MUFINDER_PARALLEL=1 bash mufinder_help_loop.sh srun -n 1" in runner
    assert "srun -n 1 castep.mpi keywords" not in runner
    assert "python" not in runner.lower()
    # Nothing but the runner script is uploaded.
    assert [args for kind, args in ssh.commands if kind == "put"] == [request.script_path]

    assert "Keyword job 42 is queued..." in progress
    # A running job reports how far it has got, so a slow one is not mistaken
    # for a stuck one.
    assert "Keyword job 42 is running... (137 keywords described so far)" in progress
    assert progress[-1] == "CASTEP keyword generation complete."
    assert ssh.deleted == [request.working_dir]
    assert ssh.disconnected is True


def test_scheduler_keyword_job_runs_on_the_default_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
) -> None:
    from mufinder.domain.cluster import ClusterQueue

    scheduler = _Scheduler(dump=_scheduler_dump(_ALL_NAMES))
    _use_scheduler(monkeypatch, scheduler)

    generate_cluster_castep_keywords_json(
        config=_scheduler_config(
            name="PSI",
            template_script_path="",
            default_queue_name="daily",
            queues=(
                ClusterQueue(name="hourly"),
                ClusterQueue(name="daily", castep_executable="/opt/castep-daily/castep.mpi"),
            ),
        ),
        output_path=tmp_path / "castep_keywords.json",
    )

    request = scheduler.submit_requests[0]
    runner = _SchedulerSSH.last.remote_files[request.script_path].decode("utf-8")
    preamble = runner.split("\n\n", 2)[1].splitlines()
    assert preamble[-1] == "#SBATCH -p daily"
    assert "hourly" not in runner
    # The default queue's executable override applies to the keyword job.
    assert "/opt/castep-daily/castep.mpi" in runner


def test_scheduler_keyword_job_keeps_a_template_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
) -> None:
    scheduler = _Scheduler(dump=_scheduler_dump(_ALL_NAMES))
    _use_scheduler(monkeypatch, scheduler)

    generate_cluster_castep_keywords_json(
        config=_scheduler_config(default_queue_name="devel"),
        output_path=tmp_path / "castep_keywords.json",
    )

    request = scheduler.submit_requests[0]
    runner = _SchedulerSSH.last.remote_files[request.script_path].decode("utf-8")
    assert "#SBATCH -p scarf" in runner
    assert "devel" not in runner


@pytest.mark.parametrize("grep_result", [(1, "0\n", ""), (2, "", "No such file"), (0, "garbage", "")])
def test_running_job_progress_detail_is_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
    grep_result: tuple[int, str, str],
) -> None:
    scheduler = _Scheduler([JobStatus.RUNNING], dump=_scheduler_dump(_ALL_NAMES))
    _use_scheduler(monkeypatch, scheduler)
    monkeypatch.setattr(_SchedulerSSH, "grep_result", grep_result)
    progress: list[str] = []

    generate_cluster_castep_keywords_json(
        config=_scheduler_config(),
        output_path=tmp_path / "castep_keywords.json",
        progress=lambda message, _percent: progress.append(message),
    )

    assert "Keyword job 42 is running..." in progress


def test_scheduler_polling_backs_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]) -> None:
    scheduler = _Scheduler([JobStatus.QUEUED] * 6, dump=_scheduler_dump(_ALL_NAMES))
    _use_scheduler(monkeypatch, scheduler)

    generate_cluster_castep_keywords_json(
        config=_scheduler_config(),
        output_path=tmp_path / "castep_keywords.json",
    )

    # Waits of 2, 2, 4, 8, 15, 15 seconds between the seven status checks,
    # each slept in slices of at most a second so cancel stays responsive.
    assert scheduler.status_calls == 7
    assert sum(no_sleep) == pytest.approx(46.0)
    assert max(no_sleep) <= 1.0


def test_addqueue_uses_scheduler_mode_with_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
) -> None:
    scheduler = _Scheduler(dump=_scheduler_dump(_ALL_NAMES))
    _use_scheduler(monkeypatch, scheduler)

    generate_cluster_castep_keywords_json(
        config=_scheduler_config(
            name="Redwood",
            queue_system="addqueue",
            keyword_execution_mode="direct",
            default_walltime="",
        ),
        output_path=tmp_path / "castep_keywords.json",
    )

    request = scheduler.submit_requests[0]
    assert request.castep_executable == "/bin/bash"
    assert request.seed_name.endswith("/run_keywords.sh")
    assert request.walltime == ""
    runner = _SchedulerSSH.last.remote_files[request.seed_name].decode("utf-8")
    assert "#SBATCH" not in runner
    # addqueue starts the script inside an MPI job step, where a bare CASTEP
    # that fails to start takes the whole step down: no probing, launcher only.
    assert (
        "mufinder_try 'standalone CASTEP processes inside one launcher' "
        "mufinder_limit 900 mpirun -np 1 bash mufinder_help_loop.sh --standalone castep.mpi \\\n"
        "  || mufinder_try 'the launcher for every query' env MUFINDER_PARALLEL=1 MUFINDER_FAST_START=0 "
        "bash mufinder_help_loop.sh mpirun -np 1 castep.mpi || true"
    ) in runner
    assert "the bare CASTEP executable" not in runner
    assert "a single job step" not in runner
    assert "mufinder_castep --version 2>&1 || true" in runner
    assert "$(mufinder_castep -help all 2>&1 | awk" in runner


def test_scheduler_mode_reports_a_truncated_dump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
) -> None:
    scheduler = _Scheduler(
        dump=_scheduler_dump(_ALL_NAMES[:20], complete=False),
        log=b"slurmstepd: error: *** JOB 42 CANCELLED DUE TO TIME LIMIT ***\n",
    )
    _use_scheduler(monkeypatch, scheduler)
    output_path = tmp_path / "castep_keywords.json"

    with pytest.raises(ClusterKeywordGenerationError) as excinfo:
        generate_cluster_castep_keywords_json(config=_scheduler_config(), output_path=output_path)

    message = str(excinfo.value)
    assert "ended before it had described every keyword" in message
    assert "CANCELLED DUE TO TIME LIMIT" in message
    assert not output_path.exists()
    # A failed job's files stay on the cluster, and the message says where.
    ssh = _SchedulerSSH.last
    working_dir = scheduler.submit_requests[0].working_dir
    assert ssh.deleted == []
    assert f"The job's files were left in {working_dir} on SCARF for inspection (scheduler job 42)." in message
    assert ssh.disconnected is True


def test_scheduler_mode_reports_missing_dump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep) -> None:
    _use_scheduler(monkeypatch, _Scheduler())

    with pytest.raises(ClusterKeywordGenerationError, match="ended before it had described every keyword"):
        generate_cluster_castep_keywords_json(
            config=_scheduler_config(),
            output_path=tmp_path / "castep_keywords.json",
        )


def test_scheduler_mode_explains_a_castep_that_never_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
) -> None:
    """Regression: Redwood's first failure read "described only 0 of 0 keywords"."""

    dump = (
        "@@MUFINDER_VERSION_BEGIN@@\n*** An error occurred in MPI_Init\n\n@@MUFINDER_VERSION_END@@\n"
        "@@MUFINDER_DUMP_COMPLETE@@\n"
    ).encode("utf-8")
    _use_scheduler(monkeypatch, _Scheduler(dump=dump))

    with pytest.raises(ClusterKeywordGenerationError) as excinfo:
        generate_cluster_castep_keywords_json(
            config=_scheduler_config(),
            output_path=tmp_path / "castep_keywords.json",
        )

    message = str(excinfo.value)
    assert message.startswith("CASTEP on SCARF did not list any keywords, so no keyword catalog was written.")
    assert "usually means CASTEP could not start" in message
    assert "0 of 0" not in message
    assert "--- keyword_help.txt ---\n*** An error occurred in MPI_Init" in message


# The stand-in for an MPI-built CASTEP on an addqueue cluster. `in_job` is true
# when the process still carries the variables of the MPI job it was started in.
_ADDQUEUE_SCENARIOS = {
    # Stripped of the MPI job's variables CASTEP starts standalone, so the
    # whole loop runs inside a single launcher.
    "standalone_works": {
        "may_start": "! in_job",
        "tried": ["standalone CASTEP processes inside one launcher"],
        "launches": 1,
    },
    # CASTEP cannot start standalone here; only a process launched directly
    # works, so the proven path takes over: one launch per query.
    "standalone_fails": {
        "may_start": 'in_job && [ -n "$DIRECT" ]',
        "tried": ["standalone CASTEP processes inside one launcher", "the launcher for every query"],
        # The container, then 211 queries, the listing, the version and the timed query.
        "launches": 215,
    },
}


@pytest.mark.skipif(sys.platform == "win32", reason="the scheduler runner is a POSIX shell script")
@pytest.mark.parametrize("scenario", sorted(_ADDQUEUE_SCENARIOS))
def test_addqueue_job_never_starts_castep_bare_in_the_job_step(tmp_path: Path, monkeypatch, scenario: str) -> None:
    """Inside addqueue's job step a failed bare start would cancel the whole job."""

    spec = _ADDQUEUE_SCENARIOS[scenario]
    real = _write_fake_castep(tmp_path / "castep")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bare_starts = bin_dir / "bare.log"
    picky = bin_dir / "castep.mpi"
    picky.write_text(
        "#!/bin/bash\n"
        'in_job() { [ -n "$PMIX_NAMESPACE" ] || [ -n "$SLURM_STEP_ID" ]; }\n'
        # Started straight from the job script, without any launcher.
        f'if [ -z "$UNDER_LAUNCHER" ]; then echo bare >> {str(bare_starts)!r}; exit 1; fi\n'
        f"if {spec['may_start']}; then exec {str(real)!r} \"$@\"; fi\n"
        'echo "MPI_Init failed" >&2; exit 1\n',
        encoding="utf-8",
    )
    picky.chmod(0o755)
    log = bin_dir / "launcher.log"
    launcher = bin_dir / "mpirun"
    launcher.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$*" >> {str(log)!r}\n'
        "shift 2\n"
        'if [ "$1" != bash ]; then export DIRECT=1; fi\n'
        'UNDER_LAUNCHER=1 PMIX_NAMESPACE=prterun-1 exec "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    _write_timeout(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    # The job script itself runs as a task of addqueue's srun step.
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("PMIX_NAMESPACE", "slurm.pmix.932267.0")
    config = _scheduler_config(
        name="Redwood",
        queue_system="addqueue",
        castep_executable="castep.mpi",
        keyword_shell_init="",
        template_script_path="",
    )

    _script, dump = _run_scheduler_script(tmp_path, config)

    _assert_complete_dump(dump)
    assert not bare_starts.exists()
    job_log = (tmp_path / "work dir" / "keyword_generation.log").read_text(encoding="utf-8")
    tried = [line.removeprefix("mufinder: trying ") for line in job_log.splitlines() if line.startswith("mufinder: trying ")]
    assert tried == spec["tried"]
    assert len(log.read_text(encoding="utf-8").splitlines()) == spec["launches"]


def test_mpi_job_environment_pattern_selects_only_job_bound_variables() -> None:
    pattern = re.compile(scripts_module._MPI_JOB_ENVIRONMENT_RE)

    for name in (
        "PMIX_NAMESPACE",
        "PMIX_RANK",
        "PMI_FD",
        "PRTE_LAUNCHED",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_MCA_ess_base_jobid",
        "SLURM_STEP_ID",
        "SLURM_STEPID",
        "SLURM_PMIX_MAPPING_SERV",
        "SLURM_TASK_PID",
        "SLURM_PROCID",
    ):
        assert pattern.match(name), name
    # The allocation itself, the user's environment and the site's transport
    # choice (Redwood sets OMPI_MCA_pml) must survive.
    for name in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "PATH", "LD_LIBRARY_PATH", "OMPI_MCA_pml", "OMPI_MCA_btl"):
        assert not pattern.match(name), name


def test_scheduler_failed_state_raises_with_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep) -> None:
    scheduler = _Scheduler([JobStatus.FAILED], log=b"castep.mpi: error while loading shared libraries\n")
    _use_scheduler(monkeypatch, scheduler)

    with pytest.raises(ClusterKeywordGenerationError) as excinfo:
        generate_cluster_castep_keywords_json(
            config=_scheduler_config(),
            output_path=tmp_path / "castep_keywords.json",
        )

    assert "Cluster keyword generation job failed on SCARF." in str(excinfo.value)
    assert "error while loading shared libraries" in str(excinfo.value)


def test_failed_submission_reports_the_scheduler_error_and_keeps_the_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job that never starts used to fail with a bare one-line error."""

    class _RejectingScheduler(_Scheduler):
        def submit(self, ssh: _SchedulerSSH, request) -> str:
            raise RuntimeError("addqueue: unknown queue 'redwood'")

    _use_scheduler(monkeypatch, _RejectingScheduler())

    with pytest.raises(ClusterKeywordGenerationError) as excinfo:
        generate_cluster_castep_keywords_json(
            config=_scheduler_config(name="Redwood", queue_system="addqueue"),
            output_path=tmp_path / "castep_keywords.json",
        )

    message = str(excinfo.value)
    ssh = _SchedulerSSH.last
    assert message.startswith("addqueue: unknown queue 'redwood'")
    assert "The job's files were left in /work4/isis/alice/.mufinder-keywords-" in message
    assert "(scheduler job" not in message
    # The scripts MuFinder wrote itself are not quoted back.
    assert "run_keywords.sh ---" not in message
    assert ssh.deleted == []


def test_failure_report_quotes_the_scheduler_output_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
) -> None:
    """The scheduler's own output is where a job that died early explains itself."""

    class _SchedulerWithOutput(_Scheduler):
        def submit(self, ssh: _SchedulerSSH, request) -> str:
            job_id = super().submit(ssh, request)
            ssh.remote_files[f"{request.working_dir}/slurm-42.out"] = b"/bin/bash: module: command not found\n"
            ssh.remote_files[f"{request.working_dir}/empty.err"] = b""
            return job_id

    scheduler = _SchedulerWithOutput(log=b"mufinder: trying the bare CASTEP executable\n")
    _use_scheduler(monkeypatch, scheduler)

    with pytest.raises(ClusterKeywordGenerationError) as excinfo:
        generate_cluster_castep_keywords_json(
            config=_scheduler_config(),
            output_path=tmp_path / "castep_keywords.json",
        )

    message = str(excinfo.value)
    assert "ended before it had described every keyword" in message
    assert "--- slurm-42.out ---\n/bin/bash: module: command not found" in message
    assert "empty.err" not in message
    # The job log was already part of the message and is not repeated.
    assert message.count("mufinder: trying the bare CASTEP executable") == 1


def test_failure_report_survives_an_unlistable_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
) -> None:
    class _NoListSSH(_SchedulerSSH):
        def listdir(self, remote_path: str) -> list[str]:
            raise OSError("sftp session lost")

    monkeypatch.setattr(generator_module, "SSHConnection", _NoListSSH)
    monkeypatch.setattr(generator_module, "create_scheduler", lambda *_args, **_kwargs: _Scheduler())

    with pytest.raises(ClusterKeywordGenerationError, match="The job's files were left in"):
        generate_cluster_castep_keywords_json(
            config=_scheduler_config(),
            output_path=tmp_path / "castep_keywords.json",
        )


def test_scheduler_cancelled_state_raises_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep) -> None:
    scheduler = _Scheduler([JobStatus.CANCELLED])
    _use_scheduler(monkeypatch, scheduler)

    with pytest.raises(ClusterKeywordGenerationCancelled, match="job was cancelled"):
        generate_cluster_castep_keywords_json(
            config=_scheduler_config(),
            output_path=tmp_path / "castep_keywords.json",
        )
    assert scheduler.cancelled_jobs == ["42"]


def test_cancel_while_waiting_cancels_the_remote_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
) -> None:
    scheduler = _Scheduler([JobStatus.QUEUED] * 50)
    _use_scheduler(monkeypatch, scheduler)

    with pytest.raises(ClusterKeywordGenerationCancelled):
        generate_cluster_castep_keywords_json(
            config=_scheduler_config(),
            output_path=tmp_path / "castep_keywords.json",
            cancel_check=lambda: scheduler.status_calls >= 1,
        )

    assert scheduler.cancelled_jobs == ["42"]
    assert len(_SchedulerSSH.last.deleted) == 1


def test_cancel_and_cleanup_errors_do_not_mask_the_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
) -> None:
    class _FragileScheduler(_Scheduler):
        def cancel(self, _ssh, job_id: str) -> None:
            raise RuntimeError("scancel unavailable")

    class _FragileSSH(_SchedulerSSH):
        def rmtree(self, remote_path: str) -> None:
            raise OSError("permission denied")

    scheduler = _FragileScheduler([JobStatus.QUEUED] * 5)
    monkeypatch.setattr(generator_module, "SSHConnection", _FragileSSH)
    monkeypatch.setattr(generator_module, "create_scheduler", lambda *_args, **_kwargs: scheduler)

    with pytest.raises(ClusterKeywordGenerationCancelled):
        generate_cluster_castep_keywords_json(
            config=_scheduler_config(),
            output_path=tmp_path / "castep_keywords.json",
            cancel_check=lambda: scheduler.status_calls >= 1,
        )


# --- the generated shell, run for real ----------------------------------------


def _write_fake_castep(directory: Path, *, broken: tuple[str, ...] = ()) -> Path:
    """Write a stand-in ``castep`` that serves the synthetic help text."""

    help_dir = directory / "help"
    help_dir.mkdir(parents=True)
    (directory / "all.txt").write_text(help_all_text(), encoding="utf-8")
    for name in _ALL_NAMES:
        if name not in broken:
            (help_dir / f"{name}.txt").write_text(help_keyword_text(name), encoding="utf-8")
    executable = directory / "fake castep"  # the space proves the quoting
    executable.write_text(
        "#!/bin/bash\n"
        f'root={str(directory)!r}\n'
        'if [ "$1" = "--version" ]; then echo " CASTEP version: 25.12 "; exit 0; fi\n'
        'if [ "$2" = "all" ]; then cat "$root/all.txt"; exit 0; fi\n'
        'if [ -f "$root/help/$2.txt" ]; then cat "$root/help/$2.txt"; exit 0; fi\n'
        'echo "no help for $2" >&2; exit 3\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


@pytest.mark.skipif(sys.platform == "win32", reason="the remote help loop is a POSIX shell script")
def test_help_loop_command_runs_under_real_bash(tmp_path: Path) -> None:
    executable = _write_fake_castep(tmp_path / "castep", broken=("PARAM_FILLER_2",))
    config = _direct_config(castep_executable=str(executable))
    command = scripts_module.help_loop_command(
        scripts_module.build_castep_command_with_launcher(config),
        ["TASK", "LATTICE_CART", "PARAM_FILLER_2"],
    )

    completed = subprocess.run(["/bin/bash", "-c", f"set -e\n{command}"], capture_output=True, text=True, check=True)

    dump = generator_module._HelpDump(cell={}, param={})
    dump.add(completed.stdout)
    assert dump.requested == 3
    assert dump.param == {
        "task": {"keyword": "task", "option_type": "String", "level": "Basic", "docstring": "Help for task."}
    }
    assert list(dump.cell) == ["lattice_cart"]
    # stderr is folded into the dump so a failing call explains itself.
    assert dump.first_failure == "castep -help PARAM_FILLER_2:\nno help for PARAM_FILLER_2"


@pytest.mark.skipif(sys.platform == "win32", reason="the remote help loop is a POSIX shell script")
@pytest.mark.parametrize(("parallel", "expected_peak"), [("1", 1), ("4", 4)])
def test_help_loop_runs_queries_side_by_side_and_prints_them_in_order(
    tmp_path: Path,
    parallel: str,
    expected_peak: int,
) -> None:
    """Queries overlap, yet each description is printed whole and in order."""

    active = tmp_path / "active"
    active.mkdir()
    peak = tmp_path / "peak.log"
    env_log = tmp_path / "env.log"
    executable = tmp_path / "castep"
    executable.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "--version" ]; then echo " CASTEP version: 25.12 "; exit 0; fi\n'
        f'echo "$OMPI_MCA_pml/$OMPI_MCA_btl/$OMPI_MCA_mpi_yield_when_idle" >> {str(env_log)!r}\n'
        f'touch {str(active)!r}/$2\n'
        "sleep 0.3\n"
        f'ls {str(active)!r} | wc -l >> {str(peak)!r}\n'
        f'rm {str(active)!r}/$2\n'
        'echo "help for $2"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    names = [f"KEY_{index}" for index in range(8)]
    command = scripts_module.help_loop_command(shlex.quote(str(executable)), names)

    completed = subprocess.run(
        ["/bin/bash", "-c", f"set -e\n{command}"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "MUFINDER_PARALLEL": parallel},
    )

    sections = generator_module._HELP_SECTION_RE.findall(completed.stdout)
    assert [(name, doc, status) for name, doc, _tail, status in _sections(completed.stdout)] == [
        (name, f"help for {name}", "0") for name in names
    ]
    assert len(sections) == 8
    assert max(int(line) for line in peak.read_text(encoding="utf-8").split()) == expected_peak
    # CASTEP answered with the Open MPI fast-start settings, so they were kept.
    assert set(env_log.read_text(encoding="utf-8").split()) == {"ob1/self/1"}


@pytest.mark.skipif(sys.platform == "win32", reason="the remote help loop is a POSIX shell script")
@pytest.mark.parametrize("pkill", ["real", "finds_nothing", "missing"])
def test_help_loop_stops_grouping_queries_when_it_does_not_pay(tmp_path: Path, pkill: str) -> None:
    """Regression: on Redwood four queries sharing one CPU were slower than four in turn.

    The login node runs this loop under ``set -e``, so abandoning the trial
    group must survive a ``pkill`` that finds nothing to kill or is not
    installed.
    """

    if not subprocess.run(["date", "+%s%N"], capture_output=True, text=True).stdout.strip().isdigit():
        pytest.skip("this platform's date has no nanosecond clock, so the loop does not calibrate")

    active = tmp_path / "active"
    active.mkdir()
    peak = tmp_path / "peak.log"
    executable = tmp_path / "castep"
    # A query takes about 0.15 s alone but 1.65 s when four overlap: running
    # side by side is slower than taking turns, as on a single contended core.
    executable.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "--version" ]; then echo " CASTEP version: 25.12 "; exit 0; fi\n'
        f'marker={str(active)!r}/$2\n'
        'touch "$marker"\n'
        # Exit promptly when the loop abandons the group, as a real process would.
        'trap \'rm -f "$marker"; kill "$nap" 2>/dev/null; exit 143\' TERM\n'
        "sleep 0.05 & nap=$!; wait $nap\n"
        f'busy=$(ls {str(active)!r} | wc -l)\n'
        f'echo "$busy" >> {str(peak)!r}\n'
        'sleep "$(awk "BEGIN { print 0.1 * $busy * $busy }")" & nap=$!; wait $nap\n'
        'rm -f "$marker"\n'
        'echo "help for $2"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    names = [f"KEY_{index}" for index in range(13)]
    command = scripts_module.help_loop_command(shlex.quote(str(executable)), names)

    env = dict(os.environ)
    if pkill != "real":
        tools = tmp_path / "tools"
        tools.mkdir()
        if pkill == "finds_nothing":
            (tools / "pkill").write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
            (tools / "pkill").chmod(0o755)
            env["PATH"] = f"{tools}:{env['PATH']}"
        else:
            for name in ("bash", "cat", "date", "sleep", "mktemp", "rm", "ls", "wc", "awk", "touch", "grep"):
                (tools / name).symlink_to(shutil.which(name))
            env["PATH"] = str(tools)

    completed = subprocess.run(
        ["/bin/bash", "-c", f"set -e\n{command}"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    assert [(name, doc) for name, doc, _tail, _status in _sections(completed.stdout)] == [
        (name, f"help for {name}") for name in names
    ]
    match = re.search(
        r"mufinder: 4 queries at once had not finished after (\d+) ms against (\d+) ms for one; "
        r"running one at a time",
        completed.stderr,
    )
    assert match is not None
    # The trial group is abandoned as soon as it has used three quarters of
    # the time four queries take in turn, not waited out (it would take 1.65 s).
    abandoned_after, single = int(match.group(1)), int(match.group(2))
    assert abandoned_after < 1200
    assert abandoned_after >= single * 3
    # One alone, the abandoned group of four, then all twelve one at a time:
    # the abandoned queries are described again, and nothing is lost.
    if pkill == "real":
        peaks = [int(line) for line in peak.read_text(encoding="utf-8").split()]
        assert peaks.count(4) == 4
        assert peaks.count(1) == 13
        # Nothing is left behind in the scratch directories.
        assert not list(active.iterdir())


def _write_real_timeout(directory: Path) -> None:
    """Provide a working ``timeout`` (absent on macOS): exit 124 when the limit passes."""

    command = directory / "timeout"
    command.write_text(
        "#!/bin/bash\n"
        "limit=$1; shift\n"
        '"$@" & pid=$!\n'
        '( sleep "$limit"; kill -TERM "$pid" 2>/dev/null ) & watcher=$!\n'
        "wait $pid; status=$?\n"
        "kill $watcher 2>/dev/null\n"
        'if [ "$status" -eq 143 ]; then exit 124; fi\n'
        "exit $status\n",
        encoding="utf-8",
    )
    command.chmod(0o755)


@pytest.mark.skipif(sys.platform == "win32", reason="the remote help loop is a POSIX shell script")
def test_help_loop_gives_up_on_a_castep_that_stops_responding(tmp_path: Path) -> None:
    """A binary that answers --version and then hangs must not hold the job to its walltime."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_real_timeout(bin_dir)
    started = tmp_path / "started.log"
    executable = tmp_path / "castep"
    executable.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "--version" ]; then echo " CASTEP version: 25.12 "; exit 0; fi\n'
        f'echo "$2" >> {str(started)!r}\n'
        'if [ "$2" = "KEY_0" ]; then echo "help for $2"; exit 0; fi\n'
        "exec sleep 30\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    names = [f"KEY_{index}" for index in range(40)]
    command = scripts_module.help_loop_command(shlex.quote(str(executable)), names)

    completed = subprocess.run(
        ["/bin/bash", "-c", command],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "MUFINDER_QUERY_TIMEOUT": "0.3",
            "MUFINDER_PARALLEL": "1",
        },
        timeout=60,
    )

    assert completed.returncode == 75
    assert "mufinder: CASTEP stopped responding: 3 queries were cut off after 0.3 s each" in completed.stderr
    # It stopped after three hung queries instead of working through all forty.
    assert started.read_text(encoding="utf-8").split() == ["KEY_0", "KEY_1", "KEY_2", "KEY_3"]
    statuses = [status for _name, _doc, _tail, status in _sections(completed.stdout)]
    assert statuses == ["0", "124", "124", "124"]


@pytest.mark.skipif(sys.platform == "win32", reason="the scheduler runner is a POSIX shell script")
def test_scheduler_job_moves_on_from_a_strategy_that_stops_responding(tmp_path: Path, monkeypatch) -> None:
    """Bare CASTEP passes the probe, then hangs; the launcher path still delivers."""

    real = _write_fake_castep(tmp_path / "castep")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_real_timeout(bin_dir)
    flaky = bin_dir / "castep.mpi"
    flaky.write_text(
        "#!/bin/bash\n"
        f'if [ -n "$LAUNCHED" ] || [ "$1" = "--version" ]; then exec {str(real)!r} "$@"; fi\n'
        "exec sleep 30\n",
        encoding="utf-8",
    )
    flaky.chmod(0o755)
    launcher = bin_dir / "site_launch"
    launcher.write_text('#!/bin/bash\nshift 2\nLAUNCHED=1 exec "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("MUFINDER_QUERY_TIMEOUT", "0.3")
    template = tmp_path / "hardcoded.sh"
    template.write_text(
        "#!/bin/bash\n#SBATCH -p short\n\nsite_launch -n $num_processes castep.mpi $seed_name\n",
        encoding="utf-8",
    )
    config = _scheduler_config(castep_executable="castep.mpi", template_script_path=str(template), keyword_shell_init="")

    _script, dump = _run_scheduler_script(tmp_path, config)

    _assert_complete_dump(dump)
    job_log = (tmp_path / "work dir" / "keyword_generation.log").read_text(encoding="utf-8")
    tried = [line.removeprefix("mufinder: trying ") for line in job_log.splitlines() if line.startswith("mufinder: trying ")]
    assert tried == ["the bare CASTEP executable", "a single job step"]
    # The hung listing was cut off, leaving the bare strategy nothing to describe.
    assert "mufinder: CASTEP listed no keywords" in job_log


def _sections(text: str) -> list[tuple[str, str, str, str]]:
    return [
        (match.group("name"), match.group("doc").strip(), "", match.group("status"))
        for match in generator_module._HELP_SECTION_RE.finditer(text)
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="the remote help loop is a POSIX shell script")
def test_fast_start_settings_are_dropped_when_castep_rejects_them(tmp_path: Path) -> None:
    env_log = tmp_path / "env.log"
    executable = tmp_path / "castep"
    executable.write_text(
        "#!/bin/bash\n"
        'if [ -n "$OMPI_MCA_pml" ] && [ "$1" = "--version" ]; then echo "MPI_Init failed" >&2; exit 1; fi\n'
        f'echo "[$OMPI_MCA_pml]" >> {str(env_log)!r}\n'
        'echo "help for $2"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    command = scripts_module.help_loop_command(shlex.quote(str(executable)), ["TASK"])

    env = {name: value for name, value in os.environ.items() if not name.startswith("OMPI_MCA_")}
    subprocess.run(["/bin/bash", "-c", command], capture_output=True, text=True, check=True, env=env)

    assert env_log.read_text(encoding="utf-8").split() == ["[]"]


def _run_scheduler_script(tmp_path: Path, config: ClusterConfig) -> tuple[str, str]:
    """Run the keyword job script under bash and return ``(script, dump)``."""

    work = tmp_path / "work dir"
    work.mkdir()
    dump_path = work / "keyword_help.txt"
    script = scripts_module.scheduler_runner_script(
        config=config,
        remote_directory=str(work),
        remote_dump_path=str(dump_path),
        remote_log_path=str(work / "keyword_generation.log"),
    )
    runner = work / "run_keywords.sh"
    runner.write_text(script, encoding="utf-8")
    subprocess.run(["/bin/bash", str(runner)], check=True)
    return script, dump_path.read_text(encoding="utf-8")


def _assert_complete_dump(text: str) -> None:
    dump = generator_module._HelpDump(cell={}, param={})
    dump.add(text)
    names = complete_keyword_names()
    assert scripts_module.DUMP_COMPLETE_MARKER in text
    assert set(dump.cell) == set(names["cell"])
    assert set(dump.param) == set(names["param"])
    version = generator_module._VERSION_SECTION_RE.search(text)
    assert version is not None
    assert generator_module._parse_castep_version(version.group("text")) == "25.12"
    payload = generator_module._checked_payload(dump, "25.12", _scheduler_config())
    assert payload["castep_version"] == "25.12"


def _write_launcher(directory: Path, name: str) -> tuple[Path, Path]:
    """Write a stand-in launcher that logs each use and runs its command."""

    log = directory / f"{name}.log"
    launcher = directory / name
    launcher.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$*" >> {str(log)!r}\nshift 2\nexec "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher, log


def _write_timeout(directory: Path) -> None:
    """Provide ``timeout`` (absent on macOS) so the bare-binary probe can run."""

    command = directory / "timeout"
    command.write_text('#!/bin/bash\nshift\nexec "$@"\n', encoding="utf-8")
    command.chmod(0o755)


@pytest.mark.skipif(sys.platform == "win32", reason="the scheduler runner is a POSIX shell script")
def test_scheduler_job_follows_a_template_that_uses_castep_command(tmp_path: Path, monkeypatch) -> None:
    """The job script comes from the cluster's own template, launcher included."""

    executable = _write_fake_castep(tmp_path / "castep")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _launcher, launcher_log = _write_launcher(bin_dir, "srun")
    _write_timeout(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    template = tmp_path / "site.sh"
    template.write_text(
        "#!/bin/bash\n"
        "#SBATCH -J $job_name\n"
        "#SBATCH -n $num_processes\n"
        "#SBATCH -A muons\n"
        "#SBATCH -t $walltime\n"
        "\n"
        "export SITE_SETUP=1\n"
        "cd $working_directory\n"
        "$castep_command\n"
        "rm -f $seed_name.check\n",
        encoding="utf-8",
    )
    config = _scheduler_config(
        castep_executable=str(executable),
        template_script_path=str(template),
        keyword_shell_init="",
    )

    script, dump = _run_scheduler_script(tmp_path, config)

    assert script.startswith(
        "#!/bin/bash\n#SBATCH -J mufinder-keywords\n#SBATCH -n 1\n#SBATCH -A muons\n#SBATCH -t 01:00:00\n\n"
        "export SITE_SETUP=1\n"
    )
    # The calculation's own run line and clean-up are not part of the job.
    assert "keywords.check" not in script
    _assert_complete_dump(dump)
    # A bare binary answered inside the "allocation", so no job step was
    # started at all.
    assert not launcher_log.exists()
    job_log = (tmp_path / "work dir" / "keyword_generation.log").read_text(encoding="utf-8").splitlines()
    assert job_log[0] == "mufinder: trying the bare CASTEP executable"
    # The log records what a query costs on the node, for diagnosing a slow run.
    assert re.fullmatch(r"mufinder: \S+ cpu\(s\) available, Open MPI fast start applied", job_log[1])
    assert re.fullmatch(r"mufinder: one query took [\d.]+ s \([\d.]+ s user, [\d.]+ s system\)", job_log[2])
    # ... and whether running queries side by side paid off on this node.
    assert re.fullmatch(
        r"mufinder: 4 queries at once took \d+ ms against \d+ ms for one; (keeping 4|running one) at a time",
        job_log[3],
    )
    assert len(job_log) == 4


# How the stand-in CASTEP decides whether it may start, per scenario. It is an
# "MPI binary" that fails unless the stand-in launcher prepared its environment.
_STRATEGY_SCENARIOS = {
    # Any process started under the launcher works, including one started by a
    # shell inside the step: the whole loop runs in a single job step.
    "single_step": {
        "launcher": 'LAUNCHED=1 exec "$@"',
        "may_start": '[ -n "$LAUNCHED" ]',
        "timeout": True,
        "tried": ["a single job step"],
        # The two-start probe and the step that runs the loop.
        "launches": 2,
    },
    # CASTEP only starts as the launcher's own child, so the probe inside a
    # step fails and every query is launched separately.
    "step_probe_fails": {
        "launcher": 'if [ "$1" = bash ]; then exec "$@"; else LAUNCHED=1 exec "$@"; fi',
        "may_start": '[ -n "$LAUNCHED" ]',
        "timeout": True,
        "tried": ["the launcher for every query"],
        # The failed step probe, then the start-up probe, 211 queries, the
        # listing, the version, and the timed query.
        "launches": 216,
    },
    # The probe passes but the loop then fails inside the step (only the
    # version query works there), so its dump is rejected and replaced.
    "step_dump_rejected": {
        "launcher": 'if [ "$1" = bash ]; then IN_STEP=1 exec "$@"; else LAUNCHED=1 exec "$@"; fi',
        "may_start": '[ -n "$LAUNCHED" ] || { [ -n "$IN_STEP" ] && [ "$1" = "--version" ]; }',
        "timeout": True,
        "tried": ["a single job step", "the launcher for every query"],
        "launches": 217,
    },
    # Without `timeout` a hanging binary could stall the job, so neither
    # shortcut is attempted.
    "no_timeout": {
        "launcher": 'LAUNCHED=1 exec "$@"',
        "may_start": '[ -n "$LAUNCHED" ]',
        "timeout": False,
        "tried": ["the launcher for every query"],
        "launches": 215,
    },
}


@pytest.mark.skipif(sys.platform == "win32", reason="the scheduler runner is a POSIX shell script")
@pytest.mark.parametrize("scenario", sorted(_STRATEGY_SCENARIOS))
def test_scheduler_job_uses_the_launcher_as_little_as_the_cluster_allows(
    tmp_path: Path,
    monkeypatch,
    scenario: str,
) -> None:
    spec = _STRATEGY_SCENARIOS[scenario]
    real = _write_fake_castep(tmp_path / "castep")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    picky = bin_dir / "castep.mpi"
    picky.write_text(
        "#!/bin/bash\n"
        f"if {spec['may_start']}; then exec {str(real)!r} \"$@\"; fi\n"
        'echo "MPI_Init failed" >&2; exit 1\n',
        encoding="utf-8",
    )
    picky.chmod(0o755)
    log = bin_dir / "launcher.log"
    launcher = bin_dir / "site_launch"
    launcher.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$*" >> {str(log)!r}\nshift 2\n{spec["launcher"]}\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    path = f"{bin_dir}:{os.environ['PATH']}"
    if spec["timeout"]:
        _write_timeout(bin_dir)
    else:
        # Hide any real `timeout` (present on Linux) behind a PATH without it.
        tools = tmp_path / "tools"
        tools.mkdir()
        for name in ("bash", "cat", "grep", "awk", "printf", "env", "mktemp", "rm"):
            located = shutil.which(name)
            if located:
                (tools / name).symlink_to(located)
        path = f"{bin_dir}:{tools}"
    monkeypatch.setenv("PATH", path)
    template = tmp_path / "hardcoded.sh"
    template.write_text(
        "#!/bin/bash\n#SBATCH -p short\n\nsite_launch -n $num_processes castep.mpi $seed_name\n",
        encoding="utf-8",
    )
    config = _scheduler_config(
        castep_executable="/configured/but/unused/castep",
        template_script_path=str(template),
        keyword_shell_init="",
    )

    script, dump = _run_scheduler_script(tmp_path, config)

    # Launcher and executable are both read off the template's hard-coded line.
    assert "env MUFINDER_PARALLEL=1 bash mufinder_help_loop.sh site_launch -n 1 castep.mpi || true" in script
    _assert_complete_dump(dump)
    job_log = (tmp_path / "work dir" / "keyword_generation.log").read_text(encoding="utf-8")
    tried = [line.removeprefix("mufinder: trying ") for line in job_log.splitlines() if line.startswith("mufinder: trying ")]
    assert tried == spec["tried"]
    assert len(log.read_text(encoding="utf-8").splitlines()) == spec["launches"]


@pytest.mark.skipif(sys.platform == "win32", reason="the scheduler runner is a POSIX shell script")
def test_scheduler_job_without_a_usable_template_falls_back_to_a_generic_script(
    tmp_path: Path,
    monkeypatch,
) -> None:
    executable = _write_fake_castep(tmp_path / "castep")
    no_run_line = tmp_path / "odd.sh"
    no_run_line.write_text("#!/bin/bash\n#SBATCH -p short\necho hello\n", encoding="utf-8")

    for template_path in (str(no_run_line), str(tmp_path / "missing.sh")):
        config = _scheduler_config(
            castep_executable=str(executable),
            template_script_path=template_path,
            keyword_shell_init="export MUFINDER_TEST=1",
        )
        assert scripts_module.job_template_plan(config) is None
        script = scripts_module.scheduler_runner_script(
            config=config,
            remote_directory="/work",
            remote_dump_path="/work/keyword_help.txt",
            remote_log_path="/work/keyword_generation.log",
        )
        # Slurm ignores #SBATCH lines that follow a command.
        assert script.startswith(
            "#!/bin/bash\n#SBATCH -J mufinder-keywords\n#SBATCH -n 1\n#SBATCH -N 1\n#SBATCH -t 01:00:00\n\ncd /work\n"
        )
        assert "export MUFINDER_TEST=1" in script

    # With no template to read a launcher from, Slurm's own is the fallback
    # for an MPI build that will not start bare.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    picky = bin_dir / "castep.mpi"
    picky.write_text(
        f'#!/bin/bash\nif [ -z "$LAUNCHED" ]; then echo "MPI_Init failed" >&2; exit 1; fi\nexec {str(executable)!r} "$@"\n',
        encoding="utf-8",
    )
    picky.chmod(0o755)
    srun = bin_dir / "srun"
    srun.write_text('#!/bin/bash\nshift 2\nLAUNCHED=1 exec "$@"\n', encoding="utf-8")
    srun.chmod(0o755)
    _write_timeout(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    config = _scheduler_config(
        castep_executable="castep.mpi",
        template_script_path=str(no_run_line),
        keyword_shell_init="",
    )
    script, dump = _run_scheduler_script(tmp_path, config)
    assert "mufinder_try 'a single job step' srun -n 1 bash mufinder_help_loop.sh castep.mpi; then" in script
    _assert_complete_dump(dump)
    job_log = (tmp_path / "work dir" / "keyword_generation.log").read_text(encoding="utf-8")
    assert "mufinder: trying a single job step" in job_log
    assert "mufinder: trying the bare CASTEP executable" not in job_log


def test_login_node_setup_is_inferred_from_the_job_template(tmp_path: Path) -> None:
    """Direct mode reuses the template's environment setup and executable."""

    scarf = _scheduler_config(keyword_shell_init="", keyword_execution_mode="direct")
    plan = scripts_module.job_template_plan(scarf)
    assert plan.launcher == ("srun", "-n", "1")
    assert plan.executable == "castep.mpi"
    assert plan.setup_lines == ("module load CASTEP",)
    assert scripts_module.login_shell_init(scarf) == "module load CASTEP"
    # On the login node the job launcher is not used.
    assert scripts_module.build_castep_command_with_launcher(scarf, "-help", "all") == "castep.mpi -help all"

    # An explicit shell init always wins over the inferred one.
    explicit = _scheduler_config(keyword_shell_init="module load CASTEP/25")
    assert scripts_module.login_shell_init(explicit) == "module load CASTEP/25"

    # No template at all (addqueue): nothing to infer, the config is used as is.
    redwood = _direct_config(queue_system="addqueue", castep_executable="/mnt/redwoodfs/muon/castep.mpi")
    assert scripts_module.job_template_plan(redwood) is None
    assert scripts_module.login_shell_init(redwood) == ""
    assert scripts_module.build_castep_command_with_launcher(redwood, "--version") == (
        "mpirun -np 1 /mnt/redwoodfs/muon/castep.mpi --version"
    )

    # A run line that cannot be tokenised, or names no CASTEP binary, falls
    # back to the configured executable and the scheduler's usual launcher.
    for run_line in ("srun 'unterminated $seed_name", "run_my_code $seed_name"):
        template = tmp_path / "template.sh"
        template.write_text(f"#!/bin/bash\nmodule load X\n{run_line}\n", encoding="utf-8")
        plan = scripts_module.job_template_plan(
            _scheduler_config(template_script_path=str(template), castep_executable="/opt/castep")
        )
        assert (plan.launcher, plan.executable) == (("srun", "-n", "1"), "/opt/castep")
        assert plan.setup_lines == ("module load X",)


def test_direct_mode_runs_inferred_setup_on_the_login_node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator_module, "SSHConnection", _DirectSSH)
    monkeypatch.setattr(_DirectSSH, "failing", frozenset())

    generate_cluster_castep_keywords_json(
        config=_scheduler_config(keyword_shell_init="", keyword_execution_mode="direct"),
        output_path=tmp_path / "castep_keywords.json",
    )

    assert all("module load CASTEP" in command for command in _DirectSSH.last.commands)


# --- helpers ------------------------------------------------------------------


def test_build_shell_command_applies_working_directory_and_shell_init() -> None:
    wrapped = scripts_module.build_shell_command(
        command="castep.mpi --version",
        working_directory="/scratch/alice",
        shell_init="module load CASTEP",
    )

    assert wrapped.startswith("/bin/bash -lc ")
    assert "cd /scratch/alice" in wrapped
    assert "module load CASTEP" in wrapped
    assert "castep.mpi --version" in wrapped


def test_build_shell_command_drops_unset_working_directory_and_shell_init() -> None:
    wrapped = scripts_module.build_shell_command(command="castep.mpi --version", shell_init="  \n")

    assert shlex.split(wrapped) == [
        "/bin/bash",
        "-lc",
        "set -e\nsource /etc/profile >/dev/null 2>&1 || true\ncastep.mpi --version",
    ]


_SYNTAX_CHECK_TEMPLATE = (
    "#!/bin/bash\n#SBATCH -J $job_name\n#SBATCH -A muons\n\nexport SITE_SETUP=1\ncd $working_directory\n$castep_command\n"
)


def _rendered_keyword_scripts(tmp_path: Path) -> dict[str, str]:
    """Return every script cluster keyword generation can send, by a descriptive name."""

    template = tmp_path / "site.sh"
    template.write_text(_SYNTAX_CHECK_TEMPLATE, encoding="utf-8")
    configs = {
        "scarf": _scheduler_config(),
        "site-template": _scheduler_config(template_script_path=str(template), castep_executable="/opt/it's/castep"),
        "no-template": _scheduler_config(template_script_path=str(tmp_path / "missing.sh"), keyword_shell_init=""),
        "addqueue": _scheduler_config(queue_system="addqueue", keyword_shell_init="module load mpi\n"),
        "no-launcher": _scheduler_config(queue_system="pbs"),
        "direct": _direct_config(),
        "direct-addqueue": _direct_config(queue_system="addqueue"),
    }
    scripts: dict[str, str] = {"job help loop": scripts_module._job_help_loop_script()}
    for name, config in configs.items():
        castep = scripts_module.build_castep_command_with_launcher(config)
        for label, command in (
            ("list", scripts_module.build_castep_command_with_launcher(config, "-help", "all")),
            ("batch", scripts_module.help_loop_command(castep, ["TASK", "LATTICE_CART"])),
        ):
            wrapped = scripts_module.build_shell_command(
                command=command,
                working_directory="/scratch/a b",
                shell_init=scripts_module.login_shell_init(config),
            )
            scripts[f"{name} login {label}"] = shlex.split(wrapped)[2]
        if config.keyword_execution_mode == "scheduler":
            scripts[f"{name} job"] = scripts_module.scheduler_runner_script(
                config=config,
                remote_directory="/work dir/job",
                remote_dump_path="/work dir/job/keyword_help.txt",
                remote_log_path="/work dir/job/keyword_generation.log",
            )
    return scripts


def test_every_rendered_keyword_script_is_valid_bash(tmp_path: Path) -> None:
    """A template edit that breaks the shell syntax of any script fails here."""

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    scripts = _rendered_keyword_scripts(tmp_path)
    # Each strategy is covered: no launcher, job steps, and inside a launcher step.
    assert "mufinder_try 'the CASTEP executable'" in scripts["no-launcher job"]
    assert "mufinder_try 'a single job step'" in scripts["site-template job"]
    assert "--standalone" in scripts["addqueue job"]

    for name, script in scripts.items():
        path = tmp_path / "script.sh"
        path.write_text(script, encoding="utf-8")
        completed = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
        assert completed.returncode == 0, f"{name}:\n{completed.stderr}"


def test_run_remote_checked_prefers_stderr_then_stdout_then_command() -> None:
    class _SSH:
        def __init__(self, result) -> None:
            self.result = result

        def exec_command(self, _command: str, timeout: int = 30):
            return self.result

    for result, expected in (
        ((1, "out", "err"), "err"),
        ((1, "out", ""), "out"),
        ((1, "", ""), "Remote command failed: castep"),
    ):
        with pytest.raises(ClusterKeywordGenerationError, match=expected):
            generator_module._run_remote_checked(_SSH(result), _direct_config(), "castep")


def test_version_detection_is_best_effort() -> None:
    class _SSH:
        def __init__(self, result) -> None:
            self.result = result

        def exec_command(self, _command: str, timeout: int = 30):
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    detect = generator_module._detect_remote_castep_version
    assert detect(_SSH((0, " CASTEP version: 24.10 \n", "")), _direct_config()) == "24.10"
    assert detect(_SSH((0, "", "CASTEP version 23.1.1")), _direct_config()) == "23.1.1"
    assert detect(_SSH((0, "no version here", "")), _direct_config()) == ""
    assert detect(_SSH(TimeoutError("slow")), _direct_config()) == ""


def test_parse_keyword_doc_edge_cases() -> None:
    parse = generator_module._parse_keyword_doc
    assert parse("TASK", "no type information") is None
    # A keyword reported for both file kinds cannot be assigned to one.
    assert parse("TASK", "Type: String\nLevel: Basic\n\nDoc\n\n") is None
    assert parse("TASK", help_keyword_text("TASK"))[0] == "param"
    assert parse("LATTICE_CART", help_keyword_text("LATTICE_CART"))[0] == "cell"


def test_remote_log_excerpt_skips_markers_and_falls_through_empty_files() -> None:
    ssh = _SchedulerSSH(None)
    ssh.remote_files["/job/log"] = b"\n\n"
    ssh.remote_files["/job/dump"] = b"@@MUFINDER_VERSION_BEGIN@@\nmpirun: not found\r\n@@MUFINDER_VERSION_END@@\n"

    assert generator_module._remote_log_excerpt(ssh, "/job/missing", "/job/log", "/job/dump") == "mpirun: not found"
    assert generator_module._remote_log_excerpt(ssh, "/job/missing") == ""


def test_keyword_job_walltime_is_short_whatever_the_cluster_default() -> None:
    walltime = scripts_module.scheduler_walltime
    # A production-length request would only hold the job back in the queue.
    assert walltime(_scheduler_config(default_walltime="12:00:00")) == "01:00:00"
    assert walltime(_scheduler_config(default_walltime="")) == "01:00:00"
    assert walltime(_scheduler_config(default_walltime="", queue_system="addqueue")) == ""
    assert walltime(_scheduler_config(default_walltime="02:00:00", queue_system="addqueue")) == "02:00:00"


def test_emit_progress_and_check_cancel_helpers() -> None:
    generator_module._emit_progress(None, "ignored")
    generator_module._check_cancel(None)
    generator_module._check_cancel(lambda: False)
    with pytest.raises(ClusterKeywordGenerationCancelled):
        generator_module._check_cancel(lambda: True)


# --- worker -------------------------------------------------------------------


def _worker(tmp_path: Path) -> tuple[ClusterKeywordGenerationWorker, dict[str, list]]:
    worker = ClusterKeywordGenerationWorker(
        config=_direct_config(),
        output_path=tmp_path / "castep_keywords.json",
        password="secret",
    )
    events: dict[str, list] = {"finished": [], "failed": [], "cancelled": [], "progress": []}
    worker.finished.connect(events["finished"].append)
    worker.failed.connect(events["failed"].append)
    worker.cancelled.connect(lambda: events["cancelled"].append(True))
    worker.progress_changed.connect(lambda message, percent: events["progress"].append((message, percent)))
    return worker, events


def test_worker_run_success_emits_finished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_generate(*, output_path, progress, cancel_check, **_kwargs):
        progress("working", 50)
        assert cancel_check() is False
        return Path(output_path)

    monkeypatch.setattr(generator_module, "generate_cluster_castep_keywords_json", fake_generate)
    worker, events = _worker(tmp_path)

    worker.run()

    assert events["finished"] == [str(tmp_path / "castep_keywords.json")]
    assert events["progress"] == [("working", 50)]
    assert events["failed"] == [] and events["cancelled"] == []


def test_worker_run_cancelled_emits_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_generate(*, cancel_check, **_kwargs):
        assert cancel_check() is True
        raise ClusterKeywordGenerationCancelled("cancelled")

    monkeypatch.setattr(generator_module, "generate_cluster_castep_keywords_json", fake_generate)
    worker, events = _worker(tmp_path)
    worker.cancel()

    worker.run()

    assert events["cancelled"] == [True]
    assert events["finished"] == [] and events["failed"] == []


def test_worker_run_failure_emits_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_generate(**_kwargs):
        raise RuntimeError("ssh exploded")

    monkeypatch.setattr(generator_module, "generate_cluster_castep_keywords_json", fake_generate)
    worker, events = _worker(tmp_path)

    worker.run()

    assert events["failed"] == ["ssh exploded"]
