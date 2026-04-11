"""Branch coverage for cluster keyword generator helper functions."""

from __future__ import annotations

import pytest

import mufinder.services.cluster_keyword_generator as generator_module
from mufinder.cluster_setup import ClusterConfig
from mufinder.services.cluster_keyword_generator import (
    ClusterKeywordGenerationCancelled,
    ClusterKeywordGenerationError,
)


def _config(*, queue_system: str = "SLURM", walltime: str = "", shell_init: str = "") -> ClusterConfig:
    return ClusterConfig(
        name="SCARF",
        hostname="ui1.scarf.rl.ac.uk",
        username="alice",
        auth_method="key",
        default_working_directory="/work/alice",
        queue_system=queue_system,
        castep_executable="castep.mpi",
        default_walltime=walltime,
        keyword_shell_init=shell_init,
    )


class _ExecSSH:
    def __init__(self, exit_code: int, stdout: str, stderr: str) -> None:
        self._exit_code = exit_code
        self._stdout = stdout
        self._stderr = stderr
        self.commands: list[str] = []

    def exec_command(self, command: str, timeout: int = 60):
        _ = timeout
        self.commands.append(command)
        return self._exit_code, self._stdout, self._stderr


class _LogSSH:
    def __init__(self, text: str, exists: bool = True) -> None:
        self._text = text
        self._exists = exists

    def exists(self, _path: str) -> bool:
        return self._exists

    def read_text(self, _path: str) -> str:
        return self._text


def test_check_cancel_raises_cancelled_error() -> None:
    with pytest.raises(ClusterKeywordGenerationCancelled):
        generator_module._check_cancel(lambda: True)


def test_build_shell_command_applies_working_directory_and_shell_init() -> None:
    command = generator_module._build_shell_command(
        command="echo hello",
        working_directory="/work/alice",
        shell_init="module load CASTEP",
    )

    assert "cd /work/alice" in command
    assert "module load CASTEP" in command
    assert "echo hello" in command


def test_run_remote_checked_raises_cluster_error_with_best_details() -> None:
    ssh = _ExecSSH(exit_code=1, stdout="", stderr="permission denied")

    with pytest.raises(ClusterKeywordGenerationError, match="permission denied"):
        generator_module._run_remote_checked(ssh, _config(), "whoami")


def test_detect_remote_castep_version_handles_error_and_non_numeric(monkeypatch) -> None:
    ssh = _ExecSSH(exit_code=0, stdout="", stderr="")
    config = _config()

    monkeypatch.setattr(
        generator_module,
        "_run_remote_checked",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ClusterKeywordGenerationError("boom")),
    )
    assert generator_module._detect_remote_castep_version(ssh, config) == ""

    monkeypatch.setattr(
        generator_module,
        "_run_remote_checked",
        lambda *_args, **_kwargs: "CASTEP version: development",
    )
    assert generator_module._detect_remote_castep_version(ssh, config) == ""

    monkeypatch.setattr(
        generator_module,
        "_run_remote_checked",
        lambda *_args, **_kwargs: "CASTEP version: 25.1",
    )
    assert generator_module._detect_remote_castep_version(ssh, config) == 25.1


def test_scheduler_walltime_and_python_parsing_helpers() -> None:
    assert generator_module._scheduler_walltime(_config(queue_system="ADDQUEUE", walltime="")) == ""
    assert generator_module._scheduler_walltime(_config(queue_system="SLURM", walltime="")) == "00:10:00"
    assert generator_module._scheduler_walltime(_config(queue_system="SLURM", walltime="02:00:00")) == "02:00:00"

    assert generator_module._parse_python_version("Python 3.11.7") == (3, 11, 7)
    assert generator_module._parse_python_version("no version here") is None


def test_remote_log_excerpt_filters_progress_and_annotation_hint() -> None:
    log_text = (
        "Processed: [#####               ]  25%\n"
        "Processed: [##########          ]  50%\n"
        "Traceback (most recent call last):\n"
        "TypeError: unsupported operand type(s) for |: 'type' and 'type'\n"
    )
    excerpt = generator_module._remote_log_excerpt(_LogSSH(log_text), "/tmp/log")

    assert "Processed:" not in excerpt
    assert "TypeError" in excerpt
    hint = generator_module._annotation_compatibility_hint(excerpt, "python3")
    assert "Compatibility hint:" in hint
    assert "python3" in hint
    assert generator_module._annotation_compatibility_hint("regular error", "python3") == ""
    assert generator_module._remote_log_excerpt(_LogSSH("", exists=False), "/tmp/log") == ""


def test_emit_progress_and_parse_keyword_doc_edge_cases() -> None:
    updates: list[tuple[str, int | None]] = []
    generator_module._emit_progress(lambda msg, pct: updates.append((msg, pct)), "hello", 42)
    generator_module._emit_progress(None, "ignored")
    assert updates == [("hello", 42)]

    assert generator_module._parse_keyword_doc("TASK", "unstructured help output") is None
    no_suffix_doc = "Type: Integer\nLevel: Basic\n\nNo suffix section here\n\n"
    assert generator_module._parse_keyword_doc("TASK", no_suffix_doc) is None

    assert generator_module._annotation_compatibility_hint("", "python3") == ""
