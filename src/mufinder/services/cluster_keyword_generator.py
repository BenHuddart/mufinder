"""Cluster-backed CASTEP keyword generation helpers."""

from __future__ import annotations

import json
from pathlib import Path
import posixpath
import re
import shlex
import tempfile
import time
from typing import Callable
from uuid import uuid4

from PySide6.QtCore import QObject, Signal, Slot

from mufinder.cluster_setup import ClusterConfig
from mufinder.domain.enums import JobStatus
from mufinder.services.scheduler import SchedulerSubmitRequest, create_scheduler
from mufinder.services.ssh_client import SSHConnection


ProgressCallback = Callable[[str, int | None], None]
CancelCheck = Callable[[], bool]

_HELP_OPTION_RE = re.compile(r"((?<=^ )[A-Z_\d]{2,}|(?<=^)[A-Z_\d]{2,})", re.MULTILINE)
_TYPE_DOC_RE = re.compile(
    r"(?P<before_type>.*)Type: (?P<type>.+?)\s+"
    r"Level: (?P<level>[^ ]+)\n\s*\n"
    r"(?P<doc>.*?)(\n\s*\n|$)",
    re.DOTALL,
)
_CASTEP_VERSION_RE = re.compile(r"CASTEP version[: ]+([0-9.]+)")
_NONE_FOUND_PARAM_RE = re.compile(r"PARAMETERS keywords:\n\n\s?None found")
_NONE_FOUND_CELL_RE = re.compile(r"CELL keywords:\n\n\s?None found")
_DEFAULT_SCHEDULER_WALLTIME = "00:10:00"
_REMOTE_JSON_FILENAME = "castep_keywords.json"
_REMOTE_GENERATOR_NAME = "generate_keywords.py"
_REMOTE_RUNNER_NAME = "run_keywords.sh"
_REMOTE_LOG_NAME = "keyword_generation.log"
_PROGRESS_LINE_RE = re.compile(r"^\s*Processed:\s*\[[# ]+\]\s*\d+%\s*$")

_REMOTE_GENERATOR_SCRIPT = r"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

HELP_OPTION_RE = re.compile(r"((?<=^ )[A-Z_\d]{2,}|(?<=^)[A-Z_\d]{2,})", re.MULTILINE)
TYPE_DOC_RE = re.compile(
    r"(?P<before_type>.*)Type: (?P<type>.+?)\s+"
    r"Level: (?P<level>[^ ]+)\n\s*\n"
    r"(?P<doc>.*?)(\n\s*\n|$)",
    re.DOTALL,
)
CASTEP_VERSION_RE = re.compile(r"CASTEP version[: ]+([0-9.]+)")
NONE_FOUND_PARAM_RE = re.compile(r"PARAMETERS keywords:\n\n\s?None found")
NONE_FOUND_CELL_RE = re.compile(r"CELL keywords:\n\n\s?None found")


def run(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
    )
    stdout, stderr = process.communicate()
    return int(process.returncode), stdout.strip(), stderr.strip()


def build_command(executable, *args):
    launcher = os.environ.get("MUFINDER_CASTEP_LAUNCHER", "").strip()
    launcher_parts = shlex.split(launcher) if launcher else []
    parts = [*(shlex_quote(part) for part in launcher_parts), shlex_quote(executable), *(shlex_quote(arg) for arg in args)]
    return " ".join(parts)


def shlex_quote(value):
    import shlex

    return shlex.quote(value)


def detect_version(executable):
    exit_code, stdout, stderr = run(build_command(executable, "--version"))
    if exit_code != 0:
        return ""
    match = CASTEP_VERSION_RE.search(stdout) or CASTEP_VERSION_RE.search(stderr)
    if not match:
        return ""
    try:
        return float(match.group(1))
    except ValueError:
        return match.group(1)


def run_checked(executable, *args):
    command = build_command(executable, *args)
    exit_code, stdout, stderr = run(command)
    if exit_code != 0:
        details = stderr.strip() or stdout.strip() or f"Command failed: {command}"
        raise RuntimeError(details)
    return stdout


def parse_doc(option, doc):
    match = TYPE_DOC_RE.match(doc)
    if match is None:
        return None
    suffix = None
    if NONE_FOUND_PARAM_RE.findall(doc):
        suffix = "cell"
    if NONE_FOUND_CELL_RE.findall(doc):
        suffix = "param"
    if suffix is None:
        return None
    groups = match.groupdict()
    option_name = option.lower()
    return (
        suffix,
        {
            "keyword": option_name,
            "option_type": groups["type"],
            "level": groups["level"],
            "docstring": groups["doc"],
        },
    )


def main():
    executable = sys.argv[1]
    output_path = Path(sys.argv[2]).expanduser()
    help_all = run_checked(executable, "-help", "all")
    raw_options = HELP_OPTION_RE.findall(help_all)
    if not raw_options:
        raise RuntimeError("CASTEP help output did not contain any keywords.")

    payload = {"cell": {}, "param": {}, "types": [], "levels": [], "castep_version": detect_version(executable)}
    types = set()
    levels = set()
    last_percent = -1

    total = len(raw_options)
    for index, option in enumerate(raw_options, start=1):
        doc = run_checked(executable, "-help", option)
        parsed = parse_doc(option, doc)
        if parsed is None:
            continue
        suffix, entry = parsed
        payload[suffix][entry["keyword"]] = entry
        types.add(entry["option_type"])
        levels.add(entry["level"])
        percent = int((index / total) * 100)
        if percent != last_percent:
            sys.stdout.write(f"Processed: [{'#' * int(percent / 5):<20}] {percent:>3d}%\n")
            sys.stdout.flush()
            last_percent = percent

    payload["types"] = sorted(types)
    payload["levels"] = sorted(levels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".strip()


class ClusterKeywordGenerationError(RuntimeError):
    """Raised when remote CASTEP keyword generation fails."""


class ClusterKeywordGenerationCancelled(ClusterKeywordGenerationError):
    """Raised when remote CASTEP keyword generation is cancelled."""


def _effective_keyword_mode(config: ClusterConfig) -> str:
    mode = config.keyword_execution_mode.strip().lower() or "direct"
    if config.queue_system.strip().lower() == "addqueue":
        return "scheduler"
    return mode


def validate_cluster_keyword_config(config: ClusterConfig) -> list[str]:
    """Return blocking issues for cluster-backed keyword generation."""

    issues: list[str] = []
    if not config.hostname.strip():
        issues.append("Cluster hostname is required.")
    if not config.username.strip():
        issues.append("Cluster username is required.")
    if not config.castep_executable.strip():
        issues.append("CASTEP executable is required.")

    mode = _effective_keyword_mode(config)
    if mode == "scheduler":
        if not config.default_working_directory.strip():
            issues.append("Default working directory is required for scheduler keyword generation.")
        if not config.keyword_python_command.strip():
            issues.append("Keyword Python command is required for scheduler keyword generation.")
        executable = config.castep_executable.strip()
        if "/" not in executable and not config.keyword_shell_init.strip():
            issues.append(
                "Keyword shell init is required when scheduler keyword generation uses a module-provided CASTEP command."
            )
    return issues


def _emit_progress(callback: ProgressCallback | None, message: str, percent: int | None = None) -> None:
    if callback is not None:
        callback(message, percent)


def _check_cancel(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise ClusterKeywordGenerationCancelled("CASTEP keyword generation cancelled.")


def _build_shell_command(
    *,
    command: str,
    working_directory: str = "",
    shell_init: str = "",
) -> str:
    script_lines = [
        "set -e",
        "source /etc/profile >/dev/null 2>&1 || true",
    ]
    if working_directory.strip():
        script_lines.append(f"cd {shlex.quote(working_directory.strip())}")
    if shell_init.strip():
        script_lines.append(shell_init.rstrip())
    script_lines.append(command)
    return f"/bin/bash -lc {shlex.quote(chr(10).join(script_lines))}"


def _run_remote_checked(
    ssh: SSHConnection,
    config: ClusterConfig,
    command: str,
    *,
    timeout: int = 60,
    working_directory: str = "",
) -> str:
    wrapped = _build_shell_command(
        command=command,
        working_directory=working_directory,
        shell_init=config.keyword_shell_init,
    )
    exit_code, stdout, stderr = ssh.exec_command(wrapped, timeout=timeout)
    if exit_code != 0:
        details = stderr.strip() or stdout.strip() or f"Remote command failed: {command}"
        raise ClusterKeywordGenerationError(details)
    return stdout


def _keyword_castep_launcher(config: ClusterConfig) -> str:
    if config.queue_system.strip().lower() == "addqueue":
        return "mpirun -np 1"
    return ""


def _build_castep_command_with_launcher(config: ClusterConfig, *args: str) -> str:
    launcher = _keyword_castep_launcher(config)
    launcher_parts = shlex.split(launcher) if launcher else []
    executable = config.castep_executable.strip()
    parts = [*launcher_parts, executable, *args]
    return " ".join(shlex.quote(part) for part in parts)


def _detect_remote_castep_version(ssh: SSHConnection, config: ClusterConfig) -> float | str:
    try:
        stdout = _run_remote_checked(
            ssh,
            config,
            _build_castep_command_with_launcher(config, "--version"),
            timeout=30,
        )
    except ClusterKeywordGenerationError:
        return ""

    match = _CASTEP_VERSION_RE.search(stdout)
    if not match:
        return ""
    try:
        return float(match.group(1))
    except ValueError:
        return match.group(1)


def _parse_keyword_doc(option: str, doc: str) -> tuple[str, dict[str, str]] | None:
    match = _TYPE_DOC_RE.match(doc)
    if match is None:
        return None

    suffix = None
    if _NONE_FOUND_PARAM_RE.findall(doc):
        suffix = "cell"
    if _NONE_FOUND_CELL_RE.findall(doc):
        suffix = "param"
    if suffix is None:
        return None

    groups = match.groupdict()
    option_name = option.lower()
    return (
        suffix,
        {
            "keyword": option_name,
            "option_type": groups["type"],
            "level": groups["level"],
            "docstring": groups["doc"],
        },
    )


def _build_remote_keyword_payload(
    ssh: SSHConnection,
    config: ClusterConfig,
    *,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, object]:
    _emit_progress(progress, f"Querying CASTEP help on {config.name}...", 5)
    help_all = _run_remote_checked(
        ssh,
        config,
        _build_castep_command_with_launcher(config, "-help", "all"),
        timeout=120,
    )
    raw_options = _HELP_OPTION_RE.findall(help_all)
    if not raw_options:
        raise ClusterKeywordGenerationError("CASTEP help output did not contain any keywords.")

    payload: dict[str, object] = {
        "cell": {},
        "param": {},
        "types": [],
        "levels": [],
        "castep_version": _detect_remote_castep_version(ssh, config),
    }
    types: set[str] = set()
    levels: set[str] = set()

    for index, option in enumerate(raw_options, start=1):
        _check_cancel(cancel_check)
        doc = _run_remote_checked(
            ssh,
            config,
            _build_castep_command_with_launcher(config, "-help", option),
            timeout=120,
        )
        parsed = _parse_keyword_doc(option, doc)
        if parsed is None:
            continue
        suffix, entry = parsed
        entries = payload[suffix]
        assert isinstance(entries, dict)
        entries[entry["keyword"]] = entry
        types.add(entry["option_type"])
        levels.add(entry["level"])
        percent = max(5, min(100, int((index / len(raw_options)) * 100)))
        _emit_progress(progress, f"Generating CASTEP keywords... {percent}%", percent)

    payload["types"] = sorted(types)
    payload["levels"] = sorted(levels)
    return payload


def _write_payload(payload: dict[str, object], output_path: Path) -> Path:
    output = output_path.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return output.resolve()


def _scheduler_walltime(config: ClusterConfig) -> str:
    walltime = config.default_walltime.strip()
    if walltime:
        return walltime
    if config.queue_system.strip().lower() == "addqueue":
        return ""
    return _DEFAULT_SCHEDULER_WALLTIME


def _scheduler_runner_script(
    *,
    config: ClusterConfig,
    remote_directory: str,
    remote_generator_path: str,
    remote_json_path: str,
    remote_log_path: str,
) -> str:
    lines = [
        "#!/bin/bash",
        "set -e",
    ]
    if config.queue_system.strip().lower() == "slurm":
        lines.extend(
            [
                "#SBATCH -J mufinder-keywords",
                "#SBATCH -n 1",
                "#SBATCH -N 1",
                f"#SBATCH -t {_scheduler_walltime(config)}",
                "",
            ]
        )
    lines.append(f"cd {shlex.quote(remote_directory)}")
    if config.keyword_shell_init.strip():
        lines.append(config.keyword_shell_init.rstrip())
    launcher = _keyword_castep_launcher(config)
    if launcher:
        lines.append(f"export MUFINDER_CASTEP_LAUNCHER={shlex.quote(launcher)}")
    python_command = config.keyword_python_command.strip() or "python3"
    command = " ".join(
        [
            shlex.quote(python_command),
            shlex.quote(remote_generator_path),
            shlex.quote(config.castep_executable.strip()),
            shlex.quote(remote_json_path),
            ">",
            shlex.quote(remote_log_path),
            "2>&1",
        ]
    )
    lines.append(command)
    lines.append("")
    return "\n".join(lines)


def _remote_log_excerpt(ssh: SSHConnection, remote_log_path: str) -> str:
    if not ssh.exists(remote_log_path):
        return ""
    text = ssh.read_text(remote_log_path)
    if not text:
        return ""

    normalized = text.replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.splitlines()]
    non_progress = [line for line in lines if not _PROGRESS_LINE_RE.match(line.strip()) and line.strip()]
    if non_progress:
        excerpt = "\n".join(non_progress[-80:])
    else:
        excerpt = "\n".join(line for line in lines[-80:] if line.strip())

    excerpt = excerpt.strip()
    return excerpt[-4000:] if excerpt else ""


def _scheduler_python_probe_command(python_command: str) -> str:
    quoted_python = shlex.quote(python_command)
    script = "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"
    return f"{quoted_python} -c {shlex.quote(script)}"


def _parse_python_version(raw: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", raw)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _ensure_scheduler_python_compatible(
    ssh: SSHConnection,
    config: ClusterConfig,
) -> None:
    python_command = config.keyword_python_command.strip() or "python3"
    probe = _scheduler_python_probe_command(python_command)
    wrapped = _build_shell_command(command=probe, shell_init=config.keyword_shell_init)
    exit_code, stdout, stderr = ssh.exec_command(wrapped, timeout=30)
    combined = "\n".join(part.strip() for part in [stdout, stderr] if part.strip()).strip()
    if exit_code != 0:
        details = combined or "probe command failed"
        raise ClusterKeywordGenerationError(
            f"Keyword Python preflight failed for `{python_command}` on {config.name}: {details}"
        )

    parsed = _parse_python_version(combined)
    if parsed is None:
        raise ClusterKeywordGenerationError(
            f"Keyword Python preflight could not parse a version for `{python_command}` on {config.name}. "
            f"Output: {combined or '(empty)'}"
        )

    major, minor, _patch = parsed
    if (major, minor) < (3, 6):
        raise ClusterKeywordGenerationError(
            f"Keyword Python preflight requires Python >= 3.6, but `{python_command}` on {config.name} "
            f"reported {major}.{minor}."
        )


def _annotation_compatibility_hint(log_excerpt: str, python_command: str) -> str:
    if not log_excerpt.strip():
        return ""

    lowered = log_excerpt.lower()
    markers = (
        "unsupported operand type(s) for |",
        "typeerror",
        "def detect_version(",
        "syntaxerror",
        "invalid syntax",
    )
    if any(marker in lowered for marker in markers):
        return (
            "Compatibility hint: cluster Python may be older than expected for the helper script. "
            f"Check `keyword_python_command` (currently `{python_command}`) and try a newer interpreter."
        )
    return ""


def generate_cluster_castep_keywords_json(
    *,
    config: ClusterConfig,
    output_path: str | Path,
    password: str | None = None,
    settings=None,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> Path:
    """Generate CASTEP keyword JSON using a remote cluster."""

    issues = validate_cluster_keyword_config(config)
    if issues:
        raise ClusterKeywordGenerationError("\n".join(f"- {issue}" for issue in issues))

    output = Path(output_path).expanduser()
    if output.suffix.lower() != ".json":
        raise ClusterKeywordGenerationError("CASTEP keywords output path must end with .json.")

    ssh = SSHConnection(config, settings=settings)
    _emit_progress(progress, f"Connecting to {config.name}...", 0)
    ssh.connect(password=password)
    try:
        _check_cancel(cancel_check)
        mode = _effective_keyword_mode(config)
        if mode == "scheduler":
            if config.queue_system.strip().lower() == "addqueue":
                _emit_progress(progress, f"Using {config.queue_system.upper()} job submission for keyword generation.", 5)
            return _generate_scheduler_keywords_json(
                ssh=ssh,
                config=config,
                output_path=output,
                progress=progress,
                cancel_check=cancel_check,
            )
        return _generate_direct_keywords_json(
            ssh=ssh,
            config=config,
            output_path=output,
            progress=progress,
            cancel_check=cancel_check,
        )
    finally:
        ssh.disconnect()


def _generate_direct_keywords_json(
    *,
    ssh: SSHConnection,
    config: ClusterConfig,
    output_path: Path,
    progress: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> Path:
    payload = _build_remote_keyword_payload(
        ssh,
        config,
        progress=progress,
        cancel_check=cancel_check,
    )
    _check_cancel(cancel_check)
    _emit_progress(progress, f"Writing CASTEP keywords to {output_path.name}...", 100)
    return _write_payload(payload, output_path)


def _generate_scheduler_keywords_json(
    *,
    ssh: SSHConnection,
    config: ClusterConfig,
    output_path: Path,
    progress: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> Path:
    _ensure_scheduler_python_compatible(ssh, config)
    scheduler = create_scheduler(
        config.queue_system,
        submit_command_path=config.submit_command_path,
        queue_check_command_path=config.queue_check_command_path,
        cancel_command_path=config.cancel_command_path,
    )
    remote_directory = posixpath.join(
        config.default_working_directory.strip(),
        f".mufinder-keywords-{uuid4().hex}",
    )
    remote_generator_path = posixpath.join(remote_directory, _REMOTE_GENERATOR_NAME)
    remote_runner_path = posixpath.join(remote_directory, _REMOTE_RUNNER_NAME)
    remote_json_path = posixpath.join(remote_directory, _REMOTE_JSON_FILENAME)
    remote_log_path = posixpath.join(remote_directory, _REMOTE_LOG_NAME)
    ssh.mkdir(remote_directory, exist_ok=True)

    remote_job_id = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".py", delete=False) as handle:
            handle.write(_REMOTE_GENERATOR_SCRIPT)
            generator_path = Path(handle.name)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".sh", delete=False) as handle:
            handle.write(
                _scheduler_runner_script(
                    config=config,
                    remote_directory=remote_directory,
                    remote_generator_path=remote_generator_path,
                    remote_json_path=remote_json_path,
                    remote_log_path=remote_log_path,
                )
            )
            runner_path = Path(handle.name)
        try:
            _emit_progress(progress, f"Uploading keyword generation helper to {config.name}...", 10)
            ssh.put(generator_path, remote_generator_path)
            ssh.put(runner_path, remote_runner_path)
            ssh.chmod(remote_runner_path, 0o755)
        finally:
            generator_path.unlink(missing_ok=True)
            runner_path.unlink(missing_ok=True)

        request = SchedulerSubmitRequest(
            working_dir=remote_directory,
            seed_name=remote_runner_path if config.queue_system.strip().lower() == "addqueue" else "keywords",
            num_processes=1,
            cores_per_node=1,
            walltime=_scheduler_walltime(config),
            castep_executable="/bin/bash" if config.queue_system.strip().lower() == "addqueue" else "",
            script_path=remote_runner_path if config.queue_system.strip().lower() == "slurm" else "",
            memory_gb=config.default_memory_gb,
            queue_name=config.default_queue_name,
            comment="MuFinder CASTEP keywords",
        )

        _check_cancel(cancel_check)
        _emit_progress(progress, f"Submitting keyword generation job on {config.name}...", 20)
        remote_job_id = scheduler.submit(ssh, request)
        _emit_progress(progress, f"Waiting for cluster job {remote_job_id}...", 30)

        while True:
            _check_cancel(cancel_check)
            statuses = scheduler.status(ssh, [remote_job_id])
            info = statuses.get(remote_job_id)
            mapped = JobStatus.COMPLETED if info is None else scheduler.map_to_job_status(info.state)
            if mapped == JobStatus.QUEUED:
                _emit_progress(progress, f"Keyword job {remote_job_id} is queued...", 35)
            elif mapped == JobStatus.RUNNING:
                _emit_progress(progress, f"Keyword job {remote_job_id} is running...", 60)
            elif mapped == JobStatus.COMPLETED:
                break
            elif mapped == JobStatus.CANCELLED:
                raise ClusterKeywordGenerationCancelled("Cluster keyword generation job was cancelled.")
            elif mapped in {JobStatus.FAILED, JobStatus.STOPPED}:
                details = _remote_log_excerpt(ssh, remote_log_path)
                message = f"Cluster keyword generation job failed on {config.name}."
                hint = _annotation_compatibility_hint(
                    details,
                    config.keyword_python_command.strip() or "python3",
                )
                if hint:
                    message = f"{message}\n\n{hint}"
                if details:
                    message = f"{message}\n\n{details}"
                raise ClusterKeywordGenerationError(message)
            time.sleep(2)

        _emit_progress(progress, f"Downloading CASTEP keywords from {config.name}...", 90)
        if not ssh.exists(remote_json_path):
            details = _remote_log_excerpt(ssh, remote_log_path)
            message = "Cluster keyword generation finished without producing castep_keywords.json."
            hint = _annotation_compatibility_hint(
                details,
                config.keyword_python_command.strip() or "python3",
            )
            if hint:
                message = f"{message}\n\n{hint}"
            if details:
                message = f"{message}\n\n{details}"
            raise ClusterKeywordGenerationError(message)
        ssh.get(remote_json_path, output_path)
        _emit_progress(progress, "CASTEP keyword generation complete.", 100)
        return output_path.resolve()
    except ClusterKeywordGenerationCancelled:
        if remote_job_id:
            try:
                scheduler.cancel(ssh, remote_job_id)
            except Exception:
                pass
        raise
    finally:
        try:
            ssh.rmtree(remote_directory)
        except Exception:
            pass


class ClusterKeywordGenerationWorker(QObject):
    """Run cluster keyword generation off the UI thread."""

    progress_changed = Signal(str, object)
    finished = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        *,
        config: ClusterConfig,
        output_path: str | Path,
        password: str | None = None,
        settings=None,
    ) -> None:
        super().__init__()
        self._config = config
        self._output_path = Path(output_path)
        self._password = password
        self._settings = settings
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    @Slot()
    def run(self) -> None:
        try:
            output = generate_cluster_castep_keywords_json(
                config=self._config,
                output_path=self._output_path,
                password=self._password,
                settings=self._settings,
                progress=self._on_progress,
                cancel_check=lambda: self._cancel_requested,
            )
        except ClusterKeywordGenerationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        else:
            self.finished.emit(str(output))

    def _on_progress(self, message: str, percent: int | None) -> None:
        self.progress_changed.emit(message, percent)
