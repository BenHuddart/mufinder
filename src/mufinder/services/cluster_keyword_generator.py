"""Cluster-backed CASTEP keyword generation helpers.

CASTEP documents its keywords one at a time: ``castep -help all`` lists the
names and ``castep -help <NAME>`` describes one. A full catalog therefore needs
several hundred ``-help`` calls. They are all issued from a small shell loop on
the cluster, which prints every description between marker lines; MuFinder
parses that dump locally. The cluster needs nothing but ``bash`` and CASTEP, and
the number of SSH round trips is a handful instead of one per keyword.

The scripts are built by :mod:`mufinder.services.cluster_keyword_scripts`; this
module runs them over SSH or through the scheduler and parses what they print.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import posixpath
import re
import shlex
import tempfile
import time
from typing import Callable
from uuid import uuid4

from PySide6.QtCore import QObject, Signal, Slot

from mufinder.castep_io.keyword_catalog_file import (
    KeywordCatalogValidationError,
    validate_keyword_catalog_payload,
    write_keyword_catalog_file,
)
from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import JobStatus
from mufinder.services.cluster_keyword_scripts import (
    DUMP_COMPLETE_MARKER,
    HELP_END_MARKER,
    JOB_HELP_LOOP_NAME,
    build_castep_command_with_launcher,
    build_shell_command,
    help_loop_command,
    login_shell_init,
    scheduler_runner_script,
    scheduler_walltime,
)
from mufinder.services.scheduler import SchedulerSubmitRequest, create_scheduler
from mufinder.services.ssh_client import SSHConnection, SSHCredentials


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
_REMOTE_DUMP_FILENAME = "keyword_help.txt"
_REMOTE_RUNNER_NAME = "run_keywords.sh"
_REMOTE_LOG_NAME = "keyword_generation.log"

_HELP_SECTION_RE = re.compile(
    r"^@@MUFINDER_HELP_BEGIN (?P<name>[A-Z_\d]+)@@\n"
    r"(?P<doc>.*?)"
    r"\n@@MUFINDER_HELP_END (?P=name) (?P<status>-?\d+)@@$",
    re.DOTALL | re.MULTILINE,
)
_VERSION_SECTION_RE = re.compile(
    r"^@@MUFINDER_VERSION_BEGIN@@\n(?P<text>.*?)\n@@MUFINDER_VERSION_END@@$",
    re.DOTALL | re.MULTILINE,
)
#: Keywords described per SSH round trip in direct mode. Large enough that a
#: catalog takes about a dozen round trips, small enough that progress updates
#: and cancellation stay responsive.
_DIRECT_BATCH_SIZE = 50
#: Generous allowance per ``-help`` call when sizing a batch timeout; a call
#: normally takes a fraction of a second.
_HELP_CALL_TIMEOUT_SECONDS = 15
_SCHEDULER_POLL_INTERVALS = (2.0, 2.0, 4.0, 8.0, 15.0)


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
    return issues


def _emit_progress(
    callback: ProgressCallback | None, message: str, percent: int | None = None
) -> None:
    if callback is not None:
        callback(message, percent)


def _check_cancel(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise ClusterKeywordGenerationCancelled("CASTEP keyword generation cancelled.")


def _run_remote_checked(
    ssh: SSHConnection,
    config: ClusterConfig,
    command: str,
    *,
    timeout: int = 60,
    working_directory: str = "",
) -> str:
    wrapped = build_shell_command(
        command=command,
        working_directory=working_directory,
        shell_init=login_shell_init(config),
    )
    exit_code, stdout, stderr = ssh.exec_command(wrapped, timeout=timeout)
    if exit_code != 0:
        details = stderr.strip() or stdout.strip() or f"Remote command failed: {command}"
        raise ClusterKeywordGenerationError(details)
    return stdout


def _parse_castep_version(text: str) -> str:
    """Return the CASTEP version named in ``--version`` output, or ``""``.

    Kept as text: ASE's generator stores a float, which turns ``24.10`` into
    ``24.1``.
    """

    match = _CASTEP_VERSION_RE.search(text)
    return match.group(1).rstrip(".") if match else ""


def _detect_remote_castep_version(ssh: SSHConnection, config: ClusterConfig) -> str:
    wrapped = build_shell_command(
        command=build_castep_command_with_launcher(config, "--version"),
        shell_init=login_shell_init(config),
    )
    try:
        _exit_code, stdout, stderr = ssh.exec_command(wrapped, timeout=30)
    except Exception:  # noqa: BLE001 - the version is informational only.
        return ""
    return _parse_castep_version(stdout) or _parse_castep_version(stderr)


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


@dataclass(slots=True)
class _HelpDump:
    """Keyword descriptions recovered from remote ``-help`` output."""

    cell: dict[str, dict[str, str]]
    param: dict[str, dict[str, str]]
    requested: int = 0
    #: Output of the first keyword whose description could not be used, kept to
    #: explain a failed generation (an MPI launcher error, for example).
    first_failure: str = ""

    @property
    def described(self) -> int:
        return len(self.cell) + len(self.param)

    def add(self, text: str) -> None:
        """Fold the marked-up help sections in ``text`` into this dump."""

        for section in _HELP_SECTION_RE.finditer(text):
            self.requested += 1
            name = section.group("name")
            doc = section.group("doc")
            parsed = _parse_keyword_doc(name, doc) if section.group("status") == "0" else None
            if parsed is None:
                if not self.first_failure:
                    self.first_failure = f"castep -help {name}:\n{doc.strip()[-1500:]}"
                continue
            suffix, entry = parsed
            target = self.cell if suffix == "cell" else self.param
            target[entry["keyword"]] = entry

    def payload(self, castep_version: str) -> dict[str, object]:
        entries = [*self.cell.values(), *self.param.values()]
        return {
            "cell": self.cell,
            "param": self.param,
            "types": sorted({entry["option_type"] for entry in entries}),
            "levels": sorted({entry["level"] for entry in entries}),
            "castep_version": castep_version,
        }


def _checked_payload(
    dump: _HelpDump, castep_version: str, config: ClusterConfig
) -> dict[str, object]:
    """Return the catalog payload, refusing one that is visibly incomplete.

    Individual ``-help`` calls that fail are tolerated, but a run that lost most
    of them must not be written out as a catalog.
    """

    payload = dump.payload(castep_version)
    try:
        validate_keyword_catalog_payload(payload, require_complete=True)
    except KeywordCatalogValidationError as exc:
        if dump.requested == 0:
            message = (
                f"CASTEP on {config.name} did not list any keywords, so no keyword catalog was written. "
                "`castep -help all` produced no keyword names, which usually means CASTEP could not start."
            )
        else:
            message = (
                f"CASTEP on {config.name} described only {dump.described} of {dump.requested} keywords, "
                f"so no keyword catalog was written. {exc}"
            )
        if dump.first_failure:
            message = f"{message}\n\nFirst failing query:\n{dump.first_failure}"
        raise ClusterKeywordGenerationError(message) from exc
    return payload


def _remote_log_excerpt(ssh: SSHConnection, *remote_paths: str) -> str:
    """Return the tail of the first non-empty remote file among ``remote_paths``."""

    for remote_path in remote_paths:
        if not ssh.exists(remote_path):
            continue
        text = ssh.read_text(remote_path)
        lines = [line.rstrip() for line in text.replace("\r", "\n").splitlines() if line.strip()]
        lines = [line for line in lines if not line.startswith("@@MUFINDER_")]
        excerpt = "\n".join(lines[-80:]).strip()
        if excerpt:
            return excerpt[-4000:]
    return ""


def _remove_remote_directory(ssh: SSHConnection, remote_directory: str) -> None:
    try:
        ssh.rmtree(remote_directory)
    except Exception:  # noqa: BLE001 - clean-up must not mask the outcome.
        pass


def _failure_report(
    message: str,
    *,
    ssh: SSHConnection,
    config: ClusterConfig,
    remote_directory: str,
    remote_job_id: str,
) -> str:
    """Return ``message`` with everything needed to troubleshoot a failed job.

    Adds the scheduler's own output files (where a job that never reached the
    keyword script explains itself), says where the job's files were left on
    the cluster, and names the job so it can be looked up in the queue history.
    """

    parts = [message.strip() or f"Cluster keyword generation failed on {config.name}."]
    try:
        names = sorted(ssh.listdir(remote_directory))
    except Exception:  # noqa: BLE001 - diagnostics are best effort.
        names = []
    generated = {_REMOTE_RUNNER_NAME, JOB_HELP_LOOP_NAME}
    for name in names:
        if name in generated:
            continue
        try:
            excerpt = _remote_log_excerpt(ssh, posixpath.join(remote_directory, name))
        except Exception:  # noqa: BLE001
            excerpt = ""
        if excerpt and excerpt not in message:
            parts.append(f"--- {name} ---\n{excerpt}")

    where = f"The job's files were left in {remote_directory} on {config.name} for inspection"
    if remote_job_id:
        where += f" (scheduler job {remote_job_id})"
    parts.append(f"{where}. They can be deleted once the problem is understood.")
    return "\n\n".join(parts)


def generate_cluster_castep_keywords_json(
    *,
    config: ClusterConfig,
    output_path: str | Path,
    password: str | None = None,
    credentials: SSHCredentials | None = None,
    settings=None,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> Path:
    """Generate CASTEP keyword JSON using a remote cluster.

    The keyword job runs on the cluster's default queue, with that queue's
    overrides (executable, template, walltime, memory) applied.
    """

    config = config.for_queue(config.default_queue_name)
    issues = validate_cluster_keyword_config(config)
    if issues:
        raise ClusterKeywordGenerationError("\n".join(f"- {issue}" for issue in issues))

    output = Path(output_path).expanduser()
    if output.suffix.lower() != ".json":
        raise ClusterKeywordGenerationError("CASTEP keywords output path must end with .json.")

    ssh = SSHConnection(config, settings=settings)
    _emit_progress(progress, f"Connecting to {config.name}...", 0)
    resolved_credentials = credentials or SSHCredentials.from_legacy_password(password)
    if config.jump_host_enabled:
        ssh.connect(credentials=resolved_credentials)
    else:
        ssh.connect(password=resolved_credentials.target_secret)
    try:
        _check_cancel(cancel_check)
        mode = _effective_keyword_mode(config)
        if mode == "scheduler":
            if config.queue_system.strip().lower() == "addqueue":
                _emit_progress(
                    progress,
                    f"Using {config.queue_system.upper()} job submission for keyword generation.",
                    5,
                )
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
    _emit_progress(progress, f"Querying CASTEP help on {config.name}...", 5)
    help_all = _run_remote_checked(
        ssh,
        config,
        build_castep_command_with_launcher(config, "-help", "all"),
        timeout=120,
    )
    keywords = list(dict.fromkeys(_HELP_OPTION_RE.findall(help_all)))
    if not keywords:
        raise ClusterKeywordGenerationError("CASTEP help output did not contain any keywords.")
    castep_version = _detect_remote_castep_version(ssh, config)

    castep_command = build_castep_command_with_launcher(config)
    dump = _HelpDump(cell={}, param={})
    for start in range(0, len(keywords), _DIRECT_BATCH_SIZE):
        _check_cancel(cancel_check)
        batch = keywords[start : start + _DIRECT_BATCH_SIZE]
        dump.add(
            _run_remote_checked(
                ssh,
                config,
                help_loop_command(castep_command, batch),
                timeout=max(120, len(batch) * _HELP_CALL_TIMEOUT_SECONDS),
            )
        )
        done = min(start + len(batch), len(keywords))
        percent = max(5, min(99, int((done / len(keywords)) * 100)))
        _emit_progress(
            progress, f"Generating CASTEP keywords... {done} of {len(keywords)}", percent
        )

    payload = _checked_payload(dump, castep_version, config)
    _check_cancel(cancel_check)
    _emit_progress(progress, f"Writing CASTEP keywords to {output_path.name}...", 100)
    return write_keyword_catalog_file(payload, output_path)


def _remote_described_count(ssh: SSHConnection, remote_dump_path: str) -> int:
    """Return how many keywords the running job has described, or 0 if unknown.

    The job writes its dump as it goes, so counting the end markers shows that
    a long-running job is making progress rather than stuck.
    """

    command = f"grep -c {shlex.quote(HELP_END_MARKER)} {shlex.quote(remote_dump_path)}"
    try:
        _exit_code, stdout, _stderr = ssh.exec_command(command, timeout=30)
        return int(stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001 - progress detail is best effort.
        return 0


def _wait_for_scheduler_job(
    *,
    ssh: SSHConnection,
    scheduler,
    config: ClusterConfig,
    remote_job_id: str,
    remote_log_path: str,
    remote_dump_path: str,
    progress: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> None:
    """Block until the keyword job leaves the queue, backing off between polls."""

    poll = 0
    while True:
        _check_cancel(cancel_check)
        statuses = scheduler.status(ssh, [remote_job_id])
        info = statuses.get(remote_job_id)
        mapped = JobStatus.COMPLETED if info is None else scheduler.map_to_job_status(info.state)
        if mapped == JobStatus.QUEUED:
            _emit_progress(progress, f"Keyword job {remote_job_id} is queued...", 35)
        elif mapped == JobStatus.RUNNING:
            described = _remote_described_count(ssh, remote_dump_path)
            detail = f" ({described} keywords described so far)" if described else ""
            _emit_progress(progress, f"Keyword job {remote_job_id} is running...{detail}", 60)
        elif mapped == JobStatus.COMPLETED:
            return
        elif mapped == JobStatus.CANCELLED:
            raise ClusterKeywordGenerationCancelled("Cluster keyword generation job was cancelled.")
        elif mapped in {JobStatus.FAILED, JobStatus.STOPPED}:
            message = f"Cluster keyword generation job failed on {config.name}."
            details = _remote_log_excerpt(ssh, remote_log_path, remote_dump_path)
            if details:
                message = f"{message}\n\n{details}"
            raise ClusterKeywordGenerationError(message)

        # Poll gently: a queued job can wait a long time, and schedulers dislike
        # being asked every couple of seconds. Sleep in short slices so that a
        # cancel request is still honoured promptly.
        interval = _SCHEDULER_POLL_INTERVALS[min(poll, len(_SCHEDULER_POLL_INTERVALS) - 1)]
        poll += 1
        waited = 0.0
        while waited < interval:
            _check_cancel(cancel_check)
            time.sleep(min(1.0, interval - waited))
            waited += 1.0


def _generate_scheduler_keywords_json(
    *,
    ssh: SSHConnection,
    config: ClusterConfig,
    output_path: Path,
    progress: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> Path:
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
    remote_runner_path = posixpath.join(remote_directory, _REMOTE_RUNNER_NAME)
    remote_dump_path = posixpath.join(remote_directory, _REMOTE_DUMP_FILENAME)
    remote_log_path = posixpath.join(remote_directory, _REMOTE_LOG_NAME)
    ssh.mkdir(remote_directory, exist_ok=True)

    remote_job_id = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".sh",
            delete=False,
        ) as handle:
            handle.write(
                scheduler_runner_script(
                    config=config,
                    remote_directory=remote_directory,
                    remote_dump_path=remote_dump_path,
                    remote_log_path=remote_log_path,
                )
            )
            runner_path = Path(handle.name)
        try:
            _emit_progress(progress, f"Uploading keyword generation script to {config.name}...", 10)
            ssh.put(runner_path, remote_runner_path)
            ssh.chmod(remote_runner_path, 0o755)
        finally:
            runner_path.unlink(missing_ok=True)

        request = SchedulerSubmitRequest(
            working_dir=remote_directory,
            seed_name=remote_runner_path
            if config.queue_system.strip().lower() == "addqueue"
            else "keywords",
            num_processes=1,
            cores_per_node=1,
            walltime=scheduler_walltime(config),
            castep_executable="/bin/bash"
            if config.queue_system.strip().lower() == "addqueue"
            else "",
            script_path=remote_runner_path
            if config.queue_system.strip().lower() == "slurm"
            else "",
            memory_gb=config.default_memory_gb,
            queue_name=config.default_queue_name,
            comment="MuFinder CASTEP keywords",
        )

        _check_cancel(cancel_check)
        _emit_progress(progress, f"Submitting keyword generation job on {config.name}...", 20)
        remote_job_id = scheduler.submit(ssh, request)
        _emit_progress(progress, f"Waiting for cluster job {remote_job_id}...", 30)
        _wait_for_scheduler_job(
            ssh=ssh,
            scheduler=scheduler,
            config=config,
            remote_job_id=remote_job_id,
            remote_log_path=remote_log_path,
            remote_dump_path=remote_dump_path,
            progress=progress,
            cancel_check=cancel_check,
        )

        _emit_progress(
            progress, f"Downloading CASTEP keyword descriptions from {config.name}...", 90
        )
        dump_text = ssh.read_text(remote_dump_path) if ssh.exists(remote_dump_path) else ""
        if DUMP_COMPLETE_MARKER not in dump_text:
            message = (
                f"The keyword generation job on {config.name} ended before it had described every keyword "
                "(it may have hit its walltime or failed to start CASTEP)."
            )
            details = _remote_log_excerpt(ssh, remote_log_path, remote_dump_path)
            if details:
                message = f"{message}\n\n{details}"
            raise ClusterKeywordGenerationError(message)

        dump = _HelpDump(cell={}, param={})
        dump.add(dump_text)
        version_section = _VERSION_SECTION_RE.search(dump_text)
        castep_version = (
            _parse_castep_version(version_section.group("text")) if version_section else ""
        )
        payload = _checked_payload(dump, castep_version, config)
        written = write_keyword_catalog_file(payload, output_path)
        _emit_progress(progress, "CASTEP keyword generation complete.", 100)
    except ClusterKeywordGenerationCancelled:
        if remote_job_id:
            try:
                scheduler.cancel(ssh, remote_job_id)
            except Exception:
                pass
        _remove_remote_directory(ssh, remote_directory)
        raise
    except Exception as exc:  # noqa: BLE001 - every failure gets the same diagnostics.
        # Leave the job's files on the cluster: they are the only record of
        # what went wrong, and the message alone is often not enough.
        raise ClusterKeywordGenerationError(
            _failure_report(
                str(exc),
                ssh=ssh,
                config=config,
                remote_directory=remote_directory,
                remote_job_id=remote_job_id,
            )
        ) from exc
    _remove_remote_directory(ssh, remote_directory)
    return written


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
        credentials: SSHCredentials | None = None,
        settings=None,
    ) -> None:
        super().__init__()
        self._config = config
        self._output_path = Path(output_path)
        self._credentials = credentials or SSHCredentials.from_legacy_password(password)
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
                credentials=self._credentials,
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
