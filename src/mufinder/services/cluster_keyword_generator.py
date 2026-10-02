"""Cluster-backed CASTEP keyword generation helpers.

CASTEP documents its keywords one at a time: ``castep -help all`` lists the
names and ``castep -help <NAME>`` describes one. A full catalog therefore needs
several hundred ``-help`` calls. They are all issued from a small shell loop on
the cluster, which prints every description between marker lines; MuFinder
parses that dump locally. The cluster needs nothing but ``bash`` and CASTEP, and
the number of SSH round trips is a handful instead of one per keyword.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import posixpath
import re
import shlex
import tempfile
import time
from string import Template
from typing import Callable
from uuid import uuid4

from PySide6.QtCore import QObject, Signal, Slot

from mufinder.castep_io.keyword_catalog_file import (
    KeywordCatalogValidationError,
    validate_keyword_catalog_payload,
    write_keyword_catalog_file,
)
from mufinder.cluster_setup import ClusterConfig
from mufinder.domain.enums import JobStatus
from mufinder.services.scheduler import SchedulerSubmitRequest, create_scheduler
from mufinder.services.script_template import load_template_for_cluster, populate_template
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
#: Walltime requested for the keyword job, independent of the cluster's default
#: job walltime. A job that can run the bare binary finishes in a few minutes,
#: but one that must start a job step per query is far slower: on SCARF it
#: managed roughly 20 queries a minute, which puts ~570 queries at about half an
#: hour. An hour leaves room for that without asking for a production-length slot.
_KEYWORD_JOB_WALLTIME = "01:00:00"
_KEYWORD_JOB_NAME = "mufinder-keywords"
_TEMPLATE_RUN_LINE_RE = re.compile(r"\$\{?(castep_command|seed_name)\}?")
_TEMPLATE_CASTEP_COMMAND_RE = re.compile(r"\$\{?castep_command\}?")
_REMOTE_DUMP_FILENAME = "keyword_help.txt"
_REMOTE_RUNNER_NAME = "run_keywords.sh"
_REMOTE_LOG_NAME = "keyword_generation.log"
_JOB_HELP_LOOP_NAME = "mufinder_help_loop.sh"

# Marker lines the remote shell loop prints around each piece of CASTEP output.
_HELP_BEGIN_MARKER = "@@MUFINDER_HELP_BEGIN"
_HELP_END_MARKER = "@@MUFINDER_HELP_END"
_VERSION_BEGIN_MARKER = "@@MUFINDER_VERSION_BEGIN@@"
_VERSION_END_MARKER = "@@MUFINDER_VERSION_END@@"
_DUMP_COMPLETE_MARKER = "@@MUFINDER_DUMP_COMPLETE@@"
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
# Awk equivalent of ``_HELP_OPTION_RE`` for the scheduler job, which has to pick
# the keyword names out of ``castep -help all`` without any help from MuFinder.
_AWK_KEYWORD_NAMES = (
    "{ if (match($0, /^ ?[A-Z_0-9][A-Z_0-9]+/)) "
    '{ name = substr($0, RSTART, RLENGTH); sub(/^ /, "", name); print name } }'
)
#: Keywords described per SSH round trip in direct mode. Large enough that a
#: catalog takes about a dozen round trips, small enough that progress updates
#: and cancellation stay responsive.
_DIRECT_BATCH_SIZE = 50
#: Generous allowance per ``-help`` call when sizing a batch timeout; a call
#: normally takes a fraction of a second.
_HELP_CALL_TIMEOUT_SECONDS = 15
_SCHEDULER_POLL_INTERVALS = (2.0, 2.0, 4.0, 8.0, 15.0)
#: ``-help`` queries run at once. They mostly wait rather than compute (0.8 s of
#: CPU in 2.8 s of wall time on SCARF), so a few at a time pays off even on the
#: single core a keyword job asks for, without loading a login node.
_DEFAULT_PARALLEL_QUERIES = 4
#: Environment variables that tie a process to the MPI job (launcher step) it
#: was started in: the PMIx/PMI rendezvous of ``srun`` and ``mpirun``, and the
#: Slurm variables that mark a process as a task of a job step. Without them an
#: MPI program starts as a standalone single process.
_MPI_JOB_ENVIRONMENT_RE = (
    "^(PMIX_|PMI_|PRTE_|PRRTE_|ORTE_|HYDRA_|OMPI_COMM_WORLD_|OMPI_UNIVERSE_SIZE|OMPI_APP_CTX_|"
    "OMPI_NUM_APP_CTX|OMPI_FIRST_RANKS|OMPI_COMMAND|OMPI_ARGV|OMPI_FILE_LOCATION|"
    "OMPI_MCA_(ess|orte|prte|pmix|plm|ras|rmaps|schizo)_|"
    "SLURM_STEP|SLURM_PMIX|SLURM_PMI|SLURM_TASK|SLURM_PROCID|SLURM_LOCALID|SLURM_NODEID|SLURM_GTIDS|"
    "SLURM_SRUN_COMM|SLURM_LAUNCH_NODE_IPADDR|SLURM_DISTRIBUTION|SLURM_MPI_TYPE)"
)
#: Seconds allowed for the standalone attempt on an ``addqueue`` cluster before
#: it is abandoned for the launcher-per-query fallback.
_STANDALONE_ATTEMPT_SECONDS = 900
#: Seconds one ``-help`` query may take before it is cut off, how many cut-off
#: queries are tolerated before a run gives up, and the exit status it gives up
#: with. A query normally takes well under three seconds.
_QUERY_TIMEOUT_SECONDS = 120
_QUERY_TIMEOUTS_TOLERATED = 3
_STOPPED_RESPONDING_EXIT_STATUS = 75
#: Open MPI settings that keep a single process off the network fabric, and
#: make it give up the CPU while it waits rather than spin, so that queries
#: sharing a core do not starve each other.
_OPENMPI_LOCAL_SETTINGS = (
    ("OMPI_MCA_pml", "ob1"),
    ("OMPI_MCA_btl", "self"),
    ("OMPI_MCA_mpi_yield_when_idle", "1"),
)


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
        shell_init=_login_shell_init(config),
    )
    exit_code, stdout, stderr = ssh.exec_command(wrapped, timeout=timeout)
    if exit_code != 0:
        details = stderr.strip() or stdout.strip() or f"Remote command failed: {command}"
        raise ClusterKeywordGenerationError(details)
    return stdout


@dataclass(frozen=True, slots=True)
class _JobTemplatePlan:
    """What the cluster's own job script template says about running CASTEP.

    MuFinder already knows how to run CASTEP on a cluster: the template used for
    every other job carries the scheduler directives (partition, account), the
    environment setup (``module load``), and the launcher. Keyword generation
    reuses all of it instead of asking the user to describe the cluster twice.
    """

    #: Template lines before the CASTEP run line, with variables filled in.
    preamble: tuple[str, ...]
    #: Launcher words preceding the executable on the run line (``srun -n 1``).
    launcher: tuple[str, ...]
    #: The executable the template runs; may differ from the configured one
    #: when the template hard-codes it.
    executable: str

    @property
    def setup_lines(self) -> tuple[str, ...]:
        """Environment setup commands, usable on a login node as well."""

        lines = []
        for line in self.preamble:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "cd" or stripped.startswith("cd "):
                continue
            lines.append(stripped)
        return tuple(lines)


def _job_template_plan(config: ClusterConfig, *, working_directory: str = "") -> _JobTemplatePlan | None:
    """Read the cluster's job template, or return ``None`` if it has none.

    The run line is the last command that mentions ``$castep_command`` or
    ``$seed_name``. Everything above it is kept; it and everything below are
    dropped, because they act on a calculation that the keyword job does not
    run.
    """

    try:
        template = load_template_for_cluster(
            config.template_script_path,
            cluster_name=config.name,
            queue_system=config.queue_system,
        )
    except (ValueError, OSError):
        return None

    raw_lines = template.template.splitlines()
    run_index = next(
        (
            index
            for index in range(len(raw_lines) - 1, -1, -1)
            if not raw_lines[index].lstrip().startswith("#") and _TEMPLATE_RUN_LINE_RE.search(raw_lines[index])
        ),
        None,
    )
    if run_index is None:
        return None

    variables = {
        "job_name": _KEYWORD_JOB_NAME,
        "num_processes": "1",
        "num_nodes": "1",
        "cores_per_node": "1",
        "walltime": _KEYWORD_JOB_WALLTIME,
        "working_directory": shlex.quote(working_directory) if working_directory else ".",
        "seed_name": "keywords",
    }
    configured_executable = config.castep_executable.strip()
    is_slurm = config.queue_system.strip().lower() == "slurm"
    default_launcher = ("srun", "-n", "1") if is_slurm else ("mpirun", "-np", "1")

    run_line = raw_lines[run_index]
    if _TEMPLATE_CASTEP_COMMAND_RE.search(run_line):
        # MuFinder builds this command itself for real jobs (see
        # ``render_job_script``), so its launcher is known exactly.
        launcher, executable = default_launcher, configured_executable
    else:
        try:
            words = shlex.split(Template(run_line).safe_substitute(variables))
        except ValueError:
            words = []
        index = next(
            (
                position
                for position, word in enumerate(words)
                if posixpath.basename(word).lower().startswith("castep")
                or (configured_executable and word == configured_executable)
            ),
            None,
        )
        if index is None:
            launcher, executable = default_launcher, configured_executable
        else:
            launcher, executable = tuple(words[:index]), words[index]

    preamble = populate_template(Template("\n".join(raw_lines[:run_index])), variables).splitlines()
    return _JobTemplatePlan(preamble=tuple(preamble), launcher=launcher, executable=executable)


def _login_shell_init(config: ClusterConfig) -> str:
    """Return the environment setup to run before CASTEP on the login node.

    An explicit ``keyword_shell_init`` wins. Otherwise the setup commands of the
    cluster's job template are reused, so a template that says
    ``module load CASTEP`` does not have to be repeated in cluster setup.
    """

    explicit = config.keyword_shell_init.strip()
    if explicit:
        return config.keyword_shell_init
    plan = _job_template_plan(config)
    return "\n".join(plan.setup_lines) if plan is not None else ""


def _login_castep_executable(config: ClusterConfig) -> str:
    plan = _job_template_plan(config)
    if plan is not None and plan.executable:
        return plan.executable
    return config.castep_executable.strip()


def _keyword_castep_launcher(config: ClusterConfig) -> str:
    if config.queue_system.strip().lower() == "addqueue":
        return "mpirun -np 1"
    return ""


def _build_castep_command_with_launcher(config: ClusterConfig, *args: str) -> str:
    launcher = _keyword_castep_launcher(config)
    launcher_parts = shlex.split(launcher) if launcher else []
    executable = _login_castep_executable(config)
    parts = [*launcher_parts, executable, *args]
    return " ".join(shlex.quote(part) for part in parts)


def _parse_castep_version(text: str) -> str:
    """Return the CASTEP version named in ``--version`` output, or ``""``.

    Kept as text: ASE's generator stores a float, which turns ``24.10`` into
    ``24.1``.
    """

    match = _CASTEP_VERSION_RE.search(text)
    return match.group(1).rstrip(".") if match else ""


def _detect_remote_castep_version(ssh: SSHConnection, config: ClusterConfig) -> str:
    wrapped = _build_shell_command(
        command=_build_castep_command_with_launcher(config, "--version"),
        shell_init=_login_shell_init(config),
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


def _fast_start_lines(castep_command: str) -> list[str]:
    """Return shell that trims MPI start-up for the ``-help`` queries.

    A ``-help`` query needs no network, but an MPI-built CASTEP still brings up
    its fabric first. On SCARF that cost about two of the 2.8 seconds each query
    took: Open MPI tried its OFI transport, failed, and fell back. Restricting
    Open MPI to its in-process transports skips that. The settings are applied
    only if CASTEP still answers with them, and other MPI libraries ignore them.
    """

    settings = " ".join(f"{name}={value}" for name, value in _OPENMPI_LOCAL_SETTINGS)
    return [
        'if [ "${MUFINDER_FAST_START:-1}" = 1 ] \\',
        f"    && {settings} {castep_command} --version 2>/dev/null | grep -q 'CASTEP version'; then",
        f"  export {settings}",
        "fi",
    ]


def _timing_lines(castep_command: str) -> list[str]:
    """Return shell that records, on stderr, what one query costs on this node.

    Generation time is start-up time multiplied by several hundred, and whether
    that start-up is spent computing or waiting decides what can speed it up.
    One timed query in the job log answers that without anyone having to
    reproduce the job's environment by hand.
    """

    return [
        "{",
        '  echo "mufinder: $(nproc 2>/dev/null || echo \'?\') cpu(s) available,'
        " Open MPI fast start $([ -n \"$OMPI_MCA_btl\" ] && echo applied || echo 'not applied')\"",
        "  TIMEFORMAT='mufinder: one query took %R s (%U s user, %S s system)'",
        f"  time {castep_command} -help TASK >/dev/null 2>&1 || true",
        "} >&2",
    ]


def _query_guard_lines() -> list[str]:
    """Return shell that sets ``$mufinder_guard``, the time limit on one query.

    A binary can answer ``--version`` and then hang in ``MPI_Init`` on a later
    start. Without a limit that would hold a strategy, and so the job, until
    the walltime. Every query is run under ``timeout`` where it exists, and the
    loop gives up once several have been cut off.
    """

    return [
        'mufinder_guard=""',
        "if command -v timeout >/dev/null 2>&1; then",
        f'  mufinder_guard="timeout ${{MUFINDER_QUERY_TIMEOUT:-{_QUERY_TIMEOUT_SECONDS}}}"',
        "fi",
        "mufinder_timeouts=0",
    ]


def _help_loop_command(
    castep_command: str,
    keywords: list[str] | None = None,
    *,
    record_timing: bool = False,
    command_applies_guard: bool = False,
) -> str:
    """Return shell that prints every keyword description between marker lines.

    Queries are started a few at a time where that pays, and their output is
    collected in files so that it is still printed whole and in order. Each
    group is printed as soon as it finishes, which keeps progress visible.

    Whether it pays depends on the node, so the loop measures it: it times one
    query alone, then starts a full group, and drops to one at a time unless the
    group finishes in clearly less time than its queries take in sequence. A
    group that overruns is abandoned at that point and its queries are run
    again in turn, so the trial costs a second or two at most. On SCARF a
    query mostly waited on MPI start-up, and four at once were much faster. On
    Redwood a lone query took 0.5 s but four sharing the job's single CPU took
    several seconds between them, slower than running them in turn.
    ``MUFINDER_PARALLEL`` sets the group size (1 disables grouping).

    Args:
        castep_command: The already-quoted CASTEP invocation, launcher included.
        keywords: The keyword names to describe. ``None`` makes the loop ask
            CASTEP for the list itself, for a scheduler job that runs unattended.
        record_timing: Write the cost of one query to stderr first (the job log,
            for a scheduler job).
        command_applies_guard: ``castep_command`` is a shell function that puts
            ``$mufinder_guard`` in front of CASTEP itself. Otherwise the guard
            is placed in front of ``castep_command`` here.
    """

    guard = "" if command_applies_guard else "$mufinder_guard "

    if keywords is None:
        source = f"$({castep_command} -help all 2>&1 | awk {shlex.quote(_AWK_KEYWORD_NAMES)})"
    else:
        source = " ".join(shlex.quote(name) for name in keywords)
    return "\n".join(
        [
            *_query_guard_lines(),
            *_fast_start_lines(castep_command),
            *(_timing_lines(castep_command) if record_timing else ()),
            "mufinder_scratch=$(mktemp -d)",
            f'mufinder_parallel="${{MUFINDER_PARALLEL:-{_DEFAULT_PARALLEL_QUERIES}}}"',
            # Nanosecond clock for the calibration below; BSD date prints a
            # literal N, in which case calibration is skipped.
            "mufinder_now() { date +%s%N 2>/dev/null; }",
            'case "$(mufinder_now)" in "" | *[!0-9]*) mufinder_calibrate=0 ;; *) mufinder_calibrate=1 ;; esac',
            '[ "$mufinder_parallel" -gt 1 ] || mufinder_calibrate=0',
            "mufinder_round=0",
            f"set -- {source}",
            'if [ "$#" -eq 0 ]; then echo "mufinder: CASTEP listed no keywords" >&2; fi',
            'while [ "$#" -gt 0 ]; do',
            # While calibrating, the first round is a single query and the
            # second a full group, so the two can be compared.
            '  size="$mufinder_parallel"',
            '  if [ "$mufinder_calibrate" = 1 ] && [ "$mufinder_round" = 0 ]; then size=1; fi',
            '  chunk=""',
            "  count=0",
            '  while [ "$#" -gt 0 ] && [ "$count" -lt "$size" ]; do',
            '    chunk="$chunk $1"',
            "    shift",
            "    count=$((count + 1))",
            "  done",
            '  if [ "$mufinder_calibrate" = 1 ]; then mufinder_started=$(mufinder_now); fi',
            '  pids=""',
            "  for keyword in $chunk; do",
            "    (",
            "      status=0",
            f'      {guard}{castep_command} -help "$keyword" > "$mufinder_scratch/$keyword.out" 2>&1 || status=$?',
            '      printf \'%s\' "$status" > "$mufinder_scratch/$keyword.status"',
            "    ) &",
            '    pids="$pids $!"',
            "  done",
            '  if [ "$mufinder_calibrate" = 1 ] && [ "$mufinder_round" = 1 ] && [ "$count" = "$mufinder_parallel" ]; then',
            # The trial group. It is worth keeping only if it finishes in
            # clearly less time than its queries take in turn; the moment it
            # has used that much time it is abandoned rather than waited for
            # (on Redwood such a group took 38 s against 0.5 s for one query).
            "    mufinder_limit_ns=$((mufinder_single * count * 3 / 4))",
            "    while :; do",
            "      pending=0",
            "      for keyword in $chunk; do",
            '        [ -f "$mufinder_scratch/$keyword.status" ] || pending=1',
            "      done",
            '      [ "$pending" = 1 ] || break',
            "      mufinder_elapsed=$(( $(mufinder_now) - mufinder_started ))",
            '      if [ "$mufinder_elapsed" -gt "$mufinder_limit_ns" ]; then',
            '        echo "mufinder: $count queries at once had not finished after $((mufinder_elapsed / 1000000)) ms'
            ' against $((mufinder_single / 1000000)) ms for one; running one at a time" >&2',
            "        for pid in $pids; do",
            # Either may find nothing left to kill, or be missing; neither
            # is an error, and the caller may be running under `set -e`.
            '          pkill -TERM -P "$pid" 2>/dev/null || true',
            '          kill "$pid" 2>/dev/null || true',
            "        done",
            # Anything that outlives the kill keeps writing into the old
            # scratch directory, away from the queries that are run again.
            '        mufinder_abandoned="$mufinder_scratch"',
            "        mufinder_scratch=$(mktemp -d)",
            "        mufinder_parallel=1",
            "        mufinder_calibrate=0",
            '        set -- $chunk "$@"',
            "        continue 2",
            "      fi",
            "      sleep 0.05",
            "    done",
            "  fi",
            # Only this round's queries: an abandoned one that survived being
            # killed must not hold the loop up.
            "  wait $pids || true",
            '  if [ "$mufinder_calibrate" = 1 ]; then',
            "    mufinder_elapsed=$(( $(mufinder_now) - mufinder_started ))",
            '    if [ "$mufinder_round" = 0 ]; then',
            "      mufinder_single=$mufinder_elapsed",
            "    else",
            '      if [ "$count" = "$mufinder_parallel" ]; then',
            '        echo "mufinder: $count queries at once took $((mufinder_elapsed / 1000000)) ms against'
            ' $((mufinder_single / 1000000)) ms for one; keeping $count at a time" >&2',
            "      fi",
            "      mufinder_calibrate=0",
            "    fi",
            "    mufinder_round=$((mufinder_round + 1))",
            "  fi",
            "  for keyword in $chunk; do",
            f"    printf '{_HELP_BEGIN_MARKER} %s@@\\n' \"$keyword\"",
            '    cat "$mufinder_scratch/$keyword.out"',
            '    query_status=$(cat "$mufinder_scratch/$keyword.status")',
            f"    printf '\\n{_HELP_END_MARKER} %s %s@@\\n' \"$keyword\" \"$query_status\"",
            # 124 is how `timeout` reports that it cut the query off.
            '    if [ -n "$mufinder_guard" ] && [ "$query_status" = 124 ]; then',
            "      mufinder_timeouts=$((mufinder_timeouts + 1))",
            "    fi",
            "  done",
            f'  if [ "$mufinder_timeouts" -ge {_QUERY_TIMEOUTS_TOLERATED} ]; then',
            '    echo "mufinder: CASTEP stopped responding: $mufinder_timeouts queries were cut off after'
            f' ${{MUFINDER_QUERY_TIMEOUT:-{_QUERY_TIMEOUT_SECONDS}}} s each" >&2',
            '    rm -rf "$mufinder_scratch" ${mufinder_abandoned:+"$mufinder_abandoned"}',
            f"    exit {_STOPPED_RESPONDING_EXIT_STATUS}",
            "  fi",
            "done",
            'rm -rf "$mufinder_scratch" ${mufinder_abandoned:+"$mufinder_abandoned"}',
        ]
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


def _checked_payload(dump: _HelpDump, castep_version: str, config: ClusterConfig) -> dict[str, object]:
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


def _scheduler_walltime(config: ClusterConfig) -> str:
    if config.queue_system.strip().lower() == "addqueue":
        return config.default_walltime.strip()
    return _KEYWORD_JOB_WALLTIME


def _job_help_loop_script() -> str:
    """Return the script that writes one complete dump to standard output.

    It runs CASTEP as whatever command line it is given, so the same script
    serves every strategy in :func:`_job_strategy_lines`.
    """

    return "\n".join(
        [
            'if [ "$1" = "--standalone" ]; then',
            "  shift",
            # Forget the MPI job this shell belongs to, so that each CASTEP
            # starts as an independent single process, exactly as it does in a
            # plain batch script.
            f"  for name in $(compgen -e | grep -E {shlex.quote(_MPI_JOB_ENVIRONMENT_RE)}); do",
            '    unset "$name"',
            "  done",
            "fi",
            # $mufinder_guard is empty until the help loop sets the time limit.
            'mufinder_castep() { $mufinder_guard "${MUFINDER_CASTEP[@]}" "$@"; }',
            'MUFINDER_CASTEP=("$@")',
            f"printf '{_VERSION_BEGIN_MARKER}\\n'",
            "mufinder_castep --version 2>&1 || true",
            f"printf '\\n{_VERSION_END_MARKER}\\n'",
            _help_loop_command("mufinder_castep", record_timing=True, command_applies_guard=True),
            f"printf '{_DUMP_COMPLETE_MARKER}\\n'",
        ]
    )


def _job_strategy_lines(
    *,
    launcher: tuple[str, ...],
    executable: str,
    remote_dump_path: str,
    remote_log_path: str,
    inside_launcher_step: bool = False,
) -> list[str]:
    """Return shell that produces the dump by the fastest strategy that works.

    Starting CASTEP costs a fraction of a second, but starting a *job step* for
    it costs seconds: on SCARF, launching each of ~570 queries through ``srun``
    ran at about 20 queries a minute. So the launcher is used as little as the
    cluster allows, trying in order:

    1. the bare executable, when it starts inside the allocation;
    2. one job step that runs the whole loop, with the bare executable inside
       it, when two consecutive starts inside a single step both work;
    3. the launcher for every query, which mirrors how real jobs run and is the
       fallback that always applies.

    The first two are attempted only when ``timeout`` exists: it bounds the
    probe, and inside the loop it bounds every query, so a binary that hangs
    instead of failing costs a strategy a few minutes rather than the job its
    walltime (see :func:`_query_guard_lines`). Each is kept only if its dump is
    complete and at least nine queries in ten succeeded. Otherwise the next
    strategy overwrites the dump.

    ``inside_launcher_step`` rules all of that out. ``addqueue`` does not run
    the job script as a batch script: it starts it with ``srun --mpi=pmix``, so
    the script is already a task of an MPI job step. There a bare CASTEP is not
    an independent process but that step's rank 0; only its first start works,
    and when a later one fails in ``MPI_Init`` the scheduler cancels the whole
    step, script included (seen on Redwood: "STEP ... CANCELLED DUE TO TASK
    FAILURE", four seconds into the job). Starting CASTEP bare in the script is
    therefore never safe there. Launching every query through ``mpirun`` is
    safe and proven, but slow (Redwood: 2.4 s a query, over twenty minutes).

    So one launcher invocation is used as a *container*: ``mpirun -np 1`` runs
    the whole loop, and inside it the loop drops the variables that tie a
    process to an MPI job, so each CASTEP starts as a standalone single process
    as it would in a plain batch script, several at a time. If that goes wrong
    the damage is confined to that one ``mpirun``, the dump is rejected, and
    the proven launcher-per-query path (one at a time, no Open MPI overrides)
    takes over.
    """

    dump = shlex.quote(remote_dump_path)
    log = shlex.quote(remote_log_path)
    quoted_executable = shlex.quote(executable)
    quoted_launcher = " ".join(shlex.quote(part) for part in launcher)
    script_name = _JOB_HELP_LOOP_NAME
    probe = f"{quoted_executable} --version"
    lines = [
        f"cat > {script_name} <<'MUFINDER_HELP_LOOP'",
        _job_help_loop_script(),
        "MUFINDER_HELP_LOOP",
        f": > {log}",
        "mufinder_dump_ok() {",
        f"  grep -q '^{_DUMP_COMPLETE_MARKER}$' {dump} || return 1",
        f"  total=$(grep -c '^{_HELP_BEGIN_MARKER} ' {dump})",
        f"  good=$(grep -c '^{_HELP_END_MARKER} [A-Z_0-9]* 0@@$' {dump})",
        '  [ "$total" -gt 0 ] && [ $((good * 10)) -ge $((total * 9)) ]',
        "}",
        "mufinder_try() {",
        '  echo "mufinder: trying $1" >> ' + log,
        "  shift",
        f'  "$@" > {dump} 2>> {log} && mufinder_dump_ok',
        "}",
    ]
    if not launcher:
        lines.append(f"mufinder_try 'the CASTEP executable' bash {script_name} {quoted_executable} || true")
        return lines
    if inside_launcher_step:
        lines.extend(
            [
                "mufinder_limit() {",
                '  if command -v timeout >/dev/null 2>&1; then timeout "$@"; else shift; "$@"; fi',
                "}",
                "mufinder_try 'standalone CASTEP processes inside one launcher' "
                f"mufinder_limit {_STANDALONE_ATTEMPT_SECONDS} {quoted_launcher} "
                f"bash {script_name} --standalone {quoted_executable} \\",
                "  || mufinder_try 'the launcher for every query' env MUFINDER_PARALLEL=1 MUFINDER_FAST_START=0 "
                f"bash {script_name} {quoted_launcher} {quoted_executable} || true",
            ]
        )
        return lines

    # One at a time: a one-task allocation cannot run job steps side by side.
    per_query = (
        "mufinder_try 'the launcher for every query' "
        f"env MUFINDER_PARALLEL=1 bash {script_name} {quoted_launcher} {quoted_executable} || true"
    )
    step_probe = shlex.quote(f"{probe} >/dev/null 2>&1 && {probe}")
    lines.extend(
        [
            "if ! command -v timeout >/dev/null 2>&1; then",
            f"  {per_query}",
            f"elif timeout 30 {probe} 2>/dev/null | grep -q 'CASTEP version' \\",
            f"    && mufinder_try 'the bare CASTEP executable' bash {script_name} {quoted_executable}; then",
            "  :",
            f"elif timeout 120 {quoted_launcher} bash -c {step_probe} 2>/dev/null | grep -q 'CASTEP version' \\",
            f"    && mufinder_try 'a single job step' {quoted_launcher} bash {script_name} {quoted_executable}; then",
            "  :",
            "else",
            f"  {per_query}",
            "fi",
        ]
    )
    return lines


def _scheduler_runner_script(
    *,
    config: ClusterConfig,
    remote_directory: str,
    remote_dump_path: str,
    remote_log_path: str,
) -> str:
    """Return the keyword job script, modelled on the cluster's own job script."""

    is_slurm = config.queue_system.strip().lower() == "slurm"
    plan = _job_template_plan(config, working_directory=remote_directory) if is_slurm else None
    if plan is not None:
        # Scheduler directives, environment setup and launcher all come from
        # the template the user's real jobs are submitted with.
        lines = list(plan.preamble)
        if not lines or not lines[0].startswith("#!"):
            lines.insert(0, "#!/bin/bash")
        launcher, executable = plan.launcher, plan.executable
    else:
        lines = ["#!/bin/bash"]
        if is_slurm:
            # Slurm stops reading #SBATCH directives at the first command, so
            # they must come before anything else.
            lines.extend(
                [
                    f"#SBATCH -J {_KEYWORD_JOB_NAME}",
                    "#SBATCH -n 1",
                    "#SBATCH -N 1",
                    f"#SBATCH -t {_scheduler_walltime(config)}",
                    "",
                ]
            )
        launcher_text = _keyword_castep_launcher(config)
        if is_slurm:
            # With no template to read a launcher from, offer Slurm's own as
            # the fallback for an MPI build that will not start bare.
            launcher = ("srun", "-n", "1")
        else:
            launcher = tuple(shlex.split(launcher_text)) if launcher_text else ()
        executable = config.castep_executable.strip()

    lines.append(f"cd {shlex.quote(remote_directory)}")
    if config.keyword_shell_init.strip():
        lines.append(config.keyword_shell_init.rstrip())
    lines.extend(
        _job_strategy_lines(
            launcher=launcher,
            executable=executable,
            remote_dump_path=remote_dump_path,
            remote_log_path=remote_log_path,
            inside_launcher_step=config.queue_system.strip().lower() == "addqueue",
        )
    )
    lines.append("")
    return "\n".join(lines)


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
    generated = {_REMOTE_RUNNER_NAME, _JOB_HELP_LOOP_NAME}
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
    """Generate CASTEP keyword JSON using a remote cluster."""

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
    _emit_progress(progress, f"Querying CASTEP help on {config.name}...", 5)
    help_all = _run_remote_checked(
        ssh,
        config,
        _build_castep_command_with_launcher(config, "-help", "all"),
        timeout=120,
    )
    keywords = list(dict.fromkeys(_HELP_OPTION_RE.findall(help_all)))
    if not keywords:
        raise ClusterKeywordGenerationError("CASTEP help output did not contain any keywords.")
    castep_version = _detect_remote_castep_version(ssh, config)

    castep_command = _build_castep_command_with_launcher(config)
    dump = _HelpDump(cell={}, param={})
    for start in range(0, len(keywords), _DIRECT_BATCH_SIZE):
        _check_cancel(cancel_check)
        batch = keywords[start : start + _DIRECT_BATCH_SIZE]
        dump.add(
            _run_remote_checked(
                ssh,
                config,
                _help_loop_command(castep_command, batch),
                timeout=max(120, len(batch) * _HELP_CALL_TIMEOUT_SECONDS),
            )
        )
        done = min(start + len(batch), len(keywords))
        percent = max(5, min(99, int((done / len(keywords)) * 100)))
        _emit_progress(progress, f"Generating CASTEP keywords... {done} of {len(keywords)}", percent)

    payload = _checked_payload(dump, castep_version, config)
    _check_cancel(cancel_check)
    _emit_progress(progress, f"Writing CASTEP keywords to {output_path.name}...", 100)
    return write_keyword_catalog_file(payload, output_path)


def _remote_described_count(ssh: SSHConnection, remote_dump_path: str) -> int:
    """Return how many keywords the running job has described, or 0 if unknown.

    The job writes its dump as it goes, so counting the end markers shows that
    a long-running job is making progress rather than stuck.
    """

    command = f"grep -c {shlex.quote(_HELP_END_MARKER)} {shlex.quote(remote_dump_path)}"
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
                _scheduler_runner_script(
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

        _emit_progress(progress, f"Downloading CASTEP keyword descriptions from {config.name}...", 90)
        dump_text = ssh.read_text(remote_dump_path) if ssh.exists(remote_dump_path) else ""
        if _DUMP_COMPLETE_MARKER not in dump_text:
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
        castep_version = _parse_castep_version(version_section.group("text")) if version_section else ""
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
