"""The shell scripts that dump CASTEP's keyword descriptions on a cluster.

The scripts themselves are templates in ``mufinder/assets/shell/``, rendered by
:func:`~mufinder.services.shell_templates.render_shell_template`. This module
works out what goes into them: the cluster's environment setup, launcher and
executable (read from its job script template where it has one), quoted for the
shell, and the markers and limits that :mod:`cluster_keyword_generator` relies
on when it reads the dump back.
"""

from __future__ import annotations

from dataclasses import dataclass
import posixpath
import re
import shlex
from string import Template

from mufinder.domain.cluster import ClusterConfig
from mufinder.services.script_template import (
    format_mem_per_cpu_directive,
    insert_partition_directive,
    insert_sbatch_directive,
    load_template_for_cluster,
    populate_template,
)
from mufinder.services.shell_templates import render_shell_template


#: Walltime requested for the keyword job, independent of the cluster's default
#: job walltime. A job that can run the bare binary finishes in a few minutes,
#: but one that must start a job step per query is far slower: on SCARF it
#: managed roughly 20 queries a minute, which puts ~570 queries at about half an
#: hour. An hour leaves room for that without asking for a production-length slot.
_KEYWORD_JOB_WALLTIME = "01:00:00"
_KEYWORD_JOB_NAME = "mufinder-keywords"
_TEMPLATE_RUN_LINE_RE = re.compile(r"\$\{?(castep_command|seed_name)\}?")
_TEMPLATE_CASTEP_COMMAND_RE = re.compile(r"\$\{?castep_command\}?")
JOB_HELP_LOOP_NAME = "mufinder_help_loop.sh"

#: Marker lines the remote shell loop prints around each piece of CASTEP output,
#: which ``cluster_keyword_generator`` looks for when it reads the dump back.
HELP_BEGIN_MARKER = "@@MUFINDER_HELP_BEGIN"
HELP_END_MARKER = "@@MUFINDER_HELP_END"
VERSION_BEGIN_MARKER = "@@MUFINDER_VERSION_BEGIN@@"
VERSION_END_MARKER = "@@MUFINDER_VERSION_END@@"
DUMP_COMPLETE_MARKER = "@@MUFINDER_DUMP_COMPLETE@@"
#: Awk equivalent of ``_HELP_OPTION_RE`` in ``cluster_keyword_generator``, for the
#: scheduler job, which has to pick the keyword names out of ``castep -help all``
#: without any help from MuFinder.
_AWK_KEYWORD_NAMES = (
    "{ if (match($0, /^ ?[A-Z_0-9][A-Z_0-9]+/)) "
    '{ name = substr($0, RSTART, RLENGTH); sub(/^ /, "", name); print name } }'
)
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


def build_shell_command(
    *,
    command: str,
    working_directory: str = "",
    shell_init: str = "",
) -> str:
    """Return ``command`` wrapped to run in a login shell on the cluster."""

    script = render_shell_template(
        "login_command.sh",
        cd=f"cd {shlex.quote(working_directory.strip())}" if working_directory.strip() else "",
        shell_init=shell_init.rstrip() if shell_init.strip() else "",
        command=command,
    )
    return f"/bin/bash -lc {shlex.quote(script)}"


@dataclass(frozen=True, slots=True)
class JobTemplatePlan:
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


def job_template_plan(config: ClusterConfig, *, working_directory: str = "") -> JobTemplatePlan | None:
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
    return JobTemplatePlan(preamble=tuple(preamble), launcher=launcher, executable=executable)


def login_shell_init(config: ClusterConfig) -> str:
    """Return the environment setup to run before CASTEP on the login node.

    An explicit ``keyword_shell_init`` wins. Otherwise the setup commands of the
    cluster's job template are reused, so a template that says
    ``module load CASTEP`` does not have to be repeated in cluster setup.
    """

    explicit = config.keyword_shell_init.strip()
    if explicit:
        return config.keyword_shell_init
    plan = job_template_plan(config)
    return "\n".join(plan.setup_lines) if plan is not None else ""


def _login_castep_executable(config: ClusterConfig) -> str:
    plan = job_template_plan(config)
    if plan is not None and plan.executable:
        return plan.executable
    return config.castep_executable.strip()


def _keyword_castep_launcher(config: ClusterConfig) -> str:
    if config.queue_system.strip().lower() == "addqueue":
        return "mpirun -np 1"
    return ""


def build_castep_command_with_launcher(config: ClusterConfig, *args: str) -> str:
    """Return the quoted CASTEP command line for the login node, with ``args``."""

    launcher = _keyword_castep_launcher(config)
    launcher_parts = shlex.split(launcher) if launcher else []
    executable = _login_castep_executable(config)
    parts = [*launcher_parts, executable, *args]
    return " ".join(shlex.quote(part) for part in parts)


def help_loop_command(
    castep_command: str,
    keywords: list[str] | None = None,
    *,
    record_timing: bool = False,
    command_applies_guard: bool = False,
) -> str:
    """Return shell that prints every keyword description between marker lines.

    The loop is ``assets/shell/help_loop.sh``, which explains how it groups
    queries, limits each one and gives up on a CASTEP that stops responding.

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

    if keywords is None:
        source = f"$({castep_command} -help all 2>&1 | awk {shlex.quote(_AWK_KEYWORD_NAMES)})"
    else:
        source = " ".join(shlex.quote(name) for name in keywords)
    return render_shell_template(
        "help_loop.sh",
        query_timeout=_QUERY_TIMEOUT_SECONDS,
        openmpi_settings=" ".join(f"{name}={value}" for name, value in _OPENMPI_LOCAL_SETTINGS),
        castep=castep_command,
        timing=render_shell_template("help_loop_timing.sh", castep=castep_command) if record_timing else "",
        parallel_default=_DEFAULT_PARALLEL_QUERIES,
        keywords=source,
        guard="" if command_applies_guard else "$mufinder_guard ",
        help_begin=HELP_BEGIN_MARKER,
        help_end=HELP_END_MARKER,
        timeouts_tolerated=_QUERY_TIMEOUTS_TOLERATED,
        stopped_status=_STOPPED_RESPONDING_EXIT_STATUS,
    )


def scheduler_walltime(config: ClusterConfig) -> str:
    """Return the walltime the keyword job asks for."""

    if config.queue_system.strip().lower() == "addqueue":
        return config.default_walltime.strip()
    return _KEYWORD_JOB_WALLTIME


def _job_help_loop_script() -> str:
    """Return the script that writes one complete dump to standard output."""

    return render_shell_template(
        "job_help_loop.sh",
        mpi_job_environment_re=shlex.quote(_MPI_JOB_ENVIRONMENT_RE),
        version_begin=VERSION_BEGIN_MARKER,
        version_end=VERSION_END_MARKER,
        help_loop=help_loop_command("mufinder_castep", record_timing=True, command_applies_guard=True),
        dump_complete=DUMP_COMPLETE_MARKER,
    )


def _job_strategy_script(
    *,
    launcher: tuple[str, ...],
    executable: str,
    inside_launcher_step: bool = False,
) -> str:
    """Return shell that produces the dump by the fastest strategy that works.

    The strategies, and why each cluster gets the one it does, are described in
    ``assets/shell/keyword_job_strategy_*.sh``.
    """

    quoted_executable = shlex.quote(executable)
    if not launcher:
        return render_shell_template(
            "keyword_job_strategy_bare.sh",
            help_loop_name=JOB_HELP_LOOP_NAME,
            executable=quoted_executable,
        )
    quoted_launcher = " ".join(shlex.quote(part) for part in launcher)
    if inside_launcher_step:
        return render_shell_template(
            "keyword_job_strategy_launcher_step.sh",
            standalone_seconds=_STANDALONE_ATTEMPT_SECONDS,
            launcher=quoted_launcher,
            help_loop_name=JOB_HELP_LOOP_NAME,
            executable=quoted_executable,
        )
    probe = f"{quoted_executable} --version"
    return render_shell_template(
        "keyword_job_strategy_job_steps.sh",
        help_loop_name=JOB_HELP_LOOP_NAME,
        launcher=quoted_launcher,
        executable=quoted_executable,
        step_probe=shlex.quote(f"{probe} >/dev/null 2>&1 && {probe}"),
    )


def scheduler_runner_script(
    *,
    config: ClusterConfig,
    remote_directory: str,
    remote_dump_path: str,
    remote_log_path: str,
) -> str:
    """Return the keyword job script, modelled on the cluster's own job script."""

    is_slurm = config.queue_system.strip().lower() == "slurm"
    plan = job_template_plan(config, working_directory=remote_directory) if is_slurm else None
    if plan is not None:
        # Scheduler directives, environment setup and launcher all come from
        # the template the user's real jobs are submitted with.
        preamble = list(plan.preamble)
        if not preamble or not preamble[0].startswith("#!"):
            preamble.insert(0, "#!/bin/bash")
        launcher, executable = plan.launcher, plan.executable
    else:
        preamble = ["#!/bin/bash"]
        if is_slurm:
            # Slurm stops reading #SBATCH directives at the first command, so
            # they must come before anything else.
            preamble.extend(
                [
                    f"#SBATCH -J {_KEYWORD_JOB_NAME}",
                    "#SBATCH -n 1",
                    "#SBATCH -N 1",
                    f"#SBATCH -t {scheduler_walltime(config)}",
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

    preamble_text = "\n".join(preamble)
    if is_slurm:
        # The keyword job asks for the default queue's partition and memory,
        # as a real job on that queue would; a template line wins for either.
        preamble_text = insert_partition_directive(preamble_text, config.default_queue_name)
        preamble_text = insert_sbatch_directive(
            preamble_text, format_mem_per_cpu_directive(config.default_mem_per_cpu_gb)
        )
    script = render_shell_template(
        "keyword_job.sh",
        preamble=preamble_text,
        directory=shlex.quote(remote_directory),
        shell_init=config.keyword_shell_init.rstrip() if config.keyword_shell_init.strip() else "",
        help_loop_name=JOB_HELP_LOOP_NAME,
        help_loop_script=_job_help_loop_script(),
        log=shlex.quote(remote_log_path),
        dump=shlex.quote(remote_dump_path),
        dump_complete=DUMP_COMPLETE_MARKER,
        help_begin=HELP_BEGIN_MARKER,
        help_end=HELP_END_MARKER,
        strategy=_job_strategy_script(
            launcher=launcher,
            executable=executable,
            inside_launcher_step=config.queue_system.strip().lower() == "addqueue",
        ),
    )
    return f"{script}\n"
