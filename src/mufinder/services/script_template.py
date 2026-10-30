"""Cluster script template helpers."""

from __future__ import annotations

from math import ceil
from pathlib import Path
import re
import shlex
from string import Template


# Any leading-block ``#SBATCH`` line, any ``#SBATCH`` line that already
# carries a memory request (``--mem``, ``--mem-per-cpu``, ``--mem-per-gpu``),
# and any that already names a partition (``-p``, ``--partition``).
_SBATCH_LINE_RE = re.compile(r"^\s*#SBATCH\b")
_SBATCH_MEMORY_LINE_RE = re.compile(r"^\s*#SBATCH\b.*--mem", re.IGNORECASE)
_SBATCH_PARTITION_LINE_RE = re.compile(r"^\s*#SBATCH\b.*\s(?:-p|--partition\b)")

_BUILTIN_TEMPLATE_TOKEN_SLURM = "builtin:slurm-basic"
_BUILTIN_TEMPLATE_TOKEN_SCARF = "builtin:scarf-slurm"
_BUILTIN_TEMPLATE_LABELS = {
    _BUILTIN_TEMPLATE_TOKEN_SLURM: "Built-in SLURM template",
    _BUILTIN_TEMPLATE_TOKEN_SCARF: "Built-in SCARF template",
}

_BUILTIN_TEMPLATES: dict[str, str] = {
    _BUILTIN_TEMPLATE_TOKEN_SLURM: """#!/bin/bash

#SBATCH -J $job_name
#SBATCH -n $num_processes
#SBATCH -N $num_nodes
#SBATCH --ntasks-per-node=$cores_per_node
#SBATCH -t $walltime

cd $working_directory

$castep_command
""",
    _BUILTIN_TEMPLATE_TOKEN_SCARF: """#!/bin/bash

#SBATCH -J $job_name
#SBATCH -n $num_processes
#SBATCH -p scarf
#SBATCH -t $walltime

module load CASTEP
cd $working_directory

srun -n $num_processes castep.mpi $seed_name
""",
}


def format_mem_per_cpu_directive(mem_per_cpu_gb: float | None) -> str:
    """Return the ``#SBATCH`` memory directive for a per-CPU request in GB.

    Returns ``""`` when no positive request is set, so callers can treat "no
    memory requested" as "emit nothing" and leave scheduler defaults in place.

    SLURM's ``--mem-per-cpu`` takes an integer with an optional unit suffix, so
    ``7.5`` GB cannot be written as ``7.5G``. Whole numbers of gigabytes render
    as ``G``; anything else is converted to whole megabytes (``7.5`` -> ``7680M``).

    ``--mem-per-cpu`` is per *allocated CPU*, not per task. MuFinder submits
    pure-MPI CASTEP runs and never sets ``--cpus-per-task``, which SLURM
    defaults to 1, so one CPU is one MPI rank and this value is the per-rank
    memory ceiling enforced by the cgroup.
    """

    if mem_per_cpu_gb is None:
        return ""
    try:
        value = float(mem_per_cpu_gb)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    megabytes = int(round(value * 1024))
    if megabytes <= 0:
        return ""
    if megabytes % 1024 == 0:
        return f"#SBATCH --mem-per-cpu={megabytes // 1024}G"
    return f"#SBATCH --mem-per-cpu={megabytes}M"


def format_partition_directive(queue_name: str) -> str:
    """Return the ``#SBATCH -p`` directive for a queue, or ``""`` for none."""

    name = queue_name.strip()
    return f"#SBATCH -p {name}" if name else ""


def template_sets_partition(
    template_script_path: str, *, cluster_name: str, queue_system: str
) -> bool:
    """Return whether the cluster's job template hard-codes a partition.

    Such a template wins over the queue chosen for a submission, so the dialogs
    use this to say the choice will not reach the script. A template that
    cannot be read sets nothing.
    """

    if queue_system.strip().lower() != "slurm":
        return False
    try:
        template = load_template_for_cluster(
            template_script_path,
            cluster_name=cluster_name,
            queue_system=queue_system,
        )
    except (ValueError, OSError):
        return False
    return any(_SBATCH_PARTITION_LINE_RE.match(line) for line in template.template.splitlines())


def insert_sbatch_directive(
    script: str, directive: str, *, unless: re.Pattern[str] | None = None
) -> str:
    """Insert one ``#SBATCH`` directive into a rendered batch script.

    The directive is placed after the last existing ``#SBATCH`` line so it stays
    inside the leading comment block, where ``sbatch`` stops parsing directives.
    A script with a line matching ``unless`` is left untouched, so a
    hand-written template line always wins over the dialog. ``unless`` defaults
    to any memory request (``--mem``/``--mem-per-cpu``/``--mem-per-gpu``).

    Templates with no ``#SBATCH`` line at all still receive the directive, just
    after the shebang (or at the top when there is none). ``#SBATCH`` is only a
    comment to the shell, so this cannot break a script that the scheduler
    ignores it in.
    """

    if not directive:
        return script

    lines = script.splitlines(keepends=True)
    existing = _SBATCH_MEMORY_LINE_RE if unless is None else unless
    if any(existing.match(line) for line in lines):
        return script

    insert_at = 0
    for index, line in enumerate(lines):
        if _SBATCH_LINE_RE.match(line):
            insert_at = index + 1
    if insert_at == 0 and lines and lines[0].startswith("#!"):
        insert_at = 1

    if insert_at > 0 and not lines[insert_at - 1].endswith("\n"):
        lines[insert_at - 1] = f"{lines[insert_at - 1]}\n"
    lines.insert(insert_at, f"{directive}\n")
    return "".join(lines)


def insert_partition_directive(script: str, queue_name: str) -> str:
    """Add ``#SBATCH -p <queue>`` unless the script already names a partition."""

    return insert_sbatch_directive(
        script,
        format_partition_directive(queue_name),
        unless=_SBATCH_PARTITION_LINE_RE,
    )


def load_template(path: Path | str) -> Template:
    """Load a script template from disk."""

    token = str(path).strip()
    if token in _BUILTIN_TEMPLATES:
        return Template(_BUILTIN_TEMPLATES[token])
    content = Path(path).read_text(encoding="utf-8")
    return Template(content)


def is_builtin_template(template_script_path: str) -> bool:
    """Return whether a configured template names a built-in, not a file."""

    return template_script_path.strip() in _BUILTIN_TEMPLATE_LABELS


def template_display_name(template_script_path: str, *, cluster_name: str) -> str:
    """Name the template a SLURM profile uses, as the user knows it.

    A file is shown by its name. An empty setting names the built-in that
    ``load_template_for_cluster`` falls back to.
    """

    configured = template_script_path.strip()
    if configured in _BUILTIN_TEMPLATE_LABELS:
        return _BUILTIN_TEMPLATE_LABELS[configured]
    if configured:
        return Path(configured).name or configured
    token = (
        _BUILTIN_TEMPLATE_TOKEN_SCARF
        if cluster_name.strip().lower() == "scarf"
        else _BUILTIN_TEMPLATE_TOKEN_SLURM
    )
    return _BUILTIN_TEMPLATE_LABELS[token]


def load_template_for_cluster(
    template_script_path: str,
    *,
    cluster_name: str,
    queue_system: str,
) -> Template:
    """Load either a configured template path or a sensible built-in fallback."""

    configured = template_script_path.strip()
    if configured:
        return load_template(configured)

    if cluster_name.strip().lower() == "scarf":
        return load_template(_BUILTIN_TEMPLATE_TOKEN_SCARF)

    if queue_system.strip().lower() == "slurm":
        return load_template(_BUILTIN_TEMPLATE_TOKEN_SLURM)

    raise ValueError("No template script is configured for this cluster.")


def populate_template(template: Template, variables: dict[str, str]) -> str:
    """Populate a loaded template with runtime variables."""

    stringified = {key: str(value) for key, value in variables.items()}
    return template.safe_substitute(stringified)


def render_job_script(
    *,
    template_script_path: str,
    cluster_name: str,
    queue_system: str,
    seed_name: str,
    num_processes: int,
    cores_per_node: int,
    walltime: str,
    remote_dir: str,
    castep_executable: str,
    mem_per_cpu_gb: float | None = None,
    queue_name: str = "",
) -> str:
    """Render a cluster scheduler submission script from its runtime inputs.

    This is the rendering step that ``ClusterJobRunner.submit`` used to
    inline. It loads the configured (or built-in) template — reading from
    disk when ``template_script_path`` names a file — derives the node count
    from ``num_processes`` / ``cores_per_node``, builds the launcher command
    (``srun`` for slurm, ``mpirun`` otherwise), and substitutes the eight
    template variables. It returns only the rendered script text; the caller
    remains responsible for writing, uploading, and chmod-ing it. Beyond the
    template read it is stateless and touches no runner or SSH machinery.

    Design note — why this lives here and not in
    ``JobExecutionController.prepare_cluster_submission``: rendering consumes
    values that are only resolved at submit time. ``remote_dir`` is computed
    inside ``submit`` (from the working directory and seed name), and the
    CASTEP executable is resolved there against the runner's cluster config
    fallback. Relocating rendering to the controller's GUI-thread prepare step
    would force that same resolution to be duplicated there. So the extraction
    keeps this a self-contained rendering function called from ``submit`` at
    exactly the same point, with the already-resolved ``remote_dir`` and
    ``castep_executable`` passed in.

    :param template_script_path: Configured template path, or "" to fall back
        to a built-in template for the cluster/queue.
    :param cluster_name: Raw cluster name (normalized by the template loader).
    :param queue_system: Raw queue system (normalized here for the launcher
        choice and by the template loader for its fallback selection).
    :param seed_name: CASTEP seed; populates both ``job_name`` and
        ``seed_name`` template variables.
    :param num_processes: Total MPI ranks (clamped to at least 1).
    :param cores_per_node: Cores per node used for the node ceiling math
        (clamped to at least 1).
    :param walltime: Scheduler walltime string.
    :param remote_dir: Resolved remote working directory; populates the
        ``working_directory`` template variable.
    :param castep_executable: Already-resolved CASTEP executable (the submit
        caller applies its config fallback before calling this). Shell-quoted
        into the launcher command.
    :param mem_per_cpu_gb: Optional per-CPU memory request in GB, emitted as an
        ``#SBATCH --mem-per-cpu`` directive on the slurm path only. ``None``
        (the default) emits nothing and leaves the rendered script byte-for-byte
        as it was before this option existed.
    :param queue_name: Optional SLURM partition, emitted as ``#SBATCH -p`` on
        the slurm path only, unless the template already names a partition.
        ``""`` (the default) emits nothing.
    """

    template = load_template_for_cluster(
        template_script_path,
        cluster_name=cluster_name,
        queue_system=queue_system,
    )
    nodes = max(1, int(ceil(max(1, int(num_processes)) / max(1, int(cores_per_node)))))
    is_slurm = queue_system.strip().lower() == "slurm"
    if is_slurm:
        command = (
            f"srun -n {max(1, int(num_processes))} {shlex.quote(castep_executable)} {seed_name}"
        )
    else:
        command = (
            f"mpirun -np {max(1, int(num_processes))} {shlex.quote(castep_executable)} {seed_name}"
        )
    script = populate_template(
        template,
        {
            "job_name": seed_name,
            "num_processes": str(max(1, int(num_processes))),
            "num_nodes": str(nodes),
            "cores_per_node": str(max(1, int(cores_per_node))),
            "walltime": walltime,
            "working_directory": remote_dir,
            "seed_name": seed_name,
            "castep_command": command,
        },
    )
    if not is_slurm:
        return script
    script = insert_partition_directive(script, queue_name)
    return insert_sbatch_directive(script, format_mem_per_cpu_directive(mem_per_cpu_gb))
