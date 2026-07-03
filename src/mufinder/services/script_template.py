"""Cluster script template helpers."""

from __future__ import annotations

from math import ceil
from pathlib import Path
import re
import shlex
from string import Template


_ALLOWED_TEMPLATE_VARIABLES = {
    "job_name",
    "num_processes",
    "num_nodes",
    "cores_per_node",
    "walltime",
    "working_directory",
    "seed_name",
    "castep_command",
}


_TEMPLATE_VAR_RE = re.compile(r"\$\{?([a-zA-Z_][a-zA-Z0-9_]*)\}?")

_BUILTIN_TEMPLATE_TOKEN_SLURM = "builtin:slurm-basic"
_BUILTIN_TEMPLATE_TOKEN_SCARF = "builtin:scarf-slurm"

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


def load_template(path: Path | str) -> Template:
    """Load a script template from disk."""

    token = str(path).strip()
    if token in _BUILTIN_TEMPLATES:
        return Template(_BUILTIN_TEMPLATES[token])
    content = Path(path).read_text(encoding="utf-8")
    return Template(content)


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
) -> str:
    """Render a cluster scheduler submission script from its runtime inputs.

    This is the pure rendering step that ``ClusterJobRunner.submit`` used to
    inline. It loads the configured (or built-in) template, derives the node
    count from ``num_processes`` / ``cores_per_node``, builds the launcher
    command (``srun`` for slurm, ``mpirun`` otherwise), and substitutes the
    eight template variables. It returns only the rendered script text; the
    caller remains responsible for writing, uploading, and chmod-ing it.

    Design note — why this lives here and not in
    ``JobExecutionController.prepare_cluster_submission``: rendering consumes
    values that are only resolved at submit time. ``remote_dir`` is computed
    inside ``submit`` (from the working directory and seed name), and the
    CASTEP executable is resolved there against the runner's cluster config
    fallback. Relocating rendering to the controller's GUI-thread prepare step
    would force that same resolution to be duplicated there. So the extraction
    keeps this a pure function called from ``submit`` at exactly the same
    point, with the already-resolved ``remote_dir`` and ``castep_executable``
    passed in.

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
    """

    template = load_template_for_cluster(
        template_script_path,
        cluster_name=cluster_name,
        queue_system=queue_system,
    )
    nodes = max(1, int(ceil(max(1, int(num_processes)) / max(1, int(cores_per_node)))))
    if queue_system.strip().lower() == "slurm":
        command = (
            f"srun -n {max(1, int(num_processes))} "
            f"{shlex.quote(castep_executable)} {seed_name}"
        )
    else:
        command = (
            f"mpirun -np {max(1, int(num_processes))} "
            f"{shlex.quote(castep_executable)} {seed_name}"
        )
    return populate_template(
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


def validate_template(template: Template) -> list[str]:
    """Return warnings for unknown placeholders used in a template."""

    warnings: list[str] = []
    seen: set[str] = set()
    for match in _TEMPLATE_VAR_RE.finditer(template.template):
        variable = match.group(1)
        if variable in _ALLOWED_TEMPLATE_VARIABLES or variable in seen:
            continue
        seen.add(variable)
        warnings.append(f"Unknown template variable: {variable}")
    return warnings
