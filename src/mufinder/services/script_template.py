"""Cluster script template helpers."""

from __future__ import annotations

from pathlib import Path
import re
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
