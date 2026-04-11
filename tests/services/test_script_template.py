"""Tests for script template helpers."""

from __future__ import annotations

from pathlib import Path
from string import Template

import pytest

from mufinder.services.script_template import (
    load_template,
    load_template_for_cluster,
    populate_template,
    validate_template,
)


def test_load_template_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "run.sh"
    path.write_text("#!/bin/sh\necho $job_name\n", encoding="utf-8")

    template = load_template(path)

    assert isinstance(template, Template)
    assert "$job_name" in template.template


def test_populate_template_substitutes_known_variables() -> None:
    template = Template("run $job_name with $num_processes")

    rendered = populate_template(template, {"job_name": "test", "num_processes": "16"})

    assert rendered == "run test with 16"


def test_populate_template_safe_substitute_keeps_unknowns() -> None:
    template = Template("run $job_name and $unknown")

    rendered = populate_template(template, {"job_name": "test"})

    assert rendered == "run test and $unknown"


def test_validate_template_reports_unknown_placeholders() -> None:
    template = Template("$job_name $not_supported")

    warnings = validate_template(template)

    assert warnings == ["Unknown template variable: not_supported"]


def test_load_template_supports_builtin_tokens() -> None:
    template = load_template("builtin:scarf-slurm")

    assert "#SBATCH -p scarf" in template.template
    assert "castep.mpi $seed_name" in template.template


def test_load_template_for_cluster_prefers_scarf_builtin_when_unset() -> None:
    template = load_template_for_cluster(
        "",
        cluster_name="SCARF",
        queue_system="SLURM",
    )

    assert "module load CASTEP" in template.template


def test_load_template_for_cluster_uses_generic_slurm_when_unset() -> None:
    template = load_template_for_cluster(
        "",
        cluster_name="alpha",
        queue_system="slurm",
    )

    assert "#SBATCH --ntasks-per-node=$cores_per_node" in template.template


def test_load_template_for_cluster_raises_for_unknown_queue() -> None:
    with pytest.raises(ValueError, match="No template script"):
        load_template_for_cluster(
            "",
            cluster_name="alpha",
            queue_system="pbs",
        )
