"""Tests for script template helpers."""

from __future__ import annotations

from pathlib import Path
from string import Template

import pytest

from mufinder.services.script_template import (
    load_template,
    load_template_for_cluster,
    populate_template,
    render_job_script,
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


def _render_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "template_script_path": "",
        "cluster_name": "alpha",
        "queue_system": "slurm",
        "seed_name": "job1",
        "num_processes": 8,
        "cores_per_node": 4,
        "walltime": "01:00:00",
        "remote_dir": "/remote/job1",
        "castep_executable": "castep.mpi",
    }
    kwargs.update(overrides)
    return kwargs


def test_render_job_script_slurm_uses_srun_command() -> None:
    script = render_job_script(**_render_kwargs(queue_system="slurm", num_processes=8))

    assert "srun -n 8 castep.mpi job1" in script
    assert "mpirun" not in script


def test_render_job_script_non_slurm_uses_mpirun_command(
    tmp_path: Path,
) -> None:
    # The generic slurm built-in only fires for slurm; give a custom template so
    # the non-slurm branch has something to render into.
    template_path = tmp_path / "custom.sh"
    template_path.write_text("cmd: $castep_command\n", encoding="utf-8")

    script = render_job_script(
        **_render_kwargs(
            template_script_path=str(template_path),
            queue_system="pbs",
            num_processes=6,
        )
    )

    assert "mpirun -np 6 castep.mpi job1" in script
    assert "srun" not in script


def test_render_job_script_computes_node_ceiling() -> None:
    def _render(procs: int, cores: int) -> str:
        return render_job_script(
            **_render_kwargs(
                template_script_path="",
                queue_system="slurm",
                num_processes=procs,
                cores_per_node=cores,
            )
        )

    # 10 processes / 4 per node -> ceil(2.5) == 3 nodes.
    assert "#SBATCH -N 3" in _render(10, 4)
    # Exact division stays exact.
    assert "#SBATCH -N 2" in _render(8, 4)
    # Fewer processes than a node still yields at least one node.
    assert "#SBATCH -N 1" in _render(1, 4)


def test_render_job_script_substitutes_all_variables(tmp_path: Path) -> None:
    template_path = tmp_path / "full.sh"
    template_path.write_text(
        "job=$job_name procs=$num_processes nodes=$num_nodes "
        "cores=$cores_per_node wall=$walltime dir=$working_directory "
        "seed=$seed_name cmd=$castep_command\n",
        encoding="utf-8",
    )

    script = render_job_script(
        **_render_kwargs(
            template_script_path=str(template_path),
            queue_system="slurm",
            seed_name="myseed",
            num_processes=10,
            cores_per_node=4,
            walltime="12:00:00",
            remote_dir="/scratch/myseed",
            castep_executable="castep.mpi",
        )
    )

    assert "job=myseed" in script
    assert "procs=10" in script
    assert "nodes=3" in script
    assert "cores=4" in script
    assert "wall=12:00:00" in script
    assert "dir=/scratch/myseed" in script
    assert "seed=myseed" in script
    assert "cmd=srun -n 10 castep.mpi myseed" in script


def test_render_job_script_scarf_builtin_fallback() -> None:
    script = render_job_script(
        **_render_kwargs(
            template_script_path="",
            cluster_name="SCARF",
            queue_system="slurm",
        )
    )

    assert "module load CASTEP" in script
    assert "#SBATCH -p scarf" in script


def test_render_job_script_generic_slurm_builtin_fallback() -> None:
    script = render_job_script(
        **_render_kwargs(
            template_script_path="",
            cluster_name="alpha",
            queue_system="SLURM",
        )
    )

    assert "#SBATCH --ntasks-per-node=4" in script
    assert "srun -n 8 castep.mpi job1" in script


def test_render_job_script_shell_quotes_executable_with_spaces(tmp_path: Path) -> None:
    template_path = tmp_path / "quote.sh"
    template_path.write_text("$castep_command\n", encoding="utf-8")

    script = render_job_script(
        **_render_kwargs(
            template_script_path=str(template_path),
            queue_system="slurm",
            num_processes=4,
            castep_executable="/opt/my castep/castep.mpi",
        )
    )

    assert "srun -n 4 '/opt/my castep/castep.mpi' job1" in script
