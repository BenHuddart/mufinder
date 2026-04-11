"""Rendering boundary from resolved JobSpec to CASTEP text files."""

from __future__ import annotations

from mufinder.castep_io import write_cell, write_param

from .types import JobSpec


def render_job(job_spec: JobSpec) -> tuple[str, str]:
    """Render one resolved JobSpec into CASTEP .cell and .param text."""

    return (write_cell(job_spec.cell_file), write_param(job_spec.param_file))
