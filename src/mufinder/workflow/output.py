"""Local filesystem output for resolved CASTEP jobs."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Sequence

from mufinder.persistence.codec import to_jsonable

from .renderer import render_job
from .types import JobSpec


def write_job_directory(job_spec: JobSpec, base_path: Path) -> Path:
    """Create one directory for a resolved job and write CASTEP inputs."""

    job_dir = base_path / job_spec.name
    job_dir.mkdir(parents=True, exist_ok=True)

    cell_text, param_text = render_job(job_spec)
    (job_dir / f"{job_spec.name}.cell").write_text(cell_text, encoding="utf-8")
    (job_dir / f"{job_spec.name}.param").write_text(param_text, encoding="utf-8")

    return job_dir


def write_manifest(jobs: Sequence[JobSpec], base_path: Path, *, file_name: str = "manifest.json") -> Path:
    """Write a reproducibility manifest for a set of resolved jobs."""

    base_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "jobs": [
            {
                "name": job.name,
                "directory": job.name,
                "cell_file": f"{job.name}/{job.name}.cell",
                "param_file": f"{job.name}/{job.name}.param",
                "workflow_stage": job.workflow_stage.value,
                "tags": list(job.tags),
                "source_structure_id": str(job.structure.source_structure_id)
                if job.structure.source_structure_id is not None
                else None,
                "provenance": to_jsonable(job.provenance),
            }
            for job in jobs
        ],
    }

    manifest_path = base_path / file_name
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def write_job_set(jobs: Sequence[JobSpec], base_path: Path) -> Path:
    """Write all job directories and a manifest to the local filesystem."""

    base_path.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        write_job_directory(job, base_path)
    return write_manifest(jobs, base_path)
