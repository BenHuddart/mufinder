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
    (job_dir / f"{job_spec.name}.cell").write_text(cell_text, encoding="utf-8", newline="\n")
    (job_dir / f"{job_spec.name}.param").write_text(param_text, encoding="utf-8", newline="\n")

    return job_dir


def write_job_directories(
    items: Sequence[tuple[JobSpec, Path]],
    *,
    previous_inputs: dict[Path, bytes | None] | None = None,
    made_folders: list[Path] | None = None,
) -> list[Path]:
    """Write the inputs of every ``(spec, base_path)`` in ``items``, all or none.

    Every job is rendered before anything is written, so a job that cannot be
    rendered writes nothing. When a write then fails (a full disk, a folder that
    cannot be made), every file this call wrote is put back as it was, every
    folder it made is removed, and the error is raised again. Returns the job
    directories in the order given.

    ``previous_inputs``, when given, receives each written file's bytes from
    before the write (``None`` for a file that was not there), and
    ``made_folders`` the folders the call made, so one job's inputs can be put
    back later (:func:`restore_job_inputs`), or all of them
    (:func:`undo_job_writes`).
    """

    rendered = [(spec, base_path / spec.name, render_job(spec)) for spec, base_path in items]
    made_dirs: list[Path] = []
    previous: list[tuple[Path, bytes | None]] = []
    try:
        for spec, job_dir, (cell_text, param_text) in rendered:
            missing = []
            parent = job_dir
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            # Listed before they are made: a mkdir that fails part-way leaves no parent.
            made_dirs.extend(reversed(missing))
            job_dir.mkdir(parents=True, exist_ok=True)
            for suffix, text in ((".cell", cell_text), (".param", param_text)):
                path = job_dir / f"{spec.name}{suffix}"
                previous.append((path, path.read_bytes() if path.is_file() else None))
                path.write_text(text, encoding="utf-8", newline="\n")
    except BaseException:
        for path, data in reversed(previous):
            try:
                if data is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(data)
            except OSError:
                pass
        for directory in reversed(made_dirs):
            try:
                directory.rmdir()
            except OSError:
                pass
        raise
    if previous_inputs is not None:
        for path, data in previous:
            previous_inputs.setdefault(path, data)
    if made_folders is not None:
        made_folders.extend(made_dirs)
    return [job_dir for _spec, job_dir, _texts in rendered]


def restore_job_inputs(previous_inputs: dict[Path, bytes | None], job_dir: Path) -> list[str]:
    """Put back the input files in ``job_dir`` that :func:`write_job_directories` wrote.

    Each goes back to the bytes ``previous_inputs`` recorded, and one that was
    not there before is removed. Returns one line for each file that could not
    be put back; nothing is raised.
    """

    errors: list[str] = []
    for path, data in previous_inputs.items():
        if path.parent != job_dir:
            continue
        try:
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(data)
        except OSError as exc:
            errors.append(f"{path.name} could not be put back ({exc})")
    return errors


def undo_job_writes(
    previous_inputs: dict[Path, bytes | None], made_folders: Sequence[Path]
) -> None:
    """Put back every file :func:`write_job_directories` wrote and remove the folders it made.

    For a later step of the same change that failed. Nothing is raised.
    """

    for path, data in previous_inputs.items():
        try:
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(data)
        except OSError:
            pass
    for folder in reversed(list(made_folders)):
        try:
            folder.rmdir()
        except OSError:
            pass


def write_manifest(
    jobs: Sequence[JobSpec], base_path: Path, *, file_name: str = "manifest.json"
) -> Path:
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
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def write_job_set(jobs: Sequence[JobSpec], base_path: Path) -> Path:
    """Write all job directories and a manifest to the local filesystem."""

    base_path.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        write_job_directory(job, base_path)
    return write_manifest(jobs, base_path)
