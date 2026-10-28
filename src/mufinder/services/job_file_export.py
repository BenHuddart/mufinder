"""Collect a job's CASTEP input and output files and copy them to another folder.

:func:`job_files_export` lists the standard CASTEP files a job left in its
directory, or says why there are none; :func:`plan_job_file_copy` decides where
each file goes in a chosen folder (one sub-folder per job, named after its
seed); :func:`copy_job_files` copies them. Checkpoints (``.check``,
``.castep_bin``) are left out: they are large binaries that only CASTEP reads.
Asking for the folder and confirming replacements is the caller's.

Layering: this lives in ``mufinder.services`` and imports ``domain`` only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import shutil

from mufinder.domain.models import Job

NO_FOLDER_REASON = "This job has no local folder yet."
NO_FILES_REASON = "This job's folder holds no CASTEP input or output files."

#: Suffixes after the seed (or after ``<seed>_farmNNN`` for task farming) that are copied.
_EXPORT_SUFFIXES = (
    ".cell",
    ".param",
    "-in.cell",
    ".castep",
    "-out.cell",
    "-out.cif",
    ".geom",
    ".md",
    ".ts",
    ".bands",
    ".phonon",
    ".den_fmt",
)
#: Error files: the run's ``<seed>.err`` and CASTEP's per-process ``<seed>.0001.err``.
_ERROR_FILE_SUFFIX = re.compile(r"(\.\d+)?\.err")
_FARM_PREFIX = re.compile(r"_farm\d+")


@dataclass(frozen=True, slots=True)
class JobFilesExport:
    """One job's files ready to copy, or the reason there are none.

    ``files`` is empty exactly when ``unavailable_reason`` is set. ``directory``
    is the job's folder when it exists, even when it holds nothing to copy.
    """

    job: Job
    seed: str
    directory: Path | None
    files: tuple[Path, ...]
    unavailable_reason: str | None


def job_files_export(job: Job, directory: Path | None, seed: str) -> JobFilesExport:
    """Return the standard CASTEP files for ``seed`` in ``directory``, sorted by name."""

    if directory is None or not directory.is_dir():
        return JobFilesExport(job, seed, None, (), NO_FOLDER_REASON)
    try:
        files = tuple(
            sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file() and _is_export_file(path, seed)
                ),
                key=lambda path: path.name,
            )
        )
    except OSError as exc:
        return JobFilesExport(
            job, seed, directory, (), f"Could not read this job's folder: {exc.strerror or exc}"
        )
    return JobFilesExport(job, seed, directory, files, None if files else NO_FILES_REASON)


def _is_export_file(path: Path, seed: str) -> bool:
    name = path.name
    if not seed or not name.startswith(seed):
        return False
    rest = name[len(seed) :]
    farm = _FARM_PREFIX.match(rest)
    if farm is not None:
        rest = rest[farm.end() :]
    return rest in _EXPORT_SUFFIXES or _ERROR_FILE_SUFFIX.fullmatch(rest) is not None


def plan_job_file_copy(
    exports: Sequence[JobFilesExport], destination: Path
) -> list[tuple[Path, Path]]:
    """Pair every file in ``exports`` with its target under ``destination``.

    Each job gets its own sub-folder named after its seed (the job label when
    there is no seed), with ``_2``, ``_3``… added where two jobs share a name.
    Jobs with nothing to copy are skipped.
    """

    used: set[str] = set()
    pairs: list[tuple[Path, Path]] = []
    for export in exports:
        if not export.files:
            continue
        stem = export.seed.strip() or _folder_name(export.job.label) or "job"
        name, counter = stem, 1
        while name.lower() in used:
            counter += 1
            name = f"{stem}_{counter}"
        used.add(name.lower())
        pairs.extend((source, destination / name / source.name) for source in export.files)
    return pairs


def _folder_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]+", "_", text.strip()).strip("_.")


def copies_onto_source(pairs: Sequence[tuple[Path, Path]]) -> bool:
    """Return whether any target in ``pairs`` is its own source (the folder holds the job folders)."""

    return any(source.resolve() == target.resolve() for source, target in pairs)


def copy_job_files(pairs: Sequence[tuple[Path, Path]]) -> tuple[list[Path], list[str]]:
    """Copy each ``(source, target)`` pair; return the written targets and the failures.

    A failure reads ``"<job folder>/<file>: <reason>"``, and does not stop the
    remaining copies.
    """

    copied: list[Path] = []
    failures: list[str] = []
    for source, target in pairs:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError as exc:
            failures.append(f"{target.parent.name}/{target.name}: {exc.strerror or exc}")
        else:
            copied.append(target)
    return copied, failures


__all__ = [
    "NO_FILES_REASON",
    "NO_FOLDER_REASON",
    "JobFilesExport",
    "copies_onto_source",
    "copy_job_files",
    "job_files_export",
    "plan_job_file_copy",
]
