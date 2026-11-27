"""Tests for collecting a job's CASTEP files and copying them to a chosen folder."""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.services.job_file_export import (
    NO_FILES_REASON,
    NO_FOLDER_REASON,
    JobFilesExport,
    copies_onto_source,
    copy_job_files,
    job_files_export,
    plan_job_file_copy,
)
from tests.support.jobs import make_job


def _touch(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text(name, encoding="utf-8")


def test_job_files_export_lists_the_standard_files_and_skips_checkpoints(tmp_path: Path) -> None:
    job_dir = tmp_path / "files" / "Si"
    _touch(
        job_dir,
        "Si.cell",
        "Si.param",
        "Si.castep",
        "Si-out.cell",
        "Si-in.cell",
        "Si.geom",
        "Si.0001.err",
        "Si.err",
        "Si_farm002.castep",
        "Si.check",
        "Si.castep_bin",
        "Other.castep",
        "notes.txt",
    )
    job = make_job("Si")

    export = job_files_export(job, job_dir, "Si")

    assert export.unavailable_reason is None
    assert export.directory == job_dir
    assert [path.name for path in export.files] == [
        "Si-in.cell",
        "Si-out.cell",
        "Si.0001.err",
        "Si.castep",
        "Si.cell",
        "Si.err",
        "Si.geom",
        "Si.param",
        "Si_farm002.castep",
    ]


def test_job_files_export_explains_a_missing_folder_or_an_empty_one(tmp_path: Path) -> None:
    job = make_job("Si")

    assert job_files_export(job, None, "Si").unavailable_reason == NO_FOLDER_REASON
    assert job_files_export(job, tmp_path / "absent", "Si").unavailable_reason == NO_FOLDER_REASON

    empty = tmp_path / "empty"
    _touch(empty, "Si.check")
    export = job_files_export(job, empty, "Si")
    assert export.directory == empty
    assert export.files == ()
    assert export.unavailable_reason == NO_FILES_REASON


def test_plan_job_file_copy_puts_each_job_in_its_own_seed_folder(tmp_path: Path) -> None:
    first_dir = tmp_path / "a" / "Si"
    second_dir = tmp_path / "b" / "Si"
    _touch(first_dir, "Si.cell", "Si.castep")
    _touch(second_dir, "Si.cell")
    exports = [
        job_files_export(make_job("one"), first_dir, "Si"),
        JobFilesExport(make_job("none"), "Si", None, (), NO_FOLDER_REASON),
        job_files_export(make_job("two"), second_dir, "Si"),
    ]
    destination = tmp_path / "out"

    pairs = plan_job_file_copy(exports, destination)

    assert pairs == [
        (first_dir / "Si.castep", destination / "Si" / "Si.castep"),
        (first_dir / "Si.cell", destination / "Si" / "Si.cell"),
        (second_dir / "Si.cell", destination / "Si_2" / "Si.cell"),
    ]


def test_copy_job_files_copies_and_reports_each_failure(tmp_path: Path) -> None:
    source = tmp_path / "src"
    _touch(source, "Si.cell")
    destination = tmp_path / "out" / "Si"
    pairs = [
        (source / "Si.cell", destination / "Si.cell"),
        (source / "Si.castep", destination / "Si.castep"),
    ]

    copied, failures = copy_job_files(pairs)

    assert copied == [destination / "Si.cell"]
    assert (destination / "Si.cell").read_text(encoding="utf-8") == "Si.cell"
    assert len(failures) == 1
    assert failures[0].startswith("Si/Si.castep: ")


@pytest.mark.parametrize("seed", ["", "  "])
def test_plan_job_file_copy_names_a_folder_without_a_seed_after_the_job(
    tmp_path: Path, seed: str
) -> None:
    job_dir = tmp_path / "job"
    _touch(job_dir, "x.cell")
    export = JobFilesExport(make_job("Mu 1"), seed, job_dir, (job_dir / "x.cell",), None)

    pairs = plan_job_file_copy([export], tmp_path / "out")

    assert pairs == [(job_dir / "x.cell", tmp_path / "out" / "Mu_1" / "x.cell")]


def test_job_files_export_explains_a_folder_it_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = tmp_path / "Si"
    _touch(job_dir, "Si.cell")

    def refuse(_self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "iterdir", refuse)

    export = job_files_export(make_job("Si"), job_dir, "Si")

    assert export.files == ()
    assert export.directory == job_dir
    assert export.unavailable_reason == "Could not read this job's folder: Permission denied"


def test_copies_onto_source_spots_a_destination_that_holds_the_job_folder(tmp_path: Path) -> None:
    job_dir = tmp_path / "files" / "Si"
    _touch(job_dir, "Si.cell")
    export = job_files_export(make_job("Si"), job_dir, "Si")

    assert copies_onto_source(plan_job_file_copy([export], job_dir.parent))
    assert not copies_onto_source(plan_job_file_copy([export], tmp_path / "out"))
