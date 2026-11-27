"""Tests for rendering resolved jobs to local filesystem output."""

from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from mufinder.castep_io.types import KeywordAssignment
from mufinder.domain.models import Structure
from mufinder.workflow.bridge import structure_to_spec
from mufinder.workflow.convergence import expand_convergence_sweep_jobs
from mufinder.workflow.output import restore_job_inputs, write_job_directories, write_job_set
from mufinder.workflow.types import InputTemplate, ParamTemplate


def _jobs() -> list[object]:
    structure = Structure(
        label="Si",
        lattice_vectors=[[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    return expand_convergence_sweep_jobs(
        structure=structure_to_spec(structure),
        sweep_parameter="cut_off_energy",
        sweep_values=[300.0, 340.0],
        name_prefix="conv",
        workflow_defaults=InputTemplate(
            param=ParamTemplate(
                keywords=(
                    KeywordAssignment("task", "SINGLEPOINT"),
                    KeywordAssignment("xc_functional", "PBE"),
                )
            )
        ),
    )


def test_write_job_set_creates_input_files_and_manifest(tmp_path: Path) -> None:
    jobs = _jobs()
    out_dir = tmp_path / "jobs"

    manifest_path = write_job_set(jobs, out_dir)

    assert manifest_path == out_dir / "manifest.json"
    assert manifest_path.exists()

    for job in jobs:
        job_dir = out_dir / job.name
        assert job_dir.exists()

        cell_path = job_dir / f"{job.name}.cell"
        param_path = job_dir / f"{job.name}.param"
        assert cell_path.exists()
        assert param_path.exists()

        assert "%BLOCK LATTICE_CART" in cell_path.read_text(encoding="utf-8")
        assert "TASK : SINGLEPOINT" in param_path.read_text(encoding="utf-8")
        assert b"\r\n" not in cell_path.read_bytes()
        assert b"\r\n" not in param_path.read_bytes()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert len(manifest["jobs"]) == 2
    assert manifest["jobs"][0]["name"] == jobs[0].name
    assert b"\r\n" not in manifest_path.read_bytes()


def test_write_job_directories_writes_every_job(tmp_path: Path) -> None:
    jobs = _jobs()

    directories = write_job_directories([(job, tmp_path) for job in jobs])

    assert directories == [tmp_path / job.name for job in jobs]
    assert all((tmp_path / job.name / f"{job.name}.param").is_file() for job in jobs)


def test_write_job_directories_that_fails_part_way_leaves_the_folder_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _jobs()[:2]
    existing = tmp_path / first.name
    existing.mkdir()
    (existing / f"{first.name}.cell").write_text("old cell", encoding="utf-8")
    write_text = Path.write_text

    def full_disk(path: Path, *args, **kwargs):
        if path.parent.name == second.name and path.suffix == ".param":
            raise OSError(errno.ENOSPC, "No space left on device")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", full_disk)

    with pytest.raises(OSError):
        write_job_directories([(first, tmp_path), (second, tmp_path)])

    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == [
        first.name,
        f"{first.name}/{first.name}.cell",
    ]
    assert (existing / f"{first.name}.cell").read_text(encoding="utf-8") == "old cell"


def test_write_job_directories_that_fails_removes_the_parent_folders_it_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (job,) = _jobs()[:1]
    root = tmp_path / "files" / "pristine_relaxation"
    write_text = Path.write_text

    def full_disk(path: Path, *args, **kwargs):
        if path.suffix == ".param":
            raise OSError(errno.ENOSPC, "No space left on device")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", full_disk)

    with pytest.raises(OSError):
        write_job_directories([(job, root)])

    assert list(tmp_path.iterdir()) == []


def test_restore_job_inputs_puts_back_one_jobs_inputs(tmp_path: Path) -> None:
    first, second = _jobs()[:2]
    (tmp_path / first.name).mkdir()
    (tmp_path / first.name / f"{first.name}.cell").write_text("old cell", encoding="utf-8")
    previous: dict[Path, bytes | None] = {}

    write_job_directories([(first, tmp_path), (second, tmp_path)], previous_inputs=previous)
    restore_job_inputs(previous, tmp_path / first.name)

    assert sorted(path.name for path in (tmp_path / first.name).iterdir()) == [f"{first.name}.cell"]
    assert (tmp_path / first.name / f"{first.name}.cell").read_text(encoding="utf-8") == "old cell"
    assert (tmp_path / second.name / f"{second.name}.param").is_file()
