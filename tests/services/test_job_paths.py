"""Tests for resolving a job's directory, seed name and output files."""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.domain.charge_states import MUON_RELAXATION_CALC_TYPES
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Project
from mufinder.services.job_paths import (
    definition_output_directory,
    job_output_directory,
    job_output_files,
    job_seed_name,
    remote_directory_seed_name,
)
from mufinder.services.project_service import ProjectDocument
from mufinder.workflow.naming import job_name
from tests.support.jobs import make_job


def _no_quantum_seed(_definition: CalculationDefinition, key: str) -> str:
    raise AssertionError(f"unexpected quantum lookup for {key}")


@pytest.fixture
def document(tmp_path: Path) -> ProjectDocument:
    root = tmp_path / "project"
    root.mkdir()
    return ProjectDocument(project_root=root, project=Project())


def _add_definition(document: ProjectDocument, **fields) -> CalculationDefinition:
    definition = CalculationDefinition(**fields)
    document.calculations[definition.id] = definition
    return definition


def test_definition_output_directory_resolves_relative_paths(
    document: ProjectDocument, tmp_path: Path
) -> None:
    relative = CalculationDefinition(parameters={"output_directory": "files/uep"})
    absolute = CalculationDefinition(parameters={"output_directory": str(tmp_path / "elsewhere")})
    blank = CalculationDefinition(parameters={"output_directory": "  "})

    assert (
        definition_output_directory(document, relative) == document.project_root / "files" / "uep"
    )
    assert definition_output_directory(document, absolute) == tmp_path / "elsewhere"
    assert definition_output_directory(document, blank) is None
    assert definition_output_directory(None, relative) is None


def test_cutoff_and_label_mappings_resolve_to_label_directories(document: ProjectDocument) -> None:
    cutoff_job = make_job("cutoff_300")
    label_job = make_job("relax_a", workflow_stage=WorkflowStage.PRISTINE_RELAXATION)
    _add_definition(
        document,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        parameters={
            "output_directory": "files/cutoff",
            "job_ids_by_cutoff": {"300": str(cutoff_job.id)},
        },
    )
    _add_definition(
        document,
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        parameters={
            "output_directory": "files/relax",
            "job_ids_by_label": {"a": str(label_job.id)},
        },
    )

    root = document.project_root
    assert job_output_directory(document, cutoff_job, quantum_seed_name=_no_quantum_seed) == (
        root / "files" / "cutoff" / "cutoff_300"
    )
    assert job_output_directory(document, label_job, quantum_seed_name=_no_quantum_seed) == (
        root / "files" / "relax" / "relax_a"
    )


def test_muon_relaxation_job_uses_charge_state_subdirectory_and_seed(
    document: ProjectDocument,
) -> None:
    stored_seed = make_job("site_1", calc_type=MUON_RELAXATION_CALC_TYPES["positive_muon"])
    remote_seed = make_job(
        "site_2",
        calc_type=MUON_RELAXATION_CALC_TYPES["muonium"],
        remote_directory="/scratch/run/muonium_site_2_x/",
    )
    plain = make_job("site_3", calc_type=MUON_RELAXATION_CALC_TYPES["muonium"])
    _add_definition(
        document,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        parameters={
            "output_directory": "files/muon",
            "job_ids_by_label": {
                "s1": str(stored_seed.id),
                "s2": str(remote_seed.id),
                "s3": str(plain.id),
            },
            "job_seed_names_by_label": {"s1": " positivemuon_site_1_seed "},
        },
    )
    base = document.project_root / "files" / "muon"

    assert job_output_directory(document, stored_seed, quantum_seed_name=_no_quantum_seed) == (
        base / "positivemuon" / "positivemuon_site_1_seed"
    )
    assert job_output_directory(document, remote_seed, quantum_seed_name=_no_quantum_seed) == (
        base / "muonium" / "muonium_site_2_x"
    )
    assert (
        job_output_directory(document, plain, quantum_seed_name=_no_quantum_seed)
        == base / "muonium" / "site_3"
    )

    inferred = base / "muonium" / job_name("muonium", "site_3")
    inferred.mkdir(parents=True)
    assert job_output_directory(document, plain, quantum_seed_name=_no_quantum_seed) == inferred


def test_quantum_job_seed_comes_from_the_callback(document: ProjectDocument) -> None:
    job = make_job("phonon", workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS)
    definition = _add_definition(
        document,
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        parameters={
            "output_directory": "files/quantum",
            "phonon_job_ids_by_label": {"key_1": str(job.id)},
        },
    )
    calls: list[tuple[CalculationDefinition, str]] = []

    def seed(owner: CalculationDefinition, key: str) -> str:
        calls.append((owner, key))
        return "quantum_seed"

    assert job_output_directory(document, job, quantum_seed_name=seed) == (
        document.project_root / "files" / "quantum" / "quantum_seed"
    )
    assert calls == [(definition, "key_1")]


def test_legacy_quantum_singlepoint_keeps_its_own_recorded_seed(document: ProjectDocument) -> None:
    """A row given a fresh phonon seed records the seed its legacy singlepoint ran under."""

    singlepoint = make_job("singlepoint", workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS)
    phonon = make_job("phonon", workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS)
    _add_definition(
        document,
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        parameters={
            "output_directory": "files/quantum",
            "singlepoint_job_ids_by_label": {"key_1": str(singlepoint.id)},
            "phonon_job_ids_by_label": {"key_1": str(phonon.id)},
            "singlepoint_seed_names_by_label": {"key_1": "quantum_site"},
        },
    )
    base = document.project_root / "files" / "quantum"

    def seed(_owner: CalculationDefinition, _key: str) -> str:
        return "quantum_site_phonon"

    assert (
        job_output_directory(document, singlepoint, quantum_seed_name=seed) == base / "quantum_site"
    )
    assert (
        job_output_directory(document, phonon, quantum_seed_name=seed)
        == base / "quantum_site_phonon"
    )


def test_legacy_quantum_singlepoint_without_a_recorded_seed_shares_the_rows_seed(
    document: ProjectDocument,
) -> None:
    singlepoint = make_job("singlepoint", workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS)
    _add_definition(
        document,
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        parameters={
            "output_directory": "files/quantum",
            "singlepoint_job_ids_by_label": {"key_1": str(singlepoint.id)},
            "singlepoint_seed_names_by_label": {"other": "quantum_other", "key_1": "  "},
        },
    )

    assert job_output_directory(
        document, singlepoint, quantum_seed_name=lambda _owner, _key: "quantum_site"
    ) == (document.project_root / "files" / "quantum" / "quantum_site")


def test_job_labels_and_fallbacks_require_an_existing_directory(document: ProjectDocument) -> None:
    listed = make_job("listed", workflow_stage=WorkflowStage.UEP)
    _add_definition(
        document,
        workflow_stage=WorkflowStage.UEP,
        parameters={"output_directory": "files/uep", "job_labels": ["listed"]},
    )
    orphan = make_job("orphan", workflow_stage=WorkflowStage.UEP)

    assert job_output_directory(document, listed, quantum_seed_name=_no_quantum_seed) is None
    assert job_output_directory(document, orphan, quantum_seed_name=_no_quantum_seed) is None

    listed_dir = document.project_root / "files" / "uep" / "listed"
    listed_dir.mkdir(parents=True)
    fallback = document.project_root / "files" / "orphan"
    fallback.mkdir(parents=True)

    assert job_output_directory(document, listed, quantum_seed_name=_no_quantum_seed) == listed_dir
    assert job_output_directory(document, orphan, quantum_seed_name=_no_quantum_seed) == fallback
    assert job_output_directory(None, orphan, quantum_seed_name=_no_quantum_seed) is None


def test_job_seed_name_prefers_remote_directory_then_output_directory() -> None:
    remote = make_job("label", remote_directory="/scratch/run/seed_remote")
    local = make_job("label")
    calls: list[str] = []

    def resolve(job) -> Path | None:
        calls.append(job.label)
        return Path("/project/files/seed_local")

    assert job_seed_name(remote, resolve) == "seed_remote"
    assert calls == []
    assert job_seed_name(local, resolve) == "seed_local"
    assert job_seed_name(local, lambda _job: None) == "label"
    assert remote_directory_seed_name(make_job("x", remote_directory="/")) == ""
    assert remote_directory_seed_name(make_job("x", remote_directory="  ")) == ""


def test_job_output_files_lists_present_files_in_order(tmp_path: Path) -> None:
    root = tmp_path / "project"
    output_dir = root / "files" / "job"
    output_dir.mkdir(parents=True)
    for name in ("seed.castep", "seed.cell", "seed_farm001.ts", "other.castep"):
        (output_dir / name).write_text("")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "seed.param").write_text("")

    assert job_output_files(root, output_dir, "seed") == [
        "files/job/seed.cell",
        "files/job/seed.castep",
        "files/job/seed_farm001.ts",
    ]
    assert job_output_files(root, outside, "seed") == [str(outside / "seed.param")]
    assert job_output_files(root, None, "seed") == []
