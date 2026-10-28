"""Tests for Hyperfine Coupling CASTEP magres job expansion."""

from __future__ import annotations

from uuid import UUID

import pytest

from mufinder.castep_io import write_cell, write_param
from mufinder.castep_io.types import KeywordAssignment, ValueWithUnit
from mufinder.domain.enums import WorkflowStage
from mufinder.workflow.bridge import structure_to_spec
from mufinder.workflow.hyperfine import (
    HYPERFINE_CALC_TYPE,
    HyperfineJobConfig,
    expand_hyperfine_jobs,
)
from mufinder.workflow.quantum import QuantumPhononSource
from mufinder.workflow.types import InputTemplate, ParamTemplate
from tests.support.structures import QUARTER_DIAGONAL, host_structure

_RUN_ID = UUID("11111111-1111-1111-1111-111111111111")
_SITE_ID = UUID("22222222-2222-2222-2222-222222222222")


def _source(label: str = "site1", charge_state: str | None = "positive_muon"):
    structure = host_structure(
        label="relaxed site",
        origin="muon_relaxation",
        species=("Si", "H:mu"),
        positions=QUARTER_DIAGONAL,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    return QuantumPhononSource(
        label=label,
        structure=structure_to_spec(structure),
        source_site_id=_SITE_ID,
        source_relaxation_run_id=_RUN_ID,
        charge_state=charge_state,
    )


def test_expands_one_spin_polarised_magres_job_per_site() -> None:
    (job,) = expand_hyperfine_jobs(
        jobs=[HyperfineJobConfig(source=_source())], name_prefix="hyperfine"
    )

    param = write_param(job.param_file)
    cell = write_cell(job.cell_file)
    assert "TASK : MAGRES" in param
    assert "MAGRES_TASK : HYPERFINE" in param
    assert "SPIN_POLARIZED : TRUE" in param
    assert "WRITE_FORMATTED_DENSITY : TRUE" in param
    assert "CHARGE : 1" in param
    assert "%BLOCK SPECIES_GAMMA" in cell
    assert "H:mu" in cell.split("%BLOCK SPECIES_GAMMA")[1]
    assert job.workflow_stage == WorkflowStage.HYPERFINE_COUPLING
    assert job.provenance["calc_type"] == HYPERFINE_CALC_TYPE
    assert job.provenance["source_site_id"] == str(_SITE_ID)
    assert job.name == "hyperfine_site1"


def test_muonium_is_neutral_with_a_seeded_muon_spin() -> None:
    (job,) = expand_hyperfine_jobs(
        jobs=[HyperfineJobConfig(source=_source("mu1", "muonium"))], name_prefix="hyperfine"
    )

    assert "CHARGE : 0" in write_param(job.param_file)
    muon_row = next(
        line for line in write_cell(job.cell_file).splitlines() if line.startswith("H:mu")
    )
    assert "SPIN=1" in muon_row


def test_row_settings_are_layered_but_controlled_keywords_are_refused() -> None:
    tolerance = ParamTemplate(
        keywords=(KeywordAssignment("elec_energy_tol", ValueWithUnit(1e-8, "eV")),)
    )
    (job,) = expand_hyperfine_jobs(
        jobs=[HyperfineJobConfig(source=_source(), param_overrides=tolerance)],
        name_prefix="hyperfine",
    )
    assert "ELEC_ENERGY_TOL : 1e-08 eV" in write_param(job.param_file)

    for name, value in (("task", "SINGLEPOINT"), ("spin_polarized", False)):
        override = ParamTemplate(keywords=(KeywordAssignment(name, value),))
        with pytest.raises(ValueError, match="set by the Hyperfine Coupling workflow"):
            expand_hyperfine_jobs(
                jobs=[HyperfineJobConfig(source=_source(), param_overrides=override)],
                name_prefix="hyperfine",
            )
        with pytest.raises(ValueError, match="set by the Hyperfine Coupling workflow"):
            expand_hyperfine_jobs(
                jobs=[HyperfineJobConfig(source=_source())],
                name_prefix="hyperfine",
                user_overrides=InputTemplate(param=override),
            )


def test_requires_jobs_and_labels() -> None:
    with pytest.raises(ValueError, match="At least one hyperfine job"):
        expand_hyperfine_jobs(jobs=[], name_prefix="hyperfine")
    with pytest.raises(ValueError, match="labels must not be empty"):
        expand_hyperfine_jobs(
            jobs=[HyperfineJobConfig(source=_source(label="  "))], name_prefix="hyperfine"
        )
