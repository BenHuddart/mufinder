"""Tests for convergence workflow expansion."""

from __future__ import annotations

from mufinder.castep_io.types import KeywordAssignment, ValueWithUnit
from mufinder.domain.models import Structure
from mufinder.workflow.bridge import structure_to_spec
from mufinder.workflow.convergence import ConvergenceWorkflow
from mufinder.workflow.types import InputTemplate, ParamTemplate


def _base_structure_spec() -> object:
    structure = Structure(
        label="Si",
        lattice_vectors=[[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    return structure_to_spec(structure)


def _find_value(keywords: tuple[KeywordAssignment, ...], name: str) -> object:
    for item in keywords:
        if item.name == name:
            return item.value
    raise AssertionError(f"Keyword {name} missing")


def test_convergence_workflow_expands_cutoff_sweep() -> None:
    workflow = ConvergenceWorkflow(
        structure=_base_structure_spec(),
        sweep_parameter="cut_off_energy",
        sweep_values=[300.0, 360.0],
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

    jobs = workflow.expand()

    assert [job.name for job in jobs] == ["conv_ecut_300", "conv_ecut_360"]
    assert len(jobs) == 2

    first_cutoff = _find_value(jobs[0].param_file.keywords, "cut_off_energy")
    assert isinstance(first_cutoff, ValueWithUnit)
    assert first_cutoff.value == 300.0


def test_convergence_workflow_rejects_empty_sweep() -> None:
    workflow = ConvergenceWorkflow(
        structure=_base_structure_spec(),
        sweep_parameter="cut_off_energy",
        sweep_values=[],
    )

    try:
        workflow.expand()
    except ValueError as exc:
        assert "at least one sweep value" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty sweep")
