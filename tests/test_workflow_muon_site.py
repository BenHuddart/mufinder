"""Tests for muon-site workflow expansion."""

from __future__ import annotations

from mufinder.castep_io.types import KeywordAssignment
from mufinder.domain.models import Structure
from mufinder.workflow.bridge import structure_to_spec
from mufinder.workflow.muon_site import MuonSiteWorkflow
from mufinder.workflow.types import InputTemplate, ParamTemplate


def _base_structure_spec() -> object:
    structure = Structure(
        label="Host",
        lattice_vectors=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    return structure_to_spec(structure)


def test_muon_site_workflow_expands_structure_variations() -> None:
    workflow = MuonSiteWorkflow(
        host_structure=_base_structure_spec(),
        muon_positions_frac=[(0.1, 0.2, 0.3), (0.8, 0.4, 0.6)],
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

    assert len(jobs) == 2
    assert [job.name for job in jobs] == ["muon_site_site_001", "muon_site_site_002"]
    assert len(jobs[0].cell_file.positions_frac.positions) == 3
    assert jobs[0].cell_file.positions_frac.positions[-1].species == "H:mu"


def test_muon_site_workflow_rejects_bad_position_shape() -> None:
    workflow = MuonSiteWorkflow(
        host_structure=_base_structure_spec(),
        muon_positions_frac=[(0.1, 0.2)],
    )

    try:
        workflow.expand()
    except ValueError as exc:
        assert "three components" in str(exc)
    else:
        raise AssertionError("Expected ValueError for malformed muon position")
