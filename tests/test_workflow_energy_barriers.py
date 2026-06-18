from __future__ import annotations

from uuid import uuid4

from mufinder.castep_io.types import AtomicPosition, LatticeCart, PositionBlock
from mufinder.domain.enums import WorkflowStage
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    NudgedElasticBandJobConfig,
    NudgedElasticBandSource,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
    TransitionStateSearchJobConfig,
    TransitionStateSearchSource,
    expand_nudged_elastic_band_jobs,
    expand_transition_state_search_jobs,
)
from mufinder.workflow.types import StructureSpec


def _structure_spec() -> StructureSpec:
    return StructureSpec(
        lattice_cart=LatticeCart(
            vectors=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
            unit="ang",
        ),
        positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (0.5, 0.5, 0.5)),
                AtomicPosition("H:mu", (0.25, 0.25, 0.25)),
            )
        ),
        species=("Si", "H:mu"),
        atomic_spins=(1.0, -1.0, 0.0),
        label="reactant",
    )


def test_expand_transition_state_search_jobs_renders_product_block() -> None:
    source = TransitionStateSearchSource(
        label="siteA_to_siteB_path1",
        reactant_structure=_structure_spec(),
        product_positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (0.5, 0.5, 0.5)),
                AtomicPosition("H:mu", (0.75, 0.25, 0.25)),
            )
        ),
        product_atomic_spins=(1.0, -1.0, 0.0),
        source_site_ids=(uuid4(), uuid4()),
        source_relaxation_run_ids=(uuid4(), uuid4()),
        source_relaxation_job_ids=(uuid4(), uuid4()),
        path_start_fractional=(0.25, 0.25, 0.25),
        path_end_fractional=(0.75, 0.25, 0.25),
        path_length_ang=2.0,
    )

    jobs = expand_transition_state_search_jobs(
        jobs=[TransitionStateSearchJobConfig(source=source)],
        name_prefix="barrier",
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job.workflow_stage == WorkflowStage.ENERGY_BARRIERS
    assert job.provenance["calc_type"] == TRANSITION_STATE_SEARCH_CALC_TYPE
    assert any(block.name == "positions_frac_product" for block in job.cell_file.blocks)
    assert any(keyword.name == "task" and keyword.value == "TRANSITIONSTATESEARCH" for keyword in job.param_file.keywords)


def test_expand_nudged_elastic_band_jobs_renders_intermediate_blocks() -> None:
    source = NudgedElasticBandSource(
        label="siteA_to_siteB_path1_neb",
        reactant_structure=_structure_spec(),
        product_positions_abs=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (2.0, 2.0, 2.0)),
                AtomicPosition("H:mu", (3.0, 1.0, 1.0)),
            ),
            unit="ang",
        ),
        intermediate_positions_abs=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (2.0, 2.0, 2.0)),
                AtomicPosition("H:mu", (2.0, 1.5, 1.0)),
            ),
            unit="ang",
        ),
        product_atomic_spins=(1.0, -1.0, 0.0),
        intermediate_atomic_spins=(1.0, -1.0, 0.0),
        source_site_ids=(uuid4(), uuid4()),
        source_relaxation_run_ids=(uuid4(), uuid4()),
        source_relaxation_job_ids=(uuid4(), uuid4()),
        source_transition_state_run_id=uuid4(),
        source_transition_state_job_id=uuid4(),
        path_length_ang=2.0,
    )

    jobs = expand_nudged_elastic_band_jobs(
        jobs=[NudgedElasticBandJobConfig(source=source)],
        name_prefix="barrier",
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job.workflow_stage == WorkflowStage.ENERGY_BARRIERS
    assert job.provenance["calc_type"] == NUDGED_ELASTIC_BAND_CALC_TYPE
    assert any(block.name == "positions_abs_product" for block in job.cell_file.blocks)
    assert any(block.name == "positions_abs_intermediate" for block in job.cell_file.blocks)
    assert any(keyword.name == "tssearch_method" and keyword.value == "NEB" for keyword in job.param_file.keywords)


def test_expand_nudged_elastic_band_jobs_without_intermediate_seeds_from_endpoints() -> None:
    # Direct endpoint seeding: no transition-state intermediate is supplied, so CASTEP
    # interpolates the band itself. The cell must carry only the product block.
    source = NudgedElasticBandSource(
        label="siteA_to_siteB_path1_neb",
        reactant_structure=_structure_spec(),
        product_positions_abs=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (2.0, 2.0, 2.0)),
                AtomicPosition("H:mu", (3.0, 1.0, 1.0)),
            ),
            unit="ang",
        ),
        intermediate_positions_abs=None,
        product_atomic_spins=(1.0, -1.0, 0.0),
        source_site_ids=(uuid4(), uuid4()),
        source_relaxation_run_ids=(uuid4(), uuid4()),
        source_relaxation_job_ids=(uuid4(), uuid4()),
        path_length_ang=2.0,
    )

    jobs = expand_nudged_elastic_band_jobs(
        jobs=[NudgedElasticBandJobConfig(source=source)],
        name_prefix="barrier",
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job.provenance["calc_type"] == NUDGED_ELASTIC_BAND_CALC_TYPE
    assert job.provenance["source_transition_state_job_id"] is None
    assert any(block.name == "positions_abs_product" for block in job.cell_file.blocks)
    assert not any(block.name == "positions_abs_intermediate" for block in job.cell_file.blocks)
    assert any(keyword.name == "tssearch_method" and keyword.value == "NEB" for keyword in job.param_file.keywords)
    # Climbing image is enabled by default so the band yields the saddle energy.
    assert any(
        keyword.name == "tssearch_neb_climbing" and keyword.value is True for keyword in job.param_file.keywords
    )