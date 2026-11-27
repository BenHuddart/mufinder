from __future__ import annotations

from uuid import uuid4

from mufinder.castep_io.types import AtomicPosition, LatticeCart, PositionBlock, ValueWithUnit
from mufinder.domain.enums import WorkflowStage
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    NudgedElasticBandJobConfig,
    NudgedElasticBandSource,
    expand_nudged_elastic_band_jobs,
    merge_serialized_sources,
    replace_serialized_sources_for_pair,
    strip_legacy_energy_barrier_params,
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


def test_expand_nudged_elastic_band_jobs_renders_intermediate_blocks() -> None:
    source = NudgedElasticBandSource(
        label="siteA_to_siteB_path1_neb",
        reactant_structure=_structure_spec(),
        product_positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (2.0, 2.0, 2.0)),
                AtomicPosition("H:mu", (3.0, 1.0, 1.0)),
            ),
            unit="ang",
        ),
        intermediate_positions_frac=PositionBlock(
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
    assert any(block.name == "positions_frac_product" for block in job.cell_file.blocks)
    assert any(block.name == "positions_frac_intermediate" for block in job.cell_file.blocks)
    assert any(
        keyword.name == "tssearch_method" and keyword.value == "NEB"
        for keyword in job.param_file.keywords
    )


def test_expand_nudged_elastic_band_jobs_sets_task_and_spin_polarisation_from_reactant_spins() -> (
    None
):
    spinless = _structure_spec()
    spinless = StructureSpec(
        lattice_cart=spinless.lattice_cart,
        positions_frac=spinless.positions_frac,
        species=spinless.species,
        atomic_spins=(0.0, 0.0, 0.0),
        label=spinless.label,
    )

    def _params(reactant: StructureSpec) -> dict[str, object]:
        source = NudgedElasticBandSource(
            label="path",
            reactant_structure=reactant,
            product_positions_frac=reactant.positions_frac,
        )
        (job,) = expand_nudged_elastic_band_jobs(
            jobs=[NudgedElasticBandJobConfig(source=source)], name_prefix="barrier"
        )
        return {keyword.name: keyword.value for keyword in job.param_file.keywords}

    magnetic = _params(_structure_spec())
    assert magnetic["task"] == "TRANSITIONSTATESEARCH"
    assert magnetic["tssearch_method"] == "NEB"
    assert magnetic["spin_polarized"] is True
    assert "spin_polarized" not in _params(spinless)


def test_expand_nudged_elastic_band_jobs_without_intermediate_omits_block() -> None:
    # The renderer faithfully omits the intermediate block when none is supplied. (A real CASTEP
    # NEB run requires the intermediate; the app always supplies the minimum-image midpoint, but
    # the lower-level expander must not invent one.)
    source = NudgedElasticBandSource(
        label="siteA_to_siteB_path1_neb",
        reactant_structure=_structure_spec(),
        product_positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (2.0, 2.0, 2.0)),
                AtomicPosition("H:mu", (3.0, 1.0, 1.0)),
            ),
            unit="ang",
        ),
        intermediate_positions_frac=None,
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
    assert "source_transition_state_job_id" not in job.provenance
    assert any(block.name == "positions_frac_product" for block in job.cell_file.blocks)
    assert not any(block.name == "positions_frac_intermediate" for block in job.cell_file.blocks)
    assert any(
        keyword.name == "tssearch_method" and keyword.value == "NEB"
        for keyword in job.param_file.keywords
    )
    # Climbing image is enabled by default so the band yields the saddle energy.
    assert any(
        keyword.name == "tssearch_neb_climbing" and keyword.value is True
        for keyword in job.param_file.keywords
    )


def _neb_source() -> NudgedElasticBandSource:
    return NudgedElasticBandSource(
        label="siteA_to_siteB_path1_neb",
        reactant_structure=_structure_spec(),
        product_positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (2.0, 2.0, 2.0)),
                AtomicPosition("H:mu", (3.0, 1.0, 1.0)),
            ),
            unit="ang",
        ),
        intermediate_positions_frac=None,
        product_atomic_spins=(1.0, -1.0, 0.0),
        source_site_ids=(uuid4(), uuid4()),
        source_relaxation_run_ids=(uuid4(), uuid4()),
        source_relaxation_job_ids=(uuid4(), uuid4()),
        path_length_ang=2.0,
    )


def _param_value(job, name: str):
    for keyword in job.param_file.keywords:
        if keyword.name == name:
            return keyword.value
    return None


def test_expand_nudged_elastic_band_jobs_emits_num_farms_when_farming() -> None:
    jobs = expand_nudged_elastic_band_jobs(
        jobs=[NudgedElasticBandJobConfig(source=_neb_source(), num_farms=3)],
        name_prefix="barrier",
    )

    assert _param_value(jobs[0], "num_farms") == 3
    assert jobs[0].provenance["num_farms"] == 3


def test_expand_nudged_elastic_band_jobs_omits_num_farms_when_disabled() -> None:
    # CASTEP defaults NUM_FARMS to 1; writing it explicitly would be noise, and a
    # value of 1 is exactly "farming off".
    jobs = expand_nudged_elastic_band_jobs(
        jobs=[NudgedElasticBandJobConfig(source=_neb_source())],
        name_prefix="barrier",
    )

    assert _param_value(jobs[0], "num_farms") is None
    assert jobs[0].provenance["num_farms"] == 1


def test_merge_serialized_sources_replaces_in_place_and_appends_new_keys() -> None:
    merged = merge_serialized_sources(
        [{"key": "a", "v": 1}, {"key": ""}, "junk", {"key": "b", "v": 2}],
        [{"key": "b", "v": 3}, {"key": "c", "v": 4}, {"v": 5}],
        key_field="key",
    )

    assert merged == [{"key": "a", "v": 1}, {"key": "b", "v": 3}, {"key": "c", "v": 4}]
    assert merge_serialized_sources("not a list", [{"key": "x"}], key_field="key") == [{"key": "x"}]


def test_replace_serialized_sources_for_pair_drops_only_that_pair() -> None:
    existing = [
        {"key": "r:p:1", "reactant_site_id": "r", "product_site_id": "p"},
        {"key": "r:q:1", "reactant_site_id": "r", "product_site_id": "q"},
        {"key": "p:r:1", "reactant_site_id": "p", "product_site_id": "r"},
        {"key": "r:p:2", "reactant_site_id": "r", "product_site_id": "p"},
    ]

    replaced = replace_serialized_sources_for_pair(
        existing,
        reactant_site_id="r",
        product_site_id="p",
        new_sources=[
            {"key": "r:p:1", "reactant_site_id": "r", "product_site_id": "p", "new": True}
        ],
    )

    assert [source["key"] for source in replaced] == ["r:q:1", "p:r:1", "r:p:1"]
    assert replaced[-1]["new"] is True


def test_replace_serialized_sources_for_pair_keeps_the_listed_keys() -> None:
    existing = [
        {"key": "r:p:1", "reactant_site_id": "r", "product_site_id": "p"},
        {"key": "r:p:2", "reactant_site_id": "r", "product_site_id": "p"},
    ]

    replaced = replace_serialized_sources_for_pair(
        existing,
        reactant_site_id="r",
        product_site_id="p",
        new_sources=[{"key": "r:p:3", "reactant_site_id": "r", "product_site_id": "p"}],
        keep_keys={"r:p:2"},
    )

    assert [source["key"] for source in replaced] == ["r:p:2", "r:p:3"]


def test_strip_legacy_energy_barrier_params_drops_placeholders_only() -> None:
    params = {
        "tssearch_lst_max_iter": 10,
        "tssearch_force_tol": ValueWithUnit(0.0, "eV/ang"),
        "tssearch_disp_tol": -1,
        "tssearch_energy_tol": ValueWithUnit(1e-5, "eV"),
        "fine_cut_off_energy": 0,
        "cut_off_energy": 0,
        "tssearch_method": "NEB",
        "tssearch_neb_max_iter": "not-a-number",
    }

    assert strip_legacy_energy_barrier_params(params) == {
        "tssearch_energy_tol": ValueWithUnit(1e-5, "eV"),
        "cut_off_energy": 0,
        "tssearch_method": "NEB",
        "tssearch_neb_max_iter": "not-a-number",
    }


def test_replace_serialized_sources_for_pair_keeps_the_other_charge_states_paths() -> None:
    existing = [
        {
            "key": "r:p:1",
            "reactant_site_id": "r",
            "product_site_id": "p",
            "charge_state": "positive_muon",
        },
        {
            "key": "r:mu:p:mu:1",
            "reactant_site_id": "r",
            "product_site_id": "p",
            "charge_state": "muonium",
        },
        {"key": "legacy", "reactant_site_id": "r", "product_site_id": "p"},
    ]

    replaced = replace_serialized_sources_for_pair(
        existing,
        reactant_site_id="r",
        product_site_id="p",
        new_sources=[{"key": "r:mu:p:mu:2", "reactant_site_id": "r", "product_site_id": "p"}],
        charge_state="muonium",
    )

    assert [source["key"] for source in replaced] == ["r:p:1", "r:mu:p:mu:2"]
