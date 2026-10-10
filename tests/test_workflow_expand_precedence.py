"""Per-workflow template-precedence regression tests for the ``expand_*`` API.

These pin the 5-layer template stack for the function-based workflows
(relaxation, energy_barriers, quantum). For each
``expand_*`` function they prove:

(a) ``structure_defaults`` is honored and sits *between* ``workflow_defaults``
    and the per-job patch layer (peel-off style, on a keyword no base template
    sets);
(b) the function's OWN base template still beats caller ``workflow_defaults``
    AND ``structure_defaults`` that try to set the same keyword — this pins the
    base template's placement in the PATCH layer and is the regression trap this
    PR exists to guard;
(c) per-config ``param_overrides``/``cell_overrides`` still beat the function's
    base template.

The base templates are deliberately placed in the patch layer (above
``workflow_defaults``/``structure_defaults``); moving them into
``workflow_defaults`` would flip (b) and is exactly what these tests catch.
"""

from __future__ import annotations

from typing import Callable
from uuid import uuid4

import pytest

from mufinder.castep_io.types import (
    AtomicPosition,
    KeywordAssignment,
    LatticeCart,
    PositionBlock,
)
from mufinder.workflow.energy_barriers import (
    NudgedElasticBandJobConfig,
    NudgedElasticBandSource,
    expand_nudged_elastic_band_jobs,
)
from mufinder.workflow.quantum import (
    QuantumPhononJobConfig,
    QuantumPhononSource,
    expand_quantum_phonon_jobs,
    expand_quantum_singlepoint_jobs,
)
from mufinder.workflow.relaxation import RelaxationJobConfig, expand_relaxation_jobs
from mufinder.workflow.types import (
    CellTemplate,
    InputTemplate,
    ParamTemplate,
    StructureSpec,
)


def _structure() -> StructureSpec:
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
        atomic_spins=(0.0, 0.0, 0.0),
        label="src",
    )


def _param(name: str, value: object) -> InputTemplate:
    return InputTemplate(param=ParamTemplate(keywords=(KeywordAssignment(name, value),)))


def _param_value(job, name: str):
    for keyword in job.param_file.keywords:
        if keyword.name == name:
            return keyword.value
    raise AssertionError(f"missing param keyword {name}")


# One expander adapter per workflow: given the layer kwargs, return a single
# JobSpec. ``param_overrides`` lets test (c) push a per-config override.
def _run_relaxation(**layers) -> object:
    overrides = layers.pop("param_overrides", None)
    jobs = expand_relaxation_jobs(
        jobs=[RelaxationJobConfig(label="j", structure=_structure(), param_overrides=overrides)],
        name_prefix="relax",
        **layers,
    )
    return jobs[0]


def _neb_source() -> NudgedElasticBandSource:
    return NudgedElasticBandSource(
        label="path_neb",
        reactant_structure=_structure(),
        product_positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (0.5, 0.5, 0.5)),
                AtomicPosition("H:mu", (0.30, 0.30, 0.30)),
            )
        ),
        source_site_ids=(uuid4(), uuid4()),
        source_relaxation_run_ids=(uuid4(), uuid4()),
        source_relaxation_job_ids=(uuid4(), uuid4()),
    )


def _run_neb(**layers) -> object:
    overrides = layers.pop("param_overrides", None)
    jobs = expand_nudged_elastic_band_jobs(
        jobs=[NudgedElasticBandJobConfig(source=_neb_source(), param_overrides=overrides)],
        name_prefix="barrier",
        **layers,
    )
    return jobs[0]


def _quantum_source() -> QuantumPhononSource:
    return QuantumPhononSource(
        label="site",
        structure=_structure(),
        source_site_id=uuid4(),
        source_relaxation_run_id=uuid4(),
    )


def _run_quantum_singlepoint(**layers) -> object:
    overrides = layers.pop("param_overrides", None)
    jobs = expand_quantum_singlepoint_jobs(
        jobs=[QuantumPhononJobConfig(source=_quantum_source(), param_overrides=overrides)],
        name_prefix="q",
        **layers,
    )
    return jobs[0]


def _run_quantum_phonon(**layers) -> object:
    overrides = layers.pop("param_overrides", None)
    jobs = expand_quantum_phonon_jobs(
        jobs=[QuantumPhononJobConfig(source=_quantum_source(), param_overrides=overrides)],
        name_prefix="q",
        **layers,
    )
    return jobs[0]


# (runner, base-template keyword the workflow sets, expected base value)
_WORKFLOWS: list[tuple[str, Callable[..., object], str, object]] = [
    ("relaxation", _run_relaxation, "task", "GEOMETRYOPTIMIZATION"),
    ("neb", _run_neb, "task", "TRANSITIONSTATESEARCH"),
    ("quantum_singlepoint", _run_quantum_singlepoint, "task", "SINGLEPOINT"),
    ("quantum_phonon", _run_quantum_phonon, "task", "PHONON"),
]


@pytest.mark.parametrize("name, run, base_keyword, base_value", _WORKFLOWS, ids=[w[0] for w in _WORKFLOWS])
def test_structure_defaults_sits_between_workflow_defaults_and_patch(
    name: str, run: Callable[..., object], base_keyword: str, base_value: object
) -> None:
    # (a) On a keyword no base template sets (max_scf_cycles), peel the layers
    # off from the top and assert the next-highest present layer wins:
    # workflow_defaults(20) < structure_defaults(30) < patch(config override 40).
    override = _param("max_scf_cycles", 40)

    # structure_defaults present and above workflow_defaults -> 30.
    job = run(
        workflow_defaults=_param("max_scf_cycles", 20),
        structure_defaults=_param("max_scf_cycles", 30),
    )
    assert _param_value(job, "max_scf_cycles") == 30

    # Drop structure_defaults -> workflow_defaults wins (20). Proves the layer
    # was actually being honored above.
    job = run(workflow_defaults=_param("max_scf_cycles", 20))
    assert _param_value(job, "max_scf_cycles") == 20

    # Per-config override lands in the patch layer, above structure_defaults -> 40.
    job = run(
        workflow_defaults=_param("max_scf_cycles", 20),
        structure_defaults=_param("max_scf_cycles", 30),
        param_overrides=override.param,
    )
    assert _param_value(job, "max_scf_cycles") == 40


@pytest.mark.parametrize("name, run, base_keyword, base_value", _WORKFLOWS, ids=[w[0] for w in _WORKFLOWS])
def test_base_template_beats_workflow_and_structure_defaults(
    name: str, run: Callable[..., object], base_keyword: str, base_value: object
) -> None:
    # (b) THE REGRESSION TRAP. workflow_defaults AND structure_defaults both try
    # to set the base-template keyword; the function's base value must still win,
    # because the base template is composed into the PATCH layer (above both).
    job = run(
        workflow_defaults=_param(base_keyword, "WORKFLOW_OVERRIDE"),
        structure_defaults=_param(base_keyword, "STRUCTURE_OVERRIDE"),
    )
    assert _param_value(job, base_keyword) == base_value


@pytest.mark.parametrize("name, run, base_keyword, base_value", _WORKFLOWS, ids=[w[0] for w in _WORKFLOWS])
def test_config_overrides_beat_base_template(
    name: str, run: Callable[..., object], base_keyword: str, base_value: object
) -> None:
    # (c) A per-config param override for the SAME base keyword wins, because the
    # base template is composed first and config overrides last within the patch.
    job = run(param_overrides=ParamTemplate(keywords=(KeywordAssignment(base_keyword, "CONFIG_OVERRIDE"),)))
    assert _param_value(job, base_keyword) == "CONFIG_OVERRIDE"


def test_relaxation_cell_override_beats_base_and_defaults_via_kpoint_grid() -> None:
    # cell_overrides path: relaxation's cell base has no kpoint grid unless the
    # config supplies one; a cell_override grid must win over workflow/structure
    # defaults (it lands in the patch layer).
    jobs = expand_relaxation_jobs(
        jobs=[
            RelaxationJobConfig(
                label="j",
                structure=_structure(),
                cell_overrides=CellTemplate(kpoint_mp_grid=(5, 5, 5)),
            )
        ],
        name_prefix="relax",
        workflow_defaults=InputTemplate(cell=CellTemplate(kpoint_mp_grid=(2, 2, 2))),
        structure_defaults=InputTemplate(cell=CellTemplate(kpoint_mp_grid=(3, 3, 3))),
    )
    assert jobs[0].cell_file.kpoint_mp_grid == (5, 5, 5)
