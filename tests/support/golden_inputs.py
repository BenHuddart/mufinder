"""Deterministic input builders for CASTEP golden-rendering characterization.

Each ``build_*`` function returns a list of ``(job_name, JobSpec)`` pairs. The
same builders are used both to (re)generate golden ``.cell`` / ``.param`` files
and to compare against them, so the fixtures cannot drift from the code.

Inputs are fully self-contained: no network, no CASTEP binary, no
``castep_keywords.json``. All UUIDs used by adapters live only in job
``provenance`` (never rendered), so ``render_job`` output is deterministic.

Run ``python -m tests.support.golden_inputs`` to (re)write the golden files
under ``tests/data/golden_inputs/`` after eyeballing the change.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from mufinder.app.convergence_adapter import expand_convergence_jobs
from mufinder.app.convergence_types import ConvergenceConfig
from mufinder.app.relaxation_adapter import (
    expand_muon_relaxation_jobs,
    expand_pristine_relaxation_jobs,
)
from mufinder.app.relaxation_types import (
    MuonRelaxationConfig,
    MuonRelaxationRow,
    PristineRelaxationConfig,
    PristineRelaxationRow,
)
from mufinder.app.uep_adapter import expand_uep_jobs
from mufinder.app.uep_types import UepConfig, UepSourceRow
from mufinder.castep_io.types import AtomicPosition, LatticeCart, PositionBlock
from mufinder.domain.models import Structure
from mufinder.workflow.bridge import structure_to_spec
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
from mufinder.workflow.renderer import render_job
from mufinder.workflow.types import JobSpec, StructureSpec

GOLDEN_DIR = Path("tests/data/golden_inputs")

# Fixed UUIDs so provenance is stable; these are never rendered into cell/param.
_SITE_A = UUID("00000000-0000-0000-0000-0000000000a1")
_SITE_B = UUID("00000000-0000-0000-0000-0000000000b2")
_RUN_A = UUID("00000000-0000-0000-0000-0000000000c3")
_RUN_B = UUID("00000000-0000-0000-0000-0000000000d4")
_JOB_A = UUID("00000000-0000-0000-0000-0000000000e5")
_JOB_B = UUID("00000000-0000-0000-0000-0000000000f6")


def _si_structure() -> Structure:
    return Structure(
        label="Si",
        lattice_vectors=[[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _relaxed_muon_structure() -> Structure:
    from mufinder.domain.enums import WorkflowStage

    return Structure(
        label="relaxed site",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )


# --------------------------------------------------------------------------
# Convergence
# --------------------------------------------------------------------------
def build_convergence_cutoff() -> list[tuple[str, JobSpec]]:
    config = ConvergenceConfig(
        sweep_parameter="cut_off_energy",
        sweep_values=(300.0, 400.0),
        cutoff_energies=(300.0, 400.0),
        shared_params={
            "task": "SINGLEPOINT",
            "xc_functional": "PBE",
            "max_scf_cycles": 80,
            "kpoint_mp_grid": (4, 4, 4),
        },
        name_prefix="convergence",
    )
    jobs = expand_convergence_jobs(structure=structure_to_spec(_si_structure()), config=config)
    return [(job.name, job) for job in jobs]


def build_convergence_kpoint() -> list[tuple[str, JobSpec]]:
    config = ConvergenceConfig(
        sweep_parameter="kpoint_mp_grid",
        sweep_values=((2, 2, 2), (4, 4, 4)),
        shared_params={
            "task": "SINGLEPOINT",
            "xc_functional": "PBE",
            "cut_off_energy": 400.0,
        },
        name_prefix="convergence",
    )
    jobs = expand_convergence_jobs(structure=structure_to_spec(_si_structure()), config=config)
    return [(job.name, job) for job in jobs]


# --------------------------------------------------------------------------
# Relaxation
# --------------------------------------------------------------------------
def build_pristine_relaxation() -> list[tuple[str, JobSpec]]:
    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(
            PristineRelaxationRow(
                label="unit",
                fix_all_cell=True,
                supercell_matrix=None,
                parameter_values={"task": "GEOMETRYOPTIMIZATION"},
            ),
            # Supercell + rescaled k-point grid case.
            PristineRelaxationRow(
                label="super2x1x1",
                fix_all_cell=False,
                supercell_matrix=((2, 0, 0), (0, 1, 0), (0, 0, 1)),
                parameter_values={"task": "GEOMETRYOPTIMIZATION"},
            ),
        ),
    )
    host = _si_structure()
    expansion = expand_pristine_relaxation_jobs(
        host_structure=host,
        config=config,
        project_cutoff_energy=600.0,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=host.lattice_vectors,
    )
    return [(job.name, job) for job in expansion.jobs]


def build_muon_relaxation() -> list[tuple[str, JobSpec]]:
    config = MuonRelaxationConfig(
        rows=(
            # Diamagnetic positive muon.
            MuonRelaxationRow(
                site_id=_SITE_A,
                label="site_a",
                initial_fractional_position=(0.2, 0.4, 0.6),
                charge_state="positive_muon",
            ),
            # Open-shell muonium (spin-seeded).
            MuonRelaxationRow(
                site_id=_SITE_B,
                label="site_b",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state="muonium",
            ),
        )
    )
    host = _si_structure()
    expansion = expand_muon_relaxation_jobs(
        host_structure=host,
        config=config,
        project_cutoff_energy=600.0,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=host.lattice_vectors,
    )
    return [(job.name, job) for job in expansion.jobs]


# --------------------------------------------------------------------------
# UEP
# --------------------------------------------------------------------------
def build_uep() -> list[tuple[str, JobSpec]]:
    config = UepConfig(
        name_prefix="uep",
        rows=(UepSourceRow(label="source_norelax", parameter_values={}),),
    )
    host = _si_structure()
    expansion = expand_uep_jobs(
        host_structure=host,
        config=config,
        project_cutoff_energy=650.0,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=host.lattice_vectors,
    )
    return [(job.name, job) for job in expansion.jobs]


# --------------------------------------------------------------------------
# Quantum corrections
# --------------------------------------------------------------------------
def _quantum_source(*, muonium: bool) -> QuantumPhononSource:
    return QuantumPhononSource(
        label="mu1" if muonium else "site1",
        structure=structure_to_spec(_relaxed_muon_structure()),
        source_site_id=_SITE_A,
        source_relaxation_run_id=_RUN_A,
        source_relaxation_job_id=_JOB_A,
        charge_state="muonium" if muonium else "positive_muon",
        relative_energy_eV=-10.0,
    )


def build_quantum_singlepoint() -> list[tuple[str, JobSpec]]:
    jobs = expand_quantum_singlepoint_jobs(
        jobs=(
            QuantumPhononJobConfig(source=_quantum_source(muonium=False)),
            QuantumPhononJobConfig(source=_quantum_source(muonium=True)),
        ),
        name_prefix="quantum_singlepoint",
    )
    return [(job.name, job) for job in jobs]


def build_quantum_phonon() -> list[tuple[str, JobSpec]]:
    jobs = expand_quantum_phonon_jobs(
        jobs=(
            QuantumPhononJobConfig(source=_quantum_source(muonium=False)),
            QuantumPhononJobConfig(source=_quantum_source(muonium=True)),
        ),
        name_prefix="quantum_phonon",
    )
    return [(job.name, job) for job in jobs]


# --------------------------------------------------------------------------
# NEB (energy barriers)
# --------------------------------------------------------------------------
def _neb_reactant_spec(*, spin: bool) -> StructureSpec:
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
        atomic_spins=(1.0, -1.0, 0.0) if spin else None,
        label="reactant",
    )


def build_neb() -> list[tuple[str, JobSpec]]:
    # Non-spin case, with intermediate image.
    plain = NudgedElasticBandSource(
        label="siteA_to_siteB_path1_neb",
        reactant_structure=_neb_reactant_spec(spin=False),
        product_positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (0.5, 0.5, 0.5)),
                AtomicPosition("H:mu", (0.75, 0.25, 0.25)),
            )
        ),
        intermediate_positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (0.5, 0.5, 0.5)),
                AtomicPosition("H:mu", (0.5, 0.25, 0.25)),
            )
        ),
        source_site_ids=(_SITE_A, _SITE_B),
        source_relaxation_run_ids=(_RUN_A, _RUN_B),
        source_relaxation_job_ids=(_JOB_A, _JOB_B),
        path_length_ang=2.0,
    )
    # Spin-polarized case, with intermediate image and seeded spins.
    spin = NudgedElasticBandSource(
        label="siteA_to_siteB_spin_neb",
        reactant_structure=_neb_reactant_spec(spin=True),
        product_positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (0.5, 0.5, 0.5)),
                AtomicPosition("H:mu", (0.75, 0.25, 0.25)),
            )
        ),
        intermediate_positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (0.5, 0.5, 0.5)),
                AtomicPosition("H:mu", (0.5, 0.25, 0.25)),
            )
        ),
        product_atomic_spins=(1.0, -1.0, 0.0),
        intermediate_atomic_spins=(1.0, -1.0, 0.0),
        source_site_ids=(_SITE_A, _SITE_B),
        source_relaxation_run_ids=(_RUN_A, _RUN_B),
        source_relaxation_job_ids=(_JOB_A, _JOB_B),
        path_length_ang=2.0,
    )
    jobs = expand_nudged_elastic_band_jobs(
        jobs=[
            NudgedElasticBandJobConfig(source=plain),
            NudgedElasticBandJobConfig(source=spin),
        ],
        name_prefix="barrier",
    )
    return [(job.name, job) for job in jobs]


# Workflow name -> builder. The workflow name is the golden subdirectory.
BUILDERS = {
    "convergence_cutoff": build_convergence_cutoff,
    "convergence_kpoint": build_convergence_kpoint,
    "pristine_relaxation": build_pristine_relaxation,
    "muon_relaxation": build_muon_relaxation,
    "uep": build_uep,
    "quantum_singlepoint": build_quantum_singlepoint,
    "quantum_phonon": build_quantum_phonon,
    "neb": build_neb,
}


def iter_golden_cases():
    """Yield (workflow, job_name, cell_text, param_text) for every golden job."""

    for workflow, builder in BUILDERS.items():
        for job_name, job in builder():
            cell_text, param_text = render_job(job)
            yield workflow, job_name, cell_text, param_text


def write_golden_files(base_dir: Path = GOLDEN_DIR) -> list[Path]:
    """(Re)generate all golden files. Returns the list of written paths."""

    written: list[Path] = []
    for workflow, job_name, cell_text, param_text in iter_golden_cases():
        workflow_dir = base_dir / workflow
        workflow_dir.mkdir(parents=True, exist_ok=True)
        cell_path = workflow_dir / f"{job_name}.cell"
        param_path = workflow_dir / f"{job_name}.param"
        cell_path.write_text(cell_text, encoding="utf-8")
        param_path.write_text(param_text, encoding="utf-8")
        written.extend([cell_path, param_path])
    return written


if __name__ == "__main__":
    paths = write_golden_files()
    for path in paths:
        print(path)
