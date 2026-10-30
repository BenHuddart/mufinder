"""Cobalt difluoride: the worked example behind every screenshot but those of the iron, MnSi and GdRu2Si2 examples.

CoF2 is a rutile antiferromagnet (P4_2/mnm, a = 4.695 Å, c = 3.180 Å, Co at 2a,
F at 4f with x = 0.306; Baur and Khan 1971). The ordered Co moments, 2.64 μB
(Lines 1965), point along c, and the corner and body-centre sublattices are
antiparallel (T_N = 37.85 K). The positive muon stops at the 4c site
(0, 1/2, 0) and pulls its two nearest fluorines into a linear F-mu-F complex
(De Renzi et al. 1984; Möller et al. 2013; Huddart et al. 2022, the MuFinder
paper). ``docs/user_guide/examples/cof2.rst`` tells the story these scenes
illustrate.

:func:`fixture_project` is the configured-but-unsubmitted project the
getting-started and setup scenes open: the host with ``PBE`` and per-atom
spins, a saved 2x2x2 supercell, cutoff and k-point sweeps, and candidate muon
sites. :func:`converged_project` adds adopted convergence defaults with
illustrative pinned plots, and :func:`dashboard_project` binds failed, stopped
and cancelled jobs to three cutoff rows.

:func:`results_project` drives that converged project through every results
workflow in a ``MainWindow``, as the buttons do. No scene runs CASTEP: each
job is submitted through ``JobExecutionController`` with a local runner whose
``submit`` writes the job's outputs (see
:func:`~docs_screenshots.render.synthetic_runs`), and is then finished through
the window's completion handler. The outputs are synthetic, written by
:mod:`docs_screenshots.synthetic` and tuned so that the application's own
analysis reproduces the published numbers:

- the pristine relaxation converges to the host cell with sub-percent changes;
- the UEP source job is a model density of superposed Gaussian Co2+ and F-
  ionic densities (``UEP_SHELLS``), not a CASTEP density;
- each positive-muon start relaxes to one of the three clusters of the
  MuFinder paper (``POSITIVE_SITES``): the 4c F-mu-F site with the paper's
  displacements, and De Renzi's rejected T_A and T_F candidates 0.34 and
  0.82 eV higher; the same starts also relax as muonium, each to the nearer
  of Möller's interstitial site and T_A (``MUONIUM_SITES``);
- the phonon run of the 4c site is a linear F-mu-F complex embedded in the
  supercell with Möller's mode frequencies (``FMUF_MODES_CM``).

The Magnetic Fields scenes run the real ``muesr`` calculation on that project.
"""

from __future__ import annotations

# The render module configures Qt and VTK, so it is imported before them.
from ..render import (
    XC_FUNCTIONAL,
    Dialog,
    Floating,
    Outputs,
    Page,
    Scene,
    Window,
    calculate_field,
    cluster_and_pin,
    current_page,
    focus_default_button,
    frame_viewers,
    hide_catalog_notice,
    include_every_job,
    magnetic_project,
    new_project,
    opened,
    refuse_message_boxes,
    run_uep_analysis,
    save_magnetic_workspace,
    scratch_directory,
    select_all_sites,
    settle,
    show_host_with_spins,
    synthetic_runs,
    view_along_a,
    when_modal,
)
from ..synthetic import (
    bfgs_log_text,
    castep_text,
    cell_text,
    den_fmt_text,
    legacy_castep_text,
    triatomic_phonon_text,
    uep_castep_text,
)

from collections.abc import Callable
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtWidgets import QApplication, QWidget

from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CONVERGED_DEFAULT_METHOD, Structure
from mufinder.domain.text import natural_sort_key
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    build_collinear_component,
)
from mufinder.services.project_service import ProjectDocument, ProjectService

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow

# --- The host -------------------------------------------------------------------

COF2_A = 4.695  # Å, Baur and Khan 1971
COF2_C = 3.180
COF2_X = 0.306  # F at 4f (x, x, 0)
CO_SPIN = 3.0  # hbar/2 on the corner Co; the body-centre Co carries -CO_SPIN
CO_MOMENT_MU_B = 2.64  # ordered moment per Co, Lines 1965
SUPERCELL_2X2X2 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
SUPERCELL_LABEL = "CoF2 2x2x2"


def cof2_host() -> Structure:
    """The CoF2 unit cell, with antiparallel spins on its two Co."""

    x = COF2_X
    return Structure(
        label="CoF2",
        lattice_vectors=[[COF2_A, 0.0, 0.0], [0.0, COF2_A, 0.0], [0.0, 0.0, COF2_C]],
        species=["Co", "Co", "F", "F", "F", "F"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [x, x, 0.0],
            [1.0 - x, 1.0 - x, 0.0],
            [0.5 + x, 0.5 - x, 0.5],
            [0.5 - x, 0.5 + x, 0.5],
        ],
        properties={"atomic_spins": [CO_SPIN, -CO_SPIN, 0.0, 0.0, 0.0, 0.0]},
    )


def cof2_magnetic_model() -> MagneticStructureModel:
    """The collinear antiferromagnet: 2.64 μB along c, corner Co up, body-centre Co down."""

    return MagneticStructureModel(
        components=(
            build_collinear_component(
                label="Co corner",
                selector=AtomSelector.indices([0]),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, CO_MOMENT_MU_B),
                component_id="co-corner",
            ),
            build_collinear_component(
                label="Co body centre",
                selector=AtomSelector.indices([1]),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, -CO_MOMENT_MU_B),
                component_id="co-body-centre",
            ),
        ),
    )


def cof2_single_component_model() -> MagneticStructureModel:
    """The same antiferromagnet as one component on both Co, the body-centre Co half a turn on."""

    component = build_collinear_component(
        label="Co antiferromagnet",
        selector=AtomSelector.species("Co"),
        q_coeff=(0.0, 0.0, 0.0),
        moment=(0.0, 0.0, CO_MOMENT_MU_B),
        component_id="co-antiferromagnet",
    )
    return MagneticStructureModel(components=(replace(component, atom_phase_turns={1: 0.5}),))


# --- The fixture project ----------------------------------------------------------

CUTOFFS_EV = (300.0, 400.0, 500.0, 600.0, 700.0, 800.0)
KPOINT_MIN_SPACING = 0.05  # 1/Å, the Generate From Spacing... value
RANDOM_STARTS = 6
RANDOM_SEED = "7"


def _configure_sweep(
    service: ProjectService,
    document: ProjectDocument,
    *,
    stage: WorkflowStage,
    form_state: dict[str, object],
) -> None:
    """Configure a convergence sweep as its page's configure action does.

    The page widget turns ``form_state`` into a configuration with the
    catalog's default parameters and the project functional, and the
    definition is registered for the unit cell, leaving every row ready to run.
    """

    from mufinder.app.convergence_widget import CutoffConvergenceWidget
    from mufinder.app.kpoint_convergence_widget import KPointConvergenceWidget
    from mufinder.app.workflow_config_codecs import serialize_convergence_config
    from mufinder.app.workflow_controllers.convergence_cutoff import ConvergenceCutoffController
    from mufinder.app.workflow_controllers.convergence_kpoint import ConvergenceKpointController

    cutoff = stage == WorkflowStage.CONVERGENCE_CUTOFF
    widget = CutoffConvergenceWidget() if cutoff else KPointConvergenceWidget()
    controller = ConvergenceCutoffController if cutoff else ConvergenceKpointController
    try:
        widget.set_xc_functional_default(document.project.recommended_xc_functional)
        widget.set_form_state(form_state)
        config = widget.build_config()
        if config is None:
            raise RuntimeError(f"The {stage.value} sweep has no valid configuration")
        resolution = service.resolve_workflow_structure(document, stage, normalize_state=True)
        register = (
            service.register_convergence_cutoff_definition
            if cutoff
            else service.register_convergence_kpoint_definition
        )
        definition = register(
            document,
            input_structure_id=resolution.structure.id,
            parameters=serialize_convergence_config(config),
            output_directory=document.project_root / "files" / stage.value,
            simulation_cell_mode=resolution.normalized_selection.mode,
        )
        document.ui_state[controller.form_state_key] = widget.collect_state()
        document.ui_state[controller.active_key] = str(definition.id)
    finally:
        widget.deleteLater()


def _build_fixture(directory: Path) -> tuple[ProjectService, ProjectDocument]:
    """The fixture project in memory; :func:`fixture_project` saves it."""

    from mufinder.science.kpoint_grids import generate_kpoint_grids

    host = cof2_host()
    service, document = new_project(directory, "CoF2", host)
    service.set_recommended_xc_functional(document, XC_FUNCTIONAL)

    service.create_workflow_supercell_asset(
        document,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        host_structure_id=host.id,
        matrix=SUPERCELL_2X2X2,
        label=SUPERCELL_LABEL,
    )
    document.project.supercell_matrix = [list(row) for row in SUPERCELL_2X2X2]

    _configure_sweep(
        service,
        document,
        stage=WorkflowStage.CONVERGENCE_CUTOFF,
        form_state={"cutoff_energies": list(CUTOFFS_EV)},
    )
    grids = generate_kpoint_grids(host.lattice_vectors, KPOINT_MIN_SPACING)
    _configure_sweep(
        service,
        document,
        stage=WorkflowStage.CONVERGENCE_KPOINT,
        form_state={
            "kpoint_grids": [list(grid) for grid in grids],
            "generated_for_cell_id": str(host.id),
            "generated_for_lattice": host.lattice_vectors,
        },
    )

    # One muon 1 Å from each fluorine, the F-mu starts, then random starts by
    # the MuFinder paper's rules: at least 1 Å from every atom, a fixed seed.
    service.generate_candidate_sites(
        document,
        method="near_element",
        parameters={
            "element": "F",
            "muons_per_atom": 1,
            "bond_distance": 1.0,
            "min_muon_atom": 0.8,
        },
    )
    service.generate_candidate_sites(
        document,
        method="random",
        parameters={
            "selection_mode": "count",
            "count": RANDOM_STARTS,
            "min_muon_atom": 1.0,
            "max_attempts": 50000,
            "seed": RANDOM_SEED,
            "include_symmetry_equivalents": False,
        },
    )
    return service, document


def fixture_project(directory: Path) -> Path:
    """The CoF2 project most setup scenes open; see the module docstring."""

    service, document = _build_fixture(directory)
    service.save_project(document)
    return document.project_root


# The Convergence Testing overview pins the examiner plot of each adopted
# default. Pinning one needs a finished sweep, so the overview adopts a cutoff
# and a grid with illustrative sweep data: smooth decays towards the converged
# values, not CASTEP results. The examiner recommends exactly the adopted
# values for this data at the default thresholds: 600 eV (the first cutoff
# within 1 meV/atom and 0.05 eV/Ang of 800 eV) and 4 x 4 x 6.

ADOPTED_CUTOFF_EV = 600.0
ADOPTED_GRID = (4, 4, 6)
ENERGY_THRESHOLD = 0.001  # eV per atom
FORCE_THRESHOLD = 0.05  # eV/Å
_CUTOFF_SWEEP = {  # cutoff: (energy above the 800 eV value, max force)
    300.0: (1.84, 0.62),
    400.0: (0.37, 0.21),
    500.0: (0.061, 0.083),
    600.0: (0.0049, 0.034),
    700.0: (0.0007, 0.013),
    800.0: (0.0, 0.011),
}
_KPOINT_SWEEP = {  # grid: (energy above the densest grid's value, max force)
    (2, 2, 3): (0.214, 0.046),
    (3, 3, 4): (0.0112, 0.012),
    (4, 4, 6): (0.0006, 0.004),
    (5, 5, 7): (0.0, 0.003),
}
PRISTINE_ENERGY_EV = -5621.8743  # unit cell, two formula units
_CONVERGED_ENERGY_EV = PRISTINE_ENERGY_EV


def _converged_fixture(directory: Path) -> tuple[ProjectService, ProjectDocument]:
    """The fixture project with a cutoff and a k-point grid adopted and pinned."""

    from mufinder.science.kpoint_grids import format_kpoint_grid

    service, document = _build_fixture(directory)
    host = service.active_host_structure(document)
    host_id = str(host.id)
    criteria = {
        "method": CONVERGED_DEFAULT_METHOD,
        "energy_threshold": ENERGY_THRESHOLD,
        "force_threshold": FORCE_THRESHOLD,
        "cell_id": host_id,
        "xc_functional": XC_FUNCTIONAL,
    }
    service.set_recommended_cutoff(document, ADOPTED_CUTOFF_EV, **criteria)
    service.set_recommended_kpoint_grid(
        document, ADOPTED_GRID, lattice_vectors=host.lattice_vectors, **criteria
    )

    def snapshot(
        default_value: object, sweep: dict[Any, tuple[float, float]], x: Callable[[Any], float]
    ) -> dict:
        return {
            "host_structure_id": host_id,
            "simulation_cell_structure_id": host_id,
            "default_value": default_value,
            "energy_threshold": ENERGY_THRESHOLD,
            "force_threshold": FORCE_THRESHOLD,
            "plot_data": {
                "cutoffs": [x(key) for key in sweep],
                "energies": [_CONVERGED_ENERGY_EV + energy for energy, _force in sweep.values()],
                "forces": [force for _energy, force in sweep.values()],
                "energy_threshold": ENERGY_THRESHOLD,
                "force_threshold": FORCE_THRESHOLD,
            },
        }

    kpoint = snapshot(list(ADOPTED_GRID), _KPOINT_SWEEP, lambda grid: float(np.prod(grid)))
    kpoint["point_labels"] = {
        str(float(np.prod(grid))): format_kpoint_grid(grid) for grid in _KPOINT_SWEEP
    }
    document.ui_state["convergence_overview_snapshots"] = {
        "cutoff": snapshot(ADOPTED_CUTOFF_EV, _CUTOFF_SWEEP, float),
        "kpoint": kpoint,
    }
    return service, document


def converged_project(directory: Path) -> Path:
    """The fixture project with a cutoff and a k-point grid adopted and pinned."""

    service, document = _converged_fixture(directory)
    service.save_project(document)
    return document.project_root


def sweep_job_labels(document: ProjectDocument, stage: WorkflowStage) -> list[str]:
    """The job labels a submission of the ``stage`` sweep's rows would create."""

    from mufinder.app.convergence_adapter import expand_convergence_jobs
    from mufinder.app.workflow_config_codecs import deserialize_convergence_config
    from mufinder.workflow.bridge import structure_to_spec

    definition = next(
        item for item in document.calculations.values() if item.workflow_stage == stage
    )
    config = deserialize_convergence_config(definition.parameters["configuration"])
    structure = document.structures[definition.input_structure_ids[0]]
    expansion = expand_convergence_jobs(structure=structure_to_spec(structure), config=config)
    return [job.name for job in expansion.jobs]


def dashboard_project(directory: Path) -> Path:
    """The fixture project after a first cutoff run that needs attention.

    Three cutoff rows ran and ended failed, stopped and cancelled, the states
    that keep a job on the Job Dashboard.
    """

    from datetime import timedelta

    from mufinder.domain.enums import FailureReason, JobStatus
    from mufinder.domain.models import utc_now

    service, document = _build_fixture(directory)
    definition = next(
        definition
        for definition in document.calculations.values()
        if definition.workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF
    )
    outcomes = (  # cutoff, status, cause, message, minutes run
        (
            300.0,
            JobStatus.FAILED,
            FailureReason.CASTEP_ERROR,
            "SCF did not converge in 100 cycles",
            9,
        ),
        (400.0, JobStatus.STOPPED, FailureReason.WALLTIME, "Stopped at the walltime limit", 60),
        (500.0, JobStatus.CANCELLED, None, None, 4),
    )
    labels = dict(
        zip(CUTOFFS_EV, sweep_job_labels(document, WorkflowStage.CONVERGENCE_CUTOFF), strict=True)
    )
    mapping = definition.parameters.setdefault("job_ids_by_cutoff", {})
    finished = utc_now()
    for cutoff, status, reason, message, minutes in outcomes:
        job = service.add_job(
            document,
            label=labels[cutoff],
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        job.status = status
        job.failure_reason = reason
        job.error_message = message
        job.num_processes = 4
        job.submitted_at = job.started_at = finished - timedelta(minutes=minutes)
        job.completed_at = finished
        mapping[f"{cutoff:.12g}"] = str(job.id)
    service.save_project(document)
    return document.project_root


# --- The results project ------------------------------------------------------------

# The pristine relaxation: a variable-cell BFGS run that ends on the host cell
# with sub-percent changes, as a PBE relaxation of CoF2 might.
PRISTINE_LATTICE = (4.7091, 4.7091, 3.1878)
PRISTINE_X = 0.3047

# The model density of the UEP source job: per species, (electrons, width in Å)
# of each Gaussian shell. Co2+ keeps 15 of its 17 valence electrons (3s2 3p6
# compact, 3d7 wider) and F- holds 8 (2s2 compact, 2p6 diffuse), so the cell is
# neutral against the pseudopotentials' ionic charges, Co 17 and F 7.
UEP_SHELLS = {"Co": ((8.0, 0.22), (7.0, 0.42)), "F": ((2.0, 0.25), (6.0, 0.60))}
UEP_PSEUDOPOTENTIALS = {"Co": (17.0, 1.8), "F": (7.0, 1.4)}  # ionic charge, core radius in bohr
UEP_GRID = (32, 32, 24)
UEP_ROW = "source_01"  # the UEP page's first row label

# Positive-muon clusters (Huddart et al. 2022): each site, its energy above the
# 4c site, and the radial displacements of its neighbours. A displacement
# entry is (species, rank by distance from the muon, change in Å): negative
# pulls the ion towards the muon. At 4c the two nearest F move in by 0.50 and
# 0.57 Å, forming the F-mu-F complex, and the two nearest Co move out by 0.14
# and 0.17 Å. T_A and T_F are De Renzi's rejected candidates.
POSITIVE_SITES: dict[
    str, tuple[tuple[float, float, float], float, tuple[tuple[str, int, float], ...]]
] = {
    "4c": (
        (0.0, 0.5, 0.0),
        0.0,
        (("F", 0, -0.50), ("F", 1, -0.57), ("Co", 0, 0.14), ("Co", 1, 0.17)),
    ),
    "T_A": (
        (0.5, 0.0, 0.25),
        0.34,
        (("F", 0, -0.12), ("F", 1, -0.12), ("Co", 0, 0.06), ("Co", 1, 0.06)),
    ),
    "T_F": ((0.65, 0.84, 0.0), 0.82, (("F", 0, 0.25), ("Co", 0, 0.05), ("Co", 1, 0.04))),
}
# A positive-muon start within this distance (Å) of a T_F or T_A site relaxes
# there; every other start falls into the wide 4c F-mu-F basin.
POSITIVE_BASIN_RADIUS = {"T_F": 0.45, "T_A": 0.60}
POSITIVE_ENERGY_EV = -44985.7412  # the relaxed 4c supercell

# Muonium clusters (Möller et al. 2013): the interstitial site and T_A, a few
# tenths of an eV higher, with small outward relaxations of every ion within
# MUONIUM_NEIGHBOUR_RADIUS. Each start relaxes to the nearer of the two.
MUONIUM_SITES: dict[str, tuple[tuple[float, float, float], float, float]] = {
    "Mu": ((0.56, 0.84, 0.50), 0.0, 0.04),
    "T_A": ((0.5, 0.0, 0.25), 0.27, 0.03),
}
MUONIUM_NEIGHBOUR_RADIUS = 2.4
MUONIUM_ENERGY_EV = -44989.4537

# The F-mu-F phonon of the 4c site (Möller et al. 2013): the symmetric stretch,
# the two bends and the antisymmetric stretch, below a host band.
FMUF_MODES_CM = {"stretch": 585.0, "bends": (3076.0, 3473.0), "antisymmetric": 4570.0}
HOST_BAND_TOP_CM = 600.0


@cache
def _host_orbit(site: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
    """The positions symmetry-equivalent to ``site`` in the host cell."""

    from mufinder.science.spglib_compat import spglib_cell, symmetry_operations

    host = cof2_host()
    rotations, translations = symmetry_operations(spglib_cell(host), symprec=1e-5)
    orbit: list[np.ndarray] = []
    for rotation, translation in zip(rotations, translations, strict=True):
        image = (np.asarray(rotation) @ np.asarray(site) + translation) % 1.0
        if not any(
            np.allclose((image - other + 0.5) % 1.0 - 0.5, 0.0, atol=1e-6) for other in orbit
        ):
            orbit.append(image)
    return tuple(tuple(float(value) for value in image) for image in orbit)


def _nearest_image(start: Any, site: tuple[float, float, float]) -> tuple[np.ndarray, float]:
    """The member of ``site``'s orbit nearest ``start`` (host fractional), and its distance in Å."""

    lattice = np.asarray(cof2_host().lattice_vectors, dtype=float)
    best: tuple[np.ndarray, float] | None = None
    for image in _host_orbit(site):
        shift = (np.asarray(image) - np.asarray(start) + 0.5) % 1.0 - 0.5
        distance = float(np.linalg.norm(shift @ lattice))
        if best is None or distance < best[1]:
            best = (np.asarray(start) + shift, distance)
    assert best is not None
    return best


def positive_site_kind(start: Any) -> str:
    """The positive-muon cluster a start relaxes to; see ``POSITIVE_BASIN_RADIUS``."""

    for kind, radius in POSITIVE_BASIN_RADIUS.items():
        if _nearest_image(start, POSITIVE_SITES[kind][0])[1] < radius:
            return kind
    return "4c"


def muonium_site_kind(start: Any) -> str:
    """The muonium cluster a start relaxes to: the nearer of ``MUONIUM_SITES``."""

    return min(MUONIUM_SITES, key=lambda kind: _nearest_image(start, MUONIUM_SITES[kind][0])[1])


def _job_cell(job_dir: Path, seed: str):
    from mufinder.castep_io.cell_parser import parse_cell_file

    return parse_cell_file(job_dir / f"{seed}.cell")


def _relaxed_muon_cell(
    cell: Any,
    start: Any,
    site: np.ndarray,
    displacements: Callable[[list[str], np.ndarray], np.ndarray],
) -> str:
    """The input supercell with its muon moved to ``site`` and its neighbours displaced.

    ``start`` and ``site`` are host fractional coordinates; ``displacements``
    maps the species and the muon-to-ion vectors (Å) to each ion's radial
    change in Å.
    """

    lattice = np.asarray(cell.lattice_vectors, dtype=float)
    fractional = np.asarray(cell.fractional_positions, dtype=float)
    species = list(cell.species)
    muon = species.index("H:mu")
    repeats = np.rint(
        np.linalg.norm(lattice, axis=1) / np.linalg.norm(cof2_host().lattice_vectors, axis=1)
    )
    fractional[muon] = fractional[muon] + (np.asarray(site) - np.asarray(start)) / repeats
    separation = (fractional - fractional[muon] + 0.5) % 1.0 - 0.5
    vectors = separation @ lattice
    changes = displacements(species, vectors)
    changes[muon] = 0.0
    distances = np.linalg.norm(vectors, axis=1)
    distances[muon] = 1.0
    moved = vectors * (1.0 + changes / distances)[:, None]
    fractional = fractional[muon] + np.linalg.solve(lattice.T, moved.T).T
    return cell_text(lattice.tolist(), species, (fractional % 1.0).tolist())


def _ranked_displacements(
    entries: tuple[tuple[str, int, float], ...],
) -> Callable[[list[str], np.ndarray], np.ndarray]:
    """Radial changes for the ``rank``-th nearest ion of each species, ties broken by atom index."""

    def changes(species: list[str], vectors: np.ndarray) -> np.ndarray:
        distances = np.round(np.linalg.norm(vectors, axis=1), 6)
        result = np.zeros(len(species))
        for symbol, rank, change in entries:
            candidates = sorted(
                (distance, index)
                for index, (other, distance) in enumerate(zip(species, distances, strict=True))
                if other == symbol and distance > 0.0
            )
            result[candidates[rank][1]] = change
        return result

    return changes


def _outward_displacements(
    change: float, radius: float
) -> Callable[[list[str], np.ndarray], np.ndarray]:
    """The same outward change for every ion within ``radius`` Å of the muon."""

    def changes(species: list[str], vectors: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(vectors, axis=1)
        return np.where((distances > 0.0) & (distances < radius), change, 0.0)

    return changes


def _rank_in_cluster(
    window: MainWindow, definition: Any, row: Any, kind_of: Callable[[Any], str]
) -> int:
    """How many of the definition's earlier rows of this charge state relax to the same cluster.

    Each cluster's first row, in the Relaxations table's order, takes the
    cluster's energy exactly; later members sit a little above it.
    """

    from mufinder.app.workflow_config_codecs import deserialize_muon_relaxation_config

    rows = [
        other
        for other in deserialize_muon_relaxation_config(definition.parameters["configuration"]).rows
        if other.charge_state == row.charge_state
    ]
    rows.sort(key=lambda other: natural_sort_key(other.label))
    kind = kind_of(row.initial_fractional_position)
    earlier = rows[: next(index for index, other in enumerate(rows) if other.label == row.label)]
    return sum(1 for other in earlier if kind_of(other.initial_fractional_position) == kind)


def _muon_outputs(window: MainWindow) -> Outputs:
    """A relaxed cell and a CASTEP output for each staged start, by charge state and basin."""

    def write(job: Any, job_dir: Path, seed: str) -> bool:
        definition, row = window._muon_relaxation_definition_and_row_for_job(job)
        start = row.initial_fractional_position
        if row.charge_state == MUONIUM_CHARGE_STATE:
            kind = muonium_site_kind(start)
            site, energy_above, change = MUONIUM_SITES[kind]
            displacements = _outward_displacements(change, MUONIUM_NEIGHBOUR_RADIUS)
            rank = _rank_in_cluster(window, definition, row, muonium_site_kind)
            energy = MUONIUM_ENERGY_EV + energy_above + 0.0004 * rank
        else:
            kind = positive_site_kind(start)
            site, energy_above, entries = POSITIVE_SITES[kind]
            displacements = _ranked_displacements(entries)
            rank = _rank_in_cluster(window, definition, row, positive_site_kind)
            energy = POSITIVE_ENERGY_EV + energy_above + 0.0006 * rank
        number = int("".join(character for character in row.label if character.isdigit()) or 0)
        relaxed, _distance = _nearest_image(start, site)
        cell = _job_cell(job_dir, seed)
        (job_dir / f"{seed}-out.cell").write_text(
            _relaxed_muon_cell(cell, start, relaxed, displacements), encoding="utf-8"
        )
        text = castep_text(
            energy=energy,
            species=list(cell.species),
            geometry=True,
            wall_time=21843.2 + 977.0 * number,
        )
        (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
        return True

    return write


def _pristine_outputs(*, running: bool) -> Outputs:
    """A BFGS log of the host cell ending on ``PRISTINE_LATTICE``; cut short while running."""

    def write(_job: Any, job_dir: Path, seed: str) -> bool:
        x = PRISTINE_X
        a, b, c = PRISTINE_LATTICE
        lattice = [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
        species = cof2_host().species
        fractional = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [x, x, 0.0],
            [1 - x, 1 - x, 0.0],
            [0.5 + x, 0.5 - x, 0.5],
            [0.5 - x, 0.5 + x, 0.5],
        ]
        text = bfgs_log_text(
            lattice=lattice,
            species=species,
            fractional=fractional,
            iterations=3 if running else 7,
            initial_enthalpy=PRISTINE_ENERGY_EV + 0.2214,
            final_enthalpy=PRISTINE_ENERGY_EV if not running else PRISTINE_ENERGY_EV + 0.0391,
            initial_force=0.8214,
            final_force=0.0123 if not running else 0.1342,
            finished=not running,
            wall_time=4127.6,
        )
        (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
        if not running:
            (job_dir / f"{seed}-out.cell").write_text(
                cell_text(lattice, species, fractional), encoding="utf-8"
            )
        return not running

    return write


def _uep_outputs(_job: Any, job_dir: Path, seed: str) -> bool:
    """The model density of the host cell and the single-point output the UEP analysis reads."""

    host = cof2_host()
    arguments = {
        "lattice": host.lattice_vectors,
        "species": host.species,
        "fractional": host.fractional_positions,
    }
    (job_dir / f"{seed}.den_fmt").write_text(
        den_fmt_text(**arguments, grid=UEP_GRID, shells=UEP_SHELLS), encoding="utf-8"
    )
    text = uep_castep_text(
        **arguments,
        pseudopotentials=UEP_PSEUDOPOTENTIALS,
        energy=PRISTINE_ENERGY_EV + 0.0021,
        wall_time=1843.2,
    )
    (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
    return True


def _phonon_outputs(job: Any, job_dir: Path, seed: str) -> bool:
    """The phonon run of the 4c site: an F-mu-F complex in the relaxed supercell."""

    cell = _job_cell(job_dir, seed)
    phonon = triatomic_phonon_text(
        cell,
        stretch_cm=FMUF_MODES_CM["stretch"],
        bend_cm=FMUF_MODES_CM["bends"],
        antisymmetric_cm=FMUF_MODES_CM["antisymmetric"],
        host_top_cm=HOST_BAND_TOP_CM,
    )
    (job_dir / f"{seed}.phonon").write_text(phonon, encoding="utf-8")
    text = castep_text(
        energy=POSITIVE_ENERGY_EV - 0.0004,
        species=list(cell.species),
        geometry=False,
        wall_time=86126.4,
    )
    (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
    return True


def _spin_polarise(view: Any, table: Any = None) -> None:
    """Tick ``Spin Polarized`` for every row (of ``table``) in ``Configure Selected Parameters...``.

    CoF2 is an antiferromagnet, so its calculations are spin-polarised; the
    rows keep their other inherited parameters.
    """

    row_ids = tuple(
        entry.row_id for entry in view._row_entries() if table is None or entry.table is table
    )
    when_modal(
        "ParameterDialog",
        lambda dialog: (dialog.set_values({"spin_polarized": True}), dialog._accept_if_valid()),
    )
    view._open_parameter_dialog(row_ids)


def _choose_relaxed_4c_site(dialog: QWidget) -> None:
    """In ``Select Muon Site``, take the pinned positive-muon cluster 1 and correct for its distortion."""

    dialog._site_table.selectRow(_positive_cluster_row(dialog))
    dialog._distortion_checkbox.setChecked(True)
    dialog.accept()


def _positive_cluster_row(dialog: QWidget) -> int:
    """The row of the lowest-energy pinned positive-muon site, which the dialog lists first."""

    from mufinder.app import magnetic_site_dialog

    for row in range(dialog._site_table.rowCount()):
        option = dialog._site_table.item(row, 0).data(magnetic_site_dialog._SITE_OPTION_ROLE)
        if (
            option.distortion_structure_id is not None
            and "muonium" not in (option.source_label or "").lower()
        ):
            return row
    raise RuntimeError("No pinned positive-muon site with a relaxed structure")


def _build_results(directory: Path, *, running_pristine: bool = False) -> Path:
    """Build and save the results project; see the module docstring."""

    app = QApplication.instance()
    service, document = _converged_fixture(directory)
    document.project.magnetic_structure = cof2_magnetic_model().to_json()
    document.ui_state["magnetic_field_form"] = {"demag_shape": "none"}
    service.save_project(document)

    # The muon relaxations run in the 2x2x2 supercell, so configuring them
    # rescales the adopted unit-cell k-point grid and says so.
    with (
        opened(app, document.project_root) as window,
        refuse_message_boxes(app, expected=("K-point grids adjusted",)),
    ):
        document = window._document
        # Pristine relaxation: one variable-cell row, run as Run All Locally runs it.
        pristine = window._pristine_relaxation_view
        window._set_active_route("pristine_relaxation")
        pristine._add_row()
        _spin_polarise(pristine)
        with synthetic_runs(window, _pristine_outputs(running=running_pristine)):
            pristine._run_local_selected_rows()
        if running_pristine:
            window._save_project()
            return document.project_root
        # A comparison table of the relaxed cell against the host.
        when_modal("ComparisonJobDialog", include_every_job)
        pristine.generate_comparison_requested.emit()

        # UEP: one source row on the host, then a batch search and its clustering.
        window._set_active_route("uep")
        window._uep_view._add_row()
        _spin_polarise(window._uep_view)
        with synthetic_runs(window, _uep_outputs):
            window._uep_view._run_local_selected_rows()
        run_uep_analysis(window, UEP_ROW)

        # Every start, in the Generation table's order, staged in both charge
        # states as relaxations in the 2x2x2 supercell.
        relaxations = window._muon_relaxation_view
        when_modal(
            "MuonChargeStateDialog", lambda dialog: (dialog._both.setChecked(True), dialog.accept())
        )
        open_charge_state_dialog(window)
        for entry in list(relaxations._row_entries()):
            relaxations.set_row_supercell(entry.row_id, [list(row) for row in SUPERCELL_2X2X2])
        # Muonium rows are spin-polarised by default; the positive muon's are ticked here.
        _spin_polarise(relaxations, relaxations._table_for_charge_state(POSITIVE_MUON_CHARGE_STATE))
        for charge_state in (POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE):
            relaxations._charge_tabs.setCurrentWidget(
                relaxations._table_for_charge_state(charge_state)
            )
            relaxations._active_table().clearSelection()
            with synthetic_runs(window, _muon_outputs(window)):
                relaxations._run_local_selected_rows()

        # An analysis workspace per charge state, clustered and pinned.
        cluster_and_pin(window, POSITIVE_MUON_CHARGE_STATE, "Positive muon sites")
        cluster_and_pin(window, MUONIUM_CHARGE_STATE, "Muonium sites")

        # Quantum corrections: the phonon job of the positive-muon 4c site.
        from mufinder.services.source_candidates import charge_state_for_job

        window._set_active_route("quantum_corrections")
        quantum = window._quantum_corrections_controller
        positive = [
            candidate
            for candidate in quantum.candidates()
            if charge_state_for_job(candidate.job) == POSITIVE_MUON_CHARGE_STATE
        ]
        lowest = min(positive, key=lambda candidate: candidate.total_energy_eV)
        with synthetic_runs(window, _phonon_outputs):
            quantum.submit([lowest.key], target="local")

        # Magnetic Fields: the relaxed 4c site from the summary, distortion-corrected.
        window._set_active_route("magnetic_fields")
        when_modal("MuonSiteSelectionDialog", _choose_relaxed_4c_site)
        window._magnetic_field_view._select_site_button.click()

        # Energy barriers: the paths from the 4c site to its neighbours.
        window._set_active_route("energy_barriers")
        barriers = window._energy_barriers_controller
        barriers.select_endpoints(lowest.key, lowest.key)
        barriers.generate_paths(lowest.key, lowest.key)
        window._save_project()
        return document.project_root


def results_project(directory: Path) -> Path:
    """The CoF2 project with every results workflow finished; see the module docstring."""

    return _build_results(directory)


def running_relaxation_project(directory: Path) -> Path:
    """The CoF2 project with its pristine relaxation part-way through."""

    return _build_results(directory, running_pristine=True)


# --- The legacy folder ----------------------------------------------------------

_LEGACY_RUNS = (  # start in host fractional coordinates, site, converged
    ((0.03, 0.47, 0.06), "4c", True),
    ((0.47, 0.04, 0.29), "T_A", True),
    ((0.61, 0.80, 0.07), "T_F", False),
)


def _legacy_folder(root: Path) -> Path:
    """A legacy working folder of three CoF2 positive-muon relaxations, in the documented layout.

    Hand-authored, with the positive-muon cluster energies: see
    docs/legacy_import.rst, "Research data boundary". Site 3 stopped before
    its geometry converged.
    """

    from mufinder.science.supercell import build_supercell_structure

    supercell, _spins = build_supercell_structure(cof2_host(), SUPERCELL_2X2X2)
    for index, (start, kind, converged) in enumerate(_LEGACY_RUNS, start=1):
        site, energy_above, _entries = POSITIVE_SITES[kind]
        relaxed, _distance = _nearest_image(start, site)
        species = [*supercell.species, "H:mu"]
        initial = [*supercell.fractional_positions, [value / 2.0 for value in start]]
        final = [*supercell.fractional_positions, [float(value) % 1.0 / 2.0 for value in relaxed]]
        stem = f"CoF2_{index}"
        for folder in ("input", "results", f"run/site{index}"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        (root / "input" / f"{stem}.cell").write_text(
            cell_text(supercell.lattice_vectors, species, initial) + "\nFIX_ALL_CELL : TRUE\n",
            encoding="utf-8",
        )
        (root / "results" / f"{stem}-out.cell").write_text(
            cell_text(supercell.lattice_vectors, species, final), encoding="utf-8"
        )
        energy = POSITIVE_ENERGY_EV + energy_above + 0.0011 * index
        (root / "results" / f"{stem}.castep").write_text(
            legacy_castep_text(
                energy=energy,
                converged=converged,
                cut_off_ev=ADOPTED_CUTOFF_EV,
                spin_polarized=True,
            ),
            encoding="utf-8",
        )
        (root / f"run/site{index}" / f"{stem}.param").write_text(
            "task : geometryoptimisation\ncharge : +1\nxc_functional : PBE\ncut_off_energy : 600 eV\n"
            "spin_polarized : true\nwrite_cell_structure : true\ngeom_max_iter : 300\n",
            encoding="utf-8",
        )
    x = COF2_X
    (root / "CoF2.cif").write_text(
        f"data_CoF2\n_symmetry_space_group_name_H-M 'P 42/m n m'\n_cell_length_a {COF2_A:.6f}\n"
        f"_cell_length_b {COF2_A:.6f}\n_cell_length_c {COF2_C:.6f}\n_cell_angle_alpha 90\n_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\nloop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        f"Co 0 0 0\nF {x:.4f} {x:.4f} 0\n",
        encoding="utf-8",
    )
    return root


# --- Scene steps --------------------------------------------------------------


def open_supercell_editor(window: MainWindow) -> None:
    """Click ``Open Supercell Editor`` on the Supercell Editor page."""

    window._on_host_supercell_editor_requested()


def show_saved_supercell(dialog: QWidget) -> None:
    """Select the saved supercell, which loads its matrix and preview."""

    dialog._asset_table.selectRow(0)
    settle(QApplication.instance())
    frame_viewers(dialog)


def open_supercell_suggestions(window: MainWindow) -> None:
    """Click ``Suggest…`` in the Supercell Editor."""

    when_modal("SupercellEditorDialog", lambda editor: editor._suggest_button.click())
    open_supercell_editor(window)


def open_cutoff_parameters(window: MainWindow) -> None:
    """Click the ``Configure...`` button of the 500 eV cutoff row."""

    view = window._convergence_view
    view._table.cellWidget(CUTOFFS_EV.index(500.0), view._CONFIGURE_COLUMN).click()


def open_local_submission(window: MainWindow) -> None:
    """Open the dialog ``Run All Locally`` shows for the cutoff sweep.

    The run button first checks for a CASTEP executable, which the scratch
    settings do not have, so the dialog is opened with the sweep's job labels
    directly. It reports the computer's core count; an eight-core computer
    (eight logical and eight physical cores) keeps the image the same on every
    build machine.
    """

    from mufinder.app import submission_dialog
    from mufinder.services.project_summary import job_stage_title

    labels = sweep_job_labels(window._document, WorkflowStage.CONVERGENCE_CUTOFF)
    ideal_thread_count = submission_dialog.QThread.idealThreadCount
    physical_core_count = submission_dialog.physical_core_count
    submission_dialog.QThread.idealThreadCount = staticmethod(lambda: 8)
    submission_dialog.physical_core_count = lambda: 8
    try:
        dialog = submission_dialog.LocalSubmissionDialog(
            job_labels=labels,
            mpirun_configured=True,
            stage_title=job_stage_title([WorkflowStage.CONVERGENCE_CUTOFF]),
            parent=window,
        )
    finally:
        submission_dialog.QThread.idealThreadCount = ideal_thread_count
        submission_dialog.physical_core_count = physical_core_count
    dialog.exec()


def show_saved_supercell_along_a(dialog: QWidget) -> None:
    show_saved_supercell(dialog)
    view_along_a(dialog)


def open_charge_state_dialog(window: MainWindow) -> None:
    """Select every generated site and click ``Create Relaxation Jobs``, which asks for the charge state."""

    sites = sorted(window._document.sites.values(), key=lambda site: natural_sort_key(site.label))
    window._on_create_muon_relaxation_jobs_requested([str(site.id) for site in sites])


def choose_both_charge_states(dialog: QWidget) -> None:
    dialog._both.setChecked(True)
    focus_default_button(dialog)


def show_muonium_tab(page: QWidget) -> None:
    """Show the Muonium tab of the Relaxations page."""

    page._charge_tabs.setCurrentWidget(page._table_for_charge_state(MUONIUM_CHARGE_STATE))


def _workspace_id(window: MainWindow, charge_state: str) -> str:
    index = 0 if charge_state == POSITIVE_MUON_CHARGE_STATE else 1
    return str(window._relaxation_analysis_workspaces()[index]["id"])


def open_analysis_workspace(charge_state: str) -> Callable[[MainWindow], None]:
    """Open the results project's analysis workspace of ``charge_state`` in the navigator."""

    def open_workspace(window: MainWindow) -> None:
        window._set_active_route(
            window._relaxation_analysis_workspace_route(_workspace_id(window, charge_state))
        )

    return open_workspace


def open_relaxation_clustering(window: MainWindow) -> None:
    """Click ``Clustering...`` in the positive-muon workspace."""

    workspace_id = _workspace_id(window, POSITIVE_MUON_CHARGE_STATE)
    window._set_active_route(window._relaxation_analysis_workspace_route(workspace_id))
    window._relaxation_analysis_view._open_clustering(workspace_id)


def open_displacement_analysis(window: MainWindow) -> None:
    """Click ``Analyse Displacements`` in the positive-muon workspace."""

    workspace_id = _workspace_id(window, POSITIVE_MUON_CHARGE_STATE)
    window._set_active_route(window._relaxation_analysis_workspace_route(workspace_id))
    window._relaxation_analysis_view._open_displacements(workspace_id)


def open_comparison_table(window: MainWindow) -> None:
    """Open the results project's comparison table in the navigator."""

    table_id = next(iter(window._document.comparison_tables))
    window._set_active_route(window._comparison_route(table_id))


def open_comparison_job_dialog(window: MainWindow) -> None:
    """Click ``Generate Comparison Table`` on the Pristine Relaxation page."""

    window._pristine_relaxation_view.generate_comparison_requested.emit()


def tick_every_job(dialog: QWidget) -> None:
    from PySide6.QtCore import Qt

    for row in range(dialog._table.rowCount()):
        dialog._table.item(row, dialog._CHECK_COLUMN).setCheckState(Qt.CheckState.Checked)
    focus_default_button(dialog)


def show_running_progress(window: MainWindow) -> QWidget:
    """Inspect the running pristine relaxation, as its dashboard Inspect button does.

    Reopening a project ends its local jobs, since no local runner survives
    the application, so the scene puts the job back in the state the local
    runner keeps it in while CASTEP writes its ``.castep``.
    """

    from mufinder.domain.enums import JobStatus

    job = next(
        job
        for job in window._document.jobs.values()
        if job.workflow_stage == WorkflowStage.PRISTINE_RELAXATION
    )
    job.status = JobStatus.RUNNING
    castep = window._job_output_directory(job) / f"{window._job_seed_name(job)}.castep"
    monitor = window._job_progress
    monitor._local_output_paths = lambda job_id: (castep,) if job_id == job.id else ()
    monitor._run_in_background = lambda task: task()
    window._refresh_jobs_table()
    return window._show_geometry_progress(job)


def _open_uep_analysis(window: MainWindow) -> QWidget:
    window._on_open_uep_analysis_requested(UEP_ROW)
    return next(iter(window._uep_analysis_dialogs.values()))


def open_uep_clustering(window: MainWindow) -> QWidget:
    """Open the UEP analysis of the source job, then its saved batch search's clustering."""

    analysis = _open_uep_analysis(window)
    analysis._open_clustering_dialog()
    analysis.hide()
    return analysis._clustering_dialog


def open_uep_batch_search(window: MainWindow) -> QWidget:
    """Open the UEP analysis window on its batch search, with the saved minima listed."""

    analysis = _open_uep_analysis(window)
    _show_batch_tab(analysis)
    return analysis


def _show_batch_tab(analysis: QWidget) -> None:
    from PySide6.QtWidgets import QTabWidget

    for tabs in analysis.findChildren(QTabWidget):
        for index in range(tabs.count()):
            if "batch" in tabs.tabText(index).lower():
                tabs.setCurrentIndex(index)
                return


def open_phonon_analysis(window: MainWindow) -> QWidget:
    """Click ``Analyse...`` on the completed phonon result."""

    result = next(
        result
        for result in window._document.results.values()
        if isinstance(result.data, dict) and "muon_zpe_total_eV" in result.data
    )
    window._on_launch_quantum_phonon_analysis(str(result.id))
    return window._quantum_analysis_dialogs[result.id]


def calculate_corner_tensor(dialog: QWidget) -> None:
    """Take the corner Co sublattice (atom 1) and click ``Calculate``."""

    from PySide6.QtCore import QEvent

    dialog._indices_radio.setChecked(True)
    dialog._indices_edit.setText("1")
    dialog._calculate_button.click()
    # The previous result's placeholder goes through deleteLater, which an
    # event loop would run before the next paint.
    QApplication.instance().sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def calculate_fluorine_second_moment(dialog: QWidget) -> None:
    """Zero-field polycrystal, corrected for the F-mu-F distortion, then ``Calculate``."""

    dialog._regime_combo.setCurrentIndex(dialog._regime_combo.findData("zf"))
    dialog._sample_combo.setCurrentIndex(dialog._sample_combo.findData("polycrystal"))
    dialog._distortion.setChecked(True)
    dialog._calculate_button.click()


def open_legacy_import(window: MainWindow) -> None:
    """Scan a synthetic legacy folder and open ``Import Legacy Relaxations`` on it.

    ``File > Import Legacy Relaxations...`` scans on a worker thread after a
    folder dialog; the scene runs the same scan in this thread. Its notice that
    no keyword catalog is installed, true only of the scratch settings, is
    dropped as the other scenes hide it.
    """

    from dataclasses import replace

    from mufinder.app.legacy_import_dialog import LegacyImportDialog
    from mufinder.services.legacy_import import scan_legacy_folder

    host = window._project_service.active_host_structure(window._document)
    root = _legacy_folder(scratch_directory("legacy-") / "cof2-muon-sites")
    scan = scan_legacy_folder(
        root,
        host_lattice=[list(row) for row in host.lattice_vectors],
        host_species=list(host.species),
    )
    scan = replace(
        scan,
        warnings=tuple(warning for warning in scan.warnings if "keyword catalog" not in warning),
    )
    LegacyImportDialog(scan, parent=window).exec()


# --- Scenes -------------------------------------------------------------------


def _magnetic(open_: Callable[[Any], object]) -> Callable[[MainWindow], object]:
    return lambda window: open_(window._magnetic_field_view)


def open_atom_phases(window: MainWindow) -> None:
    """Open the builder, edit the one component and click ``Per-Atom Phases...``."""

    def edit(builder: QWidget) -> None:
        when_modal("MagneticComponentDialog", lambda dialog: dialog._atom_phase_button.click())
        builder._component_table.selectRow(0)
        builder._edit_button.click()

    when_modal("MagneticStructureBuilderDialog", edit)
    window._magnetic_field_view._edit_structure_button.click()


SCENES: dict[str, Scene] = {
    # Getting started and setup.
    "project_home": Scene(converged_project, None, Window()),
    "host_structure_page": Scene(
        fixture_project,
        "host_structure",
        Page("_host_structure_view", (1100, 760), prepare=show_host_with_spins),
    ),
    "supercell_editor": Scene(
        fixture_project,
        "supercell_editor",
        Dialog(open_supercell_editor, size=(1180, 760), prepare=show_saved_supercell_along_a),
    ),
    "supercell_suggestions": Scene(
        fixture_project,
        "supercell_editor",
        Dialog(open_supercell_suggestions, kind="SupercellSuggestionDialog"),
    ),
    "convergence_overview": Scene(
        converged_project, "convergence", Page("_convergence_overview_view", (1100, 760))
    ),
    "cutoff_convergence_page": Scene(
        fixture_project, "convergence_cutoff", Page("_convergence_view", (1100, 720))
    ),
    "kpoint_convergence_page": Scene(
        fixture_project, "convergence_kpoint", Page("_kpoint_convergence_view", (1100, 760))
    ),
    "parameter_dialog": Scene(
        fixture_project,
        "convergence_cutoff",
        Dialog(
            open_cutoff_parameters,
            kind="ParameterDialog",
            size=(900, 520),
            prepare=hide_catalog_notice,
        ),
    ),
    "site_generation_page": Scene(
        fixture_project,
        "muon_candidates",
        Page("_muon_generation_view", (1160, 740), prepare=select_all_sites),
    ),
    "charge_state_dialog": Scene(
        fixture_project,
        "muon_candidates",
        Dialog(
            open_charge_state_dialog,
            kind="MuonChargeStateDialog",
            prepare=choose_both_charge_states,
        ),
    ),
    "journal_page": Scene(results_project, "provenance_log", Page("_journal_view", (1100, 800))),
    "castep_setup_dialog": Scene(
        fixture_project, None, Dialog(lambda window: window._open_castep_setup_dialog())
    ),
    "cluster_setup_dialog": Scene(
        fixture_project, None, Dialog(lambda window: window._open_cluster_setup_dialog())
    ),
    "local_submission_dialog": Scene(
        fixture_project,
        "convergence_cutoff",
        Dialog(open_local_submission, prepare=focus_default_button),
    ),
    "job_dashboard": Scene(dashboard_project, "convergence_cutoff", Window(dashboard=True)),
    # Pristine relaxation and the comparison table.
    "pristine_relaxation_page": Scene(
        results_project, "pristine_relaxation", Page("_pristine_relaxation_view", (1100, 540))
    ),
    "geometry_progress_dialog": Scene(
        running_relaxation_project,
        "pristine_relaxation",
        Floating(show_running_progress, size=(820, 640)),
    ),
    "comparison_job_dialog": Scene(
        results_project,
        "pristine_relaxation",
        Dialog(
            open_comparison_job_dialog,
            kind="ComparisonJobDialog",
            size=(900, 380),
            prepare=tick_every_job,
        ),
    ),
    "comparison_table": Scene(
        results_project, None, Page(current_page, (780, 680)), setup=open_comparison_table
    ),
    # UEP.
    "uep_page": Scene(results_project, "uep", Page("_uep_view", (1100, 540))),
    "uep_batch_search": Scene(
        results_project,
        "uep",
        Floating(open_uep_batch_search, size=(1280, 820), prepare=frame_viewers),
    ),
    "uep_clustering_dialog": Scene(
        results_project,
        "uep",
        Floating(open_uep_clustering, size=(1180, 760), prepare=frame_viewers),
    ),
    # Muon relaxations and their analysis.
    "muon_relaxations_page": Scene(
        results_project, "muon_relaxations", Page("_muon_relaxation_view", (1100, 760))
    ),
    "muonium_relaxations_page": Scene(
        results_project,
        "muon_relaxations",
        Page("_muon_relaxation_view", (1100, 760), prepare=show_muonium_tab),
    ),
    "relaxation_analysis_workspace": Scene(
        results_project,
        None,
        Page("_relaxation_analysis_view", (1100, 720), prepare=frame_viewers),
        setup=open_analysis_workspace(POSITIVE_MUON_CHARGE_STATE),
    ),
    "muonium_analysis_workspace": Scene(
        results_project,
        None,
        Page("_relaxation_analysis_view", (1100, 720), prepare=frame_viewers),
        setup=open_analysis_workspace(MUONIUM_CHARGE_STATE),
    ),
    "relaxation_clustering_dialog": Scene(
        results_project,
        None,
        Dialog(
            open_relaxation_clustering,
            kind="RelaxationClusteringDialog",
            size=(1280, 820),
            prepare=frame_viewers,
        ),
    ),
    "displacement_analysis_dialog": Scene(
        results_project,
        None,
        Dialog(
            open_displacement_analysis,
            kind="DisplacementAnalysisDialog",
            size=(1180, 720),
            prepare=frame_viewers,
        ),
    ),
    "muon_sites_summary": Scene(
        results_project,
        "muon_sites",
        Page("_muon_sites_summary_view", (1100, 800), prepare=frame_viewers),
    ),
    # Quantum corrections.
    "quantum_corrections_page": Scene(
        results_project, "quantum_corrections", Page("_quantum_corrections_view", (1100, 680))
    ),
    "phonon_analysis_window": Scene(
        results_project,
        "quantum_corrections",
        Floating(open_phonon_analysis, size=(1180, 780), prepare=frame_viewers),
    ),
    # Magnetic fields.
    "cof2_magnetic_fields_page": Scene(
        results_project,
        "magnetic_fields",
        Page("_magnetic_field_view", (1100, 1140)),
        setup=lambda window: calculate_field(window._magnetic_field_view),
    ),
    "cof2_magnetic_structure_builder": Scene(
        results_project,
        "magnetic_fields",
        Dialog(_magnetic(lambda page: page._edit_structure_button.click()), size=(1180, 900)),
    ),
    "cof2_atom_phases": Scene(
        magnetic_project("CoF2 one component", cof2_host(), cof2_single_component_model(), {}),
        "magnetic_fields",
        Dialog(open_atom_phases, kind="AtomPhaseDialog", size=(620, 400)),
    ),
    "magnetic_muon_site_dialog": Scene(
        results_project,
        "magnetic_fields",
        Dialog(
            _magnetic(lambda page: page._select_site_button.click()),
            kind="MuonSiteSelectionDialog",
            size=(900, 460),
            prepare=focus_default_button,
        ),
    ),
    "dipolar_tensor_dialog": Scene(
        results_project,
        "magnetic_fields",
        Dialog(
            _magnetic(lambda page: page._dipolar_tensor_button.click()),
            kind="DipolarTensorDialog",
            size=(900, 820),
            prepare=calculate_corner_tensor,
        ),
    ),
    "nuclear_relaxation_dialog": Scene(
        results_project,
        "magnetic_fields",
        Dialog(
            _magnetic(lambda page: page._nuclear_relaxation_button.click()),
            kind="NuclearRelaxationDialog",
            size=(760, 980),
            prepare=calculate_fluorine_second_moment,
        ),
    ),
    "cof2_magnetic_field_workspace": Scene(
        results_project,
        "magnetic_fields",
        Page(current_page, (1100, 1140)),
        setup=save_magnetic_workspace("CoF2 4c site"),
    ),
    # Energy barriers.
    "energy_barriers_page": Scene(
        results_project, "energy_barriers", Page("_energy_barriers_view", (1100, 740))
    ),
    # The legacy importer.
    "legacy_import_dialog": Scene(
        fixture_project,
        None,
        Dialog(open_legacy_import, kind="LegacyImportDialog", prepare=focus_default_button),
    ),
}
