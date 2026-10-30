"""Gadolinium ruthenium silicide: the Energy Barriers worked example, from real CASTEP outputs.

GdRu2Si2 is a centrosymmetric tetragonal magnet (I4/mmm, a = 4.1634 Å,
c = 9.6102 Å; Hiebl et al. 1983) with Gd at 2a, Ru at 4d and Si at 4e. Huddart
et al. (2025) found two positive-muon sites in a 2x2x1 supercell with
ferromagnetic Gd spins: site 1 at (0.50, 0.00, 0.07) and site 2 at
(0.21, 0.21, 0.16), 0.26 eV higher. Their NEB band from site 1 to site 2 and
their Gamma-point phonons of both sites show that site 2 is the transition
state of the hop between neighbouring site 1 positions and is not stable once
the muon's zero-point energy is counted. ``docs/user_guide/examples/gdru2si2.rst``
tells the story these scenes illustrate.

Unlike the CoF2 example, the outputs here are the paper's own CASTEP outputs,
kept under ``tests/fixtures/gdru2si2`` (see its ``README.md``): the two relaxed
supercells, the NEB band trimmed to its final iteration, and the two
``.phonon`` files. Only the ``.castep`` logs are synthesised, around the real
final energies.

:func:`generation_project` is the host, the supercell and the random starts
of the paper's generation rules. :func:`results_project` adds the two Table SI
starts as custom positions and drives the window through their relaxations,
the analysis workspace and its pin, the phonon jobs of both sites, and the
Energy Barriers paths and NEB band. :func:`configured_project` stops before
the NEB job is submitted, for the configuration dialog.
"""

from __future__ import annotations

# The render module configures Qt and VTK, so it is imported before them.
from ..render import (
    PROJECT_ROOT,
    XC_FUNCTIONAL,
    Dialog,
    Page,
    Scene,
    cluster_and_pin,
    focus_default_button,
    frame_viewers,
    hide_catalog_notice,
    new_project,
    opened,
    refuse_message_boxes,
    select_all_sites,
    settle,
    synthetic_runs,
    view_along_a,
    when_modal,
)
from ..synthetic import castep_text

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication, QWidget

from mufinder.castep_io.types import ValueWithUnit
from mufinder.domain.charge_states import POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Structure
from mufinder.domain.text import natural_sort_key
from mufinder.services.project_service import ProjectDocument, ProjectService

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "gdru2si2"

# --- The host -------------------------------------------------------------------

GDRU2SI2_A = 4.1634  # Å, Hiebl et al. 1983
GDRU2SI2_C = 9.6102
SI_Z = 0.3682  # Si at 4e (0, 0, z), from the atoms of the relaxed cells far from the muon
GD_SPIN = 7.0  # hbar/2 on every Gd: the ferromagnetic arrangement the calculations used
SUPERCELL_2X2X1 = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
SUPERCELL_LABEL = "GdRu2Si2 2x2x1"


def gdru2si2_host() -> Structure:
    """The conventional I4/mmm cell, ten atoms, with a ferromagnetic spin on each Gd."""

    z = SI_Z
    return Structure(
        label="GdRu2Si2",
        lattice_vectors=[[GDRU2SI2_A, 0.0, 0.0], [0.0, GDRU2SI2_A, 0.0], [0.0, 0.0, GDRU2SI2_C]],
        species=["Gd", "Gd", "Ru", "Ru", "Ru", "Ru", "Si", "Si", "Si", "Si"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 0.25],
            [0.5, 0.0, 0.25],
            [0.0, 0.5, 0.75],
            [0.5, 0.0, 0.75],
            [0.0, 0.0, z],
            [0.0, 0.0, 1.0 - z],
            [0.5, 0.5, 0.5 + z],
            [0.5, 0.5, 0.5 - z],
        ],
        properties={"atomic_spins": [GD_SPIN, GD_SPIN, *[0.0] * 8]},
    )


# --- The calculations -------------------------------------------------------------

CUTOFF_EV = 1000.0
RELAXATION_GRID = (5, 5, 5)  # in the 2x2x1 supercell
FORCE_TOL_EV_ANG = 0.05
# The paper's random generation rules: every start at least 0.5 Å from the
# earlier starts and their symmetry equivalents, and 1.0 Å from every atom.
# The paper's draw stopped at 30 starts; MuFinder fills the cell, which with
# this seed takes 38.
MIN_MUON_MUON_ANG = 0.5
MIN_MUON_ATOM_ANG = 1.0
RANDOM_SEED = "2025"


@dataclass(frozen=True)
class MuonSite:
    """A Table SI site: its start in conventional fractional coordinates and its real outputs."""

    start: tuple[float, float, float]
    energy_eV: float  # the relaxed supercell's final energy
    seed: str  # the fixture files' seed name


SITES = {
    1: MuonSite((0.50, 0.00, 0.07), -69184.849, "GdRu2Si2_site1"),
    2: MuonSite((0.21, 0.21, 0.16), -69184.592, "GdRu2Si2_site2"),
}

# The NEB band from site 1 to site 2: 40 images, 4 task farms, a 0.08 eV/Å
# force tolerance and a 3x3x3 grid, as the paper's run used.
NEB_FIXTURE = "GdRu2Si2_12_neb.ts"
NEB_IMAGES = 40
NEB_FARMS = 4
NEB_FORCE_TOL_EV_ANG = 0.08
NEB_GRID = (3, 3, 3)
PHONON_GRID = (3, 3, 3)
# The automatic cutoff keeps only the 1.375 Å hop of site 1 across the Gd plane;
# 3 Å also takes the 2.944 Å in-plane hop to 1', whose transition state is site 2.
SAME_SITE_MAX_PATH_ANG = 3.0
PHONON_FARMS = 4


def _fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _job_cell(job_dir: Path, seed: str) -> Any:
    from mufinder.castep_io.cell_parser import parse_cell_file

    return parse_cell_file(job_dir / f"{seed}.cell")


def _muon_position(cell: Any) -> np.ndarray:
    return np.asarray(cell.fractional_positions[list(cell.species).index("H:mu")], dtype=float)


def _nearest_site(cell: Any) -> MuonSite:
    """The Table SI site whose relaxed muon is nearest the muon of a job's input cell."""

    from mufinder.castep_io.cell_parser import parse_cell_file

    lattice = np.asarray(cell.lattice_vectors, dtype=float)
    muon = _muon_position(cell)

    def distance(site: MuonSite) -> float:
        other = _muon_position(parse_cell_file(FIXTURES / f"{site.seed}-out.cell"))
        return float(np.linalg.norm(((other - muon + 0.5) % 1.0 - 0.5) @ lattice))

    return min(SITES.values(), key=distance)


def _relaxation_outputs(_job: Any, job_dir: Path, seed: str) -> bool:
    """The paper's relaxed supercell of the job's site, and a log ending on its real energy."""

    cell = _job_cell(job_dir, seed)
    site = _nearest_site(cell)
    (job_dir / f"{seed}-out.cell").write_text(
        _fixture_text(f"{site.seed}-out.cell"), encoding="utf-8"
    )
    text = castep_text(
        energy=site.energy_eV, species=list(cell.species), geometry=True, wall_time=171236.4
    )
    (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
    return True


def _phonon_outputs(_job: Any, job_dir: Path, seed: str) -> bool:
    """The paper's Gamma-point ``.phonon`` file of the job's site."""

    cell = _job_cell(job_dir, seed)
    site = _nearest_site(cell)
    (job_dir / f"{seed}.phonon").write_text(_fixture_text(f"{site.seed}.phonon"), encoding="utf-8")
    text = castep_text(
        energy=site.energy_eV, species=list(cell.species), geometry=False, wall_time=259804.1
    )
    for name in (f"{seed}.castep", f"{seed}_farm001.castep"):
        (job_dir / name).write_text(text, encoding="utf-8")
    return True


def _neb_outputs(_job: Any, job_dir: Path, seed: str) -> bool:
    """The paper's NEB band from site 1 to site 2, trimmed to its final iteration.

    A task-farmed run writes its band to ``<seed>_farm001.ts`` and no
    ``<seed>.ts``.
    """

    (job_dir / f"{seed}_farm001.ts").write_text(_fixture_text(NEB_FIXTURE), encoding="utf-8")
    species = list(_job_cell(job_dir, seed).species)
    text = castep_text(
        energy=SITES[1].energy_eV, species=species, geometry=False, wall_time=412833.9
    )
    for name in (f"{seed}.castep", f"{seed}_farm001.castep"):
        (job_dir / name).write_text(text, encoding="utf-8")
    return True


# --- The projects -------------------------------------------------------------------


def _build_generation(directory: Path) -> tuple[ProjectService, ProjectDocument]:
    """The host, its supercell, the calculation defaults and the random starts."""

    host = gdru2si2_host()
    service, document = new_project(directory, "GdRu2Si2", host)
    service.set_recommended_xc_functional(document, XC_FUNCTIONAL)
    service.create_workflow_supercell_asset(
        document,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        host_structure_id=host.id,
        matrix=SUPERCELL_2X2X1,
        label=SUPERCELL_LABEL,
    )
    document.project.supercell_matrix = [list(row) for row in SUPERCELL_2X2X1]
    # The paper's settings, set by hand: no convergence sweep ran in this project.
    supercell_lattice = (
        np.asarray(SUPERCELL_2X2X1, dtype=float) @ np.asarray(host.lattice_vectors)
    ).tolist()
    service.set_recommended_cutoff(
        document, CUTOFF_EV, method="manual", xc_functional=XC_FUNCTIONAL
    )
    service.set_recommended_kpoint_grid(
        document,
        RELAXATION_GRID,
        lattice_vectors=supercell_lattice,
        method="manual",
        xc_functional=XC_FUNCTIONAL,
    )
    random = {
        "selection_mode": "separation",
        "min_muon_muon": MIN_MUON_MUON_ANG,
        "min_muon_atom": MIN_MUON_ATOM_ANG,
        "max_attempts": 50000,
        "seed": RANDOM_SEED,
    }
    service.generate_candidate_sites(document, method="random", parameters=random)
    document.ui_state["muon_generation_form"] = {"method": "random", "random": random}
    return service, document


def generation_project(directory: Path) -> Path:
    """The GdRu2Si2 project with the random starts generated and nothing run."""

    service, document = _build_generation(directory)
    service.save_project(document)
    return document.project_root


def _configure_relaxations(window: MainWindow) -> None:
    """Spin-polarise every relaxation row and set the paper's force tolerance."""

    view = window._muon_relaxation_view
    row_ids = tuple(entry.row_id for entry in view._row_entries())
    values = {"spin_polarized": True, "geom_force_tol": ValueWithUnit(FORCE_TOL_EV_ANG, "eV/ang")}
    when_modal(
        "ParameterDialog", lambda dialog: (dialog.set_values(values), dialog._accept_if_valid())
    )
    view._open_parameter_dialog(row_ids)


def _configure_neb(window: MainWindow, key: str) -> None:
    """Set the NEB row's images, force tolerance, k-point grid and task farms in its dialog."""

    values = {
        "tssearch_max_path_points": NEB_IMAGES,
        "tssearch_force_tol": ValueWithUnit(NEB_FORCE_TOL_EV_ANG, "eV/ang"),
        "kpoint_mp_grid": NEB_GRID,
    }

    def configure(dialog: Any) -> None:
        dialog.set_values(values)
        dialog._num_farms_spin.setValue(NEB_FARMS)
        dialog._accept_if_valid()

    when_modal("NudgedElasticBandConfigDialog", configure)
    window._on_configure_energy_barrier_result([key])


def _configure_phonons(window: MainWindow, keys: list[str]) -> None:
    """Set the phonon rows' k-point grid and task farms in their dialog."""

    def configure(dialog: Any) -> None:
        dialog.set_values({"kpoint_mp_grid": PHONON_GRID})
        dialog._num_farms_spin.setValue(PHONON_FARMS)
        dialog._accept_if_valid()

    when_modal("QuantumCorrectionsConfigDialog", configure)
    window._on_configure_quantum_corrections_source(keys)


def site_keys(window: MainWindow) -> dict[int, str]:
    """The relaxed-site key of each Table SI site, by its rank in energy."""

    candidates = sorted(
        window._quantum_corrections_controller.candidates(), key=lambda item: item.total_energy_eV
    )
    return {number: candidates[number - 1].key for number in SITES}


def path_key(window: MainWindow, reactant: str, product: str) -> str:
    """The shortest generated path between two sites."""

    controller = window._energy_barriers_controller
    paths = [
        candidate
        for candidate in controller.candidates()
        if candidate.reactant.key == reactant and candidate.product.key == product
    ]
    if not paths:
        raise RuntimeError("No path was generated between the endpoints")
    return min(paths, key=lambda candidate: candidate.path.length_ang).key


def _build_results(directory: Path, *, run_neb: bool) -> Path:
    """Build and save the results project; see the module docstring."""

    app = QApplication.instance()
    service, document = _build_generation(directory)
    # The two Table SI sites, entered as custom coordinates.
    for site in SITES.values():
        x, y, z = site.start
        service.generate_candidate_sites(
            document,
            method="manual",
            parameters={"coord_system": "fractional", "x": x, "y": y, "z": z},
        )
    service.save_project(document)

    with opened(app, document.project_root) as window, refuse_message_boxes(app):
        document = window._document
        manual = sorted(
            (site for site in document.sites.values() if site.generation_method == "manual"),
            key=lambda site: natural_sort_key(site.label),
        )
        # Positive-muon relaxations of the two starts in the 2x2x1 supercell.
        when_modal("MuonChargeStateDialog", lambda dialog: dialog.accept())
        window._on_create_muon_relaxation_jobs_requested([str(site.id) for site in manual])
        relaxations = window._muon_relaxation_view
        for entry in list(relaxations._row_entries()):
            relaxations.set_row_supercell(entry.row_id, [list(row) for row in SUPERCELL_2X2X1])
        _configure_relaxations(window)
        relaxations._active_table().clearSelection()
        with synthetic_runs(window, _relaxation_outputs):
            relaxations._run_local_selected_rows()
        cluster_and_pin(window, POSITIVE_MUON_CHARGE_STATE, "GdRu2Si2 sites")

        # Quantum corrections: the phonon jobs of both pinned sites.
        window._set_active_route("quantum_corrections")
        keys = site_keys(window)
        _configure_phonons(window, list(keys.values()))
        with synthetic_runs(window, _phonon_outputs):
            window._quantum_corrections_controller.submit(list(keys.values()), target="local")

        # Energy barriers: the 1 -> 1' paths up to the in-plane hop between
        # neighbouring site 1 positions, then the 1 -> 2 paths, the shortest
        # configured and run as the paper's NEB band.
        window._set_active_route("energy_barriers")
        barriers = window._energy_barriers_controller
        barriers.generate_paths(keys[1], keys[1], f"{SAME_SITE_MAX_PATH_ANG:g}")
        barriers.select_endpoints(keys[1], keys[2])
        barriers.generate_paths(keys[1], keys[2])
        neb = path_key(window, keys[1], keys[2])
        _configure_neb(window, neb)
        if run_neb:
            with synthetic_runs(window, _neb_outputs):
                barriers.submit([neb], target="local")
        settle(app)
        window._save_project()
        return document.project_root


def results_project(directory: Path) -> Path:
    """The GdRu2Si2 project with both sites relaxed, their phonons and the NEB band finished."""

    return _build_results(directory, run_neb=True)


def configured_project(directory: Path) -> Path:
    """The results project with the NEB row configured but not yet submitted."""

    return _build_results(directory, run_neb=False)


# --- Scene steps --------------------------------------------------------------


def open_workspace(window: MainWindow) -> None:
    """Open the analysis workspace of the two relaxed sites in the navigator."""

    workspace_id = str(window._relaxation_analysis_workspaces()[0]["id"])
    window._set_active_route(window._relaxation_analysis_workspace_route(workspace_id))


def _neb_path_key(window: MainWindow) -> str:
    keys = site_keys(window)
    return path_key(window, keys[1], keys[2])


def open_path_preview(window: MainWindow) -> None:
    """Click ``View...`` on the site 1 to site 2 path that the NEB band follows."""

    window._on_view_energy_barrier_source(_neb_path_key(window))


def open_neb_configuration(window: MainWindow) -> None:
    """Click ``Configure...`` on the site 1 to site 2 path."""

    window._on_configure_energy_barrier_result([_neb_path_key(window)])


def show_transition_state(page: QWidget) -> None:
    """Show the NEB Analysis tab with the highest-energy image in the inspector, viewed down a."""

    page._tabs.setCurrentIndex(1)
    settle(QApplication.instance())
    energies = [row.energy_value for row in page._preview_rows]
    page._apply_preview_index(int(np.argmax(energies)))
    settle(QApplication.instance())
    # Down a the muon sits clear of the Gd, Ru and Si columns.
    view_along_a(page)


def prepare_neb_dialog(dialog: QWidget) -> None:
    hide_catalog_notice(dialog)
    focus_default_button(dialog)


def task_farm_row(dialog: Any) -> QRect:
    """The ``Task farms`` row of the NEB configuration dialog, with its split of the images."""

    spin = dialog._num_farms_spin
    top_left = spin.parentWidget().mapTo(dialog, spin.geometry().topLeft())
    margin = 4
    return QRect(0, top_left.y() - margin, dialog.width(), spin.height() + 2 * margin)


# --- Scenes -------------------------------------------------------------------

SCENES: dict[str, Scene] = {
    "gdru2si2_site_generation": Scene(
        generation_project,
        "muon_candidates",
        Page("_muon_generation_view", (1160, 740), prepare=select_all_sites),
    ),
    "gdru2si2_muon_relaxations": Scene(
        results_project, "muon_relaxations", Page("_muon_relaxation_view", (1100, 440))
    ),
    "gdru2si2_analysis_workspace": Scene(
        results_project,
        None,
        Page("_relaxation_analysis_view", (1100, 720), prepare=frame_viewers),
        setup=open_workspace,
    ),
    "gdru2si2_energy_barrier_paths": Scene(
        results_project, "energy_barriers", Page("_energy_barriers_view", (1100, 740))
    ),
    "gdru2si2_path_preview": Scene(
        results_project,
        "energy_barriers",
        Dialog(
            open_path_preview,
            kind="EnergyBarrierPathViewerDialog",
            size=(860, 720),
            prepare=view_along_a,
        ),
    ),
    "gdru2si2_neb_configuration": Scene(
        configured_project,
        "energy_barriers",
        Dialog(
            open_neb_configuration,
            kind="NudgedElasticBandConfigDialog",
            size=(900, 790),
            prepare=prepare_neb_dialog,
        ),
    ),
    "gdru2si2_neb_task_farms": Scene(
        configured_project,
        "energy_barriers",
        Dialog(
            open_neb_configuration,
            kind="NudgedElasticBandConfigDialog",
            size=(900, 790),
            prepare=prepare_neb_dialog,
            region=task_farm_row,
        ),
    ),
    "gdru2si2_neb_analysis": Scene(
        results_project,
        "energy_barriers",
        Page("_energy_barriers_view", (1100, 860), prepare=show_transition_state),
    ),
    "gdru2si2_quantum_corrections": Scene(
        results_project, "quantum_corrections", Page("_quantum_corrections_view", (1100, 680))
    ),
}
