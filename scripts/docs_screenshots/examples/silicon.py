"""Diamond silicon: the results and analysis scenes, and the legacy importer.

The results pages need finished calculations. No scene runs CASTEP:
:func:`results_project` submits each workflow's jobs through the application's
own controllers and ``JobExecutionController``, with a local runner whose
``submit`` writes the job's CASTEP outputs instead of starting a process (see
:func:`~docs_screenshots.render.synthetic_runs`), and then finishes each job
through ``MainWindow``'s completion handler. Result dispatch, the stored
results, derived structures and relaxed sites are therefore exactly what the
application builds after a real run.

The outputs come from the repository's synthetic fixtures where one fits: the
UEP source job is ``tests/fixtures/uep/Si2.castep`` with its ``.den_fmt``, and
the pristine relaxation is ``tests/fixtures/geometry/Si2_bfgs.castep``. No
fixture holds a muon relaxation, a phonon run of a muoniated cell or a silicon
NEB path, so :mod:`docs_screenshots.synthetic` writes those in CASTEP's file
shapes with illustrative numbers (see ``SI_MUON_SITES``, ``SI_MUON_MODES_CM``
and ``NEB_BARRIER_EV``); the captions of the pages that show them say so.

:func:`running_relaxation_project` stops the pristine relaxation part-way
through, and :func:`silicon_project` holds the host alone for the legacy
importer.
"""

from __future__ import annotations

# The render module configures Qt and VTK, so it is imported before them.
from ..render import (
    FIXTURES,
    XC_FUNCTIONAL,
    Dialog,
    Floating,
    Outputs,
    Page,
    Scene,
    current_page,
    focus_default_button,
    frame_viewers,
    include_every_job,
    new_project,
    opened,
    refuse_message_boxes,
    run_uep_analysis,
    scratch_directory,
    settle,
    synthetic_runs,
    when_modal,
)
from ..synthetic import (
    castep_text,
    cell_text,
    final_configuration,
    legacy_castep_text,
    phonon_text,
    ts_text,
)

from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np
from PySide6.QtWidgets import QApplication, QWidget

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectDocument, ProjectService

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow

SI_PRIMITIVE = 2.6954645  # Å, half the cubic edge of tests/fixtures/uep/Si2.cell
SI_SUPERCELL = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

# Each candidate start relaxes to the nearest bond-centre (BC) or tetrahedral
# (T) interstitial, given in host fractional coordinates; BC is the lower site
# for a positive muon in silicon. The energies, and how far the muon pushes
# its nearest silicon neighbours outwards, are illustrative.
SI_MUON_SITES = (  # start, site kind, relaxed site
    ((0.14, 0.11, 0.13), "BC", (0.125, 0.125, 0.125)),
    ((0.61, 0.14, 0.12), "BC", (0.625, 0.125, 0.125)),
    ((0.13, 0.62, 0.14), "BC", (0.125, 0.625, 0.125)),
    ((0.48, 0.52, 0.50), "T", (0.5, 0.5, 0.5)),
    ((0.74, 0.77, 0.75), "T", (0.75, 0.75, 0.75)),
)
SI_SITE_ENERGY_EV = {"BC": -2741.2364, "T": -2740.9271}
SI_SITE_PUSH_ANG = {"BC": 0.42, "T": 0.03}
SI_NEIGHBOUR_RADIUS_ANG = 2.6
# The phonon of the lowest site: three muon modes at these Γ frequencies, each
# mixed with one silicon neighbour's motion so that the muon carries
# SI_MUON_PARTICIPATION of each (w1 = 3 * participation - 3 = -0.14), below a
# host band up to SI_HOST_TOP_CM. The numbers are illustrative.
SI_MUON_MODES_CM = (2742.0, 3226.0, 3226.0)
SI_MUON_PARTICIPATION = 1.0 - 0.14 / 3.0
SI_HOST_TOP_CM = 520.0
NEB_BARRIER_EV = 0.38  # illustrative height above the reactant
NEB_IMAGES = 5
UEP_ROW = "source_01"  # the UEP page's first row label


def silicon_host() -> Structure:
    """Diamond silicon's two-atom primitive cell, the cell of tests/fixtures/uep/Si2.cell."""

    a = SI_PRIMITIVE
    return Structure(
        label="Si (diamond)",
        lattice_vectors=[[a, a, 0.0], [a, 0.0, a], [0.0, a, a]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _job_cell(job_dir: Path, seed: str):
    from mufinder.castep_io.cell_parser import parse_cell_file

    return parse_cell_file(job_dir / f"{seed}.cell")


def _relaxed_muon_cell(cell: Any, kind: str, start: Any, site: Any) -> str:
    """The input supercell with its muon moved to ``site`` and its neighbours pushed out."""

    lattice = np.asarray(cell.lattice_vectors, dtype=float)
    fractional = np.asarray(cell.fractional_positions, dtype=float)
    muon = next(index for index, symbol in enumerate(cell.species) if symbol == "H:mu")
    repeats = np.rint(np.linalg.norm(lattice, axis=1) / np.linalg.norm(silicon_host().lattice_vectors, axis=1))
    shift = (np.asarray(site) - np.asarray(start) + 0.5) % 1.0 - 0.5
    fractional[muon] = fractional[muon] + shift / repeats
    for index in range(len(fractional)):
        if index == muon:
            continue
        separation = (fractional[index] - fractional[muon] + 0.5) % 1.0 - 0.5
        vector = separation @ lattice
        distance = float(np.linalg.norm(vector))
        if distance < SI_NEIGHBOUR_RADIUS_ANG:
            moved = vector * (1.0 + SI_SITE_PUSH_ANG[kind] / distance)
            fractional[index] = fractional[muon] + np.linalg.solve(lattice.T, moved)
    return cell_text(lattice.tolist(), cell.species, (fractional % 1.0).tolist())


def _pristine_outputs(*, running: bool) -> Outputs:
    """Si2_bfgs.castep and the relaxed cell of its final configuration; cut short while running."""

    def write(_job: Any, job_dir: Path, seed: str) -> bool:
        source = FIXTURES / "geometry" / "Si2_bfgs.castep"
        text = source.read_text(encoding="utf-8")
        if running:
            # The output as CASTEP has written it part-way through the optimisation.
            text = text[: text.index(" BFGS: finished iteration     5")]
        (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
        if not running:
            (job_dir / f"{seed}-out.cell").write_text(cell_text(*final_configuration(source)), encoding="utf-8")
        return not running

    return write


def _muon_outputs(window: MainWindow) -> Outputs:
    """A relaxed cell and an illustrative CASTEP output for each SI_MUON_SITES start."""

    def write(job: Any, job_dir: Path, seed: str) -> bool:
        definition, row = window._muon_relaxation_definition_and_row_for_job(job)
        index, (start, kind, site) = next(
            (index, entry)
            for index, entry in enumerate(SI_MUON_SITES)
            if np.allclose(entry[0], row.initial_fractional_position)
        )
        cell = _job_cell(job_dir, seed)
        (job_dir / f"{seed}-out.cell").write_text(_relaxed_muon_cell(cell, kind, start, site), encoding="utf-8")
        energy = SI_SITE_ENERGY_EV[kind] + 0.0007 * index
        text = castep_text(energy=energy, species=list(cell.species), geometry=True, wall_time=1843.2 + 211.0 * index)
        (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
        return True

    return write


def _uep_outputs(_job: Any, job_dir: Path, seed: str) -> bool:
    """The synthetic UEP source fixture: Si2.castep, Si2.den_fmt and Si2.cell."""

    for suffix in (".castep", ".den_fmt", ".cell"):
        shutil.copyfile(FIXTURES / "uep" / f"Si2{suffix}", job_dir / f"{seed}{suffix}")
    return True


def _phonon_outputs(job: Any, job_dir: Path, seed: str) -> bool:
    """The stage-2 phonon of a quantum-corrections row: its ``.phonon`` and ``.castep``."""

    cell = _job_cell(job_dir, seed)
    phonon = phonon_text(
        cell,
        muon_modes_cm=SI_MUON_MODES_CM,
        muon_participation=SI_MUON_PARTICIPATION,
        host_top_cm=SI_HOST_TOP_CM,
    )
    (job_dir / f"{seed}.phonon").write_text(phonon, encoding="utf-8")
    text = castep_text(energy=SI_SITE_ENERGY_EV["BC"] - 0.0004, species=list(cell.species), geometry=False, wall_time=9126.4)
    (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
    return True


def _neb_outputs(window: MainWindow) -> Outputs:
    """An illustrative NEB band through the job's reactant, intermediate and product cells.

    The images follow the input band, reactant to intermediate to product; the
    energy profile rises NEB_BARRIER_EV above the straight line between the
    two relaxed endpoint energies.
    """

    def write(job: Any, job_dir: Path, seed: str) -> bool:
        cell = _job_cell(job_dir, seed)
        lattice = np.asarray(cell.lattice_vectors, dtype=float)
        reactant = np.asarray(cell.fractional_positions, dtype=float)

        def block(name: str) -> np.ndarray:
            rows = next(item for item in cell.cell_file.blocks if item.name.lower() == name).rows
            positions = np.asarray([[float(value) for value in row[1:4]] for row in rows], dtype=float)
            return reactant + (positions - reactant + 0.5) % 1.0 - 0.5

        intermediate = block("positions_frac_intermediate")
        product = intermediate + (block("positions_frac_product") - intermediate + 0.5) % 1.0 - 0.5
        reactant_energy, product_energy = source_energies(window, job)
        images = []
        for image in range(NEB_IMAGES + 2):
            coordinate = image / (NEB_IMAGES + 1)
            tag = "REA" if image == 0 else "PRO" if image == NEB_IMAGES + 1 else "TST"
            energy = (
                reactant_energy
                + (product_energy - reactant_energy) * coordinate
                + NEB_BARRIER_EV * np.sin(np.pi * coordinate) ** 2
            )
            if coordinate <= 0.5:
                fractional = reactant + (intermediate - reactant) * 2.0 * coordinate
            else:
                fractional = intermediate + (product - intermediate) * (2.0 * coordinate - 1.0)
            images.append((tag, coordinate, energy, fractional @ lattice))
        (job_dir / f"{seed}.ts").write_text(ts_text(lattice, list(cell.species), images), encoding="utf-8")
        text = castep_text(energy=images[-1][2], species=list(cell.species), geometry=False, wall_time=15214.8)
        (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
        return True

    return write


def source_energies(window: MainWindow, job: Any) -> tuple[float, float]:
    """The relaxed energies of the NEB job's reactant and product sites."""

    controller = window._energy_barriers_controller
    _definition, key = controller.definition_and_key_for_job(job)
    reactant, product = str(key).split(":")[:2]
    sites = window._document.sites
    return sites[UUID(reactant)].total_energy_eV, sites[UUID(product)].total_energy_eV


def _silicon_document(directory: Path) -> tuple[ProjectService, ProjectDocument]:
    """The silicon host with ``PBE`` and a saved 2x2x2 supercell for the muon relaxations."""

    host = silicon_host()
    service, document = new_project(directory, "Silicon", host)
    service.set_recommended_xc_functional(document, XC_FUNCTIONAL)
    service.create_workflow_supercell_asset(
        document,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        host_structure_id=host.id,
        matrix=SI_SUPERCELL,
        label="Si 2x2x2",
    )
    document.project.supercell_matrix = [list(row) for row in SI_SUPERCELL]
    return service, document


def silicon_project(directory: Path) -> Path:
    """The silicon host alone, for the legacy importer."""

    service, document = _silicon_document(directory)
    service.save_project(document)
    return document.project_root


def _build_results(directory: Path, *, running_pristine: bool = False) -> Path:
    """Build and save the results project; see the module docstring."""

    app = QApplication.instance()
    service, document = _silicon_document(directory)
    for start, _kind, _site in SI_MUON_SITES:
        service.generate_candidate_sites(
            document,
            method="manual",
            parameters={"coord_system": "fractional", "x": start[0], "y": start[1], "z": start[2]},
        )
    service.save_project(document)

    with opened(app, document.project_root) as window, refuse_message_boxes(app):
        document = window._document
        # Pristine relaxation: one variable-cell row, run as Run All Locally runs it.
        pristine = window._pristine_relaxation_view
        window._set_active_route("pristine_relaxation")
        pristine._add_row()
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
        with synthetic_runs(window, _uep_outputs):
            window._uep_view._run_local_selected_rows()
        run_uep_analysis(window, UEP_ROW)

        # Muon relaxations of every candidate site in the 2x2x2 supercell.
        site_ids = [str(site.id) for site in document.sites.values()]
        when_modal("MuonChargeStateDialog", lambda dialog: dialog.accept())
        window._on_create_muon_relaxation_jobs_requested(site_ids)
        relaxations = window._muon_relaxation_view
        for entry in list(relaxations._row_entries()):
            relaxations.set_row_supercell(entry.row_id, [list(row) for row in SI_SUPERCELL])
        with synthetic_runs(window, _muon_outputs(window)):
            relaxations._run_local_selected_rows()

        # An analysis workspace of every relaxation, clustered and pinned.
        window._set_active_route("muon_relaxations")
        relaxations._active_table().selectAll()
        when_modal("QInputDialog", lambda dialog: (dialog.setTextValue("Positive muon sites"), dialog.accept()))
        relaxations._create_analysis_button.click()
        workspace_id = str(window._relaxation_analysis_workspaces()[0]["id"])
        when_modal("RelaxationClusteringDialog", lambda dialog: dialog._apply_button.click())
        window._relaxation_analysis_view._open_clustering(workspace_id)
        window._on_muon_sites_pin_requested(workspace_id)

        # Quantum corrections: the phonon job of the lowest-energy site.
        window._set_active_route("quantum_corrections")
        quantum = window._quantum_corrections_controller
        candidates = sorted(quantum.candidates(), key=lambda candidate: candidate.site.total_energy_eV)
        with synthetic_runs(window, _phonon_outputs):
            quantum.submit([candidates[0].key], target="local")

        # Energy barriers: the BC-to-T paths, and an NEB band along the first.
        window._set_active_route("energy_barriers")
        barriers = window._energy_barriers_controller
        energies = {row.key: document.sites[UUID(row.key)].total_energy_eV for row in barriers.endpoint_rows()}
        reactant, product = min(energies, key=energies.get), max(energies, key=energies.get)
        barriers.select_endpoints(reactant, product)
        barriers.generate_paths(reactant, product)
        with synthetic_runs(window, _neb_outputs(window)):
            barriers.submit([barriers.candidates()[0].key], target="local")
        window._save_project()
        return document.project_root


def results_project(directory: Path) -> Path:
    """The silicon project with every results workflow finished; see the module docstring."""

    return _build_results(directory)


def running_relaxation_project(directory: Path) -> Path:
    """The silicon project with its pristine relaxation part-way through."""

    return _build_results(directory, running_pristine=True)


# --- The legacy folder ----------------------------------------------------------


def _legacy_folder(root: Path) -> Path:
    """A legacy working folder of three silicon muon relaxations, in the documented layout.

    Hand-authored, with illustrative energies: see docs/legacy_import.rst,
    "Research data boundary". Site 3 stopped before its geometry converged.
    """

    from mufinder.science.supercell import build_supercell_structure

    supercell, _spins = build_supercell_structure(silicon_host(), SI_SUPERCELL)
    for index, ((start, kind, site), converged) in enumerate(zip(SI_MUON_SITES[2:], (True, True, False)), start=1):
        species = [*supercell.species, "H:mu"]
        initial = [*supercell.fractional_positions, [value / 2.0 for value in start]]
        relaxed = [*supercell.fractional_positions, [value / 2.0 for value in site]]
        stem = f"Si_{index}"
        for folder in ("input", "results", f"run/site{index}"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        (root / "input" / f"{stem}.cell").write_text(
            cell_text(supercell.lattice_vectors, species, initial) + "\nFIX_ALL_CELL : TRUE\n", encoding="utf-8"
        )
        (root / "results" / f"{stem}-out.cell").write_text(
            cell_text(supercell.lattice_vectors, species, relaxed), encoding="utf-8"
        )
        energy = SI_SITE_ENERGY_EV[kind] + 0.0011 * index
        (root / "results" / f"{stem}.castep").write_text(
            legacy_castep_text(energy=energy, converged=converged), encoding="utf-8"
        )
        (root / f"run/site{index}" / f"{stem}.param").write_text(
            "task : geometryoptimisation\ncharge : +1\nxc_functional : PBE\ncut_off_energy : 400 eV\n"
            "write_cell_structure : true\ngeom_max_iter : 300\n",
            encoding="utf-8",
        )
    a = 2.0 * SI_PRIMITIVE
    (root / "Si.cif").write_text(
        f"data_Si\n_symmetry_space_group_name_H-M 'F d -3 m'\n_cell_length_a {a:.6f}\n_cell_length_b {a:.6f}\n"
        f"_cell_length_c {a:.6f}\n_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nSi 0 0 0\n",
        encoding="utf-8",
    )
    return root


# --- Scene steps --------------------------------------------------------------


def open_analysis_workspace(window: MainWindow) -> None:
    """Open the results project's analysis workspace in the navigator."""

    workspace_id = str(window._relaxation_analysis_workspaces()[0]["id"])
    window._set_active_route(window._relaxation_analysis_workspace_route(workspace_id))


def open_comparison_table(window: MainWindow) -> None:
    """Open the results project's comparison table in the navigator."""

    table_id = next(iter(window._document.comparison_tables))
    window._set_active_route(window._comparison_route(table_id))


def show_running_progress(window: MainWindow) -> QWidget:
    """Inspect the running pristine relaxation, as its dashboard Inspect button does.

    Reopening a project ends its local jobs, since no local runner survives
    the application, so the scene puts the job back in the state the local
    runner keeps it in while CASTEP writes its ``.castep``.
    """

    from mufinder.domain.enums import JobStatus

    job = next(job for job in window._document.jobs.values() if job.workflow_stage == WorkflowStage.PRISTINE_RELAXATION)
    job.status = JobStatus.RUNNING
    castep = window._job_output_directory(job) / f"{window._job_seed_name(job)}.castep"
    monitor = window._job_progress
    monitor._local_output_paths = lambda job_id: (castep,) if job_id == job.id else ()
    monitor._run_in_background = lambda task: task()
    window._refresh_jobs_table()
    return window._show_geometry_progress(job)


def open_uep_clustering(window: MainWindow) -> QWidget:
    """Open the UEP analysis of the source job, then its saved batch search's clustering."""

    window._on_open_uep_analysis_requested(UEP_ROW)
    analysis = next(iter(window._uep_analysis_dialogs.values()))
    analysis._open_clustering_dialog()
    analysis.hide()
    return analysis._clustering_dialog


def show_neb_analysis(page: QWidget) -> None:
    """Show the NEB Analysis tab with the transition-state image in the inspector."""

    page._tabs.setCurrentIndex(1)
    settle(QApplication.instance())
    page._apply_preview_index((NEB_IMAGES + 1) // 2)
    settle(QApplication.instance())
    frame_viewers(page)


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
    root = _legacy_folder(scratch_directory("legacy-") / "si-muon-sites")
    scan = scan_legacy_folder(root, host_lattice=[list(row) for row in host.lattice_vectors], host_species=list(host.species))
    scan = replace(scan, warnings=tuple(warning for warning in scan.warnings if "keyword catalog" not in warning))
    LegacyImportDialog(scan, parent=window).exec()


# --- Scenes -------------------------------------------------------------------

SCENES: dict[str, Scene] = {
    "pristine_relaxation_page": Scene(results_project, "pristine_relaxation", Page("_pristine_relaxation_view", (1100, 540))),
    "geometry_progress_dialog": Scene(
        running_relaxation_project, "pristine_relaxation", Floating(show_running_progress, size=(820, 640))
    ),
    "comparison_table": Scene(results_project, None, Page(current_page, (780, 820)), setup=open_comparison_table),
    "uep_page": Scene(results_project, "uep", Page("_uep_view", (1100, 540))),
    "uep_clustering_dialog": Scene(results_project, "uep", Floating(open_uep_clustering, size=(1180, 760), prepare=frame_viewers)),
    "muon_relaxations_page": Scene(results_project, "muon_relaxations", Page("_muon_relaxation_view", (1100, 540))),
    "relaxation_analysis_workspace": Scene(
        results_project, None, Page("_relaxation_analysis_view", (1100, 680), prepare=frame_viewers), setup=open_analysis_workspace
    ),
    "muon_sites_summary": Scene(results_project, "muon_sites", Page("_muon_sites_summary_view", (1100, 800), prepare=frame_viewers)),
    "quantum_corrections_page": Scene(
        results_project, "quantum_corrections", Page("_quantum_corrections_view", (1100, 620))
    ),
    "energy_barriers_page": Scene(
        results_project, "energy_barriers", Page("_energy_barriers_view", (1100, 740))
    ),
    "energy_barrier_inspection": Scene(
        results_project, "energy_barriers", Page("_energy_barriers_view", (1100, 820), prepare=show_neb_analysis)
    ),
    "legacy_import_dialog": Scene(silicon_project, None, Dialog(open_legacy_import, kind="LegacyImportDialog", prepare=focus_default_button)),
}
