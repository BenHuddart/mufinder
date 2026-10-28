"""Body-centred cubic iron: the Magnetic Fields worked example.

bcc Fe (Im-3m, a = 2.868 Å) is a ferromagnet with 2.2 μB per atom, here along
c. The positive muon stops in an interstitial site of the bcc lattice, either
the octahedral site (1/2, 0, 0) (Wyckoff 6b) or the tetrahedral site
(1/2, 1/4, 0) (12d). The scenes calculate the field at each with the real
``muesr`` calculation, with every equivalent site listed: the contact field is
set to the −1.11 T measured at low temperature (Nishida et al. 1977). Those
projects are a host structure and a Magnetic Fields form: no CASTEP job is
involved. :func:`fe_hyperfine_project` instead runs the muon relaxations and
the tetrahedral site's hyperfine job through the window with synthetic
outputs (see :func:`~docs_screenshots.render.synthetic_runs`), the hyperfine
job's being the CASTEP run on the Hyperfine Coupling theory page.
``docs/user_guide/examples/fe.rst`` tells the story these scenes illustrate.
"""

from __future__ import annotations

# The render module configures Qt and VTK, so it is imported before them.
from ..render import (
    PROJECT_ROOT,
    Dialog,
    Page,
    ProjectBuilder,
    Scene,
    Window,
    calculate_and_list,
    calculation_card,
    cluster_and_pin,
    current_page,
    custom_site,
    focus_default_button,
    magnetic_project,
    new_project,
    opened,
    refuse_message_boxes,
    save_magnetic_workspace,
    settle,
    show_every_result_row,
    show_host_with_spins,
    synthetic_runs,
    when_modal,
)
from ..synthetic import castep_text, cell_text, den_fmt_text

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import image as mpl_image
from matplotlib.path import Path as PolygonPath
from PySide6.QtWidgets import QApplication, QWidget

from mufinder.domain.models import Structure
from mufinder.science.demagnetization import DemagnetizingSample, PlatePhoto
from mufinder.science.sample_image import segment_plate_image
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    build_collinear_component,
)

from typing import Any

from mufinder.domain.charge_states import POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.enums import WorkflowStage
from mufinder.science.hyperfine import per_bohr3_to_per_angstrom3

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow

# --- The host -------------------------------------------------------------------

FE_A = 2.868  # Å
FE_MOMENT_MU_B = 2.2  # per Fe, along c
FE_SPIN = 2.2  # ħ/2 per Fe: one unpaired electron per μB
OCTAHEDRAL_SITE = (0.5, 0.0, 0.0)  # 6b
TETRAHEDRAL_SITE = (0.5, 0.25, 0.0)  # 12d
# The contact field measured at low temperature is -1.11 T (Nishida et al.
# 1977). The two nearest Fe each carry 2.2 μB, so the contact term is the
# coupling times 2.2 μB: -1.11 T / 2.2 μB.
CONTACT_COUPLING_T_PER_MU_B = "-0.5045455"
# An 80 x 80 x 80 lattice sum keeps the cubic symmetry of the Lorentz sphere
# exactly, so the components that vanish by symmetry print as 0; the default
# 100 x 100 x 100 sum leaves 1e-7 T of numerical noise in them.
SUPERCELL = [80, 80, 80]
MEASURED_MAGNETISATION_MU_B = 2.0  # μB per Fe: a sample short of saturation, illustrative
# A hexagonal platelet photographed along its normal, c: illustrative, not a real crystal.
PLATE_PHOTO_SIZE = (480, 640)  # pixels, rows x columns
PLATE_WIDTH = 5.0  # along a, the width of the outline's bounding box
PLATE_THICKNESS = 0.1
PLATE_GRID = 64


def fe_host() -> Structure:
    """The conventional bcc cell with a ferromagnetic spin on each Fe."""

    return Structure(
        label="Fe",
        lattice_vectors=[[FE_A, 0.0, 0.0], [0.0, FE_A, 0.0], [0.0, 0.0, FE_A]],
        species=["Fe", "Fe"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        properties={"atomic_spins": [FE_SPIN, FE_SPIN]},
    )


def fe_magnetic_model() -> MagneticStructureModel:
    """The ferromagnet: 2.2 μB along c on every Fe."""

    return MagneticStructureModel(
        components=(
            build_collinear_component(
                label="Fe moment",
                selector=AtomSelector.all(),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, FE_MOMENT_MU_B),
            ),
        ),
    )


def fe_project(name: str, form: dict[str, object]) -> ProjectBuilder:
    """A builder for the Fe host, its ferromagnetic model and a Magnetic Fields ``form``.

    ``form`` extends the common one: the contact coupling, every equivalent
    site and a spherical sample.
    """

    settings: dict[str, object] = {
        "demag_shape": "sphere",
        "include_equivalent_sites": True,
        "supercell": SUPERCELL,
        "hyperfine_coupling": CONTACT_COUPLING_T_PER_MU_B,
        "hyperfine_coupling_unit": "T_per_muB",
        **form,
    }
    return magnetic_project(name, fe_host(), fe_magnetic_model(), settings)


def plate_photo(path: Path) -> np.ndarray:
    """Write a synthetic photo of a dark hexagonal platelet on a light, unevenly lit card.

    The platelet has a bright glare spot, which ``Fill holes`` keeps in the
    outline. Returns the grey levels written.
    """

    rows, columns = PLATE_PHOTO_SIZE
    y, x = np.mgrid[0:rows, 0:columns].astype(float)
    grey = 205.0 + 30.0 * (x / columns) - 20.0 * (y / rows)
    angles = np.radians(np.arange(6) * 60.0 + 8.0)
    radii = np.array([210.0, 190.0, 215.0, 200.0, 185.0, 205.0])
    vertices = np.column_stack(
        [320.0 + 1.15 * radii * np.cos(angles), 240.0 + 0.95 * radii * np.sin(angles)]
    )
    inside = PolygonPath(vertices).contains_points(np.column_stack([x.ravel(), y.ravel()]))
    inside = inside.reshape(rows, columns)
    grey = np.where(inside, 55.0 + 15.0 * (y / rows), grey)
    glare = (x - 260.0) ** 2 / 40.0**2 + (y - 190.0) ** 2 / 18.0**2 <= 1.0
    grey = np.where(glare & inside, 235.0, grey)
    grey += np.random.default_rng(258).normal(0.0, 6.0, grey.shape)
    levels = np.clip(np.round(grey), 0, 255).astype(np.uint8)
    mpl_image.imsave(path, levels, cmap="gray", vmin=0, vmax=255)
    return levels


def fe_plate_project(name: str) -> ProjectBuilder:
    """The Fe project whose sample is a platelet outlined in a photo beside the project."""

    def build(directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        photo_path = directory / "fe_platelet.png"
        segmentation = segment_plate_image(plate_photo(photo_path))
        plate = DemagnetizingSample.image_plate(
            segmentation.in_plane_mask(),
            (PLATE_WIDTH, PLATE_WIDTH * segmentation.aspect_ratio),
            PLATE_THICKNESS,
            fourier_grid=PLATE_GRID,
            photo=PlatePhoto(path=str(photo_path)),
        )
        form = {**_OCTAHEDRAL, "demag_shape": "image_plate", "demag_sample": plate.to_json()}
        return fe_project(name, form)(directory)

    return build


# --- Scene steps --------------------------------------------------------------


def select_parallel_site(page: QWidget) -> None:
    """Select the first result row whose dipolar field points along the moment."""

    table = page._result_table
    for row in range(table.rowCount()):
        if float(table.item(row, 3).text()) > 0:
            table.selectRow(row)
            return
    raise RuntimeError("No result row has a field along the moment")


def select_infinite_plate(window: MainWindow) -> None:
    """Calculate with the sample an infinite plate normal to c, along the moment."""

    page = window._magnetic_field_view
    page._plate_radio.setChecked(True)
    calculate_and_list(window)


def open_advanced_demagnetizing_field(window: MainWindow) -> None:
    """Open the Advanced Demagnetizing Field window from the Calculation card."""

    window._magnetic_field_view._advanced_demag_button.click()


def open_plate_photo(window: MainWindow) -> None:
    """Open Plate from Photo from the Advanced Demagnetizing Field window."""

    when_modal("DemagnetizingFieldAdvancedDialog", lambda dialog: dialog._photo_button.click())
    open_advanced_demagnetizing_field(window)


def orient_disc_normal_to_110(dialog: QWidget) -> None:
    """Make the sample an oblate disc cut with its face on (1 1 0) and its a axis along [0 0 1]."""

    dialog._ellipsoid_a.setValue(4.0)
    dialog._ellipsoid_b.setValue(4.0)
    dialog._ellipsoid_c.setValue(0.5)
    dialog._orientation_mode.setCurrentIndex(dialog._orientation_mode.findData("directions"))
    for edit, value in zip(dialog._normal_inputs, ("1", "1", "0"), strict=True):
        edit.setText(value)
    for edit, value in zip(dialog._in_plane_inputs, ("0", "0", "1"), strict=True):
        edit.setText(value)
    focus_default_button(dialog)


def save_both_workspaces(window: MainWindow) -> None:
    """Save the octahedral field as a workspace, move to the tetrahedral site and save that too."""

    page = window._magnetic_field_view
    save_magnetic_workspace("Octahedral site")(window)
    settle(QApplication.instance())

    def choose_tetrahedral(dialog: QWidget) -> None:
        for edit, value in zip(
            (dialog._custom_x, dialog._custom_y, dialog._custom_z), TETRAHEDRAL_SITE, strict=True
        ):
            edit.setText(f"{value:g}")
        dialog.accept()

    when_modal("MuonSiteSelectionDialog", choose_tetrahedral)
    page._select_site_button.click()
    save_magnetic_workspace("Tetrahedral site")(window)
    # Show the octahedral workspace, whose six rows fit the window.
    workspace_id = window._document.ui_state["magnetic_field_workspaces"][0]["id"]
    window._set_active_route(window._magnetic_field_workspace_route(workspace_id))
    show_every_result_row(current_page(window))


# --- Scenes -------------------------------------------------------------------

_OCTAHEDRAL = {"selected_muon_site": custom_site(fe_host(), OCTAHEDRAL_SITE)}
_TETRAHEDRAL = {"selected_muon_site": custom_site(fe_host(), TETRAHEDRAL_SITE)}
octahedral_project = fe_project("Iron", _OCTAHEDRAL)

# --- Hyperfine Coupling ---------------------------------------------------------
# The run on the Hyperfine Coupling theory page: the muon at the unrelaxed
# tetrahedral and octahedral sites of a 2 x 2 x 2 supercell, PBE. The
# relaxations return their starts; the tetrahedral site's magres job gives
# that run's tensor (its CASTEP output, kept under tests/fixtures/hyperfine),
# its 2.20 μB Fe moments and its pseudo spin density at the muon.
HYPERFINE_SUPERCELL = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
HYPERFINE_CUTOFF_EV = 700.0
HYPERFINE_KPOINTS = (8, 8, 8)
HYPERFINE_MAGRES = PROJECT_ROOT / "tests" / "fixtures" / "hyperfine" / "fe_tetrahedral_muon.magres"
FE_MULLIKEN_SPIN = 2.20  # μB per Fe
MUON_MULLIKEN_SPIN = -0.02  # μB
PSEUDO_SPIN_DENSITY_BOHR3 = -0.0180  # at the tetrahedral muon
# Final energies of the two relaxations in eV: DFT puts the muon at the tetrahedral site.
SITE_ENERGIES_EV = {TETRAHEDRAL_SITE: -13827.336, OCTAHEDRAL_SITE: -13827.169}
# Model atomic densities for the formatted density's charge column, (electrons, width in Å).
DENSITY_SHELLS = {"Fe": [(8.0, 0.7)], "H:mu": [(1.0, 0.5)]}
DENSITY_GRID = (24, 24, 24)


def _job_cell(job_dir: Path, seed: str) -> Any:
    from mufinder.castep_io.cell_parser import parse_cell_file

    return parse_cell_file(job_dir / f"{seed}.cell").cell_file


def _cell_atoms(cell: Any) -> tuple[list[str], list[list[float]]]:
    positions = cell.positions_frac.positions
    return (
        [position.species for position in positions],
        [list(position.coordinates) for position in positions],
    )


def _relaxation_outputs(window: MainWindow):
    """An unmoved ``-out.cell`` and a CASTEP output with Mulliken spins for each start."""

    def write(job: Any, job_dir: Path, seed: str) -> bool:
        _definition, row = window._muon_relaxation_definition_and_row_for_job(job)
        start = tuple(float(value) for value in row.initial_fractional_position)
        cell = _job_cell(job_dir, seed)
        species, fractional = _cell_atoms(cell)
        (job_dir / f"{seed}-out.cell").write_text(
            cell_text(cell.lattice_cart.vectors, species, fractional), encoding="utf-8"
        )
        spins = [MUON_MULLIKEN_SPIN if symbol == "H:mu" else FE_MULLIKEN_SPIN for symbol in species]
        energy = min(
            SITE_ENERGIES_EV.items(), key=lambda item: np.linalg.norm(np.subtract(item[0], start))
        )[1]
        text = castep_text(
            energy=energy, species=species, geometry=True, wall_time=659.4, mulliken_spins=spins
        )
        (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
        return True

    return write


def _hyperfine_outputs(_job: Any, job_dir: Path, seed: str) -> bool:
    """The tetrahedral run's ``.magres``, a CASTEP output and a spin-polarised density."""

    cell = _job_cell(job_dir, seed)
    species, fractional = _cell_atoms(cell)
    (job_dir / f"{seed}.magres").write_text(
        HYPERFINE_MAGRES.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (job_dir / f"{seed}.den_fmt").write_text(
        den_fmt_text(
            cell.lattice_cart.vectors,
            species,
            fractional,
            grid=DENSITY_GRID,
            shells=DENSITY_SHELLS,
            spin_density_angstrom3=per_bohr3_to_per_angstrom3(PSEUDO_SPIN_DENSITY_BOHR3),
        ),
        encoding="utf-8",
    )
    text = castep_text(energy=-13827.2, species=species, geometry=False, wall_time=220.0)
    (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
    return True


def fe_hyperfine_project(directory: Path) -> Path:
    """Both sites relaxed, clustered and pinned; the tetrahedral site's hyperfine job run."""

    app = QApplication.instance()
    host = fe_host()
    service, document = new_project(directory, "Iron hyperfine", host)
    document.project.magnetic_structure = fe_magnetic_model().to_json()
    service.set_recommended_xc_functional(document, "PBE")
    service.create_workflow_supercell_asset(
        document,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        host_structure_id=host.id,
        matrix=HYPERFINE_SUPERCELL,
        label="2x2x2",
    )
    lattice = (np.asarray(HYPERFINE_SUPERCELL, float) @ np.asarray(host.lattice_vectors)).tolist()
    service.set_recommended_cutoff(
        document, HYPERFINE_CUTOFF_EV, method="manual", xc_functional="PBE"
    )
    service.set_recommended_kpoint_grid(
        document, HYPERFINE_KPOINTS, lattice_vectors=lattice, method="manual", xc_functional="PBE"
    )
    for x, y, z in (TETRAHEDRAL_SITE, OCTAHEDRAL_SITE):
        service.generate_candidate_sites(
            document,
            method="manual",
            parameters={"coord_system": "fractional", "x": x, "y": y, "z": z},
        )
    service.save_project(document)

    with opened(app, document.project_root) as window, refuse_message_boxes(app):
        document = window._document
        when_modal("MuonChargeStateDialog", lambda dialog: dialog.accept())
        window._on_create_muon_relaxation_jobs_requested(
            [str(site.id) for site in document.sites.values()]
        )
        relaxations = window._muon_relaxation_view
        for entry in list(relaxations._row_entries()):
            relaxations.set_row_supercell(entry.row_id, HYPERFINE_SUPERCELL)
        relaxations._active_table().clearSelection()
        with synthetic_runs(window, _relaxation_outputs(window)):
            relaxations._run_local_selected_rows()
        cluster_and_pin(window, POSITIVE_MUON_CHARGE_STATE, "Iron sites")

        window._set_active_route("hyperfine_coupling")
        hyperfine = window._hyperfine_coupling_controller
        tetrahedral = min(hyperfine.candidates(), key=lambda candidate: candidate.total_energy_eV)
        with synthetic_runs(window, _hyperfine_outputs):
            hyperfine.submit([tetrahedral.key], target="local")
        window._save_project()
        return document.project_root


def select_first_result(page: QWidget) -> None:
    """Select the first completed result, so Use in Magnetic Fields is enabled."""

    page._results_table.selectRow(0)


SCENES: dict[str, Scene] = {
    "fe_host_structure": Scene(
        octahedral_project,
        "host_structure",
        Page("_host_structure_view", (1100, 760), prepare=show_host_with_spins),
    ),
    "fe_magnetic_fields_page": Scene(
        octahedral_project,
        "magnetic_fields",
        Page("_magnetic_field_view", (1100, 1260)),
        setup=calculate_and_list,
    ),
    "fe_tetrahedral_sites": Scene(
        fe_project("Iron tetrahedral", _TETRAHEDRAL),
        "magnetic_fields",
        Page("_magnetic_field_view", (1100, 1250)),
        setup=calculate_and_list,
    ),
    "fe_equivalent_sites": Scene(
        fe_project("Iron without demagnetizing field", {**_OCTAHEDRAL, "demag_shape": "none"}),
        "magnetic_fields",
        Page(calculation_card, (1100, 900)),
        setup=lambda window: (
            calculate_and_list(window),
            select_parallel_site(window._magnetic_field_view),
        ),
    ),
    "fe_demagnetizing_shapes": Scene(
        octahedral_project,
        "magnetic_fields",
        Page(calculation_card, (1100, 900)),
        setup=select_infinite_plate,
    ),
    "fe_oriented_demagnetizing_sample": Scene(
        octahedral_project,
        "magnetic_fields",
        Dialog(
            open_advanced_demagnetizing_field,
            kind="DemagnetizingFieldAdvancedDialog",
            size=(1000, 640),
            prepare=orient_disc_normal_to_110,
        ),
    ),
    "fe_plate_photo_outline": Scene(
        fe_plate_project("Iron platelet outline"),
        "magnetic_fields",
        Dialog(
            open_plate_photo,
            kind="SamplePhotoDialog",
            size=(1000, 600),
            prepare=focus_default_button,
        ),
    ),
    "fe_photographed_plate": Scene(
        fe_plate_project("Iron platelet"),
        "magnetic_fields",
        Dialog(
            open_advanced_demagnetizing_field,
            kind="DemagnetizingFieldAdvancedDialog",
            size=(1000, 640),
            prepare=focus_default_button,
        ),
    ),
    "fe_measured_magnetisation": Scene(
        fe_project(
            "Iron measured magnetisation",
            {
                **_OCTAHEDRAL,
                "use_measured_magnetization": True,
                "measured_magnetization": ["0", "0", str(MEASURED_MAGNETISATION_MU_B)],
            },
        ),
        "magnetic_fields",
        Page(calculation_card, (1100, 900)),
        setup=calculate_and_list,
    ),
    "fe_select_muon_site": Scene(
        octahedral_project,
        "magnetic_fields",
        Dialog(
            lambda window: window._magnetic_field_view._select_site_button.click(),
            kind="MuonSiteSelectionDialog",
            size=(760, 230),
            prepare=focus_default_button,
        ),
    ),
    "fe_hyperfine_coupling_page": Scene(
        fe_hyperfine_project,
        "hyperfine_coupling",
        Page("_hyperfine_coupling_view", (1100, 820), prepare=select_first_result),
    ),
    "fe_magnetic_workspaces": Scene(
        octahedral_project, "magnetic_fields", Window((1280, 1140)), setup=save_both_workspaces
    ),
}
