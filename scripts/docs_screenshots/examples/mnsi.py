"""Manganese silicide: the Magnetic Fields worked example for an incommensurate helix.

B20 MnSi (space group P2₁3, a = 4.558 Å) is a helimagnet below about 29 K: the
Mn moments rotate in the plane perpendicular to a propagation vector q along
[111], left-handed, with a period of about 18 nm. The positive muon stops at
the 4a site with x = 0.532 (Amato et al. 2014). The scenes build that helix
with the structure builder's preset helper, sample the field distribution at
the four 4a positions of one k domain with the real ``muesr`` calculation,
break site II's distribution into its field terms, and list the finite fields
of the helix's commensurate approximant. The project is a host structure and a
Magnetic Fields form: no CASTEP job is involved.
``docs/user_guide/examples/mnsi.rst`` tells the story these scenes illustrate.

The helix and the contact coupling are the plain model of the literature, not
a fit: Dalmas de Réotier et al. (2016) refine a second Mn orbit that this
example leaves out. Every field is computed on a grid of phases, so the
screenshots are identical from one run to the next.
"""

from __future__ import annotations

# The render module configures Qt and VTK, so it is imported before them.
from ..render import (
    Dialog,
    Page,
    ProjectBuilder,
    Scene,
    calculate_and_list,
    calculate_field,
    custom_site,
    focus_default_button,
    frame_viewers,
    magnetic_project,
    settle,
    when_modal,
)

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication, QDialog, QGroupBox, QSplitter, QWidget

from mufinder.domain.models import Structure
from mufinder.science.magnetic_structure import MagneticStructureModel

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow

# --- The host and the helix ------------------------------------------------------

MNSI_A = 4.558  # Å
MN_X = 0.138  # 4a
SI_X = 0.845  # 4a
MUON_X = 0.532  # 4a, the site Amato et al. (2014) find
# The helix: |k| = 0.035 Å⁻¹ along [111], which is 0.0147 r.l.u. along each axis.
Q_RLU = (0.0147, 0.0147, 0.0147)
MN_MOMENT_MU_B = 0.385  # per Mn at 5 K (Dalmas de Réotier et al. 2016)
# The contact coupling that makes the dipolar and contact fields reproduce the
# edges Amato et al. (2014) predict. Their transverse-field coupling,
# -0.9276 mol/emu, is -0.518 T/μB for a moment of 1 μB per Mn; -0.505 T/μB is
# 2.5 % smaller.
CONTACT_COUPLING_T_PER_MU_B = "-0.505"
# Site II of the four 4a positions, which are equivalent to III and IV and
# stand apart from site I on the k axis.
SITE_II = (0.968, 0.468, 0.032)
# 1/64 = 0.015625 is the nearest short-period fraction to 0.0147, 0.0009 r.l.u. away.
APPROXIMANT_PERIOD = 64


def _4a(x: float) -> list[list[float]]:
    """The four 4a positions (x, x, x), (-x+1/2, -x, x+1/2), (-x, x+1/2, -x+1/2), (x+1/2, -x+1/2, -x)."""

    return [
        [value % 1.0 for value in position]
        for position in (
            (x, x, x),
            (-x + 0.5, -x, x + 0.5),
            (-x, x + 0.5, -x + 0.5),
            (x + 0.5, -x + 0.5, -x),
        )
    ]


def mnsi_host() -> Structure:
    """The cubic B20 cell: four Mn and four Si on the 4a positions."""

    return Structure(
        label="MnSi",
        lattice_vectors=[[MNSI_A, 0.0, 0.0], [0.0, MNSI_A, 0.0], [0.0, 0.0, MNSI_A]],
        species=["Mn"] * 4 + ["Si"] * 4,
        fractional_positions=_4a(MN_X) + _4a(SI_X),
    )


def q_basis_model() -> MagneticStructureModel:
    """The model after the propagation vector is typed into the first row of the q basis."""

    return MagneticStructureModel(q_basis_rlu=(Q_RLU, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))


def choose_helix(dialog: QDialog) -> None:
    """Fill the ``Add Magnetic Preset`` dialog: a left-handed helix of Mn moments along the first q-basis row."""

    dialog._kind_combo.setCurrentIndex(dialog._kind_combo.findData("helix"))
    dialog._propagation_combo.setCurrentIndex(dialog._propagation_combo.findData(0))
    dialog._handedness_combo.setCurrentIndex(dialog._handedness_combo.findData(-1))
    dialog._moment.setText(str(MN_MOMENT_MU_B))
    dialog._selector_combo.setCurrentIndex(dialog._selector_combo.findText("Species: Mn"))
    focus_default_button(dialog)


def mnsi_magnetic_model() -> MagneticStructureModel:
    """The helix, built by the preset helper with the choices of :func:`choose_helix`."""

    from mufinder.app.magnetic_field_widget import MagneticPresetDialog

    model = q_basis_model()
    dialog = MagneticPresetDialog(mnsi_host(), model=model)
    choose_helix(dialog)
    helix = dialog.component(dialog.q_basis_coefficients())
    dialog.deleteLater()
    return MagneticStructureModel(q_basis_rlu=model.q_basis_rlu, components=(helix,))


def mnsi_project(
    name: str, form: dict[str, object], model: MagneticStructureModel | None = None
) -> ProjectBuilder:
    """A builder for the MnSi host, its helix (or ``model``) and a Magnetic Fields ``form``.

    ``form`` extends the common one: the muon site at 4a, every equivalent
    site, the contact coupling and a sample without a demagnetizing field.
    """

    host = mnsi_host()
    settings: dict[str, object] = {
        "demag_shape": "none",
        "include_equivalent_sites": True,
        "hyperfine_coupling": CONTACT_COUPLING_T_PER_MU_B,
        "hyperfine_coupling_unit": "T_per_muB",
        "selected_muon_site": custom_site(host, (MUON_X, MUON_X, MUON_X)),
        **form,
    }

    def build(directory: Path) -> Path:
        # The helix is built when the project is, not when the module is imported:
        # the preset helper is a Qt dialog.
        return magnetic_project(name, host, model or mnsi_magnetic_model(), settings)(directory)

    return build


# --- Scene steps --------------------------------------------------------------


def widen_atom_table(page: QWidget) -> None:
    """Widen the atom table so the ``Spin (hbar/2)`` column is not cut off."""

    page.findChild(QSplitter).setSizes([560, 540])
    frame_viewers(page)


def open_builder(window: MainWindow) -> None:
    window._magnetic_field_view._edit_structure_button.click()


def _builder_step(step) -> None:
    """Run ``step`` on the ``Magnetic Structure Builder`` once it is open."""

    when_modal("MagneticStructureBuilderDialog", step)


def open_preset_helper(window: MainWindow) -> None:
    """Open the builder of the q basis alone and click ``Add Preset...``."""

    _builder_step(lambda builder: builder._add_preset_button.click())
    open_builder(window)


def open_component_editor(window: MainWindow) -> None:
    """Open the builder, select the helix and click ``Edit...``."""

    def edit(builder: QDialog) -> None:
        builder._component_table.selectRow(0)
        builder._edit_button.click()

    _builder_step(edit)
    open_builder(window)


def preview_two_cells(builder: QDialog) -> None:
    """Repeat the preview twice along a, b and c: the helix turns slowly, so the arrows show its start."""

    for repeat in (builder._repeat_a, builder._repeat_b, builder._repeat_c):
        repeat.setValue(2)
    settle(QApplication.instance())
    frame_viewers(builder)


def select_helix(builder: QDialog) -> None:
    """Select the helix so the q-basis panel marks the row it uses."""

    builder._component_table.selectRow(0)


def q_basis_panel(builder: QDialog) -> QWidget:
    """The builder's ``Propagation-Vector Basis`` group."""

    for group in builder.findChildren(QGroupBox):
        if group.title() == "Propagation-Vector Basis":
            return group
    raise RuntimeError("The builder has no Propagation-Vector Basis group")


def open_settings(window: MainWindow) -> None:
    window._magnetic_field_view._calculation_settings_button.click()


def show_approximant_settings(dialog: QDialog) -> None:
    """The settings of the approximant run: the option on, the period 64 the model needs."""

    dialog._use_approximants.setChecked(True)
    focus_default_button(dialog)


def list_and_select_continuum(window: MainWindow) -> None:
    """Calculate, list every row and select site II, the first of the continuum."""

    calculate_and_list(window)
    window._magnetic_field_view._result_table.selectRow(1)


def open_distribution_plot(window: MainWindow) -> None:
    """Calculate, select all four distributions and open ``Open Distribution Plot``."""

    page = window._magnetic_field_view
    page._calculate_button.click()
    if page._result_table.rowCount() == 0:
        raise RuntimeError(
            f"The field calculation gave no result: {page._calculation_status.text()}"
        )
    page._result_table.selectAll()
    page._open_distribution_button.click()


def simulate_polarisation(title: str, *, collision_rate: float = 0.0):
    """A scene step: calculate the four sites' fields, open ``Polarisation →`` and add a simulation.

    The settings are those of the worked example: a polycrystal in zero field,
    16 μs on 4001 points, a 6 μs Gaussian window and four-fold zero padding.
    """

    def step(window: MainWindow) -> None:
        calculate_and_list(window)
        window._magnetic_field_view._polarisation_button.click()
        page = window._polarisation_view
        page._title_edit.setText(title)
        page._sample.setCurrentIndex(page._sample.findData(True))
        page._end_time.setValue(16.0)
        page._time_points.setValue(4001)
        page._apodisation_sigma.setText("6")
        page._padding.setValue(4)
        page._collision_rate.setValue(collision_rate)
        page._add_button.click()
        if page.selected_simulation_id() is None:
            raise RuntimeError(f"The polarisation simulation failed: {page._status.text()}")

    return step


# --- Scenes -------------------------------------------------------------------

_APPROXIMANT_PERIOD = {"approximant_max_period": APPROXIMANT_PERIOD}
helix_project = mnsi_project("MnSi", _APPROXIMANT_PERIOD)

SCENES: dict[str, Scene] = {
    "mnsi_host_structure": Scene(
        helix_project,
        "host_structure",
        Page("_host_structure_view", (1100, 760), prepare=widen_atom_table),
    ),
    "mnsi_magnetic_structure_builder": Scene(
        helix_project,
        "magnetic_fields",
        Dialog(
            open_builder,
            kind="MagneticStructureBuilderDialog",
            size=(1180, 900),
            prepare=preview_two_cells,
        ),
    ),
    "mnsi_q_basis": Scene(
        helix_project,
        "magnetic_fields",
        Dialog(
            open_builder,
            kind="MagneticStructureBuilderDialog",
            size=(1180, 900),
            prepare=select_helix,
            region=q_basis_panel,
        ),
    ),
    "mnsi_q_picker": Scene(
        helix_project,
        "magnetic_fields",
        Dialog(open_component_editor, kind="MagneticComponentDialog", prepare=focus_default_button),
    ),
    "mnsi_preset_helper": Scene(
        mnsi_project("MnSi q basis", _APPROXIMANT_PERIOD, q_basis_model()),
        "magnetic_fields",
        Dialog(open_preset_helper, kind="MagneticPresetDialog", prepare=choose_helix),
    ),
    "mnsi_magnetic_fields_page": Scene(
        helix_project,
        "magnetic_fields",
        Page("_magnetic_field_view", (1100, 1140)),
        setup=list_and_select_continuum,
    ),
    "mnsi_calculation_settings": Scene(
        helix_project,
        "magnetic_fields",
        Dialog(
            open_settings,
            kind="MagneticFieldCalculationSettingsDialog",
            prepare=show_approximant_settings,
        ),
    ),
    "mnsi_field_distribution": Scene(
        helix_project, "magnetic_fields", Dialog(open_distribution_plot)
    ),
    "mnsi_distribution_breakdown": Scene(
        mnsi_project(
            "MnSi site II",
            {
                "include_equivalent_sites": False,
                "selected_muon_site": custom_site(mnsi_host(), SITE_II),
            },
        ),
        "magnetic_fields",
        # Tall enough for the contact-shell warning under the whole breakdown.
        Page("_magnetic_field_view", (1100, 1160)),
        setup=lambda window: calculate_field(window._magnetic_field_view),
    ),
    "mnsi_polarisation": Scene(
        helix_project,
        "magnetic_fields",
        Page("_polarisation_view", (1400, 1000)),
        setup=simulate_polarisation("ZF powder"),
    ),
    "mnsi_polarisation_dynamics": Scene(
        helix_project,
        "magnetic_fields",
        Page("_polarisation_view", (1400, 1000)),
        setup=simulate_polarisation("ZF powder, ν = 2 μs⁻¹", collision_rate=2.0),
    ),
    "mnsi_approximant_fields": Scene(
        mnsi_project(
            "MnSi approximant",
            {
                **_APPROXIMANT_PERIOD,
                "use_commensurate_approximants": True,
                "include_equivalent_sites": False,
                "selected_muon_site": custom_site(mnsi_host(), SITE_II),
            },
        ),
        "magnetic_fields",
        Page("_magnetic_field_view", (1100, 1080)),
        setup=lambda window: calculate_field(window._magnetic_field_view),
    ),
}
