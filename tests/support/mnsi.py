"""B20 MnSi, the incommensurate-helix worked example, shared by the field and polarisation tests.

MnSi (P2_1 3, a = 4.558 Å; Mn and Si at 4a with x = 0.138 and 0.845) is a
left-handed helix along [111]: |k| = 0.035 Å^-1, 0.385 mu_B per Mn. The muon
stops at 4a with x = 0.532 (Amato et al. 2014), where the four positions of
one k domain split into site I, on the k axis, and the equivalent sites II-IV.
The contact coupling -0.505 T/mu_B, over the shell of the three Mn tied 1.760 Å
from the muon, reproduces the edges Amato et al. predict: an isolated line at
88.0 mT and a continuum from 90.0 to 206.5 mT.
"""

from __future__ import annotations

from mufinder.domain.models import Structure
from mufinder.science.magnetic_fields import (
    MagneticFieldOptions,
    contact_coupling_to_angstrom_minus3,
)
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    build_magnetic_preset_component,
)
from tests.support.structures import host_structure

MNSI_A = 4.558
MNSI_Q = (0.0147, 0.0147, 0.0147)
MNSI_MOMENT = 0.385
MNSI_MUON = (0.532, 0.532, 0.532)
MNSI_SITE_II = (0.968, 0.468, 0.032)
MNSI_CONTACT_T_PER_MUB = -0.505
MNSI_CONTACT_NEIGHBOURS = 3


def mnsi_4a(x: float) -> tuple[tuple[float, ...], ...]:
    """The four 4a positions of P2_1 3 with parameter ``x``."""

    positions = (
        (x, x, x),
        (-x + 0.5, -x, x + 0.5),
        (-x, x + 0.5, -x + 0.5),
        (x + 0.5, -x + 0.5, -x),
    )
    return tuple(tuple(value % 1.0 for value in position) for position in positions)


def mnsi_structure() -> Structure:
    return host_structure(
        label="MnSi",
        lattice=MNSI_A,
        species=("Mn",) * 4 + ("Si",) * 4,
        positions=mnsi_4a(0.138) + mnsi_4a(0.845),
    )


def mnsi_helix(handedness: int = -1) -> MagneticStructureModel:
    component = build_magnetic_preset_component(
        "helix",
        lattice_vectors=mnsi_structure().lattice_vectors,
        q_rlu=MNSI_Q,
        q_coeff=(1.0, 0.0, 0.0),
        moment=MNSI_MOMENT,
        reference_axis="a",
        selector=AtomSelector.species("Mn"),
        handedness=handedness,
    )
    return MagneticStructureModel(
        q_basis_rlu=(MNSI_Q, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), components=(component,)
    )


def mnsi_options(**changes) -> MagneticFieldOptions:
    contact = contact_coupling_to_angstrom_minus3(MNSI_CONTACT_T_PER_MUB, "T_per_muB")
    return MagneticFieldOptions(
        **{
            "find_equivalent_sites": True,
            "contact_coupling": contact,
            "nnn": MNSI_CONTACT_NEIGHBOURS,
            **changes,
        }
    )
