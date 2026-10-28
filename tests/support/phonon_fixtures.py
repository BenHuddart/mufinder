"""Synthetic CASTEP ``.phonon`` fixtures built from a toy force-constant model.

``n2_mu_n2_phonon_text`` returns the Gamma-point ``.phonon`` file of a reduced
[N2-mu-N2]+ complex: five atoms (N, N, H:mu, N, N) on a line along ``x`` in a
cubic box, the muon midway between two N2 molecules. It is a toy model of the
complex of Gomilsek *et al.*, Commun. Phys. 6, 142 (2023), small enough for the
science tests to check every mode of.

The force constants are pair springs plus a lattice tether on every nitrogen:

- a muon-nitrogen spring to each nearest nitrogen, stiff along the chain
  (``muon_nitrogen_stretch``) and across it (``muon_nitrogen_bend``);
- the N2 intramolecular spring, along (``nitrogen_stretch``) and across
  (``nitrogen_bend``) each molecule;
- an isotropic spring tying every nitrogen to the static host lattice
  (``nitrogen_lattice``).

Each pair spring is diagonal in ``x, y, z`` for a chain along ``x``, so the
Hessian splits into one 5 x 5 block along the chain and one transverse block
shared by ``y`` and ``z``. Diagonalising the blocks separately keeps the two
transverse modes of each pair exactly degenerate and their eigenvectors
deterministic. The eigenvectors written to the file are the orthonormal
eigenvectors of the mass-weighted dynamical matrix, as CASTEP writes them.

The constants in ``N2_MU_N2_FORCE_CONSTANTS`` were tuned once, offline, so that
the three muon-dominated modes carry a zero-point energy of 0.174, 0.203 and
0.203 eV (0.17, 0.20 and 0.20 eV, 0.58 eV in total), their summed muon
participation is 2.86, and the harmonic muon-nearest-nitrogen correlation
``|w2|`` is about 0.26: the paper's harmonic numbers. Nothing is tuned at test
time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from mufinder.domain.constants import (
    ATOMIC_MASS_KG,
    ELEMENTARY_CHARGE_C,
    LIGHT_SPEED_M_S,
    MUON_MASS_AMU,
)

#: Nitrogen mass written in the fixture header (AMU).
NITROGEN_MASS_AMU = 14.0067
#: Muon mass written in the fixture header (AMU), as CASTEP echoes it.
FIXTURE_MUON_MASS_AMU = round(MUON_MASS_AMU, 7)
#: Edge of the cubic box (Å).
BOX_EDGE_ANG = 10.0
#: Muon to nearest-nitrogen distance (Å), as in alpha-N2.
MUON_NITROGEN_DISTANCE_ANG = 1.29
#: N2 bond length (Å).
NITROGEN_BOND_ANG = 1.10
#: Species along the chain; the muon is atom 3 and its nearest nitrogens are 2 and 4.
CHAIN_SPECIES: tuple[str, ...] = ("N", "N", "H:mu", "N", "N")
MUON_ATOM_INDEX = 3
NEAREST_NITROGEN_INDEX = 2


@dataclass(frozen=True, slots=True)
class ChainForceConstants:
    """Spring constants of the [N2-mu-N2]+ toy model, in eV/Å²."""

    muon_nitrogen_stretch: float
    muon_nitrogen_bend: float
    nitrogen_stretch: float
    nitrogen_bend: float
    nitrogen_lattice: float


#: Tuned constants for the worked example (see the module docstring).
N2_MU_N2_FORCE_CONSTANTS = ChainForceConstants(
    muon_nitrogen_stretch=1.612,
    muon_nitrogen_bend=2.227,
    nitrogen_stretch=179.2,
    nitrogen_bend=0.167,
    nitrogen_lattice=0.33,
)

#: Conversion from sqrt(eV / (Å² AMU)) to angular frequency in rad/s.
_OMEGA_UNIT_RAD_S = sqrt(ELEMENTARY_CHARGE_C / (1e-20 * ATOMIC_MASS_KG))


def _chain_masses() -> np.ndarray:
    return np.array(
        [
            FIXTURE_MUON_MASS_AMU if species == "H:mu" else NITROGEN_MASS_AMU
            for species in CHAIN_SPECIES
        ]
    )


def _chain_x_positions() -> tuple[float, ...]:
    centre = BOX_EDGE_ANG / 2.0
    inner = MUON_NITROGEN_DISTANCE_ANG
    outer = inner + NITROGEN_BOND_ANG
    return (centre - outer, centre - inner, centre, centre + inner, centre + outer)


def _block_hessian(muon_nitrogen: float, nitrogen_nitrogen: float, lattice: float) -> np.ndarray:
    """Return one 5 x 5 Cartesian block of the force-constant matrix (eV/Å²)."""

    hessian = np.zeros((5, 5))
    pairs = (
        (0, 1, nitrogen_nitrogen),
        (1, 2, muon_nitrogen),
        (2, 3, muon_nitrogen),
        (3, 4, nitrogen_nitrogen),
    )
    for i, j, k in pairs:
        hessian[i, i] += k
        hessian[j, j] += k
        hessian[i, j] -= k
        hessian[j, i] -= k
    for atom in (0, 1, 3, 4):
        hessian[atom, atom] += lattice
    return hessian


def _block_modes(hessian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return frequencies (cm^-1) and sign-fixed mass-weighted eigenvectors of one block."""

    inverse_root_mass = 1.0 / np.sqrt(_chain_masses())
    dynamical = hessian * np.outer(inverse_root_mass, inverse_root_mass)
    eigenvalues, eigenvectors = np.linalg.eigh(dynamical)
    angular = np.sqrt(np.clip(eigenvalues, 0.0, None)) * _OMEGA_UNIT_RAD_S
    frequencies = angular / (2.0 * pi * LIGHT_SPEED_M_S * 100.0)
    for column in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, column]
        pivot = (
            MUON_ATOM_INDEX - 1
            if abs(vector[MUON_ATOM_INDEX - 1]) > 1e-9
            else int(np.argmax(np.abs(vector)))
        )
        if vector[pivot] < 0.0:
            eigenvectors[:, column] = -vector
    return frequencies, eigenvectors


def n2_mu_n2_modes(
    constants: ChainForceConstants = N2_MU_N2_FORCE_CONSTANTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 15 Gamma modes as ``(frequencies_cm, eigenvectors[mode, atom, xyz])``.

    Modes are in ascending frequency; ties keep the order ``x``, ``y``, ``z``.
    """

    along = _block_modes(
        _block_hessian(
            constants.muon_nitrogen_stretch, constants.nitrogen_stretch, constants.nitrogen_lattice
        )
    )
    across = _block_modes(
        _block_hessian(
            constants.muon_nitrogen_bend, constants.nitrogen_bend, constants.nitrogen_lattice
        )
    )
    frequencies: list[float] = []
    vectors: list[np.ndarray] = []
    for axis, (block_frequencies, block_vectors) in enumerate((along, across, across)):
        for column, frequency in enumerate(block_frequencies):
            mode = np.zeros((5, 3))
            mode[:, axis] = block_vectors[:, column]
            frequencies.append(float(frequency))
            vectors.append(mode)
    order = sorted(range(len(frequencies)), key=lambda index: (round(frequencies[index], 6), index))
    return np.array([frequencies[i] for i in order]), np.array([vectors[i] for i in order])


def n2_mu_n2_phonon_text(constants: ChainForceConstants = N2_MU_N2_FORCE_CONSTANTS) -> str:
    """Return the Gamma-only CASTEP ``.phonon`` text of the [N2-mu-N2]+ toy model."""

    frequencies, eigenvectors = n2_mu_n2_modes(constants)
    masses = _chain_masses()
    lines = [
        " BEGIN header",
        f" Number of ions         {len(CHAIN_SPECIES)}",
        f" Number of branches     {3 * len(CHAIN_SPECIES)}",
        " Number of wavevectors  1",
        " Frequencies in         cm-1",
        " IR intensities in      (D/A)**2/amu",
        " Unit cell vectors (A)",
        f"   {BOX_EDGE_ANG:10.6f}    0.000000    0.000000",
        f"    0.000000   {BOX_EDGE_ANG:10.6f}    0.000000",
        f"    0.000000    0.000000   {BOX_EDGE_ANG:10.6f}",
        " Fractional Co-ordinates",
    ]
    for index, (species, x, mass) in enumerate(
        zip(CHAIN_SPECIES, _chain_x_positions(), masses), start=1
    ):
        lines.append(
            f"   {index:3d}     {x / BOX_EDGE_ANG:.6f}    0.500000    0.500000   {species:<6s} {mass:12.7f}"
        )
    lines.append(" END header")
    lines.append("     q-pt=    1    0.000000  0.000000  0.000000      1.0000000000")
    for mode, frequency in enumerate(frequencies, start=1):
        lines.append(f"     {mode:3d}  {frequency:14.6f}")
    lines.append("                        Phonon Eigenvectors")
    lines.append(
        "Mode Ion                X                                   Y                                   Z"
    )
    for mode, vector in enumerate(eigenvectors, start=1):
        for atom, (x, y, z) in enumerate(vector, start=1):
            lines.append(
                f"{mode:4d}{atom:4d} {x:15.12f} {0.0:15.12f}     {y:15.12f} {0.0:15.12f}     "
                f"{z:15.12f} {0.0:15.12f}"
            )
    return "\n".join(lines) + "\n"
