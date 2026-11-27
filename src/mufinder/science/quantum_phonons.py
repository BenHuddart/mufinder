"""CASTEP phonon parsing and muon quantum-correction analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path
import re
from typing import Callable, Iterable, Sequence

import numpy as np

from mufinder.domain.constants import (
    ATOMIC_MASS_KG,
    BOLTZMANN_J_K,
    ELEMENTARY_CHARGE_C,
    HBAR_J_S,
    LIGHT_SPEED_M_S,
    MUON_MASS_AMU,
    PLANCK_CONSTANT_J_S,
)
from mufinder.domain.models import Structure
from mufinder.science.lattice import nearest_image_distance_matrix


class PhononParseError(ValueError):
    """Raised when a CASTEP ``.phonon`` file cannot be interpreted."""


#: The warning :func:`muon_zpe` raises when a selected mode is left out of the total.
NON_POSITIVE_MUON_MODES_WARNING = "Selected muon-dominated modes include non-positive frequencies."


@dataclass(frozen=True, slots=True)
class PhononAtom:
    """One atom row from the CASTEP phonon header."""

    index: int
    fractional_coordinates: tuple[float, float, float]
    species: str
    mass_amu: float


@dataclass(frozen=True, slots=True)
class PhononQPoint:
    """One CASTEP phonon q-point."""

    number: int
    q_vector: tuple[float, float, float]
    weight: float
    frequencies_cm: tuple[float, ...]
    eigenvectors: np.ndarray
    direction: tuple[float, float, float] | None = None

    @property
    def is_gamma(self) -> bool:
        """Return true when the q-vector is Gamma."""

        return all(abs(component) < 1e-9 for component in self.q_vector)


@dataclass(frozen=True, slots=True)
class PhononCalculation:
    """Parsed CASTEP phonon calculation."""

    atoms: tuple[PhononAtom, ...]
    qpoints: tuple[PhononQPoint, ...]
    lattice_vectors: tuple[tuple[float, float, float], ...]
    frequency_units: str = "cm-1"

    def gamma_point(self) -> PhononQPoint:
        """Return the Gamma q-point, raising if none is present."""

        for qpoint in self.qpoints:
            if qpoint.is_gamma:
                return qpoint
        raise PhononParseError("Gamma-point phonon data were not found.")

    def atom_index_for_species(self, species: str) -> int:
        """Return the one-based CASTEP atom index for a unique species label."""

        matches = [atom.index for atom in self.atoms if atom.species == species]
        if len(matches) != 1:
            raise PhononParseError(
                f"Expected exactly one atom labelled {species!r}; found {len(matches)}."
            )
        return matches[0]


#: Atoms this close in distance from the muon, in Å, count as tied.
_NEIGHBOUR_TIE_ANG = 1e-3


@dataclass(frozen=True, slots=True)
class MuonNeighbour:
    """One atom near the muon: its distance and its correlation with the muon.

    ``correlation`` is the harmonic ``|c|``, ``-w2`` of :func:`w2_entanglement_witness`.
    ``distance_shell`` numbers the atom's distance shell among every atom in the cell,
    nearest first from 0, as :func:`muon_nearest_atoms` forms the shells.
    """

    atom_index: int
    species: str
    distance_ang: float
    correlation: float
    distance_shell: int


@dataclass(frozen=True, slots=True)
class MuonModeContribution:
    """Muon participation, ZPE and zero-point width for one selected phonon mode.

    ``width_A`` is the muon's zero-point position spread along the mode,
    :func:`mode_width_angstrom` of ``frequency_cm``. It is last and defaults to
    ``0.0`` so construction without it keeps working;
    :func:`select_muon_modes` always fills it in.
    """

    mode_index: int
    frequency_cm: float
    participation: float
    zpe_eV: float
    width_A: float = 0.0


@dataclass(frozen=True, slots=True)
class MuonZPEResult:
    """Zero-point-energy summary for the muon-dominated Gamma modes."""

    total_eV: float
    modes: tuple[MuonModeContribution, ...]
    warnings: tuple[str, ...] = ()


#: Relative tolerance for accepting the CASTEP-reported muon mass. CASTEP echoes
#: the ``SPECIES_MASS`` value with limited precision, and runs prepared outside
#: MuFinder commonly use a rounded muon mass such as 0.1138 AMU (0.33 % high,
#: shifting frequencies by 0.16 %). One per cent accepts those while still
#: flagging the failure this guards against, the ~8.9x hydrogen-mass default.
MUON_MASS_RELATIVE_TOLERANCE = 1e-2
CM_INV_TO_EV = PLANCK_CONSTANT_J_S * LIGHT_SPEED_M_S * 100.0 / ELEMENTARY_CHARGE_C
CM_INV_ZPE_TO_EV = 0.5 * CM_INV_TO_EV

_QPOINT_RE = re.compile(
    r"^\s*q-pt=\s*(?P<number>\d+)\s+"
    r"(?P<qx>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
    r"(?P<qy>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
    r"(?P<qz>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
    r"(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
    r"(?P<rest>.*)$"
)


def parse_phonon_path(path: str | Path) -> PhononCalculation:
    """Parse a CASTEP ``.phonon`` file from disk."""

    return parse_phonon_text(Path(path).read_text(encoding="utf-8"))


def parse_phonon_text(text: str) -> PhononCalculation:
    """Parse CASTEP ``.phonon`` text."""

    lines = text.splitlines()
    if not lines or lines[0].strip() != "BEGIN header":
        raise PhononParseError("Expected CASTEP phonon header.")

    header_end = _find_line(lines, "END header", start=1)
    header = lines[1:header_end]
    lattice_vectors, atoms, frequency_units = _parse_header(header)
    qpoints = _parse_qpoints(lines[header_end + 1 :], n_atoms=len(atoms))
    if not qpoints:
        raise PhononParseError("No q-points were found in the phonon file.")
    return PhononCalculation(
        atoms=tuple(atoms),
        qpoints=tuple(qpoints),
        lattice_vectors=tuple(lattice_vectors),
        frequency_units=frequency_units,
    )


def structure_from_phonon_calculation(
    calculation: PhononCalculation,
    *,
    label: str = "Muonated phonon structure",
) -> Structure:
    """Build a MuFinder structure from a parsed CASTEP phonon calculation."""

    return Structure(
        label=label,
        origin="phonon",
        lattice_vectors=[
            [float(value) for value in vector] for vector in calculation.lattice_vectors
        ],
        species=[str(atom.species) for atom in calculation.atoms],
        fractional_positions=[
            [float(value) for value in atom.fractional_coordinates] for atom in calculation.atoms
        ],
        properties={"pbc": [True, True, True]},
    )


def muon_mode_participations(
    calculation: PhononCalculation,
    *,
    muon_species: str = "H:mu",
) -> tuple[float, ...]:
    """Return muon participation for every Gamma-point mode."""

    gamma = calculation.gamma_point()
    muon_index = calculation.atom_index_for_species(muon_species) - 1
    muon_vectors = np.asarray(gamma.eigenvectors[:, muon_index, :], dtype=complex)
    participation = np.sum(np.abs(muon_vectors) ** 2, axis=1).real
    return tuple(float(value) for value in participation.tolist())


def real_mode_displacements(
    calculation: PhononCalculation,
    mode_index: int,
    *,
    muon_species: str = "H:mu",
) -> np.ndarray:
    """Return one Gamma-point mode as deterministic real-valued cartesian vectors."""

    gamma = calculation.gamma_point()
    zero_based_mode_index = int(mode_index) - 1
    if zero_based_mode_index < 0 or zero_based_mode_index >= len(gamma.frequencies_cm):
        raise PhononParseError(
            f"Mode index {mode_index} is not present in the Gamma-point phonon data."
        )

    mode_vectors = np.asarray(gamma.eigenvectors[zero_based_mode_index, :, :], dtype=complex)
    if mode_vectors.shape != (len(calculation.atoms), 3):
        raise PhononParseError(
            f"Expected mode {mode_index} to have shape ({len(calculation.atoms)}, 3); "
            f"found {tuple(mode_vectors.shape)}."
        )

    muon_index = calculation.atom_index_for_species(muon_species) - 1
    muon_vector = np.asarray(mode_vectors[muon_index], dtype=complex).reshape(-1)
    pivot = int(np.argmax(np.abs(muon_vector)))
    pivot_value = complex(muon_vector[pivot]) if muon_vector.size else 0.0j
    if abs(pivot_value) <= 0.0:
        flattened = np.asarray(mode_vectors, dtype=complex).reshape(-1)
        global_pivot = int(np.argmax(np.abs(flattened))) if flattened.size else 0
        pivot_value = complex(flattened[global_pivot]) if flattened.size else 0.0j
    phase = np.exp(-1j * np.angle(pivot_value)) if abs(pivot_value) > 0.0 else 1.0 + 0.0j
    aligned = mode_vectors * phase
    real_vectors = np.real_if_close(aligned, tol=1000)
    if np.iscomplexobj(real_vectors):
        return np.asarray(np.real(real_vectors), dtype=float)
    return np.asarray(real_vectors, dtype=float)


def scaled_mode_displacements(
    calculation: PhononCalculation,
    mode_index: int,
    amplitude_ang: float,
    *,
    muon_species: str = "H:mu",
) -> np.ndarray:
    """Scale a mode so its largest atomic displacement norm equals ``amplitude_ang`` in Å."""

    try:
        amplitude = float(amplitude_ang)
    except (TypeError, ValueError) as exc:
        raise ValueError("Amplitude must be a finite number.") from exc
    if not np.isfinite(amplitude):
        raise ValueError("Amplitude must be finite.")
    if amplitude < 0.0:
        raise ValueError("Amplitude must be non-negative.")

    displacements = real_mode_displacements(calculation, mode_index, muon_species=muon_species)
    norms = np.linalg.norm(displacements, axis=1)
    maximum = float(np.max(norms)) if norms.size else 0.0
    if maximum <= 0.0:
        return np.zeros_like(displacements, dtype=float)
    return np.asarray(displacements, dtype=float) * (amplitude / maximum)


def select_muon_modes(
    calculation: PhononCalculation,
    *,
    muon_species: str = "H:mu",
    count: int = 3,
) -> tuple[MuonModeContribution, ...]:
    """Select the Gamma modes with the largest muon participation."""

    if count <= 0:
        raise ValueError("At least one muon mode must be selected.")
    gamma = calculation.gamma_point()
    participation = np.asarray(
        muon_mode_participations(calculation, muon_species=muon_species), dtype=float
    )
    selected = np.argsort(participation)[-count:]
    frequencies = np.asarray(gamma.frequencies_cm, dtype=float)
    selected = selected[np.argsort(frequencies[selected])]
    return tuple(
        MuonModeContribution(
            mode_index=int(mode + 1),
            frequency_cm=float(gamma.frequencies_cm[mode]),
            participation=float(participation[mode]),
            zpe_eV=zpe_from_frequency_cm(float(gamma.frequencies_cm[mode])),
            width_A=mode_width_angstrom(float(gamma.frequencies_cm[mode])),
        )
        for mode in selected
    )


def muon_mass_warning(
    calculation: PhononCalculation,
    *,
    muon_species: str = "H:mu",
) -> str | None:
    """Return a warning when CASTEP did not use the muon mass for the muon.

    CASTEP writes the mass it actually used into the ``.phonon`` header, so this is
    the authoritative check on a finished run. The mass is *not* something MuFinder
    can correct after the fact: omega scales as 1/sqrt(mass), so a phonon run that
    used the default hydrogen mass (about 8.9 times the muon mass) yields
    frequencies and a ZPE roughly a factor of three too low, and displacement
    amplitudes wrong by the same square-root factor.

    A fresh phonon run reads the muon mass from its own ``.cell``. The run can
    still end up with another mass when the row's settings override
    ``SPECIES_MASS``, or when it continued from a checkpoint written without the
    muon mass: CASTEP restores species masses from the checkpoint on a
    continuation. Such a run has to be repeated from a clean directory.

    Returns ``None`` when the mass matches, and also when the muon cannot be
    identified in the header — callers that need the muon raise their own
    :class:`PhononParseError` for that case, and a warning helper must not.
    """

    try:
        muon = _atom_by_index(calculation, calculation.atom_index_for_species(muon_species))
    except PhononParseError:
        return None
    reported = float(muon.mass_amu)
    if abs(reported - MUON_MASS_AMU) <= MUON_MASS_RELATIVE_TOLERANCE * MUON_MASS_AMU:
        return None
    return (
        f"CASTEP used {reported:.6g} AMU for {muon_species}, not the muon mass "
        f"{MUON_MASS_AMU:.6g} AMU. Frequencies scale as 1/sqrt(mass), so this run's "
        "ZPE, mode frequencies, and displacement amplitudes are wrong. Either the "
        "row's settings override SPECIES_MASS, or the run continued from a checkpoint "
        "written without the muon mass: delete the job and its files from the Job "
        "Dashboard and run the row again."
    )


def muon_zpe(
    calculation: PhononCalculation,
    *,
    muon_species: str = "H:mu",
    count: int = 3,
) -> MuonZPEResult:
    """Calculate the muon ZPE from the selected Gamma modes."""

    warnings: list[str] = []
    mass_warning = muon_mass_warning(calculation, muon_species=muon_species)
    if mass_warning is not None:
        warnings.append(mass_warning)
    modes = select_muon_modes(calculation, muon_species=muon_species, count=count)
    unstable = [mode for mode in modes if mode.frequency_cm <= 0.0]
    if unstable:
        warnings.append(NON_POSITIVE_MUON_MODES_WARNING)
    total = sum(mode.zpe_eV for mode in modes if mode.frequency_cm > 0.0)
    return MuonZPEResult(total_eV=float(total), modes=modes, warnings=tuple(warnings))


#: How far, as an absolute deviation from the identity, the muon rows of the stored
#: Gamma eigenvectors may stray from completeness, ``sum_n e_mu,n e_mu,n^dagger = 1``.
#: CASTEP writes twelve decimals per component; the GdRu2Si2 runs are complete within 1e-9.
_EIGENVECTOR_COMPLETENESS_TOLERANCE = 1e-3

#: Why :func:`muon_zpe_along` gives no value along a direction of negative curvature.
NO_REAL_FREQUENCY_ALONG_DIRECTION = (
    "The muon's curvature along this direction is not positive, so it has no real "
    "frequency along it."
)


@dataclass(frozen=True, slots=True)
class DirectionalZPE:
    """The muon's zero-point energy along one direction, or why there is none.

    ``zpe_eV`` and ``frequency_cm`` are ``None`` together, and ``reason`` then says
    why; with a value ``reason`` is empty.
    """

    zpe_eV: float | None
    frequency_cm: float | None
    reason: str = ""


def muon_dynamical_block_cm2(
    calculation: PhononCalculation,
    *,
    muon_species: str = "H:mu",
) -> np.ndarray:
    """Return the muon's 3 x 3 block D(0) of the Gamma-point dynamical matrix, in cm^-2.

    CASTEP's ``.phonon`` eigenvectors ``e_n`` are the orthonormal eigenvectors of
    the mass-weighted dynamical matrix ``D = M^-1/2 Phi M^-1/2``: each mode is
    normalised to 1 over every atom and Cartesian component (it carries no
    ``1/sqrt(m)`` factor), and its components are Cartesian in the frame of the
    file's ``Unit cell vectors``. A frequency ``nu_n`` in cm^-1 is the square root
    of the eigenvalue, and CASTEP writes an imaginary one as negative. The matrix
    is then ``D = sum_n s_n nu_n^2 e_n e_n^dagger`` with ``s_n`` the sign of
    ``nu_n``, and its muon block is

        D(0) = sum_n s_n nu_n^2 e_mu,n e_mu,n^dagger,

    summed over every mode, imaginary ones included, so the block is exact. As
    ``D(0) = Phi_mumu / m_mu``, ``v^T D(0) v`` for a unit ``v`` is the squared
    frequency of the muon moving alone along ``v`` with every other atom held
    fixed. The definition follows B. M. Huddart et al., Phys. Rev. B 111, 054440
    (2025), Supplemental Material, Eq. 3.

    Raises :class:`PhononParseError` when the muon rows of the eigenvectors are not
    complete (``sum_n e_mu,n e_mu,n^dagger`` is not the identity), which is what
    the formula needs and what a file of fewer modes or other normalisation lacks.
    """

    gamma = calculation.gamma_point()
    muon_index = calculation.atom_index_for_species(muon_species) - 1
    vectors = np.asarray(gamma.eigenvectors[:, muon_index, :], dtype=complex)
    frequencies = np.asarray(gamma.frequencies_cm, dtype=float)
    completeness = np.einsum("na,nb->ab", vectors, np.conjugate(vectors))
    if not np.allclose(completeness, np.eye(3), rtol=0.0, atol=_EIGENVECTOR_COMPLETENESS_TOLERANCE):
        raise PhononParseError(
            "The muon's Gamma-point eigenvectors are not a complete orthonormal set, so the "
            "muon block of the dynamical matrix cannot be built from them."
        )
    weights = np.sign(frequencies) * frequencies**2
    block = np.einsum("n,na,nb->ab", weights, vectors, np.conjugate(vectors))
    # At Gamma the dynamical matrix is real and symmetric; the sum is Hermitian.
    real = np.asarray(block.real, dtype=float)
    return 0.5 * (real + real.T)


def muon_zpe_along(
    calculation: PhononCalculation,
    direction: Sequence[float],
    *,
    muon_species: str = "H:mu",
) -> DirectionalZPE:
    """Return the muon's one-dimensional zero-point energy along ``direction``.

    ``direction`` is a Cartesian vector in the frame of the calculation's cell, of
    any non-zero length; it is normalised to ``v``. The frequency along it is
    ``nu_v = sqrt(v^T D(0) v)`` with ``D(0)`` from :func:`muon_dynamical_block_cm2`,
    and the zero-point energy is ``h c nu_v / 2`` (``hbar omega_v / 2``), in eV.
    Where ``v^T D(0) v`` is not positive there is no real frequency, and the
    result holds no value and :data:`NO_REAL_FREQUENCY_ALONG_DIRECTION`.
    Raises ``ValueError`` for a zero or non-finite direction and
    :class:`PhononParseError` as :func:`muon_dynamical_block_cm2` does.
    """

    vector = np.asarray(direction, dtype=float).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("A direction needs three finite Cartesian components.")
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise ValueError("A direction must not be the zero vector.")
    unit = vector / length
    curvature = float(
        unit @ muon_dynamical_block_cm2(calculation, muon_species=muon_species) @ unit
    )
    if not curvature > 0.0:
        return DirectionalZPE(None, None, NO_REAL_FREQUENCY_ALONG_DIRECTION)
    frequency = sqrt(curvature)
    return DirectionalZPE(zpe_from_frequency_cm(frequency), frequency)


#: Two eigenvalues of D(0) within this fraction of the largest magnitude are one
#: degenerate level: 1e-3 in nu^2 is 0.05 % in frequency, well under a meV at muon ZPEs.
PRINCIPAL_DEGENERACY_RELATIVE_TOLERANCE = 1e-3

#: Why :func:`muon_principal_zpes` gives no value along an axis of negative curvature.
NO_REAL_FREQUENCY_ALONG_AXIS = (
    "The muon's curvature along this principal axis is not positive, so it has no real "
    "frequency along it."
)


@dataclass(frozen=True, slots=True)
class PrincipalZPE:
    """One principal axis of the muon's well and its zero-point energy along it.

    ``curvature_cm2`` is the eigenvalue of D(0), ``nu^2`` in cm^-2. ``zpe_eV`` and
    ``frequency_cm`` are ``None`` when it is not positive, and ``reason`` then says
    why. ``axis`` is the unit eigenvector, Cartesian in the calculation's frame, with
    its first component larger than 1e-3 in size made positive (an axis has no
    sense). ``degenerate_with`` lists the positions, in the tuple
    :func:`muon_principal_zpes` returns, of the other axes of the same level; such an
    axis is any direction in the plane (or space) they span, so ``axis`` is then
    ``None`` rather than an arbitrary choice.
    """

    curvature_cm2: float
    zpe_eV: float | None
    frequency_cm: float | None
    axis: tuple[float, float, float] | None
    degenerate_with: tuple[int, ...] = ()
    reason: str = ""


def muon_principal_zpes(
    calculation: PhononCalculation,
    *,
    muon_species: str = "H:mu",
) -> tuple[PrincipalZPE, PrincipalZPE, PrincipalZPE]:
    """Return the three principal axes of D(0) and the muon's ZPE along each, ascending.

    The axes are the eigenvectors of :func:`muon_dynamical_block_cm2`, and the ZPE
    along each is ``h c sqrt(lambda) / 2`` for its eigenvalue ``lambda`` (cm^-2), the
    value :func:`muon_zpe_along` gives along that axis. They are ordered by
    eigenvalue, lowest first, so by ZPE where every one is positive. Consecutive
    eigenvalues of one sign within :data:`PRINCIPAL_DEGENERACY_RELATIVE_TOLERANCE` (of
    the largest magnitude) of each other form one degenerate level, chained so that a
    level holds every axis linked to it (see :class:`PrincipalZPE`). A non-positive eigenvalue gives no
    ZPE and :data:`NO_REAL_FREQUENCY_ALONG_AXIS`.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(
        muon_dynamical_block_cm2(calculation, muon_species=muon_species)
    )
    scale = max(float(np.max(np.abs(eigenvalues))), 1e-300)
    # Ascending eigenvalues chain into levels: each joins the previous one's level when
    # it is within the tolerance of it and of the same sign, so a level is transitive.
    level = [0, 0, 0]
    for index in (1, 2):
        joined = abs(float(eigenvalues[index] - eigenvalues[index - 1])) <= (
            PRINCIPAL_DEGENERACY_RELATIVE_TOLERANCE * scale
        ) and (eigenvalues[index] > 0.0) == (eigenvalues[index - 1] > 0.0)
        level[index] = level[index - 1] if joined else level[index - 1] + 1
    principal: list[PrincipalZPE] = []
    for index, value in enumerate(eigenvalues):
        others = tuple(
            other for other in range(3) if other != index and level[other] == level[index]
        )
        axis = np.asarray(eigenvectors[:, index], dtype=float)
        leading = next((component for component in axis if abs(component) > 1e-3), 1.0)
        axis = axis if leading > 0.0 else -axis
        frequency = sqrt(float(value)) if value > 0.0 else None
        principal.append(
            PrincipalZPE(
                curvature_cm2=float(value),
                zpe_eV=zpe_from_frequency_cm(frequency) if frequency is not None else None,
                frequency_cm=frequency,
                axis=None if others else (float(axis[0]), float(axis[1]), float(axis[2])),
                degenerate_with=others,
                reason="" if frequency is not None else NO_REAL_FREQUENCY_ALONG_AXIS,
            )
        )
    return principal[0], principal[1], principal[2]


def zpe_from_frequency_cm(frequency_cm: float) -> float:
    """Return ``1/2 h c nu`` in eV for a CASTEP frequency in cm^-1."""

    if frequency_cm <= 0.0:
        return 0.0
    return float(CM_INV_ZPE_TO_EV * frequency_cm)


def mode_width_angstrom(frequency_cm: float, mass_amu: float = MUON_MASS_AMU) -> float:
    """Return the zero-point width ``sqrt(hbar / (2 m omega))`` of one mode in Å.

    This is the root-mean-square ground-state displacement of a harmonic
    oscillator of mass ``mass_amu`` (the muon by default) and angular frequency
    ``omega = 2 pi c nu`` for a CASTEP frequency ``nu`` in cm^-1. A mode with a
    muon ZPE of 0.17 eV (about 2742 cm^-1) gives about 0.23 Å for the muon.
    A non-positive frequency has no zero-point width and returns ``0.0``.
    """

    if frequency_cm <= 0.0 or mass_amu <= 0.0:
        return 0.0
    angular = 2.0 * pi * LIGHT_SPEED_M_S * 100.0 * float(frequency_cm)
    width_m = sqrt(HBAR_J_S / (2.0 * ATOMIC_MASS_KG * float(mass_amu) * angular))
    return float(width_m * 1e10)


def covariance_matrix(
    calculation: PhononCalculation,
    atom_a_index: int,
    atom_b_index: int,
    *,
    temperature_K: float = 0.0,
) -> np.ndarray:
    """Return the 3x3 displacement covariance tensor, in Å², for two one-based atoms."""

    atom_a = _atom_by_index(calculation, atom_a_index)
    atom_b = _atom_by_index(calculation, atom_b_index)
    covariance = np.zeros((3, 3), dtype=complex)
    for qpoint in calculation.qpoints:
        vec_a = _weighted_displacement_vectors(qpoint, atom_a, atom_a_index - 1, temperature_K)
        vec_b = _weighted_displacement_vectors(qpoint, atom_b, atom_b_index - 1, temperature_K)
        covariance += qpoint.weight * np.einsum("ma,mb->ab", vec_a, np.conjugate(vec_b))
    return covariance.real


def correlation_matrix(
    calculation: PhononCalculation,
    atom_a_index: int,
    atom_b_index: int,
    *,
    temperature_K: float = 0.0,
) -> np.ndarray:
    """Return the normalized 3x3 covariance tensor for two one-based atoms."""

    cov_ab = covariance_matrix(calculation, atom_a_index, atom_b_index, temperature_K=temperature_K)
    cov_aa = covariance_matrix(calculation, atom_a_index, atom_a_index, temperature_K=temperature_K)
    cov_bb = covariance_matrix(calculation, atom_b_index, atom_b_index, temperature_K=temperature_K)
    denom = np.sqrt(np.outer(np.diag(cov_aa), np.diag(cov_bb)))
    return np.divide(cov_ab, denom, out=np.zeros_like(cov_ab), where=np.abs(denom) > 0.0)


def w1_entanglement_witness(calculation: PhononCalculation, *, muon_species: str = "H:mu") -> float:
    """Return the ``w1`` muon-host entanglement witness, ``P - 3``.

    Follows M. Gomilšek et al., Commun. Phys. 6, 142 (2023),
    doi:10.1038/s42005-023-01260-7: the three Gamma modes with the largest muon
    participation (the modes :func:`select_muon_modes` picks) are projected onto
    pure muon motion, and ``P`` is the sum of the squared norms of the projected
    (normalised) eigenvectors, each the mode's muon participation. ``w1`` is
    ``0`` when the muon's motion is fully contained in those three modes and
    becomes more negative as it mixes with host modes, down to ``-3``.
    """

    modes = select_muon_modes(calculation, muon_species=muon_species, count=3)
    return float(sum(mode.participation for mode in modes) - 3.0)


def w2_entanglement_witness(
    calculation: PhononCalculation, atom_a_index: int, atom_b_index: int
) -> float:
    """Return the legacy trace-normalized ``w2`` covariance witness."""

    cov_ab = np.trace(covariance_matrix(calculation, atom_a_index, atom_b_index))
    cov_aa = np.trace(covariance_matrix(calculation, atom_a_index, atom_a_index))
    cov_bb = np.trace(covariance_matrix(calculation, atom_b_index, atom_b_index))
    denom = np.sqrt(cov_aa * cov_bb)
    if abs(denom) <= 0.0:
        return 0.0
    return float(-abs(cov_ab / denom))


def muon_nearest_atoms(
    calculation: PhononCalculation, count: int, *, muon_species: str = "H:mu"
) -> tuple[MuonNeighbour, ...]:
    """Return the ``count`` atoms nearest the muon, closest first.

    Distances are to the nearest periodic image. Atoms in a run of distances each
    within :data:`_NEIGHBOUR_TIE_ANG` of the last are tied, and tied atoms go in CASTEP
    index order, so the order of a symmetry shell does not follow rounding noise in
    the file's coordinates. The list holds exactly ``count`` atoms, or every other
    atom when there are fewer, even if that divides a shell. Each ``correlation`` is
    ``-w2`` from :func:`w2_entanglement_witness`.
    """

    if count < 1:
        raise ValueError(f"The nearest-atom count must be positive, got {count}.")
    muon = _atom_by_index(calculation, calculation.atom_index_for_species(muon_species))
    others = [atom for atom in calculation.atoms if atom.index != muon.index]
    if not others:
        return ()
    distances = nearest_image_distance_matrix(
        [muon.fractional_coordinates],
        [atom.fractional_coordinates for atom in others],
        calculation.lattice_vectors,
    )[0]
    by_distance = sorted(
        zip(others, (float(distance) for distance in distances)), key=lambda pair: pair[1]
    )
    shells: list[list[tuple[PhononAtom, float]]] = []
    for pair in by_distance:
        if shells and pair[1] - shells[-1][-1][1] < _NEIGHBOUR_TIE_ANG:
            shells[-1].append(pair)
        else:
            shells.append([pair])
    ranked = [
        (atom, distance, shell_number)
        for shell_number, shell in enumerate(shells)
        for atom, distance in sorted(shell, key=lambda member: member[0].index)
    ]
    return tuple(
        MuonNeighbour(
            atom_index=atom.index,
            species=atom.species,
            distance_ang=distance,
            correlation=abs(w2_entanglement_witness(calculation, muon.index, atom.index)),
            distance_shell=shell_number,
        )
        for atom, distance, shell_number in ranked[:count]
    )


#: The columns :func:`order_muon_neighbours` can order by.
MUON_NEIGHBOUR_SORT_KEYS = ("atom", "species", "distance", "correlation")


def order_muon_neighbours(
    neighbours: Iterable[MuonNeighbour], key: str, *, descending: bool = False
) -> tuple[MuonNeighbour, ...]:
    """Return ``neighbours`` ordered by ``key``, one of :data:`MUON_NEIGHBOUR_SORT_KEYS`.

    This orders rows only and changes no value. ``atom`` and ``correlation`` order
    by number and ``species`` by name, ignoring case. ``distance`` orders by
    ``distance_shell``, the shells :func:`muon_nearest_atoms` formed over every atom,
    so ascending is that function's order whatever the count. Ties keep CASTEP index
    order in both directions, and a missing (NaN) ``correlation`` goes last in both.
    """

    if key not in MUON_NEIGHBOUR_SORT_KEYS:
        raise ValueError(f"Cannot order the muon's neighbours by {key!r}.")
    by_index = sorted(neighbours, key=lambda neighbour: neighbour.atom_index)
    if key == "correlation":
        present = [n for n in by_index if not np.isnan(n.correlation)]
        missing = [n for n in by_index if np.isnan(n.correlation)]
    else:
        present, missing = by_index, []
    sort_key: dict[str, Callable[[MuonNeighbour], object]] = {
        "atom": lambda neighbour: neighbour.atom_index,
        "species": lambda neighbour: neighbour.species.casefold(),
        "distance": lambda neighbour: neighbour.distance_shell,
        "correlation": lambda neighbour: neighbour.correlation,
    }
    # ``sorted`` is stable with ``reverse=True`` too, so ties stay in index order.
    return (*sorted(present, key=sort_key[key], reverse=descending), *missing)


def _find_line(lines: list[str], marker: str, *, start: int = 0) -> int:
    for index in range(start, len(lines)):
        if lines[index].strip() == marker:
            return index
    raise PhononParseError(f"Expected {marker!r}.")


def _parse_header(
    lines: Iterable[str],
) -> tuple[list[tuple[float, float, float]], list[PhononAtom], str]:
    header = list(lines)
    lattice_vectors: list[tuple[float, float, float]] = []
    atoms: list[PhononAtom] = []
    frequency_units = "cm-1"
    index = 0
    while index < len(header):
        stripped = header[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("Frequencies in"):
            parts = stripped.split()
            frequency_units = parts[-1] if parts else "cm-1"
            index += 1
            continue
        if stripped == "Unit cell vectors (A)":
            lattice_vectors = [
                _parse_float_triplet(header[index + offset].split(), "unit cell vector")
                for offset in (1, 2, 3)
            ]
            index += 4
            continue
        if stripped == "Fractional Co-ordinates":
            index += 1
            while index < len(header):
                parts = header[index].split()
                if len(parts) < 6:
                    break
                atoms.append(
                    PhononAtom(
                        index=int(parts[0]),
                        fractional_coordinates=(float(parts[1]), float(parts[2]), float(parts[3])),
                        species=parts[4],
                        mass_amu=float(parts[5]),
                    )
                )
                index += 1
            continue
        index += 1
    if not lattice_vectors:
        raise PhononParseError("Unit cell vectors were not found in the phonon header.")
    if not atoms:
        raise PhononParseError("Fractional coordinates were not found in the phonon header.")
    if frequency_units != "cm-1":
        raise PhononParseError(f"Unsupported phonon frequency units: {frequency_units!r}.")
    return lattice_vectors, atoms, frequency_units


def _parse_qpoints(lines: list[str], *, n_atoms: int) -> list[PhononQPoint]:
    qpoints: list[PhononQPoint] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _QPOINT_RE.match(line)
        if match is None:
            index += 1
            continue
        q_number = int(match.group("number"))
        q_vector = (float(match.group("qx")), float(match.group("qy")), float(match.group("qz")))
        weight = float(match.group("weight"))
        direction_values = match.group("rest").split()
        direction = (
            tuple(float(value) for value in direction_values[:3])
            if len(direction_values) >= 3
            else None
        )
        index += 1
        frequencies: list[float] = []
        while index < len(lines) and "Phonon Eigenvectors" not in lines[index]:
            parts = lines[index].split()
            if len(parts) >= 2 and parts[0].lstrip("+-").isdigit():
                frequencies.append(float(parts[1]))
            index += 1
        if index >= len(lines):
            raise PhononParseError(f"Missing eigenvector block for q-point {q_number}.")
        index += 1
        if index < len(lines) and lines[index].strip().startswith("Mode"):
            index += 1
        eigenvectors = np.zeros((len(frequencies), n_atoms, 3), dtype=complex)
        rows_seen = 0
        while index < len(lines) and not lines[index].strip().startswith("q-pt="):
            parts = lines[index].split()
            if (
                len(parts) >= 8
                and parts[0].lstrip("+-").isdigit()
                and parts[1].lstrip("+-").isdigit()
            ):
                mode = int(parts[0]) - 1
                atom = int(parts[1]) - 1
                if 0 <= mode < len(frequencies) and 0 <= atom < n_atoms:
                    eigenvectors[mode, atom, :] = (
                        complex(float(parts[2]), float(parts[3])),
                        complex(float(parts[4]), float(parts[5])),
                        complex(float(parts[6]), float(parts[7])),
                    )
                    rows_seen += 1
            index += 1
        if rows_seen != len(frequencies) * n_atoms:
            raise PhononParseError(
                f"Expected {len(frequencies) * n_atoms} eigenvector rows for q-point "
                f"{q_number}; found {rows_seen}."
            )
        qpoints.append(
            PhononQPoint(
                number=q_number,
                q_vector=q_vector,
                weight=weight,
                frequencies_cm=tuple(frequencies),
                eigenvectors=eigenvectors,
                direction=direction,  # type: ignore[arg-type]
            )
        )
    return qpoints


def _parse_float_triplet(parts: list[str], label: str) -> tuple[float, float, float]:
    if len(parts) < 3:
        raise PhononParseError(f"Expected three values for {label}.")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _atom_by_index(calculation: PhononCalculation, atom_index: int) -> PhononAtom:
    for atom in calculation.atoms:
        if atom.index == atom_index:
            return atom
    raise PhononParseError(f"Atom index {atom_index} is not present in the phonon file.")


def _weighted_displacement_vectors(
    qpoint: PhononQPoint,
    atom: PhononAtom,
    zero_based_atom_index: int,
    temperature_K: float,
) -> np.ndarray:
    frequencies = np.asarray(qpoint.frequencies_cm, dtype=float)
    vectors = np.array(qpoint.eigenvectors[:, zero_based_atom_index, :], dtype=complex)
    positive = frequencies > 0.0
    if not np.any(positive):
        return np.zeros_like(vectors)
    angular = 2.0 * pi * LIGHT_SPEED_M_S * 100.0 * frequencies
    factor = np.zeros_like(frequencies, dtype=complex)
    # The zero-point amplitude sqrt(hbar / (2 m omega)) of each mode, in Å.
    factor[positive] = 1e10 * np.sqrt(
        HBAR_J_S / (2.0 * ATOMIC_MASS_KG * atom.mass_amu * angular[positive])
    )
    if temperature_K > 0.0:
        thermal = np.ones_like(frequencies, dtype=float)
        argument = HBAR_J_S * angular[positive] / (2.0 * BOLTZMANN_J_K * temperature_K)
        thermal[positive] = np.sqrt(1.0 / np.tanh(argument))
        factor *= thermal
    phases = np.exp(
        1j * np.dot(np.asarray(qpoint.q_vector), np.asarray(atom.fractional_coordinates))
    )
    return vectors * (phases * factor)[:, None]
