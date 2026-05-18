"""CASTEP phonon parsing and muon quantum-correction analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from mufinder.domain.models import Structure


class PhononParseError(ValueError):
    """Raised when a CASTEP ``.phonon`` file cannot be interpreted."""


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


@dataclass(frozen=True, slots=True)
class MuonModeContribution:
    """Muon participation and ZPE for one selected phonon mode."""

    mode_index: int
    frequency_cm: float
    participation: float
    zpe_eV: float


@dataclass(frozen=True, slots=True)
class MuonZPEResult:
    """Zero-point-energy summary for the muon-dominated Gamma modes."""

    total_eV: float
    modes: tuple[MuonModeContribution, ...]
    warnings: tuple[str, ...] = ()


PLANCK_CONSTANT_J_S = 6.62607015e-34
LIGHT_SPEED_M_S = 299792458.0
ELEMENTARY_CHARGE_C = 1.602176634e-19
HBAR_J_S = PLANCK_CONSTANT_J_S / (2.0 * pi)
ATOMIC_MASS_KG = 1.66053906660e-27
BOLTZMANN_J_K = 1.380649e-23
CM_INV_TO_EV = PLANCK_CONSTANT_J_S * LIGHT_SPEED_M_S * 100.0 / ELEMENTARY_CHARGE_C
CM_INV_ZPE_TO_EV = 0.5 * CM_INV_TO_EV
MUON_MASS_AMU = 0.11342892566

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
        lattice_vectors=[[float(value) for value in vector] for vector in calculation.lattice_vectors],
        species=[str(atom.species) for atom in calculation.atoms],
        fractional_positions=[
            [float(value) for value in atom.fractional_coordinates]
            for atom in calculation.atoms
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
        raise PhononParseError(f"Mode index {mode_index} is not present in the Gamma-point phonon data.")

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
    participation = np.asarray(muon_mode_participations(calculation, muon_species=muon_species), dtype=float)
    selected = np.argsort(participation)[-count:]
    frequencies = np.asarray(gamma.frequencies_cm, dtype=float)
    selected = selected[np.argsort(frequencies[selected])]
    return tuple(
        MuonModeContribution(
            mode_index=int(mode + 1),
            frequency_cm=float(gamma.frequencies_cm[mode]),
            participation=float(participation[mode]),
            zpe_eV=zpe_from_frequency_cm(float(gamma.frequencies_cm[mode])),
        )
        for mode in selected
    )


def muon_zpe(
    calculation: PhononCalculation,
    *,
    muon_species: str = "H:mu",
    count: int = 3,
) -> MuonZPEResult:
    """Calculate the muon ZPE from the selected Gamma modes."""

    warnings: list[str] = []
    modes = select_muon_modes(calculation, muon_species=muon_species, count=count)
    unstable = [mode for mode in modes if mode.frequency_cm <= 0.0]
    if unstable:
        warnings.append("Selected muon-dominated modes include non-positive frequencies.")
    total = sum(mode.zpe_eV for mode in modes if mode.frequency_cm > 0.0)
    return MuonZPEResult(total_eV=float(total), modes=modes, warnings=tuple(warnings))


def zpe_from_frequency_cm(frequency_cm: float) -> float:
    """Return ``1/2 h c nu`` in eV for a CASTEP frequency in cm^-1."""

    if frequency_cm <= 0.0:
        return 0.0
    return float(CM_INV_ZPE_TO_EV * frequency_cm)


def covariance_matrix(
    calculation: PhononCalculation,
    atom_a_index: int,
    atom_b_index: int,
    *,
    temperature_K: float = 0.0,
) -> np.ndarray:
    """Return the 3x3 displacement covariance tensor for two one-based atoms."""

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
    """Return the legacy ``w1`` witness based on Gamma muon-mode localization."""

    gamma = calculation.gamma_point()
    muon_index = calculation.atom_index_for_species(muon_species) - 1
    participation = np.sum(np.abs(gamma.eigenvectors[:, muon_index, :]) ** 2, axis=1).real
    top_three = float(np.sort(participation)[-3:].sum())
    return -sqrt(max(0.0, 3.0 - top_three))


def w2_entanglement_witness(calculation: PhononCalculation, atom_a_index: int, atom_b_index: int) -> float:
    """Return the legacy trace-normalized ``w2`` covariance witness."""

    cov_ab = np.trace(covariance_matrix(calculation, atom_a_index, atom_b_index))
    cov_aa = np.trace(covariance_matrix(calculation, atom_a_index, atom_a_index))
    cov_bb = np.trace(covariance_matrix(calculation, atom_b_index, atom_b_index))
    denom = np.sqrt(cov_aa * cov_bb)
    if abs(denom) <= 0.0:
        return 0.0
    return float(-abs(cov_ab / denom))


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
            if len(parts) >= 8 and parts[0].lstrip("+-").isdigit() and parts[1].lstrip("+-").isdigit():
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
    factor[positive] = np.sqrt(HBAR_J_S / (2.0 * ATOMIC_MASS_KG * atom.mass_amu * angular[positive]))
    if temperature_K > 0.0:
        thermal = np.ones_like(frequencies, dtype=float)
        argument = HBAR_J_S * angular[positive] / (2.0 * BOLTZMANN_J_K * temperature_K)
        thermal[positive] = np.sqrt(1.0 / np.tanh(argument))
        factor *= thermal
    phases = np.exp(1j * np.dot(np.asarray(qpoint.q_vector), np.asarray(atom.fractional_coordinates)))
    return vectors * (phases * factor)[:, None]
