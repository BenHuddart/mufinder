"""CASTEP output files written for the screenshot projects, in CASTEP's file shapes.

No screenshot runs CASTEP. A project whose page shows a finished calculation
gets its outputs from these writers, or from a repository fixture where one
fits. Each writer produces the lines MuFinder's parsers read and nothing
more; the numbers are whatever the caller passes, so a page that shows them
says in its caption that they are synthetic.

This module imports neither Qt nor VTK, so tests import it directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
from typing import Any

import numpy as np


_RUN_STARTED = " Run started: Mon, 12 Oct 2026 09:14:02 +0100\n"


def cell_text(lattice: Any, species: Any, fractional: Any) -> str:
    """A minimal ``-out.cell``: the lattice and fractional positions CASTEP writes."""

    rows = "\n".join(f"   {' '.join(f'{value:17.10f}' for value in vector)}" for vector in lattice)
    atoms = "\n".join(
        f"   {symbol:<5} {' '.join(f'{float(value):17.10f}' for value in position)}"
        for symbol, position in zip(species, fractional, strict=True)
    )
    return (
        f"%BLOCK LATTICE_CART\n   ANG\n{rows}\n%ENDBLOCK LATTICE_CART\n\n"
        f"%BLOCK POSITIONS_FRAC\n{atoms}\n%ENDBLOCK POSITIONS_FRAC\n"
    )


def final_configuration(castep: Path) -> tuple[list[list[float]], list[str], list[list[float]]]:
    """The lattice and atoms of a geometry optimisation's ``Final Configuration`` block."""

    lines = castep.read_text(encoding="utf-8").splitlines()
    start = max(index for index, line in enumerate(lines) if "Final Configuration" in line)
    lattice_row = (
        next(index for index in range(start, len(lines)) if "Real Lattice" in lines[index]) + 1
    )
    lattice = [[float(value) for value in lines[lattice_row + row].split()[:3]] for row in range(3)]
    atom = re.compile(r"^\s*x\s+([A-Za-z][\w:]*)\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\s+x\s*$")
    species: list[str] = []
    fractional: list[list[float]] = []
    for line in lines[lattice_row:]:
        match = atom.match(line)
        if match:
            species.append(match.group(1))
            fractional.append([float(match.group(index)) for index in (2, 3, 4)])
        elif species and "xxxx" in line:
            break
    return lattice, species, fractional


def castep_text(*, energy: float, species: list[str], geometry: bool, wall_time: float) -> str:
    """A CASTEP output with the lines MuFinder reads: final energy, forces and timing."""

    forces = "\n".join(
        f" * {symbol:<6} {index:4d} {0.004 * np.sin(1.7 * index):12.5f} {0.003 * np.cos(2.3 * index):12.5f}"
        f" {-0.002 * np.sin(0.9 * index):12.5f} *"
        for index, symbol in enumerate(species, start=1)
    )
    converged = " BFGS: Geometry optimization completed successfully.\n\n" if geometry else ""
    return (
        f"{_RUN_STARTED}\n"
        f" Final energy, E             =  {energy:.10f}     eV\n\n"
        " ************************** Forces **************************\n"
        " *               Cartesian components (eV/A)                *\n"
        f"{forces}\n"
        " ************************************************************\n\n"
        f"{converged}"
        f" Total time          =    {wall_time:.2f} s\n"
    )


def _castep_float(value: float, digits: int) -> str:
    """``value`` as CASTEP's geometry tables print it, with a three-digit exponent."""

    mantissa, exponent = f"{value:.{digits}E}".split("E")
    return f"{mantissa}E{int(exponent):+04d}"


def _forces_block(species: Sequence[str], magnitude: float) -> list[str]:
    """A forces block whose largest atomic force has length ``magnitude``.

    The first atom carries the largest force; the others carry less, along
    fixed directions, so the block is the same on every build.
    """

    rows = []
    for index, symbol in enumerate(species, start=1):
        direction = np.array([np.sin(1.7 * index), np.cos(2.3 * index), -np.sin(0.9 * index)])
        weight = 1.0 if index == 1 else 0.4 + 0.5 * abs(np.sin(1.3 * index))
        force = magnitude * weight * direction / np.linalg.norm(direction)
        rows.append(
            f" * {symbol:<15}{index:<d}{force[0]:13.5f}{force[1]:21.5f}{force[2]:21.5f}         *"
        )
    border = " " + "*" * 84
    return [
        " ************************************** Forces **************************************",
        " *                           Cartesian components (eV/A)                            *",
        *rows,
        border,
        "",
    ]


def _energy_profile(initial: float, final: float, iterations: int, last_step: float) -> list[float]:
    """Energies from ``initial`` to ``final`` that fall by less at every step.

    The profile is geometric, ``final + (initial - final) (q^k - q^N) / (1 - q^N)``,
    with the ratio ``q`` chosen so that the last step falls by ``last_step``.
    When even an even spacing gives a smaller last step, the spacing is even.
    """

    span = initial - final
    if iterations == 0:
        return [initial]
    if span <= last_step * iterations:
        return [initial - span * step / iterations for step in range(iterations + 1)]

    def last_drop(q: float) -> float:
        return span * q ** (iterations - 1) * (1.0 - q) / (1.0 - q**iterations)

    low, high = 1e-6, 1.0 - 1e-9
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if last_drop(middle) < last_step else (low, middle)
    q = 0.5 * (low + high)
    return [
        final + span * (q**step - q**iterations) / (1.0 - q**iterations)
        for step in range(iterations + 1)
    ]


def bfgs_log_text(
    *,
    lattice: Sequence[Sequence[float]],
    species: Sequence[str],
    fractional: Sequence[Sequence[float]],
    iterations: int,
    initial_enthalpy: float,
    final_enthalpy: float,
    initial_force: float = 1.0,
    final_force: float = 0.02,
    energy_tolerance: float = 2.0e-5,
    force_tolerance: float = 0.05,
    displacement_tolerance: float = 1.0e-3,
    max_iterations: int = 100,
    finished: bool = True,
    wall_time: float = 1843.2,
) -> str:
    """A fixed-cell CASTEP BFGS geometry optimisation of ``iterations`` steps.

    The output holds what :func:`mufinder.castep_io.output_parser.parse_castep_output`
    and the geometry-progress parser read, in the layout of
    ``tests/fixtures/geometry/Si2_bfgs.castep``: the ``Run started`` banner,
    the optimisation parameter echo, and for iterations 0 to ``iterations`` an
    energy, a forces block and a convergence table. The enthalpy falls from
    ``initial_enthalpy`` to ``final_enthalpy`` by less at every step, the
    largest force falls geometrically from ``initial_force`` to
    ``final_force``, and the largest displacement follows the force.

    A ``finished`` run ends with every criterion met, the converged message,
    the final configuration of ``lattice``, ``species`` and ``fractional``,
    and the closing timing line; a ``ValueError`` is raised if
    ``final_force`` does not meet ``force_tolerance``. An unfinished run stops
    after the last table, as a file CASTEP is still writing does.
    """

    if iterations < 1:
        raise ValueError("A geometry optimisation log needs at least one iteration")
    if iterations > max_iterations:
        raise ValueError(f"{iterations} iterations exceed max_iterations = {max_iterations}")
    if finished and final_force > force_tolerance:
        raise ValueError(
            f"final_force {final_force} does not meet force_tolerance {force_tolerance}"
        )
    count = len(species)
    enthalpies = _energy_profile(
        initial_enthalpy, final_enthalpy, iterations, 0.5 * energy_tolerance * count
    )
    forces = [
        initial_force * (final_force / initial_force) ** (step / iterations)
        for step in range(iterations + 1)
    ]

    def row(name: str, value: float, tolerance: float, unit: str, *, first: bool = False) -> str:
        # Iteration 0 has no previous step, so CASTEP reports its energy
        # change and displacement as zero and not converged.
        ok = "Yes" if value <= tolerance and not first else "No "
        return (
            f" | {name:^9} |   {_castep_float(value, 6)} |   {_castep_float(tolerance, 6)} |"
            f" {unit:>10} | {ok} | <-- BFGS"
        )

    divider = " +-----------+-----------------+-----------------+------------+-----+ <-- BFGS"
    lines = [
        _RUN_STARTED,
        " ********************** Geometry Optimization Parameters ***********************",
        "  ",
        " optimization method                            : BFGS",
        " variable cell method                           : fixed basis quality",
        f" max. number of steps                           : {max_iterations:10d}",
        f" total energy convergence tolerance             : {energy_tolerance:.4E}   eV/atom",
        f" max ionic |force| tolerance                    : {force_tolerance:.4E}   eV/A",
        f" max ionic |displacement| tolerance             : {displacement_tolerance:.4E}   A",
        "",
    ]
    for step, (enthalpy, force) in enumerate(zip(enthalpies, forces, strict=True)):
        if step:
            lines += [f" Starting BFGS iteration {step:10d} ...", ""]
        energy_change = 0.0 if step == 0 else (enthalpies[step - 1] - enthalpy) / count
        displacement = 0.0 if step == 0 else 0.5 * displacement_tolerance * force / final_force
        lines += [
            f"Final energy, E             =  {enthalpy:.10f}     eV",
            "",
            *_forces_block(species, force),
            f" BFGS: finished iteration {step:5d} with enthalpy= {_castep_float(enthalpy, 8)} eV",
            "  ",
            divider,
            " | Parameter |      value      |    tolerance    |    units   | OK? | <-- BFGS",
            divider,
            row("dE/ion", energy_change, energy_tolerance, "eV", first=step == 0),
            row("|F|max", force, force_tolerance, "eV/A"),
            row("|dR|max", displacement, displacement_tolerance, "A", first=step == 0),
            divider,
            "  ",
        ]
    if not finished:
        return "\n".join(lines) + "\n"

    vectors = np.asarray(lattice, dtype=float)
    reciprocal = 2.0 * np.pi * np.linalg.inv(vectors).T
    border = "            " + "x" * 65
    atoms = [
        f"            x  {symbol:<8}{index:10d}   {position[0]:12.6f}{position[1]:11.6f}{position[2]:11.6f}   x"
        for index, (symbol, position) in enumerate(zip(species, fractional, strict=True), start=1)
    ]
    lines += [
        " BFGS: Geometry optimization completed successfully.",
        "",
        "=" * 80,
        " BFGS: Final Configuration:",
        "=" * 80,
        "",
        "                           -------------------------------",
        "                                      Unit Cell",
        "                           -------------------------------",
        "        Real Lattice(A)              Reciprocal Lattice(1/A)",
        *(
            f"   {real[0]:12.7f}{real[1]:14.7f}{real[2]:14.7f}    {back[0]:14.9f}{back[1]:14.9f}{back[2]:14.9f}"
            for real, back in zip(vectors, reciprocal, strict=True)
        ),
        "",
        border,
        "            x  Element         Atom        Fractional coordinates of atoms  x",
        "            x                 Number           u          v          w      x",
        "            x---------------------------------------------------------------x",
        *atoms,
        border,
        "",
        f" BFGS: Final Enthalpy     = {_castep_float(enthalpies[-1], 8)} eV",
        "",
        *_forces_block(species, forces[-1]),
        f"Total time          = {wall_time:9.2f} s",
    ]
    return "\n".join(lines) + "\n"


# Atomic masses in AMU, as CASTEP writes them into a ``.phonon`` header.
_MASSES_AMU = {"Co": 58.933194, "F": 18.998403163, "N": 14.0067, "Si": 28.0855}


def _phonon_header(cell: Any, modes: Sequence[tuple[float, np.ndarray]]) -> list[str]:
    """The lines of a Gamma-only ``.phonon`` file of ``cell`` with ``modes`` (frequency, eigenvectors)."""

    from mufinder.domain.constants import MUON_MASS_AMU

    species = list(cell.species)
    count = len(species)
    lines = [
        " BEGIN header",
        f" Number of ions {count:9d}",
        f" Number of branches {3 * count:5d}",
        " Number of wavevectors  1",
        " Frequencies in         cm-1",
        " Unit cell vectors (A)",
        *(
            f" {row[0]:12.6f}{row[1]:12.6f}{row[2]:12.6f}"
            for row in np.asarray(cell.lattice_vectors, dtype=float)
        ),
        " Fractional Co-ordinates",
    ]
    for index, (symbol, position) in enumerate(
        zip(species, cell.fractional_positions, strict=True), start=1
    ):
        mass = MUON_MASS_AMU if symbol == "H:mu" else _MASSES_AMU[symbol]
        lines.append(
            f" {index:5d} {position[0]:12.6f}{position[1]:12.6f}{position[2]:12.6f}   {symbol:<8}{mass:12.6f}"
        )
    lines += [" END header", "     q-pt=    1    0.000000  0.000000  0.000000      1.0000000000"]
    lines += [
        f" {number:7d} {frequency:14.6f}"
        for number, (frequency, _vector) in enumerate(modes, start=1)
    ]
    lines += [
        "                        Phonon Eigenvectors",
        "Mode Ion                X                                   Y                                   Z",
    ]
    for number, (_frequency, vector) in enumerate(modes, start=1):
        for atom in range(count):
            lines.append(
                f" {number:3d} {atom + 1:3d}"
                + "".join(f" {value:15.12f} {0.0:15.12f}" for value in vector[atom])
            )
    return lines


def _host_band(count: int, top_cm: float) -> list[float]:
    """``count`` host frequencies spread below ``top_cm``, the same on every build."""

    return [top_cm * np.sin(0.5 * np.pi * (number + 1) / (count + 1)) for number in range(count)]


def triatomic_phonon_text(
    cell: Any,
    *,
    stretch_cm: float,
    bend_cm: tuple[float, float],
    antisymmetric_cm: float,
    host_top_cm: float,
) -> str:
    """A Gamma-only CASTEP ``.phonon`` file for a linear X-mu-X complex embedded in ``cell``.

    The complex is the muon and its two nearest host atoms, which must share a
    species. Its nine degrees of freedom are written along three orthonormal
    axes: the X-X axis, the in-plane direction normal to it, and the remaining
    direction closest to the cell's ``c`` axis. Along each axis the three atoms
    span, in mass-weighted coordinates, a rigid translation, the symmetric
    combination of the two X atoms, and the vector orthogonal to both, in which
    the muon moves against the two X atoms with the centre of mass at rest.
    The last is the antisymmetric stretch at ``antisymmetric_cm`` along the
    X-X axis and the two bends ``bend_cm`` across it; the symmetric
    combination along the axis is the symmetric stretch at ``stretch_cm``.
    Every other vector, and every Cartesian direction of every other atom, is
    a host mode below ``host_top_cm``. The eigenvectors are orthonormal by
    construction, and the muon carries ``2 m_X / (2 m_X + m_mu)`` of each of
    its three modes.
    """

    from mufinder.domain.constants import MUON_MASS_AMU

    species = list(cell.species)
    count = len(species)
    lattice = np.asarray(cell.lattice_vectors, dtype=float)
    fractional = np.asarray(cell.fractional_positions, dtype=float)
    muon = species.index("H:mu")
    separation = (fractional - fractional[muon] + 0.5) % 1.0 - 0.5
    distances = np.linalg.norm(separation @ lattice, axis=1)
    distances[muon] = np.inf
    first, second = (int(index) for index in np.argsort(distances)[:2])
    if species[first] != species[second]:
        raise ValueError("The muon's two nearest atoms must share a species for an X-mu-X complex")
    axis = (separation[second] - separation[first]) @ lattice
    axis /= np.linalg.norm(axis)
    c_axis = lattice[2] / np.linalg.norm(lattice[2])
    across_c = c_axis - np.dot(c_axis, axis) * axis
    across_c /= np.linalg.norm(across_c)
    in_plane = np.cross(across_c, axis)

    mass = _MASSES_AMU[species[first]]
    roots = np.sqrt([mass, MUON_MASS_AMU, mass])
    translation = roots / np.linalg.norm(roots)
    symmetric = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    muon_against = np.cross(translation, symmetric)
    muon_against *= np.sign(muon_against[1]) / np.linalg.norm(muon_against)
    complex_atoms = (first, muon, second)

    def mode(coefficients: np.ndarray, direction: np.ndarray) -> np.ndarray:
        vector = np.zeros((count, 3))
        for atom, coefficient in zip(complex_atoms, coefficients, strict=True):
            vector[atom] = coefficient * direction
        return vector

    modes: list[tuple[float, np.ndarray]] = [
        (antisymmetric_cm, mode(muon_against, axis)),
        (stretch_cm, mode(symmetric, axis)),
        (bend_cm[0], mode(muon_against, in_plane)),
        (bend_cm[1], mode(muon_against, across_c)),
    ]
    host_vectors = [mode(translation, axis), mode(translation, in_plane), mode(symmetric, in_plane)]
    host_vectors += [mode(translation, across_c), mode(symmetric, across_c)]
    for atom in range(count):
        if atom in complex_atoms:
            continue
        for direction in np.eye(3):
            vector = np.zeros((count, 3))
            vector[atom] = direction
            host_vectors.append(vector)
    modes += list(zip(_host_band(len(host_vectors), host_top_cm), host_vectors, strict=True))
    modes.sort(key=lambda entry: entry[0])
    return "\n".join(_phonon_header(cell, modes)) + "\n"


def _muon_frame(cell: Any) -> tuple[np.ndarray, np.ndarray, list[str], int, np.ndarray]:
    """The lattice, fractional positions, species, muon index and minimum-image vectors (Å) from the muon."""

    lattice = np.asarray(cell.lattice_vectors, dtype=float)
    fractional = np.asarray(cell.fractional_positions, dtype=float)
    species = list(cell.species)
    muon = species.index("H:mu")
    vectors = ((fractional - fractional[muon] + 0.5) % 1.0 - 0.5) @ lattice
    return lattice, fractional, species, muon, vectors


def _diatomic_molecules(
    lattice: np.ndarray, fractional: np.ndarray, atoms: Sequence[int]
) -> list[tuple[int, int]]:
    """``atoms`` paired with their nearest neighbours into molecules; a ``ValueError`` if they do not pair."""

    pairs: set[tuple[int, int]] = set()
    for atom in atoms:
        others = [other for other in atoms if other != atom]
        separation = (fractional[others] - fractional[atom] + 0.5) % 1.0 - 0.5
        partner = others[int(np.argmin(np.linalg.norm(separation @ lattice, axis=1)))]
        pairs.add((min(atom, partner), max(atom, partner)))
    if 2 * len(pairs) != len(atoms):
        raise ValueError("The atoms other than the muon do not pair into diatomic molecules")
    return sorted(pairs)


def _molecule_atoms(
    lattice: np.ndarray, fractional: np.ndarray, vectors: np.ndarray, molecule: tuple[int, int]
) -> np.ndarray:
    """The two atoms of ``molecule`` as vectors (Å) from the muon, the second its own nearest image of the first."""

    first, second = molecule
    bond = ((fractional[second] - fractional[first] + 0.5) % 1.0 - 0.5) @ lattice
    return np.array([vectors[first], vectors[first] + bond])


def _bridged_pair(
    cell: Any,
) -> tuple[
    np.ndarray, np.ndarray, list[str], int, np.ndarray, list[tuple[int, int]], list[tuple[int, int]]
]:
    """The muon frame, every molecule, and the two molecules whose centres are nearest the muon.

    Each of the two is ordered (near atom, far atom) from the muon.
    """

    lattice, fractional, species, muon, vectors = _muon_frame(cell)
    molecules = _diatomic_molecules(
        lattice, fractional, [atom for atom in range(len(species)) if atom != muon]
    )
    centres = [
        np.linalg.norm(_molecule_atoms(lattice, fractional, vectors, molecule).mean(axis=0))
        for molecule in molecules
    ]
    nearest = [molecules[index] for index in np.argsort(centres, kind="stable")[:2]]
    ordered = []
    for first, second in nearest:
        near_first = np.linalg.norm(vectors[first]) <= np.linalg.norm(vectors[second])
        ordered.append((first, second) if near_first else (second, first))
    return lattice, fractional, species, muon, vectors, molecules, ordered


def bridged_diatomic_cell(
    cell: Any,
    *,
    muon_bond: float,
    molecule_bond: float,
    neighbour_shift: float,
    neighbour_radius: float,
) -> list[list[float]]:
    """The fractional positions of ``cell`` with its muon bridging the two molecules nearest it.

    Every atom but the muon (``H:mu``) belongs to a diatomic molecule, its
    nearest neighbour. The two molecules whose centres are nearest the muon
    turn to point at it, along the line from the muon to their centres: the
    near atom ``muon_bond`` Å from the muon and the far atom ``molecule_bond``
    Å beyond it. Every other molecule whose centre lies within
    ``neighbour_radius`` Å of the muon moves rigidly ``neighbour_shift`` Å
    towards it. The muon and every other atom stay where they are.
    """

    lattice, fractional, _species, muon, vectors, molecules, bridged = _bridged_pair(cell)
    moved = np.array(fractional)

    def place(atom: int, vector: np.ndarray) -> None:
        moved[atom] = (fractional[muon] + np.linalg.solve(lattice.T, vector)) % 1.0

    bridged_atoms = {atom for pair in bridged for atom in pair}
    for near, far in bridged:
        centre = _molecule_atoms(lattice, fractional, vectors, (near, far)).mean(axis=0)
        direction = centre / np.linalg.norm(centre)
        place(near, direction * muon_bond)
        place(far, direction * (muon_bond + molecule_bond))
    for molecule in molecules:
        if bridged_atoms.intersection(molecule):
            continue
        atoms = _molecule_atoms(lattice, fractional, vectors, molecule)
        centre = atoms.mean(axis=0)
        distance = float(np.linalg.norm(centre))
        if distance < neighbour_radius:
            for atom, vector in zip(
                molecule, atoms - neighbour_shift * centre / distance, strict=True
            ):
                place(atom, vector)
    return moved.tolist()


# The spring constants, in eV/Å², of the five-atom [N2-mu-N2]+ chain model in
# tests/support/phonon_fixtures.py (N2_MU_N2_FORCE_CONSTANTS), tuned there to
# the harmonic numbers of Gomilšek et al., Commun. Phys. 6, 142 (2023).
# tests/docs/test_docs_synthetic.py keeps the two equal.
N2_MU_N2_SPRINGS: dict[str, float] = {
    "muon_nitrogen_stretch": 1.612,
    "muon_nitrogen_bend": 2.227,
    "nitrogen_stretch": 179.2,
    "nitrogen_bend": 0.167,
    "nitrogen_lattice": 0.33,
}


def _chain_block_modes(
    masses: np.ndarray, muon_spring: float, molecule_spring: float, tether: float
) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies (cm^-1) and mass-weighted eigenvectors (columns) of one Cartesian block of the chain.

    The chain is far atom, near atom, muon, near atom, far atom: a muon spring
    to each near atom, a molecule spring in each molecule, and a tether of
    every non-muon atom to the lattice. Each eigenvector's muon component (or
    its largest one, where the muon is at rest) is made positive.
    """

    from mufinder.science.quantum_phonons import (
        ATOMIC_MASS_KG,
        ELEMENTARY_CHARGE_C,
        LIGHT_SPEED_M_S,
    )

    hessian = np.zeros((5, 5))
    for i, j, spring in (
        (0, 1, molecule_spring),
        (1, 2, muon_spring),
        (2, 3, muon_spring),
        (3, 4, molecule_spring),
    ):
        hessian[[i, j], [i, j]] += spring
        hessian[i, j] -= spring
        hessian[j, i] -= spring
    for atom in (0, 1, 3, 4):
        hessian[atom, atom] += tether
    inverse_root = 1.0 / np.sqrt(masses)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian * np.outer(inverse_root, inverse_root))
    angular = np.sqrt(np.clip(eigenvalues, 0.0, None)) * np.sqrt(
        ELEMENTARY_CHARGE_C / (1e-20 * ATOMIC_MASS_KG)
    )
    for column in range(5):
        vector = eigenvectors[:, column]
        pivot = 2 if abs(vector[2]) > 1e-9 else int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            eigenvectors[:, column] = -vector
    return angular / (2.0 * np.pi * LIGHT_SPEED_M_S * 100.0), eigenvectors


def _perpendicular_pair(axis: np.ndarray, lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors normal to ``axis`` and each other: the first the lattice vector most normal to it, made normal."""

    references = lattice / np.linalg.norm(lattice, axis=1)[:, None]
    reference = references[int(np.argmin(np.abs(references @ axis)))]
    first = reference - np.dot(reference, axis) * axis
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def bridged_diatomic_phonon_text(
    cell: Any,
    *,
    stretch_band_cm: tuple[float, float],
    libration_band_cm: tuple[float, float],
    translation_band_cm: tuple[float, float],
    springs: dict[str, float] = N2_MU_N2_SPRINGS,
) -> str:
    """A Gamma-only CASTEP ``.phonon`` file for a muon bridging two diatomic molecules in ``cell``.

    The complex is the muon and the two molecules whose centres are nearest
    it, which must share a species: a five-atom chain along the line through
    their near atoms. Its fifteen modes are the chain spring model of
    ``springs``, the constants of ``N2_MU_N2_SPRINGS`` by default, solved in
    three Cartesian blocks: along the chain with the stretch springs, and
    across it, along the two normals of :func:`_perpendicular_pair`, with the
    bend springs. Every other molecule vibrates on its own: its bond
    stretch in ``stretch_band_cm``, its two librations in
    ``libration_band_cm`` and its three translations in
    ``translation_band_cm``, each band spread evenly over the molecules in
    atom order. The eigenvectors are orthonormal by construction.
    """

    from mufinder.domain.constants import MUON_MASS_AMU

    lattice, fractional, species, muon, vectors, molecules, bridged = _bridged_pair(cell)
    count = len(species)
    (near_a, far_a), (near_b, far_b) = bridged
    if len({species[atom] for atom in (near_a, far_a, near_b, far_b)}) != 1:
        raise ValueError("The two molecules the muon bridges must share a species")
    mass = _MASSES_AMU[species[near_a]]
    chain = (far_a, near_a, muon, near_b, far_b)
    masses = np.array([mass, mass, MUON_MASS_AMU, mass, mass])
    axis = vectors[near_b] - vectors[near_a]
    axis /= np.linalg.norm(axis)
    across = _perpendicular_pair(axis, lattice)

    def vector_of(components: dict[int, np.ndarray]) -> np.ndarray:
        vector = np.zeros((count, 3))
        for atom, value in components.items():
            vector[atom] = value
        return vector

    modes: list[tuple[float, np.ndarray]] = []
    blocks = (
        (axis, springs["muon_nitrogen_stretch"], springs["nitrogen_stretch"]),
        (across[0], springs["muon_nitrogen_bend"], springs["nitrogen_bend"]),
        (across[1], springs["muon_nitrogen_bend"], springs["nitrogen_bend"]),
    )
    for direction, muon_spring, molecule_spring in blocks:
        frequencies, eigenvectors = _chain_block_modes(
            masses, muon_spring, molecule_spring, springs["nitrogen_lattice"]
        )
        for frequency, column in zip(frequencies, eigenvectors.T, strict=True):
            modes.append(
                (
                    float(frequency),
                    vector_of(
                        {atom: value * direction for atom, value in zip(chain, column, strict=True)}
                    ),
                )
            )

    others = [molecule for molecule in molecules if not set(molecule) & set(chain)]
    stretch = np.linspace(*stretch_band_cm, len(others))
    libration = np.linspace(*libration_band_cm, 2 * len(others))
    translation = np.linspace(*translation_band_cm, 3 * len(others))
    half = 1.0 / np.sqrt(2.0)
    for index, (first, second) in enumerate(others):
        bond = np.diff(_molecule_atoms(lattice, fractional, vectors, (first, second)), axis=0)[0]
        bond /= np.linalg.norm(bond)
        modes.append((float(stretch[index]), vector_of({first: -half * bond, second: half * bond})))
        for turn, normal in enumerate(_perpendicular_pair(bond, lattice)):
            modes.append(
                (
                    float(libration[2 * index + turn]),
                    vector_of({first: -half * normal, second: half * normal}),
                )
            )
        for shift, direction in enumerate(np.eye(3)):
            modes.append(
                (
                    float(translation[3 * index + shift]),
                    vector_of({first: half * direction, second: half * direction}),
                )
            )
    modes.sort(key=lambda entry: entry[0])
    return "\n".join(_phonon_header(cell, modes)) + "\n"


def den_fmt_text(
    lattice: Sequence[Sequence[float]],
    species: Sequence[str],
    fractional: Sequence[Sequence[float]],
    *,
    grid: tuple[int, int, int],
    shells: dict[str, Sequence[tuple[float, float]]],
) -> str:
    """A CASTEP ``.den_fmt`` of superposed spherical Gaussian atomic densities.

    ``shells`` gives, per species, ``(electrons, width in Å)`` pairs: each atom
    carries one normalised Gaussian of that width per pair. The density is
    built from its Fourier coefficients on ``grid``, so it is periodic and band
    limited, and is written as CASTEP writes a density: electrons per grid
    point times the number of grid points, whose cell average is the electron
    count. This is a model density, not a self-consistent one.
    """

    vectors = np.asarray(lattice, dtype=float)
    positions = np.asarray(fractional, dtype=float)
    indices = np.array(np.meshgrid(*(np.fft.fftfreq(size) * size for size in grid), indexing="ij"))
    reciprocal = 2.0 * np.pi * np.linalg.inv(vectors).T
    wavevectors = np.tensordot(reciprocal.T, indices, axes=(1, 0))
    g_squared = np.sum(wavevectors**2, axis=0)
    coefficients = np.zeros(grid, dtype=complex)
    for symbol, position in zip(species, positions, strict=True):
        phase = np.exp(-2.0j * np.pi * np.tensordot(position, indices, axes=(0, 0)))
        for electrons, width in shells[symbol]:
            coefficients += electrons * phase * np.exp(-0.5 * width**2 * g_squared)
    density = np.real(np.fft.ifftn(coefficients)) * np.prod(grid)
    if density.min() <= 0.0:
        raise ValueError(
            "The Gaussian widths are too narrow for the grid: the density goes negative"
        )

    lines = [
        " BEGIN header",
        " ",
        "           Real Lattice(A)               Lattice parameters(A)    Cell Angles",
        *(f"  {row[0]:12.7f}  {row[1]:12.7f}  {row[2]:12.7f}" for row in vectors),
        " ",
        "   1                            ! nspins",
        f"  {grid[0]:3d}  {grid[1]:3d}  {grid[2]:3d}                ! fine FFT grid along <a,b,c>",
        ' END header: data is "<a b c> charge" in units of electrons/grid_point * number of grid_points',
        " ",
    ]
    lines += [
        f"{a + 1:6d}{b + 1:6d}{c + 1:6d}{density[a, b, c]:20.6f}"
        for c in range(grid[2])
        for b in range(grid[1])
        for a in range(grid[0])
    ]
    return "\n".join(lines) + "\n"


def uep_castep_text(
    *,
    lattice: Sequence[Sequence[float]],
    species: Sequence[str],
    fractional: Sequence[Sequence[float]],
    pseudopotentials: dict[str, tuple[float, float]],
    energy: float,
    wall_time: float,
) -> str:
    """A neutral single-point CASTEP output with what the UEP analysis reads.

    It holds a pseudopotential report per species, giving its ionic charge and
    core radius (``pseudopotentials`` maps each species to ``(ionic charge,
    core radius in bohr)``), the parameter header, the unit cell and cell
    contents ASE reads the host from, a final energy, and the timing lines
    that end a run.
    """

    vectors = np.asarray(lattice, dtype=float)
    reciprocal = 2.0 * np.pi * np.linalg.inv(vectors).T
    electrons = sum(pseudopotentials[symbol][0] for symbol in species)
    reports = []
    for symbol, (charge, radius) in pseudopotentials.items():
        reports += [
            "   ============================================================",
            "   | Pseudopotential Report - Date of generation 12-10-2026   |",
            "   ------------------------------------------------------------",
            f"   | Element: {symbol:<2} Ionic charge: {charge:5.2f} Level of theory: PBE     |",
            "   |                 Pseudopotential Definition               |",
            "   |        Beta     l      e      Rc     scheme   norm       |",
            f"   |          1      0   -0.400   {radius:5.3f}     qc      0         |",
            f"   |         loc     2    0.000   {radius:5.3f}     pn      0         |",
            "   ------------------------------------------------------------",
            "   |      Author: synthetic model for the documentation       |",
            "   ============================================================",
            "",
        ]
    atoms = [
        f"            x  {symbol:<2}  {index:10d}   {position[0]:10.6f} {position[1]:10.6f} {position[2]:10.6f}   x"
        for index, (symbol, position) in enumerate(zip(species, fractional, strict=True), start=1)
    ]
    border = "            " + "x" * 60
    lines = [
        " | Welcome to Academic Release CASTEP version 24.1 |",
        "",
        _RUN_STARTED,
        *reports,
        " ************************************ Title ************************************",
        "",
        " ***************************** General Parameters ******************************",
        "",
        " type of calculation                            : single point energy",
        "",
        " *********************** Exchange-Correlation Parameters ***********************",
        "",
        " using functional                               : Perdew Burke Ernzerhof",
        "",
        " **************************** Electronic Parameters ****************************",
        "",
        f" number of  electrons                           :  {electrons:.3f}",
        " net charge of system                           :  0.000",
        " treating system as spin-polarized",
        "",
        " *******************************************************************************",
        "",
        "                           -------------------------------",
        "                                      Unit Cell",
        "                           -------------------------------",
        "        Real Lattice(A)                      Reciprocal Lattice(1/A)",
        *(
            f"  {real[0]:12.7f}  {real[1]:12.7f}  {real[2]:12.7f}     {back[0]:14.9f}{back[1]:14.9f}{back[2]:14.9f}"
            for real, back in zip(vectors, reciprocal, strict=True)
        ),
        "",
        "                           -------------------------------",
        "                                     Cell Contents",
        "                           -------------------------------",
        f"                         Total number of ions in cell = {len(species):4d}",
        f"                      Total number of species in cell = {len(pseudopotentials):4d}",
        "",
        border,
        "            x  Element    Atom        Fractional coordinates of atoms  x",
        "            x            Number           u          v          w      x",
        "            x----------------------------------------------------------x",
        *atoms,
        border,
        "",
        f"Final energy =  {energy:.10f}     eV",
        "",
        "Initialisation time =      5.44 s",
        f"Calculation time    = {wall_time - 5.48:9.2f} s",
        "Finalisation time   =      0.04 s",
        f"Total time          = {wall_time:9.2f} s",
    ]
    return "\n".join(lines) + "\n"


def legacy_castep_text(
    *, energy: float, converged: bool, cut_off_ev: float, spin_polarized: bool
) -> str:
    """A legacy run's CASTEP output: the header lines the importer reads and the result."""

    tail = " Geometry optimization completed successfully.\n\n" if converged else ""
    spin = "spin-polarized" if spin_polarized else "non-spin-polarized"
    return (
        " Run started: Tue, 14 Mar 2023 10:02:41 +0000\n\n"
        " type of calculation                            : geometry optimization\n"
        " using functional                               : Perdew Burke Ernzerhof\n"
        f" plane wave basis set cut-off                   :   {cut_off_ev:.4f}   eV\n"
        " net charge of system                           :  1.000\n"
        f" treating system as {spin}\n\n"
        "                       MP grid size for SCF calculation is  2  2  3\n\n"
        f" Final energy, E             =  {energy:.4f}     eV\n\n"
        f"{tail} Total time          =   22143.70 s\n"
    )
