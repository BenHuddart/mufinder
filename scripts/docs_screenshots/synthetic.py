"""CASTEP output files written for the screenshot projects, in CASTEP's file shapes.

No screenshot runs CASTEP. A project whose page shows a finished calculation
gets its outputs from these writers, or from a repository fixture where one
fits. Each writer produces the lines MuFinder's parsers read and nothing
more; the numbers are whatever the caller passes, so a page that shows them
says in its caption that they are illustrative.

This module imports neither Qt nor VTK, so tests import it directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
from typing import Any

import numpy as np

BOHR_ANG = 0.529177210903
HARTREE_EV = 27.211386245988

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
    lattice_row = next(index for index in range(start, len(lines)) if "Real Lattice" in lines[index]) + 1
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
        rows.append(f" * {symbol:<15}{index:<d}{force[0]:13.5f}{force[1]:21.5f}{force[2]:21.5f}         *")
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
    return [final + span * (q**step - q**iterations) / (1.0 - q**iterations) for step in range(iterations + 1)]


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
        raise ValueError(f"final_force {final_force} does not meet force_tolerance {force_tolerance}")
    count = len(species)
    enthalpies = _energy_profile(initial_enthalpy, final_enthalpy, iterations, 0.5 * energy_tolerance * count)
    forces = [initial_force * (final_force / initial_force) ** (step / iterations) for step in range(iterations + 1)]

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


def phonon_text(
    cell: Any,
    *,
    muon_modes_cm: Sequence[float],
    muon_participation: float,
    host_top_cm: float,
) -> str:
    """A Gamma-only CASTEP ``.phonon`` file for ``cell`` with ``muon_modes_cm`` muon modes.

    The eigenvectors are orthonormal: each muon direction is rotated with the
    same direction of the first host atom, which gives a muon mode, carrying
    ``muon_participation`` of the muon's motion, and a host partner just below
    ``host_top_cm``; every other degree of freedom is a host mode of its own,
    spread below ``host_top_cm``.
    """

    from mufinder.domain.constants import MUON_MASS_AMU

    species = list(cell.species)
    count = len(species)
    muon = species.index("H:mu")
    neighbour = next(index for index, symbol in enumerate(species) if symbol != "H:mu")
    mix = np.sqrt(muon_participation)
    other = np.sqrt(1.0 - muon_participation)
    modes: list[tuple[float, np.ndarray]] = []
    paired: set[tuple[int, int]] = set()
    for axis, frequency in enumerate(muon_modes_cm):
        muon_mode = np.zeros((count, 3))
        muon_mode[muon, axis], muon_mode[neighbour, axis] = mix, other
        partner = np.zeros((count, 3))
        partner[muon, axis], partner[neighbour, axis] = -other, mix
        modes += [(frequency, muon_mode), (host_top_cm * (0.9 + 0.02 * axis), partner)]
        paired |= {(muon, axis), (neighbour, axis)}
    host = [(atom, axis) for atom in range(count) for axis in range(3) if (atom, axis) not in paired]
    for number, (atom, axis) in enumerate(host):
        vector = np.zeros((count, 3))
        vector[atom, axis] = 1.0
        modes.append((host_top_cm * np.sin(0.5 * np.pi * (number + 1) / (len(host) + 1)), vector))
    modes.sort(key=lambda mode: mode[0])

    lines = [
        " BEGIN header",
        f" Number of ions {count:9d}",
        f" Number of branches {3 * count:5d}",
        " Number of wavevectors  1",
        " Frequencies in         cm-1",
        " Unit cell vectors (A)",
        *(f" {row[0]:12.6f}{row[1]:12.6f}{row[2]:12.6f}" for row in np.asarray(cell.lattice_vectors, dtype=float)),
        " Fractional Co-ordinates",
    ]
    for index, (symbol, position) in enumerate(zip(species, cell.fractional_positions, strict=True), start=1):
        mass = MUON_MASS_AMU if symbol == "H:mu" else 28.0855
        lines.append(f" {index:5d} {position[0]:12.6f}{position[1]:12.6f}{position[2]:12.6f}   {symbol:<8}{mass:12.6f}")
    lines += [" END header", "     q-pt=    1    0.000000  0.000000  0.000000      1.0000000000"]
    lines += [f" {number:7d} {frequency:14.6f}" for number, (frequency, _vector) in enumerate(modes, start=1)]
    lines += ["                        Phonon Eigenvectors", "Mode Ion                X                                   Y                                   Z"]
    for number, (_frequency, vector) in enumerate(modes, start=1):
        for atom in range(count):
            lines.append(f" {number:3d} {atom + 1:3d}" + "".join(f" {value:15.12f} {0.0:15.12f}" for value in vector[atom]))
    return "\n".join(lines) + "\n"


def ts_text(lattice: np.ndarray, species: list[str], images: list[tuple[str, float, float, np.ndarray]]) -> str:
    """A CASTEP ``.ts`` file in atomic units: one REA, the TST images and one PRO block."""

    def number(value: float) -> str:
        mantissa, exponent = f"{value:.8E}".split("E")
        return f"{mantissa}E{int(exponent):+04d}".rjust(18)

    def row(values: Any) -> str:
        return "".join(f"   {number(float(value))}" for value in values)

    lines = [" TSTYPE TSConfirmation"]
    for tag, coordinate, energy_ev, cartesian in images:
        lines.append(f" {tag:<3} {1:4d}   {number(coordinate)}")
        energy = energy_ev / HARTREE_EV
        lines.append(" " * 9 + row((energy, energy)) + " " * 21 + "  <-- E")
        lines.extend(" " * 9 + row(vector / BOHR_ANG) + "  <-- h" for vector in lattice)
        lines.extend(
            f" {symbol:<8} {1:4d}" + row(position / BOHR_ANG) + "  <-- R"
            for symbol, position in zip(species, cartesian, strict=True)
        )
        lines.extend(f" {symbol:<8} {1:4d}" + row((0.0, 0.0, 0.0)) + "  <-- F" for symbol in species)
        lines.append("  ")
    return "\n".join(lines) + "\n"


def legacy_castep_text(*, energy: float, converged: bool) -> str:
    """A legacy run's CASTEP output: the header lines the importer reads and the result."""

    tail = " Geometry optimization completed successfully.\n\n" if converged else ""
    return (
        " Run started: Tue, 14 Mar 2023 10:02:41 +0000\n\n"
        " type of calculation                            : geometry optimization\n"
        " using functional                               : Perdew Burke Ernzerhof\n"
        " plane wave basis set cut-off                   :   400.0000   eV\n"
        " net charge of system                           :  1.000\n"
        " treating system as non-spin-polarized\n\n"
        "                       MP grid size for SCF calculation is  2  2  2\n\n"
        f" Final energy, E             =  {energy:.4f}     eV\n\n"
        f"{tail} Total time          =    2214.37 s\n"
    )
