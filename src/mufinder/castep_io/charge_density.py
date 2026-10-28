"""Readers for the CASTEP outputs that describe a host charge density.

The unperturbed electrostatic potential (UEP) analysis in
``mufinder.science.uep`` needs three things from a CASTEP calculation: the
formatted charge density (``.den_fmt``), the ionic charge and core radii from
the pseudopotential reports in ``.castep`` output (or a pseudopotential file),
and the total cell charge CASTEP reports. This module parses that CASTEP text;
the science layer interprets it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from mufinder.domain.constants import BOHR_RADIUS_ANGSTROM


_FLOAT_RE = r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?"
_PSEUDOPOTENTIAL_HEADER_RE = re.compile(r"Pseudopotential\s+Report", re.IGNORECASE)
_PSEUDOPOTENTIAL_END_RE = re.compile(r"Author:", re.IGNORECASE)
_ELEMENT_RE = re.compile(r"Element:\s+([A-Za-z]{1,2})\s+Ionic charge:\s+([0-9.]+)")
_CORE_RADIUS_RE = re.compile(r"(?:[0-9]+|loc)\s+[0-9]\s+[\-0-9.]+\s+([0-9.]+)")
_NEUTRALITY_RE = re.compile(r"unit\s+cell\s+is\s+neutral", re.IGNORECASE)
_TOTAL_CHARGE_RE = re.compile(rf"total\s+charge[^\n=]*=\s*({_FLOAT_RE})", re.IGNORECASE)


class ChargeDensityInputError(ValueError):
    """Raised when a formatted density or pseudopotential report cannot be read."""


@dataclass(frozen=True, slots=True)
class FormattedDensity:
    """Parsed CASTEP formatted density grid.

    ``values[..., 0]`` is the charge and, for a collinear spin-polarised run
    (``nspins == 2``), ``values[..., 1]`` is the spin density ρ↑ − ρ↓. Both are in
    CASTEP's units of electrons per grid point times the number of grid points,
    that is the density times the cell volume.
    """

    lattice_vectors: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]
    grid: tuple[int, int, int]
    nspins: int
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class IonicPseudopotential:
    """Ionic metadata needed for the Gaussian ionic model."""

    species: str
    ionic_charge: float
    core_radius_ang: float

    @property
    def gaussian_width_ang(self) -> float:
        raise AttributeError("gaussian_width_ang depends on the chosen gw_factor.")


def parse_formatted_density(path: str | Path) -> FormattedDensity:
    """Parse a CASTEP `.den_fmt` file into a dense FFT-ready grid."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ChargeDensityInputError(
            f"Failed to read formatted density '{source}': {exc}"
        ) from exc

    lattice_rows: list[tuple[float, float, float]] = []
    grid: tuple[int, int, int] | None = None
    nspins: int | None = None
    data_start: int | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if "Real Lattice" in line:
            for offset in range(1, 4):
                row_line = lines[index + offset]
                parts = row_line.split()[:3]
                if len(parts) != 3:
                    raise ChargeDensityInputError(
                        f"Could not parse lattice vectors from '{source.name}'."
                    )
                lattice_rows.append(tuple(float(value) for value in parts))
            continue
        if "! nspins" in line:
            try:
                nspins = int(line.split()[0])
            except (IndexError, ValueError) as exc:
                raise ChargeDensityInputError(
                    f"Could not parse nspins from '{source.name}'."
                ) from exc
            continue
        if "! fine FFT grid" in line:
            parts = line.split()[:3]
            if len(parts) != 3:
                raise ChargeDensityInputError(f"Could not parse FFT grid from '{source.name}'.")
            grid = (int(parts[0]), int(parts[1]), int(parts[2]))
            continue
        if stripped.startswith("END header"):
            data_start = index + 1
            break

    if len(lattice_rows) != 3 or grid is None or nspins is None or data_start is None:
        raise ChargeDensityInputError(
            f"Formatted density '{source.name}' does not contain a complete CASTEP header."
        )

    values = np.zeros((grid[0], grid[1], grid[2], nspins), dtype=float)
    count = 0
    expected_count = grid[0] * grid[1] * grid[2]
    for line in lines[data_start:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        if nspins == 2 and len(parts) < 5:
            raise ChargeDensityInputError(
                f"Spin-polarised density '{source.name}' has no spin column: {line.strip()}"
            )
        try:
            ia = int(parts[0]) - 1
            ib = int(parts[1]) - 1
            ic = int(parts[2]) - 1
            charge = float(parts[3])
            spin = float(parts[4]) if nspins == 2 else 0.0
        except ValueError as exc:
            raise ChargeDensityInputError(
                f"Invalid density row in '{source.name}': {line.strip()}"
            ) from exc
        if not (0 <= ia < grid[0] and 0 <= ib < grid[1] and 0 <= ic < grid[2]):
            raise ChargeDensityInputError(
                f"Density row index out of range in '{source.name}': {line.strip()}"
            )
        values[ia, ib, ic, 0] = charge
        if nspins == 2:
            values[ia, ib, ic, 1] = spin
        count += 1

    if count != expected_count:
        raise ChargeDensityInputError(
            f"Formatted density '{source.name}' is incomplete: expected {expected_count} points, found {count}."
        )

    return FormattedDensity(
        lattice_vectors=(lattice_rows[0], lattice_rows[1], lattice_rows[2]),
        grid=grid,
        nspins=nspins,
        values=values,
    )


def parse_castep_pseudopotentials(path: str | Path) -> dict[str, IonicPseudopotential]:
    """Parse ionic charges and smallest core radii from CASTEP pseudopotential reports."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ChargeDensityInputError(f"Failed to read CASTEP file '{source}': {exc}") from exc

    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if _PSEUDOPOTENTIAL_HEADER_RE.search(line):
            current = [line]
            continue
        if current is not None:
            current.append(line)
            if _PSEUDOPOTENTIAL_END_RE.search(line):
                blocks.append(current)
                current = None

    parsed: dict[str, IonicPseudopotential] = {}
    for block in blocks:
        element: str | None = None
        ionic_charge: float | None = None
        core_radius_ang = float("inf")
        for line in block:
            if (match := _ELEMENT_RE.search(line)) is not None:
                element = match.group(1)
                ionic_charge = float(match.group(2))
                continue
            if (match := _CORE_RADIUS_RE.search(line)) is not None:
                core_radius_ang = min(core_radius_ang, float(match.group(1)) * BOHR_RADIUS_ANGSTROM)
        if element and ionic_charge is not None and np.isfinite(core_radius_ang):
            parsed[element] = IonicPseudopotential(
                species=element,
                ionic_charge=ionic_charge,
                core_radius_ang=core_radius_ang,
            )

    if not parsed:
        raise ChargeDensityInputError(
            f"Could not find ionic pseudopotential metadata in '{source.name}'. "
            "UEP analysis needs the CASTEP pseudopotential report or a usable CELL/SPECIES_POT fallback."
        )
    return parsed


def castep_total_charge(path: str | Path) -> float | None:
    """Return the total cell charge reported by CASTEP when available."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if _NEUTRALITY_RE.search(text):
        return 0.0
    matches = [float(match.group(1)) for match in _TOTAL_CHARGE_RE.finditer(text)]
    return matches[-1] if matches else None
