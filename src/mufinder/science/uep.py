"""MuFinder-native Unperturbed Electrostatic Potential analysis tools.

This module loads a host's CASTEP charge density, evaluates the UEP field, and
probes, optimizes, samples and meshes it. The CASTEP readers it builds on live
in ``mufinder.castep_io.charge_density``; grouping of optimized positions lives
in ``mufinder.science.uep_clustering``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ase import io
from ase.data import atomic_numbers
import numpy as np
from scipy.optimize import minimize

from mufinder.castep_io import charge_density as _charge_density
from mufinder.castep_io.cell_parser import CellParseError, parse_cell_file
from mufinder.castep_io.charge_density import ChargeDensityInputError
from mufinder.castep_io.output_parser import parse_castep_mulliken_atomic_spins
from mufinder.domain.constants import (
    ELEMENTARY_CHARGE_C,
    HBAR_J_S,
    MUON_MASS_KG,
    VACUUM_PERMITTIVITY_F_PER_M,
)
from mufinder.science.lattice import cartesian_to_fractional, fractional_to_cartesian


_COULOMB_PREFactor = 1.0 / (4.0 * np.pi * VACUUM_PERMITTIVITY_F_PER_M)
_ISOSURFACE_PREVIEW_MAX_GRID_POINTS = 18


def _pyvista_modules():
    import pyvista as pv

    return pv


#: The gradient norm, in eV/Å, at which a search start counts as converged.
DEFAULT_FORCE_TOLERANCE_EV_PER_ANG = 1e-5
#: Below this σ·G_max the ionic Gaussian keeps over 13 % of its weight at the edge of the G
#: sphere. On the fixtures and CoF₂ a batch minimum then moves or the potential rings.
IONIC_GAUSSIAN_MIN_RESOLUTION = 2.0
POTENTIAL_LEVEL_CEILING_PERCENTILE = 95.0
#: The iteration limit of one search start.
DEFAULT_OPTIMIZER_MAX_STEPS = 300
#: How far, relative to the ionic charge, the density's cell average may sit from it before
#: the host counts as charged. 1e-4 is the relative tolerance of the published UEP method's
#: implementation (Sturniolo and Liborio, J. Chem. Phys. 153, 044111 (2020)); the density is
#: renormalised to the ionic charge afterwards, so the check only screens out a charged cell.
DENSITY_NEUTRALITY_RELATIVE_TOLERANCE = 1e-4


class UepError(ValueError):
    """Raised when UEP source data or analysis is invalid."""


@dataclass(frozen=True, slots=True)
class UepStructure:
    """Host structure information loaded from CASTEP output."""

    lattice_vectors: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]
    species: tuple[str, ...]
    positions_ang: tuple[tuple[float, float, float], ...]
    fractional_positions: tuple[tuple[float, float, float], ...]
    atomic_spins: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class UepSourceData:
    """Fully validated UEP source bundle."""

    density: _charge_density.FormattedDensity
    structure: UepStructure
    pseudopotentials: tuple[_charge_density.IonicPseudopotential, ...]
    castep_path: Path
    den_fmt_path: Path
    cell_path: Path | None
    gw_factor: float
    neutrality_warning: str | None = None


@dataclass(frozen=True, slots=True)
class StationaryPointClassification:
    """Local Hessian-based classification of a probed point."""

    label: str
    eigenvalues: tuple[float, float, float]
    positive_count: int
    negative_count: int
    near_zero_count: int


@dataclass(frozen=True, slots=True)
class UepProbeResult:
    """Direct UEP evaluation at a single Cartesian position."""

    position_ang: tuple[float, float, float]
    fractional_position: tuple[float, float, float]
    potential_total_ev: float
    potential_electronic_ev: float
    potential_ionic_ev: float
    gradient_total_ev_per_ang: tuple[float, float, float]
    gradient_electronic_ev_per_ang: tuple[float, float, float]
    gradient_ionic_ev_per_ang: tuple[float, float, float]
    hessian_total_ev_per_ang2: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]
    hessian_electronic_ev_per_ang2: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]
    hessian_ionic_ev_per_ang2: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]
    classification: StationaryPointClassification
    zero_point_energy_ev: float | None
    total_quantum_energy_ev: float | None


@dataclass(frozen=True, slots=True)
class UepOptimizationResult:
    """One optimized muon position under the UEP approximation."""

    start_position_ang: tuple[float, float, float]
    optimized_position_ang: tuple[float, float, float]
    optimized_fractional_position: tuple[float, float, float]
    classical_energy_ev: float
    zero_point_energy_ev: float | None
    total_energy_ev: float | None
    success: bool
    message: str
    iterations: int
    classification: StationaryPointClassification


@dataclass(frozen=True, slots=True)
class UepPotentialGrid:
    """Real-space total electrostatic potential sampled on the FFT grid."""

    lattice_vectors: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]
    grid: tuple[int, int, int]
    potential_total_ev: np.ndarray
    gw_factor: float | None = None


@dataclass(frozen=True, slots=True)
class UepIsosurfaceMesh:
    """Triangle mesh for one electrostatic-potential isosurface."""

    level_ev: float
    minimum_potential_ev: float
    maximum_potential_ev: float
    vertices_ang: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class UepSlicePlane:
    """Geometric definition of one sampled UEP slice plane."""

    anchor_ang: tuple[float, float, float]
    normal_ang: tuple[float, float, float]
    basis_u_ang: tuple[float, float, float]
    basis_v_ang: tuple[float, float, float]
    extent_u_ang: tuple[float, float]
    extent_v_ang: tuple[float, float]
    points_u: int
    points_v: int


@dataclass(frozen=True, slots=True)
class UepGridSliceSample:
    """Potential sampled from a cached periodic grid on one plane patch."""

    plane: UepSlicePlane
    x_coordinates_ang: tuple[tuple[float, ...], ...]
    y_coordinates_ang: tuple[tuple[float, ...], ...]
    z_coordinates_ang: tuple[tuple[float, ...], ...]
    potential_total_ev: tuple[tuple[float, ...], ...]
    minimum_potential_ev: float
    maximum_potential_ev: float


def triple_from_payload(values: object) -> tuple[float, float, float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    try:
        return (float(values[0]), float(values[1]), float(values[2]))
    except (TypeError, ValueError):
        return None


def classification_from_dict(payload: object) -> StationaryPointClassification | None:
    """Rebuild a saved stationary-point classification; ``None`` when it is malformed."""

    if not isinstance(payload, Mapping):
        return None
    eigenvalues = triple_from_payload(payload.get("eigenvalues"))
    if eigenvalues is None:
        return None
    try:
        return StationaryPointClassification(
            label=str(payload.get("label", "")).strip(),
            eigenvalues=eigenvalues,
            positive_count=int(payload.get("positive_count", 0)),
            negative_count=int(payload.get("negative_count", 0)),
            near_zero_count=int(payload.get("near_zero_count", 0)),
        )
    except (TypeError, ValueError):
        return None


def optimization_result_from_dict(payload: object) -> UepOptimizationResult | None:
    """Rebuild a saved optimisation result; ``None`` when it is malformed."""

    if not isinstance(payload, Mapping):
        return None
    start = triple_from_payload(payload.get("start_position_ang"))
    optimized = triple_from_payload(payload.get("optimized_position_ang"))
    fractional = triple_from_payload(payload.get("optimized_fractional_position"))
    classification = classification_from_dict(payload.get("classification"))
    if start is None or optimized is None or fractional is None or classification is None:
        return None
    zero_point = payload.get("zero_point_energy_ev")
    total = payload.get("total_energy_ev")
    try:
        return UepOptimizationResult(
            start_position_ang=start,
            optimized_position_ang=optimized,
            optimized_fractional_position=fractional,
            classical_energy_ev=float(payload.get("classical_energy_ev", 0.0)),
            zero_point_energy_ev=None if zero_point is None else float(zero_point),
            total_energy_ev=None if total is None else float(total),
            success=bool(payload.get("success", False)),
            message=str(payload.get("message", "")),
            iterations=int(payload.get("iterations", 0)),
            classification=classification,
        )
    except (TypeError, ValueError):
        return None


def write_potential_grid_xsf(
    path: str | Path,
    *,
    structure: UepStructure,
    potential: UepPotentialGrid,
) -> Path:
    """Write a periodic structure plus 3D electrostatic-potential grid to XSF.

    XSF volumetric data is defined as a general (non-periodic) grid. To export a
    periodic FFT mesh consistently, duplicate the periodic boundary planes so the
    written grid spans the full cell with redundant end points along each axis.
    """

    values = np.asarray(potential.potential_total_ev, dtype=float)
    if values.ndim != 3 or min(values.shape) < 1:
        raise UepError("XSF export needs a 3D electrostatic-potential grid.")

    lattice = np.asarray(potential.lattice_vectors, dtype=float)
    if lattice.shape != (3, 3):
        raise UepError("XSF export needs three lattice vectors.")
    if len(structure.species) != len(structure.positions_ang):
        raise UepError("XSF export needs one Cartesian position per structure species.")

    target = Path(path)
    xsf_values = np.pad(values, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    nx, ny, nz = (int(size) for size in xsf_values.shape)
    flattened_values = xsf_values.flatten(order="F")

    lines = [
        "CRYSTAL",
        "PRIMVEC",
        *(f"{vector[0]:.16f} {vector[1]:.16f} {vector[2]:.16f}" for vector in lattice),
        "PRIMCOORD",
        f"{len(structure.species)} 1",
    ]
    for symbol, position in zip(structure.species, structure.positions_ang, strict=True):
        atomic_number = int(atomic_numbers.get(str(symbol).capitalize(), 0))
        lines.append(
            f"{atomic_number} {float(position[0]):.16f} {float(position[1]):.16f} {float(position[2]):.16f}"
        )

    lines.extend(
        [
            "BEGIN_BLOCK_DATAGRID_3D",
            "UEP_POTENTIAL",
            "BEGIN_DATAGRID_3D_UEP_POTENTIAL",
            f"{nx} {ny} {nz}",
            "0.0000000000000000 0.0000000000000000 0.0000000000000000",
            *(f"{vector[0]:.16f} {vector[1]:.16f} {vector[2]:.16f}" for vector in lattice),
        ]
    )
    for offset in range(0, len(flattened_values), 6):
        chunk = flattened_values[offset : offset + 6]
        lines.append(" ".join(f"{float(value):.16e}" for value in chunk))
    lines.extend(
        [
            "END_DATAGRID_3D_UEP_POTENTIAL",
            "END_BLOCK_DATAGRID_3D",
            "",
        ]
    )

    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def parse_formatted_density(path: str | Path) -> _charge_density.FormattedDensity:
    """Parse a CASTEP `.den_fmt` file, raising ``UepError`` when it is unusable.

    Wraps ``mufinder.castep_io.charge_density.parse_formatted_density``.
    """

    try:
        return _charge_density.parse_formatted_density(path)
    except ChargeDensityInputError as exc:
        raise UepError(str(exc)) from exc


def parse_castep_pseudopotentials(
    path: str | Path,
) -> dict[str, _charge_density.IonicPseudopotential]:
    """Parse CASTEP pseudopotential reports, raising ``UepError`` when none is usable.

    Wraps ``mufinder.castep_io.charge_density.parse_castep_pseudopotentials``.
    """

    try:
        return _charge_density.parse_castep_pseudopotentials(path)
    except ChargeDensityInputError as exc:
        raise UepError(str(exc)) from exc


def _merge_cell_pseudopotential_metadata(
    *,
    parsed: dict[str, _charge_density.IonicPseudopotential],
    cell_path: Path,
) -> dict[str, _charge_density.IonicPseudopotential]:
    """Augment pseudopotential metadata using SPECIES_POT references from a CELL file."""

    try:
        parsed_cell = parse_cell_file(cell_path)
    except CellParseError as exc:
        raise UepError(f"Failed to parse CELL metadata from '{cell_path.name}': {exc}") from exc

    species_potentials = parsed_cell.cell_file.species_pot
    if not species_potentials:
        return parsed

    merged = dict(parsed)
    cell_dir = cell_path.parent
    for row in species_potentials:
        if row.species in merged:
            continue
        potential_path = Path(row.potential)
        if not potential_path.is_absolute():
            potential_path = cell_dir / potential_path
        if not potential_path.exists():
            continue
        try:
            merged.update(parse_castep_pseudopotentials(potential_path))
        except UepError:
            continue
    return merged


def load_uep_source(
    *,
    den_fmt_path: str | Path,
    castep_path: str | Path,
    cell_path: str | Path | None = None,
    gw_factor: float = 5.0,
) -> UepSourceData:
    """Load and validate the files required for UEP analysis."""

    density_path = Path(den_fmt_path)
    output_path = Path(castep_path)
    optional_cell_path = Path(cell_path) if cell_path is not None else None

    if gw_factor <= 0.0:
        raise UepError("gw_factor must be positive.")

    density = parse_formatted_density(density_path)
    try:
        atoms = io.read(output_path)
    except Exception as exc:  # noqa: BLE001
        raise UepError(f"Failed to read host structure from '{output_path.name}': {exc}") from exc

    try:
        pseudopotentials = parse_castep_pseudopotentials(output_path)
    except UepError:
        pseudopotentials = {}

    if optional_cell_path is None:
        candidate = output_path.with_suffix(".cell")
        optional_cell_path = candidate if candidate.exists() else None

    if optional_cell_path is not None and optional_cell_path.exists():
        pseudopotentials = _merge_cell_pseudopotential_metadata(
            parsed=pseudopotentials,
            cell_path=optional_cell_path,
        )

    total_charge = _charge_density.castep_total_charge(output_path)
    neutrality_warning = None
    if total_charge is not None and abs(total_charge) > 1e-3:
        raise UepError(
            f"UEP requires a neutral host cell, but '{output_path.name}' reports total charge {total_charge:.6g}."
        )
    if total_charge is None:
        neutrality_warning = (
            "CASTEP output did not report total cell charge explicitly; assuming a neutral host."
        )

    positions_ang = tuple(
        tuple(float(value) for value in row) for row in atoms.get_positions().tolist()
    )
    fractional_positions = tuple(
        tuple(float(value) for value in row)
        for row in atoms.get_scaled_positions(wrap=True).tolist()
    )
    atomic_spins = parse_castep_mulliken_atomic_spins(output_path)
    structure = UepStructure(
        lattice_vectors=(
            tuple(float(value) for value in atoms.cell[0]),
            tuple(float(value) for value in atoms.cell[1]),
            tuple(float(value) for value in atoms.cell[2]),
        ),
        species=tuple(str(symbol) for symbol in atoms.get_chemical_symbols()),
        positions_ang=positions_ang,
        fractional_positions=fractional_positions,
        atomic_spins=atomic_spins,
    )

    missing_species = sorted(
        {symbol for symbol in structure.species if symbol not in pseudopotentials}
    )
    if missing_species:
        joined = ", ".join(missing_species)
        raise UepError(
            "Missing pseudopotential metadata for species "
            f"{joined}. UEP analysis cannot continue without ionic charge and core-radius data."
        )

    ionic_charge_total = sum(pseudopotentials[symbol].ionic_charge for symbol in structure.species)
    average_density = float(np.average(density.values[:, :, :, 0]))
    if not np.isclose(
        average_density,
        ionic_charge_total,
        rtol=DENSITY_NEUTRALITY_RELATIVE_TOLERANCE,
        atol=0.0,
    ):
        raise UepError(
            "UEP requires a neutral host cell, but the formatted density integrates to "
            f"{average_density:.6g} e while the ionic model requires {ionic_charge_total:.6g} e."
        )

    return UepSourceData(
        density=density,
        structure=structure,
        pseudopotentials=tuple(pseudopotentials[symbol] for symbol in structure.species),
        castep_path=output_path,
        den_fmt_path=density_path,
        cell_path=optional_cell_path
        if optional_cell_path is not None and optional_cell_path.exists()
        else None,
        gw_factor=float(gw_factor),
        neutrality_warning=neutrality_warning,
    )


class UepField:
    """Reciprocal-space UEP evaluator for one host density."""

    def __init__(self, source: UepSourceData) -> None:
        self._source = source
        lattice = np.array(source.density.lattice_vectors, dtype=float)
        grid = np.array(source.density.grid, dtype=int)
        electron_density = np.array(source.density.values[:, :, :, 0], dtype=float)
        ionic_charge = sum(item.ionic_charge for item in source.pseudopotentials)
        if ionic_charge <= 0.0:
            raise UepError("Total ionic charge must be positive.")
        if np.sum(electron_density) == 0.0:
            raise UepError("Formatted density integrates to zero and cannot be normalized for UEP.")

        electron_density *= -ionic_charge / np.sum(electron_density)

        reciprocal_lattice = np.linalg.inv(lattice.T) * (2.0 * np.pi)
        fft_grid = np.array(
            np.meshgrid(
                *[np.fft.fftfreq(int(grid[index])) * int(grid[index]) for index in range(3)],
                indexing="ij",
            )
        )
        g_grid = np.tensordot(reciprocal_lattice, fft_grid, axes=(0, 0))
        g_norm = np.linalg.norm(g_grid, axis=0)
        g_norm_safe = np.where(g_norm > 0.0, g_norm, np.inf)

        volume = abs(np.dot(np.cross(lattice[0], lattice[1]), lattice[2]))
        electron_density_g = np.fft.fftn(electron_density)
        electron_potential_g = 4.0 * np.pi / (g_norm_safe**2) * (electron_density_g / volume)

        ionic_density_g = np.zeros(g_grid.shape[1:], dtype=complex)
        positions = np.array(source.structure.positions_ang, dtype=float)
        widths = np.array(
            [pp.core_radius_ang / source.gw_factor for pp in source.pseudopotentials],
            dtype=float,
        )
        ionic_charges = np.array([pp.ionic_charge for pp in source.pseudopotentials], dtype=float)
        for index, position in enumerate(positions):
            phase = np.sum(g_grid * position[:, None, None, None], axis=0)
            ionic_density_g += ionic_charges[index] * np.exp(
                -1.0j * phase - 0.5 * (widths[index] * g_norm) ** 2
            )
        ionic_potential_g = 4.0 * np.pi / (g_norm_safe**2) * (ionic_density_g / volume)

        self._lattice = lattice
        self._volume = float(volume)
        self._g_grid = g_grid
        self._g_norm = g_norm
        self._electron_density = electron_density
        self._electron_density_g = electron_density_g
        self._ionic_density_g = ionic_density_g
        self._electron_potential_g = electron_potential_g
        self._ionic_potential_g = ionic_potential_g

    @property
    def source(self) -> UepSourceData:
        return self._source

    @property
    def lattice_vectors(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        return self._source.structure.lattice_vectors

    def fractional_position(
        self, position_ang: tuple[float, float, float] | np.ndarray
    ) -> tuple[float, float, float]:
        position = np.asarray(position_ang, dtype=float)
        fractional = np.mod(np.linalg.solve(self._lattice.T, position), 1.0)
        return tuple(float(value) for value in fractional)

    def _validate_points(self, points: object) -> np.ndarray:
        array = np.asarray(points, dtype=float)
        if array.ndim == 1:
            if array.shape[0] != 3:
                raise UepError("Cartesian probe positions must have exactly three components.")
            return array[None, :]
        if array.ndim != 2 or array.shape[1] != 3:
            raise UepError("Cartesian probe positions must have shape (N, 3).")
        return array

    def _fourier_kernel(self, points: np.ndarray) -> np.ndarray:
        return np.exp(1.0j * np.tensordot(self._g_grid, points.T, axes=(0, 0)))

    def charge_density(self, points: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return total/electronic/ionic charge density in e/Ang^3."""

        coords = self._validate_points(points)
        kernel = self._fourier_kernel(coords)
        rho_electronic = (
            np.real(np.sum(self._electron_density_g[:, :, :, None] * kernel, axis=(0, 1, 2)))
            / self._volume
        )
        rho_ionic = (
            np.real(np.sum(self._ionic_density_g[:, :, :, None] * kernel, axis=(0, 1, 2)))
            / self._volume
        )
        return (rho_electronic + rho_ionic, rho_electronic, rho_ionic)

    def potential(self, points: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return total/electronic/ionic potential in eV for a unit positive charge."""

        coords = self._validate_points(points)
        kernel = self._fourier_kernel(coords)
        potential_electronic = np.real(
            np.sum(self._electron_potential_g[:, :, :, None] * kernel, axis=(0, 1, 2))
        )
        potential_ionic = np.real(
            np.sum(self._ionic_potential_g[:, :, :, None] * kernel, axis=(0, 1, 2))
        )
        scale = _COULOMB_PREFactor * ELEMENTARY_CHARGE_C * 1e10
        potential_electronic *= scale
        potential_ionic *= scale
        return (potential_electronic + potential_ionic, potential_electronic, potential_ionic)

    def gradient(self, points: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return total/electronic/ionic potential gradients in eV/Ang."""

        coords = self._validate_points(points)
        kernel = self._fourier_kernel(coords)
        derivative_kernel = 1.0j * self._g_grid[:, :, :, :, None] * kernel[None, :, :, :, :]
        gradient_electronic = np.real(
            np.sum(
                self._electron_potential_g[None, :, :, :, None] * derivative_kernel, axis=(1, 2, 3)
            )
        ).T
        gradient_ionic = np.real(
            np.sum(self._ionic_potential_g[None, :, :, :, None] * derivative_kernel, axis=(1, 2, 3))
        ).T
        # Match the Cartesian derivative of ``potential()`` with respect to
        # Angstrom coordinates, as exposed by pymuon-suite after its optimizer
        # length-unit conversion.
        scale = _COULOMB_PREFactor * ELEMENTARY_CHARGE_C * 1e10
        gradient_electronic *= scale
        gradient_ionic *= scale
        return (gradient_electronic + gradient_ionic, gradient_electronic, gradient_ionic)

    def hessian(self, points: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return total/electronic/ionic Hessians in eV/Ang^2."""

        coords = self._validate_points(points)
        kernel = self._fourier_kernel(coords)
        g_outer = self._g_grid[:, None, :, :, :] * self._g_grid[None, :, :, :, :]
        second_kernel = -g_outer[:, :, :, :, :, None] * kernel[None, None, :, :, :, :]
        hessian_electronic = np.real(
            np.sum(
                self._electron_potential_g[None, None, :, :, :, None] * second_kernel,
                axis=(2, 3, 4),
            )
        ).T
        hessian_ionic = np.real(
            np.sum(
                self._ionic_potential_g[None, None, :, :, :, None] * second_kernel, axis=(2, 3, 4)
            )
        ).T
        # Match the Cartesian second derivative of ``potential()`` with respect
        # to Angstrom coordinates, as exposed by pymuon-suite after its
        # optimizer length-unit conversion.
        scale = _COULOMB_PREFactor * ELEMENTARY_CHARGE_C * 1e10
        hessian_electronic *= scale
        hessian_ionic *= scale
        return (hessian_electronic + hessian_ionic, hessian_electronic, hessian_ionic)


def potential_grid(field: UepField) -> UepPotentialGrid:
    """Return the total electrostatic potential on the source FFT grid."""

    grid = tuple(int(value) for value in field.source.density.grid)
    point_count = int(np.prod(grid))
    scale = _COULOMB_PREFactor * ELEMENTARY_CHARGE_C * 1e10
    total = (
        np.real(np.fft.ifftn(field._electron_potential_g + field._ionic_potential_g))
        * point_count
        * scale
    )
    return UepPotentialGrid(
        lattice_vectors=field.lattice_vectors,
        grid=grid,
        potential_total_ev=np.asarray(total, dtype=float),
        gw_factor=field.source.gw_factor,
    )


def _stride_for_axis(size: int, limit: int) -> int:
    """The stride that leaves ``size`` periodic points closest, by ratio, to ``limit``.

    A stride that divides ``size`` keeps the sub-lattice's spacing uniform across the
    periodic boundary, so an axis of 25 points stays at 25 rather than falling to 5, and a
    prime size keeps every point. CASTEP's FFT grids have small factors.
    """

    divisors = [d for d in range(1, size) if size % d == 0 and size // d >= 2]
    return min(divisors, key=lambda d: (abs(np.log((size // d) / limit)), d), default=1)


def preview_potential_grid(
    potential: UepPotentialGrid,
    *,
    max_grid_points: int = _ISOSURFACE_PREVIEW_MAX_GRID_POINTS,
) -> UepPotentialGrid:
    """Return a coarse sub-lattice of ``potential`` for an isosurface preview while dragging.

    Every value is one of the grid's own, so a surface drawn from it sits at the level
    asked for at each of its points. Fourier truncation would ring around the nuclear
    spikes once the coarser grid no longer resolves them, by several eV in the range of
    interest.
    """

    values = np.asarray(potential.potential_total_ev, dtype=float)
    if values.ndim != 3 or min(values.shape) < 2:
        raise UepError(
            "Isosurface previews need a 3D potential grid with at least two points per axis."
        )

    limit = max(2, int(max_grid_points))
    strides = tuple(_stride_for_axis(int(size), limit) for size in values.shape)
    reduced = values[:: strides[0], :: strides[1], :: strides[2]].copy()
    return UepPotentialGrid(
        lattice_vectors=potential.lattice_vectors,
        grid=tuple(int(size) for size in reduced.shape),
        potential_total_ev=reduced,
        gw_factor=potential.gw_factor,
    )


def potential_above_minimum(potential: UepPotentialGrid) -> UepPotentialGrid:
    """Return ``potential`` measured from its lowest grid value.

    The potential's own zero is its cell average, because the G = 0 term is dropped, so
    only differences between its values mean anything.
    """

    values = np.asarray(potential.potential_total_ev, dtype=float)
    return UepPotentialGrid(
        lattice_vectors=potential.lattice_vectors,
        grid=potential.grid,
        potential_total_ev=values - float(np.min(values)),
        gw_factor=potential.gw_factor,
    )


def potential_level_ceiling(potential: UepPotentialGrid) -> float:
    """Return the top of a useful level range: the grid's 95th percentile.

    The nuclear spikes take the maximum far above the region a muon can reach. On a grid
    too flat for the percentile to rise above the minimum, the maximum is returned.
    """

    values = np.asarray(potential.potential_total_ev, dtype=float)
    ceiling = float(np.percentile(values, POTENTIAL_LEVEL_CEILING_PERCENTILE))
    return ceiling if ceiling > float(np.min(values)) else float(np.max(values))


def ionic_gaussian_resolution(field: UepField) -> float:
    """Return σ·G_max for the narrowest ionic Gaussian on the field's FFT grid.

    G_max is the radius of the sphere inscribed in the grid's G box. The Gaussian's
    transform falls to exp(-½(σ·G_max)²) there, so a small value means the grid truncates
    the ionic charge and the potential rings around the ions.
    """

    source = field.source
    lattice = np.asarray(source.density.lattice_vectors, dtype=float)
    grid = np.asarray(source.density.grid, dtype=float)
    g_max = float(np.min((grid // 2) * (2.0 * np.pi) / np.linalg.norm(lattice, axis=1)))
    width = min(pp.core_radius_ang for pp in source.pseudopotentials) / source.gw_factor
    return float(width) * g_max


def _slice_lattice(lattice_vectors: object) -> np.ndarray:
    lattice = np.asarray(lattice_vectors, dtype=float)
    if lattice.shape != (3, 3):
        raise UepError("Slice sampling needs three lattice vectors.")
    return lattice


def fractional_point_to_cartesian(
    lattice_vectors: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ],
    fractional_position: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    """Convert one fractional point into Cartesian coordinates."""

    point = np.asarray(fractional_position, dtype=float)
    if point.shape != (3,):
        raise UepError("Fractional positions must be 3-vectors.")
    cartesian = fractional_to_cartesian(point, _slice_lattice(lattice_vectors))
    return tuple(float(value) for value in cartesian)


def miller_indices_to_cartesian_normal(
    lattice_vectors: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ],
    miller_indices: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    """Convert Miller indices into the corresponding Cartesian plane normal."""

    lattice = _slice_lattice(lattice_vectors)
    miller = np.asarray(miller_indices, dtype=float)
    if miller.shape != (3,):
        raise UepError("Miller indices must be 3-vectors.")
    if float(np.linalg.norm(miller)) <= 1e-12:
        raise UepError("Miller indices must not all be zero.")
    normal = np.linalg.solve(lattice, miller)
    return normalize_plane_normal(normal)


def cartesian_normal_to_lattice_plane_components(
    lattice_vectors: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ],
    normal_ang: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    """Convert a Cartesian plane normal into normalized lattice-plane components."""

    lattice = _slice_lattice(lattice_vectors)
    normal = np.asarray(normalize_plane_normal(normal_ang), dtype=float)
    components = lattice @ normal
    scale = float(np.max(np.abs(components)))
    if scale <= 1e-12:
        raise UepError("Plane normals must have non-zero length.")
    normalized = components / scale
    return tuple(float(value) for value in normalized)


def _interpolate_periodic_grid_trilinear(
    potential: UepPotentialGrid,
    positions_ang: np.ndarray,
) -> np.ndarray:
    values = np.asarray(potential.potential_total_ev, dtype=float)
    if values.ndim != 3 or min(values.shape) < 2:
        raise UepError(
            "Slice sampling needs a 3D potential grid with at least two points per axis."
        )

    fractional = cartesian_to_fractional(
        positions_ang, _slice_lattice(potential.lattice_vectors), error=UepError
    )
    wrapped = np.mod(fractional, 1.0)
    grid_shape = np.asarray(values.shape, dtype=float)
    scaled = wrapped * grid_shape.reshape(1, 3)
    lower = np.floor(scaled).astype(int)
    weights = scaled - lower.astype(float)
    upper = (lower + 1) % np.asarray(values.shape, dtype=int).reshape(1, 3)
    lower = lower % np.asarray(values.shape, dtype=int).reshape(1, 3)

    ix0, iy0, iz0 = lower[:, 0], lower[:, 1], lower[:, 2]
    ix1, iy1, iz1 = upper[:, 0], upper[:, 1], upper[:, 2]
    wx, wy, wz = weights[:, 0], weights[:, 1], weights[:, 2]

    c000 = values[ix0, iy0, iz0]
    c100 = values[ix1, iy0, iz0]
    c010 = values[ix0, iy1, iz0]
    c110 = values[ix1, iy1, iz0]
    c001 = values[ix0, iy0, iz1]
    c101 = values[ix1, iy0, iz1]
    c011 = values[ix0, iy1, iz1]
    c111 = values[ix1, iy1, iz1]

    c00 = ((1.0 - wx) * c000) + (wx * c100)
    c10 = ((1.0 - wx) * c010) + (wx * c110)
    c01 = ((1.0 - wx) * c001) + (wx * c101)
    c11 = ((1.0 - wx) * c011) + (wx * c111)
    c0 = ((1.0 - wy) * c00) + (wy * c10)
    c1 = ((1.0 - wy) * c01) + (wy * c11)
    return ((1.0 - wz) * c0) + (wz * c1)


def normalize_plane_normal(
    normal_ang: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    """Return a normalized plane normal or raise if it is degenerate."""

    normal = np.asarray(normal_ang, dtype=float)
    if normal.shape != (3,):
        raise UepError("Plane normals must be 3-vectors.")
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise UepError("Plane normals must have non-zero length.")
    return tuple(float(value) for value in (normal / norm))


def plane_basis_from_normal(
    normal_ang: tuple[float, float, float] | np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Build an orthonormal in-plane basis for a plane normal."""

    normal = np.asarray(normalize_plane_normal(normal_ang), dtype=float)
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(normal, reference))) > 0.92:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    basis_u = np.cross(normal, reference)
    basis_u /= max(float(np.linalg.norm(basis_u)), 1e-12)
    basis_v = np.cross(normal, basis_u)
    basis_v /= max(float(np.linalg.norm(basis_v)), 1e-12)
    return (
        tuple(float(value) for value in basis_u),
        tuple(float(value) for value in basis_v),
    )


def plane_from_point_and_normal(
    *,
    anchor_ang: tuple[float, float, float] | np.ndarray,
    normal_ang: tuple[float, float, float] | np.ndarray,
    lattice_vectors: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ],
    points_u: int = 161,
    points_v: int = 161,
) -> UepSlicePlane:
    """Define a cell-fitted slice plane from one anchor point and a normal."""

    anchor = np.asarray(anchor_ang, dtype=float)
    if anchor.shape != (3,):
        raise UepError("Slice-plane anchors must be 3-vectors.")
    normal = np.asarray(normalize_plane_normal(normal_ang), dtype=float)
    basis_u, basis_v = plane_basis_from_normal(normal)
    basis_u_vec = np.asarray(basis_u, dtype=float)
    basis_v_vec = np.asarray(basis_v, dtype=float)
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            lattice_vectors[0],
            lattice_vectors[1],
            lattice_vectors[2],
            np.asarray(lattice_vectors[0], dtype=float)
            + np.asarray(lattice_vectors[1], dtype=float),
            np.asarray(lattice_vectors[0], dtype=float)
            + np.asarray(lattice_vectors[2], dtype=float),
            np.asarray(lattice_vectors[1], dtype=float)
            + np.asarray(lattice_vectors[2], dtype=float),
            np.asarray(lattice_vectors[0], dtype=float)
            + np.asarray(lattice_vectors[1], dtype=float)
            + np.asarray(lattice_vectors[2], dtype=float),
        ],
        dtype=float,
    )
    relative = corners - anchor.reshape(1, 3)
    extent_u = (
        float(np.min(relative @ basis_u_vec)),
        float(np.max(relative @ basis_u_vec)),
    )
    extent_v = (
        float(np.min(relative @ basis_v_vec)),
        float(np.max(relative @ basis_v_vec)),
    )
    return UepSlicePlane(
        anchor_ang=tuple(float(value) for value in anchor),
        normal_ang=tuple(float(value) for value in normal),
        basis_u_ang=basis_u,
        basis_v_ang=basis_v,
        extent_u_ang=extent_u,
        extent_v_ang=extent_v,
        points_u=int(points_u),
        points_v=int(points_v),
    )


def plane_from_fractional_anchor_and_miller(
    *,
    anchor_fractional: tuple[float, float, float] | np.ndarray,
    miller_indices: tuple[float, float, float] | np.ndarray,
    lattice_vectors: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ],
    points_u: int = 161,
    points_v: int = 161,
) -> UepSlicePlane:
    """Define a cell-fitted slice plane from a fractional anchor and Miller indices."""

    anchor_ang = fractional_point_to_cartesian(lattice_vectors, anchor_fractional)
    normal_ang = miller_indices_to_cartesian_normal(lattice_vectors, miller_indices)
    return plane_from_point_and_normal(
        anchor_ang=anchor_ang,
        normal_ang=normal_ang,
        lattice_vectors=lattice_vectors,
        points_u=points_u,
        points_v=points_v,
    )


def plane_from_three_points(
    *,
    point_a_ang: tuple[float, float, float] | np.ndarray,
    point_b_ang: tuple[float, float, float] | np.ndarray,
    point_c_ang: tuple[float, float, float] | np.ndarray,
    lattice_vectors: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ],
    points_u: int = 161,
    points_v: int = 161,
) -> UepSlicePlane:
    """Define a cell-fitted slice plane through three non-collinear points."""

    point_a = np.asarray(point_a_ang, dtype=float)
    point_b = np.asarray(point_b_ang, dtype=float)
    point_c = np.asarray(point_c_ang, dtype=float)
    normal = np.cross(point_b - point_a, point_c - point_a)
    if float(np.linalg.norm(normal)) <= 1e-12:
        raise UepError("Three atoms must not be collinear when defining a plane.")
    anchor = (point_a + point_b + point_c) / 3.0
    return plane_from_point_and_normal(
        anchor_ang=anchor,
        normal_ang=normal,
        lattice_vectors=lattice_vectors,
        points_u=points_u,
        points_v=points_v,
    )


def plane_from_best_fit(
    *,
    points_ang: tuple[tuple[float, float, float], ...]
    | list[tuple[float, float, float]]
    | np.ndarray,
    lattice_vectors: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ],
    points_u: int = 161,
    points_v: int = 161,
) -> UepSlicePlane:
    """Define a cell-fitted least-squares plane through four or more points."""

    points = np.asarray(points_ang, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise UepError("Best-fit plane construction needs at least four Cartesian points.")
    anchor = np.mean(points, axis=0)
    centered = points - anchor.reshape(1, 3)
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if len(singular_values) < 2 or singular_values[1] <= 1e-12:
        raise UepError(
            "Best-fit plane construction failed because the selected atoms are collinear."
        )
    return plane_from_point_and_normal(
        anchor_ang=anchor,
        normal_ang=normal,
        lattice_vectors=lattice_vectors,
        points_u=points_u,
        points_v=points_v,
    )


def sample_grid_slice(
    potential: UepPotentialGrid,
    plane: UepSlicePlane,
) -> UepGridSliceSample:
    """Sample a cached periodic potential grid on a defined slice plane."""

    if int(plane.points_u) < 2 or int(plane.points_v) < 2:
        raise UepError("Slice sampling needs at least two points along each axis.")
    anchor = np.asarray(plane.anchor_ang, dtype=float)
    basis_u = np.asarray(plane.basis_u_ang, dtype=float)
    basis_v = np.asarray(plane.basis_v_ang, dtype=float)
    xs = np.linspace(
        float(plane.extent_u_ang[0]), float(plane.extent_u_ang[1]), int(plane.points_u)
    )
    ys = np.linspace(
        float(plane.extent_v_ang[0]), float(plane.extent_v_ang[1]), int(plane.points_v)
    )
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    points = (
        anchor.reshape(1, 1, 3)
        + (x_grid[..., np.newaxis] * basis_u.reshape(1, 1, 3))
        + (y_grid[..., np.newaxis] * basis_v.reshape(1, 1, 3))
    )
    flat_points = points.reshape((-1, 3))
    sampled = _interpolate_periodic_grid_trilinear(potential, flat_points).reshape(
        (int(plane.points_u), int(plane.points_v))
    )
    return UepGridSliceSample(
        plane=plane,
        x_coordinates_ang=tuple(tuple(float(value) for value in row) for row in points[:, :, 0]),
        y_coordinates_ang=tuple(tuple(float(value) for value in row) for row in points[:, :, 1]),
        z_coordinates_ang=tuple(tuple(float(value) for value in row) for row in points[:, :, 2]),
        potential_total_ev=tuple(tuple(float(value) for value in row) for row in sampled),
        minimum_potential_ev=float(np.min(sampled)),
        maximum_potential_ev=float(np.max(sampled)),
    )


def extract_isosurface(
    potential: UepPotentialGrid,
    *,
    level_ev: float,
) -> UepIsosurfaceMesh:
    """Extract an isosurface mesh from a real-space electrostatic-potential grid.

    PyVista contours the periodic grid. A level no surface crosses raises
    :class:`UepError`: the grid's minimum always does, even where a flat region lies at it.
    """

    values = np.asarray(potential.potential_total_ev, dtype=float)
    if values.ndim != 3 or min(values.shape) < 2:
        raise UepError(
            "Isosurface extraction needs a 3D potential grid with at least two points per axis."
        )

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if level_ev < minimum or level_ev > maximum:
        raise UepError(
            f"Isosurface level {level_ev:.6g} eV is outside the available range [{minimum:.6g}, {maximum:.6g}] eV."
        )

    pyvista_mesh = _extract_isosurface_with_pyvista(
        potential,
        level_ev=level_ev,
        minimum=minimum,
        maximum=maximum,
    )
    if pyvista_mesh is None:
        raise UepError(
            f"No isosurface intersects the cell at {level_ev:.6g} eV. Choose a value closer to the extrema."
        )
    return pyvista_mesh


def _extract_isosurface_with_pyvista(
    potential: UepPotentialGrid,
    *,
    level_ev: float,
    minimum: float,
    maximum: float,
) -> UepIsosurfaceMesh | None:
    pv = _pyvista_modules()

    values = np.asarray(potential.potential_total_ev, dtype=float)
    nx, ny, nz = (int(size) for size in values.shape)
    extended_values = np.pad(values, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    frac_x = np.arange(nx + 1, dtype=float) / float(nx)
    frac_y = np.arange(ny + 1, dtype=float) / float(ny)
    frac_z = np.arange(nz + 1, dtype=float) / float(nz)
    fractional_points = np.stack(np.meshgrid(frac_x, frac_y, frac_z, indexing="ij"), axis=-1)
    lattice = np.asarray(potential.lattice_vectors, dtype=float)
    cartesian_points = np.einsum("...i,ij->...j", fractional_points, lattice, optimize=True)
    try:
        grid = pv.StructuredGrid(
            np.ascontiguousarray(cartesian_points[..., 0]),
            np.ascontiguousarray(cartesian_points[..., 1]),
            np.ascontiguousarray(cartesian_points[..., 2]),
        )
        grid["potential_total_ev"] = np.ascontiguousarray(extended_values).ravel(order="F")
        contour = grid.contour(isosurfaces=[float(level_ev)], scalars="potential_total_ev")
        if getattr(contour, "n_cells", 0) <= 0:
            return None
        triangulated = contour.triangulate()
    except Exception as exc:  # noqa: BLE001
        raise UepError(
            f"PyVista could not contour the potential at {level_ev:.6g} eV: {exc}"
        ) from exc

    points = np.asarray(getattr(triangulated, "points", np.empty((0, 3))), dtype=float)
    faces_array = np.asarray(getattr(triangulated, "faces", np.empty((0,), dtype=int)), dtype=int)
    if points.ndim != 2 or points.shape[1] != 3 or faces_array.size == 0:
        return None

    faces = _reshape_pyvista_faces(faces_array)
    if faces.size == 0:
        return None

    return UepIsosurfaceMesh(
        level_ev=float(level_ev),
        minimum_potential_ev=float(minimum),
        maximum_potential_ev=float(maximum),
        vertices_ang=tuple((float(point[0]), float(point[1]), float(point[2])) for point in points),
        faces=tuple((int(face[0]), int(face[1]), int(face[2])) for face in faces),
    )


def _reshape_pyvista_faces(faces: np.ndarray) -> np.ndarray:
    flat = np.asarray(faces, dtype=int).ravel()
    triangles: list[tuple[int, int, int]] = []
    index = 0
    while index < len(flat):
        vertex_count = int(flat[index])
        if vertex_count <= 0 or index + 1 + vertex_count > len(flat):
            break
        vertices = flat[index + 1 : index + 1 + vertex_count]
        if vertex_count == 3:
            triangles.append((int(vertices[0]), int(vertices[1]), int(vertices[2])))
        else:
            for offset in range(1, vertex_count - 1):
                triangles.append(
                    (int(vertices[0]), int(vertices[offset]), int(vertices[offset + 1]))
                )
        index += 1 + vertex_count
    return np.asarray(triangles, dtype=int)


def classify_stationary_point(
    hessian_ev_per_ang2: np.ndarray
    | tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    *,
    tolerance: float = 1e-6,
) -> StationaryPointClassification:
    """Classify a stationary point from its Hessian eigenvalues."""

    matrix = np.asarray(hessian_ev_per_ang2, dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    positive = int(np.count_nonzero(eigenvalues > tolerance))
    negative = int(np.count_nonzero(eigenvalues < -tolerance))
    near_zero = int(len(eigenvalues) - positive - negative)
    if near_zero:
        label = "flat"
    elif negative == 0:
        label = "minimum"
    elif positive == 0:
        label = "maximum"
    else:
        label = "saddle"
    return StationaryPointClassification(
        label=label,
        eigenvalues=tuple(float(value) for value in eigenvalues),
        positive_count=positive,
        negative_count=negative,
        near_zero_count=near_zero,
    )


def zero_point_energy_ev(
    hessian_ev_per_ang2: np.ndarray
    | tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    *,
    particle_mass_kg: float = MUON_MASS_KG,
) -> float | None:
    """Return the zero-point energy for a stable minimum, or `None` for an unstable extremum."""

    if particle_mass_kg <= 0.0:
        raise UepError("particle_mass_kg must be positive.")

    hessian = np.asarray(hessian_ev_per_ang2, dtype=float) * 1e20 * ELEMENTARY_CHARGE_C
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.any(eigenvalues < 0.0):
        return None
    value = np.sum(0.5 * HBAR_J_S * np.sqrt(eigenvalues / particle_mass_kg)) / ELEMENTARY_CHARGE_C
    return float(value)


def probe_point(
    field: UepField,
    position_ang: tuple[float, float, float] | np.ndarray,
    *,
    particle_mass_kg: float = MUON_MASS_KG,
) -> UepProbeResult:
    """Evaluate the local UEP, its derivatives, the Hessian classification and the ZPE."""

    position = np.asarray(position_ang, dtype=float)
    total_potential, electronic_potential, ionic_potential = field.potential(position)
    total_gradient, electronic_gradient, ionic_gradient = field.gradient(position)
    total_hessian, electronic_hessian, ionic_hessian = field.hessian(position)

    classification = classify_stationary_point(total_hessian[0])
    zpe = zero_point_energy_ev(total_hessian[0], particle_mass_kg=particle_mass_kg)
    total_energy = float(total_potential[0] + zpe) if zpe is not None else None

    return UepProbeResult(
        position_ang=tuple(float(value) for value in position),
        fractional_position=field.fractional_position(position),
        potential_total_ev=float(total_potential[0]),
        potential_electronic_ev=float(electronic_potential[0]),
        potential_ionic_ev=float(ionic_potential[0]),
        gradient_total_ev_per_ang=tuple(float(value) for value in total_gradient[0]),
        gradient_electronic_ev_per_ang=tuple(float(value) for value in electronic_gradient[0]),
        gradient_ionic_ev_per_ang=tuple(float(value) for value in ionic_gradient[0]),
        hessian_total_ev_per_ang2=tuple(
            tuple(float(value) for value in row) for row in total_hessian[0]
        ),
        hessian_electronic_ev_per_ang2=tuple(
            tuple(float(value) for value in row) for row in electronic_hessian[0]
        ),
        hessian_ionic_ev_per_ang2=tuple(
            tuple(float(value) for value in row) for row in ionic_hessian[0]
        ),
        classification=classification,
        zero_point_energy_ev=zpe,
        total_quantum_energy_ev=total_energy,
    )


def optimize_muon_position(
    field: UepField,
    *,
    start_position_ang: tuple[float, float, float] | np.ndarray,
    particle_mass_kg: float = MUON_MASS_KG,
    force_tolerance_ev_per_ang: float = DEFAULT_FORCE_TOLERANCE_EV_PER_ANG,
    max_steps: int = DEFAULT_OPTIMIZER_MAX_STEPS,
    method: str = "trust-exact",
) -> UepOptimizationResult:
    """Optimize one trial muon position under the UEP approximation.

    With the default ``trust-exact``, the optimizer stops when the norm of the potential's
    gradient falls below ``force_tolerance_ev_per_ang``; another method reads it as its own
    ``tol``. A converged result carries the harmonic zero-point energy from the Hessian at
    the end point, or ``None`` when a curvature there is negative. A start that did not
    converge has none, because its end point is not a stationary point.
    """

    start = np.asarray(start_position_ang, dtype=float)
    if start.shape != (3,):
        raise UepError("start_position_ang must contain exactly three Cartesian coordinates.")

    def objective(coords: np.ndarray) -> float:
        return float(field.potential(coords)[0][0])

    def gradient(coords: np.ndarray) -> np.ndarray:
        return np.asarray(field.gradient(coords)[0][0], dtype=float)

    def hessian(coords: np.ndarray) -> np.ndarray:
        return np.asarray(field.hessian(coords)[0][0], dtype=float)

    solution = minimize(
        objective,
        start,
        jac=gradient,
        hess=hessian,
        method=method,
        tol=float(force_tolerance_ev_per_ang),
        options={"maxiter": int(max_steps)},
    )

    optimized = np.asarray(solution.x, dtype=float)
    probe = probe_point(
        field,
        optimized,
        particle_mass_kg=particle_mass_kg,
    )
    converged = bool(solution.success)
    return UepOptimizationResult(
        start_position_ang=tuple(float(value) for value in start),
        optimized_position_ang=tuple(float(value) for value in optimized),
        optimized_fractional_position=probe.fractional_position,
        classical_energy_ev=probe.potential_total_ev,
        zero_point_energy_ev=probe.zero_point_energy_ev if converged else None,
        total_energy_ev=probe.total_quantum_energy_ev if converged else None,
        success=converged,
        message=str(solution.message),
        iterations=int(getattr(solution, "nit", 0) or 0),
        classification=probe.classification,
    )
