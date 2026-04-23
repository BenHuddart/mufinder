"""MuFinder-native Unperturbed Electrostatic Potential analysis tools."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isnan
from pathlib import Path
import re
from typing import Any

from ase import Atoms, io
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_numbers
import numpy as np
from scipy import constants as cnst
from scipy.cluster import hierarchy, vq
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial import distance as spdistance

from mufinder.castep_io.output_parser import parse_castep_mulliken_atomic_spins
from mufinder.castep_io.cell_parser import CellParseError, parse_cell_file


_COULOMB_PREFactor = 1.0 / (4.0 * np.pi * cnst.epsilon_0)
_FLOAT_RE = r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?"
_PSEUDOPOTENTIAL_HEADER_RE = re.compile(r"Pseudopotential\s+Report", re.IGNORECASE)
_PSEUDOPOTENTIAL_END_RE = re.compile(r"Author:", re.IGNORECASE)
_ELEMENT_RE = re.compile(r"Element:\s+([A-Za-z]{1,2})\s+Ionic charge:\s+([0-9.]+)")
_CORE_RADIUS_RE = re.compile(r"(?:[0-9]+|loc)\s+[0-9]\s+[\-0-9.]+\s+([0-9.]+)")
_NEUTRALITY_RE = re.compile(r"unit\s+cell\s+is\s+neutral", re.IGNORECASE)
_TOTAL_CHARGE_RE = re.compile(rf"total\s+charge[^\n=]*=\s*({_FLOAT_RE})", re.IGNORECASE)
_MUON_MASS_KG = cnst.physical_constants["muon mass"][0]
_AMU_TO_KG = cnst.physical_constants["atomic mass constant"][0]
_BOHR_TO_ANG = cnst.physical_constants["Bohr radius"][0] * 1e10
_ISOSURFACE_PREVIEW_MAX_GRID_POINTS = 28
_UEP_KMEANS_SEED = 0


def _pyvista_modules():
    try:
        import pyvista as pv
    except ImportError:
        return None
    return pv


class UepError(ValueError):
    """Raised when UEP source data or analysis is invalid."""


@dataclass(frozen=True, slots=True)
class FormattedDensity:
    """Parsed CASTEP formatted density grid."""

    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
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


@dataclass(frozen=True, slots=True)
class UepStructure:
    """Host structure information loaded from CASTEP output."""

    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    species: tuple[str, ...]
    positions_ang: tuple[tuple[float, float, float], ...]
    fractional_positions: tuple[tuple[float, float, float], ...]
    atomic_spins: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class UepSourceData:
    """Fully validated UEP source bundle."""

    density: FormattedDensity
    structure: UepStructure
    pseudopotentials: tuple[IonicPseudopotential, ...]
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
    hessian_total_ev_per_ang2: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    hessian_electronic_ev_per_ang2: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    hessian_ionic_ev_per_ang2: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
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
class UepLineSample:
    """Sampled UEP data along a line segment."""

    distance_ang: tuple[float, ...]
    potential_total_ev: tuple[float, ...]
    potential_electronic_ev: tuple[float, ...]
    potential_ionic_ev: tuple[float, ...]
    charge_total_e_per_ang3: tuple[float, ...]
    charge_electronic_e_per_ang3: tuple[float, ...]
    charge_ionic_e_per_ang3: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class UepPlaneSample:
    """Sampled UEP data on a plane patch."""

    x_coordinates_ang: tuple[tuple[float, ...], ...]
    y_coordinates_ang: tuple[tuple[float, ...], ...]
    potential_total_ev: tuple[tuple[float, ...], ...]
    potential_electronic_ev: tuple[tuple[float, ...], ...]
    potential_ionic_ev: tuple[tuple[float, ...], ...]
    charge_total_e_per_ang3: tuple[tuple[float, ...], ...]
    charge_electronic_e_per_ang3: tuple[tuple[float, ...], ...]
    charge_ionic_e_per_ang3: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class UepClusterSummary:
    """One cluster of optimized UEP stopping-site candidates."""

    cluster_index: int
    size: int
    representative_index: int
    representative_position_ang: tuple[float, float, float]
    representative_fractional_position: tuple[float, float, float]
    minimum_energy_ev: float | None
    average_energy_ev: float | None
    standard_deviation_ev: float | None
    member_indexes: tuple[int, ...]
    representative_asymmetric_fractional_position: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class UepClusterSimilaritySummary:
    """Distance between two clusters in the raw pymuon-style feature space."""

    left_cluster_index: int
    right_cluster_index: int
    distance: float


@dataclass(frozen=True, slots=True)
class UepBatchSummary:
    """Clustered summary of a batch UEP search."""

    threshold_ang: float
    clusters: tuple[UepClusterSummary, ...]
    clustering_method: str = "legacy_cartesian_hier"
    clustering_hier_t: float | None = None
    clustering_kmeans_k: int | None = None
    clustering_linkage_method: str | None = None
    feature_mode: str = "legacy_cartesian_positions"
    similarities: tuple[UepClusterSimilaritySummary, ...] = ()


@dataclass(frozen=True, slots=True)
class UepClusteringConfig:
    """Configuration for pymuon-suite-style clustering of optimized sites."""

    clustering_method: str = "hier"
    clustering_hier_t: float = 0.3
    clustering_kmeans_k: int = 4
    clustering_linkage_method: str = "single"
    symprec: float = 1e-5
    feature_mode: str = "soprano_defect_asymmetric_fpos"


@dataclass(frozen=True, slots=True)
class UepClusteringResult:
    """Full output of one Soprano-backed clustering pass."""

    config: UepClusteringConfig
    cluster_labels: tuple[int, ...]
    asymmetric_fractional_positions: tuple[tuple[float, float, float], ...]
    genome_vectors: tuple[tuple[float, ...], ...]
    normalized_genome_vectors: tuple[tuple[float, ...], ...]
    summary: UepBatchSummary
    linkage_matrix: tuple[tuple[float, float, float, float], ...] = ()
    kmeans_centroids: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class UepPreparedClusteringDataset:
    """Cached Soprano-derived clustering features for one relaxed-site batch."""

    config: UepClusteringConfig
    optimizations: tuple[UepOptimizationResult, ...]
    asymmetric_fractional_positions: tuple[tuple[float, float, float], ...]
    genome_vectors: tuple[tuple[float, ...], ...]
    normalized_genome_vectors: tuple[tuple[float, ...], ...]
    linkage_method: str
    linkage_matrix: tuple[tuple[float, float, float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class UepPotentialGrid:
    """Real-space total electrostatic potential sampled on the FFT grid."""

    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
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


def serialize_potential_grid(grid: UepPotentialGrid) -> dict[str, Any]:
    """Convert a potential grid into a JSON-serializable payload."""

    payload = {
        "lattice_vectors": [list(vector) for vector in grid.lattice_vectors],
        "grid": [int(value) for value in grid.grid],
        "potential_total_ev": np.asarray(grid.potential_total_ev, dtype=float).tolist(),
    }
    if grid.gw_factor is not None:
        payload["gw_factor"] = float(grid.gw_factor)
    return payload


def deserialize_potential_grid(payload: object) -> UepPotentialGrid | None:
    """Rebuild a persisted potential grid payload."""

    if not isinstance(payload, dict):
        return None
    lattice_payload = payload.get("lattice_vectors")
    grid_payload = payload.get("grid")
    values_payload = payload.get("potential_total_ev")
    if not isinstance(lattice_payload, list) or len(lattice_payload) != 3:
        return None
    if not isinstance(grid_payload, list) or len(grid_payload) != 3:
        return None
    try:
        lattice_vectors = tuple(tuple(float(value) for value in row) for row in lattice_payload)
        grid = tuple(int(value) for value in grid_payload)
        values = np.asarray(values_payload, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.shape != grid:
        return None
    raw_gw_factor = payload.get("gw_factor")
    try:
        gw_factor = None if raw_gw_factor is None else float(raw_gw_factor)
    except (TypeError, ValueError):
        return None
    return UepPotentialGrid(
        lattice_vectors=lattice_vectors,
        grid=grid,
        potential_total_ev=values,
        gw_factor=gw_factor,
    )


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


def parse_formatted_density(path: str | Path) -> FormattedDensity:
    """Parse a CASTEP `.den_fmt` file into a dense FFT-ready grid."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise UepError(f"Failed to read formatted density '{source}': {exc}") from exc

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
                    raise UepError(f"Could not parse lattice vectors from '{source.name}'.")
                lattice_rows.append(tuple(float(value) for value in parts))
            continue
        if "! nspins" in line:
            try:
                nspins = int(line.split()[0])
            except (IndexError, ValueError) as exc:
                raise UepError(f"Could not parse nspins from '{source.name}'.") from exc
            continue
        if "! fine FFT grid" in line:
            parts = line.split()[:3]
            if len(parts) != 3:
                raise UepError(f"Could not parse FFT grid from '{source.name}'.")
            grid = (int(parts[0]), int(parts[1]), int(parts[2]))
            continue
        if stripped.startswith("END header"):
            data_start = index + 1
            break

    if len(lattice_rows) != 3 or grid is None or nspins is None or data_start is None:
        raise UepError(f"Formatted density '{source.name}' does not contain a complete CASTEP header.")

    values = np.zeros((grid[0], grid[1], grid[2], nspins), dtype=float)
    count = 0
    expected_count = grid[0] * grid[1] * grid[2]
    for line in lines[data_start:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            ia = int(parts[0]) - 1
            ib = int(parts[1]) - 1
            ic = int(parts[2]) - 1
            charge = float(parts[3])
        except ValueError as exc:
            raise UepError(f"Invalid density row in '{source.name}': {line.strip()}") from exc
        if not (0 <= ia < grid[0] and 0 <= ib < grid[1] and 0 <= ic < grid[2]):
            raise UepError(f"Density row index out of range in '{source.name}': {line.strip()}")
        values[ia, ib, ic, 0] = charge
        count += 1

    if count != expected_count:
        raise UepError(
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
        raise UepError(f"Failed to read CASTEP file '{source}': {exc}") from exc

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
                core_radius_ang = min(core_radius_ang, float(match.group(1)) * _BOHR_TO_ANG)
        if element and ionic_charge is not None and np.isfinite(core_radius_ang):
            parsed[element] = IonicPseudopotential(
                species=element,
                ionic_charge=ionic_charge,
                core_radius_ang=core_radius_ang,
            )

    if not parsed:
        raise UepError(
            f"Could not find ionic pseudopotential metadata in '{source.name}'. "
            "UEP analysis needs the CASTEP pseudopotential report or a usable CELL/SPECIES_POT fallback."
        )
    return parsed


def _merge_cell_pseudopotential_metadata(
    *,
    parsed: dict[str, IonicPseudopotential],
    cell_path: Path,
) -> dict[str, IonicPseudopotential]:
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

    total_charge = castep_total_charge(output_path)
    neutrality_warning = None
    if total_charge is not None and abs(total_charge) > 1e-3:
        raise UepError(
            f"UEP requires a neutral host cell, but '{output_path.name}' reports total charge {total_charge:.6g}."
        )
    if total_charge is None:
        neutrality_warning = "CASTEP output did not report total cell charge explicitly; assuming a neutral host."

    positions_ang = tuple(tuple(float(value) for value in row) for row in atoms.get_positions().tolist())
    fractional_positions = tuple(tuple(float(value) for value in row) for row in atoms.get_scaled_positions(wrap=True).tolist())
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

    missing_species = sorted({symbol for symbol in structure.species if symbol not in pseudopotentials})
    if missing_species:
        joined = ", ".join(missing_species)
        raise UepError(
            "Missing pseudopotential metadata for species "
            f"{joined}. UEP analysis cannot continue without ionic charge and core-radius data."
        )

    ionic_charge_total = sum(pseudopotentials[symbol].ionic_charge for symbol in structure.species)
    average_density = float(np.average(density.values[:, :, :, 0]))
    if not np.isclose(average_density, ionic_charge_total, atol=1e-4):
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
        cell_path=optional_cell_path if optional_cell_path is not None and optional_cell_path.exists() else None,
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
    def lattice_vectors(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        return self._source.structure.lattice_vectors

    def fractional_position(self, position_ang: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
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
        rho_electronic = np.real(np.sum(self._electron_density_g[:, :, :, None] * kernel, axis=(0, 1, 2))) / self._volume
        rho_ionic = np.real(np.sum(self._ionic_density_g[:, :, :, None] * kernel, axis=(0, 1, 2))) / self._volume
        return (rho_electronic + rho_ionic, rho_electronic, rho_ionic)

    def potential(self, points: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return total/electronic/ionic potential in eV for a unit positive charge."""

        coords = self._validate_points(points)
        kernel = self._fourier_kernel(coords)
        potential_electronic = np.real(np.sum(self._electron_potential_g[:, :, :, None] * kernel, axis=(0, 1, 2)))
        potential_ionic = np.real(np.sum(self._ionic_potential_g[:, :, :, None] * kernel, axis=(0, 1, 2)))
        scale = _COULOMB_PREFactor * cnst.e * 1e10
        potential_electronic *= scale
        potential_ionic *= scale
        return (potential_electronic + potential_ionic, potential_electronic, potential_ionic)

    def gradient(self, points: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return total/electronic/ionic potential gradients in eV/Ang."""

        coords = self._validate_points(points)
        kernel = self._fourier_kernel(coords)
        derivative_kernel = 1.0j * self._g_grid[:, :, :, :, None] * kernel[None, :, :, :, :]
        gradient_electronic = np.real(
            np.sum(self._electron_potential_g[None, :, :, :, None] * derivative_kernel, axis=(1, 2, 3))
        ).T
        gradient_ionic = np.real(
            np.sum(self._ionic_potential_g[None, :, :, :, None] * derivative_kernel, axis=(1, 2, 3))
        ).T
        # Match the Cartesian derivative of ``potential()`` with respect to
        # Angstrom coordinates, as exposed by pymuon-suite after its optimizer
        # length-unit conversion.
        scale = _COULOMB_PREFactor * cnst.e * 1e10
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
            np.sum(self._electron_potential_g[None, None, :, :, :, None] * second_kernel, axis=(2, 3, 4))
        ).T
        hessian_ionic = np.real(
            np.sum(self._ionic_potential_g[None, None, :, :, :, None] * second_kernel, axis=(2, 3, 4))
        ).T
        # Match the Cartesian second derivative of ``potential()`` with respect
        # to Angstrom coordinates, as exposed by pymuon-suite after its
        # optimizer length-unit conversion.
        scale = _COULOMB_PREFactor * cnst.e * 1e10
        hessian_electronic *= scale
        hessian_ionic *= scale
        return (hessian_electronic + hessian_ionic, hessian_electronic, hessian_ionic)


_CUBE_TETRAHEDRA: tuple[tuple[int, int, int, int], ...] = (
    (0, 5, 1, 6),
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
)

_TETRA_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)


def potential_grid(field: UepField) -> UepPotentialGrid:
    """Return the total electrostatic potential on the source FFT grid."""

    grid = tuple(int(value) for value in field.source.density.grid)
    point_count = int(np.prod(grid))
    scale = _COULOMB_PREFactor * cnst.e * 1e10
    total = np.real(np.fft.ifftn(field._electron_potential_g + field._ionic_potential_g)) * point_count * scale
    return UepPotentialGrid(
        lattice_vectors=field.lattice_vectors,
        grid=grid,
        potential_total_ev=np.asarray(total, dtype=float),
        gw_factor=field.source.gw_factor,
    )


def _periodic_resample_axis(values: np.ndarray, target_size: int, *, axis: int) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    if axis < 0 or axis >= source.ndim:
        raise UepError("Periodic grid resampling axis is out of range.")
    if target_size < 2:
        raise UepError("Periodic grid resampling needs at least two points along each axis.")

    source_size = int(source.shape[axis])
    if target_size == source_size:
        return source.copy()

    moved = np.moveaxis(source, axis, 0)
    sample_positions = np.arange(target_size, dtype=float) * (float(source_size) / float(target_size))
    lower = np.floor(sample_positions).astype(int) % source_size
    upper = (lower + 1) % source_size
    weights = (sample_positions - np.floor(sample_positions)).astype(float)
    reshape = (target_size,) + (1,) * (moved.ndim - 1)
    resampled = (1.0 - weights).reshape(reshape) * moved[lower] + weights.reshape(reshape) * moved[upper]
    return np.moveaxis(resampled, 0, axis)


def preview_potential_grid(
    potential: UepPotentialGrid,
    *,
    max_grid_points: int = _ISOSURFACE_PREVIEW_MAX_GRID_POINTS,
) -> UepPotentialGrid:
    """Return a capped-resolution periodic grid for responsive isosurface previews."""

    values = np.asarray(potential.potential_total_ev, dtype=float)
    if values.ndim != 3 or min(values.shape) < 2:
        raise UepError("Isosurface previews need a 3D potential grid with at least two points per axis.")

    limit = max(2, int(max_grid_points))
    target_grid = tuple(min(int(size), limit) for size in values.shape)
    if target_grid == tuple(int(size) for size in values.shape):
        return UepPotentialGrid(
            lattice_vectors=potential.lattice_vectors,
            grid=tuple(int(size) for size in values.shape),
            potential_total_ev=values.copy(),
            gw_factor=potential.gw_factor,
        )

    resampled = values.copy()
    for axis, target_size in enumerate(target_grid):
        resampled = _periodic_resample_axis(resampled, int(target_size), axis=axis)

    source_min = float(np.min(values))
    source_max = float(np.max(values))
    preview_min = float(np.min(resampled))
    preview_max = float(np.max(resampled))
    if np.isclose(preview_min, preview_max):
        resampled = np.full(target_grid, 0.5 * (source_min + source_max), dtype=float)
    else:
        scale = (source_max - source_min) / (preview_max - preview_min)
        resampled = (resampled - preview_min) * scale + source_min

    return UepPotentialGrid(
        lattice_vectors=potential.lattice_vectors,
        grid=target_grid,
        potential_total_ev=np.asarray(resampled, dtype=float),
        gw_factor=potential.gw_factor,
    )


def _cartesian_to_fractional(
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    positions_ang: np.ndarray,
) -> np.ndarray:
    lattice = np.asarray(lattice_vectors, dtype=float)
    if lattice.shape != (3, 3):
        raise UepError("Slice sampling needs three lattice vectors.")
    cartesian = np.asarray(positions_ang, dtype=float)
    if cartesian.ndim != 2 or cartesian.shape[1] != 3:
        raise UepError("Slice sampling points must be Cartesian 3-vectors.")
    return np.linalg.solve(lattice.T, cartesian.T).T


def _fractional_to_cartesian(
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    fractional_positions: np.ndarray,
) -> np.ndarray:
    lattice = np.asarray(lattice_vectors, dtype=float)
    if lattice.shape != (3, 3):
        raise UepError("Slice sampling needs three lattice vectors.")
    fractional = np.asarray(fractional_positions, dtype=float)
    if fractional.ndim != 2 or fractional.shape[1] != 3:
        raise UepError("Fractional positions must be 3-vectors.")
    return fractional @ lattice


def fractional_point_to_cartesian(
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    fractional_position: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    """Convert one fractional point into Cartesian coordinates."""

    point = np.asarray(fractional_position, dtype=float)
    if point.shape != (3,):
        raise UepError("Fractional positions must be 3-vectors.")
    cartesian = _fractional_to_cartesian(lattice_vectors, point.reshape(1, 3))[0]
    return tuple(float(value) for value in cartesian)


def miller_indices_to_cartesian_normal(
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    miller_indices: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    """Convert Miller indices into the corresponding Cartesian plane normal."""

    lattice = np.asarray(lattice_vectors, dtype=float)
    if lattice.shape != (3, 3):
        raise UepError("Slice sampling needs three lattice vectors.")
    miller = np.asarray(miller_indices, dtype=float)
    if miller.shape != (3,):
        raise UepError("Miller indices must be 3-vectors.")
    if float(np.linalg.norm(miller)) <= 1e-12:
        raise UepError("Miller indices must not all be zero.")
    normal = np.linalg.solve(lattice, miller)
    return normalize_plane_normal(normal)


def cartesian_normal_to_lattice_plane_components(
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    normal_ang: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    """Convert a Cartesian plane normal into normalized lattice-plane components."""

    lattice = np.asarray(lattice_vectors, dtype=float)
    if lattice.shape != (3, 3):
        raise UepError("Slice sampling needs three lattice vectors.")
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
        raise UepError("Slice sampling needs a 3D potential grid with at least two points per axis.")

    fractional = _cartesian_to_fractional(potential.lattice_vectors, np.asarray(positions_ang, dtype=float))
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


def normalize_plane_normal(normal_ang: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    """Return a normalized plane normal or raise if it is degenerate."""

    normal = np.asarray(normal_ang, dtype=float)
    if normal.shape != (3,):
        raise UepError("Plane normals must be 3-vectors.")
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise UepError("Plane normals must have non-zero length.")
    return tuple(float(value) for value in (normal / norm))


def plane_basis_from_normal(normal_ang: tuple[float, float, float] | np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
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
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
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
            np.asarray(lattice_vectors[0], dtype=float) + np.asarray(lattice_vectors[1], dtype=float),
            np.asarray(lattice_vectors[0], dtype=float) + np.asarray(lattice_vectors[2], dtype=float),
            np.asarray(lattice_vectors[1], dtype=float) + np.asarray(lattice_vectors[2], dtype=float),
            np.asarray(lattice_vectors[0], dtype=float) + np.asarray(lattice_vectors[1], dtype=float) + np.asarray(lattice_vectors[2], dtype=float),
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
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
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
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
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
    points_ang: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]] | np.ndarray,
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
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
        raise UepError("Best-fit plane construction failed because the selected atoms are collinear.")
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
    xs = np.linspace(float(plane.extent_u_ang[0]), float(plane.extent_u_ang[1]), int(plane.points_u))
    ys = np.linspace(float(plane.extent_v_ang[0]), float(plane.extent_v_ang[1]), int(plane.points_v))
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    points = (
        anchor.reshape(1, 1, 3)
        + (x_grid[..., np.newaxis] * basis_u.reshape(1, 1, 3))
        + (y_grid[..., np.newaxis] * basis_v.reshape(1, 1, 3))
    )
    flat_points = points.reshape((-1, 3))
    sampled = _interpolate_periodic_grid_trilinear(potential, flat_points).reshape((int(plane.points_u), int(plane.points_v)))
    return UepGridSliceSample(
        plane=plane,
        x_coordinates_ang=tuple(tuple(float(value) for value in row) for row in points[:, :, 0]),
        y_coordinates_ang=tuple(tuple(float(value) for value in row) for row in points[:, :, 1]),
        z_coordinates_ang=tuple(tuple(float(value) for value in row) for row in points[:, :, 2]),
        potential_total_ev=tuple(tuple(float(value) for value in row) for row in sampled),
        minimum_potential_ev=float(np.min(sampled)),
        maximum_potential_ev=float(np.max(sampled)),
    )


def _interpolate_isosurface_vertex(
    point_a: np.ndarray,
    point_b: np.ndarray,
    value_a: float,
    value_b: float,
    *,
    level_ev: float,
) -> np.ndarray:
    if np.isclose(value_a, value_b):
        return 0.5 * (point_a + point_b)
    weight = float((level_ev - value_a) / (value_b - value_a))
    weight = min(max(weight, 0.0), 1.0)
    return point_a + weight * (point_b - point_a)


def _order_polygon_vertices(points: list[np.ndarray]) -> list[np.ndarray]:
    if len(points) <= 3:
        return points

    center = np.mean(points, axis=0)
    normal = None
    reference = None
    for first in range(len(points) - 2):
        for second in range(first + 1, len(points) - 1):
            for third in range(second + 1, len(points)):
                trial = np.cross(points[second] - points[first], points[third] - points[first])
                norm = float(np.linalg.norm(trial))
                if norm > 1e-12:
                    normal = trial / norm
                    reference = points[first] - center
                    break
            if normal is not None:
                break
        if normal is not None:
            break

    if normal is None or reference is None or float(np.linalg.norm(reference)) <= 1e-12:
        return points

    axis_u = reference / float(np.linalg.norm(reference))
    axis_v = np.cross(normal, axis_u)
    axis_v_norm = float(np.linalg.norm(axis_v))
    if axis_v_norm <= 1e-12:
        return points
    axis_v = axis_v / axis_v_norm

    def angle(point: np.ndarray) -> float:
        delta = point - center
        return atan2(float(np.dot(delta, axis_v)), float(np.dot(delta, axis_u)))

    return [points[index] for index in np.argsort([angle(point) for point in points])]


def _polygonize_tetrahedron(
    points: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    values: tuple[float, float, float, float],
    *,
    level_ev: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    intersections: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    for start_index, end_index in _TETRA_EDGES:
        value_start = float(values[start_index])
        value_end = float(values[end_index])
        if np.isclose(value_start, level_ev) and np.isclose(value_end, level_ev):
            continue
        crosses = (
            (value_start <= level_ev < value_end)
            or (value_end <= level_ev < value_start)
            or (np.isclose(value_start, level_ev) and not np.isclose(value_end, level_ev))
            or (np.isclose(value_end, level_ev) and not np.isclose(value_start, level_ev))
        )
        if not crosses:
            continue
        vertex = _interpolate_isosurface_vertex(
            points[start_index],
            points[end_index],
            value_start,
            value_end,
            level_ev=level_ev,
        )
        key = tuple(int(round(float(coord) * 1_000_000.0)) for coord in vertex)
        if key in seen:
            continue
        seen.add(key)
        intersections.append(vertex)

    if len(intersections) < 3:
        return []
    ordered = _order_polygon_vertices(intersections)
    if len(ordered) == 3:
        return [(ordered[0], ordered[1], ordered[2])]

    triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index in range(1, len(ordered) - 1):
        triangles.append((ordered[0], ordered[index], ordered[index + 1]))
    return triangles


def extract_isosurface(
    potential: UepPotentialGrid,
    *,
    level_ev: float,
) -> UepIsosurfaceMesh:
    """Extract an isosurface mesh from a real-space electrostatic-potential grid."""

    values = np.asarray(potential.potential_total_ev, dtype=float)
    if values.ndim != 3 or min(values.shape) < 2:
        raise UepError("Isosurface extraction needs a 3D potential grid with at least two points per axis.")

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
    if pyvista_mesh is not None:
        return pyvista_mesh

    return _extract_isosurface_fallback(
        potential,
        level_ev=level_ev,
        minimum=minimum,
        maximum=maximum,
    )


def _extract_isosurface_with_pyvista(
    potential: UepPotentialGrid,
    *,
    level_ev: float,
    minimum: float,
    maximum: float,
) -> UepIsosurfaceMesh | None:
    pv = _pyvista_modules()
    if pv is None:
        return None

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
    except Exception:  # noqa: BLE001
        return None

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
                triangles.append((int(vertices[0]), int(vertices[offset]), int(vertices[offset + 1])))
        index += 1 + vertex_count
    return np.asarray(triangles, dtype=int)


def _extract_isosurface_fallback(
    potential: UepPotentialGrid,
    *,
    level_ev: float,
    minimum: float,
    maximum: float,
) -> UepIsosurfaceMesh:
    values = np.asarray(potential.potential_total_ev, dtype=float)

    nx, ny, nz = (int(size) for size in values.shape)
    extended_values = np.pad(values, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    frac_x = np.arange(nx + 1, dtype=float) / float(nx)
    frac_y = np.arange(ny + 1, dtype=float) / float(ny)
    frac_z = np.arange(nz + 1, dtype=float) / float(nz)
    fractional_points = np.stack(np.meshgrid(frac_x, frac_y, frac_z, indexing="ij"), axis=-1)
    lattice = np.asarray(potential.lattice_vectors, dtype=float)
    cartesian_points = np.einsum("...i,ij->...j", fractional_points, lattice, optimize=True)

    cube_offsets = (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    )

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_index_by_key: dict[tuple[int, int, int], int] = {}

    def add_vertex(point: np.ndarray) -> int:
        key = tuple(int(round(float(coord) * 1_000_000.0)) for coord in point)
        if key in vertex_index_by_key:
            return vertex_index_by_key[key]
        index = len(vertices)
        vertices.append((float(point[0]), float(point[1]), float(point[2])))
        vertex_index_by_key[key] = index
        return index

    for ia in range(nx):
        for ib in range(ny):
            for ic in range(nz):
                cube_points = tuple(
                    np.asarray(cartesian_points[ia + dx, ib + dy, ic + dz], dtype=float) for dx, dy, dz in cube_offsets
                )
                cube_values = tuple(float(extended_values[ia + dx, ib + dy, ic + dz]) for dx, dy, dz in cube_offsets)
                if minimum > level_ev or maximum < level_ev:
                    continue
                if level_ev < min(cube_values) or level_ev > max(cube_values):
                    continue
                for tetrahedron in _CUBE_TETRAHEDRA:
                    tetra_points = tuple(cube_points[index] for index in tetrahedron)
                    tetra_values = tuple(cube_values[index] for index in tetrahedron)
                    for triangle in _polygonize_tetrahedron(tetra_points, tetra_values, level_ev=level_ev):
                        indices = tuple(add_vertex(vertex) for vertex in triangle)
                        if len({indices[0], indices[1], indices[2]}) == 3:
                            faces.append(indices)

    if not faces:
        raise UepError(
            f"No isosurface intersects the cell at {level_ev:.6g} eV. Choose a value closer to the extrema."
        )

    return UepIsosurfaceMesh(
        level_ev=float(level_ev),
        minimum_potential_ev=minimum,
        maximum_potential_ev=maximum,
        vertices_ang=tuple(vertices),
        faces=tuple(faces),
    )


def classify_stationary_point(
    hessian_ev_per_ang2: np.ndarray | tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
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
    hessian_ev_per_ang2: np.ndarray | tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    *,
    particle_mass_kg: float = _MUON_MASS_KG,
) -> float | None:
    """Return the zero-point energy for a stable minimum, or `None` for an unstable extremum."""

    if particle_mass_kg <= 0.0:
        raise UepError("particle_mass_kg must be positive.")

    hessian = np.asarray(hessian_ev_per_ang2, dtype=float) * 1e20 * cnst.e
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.any(eigenvalues < 0.0):
        return None
    value = np.sum(0.5 * cnst.hbar * np.sqrt(eigenvalues / particle_mass_kg)) / cnst.e
    return float(value)


def probe_point(
    field: UepField,
    position_ang: tuple[float, float, float] | np.ndarray,
    *,
    particle_mass_kg: float = _MUON_MASS_KG,
    include_zero_point_energy: bool = True,
) -> UepProbeResult:
    """Evaluate the local UEP, derivatives, and Hessian classification."""

    position = np.asarray(position_ang, dtype=float)
    total_potential, electronic_potential, ionic_potential = field.potential(position)
    total_gradient, electronic_gradient, ionic_gradient = field.gradient(position)
    total_hessian, electronic_hessian, ionic_hessian = field.hessian(position)

    classification = classify_stationary_point(total_hessian[0])
    zpe = (
        zero_point_energy_ev(total_hessian[0], particle_mass_kg=particle_mass_kg)
        if include_zero_point_energy
        else None
    )
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
        hessian_total_ev_per_ang2=tuple(tuple(float(value) for value in row) for row in total_hessian[0]),
        hessian_electronic_ev_per_ang2=tuple(tuple(float(value) for value in row) for row in electronic_hessian[0]),
        hessian_ionic_ev_per_ang2=tuple(tuple(float(value) for value in row) for row in ionic_hessian[0]),
        classification=classification,
        zero_point_energy_ev=zpe,
        total_quantum_energy_ev=total_energy,
    )


def optimize_muon_position(
    field: UepField,
    *,
    start_position_ang: tuple[float, float, float] | np.ndarray,
    particle_mass_kg: float = _MUON_MASS_KG,
    tolerance_ev: float = 1e-5,
    max_steps: int = 300,
    method: str = "trust-exact",
    include_zero_point_energy: bool = True,
) -> UepOptimizationResult:
    """Optimize one trial muon position under the UEP approximation."""

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
        tol=float(tolerance_ev),
        options={"maxiter": int(max_steps)},
    )

    optimized = np.asarray(solution.x, dtype=float)
    probe = probe_point(
        field,
        optimized,
        particle_mass_kg=particle_mass_kg,
        include_zero_point_energy=include_zero_point_energy,
    )
    return UepOptimizationResult(
        start_position_ang=tuple(float(value) for value in start),
        optimized_position_ang=tuple(float(value) for value in optimized),
        optimized_fractional_position=probe.fractional_position,
        classical_energy_ev=probe.potential_total_ev,
        zero_point_energy_ev=probe.zero_point_energy_ev,
        total_energy_ev=probe.total_quantum_energy_ev,
        success=bool(solution.success),
        message=str(solution.message),
        iterations=int(getattr(solution, "nit", 0) or 0),
        classification=probe.classification,
    )


def calculate_optimization_zero_point_energy(
    field: UepField,
    optimization: UepOptimizationResult,
    *,
    particle_mass_kg: float = _MUON_MASS_KG,
) -> UepOptimizationResult:
    """Re-evaluate the optimized site and attach harmonic ZPE data."""

    probe = probe_point(
        field,
        optimization.optimized_position_ang,
        particle_mass_kg=particle_mass_kg,
        include_zero_point_energy=True,
    )
    return UepOptimizationResult(
        start_position_ang=optimization.start_position_ang,
        optimized_position_ang=optimization.optimized_position_ang,
        optimized_fractional_position=probe.fractional_position,
        classical_energy_ev=probe.potential_total_ev,
        zero_point_energy_ev=probe.zero_point_energy_ev,
        total_energy_ev=probe.total_quantum_energy_ev,
        success=optimization.success,
        message=optimization.message,
        iterations=optimization.iterations,
        classification=probe.classification,
    )


def sample_line(
    field: UepField,
    *,
    start_ang: tuple[float, float, float] | np.ndarray,
    end_ang: tuple[float, float, float] | np.ndarray,
    num_points: int,
) -> UepLineSample:
    """Sample potential and charge density along a straight line."""

    if num_points < 2:
        raise UepError("Line sampling needs at least two points.")
    start = np.asarray(start_ang, dtype=float)
    end = np.asarray(end_ang, dtype=float)
    line = np.linspace(start, end, int(num_points))
    distance = np.linalg.norm(line - line[0], axis=1)
    potential_total, potential_electronic, potential_ionic = field.potential(line)
    charge_total, charge_electronic, charge_ionic = field.charge_density(line)
    return UepLineSample(
        distance_ang=tuple(float(value) for value in distance),
        potential_total_ev=tuple(float(value) for value in potential_total),
        potential_electronic_ev=tuple(float(value) for value in potential_electronic),
        potential_ionic_ev=tuple(float(value) for value in potential_ionic),
        charge_total_e_per_ang3=tuple(float(value) for value in charge_total),
        charge_electronic_e_per_ang3=tuple(float(value) for value in charge_electronic),
        charge_ionic_e_per_ang3=tuple(float(value) for value in charge_ionic),
    )


def sample_plane(
    field: UepField,
    *,
    origin_ang: tuple[float, float, float] | np.ndarray,
    corner_x_ang: tuple[float, float, float] | np.ndarray,
    corner_y_ang: tuple[float, float, float] | np.ndarray,
    points_x: int,
    points_y: int,
) -> UepPlaneSample:
    """Sample potential and charge density on a plane patch."""

    if points_x < 2 or points_y < 2:
        raise UepError("Plane sampling needs at least two points along each axis.")
    origin = np.asarray(origin_ang, dtype=float)
    corner_x = np.asarray(corner_x_ang, dtype=float)
    corner_y = np.asarray(corner_y_ang, dtype=float)
    dx = corner_x - origin
    dy = corner_y - origin
    if np.linalg.norm(dx) <= 1e-12 or np.linalg.norm(dy) <= 1e-12:
        raise UepError("Plane sampling axes must span a non-zero patch.")
    xs = np.linspace(0.0, 1.0, int(points_x))
    ys = np.linspace(0.0, 1.0, int(points_y))
    plane = np.array([origin + (x * dx) + (y * dy) for x in xs for y in ys], dtype=float)
    potential_total, potential_electronic, potential_ionic = field.potential(plane)
    charge_total, charge_electronic, charge_ionic = field.charge_density(plane)

    x_coords = np.zeros((int(points_x), int(points_y)), dtype=float)
    y_coords = np.zeros((int(points_x), int(points_y)), dtype=float)
    dx_norm = np.linalg.norm(dx)
    dy_orthogonal = dy - np.dot(dy, dx) * dx / max(dx_norm**2, 1e-12)
    dy_norm = np.linalg.norm(dy_orthogonal)
    if dy_norm <= 1e-12:
        raise UepError("Plane sampling axes must not be collinear.")
    for ix in range(int(points_x)):
        for iy in range(int(points_y)):
            point = plane[(ix * int(points_y)) + iy]
            relative = point - origin
            x_coords[ix, iy] = np.dot(relative, dx) / max(dx_norm, 1e-12)
            y_coords[ix, iy] = np.dot(relative, dy_orthogonal) / max(dy_norm, 1e-12)

    def reshape(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
        shaped = values.reshape((int(points_x), int(points_y)))
        return tuple(tuple(float(value) for value in row) for row in shaped)

    return UepPlaneSample(
        x_coordinates_ang=tuple(tuple(float(value) for value in row) for row in x_coords),
        y_coordinates_ang=tuple(tuple(float(value) for value in row) for row in y_coords),
        potential_total_ev=reshape(potential_total),
        potential_electronic_ev=reshape(potential_electronic),
        potential_ionic_ev=reshape(potential_ionic),
        charge_total_e_per_ang3=reshape(charge_total),
        charge_electronic_e_per_ang3=reshape(charge_electronic),
        charge_ionic_e_per_ang3=reshape(charge_ionic),
    )


def cluster_optimized_positions(
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
    *,
    threshold_ang: float = 0.2,
) -> UepBatchSummary:
    """Cluster optimized UEP endpoints in Cartesian space."""

    if threshold_ang <= 0.0:
        raise UepError("threshold_ang must be positive.")
    if not optimizations:
        return UepBatchSummary(
            threshold_ang=float(threshold_ang),
            clusters=(),
            clustering_method="legacy_cartesian_hier",
            clustering_hier_t=float(threshold_ang),
            clustering_linkage_method="average",
            feature_mode="legacy_cartesian_positions",
        )

    positions = np.array([entry.optimized_position_ang for entry in optimizations], dtype=float)
    if len(optimizations) == 1:
        energy = _clustering_energy(optimizations[0])
        summary = UepClusterSummary(
            cluster_index=1,
            size=1,
            representative_index=0,
            representative_position_ang=optimizations[0].optimized_position_ang,
            representative_fractional_position=optimizations[0].optimized_fractional_position,
            minimum_energy_ev=energy,
            average_energy_ev=energy,
            standard_deviation_ev=0.0,
            member_indexes=(0,),
        )
        return UepBatchSummary(
            threshold_ang=float(threshold_ang),
            clusters=(summary,),
            clustering_method="legacy_cartesian_hier",
            clustering_hier_t=float(threshold_ang),
            clustering_linkage_method="average",
            feature_mode="legacy_cartesian_positions",
        )

    linkage_matrix = linkage(positions, method="average")
    labels = fcluster(linkage_matrix, t=float(threshold_ang), criterion="distance")
    cluster_ids = sorted(set(int(value) for value in labels))

    clusters: list[UepClusterSummary] = []
    for cluster_index, cluster_id in enumerate(cluster_ids, start=1):
        member_indexes = tuple(index for index, label in enumerate(labels) if int(label) == cluster_id)
        members = [optimizations[index] for index in member_indexes]
        numeric_energies = [_clustering_energy(entry) for entry in members]
        representative_offset = min(
            range(len(members)),
            key=lambda member_index: (
                _clustering_energy(members[member_index]),
                member_index,
            ),
        )
        representative_index = member_indexes[representative_offset]
        representative = optimizations[representative_index]
        minimum = min(numeric_energies) if numeric_energies else None
        average = float(np.mean(numeric_energies)) if numeric_energies else None
        deviation = float(np.std(numeric_energies)) if numeric_energies else None
        clusters.append(
            UepClusterSummary(
                cluster_index=cluster_index,
                size=len(member_indexes),
                representative_index=representative_index,
                representative_position_ang=representative.optimized_position_ang,
                representative_fractional_position=representative.optimized_fractional_position,
                minimum_energy_ev=minimum,
                average_energy_ev=average,
                standard_deviation_ev=deviation,
                member_indexes=member_indexes,
            )
        )

    normalized, _cluster_index_map = _renumber_cluster_summaries(clusters, optimizations)
    return UepBatchSummary(
        threshold_ang=float(threshold_ang),
        clusters=normalized,
        clustering_method="legacy_cartesian_hier",
        clustering_hier_t=float(threshold_ang),
        clustering_linkage_method="average",
        feature_mode="legacy_cartesian_positions",
    )


def _import_soprano_clustering():
    try:
        from soprano.analyse.phylogen import Gene, PhylogenCluster
        from soprano.collection import AtomsCollection
        from soprano.utils import compute_asymmetric_distmat
    except ImportError as exc:  # pragma: no cover - exercised when dependency is unavailable.
        raise UepError(
            "pymuon-suite-compatible UEP clustering requires the optional 'Soprano' dependency."
        ) from exc
    return Gene, PhylogenCluster, AtomsCollection, compute_asymmetric_distmat


def _structure_like_to_ase_atoms(source_structure: object) -> Atoms:
    lattice_vectors = getattr(source_structure, "lattice_vectors", None)
    species = getattr(source_structure, "species", None)
    fractional_positions = getattr(source_structure, "fractional_positions", None)
    if lattice_vectors is None or species is None or fractional_positions is None:
        raise UepError("source_structure must define lattice_vectors, species, and fractional_positions.")
    if len(species) != len(fractional_positions):
        raise UepError("source_structure species and fractional_positions must have matching lengths.")
    return Atoms(
        list(str(symbol) for symbol in species),
        scaled_positions=[tuple(float(value) for value in row) for row in fractional_positions],
        cell=np.asarray(lattice_vectors, dtype=float),
        pbc=True,
    )


def _clustering_energy(entry: UepOptimizationResult) -> float:
    if entry.total_energy_ev is not None and not isnan(entry.total_energy_ev):
        return float(entry.total_energy_ev)
    if not isnan(entry.classical_energy_ev):
        return float(entry.classical_energy_ev)
    raise UepError("UEP clustering requires a finite classical or total energy for every optimization.")


def relaxed_site_number_map(
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
) -> dict[int, int]:
    """Return 1-based relaxed-site numbers ordered globally by classical energy."""

    ordered_indexes = sorted(
        range(len(optimizations)),
        key=lambda idx: (
            float(optimizations[idx].classical_energy_ev)
            if not isnan(optimizations[idx].classical_energy_ev)
            else float("inf"),
            int(idx),
        ),
    )
    return {int(result_index): int(site_number) for site_number, result_index in enumerate(ordered_indexes, start=1)}


def _cluster_average_classical_energy(
    cluster: UepClusterSummary,
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
) -> float:
    classical_energies = [
        float(optimizations[int(index)].classical_energy_ev)
        for index in cluster.member_indexes
        if not isnan(optimizations[int(index)].classical_energy_ev)
    ]
    if not classical_energies:
        return float("inf")
    return float(np.mean(classical_energies))


def _cluster_minimum_classical_energy(
    cluster: UepClusterSummary,
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
) -> float:
    classical_energies = [
        float(optimizations[int(index)].classical_energy_ev)
        for index in cluster.member_indexes
        if not isnan(optimizations[int(index)].classical_energy_ev)
    ]
    if not classical_energies:
        return float("inf")
    return min(classical_energies)


def _renumber_cluster_summaries(
    clusters: list[UepClusterSummary],
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
) -> tuple[tuple[UepClusterSummary, ...], dict[int, int]]:
    ordered = sorted(
        clusters,
        key=lambda cluster: (
            _cluster_average_classical_energy(cluster, optimizations),
            _cluster_minimum_classical_energy(cluster, optimizations),
            int(cluster.cluster_index),
        ),
    )
    cluster_index_map = {int(cluster.cluster_index): index + 1 for index, cluster in enumerate(ordered)}
    normalized = tuple(
        UepClusterSummary(
            cluster_index=index + 1,
            size=cluster.size,
            representative_index=cluster.representative_index,
            representative_position_ang=cluster.representative_position_ang,
            representative_fractional_position=cluster.representative_fractional_position,
            minimum_energy_ev=cluster.minimum_energy_ev,
            average_energy_ev=cluster.average_energy_ev,
            standard_deviation_ev=cluster.standard_deviation_ev,
            member_indexes=cluster.member_indexes,
            representative_asymmetric_fractional_position=cluster.representative_asymmetric_fractional_position,
        )
        for index, cluster in enumerate(ordered)
    )
    return normalized, cluster_index_map


def _renumber_cluster_similarity_summaries(
    similarities: tuple[UepClusterSimilaritySummary, ...],
    cluster_index_map: dict[int, int],
) -> tuple[UepClusterSimilaritySummary, ...]:
    remapped: list[UepClusterSimilaritySummary] = []
    for similarity in similarities:
        left_index = cluster_index_map.get(int(similarity.left_cluster_index), int(similarity.left_cluster_index))
        right_index = cluster_index_map.get(int(similarity.right_cluster_index), int(similarity.right_cluster_index))
        ordered_left, ordered_right = sorted((left_index, right_index))
        remapped.append(
            UepClusterSimilaritySummary(
                left_cluster_index=int(ordered_left),
                right_cluster_index=int(ordered_right),
                distance=float(similarity.distance),
            )
        )
    return tuple(sorted(remapped, key=lambda item: (item.distance, item.left_cluster_index, item.right_cluster_index)))


def _asymmetric_images_centre(
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
) -> np.ndarray:
    reference_index = min(range(len(optimizations)), key=lambda idx: (_clustering_energy(optimizations[idx]), idx))
    reference = np.asarray(optimizations[reference_index].optimized_fractional_position, dtype=float)
    # Avoid high-symmetry centres, which Soprano warns can produce unstable image selection.
    jitter = np.array((1.61803398875e-6, 2.41421356237e-6, 3.14159265359e-6), dtype=float)
    return (reference + jitter) % 1.0


def _muonated_atoms_for_clustering(
    host_atoms: Atoms,
    optimization: UepOptimizationResult,
    *,
    index: int,
) -> Atoms:
    scaled_positions = np.vstack(
        [
            host_atoms.get_scaled_positions(wrap=True),
            np.asarray(optimization.optimized_fractional_position, dtype=float),
        ]
    )
    structure = Atoms(
        host_atoms.get_chemical_symbols() + ["H"],
        scaled_positions=scaled_positions,
        cell=host_atoms.cell.copy(),
        pbc=True,
        info={"name": f"uep_{index + 1}"},
    )
    structure.calc = SinglePointCalculator(structure, energy=_clustering_energy(optimization))
    return structure


def _cluster_similarity_summaries(
    cluster_groups: list[np.ndarray],
    genome_vectors: np.ndarray,
) -> tuple[UepClusterSimilaritySummary, ...]:
    if len(cluster_groups) < 2:
        return ()
    centres = np.array([np.average(genome_vectors[group], axis=0) for group in cluster_groups], dtype=float)
    distances = np.linalg.norm(centres[:, None] - centres[None, :], axis=-1)
    pairs: list[UepClusterSimilaritySummary] = []
    for left_index in range(len(cluster_groups) - 1):
        for right_index in range(left_index + 1, len(cluster_groups)):
            pairs.append(
                UepClusterSimilaritySummary(
                    left_cluster_index=left_index + 1,
                    right_cluster_index=right_index + 1,
                    distance=float(distances[left_index, right_index]),
                )
            )
    return tuple(sorted(pairs, key=lambda item: (item.distance, item.left_cluster_index, item.right_cluster_index)))


def _validate_clustering_config(config: UepClusteringConfig) -> UepClusteringConfig:
    active = config or UepClusteringConfig()
    method = active.clustering_method.strip().lower()
    if method not in {"hier", "kmeans"}:
        raise UepError("clustering_method must be 'hier' or 'kmeans'.")
    if active.clustering_hier_t <= 0.0:
        raise UepError("clustering_hier_t must be positive.")
    if active.clustering_kmeans_k <= 0:
        raise UepError("clustering_kmeans_k must be positive.")
    return UepClusteringConfig(
        clustering_method=method,
        clustering_hier_t=float(active.clustering_hier_t),
        clustering_kmeans_k=int(active.clustering_kmeans_k),
        clustering_linkage_method=str(active.clustering_linkage_method).strip().lower() or "single",
        symprec=float(active.symprec),
        feature_mode=active.feature_mode,
    )


def prepare_uep_clustering_dataset(
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
    *,
    source_structure: object,
    config: UepClusteringConfig | None = None,
) -> UepPreparedClusteringDataset:
    """Prepare reusable Soprano-derived clustering features for a relaxed-site batch."""

    active = _validate_clustering_config(config or UepClusteringConfig())
    if not optimizations:
        return UepPreparedClusteringDataset(
            config=active,
            optimizations=(),
            asymmetric_fractional_positions=(),
            genome_vectors=(),
            normalized_genome_vectors=(),
            linkage_method=active.clustering_linkage_method,
            linkage_matrix=(),
        )

    Gene, PhylogenCluster, AtomsCollection, compute_asymmetric_distmat = _import_soprano_clustering()
    host_atoms = _structure_like_to_ase_atoms(source_structure)
    collection = AtomsCollection(
        [
            _muonated_atoms_for_clustering(host_atoms, optimization, index=index)
            for index, optimization in enumerate(optimizations)
        ]
    )
    images_centre = _asymmetric_images_centre(optimizations)

    def _parse_asymmetric_fpos(coll):
        points = coll.all.get_scaled_positions()[:, -1, :]
        _, images = compute_asymmetric_distmat(
            host_atoms,
            points,
            return_images=True,
            images_centre=images_centre,
            symprec=active.symprec,
        )
        return images

    genes = [
        Gene("energy", 1.0, {}),
        Gene("mufinder_defect_asymmetric_fpos", 1.0, parser=_parse_asymmetric_fpos),
    ]
    clusterer = PhylogenCluster(collection, genes=genes)
    genome_vectors, _ = clusterer.get_genome_vectors()
    normalized_vectors, _ = clusterer.get_genome_vectors_norm()
    linkage_matrix = np.empty((0, 4), dtype=float)
    if len(optimizations) > 1:
        linkage_matrix = np.asarray(
            clusterer.get_linkage(method=active.clustering_linkage_method),
            dtype=float,
        )
    return UepPreparedClusteringDataset(
        config=active,
        optimizations=tuple(optimizations),
        asymmetric_fractional_positions=tuple(
            tuple(float(value) for value in row.tolist()) for row in np.asarray(genome_vectors[:, 1:4], dtype=float)
        ),
        genome_vectors=tuple(
            tuple(float(value) for value in row.tolist()) for row in np.asarray(genome_vectors, dtype=float)
        ),
        normalized_genome_vectors=tuple(
            tuple(float(value) for value in row.tolist()) for row in np.asarray(normalized_vectors, dtype=float)
        ),
        linkage_method=active.clustering_linkage_method,
        linkage_matrix=tuple(
            tuple(float(value) for value in row.tolist()) for row in np.asarray(linkage_matrix, dtype=float)
        ),
    )


def _linkage_matrix_for_prepared(
    prepared: UepPreparedClusteringDataset,
    *,
    method: str,
) -> np.ndarray:
    if len(prepared.optimizations) <= 1:
        return np.empty((0, 4), dtype=float)
    if method == prepared.linkage_method and prepared.linkage_matrix:
        return np.asarray(prepared.linkage_matrix, dtype=float)
    normalized_vectors = np.asarray(prepared.normalized_genome_vectors, dtype=float)
    condensed = spdistance.pdist(normalized_vectors, metric="euclidean")
    return np.asarray(hierarchy.linkage(condensed, method=method), dtype=float)


def _cluster_groups_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for cluster_id in sorted(set(int(value) for value in labels)):
        groups.append(np.where(labels == cluster_id)[0].astype(int))
    return groups


def _build_clustering_result(
    prepared: UepPreparedClusteringDataset,
    *,
    config: UepClusteringConfig,
    cluster_labels: np.ndarray,
    cluster_groups: list[np.ndarray],
    linkage_matrix: np.ndarray | None = None,
    kmeans_centroids: np.ndarray | None = None,
) -> UepClusteringResult:
    method = config.clustering_method
    optimizations = prepared.optimizations
    genome_vectors = np.asarray(prepared.genome_vectors, dtype=float)
    asymmetric_positions = np.asarray(prepared.asymmetric_fractional_positions, dtype=float)
    clusters: list[UepClusterSummary] = []
    for cluster_index, members in enumerate(cluster_groups, start=1):
        group = np.asarray(members, dtype=int)
        if group.size == 0:
            continue
        member_indexes = tuple(int(value) for value in group.tolist())
        cluster_members = [optimizations[index] for index in member_indexes]
        finite_total_energies = [
            float(member.total_energy_ev)
            for member in cluster_members
            if member.total_energy_ev is not None and not isnan(member.total_energy_ev)
        ]
        representative_index = min(member_indexes, key=lambda idx: (_clustering_energy(optimizations[idx]), idx))
        representative = optimizations[representative_index]
        clusters.append(
            UepClusterSummary(
                cluster_index=cluster_index,
                size=len(member_indexes),
                representative_index=representative_index,
                representative_position_ang=representative.optimized_position_ang,
                representative_fractional_position=representative.optimized_fractional_position,
                minimum_energy_ev=min(finite_total_energies) if finite_total_energies else None,
                average_energy_ev=float(np.mean(finite_total_energies)) if finite_total_energies else None,
                standard_deviation_ev=float(np.std(finite_total_energies)) if finite_total_energies else None,
                member_indexes=member_indexes,
                representative_asymmetric_fractional_position=tuple(
                    float(value) for value in asymmetric_positions[representative_index].tolist()
                ),
            )
        )

    normalized_clusters, cluster_index_map = _renumber_cluster_summaries(clusters, optimizations)
    summary = UepBatchSummary(
        threshold_ang=float(config.clustering_hier_t if method == "hier" else 0.0),
        clusters=normalized_clusters,
        clustering_method=method,
        clustering_hier_t=float(config.clustering_hier_t),
        clustering_kmeans_k=(int(len(cluster_groups)) if method == "kmeans" else None),
        clustering_linkage_method=config.clustering_linkage_method if method == "hier" else None,
        feature_mode=config.feature_mode,
        similarities=_renumber_cluster_similarity_summaries(
            _cluster_similarity_summaries(cluster_groups, genome_vectors),
            cluster_index_map,
        ),
    )
    return UepClusteringResult(
        config=config,
        cluster_labels=tuple(int(value) for value in np.asarray(cluster_labels, dtype=int).tolist()),
        asymmetric_fractional_positions=prepared.asymmetric_fractional_positions,
        genome_vectors=prepared.genome_vectors,
        normalized_genome_vectors=prepared.normalized_genome_vectors,
        linkage_matrix=tuple(
            tuple(float(value) for value in row.tolist())
            for row in np.asarray(np.empty((0, 4), dtype=float) if linkage_matrix is None else linkage_matrix, dtype=float)
        ),
        kmeans_centroids=tuple(
            tuple(float(value) for value in row.tolist())
            for row in np.asarray(np.empty((0, 0), dtype=float) if kmeans_centroids is None else kmeans_centroids, dtype=float)
        ),
        summary=summary,
    )


def cluster_prepared_uep_dataset(
    prepared: UepPreparedClusteringDataset,
    *,
    config: UepClusteringConfig | None = None,
) -> UepClusteringResult:
    """Replay pymuon-suite-style clustering over a prepared relaxed-site dataset."""

    active = _validate_clustering_config(config or prepared.config)
    method = active.clustering_method
    if not prepared.optimizations:
        return UepClusteringResult(
            config=active,
            cluster_labels=(),
            asymmetric_fractional_positions=(),
            genome_vectors=(),
            normalized_genome_vectors=(),
            linkage_matrix=(),
            kmeans_centroids=(),
            summary=UepBatchSummary(
                threshold_ang=float(active.clustering_hier_t if method == "hier" else 0.0),
                clusters=(),
                clustering_method=method,
                clustering_hier_t=float(active.clustering_hier_t),
                clustering_kmeans_k=(int(active.clustering_kmeans_k) if method == "kmeans" else None),
                clustering_linkage_method=active.clustering_linkage_method if method == "hier" else None,
                feature_mode=active.feature_mode,
                similarities=(),
            ),
        )

    if method == "hier":
        if len(prepared.optimizations) == 1:
            cluster_labels = np.array([1], dtype=int)
            cluster_groups = [np.array([0], dtype=int)]
            linkage_matrix = np.empty((0, 4), dtype=float)
        else:
            linkage_matrix = _linkage_matrix_for_prepared(prepared, method=active.clustering_linkage_method)
            cluster_labels = hierarchy.fcluster(
                linkage_matrix,
                t=max(float(active.clustering_hier_t), 1e-12),
                criterion="distance",
            ).astype(int)
            cluster_groups = _cluster_groups_from_labels(cluster_labels)
        return _build_clustering_result(
            prepared,
            config=active,
            cluster_labels=cluster_labels,
            cluster_groups=cluster_groups,
            linkage_matrix=linkage_matrix,
        )

    cluster_count = int(active.clustering_kmeans_k)
    if cluster_count > len(prepared.optimizations):
        raise UepError("clustering_kmeans_k must not exceed the number of optimized sites.")
    normalized_vectors = np.asarray(prepared.normalized_genome_vectors, dtype=float)
    centroids, _distortion = vq.kmeans(
        normalized_vectors,
        cluster_count,
        seed=_UEP_KMEANS_SEED,
    )
    zero_based_labels, _distances = vq.vq(normalized_vectors, centroids)
    cluster_labels = zero_based_labels.astype(int) + 1
    cluster_groups = [np.where(zero_based_labels == index)[0].astype(int) for index in range(int(np.max(zero_based_labels)) + 1)]
    return _build_clustering_result(
        prepared,
        config=active,
        cluster_labels=cluster_labels,
        cluster_groups=cluster_groups,
        kmeans_centroids=np.asarray(centroids, dtype=float),
    )


def cluster_uep_optimizations(
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
    *,
    source_structure: object,
    config: UepClusteringConfig | None = None,
) -> UepClusteringResult:
    """Cluster optimized UEP endpoints using the Soprano machinery used by pymuon-suite."""

    prepared = prepare_uep_clustering_dataset(
        optimizations,
        source_structure=source_structure,
        config=config,
    )
    return cluster_prepared_uep_dataset(prepared, config=config)


def particle_mass_kg_from_amu(mass_amu: float) -> float:
    """Convert a particle mass from amu to kg."""

    return float(mass_amu) * _AMU_TO_KG
