"""Magnetic-structure helpers for propagation-vector based field workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from uuid import uuid4

from ase import Atoms
import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.spglib_compat import get_crystal_symmetry_dataset

MAGNETIC_STRUCTURE_SCHEMA_VERSION = 2


class MagneticStructureError(ValueError):
    """Raised when a magnetic-structure definition is invalid."""


@dataclass(frozen=True, slots=True)
class ComplexVector:
    """JSON-safe complex three-vector."""

    real: tuple[float, float, float]
    imaginary: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_vector(cls, value: Sequence[complex | float]) -> "ComplexVector":
        arr = _as_complex_vector(value, "fourier vector")
        return cls(
            real=tuple(float(x) for x in arr.real),
            imaginary=tuple(float(x) for x in arr.imag),
        )

    @classmethod
    def from_json(cls, payload: object) -> "ComplexVector":
        if isinstance(payload, Mapping):
            real = _as_float_tuple(payload.get("real", (0.0, 0.0, 0.0)), "real")
            imag = _as_float_tuple(payload.get("imaginary", (0.0, 0.0, 0.0)), "imaginary")
            return cls(real=real, imaginary=imag)
        return cls.from_vector(payload)  # type: ignore[arg-type]

    def to_array(self) -> np.ndarray:
        return np.asarray(self.real, dtype=float) + 1j * np.asarray(self.imaginary, dtype=float)

    def to_json(self) -> dict[str, list[float]]:
        return {
            "real": [float(x) for x in self.real],
            "imaginary": [float(x) for x in self.imaginary],
        }


@dataclass(frozen=True, slots=True)
class FourierTerm:
    """One Fourier coefficient attached to a magnetic component."""

    q_coeff: tuple[float, float, float]
    coefficient: ComplexVector

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "FourierTerm":
        return cls(
            q_coeff=_as_float_tuple(payload.get("q_coeff", (0.0, 0.0, 0.0)), "q_coeff"),
            coefficient=ComplexVector.from_json(payload.get("coefficient", {})),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "q_coeff": [float(x) for x in self.q_coeff],
            "coefficient": self.coefficient.to_json(),
        }


@dataclass(frozen=True, slots=True)
class MagneticTextureTerm:
    """One abstract Fourier term in a spin texture."""

    q_coeff: tuple[float, float, float]
    coefficient: ComplexVector

    @classmethod
    def from_vector(
        cls,
        q_coeff: Sequence[float],
        coefficient: Sequence[complex | float],
    ) -> "MagneticTextureTerm":
        return cls(
            q_coeff=_as_float_tuple(q_coeff, "texture q_coeff"),
            coefficient=ComplexVector.from_vector(coefficient),
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "MagneticTextureTerm":
        return cls(
            q_coeff=_as_float_tuple(payload.get("q_coeff", (0.0, 0.0, 0.0)), "texture q_coeff"),
            coefficient=ComplexVector.from_json(payload.get("coefficient", {})),
        )

    def to_fourier_term(self, q_coeff: Sequence[float] | None = None) -> FourierTerm:
        return FourierTerm(
            q_coeff=self.q_coeff if q_coeff is None else _as_float_tuple(q_coeff, "q_coeff"),
            coefficient=self.coefficient,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "q_coeff": [float(x) for x in self.q_coeff],
            "coefficient": self.coefficient.to_json(),
        }


@dataclass(frozen=True, slots=True)
class MagneticTexture:
    """Abstract multi-q spin texture before it is applied to atom sites."""

    terms: tuple[MagneticTextureTerm, ...] = ()

    @classmethod
    def from_json(cls, payload: object) -> "MagneticTexture":
        if not isinstance(payload, Mapping):
            return cls()
        raw_terms = payload.get("terms", ())
        terms = tuple(
            MagneticTextureTerm.from_json(item)
            for item in raw_terms
            if isinstance(item, Mapping)
        )
        return cls(terms=terms)

    def to_json(self) -> dict[str, object]:
        return {"terms": [term.to_json() for term in self.terms]}

    def combine_terms(self, *, atol: float = 1e-12) -> "MagneticTexture":
        """Return a texture where terms with identical q coefficients are summed."""

        combined: list[tuple[np.ndarray, np.ndarray]] = []
        for term in self.terms:
            q = np.asarray(term.q_coeff, dtype=float)
            coefficient = term.coefficient.to_array()
            for index, (existing_q, existing_fc) in enumerate(combined):
                if np.allclose(existing_q, q, atol=atol, rtol=0.0):
                    combined[index] = (existing_q, existing_fc + coefficient)
                    break
            else:
                combined.append((q, coefficient))
        return MagneticTexture(
            terms=tuple(
                MagneticTextureTerm.from_vector(q, fc)
                for q, fc in combined
                if not np.allclose(fc, 0.0, atol=atol, rtol=0.0)
            )
        )


@dataclass(frozen=True, slots=True)
class MagneticTextureDiagnostics:
    """Moment-magnitude diagnostics for a sampled spin texture."""

    term_count: int
    magnitude_min: float
    magnitude_mean: float
    magnitude_max: float
    magnitude_std: float


@dataclass(frozen=True, slots=True)
class AtomSelector:
    """Selector describing which host atoms receive a magnetic component."""

    kind: str = "all"
    values: tuple[str | int, ...] = ()

    @classmethod
    def all(cls) -> "AtomSelector":
        return cls("all", ())

    @classmethod
    def species(cls, *symbols: str) -> "AtomSelector":
        return cls("species", tuple(str(symbol) for symbol in symbols))

    @classmethod
    def indices(cls, indices: Sequence[int]) -> "AtomSelector":
        return cls("indices", tuple(int(index) for index in indices))

    @classmethod
    def symmetry_equivalent(cls, representative: int) -> "AtomSelector":
        return cls("symmetry_equivalent", (int(representative),))

    @classmethod
    def from_json(cls, payload: object) -> "AtomSelector":
        if not isinstance(payload, Mapping):
            return cls.all()
        kind = str(payload.get("kind", "all")).strip() or "all"
        raw_values = payload.get("values", ())
        if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
            values: tuple[str | int, ...] = tuple(raw_values)  # type: ignore[arg-type]
        elif raw_values in (None, ""):
            values = ()
        else:
            values = (str(raw_values),)
        return cls(kind=kind, values=values)

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind, "values": list(self.values)}


@dataclass(frozen=True, slots=True)
class MagneticComponent:
    """A user-facing component in a magnetic structure."""

    component_id: str
    label: str
    kind: str
    selector: AtomSelector
    terms: tuple[FourierTerm, ...]
    phase_turns: float = 0.0
    enabled: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "MagneticComponent":
        raw_terms = payload.get("terms", ())
        terms = tuple(
            FourierTerm.from_json(term)
            for term in raw_terms
            if isinstance(term, Mapping)
        )
        return cls(
            component_id=str(payload.get("component_id", "")).strip(),
            label=str(payload.get("label", "")).strip(),
            kind=str(payload.get("kind", "custom")).strip() or "custom",
            selector=AtomSelector.from_json(payload.get("selector", {})),
            terms=terms,
            phase_turns=float(payload.get("phase_turns", 0.0) or 0.0),
            enabled=bool(payload.get("enabled", True)),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), Mapping) else {},
        )

    def to_json(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "label": self.label,
            "kind": self.kind,
            "selector": self.selector.to_json(),
            "terms": [term.to_json() for term in self.terms],
            "phase_turns": float(self.phase_turns),
            "enabled": bool(self.enabled),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MuesrModelData:
    """muesr-compatible magnetic model data."""

    k_rlu: np.ndarray
    fc: np.ndarray
    phi: np.ndarray
    component_ids: tuple[str, ...]
    q_coeffs: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class AtomResolvedFourierComponent:
    """Fourier data after applying selectors to a concrete host structure."""

    component_id: str
    q_coeff: tuple[float, float, float]
    q_rlu: np.ndarray
    atom_fcs: np.ndarray


@dataclass(frozen=True, slots=True)
class MagneticStructureModel:
    """Persisted magnetic structure in MuFinder's propagation-vector format."""

    q_basis_rlu: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    components: tuple[MagneticComponent, ...] = ()
    schema_version: int = MAGNETIC_STRUCTURE_SCHEMA_VERSION
    global_phase_turns: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: object) -> "MagneticStructureModel":
        if not isinstance(payload, Mapping):
            return cls()
        q_basis = _as_q_basis(payload.get("q_basis_rlu", payload.get("q_basis", ())))
        raw_components = payload.get("components", ())
        components = tuple(
            component
            for component in (
                MagneticComponent.from_json(item)
                for item in raw_components
                if isinstance(item, Mapping)
            )
            if component.component_id and component.terms
        )
        metadata = payload.get("metadata", {})
        return cls(
            q_basis_rlu=q_basis,
            components=components,
            schema_version=int(payload.get("schema_version", MAGNETIC_STRUCTURE_SCHEMA_VERSION)),
            global_phase_turns=float(payload.get("global_phase_turns", 0.0) or 0.0),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": int(self.schema_version),
            "q_basis_rlu": [[float(x) for x in row] for row in self.q_basis_rlu],
            "components": [component.to_json() for component in self.components],
            "global_phase_turns": float(self.global_phase_turns),
            "metadata": dict(self.metadata),
        }

    @property
    def q_basis_array(self) -> np.ndarray:
        return np.asarray(self.q_basis_rlu, dtype=float).reshape(3, 3)

    def q_coeff_to_rlu(self, q_coeff: Sequence[float]) -> np.ndarray:
        return _as_float_array(q_coeff, "q_coeff") @ self.q_basis_array


def viable_atom_selectors(
    structure: Structure,
    *,
    selected_indices: Sequence[int] = (),
) -> list[AtomSelector]:
    """Return selector options that are meaningful for the current host structure."""

    selectors: list[AtomSelector] = [AtomSelector.all()]
    for symbol in sorted(set(structure.species)):
        selectors.append(AtomSelector.species(symbol))

    for representative in _equivalent_atom_representatives(structure):
        selectors.append(AtomSelector.symmetry_equivalent(int(representative)))

    valid_selection = tuple(
        int(index)
        for index in selected_indices
        if 0 <= int(index) < len(structure.species)
    )
    if valid_selection:
        selectors.append(AtomSelector.indices(valid_selection))
    return selectors


def resolve_atom_selector(structure: Structure, selector: AtomSelector) -> tuple[int, ...]:
    """Resolve an atom selector to sorted unit-cell atom indices."""

    atom_count = len(structure.species)
    kind = selector.kind.strip().lower()
    if kind == "all":
        return tuple(range(atom_count))
    if kind == "species":
        species = {str(value) for value in selector.values}
        return tuple(index for index, symbol in enumerate(structure.species) if symbol in species)
    if kind == "indices":
        indices = sorted({int(value) for value in selector.values})
        if any(index < 0 or index >= atom_count for index in indices):
            raise MagneticStructureError("atom selector contains an out-of-range index")
        return tuple(indices)
    if kind == "symmetry_equivalent":
        if not selector.values:
            raise MagneticStructureError("symmetry-equivalent selector requires a representative index")
        representative = int(selector.values[0])
        equivalents = _equivalent_atoms(structure)
        if equivalents is None or representative < 0 or representative >= atom_count:
            raise MagneticStructureError("symmetry-equivalent atom groups could not be resolved")
        return tuple(int(index) for index in np.where(equivalents == equivalents[representative])[0])
    raise MagneticStructureError(f"unknown atom selector kind '{selector.kind}'")


def atom_resolved_fourier_components(
    structure: Structure,
    model: MagneticStructureModel,
    *,
    component_ids: Sequence[str] | None = None,
) -> tuple[AtomResolvedFourierComponent, ...]:
    """Apply selectors and phases to produce atom-resolved Fourier components."""

    atom_count = len(structure.species)
    positions = np.asarray(structure.fractional_positions, dtype=float)
    if positions.shape != (atom_count, 3):
        raise MagneticStructureError("structure fractional positions must have shape (n_atoms, 3)")

    selected_ids = None if component_ids is None else {str(value) for value in component_ids}
    resolved: list[AtomResolvedFourierComponent] = []
    for component in model.components:
        if not component.enabled:
            continue
        if selected_ids is not None and component.component_id not in selected_ids:
            continue
        indices = resolve_atom_selector(structure, component.selector)
        if not indices:
            continue
        index_array = np.asarray(indices, dtype=int)
        component_phase = float(model.global_phase_turns) + float(component.phase_turns)
        for term in component.terms:
            q_rlu = model.q_coeff_to_rlu(term.q_coeff)
            fc = np.zeros((atom_count, 3), dtype=complex)
            phase_turns = positions[index_array] @ q_rlu + component_phase
            phase = np.exp(-2j * np.pi * phase_turns)
            fc[index_array] = phase[:, np.newaxis] * term.coefficient.to_array()[np.newaxis, :]
            resolved.append(
                AtomResolvedFourierComponent(
                    component_id=component.component_id,
                    q_coeff=term.q_coeff,
                    q_rlu=q_rlu,
                    atom_fcs=fc,
                )
            )
    return tuple(resolved)


def moment_at_atom(
    structure: Structure,
    model: MagneticStructureModel,
    atom_index: int,
    *,
    cell_offset: Sequence[float] = (0.0, 0.0, 0.0),
    component_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Evaluate the real magnetic moment for one atom and cell offset."""

    atom_index = int(atom_index)
    if atom_index < 0 or atom_index >= len(structure.species):
        raise MagneticStructureError("atom_index is out of range")
    offset = _as_float_array(cell_offset, "cell_offset")
    moment = np.zeros(3, dtype=complex)
    for component in atom_resolved_fourier_components(structure, model, component_ids=component_ids):
        fc = component.atom_fcs[atom_index]
        if np.allclose(fc, 0.0):
            continue
        phase = np.exp(-2j * np.pi * float(component.q_rlu @ offset))
        moment += fc * phase
    return moment.real


def moments_in_supercell(
    structure: Structure,
    model: MagneticStructureModel,
    supercell: Sequence[int] = (1, 1, 1),
    *,
    component_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Cartesian positions, moments, and unit-cell atom indices for a diagonal supercell."""

    repeats = _as_supercell_repeats(supercell)
    cell = np.asarray(structure.lattice_vectors, dtype=float)
    frac = np.asarray(structure.fractional_positions, dtype=float)
    offsets = np.array(
        [(i, j, k) for i in range(repeats[0]) for j in range(repeats[1]) for k in range(repeats[2])],
        dtype=float,
    )
    atom_count = len(structure.species)
    positions_frac = (frac[np.newaxis, :, :] + offsets[:, np.newaxis, :]).reshape(-1, 3)
    atom_indices = np.tile(np.arange(atom_count, dtype=int), len(offsets))
    positions_cart = positions_frac @ cell
    moments = np.zeros((positions_frac.shape[0], 3), dtype=float)
    for row, atom_index in enumerate(atom_indices):
        offset = np.floor(positions_frac[row])
        moments[row] = moment_at_atom(
            structure,
            model,
            int(atom_index),
            cell_offset=offset,
            component_ids=component_ids,
        )
    return positions_cart, moments, atom_indices


def get_muesr_model_data(
    structure: Structure,
    model: MagneticStructureModel,
    *,
    merge_equivalent_k: bool = True,
    component_ids: Sequence[str] | None = None,
    atol: float = 1e-10,
) -> tuple[MuesrModelData, ...]:
    """Return magnetic model data that can be passed to muesr."""

    grouped: list[dict[str, object]] = []
    for component in atom_resolved_fourier_components(structure, model, component_ids=component_ids):
        if np.allclose(component.atom_fcs, 0.0):
            continue
        target: dict[str, object] | None = None
        if merge_equivalent_k:
            for entry in grouped:
                if np.allclose(entry["k"], component.q_rlu, atol=atol, rtol=0.0):
                    target = entry
                    break
        if target is None:
            target = {
                "k": component.q_rlu.copy(),
                "fc": np.zeros((len(structure.species), 3), dtype=complex),
                "component_ids": [],
                "q_coeffs": [],
            }
            grouped.append(target)
        target["fc"] = np.asarray(target["fc"], dtype=complex) + component.atom_fcs
        target["component_ids"].append(component.component_id)  # type: ignore[index]
        target["q_coeffs"].append(component.q_coeff)  # type: ignore[index]

    out: list[MuesrModelData] = []
    for entry in grouped:
        fc = np.asarray(entry["fc"], dtype=complex)
        if np.allclose(fc, 0.0):
            continue
        out.append(
            MuesrModelData(
                k_rlu=np.asarray(entry["k"], dtype=float),
                fc=fc,
                phi=np.zeros(len(structure.species), dtype=float),
                component_ids=tuple(sorted(set(entry["component_ids"]))),  # type: ignore[arg-type]
                q_coeffs=tuple(sorted(set(entry["q_coeffs"]))),  # type: ignore[arg-type]
            )
        )
    return tuple(out)


def split_muesr_model_data_into_helices(
    model_data: MuesrModelData,
    *,
    atol: float = 1e-10,
) -> tuple[MuesrModelData, ...]:
    """Split muesr data into per-atom circular helices when needed.

    muesr's incommensurate field routine expects each atom's complex Fourier
    coefficient to describe a circular helix: real and imaginary parts must be
    orthogonal and have equal magnitude. MuFinder permits more general complex
    coefficients, so elliptical or linearly polarised coefficients are resolved
    into circular helical pieces at the hand-off to muesr.
    """

    fc = np.asarray(model_data.fc, dtype=complex)
    if _fourier_components_are_circular_helices(fc, atol=atol):
        return (model_data,)

    return tuple(
        MuesrModelData(
            k_rlu=np.asarray(model_data.k_rlu, dtype=float).copy(),
            fc=helix_fc,
            phi=np.asarray(model_data.phi, dtype=float).copy(),
            component_ids=model_data.component_ids,
            q_coeffs=model_data.q_coeffs,
        )
        for helix_fc in _split_fourier_array_into_circular_helices(fc, atol=atol)
    )


def _fourier_components_are_circular_helices(fc: np.ndarray, *, atol: float = 1e-10) -> bool:
    arr = np.asarray(fc, dtype=complex)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise MagneticStructureError(f"muesr Fourier coefficients must have shape (n_atoms, 3), got {arr.shape}")
    return all(_is_circular_helix_vector(vector, atol=atol) for vector in arr)


def _is_circular_helix_vector(fc: np.ndarray, *, atol: float = 1e-10) -> bool:
    vector = np.asarray(fc, dtype=complex).reshape(3)
    if atol <= 0.0:
        raise MagneticStructureError("helical decomposition tolerance must be positive")

    real = vector.real
    imag = vector.imag
    real_norm = float(np.linalg.norm(real))
    imag_norm = float(np.linalg.norm(imag))
    scale = max(1.0, real_norm, imag_norm)

    if real_norm <= atol * scale and imag_norm <= atol * scale:
        return True
    if real_norm <= atol * scale or imag_norm <= atol * scale:
        return False

    equal_norm = abs(real_norm - imag_norm) <= atol * scale
    orthogonal = abs(float(np.dot(real, imag))) <= atol * max(1.0, real_norm * imag_norm)
    return bool(equal_norm and orthogonal)


def _split_fourier_array_into_circular_helices(
    fc: np.ndarray,
    *,
    atol: float = 1e-10,
) -> tuple[np.ndarray, ...]:
    arr = np.asarray(fc, dtype=complex)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise MagneticStructureError(f"muesr Fourier coefficients must have shape (n_atoms, 3), got {arr.shape}")
    if atol <= 0.0:
        raise MagneticStructureError("helical decomposition tolerance must be positive")

    components: list[np.ndarray] = []
    for atom_index in range(arr.shape[0]):
        parts = _split_complex_vector_into_circular_helices(arr[atom_index], atol=atol)
        while len(components) < len(parts):
            components.append(np.zeros_like(arr, dtype=complex))
        for part_index, part in enumerate(parts):
            components[part_index][atom_index] = part

    scale = max(1.0, float(np.linalg.norm(arr)))
    kept = [component for component in components if float(np.linalg.norm(component)) > atol * scale]
    if not kept:
        kept = [np.zeros_like(arr, dtype=complex)]

    reconstructed = np.sum(np.asarray(kept, dtype=complex), axis=0)
    if float(np.linalg.norm(reconstructed - arr)) > 100.0 * atol * scale:
        raise MagneticStructureError("helical decomposition failed to reconstruct Fourier coefficients")

    for component in kept:
        if not _fourier_components_are_circular_helices(component, atol=atol):
            raise MagneticStructureError("helical decomposition produced a non-helical component")

    return tuple(component.copy() for component in kept)


def _split_complex_vector_into_circular_helices(
    fc: np.ndarray,
    *,
    atol: float = 1e-10,
) -> tuple[np.ndarray, ...]:
    vector = np.asarray(fc, dtype=complex).reshape(3)
    if atol <= 0.0:
        raise MagneticStructureError("helical decomposition tolerance must be positive")

    scale = max(1.0, float(np.linalg.norm(vector)))
    if float(np.linalg.norm(vector)) <= atol * scale:
        return (np.zeros(3, dtype=complex),)
    if _is_circular_helix_vector(vector, atol=atol):
        return (vector.copy(),)

    real = vector.real
    imag = vector.imag
    real_norm = float(np.linalg.norm(real))
    imag_norm = float(np.linalg.norm(imag))

    e1 = real / real_norm if real_norm > atol * scale else imag / imag_norm
    imag_perp = imag - float(np.dot(imag, e1)) * e1
    imag_perp_norm = float(np.linalg.norm(imag_perp))
    if imag_perp_norm > atol * scale:
        e2 = imag_perp / imag_perp_norm
    else:
        e2 = _orthogonal_unit_vector(e1)

    c1 = np.dot(vector, e1)
    c2 = np.dot(vector, e2)
    w_plus = 0.5 * (c1 - 1j * c2)
    w_minus = 0.5 * (c1 + 1j * c2)

    plus = w_plus * (e1 + 1j * e2)
    minus = w_minus * (e1 - 1j * e2)

    out: list[np.ndarray] = []
    if float(np.linalg.norm(plus)) > atol * scale:
        out.append(plus.astype(complex, copy=False))
    if float(np.linalg.norm(minus)) > atol * scale:
        out.append(minus.astype(complex, copy=False))
    if not out:
        out = [np.zeros(3, dtype=complex)]

    reconstructed = np.sum(np.asarray(out, dtype=complex), axis=0)
    if float(np.linalg.norm(reconstructed - vector)) > 100.0 * atol * scale:
        raise MagneticStructureError("helical decomposition failed to reconstruct Fourier vector")
    return tuple(part.copy() for part in out)


def _orthogonal_unit_vector(vector: np.ndarray) -> np.ndarray:
    unit = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(unit))
    if norm <= 0.0:
        raise MagneticStructureError("cannot find an orthogonal vector to a zero vector")
    unit = unit / norm
    axis = np.eye(3, dtype=float)[int(np.argmin(np.abs(unit)))]
    orthogonal = np.cross(unit, axis)
    orthogonal_norm = float(np.linalg.norm(orthogonal))
    if orthogonal_norm <= 0.0:
        axis = np.asarray((1.0, 0.0, 0.0), dtype=float)
        orthogonal = np.cross(unit, axis)
        orthogonal_norm = float(np.linalg.norm(orthogonal))
    return orthogonal / orthogonal_norm


def build_collinear_component(
    *,
    label: str,
    selector: AtomSelector,
    q_coeff: Sequence[float],
    moment: Sequence[float],
    phase_turns: float = 0.0,
    component_id: str | None = None,
) -> MagneticComponent:
    """Build a collinear magnetic component."""

    return _build_single_term_component(
        kind="collinear",
        label=label,
        selector=selector,
        q_coeff=q_coeff,
        coefficient=_as_float_array(moment, "moment").astype(complex),
        phase_turns=phase_turns,
        component_id=component_id,
    )


def build_spin_density_wave_component(
    *,
    label: str,
    selector: AtomSelector,
    q_coeff: Sequence[float],
    amplitude_vector: Sequence[float],
    phase_turns: float = 0.0,
    component_id: str | None = None,
) -> MagneticComponent:
    """Build a spin-density-wave component."""

    return _build_single_term_component(
        kind="spin_density_wave",
        label=label,
        selector=selector,
        q_coeff=q_coeff,
        coefficient=_as_float_array(amplitude_vector, "amplitude_vector").astype(complex),
        phase_turns=phase_turns,
        component_id=component_id,
    )


def build_helix_component(
    *,
    label: str,
    selector: AtomSelector,
    q_coeff: Sequence[float],
    real_axis_vector: Sequence[float],
    imaginary_axis_vector: Sequence[float],
    phase_turns: float = 0.0,
    component_id: str | None = None,
) -> MagneticComponent:
    """Build a helical magnetic component."""

    coefficient = _as_float_array(real_axis_vector, "real_axis_vector").astype(complex)
    coefficient += 1j * _as_float_array(imaginary_axis_vector, "imaginary_axis_vector")
    return _build_single_term_component(
        kind="helix",
        label=label,
        selector=selector,
        q_coeff=q_coeff,
        coefficient=coefficient,
        phase_turns=phase_turns,
        component_id=component_id,
    )


def build_cycloid_component(
    *,
    label: str,
    selector: AtomSelector,
    q_coeff: Sequence[float],
    real_axis_vector: Sequence[float],
    imaginary_axis_vector: Sequence[float],
    phase_turns: float = 0.0,
    component_id: str | None = None,
) -> MagneticComponent:
    """Build a cycloidal magnetic component."""

    coefficient = _as_float_array(real_axis_vector, "real_axis_vector").astype(complex)
    coefficient += 1j * _as_float_array(imaginary_axis_vector, "imaginary_axis_vector")
    return _build_single_term_component(
        kind="cycloid",
        label=label,
        selector=selector,
        q_coeff=q_coeff,
        coefficient=coefficient,
        phase_turns=phase_turns,
        component_id=component_id,
    )


def build_conical_component(
    *,
    label: str,
    selector: AtomSelector,
    q_coeff: Sequence[float],
    total_moment: float,
    real_axis_direction: Sequence[float],
    imaginary_axis_direction: Sequence[float],
    static_axis_direction: Sequence[float],
    cone_angle_degrees: float,
    phase_turns: float = 0.0,
    component_id: str | None = None,
) -> MagneticComponent:
    """Build a conical magnetic component.

    The component stores a q=0 term along the cone axis and a finite-q helical
    term in the perpendicular rotation plane. ``cone_angle_degrees`` is the
    angle between the moment and the cone axis; 90 degrees gives a pure helix.
    """

    angle = np.deg2rad(float(cone_angle_degrees))
    moment = float(total_moment)
    static = moment * float(np.cos(angle)) * _unit_vector(static_axis_direction, "static_axis_direction")
    rotating_real = moment * float(np.sin(angle)) * _unit_vector(real_axis_direction, "real_axis_direction")
    rotating_imag = moment * float(np.sin(angle)) * _unit_vector(imaginary_axis_direction, "imaginary_axis_direction")
    coefficient = rotating_real.astype(complex) + 1j * rotating_imag
    return MagneticComponent(
        component_id=component_id or f"component-{uuid4()}",
        label=str(label).strip() or "Conical",
        kind="conical",
        selector=selector,
        terms=(
            FourierTerm(q_coeff=(0.0, 0.0, 0.0), coefficient=ComplexVector.from_vector(static.astype(complex))),
            FourierTerm(q_coeff=_as_float_tuple(q_coeff, "q_coeff"), coefficient=ComplexVector.from_vector(coefficient)),
        ),
        phase_turns=float(phase_turns),
        enabled=True,
    )


def build_texture_component(
    *,
    label: str,
    selector: AtomSelector,
    texture: MagneticTexture,
    q_coeff_transform: Sequence[Sequence[float]] | None = None,
    phase_turns: float = 0.0,
    component_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MagneticComponent:
    """Build one grouped component by applying an abstract multi-q texture."""

    if not texture.terms:
        raise MagneticStructureError("texture must contain at least one Fourier term")
    transform = np.eye(3, dtype=float) if q_coeff_transform is None else np.asarray(q_coeff_transform, dtype=float)
    if transform.shape != (3, 3):
        raise MagneticStructureError("q_coeff_transform must have shape (3, 3)")
    terms = tuple(
        term.to_fourier_term(np.asarray(term.q_coeff, dtype=float) @ transform)
        for term in texture.combine_terms().terms
    )
    return MagneticComponent(
        component_id=component_id or f"component-{uuid4()}",
        label=str(label).strip() or "Multi-q texture",
        kind="texture",
        selector=selector,
        terms=terms,
        phase_turns=float(phase_turns),
        enabled=True,
        metadata=dict(metadata or {}),
    )


def build_hexagonal_skyrmion_texture(
    *,
    magnitude: float,
    q_directions: Sequence[Sequence[float]] | None = None,
    core_direction: Sequence[float] = (0.0, 0.0, -1.0),
    cone_angle_degrees: float = 90.0,
    skyrmion_type: str = "bloch",
    helicity: int = 1,
) -> MagneticTexture:
    """Build a compact three-q hexagonal skyrmion-lattice texture."""

    core = _unit_vector(core_direction, "core_direction")
    if q_directions is None:
        q1 = _unit_vector((1.0, 0.0, 0.0), "q1_direction")
        q_dirs = (q1, _rotate_vector(q1, core, 2.0 * np.pi / 3.0), _rotate_vector(q1, core, 4.0 * np.pi / 3.0))
    else:
        if len(q_directions) < 3:
            raise MagneticStructureError("hexagonal skyrmion texture requires three q directions")
        q_dirs = tuple(_unit_vector(direction, "q_direction") for direction in q_directions[:3])
    return _build_skyrmion_texture(
        magnitude=magnitude,
        q_coeffs=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, -1.0, 0.0)),
        q_directions=q_dirs,
        core_direction=core,
        cone_angle_degrees=cone_angle_degrees,
        skyrmion_type=skyrmion_type,
        helicity=helicity,
        ferromagnetic_background=0.0,
    )


def build_square_skyrmion_texture(
    *,
    magnitude: float,
    q_directions: Sequence[Sequence[float]] | None = None,
    core_direction: Sequence[float] = (0.0, 0.0, -1.0),
    cone_angle_degrees: float = 90.0,
    skyrmion_type: str = "bloch",
    ferromagnetic_background: float = 0.0,
    helicity: int = 1,
) -> MagneticTexture:
    """Build a compact two-q square skyrmion-lattice texture."""

    core = _unit_vector(core_direction, "core_direction")
    if q_directions is None:
        q1 = _unit_vector((1.0, 0.0, 0.0), "q1_direction")
        q_dirs = (q1, _rotate_vector(q1, core, np.pi / 2.0))
    else:
        if len(q_directions) < 2:
            raise MagneticStructureError("square skyrmion texture requires two q directions")
        q_dirs = tuple(_unit_vector(direction, "q_direction") for direction in q_directions[:2])
    return _build_skyrmion_texture(
        magnitude=magnitude,
        q_coeffs=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        q_directions=q_dirs,
        core_direction=core,
        cone_angle_degrees=cone_angle_degrees,
        skyrmion_type=skyrmion_type,
        helicity=helicity,
        ferromagnetic_background=ferromagnetic_background,
    )


def skyrmion_q_basis_rlu(
    structure: Structure,
    *,
    q1_rlu: Sequence[float],
    core_direction: Sequence[float],
    lattice: str,
) -> tuple[tuple[float, float, float], ...]:
    """Return the physical q-basis rows for a square or hexagonal skyrmion."""

    q1 = _as_float_array(q1_rlu, "q1_rlu")
    core = _unit_vector(core_direction, "core_direction")
    reciprocal = np.linalg.inv(np.asarray(structure.lattice_vectors, dtype=float)).T
    q1_cart = q1 @ reciprocal
    q1_norm = float(np.linalg.norm(q1_cart))
    if q1_norm <= 0.0:
        raise MagneticStructureError("q1 vector must be non-zero")
    q1_dir = q1_cart / q1_norm
    if abs(float(q1_dir @ core)) > 1e-6:
        raise MagneticStructureError("q1 must lie in the plane perpendicular to the skyrmion core direction")
    angle = np.pi / 2.0 if lattice == "square" else 2.0 * np.pi / 3.0
    q2_cart = q1_norm * _rotate_vector(q1_dir, core, angle)
    q2_rlu = q2_cart @ np.linalg.inv(reciprocal)
    return (
        tuple(float(value) for value in q1),
        tuple(float(value) for value in q2_rlu),
        (0.0, 0.0, 0.0),
    )


def evaluate_texture_moment(texture: MagneticTexture, q_dots: Sequence[float]) -> np.ndarray:
    """Evaluate a texture in reduced phase coordinates ``k_i.r``."""

    dots = np.asarray(q_dots, dtype=float)
    moment = np.zeros(3, dtype=complex)
    for term in texture.terms:
        q = np.asarray(term.q_coeff, dtype=float)
        q_dot = float(q @ dots[:3])
        moment += term.coefficient.to_array() * np.exp(-2j * np.pi * q_dot)
    return moment.real


def evaluate_texture_grid(texture: MagneticTexture, resolution: Sequence[int] = (32, 32, 1)) -> np.ndarray:
    """Evaluate a texture on a regular reduced-coordinate grid."""

    res = _texture_grid_resolution(resolution)
    axes = [np.linspace(0.0, 1.0, value, endpoint=False) for value in res]
    uu, vv, ww = np.meshgrid(*axes, indexing="ij")
    dots = np.stack([uu, vv, ww], axis=-1)
    moments = np.zeros(res + (3,), dtype=complex)
    for term in texture.terms:
        q = np.asarray(term.q_coeff, dtype=float)
        q_dot = np.tensordot(dots, q, axes=([3], [0]))
        moments += term.coefficient.to_array()[np.newaxis, np.newaxis, np.newaxis, :] * np.exp(-2j * np.pi * q_dot)[..., np.newaxis]
    return moments.real


def texture_diagnostics(texture: MagneticTexture, resolution: Sequence[int] = (32, 32, 1)) -> MagneticTextureDiagnostics:
    """Return sampled moment-magnitude diagnostics for a texture."""

    if not texture.terms:
        return MagneticTextureDiagnostics(0, 0.0, 0.0, 0.0, 0.0)
    magnitudes = np.linalg.norm(evaluate_texture_grid(texture, resolution), axis=-1)
    return MagneticTextureDiagnostics(
        term_count=len(texture.terms),
        magnitude_min=float(np.min(magnitudes)),
        magnitude_mean=float(np.mean(magnitudes)),
        magnitude_max=float(np.max(magnitudes)),
        magnitude_std=float(np.std(magnitudes)),
    )


def normalize_texture_magnitude(
    texture: MagneticTexture,
    *,
    grid_resolution: Sequence[int] = (32, 32, 1),
    threshold: float = 1e-3,
    target_magnitude: float = 1.0,
) -> MagneticTexture:
    """Return an FFT-modulated texture that approximates constant magnitude."""

    if not texture.terms:
        return texture
    res = _texture_grid_resolution(grid_resolution)
    moments = evaluate_texture_grid(texture, res)
    magnitudes = np.linalg.norm(moments, axis=-1)
    inverse = np.zeros(res, dtype=float)
    mask = magnitudes > 1e-10
    inverse[mask] = 1.0 / magnitudes[mask]
    inverse[~mask] = 1e6
    f_coeffs = np.fft.fftn(inverse) / np.prod(res)
    max_coeff = float(np.max(np.abs(f_coeffs)))
    if max_coeff <= 0.0:
        return texture
    significant = np.where(np.abs(f_coeffs) > float(threshold) * max_coeff)
    freqs = [np.fft.fftfreq(value, d=1.0 / value).astype(int) for value in res]
    grid_coords = np.array(np.meshgrid(*freqs, indexing="ij"))
    out: list[MagneticTextureTerm] = []
    for term in texture.terms:
        base_q = np.asarray(term.q_coeff, dtype=float)
        coefficient = term.coefficient.to_array()
        for row in range(len(significant[0])):
            idx = tuple(int(values[row]) for values in significant)
            harmonic = grid_coords[:, idx[0], idx[1], idx[2]]
            out.append(MagneticTextureTerm.from_vector(base_q + harmonic, coefficient * f_coeffs[idx]))
    normalized = MagneticTexture(tuple(out)).combine_terms()
    scale = float(target_magnitude)
    if np.isclose(scale, 1.0):
        return normalized
    return MagneticTexture(
        tuple(
            MagneticTextureTerm.from_vector(term.q_coeff, scale * term.coefficient.to_array())
            for term in normalized.terms
        )
    ).combine_terms()


def prune_texture_terms(
    texture: MagneticTexture,
    *,
    threshold: float | None = None,
    top_n: int | None = None,
) -> MagneticTexture:
    """Remove weak texture terms by absolute coefficient norm or top-N rank."""

    terms = list(texture.terms)
    if threshold is not None:
        terms = [term for term in terms if float(np.linalg.norm(term.coefficient.to_array())) >= float(threshold)]
    if top_n is not None and int(top_n) >= 0 and len(terms) > int(top_n):
        terms = sorted(terms, key=lambda term: float(np.linalg.norm(term.coefficient.to_array())), reverse=True)[: int(top_n)]
    return MagneticTexture(tuple(terms)).combine_terms()


def structure_to_ase_atoms(structure: Structure) -> Atoms:
    """Convert a MuFinder structure entity to an ASE Atoms object."""

    return Atoms(
        symbols=list(structure.species),
        scaled_positions=np.asarray(structure.fractional_positions, dtype=float),
        cell=np.asarray(structure.lattice_vectors, dtype=float),
        pbc=[True, True, True],
    )


def magnetic_model_from_legacy_structure(structure: Structure) -> MagneticStructureModel:
    """Build a v1 magnetic model from current scalar-spin or MCIF metadata."""

    q_basis = _legacy_q_basis_rlu(structure)
    propagation = q_basis[0]
    q_coeff = (1.0, 0.0, 0.0) if any(abs(value) > 1e-12 for value in propagation) else (0.0, 0.0, 0.0)
    moment_entries = _legacy_moment_entries(structure)
    components: list[MagneticComponent] = []
    for entry in moment_entries:
        index = int(entry["atom_index"])
        vector = entry["vector"]
        if not any(abs(value) > 1e-12 for value in vector):
            continue
        label = str(entry.get("label") or f"{structure.species[index]}{index + 1}")
        coefficient = _legacy_fourier_coefficient(structure, atom_index=index, moment=vector, q_rlu=propagation)
        metadata: dict[str, object] = {}
        if entry.get("source"):
            metadata["source"] = entry["source"]
        if entry.get("mcif_label"):
            metadata["mcif_label"] = entry["mcif_label"]
        components.append(
            MagneticComponent(
                component_id=f"legacy_atom_{index + 1}",
                label=label,
                kind="legacy_moment",
                selector=AtomSelector.indices([index]),
                terms=(FourierTerm(q_coeff=q_coeff, coefficient=ComplexVector.from_vector(coefficient)),),
                metadata=metadata,
            )
        )
    return MagneticStructureModel(
        q_basis_rlu=q_basis,
        components=tuple(components),
        metadata={"source": "legacy_host_magnetic_structure"},
    )


def _build_single_term_component(
    *,
    kind: str,
    label: str,
    selector: AtomSelector,
    q_coeff: Sequence[float],
    coefficient: np.ndarray,
    phase_turns: float,
    component_id: str | None,
) -> MagneticComponent:
    return MagneticComponent(
        component_id=component_id or f"component-{uuid4()}",
        label=str(label).strip() or kind.replace("_", " ").title(),
        kind=kind,
        selector=selector,
        terms=(
            FourierTerm(
                q_coeff=_as_float_tuple(q_coeff, "q_coeff"),
                coefficient=ComplexVector.from_vector(coefficient),
            ),
        ),
        phase_turns=float(phase_turns),
        enabled=True,
    )


def _build_skyrmion_texture(
    *,
    magnitude: float,
    q_coeffs: Sequence[Sequence[float]],
    q_directions: Sequence[Sequence[float]],
    core_direction: np.ndarray,
    cone_angle_degrees: float,
    skyrmion_type: str,
    helicity: int,
    ferromagnetic_background: float,
) -> MagneticTexture:
    helicity = int(helicity)
    if helicity not in {-1, 1}:
        raise MagneticStructureError("helicity must be +1 or -1")
    kind = str(skyrmion_type).strip().lower()
    if kind not in {"bloch", "neel"}:
        raise MagneticStructureError("skyrmion_type must be 'bloch' or 'neel'")
    angle = np.deg2rad(float(cone_angle_degrees))
    q_dirs = tuple(_unit_vector(direction, "q_direction") for direction in q_directions)
    rot_mag = float(magnitude) * float(np.sin(angle)) / max(1, len(q_dirs))
    terms: list[MagneticTextureTerm] = []
    for q_coeff, q_dir in zip(q_coeffs, q_dirs, strict=True):
        if abs(float(q_dir @ core_direction)) > 1e-6:
            raise MagneticStructureError("skyrmion q directions must be perpendicular to the core direction")
        if kind == "bloch":
            e1 = core_direction
            e2 = helicity * _unit_vector(np.cross(core_direction, q_dir), "bloch_direction")
        else:
            e1 = core_direction
            e2 = helicity * q_dir
        terms.append(MagneticTextureTerm.from_vector(q_coeff, rot_mag * (e1 + 1j * e2)))
    static_mag = float(magnitude) * float(np.cos(angle)) + float(ferromagnetic_background)
    if abs(static_mag) > 1e-12:
        terms.append(MagneticTextureTerm.from_vector((0.0, 0.0, 0.0), static_mag * core_direction.astype(complex)))
    return MagneticTexture(tuple(terms)).combine_terms()


def _rotate_vector(vector: Sequence[float], axis: Sequence[float], angle_rad: float) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    a = _unit_vector(axis, "rotation_axis")
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    return v * cos_a + np.cross(a, v) * sin_a + a * float(np.dot(a, v)) * (1.0 - cos_a)


def _texture_grid_resolution(resolution: Sequence[int]) -> tuple[int, int, int]:
    raw = list(resolution) + [1, 1, 1]
    res = tuple(max(1, int(value)) for value in raw[:3])
    return res


def _legacy_propagation_vector(structure: Structure) -> tuple[float, float, float]:
    source: object = structure.properties.get("propagation_vector")
    if source is None:
        magnetic = structure.properties.get("magnetic_structure")
        if isinstance(magnetic, Mapping):
            source = magnetic.get("propagation_vector")
    try:
        return _as_float_tuple(source, "propagation_vector")
    except MagneticStructureError:
        return (0.0, 0.0, 0.0)


def _legacy_q_basis_rlu(structure: Structure) -> tuple[tuple[float, float, float], ...]:
    """Return up to three legacy/MCIF propagation vectors as a q basis."""
    vectors: list[tuple[float, float, float]] = []
    magnetic = structure.properties.get("magnetic_structure")
    if isinstance(magnetic, Mapping):
        raw_vectors = magnetic.get("propagation_vectors_rlu")
        if isinstance(raw_vectors, Sequence) and not isinstance(raw_vectors, (str, bytes)):
            for raw_vector in list(raw_vectors)[:3]:
                try:
                    vectors.append(_as_float_tuple(raw_vector, "propagation vector"))
                except MagneticStructureError:
                    continue

    if not vectors:
        vectors.append(_legacy_propagation_vector(structure))
    while len(vectors) < 3:
        vectors.append((0.0, 0.0, 0.0))
    return tuple(vectors[:3])


def _legacy_moment_entries(structure: Structure) -> list[dict[str, object]]:
    atom_count = len(structure.species)
    entries: list[dict[str, object]] = []
    magnetic = structure.properties.get("magnetic_structure")
    if isinstance(magnetic, Mapping):
        moments = magnetic.get("moments")
        if isinstance(moments, Sequence) and not isinstance(moments, (str, bytes)):
            for index, moment in enumerate(moments):
                if not isinstance(moment, Mapping):
                    continue
                atom_index = _legacy_moment_atom_index(moment, fallback=index, atom_count=atom_count)
                if atom_index is None:
                    continue
                try:
                    vector = _as_float_tuple(moment.get("vector", (0.0, 0.0, 0.0)), "moment vector")
                except MagneticStructureError:
                    continue
                label = str(moment.get("label") or moment.get("atom_label") or f"{structure.species[atom_index]}{atom_index + 1}")
                entries.append(
                    {
                        "atom_index": atom_index,
                        "label": label,
                        "vector": vector,
                        "source": "mcif",
                        "mcif_label": str(moment.get("label", label)),
                    }
                )
    if any(any(abs(value) > 1e-12 for value in entry["vector"]) for entry in entries):
        return entries

    raw_spins = structure.properties.get("atomic_spins", structure.properties.get("initial_magnetic_moments"))
    if isinstance(raw_spins, Sequence) and not isinstance(raw_spins, (str, bytes)):
        for index, value in enumerate(list(raw_spins)[:atom_count]):
            try:
                vector = (0.0, 0.0, float(value))
            except (TypeError, ValueError):
                vector = (0.0, 0.0, 0.0)
            entries.append(
                {
                    "atom_index": index,
                    "label": f"{structure.species[index]}{index + 1}",
                    "vector": vector,
                }
            )
    return entries


def _legacy_moment_atom_index(moment: Mapping[str, object], *, fallback: int, atom_count: int) -> int | None:
    raw_index = moment.get("atom_index", fallback)
    try:
        atom_index = int(raw_index)
    except (TypeError, ValueError):
        atom_index = fallback
    if atom_index < 0 or atom_index >= atom_count:
        return None
    return atom_index


def _legacy_fourier_coefficient(
    structure: Structure,
    *,
    atom_index: int,
    moment: Sequence[float],
    q_rlu: Sequence[float],
) -> np.ndarray:
    """Return an imported Fourier coefficient that reconstructs the unit-cell moment."""
    coefficient = np.asarray(moment, dtype=float).astype(complex)
    q = np.asarray(q_rlu, dtype=float)
    if not np.any(np.abs(q) > 1e-12):
        return coefficient
    frac = np.asarray(structure.fractional_positions, dtype=float)
    if frac.shape != (len(structure.species), 3):
        return coefficient
    phase_turns = float(q @ frac[int(atom_index)])
    return coefficient * np.exp(2j * np.pi * phase_turns)


def _equivalent_atom_representatives(structure: Structure) -> tuple[int, ...]:
    equivalents = _equivalent_atoms(structure)
    if equivalents is None:
        return ()
    return tuple(int(index) for index in sorted(set(int(value) for value in equivalents)))


def _equivalent_atoms(structure: Structure) -> np.ndarray | None:
    dataset = get_crystal_symmetry_dataset(
        (
            np.asarray(structure.lattice_vectors, dtype=float),
            np.asarray(structure.fractional_positions, dtype=float),
            np.asarray(structure_to_ase_atoms(structure).get_atomic_numbers(), dtype=int),
        )
    )
    if dataset is None:
        return None
    if hasattr(dataset, "equivalent_atoms"):
        return np.asarray(dataset.equivalent_atoms, dtype=int)
    return np.asarray(dataset["equivalent_atoms"], dtype=int)


def _as_float_array(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,):
        raise MagneticStructureError(f"{name} must be a three-vector")
    return arr


def _unit_vector(values: Sequence[float], name: str) -> np.ndarray:
    arr = _as_float_array(values, name)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise MagneticStructureError(f"{name} must be non-zero")
    return arr / norm


def _as_complex_vector(values: Sequence[complex | float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=complex)
    if arr.shape != (3,):
        raise MagneticStructureError(f"{name} must be a complex three-vector")
    return arr


def _as_float_tuple(values: object, name: str) -> tuple[float, float, float]:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MagneticStructureError(f"{name} must be a three-vector") from exc
    if arr.shape != (3,):
        raise MagneticStructureError(f"{name} must be a three-vector")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _as_q_basis(values: object) -> tuple[tuple[float, float, float], ...]:
    if values in (None, ""):
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MagneticStructureError("q_basis_rlu must have shape (3, 3)") from exc
    if arr.shape != (3, 3):
        raise MagneticStructureError("q_basis_rlu must have shape (3, 3)")
    return tuple(tuple(float(value) for value in row) for row in arr)


def _as_supercell_repeats(supercell: Sequence[int]) -> tuple[int, int, int]:
    if len(supercell) != 3:
        raise MagneticStructureError("supercell must contain three repeat counts")
    repeats = tuple(int(value) for value in supercell)
    if any(value <= 0 for value in repeats):
        raise MagneticStructureError("supercell repeat counts must be positive")
    return repeats
