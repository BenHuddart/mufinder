"""Magnetic-structure helpers for propagation-vector based field workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from uuid import uuid4

from ase import Atoms
import numpy as np

from mufinder.domain.magnetic_properties import propagation_vector_from_structure
from mufinder.domain.models import Structure
from mufinder.domain.text import plural
from mufinder.science.ase_bridge import structure_to_ase_atoms
from mufinder.science.lattice import (
    cartesian_to_reciprocal,
    crystal_axis_to_cartesian,
    lattice_matrix,
    reciprocal_to_cartesian,
)
from mufinder.science.spglib_compat import get_crystal_symmetry_dataset, spglib_cell

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
            MagneticTextureTerm.from_json(item) for item in raw_terms if isinstance(item, Mapping)
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
    """A user-facing component in a magnetic structure.

    ``atom_phase_turns`` maps a unit-cell atom index to an extra phase in turns
    that applies to that atom only, added to ``phase_turns``; it sets sublattice
    phases without one component per atom.
    """

    component_id: str
    label: str
    kind: str
    selector: AtomSelector
    terms: tuple[FourierTerm, ...]
    phase_turns: float = 0.0
    enabled: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)
    atom_phase_turns: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "atom_phase_turns", _normalized_atom_phase_turns(self.atom_phase_turns)
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "MagneticComponent":
        raw_terms = payload.get("terms", ())
        terms = tuple(
            FourierTerm.from_json(term) for term in raw_terms if isinstance(term, Mapping)
        )
        return cls(
            component_id=str(payload.get("component_id", "")).strip(),
            label=str(payload.get("label", "")).strip(),
            kind=str(payload.get("kind", "custom")).strip() or "custom",
            selector=AtomSelector.from_json(payload.get("selector", {})),
            terms=terms,
            phase_turns=float(payload.get("phase_turns", 0.0) or 0.0),
            enabled=bool(payload.get("enabled", True)),
            metadata=dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata", {}), Mapping)
            else {},
            atom_phase_turns=_saved_atom_phase_turns(payload.get("atom_phase_turns")),
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
            "atom_phase_turns": {
                str(index): float(phase) for index, phase in self.atom_phase_turns.items()
            },
        }


def _saved_atom_phase_turns(values: object) -> dict[int, float]:
    """Return the valid entries of saved ``atom_phase_turns``, dropping malformed ones as loading does."""

    if not isinstance(values, Mapping):
        return {}
    kept: dict[int, float] = {}
    for raw_index, raw_phase in values.items():
        try:
            kept.update(_normalized_atom_phase_turns({raw_index: raw_phase}))
        except MagneticStructureError:
            continue
    return kept


def _normalized_atom_phase_turns(values: object) -> dict[int, float]:
    if not isinstance(values, Mapping):
        raise MagneticStructureError("atom_phase_turns must map atom indices to phases in turns")
    normalized: dict[int, float] = {}
    for raw_index, raw_phase in values.items():
        try:
            index = int(raw_index)
            phase = float(raw_phase)
        except (TypeError, ValueError) as exc:
            raise MagneticStructureError(
                "atom_phase_turns must map atom indices to phases in turns"
            ) from exc
        if index < 0:
            raise MagneticStructureError("atom_phase_turns atom indices must not be negative")
        if not np.isfinite(phase):
            raise MagneticStructureError("atom_phase_turns phases must be finite")
        normalized[index] = phase
    return normalized


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


def enabled_magnetic_components(model: MagneticStructureModel) -> tuple[MagneticComponent, ...]:
    """Return the components that contribute to moments and fields, in model order."""

    return tuple(component for component in model.components if component.enabled)


def duplicate_magnetic_components(
    model: MagneticStructureModel,
    component_ids: Sequence[str],
) -> tuple[MagneticStructureModel, tuple[str, ...]]:
    """Insert a copy of each listed component directly after its original.

    A copy keeps the original's selector, Fourier terms, phase, enabled state and
    metadata, so it shares the original's q-basis slots. It gets a fresh
    ``component_id`` and a ``" (copy)"`` label suffix. Returns the new model and
    the copies' ids in model order.
    """

    selected = {str(value) for value in component_ids}
    components: list[MagneticComponent] = []
    copy_ids: list[str] = []
    for component in model.components:
        components.append(component)
        if component.component_id not in selected:
            continue
        duplicate = replace(
            component,
            component_id=f"component-{uuid4()}",
            label=f"{component.label} (copy)",
            metadata=deepcopy(dict(component.metadata)),
        )
        components.append(duplicate)
        copy_ids.append(duplicate.component_id)
    return replace(model, components=tuple(components)), tuple(copy_ids)


def set_magnetic_components_enabled(
    model: MagneticStructureModel,
    component_ids: Sequence[str],
    enabled: bool,
) -> MagneticStructureModel:
    """Enable or disable the listed components; other components are unchanged."""

    selected = {str(value) for value in component_ids}
    components = tuple(
        replace(component, enabled=bool(enabled))
        if component.component_id in selected
        else component
        for component in model.components
    )
    return replace(model, components=components)


def move_magnetic_components(
    model: MagneticStructureModel,
    component_ids: Sequence[str],
    offset: int,
) -> MagneticStructureModel:
    """Move the listed components ``offset`` places (negative is towards the start).

    Listed components keep their relative order. A component stops when it
    reaches either end or another listed component that has already stopped, so
    a selection never wraps round. Component order does not change the summed
    moments; it is the order the builder lists and evaluates components in.
    """

    selected = {str(value) for value in component_ids}
    components = list(model.components)
    step = -1 if offset < 0 else 1
    for _ in range(abs(int(offset))):
        indices = range(1, len(components)) if step < 0 else range(len(components) - 2, -1, -1)
        moved = False
        for index in indices:
            neighbour = index + step
            if (
                components[index].component_id in selected
                and components[neighbour].component_id not in selected
            ):
                components[index], components[neighbour] = components[neighbour], components[index]
                moved = True
        if not moved:
            break
    return replace(model, components=tuple(components))


_Q_COEFF_TOLERANCE = 1e-12
_Q_BASIS_MATCH_TOLERANCE = 1e-9
# Component kinds whose moment plane or cone axis is constructed from the q direction.
_Q_DIRECTION_COMPONENT_KINDS = frozenset({"helix", "cycloid", "conical"})


@dataclass(frozen=True, slots=True)
class QBasisSlot:
    """One row of a model's stored propagation-vector basis and the components that use it.

    ``component_ids`` lists, in model order, every component (enabled or not)
    with a Fourier term whose ``q_coeff`` has a non-zero entry for this row.
    ``texture_component_ids`` is the subset that are grouped textures.
    ``direction_component_ids`` is the subset whose moment plane or cone axis
    was built from the q direction (helix, cycloid and conical components).
    """

    index: int
    q_rlu: tuple[float, float, float]
    component_ids: tuple[str, ...] = ()
    texture_component_ids: tuple[str, ...] = ()
    direction_component_ids: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return f"q{self.index + 1}"

    @property
    def in_use(self) -> bool:
        return bool(self.component_ids)


def is_texture_component(component: MagneticComponent) -> bool:
    """Return whether ``component`` is a grouped multi-q texture."""

    return component.kind == "texture" or component.metadata.get("type") == "texture"


def q_basis_slots(model: MagneticStructureModel) -> tuple[QBasisSlot, ...]:
    """Describe each of the model's three q-basis rows and which components reference it."""

    users: list[list[str]] = [[], [], []]
    texture_users: list[list[str]] = [[], [], []]
    direction_users: list[list[str]] = [[], [], []]
    for component in model.components:
        for index in component_q_basis_indices(component):
            users[index].append(component.component_id)
            if is_texture_component(component):
                texture_users[index].append(component.component_id)
            elif component.kind in _Q_DIRECTION_COMPONENT_KINDS:
                direction_users[index].append(component.component_id)
    return tuple(
        QBasisSlot(
            index=index,
            q_rlu=tuple(float(value) for value in row),
            component_ids=tuple(users[index]),
            texture_component_ids=tuple(texture_users[index]),
            direction_component_ids=tuple(direction_users[index]),
        )
        for index, row in enumerate(model.q_basis_array)
    )


def component_q_basis_indices(component: MagneticComponent) -> tuple[int, ...]:
    """Return the sorted q-basis row indices that any of ``component``'s terms reference."""

    used: set[int] = set()
    for term in component.terms:
        for index, value in enumerate(tuple(term.q_coeff)[:3]):
            if abs(float(value)) > _Q_COEFF_TOLERANCE:
                used.add(index)
    return tuple(sorted(used))


def q_basis_placeholder(model: MagneticStructureModel) -> str | None:
    """Return the line that stands in for the q-basis rows, or ``None`` when they are needed.

    The rows are needed as soon as an enabled component references one. That
    covers a reference to a row holding no vector, whose q is zero until the
    row is filled in, because the builder checklist asks for exactly that.
    """

    if not model.components:
        return "No components yet"
    if any(component_q_basis_indices(c) for c in enabled_magnetic_components(model)):
        return None
    if used_q_basis_indices(model):
        return "All enabled components have q = 0"
    return "All components have q = 0"


def q_coeff_expression(q_coeff: Sequence[float]) -> str:
    """Write a q-basis coefficient vector as a combination of ``q1``, ``q2`` and ``q3``.

    ``(1, 0, 0)`` is ``q1``, ``(1, -1, 0)`` is ``q1 − q2``, ``(-1, -1, 0)`` is
    ``−q1 − q2`` and ``(0.5, 0, 0)`` is ``0.5q1``. An all-zero vector is ``0``.
    """

    parts: list[str] = []
    for index, raw in enumerate(tuple(q_coeff)[:3]):
        value = float(raw)
        if abs(value) <= _Q_COEFF_TOLERANCE:
            continue
        magnitude = abs(value)
        scale = "" if abs(magnitude - 1.0) <= _Q_COEFF_TOLERANCE else f"{magnitude:.4g}"
        term = f"{scale}q{index + 1}"
        if not parts:
            parts.append(term if value > 0 else f"−{term}")
        else:
            parts.append(f"+ {term}" if value > 0 else f"− {term}")
    return " ".join(parts) or "0"


def set_q_basis_vector(
    model: MagneticStructureModel,
    index: int,
    q_rlu: Sequence[float],
) -> MagneticStructureModel:
    """Store ``q_rlu`` in q-basis row ``index``; every component using that row follows it.

    Components keep their ``q_coeff`` values, so a component that references the
    row changes wavevector with it. A row used by a grouped texture cannot be
    edited here, because the texture's moment directions were built from its
    q directions; edit the texture instead. An edit must keep the direction of
    every wavevector of each helix, cycloid or conical component on the row,
    since those moment planes and cone axes were built from it; a component
    that combines several rows therefore blocks any change to them. A row that a
    component uses cannot be set to zero, and no two rows may hold the same
    non-zero vector.
    """

    if not 0 <= int(index) < 3:
        raise MagneticStructureError("q-basis row index must be 0, 1 or 2")
    q = _as_float_array(q_rlu, "q_rlu")
    if not np.all(np.isfinite(q)):
        raise MagneticStructureError("q_rlu must be finite")
    slot = q_basis_slots(model)[int(index)]
    if slot.texture_component_ids:
        raise MagneticStructureError(
            f"{slot.label} is used by a texture; edit the texture to change its propagation vectors"
        )
    if slot.in_use and np.allclose(q, 0.0, atol=_Q_COEFF_TOLERANCE):
        raise MagneticStructureError(
            f"{slot.label} is used by a component, so it cannot be zero; edit the component to make it q = 0"
        )
    basis = [tuple(float(value) for value in row) for row in model.q_basis_array]
    if not np.allclose(q, 0.0, atol=_Q_COEFF_TOLERANCE):
        for other, row in enumerate(basis):
            if other != int(index) and np.allclose(row, q, atol=_Q_BASIS_MATCH_TOLERANCE, rtol=0.0):
                raise MagneticStructureError(f"q{other + 1} already holds this propagation vector")
    basis[int(index)] = tuple(float(value) for value in q)
    edited = replace(model, q_basis_rlu=tuple(basis))
    # A direction-built component may combine several rows, so compare each term's full wavevector.
    direction_ids = set(slot.direction_component_ids)
    for component in model.components:
        if component.component_id not in direction_ids:
            continue
        for term in component.terms:
            before = model.q_coeff_to_rlu(term.q_coeff)
            if np.allclose(before, 0.0, atol=_Q_COEFF_TOLERANCE):
                continue
            if not _is_positive_multiple(edited.q_coeff_to_rlu(term.q_coeff), before):
                raise MagneticStructureError(
                    f"this edit would turn the q of '{component.label}', a helix, cycloid or conical "
                    "component whose moments were built around its direction; only rescalings that keep "
                    "its direction are allowed here, so edit the component instead"
                )
    return edited


def q_basis_combination_rlu(model: MagneticStructureModel, q_coeff: Sequence[float]) -> np.ndarray:
    """Return the wavevector, in r.l.u., of a combination of the model's stored q-basis rows.

    Every row the combination uses must hold a non-zero vector, and the
    combination itself must be non-zero: a component with q = 0 needs no row.
    """

    coeff = _as_float_array(q_coeff, "q_coeff")
    if not np.all(np.isfinite(coeff)):
        raise MagneticStructureError("q-basis coefficients must be finite")
    used = [index for index, value in enumerate(coeff) if abs(float(value)) > _Q_COEFF_TOLERANCE]
    if not used:
        raise MagneticStructureError("choose at least one q-basis row, or enter q = 0 directly")
    basis = model.q_basis_array
    for index in used:
        if np.allclose(basis[index], 0.0, atol=_Q_COEFF_TOLERANCE):
            raise MagneticStructureError(
                f"q{index + 1} is not set; store a propagation vector in it first"
            )
    q_rlu = coeff @ basis
    if np.allclose(q_rlu, 0.0, atol=_Q_COEFF_TOLERANCE):
        raise MagneticStructureError(
            f"{q_coeff_expression(coeff)} gives q = 0; enter q = 0 directly instead of combining rows"
        )
    return q_rlu


@dataclass(frozen=True, slots=True)
class BuilderChecklistItem:
    """One step of the magnetic-structure builder's checklist, derived from model state.

    ``state`` is ``"done"``, ``"todo"`` or ``"problem"``; ``detail`` says why.
    """

    key: str
    title: str
    state: str
    detail: str = ""


CHECKLIST_DONE = "done"
CHECKLIST_TODO = "todo"
CHECKLIST_PROBLEM = "problem"
_CRYSTAL_AXIS_NAMES = ("a", "b", "c")


def magnetic_period_along_axes(
    q_vectors_rlu: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-8,
    max_period: int = 64,
) -> tuple[int | None, ...]:
    """Return the cell repeats along a, b and c that one magnetic period needs.

    Each axis's period is the least common multiple, over ``q_vectors_rlu``, of
    the smallest ``n <= max_period`` with ``n * q`` an integer along that axis.
    An axis is ``None`` when some wavevector is incommensurate along it.
    """

    periods: list[int | None] = [1, 1, 1]
    for raw in q_vectors_rlu:
        q = _as_float_array(raw, "q_rlu")
        for axis in range(3):
            if periods[axis] is None:
                continue
            period = next(
                (
                    n
                    for n in range(1, int(max_period) + 1)
                    if abs(n * float(q[axis]) - round(n * float(q[axis]))) <= float(tolerance)
                ),
                None,
            )
            periods[axis] = None if period is None else int(np.lcm(int(periods[axis]), period))
    return tuple(periods)


def magnetic_builder_checklist(
    model: MagneticStructureModel,
    *,
    has_host: bool,
    preview_repeats: Sequence[int],
    max_preview_repeat: int = 12,
) -> tuple[BuilderChecklistItem, ...]:
    """Describe how far a builder model is from being ready, and what is wrong with it.

    The steps are: components added (from host spins or by hand), the q basis
    set, the preview supercell checked, and ready to use. A step is a problem
    when every component is disabled, an enabled component has zero moment, an
    enabled component uses a zero q-basis row, or the preview does not repeat
    along an axis where a wavevector is incommensurate. A commensurate period
    longer than ``max_preview_repeat`` only asks for the largest preview.
    """

    repeats = _as_supercell_repeats(preview_repeats)
    enabled = enabled_magnetic_components(model)
    items: list[BuilderChecklistItem] = []

    components_title = "Add components or build them from host spins"
    zero_moment = [
        component.label
        for component in enabled
        if all(
            np.allclose(term.coefficient.to_array(), 0.0, atol=_Q_COEFF_TOLERANCE)
            for term in component.terms
        )
    ]
    if not has_host:
        items.append(
            BuilderChecklistItem(
                "components", components_title, CHECKLIST_TODO, "Load a host structure first."
            )
        )
    elif not model.components:
        items.append(
            BuilderChecklistItem(
                "components",
                components_title,
                CHECKLIST_TODO,
                "Use the 'Add' menu.",
            )
        )
    elif not enabled:
        items.append(
            BuilderChecklistItem(
                "components",
                components_title,
                CHECKLIST_PROBLEM,
                "Every component is disabled, so the structure has no moments.",
            )
        )
    elif zero_moment:
        items.append(
            BuilderChecklistItem(
                "components",
                components_title,
                CHECKLIST_PROBLEM,
                "Zero moment: " + ", ".join(f"'{label}'" for label in zero_moment) + ".",
            )
        )
    else:
        items.append(
            BuilderChecklistItem(
                "components",
                components_title,
                CHECKLIST_DONE,
                f"{plural(len(model.components), 'component')}, {len(enabled)} enabled.",
            )
        )

    basis_title = "Set the propagation-vector basis"
    basis = model.q_basis_array
    used_rows = sorted(
        {index for component in enabled for index in component_q_basis_indices(component)}
    )
    zero_rows = [
        index for index in used_rows if np.allclose(basis[index], 0.0, atol=_Q_COEFF_TOLERANCE)
    ]
    if not enabled:
        items.append(
            BuilderChecklistItem(
                "q_basis", basis_title, CHECKLIST_TODO, "Add or enable a component first."
            )
        )
    elif zero_rows:
        rows = ", ".join(f"q{index + 1}" for index in zero_rows)
        items.append(
            BuilderChecklistItem(
                "q_basis",
                basis_title,
                CHECKLIST_PROBLEM,
                f"{rows} is used by an enabled component but holds no propagation vector.",
            )
        )
    elif not used_rows:
        items.append(
            BuilderChecklistItem(
                "q_basis",
                basis_title,
                CHECKLIST_DONE,
                "Not needed: every enabled component has q = 0.",
            )
        )
    else:
        rows = ", ".join(f"q{index + 1}" for index in used_rows)
        items.append(
            BuilderChecklistItem(
                "q_basis", basis_title, CHECKLIST_DONE, f"Enabled components use {rows}."
            )
        )

    preview_title = "Check the preview supercell"
    wavevectors = [
        model.q_coeff_to_rlu(term.q_coeff)
        for component in enabled
        for term in component.terms
        if not np.allclose(term.coefficient.to_array(), 0.0, atol=_Q_COEFF_TOLERANCE)
    ]
    repeat_text = "×".join(str(value) for value in repeats)
    if not wavevectors:
        items.append(
            BuilderChecklistItem(
                "preview", preview_title, CHECKLIST_TODO, "Nothing to preview yet."
            )
        )
    else:
        periods = magnetic_period_along_axes(wavevectors)
        needed = [
            None if period is None else min(period, int(max_preview_repeat)) for period in periods
        ]
        partial = any(period is not None and period > int(max_preview_repeat) for period in periods)
        unrepeated = [
            _CRYSTAL_AXIS_NAMES[axis]
            for axis, period in enumerate(periods)
            if period is None and repeats[axis] < 2
        ]
        short = [
            axis for axis, count in enumerate(needed) if count is not None and repeats[axis] < count
        ]
        if unrepeated:
            axes = ", ".join(unrepeated)
            items.append(
                BuilderChecklistItem(
                    "preview",
                    preview_title,
                    CHECKLIST_PROBLEM,
                    f"q is incommensurate along {axes} but the preview does not repeat along it; "
                    "raise that repeat to see the modulation.",
                )
            )
        elif short:
            target = "×".join(
                str(repeats[axis] if count is None else max(repeats[axis], count))
                for axis, count in enumerate(needed)
            )
            goal = "The largest preview needs" if partial else "One magnetic cell needs"
            items.append(
                BuilderChecklistItem(
                    "preview",
                    preview_title,
                    CHECKLIST_TODO,
                    f"{goal} a {target} preview; it shows {repeat_text}.",
                )
            )
        elif any(period is None for period in periods):
            items.append(
                BuilderChecklistItem(
                    "preview",
                    preview_title,
                    CHECKLIST_DONE,
                    f"The {repeat_text} preview repeats along every incommensurate axis.",
                )
            )
        elif partial:
            items.append(
                BuilderChecklistItem(
                    "preview",
                    preview_title,
                    CHECKLIST_DONE,
                    f"The magnetic cell is longer than the largest preview; {repeat_text} shows part of it.",
                )
            )
        else:
            items.append(
                BuilderChecklistItem(
                    "preview",
                    preview_title,
                    CHECKLIST_DONE,
                    f"The {repeat_text} preview shows at least one whole magnetic cell.",
                )
            )

    ready_title = "Use the structure"
    if all(item.state == CHECKLIST_DONE for item in items):
        items.append(
            BuilderChecklistItem(
                "ready",
                ready_title,
                CHECKLIST_DONE,
                "Ready: Use Structure stores it for field calculations.",
            )
        )
    elif any(item.state == CHECKLIST_PROBLEM for item in items):
        items.append(
            BuilderChecklistItem(
                "ready", ready_title, CHECKLIST_TODO, "Fix the flagged steps first."
            )
        )
    else:
        items.append(
            BuilderChecklistItem(
                "ready", ready_title, CHECKLIST_TODO, "Finish the steps above first."
            )
        )
    return tuple(items)


MAGNETIC_PRESET_KINDS = ("helix", "conical", "cycloid", "spin_density_wave")


def q_cartesian_direction(
    lattice_vectors: Sequence[Sequence[float]], q_rlu: Sequence[float]
) -> np.ndarray:
    """Return the Cartesian unit vector along ``q_rlu`` for a lattice whose rows are a, b and c."""

    q = _as_float_array(q_rlu, "q_rlu")
    if np.allclose(q, 0.0, atol=_Q_COEFF_TOLERANCE):
        raise MagneticStructureError("q must be non-zero to define a propagation direction")
    cartesian = reciprocal_to_cartesian(q, lattice_vectors, error=MagneticStructureError)
    return _unit_vector(cartesian, "q direction")


def crystal_axis_direction(lattice_vectors: Sequence[Sequence[float]], axis: str) -> np.ndarray:
    """Return the Cartesian unit vector along crystal axis ``"a"``, ``"b"`` or ``"c"``."""

    name = str(axis).strip().lower()
    if name not in _CRYSTAL_AXIS_NAMES:
        raise MagneticStructureError("crystal axis must be 'a', 'b' or 'c'")
    lattice = lattice_matrix(lattice_vectors, error=MagneticStructureError)
    return _unit_vector(lattice[_CRYSTAL_AXIS_NAMES.index(name)], f"axis {name}")


def reciprocal_axis_q_rlu(axis: str, wavevector: float) -> tuple[float, float, float]:
    """Return the r.l.u. wavevector ``wavevector`` along reciprocal axis ``a*``, ``b*`` or ``c*``."""

    name = str(axis).strip().lower().rstrip("*")
    if name not in _CRYSTAL_AXIS_NAMES:
        raise MagneticStructureError("reciprocal axis must be 'a*', 'b*' or 'c*'")
    value = float(wavevector)
    if not np.isfinite(value) or abs(value) <= _Q_COEFF_TOLERANCE:
        raise MagneticStructureError("the wavevector must be a non-zero number of r.l.u.")
    q = [0.0, 0.0, 0.0]
    q[_CRYSTAL_AXIS_NAMES.index(name)] = value
    return (q[0], q[1], q[2])


def reciprocal_axis_length(lattice_vectors: Sequence[Sequence[float]], axis: str) -> float:
    """Return ``|a*|`` (or ``|b*|``, ``|c*|``) in inverse Angstrom, with ``2π``: the ``|k|`` of 1 r.l.u. along that axis."""

    q = reciprocal_axis_q_rlu(axis, 1.0)
    cartesian = reciprocal_to_cartesian(q, lattice_vectors, error=MagneticStructureError)
    return 2.0 * np.pi * float(np.linalg.norm(cartesian))


def wavevector_rlu_from_inverse_angstrom(
    lattice_vectors: Sequence[Sequence[float]],
    magnitude: float,
    direction_rlu: Sequence[float],
) -> tuple[float, float, float]:
    """Return the r.l.u. wavevector of length ``magnitude`` (inverse Angstrom) along ``direction_rlu``.

    ``magnitude`` follows the crystallographic convention with ``2π``, so the
    Cartesian wavevector is ``k = magnitude x unit(direction)`` and its
    components are ``q_i = k · a_i / 2π``.
    """

    value = float(magnitude)
    if not np.isfinite(value) or abs(value) <= _Q_COEFF_TOLERANCE:
        raise MagneticStructureError("the wavevector must be a non-zero number of inverse Angstrom")
    k_cartesian = value * q_cartesian_direction(lattice_vectors, direction_rlu)
    q = cartesian_to_reciprocal(
        k_cartesian / (2.0 * np.pi), lattice_vectors, error=MagneticStructureError
    )
    return (float(q[0]), float(q[1]), float(q[2]))


def default_preset_reference_axis(
    lattice_vectors: Sequence[Sequence[float]], q_rlu: Sequence[float]
) -> str:
    """Return the crystal axis most nearly perpendicular to ``q_rlu``, the default reference for a preset."""

    q_direction = q_cartesian_direction(lattice_vectors, q_rlu)
    return min(
        _CRYSTAL_AXIS_NAMES,
        # Rounding lets ties between equally perpendicular axes fall to the first, a before b before c.
        key=lambda name: round(
            abs(float(crystal_axis_direction(lattice_vectors, name) @ q_direction)), 9
        ),
    )


def build_magnetic_preset_component(
    kind: str,
    *,
    lattice_vectors: Sequence[Sequence[float]],
    q_rlu: Sequence[float],
    q_coeff: Sequence[float],
    moment: float,
    reference_axis: str,
    selector: AtomSelector,
    label: str = "",
    handedness: int = 1,
    cone_angle_degrees: float = 45.0,
    phase_turns: float = 0.0,
    component_id: str | None = None,
) -> MagneticComponent:
    """Build a helix, conical, cycloid or spin-density-wave component from crystal axes.

    ``q_rlu`` sets the propagation direction and ``q_coeff`` is how the
    component stores it in the model's q basis. ``reference_axis`` (``"a"``,
    ``"b"`` or ``"c"``) places the moments relative to the crystal:

    * helix and conical: the rotation plane is perpendicular to q and the
      moment points along the reference axis, projected into that plane, at
      zero phase; a conical component's cone axis is q;
    * cycloid: the rotation plane holds q and the reference axis;
    * spin-density wave: the moment is modulated along the reference axis.

    ``handedness`` (+1 or -1) sets the sense of rotation, as in the
    component dialog. The moment maths is the existing component builders'.
    """

    preset = str(kind).strip().lower()
    if preset not in MAGNETIC_PRESET_KINDS:
        raise MagneticStructureError(
            "preset kind must be helix, conical, cycloid or spin_density_wave"
        )
    size = float(moment)
    if not np.isfinite(size) or size <= 0.0:
        raise MagneticStructureError("the moment must be positive")
    if int(handedness) not in {-1, 1}:
        raise MagneticStructureError("handedness must be +1 or -1")
    q_direction = q_cartesian_direction(lattice_vectors, q_rlu)
    reference = crystal_axis_direction(lattice_vectors, reference_axis)
    kwargs = {
        "selector": selector,
        "q_coeff": q_coeff,
        "phase_turns": phase_turns,
        "component_id": component_id,
    }
    if preset == "spin_density_wave":
        return build_spin_density_wave_component(
            label=label or "Spin-density wave", amplitude_vector=size * reference, **kwargs
        )
    projected = reference - float(reference @ q_direction) * q_direction
    if float(np.linalg.norm(projected)) <= 1e-6:
        raise MagneticStructureError(
            f"axis {str(reference_axis).strip().lower()} is parallel to q; choose a reference axis that is not"
        )
    primary = projected / float(np.linalg.norm(projected))
    sign = float(int(handedness))
    if preset == "cycloid":
        return build_cycloid_component(
            label=label or "Cycloid",
            real_axis_vector=size * primary,
            imaginary_axis_vector=size * sign * q_direction,
            **kwargs,
        )
    secondary = sign * np.cross(q_direction, primary)
    if preset == "conical":
        return build_conical_component(
            label=label or "Conical",
            total_moment=size,
            real_axis_direction=primary,
            imaginary_axis_direction=secondary,
            static_axis_direction=q_direction,
            cone_angle_degrees=cone_angle_degrees,
            **kwargs,
        )
    return build_helix_component(
        label=label or "Helix",
        real_axis_vector=size * primary,
        imaginary_axis_vector=size * secondary,
        **kwargs,
    )


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
        int(index) for index in selected_indices if 0 <= int(index) < len(structure.species)
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
            raise MagneticStructureError(
                "symmetry-equivalent selector requires a representative index"
            )
        representative = int(selector.values[0])
        equivalents = _equivalent_atoms(structure)
        if equivalents is None or representative < 0 or representative >= atom_count:
            raise MagneticStructureError("symmetry-equivalent atom groups could not be resolved")
        return tuple(
            int(index) for index in np.where(equivalents == equivalents[representative])[0]
        )
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
        atom_phases = _component_atom_phases(component, index_array, atom_count)
        for term in component.terms:
            q_rlu = model.q_coeff_to_rlu(term.q_coeff)
            fc = np.zeros((atom_count, 3), dtype=complex)
            phase_turns = positions[index_array] @ q_rlu + component_phase + atom_phases
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


def _component_atom_phases(
    component: MagneticComponent, index_array: np.ndarray, atom_count: int
) -> np.ndarray:
    """Return each selected atom's extra phase in turns, in the order of ``index_array``."""

    out_of_range = sorted(index for index in component.atom_phase_turns if index >= atom_count)
    if out_of_range:
        raise MagneticStructureError(
            f"component '{component.label}' sets a phase on atom {out_of_range[0]}, "
            f"but the structure has {atom_count} atoms"
        )
    return np.array(
        [component.atom_phase_turns.get(int(index), 0.0) for index in index_array], dtype=float
    )


def atom_phases_for_selection(
    structure: Structure, atom_phase_turns: Mapping[int, float], selector: AtomSelector
) -> dict[int, float]:
    """Return the non-zero ``atom_phase_turns`` on the atoms ``selector`` picks in ``structure``.

    Re-targeting a component keeps the phases of the atoms it still covers and
    drops the rest.
    """

    kept = set(resolve_atom_selector(structure, selector))
    return {
        int(index): float(phase)
        for index, phase in atom_phase_turns.items()
        if int(index) in kept and float(phase) != 0.0
    }


def moment_at_atom(
    structure: Structure,
    model: MagneticStructureModel,
    atom_index: int,
    *,
    cell_offset: Sequence[float] = (0.0, 0.0, 0.0),
    component_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Evaluate the real magnetic moment for one atom and cell offset."""

    offset = _as_float_array(cell_offset, "cell_offset")
    return moments_at_atoms(structure, model, [atom_index], [offset], component_ids=component_ids)[
        0
    ]


def moments_at_atoms(
    structure: Structure,
    model: MagneticStructureModel,
    atom_indices: Sequence[int],
    cell_offsets: Sequence[Sequence[float]],
    *,
    component_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Evaluate real magnetic moments, one row per (unit-cell atom index, cell offset) pair."""

    indices = np.asarray(atom_indices, dtype=int).reshape(-1)
    offsets = np.asarray(cell_offsets, dtype=float).reshape(-1, 3)
    if offsets.shape[0] != indices.shape[0]:
        raise MagneticStructureError("atom_indices and cell_offsets must have the same length")
    if np.any(indices < 0) or np.any(indices >= len(structure.species)):
        raise MagneticStructureError("atom_index is out of range")
    moments = np.zeros((indices.shape[0], 3), dtype=complex)
    for component in atom_resolved_fourier_components(
        structure, model, component_ids=component_ids
    ):
        fc = component.atom_fcs[indices]
        if np.allclose(fc, 0.0):
            continue
        phase = np.exp(-2j * np.pi * (offsets @ component.q_rlu))
        moments += fc * phase[:, np.newaxis]
    return moments.real


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
        [
            (i, j, k)
            for i in range(repeats[0])
            for j in range(repeats[1])
            for k in range(repeats[2])
        ],
        dtype=float,
    )
    atom_count = len(structure.species)
    positions_frac = (frac[np.newaxis, :, :] + offsets[:, np.newaxis, :]).reshape(-1, 3)
    atom_indices = np.tile(np.arange(atom_count, dtype=int), len(offsets))
    positions_cart = positions_frac @ cell
    moments = moments_at_atoms(
        structure,
        model,
        atom_indices,
        np.floor(positions_frac),
        component_ids=component_ids,
    )
    return positions_cart, moments, atom_indices


def magnetic_structure_to_ase_atoms(
    structure: Structure,
    model: MagneticStructureModel,
    supercell: Sequence[int] = (1, 1, 1),
    *,
    component_ids: Sequence[str] | None = None,
) -> Atoms:
    """Return the diagonal ``supercell`` of ``structure`` as ASE ``Atoms`` carrying the model's moments.

    ``initial_magnetic_moments`` is an ``(n, 3)`` array of Cartesian moments in
    Bohr magnetons from :func:`moments_in_supercell`, in the same atom order.
    Species labels are reduced to element symbols and kept in
    ``atoms.info["mufinder_species_labels"]``.
    """

    repeats = _as_supercell_repeats(supercell)
    unit_cell = structure_to_ase_atoms(
        structure, sanitize_symbols=True, record_species_labels=True, error=MagneticStructureError
    )
    positions, moments, atom_indices = moments_in_supercell(
        structure, model, repeats, component_ids=component_ids
    )
    symbols = unit_cell.get_chemical_symbols()
    labels = unit_cell.info["mufinder_species_labels"]
    atoms = Atoms(
        symbols=[symbols[index] for index in atom_indices],
        positions=positions,
        cell=np.asarray(repeats, dtype=float)[:, np.newaxis] * np.asarray(unit_cell.cell),
        pbc=True,
    )
    atoms.set_initial_magnetic_moments(moments)
    atoms.info["mufinder_species_labels"] = [labels[index] for index in atom_indices]
    return atoms


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
    for component in atom_resolved_fourier_components(
        structure, model, component_ids=component_ids
    ):
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
        raise MagneticStructureError(
            f"muesr Fourier coefficients must have shape (n_atoms, 3), got {arr.shape}"
        )
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
        raise MagneticStructureError(
            f"muesr Fourier coefficients must have shape (n_atoms, 3), got {arr.shape}"
        )
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
    kept = [
        component for component in components if float(np.linalg.norm(component)) > atol * scale
    ]
    if not kept:
        kept = [np.zeros_like(arr, dtype=complex)]

    reconstructed = np.sum(np.asarray(kept, dtype=complex), axis=0)
    if float(np.linalg.norm(reconstructed - arr)) > 100.0 * atol * scale:
        raise MagneticStructureError(
            "helical decomposition failed to reconstruct Fourier coefficients"
        )

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
    static = (
        moment * float(np.cos(angle)) * _unit_vector(static_axis_direction, "static_axis_direction")
    )
    rotating_real = (
        moment * float(np.sin(angle)) * _unit_vector(real_axis_direction, "real_axis_direction")
    )
    rotating_imag = (
        moment
        * float(np.sin(angle))
        * _unit_vector(imaginary_axis_direction, "imaginary_axis_direction")
    )
    coefficient = rotating_real.astype(complex) + 1j * rotating_imag
    return MagneticComponent(
        component_id=component_id or f"component-{uuid4()}",
        label=str(label).strip() or "Conical",
        kind="conical",
        selector=selector,
        terms=(
            FourierTerm(
                q_coeff=(0.0, 0.0, 0.0),
                coefficient=ComplexVector.from_vector(static.astype(complex)),
            ),
            FourierTerm(
                q_coeff=_as_float_tuple(q_coeff, "q_coeff"),
                coefficient=ComplexVector.from_vector(coefficient),
            ),
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
    transform = (
        np.eye(3, dtype=float)
        if q_coeff_transform is None
        else np.asarray(q_coeff_transform, dtype=float)
    )
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
        q_dirs = (
            q1,
            _rotate_vector(q1, core, 2.0 * np.pi / 3.0),
            _rotate_vector(q1, core, 4.0 * np.pi / 3.0),
        )
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
        raise MagneticStructureError(
            "q1 must lie in the plane perpendicular to the skyrmion core direction"
        )
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


def evaluate_texture_grid(
    texture: MagneticTexture, resolution: Sequence[int] = (32, 32, 1)
) -> np.ndarray:
    """Evaluate a texture on a regular reduced-coordinate grid."""

    res = _texture_grid_resolution(resolution)
    axes = [np.linspace(0.0, 1.0, value, endpoint=False) for value in res]
    uu, vv, ww = np.meshgrid(*axes, indexing="ij")
    dots = np.stack([uu, vv, ww], axis=-1)
    moments = np.zeros(res + (3,), dtype=complex)
    for term in texture.terms:
        q = np.asarray(term.q_coeff, dtype=float)
        q_dot = np.tensordot(dots, q, axes=([3], [0]))
        moments += (
            term.coefficient.to_array()[np.newaxis, np.newaxis, np.newaxis, :]
            * np.exp(-2j * np.pi * q_dot)[..., np.newaxis]
        )
    return moments.real


def texture_diagnostics(
    texture: MagneticTexture, resolution: Sequence[int] = (32, 32, 1)
) -> MagneticTextureDiagnostics:
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


@dataclass(frozen=True, slots=True)
class TopologicalDensity2D:
    """Topological-charge density sampled on a real-space slice of a texture.

    ``points`` holds the Cartesian sample points (Å), shape ``(nu, nv, 3)``;
    ``density`` holds the charge density of each grid cell (Å⁻²), shape
    ``(nu - 1, nv - 1)``; ``charge`` is the total signed charge on the slice.
    """

    points: np.ndarray
    density: np.ndarray
    charge: float


def topological_charge_magnetic_unit_cell(
    texture: MagneticTexture,
    q_basis: Sequence[Sequence[float]],
    resolution: Sequence[int] = (64, 64),
    *,
    max_resolution: int | None = None,
) -> int:
    """Return the topological charge of one magnetic unit cell of a 2-D texture.

    ``q_basis`` holds the Cartesian propagation-vector rows (Å⁻¹, in the
    ``exp(-2πi q·r)`` convention, without a 2π factor). Rows 1 and 2 span the
    texture plane; the magnetic cell is spanned by the in-plane real-space
    vectors ``a_j`` dual to them (``q_i · a_j = δ_ij``). The direction field
    ``n = m/|m|`` is sampled on an open grid over that cell and the signed solid
    angle of each grid triangle is summed with
    ``tan(Ω/2) = n₁·(n₂×n₃) / (1 + n₁·n₂ + n₂·n₃ + n₃·n₁)``, giving
    ``Q = ΣΩ / 4π``.

    The sign is taken looking down the plane normal oriented towards +z (towards
    +y, then +x, for planes that contain z): a skyrmion whose core points along
    −z in a +z background has ``Q = −1``.

    The grid is under-resolved when it has no more than twice as many points
    along an axis as the highest Fourier frequency along it (the samples would
    alias), or when two neighbouring samples are 90° or more apart. The grid is
    then doubled until it exceeds ``max_resolution`` (by
    default the grid is not refined) and an error is raised. A third q vector
    that lies in the plane as a combination of q1 and q2 is folded into them.
    An error is also raised for a singular q pair, a texture that uses a third q
    vector out of the plane or is not periodic on the cell, and a texture whose
    magnitude vanishes on the grid, where the charge is undefined.
    """

    if not texture.terms:
        raise MagneticStructureError("texture must contain at least one Fourier term")
    texture, normal_sign = _planar_texture(texture, q_basis)
    nu, nv = _plane_grid_resolution(resolution)
    if min(nu, nv) < 3:
        raise MagneticStructureError("topological charge needs a grid of at least 3 × 3")
    limit = max(nu, nv) if max_resolution is None else int(max_resolution)
    frequencies = np.abs(np.array([term.q_coeff[:2] for term in texture.terms], dtype=float))
    highest_u, highest_v = (float(value) for value in np.max(frequencies, axis=0))
    while True:
        # Below the Nyquist limit of every harmonic the samples cannot alias, so
        # the neighbour-angle test sees the texture itself.
        if nu > 2.0 * highest_u and nv > 2.0 * highest_v:
            moments = evaluate_texture_grid(texture, (nu, nv, 1))[:, :, 0, :]
            directions = _unit_texture_directions(moments)
            corners = (
                directions,
                np.roll(directions, -1, axis=0),
                np.roll(np.roll(directions, -1, axis=0), -1, axis=1),
                np.roll(directions, -1, axis=1),
            )
            if _directions_resolved(*corners):
                break
        if 2 * max(nu, nv) > limit:
            raise MagneticStructureError(
                f"the {nu} × {nv} grid is too coarse for this texture: it needs more than "
                f"{2.0 * highest_u:g} × {2.0 * highest_v:g} points to resolve every Fourier "
                "term, with neighbouring moments under 90° apart; increase the resolution"
            )
        nu, nv = 2 * nu, 2 * nv
    first, second, third, fourth = corners
    solid_angle = _triangle_solid_angles(first, second, third) + _triangle_solid_angles(
        first, third, fourth
    )
    charge = normal_sign * float(np.sum(solid_angle)) / (4.0 * np.pi)
    rounded = round(charge)
    if abs(charge - rounded) > 1e-6:
        raise MagneticStructureError(
            f"topological charge {charge:.6g} is not an integer; increase the resolution"
        )
    return int(rounded)


def topological_density_2d(
    texture: MagneticTexture,
    q_basis: Sequence[Sequence[float]],
    *,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    u_vector: Sequence[float],
    v_vector: Sequence[float],
    resolution: Sequence[int] = (64, 64),
) -> TopologicalDensity2D:
    """Return the topological-charge density on a real-space slice of a texture.

    The slice is the parallelogram ``origin + s·u_vector + t·v_vector`` with
    ``s, t`` in ``[0, 1]`` (Cartesian Å), sampled at ``resolution`` points per
    side including both edges. ``q_basis`` holds the Cartesian propagation-vector
    rows (Å⁻¹, as for :func:`topological_charge_magnetic_unit_cell`), so the
    slice may cut a 3-D texture. Each grid cell's density is its solid angle,
    summed over its two triangles, divided by ``4π`` times the cell area, with
    the same sign convention as :func:`topological_charge_magnetic_unit_cell`.
    Raises when the slice vectors are parallel, the magnitude vanishes, the grid
    has no more than two intervals per Fourier cycle along a side (the samples
    would alias), or neighbouring samples are 90° or more apart.
    """

    basis = np.zeros((3, 3), dtype=float)
    rows = np.asarray(q_basis, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 3 or not 1 <= rows.shape[0] <= 3:
        raise MagneticStructureError("q_basis must have between one and three rows of 3 values")
    basis[: rows.shape[0]] = rows
    u = _as_float_array(u_vector, "u_vector")
    v = _as_float_array(v_vector, "v_vector")
    start = _as_float_array(origin, "origin")
    normal = np.cross(u, v)
    area = float(np.linalg.norm(normal))
    if area <= 1e-12 * max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-300):
        raise MagneticStructureError("slice vectors must be non-zero and not parallel")
    nu, nv = _plane_grid_resolution(resolution)
    if min(nu, nv) < 2:
        raise MagneticStructureError("topological density needs at least 2 points per side")
    s_axis = np.linspace(0.0, 1.0, nu)
    t_axis = np.linspace(0.0, 1.0, nv)
    points = start + s_axis[:, np.newaxis, np.newaxis] * u + t_axis[np.newaxis, :, np.newaxis] * v
    q_cartesian = np.array([term.q_coeff for term in texture.terms], dtype=float) @ basis
    # Fourier cycles of each term along each side of the slice; fewer than two
    # intervals per cycle would alias.
    highest_u = float(np.max(np.abs(q_cartesian @ u))) if texture.terms else 0.0
    highest_v = float(np.max(np.abs(q_cartesian @ v))) if texture.terms else 0.0
    if nu - 1 <= 2.0 * highest_u or nv - 1 <= 2.0 * highest_v:
        raise MagneticStructureError(
            f"the {nu} × {nv} slice grid is too coarse for this texture: it needs more than "
            f"{2.0 * highest_u:.4g} × {2.0 * highest_v:.4g} intervals to resolve every "
            "Fourier term; increase the resolution"
        )
    phases = points @ basis.T
    moments = np.zeros(points.shape, dtype=complex)
    for term in texture.terms:
        q_dot = phases @ np.asarray(term.q_coeff, dtype=float)
        moments += term.coefficient.to_array() * np.exp(-2j * np.pi * q_dot)[..., np.newaxis]
    directions = _unit_texture_directions(moments.real)
    first = directions[:-1, :-1]
    second = directions[1:, :-1]
    third = directions[1:, 1:]
    fourth = directions[:-1, 1:]
    if not _directions_resolved(first, second, third, fourth):
        raise MagneticStructureError(
            f"the {nu} × {nv} slice grid is too coarse for this texture: neighbouring "
            "moments are 90° or more apart; increase the resolution"
        )
    solid_angle = _triangle_solid_angles(first, second, third) + _triangle_solid_angles(
        first, third, fourth
    )
    sign = _plane_normal_sign(normal / area)
    cell_area = area / ((nu - 1) * (nv - 1))
    density = sign * solid_angle / (4.0 * np.pi * cell_area)
    return TopologicalDensity2D(
        points=points,
        density=density,
        charge=float(sign * np.sum(solid_angle) / (4.0 * np.pi)),
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
            out.append(
                MagneticTextureTerm.from_vector(base_q + harmonic, coefficient * f_coeffs[idx])
            )
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
        terms = [
            term
            for term in terms
            if float(np.linalg.norm(term.coefficient.to_array())) >= float(threshold)
        ]
    if top_n is not None and int(top_n) >= 0 and len(terms) > int(top_n):
        terms = sorted(
            terms, key=lambda term: float(np.linalg.norm(term.coefficient.to_array())), reverse=True
        )[: int(top_n)]
    return MagneticTexture(tuple(terms)).combine_terms()


def magnetic_model_from_host_spins(structure: Structure) -> MagneticStructureModel:
    """Build a magnetic model from the host's scalar-spin or MCIF metadata."""

    q_basis = _legacy_q_basis_rlu(structure)
    propagation = q_basis[0]
    q_coeff = (
        (1.0, 0.0, 0.0) if any(abs(value) > 1e-12 for value in propagation) else (0.0, 0.0, 0.0)
    )
    moment_entries = _legacy_moment_entries(structure)
    components: list[MagneticComponent] = []
    for entry in moment_entries:
        index = int(entry["atom_index"])
        vector = entry["vector"]
        if not any(abs(value) > 1e-12 for value in vector):
            continue
        label = str(entry.get("label") or f"{structure.species[index]}{index + 1}")
        coefficient = _legacy_fourier_coefficient(
            structure, atom_index=index, moment=vector, q_rlu=propagation
        )
        metadata: dict[str, object] = {}
        if entry.get("source"):
            metadata["source"] = entry["source"]
        if entry.get("mcif_label"):
            metadata["mcif_label"] = entry["mcif_label"]
        if entry.get("source_coordinate_system"):
            metadata["source_coordinate_system"] = entry["source_coordinate_system"]
        components.append(
            MagneticComponent(
                component_id=f"legacy_atom_{index + 1}",
                label=label,
                kind="legacy_moment",
                selector=AtomSelector.indices([index]),
                terms=(
                    FourierTerm(
                        q_coeff=q_coeff, coefficient=ComplexVector.from_vector(coefficient)
                    ),
                ),
                metadata=metadata,
            )
        )
    return MagneticStructureModel(
        q_basis_rlu=q_basis,
        components=tuple(components),
        metadata={"source": "legacy_host_magnetic_structure"},
    )


# Public import alias; see docs/compatibility_policy.rst.
magnetic_model_from_legacy_structure = magnetic_model_from_host_spins


def magnetic_model_for_host(stored: object, host: Structure | None) -> MagneticStructureModel:
    """Return the magnetic model saved for ``host``, or one built from its spins.

    ``stored`` is the project's (or a workspace's) saved model payload. When it
    is not a mapping, the host's own ``magnetic_structure`` property is used. A
    payload with ``components`` is read as a model; anything else falls back to
    :func:`magnetic_model_from_host_spins`. With no host the model is
    empty.
    """

    if host is None:
        return MagneticStructureModel()
    payload = stored if isinstance(stored, dict) else host.properties.get("magnetic_structure")
    if isinstance(payload, dict) and "components" in payload:
        return MagneticStructureModel.from_json(payload)
    return magnetic_model_from_host_spins(host)


# --- Builder geometry: q-basis rows, propagation and moment directions ---------

# Below this length a direction or a projection counts as zero.
_DIRECTION_TOLERANCE = 1e-6


def unit_q_coeff(index: int) -> tuple[float, float, float]:
    """Return the q-basis coefficients that select basis row ``index`` alone."""

    rows = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    return rows[int(index)]


def _float_triple(values: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def component_propagation_rlu(
    model: MagneticStructureModel, component: MagneticComponent
) -> tuple[float, float, float]:
    """Return a component's q in r.l.u.: its first finite-q term's, else its first term's, else zero."""

    for term in component.terms:
        if any(abs(float(value)) > 1e-12 for value in term.q_coeff):
            return _float_triple(model.q_coeff_to_rlu(term.q_coeff))
    if component.terms:
        return _float_triple(model.q_coeff_to_rlu(component.terms[0].q_coeff))
    return (0.0, 0.0, 0.0)


def component_propagation_coeff(
    component: MagneticComponent,
) -> tuple[float, float, float] | None:
    """Return the q-basis coefficients of a component's first finite-q term, or ``None`` when it has none."""

    for term in component.terms:
        if any(abs(float(value)) > 1e-12 for value in term.q_coeff):
            return _float_triple(term.q_coeff)
    return None


def used_q_basis_indices(
    model: MagneticStructureModel, *, ignored_component_ids: set[str] | None = None
) -> set[int]:
    """Return the q-basis rows the model's components use, leaving out ``ignored_component_ids``."""

    used: set[int] = set()
    for component in model.components:
        if ignored_component_ids is not None and component.component_id in ignored_component_ids:
            continue
        used.update(component_q_basis_indices(component))
    return used


def allocate_q_basis_row(
    model: MagneticStructureModel,
    q_vector: Sequence[float],
    *,
    ignored_component_ids: set[str] | None = None,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float] | None]:
    """Find or free a q-basis row for ``q_vector``.

    Returns the q basis, with ``q_vector`` written into a free row when no row
    already holds it, and the coefficients that select that row. A used row
    that matches comes first, then a free row that matches, then the first
    free row. The coefficients are ``None`` when every row is used by another
    q; a zero ``q_vector`` needs no row and gets zero coefficients.
    """

    q = np.asarray(q_vector, dtype=float)
    basis = list(model.q_basis_rlu)
    if np.allclose(q, 0.0, atol=1e-12):
        return tuple(basis), (0.0, 0.0, 0.0)
    used_indices = used_q_basis_indices(model, ignored_component_ids=ignored_component_ids)
    for index, row in enumerate(basis):
        if index in used_indices and np.allclose(row, q, atol=1e-9):
            return tuple(basis), unit_q_coeff(index)
    for index, row in enumerate(basis):
        if index not in used_indices and np.allclose(row, q, atol=1e-9):
            return tuple(basis), unit_q_coeff(index)
    for index in range(len(basis)):
        if index not in used_indices:
            basis[index] = tuple(float(value) for value in q)
            return tuple(basis), unit_q_coeff(index)
    return tuple(basis), None


def allocate_texture_q_basis_rows(
    model: MagneticStructureModel,
    q_vectors: Sequence[Sequence[float]],
    *,
    ignored_component_ids: set[str] | None = None,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...] | None]:
    """Find or free a q-basis row for each of a texture's (at most three) q vectors.

    Returns the q basis and the three rows of coefficients that map the
    texture's own q basis onto it; a zero vector maps to zero. The rows are
    ``None`` when a vector finds neither a matching row nor a free one.
    """

    basis = [tuple(float(value) for value in row) for row in model.q_basis_rlu]
    used_indices = used_q_basis_indices(model, ignored_component_ids=ignored_component_ids)
    transform_rows: list[tuple[float, float, float]] = []
    allocated: set[int] = set()
    for q_vector in q_vectors[:3]:
        q = np.asarray(q_vector, dtype=float)
        if np.allclose(q, 0.0, atol=1e-12):
            transform_rows.append((0.0, 0.0, 0.0))
            continue
        matched = None
        for index, row in enumerate(basis):
            if np.allclose(row, q, atol=1e-9, rtol=0.0):
                matched = index
                break
        if matched is None:
            for index in range(3):
                if index not in used_indices and index not in allocated:
                    matched = index
                    basis[index] = tuple(float(value) for value in q)
                    break
        if matched is None:
            return tuple(basis), None
        allocated.add(matched)
        transform_rows.append(unit_q_coeff(matched))
    while len(transform_rows) < 3:
        transform_rows.append((0.0, 0.0, 0.0))
    return tuple(basis), tuple(transform_rows)


def unit_direction_or(
    vector: Sequence[float], fallback: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Return ``vector`` normalised, or ``fallback`` when it is (almost) zero."""

    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= _DIRECTION_TOLERANCE:
        return fallback
    return tuple(float(value) for value in arr / norm)  # type: ignore[return-value]


def perpendicular_unit_direction(vector: Sequence[float]) -> np.ndarray:
    """Return a unit vector perpendicular to ``vector``, built from x (or y when x is nearly parallel).

    A zero ``vector`` is taken as z.
    """

    axis = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= _DIRECTION_TOLERANCE:
        axis = np.asarray((0.0, 0.0, 1.0), dtype=float)
    else:
        axis = axis / norm
    trial = np.asarray((1.0, 0.0, 0.0), dtype=float)
    if abs(float(axis @ trial)) > 0.9:
        trial = np.asarray((0.0, 1.0, 0.0), dtype=float)
    projected = trial - float(trial @ axis) * axis
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm <= _DIRECTION_TOLERANCE:
        return np.asarray((0.0, 0.0, 1.0), dtype=float)
    return projected / projected_norm


def nonzero_unit_direction(vector: Sequence[float], label: str) -> np.ndarray:
    """Return ``vector`` normalised, raising ``ValueError`` naming ``label`` when it is (almost) zero."""

    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= _DIRECTION_TOLERANCE:
        raise ValueError(f"{label} must be a non-zero direction")
    return arr / norm


def propagation_direction(
    lattice_vectors: Sequence[Sequence[float]],
    q_rlu: Sequence[float],
    *,
    allow_fallback: bool = False,
) -> np.ndarray:
    """Return the Cartesian unit vector along a helix, cycloid or conical component's q.

    A zero q, or one with no Cartesian direction, raises ``ValueError``; with
    ``allow_fallback`` it gives z instead.
    """

    q = np.asarray(q_rlu, dtype=float)
    if float(np.linalg.norm(q)) <= _DIRECTION_TOLERANCE:
        if allow_fallback:
            return np.asarray((0.0, 0.0, 1.0), dtype=float)
        raise ValueError("q must be non-zero for helical, cycloidal, and conical components")
    try:
        return q_cartesian_direction(lattice_vectors, q)
    except MagneticStructureError:
        if allow_fallback:
            return np.asarray((0.0, 0.0, 1.0), dtype=float)
        raise ValueError("q must define a non-zero propagation direction") from None


def rotation_handedness(
    primary: Sequence[float], secondary: Sequence[float], q_direction: Sequence[float]
) -> int:
    """Return the sense of rotation about q: +1 when the triple product of q, primary and secondary is not negative, else -1."""

    primary_arr = np.asarray(primary, dtype=float)
    secondary_arr = np.asarray(secondary, dtype=float)
    q_arr = np.asarray(q_direction, dtype=float)
    return 1 if float(np.dot(np.cross(q_arr, primary_arr), secondary_arr)) >= 0.0 else -1


def reference_direction_perpendicular_to(
    reference: Sequence[float], q_direction: np.ndarray
) -> np.ndarray:
    """Return the unit part of ``reference`` perpendicular to the unit ``q_direction``.

    When ``reference`` is (almost) parallel to q, any unit vector perpendicular
    to q is returned instead (:func:`perpendicular_unit_direction`).
    """

    raw = np.asarray(reference, dtype=float)
    projected = raw - float(raw @ q_direction) * q_direction
    norm = float(np.linalg.norm(projected))
    if norm <= _DIRECTION_TOLERANCE:
        return perpendicular_unit_direction(q_direction)
    return projected / norm


def constrained_secondary_direction(
    kind: str, q_direction: np.ndarray, primary: np.ndarray, handedness: float
) -> np.ndarray:
    """Return the second moment direction of a helix, conical or cycloid component.

    A cycloid rotates in the plane holding q, so its second direction is
    ``handedness * q``. A helix or conical component rotates about q, so it is
    ``handedness * q x primary``, normalised.
    """

    if kind == "cycloid":
        return handedness * q_direction
    secondary = handedness * np.cross(q_direction, primary)
    secondary_norm = float(np.linalg.norm(secondary))
    if secondary_norm <= _DIRECTION_TOLERANCE:
        return perpendicular_unit_direction(primary)
    return secondary / secondary_norm


def conical_component_geometry(
    component: MagneticComponent, q_direction: np.ndarray | None
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Read a conical component's cone back from its two terms.

    Returns the moment amplitude, the cone angle in degrees, the unit real and
    imaginary directions of the rotating term and the cone axis. The axis is
    ``q_direction``, or the real direction crossed with the imaginary one when
    q is not known. The component must have at least two terms: the static
    term first, then the rotating one.
    """

    static = np.asarray(component.terms[0].coefficient.real, dtype=float)
    rotating = component.terms[1].coefficient
    real = np.asarray(rotating.real, dtype=float)
    imag = np.asarray(rotating.imaginary, dtype=float)
    real_direction = np.asarray(unit_direction_or(real, (0.0, 0.0, 1.0)), dtype=float)
    imag_direction = np.asarray(unit_direction_or(imag, (1.0, 0.0, 0.0)), dtype=float)
    axis = np.cross(real_direction, imag_direction) if q_direction is None else q_direction
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= _DIRECTION_TOLERANCE:
        axis = np.asarray((0.0, 0.0, 1.0), dtype=float)
    else:
        axis = axis / axis_norm
    signed_static = float(static @ axis)
    rotating_norm = 0.5 * (float(np.linalg.norm(real)) + float(np.linalg.norm(imag)))
    amplitude = float(np.hypot(signed_static, rotating_norm))
    cone_angle = float(np.degrees(np.arctan2(rotating_norm, signed_static)))
    return amplitude, cone_angle, real_direction, imag_direction, axis


# --- Previews ------------------------------------------------------------------


def tiled_supercell_structure(
    structure: Structure,
    repeats: tuple[int, int, int],
) -> tuple[Structure, np.ndarray]:
    """Tile ``structure`` ``repeats`` times along a, b and c for a moment preview.

    The atoms come in the order :func:`moments_in_supercell` gives its moments:
    cell by cell (c fastest), then atom by atom, at unwrapped positions. Returns
    the tiled structure and, for each of its atoms, the index of the unit-cell
    atom it copies.
    """

    repeat_array = np.asarray(repeats, dtype=float)
    base_frac = np.asarray(structure.fractional_positions, dtype=float)
    offsets = np.array(
        [
            (i, j, k)
            for i in range(repeats[0])
            for j in range(repeats[1])
            for k in range(repeats[2])
        ],
        dtype=float,
    )
    atom_count = len(structure.species)
    positions_frac = (
        (base_frac[np.newaxis, :, :] + offsets[:, np.newaxis, :]) / repeat_array
    ).reshape(-1, 3)
    atom_indices = np.tile(np.arange(atom_count, dtype=int), len(offsets))
    lattice = np.asarray(structure.lattice_vectors, dtype=float).copy()
    lattice[0] *= repeats[0]
    lattice[1] *= repeats[1]
    lattice[2] *= repeats[2]
    preview = Structure(
        label=f"{structure.label or 'structure'} preview {repeats[0]}x{repeats[1]}x{repeats[2]}",
        origin=structure.origin,
        lattice_vectors=lattice.tolist(),
        species=[structure.species[int(index)] for index in atom_indices],
        fractional_positions=positions_frac.tolist(),
        source_file=structure.source_file,
        parent_structure_id=structure.id,
        properties={
            **structure.properties,
            "pbc": structure.properties.get("pbc", [True, True, True]),
        },
    )
    return preview, atom_indices


def evaluate_texture_preview_grid(
    texture: MagneticTexture,
    *,
    mode: str = "custom",
    core_direction: Sequence[float] = (0.0, 0.0, -1.0),
    resolution: int = 73,
    repeats: float = 3.0,
) -> np.ndarray:
    """Evaluate a texture on a square window ``repeats`` periods wide, for a 2D preview.

    The texture's q-basis rows are drawn as 2D vectors: at 120 degrees for a
    hexagonal skyrmion, at right angles otherwise, with the second row flipped
    for a skyrmion whose core points along -z, so the drawn sense of rotation
    matches the moments. Returns the real moments, shape
    ``(resolution, resolution, 3)``.
    """

    axis = np.linspace(-0.5 * repeats, 0.5 * repeats, int(resolution), endpoint=True)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    core = np.asarray(core_direction, dtype=float)
    handed_plot_sign = -1.0 if core.size >= 3 and float(core[2]) < 0.0 else 1.0
    if mode == "hexagonal_skyrmion":
        preview_q_basis = np.array(
            (
                (1.0, 0.0),
                (-0.5, handed_plot_sign * np.sqrt(3.0) / 2.0),
                (0.0, 0.0),
            ),
            dtype=float,
        )
    elif mode == "square_skyrmion":
        preview_q_basis = np.array(
            (
                (1.0, 0.0),
                (0.0, handed_plot_sign),
                (0.0, 0.0),
            ),
            dtype=float,
        )
    else:
        preview_q_basis = np.array(
            (
                (1.0, 0.0),
                (0.0, 1.0),
                (0.0, 0.0),
            ),
            dtype=float,
        )
    moments = np.zeros((axis.size, axis.size, 3), dtype=complex)
    for term in texture.terms:
        q = np.asarray(term.q_coeff, dtype=float)
        q_xy = q[:3] @ preview_q_basis
        q_dot = q_xy[0] * xx + q_xy[1] * yy
        moments += (
            term.coefficient.to_array()[np.newaxis, np.newaxis, :]
            * np.exp(-2j * np.pi * q_dot)[..., np.newaxis]
        )
    return moments.real


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
            raise MagneticStructureError(
                "skyrmion q directions must be perpendicular to the core direction"
            )
        if kind == "bloch":
            e1 = core_direction
            e2 = helicity * _unit_vector(np.cross(core_direction, q_dir), "bloch_direction")
        else:
            e1 = core_direction
            e2 = helicity * q_dir
        terms.append(MagneticTextureTerm.from_vector(q_coeff, rot_mag * (e1 + 1j * e2)))
    static_mag = float(magnitude) * float(np.cos(angle)) + float(ferromagnetic_background)
    if abs(static_mag) > 1e-12:
        terms.append(
            MagneticTextureTerm.from_vector(
                (0.0, 0.0, 0.0), static_mag * core_direction.astype(complex)
            )
        )
    return MagneticTexture(tuple(terms)).combine_terms()


def _rotate_vector(vector: Sequence[float], axis: Sequence[float], angle_rad: float) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    a = _unit_vector(axis, "rotation_axis")
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    return v * cos_a + np.cross(a, v) * sin_a + a * float(np.dot(a, v)) * (1.0 - cos_a)


def _plane_grid_resolution(resolution: Sequence[int]) -> tuple[int, int]:
    values = [int(value) for value in resolution]
    if len(values) != 2:
        raise MagneticStructureError("resolution must have two values")
    return values[0], values[1]


def _plane_normal_sign(normal: np.ndarray) -> float:
    """Return +1 when ``normal`` points towards +z (or +y, then +x, in the plane z = 0)."""

    for axis in (2, 1, 0):
        if abs(float(normal[axis])) > 1e-9:
            return 1.0 if float(normal[axis]) > 0.0 else -1.0
    return 1.0


def _planar_texture(
    texture: MagneticTexture, q_basis: Sequence[Sequence[float]]
) -> tuple[MagneticTexture, float]:
    """Return a 2-D texture with q coefficients on q1 and q2 only, and the sign of ``q1 × q2``.

    A term on an in-plane q3 is rewritten on q1 and q2; a term on an out-of-plane
    q3, a singular q pair, and a non-integer coefficient (a texture that is not
    periodic on the magnetic cell) raise.
    """

    rows = np.asarray(q_basis, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 3 or not 2 <= rows.shape[0] <= 3:
        raise MagneticStructureError("q_basis must have two or three rows of 3 values")
    q1, q2 = rows[0], rows[1]
    normal = np.cross(q1, q2)
    scale = float(np.linalg.norm(q1) * np.linalg.norm(q2))
    if scale <= 0.0 or float(np.linalg.norm(normal)) <= 1e-9 * scale:
        raise MagneticStructureError(
            "the in-plane q vectors q1 and q2 must be non-zero and not parallel"
        )
    q3 = rows[2] if rows.shape[0] == 3 else np.zeros(3)
    q3_norm = float(np.linalg.norm(q3))
    q3_in_plane: np.ndarray | None = None
    if q3_norm > 1e-12:
        in_plane, *_ = np.linalg.lstsq(rows[:2].T, q3, rcond=None)
        if float(np.linalg.norm(rows[:2].T @ in_plane - q3)) <= 1e-9 * q3_norm:
            q3_in_plane = in_plane
    terms: list[MagneticTextureTerm] = []
    for term in texture.terms:
        q_coeff = np.asarray(term.q_coeff, dtype=float)
        planar = q_coeff[:2].copy()
        if q3_norm > 1e-12 and abs(float(q_coeff[2])) > 1e-9:
            if q3_in_plane is None:
                raise MagneticStructureError(
                    "the texture uses a third q vector out of the q1-q2 plane, "
                    "so it is not two-dimensional"
                )
            planar += float(q_coeff[2]) * q3_in_plane
        if not np.allclose(planar, np.round(planar), atol=1e-9, rtol=0.0):
            raise MagneticStructureError(
                "the texture is not periodic on the magnetic cell spanned by q1 and q2"
            )
        terms.append(
            MagneticTextureTerm(
                q_coeff=(float(round(planar[0])), float(round(planar[1])), 0.0),
                coefficient=term.coefficient,
            )
        )
    sign = _plane_normal_sign(normal / float(np.linalg.norm(normal)))
    return MagneticTexture(tuple(terms)), sign


def _unit_texture_directions(moments: np.ndarray) -> np.ndarray:
    magnitudes = np.linalg.norm(moments, axis=-1)
    largest = float(np.max(magnitudes)) if magnitudes.size else 0.0
    if largest <= 0.0 or float(np.min(magnitudes)) <= 1e-6 * largest:
        raise MagneticStructureError(
            "the moment magnitude vanishes on the sampling grid, so the direction m/|m| "
            "is undefined there"
        )
    return moments / magnitudes[..., np.newaxis]


def _directions_resolved(
    first: np.ndarray, second: np.ndarray, third: np.ndarray, fourth: np.ndarray
) -> bool:
    """Return whether every edge and diagonal of each grid cell is under 90°."""

    pairs = ((first, second), (second, third), (third, fourth), (fourth, first), (first, third))
    return all(float(np.min(np.sum(a * b, axis=-1))) > 0.0 for a, b in pairs)


def _triangle_solid_angles(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
    numerator = np.sum(first * np.cross(second, third), axis=-1)
    denominator = (
        1.0
        + np.sum(first * second, axis=-1)
        + np.sum(second * third, axis=-1)
        + np.sum(third * first, axis=-1)
    )
    return 2.0 * np.arctan2(numerator, denominator)


def _texture_grid_resolution(resolution: Sequence[int]) -> tuple[int, int, int]:
    raw = list(resolution) + [1, 1, 1]
    res = tuple(max(1, int(value)) for value in raw[:3])
    return res


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
        vectors.append(propagation_vector_from_structure(structure, strict=True))
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
                atom_index = _legacy_moment_atom_index(
                    moment, fallback=index, atom_count=atom_count
                )
                if atom_index is None:
                    continue
                try:
                    vector = _as_float_tuple(moment.get("vector", (0.0, 0.0, 0.0)), "moment vector")
                except MagneticStructureError:
                    continue
                if moment.get("coordinate_system") == "crystalaxis":
                    # Stored as written in the file (components along the unit lattice
                    # directions) by imports that did not convert them.
                    try:
                        vector = tuple(
                            crystal_axis_to_cartesian(
                                vector, structure.lattice_vectors, error=MagneticStructureError
                            )
                        )
                    except MagneticStructureError:
                        continue
                source_system = moment.get("source_coordinate_system") or moment.get(
                    "coordinate_system"
                )
                label = str(
                    moment.get("label")
                    or moment.get("atom_label")
                    or f"{structure.species[atom_index]}{atom_index + 1}"
                )
                entries.append(
                    {
                        "atom_index": atom_index,
                        "label": label,
                        "vector": vector,
                        "source": "mcif",
                        "mcif_label": str(moment.get("label", label)),
                        "source_coordinate_system": source_system,
                    }
                )
    if any(any(abs(value) > 1e-12 for value in entry["vector"]) for entry in entries):
        return entries

    raw_spins = structure.properties.get(
        "atomic_spins", structure.properties.get("initial_magnetic_moments")
    )
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


def _legacy_moment_atom_index(
    moment: Mapping[str, object], *, fallback: int, atom_count: int
) -> int | None:
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
    cell = spglib_cell(structure)
    if cell is None:
        return None
    dataset = get_crystal_symmetry_dataset(cell)
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


def _is_positive_multiple(vector: np.ndarray, reference: np.ndarray) -> bool:
    """Return whether non-zero ``vector`` points the same way as non-zero ``reference``."""

    norm = float(np.linalg.norm(vector))
    reference_norm = float(np.linalg.norm(reference))
    if norm <= _Q_COEFF_TOLERANCE or reference_norm <= _Q_COEFF_TOLERANCE:
        return False
    return bool(np.allclose(vector / norm, reference / reference_norm, atol=1e-9, rtol=0.0))


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
