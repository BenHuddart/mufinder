"""Cheap change-detection fingerprints for structure-driven view updates.

Flush-path refreshes (see :meth:`MainWindow._flush_dashboard`) re-push the same
host :class:`~mufinder.domain.models.Structure` into the 3D atomic viewer on
every tick, forcing a full mesh rebuild even when nothing changed. These helpers
produce a cheap fingerprint so a widget entry point can skip the expensive
re-render when its input is unchanged.

**Contract: content-based fingerprint.** The fingerprint carries the *content*
of the rendered geometry fields (``species``, ``fractional_positions``,
``lattice_vectors``) as normalized tuples together with ``Structure.id``. Two
calls return equal fingerprints iff the id and the geometry values are equal,
regardless of which list objects hold them — an exact equality guarantee, not a
probabilistic one (storing ``hash(...)`` digests instead would admit a hash
collision falsely skipping a render). This makes the guard safe against every
mutation pattern:

- reassignment of the geometry lists (the pattern used by
  ``ProjectService._repair_supercell_asset`` / ``regenerate_supercell_asset``,
  which swap geometry while keeping the same ``Structure.id``),
- in-place element mutation of those lists, and
- CPython ``id()`` recycling — an identity-based fingerprint could falsely
  match when a freed list's address is reused by a later replacement list
  (list_A -> list_B -> list_C between two renders can leave
  ``id(list_C) == id(list_A)``), so identities must not be used here.

Building tuples of a few thousand floats/strings costs tens of microseconds and
a few kilobytes per cached fingerprint — negligible next to the multi-second
viewer rebuild the guard exists to skip.

``properties`` is left out of the geometry fingerprint because pages show
different slices of it. What the 3D viewer draws from them (spins, atom colours
and periodicity, :data:`VIEWER_PROPERTY_KEYS`) is snapshotted by content in
:func:`structure_properties_snapshot`; ``properties`` values are mutated in
place elsewhere (e.g. ``ProjectService.set_host_magnetic_configuration``), so
reference- or identity-based snapshots would miss changes.

A page that feeds a viewer uses :class:`ViewerStructureCheck`, which says
whether the geometry, the spins alone or other viewer properties changed since
the viewer was last given a structure, or :func:`viewer_structure_fingerprint`
when it folds its own page-specific snapshot in on top.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from mufinder.domain.models import Structure

# Sentinel distinguishing "no structure" from any real fingerprint tuple.
NO_STRUCTURE: tuple[object, ...] = ()


def structure_geometry_fingerprint(structure: Structure | None) -> tuple[object, ...]:
    """Return a content-based fingerprint of a structure's rendered geometry.

    The fingerprint combines ``structure.id`` with normalized tuples of the
    ``species``, ``fractional_positions`` and ``lattice_vectors`` contents, so
    it compares unequal whenever the rendered geometry values change — however
    that change was made (list reassignment or in-place element mutation), and
    with exact equality semantics (no hash-collision false matches). ``None``
    maps to :data:`NO_STRUCTURE`, which never compares equal to a real
    structure's fingerprint (a real structure always contributes its ``id`` as
    the first element).
    """

    if structure is None:
        return NO_STRUCTURE
    return (
        structure.id,
        tuple(structure.species),
        tuple(float(value) for position in structure.fractional_positions for value in position),
        tuple(float(value) for vector in structure.lattice_vectors for value in vector),
    )


#: The ``Structure.properties`` keys the 3D atomic viewer draws: spin arrows, atom
#: colour overrides and the periodic boundary flags.
VIEWER_PROPERTY_KEYS: tuple[str, ...] = ("atomic_spins", "atom_colors", "pbc")


def _frozen(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _frozen(item)) for key, item in value.items()))
    if hasattr(value, "tolist"):
        # A NumPy array or scalar: compared element by element, it has no truth value.
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def structure_properties_snapshot(
    structure: Structure | None, keys: Iterable[str] = VIEWER_PROPERTY_KEYS
) -> tuple[object, ...]:
    """Return a content-based snapshot of ``keys`` in ``structure.properties``.

    Nested lists and mappings become tuples, so the snapshot compares equal
    exactly when the values do, however they were changed (in place or
    reassigned). ``None`` maps to :data:`NO_STRUCTURE`.
    """

    if structure is None:
        return NO_STRUCTURE
    return tuple((key, _frozen(structure.properties.get(key))) for key in keys)


def viewer_structure_fingerprint(structure: Structure | None) -> tuple[object, ...]:
    """Return a content-based fingerprint of everything the 3D viewer draws from ``structure``.

    The geometry (:func:`structure_geometry_fingerprint`) and the viewer's
    properties (:data:`VIEWER_PROPERTY_KEYS`). A page that shows more than the
    viewer folds its own snapshot in on top. ``None`` maps to :data:`NO_STRUCTURE`.
    """

    if structure is None:
        return NO_STRUCTURE
    return (structure_geometry_fingerprint(structure), structure_properties_snapshot(structure))


@dataclass(frozen=True)
class ViewerStructureChange:
    """What changed in a viewer's structure since a :class:`ViewerStructureCheck` last saw it."""

    #: The geometry changed, or nothing was given before: the viewer reloads.
    geometry: bool
    #: The :data:`VIEWER_PROPERTY_KEYS` whose values changed, on the same geometry.
    properties: frozenset[str] = frozenset()
    #: The structure last given had a non-zero spin, so the user has seen spins.
    had_spins: bool = False

    @property
    def unchanged(self) -> bool:
        return not self.geometry and not self.properties

    @property
    def spins_only(self) -> bool:
        """Only the spins changed, which the viewer redraws in place."""

        return not self.geometry and self.properties == frozenset({"atomic_spins"})


class ViewerStructureCheck:
    """Tell a page what changed in the structure it last gave its 3D viewer.

    A page asks :meth:`change` before giving its viewer a structure and calls
    :meth:`remember` once it has, so a refresh that brings the same content
    leaves the viewer, its camera, selection and overlays alone. Content is
    compared, not identity: structures are edited in place.
    """

    def __init__(self) -> None:
        self._geometry: tuple[object, ...] | None = None
        self._properties: tuple[object, ...] = NO_STRUCTURE
        # The snapshot the last ``change`` took, which ``remember`` reuses.
        self._seen: tuple[Structure | None, tuple[object, ...], tuple[object, ...]] | None = None

    def change(self, structure: Structure | None) -> ViewerStructureChange:
        geometry = structure_geometry_fingerprint(structure)
        properties = structure_properties_snapshot(structure)
        self._seen = (structure, geometry, properties)
        had_spins = _any_nonzero(dict(self._properties).get("atomic_spins"))
        if self._geometry is None or geometry != self._geometry:
            return ViewerStructureChange(geometry=True, had_spins=had_spins)
        changed = frozenset(
            str(key)
            for (key, value), (_key, previous) in zip(properties, self._properties, strict=True)
            if value != previous
        )
        return ViewerStructureChange(geometry=False, properties=changed, had_spins=had_spins)

    def remember(self, structure: Structure | None) -> None:
        seen = self._seen
        self._seen = None
        if seen is not None and seen[0] is structure:
            self._geometry, self._properties = seen[1], seen[2]
            return
        self._geometry = structure_geometry_fingerprint(structure)
        self._properties = structure_properties_snapshot(structure)

    def forget(self) -> None:
        """Treat the next structure as new, so the viewer reloads it."""

        self._geometry = None
        self._properties = NO_STRUCTURE
        self._seen = None


def _any_nonzero(values: object) -> bool:
    """Whether a frozen spin list holds a number other than zero."""

    if not isinstance(values, tuple):
        return False
    for value in values:
        try:
            if abs(float(value)) > 1e-10:  # type: ignore[arg-type]
                return True
        except (TypeError, ValueError):
            continue
    return False


__all__ = [
    "NO_STRUCTURE",
    "VIEWER_PROPERTY_KEYS",
    "ViewerStructureChange",
    "ViewerStructureCheck",
    "structure_geometry_fingerprint",
    "structure_properties_snapshot",
    "viewer_structure_fingerprint",
]
