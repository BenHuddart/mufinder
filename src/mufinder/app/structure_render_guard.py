"""Cheap change-detection fingerprints for structure-driven view updates.

Flush-path refreshes (see :meth:`MainWindow._flush_dashboard`) re-push the same
host :class:`~mufinder.domain.models.Structure` into the 3D atomic viewer on
every tick, forcing a full mesh rebuild even when nothing changed. These helpers
produce a cheap fingerprint so a widget entry point can skip the expensive
re-render when its input is unchanged.

**Contract: content-based fingerprint.** The fingerprint hashes the *content* of
the rendered geometry fields (``species``, ``fractional_positions``,
``lattice_vectors``) together with ``Structure.id``. Two calls return equal
fingerprints iff the id and the geometry values are equal, regardless of which
list objects hold them. This makes the guard safe against every mutation
pattern:

- reassignment of the geometry lists (the pattern used by
  ``ProjectService._repair_supercell_asset`` / ``regenerate_supercell_asset``,
  which swap geometry while keeping the same ``Structure.id``),
- in-place element mutation of those lists, and
- CPython ``id()`` recycling — an identity-based fingerprint could falsely
  match when a freed list's address is reused by a later replacement list
  (list_A -> list_B -> list_C between two renders can leave
  ``id(list_C) == id(list_A)``), so identities must not be used here.

Hashing tuples of a few thousand floats/strings costs tens of microseconds —
negligible next to the multi-second viewer rebuild the guard exists to skip.

``properties`` is deliberately excluded here because widgets render different
slices of it. Callers that render property-derived data (e.g. atomic spins) must
fold a *content-based* snapshot of the relevant keys into their own guard on top
of :func:`structure_geometry_fingerprint` — ``properties`` values are mutated in
place elsewhere (e.g. ``MainWindow._on_host_magnetic_configuration_changed``),
so reference- or identity-based snapshots would miss changes.
"""

from __future__ import annotations

from mufinder.domain.models import Structure

# Sentinel distinguishing "no structure" from any real fingerprint tuple.
NO_STRUCTURE: tuple[object, ...] = ()


def structure_geometry_fingerprint(structure: Structure | None) -> tuple[object, ...]:
    """Return a content-based fingerprint of a structure's rendered geometry.

    The fingerprint combines ``structure.id`` with hashes of the ``species``,
    ``fractional_positions`` and ``lattice_vectors`` contents, so it changes
    whenever the rendered geometry values change — however that change was made
    (list reassignment or in-place element mutation). ``None`` maps to
    :data:`NO_STRUCTURE`, which never compares equal to a real structure's
    fingerprint (a real structure always contributes its ``id`` as the first
    element).
    """

    if structure is None:
        return NO_STRUCTURE
    return (
        structure.id,
        hash(tuple(structure.species)),
        hash(
            tuple(
                float(value)
                for position in structure.fractional_positions
                for value in position
            )
        ),
        hash(
            tuple(
                float(value)
                for vector in structure.lattice_vectors
                for value in vector
            )
        ),
    )


__all__ = ["NO_STRUCTURE", "structure_geometry_fingerprint"]
