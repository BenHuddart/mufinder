"""Load-time migrations between saved-project schema versions.

``ProjectService.load_project`` reads the manifest and every entity payload off
disk, then passes them to :func:`migrate_project_payloads` before it builds any
domain object. Each entry in ``_MIGRATIONS`` upgrades payloads saved at one
schema version to the next, in place, and reports which entity files it
changed so the caller can make the next save rewrite them. Migrations never
touch provenance: those records are history and keep the names they were
written with.

A project saved at the current version, or at a newer one, is left untouched.
A manifest without ``schema_version`` predates the field and is read as
schema 1.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mufinder.domain.models import PROJECT_SCHEMA_VERSION


EntityPayloads = dict[str, list[dict[str, Any]]]
"""Raw entity payloads keyed by entity type, as ``ProjectStore`` loads them."""

EntityKey = tuple[str, str]
"""An ``(entity_type, entity_id)`` pair naming one entity file."""

_SWEEP_STAGES = frozenset({"convergence_cutoff", "convergence_kpoint"})


def stored_schema_version(manifest: dict[str, Any]) -> int:
    """Return the schema version ``manifest`` was saved at."""

    return int(manifest.get("schema_version", 1))


def migrate_project_payloads(manifest: dict[str, Any], entities: EntityPayloads) -> set[EntityKey]:
    """Upgrade ``manifest`` and ``entities`` to ``PROJECT_SCHEMA_VERSION`` in place.

    Returns the ``(entity_type, entity_id)`` keys of every entity payload a
    migration step changed. The manifest's ``schema_version`` is set to the
    version the payloads now conform to.
    """

    version = stored_schema_version(manifest)
    changed: set[EntityKey] = set()
    while version < PROJECT_SCHEMA_VERSION:
        changed |= _MIGRATIONS[version](entities)
        version += 1
    manifest["schema_version"] = version
    return changed


def _sweep_values_replace_cutoff_energies(entities: EntityPayloads) -> set[EntityKey]:
    """Schema 1 to 2: convergence definitions keep their sweep in ``sweep_values`` only.

    Schema 1 wrote a convergence definition's configuration with both
    ``sweep_values`` and the legacy ``cutoff_energies``, and read a cutoff
    sweep from ``cutoff_energies`` first. The values that reader used become
    ``sweep_values`` and ``cutoff_energies`` is removed, for cutoff and k-point
    sweeps alike.
    """

    changed: set[EntityKey] = set()
    for payload in entities.get("calculations", []):
        if payload.get("workflow_stage") not in _SWEEP_STAGES:
            continue
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            continue
        configuration = parameters.get("configuration")
        if not isinstance(configuration, dict) or "cutoff_energies" not in configuration:
            continue
        legacy_values = configuration.pop("cutoff_energies")
        sweep_parameter = str(configuration.get("sweep_parameter", "cut_off_energy")).strip()
        cutoff_sweep = sweep_parameter in {"", "cut_off_energy"}
        if isinstance(legacy_values, list) and (
            cutoff_sweep or not isinstance(configuration.get("sweep_values"), list)
        ):
            configuration["sweep_values"] = legacy_values
        changed.add(("calculations", str(payload.get("id"))))
    return changed


_MIGRATIONS: dict[int, Callable[[EntityPayloads], set[EntityKey]]] = {
    1: _sweep_values_replace_cutoff_energies,
}
"""The step that upgrades payloads saved at each version to the next one."""
