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


_LEGACY_SITE_POSITION_KEYS = {
    "fractional_position": "initial_fractional_position",
    "cartesian_position": "initial_cartesian_position",
}


def _site_positions_split_initial_and_relaxed(entities: EntityPayloads) -> set[EntityKey]:
    """Schema 2 to 3: a muon site keeps its initial and relaxed positions apart.

    Schema 2 stored the position a site was generated at as
    ``fractional_position`` / ``cartesian_position`` and never updated it when
    the site was relaxed. Those keys become ``initial_fractional_position`` /
    ``initial_cartesian_position``. A site linked to a relaxation run gains
    ``relaxed_fractional_position``, copied from that run's result
    ``muon_final_fractional_position_host`` when the result records one.
    """

    results_by_id = {str(payload.get("id")): payload for payload in entities.get("results", [])}
    results_by_run = {
        str(payload.get("calculation_run_id")): payload
        for payload in entities.get("results", [])
        if payload.get("calculation_run_id") is not None
    }
    runs_by_id = {str(payload.get("id")): payload for payload in entities.get("runs", [])}

    def relaxed_position(run_id: object) -> list[float] | None:
        run = runs_by_id.get(str(run_id))
        if run is None:
            return None
        result = results_by_id.get(str(run.get("result_id"))) or results_by_run.get(str(run_id))
        data = result.get("data") if result is not None else None
        position = data.get("muon_final_fractional_position_host") if isinstance(data, dict) else None
        if not isinstance(position, list) or len(position) != 3:
            return None
        try:
            return [float(value) for value in position]
        except (TypeError, ValueError):
            return None

    changed: set[EntityKey] = set()
    for payload in entities.get("sites", []):
        site_changed = False
        for legacy_key, key in _LEGACY_SITE_POSITION_KEYS.items():
            if legacy_key in payload:
                value = payload.pop(legacy_key)
                payload.setdefault(key, value)
                site_changed = True
        if payload.get("relaxed_fractional_position") is None and payload.get("relaxation_run_id"):
            position = relaxed_position(payload["relaxation_run_id"])
            if position is not None:
                payload["relaxed_fractional_position"] = position
                site_changed = True
        if site_changed:
            changed.add(("sites", str(payload.get("id"))))
    return changed


_MIGRATIONS: dict[int, Callable[[EntityPayloads], set[EntityKey]]] = {
    1: _sweep_values_replace_cutoff_energies,
    2: _site_positions_split_initial_and_relaxed,
}
"""The step that upgrades payloads saved at each version to the next one."""
