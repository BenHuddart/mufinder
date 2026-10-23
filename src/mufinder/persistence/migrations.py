"""Load-time migrations between saved-project schema versions.

``ProjectService.load_project`` reads the manifest and every entity payload off
disk, together with the saved ``ui_state``, then passes them to
:func:`migrate_project_payloads` before it builds any domain object. Each entry
in ``_MIGRATIONS`` upgrades payloads saved at one schema version to the next,
in place, and reports which entity files it changed (and :data:`UI_STATE_KEY`
when it changed ``ui_state``) so the caller can make the next save rewrite
them. Migrations never
touch provenance: those records are history and keep the names they were
written with.

A project saved at the current version, or at a newer one, is left untouched.
A manifest without ``schema_version`` predates the field and is read as
schema 1.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from mufinder.domain.models import PROJECT_SCHEMA_VERSION


EntityPayloads = dict[str, list[dict[str, Any]]]
"""Raw entity payloads keyed by entity type, as ``ProjectStore`` loads them."""

EntityKey = tuple[str, str]
"""An ``(entity_type, entity_id)`` pair naming one entity file."""

UI_STATE_KEY: EntityKey = ("ui_state", "ui_state")
"""The key a migration step reports when it changed ``ui_state``."""

MigrationStep = Callable[[EntityPayloads, dict[str, Any]], set[EntityKey]]

_SWEEP_STAGES = frozenset({"convergence_cutoff", "convergence_kpoint"})


def stored_schema_version(manifest: dict[str, Any]) -> int:
    """Return the schema version ``manifest`` was saved at."""

    return int(manifest.get("schema_version", 1))


def migrate_project_payloads(
    manifest: dict[str, Any],
    entities: EntityPayloads,
    ui_state: dict[str, Any] | None = None,
) -> set[EntityKey]:
    """Upgrade ``manifest``, ``entities`` and ``ui_state`` to ``PROJECT_SCHEMA_VERSION`` in place.

    Returns the ``(entity_type, entity_id)`` keys of every entity payload a
    migration step changed or added, plus :data:`UI_STATE_KEY` when a step
    changed ``ui_state``. The manifest's ``schema_version`` is set to the
    version the payloads now conform to.
    """

    version = stored_schema_version(manifest)
    changed: set[EntityKey] = set()
    state = ui_state if ui_state is not None else {}
    while version < PROJECT_SCHEMA_VERSION:
        changed |= _MIGRATIONS[version](entities, state)
        version += 1
    manifest["schema_version"] = version
    return changed


def _sweep_values_replace_cutoff_energies(
    entities: EntityPayloads, ui_state: dict[str, Any]
) -> set[EntityKey]:
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


def _site_positions_split_initial_and_relaxed(
    entities: EntityPayloads, ui_state: dict[str, Any]
) -> set[EntityKey]:
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
        position = (
            data.get("muon_final_fractional_position_host") if isinstance(data, dict) else None
        )
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


#: Magnetic Fields form-state key, result type and how its value becomes record data.
_MAGNETIC_FIELD_FORM_RESULTS: tuple[
    tuple[str, str, Callable[[object], dict[str, Any] | None]], ...
] = (
    (
        "magnetic_field_results",
        "magnetic_field",
        lambda value: (
            {"kind": str(value.get("kind", "")), "results": value["results"]}
            if isinstance(value, dict)
            and isinstance(value.get("results"), list)
            and value["results"]
            else None
        ),
    ),
    (
        "dipolar_tensor_results",
        "magnetic_dipolar_tensor",
        lambda value: {"results": value} if isinstance(value, list) and value else None,
    ),
    (
        "nuclear_relaxation_results",
        "nuclear_relaxation",
        lambda value: {"result": value} if isinstance(value, dict) and value else None,
    ),
)


def _magnetic_field_results_become_results(
    entities: EntityPayloads, ui_state: dict[str, Any]
) -> set[EntityKey]:
    """Schema 3 to 4: Magnetic Fields results become ``results`` entities.

    Schema 3 kept the Magnetic Fields page's latest field, dipolar-tensor and
    nuclear-relaxation results inline in its form state:
    ``ui_state["magnetic_field_form"]`` for the page and the ``form`` of each
    entry in ``ui_state["magnetic_field_workspaces"]``. Each non-empty value
    becomes one result entity with no ``calculation_run_id``, whose
    ``data["owner"]`` is ``"magnetic_fields"`` for the page or the workspace
    id, and its key is removed from the form.
    """

    forms: list[tuple[str, dict[str, Any]]] = []
    page_form = ui_state.get("magnetic_field_form")
    if isinstance(page_form, dict):
        forms.append(("magnetic_fields", page_form))
    workspaces = ui_state.get("magnetic_field_workspaces")
    for workspace in workspaces if isinstance(workspaces, list) else []:
        form = workspace.get("form") if isinstance(workspace, dict) else None
        owner = str(workspace.get("id", "")).strip() if isinstance(workspace, dict) else ""
        if isinstance(form, dict) and owner:
            forms.append((owner, form))

    changed: set[EntityKey] = set()
    created_at = datetime.now(UTC).isoformat()
    for owner, form in forms:
        for key, result_type, to_data in _MAGNETIC_FIELD_FORM_RESULTS:
            if key not in form:
                continue
            data = to_data(form.pop(key))
            changed.add(UI_STATE_KEY)
            if data is None:
                continue
            result_id = str(uuid4())
            entities.setdefault("results", []).append(
                {
                    "id": result_id,
                    "calculation_run_id": None,
                    "result_type": result_type,
                    "data": {"owner": owner, **data},
                    "output_files": [],
                    "derived_structure_ids": [],
                    "created_at": created_at,
                }
            )
            changed.add(("results", result_id))
    return changed


#: mu0 mu_B / (4 pi Å^3) in T per mu_B: the factor that turns muesr's raw dipolar
#: tensor, in Å^-3, into T/mu_B. ``mufinder.science.magnetic_fields`` applies the
#: same factor, ``DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3``, to new tensors.
_DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3 = 1e-7 * 9.274_010_078_3e-24 / 1e-30


def _dipolar_tensors_convert_to_tesla_per_mu_b(
    entities: EntityPayloads, ui_state: dict[str, Any]
) -> set[EntityKey]:
    """Schema 4 to 5: saved dipolar tensors are converted from Å^-3 to T/mu_B.

    Schema 4 stored muesr's dipolar tensor under ``dipolar_tensor_T_per_muB``
    without converting it from Å^-3. Every 3 x 3 tensor in a
    ``magnetic_dipolar_tensor`` result is multiplied by
    :data:`_DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3`; an entry without a
    numeric 3 x 3 tensor is left as it is.
    """

    changed: set[EntityKey] = set()
    for payload in entities.get("results", []):
        if payload.get("result_type") != "magnetic_dipolar_tensor":
            continue
        data = payload.get("data")
        entries = data.get("results") if isinstance(data, dict) else None
        for entry in entries if isinstance(entries, list) else []:
            tensor = entry.get("dipolar_tensor_T_per_muB") if isinstance(entry, dict) else None
            if not _is_numeric_3x3(tensor):
                continue
            entry["dipolar_tensor_T_per_muB"] = [
                [float(value) * _DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3 for value in row]
                for row in tensor
            ]
            changed.add(("results", str(payload.get("id"))))
    return changed


def _is_numeric_3x3(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(row, list)
            and len(row) == 3
            and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in row)
            for row in value
        )
    )


_MIGRATIONS: dict[int, MigrationStep] = {
    1: _sweep_values_replace_cutoff_energies,
    2: _site_positions_split_initial_and_relaxed,
    3: _magnetic_field_results_become_results,
    4: _dipolar_tensors_convert_to_tesla_per_mu_b,
}
"""The step that upgrades payloads saved at each version to the next one."""
