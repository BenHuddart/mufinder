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
from datetime import UTC, datetime, timedelta
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

RETIRED_UI_STATE_KEYS = frozenset(
    {
        "default_cut_off_energy",
        "default_cut_off_energy_method",
        "default_cut_off_energy_threshold",
        "default_cut_off_force_threshold",
        "default_cut_off_stress_threshold",
        "default_kpoint_stress_threshold",
        "_integrity_warnings",
    }
)
"""Top-level ``ui_state`` keys that nothing reads any more.

The ``default_*`` keys copied the adopted convergence defaults, which the
project's ``recommended_*`` fields record. ``_integrity_warnings`` held what the
integrity check found at a load and was saved with the rest of the state, so the
warning came back after the problem was mended; the check now runs at each load
and its findings are never saved. :func:`drop_retired_ui_state_keys` removes
them when a project loads, whatever its schema version.
"""


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


def drop_retired_ui_state_keys(ui_state: dict[str, Any]) -> set[EntityKey]:
    """Remove every :data:`RETIRED_UI_STATE_KEYS` entry from ``ui_state`` in place.

    Returns ``{UI_STATE_KEY}`` when a key was removed, so the next save
    rewrites ``ui_state`` without it, and an empty set otherwise.
    """

    if not isinstance(ui_state, dict):
        return set()
    retired = RETIRED_UI_STATE_KEYS.intersection(ui_state)
    for key in retired:
        del ui_state[key]
    return {UI_STATE_KEY} if retired else set()


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


def _relaxed_structure_spins_are_read_from_job_files(
    entities: EntityPayloads, ui_state: dict[str, Any]
) -> set[EntityKey]:
    """Schema 5 to 6: relaxed structures keep their run's ``SPIN=`` values.

    The values live in each relaxation's job files, which payloads cannot
    reach, so this step changes nothing. ``ProjectService.load_project``
    recovers them for a project stored below schema 6
    (``mufinder.services.relaxed_structure_spins``).
    """

    return set()


#: Stored job statuses of schema 6 and below that became ``active``; each names the job's activity.
_ACTIVE_STATUSES = frozenset({"submitting", "queued", "running", "retrieving"})
#: Schema 6 ``failure_reason`` values; each is kept as the ``end_reason`` of an ended job.
_FAILURE_REASONS = frozenset(
    {
        "walltime",
        "out_of_memory",
        "node_failure",
        "castep_error",
        "output_unavailable",
        "interrupted",
        "submission",
    }
)


def legacy_job_state(status: object, failure_reason: object = None) -> dict[str, str | None] | None:
    """Return the schema 7 state of a job stored at schema 6 with ``status`` and ``failure_reason``.

    The result holds ``status`` (``defined``, ``active``, ``completed`` or
    ``ended``), ``activity`` (the old status of an active job, else ``None``)
    and ``end_reason`` (else ``None``). A stopped or failed job keeps its
    failure reason; one with none, or with a value schema 6 did not know, ended
    for an ``unknown`` reason. A cancelled job ended because it was
    ``cancelled``. ``None`` means ``status`` is not a schema 6 status, and the
    job is left as it is.
    """

    if status in {"defined", "completed"}:
        return {"status": str(status), "activity": None, "end_reason": None}
    if status in _ACTIVE_STATUSES:
        return {"status": "active", "activity": str(status), "end_reason": None}
    if status == "cancelled":
        return {"status": "ended", "activity": None, "end_reason": "cancelled"}
    if status in {"stopped", "failed"}:
        reason = failure_reason if failure_reason in _FAILURE_REASONS else "unknown"
        return {"status": "ended", "activity": None, "end_reason": str(reason)}
    return None


def _job_states_collapse_to_four(
    entities: EntityPayloads, ui_state: dict[str, Any]
) -> set[EntityKey]:
    """Schema 6 to 7: a job stores one of four states, with its activity or end reason.

    Schema 6 stored nine job statuses and a ``failure_reason``. Each job
    payload's ``status``, ``activity`` and ``end_reason`` are set by
    :func:`legacy_job_state` and ``failure_reason`` is removed. A run stores
    no activity or reason, so only its ``status`` is rewritten. A payload with
    no status, or one schema 6 did not know, is left as it is.
    """

    changed: set[EntityKey] = set()
    for payload in entities.get("jobs", []):
        state = legacy_job_state(payload.get("status"), payload.get("failure_reason"))
        if state is None:
            continue
        payload.pop("failure_reason", None)
        payload.update(state)
        changed.add(("jobs", str(payload.get("id"))))
    for payload in entities.get("runs", []):
        state = legacy_job_state(payload.get("status"))
        if state is None:
            continue
        payload["status"] = state["status"]
        changed.add(("runs", str(payload.get("id"))))
    return changed


UEP_ANALYSIS_STATES_KEY = "uep_analysis_states"
"""The ``ui_state`` key of the UEP analysis window's state of each source job."""

_UEP_SEARCH_SETTING_KEYS = (
    "gw_factor",
    "batch_count",
    "batch_min_muon_muon",
    "batch_min_muon_atom",
    "batch_max_attempts",
    "batch_optimization_tolerance_ev",
    "batch_max_steps",
)

_UEP_SEARCH_KEYS = frozenset({"batch_results", "batch_summary_data", "batch_summary_text"})

RETIRED_UEP_STATE_KEYS = frozenset(
    {
        # The retired summary, point-probe, line and plane pages.
        "probe_point",
        "latest_probe",
        "probe_output",
        "line_start",
        "line_end",
        "line_points",
        "line_sample",
        "line_data",
        "plane_origin",
        "plane_x",
        "plane_y",
        "plane_points_x",
        "plane_points_y",
        "plane_sample",
        "plane_data",
        # A potential grid kept inline, the retired clustering window, and what
        # older windows saved of their statuses and sliders.
        "potential_grid",
        "clustering_window_state",
        "batch_zpe_completed",
        "isosurface_slider_value",
        "isosurface_status_text",
        "isosurface_status_kind",
        "slice_status_text",
        "slice_status_kind",
    }
)
"""Keys of a UEP analysis state that schema 8 drops: nothing reads them."""

_UEP_LAYER_DROPPED_KEYS = frozenset({"mesh", "status_text", "status_kind"})
_UEP_PLANE_DROPPED_KEYS = frozenset({"sample", "status_text", "status_kind"})

UEP_POTENTIAL_GRID_RESULT_TYPE = "uep_potential_grid"
"""The result type older builds stored the full-cell UEP potential grid as."""

UEP_BATCH_SEARCH_RESULT_TYPE = "uep_batch_search"
"""The result type of a stored UEP batch search (``mufinder.services.uep_results``)."""


def _uep_state_understood(state: object) -> bool:
    """Whether :func:`_uep_batch_searches_become_results` can move ``state`` whole."""

    if not isinstance(state, dict):
        return False
    results = state.get("batch_results")
    if results is not None and not (
        isinstance(results, list) and all(isinstance(entry, dict) for entry in results)
    ):
        return False
    summary = state.get("batch_summary_data")
    if summary is not None and not isinstance(summary, dict):
        return False
    text = state.get("batch_summary_text")
    if text is not None and not isinstance(text, str):
        return False
    layers = state.get("isosurface_layers")
    if layers is not None and not isinstance(layers, list):
        return False
    planes = state.get("slice_planes")
    if planes is not None and not isinstance(planes, list):
        return False
    mesh = state.get("isosurface_mesh")
    return mesh is None or isinstance(mesh, dict)


def _uep_display_state(state: dict[str, Any]) -> dict[str, Any]:
    """``state`` without its search, retired keys, meshes and slice samples."""

    display = {
        key: value
        for key, value in state.items()
        if key not in _UEP_SEARCH_KEYS and key not in RETIRED_UEP_STATE_KEYS
    }
    if isinstance(display.get("isosurface_layers"), list):
        display["isosurface_layers"] = [
            (
                {k: v for k, v in layer.items() if k not in _UEP_LAYER_DROPPED_KEYS}
                if isinstance(layer, dict)
                else layer
            )
            for layer in display["isosurface_layers"]
        ]
    if isinstance(display.get("slice_planes"), list):
        display["slice_planes"] = [
            (
                {k: v for k, v in plane.items() if k not in _UEP_PLANE_DROPPED_KEYS}
                if isinstance(plane, dict)
                else plane
            )
            for plane in display["slice_planes"]
        ]
    mesh = display.get("isosurface_mesh")
    if isinstance(mesh, dict):
        # The level is all a window reads of a single saved mesh.
        display["isosurface_mesh"] = {"level_ev": mesh.get("level_ev")}
    return display


def _uep_batch_searches_become_results(
    entities: EntityPayloads, ui_state: dict[str, Any]
) -> set[EntityKey]:
    """Schema 7 to 8: UEP batch searches become results; the potential grid is not stored.

    The step is :func:`move_uep_searches_to_results`, which loading also runs at
    every schema version.
    """

    return move_uep_searches_to_results(entities, ui_state)


def move_uep_searches_to_results(
    entities: EntityPayloads, ui_state: dict[str, Any]
) -> set[EntityKey]:
    """Move UEP batch searches kept in ``ui_state`` into results, in place.

    Schema 7 kept each UEP source job's batch search in its analysis state,
    ``ui_state["uep_analysis_states"][job_id]``: ``batch_results``,
    ``batch_summary_data`` and ``batch_summary_text``, beside the form's search
    settings. Every state with results becomes one ``uep_batch_search`` result whose
    ``data`` holds ``source_job_id``, ``settings`` (the form values and ``gw_factor``,
    which the state keeps too), ``results``, ``grouping`` and ``log``, moved as they
    were; its ``calculation_run_id`` is the job's latest run, or ``None``. Records
    are created in the order of the states. The three keys leave every state, with
    :data:`RETIRED_UEP_STATE_KEYS`, the isosurface meshes and the slice samples.
    Every ``uep_potential_grid`` result is removed.

    A build from before schema 8 reads and writes searches in the state only, and
    keeps the manifest's schema version, so loading runs this at every version. A
    ``uep_batch_search`` record already stored for a job whose state holds a search
    is replaced under its id by that later search. An empty ``batch_results`` never
    removes a record: such a build writes one whenever its window closes, as it does
    not read the record, so a search it discarded comes back here rather than one it
    never showed being lost. Run again on its own output, it changes nothing. A state it does not understand (see
    :func:`_uep_state_understood`) is left as it is. The move is made on copies and
    applied only once every state has been read.

    Returns the keys of the result payloads added or replaced, and
    :data:`UI_STATE_KEY` when ``ui_state`` changed.
    """

    changed: set[EntityKey] = set()
    results = entities.setdefault("results", [])
    states = ui_state.get(UEP_ANALYSIS_STATES_KEY)
    if not isinstance(states, dict):
        states = {}

    # A build from before schema 8 may have saved a search over one this step moved
    # earlier; the state's search is the later one and takes the record's place.
    stored = {
        str(payload["data"].get("source_job_id")): str(payload.get("id"))
        for payload in results
        if isinstance(payload, dict)
        and payload.get("result_type") == UEP_BATCH_SEARCH_RESULT_TYPE
        and isinstance(payload.get("data"), dict)
    }
    latest_run: dict[str, tuple[int, str]] = {}
    for run in entities.get("runs", []):
        if not isinstance(run, dict):
            continue
        job_id = str(run.get("job_id"))
        try:
            index = int(run.get("run_index", 0))
        except (TypeError, ValueError):
            index = 0
        if job_id not in latest_run or index > latest_run[job_id][0]:
            latest_run[job_id] = (index, str(run.get("id")))

    created = datetime.now(UTC)
    new_states = dict(states)
    new_results: list[dict[str, Any]] = []
    for job_id, state in states.items():
        if not _uep_state_understood(state):
            continue
        display = _uep_display_state(state)
        search = state.get("batch_results")
        if search:
            run = latest_run.get(str(job_id))
            new_results.append(
                {
                    "id": stored.get(str(job_id)) or str(uuid4()),
                    "calculation_run_id": None if run is None else run[1],
                    "result_type": UEP_BATCH_SEARCH_RESULT_TYPE,
                    "data": {
                        "source_job_id": str(job_id),
                        "settings": {
                            key: state[key] for key in _UEP_SEARCH_SETTING_KEYS if key in state
                        },
                        "results": search,
                        "grouping": state.get("batch_summary_data"),
                        "log": state.get("batch_summary_text") or "",
                    },
                    "output_files": [],
                    "derived_structure_ids": [],
                    "created_at": (created + timedelta(microseconds=len(new_results))).isoformat(),
                }
            )
        if display != state:
            new_states[job_id] = display

    if new_states != states:
        ui_state[UEP_ANALYSIS_STATES_KEY] = new_states
        changed.add(UI_STATE_KEY)
    replaced = {payload["id"] for payload in new_results}
    results[:] = [
        payload
        for payload in results
        if not (
            isinstance(payload, dict)
            and (
                payload.get("result_type") == UEP_POTENTIAL_GRID_RESULT_TYPE
                or payload.get("id") in replaced
            )
        )
    ]
    results.extend(new_results)
    changed |= {("results", payload["id"]) for payload in new_results}
    return changed


_MIGRATIONS: dict[int, MigrationStep] = {
    1: _sweep_values_replace_cutoff_energies,
    2: _site_positions_split_initial_and_relaxed,
    3: _magnetic_field_results_become_results,
    4: _dipolar_tensors_convert_to_tesla_per_mu_b,
    5: _relaxed_structure_spins_are_read_from_job_files,
    6: _job_states_collapse_to_four,
    7: _uep_batch_searches_become_results,
}
"""The step that upgrades payloads saved at each version to the next one."""
