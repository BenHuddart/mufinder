"""Keep calculation definitions' job mappings and rows consistent.

A workflow can hold several calculation definitions for the same context: a
user who reconfigures a table creates a new definition rather than editing the
old one in place. Each definition binds its rows to jobs through a
``job_ids_by_<key>`` mapping in its parameters. Two operations keep those
definitions consistent, and every workflow runs them through this module:

- :func:`sync_equivalent_mappings` copies job bindings from every equivalent
  definition into the one a workflow is working on, so a row whose
  configuration is unchanged keeps the job an older definition ran for it.
- :func:`remove_labels_from_definition` drops rows and their job bindings from
  a definition, and :func:`remove_definition_if_no_rows` deletes a definition
  left with no rows. The convergence sweeps, whose configurations list sweep
  values rather than rows, and the energy-barrier definitions, which list
  sources, have their own removals: :func:`remove_grid_keys_from_definition`,
  :func:`remove_cutoff_from_definition` and
  :func:`remove_source_keys_from_definition`; the generated paths those sources
  come from are dropped from the Energy Barriers form state by
  :func:`remove_generated_sources`.

The convergence sweeps match definitions by context (input structure, output
directory, simulation cell mode and supercell source) through
:func:`sweep_definition_has_context` and :func:`sweep_definition_context_matches`;
:func:`matching_sweep_definition` finds the definition a new cutoff
configuration would duplicate, so reconfiguring it unchanged reuses that
definition.

The functions take a workflow's configuration codec and row helpers as
callables, because the configuration types belong to the application layer.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, TypeVar
from uuid import UUID

from mufinder.domain.enums import JobStatus
from mufinder.domain.models import CalculationDefinition
from mufinder.services.job_paths import definition_output_directory
from mufinder.services.project_service import ProjectDocument
from mufinder.services.source_candidates import ENERGY_BARRIERS_FORM_STATE_KEY

ConfigT = TypeVar("ConfigT")
KeyT = TypeVar("KeyT", bound=Hashable)

#: The mapping that binds a row-keyed workflow's rows to their jobs.
LABEL_MAPPING = "job_ids_by_label"

_IN_FLIGHT_STATUSES = frozenset(
    {JobStatus.SUBMITTING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING}
)
_TERMINAL_UNSUCCESSFUL_STATUSES = frozenset(
    {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
)


class _RowConfig(Protocol):
    rows: tuple[Any, ...]


def _mapped_job_status(document: ProjectDocument, raw_job_id: str) -> JobStatus | None:
    try:
        job_id = UUID(str(raw_job_id))
    except ValueError:
        return None
    job = document.jobs.get(job_id)
    return None if job is None else job.status


def convergence_mapping_job_priority(document: ProjectDocument, raw_job_id: str) -> int:
    """Rank the job ``raw_job_id`` names when two definitions bind the same row.

    In flight (50) beats completed (40), which beats failed, stopped or
    cancelled (30), which beats defined (10). A malformed id or a missing job
    ranks 0. The convergence sweeps and the UEP workflow use this order.
    """

    status = _mapped_job_status(document, raw_job_id)
    if status in _IN_FLIGHT_STATUSES:
        return 50
    if status == JobStatus.COMPLETED:
        return 40
    if status in _TERMINAL_UNSUCCESSFUL_STATUSES:
        return 30
    if status == JobStatus.DEFINED:
        return 10
    return 0


def relaxation_mapping_job_priority(document: ProjectDocument, raw_job_id: str) -> int:
    """Rank the job ``raw_job_id`` names, as the two relaxation workflows do.

    The same order as :func:`convergence_mapping_job_priority` except that a
    defined job (35) beats a failed, stopped or cancelled one (30): a
    relaxation row the user has re-staged keeps its fresh job.
    """

    status = _mapped_job_status(document, raw_job_id)
    if status in _IN_FLIGHT_STATUSES:
        return 50
    if status == JobStatus.COMPLETED:
        return 40
    if status == JobStatus.DEFINED:
        return 35
    if status in _TERMINAL_UNSUCCESSFUL_STATUSES:
        return 30
    return 0


def definition_workflow_context_matches(
    document: ProjectDocument,
    source: CalculationDefinition,
    target: CalculationDefinition,
) -> bool:
    """Return whether ``source`` shares ``target``'s workflow context.

    Both must have the same workflow stage, calc type, input structures and
    output directory. The relaxation and UEP workflows sync mappings between
    definitions that match this way.
    """

    return (
        source.workflow_stage == target.workflow_stage
        and source.calc_type == target.calc_type
        and source.input_structure_ids == target.input_structure_ids
        and definition_output_directory(document, source)
        == definition_output_directory(document, target)
    )


def sweep_definition_has_context(
    document: ProjectDocument,
    definition: CalculationDefinition,
    *,
    input_structure_id: UUID,
    output_directory: Path,
    simulation_cell_mode: str,
    supercell_source_workflow: str | None,
) -> bool:
    """Return whether convergence sweep ``definition`` runs in the given context.

    ``definition`` must have exactly the input structure ``input_structure_id``,
    the output directory ``output_directory``, the simulation cell mode
    ``simulation_cell_mode`` (``"unit_cell"`` when unset) and the supercell
    source workflow ``supercell_source_workflow``.
    """

    return (
        definition.input_structure_ids == [input_structure_id]
        and definition_output_directory(document, definition) == output_directory
        and str(definition.parameters.get("simulation_cell_mode", "unit_cell"))
        == simulation_cell_mode
        and definition.parameters.get("supercell_source_workflow") == supercell_source_workflow
    )


def sweep_definition_context_matches(
    document: ProjectDocument,
    source: CalculationDefinition,
    target: CalculationDefinition,
) -> bool:
    """Return whether ``source`` shares a convergence sweep ``target``'s context.

    ``target`` needs an input structure and an output directory. ``source``
    must then have ``target``'s context, as
    :func:`sweep_definition_has_context` checks it: exactly that input
    structure, the same output directory, the same ``simulation_cell_mode``
    (``"unit_cell"`` when unset) and the same ``supercell_source_workflow``.
    """

    if not target.input_structure_ids:
        return False
    output_directory = definition_output_directory(document, target)
    if output_directory is None:
        return False
    return sweep_definition_has_context(
        document,
        source,
        input_structure_id=target.input_structure_ids[0],
        output_directory=output_directory,
        simulation_cell_mode=str(target.parameters.get("simulation_cell_mode", "unit_cell")),
        supercell_source_workflow=target.parameters.get("supercell_source_workflow"),
    )


def matching_sweep_definition(
    document: ProjectDocument,
    definitions: Iterable[CalculationDefinition | None],
    *,
    parameters: Mapping[str, object],
    input_structure_id: UUID,
    output_directory: Path,
    simulation_cell_mode: str,
    supercell_source_workflow: str | None,
) -> CalculationDefinition | None:
    """Return the first sweep definition a new configuration would duplicate.

    That is the first of ``definitions`` whose stored ``configuration`` equals
    ``parameters`` and that runs in the given context
    (:func:`sweep_definition_has_context`). ``None`` entries and repeated
    definitions are skipped, so a caller can put the active definition first
    and then list them all. Returns ``None`` when no definition matches.
    """

    seen: set[UUID] = set()
    for definition in definitions:
        if definition is None or definition.id in seen:
            continue
        seen.add(definition.id)
        if definition.parameters.get("configuration") == parameters and (
            sweep_definition_has_context(
                document,
                definition,
                input_structure_id=input_structure_id,
                output_directory=output_directory,
                simulation_cell_mode=simulation_cell_mode,
                supercell_source_workflow=supercell_source_workflow,
            )
        ):
            return definition
    return None


def ensure_job_mapping(
    definition: CalculationDefinition, mapping_name: str = LABEL_MAPPING
) -> dict[str, object]:
    """Return ``definition``'s ``mapping_name`` mapping, creating it when absent.

    A value that is not a mapping is replaced by an empty one.
    """

    mapping = definition.parameters.setdefault(mapping_name, {})
    if not isinstance(mapping, dict):
        mapping = {}
        definition.parameters[mapping_name] = mapping
    return mapping


def _string_key(raw_key: object) -> str | None:
    return raw_key if isinstance(raw_key, str) else None


def sync_equivalent_mappings(
    document: ProjectDocument,
    definitions: Iterable[CalculationDefinition],
    target: CalculationDefinition,
    *,
    deserialize: Callable[[object], ConfigT | None],
    signatures: Callable[[ConfigT], Mapping[KeyT, object]],
    priority: Callable[[ProjectDocument, str], int],
    mapping_name: str = LABEL_MAPPING,
    context_matches: Callable[
        [ProjectDocument, CalculationDefinition, CalculationDefinition], bool
    ] = definition_workflow_context_matches,
    parse_key: Callable[[object], KeyT | None] = _string_key,  # type: ignore[assignment]
    format_key: Callable[[KeyT], str] = str,
    unlisted_keys_match: bool = True,
    accept_job: Callable[[str, str], bool] | None = None,
    legacy_keys: Callable[[ConfigT], Mapping[str, str]] | None = None,
    seed_mapping_name: str | None = None,
) -> None:
    """Copy job bindings from definitions equivalent to ``target`` into it.

    Every definition in ``definitions`` (``target`` included) whose context
    matches ``target``'s and whose configuration deserialises is a source. A
    source's ``mapping_name`` entry is a candidate for ``target`` when its key
    parses and the row it names has the same signature in both
    configurations. Among candidates for one key the job with the highest
    ``priority`` wins (the first one seen on a tie), and it replaces
    ``target``'s own binding only when that binding is missing or ranks
    strictly lower. ``target`` is not touched when there is no candidate.

    Parameters:

    - ``signatures`` maps each row key a configuration lists to the row's
      signature; two rows are equivalent when their signatures are equal.
      With ``unlisted_keys_match`` a key neither configuration lists still
      counts as equivalent (the row workflows); without it, both must list it
      (the convergence sweeps).
    - ``parse_key`` turns a raw mapping key into a row key, or ``None`` to
      skip it; ``format_key`` spells the row key in ``target``'s mapping. The
      defaults accept string keys unchanged.
    - ``accept_job(key, raw_job_id)`` rejects a candidate before it is
      ranked.
    - ``legacy_keys(config)`` maps a source's older key spellings to the row
      keys its configuration now uses; a string mapping key found there is
      translated before it is parsed.
    - ``seed_mapping_name`` names a mapping of per-row seed names that travels
      with the job binding. A winning candidate brings the seed name its
      source stored under the raw key or the translated one, stripped; a blank
      one is ignored. ``target``'s seed mapping is created whenever a binding
      is copied.
    """

    target_config = deserialize(target.parameters.get("configuration", {}))
    if target_config is None:
        return
    target_signatures = signatures(target_config)

    def equivalent(key: KeyT, source_signatures: Mapping[KeyT, object]) -> bool:
        if unlisted_keys_match:
            return target_signatures.get(key) == source_signatures.get(key)
        return (
            key in target_signatures
            and key in source_signatures
            and target_signatures[key] == source_signatures[key]
        )

    candidates: dict[str, str] = {}
    seed_candidates: dict[str, str] = {}
    for definition in definitions:
        if not context_matches(document, definition, target):
            continue
        source_config = deserialize(definition.parameters.get("configuration", {}))
        if source_config is None:
            continue
        source_signatures = signatures(source_config)
        translations = legacy_keys(source_config) if legacy_keys is not None else {}
        mapping = definition.parameters.get(mapping_name, {})
        if not isinstance(mapping, dict):
            continue
        seed_mapping = (
            definition.parameters.get(seed_mapping_name, {})
            if seed_mapping_name is not None
            else None
        )
        for raw_key, raw_job_id in mapping.items():
            if not isinstance(raw_job_id, str):
                continue
            translated = translations.get(raw_key, raw_key) if isinstance(raw_key, str) else raw_key
            key = parse_key(translated)
            if key is None or not equivalent(key, source_signatures):
                continue
            mapping_key = format_key(key)
            if accept_job is not None and not accept_job(mapping_key, raw_job_id):
                continue
            existing = candidates.get(mapping_key)
            if existing is None or priority(document, raw_job_id) > priority(document, existing):
                candidates[mapping_key] = raw_job_id
                if isinstance(seed_mapping, dict):
                    raw_seed = seed_mapping.get(raw_key, seed_mapping.get(mapping_key))
                    if isinstance(raw_seed, str) and raw_seed.strip():
                        seed_candidates[mapping_key] = raw_seed.strip()

    if not candidates:
        return
    target_mapping = ensure_job_mapping(target, mapping_name)
    target_seed_mapping = (
        ensure_job_mapping(target, seed_mapping_name) if seed_mapping_name is not None else None
    )
    for mapping_key, raw_job_id in candidates.items():
        existing = target_mapping.get(mapping_key)
        if not isinstance(existing, str) or priority(document, raw_job_id) > priority(
            document, existing
        ):
            target_mapping[mapping_key] = raw_job_id
            raw_seed = seed_candidates.get(mapping_key)
            if raw_seed and target_seed_mapping is not None:
                target_seed_mapping[mapping_key] = raw_seed


def remove_definition_if_no_rows(
    document: ProjectDocument,
    definition: CalculationDefinition,
    active_key: str | None,
) -> bool:
    """Delete ``definition`` when its configuration's ``rows`` list is empty.

    A configuration that is not a mapping, or whose ``rows`` is not a list,
    keeps the definition. When ``ui_state[active_key]`` names the deleted
    definition it is cleared too. Returns whether the definition was deleted.
    """

    configuration = definition.parameters.get("configuration")
    if not isinstance(configuration, dict):
        return False
    rows = configuration.get("rows")
    if not isinstance(rows, list) or rows:
        return False
    document.calculations.pop(definition.id, None)
    if active_key is not None and document.ui_state.get(active_key) == str(definition.id):
        document.ui_state.pop(active_key, None)
    return True


def remove_labels_from_definition(
    document: ProjectDocument,
    definition: CalculationDefinition,
    labels: Collection[str],
    *,
    deserialize: Callable[[object], ConfigT | None],
    serialize: Callable[[ConfigT], dict[str, object]],
    active_key: str | None,
    row_matches: Callable[[Any], bool] | None = None,
    key_matches: Callable[[str], bool] | None = None,
    mapping_names: Iterable[str] = (LABEL_MAPPING,),
) -> bool:
    """Drop the rows ``labels`` name from ``definition`` and unbind their jobs.

    ``row_matches(row)`` picks the configuration rows to drop (default: the
    row's ``label`` is in ``labels``); the configuration is rewritten only
    when a row is dropped. ``key_matches(key)`` picks the string keys to drop
    from each of ``mapping_names`` (default: the key is in ``labels``). A
    definition left with no rows is then deleted through
    :func:`remove_definition_if_no_rows`. Returns whether it was deleted.
    """

    if row_matches is None:
        row_matches = lambda row: row.label in labels  # noqa: E731
    if key_matches is None:
        key_matches = lambda key: key in labels  # noqa: E731

    config: _RowConfig | None = deserialize(definition.parameters.get("configuration", {}))  # type: ignore[assignment]
    if config is not None:
        rows = tuple(row for row in config.rows if not row_matches(row))
        if len(rows) != len(config.rows):
            definition.parameters["configuration"] = serialize(replace(config, rows=rows))  # type: ignore[type-var]

    for mapping_name in mapping_names:
        mapping = definition.parameters.get(mapping_name, {})
        if not isinstance(mapping, dict):
            continue
        for key in [key for key in mapping if isinstance(key, str) and key_matches(key)]:
            mapping.pop(key, None)

    return remove_definition_if_no_rows(document, definition, active_key)


def remove_grid_keys_from_definition(
    definition: CalculationDefinition,
    keys: Collection[str],
    *,
    mapping_name: str,
    mapping_key: Callable[[tuple[int, int, int]], str],
) -> None:
    """Drop the k-point grids whose mapping keys are ``keys`` from a sweep definition.

    A ``sweep_values`` entry is dropped when it is a three-item list of
    integers whose ``mapping_key`` is in ``keys``; any other entry is kept.
    The grids' entries in ``mapping_name`` are dropped too. The definition is
    kept even when no grid remains.
    """

    if not keys:
        return
    configuration = definition.parameters.get("configuration")
    if isinstance(configuration, dict):
        raw_values = configuration.get("sweep_values", [])
        updated_values: list[object] = []
        if isinstance(raw_values, list):
            for raw_value in raw_values:
                if not isinstance(raw_value, list) or len(raw_value) != 3:
                    updated_values.append(raw_value)
                    continue
                try:
                    key = mapping_key((int(raw_value[0]), int(raw_value[1]), int(raw_value[2])))
                except (TypeError, ValueError):
                    updated_values.append(raw_value)
                    continue
                if key in keys:
                    continue
                updated_values.append(raw_value)
            configuration["sweep_values"] = updated_values

    mapping = definition.parameters.get(mapping_name, {})
    if isinstance(mapping, dict):
        for key in keys:
            mapping.pop(key, None)


def remove_cutoff_from_definition(
    document: ProjectDocument,
    definition: CalculationDefinition,
    cutoff: float,
    *,
    mapping_name: str,
    mapping_key: Callable[[float], str],
    active_key: str | None,
) -> bool:
    """Drop one cutoff energy from a cutoff-sweep definition.

    Every ``sweep_values`` entry equal to ``cutoff`` is dropped; the position
    of the first dropped entry also drops the matching ``job_labels`` entry.
    Override groups lose the cutoff and an emptied group is dropped; a
    malformed group or value in one is dropped as well. The cutoff's ``mapping_name`` entry is removed.
    A definition left with no cutoff is deleted, clearing
    ``ui_state[active_key]`` when it names the definition. Returns whether the
    definition was deleted.
    """

    configuration = definition.parameters.get("configuration")
    if not isinstance(configuration, dict):
        return False

    cutoff_to_remove = float(cutoff)
    remove_index: int | None = None
    raw_sweep_values = configuration.get("sweep_values")
    if isinstance(raw_sweep_values, list):
        updated_sweep_values: list[object] = []
        for index, raw_value in enumerate(raw_sweep_values):
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                updated_sweep_values.append(raw_value)
                continue
            if value == cutoff_to_remove:
                if remove_index is None:
                    remove_index = index
                continue
            updated_sweep_values.append(value)
        configuration["sweep_values"] = updated_sweep_values

    raw_overrides = configuration.get("overrides", [])
    if isinstance(raw_overrides, list):
        updated_overrides: list[dict[str, object]] = []
        for entry in raw_overrides:
            if not isinstance(entry, dict):
                continue
            group_key = "cutoffs" if "cutoffs" in entry else "sweep_values"
            raw_group = entry.get(group_key, [])
            if not isinstance(raw_group, list):
                continue
            remaining: list[float] = []
            for raw_value in raw_group:
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if value == cutoff_to_remove:
                    continue
                remaining.append(value)
            if not remaining:
                continue
            updated_entry = dict(entry)
            updated_entry[group_key] = remaining
            updated_overrides.append(updated_entry)
        configuration["overrides"] = updated_overrides

    mapping = definition.parameters.get(mapping_name, {})
    if isinstance(mapping, dict):
        mapping.pop(mapping_key(cutoff), None)

    labels = definition.parameters.get("job_labels", [])
    if isinstance(labels, list) and remove_index is not None and remove_index < len(labels):
        labels.pop(remove_index)

    if configuration.get("sweep_values"):
        return False
    document.calculations.pop(definition.id, None)
    if active_key is not None and document.ui_state.get(active_key) == str(definition.id):
        document.ui_state.pop(active_key, None)
    return True


def remove_source_keys_from_definition(
    document: ProjectDocument,
    definition: CalculationDefinition,
    keys: Collection[str],
) -> bool:
    """Drop the source rows ``keys`` name from a source-keyed definition.

    The energy-barrier definitions list their rows as ``configuration
    ["sources"]`` entries with a ``key``, and may hold per-row
    ``row_settings``. Matching sources, their row settings (the whole
    ``row_settings`` entry once it is empty) and their ``job_ids_by_label``
    bindings are dropped. A definition with no source left is deleted.
    Returns whether it was deleted.
    """

    config = definition.parameters.get("configuration", {})
    if isinstance(config, dict):
        raw_sources = config.get("sources", [])
        if isinstance(raw_sources, list):
            config["sources"] = [
                source
                for source in raw_sources
                if not isinstance(source, dict) or str(source.get("key", "")).strip() not in keys
            ]
        row_settings = config.get("row_settings", {})
        if isinstance(row_settings, dict):
            for key in keys:
                row_settings.pop(key, None)
            if not row_settings:
                config.pop("row_settings", None)
    mapping = definition.parameters.get(LABEL_MAPPING, {})
    if isinstance(mapping, dict):
        for key in keys:
            mapping.pop(key, None)

    config = definition.parameters.get("configuration", {})
    if not isinstance(config, dict):
        return False
    raw_sources = config.get("sources", [])
    if isinstance(raw_sources, list) and raw_sources:
        return False
    document.calculations.pop(definition.id, None)
    return True


def remove_generated_sources(document: ProjectDocument, keys: Collection[str]) -> None:
    """Drop the generated paths ``keys`` name from the Energy Barriers form state."""

    state = document.ui_state.get(ENERGY_BARRIERS_FORM_STATE_KEY, {})
    if not isinstance(state, dict):
        return
    raw_sources = state.get("generated_sources", [])
    if isinstance(raw_sources, list):
        state["generated_sources"] = [
            source
            for source in raw_sources
            if not isinstance(source, dict) or str(source.get("key", "")).strip() not in keys
        ]


__all__ = [
    "LABEL_MAPPING",
    "convergence_mapping_job_priority",
    "definition_workflow_context_matches",
    "ensure_job_mapping",
    "matching_sweep_definition",
    "relaxation_mapping_job_priority",
    "remove_cutoff_from_definition",
    "remove_definition_if_no_rows",
    "remove_generated_sources",
    "remove_grid_keys_from_definition",
    "remove_labels_from_definition",
    "remove_source_keys_from_definition",
    "sweep_definition_context_matches",
    "sweep_definition_has_context",
    "sync_equivalent_mappings",
]
