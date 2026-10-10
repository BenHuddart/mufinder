"""Shared helpers for workflow-specific parameter dialog surfaces."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import replace
from typing import Any

from mufinder.castep_io.gui_metadata import (
    get_gui_keyword_specs_for_files,
    infer_task,
    keyword_visible_in_workflow,
    normalize_gui_value,
)
from mufinder.castep_io.schema import KeywordSpec, get_keyword_spec
from mufinder.castep_io.types import BlockSection, ValueWithUnit

# Cell keywords the workflow derives from the row's structure; a stored value
# for one of them is never offered for editing or carried into a row.
STRUCTURE_CONTROLLED_CELL_KEYWORDS = frozenset(
    {"lattice_cart", "lattice_abc", "positions_frac", "positions_abs"}
)

# Inside a mutually exclusive group, the default that survives is the first of
# these names present; otherwise the alphabetically first name.
_PREFERRED_DEFAULT_PARAMETER_NAMES = (
    "cut_off_energy",
    "kpoint_mp_grid",
    "kpoint_mp_spacing",
    "kpoints_list",
)


def kpoint_exclusive_hint(
    effective_params_sequence: Iterable[Mapping[str, object]],
) -> str | None:
    """Return the preferred_exclusive hint for the kpoint grid/spacing pair.

    Returns ``"kpoint_mp_spacing"`` when every dict in the sequence contains
    ``kpoint_mp_spacing``, indicating the user's active kpoint choice is spacing
    across all rows/cutoffs.  Returns ``None`` for grid mode, mixed rows, or an
    empty sequence.

    Pass the effective (pre-prefill) params for each row so the hint reflects
    the user's stored choice rather than the pre-filled inactive half added for
    UX toggle convenience.
    """
    params = list(effective_params_sequence)
    if not params:
        return None
    if all("kpoint_mp_spacing" in p for p in params):
        return "kpoint_mp_spacing"
    return None


def mixed_parameter_names(
    effective_params_sequence: Iterable[Mapping[str, object]],
) -> set[str]:
    """Return the parameters whose value differs between rows edited together.

    A parameter is mixed when the rows do not all hold the same value for it,
    which includes rows where it is absent. ``task`` is never reported: the
    parameter dialog handles it separately.
    """

    params = [dict(p) for p in effective_params_sequence]
    if len(params) < 2:
        return set()
    names = set().union(*params) - {"task"}
    missing = object()
    return {
        name
        for name in names
        if any(p.get(name, missing) != params[0].get(name, missing) for p in params[1:])
    }


def merge_batch_values(
    collected: Mapping[str, object],
    row_values: Mapping[str, object],
    untouched_mixed: Iterable[str],
) -> dict[str, object]:
    """Return one row's parameters after a batch edit.

    ``collected`` is what the dialog returned, which leaves out the mixed
    parameters the user did not touch; for those the row keeps the value it had
    (or stays without one). Everything else comes from the dialog, so a
    parameter the user edited or removed applies to every row.
    """

    merged = dict(collected)
    for name in untouched_mixed:
        if name in row_values:
            merged[name] = row_values[name]
        else:
            merged.pop(name, None)
    return merged


def merge_batch_advanced(
    dialog_advanced: Iterable[str],
    row_advanced: Iterable[str],
    untouched_mixed: Iterable[str],
) -> set[str]:
    """Return one row's visible advanced parameters after a batch edit.

    An advanced parameter that is mixed and untouched stays shown only on the
    rows that already showed it; the rest follow the dialog.
    """

    untouched = set(untouched_mixed)
    return (set(dialog_advanced) - untouched) | (set(row_advanced) & untouched)


COMMON_WORKFLOW_PARAMETER_NAMES = (
    "basis_precision",
    "cut_off_energy",
    "elec_energy_tol",
    "kpoint_mp_grid",
    "kpoint_mp_spacing",
)


def seed_xc_functional_default(
    default_params: dict[str, Any],
    xc_functional: str | None,
    *,
    schema_default: object | None,
) -> None:
    """Make the project default XC functional a workflow widget's row default.

    ``default_params`` is the widget's shared row-default mapping, updated in
    place. With no project functional the row default falls back to
    ``schema_default`` (the value the widget started with), so clearing the
    project default restores the pre-project behaviour.
    """

    if xc_functional:
        default_params["xc_functional"] = xc_functional
    elif schema_default is not None:
        default_params["xc_functional"] = schema_default
    else:
        default_params.pop("xc_functional", None)


def _append_workflow_stage(values: tuple[str, ...], workflow_stage: str) -> tuple[str, ...]:
    if workflow_stage in values:
        return values
    return (*values, workflow_stage)


def clone_spec_for_workflow(
    spec: KeywordSpec,
    *,
    workflow_stage: str,
    basic: bool,
    advanced: bool,
    default_overrides: Mapping[str, object] | None = None,
) -> KeywordSpec:
    """Return ``spec`` made visible in ``workflow_stage``'s parameter dialog.

    ``basic`` and ``advanced`` add the stage to the spec's basic and advanced
    workflows. ``default_overrides`` replaces the schema default for the spec
    named in it (a workflow-specific starting value).
    """

    basic_workflows = spec.basic_workflows
    advanced_workflows = spec.advanced_workflows

    if basic:
        basic_workflows = _append_workflow_stage(basic_workflows, workflow_stage)
    if advanced:
        advanced_workflows = _append_workflow_stage(advanced_workflows, workflow_stage)

    default_value = spec.default_value
    if default_overrides is not None and spec.name in default_overrides:
        default_value = default_overrides[spec.name]

    return replace(
        spec,
        basic_workflows=basic_workflows,
        advanced_workflows=advanced_workflows,
        default_value=default_value,
    )


def spec_is_blocked_by_locked(spec: KeywordSpec, locked_names: Collection[str]) -> bool:
    """Return whether ``spec`` is mutually exclusive with a locked parameter."""

    if spec.name in locked_names:
        return False
    return any(conflict in locked_names for conflict in spec.mutually_exclusive_with)


def workflow_default_params(
    specs: Iterable[KeywordSpec],
    *,
    workflow_stage: str,
    locked_names: Collection[str] = (),
    skip_names: Collection[str] = (),
) -> dict[str, Any]:
    """Return the schema defaults a workflow's rows start from.

    The mapping holds the stage's ``task`` and the normalised schema default of
    every spec with one that is basic in ``workflow_stage``, except the locked
    names, the specs mutually exclusive with a locked name, and ``skip_names``
    (parameters the widget seeds from the project defaults instead).
    """

    defaults: dict[str, Any] = {"task": infer_task(workflow_stage)}
    for spec in specs:
        if spec.default_value is None or spec.name in skip_names:
            continue
        if not keyword_visible_in_workflow(spec, workflow_stage=workflow_stage, include_advanced=False):
            continue
        if spec.name in locked_names or spec_is_blocked_by_locked(spec, locked_names):
            continue
        defaults[spec.name] = normalize_gui_value(spec, spec.default_value)
    return defaults


def _preferred_default_name(left: str, right: str) -> str:
    pair = {left, right}
    for preferred in _PREFERRED_DEFAULT_PARAMETER_NAMES:
        if preferred in pair:
            return preferred
    return min(left, right)


def drop_default_conflicts(
    values: dict[str, Any],
    specs_by_name: Mapping[str, KeywordSpec],
    *,
    locked_names: Collection[str] | None = None,
) -> dict[str, Any]:
    """Remove conflicting defaults from ``values`` in place and return it.

    A value mutually exclusive with a locked name is dropped. Within each
    remaining mutually exclusive pair one value survives, chosen
    deterministically (see ``_PREFERRED_DEFAULT_PARAMETER_NAMES``); a locked name
    always survives.
    """

    locked = set(locked_names or ())
    for name in tuple(values):
        spec = specs_by_name.get(name)
        if spec is not None and spec_is_blocked_by_locked(spec, locked):
            values.pop(name, None)

    changed = True
    while changed:
        changed = False
        for name in tuple(values):
            if name in locked:
                continue
            spec = specs_by_name.get(name)
            if spec is None or name not in values:
                continue
            for conflict in spec.mutually_exclusive_with:
                if conflict not in values:
                    continue
                keep = _preferred_default_name(name, conflict)
                drop = conflict if keep == name else name
                if drop in locked:
                    drop = name
                if drop in values:
                    values.pop(drop, None)
                    changed = True
                    break
            if changed:
                break
    return values


def restore_structured_value(
    specs_by_name: Mapping[str, KeywordSpec] | None,
    name: str,
    value: object,
) -> object:
    """Return a persisted parameter value in the structured form its spec takes.

    JSON storage flattens block values, values with units and vectors into
    dicts and lists; this rebuilds ``BlockSection``, ``ValueWithUnit`` and
    tuples and normalises the result. The spec comes from ``specs_by_name``, or
    from the CASTEP keyword catalog (``param`` before ``cell``) when that is
    ``None``. A name with no spec is returned unchanged, and a value its spec
    rejects is returned without normalising.
    """

    if specs_by_name is None:
        spec = get_keyword_spec("param", name) or get_keyword_spec("cell", name)
    else:
        spec = specs_by_name.get(name)
    if spec is None:
        return value

    if isinstance(value, dict):
        if spec.kind == "block" and "rows" in value:
            return BlockSection(
                name=str(value.get("name", spec.name)),
                rows=tuple(tuple(row) for row in value.get("rows", ())),
                unit=str(value["unit"]) if value.get("unit") is not None else None,
            )
        if spec.value_type in {"physical", "positive_float_with_optional_unit"} and {"value", "unit"} <= value.keys():
            return ValueWithUnit(value=value["value"], unit=str(value["unit"]))

    if isinstance(value, list) and spec.value_type in {"int_triplet", "real_vector"}:
        value = tuple(value)

    try:
        return normalize_gui_value(spec, value)
    except (TypeError, ValueError):
        return value


def sanitize_parameter_mapping(
    mapping: Mapping[str, object],
    specs_by_name: Mapping[str, KeywordSpec],
    *,
    keep_unknown: bool = False,
    keep: Collection[str] = (),
    drop: Collection[str] = (),
) -> dict[str, object]:
    """Return a stored parameter mapping restored to its structured values.

    Names with no spec in ``specs_by_name`` are dropped, because a row the user
    can still edit must not carry a value no editor can show. ``keep`` names
    survive without a spec (a widget's ``task``), and ``drop`` names are removed
    even when they have a spec.

    ``keep_unknown`` passes every unknown name through unchanged, for imported
    rows. The keyword catalog is discovered from the environment, so the same
    project can see a thinner one on another machine, and a name dropped from
    the widget's state while the definition row keeps it desynchronises the two,
    which drops a completed import to a "ready" row with no energy. The
    passed-through values are inert: imported rows are never edited, expanded
    or submitted.
    """

    cleaned: dict[str, object] = {}
    for name, value in mapping.items():
        if name in drop:
            continue
        if keep_unknown or name in specs_by_name or name in keep:
            cleaned[name] = restore_structured_value(specs_by_name, name, value)
    return cleaned


def workflow_basic_parameter_names(*workflow_specific_names: str) -> tuple[str, ...]:
    """Return the shared common workflow rows followed by workflow-specific rows.

    Project policy: workflow dialogs present common parameters first, in a
    stable order, so users can find them in the same place across workflows.
    Workflow-specific rows follow after the common prefix.
    """

    ordered: list[str] = []
    seen: set[str] = set()
    for name in (*COMMON_WORKFLOW_PARAMETER_NAMES, *workflow_specific_names):
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def reorder_workflow_parameter_specs(
    specs: list[KeywordSpec],
    *,
    ordered_names: tuple[str, ...] = COMMON_WORKFLOW_PARAMETER_NAMES,
) -> list[KeywordSpec]:
    """Return specs ordered by a shared workflow prefix and then original order."""

    ordered_specs: list[KeywordSpec] = []
    seen: set[str] = set()

    for name in ordered_names:
        for spec in specs:
            if spec.name != name or spec.name in seen:
                continue
            ordered_specs.append(spec)
            seen.add(spec.name)
            break

    for spec in specs:
        if spec.name in seen:
            continue
        ordered_specs.append(spec)
        seen.add(spec.name)

    return ordered_specs


def build_workflow_parameter_surface(
    *,
    workflow_stage: str,
    basic_names: tuple[str, ...],
    files: tuple[str, ...] = ("param", "cell"),
    excluded_names: tuple[str, ...] = ("task",),
) -> list[KeywordSpec]:
    """Return dialog specs with common-first basic rows and searchable advanced rows."""

    visible_specs = get_gui_keyword_specs_for_files(
        files=files,
        workflow_stage=workflow_stage,
        include_advanced=True,
    )

    specs_by_name: dict[str, KeywordSpec] = {}
    for spec in visible_specs:
        if spec.name in excluded_names or spec.name in specs_by_name:
            continue
        specs_by_name[spec.name] = spec

    for name in basic_names:
        if name in excluded_names or name in specs_by_name:
            continue
        spec = get_keyword_spec("param", name) or get_keyword_spec("cell", name)
        if spec is None:
            continue
        specs_by_name[name] = spec

    specs: list[KeywordSpec] = []
    seen: set[str] = set()
    for name in (*basic_names, *specs_by_name):
        if name in excluded_names or name in seen:
            continue
        spec = specs_by_name.get(name)
        if spec is None:
            continue
        specs.append(
            clone_spec_for_workflow(
                spec,
                workflow_stage=workflow_stage,
                basic=name in basic_names,
                advanced=True,
            )
        )
        seen.add(name)
    return reorder_workflow_parameter_specs(specs, ordered_names=basic_names)