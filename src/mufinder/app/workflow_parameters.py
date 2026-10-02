"""Shared helpers for workflow-specific parameter dialog surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from mufinder.castep_io.gui_metadata import get_gui_keyword_specs_for_files
from mufinder.castep_io.schema import KeywordSpec, get_keyword_spec


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


def _clone_spec_for_workflow_surface(
    spec: KeywordSpec,
    *,
    workflow_stage: str,
    basic: bool,
    advanced: bool,
) -> KeywordSpec:
    basic_workflows = spec.basic_workflows
    advanced_workflows = spec.advanced_workflows

    if basic:
        basic_workflows = _append_workflow_stage(basic_workflows, workflow_stage)
    if advanced:
        advanced_workflows = _append_workflow_stage(advanced_workflows, workflow_stage)

    return replace(
        spec,
        basic_workflows=basic_workflows,
        advanced_workflows=advanced_workflows,
    )


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
            _clone_spec_for_workflow_surface(
                spec,
                workflow_stage=workflow_stage,
                basic=name in basic_names,
                advanced=True,
            )
        )
        seen.add(name)
    return reorder_workflow_parameter_specs(specs, ordered_names=basic_names)