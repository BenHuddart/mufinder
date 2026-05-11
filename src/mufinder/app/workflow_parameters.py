"""Shared helpers for workflow-specific parameter dialog surfaces."""

from __future__ import annotations

from dataclasses import replace

from mufinder.castep_io.gui_metadata import get_gui_keyword_specs_for_files
from mufinder.castep_io.schema import KeywordSpec, get_keyword_spec


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


def build_workflow_parameter_surface(
    *,
    workflow_stage: str,
    basic_names: tuple[str, ...],
    files: tuple[str, ...] = ("param", "cell"),
    excluded_names: tuple[str, ...] = ("task",),
) -> list[KeywordSpec]:
    """Return dialog specs with curated basic rows and searchable advanced rows."""

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
    return specs