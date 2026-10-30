"""GUI-focused helpers for workflow-aware CASTEP keyword presentation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .schema import (
    FileKind,
    KeywordSpec,
    canonical_keyword_name,
    keyword_sort_key,
    keyword_specs,
)
from .types import BlockSection, KeywordAssignment, ValueWithUnit
from .validators import CastepValidationError, validate_keyword_value


WORKFLOW_TASK_MAP: dict[str, str] = {
    "convergence_cutoff": "SINGLEPOINT",
    "convergence_kpoint": "SINGLEPOINT",
    "pristine_relaxation": "GEOMETRYOPTIMIZATION",
    "uep": "SINGLEPOINT",
    "muon_generation": "SINGLEPOINT",
    "muon_relaxation": "GEOMETRYOPTIMIZATION",
    "site_analysis": "SINGLEPOINT",
    "magnetic_field": "SINGLEPOINT",
    "hyperfine_coupling": "MAGRES",
    "quantum_corrections": "SINGLEPOINT",
    "energy_barriers": "TRANSITIONSTATESEARCH",
}


def _workflow_matches(workflow_stage: str, configured: tuple[str, ...]) -> bool:
    stage = workflow_stage.strip().lower()
    normalized = {item.strip().lower() for item in configured}
    return "all" in normalized or stage in normalized


def keyword_visible_in_workflow(
    spec: KeywordSpec,
    *,
    workflow_stage: str,
    include_advanced: bool,
) -> bool:
    """Return whether a keyword should be shown in the requested workflow context."""

    if _workflow_matches(workflow_stage, spec.basic_workflows):
        return True
    if include_advanced and _workflow_matches(workflow_stage, spec.advanced_workflows):
        return True
    return False


#: Group headings for keyword categories, in sentence case. A category that is
#: not listed is shown with its underscores as spaces and a capital first letter.
KEYWORD_CATEGORY_LABELS: dict[str, str] = {
    "constraints": "Constraints",
    "electronic": "Electronic",
    "elnes": "ELNES",
    "general": "General",
    "geometry": "Geometry optimisation",
    "io": "Input and output",
    "kpoints": "K-points",
    "magres": "Magnetic resonance",
    "molecular_dynamics": "Molecular dynamics",
    "optics": "Optics",
    "phonon": "Phonons",
    "pseudopotentials": "Species and pseudopotentials",
    "scf": "SCF",
    "spectral": "Spectral",
    "structure": "Structure",
    "symmetry": "Symmetry",
    "tddft": "TDDFT",
    "tssearch": "Transition-state search",
    "wannier": "Wannier functions",
    "workflow": "Workflow",
}


def keyword_category_label(category: str | None) -> str:
    """Return the heading a keyword category is grouped under in the parameter picker."""

    key = (category or "general").strip().lower() or "general"
    if key in KEYWORD_CATEGORY_LABELS:
        return KEYWORD_CATEGORY_LABELS[key]
    text = key.replace("_", " ")
    return text[:1].upper() + text[1:]


def exclusive_partner_labels(spec: KeywordSpec) -> tuple[str, ...]:
    """Return the labels of the keywords that cannot be set together with ``spec``.

    The partners are the keyword's ``mutually_exclusive_with`` entries, then the
    keywords whose own list names it, then the other members of its
    ``exclusive_group``, looked up in the active schema for the label; a partner
    the schema does not list is shown by its name.
    """

    schema = keyword_specs(spec.file)
    names: list[str] = [
        canonical_keyword_name(spec.file, str(item)) for item in spec.mutually_exclusive_with
    ]
    names.extend(
        name
        for name, other in schema.items()
        if spec.name
        in {canonical_keyword_name(spec.file, str(item)) for item in other.mutually_exclusive_with}
    )
    if spec.exclusive_group:
        names.extend(
            name for name, other in schema.items() if other.exclusive_group == spec.exclusive_group
        )
    labels: list[str] = []
    for name in dict.fromkeys(names):
        if not name or name == spec.name:
            continue
        other = schema.get(name)
        label = (other.label if other is not None else None) or name.upper()
        labels.append(label)
    return tuple(labels)


def keyword_visible_for_task(spec: KeywordSpec, *, task: str) -> bool:
    """Return whether a keyword should be shown for the effective CASTEP task."""

    if not spec.allowed_tasks:
        return True
    return task.strip().upper() in {item.strip().upper() for item in spec.allowed_tasks}


def keyword_locked_in_workflow(spec: KeywordSpec, *, workflow_stage: str) -> bool:
    """Return whether the workflow controls this keyword outside user editing."""

    return _workflow_matches(workflow_stage, spec.locked_in_workflows)


def infer_task(workflow_stage: str) -> str:
    """Infer CASTEP TASK from the workflow stage."""

    normalized = workflow_stage.strip().lower()
    return WORKFLOW_TASK_MAP.get(normalized, "SINGLEPOINT")


def get_gui_keyword_specs(
    *,
    file: FileKind,
    workflow_stage: str,
    include_advanced: bool = False,
) -> list[KeywordSpec]:
    """Return keyword specs visible for a given workflow and mode."""

    schema = keyword_specs(file)
    visible_names = [
        name
        for name, spec in schema.items()
        if keyword_visible_in_workflow(
            spec,
            workflow_stage=workflow_stage,
            include_advanced=include_advanced,
        )
    ]

    visible_names.sort(key=lambda name: keyword_sort_key(file, name))
    return [schema[name] for name in visible_names]


def get_gui_keyword_specs_for_files(
    *,
    files: Iterable[FileKind],
    workflow_stage: str,
    include_advanced: bool = False,
) -> list[KeywordSpec]:
    """Return visible specs across multiple CASTEP file targets."""

    specs: list[KeywordSpec] = []
    for file in files:
        specs.extend(
            get_gui_keyword_specs(
                file=file,
                workflow_stage=workflow_stage,
                include_advanced=include_advanced,
            )
        )
    return specs


def normalize_gui_value(spec: KeywordSpec, value: object) -> object:
    """Normalize a GUI value to the structured CASTEP runtime value shape."""

    if spec.kind == "block":
        if isinstance(value, BlockSection):
            canonical_name = canonical_keyword_name(spec.file, spec.name)
            if value.name == canonical_name:
                return value
            return BlockSection(name=canonical_name, rows=value.rows, unit=value.unit)
        raise ValueError(f"{spec.name} requires a BlockSection value.")

    if spec.value_type in {"physical", "positive_float_with_optional_unit"}:
        if isinstance(value, ValueWithUnit):
            return value
        if spec.unit is not None:
            return ValueWithUnit(float(value), spec.unit)
        return float(value)

    if spec.value_type == "int_triplet":
        parts = tuple(int(component) for component in value)  # type: ignore[arg-type]
        expected = spec.vector_length or 3
        if len(parts) != expected:
            raise ValueError(f"{spec.name} requires exactly {expected} integer components.")
        return parts

    if spec.value_type in {"int", "positive_int", "non_negative_int"}:
        return int(value)

    if spec.value_type in {"float", "number"}:
        return float(value)

    if spec.value_type == "real_vector":
        parts = tuple(float(component) for component in value)  # type: ignore[arg-type]
        expected = spec.vector_length or 3
        if len(parts) != expected:
            raise ValueError(f"{spec.name} requires exactly {expected} numeric components.")
        return parts

    if spec.value_type == "bool":
        return bool(value)

    if spec.value_type == "enum":
        if spec.option_type == "Integer":
            return int(value)
        if spec.option_type in {"Real", "Physical"}:
            return float(value)
        return str(value).strip().upper()

    return value


def make_keyword_assignment(spec: KeywordSpec, value: object) -> KeywordAssignment | None:
    """Build a normalized keyword assignment from a GUI-sourced value."""

    canonical_name = canonical_keyword_name(spec.file, spec.name)

    if spec.kind == "block":
        raise ValueError(f"{spec.name} is a block keyword and cannot become a scalar assignment.")

    if spec.kind == "flag":
        return KeywordAssignment(name=canonical_name, value=None) if bool(value) else None

    normalized_value = normalize_gui_value(spec, value)
    return KeywordAssignment(name=canonical_name, value=normalized_value)


def make_block_section(spec: KeywordSpec, value: object) -> BlockSection | None:
    """Build a normalized block payload from a GUI-sourced value."""

    if spec.kind != "block":
        raise ValueError(f"{spec.name} is not a block keyword.")
    normalized = normalize_gui_value(spec, value)
    if not isinstance(normalized, BlockSection):
        raise ValueError(f"{spec.name} block normalization did not return a BlockSection.")
    return normalized


def gui_value_is_active(spec: KeywordSpec, value: object) -> bool:
    """Return whether a GUI value should participate in constraints and output."""

    if value in (None, ""):
        return False
    if spec.kind == "flag":
        return bool(value)
    if spec.value_type == "bool":
        return bool(value)
    return True


def validate_gui_values(
    values: Mapping[str, object],
    specs_by_name: Mapping[str, KeywordSpec],
    *,
    task: str,
) -> list[str]:
    """Validate a dictionary of GUI-collected keyword values."""

    errors: list[str] = []
    normalized_task = task.strip().upper()
    active_names: set[str] = set()

    for name, raw_value in values.items():
        if name == "task":
            continue
        spec = specs_by_name.get(name)
        if spec is None:
            continue
        if gui_value_is_active(spec, raw_value):
            active_names.add(name)

        if spec.kind == "flag" and not bool(raw_value):
            continue

        try:
            normalized_value = normalize_gui_value(spec, raw_value)
            validate_keyword_value(spec.file, spec.name, normalized_value)
        except (TypeError, ValueError, CastepValidationError) as exc:
            errors.append(str(exc))
            continue

        if spec.allowed_tasks and normalized_task not in {
            item.upper() for item in spec.allowed_tasks
        }:
            joined = ", ".join(spec.allowed_tasks)
            tasks = "task" if len(spec.allowed_tasks) == 1 else "tasks"
            errors.append(f"{spec.name} is only available for {tasks}: {joined}.")

    for name in sorted(active_names):
        spec = specs_by_name[name]
        for conflicting_name in spec.mutually_exclusive_with:
            if conflicting_name in active_names and name < conflicting_name:
                errors.append(f"{spec.name} conflicts with {conflicting_name}.")
        for required in spec.requires:
            if required not in active_names:
                errors.append(f"{spec.name} requires {required}.")
        if spec.requires_any_of and not any(
            required in active_names for required in spec.requires_any_of
        ):
            joined = " or ".join(spec.requires_any_of)
            errors.append(f"{spec.name} requires {joined}.")

    return errors


def default_assignments(specs: Iterable[KeywordSpec]) -> tuple[KeywordAssignment, ...]:
    """Return normalized default assignments for specs that declare defaults."""

    assignments: list[KeywordAssignment] = []
    for spec in specs:
        if spec.default_value is None:
            continue
        assignment = make_keyword_assignment(spec, spec.default_value)
        if assignment is not None:
            assignments.append(assignment)
    return tuple(assignments)
