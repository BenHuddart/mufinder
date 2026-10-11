"""Translation of the bundled starter keyword catalog into runtime keyword specs.

When no ``castep_keywords.json`` can be loaded, ``mufinder.castep_io.schema``
falls back to the starter catalog (``docs/castep/keyword_catalog.yaml``, shipped
as ``mufinder/assets/keyword_catalog.yaml``). This module reads that YAML,
resolves aliases and requirement strings, merges ``catalog_overrides`` and
builds the ``KeywordSpec`` mappings; ``schema`` decides when to use them and
records the starter catalog as the active source.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any

import yaml

from .catalog_overrides import ALL_WORKFLOWS, BLOCK_SCHEMAS, KEYWORD_OVERRIDES, MANUAL_ALIAS_MAP
from .schema import (
    FileKind,
    KeywordKind,
    KeywordSpec,
    _build_block_schema,
    _first_sentence,
    _humanize,
    _infer_allowed_tasks,
    _infer_category,
    _override_canonical_names,
    _override_spec,
)


_STARTER_SOURCE_ORDER_OVERRIDES: dict[FileKind, tuple[str, ...]] = {
    "cell": (
        "lattice_cart",
        "lattice_abc",
        "positions_frac",
        "positions_abs",
        "symmetry_generate",
        "fix_all_cell",
        "species_pot",
        "kpoints_list",
        "kpoint_mp_grid",
        "kpoint_mp_spacing",
        "kpoint_mp_offset",
        "unit_cell",
    ),
    "param": (
        "task",
        "write_formatted_density",
        "calculate_polarization",
        "write_cell_structure",
        "continuation",
        "xc_functional",
        "basis_precision",
        "cut_off_energy",
        "spin_polarized",
        "spin",
        "smearing_width",
        "elec_energy_tol",
        "max_scf_cycles",
        "geom_max_iter",
        "geom_energy_tol",
        "geom_force_tol",
        "geom_stress_tol",
        "geom_disp_tol",
        "geom_frequency_est",
        "md_delta_t",
        "md_num_iter",
        "md_temperature",
    ),
}


def _starter_resolve_name(file: FileKind, name: str, alias_map: dict[str, str]) -> str:
    current = name.strip().lower()
    seen: set[str] = set()
    while current in alias_map and current not in seen:
        seen.add(current)
        current = alias_map[current].strip().lower()
    return current


def _starter_option_type(kind: KeywordKind, value_type: str) -> str:
    if kind == "block":
        return "Block"
    if kind == "flag":
        return "Defined"
    if value_type == "bool":
        return "Boolean (Logical)"
    if value_type == "int_triplet":
        return "Integer Vector"
    if value_type == "real_vector":
        return "Real Vector"
    if value_type in {"int", "positive_int", "non_negative_int"}:
        return "Integer"
    if value_type in {"float", "number"}:
        return "Real"
    if value_type in {"physical", "positive_float_with_optional_unit"}:
        return "Physical"
    return "String"


def _starter_kind_value_type(entry: dict[str, Any]) -> tuple[KeywordKind, str]:
    raw_kind = str(entry.get("kind", "scalar")).strip().lower()
    raw_value_type = str(entry.get("value_type", "text")).strip().lower()
    if raw_kind == "block":
        return ("block", "block")
    if raw_kind == "flag" or raw_value_type == "flag":
        return ("flag", "bool")
    return ("scalar", raw_value_type or "text")


def _starter_sequence(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple)):
        return tuple(str(item).strip() for item in values if str(item).strip())
    return (str(values).strip(),) if str(values).strip() else ()


def _starter_parse_requirements(
    file: FileKind,
    values: object,
    *,
    alias_map: dict[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    requires: list[str] = []
    requires_any_of: list[str] = []
    allowed_tasks: list[str] = []

    for raw in _starter_sequence(values):
        normalized = raw.strip()
        if not normalized:
            continue
        if normalized.upper().startswith("TASK="):
            allowed_tasks.append(normalized.split("=", maxsplit=1)[1].strip().upper())
            continue
        if " or " in normalized.lower():
            for part in re.split(r"\s+or\s+", normalized, flags=re.IGNORECASE):
                canonical = _starter_resolve_name(file, part.strip().lower(), alias_map)
                if canonical:
                    requires_any_of.append(canonical)
            continue
        canonical = _starter_resolve_name(file, normalized.lower(), alias_map)
        if canonical:
            requires.append(canonical)

    return (
        tuple(dict.fromkeys(requires)),
        tuple(dict.fromkeys(requires_any_of)),
        tuple(dict.fromkeys(allowed_tasks)),
    )


def _starter_default_value(spec_value_type: str, value: object) -> object | None:
    if value is None:
        return None
    if spec_value_type in {"int_triplet", "real_vector"} and isinstance(value, list):
        return tuple(value)
    if spec_value_type in {"int", "positive_int", "non_negative_int"} and value != "":
        return int(value)
    if spec_value_type in {"float", "number", "physical", "positive_float_with_optional_unit"} and value != "":
        return float(value)
    if spec_value_type == "bool":
        return bool(value)
    return value


def _starter_source_order(file: FileKind, name: str, fallback_order: int) -> int:
    overrides = {item: index for index, item in enumerate(_STARTER_SOURCE_ORDER_OVERRIDES[file])}
    if name in overrides:
        return overrides[name]
    return len(overrides) + fallback_order


def load_starter_catalog_entries(catalog_path: Path) -> dict[FileKind, list[dict[str, Any]]]:
    """Read the starter catalog YAML into per-file keyword entries in source order."""

    raw_text = catalog_path.read_text(encoding="utf-8")
    normalized_lines: list[str] = []
    quoted_keys = {
        "name",
        "canonical_name",
        "file",
        "kind",
        "value_type",
        "unit",
        "example",
        "notes",
        "label",
        "description",
        "category",
        "widget_hint",
    }
    for line in raw_text.splitlines():
        match = re.match(r"^(\s*)([A-Za-z_]+):\s*(.+)$", line)
        if not match:
            normalized_lines.append(line)
            continue
        indent, key, raw_value = match.groups()
        if key not in quoted_keys:
            normalized_lines.append(line)
            continue
        stripped = raw_value.strip()
        if not stripped or stripped in {"|", ">", "[]", "{}"} or stripped.startswith(("'", '"', "[", "{")):
            normalized_lines.append(line)
            continue
        escaped = raw_value.replace("'", "''")
        normalized_lines.append(f"{indent}{key}: '{escaped}'")
    payload = yaml.safe_load("\n".join(normalized_lines) + "\n") or {}
    keywords = payload.get("keywords", ())
    if not isinstance(keywords, list):
        raise TypeError(f"Expected 'keywords' list in starter catalog {catalog_path}.")

    entries: dict[FileKind, list[dict[str, Any]]] = {"cell": [], "param": []}
    per_file_order: dict[FileKind, int] = {"cell": 0, "param": 0}

    for raw_entry in keywords:
        if not isinstance(raw_entry, dict):
            continue
        file = str(raw_entry.get("file", "")).strip().lower()
        if file not in {"cell", "param"}:
            continue
        source_name = str(raw_entry.get("name", "")).strip().lower()
        if not source_name:
            continue
        canonical_name = str(raw_entry.get("canonical_name") or source_name).strip().lower()
        entries[file].append(
            {
                "source_name": source_name,
                "canonical_name": canonical_name,
                "payload": raw_entry,
                "source_order": per_file_order[file],
            }
        )
        per_file_order[file] += 1

    return entries


def _starter_alias_map(file: FileKind, entries: list[dict[str, Any]]) -> dict[str, str]:
    alias_map = dict(MANUAL_ALIAS_MAP.get(file, {}))
    names = {
        str(entry["source_name"]).strip().lower()
        for entry in entries
    } | {
        str(entry["canonical_name"]).strip().lower()
        for entry in entries
    }
    for entry in entries:
        source_name = str(entry["source_name"]).strip().lower()
        canonical_name = str(entry["canonical_name"]).strip().lower()
        if source_name != canonical_name:
            alias_map[source_name] = canonical_name
    for name in sorted(names):
        if name in alias_map:
            continue
        if "isation" in name:
            target = name.replace("isation", "ization")
            if target in names:
                alias_map[name] = target
                continue
        if "polarised" in name:
            target = name.replace("polarised", "polarized")
            if target in names:
                alias_map[name] = target
    return alias_map


def _starter_spec_from_catalog(
    *,
    file: FileKind,
    name: str,
    payload: dict[str, Any],
    source_order: int,
    aliases: tuple[str, ...],
    alias_map: dict[str, str],
) -> KeywordSpec:
    override = dict(KEYWORD_OVERRIDES.get(file, {}).get(name, {}))
    gui = dict(payload.get("gui") or {})
    kind, value_type = _starter_kind_value_type(payload)
    option_type = _starter_option_type(kind, value_type)

    requires, requires_any_of, allowed_tasks = _starter_parse_requirements(
        file,
        payload.get("requires"),
        alias_map=alias_map,
    )
    if not allowed_tasks:
        allowed_tasks = tuple(str(item) for item in override.get("allowed_tasks", _infer_allowed_tasks(name)))

    block_schema = None
    if kind == "block":
        override_block_schema = override.get("block_schema")
        if isinstance(override_block_schema, dict):
            block_schema = _build_block_schema(override_block_schema)
        else:
            block_schema = _build_block_schema(BLOCK_SCHEMAS.get(file, {}).get(name))

    notes = str(payload.get("notes", "") or override.get("notes", "") or "").strip() or None
    description = str(gui.get("label") or payload.get("description") or override.get("description") or _humanize(name))
    help_text = str(notes or override.get("help_text") or description)
    default_value = _starter_default_value(value_type, gui.get("default_value", override.get("default_value")))

    mutually_exclusive_with: list[str] = []
    for item in payload.get("mutually_exclusive_with", ()) or override.get("mutually_exclusive_with", ()):
        raw_other = str(item).strip().lower()
        if not raw_other:
            continue
        resolved_other = _starter_resolve_name(file, raw_other, alias_map)
        if resolved_other == name:
            continue
        mutually_exclusive_with.append(resolved_other)

    return KeywordSpec(
        name=name,
        file=file,
        kind=kind,
        value_type=value_type,
        option_type=option_type,
        level="",
        docstring=help_text,
        unit=str(payload["unit"]) if payload.get("unit") is not None else str(override["unit"]) if override.get("unit") is not None else None,
        allowed_values=tuple(str(item) for item in payload.get("allowed_values", ()) or override.get("allowed_values", ())),
        mutually_exclusive_with=tuple(dict.fromkeys(mutually_exclusive_with)),
        requires=requires,
        requires_any_of=requires_any_of,
        repeatable=bool(payload.get("repeatable", override.get("repeatable", False))),
        aliases=aliases,
        allowed_tasks=allowed_tasks,
        example=str(payload["example"]) if payload.get("example") is not None else str(override["example"]) if override.get("example") is not None else None,
        notes=notes,
        label=str(gui.get("label") or override.get("label") or _humanize(name)),
        description=str(payload.get("description") or override.get("description") or _first_sentence(help_text)),
        help_text=help_text,
        category=str(gui.get("category") or override.get("category") or _infer_category(file, name)),
        widget_hint=str(gui["widget_hint"]) if gui.get("widget_hint") is not None else str(override["widget_hint"]) if override.get("widget_hint") is not None else None,
        basic_workflows=tuple(
            str(item)
            for item in (
                gui.get("basic_workflows")
                if gui.get("basic_workflows") is not None
                else override.get("basic_workflows", ALL_WORKFLOWS if name == "task" else ())
            )
        ),
        advanced_workflows=tuple(
            str(item)
            for item in (
                gui.get("advanced_workflows")
                if gui.get("advanced_workflows") is not None
                else override.get("advanced_workflows", ALL_WORKFLOWS)
            )
        ),
        locked_in_workflows=tuple(
            str(item)
            for item in (
                payload.get("locked_in_workflows")
                if payload.get("locked_in_workflows") is not None
                else gui.get("locked_in_workflows")
                if gui.get("locked_in_workflows") is not None
                else override.get("locked_in_workflows", ())
            )
        ),
        default_value=default_value,
        min_value=payload.get("min_value", gui.get("min_value", override.get("min_value"))),
        max_value=payload.get("max_value", gui.get("max_value", override.get("max_value"))),
        step=payload.get("step", gui.get("step", override.get("step"))),
        exclusive_group=str(payload["exclusive_group"]) if payload.get("exclusive_group") is not None else str(gui["exclusive_group"]) if gui.get("exclusive_group") is not None else str(override["exclusive_group"]) if override.get("exclusive_group") is not None else None,
        block_schema=block_schema,
        source_order=source_order,
        vector_length=(
            int(payload["vector_length"])
            if payload.get("vector_length") is not None
            else int(override["vector_length"])
            if override.get("vector_length") is not None
            else (3 if value_type in {"int_triplet", "real_vector"} else None)
        ),
    )


def build_starter_schema(
    catalog_path: Path,
) -> tuple[dict[str, KeywordSpec], dict[str, KeywordSpec], dict[str, tuple[str, ...]]]:
    """Translate the starter catalog at ``catalog_path`` into keyword specs.

    Returns the ``cell`` specs, the ``param`` specs and the source keyword names
    per file, in the shape ``schema._build_schema`` returns for a full catalog.
    """

    schema_by_file: dict[str, dict[str, KeywordSpec]] = {}
    source_names: dict[str, tuple[str, ...]] = {}
    starter_entries = load_starter_catalog_entries(catalog_path)

    for file in ("cell", "param"):
        specs: dict[str, KeywordSpec] = {}
        entries = starter_entries[file]
        alias_map = _starter_alias_map(file, entries)
        entries_by_source = {str(entry["source_name"]).strip().lower(): entry for entry in entries}
        alias_groups: dict[str, list[str]] = defaultdict(list)
        canonical_order: dict[str, int] = {}
        source_name_list: list[str] = []

        for entry in entries:
            source_name = str(entry["source_name"]).strip().lower()
            declared_canonical = str(entry["canonical_name"]).strip().lower()
            resolved_canonical = _starter_resolve_name(file, declared_canonical, alias_map)
            source_name_list.append(source_name)
            if source_name != resolved_canonical:
                alias_groups[resolved_canonical].append(source_name)
            if declared_canonical != resolved_canonical and declared_canonical != source_name:
                alias_groups[resolved_canonical].append(declared_canonical)
            order = _starter_source_order(file, resolved_canonical, int(entry["source_order"]))
            canonical_order[resolved_canonical] = min(order, canonical_order.get(resolved_canonical, order))

        for alias, canonical in MANUAL_ALIAS_MAP.get(file, {}).items():
            resolved_canonical = _starter_resolve_name(file, canonical, alias_map)
            if resolved_canonical in canonical_order:
                alias_groups[resolved_canonical].append(alias)

        for canonical in sorted(canonical_order, key=canonical_order.get):
            base_entry = entries_by_source.get(canonical)
            if base_entry is None:
                base_entry = next(
                    entry
                    for entry in entries
                    if _starter_resolve_name(file, str(entry["canonical_name"]).strip().lower(), alias_map) == canonical
                )
            specs[canonical] = _starter_spec_from_catalog(
                file=file,
                name=canonical,
                payload=dict(base_entry["payload"]),
                source_order=canonical_order[canonical],
                aliases=tuple(sorted(set(alias_groups.get(canonical, ())))),
                alias_map=alias_map,
            )

        extra_names = [name for name in _override_canonical_names(file) if name not in specs]
        for index, name in enumerate(extra_names, start=len(canonical_order)):
            specs[name] = _override_spec(
                file=file,
                name=name,
                source_order=index,
                aliases=tuple(
                    sorted(
                        alias
                        for alias, canonical in MANUAL_ALIAS_MAP.get(file, {}).items()
                        if canonical == name
                    )
                ),
            )
            source_name_list.append(name)

        for alias, canonical in MANUAL_ALIAS_MAP.get(file, {}).items():
            if canonical in specs and alias not in source_name_list:
                source_name_list.append(alias)

        schema_by_file[file] = specs
        source_names[file] = tuple(dict.fromkeys(source_name_list))

    return schema_by_file["cell"], schema_by_file["param"], source_names
