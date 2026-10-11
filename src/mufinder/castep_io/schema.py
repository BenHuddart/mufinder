"""Runtime CASTEP keyword schema built from a generated base catalog plus overrides.

This module discovers and loads ``castep_keywords.json``, infers keyword specs
from it, holds the active schema and exposes the lookup API. When no catalog
can be loaded it falls back to the bundled starter catalog, whose translation
into specs lives in ``mufinder.castep_io.starter_catalog``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
from typing import Any, Literal

from mufinder.assets import starter_keyword_catalog_path

from .catalog_overrides import ALL_WORKFLOWS, BLOCK_SCHEMAS, KEYWORD_OVERRIDES, MANUAL_ALIAS_MAP
from .keyword_catalog_file import KeywordCatalogValidationError, read_keyword_catalog_file


FileKind = Literal["cell", "param"]
KeywordKind = Literal["scalar", "block", "flag"]
ValueType = Literal[
    "enum",
    "bool",
    "int",
    "float",
    "physical",
    "text",
    "int_triplet",
    "real_vector",
    "block",
]

_ENV_JSON_VARS = ("MUFINDER_CASTEP_KEYWORDS_JSON", "CASTEP_KEYWORDS_JSON")
#: Set by ``mufinder.castep_setup`` from the saved CASTEP setup. Kept separate from
#: the user-facing ``*_KEYWORDS_JSON`` variables so that those stay a genuine
#: override: an explicitly exported variable wins over the catalog MuFinder manages.
SETUP_KEYWORDS_JSON_ENV_VAR = "MUFINDER_CASTEP_KEYWORDS_JSON_SETUP"
_SETUP_CATALOG_ORIGIN = "setup"
_ENV_ROOT_VARS = ("MUFINDER_CASTEP_ROOT", "CASTEP_ROOT")
_CATALOG_RELATIVE_PATH = Path("Test/Utilities/ReadTS/castep_keywords.json")
_STARTER_CATALOG_PATH = starter_keyword_catalog_path()
_STARTER_CATALOG_ORIGIN = "bundled_starter_catalog"
_ALLOWED_VALUES_RE = re.compile(
    r"Allowed values:\s*(.+?)(?:Default value\s*:|Modifiable:|NB\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_DEFAULT_VALUE_RE = re.compile(r"Default value\s*:\s*(.+)", re.IGNORECASE)
_EXAMPLE_RE = re.compile(r"Example:\s*(.+?)(?:Default value\s*:|Allowed values:|$)", re.IGNORECASE | re.DOTALL)
_ALIAS_RE = re.compile(r"Default value:\s*Determined from ([A-Z0-9_+]+)", re.IGNORECASE)
_ENUM_TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-*/.]+")
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True, slots=True)
class BlockColumnSpec:
    """One typed column in a structured block editor."""

    name: str
    value_kind: str = "text"
    required: bool = True
    allowed_values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BlockSchema:
    """Editor metadata for a block keyword."""

    family: str = "raw_rows"
    columns: tuple[BlockColumnSpec, ...] = ()
    allow_unit: bool = False
    default_unit: str | None = None
    fixed_row_count: int | None = None
    min_rows: int = 0
    row_group_size: int | None = None
    row_labels: tuple[str, ...] = ()
    summary_mode: str = "row_count"


@dataclass(frozen=True, slots=True)
class KeywordSpec:
    """Runtime metadata for one supported keyword or block."""

    name: str
    file: FileKind
    kind: KeywordKind
    value_type: ValueType | str
    option_type: str = ""
    level: str = ""
    docstring: str = ""
    unit: str | None = None
    allowed_values: tuple[str, ...] = ()
    mutually_exclusive_with: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    requires_any_of: tuple[str, ...] = ()
    repeatable: bool = False
    aliases: tuple[str, ...] = ()
    allowed_tasks: tuple[str, ...] = ()
    example: str | None = None
    notes: str | None = None
    label: str | None = None
    description: str | None = None
    help_text: str | None = None
    category: str | None = None
    widget_hint: str | None = None
    basic_workflows: tuple[str, ...] = ()
    advanced_workflows: tuple[str, ...] = ALL_WORKFLOWS
    locked_in_workflows: tuple[str, ...] = ()
    default_value: object | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    exclusive_group: str | None = None
    block_schema: BlockSchema | None = None
    source_order: int = 0
    vector_length: int | None = None


@dataclass(frozen=True, slots=True)
class CatalogSourceInfo:
    """How the active CASTEP keyword catalog was located."""

    path: Path
    origin: str


@dataclass(frozen=True, slots=True)
class KeywordCatalogStatus:
    """What keyword metadata the running session is using, and why."""

    origin: str
    path: Path | None
    is_starter: bool
    castep_version: str
    cell_count: int
    param_count: int
    #: Why an explicitly configured catalog was skipped, if one was.
    issue: str | None = None


CATALOG_SOURCE_INFO: CatalogSourceInfo | None = None
#: Why explicitly configured catalog files were passed over during the last
#: refresh (missing or invalid), or ``None`` when nothing was skipped.
CATALOG_ISSUE: str | None = None
CATALOG_CASTEP_VERSION: str = ""


def _candidate_catalog_sources() -> tuple[CatalogSourceInfo, ...]:
    candidates: list[CatalogSourceInfo] = []

    for env_name in _ENV_JSON_VARS:
        raw = os.environ.get(env_name)
        if raw:
            candidates.append(
                CatalogSourceInfo(
                    path=Path(raw).expanduser(),
                    origin=f"env_json:{env_name}",
                )
            )

    raw_setup = os.environ.get(SETUP_KEYWORDS_JSON_ENV_VAR)
    if raw_setup:
        candidates.append(
            CatalogSourceInfo(
                path=Path(raw_setup).expanduser(),
                origin=_SETUP_CATALOG_ORIGIN,
            )
        )

    for env_name in _ENV_ROOT_VARS:
        raw = os.environ.get(env_name)
        if raw:
            candidates.append(
                CatalogSourceInfo(
                    path=Path(raw).expanduser() / _CATALOG_RELATIVE_PATH,
                    origin=f"env_root:{env_name}",
                )
            )

    repo_root = Path(__file__).resolve().parents[3]
    search_roots = (
        repo_root.parent,
        Path.home() / "Source",
    )
    for root in search_roots:
        if not root.exists():
            continue
        for entry in sorted(root.glob("CASTEP-*")):
            candidates.append(
                CatalogSourceInfo(
                    path=entry / _CATALOG_RELATIVE_PATH,
                    origin="auto_discovery",
                )
            )

    # Preserve order while removing duplicates.
    unique: list[CatalogSourceInfo] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(CatalogSourceInfo(path=resolved, origin=candidate.origin))
    return tuple(unique)


def _is_explicit_origin(origin: str) -> bool:
    """Return whether ``origin`` names a catalog the user configured by path."""

    return origin.startswith("env_json:") or origin == _SETUP_CATALOG_ORIGIN


def _load_base_catalog() -> dict[str, Any]:
    """Load the first usable ``castep_keywords.json`` among the candidates.

    A candidate that is missing or not a keyword catalog is skipped rather than
    fatal: the app must still start when a configured file was moved, deleted,
    or corrupted. Skipped *explicit* candidates are recorded in
    ``CATALOG_ISSUE`` so the UI can say why the session is degraded.

    Raises:
        FileNotFoundError: If no candidate yields a usable catalog.
    """

    global CATALOG_SOURCE_INFO
    global CATALOG_ISSUE
    global CATALOG_CASTEP_VERSION

    issues: list[str] = []
    candidates = _candidate_catalog_sources()
    for candidate in candidates:
        if not candidate.path.is_file():
            if _is_explicit_origin(candidate.origin):
                issues.append(f"The configured keyword catalog '{candidate.path}' no longer exists.")
            continue
        try:
            payload, summary = read_keyword_catalog_file(candidate.path, require_complete=False)
        except KeywordCatalogValidationError as exc:
            issues.append(f"The keyword catalog '{candidate.path}' could not be used: {exc}")
            continue

        CATALOG_SOURCE_INFO = candidate
        CATALOG_ISSUE = " ".join(issues) or None
        CATALOG_CASTEP_VERSION = summary.castep_version
        normalized: dict[str, Any] = {
            "castep_version": payload.get("castep_version", ""),
            "cell": [],
            "param": [],
        }
        for file in ("cell", "param"):
            normalized[file] = [
                {
                    "name": str(name).lower(),
                    "order": index,
                    **dict(metadata),
                }
                for index, (name, metadata) in enumerate(payload[file].items())
            ]
        return normalized

    CATALOG_ISSUE = " ".join(issues) or None
    searched = "\n".join(f"  - {source.path}" for source in candidates)
    raise FileNotFoundError(
        "MuFinder requires a local CASTEP metadata file at "
        "`Test/Utilities/ReadTS/castep_keywords.json`, but it is not bundled with the repository. "
        "Set MUFINDER_CASTEP_KEYWORDS_JSON or MUFINDER_CASTEP_ROOT, or place a CASTEP checkout next to this repo.\n"
        f"Searched:\n{searched}"
    )


def _humanize(name: str) -> str:
    return name.replace("_", " ").replace("mp", "MP").title().replace("Xc", "XC")


def _first_sentence(docstring: str) -> str:
    stripped = " ".join(line.strip() for line in docstring.splitlines() if line.strip())
    if not stripped:
        return ""
    match = re.search(r"[.?!](?:\s|$)", stripped)
    return stripped[: match.end()].strip() if match else stripped


def _extract_example(docstring: str) -> str | None:
    match = _EXAMPLE_RE.search(docstring)
    if not match:
        return None
    example = match.group(1).strip()
    return example or None


def _clean_allowed_values_segment(segment: str) -> str:
    return " ".join(segment.replace("\n", " ").split())


def _parse_allowed_values(docstring: str) -> tuple[str, ...]:
    if "Allowed values:" not in docstring:
        return ()
    match = _ALLOWED_VALUES_RE.search(docstring)
    if not match:
        return ()
    segment = _clean_allowed_values_segment(match.group(1))
    lower = segment.lower()
    if "(any" in lower or "libxc" in lower or "supported" in lower and ";" in segment:
        return ()
    candidate_text = segment.replace(" or ", ", ")
    tokens = [token.strip(" .,") for token in candidate_text.split(",")]
    tokens = [token for token in tokens if token]
    simple = [token for token in tokens if token == token.upper() or token.isdigit()]
    if not simple or len(simple) > 40:
        extracted = _ENUM_TOKEN_RE.findall(segment)
        extracted = [item for item in extracted if item == item.upper() or item.isdigit()]
        if 0 < len(extracted) <= 40:
            return tuple(dict.fromkeys(extracted))
        return ()
    return tuple(dict.fromkeys(simple))


def _parse_default_value(docstring: str, *, value_type: str, option_type: str) -> object | None:
    match = _DEFAULT_VALUE_RE.search(docstring)
    if not match:
        return None
    line = match.group(1).strip().splitlines()[0].strip()
    if not line or line.lower().startswith("set by") or "unless" in line.lower():
        return None
    if value_type == "bool":
        upper = line.upper()
        if "TRUE" in upper:
            return True
        if "FALSE" in upper:
            return False
        return None
    if value_type == "enum":
        token = line.split()[0].strip(".,")
        return token.upper()
    if value_type in {"int", "float", "physical"}:
        token = line.split()[0].strip(".,")
        if _NUMERIC_RE.match(token):
            if value_type == "int" and token.lstrip("+-").isdigit():
                return int(token)
            return float(token)
    if option_type == "String":
        token = line.split()[0].strip(".,")
        return token if token else None
    return None


def _infer_unit(docstring: str, option_type: str) -> str | None:
    if option_type != "Physical":
        return None
    for line in docstring.splitlines():
        if "Default value" in line or "Example:" in line:
            tokens = line.strip().split()
            if len(tokens) >= 2:
                candidate = tokens[-1].strip(".,")
                if candidate and any(char.isalpha() for char in candidate):
                    return candidate
    return None


def _infer_value_type(name: str, option_type: str, allowed_values: tuple[str, ...]) -> tuple[KeywordKind, str]:
    if option_type == "Block":
        return ("block", "block")
    if option_type == "Defined":
        return ("flag", "bool")
    if option_type == "Boolean (Logical)":
        return ("scalar", "bool")
    if option_type == "Integer Vector":
        return ("scalar", "int_triplet")
    if option_type == "Real Vector":
        return ("scalar", "real_vector")
    if option_type == "Integer":
        return ("scalar", "int")
    if option_type == "Real":
        return ("scalar", "float")
    if option_type == "Physical":
        return ("scalar", "physical")
    if option_type == "String" and allowed_values:
        return ("scalar", "enum")
    if option_type == "String":
        return ("scalar", "text")
    if allowed_values:
        return ("scalar", "enum")
    return ("scalar", "text")


def _infer_category(file: FileKind, name: str) -> str:
    if file == "cell":
        if "kpoint" in name:
            return "kpoints"
        if name.startswith("symmetry"):
            return "symmetry"
        if "constraint" in name or name.startswith("fix_"):
            return "constraints"
        if name.startswith("species_") or "pot" in name:
            return "pseudopotentials"
        return "structure"

    if name.startswith(("geom_", "cell_constraints", "ionic_constraints")):
        return "geometry"
    if name.startswith("md_"):
        return "molecular_dynamics"
    if name.startswith(("phonon_", "raman_", "born_", "thermo_")):
        return "phonon"
    if name.startswith(("magres_", "elnes_", "spectral_", "optics_", "tddft_", "wannier_", "tssearch_")):
        return name.split("_", maxsplit=1)[0]
    if name.startswith(("print_", "write_", "backup_", "checkpoint", "continuation", "reuse")):
        return "io"
    if "spin" in name or name.startswith(("mix_", "smearing", "elec_", "xc_", "basis_", "cut_off")):
        return "electronic"
    if "kpoint" in name:
        return "kpoints"
    return "general"


def _infer_allowed_tasks(name: str) -> tuple[str, ...]:
    if name.startswith("geom_"):
        return ("GEOMETRYOPTIMIZATION",)
    if name.startswith("md_"):
        return ("MOLECULARDYNAMICS",)
    if name.startswith(("phonon_", "raman_", "born_")):
        return ("PHONON", "PHONON+EFIELD", "THERMODYNAMICS")
    if name.startswith("thermo_"):
        return ("THERMODYNAMICS",)
    if name.startswith("magres_"):
        return ("MAGRES",)
    if name.startswith("elnes_"):
        return ("ELNES",)
    if name.startswith("spectral_"):
        return ("SPECTRAL",)
    if name.startswith("optics_"):
        return ("OPTICS",)
    if name.startswith("wannier_"):
        return ("WANNIER",)
    if name.startswith("tssearch_"):
        return ("TRANSITIONSTATESEARCH",)
    if name.startswith("efield_"):
        return ("EFIELD", "PHONON+EFIELD")
    if name.startswith(("bs_", "band_")):
        return ("BANDSTRUCTURE",)
    return ()


def _build_block_schema(payload: dict[str, object] | None) -> BlockSchema | None:
    if payload is None:
        return None
    columns = tuple(
        BlockColumnSpec(
            name=str(item.get("name", "Value")),
            value_kind=str(item.get("value_kind", "text")),
            required=bool(item.get("required", True)),
            allowed_values=tuple(str(value) for value in item.get("allowed_values", ())),
        )
        for item in payload.get("columns", ())
    )
    return BlockSchema(
        family=str(payload.get("family", "raw_rows")),
        columns=columns,
        allow_unit=bool(payload.get("allow_unit", False)),
        default_unit=str(payload["default_unit"]) if payload.get("default_unit") is not None else None,
        fixed_row_count=int(payload["fixed_row_count"]) if payload.get("fixed_row_count") is not None else None,
        min_rows=int(payload.get("min_rows", 0)),
        row_group_size=int(payload["row_group_size"]) if payload.get("row_group_size") is not None else None,
        row_labels=tuple(str(item) for item in payload.get("row_labels", ())),
        summary_mode=str(payload.get("summary_mode", "row_count")),
    )


def _apply_override(spec: KeywordSpec, override: dict[str, object]) -> KeywordSpec:
    payload = dict(override)
    if "block_schema" not in payload and spec.kind == "block":
        payload["block_schema"] = _build_block_schema(BLOCK_SCHEMAS.get(spec.file, {}).get(spec.name))
    elif "block_schema" in payload and isinstance(payload["block_schema"], dict):
        payload["block_schema"] = _build_block_schema(payload["block_schema"])
    return replace(spec, **payload)


def _base_spec(file: FileKind, entry: dict[str, object], *, aliases: tuple[str, ...]) -> KeywordSpec:
    name = str(entry["name"]).lower()
    option_type = str(entry.get("option_type", ""))
    docstring = str(entry.get("docstring", ""))
    allowed_values = _parse_allowed_values(docstring)
    kind, value_type = _infer_value_type(name, option_type, allowed_values)
    default_value = _parse_default_value(docstring, value_type=value_type, option_type=option_type)
    unit = _infer_unit(docstring, option_type)
    spec = KeywordSpec(
        name=name,
        file=file,
        kind=kind,
        value_type=value_type,
        option_type=option_type,
        level=str(entry.get("level", "")),
        docstring=docstring,
        unit=unit,
        allowed_values=allowed_values,
        aliases=aliases,
        allowed_tasks=_infer_allowed_tasks(name),
        example=_extract_example(docstring),
        label=_humanize(name),
        description=_first_sentence(docstring),
        help_text=docstring,
        category=_infer_category(file, name),
        widget_hint=None,
        basic_workflows=ALL_WORKFLOWS if name in {"task"} else (),
        advanced_workflows=ALL_WORKFLOWS,
        default_value=default_value,
        source_order=int(entry.get("order", 0)),
        vector_length=3 if value_type in {"int_triplet", "real_vector"} else None,
    )
    return _apply_override(spec, KEYWORD_OVERRIDES.get(file, {}).get(name, {}))


def _auto_alias_map(file: FileKind, entries: list[dict[str, object]]) -> dict[str, str]:
    names = {str(entry["name"]).lower() for entry in entries}
    alias_map = dict(MANUAL_ALIAS_MAP.get(file, {}))
    for entry in entries:
        name = str(entry["name"]).lower()
        if name in alias_map:
            continue
        docstring = str(entry.get("docstring", ""))
        match = _ALIAS_RE.search(docstring)
        if match:
            target = match.group(1).lower()
            if target in names:
                if alias_map.get(target) == name:
                    continue
                alias_map[name] = target
                continue
        if "isation" in name:
            target = name.replace("isation", "ization")
            if target in names:
                if alias_map.get(target) == name:
                    continue
                alias_map[name] = target
                continue
        if "polarised" in name:
            target = name.replace("polarised", "polarized")
            if target in names:
                if alias_map.get(target) == name:
                    continue
                alias_map[name] = target
    return alias_map


def _build_schema() -> tuple[dict[str, KeywordSpec], dict[str, KeywordSpec], dict[str, tuple[str, ...]]]:
    payload = _load_base_catalog()
    schema_by_file: dict[str, dict[str, KeywordSpec]] = {}
    source_names: dict[str, tuple[str, ...]] = {}

    for file in ("cell", "param"):
        entries = list(payload[file])
        alias_map = _auto_alias_map(file, entries)
        entries_by_name = {str(entry["name"]).lower(): entry for entry in entries}
        source_names[file] = tuple(entries_by_name)

        alias_groups: dict[str, list[str]] = defaultdict(list)
        canonical_order: dict[str, int] = {}
        for entry in entries:
            name = str(entry["name"]).lower()
            canonical = alias_map.get(name, name)
            if canonical != name:
                alias_groups[canonical].append(name)
            order = int(entry.get("order", 0))
            canonical_order[canonical] = min(order, canonical_order.get(canonical, order))

        specs: dict[str, KeywordSpec] = {}
        for canonical in sorted(canonical_order, key=canonical_order.get):
            base_entry = entries_by_name.get(canonical)
            if base_entry is None:
                continue
            spec = _base_spec(file, base_entry, aliases=tuple(sorted(set(alias_groups.get(canonical, ())))))
            specs[canonical] = spec

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
        if extra_names:
            source_names[file] = tuple((*source_names[file], *extra_names))

        schema_by_file[file] = specs

    return schema_by_file["cell"], schema_by_file["param"], source_names


def _override_kind_details(
    *,
    file: FileKind,
    name: str,
    override: dict[str, object],
) -> tuple[KeywordKind, str, str]:
    widget_hint = str(override.get("widget_hint", "") or "")
    allowed_values = tuple(str(item) for item in override.get("allowed_values", ()))
    has_block_schema = file == "cell" and name in BLOCK_SCHEMAS.get(file, {})

    if has_block_schema or widget_hint == "block":
        return ("block", "block", "Block")
    if widget_hint == "flag":
        return ("flag", "bool", "Defined")
    if widget_hint == "checkbox":
        return ("scalar", "bool", "Boolean (Logical)")
    if widget_hint == "int_triplet":
        return ("scalar", "int_triplet", "Integer Vector")
    if widget_hint == "real_vector":
        return ("scalar", "real_vector", "Real Vector")
    if widget_hint == "spinbox_int":
        return ("scalar", "int", "Integer")
    if widget_hint in {"physical", "unit_float"} or override.get("unit") is not None:
        return ("scalar", "physical", "Physical")
    if widget_hint == "spinbox_float":
        return ("scalar", "float", "Real")
    if widget_hint == "dropdown" or allowed_values:
        return ("scalar", "enum", "String")
    return ("scalar", "text", "String")


def _override_canonical_names(file: FileKind) -> tuple[str, ...]:
    names: list[str] = []
    if file == "cell":
        for name in BLOCK_SCHEMAS.get(file, {}):
            if name not in names:
                names.append(name)
    for name in KEYWORD_OVERRIDES.get(file, {}):
        if name not in names:
            names.append(name)
    return tuple(names)


def _override_spec(
    *,
    file: FileKind,
    name: str,
    source_order: int,
    aliases: tuple[str, ...],
) -> KeywordSpec:
    override = dict(KEYWORD_OVERRIDES.get(file, {}).get(name, {}))
    kind, value_type, option_type = _override_kind_details(file=file, name=name, override=override)

    docstring = str(
        override.get("help_text")
        or override.get("description")
        or override.get("notes")
        or override.get("label")
        or _humanize(name)
    )
    unit = override.get("unit")
    block_schema = None
    if kind == "block":
        override_block_schema = override.get("block_schema")
        if isinstance(override_block_schema, dict):
            block_schema = _build_block_schema(override_block_schema)
        else:
            block_schema = _build_block_schema(BLOCK_SCHEMAS.get(file, {}).get(name))

    return KeywordSpec(
        name=name,
        file=file,
        kind=kind,
        value_type=value_type,
        option_type=option_type,
        level="",
        docstring=docstring,
        unit=str(unit) if unit is not None else None,
        allowed_values=tuple(str(item) for item in override.get("allowed_values", ())),
        mutually_exclusive_with=tuple(str(item) for item in override.get("mutually_exclusive_with", ())),
        requires=tuple(str(item) for item in override.get("requires", ())),
        requires_any_of=tuple(str(item) for item in override.get("requires_any_of", ())),
        repeatable=bool(override.get("repeatable", False)),
        aliases=aliases,
        allowed_tasks=tuple(str(item) for item in override.get("allowed_tasks", _infer_allowed_tasks(name))),
        example=str(override["example"]) if override.get("example") is not None else None,
        notes=str(override["notes"]) if override.get("notes") is not None else None,
        label=str(override.get("label") or _humanize(name)),
        description=str(override.get("description") or _first_sentence(docstring)),
        help_text=str(override.get("help_text") or docstring),
        category=str(override.get("category") or _infer_category(file, name)),
        widget_hint=str(override["widget_hint"]) if override.get("widget_hint") is not None else None,
        basic_workflows=tuple(
            str(item) for item in override.get("basic_workflows", ALL_WORKFLOWS if name == "task" else ())
        ),
        advanced_workflows=tuple(str(item) for item in override.get("advanced_workflows", ALL_WORKFLOWS)),
        locked_in_workflows=tuple(str(item) for item in override.get("locked_in_workflows", ())),
        default_value=override.get("default_value"),
        min_value=override.get("min_value"),
        max_value=override.get("max_value"),
        step=override.get("step"),
        exclusive_group=str(override["exclusive_group"]) if override.get("exclusive_group") is not None else None,
        block_schema=block_schema,
        source_order=source_order,
        vector_length=(
            int(override["vector_length"])
            if override.get("vector_length") is not None
            else (3 if value_type in {"int_triplet", "real_vector"} else None)
        ),
    )


def _build_starter_schema() -> tuple[dict[str, KeywordSpec], dict[str, KeywordSpec], dict[str, tuple[str, ...]]]:
    """Build the schema from the bundled starter catalog and record it as the source.

    The translation lives in ``mufinder.castep_io.starter_catalog``, which imports
    this module's spec types, so it is imported here rather than at module level.
    """

    global CATALOG_SOURCE_INFO
    global CATALOG_CASTEP_VERSION

    from .starter_catalog import build_starter_schema

    CATALOG_CASTEP_VERSION = ""
    cell_keywords, param_keywords, source_keywords = build_starter_schema(_STARTER_CATALOG_PATH)
    CATALOG_SOURCE_INFO = CatalogSourceInfo(
        path=_STARTER_CATALOG_PATH,
        origin=_STARTER_CATALOG_ORIGIN,
    )
    return cell_keywords, param_keywords, source_keywords


CELL_KEYWORDS: dict[str, KeywordSpec] = {}
PARAM_KEYWORDS: dict[str, KeywordSpec] = {}
SOURCE_KEYWORDS: dict[str, tuple[str, ...]] = {}

SCHEMA: dict[FileKind, dict[str, KeywordSpec]] = {"cell": {}, "param": {}}


def _serialization_order(file: FileKind) -> tuple[str, ...]:
    names = sorted(SCHEMA[file], key=lambda item: SCHEMA[file][item].source_order)
    return tuple(names)


CELL_SERIALIZATION_ORDER: tuple[str, ...] = ()
PARAM_SERIALIZATION_ORDER: tuple[str, ...] = ()
SERIALIZATION_INDEX: dict[FileKind, dict[str, int]] = {"cell": {}, "param": {}}


def refresh_keyword_catalog(*, allow_fallback: bool = False) -> CatalogSourceInfo | None:
    """Rebuild the runtime keyword schema from the current environment settings."""

    global CELL_KEYWORDS
    global PARAM_KEYWORDS
    global SOURCE_KEYWORDS
    global SCHEMA
    global CELL_SERIALIZATION_ORDER
    global PARAM_SERIALIZATION_ORDER
    global SERIALIZATION_INDEX

    try:
        cell_keywords, param_keywords, source_keywords = _build_schema()
    except FileNotFoundError:
        if not allow_fallback:
            raise
        cell_keywords, param_keywords, source_keywords = _build_starter_schema()

    CELL_KEYWORDS = cell_keywords
    PARAM_KEYWORDS = param_keywords
    SOURCE_KEYWORDS = source_keywords
    SCHEMA = {
        "cell": CELL_KEYWORDS,
        "param": PARAM_KEYWORDS,
    }
    CELL_SERIALIZATION_ORDER = _serialization_order("cell")
    PARAM_SERIALIZATION_ORDER = _serialization_order("param")
    SERIALIZATION_INDEX = {
        "cell": {name: index for index, name in enumerate(CELL_SERIALIZATION_ORDER)},
        "param": {name: index for index, name in enumerate(PARAM_SERIALIZATION_ORDER)},
    }
    return CATALOG_SOURCE_INFO


refresh_keyword_catalog(allow_fallback=True)


def canonical_keyword_name(file: FileKind, name: str) -> str:
    """Resolve a keyword or alias to the canonical lowercase runtime name."""

    normalized = name.strip().lower()
    if normalized in SCHEMA[file]:
        return normalized
    for canonical_name, spec in SCHEMA[file].items():
        if normalized in spec.aliases:
            return canonical_name
    return normalized


def get_keyword_spec(file: FileKind, name: str) -> KeywordSpec | None:
    """Return the runtime keyword specification, if supported."""

    return SCHEMA[file].get(canonical_keyword_name(file, name))


def keyword_sort_key(file: FileKind, name: str) -> tuple[int, str]:
    """Return a deterministic sort key for a keyword based on source order."""

    canonical = canonical_keyword_name(file, name)
    return (SERIALIZATION_INDEX[file].get(canonical, 1_000_000), canonical)


def serialization_order(file: FileKind) -> tuple[str, ...]:
    """Return the current keyword serialization order for a CASTEP file kind.

    Consumers must call this rather than importing ``CELL_SERIALIZATION_ORDER`` /
    ``PARAM_SERIALIZATION_ORDER`` by value at import time: those module globals are
    reassigned by :func:`refresh_keyword_catalog`, so a value-bound copy would keep the
    stale order after a runtime catalog refresh (e.g. reconfiguring the CASTEP location).
    """

    return CELL_SERIALIZATION_ORDER if file == "cell" else PARAM_SERIALIZATION_ORDER


def keyword_specs(file: FileKind) -> dict[str, KeywordSpec]:
    """Return the live mapping of canonical keyword name to spec for a file kind.

    Like :func:`serialization_order`, this exists so consumers read the current schema
    instead of importing the ``SCHEMA`` / ``CELL_KEYWORDS`` / ``PARAM_KEYWORDS`` globals by
    value: :func:`refresh_keyword_catalog` reassigns those, so a value-bound copy would go
    stale after a runtime catalog refresh.
    """

    return SCHEMA[file]


def keyword_catalog_source_info() -> CatalogSourceInfo | None:
    """Return how the active keyword catalog was located."""

    return CATALOG_SOURCE_INFO


def keyword_catalog_status() -> KeywordCatalogStatus:
    """Describe the keyword metadata the running session is using.

    ``is_starter`` is the signal callers should branch on: it is true when no
    ``castep_keywords.json`` could be loaded and MuFinder is running on the
    small bundled starter catalog, which does not know every keyword CASTEP can
    write back into its own output files.
    """

    source = CATALOG_SOURCE_INFO
    origin = source.origin if source is not None else _STARTER_CATALOG_ORIGIN
    return KeywordCatalogStatus(
        origin=origin,
        path=source.path if source is not None else None,
        is_starter=origin == _STARTER_CATALOG_ORIGIN,
        castep_version=CATALOG_CASTEP_VERSION,
        cell_count=len(SOURCE_KEYWORDS.get("cell", ())),
        param_count=len(SOURCE_KEYWORDS.get("param", ())),
        issue=CATALOG_ISSUE,
    )
