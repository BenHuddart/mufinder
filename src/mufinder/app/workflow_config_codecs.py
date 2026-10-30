"""Persistence codecs for workflow configurations.

Each workflow stores its row configuration in
``CalculationDefinition.parameters["configuration"]``. The functions here turn
the configuration dataclasses into that stored mapping and back. Serialising
copies parameter values as they are (the JSON persistence codec flattens them
later); deserialising restores every parameter value to its structured form
through :func:`~mufinder.app.workflow_parameters.restore_structured_value`
and skips rows it cannot read rather than failing the whole configuration.

The source-keyed workflows (quantum corrections, energy barriers) have no
configuration dataclass: each row's settings live under
``configuration["row_settings"][key]`` as its parameter values, its sorted
``added_advanced`` names and workflow extras such as ``num_farms``. The
``workflow_row_configuration`` family reads and writes those entries.

The module lives in ``mufinder.app`` because the configuration dataclasses
(``mufinder.app.convergence_types``, ``relaxation_types`` and ``uep_types``)
and the parameter restorer do. It imports no Qt.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from uuid import UUID

from mufinder.domain.models import CalculationDefinition
from mufinder.persistence.codec import to_jsonable

from mufinder.app.adapter_common import coerce_supercell_matrix
from mufinder.app.convergence_types import ConvergenceConfig, SweepValue
from mufinder.app.relaxation_types import (
    MuonRelaxationConfig,
    MuonRelaxationRow,
    PristineRelaxationConfig,
    PristineRelaxationRow,
    SupercellMatrix,
)
from mufinder.app.uep_types import UepConfig, UepSourceRow
from mufinder.app.workflow_parameters import (
    merge_batch_advanced,
    merge_batch_values,
    restore_structured_value,
)


def normalized_supercell_matrix(matrix: SupercellMatrix | None) -> SupercellMatrix | None:
    """Return ``matrix`` with integer entries, or ``None`` for no matrix.

    Raises ``ValueError`` when ``matrix`` is not three rows of three
    integer-convertible values.
    """

    if matrix is None:
        return None
    normalized = coerce_supercell_matrix(matrix)
    if normalized is None:
        raise ValueError(f"Not a 3x3 integer supercell matrix: {matrix!r}")
    return normalized


def _serialize_supercell_matrix(matrix: SupercellMatrix | None) -> list[list[int]] | None:
    return [list(line) for line in matrix] if matrix is not None else None


def _deserialize_supercell_matrix(payload: object) -> SupercellMatrix | None:
    # The stored matrix is always a nested list; anything else is unreadable.
    if not isinstance(payload, list):
        return None
    return coerce_supercell_matrix(payload)


def _restore_parameter_values(payload: object) -> dict[str, object]:
    params = payload if isinstance(payload, dict) else {}
    return {
        str(name): restore_structured_value(None, str(name), raw_value)
        for name, raw_value in params.items()
    }


def _deserialize_added_advanced(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, list):
        return ()
    return tuple(str(name) for name in payload if isinstance(name, str))


def _row_label(entry: Mapping[str, object]) -> str:
    return str(entry.get("label", "")).strip()


def serialize_uep_config(config: UepConfig) -> dict[str, object]:
    """Return the stored mapping for a UEP configuration."""

    return {
        "name_prefix": "uep" if config.name_prefix == "uep_source" else config.name_prefix,
        "rows": [
            {
                "label": row.label,
                "supercell_matrix": _serialize_supercell_matrix(row.supercell_matrix),
                "parameter_values": dict(row.parameter_values),
                "added_advanced": list(row.added_advanced),
            }
            for row in config.rows
        ],
    }


def deserialize_uep_config(payload: object) -> UepConfig | None:
    """Return the UEP configuration stored in ``payload``, or ``None``."""

    if not isinstance(payload, dict):
        return None
    rows_payload = payload.get("rows", [])
    if not isinstance(rows_payload, list):
        return None

    rows: list[UepSourceRow] = []
    for entry in rows_payload:
        if not isinstance(entry, dict):
            continue
        label = _row_label(entry)
        if not label:
            continue
        rows.append(
            UepSourceRow(
                label=label,
                supercell_matrix=_deserialize_supercell_matrix(entry.get("supercell_matrix")),
                parameter_values=_restore_parameter_values(
                    entry.get("parameter_values", entry.get("params", {}))
                ),
                added_advanced=_deserialize_added_advanced(entry.get("added_advanced", [])),
            )
        )

    raw_prefix = str(payload.get("name_prefix", "uep")).strip()
    return UepConfig(
        name_prefix="uep" if raw_prefix == "uep_source" else raw_prefix or "uep",
        rows=tuple(rows),
    )


def serialize_pristine_relaxation_config(config: PristineRelaxationConfig) -> dict[str, object]:
    """Return the stored mapping for a pristine-relaxation configuration."""

    return {
        "name_prefix": config.name_prefix,
        "rows": [
            {
                "label": row.label,
                "fix_all_cell": bool(row.fix_all_cell),
                "supercell_matrix": _serialize_supercell_matrix(row.supercell_matrix),
                "parameter_values": dict(row.parameter_values),
                "added_advanced": list(row.added_advanced),
            }
            for row in config.rows
        ],
    }


def deserialize_pristine_relaxation_config(payload: object) -> PristineRelaxationConfig | None:
    """Return the pristine-relaxation configuration in ``payload``, or ``None``."""

    if not isinstance(payload, dict):
        return None
    rows_payload = payload.get("rows", [])
    if not isinstance(rows_payload, list):
        return None

    rows: list[PristineRelaxationRow] = []
    for entry in rows_payload:
        if not isinstance(entry, dict):
            continue
        label = _row_label(entry)
        if not label:
            continue
        rows.append(
            PristineRelaxationRow(
                label=label,
                fix_all_cell=bool(entry.get("fix_all_cell", False)),
                supercell_matrix=_deserialize_supercell_matrix(entry.get("supercell_matrix")),
                parameter_values=_restore_parameter_values(
                    entry.get("parameter_values", entry.get("params", {}))
                ),
                added_advanced=_deserialize_added_advanced(entry.get("added_advanced", [])),
            )
        )

    return PristineRelaxationConfig(
        name_prefix=str(payload.get("name_prefix", "pristine_relax")).strip() or "pristine_relax",
        rows=tuple(rows),
    )


def normalized_parameter_values(values: Mapping[str, object]) -> object:
    """Parameter values in the JSON form they are stored in, for comparing two rows."""

    return to_jsonable({str(name): value for name, value in values.items()})


def pristine_relaxation_row_signature(row: PristineRelaxationRow) -> tuple[object, ...]:
    """The stored settings of a row: two rows that match run the same job."""

    return (
        bool(row.fix_all_cell),
        normalized_supercell_matrix(row.supercell_matrix),
        normalized_parameter_values(row.parameter_values),
        tuple(str(name) for name in row.added_advanced),
    )


def serialize_muon_relaxation_config(config: MuonRelaxationConfig) -> dict[str, object]:
    """Return the stored mapping for a muon-relaxation configuration."""

    return {
        "rows": [
            {
                "site_id": str(row.site_id),
                "label": row.label,
                "initial_fractional_position": list(row.initial_fractional_position),
                "charge_state": row.charge_state,
                "supercell_matrix": _serialize_supercell_matrix(row.supercell_matrix),
                "parameter_values": dict(row.parameter_values),
                "added_advanced": list(row.added_advanced),
                "imported": bool(row.imported),
                "import_batch_id": (
                    str(row.import_batch_id) if row.import_batch_id is not None else None
                ),
            }
            for row in config.rows
        ]
    }


def deserialize_muon_relaxation_config(payload: object) -> MuonRelaxationConfig | None:
    """Return the muon-relaxation configuration in ``payload``, or ``None``.

    A row without a readable site id, label or three-component position is
    skipped. ``fractional_position`` is read when ``initial_fractional_position``
    is absent.
    """

    if not isinstance(payload, dict):
        return None
    rows_payload = payload.get("rows", [])
    if not isinstance(rows_payload, list):
        return None

    rows: list[MuonRelaxationRow] = []
    for entry in rows_payload:
        if not isinstance(entry, dict):
            continue
        try:
            site_id = UUID(str(entry.get("site_id")))
        except (TypeError, ValueError):
            continue
        label = _row_label(entry)
        if not label:
            continue
        raw_position = entry.get(
            "initial_fractional_position", entry.get("fractional_position", [])
        )
        if not isinstance(raw_position, (list, tuple)) or len(raw_position) != 3:
            continue
        import_batch_id: UUID | None = None
        raw_batch_id = entry.get("import_batch_id")
        if raw_batch_id is not None:
            try:
                import_batch_id = UUID(str(raw_batch_id))
            except (TypeError, ValueError):
                import_batch_id = None
        rows.append(
            MuonRelaxationRow(
                site_id=site_id,
                label=label,
                initial_fractional_position=(
                    float(raw_position[0]),
                    float(raw_position[1]),
                    float(raw_position[2]),
                ),
                charge_state=str(entry.get("charge_state", "positive_muon")),
                supercell_matrix=_deserialize_supercell_matrix(entry.get("supercell_matrix")),
                parameter_values=_restore_parameter_values(entry.get("parameter_values", {})),
                added_advanced=_deserialize_added_advanced(entry.get("added_advanced", [])),
                imported=bool(entry.get("imported", False)),
                import_batch_id=import_batch_id,
            )
        )
    return MuonRelaxationConfig(rows=tuple(rows))


def _serialize_sweep_value(value: SweepValue) -> object:
    return list(value) if isinstance(value, tuple) else value


def serialize_convergence_config(config: ConvergenceConfig) -> dict[str, object]:
    """Return the stored mapping for a cutoff or k-point convergence configuration."""

    return {
        "name_prefix": config.name_prefix,
        "sweep_parameter": config.sweep_parameter,
        "sweep_values": [_serialize_sweep_value(value) for value in config.sweep_values],
        "shared_params": config.shared_params,
        "overrides": [
            {
                "cutoffs": [_serialize_sweep_value(value) for value in sorted(group)],
                "params": payload,
            }
            for group, payload in config.overrides.items()
        ],
    }


def deserialize_convergence_config(payload: object) -> ConvergenceConfig | None:
    """Return the convergence configuration in ``payload``, or ``None``.

    The sweep is read from ``sweep_values``; an override group reads
    ``cutoffs`` in preference to ``sweep_values``. An unreadable sweep value
    rejects the configuration, while an unreadable override group is skipped.
    """

    if not isinstance(payload, dict):
        return None

    sweep_parameter = (
        str(payload.get("sweep_parameter", "cut_off_energy")).strip() or "cut_off_energy"
    )
    if sweep_parameter not in {"cut_off_energy", "kpoint_mp_grid"}:
        return None

    def parse_sweep_value(value: object) -> SweepValue:
        if sweep_parameter == "cut_off_energy":
            return float(value)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return (int(value[0]), int(value[1]), int(value[2]))
        raise ValueError

    try:
        raw_sweep_values = payload.get("sweep_values")
        if not isinstance(raw_sweep_values, list):
            raw_sweep_values = []
        sweep_values = tuple(parse_sweep_value(value) for value in raw_sweep_values)
    except (TypeError, ValueError):
        return None

    shared_params = payload.get("shared_params", {})
    if not isinstance(shared_params, dict):
        return None

    overrides: dict[frozenset[object], dict[str, object]] = {}
    raw_overrides = payload.get("overrides", [])
    if isinstance(raw_overrides, list):
        for entry in raw_overrides:
            if not isinstance(entry, dict):
                continue
            try:
                raw_group = entry.get("cutoffs", entry.get("sweep_values", []))
                cutoff_group = frozenset(parse_sweep_value(value) for value in raw_group)
            except (TypeError, ValueError):
                continue
            params = entry.get("params", {})
            if not cutoff_group or not isinstance(params, dict):
                continue
            overrides[cutoff_group] = _restore_parameter_values(params)

    default_prefix = (
        "kpoint_convergence" if sweep_parameter == "kpoint_mp_grid" else "cutoff_convergence"
    )

    return ConvergenceConfig(
        sweep_parameter=sweep_parameter,
        sweep_values=sweep_values,
        shared_params=_restore_parameter_values(shared_params),
        overrides=overrides,
        name_prefix=str(payload.get("name_prefix", default_prefix)).strip() or default_prefix,
    )


#: The row-settings keys that are not workflow extras.
_ROW_SETTINGS_PARAMETER_KEYS = frozenset({"parameter_values", "params", "added_advanced"})


def workflow_row_configuration(
    definition: CalculationDefinition | None,
    key: str,
) -> tuple[dict[str, object], tuple[str, ...], dict[str, object]]:
    """Return a source row's stored ``(parameter values, advanced names, extras)``.

    Parameter values are restored to their structured form; a missing or
    malformed entry reads as empty.
    """

    if definition is None:
        return {}, (), {}
    config = definition.parameters.get("configuration", {})
    if not isinstance(config, dict):
        return {}, (), {}
    raw_rows = config.get("row_settings", {})
    if not isinstance(raw_rows, dict):
        return {}, (), {}
    raw_entry = raw_rows.get(key, {})
    if not isinstance(raw_entry, dict):
        return {}, (), {}
    extras = {
        str(name): value
        for name, value in raw_entry.items()
        if name not in _ROW_SETTINGS_PARAMETER_KEYS
    }
    return (
        _restore_parameter_values(raw_entry.get("parameter_values", raw_entry.get("params", {}))),
        _deserialize_added_advanced(raw_entry.get("added_advanced", [])),
        extras,
    )


def source_row_parameter_values(
    definition: CalculationDefinition | None,
    key: str,
    inherited: Mapping[str, object],
    *,
    dropped: Iterable[str] = (),
    defaults: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a source row's effective parameter values.

    The row's stored parameter values sit over ``inherited`` (the settings of
    its source relaxation); the ``dropped`` names, which the workflow writes
    itself, are removed from the result, and ``defaults`` fill what is still
    unset.
    """

    params, _advanced, _extras = workflow_row_configuration(definition, key)
    values = dict(inherited)
    values.update(params)
    for name in dropped:
        values.pop(name, None)
    for name, value in (defaults or {}).items():
        values.setdefault(name, value)
    return values


def workflow_row_configuration_exists(definition: CalculationDefinition | None, key: str) -> bool:
    """Return whether a source row stores any parameter, advanced name or non-blank extra."""

    params, advanced, extras = workflow_row_configuration(definition, key)
    return bool(params) or bool(advanced) or any(str(value).strip() for value in extras.values())


def num_farms_from_extras(extras: Mapping[str, object]) -> int:
    """Return a row's CASTEP task-farm count from its extras; at least 1."""

    raw = extras.get("num_farms", 1)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def row_settings_payload(
    values: Mapping[str, object],
    advanced_names: Iterable[str],
    extras: Mapping[str, object],
) -> dict[str, object] | None:
    """Return the ``row_settings`` entry for values a configure dialog returned.

    The workflow-locked ``task`` is dropped. ``None`` means there is nothing to
    store: no parameter value, no advanced name and no extra.
    """

    parameter_values = to_jsonable(
        {str(name): value for name, value in values.items() if name != "task"}
    )
    payload: dict[str, object] = {
        "parameter_values": parameter_values if isinstance(parameter_values, dict) else {},
        "added_advanced": sorted(set(advanced_names)),
    }
    stored_extras = {str(name): value for name, value in extras.items()}
    payload.update(stored_extras)
    if payload["parameter_values"] or payload["added_advanced"] or stored_extras:
        return payload
    return None


def batch_row_payload(
    definition: CalculationDefinition | None,
    key: str,
    payload: dict[str, object] | None,
    untouched_mixed: Collection[str],
) -> dict[str, object] | None:
    """Return the ``row_settings`` entry to store for one row of a batch edit.

    ``payload`` is what the dialog produced for every row. Parameters that
    differed between the rows and were left alone are taken from the row's own
    stored configuration instead: a row that set one explicitly keeps it, and a
    row that did not keeps deriving it from its source defaults.
    """

    if not untouched_mixed:
        return payload
    row_params, row_advanced, _extras = workflow_row_configuration(definition, key)
    merged = dict(payload or {})
    collected = merged.get("parameter_values")
    merged["parameter_values"] = merge_batch_values(
        collected if isinstance(collected, dict) else {},
        to_jsonable({str(name): value for name, value in row_params.items()}) or {},
        untouched_mixed,
    )
    merged["added_advanced"] = sorted(
        merge_batch_advanced(merged.get("added_advanced", ()), row_advanced, untouched_mixed)
    )
    has_extras = any(name not in {"parameter_values", "added_advanced"} for name in merged)
    if not (merged["parameter_values"] or merged["added_advanced"] or has_extras):
        return None
    return merged


def store_workflow_row_configuration(
    definition: CalculationDefinition,
    key: str,
    payload: Mapping[str, object] | None,
) -> None:
    """Store a source row's ``row_settings`` entry; ``None`` removes it.

    Removing the last entry removes ``row_settings`` itself.
    """

    config = definition.parameters.setdefault("configuration", {})
    if not isinstance(config, dict):
        config = {}
        definition.parameters["configuration"] = config
    rows = config.setdefault("row_settings", {})
    if not isinstance(rows, dict):
        rows = {}
        config["row_settings"] = rows
    if payload is None:
        rows.pop(key, None)
        if not rows:
            config.pop("row_settings", None)
        return
    rows[key] = payload
