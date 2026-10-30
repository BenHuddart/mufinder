"""JSON serialization helpers for MuFinder dataclasses."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID


_JSON_SCALAR_TYPES = frozenset({str, int, float, bool, type(None)})


def to_jsonable(value: Any) -> Any:
    """Recursively convert Python objects to JSON-safe values."""
    # Fast path for the exact built-in containers and scalars: UI state can hold
    # hundreds of thousands of floats (isosurface meshes, slice samples), and
    # running every one through the checks below dominated its serialization.
    # Exact type matches keep IntEnum/StrEnum and container subclasses on the
    # general path.
    kind = type(value)
    if kind in _JSON_SCALAR_TYPES:
        return value
    if kind is list or kind is tuple:
        return [item if type(item) in _JSON_SCALAR_TYPES else to_jsonable(item) for item in value]
    if kind is dict:
        return {
            str(k): v if type(v) in _JSON_SCALAR_TYPES else to_jsonable(v) for k, v in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        # Walk the fields directly: ``dataclasses.asdict`` deep-copies the whole
        # tree first, which then had to be walked a second time here.
        result: dict[str, Any] = {}
        for item in fields(value):
            field_value = getattr(value, item.name)
            result[item.name] = (
                field_value if type(field_value) in _JSON_SCALAR_TYPES else to_jsonable(field_value)
            )
        return result
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO8601 datetime string."""
    if value is None:
        return None
    return datetime.fromisoformat(value)


def parse_uuid(value: str | None) -> UUID | None:
    """Parse UUID from string."""
    if value is None:
        return None
    return UUID(value)


def uuid_or_none(value: object) -> UUID | None:
    """Return ``value`` read as a UUID, or ``None`` when it is missing or not one.

    For lenient reads of loosely typed saved state such as ``ui_state`` entries;
    :func:`parse_uuid` is the strict reader for record fields.
    """

    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def int_or_none(value: object) -> int | None:
    """Return ``int(value)``, or ``None`` when ``value`` does not convert."""

    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


def float_triplet(value: object) -> tuple[float, float, float]:
    """Return ``value`` as three floats.

    Raises ``TypeError`` when ``value`` is not iterable and ``ValueError`` when it
    does not hold exactly three numbers.
    """

    values = list(value)  # type: ignore[call-overload]
    if len(values) != 3:
        raise ValueError("expected a three-vector")
    return (float(values[0]), float(values[1]), float(values[2]))


def float_triplet_or_none(value: object) -> tuple[float, float, float] | None:
    """Return a list or tuple of three numbers as three floats, or ``None`` for anything else."""

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
