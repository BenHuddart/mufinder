"""JSON serialization helpers for MuFinder dataclasses."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
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
    if is_dataclass(value):
        return to_jsonable(asdict(value))
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
