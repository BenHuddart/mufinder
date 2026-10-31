"""Navigator routes that more than the main window has to name."""

from __future__ import annotations

from uuid import UUID

_COMPARISON_ROUTE_PREFIX = "pristine_relaxation/comparison/"
MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX = "magnetic_fields/workspace/"


def magnetic_field_workspace_route(workspace_id: str) -> str:
    """The navigator route of one saved magnetic-field workspace."""

    return f"{MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX}{workspace_id}"


def comparison_route(table_id: UUID) -> str:
    """The navigator route of one comparison table."""

    return f"{_COMPARISON_ROUTE_PREFIX}{table_id}"


def comparison_table_id_from_route(route: object) -> UUID | None:
    """The table a navigator route names, or ``None`` for any other route."""

    if not isinstance(route, str) or not route.startswith(_COMPARISON_ROUTE_PREFIX):
        return None
    try:
        return UUID(route[len(_COMPARISON_ROUTE_PREFIX) :].strip())
    except ValueError:
        return None
