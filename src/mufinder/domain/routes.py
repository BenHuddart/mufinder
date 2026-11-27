"""Navigator routes that more than the main window has to name."""

from __future__ import annotations

from uuid import UUID

_COMPARISON_ROUTE_PREFIX = "pristine_relaxation/comparison/"
MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX = "magnetic_fields/workspace/"
RELAXATION_ANALYSIS_WORKSPACE_ROUTE_PREFIX = "muon_analysis/workspace/"


def magnetic_field_workspace_route(workspace_id: str) -> str:
    """The navigator route of one saved magnetic-field workspace."""

    return f"{MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX}{workspace_id}"


def magnetic_field_workspace_id_from_route(route: object) -> str | None:
    """The magnetic-field workspace a navigator route names, or ``None`` for any other route."""

    return _workspace_id_from_route(route, MAGNETIC_FIELD_WORKSPACE_ROUTE_PREFIX)


def relaxation_analysis_workspace_route(workspace_id: str) -> str:
    """The navigator route of one relaxation analysis workspace."""

    return f"{RELAXATION_ANALYSIS_WORKSPACE_ROUTE_PREFIX}{workspace_id}"


def relaxation_analysis_workspace_id_from_route(route: object) -> str | None:
    """The relaxation analysis workspace a navigator route names, or ``None`` for any other."""

    return _workspace_id_from_route(route, RELAXATION_ANALYSIS_WORKSPACE_ROUTE_PREFIX)


def _workspace_id_from_route(route: object, prefix: str) -> str | None:
    if not isinstance(route, str) or not route.startswith(prefix):
        return None
    workspace_id = route[len(prefix) :].strip()
    return workspace_id or None


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
