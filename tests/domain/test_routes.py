"""Navigator routes for saved workspaces and comparison tables."""

from __future__ import annotations

from uuid import uuid4

import pytest

from mufinder.domain.routes import (
    comparison_route,
    comparison_table_id_from_route,
    magnetic_field_workspace_id_from_route,
    magnetic_field_workspace_route,
    relaxation_analysis_workspace_id_from_route,
    relaxation_analysis_workspace_route,
)


def test_workspace_routes_name_their_workspace_and_read_back() -> None:
    assert magnetic_field_workspace_route("abc") == "magnetic_fields/workspace/abc"
    assert relaxation_analysis_workspace_route("abc") == "muon_analysis/workspace/abc"
    assert magnetic_field_workspace_id_from_route("magnetic_fields/workspace/abc ") == "abc"
    assert relaxation_analysis_workspace_id_from_route("muon_analysis/workspace/abc") == "abc"


@pytest.mark.parametrize(
    "route", [None, 3, "uep", "magnetic_fields/workspace/", "muon_analysis/workspace/ "]
)
def test_other_routes_name_no_workspace(route: object) -> None:
    assert magnetic_field_workspace_id_from_route(route) is None
    assert relaxation_analysis_workspace_id_from_route(route) is None


def test_a_workspace_route_is_not_read_as_the_other_kind() -> None:
    assert magnetic_field_workspace_id_from_route(relaxation_analysis_workspace_route("x")) is None
    assert relaxation_analysis_workspace_id_from_route(magnetic_field_workspace_route("x")) is None


def test_comparison_routes_read_back_their_table() -> None:
    table_id = uuid4()

    assert comparison_table_id_from_route(comparison_route(table_id)) == table_id
    assert comparison_table_id_from_route("pristine_relaxation/comparison/nope") is None
