"""``ViewerStructureCheck`` says what changed in what a 3D viewer draws, by content."""

from __future__ import annotations

from copy import deepcopy

import pytest

from mufinder.app.structure_render_guard import (
    ViewerStructureCheck,
    viewer_structure_fingerprint,
)
from mufinder.domain.models import Structure
from tests.support.structures import host_structure


def _host() -> Structure:
    return host_structure(
        species=("Fe", "Fe"),
        properties={"atomic_spins": [1.0, -1.0], "atom_colors": [None, None]},
    )


def _remembered(structure: Structure) -> ViewerStructureCheck:
    check = ViewerStructureCheck()
    check.remember(structure)
    return check


def test_the_first_structure_and_none_are_new_geometry() -> None:
    check = ViewerStructureCheck()

    assert check.change(None).geometry
    assert check.change(_host()).geometry


def test_the_same_content_is_unchanged_whatever_object_holds_it() -> None:
    host = _host()
    check = _remembered(host)

    assert check.change(host).unchanged
    assert check.change(deepcopy(host)).unchanged
    empty = ViewerStructureCheck()
    empty.remember(None)
    assert empty.change(None).unchanged


@pytest.mark.parametrize(
    "edit",
    [
        lambda host: host.fractional_positions[0].__setitem__(0, 0.1),
        lambda host: host.species.__setitem__(1, "Co"),
        lambda host: host.lattice_vectors[0].__setitem__(0, 6.0),
    ],
    ids=["position", "species", "lattice"],
)
def test_an_in_place_geometry_edit_is_new_geometry(edit) -> None:
    host = _host()
    check = _remembered(host)

    edit(host)

    change = check.change(host)
    assert change.geometry
    assert change.properties == frozenset()
    assert check.change(None).geometry


def test_an_in_place_spin_edit_is_spins_only() -> None:
    host = _host()
    check = _remembered(host)

    host.properties["atomic_spins"][0] = 2.0
    change = check.change(host)

    assert change.spins_only
    assert change.had_spins
    assert not change.geometry
    check.remember(host)
    assert check.change(host).unchanged


@pytest.mark.parametrize(
    ("key", "value"),
    [("atom_colors", ["#ff0000", None]), ("pbc", [True, True, False])],
)
def test_a_colour_or_periodicity_edit_is_a_property_change_not_spins_only(key, value) -> None:
    host = _host()
    check = _remembered(host)

    host.properties[key] = value
    change = check.change(host)

    assert change.properties == frozenset({key})
    assert not change.spins_only
    assert not change.unchanged


def test_had_spins_says_whether_the_last_structure_had_a_non_zero_spin() -> None:
    host = _host()
    host.properties["atomic_spins"] = [0.0, 0.0]
    check = _remembered(host)

    host.properties["atomic_spins"][0] = 1.0
    change = check.change(host)

    assert change.spins_only
    assert not change.had_spins


def test_forget_makes_the_next_structure_new() -> None:
    host = _host()
    check = _remembered(host)

    check.forget()

    assert check.change(host).geometry


def test_the_viewer_fingerprint_follows_geometry_and_viewer_properties_only() -> None:
    host = _host()
    before = viewer_structure_fingerprint(host)

    host.properties["propagation_vector"] = [0.5, 0.0, 0.0]
    assert viewer_structure_fingerprint(host) == before
    host.properties["atom_colors"][1] = "#00ff00"
    assert viewer_structure_fingerprint(host) != before
