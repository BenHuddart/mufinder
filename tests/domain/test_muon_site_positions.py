"""Tests for the initial and relaxed positions a ``MuonSite`` records."""

from __future__ import annotations

from uuid import uuid4

import pytest

from mufinder.domain.models import MuonSite


def _staged_site() -> MuonSite:
    return MuonSite(
        label="Site 1",
        status="staged",
        initial_fractional_position=[0.1, 0.2, 0.3],
        initial_cartesian_position=[0.4, 0.8, 1.2],
    )


def test_unrelaxed_site_is_where_it_started() -> None:
    site = _staged_site()

    assert site.relaxed_fractional_position is None
    assert site.current_fractional_position() == [0.1, 0.2, 0.3]


def test_recording_a_relaxation_keeps_the_initial_position() -> None:
    site = _staged_site()
    run_id = uuid4()

    site.record_relaxation(run_id, total_energy_eV=-12.0, fractional_position=(0.15, 0.25, 0.35))

    assert site.status == "relaxed"
    assert site.relaxation_run_id == run_id
    assert site.total_energy_eV == -12.0
    assert site.initial_fractional_position == [0.1, 0.2, 0.3]
    assert site.relaxed_fractional_position == [0.15, 0.25, 0.35]
    assert site.current_fractional_position() == [0.15, 0.25, 0.35]


def test_a_relaxation_without_a_final_position_falls_back_to_the_initial_one() -> None:
    site = _staged_site()

    site.record_relaxation(uuid4(), total_energy_eV=None, fractional_position=None)

    assert site.relaxed_fractional_position is None
    assert site.current_fractional_position() == [0.1, 0.2, 0.3]


def test_clearing_a_relaxation_returns_the_site_to_staged_at_its_initial_position() -> None:
    site = _staged_site()
    site.record_relaxation(uuid4(), total_energy_eV=-12.0, fractional_position=(0.15, 0.25, 0.35))

    site.clear_relaxation()

    assert site.status == "staged"
    assert site.relaxation_run_id is None
    assert site.total_energy_eV is None
    assert site.relaxed_fractional_position is None
    assert site.current_fractional_position() == [0.1, 0.2, 0.3]


def test_clearing_leaves_a_non_relaxed_status_alone() -> None:
    site = MuonSite(status="generated")

    site.clear_relaxation()

    assert site.status == "generated"


def test_legacy_fractional_position_and_cartesian_position_alias_the_initial_position() -> None:
    site = _staged_site()

    assert site.fractional_position == [0.1, 0.2, 0.3]
    assert site.cartesian_position == [0.4, 0.8, 1.2]

    site.fractional_position = [0.5, 0.5, 0.5]
    site.cartesian_position = [2.0, 2.0, 2.0]

    assert site.initial_fractional_position == [0.5, 0.5, 0.5]
    assert site.initial_cartesian_position == [2.0, 2.0, 2.0]


def test_the_legacy_names_are_not_constructor_keywords() -> None:
    with pytest.raises(TypeError):
        MuonSite(fractional_position=[0.1, 0.2, 0.3])  # type: ignore[call-arg]
