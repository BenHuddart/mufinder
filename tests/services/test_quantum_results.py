"""Tests for the quantum-phonon result readers in ``mufinder.services.quantum_results``."""

from __future__ import annotations

import math

import pytest

from mufinder.science.quantum_phonons import mode_width_angstrom
from mufinder.services.quantum_results import (
    W1_DEFINITION,
    mode_widths_from_result_data,
    w1_from_result_data,
)

_MODES = [
    {"mode_index": 4, "frequency_cm": 1600.0, "participation": 0.9, "zpe_eV": 0.099},
    {"mode_index": 5, "frequency_cm": 2500.0, "participation": 0.8, "zpe_eV": 0.155},
    {"mode_index": 6, "frequency_cm": 3600.0, "participation": 0.7, "zpe_eV": 0.223},
]


def test_w1_definition_names_participation_minus_three() -> None:
    assert W1_DEFINITION == "participation_minus_3"


def test_w1_from_result_data_returns_a_current_witness_as_stored() -> None:
    data = {
        "muon_zpe_modes": _MODES,
        "w1_entanglement_witness": -0.25,
        "w1_definition": W1_DEFINITION,
    }

    assert w1_from_result_data(data) == -0.25


def test_w1_from_result_data_recomputes_a_witness_stored_before_its_definition() -> None:
    data = {"muon_zpe_modes": _MODES, "w1_entanglement_witness": -math.sqrt(3.0 - 2.4)}

    assert w1_from_result_data(data) == pytest.approx(2.4 - 3.0)


def test_w1_from_result_data_recomputes_even_without_a_stored_witness() -> None:
    assert w1_from_result_data({"muon_zpe_modes": _MODES}) == pytest.approx(-0.6)


@pytest.mark.parametrize(
    "data",
    [
        {"w1_entanglement_witness": -0.5},
        {"w1_entanglement_witness": -0.5, "muon_zpe_modes": []},
        {"w1_entanglement_witness": -0.5, "muon_zpe_modes": "not a list"},
        {"w1_entanglement_witness": -0.5, "muon_zpe_modes": [{"mode_index": 4}]},
    ],
    ids=["no-modes", "empty-modes", "malformed-modes", "modes-without-participation"],
)
def test_w1_from_result_data_keeps_an_old_witness_it_cannot_recompute(data: dict) -> None:
    assert w1_from_result_data(data) == -0.5


def test_w1_from_result_data_is_none_without_a_witness() -> None:
    assert w1_from_result_data({}) is None
    assert (
        w1_from_result_data({"w1_entanglement_witness": "n/a", "w1_definition": W1_DEFINITION})
        is None
    )


def test_mode_widths_from_result_data_prefers_the_stored_width() -> None:
    modes = [dict(mode, width_A=0.1 * index) for index, mode in enumerate(_MODES, start=1)]

    assert mode_widths_from_result_data({"muon_zpe_modes": modes}) == pytest.approx((0.1, 0.2, 0.3))


def test_mode_widths_from_result_data_computes_a_missing_width_from_the_frequency() -> None:
    modes = [*_MODES, {"mode_index": 1, "frequency_cm": -50.0}, {"mode_index": 2}]

    widths = mode_widths_from_result_data({"muon_zpe_modes": modes})

    assert widths == pytest.approx(
        tuple(mode_width_angstrom(mode["frequency_cm"]) for mode in _MODES) + (0.0, 0.0)
    )


def test_mode_widths_from_result_data_is_empty_without_modes() -> None:
    assert mode_widths_from_result_data({}) == ()
    assert mode_widths_from_result_data({"muon_zpe_modes": None}) == ()
