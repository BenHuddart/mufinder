"""Tests for the convergence sweep key codecs and configuration."""

from __future__ import annotations

from dataclasses import fields

import pytest

from mufinder.app.convergence_types import (
    CUTOFF_KEY_CODEC,
    KPOINT_GRID_CODEC,
    ConvergenceConfig,
    parse_kpoint_grid,
)


def test_cutoff_codec_round_trips_table_text_and_state() -> None:
    assert CUTOFF_KEY_CODEC.parse("300") == 300.0
    assert CUTOFF_KEY_CODEC.format(300.0) == "300"
    assert CUTOFF_KEY_CODEC.describe(412.5) == "412.5 eV"
    assert CUTOFF_KEY_CODEC.to_state(300) == 300.0
    assert CUTOFF_KEY_CODEC.normalize("360") == 360.0


def test_cutoff_codec_coerce_rejects_non_numbers() -> None:
    assert CUTOFF_KEY_CODEC.coerce("abc") is None
    assert CUTOFF_KEY_CODEC.coerce(None) is None
    with pytest.raises(ValueError):
        CUTOFF_KEY_CODEC.parse("abc")


@pytest.mark.parametrize("text", ["2 x 2 x 1", "2x2x1", "2, 2, 1", " 2 2 1 "])
def test_kpoint_grid_parse_accepts_any_separator(text: str) -> None:
    assert parse_kpoint_grid(text) == (2, 2, 1)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("2 x 2", "exactly three integers"),
        ("2 x 2 x 2 x 2", "exactly three integers"),
        ("0 x 2 x 2", "positive integers"),
    ],
)
def test_kpoint_grid_parse_rejects_malformed_grids(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_kpoint_grid(text)


def test_kpoint_codec_normalises_lists_and_tuples_and_writes_lists() -> None:
    assert KPOINT_GRID_CODEC.normalize([3, 3, 2]) == (3, 3, 2)
    assert KPOINT_GRID_CODEC.normalize((3.0, 3.0, 2.0)) == (3, 3, 2)
    assert KPOINT_GRID_CODEC.format((3, 3, 2)) == "3 x 3 x 2"
    assert KPOINT_GRID_CODEC.describe((3, 3, 2)) == "3 x 3 x 2"
    assert KPOINT_GRID_CODEC.to_state((3, 3, 2)) == [3, 3, 2]


@pytest.mark.parametrize("raw", ["3x3x2", [3, 3], [3, 0, 2], [3, "a", 2], None])
def test_kpoint_codec_coerce_rejects_what_is_not_a_grid(raw: object) -> None:
    assert KPOINT_GRID_CODEC.coerce(raw) is None


def test_convergence_config_holds_every_sweep_in_sweep_values() -> None:
    cutoff = ConvergenceConfig(sweep_values=(300.0, 400.0))
    kpoint = ConvergenceConfig(sweep_parameter="kpoint_mp_grid", sweep_values=((2, 2, 2),))

    assert cutoff.sweep_values == (300.0, 400.0)
    assert kpoint.sweep_values == ((2, 2, 2),)
    assert "cutoff_energies" not in {field.name for field in fields(ConvergenceConfig)}
    with pytest.raises(TypeError):
        ConvergenceConfig(cutoff_energies=(300.0,))  # type: ignore[call-arg]


def test_convergence_config_rejects_an_unknown_sweep_parameter() -> None:
    with pytest.raises(ValueError, match="sweep_parameter"):
        ConvergenceConfig(sweep_parameter="smearing_width")
