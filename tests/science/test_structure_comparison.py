"""Tests for end-to-end structure comparison orchestration."""

from __future__ import annotations

from mufinder.domain.models import Structure
from mufinder.science.structure_comparison import DftComparisonInput, compare_structures


def _exp() -> Structure:
    return Structure(
        label="exp",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def _dft() -> Structure:
    return Structure(
        label="dft",
        lattice_vectors=[[5.2, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.01, 0.0, 0.0]],
    )


def test_compare_structures_builds_metric_payload() -> None:
    experimental = _exp()
    dft = _dft()

    result = compare_structures(
        experimental,
        [DftComparisonInput(label="DFT-1", structure=dft, structure_id=dft.id)],
    )

    assert result.definitions
    assert result.experimental_values["a"].display
    assert len(result.dft_columns) == 1
    assert result.dft_columns[0].values["delta_a_pct"].display.endswith("%")
