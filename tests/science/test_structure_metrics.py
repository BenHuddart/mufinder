"""Tests for comparison metric definitions and value builders."""

from __future__ import annotations

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import match_structures
from mufinder.science.structure_metrics import dft_metric_values, experimental_metric_values, metric_definitions


def _exp() -> Structure:
    return Structure(
        label="exp",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _dft() -> Structure:
    return Structure(
        label="dft",
        lattice_vectors=[[5.1, 0.0, 0.0], [0.0, 4.95, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.252, 0.248, 0.25]],
    )


def test_metric_definitions_include_expected_keys() -> None:
    keys = {definition.key for definition in metric_definitions()}

    assert {"a", "b", "c", "alpha", "beta", "gamma"} <= keys
    assert {"rmsd", "mean_displacement", "space_group", "similarity_score"} <= keys


def test_metric_value_builders_produce_required_rows() -> None:
    matching = match_structures(_exp(), _dft())
    exp_values = experimental_metric_values(_exp())
    dft_values = dft_metric_values(_exp(), _dft(), matching)

    assert exp_values["a"].display
    assert exp_values["delta_a_pct"].display == "0%"
    assert dft_values["delta_a_pct"].display.endswith("%")
    assert dft_values["rmsd"].display
    assert dft_values["similarity_score"].value is not None
