"""Tests for comparison metric definitions, info text and value builders."""

from __future__ import annotations

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science import structure_metrics as sm
from mufinder.science.atom_matching import AtomMatching, match_structures
from mufinder.science.structure_metrics import (
    MetricDefinition,
    MetricValue,
    dft_metric_values,
    experimental_metric_values,
    metric_definitions,
    metric_info_text,
)
from tests.support.structures import QUARTER_DIAGONAL, host_structure


def _cube(label: str, scale: float) -> Structure:
    return host_structure(label=label, lattice=scale, positions=QUARTER_DIAGONAL)


def _exp() -> Structure:
    return _cube("exp", 5.0)


def _dft() -> Structure:
    return host_structure(
        label="dft",
        lattice=[[5.1, 0.0, 0.0], [0.0, 4.95, 0.0], [0.0, 0.0, 5.0]],
        positions=((0.0, 0.0, 0.0), (0.252, 0.248, 0.25)),
    )


def _empty_matching() -> AtomMatching:
    return AtomMatching(pairs=(), unmatched_experimental=(0, 1), unmatched_dft=(0, 1))


def _full_definition() -> MetricDefinition:
    return MetricDefinition(
        key="demo",
        label="Demo",
        unit="Ang",
        group="Demo Group",
        meaning="A demo meaning.",
        calculation="calc",
        interpretation="An interpretation.",
        caveats="Some caveats.",
        physics="The physics line.",
        method="The method line.",
        equation=r"x = y",
        symbols=((r"x", "first"), (r"y", "second")),
    )


def _minimal_definition() -> MetricDefinition:
    return MetricDefinition(
        key="bare",
        label="Bare",
        unit="",
        group="Bare Group",
        meaning="Bare meaning.",
        calculation="calc",
        interpretation="Bare interpretation.",
        caveats="Bare caveats.",
    )


# metric_definitions and metric_info_text


def test_metric_definitions_include_expected_keys() -> None:
    keys = {definition.key for definition in metric_definitions()}

    assert {"a", "b", "c", "alpha", "beta", "gamma"} <= keys
    assert {"rmsd", "mean_displacement", "space_group", "similarity_score"} <= keys


def test_metric_info_text_lists_every_populated_section_in_order() -> None:
    text = metric_info_text(_full_definition())

    headers = [
        "Meaning:",
        "Physics:",
        "Method:",
        "Equation (LaTeX):",
        "Symbols:",
        "Interpretation:",
        "Caveats:",
    ]
    positions = [text.index(header) for header in headers]
    assert positions == sorted(positions)
    # Symbols are rendered as a bullet list of "- symbol: meaning".
    assert "- x: first" in text
    assert "- y: second" in text
    # Sections are separated by a blank line.
    assert text.count("\n\n") == len(headers) - 1


def test_metric_info_text_omits_absent_optional_sections() -> None:
    text = metric_info_text(_minimal_definition())

    assert text == "Meaning:\nBare meaning.\n\nInterpretation:\nBare interpretation.\n\nCaveats:\nBare caveats."


def test_metric_info_text_round_trips_real_definitions() -> None:
    for definition in metric_definitions():
        text = metric_info_text(definition)
        assert text.startswith("Meaning:")
        assert "Interpretation:" in text
        assert "Caveats:" in text


def test_metric_value_dataclass_defaults() -> None:
    value = MetricValue(value=1.0, display="1")
    assert value.warning is None
    assert value.value == 1.0
    assert value.display == "1"


# experimental_metric_values


def test_experimental_metric_values_are_reference_baseline() -> None:
    values = experimental_metric_values(_exp())

    assert values["delta_a_pct"].display == "0%"
    assert values["space_group"].display == "Reference"
    assert values["topology_preserved"].value is True
    assert values["similarity_score"].value == 1.0
    # Lattice values match the cube edge length.
    assert values["a"].value == 5.0
    assert isinstance(values["cell_volume"].value, float)
    assert np.isclose(values["cell_volume"].value, 125.0)


# dft_metric_values


def test_metric_value_builders_produce_required_rows() -> None:
    matching = match_structures(_exp(), _dft())
    exp_values = experimental_metric_values(_exp())
    dft_values = dft_metric_values(_exp(), _dft(), matching)

    assert exp_values["a"].display
    assert exp_values["delta_a_pct"].display == "0%"
    assert dft_values["delta_a_pct"].display.endswith("%")
    assert dft_values["rmsd"].display
    assert dft_values["similarity_score"].value is not None


def test_dft_metric_values_format_numbers_and_percent_changes() -> None:
    # Six significant digits for lattice values, four for percent changes.
    values = dft_metric_values(_exp(), _cube("dft", 5.6172839), _empty_matching())

    assert values["a"].display == "5.61728"
    assert values["delta_a_pct"].display == "12.35%"
    assert values["delta_a_pct"].value == 100.0 * (5.6172839 - 5.0) / 5.0


def test_dft_metric_values_empty_matching_yields_na_displacements() -> None:
    values = dft_metric_values(_exp(), _cube("dft", 5.0), _empty_matching())

    assert values["rmsd"].value is None
    assert values["rmsd"].display == "N/A"
    assert values["max_displacement"].value is None
    assert values["mean_displacement"].value is None
    # Identical cells give zero percent change.
    assert values["delta_a_pct"].value == 0.0
    assert values["delta_a_pct"].display == "0%"
    # Partial matching sets a warning on the lattice rows.
    assert values["a"].warning == "Partial atom matching"
    assert 0.0 <= values["similarity_score"].value <= 1.0


def test_dft_metric_values_full_match_has_no_warning() -> None:
    exp = _cube("exp", 5.0)
    dft = _cube("dft", 5.0)
    matching = match_structures(exp, dft)

    values = dft_metric_values(exp, dft, matching)

    assert not matching.unmatched_experimental
    assert not matching.unmatched_dft
    assert values["a"].warning is None
    assert values["rmsd"].value is not None
    assert values["similarity_score"].value <= 1.0


def test_dft_metric_values_space_group_and_topology_displays() -> None:
    exp = _cube("exp", 5.0)
    dft = _cube("dft", 5.05)
    matching = match_structures(exp, dft)

    values = dft_metric_values(exp, dft, matching)

    assert values["space_group"].display
    assert values["topology_preserved"].display in {"Yes", "No", "N/A"}
    # space_group / topology rows never carry the partial-matching warning.
    assert values["space_group"].warning is None
    assert values["topology_preserved"].warning is None


def test_percent_change_against_zero_reference_is_undefined_and_shown_as_na() -> None:
    # Private helpers: lattice_parameters rejects a zero-length reference axis before dft_metric_values gets here.
    assert sm._safe_percent_change(0.0, 5.0) is None
    assert sm._safe_percent_change(1e-13, 5.0) is None
    assert sm._safe_percent_change(10.0, 11.0) == 100.0 * (11.0 - 10.0) / 10.0
    assert sm._format_percent(None) == "N/A"
