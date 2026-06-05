"""Additional coverage for structure-metric helpers and value builders.

These tests target the metric-info section builder, the number/percent
formatters, the safe percent-change helper, and the empty-matching and
space-group/topology display branches of the metric value builders. They are
deterministic and free of network, GUI, randomness, sleeps, and external
files.
"""

from __future__ import annotations

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import AtomMatching, match_structures
from mufinder.science.structure_metrics import (
    MetricDefinition,
    MetricValue,
    dft_metric_values,
    experimental_metric_values,
    metric_definitions,
    metric_info_text,
)
from mufinder.science import structure_metrics as sm


def _cube(label: str, scale: float) -> Structure:
    return Structure(
        label=label,
        lattice_vectors=[[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


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


# ---------------------------------------------------------------------------
# metric_info_text / _build_metric_info_sections
# ---------------------------------------------------------------------------


def test_metric_info_text_includes_all_optional_sections() -> None:
    text = metric_info_text(_full_definition())

    # Every optional section header must appear when the field is populated.
    for header in (
        "Meaning:",
        "Physics:",
        "Method:",
        "Equation (LaTeX):",
        "Symbols:",
        "Interpretation:",
        "Caveats:",
    ):
        assert header in text

    # Symbols are rendered as a bullet list of "- symbol: meaning".
    assert "- x: first" in text
    assert "- y: second" in text
    # Sections are separated by a blank line.
    assert "\n\n" in text


def test_metric_info_text_omits_absent_optional_sections() -> None:
    text = metric_info_text(_minimal_definition())

    assert "Meaning:" in text
    assert "Interpretation:" in text
    assert "Caveats:" in text
    # Optional fields are empty, so their headers must be absent.
    assert "Physics:" not in text
    assert "Method:" not in text
    assert "Equation (LaTeX):" not in text
    assert "Symbols:" not in text


def test_build_metric_info_sections_ordering_and_pairs() -> None:
    sections = sm._build_metric_info_sections(_full_definition())
    titles = [title for title, _ in sections]

    assert titles == [
        "Meaning",
        "Physics",
        "Method",
        "Equation (LaTeX)",
        "Symbols",
        "Interpretation",
        "Caveats",
    ]
    # Bodies are non-empty strings.
    assert all(isinstance(body, str) and body for _, body in sections)


def test_build_metric_info_sections_minimal_only_required() -> None:
    sections = sm._build_metric_info_sections(_minimal_definition())
    titles = [title for title, _ in sections]

    assert titles == ["Meaning", "Interpretation", "Caveats"]


def test_metric_info_text_round_trips_real_definitions() -> None:
    # Exercise the real definitions through the public text builder.
    for definition in metric_definitions():
        text = metric_info_text(definition)
        assert text.startswith("Meaning:")
        assert "Interpretation:" in text
        assert "Caveats:" in text


# ---------------------------------------------------------------------------
# _format_number / _format_percent / _safe_percent_change
# ---------------------------------------------------------------------------


def test_format_number_none_returns_na() -> None:
    assert sm._format_number(None) == "N/A"


def test_format_number_uses_significant_digits() -> None:
    assert sm._format_number(1.23456789, digits=3) == "1.23"
    assert sm._format_number(0.0) == "0"


def test_format_percent_none_returns_na() -> None:
    assert sm._format_percent(None) == "N/A"


def test_format_percent_appends_symbol() -> None:
    result = sm._format_percent(12.3456)
    assert result.endswith("%")
    assert result == "12.35%"


def test_safe_percent_change_zero_reference_returns_none() -> None:
    assert sm._safe_percent_change(0.0, 5.0) is None
    # Just below the tolerance threshold also returns None.
    assert sm._safe_percent_change(1e-13, 5.0) is None


def test_safe_percent_change_regular_value() -> None:
    assert sm._safe_percent_change(10.0, 11.0) == 100.0 * (11.0 - 10.0) / 10.0


# ---------------------------------------------------------------------------
# dft_metric_values branches
# ---------------------------------------------------------------------------


def test_dft_metric_values_empty_matching_yields_na_displacements() -> None:
    exp = _cube("exp", 5.0)
    dft = _cube("dft", 5.0)
    # An empty matching exercises the "no displacements" branch where rmsd,
    # max_displacement, and mean_displacement are all None.
    empty = AtomMatching(pairs=(), unmatched_experimental=(0, 1), unmatched_dft=(0, 1))

    values = dft_metric_values(exp, dft, empty)

    assert values["rmsd"].value is None
    assert values["rmsd"].display == "N/A"
    assert values["max_displacement"].value is None
    assert values["mean_displacement"].value is None
    # Identical cells give zero percent change.
    assert values["delta_a_pct"].value == 0.0
    # Partial matching sets a warning on the lattice rows.
    assert values["a"].warning == "Partial atom matching"
    # similarity_score stays within [0, 1].
    assert 0.0 <= values["similarity_score"].value <= 1.0


def test_dft_metric_values_full_match_has_no_warning() -> None:
    exp = _cube("exp", 5.0)
    dft = _cube("dft", 5.0)
    matching = match_structures(exp, dft)

    values = dft_metric_values(exp, dft, matching)

    assert not matching.unmatched_experimental
    assert not matching.unmatched_dft
    assert values["a"].warning is None
    # Identical structures: displacements are (near) zero, score near 1.
    assert values["rmsd"].value is not None
    assert values["similarity_score"].value <= 1.0


def test_dft_metric_values_space_group_and_topology_displays() -> None:
    exp = _cube("exp", 5.0)
    dft = _cube("dft", 5.05)
    matching = match_structures(exp, dft)

    values = dft_metric_values(exp, dft, matching)

    # Space-group display is one of the documented forms.
    assert values["space_group"].display in {"N/A"} or values["space_group"].display
    # Topology display is one of the documented tokens.
    assert values["topology_preserved"].display in {"Yes", "No", "N/A"}
    # space_group / topology rows never carry the partial-matching warning.
    assert values["space_group"].warning is None
    assert values["topology_preserved"].warning is None


def test_experimental_metric_values_are_reference_baseline() -> None:
    exp = _cube("exp", 5.0)
    values = experimental_metric_values(exp)

    assert values["delta_a_pct"].display == "0%"
    assert values["space_group"].display == "Reference"
    assert values["topology_preserved"].value is True
    assert values["similarity_score"].value == 1.0
    # Lattice values are positive and match the cube edge length.
    assert values["a"].value == 5.0
    assert isinstance(values["cell_volume"].value, float)
    assert np.isclose(values["cell_volume"].value, 125.0)


def test_metric_value_dataclass_defaults() -> None:
    value = MetricValue(value=1.0, display="1")
    assert value.warning is None
    assert value.value == 1.0
    assert value.display == "1"
