"""Tests for comparison metric definitions, info text and value builders."""

from __future__ import annotations

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science import structure_metrics as sm
from mufinder.science.atom_matching import AtomMatching, match_structures
from mufinder.science.supercell_matching import tile_to_supercell
from mufinder.science.symmetry_analysis import SpaceGroupResult
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


def test_metric_definitions_are_lattice_displacement_and_symmetry_rows() -> None:
    definitions = metric_definitions()

    assert [definition.key for definition in definitions] == [
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "cell_volume",
        "rmsd",
        "max_displacement",
        "space_group",
        "space_group_match",
    ]
    assert [definition.group for definition in definitions] == (
        ["Lattice parameters"] * 7 + ["Atomic displacements"] * 2 + ["Symmetry"] * 2
    )
    assert [(definition.label, definition.unit) for definition in definitions[:9]] == [
        ("a", "Å"),
        ("b", "Å"),
        ("c", "Å"),
        ("α", "°"),
        ("β", "°"),
        ("γ", "°"),
        ("V", "Å³"),
        ("RMSD", "Å"),
        ("Max displacement", "Å"),
    ]


def test_metric_meanings_are_one_line_tooltip_sentences() -> None:
    for definition in metric_definitions():
        assert definition.meaning.endswith("."), definition.key
        assert "\n" not in definition.meaning
        assert definition.meaning[0].isupper()


def test_dropped_metrics_are_not_computed() -> None:
    matching = match_structures(_exp(), _dft())
    keys = set(dft_metric_values(_exp(), _dft(), matching)) | set(
        experimental_metric_values(_exp())
    )

    assert keys == {definition.key for definition in metric_definitions()}
    for dropped in (
        "similarity_score",
        "mean_displacement",
        "delta_a_pct",
        "coordination_changes",
        "topology_preserved",
    ):
        assert dropped not in keys


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

    assert (
        text
        == "Meaning:\nBare meaning.\n\nInterpretation:\nBare interpretation.\n\nCaveats:\nBare caveats."
    )


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

    assert values["a"].value == 5.0
    assert values["a"].display == "5"
    assert isinstance(values["cell_volume"].value, float)
    assert np.isclose(values["cell_volume"].value, 125.0)
    assert values["rmsd"].display == "0"
    # The experimental column names its own space group.
    assert values["space_group"].display == "R-3m (166)"
    assert values["space_group_match"].display == "Reference"


# dft_metric_values


def test_dft_lattice_values_carry_their_change_against_experiment() -> None:
    values = dft_metric_values(_exp(), _cube("dft", 5.015), _empty_matching())

    assert values["a"].value == 5.015
    assert values["a"].display == "5.015 (+0.30 %)"
    assert values["alpha"].display == "90 (+0.00 %)"
    assert values["cell_volume"].display.endswith("(+0.90 %)")

    shrunk = dft_metric_values(_exp(), _cube("dft", 4.9), _empty_matching())
    assert shrunk["c"].display == "4.9 (-2.00 %)"


def test_dft_metric_values_format_lattice_values_to_six_significant_digits() -> None:
    values = dft_metric_values(_exp(), _cube("dft", 5.6172839), _empty_matching())

    assert values["a"].display == "5.61728 (+12.35 %)"


def test_dft_metric_values_empty_matching_yields_na_displacements() -> None:
    values = dft_metric_values(_exp(), _cube("dft", 5.0), _empty_matching())

    assert values["rmsd"].value is None
    assert values["rmsd"].display == "N/A"
    assert values["max_displacement"].value is None
    assert values["max_displacement"].display == "N/A"
    # Partial matching sets a warning on the lattice and displacement rows.
    assert values["a"].warning == "Partial atom matching"
    assert values["rmsd"].warning == "Partial atom matching"


def test_dft_metric_values_full_match_has_no_warning() -> None:
    exp = _cube("exp", 5.0)
    dft = _cube("dft", 5.0)
    matching = match_structures(exp, dft)

    values = dft_metric_values(exp, dft, matching)

    assert not matching.unmatched_experimental
    assert not matching.unmatched_dft
    assert values["a"].warning is None
    assert values["rmsd"].value == 0.0
    assert values["max_displacement"].value == 0.0


def test_dft_metric_values_report_rmsd_and_max_displacement() -> None:
    matching = match_structures(_exp(), _dft())
    values = dft_metric_values(_exp(), _dft(), matching)

    displacements = [pair.displacement for pair in matching.pairs]
    assert values["rmsd"].value == float(np.sqrt(np.mean(np.square(displacements))))
    assert values["max_displacement"].value == max(displacements)
    assert values["max_displacement"].value >= values["rmsd"].value > 0.0


def test_dft_metric_values_name_the_space_group_and_whether_it_matches() -> None:
    exp = _cube("exp", 5.0)
    kept = dft_metric_values(exp, _cube("dft", 5.05), match_structures(exp, _cube("dft", 5.05)))

    assert kept["space_group"].display == "R-3m (166)"
    assert kept["space_group_match"].display == "Yes"
    assert kept["space_group_match"].value is True
    # The symmetry rows never carry the partial-matching warning.
    assert kept["space_group"].warning is None
    assert kept["space_group_match"].warning is None

    tetragonal = host_structure(
        label="dft", lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.4]], positions=QUARTER_DIAGONAL
    )
    broken = dft_metric_values(exp, tetragonal, match_structures(exp, tetragonal))
    assert broken["space_group_match"].display == "No"
    assert broken["space_group_match"].value is False
    assert broken["space_group"].display != kept["space_group"].display


def test_space_group_symbols_show_screw_axes_as_subscripts() -> None:
    group = SpaceGroupResult(number=136, symbol="P4_2/mnm", hall_symbol=None, available=True)

    assert sm._format_space_group(group) == "P4₂/mnm (136)"
    assert (
        sm._format_space_group(
            SpaceGroupResult(number=19, symbol="P2_12_12_1", hall_symbol=None, available=True)
        )
        == "P2₁2₁2₁ (19)"
    )
    unavailable = SpaceGroupResult(number=None, symbol=None, hall_symbol=None, available=False)
    assert sm._format_space_group(unavailable) == "N/A"


def test_percent_change_against_zero_reference_is_undefined_and_shown_without_change() -> None:
    # Private helpers: lattice_parameters rejects a zero-length reference axis before dft_metric_values gets here.
    assert sm._safe_percent_change(0.0, 5.0) is None
    assert sm._safe_percent_change(1e-13, 5.0) is None
    assert sm._safe_percent_change(10.0, 11.0) == 100.0 * (11.0 - 10.0) / 10.0
    assert sm._format_with_change(5.0, 0.0) == "5"


def test_a_change_that_rounds_to_zero_is_never_negative() -> None:
    assert sm._format_with_change(89.99999, 90.0) == "90 (+0.00 %)"
    assert sm._format_with_change(90.0, 90.0) == "90 (+0.00 %)"
    assert sm._format_with_change(89.99, 90.0) == "89.99 (-0.01 %)"


def _strained_supercell(matrix: tuple[tuple[int, int, int], ...], strain: np.ndarray) -> Structure:
    """The ``_exp`` cube tiled to ``matrix``, its Cartesian frame strained by ``strain``."""

    tiled = tile_to_supercell(_exp(), matrix)
    tiled.lattice_vectors = (np.asarray(tiled.lattice_vectors) @ strain).tolist()
    return tiled


def test_a_2x2x2_supercell_column_reads_per_host_cell() -> None:
    matrix = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    dft = _strained_supercell(matrix, np.eye(3) * 1.003)

    values = dft_metric_values(
        tile_to_supercell(_exp(), matrix),
        dft,
        _empty_matching(),
        host=_exp(),
        supercell_matrix=matrix,
    )

    assert np.isclose(values["a"].value, 5.015)
    assert values["a"].display == "5.015 (+0.30 %)"
    assert values["gamma"].display == "90 (+0.00 %)"
    assert np.isclose(values["cell_volume"].value, 125.0 * 1.003**3)
    assert values["cell_volume"].display.endswith("(+0.90 %)")


def test_a_non_diagonal_supercell_column_reads_the_host_cell_it_implies() -> None:
    # A sqrt(2) x sqrt(2) x 1 cell rotated 45° from the host axes, strained 1 %
    # along the host a only. Per supercell vector a would read +0.50 %.
    matrix = ((1, 1, 0), (-1, 1, 0), (0, 0, 1))
    dft = _strained_supercell(matrix, np.diag([1.01, 1.0, 1.0]))

    values = dft_metric_values(
        tile_to_supercell(_exp(), matrix),
        dft,
        _empty_matching(),
        host=_exp(),
        supercell_matrix=matrix,
    )

    assert values["a"].display == "5.05 (+1.00 %)"
    assert values["b"].display == "5 (+0.00 %)"
    assert values["c"].display == "5 (+0.00 %)"
    assert values["gamma"].display == "90 (+0.00 %)"
    assert values["cell_volume"].display == "126.25 (+1.00 %)"
