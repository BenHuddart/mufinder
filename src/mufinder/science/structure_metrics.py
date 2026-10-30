"""Configuration-driven metric definitions for structure comparison."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import AtomMatching
from mufinder.science.kpoint_grids import lattice_parameters
from mufinder.science.spglib_compat import SYMMETRY_TOLERANCE_ANG
from mufinder.science.symmetry_analysis import (
    SpaceGroupResult,
    compare_space_group,
    detect_space_group,
)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """One table metric definition."""

    key: str
    label: str
    unit: str
    group: str
    meaning: str
    calculation: str
    interpretation: str
    caveats: str
    physics: str = ""
    method: str = ""
    equation: str = ""
    symbols: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class MetricValue:
    """Rendered metric payload."""

    value: Any
    display: str
    warning: str | None = None


_LATTICE_METHOD = (
    "Computed on the normalized comparison structures after optional experimental tiling to a "
    "detected integer supercell. These quantities are read directly from the comparison-cell "
    "lattice rather than from a fitted strain tensor."
)

_MATCHING_METHOD = (
    "Atoms are matched after optional supercell normalization. Matching is performed separately "
    "within each chemical species by minimizing the periodic minimum-image distance matrix with a "
    "Hungarian assignment. Displacements therefore depend on both periodic wrapping and the matched "
    "atom correspondence."
)

_SYMMETRY_METHOD = (
    "Space groups are identified with spglib on the normalized comparison structures, using "
    f"symprec = {SYMMETRY_TOLERANCE_ANG:g} Å, and compared by their international numbers."
)


def _build_metric_info_sections(definition: MetricDefinition) -> list[tuple[str, str]]:
    sections = [("Meaning", definition.meaning)]
    if definition.physics:
        sections.append(("Physics", definition.physics))
    if definition.method:
        sections.append(("Method", definition.method))
    if definition.equation:
        sections.append(("Equation (LaTeX)", definition.equation))
    if definition.symbols:
        sections.append(
            (
                "Symbols",
                "\n".join(f"- {symbol}: {meaning}" for symbol, meaning in definition.symbols),
            )
        )
    sections.append(("Interpretation", definition.interpretation))
    sections.append(("Caveats", definition.caveats))
    return sections


def metric_info_text(definition: MetricDefinition) -> str:
    """Build the info-dialog content for a metric."""

    return "\n\n".join(
        f"{title}:\n{body}" for title, body in _build_metric_info_sections(definition)
    )


_METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        key="a",
        label="a",
        unit="Å",
        group="Lattice parameters",
        meaning="Length of the first lattice vector, with the change against experiment for each DFT column.",
        calculation="Computed from |a_vector|.",
        interpretation="Closer values imply less lattice distortion.",
        caveats="Sensitive to supercell scaling.",
        physics="Tracks axial expansion or contraction along the first direct-lattice basis vector.",
        method=_LATTICE_METHOD,
        equation=r"a = \lVert \mathbf{a} \rVert_2",
        symbols=(
            (r"a", "Length of the first direct-lattice basis vector."),
            (r"\mathbf{a}", "First direct-lattice basis vector of the normalized comparison cell."),
        ),
    ),
    MetricDefinition(
        key="b",
        label="b",
        unit="Å",
        group="Lattice parameters",
        meaning="Length of the second lattice vector, with the change against experiment for each DFT column.",
        calculation="Computed from |b_vector|.",
        interpretation="Closer values imply less lattice distortion.",
        caveats="Sensitive to supercell scaling.",
        physics="Tracks axial expansion or contraction along the second direct-lattice basis vector.",
        method=_LATTICE_METHOD,
        equation=r"b = \lVert \mathbf{b} \rVert_2",
        symbols=(
            (r"b", "Length of the second direct-lattice basis vector."),
            (
                r"\mathbf{b}",
                "Second direct-lattice basis vector of the normalized comparison cell.",
            ),
        ),
    ),
    MetricDefinition(
        key="c",
        label="c",
        unit="Å",
        group="Lattice parameters",
        meaning="Length of the third lattice vector, with the change against experiment for each DFT column.",
        calculation="Computed from |c_vector|.",
        interpretation="Closer values imply less lattice distortion.",
        caveats="Sensitive to supercell scaling.",
        physics="Tracks axial expansion or contraction along the third direct-lattice basis vector.",
        method=_LATTICE_METHOD,
        equation=r"c = \lVert \mathbf{c} \rVert_2",
        symbols=(
            (r"c", "Length of the third direct-lattice basis vector."),
            (r"\mathbf{c}", "Third direct-lattice basis vector of the normalized comparison cell."),
        ),
    ),
    MetricDefinition(
        key="alpha",
        label="α",
        unit="°",
        group="Lattice parameters",
        meaning="Angle between the b and c lattice vectors, with the change against experiment for each DFT column.",
        calculation="Derived from lattice dot products.",
        interpretation="Stable values indicate low shear distortion.",
        caveats="Angles can be noisy for low-symmetry cells.",
        physics="Measures cell-shape shear involving the second and third basis directions.",
        method=_LATTICE_METHOD,
        equation=(
            r"\alpha = \cos^{-1}\left("
            r"\frac{\mathbf{b}\cdot\mathbf{c}}{\lVert \mathbf{b} \rVert_2 \lVert \mathbf{c} \rVert_2}"
            r"\right)"
        ),
        symbols=(
            (r"\alpha", "Angle between the second and third lattice vectors."),
            (r"\mathbf{b}, \mathbf{c}", "Second and third direct-lattice basis vectors."),
        ),
    ),
    MetricDefinition(
        key="beta",
        label="β",
        unit="°",
        group="Lattice parameters",
        meaning="Angle between the a and c lattice vectors, with the change against experiment for each DFT column.",
        calculation="Derived from lattice dot products.",
        interpretation="Stable values indicate low shear distortion.",
        caveats="Angles can be noisy for low-symmetry cells.",
        physics="Measures cell-shape shear involving the first and third basis directions.",
        method=_LATTICE_METHOD,
        equation=(
            r"\beta = \cos^{-1}\left("
            r"\frac{\mathbf{a}\cdot\mathbf{c}}{\lVert \mathbf{a} \rVert_2 \lVert \mathbf{c} \rVert_2}"
            r"\right)"
        ),
        symbols=(
            (r"\beta", "Angle between the first and third lattice vectors."),
            (r"\mathbf{a}, \mathbf{c}", "First and third direct-lattice basis vectors."),
        ),
    ),
    MetricDefinition(
        key="gamma",
        label="γ",
        unit="°",
        group="Lattice parameters",
        meaning="Angle between the a and b lattice vectors, with the change against experiment for each DFT column.",
        calculation="Derived from lattice dot products.",
        interpretation="Stable values indicate low shear distortion.",
        caveats="Angles can be noisy for low-symmetry cells.",
        physics="Measures cell-shape shear involving the first and second basis directions.",
        method=_LATTICE_METHOD,
        equation=(
            r"\gamma = \cos^{-1}\left("
            r"\frac{\mathbf{a}\cdot\mathbf{b}}{\lVert \mathbf{a} \rVert_2 \lVert \mathbf{b} \rVert_2}"
            r"\right)"
        ),
        symbols=(
            (r"\gamma", "Angle between the first and second lattice vectors."),
            (r"\mathbf{a}, \mathbf{b}", "First and second direct-lattice basis vectors."),
        ),
    ),
    MetricDefinition(
        key="cell_volume",
        label="V",
        unit="Å³",
        group="Lattice parameters",
        meaning="Volume of the comparison cell, with the change against experiment for each DFT column.",
        calculation="Absolute determinant of lattice matrix.",
        interpretation="Large changes often indicate over/under relaxation.",
        caveats="Volume scales with supercell size.",
        physics="Captures isotropic expansion or contraction of the comparison cell and any coupled shear contributions.",
        method=_LATTICE_METHOD,
        equation=r"V = |\det L|",
        symbols=(
            (r"V", "Cell volume of the normalized comparison structure."),
            (r"L", "3x3 lattice matrix whose rows are the direct-lattice basis vectors."),
        ),
    ),
    MetricDefinition(
        key="rmsd",
        label="RMSD",
        unit="Å",
        group="Atomic displacements",
        meaning="Root-mean-square displacement of the matched atoms from their experimental positions.",
        calculation="sqrt(mean(d_i^2)).",
        interpretation="Lower is better; captures average structural drift.",
        caveats="Depends on atom matching quality.",
        physics="Measures the characteristic amplitude of atomic relaxation after periodic wrapping and species-resolved atom correspondence have been accounted for.",
        method=_MATCHING_METHOD,
        equation=(
            r"\Delta \mathbf{f}_i = \mathbf{f}^{\mathrm{DFT}}_{m(i)} - \mathbf{f}^{\mathrm{exp}}_i"
            "\n"
            r"d_i = \left\lVert \left(\Delta \mathbf{f}_i - \mathrm{round}(\Delta \mathbf{f}_i)\right) L_{\mathrm{DFT}} \right\rVert_2"
            "\n"
            r"\mathrm{RMSD} = \sqrt{\frac{1}{N_{\mathrm{match}}}\sum_{i=1}^{N_{\mathrm{match}}} d_i^2}"
        ),
        symbols=(
            (r"\mathbf{f}^{\mathrm{exp}}_i", "Fractional position of experimental atom i."),
            (
                r"\mathbf{f}^{\mathrm{DFT}}_{m(i)}",
                "Fractional position of the DFT atom matched to experimental atom i.",
            ),
            (
                r"m(i)",
                "Species-preserving Hungarian assignment map from experimental to DFT atoms.",
            ),
            (
                r"L_{\mathrm{DFT}}",
                "DFT lattice matrix used to convert minimum-image fractional displacements to Cartesian vectors.",
            ),
            (r"d_i", "Minimum-image Cartesian displacement of matched atom pair i."),
            (r"N_{\mathrm{match}}", "Number of matched atom pairs."),
        ),
    ),
    MetricDefinition(
        key="max_displacement",
        label="Max displacement",
        unit="Å",
        group="Atomic displacements",
        meaning="Largest displacement of any matched atom from its experimental position.",
        calculation="max(d_i).",
        interpretation="Highlights worst local relaxation changes.",
        caveats="Single outlier can dominate.",
        physics="Isolates the single largest matched-atom relaxation, which is often useful for spotting local defects or unstable coordination environments.",
        method=_MATCHING_METHOD,
        equation=r"d_{\max} = \max_{1 \le i \le N_{\mathrm{match}}} d_i",
        symbols=(
            (r"d_i", "Minimum-image Cartesian displacement of matched atom pair i."),
            (r"d_{\max}", "Largest matched-atom displacement."),
            (r"N_{\mathrm{match}}", "Number of matched atom pairs."),
        ),
    ),
    MetricDefinition(
        key="space_group",
        label="Space group",
        unit="",
        group="Symmetry",
        meaning="Space group that spglib finds for each structure.",
        calculation="spglib identification of the international symbol and number.",
        interpretation="The DFT structure should keep the experimental space group.",
        caveats="Depends on the symmetry tolerance.",
        physics="Names the crystallographic symmetry of each periodic structure.",
        method=_SYMMETRY_METHOD,
        equation=r"n = \mathrm{spglib}(L, \{\mathbf{f}_i\}, \{Z_i\})",
        symbols=(
            (r"n", "International space-group number reported by spglib."),
            (r"L", "Lattice matrix of the structure."),
            (r"\mathbf{f}_i, Z_i", "Fractional position and species of atom i."),
        ),
    ),
    MetricDefinition(
        key="space_group_match",
        label="Same space group",
        unit="",
        group="Symmetry",
        meaning="Whether the DFT structure keeps the experimental space group.",
        calculation="Comparison of the spglib space-group numbers.",
        interpretation="Yes means the relaxation kept the experimental symmetry.",
        caveats="Depends on the symmetry tolerance.",
        physics="Captures whether the relaxed crystal retains the symmetry of the experimental reference.",
        method=_SYMMETRY_METHOD,
        equation=r"\mathrm{match}_{\mathrm{SG}} = \mathbf{1}\left[n_{\mathrm{exp}} = n_{\mathrm{DFT}}\right]",
        symbols=(
            (r"n_{\mathrm{exp}}", "Experimental space-group number reported by spglib."),
            (r"n_{\mathrm{DFT}}", "DFT space-group number reported by spglib."),
            (r"\mathrm{match}_{\mathrm{SG}}", "Boolean indicator of space-group-number equality."),
        ),
    ),
)


def metric_definitions() -> tuple[MetricDefinition, ...]:
    """Return the ordered comparison-table metric definitions."""

    return _METRIC_DEFINITIONS


def _format_number(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}g}"


def _safe_percent_change(reference: float, value: float) -> float | None:
    if abs(reference) <= 1e-12:
        return None
    return 100.0 * (value - reference) / reference


def _format_with_change(value: float, reference: float) -> str:
    """``value`` with its signed percent change against ``reference``: ``4.7091 (+0.30 %)``."""

    change = _safe_percent_change(reference, value)
    if change is None:
        return _format_number(value)
    # A change that rounds to zero reads "+0.00 %", never "-0.00 %".
    if abs(change) < 0.005:
        change = 0.0
    return f"{_format_number(value)} ({change:+.2f} %)"


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _format_space_group(group: SpaceGroupResult) -> str:
    """The spglib symbol with screw-axis subscripts and the number: ``P4₂/mnm (136)``."""

    if not group.available or group.number is None:
        return "N/A"
    symbol = re.sub(
        r"_(\d)", lambda match: match.group(1).translate(_SUBSCRIPT_DIGITS), group.symbol or ""
    )
    return f"{symbol} ({group.number})" if symbol else str(group.number)


def _structure_lattice_metrics(structure: Structure) -> dict[str, float]:
    a, b, c, alpha, beta, gamma = lattice_parameters(structure.lattice_vectors)
    matrix = np.asarray(structure.lattice_vectors, dtype=float)
    volume = abs(float(np.linalg.det(matrix)))
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "cell_volume": volume,
    }


_LATTICE_KEYS = ("a", "b", "c", "alpha", "beta", "gamma", "cell_volume")


def experimental_metric_values(experimental: Structure) -> dict[str, MetricValue]:
    """Compute metric values for the experimental reference column."""

    lattice_values = _structure_lattice_metrics(experimental)
    space_group = detect_space_group(experimental)
    values: dict[str, MetricValue] = {
        key: MetricValue(lattice_values[key], _format_number(lattice_values[key]))
        for key in _LATTICE_KEYS
    }
    values["rmsd"] = MetricValue(0.0, "0")
    values["max_displacement"] = MetricValue(0.0, "0")
    space_group_display = _format_space_group(space_group)
    values["space_group"] = MetricValue(space_group_display, space_group_display)
    values["space_group_match"] = MetricValue("reference", "Reference")
    return values


def dft_metric_values(
    experimental: Structure,
    dft: Structure,
    matching: AtomMatching,
) -> dict[str, MetricValue]:
    """Compute DFT-column metric values against the experimental structure.

    Each lattice value's display carries its percent change against experiment.
    """

    experimental_lattice = _structure_lattice_metrics(experimental)
    dft_lattice = _structure_lattice_metrics(dft)

    displacements = [pair.displacement for pair in matching.pairs]
    rmsd = float(np.sqrt(np.mean(np.square(displacements)))) if displacements else None
    max_displacement = float(np.max(displacements)) if displacements else None

    _exp_sg, dft_sg, sg_match = compare_space_group(experimental, dft)
    match_display = "N/A" if sg_match is None else ("Yes" if sg_match else "No")

    warning = None
    if matching.unmatched_experimental or matching.unmatched_dft:
        warning = "Partial atom matching"

    values: dict[str, MetricValue] = {
        key: MetricValue(
            dft_lattice[key],
            _format_with_change(dft_lattice[key], experimental_lattice[key]),
            warning=warning,
        )
        for key in _LATTICE_KEYS
    }
    values["rmsd"] = MetricValue(rmsd, _format_number(rmsd), warning=warning)
    values["max_displacement"] = MetricValue(
        max_displacement, _format_number(max_displacement), warning=warning
    )
    dft_space_group = _format_space_group(dft_sg)
    values["space_group"] = MetricValue(dft_space_group, dft_space_group)
    values["space_group_match"] = MetricValue(sg_match, match_display)
    return values
