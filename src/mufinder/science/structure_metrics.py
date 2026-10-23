"""Configuration-driven metric definitions for structure comparison."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import AtomMatching
from mufinder.science.bonding_analysis import (
    compare_bond_angles,
    compare_coordination_numbers,
    compare_nearest_neighbour_bond_lengths,
    nearest_neighbour_distances,
)
from mufinder.science.kpoint_grids import lattice_parameters
from mufinder.science.symmetry_analysis import compare_space_group, compare_topology


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

_BONDING_METHOD = (
    "Bonding-derived metrics are evaluated on the normalized structures after atom matching. "
    "Neighbour relations use a radius-based cutoff of 1.3 times the sum of ASE covalent radii, "
    "applied with minimum-image periodic distances."
)

_SYMMETRY_METHOD = (
    "Symmetry and topology are evaluated on the normalized comparison structures. Space groups are "
    "identified with spglib, while topology is checked by comparing mapped bonding graphs built from "
    "the same radius-based neighbour criterion used for coordination metrics."
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
        unit="Ang",
        group="Lattice Parameters",
        meaning="First lattice-vector length.",
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
        unit="Ang",
        group="Lattice Parameters",
        meaning="Second lattice-vector length.",
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
        unit="Ang",
        group="Lattice Parameters",
        meaning="Third lattice-vector length.",
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
        label="alpha",
        unit="deg",
        group="Lattice Parameters",
        meaning="Angle between b and c vectors.",
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
        label="beta",
        unit="deg",
        group="Lattice Parameters",
        meaning="Angle between a and c vectors.",
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
        label="gamma",
        unit="deg",
        group="Lattice Parameters",
        meaning="Angle between a and b vectors.",
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
        label="Cell Volume",
        unit="Ang^3",
        group="Lattice Parameters",
        meaning="Unit-cell volume.",
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
        key="delta_a_pct",
        label="Change in a vs experiment",
        unit="%",
        group="Lattice Changes",
        meaning="Relative change in lattice a.",
        calculation="100*(a_dft-a_exp)/a_exp.",
        interpretation="Near-zero values indicate good agreement.",
        caveats="Requires consistent experimental normalization.",
        physics="Reports signed tensile or compressive relaxation relative to the experimental reference along the a direction.",
        method="Computed from the normalized experimental and DFT lattice lengths. A positive value means the relaxed DFT cell is longer than the experimental reference along a.",
        equation=r"\Delta a_{\%} = 100\,\frac{a_{\mathrm{DFT}} - a_{\mathrm{exp}}}{a_{\mathrm{exp}}}",
        symbols=(
            (r"a_{\mathrm{DFT}}", "DFT value of the a lattice parameter after normalization."),
            (r"a_{\mathrm{exp}}", "Experimental reference value of the a lattice parameter."),
            (r"\Delta a_{\%}", "Percent change in a relative to experiment."),
        ),
    ),
    MetricDefinition(
        key="delta_b_pct",
        label="Change in b vs experiment",
        unit="%",
        group="Lattice Changes",
        meaning="Relative change in lattice b.",
        calculation="100*(b_dft-b_exp)/b_exp.",
        interpretation="Near-zero values indicate good agreement.",
        caveats="Requires consistent experimental normalization.",
        physics="Reports signed tensile or compressive relaxation relative to the experimental reference along the b direction.",
        method="Computed from the normalized experimental and DFT lattice lengths. A positive value means the relaxed DFT cell is longer than the experimental reference along b.",
        equation=r"\Delta b_{\%} = 100\,\frac{b_{\mathrm{DFT}} - b_{\mathrm{exp}}}{b_{\mathrm{exp}}}",
        symbols=(
            (r"b_{\mathrm{DFT}}", "DFT value of the b lattice parameter after normalization."),
            (r"b_{\mathrm{exp}}", "Experimental reference value of the b lattice parameter."),
            (r"\Delta b_{\%}", "Percent change in b relative to experiment."),
        ),
    ),
    MetricDefinition(
        key="delta_c_pct",
        label="Change in c vs experiment",
        unit="%",
        group="Lattice Changes",
        meaning="Relative change in lattice c.",
        calculation="100*(c_dft-c_exp)/c_exp.",
        interpretation="Near-zero values indicate good agreement.",
        caveats="Requires consistent experimental normalization.",
        physics="Reports signed tensile or compressive relaxation relative to the experimental reference along the c direction.",
        method="Computed from the normalized experimental and DFT lattice lengths. A positive value means the relaxed DFT cell is longer than the experimental reference along c.",
        equation=r"\Delta c_{\%} = 100\,\frac{c_{\mathrm{DFT}} - c_{\mathrm{exp}}}{c_{\mathrm{exp}}}",
        symbols=(
            (r"c_{\mathrm{DFT}}", "DFT value of the c lattice parameter after normalization."),
            (r"c_{\mathrm{exp}}", "Experimental reference value of the c lattice parameter."),
            (r"\Delta c_{\%}", "Percent change in c relative to experiment."),
        ),
    ),
    MetricDefinition(
        key="delta_volume_pct",
        label="Change in volume vs experiment",
        unit="%",
        group="Lattice Changes",
        meaning="Relative change in cell volume.",
        calculation="100*(V_dft-V_exp)/V_exp.",
        interpretation="Near-zero values indicate good agreement.",
        caveats="Requires consistent experimental normalization.",
        physics="Summarizes net volumetric expansion or compression of the relaxed DFT cell relative to the experimental reference.",
        method="Computed from the normalized experimental and DFT cell volumes. The sign distinguishes volumetric expansion from contraction.",
        equation=r"\Delta V_{\%} = 100\,\frac{V_{\mathrm{DFT}} - V_{\mathrm{exp}}}{V_{\mathrm{exp}}}",
        symbols=(
            (r"V_{\mathrm{DFT}}", "DFT cell volume after normalization."),
            (r"V_{\mathrm{exp}}", "Experimental reference cell volume."),
            (r"\Delta V_{\%}", "Percent change in cell volume relative to experiment."),
        ),
    ),
    MetricDefinition(
        key="rmsd",
        label="RMSD (matched atoms)",
        unit="Ang",
        group="Atomic Displacements",
        meaning="Root-mean-square displacement over matched atoms.",
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
        unit="Ang",
        group="Atomic Displacements",
        meaning="Largest displacement among matched atoms.",
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
        key="mean_displacement",
        label="Mean displacement",
        unit="Ang",
        group="Atomic Displacements",
        meaning="Average displacement among matched atoms.",
        calculation="mean(d_i).",
        interpretation="Lower values indicate better atomic retention.",
        caveats="Depends on atom matching quality.",
        physics="Measures the average magnitude of the matched-atom relaxations and is less sensitive to a single outlier than the maximum displacement.",
        method=_MATCHING_METHOD,
        equation=r"\bar{d} = \frac{1}{N_{\mathrm{match}}}\sum_{i=1}^{N_{\mathrm{match}}} d_i",
        symbols=(
            (r"d_i", "Minimum-image Cartesian displacement of matched atom pair i."),
            (r"\bar{d}", "Mean matched-atom displacement."),
            (r"N_{\mathrm{match}}", "Number of matched atom pairs."),
        ),
    ),
    MetricDefinition(
        key="mean_bond_length_diff",
        label="Mean nearest-neighbour bond-length difference",
        unit="Ang",
        group="Bonding",
        meaning="Average change in nearest-neighbour distances.",
        calculation="Mean absolute difference of nearest-neighbour distances.",
        interpretation="Lower is better.",
        caveats="Sensitive to neighbour-definition thresholds.",
        physics="Tracks how much the shortest local bond length around each matched site changes between experiment and the relaxed DFT structure.",
        method=(
            _BONDING_METHOD
            + " The nearest-neighbour length for one atom is the smallest non-zero periodic distance to any other atom in the same structure."
        ),
        equation=(
            r"r_i^{\mathrm{NN}} = \min_{j \ne i} r_{ij}"
            "\n"
            r"\Delta r_{\mathrm{mean}} = \frac{1}{N_{\mathrm{match}}}\sum_{i=1}^{N_{\mathrm{match}}} "
            r"\left| r_{m(i)}^{\mathrm{NN,DFT}} - r_i^{\mathrm{NN,exp}} \right|"
        ),
        symbols=(
            (r"r_{ij}", "Periodic minimum-image distance between atoms i and j."),
            (r"r_i^{\mathrm{NN}}", "Nearest-neighbour distance associated with atom i."),
            (r"m(i)", "Matched DFT atom for experimental atom i."),
            (
                r"\Delta r_{\mathrm{mean}}",
                "Mean absolute nearest-neighbour bond-length change over matched sites.",
            ),
        ),
    ),
    MetricDefinition(
        key="max_bond_length_diff",
        label="Max nearest-neighbour bond-length difference",
        unit="Ang",
        group="Bonding",
        meaning="Largest change in nearest-neighbour distances.",
        calculation="Max absolute difference of nearest-neighbour distances.",
        interpretation="Highlights local bonding outliers.",
        caveats="Sensitive to neighbour-definition thresholds.",
        physics="Highlights the strongest local change in shortest-bond scale among the matched sites.",
        method=(
            _BONDING_METHOD
            + " The compared quantity is still the nearest-neighbour distance for each matched site, but the metric reports the largest absolute change rather than the average."
        ),
        equation=(
            r"\Delta r_{\max} = \max_{1 \le i \le N_{\mathrm{match}}} "
            r"\left| r_{m(i)}^{\mathrm{NN,DFT}} - r_i^{\mathrm{NN,exp}} \right|"
        ),
        symbols=(
            (r"r_i^{\mathrm{NN}}", "Nearest-neighbour distance associated with atom i."),
            (r"m(i)", "Matched DFT atom for experimental atom i."),
            (
                r"\Delta r_{\max}",
                "Largest absolute nearest-neighbour bond-length change over matched sites.",
            ),
        ),
    ),
    MetricDefinition(
        key="mean_bond_angle_diff",
        label="Mean bond-angle difference",
        unit="deg",
        group="Bonding",
        meaning="Average change in local bond-angle environments.",
        calculation="Mean absolute difference of local mean bond angles.",
        interpretation="Lower is better.",
        caveats="Angles are unstable with very low coordination.",
        physics="Measures how much the mean angular opening of each local coordination environment changes during relaxation.",
        method=(
            _BONDING_METHOD
            + " For one site, all neighbour-neighbour angles formed by periodic bond vectors are averaged to give a local mean angle. Sites with fewer than two neighbours do not contribute."
        ),
        equation=(
            r"\bar{\theta}_i = \frac{1}{M_i}\sum_{(j,k) \in \mathcal{P}_i} "
            r"\cos^{-1}\left(\frac{\mathbf{r}_{ij}\cdot\mathbf{r}_{ik}}{\lVert \mathbf{r}_{ij} \rVert_2 \lVert \mathbf{r}_{ik} \rVert_2}\right)"
            "\n"
            r"\Delta \theta_{\mathrm{mean}} = \frac{1}{N_{\theta}}\sum_i \left| \bar{\theta}_{m(i)}^{\mathrm{DFT}} - \bar{\theta}_i^{\mathrm{exp}} \right|"
        ),
        symbols=(
            (r"\mathbf{r}_{ij}", "Periodic bond vector from site i to neighbour j."),
            (r"\mathcal{P}_i", "Set of unique neighbour pairs around site i."),
            (r"M_i", "Number of neighbour-pair angles included around site i."),
            (r"\bar{\theta}_i", "Mean local bond angle around site i."),
            (
                r"N_{\theta}",
                "Number of matched sites that contribute a valid local mean angle in both structures.",
            ),
        ),
    ),
    MetricDefinition(
        key="coordination_changes",
        label="Coordination-number changes",
        unit="count",
        group="Bonding",
        meaning="Number of matched atoms whose coordination changed.",
        calculation="Count of matched atoms with unequal neighbour counts.",
        interpretation="0 indicates full coordination preservation.",
        caveats="Depends on neighbour cutoff policy.",
        physics="Counts local coordination-shell changes, which are often the clearest signal of bond breaking, bond formation, or major local reconstruction.",
        method=_BONDING_METHOD,
        equation=(
            r"\mathrm{CN}_i = \sum_{j \ne i} \mathbf{1}\left[r_{ij} \le 1.3\,(R_i + R_j)\right]"
            "\n"
            r"N_{\Delta \mathrm{CN}} = \sum_{i=1}^{N_{\mathrm{match}}} \mathbf{1}\left["
            r"\mathrm{CN}_i^{\mathrm{exp}} \ne \mathrm{CN}_{m(i)}^{\mathrm{DFT}}\right]"
        ),
        symbols=(
            (
                r"\mathrm{CN}_i",
                "Coordination number of site i under the radius-based neighbour rule.",
            ),
            (r"R_i", "ASE covalent radius for the species on site i."),
            (
                r"\mathbf{1}[\cdot]",
                "Indicator function equal to 1 when the condition is true and 0 otherwise.",
            ),
            (
                r"N_{\Delta \mathrm{CN}}",
                "Number of matched sites whose coordination number changes.",
            ),
        ),
    ),
    MetricDefinition(
        key="space_group",
        label="Space group comparison",
        unit="",
        group="Symmetry",
        meaning="Space-group match between experimental and DFT structures.",
        calculation="spglib identification and number comparison.",
        interpretation="Matching groups suggest symmetry retention.",
        caveats="Requires spglib and tolerance tuning.",
        physics="Captures whether the relaxed periodic crystal retains the same crystallographic symmetry class as the experimental reference.",
        method=_SYMMETRY_METHOD
        + " The current implementation compares international space-group numbers with spglib using symprec = 1e-2.",
        equation=r"\mathrm{match}_{\mathrm{SG}} = \mathbf{1}\left[n_{\mathrm{exp}} = n_{\mathrm{DFT}}\right]",
        symbols=(
            (r"n_{\mathrm{exp}}", "Experimental space-group number reported by spglib."),
            (r"n_{\mathrm{DFT}}", "DFT space-group number reported by spglib."),
            (r"\mathrm{match}_{\mathrm{SG}}", "Boolean indicator of space-group-number equality."),
        ),
    ),
    MetricDefinition(
        key="topology_preserved",
        label="Topology/connectivity preserved",
        unit="",
        group="Symmetry",
        meaning="Whether bonding connectivity is preserved.",
        calculation="Compare matched-atom connectivity graphs.",
        interpretation="True indicates preserved bonding topology.",
        caveats="Depends on neighbour-definition thresholds.",
        physics="Tests whether the bond network implied by the chosen neighbour rule survives relaxation after mapped atom identities are taken into account.",
        method=_SYMMETRY_METHOD,
        equation=(
            r"E = \{(i,j): r_{ij} \le 1.3\,(R_i + R_j)\}"
            "\n"
            r"\mathrm{preserved}_{\mathrm{topo}} = \mathbf{1}\left[E_{\mathrm{exp}}^{\mathrm{mapped}} = E_{\mathrm{DFT}}^{\mathrm{matched}}\right]"
        ),
        symbols=(
            (r"E", "Undirected bond-edge set implied by the radius-based neighbour criterion."),
            (
                r"E_{\mathrm{exp}}^{\mathrm{mapped}}",
                "Experimental bond-edge set after mapping matched experimental indices into DFT index space.",
            ),
            (
                r"E_{\mathrm{DFT}}^{\mathrm{matched}}",
                "DFT bond-edge set restricted to matched atoms.",
            ),
            (
                r"\mathrm{preserved}_{\mathrm{topo}}",
                "Boolean indicator of exact connectivity preservation under the matched mapping.",
            ),
        ),
    ),
    MetricDefinition(
        key="similarity_score",
        label="Similarity score",
        unit="0-1",
        group="Overall",
        meaning="Composite structural similarity index.",
        calculation="Weighted geometric mean of normalized lattice, displacement, and bonding terms.",
        interpretation="1 is best; values closer to 0 indicate larger differences.",
        caveats="Heuristic metric; compare alongside raw metrics.",
        physics="Provides a single heuristic summary that balances lattice agreement, matched-atom relaxation amplitude, and coordination preservation, with a small symmetry bonus.",
        method=(
            "The score uses the mean absolute percent changes in a, b, and c to form a lattice term, the RMSD normalized by the mean experimental nearest-neighbour distance to form a displacement term, and the preserved coordination fraction to form a bonding term. "
            "Topology and bond-angle metrics remain diagnostic outputs but do not currently enter the score directly."
        ),
        equation=(
            r"\Delta_{\mathrm{lat}} = \frac{1}{3}\sum_{x \in \{a,b,c\}} \left|\Delta x_{\%}\right|, "
            r"\quad S_{\mathrm{lat}} = \max\left(0, 1 - \frac{\Delta_{\mathrm{lat}}}{100}\right)"
            "\n"
            r"\sigma = \left\langle r_i^{\mathrm{NN,exp}} \right\rangle, "
            r"\quad S_{\mathrm{disp}} = \exp\left(-\frac{\mathrm{RMSD}}{\max(\sigma, 10^{-8})}\right)"
            "\n"
            r"S_{\mathrm{bond}} = f_{\mathrm{coord}}"
            "\n"
            r"S = \min\left(1, \max\left(0, S_{\mathrm{lat}}^{0.35} S_{\mathrm{disp}}^{0.35} S_{\mathrm{bond}}^{0.30} + 0.1\,\mathbf{1}[n_{\mathrm{exp}} = n_{\mathrm{DFT}}]\right)\right)"
        ),
        symbols=(
            (
                r"\Delta x_{\%}",
                "Percent change in lattice parameter x relative to experiment, for x in {a, b, c}.",
            ),
            (r"\Delta_{\mathrm{lat}}", "Mean absolute lattice percent change across a, b, and c."),
            (r"S_{\mathrm{lat}}", "Normalized lattice-similarity factor."),
            (
                r"\sigma",
                "Mean experimental nearest-neighbour distance, used as a characteristic length scale.",
            ),
            (r"S_{\mathrm{disp}}", "Displacement-similarity factor derived from RMSD."),
            (
                r"f_{\mathrm{coord}}",
                "Fraction of matched sites whose coordination numbers are preserved.",
            ),
            (
                r"S_{\mathrm{bond}}",
                "Bonding-similarity factor, equal to the preserved coordination fraction.",
            ),
            (r"S", "Final similarity score clipped to the interval [0, 1]."),
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


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4g}%"


def _safe_percent_change(reference: float, value: float) -> float | None:
    if abs(reference) <= 1e-12:
        return None
    return 100.0 * (value - reference) / reference


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


def experimental_metric_values(experimental: Structure) -> dict[str, MetricValue]:
    """Compute metric values for the experimental reference column."""

    lattice_values = _structure_lattice_metrics(experimental)
    values: dict[str, MetricValue] = {
        "a": MetricValue(lattice_values["a"], _format_number(lattice_values["a"])),
        "b": MetricValue(lattice_values["b"], _format_number(lattice_values["b"])),
        "c": MetricValue(lattice_values["c"], _format_number(lattice_values["c"])),
        "alpha": MetricValue(lattice_values["alpha"], _format_number(lattice_values["alpha"])),
        "beta": MetricValue(lattice_values["beta"], _format_number(lattice_values["beta"])),
        "gamma": MetricValue(lattice_values["gamma"], _format_number(lattice_values["gamma"])),
        "cell_volume": MetricValue(
            lattice_values["cell_volume"], _format_number(lattice_values["cell_volume"])
        ),
        "delta_a_pct": MetricValue(0.0, "0%"),
        "delta_b_pct": MetricValue(0.0, "0%"),
        "delta_c_pct": MetricValue(0.0, "0%"),
        "delta_volume_pct": MetricValue(0.0, "0%"),
        "rmsd": MetricValue(0.0, "0"),
        "max_displacement": MetricValue(0.0, "0"),
        "mean_displacement": MetricValue(0.0, "0"),
        "mean_bond_length_diff": MetricValue(0.0, "0"),
        "max_bond_length_diff": MetricValue(0.0, "0"),
        "mean_bond_angle_diff": MetricValue(0.0, "0"),
        "coordination_changes": MetricValue(0, "0"),
        "space_group": MetricValue("reference", "Reference"),
        "topology_preserved": MetricValue(True, "Reference"),
        "similarity_score": MetricValue(1.0, "1"),
    }
    return values


def dft_metric_values(
    experimental: Structure,
    dft: Structure,
    matching: AtomMatching,
) -> dict[str, MetricValue]:
    """Compute DFT-column metric values against the experimental structure."""

    experimental_lattice = _structure_lattice_metrics(experimental)
    dft_lattice = _structure_lattice_metrics(dft)

    displacements = [pair.displacement for pair in matching.pairs]
    rmsd = float(np.sqrt(np.mean(np.square(displacements)))) if displacements else None
    max_displacement = float(np.max(displacements)) if displacements else None
    mean_displacement = float(np.mean(displacements)) if displacements else None

    bond_lengths = compare_nearest_neighbour_bond_lengths(experimental, dft, matching)
    bond_angles = compare_bond_angles(experimental, dft, matching)
    coordination = compare_coordination_numbers(experimental, dft, matching)
    exp_sg, dft_sg, sg_match = compare_space_group(experimental, dft)
    topology = compare_topology(experimental, dft, matching)

    reference_nn = [
        value for value in nearest_neighbour_distances(experimental) if value is not None
    ]
    sigma = float(np.mean(reference_nn)) if reference_nn else 1.0

    lattice_delta = np.mean(
        [
            abs(_safe_percent_change(experimental_lattice[name], dft_lattice[name]) or 0.0)
            for name in ("a", "b", "c")
        ]
    )
    lattice_similarity = max(0.0, 1.0 - (float(lattice_delta) / 100.0))
    displacement_similarity = exp(-(rmsd or 0.0) / max(sigma, 1e-8)) if rmsd is not None else 0.0
    bonding_similarity = (
        coordination.preserved_fraction if coordination.preserved_fraction is not None else 0.0
    )
    symmetry_bonus = 0.1 if sg_match else 0.0

    weighted_product = (
        max(lattice_similarity, 1e-8) ** 0.35
        * max(displacement_similarity, 1e-8) ** 0.35
        * max(bonding_similarity, 1e-8) ** 0.3
    )
    similarity_score = min(1.0, max(0.0, weighted_product + symmetry_bonus))

    sg_display = "N/A"
    if exp_sg.available and dft_sg.available:
        exp_token = exp_sg.symbol or "?"
        dft_token = dft_sg.symbol or "?"
        if sg_match is True:
            sg_display = f"Match ({exp_token})"
        elif sg_match is False:
            sg_display = f"{exp_token} vs {dft_token}"

    topology_display = "N/A"
    if topology.preserved is True:
        topology_display = "Yes"
    elif topology.preserved is False:
        topology_display = "No"

    warning = None
    if matching.unmatched_experimental or matching.unmatched_dft:
        warning = "Partial atom matching"

    values: dict[str, MetricValue] = {
        "a": MetricValue(dft_lattice["a"], _format_number(dft_lattice["a"]), warning=warning),
        "b": MetricValue(dft_lattice["b"], _format_number(dft_lattice["b"]), warning=warning),
        "c": MetricValue(dft_lattice["c"], _format_number(dft_lattice["c"]), warning=warning),
        "alpha": MetricValue(
            dft_lattice["alpha"], _format_number(dft_lattice["alpha"]), warning=warning
        ),
        "beta": MetricValue(
            dft_lattice["beta"], _format_number(dft_lattice["beta"]), warning=warning
        ),
        "gamma": MetricValue(
            dft_lattice["gamma"], _format_number(dft_lattice["gamma"]), warning=warning
        ),
        "cell_volume": MetricValue(
            dft_lattice["cell_volume"],
            _format_number(dft_lattice["cell_volume"]),
            warning=warning,
        ),
        "delta_a_pct": MetricValue(
            _safe_percent_change(experimental_lattice["a"], dft_lattice["a"]),
            _format_percent(_safe_percent_change(experimental_lattice["a"], dft_lattice["a"])),
            warning=warning,
        ),
        "delta_b_pct": MetricValue(
            _safe_percent_change(experimental_lattice["b"], dft_lattice["b"]),
            _format_percent(_safe_percent_change(experimental_lattice["b"], dft_lattice["b"])),
            warning=warning,
        ),
        "delta_c_pct": MetricValue(
            _safe_percent_change(experimental_lattice["c"], dft_lattice["c"]),
            _format_percent(_safe_percent_change(experimental_lattice["c"], dft_lattice["c"])),
            warning=warning,
        ),
        "delta_volume_pct": MetricValue(
            _safe_percent_change(experimental_lattice["cell_volume"], dft_lattice["cell_volume"]),
            _format_percent(
                _safe_percent_change(
                    experimental_lattice["cell_volume"], dft_lattice["cell_volume"]
                )
            ),
            warning=warning,
        ),
        "rmsd": MetricValue(rmsd, _format_number(rmsd), warning=warning),
        "max_displacement": MetricValue(
            max_displacement, _format_number(max_displacement), warning=warning
        ),
        "mean_displacement": MetricValue(
            mean_displacement, _format_number(mean_displacement), warning=warning
        ),
        "mean_bond_length_diff": MetricValue(
            bond_lengths.mean_difference,
            _format_number(bond_lengths.mean_difference),
            warning=warning,
        ),
        "max_bond_length_diff": MetricValue(
            bond_lengths.max_difference,
            _format_number(bond_lengths.max_difference),
            warning=warning,
        ),
        "mean_bond_angle_diff": MetricValue(
            bond_angles.mean_difference,
            _format_number(bond_angles.mean_difference),
            warning=warning,
        ),
        "coordination_changes": MetricValue(
            coordination.changed_sites,
            str(coordination.changed_sites),
            warning=warning,
        ),
        "space_group": MetricValue(sg_display, sg_display, warning=None),
        "topology_preserved": MetricValue(topology.preserved, topology_display, warning=None),
        "similarity_score": MetricValue(
            similarity_score, _format_number(similarity_score, digits=4), warning=warning
        ),
    }

    return values
