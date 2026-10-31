"""The metrics of the structure comparison table and how their values are computed."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import AtomMatching
from mufinder.science.kpoint_grids import lattice_parameters
from mufinder.science.supercell_matching import host_cell_lattice
from mufinder.science.symmetry_analysis import (
    SpaceGroupResult,
    compare_space_group,
    detect_space_group,
)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """One row of the comparison table; the method behind it is in the user guide."""

    key: str
    label: str
    unit: str
    group: str
    tooltip: str


@dataclass(frozen=True, slots=True)
class MetricValue:
    """Rendered metric payload."""

    value: Any
    display: str
    warning: str | None = None


_METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        key="a",
        label="a",
        unit="Å",
        group="Lattice parameters",
        tooltip="Length of the first lattice vector, with the change against experiment for each DFT column.",
    ),
    MetricDefinition(
        key="b",
        label="b",
        unit="Å",
        group="Lattice parameters",
        tooltip="Length of the second lattice vector, with the change against experiment for each DFT column.",
    ),
    MetricDefinition(
        key="c",
        label="c",
        unit="Å",
        group="Lattice parameters",
        tooltip="Length of the third lattice vector, with the change against experiment for each DFT column.",
    ),
    MetricDefinition(
        key="alpha",
        label="α",
        unit="°",
        group="Lattice parameters",
        tooltip="Angle between the b and c lattice vectors, with the change against experiment for each DFT column.",
    ),
    MetricDefinition(
        key="beta",
        label="β",
        unit="°",
        group="Lattice parameters",
        tooltip="Angle between the a and c lattice vectors, with the change against experiment for each DFT column.",
    ),
    MetricDefinition(
        key="gamma",
        label="γ",
        unit="°",
        group="Lattice parameters",
        tooltip="Angle between the a and b lattice vectors, with the change against experiment for each DFT column.",
    ),
    MetricDefinition(
        key="cell_volume",
        label="V",
        unit="Å³",
        group="Lattice parameters",
        tooltip="Volume of the host cell, with the change against experiment for each DFT column.",
    ),
    MetricDefinition(
        key="rmsd",
        label="RMSD",
        unit="Å",
        group="Atomic displacements",
        tooltip="Root-mean-square displacement of the matched atoms from their experimental positions.",
    ),
    MetricDefinition(
        key="max_displacement",
        label="Max displacement",
        unit="Å",
        group="Atomic displacements",
        tooltip="Largest displacement of any matched atom from its experimental position.",
    ),
    MetricDefinition(
        key="space_group",
        label="Space group",
        unit="",
        group="Symmetry",
        tooltip="Space group that spglib finds for each structure.",
    ),
    MetricDefinition(
        key="space_group_match",
        label="Same space group",
        unit="",
        group="Symmetry",
        tooltip="Whether the DFT structure keeps the experimental space group.",
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


def format_with_change(value: float, reference: float | None, *, digits: int = 6) -> str:
    """``value`` with its signed percent change against ``reference``: ``4.7091 (+0.30 %)``.

    Without a usable ``reference`` it is the value alone. ``digits`` is the
    number of significant figures of the value.
    """

    change = None if reference is None else _safe_percent_change(reference, value)
    if change is None:
        return _format_number(value, digits=digits)
    # A change that rounds to zero reads "+0.00 %", never "-0.00 %".
    if abs(change) < 0.005:
        change = 0.0
    return f"{_format_number(value, digits=digits)} ({change:+.2f} %)"


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _format_space_group(group: SpaceGroupResult) -> str:
    """The spglib symbol with screw-axis subscripts and the number: ``P4₂/mnm (136)``."""

    if not group.available or group.number is None:
        return "N/A"
    symbol = re.sub(
        r"_(\d)", lambda match: match.group(1).translate(_SUBSCRIPT_DIGITS), group.symbol or ""
    )
    return f"{symbol} ({group.number})" if symbol else str(group.number)


def _lattice_metrics(lattice_vectors: Sequence[Sequence[float]] | np.ndarray) -> dict[str, float]:
    matrix = np.asarray(lattice_vectors, dtype=float)
    a, b, c, alpha, beta, gamma = lattice_parameters(matrix.tolist())
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

    lattice_values = _lattice_metrics(experimental.lattice_vectors)
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
    *,
    host: Structure | None = None,
    supercell_matrix: Sequence[Sequence[int]] | None = None,
) -> dict[str, MetricValue]:
    """Compute DFT-column metric values against the experimental structure.

    ``experimental`` and ``dft`` are the comparison pair, with the host tiled to
    the DFT cell when that is a supercell; the displacement and symmetry rows
    use them. The lattice rows are read per host cell: ``supercell_matrix`` is
    the detected ``M`` with ``L_DFT ≈ M L_host``, the DFT lattice is reduced to
    the host cell as ``M⁻¹ L_DFT`` (:func:`host_cell_lattice`) and compared with
    ``host``, the untiled experimental structure. Without them the lattice rows
    compare ``dft`` with ``experimental`` as given. Each lattice value's display
    carries its percent change against experiment.
    """

    reference = experimental if host is None else host
    experimental_lattice = _lattice_metrics(reference.lattice_vectors)
    dft_lattice_vectors: Sequence[Sequence[float]] | np.ndarray = dft.lattice_vectors
    if supercell_matrix is not None:
        dft_lattice_vectors = host_cell_lattice(dft_lattice_vectors, supercell_matrix)
    dft_lattice = _lattice_metrics(dft_lattice_vectors)

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
            format_with_change(dft_lattice[key], experimental_lattice[key]),
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
