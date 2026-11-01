"""Group the minima of a UEP batch search into distinct muon sites.

Random starts relax to exact stationary points of one smooth potential, so two
results are one site when they lie on the same symmetry orbit. Only converged
minima are grouped. Two minima are linked when one lies within a distance in
ångström of any member of the other's symmetry orbit and their classical energies
differ by less than a tolerance in eV, and the groups are the connected components
of those links (:func:`~mufinder.science.relaxation_analysis.cluster_connected_components`).

The classical energy is the one every search has, so a zero-point energy added
later changes neither the groups nor their order. Groups are numbered from the
lowest energy. No normalisation or batch statistic enters, so a group does not depend
on which other sites the batch found.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np

from mufinder.domain.text import plural
from mufinder.science.lattice import lattice_cart_to_frac, lattice_frac_to_cart, wrap_fractional
from mufinder.science.relaxation_analysis import (
    RelaxedMuonSite,
    cluster_connected_components,
    symmetry_distance_matrix,
)
from mufinder.science.site_generation import asymmetric_unit_position
from mufinder.science.spglib_compat import SYMMETRY_TOLERANCE_ANG, symmetry_detection_warning
from mufinder.science.uep import UepError, UepOptimizationResult

#: The tolerance a search is grouped at until the user changes it, well under the
#: 0.8 Å minimum muon-muon distance of the starts.
DEFAULT_GROUPING_TOLERANCE_ANG = 0.25
#: Minima further apart in classical energy than this are distinct, however close.
GROUPING_ENERGY_TOLERANCE_EV = 1e-3
#: The ``clustering_method`` of a :class:`UepBatchSummary`.
UEP_GROUPING_METHOD = "symmetry_distance"
#: The start of the log line :func:`uep_grouping_report` writes.
GROUPING_LOG_PREFIX = "Grouping: "

#: Log lines of an earlier search or grouping, replaced when a search is grouped again.
_REPLACED_LOG_PREFIXES = (GROUPING_LOG_PREFIX, "Relaxed sites available for clustering:")
#: The identifiers a minimum has no use for when it is treated as a relaxed site.
_NO_ID = UUID(int=0)
#: A position's sort key is rounded to this, so optimiser scatter does not decide the order.
_ORDER_DECIMALS = 6


@dataclass(frozen=True, slots=True)
class UepClusterSummary:
    """One group of minima: one distinct muon site.

    ``representative_index`` is the lowest-energy member. The representative
    positions are that member's image in the asymmetric unit
    (:func:`~mufinder.science.site_generation.asymmetric_unit_position`), and the energies
    are classical.
    """

    cluster_index: int
    size: int
    representative_index: int
    representative_position_ang: tuple[float, float, float]
    representative_fractional_position: tuple[float, float, float]
    minimum_energy_ev: float
    average_energy_ev: float
    standard_deviation_ev: float
    member_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class UepBatchSummary:
    """How a batch search was grouped and what that found, lowest energy first.

    ``failed_count`` and ``non_minimum_count`` are the optimisations left out for
    not converging or for ending on another kind of stationary point.
    ``symmetry_warning`` says why the host's symmetry was not used, if it was not.
    """

    threshold_ang: float
    clusters: tuple[UepClusterSummary, ...]
    clustering_method: str = UEP_GROUPING_METHOD
    energy_tolerance_ev: float = GROUPING_ENERGY_TOLERANCE_EV
    failed_count: int = 0
    non_minimum_count: int = 0
    symmetry_warning: str | None = None


@dataclass(frozen=True, slots=True)
class UepMinimaDistances:
    """The symmetry distances of a batch's minima, measured once for any number of groupings.

    ``result_indexes`` gives the result of each row of ``matrix_ang``.
    """

    result_indexes: tuple[int, ...]
    matrix_ang: tuple[tuple[float, ...], ...]
    symmetry_warning: str | None = field(default=None)


def is_converged_minimum(result: UepOptimizationResult) -> bool:
    """Return whether an optimisation converged on a true minimum with a finite energy."""

    return (
        bool(result.success)
        and result.classification.label == "minimum"
        and bool(np.isfinite(result.classical_energy_ev))
    )


def _ordered_minima(
    optimizations: Sequence[UepOptimizationResult], source_structure: Any
) -> tuple[tuple[int, ...], tuple[RelaxedMuonSite, ...]]:
    """The minima as relaxed sites in the source structure's cell, in an order of their own.

    Positions are taken from the Cartesian result, because a source job may have run
    on a supercell of the structure. The order depends only on the energies and
    positions, so the result does not depend on the order of the input.
    """

    lattice = source_structure.lattice_vectors
    rows = []
    for index, result in enumerate(optimizations):
        if not is_converged_minimum(result):
            continue
        fractional = wrap_fractional(lattice_cart_to_frac(result.optimized_position_ang, lattice))
        order = (float(result.classical_energy_ev), *np.round(fractional, _ORDER_DECIMALS).tolist())
        rows.append((order, index, fractional))
    rows.sort(key=lambda row: (row[0], row[1]))
    sites = tuple(
        RelaxedMuonSite(
            job_id=_NO_ID,
            job_label=f"{rank:06d}",
            charge_state="",
            structure_id=_NO_ID,
            site_id=None,
            folded_fractional_position=(fractional[0], fractional[1], fractional[2]),
            cartesian_position_ang=optimizations[index].optimized_position_ang,
            total_energy_ev=float(optimizations[index].classical_energy_ev),
        )
        for rank, (_order, index, fractional) in enumerate(rows)
    )
    return tuple(index for _order, index, _fractional in rows), sites


def measure_uep_minima(
    optimizations: Sequence[UepOptimizationResult],
    *,
    source_structure: Any,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
) -> UepMinimaDistances:
    """Return the symmetry distances between a batch's converged minima."""

    result_indexes, sites = _ordered_minima(optimizations, source_structure)
    return UepMinimaDistances(
        result_indexes=result_indexes,
        matrix_ang=symmetry_distance_matrix(sites, source_structure, symprec=symprec),
        symmetry_warning=symmetry_detection_warning(source_structure, symprec=symprec),
    )


def group_uep_minima(
    optimizations: Sequence[UepOptimizationResult],
    *,
    source_structure: Any,
    tolerance_ang: float = DEFAULT_GROUPING_TOLERANCE_ANG,
    energy_tolerance_ev: float = GROUPING_ENERGY_TOLERANCE_EV,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
    distances: UepMinimaDistances | None = None,
) -> UepBatchSummary:
    """Group a batch's converged minima into distinct sites.

    ``source_structure`` needs ``lattice_vectors``, ``species`` and
    ``fractional_positions``. ``distances`` reuses an earlier
    :func:`measure_uep_minima` of the same results, so regrouping at another tolerance
    measures nothing; the results' minima are measured here when they differ from it.
    """

    if tolerance_ang <= 0.0:
        raise UepError("The grouping tolerance must be positive.")
    if energy_tolerance_ev <= 0.0:
        raise UepError("The grouping energy tolerance must be positive.")
    result_indexes, sites = _ordered_minima(optimizations, source_structure)
    if distances is None or distances.result_indexes != result_indexes:
        distances = measure_uep_minima(
            optimizations, source_structure=source_structure, symprec=symprec
        )
    grouped = cluster_connected_components(
        sites,
        source_structure,
        tolerance_ang=tolerance_ang,
        symprec=symprec,
        energy_tolerance_ev=energy_tolerance_ev,
        distance_matrix_ang=distances.matrix_ang,
    )
    clusters = tuple(
        _cluster_summary(
            optimizations,
            source_structure,
            cluster_index=cluster.cluster_index,
            representative_index=result_indexes[cluster.representative_index],
            member_indexes=[result_indexes[member] for member in cluster.member_indexes],
            symprec=symprec,
        )
        for cluster in grouped.clusters
    )
    failed = sum(not result.success for result in optimizations)
    return UepBatchSummary(
        threshold_ang=float(tolerance_ang),
        clusters=clusters,
        energy_tolerance_ev=float(energy_tolerance_ev),
        failed_count=failed,
        non_minimum_count=len(optimizations) - failed - len(result_indexes),
        symmetry_warning=distances.symmetry_warning,
    )


def _cluster_summary(
    optimizations: Sequence[UepOptimizationResult],
    source_structure: Any,
    *,
    cluster_index: int,
    representative_index: int,
    member_indexes: Sequence[int],
    symprec: float,
) -> UepClusterSummary:
    lattice = source_structure.lattice_vectors
    fractional = asymmetric_unit_position(
        lattice_cart_to_frac(optimizations[representative_index].optimized_position_ang, lattice),
        lattice_vectors=lattice,
        host_species=source_structure.species,
        host_positions_frac=source_structure.fractional_positions,
        symprec=symprec,
    )
    cartesian = lattice_frac_to_cart(fractional, lattice)
    energies = [float(optimizations[member].classical_energy_ev) for member in member_indexes]
    return UepClusterSummary(
        cluster_index=int(cluster_index),
        size=len(member_indexes),
        representative_index=int(representative_index),
        representative_position_ang=(cartesian[0], cartesian[1], cartesian[2]),
        representative_fractional_position=(fractional[0], fractional[1], fractional[2]),
        minimum_energy_ev=min(energies),
        average_energy_ev=float(np.mean(energies)),
        standard_deviation_ev=float(np.std(energies)),
        member_indexes=tuple(sorted(int(member) for member in member_indexes)),
    )


def relaxed_site_number_map(
    optimizations: Sequence[UepOptimizationResult],
    *,
    included: Iterable[int] | None = None,
) -> dict[int, int]:
    """Return 1-based site numbers by classical energy, for the ``included`` result indexes."""

    indexes = range(len(optimizations)) if included is None else {int(index) for index in included}
    ordered = sorted(
        indexes,
        key=lambda index: (
            float(optimizations[index].classical_energy_ev)
            if np.isfinite(optimizations[index].classical_energy_ev)
            else float("inf"),
            int(index),
        ),
    )
    return {int(index): number for number, index in enumerate(ordered, start=1)}


def uep_grouping_report(summary: UepBatchSummary) -> str:
    """Return the search log's account of a grouping, including what it left out."""

    minima = sum(cluster.size for cluster in summary.clusters)
    text = (
        f"{GROUPING_LOG_PREFIX}{plural(minima, 'minimum', 'minima')} in "
        f"{plural(len(summary.clusters), 'site')}, within {summary.threshold_ang:g} Å and "
        f"{summary.energy_tolerance_ev * 1000.0:g} meV."
    )
    left_out = []
    if summary.failed_count:
        left_out.append(plural(summary.failed_count, "failed optimisation"))
    if summary.non_minimum_count:
        left_out.append(
            plural(
                summary.non_minimum_count,
                "result that is not a minimum",
                "results that are not minima",
            )
        )
    if left_out:
        text += f" Left out {' and '.join(left_out)}."
    if summary.symmetry_warning:
        text += f" {summary.symmetry_warning}"
    return text


def uep_search_log(existing: str, summary: UepBatchSummary) -> str:
    """Return a search log with its account of the grouping brought up to date."""

    lines = [
        line
        for line in existing.splitlines()
        if line.strip() and not line.startswith(_REPLACED_LOG_PREFIXES)
    ]
    return "\n".join([*lines, uep_grouping_report(summary)])


def _triple(values: object) -> tuple[float, float, float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _cluster_summary_from_dict(payload: object) -> UepClusterSummary | None:
    if not isinstance(payload, Mapping):
        return None
    position = _triple(payload.get("representative_position_ang"))
    fractional = _triple(payload.get("representative_fractional_position"))
    members = payload.get("member_indexes")
    if position is None or fractional is None or not isinstance(members, (list, tuple)):
        return None
    return UepClusterSummary(
        cluster_index=int(payload["cluster_index"]),
        size=int(payload["size"]),
        representative_index=int(payload["representative_index"]),
        representative_position_ang=position,
        representative_fractional_position=fractional,
        minimum_energy_ev=float(payload["minimum_energy_ev"]),
        average_energy_ev=float(payload["average_energy_ev"]),
        standard_deviation_ev=float(payload["standard_deviation_ev"]),
        member_indexes=tuple(int(member) for member in members),
    )


def uep_batch_summary_from_dict(payload: object) -> UepBatchSummary | None:
    """Read a saved grouping; ``None`` when it is malformed or used a retired method."""

    if not isinstance(payload, Mapping) or payload.get("clustering_method") != UEP_GROUPING_METHOD:
        return None
    raw_clusters = payload.get("clusters")
    if not isinstance(raw_clusters, (list, tuple)):
        return None
    try:
        clusters = tuple(_cluster_summary_from_dict(entry) for entry in raw_clusters)
        if any(cluster is None for cluster in clusters):
            return None
        warning = payload.get("symmetry_warning")
        return UepBatchSummary(
            threshold_ang=float(payload["threshold_ang"]),
            clusters=clusters,  # type: ignore[arg-type]
            energy_tolerance_ev=float(
                payload.get("energy_tolerance_ev", GROUPING_ENERGY_TOLERANCE_EV)
            ),
            failed_count=int(payload.get("failed_count", 0)),
            non_minimum_count=int(payload.get("non_minimum_count", 0)),
            symmetry_warning=None if warning is None else str(warning),
        )
    except (KeyError, TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_GROUPING_TOLERANCE_ANG",
    "GROUPING_ENERGY_TOLERANCE_EV",
    "GROUPING_LOG_PREFIX",
    "UEP_GROUPING_METHOD",
    "UepBatchSummary",
    "UepClusterSummary",
    "UepMinimaDistances",
    "group_uep_minima",
    "is_converged_minimum",
    "measure_uep_minima",
    "relaxed_site_number_map",
    "uep_batch_summary_from_dict",
    "uep_grouping_report",
    "uep_search_log",
]
