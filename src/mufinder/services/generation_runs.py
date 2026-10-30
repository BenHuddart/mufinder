"""Plain-language descriptions of muon generation runs, and the runs' groups of starts.

A :class:`~mufinder.domain.models.GenerationRun` records one ``Generate Sites``
request. This module turns it into the one-line summary the Generation page
heads the run's starts with and the Journal prints, and into a sentence for a
methods section (``Copy description``). It also groups the generated starts on
the page by run. Starts made before runs were recorded have no run; they are
grouped as "Earlier starts" with a summary built from the inputs each start
kept in its ``properties``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from mufinder.domain.models import GenerationRun, MuonSite
from mufinder.domain.text import natural_sort_key, plural, space_group_text
from mufinder.persistence import project_codec
from mufinder.services.project_document import ProjectDocument

#: Name of each generation method as the summary line starts it.
_METHOD_TITLES = {
    "random": "Random",
    "manual": "Custom coordinates",
    "uep_clusters": "UEP clusters",
}

#: Site ``generation_method`` values that a run records as ``uep_clusters``.
_UEP_SITE_METHODS = frozenset({"uep_cluster", "uep_cluster_member", "uep_batch_cluster"})

#: Heading word of the group of starts made before generation runs were recorded.
EARLIER_STARTS_TITLE = "Earlier starts"

#: UEP clustering method names, as the methods sentence writes them.
_UEP_CLUSTERING_NAMES = {"hier": "hierarchical", "kmeans": "k-means"}


def _distance(value: object) -> str:
    """Write a distance to at most two decimals, keeping one (``1.0``, ``1.79``)."""

    text = f"{float(value):.2f}".rstrip("0")  # type: ignore[arg-type]
    return f"{text}0" if text.endswith(".") else text


def _number(value: object) -> str:
    return f"{float(value):g}"  # type: ignore[arg-type]


def _start_count(run: GenerationRun, count: int | None) -> int:
    return len(run.site_ids) if count is None else count


def _symmetry_summary(run: GenerationRun) -> str | None:
    """Return the symmetry part of the summary line, or ``None`` when it has none."""

    if run.method not in {"random", "near_element"} or run.symmetry_tolerance_ang is None:
        return None
    if not run.uses_symmetry:
        return "symmetry not used"
    if not run.space_group_symbol:
        return "no symmetry detected (P1)"
    operations = run.symmetry_operation_count or 1
    return f"{space_group_text(run.space_group_symbol)} ({operations} ops)"


def _uep_source_text(run: GenerationRun) -> str | None:
    jobs = run.parameters.get("source_jobs")
    labels = [
        str(job.get("label", "")).strip()
        for job in (jobs if isinstance(jobs, list) else [])
        if isinstance(job, Mapping) and str(job.get("label", "")).strip()
    ]
    if not labels:
        return None
    if len(labels) == 1:
        return f"'{labels[0]}'"
    return plural(len(labels), "UEP job")


def run_summary_line(run: GenerationRun, *, count: int | None = None) -> str:
    """Return the run's parameters in one line.

    For example ``"Random, 6 starts, ≥ 1.79 Å apart, ≥ 1.0 Å from atoms,
    P4₂/mnm (16 ops)"``. ``count`` overrides the number of starts (the
    run's own by default). The seed is not part of the line.
    """

    parameters = run.parameters
    starts = plural(_start_count(run, count), "start")
    if run.method == "random":
        parts = [_METHOD_TITLES["random"], starts]
        if parameters.get("min_muon_muon") is not None:
            parts.append(f"≥ {_distance(parameters['min_muon_muon'])} Å apart")
        if parameters.get("min_muon_atom") is not None:
            parts.append(f"≥ {_distance(parameters['min_muon_atom'])} Å from atoms")
    elif run.method == "near_element":
        element = str(parameters.get("element", "")).strip() or "element"
        parts = [f"Attached to {element}", starts]
        if parameters.get("muons_per_atom") is not None:
            parts.append(f"{int(parameters['muons_per_atom'])} per atom")
        if parameters.get("bond_distance") is not None:
            parts.append(f"{_distance(parameters['bond_distance'])} Å from {element}")
        if parameters.get("min_muon_atom") is not None:
            parts.append(f"≥ {_distance(parameters['min_muon_atom'])} Å from atoms")
    elif run.method == "manual":
        parts = [_METHOD_TITLES["manual"], starts]
        coord_system = str(parameters.get("coord_system", "")).strip()
        if coord_system:
            parts.append(coord_system)
    elif run.method == "uep_clusters":
        parts = [_METHOD_TITLES["uep_clusters"], starts]
        source = _uep_source_text(run)
        if source:
            parts.append(f"from {source}")
    else:
        parts = [run.method or "Unknown method", starts]
    symmetry = _symmetry_summary(run)
    if symmetry:
        parts.append(symmetry)
    return ", ".join(parts)


def run_journal_line(run: GenerationRun) -> str:
    """Return the summary line with the seed, as the Journal prints it."""

    line = run_summary_line(run)
    if run.seed is not None:
        line += f", seed {run.seed}"
    return f"{line}."


def _symmetry_sentence(run: GenerationRun, *, subject: str) -> str:
    if run.symmetry_tolerance_ang is None:
        return ""
    tolerance = _number(run.symmetry_tolerance_ang)
    if not run.uses_symmetry:
        return " Symmetry-equivalent positions were not excluded."
    if not run.space_group_symbol:
        return (
            f" No host symmetry was detected by spglib at a tolerance of {tolerance} Å, "
            "so the host was treated as P1."
        )
    operations = run.symmetry_operation_count or 1
    return (
        f" {subject} the {plural(operations, 'symmetry operation')} of the host cell "
        f"(space group {space_group_text(run.space_group_symbol)}), detected by spglib at a "
        f"tolerance of {tolerance} Å."
    )


def run_methods_sentence(
    run: GenerationRun, *, host_label: str | None = None, count: int | None = None
) -> str:
    """Return a methods-section description of the run, built from its record.

    ``host_label`` names the host structure; ``count`` overrides the number of
    starts, as in :func:`run_summary_line`.
    """

    parameters = run.parameters
    n = _start_count(run, count)
    host = f" in {host_label}" if host_label else ""
    if run.method == "random":
        symmetric = run.uses_symmetry and bool(run.space_group_symbol)
        images = " and their symmetry images" if symmetric else ""
        text = f"MuFinder generated {plural(n, 'initial muon position')} at random{host}"
        if parameters.get("min_muon_muon") is not None:
            text += (
                f", each at least {_distance(parameters['min_muon_muon'])} Å from the other "
                f"starts{images}"
            )
        if parameters.get("min_muon_atom") is not None:
            text += f" and at least {_distance(parameters['min_muon_atom'])} Å from every host atom"
        text += "."
        estimated = parameters.get("estimated_min_muon_muon")
        if parameters.get("selection_mode") == "count" and estimated is not None:
            text += (
                f" The separation was estimated as {_distance(estimated)} Å from the requested "
                f"{plural(int(parameters.get('count', n)), 'start')}"
            )
            if run.retries:
                text += (
                    f" and reduced to {_distance(parameters['min_muon_muon'])} Å after "
                    f"{plural(run.retries, 'retry', 'retries')} so that they fitted"
                )
            text += "."
        if run.attempts_used is not None:
            text += (
                f" {plural(run.attempts_used, 'trial position was', 'trial positions were')} drawn"
            )
            if parameters.get("max_attempts") is not None:
                text += f" (limit {int(parameters['max_attempts'])} per try)"
            text += "."
        text += _symmetry_sentence(run, subject="Symmetry images were taken from")
        if run.seed is not None:
            text += f" The random seed was {run.seed}."
    elif run.method == "near_element":
        element = str(parameters.get("element", "")).strip() or "the chosen element"
        text = f"MuFinder placed {plural(n, 'initial muon position')}{host}"
        if parameters.get("bond_distance") is not None:
            text += f", {_distance(parameters['bond_distance'])} Å from {element}"
        if parameters.get("muons_per_atom") is not None:
            text += (
                f": {plural(int(parameters['muons_per_atom']), 'start')} on one {element} atom "
                "of each set of symmetry-equivalent atoms"
            )
        text += ", along the directions with the most clearance from the host"
        if parameters.get("neighbour_cutoff_ang") is not None:
            text += f" (neighbours within {_distance(parameters['neighbour_cutoff_ang'])} Å)"
        if parameters.get("min_muon_atom") is not None:
            text += (
                f", each at least {_distance(parameters['min_muon_atom'])} Å from every host atom"
            )
        text += "."
        text += _symmetry_sentence(run, subject="Equivalent atoms were taken from")
    elif run.method == "manual":
        coordinates = parameters.get("coordinates")
        coord_system = str(parameters.get("coord_system", "")).strip() or "fractional"
        if isinstance(coordinates, list) and len(coordinates) == 3:
            values = ", ".join(_number(value) for value in coordinates)
            unit = " Å" if coord_system == "cartesian" else ""
            text = (
                f"The initial muon position{host} was entered by hand at {coord_system} "
                f"coordinates ({values}){unit}."
            )
        else:
            text = (
                f"{plural(n, 'initial muon position was', 'initial muon positions were')} "
                f"entered by hand{host} as {coord_system} coordinates."
            )
    elif run.method == "uep_clusters":
        source = _uep_source_text(run)
        text = (
            f"MuFinder took {plural(n, 'initial muon position')}{host} from the clustered "
            "minima of the unperturbed electrostatic potential (UEP)"
        )
        if source:
            text += f" of {source}"
        grouping = parameters.get("grouping")
        method = grouping.get("clustering_method") if isinstance(grouping, Mapping) else None
        if method:
            name = _UEP_CLUSTERING_NAMES.get(str(method), str(method))
            text += f", grouped by {name} clustering"
            if str(method) == "hier" and grouping.get("clustering_hier_t") is not None:
                text += f" (t = {_number(grouping['clustering_hier_t'])})"
            elif str(method) == "kmeans" and grouping.get("clustering_kmeans_k") is not None:
                text += f" (k = {int(grouping['clustering_kmeans_k'])})"
        text += "."
    else:
        text = f"MuFinder generated {plural(n, 'initial muon position')}{host}."
    if run.distinct_site_count is not None and run.distinct_site_count != n:
        text += f" {run.distinct_site_count} of them are symmetry-distinct."
    return text


def generation_run_from_provenance(parameters: Mapping[str, Any]) -> GenerationRun | None:
    """Return the run a ``generate_candidates`` record keeps, or ``None`` when it keeps none."""

    payload = parameters.get("generation_run")
    if not isinstance(payload, dict):
        return None
    try:
        return project_codec.generation_run_from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class GenerationRunGroup:
    """The generated starts of one run, as the Generation page lists them.

    ``run_id`` is ``None`` for a group of earlier starts, which had no run.
    ``title`` is the group's header line and ``description`` its methods
    sentence; ``site_ids`` are the starts in label order.
    """

    key: str
    run_id: UUID | None
    title: str
    description: str
    site_ids: tuple[UUID, ...]


def _run_method(site_method: str | None) -> str:
    method = str(site_method or "")
    if method in _UEP_SITE_METHODS:
        return "uep_clusters"
    return method


def _earlier_inputs(method: str, properties: Mapping[str, Any]) -> dict[str, Any]:
    """Return the run inputs an earlier start kept in its ``properties``."""

    keys: dict[str, str]
    if method == "random":
        keys = {
            "selection_mode": "selection_mode",
            "min_muon_muon": "min_muon_muon",
            "min_muon_atom": "min_muon_atom",
            "count": "target_count",
            "estimated_min_muon_muon": "estimated_min_muon_muon",
        }
    elif method == "near_element":
        keys = {
            "element": "target_element",
            "muons_per_atom": "muons_per_atom",
            "bond_distance": "bond_distance",
            "min_muon_atom": "min_muon_atom",
        }
    elif method == "manual":
        keys = {"coord_system": "coordinate_system"}
    else:
        keys = {}
    inputs = {
        name: properties[key] for name, key in keys.items() if properties.get(key) is not None
    }
    if method == "uep_clusters" and properties.get("source_job_label"):
        inputs["source_jobs"] = [
            {
                "id": str(properties.get("source_job_id", "") or ""),
                "label": str(properties["source_job_label"]),
            }
        ]
    return inputs


def _earlier_key(site: MuonSite) -> str:
    method = _run_method(site.generation_method)
    inputs = _earlier_inputs(method, site.properties)
    return repr(
        (
            method,
            str(site.source_structure_id or ""),
            sorted((key, repr(value)) for key, value in inputs.items()),
        )
    )


def _earlier_group(sites: Sequence[MuonSite], host_label: str | None) -> GenerationRunGroup:
    first = sites[0]
    method = _run_method(first.generation_method)
    run = GenerationRun(method=method, parameters=_earlier_inputs(method, first.properties))
    line = run_summary_line(run, count=len(sites))
    sentence = run_methods_sentence(run, host_label=host_label, count=len(sites))
    return GenerationRunGroup(
        key=f"earlier:{_earlier_key(first)}",
        run_id=None,
        title=f"{EARLIER_STARTS_TITLE}: {line}",
        description=f"{sentence} The seed and the host symmetry were not recorded.",
        site_ids=tuple(site.id for site in sites),
    )


def _host_label(document: ProjectDocument, structure_id: UUID | None) -> str | None:
    structure = document.structures.get(structure_id) if structure_id is not None else None
    if structure is None:
        return None
    return structure.label.strip() or None


def generation_run_groups(
    document: ProjectDocument, sites: Iterable[MuonSite]
) -> list[GenerationRunGroup]:
    """Group generated starts by the run that made them, in the order the page lists them.

    Earlier starts (no run, or a run that is not in the document) come first,
    one group per method and set of recorded inputs; then each run, oldest
    first. Starts are in natural label order within a group.
    """

    by_run: dict[UUID, list[MuonSite]] = {}
    earlier: dict[str, list[MuonSite]] = {}
    for site in sorted(sites, key=lambda item: natural_sort_key(item.label)):
        run_id = site.source_generation_batch_id
        if run_id is not None and run_id in document.generation_runs:
            by_run.setdefault(run_id, []).append(site)
        else:
            earlier.setdefault(_earlier_key(site), []).append(site)
    groups = [
        _earlier_group(members, _host_label(document, members[0].source_structure_id))
        for members in earlier.values()
    ]
    runs = sorted(
        (document.generation_runs[run_id] for run_id in by_run),
        key=lambda run: (run.created_at, str(run.id)),
    )
    for run in runs:
        groups.append(
            GenerationRunGroup(
                key=str(run.id),
                run_id=run.id,
                title=run_summary_line(run),
                description=run_methods_sentence(
                    run, host_label=_host_label(document, run.source_structure_id)
                ),
                site_ids=tuple(site.id for site in by_run[run.id]),
            )
        )
    return groups


__all__ = [
    "EARLIER_STARTS_TITLE",
    "GenerationRunGroup",
    "generation_run_from_provenance",
    "generation_run_groups",
    "run_journal_line",
    "run_methods_sentence",
    "run_summary_line",
]
