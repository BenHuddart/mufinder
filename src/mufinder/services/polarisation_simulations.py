"""Polarisation simulations of stored magnetic-field results.

A simulation combines the field results of one or more *sources*: the
Magnetic Fields page and each saved magnetic-field workspace hold at most one
field result (:mod:`mufinder.services.magnetic_field_results`), computed at
one muon site and its symmetry equivalents. :func:`field_sources` lists the
sources a project has, and :func:`run_polarisation_simulation` simulates a
weighted selection of them with
:func:`~mufinder.science.polarisation_simulation.simulate_polarisation`. A
source's weight is shared equally by the equivalent positions it holds.

Each simulation is a ``muon_polarisation`` record whose ``data["owner"]`` is
:func:`simulation_owner` of its id, so
:func:`~mufinder.services.magnetic_field_results.polarisation_result` reads its
static trace. The record also holds the title, the settings, the sources with
the id of the field record each was computed from, the dynamic trace and the
site source, which is :data:`UEP_SITE_SOURCE` when any source was computed at
an unrelaxed UEP site.

Recalculating or removing a source's fields replaces or deletes its field
record. The simulation is then stale (:func:`stale_sources`): it is kept,
and can be recalculated once every source has fields again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from mufinder.domain.models import Result
from mufinder.science.magnetic_fields import (
    MagneticFieldDistributionResult,
    MagneticFieldResult,
)
from mufinder.science.nuclear_relaxation import calculate_nuclear_width_tensors
from mufinder.science.polarisation import FieldResult, PolarisationError
from mufinder.science.polarisation_simulation import (
    PolarisationSettings,
    PolarisationSimulation,
    simulate_polarisation,
)
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    MAGNETIC_FIELD_RESULT_TYPE,
    MUON_POLARISATION_RESULT_TYPE,
    UEP_SITE_SOURCE,
    field_result_site_source,
    is_unrelaxed_site_source,
    magnetic_field_result,
)
from mufinder.services.project_document import ProjectDocument, active_host_structure
from mufinder.services.project_summary import magnetic_field_workspaces

#: Prefix of a simulation record's ``data["owner"]``; the simulation id follows it.
SIMULATION_OWNER_PREFIX = "polarisation:"
#: Title of the Magnetic Fields page as a source.
MAGNETIC_FIELD_PAGE_TITLE = "Magnetic Fields"


@dataclass(frozen=True, slots=True)
class FieldSource:
    """A stored field result a simulation can use.

    ``owner`` is :data:`MAGNETIC_FIELD_PAGE_OWNER` or a workspace id, and
    ``record_id`` the field record's id, which changes whenever the fields are
    recalculated. ``kind`` is ``"commensurate"`` or ``"distribution"``.
    ``applied_T`` is the applied field the results carry, ``None`` when they
    disagree.
    """

    owner: str
    record_id: str
    title: str
    kind: str
    site_labels: tuple[str, ...]
    site_source: str | None
    applied_T: tuple[float, float, float] | None

    @property
    def estimate(self) -> bool:
        return is_unrelaxed_site_source(self.site_source)

    @property
    def label(self) -> str:
        count = len(self.site_labels)
        site = self.site_labels[0] if self.site_labels else "-"
        kind = "distribution" if self.kind == "distribution" else "field"
        noun = f"{kind}s" if count != 1 else kind
        text = f"{self.title} · {count} {noun} at {site}"
        return f"{text} (unrelaxed estimate)" if self.estimate else text


@dataclass(frozen=True, slots=True)
class SimulationSource:
    """One source of a simulation, with its weight and the field record it used."""

    owner: str
    record_id: str
    title: str
    weight: float = 1.0

    def to_json(self) -> dict[str, object]:
        return {
            "owner": self.owner,
            "record_id": self.record_id,
            "title": self.title,
            "weight": float(self.weight),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> SimulationSource:
        return cls(
            owner=str(payload["owner"]),
            record_id=str(payload["record_id"]),
            title=str(payload.get("title", "")),
            weight=float(payload.get("weight", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class PolarisationSimulationRecord:
    """A stored simulation: what was asked for and what it gave."""

    simulation_id: str
    title: str
    settings: PolarisationSettings
    sources: tuple[SimulationSource, ...]
    simulation: PolarisationSimulation
    site_source: str | None = None
    created_at: datetime | None = field(default=None, compare=False)

    @property
    def owner(self) -> str:
        return simulation_owner(self.simulation_id)

    @property
    def estimate(self) -> bool:
        return is_unrelaxed_site_source(self.site_source)


def simulation_owner(simulation_id: str) -> str:
    """Return the ``data["owner"]`` of the simulation ``simulation_id``."""

    return f"{SIMULATION_OWNER_PREFIX}{simulation_id}"


def field_sources(document: ProjectDocument) -> list[FieldSource]:
    """Return the stored field results, the page first, then the workspaces in order."""

    owners = [(MAGNETIC_FIELD_PAGE_OWNER, MAGNETIC_FIELD_PAGE_TITLE)]
    for workspace in magnetic_field_workspaces(document):
        workspace_id = str(workspace.get("id", "")).strip()
        if workspace_id:
            owners.append((workspace_id, str(workspace.get("title") or workspace_id)))
    sources = []
    for owner, title in owners:
        source = _field_source(document, owner, title)
        if source is not None:
            sources.append(source)
    return sources


def field_results_for_source(document: ProjectDocument, owner: str) -> list[FieldResult]:
    """Return ``owner``'s stored field results as result objects."""

    record = magnetic_field_result(document, owner, MAGNETIC_FIELD_RESULT_TYPE)
    if record is None:
        raise PolarisationError(f"{owner!r} has no field results")
    return _field_results(record)


def run_polarisation_simulation(
    document: ProjectDocument,
    *,
    title: str,
    settings: PolarisationSettings,
    sources: Sequence[SimulationSource],
    simulation_id: str | None = None,
) -> PolarisationSimulationRecord:
    """Simulate ``sources`` with ``settings`` and return the record, not yet stored.

    Each source's ``record_id`` is set to the field record it is simulated
    from. ``simulation_id`` recalculates an existing simulation in place.
    """

    if not sources:
        raise PolarisationError("select at least one field result")
    host = active_host_structure(document)
    if host is None:
        raise PolarisationError("the project has no host structure")
    known = {source.owner: source for source in field_sources(document)}
    results: list[FieldResult] = []
    weights: list[float] = []
    used: list[SimulationSource] = []
    site_sources: list[str | None] = []
    for source in sources:
        current = known.get(source.owner)
        if current is None:
            raise PolarisationError(f"{source.title or source.owner} has no field results")
        if not source.weight >= 0.0:
            raise PolarisationError("source weights must be >= 0")
        entries = field_results_for_source(document, source.owner)
        results.extend(entries)
        weights.extend([float(source.weight) / len(entries)] * len(entries))
        used.append(
            SimulationSource(
                owner=current.owner,
                record_id=current.record_id,
                title=current.title,
                weight=float(source.weight),
            )
        )
        site_sources.append(current.site_source)
    if sum(weights) <= 0.0:
        raise PolarisationError("at least one source needs a positive weight")
    tensors = None
    if settings.sigma_from_nuclear_widths:
        tensors = [
            calculate_nuclear_width_tensors(
                host, result.muon_position, site_label=result.site_label
            )
            for result in results
        ]
    simulation = simulate_polarisation(
        settings,
        results,
        lattice_vectors=host.lattice_vectors,
        site_weights=weights,
        nuclear_width_tensors=tensors,
    )
    return PolarisationSimulationRecord(
        simulation_id=simulation_id or uuid4().hex,
        title=title.strip() or "Simulation",
        settings=settings,
        sources=tuple(used),
        simulation=simulation,
        site_source=UEP_SITE_SOURCE
        if any(is_unrelaxed_site_source(item) for item in site_sources)
        else next((item for item in site_sources if item), None),
    )


def store_polarisation_simulation(
    document: ProjectDocument, record: PolarisationSimulationRecord
) -> Result:
    """Store ``record``, replacing the simulation of the same id, and return the record."""

    existing = _simulation_result(document, record.simulation_id)
    data = {
        "owner": record.owner,
        "simulation_id": record.simulation_id,
        "title": record.title,
        "settings": record.settings.to_json(),
        "sources": [source.to_json() for source in record.sources],
        "result": record.simulation.static.to_json(),
        "dynamic": None
        if record.simulation.dynamic is None
        else record.simulation.dynamic.to_json(),
        "warnings": list(record.simulation.warnings),
    }
    if record.site_source:
        data["site_source"] = record.site_source
    if existing is not None:
        if existing.data == data:
            return existing
        del document.results[existing.id]
    result = Result(result_type=MUON_POLARISATION_RESULT_TYPE, data=data)
    if existing is not None:
        result.created_at = existing.created_at
    document.results[result.id] = result
    return result


def polarisation_simulations(document: ProjectDocument) -> list[PolarisationSimulationRecord]:
    """Return the stored simulations, oldest first; malformed records are skipped."""

    records = []
    for result in sorted(document.results.values(), key=lambda item: item.created_at):
        record = _simulation_from_result(result)
        if record is not None:
            records.append(record)
    return records


def polarisation_simulation(
    document: ProjectDocument, simulation_id: str
) -> PolarisationSimulationRecord | None:
    """Return the stored simulation ``simulation_id``, or ``None``."""

    result = _simulation_result(document, simulation_id)
    return None if result is None else _simulation_from_result(result)


def remove_polarisation_simulation(document: ProjectDocument, simulation_id: str) -> None:
    """Delete the stored simulation ``simulation_id``, if there is one."""

    result = _simulation_result(document, simulation_id)
    if result is not None:
        del document.results[result.id]


def stale_sources(
    document: ProjectDocument, record: PolarisationSimulationRecord
) -> list[SimulationSource]:
    """Return the sources whose fields were recalculated or removed since ``record`` ran."""

    stale = []
    for source in record.sources:
        current = magnetic_field_result(document, source.owner, MAGNETIC_FIELD_RESULT_TYPE)
        if current is None or str(current.id) != source.record_id:
            stale.append(source)
    return stale


def _field_source(document: ProjectDocument, owner: str, title: str) -> FieldSource | None:
    record = magnetic_field_result(document, owner, MAGNETIC_FIELD_RESULT_TYPE)
    if record is None:
        return None
    try:
        results = _field_results(record)
    except PolarisationError:
        return None
    applied = {_applied(result) for result in results}
    return FieldSource(
        owner=owner,
        record_id=str(record.id),
        title=title,
        kind=str(record.data.get("kind", "")),
        site_labels=tuple(result.site_label for result in results),
        site_source=field_result_site_source(record.data),
        applied_T=applied.pop() if len(applied) == 1 else None,
    )


def _field_results(record: Result) -> list[FieldResult]:
    data = record.data if isinstance(record.data, Mapping) else {}
    kind = data.get("kind")
    payloads = data.get("results")
    if not isinstance(payloads, list) or not payloads:
        raise PolarisationError("the field record holds no results")
    decode = {
        "commensurate": MagneticFieldResult.from_json,
        "distribution": MagneticFieldDistributionResult.from_json,
    }.get(str(kind))
    if decode is None:
        raise PolarisationError(f"unknown field result kind {kind!r}")
    try:
        results = [decode(payload) for payload in payloads]
    except (KeyError, TypeError, ValueError) as exc:
        raise PolarisationError(f"malformed field results: {exc}") from exc
    if any(
        isinstance(result, MagneticFieldDistributionResult) and result.fields_T.shape[0] == 0
        for result in results
    ):
        raise PolarisationError("a field distribution holds no samples")
    return results


def _applied(result: FieldResult) -> tuple[float, float, float]:
    applied = result.applied if isinstance(result, MagneticFieldResult) else result.applied_T
    if applied is None:
        return (0.0, 0.0, 0.0)
    x, y, z = (float(value) for value in applied)
    return (x, y, z)


def _simulation_result(document: ProjectDocument, simulation_id: str) -> Result | None:
    owner = simulation_owner(simulation_id)
    return next(
        (
            result
            for result in document.results.values()
            if result.result_type == MUON_POLARISATION_RESULT_TYPE
            and isinstance(result.data, Mapping)
            and result.data.get("owner") == owner
        ),
        None,
    )


def _simulation_from_result(result: Result) -> PolarisationSimulationRecord | None:
    data = result.data
    if result.result_type != MUON_POLARISATION_RESULT_TYPE or not isinstance(data, Mapping):
        return None
    owner = str(data.get("owner", ""))
    if not owner.startswith(SIMULATION_OWNER_PREFIX):
        return None
    try:
        settings = PolarisationSettings.from_json(data["settings"])
        simulation = PolarisationSimulation.from_json(
            {
                "static": data["result"],
                "dynamic": data.get("dynamic"),
                "warnings": data.get("warnings"),
            },
            settings,
        )
        sources = tuple(SimulationSource.from_json(item) for item in data["sources"])
    except (KeyError, TypeError, ValueError, PolarisationError):
        return None
    return PolarisationSimulationRecord(
        simulation_id=owner.removeprefix(SIMULATION_OWNER_PREFIX),
        title=str(data.get("title", "Simulation")),
        settings=settings,
        sources=sources,
        simulation=simulation,
        site_source=str(data["site_source"]) if data.get("site_source") else None,
        created_at=result.created_at,
    )
