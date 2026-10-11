"""Read-only source-candidate queries over a :class:`ProjectDocument`.

The quantum-corrections, energy-barriers and relaxation-analysis surfaces each
start from a list of completed upstream calculations: relaxed muon sites, the
diffusion paths generated between two of them, completed NEB results, and the
completed pristine and muon relaxations. This module builds those lists from the
project document. Every function here only reads the document; none mutates
project state or touches a widget.

Inputs that belong to a workflow controller (a definition's deserialized rows,
the row-to-job binding, the live row states) are resolved by the caller and
passed in, so this module depends on nothing above the ``services`` layer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import numpy as np

from mufinder.domain.charge_states import (
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    muon_charge_state_prefix,
    muon_relaxation_calc_type,
)
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Result, Structure
from mufinder.science.energy_barriers import (
    EnergyBarrierError,
    MuonPath,
    enumerate_distinct_muon_paths,
)
from mufinder.science.relaxation_analysis import (
    RelaxedMuonSite,
    folded_muon_position_from_structure,
    grouped_relative_energies,
)
from mufinder.services.project_service import ProjectDocument
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_SET_CALC_TYPE,
    merge_serialized_sources,
)
from mufinder.workflow.quantum import quantum_source_eligibility

#: Stage identifiers for a two-stage quantum-corrections source row.
QUANTUM_STAGE_SINGLEPOINT = "singlepoint"
QUANTUM_STAGE_PHONON = "phonon"

#: ``ui_state`` key of the Energy Barriers form, which holds the generated paths.
ENERGY_BARRIERS_FORM_STATE_KEY = "energy_barriers_form"


@dataclass(frozen=True, slots=True)
class QuantumSourceCandidate:
    """A relaxed muon site eligible as a quantum-corrections or barrier endpoint."""

    key: str
    site: MuonSite
    run: CalculationRun
    job: Job
    structure: Structure


@dataclass(frozen=True, slots=True)
class QuantumStageState:
    """Current two-stage quantum-corrections state for one source row."""

    stage: str
    status: str
    current_job: Job | None = None


@dataclass(frozen=True, slots=True)
class EnergyBarrierSourceCandidate:
    """One generated diffusion path between two relaxed endpoints."""

    key: str
    reactant: QuantumSourceCandidate
    product: QuantumSourceCandidate
    path: MuonPath
    path_index: int


@dataclass(frozen=True, slots=True)
class EnergyBarrierNebCandidate:
    """Endpoint path source eligible for NEB generation.

    NEB is seeded directly from the two relaxed endpoints; CASTEP interpolates
    the band between them, so no transition-state search is required.
    """

    key: str
    source: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class EnergyBarrierCompletedNebCandidate:
    """Completed NEB result eligible for preview."""

    key: str
    run: CalculationRun
    job: Job
    result: Result


@dataclass(frozen=True, slots=True)
class CompletedRelaxationAnalysisSource:
    """Source row for a completed muon structural relaxation."""

    job_id: UUID
    job_label: str
    charge_state: str
    structure_id: UUID
    site_id: UUID | None
    folded_fractional_position: tuple[float, float, float]
    total_energy_ev: float | None

    def to_relaxed_site(self, host_structure: Structure) -> RelaxedMuonSite:
        cart = tuple(
            float(value)
            for value in (
                np.asarray(self.folded_fractional_position, dtype=float)
                @ np.asarray(host_structure.lattice_vectors, dtype=float)
            ).tolist()
        )
        return RelaxedMuonSite(
            job_id=self.job_id,
            job_label=self.job_label,
            charge_state=self.charge_state,
            structure_id=self.structure_id,
            site_id=self.site_id,
            folded_fractional_position=self.folded_fractional_position,
            cartesian_position_ang=cart,
            total_energy_ev=self.total_energy_ev,
        )


@dataclass(frozen=True, slots=True)
class CompletedPristineJob:
    """Display row describing one completed pristine-relaxation job."""

    job_id: UUID
    job_label: str
    structure_id: UUID
    supercell_label: str
    status: str
    total_energy: float | None
    lattice_a: float | None
    lattice_b: float | None
    lattice_c: float | None
    parameter_values: dict[str, object]


class PristineRelaxationRowLike(Protocol):
    """The pristine-relaxation row fields :func:`completed_pristine_relaxation_jobs` reads."""

    label: str
    parameter_values: Mapping[str, object]
    supercell_matrix: Sequence[Sequence[int]] | None


class RelaxationRowStateLike(Protocol):
    """The row-state fields :func:`completed_pristine_relaxation_jobs` reads."""

    status: str
    job_id: UUID | None
    total_energy: float | None
    lattice_a: float | None
    lattice_b: float | None
    lattice_c: float | None


#: One muon-relaxation row resolved by its controller:
#: ``(row label, charge state, source site id, bound job)``.
MuonRelaxationRowJob = tuple[str, str, UUID | None, Job | None]


# --- serialized source fields -------------------------------------------------


def uuid_from_source(source: Mapping[str, object], key: str) -> UUID | None:
    """Return ``source[key]`` parsed as a UUID, or ``None`` when absent or invalid."""

    raw = source.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return UUID(raw)
    except ValueError:
        return None


def fractional_triplet_from_source(source: Mapping[str, object], key: str) -> tuple[float, float, float]:
    """Return ``source[key]`` as three floats; raise ``ValueError`` when malformed."""

    raw = source.get(key)
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"Expected three fractional coordinates for {key}.")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def float_from_source(source: Mapping[str, object], key: str) -> float | None:
    """Return ``source[key]`` as a float, or ``None`` when absent or not numeric."""

    raw = source.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


# --- quantum-corrections sources ----------------------------------------------


def charge_state_for_job(job: Job | None) -> str | None:
    """Return the muon charge state a muon-relaxation job ran in, if any."""

    if job is None:
        return None
    if job.calc_type == muon_relaxation_calc_type(POSITIVE_MUON_CHARGE_STATE):
        return POSITIVE_MUON_CHARGE_STATE
    if job.calc_type == muon_relaxation_calc_type(MUONIUM_CHARGE_STATE):
        return MUONIUM_CHARGE_STATE
    return None


def eligible_quantum_source_candidates(document: ProjectDocument | None) -> list[QuantumSourceCandidate]:
    """Return every relaxed muon site that can seed a quantum-corrections row.

    A site qualifies when :func:`~mufinder.workflow.quantum.quantum_source_eligibility`
    accepts it and its relaxation run, job and first derived structure all exist.
    Candidates are ordered by charge state, then relative energy within their
    comparable group (sources without an energy last), then site label.
    """

    if document is None:
        return []
    candidates: list[QuantumSourceCandidate] = []
    for site in document.sites.values():
        run = document.runs.get(site.relaxation_run_id) if site.relaxation_run_id is not None else None
        job = document.jobs.get(run.job_id) if run is not None and run.job_id is not None else None
        result = document.results.get(run.result_id) if run is not None and run.result_id is not None else None
        relaxed_structure = None
        if result is not None and result.derived_structure_ids:
            relaxed_structure = document.structures.get(result.derived_structure_ids[0])
        verdict = quantum_source_eligibility(
            site=site,
            run=run,
            job=job,
            relaxed_structure=relaxed_structure,
        )
        if not verdict.eligible or run is None or job is None or relaxed_structure is None:
            continue
        candidates.append(
            QuantumSourceCandidate(
                key=str(site.id),
                site=site,
                run=run,
                job=job,
                structure=relaxed_structure,
            )
        )
    relative = quantum_source_relative_energies(candidates)

    def order(item: QuantumSourceCandidate) -> tuple[object, ...]:
        energy = relative.get(item.key)
        charge_state = charge_state_for_job(item.job) or ""
        return (charge_state, energy is None, energy or 0.0, item.site.label)

    return sorted(candidates, key=order)


def quantum_source_relative_energies(
    candidates: Iterable[QuantumSourceCandidate],
) -> dict[str, float | None]:
    """Return each relaxed source's energy relative to its comparable minimum.

    ``MuonSite.total_energy_eV`` holds the relaxation's absolute final total
    energy, which is only comparable between relaxations of the same cell in the
    same charge state. Sources are therefore grouped by charge state and cell
    composition (which separates different supercells), and each is reported
    relative to the lowest-energy source in its group.
    """

    entries: dict[str, tuple[object, float | None]] = {}
    for candidate in candidates:
        composition = tuple(sorted(Counter(candidate.structure.species).items()))
        group = (charge_state_for_job(candidate.job), composition)
        entries[candidate.key] = (group, candidate.site.total_energy_eV)
    return grouped_relative_energies(entries)


def quantum_source_candidate_by_key(
    document: ProjectDocument | None,
    key: str,
) -> QuantumSourceCandidate | None:
    """Return the eligible quantum source with ``key``, or ``None``."""

    for candidate in eligible_quantum_source_candidates(document):
        if candidate.key == key:
            return candidate
    return None


def quantum_source_labels_by_key(candidates: Iterable[QuantumSourceCandidate]) -> dict[str, str]:
    """Return a unique display label per source key.

    The base label is the site label, else the relaxation job label, else
    ``"site"``; repeated base labels get ``_2``, ``_3``, ... in candidate order.
    """

    labels_by_key: dict[str, str] = {}
    counts: dict[str, int] = {}
    for candidate in candidates:
        base = str(candidate.site.label or candidate.job.label or "site").strip() or "site"
        count = counts.get(base, 0) + 1
        counts[base] = count
        labels_by_key[candidate.key] = base if count == 1 else f"{base}_{count}"
    return labels_by_key


def quantum_stage_state(
    *,
    singlepoint_job: Job | None,
    phonon_job: Job | None,
    row_status: Callable[[Job], str],
) -> QuantumStageState:
    """Resolve one quantum-corrections row's current stage from its two jobs.

    A phonon job puts the row on the phonon stage with that job's status. With
    no phonon job, a completed singlepoint makes the phonon stage ready, any
    other singlepoint keeps the row on the singlepoint stage with its status,
    and no singlepoint at all leaves the singlepoint stage ready. ``row_status``
    projects a job onto the status string its workflow-table row shows.
    """

    if phonon_job is not None:
        return QuantumStageState(
            stage=QUANTUM_STAGE_PHONON,
            status=row_status(phonon_job),
            current_job=phonon_job,
        )
    if singlepoint_job is None:
        return QuantumStageState(stage=QUANTUM_STAGE_SINGLEPOINT, status="ready")
    if singlepoint_job.status == JobStatus.COMPLETED:
        return QuantumStageState(stage=QUANTUM_STAGE_PHONON, status="ready")
    return QuantumStageState(
        stage=QUANTUM_STAGE_SINGLEPOINT,
        status=row_status(singlepoint_job),
        current_job=singlepoint_job,
    )


# --- energy-barrier sources ---------------------------------------------------


def energy_barrier_generated_sources(document: ProjectDocument | None) -> list[dict[str, object]]:
    """Return every generated diffusion-path source, serialized.

    The union, merged by ``key``, of the paths held in the Energy Barriers form
    state and the ``sources`` stored in each energy-barriers definition's
    configuration, in document order.
    """

    if document is None:
        return []
    state = document.ui_state.get(ENERGY_BARRIERS_FORM_STATE_KEY, {})
    raw_sources = state.get("generated_sources", []) if isinstance(state, dict) else []
    sources = (
        [dict(source) for source in raw_sources if isinstance(source, dict)]
        if isinstance(raw_sources, list)
        else []
    )
    for definition in document.calculations.values():
        if (
            definition.workflow_stage != WorkflowStage.ENERGY_BARRIERS
            or definition.calc_type != TRANSITION_STATE_SEARCH_SET_CALC_TYPE
        ):
            continue
        config = definition.parameters.get("configuration", {})
        if not isinstance(config, dict):
            continue
        raw_sources = config.get("sources", [])
        if not isinstance(raw_sources, list):
            continue
        sources = merge_serialized_sources(
            sources,
            [dict(source) for source in raw_sources if isinstance(source, dict)],
            key_field="key",
        )
    return sources


def enumerate_energy_barrier_source_candidates(
    document: ProjectDocument | None,
    reactant_key: str,
    product_key: str,
    *,
    max_distance_ang: float | None = None,
) -> list[EnergyBarrierSourceCandidate]:
    """Enumerate the distinct muon paths between two eligible quantum sources.

    Returns nothing when either endpoint is not eligible, the endpoints differ in
    charge state, either relaxed structure does not descend from the project
    host, or the path enumeration fails. Paths are numbered from 1.
    """

    if document is None:
        return []
    host_id = document.project.host_structure_id
    host = document.structures.get(host_id) if host_id is not None else None
    candidates = eligible_quantum_source_candidates(document)
    by_key = {candidate.key: candidate for candidate in candidates}
    reactant = by_key.get(reactant_key)
    product = by_key.get(product_key)
    if host is None or reactant is None or product is None:
        return []
    if charge_state_for_job(reactant.job) != charge_state_for_job(product.job):
        return []
    if reactant.structure.parent_structure_id != host.id or product.structure.parent_structure_id != host.id:
        return []
    try:
        paths = enumerate_distinct_muon_paths(
            pristine_structure=host,
            reactant_structure=reactant.structure,
            product_structure=product.structure,
            max_distance_ang=max_distance_ang,
        )
    except EnergyBarrierError:
        return []
    return [
        EnergyBarrierSourceCandidate(
            key=f"{reactant.site.id}:{product.site.id}:{path_index}",
            reactant=reactant,
            product=product,
            path=path,
            path_index=path_index,
        )
        for path_index, path in enumerate(paths, start=1)
    ]


_ENDPOINT_FIELDS = ("site", "run", "job", "structure")


def eligible_energy_barrier_source_candidates(
    document: ProjectDocument | None,
) -> list[EnergyBarrierSourceCandidate]:
    """Rebuild the generated diffusion paths whose endpoint records still exist.

    Each serialized source from :func:`energy_barrier_generated_sources` is
    resolved against the document; a source with a missing or malformed id, a
    deleted site, run, job or structure, or an unreadable path is skipped.
    """

    if document is None:
        return []
    stores = {
        "site": document.sites,
        "run": document.runs,
        "job": document.jobs,
        "structure": document.structures,
    }
    candidates: list[EnergyBarrierSourceCandidate] = []
    for raw_source in energy_barrier_generated_sources(document):
        ids = {
            (side, field): uuid_from_source(raw_source, f"{side}_{field}_id")
            for side in ("reactant", "product")
            for field in _ENDPOINT_FIELDS
        }
        if any(value is None for value in ids.values()):
            continue
        records = {
            (side, field): stores[field].get(record_id)
            for (side, field), record_id in ids.items()
        }
        if any(value is None for value in records.values()):
            continue
        try:
            path_index = int(raw_source.get("path_index", 1))
            path = MuonPath(
                start_fractional=fractional_triplet_from_source(raw_source, "path_start_fractional"),
                end_fractional=fractional_triplet_from_source(raw_source, "path_end_fractional"),
                length_ang=float(raw_source.get("path_length_ang", 0.0)),
            )
        except (TypeError, ValueError):
            continue
        endpoints = {
            side: QuantumSourceCandidate(
                key=str(records[(side, "site")].id),
                site=records[(side, "site")],
                run=records[(side, "run")],
                job=records[(side, "job")],
                structure=records[(side, "structure")],
            )
            for side in ("reactant", "product")
        }
        candidates.append(
            EnergyBarrierSourceCandidate(
                key=str(raw_source.get("key", "")).strip(),
                reactant=endpoints["reactant"],
                product=endpoints["product"],
                path=path,
                path_index=path_index,
            )
        )
    return candidates


def energy_barrier_neb_source_candidates(document: ProjectDocument | None) -> list[EnergyBarrierNebCandidate]:
    """Return every keyed generated path as an NEB source, sorted by key."""

    if document is None:
        return []
    candidates: list[EnergyBarrierNebCandidate] = []
    for source in energy_barrier_generated_sources(document):
        key = str(source.get("key", "")).strip()
        if not key:
            continue
        candidates.append(EnergyBarrierNebCandidate(key=key, source=source))
    return sorted(candidates, key=lambda item: item.key)


def completed_energy_barrier_neb_candidates(
    document: ProjectDocument | None,
) -> list[EnergyBarrierCompletedNebCandidate]:
    """Return every NEB result whose run and job exist, sorted by job label."""

    if document is None:
        return []
    candidates: list[EnergyBarrierCompletedNebCandidate] = []
    for result in document.results.values():
        if result.result_type != NUDGED_ELASTIC_BAND_CALC_TYPE:
            continue
        run = document.runs.get(result.calculation_run_id) if result.calculation_run_id is not None else None
        job = document.jobs.get(run.job_id) if run is not None and run.job_id is not None else None
        if run is None or job is None:
            continue
        candidates.append(
            EnergyBarrierCompletedNebCandidate(
                key=str(result.id),
                run=run,
                job=job,
                result=result,
            )
        )
    return sorted(candidates, key=lambda item: (item.job.label, item.key))


# --- completed relaxations ----------------------------------------------------


def coerce_fractional_triplet(value: object) -> tuple[float, float, float] | None:
    """Return ``value`` as three floats, or ``None`` when it is not a numeric triplet."""

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def completed_muon_relaxation_sources(
    document: ProjectDocument | None,
    rows: Iterable[MuonRelaxationRowJob],
    *,
    run_for_job: Callable[[UUID], CalculationRun | None],
    total_energy_for_job: Callable[[UUID], float | None],
) -> list[CompletedRelaxationAnalysisSource]:
    """Return the converged, completed muon relaxations as analysis sources.

    ``rows`` are the host's muon-relaxation rows with their bound jobs. A row
    contributes once per job when the job completed, its run converged and has
    a result with a derived structure, and a folded host muon position is known
    (from the result payload, else from the relaxed structure). Sources are
    sorted by total energy (unknown last), then case-insensitive label.
    """

    if document is None:
        return []
    sources: list[CompletedRelaxationAnalysisSource] = []
    seen: set[UUID] = set()
    for label, charge_state, site_id, job in rows:
        if job is None or job.status != JobStatus.COMPLETED or job.id in seen:
            continue
        run = run_for_job(job.id)
        if run is None or run.result_id is None or run.convergence_achieved is False:
            continue
        result = document.results.get(run.result_id)
        if result is None or not result.derived_structure_ids:
            continue
        structure_id = result.derived_structure_ids[0]
        structure = document.structures.get(structure_id)
        if structure is None:
            continue
        fractional = None
        if isinstance(result.data, dict):
            fractional = coerce_fractional_triplet(result.data.get("muon_final_fractional_position_host"))
        if fractional is None:
            fractional = folded_muon_position_from_structure(structure)
        if fractional is None:
            continue
        sources.append(
            CompletedRelaxationAnalysisSource(
                job_id=job.id,
                job_label=label,
                charge_state=charge_state,
                structure_id=structure_id,
                site_id=site_id,
                folded_fractional_position=fractional,
                total_energy_ev=total_energy_for_job(job.id),
            )
        )
        seen.add(job.id)
    return sorted(
        sources,
        key=lambda source: (
            float("inf") if source.total_energy_ev is None else float(source.total_energy_ev),
            source.job_label.casefold(),
        ),
    )


def relaxation_sources_for_labels(
    sources: Iterable[CompletedRelaxationAnalysisSource],
    labels: Iterable[str],
) -> list[CompletedRelaxationAnalysisSource]:
    """Select sources by ``"{charge prefix}:{label}"`` key or by bare job label."""

    wanted = {str(label) for label in labels}
    selected: list[CompletedRelaxationAnalysisSource] = []
    for source in sources:
        key = f"{muon_charge_state_prefix(source.charge_state)}:{source.job_label}"
        if key in wanted or source.job_label in wanted:
            selected.append(source)
    return selected


def _supercell_label(matrix: Sequence[Sequence[int]] | None) -> str:
    if matrix is None:
        return "Unit cell"
    return "[" + "; ".join(",".join(str(int(value)) for value in vector) for vector in matrix) + "]"


def completed_pristine_relaxation_jobs(
    document: ProjectDocument | None,
    rows: Iterable[PristineRelaxationRowLike],
    states: Mapping[str, RelaxationRowStateLike],
    *,
    run_for_job: Callable[[UUID], CalculationRun | None],
) -> list[CompletedPristineJob]:
    """Return the completed pristine relaxations that produced a relaxed structure.

    ``rows`` are the active pristine-relaxation definition's rows and ``states``
    its live row states keyed by row label. Rows are returned in label order.
    """

    if document is None:
        return []
    row_list = list(rows)
    parameter_values_by_label = {row.label: dict(row.parameter_values) for row in row_list}
    supercell_label_by_row = {row.label: _supercell_label(row.supercell_matrix) for row in row_list}

    completed: list[CompletedPristineJob] = []
    for row_label, state in sorted(states.items()):
        if state.status != JobStatus.COMPLETED.value:
            continue
        if state.job_id is None:
            continue
        run = run_for_job(state.job_id)
        if run is None or run.result_id is None:
            continue
        result = document.results.get(run.result_id)
        if result is None or not result.derived_structure_ids:
            continue
        completed.append(
            CompletedPristineJob(
                job_id=state.job_id,
                job_label=row_label,
                structure_id=result.derived_structure_ids[0],
                supercell_label=supercell_label_by_row.get(row_label, "Unit cell"),
                status=state.status,
                total_energy=state.total_energy,
                lattice_a=state.lattice_a,
                lattice_b=state.lattice_b,
                lattice_c=state.lattice_c,
                parameter_values=parameter_values_by_label.get(row_label, {}),
            )
        )
    return completed


__all__ = [
    "CompletedPristineJob",
    "CompletedRelaxationAnalysisSource",
    "ENERGY_BARRIERS_FORM_STATE_KEY",
    "EnergyBarrierCompletedNebCandidate",
    "EnergyBarrierNebCandidate",
    "EnergyBarrierSourceCandidate",
    "MuonRelaxationRowJob",
    "PristineRelaxationRowLike",
    "QUANTUM_STAGE_PHONON",
    "QUANTUM_STAGE_SINGLEPOINT",
    "QuantumSourceCandidate",
    "QuantumStageState",
    "RelaxationRowStateLike",
    "charge_state_for_job",
    "coerce_fractional_triplet",
    "completed_energy_barrier_neb_candidates",
    "completed_muon_relaxation_sources",
    "completed_pristine_relaxation_jobs",
    "eligible_energy_barrier_source_candidates",
    "eligible_quantum_source_candidates",
    "energy_barrier_generated_sources",
    "energy_barrier_neb_source_candidates",
    "enumerate_energy_barrier_source_candidates",
    "float_from_source",
    "fractional_triplet_from_source",
    "quantum_source_candidate_by_key",
    "quantum_source_labels_by_key",
    "quantum_source_relative_energies",
    "quantum_stage_state",
    "relaxation_sources_for_labels",
    "uuid_from_source",
]
