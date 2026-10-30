"""Read-only source-candidate queries over a :class:`ProjectDocument`.

The quantum-corrections, energy-barriers and relaxation-analysis surfaces each
start from a list of completed upstream calculations: relaxed muon sites, the
diffusion paths generated between two of them, completed NEB results, and the
completed pristine and muon relaxations. This module builds those lists from the
project document, together with the labels, seed names and stored ``sources``
entries derived from them and the settings a follow-on row inherits from its
source relaxation's input files. Every function here only reads; none mutates
project state or touches a widget.

Inputs that belong to a workflow controller (a definition's deserialized rows,
the row-to-job binding, the live row states) are resolved by the caller and
passed in, so this module depends on nothing above the ``services`` layer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol
from uuid import UUID

import numpy as np

from mufinder.castep_io.cell_parser import CellParseError, ParsedCell, parse_cell_file
from mufinder.castep_io.param_parser import parse_param_assignments, run_is_spin_polarized
from mufinder.domain.charge_states import (
    MUONIUM_CHARGE_STATE,
    canonical_charge_state,
    muon_charge_state_label,
    muon_charge_state_prefix,
    muon_relaxation_charge_state,
)
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    Job,
    MuonSite,
    Result,
    Structure,
)
from mufinder.persistence.codec import float_triplet_or_none
from mufinder.science.energy_barriers import (
    EnergyBarrierError,
    MuonPath,
    enumerate_distinct_muon_paths,
    equivalent_path_indexes,
    path_symmetry_uses_magnetic_order,
)
from mufinder.science.lattice import lattice_frac_to_cart, wrap_fractional
from mufinder.science.relaxation_analysis import (
    RelaxedMuonSite,
    grouped_relative_energies,
    muon_fractional_position_from_cell,
)
from mufinder.science.supercell import build_supercell_structure
from mufinder.services.job_paths import job_seed_name
from mufinder.services.job_results import total_energy_for_job
from mufinder.services.project_document import definitions_for_stage
from mufinder.services.project_service import ProjectDocument
from mufinder.services.project_summary import (
    latest_run_for_job,
    muon_relaxation_job_for_row,
    muon_relaxation_row_jobs,
)
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_SET_CALC_TYPE,
    is_pair_source,
    merge_serialized_sources,
)
from mufinder.workflow.muon_site import MUON_RELAXATION_DEFINITION_CALC_TYPE
from mufinder.workflow.naming import job_name
from mufinder.workflow.quantum import quantum_source_eligibility

#: ``ui_state`` key of the Energy Barriers form, which holds the generated paths.
ENERGY_BARRIERS_FORM_STATE_KEY = "energy_barriers_form"

#: ``MuonSite.properties`` key naming the charge state whose relaxation of the
#: site keys by the bare site id. Written when the site is first relaxed and
#: never changed, so that key keeps naming one charge state.
SITE_KEY_CHARGE_STATE_PROPERTY = "source_key_charge_state"


@dataclass(frozen=True, slots=True)
class QuantumSourceCandidate:
    """A relaxed muon site in one charge state, eligible as a quantum-corrections or barrier endpoint.

    A site relaxed in both charge states gives one candidate per charge state.
    ``key`` is the site id for the relaxation the site's own link records and
    ``"{site id}:{charge-state prefix}"`` for a relaxation in another charge
    state (see :func:`quantum_source_key`). ``total_energy_eV`` is that
    relaxation's absolute final total energy.
    """

    key: str
    site: MuonSite
    run: CalculationRun
    job: Job
    structure: Structure
    total_energy_eV: float | None = None


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
    #: What makes two relaxations' total energies comparable: the supercell and the
    #: energy-defining settings their row stores (:func:`relaxation_calculation_key`).
    calculation: tuple[object, ...] = ()

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


def fractional_triplet_from_source(
    source: Mapping[str, object], key: str
) -> tuple[float, float, float]:
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

    return None if job is None else muon_relaxation_charge_state(job.calc_type)


def charge_state_label_for_job(job: Job | None) -> str:
    """Return the user-facing charge-state label of a muon-relaxation job, or ``""``."""

    charge_state = charge_state_for_job(job)
    return muon_charge_state_label(charge_state) if charge_state is not None else ""


def site_linked_charge_state(document: ProjectDocument, site: MuonSite) -> str | None:
    """Return the charge state of the relaxation ``site``'s own link records, if any."""

    run = document.runs.get(site.relaxation_run_id) if site.relaxation_run_id is not None else None
    job = document.jobs.get(run.job_id) if run is not None and run.job_id is not None else None
    return charge_state_for_job(job)


def site_key_charge_state(document: ProjectDocument, site: MuonSite) -> str | None:
    """Return the charge state whose relaxation of ``site`` keys by the bare site id.

    The charge state recorded under :data:`SITE_KEY_CHARGE_STATE_PROPERTY`, else
    (a site linked before the property existed) the charge state of the
    relaxation the site's own link records.
    """

    raw = site.properties.get(SITE_KEY_CHARGE_STATE_PROPERTY)
    stored = canonical_charge_state(raw) if isinstance(raw, str) else None
    return stored or site_linked_charge_state(document, site)


def quantum_source_key(document: ProjectDocument, site: MuonSite, job: Job | None) -> str:
    """Return the source key of ``site`` relaxed by ``job``.

    The relaxation in the charge state :func:`site_key_charge_state` names keys
    by the site id, so sources stored before a site could be relaxed in both
    charge states keep their keys, and resetting that relaxation never hands
    the key to the other charge state; a relaxation in another charge state
    keys by ``"{site id}:{charge-state prefix}"``.
    """

    charge_state = charge_state_for_job(job)
    if charge_state is None or charge_state == site_key_charge_state(document, site):
        return str(site.id)
    return f"{site.id}:{muon_charge_state_prefix(charge_state)}"


def site_charge_state_relaxation_jobs(document: ProjectDocument) -> dict[UUID, dict[str, Job]]:
    """Return each site's completed muon-relaxation job per charge state.

    Read from the muon-relaxation rows bound to a site: a start relaxed as both
    a positive muon and muonium has one completed job in each charge state,
    while the site's own link records only one of them. The first completed
    job in document order wins when several rows relax a site in the same
    charge state.
    """

    jobs: dict[UUID, dict[str, Job]] = {}
    for definition in definitions_for_stage(
        document,
        WorkflowStage.MUON_RELAXATION,
        calc_type=MUON_RELAXATION_DEFINITION_CALC_TYPE,
    ):
        for _label, _row_charge_state, site_id, job in muon_relaxation_row_jobs(
            document, definition
        ):
            charge_state = charge_state_for_job(job)
            if (
                site_id is None
                or job is None
                or charge_state is None
                or job.status != JobStatus.COMPLETED
            ):
                continue
            jobs.setdefault(site_id, {}).setdefault(charge_state, job)
    return jobs


def _relaxed_structure_for_run(
    document: ProjectDocument, run: CalculationRun | None
) -> Structure | None:
    result = (
        document.results.get(run.result_id)
        if run is not None and run.result_id is not None
        else None
    )
    if result is None or not result.derived_structure_ids:
        return None
    return document.structures.get(result.derived_structure_ids[0])


def eligible_quantum_source_candidates(
    document: ProjectDocument | None,
) -> list[QuantumSourceCandidate]:
    """Return every relaxed muon site, per charge state, that can seed a quantum-corrections row.

    A site qualifies through the relaxation its own link records, and through
    the latest run of each completed muon-relaxation job of another charge
    state bound to it by a muon-relaxation row (see
    :func:`site_charge_state_relaxation_jobs`), when
    :func:`~mufinder.workflow.quantum.quantum_source_eligibility` accepts it and
    the relaxation run, job and first derived structure all exist. Candidates
    are ordered by charge state, then relative energy within their comparable
    group (sources without an energy last), then site label.
    """

    if document is None:
        return []
    row_jobs = site_charge_state_relaxation_jobs(document)
    candidates: list[QuantumSourceCandidate] = []
    for site in document.sites.values():
        run = (
            document.runs.get(site.relaxation_run_id)
            if site.relaxation_run_id is not None
            else None
        )
        job = document.jobs.get(run.job_id) if run is not None and run.job_id is not None else None
        relaxed_structure = _relaxed_structure_for_run(document, run)
        verdict = quantum_source_eligibility(
            site=site,
            run=run,
            job=job,
            relaxed_structure=relaxed_structure,
        )
        linked_charge_state = charge_state_for_job(job)
        if (
            verdict.eligible
            and run is not None
            and job is not None
            and relaxed_structure is not None
        ):
            candidates.append(
                QuantumSourceCandidate(
                    key=quantum_source_key(document, site, job),
                    site=site,
                    run=run,
                    job=job,
                    structure=relaxed_structure,
                    total_energy_eV=site.total_energy_eV,
                )
            )
        for charge_state, row_job in row_jobs.get(site.id, {}).items():
            if charge_state == linked_charge_state or (job is not None and row_job.id == job.id):
                continue
            row_run = latest_run_for_job(document, row_job.id)
            row_structure = _relaxed_structure_for_run(document, row_run)
            verdict = quantum_source_eligibility(
                site=site,
                run=row_run,
                job=row_job,
                relaxed_structure=row_structure,
                site_link=False,
            )
            if not verdict.eligible or row_run is None or row_structure is None:
                continue
            candidates.append(
                QuantumSourceCandidate(
                    key=quantum_source_key(document, site, row_job),
                    site=site,
                    run=row_run,
                    job=row_job,
                    structure=row_structure,
                    total_energy_eV=total_energy_for_job(document, row_job.id),
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

    ``QuantumSourceCandidate.total_energy_eV`` holds the relaxation's absolute
    final total energy, which is only comparable between relaxations of the same cell in the
    same charge state. Sources are therefore grouped by charge state and cell
    composition (which separates different supercells), and each is reported
    relative to the lowest-energy source in its group.
    """

    entries: dict[str, tuple[object, float | None]] = {}
    for candidate in candidates:
        composition = tuple(sorted(Counter(candidate.structure.species).items()))
        group = (charge_state_for_job(candidate.job), composition)
        entries[candidate.key] = (group, candidate.total_energy_eV)
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


def quantum_source_base_label(candidate: QuantumSourceCandidate, fallback: str = "site") -> str:
    """Return a source's label before de-duplication.

    The site label, else the relaxation job label, else ``fallback``. A
    relaxation in a charge state other than the site's own link adds the
    charge-state prefix (``"O1_muonium"``), so one site's two charge states
    name their follow-on jobs apart.
    """

    base = str(candidate.site.label or candidate.job.label or fallback).strip() or fallback
    if candidate.key != str(candidate.site.id):
        charge_state = charge_state_for_job(candidate.job)
        if charge_state is not None:
            base = f"{base}_{muon_charge_state_prefix(charge_state)}"
    return base


def quantum_source_labels_by_key(candidates: Iterable[QuantumSourceCandidate]) -> dict[str, str]:
    """Return a unique display label per source key.

    The base label is :func:`quantum_source_base_label`; repeated base labels
    get ``_2``, ``_3``, ... in candidate order.
    """

    labels_by_key: dict[str, str] = {}
    counts: dict[str, int] = {}
    for candidate in candidates:
        base = quantum_source_base_label(candidate)
        count = counts.get(base, 0) + 1
        counts[base] = count
        labels_by_key[candidate.key] = base if count == 1 else f"{base}_{count}"
    return labels_by_key


def quantum_source_label(
    document: ProjectDocument | None, candidate: QuantumSourceCandidate
) -> str:
    """Return ``candidate``'s unique display label among the eligible sources."""

    labels = quantum_source_labels_by_key(eligible_quantum_source_candidates(document))
    return labels.get(candidate.key, "site")


def quantum_seed_name_for_key(
    document: ProjectDocument | None,
    definition: CalculationDefinition | None,
    key: str,
) -> str:
    """Return the seed name a quantum-corrections row's phonon job runs under.

    The name recorded in the definition's ``job_seed_names_by_label`` wins;
    otherwise it is derived from the source's display label, or ``"quantum"``
    when the source is no longer eligible.
    """

    mapping = (
        definition.parameters.get("job_seed_names_by_label", {}) if definition is not None else {}
    )
    raw_seed = mapping.get(key) if isinstance(mapping, dict) else None
    if isinstance(raw_seed, str) and raw_seed.strip():
        return raw_seed.strip()
    candidate = quantum_source_candidate_by_key(document, key)
    if candidate is None:
        return "quantum"
    return job_name("quantum", quantum_source_label(document, candidate))


def serialize_quantum_sources(
    document: ProjectDocument | None,
    keys: Iterable[str | UUID],
) -> list[dict[str, object]]:
    """Return the stored ``sources`` entries of the eligible sources ``keys`` name.

    A key is a source key (a bare site id names the site's own relaxation);
    repeated keys are stored once.
    """

    serialized: list[dict[str, object]] = []
    for key in dict.fromkeys(str(raw_key) for raw_key in keys):
        candidate = quantum_source_candidate_by_key(document, key)
        if candidate is None:
            continue
        serialized.append(
            {
                "key": candidate.key,
                "site_id": str(candidate.site.id),
                "source_relaxation_run_id": str(candidate.run.id),
                "source_relaxation_job_id": str(candidate.job.id),
                "relaxed_structure_id": str(candidate.structure.id),
                "label": quantum_source_label(document, candidate),
                "charge_state": charge_state_for_job(candidate.job),
            }
        )
    return serialized


def source_relaxation_parameter_values(
    job: Job,
    output_directory: Callable[[Job], Path | None],
) -> dict[str, object]:
    """Return the settings a follow-on row inherits from its source relaxation.

    Read from the relaxation's own ``.cell`` and ``.param``: the k-point grid,
    the cut-off energy, the exchange-correlation functional and the spin
    treatment, so the follow-on calculation stays comparable with the
    relaxation. A file that is missing or does not parse contributes nothing,
    except that a muonium source without a readable ``.param`` is taken to be
    spin-polarised, the default its relaxation is written with.
    """

    values: dict[str, object] = {}
    output_dir = output_directory(job)
    if output_dir is None:
        return values

    seed_name = job_seed_name(job, output_directory)
    cell_path = output_dir / f"{seed_name}.cell"
    parsed_cell: ParsedCell | None = None
    if cell_path.is_file():
        try:
            parsed_cell = parse_cell_file(cell_path)
        except (CellParseError, OSError, ValueError):
            parsed_cell = None
        if parsed_cell is not None and parsed_cell.cell_file.kpoint_mp_grid is not None:
            values["kpoint_mp_grid"] = tuple(
                int(component) for component in parsed_cell.cell_file.kpoint_mp_grid
            )

    param_path = output_dir / f"{seed_name}.param"
    assignments: dict[str, object] | None = None
    if param_path.is_file():
        try:
            assignments = parse_param_assignments(param_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            assignments = None
    if assignments is not None:
        cutoff = assignments.get("cut_off_energy")
        if cutoff is not None:
            values["cut_off_energy"] = cutoff
        xc_functional = assignments.get("xc_functional")
        if xc_functional is not None:
            values["xc_functional"] = xc_functional
        values["spin_polarized"] = run_is_spin_polarized(
            assignments,
            parsed_cell.atomic_spins if parsed_cell is not None else None,
        )
    elif charge_state_for_job(job) == MUONIUM_CHARGE_STATE:
        # The value also seeds the always-shown Spin Polarized checkbox, which
        # would otherwise save an unticked box over the workflow's own default.
        values["spin_polarized"] = True
    return values


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


def energy_barrier_form_state(document: ProjectDocument | None) -> dict[str, object]:
    """Return a copy of the Energy Barriers form state, or ``{}``."""

    if document is None:
        return {}
    raw = document.ui_state.get(ENERGY_BARRIERS_FORM_STATE_KEY, {})
    return dict(raw) if isinstance(raw, dict) else {}


def energy_barrier_max_path_length_text(document: ProjectDocument | None) -> str:
    """Return the stored maximum path length as the form shows it, or ``""``."""

    raw = energy_barrier_form_state(document).get("max_path_length_ang")
    if raw in (None, ""):
        return ""
    try:
        return f"{float(raw):.6g}"
    except (TypeError, ValueError):
        return ""


def selected_energy_barrier_endpoint_keys(
    document: ProjectDocument | None,
    available_keys: Sequence[str] | None = None,
) -> tuple[str, str]:
    """Return the ``(initial, final)`` endpoint keys the form selects.

    ``available_keys`` are the endpoints the pickers offer, in order (default:
    every eligible source). A missing selection, or one not among them, falls
    back to the first available key for the initial endpoint and the first other
    available key for the final one.
    """

    if available_keys is None:
        available_keys = [
            candidate.key for candidate in eligible_quantum_source_candidates(document)
        ]
    available = list(available_keys)
    state = energy_barrier_form_state(document)
    reactant_key = str(state.get("selected_reactant_key", "")).strip()
    product_key = str(state.get("selected_product_key", "")).strip()
    if reactant_key not in available:
        reactant_key = available[0] if available else ""
    if product_key not in available:
        product_key = next((key for key in available if key != reactant_key), "")
    return reactant_key, product_key


def energy_barrier_source_label(candidate: EnergyBarrierSourceCandidate) -> str:
    """Return a generated path's label, ``"{initial}_to_{final}_path{n}"``."""

    reactant = quantum_source_base_label(candidate.reactant, "reactant")
    product = quantum_source_base_label(candidate.product, "product")
    return f"{reactant}_to_{product}_path{candidate.path_index}"


def serialize_energy_barrier_source_candidates(
    candidates: Iterable[EnergyBarrierSourceCandidate],
) -> list[dict[str, object]]:
    """Return the stored form of generated paths, as the form state holds them."""

    return [
        {
            "key": candidate.key,
            "label": energy_barrier_source_label(candidate),
            "path_index": candidate.path_index,
            "reactant_site_id": str(candidate.reactant.site.id),
            "product_site_id": str(candidate.product.site.id),
            "reactant_run_id": str(candidate.reactant.run.id),
            "product_run_id": str(candidate.product.run.id),
            "reactant_job_id": str(candidate.reactant.job.id),
            "product_job_id": str(candidate.product.job.id),
            "reactant_structure_id": str(candidate.reactant.structure.id),
            "product_structure_id": str(candidate.product.structure.id),
            "path_start_fractional": list(candidate.path.start_fractional),
            "path_end_fractional": list(candidate.path.end_fractional),
            "path_length_ang": candidate.path.length_ang,
            "charge_state": charge_state_for_job(candidate.reactant.job),
        }
        for candidate in candidates
    ]


def energy_barrier_path_preview_structure(
    document: ProjectDocument | None,
    candidate: EnergyBarrierSourceCandidate,
) -> Structure:
    """Return the pristine supercell a generated path is drawn in.

    The host is expanded to the initial endpoint's relaxed cell. Raises
    ``ValueError`` with a user-facing reason when there is no document or host,
    a lattice is malformed, or the endpoint cell is not an integer supercell of
    the host.
    """

    if document is None:
        raise ValueError("No project document is loaded.")
    host_id = document.project.host_structure_id
    host = document.structures.get(host_id) if host_id is not None else None
    if host is None:
        raise ValueError("A pristine host structure is required for the path preview.")

    pristine_lattice = np.asarray(host.lattice_vectors, dtype=float)
    target_lattice = np.asarray(candidate.reactant.structure.lattice_vectors, dtype=float)
    if pristine_lattice.shape != (3, 3) or target_lattice.shape != (3, 3):
        raise ValueError("Energy barrier path preview requires 3x3 lattice vectors.")

    matrix = np.rint(target_lattice @ np.linalg.inv(pristine_lattice)).astype(int)
    if not np.allclose(matrix @ pristine_lattice, target_lattice, atol=1e-6):
        raise ValueError(
            "Could not construct a pristine supercell matching the relaxed endpoint cell."
        )

    expanded_structure, _spins = build_supercell_structure(host, matrix.tolist())
    return replace(
        expanded_structure,
        label=f"{energy_barrier_source_label(candidate)} preview",
        origin="energy_barriers_path_preview",
        parent_structure_id=host.id,
    )


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
    if (
        reactant.structure.parent_structure_id != host.id
        or product.structure.parent_structure_id != host.id
    ):
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
            key=_path_key(reactant.key, product.key, path_index),
            reactant=reactant,
            product=product,
            path=path,
            path_index=path_index,
        )
        for path_index, path in enumerate(paths, start=1)
    ]


def _path_key(reactant_key: str, product_key: str, path_index: int) -> str:
    return f"{reactant_key}:{product_key}:{path_index}"


def keep_stored_path_keys(
    document: ProjectDocument | None,
    candidates: Sequence[EnergyBarrierSourceCandidate],
    stored_sources: object,
    *,
    durable_keys: Collection[str] = (),
) -> list[EnergyBarrierSourceCandidate]:
    """Give regenerated paths the keys of the stored paths they repeat.

    A key binds a path to its NEB job and settings, so a path generated again
    must keep its key, and a new path must not take the key of a stored path
    that keeps its row. A candidate symmetry-equivalent to a stored path of the
    same pair, charge state and endpoint structures
    (:func:`~mufinder.science.energy_barriers.equivalent_path_indexes`) takes
    that path's index and key, preferring a stored path whose key is in
    ``durable_keys`` (one with a job or saved settings, which stays although no
    longer generated). Every other candidate takes the next index above the
    matched and durable paths whose key no kept path holds. With nothing stored
    the candidates are returned unchanged.
    """

    if not candidates:
        return []
    first = candidates[0]
    pair = [
        source
        for source in (stored_sources if isinstance(stored_sources, list) else ())
        if isinstance(source, dict)
        and is_pair_source(
            source,
            reactant_site_id=str(first.reactant.site.id),
            product_site_id=str(first.product.site.id),
            charge_state=charge_state_for_job(first.reactant.job),
        )
    ]
    if not pair:
        return list(candidates)

    def stored_key(source: Mapping[str, object]) -> str:
        return str(source.get("key", "")).strip()

    # Only paths stored for these endpoint structures share their cell, so only
    # their positions can be compared with the candidates'.
    same_cell = [
        source
        for source in pair
        if _stored_path_index(source)
        and stored_key(source)
        and str(source.get("reactant_structure_id", "")) == str(first.reactant.structure.id)
        and str(source.get("product_structure_id", "")) == str(first.product.structure.id)
    ]
    hops: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    readable: list[Mapping[str, object]] = []
    for source in same_cell:
        try:
            hops.append(
                (
                    fractional_triplet_from_source(source, "path_start_fractional"),
                    fractional_triplet_from_source(source, "path_end_fractional"),
                )
            )
        except (TypeError, ValueError):
            continue
        readable.append(source)
    host_id = document.project.host_structure_id if document is not None else None
    host = document.structures.get(host_id) if document is not None and host_id else None
    indexes: list[int | None] = [None] * len(hops)
    if host is not None and hops:
        try:
            indexes = equivalent_path_indexes(
                pristine_structure=host,
                reactant_structure=first.reactant.structure,
                product_structure=first.product.structure,
                paths=[candidate.path for candidate in candidates],
                hops=hops,
            )
        except ValueError:
            pass
    matched: dict[int, Mapping[str, object]] = {}
    for source, index in zip(readable, indexes, strict=True):
        if index is None:
            continue
        current = matched.get(index)
        if current is None or (
            stored_key(source) in durable_keys and stored_key(current) not in durable_keys
        ):
            matched[index] = source

    kept_sources = [
        source
        for source in pair
        if any(source is match for match in matched.values()) or stored_key(source) in durable_keys
    ]
    taken_keys = {stored_key(source) for source in kept_sources}
    next_index = max((_stored_path_index(source) for source in kept_sources), default=0) + 1
    kept: list[EnergyBarrierSourceCandidate] = []
    for position, candidate in enumerate(candidates):
        match = matched.get(position)
        if match is not None:
            kept.append(
                replace(candidate, key=stored_key(match), path_index=_stored_path_index(match))
            )
            continue
        key = _path_key(candidate.reactant.key, candidate.product.key, next_index)
        while key in taken_keys:
            next_index += 1
            key = _path_key(candidate.reactant.key, candidate.product.key, next_index)
        kept.append(replace(candidate, key=key, path_index=next_index))
        taken_keys.add(key)
        next_index += 1
    return kept


def _stored_path_index(source: Mapping[str, object]) -> int:
    value = source.get("path_index")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def energy_barrier_path_symmetry_uses_magnetic_order(
    document: ProjectDocument | None, reactant_key: str
) -> bool | None:
    """Return whether paths from ``reactant_key`` use the host's magnetic order symmetry.

    ``True`` when the host's collinear spin pattern removes crystal operations
    in the reactant's supercell (see
    :func:`mufinder.science.energy_barriers.path_symmetry_uses_magnetic_order`),
    ``False`` when paths use the crystal symmetry, and ``None`` when there is no
    host or no eligible reactant, or the reactant does not descend from the host.
    """

    if document is None:
        return None
    host_id = document.project.host_structure_id
    host = document.structures.get(host_id) if host_id is not None else None
    reactant = quantum_source_candidate_by_key(document, reactant_key) if reactant_key else None
    if host is None or reactant is None or reactant.structure.parent_structure_id != host.id:
        return None
    try:
        return path_symmetry_uses_magnetic_order(
            pristine_structure=host, reactant_structure=reactant.structure
        )
    except ValueError:  # EnergyBarrierError, or a malformed host or endpoint cell
        return None


_ENDPOINT_FIELDS = ("site", "run", "job", "structure")


def _stored_endpoint(
    document: ProjectDocument,
    site: MuonSite,
    run: CalculationRun,
    job: Job,
    structure: Structure,
) -> QuantumSourceCandidate:
    key = quantum_source_key(document, site, job)
    return QuantumSourceCandidate(
        key=key,
        site=site,
        run=run,
        job=job,
        structure=structure,
        total_energy_eV=site.total_energy_eV
        if key == str(site.id)
        else total_energy_for_job(document, job.id),
    )


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
            (side, field): stores[field].get(record_id) for (side, field), record_id in ids.items()
        }
        if any(value is None for value in records.values()):
            continue
        try:
            path_index = int(raw_source.get("path_index", 1))
            path = MuonPath(
                start_fractional=fractional_triplet_from_source(
                    raw_source, "path_start_fractional"
                ),
                end_fractional=fractional_triplet_from_source(raw_source, "path_end_fractional"),
                length_ang=float(raw_source.get("path_length_ang", 0.0)),
            )
        except (TypeError, ValueError):
            continue
        endpoints = {
            side: _stored_endpoint(
                document,
                records[(side, "site")],
                records[(side, "run")],
                records[(side, "job")],
                records[(side, "structure")],
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


def energy_barrier_neb_source_candidates(
    document: ProjectDocument | None,
) -> list[EnergyBarrierNebCandidate]:
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


def serialize_energy_barrier_neb_sources(
    document: ProjectDocument | None,
    keys: Iterable[str],
) -> list[dict[str, object]]:
    """Return copies of the generated-path sources ``keys`` name, in that order."""

    candidates_by_key = {
        candidate.key: candidate for candidate in energy_barrier_neb_source_candidates(document)
    }
    return [dict(candidates_by_key[key].source) for key in keys if key in candidates_by_key]


def energy_barrier_neb_sources_by_job_id(
    document: ProjectDocument | None,
    definitions: Iterable[CalculationDefinition],
) -> dict[UUID, dict[str, object]]:
    """Map each NEB job ``definitions`` bind back to its generated-path source."""

    sources_by_key = {
        str(source.get("key", "")).strip(): source
        for source in energy_barrier_generated_sources(document)
    }
    by_job_id: dict[UUID, dict[str, object]] = {}
    for definition in definitions:
        mapping = definition.parameters.get("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            continue
        for key, raw_job_id in mapping.items():
            source = sources_by_key.get(str(key).strip())
            if source is None or not isinstance(raw_job_id, str):
                continue
            try:
                by_job_id[UUID(raw_job_id)] = source
            except ValueError:
                continue
    return by_job_id


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
        run = (
            document.runs.get(result.calculation_run_id)
            if result.calculation_run_id is not None
            else None
        )
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


def energy_barrier_neb_result_path_label(
    document: ProjectDocument | None,
    candidate: EnergyBarrierCompletedNebCandidate,
    source: Mapping[str, object] | None,
) -> str:
    """Describe a completed NEB result as ``initial → final``, else by its job label."""

    if source is not None and document is not None:
        labels: list[str] = []
        for field in ("reactant_site_id", "product_site_id"):
            site_id = uuid_from_source(source, field)
            site = document.sites.get(site_id) if site_id is not None else None
            labels.append((site.label or "").strip() if site is not None else "")
        if all(labels):
            return f"{labels[0]} → {labels[1]}"
    return candidate.job.label


# --- completed relaxations ----------------------------------------------------


def coerce_fractional_triplet(value: object) -> tuple[float, float, float] | None:
    """Return ``value`` as three floats, or ``None`` when it is not a numeric triplet."""

    return float_triplet_or_none(value)


#: Row settings that change a relaxation's total energy, so relaxations that
#: differ in any of them are different calculations.
RELAXATION_ENERGY_PARAMETERS = (
    "cut_off_energy",
    "basis_precision",
    "kpoint_mp_grid",
    "kpoint_mp_spacing",
    "xc_functional",
    "spin_polarized",
)


def _muon_relaxation_rows_by_job(document: ProjectDocument) -> dict[UUID, Mapping[str, object]]:
    """Map each bound muon-relaxation job to the row it ran, from the newest definition."""

    rows_by_job: dict[UUID, Mapping[str, object]] = {}
    for definition in definitions_for_stage(
        document, WorkflowStage.MUON_RELAXATION, calc_type=MUON_RELAXATION_DEFINITION_CALC_TYPE
    ):
        configuration = definition.parameters.get("configuration")
        rows = configuration.get("rows") if isinstance(configuration, Mapping) else None
        for entry in rows if isinstance(rows, list) else []:
            if not isinstance(entry, Mapping):
                continue
            label = str(entry.get("label", "")).strip()
            charge_state = (
                canonical_charge_state(str(entry.get("charge_state") or ""))
                or POSITIVE_MUON_CHARGE_STATE
            )
            job = muon_relaxation_job_for_row(document, definition, label, charge_state)
            if label and job is not None:
                rows_by_job.setdefault(job.id, entry)
    return rows_by_job


def _stored_supercell_matrix(entry: Mapping[str, object]) -> list[list[float]] | None:
    raw = entry.get("supercell_matrix")
    if not isinstance(raw, list) or len(raw) != 3:
        return None
    try:
        return [[float(value) for value in row] for row in raw]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None


def host_muon_position_from_relaxed_structure(
    structure: Structure, entry: Mapping[str, object] | None
) -> tuple[float, float, float] | None:
    """Return the relaxed muon's host-cell position, read from its relaxed supercell.

    The relaxed structure is the supercell the row ran in, so its ``H:mu``
    fractional coordinate is mapped to the host cell with the supercell matrix
    the row stores (no matrix: the row ran in the host cell). ``None`` when
    the row is unknown or the cell holds no muon: the position is unavailable.
    """

    if entry is None:
        return None
    supercell_position = muon_fractional_position_from_cell(structure)
    if supercell_position is None:
        return None
    matrix = _stored_supercell_matrix(entry)
    host = (
        supercell_position
        if matrix is None
        else lattice_frac_to_cart(supercell_position, matrix)  # row vector times the matrix
    )
    folded = wrap_fractional(host)
    return (float(folded[0]), float(folded[1]), float(folded[2]))


def _comparable_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_comparable_value(item) for item in value)
    if isinstance(value, str):
        return value.strip().casefold()
    if isinstance(value, bool) or value is None:
        return value
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def relaxation_calculation_key(
    entry: Mapping[str, object] | None, structure: Structure
) -> tuple[object, ...]:
    """Return what must match for two relaxations' total energies to be comparable.

    ``(composition, supercell, settings)``: the relaxed cell's composition,
    which separates supercells of different sizes, then the supercell matrix
    and the energy-defining settings (:data:`RELAXATION_ENERGY_PARAMETERS`)
    the row stores, both ``None`` when the row is unknown. A row stores only
    the settings it overrides, so a setting given explicitly and the same value
    inherited from the project defaults compare as different.
    """

    composition = tuple(sorted(Counter(structure.species).items()))
    if entry is None:
        return (composition, None, None)
    values = entry.get("parameter_values")
    parameters = values if isinstance(values, Mapping) else {}
    return (
        composition,
        ("supercell", _comparable_value(_stored_supercell_matrix(entry))),
        tuple(
            (name, _comparable_value(parameters.get(name))) for name in RELAXATION_ENERGY_PARAMETERS
        ),
    )


def mixes_calculations(sources: Iterable[CompletedRelaxationAnalysisSource]) -> bool:
    """Return whether ``sources`` come from more than one comparable calculation.

    Cells are compared for every source; the supercell and settings only
    between sources whose rows are known (:func:`relaxation_calculation_key`).
    """

    keys = [source.calculation for source in sources if len(source.calculation) == 3]
    return any(len({key[part] for key in keys if key[part] is not None}) > 1 for part in range(3))


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
    a result with a derived structure, and a folded host muon position is known:
    from the result payload, else from the relaxed supercell mapped to the host
    cell with the row's stored supercell matrix
    (:func:`host_muon_position_from_relaxed_structure`). Each source carries
    its calculation key (:func:`relaxation_calculation_key`). Sources are
    sorted by total energy (unknown last), then case-insensitive label.
    """

    if document is None:
        return []
    sources: list[CompletedRelaxationAnalysisSource] = []
    seen: set[UUID] = set()
    rows_by_job = _muon_relaxation_rows_by_job(document)
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
        entry = rows_by_job.get(job.id)
        fractional = None
        if isinstance(result.data, dict):
            fractional = coerce_fractional_triplet(
                result.data.get("muon_final_fractional_position_host")
            )
        if fractional is None:
            fractional = host_muon_position_from_relaxed_structure(structure, entry)
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
                calculation=relaxation_calculation_key(entry, structure),
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


def newest_completed_structure_id(
    document: ProjectDocument | None, jobs: Iterable[CompletedPristineJob]
) -> UUID | None:
    """Return the relaxed structure of the most recently completed job in ``jobs``."""

    if document is None:
        return None
    newest: tuple[object, UUID] | None = None
    for job in jobs:
        record = document.jobs.get(job.job_id)
        completed_at = record.completed_at if record is not None else None
        if completed_at is None:
            continue
        if newest is None or completed_at > newest[0]:  # type: ignore[operator]
            newest = (completed_at, job.structure_id)
    return newest[1] if newest is not None else None


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
    "RELAXATION_ENERGY_PARAMETERS",
    "PristineRelaxationRowLike",
    "QuantumSourceCandidate",
    "RelaxationRowStateLike",
    "SITE_KEY_CHARGE_STATE_PROPERTY",
    "charge_state_for_job",
    "charge_state_label_for_job",
    "coerce_fractional_triplet",
    "completed_energy_barrier_neb_candidates",
    "completed_muon_relaxation_sources",
    "completed_pristine_relaxation_jobs",
    "eligible_energy_barrier_source_candidates",
    "eligible_quantum_source_candidates",
    "energy_barrier_form_state",
    "energy_barrier_generated_sources",
    "energy_barrier_max_path_length_text",
    "energy_barrier_neb_result_path_label",
    "energy_barrier_neb_source_candidates",
    "energy_barrier_neb_sources_by_job_id",
    "energy_barrier_path_preview_structure",
    "energy_barrier_path_symmetry_uses_magnetic_order",
    "energy_barrier_source_label",
    "enumerate_energy_barrier_source_candidates",
    "float_from_source",
    "host_muon_position_from_relaxed_structure",
    "fractional_triplet_from_source",
    "keep_stored_path_keys",
    "mixes_calculations",
    "newest_completed_structure_id",
    "quantum_seed_name_for_key",
    "quantum_source_base_label",
    "quantum_source_candidate_by_key",
    "quantum_source_label",
    "quantum_source_labels_by_key",
    "quantum_source_key",
    "quantum_source_relative_energies",
    "relaxation_calculation_key",
    "relaxation_sources_for_labels",
    "selected_energy_barrier_endpoint_keys",
    "serialize_energy_barrier_neb_sources",
    "serialize_energy_barrier_source_candidates",
    "serialize_quantum_sources",
    "site_charge_state_relaxation_jobs",
    "site_key_charge_state",
    "site_linked_charge_state",
    "source_relaxation_parameter_values",
    "uuid_from_source",
]
