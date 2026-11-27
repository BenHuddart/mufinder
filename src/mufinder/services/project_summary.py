"""Where a project stands: the workflow's stages, their progress and what they found.

This module is the one home for the workflow-stage catalogue the project
navigator and the project home are built from (``WORKFLOW_STAGES``,
``STAGE_GROUPS``, ``STAGE_PHASES``), the completion rule of each stage
(``*_complete``), the summary the project home shows (``summarize_project``):
the host line, the stage of the latest activity, the muon-site progress and the
later workflows' results, and the status of every stage and group built from
that summary (``stage_statuses``). It is read-only and UI-free.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from functools import lru_cache
import math
from uuid import UUID

from mufinder.domain.charge_states import (
    POSITIVE_MUON_CHARGE_STATE,
    canonical_charge_state,
    is_muon_relaxation_calc_type,
    muon_charge_state_label,
    muon_charge_state_prefix,
    muon_relaxation_calc_type,
)
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Result, Structure
from mufinder.domain.text import plural, space_group_text
from mufinder.persistence.codec import uuid_or_none
from mufinder.science.ase_bridge import is_muon_label, reduced_formula
from mufinder.science.spglib_compat import host_symmetry
from mufinder.services import supercell_assets
from mufinder.services.job_monitoring import job_status_word
from mufinder.services.magnetic_field_results import (
    MUON_POLARISATION_RESULT_TYPE,
    SIMULATION_OWNER_PREFIX,
)
from mufinder.services.uep_results import uep_batch_searches
from mufinder.services.project_document import (
    ProjectDocument,
    active_definition,
    active_host_structure,
    job_for_mapping_key,
)
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
)
from mufinder.workflow.muon_site import (
    ACTIVE_MUON_RELAXATION_DEFINITION_KEY,
    MUON_RELAXATION_DEFINITION_CALC_TYPE,
)
from mufinder.workflow.hyperfine import HYPERFINE_CALC_TYPE
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE

#: ``ui_state`` key holding the relaxation-analysis workspaces.
RELAXATION_ANALYSIS_WORKSPACES_KEY = "relaxation_analysis_workspaces"
#: ``ui_state`` key holding the magnetic-field workspaces.
MAGNETIC_FIELD_WORKSPACES_KEY = "magnetic_field_workspaces"
#: ``ui_state`` key holding the pinned Muon Sites summary, one snapshot per charge state.
MUON_SITES_SUMMARY_SNAPSHOTS_KEY = "muon_sites_summary_snapshots"


class StageStatus(StrEnum):
    """How far a workflow stage has got."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class StageGroup:
    """A navigator section that holds several stages, such as Convergence Testing."""

    key: str
    title: str
    completion_goal: str


@dataclass(frozen=True, slots=True)
class StagePhase:
    """A phase of the workflow over consecutive top-level entries, such as Find sites.

    ``members`` are top-level entries in workflow order: a stage key, or a
    group key standing for the stages of that group.
    """

    key: str
    title: str
    members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkflowStageInfo:
    """One workflow stage: its navigator route, title and completion rule.

    ``key`` is also the stage's navigator route. An ``optional`` stage is a side
    route the main workflow does not pass through.
    """

    key: str
    title: str
    completion_goal: str
    group: str | None = None
    optional: bool = False


@dataclass(frozen=True, slots=True)
class StageSummary:
    """Where one stage stands: its progress and a few facts about the project."""

    key: str
    title: str
    status: StageStatus
    facts: tuple[str, ...]

    @property
    def fact_line(self) -> str:
        """The stage's facts as one line: ``"Default cutoff 600 eV; 6 tests complete"``."""

        return "; ".join(self.facts)


@dataclass(frozen=True, slots=True)
class LastStage:
    """The stage the user left off at, and when.

    ``timestamp`` is the stage's latest activity, or ``None`` when the project
    has none and the stage is the first one still to do.
    """

    key: str
    title: str
    timestamp: datetime | None
    fact_line: str


@dataclass(frozen=True, slots=True)
class ProjectSummary:
    """What the project home shows: every stage's summary and the lines built from them.

    ``host_line`` names the host and the adopted convergence defaults;
    ``last_stage`` is where the user left off; ``site_progress`` says how far
    the muon-site search has got, for a project with no pinned sites;
    ``other_results`` joins the facts of the later workflows that have started,
    and is empty when none has.
    """

    stages: tuple[StageSummary, ...]
    host_line: str
    last_stage: LastStage
    site_progress: str
    other_results: str

    def stage(self, key: str) -> StageSummary | None:
        return next((stage for stage in self.stages if stage.key == key), None)


STAGE_GROUPS: tuple[StageGroup, ...] = (
    StageGroup(
        "convergence",
        "Convergence Testing",
        "Set both project default convergence values (cutoff and k-point).",
    ),
    StageGroup(
        "muon_sites",
        "Muon Sites",
        "Generate muon sites, complete all their relaxations and create an analysis workspace.",
    ),
)

WORKFLOW_STAGES: tuple[WorkflowStageInfo, ...] = (
    WorkflowStageInfo(
        "host_structure",
        "Host Structure",
        "Load a host structure.",
    ),
    WorkflowStageInfo(
        "convergence_cutoff",
        "Cutoff Energy",
        "Set a project default cutoff value.",
        group="convergence",
    ),
    WorkflowStageInfo(
        "convergence_kpoint",
        "K-Point Grid",
        "Set a project default k-point grid.",
        group="convergence",
    ),
    WorkflowStageInfo(
        "pristine_relaxation",
        "Pristine Relaxation",
        "Generate at least one comparison table.",
        optional=True,
    ),
    WorkflowStageInfo(
        "uep",
        "UEP",
        "Complete at least one UEP source job.",
        optional=True,
    ),
    WorkflowStageInfo(
        "muon_candidates",
        "Generation",
        "Generate at least one muon site.",
        group="muon_sites",
    ),
    WorkflowStageInfo(
        "muon_relaxations",
        "Relaxations",
        "All structural relaxations completed.",
        group="muon_sites",
    ),
    WorkflowStageInfo(
        "muon_analysis",
        "Analysis",
        "Create at least one structural-relaxation analysis workspace.",
        group="muon_sites",
    ),
    WorkflowStageInfo(
        "hyperfine_coupling",
        "Hyperfine Coupling",
        "Complete at least one hyperfine calculation.",
    ),
    WorkflowStageInfo(
        "magnetic_fields",
        "Magnetic Fields",
        "Define at least one magnetic component.",
    ),
    WorkflowStageInfo(
        "polarisation",
        "Polarisation",
        "Run at least one polarisation simulation.",
    ),
    WorkflowStageInfo(
        "quantum_corrections",
        "Quantum Corrections",
        "Complete at least one quantum phonon analysis.",
    ),
    WorkflowStageInfo(
        "energy_barriers",
        "Energy Barriers",
        "Complete at least one NEB barrier calculation.",
    ),
)

#: The workflow's phases, in workflow order.
STAGE_PHASES: tuple[StagePhase, ...] = (
    StagePhase("prepare", "Prepare", ("host_structure", "convergence", "pristine_relaxation")),
    StagePhase("find_sites", "Find sites", ("uep", "muon_sites")),
    StagePhase(
        "interpret",
        "Interpret",
        (
            "hyperfine_coupling",
            "magnetic_fields",
            "polarisation",
            "quantum_corrections",
            "energy_barriers",
        ),
    ),
)

_STAGE_JOB_STAGES: dict[str, tuple[WorkflowStage, ...]] = {
    "convergence_cutoff": (WorkflowStage.CONVERGENCE_CUTOFF,),
    "convergence_kpoint": (WorkflowStage.CONVERGENCE_KPOINT,),
    "pristine_relaxation": (WorkflowStage.PRISTINE_RELAXATION,),
    "uep": (WorkflowStage.UEP,),
    "muon_relaxations": (WorkflowStage.MUON_RELAXATION,),
    "hyperfine_coupling": (WorkflowStage.HYPERFINE_COUPLING,),
    "magnetic_fields": (WorkflowStage.MAGNETIC_FIELD,),
    "quantum_corrections": (WorkflowStage.QUANTUM_CORRECTIONS,),
    "energy_barriers": (WorkflowStage.ENERGY_BARRIERS,),
}

#: The navigator stage whose page holds a job of each workflow stage.
_JOB_STAGE_KEYS: dict[WorkflowStage, str] = {
    **{job_stage: key for key, job_stages in _STAGE_JOB_STAGES.items() for job_stage in job_stages},
    WorkflowStage.MUON_GENERATION: "muon_candidates",
    WorkflowStage.SITE_ANALYSIS: "muon_analysis",
}

_STAGES_BY_KEY = {stage.key: stage for stage in WORKFLOW_STAGES}
_GROUPS_BY_KEY = {group.key: group for group in STAGE_GROUPS}
_PHASES_BY_KEY = {phase.key: phase for phase in STAGE_PHASES}


def stage_info(key: str) -> WorkflowStageInfo:
    """Return the stage with ``key``; raises ``KeyError`` for an unknown key."""

    return _STAGES_BY_KEY[key]


def job_stage_key(workflow_stage: WorkflowStage) -> str:
    """Return the navigator stage key (a page route) of a job's ``workflow_stage``."""

    return _JOB_STAGE_KEYS[workflow_stage]


def job_stage_name(workflow_stage: WorkflowStage) -> str:
    """Return the navigator's name for a job's ``workflow_stage``."""

    return stage_info(job_stage_key(workflow_stage)).title


def stage_group(key: str) -> StageGroup:
    """Return the stage group with ``key``; raises ``KeyError`` for an unknown key."""

    return _GROUPS_BY_KEY[key]


def group_stage_keys(group_key: str) -> tuple[str, ...]:
    """Return the keys of the stages in a group, in workflow order."""

    return tuple(stage.key for stage in WORKFLOW_STAGES if stage.group == group_key)


def stage_phase(key: str) -> StagePhase:
    """Return the phase with ``key``; raises ``KeyError`` for an unknown key."""

    return _PHASES_BY_KEY[key]


# A page whose own title reads as a job name ("Cutoff Energy jobs") lends it;
# "Relaxations" names the page only under its Muon Sites group.
_JOB_STAGE_TITLE_OVERRIDES: dict[WorkflowStage, str] = {
    WorkflowStage.MUON_RELAXATION: "Muon Relaxation",
}


def job_stage_title(stages: Iterable[WorkflowStage]) -> str | None:
    """Return the name of the page that made a batch of jobs, for "Run 6 Cutoff Energy jobs".

    Returns ``None`` when the batch is empty, mixes stages, or comes from a
    stage no page submits jobs for, so the caller says plain "jobs".
    """

    distinct = set(stages)
    if len(distinct) != 1:
        return None
    (stage,) = distinct
    if stage in _JOB_STAGE_TITLE_OVERRIDES:
        return _JOB_STAGE_TITLE_OVERRIDES[stage]
    for key, job_stages in _STAGE_JOB_STAGES.items():
        if stage in job_stages:
            return stage_info(key).title
    return None


# --- Completion rules ---------------------------------------------------------


def host_complete(document: ProjectDocument) -> bool:
    return document.project.host_structure_id is not None


def cutoff_complete(document: ProjectDocument) -> bool:
    return document.project.recommended_cutoff_energy is not None


def kpoint_complete(document: ProjectDocument) -> bool:
    return document.project.recommended_kpoint_grid is not None


def pristine_relaxation_complete(document: ProjectDocument) -> bool:
    return bool(document.comparison_tables)


def uep_complete(document: ProjectDocument) -> bool:
    return any(
        job.workflow_stage == WorkflowStage.UEP and job.status == JobStatus.COMPLETED
        for job in document.jobs.values()
    )


def muon_candidates_complete(document: ProjectDocument) -> bool:
    return bool(document.sites)


def muon_relaxations_complete(document: ProjectDocument) -> bool:
    """Return whether every row of the active muon-relaxation definition relaxed and converged."""

    definition = active_muon_relaxation_definition(document)
    if definition is None:
        return False
    saw_row = False
    for _label, _charge_state, _site_id, job in muon_relaxation_row_jobs(document, definition):
        saw_row = True
        if job is None or job.status != JobStatus.COMPLETED:
            return False
        latest_run = latest_run_for_job(document, job.id)
        if latest_run is not None and latest_run.convergence_achieved is False:
            return False
    return saw_row


def relaxation_analysis_complete(document: ProjectDocument) -> bool:
    return bool(relaxation_analysis_workspaces(document))


def magnetic_fields_complete(document: ProjectDocument) -> bool:
    return magnetic_structure_has_components(document.project.magnetic_structure)


def magnetic_structure_has_components(magnetic_structure: object) -> bool:
    if not isinstance(magnetic_structure, Mapping):
        return False
    components = magnetic_structure.get("components")
    if isinstance(components, list):
        return any(isinstance(component, dict) for component in components)
    moments = magnetic_structure.get("moments")
    return isinstance(moments, list) and bool(moments)


def polarisation_complete(document: ProjectDocument) -> bool:
    return any(_is_polarisation_simulation(result) for result in document.results.values())


def _is_polarisation_simulation(result: Result) -> bool:
    if result.result_type != MUON_POLARISATION_RESULT_TYPE or not isinstance(result.data, Mapping):
        return False
    return str(result.data.get("owner", "")).startswith(SIMULATION_OWNER_PREFIX)


def hyperfine_coupling_complete(document: ProjectDocument) -> bool:
    return any(result.result_type == HYPERFINE_CALC_TYPE for result in document.results.values())


def quantum_corrections_complete(document: ProjectDocument) -> bool:
    return any(
        result.result_type == QUANTUM_PHONON_CALC_TYPE for result in document.results.values()
    )


#: Result types that hold a barrier: NEB results, plus the search results an older
#: project may still store (see ``docs/compatibility_policy.rst``).
_BARRIER_RESULT_TYPES = frozenset(
    {NUDGED_ELASTIC_BAND_CALC_TYPE, TRANSITION_STATE_SEARCH_CALC_TYPE}
)


def energy_barriers_complete(document: ProjectDocument) -> bool:
    return any(result.result_type in _BARRIER_RESULT_TYPES for result in document.results.values())


_STAGE_COMPLETION_CHECKS: dict[str, Callable[[ProjectDocument], bool]] = {
    "host_structure": host_complete,
    "convergence_cutoff": cutoff_complete,
    "convergence_kpoint": kpoint_complete,
    "pristine_relaxation": pristine_relaxation_complete,
    "uep": uep_complete,
    "muon_candidates": muon_candidates_complete,
    "muon_relaxations": muon_relaxations_complete,
    "muon_analysis": relaxation_analysis_complete,
    "hyperfine_coupling": hyperfine_coupling_complete,
    "magnetic_fields": magnetic_fields_complete,
    "polarisation": polarisation_complete,
    "quantum_corrections": quantum_corrections_complete,
    "energy_barriers": energy_barriers_complete,
}


def _group_completion_check(group_key: str) -> Callable[[ProjectDocument], bool]:
    """A group is complete when every stage in it is complete."""

    checks = tuple(_STAGE_COMPLETION_CHECKS[key] for key in group_stage_keys(group_key))
    return lambda document: all(check(document) for check in checks)


#: The completion rule of every stage and stage group, by key. A group's rule is
#: that every stage in it is complete.
COMPLETION_CHECKS: dict[str, Callable[[ProjectDocument], bool]] = {
    **_STAGE_COMPLETION_CHECKS,
    **{group.key: _group_completion_check(group.key) for group in STAGE_GROUPS},
}


# --- Shared document lookups --------------------------------------------------


def active_muon_relaxation_definition(document: ProjectDocument) -> CalculationDefinition | None:
    return active_definition(
        document,
        WorkflowStage.MUON_RELAXATION,
        calc_type=MUON_RELAXATION_DEFINITION_CALC_TYPE,
        active_key=ACTIVE_MUON_RELAXATION_DEFINITION_KEY,
    )


def muon_relaxation_job_for_row(
    document: ProjectDocument,
    definition: CalculationDefinition | None,
    label: str,
    charge_state: str,
) -> Job | None:
    """Return the muon-relaxation job ``definition`` binds to one row.

    The charge-qualified ``"{prefix}:{label}"`` key is tried first and must hold
    a job of that charge state's calc type; the bare ``label`` key, kept by
    older definitions, accepts any muon-relaxation calc type.
    """

    keys = (
        (f"{muon_charge_state_prefix(charge_state)}:{label}", label) if charge_state else (label,)
    )
    calc_type = muon_relaxation_calc_type(charge_state) if charge_state else None
    for key in keys:
        job = job_for_mapping_key(
            document, definition, key, calc_type=calc_type if key != label else None
        )
        if job is not None and is_muon_relaxation_calc_type(job.calc_type):
            return job
    return None


def muon_relaxation_row_jobs(
    document: ProjectDocument,
    definition: CalculationDefinition,
) -> list[tuple[str, str, UUID | None, Job | None]]:
    """Return ``(label, charge_state, site_id, job)`` for each row of a muon-relaxation definition."""

    configuration = definition.parameters.get("configuration", {})
    rows = configuration.get("rows", []) if isinstance(configuration, dict) else []
    if not isinstance(rows, list):
        return []
    row_jobs: list[tuple[str, str, UUID | None, Job | None]] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", "")).strip()
        if not label:
            continue
        charge_state = canonical_charge_state(
            str(entry.get("charge_state") or POSITIVE_MUON_CHARGE_STATE)
        )
        charge_state = charge_state or POSITIVE_MUON_CHARGE_STATE
        site_id = uuid_or_none(entry.get("site_id"))
        row_jobs.append(
            (
                label,
                charge_state,
                site_id,
                muon_relaxation_job_for_row(document, definition, label, charge_state),
            )
        )
    return row_jobs


def latest_run_for_job(document: ProjectDocument, job_id: UUID) -> CalculationRun | None:
    runs = [run for run in document.runs.values() if run.job_id == job_id]
    return max(runs, key=lambda run: run.run_index) if runs else None


def relaxation_analysis_workspaces(document: ProjectDocument) -> list[dict[str, object]]:
    return _dict_list(document.ui_state.get(RELAXATION_ANALYSIS_WORKSPACES_KEY))


def magnetic_field_workspaces(document: ProjectDocument) -> list[dict[str, object]]:
    return _dict_list(document.ui_state.get(MAGNETIC_FIELD_WORKSPACES_KEY))


def structure_formula(structure: Structure) -> str:
    """Return the reduced chemical formula of a structure's non-muon atoms (``SrTiO3``)."""

    return reduced_formula(structure.species)[0]


# --- The project summary ------------------------------------------------------


def summarize_project(document: ProjectDocument) -> ProjectSummary:
    """Return every stage's status and facts, and the lines the project home shows."""

    complete = {key: check(document) for key, check in _STAGE_COMPLETION_CHECKS.items()}
    jobs_by_stage = Counter(job.workflow_stage for job in document.jobs.values())
    stages: list[StageSummary] = []
    for info in WORKFLOW_STAGES:
        job_stages = _STAGE_JOB_STAGES.get(info.key, ())
        started = any(jobs_by_stage[stage] for stage in job_stages) or _has_partial_work(
            document, info.key
        )
        if complete[info.key]:
            status = StageStatus.COMPLETE
        elif started:
            status = StageStatus.IN_PROGRESS
        else:
            status = StageStatus.NOT_STARTED
        stages.append(StageSummary(info.key, info.title, status, _FACTS[info.key](document)))
    return ProjectSummary(
        stages=tuple(stages),
        host_line=_host_line(document),
        last_stage=_last_stage(document, stages),
        site_progress=_site_progress(document),
        other_results=_other_results(stages),
    )


def group_status(summary: ProjectSummary, group_key: str) -> StageStatus:
    """Return how far a stage group has got.

    A group is complete when every stage in it is complete, in progress when
    any of its stages has started, and otherwise not started. Raises
    ``KeyError`` for an unknown key.
    """

    stage_group(group_key)
    members = group_stage_keys(group_key)
    statuses = [stage.status for stage in summary.stages if stage.key in members]
    if all(status == StageStatus.COMPLETE for status in statuses):
        return StageStatus.COMPLETE
    if any(status != StageStatus.NOT_STARTED for status in statuses):
        return StageStatus.IN_PROGRESS
    return StageStatus.NOT_STARTED


def stage_statuses(summary: ProjectSummary) -> dict[str, StageStatus]:
    """Return the status of every stage key and every group key in a project summary.

    Each stage's status is the summary's own, so a caller that also shows the
    summary gets the same status for a stage wherever it appears.
    """

    statuses = {stage.key: stage.status for stage in summary.stages}
    statuses.update({group.key: group_status(summary, group.key) for group in STAGE_GROUPS})
    return statuses


#: The later workflows whose facts the project home joins into one line, in its order.
OTHER_RESULT_STAGES = (
    "magnetic_fields",
    "polarisation",
    "hyperfine_coupling",
    "quantum_corrections",
    "energy_barriers",
)


#: What each later workflow's facts say while it has nothing to show.
_NOTHING_YET = {
    "magnetic_fields": "No magnetic model yet",
    "polarisation": "No polarisation simulations yet",
    "hyperfine_coupling": "No hyperfine calculations yet",
    "quantum_corrections": "No quantum calculations yet",
    "energy_barriers": "No barrier calculations yet",
}


def _other_results(stages: Iterable[StageSummary]) -> str:
    """The later workflows' facts, leaving out a workflow with nothing to show.

    A workflow whose rows are configured but not run has started and still has
    no result, so its status alone does not decide.
    """

    by_key = {stage.key: stage for stage in stages}
    return " · ".join(
        f"{by_key[key].title}: {by_key[key].fact_line}"
        for key in OTHER_RESULT_STAGES
        if by_key[key].status != StageStatus.NOT_STARTED
        and by_key[key].facts != (_NOTHING_YET[key],)
    )


def _last_stage(document: ProjectDocument, stages: list[StageSummary]) -> LastStage:
    """The stage of the newest job or Journal record, else the first stage still to do."""

    latest = _latest_activity(document)
    if latest is not None:
        key, timestamp = latest
    else:
        timestamp = None
        key = next(
            (
                stage.key
                for stage in stages
                if stage.status != StageStatus.COMPLETE and not stage_info(stage.key).optional
            ),
            stages[-1].key,
        )
    stage = next(stage for stage in stages if stage.key == key)
    return LastStage(stage.key, stage.title, timestamp, stage.fact_line)


#: Journal records whose page is not their workflow stage's: a polarisation
#: simulation is recorded under Magnetic Fields but made on the Polarisation page.
_RECORD_ACTION_STAGE_KEYS = {"polarisation_simulated": "polarisation"}


def _latest_activity(document: ProjectDocument) -> tuple[str, datetime] | None:
    """The stage key and time of the newest job event or workflow Journal record."""

    latest: tuple[str, datetime] | None = None

    def consider(key: str, stamp: datetime | None) -> None:
        nonlocal latest
        if stamp is None:
            return
        stamp = _aware(stamp)
        if latest is None or stamp > latest[1]:
            latest = (key, stamp)

    for job in document.jobs.values():
        key = job_stage_key(job.workflow_stage)
        for stamp in (job.created_at, job.submitted_at, job.started_at, job.completed_at):
            consider(key, stamp)
    for record in document.provenance:
        if record.workflow_stage is not None:
            key = _RECORD_ACTION_STAGE_KEYS.get(record.action, job_stage_key(record.workflow_stage))
            consider(key, record.timestamp)
    return latest


def _aware(stamp: datetime) -> datetime:
    """A naive time, from an older project, is UTC."""

    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)


def _host_line(document: ProjectDocument) -> str:
    """``"CoF₂ · P4₂/mnm · 6 atoms · cutoff 600 eV, k-point grid 4×4×6"``."""

    host = active_host_structure(document)
    if host is None:
        return "No host structure yet"
    parts = [structure_formula(host) or host.label]
    symbol = _space_group_symbol(
        tuple(tuple(float(value) for value in vector) for vector in host.lattice_vectors),
        tuple(host.species),
        tuple(tuple(float(value) for value in position) for position in host.fractional_positions),
    )
    if symbol:
        parts.append(space_group_text(symbol))
    parts.append(plural(sum(not is_muon_label(label) for label in host.species), "atom"))
    cutoff = document.project.recommended_cutoff_energy
    grid = document.project.recommended_kpoint_grid
    if cutoff is None and grid is None:
        parts.append("not converged yet")
    else:
        cutoff_text = f"cutoff {cutoff:g} eV" if cutoff is not None else "no default cutoff yet"
        grid_text = (
            f"k-point grid {'×'.join(str(value) for value in grid)}"
            if grid is not None
            else "no default k-point grid yet"
        )
        parts.append(f"{cutoff_text}, {grid_text}")
    return " · ".join(parts)


@lru_cache(maxsize=8)
def _space_group_symbol(
    lattice_vectors: tuple[tuple[float, ...], ...],
    species: tuple[str, ...],
    fractional_positions: tuple[tuple[float, ...], ...],
) -> str | None:
    """The host's space group, cached because the project home asks on every refresh."""

    return host_symmetry(
        lattice_vectors=lattice_vectors,
        species=species,
        fractional_positions=fractional_positions,
    ).symbol


def _site_progress(document: ProjectDocument) -> str:
    """``"7 candidate sites, 4 of 14 relaxed"``, for a project with nothing pinned."""

    host_id = document.project.host_structure_id
    candidates = sum(
        site.source_structure_id in (None, host_id) for site in document.sites.values()
    )
    if not candidates:
        return "No candidate muon sites yet."
    text = plural(candidates, "candidate site")
    definition = active_muon_relaxation_definition(document)
    # Relaxations configured on a previous host are not this host's progress.
    if definition is not None and host_id in definition.input_structure_ids:
        rows = muon_relaxation_row_jobs(document, definition)
        relaxed = sum(_relaxed(document, job) for _label, _charge, _site, job in rows)
        if rows:
            text += f", {relaxed} of {len(rows)} relaxed"
    return f"{text}. No sites pinned to the Muon Sites summary yet."


def _relaxed(document: ProjectDocument, job: Job | None) -> bool:
    """Whether a relaxation row's job finished without a non-converged run."""

    if job is None or job.status != JobStatus.COMPLETED:
        return False
    run = latest_run_for_job(document, job.id)
    return run is None or run.convergence_achieved is not False


def _has_partial_work(document: ProjectDocument, key: str) -> bool:
    if key == "muon_relaxations":
        return any(site.status in {"staged", "relaxed"} for site in document.sites.values())
    if key == "magnetic_fields":
        return bool(magnetic_field_workspaces(document))
    return False


def _host_facts(document: ProjectDocument) -> tuple[str, ...]:
    host = active_host_structure(document)
    if host is None:
        return ("No host structure yet",)
    formula = structure_formula(host) or host.label
    atoms = sum(not is_muon_label(label) for label in host.species)
    facts = [f"{formula} · {plural(atoms, 'atom')}"]
    if host.origin == "pristine_relaxation":
        facts.append("Active host is a relaxed structure")
    facts.append(_saved_supercells_fact(document))
    return tuple(facts)


def _saved_supercells_fact(document: ProjectDocument) -> str:
    assets = supercell_assets.list_workflow_supercell_assets(
        document,
        host_structure_id=document.project.host_structure_id,
    )
    if not assets:
        return "No saved supercells"
    labels = ", ".join(asset.label for asset in assets[:3])
    more = f" and {len(assets) - 3} more" if len(assets) > 3 else ""
    return f"{plural(len(assets), 'saved supercell')}: {labels}{more}"


def _cutoff_facts(document: ProjectDocument) -> tuple[str, ...]:
    cutoff = document.project.recommended_cutoff_energy
    facts = [f"Default cutoff {cutoff:g} eV" if cutoff is not None else "No default cutoff yet"]
    facts.extend(_job_facts(document, WorkflowStage.CONVERGENCE_CUTOFF, "test"))
    return tuple(facts)


def _kpoint_facts(document: ProjectDocument) -> tuple[str, ...]:
    grid = document.project.recommended_kpoint_grid
    facts = [
        f"Default grid {'×'.join(str(value) for value in grid)}"
        if grid is not None
        else "No default grid yet"
    ]
    facts.extend(_job_facts(document, WorkflowStage.CONVERGENCE_KPOINT, "test"))
    return tuple(facts)


def _pristine_facts(document: ProjectDocument) -> tuple[str, ...]:
    relaxed = [
        structure
        for structure in document.structures.values()
        if structure.origin == "pristine_relaxation"
    ]
    facts = list(_job_facts(document, WorkflowStage.PRISTINE_RELAXATION, "relaxation"))
    if relaxed:
        facts.append(plural(len(relaxed), "relaxed structure"))
    if document.comparison_tables:
        facts.append(plural(len(document.comparison_tables), "comparison table"))
    return tuple(facts) or ("Not visited",)


def _uep_facts(document: ProjectDocument) -> tuple[str, ...]:
    facts = list(_job_facts(document, WorkflowStage.UEP, "source job"))
    minima = 0
    for search in uep_batch_searches(document):
        clusters = None if search.grouping is None else search.grouping.get("clusters")
        if isinstance(clusters, list):
            minima += sum(isinstance(cluster, Mapping) for cluster in clusters)
    if minima:
        facts.append(plural(minima, "UEP site"))
    return tuple(facts) or ("Not visited",)


def _candidate_facts(document: ProjectDocument) -> tuple[str, ...]:
    host_id = document.project.host_structure_id
    sites = [
        site for site in document.sites.values() if site.source_structure_id in (None, host_id)
    ]
    if not sites:
        return ("No candidate sites yet",)
    from_uep = sum((site.generation_method or "").startswith("uep_cluster") for site in sites)
    facts = [plural(len(sites), "candidate site")]
    if from_uep:
        facts.append(f"{from_uep} from UEP clusters")
    other_hosts = len(document.sites) - len(sites)
    if other_hosts:
        facts.append(f"{other_hosts} from a previous host")
    return tuple(facts)


def _relaxation_facts(document: ProjectDocument) -> tuple[str, ...]:
    definition = active_muon_relaxation_definition(document)
    if definition is None:
        return ("No relaxations configured",)
    totals: Counter[str] = Counter()
    relaxed: Counter[str] = Counter()
    active: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    for _label, charge_state, _site_id, job in muon_relaxation_row_jobs(document, definition):
        totals[charge_state] += 1
        if job is None:
            continue
        if job.status == JobStatus.COMPLETED:
            relaxed[charge_state] += _relaxed(document, job)
        elif job.status == JobStatus.ACTIVE:
            active[charge_state] += 1
        elif job_status_word(job) == "failed":
            failed[charge_state] += 1
    facts: list[str] = []
    for charge_state, total in totals.items():
        fact = f"{muon_charge_state_label(charge_state).capitalize()}: {relaxed[charge_state]} of {total} relaxed"
        extras = [f"{active[charge_state]} active"] if active[charge_state] else []
        if failed[charge_state]:
            extras.append(f"{failed[charge_state]} failed")
        facts.append(f"{fact}, {', '.join(extras)}" if extras else fact)
    return tuple(facts) or ("No relaxations configured",)


def _analysis_facts(document: ProjectDocument) -> tuple[str, ...]:
    workspaces = relaxation_analysis_workspaces(document)
    if not workspaces:
        return ("No analysis workspaces yet",)
    facts = [plural(len(workspaces), "analysis workspace")]
    snapshots = document.ui_state.get(MUON_SITES_SUMMARY_SNAPSHOTS_KEY)
    if isinstance(snapshots, Mapping):
        for charge_state, snapshot in snapshots.items():
            sites = snapshot.get("sites") if isinstance(snapshot, Mapping) else None
            if isinstance(sites, list) and sites:
                label = muon_charge_state_label(str(charge_state)).capitalize()
                facts.append(f"{label}: {plural(len(sites), 'distinct site')} pinned")
    return tuple(facts)


def _magnetic_facts(document: ProjectDocument) -> tuple[str, ...]:
    facts: list[str] = []
    magnetic_structure = document.project.magnetic_structure
    components = (
        magnetic_structure.get("components") if isinstance(magnetic_structure, Mapping) else None
    )
    if isinstance(components, list) and components:
        facts.append(plural(len(components), "magnetic component"))
    workspaces = magnetic_field_workspaces(document)
    if workspaces:
        facts.append(plural(len(workspaces), "saved workspace"))
    return tuple(facts) or (_NOTHING_YET["magnetic_fields"],)


def _polarisation_facts(document: ProjectDocument) -> tuple[str, ...]:
    count = sum(1 for result in document.results.values() if _is_polarisation_simulation(result))
    return (plural(count, "simulation"),) if count else (_NOTHING_YET["polarisation"],)


def _hyperfine_facts(document: ProjectDocument) -> tuple[str, ...]:
    count = sum(
        1 for result in document.results.values() if result.result_type == HYPERFINE_CALC_TYPE
    )
    facts = [plural(count, "hyperfine result")] if count else []
    facts.extend(_job_facts(document, WorkflowStage.HYPERFINE_COUPLING, "job", completed=False))
    return tuple(facts) or (_NOTHING_YET["hyperfine_coupling"],)


def _quantum_facts(document: ProjectDocument) -> tuple[str, ...]:
    results = [
        result
        for result in document.results.values()
        if result.result_type == QUANTUM_PHONON_CALC_TYPE
    ]
    facts = [plural(len(results), "phonon result")] if results else []
    facts.extend(_job_facts(document, WorkflowStage.QUANTUM_CORRECTIONS, "job", completed=False))
    return tuple(facts) or (_NOTHING_YET["quantum_corrections"],)


def _barrier_facts(document: ProjectDocument) -> tuple[str, ...]:
    heights: list[float] = []
    for result in document.results.values():
        if result.result_type not in _BARRIER_RESULT_TYPES or not isinstance(result.data, Mapping):
            continue
        height = result.data.get("barrier_height_eV")
        if isinstance(height, (int, float)) and math.isfinite(height):
            heights.append(float(height))
    facts = []
    if heights:
        facts.append(f"{plural(len(heights), 'barrier')}, lowest {min(heights):.2f} eV")
    facts.extend(_job_facts(document, WorkflowStage.ENERGY_BARRIERS, "job", completed=False))
    return tuple(facts) or (_NOTHING_YET["energy_barriers"],)


_FACTS: dict[str, Callable[[ProjectDocument], tuple[str, ...]]] = {
    "host_structure": _host_facts,
    "convergence_cutoff": _cutoff_facts,
    "convergence_kpoint": _kpoint_facts,
    "pristine_relaxation": _pristine_facts,
    "uep": _uep_facts,
    "muon_candidates": _candidate_facts,
    "muon_relaxations": _relaxation_facts,
    "muon_analysis": _analysis_facts,
    "hyperfine_coupling": _hyperfine_facts,
    "magnetic_fields": _magnetic_facts,
    "polarisation": _polarisation_facts,
    "quantum_corrections": _quantum_facts,
    "energy_barriers": _barrier_facts,
}


def _job_facts(
    document: ProjectDocument,
    stage: WorkflowStage,
    noun: str,
    *,
    completed: bool = True,
) -> list[str]:
    jobs = [job for job in document.jobs.values() if job.workflow_stage == stage]
    if not jobs:
        return []
    statuses = Counter(job.status for job in jobs)
    failed = sum(1 for job in jobs if job_status_word(job) == "failed")
    parts: list[str] = []
    if completed and statuses[JobStatus.COMPLETED]:
        parts.append(f"{plural(statuses[JobStatus.COMPLETED], f'{noun}')} complete")
    if statuses[JobStatus.ACTIVE]:
        parts.append(f"{statuses[JobStatus.ACTIVE]} active")
    if failed:
        parts.append(f"{failed} failed")
    return [", ".join(parts)] if parts else []


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


__all__: Iterable[str] = (
    "COMPLETION_CHECKS",
    "OTHER_RESULT_STAGES",
    "MAGNETIC_FIELD_WORKSPACES_KEY",
    "MUON_SITES_SUMMARY_SNAPSHOTS_KEY",
    "RELAXATION_ANALYSIS_WORKSPACES_KEY",
    "STAGE_GROUPS",
    "STAGE_PHASES",
    "WORKFLOW_STAGES",
    "ProjectSummary",
    "LastStage",
    "StageGroup",
    "StagePhase",
    "StageStatus",
    "StageSummary",
    "WorkflowStageInfo",
    "group_status",
    "stage_phase",
    "stage_statuses",
    "summarize_project",
)
