"""Where a project stands: the workflow's stages, their dependencies and their progress.

This module is the one home for the workflow-stage catalogue the project
navigator and the project home are built from (``WORKFLOW_STAGES``,
``STAGE_GROUPS``), the dependency graph between stages
(``STAGE_DEPENDENCIES``), the completion rule of each stage (``*_complete``) and
the per-stage summary the project home shows (``summarize_project``). It is
read-only and UI-free.

A dependency is ``required`` when the code blocks the later stage without the
earlier one, ``recommended`` when the earlier stage only prefills defaults, and
``optional`` when it is one of several routes into the later stage.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
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
from mufinder.domain.enums import ACTIVE_JOB_STATUSES, JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Structure
from mufinder.science.ase_bridge import is_muon_label, reduced_formula
from mufinder.services import supercell_assets
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
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE

#: ``ui_state`` key holding the relaxation-analysis workspaces.
RELAXATION_ANALYSIS_WORKSPACES_KEY = "relaxation_analysis_workspaces"
#: ``ui_state`` key holding the magnetic-field workspaces.
MAGNETIC_FIELD_WORKSPACES_KEY = "magnetic_field_workspaces"
#: ``ui_state`` key holding the pinned Muon Sites summary, one snapshot per charge state.
MUON_SITES_SUMMARY_SNAPSHOTS_KEY = "muon_sites_summary_snapshots"
#: ``ui_state`` key holding the stage the user last worked on, shown first on the project home.
PROJECT_HOME_STAGE_KEY = "project_home_stage"


class StageStatus(StrEnum):
    """How far a workflow stage has got."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class DependencyKind(StrEnum):
    """How strongly a later stage depends on an earlier one."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


@dataclass(frozen=True, slots=True)
class StageGroup:
    """A navigator section that holds several stages, such as Convergence Testing."""

    key: str
    title: str
    completion_goal: str


@dataclass(frozen=True, slots=True)
class WorkflowStageInfo:
    """One workflow stage: its navigator route, title, completion rule and guidance.

    ``key`` is also the stage's navigator route. ``guidance`` says what the
    stage is for and what it needs, in one or two sentences. An ``optional``
    stage is a side route the main workflow does not pass through.
    """

    key: str
    title: str
    completion_goal: str
    guidance: str
    group: str | None = None
    optional: bool = False


@dataclass(frozen=True, slots=True)
class StageDependency:
    """An edge of the workflow graph: ``target`` builds on ``source``.

    ``source`` is a stage key or a :class:`StageGroup` key; a group counts as
    complete when every stage in it is.
    """

    source: str
    target: str
    kind: DependencyKind


@dataclass(frozen=True, slots=True)
class StageSummary:
    """Where one stage stands, for the project home."""

    key: str
    title: str
    status: StageStatus
    facts: tuple[str, ...]
    guidance: str
    active_jobs: int = 0
    unmet_requirements: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectSummary:
    """Every stage's summary in workflow order, and the stage to show first."""

    stages: tuple[StageSummary, ...]
    current_stage: str

    def stage(self, key: str) -> StageSummary | None:
        return next((stage for stage in self.stages if stage.key == key), None)

    @property
    def completed_count(self) -> int:
        return sum(stage.status == StageStatus.COMPLETE for stage in self.stages)

    @property
    def active_jobs(self) -> int:
        return sum(stage.active_jobs for stage in self.stages)


STAGE_GROUPS: tuple[StageGroup, ...] = (
    StageGroup(
        "convergence",
        "Convergence Testing",
        "Set both project default convergence values (cutoff and k-point).",
    ),
    StageGroup("muon_sites", "Muon Sites", "Generate at least one muon site."),
)

WORKFLOW_STAGES: tuple[WorkflowStageInfo, ...] = (
    WorkflowStageInfo(
        "host_structure",
        "Host Structure",
        "Load a host structure.",
        "Start here: import the crystal the project studies. Every other stage works on the active host.",
    ),
    WorkflowStageInfo(
        "supercell_editor",
        "Supercell Editor",
        "Save reusable supercells for workflow setup.",
        "Optional. Any calculation can run in a saved supercell instead of the unit cell.",
        optional=True,
    ),
    WorkflowStageInfo(
        "convergence_cutoff",
        "Cutoff Energy",
        "Set a project default cutoff value.",
        "Recommended before other calculations: the converged cutoff becomes the project default.",
        group="convergence",
    ),
    WorkflowStageInfo(
        "convergence_kpoint",
        "K-Point Grid",
        "Set a project default k-point grid.",
        "Recommended before other calculations: the converged grid becomes the project default.",
        group="convergence",
    ),
    WorkflowStageInfo(
        "pristine_relaxation",
        "Pristine Relaxation",
        "Generate at least one comparison table.",
        "Optional detour. To carry on with the relaxed host, make it the active host on the "
        "Host Structure page; UEP can also use it as a source.",
        optional=True,
    ),
    WorkflowStageInfo(
        "uep",
        "UEP",
        "Complete at least one UEP source job.",
        "Optional route to candidate sites through the unperturbed electrostatic potential, and a "
        "source of estimated sites for magnetic fields.",
        optional=True,
    ),
    WorkflowStageInfo(
        "muon_candidates",
        "Generation",
        "Generate at least one muon site.",
        "Place candidate muon sites: from UEP clusters, at random, near an element or by hand.",
        group="muon_sites",
    ),
    WorkflowStageInfo(
        "muon_relaxations",
        "Relaxations",
        "All structural relaxations completed.",
        "Relax each staged candidate with the muon in place. Needs generated sites.",
        group="muon_sites",
    ),
    WorkflowStageInfo(
        "muon_analysis",
        "Analysis",
        "Create at least one structural-relaxation analysis workspace.",
        "Cluster the converged relaxations into distinct sites and pin them on the Muon Sites page. "
        "Needs completed relaxations of one charge state.",
        group="muon_sites",
    ),
    WorkflowStageInfo(
        "magnetic_fields",
        "Magnetic Fields",
        "Define at least one magnetic component.",
        "Needs a magnetic model and distinct sites from Analysis; UEP clusters give unrelaxed estimates.",
    ),
    WorkflowStageInfo(
        "quantum_corrections",
        "Quantum Corrections",
        "Complete at least one quantum phonon analysis.",
        "Zero-point energy, spread and host entanglement of each distinct relaxed site. Needs relaxed muon sites.",
    ),
    WorkflowStageInfo(
        "energy_barriers",
        "Energy Barriers",
        "Complete at least one NEB barrier calculation.",
        "Barriers for a muon hopping between sites. Needs two relaxed sites of the same charge state.",
    ),
)

STAGE_DEPENDENCIES: tuple[StageDependency, ...] = (
    StageDependency("host_structure", "supercell_editor", DependencyKind.OPTIONAL),
    StageDependency("host_structure", "convergence_cutoff", DependencyKind.REQUIRED),
    StageDependency("host_structure", "convergence_kpoint", DependencyKind.REQUIRED),
    StageDependency("convergence", "pristine_relaxation", DependencyKind.OPTIONAL),
    StageDependency("convergence", "uep", DependencyKind.RECOMMENDED),
    StageDependency("convergence", "muon_candidates", DependencyKind.RECOMMENDED),
    StageDependency("pristine_relaxation", "uep", DependencyKind.OPTIONAL),
    StageDependency("pristine_relaxation", "muon_candidates", DependencyKind.OPTIONAL),
    StageDependency("uep", "muon_candidates", DependencyKind.OPTIONAL),
    StageDependency("muon_candidates", "muon_relaxations", DependencyKind.REQUIRED),
    StageDependency("muon_relaxations", "muon_analysis", DependencyKind.REQUIRED),
    StageDependency("muon_analysis", "magnetic_fields", DependencyKind.REQUIRED),
    StageDependency("muon_analysis", "quantum_corrections", DependencyKind.REQUIRED),
    StageDependency("muon_analysis", "energy_barriers", DependencyKind.REQUIRED),
    StageDependency("uep", "magnetic_fields", DependencyKind.OPTIONAL),
)

_STAGE_JOB_STAGES: dict[str, tuple[WorkflowStage, ...]] = {
    "convergence_cutoff": (WorkflowStage.CONVERGENCE_CUTOFF,),
    "convergence_kpoint": (WorkflowStage.CONVERGENCE_KPOINT,),
    "pristine_relaxation": (WorkflowStage.PRISTINE_RELAXATION,),
    "uep": (WorkflowStage.UEP,),
    "muon_relaxations": (WorkflowStage.MUON_RELAXATION,),
    "magnetic_fields": (WorkflowStage.MAGNETIC_FIELD,),
    "quantum_corrections": (WorkflowStage.QUANTUM_CORRECTIONS,),
    "energy_barriers": (WorkflowStage.ENERGY_BARRIERS,),
}

_STAGES_BY_KEY = {stage.key: stage for stage in WORKFLOW_STAGES}
_GROUPS_BY_KEY = {group.key: group for group in STAGE_GROUPS}


def stage_info(key: str) -> WorkflowStageInfo:
    """Return the stage with ``key``; raises ``KeyError`` for an unknown key."""

    return _STAGES_BY_KEY[key]


def stage_group(key: str) -> StageGroup:
    """Return the stage group with ``key``; raises ``KeyError`` for an unknown key."""

    return _GROUPS_BY_KEY[key]


def group_stage_keys(group_key: str) -> tuple[str, ...]:
    """Return the keys of the stages in a group, in workflow order."""

    return tuple(stage.key for stage in WORKFLOW_STAGES if stage.group == group_key)


# --- Completion rules ---------------------------------------------------------


def host_complete(document: ProjectDocument) -> bool:
    return document.project.host_structure_id is not None


def supercell_editor_complete(document: ProjectDocument) -> bool:
    return bool(
        supercell_assets.list_workflow_supercell_assets(
            document,
            host_structure_id=document.project.host_structure_id,
        )
    )


def cutoff_complete(document: ProjectDocument) -> bool:
    return document.project.recommended_cutoff_energy is not None


def kpoint_complete(document: ProjectDocument) -> bool:
    return document.project.recommended_kpoint_grid is not None


def convergence_complete(document: ProjectDocument) -> bool:
    return cutoff_complete(document) and kpoint_complete(document)


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


def relaxation_analysis_workspace_clustered(document: ProjectDocument, workspace_id: str) -> bool:
    for workspace in relaxation_analysis_workspaces(document):
        if str(workspace.get("id", "")) == workspace_id:
            clustering = workspace.get("clustering")
            return isinstance(clustering, dict) and clustering.get("result") is not None
    return False


def magnetic_fields_complete(document: ProjectDocument) -> bool:
    return magnetic_structure_has_components(document.project.magnetic_structure)


def magnetic_field_workspace_complete(document: ProjectDocument, workspace_id: str) -> bool:
    for workspace in magnetic_field_workspaces(document):
        if str(workspace.get("id", "")).strip() == workspace_id:
            return magnetic_structure_has_components(workspace.get("magnetic_structure"))
    return False


def magnetic_structure_has_components(magnetic_structure: object) -> bool:
    if not isinstance(magnetic_structure, Mapping):
        return False
    components = magnetic_structure.get("components")
    if isinstance(components, list):
        return any(isinstance(component, dict) for component in components)
    moments = magnetic_structure.get("moments")
    return isinstance(moments, list) and bool(moments)


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


#: The completion rule of every stage and stage group, by key.
COMPLETION_CHECKS: dict[str, Callable[[ProjectDocument], bool]] = {
    "host_structure": host_complete,
    "supercell_editor": supercell_editor_complete,
    "convergence": convergence_complete,
    "convergence_cutoff": cutoff_complete,
    "convergence_kpoint": kpoint_complete,
    "pristine_relaxation": pristine_relaxation_complete,
    "uep": uep_complete,
    "muon_sites": muon_candidates_complete,
    "muon_candidates": muon_candidates_complete,
    "muon_relaxations": muon_relaxations_complete,
    "muon_analysis": relaxation_analysis_complete,
    "magnetic_fields": magnetic_fields_complete,
    "quantum_corrections": quantum_corrections_complete,
    "energy_barriers": energy_barriers_complete,
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
        site_id = _uuid_or_none(entry.get("site_id"))
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
    """Return every stage's status and facts, and the stage the project home shows first.

    A stage that has not started lists the required earlier stages that have
    not started either, plus the host when there is none, as its unmet
    requirements. A started earlier stage counts as met: relaxation analysis,
    for example, can begin once some relaxations have finished.
    """

    complete = {key: check(document) for key, check in COMPLETION_CHECKS.items()}
    active_by_stage, any_jobs_by_stage = _job_counts(document)
    statuses: dict[str, StageStatus] = {}
    active_jobs: dict[str, int] = {}
    for info in WORKFLOW_STAGES:
        job_stages = _STAGE_JOB_STAGES.get(info.key, ())
        active_jobs[info.key] = sum(active_by_stage[stage] for stage in job_stages)
        started = any(any_jobs_by_stage[stage] for stage in job_stages) or _has_partial_work(
            document, info.key
        )
        if complete[info.key]:
            statuses[info.key] = StageStatus.COMPLETE
        elif started:
            statuses[info.key] = StageStatus.IN_PROGRESS
        else:
            statuses[info.key] = StageStatus.NOT_STARTED
    stages = [
        StageSummary(
            key=info.key,
            title=info.title,
            status=statuses[info.key],
            facts=_FACTS[info.key](document),
            guidance=info.guidance,
            active_jobs=active_jobs[info.key],
            unmet_requirements=(
                _unmet_requirements(info.key, statuses)
                if statuses[info.key] == StageStatus.NOT_STARTED
                else ()
            ),
        )
        for info in WORKFLOW_STAGES
    ]
    return ProjectSummary(stages=tuple(stages), current_stage=_current_stage(document, stages))


def _unmet_requirements(key: str, statuses: Mapping[str, StageStatus]) -> tuple[str, ...]:
    def started(source: str) -> bool:
        members = group_stage_keys(source) if source in _GROUPS_BY_KEY else (source,)
        return any(statuses[member] != StageStatus.NOT_STARTED for member in members)

    unmet: list[str] = []
    if key != "host_structure" and statuses["host_structure"] != StageStatus.COMPLETE:
        unmet.append(stage_info("host_structure").title)
    for dependency in STAGE_DEPENDENCIES:
        if (
            dependency.target != key
            or dependency.kind != DependencyKind.REQUIRED
            or started(dependency.source)
        ):
            continue
        title = (_STAGES_BY_KEY.get(dependency.source) or _GROUPS_BY_KEY[dependency.source]).title
        if title not in unmet:
            unmet.append(title)
    return tuple(unmet)


def _current_stage(document: ProjectDocument, stages: list[StageSummary]) -> str:
    saved = document.ui_state.get(PROJECT_HOME_STAGE_KEY)
    if isinstance(saved, str) and saved in _STAGES_BY_KEY:
        return saved
    in_progress = [stage.key for stage in stages if stage.status == StageStatus.IN_PROGRESS]
    if in_progress:
        return in_progress[-1]
    for stage in stages:
        if (
            stage.status != StageStatus.COMPLETE
            and not stage.unmet_requirements
            and not stage_info(stage.key).optional
        ):
            return stage.key
    return stages[-1].key


def _job_counts(document: ProjectDocument) -> tuple[Counter[WorkflowStage], Counter[WorkflowStage]]:
    active: Counter[WorkflowStage] = Counter()
    present: Counter[WorkflowStage] = Counter()
    for job in document.jobs.values():
        present[job.workflow_stage] += 1
        if job.status in ACTIVE_JOB_STATUSES:
            active[job.workflow_stage] += 1
    return active, present


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
    facts = [f"{formula} · {_count(atoms, 'atom')}"]
    if host.origin == "pristine_relaxation":
        facts.append("Active host is a relaxed structure")
    return tuple(facts)


def _supercell_facts(document: ProjectDocument) -> tuple[str, ...]:
    assets = supercell_assets.list_workflow_supercell_assets(
        document,
        host_structure_id=document.project.host_structure_id,
    )
    if not assets:
        return ("No saved supercells",)
    labels = ", ".join(asset.label for asset in assets[:3])
    more = f" and {len(assets) - 3} more" if len(assets) > 3 else ""
    return (f"{_count(len(assets), 'saved supercell')}: {labels}{more}",)


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
        facts.append(_count(len(relaxed), "relaxed structure"))
    if document.comparison_tables:
        facts.append(_count(len(document.comparison_tables), "comparison table"))
    return tuple(facts) or ("Not visited",)


def _uep_facts(document: ProjectDocument) -> tuple[str, ...]:
    facts = list(_job_facts(document, WorkflowStage.UEP, "source job"))
    states = document.ui_state.get("uep_analysis_states")
    minima = 0
    if isinstance(states, Mapping):
        for state in states.values():
            summary = state.get("batch_summary_data") if isinstance(state, Mapping) else None
            clusters = summary.get("clusters") if isinstance(summary, Mapping) else None
            if isinstance(clusters, list):
                minima += sum(isinstance(cluster, Mapping) for cluster in clusters)
    if minima:
        facts.append(_count(minima, "clustered minimum", "clustered minima"))
    return tuple(facts) or ("Not visited",)


def _candidate_facts(document: ProjectDocument) -> tuple[str, ...]:
    host_id = document.project.host_structure_id
    sites = [
        site for site in document.sites.values() if site.source_structure_id in (None, host_id)
    ]
    if not sites:
        return ("No candidate sites yet",)
    from_uep = sum(site.generation_method.startswith("uep_cluster") for site in sites)
    facts = [_count(len(sites), "candidate site")]
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
            run = latest_run_for_job(document, job.id)
            if run is None or run.convergence_achieved is not False:
                relaxed[charge_state] += 1
        elif job.status in ACTIVE_JOB_STATUSES:
            active[charge_state] += 1
        elif job.status == JobStatus.FAILED:
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
    facts = [_count(len(workspaces), "analysis workspace")]
    snapshots = document.ui_state.get(MUON_SITES_SUMMARY_SNAPSHOTS_KEY)
    if isinstance(snapshots, Mapping):
        for charge_state, snapshot in snapshots.items():
            sites = snapshot.get("sites") if isinstance(snapshot, Mapping) else None
            if isinstance(sites, list) and sites:
                label = muon_charge_state_label(str(charge_state)).capitalize()
                facts.append(f"{label}: {_count(len(sites), 'distinct site')} pinned")
    return tuple(facts)


def _magnetic_facts(document: ProjectDocument) -> tuple[str, ...]:
    facts: list[str] = []
    magnetic_structure = document.project.magnetic_structure
    components = (
        magnetic_structure.get("components") if isinstance(magnetic_structure, Mapping) else None
    )
    if isinstance(components, list) and components:
        facts.append(_count(len(components), "magnetic component"))
    workspaces = magnetic_field_workspaces(document)
    if workspaces:
        facts.append(_count(len(workspaces), "saved workspace"))
    return tuple(facts) or ("No magnetic model yet",)


def _quantum_facts(document: ProjectDocument) -> tuple[str, ...]:
    results = [
        result
        for result in document.results.values()
        if result.result_type == QUANTUM_PHONON_CALC_TYPE
    ]
    facts = [_count(len(results), "phonon result")] if results else []
    facts.extend(_job_facts(document, WorkflowStage.QUANTUM_CORRECTIONS, "job", completed=False))
    return tuple(facts) or ("No quantum calculations yet",)


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
        facts.append(f"{_count(len(heights), 'barrier')}, lowest {min(heights):.2f} eV")
    facts.extend(_job_facts(document, WorkflowStage.ENERGY_BARRIERS, "job", completed=False))
    return tuple(facts) or ("No barrier calculations yet",)


_FACTS: dict[str, Callable[[ProjectDocument], tuple[str, ...]]] = {
    "host_structure": _host_facts,
    "supercell_editor": _supercell_facts,
    "convergence_cutoff": _cutoff_facts,
    "convergence_kpoint": _kpoint_facts,
    "pristine_relaxation": _pristine_facts,
    "uep": _uep_facts,
    "muon_candidates": _candidate_facts,
    "muon_relaxations": _relaxation_facts,
    "muon_analysis": _analysis_facts,
    "magnetic_fields": _magnetic_facts,
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
    statuses = Counter(job.status for job in document.jobs.values() if job.workflow_stage == stage)
    if not statuses:
        return []
    parts: list[str] = []
    if completed and statuses[JobStatus.COMPLETED]:
        parts.append(f"{_count(statuses[JobStatus.COMPLETED], f'{noun}')} complete")
    active = sum(statuses[status] for status in ACTIVE_JOB_STATUSES)
    if active:
        parts.append(f"{active} active")
    if statuses[JobStatus.FAILED]:
        parts.append(f"{statuses[JobStatus.FAILED]} failed")
    return [", ".join(parts)] if parts else []


def _count(value: int, singular: str, plural: str | None = None) -> str:
    return f"{value} {singular if value == 1 else plural or singular + 's'}"


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _uuid_or_none(value: object) -> UUID | None:
    if not value:
        return None
    try:
        return UUID(str(value))
    except ValueError:
        return None


__all__: Iterable[str] = (
    "COMPLETION_CHECKS",
    "MAGNETIC_FIELD_WORKSPACES_KEY",
    "MUON_SITES_SUMMARY_SNAPSHOTS_KEY",
    "PROJECT_HOME_STAGE_KEY",
    "RELAXATION_ANALYSIS_WORKSPACES_KEY",
    "STAGE_DEPENDENCIES",
    "STAGE_GROUPS",
    "WORKFLOW_STAGES",
    "DependencyKind",
    "ProjectSummary",
    "StageDependency",
    "StageGroup",
    "StageStatus",
    "StageSummary",
    "WorkflowStageInfo",
    "summarize_project",
)
