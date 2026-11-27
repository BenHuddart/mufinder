"""Quantum-corrections CASTEP workflow expansion."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence
from uuid import UUID

from mufinder.castep_io.schema import canonical_keyword_name
from mufinder.castep_io.types import BlockSection, KeywordAssignment, ValueWithUnit
from mufinder.domain.charge_states import is_muonium, seed_muon_spins
from mufinder.domain.constants import MUON_GAMMA_RAD_S_T, MUON_MASS_AMU
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Structure

from ._common import require_jobs
from .merge import compose_patch, resolve_job
from .naming import job_name
from .types import CellTemplate, InputTemplate, JobPatch, JobSpec, ParamTemplate, StructureSpec

#: Calc type of the singlepoint job older projects ran before the phonon job.
#: MuFinder no longer creates such jobs; the name is kept so a stored job or
#: result of that type is still recognised (see ``docs/compatibility_policy.rst``).
QUANTUM_SINGLEPOINT_CALC_TYPE = "castep_quantum_singlepoint"
QUANTUM_PHONON_CALC_TYPE = "castep_quantum_phonon"
QUANTUM_PHONON_SET_CALC_TYPE = "castep_quantum_phonon_set"
QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL = ValueWithUnit(1e-9, "eV")
#: The only phonon method Quantum Corrections runs. Finite displacement works
#: with ultrasoft pseudopotentials and with every setting CASTEP's DFPT does not
#: support, so every phonon job writes it explicitly.
QUANTUM_PHONON_METHOD = "FINITEDISPLACEMENT"


def seed_muonium_spin(structure: StructureSpec, muon_species: str) -> StructureSpec:
    """Seed +1 on the muon so a muonium job stays open-shell.

    Mirrors the muon-relaxation adapter: neutral muonium binds one unpaired electron
    (S = 1/2, ~1 muB). Host moments are preserved, and this is idempotent if the relaxed
    source structure already carried the seed.
    """

    positions = structure.positions_frac.positions
    muon_indices = [
        index for index, position in enumerate(positions) if position.species == muon_species
    ]
    # Fail loudly rather than emit a spin-polarised muonium job with no seeded moment: a
    # zero net moment on an odd-electron cell silently converges to the wrong (closed-shell)
    # solution. This catches a species-label mismatch between the structure and muon_species.
    if not muon_indices:
        raise ValueError(
            f"Muonium structure has no atom matching muon species {muon_species!r}; "
            f"cannot seed the +1 muon spin. Present species: "
            f"{sorted({position.species for position in positions})}."
        )
    spins = seed_muon_spins(
        structure.atomic_spins, atom_count=len(positions), muon_indices=muon_indices
    )
    return replace(structure, atomic_spins=tuple(spins))


@dataclass(frozen=True, slots=True)
class QuantumPhononSource:
    """One eligible relaxed muon site for a phonon calculation."""

    label: str
    structure: StructureSpec
    source_site_id: UUID
    source_relaxation_run_id: UUID
    source_relaxation_job_id: UUID | None = None
    charge_state: str | None = None
    total_energy_eV: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QuantumPhononJobConfig:
    """Configuration shared by one generated Quantum Corrections job."""

    source: QuantumPhononSource
    muon_species: str = "H:mu"
    num_farms: int = 1
    cell_overrides: CellTemplate | None = None
    param_overrides: ParamTemplate | None = None


@dataclass(frozen=True, slots=True)
class QuantumSourceEligibility:
    """Eligibility verdict for a possible quantum-corrections source."""

    eligible: bool
    reason: str = ""


def quantum_source_eligibility(
    *,
    site: MuonSite,
    run: CalculationRun | None,
    job: Job | None,
    relaxed_structure: Structure | None,
    site_link: bool = True,
) -> QuantumSourceEligibility:
    """Validate that a site came from a completed structural muon relaxation.

    With ``site_link`` (the default) ``run`` must be the relaxation the site's
    own link records and the site must be marked relaxed. Without it ``run`` is
    a relaxation of the site in another charge state, found through the
    muon-relaxation row bound to the site, so the site's link and status are not
    checked; ``run`` must still belong to ``job``.
    """

    if site_link:
        if site.status != "relaxed":
            return QuantumSourceEligibility(False, "Muon site has not been structurally relaxed.")
        if site.relaxation_run_id is None:
            return QuantumSourceEligibility(
                False, "Muon site has no structural-relaxation run provenance."
            )
        if run is None or run.id != site.relaxation_run_id:
            return QuantumSourceEligibility(False, "The recorded relaxation run is missing.")
    elif run is None or job is None or run.job_id != job.id:
        return QuantumSourceEligibility(False, "The recorded relaxation run is missing.")
    if run.status != JobStatus.COMPLETED:
        return QuantumSourceEligibility(False, "The structural-relaxation run is not complete.")
    if job is None or job.workflow_stage != WorkflowStage.MUON_RELAXATION:
        return QuantumSourceEligibility(
            False, "The source job is not a MuFinder muon structural relaxation."
        )
    if job.status != JobStatus.COMPLETED:
        return QuantumSourceEligibility(False, "The source muon relaxation job is not complete.")
    if relaxed_structure is None:
        return QuantumSourceEligibility(
            False, "The source relaxation has no parsed relaxed structure."
        )
    if relaxed_structure.origin != "muon_relaxation":
        return QuantumSourceEligibility(
            False, "The source structure was not produced by muon structural relaxation."
        )
    if (
        relaxed_structure.properties.get("source_workflow_stage")
        != WorkflowStage.MUON_RELAXATION.value
    ):
        return QuantumSourceEligibility(
            False, "The source structure does not carry muon-relaxation provenance."
        )
    if "H:mu" not in relaxed_structure.species:
        return QuantumSourceEligibility(
            False, "The relaxed structure does not contain an H:mu atom."
        )
    return QuantumSourceEligibility(True)


def expand_quantum_phonon_jobs(
    *,
    jobs: Sequence[QuantumPhononJobConfig],
    name_prefix: str,
    global_defaults: InputTemplate | None = None,
    workflow_defaults: InputTemplate | None = None,
    structure_defaults: InputTemplate | None = None,
    user_overrides: InputTemplate | None = None,
    tags: tuple[str, ...] = (),
) -> list[JobSpec]:
    """Expand quantum-corrections configurations into one CASTEP phonon job each.

    Each job is a fresh ``TASK : PHONON`` run: CASTEP converges the ground state
    and then computes the phonons in the same job, with the muon mass and
    gyromagnetic ratio written in its own ``.cell``. ``CONTINUATION : DEFAULT``
    makes a rerun in a directory that still holds the job's ``.check`` resume
    from it, and is a new run otherwise. Every job runs finite-displacement
    phonons (:data:`QUANTUM_PHONON_METHOD`); a ``PHONON_METHOD`` in a layer
    applied after it (a config's ``param_overrides`` or ``user_overrides``) is
    refused. The defaults layers sit below the workflow's own value, so theirs
    never reaches the file. The method and the farm count are recorded in the
    provenance.
    """

    require_jobs(jobs, "quantum phonon job")
    _refuse_phonon_method_override(user_overrides.param if user_overrides is not None else None)

    resolved: list[JobSpec] = []
    for config in jobs:
        label = config.source.label.strip()
        if not label:
            raise ValueError("Quantum phonon source labels must not be empty.")
        _refuse_phonon_method_override(config.param_overrides)

        structure = config.source.structure
        if is_muonium(config.source.charge_state):
            structure = seed_muonium_spin(structure, config.muon_species)
        base = InputTemplate(cell=_cell_base_template(config), param=_param_base_template(config))
        patch_template = compose_patch(
            base,
            InputTemplate(cell=config.cell_overrides, param=config.param_overrides),
        )
        patch = JobPatch(
            label_suffix=label,
            structure=structure,
            cell=patch_template.cell,
            param=patch_template.param,
        )

        provenance: dict[str, Any] = {
            "workflow": "quantum_corrections",
            "calc_type": QUANTUM_PHONON_CALC_TYPE,
            "label": label,
            # The provenance spelling older jobs recorded; the .param gets QUANTUM_PHONON_METHOD.
            "phonon_method": "finite_displacement",
            "phonon_kpoint_mp_grid": (1, 1, 1),
            "muon_species": config.muon_species,
            "muon_mass_amu": MUON_MASS_AMU,
            "muon_gamma_rad_s_T": MUON_GAMMA_RAD_S_T,
            "num_farms": max(1, int(config.num_farms)),
        }
        provenance["source_site_id"] = str(config.source.source_site_id)
        provenance["source_relaxation_run_id"] = str(config.source.source_relaxation_run_id)
        provenance["source_relaxation_job_id"] = (
            str(config.source.source_relaxation_job_id)
            if config.source.source_relaxation_job_id is not None
            else None
        )
        provenance["charge_state"] = config.source.charge_state
        provenance.update(config.source.provenance)

        resolved.append(
            resolve_job(
                name=job_name(name_prefix, label),
                base_structure=structure,
                workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
                global_defaults=global_defaults,
                workflow_defaults=workflow_defaults,
                structure_defaults=structure_defaults,
                patch=patch,
                user_overrides=user_overrides,
                tags=("quantum_corrections", "phonon", *tags),
                provenance=provenance,
            )
        )
    return resolved


def muon_species_blocks(muon_species: str) -> tuple[BlockSection, ...]:
    """Return the muon ``SPECIES_MASS`` / ``SPECIES_GAMMA`` blocks.

    Without them CASTEP gives ``H:mu`` the default hydrogen mass and gyromagnetic
    ratio: phonon frequencies and the muon ZPE come out roughly a factor of three
    too low (omega scales as 1/sqrt(mass)), and a hyperfine tensor is scaled by
    the proton's gamma instead of the muon's. A fresh run reads them from the
    job's own ``.cell``.
    """

    return (
        BlockSection("species_mass", ((muon_species, MUON_MASS_AMU),), unit="AMU"),
        BlockSection("species_gamma", ((muon_species, MUON_GAMMA_RAD_S_T),), unit="radsectesla"),
    )


def _cell_base_template(config: QuantumPhononJobConfig) -> CellTemplate:
    return CellTemplate(
        extra_scalars=(KeywordAssignment("phonon_kpoint_mp_grid", (1, 1, 1)),),
        extra_blocks=muon_species_blocks(config.muon_species),
    )


def _param_base_template(config: QuantumPhononJobConfig) -> ParamTemplate:
    keywords = [
        KeywordAssignment("task", "PHONON"),
        KeywordAssignment("phonon_method", QUANTUM_PHONON_METHOD),
        KeywordAssignment("continuation", "DEFAULT"),
        KeywordAssignment("nonscf_eigenvalue_tol", _phonon_nonscf_eigenvalue_tol(config)),
    ]
    if int(config.num_farms) > 1:
        keywords.append(KeywordAssignment("num_farms", max(1, int(config.num_farms))))
    if config.source.charge_state:
        charge = 0 if is_muonium(config.source.charge_state) else 1
        keywords.append(KeywordAssignment("charge", charge))
    if is_muonium(config.source.charge_state):
        # Muonium is open-shell: default to spin-polarised (overridable via param_overrides).
        keywords.append(KeywordAssignment("spin_polarized", True))
    return ParamTemplate(keywords=tuple(keywords))


def _phonon_nonscf_eigenvalue_tol(config: QuantumPhononJobConfig) -> ValueWithUnit:
    elec_energy_tol = _param_keyword_value(config.param_overrides, "elec_energy_tol")
    if isinstance(elec_energy_tol, ValueWithUnit):
        normalized_unit = elec_energy_tol.unit.strip().lower().replace(" ", "")
        if normalized_unit == "ev":
            return ValueWithUnit(
                min(
                    float(elec_energy_tol.value), float(QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL.value)
                ),
                "eV",
            )
    if isinstance(elec_energy_tol, (int, float)) and not isinstance(elec_energy_tol, bool):
        return ValueWithUnit(
            min(float(elec_energy_tol), float(QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL.value)),
            "eV",
        )
    return QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL


def _refuse_phonon_method_override(template: ParamTemplate | None) -> None:
    """Raise when ``template`` sets ``PHONON_METHOD``, which the workflow writes itself."""

    override_method = _param_keyword_value(template, "phonon_method")
    if override_method is not None:
        raise ValueError(
            f"Quantum Corrections writes PHONON_METHOD : {QUANTUM_PHONON_METHOD} itself; "
            f"remove the PHONON_METHOD override ({override_method!r})."
        )


def _param_keyword_value(template: ParamTemplate | None, name: str) -> object | None:
    if template is None:
        return None
    target = canonical_keyword_name("param", name)
    for assignment in template.keywords:
        if canonical_keyword_name("param", assignment.name) == target:
            return assignment.value
    return None
