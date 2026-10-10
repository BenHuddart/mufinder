"""Quantum-corrections CASTEP workflow expansion."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Sequence
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

QuantumPhononMethod = Literal["finite_displacement", "dfpt", "linear_response"]
PseudopotentialFamily = Literal["unknown", "norm_conserving", "ultrasoft"]

QUANTUM_SINGLEPOINT_CALC_TYPE = "castep_quantum_singlepoint"
QUANTUM_PHONON_CALC_TYPE = "castep_quantum_phonon"
QUANTUM_PHONON_SET_CALC_TYPE = "castep_quantum_phonon_set"
QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL = ValueWithUnit(1e-9, "eV")

_METHOD_TO_CASTEP = {
    "finite_displacement": "FINITEDISPLACEMENT",
    "dfpt": "DFPT",
    "linear_response": "LINEARRESPONSE",
}


def _seed_muonium_spin(structure: StructureSpec, muon_species: str) -> StructureSpec:
    """Seed +1 on the muon so a muonium phonon/singlepoint stays open-shell.

    Mirrors the muon-relaxation adapter: neutral muonium binds one unpaired electron
    (S = 1/2, ~1 muB). Host moments are preserved, and this is idempotent if the relaxed
    source structure already carried the seed.
    """

    positions = structure.positions_frac.positions
    muon_indices = [index for index, position in enumerate(positions) if position.species == muon_species]
    # Fail loudly rather than emit a spin-polarised muonium job with no seeded moment: a
    # zero net moment on an odd-electron cell silently converges to the wrong (closed-shell)
    # solution. This catches a species-label mismatch between the structure and muon_species.
    if not muon_indices:
        raise ValueError(
            f"Muonium structure has no atom matching muon species {muon_species!r}; "
            f"cannot seed the +1 muon spin. Present species: "
            f"{sorted({position.species for position in positions})}."
        )
    spins = seed_muon_spins(structure.atomic_spins, atom_count=len(positions), muon_indices=muon_indices)
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
    method: QuantumPhononMethod = "finite_displacement"
    muon_species: str = "H:mu"
    pseudopotential_family: PseudopotentialFamily = "unknown"
    num_farms: int = 1
    cell_overrides: CellTemplate | None = None
    param_overrides: ParamTemplate | None = None


@dataclass(frozen=True, slots=True)
class QuantumSourceEligibility:
    """Eligibility verdict for a possible quantum-corrections source."""

    eligible: bool
    reason: str = ""


def method_to_castep(method: QuantumPhononMethod) -> str:
    """Return the CASTEP ``PHONON_METHOD`` keyword value."""

    try:
        return _METHOD_TO_CASTEP[method]
    except KeyError as exc:
        raise ValueError(f"Unsupported phonon method: {method!r}.") from exc


def phonon_method_compatibility(
    method: QuantumPhononMethod,
    *,
    pseudopotential_family: PseudopotentialFamily = "unknown",
    uses_hubbard_u: bool = False,
    uses_exact_exchange: bool = False,
    uses_spin_orbit: bool = False,
) -> tuple[bool, str | None]:
    """Return whether a CASTEP phonon method is compatible with known settings."""

    if method == "finite_displacement":
        return True, None
    if pseudopotential_family == "ultrasoft":
        return False, "CASTEP DFPT/linear-response phonons require norm-conserving pseudopotentials."
    if uses_hubbard_u:
        return False, "CASTEP DFPT/linear-response phonons are not implemented with Hubbard U."
    if uses_exact_exchange:
        return False, "CASTEP DFPT/linear-response phonons are not implemented with exact exchange."
    if uses_spin_orbit:
        return False, "CASTEP DFPT/linear-response phonons are not implemented with spin-orbit coupling."
    if pseudopotential_family == "unknown":
        return True, "Pseudopotential family is unknown; verify norm-conserving pseudopotentials before running DFPT."
    return True, None


def quantum_source_eligibility(
    *,
    site: MuonSite,
    run: CalculationRun | None,
    job: Job | None,
    relaxed_structure: Structure | None,
) -> QuantumSourceEligibility:
    """Validate that a site came from a completed structural muon relaxation."""

    if site.status != "relaxed":
        return QuantumSourceEligibility(False, "Muon site has not been structurally relaxed.")
    if site.relaxation_run_id is None:
        return QuantumSourceEligibility(False, "Muon site has no structural-relaxation run provenance.")
    if run is None or run.id != site.relaxation_run_id:
        return QuantumSourceEligibility(False, "The recorded relaxation run is missing.")
    if run.status != JobStatus.COMPLETED:
        return QuantumSourceEligibility(False, "The structural-relaxation run is not complete.")
    if job is None or job.workflow_stage != WorkflowStage.MUON_RELAXATION:
        return QuantumSourceEligibility(False, "The source job is not a MuFinder muon structural relaxation.")
    if job.status != JobStatus.COMPLETED:
        return QuantumSourceEligibility(False, "The source muon relaxation job is not complete.")
    if relaxed_structure is None:
        return QuantumSourceEligibility(False, "The source relaxation has no parsed relaxed structure.")
    if relaxed_structure.origin != "muon_relaxation":
        return QuantumSourceEligibility(False, "The source structure was not produced by muon structural relaxation.")
    if relaxed_structure.properties.get("source_workflow_stage") != WorkflowStage.MUON_RELAXATION.value:
        return QuantumSourceEligibility(False, "The source structure does not carry muon-relaxation provenance.")
    if "H:mu" not in relaxed_structure.species:
        return QuantumSourceEligibility(False, "The relaxed structure does not contain an H:mu atom.")
    return QuantumSourceEligibility(True)


def expand_quantum_singlepoint_jobs(
    *,
    jobs: Sequence[QuantumPhononJobConfig],
    name_prefix: str,
    global_defaults: InputTemplate | None = None,
    workflow_defaults: InputTemplate | None = None,
    structure_defaults: InputTemplate | None = None,
    user_overrides: InputTemplate | None = None,
    tags: tuple[str, ...] = (),
) -> list[JobSpec]:
    """Expand quantum-corrections singlepoint configurations into CASTEP jobs."""

    return _expand_quantum_jobs(
        "singlepoint",
        jobs=jobs,
        name_prefix=name_prefix,
        global_defaults=global_defaults,
        workflow_defaults=workflow_defaults,
        structure_defaults=structure_defaults,
        user_overrides=user_overrides,
        tags=tags,
    )


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
    """Expand quantum-corrections phonon configurations into CASTEP jobs."""

    return _expand_quantum_jobs(
        "phonon",
        jobs=jobs,
        name_prefix=name_prefix,
        global_defaults=global_defaults,
        workflow_defaults=workflow_defaults,
        structure_defaults=structure_defaults,
        user_overrides=user_overrides,
        tags=tags,
    )


def _expand_quantum_jobs(
    stage: Literal["singlepoint", "phonon"],
    *,
    jobs: Sequence[QuantumPhononJobConfig],
    name_prefix: str,
    global_defaults: InputTemplate | None,
    workflow_defaults: InputTemplate | None,
    structure_defaults: InputTemplate | None,
    user_overrides: InputTemplate | None,
    tags: tuple[str, ...],
) -> list[JobSpec]:
    """Expand one quantum-corrections stage: the stage-1 singlepoint or the stage-2 phonon.

    The stages share labels, muonium spin seeding, override composition and the
    source provenance. The phonon stage also checks the phonon method against
    the pseudopotential family and records the method, its warning and the farm
    count. The singlepoint cell base carries the muon species blocks only: the
    phonon-specific scalars belong to stage 2 (see ``_muon_species_blocks`` for
    why the mass has to be written in stage 1 as well).
    """

    phonon = stage == "phonon"
    require_jobs(jobs, f"quantum {stage} job")

    resolved: list[JobSpec] = []
    for config in jobs:
        label = config.source.label.strip()
        if not label:
            raise ValueError(f"Quantum {stage} source labels must not be empty.")
        method_warning: str | None = None
        if phonon:
            compatible, method_warning = phonon_method_compatibility(
                config.method,
                pseudopotential_family=config.pseudopotential_family,
            )
            if not compatible:
                raise ValueError(method_warning or "Incompatible CASTEP phonon method.")

        structure = config.source.structure
        if is_muonium(config.source.charge_state):
            structure = _seed_muonium_spin(structure, config.muon_species)
        base = (
            InputTemplate(cell=_cell_base_template(config), param=_param_base_template(config))
            if phonon
            else InputTemplate(
                cell=_singlepoint_cell_base_template(config),
                param=_singlepoint_param_base_template(config),
            )
        )
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
            "calc_type": QUANTUM_PHONON_CALC_TYPE if phonon else QUANTUM_SINGLEPOINT_CALC_TYPE,
            "label": label,
        }
        if phonon:
            provenance["phonon_method"] = config.method
            provenance["phonon_kpoint_mp_grid"] = (1, 1, 1)
        provenance["muon_species"] = config.muon_species
        provenance["muon_mass_amu"] = MUON_MASS_AMU
        provenance["muon_gamma_rad_s_T"] = MUON_GAMMA_RAD_S_T
        if phonon:
            provenance["num_farms"] = max(1, int(config.num_farms))
        provenance["source_site_id"] = str(config.source.source_site_id)
        provenance["source_relaxation_run_id"] = str(config.source.source_relaxation_run_id)
        provenance["source_relaxation_job_id"] = (
            str(config.source.source_relaxation_job_id)
            if config.source.source_relaxation_job_id is not None
            else None
        )
        provenance["charge_state"] = config.source.charge_state
        if phonon:
            provenance["method_warning"] = method_warning
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
                tags=("quantum_corrections", stage, *tags),
                provenance=provenance,
            )
        )
    return resolved


def _muon_species_blocks(config: QuantumPhononJobConfig) -> tuple[BlockSection, ...]:
    """Return the muon ``SPECIES_MASS`` / ``SPECIES_GAMMA`` blocks.

    **Both** stages must emit these, not just the phonon stage. Stage 2 runs as a
    ``CONTINUATION``, and CASTEP restores the model — species masses included —
    from the stage-1 ``.check`` file. A mass written only into the stage-2 cell
    therefore arrives too late: the phonons run with the default hydrogen mass for
    ``H:mu``, and because omega scales as 1/sqrt(mass) the frequencies and the muon
    ZPE come out roughly a factor of three too low. Emitting the blocks in the
    stage-1 singlepoint is what puts the muon mass into the checkpoint the
    continuation reads. Masses do not affect a singlepoint SCF energy, so this is
    inert for the stage-1 result itself.
    """

    return (
        BlockSection("species_mass", ((config.muon_species, MUON_MASS_AMU),), unit="AMU"),
        BlockSection("species_gamma", ((config.muon_species, MUON_GAMMA_RAD_S_T),), unit="radsectesla"),
    )


def _singlepoint_cell_base_template(config: QuantumPhononJobConfig) -> CellTemplate:
    return CellTemplate(extra_blocks=_muon_species_blocks(config))


def _cell_base_template(config: QuantumPhononJobConfig) -> CellTemplate:
    return CellTemplate(
        extra_scalars=(KeywordAssignment("phonon_kpoint_mp_grid", (1, 1, 1)),),
        extra_blocks=_muon_species_blocks(config),
    )


def _param_base_template(config: QuantumPhononJobConfig) -> ParamTemplate:
    keywords = [
        KeywordAssignment("task", "PHONON"),
        KeywordAssignment("phonon_method", method_to_castep(config.method)),
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


def _singlepoint_param_base_template(config: QuantumPhononJobConfig) -> ParamTemplate:
    keywords = [KeywordAssignment("task", "SINGLEPOINT")]
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
                min(float(elec_energy_tol.value), float(QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL.value)),
                "eV",
            )
    if isinstance(elec_energy_tol, (int, float)) and not isinstance(elec_energy_tol, bool):
        return ValueWithUnit(
            min(float(elec_energy_tol), float(QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL.value)),
            "eV",
        )
    return QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL


def _param_keyword_value(template: ParamTemplate | None, name: str) -> object | None:
    if template is None:
        return None
    target = canonical_keyword_name("param", name)
    for assignment in template.keywords:
        if canonical_keyword_name("param", assignment.name) == target:
            return assignment.value
    return None
