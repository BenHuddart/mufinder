"""Quantum-corrections CASTEP workflow expansion."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Sequence
from uuid import UUID

from mufinder.castep_io.schema import canonical_keyword_name
from mufinder.castep_io.types import BlockSection, KeywordAssignment, ValueWithUnit
from mufinder.domain.charge_states import is_muonium
from mufinder.domain.constants import MUON_GAMMA_RAD_S_T, MUON_MASS_AMU
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Structure

from .merge import merge_templates, resolve_job
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
    # Fail loudly rather than emit a spin-polarised muonium job with no seeded moment: a
    # zero net moment on an odd-electron cell silently converges to the wrong (closed-shell)
    # solution. This catches a species-label mismatch between the structure and muon_species.
    if not any(position.species == muon_species for position in positions):
        raise ValueError(
            f"Muonium structure has no atom matching muon species {muon_species!r}; "
            f"cannot seed the +1 muon spin. Present species: "
            f"{sorted({position.species for position in positions})}."
        )
    existing = (
        list(structure.atomic_spins)
        if structure.atomic_spins is not None
        else [0.0] * len(positions)
    )
    spins = tuple(
        1.0 if position.species == muon_species else existing[index]
        for index, position in enumerate(positions)
    )
    return replace(structure, atomic_spins=spins)


@dataclass(frozen=True, slots=True)
class QuantumPhononSource:
    """One eligible relaxed muon site for a phonon calculation."""

    label: str
    structure: StructureSpec
    source_site_id: UUID
    source_relaxation_run_id: UUID
    source_relaxation_job_id: UUID | None = None
    charge_state: str | None = None
    relative_energy_eV: float | None = None
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
    user_overrides: InputTemplate | None = None,
    tags: tuple[str, ...] = (),
) -> list[JobSpec]:
    """Expand quantum-corrections singlepoint configurations into CASTEP jobs."""

    if not jobs:
        raise ValueError("At least one quantum singlepoint job must be configured.")

    resolved: list[JobSpec] = []
    for config in jobs:
        label = config.source.label.strip()
        if not label:
            raise ValueError("Quantum singlepoint source labels must not be empty.")

        structure = config.source.structure
        if is_muonium(config.source.charge_state):
            structure = _seed_muonium_spin(structure, config.muon_species)
        patch = JobPatch(
            label_suffix=label,
            structure=structure,
            cell=config.cell_overrides,
            param=_singlepoint_param_template(config),
        )
        resolved.append(
            resolve_job(
                name=job_name(name_prefix, label),
                base_structure=structure,
                workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
                global_defaults=global_defaults,
                workflow_defaults=workflow_defaults,
                patch=patch,
                user_overrides=user_overrides,
                tags=("quantum_corrections", "singlepoint", *tags),
                provenance={
                    "workflow": "quantum_corrections",
                    "calc_type": QUANTUM_SINGLEPOINT_CALC_TYPE,
                    "label": label,
                    "source_site_id": str(config.source.source_site_id),
                    "source_relaxation_run_id": str(config.source.source_relaxation_run_id),
                    "source_relaxation_job_id": (
                        str(config.source.source_relaxation_job_id)
                        if config.source.source_relaxation_job_id is not None
                        else None
                    ),
                    "charge_state": config.source.charge_state,
                    **dict(config.source.provenance),
                },
            )
        )
    return resolved


def expand_quantum_phonon_jobs(
    *,
    jobs: Sequence[QuantumPhononJobConfig],
    name_prefix: str,
    global_defaults: InputTemplate | None = None,
    workflow_defaults: InputTemplate | None = None,
    user_overrides: InputTemplate | None = None,
    tags: tuple[str, ...] = (),
) -> list[JobSpec]:
    """Expand quantum-corrections phonon configurations into CASTEP jobs."""

    if not jobs:
        raise ValueError("At least one quantum phonon job must be configured.")

    resolved: list[JobSpec] = []
    for config in jobs:
        label = config.source.label.strip()
        if not label:
            raise ValueError("Quantum phonon source labels must not be empty.")
        compatible, message = phonon_method_compatibility(
            config.method,
            pseudopotential_family=config.pseudopotential_family,
        )
        if not compatible:
            raise ValueError(message or "Incompatible CASTEP phonon method.")

        structure = config.source.structure
        if is_muonium(config.source.charge_state):
            structure = _seed_muonium_spin(structure, config.muon_species)
        patch = JobPatch(
            label_suffix=label,
            structure=structure,
            cell=_cell_template(config),
            param=_param_template(config),
        )
        resolved.append(
            resolve_job(
                name=job_name(name_prefix, label),
                base_structure=structure,
                workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
                global_defaults=global_defaults,
                workflow_defaults=workflow_defaults,
                patch=patch,
                user_overrides=user_overrides,
                tags=("quantum_corrections", "phonon", *tags),
                provenance={
                    "workflow": "quantum_corrections",
                    "calc_type": QUANTUM_PHONON_CALC_TYPE,
                    "label": label,
                    "phonon_method": config.method,
                    "phonon_kpoint_mp_grid": (1, 1, 1),
                    "muon_species": config.muon_species,
                    "muon_mass_amu": MUON_MASS_AMU,
                    "muon_gamma_rad_s_T": MUON_GAMMA_RAD_S_T,
                    "num_farms": max(1, int(config.num_farms)),
                    "source_site_id": str(config.source.source_site_id),
                    "source_relaxation_run_id": str(config.source.source_relaxation_run_id),
                    "source_relaxation_job_id": (
                        str(config.source.source_relaxation_job_id)
                        if config.source.source_relaxation_job_id is not None
                        else None
                    ),
                    "charge_state": config.source.charge_state,
                    "method_warning": message,
                    **dict(config.source.provenance),
                },
            )
        )
    return resolved


def _cell_template(config: QuantumPhononJobConfig) -> CellTemplate:
    base = CellTemplate(
        extra_scalars=(KeywordAssignment("phonon_kpoint_mp_grid", (1, 1, 1)),),
        extra_blocks=(
            BlockSection("species_mass", ((config.muon_species, MUON_MASS_AMU),), unit="AMU"),
            BlockSection("species_gamma", ((config.muon_species, MUON_GAMMA_RAD_S_T),), unit="radsectesla"),
        ),
    )
    if config.cell_overrides is None:
        return base
    merged = merge_templates(InputTemplate(cell=base), InputTemplate(cell=config.cell_overrides))
    return merged.cell or base


def _param_template(config: QuantumPhononJobConfig) -> ParamTemplate:
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
    base = ParamTemplate(keywords=tuple(keywords))
    if config.param_overrides is None:
        return base
    merged = merge_templates(InputTemplate(param=base), InputTemplate(param=config.param_overrides))
    return merged.param or base


def _singlepoint_param_template(config: QuantumPhononJobConfig) -> ParamTemplate:
    keywords = [KeywordAssignment("task", "SINGLEPOINT")]
    if config.source.charge_state:
        charge = 0 if is_muonium(config.source.charge_state) else 1
        keywords.append(KeywordAssignment("charge", charge))
    if is_muonium(config.source.charge_state):
        # Muonium is open-shell: default to spin-polarised (overridable via param_overrides).
        keywords.append(KeywordAssignment("spin_polarized", True))
    base = ParamTemplate(keywords=tuple(keywords))
    if config.param_overrides is None:
        return base
    merged = merge_templates(InputTemplate(param=base), InputTemplate(param=config.param_overrides))
    return merged.param or base


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
