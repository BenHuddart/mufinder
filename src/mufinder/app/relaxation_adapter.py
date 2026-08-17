"""Adapter from pristine-relaxation UI state into workflow job expansion."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np

from mufinder.app.adapter_common import (
    STRUCTURE_CONTROLLED_CELL_KEYWORDS,
    AdapterExpansion,
    build_input_template,
    rescale_kpoint_grid,
    row_values_has_keyword,
    template_has_explicit_kpoints,
)
from mufinder.app.relaxation_types import (
    MuonRelaxationConfig,
    MuonRelaxationRow,
    PristineRelaxationConfig,
    PristineRelaxationRow,
    SupercellMatrix,
)
from mufinder.castep_io.types import (
    BlockSection,
    KeywordAssignment,
)
from mufinder.domain.models import Structure
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.charge_states import (
    MUON_RELAXATION_CALC_TYPES,
    MUON_RELAXATION_CHARGES,
    MUON_RELAXATION_JOB_PREFIXES,
    MUONIUM_CHARGE_STATE,
    MUONIUM_MUON_SPIN,
    POSITIVE_MUON_CHARGE_STATE,
    _charge_state_key,
    canonical_charge_state,
    is_muon_relaxation_calc_type,
    muon_charge_for_state,
    muon_charge_state_label,
    muon_charge_state_prefix,
    muon_relaxation_calc_type,
    seed_muon_spins,
)

# These names moved to ``mufinder.domain.charge_states`` (durable persisted
# vocabulary, not presentation detail) and are re-exported here because app-layer
# call sites and tests import them from this module.
__all__ = [
    "MUON_RELAXATION_CALC_TYPES",
    "MUON_RELAXATION_CHARGES",
    "MUON_RELAXATION_JOB_PREFIXES",
    "MUONIUM_CHARGE_STATE",
    "POSITIVE_MUON_CHARGE_STATE",
    "RelaxationExpansion",
    "expand_muon_relaxation_jobs",
    "expand_pristine_relaxation_jobs",
    "is_muon_relaxation_calc_type",
    "muon_charge_for_state",
    "muon_charge_state_label",
    "muon_charge_state_prefix",
    "muon_relaxation_calc_type",
    "suggested_grid_for_row",
]
from mufinder.science.kpoint_grids import (
    equivalent_grid_for_supercell,
    supercell_lattice_vectors,
)
from mufinder.science.supercell import build_supercell_structure
from mufinder.workflow import StructureBridgeError, structure_to_spec
from mufinder.workflow.naming import job_name
from mufinder.domain.constants import MUON_MASS_AMU
from mufinder.workflow.merge import merge_templates
from mufinder.workflow.relaxation import RelaxationJobConfig, expand_relaxation_jobs
from mufinder.workflow.types import CellTemplate, InputTemplate, JobSpec, ParamTemplate


# ``RelaxationExpansion`` is a thin backward-compatible alias for the generic
# :class:`~mufinder.app.adapter_common.AdapterExpansion`. The former per-workflow
# ``kpoint_messages`` field is exposed as a property on ``AdapterExpansion``.
RelaxationExpansion = AdapterExpansion


def _template_from_values(values: Mapping[str, object]) -> InputTemplate:
    forbidden: dict[str, str] = {
        name: f"{name} is controlled by the selected structure."
        for name in STRUCTURE_CONTROLLED_CELL_KEYWORDS
    }
    forbidden["fix_all_cell"] = "fix_all_cell is controlled by the row controls."

    return build_input_template(
        values,
        context_label="pristine-relaxation settings",
        forbidden=forbidden,
        kpoints_list_reset=None,
    )


def _is_identity_supercell(matrix: SupercellMatrix | None) -> bool:
    return matrix is None or matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _matrix_token(matrix: SupercellMatrix | None) -> str:
    if _is_identity_supercell(matrix):
        return "unit cell"
    if matrix is None:
        return "unit cell"
    is_diagonal = (
        matrix[0][1] == 0
        and matrix[0][2] == 0
        and matrix[1][0] == 0
        and matrix[1][2] == 0
        and matrix[2][0] == 0
        and matrix[2][1] == 0
    )
    if is_diagonal:
        return f"{matrix[0][0]}x{matrix[1][1]}x{matrix[2][2]}"
    return (
        f"[{matrix[0][0]} {matrix[0][1]} {matrix[0][2]}; "
        f"{matrix[1][0]} {matrix[1][1]} {matrix[1][2]}; "
        f"{matrix[2][0]} {matrix[2][1]} {matrix[2][2]}]"
    )


def _structure_for_row(host_structure: Structure, row: PristineRelaxationRow) -> Structure:
    if _is_identity_supercell(row.supercell_matrix):
        return host_structure

    matrix = row.supercell_matrix
    if matrix is None:
        return host_structure

    structure, _spins = build_supercell_structure(
        host_structure,
        [list(line) for line in matrix],
        atomic_spins=host_structure.properties.get("atomic_spins"),
        propagation_vector=host_structure.properties.get("propagation_vector"),
    )
    structure.label = f"{host_structure.label}_{_matrix_token(matrix)}"
    return structure








def _muon_supercell_fractional_position(
    host_fractional_position: tuple[float, float, float],
    matrix: SupercellMatrix | None,
) -> tuple[float, float, float]:
    wrapped = np.mod(np.asarray(host_fractional_position, dtype=float), 1.0)
    if _is_identity_supercell(matrix):
        return tuple(float(component) for component in wrapped)  # type: ignore[return-value]
    parsed = np.asarray(matrix, dtype=float)
    supercell_fractional = wrapped @ np.linalg.inv(parsed)
    return tuple(float(component % 1.0) for component in supercell_fractional)  # type: ignore[return-value]


def _structure_for_muon_row(host_structure: Structure, row: MuonRelaxationRow) -> Structure:
    base = _structure_for_row(
        host_structure,
        PristineRelaxationRow(label=row.label, supercell_matrix=row.supercell_matrix),
    )
    muon_position = _muon_supercell_fractional_position(row.initial_fractional_position, row.supercell_matrix)
    properties = dict(base.properties)
    # Neutral muonium is open-shell: mu+ binds one electron (S = 1/2, ~1 muB). Seed +1 on the
    # muon so the spin-polarised SCF localises the unpaired electron there; the diamagnetic mu+
    # has no bound electron and stays 0.
    is_muonium = _charge_state_key(row.charge_state) == MUONIUM_CHARGE_STATE
    muon_spin = MUONIUM_MUON_SPIN if is_muonium else 0.0
    host_spins = list(properties.get("atomic_spins", host_structure.properties.get("atomic_spins", [])))
    # The muon is appended as the last atom (see below), so seed it there. Only materialise the
    # array when it carries information: a magnetic host (host_spins) or a muonium seed. A
    # diamagnetic muon on a non-magnetic host needs no spin array at all.
    if host_spins or is_muonium:
        atom_count = len(base.fractional_positions) + 1
        properties["atomic_spins"] = seed_muon_spins(
            host_spins or None,
            atom_count=atom_count,
            muon_indices=(atom_count - 1,),
            muon_spin=muon_spin,
        )

    return Structure(
        label=f"{base.label}_{muon_charge_state_prefix(row.charge_state)}_{row.label}",
        origin=base.origin,
        lattice_vectors=[list(vector) for vector in base.lattice_vectors],
        species=list(base.species) + ["H:mu"],
        fractional_positions=[list(position) for position in base.fractional_positions] + [list(muon_position)],
        source_file=base.source_file,
        parent_structure_id=base.parent_structure_id,
        properties={
            **properties,
            "muon_site_id": str(row.site_id),
            "muon_charge_state": _charge_state_key(row.charge_state),
            "muon_initial_fractional_position_host": [float(value) for value in row.initial_fractional_position],
            "muon_initial_fractional_position_supercell": [float(value) for value in muon_position],
        },
    )


def expand_pristine_relaxation_jobs(
    *,
    host_structure: Structure,
    config: PristineRelaxationConfig,
    project_cutoff_energy: float | None,
    project_kpoint_grid: tuple[int, int, int] | None,
    project_kpoint_reference_lattice: list[list[float]] | None,
) -> RelaxationExpansion:
    """Expand pristine-relaxation UI config to workflow job specs."""

    if not config.rows:
        raise ValueError("At least one pristine-relaxation row must be configured.")

    workflow_defaults_values: dict[str, object] = {
        "task": "GEOMETRYOPTIMIZATION",
        "geom_max_iter": 300,
        "symmetry_generate": True,
        "write_cell_structure": True,
    }
    if project_cutoff_energy is not None:
        workflow_defaults_values["cut_off_energy"] = float(project_cutoff_energy)
    workflow_defaults = _template_from_values(workflow_defaults_values)

    reference_lattice = (
        project_kpoint_reference_lattice
        if isinstance(project_kpoint_reference_lattice, list) and len(project_kpoint_reference_lattice) == 3
        else host_structure.lattice_vectors
    )

    jobs: list[RelaxationJobConfig] = []
    messages: list[str] = []
    seen_labels: set[str] = set()
    for row in config.rows:
        label = row.label.strip()
        if not label:
            raise ValueError("Pristine-relaxation row labels must not be empty.")
        if label in seen_labels:
            raise ValueError(f"Duplicate pristine-relaxation row label '{label}'.")
        seen_labels.add(label)

        row_structure = _structure_for_row(host_structure, row)
        try:
            structure_spec = structure_to_spec(row_structure)
        except StructureBridgeError as exc:
            raise ValueError(f"Could not convert row '{label}' structure: {exc}") from exc

        # WRITE_CELL_STRUCTURE is controlled by the pristine-relaxation workflow
        # and must stay enabled to guarantee -out.cell output files.
        row_values = dict(row.parameter_values)
        row_values.pop("write_cell_structure", None)
        row_template = _template_from_values(row_values)

        kpoint_grid = None
        if not template_has_explicit_kpoints(row_template):
            kpoint_grid, message = rescale_kpoint_grid(
                project_kpoint_grid=project_kpoint_grid,
                reference_lattice=reference_lattice,
                row_lattice=row_structure.lattice_vectors,
                is_identity_supercell=_is_identity_supercell(row.supercell_matrix),
                has_supercell=row.supercell_matrix is not None,
                target_description=f"{_matrix_token(row.supercell_matrix)} (equivalent density).",
            )
            if message is not None:
                messages.append(message)

        jobs.append(
            RelaxationJobConfig(
                label=label,
                structure=structure_spec,
                fix_all_cell=row.fix_all_cell,
                kpoint_grid=kpoint_grid,
                cell_overrides=row_template.cell,
                param_overrides=row_template.param,
                provenance={
                    "supercell_matrix": (
                        [list(line) for line in row.supercell_matrix]
                        if row.supercell_matrix is not None
                        else None
                    )
                },
            )
        )

    expanded = expand_relaxation_jobs(
        jobs=jobs,
        name_prefix=config.name_prefix,
        workflow_defaults=workflow_defaults,
        tags=("pristine_relaxation",),
    )
    return AdapterExpansion(jobs=tuple(expanded), messages=tuple(messages))


def expand_muon_relaxation_jobs(
    *,
    host_structure: Structure,
    config: MuonRelaxationConfig,
    project_cutoff_energy: float | None,
    project_kpoint_grid: tuple[int, int, int] | None,
    project_kpoint_reference_lattice: list[list[float]] | None,
) -> RelaxationExpansion:
    """Expand staged muon sites to fixed-cell CASTEP geometry-optimization jobs."""

    if not config.rows:
        raise ValueError("At least one muon-relaxation row must be staged.")

    workflow_defaults_values: dict[str, object] = {
        "task": "GEOMETRYOPTIMIZATION",
        "geom_max_iter": 300,
        "write_cell_structure": True,
    }
    if project_cutoff_energy is not None:
        workflow_defaults_values["cut_off_energy"] = float(project_cutoff_energy)
    species_mass_block = BlockSection("species_mass", (("H:mu", MUON_MASS_AMU),), unit="AMU")
    workflow_defaults = merge_templates(
        _template_from_values(workflow_defaults_values),
        InputTemplate(cell=CellTemplate(extra_blocks=(species_mass_block,))),
    )

    reference_lattice = (
        project_kpoint_reference_lattice
        if isinstance(project_kpoint_reference_lattice, list) and len(project_kpoint_reference_lattice) == 3
        else host_structure.lattice_vectors
    )

    messages: list[str] = []
    expanded_jobs: list[JobSpec] = []
    seen_names: set[str] = set()
    for row in config.rows:
        label = row.label.strip()
        if not label:
            raise ValueError("Muon-relaxation row labels must not be empty.")
        charge_state = _charge_state_key(row.charge_state)
        charge_prefix = muon_charge_state_prefix(charge_state)
        seed_name = job_name(charge_prefix, label)
        if seed_name in seen_names:
            raise ValueError(f"Duplicate muon-relaxation job seed name '{seed_name}'.")
        seen_names.add(seed_name)

        row_structure = _structure_for_muon_row(host_structure, row)
        try:
            structure_spec = structure_to_spec(row_structure)
        except StructureBridgeError as exc:
            raise ValueError(f"Could not convert row '{label}' structure: {exc}") from exc

        row_values = dict(row.parameter_values)
        for locked in ("task", "write_cell_structure", "charge", "fix_all_cell"):
            row_values.pop(locked, None)
        row_template = _template_from_values(row_values)

        kpoint_grid = None
        if not template_has_explicit_kpoints(row_template):
            kpoint_grid, message = rescale_kpoint_grid(
                project_kpoint_grid=project_kpoint_grid,
                reference_lattice=reference_lattice,
                row_lattice=row_structure.lattice_vectors,
                is_identity_supercell=_is_identity_supercell(row.supercell_matrix),
                has_supercell=row.supercell_matrix is not None,
                target_description=f"{_matrix_token(row.supercell_matrix)} (equivalent density).",
            )
            if message is not None:
                messages.append(message)

        param = row_template.param
        forced_keywords = [KeywordAssignment("charge", muon_charge_for_state(charge_state))]
        # Muonium is open-shell, so default to a spin-polarised SCF. Left as a default (not locked
        # like charge) so the user can still override it from the param dialog.
        if charge_state == MUONIUM_CHARGE_STATE and not row_values_has_keyword(row_values, "spin_polarized"):
            forced_keywords.append(KeywordAssignment("spin_polarized", True))
        param_overrides = ParamTemplate(
            keywords=(tuple(param.keywords) if param is not None else ()) + tuple(forced_keywords),
            extra_blocks=tuple(param.extra_blocks) if param is not None else (),
        )

        row_jobs = expand_relaxation_jobs(
            jobs=(
                RelaxationJobConfig(
                    label=label,
                    structure=structure_spec,
                    fix_all_cell=True,
                    kpoint_grid=kpoint_grid,
                    cell_overrides=row_template.cell,
                    param_overrides=param_overrides,
                    provenance={
                        "site_id": str(row.site_id),
                        "charge_state": charge_state,
                        "calc_type": muon_relaxation_calc_type(charge_state),
                        "supercell_matrix": (
                            [list(line) for line in row.supercell_matrix]
                            if row.supercell_matrix is not None
                            else None
                        ),
                        "muon_initial_fractional_position_host": [
                            float(value) for value in row.initial_fractional_position
                        ],
                        "muon_initial_fractional_position_supercell": list(
                            _muon_supercell_fractional_position(
                                row.initial_fractional_position,
                                row.supercell_matrix,
                            )
                        ),
                    },
                ),
            ),
            name_prefix=charge_prefix,
            workflow_defaults=workflow_defaults,
            tags=("muon_relaxation", charge_prefix),
        )
        expanded_jobs.extend(
            replace(
                spec,
                workflow_stage=WorkflowStage.MUON_RELAXATION,
                provenance={**spec.provenance, "workflow": "muon_relaxation"},
            )
            for spec in row_jobs
        )

    return AdapterExpansion(jobs=tuple(expanded_jobs), messages=tuple(messages))


def suggested_grid_for_row(
    *,
    host_lattice: list[list[float]],
    row_matrix: SupercellMatrix | None,
    reference_lattice: list[list[float]] | None,
    reference_grid: tuple[int, int, int] | None,
) -> tuple[int, int, int] | None:
    """Return the suggested k-point grid for one row based on supercell choice."""

    if reference_grid is None:
        return None

    if _is_identity_supercell(row_matrix):
        return reference_grid

    row_lattice = [
        [float(value) for value in vector]
        for vector in supercell_lattice_vectors(
            host_lattice,
            row_matrix or ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )
    ]
    base_lattice = reference_lattice if reference_lattice is not None else host_lattice
    return equivalent_grid_for_supercell(base_lattice, reference_grid, row_lattice)
