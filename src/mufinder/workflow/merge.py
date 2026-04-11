"""Template merging and job resolution helpers."""

from __future__ import annotations

from typing import Any

from mufinder.castep_io.schema import canonical_keyword_name, keyword_sort_key
from mufinder.castep_io.types import CellFile, KeywordAssignment, ParamFile
from mufinder.domain.enums import WorkflowStage

from .types import CellTemplate, InputTemplate, JobPatch, JobSpec, ParamTemplate, StructureSpec

MERGE_PRECEDENCE_ORDER: tuple[str, ...] = (
    "global_defaults",
    "workflow_defaults",
    "structure_defaults",
    "sweep_overrides",
    "user_overrides",
)


def _merge_assignments(
    file_kind: str,
    current: tuple[KeywordAssignment, ...],
    incoming: tuple[KeywordAssignment, ...],
) -> tuple[KeywordAssignment, ...]:
    merged: dict[str, KeywordAssignment] = {
        canonical_keyword_name(file_kind, assignment.name): assignment
        for assignment in current
    }
    for assignment in incoming:
        merged[canonical_keyword_name(file_kind, assignment.name)] = assignment
    ordered_keys = sorted(merged, key=lambda name: keyword_sort_key(file_kind, name))
    return tuple(merged[key] for key in ordered_keys)


def _merge_blocks(file_kind: str, current: tuple[Any, ...], incoming: tuple[Any, ...]) -> tuple[Any, ...]:
    merged: dict[str, Any] = {
        canonical_keyword_name(file_kind, block.name): block
        for block in current
    }
    for block in incoming:
        merged[canonical_keyword_name(file_kind, block.name)] = block
    ordered_keys = sorted(merged, key=lambda name: keyword_sort_key(file_kind, name))
    return tuple(merged[key] for key in ordered_keys)


def _param_spin_polarized(keywords: tuple[KeywordAssignment, ...]) -> bool:
    for keyword in keywords:
        if canonical_keyword_name("param", keyword.name) != "spin_polarized":
            continue
        return bool(keyword.value)
    return False


def merge_templates(*layers: InputTemplate | None) -> InputTemplate:
    """Merge template layers from low precedence to high precedence."""

    has_cell_layer = False
    lattice_cart = None
    lattice_abc = None
    positions_frac = None
    positions_abs = None
    atomic_spins = None
    species_pot = None
    kpoint_mp_grid = None
    kpoint_mp_spacing = None
    kpoints_list = None
    symmetry_generate = None
    extra_scalars: tuple[KeywordAssignment, ...] = ()
    extra_cell_blocks: tuple[Any, ...] = ()

    has_param_layer = False
    keywords: tuple[KeywordAssignment, ...] = ()
    extra_param_blocks: tuple[Any, ...] = ()

    for layer in layers:
        if layer is None:
            continue

        if layer.cell is not None:
            has_cell_layer = True
            cell = layer.cell

            if cell.lattice_cart is not None:
                lattice_cart = cell.lattice_cart
                lattice_abc = None
            if cell.lattice_abc is not None:
                lattice_abc = cell.lattice_abc
                lattice_cart = None

            if cell.positions_frac is not None:
                positions_frac = cell.positions_frac
                positions_abs = None
            if cell.positions_abs is not None:
                positions_abs = cell.positions_abs
                positions_frac = None

            if cell.atomic_spins is not None:
                atomic_spins = cell.atomic_spins

            if cell.species_pot is not None:
                species_pot = cell.species_pot

            if cell.kpoint_mp_grid is not None:
                kpoint_mp_grid = cell.kpoint_mp_grid
                kpoint_mp_spacing = None
                kpoints_list = ()
            elif cell.kpoint_mp_spacing is not None:
                kpoint_mp_spacing = cell.kpoint_mp_spacing
                kpoint_mp_grid = None
                kpoints_list = ()
            elif cell.kpoints_list is not None:
                kpoints_list = cell.kpoints_list
                kpoint_mp_grid = None
                kpoint_mp_spacing = None

            if cell.symmetry_generate is not None:
                symmetry_generate = cell.symmetry_generate

            extra_scalars = _merge_assignments("cell", extra_scalars, cell.extra_scalars)
            extra_cell_blocks = _merge_blocks("cell", extra_cell_blocks, cell.extra_blocks)

        if layer.param is not None:
            has_param_layer = True
            param = layer.param
            keywords = _merge_assignments("param", keywords, param.keywords)
            extra_param_blocks = _merge_blocks("param", extra_param_blocks, param.extra_blocks)

    merged_cell = None
    if has_cell_layer:
        merged_cell = CellTemplate(
            lattice_cart=lattice_cart,
            lattice_abc=lattice_abc,
            positions_frac=positions_frac,
            positions_abs=positions_abs,
            atomic_spins=atomic_spins,
            species_pot=species_pot,
            kpoint_mp_grid=kpoint_mp_grid,
            kpoint_mp_spacing=kpoint_mp_spacing,
            kpoints_list=kpoints_list,
            symmetry_generate=symmetry_generate,
            extra_scalars=extra_scalars,
            extra_blocks=extra_cell_blocks,
        )

    merged_param = None
    if has_param_layer:
        merged_param = ParamTemplate(
            keywords=keywords,
            extra_blocks=extra_param_blocks,
        )

    return InputTemplate(cell=merged_cell, param=merged_param)


def merge_with_precedence(
    *,
    global_defaults: InputTemplate | None = None,
    workflow_defaults: InputTemplate | None = None,
    structure_defaults: InputTemplate | None = None,
    sweep_overrides: InputTemplate | None = None,
    user_overrides: InputTemplate | None = None,
) -> InputTemplate:
    """Merge layers using the framework's explicit precedence order."""

    return merge_templates(
        global_defaults,
        workflow_defaults,
        structure_defaults,
        sweep_overrides,
        user_overrides,
    )


def resolve_job(
    *,
    name: str,
    base_structure: StructureSpec,
    workflow_stage: WorkflowStage,
    global_defaults: InputTemplate | None = None,
    workflow_defaults: InputTemplate | None = None,
    structure_defaults: InputTemplate | None = None,
    patch: JobPatch | None = None,
    user_overrides: InputTemplate | None = None,
    tags: tuple[str, ...] = (),
    provenance: dict[str, Any] | None = None,
) -> JobSpec:
    """Resolve one concrete JobSpec from layered templates and an optional patch."""

    resolved_structure = patch.structure if patch and patch.structure is not None else base_structure

    sweep_overrides = None
    if patch is not None and (patch.cell is not None or patch.param is not None):
        sweep_overrides = InputTemplate(cell=patch.cell, param=patch.param)

    merged = merge_with_precedence(
        global_defaults=global_defaults,
        workflow_defaults=workflow_defaults,
        structure_defaults=structure_defaults,
        sweep_overrides=sweep_overrides,
        user_overrides=user_overrides,
    )

    cell_template = merged.cell or CellTemplate()
    param_template = merged.param or ParamTemplate()

    known_cell_scalar_names = {"kpoint_mp_grid", "kpoint_mp_spacing", "symmetry_generate"}
    known_cell_block_names = {
        "lattice_cart",
        "lattice_abc",
        "positions_frac",
        "positions_abs",
        "species_pot",
        "kpoints_list",
    }
    extra_scalars = tuple(
        assignment
        for assignment in cell_template.extra_scalars
        if canonical_keyword_name("cell", assignment.name) not in known_cell_scalar_names
    )
    extra_blocks = tuple(
        block
        for block in cell_template.extra_blocks
        if canonical_keyword_name("cell", block.name) not in known_cell_block_names
    )

    cell_file = CellFile(
        lattice_cart=(
            cell_template.lattice_cart
            if cell_template.lattice_cart is not None
            else None if cell_template.lattice_abc is not None else resolved_structure.lattice_cart
        ),
        lattice_abc=cell_template.lattice_abc,
        positions_frac=(
            cell_template.positions_frac
            if cell_template.positions_frac is not None
            else None if cell_template.positions_abs is not None else resolved_structure.positions_frac
        ),
        positions_abs=cell_template.positions_abs,
        atomic_spins=(
            None
            if not _param_spin_polarized(param_template.keywords)
            else (
                cell_template.atomic_spins
                if cell_template.atomic_spins is not None
                else resolved_structure.atomic_spins
            )
        ),
        species_pot=cell_template.species_pot or (),
        kpoints_list=cell_template.kpoints_list or (),
        kpoint_mp_grid=cell_template.kpoint_mp_grid,
        kpoint_mp_spacing=cell_template.kpoint_mp_spacing,
        symmetry_generate=cell_template.symmetry_generate if cell_template.symmetry_generate is not None else False,
        scalars=extra_scalars,
        blocks=extra_blocks,
    )

    param_file = ParamFile(
        keywords=param_template.keywords,
        blocks=param_template.extra_blocks,
    )

    resolved_provenance = dict(provenance or {})
    resolved_provenance.setdefault("merge_precedence", list(MERGE_PRECEDENCE_ORDER))
    if patch is not None:
        resolved_provenance.setdefault("patch_label_suffix", patch.label_suffix)

    return JobSpec(
        name=name,
        structure=resolved_structure,
        cell_file=cell_file,
        param_file=param_file,
        workflow_stage=workflow_stage,
        tags=tags,
        provenance=resolved_provenance,
    )
