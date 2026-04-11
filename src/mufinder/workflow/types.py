"""Core types for the CASTEP calculation workflow framework.

This module defines dataclasses for structure bridging, template layering,
workflow variations, and resolved job specifications.

A future ExecutionSpec dataclass can accompany JobSpec with scheduler and
remote-execution metadata without replacing JobSpec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from mufinder.castep_io.types import (
    BlockSection,
    CellFile,
    KeywordAssignment,
    KPoint,
    LatticeABC,
    LatticeCart,
    ParamFile,
    PositionBlock,
    SpeciesPotential,
    ValueWithUnit,
)
from mufinder.domain.enums import WorkflowStage


@dataclass(frozen=True, slots=True)
class StructureSpec:
    """Workflow-ready crystallographic structure."""

    lattice_cart: LatticeCart
    positions_frac: PositionBlock
    species: tuple[str, ...]
    atomic_spins: tuple[float, ...] | None = None
    label: str = ""
    source_structure_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class CellTemplate:
    """Partial cell settings for one precedence layer.

    None fields are transparent and do not override lower-precedence layers.
    """

    lattice_cart: LatticeCart | None = None
    lattice_abc: LatticeABC | None = None
    positions_frac: PositionBlock | None = None
    positions_abs: PositionBlock | None = None
    atomic_spins: tuple[float, ...] | None = None
    species_pot: tuple[SpeciesPotential, ...] | None = None
    kpoint_mp_grid: tuple[int, int, int] | None = None
    kpoint_mp_spacing: float | ValueWithUnit | None = None
    kpoints_list: tuple[KPoint, ...] | None = None
    symmetry_generate: bool | None = None
    extra_scalars: tuple[KeywordAssignment, ...] = ()
    extra_blocks: tuple[BlockSection, ...] = ()


@dataclass(frozen=True, slots=True)
class ParamTemplate:
    """Partial param settings for one precedence layer.

    None fields are transparent and do not override lower-precedence layers.
    """

    keywords: tuple[KeywordAssignment, ...] = ()
    extra_blocks: tuple[BlockSection, ...] = ()


@dataclass(frozen=True, slots=True)
class InputTemplate:
    """Combined cell and param settings at one precedence layer."""

    cell: CellTemplate | None = None
    param: ParamTemplate | None = None


@dataclass(frozen=True, slots=True)
class JobPatch:
    """One workflow variation override.

    A patch can vary the structure, cell, and/or param settings.
    """

    label_suffix: str
    structure: StructureSpec | None = None
    cell: CellTemplate | None = None
    param: ParamTemplate | None = None


@dataclass(frozen=True, slots=True)
class JobSpec:
    """Fully resolved immutable job specification.

    Workflows expand into JobSpec objects, and only JobSpec objects are
    rendered into CASTEP files.
    """

    name: str
    structure: StructureSpec
    cell_file: CellFile
    param_file: ParamFile
    workflow_stage: WorkflowStage
    tags: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
