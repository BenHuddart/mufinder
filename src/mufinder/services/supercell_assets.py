"""Saved workflow supercell assets and each workflow's simulation-cell selection.

A supercell asset is a :class:`~mufinder.domain.models.Structure` built from a
host structure by an integer 3x3 matrix. It records the matrix under
``properties["supercell_asset"]``, with the workflow stage it was saved from
(none when it was saved from the host page), and points at the host through
``parent_structure_id``. Assets are project-wide: the library holds one entry per
cell, and every workflow selects that entry. Each workflow stage persists which
cell it simulates (the single unit cell or one saved asset) in the
``workflow_supercells`` ``ui_state`` entry. An asset referenced by a calculation
definition is frozen: it can be renamed but not regenerated or deleted.

:class:`~mufinder.services.project_service.ProjectService` delegates its
supercell methods to the functions here, under the same names and signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re
from typing import Any
from uuid import UUID

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.magnetic_properties import (
    atomic_spins_from_structure,
    has_nonzero_spins,
    propagation_vector_from_structure,
)
from mufinder.domain.models import Structure, utc_now
from mufinder.persistence.codec import parse_datetime
from mufinder.science.supercell import build_supercell_structure
from mufinder.services.project_document import ProjectDocument, active_host_structure

_WORKFLOW_SUPERCELLS_STATE_KEY = "workflow_supercells"
#: Selection mode: simulate the host unit cell.
WORKFLOW_SUPERCELL_UNIT_CELL = "unit_cell"
#: Selection mode: simulate one saved supercell asset.
WORKFLOW_SUPERCELL_SAVED = "saved_supercell"
# Past tense: pages keep showing it after the supercell becomes available again.
_FALLBACK_STATUS = "Using single unit cell: the saved supercell was not available."


@dataclass(slots=True, frozen=True)
class WorkflowSupercellSelection:
    """Persisted workflow simulation-cell selection."""

    mode: str = WORKFLOW_SUPERCELL_UNIT_CELL
    structure_id: UUID | None = None


@dataclass(slots=True, frozen=True)
class WorkflowSupercellAsset:
    """Metadata for one saved reusable supercell structure."""

    structure_id: UUID
    label: str
    parent_structure_id: UUID | None
    owner_workflow_stage: WorkflowStage | None
    matrix: list[list[int]]
    source_asset_structure_id: UUID | None = None
    saved_at: datetime | None = None


@dataclass(slots=True)
class WorkflowStructureResolution:
    """Resolved workflow structure plus UI-facing selection state."""

    structure: Structure | None
    available_assets: list[WorkflowSupercellAsset]
    selection: WorkflowSupercellSelection
    normalized_selection: WorkflowSupercellSelection
    selected_asset: WorkflowSupercellAsset | None = None
    status_message: str = ""
    source_workflow_stage: WorkflowStage | None = None

    @property
    def fell_back_to_unit_cell(self) -> bool:
        """Whether a saved selection no longer resolves, so the host unit cell is used."""

        return self.structure is not None and self.normalized_selection != self.selection


def _coerce_supercell_matrix(matrix: Any) -> list[list[int]] | None:
    if not isinstance(matrix, list) or len(matrix) != 3:
        return None
    parsed: list[list[int]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != 3:
            return None
        parsed.append([int(row[0]), int(row[1]), int(row[2])])
    return parsed


def _parse_saved_at(value: Any) -> datetime | None:
    try:
        parsed = parse_datetime(value) if isinstance(value, str) else None
    except ValueError:
        return None
    return parsed if parsed is not None and parsed.tzinfo is not None else None


def _parse_optional_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def _supercell_asset_label(matrix: list[list[int]]) -> str:
    if (
        matrix[0][1] == 0
        and matrix[0][2] == 0
        and matrix[1][0] == 0
        and matrix[1][2] == 0
        and matrix[2][0] == 0
        and matrix[2][1] == 0
    ):
        token = f"{matrix[0][0]}x{matrix[1][1]}x{matrix[2][2]}"
    else:
        token = (
            f"[{matrix[0][0]} {matrix[0][1]} {matrix[0][2]}; "
            f"{matrix[1][0]} {matrix[1][1]} {matrix[1][2]}; "
            f"{matrix[2][0]} {matrix[2][1]} {matrix[2][2]}]"
        )
    return f"{token} supercell"


def _supercell_asset_properties(
    matrix: list[list[int]],
    owner_workflow_stage: WorkflowStage | None,
    source_asset_structure_id: UUID | None,
    saved_at: datetime,
) -> dict[str, Any]:
    return {
        "matrix": [list(row) for row in matrix],
        "owner_workflow_stage": (
            owner_workflow_stage.value if owner_workflow_stage is not None else None
        ),
        "source_asset_structure_id": (
            str(source_asset_structure_id) if source_asset_structure_id is not None else None
        ),
        "saved_at": saved_at.isoformat(),
    }


def _workflow_supercells_state(document: ProjectDocument) -> dict[str, Any]:
    raw_state = document.ui_state.get(_WORKFLOW_SUPERCELLS_STATE_KEY)
    if isinstance(raw_state, dict):
        return raw_state
    state: dict[str, Any] = {}
    document.ui_state[_WORKFLOW_SUPERCELLS_STATE_KEY] = state
    return state


def get_workflow_supercell_selection(
    document: ProjectDocument,
    workflow_stage: WorkflowStage,
) -> WorkflowSupercellSelection:
    """Return ``workflow_stage``'s persisted selection; anything malformed reads as the unit cell."""

    raw = _workflow_supercells_state(document).get(workflow_stage.value, {})
    if not isinstance(raw, dict):
        return WorkflowSupercellSelection()

    mode = str(raw.get("mode", WORKFLOW_SUPERCELL_UNIT_CELL)).strip().lower()
    if mode != WORKFLOW_SUPERCELL_SAVED:
        return WorkflowSupercellSelection()

    structure_id = _parse_optional_uuid(raw.get("structure_id"))
    if structure_id is None:
        return WorkflowSupercellSelection()
    return WorkflowSupercellSelection(mode=WORKFLOW_SUPERCELL_SAVED, structure_id=structure_id)


def set_workflow_supercell_selection(
    document: ProjectDocument,
    workflow_stage: WorkflowStage,
    selection: WorkflowSupercellSelection,
) -> None:
    """Persist ``workflow_stage``'s selection; a saved selection without an id stores the unit cell."""

    state = _workflow_supercells_state(document)
    if selection.mode == WORKFLOW_SUPERCELL_SAVED and selection.structure_id is not None:
        state[workflow_stage.value] = {
            "mode": WORKFLOW_SUPERCELL_SAVED,
            "structure_id": str(selection.structure_id),
        }
        return

    state[workflow_stage.value] = {"mode": WORKFLOW_SUPERCELL_UNIT_CELL}


def supercell_asset(structure: Structure) -> WorkflowSupercellAsset | None:
    """Return the asset metadata ``structure`` carries, or ``None`` when it is not a valid asset."""

    raw = structure.properties.get("supercell_asset")
    if not isinstance(raw, dict):
        return None

    matrix = _coerce_supercell_matrix(raw.get("matrix"))
    if matrix is None:
        return None

    owner = raw.get("owner_workflow_stage")
    try:
        owner_stage = WorkflowStage(str(owner)) if owner is not None else None
    except ValueError:
        return None

    return WorkflowSupercellAsset(
        structure_id=structure.id,
        label=structure.label,
        parent_structure_id=structure.parent_structure_id,
        owner_workflow_stage=owner_stage,
        matrix=matrix,
        source_asset_structure_id=_parse_optional_uuid(raw.get("source_asset_structure_id")),
        saved_at=_parse_saved_at(raw.get("saved_at")),
    )


def list_workflow_supercell_assets(
    document: ProjectDocument,
    *,
    host_structure_id: UUID | None,
) -> list[WorkflowSupercellAsset]:
    """Return the assets built from ``host_structure_id``, ordered by label."""

    if host_structure_id is None:
        return []

    assets: list[WorkflowSupercellAsset] = []
    for structure in document.structures.values():
        if structure.parent_structure_id != host_structure_id:
            continue
        asset = supercell_asset(structure)
        if asset is None:
            continue
        assets.append(asset)

    assets.sort(key=lambda asset: (asset.label.casefold(), str(asset.structure_id)))
    return assets


def latest_workflow_supercell_asset(
    document: ProjectDocument,
    *,
    host_structure_id: UUID | None,
    matrix: list[list[int]] | None = None,
) -> WorkflowSupercellAsset | None:
    """Return the last saved asset of ``host_structure_id``, of ``matrix`` when one is given.

    An asset saved before ``saved_at`` was recorded counts as older than any that has it.
    """

    matches = [
        asset
        for asset in list_workflow_supercell_assets(document, host_structure_id=host_structure_id)
        if matrix is None or asset.matrix == matrix
    ]
    oldest = datetime.min.replace(tzinfo=UTC)
    return max(matches, key=lambda asset: asset.saved_at or oldest, default=None)


def supercell_editor_seed_matrix(document: ProjectDocument) -> list[list[int]]:
    """Return the matrix the Supercell Editor opens on when no cell is in use.

    That is the active host's most recently saved supercell. A project
    without one reads its legacy ``Project.supercell_matrix``, which new projects
    leave at the identity.
    """

    host = active_host_structure(document)
    latest = latest_workflow_supercell_asset(
        document, host_structure_id=host.id if host is not None else None
    )
    if latest is not None:
        return [list(row) for row in latest.matrix]
    legacy = _coerce_supercell_matrix(document.project.supercell_matrix)
    return legacy if legacy is not None else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def structure_atomic_spins(structure: Structure) -> list[float]:
    """Return numeric per-atom spins for a structure, defaulting missing values to zero."""

    raw = structure.properties.get("atomic_spins")
    if raw is None:
        raw = structure.properties.get("initial_magnetic_moments")

    spins = [0.0] * len(structure.species)
    saw_numeric = False
    if isinstance(raw, list):
        for index, value in enumerate(raw[: len(spins)]):
            try:
                spins[index] = float(value)
                saw_numeric = True
            except (TypeError, ValueError):
                spins[index] = 0.0

    fallback = atomic_spins_from_structure(structure, atom_count=len(spins))
    if saw_numeric:
        if supercell_asset(structure) is not None:
            return spins
        if fallback is not None and not has_nonzero_spins(spins) and has_nonzero_spins(fallback):
            return fallback
        return spins

    if fallback is not None:
        return fallback
    return spins


def _repair_legacy_supercell_asset_spins(
    *,
    host: Structure,
    structure: Structure,
    asset: WorkflowSupercellAsset,
) -> bool:
    host_spins = structure_atomic_spins(host)
    if not has_nonzero_spins(host_spins):
        return False

    raw = structure.properties.get("atomic_spins")
    raw_spins = [0.0] * len(structure.species)
    saw_numeric = False
    if isinstance(raw, list):
        for index, value in enumerate(raw[: len(raw_spins)]):
            try:
                raw_spins[index] = float(value)
                saw_numeric = True
            except (TypeError, ValueError):
                raw_spins[index] = 0.0
    if saw_numeric and has_nonzero_spins(raw_spins):
        return False

    try:
        repaired, expanded_spins = build_supercell_structure(
            host,
            asset.matrix,
            atomic_spins=host_spins,
            propagation_vector=list(propagation_vector_from_structure(host)),
        )
    except ValueError:
        return False

    if not has_nonzero_spins(expanded_spins):
        return False

    structure.lattice_vectors = [list(vector) for vector in repaired.lattice_vectors]
    structure.species = list(repaired.species)
    structure.fractional_positions = [list(position) for position in repaired.fractional_positions]
    repaired_properties = dict(repaired.properties)
    repaired_properties["supercell_asset"] = dict(structure.properties["supercell_asset"])
    structure.properties = repaired_properties
    return True


def repair_supercell_asset_if_needed(
    document: ProjectDocument,
    *,
    structure_id: UUID,
) -> bool:
    """Rebuild an asset that lacks its magnetic host's spins; return whether it was rebuilt."""

    structure = document.structures.get(structure_id)
    if structure is None:
        return False
    asset = supercell_asset(structure)
    if asset is None:
        return False
    parent_id = structure.parent_structure_id
    if parent_id is None:
        return False
    host = document.structures.get(parent_id)
    if host is None:
        return False
    return _repair_legacy_supercell_asset_spins(host=host, structure=structure, asset=asset)


def _refresh_supercell_asset_if_needed(
    *,
    host: Structure,
    structure: Structure,
) -> None:
    if structure.parent_structure_id != host.id:
        return
    asset = supercell_asset(structure)
    if asset is None:
        return
    _repair_legacy_supercell_asset_spins(host=host, structure=structure, asset=asset)


def create_workflow_supercell_asset(
    document: ProjectDocument,
    *,
    workflow_stage: WorkflowStage | None,
    host_structure_id: UUID,
    matrix: list[list[int]],
    label: str | None = None,
    source_asset_structure_id: UUID | None = None,
    reuse_existing: bool = True,
) -> Structure:
    """Build and register an asset of ``host_structure_id``, reusing a matching one by default.

    ``workflow_stage`` records the workflow it was saved from, or ``None`` for no
    workflow. By default a saved asset with the same matrix is returned, whichever
    workflow it was saved from.
    """

    host = document.structures.get(host_structure_id)
    if host is None:
        raise ValueError("host_structure_id must reference a known structure.")

    parsed_matrix = _coerce_supercell_matrix(matrix)
    if parsed_matrix is None:
        raise ValueError("Supercell matrix must be a 3x3 integer matrix.")

    if reuse_existing:
        match = latest_workflow_supercell_asset(
            document, host_structure_id=host_structure_id, matrix=parsed_matrix
        )
        if match is not None:
            existing = document.structures[match.structure_id]
            _refresh_supercell_asset_if_needed(host=host, structure=existing)
            return existing

    structure, _expanded_spins = build_supercell_structure(
        host,
        parsed_matrix,
        atomic_spins=structure_atomic_spins(host),
        propagation_vector=list(propagation_vector_from_structure(host)),
    )
    structure.label = str(label or "").strip() or _supercell_asset_label(parsed_matrix)
    structure.parent_structure_id = host_structure_id
    structure.properties["supercell_asset"] = _supercell_asset_properties(
        parsed_matrix, workflow_stage, source_asset_structure_id, utc_now()
    )
    document.structures[structure.id] = structure
    return structure


def supercell_asset_references(
    document: ProjectDocument,
    *,
    asset_structure_id: UUID,
) -> list[str]:
    """Return user-facing references to a saved supercell asset."""

    references: list[str] = []
    for definition in document.calculations.values():
        if asset_structure_id in definition.input_structure_ids:
            references.append(
                f"{definition.workflow_stage.value} calculation '{definition.calc_type}'"
            )

    for stage_value, raw in _workflow_supercells_state(document).items():
        if not isinstance(raw, dict):
            continue
        raw_id = _parse_optional_uuid(raw.get("structure_id"))
        if raw_id == asset_structure_id:
            references.append(f"{stage_value} active selection")

    return references


def workflows_selecting_supercell_asset(
    document: ProjectDocument,
    *,
    asset_structure_id: UUID,
) -> list[WorkflowStage]:
    """Return the workflow stages that simulate the asset, in workflow order.

    A selection of an asset of another host resolves to the unit cell, so it does not count.
    """

    structure = document.structures.get(asset_structure_id)
    host = active_host_structure(document)
    if structure is None or host is None or structure.parent_structure_id != host.id:
        return []
    return [
        stage
        for stage in WorkflowStage
        if get_workflow_supercell_selection(document, stage).structure_id == asset_structure_id
    ]


def distinct_supercell_asset_label(
    document: ProjectDocument,
    *,
    host_structure_id: UUID,
    label: str,
) -> str:
    """Return ``label``, with ``copy`` added when a saved asset of the host already has it."""

    taken = {
        asset.label.casefold()
        for asset in list_workflow_supercell_assets(document, host_structure_id=host_structure_id)
    }
    if label.casefold() not in taken:
        return label
    base = re.sub(r" copy( \d+)?$", "", label, flags=re.IGNORECASE)
    candidate = f"{base} copy"
    number = 2
    while candidate.casefold() in taken:
        candidate = f"{base} copy {number}"
        number += 1
    return candidate


def supercell_asset_is_frozen(
    document: ProjectDocument,
    *,
    asset_structure_id: UUID,
) -> bool:
    """Return whether the asset is frozen into a calculation definition."""

    return any(
        asset_structure_id in definition.input_structure_ids
        for definition in document.calculations.values()
    )


def rename_workflow_supercell_asset(
    document: ProjectDocument,
    *,
    asset_structure_id: UUID,
    label: str,
) -> Structure:
    """Rename a saved supercell asset."""

    structure = document.structures.get(asset_structure_id)
    if structure is None or supercell_asset(structure) is None:
        raise ValueError("Selected structure is not a saved supercell asset.")
    cleaned = str(label).strip()
    if not cleaned:
        raise ValueError("Supercell name must not be empty.")
    structure.label = cleaned
    structure.properties["supercell_asset"]["saved_at"] = utc_now().isoformat()
    return structure


def update_workflow_supercell_asset_matrix(
    document: ProjectDocument,
    *,
    asset_structure_id: UUID,
    matrix: list[list[int]],
    label: str | None = None,
) -> Structure:
    """Regenerate an unfrozen saved supercell asset with a new matrix."""

    structure = document.structures.get(asset_structure_id)
    if structure is None:
        raise ValueError("Selected supercell does not exist in this project.")
    asset = supercell_asset(structure)
    if asset is None:
        raise ValueError("Selected structure is not a saved supercell asset.")
    if supercell_asset_is_frozen(document, asset_structure_id=asset_structure_id):
        raise ValueError(
            "This supercell is already used by configured calculations; save a new copy instead."
        )
    if structure.parent_structure_id is None:
        raise ValueError("Saved supercell is missing its source structure.")
    host = document.structures.get(structure.parent_structure_id)
    if host is None:
        raise ValueError("Saved supercell source structure is no longer available.")

    parsed_matrix = _coerce_supercell_matrix(matrix)
    if parsed_matrix is None:
        raise ValueError("Supercell matrix must be a 3x3 integer matrix.")

    regenerated, _expanded_spins = build_supercell_structure(
        host,
        parsed_matrix,
        atomic_spins=structure_atomic_spins(host),
        propagation_vector=list(propagation_vector_from_structure(host)),
    )
    structure.lattice_vectors = [list(vector) for vector in regenerated.lattice_vectors]
    structure.species = list(regenerated.species)
    structure.fractional_positions = [
        list(position) for position in regenerated.fractional_positions
    ]
    structure.label = (
        str(label or "").strip() or structure.label or _supercell_asset_label(parsed_matrix)
    )
    structure.properties = dict(regenerated.properties)
    structure.properties["supercell_asset"] = _supercell_asset_properties(
        parsed_matrix, asset.owner_workflow_stage, asset.source_asset_structure_id, utc_now()
    )
    structure.parent_structure_id = host.id
    return structure


def delete_workflow_supercell_asset(
    document: ProjectDocument,
    *,
    asset_structure_id: UUID,
) -> Structure:
    """Delete an unused saved supercell asset and clear stale active selections."""

    structure = document.structures.get(asset_structure_id)
    if structure is None or supercell_asset(structure) is None:
        raise ValueError("Selected structure is not a saved supercell asset.")
    if supercell_asset_is_frozen(document, asset_structure_id=asset_structure_id):
        raise ValueError(
            "This supercell is already used by configured calculations and cannot be deleted."
        )

    removed = document.structures.pop(asset_structure_id)
    state = _workflow_supercells_state(document)
    for stage_value, raw in list(state.items()):
        if not isinstance(raw, dict):
            continue
        if _parse_optional_uuid(raw.get("structure_id")) == asset_structure_id:
            state[stage_value] = {"mode": WORKFLOW_SUPERCELL_UNIT_CELL}
    return removed


def use_workflow_supercell_asset(
    document: ProjectDocument,
    *,
    workflow_stage: WorkflowStage,
    asset_structure_id: UUID,
) -> Structure:
    """Select a saved asset for ``workflow_stage``; the asset is shared, not copied."""

    host = active_host_structure(document)
    if host is None:
        raise ValueError("Load a host structure before selecting a saved supercell.")

    asset_structure = document.structures.get(asset_structure_id)
    if asset_structure is None:
        raise ValueError("Selected supercell does not exist in this project.")

    asset = supercell_asset(asset_structure)
    if asset is None:
        raise ValueError("Selected structure is not a saved supercell asset.")
    if asset_structure.parent_structure_id != host.id:
        raise ValueError("Selected supercell does not belong to the active host structure.")

    _refresh_supercell_asset_if_needed(host=host, structure=asset_structure)

    set_workflow_supercell_selection(
        document,
        workflow_stage,
        WorkflowSupercellSelection(
            mode=WORKFLOW_SUPERCELL_SAVED,
            structure_id=asset_structure.id,
        ),
    )
    return asset_structure


def clear_workflow_supercell_selection(
    document: ProjectDocument,
    workflow_stage: WorkflowStage,
) -> None:
    """Reset ``workflow_stage`` to simulate the single unit cell."""

    set_workflow_supercell_selection(document, workflow_stage, WorkflowSupercellSelection())


def resolve_workflow_structure(
    document: ProjectDocument,
    workflow_stage: WorkflowStage,
    *,
    normalize_state: bool = False,
) -> WorkflowStructureResolution:
    """Return the structure ``workflow_stage`` simulates and its selection state.

    A saved selection that no longer resolves to an asset of the active host
    falls back to the host unit cell; ``normalize_state`` also writes that
    fallback back to ``ui_state``.
    """

    host = active_host_structure(document)
    selection = get_workflow_supercell_selection(document, workflow_stage)
    normalized = selection
    available_assets = list_workflow_supercell_assets(
        document,
        host_structure_id=host.id if host is not None else None,
    )

    if host is None:
        normalized = WorkflowSupercellSelection()
        if normalize_state:
            set_workflow_supercell_selection(document, workflow_stage, normalized)
        return WorkflowStructureResolution(
            structure=None,
            available_assets=[],
            selection=selection,
            normalized_selection=normalized,
            status_message="Load a host structure to choose a simulation cell.",
        )

    if selection.mode == WORKFLOW_SUPERCELL_SAVED and selection.structure_id is not None:
        selected_structure = document.structures.get(selection.structure_id)
        selected_asset = (
            supercell_asset(selected_structure) if selected_structure is not None else None
        )
        if (
            selected_structure is not None
            and selected_asset is not None
            and selected_structure.parent_structure_id == host.id
        ):
            _refresh_supercell_asset_if_needed(host=host, structure=selected_structure)
            selected_asset = supercell_asset(selected_structure)
            if selected_asset is None:
                normalized = WorkflowSupercellSelection()
                if normalize_state:
                    set_workflow_supercell_selection(document, workflow_stage, normalized)
                return WorkflowStructureResolution(
                    structure=host,
                    available_assets=available_assets,
                    selection=selection,
                    normalized_selection=normalized,
                    status_message=_FALLBACK_STATUS,
                )
            source_workflow_stage = selected_asset.owner_workflow_stage
            if selected_asset.source_asset_structure_id is not None:
                source_structure = document.structures.get(selected_asset.source_asset_structure_id)
                source_asset = (
                    supercell_asset(source_structure) if source_structure is not None else None
                )
                if source_asset is not None and source_asset.owner_workflow_stage is not None:
                    source_workflow_stage = source_asset.owner_workflow_stage
            return WorkflowStructureResolution(
                structure=selected_structure,
                available_assets=available_assets,
                selection=selection,
                normalized_selection=selection,
                selected_asset=selected_asset,
                status_message=f"Using saved supercell: {selected_structure.label}",
                source_workflow_stage=source_workflow_stage,
            )

        normalized = WorkflowSupercellSelection()
        if normalize_state:
            set_workflow_supercell_selection(document, workflow_stage, normalized)
        return WorkflowStructureResolution(
            structure=host,
            available_assets=available_assets,
            selection=selection,
            normalized_selection=normalized,
            status_message=_FALLBACK_STATUS,
        )

    if normalize_state and normalized != selection:
        set_workflow_supercell_selection(document, workflow_stage, normalized)

    return WorkflowStructureResolution(
        structure=host,
        available_assets=available_assets,
        selection=selection,
        normalized_selection=normalized,
        status_message="Using single unit cell.",
    )
