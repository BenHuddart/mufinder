"""The pinned convergence plots on the Convergence Testing overview.

Adopting a cutoff energy or a k-point grid as the project default from the
convergence examiner pins a snapshot of the examiner's plot to the overview
page. This module builds, stores and validates those snapshots on the project
document: a snapshot shows only while it belongs to the active host and still
describes the project default, and a k-point snapshot's spacings are recomputed
from the cell its sweep ran on each time it is read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from mufinder.domain.models import Structure
from mufinder.persistence.codec import to_jsonable
from mufinder.science.kpoint_grids import distinct_grid_spacings, format_kpoint_grid
from mufinder.services.project_document import ProjectDocument

#: ``ui_state`` key of the pinned overview snapshots, keyed by sweep kind.
CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY = "convergence_overview_snapshots"

#: ``ui_state`` keys of the cutoff and k-point examiners' saved state.
CUTOFF_EXAMINER_STATE_KEY = "convergence_examiner_state"
KPOINT_EXAMINER_STATE_KEY = "kpoint_examiner_state"

#: Sweep kinds, used as keys of the stored snapshots.
CUTOFF_SNAPSHOT = "cutoff"
KPOINT_SNAPSHOT = "kpoint"

_CUTOFF_MATCH_TOLERANCE_EV = 1e-9


# --- storage --------------------------------------------------------------------


def convergence_overview_snapshots(document: ProjectDocument | None) -> dict[str, object]:
    """Return the stored snapshots keyed by sweep kind (empty when none)."""

    if document is None:
        return {}
    raw_value = document.ui_state.get(CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY, {})
    return raw_value if isinstance(raw_value, dict) else {}


def store_convergence_overview_snapshot(
    document: ProjectDocument | None,
    kind: str,
    snapshot: Mapping[str, object] | None,
) -> None:
    """Store ``snapshot`` as the ``kind`` pin, or remove that pin for ``None``."""

    if document is None:
        return
    snapshots = dict(convergence_overview_snapshots(document))
    if snapshot is None:
        snapshots.pop(kind, None)
    else:
        serialized = to_jsonable(dict(snapshot))
        if isinstance(serialized, dict):
            snapshots[kind] = serialized
    document.ui_state[CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] = snapshots


# --- building -------------------------------------------------------------------


def build_cutoff_overview_snapshot(
    document: ProjectDocument | None,
    *,
    cutoff_energy: float,
    energy_threshold: float,
    force_threshold: float,
    simulation_cell_structure_id: str | None,
    stress_threshold: float | None = None,
) -> dict[str, object] | None:
    """Return the snapshot that pins an adopted cutoff, from the examiner's plot data.

    ``None`` when there is no active host or the cutoff examiner has saved no
    plot data.
    """

    return _build_overview_snapshot(
        document,
        examiner_state_key=CUTOFF_EXAMINER_STATE_KEY,
        default_value=float(cutoff_energy),
        energy_threshold=energy_threshold,
        force_threshold=force_threshold,
        simulation_cell_structure_id=simulation_cell_structure_id,
        stress_threshold=stress_threshold,
    )


def build_kpoint_overview_snapshot(
    document: ProjectDocument | None,
    *,
    grid: tuple[int, int, int],
    energy_threshold: float,
    force_threshold: float,
    simulation_cell_structure_id: str | None,
    point_grids: Mapping[float, tuple[int, int, int]],
    stress_threshold: float | None = None,
) -> dict[str, object] | None:
    """Return the snapshot that pins an adopted k-point grid, with the point labels.

    ``point_grids`` maps each examined k-point count to its grid; the snapshot
    stores them as ``point_labels`` (``{"8.0": "Grid: 2 x 2 x 2"}``), which the
    overview shows and :func:`recompute_snapshot_spacings` reads back. ``None``
    when there is no active host or the k-point examiner has saved no plot data.
    """

    return _build_overview_snapshot(
        document,
        examiner_state_key=KPOINT_EXAMINER_STATE_KEY,
        default_value=[int(grid[0]), int(grid[1]), int(grid[2])],
        energy_threshold=energy_threshold,
        force_threshold=force_threshold,
        simulation_cell_structure_id=simulation_cell_structure_id,
        stress_threshold=stress_threshold,
        point_labels={
            str(float(key)): kpoint_grid_label(grid) for key, grid in point_grids.items()
        },
    )


def _build_overview_snapshot(
    document: ProjectDocument | None,
    *,
    examiner_state_key: str,
    default_value: object,
    energy_threshold: float,
    force_threshold: float,
    simulation_cell_structure_id: str | None,
    stress_threshold: float | None,
    point_labels: Mapping[str, str] | None = None,
) -> dict[str, object] | None:
    if document is None:
        return None
    host_structure_id = _active_host_id_text(document)
    if host_structure_id is None:
        return None

    state = document.ui_state.get(examiner_state_key, {})
    if not isinstance(state, Mapping):
        return None
    plot_data = state.get("plot_data")
    if not isinstance(plot_data, Mapping):
        return None

    snapshot: dict[str, object] = {
        "host_structure_id": host_structure_id,
        "simulation_cell_structure_id": simulation_cell_structure_id,
        "default_value": default_value,
        "energy_threshold": float(energy_threshold),
        "force_threshold": float(force_threshold),
        "plot_data": dict(plot_data),
    }
    if point_labels is not None:
        snapshot["point_labels"] = dict(point_labels)
    if stress_threshold is not None:
        snapshot["stress_threshold"] = float(stress_threshold)
    return snapshot


# --- validation -----------------------------------------------------------------


def validated_cutoff_overview_snapshot(
    document: ProjectDocument | None,
) -> dict[str, object] | None:
    """Return the cutoff snapshot while it belongs to the active host and the default.

    ``None`` when it is missing, was pinned for another host, or no longer
    matches the project's recommended cutoff energy.
    """

    snapshot = _host_snapshot(document, CUTOFF_SNAPSHOT)
    if snapshot is None or document is None:
        return None
    try:
        snapshot_value = float(snapshot.get("default_value"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    current_value = document.project.recommended_cutoff_energy
    if (
        current_value is None
        or abs(snapshot_value - float(current_value)) > _CUTOFF_MATCH_TOLERANCE_EV
    ):
        return None
    return snapshot


def validated_kpoint_overview_snapshot(
    document: ProjectDocument | None,
) -> dict[str, object] | None:
    """Return the k-point snapshot while it belongs to the active host and the default.

    ``None`` when it is missing, was pinned for another host, or no longer
    matches the project's recommended grid. The returned copy carries
    ``point_spacings`` recomputed on the cell the sweep ran on (a saved
    supercell, or the host for a snapshot that records none), so values stored
    with an older spacing formula are always refreshed.
    """

    snapshot = _host_snapshot(document, KPOINT_SNAPSHOT)
    if snapshot is None or document is None:
        return None

    raw_value = snapshot.get("default_value")
    if not isinstance(raw_value, list) or len(raw_value) != 3:
        return None
    try:
        snapshot_grid = (int(raw_value[0]), int(raw_value[1]), int(raw_value[2]))
    except (TypeError, ValueError):
        return None

    current_grid = document.project.recommended_kpoint_grid
    if current_grid is None or snapshot_grid != current_grid:
        return None

    cell = kpoint_snapshot_cell(document, snapshot)
    if cell is not None:
        snapshot = recompute_snapshot_spacings(snapshot, cell.lattice_vectors)
    return snapshot


def kpoint_snapshot_cell(
    document: ProjectDocument | None, snapshot: Mapping[str, object]
) -> Structure | None:
    """Return the structure a k-point snapshot's sweep ran on, else the active host."""

    if document is None:
        return None
    raw_cell_id = snapshot.get("simulation_cell_structure_id")
    if isinstance(raw_cell_id, str) and raw_cell_id.strip():
        try:
            cell = document.structures.get(UUID(raw_cell_id.strip()))
        except ValueError:
            cell = None
        if cell is not None:
            return cell
    host_id = document.project.host_structure_id
    return document.structures.get(host_id) if host_id is not None else None


def _host_snapshot(document: ProjectDocument | None, kind: str) -> dict[str, object] | None:
    """Return the stored ``kind`` snapshot when it was pinned for the active host."""

    if document is None:
        return None
    snapshot = convergence_overview_snapshots(document).get(kind)
    if not isinstance(snapshot, dict):
        return None
    if _snapshot_host_id(snapshot) != _active_host_id_text(document):
        return None
    return snapshot


def _snapshot_host_id(snapshot: Mapping[str, object]) -> str | None:
    raw_value = snapshot.get("host_structure_id")
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    return None


def _active_host_id_text(document: ProjectDocument) -> str | None:
    host_id = document.project.host_structure_id
    return None if host_id is None else str(host_id)


# --- k-point spacings -----------------------------------------------------------


def kpoint_grid_label(grid: tuple[int, int, int]) -> str:
    """Return the ``Grid: a x b x c`` label a k-point snapshot stores for ``grid``.

    :func:`kpoint_grids_from_labels` reads it back.
    """

    return f"Grid: {format_kpoint_grid(grid)}"


def kpoint_grids_from_labels(
    point_labels: Mapping[float, str],
) -> dict[float, tuple[int, int, int]]:
    """Parse a k-point snapshot's ``Grid: a x b x c`` labels into grids keyed by count.

    Labels that are not a grid of three positive integers are skipped.
    """

    grids: dict[float, tuple[int, int, int]] = {}
    for x_value, label in point_labels.items():
        if not label.startswith("Grid:"):
            continue
        payload = label.split(":", maxsplit=1)[1].strip().replace(" ", "")
        parts = payload.split("x")
        if len(parts) != 3:
            continue
        try:
            grid = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        if grid[0] <= 0 or grid[1] <= 0 or grid[2] <= 0:
            continue
        grids[float(x_value)] = grid
    return grids


def recompute_snapshot_spacings(
    snapshot: Mapping[str, object],
    lattice_vectors: Sequence[Sequence[float]],
) -> dict[str, object]:
    """Return a copy of ``snapshot`` with ``point_spacings`` recomputed.

    Re-derives the spacings from ``lattice_vectors`` and the grid labels stored
    in the snapshot, so that stale values persisted with an older formula are
    replaced without mutating the caller's dict. Returns a shallow copy
    unchanged when the snapshot lacks ``point_labels`` or ``plot_data``, or if
    recomputation raises an unexpected error.
    """

    raw_labels = snapshot.get("point_labels")
    plot_data = snapshot.get("plot_data")
    if not isinstance(raw_labels, Mapping) or not isinstance(plot_data, Mapping):
        return dict(snapshot)

    float_labels: dict[float, str] = {}
    for key, value in raw_labels.items():
        try:
            float_labels[float(key)] = str(value)
        except (TypeError, ValueError):
            continue

    grids = kpoint_grids_from_labels(float_labels)
    try:
        fresh_spacings = distinct_grid_spacings(lattice_vectors, grids)
    except Exception:
        return dict(snapshot)

    return {
        **dict(snapshot),
        "plot_data": {**dict(plot_data), "point_spacings": fresh_spacings},
    }
