"""Behaviour tests for the pinned convergence plots on the overview page."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mufinder.science.kpoint_grids import distinct_grid_spacings, spacing_from_grid
from mufinder.services.convergence_defaults import (
    CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY,
    CUTOFF_EXAMINER_STATE_KEY,
    CUTOFF_SNAPSHOT,
    KPOINT_EXAMINER_STATE_KEY,
    KPOINT_SNAPSHOT,
    build_cutoff_overview_snapshot,
    build_kpoint_overview_snapshot,
    convergence_overview_snapshots,
    kpoint_grids_from_labels,
    kpoint_snapshot_cell,
    recompute_snapshot_spacings,
    store_convergence_overview_snapshot,
    validated_cutoff_overview_snapshot,
    validated_kpoint_overview_snapshot,
)
from mufinder.services.project_service import ProjectService
from tests.support.structures import host_structure

_PLOT_DATA = {
    "cutoffs": [300.0, 400.0, 550.0],
    "energies": [-10.0, -10.1, -10.15],
    "forces": [0.3, 0.18, 0.05],
    "energy_threshold": 0.001,
    "force_threshold": 0.05,
}


@pytest.fixture
def project(tmp_path: Path):
    service = ProjectService()
    doc = service.create_project(tmp_path / "defaults.mufinder", "Defaults")
    host = host_structure(species=("Si",), positions=([0.0, 0.0, 0.0],))
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    return service, doc, host


def _supercell(service, doc, host):
    supercell = host_structure(
        label="2x1x1",
        lattice=[[10.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=("Si", "Si"),
        positions=([0.0, 0.0, 0.0], [0.5, 0.0, 0.0]),
        parent_structure_id=host.id,
    )
    service.add_structure(doc, supercell)
    return supercell


# --- building -------------------------------------------------------------------


def test_cutoff_snapshot_pins_the_examiner_plot_for_the_active_host(project) -> None:
    _service, doc, host = project
    doc.ui_state[CUTOFF_EXAMINER_STATE_KEY] = {"plot_data": _PLOT_DATA}

    snapshot = build_cutoff_overview_snapshot(
        doc,
        cutoff_energy=550,
        energy_threshold=0.001,
        force_threshold=0.05,
        simulation_cell_structure_id=str(host.id),
    )

    assert snapshot == {
        "host_structure_id": str(host.id),
        "simulation_cell_structure_id": str(host.id),
        "default_value": 550.0,
        "energy_threshold": 0.001,
        "force_threshold": 0.05,
        "plot_data": _PLOT_DATA,
    }
    assert "method" not in snapshot


def test_cutoff_snapshot_records_a_stress_threshold_when_given(project) -> None:
    _service, doc, _host = project
    doc.ui_state[CUTOFF_EXAMINER_STATE_KEY] = {"plot_data": _PLOT_DATA}

    snapshot = build_cutoff_overview_snapshot(
        doc,
        cutoff_energy=550.0,
        energy_threshold=0.001,
        force_threshold=0.05,
        simulation_cell_structure_id=None,
        stress_threshold=0.1,
    )

    assert snapshot is not None
    assert snapshot["stress_threshold"] == pytest.approx(0.1)
    assert snapshot["simulation_cell_structure_id"] is None


def test_kpoint_snapshot_stores_the_grid_and_the_point_labels(project) -> None:
    _service, doc, host = project
    doc.ui_state[KPOINT_EXAMINER_STATE_KEY] = {"plot_data": {"cutoffs": [8.0]}}

    snapshot = build_kpoint_overview_snapshot(
        doc,
        grid=(2, 2, 2),
        energy_threshold=0.001,
        force_threshold=0.05,
        simulation_cell_structure_id=str(host.id),
        labels={8: "Grid: 2 x 2 x 2"},
    )

    assert snapshot is not None
    assert snapshot["default_value"] == [2, 2, 2]
    assert snapshot["point_labels"] == {"8.0": "Grid: 2 x 2 x 2"}
    assert snapshot["plot_data"] == {"cutoffs": [8.0]}
    assert "stress_threshold" not in snapshot


@pytest.mark.parametrize(
    ("builder", "state_key", "kwargs"),
    [
        (build_cutoff_overview_snapshot, CUTOFF_EXAMINER_STATE_KEY, {"cutoff_energy": 550.0}),
        (
            build_kpoint_overview_snapshot,
            KPOINT_EXAMINER_STATE_KEY,
            {"grid": (2, 2, 2), "labels": {}},
        ),
    ],
)
@pytest.mark.parametrize("state", [None, "invalid", {}, {"plot_data": "invalid"}])
def test_no_snapshot_without_saved_examiner_plot_data(
    project, builder, state_key, kwargs, state
) -> None:
    _service, doc, _host = project
    if state is not None:
        doc.ui_state[state_key] = state

    snapshot = builder(
        doc,
        energy_threshold=0.001,
        force_threshold=0.05,
        simulation_cell_structure_id=None,
        **kwargs,
    )

    assert snapshot is None


def test_no_snapshot_without_a_document_or_an_active_host(project) -> None:
    _service, doc, _host = project
    doc.ui_state[CUTOFF_EXAMINER_STATE_KEY] = {"plot_data": _PLOT_DATA}
    kwargs = {
        "cutoff_energy": 550.0,
        "energy_threshold": 0.001,
        "force_threshold": 0.05,
        "simulation_cell_structure_id": None,
    }

    assert build_cutoff_overview_snapshot(None, **kwargs) is None
    doc.project.host_structure_id = None
    assert build_cutoff_overview_snapshot(doc, **kwargs) is None


# --- storage --------------------------------------------------------------------


def test_storing_a_snapshot_serialises_it_and_none_removes_it(project) -> None:
    _service, doc, _host = project

    store_convergence_overview_snapshot(
        doc, KPOINT_SNAPSHOT, {"default_value": (2, 2, 2), "plot_data": {"cutoffs": (8.0,)}}
    )
    store_convergence_overview_snapshot(doc, CUTOFF_SNAPSHOT, {"default_value": 550.0})

    stored = doc.ui_state[CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY]
    assert stored[KPOINT_SNAPSHOT] == {"default_value": [2, 2, 2], "plot_data": {"cutoffs": [8.0]}}
    assert stored[CUTOFF_SNAPSHOT] == {"default_value": 550.0}

    store_convergence_overview_snapshot(doc, CUTOFF_SNAPSHOT, None)
    store_convergence_overview_snapshot(doc, CUTOFF_SNAPSHOT, None)

    assert set(doc.ui_state[CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY]) == {KPOINT_SNAPSHOT}
    store_convergence_overview_snapshot(None, CUTOFF_SNAPSHOT, {"default_value": 1.0})


def test_malformed_stored_snapshots_read_as_none(project) -> None:
    _service, doc, _host = project
    doc.ui_state[CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] = "invalid"

    assert convergence_overview_snapshots(doc) == {}
    assert convergence_overview_snapshots(None) == {}
    assert validated_cutoff_overview_snapshot(doc) is None
    assert validated_kpoint_overview_snapshot(doc) is None
    store_convergence_overview_snapshot(doc, CUTOFF_SNAPSHOT, {"default_value": 1.0})
    assert doc.ui_state[CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] == {
        CUTOFF_SNAPSHOT: {"default_value": 1.0}
    }


# --- validation -----------------------------------------------------------------


def test_cutoff_snapshot_shows_while_it_matches_the_host_and_the_default(project) -> None:
    _service, doc, host = project
    doc.project.recommended_cutoff_energy = 550.0
    doc.ui_state[CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] = {
        CUTOFF_SNAPSHOT: {"host_structure_id": f" {host.id} ", "default_value": 550.0}
    }

    assert validated_cutoff_overview_snapshot(doc) is not None
    assert validated_cutoff_overview_snapshot(None) is None

    doc.project.recommended_cutoff_energy = 600.0
    assert validated_cutoff_overview_snapshot(doc) is None

    doc.project.recommended_cutoff_energy = None
    assert validated_cutoff_overview_snapshot(doc) is None


@pytest.mark.parametrize(
    "snapshot",
    [
        {"host_structure_id": "different-host", "default_value": 550.0},
        {"host_structure_id": "", "default_value": 550.0},
        {"default_value": 550.0},
        {"host_structure_id": "HOST", "default_value": "not a number"},
        {"host_structure_id": "HOST"},
    ],
)
def test_cutoff_snapshot_hidden_for_another_host_or_an_unreadable_value(project, snapshot) -> None:
    _service, doc, host = project
    doc.project.recommended_cutoff_energy = 550.0
    if snapshot.get("host_structure_id") == "HOST":
        snapshot = {**snapshot, "host_structure_id": str(host.id)}
    doc.ui_state[CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] = {CUTOFF_SNAPSHOT: snapshot}

    assert validated_cutoff_overview_snapshot(doc) is None


@pytest.mark.parametrize(
    "default_value",
    [[2, 2, 4], [2, 2], (2, 2, 2), ["a", 2, 2], None],
)
def test_kpoint_snapshot_hidden_unless_its_grid_is_the_default(project, default_value) -> None:
    _service, doc, host = project
    doc.project.recommended_kpoint_grid = (2, 2, 2)
    doc.ui_state[CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] = {
        KPOINT_SNAPSHOT: {"host_structure_id": str(host.id), "default_value": default_value}
    }

    assert validated_kpoint_overview_snapshot(doc) is None


def test_kpoint_snapshot_hidden_for_another_host(project) -> None:
    _service, doc, _host = project
    doc.project.recommended_kpoint_grid = (2, 2, 2)
    doc.ui_state[CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] = {
        KPOINT_SNAPSHOT: {"host_structure_id": "different-host", "default_value": [2, 2, 2]}
    }

    assert validated_kpoint_overview_snapshot(doc) is None
    doc.project.recommended_kpoint_grid = None
    assert validated_kpoint_overview_snapshot(doc) is None


def test_kpoint_overview_recomputes_spacings_on_the_sweep_cell(project) -> None:
    """The overview refreshes stored spacings from the cell the snapshot was swept on."""

    service, doc, host = project
    supercell = _supercell(service, doc, host)
    doc.project.recommended_kpoint_grid = (4, 2, 2)
    doc.ui_state[CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY] = {
        KPOINT_SNAPSHOT: {
            "host_structure_id": str(host.id),
            "simulation_cell_structure_id": str(supercell.id),
            "default_value": [4, 2, 2],
            "point_labels": {"16.0": "Grid: 4 x 2 x 2"},
            "plot_data": {
                "cutoffs": [16.0],
                "energies": [-20.0],
                "point_spacings": {"16.0": 999.0},
            },
        }
    }

    snapshot = validated_kpoint_overview_snapshot(doc)

    assert snapshot is not None
    assert snapshot["plot_data"]["point_spacings"][16.0] == pytest.approx(
        spacing_from_grid(supercell.lattice_vectors, (4, 2, 2))
    )
    assert snapshot["plot_data"]["point_spacings"][16.0] != pytest.approx(
        spacing_from_grid(host.lattice_vectors, (4, 2, 2))
    )
    # The stored snapshot keeps its values; only the returned copy is refreshed.
    stored = doc.ui_state[CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY][KPOINT_SNAPSHOT]
    assert stored["plot_data"]["point_spacings"] == {"16.0": 999.0}


@pytest.mark.parametrize("cell_id", [None, "", "not-a-uuid", "SUPERCELL_MISSING"])
def test_kpoint_snapshot_cell_falls_back_to_the_host(project, cell_id) -> None:
    _service, doc, host = project
    if cell_id == "SUPERCELL_MISSING":
        cell_id = "00000000-0000-0000-0000-000000000001"

    assert kpoint_snapshot_cell(doc, {"simulation_cell_structure_id": cell_id}) is host


def test_kpoint_snapshot_cell_is_the_recorded_supercell(project) -> None:
    service, doc, host = project
    supercell = _supercell(service, doc, host)

    snapshot = {"simulation_cell_structure_id": f" {supercell.id} "}

    assert kpoint_snapshot_cell(doc, snapshot) is supercell
    assert kpoint_snapshot_cell(None, snapshot) is None
    doc.project.host_structure_id = None
    assert kpoint_snapshot_cell(doc, {}) is None


# --- k-point spacings -----------------------------------------------------------


def test_grids_parse_from_grid_labels_and_skip_anything_else() -> None:
    grids = kpoint_grids_from_labels(
        {
            8.0: "Grid: 2 x 2 x 2",
            12.0: "Grid:2x2x3",
            1.0: "Grid: 1 k-points",
            4.0: "Grid: 0 x 2 x 2",
            6.0: "Grid: a x 2 x 3",
            9.0: "3 x 3 x 1",
        }
    )

    assert grids == {8.0: (2, 2, 2), 12.0: (2, 2, 3)}


def test_spacings_suppress_duplicates_from_a_dominant_axis() -> None:
    # Hexagonal AlF3 (a=4.872, c=12.466 Å). Under the round-trip formula the
    # c-axis at N_c=3 contributes 1/(2*c)*1/(N_c-1)·|b_c| ≈ 0.0401, while the
    # ab axes only beat that bound once N_ab ≥ 7. So (2,2,3), (3,3,3), (4,4,3)
    # all report the same spacing — c dominates the min for each. The first
    # occurrence is kept; the rest are suppressed to avoid a non-decreasing
    # annotation series that would mislead about convergence.
    a, c = 4.872, 12.466
    lattice = [[a, 0.0, 0.0], [-a / 2.0, a * math.sqrt(3) / 2.0, 0.0], [0.0, 0.0, c]]
    grids = {12.0: (2, 2, 3), 27.0: (3, 3, 3), 48.0: (4, 4, 3)}

    spacings = distinct_grid_spacings(lattice, grids)

    assert 12.0 in spacings
    assert 27.0 not in spacings
    assert 48.0 not in spacings


def test_spacings_leave_out_gamma_only_sampling() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    spacings = distinct_grid_spacings(lattice, {1.0: (1, 1, 1), 8.0: (2, 2, 2)})

    assert set(spacings) == {8.0}


def test_recompute_replaces_stale_spacings_without_mutating_the_snapshot() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    snapshot: dict = {
        "point_labels": {"8.0": "Grid: 2 x 2 x 2", "27.0": "Grid: 3 x 3 x 3", "x": "Grid: 1"},
        "plot_data": {
            "cutoffs": [8.0, 27.0],
            "energies": [-10.0, -10.01],
            # Deliberately wrong ("stale") spacing values:
            "point_spacings": {"8.0": 999.0, "27.0": 999.0},
        },
    }

    refreshed = recompute_snapshot_spacings(snapshot, lattice)

    assert refreshed["plot_data"]["point_spacings"][8.0] == pytest.approx(
        spacing_from_grid(lattice, (2, 2, 2))
    )
    assert refreshed["plot_data"]["point_spacings"][27.0] == pytest.approx(
        spacing_from_grid(lattice, (3, 3, 3))
    )
    assert snapshot["plot_data"]["point_spacings"] == {"8.0": 999.0, "27.0": 999.0}


def test_recompute_returns_a_copy_without_point_labels() -> None:
    # Snapshots that pre-date point_labels (or cutoff snapshots) are returned
    # as-is so the caller can still display them.
    snapshot: dict = {"plot_data": {"cutoffs": [300.0, 400.0], "point_spacings": {"300.0": 0.5}}}
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    result = recompute_snapshot_spacings(snapshot, lattice)

    assert result["plot_data"]["point_spacings"] == {"300.0": 0.5}
    assert result is not snapshot
