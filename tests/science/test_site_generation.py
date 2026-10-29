"""Tests for periodic muon site-generation algorithms."""

from __future__ import annotations

from itertools import combinations, product
import math

import numpy as np
import pytest

from mufinder.science import site_generation, spglib_compat
from mufinder.science.lattice import nearest_image_distance_matrix
from mufinder.science.site_generation import (
    SiteGenerationError,
    closest_symmetry_equivalent_position,
    estimate_free_volume,
    estimate_muon_muon_separation_for_count,
    estimate_random_site_count_from_separation,
    generate_attached_element_sites,
    generate_random_sites,
    generate_random_sites_by_distance,
    lattice_cart_to_frac,
    lattice_frac_to_cart,
    symmetry_distinct_atom_indexes,
    symmetry_equivalent_positions,
)
from tests.support.symmetry_hosts import cof2_host, si_primitive_host, zno_host

CUBIC_8 = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
_CUBIC = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
TINY_2 = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
# Determinant zero: rows 0 and 1 are colinear.
SINGULAR = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def _pbc_distance(a: list[float], b: list[float], lattice: list[list[float]]) -> float:
    delta = [a[i] - b[i] for i in range(3)]
    delta = [d - round(d) for d in delta]
    cart = lattice_frac_to_cart(delta, lattice)
    return math.sqrt(sum(v * v for v in cart))


_BRUTE_FORCE_OFFSETS = np.asarray(list(product((-2, -1, 0, 1, 2), repeat=3)), dtype=float)


def _brute_force_distance(a, b, lattice) -> float:
    """Shortest distance between ``a`` and any image of ``b`` within two cells, by enumeration."""

    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    cartesian = (delta + _BRUTE_FORCE_OFFSETS) @ np.asarray(lattice, dtype=float)
    return float(np.min(np.linalg.norm(cartesian, axis=1)))


# Lattice conversions


@pytest.mark.parametrize(
    "lattice",
    [
        [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]],
        [[3.0, 0.0, 0.0], [-1.5, 2.598, 0.0], [0.0, 0.0, 5.0]],
    ],
    ids=["orthorhombic", "hexagonal"],
)
def test_fractional_cartesian_roundtrip(lattice: list[list[float]]) -> None:
    frac = [0.25, 0.5, 0.75]
    cart = lattice_frac_to_cart(frac, lattice)
    recovered = lattice_cart_to_frac(cart, lattice)

    assert recovered == pytest.approx(frac)


def test_lattice_frac_to_cart_rejects_non_3x3_matrix() -> None:
    with pytest.raises(SiteGenerationError, match="3x3 matrix"):
        lattice_frac_to_cart([0.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_lattice_cart_to_frac_rejects_ragged_rows() -> None:
    with pytest.raises(SiteGenerationError, match="3x3 matrix"):
        lattice_cart_to_frac([1.0, 2.0, 3.0], [[1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_lattice_cart_to_frac_rejects_singular_lattice() -> None:
    with pytest.raises(SiteGenerationError, match="singular"):
        lattice_cart_to_frac([1.0, 2.0, 3.0], SINGULAR)


# Symmetry equivalents


def test_closest_symmetry_equivalent_position_selects_representative_near_anchor() -> None:
    lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    clustered = closest_symmetry_equivalent_position(
        [0.1, 0.2, 0.3],
        anchor_frac=[0.9, 0.8, 0.7],
        lattice_vectors=lattice,
        host_species=["Si"],
        host_positions_frac=[[0.0, 0.0, 0.0]],
    )

    assert clustered == pytest.approx([0.9, 0.8, 0.7])


def test_symmetry_equivalent_positions_unknown_species_returns_single_rep() -> None:
    equivalents = symmetry_equivalent_positions(
        [1.1, -0.2, 0.3],
        lattice_vectors=_CUBIC,
        host_species=["Xx"],
        host_positions_frac=[[0.0, 0.0, 0.0]],
    )
    # No symmetry available -> the wrapped input position is the only rep.
    assert equivalents == [pytest.approx([0.1, 0.8, 0.3])]


def test_symmetry_equivalent_positions_species_count_mismatch_returns_single_rep() -> None:
    equivalents = symmetry_equivalent_positions(
        [0.1, 0.2, 0.3],
        lattice_vectors=_CUBIC,
        host_species=["Si", "O"],
        host_positions_frac=[[0.0, 0.0, 0.0]],
    )
    assert equivalents == [pytest.approx([0.1, 0.2, 0.3])]


def test_host_symmetry_is_detected_once_per_host(monkeypatch: pytest.MonkeyPatch) -> None:
    spglib_compat.clear_symmetry_operations_cache()
    calls: list[float] = []
    real = spglib_compat.get_crystal_symmetry

    def counting(cell, *, symprec):
        calls.append(symprec)
        return real(cell, symprec=symprec)

    monkeypatch.setattr(spglib_compat, "get_crystal_symmetry", counting)
    host = {
        "lattice_vectors": _CUBIC,
        "host_species": ["Si", "Si"],
        "host_positions_frac": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    }
    try:
        first = symmetry_equivalent_positions([0.1, 0.1, 0.1], **host)
        second = symmetry_equivalent_positions([0.3, 0.2, 0.1], **host)
        operations = site_generation._symmetry_operations(**host)
    finally:
        spglib_compat.clear_symmetry_operations_cache()

    assert len(first) == 16
    assert len(second) == 96
    assert calls == [spglib_compat.SYMMETRY_TOLERANCE_ANG]
    assert operations is not None
    # Shared cached arrays must not be writable by callers.
    assert all(not rotation.flags.writeable for rotation in operations[0])


def test_symmetry_equivalents_fall_back_to_input_when_spglib_finds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spglib_compat.clear_symmetry_operations_cache()
    monkeypatch.setattr(spglib_compat, "get_crystal_symmetry", lambda cell, *, symprec: None)
    try:
        equivalents = symmetry_equivalent_positions(
            [1.1, -0.2, 0.3],
            lattice_vectors=_CUBIC,
            host_species=["Si"],
            host_positions_frac=[[0.0, 0.0, 0.0]],
        )
    finally:
        spglib_compat.clear_symmetry_operations_cache()

    assert equivalents == [pytest.approx([0.1, 0.8, 0.3])]


def test_cluster_positions_picks_the_true_nearest_image_in_a_hexagonal_cell() -> None:
    host = zno_host()
    rng = np.random.default_rng(320)

    for position, anchor in rng.uniform(0.0, 1.0, size=(400, 2, 3)):
        chosen = closest_symmetry_equivalent_position(
            position.tolist(),
            anchor_frac=anchor.tolist(),
            lattice_vectors=host.lattice_vectors,
            host_species=host.species,
            host_positions_frac=host.fractional_positions,
        )
        orbit = symmetry_equivalent_positions(
            position.tolist(),
            lattice_vectors=host.lattice_vectors,
            host_species=host.species,
            host_positions_frac=host.fractional_positions,
        )
        closest = min(_brute_force_distance(image, anchor, host.lattice_vectors) for image in orbit)
        assert _brute_force_distance(chosen, anchor, host.lattice_vectors) == pytest.approx(
            closest, abs=1e-9
        )


# Free volume, site-count and separation estimates


def test_estimate_free_volume_none_lattice_uses_unit_cell_volume() -> None:
    # exclusion_radius <= 0 returns the full (unit) cell volume immediately.
    volume = estimate_free_volume(
        lattice_vectors=None,
        host_positions_frac=[[0.0, 0.0, 0.0]],
        exclusion_radius=0.0,
    )
    assert volume == pytest.approx(1.0)


def test_estimate_free_volume_no_hosts_returns_full_volume() -> None:
    volume = estimate_free_volume(
        lattice_vectors=_CUBIC,
        host_positions_frac=[],
        exclusion_radius=1.5,
    )
    assert volume == pytest.approx(125.0)


def test_estimate_free_volume_excludes_atom_neighbourhood() -> None:
    full = 125.0
    free = estimate_free_volume(
        lattice_vectors=_CUBIC,
        host_positions_frac=[[0.0, 0.0, 0.0]],
        exclusion_radius=1.0,
    )
    assert 0.0 < free < full


def test_estimate_free_volume_rejects_nonpositive_volume() -> None:
    with pytest.raises(SiteGenerationError, match="Cell volume must be positive"):
        estimate_free_volume(
            lattice_vectors=SINGULAR,
            host_positions_frac=[[0.0, 0.0, 0.0]],
            exclusion_radius=1.0,
        )


def test_estimate_random_site_count_applies_symmetry_multiplicity() -> None:
    common = dict(
        lattice_vectors=_CUBIC,
        host_positions_frac=[[0.0, 0.0, 0.0]],
        min_muon_muon=1.0,
        min_muon_atom=0.5,
        host_species=["Si"],
    )
    with_sym = estimate_random_site_count_from_separation(
        include_symmetry_equivalents=True, **common
    )
    without_sym = estimate_random_site_count_from_separation(
        include_symmetry_equivalents=False, **common
    )
    # Reserving full symmetry orbits consumes more single-site volume, so the
    # symmetry-aware estimate must not exceed the symmetry-blind one.
    assert with_sym > 0.0
    assert without_sym > 0.0
    assert with_sym <= without_sym


def test_estimate_random_site_count_rejects_nonpositive_muon_muon() -> None:
    with pytest.raises(SiteGenerationError, match="greater than zero"):
        estimate_random_site_count_from_separation(
            lattice_vectors=_CUBIC,
            host_positions_frac=[],
            min_muon_muon=0.0,
            min_muon_atom=1.0,
        )


def test_estimate_random_site_count_rejects_negative_muon_atom() -> None:
    with pytest.raises(SiteGenerationError, match="non-negative"):
        estimate_random_site_count_from_separation(
            lattice_vectors=_CUBIC,
            host_positions_frac=[],
            min_muon_muon=1.0,
            min_muon_atom=-0.1,
        )


def test_estimate_muon_muon_separation_inverts_expected_random_count() -> None:
    lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    target_count = 30

    separation = estimate_muon_muon_separation_for_count(
        lattice_vectors=lattice,
        host_positions_frac=[],
        target_count=target_count,
        min_muon_atom=1.0,
        include_symmetry_equivalents=False,
    )
    estimated_count = estimate_random_site_count_from_separation(
        lattice_vectors=lattice,
        host_positions_frac=[],
        min_muon_muon=separation,
        min_muon_atom=1.0,
        include_symmetry_equivalents=False,
    )

    assert separation == pytest.approx(2.82, rel=0.03)
    assert estimated_count == pytest.approx(target_count, rel=0.02)


def test_estimate_separation_none_lattice_uses_identity_cell() -> None:
    separation = estimate_muon_muon_separation_for_count(
        lattice_vectors=None,
        host_positions_frac=[],
        target_count=2,
        min_muon_atom=0.5,
    )
    assert separation >= 0.1
    assert math.isfinite(separation)


def test_estimate_separation_rejects_nonpositive_target() -> None:
    with pytest.raises(SiteGenerationError, match="Target count"):
        estimate_muon_muon_separation_for_count(
            lattice_vectors=_CUBIC,
            host_positions_frac=[],
            target_count=0,
            min_muon_atom=1.0,
        )


def test_estimate_separation_rejects_negative_muon_atom() -> None:
    with pytest.raises(SiteGenerationError, match="non-negative"):
        estimate_muon_muon_separation_for_count(
            lattice_vectors=_CUBIC,
            host_positions_frac=[],
            target_count=5,
            min_muon_atom=-1.0,
        )


# Random site generation


def test_generate_random_sites_respects_distance_constraints() -> None:
    lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
    host_positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    min_muon_atom = 0.8
    min_muon_muon = 0.8

    points, attempts = generate_random_sites(
        lattice_vectors=lattice,
        host_positions_frac=host_positions,
        count=20,
        min_muon_muon=min_muon_muon,
        min_muon_atom=min_muon_atom,
        max_attempts=50000,
        seed=7,
    )

    assert len(points) == 20
    assert attempts > 0

    for idx, point in enumerate(points):
        for host in host_positions:
            assert _pbc_distance(point.fractional, host, lattice) >= min_muon_atom
        for other in points[idx + 1 :]:
            assert _pbc_distance(point.fractional, other.fractional, lattice) >= min_muon_muon


def test_symmetry_equivalent_random_sites_reserve_full_orbits() -> None:
    lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    host_species = ["Si"]
    host_positions = [[0.0, 0.0, 0.0]]

    equivalents = symmetry_equivalent_positions(
        [0.1, 0.2, 0.3],
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions,
    )
    assert len(equivalents) > 1

    points, _ = generate_random_sites(
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions,
        count=3,
        min_muon_muon=1.0,
        min_muon_atom=0.5,
        max_attempts=50000,
        seed=11,
        include_symmetry_equivalents=True,
    )

    orbits = [
        symmetry_equivalent_positions(
            point.fractional,
            lattice_vectors=lattice,
            host_species=host_species,
            host_positions_frac=host_positions,
        )
        for point in points
    ]
    for index, orbit in enumerate(orbits):
        for other_orbit in orbits[index + 1 :]:
            assert min(_pbc_distance(a, b, lattice) for a in orbit for b in other_orbit) >= 1.0


def test_generate_random_sites_symmetry_without_species_falls_back() -> None:
    # include_symmetry_equivalents requested but host_species is None: the
    # operations branch is skipped and generation still succeeds.
    points, _attempts = generate_random_sites(
        lattice_vectors=CUBIC_8,
        host_positions_frac=[],
        count=2,
        min_muon_muon=0.8,
        min_muon_atom=0.0,
        max_attempts=5000,
        seed=3,
        include_symmetry_equivalents=True,
        host_species=None,
    )
    assert len(points) == 2


def test_random_generation_keeps_whole_orbits_apart_when_identity_is_not_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = {
        "lattice_vectors": _CUBIC,
        "host_species": ["Si", "Si"],
        "host_positions_frac": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    }
    real = site_generation._symmetry_operations

    def identity_last(**kwargs):
        rotations, translations = real(**kwargs)
        return rotations[::-1], translations[::-1]

    monkeypatch.setattr(site_generation, "_symmetry_operations", identity_last)

    points, _attempts = generate_random_sites(
        **host,
        min_muon_muon=0.8,
        min_muon_atom=1.0,
        max_attempts=300,
        seed=3,
        include_symmetry_equivalents=True,
    )

    assert points
    orbits = [
        spglib_compat.symmetry_orbit(
            point.fractional, identity_last(**host), lattice_vectors=_CUBIC, tolerance=1e-5
        )
        for point in points
    ]
    for orbit in orbits:
        for image in orbit:
            assert all(
                _pbc_distance(image, atom, _CUBIC) >= 1.0 for atom in host["host_positions_frac"]
            )
    for first, second in combinations(range(len(orbits)), 2):
        assert all(
            _pbc_distance(a, b, _CUBIC) >= 0.8 for a in orbits[first] for b in orbits[second]
        )


def test_random_generation_reports_progress_as_sites_are_accepted() -> None:
    updates: list[tuple[int, int]] = []

    points, _attempts = generate_random_sites(
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        host_positions_frac=[],
        count=4,
        min_muon_muon=0.8,
        min_muon_atom=0.0,
        max_attempts=5000,
        seed=4,
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert len(points) == 4
    assert updates[0] == (0, 4)
    assert updates[-1] == (4, 4)
    assert [completed for completed, _total in updates] == [0, 1, 2, 3, 4]


def test_generate_random_sites_can_return_partial_results_when_enabled() -> None:
    lattice = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]

    points, attempts = generate_random_sites(
        lattice_vectors=lattice,
        host_positions_frac=[],
        count=200,
        min_muon_muon=0.8,
        min_muon_atom=0.0,
        max_attempts=5000,
        seed=7,
        allow_partial=True,
    )

    assert 0 < len(points) < 200
    assert attempts == 5000


def test_invalid_generation_arguments_raise() -> None:
    lattice = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]
    with pytest.raises(SiteGenerationError):
        generate_random_sites(
            lattice_vectors=lattice,
            host_positions_frac=[],
            count=0,
            min_muon_muon=0.5,
            min_muon_atom=0.5,
        )

    with pytest.raises(SiteGenerationError):
        generate_attached_element_sites(
            lattice_vectors=lattice,
            host_species=["O"],
            host_positions_frac=[[0.0, 0.0, 0.0]],
            target_element="O",
            muons_per_atom=0,
            bond_distance=1.0,
            min_muon_atom=0.5,
        )


def test_generate_random_sites_rejects_negative_distance() -> None:
    with pytest.raises(SiteGenerationError, match="non-negative"):
        generate_random_sites(
            lattice_vectors=TINY_2,
            host_positions_frac=[],
            count=1,
            min_muon_muon=-1.0,
            min_muon_atom=0.5,
        )


def test_random_generation_rejects_malformed_host_positions() -> None:
    with pytest.raises(SiteGenerationError, match="triples"):
        generate_random_sites(
            lattice_vectors=_CUBIC,
            host_positions_frac=[[0.0, 0.0], [0.5, 0.5]],
            count=1,
            min_muon_muon=0.8,
            min_muon_atom=1.0,
        )


def test_generate_random_sites_raises_when_requested_count_cannot_be_met() -> None:
    lattice = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]

    with pytest.raises(SiteGenerationError, match="Could only generate"):
        generate_random_sites(
            lattice_vectors=lattice,
            host_positions_frac=[],
            count=200,
            min_muon_muon=0.8,
            min_muon_atom=0.0,
            max_attempts=5000,
            seed=7,
        )


def test_generate_random_sites_raises_when_nothing_can_be_placed() -> None:
    # A huge muon-atom exclusion forbids every candidate, so zero are accepted.
    with pytest.raises(SiteGenerationError, match="Could not generate any"):
        generate_random_sites(
            lattice_vectors=TINY_2,
            host_positions_frac=[[0.5, 0.5, 0.5]],
            count=3,
            min_muon_muon=0.5,
            min_muon_atom=5.0,
            max_attempts=200,
            seed=1,
        )


# Random filling (no count)


def test_generate_random_sites_without_a_count_fills_until_attempt_limit() -> None:
    lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
    host_positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    min_muon_atom = 0.8
    min_muon_muon = 1.2

    points, attempts = generate_random_sites(
        lattice_vectors=lattice,
        host_positions_frac=host_positions,
        min_muon_muon=min_muon_muon,
        min_muon_atom=min_muon_atom,
        max_attempts=5000,
        seed=7,
    )

    assert len(points) > 1
    assert attempts == 5000

    for idx, point in enumerate(points):
        for host in host_positions:
            assert _pbc_distance(point.fractional, host, lattice) >= min_muon_atom
        for other in points[idx + 1 :]:
            assert _pbc_distance(point.fractional, other.fractional, lattice) >= min_muon_muon


def test_deprecated_distance_generator_fills_the_cell_like_count_none() -> None:
    options = {
        "lattice_vectors": CUBIC_8,
        "host_positions_frac": [[0.0, 0.0, 0.0]],
        "min_muon_muon": 1.5,
        "min_muon_atom": 1.0,
        "max_attempts": 300,
        "seed": 4,
    }

    with pytest.warns(DeprecationWarning, match="count=None"):
        old_points, old_attempts = generate_random_sites_by_distance(**options)
    points, attempts = generate_random_sites(count=None, **options)

    assert old_attempts == attempts == 300
    assert [point.fractional for point in old_points] == [point.fractional for point in points]


def test_random_filling_uses_the_predicted_progress_total() -> None:
    updates: list[tuple[int, int]] = []

    points, _attempts = generate_random_sites(
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        host_positions_frac=[],
        min_muon_muon=1.4,
        min_muon_atom=0.0,
        max_attempts=500,
        seed=5,
        expected_count=5,
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert len(points) > 0
    assert updates[0] == (0, 5)
    assert updates[-1][0] == len(points)
    assert updates[-1][1] >= 5


def test_random_filling_rejects_nonpositive_muon_muon() -> None:
    with pytest.raises(SiteGenerationError, match="greater than zero"):
        generate_random_sites(
            lattice_vectors=TINY_2,
            host_positions_frac=[],
            min_muon_muon=0.0,
            min_muon_atom=0.5,
        )


def test_random_filling_rejects_negative_muon_atom() -> None:
    with pytest.raises(SiteGenerationError, match="non-negative"):
        generate_random_sites(
            lattice_vectors=TINY_2,
            host_positions_frac=[],
            min_muon_muon=1.0,
            min_muon_atom=-1.0,
        )


def test_random_filling_raises_when_nothing_can_be_placed() -> None:
    with pytest.raises(SiteGenerationError, match="Could not generate any"):
        generate_random_sites(
            lattice_vectors=TINY_2,
            host_positions_frac=[[0.5, 0.5, 0.5]],
            min_muon_muon=0.5,
            min_muon_atom=5.0,
            max_attempts=200,
            seed=1,
        )


@pytest.mark.parametrize("symmetry", [False, True], ids=["no-symmetry", "symmetry"])
@pytest.mark.parametrize(
    ("host_factory", "min_muon_muon"),
    [(zno_host, 2.2), (si_primitive_host, 2.0)],
    ids=["hexagonal-ZnO", "fcc-primitive-Si"],
)
def test_random_generation_keeps_both_distances_in_skewed_cells(
    host_factory, min_muon_muon: float, symmetry: bool
) -> None:
    host = host_factory()
    lattice = host.lattice_vectors
    min_muon_atom = 1.0

    points, _attempts = generate_random_sites(
        lattice_vectors=lattice,
        host_positions_frac=host.fractional_positions,
        host_species=host.species,
        min_muon_muon=min_muon_muon,
        min_muon_atom=min_muon_atom,
        max_attempts=2000,
        seed=0,
        include_symmetry_equivalents=symmetry,
    )

    orbits = [
        symmetry_equivalent_positions(
            point.fractional,
            lattice_vectors=lattice,
            host_species=host.species,
            host_positions_frac=host.fractional_positions,
        )
        if symmetry
        else [point.fractional]
        for point in points
    ]
    # At these spacings one symmetric orbit can fill the cell on its own.
    assert len(points) >= (1 if symmetry else 2)
    for orbit in orbits:
        for image in orbit:
            for atom in host.fractional_positions:
                assert _brute_force_distance(image, atom, lattice) >= min_muon_atom
    for first, second in combinations(orbits, 2):
        for a in first:
            for b in second:
                assert _brute_force_distance(a, b, lattice) >= min_muon_muon


# Attached-element sites


def test_generate_attached_element_sites_uses_host_geometry_per_atom() -> None:
    lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
    host_species = ["O", "Si", "Si", "O"]
    host_positions = [[0.5, 0.5, 0.5], [0.65, 0.5, 0.5], [0.5, 0.65, 0.5], [0.25, 0.25, 0.25]]

    points, attempts = generate_attached_element_sites(
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions,
        target_element="O",
        muons_per_atom=2,
        bond_distance=1.0,
        min_muon_atom=0.5,
    )

    assert len(points) == 4
    assert attempts >= 4
    for anchor in (host_positions[0], host_positions[3]):
        distances = [_pbc_distance(point.fractional, anchor, lattice) for point in points]
        assert sum(distance == pytest.approx(1.0) for distance in distances) == 2


def test_attached_sites_sit_at_the_bond_distance_in_a_non_orthogonal_cell() -> None:
    # Hexagonal axes: converting the bonded Cartesian point back to fractional must use the
    # inverse lattice the right way round, or the site lands away from its anchor.
    a, c = 6.0, 7.0
    lattice = [[a, 0.0, 0.0], [-a / 2.0, a * 3.0**0.5 / 2.0, 0.0], [0.0, 0.0, c]]
    anchor = [1.0 / 3.0, 2.0 / 3.0, 0.25]

    points, _attempts = generate_attached_element_sites(
        lattice_vectors=lattice,
        host_species=["O"],
        host_positions_frac=[anchor],
        target_element="O",
        muons_per_atom=3,
        bond_distance=1.0,
        min_muon_atom=0.5,
    )

    assert len(points) == 3
    for point in points:
        assert _pbc_distance(point.fractional, anchor, lattice) == pytest.approx(1.0)
        assert lattice_cart_to_frac(point.cartesian, lattice) == pytest.approx(
            point.fractional, abs=1e-12
        )


def test_attached_sites_none_lattice_uses_neighbour_geometry() -> None:
    # lattice_vectors=None -> identity cell; the nearby Si neighbour drives the
    # neighbour-direction and summed-away branches before placement succeeds.
    points, attempts = generate_attached_element_sites(
        lattice_vectors=None,
        host_species=["O", "Si"],
        host_positions_frac=[[0.5, 0.5, 0.5], [0.6, 0.5, 0.5]],
        target_element="O",
        muons_per_atom=1,
        bond_distance=0.2,
        min_muon_atom=0.05,
        neighbor_cutoff=2.6,
    )
    assert len(points) == 1
    assert attempts >= 1
    fractional = points[0].fractional
    assert all(0.0 <= component < 1.0 for component in fractional)


def test_generate_attached_element_sites_missing_element_error() -> None:
    lattice = [[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 7.0]]
    with pytest.raises(SiteGenerationError, match="No host atoms"):
        generate_attached_element_sites(
            lattice_vectors=lattice,
            host_species=["O", "Si"],
            host_positions_frac=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            target_element="Fe",
            muons_per_atom=1,
            bond_distance=1.0,
            min_muon_atom=0.6,
        )


def test_generate_attached_element_sites_raises_when_requested_count_cannot_be_met() -> None:
    lattice = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    host_species = ["O"]
    host_positions = [[0.0, 0.0, 0.0]]

    with pytest.raises(SiteGenerationError, match="attached-element candidate"):
        generate_attached_element_sites(
            lattice_vectors=lattice,
            host_species=host_species,
            host_positions_frac=host_positions,
            target_element="O",
            muons_per_atom=1,
            bond_distance=0.5,
            min_muon_atom=0.8,
        )


def test_attached_sites_take_one_atom_from_each_symmetry_equivalent_set() -> None:
    host = cof2_host()

    distinct = symmetry_distinct_atom_indexes(
        lattice_vectors=host.lattice_vectors,
        host_species=host.species,
        host_positions_frac=host.fractional_positions,
        target_element="F",
    )
    points, _attempts = generate_attached_element_sites(
        lattice_vectors=host.lattice_vectors,
        host_species=host.species,
        host_positions_frac=host.fractional_positions,
        target_element="F",
        muons_per_atom=2,
        bond_distance=1.0,
        min_muon_atom=0.8,
    )

    # The four F of CoF2 are one 4f orbit, so only the first carries starts.
    assert distinct == [2]
    assert len(points) == 2
    anchor = host.fractional_positions[2]
    for point in points:
        assert _brute_force_distance(point.fractional, anchor, host.lattice_vectors) == (
            pytest.approx(1.0)
        )


def test_attached_sites_use_every_atom_when_symmetry_is_not_detected() -> None:
    host = cof2_host(cobalt_label="Xx")

    distinct = symmetry_distinct_atom_indexes(
        lattice_vectors=host.lattice_vectors,
        host_species=host.species,
        host_positions_frac=host.fractional_positions,
        target_element="F",
    )

    assert distinct == [2, 3, 4, 5]


def test_attached_sites_rejects_empty_element() -> None:
    with pytest.raises(SiteGenerationError, match="cannot be empty"):
        generate_attached_element_sites(
            lattice_vectors=TINY_2,
            host_species=["O"],
            host_positions_frac=[[0.0, 0.0, 0.0]],
            target_element="   ",
        )


def test_attached_sites_rejects_nonpositive_bond_distance() -> None:
    with pytest.raises(SiteGenerationError, match="Bond distance"):
        generate_attached_element_sites(
            lattice_vectors=TINY_2,
            host_species=["O"],
            host_positions_frac=[[0.0, 0.0, 0.0]],
            target_element="O",
            bond_distance=0.0,
        )


def test_attached_sites_rejects_negative_muon_atom_distance() -> None:
    with pytest.raises(SiteGenerationError, match="non-negative"):
        generate_attached_element_sites(
            lattice_vectors=TINY_2,
            host_species=["O"],
            host_positions_frac=[[0.0, 0.0, 0.0]],
            target_element="O",
            min_muon_atom=-1.0,
        )


def test_attached_sites_raises_when_no_candidate_clears_hosts() -> None:
    # Bond distance places the muon well inside the required min_muon_atom
    # shell, so no candidate is accepted for the single target atom.
    with pytest.raises(SiteGenerationError, match="Could not place any"):
        generate_attached_element_sites(
            lattice_vectors=TINY_2,
            host_species=["O"],
            host_positions_frac=[[0.0, 0.0, 0.0]],
            target_element="O",
            muons_per_atom=1,
            bond_distance=0.4,
            min_muon_atom=2.0,
        )


# Spacing checks


def _violates(point, references, lattice, limit) -> bool:
    return site_generation._violates_spacing(
        np.asarray(point, dtype=float), ((np.asarray(references, dtype=float), limit),), lattice
    )


def test_spacing_check_measures_the_nearest_image_in_a_skewed_cell() -> None:
    lattice = zno_host().lattice_vectors
    point, reference = [0.0, 0.0, 0.0], [[0.45, 0.55, 0.0]]

    # The per-component fold puts these 2.53 A apart; the nearest image is 1.65 A away.
    assert _brute_force_distance(point, reference[0], lattice) == pytest.approx(
        3.25 * math.sqrt(0.2575)
    )
    assert _violates(point, reference, lattice, 2.0) is True
    assert _violates(point, reference, lattice, 1.6) is False
    assert _violates(point, np.empty((0, 3)), _CUBIC, 1.0) is False


def test_spacing_check_is_strict_at_the_limit() -> None:
    point, reference = [0.1, 0.2, 0.3], [[0.4, 0.45, 0.9]]
    limit = float(nearest_image_distance_matrix(point, reference, _CUBIC)[0, 0])

    assert _violates(point, reference, _CUBIC, limit) is False
    assert _violates(point, reference, _CUBIC, math.nextafter(limit, 10.0)) is True


def test_spacing_check_applies_each_limit_to_its_own_references() -> None:
    lattice = zno_host().lattice_vectors
    point = np.asarray([0.0, 0.0, 0.0])
    near = np.asarray([[0.45, 0.55, 0.0]])
    far = np.asarray([[0.5, 0.0, 0.5]])

    # The near reference only breaks the larger limit; the far one breaks neither.
    assert site_generation._violates_spacing(point, ((near, 1.6), (far, 2.0)), lattice) is False
    assert site_generation._violates_spacing(point, ((far, 1.6), (near, 2.0)), lattice) is True
