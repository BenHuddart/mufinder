"""Edge-case and guard coverage for periodic muon site-generation.

These tests complement ``test_site_generation.py`` by exercising the
validation guards, singular-lattice handling, symmetry-fallback branches,
free-volume early returns, and the zero-yield error paths that the primary
suite does not reach.  They are fully deterministic (fixed seeds, no I/O).
"""

from __future__ import annotations

import math

import pytest

from mufinder.science.site_generation import (
    SiteGenerationError,
    estimate_free_volume,
    estimate_muon_muon_separation_for_count,
    estimate_random_site_count_from_separation,
    generate_attached_element_sites,
    generate_random_sites,
    generate_random_sites_by_distance,
    lattice_cart_to_frac,
    lattice_frac_to_cart,
    symmetry_equivalent_positions,
)

CUBIC_8 = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
CUBIC_5 = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
TINY_2 = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
# Determinant zero: rows 0 and 1 are colinear.
SINGULAR = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


# --- _validate_lattice guards ---------------------------------------------


def test_lattice_frac_to_cart_rejects_non_3x3_matrix() -> None:
    with pytest.raises(SiteGenerationError, match="3x3 matrix"):
        lattice_frac_to_cart([0.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_lattice_cart_to_frac_rejects_ragged_rows() -> None:
    with pytest.raises(SiteGenerationError, match="3x3 matrix"):
        lattice_cart_to_frac([1.0, 2.0, 3.0], [[1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# --- singular-matrix inverse guard ----------------------------------------


def test_lattice_cart_to_frac_rejects_singular_lattice() -> None:
    with pytest.raises(SiteGenerationError, match="singular"):
        lattice_cart_to_frac([1.0, 2.0, 3.0], SINGULAR)


# --- symmetry fallback when species cannot be resolved ---------------------


def test_symmetry_equivalent_positions_unknown_species_returns_single_rep() -> None:
    equivalents = symmetry_equivalent_positions(
        [1.1, -0.2, 0.3],
        lattice_vectors=CUBIC_5,
        host_species=["Xx"],
        host_positions_frac=[[0.0, 0.0, 0.0]],
    )
    # No symmetry available -> the wrapped input position is the only rep.
    assert equivalents == [pytest.approx([0.1, 0.8, 0.3])]


def test_symmetry_equivalent_positions_species_count_mismatch_returns_single_rep() -> None:
    equivalents = symmetry_equivalent_positions(
        [0.1, 0.2, 0.3],
        lattice_vectors=CUBIC_5,
        host_species=["Si", "O"],
        host_positions_frac=[[0.0, 0.0, 0.0]],
    )
    assert equivalents == [pytest.approx([0.1, 0.2, 0.3])]


# --- estimate_free_volume early returns and guards ------------------------


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
        lattice_vectors=CUBIC_5,
        host_positions_frac=[],
        exclusion_radius=1.5,
    )
    assert volume == pytest.approx(125.0)


def test_estimate_free_volume_excludes_atom_neighbourhood() -> None:
    full = 125.0
    free = estimate_free_volume(
        lattice_vectors=CUBIC_5,
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


# --- estimate_random_site_count_from_separation guards --------------------


def test_estimate_random_site_count_rejects_nonpositive_muon_muon() -> None:
    with pytest.raises(SiteGenerationError, match="greater than zero"):
        estimate_random_site_count_from_separation(
            lattice_vectors=CUBIC_5,
            host_positions_frac=[],
            min_muon_muon=0.0,
            min_muon_atom=1.0,
        )


def test_estimate_random_site_count_rejects_negative_muon_atom() -> None:
    with pytest.raises(SiteGenerationError, match="non-negative"):
        estimate_random_site_count_from_separation(
            lattice_vectors=CUBIC_5,
            host_positions_frac=[],
            min_muon_muon=1.0,
            min_muon_atom=-0.1,
        )


def test_estimate_random_site_count_applies_symmetry_multiplicity() -> None:
    common = dict(
        lattice_vectors=CUBIC_5,
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


# --- estimate_muon_muon_separation_for_count guards -----------------------


def test_estimate_separation_rejects_nonpositive_target() -> None:
    with pytest.raises(SiteGenerationError, match="Target count"):
        estimate_muon_muon_separation_for_count(
            lattice_vectors=CUBIC_5,
            host_positions_frac=[],
            target_count=0,
            min_muon_atom=1.0,
        )


def test_estimate_separation_rejects_negative_muon_atom() -> None:
    with pytest.raises(SiteGenerationError, match="non-negative"):
        estimate_muon_muon_separation_for_count(
            lattice_vectors=CUBIC_5,
            host_positions_frac=[],
            target_count=5,
            min_muon_atom=-1.0,
        )


def test_estimate_separation_none_lattice_uses_identity_cell() -> None:
    separation = estimate_muon_muon_separation_for_count(
        lattice_vectors=None,
        host_positions_frac=[],
        target_count=2,
        min_muon_atom=0.5,
    )
    assert separation >= 0.1
    assert math.isfinite(separation)


# --- generate_random_sites guards and zero-yield path ---------------------


def test_generate_random_sites_rejects_negative_distance() -> None:
    with pytest.raises(SiteGenerationError, match="non-negative"):
        generate_random_sites(
            lattice_vectors=TINY_2,
            host_positions_frac=[],
            count=1,
            min_muon_muon=-1.0,
            min_muon_atom=0.5,
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


# --- generate_random_sites_by_distance guards and zero-yield path ---------


def test_generate_by_distance_rejects_nonpositive_muon_muon() -> None:
    with pytest.raises(SiteGenerationError, match="greater than zero"):
        generate_random_sites_by_distance(
            lattice_vectors=TINY_2,
            host_positions_frac=[],
            min_muon_muon=0.0,
            min_muon_atom=0.5,
        )


def test_generate_by_distance_rejects_negative_muon_atom() -> None:
    with pytest.raises(SiteGenerationError, match="non-negative"):
        generate_random_sites_by_distance(
            lattice_vectors=TINY_2,
            host_positions_frac=[],
            min_muon_muon=1.0,
            min_muon_atom=-1.0,
        )


def test_generate_by_distance_raises_when_nothing_can_be_placed() -> None:
    with pytest.raises(SiteGenerationError, match="Could not generate any"):
        generate_random_sites_by_distance(
            lattice_vectors=TINY_2,
            host_positions_frac=[[0.5, 0.5, 0.5]],
            min_muon_muon=0.5,
            min_muon_atom=5.0,
            max_attempts=200,
            seed=1,
        )


# --- generate_attached_element_sites guards -------------------------------


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
