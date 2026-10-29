from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from mufinder.domain.models import Structure
from mufinder.science import energy_barriers as eb
from mufinder.science.energy_barriers import (
    EnergyBarrierError,
    _linear_midpoint_atoms,
    _muon_host_clearance,
    _structure_to_atoms,
    enumerate_distinct_muon_paths,
    idpp_intermediate_structure,
    neb_farm_distribution,
    path_symmetry_uses_magnetic_order,
    prepare_transition_state_structures,
)
from mufinder.science.lattice import nearest_image_distance_matrix
from mufinder.science.spglib_compat import species_to_atomic_numbers, symmetry_operations
from tests.support.structures import BODY_CENTRE, relaxed_muon_supercell, si_pair_host
from tests.support.symmetry_hosts import cof2_host, hcp_host


def _relaxed_structure(label: str, muon_fractional: list[float]) -> Structure:
    return Structure(
        label=label,
        origin="muon_relaxation",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], muon_fractional],
        properties={"atomic_spins": [1.0, -1.0, 0.0]},
    )


def test_enumerate_distinct_muon_paths_returns_ordered_paths() -> None:
    pristine = si_pair_host(label="host", positions=BODY_CENTRE)
    reactant = _relaxed_structure("siteA", [0.25, 0.25, 0.25])
    product = _relaxed_structure("siteB", [0.75, 0.25, 0.25])

    paths = enumerate_distinct_muon_paths(
        pristine_structure=pristine,
        reactant_structure=reactant,
        product_structure=product,
    )

    assert paths
    assert all(path.length_ang > 0.0 for path in paths)
    assert paths[0].start_fractional == (0.25, 0.25, 0.25)


def test_prepare_transition_state_structures_aligns_product_to_path() -> None:
    pristine = si_pair_host(label="host", positions=BODY_CENTRE)
    reactant = _relaxed_structure("siteA", [0.25, 0.25, 0.25])
    product = _relaxed_structure("siteB", [0.75, 0.25, 0.25])

    prepared = prepare_transition_state_structures(
        pristine_structure=pristine,
        reactant_structure=reactant,
        product_structure=product,
        reactant_anchor_fractional=(0.25, 0.25, 0.25),
        product_anchor_fractional=(0.75, 0.25, 0.25),
    )

    assert prepared.reactant.species == reactant.species
    assert prepared.product.species == product.species
    assert prepared.product.fractional_positions[-1] == [0.75, 0.25, 0.25]
    assert prepared.reactant.properties["atomic_spins"] == [1.0, -1.0, 0.0]


def test_enumerate_distinct_muon_paths_respects_max_distance() -> None:
    pristine = si_pair_host(label="host", positions=BODY_CENTRE)
    reactant = _relaxed_structure("siteA", [0.25, 0.25, 0.25])
    product = _relaxed_structure("siteB", [0.75, 0.25, 0.25])

    paths = enumerate_distinct_muon_paths(
        pristine_structure=pristine,
        reactant_structure=reactant,
        product_structure=product,
        max_distance_ang=1.9,
    )

    assert paths == ()


def _overlapping_midpoint_endpoints() -> tuple[Structure, Structure]:
    # A host Cu sits just off the straight muon path so the *linear* midpoint lands almost on
    # top of it (~0.2 A) -- the exact pathology that blew up the Oct->Oct NEB (191 eV/A initial
    # force). Placing the Cu off-axis gives IDPP a well-defined perpendicular escape direction.
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    def endpoint(label: str, muon: list[float]) -> Structure:
        return Structure(
            label=label,
            origin="muon_relaxation",
            lattice_vectors=lattice,
            species=["Cu", "H:mu"],
            fractional_positions=[[0.5, 0.30, 0.25], muon],
            properties={"atomic_spins": [0.0, 0.0]},
        )

    # Avoid an exactly-half-cell hop (0.25->0.75): a 0.5 displacement is min-image ambiguous and
    # wraps the muon the other way round the cell. A 0.30->0.70 hop puts the linear midpoint at
    # x=0.5, right next to the Cu at x=0.5.
    return endpoint("siteA", [0.30, 0.25, 0.25]), endpoint("siteB", [0.70, 0.25, 0.25])


def test_idpp_intermediate_bows_muon_off_host_atom() -> None:
    reactant, product = _overlapping_midpoint_endpoints()

    linear_clearance = _muon_host_clearance(
        _linear_midpoint_atoms(
            _structure_to_atoms(reactant, require_muon=True),
            _structure_to_atoms(product, require_muon=True),
        )
    )
    assert linear_clearance < 0.3  # linear midpoint overlaps the host

    intermediate = idpp_intermediate_structure(reactant, product)

    # Structure integrity: same atoms, same order, muon preserved as a tagged species.
    assert intermediate.species == reactant.species
    assert len(intermediate.fractional_positions) == len(reactant.fractional_positions)

    idpp_clearance = _muon_host_clearance(_structure_to_atoms(intermediate, require_muon=True))
    # IDPP must pull the muon well clear of the host (target Cu-mu distance is ~1 A here),
    # and by contract it never returns something worse than the linear midpoint.
    assert idpp_clearance > 0.8
    assert idpp_clearance >= linear_clearance


def test_idpp_intermediate_preserves_clean_short_hop() -> None:
    # When the linear midpoint is already clear, the result must still be a valid single-muon
    # intermediate that is no worse than linear (the acceptance gate's contract).
    lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]

    def endpoint(label: str, muon: list[float]) -> Structure:
        return Structure(
            label=label,
            origin="muon_relaxation",
            lattice_vectors=lattice,
            species=["Cu", "Cu", "H:mu"],
            fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], muon],
            properties={"atomic_spins": [0.0, 0.0, 0.0]},
        )

    reactant = endpoint("siteA", [0.20, 0.50, 0.50])
    product = endpoint("siteB", [0.30, 0.50, 0.50])

    intermediate = idpp_intermediate_structure(reactant, product)
    assert intermediate.species == reactant.species
    assert _muon_host_clearance(_structure_to_atoms(intermediate, require_muon=True)) > 0.0


def test_idpp_intermediate_warns_and_falls_back_when_optimiser_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A silent fall-back would reintroduce the muon-on-host overlap IDPP exists to prevent, so the
    # failure path must log a warning and still return the linear midpoint (job generation unblocked).
    reactant, product = _overlapping_midpoint_endpoints()

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("optimiser diverged")

    monkeypatch.setattr(eb, "_idpp_central_image", _boom)

    with caplog.at_level(logging.WARNING, logger="mufinder.science.energy_barriers"):
        intermediate = idpp_intermediate_structure(reactant, product)

    assert any("falling back to the linear midpoint" in record.message for record in caplog.records)

    # Result equals the linear midpoint (muon clearance matches, structure still valid).
    linear = _linear_midpoint_atoms(
        _structure_to_atoms(reactant, require_muon=True),
        _structure_to_atoms(product, require_muon=True),
    )
    result_clearance = _muon_host_clearance(_structure_to_atoms(intermediate, require_muon=True))
    assert result_clearance == pytest.approx(_muon_host_clearance(linear))


def test_neb_farm_distribution_matches_castep_block_split() -> None:
    # CASTEP's tssearch_neb_farm_indices gives the first (n mod F) farms one
    # extra image; the rest get the floor. 11 images over 3 farms is 4/4/3.
    distribution = neb_farm_distribution(images=11, farms=3)

    assert distribution.steady_counts == (4, 4, 3)
    # The first sweep also computes the two endpoints: 13 images over 3 farms.
    assert distribution.first_sweep_counts == (5, 4, 4)
    assert distribution.efficiency == pytest.approx(11 / 12)
    assert distribution.idle_farms == 0


def test_neb_farm_distribution_flags_poor_pairings() -> None:
    # 11 images over 5 farms leaves the busiest farm with 3 while others hold 2:
    # over a quarter of the allocation is wasted every sweep.
    assert neb_farm_distribution(images=11, farms=5).efficiency == pytest.approx(11 / 15)
    # Even splits are perfect regardless of size.
    assert neb_farm_distribution(images=12, farms=4).efficiency == pytest.approx(1.0)
    assert neb_farm_distribution(images=11, farms=1).steady_counts == (11,)


def test_neb_farm_distribution_reports_idle_farms() -> None:
    # CASTEP warns (but does not abort) when farms outnumber images.
    distribution = neb_farm_distribution(images=3, farms=5)

    assert distribution.steady_counts == (1, 1, 1, 0, 0)
    assert distribution.idle_farms == 2


# --- Path symmetry: full pair orbits and collinear magnetic order ---------------

COF2_4C = (0.0, 0.5, 0.0)
COF2_SUPERCELL = (2, 2, 2)
HCP_SUPERCELL = (2, 2, 2)
PAIR_ORBIT_CUTOFF_ANG = 4.0


def _cof2_with_spins(spins: list[float] | None) -> Structure:
    host = cof2_host()
    if spins is not None:
        host.properties["atomic_spins"] = [*spins, 0.0, 0.0, 0.0, 0.0]
    return host


def _paths(host: Structure, repeats: Any, start: Any, end: Any) -> list[float]:
    paths = enumerate_distinct_muon_paths(
        pristine_structure=host,
        reactant_structure=relaxed_muon_supercell(host, repeats, start),
        product_structure=relaxed_muon_supercell(host, repeats, end),
        max_distance_ang=PAIR_ORBIT_CUTOFF_ANG,
    )
    return [round(path.length_ang, 3) for path in paths]


def _cof2_paths(host: Structure, start: Any, end: Any) -> list[float]:
    return _paths(host, COF2_SUPERCELL, start, end)


def _periodic(delta: np.ndarray) -> np.ndarray:
    return delta - np.rint(delta)


def _brute_force_path_lengths(
    host: Structure, start: Any, end: Any, repeats: Any = COF2_SUPERCELL
) -> list[float]:
    """Union every (start, end) pair with its images under every crystal operation.

    Written apart from the code under test: one class per pair orbit, with a hop
    and its reverse in one class when both ends lie on one orbit. Each pair is
    the hop to the end's nearest image.
    """

    supercell = relaxed_muon_supercell(host, repeats, start)
    lattice = np.asarray(supercell.lattice_vectors, dtype=float)
    positions = np.asarray(supercell.fractional_positions[:-1], dtype=float)
    numbers = species_to_atomic_numbers(supercell.species[:-1])
    operations = symmetry_operations((lattice, positions, numbers), symprec=1e-3)
    assert operations is not None
    rotations, translations = operations
    counts = np.asarray(repeats, dtype=float)

    def orbit(position: Any) -> np.ndarray:
        scaled = np.asarray(position, dtype=float) / counts
        images = np.mod(rotations @ scaled + translations, 1.0)
        keys = np.mod(np.rint(images * 1e6).astype(np.int64), 1_000_000)
        _keys, first = np.unique(keys, axis=0, return_index=True)
        return images[np.sort(first)]

    def index_in(members: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Index in ``members`` of each position, periodically (-1 if none)."""

        close = np.all(np.abs(_periodic(positions[..., None, :] - members)) < 1e-6, axis=-1)
        return np.where(close.any(axis=-1), close.argmax(axis=-1), -1)

    def images_in(members: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Index in ``members`` of each operation's image of each position (-1 if none)."""

        images = np.einsum("oij,pj->opi", rotations, positions) + translations[:, None, :]
        return index_in(members, images)

    starts, ends = orbit(start), orbit(end)
    start_images, end_images = images_in(starts, starts), images_in(ends, ends)
    same = bool(np.all(index_in(ends, starts) >= 0))
    distances = nearest_image_distance_matrix(starts, ends, lattice)
    first, last = np.nonzero((distances > 1e-4) & (distances <= PAIR_ORBIT_CUTOFF_ANG))
    pair = np.full(distances.shape, -1)
    pair[first, last] = np.arange(len(first))
    # Every operation joins each pair to its image pair (and, for one orbit, to
    # the reversed image); the classes are the connected components.
    image_pairs = [pair[start_images[:, first], end_images[:, last]]]
    if same:
        end_as_start, start_as_end = index_in(starts, ends), index_in(ends, starts)
        image_pairs.append(
            pair[end_as_start[end_images[:, last]], start_as_end[start_images[:, first]]]
        )
    sources = np.concatenate(
        [np.broadcast_to(np.arange(len(first)), image.shape).ravel() for image in image_pairs]
    )
    targets = np.concatenate([image.ravel() for image in image_pairs])
    linked = targets >= 0
    graph = coo_matrix(
        (np.ones(int(linked.sum())), (sources[linked], targets[linked])),
        shape=(len(first), len(first)),
    )
    _count, labels = connected_components(graph, directed=False)
    lengths: dict[int, float] = {}
    for label, i, j in zip(labels, first, last):
        lengths.setdefault(int(label), float(distances[i, j]))
    return sorted(round(length, 3) for length in lengths.values())


def _generic_sites() -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(1)
    return [(float(x), float(y), float(z)) for x, y, z in rng.random((6, 3))]


@pytest.mark.parametrize("site", [COF2_4C, *_generic_sites()[:4]])
def test_same_site_paths_match_brute_force_pair_orbits(site: tuple[float, float, float]) -> None:
    # Keying hops by their rotated hop vector merged hops that no single operation
    # relates (4c: 4 paths for 5 classes; generic sites lost up to 5 each).
    host = _cof2_with_spins(None)

    assert _cof2_paths(host, site, site) == _brute_force_path_lengths(host, site, site)


@pytest.mark.parametrize(
    ("start", "end"),
    [((0.0, 0.0, 0.5), (0.0, 0.0, 0.5)), _generic_sites()[0:2], (_generic_sites()[2],) * 2],
)
def test_hexagonal_paths_match_brute_force_pair_orbits(start: Any, end: Any) -> None:
    # In a 120-degree cell the fractional fold is not the nearest image, so hops
    # are measured to the nearest image; one hop must not come back as several.
    host = hcp_host()

    assert _paths(host, HCP_SUPERCELL, start, end) == _brute_force_path_lengths(
        host, start, end, HCP_SUPERCELL
    )


def test_cross_site_paths_match_brute_force_pair_orbits() -> None:
    host = _cof2_with_spins(None)
    start, end = _generic_sites()[4:]

    assert _cof2_paths(host, start, end) == _brute_force_path_lengths(host, start, end)


def test_cof2_4c_hops_of_equal_length_past_different_fluorines_are_two_paths() -> None:
    # From one 4c site the +-[110] and +-[1-10] hops are both 3.32 A long, but one
    # passes 1.63 A from the nearest F at its midpoint and the other 0.37 A.
    host = _cof2_with_spins(None)

    assert _cof2_paths(host, COF2_4C, COF2_4C) == [1.59, 3.18, 3.32, 3.32, 3.681]


@pytest.mark.parametrize(
    "spins",
    # Spins read from calculated populations differ slightly between sublattices.
    [[3.0, -3.0], [2.971, -2.968]],
)
def test_two_sublattice_antiferromagnet_keeps_the_crystal_paths(spins: list[float]) -> None:
    # Every CoF2 operation keeps the spin pattern or flips it as a whole, and a
    # global flip leaves a collinear calculation's energy unchanged.
    plain = _cof2_with_spins(None)
    antiferromagnet = _cof2_with_spins(spins)
    reactant = relaxed_muon_supercell(antiferromagnet, COF2_SUPERCELL, COF2_4C)

    assert _cof2_paths(antiferromagnet, COF2_4C, COF2_4C) == _cof2_paths(plain, COF2_4C, COF2_4C)
    assert not path_symmetry_uses_magnetic_order(
        pristine_structure=antiferromagnet, reactant_structure=reactant
    )


def _simple_cubic_iron(**properties: Any) -> Structure:
    return Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        properties=properties,
    )


def _cube_centre_hops(host: Structure) -> list[np.ndarray]:
    reactant = relaxed_muon_supercell(host, (4, 4, 4), (0.5, 0.5, 0.5))
    paths = enumerate_distinct_muon_paths(
        pristine_structure=host,
        reactant_structure=reactant,
        product_structure=reactant,
        max_distance_ang=3.1,
    )
    return [_periodic(np.subtract(path.end_fractional, path.start_fractional)) for path in paths]


def test_stripe_order_splits_a_hop_the_non_magnetic_host_merges() -> None:
    # Spins alternate along a and repeat along b and c. The fourfold axes that
    # turn a into b map the stripes onto neither themselves nor their flip, so the
    # cube-centre hop along a and the one across the stripes are distinct.
    plain = _simple_cubic_iron()
    stripe = _simple_cubic_iron(atomic_spins=[2.0], propagation_vector=[0.5, 0.0, 0.0])

    assert len(_cube_centre_hops(plain)) == 1
    hops = _cube_centre_hops(stripe)
    assert len(hops) == 2
    assert sum(abs(hop[0]) > 0.1 for hop in hops) == 1
    assert path_symmetry_uses_magnetic_order(
        pristine_structure=stripe,
        reactant_structure=relaxed_muon_supercell(stripe, (4, 4, 4), (0.5, 0.5, 0.5)),
    )
    assert not path_symmetry_uses_magnetic_order(
        pristine_structure=plain,
        reactant_structure=relaxed_muon_supercell(plain, (4, 4, 4), (0.5, 0.5, 0.5)),
    )


def test_non_collinear_moments_fall_back_to_the_crystal_symmetry() -> None:
    stripe = _simple_cubic_iron(
        atomic_spins=[2.0],
        propagation_vector=[0.5, 0.0, 0.0],
        magnetic_structure={"moments": [{"vector": [1.0, 0.0, 0.0]}, {"vector": [0.0, 1.0, 0.0]}]},
    )

    assert len(_cube_centre_hops(stripe)) == 1


def test_ferrimagnet_paths_cover_only_the_magnetic_orbit_of_the_start() -> None:
    # Unequal Co moments: the operations that swap the two Co sublattices map the
    # spins onto neither pattern, so the 4c sites beside each Co are distinct and
    # the hops to the other kind (1.59 A along c) are not paths from this site.
    ferrimagnet = _cof2_with_spins([3.0, -1.0])
    reactant = relaxed_muon_supercell(ferrimagnet, COF2_SUPERCELL, COF2_4C)

    assert _cof2_paths(ferrimagnet, COF2_4C, COF2_4C) == [3.18, 3.32, 3.32]
    assert path_symmetry_uses_magnetic_order(
        pristine_structure=ferrimagnet, reactant_structure=reactant
    )


def _cof2_4c_hop(host: Structure, length_ang: float) -> Any:
    reactant = relaxed_muon_supercell(host, COF2_SUPERCELL, COF2_4C)
    paths = enumerate_distinct_muon_paths(
        pristine_structure=host,
        reactant_structure=reactant,
        product_structure=reactant,
        max_distance_ang=PAIR_ORBIT_CUTOFF_ANG,
    )
    return next(path for path in paths if round(path.length_ang, 2) == length_ang)


def test_alignment_uses_spin_flipping_operations_in_an_antiferromagnet() -> None:
    # The 1.59 A hop ends beside the other Co sublattice: only an operation that
    # flips every spin moves the relaxed 4c endpoint onto it.
    antiferromagnet = _cof2_with_spins([3.0, -3.0])
    reactant = relaxed_muon_supercell(antiferromagnet, COF2_SUPERCELL, COF2_4C)
    hop = _cof2_4c_hop(antiferromagnet, 1.59)

    prepared = prepare_transition_state_structures(
        pristine_structure=antiferromagnet,
        reactant_structure=reactant,
        product_structure=reactant,
        reactant_anchor_fractional=hop.start_fractional,
        product_anchor_fractional=hop.end_fractional,
    )

    end = np.subtract(prepared.product.fractional_positions[-1], hop.end_fractional)
    assert np.allclose(_periodic(end), 0.0, atol=1e-6)


def test_alignment_refuses_an_anchor_outside_the_magnetic_orbit() -> None:
    ferrimagnet = _cof2_with_spins([3.0, -1.0])
    reactant = relaxed_muon_supercell(ferrimagnet, COF2_SUPERCELL, COF2_4C)
    hop = _cof2_4c_hop(_cof2_with_spins([3.0, -3.0]), 1.59)

    with pytest.raises(EnergyBarrierError, match="change the host.s magnetic order"):
        prepare_transition_state_structures(
            pristine_structure=ferrimagnet,
            reactant_structure=reactant,
            product_structure=reactant,
            reactant_anchor_fractional=hop.start_fractional,
            product_anchor_fractional=hop.end_fractional,
        )


def test_stored_hops_match_the_path_they_are_equivalent_to() -> None:
    host = _cof2_with_spins(None)
    reactant = relaxed_muon_supercell(host, COF2_SUPERCELL, COF2_4C)
    paths = enumerate_distinct_muon_paths(
        pristine_structure=host,
        reactant_structure=reactant,
        product_structure=reactant,
        max_distance_ang=PAIR_ORBIT_CUTOFF_ANG,
    )
    shift = np.asarray([0.5, 0.5, 0.5])  # one host lattice vector along each axis
    reversed_hop = (paths[2].end_fractional, paths[2].start_fractional)
    shifted_hop = (
        np.mod(np.add(paths[3].start_fractional, shift), 1.0),
        np.mod(np.add(paths[3].end_fractional, shift), 1.0),
    )
    unrelated = (paths[0].start_fractional, (0.1, 0.2, 0.3))

    assert eb.equivalent_path_indexes(
        pristine_structure=host,
        reactant_structure=reactant,
        product_structure=reactant,
        paths=paths,
        hops=[reversed_hop, shifted_hop, unrelated],
    ) == [2, 3, None]
