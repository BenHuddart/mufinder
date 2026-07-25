from __future__ import annotations

import logging

import pytest

from mufinder.domain.models import Structure
from mufinder.science import energy_barriers as eb
from mufinder.science.energy_barriers import (
    _linear_midpoint_atoms,
    _muon_host_clearance,
    _structure_to_atoms,
    enumerate_distinct_muon_paths,
    idpp_intermediate_structure,
    neb_farm_distribution,
    prepare_transition_state_structures,
)


def _host_structure() -> Structure:
    return Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


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
    pristine = _host_structure()
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
    pristine = _host_structure()
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
    pristine = _host_structure()
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
