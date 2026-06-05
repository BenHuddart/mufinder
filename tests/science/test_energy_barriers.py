from __future__ import annotations

from mufinder.domain.models import Structure
from mufinder.science.energy_barriers import enumerate_distinct_muon_paths, prepare_transition_state_structures


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
