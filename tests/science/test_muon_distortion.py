"""Tests for pairing a relaxed muon supercell with its pristine host."""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.muon_distortion import (
    MuonDistortionError,
    equivalent_site_distortion,
    map_muon_distortion,
)
from tests.support.structures import host_structure, relaxed_muon_supercell

_IRON = host_structure(label="Fe", lattice=3.0, species=("Fe",), positions=((0.0, 0.0, 0.0),))
_SITE = (0.5, 0.0, 0.0)


def test_undisplaced_supercell_pairs_every_atom_with_zero_displacement() -> None:
    distortion = map_muon_distortion(_IRON, relaxed_muon_supercell(_IRON, (3, 3, 3), _SITE), _SITE)

    assert distortion.supercell_matrix == ((3, 0, 0), (0, 3, 0), (0, 0, 3))
    assert len(distortion.symbols) == 27
    assert distortion.muon_fractional == pytest.approx(_SITE)
    assert distortion.muon_offset_ang == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(distortion.displacements, 0.0, atol=1e-12)
    # Every ion sits within the correction cell centred on the muon.
    assert np.max(np.abs(distortion.pristine_vectors)) <= 4.5 + 1e-9


def test_displaced_atom_keeps_its_pristine_site_and_lattice_offset() -> None:
    relaxed = relaxed_muon_supercell(_IRON, (3, 3, 3), _SITE, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)})

    distortion = map_muon_distortion(_IRON, relaxed, _SITE)

    moved = np.flatnonzero(np.linalg.norm(distortion.displacements, axis=1) > 1e-9)
    assert moved.size == 1
    row = int(moved[0])
    assert distortion.atom_indices[row] == 0
    assert distortion.cell_offsets[row] == pytest.approx((1.0, 0.0, 0.0))
    assert distortion.pristine_vectors[row] == pytest.approx((1.5, 0.0, 0.0))
    assert distortion.displacements[row] == pytest.approx((0.1, 0.0, 0.0))


def test_rigid_drift_of_the_relaxed_cell_cancels() -> None:
    relaxed = relaxed_muon_supercell(_IRON, (2, 2, 2), _SITE)
    drift = np.array([0.02, -0.01, 0.015])
    drifted = Structure(
        label="drifted",
        lattice_vectors=relaxed.lattice_vectors,
        species=list(relaxed.species),
        fractional_positions=[
            list(np.mod(np.asarray(row) + drift, 1.0)) for row in relaxed.fractional_positions
        ],
        parent_structure_id=_IRON.id,
    )

    distortion = map_muon_distortion(_IRON, drifted, _SITE)

    np.testing.assert_allclose(distortion.displacements, 0.0, atol=1e-9)
    assert distortion.muon_offset_ang == pytest.approx(float(np.linalg.norm(drift * 6.0)))


def test_mapping_ignores_atom_order() -> None:
    relaxed = relaxed_muon_supercell(_IRON, (2, 2, 2), _SITE, {(0.0, 0.0, 0.0): (-0.05, 0.02, 0.0)})
    order = [5, 2, 7, 0, 3, 6, 1, 4, 8]
    shuffled = Structure(
        label="shuffled",
        lattice_vectors=relaxed.lattice_vectors,
        species=[relaxed.species[index] for index in order],
        fractional_positions=[relaxed.fractional_positions[index] for index in order],
        parent_structure_id=_IRON.id,
    )

    expected = map_muon_distortion(_IRON, relaxed, _SITE)
    result = map_muon_distortion(_IRON, shuffled, _SITE)

    def _by_site(distortion):
        return sorted(
            (tuple(np.round(p, 6)), tuple(np.round(d, 6)))
            for p, d in zip(distortion.pristine_vectors, distortion.displacements)
        )

    assert _by_site(result) == _by_site(expected)


def test_site_far_from_the_relaxed_muon_is_rejected() -> None:
    relaxed = relaxed_muon_supercell(_IRON, (2, 2, 2), _SITE)

    with pytest.raises(MuonDistortionError, match="1.500 Å from the muon"):
        map_muon_distortion(_IRON, relaxed, (0.0, 0.0, 0.0))


def test_lattice_translation_of_the_site_is_accepted() -> None:
    relaxed = relaxed_muon_supercell(_IRON, (2, 2, 2), (1.5, 1.0, 0.0))

    distortion = map_muon_distortion(_IRON, relaxed, _SITE)

    assert distortion.muon_offset_ang == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(distortion.displacements, 0.0, atol=1e-12)


@pytest.mark.parametrize(("y", "home_y"), [(-1e-7, 0.0), (-5e-6, 1.0 - 5e-6), (1.0 - 1e-12, 0.0)])
def test_site_just_below_a_cell_face_matches_the_relaxed_muon(y, home_y) -> None:
    # Within 1e-6 below 1 a coordinate is taken as 0; either way the site matches.
    relaxed = relaxed_muon_supercell(_IRON, (3, 3, 3), _SITE)

    distortion = map_muon_distortion(_IRON, relaxed, (0.5, y, 0.0))

    assert distortion.muon_offset_ang == pytest.approx(0.0, abs=3.0 * 5e-6 + 1e-12)
    assert distortion.muon_fractional == pytest.approx((0.5, home_y, 0.0), abs=1e-12)


@pytest.mark.parametrize(
    ("species", "message"),
    [
        (
            ("Fe", "Fe", "Fe", "Fe", "Fe", "Fe", "Fe", "O"),
            "Pristine structure has no atoms of species 'O'",
        ),
        (("Fe",) * 7, "has 7 Fe atom"),
    ],
    ids=["foreign-species", "missing-atom"],
)
def test_composition_must_match_the_pristine_supercell(species, message) -> None:
    relaxed = relaxed_muon_supercell(_IRON, (2, 2, 2), _SITE)
    host_rows = list(range(len(species)))
    altered = Structure(
        label="altered",
        lattice_vectors=relaxed.lattice_vectors,
        species=[*species, "H:mu"],
        fractional_positions=[relaxed.fractional_positions[index] for index in host_rows]
        + [relaxed.fractional_positions[-1]],
        parent_structure_id=_IRON.id,
    )

    with pytest.raises(MuonDistortionError, match=message):
        map_muon_distortion(_IRON, altered, _SITE)


def test_two_atoms_on_one_pristine_site_are_rejected() -> None:
    relaxed = relaxed_muon_supercell(_IRON, (2, 2, 2), _SITE)
    positions = [list(row) for row in relaxed.fractional_positions]
    positions[1] = list(positions[0])
    collided = Structure(
        label="collided",
        lattice_vectors=relaxed.lattice_vectors,
        species=list(relaxed.species),
        fractional_positions=positions,
        parent_structure_id=_IRON.id,
    )

    with pytest.raises(MuonDistortionError, match="not one-to-one"):
        map_muon_distortion(_IRON, collided, _SITE)


def test_unrelated_cell_is_rejected() -> None:
    relaxed = relaxed_muon_supercell(
        host_structure(label="Fe", lattice=3.7, species=("Fe",), positions=((0.0, 0.0, 0.0),)),
        (1, 1, 1),
        _SITE,
    )

    with pytest.raises(MuonDistortionError, match="matching pristine supercell"):
        map_muon_distortion(_IRON, relaxed, _SITE)


@pytest.mark.parametrize(
    ("species", "message"),
    [
        (("Fe",), "does not contain a muon marker"),
        (("H:mu", "mu"), "more than one muon marker"),
        (("H:mu",), "does not contain any host atoms"),
    ],
)
def test_muon_marker_is_validated(species, message) -> None:
    structure = host_structure(
        label="relaxed",
        lattice=3.0,
        species=species,
        positions=[(0.5, 0.0, 0.0), (0.25, 0.25, 0.25)][: len(species)],
        parent_structure_id=_IRON.id,
    )

    with pytest.raises(MuonDistortionError, match=message):
        map_muon_distortion(_IRON, structure, _SITE)


def test_pristine_host_needs_host_atoms() -> None:
    muon_only = host_structure(
        label="empty", lattice=3.0, species=("H:mu",), positions=((0.0, 0.0, 0.0),)
    )

    with pytest.raises(
        MuonDistortionError, match="Pristine structure does not contain any host atoms"
    ):
        map_muon_distortion(muon_only, relaxed_muon_supercell(_IRON, (1, 1, 1), _SITE), _SITE)


def test_errors_use_the_callers_exception_type() -> None:
    with pytest.raises(KeyError):
        map_muon_distortion(
            _IRON, relaxed_muon_supercell(_IRON, (2, 2, 2), _SITE), (0.0, 0.0, 0.0), error=KeyError
        )


def test_equivalent_site_rotates_the_displacement_pattern() -> None:
    relaxed = relaxed_muon_supercell(_IRON, (3, 3, 3), _SITE, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)})
    distortion = map_muon_distortion(_IRON, relaxed, _SITE)

    image = equivalent_site_distortion(_IRON, distortion, (0.0, 0.5, 0.0))
    direct = map_muon_distortion(
        _IRON,
        relaxed_muon_supercell(
            _IRON, (3, 3, 3), (0.0, 0.5, 0.0), {(0.0, 1.0, 0.0): (0.0, 0.1, 0.0)}
        ),
        (0.0, 0.5, 0.0),
    )

    def _by_site(item):
        return sorted(
            (int(index), tuple(np.round(offset, 6)), tuple(np.round(vector, 6)))
            for index, offset, vector in zip(
                item.atom_indices, item.cell_offsets, item.distorted_vectors
            )
        )

    assert image.muon_fractional == pytest.approx((0.0, 0.5, 0.0))
    assert _by_site(image) == _by_site(direct)
    assert equivalent_site_distortion(_IRON, distortion, _SITE) is distortion


def test_equivalent_site_must_be_in_the_symmetry_orbit() -> None:
    distortion = map_muon_distortion(_IRON, relaxed_muon_supercell(_IRON, (2, 2, 2), _SITE), _SITE)

    with pytest.raises(MuonDistortionError, match="not symmetry-equivalent"):
        equivalent_site_distortion(_IRON, distortion, (0.25, 0.0, 0.0))
