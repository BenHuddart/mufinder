"""Tests for the shared spglib plumbing in :mod:`mufinder.science.spglib_compat`.

The orbit and cell helpers replaced per-module copies in site generation, Wyckoff
assignment, magnetic fields, magnetic structures and space-group detection; each
caller gets an equivalence test against a copy of the code it used before.
"""

from __future__ import annotations

from ase import Atoms
from ase.data import atomic_numbers
import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science import spglib_compat
from mufinder.science.lattice import minimum_image_distance, minimum_image_distance_matrix
from mufinder.science.magnetic_fields import find_equivalent_muon_sites
from mufinder.science.spglib_compat import (
    SYMMETRY_TOLERANCE_ANG,
    get_crystal_symmetry,
    get_crystal_symmetry_dataset,
    operation_images,
    species_element_symbol,
    species_to_atomic_numbers,
    spglib_cell,
    symmetry_detection_warning,
    symmetry_operations,
    symmetry_orbit,
)
from tests.support.symmetry_hosts import (
    bcc_host,
    cof2_host,
    fcc_primitive_host,
    hcp_host,
    rutile_host,
    triclinic_host,
    zno_host,
)

_HOSTS = [bcc_host, triclinic_host, hcp_host, rutile_host, fcc_primitive_host]
_POINTS = [
    (0.1, 0.1, 0.1),
    (0.25, 0.25, 0.25),
    (0.12, 0.27, 0.41),
    (0.5, 0.0, 0.0),
    (0.0, 0.0, 0.999999999),
    (-1e-17, 0.5, 0.25),
]


@pytest.fixture(autouse=True)
def _fresh_symmetry_cache():
    spglib_compat.clear_symmetry_operations_cache()
    yield
    spglib_compat.clear_symmetry_operations_cache()


def _operations(host: Structure) -> tuple[np.ndarray, np.ndarray]:
    operations = symmetry_operations(spglib_cell(host), symprec=1e-5)
    assert operations is not None
    return operations


# Cells


def test_species_to_atomic_numbers_strips_and_capitalises_and_rejects_non_elements() -> None:
    assert species_to_atomic_numbers(["Fe", " o ", "SI"]) == [26, 8, 14]
    assert species_to_atomic_numbers(["Xx"]) is None
    assert species_to_atomic_numbers(["Fe", "Mu"]) is None
    assert species_to_atomic_numbers(["X"]) is None
    assert species_to_atomic_numbers([]) == []


@pytest.mark.parametrize(
    ("label", "element"),
    [
        ("Co:1", "Co"),
        ("Fe1", "Fe"),
        ("fe2", "Fe"),
        ("H:mu", "H"),
        ("O2a", "O"),
        ("C1", "C"),
        ("Cs", "Cs"),
        (" Sc ", "Sc"),
        ("Xx", None),
        ("1Fe", None),
        ("", None),
    ],
)
def test_labelled_species_read_as_their_element(label: str, element: str | None) -> None:
    assert species_element_symbol(label) == element


def test_labelled_species_host_keeps_its_space_group() -> None:
    plain = symmetry_operations(spglib_cell(cof2_host()), symprec=SYMMETRY_TOLERANCE_ANG)
    labelled = symmetry_operations(
        spglib_cell(cof2_host(cobalt_label="Co:1")), symprec=SYMMETRY_TOLERANCE_ANG
    )
    cif_labelled = spglib_cell(cof2_host(cobalt_label="Co1"))

    assert plain is not None and labelled is not None
    assert len(labelled[0]) == len(plain[0]) == 16
    assert cif_labelled is not None
    assert cif_labelled[2].tolist() == [27, 27, 9, 9, 9, 9]
    assert symmetry_detection_warning(cof2_host(cobalt_label="Co:1")) is None


def test_cif_rounded_zno_is_found_to_be_p63mc() -> None:
    dataset = get_crystal_symmetry_dataset(spglib_cell(zno_host(cif_rounded=True)))

    assert dataset is not None
    assert (dataset.number, dataset.international) == (186, "P6_3mc")
    assert len(dataset.rotations) == 12
    # The default the symmetry callers use, not only spglib's.
    operations = symmetry_operations(
        spglib_cell(zno_host(cif_rounded=True)), symprec=SYMMETRY_TOLERANCE_ANG
    )
    assert operations is not None and len(operations[0]) == 12


def test_symmetry_detection_warning_names_why_the_host_is_treated_as_p1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]

    unknown = symmetry_detection_warning(
        lattice_vectors=lattice, species=["Si", "Xx", "Xx"], fractional_positions=[[0.0] * 3] * 3
    )
    mismatched = symmetry_detection_warning(
        lattice_vectors=lattice, species=["Si", "Si"], fractional_positions=[[0.0, 0.0, 0.0]]
    )
    monkeypatch.setattr(spglib_compat, "get_crystal_symmetry", lambda cell, *, symprec: None)
    undetected = symmetry_detection_warning(
        lattice_vectors=lattice, species=["Si"], fractional_positions=[[0.0, 0.0, 0.0]]
    )

    assert unknown is not None and "'Xx' name no element" in unknown and "P1" in unknown
    assert mismatched is not None and "2 species for 1 positions" in mismatched
    assert undetected is not None and "spglib" in undetected and "0.001" in undetected


def test_spglib_cell_matches_the_ase_numbers_the_magnetic_modules_used() -> None:
    host = rutile_host()

    lattice, positions, numbers = spglib_cell(host)

    ase_numbers = Atoms(
        symbols=host.species, scaled_positions=host.fractional_positions, cell=host.lattice_vectors
    ).numbers
    assert numbers.tolist() == ase_numbers.tolist()
    assert lattice.tolist() == host.lattice_vectors
    assert positions.tolist() == host.fractional_positions


def test_spglib_cell_accepts_parts_and_rejects_unknown_species_or_mismatched_counts() -> None:
    lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]

    cell = spglib_cell(
        lattice_vectors=lattice, species=["si"], fractional_positions=[[0.0, 0.0, 0.0]]
    )

    assert cell is not None
    assert cell[2].tolist() == [atomic_numbers["Si"]]
    assert (
        spglib_cell(lattice_vectors=lattice, species=["Xx"], fractional_positions=[[0.0, 0.0, 0.0]])
        is None
    )
    assert (
        spglib_cell(
            lattice_vectors=lattice, species=["Si", "Si"], fractional_positions=[[0.0, 0.0, 0.0]]
        )
        is None
    )
    with pytest.raises(ValueError, match="needs lattice_vectors"):
        spglib_cell(lattice_vectors=lattice, species=["Si"])


# Symmetry operations


def test_symmetry_operations_match_spglib_and_are_cached_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []
    real = spglib_compat.get_crystal_symmetry

    def counting(cell, *, symprec):
        calls.append(symprec)
        return real(cell, symprec=symprec)

    monkeypatch.setattr(spglib_compat, "get_crystal_symmetry", counting)
    host = bcc_host()

    rotations, translations = symmetry_operations(spglib_cell(host), symprec=1e-5)
    again = symmetry_operations(spglib_cell(bcc_host()), symprec=1e-5)
    symmetry_operations(spglib_cell(host), symprec=1e-3)

    expected = get_crystal_symmetry(spglib_cell(host), symprec=1e-5)
    assert rotations.shape == (96, 3, 3)
    assert np.array_equal(rotations, expected["rotations"])
    assert np.array_equal(translations, expected["translations"])
    assert again is not None and again[0] is rotations
    assert calls == [1e-5, 1e-3]
    assert not rotations.flags.writeable and not translations.flags.writeable


def test_symmetry_operations_are_none_when_spglib_finds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(spglib_compat, "get_crystal_symmetry", lambda cell, *, symprec: None)

    assert symmetry_operations(spglib_cell(bcc_host()), symprec=1e-5) is None


def test_operation_images_apply_every_operation() -> None:
    rotations, translations = _operations(hcp_host())
    point = np.asarray((0.12, 0.27, 0.41))

    images = operation_images(point, rotations, translations)

    expected = [
        np.dot(rotation, point) + translation
        for rotation, translation in zip(rotations, translations)
    ]
    assert images == pytest.approx(np.asarray(expected), abs=1e-15)
    assert operation_images(point, [], []).shape == (0, 3)


# Orbits: one equivalence test per migrated caller


def _site_generation_orbit(position, rotations, translations, lattice):
    """Per-operation images and the greedy pairwise scan site generation used."""

    kept: list[list[float]] = []
    for rotation, translation in zip(rotations, translations):
        image = [
            float(value) % 1.0
            for value in np.dot(rotation, np.asarray(position, dtype=float)) + translation
        ]
        image = [value % 1.0 for value in image]
        if not any(minimum_image_distance(image, existing, lattice) <= 1e-5 for existing in kept):
            kept.append(image)
    return kept


def _wyckoff_orbit(position, rotations, translations, lattice, tolerance):
    """Batched images and the broadcast-only first-wins scan Wyckoff assignment used."""

    images = np.mod(operation_images(position, rotations, translations), 1.0)
    wrapped = list(dict.fromkeys(tuple(float(value) % 1.0 for value in image) for image in images))
    distances = minimum_image_distance_matrix(wrapped, wrapped, lattice)
    covered = np.zeros(len(wrapped), dtype=bool)
    kept = []
    for index in range(len(wrapped)):
        if covered[index]:
            continue
        kept.append(wrapped[index])
        covered |= distances[index] <= tolerance
    return kept


def _magnetic_fields_orbit(structure, position, tolerance):
    """The canonicalise-then-scan loop magnetic-field site expansion used."""

    def canonical(values):
        out = np.mod(np.asarray(values, dtype=float), 1.0)
        out[np.isclose(out, 1.0, atol=1e-10)] = 0.0
        return out

    symmetry = get_crystal_symmetry(spglib_cell(structure))
    unique: list[np.ndarray] = []
    mu = canonical(position)
    for rotation, translation in zip(symmetry["rotations"], symmetry["translations"]):
        candidate = canonical(rotation @ mu + translation)
        delta = np.mod(candidate[None, :] - np.asarray(unique).reshape(-1, 3) + 0.5, 1.0) - 0.5
        if np.any(np.linalg.norm(delta, axis=1) <= tolerance):
            continue
        unique.append(candidate)
    return [tuple(float(value) for value in row) for row in unique]


@pytest.mark.parametrize("host_factory", _HOSTS)
def test_orbit_matches_the_site_generation_scan_bit_for_bit(host_factory) -> None:
    host = host_factory()
    rotations, translations = _operations(host)

    for point in _POINTS:
        orbit = symmetry_orbit(
            point, (rotations, translations), lattice_vectors=host.lattice_vectors, tolerance=1e-5
        )
        assert orbit == _site_generation_orbit(point, rotations, translations, host.lattice_vectors)


@pytest.mark.parametrize("host_factory", _HOSTS)
@pytest.mark.parametrize("tolerance", [1e-5, 1e-3])
def test_orbit_matches_the_wyckoff_scan(host_factory, tolerance) -> None:
    host = host_factory()
    rotations, translations = _operations(host)

    for point in _POINTS:
        orbit = symmetry_orbit(
            point,
            (rotations, translations),
            lattice_vectors=host.lattice_vectors,
            tolerance=tolerance,
        )
        expected = _wyckoff_orbit(point, rotations, translations, host.lattice_vectors, tolerance)
        assert len(orbit) == len(expected)
        for actual_position, expected_position in zip(orbit, expected, strict=True):
            assert actual_position == pytest.approx(expected_position, abs=1e-12)


@pytest.mark.parametrize("host_factory", _HOSTS)
@pytest.mark.parametrize("tolerance", [1e-3, 0.05])
def test_find_equivalent_muon_sites_matches_the_magnetic_fields_scan(
    host_factory, tolerance
) -> None:
    host = host_factory()

    for point in _POINTS:
        actual = find_equivalent_muon_sites(host, point, tolerance=tolerance)
        expected = _magnetic_fields_orbit(host, point, tolerance)
        assert len(actual) == len(expected)
        for actual_position, expected_position in zip(actual, expected, strict=True):
            assert actual_position == pytest.approx(expected_position, abs=1e-12)


def test_orbit_without_operations_is_the_wrapped_position() -> None:
    assert symmetry_orbit([1.25, -0.5, 0.0], None, lattice_vectors=None, tolerance=1e-5) == [
        [0.25, 0.5, 0.0]
    ]
    assert (
        symmetry_orbit(
            [0.1, 0.2, 0.3],
            (np.empty((0, 3, 3)), np.empty((0, 3))),
            lattice_vectors=None,
            tolerance=1e-5,
        )
        == []
    )


def test_orbit_tolerance_units_follow_the_lattice_argument() -> None:
    host = bcc_host()
    operations = _operations(host)
    # 5e-7 off the 12d position: merged images lie within 1e-5 Å (2e-6 fractional) of each other.
    point = (0.25, 0.5000005, 0.0)

    in_angstrom = symmetry_orbit(
        point, operations, lattice_vectors=host.lattice_vectors, tolerance=1e-5
    )
    in_fractions = symmetry_orbit(point, operations, lattice_vectors=None, tolerance=1e-7)

    assert len(in_angstrom) == 12
    assert len(in_fractions) == 48
    closest = min(minimum_image_distance(in_fractions[0], other) for other in in_fractions[1:])
    assert 1e-7 < closest < 2e-6


def test_host_symmetry_reports_the_space_group_and_its_operation_count() -> None:
    symmetry = spglib_compat.host_symmetry(cof2_host())

    assert (symmetry.symbol, symmetry.number, symmetry.operation_count) == ("P4_2/mnm", 136, 16)
    assert symmetry.symprec == SYMMETRY_TOLERANCE_ANG


def test_host_symmetry_without_detected_symmetry_is_p1_with_one_operation() -> None:
    symmetry = spglib_compat.host_symmetry(cof2_host(cobalt_label="Xx"))

    assert (symmetry.symbol, symmetry.number, symmetry.operation_count) == (None, None, 1)
