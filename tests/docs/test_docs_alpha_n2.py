"""The numbers the α-N2 worked example reports, read from the paper's real CASTEP outputs.

``docs/user_guide/examples/alpha_n2.rst`` and the α-N2 passages of
``docs/user_guide/quantum_corrections_theory.rst`` quote the geometry, zero-point
energies, mode frequencies, widths and witnesses that MuFinder's parsers give for
the outputs under ``tests/fixtures/alpha_n2`` (Gomilšek et al., Commun. Phys. 6,
142 (2023)). These tests pin them, so a parser change that moves a number on the
page fails here first.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from mufinder.castep_io.cell_parser import parse_cell_file
from mufinder.castep_io.output_parser import parse_castep_output
from mufinder.domain.models import Structure
from mufinder.science.quantum_phonons import (
    muon_mass_warning,
    muon_zpe,
    parse_phonon_path,
    w1_entanglement_witness,
    w2_entanglement_witness,
)
from mufinder.science.relaxation_analysis import analyze_displacements

FIXTURES = Path(__file__).parents[1] / "fixtures" / "alpha_n2"
PHONON = FIXTURES / "aN2_mu_phonon.phonon"
OUT_CELL = FIXTURES / "aN2_mu_phonon-out.cell"
MUON = 65
# The two nitrogens of the complex nearest the muon, one-based as in the files.
NEAR_N = (57, 1)


def _min_image(fractional: np.ndarray) -> np.ndarray:
    return (fractional + 0.5) % 1.0 - 0.5


@pytest.fixture(scope="module")
def relaxed() -> tuple[np.ndarray, np.ndarray, list[str]]:
    cell = parse_cell_file(OUT_CELL, geometry_only=True)
    return (
        np.asarray(cell.lattice_vectors, dtype=float),
        np.asarray(cell.fractional_positions, dtype=float),
        list(cell.species),
    )


@pytest.mark.parametrize(
    ("name", "md5"),
    [
        ("aN2_mu_phonon.castep", "5e8bc7ed145a58775768be408cc52b84"),
        ("aN2_mu_phonon.cell", "3c01bb5df02790abf3942a05ee2eb595"),
        ("aN2_mu_phonon-out.cell", "7455ac9332b7553f3f48a9160a406385"),
        ("aN2_mu_phonon.param", "55c867c6f745903471b2a1abe2d83e79"),
        ("aN2_mu_phonon.phonon", "6a3ff3a24f88c5b1cb85a30aa724abdd"),
    ],
)
def test_the_fixtures_are_the_data_set_files_unchanged(name: str, md5: str) -> None:
    assert hashlib.md5((FIXTURES / name).read_bytes()).hexdigest() == md5


def test_the_muon_sits_at_the_quarter_quarter_site_in_a_linear_complex(relaxed) -> None:
    lattice, fractional, species = relaxed
    muon = fractional[MUON - 1]
    distances = np.linalg.norm(_min_image(fractional - muon) @ lattice, axis=1)

    assert species.count("N") == 64 and species[MUON - 1] == "H:mu"
    assert lattice[0][0] / 2.0 == pytest.approx(5.66353, abs=5e-6)
    # The 2x2x2 supercell's (0, 1/8, 1/8) is the conventional (0, 1/4, 1/4).
    assert np.allclose((2.0 * muon) % 1.0, (0.0018, 0.2498, 0.2482), atol=5e-5)
    assert [distances[index - 1] for index in NEAR_N] == pytest.approx((1.268, 1.288), abs=5e-4)
    # Every other nitrogen is at least 2.36 Å away.
    others = np.delete(distances, [MUON - 1, *(index - 1 for index in NEAR_N)])
    assert others.min() == pytest.approx(2.367, abs=5e-4)


def _molecules(lattice: np.ndarray, fractional: np.ndarray, species: list[str]):
    """Each N2 molecule as (its centre's distance from the muon in Å, its bond vector in Å)."""

    nitrogens = [index for index, name in enumerate(species) if name == "N"]
    muon = fractional[species.index("H:mu")]
    molecules = []
    for index in nitrogens:
        separations = _min_image(fractional[nitrogens] - fractional[index]) @ lattice
        lengths = np.linalg.norm(separations, axis=1)
        lengths[nitrogens.index(index)] = np.inf
        partner = nitrogens[int(np.argmin(lengths))]
        if partner < index:
            continue
        bond = _min_image(fractional[partner] - fractional[index])
        centre = fractional[index] + bond / 2.0
        molecules.append(
            (float(np.linalg.norm(_min_image(centre - muon) @ lattice)), bond @ lattice)
        )
    return molecules


def test_the_complex_bonds_shorten_and_every_molecule_turns(relaxed) -> None:
    molecules = _molecules(*relaxed)
    complex_bonds = [np.linalg.norm(bond) for distance, bond in molecules if distance < 2.0]
    outer_bonds = [np.linalg.norm(bond) for distance, bond in molecules if distance > 4.5]
    axes = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]) / np.sqrt(3.0)
    tilts = [
        np.degrees(np.arccos(np.max(np.abs(axes @ (bond / np.linalg.norm(bond))))))
        for _distance, bond in molecules
    ]

    assert len(molecules) == 32
    assert complex_bonds == pytest.approx((1.0994, 1.0994), abs=5e-5)
    assert len(outer_bonds) == 22
    assert np.mean(outer_bonds) == pytest.approx(1.1037, abs=5e-5)
    # The charge of the complex turns every molecule of the supercell off its
    # <111> axis, by 16 to 50 degrees.
    assert min(tilts) == pytest.approx(16.0, abs=0.5)
    assert max(tilts) == pytest.approx(50.0, abs=0.5)


def test_the_displacements_against_the_pa3_host_of_the_calculation(relaxed) -> None:
    """The host of the screenshots: the calculation's cell, x from the 1.1037 Å bond."""

    lattice, fractional, species = relaxed
    edge = lattice[0][0] / 2.0
    x = 1.1037 / (2.0 * np.sqrt(3.0) * edge)
    molecules = (
        ((0.0, 0.0, 0.0), (1, 1, 1)),
        ((0.5, 0.0, 0.5), (-1, -1, 1)),
        ((0.0, 0.5, 0.5), (-1, 1, -1)),
        ((0.5, 0.5, 0.0), (1, -1, -1)),
    )
    host = Structure(
        label="alpha-N2",
        lattice_vectors=(edge * np.eye(3)).tolist(),
        species=["N"] * 8,
        fractional_positions=[
            ((np.asarray(centre) + sign * x * np.asarray(axis)) % 1.0).tolist()
            for centre, axis in molecules
            for sign in (1.0, -1.0)
        ],
    )
    muon = (2.0 * fractional[MUON - 1]) % 1.0
    result = analyze_displacements(
        relaxed_structure=Structure(
            label="relaxed",
            lattice_vectors=lattice.tolist(),
            species=species,
            fractional_positions=fractional.tolist(),
        ),
        comparison_structure=host,
        muon_fractional_position=muon.tolist(),
    )
    radial = [point.radial_displacement_ang for point in result.points]
    moved = [point.displacement_ang for point in result.points]

    assert x == pytest.approx(0.0563, abs=5e-5)
    assert len(result.points) == 64
    assert max(point.distance_from_muon_ang for point in result.points) < 8.5
    assert min(radial) == pytest.approx(-0.86, abs=5e-3)
    assert max(radial) == pytest.approx(0.57, abs=5e-3)
    assert np.mean(moved) == pytest.approx(0.47, abs=5e-3)
    assert max(moved) == pytest.approx(0.93, abs=5e-3)


def test_the_phonon_run_finished_and_used_the_muon_mass() -> None:
    result = parse_castep_output(FIXTURES / "aN2_mu_phonon.castep")
    calculation = parse_phonon_path(PHONON)
    muon = calculation.atoms[MUON - 1]

    assert result.completed is True
    assert result.error_message is None
    assert muon.mass_amu == pytest.approx(0.113429)
    assert muon_mass_warning(calculation) is None


def test_the_harmonic_muon_modes_widths_and_zero_point_energy() -> None:
    calculation = parse_phonon_path(PHONON)
    zpe = muon_zpe(calculation)

    # The three most muon-like modes are the three highest of the 195.
    assert [mode.mode_index for mode in zpe.modes] == [193, 194, 195]
    assert [mode.frequency_cm for mode in zpe.modes] == pytest.approx(
        (2749.6, 3246.4, 3301.1), abs=0.05
    )
    assert [mode.zpe_eV for mode in zpe.modes] == pytest.approx((0.1705, 0.2013, 0.2046), abs=5e-5)
    assert [mode.width_A for mode in zpe.modes] == pytest.approx((0.232, 0.214, 0.212), abs=5e-4)
    assert [mode.participation for mode in zpe.modes] == pytest.approx(
        (0.878, 0.992, 0.992), abs=5e-4
    )
    assert zpe.total_eV == pytest.approx(0.576, abs=5e-4)
    assert zpe.warnings == ()
    assert w1_entanglement_witness(calculation) == pytest.approx(-0.1377, abs=5e-5)


def test_the_hybridised_stretch_and_the_unstable_librations() -> None:
    calculation = parse_phonon_path(PHONON)
    gamma = calculation.gamma_point()
    frequencies = np.asarray(gamma.frequencies_cm)
    participation = np.sum(np.abs(np.asarray(gamma.eigenvectors)[:, MUON - 1, :]) ** 2, axis=1)

    # Modes 161 to 192 are the stretches of the 32 molecules. The lowest,
    # mode 161, carries most of the L mode's missing 12 %; 30 more lie from
    # 2355.7 to 2363.5 cm^-1, and mode 192 stands apart at 2428.7 cm^-1.
    assert frequencies[159] < 500.0
    assert frequencies[160] == pytest.approx(2345.2, abs=0.05)
    assert participation[160] == pytest.approx(0.118, abs=5e-4)
    assert int(np.argsort(participation)[-4]) == 160
    assert frequencies[161:191] == pytest.approx(np.clip(frequencies[161:191], 2355.65, 2363.55))
    assert frequencies[191] == pytest.approx(2428.7, abs=0.05)
    assert participation[191] < 1e-3
    # Eighteen librations come out imaginary, the softest at -38.9 cm^-1;
    # together they carry under 0.1 % of the muon's motion.
    assert int(np.sum(frequencies < 0.0)) == 18
    assert frequencies.min() == pytest.approx(-38.9, abs=0.05)
    assert participation[frequencies < 0.0].sum() < 1e-3


@pytest.mark.parametrize(("nitrogen", "w2"), [(57, -0.3105), (1, -0.2984)])
def test_the_muon_correlates_with_its_nearest_nitrogens(nitrogen: int, w2: float) -> None:
    calculation = parse_phonon_path(PHONON)

    assert w2_entanglement_witness(calculation, MUON, nitrogen) == pytest.approx(w2, abs=5e-5)


@pytest.mark.parametrize(("nitrogen", "correlation"), [(57, 0.263), (1, 0.244)])
def test_the_papers_mode_cutoff_gives_its_harmonic_26_percent(
    nitrogen: int, correlation: float
) -> None:
    """The paper's 26 % sums only the modes with a muon participation of at least 1e-4.

    Its analysis notebook in the data set builds the harmonic covariance from
    those 20 of the 195 modes; MuFinder sums every stable mode.
    """

    calculation = parse_phonon_path(PHONON)
    gamma = calculation.gamma_point()
    frequencies = np.asarray(gamma.frequencies_cm)
    vectors = np.asarray(gamma.eigenvectors)
    masses = np.array([atom.mass_amu for atom in calculation.atoms])
    kept = (np.sum(np.abs(vectors[:, MUON - 1, :]) ** 2, axis=1) >= 1e-4) & (frequencies > 0.0)
    weight = np.zeros_like(frequencies)
    weight[kept] = 1.0 / np.sqrt(frequencies[kept])

    def covariance(a: int, b: int) -> float:
        u_a = vectors[:, a - 1, :] * (weight / np.sqrt(masses[a - 1]))[:, None]
        u_b = vectors[:, b - 1, :] * (weight / np.sqrt(masses[b - 1]))[:, None]
        return float(np.real(np.sum(u_a * np.conj(u_b))))

    pearson = covariance(MUON, nitrogen) / np.sqrt(
        covariance(MUON, MUON) * covariance(nitrogen, nitrogen)
    )

    assert int(kept.sum()) == 20
    assert pearson == pytest.approx(correlation, abs=5e-4)
