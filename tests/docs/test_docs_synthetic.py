"""The CASTEP outputs ``docs_screenshots.synthetic`` writes for the screenshot projects.

The module imports no Qt, so it is imported here directly; the outputs must
read back through the parsers the application uses on real CASTEP output.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mufinder.castep_io.geometry_progress import parse_geometry_progress_text
from mufinder.castep_io.output_parser import parse_castep_output
from mufinder.science.quantum_phonons import (
    muon_zpe,
    parse_phonon_text,
    w1_entanglement_witness,
    w2_entanglement_witness,
    zpe_from_frequency_cm,
)
from mufinder.science.uep import UepField, load_uep_source
from mufinder.services.job_progress import GeometryProgressReader
from tests.support.phonon_fixtures import N2_MU_N2_FORCE_CONSTANTS, n2_mu_n2_modes

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from docs_screenshots.synthetic import (  # noqa: E402
    N2_MU_N2_SPRINGS,
    bfgs_log_text,
    bridged_diatomic_cell,
    bridged_diatomic_phonon_text,
    den_fmt_text,
    final_configuration,
    triatomic_phonon_text,
    uep_castep_text,
)

LATTICE = [[2.6954645, 2.6954645, 0.0], [2.6954645, 0.0, 2.6954645], [0.0, 2.6954645, 2.6954645]]
SPECIES = ["Si", "Si"]
FRACTIONAL = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]


def _flat(rows) -> list[float]:
    return [float(value) for row in rows for value in row]


def _log(**overrides: object) -> str:
    arguments: dict[str, object] = {
        "lattice": LATTICE,
        "species": SPECIES,
        "fractional": FRACTIONAL,
        "iterations": 6,
        "initial_enthalpy": -340.682656,
        "final_enthalpy": -340.953487,
        "initial_force": 1.025682,
        "final_force": 0.0101,
    }
    arguments.update(overrides)
    return bfgs_log_text(**arguments)


def test_a_finished_bfgs_log_reads_as_a_converged_geometry_optimisation(tmp_path: Path) -> None:
    castep = tmp_path / "Si2.castep"
    castep.write_text(_log(), encoding="utf-8")

    result = parse_castep_output(castep)

    assert result.completed
    assert result.geom_converged is True
    assert result.error_message is None
    assert result.total_energy == pytest.approx(-340.953487)
    assert result.enthalpy == pytest.approx(-340.953487)
    assert result.max_force == pytest.approx(0.0101, abs=1e-5)
    assert result.wall_time == pytest.approx(1843.2)
    assert _flat(result.final_lattice_vectors) == pytest.approx(_flat(LATTICE), abs=1e-6)
    lattice, species, fractional = final_configuration(castep)
    assert species == SPECIES
    assert _flat(lattice) == pytest.approx(_flat(LATTICE), abs=1e-6)
    assert _flat(fractional) == pytest.approx(_flat(FRACTIONAL), abs=1e-6)


def test_the_progress_parser_reads_every_iteration_with_falling_energy_and_force() -> None:
    progress = parse_geometry_progress_text(_log())

    assert progress is not None
    assert (progress.method, progress.max_iterations, progress.converged) == ("BFGS", 100, True)
    assert [iteration.iteration for iteration in progress.iterations] == list(range(7))
    enthalpies = [iteration.enthalpy for iteration in progress.iterations]
    assert enthalpies[0] == pytest.approx(-340.682656) and enthalpies[-1] == pytest.approx(-340.953487)
    assert all(later < earlier for earlier, later in zip(enthalpies, enthalpies[1:]))
    drops = [earlier - later for earlier, later in zip(enthalpies, enthalpies[1:])]
    assert all(later < earlier for earlier, later in zip(drops, drops[1:]))
    forces = [criterion.value for iteration in progress.iterations for criterion in iteration.criteria if criterion.name == "|F|max"]
    assert all(later < earlier for earlier, later in zip(forces, forces[1:]))
    assert [criterion.name for criterion in progress.latest.criteria] == ["dE/ion", "|F|max", "|dR|max"]
    assert progress.latest.criteria_met == 3
    assert progress.iterations[0].criteria_met == 0


def test_an_unfinished_bfgs_log_reads_as_a_running_optimisation(tmp_path: Path) -> None:
    castep = tmp_path / "Si2.castep"
    castep.write_text(_log(iterations=3, finished=False, final_force=0.2), encoding="utf-8")

    progress = GeometryProgressReader().read_local("job", [castep])
    result = parse_castep_output(castep)

    assert progress is not None
    assert progress.converged is None
    assert [iteration.iteration for iteration in progress.iterations] == [0, 1, 2, 3]
    assert progress.latest.criteria_met < 3
    assert not result.completed
    assert result.geom_converged is None


def test_a_finished_log_whose_last_force_misses_the_tolerance_is_refused() -> None:
    with pytest.raises(ValueError, match="force_tolerance"):
        _log(final_force=0.2)


@pytest.mark.parametrize("iterations", [0, 101])
def test_an_iteration_count_outside_the_run_is_refused(iterations: int) -> None:
    with pytest.raises(ValueError, match="iteration"):
        _log(iterations=iterations)


# --- CoF2: the F-mu-F phonon, the model density and its single point ----------------

COF2_LATTICE = [[4.695, 0.0, 0.0], [0.0, 4.695, 0.0], [0.0, 0.0, 3.180]]
COF2_SPECIES = ["Co", "Co", "F", "F", "F", "F"]
COF2_X = 0.306
COF2_FRACTIONAL = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [COF2_X, COF2_X, 0.0],
    [1 - COF2_X, 1 - COF2_X, 0.0],
    [0.5 + COF2_X, 0.5 - COF2_X, 0.5],
    [0.5 - COF2_X, 0.5 + COF2_X, 0.5],
]
# Möller et al., PRB 87, 121108 (2013): the F-mu-F modes in CoF2.
FMUF_STRETCH_CM = 585.0
FMUF_MUON_MODES_CM = (3076.0, 3473.0, 4570.0)


def _fmuf_cell() -> SimpleNamespace:
    """A 2x2x2 CoF2 supercell with a muon at 4c and its two nearest F pulled in to 1.17 Å."""

    lattice = 2.0 * np.asarray(COF2_LATTICE)
    species, fractional = [], []
    for shift in np.ndindex(2, 2, 2):
        for symbol, position in zip(COF2_SPECIES, COF2_FRACTIONAL, strict=True):
            species.append(symbol)
            fractional.append((np.asarray(position) + shift) / 2.0)
    fractional = np.asarray(fractional)
    muon = np.array([0.0, 0.25, 0.0])
    separation = (fractional - muon + 0.5) % 1.0 - 0.5
    vectors = separation @ lattice
    distances = np.linalg.norm(vectors, axis=1)
    for index in np.argsort(distances)[:2]:
        fractional[index] = muon + np.linalg.solve(lattice.T, vectors[index] * 1.17 / distances[index])
    return SimpleNamespace(
        lattice_vectors=lattice.tolist(),
        species=[*species, "H:mu"],
        fractional_positions=[*(fractional % 1.0).tolist(), muon.tolist()],
    )


def _fmuf_phonon():
    return parse_phonon_text(
        triatomic_phonon_text(
            _fmuf_cell(),
            stretch_cm=FMUF_STRETCH_CM,
            bend_cm=FMUF_MUON_MODES_CM[:2],
            antisymmetric_cm=FMUF_MUON_MODES_CM[2],
            host_top_cm=600.0,
        )
    )


def test_the_fmuf_phonon_has_orthonormal_eigenvectors_one_per_degree_of_freedom() -> None:
    gamma = _fmuf_phonon().gamma_point()
    count = len(_fmuf_cell().species)

    vectors = np.asarray(gamma.eigenvectors).reshape(3 * count, 3 * count)
    assert len(gamma.frequencies_cm) == 3 * count
    np.testing.assert_allclose(vectors @ vectors.conj().T, np.eye(3 * count), atol=1e-9)
    assert list(gamma.frequencies_cm) == sorted(gamma.frequencies_cm)


def test_the_fmuf_phonon_gives_moller_s_muon_modes_and_zero_point_energy() -> None:
    phonon = _fmuf_phonon()
    zpe = muon_zpe(phonon)

    assert [mode.frequency_cm for mode in zpe.modes] == pytest.approx(FMUF_MUON_MODES_CM, abs=0.01)
    assert all(mode.participation == pytest.approx(0.997, abs=1e-3) for mode in zpe.modes)
    assert zpe.warnings == ()
    # MuFinder's muon ZPE sums the three muon modes; Möller's 0.73 eV also counts
    # the symmetric stretch of the two fluorines.
    assert zpe.total_eV == pytest.approx(0.689, abs=0.01)
    assert zpe.total_eV + zpe_from_frequency_cm(FMUF_STRETCH_CM) == pytest.approx(0.73, abs=0.01)
    assert np.isclose(phonon.gamma_point().frequencies_cm, FMUF_STRETCH_CM).any()


def test_the_model_density_and_single_point_load_as_a_neutral_uep_source(tmp_path: Path) -> None:
    shells = {"Co": ((8.0, 0.22), (7.0, 0.42)), "F": ((2.0, 0.25), (6.0, 0.60))}
    arguments = {"lattice": COF2_LATTICE, "species": COF2_SPECIES, "fractional": COF2_FRACTIONAL}
    (tmp_path / "CoF2.den_fmt").write_text(den_fmt_text(**arguments, grid=(32, 32, 24), shells=shells), encoding="utf-8")
    (tmp_path / "CoF2.castep").write_text(
        uep_castep_text(**arguments, pseudopotentials={"Co": (17.0, 1.8), "F": (7.0, 1.4)}, energy=-5621.8722, wall_time=1843.2),
        encoding="utf-8",
    )

    source = load_uep_source(den_fmt_path=tmp_path / "CoF2.den_fmt", castep_path=tmp_path / "CoF2.castep")

    assert source.density.grid == (32, 32, 24)
    assert float(np.mean(source.density.values[..., 0])) == pytest.approx(62.0)
    assert float(np.min(source.density.values[..., 0])) > 0.0
    assert source.structure.species == tuple(COF2_SPECIES)
    assert [item.ionic_charge for item in source.pseudopotentials] == [17.0, 17.0, 7.0, 7.0, 7.0, 7.0]
    # The UEP of this model density is lowest in the channel along c, at 4d,
    # below the 4c site the positive muon takes once the fluorines relax.
    field = UepField(source)
    lattice = np.asarray(COF2_LATTICE)
    at_4d, at_4c = (float(field.potential(np.asarray(site) @ lattice)[0][0]) for site in ((0.0, 0.5, 0.25), (0.0, 0.5, 0.0)))
    assert at_4d < at_4c


def test_a_density_too_narrow_for_its_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="too narrow"):
        den_fmt_text(COF2_LATTICE, COF2_SPECIES, COF2_FRACTIONAL, grid=(8, 8, 6), shells={"Co": ((17.0, 0.05),), "F": ((7.0, 0.05),)})


# --- alpha-N2: the [N2-mu-N2]+ complex and its phonon ----------------------------------

# Venables and English, Acta Cryst. B 30, 929 (1974): Pa-3, a = 5.660 Å, N at
# 8c (x, x, x). Each molecule's centre and the <111> direction of its bond.
N2_A = 5.660
N2_X = 0.056
N2_MOLECULES = (
    ((0.0, 0.0, 0.0), (1, 1, 1)),
    ((0.5, 0.0, 0.5), (-1, -1, 1)),
    ((0.0, 0.5, 0.5), (-1, 1, -1)),
    ((0.5, 0.5, 0.0), (1, -1, -1)),
)
N2_COMPLEX = {"muon_bond": 1.29, "molecule_bond": 1.115, "neighbour_shift": 0.03, "neighbour_radius": 4.5}
N2_BANDS_CM = {"stretch_band_cm": (2330.0, 2350.0), "libration_band_cm": (30.0, 70.0), "translation_band_cm": (15.0, 95.0)}


def _n2_supercell() -> SimpleNamespace:
    """The 2x2x2 alpha-N2 supercell, 64 N, with a muon at (0, 1/4, 1/4) between two molecules."""

    host = [
        (np.asarray(centre) + sign * N2_X * np.asarray(axis)) % 1.0 for centre, axis in N2_MOLECULES for sign in (1.0, -1.0)
    ]
    fractional = [(position + shift) / 2.0 for shift in np.ndindex(2, 2, 2) for position in host]
    return SimpleNamespace(
        lattice_vectors=(2.0 * N2_A * np.eye(3)).tolist(),
        species=["N"] * len(fractional) + ["H:mu"],
        fractional_positions=[*np.asarray(fractional).tolist(), [0.0, 0.125, 0.125]],
    )


def _n2_relaxed() -> SimpleNamespace:
    cell = _n2_supercell()
    return SimpleNamespace(
        lattice_vectors=cell.lattice_vectors,
        species=cell.species,
        fractional_positions=bridged_diatomic_cell(cell, **N2_COMPLEX),
    )


def _vectors_from_muon(cell: SimpleNamespace) -> np.ndarray:
    fractional = np.asarray(cell.fractional_positions)
    return ((fractional - fractional[-1] + 0.5) % 1.0 - 0.5) @ np.asarray(cell.lattice_vectors)


def _n2_phonon():
    return parse_phonon_text(bridged_diatomic_phonon_text(_n2_relaxed(), **N2_BANDS_CM))


def test_the_muon_bridges_two_molecules_into_a_linear_complex_along_011() -> None:
    vectors = _vectors_from_muon(_n2_relaxed())
    distances = np.linalg.norm(vectors, axis=1)
    order = np.argsort(distances)

    # The muon (itself, at zero) and its two nearest nitrogens, each 1.29 Å away.
    assert distances[order[1]] == pytest.approx(1.29, abs=1e-6)
    assert distances[order[2]] == pytest.approx(1.29, abs=1e-6)
    assert distances[order[3]] == pytest.approx(1.29 + 1.115, abs=1e-6)
    axis = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    for index in order[1:5]:
        assert abs(np.dot(vectors[index], axis)) == pytest.approx(distances[index], abs=1e-6)
    # The molecules around the complex move rigidly 0.03 Å towards the muon.
    original = _vectors_from_muon(_n2_supercell())
    moved = np.linalg.norm(vectors - original, axis=1)
    assert np.count_nonzero(moved > 1e-6) > 4
    assert moved[order[5:]].max() == pytest.approx(0.03, abs=1e-6)


def test_the_complex_phonon_keeps_the_chain_springs_of_the_phonon_fixture() -> None:
    assert N2_MU_N2_SPRINGS == {
        "muon_nitrogen_stretch": N2_MU_N2_FORCE_CONSTANTS.muon_nitrogen_stretch,
        "muon_nitrogen_bend": N2_MU_N2_FORCE_CONSTANTS.muon_nitrogen_bend,
        "nitrogen_stretch": N2_MU_N2_FORCE_CONSTANTS.nitrogen_stretch,
        "nitrogen_bend": N2_MU_N2_FORCE_CONSTANTS.nitrogen_bend,
        "nitrogen_lattice": N2_MU_N2_FORCE_CONSTANTS.nitrogen_lattice,
    }
    frequencies = np.asarray(_n2_phonon().gamma_point().frequencies_cm)
    chain, _vectors = n2_mu_n2_modes()
    for frequency in chain:
        assert np.isclose(frequencies, frequency, atol=1e-6).any()


def test_the_complex_phonon_has_orthonormal_eigenvectors_and_the_n2_bands() -> None:
    gamma = _n2_phonon().gamma_point()
    count = 65

    vectors = np.asarray(gamma.eigenvectors).reshape(3 * count, 3 * count)
    np.testing.assert_allclose(vectors @ vectors.conj().T, np.eye(3 * count), atol=1e-9)
    frequencies = np.asarray(gamma.frequencies_cm)
    assert list(frequencies) == sorted(frequencies)
    # The 30 molecules outside the complex: one stretch each in the band, and
    # their librations and translations below 100 cm^-1.
    assert np.count_nonzero((frequencies >= 2330.0) & (frequencies <= 2350.0)) == 30
    assert np.count_nonzero(frequencies < 100.0) >= 150


def test_the_complex_phonon_gives_the_harmonic_numbers_of_gomilsek_et_al() -> None:
    phonon = _n2_phonon()
    zpe = muon_zpe(phonon)

    assert zpe.warnings == ()
    assert zpe.total_eV == pytest.approx(0.58, abs=0.01)
    assert [round(mode.zpe_eV, 2) for mode in zpe.modes] == [0.17, 0.20, 0.20]
    assert [mode.frequency_cm for mode in zpe.modes] == pytest.approx([2807.05, 3274.32, 3274.32], abs=0.01)
    assert [round(mode.width_A, 2) for mode in zpe.modes] == [0.23, 0.21, 0.21]
    assert w1_entanglement_witness(phonon) == pytest.approx(-0.14, abs=0.01)
    # The muon is the last atom; its nearest nitrogens are the near atoms of the complex.
    distances = np.linalg.norm(_vectors_from_muon(_n2_relaxed()), axis=1)
    nearest = int(np.argsort(distances)[1]) + 1
    assert w2_entanglement_witness(phonon, 65, nearest) == pytest.approx(-0.26, abs=0.01)


def test_a_muon_between_molecules_of_two_species_is_refused() -> None:
    cell = _n2_relaxed()
    distances = np.linalg.norm(_vectors_from_muon(cell), axis=1)
    species = list(cell.species)
    for index in np.argsort(distances)[1:4:2]:
        species[int(index)] = "F"
    with pytest.raises(ValueError, match="share a species"):
        bridged_diatomic_phonon_text(
            SimpleNamespace(lattice_vectors=cell.lattice_vectors, species=species, fractional_positions=cell.fractional_positions),
            **N2_BANDS_CM,
        )
