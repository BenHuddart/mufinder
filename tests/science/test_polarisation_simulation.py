"""Tests for polarisation simulations: settings, crystal orientation, traces and spectra."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.dynamics import calculate_dynamic_polarisation
from mufinder.science.magnetic_fields import (
    calculate_incommensurate_field_distributions,
)
from mufinder.science.nuclear_relaxation import calculate_nuclear_width_tensors
from mufinder.science.polarisation import PolarisationError, calculate_polarisation
from mufinder.science.polarisation_simulation import (
    PolarisationSettings,
    PolarisationSimulation,
    crystal_rotation_from_directions,
    simulate_polarisation,
)
from tests.support.codata import MUON_GAMMA_RAD_S_T
from tests.support.field_results import field_result
from tests.support.mnsi import MNSI_MUON, mnsi_helix, mnsi_options, mnsi_structure

GAMMA = MUON_GAMMA_RAD_S_T * 1e-6  # rad μs⁻¹ T⁻¹
ZERO = (0.0, 0.0, 0.0)
CUBIC = np.eye(3) * 4.0


_field = field_result


# --- settings ---------------------------------------------------------------------------


def test_settings_round_trip_through_json() -> None:
    settings = PolarisationSettings(
        geometry="transverse",
        polarisation_direction=(1.0, 0.0, 0.0),
        detector_direction=(0.0, 1.0, 0.0),
        crystal_along_z_uvw=(1.0, 1.0, 1.0),
        crystal_along_x_uvw=(1.0, -1.0, 0.0),
        added_applied_field_T=(0.0, 0.0, 0.02),
        polycrystalline=True,
        sphere_points=200,
        lambda_z_per_us=0.1,
        sigma_per_us=0.3,
        end_time_us=8.0,
        time_points=801,
        collision_rate_per_us=2.0,
        dynamics_method="fft",
        subtract_baseline=False,
        apodisation_sigma_us=4.0,
        apodisation_lambda_us=10.0,
        padding_factor=4,
    )

    restored = PolarisationSettings.from_json(json.loads(json.dumps(settings.to_json())))

    assert restored == settings


def test_settings_missing_from_a_payload_take_their_defaults() -> None:
    assert PolarisationSettings.from_json({"time_points": 11}) == PolarisationSettings(
        time_points=11
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"geometry": "sideways"}, "geometry"),
        ({"dynamics_method": "slow"}, "dynamics method"),
        ({"polarisation_direction": (0.0, 0.0, 0.0)}, "polarisation_direction"),
        ({"crystal_along_z_uvw": (0.0, float("nan"), 1.0)}, "crystal_along_z_uvw"),
        ({"time_points": 1}, "time_points"),
        ({"sphere_points": 0}, "sphere_points"),
        ({"padding_factor": 100}, "padding_factor"),
        ({"end_time_us": 0.0}, "end_time_us"),
        ({"sigma_per_us": -1.0}, "sigma_per_us"),
        ({"collision_rate_per_us": float("inf")}, "collision_rate_per_us"),
        ({"apodisation_sigma_us": 0.0}, "apodisation_sigma_us"),
    ],
)
def test_invalid_settings_are_refused(changes: dict, message: str) -> None:
    with pytest.raises(PolarisationError, match=message):
        PolarisationSettings(**changes)


def test_malformed_settings_payload_is_refused() -> None:
    with pytest.raises(PolarisationError, match="malformed"):
        PolarisationSettings.from_json({"time_points": 10.5})


# --- crystal orientation ----------------------------------------------------------------


def test_default_orientation_of_a_cubic_cell_is_the_identity() -> None:
    np.testing.assert_allclose(
        crystal_rotation_from_directions(CUBIC, (0, 0, 1), (1, 0, 0)), np.eye(3), atol=1e-12
    )


def test_orientation_puts_the_lattice_directions_on_the_laboratory_axes() -> None:
    hexagonal = np.array([[3.0, 0.0, 0.0], [-1.5, 3.0 * np.sqrt(3) / 2, 0.0], [0.0, 0.0, 5.0]])
    rotation = crystal_rotation_from_directions(hexagonal, (1, 0, 0), (1, 2, 0))

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    a_axis = hexagonal[0] / np.linalg.norm(hexagonal[0])
    np.testing.assert_allclose(rotation @ a_axis, (0.0, 0.0, 1.0), atol=1e-12)
    # [120] is perpendicular to a in the hexagonal cell, so it lies exactly along lab x.
    b_star = np.array([1, 2, 0]) @ hexagonal
    np.testing.assert_allclose(rotation @ (b_star / np.linalg.norm(b_star)), (1, 0, 0), atol=1e-12)


def test_parallel_orientation_directions_are_refused() -> None:
    with pytest.raises(PolarisationError, match="parallel"):
        crystal_rotation_from_directions(CUBIC, (1, 1, 0), (2, 2, 0))


# --- simulations ------------------------------------------------------------------------


def test_zero_field_simulation_matches_the_polarisation_function() -> None:
    sites = [_field((0.0, 0.0, 0.01), label="a"), _field((0.02, 0.0, 0.0), label="b")]
    settings = PolarisationSettings(end_time_us=5.0, time_points=501, lambda_z_per_us=0.2)

    simulation = simulate_polarisation(
        settings, sites, lattice_vectors=CUBIC, site_weights=(1.0, 3.0)
    )

    expected = calculate_polarisation(
        sites, settings.times_us, lambda_z=0.2, site_weights=(1.0, 3.0)
    )
    np.testing.assert_allclose(simulation.static.polarisation, expected.polarisation)
    assert simulation.dynamic is None
    np.testing.assert_allclose(simulation.shown_polarisation, expected.polarisation)
    assert simulation.warnings == ()


def test_crystal_orientation_rotates_the_fields_into_the_laboratory() -> None:
    # A field along the crystal [100], with [100] put along the beam, does not depolarise.
    settings = PolarisationSettings(
        crystal_along_z_uvw=(1, 0, 0), crystal_along_x_uvw=(0, 1, 0), time_points=101
    )

    simulation = simulate_polarisation(settings, [_field((0.05, 0, 0))], lattice_vectors=CUBIC)

    np.testing.assert_allclose(simulation.static.polarisation, 1.0, atol=1e-9)


def test_longitudinal_and_transverse_geometries_take_the_axis_from_the_applied_field() -> None:
    site = _field((0.0, 0.0, 0.01), applied=(0.0, 0.0, 0.01))
    times = PolarisationSettings(time_points=401, end_time_us=4.0).times_us

    lf = simulate_polarisation(
        PolarisationSettings(geometry="longitudinal", time_points=401, end_time_us=4.0),
        [site],
        lattice_vectors=CUBIC,
    )
    tf = simulate_polarisation(
        PolarisationSettings(geometry="transverse", time_points=401, end_time_us=4.0),
        [site],
        lattice_vectors=CUBIC,
    )

    np.testing.assert_allclose(lf.static.polarisation, 1.0, atol=1e-9)
    np.testing.assert_allclose(tf.static.polarisation, np.cos(GAMMA * 0.01 * times), atol=1e-6)


def test_added_applied_field_serves_zero_field_results() -> None:
    settings = PolarisationSettings(
        geometry="transverse",
        added_applied_field_T=(0.0, 0.0, 0.02),
        time_points=401,
        end_time_us=4.0,
    )

    simulation = simulate_polarisation(settings, [_field((0.0, 0.0, 0.0))], lattice_vectors=CUBIC)

    np.testing.assert_allclose(
        simulation.static.polarisation, np.cos(GAMMA * 0.02 * settings.times_us), atol=1e-6
    )
    assert simulation.static.applied_field_T == pytest.approx((0.0, 0.0, 0.02))


@pytest.mark.parametrize(
    ("settings", "sites", "message"),
    [
        (PolarisationSettings(), [], "at least one"),
        (
            PolarisationSettings(),
            [_field((0, 0, 0.02), applied=(0, 0, 0.01))],
            "zero-field geometry",
        ),
        (
            PolarisationSettings(added_applied_field_T=(0.0, 0.0, 0.01)),
            [_field((0, 0, 0.02))],
            "zero-field geometry",
        ),
        (PolarisationSettings(geometry="longitudinal"), [_field((0, 0, 0.02))], "applied field"),
        (
            PolarisationSettings(geometry="longitudinal"),
            [
                _field((0, 0, 0.02), applied=(0, 0, 0.01)),
                _field((0, 0, 0.02), applied=(0, 0, 0.02)),
            ],
            "different applied fields",
        ),
        (PolarisationSettings(sigma_from_nuclear_widths=True), [_field((0, 0, 0.02))], "width"),
    ],
)
def test_simulations_refuse_inconsistent_inputs(settings, sites, message: str) -> None:
    with pytest.raises(PolarisationError, match=message):
        simulate_polarisation(settings, sites, lattice_vectors=CUBIC)


def test_nuclear_widths_damp_each_field_with_its_own_width() -> None:
    host = Structure(
        label="H",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
        species=["H"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    tensors = calculate_nuclear_width_tensors(host, (0.3, 0.4, 0.2), radius=10.0)
    settings = PolarisationSettings(
        sigma_from_nuclear_widths=True, sigma_per_us=5.0, time_points=501, end_time_us=5.0
    )

    simulation = simulate_polarisation(
        settings,
        [_field((0.1, 0.0, 0.0))],
        lattice_vectors=host.lattice_vectors,
        nuclear_width_tensors=[tensors],
    )

    times = settings.times_us
    sigma_squared = tensors.tf_sigma_squared((1.0, 0.0, 0.0))
    expected = np.cos(GAMMA * 0.1 * times) * np.exp(-0.5 * sigma_squared * times**2)
    np.testing.assert_allclose(simulation.static.polarisation, expected, atol=1e-5)


def test_dynamics_act_on_the_weighted_average() -> None:
    sites = [_field((0.0, 0.0, 0.01), label="a"), _field((0.01, 0.0, 0.0), label="b")]
    settings = PolarisationSettings(
        time_points=401, end_time_us=4.0, collision_rate_per_us=1.5, polycrystalline=True
    )

    simulation = simulate_polarisation(settings, sites, lattice_vectors=CUBIC)

    expected = calculate_dynamic_polarisation(
        simulation.static.polarisation, settings.times_us, 1.5
    )
    assert simulation.dynamic is not None
    np.testing.assert_allclose(simulation.dynamic.polarisation, expected.polarisation)
    np.testing.assert_allclose(simulation.shown_polarisation, expected.polarisation)


def test_coarse_time_grid_warns_that_the_largest_field_aliases() -> None:
    settings = PolarisationSettings(end_time_us=10.0, time_points=101)

    simulation = simulate_polarisation(settings, [_field((0.0, 0.0, 0.5))], lattice_vectors=CUBIC)

    assert len(simulation.warnings) == 1
    assert "MHz" in simulation.warnings[0]


def test_simulation_round_trips_and_recomputes_its_spectrum() -> None:
    settings = PolarisationSettings(
        time_points=401,
        end_time_us=4.0,
        collision_rate_per_us=0.5,
        padding_factor=2,
        polycrystalline=True,
    )
    simulation = simulate_polarisation(settings, [_field((0.01, 0.0, 0.01))], lattice_vectors=CUBIC)

    restored = PolarisationSimulation.from_json(
        json.loads(json.dumps(simulation.to_json())), settings
    )

    np.testing.assert_allclose(restored.shown_polarisation, simulation.shown_polarisation)
    np.testing.assert_allclose(restored.spectrum.amplitudes, simulation.spectrum.amplitudes)


def test_malformed_simulation_payload_is_refused() -> None:
    with pytest.raises(PolarisationError, match="malformed"):
        PolarisationSimulation.from_json({"dynamic": None}, PolarisationSettings())


# --- MnSi, the incommensurate-helix worked example (tests/support/mnsi.py) ------------


@pytest.fixture
def mnsi_sites(real_muesr):
    return calculate_incommensurate_field_distributions(
        mnsi_structure(), mnsi_helix(), [MNSI_MUON], options=mnsi_options()
    )


MNSI_SPECTRUM = {
    "end_time_us": 16.0,
    "time_points": 4001,
    "apodisation_sigma_us": 6.0,
    "padding_factor": 4,
}


def _edges_mT(simulation: PolarisationSimulation) -> tuple[float, float]:
    fields_mT = simulation.spectrum.fields_T * 1e3
    amplitudes = simulation.spectrum.amplitudes
    upper = (fields_mT > 150.0) & (fields_mT < 250.0)
    return (
        float(fields_mT[np.argmax(amplitudes)]),
        float(fields_mT[upper][np.argmax(amplitudes[upper])]),
    )


def test_mnsi_four_sites_give_the_two_edged_spectrum(mnsi_sites) -> None:
    assert len(mnsi_sites) == 4
    simulation = simulate_polarisation(
        PolarisationSettings(polycrystalline=True, **MNSI_SPECTRUM),
        mnsi_sites,
        lattice_vectors=mnsi_structure().lattice_vectors,
    )

    lower, upper = _edges_mT(simulation)
    assert lower == pytest.approx(89.0, abs=2.0)
    assert upper == pytest.approx(206.8, abs=2.0)
    assert simulation.warnings == ()


def test_mnsi_powder_average_is_flat_in_field_direction(mnsi_sites) -> None:
    """The zero-field powder trace is the mean of the single-crystal traces along x, y and z.

    In zero field the powder average depends on the fields only through ``|B|``:
    no direction of the local field is preferred, so it equals the average of
    the three orthogonal single-crystal geometries, sample by sample.
    """

    lattice = mnsi_structure().lattice_vectors
    powder = simulate_polarisation(
        PolarisationSettings(polycrystalline=True, **MNSI_SPECTRUM),
        mnsi_sites,
        lattice_vectors=lattice,
    )
    single = [
        simulate_polarisation(
            PolarisationSettings(polarisation_direction=tuple(axis), **MNSI_SPECTRUM),
            mnsi_sites,
            lattice_vectors=lattice,
        ).static.polarisation
        for axis in np.eye(3)
    ]

    np.testing.assert_allclose(powder.static.polarisation, np.mean(single, axis=0), atol=1e-6)
    lower, upper = _edges_mT(powder)
    assert (lower, upper) == pytest.approx((89.0, 206.8), abs=2.0)


def test_an_all_zero_added_field_adds_nothing_to_results_that_carry_one() -> None:
    site = _field((0.0, 0.0, 0.01), applied=(0.0, 0.0, 0.01))
    settings = PolarisationSettings(
        geometry="longitudinal", added_applied_field_T=(0.0, 0.0, 0.0), time_points=101
    )

    simulation = simulate_polarisation(settings, [site], lattice_vectors=CUBIC)

    np.testing.assert_allclose(simulation.static.polarisation, 1.0, atol=1e-9)


def test_powder_in_a_field_warns_for_the_largest_grain_field() -> None:
    # 0.3 T internal across a 0.3 T applied field: the stored total is 0.42 T, but a grain
    # with its internal field along the applied one sees 0.6 T, 81 MHz.
    site = _field((0.3, 0.0, 0.3), applied=(0.0, 0.0, 0.3))
    settings = PolarisationSettings(
        geometry="longitudinal",
        polycrystalline=True,
        sphere_points=50,
        end_time_us=1.0,
        time_points=121,  # Nyquist 60 MHz
    )

    simulation = simulate_polarisation(settings, [site], lattice_vectors=CUBIC)

    assert len(simulation.warnings) == 1
    assert "600 mT" in simulation.warnings[0]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"polycrystalline": "false"}, "polycrystalline"),
        ({"subtract_baseline": 1}, "subtract_baseline"),
    ],
)
def test_settings_payload_needs_real_booleans(payload: dict, message: str) -> None:
    with pytest.raises(PolarisationError, match=message):
        PolarisationSettings.from_json(payload)


def test_simulation_payload_with_string_warnings_is_refused() -> None:
    settings = PolarisationSettings(time_points=11)
    simulation = simulate_polarisation(settings, [_field((0.0, 0.0, 0.01))], lattice_vectors=CUBIC)
    payload = {**simulation.to_json(), "warnings": "aliased"}

    with pytest.raises(PolarisationError, match="warnings"):
        PolarisationSimulation.from_json(payload, settings)


@pytest.mark.parametrize(
    "changes",
    [
        {},  # a single crystal
        {"polycrystalline": True, "detector_direction": (1.0, 0.0, 0.0)},
        {"polycrystalline": True, "geometry": "transverse"},
    ],
)
def test_dynamics_are_refused_where_the_spin_does_not_reduce_to_one_component(changes) -> None:
    # One field across the spin: every redraw gives the same field, so a scalar renewal
    # of the projected cosine would invent relaxation that the model does not have.
    applied = (0.0, 0.0, 0.01) if changes.get("geometry") == "transverse" else ZERO
    site = _field((0.02, 0.0, applied[2]), applied=applied)
    settings = PolarisationSettings(time_points=101, collision_rate_per_us=1.0, **changes)

    with pytest.raises(PolarisationError, match="strong-collision dynamics"):
        simulate_polarisation(settings, [site], lattice_vectors=CUBIC)


@pytest.mark.parametrize("geometry", ["zero_field", "longitudinal"])
def test_dynamics_run_for_a_powder_in_zero_or_longitudinal_field(geometry: str) -> None:
    applied = (0.0, 0.0, 0.01) if geometry == "longitudinal" else ZERO
    site = _field((0.02, 0.0, applied[2]), applied=applied)
    settings = PolarisationSettings(
        geometry=geometry,
        polycrystalline=True,
        sphere_points=50,
        time_points=101,
        collision_rate_per_us=1.0,
    )

    simulation = simulate_polarisation(settings, [site], lattice_vectors=CUBIC)

    assert simulation.dynamic is not None
