"""Additional coverage for magnetic field helpers and validation branches.

These tests deliberately avoid the heavy muesr backend by exercising the pure
helper functions directly and by monkeypatching the engine loader where a
public entry point is required. They focus on currently uncovered branches:
dipolar/contact field assembly helpers, demagnetisation tensors, muon-position
expansion, magnetic-ion resolution, phase sampling, and validation errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.domain.models import MuonSite, Structure
from mufinder.science.demagnetization import DemagnetizingSample
from mufinder.science import magnetic_fields as module
from mufinder.science.magnetic_fields import (
    DipolarTensorResult,
    MagneticFieldDistributionResult,
    MagneticFieldError,
    MagneticFieldOptions,
    MagneticFieldResult,
    calculate_commensurate_fields,
    calculate_finite_dipolar_field,
    classify_magnetic_model,
    is_commensurate_propagation_vector,
)
from mufinder.science.magnetic_structure import (
    AtomSelector,
    ComplexVector,
    FourierTerm,
    MagneticComponent,
    MagneticStructureModel,
)


def _structure() -> Structure:
    return Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def _two_atom_structure() -> Structure:
    return Structure(
        label="FeO",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _model(q=(0.0, 0.0, 0.0)) -> MagneticStructureModel:
    q_coeff = (1.0, 0.0, 0.0) if any(abs(float(v)) > 1e-12 for v in q) else (0.0, 0.0, 0.0)
    return MagneticStructureModel(
        q_basis_rlu=(tuple(float(v) for v in q), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="moment",
                label="Moment",
                kind="custom",
                selector=AtomSelector.all(),
                terms=(FourierTerm(q_coeff, ComplexVector.from_vector([0.0, 0.0, 1.0])),),
            ),
        ),
    )


def _empty_model() -> MagneticStructureModel:
    """A model whose only component carries a zero moment (no Fourier data)."""

    return MagneticStructureModel(
        components=(
            MagneticComponent(
                component_id="zero",
                label="Zero",
                kind="custom",
                selector=AtomSelector.all(),
                terms=(FourierTerm((0.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 0.0])),),
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# Dataclass derived-value helpers
# --------------------------------------------------------------------------- #


def test_magnetic_field_result_total_magnitude_and_json() -> None:
    result = MagneticFieldResult(
        site_label="A",
        muon_position=(0.1, 0.2, 0.3),
        dipolar=(1.0, 0.0, 0.0),
        lorentz=(0.0, 2.0, 0.0),
        contact=(0.0, 0.0, 3.0),
        demag=(0.0, 0.0, -1.0),
        applied=(0.0, 0.0, 1.0),
    )

    assert result.total == pytest.approx((1.0, 2.0, 3.0))
    assert result.magnitude == pytest.approx(np.sqrt(14.0))

    payload = result.to_json()
    assert payload["site_label"] == "A"
    assert payload["dipolar_T"] == [1.0, 0.0, 0.0]
    assert payload["total_T"] == pytest.approx([1.0, 2.0, 3.0])
    assert payload["magnitude_T"] == pytest.approx(np.sqrt(14.0))


def test_distribution_result_magnitudes_and_json() -> None:
    fields = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=float)
    result = MagneticFieldDistributionResult(
        site_label="B",
        muon_position=(0.0, 0.0, 0.0),
        fields_T=fields,
        histogram_centres_T=np.array([0.0, 1.0], dtype=float),
        histogram_counts=np.array([1, 1], dtype=float),
        calculation_parameters={"k": "v"},
    )

    np.testing.assert_allclose(result.magnitudes_T, [5.0, 5.0])
    payload = result.to_json()
    assert payload["fields_T"] == fields.tolist()
    assert payload["mean_T"] == pytest.approx([1.5, 2.0, 2.5])
    assert payload["calculation_parameters"] == {"k": "v"}


def test_dipolar_tensor_result_to_json_round_trips() -> None:
    result = DipolarTensorResult(
        site_label="C",
        muon_position=(0.5, 0.5, 0.5),
        tensor_T_per_muB=((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)),
    )
    payload = result.to_json()
    assert payload["dipolar_tensor_T_per_muB"] == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]


# --------------------------------------------------------------------------- #
# calculate_finite_dipolar_field validation and edge cases
# --------------------------------------------------------------------------- #


def test_finite_dipolar_field_rejects_bad_position_shape() -> None:
    with pytest.raises(MagneticFieldError, match="positions_cart_ang must have shape"):
        calculate_finite_dipolar_field(
            positions_cart_ang=[[0.0, 0.0]],
            moments_mu_b=[[0.0, 0.0]],
            muon_cart_ang=[0.0, 0.0, 1.0],
        )


def test_finite_dipolar_field_rejects_mismatched_moments() -> None:
    with pytest.raises(MagneticFieldError, match="moments_mu_b must match"):
        calculate_finite_dipolar_field(
            positions_cart_ang=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            moments_mu_b=[[0.0, 0.0, 1.0]],
            muon_cart_ang=[0.0, 0.0, 1.0],
        )


def test_finite_dipolar_field_zero_when_muon_coincides_with_only_moment() -> None:
    field = calculate_finite_dipolar_field(
        positions_cart_ang=[[0.0, 0.0, 0.0]],
        moments_mu_b=[[0.0, 0.0, 1.0]],
        muon_cart_ang=[0.0, 0.0, 0.0],
    )
    assert field == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# is_commensurate / classify / rank helpers
# --------------------------------------------------------------------------- #


def test_is_commensurate_handles_zero_vector() -> None:
    assert is_commensurate_propagation_vector([0.0, 0.0, 0.0])


def test_classify_magnetic_model_raises_for_zero_moment_model() -> None:
    with pytest.raises(MagneticFieldError, match="no non-zero Fourier components"):
        classify_magnetic_model(_structure(), _empty_model())


def test_rank_increases_rejects_zero_candidate() -> None:
    assert module._rank_increases([np.array([1.0, 0.0, 0.0])], np.zeros(3), tolerance=1e-9) is False


def test_rank_increases_for_first_and_dependent_vectors() -> None:
    first = np.array([1.0, 0.0, 0.0])
    assert module._rank_increases([], first, tolerance=1e-9) is True
    assert module._rank_increases([first], np.array([2.0, 0.0, 0.0]), tolerance=1e-9) is False
    assert module._rank_increases([first], np.array([0.0, 1.0, 0.0]), tolerance=1e-9) is True


def test_active_component_q_indices_collects_nonzero_axes() -> None:
    class _Model:
        def __init__(self, q_coeffs):
            self.q_coeffs = q_coeffs

    models = [
        _Model([(1.0, 0.0, 0.0)]),
        _Model([(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]),
        _Model([(0.0, 0.0)]),  # wrong shape -> skipped
        _Model(None),  # falsy q_coeffs -> skipped
    ]
    assert module._active_component_q_indices(models) == (0, 1, 2)


def test_active_phase_basis_returns_empty_for_all_commensurate() -> None:
    class _Model:
        def __init__(self, k):
            self.k_rlu = k

    models = [_Model((0.0, 0.0, 0.5)), _Model((0.5, 0.0, 0.0))]
    basis, coeffs = module._active_phase_basis_and_coefficients(models, tolerance=1e-9, max_period=64)
    assert basis.shape == (0, 3)
    assert coeffs == [(), ()]


def test_active_phase_basis_assigns_zero_coeff_to_commensurate_components() -> None:
    class _Model:
        def __init__(self, k):
            self.k_rlu = k

    models = [_Model((0.12345, 0.0, 0.0)), _Model((0.0, 0.0, 0.5))]
    basis, coeffs = module._active_phase_basis_and_coefficients(models, tolerance=1e-9, max_period=8)
    assert basis.shape == (1, 3)
    assert coeffs[0] == pytest.approx((1.0,))
    assert coeffs[1] == (0.0,)


# --------------------------------------------------------------------------- #
# Phase sampling
# --------------------------------------------------------------------------- #


def test_build_phase_samples_returns_singleton_for_zero_dimensions() -> None:
    phases = module._build_phase_samples_turns(
        n_dim=0,
        nangles=4,
        phase_sample_count=10,
        sampling="auto",
        random_seed=0,
        max_grid_samples=100,
    )
    assert phases.shape == (1, 0)


def test_build_phase_samples_montecarlo_is_deterministic_with_seed() -> None:
    kwargs = dict(
        n_dim=2,
        nangles=4,
        phase_sample_count=7,
        sampling="montecarlo",
        random_seed=123,
        max_grid_samples=100,
    )
    a = module._build_phase_samples_turns(**kwargs)
    b = module._build_phase_samples_turns(**kwargs)
    assert a.shape == (7, 2)
    np.testing.assert_array_equal(a, b)


def test_build_phase_samples_random_alias_maps_to_montecarlo() -> None:
    phases = module._build_phase_samples_turns(
        n_dim=1,
        nangles=4,
        phase_sample_count=5,
        sampling="random",
        random_seed=0,
        max_grid_samples=100,
    )
    assert phases.shape == (5, 1)


def test_build_phase_samples_rejects_unknown_sampling() -> None:
    with pytest.raises(MagneticFieldError, match="phase_sampling must be one of"):
        module._build_phase_samples_turns(
            n_dim=1,
            nangles=4,
            phase_sample_count=5,
            sampling="bogus",
            random_seed=0,
            max_grid_samples=100,
        )


def test_build_phase_samples_rejects_nonpositive_nangles_and_count() -> None:
    with pytest.raises(MagneticFieldError, match="nangles must be positive"):
        module._build_phase_samples_turns(
            n_dim=1, nangles=0, phase_sample_count=5, sampling="grid",
            random_seed=0, max_grid_samples=100,
        )
    with pytest.raises(MagneticFieldError, match="phase_sample_count must be positive"):
        module._build_phase_samples_turns(
            n_dim=1, nangles=4, phase_sample_count=0, sampling="grid",
            random_seed=0, max_grid_samples=100,
        )


def test_build_phase_samples_grid_too_large_raises() -> None:
    with pytest.raises(MagneticFieldError, match="grid phase sampling is too large"):
        module._build_phase_samples_turns(
            n_dim=2,
            nangles=4096,
            phase_sample_count=1_000_000,
            sampling="grid",
            random_seed=0,
            max_grid_samples=4,
        )


def test_build_phase_samples_grid_produces_expected_axis_grid() -> None:
    phases = module._build_phase_samples_turns(
        n_dim=2,
        nangles=4,
        phase_sample_count=9,
        sampling="grid",
        random_seed=0,
        max_grid_samples=100,
    )
    # axis_count = min(nangles=4, floor(9**0.5)=3) = 3 -> 3*3 grid
    assert phases.shape == (9, 2)
    np.testing.assert_allclose(np.unique(phases[:, 0]), [0.0, 1.0 / 3.0, 2.0 / 3.0])


def test_grid_axis_count_clamps_to_at_least_one() -> None:
    assert module._grid_axis_count(n_dim=3, nangles=4, phase_sample_count=1) == 1
    assert module._grid_axis_count(n_dim=1, nangles=2, phase_sample_count=1000) == 2


# --------------------------------------------------------------------------- #
# _combine_component_samples
# --------------------------------------------------------------------------- #


def test_combine_component_samples_rejects_non_2d_phases() -> None:
    with pytest.raises(MagneticFieldError, match="phase_samples_turns must have shape"):
        module._combine_component_samples([], [], np.zeros(3), nangles=4)


def test_combine_component_samples_rejects_bad_lookup_shape() -> None:
    phases = np.zeros((2, 1), dtype=float)
    with pytest.raises(MagneticFieldError, match="component field lookup must have shape"):
        module._combine_component_samples([np.zeros((4, 2))], [(1.0,)], phases, nangles=4)


def test_combine_component_samples_rejects_coeff_length_mismatch() -> None:
    phases = np.zeros((2, 2), dtype=float)
    with pytest.raises(MagneticFieldError, match="phase coefficient length"):
        module._combine_component_samples([np.zeros((4, 3))], [(1.0,)], phases, nangles=4)


def test_combine_component_samples_indexes_lookup_by_phase() -> None:
    lookup = np.arange(12, dtype=float).reshape(4, 3)
    phases = np.array([[0.0], [0.25], [0.5], [0.75]], dtype=float)
    combined = module._combine_component_samples([lookup], [(1.0,)], phases, nangles=4)
    np.testing.assert_allclose(combined, lookup)


# --------------------------------------------------------------------------- #
# Histogram helper
# --------------------------------------------------------------------------- #


def test_field_magnitude_histogram_rejects_nonpositive_bins() -> None:
    with pytest.raises(MagneticFieldError, match="histogram_bins must be positive"):
        module._field_magnitude_histogram(np.array([1.0, 2.0]), bins=0)


def test_field_magnitude_histogram_rejects_non_finite_values() -> None:
    with pytest.raises(MagneticFieldError, match="non-finite magnitudes"):
        module._field_magnitude_histogram(np.array([1.0, np.inf]), bins=4)


def test_field_magnitude_histogram_rejects_empty() -> None:
    with pytest.raises(MagneticFieldError, match="no samples"):
        module._field_magnitude_histogram(np.array([], dtype=float), bins=4)


def test_field_magnitude_histogram_normal_spread() -> None:
    magnitudes = np.linspace(0.0, 1.0, 100)
    centres, counts = module._field_magnitude_histogram(magnitudes, bins=10)
    assert centres.shape == (10,)
    assert int(np.sum(counts)) == magnitudes.size


def test_field_magnitude_histogram_too_many_bins_fallback() -> None:
    magnitudes = np.array([0.5, 0.5, 0.5], dtype=float)
    # Requesting a huge number of bins on a near-delta range triggers the
    # numpy "Too many bins" ValueError, which the helper recovers from.
    centres, counts = module._field_magnitude_histogram(magnitudes, bins=10_000_000)
    assert centres.shape == (10_000_000,)
    assert int(np.sum(counts)) == magnitudes.size


# --------------------------------------------------------------------------- #
# Demagnetisation helpers
# --------------------------------------------------------------------------- #


def test_demag_tensor_none_for_default_options() -> None:
    assert module._demag_tensor(MagneticFieldOptions()) is None


def test_demag_tensor_sphere() -> None:
    tensor = module._demag_tensor(MagneticFieldOptions(demag_shape="sphere"))
    np.testing.assert_allclose(tensor, np.eye(3) / 3.0)


@pytest.mark.parametrize(
    "axis,index",
    [("a", 0), ("x", 0), ("b", 1), ("y", 1), ("c", 2), ("z", 2), ("unknown", 2)],
)
def test_demag_tensor_plate_axes(axis: str, index: int) -> None:
    tensor = module._demag_tensor(
        MagneticFieldOptions(demag_shape="infinite_plate", plate_normal_axis=axis)
    )
    expected = np.zeros((3, 3))
    expected[index, index] = 1.0
    np.testing.assert_allclose(tensor, expected)


def test_demag_tensor_rejects_unknown_shape() -> None:
    with pytest.raises(MagneticFieldError, match="unknown demagnetizing shape"):
        module._demag_tensor(MagneticFieldOptions(demag_shape="cube"))


def test_demag_tensor_prefers_explicit_sample() -> None:
    options = MagneticFieldOptions(
        demag_shape="sphere",
        demag_sample=DemagnetizingSample(shape="ellipsoid", semi_axes=(1.0, 1.0, 1.0)),
    )
    tensor = module._demag_tensor(options)
    np.testing.assert_allclose(tensor, np.eye(3) / 3.0, atol=1e-9)


def test_demag_field_zero_without_tensor() -> None:
    field = module._demag_field(np.array([1.0, 2.0, 3.0]), MagneticFieldOptions())
    np.testing.assert_array_equal(field, np.zeros(3))


def test_demag_field_applies_sphere_factor() -> None:
    field = module._demag_field(
        np.array([3.0, 0.0, 0.0]), MagneticFieldOptions(demag_shape="sphere")
    )
    np.testing.assert_allclose(field, [-1.5 * (1.0 / 3.0) * 3.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# Field-vector / field-array extraction
# --------------------------------------------------------------------------- #


def test_field_vector_uses_fallback_attribute() -> None:
    obj = type("F", (), {"D": np.array([1.0, 2.0, 3.0])})()
    np.testing.assert_array_equal(module._field_vector(obj, "Dipolar", "D"), [1.0, 2.0, 3.0])


def test_field_vector_returns_zero_when_missing() -> None:
    obj = type("F", (), {})()
    np.testing.assert_array_equal(module._field_vector(obj, "Dipolar", "D"), np.zeros(3))


def test_field_vector_flattens_2d_input() -> None:
    obj = type("F", (), {"Dipolar": np.array([[1.0, 2.0, 3.0]])})()
    np.testing.assert_array_equal(module._field_vector(obj, "Dipolar", "D"), [1.0, 2.0, 3.0])


def test_field_array_returns_zero_when_missing() -> None:
    obj = type("F", (), {})()
    arr = module._field_array(obj, "Dipolar", "D", nangles=5)
    assert arr.shape == (5, 3)
    np.testing.assert_array_equal(arr, np.zeros((5, 3)))


def test_field_array_broadcasts_single_vector() -> None:
    obj = type("F", (), {"Dipolar": np.array([1.0, 2.0, 3.0])})()
    arr = module._field_array(obj, "Dipolar", "D", nangles=3)
    np.testing.assert_array_equal(arr, np.tile([1.0, 2.0, 3.0], (3, 1)))


def test_field_array_passes_matching_nangles() -> None:
    values = np.arange(9, dtype=float).reshape(3, 3)
    obj = type("F", (), {"Dipolar": values})()
    arr = module._field_array(obj, "Dipolar", "D", nangles=3)
    np.testing.assert_array_equal(arr, values)


def test_field_array_broadcasts_single_row() -> None:
    obj = type("F", (), {"Dipolar": np.array([[1.0, 2.0, 3.0]])})()
    arr = module._field_array(obj, "Dipolar", "D", nangles=4)
    np.testing.assert_array_equal(arr, np.tile([1.0, 2.0, 3.0], (4, 1)))


def test_field_array_rejects_unexpected_shape() -> None:
    obj = type("F", (), {"Dipolar": np.zeros((2, 3))})()
    with pytest.raises(MagneticFieldError, match="unexpected field shape"):
        module._field_array(obj, "Dipolar", "D", nangles=4)


# --------------------------------------------------------------------------- #
# Muon-position expansion and magnetic-ion resolution
# --------------------------------------------------------------------------- #


def test_expanded_muon_positions_labels_sites() -> None:
    expanded = module._expanded_muon_positions(
        _structure(),
        [MuonSite(label="", fractional_position=[0.25, 0.25, 0.25]), [0.5, 0.5, 0.5]],
        MagneticFieldOptions(),
    )
    assert expanded[0][0] == "site 1"
    np.testing.assert_allclose(expanded[0][1], [0.25, 0.25, 0.25])
    assert expanded[1][0] == "site 2"
    np.testing.assert_allclose(expanded[1][1], [0.5, 0.5, 0.5])


def test_expanded_muon_positions_named_site_label() -> None:
    expanded = module._expanded_muon_positions(
        _structure(),
        [MuonSite(label="Mu1", fractional_position=[0.1, 0.2, 0.3])],
        MagneticFieldOptions(),
    )
    assert expanded[0][0] == "Mu1"


def test_expanded_muon_positions_expands_equivalents() -> None:
    expanded = module._expanded_muon_positions(
        _structure(),
        [MuonSite(label="Mu", fractional_position=[0.25, 0.25, 0.25])],
        MagneticFieldOptions(find_equivalent_sites=True),
    )
    assert len(expanded) == 8
    assert expanded[0][0] == "Mu equiv 1"
    labels = {label for label, _ in expanded}
    assert len(labels) == 8


def test_expanded_muon_positions_requires_at_least_one() -> None:
    with pytest.raises(MagneticFieldError, match="no muon positions provided"):
        module._expanded_muon_positions(_structure(), [], MagneticFieldOptions())


def test_resolve_magnetic_ion_indices_rejects_both_inputs() -> None:
    with pytest.raises(MagneticFieldError, match="not both"):
        module._resolve_magnetic_ion_indices(
            _two_atom_structure(),
            magnetic_ion_indices=[0],
            magnetic_ion_selector=AtomSelector.species("Fe"),
        )


def test_resolve_magnetic_ion_indices_via_selector() -> None:
    indices = module._resolve_magnetic_ion_indices(
        _two_atom_structure(),
        magnetic_ion_indices=None,
        magnetic_ion_selector=AtomSelector.species("Fe"),
    )
    assert indices == (0,)


def test_resolve_magnetic_ion_indices_dedupes_and_sorts() -> None:
    indices = module._resolve_magnetic_ion_indices(
        _two_atom_structure(),
        magnetic_ion_indices=[1, 0, 1],
        magnetic_ion_selector=None,
    )
    assert indices == (0, 1)


def test_resolve_magnetic_ion_indices_rejects_non_integers() -> None:
    with pytest.raises(MagneticFieldError, match="must be integers"):
        module._resolve_magnetic_ion_indices(
            _two_atom_structure(),
            magnetic_ion_indices=["not-an-int"],
            magnetic_ion_selector=None,
        )


def test_resolve_magnetic_ion_indices_requires_selection() -> None:
    with pytest.raises(MagneticFieldError, match="at least one magnetic ion"):
        module._resolve_magnetic_ion_indices(
            _two_atom_structure(),
            magnetic_ion_indices=None,
            magnetic_ion_selector=None,
        )


def test_resolve_magnetic_ion_indices_rejects_out_of_range() -> None:
    with pytest.raises(MagneticFieldError, match="out-of-range atom index"):
        module._resolve_magnetic_ion_indices(
            _two_atom_structure(),
            magnetic_ion_indices=[5],
            magnetic_ion_selector=None,
        )


def test_unit_tensor_fourier_components_marks_selected_indices() -> None:
    fc = module._unit_tensor_fourier_components(_two_atom_structure(), [0])
    assert fc.shape == (2, 3)
    assert fc[0, 2] == pytest.approx(1.0 + 0.0j)
    assert fc[1, 2] == pytest.approx(0.0 + 0.0j)


# --------------------------------------------------------------------------- #
# Small geometric helpers
# --------------------------------------------------------------------------- #


def test_canonical_fractional_wraps_and_snaps_to_zero() -> None:
    out = module._canonical_fractional([1.5, -0.25, 1.0])
    np.testing.assert_allclose(out, [0.5, 0.75, 0.0])


def test_fractional_distance_wraps_across_cell_boundary() -> None:
    dist = module._fractional_distance(np.array([0.05, 0.0, 0.0]), np.array([0.95, 0.0, 0.0]))
    assert dist == pytest.approx(0.1)


def test_as_vector_rejects_wrong_shape() -> None:
    with pytest.raises(MagneticFieldError, match="must be a three-vector"):
        module._as_vector([1.0, 2.0], "thing")


def test_tuple3_returns_python_floats() -> None:
    result = module._tuple3(np.array([1, 2, 3]))
    assert result == (1.0, 2.0, 3.0)
    assert all(isinstance(value, float) for value in result)


def test_tensor3x3_rejects_wrong_shape() -> None:
    with pytest.raises(MagneticFieldError, match="unexpected dipolar tensor shape"):
        module._tensor3x3(np.zeros((2, 2)))


# --------------------------------------------------------------------------- #
# calculate_commensurate_fields branches (via monkeypatched engines)
# --------------------------------------------------------------------------- #


def test_commensurate_fields_rejects_empty_model() -> None:
    with pytest.raises(MagneticFieldError, match="no non-zero Fourier components"):
        calculate_commensurate_fields(
            _structure(),
            _empty_model(),
            [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
        )


def test_commensurate_fields_rejects_incommensurate_model(monkeypatch) -> None:
    monkeypatch.setattr(
        module, "_load_muesr_engines", lambda: pytest.fail("muesr should not be loaded")
    )
    with pytest.raises(MagneticFieldError, match="received an incommensurate model"):
        calculate_commensurate_fields(
            _structure(),
            _model([0.12345, 0.0, 0.0]),
            [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
            options=MagneticFieldOptions(commensurate_max_period=8),
        )


def test_commensurate_fields_applies_contact_coupling_and_radius(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Sample:
        def new_mm(self):
            self.mm = type("_MM", (), {})()

    class _Field:
        def __init__(self):
            self.Dipolar = np.array([1.0, 0.0, 0.0])
            self.Lorentz = np.array([0.0, 1.0, 0.0])
            self.Contact = np.array([0.0, 0.0, 1.0])
            self.ACont = None

    field = _Field()

    def _locfield(_sample, _mode, _supercell, radius, **_kwargs):
        captured["radius"] = radius
        field.ACont = None
        return [field]

    monkeypatch.setattr(module, "_build_muesr_sample", lambda *_a: _Sample())
    monkeypatch.setattr(
        module,
        "_load_muesr_engines",
        lambda: (_locfield, lambda *_a: 7.5, None),
    )

    results = calculate_commensurate_fields(
        _structure(),
        _model([0.0, 0.0, 0.5]),
        [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1),
            include_lorentz=False,
            contact_coupling=0.42,
            demag_shape="sphere",
            applied_field=(0.0, 0.0, 0.1),
        ),
    )

    assert captured["radius"] == pytest.approx(7.5)
    assert field.ACont == pytest.approx(0.42)
    result = results[0]
    # Lorentz excluded -> zero, contact carried through, applied field present.
    np.testing.assert_allclose(result.lorentz, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.contact, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(result.applied, [0.0, 0.0, 0.1])
    # demag derives from (zero) lorentz, so it is zero here.
    np.testing.assert_allclose(result.demag, [0.0, 0.0, 0.0])


def test_commensurate_fields_explicit_radius_skips_largest_sphere(monkeypatch) -> None:
    class _Sample:
        def new_mm(self):
            self.mm = type("_MM", (), {})()

    class _Field:
        Dipolar = np.zeros(3)
        Lorentz = np.zeros(3)
        Contact = np.zeros(3)

    seen: dict[str, object] = {}

    def _locfield(_sample, _mode, _supercell, radius, **_kwargs):
        seen["radius"] = radius
        return [_Field()]

    def _largest(*_a):
        raise AssertionError("should not be called when radius is explicit")

    monkeypatch.setattr(module, "_build_muesr_sample", lambda *_a: _Sample())
    monkeypatch.setattr(module, "_load_muesr_engines", lambda: (_locfield, _largest, None))

    calculate_commensurate_fields(
        _structure(),
        _model([0.0, 0.0, 0.5]),
        [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
        options=MagneticFieldOptions(supercell=(1, 1, 1), radius=4.0),
    )
    assert seen["radius"] == pytest.approx(4.0)
