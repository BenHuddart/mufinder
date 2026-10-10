"""Tests for magnetic field helper calculations."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from mufinder.domain.models import MuonSite, Structure
from mufinder.science._muesr_compat import patch_numpy_aliases_for_muesr
from mufinder.science.demagnetization import DemagnetizingSample
from mufinder.science.magnetic_fields import (
    CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB,
    MagneticFieldOptions,
    assigned_magnetic_atom_indices,
    calculate_commensurate_fields,
    calculate_dipolar_tensors,
    calculate_dipolar_tensors_for_atoms,
    calculate_finite_dipolar_field,
    calculate_incommensurate_field_distributions,
    classify_magnetic_model,
    contact_coupling_to_angstrom_minus3,
    find_equivalent_muon_sites,
    is_commensurate_propagation_vector,
)
from mufinder.science.magnetic_structure import (
    AtomSelector,
    ComplexVector,
    FourierTerm,
    MagneticComponent,
    MagneticStructureModel,
)
from tests.support.structures import host_structure


def test_patch_numpy_aliases_for_muesr_restores_removed_aliases(monkeypatch) -> None:
    monkeypatch.delattr(np, "complex", raising=False)
    monkeypatch.delattr(np, "float", raising=False)

    patch_numpy_aliases_for_muesr()

    assert np.complex is complex
    assert np.float is float


def test_load_muesr_engines_patches_numpy_aliases(monkeypatch) -> None:
    from mufinder.science import magnetic_fields as module

    monkeypatch.delattr(np, "complex", raising=False)
    monkeypatch.delattr(np, "float", raising=False)

    locfield, find_largest_sphere, dipten = module._load_muesr_engines()

    assert callable(locfield)
    assert callable(find_largest_sphere)
    assert callable(dipten)
    assert np.complex is complex
    assert np.float is float


_structure = partial(host_structure, label="Fe", lattice=3.0, species=("Fe",), positions=((0.0, 0.0, 0.0),))


def _two_atom_structure() -> Structure:
    return Structure(
        label="FeO",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _model(q=(0.0, 0.0, 0.0)) -> MagneticStructureModel:
    q_coeff = (1.0, 0.0, 0.0) if any(abs(float(value)) > 1e-12 for value in q) else (0.0, 0.0, 0.0)
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


def _circular_helix_model(q=(0.12345, 0.0, 0.0)) -> MagneticStructureModel:
    return MagneticStructureModel(
        q_basis_rlu=(tuple(float(v) for v in q), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="helix",
                label="Helix",
                kind="helix",
                selector=AtomSelector.all(),
                terms=(
                    FourierTerm(
                        (1.0, 0.0, 0.0),
                        ComplexVector(real=(1.0, 0.0, 0.0), imaginary=(0.0, 1.0, 0.0)),
                    ),
                ),
            ),
        ),
    )


def _component(selector: AtomSelector, *, enabled: bool = True, moment: float = 1.0) -> MagneticComponent:
    return MagneticComponent(
        component_id="moment",
        label="Moment",
        kind="custom",
        selector=selector,
        terms=(FourierTerm((0.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, moment])),),
        enabled=enabled,
    )


def test_commensurability_detection() -> None:
    assert is_commensurate_propagation_vector([0.0, 0.0, 0.5])
    assert is_commensurate_propagation_vector([1.0 / 3.0, 0.0, 0.0])
    assert not is_commensurate_propagation_vector([0.12345, 0.0, 0.0], max_period=8)


def test_classify_magnetic_model_uses_all_q_vectors() -> None:
    assert classify_magnetic_model(_structure(), _model([0.0, 0.0, 0.5])) == "commensurate"
    assert classify_magnetic_model(_structure(), _model([0.12345, 0.0, 0.0]), max_period=8) == "incommensurate"


def test_find_equivalent_muon_sites_deduplicates_positions() -> None:
    structure = Structure(
        label="P1",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )

    equivalents = find_equivalent_muon_sites(structure, [0.25, 0.25, 0.25])

    assert len(equivalents) == 8
    assert np.asarray(equivalents[0]) == pytest.approx(np.asarray((0.25, 0.25, 0.25)))
    assert len({tuple(position) for position in equivalents}) == len(equivalents)


def test_finite_dipolar_field_matches_axial_formula() -> None:
    field = calculate_finite_dipolar_field(
        positions_cart_ang=[[0.0, 0.0, 0.0]],
        moments_mu_b=[[0.0, 0.0, 1.0]],
        muon_cart_ang=[0.0, 0.0, 1.0],
    )

    expected = 1e-7 * 2.0 * 9.274_010_078_3e-24 / ((1e-10) ** 3)
    assert field[0] == pytest.approx(0.0)
    assert field[1] == pytest.approx(0.0)
    assert field[2] == pytest.approx(expected)


def test_calculate_commensurate_fields_reports_missing_muesr_cleanly(monkeypatch) -> None:
    from mufinder.science import magnetic_fields as module

    def _raise():
        raise module.MagneticFieldDependencyError("missing muesr")

    monkeypatch.setattr(module, "_load_muesr_engines", _raise)

    with pytest.raises(module.MagneticFieldDependencyError, match="missing muesr"):
        calculate_commensurate_fields(
            _structure(),
            _model(),
            [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
            options=MagneticFieldOptions(supercell=(1, 1, 1)),
        )


def test_commensurate_field_calculation_merges_equivalent_k_calls(monkeypatch) -> None:
    from mufinder.science import magnetic_fields as module

    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="a",
                label="A",
                kind="custom",
                selector=AtomSelector.all(),
                terms=(FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 1.0])),),
            ),
            MagneticComponent(
                component_id="b",
                label="B",
                kind="custom",
                selector=AtomSelector.all(),
                terms=(FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([1.0, 0.0, 0.0])),),
            ),
        ),
    )
    calls: list[np.ndarray] = []

    class _Sample:
        def new_mm(self):
            self.mm = type("_MM", (), {})()

    class _Field:
        Dipolar = np.zeros(3, dtype=float)
        Lorentz = np.zeros(3, dtype=float)
        Contact = np.zeros(3, dtype=float)

    def _sample(_structure, _muons):
        return _Sample()

    def _locfield(sample, *_args, **_kwargs):
        calls.append(np.asarray(sample.mm.fc, dtype=complex).copy())
        return [_Field()]

    monkeypatch.setattr(module, "_build_muesr_sample", _sample)
    monkeypatch.setattr(module, "_load_muesr_engines", lambda: (_locfield, lambda *_args: 1.0, None))

    calculate_commensurate_fields(
        _structure(),
        model,
        [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
        options=MagneticFieldOptions(supercell=(1, 1, 1)),
    )

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0].real, [1.0, 0.0, 1.0])


def test_incommensurate_distribution_options_use_fine_defaults() -> None:
    options = MagneticFieldOptions()

    assert options.nangles == 4096
    assert options.phase_sample_count == 200_000
    assert options.phase_sampling == "auto"
    assert options.max_grid_samples == 1_000_000
    assert options.histogram_bins == 512


def test_auto_two_q_sampling_is_bounded_by_phase_sample_count() -> None:
    from mufinder.science import magnetic_fields as module

    phases = module._build_phase_samples_turns(
        n_dim=2,
        nangles=4096,
        phase_sample_count=200_000,
        sampling="auto",
        random_seed=None,
        max_grid_samples=1_000_000,
    )

    assert phases.shape == (199_809, 2)
    assert phases.shape[0] < 4096**2


def test_auto_two_q_sampling_ignores_large_legacy_grid_cap() -> None:
    from mufinder.science import magnetic_fields as module

    phases = module._build_phase_samples_turns(
        n_dim=2,
        nangles=4096,
        phase_sample_count=200_000,
        sampling="auto",
        random_seed=None,
        max_grid_samples=20_000_000,
    )

    assert phases.shape == (199_809, 2)
    assert phases.shape[0] < 4096**2


def test_contact_coupling_unit_conversion_to_muesr_acont() -> None:
    assert contact_coupling_to_angstrom_minus3(2.0, "angstrom^-3") == pytest.approx(2.0)
    assert contact_coupling_to_angstrom_minus3(1.0, "T_per_muB") == pytest.approx(
        1.0 / CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB
    )
    assert contact_coupling_to_angstrom_minus3(135.538_96, "MHz_per_muB") == pytest.approx(
        contact_coupling_to_angstrom_minus3(1.0, "T_per_muB")
    )


def test_contact_coupling_rejects_unknown_units() -> None:
    with pytest.raises(ValueError, match="unknown contact coupling unit"):
        contact_coupling_to_angstrom_minus3(1.0, "bananas")


def test_assigned_magnetic_atom_indices_uses_enabled_nonzero_components() -> None:
    structure = _two_atom_structure()
    model = MagneticStructureModel(
        components=(
            _component(AtomSelector.indices([0])),
            _component(AtomSelector.indices([1]), enabled=False),
            _component(AtomSelector.species("O"), moment=0.0),
        )
    )

    assert assigned_magnetic_atom_indices(structure, model) == (0,)


def test_calculate_dipolar_tensors_for_atoms_uses_explicit_selector(monkeypatch) -> None:
    from mufinder.science import magnetic_fields as module

    calls: dict[str, object] = {}

    class _Sample:
        def add_muon(self, position):
            calls.setdefault("muons", []).append(tuple(position))

        def new_mm(self):
            self.mm = type("_MM", (), {})()

    def _sample(_structure, muons):
        sample = _Sample()
        for muon in muons:
            sample.add_muon(muon)
        return sample

    def _find_largest_sphere(_sample, supercell):
        calls["supercell"] = tuple(supercell)
        return 12.0

    def _dipten(sample, supercell, radius):
        calls["fc"] = np.asarray(sample.mm.fc).copy()
        calls["radius"] = radius
        return [np.eye(3)]

    monkeypatch.setattr(module, "_build_muesr_sample", _sample)
    monkeypatch.setattr(module, "_load_muesr_engines", lambda: (None, _find_largest_sphere, _dipten))

    results = calculate_dipolar_tensors_for_atoms(
        _two_atom_structure(),
        [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
        magnetic_ion_selector=AtomSelector.species("Fe"),
        options=MagneticFieldOptions(supercell=(2, 3, 4)),
    )

    assert calls["supercell"] == (2, 3, 4)
    assert calls["radius"] == pytest.approx(12.0)
    assert calls["fc"].shape == (2, 3)
    assert calls["fc"][0, 2] == pytest.approx(1.0)
    assert calls["fc"][1, 2] == pytest.approx(0.0)
    assert results[0].site_label == "A"
    assert results[0].muon_position == pytest.approx((0.25, 0.25, 0.25))
    assert np.asarray(results[0].tensor_T_per_muB) == pytest.approx(np.eye(3))
    assert np.asarray(results[0].to_json()["dipolar_tensor_T_per_muB"]) == pytest.approx(np.eye(3))


def test_calculate_dipolar_tensors_requires_selected_magnetic_ions(monkeypatch) -> None:
    from mufinder.science import magnetic_fields as module

    monkeypatch.setattr(module, "_load_muesr_engines", lambda: pytest.fail("muesr should not be loaded"))

    with pytest.raises(ValueError, match="choose at least one magnetic ion"):
        calculate_dipolar_tensors_for_atoms(
            _two_atom_structure(),
            [[0.25, 0.25, 0.25]],
            magnetic_ion_indices=[],
        )


def test_calculate_dipolar_tensors_compatibility_uses_assigned_atoms(monkeypatch) -> None:
    from mufinder.science import magnetic_fields as module

    captured: dict[str, object] = {}

    class _Result:
        def to_json(self):
            return {"site_label": "site 1", "dipolar_tensor_T_per_muB": [[1.0, 0.0, 0.0]] * 3}

    def _calculate(structure, muon_sites, *, magnetic_ion_indices, options):
        captured["indices"] = tuple(magnetic_ion_indices)
        captured["muons"] = tuple(tuple(site) for site in muon_sites)
        return (_Result(),)

    monkeypatch.setattr(module, "calculate_dipolar_tensors_for_atoms", _calculate)

    payload = calculate_dipolar_tensors(
        _two_atom_structure(),
        MagneticStructureModel(components=(_component(AtomSelector.species("Fe")),)),
        [[0.25, 0.25, 0.25]],
    )

    assert captured["indices"] == (0,)
    assert payload[0]["site_label"] == "site 1"


def test_incommensurate_distribution_rejects_invalid_histogram_bins() -> None:
    with pytest.raises(ValueError, match="histogram_bins must be positive"):
        calculate_incommensurate_field_distributions(
            _structure(),
            _model([0.12345, 0.0, 0.0]),
            [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
            options=MagneticFieldOptions(histogram_bins=0),
        )


def test_incommensurate_distribution_splits_elliptical_components_before_muesr(monkeypatch) -> None:
    from mufinder.science import magnetic_fields as module

    structure = _structure()
    model = MagneticStructureModel(
        q_basis_rlu=((0.12345, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="elliptical",
                label="Elliptical",
                kind="custom",
                selector=AtomSelector.all(),
                terms=(
                    FourierTerm(
                        (1.0, 0.0, 0.0),
                        ComplexVector(real=(1.0, 0.2, 0.0), imaginary=(0.3, 0.5, 0.1)),
                    ),
                ),
            ),
        ),
    )
    captured_fcs: list[np.ndarray] = []

    class _Sample:
        def new_mm(self):
            self.mm = type("_MM", (), {})()

    class _Field:
        Dipolar = np.zeros((4, 3), dtype=float)
        Lorentz = np.zeros((4, 3), dtype=float)
        Contact = np.zeros((4, 3), dtype=float)

    def _sample(_structure, _muons):
        return _Sample()

    def _locfield(sample, mode, *_args, **_kwargs):
        assert mode == "i"
        captured_fcs.append(np.asarray(sample.mm.fc, dtype=complex).copy())
        return [_Field()]

    monkeypatch.setattr(module, "_build_muesr_sample", _sample)
    monkeypatch.setattr(module, "_load_muesr_engines", lambda: (_locfield, lambda *_args: 1.0, None))

    calculate_incommensurate_field_distributions(
        structure,
        model,
        [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, histogram_bins=4),
    )

    assert len(captured_fcs) == 2
    original_fc = np.asarray(captured_fcs).sum(axis=0)
    for fc in captured_fcs:
        real = fc[0].real
        imag = fc[0].imag
        assert np.linalg.norm(real) == pytest.approx(np.linalg.norm(imag), abs=1e-10)
        assert float(np.dot(real, imag)) == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(original_fc[0], [1.0 + 0.3j, 0.2 + 0.5j, 0.1j], atol=1e-11)


def test_incommensurate_distribution_combines_independent_multi_q_phase_grid(monkeypatch) -> None:
    from mufinder.science import magnetic_fields as module

    structure = _structure()
    model = MagneticStructureModel(
        q_basis_rlu=((0.12345, 0.0, 0.0), (0.0, 0.23456, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="q1",
                label="q1",
                kind="helix",
                selector=AtomSelector.all(),
                terms=(FourierTerm((1.0, 0.0, 0.0), ComplexVector(real=(1.0, 0.0, 0.0), imaginary=(0.0, 1.0, 0.0))),),
            ),
            MagneticComponent(
                component_id="q2",
                label="q2",
                kind="helix",
                selector=AtomSelector.all(),
                terms=(FourierTerm((0.0, 1.0, 0.0), ComplexVector(real=(1.0, 0.0, 0.0), imaginary=(0.0, 1.0, 0.0))),),
            ),
        ),
    )

    class _Sample:
        def new_mm(self):
            self.mm = type("_MM", (), {})()

    class _Field:
        def __init__(self, values: np.ndarray):
            self.Dipolar = values
            self.Lorentz = np.zeros_like(values)
            self.Contact = np.zeros_like(values)

    def _sample(_structure, _muons):
        return _Sample()

    def _locfield(sample, mode, *_args, **kwargs):
        assert mode == "i"
        nangles = int(kwargs["nangles"])
        values = np.zeros((nangles, 3), dtype=float)
        offset = 100.0 if abs(float(sample.mm.k[1])) > 0.0 else 0.0
        values[:, 0] = offset + np.arange(nangles, dtype=float)
        return [_Field(values)]

    monkeypatch.setattr(module, "_build_muesr_sample", _sample)
    monkeypatch.setattr(module, "_load_muesr_engines", lambda: (_locfield, lambda *_args: 1.0, None))

    result = calculate_incommensurate_field_distributions(
        structure,
        model,
        [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1),
            nangles=4,
            phase_sampling="grid",
            histogram_bins=4,
        ),
    )[0]

    assert result.fields_T.shape == (16, 3)
    expected_x = []
    for q1_index in range(4):
        for q2_index in range(4):
            expected_x.append(float(q1_index + 100.0 + q2_index))
    np.testing.assert_allclose(result.fields_T[:, 0], np.asarray(expected_x), atol=1e-12)
    assert result.calculation_parameters["phase_sampling"] == "grid"
    assert result.calculation_parameters["independent_q_indices"] == [0, 1]
    assert result.calculation_parameters["phase_sample_count"] == 16


def test_incommensurate_distribution_ignores_unused_q_basis_rows(monkeypatch) -> None:
    from mufinder.science import magnetic_fields as module

    model = MagneticStructureModel(
        q_basis_rlu=((0.12345, 0.0, 0.0), (0.0, 0.23456, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="q1",
                label="q1",
                kind="helix",
                selector=AtomSelector.all(),
                terms=(FourierTerm((1.0, 0.0, 0.0), ComplexVector(real=(1.0, 0.0, 0.0), imaginary=(0.0, 1.0, 0.0))),),
            ),
        ),
    )

    class _Sample:
        def new_mm(self):
            self.mm = type("_MM", (), {})()

    class _Field:
        Dipolar = np.zeros((4, 3), dtype=float)
        Lorentz = np.zeros((4, 3), dtype=float)
        Contact = np.zeros((4, 3), dtype=float)

    monkeypatch.setattr(module, "_build_muesr_sample", lambda *_args: _Sample())
    monkeypatch.setattr(module, "_load_muesr_engines", lambda: (lambda *_args, **_kwargs: [_Field()], lambda *_args: 1.0, None))

    result = calculate_incommensurate_field_distributions(
        _structure(),
        model,
        [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, phase_sampling="grid", histogram_bins=4),
    )[0]

    assert result.fields_T.shape == (4, 3)
    assert result.calculation_parameters["independent_q_indices"] == [0]
    assert result.calculation_parameters["phase_sample_count"] == 4


def test_incommensurate_distribution_collapses_dependent_physical_q_rows(monkeypatch) -> None:
    from mufinder.science import magnetic_fields as module

    model = MagneticStructureModel(
        q_basis_rlu=((0.12345, 0.0, 0.0), (0.24690, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="q1",
                label="q1",
                kind="helix",
                selector=AtomSelector.all(),
                terms=(FourierTerm((1.0, 0.0, 0.0), ComplexVector(real=(1.0, 0.0, 0.0), imaginary=(0.0, 1.0, 0.0))),),
            ),
            MagneticComponent(
                component_id="q2",
                label="q2",
                kind="helix",
                selector=AtomSelector.all(),
                terms=(FourierTerm((0.0, 1.0, 0.0), ComplexVector(real=(1.0, 0.0, 0.0), imaginary=(0.0, 1.0, 0.0))),),
            ),
        ),
    )

    class _Sample:
        def new_mm(self):
            self.mm = type("_MM", (), {})()

    class _Field:
        def __init__(self, offset: float):
            values = np.zeros((4, 3), dtype=float)
            values[:, 0] = offset + np.arange(4, dtype=float)
            self.Dipolar = values
            self.Lorentz = np.zeros_like(values)
            self.Contact = np.zeros_like(values)

    def _sample(_structure, _muons):
        return _Sample()

    def _locfield(sample, *_args, **_kwargs):
        offset = 100.0 if float(sample.mm.k[0]) > 0.2 else 0.0
        return [_Field(offset)]

    monkeypatch.setattr(module, "_build_muesr_sample", _sample)
    monkeypatch.setattr(module, "_load_muesr_engines", lambda: (_locfield, lambda *_args: 1.0, None))

    result = calculate_incommensurate_field_distributions(
        _structure(),
        model,
        [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, phase_sampling="grid", histogram_bins=4),
    )[0]

    assert result.fields_T.shape == (4, 3)
    np.testing.assert_allclose(result.fields_T[:, 0], [100.0, 103.0, 102.0, 105.0], atol=1e-12)
    assert result.calculation_parameters["independent_q_indices"] == [0, 1]
    assert result.calculation_parameters["phase_sample_count"] == 4


def test_incommensurate_distribution_applies_demag_to_sampled_lorentz_field(monkeypatch) -> None:
    from mufinder.science import magnetic_fields as module

    class _Sample:
        def new_mm(self):
            self.mm = type("_MM", (), {})()

    class _Field:
        def __init__(self, nangles: int):
            self.Dipolar = np.zeros((nangles, 3), dtype=float)
            self.Lorentz = np.zeros((nangles, 3), dtype=float)
            self.Lorentz[:, 0] = np.arange(nangles, dtype=float)
            self.Lorentz[:, 1] = 2.0 * np.arange(nangles, dtype=float)
            self.Lorentz[:, 2] = 3.0 * np.arange(nangles, dtype=float)
            self.Contact = np.zeros((nangles, 3), dtype=float)

    def _sample(_structure, _muons):
        return _Sample()

    def _locfield(_sample, mode, *_args, **kwargs):
        assert mode == "i"
        return [_Field(int(kwargs["nangles"]))]

    monkeypatch.setattr(module, "_build_muesr_sample", _sample)
    monkeypatch.setattr(module, "_load_muesr_engines", lambda: (_locfield, lambda *_args: 1.0, None))

    result = calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1),
            nangles=4,
            phase_sampling="grid",
            demag_shape="sphere",
            histogram_bins=4,
        ),
    )[0]

    lookup = np.column_stack(
        [
            np.arange(4, dtype=float),
            2.0 * np.arange(4, dtype=float),
            3.0 * np.arange(4, dtype=float),
        ]
    )
    np.testing.assert_allclose(result.fields_T, 0.5 * lookup, atol=1e-12)


def test_incommensurate_distribution_applies_advanced_demag_sample(monkeypatch) -> None:
    from mufinder.science import magnetic_fields as module

    class _Sample:
        def new_mm(self):
            self.mm = type("_MM", (), {})()

    class _Field:
        Dipolar = np.zeros((3,), dtype=float)
        Lorentz = np.array([0.0, 0.0, 2.0], dtype=float)
        Contact = np.zeros((3,), dtype=float)

    def _sample(_structure, _muons):
        return _Sample()

    def _locfield(_sample, mode, *_args, **kwargs):
        assert mode == "i"
        assert kwargs["nangles"] == 3
        return [_Field()]

    monkeypatch.setattr(module, "_build_muesr_sample", _sample)
    monkeypatch.setattr(module, "_load_muesr_engines", lambda: (_locfield, lambda *_args: 1.0, None))

    result = calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1),
            nangles=3,
            phase_sampling="grid",
            demag_sample=DemagnetizingSample(shape="ellipsoid", semi_axes=(1.0, 1.0, 1.0)),
            histogram_bins=4,
        ),
    )[0]

    np.testing.assert_allclose(result.fields_T, np.full((3, 3), [0.0, 0.0, 1.0]), atol=1e-12)
    assert result.calculation_parameters["demag_sample"]["shape"] == "ellipsoid"


def test_field_magnitude_histogram_handles_near_delta_distribution() -> None:
    from mufinder.science import magnetic_fields as module

    magnitudes = np.asarray(
        [
            0.19073784712037425,
            0.19073784712037958,
            0.19073784712038494,
        ],
        dtype=float,
    )

    centres, counts = module._field_magnitude_histogram(magnitudes, bins=512)

    assert centres.shape == (512,)
    assert counts.shape == (512,)
    assert int(np.sum(counts)) == magnitudes.size
    assert np.ptp(centres) > 1e-7
