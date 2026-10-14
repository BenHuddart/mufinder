"""Tests for magnetic field helper calculations.

The periodic calculations run through muesr. Apart from the engine-loading
tests, every test here swaps in a fake muesr (``_install_fake_muesr``) so the
public entry points run end to end on fields the test controls.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import importlib
import sys
from types import ModuleType

import numpy as np
import pytest

from mufinder.domain.models import MuonSite
from mufinder.science import magnetic_fields as module
from mufinder.science._muesr_compat import muesr_imports, patch_numpy_aliases_for_muesr
from mufinder.science.demagnetization import DemagnetizingSample
from mufinder.science.magnetic_fields import (
    CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB,
    DipolarTensorResult,
    MagneticFieldDependencyError,
    MagneticFieldDistributionResult,
    MagneticFieldError,
    MagneticFieldOptions,
    MagneticFieldResult,
    assigned_magnetic_atom_indices,
    calculate_commensurate_fields,
    calculate_dipolar_tensors,
    calculate_distortion_field_correction,
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
from mufinder.science.muon_distortion import map_muon_distortion
from tests.support.structures import host_structure, relaxed_muon_supercell

_structure = partial(host_structure, label="Fe", lattice=3.0, species=("Fe",), positions=((0.0, 0.0, 0.0),))
_two_atom_structure = partial(host_structure, label="FeO", lattice=3.0, species=("Fe", "O"))
_SITE_A = MuonSite(label="A", fractional_position=[0.25, 0.25, 0.25])


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


def _helix(component_id: str, q_coeff: tuple[float, float, float]) -> MagneticComponent:
    return MagneticComponent(
        component_id=component_id,
        label=component_id,
        kind="helix",
        selector=AtomSelector.all(),
        terms=(FourierTerm(q_coeff, ComplexVector(real=(1.0, 0.0, 0.0), imaginary=(0.0, 1.0, 0.0))),),
    )


def _circular_helix_model(q=(0.12345, 0.0, 0.0)) -> MagneticStructureModel:
    return MagneticStructureModel(
        q_basis_rlu=(tuple(float(v) for v in q), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(_helix("helix", (1.0, 0.0, 0.0)),),
    )


def _two_q_helix_model() -> MagneticStructureModel:
    return MagneticStructureModel(
        q_basis_rlu=((0.12345, 0.0, 0.0), (0.0, 0.23456, 0.0), (0.0, 0.0, 0.0)),
        components=(_helix("q1", (1.0, 0.0, 0.0)), _helix("q2", (0.0, 1.0, 0.0))),
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


# Fake muesr backend


class _FakeSample:
    def __init__(self) -> None:
        self.muons: list[tuple[float, ...]] = []

    def add_muon(self, position) -> None:
        self.muons.append(tuple(float(value) for value in position))

    def new_mm(self) -> None:
        self.mm = type("_MM", (), {})()


class _Field:
    """A muesr local-field record carrying only the attributes a test sets."""

    def __init__(self, **values) -> None:
        for name, value in values.items():
            setattr(self, name, np.asarray(value, dtype=float))


def _fields_per_muon(**values) -> Callable[..., list[_Field]]:
    """A fake ``locfield`` returning the same field record for every muon."""

    def _locfield(sample, *_args, **_kwargs):
        return [_Field(**values) for _ in sample.muons]

    return _locfield


def _install_fake_muesr(
    monkeypatch: pytest.MonkeyPatch,
    *,
    locfield: Callable[..., object] | None = None,
    largest_sphere: Callable[..., float] = lambda *_args: 1.0,
    dipten: Callable[..., object] | None = None,
) -> None:
    def _sample(_structure, muons):
        sample = _FakeSample()
        for muon in muons:
            sample.add_muon(muon)
        return sample

    monkeypatch.setattr(module, "_build_muesr_sample", _sample)
    monkeypatch.setattr(module, "_load_muesr_engines", lambda: (locfield, largest_sphere, dipten))


def _forbid_muesr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_load_muesr_engines", lambda: pytest.fail("muesr should not be loaded"))


# muesr engine loading


def test_patch_numpy_aliases_for_muesr_restores_removed_aliases(monkeypatch) -> None:
    monkeypatch.delattr(np, "complex", raising=False)
    monkeypatch.delattr(np, "float", raising=False)

    patch_numpy_aliases_for_muesr()

    assert np.complex is complex
    assert np.float is float


def test_load_muesr_engines_patches_numpy_aliases(monkeypatch) -> None:
    monkeypatch.delattr(np, "complex", raising=False)
    monkeypatch.delattr(np, "float", raising=False)

    locfield, find_largest_sphere, dipten = module._load_muesr_engines()

    assert callable(locfield)
    assert callable(find_largest_sphere)
    assert callable(dipten)
    assert np.complex is complex
    assert np.float is float



def test_muesr_imports_supplies_cgi_while_importing_muesr_when_cgi_is_missing(monkeypatch) -> None:
    # Python 3.13 removed cgi; a None entry makes ``import cgi`` fail on any version.
    monkeypatch.setitem(sys.modules, "cgi", None)
    core = importlib.import_module("muesr.core")
    monkeypatch.setattr(core, "prettytable", core.prettytable)
    monkeypatch.delitem(sys.modules, "muesr.core.prettytable")

    with muesr_imports():
        prettytable = importlib.import_module("muesr.core.prettytable")

    assert isinstance(prettytable.cgi, ModuleType)
    assert callable(prettytable.PrettyTable)
    assert sys.modules["cgi"] is None


def test_muesr_imports_keeps_an_importable_cgi(monkeypatch) -> None:
    cgi = ModuleType("cgi")
    monkeypatch.setitem(sys.modules, "cgi", cgi)

    with muesr_imports():
        assert sys.modules["cgi"] is cgi

    assert sys.modules["cgi"] is cgi


# Result records


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
    assert "distortion_T" not in payload


def test_magnetic_field_result_adds_the_distortion_correction_to_the_total() -> None:
    result = MagneticFieldResult(
        site_label="A",
        muon_position=(0.1, 0.2, 0.3),
        dipolar=(1.0, 0.0, 0.0),
        lorentz=(0.0, 0.0, 0.0),
        contact=(0.0, 0.0, 0.0),
        demag=(0.0, 0.0, 0.0),
        applied=(0.0, 0.0, 0.0),
        distortion=(0.5, 0.0, 0.25),
    )

    assert result.total == pytest.approx((1.5, 0.0, 0.25))
    assert result.to_json()["distortion_T"] == [0.5, 0.0, 0.25]


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


# Commensurability and model classification


def test_commensurability_detection() -> None:
    assert is_commensurate_propagation_vector([0.0, 0.0, 0.5])
    assert is_commensurate_propagation_vector([1.0 / 3.0, 0.0, 0.0])
    assert is_commensurate_propagation_vector([0.0, 0.0, 0.0])
    assert not is_commensurate_propagation_vector([0.12345, 0.0, 0.0], max_period=8)


def test_commensurability_rejects_a_non_three_vector() -> None:
    with pytest.raises(MagneticFieldError, match="k_rlu must be a three-vector"):
        is_commensurate_propagation_vector([1.0, 2.0])


def test_classify_magnetic_model_uses_all_q_vectors() -> None:
    assert classify_magnetic_model(_structure(), _model([0.0, 0.0, 0.5])) == "commensurate"
    assert classify_magnetic_model(_structure(), _model([0.12345, 0.0, 0.0]), max_period=8) == "incommensurate"


def test_classify_magnetic_model_raises_for_zero_moment_model() -> None:
    with pytest.raises(MagneticFieldError, match="no non-zero Fourier components"):
        classify_magnetic_model(_structure(), _empty_model())


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


# Equivalent muon sites


def test_find_equivalent_muon_sites_deduplicates_positions() -> None:
    equivalents = find_equivalent_muon_sites(_structure(label="P1"), [0.25, 0.25, 0.25])

    assert len(equivalents) == 8
    assert np.asarray(equivalents[0]) == pytest.approx(np.asarray((0.25, 0.25, 0.25)))
    assert len({tuple(position) for position in equivalents}) == len(equivalents)


def test_find_equivalent_muon_sites_deduplicates_across_the_cell_boundary() -> None:
    # The images of a site just off the origin wrap to ~0.9999; the periodic
    # distance keeps them within tolerance of the original.
    equivalents = find_equivalent_muon_sites(_structure(), [1e-4, 0.0, 0.0], tolerance=1e-3)

    assert len(equivalents) == 1


# Finite dipolar field


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


def test_finite_dipolar_field_zero_when_muon_coincides_with_only_moment() -> None:
    field = calculate_finite_dipolar_field(
        positions_cart_ang=[[0.0, 0.0, 0.0]],
        moments_mu_b=[[0.0, 0.0, 1.0]],
        muon_cart_ang=[0.0, 0.0, 0.0],
    )
    assert field == (0.0, 0.0, 0.0)


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


# Contact coupling units


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


# calculate_commensurate_fields


def test_calculate_commensurate_fields_reports_missing_muesr_cleanly(monkeypatch) -> None:
    def _raise():
        raise MagneticFieldDependencyError("missing muesr")

    monkeypatch.setattr(module, "_load_muesr_engines", _raise)

    with pytest.raises(MagneticFieldDependencyError, match="missing muesr"):
        calculate_commensurate_fields(
            _structure(),
            _model(),
            [_SITE_A],
            options=MagneticFieldOptions(supercell=(1, 1, 1)),
        )


def test_commensurate_fields_rejects_empty_model() -> None:
    with pytest.raises(MagneticFieldError, match="no non-zero Fourier components"):
        calculate_commensurate_fields(_structure(), _empty_model(), [_SITE_A])


def test_commensurate_fields_rejects_incommensurate_model(monkeypatch) -> None:
    _forbid_muesr(monkeypatch)

    with pytest.raises(MagneticFieldError, match="received an incommensurate model"):
        calculate_commensurate_fields(
            _structure(),
            _model([0.12345, 0.0, 0.0]),
            [_SITE_A],
            options=MagneticFieldOptions(commensurate_max_period=8),
        )


def test_commensurate_fields_require_a_muon_position(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon())

    with pytest.raises(MagneticFieldError, match="no muon positions provided"):
        calculate_commensurate_fields(_structure(), _model([0.0, 0.0, 0.5]), [])


def test_commensurate_field_calculation_merges_equivalent_k_calls(monkeypatch) -> None:
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

    def _locfield(sample, *_args, **_kwargs):
        calls.append(np.asarray(sample.mm.fc, dtype=complex).copy())
        return [_Field(Dipolar=np.zeros(3), Lorentz=np.zeros(3), Contact=np.zeros(3))]

    _install_fake_muesr(monkeypatch, locfield=_locfield)

    calculate_commensurate_fields(
        _structure(),
        model,
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1)),
    )

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0].real, [1.0, 0.0, 1.0])


def test_commensurate_fields_label_and_wrap_muon_positions(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(1, 2, 3)))

    results = calculate_commensurate_fields(
        _structure(),
        _model([0.0, 0.0, 0.5]),
        [
            MuonSite(label="", fractional_position=[1.5, -0.25, -1e-12]),
            [0.5, 0.5, 0.5],
            MuonSite(label="Mu1", fractional_position=[0.1, 0.2, 0.3]),
        ],
        options=MagneticFieldOptions(supercell=(1, 1, 1)),
    )

    assert [result.site_label for result in results] == ["site 1", "site 2", "Mu1"]
    # Positions are wrapped into [0, 1), with values a rounding error below 1 snapped to 0.
    assert results[0].muon_position == pytest.approx((0.5, 0.75, 0.0))
    assert results[1].muon_position == pytest.approx((0.5, 0.5, 0.5))
    assert all(isinstance(value, float) for value in results[0].muon_position + results[0].dipolar)


def test_commensurate_fields_expand_symmetry_equivalent_sites(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0, 0, 1)))

    results = calculate_commensurate_fields(
        _structure(),
        _model([0.0, 0.0, 0.5]),
        [MuonSite(label="Mu", fractional_position=[0.25, 0.25, 0.25])],
        options=MagneticFieldOptions(supercell=(1, 1, 1), find_equivalent_sites=True),
    )

    assert len(results) == 8
    assert results[0].site_label == "Mu equiv 1"
    assert len({result.site_label for result in results}) == 8


@pytest.mark.parametrize(
    ("values", "expected_dipolar"),
    [
        ({"Dipolar": (1, 2, 3)}, (1.0, 2.0, 3.0)),
        ({"D": (4, 5, 6)}, (4.0, 5.0, 6.0)),
        ({"Dipolar": [[7, 8, 9]]}, (7.0, 8.0, 9.0)),
        ({}, (0.0, 0.0, 0.0)),
    ],
    ids=["primary-attribute", "fallback-attribute", "single-row", "missing"],
)
def test_commensurate_fields_read_each_muesr_field_layout(monkeypatch, values, expected_dipolar) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(**values))

    (result,) = calculate_commensurate_fields(
        _structure(),
        _model([0.0, 0.0, 0.5]),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1)),
    )

    assert result.dipolar == pytest.approx(expected_dipolar)
    assert result.contact == (0.0, 0.0, 0.0)


def test_commensurate_fields_applies_contact_coupling_and_radius(monkeypatch) -> None:
    captured: dict[str, object] = {}
    field = _Field(Dipolar=(1, 0, 0), Lorentz=(0, 1, 0), Contact=(0, 0, 1))

    def _locfield(_sample, _mode, _supercell, radius, **_kwargs):
        captured["radius"] = radius
        field.ACont = None
        return [field]

    _install_fake_muesr(monkeypatch, locfield=_locfield, largest_sphere=lambda *_a: 7.5)

    results = calculate_commensurate_fields(
        _structure(),
        _model([0.0, 0.0, 0.5]),
        [_SITE_A],
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
    seen: dict[str, object] = {}

    def _locfield(_sample, _mode, _supercell, radius, **_kwargs):
        seen["radius"] = radius
        return [_Field(Dipolar=np.zeros(3))]

    def _largest(*_a):
        raise AssertionError("should not be called when radius is explicit")

    _install_fake_muesr(monkeypatch, locfield=_locfield, largest_sphere=_largest)

    calculate_commensurate_fields(
        _structure(),
        _model([0.0, 0.0, 0.5]),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), radius=4.0),
    )
    assert seen["radius"] == pytest.approx(4.0)


# Muon-induced distortion correction: finite-cell field of displaced minus pristine ions

_MU_X = (0.5, 0.0, 0.0)


def _axial_dipole_change(moment: float, before: float, after: float) -> float:
    """B_z change at a muon on the x axis when a z moment moves from ``before`` to ``after`` Å away."""

    prefactor = -1e-7 * 9.274_010_078_3e-24 / ((1e-10) ** 3)
    return prefactor * moment * (1.0 / abs(after) ** 3 - 1.0 / abs(before) ** 3)


def _ferromagnet() -> MagneticStructureModel:
    return _model()


def _antiferromagnet() -> MagneticStructureModel:
    return _model([0.5, 0.5, 0.5])


@pytest.mark.parametrize(
    ("model", "moment"),
    [(_ferromagnet, 1.0), (_antiferromagnet, -1.0)],
    ids=["ferromagnet", "antiferromagnet"],
)
def test_distortion_correction_matches_one_displaced_dipole(model, moment) -> None:
    # The Fe one cell along +x sits 1.5 Å from the muon and moves 0.1 Å away;
    # in the antiferromagnet its moment is reversed.
    relaxed = relaxed_muon_supercell(_structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)})
    distortion = map_muon_distortion(_structure(), relaxed, _MU_X)

    correction = calculate_distortion_field_correction(_structure(), model(), distortion)

    assert correction[0] == pytest.approx(0.0, abs=1e-12)
    assert correction[1] == pytest.approx(0.0, abs=1e-12)
    assert correction[2] == pytest.approx(_axial_dipole_change(moment, 1.5, 1.6))


def test_distortion_correction_is_zero_without_displacements() -> None:
    distortion = map_muon_distortion(_structure(), relaxed_muon_supercell(_structure(), (2, 2, 2), _MU_X), _MU_X)

    assert calculate_distortion_field_correction(_structure(), _antiferromagnet(), distortion) == pytest.approx(
        (0.0, 0.0, 0.0), abs=1e-15
    )


def test_commensurate_fields_add_the_distortion_correction(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0, 0, 1)))
    relaxed = relaxed_muon_supercell(_structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)})

    corrected, plain = calculate_commensurate_fields(
        _structure(),
        _antiferromagnet(),
        [_MU_X, _MU_X],
        options=MagneticFieldOptions(supercell=(1, 1, 1)),
        distorted_structures=[relaxed, None],
    )

    change = _axial_dipole_change(-1.0, 1.5, 1.6)
    assert corrected.dipolar == pytest.approx((0.0, 0.0, 1.0))
    assert corrected.distortion == pytest.approx((0.0, 0.0, change), abs=1e-12)
    assert corrected.total == pytest.approx((0.0, 0.0, 1.0 + change))
    assert plain.distortion is None
    assert plain.total == pytest.approx((0.0, 0.0, 1.0))


def test_distortion_correction_follows_each_symmetry_equivalent_site(monkeypatch) -> None:
    # The cubic host maps the x-axis site and its displaced neighbour onto the
    # y and z axes. The z moments make the z-axis image differ, so the result
    # shows the pattern was rotated rather than copied.
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0, 0, 0)))
    relaxed = relaxed_muon_supercell(_structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)})

    results = calculate_commensurate_fields(
        _structure(),
        _antiferromagnet(),
        [MuonSite(label="Mu", fractional_position=list(_MU_X))],
        options=MagneticFieldOptions(supercell=(1, 1, 1), find_equivalent_sites=True),
        distorted_structures=[relaxed],
    )

    by_position = {tuple(np.round(result.muon_position, 6)): result.distortion for result in results}
    assert set(by_position) == {(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)}
    for position in ((0.0, 0.5, 0.0), (0.0, 0.0, 0.5)):
        displaced = tuple(2.0 * value for value in position)
        direct = map_muon_distortion(
            _structure(),
            relaxed_muon_supercell(_structure(), (3, 3, 3), position, {displaced: tuple(0.2 * v for v in position)}),
            position,
        )
        expected = calculate_distortion_field_correction(_structure(), _antiferromagnet(), direct)
        assert by_position[position] == pytest.approx(expected, abs=1e-12)
    assert by_position[(0.0, 0.0, 0.5)][2] == pytest.approx(-2.0 * by_position[(0.5, 0.0, 0.0)][2])


def test_commensurate_fields_report_an_invalid_distorted_structure(monkeypatch) -> None:
    _forbid_muesr(monkeypatch)
    relaxed = relaxed_muon_supercell(_structure(), (2, 2, 2), _MU_X)

    with pytest.raises(MagneticFieldError, match="from the muon in the distorted structure"):
        calculate_commensurate_fields(
            _structure(),
            _model(),
            [(0.0, 0.0, 0.0)],
            distorted_structures=[relaxed],
        )


def test_commensurate_fields_need_one_distorted_entry_per_site(monkeypatch) -> None:
    _forbid_muesr(monkeypatch)

    with pytest.raises(MagneticFieldError, match="one entry per muon site"):
        calculate_commensurate_fields(_structure(), _model(), [_MU_X, _MU_X], distorted_structures=[None])


def test_incommensurate_distributions_refuse_distortion_corrections(monkeypatch) -> None:
    _forbid_muesr(monkeypatch)
    relaxed = relaxed_muon_supercell(_structure(), (2, 2, 2), _MU_X)

    with pytest.raises(MagneticFieldError, match="commensurate magnetic structures only"):
        calculate_incommensurate_field_distributions(
            _structure(),
            _circular_helix_model(),
            [_MU_X],
            distorted_structures=[relaxed],
        )


# Demagnetising correction: B_demag = -3/2 N . B_Lorentz


def _commensurate_demag(monkeypatch, **options) -> tuple[float, float, float]:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Lorentz=(1, 2, 3)))
    (result,) = calculate_commensurate_fields(
        _structure(),
        _model([0.0, 0.0, 0.5]),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), **options),
    )
    return result.demag


def test_demag_field_is_zero_without_a_sample_shape(monkeypatch) -> None:
    assert _commensurate_demag(monkeypatch) == (0.0, 0.0, 0.0)


def test_demag_field_applies_sphere_factor(monkeypatch) -> None:
    assert _commensurate_demag(monkeypatch, demag_shape="sphere") == pytest.approx((-0.5, -1.0, -1.5))


@pytest.mark.parametrize(
    ("axis", "index"),
    [("a", 0), ("x", 0), ("b", 1), ("y", 1), ("c", 2), ("z", 2), ("unknown", 2)],
)
def test_demag_field_for_infinite_plate_acts_along_the_normal(monkeypatch, axis: str, index: int) -> None:
    expected = [0.0, 0.0, 0.0]
    expected[index] = -1.5 * (index + 1)

    demag = _commensurate_demag(monkeypatch, demag_shape="infinite_plate", plate_normal_axis=axis)

    assert demag == pytest.approx(tuple(expected))


def test_demag_sample_takes_precedence_over_demag_shape(monkeypatch) -> None:
    demag = _commensurate_demag(
        monkeypatch,
        demag_shape="plate",
        plate_normal_axis="a",
        demag_sample=DemagnetizingSample(shape="ellipsoid", semi_axes=(1.0, 1.0, 1.0)),
    )

    assert demag == pytest.approx((-0.5, -1.0, -1.5))


def test_demag_field_rejects_unknown_shape(monkeypatch) -> None:
    with pytest.raises(MagneticFieldError, match="unknown demagnetizing shape"):
        _commensurate_demag(monkeypatch, demag_shape="cube")


# Dipolar tensors


def test_calculate_dipolar_tensors_for_atoms_uses_explicit_selector(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def _find_largest_sphere(_sample, supercell):
        calls["supercell"] = tuple(supercell)
        return 12.0

    def _dipten(sample, supercell, radius):
        calls["muons"] = list(sample.muons)
        calls["fc"] = np.asarray(sample.mm.fc).copy()
        calls["radius"] = radius
        return [np.eye(3)]

    _install_fake_muesr(monkeypatch, largest_sphere=_find_largest_sphere, dipten=_dipten)

    results = calculate_dipolar_tensors_for_atoms(
        _two_atom_structure(),
        [_SITE_A],
        magnetic_ion_selector=AtomSelector.species("Fe"),
        options=MagneticFieldOptions(supercell=(2, 3, 4)),
    )

    assert calls["supercell"] == (2, 3, 4)
    assert calls["radius"] == pytest.approx(12.0)
    assert calls["muons"] == [(0.25, 0.25, 0.25)]
    assert calls["fc"].shape == (2, 3)
    assert calls["fc"][0, 2] == pytest.approx(1.0)
    assert calls["fc"][1, 2] == pytest.approx(0.0)
    assert results[0].site_label == "A"
    assert results[0].muon_position == pytest.approx((0.25, 0.25, 0.25))
    assert np.asarray(results[0].tensor_T_per_muB) == pytest.approx(np.eye(3))
    assert np.asarray(results[0].to_json()["dipolar_tensor_T_per_muB"]) == pytest.approx(np.eye(3))


def test_calculate_dipolar_tensors_for_atoms_dedupes_and_sorts_indices(monkeypatch) -> None:
    captured: dict[str, np.ndarray] = {}

    def _dipten(sample, *_args):
        captured["fc"] = np.asarray(sample.mm.fc).copy()
        return [np.eye(3)]

    _install_fake_muesr(monkeypatch, dipten=_dipten)

    calculate_dipolar_tensors_for_atoms(
        _two_atom_structure(),
        [_SITE_A],
        magnetic_ion_indices=[1, 0, 1],
        options=MagneticFieldOptions(supercell=(1, 1, 1)),
    )

    np.testing.assert_allclose(captured["fc"][:, 2].real, [1.0, 1.0])


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ({"magnetic_ion_indices": []}, "choose at least one magnetic ion"),
        ({}, "choose at least one magnetic ion"),
        ({"magnetic_ion_indices": [0], "magnetic_ion_selector": AtomSelector.species("Fe")}, "not both"),
        ({"magnetic_ion_indices": ["not-an-int"]}, "must be integers"),
        ({"magnetic_ion_indices": [5]}, "out-of-range atom index"),
    ],
    ids=["empty-indices", "no-selection", "both", "non-integer", "out-of-range"],
)
def test_calculate_dipolar_tensors_for_atoms_validates_magnetic_ion_selection(monkeypatch, selection, message) -> None:
    _forbid_muesr(monkeypatch)

    with pytest.raises(MagneticFieldError, match=message):
        calculate_dipolar_tensors_for_atoms(_two_atom_structure(), [[0.25, 0.25, 0.25]], **selection)


@pytest.mark.parametrize(
    ("tensors", "message"),
    [
        ([np.zeros((2, 2))], "unexpected dipolar tensor shape"),
        ([np.eye(3), np.eye(3)], "unexpected number of dipolar tensors"),
    ],
    ids=["bad-shape", "wrong-count"],
)
def test_calculate_dipolar_tensors_for_atoms_rejects_malformed_muesr_output(monkeypatch, tensors, message) -> None:
    _install_fake_muesr(monkeypatch, dipten=lambda *_args: tensors)

    with pytest.raises(MagneticFieldError, match=message):
        calculate_dipolar_tensors_for_atoms(
            _two_atom_structure(),
            [_SITE_A],
            magnetic_ion_indices=[0],
            options=MagneticFieldOptions(supercell=(1, 1, 1)),
        )


def test_calculate_dipolar_tensors_compatibility_uses_assigned_atoms(monkeypatch) -> None:
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


# calculate_incommensurate_field_distributions: options and validation


def test_incommensurate_distribution_options_use_fine_defaults() -> None:
    options = MagneticFieldOptions()

    assert options.nangles == 4096
    assert options.phase_sample_count == 200_000
    assert options.phase_sampling == "auto"
    assert options.max_grid_samples == 1_000_000
    assert options.histogram_bins == 512


def test_incommensurate_distribution_rejects_invalid_histogram_bins() -> None:
    with pytest.raises(ValueError, match="histogram_bins must be positive"):
        calculate_incommensurate_field_distributions(
            _structure(),
            _model([0.12345, 0.0, 0.0]),
            [_SITE_A],
            options=MagneticFieldOptions(histogram_bins=0),
        )


def test_incommensurate_distribution_rejects_unknown_phase_sampling(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon())

    with pytest.raises(MagneticFieldError, match="phase_sampling must be one of"):
        calculate_incommensurate_field_distributions(
            _structure(),
            _circular_helix_model(),
            [_SITE_A],
            options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, phase_sampling="bogus"),
        )


def test_incommensurate_distribution_rejects_oversized_phase_grid(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon())

    with pytest.raises(MagneticFieldError, match="grid phase sampling is too large"):
        calculate_incommensurate_field_distributions(
            _structure(),
            _circular_helix_model(),
            [_SITE_A],
            options=MagneticFieldOptions(
                supercell=(1, 1, 1), nangles=8, phase_sample_count=8, phase_sampling="grid", max_grid_samples=4
            ),
        )


def test_incommensurate_distribution_rejects_non_finite_fields(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(np.inf, 0.0, 0.0)))

    with pytest.raises(MagneticFieldError, match="non-finite magnitudes"):
        calculate_incommensurate_field_distributions(
            _structure(),
            _circular_helix_model(),
            [_SITE_A],
            options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, phase_sampling="grid"),
        )


# calculate_incommensurate_field_distributions: muesr lookups


def test_incommensurate_distribution_splits_elliptical_components_before_muesr(monkeypatch) -> None:
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

    def _locfield(sample, mode, *_args, **_kwargs):
        assert mode == "i"
        captured_fcs.append(np.asarray(sample.mm.fc, dtype=complex).copy())
        return [_Field(Dipolar=np.zeros((4, 3)), Lorentz=np.zeros((4, 3)), Contact=np.zeros((4, 3)))]

    _install_fake_muesr(monkeypatch, locfield=_locfield)

    calculate_incommensurate_field_distributions(
        _structure(),
        model,
        [_SITE_A],
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


_LOOKUP = np.arange(12, dtype=float).reshape(4, 3)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"Dipolar": _LOOKUP}, _LOOKUP),
        ({"D": _LOOKUP}, _LOOKUP),
        ({"Dipolar": (1.0, 2.0, 3.0)}, np.tile([1.0, 2.0, 3.0], (4, 1))),
        ({"Dipolar": [[1.0, 2.0, 3.0]]}, np.tile([1.0, 2.0, 3.0], (4, 1))),
        ({}, np.zeros((4, 3))),
    ],
    ids=["per-angle", "fallback-attribute", "single-vector", "single-row", "missing"],
)
def test_incommensurate_distribution_reads_each_muesr_field_layout(monkeypatch, values, expected) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(**values))

    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, phase_sampling="grid", histogram_bins=4),
    )

    np.testing.assert_allclose(result.fields_T, expected)
    assert int(np.sum(result.histogram_counts)) == 4


def test_incommensurate_distribution_rejects_unexpected_muesr_field_shape(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=np.zeros((2, 3))))

    with pytest.raises(MagneticFieldError, match="unexpected field shape"):
        calculate_incommensurate_field_distributions(
            _structure(),
            _circular_helix_model(),
            [_SITE_A],
            options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, phase_sampling="grid"),
        )


# calculate_incommensurate_field_distributions: phase sampling


def test_incommensurate_distribution_combines_independent_multi_q_phase_grid(monkeypatch) -> None:
    def _locfield(sample, mode, *_args, **kwargs):
        assert mode == "i"
        nangles = int(kwargs["nangles"])
        values = np.zeros((nangles, 3), dtype=float)
        offset = 100.0 if abs(float(sample.mm.k[1])) > 0.0 else 0.0
        values[:, 0] = offset + np.arange(nangles, dtype=float)
        return [_Field(Dipolar=values, Lorentz=np.zeros_like(values), Contact=np.zeros_like(values))]

    _install_fake_muesr(monkeypatch, locfield=_locfield)

    result = calculate_incommensurate_field_distributions(
        _structure(),
        _two_q_helix_model(),
        [_SITE_A],
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


def test_incommensurate_distribution_grid_axis_count_follows_sample_budget(monkeypatch) -> None:
    # Two phase dimensions with 9 samples: min(nangles=4, floor(sqrt(9))=3) per axis.
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=_LOOKUP))

    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _two_q_helix_model(),
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1), nangles=4, phase_sample_count=9, phase_sampling="grid", histogram_bins=4
        ),
    )

    assert result.fields_T.shape == (9, 3)
    assert result.calculation_parameters["phase_sample_count"] == 9


def test_incommensurate_distribution_ignores_unused_q_basis_rows(monkeypatch) -> None:
    model = MagneticStructureModel(
        q_basis_rlu=((0.12345, 0.0, 0.0), (0.0, 0.23456, 0.0), (0.0, 0.0, 0.0)),
        components=(_helix("q1", (1.0, 0.0, 0.0)),),
    )
    _install_fake_muesr(
        monkeypatch,
        locfield=_fields_per_muon(Dipolar=np.zeros((4, 3)), Lorentz=np.zeros((4, 3)), Contact=np.zeros((4, 3))),
    )

    result = calculate_incommensurate_field_distributions(
        _structure(),
        model,
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, phase_sampling="grid", histogram_bins=4),
    )[0]

    assert result.fields_T.shape == (4, 3)
    assert result.calculation_parameters["independent_q_indices"] == [0]
    assert result.calculation_parameters["phase_sample_count"] == 4


def test_incommensurate_distribution_collapses_dependent_physical_q_rows(monkeypatch) -> None:
    model = MagneticStructureModel(
        q_basis_rlu=((0.12345, 0.0, 0.0), (0.24690, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(_helix("q1", (1.0, 0.0, 0.0)), _helix("q2", (0.0, 1.0, 0.0))),
    )

    def _locfield(sample, *_args, **_kwargs):
        offset = 100.0 if float(sample.mm.k[0]) > 0.2 else 0.0
        values = np.zeros((4, 3), dtype=float)
        values[:, 0] = offset + np.arange(4, dtype=float)
        return [_Field(Dipolar=values, Lorentz=np.zeros_like(values), Contact=np.zeros_like(values))]

    _install_fake_muesr(monkeypatch, locfield=_locfield)

    result = calculate_incommensurate_field_distributions(
        _structure(),
        model,
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, phase_sampling="grid", histogram_bins=4),
    )[0]

    assert result.fields_T.shape == (4, 3)
    np.testing.assert_allclose(result.fields_T[:, 0], [100.0, 103.0, 102.0, 105.0], atol=1e-12)
    assert result.calculation_parameters["independent_q_indices"] == [0, 1]
    assert result.calculation_parameters["phase_sample_count"] == 4


def test_incommensurate_distribution_holds_commensurate_components_at_zero_phase(monkeypatch) -> None:
    model = MagneticStructureModel(
        q_basis_rlu=((0.12345, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 0.0)),
        components=(
            _helix("incommensurate", (1.0, 0.0, 0.0)),
            MagneticComponent(
                component_id="commensurate",
                label="commensurate",
                kind="custom",
                selector=AtomSelector.all(),
                terms=(FourierTerm((0.0, 1.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 1.0])),),
            ),
        ),
    )

    def _locfield(_sample, mode, *_args, **_kwargs):
        if mode == "s":
            return [_Field(Dipolar=(0.0, 0.0, 7.0))]
        return [_Field(Dipolar=np.column_stack([np.arange(4.0), np.zeros(4), np.zeros(4)]))]

    _install_fake_muesr(monkeypatch, locfield=_locfield)

    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        model,
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, phase_sampling="grid", histogram_bins=4),
    )

    np.testing.assert_allclose(result.calculation_parameters["phase_basis_rlu"], [[0.12345, 0.0, 0.0]])
    np.testing.assert_allclose(result.fields_T[:, 0], np.arange(4.0))
    np.testing.assert_allclose(result.fields_T[:, 2], 7.0)


def test_incommensurate_distribution_of_commensurate_model_has_a_single_phase(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0.0, 0.0, 2.0)))

    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _model([0.0, 0.0, 0.5]),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, histogram_bins=4),
    )

    assert result.calculation_parameters["phase_basis_rlu"] == []
    assert result.calculation_parameters["phase_sample_count"] == 1
    np.testing.assert_allclose(result.fields_T, [[0.0, 0.0, 2.0]])


@pytest.mark.parametrize("sampling", ["montecarlo", "random"])
def test_incommensurate_distribution_monte_carlo_sampling_is_seeded(monkeypatch, sampling: str) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=_LOOKUP))
    options = MagneticFieldOptions(
        supercell=(1, 1, 1), nangles=4, phase_sample_count=7, phase_sampling=sampling, random_seed=123, histogram_bins=4
    )

    first = calculate_incommensurate_field_distributions(_structure(), _two_q_helix_model(), [_SITE_A], options=options)
    second = calculate_incommensurate_field_distributions(_structure(), _two_q_helix_model(), [_SITE_A], options=options)

    assert first[0].fields_T.shape == (7, 3)
    np.testing.assert_array_equal(first[0].fields_T, second[0].fields_T)


def test_auto_two_q_sampling_is_bounded_by_phase_sample_count() -> None:
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


# calculate_incommensurate_field_distributions: demagnetising correction


def test_incommensurate_distribution_applies_demag_to_sampled_lorentz_field(monkeypatch) -> None:
    def _locfield(_sample, mode, *_args, **kwargs):
        assert mode == "i"
        nangles = int(kwargs["nangles"])
        lorentz = np.column_stack([np.arange(nangles), 2.0 * np.arange(nangles), 3.0 * np.arange(nangles)])
        return [_Field(Dipolar=np.zeros((nangles, 3)), Lorentz=lorentz, Contact=np.zeros((nangles, 3)))]

    _install_fake_muesr(monkeypatch, locfield=_locfield)

    result = calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [_SITE_A],
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
    def _locfield(_sample, mode, *_args, **kwargs):
        assert mode == "i"
        assert kwargs["nangles"] == 3
        return [_Field(Dipolar=np.zeros(3), Lorentz=(0.0, 0.0, 2.0), Contact=np.zeros(3))]

    _install_fake_muesr(monkeypatch, locfield=_locfield)

    result = calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [_SITE_A],
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


# Internal guards the public entry points never reach


def test_field_magnitude_histogram_handles_near_delta_distribution() -> None:
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


def test_field_magnitude_histogram_guards_bin_count_and_empty_input() -> None:
    # Private helper: the public entry point validates histogram_bins and always samples at least one phase.
    with pytest.raises(MagneticFieldError, match="histogram_bins must be positive"):
        module._field_magnitude_histogram(np.array([1.0, 2.0]), bins=0)
    with pytest.raises(MagneticFieldError, match="no samples"):
        module._field_magnitude_histogram(np.array([], dtype=float), bins=4)


def test_phase_sampling_guards_angle_and_sample_counts() -> None:
    # Private helper: calculate_incommensurate_field_distributions rejects these counts before sampling.
    with pytest.raises(MagneticFieldError, match="nangles must be positive"):
        module._build_phase_samples_turns(
            n_dim=1, nangles=0, phase_sample_count=5, sampling="grid", random_seed=0, max_grid_samples=100
        )
    with pytest.raises(MagneticFieldError, match="phase_sample_count must be positive"):
        module._build_phase_samples_turns(
            n_dim=1, nangles=4, phase_sample_count=0, sampling="grid", random_seed=0, max_grid_samples=100
        )


def test_phase_basis_skips_zero_q_vectors() -> None:
    # Private helper: a zero q vector is commensurate, so the public path filters it out first.
    assert module._rank_increases([np.array([1.0, 0.0, 0.0])], np.zeros(3), tolerance=1e-9) is False


def test_active_q_indices_skip_malformed_q_coefficients() -> None:
    # Private helper: muesr model data built from a MagneticStructureModel always carries 3-vector q_coeffs.
    class _Model:
        def __init__(self, q_coeffs):
            self.q_coeffs = q_coeffs

    models = [
        _Model([(1.0, 0.0, 0.0)]),
        _Model([(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]),
        _Model([(0.0, 0.0)]),
        _Model(None),
    ]
    assert module._active_component_q_indices(models) == (0, 1, 2)


def test_component_sample_combination_guards_array_shapes() -> None:
    # Private helper: the public path always builds 2-D phases, (nangles, 3) lookups and matching coefficients.
    with pytest.raises(MagneticFieldError, match="phase_samples_turns must have shape"):
        module._combine_component_samples([], [], np.zeros(3), nangles=4)
    with pytest.raises(MagneticFieldError, match="component field lookup must have shape"):
        module._combine_component_samples([np.zeros((4, 2))], [(1.0,)], np.zeros((2, 1)), nangles=4)
    with pytest.raises(MagneticFieldError, match="phase coefficient length"):
        module._combine_component_samples([np.zeros((4, 3))], [(1.0,)], np.zeros((2, 2)), nangles=4)
