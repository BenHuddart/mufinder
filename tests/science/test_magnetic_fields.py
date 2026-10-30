"""Tests for magnetic field helper calculations.

The periodic calculations run through muesr. Most tests swap in a fake muesr
(``_install_fake_muesr``) so the public entry points run end to end on fields
the test controls. The engine-loading tests and the closing real-muesr section
run the installed engines; that section checks them against closed-form results
and direct lattice sums, and skips when muesr or muLFC cannot be imported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import cache, partial
import importlib
import json
import sys
from types import ModuleType

import numpy as np
import pytest

from mufinder.domain.constants import MUON_GAMMA_OVER_2PI_MHZ_T
from mufinder.domain.models import MuonSite, Structure
from mufinder.science import magnetic_fields as module
from mufinder.science._muesr_compat import muesr_imports, patch_numpy_aliases_for_muesr
from mufinder.science.demagnetization import DemagnetizingSample, principal_demagnetizing_tensor
from mufinder.science.magnetic_fields import (
    CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB,
    DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3,
    MAGNETIZATION_FIELD_T_PER_MUB_PER_ANGSTROM_MINUS3,
    MIN_MUON_ATOM_DISTANCE_ANG,
    DipolarTensorResult,
    MagneticFieldDependencyError,
    MagneticFieldDistributionResult,
    MagneticFieldError,
    MagneticFieldOptions,
    MagneticFieldResult,
    assigned_magnetic_atom_indices,
    calculate_commensurate_approximant_fields,
    calculate_commensurate_fields,
    calculate_distortion_field_correction,
    calculate_dipolar_tensors_for_atoms,
    calculate_finite_dipolar_field,
    calculate_incommensurate_field_distributions,
    classify_magnetic_model,
    contact_coupling_to_angstrom_minus3,
    find_commensurate_approximant,
    find_equivalent_muon_sites,
    is_commensurate_propagation_vector,
    muon_site_clearance_error,
    nearest_host_atom,
    regenerate_phase_samples,
)
from mufinder.science.magnetic_structure import (
    AtomSelector,
    ComplexVector,
    FourierTerm,
    MagneticComponent,
    MagneticStructureModel,
    moments_at_atoms,
)
from mufinder.science.muon_distortion import map_muon_distortion
from tests.support.codata import BOHR_MAGNETON_J_PER_T, MU0_OVER_4PI_T_M_PER_A, MU0_T_M_PER_A
from tests.support.mnsi import (
    MNSI_MOMENT,
    MNSI_MUON,
    MNSI_SITE_II,
    mnsi_helix,
    mnsi_options,
    mnsi_structure,
)
from tests.support.structures import BODY_CENTRE, host_structure, relaxed_muon_supercell

_structure = partial(
    host_structure, label="Fe", lattice=3.0, species=("Fe",), positions=((0.0, 0.0, 0.0),)
)
_two_atom_structure = partial(host_structure, label="FeO", lattice=3.0, species=("Fe", "O"))
_SITE_A = MuonSite(label="A", initial_fractional_position=[0.25, 0.25, 0.25])


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
        terms=(
            FourierTerm(q_coeff, ComplexVector(real=(1.0, 0.0, 0.0), imaginary=(0.0, 1.0, 0.0))),
        ),
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


def _component(
    selector: AtomSelector, *, enabled: bool = True, moment: float = 1.0
) -> MagneticComponent:
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
    monkeypatch.setattr(
        module, "_load_muesr_engines", lambda: pytest.fail("muesr should not be loaded")
    )


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
    # Import muesr.core first so the test does not depend on an earlier test having imported it.
    with muesr_imports():
        core = importlib.import_module("muesr.core")
    # Python 3.13 removed cgi; a None entry makes ``import cgi`` fail on any version.
    monkeypatch.setitem(sys.modules, "cgi", None)
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
    assert "calculation_parameters" not in payload


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


@pytest.mark.parametrize(
    ("distortion", "parameters"),
    [(None, {}), ((0.5, 0.0, 0.25), {"contact_coupling": 0.1, "nnn": 4, "rcont": 3.5})],
)
def test_magnetic_field_result_json_round_trips(distortion, parameters) -> None:
    result = MagneticFieldResult(
        site_label="A",
        muon_position=(0.1, 0.2, 0.3),
        dipolar=(1.0, -2.0, 0.0),
        lorentz=(0.1, 0.0, 0.0),
        contact=(0.0, 0.2, 0.0),
        demag=(0.0, 0.0, -0.3),
        applied=(0.0, 0.0, 0.05),
        distortion=distortion,
        calculation_parameters=parameters,
    )

    restored = MagneticFieldResult.from_json(json.loads(json.dumps(result.to_json())))

    assert restored == result


def test_magnetic_field_result_from_json_refuses_a_payload_without_its_terms() -> None:
    payload = MagneticFieldResult(
        site_label="A",
        muon_position=(0.0, 0.0, 0.0),
        dipolar=(1.0, 0.0, 0.0),
        lorentz=(0.0, 0.0, 0.0),
        contact=(0.0, 0.0, 0.0),
        demag=(0.0, 0.0, 0.0),
        applied=(0.0, 0.0, 0.0),
    ).to_json()
    del payload["dipolar_T"]

    with pytest.raises(KeyError):
        MagneticFieldResult.from_json(payload)


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
    assert payload["dipolar_tensor_T_per_muB"] == [
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ]


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
    assert (
        classify_magnetic_model(_structure(), _model([0.12345, 0.0, 0.0]), max_period=8)
        == "incommensurate"
    )


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

    expected = MU0_OVER_4PI_T_M_PER_A * 2.0 * BOHR_MAGNETON_J_PER_T / ((1e-10) ** 3)
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
    assert contact_coupling_to_angstrom_minus3(
        MUON_GAMMA_OVER_2PI_MHZ_T, "MHz_per_muB"
    ) == pytest.approx(contact_coupling_to_angstrom_minus3(1.0, "T_per_muB"))


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


def test_nearest_host_atom_finds_the_closest_periodic_image() -> None:
    # Fe sits at the origin of a 3 Å cube, so (0.95, 0, 0) is 0.15 Å from its image at (1, 0, 0).
    nearest = nearest_host_atom(_two_atom_structure(), [0.95, 0.0, 0.0])

    assert nearest is not None
    assert nearest.index == 0
    assert nearest.species == "Fe"
    assert nearest.distance_ang == pytest.approx(0.15)
    assert nearest_host_atom(_structure(species=(), positions=()), [0.1, 0.1, 0.1]) is None


def test_muon_site_clearance_names_the_nearest_atom_and_its_distance() -> None:
    message = muon_site_clearance_error(_two_atom_structure(), [0.5, 0.5, 0.55], label="Mu1")

    assert message == (
        "Mu1 is 0.15 Å from O (atom 2); a muon site must be at least 0.5 Å from every atom."
    )
    assert muon_site_clearance_error(_two_atom_structure(), [0.25, 0.25, 0.25]) is None
    assert MIN_MUON_ATOM_DISTANCE_ANG == 0.5


@pytest.mark.parametrize(
    "calculate",
    [
        calculate_commensurate_fields,
        calculate_commensurate_approximant_fields,
        calculate_incommensurate_field_distributions,
    ],
)
def test_field_calculations_refuse_a_muon_site_on_an_atom(monkeypatch, calculate) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon())

    with pytest.raises(MagneticFieldError, match=r"site 1 is 0\.00 Å from Fe \(atom 1\)"):
        calculate(
            _two_atom_structure(),
            _model([0.0, 0.0, 0.5]),
            [[0.0, 0.0, 0.0]],
            options=MagneticFieldOptions(supercell=(1, 1, 1)),
        )


def test_dipolar_tensors_refuse_a_muon_site_within_the_minimum_distance(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, dipten=lambda *_args: [np.eye(3)])

    with pytest.raises(MagneticFieldError, match=r"Mu is 0\.30 Å from O \(atom 2\)"):
        calculate_dipolar_tensors_for_atoms(
            _two_atom_structure(),
            [MuonSite(label="Mu", initial_fractional_position=[0.6, 0.5, 0.5])],
            magnetic_ion_indices=[0],
            options=MagneticFieldOptions(supercell=(1, 1, 1)),
        )


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
            MuonSite(label="", initial_fractional_position=[1.5, -0.25, -1e-12]),
            [0.5, 0.5, 0.5],
            MuonSite(label="Mu1", initial_fractional_position=[0.1, 0.2, 0.3]),
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
        [MuonSite(label="Mu", initial_fractional_position=[0.25, 0.25, 0.25])],
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
def test_commensurate_fields_read_each_muesr_field_layout(
    monkeypatch, values, expected_dipolar
) -> None:
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
    # The q = (0, 0, 1/2) order has no bulk magnetisation, so no demagnetizing field.
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


# The continuum outside the Lorentz sphere, -mu0 (3 q̂q̂ - 1) . M j1(qR) / (qR) for a mode
# at q: muesr's Lorentz term is the isotropic part, MuFinder adds -3 q̂ (q̂ . B_Lorentz).


@pytest.mark.parametrize(
    ("q", "lorentz", "modulated"),
    [
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), False),
        ((0.0, 0.0, 0.25), (1.0, 1.0, -2.0), True),
        ((0.0, 0.0, 0.9), (1.0, 1.0, -2.0), True),
        ((1.0, 0.0, 0.25), (1.0, 1.0, -2.0), True),
        ((0.25, 0.0, 0.0), (-2.0, 1.0, 1.0), True),
        ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), False),
    ],
    ids=["uniform", "along-c", "near-a-zone-centre", "shifted-by-a", "along-a", "at-G"],
)
def test_commensurate_lorentz_term_includes_the_continuum_along_q(
    monkeypatch, q, lorentz, modulated
) -> None:
    # The mode's shortest wavevector sets q̂: q = (0, 0, 0.9) acts as (0, 0, -0.1),
    # (1, 0, 0.25) as (0, 0, 0.25), and q = (0, 0, 1) is uniform on the lattice.
    _install_fake_muesr(
        monkeypatch,
        locfield=_fields_per_muon(
            Dipolar=(0.5, 0.0, 0.0), Lorentz=(1.0, 1.0, 1.0), Contact=(0.0, 0.0, 0.0)
        ),
    )

    (result,) = calculate_commensurate_fields(
        _structure(), _model(q), [_SITE_A], options=MagneticFieldOptions(supercell=(1, 1, 1))
    )

    assert result.dipolar == pytest.approx((0.5, 0.0, 0.0))
    assert result.lorentz == pytest.approx(lorentz)
    assert result.calculation_parameters["modulated"] is modulated
    assert MagneticFieldResult.from_json(result.to_json()).calculation_parameters == (
        result.calculation_parameters
    )


@pytest.mark.parametrize(
    ("lattice", "q", "lorentz"),
    [
        # A sheared basis of the 3 Å simple cubic lattice: q = 0.4 a* is 0.4 x̂ / 3 Å
        # plus 2 b*, so the shortest equivalent wavevector lies along x.
        ([[3.0, 0.0, 0.0], [15.0, 3.0, 0.0], [0.0, 0.0, 3.0]], (0.4, 0.0, 0.0), (-2.0, 1.0, 1.0)),
        # At the zone-boundary point (1/2, 1/2, 0) four wavevectors tie; their
        # average q̂q̂ is diag(1/2, 1/2, 0).
        (3.0, (0.5, 0.5, 0.0), (-0.5, -0.5, 1.0)),
    ],
    ids=["sheared-cell", "zone-boundary-tie"],
)
def test_commensurate_continuum_term_uses_the_shortest_equivalent_wavevector(
    monkeypatch, lattice, q, lorentz
) -> None:
    _install_fake_muesr(
        monkeypatch,
        locfield=_fields_per_muon(
            Dipolar=(0.0, 0.0, 0.0), Lorentz=(1.0, 1.0, 1.0), Contact=(0.0, 0.0, 0.0)
        ),
    )

    (result,) = calculate_commensurate_fields(
        _structure(lattice=lattice),
        _model(q),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1)),
    )

    assert result.lorentz == pytest.approx(lorentz)


def test_commensurate_continuum_term_is_left_out_with_the_lorentz_field(monkeypatch) -> None:
    _install_fake_muesr(
        monkeypatch,
        locfield=_fields_per_muon(
            Dipolar=(0.5, 0.0, 0.0), Lorentz=(1.0, 1.0, 1.0), Contact=(0.0, 0.0, 0.0)
        ),
    )

    (result,) = calculate_commensurate_fields(
        _structure(),
        _model((0.0, 0.0, 0.25)),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), include_lorentz=False),
    )

    assert result.lorentz == (0.0, 0.0, 0.0)
    assert result.total == pytest.approx((0.5, 0.0, 0.0))


def test_distribution_lorentz_term_includes_the_continuum_along_q(monkeypatch) -> None:
    def _locfield(_sample, mode, *_args, **kwargs):
        assert mode == "i"
        nangles = int(kwargs["nangles"])
        lorentz = np.tile([1.0, 1.0, 1.0], (nangles, 1))
        return [
            _Field(Dipolar=np.zeros((nangles, 3)), Lorentz=lorentz, Contact=np.zeros((nangles, 3)))
        ]

    _install_fake_muesr(monkeypatch, locfield=_locfield)

    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _model((0.12345, 0.0, 0.0)),
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1), nangles=4, phase_sampling="grid", histogram_bins=4
        ),
    )

    # Each circular part of the spin-density wave contributes (-2, 1, 1).
    np.testing.assert_allclose(result.fields_T, np.tile([-4.0, 2.0, 2.0], (4, 1)), atol=1e-12)
    assert result.terms["lorentz"].mean_T == pytest.approx((-4.0, 2.0, 2.0))
    assert result.calculation_parameters["modulated"] is True


# Muon-induced distortion correction: finite-cell field of displaced minus pristine ions

_MU_X = (0.5, 0.0, 0.0)


def _axial_dipole_change(moment: float, before: float, after: float) -> float:
    """B_z change at a muon on the x axis when a z moment moves from ``before`` to ``after`` Å away."""

    prefactor = -MU0_OVER_4PI_T_M_PER_A * BOHR_MAGNETON_J_PER_T / ((1e-10) ** 3)
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
    relaxed = relaxed_muon_supercell(
        _structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)}
    )
    distortion = map_muon_distortion(_structure(), relaxed, _MU_X)

    correction = calculate_distortion_field_correction(_structure(), model(), distortion)

    assert correction[0] == pytest.approx(0.0, abs=1e-12)
    assert correction[1] == pytest.approx(0.0, abs=1e-12)
    assert correction[2] == pytest.approx(_axial_dipole_change(moment, 1.5, 1.6))


def test_distortion_correction_is_zero_without_displacements() -> None:
    distortion = map_muon_distortion(
        _structure(), relaxed_muon_supercell(_structure(), (2, 2, 2), _MU_X), _MU_X
    )

    assert calculate_distortion_field_correction(
        _structure(), _antiferromagnet(), distortion
    ) == pytest.approx((0.0, 0.0, 0.0), abs=1e-15)


def test_distortion_correction_indexes_moments_past_a_muon_marker_in_the_host() -> None:
    # A muon marker listed before the Fe must not shift which atom each ion takes its moment from.
    marked = host_structure(
        label="Fe+mu",
        lattice=3.0,
        species=("H:mu", "Fe"),
        positions=((0.25, 0.25, 0.25), (0.0, 0.0, 0.0)),
    )
    model = MagneticStructureModel(
        q_basis_rlu=((0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="fe",
                label="Fe",
                kind="custom",
                selector=AtomSelector.species("Fe"),
                terms=(FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 1.0])),),
            ),
        ),
    )
    relaxed = relaxed_muon_supercell(
        _structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)}
    )
    distortion = map_muon_distortion(marked, relaxed, _MU_X)

    assert set(distortion.atom_indices.tolist()) == {1}
    correction = calculate_distortion_field_correction(marked, model, distortion)
    assert correction[2] == pytest.approx(_axial_dipole_change(-1.0, 1.5, 1.6))


def test_commensurate_fields_add_the_distortion_correction(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0, 0, 1)))
    relaxed = relaxed_muon_supercell(
        _structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)}
    )

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
    relaxed = relaxed_muon_supercell(
        _structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)}
    )

    results = calculate_commensurate_fields(
        _structure(),
        _antiferromagnet(),
        [MuonSite(label="Mu", initial_fractional_position=list(_MU_X))],
        options=MagneticFieldOptions(supercell=(1, 1, 1), find_equivalent_sites=True),
        distorted_structures=[relaxed],
    )

    by_position = {
        tuple(np.round(result.muon_position, 6)): result.distortion for result in results
    }
    assert set(by_position) == {(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)}
    for position in ((0.0, 0.5, 0.0), (0.0, 0.0, 0.5)):
        displaced = tuple(2.0 * value for value in position)
        direct = map_muon_distortion(
            _structure(),
            relaxed_muon_supercell(
                _structure(), (3, 3, 3), position, {displaced: tuple(0.2 * v for v in position)}
            ),
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
            [(0.25, 0.25, 0.25)],
            distorted_structures=[relaxed],
        )


def test_commensurate_fields_need_one_distorted_entry_per_site(monkeypatch) -> None:
    _forbid_muesr(monkeypatch)

    with pytest.raises(MagneticFieldError, match="one entry per muon site"):
        calculate_commensurate_fields(
            _structure(), _model(), [_MU_X, _MU_X], distorted_structures=[None]
        )


def _contact_change(coupling: float, before, after) -> float:
    """Contact field change in T for muLFC's 1/r^3-weighted mean of the shell moments.

    ``before`` and ``after`` list each shell ion as ``(distance in Å, z moment)``.
    """

    def _mean(shell) -> float:
        weights = [distance**-3 for distance, _moment in shell]
        return sum(w * m for w, (_d, m) in zip(weights, shell)) / sum(weights)

    per_mub = 2.0 * MU0_T_M_PER_A / 3.0 * BOHR_MAGNETON_J_PER_T / (1e-10) ** 3
    return coupling * per_mub * (_mean(after) - _mean(before))


@pytest.mark.parametrize(
    ("shift", "after"),
    [(0.6, [(1.5, 1.0)]), (0.3, [(1.5, 1.0), (1.8, -1.0)])],
    ids=["leaves-the-shell", "moves-within-the-shell"],
)
def test_commensurate_distortion_corrects_the_contact_term(monkeypatch, shift, after) -> None:
    # The muon sits between two Fe 1.5 Å away with opposite moments. Moving the
    # +x one out of the 2 Å contact radius leaves only the other in the shell;
    # moving it less reweights the pair by 1/r^3.
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0, 0, 0)))
    relaxed = relaxed_muon_supercell(
        _structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (shift, 0.0, 0.0)}
    )
    options = MagneticFieldOptions(supercell=(1, 1, 1), contact_coupling=0.1, nnn=2, rcont=2.0)

    (result,) = calculate_commensurate_fields(
        _structure(), _antiferromagnet(), [_MU_X], options=options, distorted_structures=[relaxed]
    )

    dipolar_change = _axial_dipole_change(-1.0, 1.5, 1.5 + shift)
    contact_change = _contact_change(0.1, [(1.5, 1.0), (1.5, -1.0)], after)
    assert result.distortion == pytest.approx(
        (0.0, 0.0, dipolar_change + contact_change), abs=1e-12
    )


def test_distortion_without_a_contact_coupling_leaves_the_contact_term(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0, 0, 0)))
    relaxed = relaxed_muon_supercell(
        _structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (0.6, 0.0, 0.0)}
    )

    (result,) = calculate_commensurate_fields(
        _structure(),
        _antiferromagnet(),
        [_MU_X],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nnn=2, rcont=2.0),
        distorted_structures=[relaxed],
    )

    assert result.distortion == pytest.approx(
        (0.0, 0.0, _axial_dipole_change(-1.0, 1.5, 2.1)), abs=1e-12
    )


def test_contact_distortion_needs_the_shell_inside_the_relaxed_supercell(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0, 0, 0)))
    relaxed = relaxed_muon_supercell(_structure(), (2, 2, 2), _MU_X)

    # Fewer than 50 ions lie within 10 Å of the muon, so the shell reaches the radius.
    with pytest.raises(
        MagneticFieldError,
        match=r"contact shell reaches 10\.00 Å.*a smaller Contact radius \(only \d+ magnetic ions",
    ):
        calculate_commensurate_fields(
            _structure(),
            _antiferromagnet(),
            [_MU_X],
            options=MagneticFieldOptions(supercell=(1, 1, 1), contact_coupling=0.1, nnn=50),
            distorted_structures=[relaxed],
        )


def test_a_full_contact_shell_past_the_relaxed_supercell_asks_for_fewer_neighbours(
    monkeypatch,
) -> None:
    # The four Fe after the two 1.5 Å from the muon lie 3.35 Å away, past the 2x2x2 supercell.
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0, 0, 0)))
    relaxed = relaxed_muon_supercell(_structure(), (2, 2, 2), _MU_X)

    with pytest.raises(
        MagneticFieldError, match=r"contact shell reaches 3\.35 Å.*fewer Contact neighbours"
    ):
        calculate_commensurate_fields(
            _structure(),
            _antiferromagnet(),
            [_MU_X],
            options=MagneticFieldOptions(supercell=(1, 1, 1), contact_coupling=0.1, nnn=6),
            distorted_structures=[relaxed],
        )


def test_contact_distortion_tie_is_resolved_by_including_the_whole_tied_set(monkeypatch) -> None:
    # The two Fe that tie 1.5 Å from the muon for one place both fit in a two-place shell.
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0, 0, 0)))
    relaxed = relaxed_muon_supercell(_structure(), (3, 3, 3), _MU_X)
    options = MagneticFieldOptions(supercell=(1, 1, 1), contact_coupling=0.1, nnn=2, rcont=2.0)

    (result,) = calculate_commensurate_fields(
        _structure(), _antiferromagnet(), [_MU_X], options=options, distorted_structures=[relaxed]
    )

    assert result.distortion == pytest.approx((0.0, 0.0, 0.0), abs=1e-15)


def test_distribution_checks_the_contact_shell_before_any_lattice_sum(monkeypatch) -> None:
    _install_fake_muesr(
        monkeypatch, locfield=lambda *_a, **_k: pytest.fail("no lattice sum should run")
    )

    with pytest.raises(MagneticFieldError, match="contact shell reaches"):
        calculate_incommensurate_field_distributions(
            _structure(),
            _circular_helix_model(),
            [_MU_X],
            options=MagneticFieldOptions(supercell=(1, 1, 1), contact_coupling=0.1, nnn=50),
            distorted_structures=[relaxed_muon_supercell(_structure(), (2, 2, 2), _MU_X)],
        )


def _zero_field_locfield(sample, mode, *_args, **kwargs):
    rows = int(kwargs["nangles"]) if mode == "i" else 1
    return [_Field(Dipolar=np.zeros((rows, 3))) for _ in sample.muons]


@pytest.mark.parametrize("coupling", [None, 0.1], ids=["dipolar", "with-contact"])
def test_distribution_of_a_uniform_order_gets_the_commensurate_correction(
    monkeypatch, coupling
) -> None:
    # A q = 0 order runs through the distribution path as a single phase.
    _install_fake_muesr(monkeypatch, locfield=_zero_field_locfield)
    relaxed = relaxed_muon_supercell(
        _structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (0.3, 0.1, 0.0)}
    )
    options = MagneticFieldOptions(supercell=(1, 1, 1), contact_coupling=coupling, rcont=2.0)

    (distribution,) = calculate_incommensurate_field_distributions(
        _structure(), _ferromagnet(), [_MU_X], options=options, distorted_structures=[relaxed]
    )
    (commensurate,) = calculate_commensurate_fields(
        _structure(), _ferromagnet(), [_MU_X], options=options, distorted_structures=[relaxed]
    )

    assert np.linalg.norm(commensurate.distortion) > 1e-3
    np.testing.assert_allclose(
        distribution.fields_T, [commensurate.distortion], rtol=1e-12, atol=1e-15
    )


def _helix_distribution(monkeypatch, relaxed, **overrides) -> MagneticFieldDistributionResult:
    _install_fake_muesr(monkeypatch, locfield=_zero_field_locfield)
    options = {"supercell": (1, 1, 1), "nangles": 16, "phase_sampling": "grid", **overrides}
    return calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [_MU_X],
        options=MagneticFieldOptions(**options),
        distorted_structures=[relaxed],
    )[0]


def test_distribution_without_displacements_is_unchanged(monkeypatch) -> None:
    result = _helix_distribution(
        monkeypatch, relaxed_muon_supercell(_structure(), (3, 3, 3), _MU_X), contact_coupling=0.1
    )

    np.testing.assert_allclose(result.fields_T, 0.0, atol=1e-15)
    assert result.calculation_parameters["distortion_corrected"] is True
    assert MagneticFieldDistributionResult.from_json(result.to_json()).calculation_parameters[
        "distortion_corrected"
    ]


def test_distribution_correction_walks_the_helix_phase(monkeypatch) -> None:
    # One Fe moves along the muon-Fe axis; its helical moment turns in the ab
    # plane with the phase, so the correction is that one dipole change, rotated.
    relaxed = relaxed_muon_supercell(
        _structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)}
    )

    result = _helix_distribution(monkeypatch, relaxed)

    magnitudes = np.linalg.norm(result.fields_T, axis=1)
    radial = 2.0 * abs(_axial_dipole_change(1.0, 1.5, 1.6))
    transverse = abs(_axial_dipole_change(1.0, 1.5, 1.6))
    assert np.all(magnitudes <= radial * (1.0 + 1e-9))
    assert np.all(magnitudes >= transverse * (1.0 - 1e-9))
    assert magnitudes.max() > 0.95 * radial
    assert magnitudes.min() < 1.05 * transverse
    assert result.terms["dipolar"].rms_T > 0.0
    assert {check.setting for check in result.convergence} == {
        "nangles",
        "phase_sample_count",
        "histogram_bins",
    }


# Demagnetising correction: B_demag = -mu0 N M_meas, with M_meas the bulk
# magnetisation of the model's uniform (q = 0) order in its (3 Å)^3 cell.

_UNIFORM_MOMENT = (1.0, 2.0, 3.0)


def _bulk_field(moment=_UNIFORM_MOMENT) -> np.ndarray:
    """mu0 M in T for ``moment`` mu_B per (3 Å)^3 cell."""

    return (
        module.MAGNETIZATION_FIELD_T_PER_MUB_PER_ANGSTROM_MINUS3
        * np.asarray(moment, dtype=float)
        / 27.0
    )


def _commensurate_demag(monkeypatch, model=None, **options) -> tuple[float, float, float]:
    # The fake Lorentz field is deliberately unrelated to the model's magnetisation.
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Lorentz=(5, 6, 7)))
    (result,) = calculate_commensurate_fields(
        _structure(),
        model or _single_q_model((0.0, 0.0, 0.0), _UNIFORM_MOMENT),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), **options),
    )
    return result.demag


def test_demag_field_is_zero_without_a_sample_shape(monkeypatch) -> None:
    assert _commensurate_demag(monkeypatch) == (0.0, 0.0, 0.0)


def test_demag_field_for_a_sphere_is_a_third_of_the_bulk_magnetisation(monkeypatch) -> None:
    assert _commensurate_demag(monkeypatch, demag_shape="sphere") == pytest.approx(
        tuple(-_bulk_field() / 3.0)
    )


@pytest.mark.parametrize(
    ("axis", "index"),
    [
        ("a", 0),
        ("x", 0),
        ("A", 0),
        ("b", 1),
        ("y", 1),
        ("c", 2),
        ("z", 2),
        ("unknown", 2),
        (" a", 2),
    ],
)
def test_demag_field_for_infinite_plate_acts_along_the_normal(
    monkeypatch, axis: str, index: int
) -> None:
    expected = [0.0, 0.0, 0.0]
    expected[index] = -_bulk_field()[index]

    demag = _commensurate_demag(monkeypatch, demag_shape="infinite_plate", plate_normal_axis=axis)

    assert demag == pytest.approx(tuple(expected))


@pytest.mark.parametrize(
    ("demag_shape", "plate_normal_axis", "sample"),
    [
        ("sphere", "c", DemagnetizingSample(shape="sphere")),
        ("Sphere ", "c", DemagnetizingSample(shape="sphere")),
        *(
            (shape, axis, DemagnetizingSample(shape="infinite_plate", plate_normal_axis=axis))
            for shape in ("infinite_plate", "plate", "infinite plate")
            for axis in ("a", "b", "c", "x", "y", "z")
        ),
        ("plate", "unknown", DemagnetizingSample(shape="infinite_plate", plate_normal_axis="c")),
    ],
)
def test_demag_shape_gives_the_same_field_as_the_equivalent_sample(
    monkeypatch, demag_shape: str, plate_normal_axis: str, sample: DemagnetizingSample
) -> None:
    by_shape = _commensurate_demag(
        monkeypatch, demag_shape=demag_shape, plate_normal_axis=plate_normal_axis
    )

    assert by_shape == _commensurate_demag(monkeypatch, demag_sample=sample)


def test_demag_sample_takes_precedence_over_demag_shape(monkeypatch) -> None:
    demag = _commensurate_demag(
        monkeypatch,
        demag_shape="plate",
        plate_normal_axis="a",
        demag_sample=DemagnetizingSample(shape="ellipsoid", semi_axes=(1.0, 1.0, 1.0)),
    )

    assert demag == pytest.approx(tuple(-_bulk_field() / 3.0))


def test_demag_field_of_an_oriented_sample_uses_the_rotated_tensor(monkeypatch) -> None:
    # A plate normal to (110) of the cubic cell takes only the [110] part of mu0 M.
    plate = DemagnetizingSample.oriented(
        np.eye(3) * 3.0, (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), shape="infinite_plate"
    )
    normal = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)

    demag = _commensurate_demag(monkeypatch, demag_sample=plate)

    assert demag == pytest.approx(tuple(-normal * (normal @ _bulk_field())))


def test_demag_field_of_a_photographed_plate_uses_its_fourier_tensor(monkeypatch) -> None:
    x = (np.arange(48) + 0.5) / 48 * 2.0 - 1.0
    outline = x[:, None] ** 2 + (x[None, :] / 0.8) ** 2 <= 1.0
    plate = DemagnetizingSample.image_plate(outline, (5.0, 4.0), 0.1, padding=2.0)

    demag = _commensurate_demag(monkeypatch, demag_sample=plate)

    tensor = principal_demagnetizing_tensor(plate)
    assert tensor[2, 2] > 0.9
    assert demag == pytest.approx(tuple(-tensor @ _bulk_field()))


def test_demag_field_is_zero_for_order_without_a_uniform_moment(monkeypatch) -> None:
    # An antiferromagnet has no bulk magnetisation, whatever Lorentz field muesr reports.
    demag = _commensurate_demag(
        monkeypatch, _single_q_model((0.0, 0.0, 0.5), _UNIFORM_MOMENT), demag_shape="sphere"
    )

    assert demag == (0.0, 0.0, 0.0)


def test_demag_field_counts_order_at_a_reciprocal_lattice_vector_as_uniform(monkeypatch) -> None:
    # q = (1, 0, 0) gives every cell the same moment, so it magnetises the sample.
    demag = _commensurate_demag(
        monkeypatch, _single_q_model((1.0, 0.0, 0.0), _UNIFORM_MOMENT), demag_shape="sphere"
    )

    assert demag == pytest.approx(tuple(-_bulk_field() / 3.0))


def test_demag_field_rejects_unknown_shape(monkeypatch) -> None:
    with pytest.raises(MagneticFieldError, match="unknown demagnetizing shape"):
        _commensurate_demag(monkeypatch, demag_shape="cube")


# A measured magnetisation, in mu_B per formula unit, replaces the model's uniform moment.

_MEASURED_MOMENT = (0.1, -0.2, 0.05)


def test_measured_magnetisation_replaces_the_model_uniform_moment(monkeypatch) -> None:
    demag = _commensurate_demag(
        monkeypatch, demag_shape="sphere", measured_magnetization=_MEASURED_MOMENT
    )

    assert demag == pytest.approx(tuple(-_bulk_field(_MEASURED_MOMENT) / 3.0))


def test_measured_magnetisation_gives_a_demag_field_for_order_without_a_uniform_moment(
    monkeypatch,
) -> None:
    demag = _commensurate_demag(
        monkeypatch,
        _single_q_model((0.0, 0.0, 0.5), _UNIFORM_MOMENT),
        demag_shape="infinite_plate",
        plate_normal_axis="c",
        measured_magnetization=_MEASURED_MOMENT,
    )

    assert demag == pytest.approx((0.0, 0.0, -_bulk_field(_MEASURED_MOMENT)[2]))


def test_measured_magnetisation_has_no_effect_without_a_sample_shape(monkeypatch) -> None:
    assert _commensurate_demag(monkeypatch, measured_magnetization=_MEASURED_MOMENT) == (
        0.0,
        0.0,
        0.0,
    )


def test_measured_magnetisation_counts_formula_units_per_cell(monkeypatch) -> None:
    # Two Fe atoms in the (3 Å)^3 cell are two formula units of Fe.
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Lorentz=(5, 6, 7)))
    (result,) = calculate_commensurate_fields(
        host_structure(label="Fe2", lattice=3.0, species=("Fe", "Fe"), positions=BODY_CENTRE),
        _single_q_model((0.0, 0.0, 0.0), _UNIFORM_MOMENT),
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1),
            demag_shape="sphere",
            measured_magnetization=_MEASURED_MOMENT,
        ),
    )

    assert result.demag == pytest.approx(tuple(-2.0 * _bulk_field(_MEASURED_MOMENT) / 3.0))


@pytest.mark.parametrize(
    ("measured", "message"),
    [
        (1.0, "three-vector"),
        ((1.0, 2.0), "three-vector"),
        ((1.0, float("nan"), 0.0), "finite"),
        ((1.0, float("inf"), 0.0), "finite"),
        (("x", 0.0, 0.0), "three numbers"),
    ],
)
def test_measured_magnetisation_is_validated(monkeypatch, measured, message) -> None:
    with pytest.raises(MagneticFieldError, match=message):
        _commensurate_demag(monkeypatch, demag_shape="sphere", measured_magnetization=measured)


def test_measured_magnetisation_applies_to_every_approximant_translation(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_phase_recording_locfield)

    results = calculate_commensurate_approximant_fields(
        _structure(),
        _circular_helix_model((0.3334, 0.0, 0.0)),
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1), demag_shape="sphere", measured_magnetization=_MEASURED_MOMENT
        ),
    )

    # The 3 x 1 x 1 approximant holds three formula units in three times the volume.
    assert len(results) == 3
    for result in results:
        assert result.demag == pytest.approx(tuple(-_bulk_field(_MEASURED_MOMENT) / 3.0))


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
    # muesr's tensor is in Å^-3; the result carries it in T/mu_B.
    expected = DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3 * np.eye(3)
    assert DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3 == pytest.approx(0.9274, abs=1e-4)
    assert np.asarray(results[0].tensor_T_per_muB) == pytest.approx(expected)
    assert np.asarray(results[0].to_json()["dipolar_tensor_T_per_muB"]) == pytest.approx(expected)


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
        (
            {"magnetic_ion_indices": [0], "magnetic_ion_selector": AtomSelector.species("Fe")},
            "not both",
        ),
        ({"magnetic_ion_indices": ["not-an-int"]}, "must be integers"),
        ({"magnetic_ion_indices": [5]}, "out-of-range atom index"),
    ],
    ids=["empty-indices", "no-selection", "both", "non-integer", "out-of-range"],
)
def test_calculate_dipolar_tensors_for_atoms_validates_magnetic_ion_selection(
    monkeypatch, selection, message
) -> None:
    _forbid_muesr(monkeypatch)

    with pytest.raises(MagneticFieldError, match=message):
        calculate_dipolar_tensors_for_atoms(
            _two_atom_structure(), [[0.25, 0.25, 0.25]], **selection
        )


@pytest.mark.parametrize(
    ("tensors", "message"),
    [
        ([np.zeros((2, 2))], "unexpected dipolar tensor shape"),
        ([np.eye(3), np.eye(3)], "unexpected number of dipolar tensors"),
    ],
    ids=["bad-shape", "wrong-count"],
)
def test_calculate_dipolar_tensors_for_atoms_rejects_malformed_muesr_output(
    monkeypatch, tensors, message
) -> None:
    _install_fake_muesr(monkeypatch, dipten=lambda *_args: tensors)

    with pytest.raises(MagneticFieldError, match=message):
        calculate_dipolar_tensors_for_atoms(
            _two_atom_structure(),
            [_SITE_A],
            magnetic_ion_indices=[0],
            options=MagneticFieldOptions(supercell=(1, 1, 1)),
        )


# calculate_incommensurate_field_distributions: options and validation


def test_incommensurate_distribution_options_use_fine_defaults() -> None:
    options = MagneticFieldOptions()

    assert options.nangles == 32_768
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
                supercell=(1, 1, 1),
                nangles=8,
                phase_sample_count=8,
                phase_sampling="grid",
                max_grid_samples=4,
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
        return [
            _Field(Dipolar=np.zeros((4, 3)), Lorentz=np.zeros((4, 3)), Contact=np.zeros((4, 3)))
        ]

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
def test_incommensurate_distribution_reads_each_muesr_field_layout(
    monkeypatch, values, expected
) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(**values))

    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1), nangles=4, phase_sampling="grid", histogram_bins=4
        ),
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
        return [
            _Field(Dipolar=values, Lorentz=np.zeros_like(values), Contact=np.zeros_like(values))
        ]

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
            supercell=(1, 1, 1),
            nangles=4,
            phase_sample_count=9,
            phase_sampling="grid",
            histogram_bins=4,
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
        locfield=_fields_per_muon(
            Dipolar=np.zeros((4, 3)), Lorentz=np.zeros((4, 3)), Contact=np.zeros((4, 3))
        ),
    )

    result = calculate_incommensurate_field_distributions(
        _structure(),
        model,
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1), nangles=4, phase_sampling="grid", histogram_bins=4
        ),
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
        return [
            _Field(Dipolar=values, Lorentz=np.zeros_like(values), Contact=np.zeros_like(values))
        ]

    _install_fake_muesr(monkeypatch, locfield=_locfield)

    result = calculate_incommensurate_field_distributions(
        _structure(),
        model,
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1), nangles=4, phase_sampling="grid", histogram_bins=4
        ),
    )[0]

    assert result.fields_T.shape == (4, 3)
    np.testing.assert_allclose(result.fields_T[:, 0], [100.0, 103.0, 102.0, 105.0], atol=1e-12)
    assert result.calculation_parameters["independent_q_indices"] == [0, 1]
    assert result.calculation_parameters["phase_sample_count"] == 4


def test_incommensurate_distribution_holds_commensurate_components_at_zero_phase(
    monkeypatch,
) -> None:
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
        options=MagneticFieldOptions(
            supercell=(1, 1, 1), nangles=4, phase_sampling="grid", histogram_bins=4
        ),
    )

    np.testing.assert_allclose(
        result.calculation_parameters["phase_basis_rlu"], [[0.12345, 0.0, 0.0]]
    )
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
def test_incommensurate_distribution_monte_carlo_sampling_is_seeded(
    monkeypatch, sampling: str
) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=_LOOKUP))
    options = MagneticFieldOptions(
        supercell=(1, 1, 1),
        nangles=4,
        phase_sample_count=7,
        phase_sampling=sampling,
        random_seed=123,
        histogram_bins=4,
    )

    first = calculate_incommensurate_field_distributions(
        _structure(), _two_q_helix_model(), [_SITE_A], options=options
    )
    second = calculate_incommensurate_field_distributions(
        _structure(), _two_q_helix_model(), [_SITE_A], options=options
    )

    assert first[0].fields_T.shape == (7, 3)
    np.testing.assert_array_equal(first[0].fields_T, second[0].fields_T)


# calculate_incommensurate_field_distributions: term breakdown and phase sampling


def _helix_term_locfield(_sample, mode, *_args, **kwargs):
    """A fake ``locfield`` whose dipolar term rotates in the ab plane and whose
    Lorentz and contact terms are fixed along c."""

    assert mode == "i"
    nangles = int(kwargs["nangles"])
    angles = 2.0 * np.pi * np.arange(nangles) / nangles
    dipolar = 0.2 * np.column_stack([np.cos(angles), np.sin(angles), np.zeros(nangles)])
    lorentz = np.tile([0.0, 0.0, 0.01], (nangles, 1))
    contact = np.tile([0.0, 0.0, -0.03], (nangles, 1))
    return [_Field(Dipolar=dipolar, Lorentz=lorentz, Contact=contact)]


def _helix_term_distribution(monkeypatch, **overrides) -> MagneticFieldDistributionResult:
    _install_fake_muesr(monkeypatch, locfield=_helix_term_locfield)
    options = {
        "supercell": (1, 1, 1),
        "nangles": 8,
        "phase_sampling": "grid",
        "demag_shape": "infinite_plate",
        "plate_normal_axis": "a",
        "measured_magnetization": _MEASURED_MOMENT,
        "applied_field": (0.0, 0.0, 0.05),
        "histogram_bins": 4,
    }
    options.update(overrides)
    return calculate_incommensurate_field_distributions(
        _structure(), _circular_helix_model(), [_SITE_A], options=MagneticFieldOptions(**options)
    )[0]


def test_incommensurate_distribution_keeps_the_mean_and_rms_of_each_term(monkeypatch) -> None:
    result = _helix_term_distribution(monkeypatch)

    assert tuple(result.terms) == ("dipolar", "lorentz", "contact", "demag", "applied")
    # The rotating dipolar term averages to zero while every sample has 0.2 T of it.
    assert result.terms["dipolar"].mean_T == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert result.terms["dipolar"].rms_T == pytest.approx(0.2)
    assert result.terms["lorentz"].mean_T == pytest.approx((0.0, 0.0, 0.01))
    assert result.terms["lorentz"].rms_T == pytest.approx(0.01)
    assert result.terms["contact"].mean_T == pytest.approx((0.0, 0.0, -0.03))
    demag = np.array([-_bulk_field(_MEASURED_MOMENT)[0], 0.0, 0.0])
    np.testing.assert_allclose(result.demag_T, demag, atol=1e-15)
    assert result.terms["demag"].rms_T == pytest.approx(float(np.linalg.norm(demag)))
    assert result.applied_T == pytest.approx((0.0, 0.0, 0.05))
    assert result.terms["applied"].rms_T == pytest.approx(0.05)
    summed = np.sum([term.mean_T for term in result.terms.values()], axis=0)
    np.testing.assert_allclose(summed, np.mean(result.fields_T, axis=0), atol=1e-15)
    assert result.total_rms_T == pytest.approx(
        float(np.sqrt(np.mean(np.sum(result.fields_T**2, axis=1))))
    )


def test_incommensurate_distribution_sums_dipolar_and_lorentz_sample_by_sample(
    monkeypatch,
) -> None:
    result = _helix_term_distribution(monkeypatch)

    # The rotating 0.2 T dipolar field and the constant 0.01 T Lorentz field are
    # perpendicular in every sample, so their sum has an RMS of hypot(0.2, 0.01).
    assert result.calculation_parameters["modulated"] is True
    assert result.dipolar_lorentz is not None
    assert result.dipolar_lorentz.mean_T == pytest.approx((0.0, 0.0, 0.01), abs=1e-12)
    assert result.dipolar_lorentz.rms_T == pytest.approx(float(np.hypot(0.2, 0.01)))


def test_incommensurate_distribution_without_lorentz_keeps_a_zero_lorentz_term(
    monkeypatch,
) -> None:
    result = _helix_term_distribution(monkeypatch, include_lorentz=False)

    assert result.terms["lorentz"].mean_T == (0.0, 0.0, 0.0)
    assert result.terms["lorentz"].rms_T == 0.0
    summed = np.sum([term.mean_T for term in result.terms.values()], axis=0)
    np.testing.assert_allclose(summed, np.mean(result.fields_T, axis=0), atol=1e-15)


def test_distribution_term_breakdown_round_trips_through_json(monkeypatch) -> None:
    result = _helix_term_distribution(monkeypatch)

    payload = json.loads(json.dumps(result.to_json()))
    assert payload["terms"]["dipolar"]["rms_T"] == pytest.approx(0.2)
    assert payload["applied_T"] == pytest.approx([0.0, 0.0, 0.05])
    restored = MagneticFieldDistributionResult.from_json(payload)

    np.testing.assert_array_equal(restored.fields_T, result.fields_T)
    np.testing.assert_array_equal(restored.histogram_counts, result.histogram_counts)
    assert restored.terms == result.terms
    assert restored.dipolar_lorentz == result.dipolar_lorentz
    assert restored.total_rms_T == result.total_rms_T
    assert restored.applied_T == result.applied_T
    assert restored.demag_T == result.demag_T
    assert restored.convergence == result.convergence
    assert restored.calculation_parameters == payload["calculation_parameters"]
    assert restored.to_json() == payload


def test_distribution_saved_without_a_breakdown_loads_with_none() -> None:
    payload = {
        "site_label": "A",
        "muon_position": [0.25, 0.25, 0.25],
        "fields_T": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        "histogram_centres_T": [1.0],
        "histogram_counts": [2.0],
        "mean_T": [0.0, 0.5, 0.5],
        "std_T": [0.0, 0.5, 0.5],
        "calculation_parameters": {"phase_sampling": "grid"},
        "convergence": [],
    }

    restored = MagneticFieldDistributionResult.from_json(payload)

    assert restored.terms == {}
    assert restored.total_rms_T is None
    assert restored.applied_T is None
    assert restored.demag_T is None
    assert restored.dipolar_lorentz is None
    saved = restored.to_json()
    assert "terms" not in saved
    assert "applied_T" not in saved
    assert "dipolar_lorentz" not in saved
    with pytest.raises(MagneticFieldError, match="recalculate"):
        regenerate_phase_samples(restored)


def test_distribution_from_json_rejects_fields_that_are_not_vectors() -> None:
    with pytest.raises(ValueError, match="fields_T must have shape"):
        MagneticFieldDistributionResult.from_json(
            {"site_label": "A", "fields_T": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}
        )


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda terms: terms["dipolar"].pop("rms_T"),
        lambda terms: terms.pop("contact"),
        lambda terms: terms.update(extra={"mean_T": [0.0, 0.0, 0.0], "rms_T": 0.0}),
    ],
    ids=["entry-missing-rms", "term-missing", "unknown-term"],
)
def test_distribution_from_json_drops_a_malformed_breakdown_whole(monkeypatch, corrupt) -> None:
    payload = json.loads(json.dumps(_helix_term_distribution(monkeypatch).to_json()))
    corrupt(payload["terms"])

    restored = MagneticFieldDistributionResult.from_json(payload)

    assert restored.terms == {}
    np.testing.assert_array_equal(restored.fields_T, np.asarray(payload["fields_T"]))


@pytest.mark.parametrize(
    "change",
    [
        {"phase_sampling_seed": None},
        {"phase_sample_count": None},
        {"component_phase_coefficients": [[1.0, 0.0, 5.0]]},
    ],
)
def test_regenerate_phase_samples_reports_malformed_parameters(
    monkeypatch, change: dict[str, object]
) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=_LOOKUP))
    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _two_q_helix_model(),
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1),
            nangles=4,
            phase_sample_count=7,
            phase_sampling="montecarlo",
            histogram_bins=4,
        ),
    )
    broken = replace(result, calculation_parameters={**result.calculation_parameters, **change})

    with pytest.raises(MagneticFieldError, match="phase sampling is malformed"):
        regenerate_phase_samples(broken)


def _capture_phase_samples(monkeypatch) -> list[np.ndarray]:
    captured: list[np.ndarray] = []
    build = module._build_phase_samples_turns

    def _build(**kwargs):
        phases = build(**kwargs)
        captured.append(phases)
        return phases

    monkeypatch.setattr(module, "_build_phase_samples_turns", _build)
    return captured


@pytest.mark.parametrize(
    ("sampling", "sample_count", "seed"),
    [("grid", 9, None), ("montecarlo", 7, 123), ("montecarlo", 7, None)],
)
def test_regenerate_phase_samples_reproduces_the_sampling_bit_for_bit(
    monkeypatch, sampling: str, sample_count: int, seed: int | None
) -> None:
    captured = _capture_phase_samples(monkeypatch)
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=_LOOKUP))
    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _two_q_helix_model(),
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1),
            nangles=4,
            phase_sample_count=sample_count,
            phase_sampling=sampling,
            random_seed=seed,
            histogram_bins=4,
        ),
    )
    saved = MagneticFieldDistributionResult.from_json(json.loads(json.dumps(result.to_json())))

    regenerated = regenerate_phase_samples(saved)

    np.testing.assert_array_equal(regenerated.phases_turns, captured[0])
    assert regenerated.phases_turns.shape == (sample_count, 2)
    np.testing.assert_allclose(regenerated.component_phase_coefficients, np.eye(2), atol=1e-12)
    assert result.calculation_parameters["random_seed"] == seed
    if sampling == "grid":
        assert result.calculation_parameters["phase_sampling_seed"] is None
    else:
        assert isinstance(result.calculation_parameters["phase_sampling_seed"], int)


def test_unseeded_monte_carlo_records_the_seed_that_reproduces_it(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=_LOOKUP))
    options = MagneticFieldOptions(
        supercell=(1, 1, 1),
        nangles=4,
        phase_sample_count=7,
        phase_sampling="montecarlo",
        histogram_bins=4,
    )

    (first,) = calculate_incommensurate_field_distributions(
        _structure(), _two_q_helix_model(), [_SITE_A], options=options
    )
    seed = first.calculation_parameters["phase_sampling_seed"]
    (second,) = calculate_incommensurate_field_distributions(
        _structure(),
        _two_q_helix_model(),
        [_SITE_A],
        options=replace(options, random_seed=seed),
    )

    assert 0 <= seed < 2**63
    np.testing.assert_array_equal(first.fields_T, second.fields_T)


def test_regenerate_phase_samples_of_a_single_phase_is_one_empty_row(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0.0, 0.0, 2.0)))
    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _model([0.0, 0.0, 0.5]),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, histogram_bins=4),
    )

    regenerated = regenerate_phase_samples(result)

    assert regenerated.phases_turns.shape == (1, 0)
    assert regenerated.component_phase_coefficients.shape == (1, 0)


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


def test_incommensurate_distribution_has_no_demag_field_without_a_uniform_moment(
    monkeypatch,
) -> None:
    def _locfield(_sample, mode, *_args, **kwargs):
        assert mode == "i"
        nangles = int(kwargs["nangles"])
        # A helix's moments, and so its Lorentz field, lie in the plane perpendicular to q.
        lorentz = np.column_stack(
            [np.zeros(nangles), 2.0 * np.arange(nangles), 3.0 * np.arange(nangles)]
        )
        return [
            _Field(Dipolar=np.zeros((nangles, 3)), Lorentz=lorentz, Contact=np.zeros((nangles, 3)))
        ]

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

    # A helix has no bulk magnetisation: the sampled fields are muesr's Lorentz lookup alone.
    lookup = np.column_stack([np.zeros(4), 2.0 * np.arange(4.0), 3.0 * np.arange(4.0)])
    np.testing.assert_allclose(result.fields_T, lookup, atol=1e-12)


def test_incommensurate_distribution_applies_advanced_demag_sample_to_the_uniform_moment(
    monkeypatch,
) -> None:
    def _locfield(sample, mode, *_args, **kwargs):
        assert mode in {"i", "s"}
        return [
            _Field(Dipolar=np.zeros(3), Lorentz=np.zeros(3), Contact=np.zeros(3))
            for _ in sample.muons
        ]

    _install_fake_muesr(monkeypatch, locfield=_locfield)
    model = MagneticStructureModel(
        q_basis_rlu=((0.12345, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            _helix("helix", (1.0, 0.0, 0.0)),
            MagneticComponent(
                component_id="uniform",
                label="Uniform",
                kind="custom",
                selector=AtomSelector.all(),
                terms=(FourierTerm((0.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 2.0])),),
            ),
        ),
    )

    result = calculate_incommensurate_field_distributions(
        _structure(),
        model,
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1),
            nangles=3,
            phase_sampling="grid",
            demag_sample=DemagnetizingSample(shape="infinite_plate", plate_normal_axis="c"),
            histogram_bins=4,
        ),
    )[0]

    # N = 1 along the plate normal: every sampled field is -mu0 M of the 2 mu_B uniform moment.
    expected = -_bulk_field((0.0, 0.0, 2.0))
    np.testing.assert_allclose(
        result.fields_T, np.full((len(result.fields_T), 3), expected), atol=1e-12
    )
    assert result.calculation_parameters["demag_sample"]["shape"] == "infinite_plate"


def _zero_lookup_locfield(_sample, mode, *_args, **kwargs):
    """A fake incommensurate ``locfield`` whose lookup tables are all zero."""

    assert mode == "i"
    zeros = np.zeros((int(kwargs["nangles"]), 3))
    return [_Field(Dipolar=zeros, Lorentz=zeros, Contact=zeros)]


def test_incommensurate_distribution_applies_a_measured_magnetisation(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_zero_lookup_locfield)

    result = calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1),
            nangles=4,
            phase_sampling="grid",
            demag_shape="infinite_plate",
            plate_normal_axis="a",
            measured_magnetization=_MEASURED_MOMENT,
            histogram_bins=4,
        ),
    )[0]

    # The helix has no uniform moment, so the whole demag field comes from the measured value.
    expected = np.array([-_bulk_field(_MEASURED_MOMENT)[0], 0.0, 0.0])
    np.testing.assert_allclose(
        result.fields_T, np.full((len(result.fields_T), 3), expected), atol=1e-12
    )
    saved = json.loads(json.dumps(result.to_json()))
    assert saved["calculation_parameters"]["measured_magnetization"] == list(_MEASURED_MOMENT)


def test_distribution_parameters_record_a_model_derived_magnetisation_as_none(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_zero_lookup_locfield)

    result = calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1), nangles=4, phase_sampling="grid", histogram_bins=4
        ),
    )[0]

    assert result.calculation_parameters["measured_magnetization"] is None


@pytest.mark.parametrize(
    ("measured", "message"),
    [
        (1.0, "three-vector"),
        ((1.0, 2.0), "three-vector"),
        ((0.0, float("nan"), 0.0), "finite"),
        (("x", 0.0, 0.0), "three numbers"),
    ],
)
def test_incommensurate_distribution_validates_a_measured_magnetisation(
    monkeypatch, measured, message
) -> None:
    _install_fake_muesr(monkeypatch, locfield=_zero_lookup_locfield)

    with pytest.raises(MagneticFieldError, match=message):
        calculate_incommensurate_field_distributions(
            _structure(),
            _circular_helix_model(),
            [_SITE_A],
            options=MagneticFieldOptions(
                supercell=(1, 1, 1),
                nangles=4,
                phase_sampling="grid",
                histogram_bins=4,
                measured_magnetization=measured,
            ),
        )


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
            n_dim=1,
            nangles=0,
            phase_sample_count=5,
            sampling="grid",
            random_seed=0,
            max_grid_samples=100,
        )
    with pytest.raises(MagneticFieldError, match="phase_sample_count must be positive"):
        module._build_phase_samples_turns(
            n_dim=1,
            nangles=4,
            phase_sample_count=0,
            sampling="grid",
            random_seed=0,
            max_grid_samples=100,
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
    with pytest.raises(MagneticFieldError, match="lookup stride must divide"):
        module._combine_component_samples(
            [np.zeros((5, 3))], [(1.0,)], np.zeros((2, 1)), nangles=5, stride=2
        )


# Commensurate approximants


def _phase_recording_locfield(sample, mode, *_args, **_kwargs):
    """A fake commensurate ``locfield`` reporting the phase and wavevector it was given."""

    assert mode == "s"
    record = (float(sample.mm.phi[0]), float(sample.mm.k[0]), float(sample.mm.k[1]))
    return [
        _Field(Dipolar=record, Lorentz=(0.0, 0.0, 1.0), Contact=(0.0, 0.0, 0.0))
        for _ in sample.muons
    ]


def test_find_commensurate_approximant_rounds_each_axis_to_a_short_period() -> None:
    approximant = find_commensurate_approximant(
        _structure(), _circular_helix_model((0.3334, 0.2499, 0.0))
    )

    assert approximant is not None
    np.testing.assert_allclose(approximant.approximant_rlu, [(1.0 / 3.0, 0.25, 0.0)])
    assert approximant.denominators == ((3, 4, 1),)
    assert approximant.periods == (3, 4, 1)
    assert approximant.max_deviation_rlu == pytest.approx(1e-4, abs=1e-6)
    assert approximant.is_exact is False
    assert approximant.describe() == "q = (1/3, 1/4, 0) with a 3 x 4 x 1 magnetic supercell"
    assert approximant.to_json()["periods"] == [3, 4, 1]


def test_find_commensurate_approximant_combines_periods_across_wavevectors() -> None:
    model = MagneticStructureModel(
        q_basis_rlu=((0.5, 0.0, 0.0), (0.0, 0.3333, 0.0), (0.0, 0.0, 0.0)),
        components=(_helix("q1", (1.0, 0.0, 0.0)), _helix("q2", (0.0, 1.0, 0.0))),
    )

    approximant = find_commensurate_approximant(_structure(), model)

    assert approximant is not None
    assert approximant.periods == (2, 3, 1)


def test_find_commensurate_approximant_reports_a_genuinely_incommensurate_q() -> None:
    # 0.12345 is 1.5e-3 from 1/8, outside the default 1e-3 tolerance.
    assert find_commensurate_approximant(_structure(), _circular_helix_model()) is None
    assert (
        find_commensurate_approximant(_structure(), _circular_helix_model(), tolerance=2e-3)
        is not None
    )
    assert (
        find_commensurate_approximant(
            _structure(), _circular_helix_model(), tolerance=2e-3, max_period=7
        )
        is None
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tolerance": -1.0}, "tolerance must not be negative"),
        ({"max_period": 0}, "maximum period must be positive"),
    ],
)
def test_find_commensurate_approximant_validates_its_limits(kwargs, message) -> None:
    with pytest.raises(MagneticFieldError, match=message):
        find_commensurate_approximant(_structure(), _circular_helix_model(), **kwargs)


def test_find_commensurate_approximant_rejects_an_empty_model() -> None:
    with pytest.raises(MagneticFieldError, match="no non-zero Fourier components"):
        find_commensurate_approximant(_structure(), _empty_model())


def test_approximant_fields_list_each_translation_with_advanced_phases(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_phase_recording_locfield)

    results = calculate_commensurate_approximant_fields(
        _structure(),
        _circular_helix_model((0.3334, 0.0, 0.0)),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1)),
    )

    assert [result.site_label for result in results] == [
        "A + (0, 0, 0)",
        "A + (1, 0, 0)",
        "A + (2, 0, 0)",
    ]
    np.testing.assert_allclose(
        [result.muon_position for result in results],
        [(0.25, 0.25, 0.25), (1.25, 0.25, 0.25), (2.25, 0.25, 0.25)],
    )
    # The backend receives the rational q = 1/3 and a phase advanced by q . T (in turns).
    np.testing.assert_allclose(
        [result.dipolar[0] for result in results], [0.0, 1.0 / 3.0, 2.0 / 3.0]
    )
    np.testing.assert_allclose([result.dipolar[1] for result in results], [1.0 / 3.0] * 3)
    assert all(result.lorentz == (0.0, 0.0, 1.0) for result in results)
    assert all(result.distortion is None for result in results)


def test_approximant_fields_keep_one_translation_per_distinct_phase(monkeypatch) -> None:
    # q = (1/2, 1/2, 0) has a 2 x 2 x 1 supercell but only two distinct phases, 0 and 1/2.
    _install_fake_muesr(monkeypatch, locfield=_phase_recording_locfield)

    results = calculate_commensurate_approximant_fields(
        _structure(),
        _circular_helix_model((0.5, 0.4999, 0.0)),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), find_equivalent_sites=False),
    )

    assert [result.site_label for result in results] == ["A + (0, 0, 0)", "A + (1, 0, 0)"]
    np.testing.assert_allclose([result.dipolar[0] for result in results], [0.0, 0.5])


def test_distinct_phase_translations_walk_the_phase_group_not_the_supercell() -> None:
    # q = (1/12, 1/12, 1/12) repeats over 12 x 12 x 12 cells but reaches only 12 phases.
    twelfths = (np.array([1, 1, 1]), np.array([12, 12, 12]))

    translations = module._distinct_phase_translations([twelfths], limit=512)

    assert translations is not None
    assert len(translations) == 12
    assert translations == tuple((n, 0, 0) for n in range(12))


def test_distinct_phase_translations_stop_at_the_limit() -> None:
    # Three independent wavevectors with coprime periods reach 11 x 12 x 13 phase sets.
    rationals = [
        (np.array([1, 0, 0]), np.array([11, 1, 1])),
        (np.array([0, 1, 0]), np.array([1, 12, 1])),
        (np.array([0, 0, 1]), np.array([1, 1, 13])),
    ]

    assert module._distinct_phase_translations(rationals, limit=512) is None
    assert len(module._distinct_phase_translations(rationals, limit=11 * 12 * 13)) == 11 * 12 * 13


def test_approximant_fields_of_a_single_phase_keep_the_site_label(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_phase_recording_locfield)

    (result,) = calculate_commensurate_approximant_fields(
        _structure(),
        _circular_helix_model((0.0002, 0.0, 0.0)),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1)),
    )

    assert result.site_label == "A"
    assert result.dipolar[1] == 0.0


def test_approximant_fields_need_an_approximant(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_phase_recording_locfield)

    with pytest.raises(
        MagneticFieldError, match="no commensurate approximant within 0.001 r.l.u. and period 12"
    ):
        calculate_commensurate_approximant_fields(_structure(), _circular_helix_model(), [_SITE_A])


def test_approximant_fields_cap_the_number_of_translations(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_phase_recording_locfield)

    with pytest.raises(MagneticFieldError, match="more than 2 inequivalent translations"):
        calculate_commensurate_approximant_fields(
            _structure(),
            _circular_helix_model((0.3334, 0.0, 0.0)),
            [_SITE_A],
            options=MagneticFieldOptions(max_approximant_fields=2),
        )


def test_approximant_correction_follows_each_translated_site(monkeypatch) -> None:
    # At r + T each ion carries the moment of its pristine site one translation along.
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0, 0, 0)))
    model = _circular_helix_model((0.25, 0.0, 0.0))
    relaxed = relaxed_muon_supercell(
        _structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (0.1, 0.05, 0.0)}
    )

    results = calculate_commensurate_approximant_fields(
        _structure(),
        model,
        [_MU_X],
        options=MagneticFieldOptions(supercell=(1, 1, 1)),
        distorted_structures=[relaxed],
    )

    distortion = map_muon_distortion(_structure(), relaxed, _MU_X)
    assert len(results) == 4
    for result in results:
        translation = np.round(np.asarray(result.muon_position) - _MU_X)
        moments = moments_at_atoms(
            _structure(), model, distortion.atom_indices, distortion.cell_offsets + translation
        )
        expected = np.asarray(
            calculate_finite_dipolar_field(distortion.distorted_vectors, moments, np.zeros(3))
        ) - np.asarray(
            calculate_finite_dipolar_field(distortion.pristine_vectors, moments, np.zeros(3))
        )
        np.testing.assert_allclose(result.distortion, expected, rtol=1e-9, atol=1e-15)
    assert len({tuple(np.round(result.distortion, 9)) for result in results}) == 4


def test_approximant_fields_reject_an_empty_model(monkeypatch) -> None:
    _forbid_muesr(monkeypatch)

    with pytest.raises(MagneticFieldError, match="no non-zero Fourier components"):
        calculate_commensurate_approximant_fields(_structure(), _empty_model(), [_SITE_A])


# Distribution convergence checks


def _ramp_locfield(sample, mode, *_args, **kwargs):
    """A fake incommensurate lookup whose |B| rises smoothly with the phase angle."""

    assert mode == "i"
    nangles = int(kwargs["nangles"])
    values = np.zeros((nangles, 3), dtype=float)
    values[:, 0] = 1.0 + np.arange(nangles, dtype=float) / nangles
    return [_Field(Dipolar=values) for _ in sample.muons]


def _alternating_locfield(sample, mode, *_args, **kwargs):
    """A fake lookup whose odd angles differ from its even ones, so half the angles miss half of p(|B|)."""

    assert mode == "i"
    nangles = int(kwargs["nangles"])
    values = np.zeros((nangles, 3), dtype=float)
    values[:, 0] = 1.0 + (np.arange(nangles) % 2)
    return [_Field(Dipolar=values) for _ in sample.muons]


def test_distribution_convergence_checks_pass_for_a_resolved_distribution(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_ramp_locfield)

    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=1024, histogram_bins=16),
    )

    assert [check.setting for check in result.convergence] == [
        "nangles",
        "phase_sample_count",
        "histogram_bins",
    ]
    assert all(check.converged for check in result.convergence)
    assert result.convergence[0].message.startswith("Halving the lookup angles to 512 shifts")
    assert result.convergence[2].metric == pytest.approx(64.0)
    assert result.calculation_parameters["resolved_phase_sampling"] == "grid"
    payload = result.to_json()["convergence"]
    assert payload[0]["setting"] == "nangles"
    assert payload[0]["converged"] is True


def test_distribution_convergence_skips_the_lookup_check_for_an_odd_lookup_count(
    monkeypatch,
) -> None:
    _install_fake_muesr(monkeypatch, locfield=_ramp_locfield)

    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=1023, histogram_bins=16),
    )

    assert [check.setting for check in result.convergence] == [
        "phase_sample_count",
        "histogram_bins",
    ]


def test_distribution_convergence_flags_a_coarse_lookup(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_alternating_locfield)

    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=1024, histogram_bins=16),
    )

    lookup = result.convergence[0]
    assert lookup.converged is False
    assert lookup.metric == pytest.approx(0.5)
    assert lookup.message.endswith("(more than 1%); double Lookup angles per q.")


@pytest.mark.parametrize(
    ("sampling", "advice"),
    [("grid", "more lookup angles per q"), ("montecarlo", "more distribution samples")],
)
def test_distribution_convergence_flags_sparse_histogram_bins(
    monkeypatch, sampling, advice
) -> None:
    _install_fake_muesr(monkeypatch, locfield=_ramp_locfield)

    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _circular_helix_model(),
        [_SITE_A],
        options=MagneticFieldOptions(
            supercell=(1, 1, 1),
            nangles=64,
            phase_sample_count=64,
            phase_sampling=sampling,
            random_seed=3,
            histogram_bins=64,
        ),
    )

    bins = result.convergence[-1]
    assert bins.setting == "histogram_bins"
    assert bins.converged is False
    assert bins.message.endswith(f"use fewer histogram bins or {advice}.")
    samples = result.convergence[1]
    assert samples.converged is False
    assert samples.message.endswith(f"use {advice}.")
    assert result.calculation_parameters["resolved_phase_sampling"] == sampling


def test_distribution_convergence_of_a_single_phase_checks_only_the_histogram(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0.0, 0.0, 2.0)))

    (result,) = calculate_incommensurate_field_distributions(
        _structure(),
        _model([0.0, 0.0, 0.5]),
        [_SITE_A],
        options=MagneticFieldOptions(supercell=(1, 1, 1), nangles=4, histogram_bins=4),
    )

    assert [check.setting for check in result.convergence] == ["histogram_bins"]
    assert result.convergence[0].converged is True


def test_half_resolution_grid_keeps_every_other_point_per_axis() -> None:
    axis = np.arange(4, dtype=float) / 4.0
    grid = np.stack([mesh.ravel() for mesh in np.meshgrid(axis, axis, indexing="ij")], axis=1)

    half = module._half_resolution_phase_samples(grid, "grid")

    np.testing.assert_allclose(half, [(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)])
    assert module._half_resolution_phase_samples(grid[:15], "grid") is None
    assert module._half_resolution_phase_samples(np.zeros((1, 0)), "grid") is None


def test_cdf_distance_measures_the_largest_cumulative_difference() -> None:
    assert module._cdf_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
    assert module._cdf_distance(np.array([1.0, 2.0]), np.array([1.0, 1.0])) == pytest.approx(0.5)


# Real muesr lattice sums against independent references
#
# These tests run the installed muesr/muLFC engines. The reference for each is
# either a closed-form result or a direct sum over every ion inside the
# Lorentz sphere, with the moments taken from ``moments_at_atoms``, so they pin
# both muesr's numbers and MuFinder's hand-off of the magnetic model to it.

_RADIUS = 12.0
_GENERAL_SITE = (0.1, 0.2, 0.3)
_bcc_structure = partial(
    host_structure, label="Fe2", lattice=3.0, species=("Fe", "Fe"), positions=BODY_CENTRE
)


def _single_q_model(q, real, imaginary=(0.0, 0.0, 0.0)) -> MagneticStructureModel:
    """One component on every atom with Fourier coefficient ``real + i imaginary`` at ``q``."""

    return MagneticStructureModel(
        q_basis_rlu=(tuple(float(value) for value in q), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="moment",
                label="Moment",
                kind="custom",
                selector=AtomSelector.all(),
                terms=(
                    FourierTerm(
                        (1.0, 0.0, 0.0), ComplexVector(real=tuple(real), imaginary=tuple(imaginary))
                    ),
                ),
            ),
        ),
    )


def _direct_lattice_field(structure, model, muon_frac, radius, *, displacements=None) -> np.ndarray:
    """Dipolar field in T at ``muon_frac`` summed directly over every ion within ``radius`` Å.

    Moments follow MuFinder's convention, ``moments_at_atoms`` with cell offsets
    measured from the origin cell. ``displacements`` maps ``(atom index, cell
    offset)`` to a Cartesian shift in Å; an ion's pristine site decides whether
    it lies inside the sphere.
    """

    lattice = np.asarray(structure.lattice_vectors, dtype=float)
    fractional = np.asarray(structure.fractional_positions, dtype=float)
    muon = np.asarray(muon_frac, dtype=float)
    reach = int(np.ceil(radius / np.min(np.linalg.norm(lattice, axis=1)))) + 1
    steps = range(-reach, reach + 1)
    offsets = np.floor(muon) + np.array(
        [(i, j, k) for i in steps for j in steps for k in steps], dtype=float
    )
    indices = np.repeat(np.arange(len(fractional)), len(offsets))
    cells = np.tile(offsets, (len(fractional), 1))
    positions = (fractional[indices] + cells) @ lattice
    muon_cart = muon @ lattice
    inside = np.linalg.norm(positions - muon_cart, axis=1) < radius
    for (atom, offset), shift in (displacements or {}).items():
        match = (indices == atom) & np.all(cells == np.asarray(offset, dtype=float), axis=1)
        positions[match] += np.asarray(shift, dtype=float)
    moments = moments_at_atoms(structure, model, indices[inside], cells[inside])
    return np.asarray(calculate_finite_dipolar_field(positions[inside], moments, muon_cart))


def test_real_muesr_ferromagnet_body_centre_sees_only_the_lorentz_field(real_muesr) -> None:
    # Cubic symmetry cancels the dipolar sum at the body centre, leaving the
    # Lorentz field mu0 M / 3 of 1 mu_B per (3 Å)^3, 0.1439 T.
    (result,) = calculate_commensurate_fields(_structure(), _model(), [(0.5, 0.5, 0.5)])

    lorentz = MU0_T_M_PER_A * BOHR_MAGNETON_J_PER_T / (3.0e-10) ** 3 / 3.0
    assert result.dipolar == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert result.lorentz == pytest.approx((0.0, 0.0, lorentz), rel=5e-3)
    assert lorentz == pytest.approx(0.1439, abs=1e-4)
    assert result.contact == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("demag_shape", "factor"), [("sphere", 1 / 3), ("infinite_plate", 1.0)], ids=["sphere", "plate"]
)
def test_real_muesr_ferromagnet_demagnetizing_field_follows_the_muesr_convention(
    real_muesr, demag_shape, factor
) -> None:
    # muesr's convention: B_Lor = mu0 M_Lor / 3 from the moments inside the
    # Lorentz sphere, B_dem = -mu0 N M_meas from the bulk magnetisation, here
    # mu0 M_meas = 0.4316 T for 1 mu_B per (3 Å)^3. For a sphere the two cancel
    # up to muesr's finite-sphere count of the ions, 0.1437 T against 0.1439 T.
    options = MagneticFieldOptions(demag_shape=demag_shape, plate_normal_axis="c")
    (result,) = calculate_commensurate_fields(
        _structure(), _model(), [(0.5, 0.5, 0.5)], options=options
    )

    bulk = MU0_T_M_PER_A * BOHR_MAGNETON_J_PER_T / (3.0e-10) ** 3
    assert bulk == pytest.approx(0.4316, abs=1e-4)
    assert result.demag == pytest.approx((0.0, 0.0, -factor * bulk), rel=1e-12)
    assert result.lorentz[2] + result.demag[2] == pytest.approx((1 / 3 - factor) * bulk, abs=5e-4)


def test_real_muesr_antiferromagnet_has_no_demagnetizing_field(real_muesr) -> None:
    # muesr's finite Lorentz sphere leaves a small Lorentz field at a general
    # site, but the A-type order has no bulk magnetisation to demagnetize.
    model = _single_q_model((0.0, 0.0, 0.5), (0.0, 0.0, 1.0))
    options = MagneticFieldOptions(demag_shape="sphere")
    (result,) = calculate_commensurate_fields(_structure(), model, [_GENERAL_SITE], options=options)

    assert np.linalg.norm(result.lorentz) > 0.0
    assert result.demag == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("structure", "model"),
    [
        (_structure, partial(_single_q_model, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        (_structure, partial(_single_q_model, (0.0, 0.0, 0.5), (0.0, 0.0, 1.0))),
        (_structure, partial(_single_q_model, (0.5, 0.0, 0.5), (1.0, 0.0, 0.0))),
        (_structure, partial(_single_q_model, (0.0, 0.0, 0.25), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        (
            _structure,
            partial(_single_q_model, (1 / 3, 0.2, 0.25), (0.0, 0.0, 1.0), (0.0, 0.5, 0.0)),
        ),
        (_bcc_structure, partial(_single_q_model, (0.0, 0.0, 1 / 3), (1.0, 0.0, 0.0))),
    ],
    ids=["ferromagnet", "a-type", "c-type", "quarter-helix", "elliptical-three-q", "bcc-third"],
)
@pytest.mark.parametrize("supercell", [(15, 15, 15), (14, 16, 17)], ids=["odd", "anisotropic"])
def test_real_muesr_commensurate_dipolar_field_matches_a_direct_lattice_sum(
    real_muesr, structure, model, supercell
) -> None:
    (result,) = calculate_commensurate_fields(
        structure(),
        model(),
        [_GENERAL_SITE],
        options=MagneticFieldOptions(supercell=supercell, radius=_RADIUS),
    )

    expected = _direct_lattice_field(structure(), model(), _GENERAL_SITE, _RADIUS)
    np.testing.assert_allclose(result.dipolar, expected, rtol=1e-6, atol=1e-9)


def test_real_muesr_commensurate_field_does_not_depend_on_the_supercell_size(real_muesr) -> None:
    # muesr puts the muon in cell floor(N / 2); with q = 1/3 every N in 14..16
    # lands that cell on a different magnetic phase unless MuFinder compensates.
    model = _single_q_model((0.0, 0.0, 1 / 3), (0.0, 0.0, 1.0))

    dipolar = [
        calculate_commensurate_fields(
            _structure(),
            model,
            [_GENERAL_SITE],
            options=MagneticFieldOptions(supercell=(n, n, n), radius=_RADIUS),
        )[0].dipolar
        for n in (14, 15, 16)
    ]

    np.testing.assert_allclose(dipolar[1], dipolar[0], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(dipolar[2], dipolar[0], rtol=1e-6, atol=1e-9)


def test_real_muesr_contact_field_converts_the_coupling_unit(real_muesr) -> None:
    # A uniform 1 mu_B ferromagnet gives a contact field equal to the coupling in T per mu_B.
    options = MagneticFieldOptions(
        supercell=(15, 15, 15),
        radius=_RADIUS,
        contact_coupling=contact_coupling_to_angstrom_minus3(0.5, "T_per_muB"),
    )

    results = calculate_commensurate_fields(
        _structure(), _model(), [(0.5, 0.5, 0.5), _GENERAL_SITE], options=options
    )

    for result in results:
        assert result.contact == pytest.approx((0.0, 0.0, 0.5), rel=1e-6, abs=1e-9)


def _direct_contact_field(structure, model, muon_frac, coupling, nnn, rcont) -> np.ndarray:
    """muesr's contact field from a direct 1/r^3-weighted mean over the ``nnn`` nearest moments."""

    lattice = np.asarray(structure.lattice_vectors, dtype=float)
    fractional = np.asarray(structure.fractional_positions, dtype=float)
    steps = range(-3, 4)
    offsets = np.array([(i, j, k) for i in steps for j in steps for k in steps], dtype=float)
    indices = np.repeat(np.arange(len(fractional)), len(offsets))
    cells = np.tile(offsets, (len(fractional), 1))
    distances = np.linalg.norm(
        (fractional[indices] + cells) @ lattice - np.asarray(muon_frac) @ lattice, axis=1
    )
    order = np.argsort(distances, kind="stable")
    shell = order[distances[order] < rcont][:nnn]
    moments = moments_at_atoms(structure, model, indices[shell], cells[shell])
    weights = distances[shell] ** -3
    mean = np.sum(weights[:, np.newaxis] * moments, axis=0) / np.sum(weights)
    return coupling * CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB * mean


@pytest.mark.parametrize(("nnn", "rcont"), [(1, 10.0), (2, 10.0), (4, 10.0), (4, 2.5)])
def test_real_muesr_contact_field_follows_the_contact_shell(real_muesr, nnn, rcont) -> None:
    model = _single_q_model((0.5, 0.0, 0.25), (0.0, 0.0, 1.0), (0.3, 0.0, 0.0))
    options = MagneticFieldOptions(
        supercell=(15, 15, 15), radius=_RADIUS, contact_coupling=0.1, nnn=nnn, rcont=rcont
    )

    (result,) = calculate_commensurate_fields(_structure(), model, [_GENERAL_SITE], options=options)

    expected = _direct_contact_field(_structure(), model, _GENERAL_SITE, 0.1, nnn, rcont)
    assert np.linalg.norm(expected) > 1e-3
    np.testing.assert_allclose(result.contact, expected, rtol=1e-6, atol=1e-9)
    assert result.calculation_parameters["nnn"] == nnn
    assert result.calculation_parameters["rcont"] == rcont


@pytest.mark.parametrize(("nnn", "rcont"), [(1, 10.0), (6, 10.0), (3, 3.5)])
def test_real_muesr_ferromagnet_contact_field_does_not_depend_on_the_shell(
    real_muesr, nnn, rcont
) -> None:
    options = MagneticFieldOptions(
        supercell=(15, 15, 15),
        radius=_RADIUS,
        contact_coupling=contact_coupling_to_angstrom_minus3(0.5, "T_per_muB"),
        nnn=nnn,
        rcont=rcont,
    )

    (result,) = calculate_commensurate_fields(
        _structure(), _model(), [_GENERAL_SITE], options=options
    )

    assert result.contact == pytest.approx((0.0, 0.0, 0.5), rel=1e-6, abs=1e-9)


def test_real_muesr_distortion_corrected_field_matches_the_distorted_lattice(real_muesr) -> None:
    # The Fe one cell along +x moves 0.1 Å away from the muon in a q = 1/4 helix,
    # where the lattice sum and the correction must agree on that Fe's moment.
    model = _single_q_model((0.25, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    relaxed = relaxed_muon_supercell(
        _structure(), (3, 3, 3), _MU_X, {(1.0, 0.0, 0.0): (0.1, 0.0, 0.0)}
    )

    (result,) = calculate_commensurate_fields(
        _structure(),
        model,
        [_MU_X],
        options=MagneticFieldOptions(supercell=(15, 15, 15), radius=_RADIUS),
        distorted_structures=[relaxed],
    )

    distorted = _direct_lattice_field(
        _structure(), model, _MU_X, _RADIUS, displacements={(0, (1.0, 0.0, 0.0)): (0.1, 0.0, 0.0)}
    )
    assert np.linalg.norm(result.distortion) > 1e-3
    np.testing.assert_allclose(
        np.asarray(result.dipolar) + np.asarray(result.distortion), distorted, rtol=1e-6, atol=1e-9
    )


def test_real_muesr_distribution_correction_matches_the_approximant_per_translation(
    real_muesr,
) -> None:
    # q = 1/4 counts as incommensurate below period 4, so the distribution path
    # samples the helix with muLFC's lookup rows while the approximant path sums
    # it exactly at each of four translations. 40 lookup rows put a phase sample
    # on every translation's phase, whatever muLFC's row offset. Matching the
    # samples by their uncorrected field pins the row offset of the correction;
    # test_approximant_correction_follows_each_translated_site pins its sign.
    model = _single_q_model((0.25, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    relaxed = relaxed_muon_supercell(
        _structure(), (3, 3, 3), _GENERAL_SITE, {(1.0, 0.0, 0.0): (0.1, 0.05, -0.05)}
    )
    common = {
        "supercell": (15, 15, 15),
        "radius": _RADIUS,
        "contact_coupling": 0.1,
        "nnn": 2,
        "rcont": 3.0,
        "include_lorentz": False,
    }
    distribution_options = MagneticFieldOptions(
        **common, nangles=40, phase_sampling="grid", commensurate_max_period=2
    )

    def _distribution(distorted):
        return calculate_incommensurate_field_distributions(
            _structure(),
            model,
            [_GENERAL_SITE],
            options=distribution_options,
            distorted_structures=distorted,
        )[0].fields_T

    plain = _distribution(None)
    corrected = _distribution([relaxed])
    translated = calculate_commensurate_approximant_fields(
        _structure(),
        model,
        [_GENERAL_SITE],
        options=MagneticFieldOptions(**common),
        distorted_structures=[relaxed],
    )

    assert len(translated) == 4
    for result in translated:
        uncorrected = np.asarray(result.dipolar) + np.asarray(result.contact)
        sample = int(np.argmin(np.linalg.norm(plain - uncorrected, axis=1)))
        np.testing.assert_allclose(plain[sample], uncorrected, rtol=1e-6, atol=1e-9)
        assert np.linalg.norm(result.distortion) > 1e-3
        np.testing.assert_allclose(
            corrected[sample] - plain[sample], result.distortion, rtol=1e-6, atol=1e-9
        )


def test_real_muesr_contact_shell_matches_the_contact_field(real_muesr) -> None:
    # The pristine half of the contact correction must be muLFC's own contact
    # average, 1/r^3-weighted over the nnn nearest moments, or the correction
    # would not difference cleanly against muesr's contact field.
    model = _single_q_model((0.5, 0.0, 0.25), (0.0, 0.0, 1.0), (0.3, 0.0, 0.0))
    options = MagneticFieldOptions(
        supercell=(15, 15, 15), radius=_RADIUS, contact_coupling=0.1, nnn=3, rcont=4.0
    )
    (result,) = calculate_commensurate_fields(_structure(), model, [_GENERAL_SITE], options=options)
    distortion = map_muon_distortion(
        _structure(), relaxed_muon_supercell(_structure(), (4, 4, 4), _GENERAL_SITE), _GENERAL_SITE
    )
    (model_data,) = module.get_muesr_model_data(_structure(), model)

    _dipolar, change = module._component_distortion_changes(model_data, distortion, options)
    coefficients = (
        model_data.fc[distortion.atom_indices]
        * np.exp(-2j * np.pi * (distortion.cell_offsets @ model_data.k_rlu))[:, np.newaxis]
    )
    shell = module.contact_shell_mean(
        distortion.pristine_vectors,
        coefficients,
        np.ones(len(coefficients), dtype=bool),
        nnn=3,
        rcont=4.0,
        reach_limit=6.0,
    )

    assert change == pytest.approx(np.zeros(3), abs=1e-15)
    expected = 0.1 * module.CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB * shell.real
    assert np.linalg.norm(expected) > 1e-2
    np.testing.assert_allclose(result.contact, expected, rtol=1e-6, atol=1e-9)


def test_real_muesr_approximant_lists_the_field_at_each_translated_site(real_muesr) -> None:
    model = _single_q_model((0.0, 0.0, 0.25), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    results = calculate_commensurate_approximant_fields(
        _structure(),
        model,
        [_GENERAL_SITE],
        options=MagneticFieldOptions(
            supercell=(15, 15, 15), radius=_RADIUS, contact_coupling=0.1, nnn=3, rcont=4.0
        ),
    )

    assert [result.site_label for result in results] == [
        f"site 1 + (0, 0, {step})" for step in range(4)
    ]
    for result in results:
        expected = _direct_lattice_field(_structure(), model, result.muon_position, _RADIUS)
        np.testing.assert_allclose(result.dipolar, expected, rtol=1e-6, atol=1e-9)
        assert result.calculation_parameters == {
            "contact_coupling": 0.1,
            "nnn": 3,
            "rcont": 4.0,
            "modulated": True,
        }


@pytest.mark.parametrize(
    ("real", "imaginary"),
    [
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, 1.0), (0.0, 0.5, 0.0)),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
    ],
    ids=["circular-helix", "elliptical", "spin-density-wave"],
)
def test_real_muesr_incommensurate_fields_trace_the_direct_sum_ellipse(
    real_muesr, real, imaginary
) -> None:
    # A single-q structure gives B(theta) = cos(theta) B0 + sin(theta) B90, where
    # B0 is the field of the model and B90 that of the model a quarter turn on.
    q = (0.1234, 0.0, 0.0)
    coefficient = np.asarray(real) + 1j * np.asarray(imaginary)
    quarter = -1j * coefficient
    options = MagneticFieldOptions(
        supercell=(15, 15, 15), radius=_RADIUS, include_lorentz=False, nangles=360
    )

    (result,) = calculate_incommensurate_field_distributions(
        _structure(), _single_q_model(q, real, imaginary), [_GENERAL_SITE], options=options
    )

    basis = np.stack(
        [
            _direct_lattice_field(
                _structure(), _single_q_model(q, real, imaginary), _GENERAL_SITE, _RADIUS
            ),
            _direct_lattice_field(
                _structure(), _single_q_model(q, quarter.real, quarter.imag), _GENERAL_SITE, _RADIUS
            ),
        ],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(basis, result.fields_T.T, rcond=None)
    np.testing.assert_allclose(basis @ coefficients, result.fields_T.T, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(coefficients, axis=0), 1.0, rtol=0.0, atol=1e-6)


def test_real_muesr_dipolar_tensor_maps_a_moment_onto_its_lattice_field(real_muesr) -> None:
    options = MagneticFieldOptions(supercell=(15, 15, 15), radius=_RADIUS)

    (result,) = calculate_dipolar_tensors_for_atoms(
        _structure(), [_GENERAL_SITE], magnetic_ion_indices=[0], options=options
    )

    tensor = np.asarray(result.tensor_T_per_muB)
    np.testing.assert_allclose(tensor, tensor.T, rtol=0.0, atol=1e-12)
    assert np.trace(tensor) == pytest.approx(0.0, abs=1e-9)
    for axis in np.eye(3):
        expected = _direct_lattice_field(
            _structure(), _single_q_model((0.0, 0.0, 0.0), axis), _GENERAL_SITE, _RADIUS
        )
        np.testing.assert_allclose(tensor @ axis, expected, rtol=1e-6, atol=1e-9)


# Long-wavelength order along c in simple cubic, against a sum over whole ab planes
#
# The field of order that varies only along c is summed plane by plane: each ab
# plane of ions is summed directly over a disc around the muon and as a continuum
# sheet beyond it. A plane's field falls off exponentially with its distance, so
# a few planes either side of the muon give the field of the infinite crystal,
# with no sphere and no Lorentz construction. In the continuum limit a
# longitudinal mode gives -2 mu0 M / 3 at the body centre and a transverse one
# +mu0 M / 3 (the outside field -mu0 (3 q̂q̂ - 1) . M / 3 as qR -> 0).

_SC_LATTICE = 4.0
_SC_WAVEVECTOR = 0.0137  # r.l.u. along c, a 292 Å period
_SC_BODY_CENTRE = (0.5, 0.5, 0.5)
_sc_structure = partial(
    host_structure, label="Fe", lattice=_SC_LATTICE, species=("Fe",), positions=((0.0, 0.0, 0.0),)
)


@cache
def _ab_plane_tensors(muon_frac: tuple[float, float, float]) -> dict[int, np.ndarray]:
    """Dipolar tensor in T/μB of each nearby ab plane of the simple cubic lattice at the muon."""

    a = _SC_LATTICE
    lateral, planes = 200, 12
    muon = np.asarray(muon_frac, dtype=float) * a
    base = np.floor(muon_frac)
    steps = np.arange(-lateral, lateral + 1)
    i, j = np.meshgrid(steps, steps, indexing="ij")
    x = (i.ravel() + base[0]) * a - muon[0]
    y = (j.ravel() + base[1]) * a - muon[1]
    disc_radius = (lateral - 1) * a
    inside = x**2 + y**2 <= disc_radius**2
    x, y = x[inside], y[inside]
    tensors: dict[int, np.ndarray] = {}
    for n in range(int(base[2]) - planes, int(base[2]) + planes + 1):
        height = n * a - muon[2]
        r = np.column_stack([x, y, np.full_like(x, height)])
        r2 = np.sum(r**2, axis=1)
        tensor = 3.0 * np.einsum("ni,nj,n->ij", r, r, r2**-2.5) - np.eye(3) * np.sum(r2**-1.5)
        # The sheet beyond the disc, with 1 / a^2 ions per unit area.
        u = disc_radius**2 + height**2
        in_plane = np.pi * (u**-0.5 - height**2 * u**-1.5)
        tensor += np.diag([in_plane, in_plane, -2.0 * in_plane]) / a**2
        tensors[n] = DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3 * tensor
    return tensors


def _ab_plane_field(model, muon_frac) -> np.ndarray:
    """Field in T at ``muon_frac`` of the moments ``model`` gives each ab plane."""

    structure = _sc_structure()
    return sum(
        tensor @ moments_at_atoms(structure, model, [0], [[0.0, 0.0, float(n)]])[0]
        for n, tensor in _ab_plane_tensors(tuple(float(v) for v in muon_frac)).items()
    )


def _ab_plane_field_ellipse(model, q, muon_frac) -> np.ndarray:
    """Semi-axes in T of the field ellipse traced as a single-q model's phase turns."""

    structure = _sc_structure()
    first, second = moments_at_atoms(structure, model, [0, 0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # Moments Re[F exp(-2 pi i q n)] on plane n.
    angle = 2.0 * np.pi * q
    coefficient = first + 1j * (second - first * np.cos(angle)) / np.sin(angle)
    field = sum(
        tensor @ (coefficient * np.exp(-2j * np.pi * q * n))
        for n, tensor in _ab_plane_tensors(tuple(float(v) for v in muon_frac)).items()
    )
    return np.linalg.svd(np.column_stack([field.real, field.imag]), compute_uv=False)


def _sc_options(radius: float, **changes) -> MagneticFieldOptions:
    cells = int(2 * radius / _SC_LATTICE) + 4
    return MagneticFieldOptions(supercell=(cells, cells, cells), radius=radius, **changes)


def test_ab_plane_sum_gives_the_continuum_fields_of_a_long_wavelength_mode() -> None:
    bulk = MAGNETIZATION_FIELD_T_PER_MUB_PER_ANGSTROM_MINUS3 / _SC_LATTICE**3
    for moment, expected in (((0.0, 0.0, 1.0), 2.0 / 3.0), ((1.0, 0.0, 0.0), 1.0 / 3.0)):
        model = _single_q_model((0.0, 0.0, _SC_WAVEVECTOR), moment)
        major, _minor = _ab_plane_field_ellipse(model, _SC_WAVEVECTOR, _SC_BODY_CENTRE)
        assert major == pytest.approx(expected * bulk, rel=2e-3)


@pytest.mark.parametrize("radius", [40.0, 100.0, 200.0])
@pytest.mark.parametrize(
    ("real", "imaginary", "continuum"),
    [
        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (2.0 / 3.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0 / 3.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0 / 3.0, 1.0 / 3.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0 / 3.0, 1.0 / 3.0)),
    ],
    ids=["longitudinal-sdw", "transverse-sdw", "cycloid", "helix"],
)
def test_real_muesr_long_wavelength_distribution_matches_the_plane_sum_at_any_radius(
    real_muesr, radius, real, imaginary, continuum
) -> None:
    model = _single_q_model((0.0, 0.0, _SC_WAVEVECTOR), real, imaginary)

    (result,) = calculate_incommensurate_field_distributions(
        _sc_structure(),
        model,
        [_SC_BODY_CENTRE],
        options=_sc_options(radius, nangles=1024, phase_sample_count=1024),
    )

    major, minor = _ab_plane_field_ellipse(model, _SC_WAVEVECTOR, _SC_BODY_CENTRE)
    magnitudes = result.magnitudes_T
    assert magnitudes.max() == pytest.approx(major, rel=1e-2)
    assert magnitudes.min() == pytest.approx(minor, abs=1e-2 * major)
    bulk = MAGNETIZATION_FIELD_T_PER_MUB_PER_ANGSTROM_MINUS3 / _SC_LATTICE**3
    np.testing.assert_allclose(
        [magnitudes.max(), magnitudes.min()], np.asarray(continuum) * bulk, rtol=0, atol=1e-2 * bulk
    )
    assert result.calculation_parameters["modulated"] is True


@pytest.mark.parametrize("radius", [40.0, 100.0, 200.0])
@pytest.mark.parametrize(
    "moment", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)], ids=["longitudinal", "transverse"]
)
def test_real_muesr_commensurate_long_wavelength_field_matches_the_plane_sum(
    real_muesr, radius, moment
) -> None:
    model = _single_q_model((0.0, 0.0, 1 / 64), moment)

    (result,) = calculate_commensurate_fields(
        _sc_structure(), model, [_SC_BODY_CENTRE], options=_sc_options(radius)
    )

    expected = _ab_plane_field(model, _SC_BODY_CENTRE)
    np.testing.assert_allclose(
        np.add(result.dipolar, result.lorentz),
        expected,
        rtol=0,
        atol=1e-2 * np.linalg.norm(expected),
    )
    assert result.calculation_parameters["modulated"] is True


@pytest.mark.parametrize(
    "moment", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)], ids=["longitudinal", "transverse"]
)
def test_real_muesr_approximant_field_matches_the_plane_sum_at_each_translation(
    real_muesr, moment
) -> None:
    model = _single_q_model((0.0, 0.0, 1 / 12), moment)

    results = calculate_commensurate_approximant_fields(
        _sc_structure(), model, [_SC_BODY_CENTRE], options=_sc_options(40.0)
    )

    assert len(results) == 12
    scale = max(np.linalg.norm(_ab_plane_field(model, r.muon_position)) for r in results)
    for result in results:
        np.testing.assert_allclose(
            np.add(result.dipolar, result.lorentz),
            _ab_plane_field(model, result.muon_position),
            rtol=0,
            atol=1e-2 * scale,
        )


# Worked examples in docs/user_guide/magnetic_fields.rst, at the default calculation settings

_FACE_CENTRE = (0.5, 0.5, 0.0)
_EDGE_CENTRE = (0.5, 0.0, 0.0)
_DOC_PRECISION = 5e-5


# --- bcc Fe, the ferromagnet worked example ---------------------------------------
#
# a = 2.868 Å with 2.2 mu_B per Fe along c, so mu0 M = 2.17 T. The Lorentz field
# mu0 M / 3 = 0.72 T and the demagnetizing field of a sphere cancel. The
# dipolar field at the interstitial sites is a traceless tensor that averages
# to zero over the equivalent sites (Schmolz et al. 1986; Onuorah et al. 2018).
# An 80 x 80 x 80 lattice sum keeps the cubic symmetry of the Lorentz sphere
# exactly, so the components that vanish by symmetry are zero.

_FE_A = 2.868
_FE_CELLS = (80, 80, 80)
_FE_OCTAHEDRAL = (0.5, 0.0, 0.0)  # 6b, Fe-Fe axis along a
_FE_OCTAHEDRAL_ALONG_C = (0.0, 0.0, 0.5)  # 6b, Fe-Fe axis along the moment
_FE_TETRAHEDRAL = (0.5, 0.25, 0.0)  # 12d
_FE_TETRAHEDRAL_ALONG_C = (0.5, 0.0, 0.25)
# The contact field measured at low temperature, -1.11 T (Nishida et al. 1977),
# is the coupling times the 2.2 mu_B of the two nearest Fe.
_FE_CONTACT_T_PER_MUB = -1.11 / 2.2


def _iron() -> Structure:
    return host_structure(
        label="Fe",
        lattice=_FE_A,
        species=("Fe", "Fe"),
        positions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    )


def _iron_model() -> MagneticStructureModel:
    return _single_q_model((0.0, 0.0, 0.0), (0.0, 0.0, 2.2))


def _iron_options(**changes) -> MagneticFieldOptions:
    return MagneticFieldOptions(
        supercell=_FE_CELLS,
        contact_coupling=contact_coupling_to_angstrom_minus3(_FE_CONTACT_T_PER_MUB, "T_per_muB"),
        **changes,
    )


def test_worked_example_iron_dipolar_fields(real_muesr) -> None:
    sites = [_FE_OCTAHEDRAL, _FE_OCTAHEDRAL_ALONG_C, _FE_TETRAHEDRAL, _FE_TETRAHEDRAL_ALONG_C]

    results = calculate_commensurate_fields(
        _iron(), _iron_model(), sites, options=MagneticFieldOptions(supercell=_FE_CELLS)
    )

    # Octahedral: +1.85 T where the Fe-Fe axis lies along M, -0.93 T for the other two sublattices.
    assert [result.dipolar[2] for result in results[:2]] == pytest.approx([-0.93, 1.85], abs=5e-3)
    # Tetrahedral: +0.26 T and -0.53 T.
    assert [result.dipolar[2] for result in results[2:]] == pytest.approx([0.26, -0.53], abs=5e-3)
    for result in results:
        assert result.dipolar[:2] == pytest.approx((0.0, 0.0), abs=1e-9)
        assert result.lorentz[2] == pytest.approx(0.72, abs=5e-3)


@pytest.mark.parametrize(
    ("site", "multiplicity"),
    [(_FE_OCTAHEDRAL, 6), (_FE_TETRAHEDRAL, 12)],
    ids=["octahedral", "tetrahedral"],
)
def test_worked_example_iron_dipolar_field_averages_to_zero_over_the_equivalent_sites(
    real_muesr, site, multiplicity
) -> None:
    results = calculate_commensurate_fields(
        _iron(),
        _iron_model(),
        [site],
        options=MagneticFieldOptions(supercell=_FE_CELLS, find_equivalent_sites=True),
    )

    assert len(results) == multiplicity
    assert np.mean([result.dipolar for result in results], axis=0) == pytest.approx(
        (0.0, 0.0, 0.0), abs=5e-3
    )


def test_worked_example_iron_lorentz_and_demagnetizing_fields_cancel_for_a_sphere(
    real_muesr,
) -> None:
    sphere = calculate_commensurate_fields(
        _iron(), _iron_model(), [_FE_OCTAHEDRAL], options=_iron_options(demag_shape="sphere")
    )[0]
    bare = calculate_commensurate_fields(
        _iron(), _iron_model(), [_FE_OCTAHEDRAL], options=_iron_options()
    )[0]
    plate = calculate_commensurate_fields(
        _iron(),
        _iron_model(),
        [_FE_OCTAHEDRAL],
        options=_iron_options(demag_shape="infinite_plate"),
    )[0]

    assert bare.contact == pytest.approx((0.0, 0.0, -1.11), abs=1e-6)
    assert sphere.lorentz[2] == pytest.approx(0.7243, abs=5e-4)
    assert sphere.demag[2] == pytest.approx(-0.7246, abs=5e-4)
    # Total at the 6b site (axis along a) with the measured contact field: the sphere and a bare lattice differ by -0.72 T.
    assert bare.total[2] == pytest.approx(-1.3118, abs=5e-4)
    assert sphere.total[2] == pytest.approx(-2.0364, abs=5e-4)
    assert plate.demag[2] == pytest.approx(-2.1737, abs=5e-4)


def test_worked_example_iron_measured_magnetisation_replaces_the_model_value(real_muesr) -> None:
    options = _iron_options(demag_shape="sphere", measured_magnetization=(0.0, 0.0, 2.0))

    (result,) = calculate_commensurate_fields(
        _iron(), _iron_model(), [_FE_OCTAHEDRAL], options=options
    )

    assert result.demag[2] == pytest.approx(-0.6587, abs=5e-4)
    assert result.total[2] == pytest.approx(-1.9705, abs=5e-4)


def test_worked_example_iron_dipolar_tensors(real_muesr) -> None:
    octahedral, tetrahedral = calculate_dipolar_tensors_for_atoms(
        _iron(),
        [_FE_OCTAHEDRAL, _FE_TETRAHEDRAL],
        magnetic_ion_indices=[0, 1],
        options=MagneticFieldOptions(supercell=_FE_CELLS),
    )

    np.testing.assert_allclose(
        octahedral.tensor_T_per_muB, np.diag([0.8419, -0.4210, -0.4210]), atol=_DOC_PRECISION * 4
    )
    np.testing.assert_allclose(
        tetrahedral.tensor_T_per_muB, np.diag([0.1194, -0.2387, 0.1194]), atol=_DOC_PRECISION * 4
    )


def test_worked_example_a_type_antiferromagnet(real_muesr) -> None:
    model = _single_q_model((0.0, 0.0, 0.5), (0.0, 0.0, 1.0))

    results = calculate_commensurate_fields(
        _structure(), model, [(0.5, 0.5, 0.5), _FACE_CENTRE, _EDGE_CENTRE]
    )

    for result, total in zip(results, (0.0, -0.5493, -0.8003), strict=True):
        # The finite sphere's count of the alternating layers leaves under 0.1 mT.
        assert result.lorentz == pytest.approx((0.0, 0.0, 0.0), abs=1e-4)
        assert result.total == pytest.approx((0.0, 0.0, total), abs=_DOC_PRECISION)


# --- MnSi, the incommensurate helix worked example (tests/support/mnsi.py) ---------


def test_worked_example_mnsi_helix_is_left_handed(real_muesr) -> None:
    for handedness in (-1, 1):
        moments = moments_at_atoms(
            mnsi_structure(), mnsi_helix(handedness), [0] * 4, [(n, n, n) for n in range(4)]
        )
        axis = np.ones(3) / np.sqrt(3.0)

        assert np.linalg.norm(moments, axis=1) == pytest.approx(MNSI_MOMENT)
        assert moments @ axis == pytest.approx(0.0, abs=1e-12)
        # Left-handed: the moment turns against the screw sense along +q.
        turns = [
            float(np.cross(first, second) @ axis) for first, second in zip(moments, moments[1:])
        ]
        assert all(turn * handedness > 0.0 for turn in turns)


def test_worked_example_mnsi_distribution_edges_match_the_amato_prediction(real_muesr) -> None:
    site_one, *others = calculate_incommensurate_field_distributions(
        mnsi_structure(), mnsi_helix(), [MNSI_MUON], options=mnsi_options()
    )

    # The grid of phases is deterministic. Site I, on the k axis, is one line at 88 mT.
    np.testing.assert_allclose(site_one.magnitudes_T, 0.0879, atol=5e-4)
    # Sites II-IV are equivalent and span 90 mT to 207 mT, the two horns of the lineshape.
    assert len(others) == 3
    for site in others:
        assert site.magnitudes_T.min() == pytest.approx(0.0901, abs=5e-4)
        assert site.magnitudes_T.max() == pytest.approx(0.2066, abs=5e-4)
    # The singleton lies below the continuum's lower edge, as measured (91 mT and 96 mT).
    assert site_one.magnitudes_T.max() < others[0].magnitudes_T.min()
    assert all(check.converged for check in site_one.convergence)


def test_worked_example_mnsi_site_two_terms_average_to_zero_but_not_their_size(
    real_muesr,
) -> None:
    (site_two,) = calculate_incommensurate_field_distributions(
        mnsi_structure(),
        mnsi_helix(),
        [MNSI_SITE_II],
        options=mnsi_options(find_equivalent_sites=False),
    )

    # Every term of the helix averages to zero over phase; its RMS keeps its size.
    for name in ("dipolar", "lorentz", "contact"):
        assert np.linalg.norm(site_two.terms[name].mean_T) < 1e-9
    assert site_two.terms["dipolar"].rms_T == pytest.approx(0.0810, abs=5e-4)
    assert site_two.terms["contact"].rms_T == pytest.approx(0.1943, abs=5e-4)
    assert site_two.terms["lorentz"].rms_T == pytest.approx(0.00036, abs=5e-5)
    assert site_two.total_rms_T == pytest.approx(0.1594, abs=5e-4)


def test_worked_example_mnsi_contact_term_is_applied_to_the_sampled_distribution(
    real_muesr,
) -> None:
    dipolar_only = calculate_incommensurate_field_distributions(
        mnsi_structure(),
        mnsi_helix(),
        [MNSI_MUON],
        options=MagneticFieldOptions(find_equivalent_sites=True),
    )

    # Without the contact coupling the dipolar field alone gives 106.5 mT at site I and 47.8 mT to 104.4 mT at II-IV.
    np.testing.assert_allclose(dipolar_only[0].magnitudes_T, 0.1065, atol=5e-4)
    assert dipolar_only[1].magnitudes_T.min() == pytest.approx(0.0478, abs=5e-4)
    assert dipolar_only[1].magnitudes_T.max() == pytest.approx(0.1044, abs=5e-4)


def test_worked_example_mnsi_the_helix_sense_matters_for_sites_two_to_four(real_muesr) -> None:
    right = calculate_incommensurate_field_distributions(
        mnsi_structure(), mnsi_helix(1), [MNSI_MUON], options=mnsi_options()
    )

    # Site I is on the k axis and does not tell the senses apart; sites II-IV do.
    np.testing.assert_allclose(right[0].magnitudes_T, 0.0879, atol=5e-4)
    assert right[1].magnitudes_T.min() == pytest.approx(0.0856, abs=5e-4)
    assert right[1].magnitudes_T.max() == pytest.approx(0.2086, abs=5e-4)


def test_worked_example_mnsi_has_a_commensurate_approximant_of_period_64(real_muesr) -> None:
    approximant = find_commensurate_approximant(
        mnsi_structure(), mnsi_helix(), tolerance=1e-3, max_period=64
    )
    nothing_shorter = find_commensurate_approximant(
        mnsi_structure(), mnsi_helix(), tolerance=1e-3, max_period=12
    )

    assert nothing_shorter is None
    assert approximant is not None
    assert approximant.approximant_rlu == ((1 / 64,) * 3,)
    assert approximant.periods == (64, 64, 64)
    assert approximant.max_deviation_rlu == pytest.approx(0.000925)


def test_worked_example_mnsi_approximant_fields_span_the_continuum(real_muesr) -> None:
    results = calculate_commensurate_approximant_fields(
        mnsi_structure(),
        mnsi_helix(),
        [MNSI_SITE_II],
        options=mnsi_options(find_equivalent_sites=False, approximant_max_period=64),
    )

    magnitudes = np.array([result.magnitude for result in results])
    # One field per translation of the 64 x 64 x 64 supercell with a distinct phase.
    assert len(results) == 64
    assert results[1].site_label == "site 1 + (1, 0, 0)"
    assert results[1].muon_position == pytest.approx((1.968, 0.468, 0.032))
    # The 64 sampled phases come within 0.7 mT of the analytic edges of the sampled distribution.
    assert magnitudes.min() == pytest.approx(0.0897, abs=1e-3)
    assert magnitudes.max() == pytest.approx(0.2072, abs=1e-3)


# --- CoF2, the user guide's worked example ----------------------------------------
#
# Rutile CoF2 (a = 4.695 Å, c = 3.180 Å, F at 4f with x = 0.306; Baur and Khan
# 1971) with antiparallel 2.64 μB moments along c on the corner and body-centre
# Co (Lines 1965). The positive muon sits at the 4c site (0, 1/2, 0); the
# published rigid-lattice dipolar field there is 0.265 T (De Renzi et al. 1984;
# Huddart et al. 2022), and De Renzi et al. tabulate the sublattice dipolar
# tensors at the equivalent site (1/2, 0, 0) in Å⁻³ per μB.

_COF2_X = 0.306
_COF2_A = 4.695
_cof2 = partial(
    host_structure,
    label="CoF2",
    lattice=[[_COF2_A, 0.0, 0.0], [0.0, _COF2_A, 0.0], [0.0, 0.0, 3.180]],
    species=("Co", "Co", "F", "F", "F", "F"),
    positions=(
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (_COF2_X, _COF2_X, 0.0),
        (1.0 - _COF2_X, 1.0 - _COF2_X, 0.0),
        (0.5 + _COF2_X, 0.5 - _COF2_X, 0.5),
        (0.5 - _COF2_X, 0.5 + _COF2_X, 0.5),
    ),
)
_COF2_4C = (0.0, 0.5, 0.0)


def _cof2_antiferromagnet() -> MagneticStructureModel:
    def sublattice(index: int, moment: float) -> MagneticComponent:
        return MagneticComponent(
            component_id=f"co-{index}",
            label=f"Co {index}",
            kind="collinear",
            selector=AtomSelector.indices([index]),
            terms=(FourierTerm((0.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, moment])),),
        )

    return MagneticStructureModel(components=(sublattice(0, 2.64), sublattice(1, -2.64)))


def test_cof2_rigid_lattice_dipolar_field_at_the_4c_site(real_muesr) -> None:
    (result,) = calculate_commensurate_fields(_cof2(), _cof2_antiferromagnet(), [_COF2_4C])

    assert result.dipolar[2] == pytest.approx(-0.2655, abs=5e-4)
    assert abs(result.dipolar[0]) < 1e-9 and abs(result.dipolar[1]) < 1e-9
    assert result.magnitude == pytest.approx(0.265, abs=1e-3)


def test_cof2_sublattice_dipolar_tensors_match_de_renzi(real_muesr) -> None:
    corner, body = (
        calculate_dipolar_tensors_for_atoms(
            _cof2(), [(0.5, 0.0, 0.0)], magnetic_ion_indices=[index]
        )[0]
        for index in (0, 1)
    )

    per_angstrom3 = 1.0 / DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3
    np.testing.assert_allclose(
        np.diag(corner.tensor_T_per_muB) * per_angstrom3, (0.292, -0.177, -0.115), atol=0.004
    )
    np.testing.assert_allclose(
        np.diag(body.tensor_T_per_muB) * per_angstrom3, (-0.152, 0.159, -0.006), atol=0.004
    )


def test_cof2_distortion_correction_from_the_mufinder_paper_displacements(real_muesr) -> None:
    # The two nearest F move in by 0.50 and 0.57 Å and the two nearest Co out by
    # 0.14 and 0.17 Å (Huddart et al. 2022), in a 2x2x2 supercell.
    lattice = np.diag([_COF2_A, _COF2_A, 3.180])
    muon = np.asarray(_COF2_4C) @ lattice

    def radial(site: tuple[float, float, float], change: float) -> tuple[float, ...]:
        vector = np.asarray(site) @ lattice - muon
        return tuple(change * vector / np.linalg.norm(vector))

    moves = {
        (_COF2_X, _COF2_X, 0.0): radial((_COF2_X, _COF2_X, 0.0), -0.50),
        (-_COF2_X, 1.0 - _COF2_X, 0.0): radial((-_COF2_X, 1.0 - _COF2_X, 0.0), -0.57),
        (0.0, 0.0, 0.0): radial((0.0, 0.0, 0.0), 0.14),
        (0.0, 1.0, 0.0): radial((0.0, 1.0, 0.0), 0.17),
    }
    relaxed = relaxed_muon_supercell(_cof2(), (2, 2, 2), _COF2_4C, moves)

    (result,) = calculate_commensurate_fields(
        _cof2(), _cof2_antiferromagnet(), [_COF2_4C], distorted_structures=[relaxed]
    )

    assert result.distortion is not None
    assert result.distortion[2] == pytest.approx(0.0660, abs=5e-4)
    assert result.magnitude == pytest.approx(0.1996, abs=5e-4)


# --- The contact shell: ties across its edge and one shell for every component ------
#
# B2 FeMn (a = 2.868 Å) with Fe up and Mn down along c. The tetrahedral site
# (1/2, 1/4, 0) has two Fe and two Mn tied at 1.603 Å; the octahedral site
# (1/2, 0, 0) has two Fe at 1.434 Å and four Mn behind them at 2.028 Å.

_B2_TETRAHEDRAL = (0.5, 0.25, 0.0)
_B2_OCTAHEDRAL = (0.5, 0.0, 0.0)
_b2_structure = partial(
    host_structure, label="B2", lattice=2.868, species=("Fe", "Mn"), positions=BODY_CENTRE
)


def _b2_component(species: str, moment: float, q_coeff=(0.0, 0.0, 0.0)) -> MagneticComponent:
    return MagneticComponent(
        component_id=species,
        label=species,
        kind="custom",
        selector=AtomSelector.species(species),
        terms=(FourierTerm(q_coeff, ComplexVector.from_vector([0.0, 0.0, moment])),),
    )


def _b2_antiferromagnet(mn_q=None) -> MagneticStructureModel:
    """Fe up and Mn down as two components; ``mn_q`` puts the Mn one at a reciprocal lattice vector."""

    if mn_q is None:
        return MagneticStructureModel(
            components=(_b2_component("Fe", 2.0), _b2_component("Mn", -2.0))
        )
    return MagneticStructureModel(
        q_basis_rlu=(tuple(mn_q), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(_b2_component("Fe", 2.0), _b2_component("Mn", -2.0, (1.0, 0.0, 0.0))),
    )


def _b2_ferromagnet() -> MagneticStructureModel:
    return MagneticStructureModel(components=(_b2_component("Fe", 2.0), _b2_component("Mn", 2.0)))


_B2_COUPLING = contact_coupling_to_angstrom_minus3(-0.5, "T_per_muB")


def _zero_locfield(sample, mode, *_args, **kwargs):
    rows = int(kwargs["nangles"]) if mode == "i" else 1
    return [_Field(Dipolar=np.zeros((rows, 3)), Contact=np.zeros((rows, 3))) for _ in sample.muons]


def _contact_warnings(result) -> list[str]:
    return list(result.calculation_parameters.get("contact_shell_warnings", []))


def test_a_contact_shell_that_splits_a_tie_of_different_moments_is_warned(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_zero_locfield)
    options = MagneticFieldOptions(supercell=(1, 1, 1), contact_coupling=_B2_COUPLING, nnn=2)

    (result,) = calculate_commensurate_fields(
        _b2_structure(), _b2_antiferromagnet(), [_B2_TETRAHEDRAL], options=options
    )

    (warning,) = _contact_warnings(result)
    assert "Contact neighbours (nnn) = 2 keeps 2 of the 4 magnetic ions" in warning
    assert "1.603 Å" in warning
    assert "2 × Fe 1, 2 × Mn 2" in warning
    assert "nnn = 4 includes them all" in warning


@pytest.mark.parametrize(
    ("model", "nnn", "coupling"),
    [
        (_b2_antiferromagnet, 4, _B2_COUPLING),
        (_b2_antiferromagnet, 8, _B2_COUPLING),
        (_b2_ferromagnet, 2, _B2_COUPLING),
        (_b2_antiferromagnet, 2, None),
    ],
    ids=["whole-tie", "two-whole-shells", "equal-moments", "no-contact-term"],
)
def test_a_contact_shell_that_keeps_a_tie_whole_or_ties_equal_moments_is_not_warned(
    monkeypatch, model, nnn, coupling
) -> None:
    _install_fake_muesr(monkeypatch, locfield=_zero_locfield)
    options = MagneticFieldOptions(supercell=(1, 1, 1), contact_coupling=coupling, nnn=nnn)

    (result,) = calculate_commensurate_fields(
        _b2_structure(), model(), [_B2_TETRAHEDRAL], options=options
    )

    assert _contact_warnings(result) == []


def test_the_tie_warning_names_ions_within_the_tolerance(monkeypatch) -> None:
    # Moving the site 0.0004 Å along b leaves the four ions within 1e-3 Å of each other.
    _install_fake_muesr(monkeypatch, locfield=_zero_locfield)
    nudged = (0.5, 0.25 + 0.0004 / 2.868, 0.0)
    options = MagneticFieldOptions(supercell=(1, 1, 1), contact_coupling=_B2_COUPLING, nnn=3)

    (result,) = calculate_commensurate_fields(
        _b2_structure(), _b2_antiferromagnet(), [nudged], options=options
    )

    (warning,) = _contact_warnings(result)
    assert "keeps 3 of the 4 magnetic ions" in warning
    assert module.CONTACT_SHELL_TIE_TOLERANCE_ANG == pytest.approx(1e-3)


def test_the_mnsi_contact_shell_warns_at_two_neighbours_and_not_at_three(monkeypatch) -> None:
    # The three nearest Mn tie at 1.760 Å. At site I, on the helix axis, they
    # share one helix phase and so one moment; at sites II to IV they differ.
    _install_fake_muesr(monkeypatch, locfield=_zero_locfield)

    def _warnings(nnn: int) -> list[list[str]]:
        results = calculate_incommensurate_field_distributions(
            mnsi_structure(),
            mnsi_helix(),
            [MNSI_MUON],
            options=mnsi_options(supercell=(1, 1, 1), nnn=nnn, nangles=16, phase_sample_count=16),
        )
        return [_contact_warnings(result) for result in results]

    site_one, *others = _warnings(2)
    assert site_one == []
    assert len(others) == 3
    assert all(len(site) == 1 and "keeps 2 of the 3 magnetic ions" in site[0] for site in others)
    assert "1.760 Å" in others[0][0] and "nnn = 3 includes them all" in others[0][0]
    assert _warnings(3) == [[], [], [], []]


def test_the_tie_checker_reports_each_site_for_a_live_setting() -> None:
    check = module.contact_shell_tie_checker(
        _b2_structure(), _b2_antiferromagnet(), [_B2_TETRAHEDRAL], find_equivalent_sites=False
    )

    (warning,) = check(2, 10.0)
    assert "nnn = 4 includes them all" in warning
    assert check(4, 10.0) == ()
    # A radius inside the tie leaves no ion to split.
    assert check(2, 1.5) == ()


def test_the_tie_checker_labels_equivalent_sites() -> None:
    check = module.contact_shell_tie_checker(
        _b2_structure(), _b2_antiferromagnet(), [_B2_TETRAHEDRAL], find_equivalent_sites=True
    )

    warnings = check(2, 10.0)

    assert len(warnings) == 12
    assert warnings[0].startswith("site 1 equiv 1: ")


def test_the_tie_checker_has_nothing_to_check_without_moments() -> None:
    check = module.contact_shell_tie_checker(
        _structure(), _empty_model(), [_MU_X], find_equivalent_sites=False
    )

    assert check(2, 10.0) == ()


@pytest.mark.parametrize(
    ("model", "warned"), [(_antiferromagnet, True), (_ferromagnet, False)], ids=["afm", "fm"]
)
def test_contact_distortion_warns_of_a_tie_across_the_shell_edge(
    monkeypatch, model, warned
) -> None:
    # Two Fe tie 1.5 Å from the muon in the relaxed supercell for one contact place.
    _install_fake_muesr(monkeypatch, locfield=_fields_per_muon(Dipolar=(0, 0, 0)))
    relaxed = relaxed_muon_supercell(_structure(), (3, 3, 3), _MU_X)
    options = MagneticFieldOptions(supercell=(1, 1, 1), contact_coupling=0.1, nnn=1, rcont=2.0)

    (result,) = calculate_commensurate_fields(
        _structure(), model(), [_MU_X], options=options, distorted_structures=[relaxed]
    )

    warnings = _contact_warnings(result)
    if warned:
        assert any(
            text.startswith(
                "In the relaxed supercell, Contact neighbours (nnn) = 1 keeps 1 of the 2"
            )
            for text in warnings
        )
    else:
        assert warnings == []
        assert result.distortion == pytest.approx((0.0, 0.0, 0.0), abs=1e-15)


def test_contact_shell_warnings_survive_a_save(monkeypatch) -> None:
    _install_fake_muesr(monkeypatch, locfield=_zero_locfield)
    options = MagneticFieldOptions(supercell=(1, 1, 1), contact_coupling=_B2_COUPLING, nnn=2)
    (result,) = calculate_commensurate_fields(
        _b2_structure(), _b2_antiferromagnet(), [_B2_TETRAHEDRAL], options=options
    )

    restored = MagneticFieldResult.from_json(json.loads(json.dumps(result.to_json())))

    assert _contact_warnings(restored) == _contact_warnings(result)


def test_real_muesr_distribution_contact_averages_one_shell_over_every_component(
    real_muesr,
) -> None:
    # The Fe and Mn components are summed separately on the distribution path.
    # Averaging each component's own shell gave the Fe pair's -1000 mT plus the
    # Mn shell's +1000 mT; the octahedral shell holds only the two Fe.
    options = MagneticFieldOptions(
        supercell=(20, 20, 20), contact_coupling=_B2_COUPLING, nangles=16, phase_sample_count=16
    )

    (commensurate,) = calculate_commensurate_fields(
        _b2_structure(), _b2_antiferromagnet(), [_B2_OCTAHEDRAL], options=options
    )
    (distribution,) = calculate_incommensurate_field_distributions(
        _b2_structure(), _b2_antiferromagnet(), [_B2_OCTAHEDRAL], options=options
    )

    assert commensurate.contact == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert distribution.terms["contact"].mean_T == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert distribution.terms["contact"].rms_T == pytest.approx(1.0, abs=1e-6)


def test_real_muesr_commensurate_contact_averages_one_shell_over_every_wavevector(
    real_muesr,
) -> None:
    # The Mn component at q = (0, 0, 1) needs its own lattice sum; its moments
    # point up once the phase of its body-centre position is applied, so all
    # six ions are up and the octahedral shell gives -1000 mT, not twice that.
    model = _b2_antiferromagnet(mn_q=(0.0, 0.0, 1.0))
    options = MagneticFieldOptions(supercell=(20, 20, 20), contact_coupling=_B2_COUPLING)

    (result,) = calculate_commensurate_fields(
        _b2_structure(), model, [_B2_OCTAHEDRAL], options=options
    )

    assert result.contact == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)


def test_real_muesr_shared_shell_matches_the_backend_when_every_call_sees_the_shell(
    real_muesr, monkeypatch
) -> None:
    # A linear SDW is summed as two circular helices on the same ions, so the
    # backend's own per-call average is already right there; the shared shell
    # must reproduce it row for row, which pins its lookup phases.
    model = _single_q_model((0.3, 0.0, 0.0), (0.0, 0.0, 1.0))
    options = MagneticFieldOptions(
        supercell=(15, 15, 15),
        radius=_RADIUS,
        contact_coupling=0.1,
        nnn=3,
        nangles=64,
        phase_sampling="grid",
        phase_sample_count=64,
    )

    def _contact():
        (result,) = calculate_incommensurate_field_distributions(
            _structure(), model, [_GENERAL_SITE], options=options
        )
        return result.terms["contact"]

    shared = _contact()
    monkeypatch.setattr(module, "_needs_shared_contact_shell", lambda entries: False)
    backend = _contact()

    assert backend.rms_T > 1e-3
    assert shared.mean_T == pytest.approx(backend.mean_T, abs=1e-9)
    assert shared.rms_T == pytest.approx(backend.rms_T, rel=1e-6)


def test_the_tie_warning_does_not_call_every_tied_moment_different(monkeypatch) -> None:
    # The four ions tied at 1.603 Å are two Fe up and two Mn down: their
    # moments are not all different, only not all equal.
    _install_fake_muesr(monkeypatch, locfield=_zero_locfield)
    options = MagneticFieldOptions(supercell=(1, 1, 1), contact_coupling=_B2_COUPLING, nnn=3)

    (result,) = calculate_commensurate_fields(
        _b2_structure(), _b2_antiferromagnet(), [_B2_TETRAHEDRAL], options=options
    )

    (warning,) = _contact_warnings(result)
    assert "magnetic ions that tie at 1.603 Å from the muon (2 × Fe 1, 2 × Mn 2)" in warning
    assert "whose moments are not all equal" in warning
    assert "with different moments" not in warning


def test_the_tie_checker_of_a_commensurate_order_compares_the_summed_moments() -> None:
    # Fe and Mn both up, as two k = 0 components: the commensurate sum merges
    # them, so the four tied ions carry one moment and the result has no warning.
    check = module.contact_shell_tie_checker(
        _b2_structure(), _b2_ferromagnet(), [_B2_TETRAHEDRAL], find_equivalent_sites=False
    )

    assert check(2, 10.0) == ()
