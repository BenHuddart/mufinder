"""The Magnetic Fields calculations the page runs off the GUI thread."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import mufinder.services.magnetic_field_calculation as calculation_module
from mufinder.science.magnetic_fields import (
    MagneticFieldError,
    MagneticFieldOptions,
    calculate_commensurate_approximant_fields,
    calculate_commensurate_fields,
    calculate_dipolar_tensors_for_atoms,
    calculate_incommensurate_field_distributions,
)
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    build_collinear_component,
    build_magnetic_preset_component,
)
from mufinder.science.nuclear_relaxation import (
    NuclearRelaxationError,
    NuclearRelaxationOptions,
    calculate_nuclear_relaxation,
)
from mufinder.services.magnetic_field_calculation import (
    COMMENSURATE_RESULTS,
    DISTRIBUTION_RESULTS,
    NonFiniteFieldError,
    equivalent_site_count,
    non_finite_field_label,
    cancelled_status,
    prepare_dipolar_tensor_calculation,
    prepare_magnetic_field_calculation,
    prepare_nuclear_relaxation_calculation,
    run_dipolar_tensor_calculation,
    run_magnetic_field_calculation,
    run_nuclear_relaxation_calculation,
    run_prepared_magnetic_field_calculation,
    site_count_text,
)
from mufinder.services.magnetic_field_results import UEP_SITE_SOURCE, UNRELAXED_ESTIMATE_SUFFIX
from mufinder.services.magnetic_field_sites import MagneticMuonSiteOption
from mufinder.services.project_service import ProjectService
from tests.support.magnetic_fields import custom_muon_site
from tests.support.mnsi import MNSI_MUON, mnsi_helix, mnsi_options, mnsi_structure
from tests.support.structures import host_structure

_FE_TETRAHEDRAL = (0.5, 0.25, 0.0)


def _fe():
    return host_structure(
        label="Fe",
        lattice=2.868,
        species=("Fe", "Fe"),
        positions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    )


def _fe_ferromagnet() -> MagneticStructureModel:
    return MagneticStructureModel(
        components=(
            build_collinear_component(
                label="m",
                selector=AtomSelector.species("Fe"),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, 2.2),
            ),
        )
    )


def _site(position) -> MagneticMuonSiteOption:
    return custom_muon_site(tuple(float(value) for value in position))


def test_fe_fields_are_those_of_the_science_call(real_muesr) -> None:
    options = MagneticFieldOptions(find_equivalent_sites=True, supercell=(30, 30, 30))
    expected = calculate_commensurate_fields(
        _fe(), _fe_ferromagnet(), [_FE_TETRAHEDRAL], options=options, distorted_structures=[None]
    )

    calculation = run_magnetic_field_calculation(
        None, _fe(), _fe_ferromagnet(), _site(_FE_TETRAHEDRAL), options, use_approximants=False
    )

    assert calculation.kind == COMMENSURATE_RESULTS
    assert calculation.results == expected
    assert len(expected) == 12
    assert calculation.status == "Results: calculated 12 commensurate field values."


def test_mnsi_distributions_are_those_of_the_science_call(real_muesr) -> None:
    options = mnsi_options(supercell=(30, 30, 30), approximant_max_period=64)
    expected = calculate_incommensurate_field_distributions(
        mnsi_structure(), mnsi_helix(), [MNSI_MUON], options=options, distorted_structures=[None]
    )

    calculation = run_magnetic_field_calculation(
        None, mnsi_structure(), mnsi_helix(), _site(MNSI_MUON), options, use_approximants=False
    )

    assert calculation.kind == DISTRIBUTION_RESULTS
    assert [result.site_label for result in calculation.results] == [
        result.site_label for result in expected
    ]
    for result, reference in zip(calculation.results, expected, strict=True):
        np.testing.assert_array_equal(result.fields_T, reference.fields_T)
    assert calculation.status.startswith("Results: sampled 4 incommensurate field distributions")
    # MnSi's q is close to 1/64: the status says so and how to list its fields.
    assert "turn on commensurate approximants" in calculation.status


def test_mnsi_lists_its_approximant_when_the_settings_allow_one(real_muesr) -> None:
    options = mnsi_options(
        supercell=(8, 8, 8), find_equivalent_sites=False, approximant_max_period=64
    )
    expected = calculate_commensurate_approximant_fields(
        mnsi_structure(), mnsi_helix(), [MNSI_MUON], options=options, distorted_structures=[None]
    )

    calculation = run_magnetic_field_calculation(
        None, mnsi_structure(), mnsi_helix(), _site(MNSI_MUON), options, use_approximants=True
    )

    assert calculation.kind == COMMENSURATE_RESULTS
    assert calculation.results == expected
    assert "of the commensurate approximant" in calculation.status


def test_a_field_at_a_uep_site_is_marked_as_an_estimate(real_muesr) -> None:
    site = replace(_site((0.5, 0.0, 0.0)), source=UEP_SITE_SOURCE)

    calculation = run_magnetic_field_calculation(
        None,
        _fe(),
        _fe_ferromagnet(),
        site,
        MagneticFieldOptions(supercell=(10, 10, 10)),
        use_approximants=False,
    )

    assert calculation.status.endswith(UNRELAXED_ESTIMATE_SUFFIX)


def test_a_non_finite_field_is_refused_with_its_site(monkeypatch) -> None:
    class _Result:
        site_label = "Custom site equiv 2"
        total = (np.nan, 0.0, 0.0)
        magnitude = np.nan

    monkeypatch.setattr(
        calculation_module, "classify_magnetic_model", lambda *_a, **_k: "commensurate"
    )
    monkeypatch.setattr(
        calculation_module, "calculate_commensurate_fields", lambda *_a, **_k: (_Result(),)
    )

    with pytest.raises(NonFiniteFieldError, match="Custom site equiv 2"):
        run_magnetic_field_calculation(
            None,
            _fe(),
            _fe_ferromagnet(),
            _site((0.5, 0.0, 0.0)),
            MagneticFieldOptions(),
            use_approximants=False,
        )
    assert non_finite_field_label([{"site_label": "saved", "total_T": [0.0, np.inf, 0.0]}]) == (
        "saved"
    )
    factored = {
        "site_label": "factored",
        "fields_T_factors": {"continuous_T": [[0.0, 0.0, 1.0]], "listed_T": [[np.nan, 0.0, 0.0]]},
    }
    assert non_finite_field_label([factored]) == "factored"


def test_a_prepared_calculation_does_not_see_later_changes_to_the_host() -> None:
    host = _fe()
    request = prepare_magnetic_field_calculation(
        None,
        host,
        _fe_ferromagnet(),
        _site(_FE_TETRAHEDRAL),
        MagneticFieldOptions(find_equivalent_sites=True),
        use_approximants=False,
    )
    host.species[0] = "Co"

    assert request.host.species == ["Fe", "Fe"]
    assert request.site_count == 12


def test_a_distortion_correction_needs_its_relaxed_structure(tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "fields.mufinder", "Fields")
    site = replace(_site((0.5, 0.0, 0.0)), include_distortion=True)

    with pytest.raises(MagneticFieldError, match="no longer available"):
        prepare_magnetic_field_calculation(
            document,
            _fe(),
            _fe_ferromagnet(),
            site,
            MagneticFieldOptions(),
            use_approximants=False,
        )


def test_backend_errors_reach_the_caller(monkeypatch) -> None:
    def _fail(*_args, **_kwargs):
        raise RuntimeError("muLFC refused the cell")

    monkeypatch.setattr(
        calculation_module, "classify_magnetic_model", lambda *_a, **_k: "commensurate"
    )
    monkeypatch.setattr(calculation_module, "calculate_commensurate_fields", _fail)
    request = prepare_magnetic_field_calculation(
        None,
        _fe(),
        _fe_ferromagnet(),
        _site((0.5, 0.0, 0.0)),
        MagneticFieldOptions(),
        use_approximants=False,
    )

    with pytest.raises(RuntimeError, match="muLFC refused the cell"):
        run_prepared_magnetic_field_calculation(request)


def test_site_counts_follow_the_equivalent_sites_option() -> None:
    site = _site(_FE_TETRAHEDRAL)

    every = MagneticFieldOptions(find_equivalent_sites=True)

    assert equivalent_site_count(_fe(), site, every) == 12
    assert (
        equivalent_site_count(_fe(), site, MagneticFieldOptions(find_equivalent_sites=False)) == 1
    )
    unknown = host_structure(species=("Xx", "Xx"))
    assert equivalent_site_count(unknown, site, every) is None
    assert site_count_text(12) == "12 muon sites"
    assert site_count_text(1) == "1 muon site"
    assert site_count_text(None) == "the muon site"


def test_dipolar_tensors_are_those_of_the_science_call(real_muesr) -> None:
    expected = calculate_dipolar_tensors_for_atoms(
        _fe(),
        [_FE_TETRAHEDRAL],
        magnetic_ion_indices=[0, 1],
        options=MagneticFieldOptions(
            supercell=(20, 20, 20), find_equivalent_sites=True, include_lorentz=False
        ),
    )

    tensors = run_dipolar_tensor_calculation(
        _fe(),
        _site(_FE_TETRAHEDRAL),
        magnetic_ion_indices=(0, 1),
        supercell=(20, 20, 20),
        include_equivalent_sites=True,
    )

    assert len(tensors) == len(expected) == 12
    for tensor, reference in zip(tensors, expected, strict=True):
        assert tensor.site_label == reference.site_label
        np.testing.assert_array_equal(tensor.tensor_T_per_muB, reference.tensor_T_per_muB)


def test_nuclear_relaxation_is_that_of_the_science_call() -> None:
    options = NuclearRelaxationOptions()
    site = _site((0.5, 0.0, 0.0))
    expected = calculate_nuclear_relaxation(
        _fe(), site.fractional_position, site_label=site.label, options=options
    )

    result = run_nuclear_relaxation_calculation(None, _fe(), site, options)

    assert result.sigma == expected.sigma
    assert result.site_label == "Custom site"


def test_a_nuclear_distortion_correction_needs_a_relaxed_structure() -> None:
    options = NuclearRelaxationOptions(use_distortion_correction=True)

    with pytest.raises(NuclearRelaxationError, match="no distortion structure"):
        prepare_nuclear_relaxation_calculation(None, _fe(), _site((0.5, 0.0, 0.0)), options)


def test_a_prepared_tensor_calculation_copies_the_host_and_counts_its_sites() -> None:
    host = _fe()
    request = prepare_dipolar_tensor_calculation(
        host,
        _site(_FE_TETRAHEDRAL),
        magnetic_ion_indices=[0, 1],
        supercell=(20, 20, 20),
        include_equivalent_sites=True,
    )
    host.species[0] = "Co"

    assert request.host.species == ["Fe", "Fe"]
    assert request.magnetic_ion_indices == (0, 1)
    assert request.options.supercell == (20, 20, 20)
    assert not request.options.include_lorentz
    assert request.site_count == 12


def test_the_cancelled_status_says_whether_results_are_kept() -> None:
    assert cancelled_status(keeps_results=True) == (
        "Calculation cancelled; the earlier results are kept"
    )
    assert cancelled_status(keeps_results=False) == "Calculation cancelled"


def _fe_helix(q: float) -> MagneticStructureModel:
    component = build_magnetic_preset_component(
        "helix",
        lattice_vectors=_fe().lattice_vectors,
        q_rlu=(q, 0.0, 0.0),
        q_coeff=(1.0, 0.0, 0.0),
        moment=2.2,
        reference_axis="b",
        selector=AtomSelector.all(),
    )
    return MagneticStructureModel(
        q_basis_rlu=((q, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), components=(component,)
    )


def test_the_calculation_classifies_q_with_the_options_it_was_given(real_muesr) -> None:
    calculation = run_magnetic_field_calculation(
        None,
        _fe(),
        _fe_helix(1.0 / 13.0),
        _site((0.1, 0.2, 0.3)),
        MagneticFieldOptions(supercell=(8, 8, 8), commensurate_max_period=13),
        use_approximants=False,
    )

    assert calculation.kind == COMMENSURATE_RESULTS
    assert len(calculation.results) == 13


@pytest.mark.parametrize(
    ("q", "kind", "rows", "status"),
    [
        (1.0 / 3.0, COMMENSURATE_RESULTS, 3, "Results: calculated 3 commensurate field values."),
        (1.0 / 12.0, COMMENSURATE_RESULTS, 12, "Results: calculated 12 commensurate field values."),
        (
            1.0 / 13.0,
            DISTRIBUTION_RESULTS,
            1,
            "Results: sampled 1 incommensurate field distribution.",
        ),
        (0.22, DISTRIBUTION_RESULTS, 1, "Results: sampled 1 incommensurate field distribution."),
    ],
    ids=["period-3", "period-12", "period-13", "period-50"],
)
def test_the_period_of_q_chooses_between_a_list_and_a_distribution(
    real_muesr, q, kind, rows, status
) -> None:
    calculation = run_magnetic_field_calculation(
        None,
        _fe(),
        _fe_helix(q),
        _site((0.1, 0.2, 0.3)),
        MagneticFieldOptions(supercell=(12, 12, 12)),
        use_approximants=False,
    )

    assert calculation.kind == kind
    assert len(calculation.results) == rows
    assert calculation.status.startswith(status)


def test_more_phase_sets_than_a_list_holds_are_a_distribution_over_every_set(real_muesr) -> None:
    # Three helices of period 9, along a, b and c: 729 sets of phases, beyond the 512 listed.
    q = 1.0 / 9.0
    components = tuple(
        build_magnetic_preset_component(
            "helix",
            lattice_vectors=_fe().lattice_vectors,
            q_rlu=tuple(q if axis == index else 0.0 for axis in range(3)),
            q_coeff=tuple(1.0 if axis == index else 0.0 for axis in range(3)),
            moment=1.0,
            reference_axis="abc"[(index + 1) % 3],
            selector=AtomSelector.all(),
        )
        for index in range(3)
    )
    model = MagneticStructureModel(
        q_basis_rlu=((q, 0.0, 0.0), (0.0, q, 0.0), (0.0, 0.0, q)), components=components
    )

    calculation = run_magnetic_field_calculation(
        None,
        _fe(),
        model,
        _site((0.1, 0.2, 0.3)),
        MagneticFieldOptions(supercell=(8, 8, 8)),
        use_approximants=False,
    )

    assert calculation.kind == DISTRIBUTION_RESULTS
    (distribution,) = calculation.results
    assert np.asarray(distribution.fields_T).shape == (729, 3)
    assert distribution.calculation_parameters["resolved_phase_sampling"] == "grid"


def test_an_approximant_of_too_many_phase_sets_is_sampled_instead(real_muesr) -> None:
    # With finite fields asked for near-commensurate q, the 729 phase sets are still a distribution.
    q = 1.0 / 9.0
    components = tuple(
        build_magnetic_preset_component(
            "helix",
            lattice_vectors=_fe().lattice_vectors,
            q_rlu=tuple(q if axis == index else 0.0 for axis in range(3)),
            q_coeff=tuple(1.0 if axis == index else 0.0 for axis in range(3)),
            moment=1.0,
            reference_axis="abc"[(index + 1) % 3],
            selector=AtomSelector.all(),
        )
        for index in range(3)
    )
    model = MagneticStructureModel(
        q_basis_rlu=((q, 0.0, 0.0), (0.0, q, 0.0), (0.0, 0.0, q)), components=components
    )

    calculation = run_magnetic_field_calculation(
        None,
        _fe(),
        model,
        _site((0.1, 0.2, 0.3)),
        MagneticFieldOptions(supercell=(8, 8, 8)),
        use_approximants=True,
    )

    assert calculation.kind == DISTRIBUTION_RESULTS
    (distribution,) = calculation.results
    assert np.asarray(distribution.fields_T).shape == (729, 3)
    assert calculation.status.startswith("Results: sampled 1 incommensurate field distribution.")
    assert calculation.status.endswith(
        "gives more than 512 sets of phases, too many to list as finite fields, so it is sampled."
    )
    assert "turn on commensurate approximants" not in calculation.status
