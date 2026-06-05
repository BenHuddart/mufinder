"""Tests for propagation-vector magnetic structure helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.magnetic_structure import (
    AtomSelector,
    ComplexVector,
    FourierTerm,
    MagneticComponent,
    MagneticStructureModel,
    atom_resolved_fourier_components,
    build_collinear_component,
    build_conical_component,
    build_cycloid_component,
    build_helix_component,
    build_hexagonal_skyrmion_texture,
    build_square_skyrmion_texture,
    build_spin_density_wave_component,
    build_texture_component,
    evaluate_texture_moment,
    get_muesr_model_data,
    magnetic_model_from_legacy_structure,
    moment_at_atom,
    moments_in_supercell,
    normalize_texture_magnitude,
    prune_texture_terms,
    skyrmion_q_basis_rlu,
    split_muesr_model_data_into_helices,
    texture_diagnostics,
    resolve_atom_selector,
    viable_atom_selectors,
)


def _structure() -> Structure:
    return Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _assert_muesr_fc_is_circular_helical(fc: np.ndarray) -> None:
    for vector in np.asarray(fc, dtype=complex):
        if np.allclose(vector, 0.0, atol=1e-12, rtol=0.0):
            continue
        real = vector.real
        imag = vector.imag
        assert np.linalg.norm(real) == pytest.approx(np.linalg.norm(imag), abs=1e-10)
        assert float(np.dot(real, imag)) == pytest.approx(0.0, abs=1e-10)


def test_magnetic_structure_model_round_trips_json() -> None:
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="helix-1",
                label="Helix",
                kind="helix",
                selector=AtomSelector.species("Fe"),
                terms=(
                    FourierTerm(
                        q_coeff=(1.0, 0.0, 0.0),
                        coefficient=ComplexVector(real=(1.0, 0.0, 0.0), imaginary=(0.0, 1.0, 0.0)),
                    ),
                ),
                phase_turns=0.25,
            ),
        ),
    )

    restored = MagneticStructureModel.from_json(model.to_json())

    assert restored.q_basis_rlu == model.q_basis_rlu
    assert restored.components[0].label == "Helix"
    assert restored.components[0].terms[0].coefficient.to_array()[1] == pytest.approx(1j)


def test_atom_selector_resolves_species_indices_and_symmetry_options() -> None:
    structure = _structure()

    assert resolve_atom_selector(structure, AtomSelector.species("Fe")) == (0,)
    assert resolve_atom_selector(structure, AtomSelector.indices([1, 0, 1])) == (0, 1)

    selectors = viable_atom_selectors(structure, selected_indices=[1])
    assert AtomSelector.all() in selectors
    assert AtomSelector.species("Fe") in selectors
    assert AtomSelector.indices([1]) in selectors


def test_atom_resolved_components_apply_basis_position_phase() -> None:
    structure = _structure()
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="sdw",
                label="SDW",
                kind="spin_density_wave",
                selector=AtomSelector.indices([1]),
                terms=(
                    FourierTerm(
                        q_coeff=(1.0, 0.0, 0.0),
                        coefficient=ComplexVector.from_vector([0.0, 0.0, 2.0]),
                    ),
                ),
            ),
        ),
    )

    components = atom_resolved_fourier_components(structure, model)

    assert len(components) == 1
    assert components[0].atom_fcs[0] == pytest.approx([0.0, 0.0, 0.0])
    assert components[0].atom_fcs[1, 2].real == pytest.approx(0.0, abs=1e-12)
    assert components[0].atom_fcs[1, 2].imag == pytest.approx(-2.0)


def test_muesr_helical_decomposition_reconstructs_elliptical_component() -> None:
    structure = _structure()
    model = MagneticStructureModel(
        q_basis_rlu=((0.123, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="elliptical",
                label="Elliptical",
                kind="custom",
                selector=AtomSelector.all(),
                terms=(
                    FourierTerm(
                        q_coeff=(1.0, 0.0, 0.0),
                        coefficient=ComplexVector(real=(1.0, 0.2, 0.0), imaginary=(0.3, 0.5, 0.1)),
                    ),
                ),
            ),
        ),
    )

    model_data = get_muesr_model_data(structure, model, merge_equivalent_k=False)[0]
    split = split_muesr_model_data_into_helices(model_data, atol=1e-12)

    assert len(split) == 2
    reconstructed = np.sum(np.asarray([part.fc for part in split]), axis=0)
    np.testing.assert_allclose(reconstructed, model_data.fc, atol=1e-11)
    for part in split:
        _assert_muesr_fc_is_circular_helical(part.fc)


def test_muesr_helical_decomposition_preserves_existing_helix() -> None:
    structure = _structure()
    model = MagneticStructureModel(
        q_basis_rlu=((0.123, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="helix",
                label="Helix",
                kind="helix",
                selector=AtomSelector.species("Fe"),
                terms=(
                    FourierTerm(
                        q_coeff=(1.0, 0.0, 0.0),
                        coefficient=ComplexVector(real=(1.0, 0.0, 0.0), imaginary=(0.0, 1.0, 0.0)),
                    ),
                ),
            ),
        ),
    )

    model_data = get_muesr_model_data(structure, model, merge_equivalent_k=False)[0]
    split = split_muesr_model_data_into_helices(model_data, atol=1e-12)

    assert split == (model_data,)
    _assert_muesr_fc_is_circular_helical(split[0].fc)


def test_muesr_helical_decomposition_splits_linear_spin_density_wave() -> None:
    structure = _structure()
    model = MagneticStructureModel(
        q_basis_rlu=((0.123, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="sdw",
                label="SDW",
                kind="spin_density_wave",
                selector=AtomSelector.species("Fe"),
                terms=(
                    FourierTerm(
                        q_coeff=(1.0, 0.0, 0.0),
                        coefficient=ComplexVector.from_vector([1.0, 0.0, 0.0]),
                    ),
                ),
            ),
        ),
    )

    model_data = get_muesr_model_data(structure, model, merge_equivalent_k=False)[0]
    split = split_muesr_model_data_into_helices(model_data, atol=1e-12)

    assert len(split) == 2
    reconstructed = np.sum(np.asarray([part.fc for part in split]), axis=0)
    np.testing.assert_allclose(reconstructed, model_data.fc, atol=1e-11)
    for part in split:
        _assert_muesr_fc_is_circular_helical(part.fc)


def test_species_component_applies_to_every_matching_unit_cell_atom() -> None:
    structure = Structure(
        label="Co",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Co", "O", "Co", "Co"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ],
    )
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="co",
                label="Co moments",
                kind="collinear",
                selector=AtomSelector.species("Co"),
                terms=(
                    FourierTerm(
                        q_coeff=(1.0, 0.0, 0.0),
                        coefficient=ComplexVector.from_vector([0.0, 0.0, 2.6]),
                    ),
                ),
            ),
        ),
    )

    _positions, moments, atom_indices = moments_in_supercell(structure, model)
    unit_moments = {int(atom_index): moments[row] for row, atom_index in enumerate(atom_indices)}

    assert unit_moments[0] == pytest.approx([0.0, 0.0, 2.6])
    assert unit_moments[1] == pytest.approx([0.0, 0.0, 0.0])
    assert unit_moments[2] == pytest.approx([0.0, 0.0, 2.6])
    assert unit_moments[3] == pytest.approx([0.0, 0.0, 2.6])


def test_basis_position_phase_opposes_cof2_cobalt_sites_for_integer_c_axis_q() -> None:
    structure = Structure(
        label="CoF2",
        lattice_vectors=[[4.7, 0.0, 0.0], [0.0, 4.7, 0.0], [0.0, 0.0, 3.2]],
        species=["Co", "Co", "F", "F"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.3, 0.3, 0.0],
            [0.7, 0.7, 0.5],
        ],
    )
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="co",
                label="Co moments",
                kind="collinear",
                selector=AtomSelector.species("Co"),
                terms=(
                    FourierTerm(
                        q_coeff=(1.0, 0.0, 0.0),
                        coefficient=ComplexVector.from_vector([0.0, 0.0, 2.6]),
                    ),
                ),
            ),
        ),
    )

    _positions, moments, atom_indices = moments_in_supercell(structure, model)
    unit_moments = {int(atom_index): moments[row] for row, atom_index in enumerate(atom_indices)}

    assert unit_moments[0] == pytest.approx([0.0, 0.0, 2.6])
    assert unit_moments[1] == pytest.approx([0.0, 0.0, -2.6])
    assert unit_moments[2] == pytest.approx([0.0, 0.0, 0.0])
    assert unit_moments[3] == pytest.approx([0.0, 0.0, 0.0])


def test_preset_component_builders_produce_expected_fourier_coefficients() -> None:
    selector = AtomSelector.species("Co")

    collinear = build_collinear_component(
        label="Collinear",
        selector=selector,
        q_coeff=(1.0, 0.0, 0.0),
        moment=(0.0, 0.0, 2.0),
        component_id="collinear",
    )
    sdw = build_spin_density_wave_component(
        label="SDW",
        selector=selector,
        q_coeff=(1.0, 0.0, 0.0),
        amplitude_vector=(0.0, 2.0, 0.0),
        component_id="sdw",
    )
    helix = build_helix_component(
        label="Helix",
        selector=selector,
        q_coeff=(1.0, 0.0, 0.0),
        real_axis_vector=(2.0, 0.0, 0.0),
        imaginary_axis_vector=(0.0, 2.0, 0.0),
        component_id="helix",
    )
    cycloid = build_cycloid_component(
        label="Cycloid",
        selector=selector,
        q_coeff=(1.0, 0.0, 0.0),
        real_axis_vector=(2.0, 0.0, 0.0),
        imaginary_axis_vector=(0.0, 0.0, 2.0),
        component_id="cycloid",
    )
    conical = build_conical_component(
        label="Conical",
        selector=selector,
        q_coeff=(1.0, 0.0, 0.0),
        total_moment=2.0,
        real_axis_direction=(1.0, 0.0, 0.0),
        imaginary_axis_direction=(0.0, 1.0, 0.0),
        static_axis_direction=(0.0, 0.0, 1.0),
        cone_angle_degrees=60.0,
        component_id="conical",
    )

    assert collinear.kind == "collinear"
    np.testing.assert_allclose(collinear.terms[0].coefficient.to_array(), [0.0, 0.0, 2.0])
    assert sdw.kind == "spin_density_wave"
    np.testing.assert_allclose(sdw.terms[0].coefficient.to_array(), [0.0, 2.0, 0.0])
    assert helix.kind == "helix"
    np.testing.assert_allclose(helix.terms[0].coefficient.to_array(), [2.0, 2.0j, 0.0])
    assert cycloid.kind == "cycloid"
    np.testing.assert_allclose(cycloid.terms[0].coefficient.to_array(), [2.0, 0.0, 2.0j])
    assert conical.kind == "conical"
    assert len(conical.terms) == 2
    np.testing.assert_allclose(conical.terms[0].coefficient.to_array(), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(conical.terms[1].coefficient.to_array(), [np.sqrt(3.0), 1j * np.sqrt(3.0), 0.0])


def test_moment_at_atom_and_supercell_propagate_phase() -> None:
    structure = Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                component_id="afm",
                label="AFM",
                kind="spin_density_wave",
                selector=AtomSelector.all(),
                terms=(FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 2.0])),),
            ),
        ),
    )

    assert moment_at_atom(structure, model, 0, cell_offset=(0.0, 0.0, 0.0)) == pytest.approx([0.0, 0.0, 2.0])
    assert moment_at_atom(structure, model, 0, cell_offset=(0.0, 0.0, 1.0)) == pytest.approx([0.0, 0.0, -2.0])

    _positions, moments, atom_indices = moments_in_supercell(structure, model, (1, 1, 2))
    assert atom_indices.tolist() == [0, 0]
    assert sorted(round(value, 6) for value in moments[:, 2]) == pytest.approx([-2.0, 2.0])


def test_legacy_magnetic_structure_converts_to_components() -> None:
    structure = _structure()
    structure.properties["magnetic_structure"] = {
        "propagation_vector": [0.0, 0.0, 0.25],
        "moments": [
            {"label": "Fe1", "vector": [0.0, 0.0, 1.5]},
            {"label": "O1", "vector": [0.0, 0.0, 0.0]},
        ],
    }

    model = magnetic_model_from_legacy_structure(structure)

    assert model.q_basis_rlu[0] == pytest.approx((0.0, 0.0, 0.25))
    assert len(model.components) == 1
    assert model.components[0].terms[0].q_coeff == pytest.approx((1.0, 0.0, 0.0))
    np.testing.assert_allclose(model.components[0].terms[0].coefficient.to_array().real, [0.0, 0.0, 1.5])


def test_mcif_magnetic_structure_preserves_q_basis_labels_and_unit_cell_moments() -> None:
    structure = _structure()
    structure.properties["magnetic_structure"] = {
        "source": "mcif",
        "atom_labels": ["Fe1", "O1"],
        "propagation_vector": [0.0, 0.0, 0.25],
        "propagation_vectors_rlu": [
            [0.0, 0.0, 0.25],
            [0.5, 0.0, 0.0],
        ],
        "moments": [
            {"label": "O1", "atom_index": 1, "atom_label": "O1", "vector": [0.0, 1.0, 2.0]},
        ],
    }

    model = magnetic_model_from_legacy_structure(structure)

    assert model.q_basis_rlu[0] == pytest.approx((0.0, 0.0, 0.25))
    assert model.q_basis_rlu[1] == pytest.approx((0.5, 0.0, 0.0))
    assert len(model.components) == 1
    component = model.components[0]
    assert component.label == "O1"
    assert component.selector.values == (1,)
    assert component.metadata["source"] == "mcif"
    assert component.metadata["mcif_label"] == "O1"
    assert moment_at_atom(structure, model, 1) == pytest.approx([0.0, 1.0, 2.0])


def test_hexagonal_skyrmion_texture_produces_three_helical_terms() -> None:
    texture = build_hexagonal_skyrmion_texture(
        magnitude=3.0,
        q_directions=((1.0, 0.0, 0.0), (-0.5, np.sqrt(3.0) / 2.0, 0.0), (-0.5, -np.sqrt(3.0) / 2.0, 0.0)),
        core_direction=(0.0, 0.0, -1.0),
        skyrmion_type="bloch",
    )

    assert len(texture.terms) == 3
    assert {term.q_coeff for term in texture.terms} == {
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, -1.0, 0.0),
    }
    first = texture.terms[0].coefficient.to_array()
    np.testing.assert_allclose(first.real, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(first.imag, [0.0, -1.0, 0.0])


def test_hexagonal_skyrmion_helicity_matches_mag2spectra_convention() -> None:
    q_basis_sky = (
        (0.05, 0.0, 0.0),
        (-0.5 * 0.05, -0.5 * np.sqrt(3.0) * 0.05, 0.0),
        (0.0, 0.0, 1.0),
    )
    sample_r = np.array((3.0, 0.0, 0.0), dtype=float)
    q_dots = np.asarray(q_basis_sky, dtype=float) @ sample_r
    plus = build_hexagonal_skyrmion_texture(
        magnitude=1.0,
        core_direction=(0.0, 0.0, -1.0),
        skyrmion_type="bloch",
        helicity=1,
    )
    minus = build_hexagonal_skyrmion_texture(
        magnitude=1.0,
        core_direction=(0.0, 0.0, -1.0),
        skyrmion_type="bloch",
        helicity=-1,
    )

    moment_plus = evaluate_texture_moment(plus, q_dots)
    moment_minus = evaluate_texture_moment(minus, q_dots)

    np.testing.assert_allclose(moment_plus[2], moment_minus[2], atol=1e-12)
    np.testing.assert_allclose(moment_plus[:2], -moment_minus[:2], atol=1e-12)


def test_square_skyrmion_texture_adds_optional_background() -> None:
    texture = build_square_skyrmion_texture(
        magnitude=2.0,
        q_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        core_direction=(0.0, 0.0, -1.0),
        cone_angle_degrees=90.0,
        ferromagnetic_background=0.25,
    )

    assert len(texture.terms) == 3
    background = next(term for term in texture.terms if term.q_coeff == (0.0, 0.0, 0.0))
    np.testing.assert_allclose(background.coefficient.to_array(), [0.0, 0.0, -0.25])


def test_skyrmion_q_basis_rotates_in_cartesian_reciprocal_space() -> None:
    structure = _structure()

    q_basis = skyrmion_q_basis_rlu(structure, q1_rlu=(0.1, 0.0, 0.0), core_direction=(0.0, 0.0, 1.0), lattice="square")

    np.testing.assert_allclose(q_basis[0], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(q_basis[1], [0.0, 0.1, 0.0], atol=1e-12)


def test_texture_component_maps_local_coefficients_to_global_basis_rows() -> None:
    texture = build_square_skyrmion_texture(
        magnitude=2.0,
        q_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    component = build_texture_component(
        label="Square",
        selector=AtomSelector.species("Fe"),
        texture=texture,
        q_coeff_transform=((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        metadata={"type": "texture"},
    )

    assert component.kind == "texture"
    assert component.terms[0].q_coeff == pytest.approx((0.0, 1.0, 0.0))
    assert component.terms[1].q_coeff == pytest.approx((0.0, 0.0, 1.0))
    assert component.metadata["type"] == "texture"
    restored = MagneticComponent.from_json(component.to_json())
    assert restored.metadata["type"] == "texture"


def test_texture_normalization_reduces_magnitude_variation_and_pruning_keeps_largest_terms() -> None:
    texture = build_square_skyrmion_texture(
        magnitude=1.0,
        q_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        skyrmion_type="bloch",
    )
    before = texture_diagnostics(texture, (16, 16, 1))
    normalized = normalize_texture_magnitude(texture, grid_resolution=(16, 16, 1), threshold=1e-3)
    after = texture_diagnostics(normalized, (16, 16, 1))

    assert after.magnitude_std < before.magnitude_std
    assert len(normalized.terms) > len(texture.terms)

    pruned = prune_texture_terms(normalized, top_n=2)
    assert len(pruned.terms) == 2
    magnitudes = [np.linalg.norm(term.coefficient.to_array()) for term in pruned.terms]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_evaluate_texture_moment_uses_negative_exponential_convention() -> None:
    texture = build_square_skyrmion_texture(
        magnitude=2.0,
        q_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        skyrmion_type="neel",
    )

    moment = evaluate_texture_moment(texture, (0.25, 0.0, 0.0))

    assert moment.shape == (3,)
    assert np.linalg.norm(moment) > 0.0
