"""Additional coverage for propagation-vector magnetic-structure helpers.

These tests target currently-uncovered JSON parsing fallbacks, error
branches, propagation-vector / moment helpers, and the helical decomposition
paths in :mod:`mufinder.science.magnetic_structure`. They are deterministic and
do not touch the filesystem, network, or any GUI layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.magnetic_structure import (
    MAGNETIC_STRUCTURE_SCHEMA_VERSION,
    AtomResolvedFourierComponent,
    AtomSelector,
    ComplexVector,
    FourierTerm,
    MagneticComponent,
    MagneticStructureError,
    MagneticStructureModel,
    MagneticTexture,
    MagneticTextureTerm,
    MuesrModelData,
    atom_resolved_fourier_components,
    build_conical_component,
    build_hexagonal_skyrmion_texture,
    build_square_skyrmion_texture,
    build_texture_component,
    evaluate_texture_grid,
    evaluate_texture_moment,
    get_muesr_model_data,
    magnetic_model_from_legacy_structure,
    moment_at_atom,
    moments_in_supercell,
    normalize_texture_magnitude,
    prune_texture_terms,
    resolve_atom_selector,
    skyrmion_q_basis_rlu,
    split_muesr_model_data_into_helices,
    texture_diagnostics,
    viable_atom_selectors,
)
from mufinder.science.magnetic_structure import (
    _orthogonal_unit_vector,
    _rotate_vector,
    _texture_grid_resolution,
)


def _structure() -> Structure:
    return Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


# --------------------------------------------------------------------------
# ComplexVector
# --------------------------------------------------------------------------


def test_complex_vector_from_json_mapping_and_sequence() -> None:
    from_map = ComplexVector.from_json({"real": [1.0, 2.0, 3.0], "imaginary": [0.0, 1.0, 0.0]})
    assert from_map.real == (1.0, 2.0, 3.0)
    assert from_map.imaginary == (0.0, 1.0, 0.0)

    # Non-mapping payload falls back to from_vector and discards imag of reals.
    from_seq = ComplexVector.from_json([1.0 + 2.0j, 0.0, 0.0])
    assert from_seq.real == (1.0, 0.0, 0.0)
    assert from_seq.imaginary == (2.0, 0.0, 0.0)


def test_complex_vector_from_json_defaults_when_keys_missing() -> None:
    cv = ComplexVector.from_json({})
    assert cv.real == (0.0, 0.0, 0.0)
    assert cv.imaginary == (0.0, 0.0, 0.0)


def test_complex_vector_to_array_and_to_json_round_trip() -> None:
    cv = ComplexVector(real=(1.0, 0.0, -1.0), imaginary=(0.0, 2.0, 0.0))
    arr = cv.to_array()
    np.testing.assert_allclose(arr, [1.0, 2.0j, -1.0])
    payload = cv.to_json()
    assert payload == {"real": [1.0, 0.0, -1.0], "imaginary": [0.0, 2.0, 0.0]}


def test_complex_vector_from_vector_rejects_wrong_shape() -> None:
    with pytest.raises(MagneticStructureError, match="complex three-vector"):
        ComplexVector.from_vector([1.0, 2.0])


# --------------------------------------------------------------------------
# FourierTerm / MagneticTextureTerm / MagneticTexture
# --------------------------------------------------------------------------


def test_fourier_term_from_and_to_json() -> None:
    term = FourierTerm.from_json(
        {"q_coeff": [1.0, 0.0, 0.0], "coefficient": {"real": [0.0, 0.0, 2.0]}}
    )
    assert term.q_coeff == (1.0, 0.0, 0.0)
    payload = term.to_json()
    assert payload["q_coeff"] == [1.0, 0.0, 0.0]
    assert payload["coefficient"]["real"] == [0.0, 0.0, 2.0]


def test_fourier_term_from_json_uses_defaults() -> None:
    term = FourierTerm.from_json({})
    assert term.q_coeff == (0.0, 0.0, 0.0)
    np.testing.assert_allclose(term.coefficient.to_array(), [0.0, 0.0, 0.0])


def test_texture_term_from_vector_json_and_to_fourier_term() -> None:
    term = MagneticTextureTerm.from_vector((1.0, 0.0, 0.0), [1.0, 1j, 0.0])
    assert term.q_coeff == (1.0, 0.0, 0.0)

    restored = MagneticTextureTerm.from_json(term.to_json())
    assert restored.q_coeff == (1.0, 0.0, 0.0)
    np.testing.assert_allclose(restored.coefficient.to_array(), [1.0, 1j, 0.0])

    # to_fourier_term without override keeps q_coeff; with override replaces it.
    same = term.to_fourier_term()
    assert same.q_coeff == (1.0, 0.0, 0.0)
    moved = term.to_fourier_term((0.0, 2.0, 0.0))
    assert moved.q_coeff == (0.0, 2.0, 0.0)


def test_magnetic_texture_from_json_non_mapping_returns_empty() -> None:
    assert MagneticTexture.from_json("not-a-mapping").terms == ()
    assert MagneticTexture.from_json(None).terms == ()


def test_magnetic_texture_from_json_skips_non_mapping_terms() -> None:
    texture = MagneticTexture.from_json(
        {"terms": [{"q_coeff": [1.0, 0.0, 0.0]}, "garbage", 42]}
    )
    assert len(texture.terms) == 1
    assert texture.to_json()["terms"][0]["q_coeff"] == [1.0, 0.0, 0.0]


def test_combine_terms_sums_duplicates_and_drops_cancelling_terms() -> None:
    texture = MagneticTexture(
        terms=(
            MagneticTextureTerm.from_vector((1.0, 0.0, 0.0), [1.0, 0.0, 0.0]),
            MagneticTextureTerm.from_vector((1.0, 0.0, 0.0), [2.0, 0.0, 0.0]),
            MagneticTextureTerm.from_vector((0.0, 1.0, 0.0), [1.0, 0.0, 0.0]),
            MagneticTextureTerm.from_vector((0.0, 1.0, 0.0), [-1.0, 0.0, 0.0]),
        )
    )
    combined = texture.combine_terms()
    # The (0,1,0) terms cancel and are dropped; the (1,0,0) pair sums to 3.
    assert len(combined.terms) == 1
    assert combined.terms[0].q_coeff == (1.0, 0.0, 0.0)
    np.testing.assert_allclose(combined.terms[0].coefficient.to_array(), [3.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# AtomSelector
# --------------------------------------------------------------------------


def test_atom_selector_from_json_variants() -> None:
    # Non-mapping -> all().
    assert AtomSelector.from_json("nope") == AtomSelector.all()
    # Sequence values preserved.
    sel = AtomSelector.from_json({"kind": "species", "values": ["Fe", "O"]})
    assert sel.kind == "species"
    assert sel.values == ("Fe", "O")
    # Empty / None scalar values -> empty tuple.
    sel_empty = AtomSelector.from_json({"kind": "all", "values": None})
    assert sel_empty.values == ()
    sel_blank = AtomSelector.from_json({"kind": "all", "values": ""})
    assert sel_blank.values == ()
    # Scalar non-empty value -> wrapped as single string.
    sel_scalar = AtomSelector.from_json({"kind": "symmetry_equivalent", "values": 3})
    assert sel_scalar.values == ("3",)
    # Blank kind falls back to "all".
    sel_kind = AtomSelector.from_json({"kind": "   "})
    assert sel_kind.kind == "all"


def test_atom_selector_to_json_round_trip() -> None:
    sel = AtomSelector.symmetry_equivalent(2)
    payload = sel.to_json()
    assert payload == {"kind": "symmetry_equivalent", "values": [2]}


# --------------------------------------------------------------------------
# resolve_atom_selector error paths
# --------------------------------------------------------------------------


def test_resolve_atom_selector_all_and_case_insensitive_kind() -> None:
    structure = _structure()
    assert resolve_atom_selector(structure, AtomSelector("ALL", ())) == (0, 1)


def test_resolve_atom_selector_indices_out_of_range_raises() -> None:
    structure = _structure()
    with pytest.raises(MagneticStructureError, match="out-of-range index"):
        resolve_atom_selector(structure, AtomSelector.indices([5]))


def test_resolve_atom_selector_symmetry_requires_representative() -> None:
    structure = _structure()
    with pytest.raises(MagneticStructureError, match="requires a representative"):
        resolve_atom_selector(structure, AtomSelector("symmetry_equivalent", ()))


def test_resolve_atom_selector_symmetry_out_of_range_raises() -> None:
    structure = _structure()
    with pytest.raises(MagneticStructureError, match="could not be resolved"):
        resolve_atom_selector(structure, AtomSelector("symmetry_equivalent", (99,)))


def test_resolve_atom_selector_unknown_kind_raises() -> None:
    structure = _structure()
    with pytest.raises(MagneticStructureError, match="unknown atom selector kind"):
        resolve_atom_selector(structure, AtomSelector("mystery", ()))


def test_resolve_symmetry_equivalent_groups_atoms() -> None:
    structure = Structure(
        label="Fe2",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe", "Fe"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    resolved = resolve_atom_selector(structure, AtomSelector.symmetry_equivalent(0))
    assert 0 in resolved


# --------------------------------------------------------------------------
# viable_atom_selectors filters invalid indices
# --------------------------------------------------------------------------


def test_viable_atom_selectors_drops_out_of_range_selection() -> None:
    structure = _structure()
    selectors = viable_atom_selectors(structure, selected_indices=[0, 99, -1])
    index_selectors = [s for s in selectors if s.kind == "indices"]
    assert index_selectors == [AtomSelector.indices([0])]


def test_viable_atom_selectors_no_selection_has_no_index_selector() -> None:
    structure = _structure()
    selectors = viable_atom_selectors(structure)
    assert all(s.kind != "indices" for s in selectors)


# --------------------------------------------------------------------------
# MagneticComponent JSON
# --------------------------------------------------------------------------


def test_magnetic_component_from_json_handles_defaults_and_bad_metadata() -> None:
    component = MagneticComponent.from_json(
        {
            "component_id": "  c1  ",
            "terms": [
                {"q_coeff": [1.0, 0.0, 0.0], "coefficient": {"real": [0.0, 0.0, 1.0]}},
                "not-a-term",
            ],
            "phase_turns": None,
            "metadata": "not-a-mapping",
        }
    )
    assert component.component_id == "c1"
    assert component.kind == "custom"  # blank/default kind
    assert component.phase_turns == 0.0
    assert component.metadata == {}
    assert len(component.terms) == 1


# --------------------------------------------------------------------------
# MagneticStructureModel JSON / helpers
# --------------------------------------------------------------------------


def test_model_from_json_non_mapping_returns_default() -> None:
    model = MagneticStructureModel.from_json("nope")
    assert model.components == ()
    assert model.schema_version == MAGNETIC_STRUCTURE_SCHEMA_VERSION


def test_model_from_json_drops_components_without_id_or_terms() -> None:
    payload = {
        "q_basis_rlu": [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "components": [
            {"component_id": "", "terms": [{"q_coeff": [1.0, 0.0, 0.0]}]},
            {"component_id": "has-id", "terms": []},
            {
                "component_id": "keep",
                "terms": [
                    {"q_coeff": [1.0, 0.0, 0.0], "coefficient": {"real": [0.0, 0.0, 1.0]}}
                ],
            },
            "garbage",
        ],
        "metadata": "not-a-mapping",
        "global_phase_turns": None,
    }
    model = MagneticStructureModel.from_json(payload)
    assert tuple(c.component_id for c in model.components) == ("keep",)
    assert model.metadata == {}
    assert model.global_phase_turns == 0.0


def test_model_from_json_supports_legacy_q_basis_key() -> None:
    model = MagneticStructureModel.from_json(
        {"q_basis": [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}
    )
    assert model.q_basis_rlu[0] == (0.5, 0.0, 0.0)


def test_model_q_basis_array_and_q_coeff_to_rlu() -> None:
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.5), (0.25, 0.0, 0.0), (0.0, 0.0, 0.0))
    )
    arr = model.q_basis_array
    assert arr.shape == (3, 3)
    rlu = model.q_coeff_to_rlu((1.0, 1.0, 0.0))
    np.testing.assert_allclose(rlu, [0.25, 0.0, 0.5])


def test_model_to_json_round_trip_preserves_schema_version() -> None:
    model = MagneticStructureModel(schema_version=7, global_phase_turns=0.5)
    payload = model.to_json()
    assert payload["schema_version"] == 7
    assert payload["global_phase_turns"] == 0.5
    assert MagneticStructureModel.from_json(payload).schema_version == 7


# --------------------------------------------------------------------------
# atom_resolved_fourier_components branches
# --------------------------------------------------------------------------


def test_atom_resolved_components_bad_positions_shape_raises() -> None:
    structure = Structure(
        label="bad",
        lattice_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        species=["Fe", "O"],
        fractional_positions=[[0.0, 0.0, 0.0]],  # only one position for two atoms
    )
    model = MagneticStructureModel()
    with pytest.raises(MagneticStructureError, match="fractional positions must have shape"):
        atom_resolved_fourier_components(structure, model)


def test_atom_resolved_components_skips_disabled_and_filters_ids() -> None:
    structure = _structure()
    term = FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 1.0]))
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent("on", "On", "collinear", AtomSelector.all(), (term,), enabled=True),
            MagneticComponent("off", "Off", "collinear", AtomSelector.all(), (term,), enabled=False),
            MagneticComponent("other", "Other", "collinear", AtomSelector.all(), (term,)),
        ),
    )
    # disabled "off" excluded; component_ids filter keeps only "on".
    resolved = atom_resolved_fourier_components(structure, model, component_ids=["on"])
    assert {c.component_id for c in resolved} == {"on"}


def test_atom_resolved_components_skips_empty_selection() -> None:
    structure = _structure()
    term = FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 1.0]))
    model = MagneticStructureModel(
        components=(
            MagneticComponent(
                "none", "None", "collinear", AtomSelector.species("Xx"), (term,)
            ),
        ),
    )
    assert atom_resolved_fourier_components(structure, model) == ()


# --------------------------------------------------------------------------
# moment_at_atom error path
# --------------------------------------------------------------------------


def test_moment_at_atom_out_of_range_raises() -> None:
    structure = _structure()
    model = MagneticStructureModel()
    with pytest.raises(MagneticStructureError, match="atom_index is out of range"):
        moment_at_atom(structure, model, 99)
    with pytest.raises(MagneticStructureError, match="atom_index is out of range"):
        moment_at_atom(structure, model, -1)


def test_moment_at_atom_zero_when_no_components() -> None:
    structure = _structure()
    model = MagneticStructureModel()
    np.testing.assert_allclose(moment_at_atom(structure, model, 0), [0.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# moments_in_supercell error / repeat helper
# --------------------------------------------------------------------------


def test_moments_in_supercell_bad_repeat_count_raises() -> None:
    structure = _structure()
    model = MagneticStructureModel()
    with pytest.raises(MagneticStructureError, match="three repeat counts"):
        moments_in_supercell(structure, model, (1, 1))
    with pytest.raises(MagneticStructureError, match="must be positive"):
        moments_in_supercell(structure, model, (1, 0, 1))


def test_moments_in_supercell_shapes() -> None:
    structure = _structure()
    model = MagneticStructureModel()
    positions, moments, atom_indices = moments_in_supercell(structure, model, (2, 1, 1))
    assert positions.shape == (4, 3)
    assert moments.shape == (4, 3)
    assert atom_indices.tolist() == [0, 1, 0, 1]


# --------------------------------------------------------------------------
# get_muesr_model_data: merging and zero filtering
# --------------------------------------------------------------------------


def test_get_muesr_model_data_merges_equal_k_and_skips_zero() -> None:
    structure = _structure()
    term_a = FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 1.0]))
    term_b = FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 1.0, 0.0]))
    zero_term = FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 0.0]))
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent("a", "A", "collinear", AtomSelector.indices([0]), (term_a,)),
            MagneticComponent("b", "B", "collinear", AtomSelector.indices([0]), (term_b,)),
            MagneticComponent("z", "Z", "collinear", AtomSelector.indices([0]), (zero_term,)),
        ),
    )
    merged = get_muesr_model_data(structure, model, merge_equivalent_k=True)
    assert len(merged) == 1
    assert merged[0].component_ids == ("a", "b")

    unmerged = get_muesr_model_data(structure, model, merge_equivalent_k=False)
    assert len(unmerged) == 2


# --------------------------------------------------------------------------
# split_muesr_model_data_into_helices: error branches
# --------------------------------------------------------------------------


def test_split_helices_bad_fc_shape_raises() -> None:
    data = MuesrModelData(
        k_rlu=np.zeros(3),
        fc=np.zeros((2, 2), dtype=complex),
        phi=np.zeros(2),
        component_ids=("c",),
        q_coeffs=((1.0, 0.0, 0.0),),
    )
    with pytest.raises(MagneticStructureError, match=r"shape \(n_atoms, 3\)"):
        split_muesr_model_data_into_helices(data)


def test_split_helices_nonpositive_atol_raises() -> None:
    structure = _structure()
    model = MagneticStructureModel(
        q_basis_rlu=((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent(
                "lin", "Lin", "spin_density_wave", AtomSelector.all(),
                (FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([1.0, 0.0, 0.0])),),
            ),
        ),
    )
    data = get_muesr_model_data(structure, model, merge_equivalent_k=False)[0]
    with pytest.raises(MagneticStructureError, match="tolerance must be positive"):
        split_muesr_model_data_into_helices(data, atol=0.0)


def test_split_helices_zero_fc_returns_single_zero_helix() -> None:
    data = MuesrModelData(
        k_rlu=np.zeros(3),
        fc=np.zeros((2, 3), dtype=complex),
        phi=np.zeros(2),
        component_ids=("c",),
        q_coeffs=((1.0, 0.0, 0.0),),
    )
    split = split_muesr_model_data_into_helices(data)
    assert len(split) == 1
    np.testing.assert_allclose(split[0].fc, np.zeros((2, 3)))


# --------------------------------------------------------------------------
# _orthogonal_unit_vector / _rotate_vector
# --------------------------------------------------------------------------


def test_orthogonal_unit_vector_is_orthonormal() -> None:
    v = np.array([0.0, 0.0, 1.0])
    ortho = _orthogonal_unit_vector(v)
    assert float(np.linalg.norm(ortho)) == pytest.approx(1.0)
    assert float(np.dot(ortho, v)) == pytest.approx(0.0, abs=1e-12)


def test_orthogonal_unit_vector_zero_raises() -> None:
    with pytest.raises(MagneticStructureError, match="orthogonal vector to a zero"):
        _orthogonal_unit_vector(np.zeros(3))


def test_rotate_vector_rotates_in_plane() -> None:
    rotated = _rotate_vector((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), np.pi / 2.0)
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)


# --------------------------------------------------------------------------
# Conical builder error branches
# --------------------------------------------------------------------------


def test_build_conical_component_pure_helix_drops_static_term() -> None:
    component = build_conical_component(
        label="",
        selector=AtomSelector.all(),
        q_coeff=(1.0, 0.0, 0.0),
        total_moment=2.0,
        real_axis_direction=(1.0, 0.0, 0.0),
        imaginary_axis_direction=(0.0, 1.0, 0.0),
        static_axis_direction=(0.0, 0.0, 1.0),
        cone_angle_degrees=90.0,
    )
    # 90-degree cone -> static (cos) component is zero but still stored as a term.
    assert component.label == "Conical"
    static_term = component.terms[0]
    np.testing.assert_allclose(static_term.coefficient.to_array(), [0.0, 0.0, 0.0], atol=1e-12)


def test_build_conical_component_zero_axis_direction_raises() -> None:
    with pytest.raises(MagneticStructureError, match="must be non-zero"):
        build_conical_component(
            label="C",
            selector=AtomSelector.all(),
            q_coeff=(1.0, 0.0, 0.0),
            total_moment=2.0,
            real_axis_direction=(0.0, 0.0, 0.0),
            imaginary_axis_direction=(0.0, 1.0, 0.0),
            static_axis_direction=(0.0, 0.0, 1.0),
            cone_angle_degrees=45.0,
        )


# --------------------------------------------------------------------------
# build_texture_component error branches
# --------------------------------------------------------------------------


def test_build_texture_component_requires_terms() -> None:
    with pytest.raises(MagneticStructureError, match="at least one Fourier term"):
        build_texture_component(
            label="X", selector=AtomSelector.all(), texture=MagneticTexture()
        )


def test_build_texture_component_bad_transform_shape() -> None:
    texture = MagneticTexture(
        terms=(MagneticTextureTerm.from_vector((1.0, 0.0, 0.0), [0.0, 0.0, 1.0]),)
    )
    with pytest.raises(MagneticStructureError, match=r"shape \(3, 3\)"):
        build_texture_component(
            label="X",
            selector=AtomSelector.all(),
            texture=texture,
            q_coeff_transform=((1.0, 0.0), (0.0, 1.0)),
        )


def test_build_texture_component_default_label_and_identity_transform() -> None:
    texture = MagneticTexture(
        terms=(MagneticTextureTerm.from_vector((1.0, 0.0, 0.0), [0.0, 0.0, 1.0]),)
    )
    component = build_texture_component(
        label="   ", selector=AtomSelector.all(), texture=texture
    )
    assert component.label == "Multi-q texture"
    assert component.terms[0].q_coeff == (1.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# Skyrmion texture builders: error branches
# --------------------------------------------------------------------------


def test_hexagonal_skyrmion_requires_three_q_directions() -> None:
    with pytest.raises(MagneticStructureError, match="three q directions"):
        build_hexagonal_skyrmion_texture(
            magnitude=1.0, q_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        )


def test_square_skyrmion_requires_two_q_directions() -> None:
    with pytest.raises(MagneticStructureError, match="two q directions"):
        build_square_skyrmion_texture(magnitude=1.0, q_directions=((1.0, 0.0, 0.0),))


def test_hexagonal_skyrmion_default_directions_generate_three_terms() -> None:
    texture = build_hexagonal_skyrmion_texture(magnitude=1.0)
    assert len(texture.terms) == 3


def test_skyrmion_invalid_helicity_raises() -> None:
    with pytest.raises(MagneticStructureError, match=r"helicity must be \+1 or -1"):
        build_square_skyrmion_texture(
            magnitude=1.0,
            q_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            helicity=2,
        )


def test_skyrmion_invalid_type_raises() -> None:
    with pytest.raises(MagneticStructureError, match="must be 'bloch' or 'neel'"):
        build_square_skyrmion_texture(
            magnitude=1.0,
            q_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            skyrmion_type="spiral",
        )


def test_skyrmion_q_direction_not_perpendicular_raises() -> None:
    with pytest.raises(MagneticStructureError, match="perpendicular to the core"):
        build_square_skyrmion_texture(
            magnitude=1.0,
            q_directions=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
            core_direction=(0.0, 0.0, 1.0),
        )


def test_square_skyrmion_neel_type_builds() -> None:
    texture = build_square_skyrmion_texture(
        magnitude=1.0,
        q_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        skyrmion_type="neel",
    )
    assert len(texture.terms) == 2


# --------------------------------------------------------------------------
# skyrmion_q_basis_rlu error branches
# --------------------------------------------------------------------------


def test_skyrmion_q_basis_zero_q1_raises() -> None:
    structure = _structure()
    with pytest.raises(MagneticStructureError, match="must be non-zero"):
        skyrmion_q_basis_rlu(
            structure, q1_rlu=(0.0, 0.0, 0.0), core_direction=(0.0, 0.0, 1.0), lattice="square"
        )


def test_skyrmion_q_basis_q1_not_in_plane_raises() -> None:
    structure = _structure()
    with pytest.raises(MagneticStructureError, match="perpendicular to the skyrmion core"):
        skyrmion_q_basis_rlu(
            structure, q1_rlu=(0.0, 0.0, 0.1), core_direction=(0.0, 0.0, 1.0), lattice="square"
        )


def test_skyrmion_q_basis_hexagonal_lattice_rotation() -> None:
    structure = _structure()
    q_basis = skyrmion_q_basis_rlu(
        structure, q1_rlu=(0.1, 0.0, 0.0), core_direction=(0.0, 0.0, 1.0), lattice="hexagonal"
    )
    # 120-degree rotation in cubic reciprocal space.
    np.testing.assert_allclose(q_basis[0], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(q_basis[1], [-0.05, np.sqrt(3.0) / 2.0 * 0.1, 0.0], atol=1e-12)
    assert q_basis[2] == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# evaluate_texture_grid / texture_diagnostics / normalize / prune
# --------------------------------------------------------------------------


def test_evaluate_texture_grid_shape() -> None:
    texture = MagneticTexture(
        terms=(MagneticTextureTerm.from_vector((1.0, 0.0, 0.0), [1.0, 0.0, 0.0]),)
    )
    grid = evaluate_texture_grid(texture, (4, 4, 1))
    assert grid.shape == (4, 4, 1, 3)


def test_texture_diagnostics_empty_texture_returns_zeros() -> None:
    diag = texture_diagnostics(MagneticTexture())
    assert diag.term_count == 0
    assert diag.magnitude_max == 0.0


def test_texture_grid_resolution_pads_and_clamps() -> None:
    assert _texture_grid_resolution((5,)) == (5, 1, 1)
    assert _texture_grid_resolution((0, -3, 7)) == (1, 1, 7)


def test_normalize_texture_empty_returns_input() -> None:
    empty = MagneticTexture()
    assert normalize_texture_magnitude(empty) is empty


def test_normalize_texture_with_target_magnitude_scales() -> None:
    texture = build_square_skyrmion_texture(
        magnitude=1.0,
        q_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    scaled = normalize_texture_magnitude(
        texture, grid_resolution=(8, 8, 1), threshold=1e-2, target_magnitude=2.0
    )
    unscaled = normalize_texture_magnitude(
        texture, grid_resolution=(8, 8, 1), threshold=1e-2, target_magnitude=1.0
    )
    scaled_max = texture_diagnostics(scaled, (8, 8, 1)).magnitude_mean
    unscaled_max = texture_diagnostics(unscaled, (8, 8, 1)).magnitude_mean
    assert scaled_max == pytest.approx(2.0 * unscaled_max, rel=1e-6)


def test_prune_texture_terms_by_threshold() -> None:
    texture = MagneticTexture(
        terms=(
            MagneticTextureTerm.from_vector((1.0, 0.0, 0.0), [2.0, 0.0, 0.0]),
            MagneticTextureTerm.from_vector((0.0, 1.0, 0.0), [0.01, 0.0, 0.0]),
        )
    )
    pruned = prune_texture_terms(texture, threshold=0.5)
    assert len(pruned.terms) == 1
    assert pruned.terms[0].q_coeff == (1.0, 0.0, 0.0)


def test_prune_texture_terms_no_args_combines() -> None:
    texture = MagneticTexture(
        terms=(
            MagneticTextureTerm.from_vector((1.0, 0.0, 0.0), [1.0, 0.0, 0.0]),
            MagneticTextureTerm.from_vector((1.0, 0.0, 0.0), [1.0, 0.0, 0.0]),
        )
    )
    pruned = prune_texture_terms(texture)
    assert len(pruned.terms) == 1
    np.testing.assert_allclose(pruned.terms[0].coefficient.to_array(), [2.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# Legacy conversion paths
# --------------------------------------------------------------------------


def test_legacy_model_from_atomic_spins_scalar() -> None:
    structure = _structure()
    structure.properties["atomic_spins"] = [1.5, -2.0]
    model = magnetic_model_from_legacy_structure(structure)
    # Two non-zero scalar spins -> two collinear components along z.
    assert len(model.components) == 2
    np.testing.assert_allclose(
        model.components[0].terms[0].coefficient.to_array().real, [0.0, 0.0, 1.5]
    )
    np.testing.assert_allclose(
        model.components[1].terms[0].coefficient.to_array().real, [0.0, 0.0, -2.0]
    )
    assert model.metadata["source"] == "legacy_host_magnetic_structure"


def test_legacy_model_no_magnetism_yields_no_components() -> None:
    structure = _structure()
    model = magnetic_model_from_legacy_structure(structure)
    assert model.components == ()
    # Default q basis is all zero.
    assert model.q_basis_rlu == (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )


def test_legacy_model_uses_top_level_propagation_vector() -> None:
    structure = _structure()
    structure.properties["propagation_vector"] = [0.0, 0.0, 0.5]
    structure.properties["atomic_spins"] = [1.0, 0.0]
    model = magnetic_model_from_legacy_structure(structure)
    assert model.q_basis_rlu[0] == pytest.approx((0.0, 0.0, 0.5))
    # Non-zero propagation -> q_coeff (1,0,0).
    assert model.components[0].terms[0].q_coeff == (1.0, 0.0, 0.0)


def test_legacy_model_skips_zero_moment_entries() -> None:
    structure = _structure()
    structure.properties["initial_magnetic_moments"] = [0.0, 0.0]
    model = magnetic_model_from_legacy_structure(structure)
    assert model.components == ()


def test_legacy_model_handles_bad_propagation_vector_gracefully() -> None:
    structure = _structure()
    structure.properties["magnetic_structure"] = {
        "propagation_vector": "not-a-vector",
        "moments": [{"label": "Fe1", "vector": [0.0, 0.0, 1.0]}],
    }
    model = magnetic_model_from_legacy_structure(structure)
    # Bad propagation vector falls back to zero -> q_coeff stays (0,0,0).
    assert model.q_basis_rlu[0] == (0.0, 0.0, 0.0)
    assert model.components[0].terms[0].q_coeff == (0.0, 0.0, 0.0)


def test_legacy_moment_entry_with_out_of_range_atom_index_is_skipped() -> None:
    structure = _structure()
    structure.properties["magnetic_structure"] = {
        "moments": [
            {"label": "ghost", "atom_index": 99, "vector": [1.0, 0.0, 0.0]},
            {"label": "Fe1", "atom_index": 0, "vector": [0.0, 0.0, 1.0]},
        ],
    }
    model = magnetic_model_from_legacy_structure(structure)
    assert len(model.components) == 1
    assert model.components[0].selector.values == (0,)


# --------------------------------------------------------------------------
# AtomResolvedFourierComponent dataclass is constructible (smoke)
# --------------------------------------------------------------------------


def test_atom_resolved_fourier_component_dataclass() -> None:
    comp = AtomResolvedFourierComponent(
        component_id="c",
        q_coeff=(1.0, 0.0, 0.0),
        q_rlu=np.zeros(3),
        atom_fcs=np.zeros((1, 3), dtype=complex),
    )
    assert comp.component_id == "c"


# --------------------------------------------------------------------------
# evaluate_texture_moment empty
# --------------------------------------------------------------------------


def test_evaluate_texture_moment_empty_texture_is_zero() -> None:
    moment = evaluate_texture_moment(MagneticTexture(), (0.1, 0.2, 0.3))
    np.testing.assert_allclose(moment, [0.0, 0.0, 0.0])
