"""Tests for propagation-vector magnetic structure helpers."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.magnetic_structure import (
    CHECKLIST_DONE,
    CHECKLIST_PROBLEM,
    CHECKLIST_TODO,
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
    allocate_q_basis_row,
    allocate_texture_q_basis_rows,
    atom_resolved_fourier_components,
    component_propagation_coeff,
    component_propagation_rlu,
    conical_component_geometry,
    constrained_secondary_direction,
    evaluate_texture_preview_grid,
    magnetic_model_for_host,
    nonzero_unit_direction,
    perpendicular_unit_direction,
    propagation_direction,
    reference_direction_perpendicular_to,
    rotation_handedness,
    tiled_supercell_structure,
    unit_direction_or,
    unit_q_coeff,
    used_q_basis_indices,
    build_collinear_component,
    build_conical_component,
    build_cycloid_component,
    build_helix_component,
    build_hexagonal_skyrmion_texture,
    build_magnetic_preset_component,
    build_square_skyrmion_texture,
    build_spin_density_wave_component,
    build_texture_component,
    component_q_basis_indices,
    crystal_axis_direction,
    default_preset_reference_axis,
    duplicate_magnetic_components,
    enabled_magnetic_components,
    evaluate_texture_grid,
    evaluate_texture_moment,
    get_muesr_model_data,
    atom_phases_for_selection,
    is_texture_component,
    magnetic_builder_checklist,
    magnetic_model_from_host_spins,
    magnetic_period_along_axes,
    magnetic_structure_to_ase_atoms,
    moment_at_atom,
    moments_in_supercell,
    move_magnetic_components,
    normalize_texture_magnitude,
    prune_texture_terms,
    q_basis_combination_rlu,
    q_basis_placeholder,
    q_basis_slots,
    q_cartesian_direction,
    q_coeff_expression,
    reciprocal_axis_length,
    reciprocal_axis_q_rlu,
    skyrmion_q_basis_rlu,
    split_muesr_model_data_into_helices,
    texture_diagnostics,
    topological_charge_magnetic_unit_cell,
    topological_density_2d,
    resolve_atom_selector,
    set_magnetic_components_enabled,
    set_q_basis_vector,
    viable_atom_selectors,
    wavevector_rlu_from_inverse_angstrom,
)
from mufinder.science.magnetic_structure import _orthogonal_unit_vector
from tests.support.structures import host_structure


_structure = partial(host_structure, label="Fe", lattice=3.0, species=("Fe", "O"))


def _assert_muesr_fc_is_circular_helical(fc: np.ndarray) -> None:
    for vector in np.asarray(fc, dtype=complex):
        if np.allclose(vector, 0.0, atol=1e-12, rtol=0.0):
            continue
        real = vector.real
        imag = vector.imag
        assert np.linalg.norm(real) == pytest.approx(np.linalg.norm(imag), abs=1e-10)
        assert float(np.dot(real, imag)) == pytest.approx(0.0, abs=1e-10)


# Fourier data types and JSON


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
                "terms": [{"q_coeff": [1.0, 0.0, 0.0], "coefficient": {"real": [0.0, 0.0, 1.0]}}],
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
    model = MagneticStructureModel(q_basis_rlu=((0.0, 0.0, 0.5), (0.25, 0.0, 0.0), (0.0, 0.0, 0.0)))
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


# Atom selectors


def test_atom_selector_resolves_species_indices_and_symmetry_options() -> None:
    structure = _structure()

    assert resolve_atom_selector(structure, AtomSelector.species("Fe")) == (0,)
    assert resolve_atom_selector(structure, AtomSelector.indices([1, 0, 1])) == (0, 1)

    selectors = viable_atom_selectors(structure, selected_indices=[1])
    assert AtomSelector.all() in selectors
    assert AtomSelector.species("Fe") in selectors
    assert AtomSelector.indices([1]) in selectors


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


def test_resolve_atom_selector_all_and_case_insensitive_kind() -> None:
    structure = _structure()
    assert resolve_atom_selector(structure, AtomSelector("ALL", ())) == (0, 1)


def test_resolve_symmetry_equivalent_groups_atoms() -> None:
    structure = _structure(label="Fe2", species=("Fe", "Fe"))
    resolved = resolve_atom_selector(structure, AtomSelector.symmetry_equivalent(0))
    assert 0 in resolved


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


def test_viable_atom_selectors_drops_out_of_range_selection() -> None:
    structure = _structure()
    selectors = viable_atom_selectors(structure, selected_indices=[0, 99, -1])
    index_selectors = [s for s in selectors if s.kind == "indices"]
    assert index_selectors == [AtomSelector.indices([0])]


def test_viable_atom_selectors_no_selection_has_no_index_selector() -> None:
    structure = _structure()
    selectors = viable_atom_selectors(structure)
    assert all(s.kind != "indices" for s in selectors)


# Atom-resolved Fourier components and moments


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


def test_atom_resolved_components_skips_disabled_and_filters_ids() -> None:
    structure = _structure()
    term = FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 1.0]))
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            MagneticComponent("on", "On", "collinear", AtomSelector.all(), (term,), enabled=True),
            MagneticComponent(
                "off", "Off", "collinear", AtomSelector.all(), (term,), enabled=False
            ),
            MagneticComponent("other", "Other", "collinear", AtomSelector.all(), (term,)),
        ),
    )
    # disabled "off" excluded; component_ids filter keeps only "on".
    resolved = atom_resolved_fourier_components(structure, model, component_ids=["on"])
    assert {c.component_id for c in resolved} == {"on"}


def _ordered_model(*component_ids: str) -> MagneticStructureModel:
    term = FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 1.0]))
    return MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=tuple(
            MagneticComponent(
                component_id,
                component_id.upper(),
                "collinear",
                AtomSelector.indices([0]),
                (term,),
                metadata={"builder": {"amplitude": 1.0}},
            )
            for component_id in component_ids
        ),
    )


def _ids(model: MagneticStructureModel) -> list[str]:
    return [component.component_id for component in model.components]


def test_duplicate_magnetic_components_inserts_copies_after_originals() -> None:
    model = set_magnetic_components_enabled(_ordered_model("a", "b", "c"), ["c"], False)

    duplicated, copy_ids = duplicate_magnetic_components(model, ["c", "a"])

    assert len(copy_ids) == 2
    assert _ids(duplicated) == ["a", copy_ids[0], "b", "c", copy_ids[1]]
    first_copy = duplicated.components[1]
    assert first_copy.label == "A (copy)"
    assert first_copy.terms == model.components[0].terms
    assert first_copy.selector == model.components[0].selector
    assert first_copy.enabled is True
    assert duplicated.components[4].enabled is False
    assert duplicated.q_basis_rlu == model.q_basis_rlu
    first_copy.metadata["builder"]["amplitude"] = 2.0
    assert model.components[0].metadata["builder"]["amplitude"] == 1.0


def test_duplicate_magnetic_components_round_trips_through_json() -> None:
    duplicated, copy_ids = duplicate_magnetic_components(_ordered_model("a"), ["a"])
    disabled = set_magnetic_components_enabled(duplicated, copy_ids, False)

    restored = MagneticStructureModel.from_json(disabled.to_json())

    assert _ids(restored) == ["a", copy_ids[0]]
    assert [component.enabled for component in restored.components] == [True, False]


def test_duplicate_magnetic_components_ignores_unknown_ids() -> None:
    model = _ordered_model("a", "b")

    duplicated, copy_ids = duplicate_magnetic_components(model, ["missing"])

    assert copy_ids == ()
    assert duplicated == model


def test_set_magnetic_components_enabled_only_touches_listed_components() -> None:
    model = _ordered_model("a", "b", "c")

    disabled = set_magnetic_components_enabled(model, ["a", "c"], False)
    assert [component.enabled for component in disabled.components] == [False, True, False]
    assert [component.component_id for component in enabled_magnetic_components(disabled)] == ["b"]

    enabled = set_magnetic_components_enabled(disabled, ["c"], True)
    assert [component.enabled for component in enabled.components] == [False, True, True]


def test_disabled_components_do_not_contribute_moments() -> None:
    structure = _structure()
    model = _ordered_model("a", "b")
    _positions, both, _indices = moments_in_supercell(structure, model, (1, 1, 1))

    disabled = set_magnetic_components_enabled(model, ["b"], False)
    _positions, one, _indices = moments_in_supercell(structure, disabled, (1, 1, 1))

    assert np.allclose(one * 2.0, both)


@pytest.mark.parametrize(
    ("selected", "offset", "expected"),
    [
        (["c"], -1, ["a", "c", "b", "d"]),
        (["b"], 1, ["a", "c", "b", "d"]),
        (["b", "c"], -1, ["b", "c", "a", "d"]),
        (["b", "c"], 1, ["a", "d", "b", "c"]),
        (["a"], -1, ["a", "b", "c", "d"]),
        (["d"], 1, ["a", "b", "c", "d"]),
        (["a", "c"], -1, ["a", "c", "b", "d"]),
        (["b", "d"], 1, ["a", "c", "b", "d"]),
        (["d"], -2, ["a", "d", "b", "c"]),
        (["a"], 10, ["b", "c", "d", "a"]),
        (["missing"], 1, ["a", "b", "c", "d"]),
        (["b"], 0, ["a", "b", "c", "d"]),
    ],
)
def test_move_magnetic_components(selected: list[str], offset: int, expected: list[str]) -> None:
    moved = move_magnetic_components(_ordered_model("a", "b", "c", "d"), selected, offset)

    assert _ids(moved) == expected


def test_move_magnetic_components_does_not_change_moments() -> None:
    structure = _structure()
    model = _ordered_model("a", "b", "c")
    _positions, before, _indices = moments_in_supercell(structure, model, (1, 1, 2))

    moved = move_magnetic_components(model, ["a"], 2)
    _positions, after, _indices = moments_in_supercell(structure, moved, (1, 1, 2))

    assert _ids(moved) == ["b", "c", "a"]
    assert np.allclose(before, after)


def _q_basis_model() -> MagneticStructureModel:
    collinear = MagneticComponent(
        "sdw",
        "SDW",
        "spin_density_wave",
        AtomSelector.indices([0]),
        (FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 1.0])),),
    )
    helix = build_helix_component(
        label="Helix",
        selector=AtomSelector.indices([0]),
        q_coeff=(0.0, 1.0, 0.0),
        real_axis_vector=(1.0, 0.0, 0.0),
        imaginary_axis_vector=(0.0, 1.0, 0.0),
        component_id="helix",
    )
    texture = MagneticComponent(
        "texture",
        "Texture",
        "texture",
        AtomSelector.all(),
        (
            FourierTerm((0.0, 0.0, 1.0), ComplexVector.from_vector([1.0, 0.0, 0.0])),
            FourierTerm((1.0, 0.0, -1.0), ComplexVector.from_vector([0.0, 1.0, 0.0])),
        ),
        enabled=False,
        metadata={"type": "texture"},
    )
    return MagneticStructureModel(
        q_basis_rlu=((0.25, 0.0, 0.0), (0.0, 0.0, 0.2), (0.0, 0.25, 0.0)),
        components=(collinear, helix, texture),
    )


def test_q_basis_slots_report_every_component_that_uses_a_row() -> None:
    slots = q_basis_slots(_q_basis_model())

    assert [slot.label for slot in slots] == ["q1", "q2", "q3"]
    assert slots[0].q_rlu == (0.25, 0.0, 0.0)
    assert slots[0].component_ids == ("sdw", "texture")
    assert slots[0].texture_component_ids == ("texture",)
    assert slots[1].component_ids == ("helix",)
    assert slots[1].direction_component_ids == ("helix",)
    assert slots[2].component_ids == ("texture",)
    assert all(slot.in_use for slot in slots)
    assert not q_basis_slots(MagneticStructureModel())[0].in_use


def test_component_q_basis_indices_and_texture_predicate() -> None:
    model = _q_basis_model()

    assert [component_q_basis_indices(component) for component in model.components] == [
        (0,),
        (1,),
        (0, 2),
    ]
    assert [is_texture_component(component) for component in model.components] == [
        False,
        False,
        True,
    ]


@pytest.mark.parametrize(
    ("q_coeff", "expected"),
    [
        ((0.0, 0.0, 0.0), "0"),
        ((1.0, 0.0, 0.0), "q1"),
        ((0.0, -1.0, 0.0), "−q2"),
        ((1.0, -1.0, 0.0), "q1 − q2"),
        ((-1.0, -1.0, 0.0), "−q1 − q2"),
        ((0.5, 0.0, 2.0), "0.5q1 + 2q3"),
    ],
)
def test_q_coeff_expression(q_coeff, expected: str) -> None:
    assert q_coeff_expression(q_coeff) == expected


def test_set_q_basis_vector_moves_every_component_on_the_row() -> None:
    structure = _structure()
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=_ordered_model("a", "b").components,
    )

    edited = set_q_basis_vector(model, 0, (0.0, 0.0, 0.25))

    assert edited.q_basis_rlu[0] == (0.0, 0.0, 0.25)
    assert edited.components == model.components
    assert np.allclose(
        edited.q_coeff_to_rlu(edited.components[1].terms[0].q_coeff), (0.0, 0.0, 0.25)
    )
    _positions, before, _indices = moments_in_supercell(structure, model, (1, 1, 4))
    _positions, after, _indices = moments_in_supercell(structure, edited, (1, 1, 4))
    assert not np.allclose(before, after)
    restored = MagneticStructureModel.from_json(edited.to_json())
    assert restored.q_basis_rlu == edited.q_basis_rlu


def test_set_q_basis_vector_allows_an_unused_row_and_rescaling_a_helix_row() -> None:
    model = _q_basis_model()
    model = replace(
        model,
        q_basis_rlu=(*model.q_basis_rlu[:2], (0.0, 0.0, 0.0)),
        components=model.components[:2],
    )

    assert set_q_basis_vector(model, 2, (0.1, 0.1, 0.0)).q_basis_rlu[2] == (0.1, 0.1, 0.0)
    assert set_q_basis_vector(model, 1, (0.0, 0.0, 0.1)).q_basis_rlu[1] == (0.0, 0.0, 0.1)


@pytest.mark.parametrize(
    ("index", "q_rlu", "message"),
    [
        (0, (0.3, 0.0, 0.0), "texture"),
        (1, (0.1, 0.0, 0.0), "keep its direction"),
        (1, (0.0, 0.0, -0.2), "keep its direction"),
        (2, (0.0, 0.0, 0.0), "texture"),
        (3, (0.1, 0.0, 0.0), "row index"),
        (1, (np.nan, 0.0, 0.0), "finite"),
        (1, (0.0, 0.0), "three-vector"),
    ],
)
def test_set_q_basis_vector_rejects_invalid_edits(index: int, q_rlu, message: str) -> None:
    with pytest.raises(MagneticStructureError, match=message):
        set_q_basis_vector(_q_basis_model(), index, q_rlu)


def test_set_q_basis_vector_checks_the_full_q_of_a_helix_that_combines_rows() -> None:
    helix = build_helix_component(
        label="Diagonal helix",
        selector=AtomSelector.indices([0]),
        q_coeff=(1.0, 1.0, 0.0),
        real_axis_vector=(0.0, 0.0, 1.0),
        imaginary_axis_vector=(1.0, -1.0, 0.0),
        component_id="diagonal",
    )
    model = MagneticStructureModel(
        q_basis_rlu=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.0)),
        components=(helix,),
    )

    # (0.2, 0, 0) is a positive multiple of q1, but it turns q1 + q2 from (0.1, 0.1, 0) to (0.2, 0.1, 0).
    with pytest.raises(MagneticStructureError, match="Diagonal helix"):
        set_q_basis_vector(model, 0, (0.2, 0.0, 0.0))
    with pytest.raises(MagneticStructureError, match="keep its direction"):
        set_q_basis_vector(model, 1, (0.0, 0.3, 0.0))
    assert set_q_basis_vector(model, 2, (0.0, 0.0, 0.5)).q_basis_rlu[2] == (0.0, 0.0, 0.5)


def test_set_q_basis_vector_rejects_zero_for_a_used_row_and_duplicate_vectors() -> None:
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.5), (0.25, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=_ordered_model("a").components,
    )

    with pytest.raises(MagneticStructureError, match="cannot be zero"):
        set_q_basis_vector(model, 0, (0.0, 0.0, 0.0))
    with pytest.raises(MagneticStructureError, match="q2 already holds"):
        set_q_basis_vector(model, 0, (0.25, 0.0, 0.0))
    assert set_q_basis_vector(model, 1, (0.0, 0.0, 0.0)).q_basis_rlu[1] == (0.0, 0.0, 0.0)


def test_atom_resolved_components_skips_empty_selection() -> None:
    structure = _structure()
    term = FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 1.0]))
    model = MagneticStructureModel(
        components=(
            MagneticComponent("none", "None", "collinear", AtomSelector.species("Xx"), (term,)),
        ),
    )
    assert atom_resolved_fourier_components(structure, model) == ()


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


def test_atom_resolved_fourier_component_dataclass() -> None:
    comp = AtomResolvedFourierComponent(
        component_id="c",
        q_coeff=(1.0, 0.0, 0.0),
        q_rlu=np.zeros(3),
        atom_fcs=np.zeros((1, 3), dtype=complex),
    )
    assert comp.component_id == "c"


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

    assert moment_at_atom(structure, model, 0, cell_offset=(0.0, 0.0, 0.0)) == pytest.approx(
        [0.0, 0.0, 2.0]
    )
    assert moment_at_atom(structure, model, 0, cell_offset=(0.0, 0.0, 1.0)) == pytest.approx(
        [0.0, 0.0, -2.0]
    )

    _positions, moments, atom_indices = moments_in_supercell(structure, model, (1, 1, 2))
    assert atom_indices.tolist() == [0, 0]
    assert sorted(round(value, 6) for value in moments[:, 2]) == pytest.approx([-2.0, 2.0])


def test_moment_at_atom_zero_when_no_components() -> None:
    structure = _structure()
    model = MagneticStructureModel()
    np.testing.assert_allclose(moment_at_atom(structure, model, 0), [0.0, 0.0, 0.0])


def test_moment_at_atom_out_of_range_raises() -> None:
    structure = _structure()
    model = MagneticStructureModel()
    with pytest.raises(MagneticStructureError, match="atom_index is out of range"):
        moment_at_atom(structure, model, 99)
    with pytest.raises(MagneticStructureError, match="atom_index is out of range"):
        moment_at_atom(structure, model, -1)


def test_moments_in_supercell_shapes() -> None:
    structure = _structure()
    model = MagneticStructureModel()
    positions, moments, atom_indices = moments_in_supercell(structure, model, (2, 1, 1))
    assert positions.shape == (4, 3)
    assert moments.shape == (4, 3)
    assert atom_indices.tolist() == [0, 1, 0, 1]


_BCC_IRON = partial(host_structure, label="Fe", lattice=2.868, species=("Fe", "Fe"))


def test_atom_phases_build_a_two_atom_antiferromagnet_as_one_component() -> None:
    structure = _BCC_IRON()
    one = MagneticStructureModel(
        components=(
            replace(
                build_collinear_component(
                    label="AFM",
                    selector=AtomSelector.all(),
                    q_coeff=(0.0, 0.0, 0.0),
                    moment=(0.0, 0.0, 2.2),
                ),
                atom_phase_turns={1: 0.5},
            ),
        )
    )
    two = MagneticStructureModel(
        components=(
            build_collinear_component(
                label="Up",
                selector=AtomSelector.indices([0]),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, 2.2),
            ),
            build_collinear_component(
                label="Down",
                selector=AtomSelector.indices([1]),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, -2.2),
            ),
        )
    )

    _positions, one_moments, _indices = moments_in_supercell(structure, one, (2, 2, 2))
    _positions, two_moments, _indices = moments_in_supercell(structure, two, (2, 2, 2))

    np.testing.assert_allclose(one_moments, two_moments, atol=1e-12)


def test_atom_phases_add_to_the_component_phase_of_a_helix() -> None:
    structure = _BCC_IRON()
    q_basis = ((0.0, 0.0, 0.2), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    helix = partial(
        build_helix_component,
        q_coeff=(1.0, 0.0, 0.0),
        real_axis_vector=(1.0, 0.0, 0.0),
        imaginary_axis_vector=(0.0, 1.0, 0.0),
    )
    one = MagneticStructureModel(
        q_basis_rlu=q_basis,
        components=(
            replace(
                helix(label="Helix", selector=AtomSelector.all(), phase_turns=0.1),
                atom_phase_turns={1: 0.25},
            ),
        ),
    )
    two = MagneticStructureModel(
        q_basis_rlu=q_basis,
        components=(
            helix(label="A", selector=AtomSelector.indices([0]), phase_turns=0.1),
            helix(label="B", selector=AtomSelector.indices([1]), phase_turns=0.35),
        ),
    )

    _positions, one_moments, _indices = moments_in_supercell(structure, one, (1, 1, 5))
    _positions, two_moments, _indices = moments_in_supercell(structure, two, (1, 1, 5))

    np.testing.assert_allclose(one_moments, two_moments, atol=1e-12)


def test_atom_phases_round_trip_json_with_string_keys() -> None:
    component = replace(
        build_collinear_component(
            label="AFM", selector=AtomSelector.all(), q_coeff=(0.0, 0.0, 0.0), moment=(0, 0, 1)
        ),
        atom_phase_turns={1: 0.5},
    )

    payload = component.to_json()
    restored = MagneticComponent.from_json(payload)

    assert payload["atom_phase_turns"] == {"1": 0.5}
    assert restored.atom_phase_turns == {1: 0.5}
    assert restored == component


def test_component_json_drops_malformed_saved_atom_phases() -> None:
    payload = build_collinear_component(
        label="AFM", selector=AtomSelector.all(), q_coeff=(0.0, 0.0, 0.0), moment=(0, 0, 1)
    ).to_json()
    payload["atom_phase_turns"] = {"two": 0.5, "-1": 0.5, "0": "nan", "3": 0.25}

    assert MagneticComponent.from_json(payload).atom_phase_turns == {3: 0.25}
    assert (
        MagneticComponent.from_json({**payload, "atom_phase_turns": [1, 2]}).atom_phase_turns == {}
    )


def test_component_json_without_atom_phases_loads_with_none() -> None:
    payload = build_collinear_component(
        label="FM", selector=AtomSelector.all(), q_coeff=(0.0, 0.0, 0.0), moment=(0, 0, 1)
    ).to_json()
    del payload["atom_phase_turns"]

    assert MagneticComponent.from_json(payload).atom_phase_turns == {}


@pytest.mark.parametrize(
    ("phases", "message"),
    [
        ({-1: 0.5}, "must not be negative"),
        ({0: float("nan")}, "must be finite"),
        ({"one": 0.5}, "must map atom indices"),
    ],
)
def test_invalid_atom_phases_are_refused(phases: dict, message: str) -> None:
    component = build_collinear_component(
        label="FM", selector=AtomSelector.all(), q_coeff=(0.0, 0.0, 0.0), moment=(0, 0, 1)
    )

    with pytest.raises(MagneticStructureError, match=message):
        replace(component, atom_phase_turns=phases)


def test_atom_phase_on_a_missing_atom_is_reported_when_resolved() -> None:
    component = replace(
        build_collinear_component(
            label="AFM", selector=AtomSelector.all(), q_coeff=(0.0, 0.0, 0.0), moment=(0, 0, 1)
        ),
        atom_phase_turns={5: 0.5},
    )

    with pytest.raises(MagneticStructureError, match="phase on atom 5, but the structure has 2"):
        atom_resolved_fourier_components(
            _BCC_IRON(), MagneticStructureModel(components=(component,))
        )


def test_atom_phases_for_selection_keeps_only_the_selected_atoms() -> None:
    component = replace(
        build_collinear_component(
            label="AFM", selector=AtomSelector.all(), q_coeff=(0.0, 0.0, 0.0), moment=(0, 0, 1)
        ),
        atom_phase_turns={0: 0.1, 1: 0.5},
    )

    kept = atom_phases_for_selection(
        _BCC_IRON(), component.atom_phase_turns, AtomSelector.indices([1])
    )
    zero_dropped = atom_phases_for_selection(_BCC_IRON(), {0: 0.0, 1: 0.5}, AtomSelector.all())

    assert kept == {1: 0.5}
    assert zero_dropped == {1: 0.5}


def test_ase_export_of_bcc_iron_carries_the_moments() -> None:
    model = MagneticStructureModel(
        components=(
            build_collinear_component(
                label="Ferromagnet",
                selector=AtomSelector.all(),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, 2.2),
            ),
        )
    )

    atoms = magnetic_structure_to_ase_atoms(_BCC_IRON(), model)
    supercell = magnetic_structure_to_ase_atoms(_BCC_IRON(), model, (2, 1, 3))

    assert atoms.get_chemical_symbols() == ["Fe", "Fe"]
    np.testing.assert_allclose(atoms.get_initial_magnetic_moments(), [[0.0, 0.0, 2.2]] * 2)
    np.testing.assert_allclose(atoms.get_positions()[1], [1.434, 1.434, 1.434])
    assert len(supercell) == 12
    np.testing.assert_allclose(supercell.cell.lengths(), [5.736, 2.868, 8.604])
    assert supercell.pbc.all()
    np.testing.assert_allclose(
        np.linalg.norm(supercell.get_initial_magnetic_moments(), axis=1), 2.2
    )


def test_ase_export_follows_the_supercell_atom_order_and_keeps_labels() -> None:
    structure = host_structure(label="FeO", lattice=3.0, species=("Fe1", "O"))
    model = MagneticStructureModel(
        q_basis_rlu=((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            build_collinear_component(
                label="AFM",
                selector=AtomSelector.species("Fe1"),
                q_coeff=(1.0, 0.0, 0.0),
                moment=(1.0, 0.0, 0.0),
            ),
        ),
    )

    atoms = magnetic_structure_to_ase_atoms(structure, model, (2, 1, 1))
    _positions, moments, _indices = moments_in_supercell(structure, model, (2, 1, 1))

    assert atoms.get_chemical_symbols() == ["Fe", "O", "Fe", "O"]
    assert atoms.info["mufinder_species_labels"] == ["Fe1", "O", "Fe1", "O"]
    np.testing.assert_allclose(atoms.get_initial_magnetic_moments(), moments)
    assert atoms.get_initial_magnetic_moments()[2, 0] == pytest.approx(-1.0)


def test_inverse_angstrom_wavevector_along_mnsi_111() -> None:
    lattice = [[4.558, 0.0, 0.0], [0.0, 4.558, 0.0], [0.0, 0.0, 4.558]]

    q = wavevector_rlu_from_inverse_angstrom(lattice, 0.035, (1.0, 1.0, 1.0))

    assert q == pytest.approx((0.0147, 0.0147, 0.0147), abs=5e-5)
    k_cartesian = 2.0 * np.pi * np.linalg.inv(np.asarray(lattice)) @ np.asarray(q)
    assert np.linalg.norm(k_cartesian) == pytest.approx(0.035)


def test_inverse_angstrom_wavevector_along_a_star_of_a_hexagonal_cell() -> None:
    a = 3.21
    lattice = [[a, 0.0, 0.0], [-a / 2.0, a * 3.0**0.5 / 2.0, 0.0], [0.0, 0.0, 5.21]]
    a_star_length = 2.0 * np.pi * 2.0 / (a * 3.0**0.5)

    q = wavevector_rlu_from_inverse_angstrom(lattice, 0.5 * a_star_length, (1.0, 0.0, 0.0))

    assert q == pytest.approx((0.5, 0.0, 0.0))


def test_reciprocal_axis_length_is_the_inverse_angstrom_of_one_rlu() -> None:
    a = 3.21
    lattice = [[a, 0.0, 0.0], [-a / 2.0, a * 3.0**0.5 / 2.0, 0.0], [0.0, 0.0, 5.21]]

    assert reciprocal_axis_length(lattice, "a*") == pytest.approx(4.0 * np.pi / (a * 3.0**0.5))
    assert reciprocal_axis_length(lattice, "c") == pytest.approx(2.0 * np.pi / 5.21)
    assert wavevector_rlu_from_inverse_angstrom(
        lattice, 0.3 * reciprocal_axis_length(lattice, "b*"), (0.0, 1.0, 0.0)
    ) == pytest.approx((0.0, 0.3, 0.0))


@pytest.mark.parametrize("magnitude", [0.0, float("inf")])
def test_inverse_angstrom_wavevector_refuses_a_zero_or_infinite_length(magnitude: float) -> None:
    with pytest.raises(MagneticStructureError, match="non-zero number of inverse Angstrom"):
        wavevector_rlu_from_inverse_angstrom(
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], magnitude, (0.0, 0.0, 1.0)
        )


def test_moments_in_supercell_bad_repeat_count_raises() -> None:
    structure = _structure()
    model = MagneticStructureModel()
    with pytest.raises(MagneticStructureError, match="three repeat counts"):
        moments_in_supercell(structure, model, (1, 1))
    with pytest.raises(MagneticStructureError, match="must be positive"):
        moments_in_supercell(structure, model, (1, 0, 1))


# muesr model data and helical decomposition


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
                "lin",
                "Lin",
                "spin_density_wave",
                AtomSelector.all(),
                (FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([1.0, 0.0, 0.0])),),
            ),
        ),
    )
    data = get_muesr_model_data(structure, model, merge_equivalent_k=False)[0]
    with pytest.raises(MagneticStructureError, match="tolerance must be positive"):
        split_muesr_model_data_into_helices(data, atol=0.0)


def test_orthogonal_helper_rejects_a_zero_vector() -> None:
    # Private helper: the helical split only passes it a unit vector, so the zero guard is unreachable publicly.
    with pytest.raises(MagneticStructureError, match="orthogonal vector to a zero"):
        _orthogonal_unit_vector(np.zeros(3))


# Preset component builders


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
    np.testing.assert_allclose(
        conical.terms[1].coefficient.to_array(), [np.sqrt(3.0), 1j * np.sqrt(3.0), 0.0]
    )


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


# Legacy and mCIF conversion


def test_legacy_magnetic_structure_converts_to_components() -> None:
    structure = _structure()
    structure.properties["magnetic_structure"] = {
        "propagation_vector": [0.0, 0.0, 0.25],
        "moments": [
            {"label": "Fe1", "vector": [0.0, 0.0, 1.5]},
            {"label": "O1", "vector": [0.0, 0.0, 0.0]},
        ],
    }

    model = magnetic_model_from_host_spins(structure)

    assert model.q_basis_rlu[0] == pytest.approx((0.0, 0.0, 0.25))
    assert len(model.components) == 1
    assert model.components[0].terms[0].q_coeff == pytest.approx((1.0, 0.0, 0.0))
    np.testing.assert_allclose(
        model.components[0].terms[0].coefficient.to_array().real, [0.0, 0.0, 1.5]
    )


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

    model = magnetic_model_from_host_spins(structure)

    assert model.q_basis_rlu[0] == pytest.approx((0.0, 0.0, 0.25))
    assert model.q_basis_rlu[1] == pytest.approx((0.5, 0.0, 0.0))
    assert len(model.components) == 1
    component = model.components[0]
    assert component.label == "O1"
    assert component.selector.values == (1,)
    assert component.metadata["source"] == "mcif"
    assert component.metadata["mcif_label"] == "O1"
    assert moment_at_atom(structure, model, 1) == pytest.approx([0.0, 1.0, 2.0])


def test_legacy_model_converts_stored_crystal_axis_moments_to_cartesian() -> None:
    # Projects imported before moments were converted on import store the
    # crystal-axis components as written; a hexagonal cell shows the difference.
    structure = _structure()
    structure.lattice_vectors = [[5.8, 0.0, 0.0], [-2.9, 2.9 * 3**0.5, 0.0], [0.0, 0.0, 11.2]]
    structure.properties["magnetic_structure"] = {
        "source": "mcif",
        "moments": [
            {
                "label": "Fe1",
                "atom_index": 0,
                "vector": [2.0, 2.0, 0.0],
                "coordinate_system": "crystalaxis",
            },
            {
                "label": "O1",
                "atom_index": 1,
                "vector": [0.0, 1.0, 0.0],
                "coordinate_system": "cartesian",
                "source_coordinate_system": "crystalaxis",
            },
        ],
    }

    model = magnetic_model_from_host_spins(structure)

    assert moment_at_atom(structure, model, 0) == pytest.approx([1.0, 3**0.5, 0.0])
    assert moment_at_atom(structure, model, 1) == pytest.approx([0.0, 1.0, 0.0])
    assert [component.metadata["source_coordinate_system"] for component in model.components] == [
        "crystalaxis",
        "crystalaxis",
    ]


def test_legacy_model_no_magnetism_yields_no_components() -> None:
    structure = _structure()
    model = magnetic_model_from_host_spins(structure)
    assert model.components == ()
    # Default q basis is all zero.
    assert model.q_basis_rlu == (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )


def test_legacy_model_from_atomic_spins_scalar() -> None:
    structure = _structure()
    structure.properties["atomic_spins"] = [1.5, -2.0]
    model = magnetic_model_from_host_spins(structure)
    # Two non-zero scalar spins -> two collinear components along z.
    assert len(model.components) == 2
    np.testing.assert_allclose(
        model.components[0].terms[0].coefficient.to_array().real, [0.0, 0.0, 1.5]
    )
    np.testing.assert_allclose(
        model.components[1].terms[0].coefficient.to_array().real, [0.0, 0.0, -2.0]
    )
    assert model.metadata["source"] == "legacy_host_magnetic_structure"


def test_legacy_model_uses_top_level_propagation_vector() -> None:
    structure = _structure()
    structure.properties["propagation_vector"] = [0.0, 0.0, 0.5]
    structure.properties["atomic_spins"] = [1.0, 0.0]
    model = magnetic_model_from_host_spins(structure)
    assert model.q_basis_rlu[0] == pytest.approx((0.0, 0.0, 0.5))
    # Non-zero propagation -> q_coeff (1,0,0).
    assert model.components[0].terms[0].q_coeff == (1.0, 0.0, 0.0)


def test_legacy_model_skips_zero_moment_entries() -> None:
    structure = _structure()
    structure.properties["initial_magnetic_moments"] = [0.0, 0.0]
    model = magnetic_model_from_host_spins(structure)
    assert model.components == ()


def test_legacy_model_handles_bad_propagation_vector_gracefully() -> None:
    structure = _structure()
    structure.properties["magnetic_structure"] = {
        "propagation_vector": "not-a-vector",
        "moments": [{"label": "Fe1", "vector": [0.0, 0.0, 1.0]}],
    }
    model = magnetic_model_from_host_spins(structure)
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
    model = magnetic_model_from_host_spins(structure)
    assert len(model.components) == 1
    assert model.components[0].selector.values == (0,)


# Skyrmion textures


def test_hexagonal_skyrmion_texture_produces_three_helical_terms() -> None:
    texture = build_hexagonal_skyrmion_texture(
        magnitude=3.0,
        q_directions=(
            (1.0, 0.0, 0.0),
            (-0.5, np.sqrt(3.0) / 2.0, 0.0),
            (-0.5, -np.sqrt(3.0) / 2.0, 0.0),
        ),
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


def test_hexagonal_skyrmion_default_directions_generate_three_terms() -> None:
    texture = build_hexagonal_skyrmion_texture(magnitude=1.0)
    assert len(texture.terms) == 3


def test_hexagonal_skyrmion_helicity_reverses_the_in_plane_moment() -> None:
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


def test_square_skyrmion_neel_type_builds() -> None:
    texture = build_square_skyrmion_texture(
        magnitude=1.0,
        q_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        skyrmion_type="neel",
    )
    assert len(texture.terms) == 2


def test_hexagonal_skyrmion_requires_three_q_directions() -> None:
    with pytest.raises(MagneticStructureError, match="three q directions"):
        build_hexagonal_skyrmion_texture(
            magnitude=1.0, q_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        )


def test_square_skyrmion_requires_two_q_directions() -> None:
    with pytest.raises(MagneticStructureError, match="two q directions"):
        build_square_skyrmion_texture(magnitude=1.0, q_directions=((1.0, 0.0, 0.0),))


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


def test_skyrmion_q_basis_rotates_in_cartesian_reciprocal_space() -> None:
    structure = _structure()

    q_basis = skyrmion_q_basis_rlu(
        structure, q1_rlu=(0.1, 0.0, 0.0), core_direction=(0.0, 0.0, 1.0), lattice="square"
    )

    np.testing.assert_allclose(q_basis[0], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(q_basis[1], [0.0, 0.1, 0.0], atol=1e-12)


def test_skyrmion_q_basis_hexagonal_lattice_rotation() -> None:
    structure = _structure()
    q_basis = skyrmion_q_basis_rlu(
        structure, q1_rlu=(0.1, 0.0, 0.0), core_direction=(0.0, 0.0, 1.0), lattice="hexagonal"
    )
    # 120-degree rotation in cubic reciprocal space.
    np.testing.assert_allclose(q_basis[0], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(q_basis[1], [-0.05, np.sqrt(3.0) / 2.0 * 0.1, 0.0], atol=1e-12)
    assert q_basis[2] == (0.0, 0.0, 0.0)


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


# Multi-q texture terms and components


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
    texture = MagneticTexture.from_json({"terms": [{"q_coeff": [1.0, 0.0, 0.0]}, "garbage", 42]})
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


def test_build_texture_component_default_label_and_identity_transform() -> None:
    texture = MagneticTexture(
        terms=(MagneticTextureTerm.from_vector((1.0, 0.0, 0.0), [0.0, 0.0, 1.0]),)
    )
    component = build_texture_component(label="   ", selector=AtomSelector.all(), texture=texture)
    assert component.label == "Multi-q texture"
    assert component.terms[0].q_coeff == (1.0, 0.0, 0.0)


def test_build_texture_component_requires_terms() -> None:
    with pytest.raises(MagneticStructureError, match="at least one Fourier term"):
        build_texture_component(label="X", selector=AtomSelector.all(), texture=MagneticTexture())


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


# Texture evaluation, normalisation and pruning


def test_evaluate_texture_moment_uses_negative_exponential_convention() -> None:
    texture = build_square_skyrmion_texture(
        magnitude=2.0,
        q_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        skyrmion_type="neel",
    )

    moment = evaluate_texture_moment(texture, (0.25, 0.0, 0.0))

    assert moment.shape == (3,)
    assert np.linalg.norm(moment) > 0.0


def test_evaluate_texture_moment_empty_texture_is_zero() -> None:
    moment = evaluate_texture_moment(MagneticTexture(), (0.1, 0.2, 0.3))
    np.testing.assert_allclose(moment, [0.0, 0.0, 0.0])


def test_evaluate_texture_grid_shape() -> None:
    texture = MagneticTexture(
        terms=(MagneticTextureTerm.from_vector((1.0, 0.0, 0.0), [1.0, 0.0, 0.0]),)
    )
    grid = evaluate_texture_grid(texture, (4, 4, 1))
    assert grid.shape == (4, 4, 1, 3)


def test_evaluate_texture_grid_pads_and_clamps_the_resolution() -> None:
    texture = MagneticTexture(
        terms=(MagneticTextureTerm.from_vector((1.0, 0.0, 0.0), [1.0, 0.0, 0.0]),)
    )

    assert evaluate_texture_grid(texture, (5,)).shape == (5, 1, 1, 3)
    assert evaluate_texture_grid(texture, (0, -3, 7)).shape == (1, 1, 7, 3)


def test_texture_diagnostics_empty_texture_returns_zeros() -> None:
    diag = texture_diagnostics(MagneticTexture())
    assert diag.term_count == 0
    assert diag.magnitude_max == 0.0


def test_texture_normalization_reduces_magnitude_variation_and_pruning_keeps_largest_terms() -> (
    None
):
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


def _rotated_about(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle)


def _skyrmion_lattice(
    lattice: str, *, core: tuple[float, float, float] = (0.0, 0.0, -1.0), **kwargs
) -> tuple[MagneticTexture, np.ndarray]:
    """Return a preset skyrmion texture and its Cartesian q basis (|q| = 0.1 Å⁻¹)."""

    axis = np.asarray(core, dtype=float)
    q1 = np.array([1.0, 0.0, 0.0])
    angle = np.pi / 2.0 if lattice == "square" else 2.0 * np.pi / 3.0
    q2 = _rotated_about(q1, axis, angle)
    if lattice == "square":
        texture = build_square_skyrmion_texture(
            magnitude=1.0, q_directions=(q1, q2), core_direction=core, **kwargs
        )
    else:
        texture = build_hexagonal_skyrmion_texture(
            magnitude=1.0, q_directions=(q1, q2, -(q1 + q2)), core_direction=core, **kwargs
        )
    return texture, 0.1 * np.array([q1, q2, np.zeros(3)])


@pytest.mark.parametrize("resolution", [(64, 64), (128, 128)])
@pytest.mark.parametrize("skyrmion_type", ["bloch", "neel"])
@pytest.mark.parametrize("helicity", [1, -1])
@pytest.mark.parametrize(("core", "expected"), [((0.0, 0.0, -1.0), -1), ((0.0, 0.0, 1.0), 1)])
def test_hexagonal_skyrmion_charge_follows_core_and_ignores_helicity(
    resolution, skyrmion_type, helicity, core, expected
) -> None:
    texture, q_basis = _skyrmion_lattice(
        "hexagonal", core=core, skyrmion_type=skyrmion_type, helicity=helicity
    )

    assert topological_charge_magnetic_unit_cell(texture, q_basis, resolution) == expected


@pytest.mark.parametrize("resolution", [(64, 64), (128, 128)])
@pytest.mark.parametrize("helicity", [1, -1])
@pytest.mark.parametrize(
    ("core", "background", "expected"),
    [
        ((0.0, 0.0, -1.0), -0.3, -1),
        ((0.0, 0.0, -1.0), 0.3, 1),
        ((0.0, 0.0, 1.0), -0.3, 1),
        ((0.0, 0.0, -1.0), 1.5, 0),
    ],
)
def test_square_skyrmion_charge_follows_core_and_background(
    resolution, helicity, core, background, expected
) -> None:
    texture, q_basis = _skyrmion_lattice(
        "square", core=core, ferromagnetic_background=background, helicity=helicity
    )

    assert topological_charge_magnetic_unit_cell(texture, q_basis, resolution) == expected


def test_square_skyrmion_without_background_has_an_undefined_charge() -> None:
    texture, q_basis = _skyrmion_lattice("square")

    with pytest.raises(MagneticStructureError, match="magnitude vanishes"):
        topological_charge_magnetic_unit_cell(texture, q_basis, (64, 64))


def test_topological_charge_ignores_q_basis_length_and_skew_but_follows_its_orientation() -> None:
    texture, q_basis = _skyrmion_lattice("hexagonal")
    # The preset's q2 is q1 turned 120° about the -z core, so q1 × q2 points along -z.
    same_orientation = np.array([[0.37, 0.0, 0.0], [0.05, -0.21, 0.0]])
    mirrored = same_orientation * np.array([1.0, -1.0, 1.0])

    assert topological_charge_magnetic_unit_cell(texture, 7.0 * q_basis) == -1
    assert topological_charge_magnetic_unit_cell(texture, same_orientation) == -1
    assert topological_charge_magnetic_unit_cell(texture, mirrored) == 1


def test_topological_charge_folds_an_in_plane_third_q_vector_into_q1_and_q2() -> None:
    preset, q_basis = _skyrmion_lattice("hexagonal")
    three_row_basis = np.array([q_basis[0], q_basis[1], -(q_basis[0] + q_basis[1])])
    on_q3 = MagneticTexture(
        tuple(
            MagneticTextureTerm(q_coeff=(0.0, 0.0, 1.0), coefficient=term.coefficient)
            if term.q_coeff == (-1.0, -1.0, 0.0)
            else term
            for term in preset.terms
        )
    )
    half_q3 = MagneticTexture((MagneticTextureTerm.from_vector((0, 0, 1), (1.0, 1.0j, 0.0)),))

    assert any(term.q_coeff == (0.0, 0.0, 1.0) for term in on_q3.terms)
    assert topological_charge_magnetic_unit_cell(on_q3, three_row_basis) == -1
    with pytest.raises(MagneticStructureError, match="not periodic"):
        topological_charge_magnetic_unit_cell(half_q3, [q_basis[0], q_basis[1], 0.5 * q_basis[0]])


def test_helix_and_spin_density_wave_have_zero_charge() -> None:
    q_basis = ((0.1, 0.0, 0.0), (0.0, 0.1, 0.0))
    helix = MagneticTexture((MagneticTextureTerm.from_vector((1, 0, 0), (0.0, 1.0, 1.0j)),))
    spin_density_wave = MagneticTexture(
        (
            MagneticTextureTerm.from_vector((0, 0, 0), (0.0, 0.0, 1.0)),
            MagneticTextureTerm.from_vector((1, 1, 0), (0.0, 0.0, 0.5)),
        )
    )

    assert topological_charge_magnetic_unit_cell(helix, q_basis) == 0
    assert topological_charge_magnetic_unit_cell(spin_density_wave, q_basis) == 0


def test_pure_spin_density_wave_has_nodes_and_an_undefined_charge() -> None:
    texture = MagneticTexture((MagneticTextureTerm.from_vector((1, 0, 0), (0.0, 0.0, 1.0)),))

    with pytest.raises(MagneticStructureError, match="magnitude vanishes"):
        topological_charge_magnetic_unit_cell(texture, ((0.1, 0, 0), (0, 0.1, 0)), (64, 64))


def test_topological_charge_raises_on_a_coarse_grid_and_refines_on_request() -> None:
    texture, q_basis = _skyrmion_lattice("hexagonal")
    normalized = normalize_texture_magnitude(texture, grid_resolution=(32, 32, 1))

    with pytest.raises(MagneticStructureError, match="too coarse"):
        topological_charge_magnetic_unit_cell(normalized, q_basis, (8, 8))
    assert (
        topological_charge_magnetic_unit_cell(normalized, q_basis, (8, 8), max_resolution=64) == -1
    )


@pytest.mark.parametrize(("scale", "expected"), [(63, -(63**2)), (64, -(64**2))])
def test_topological_charge_refines_past_aliasing_of_high_frequency_terms(scale, expected) -> None:
    preset, q_basis = _skyrmion_lattice("hexagonal")
    # Every q coefficient times `scale` tiles the magnetic cell with scale² skyrmions.
    scaled = MagneticTexture(
        tuple(
            MagneticTextureTerm(
                q_coeff=tuple(scale * value for value in term.q_coeff),
                coefficient=term.coefficient,
            )
            for term in preset.terms
        )
    )

    with pytest.raises(MagneticStructureError, match="too coarse"):
        topological_charge_magnetic_unit_cell(scaled, q_basis, (64, 64))
    assert (
        topological_charge_magnetic_unit_cell(scaled, q_basis, (64, 64), max_resolution=1024)
        == expected
    )


def test_topological_density_rejects_a_slice_grid_that_aliases() -> None:
    texture, q_basis = _skyrmion_lattice("hexagonal")
    dual = np.linalg.pinv(q_basis[:2]).T

    # Ten magnetic cells along each side need more than 20 intervals per side.
    with pytest.raises(MagneticStructureError, match="too coarse"):
        topological_density_2d(
            texture, q_basis, u_vector=10 * dual[0], v_vector=10 * dual[1], resolution=(21, 21)
        )
    tiled = topological_density_2d(
        texture, q_basis, u_vector=10 * dual[0], v_vector=10 * dual[1], resolution=(641, 641)
    )
    assert tiled.charge == pytest.approx(-100.0, abs=1e-6)


@pytest.mark.parametrize(
    ("q_basis", "message"),
    [
        (((0.1, 0.0, 0.0), (0.2, 0.0, 0.0)), "not parallel"),
        (((0.0, 0.0, 0.0), (0.0, 0.1, 0.0)), "not parallel"),
        (((0.1, 0.0, 0.0),), "two or three rows"),
    ],
)
def test_topological_charge_rejects_a_singular_q_basis(q_basis, message) -> None:
    texture, _q_basis = _skyrmion_lattice("hexagonal")

    with pytest.raises(MagneticStructureError, match=message):
        topological_charge_magnetic_unit_cell(texture, q_basis)


def test_topological_charge_rejects_textures_it_cannot_measure() -> None:
    q_basis = ((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1))
    uses_q3 = MagneticTexture((MagneticTextureTerm.from_vector((0, 0, 1), (1.0, 1.0j, 0.0)),))
    fractional = MagneticTexture((MagneticTextureTerm.from_vector((0.5, 0, 0), (0.0, 1.0, 1.0j)),))

    with pytest.raises(MagneticStructureError, match="out of the q1-q2 plane"):
        topological_charge_magnetic_unit_cell(uses_q3, q_basis)
    with pytest.raises(MagneticStructureError, match="not periodic"):
        topological_charge_magnetic_unit_cell(fractional, q_basis)
    with pytest.raises(MagneticStructureError, match="at least one Fourier term"):
        topological_charge_magnetic_unit_cell(MagneticTexture(), q_basis)
    with pytest.raises(MagneticStructureError, match="at least 3"):
        topological_charge_magnetic_unit_cell(uses_q3, q_basis[:2], (2, 64))


def test_topological_density_over_one_magnetic_cell_sums_to_the_charge() -> None:
    texture, q_basis = _skyrmion_lattice("hexagonal")
    dual = np.linalg.pinv(q_basis[:2]).T
    density_map = topological_density_2d(
        texture, q_basis, u_vector=dual[0], v_vector=dual[1], resolution=(65, 65)
    )
    swapped = topological_density_2d(
        texture, q_basis, u_vector=dual[1], v_vector=dual[0], resolution=(65, 65)
    )
    cell_area = float(np.linalg.norm(np.cross(dual[0], dual[1]))) / 64**2

    assert density_map.points.shape == (65, 65, 3)
    assert density_map.density.shape == (64, 64)
    assert density_map.charge == pytest.approx(-1.0, abs=1e-9)
    assert float(np.sum(density_map.density)) * cell_area == pytest.approx(-1.0, abs=1e-9)
    np.testing.assert_allclose(swapped.density, density_map.density.T, atol=1e-12)


def test_topological_density_rejects_bad_slices() -> None:
    texture, q_basis = _skyrmion_lattice("hexagonal")

    with pytest.raises(MagneticStructureError, match="not parallel"):
        topological_density_2d(texture, q_basis, u_vector=(1, 0, 0), v_vector=(2, 0, 0))
    with pytest.raises(MagneticStructureError, match="too coarse"):
        topological_density_2d(
            texture, q_basis, u_vector=(100, 0, 0), v_vector=(0, 100, 0), resolution=(5, 5)
        )
    with pytest.raises(MagneticStructureError, match="rows of 3"):
        topological_density_2d(texture, ((0.1, 0.0),), u_vector=(1, 0, 0), v_vector=(0, 1, 0))
    with pytest.raises(MagneticStructureError, match="at least 2"):
        topological_density_2d(
            texture, q_basis, u_vector=(1, 0, 0), v_vector=(0, 1, 0), resolution=(1, 4)
        )


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


_HEXAGONAL_LATTICE = (
    (3.0, 0.0, 0.0),
    (-1.5, 1.5 * np.sqrt(3.0), 0.0),
    (0.0, 0.0, 5.0),
)


def _basis_model(
    *rows: tuple[float, float, float], components: tuple[MagneticComponent, ...] = ()
) -> MagneticStructureModel:
    basis = list(rows) + [(0.0, 0.0, 0.0)] * (3 - len(rows))
    return MagneticStructureModel(q_basis_rlu=tuple(basis), components=components)


def test_q_basis_combination_rlu_combines_stored_rows() -> None:
    model = _basis_model((0.25, 0.0, 0.0), (0.0, 0.25, 0.0))

    assert np.allclose(q_basis_combination_rlu(model, (1.0, 0.0, 0.0)), (0.25, 0.0, 0.0))
    assert np.allclose(q_basis_combination_rlu(model, (1.0, -1.0, 0.0)), (0.25, -0.25, 0.0))
    assert np.allclose(q_basis_combination_rlu(model, (0.5, 2.0, 0.0)), (0.125, 0.5, 0.0))


@pytest.mark.parametrize(
    ("q_coeff", "message"),
    [
        ((0.0, 0.0, 0.0), "at least one q-basis row"),
        ((0.0, 0.0, 1.0), "q3 is not set"),
        ((1.0, 1.0, -1.0), "gives q = 0"),
        ((float("nan"), 0.0, 0.0), "finite"),
    ],
)
def test_q_basis_combination_rlu_rejects_unset_rows_and_zero_results(q_coeff, message) -> None:
    model = MagneticStructureModel(
        q_basis_rlu=((0.25, 0.0, 0.0), (0.0, 0.25, 0.0), (0.25, 0.25, 0.0)),
    )
    if message == "q3 is not set":
        model = _basis_model((0.25, 0.0, 0.0), (0.0, 0.25, 0.0))

    with pytest.raises(MagneticStructureError, match=message):
        q_basis_combination_rlu(model, q_coeff)


def test_magnetic_period_along_axes_combines_wavevectors_and_flags_incommensurate_axes() -> None:
    assert magnetic_period_along_axes([]) == (1, 1, 1)
    assert magnetic_period_along_axes([(0.5, 0.0, 0.25), (0.0, 1.0 / 3.0, 0.5)]) == (2, 3, 4)
    assert magnetic_period_along_axes([(0.0, 0.0, 0.123456789)]) == (1, 1, None)
    assert magnetic_period_along_axes([(0.0, 0.0, 1.0 / 70.0)]) == (1, 1, None)
    assert magnetic_period_along_axes([(0.0, 0.0, 1.0 / 70.0)], max_period=80) == (1, 1, 70)


def _checklist(
    model: MagneticStructureModel, repeats=(1, 1, 1), *, has_host: bool = True
) -> dict[str, tuple[str, str]]:
    return {
        item.key: (item.state, item.detail)
        for item in magnetic_builder_checklist(model, has_host=has_host, preview_repeats=repeats)
    }


def _helix(
    q_coeff=(1.0, 0.0, 0.0), *, label: str = "Helix", moment: float = 1.0
) -> MagneticComponent:
    return build_helix_component(
        label=label,
        selector=AtomSelector.all(),
        q_coeff=q_coeff,
        real_axis_vector=(moment, 0.0, 0.0),
        imaginary_axis_vector=(0.0, moment, 0.0),
    )


def test_magnetic_builder_checklist_starts_with_components_to_add() -> None:
    without_host = _checklist(MagneticStructureModel(), has_host=False)
    empty = _checklist(MagneticStructureModel())

    assert without_host["components"] == (CHECKLIST_TODO, "Load a host structure first.")
    assert empty["components"][0] == CHECKLIST_TODO
    assert "menu" in empty["components"][1]
    assert empty["q_basis"][0] == CHECKLIST_TODO
    assert empty["preview"][0] == CHECKLIST_TODO
    assert empty["ready"] == (CHECKLIST_TODO, "Finish the steps above first.")


def test_magnetic_builder_checklist_flags_disabled_and_zero_moment_components() -> None:
    disabled = replace(_helix(), enabled=False)
    zero = _helix(label="Zero", moment=0.0)

    all_disabled = _checklist(_basis_model((0.0, 0.0, 0.25), components=(disabled,)))
    zero_moment = _checklist(
        _basis_model((0.0, 0.0, 0.25), components=(zero, _helix(label="Kept"))), (1, 1, 4)
    )

    assert all_disabled["components"] == (
        CHECKLIST_PROBLEM,
        "Every component is disabled, so the structure has no moments.",
    )
    assert all_disabled["ready"] == (CHECKLIST_TODO, "Fix the flagged steps first.")
    assert zero_moment["components"] == (CHECKLIST_PROBLEM, "Zero moment: 'Zero'.")
    assert zero_moment["preview"][0] == CHECKLIST_DONE


def test_magnetic_builder_checklist_flags_a_zero_q_basis_row_in_use() -> None:
    checklist = _checklist(_basis_model(components=(_helix(),)))

    assert checklist["q_basis"][0] == CHECKLIST_PROBLEM
    assert "q1 is used by an enabled component" in checklist["q_basis"][1]


def test_magnetic_builder_checklist_needs_no_basis_for_q_zero_components() -> None:
    collinear = build_collinear_component(
        label="Ferro", selector=AtomSelector.all(), q_coeff=(0.0, 0.0, 0.0), moment=(0.0, 0.0, 1.0)
    )

    checklist = _checklist(MagneticStructureModel(components=(collinear,)))

    assert checklist["q_basis"] == (
        CHECKLIST_DONE,
        "Not needed: every enabled component has q = 0.",
    )
    assert checklist["preview"][0] == CHECKLIST_DONE
    assert checklist["ready"][0] == CHECKLIST_DONE


def test_magnetic_builder_checklist_asks_for_a_whole_commensurate_cell() -> None:
    model = _basis_model((0.5, 0.0, 0.25), components=(_helix(),))

    short = _checklist(model, (2, 1, 1))
    whole = _checklist(model, (2, 1, 4))

    assert short["preview"] == (
        CHECKLIST_TODO,
        "One magnetic cell needs a 2×1×4 preview; it shows 2×1×1.",
    )
    assert short["ready"][0] == CHECKLIST_TODO
    assert whole["q_basis"] == (CHECKLIST_DONE, "Enabled components use q1.")
    assert whole["preview"][0] == CHECKLIST_DONE
    assert whole["ready"][0] == CHECKLIST_DONE


def test_magnetic_builder_checklist_caps_long_commensurate_periods_at_the_largest_preview() -> None:
    model = _basis_model((0.0, 0.0, 0.05), components=(_helix(),))

    short = magnetic_builder_checklist(
        model, has_host=True, preview_repeats=(1, 1, 4), max_preview_repeat=12
    )
    longest = magnetic_builder_checklist(
        model, has_host=True, preview_repeats=(1, 1, 12), max_preview_repeat=12
    )

    assert short[2].detail == "The largest preview needs a 1×1×12 preview; it shows 1×1×4."
    assert longest[2].state == CHECKLIST_DONE
    assert "longer than the largest preview" in longest[2].detail


def test_magnetic_builder_checklist_flags_an_incommensurate_q_without_a_preview_repeat() -> None:
    model = _basis_model((0.0, 0.0, 0.123456789), components=(_helix(),))

    flat = _checklist(model)
    repeated = _checklist(model, (1, 1, 6))

    assert flat["preview"][0] == CHECKLIST_PROBLEM
    assert "incommensurate along c" in flat["preview"][1]
    assert flat["ready"] == (CHECKLIST_TODO, "Fix the flagged steps first.")
    assert repeated["preview"] == (
        CHECKLIST_DONE,
        "The 1×1×6 preview repeats along every incommensurate axis.",
    )
    assert repeated["ready"][0] == CHECKLIST_DONE


def test_q_cartesian_direction_follows_the_reciprocal_lattice() -> None:
    a_star = q_cartesian_direction(_HEXAGONAL_LATTICE, (0.25, 0.0, 0.0))

    assert np.allclose(a_star, (np.sqrt(3.0) / 2.0, 0.5, 0.0))
    assert abs(float(a_star @ crystal_axis_direction(_HEXAGONAL_LATTICE, "b"))) < 1e-12
    with pytest.raises(MagneticStructureError, match="non-zero"):
        q_cartesian_direction(_HEXAGONAL_LATTICE, (0.0, 0.0, 0.0))


def test_crystal_axis_and_reciprocal_axis_helpers_validate_their_axis() -> None:
    assert np.allclose(
        crystal_axis_direction(_HEXAGONAL_LATTICE, "B"), (-0.5, np.sqrt(3.0) / 2.0, 0.0)
    )
    assert reciprocal_axis_q_rlu("c*", 0.2) == (0.0, 0.0, 0.2)
    assert reciprocal_axis_q_rlu("a", -0.1) == (-0.1, 0.0, 0.0)
    with pytest.raises(MagneticStructureError, match="crystal axis"):
        crystal_axis_direction(_HEXAGONAL_LATTICE, "d")
    with pytest.raises(MagneticStructureError, match="reciprocal axis"):
        reciprocal_axis_q_rlu("d*", 0.2)
    with pytest.raises(MagneticStructureError, match="non-zero"):
        reciprocal_axis_q_rlu("a*", 0.0)


def test_default_preset_reference_axis_is_the_axis_most_perpendicular_to_q() -> None:
    assert default_preset_reference_axis(_HEXAGONAL_LATTICE, (0.0, 0.0, 0.25)) == "a"
    assert default_preset_reference_axis(_HEXAGONAL_LATTICE, (0.25, 0.0, 0.0)) == "b"


def _preset(kind: str, **overrides) -> MagneticComponent:
    kwargs = {
        "lattice_vectors": _HEXAGONAL_LATTICE,
        "q_rlu": (0.0, 0.0, 0.25),
        "q_coeff": (1.0, 0.0, 0.0),
        "moment": 2.0,
        "reference_axis": "a",
        "selector": AtomSelector.species("Fe"),
    }
    kwargs.update(overrides)
    return build_magnetic_preset_component(kind, **kwargs)


def test_helix_preset_rotates_perpendicular_to_q_from_the_reference_axis() -> None:
    right = _preset("helix")
    left = _preset("helix", handedness=-1)

    assert right.kind == "helix"
    assert right.label == "Helix"
    assert right.selector == AtomSelector.species("Fe")
    assert right.terms[0].q_coeff == (1.0, 0.0, 0.0)
    assert np.allclose(right.terms[0].coefficient.real, (2.0, 0.0, 0.0))
    assert np.allclose(right.terms[0].coefficient.imaginary, (0.0, 2.0, 0.0))
    assert np.allclose(left.terms[0].coefficient.imaginary, (0.0, -2.0, 0.0))


def test_helix_preset_projects_an_oblique_reference_axis_into_the_rotation_plane() -> None:
    component = _preset("helix", q_rlu=(0.25, 0.0, 0.0), reference_axis="a")
    q_direction = q_cartesian_direction(_HEXAGONAL_LATTICE, (0.25, 0.0, 0.0))
    real = np.asarray(component.terms[0].coefficient.real)
    imag = np.asarray(component.terms[0].coefficient.imaginary)

    assert abs(float(real @ q_direction)) < 1e-12
    assert abs(float(imag @ q_direction)) < 1e-12
    assert np.isclose(np.linalg.norm(real), 2.0)
    assert float(real @ crystal_axis_direction(_HEXAGONAL_LATTICE, "a")) > 0.0


def test_conical_preset_puts_the_cone_axis_along_q() -> None:
    component = _preset("conical", cone_angle_degrees=30.0, handedness=-1, label="Cone")

    assert component.kind == "conical"
    assert component.label == "Cone"
    static, rotating = component.terms
    assert static.q_coeff == (0.0, 0.0, 0.0)
    assert np.allclose(static.coefficient.real, (0.0, 0.0, 2.0 * np.cos(np.deg2rad(30.0))))
    assert np.allclose(rotating.coefficient.real, (2.0 * np.sin(np.deg2rad(30.0)), 0.0, 0.0))
    assert np.allclose(rotating.coefficient.imaginary, (0.0, -2.0 * np.sin(np.deg2rad(30.0)), 0.0))


def test_cycloid_preset_rotates_in_the_plane_holding_q_and_the_reference_axis() -> None:
    component = _preset("cycloid", reference_axis="a")

    assert component.kind == "cycloid"
    assert np.allclose(component.terms[0].coefficient.real, (2.0, 0.0, 0.0))
    assert np.allclose(component.terms[0].coefficient.imaginary, (0.0, 0.0, 2.0))


def test_spin_density_wave_preset_modulates_along_the_reference_axis() -> None:
    component = _preset("spin_density_wave", reference_axis="c")

    assert component.kind == "spin_density_wave"
    assert np.allclose(component.terms[0].coefficient.real, (0.0, 0.0, 2.0))
    assert np.allclose(component.terms[0].coefficient.imaginary, 0.0)


@pytest.mark.parametrize(
    ("kind", "overrides", "message"),
    [
        ("helix", {"reference_axis": "c"}, "parallel to q"),
        ("cycloid", {"reference_axis": "c"}, "parallel to q"),
        ("skyrmion", {}, "preset kind"),
        ("helix", {"moment": 0.0}, "moment must be positive"),
        ("helix", {"handedness": 0}, "handedness"),
        ("helix", {"q_rlu": (0.0, 0.0, 0.0)}, "non-zero"),
    ],
)
def test_magnetic_preset_rejects_invalid_choices(kind, overrides, message) -> None:
    with pytest.raises(MagneticStructureError, match=message):
        _preset(kind, **overrides)


# Builder geometry: q-basis rows, propagation and moment directions


def _ferromagnet(label: str = "fm") -> MagneticComponent:
    return build_collinear_component(
        label=label, selector=AtomSelector.all(), q_coeff=(0.0, 0.0, 0.0), moment=(0.0, 0.0, 1.0)
    )


def _sdw_on_row(row: int) -> MagneticComponent:
    return build_spin_density_wave_component(
        label=f"sdw {row}",
        selector=AtomSelector.all(),
        q_coeff=unit_q_coeff(row),
        amplitude_vector=(0.0, 0.0, 1.0),
        component_id=f"sdw-{row}",
    )


def test_allocating_a_zero_q_needs_no_basis_row() -> None:
    full_basis = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))

    q_basis, q_coeff = allocate_q_basis_row(
        MagneticStructureModel(q_basis_rlu=full_basis), (0.0, 0.0, 0.0)
    )

    assert q_basis == full_basis
    assert q_coeff == (0.0, 0.0, 0.0)


def test_a_finite_q_after_a_zero_q_component_takes_a_free_basis_row() -> None:
    stale_full_basis = ((0.25, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 0.25))
    model = MagneticStructureModel(q_basis_rlu=stale_full_basis, components=(_ferromagnet(),))

    q_basis, q_coeff = allocate_q_basis_row(model, (0.0, 0.0, 0.5))

    assert q_basis[0] == (0.0, 0.0, 0.5)
    assert q_coeff == (1.0, 0.0, 0.0)


def test_allocating_a_q_reuses_the_row_that_holds_it() -> None:
    model = MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 0.0)),
        components=(_sdw_on_row(1),),
    )

    q_basis, q_coeff = allocate_q_basis_row(model, (0.0, 0.0, 0.5))

    assert q_coeff == (0.0, 1.0, 0.0)
    assert q_basis == model.q_basis_rlu


def test_allocating_a_q_fails_when_every_row_is_used() -> None:
    basis = ((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1))
    model = MagneticStructureModel(
        q_basis_rlu=basis, components=tuple(_sdw_on_row(row) for row in range(3))
    )

    assert allocate_q_basis_row(model, (0.5, 0.5, 0.0)) == (basis, None)
    # Leaving out the component on row 3 frees that row.
    q_basis, q_coeff = allocate_q_basis_row(model, (0.5, 0.5, 0.0), ignored_component_ids={"sdw-2"})
    assert q_coeff == (0.0, 0.0, 1.0)
    assert q_basis[2] == (0.5, 0.5, 0.0)
    assert used_q_basis_indices(model) == {0, 1, 2}
    assert used_q_basis_indices(model, ignored_component_ids={"sdw-2"}) == {0, 1}


def test_texture_q_vectors_share_matching_rows_and_take_free_ones() -> None:
    model = MagneticStructureModel(
        q_basis_rlu=((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(_sdw_on_row(0),),
    )

    q_basis, transform = allocate_texture_q_basis_rows(
        model, ((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.0))
    )

    assert q_basis == ((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.0))
    assert transform == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))


def test_texture_q_vectors_fail_without_enough_free_rows() -> None:
    model = MagneticStructureModel(
        q_basis_rlu=((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(_sdw_on_row(0),),
    )

    _q_basis, transform = allocate_texture_q_basis_rows(
        model, ((0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.2))
    )

    assert transform is None


def _cone() -> MagneticComponent:
    return build_conical_component(
        label="cone",
        selector=AtomSelector.all(),
        q_coeff=(1.0, 0.0, 0.0),
        total_moment=2.0,
        real_axis_direction=(1.0, 0.0, 0.0),
        imaginary_axis_direction=(0.0, 1.0, 0.0),
        static_axis_direction=(0.0, 0.0, 1.0),
        cone_angle_degrees=30.0,
    )


def test_component_propagation_reads_the_first_finite_q_term() -> None:
    model = MagneticStructureModel(q_basis_rlu=((0.0, 0.0, 0.25), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    ferromagnet = _ferromagnet()

    assert component_propagation_rlu(model, _cone()) == (0.0, 0.0, 0.25)
    assert component_propagation_coeff(_cone()) == (1.0, 0.0, 0.0)
    assert component_propagation_rlu(model, ferromagnet) == (0.0, 0.0, 0.0)
    assert component_propagation_coeff(ferromagnet) is None
    assert component_propagation_rlu(model, replace(ferromagnet, terms=())) == (0.0, 0.0, 0.0)


def test_conical_geometry_reads_back_the_cone() -> None:
    amplitude, angle, real, imag, axis = conical_component_geometry(
        _cone(), np.asarray((0.0, 0.0, 1.0))
    )

    assert amplitude == pytest.approx(2.0)
    assert angle == pytest.approx(30.0)
    np.testing.assert_allclose(real, (1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(imag, (0.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(axis, (0.0, 0.0, 1.0))
    # Without q the axis is the rotation plane's normal.
    *_rest, axis_without_q = conical_component_geometry(_cone(), None)
    np.testing.assert_allclose(axis_without_q, (0.0, 0.0, 1.0), atol=1e-12)
    assert rotation_handedness(real, imag, axis) == 1
    assert rotation_handedness(imag, real, axis) == -1


def test_unit_directions_and_their_fallbacks() -> None:
    assert unit_direction_or((0.0, 3.0, 4.0), (1.0, 0.0, 0.0)) == pytest.approx((0.0, 0.6, 0.8))
    assert unit_direction_or((0.0, 0.0, 1e-9), (1.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)
    np.testing.assert_allclose(nonzero_unit_direction((2.0, 0.0, 0.0), "axis"), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="core direction must be a non-zero direction"):
        nonzero_unit_direction((0.0, 0.0, 0.0), "core direction")


@pytest.mark.parametrize(
    ("vector", "expected"),
    [
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ],
)
def test_perpendicular_unit_direction(vector, expected) -> None:
    np.testing.assert_allclose(perpendicular_unit_direction(vector), expected, atol=1e-12)


def test_reference_direction_is_projected_off_q() -> None:
    q = np.asarray((0.0, 0.0, 1.0))

    np.testing.assert_allclose(
        reference_direction_perpendicular_to((1.0, 0.0, 1.0), q), (1.0, 0.0, 0.0)
    )
    # A reference along q gives some direction perpendicular to it.
    np.testing.assert_allclose(
        reference_direction_perpendicular_to((0.0, 0.0, 2.0), q), (1.0, 0.0, 0.0)
    )


def test_constrained_secondary_direction_by_kind() -> None:
    q = np.asarray((0.0, 0.0, 1.0))
    primary = np.asarray((1.0, 0.0, 0.0))

    np.testing.assert_allclose(
        constrained_secondary_direction("helix", q, primary, 1.0), (0.0, 1.0, 0.0)
    )
    np.testing.assert_allclose(
        constrained_secondary_direction("conical", q, primary, -1.0), (0.0, -1.0, 0.0)
    )
    np.testing.assert_allclose(
        constrained_secondary_direction("cycloid", q, primary, -1.0), (0.0, 0.0, -1.0)
    )
    # A primary along q has no cross product; a perpendicular direction is used.
    np.testing.assert_allclose(constrained_secondary_direction("helix", q, q, 1.0), (1.0, 0.0, 0.0))


def test_propagation_direction_and_its_errors() -> None:
    lattice = ((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0))
    flat = ((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 0.0))

    np.testing.assert_allclose(propagation_direction(lattice, (0.0, 0.0, 0.5)), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="q must be non-zero"):
        propagation_direction(lattice, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="q must define a non-zero propagation direction"):
        propagation_direction(flat, (0.5, 0.0, 0.0))
    for q in ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)):
        np.testing.assert_allclose(
            propagation_direction(flat, q, allow_fallback=True), (0.0, 0.0, 1.0)
        )


# Previews


def test_tiled_supercell_lists_atoms_in_the_order_of_the_moments() -> None:
    structure = _structure()
    model = MagneticStructureModel(components=(_ferromagnet(),))

    tiled, atom_indices = tiled_supercell_structure(structure, (2, 1, 3))
    positions, _moments, moment_indices = moments_in_supercell(structure, model, (2, 1, 3))

    assert tiled.label == "Fe preview 2x1x3"
    assert tiled.parent_structure_id == structure.id
    assert len(tiled.species) == 12
    np.testing.assert_array_equal(atom_indices, moment_indices)
    assert tiled.species == [structure.species[index] for index in atom_indices]
    np.testing.assert_allclose(
        np.asarray(tiled.fractional_positions) @ np.asarray(tiled.lattice_vectors), positions
    )
    assert tiled.properties["pbc"] == [True, True, True]


def test_texture_preview_uses_hexagonal_q_basis() -> None:
    texture = MagneticTexture((MagneticTextureTerm.from_vector((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),))

    square = evaluate_texture_preview_grid(
        texture, mode="square_skyrmion", resolution=3, repeats=1.0
    )
    hexagonal = evaluate_texture_preview_grid(
        texture, mode="hexagonal_skyrmion", resolution=3, repeats=1.0
    )

    assert np.isclose(square[2, 1, 2], 1.0)
    assert abs(hexagonal[2, 1, 2]) < 1e-12


def test_texture_preview_q_basis_follows_core_direction_handedness() -> None:
    texture = MagneticTexture((MagneticTextureTerm.from_vector((0.0, 1.0, 0.0), (0.0, 0.0, 1.0j)),))

    core_down = evaluate_texture_preview_grid(
        texture,
        mode="hexagonal_skyrmion",
        core_direction=(0.0, 0.0, -1.0),
        resolution=3,
        repeats=1.0,
    )
    core_up = evaluate_texture_preview_grid(
        texture,
        mode="hexagonal_skyrmion",
        core_direction=(0.0, 0.0, 1.0),
        resolution=3,
        repeats=1.0,
    )

    assert core_down[1, 2, 2] < 0.0
    assert core_up[1, 2, 2] > 0.0
    assert np.isclose(core_down[1, 2, 2], -core_up[1, 2, 2])


# The saved model of a host


def test_saved_model_payload_is_read_before_the_host_spins() -> None:
    host = _structure(properties={"atomic_spins": [1.0, 0.0]})
    saved = MagneticStructureModel(components=(_ferromagnet("saved"),))

    model = magnetic_model_for_host(saved.to_json(), host)

    assert [component.label for component in model.components] == ["saved"]


def test_host_property_model_is_used_without_a_saved_payload() -> None:
    stored = MagneticStructureModel(components=(_ferromagnet("on host"),))
    host = _structure(properties={"magnetic_structure": stored.to_json()})

    model = magnetic_model_for_host(None, host)

    assert [component.label for component in model.components] == ["on host"]


def test_a_payload_without_components_falls_back_to_the_host_spins() -> None:
    host = _structure(properties={"atomic_spins": [2.0, 0.0]})

    model = magnetic_model_for_host({"schema_version": 2}, host)

    assert model == magnetic_model_from_host_spins(host)
    assert len(model.components) == 1


def test_the_old_host_spin_builder_name_still_imports() -> None:
    from mufinder import science
    from mufinder.science import magnetic_structure as module

    assert module.magnetic_model_from_legacy_structure is magnetic_model_from_host_spins
    assert science.magnetic_model_from_legacy_structure is magnetic_model_from_host_spins


def test_q_basis_placeholder_is_true_of_the_model() -> None:
    q_zero = _ferromagnet("on q = 0")
    on_a_row = MagneticComponent(
        component_id="row",
        label="on q1",
        kind="collinear",
        selector=AtomSelector.all(),
        terms=(FourierTerm((1.0, 0.0, 0.0), ComplexVector.from_vector([0.0, 0.0, 1.0])),),
    )
    basis = ((0.25, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    assert q_basis_placeholder(MagneticStructureModel()) == "No components yet"
    # Rows left behind in the basis do not make any component's q non-zero.
    assert q_basis_placeholder(MagneticStructureModel(q_basis_rlu=basis)) == "No components yet"
    assert (
        q_basis_placeholder(MagneticStructureModel(q_basis_rlu=basis, components=(q_zero,)))
        == "All components have q = 0"
    )
    assert (
        q_basis_placeholder(MagneticStructureModel(q_basis_rlu=basis, components=(on_a_row,)))
        is None
    )
    assert (
        q_basis_placeholder(
            MagneticStructureModel(
                q_basis_rlu=basis, components=(q_zero, replace(on_a_row, enabled=False))
            )
        )
        == "All enabled components have q = 0"
    )
    assert (
        q_basis_placeholder(MagneticStructureModel(components=(replace(q_zero, enabled=False),)))
        == "All components have q = 0"
    )


def test_no_host_gives_an_empty_model() -> None:
    saved = MagneticStructureModel(components=(_ferromagnet("saved"),))

    assert magnetic_model_for_host(saved.to_json(), None) == MagneticStructureModel()
