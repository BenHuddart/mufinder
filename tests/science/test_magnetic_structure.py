"""Tests for propagation-vector magnetic structure helpers."""

from __future__ import annotations

from functools import partial

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
    build_collinear_component,
    build_conical_component,
    build_cycloid_component,
    build_helix_component,
    build_hexagonal_skyrmion_texture,
    build_square_skyrmion_texture,
    build_spin_density_wave_component,
    build_texture_component,
    duplicate_magnetic_components,
    enabled_magnetic_components,
    evaluate_texture_grid,
    evaluate_texture_moment,
    get_muesr_model_data,
    magnetic_model_from_legacy_structure,
    moment_at_atom,
    moments_in_supercell,
    move_magnetic_components,
    normalize_texture_magnitude,
    prune_texture_terms,
    skyrmion_q_basis_rlu,
    split_muesr_model_data_into_helices,
    texture_diagnostics,
    resolve_atom_selector,
    set_magnetic_components_enabled,
    viable_atom_selectors,
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
            MagneticComponent("off", "Off", "collinear", AtomSelector.all(), (term,), enabled=False),
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

    assert moment_at_atom(structure, model, 0, cell_offset=(0.0, 0.0, 0.0)) == pytest.approx([0.0, 0.0, 2.0])
    assert moment_at_atom(structure, model, 0, cell_offset=(0.0, 0.0, 1.0)) == pytest.approx([0.0, 0.0, -2.0])

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
                "lin", "Lin", "spin_density_wave", AtomSelector.all(),
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
    np.testing.assert_allclose(conical.terms[1].coefficient.to_array(), [np.sqrt(3.0), 1j * np.sqrt(3.0), 0.0])


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


# Skyrmion textures


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


def test_hexagonal_skyrmion_default_directions_generate_three_terms() -> None:
    texture = build_hexagonal_skyrmion_texture(magnitude=1.0)
    assert len(texture.terms) == 3


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

    q_basis = skyrmion_q_basis_rlu(structure, q1_rlu=(0.1, 0.0, 0.0), core_direction=(0.0, 0.0, 1.0), lattice="square")

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
    component = build_texture_component(
        label="   ", selector=AtomSelector.all(), texture=texture
    )
    assert component.label == "Multi-q texture"
    assert component.terms[0].q_coeff == (1.0, 0.0, 0.0)


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
