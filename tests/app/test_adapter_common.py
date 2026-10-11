"""Tests for the shared app-layer adapter helpers.

These cover the three genuinely-differing parameters that each adapter passes to
:func:`~mufinder.app.adapter_common.build_input_template` (``context_label``,
``forbidden``, ``kpoints_list_reset``), plus the k-point rescaling helper's
message text, the generic ``AdapterExpansion`` shape, the expansion prologue
the supercell workflow adapters share, and the supercell-matrix helpers.
"""

from __future__ import annotations

import pytest

from mufinder.app.adapter_common import (
    AdapterExpansion,
    build_input_template,
    build_workflow_settings_template,
    claim_unique_name,
    coerce_supercell_matrix,
    is_identity_supercell,
    kpoint_reference_lattice,
    prepare_row,
    project_convergence_parameter_values,
    reference_is_supercell_of,
    require_row_label,
    rescale_kpoint_grid,
    row_values_has_keyword,
    seed_workflow_defaults,
    supercell_label,
    template_has_explicit_kpoints,
)
from mufinder.castep_io.types import BlockSection, SpeciesPotential
from mufinder.domain.models import Structure
from mufinder.workflow.types import CellTemplate, InputTemplate, ParamTemplate


def test_adapter_expansion_carries_messages() -> None:
    expansion = AdapterExpansion(jobs=(), messages=("note",))
    assert expansion.messages == ("note",)


def test_adapter_expansion_defaults_to_empty_messages() -> None:
    expansion = AdapterExpansion(jobs=())
    assert expansion.messages == ()


def test_build_input_template_parses_param_and_cell() -> None:
    template = build_input_template(
        {"task": "SINGLEPOINT", "symmetry_generate": True},
        context_label="test settings",
    )
    assert template.param is not None
    assert template.cell is not None
    assert template.cell.symmetry_generate is True


def test_build_input_template_context_label_appears_in_unsupported_error() -> None:
    # The context_label parameter is one of the genuinely-differing fragments:
    # each adapter phrases the "Unsupported CASTEP keyword" message differently.
    with pytest.raises(ValueError, match="Unsupported CASTEP keyword 'made_up' in bespoke context."):
        build_input_template({"made_up": 1}, context_label="bespoke context")


def test_build_input_template_forbidden_uses_exact_message() -> None:
    # The forbidden mapping is a genuinely-differing fragment: each adapter maps
    # canonical keyword -> its own exact validation string.
    with pytest.raises(ValueError, match="custom lock message"):
        build_input_template(
            {"cut_off_energy": 340.0},
            context_label="test settings",
            forbidden={"cut_off_energy": "custom lock message"},
        )


def test_build_input_template_kpoints_list_reset_sentinel() -> None:
    # kpoints_list_reset is the subtle genuinely-differing fragment: convergence
    # passes () ("explicitly clear the list layer"), relaxation/UEP pass None
    # ("inherit"). A scalar kpoint_mp_grid must write that sentinel into the
    # cell template's kpoints_list.
    clear_template = build_input_template(
        {"kpoint_mp_grid": (4, 4, 4)},
        context_label="test settings",
        kpoints_list_reset=(),
    )
    assert clear_template.cell is not None
    assert clear_template.cell.kpoints_list == ()

    inherit_template = build_input_template(
        {"kpoint_mp_grid": (4, 4, 4)},
        context_label="test settings",
        kpoints_list_reset=None,
    )
    assert inherit_template.cell is not None
    assert inherit_template.cell.kpoints_list is None


def test_build_workflow_settings_template_locks_structure_keywords() -> None:
    with pytest.raises(
        ValueError,
        match="positions_frac is controlled by the workflow structure and cannot be overridden.",
    ):
        build_workflow_settings_template({"positions_frac": BlockSection("positions_frac", ())})
    with pytest.raises(ValueError, match="Unsupported CASTEP keyword 'made_up' in workflow settings."):
        build_workflow_settings_template({"made_up": 1})


def test_build_workflow_settings_template_clears_kpoints_list_and_types_blocks() -> None:
    template = build_workflow_settings_template(
        {
            "kpoint_mp_grid": (3, 3, 3),
            "species_pot": BlockSection("species_pot", (("Si", "Si.recpot"),)),
            "cut_off_energy": 500.0,
        }
    )

    assert template.cell is not None
    assert template.cell.kpoint_mp_grid == (3, 3, 3)
    assert template.cell.kpoints_list == ()
    assert template.cell.species_pot == (SpeciesPotential(species="Si", potential="Si.recpot"),)
    assert template.param is not None
    assert [assignment.name for assignment in template.param.keywords] == ["cut_off_energy"]


def test_template_has_explicit_kpoints() -> None:
    assert template_has_explicit_kpoints(
        InputTemplate(cell=CellTemplate(kpoint_mp_grid=(2, 2, 2)))
    )
    assert not template_has_explicit_kpoints(InputTemplate(cell=None))
    assert not template_has_explicit_kpoints(
        InputTemplate(param=ParamTemplate(keywords=()))
    )


def test_row_values_has_keyword() -> None:
    assert row_values_has_keyword({"spin_polarized": True}, "spin_polarized")
    assert not row_values_has_keyword({"task": "SINGLEPOINT"}, "spin_polarized")


def test_rescale_kpoint_grid_no_project_grid_returns_none() -> None:
    grid, message = rescale_kpoint_grid(
        project_kpoint_grid=None,
        reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        row_lattice=[[8.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        is_identity_supercell=False,
        has_supercell=True,
        target_description="whatever.",
    )
    assert grid is None
    assert message is None


def test_rescale_kpoint_grid_identity_passes_through_without_message() -> None:
    grid, message = rescale_kpoint_grid(
        project_kpoint_grid=(4, 4, 4),
        reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        row_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        is_identity_supercell=True,
        has_supercell=False,
        target_description="unit cell.",
    )
    assert grid == (4, 4, 4)
    assert message is None


def test_rescale_kpoint_grid_supercell_builds_message_with_shared_core() -> None:
    grid, message = rescale_kpoint_grid(
        project_kpoint_grid=(4, 4, 4),
        reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        row_lattice=[[8.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        is_identity_supercell=False,
        has_supercell=True,
        target_description="2x1x1 (equivalent density).",
    )
    assert grid == (2, 4, 4)
    assert message == (
        "K-point grid adjusted from 4x4x4 to 2x4x4 for 2x1x1 (equivalent density)."
    )


def test_rescale_kpoint_grid_target_description_is_verbatim_trailer() -> None:
    # The trailing target_description is the genuinely-differing fragment for the
    # message (relaxation "(equivalent density)" vs UEP "row 'x' (equivalent
    # reciprocal-space density)"). The helper appends it verbatim after the
    # shared core.
    _grid, message = rescale_kpoint_grid(
        project_kpoint_grid=(4, 4, 4),
        reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        row_lattice=[[8.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        is_identity_supercell=False,
        has_supercell=True,
        target_description="row 'src' (equivalent reciprocal-space density).",
    )
    assert message is not None
    assert message.endswith("row 'src' (equivalent reciprocal-space density).")


def test_rescale_kpoint_grid_rescales_unit_cell_rows_for_a_supercell_reference() -> None:
    unit = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
    supercell = [[6.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]

    grid, message = rescale_kpoint_grid(
        project_kpoint_grid=(2, 3, 3),
        reference_lattice=supercell,
        row_lattice=unit,
        is_identity_supercell=True,
        has_supercell=False,
        target_description="unit cell (equivalent density).",
    )

    assert grid == (4, 3, 3)
    assert message == "K-point grid adjusted from 2x3x3 to 4x3x3 for unit cell (equivalent density)."


def test_reference_is_supercell_of_distinguishes_supercells_from_other_unit_cells() -> None:
    unit = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]

    assert reference_is_supercell_of([[6.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]], unit) is True
    assert reference_is_supercell_of(unit, unit) is False
    assert reference_is_supercell_of([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]], unit) is False
    assert reference_is_supercell_of(None, unit) is False


_CUBIC_4 = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]


def _structure(lattice: list[list[float]] = _CUBIC_4) -> Structure:
    return Structure(
        label="Si",
        lattice_vectors=[list(row) for row in lattice],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _template(values) -> InputTemplate:
    return build_input_template(values, context_label="test settings", forbidden={}, kpoints_list_reset=None)


@pytest.mark.parametrize(
    "value",
    [
        [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        ((2, 0, 0), (0, 2, 0), (0, 0, 1)),
        [["2", 0, 0], [0, 2.0, 0], [0, 0, 1]],
    ],
)
def test_coerce_supercell_matrix_accepts_lists_tuples_and_integer_strings(value) -> None:
    assert coerce_supercell_matrix(value) == ((2, 0, 0), (0, 2, 0), (0, 0, 1))


@pytest.mark.parametrize(
    "value",
    [None, "2x2x1", [[1, 0, 0]], [[1, 0, 0], [0, 1], [0, 0, 1]], [[1, 0, 0], [0, "a", 0], [0, 0, 1]]],
)
def test_coerce_supercell_matrix_rejects_malformed_values(value) -> None:
    assert coerce_supercell_matrix(value) is None


def test_supercell_label_names_unit_diagonal_and_custom_cells() -> None:
    assert is_identity_supercell(None)
    assert is_identity_supercell(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert not is_identity_supercell(((2, 0, 0), (0, 1, 0), (0, 0, 1)))

    custom = ((1, 1, 0), (0, 1, 0), (0, 0, 2))
    assert supercell_label(None) == "Unit cell"
    assert supercell_label(((1, 0, 0), (0, 1, 0), (0, 0, 1))) == "Unit cell"
    assert supercell_label(((2, 0, 0), (0, 2, 0), (0, 0, 1))) == "2x2x1"
    assert supercell_label(custom) == "Custom 3x3"
    # The adapter wording: lowercase unit cell, custom matrices spelled out
    # (this form also names the supercell structure, so it must stay stable).
    assert supercell_label(None, unit_cell="unit cell", spell_out_custom=True) == "unit cell"
    assert supercell_label(custom, unit_cell="unit cell", spell_out_custom=True) == "[1 1 0; 0 1 0; 0 0 2]"


def test_seed_workflow_defaults_layers_project_defaults_over_workflow_values() -> None:
    base = {"task": "SINGLEPOINT", "cut_off_energy": 300.0}

    seeded = seed_workflow_defaults(base, project_xc_functional="PBE", project_cutoff_energy=550)
    assert seeded == {"task": "SINGLEPOINT", "cut_off_energy": 550.0, "xc_functional": "PBE"}
    assert base == {"task": "SINGLEPOINT", "cut_off_energy": 300.0}

    unseeded = seed_workflow_defaults(base, project_xc_functional=None, project_cutoff_energy=None)
    assert unseeded == base


def test_kpoint_reference_lattice_falls_back_to_the_host_cell() -> None:
    host = _structure()
    reference = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 4.0]]

    assert kpoint_reference_lattice(reference, host) is reference
    assert kpoint_reference_lattice(None, host) == _CUBIC_4
    assert kpoint_reference_lattice([[8.0, 0.0, 0.0]], host) == _CUBIC_4


def test_row_label_and_unique_name_checks_reject_empty_and_duplicate_rows() -> None:
    seen: set[str] = set()
    assert require_row_label("  row_a ", workflow="pristine-relaxation") == "row_a"
    claim_unique_name("row_a", seen, workflow="pristine-relaxation")
    assert seen == {"row_a"}

    with pytest.raises(ValueError, match=r"^Pristine-relaxation row labels must not be empty\.$"):
        require_row_label("   ", workflow="pristine-relaxation")
    with pytest.raises(ValueError, match=r"^UEP source row labels must not be empty\.$"):
        require_row_label("", workflow="UEP source")
    with pytest.raises(ValueError, match=r"^Duplicate pristine-relaxation row label 'row_a'\.$"):
        claim_unique_name("row_a", seen, workflow="pristine-relaxation")
    with pytest.raises(ValueError, match=r"^Duplicate muon-relaxation job seed name 'mu_a'\.$"):
        claim_unique_name("mu_a", {"mu_a"}, workflow="muon-relaxation", noun="job seed name")


def test_prepare_row_drops_and_forces_values_and_rescales_the_project_grid() -> None:
    supercell = _structure([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 4.0]])

    prepared = prepare_row(
        label="row_a",
        structure=supercell,
        parameter_values={"write_cell_structure": False, "task": "GEOMETRYOPTIMIZATION"},
        template_from_values=_template,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 1)),
        project_kpoint_grid=(4, 4, 4),
        reference_lattice=_CUBIC_4,
        target_description="2x2x1 (equivalent density).",
        drop=("write_cell_structure",),
        force={"task": "SINGLEPOINT"},
    )

    assert prepared.values == {"task": "SINGLEPOINT"}
    assert prepared.kpoint_grid == (2, 2, 4)
    assert prepared.message == "K-point grid adjusted from 4x4x4 to 2x2x4 for 2x2x1 (equivalent density)."
    assert prepared.structure.lattice_cart.vectors[0] == (8.0, 0.0, 0.0)
    param_values = {keyword.name: keyword.value for keyword in prepared.template.param.keywords}
    assert param_values == {"task": "SINGLEPOINT"}


def test_prepare_row_keeps_a_row_pinned_kpoint_grid() -> None:
    prepared = prepare_row(
        label="row_a",
        structure=_structure(),
        parameter_values={"kpoint_mp_grid": (3, 3, 3)},
        template_from_values=_template,
        supercell_matrix=None,
        project_kpoint_grid=(4, 4, 4),
        reference_lattice=_CUBIC_4,
        target_description="unit cell (equivalent density).",
    )

    assert prepared.kpoint_grid is None
    assert prepared.message is None
    assert prepared.template.cell.kpoint_mp_grid == (3, 3, 3)


def test_prepare_row_reports_an_unconvertible_structure_by_row_label() -> None:
    broken = Structure(label="broken", lattice_vectors=_CUBIC_4, species=["Si"], fractional_positions=[])

    with pytest.raises(ValueError, match=r"^Could not convert row 'row_a' structure: "):
        prepare_row(
            label="row_a",
            structure=broken,
            parameter_values={},
            template_from_values=_template,
            supercell_matrix=None,
            project_kpoint_grid=None,
            reference_lattice=_CUBIC_4,
            target_description="unit cell (equivalent density).",
        )


def test_project_convergence_parameter_values_applies_only_set_defaults() -> None:
    from mufinder.castep_io.types import ValueWithUnit
    from mufinder.domain.models import Project

    assert project_convergence_parameter_values(Project(name="P")) == {}

    project = Project(
        name="P",
        recommended_xc_functional="PBE",
        recommended_cutoff_energy=500,
        recommended_kpoint_grid=(4, 4, 4),
        recommended_kpoint_lattice=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    )

    assert project_convergence_parameter_values(project) == {
        "xc_functional": "PBE",
        "cut_off_energy": ValueWithUnit(500.0, "eV"),
        "kpoint_mp_grid": (4, 4, 4),
    }
    doubled = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 5.0]]
    assert project_convergence_parameter_values(project, target_lattice=doubled)["kpoint_mp_grid"] == (2, 2, 4)
