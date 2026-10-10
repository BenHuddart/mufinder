"""Tests for the shared app-layer adapter helpers.

These cover the three genuinely-differing parameters that each adapter passes to
:func:`~mufinder.app.adapter_common.build_input_template` (``context_label``,
``forbidden``, ``kpoints_list_reset``), plus the k-point rescaling helper's
message text and the generic ``AdapterExpansion`` shape.
"""

from __future__ import annotations

import pytest

from mufinder.app.adapter_common import (
    AdapterExpansion,
    build_input_template,
    reference_is_supercell_of,
    rescale_kpoint_grid,
    row_values_has_keyword,
    template_has_explicit_kpoints,
)
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
