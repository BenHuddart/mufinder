"""Tests for the hyperfine results service: the payload and the Magnetic Fields hand-off."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.castep_io.magres import hyperfine_tensor_for_species, parse_magres_hyperfine
from mufinder.domain.models import Result, Structure
from mufinder.science.hyperfine import HyperfineError, per_bohr3_to_per_angstrom3
from mufinder.services.hyperfine_results import (
    HANDOFF_COUPLING_UNIT,
    HandoffTarget,
    NO_MULLIKEN_SPINS_WARNING,
    apply_magnetic_field_handoff,
    contact_shell_inputs,
    handoff_unavailable_reason,
    hyperfine_result_payload,
    mulliken_spins_in_structure_order,
    plan_magnetic_field_handoff,
)
from mufinder.services.project_service import ProjectService
from mufinder.workflow.hyperfine import HYPERFINE_CALC_TYPE

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hyperfine"

_EDGE = 5.0


def _structure(species=("Fe", "H:mu", "Fe")) -> Structure:
    positions = {"Fe": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], "H:mu": [[0.25, 0.25, 0.25]]}
    taken = {"Fe": 0, "H:mu": 0}
    fractional = []
    for name in species:
        fractional.append(positions[name][taken[name]])
        taken[name] += 1
    return Structure(
        lattice_vectors=[[_EDGE, 0.0, 0.0], [0.0, _EDGE, 0.0], [0.0, 0.0, _EDGE]],
        species=list(species),
        fractional_positions=fractional,
    )


def _mulliken_castep(path: Path, rows: list[tuple[str, int, float]]) -> Path:
    body = "\n".join(
        f"  {species:<8} {ion:4d}   up:   1.0  1.0  1.0  0.0  3.0  0.00  {spin:.2f}\n"
        "                      dn:   1.0  1.0  1.0  0.0  3.0"
        for species, ion, spin in rows
    )
    path.write_text(
        "     Atomic Populations (Mulliken)\n     -----------------------------\n"
        "Species   Ion Spin  s  p  d  f  Total  Charge  Spin\n"
        "==========================================================\n"
        f"{body}\n"
        "==========================================================\n",
        encoding="utf-8",
    )
    return path


def test_mulliken_spins_are_put_back_in_structure_order(tmp_path: Path) -> None:
    # CASTEP groups the populations by species in first-appearance order.
    structure = _structure(("Fe", "H:mu", "Fe"))
    castep = _mulliken_castep(
        tmp_path / "relax.castep", [("Fe", 1, 2.2), ("Fe", 2, 2.1), ("H:mu", 1, -0.02)]
    )

    assert mulliken_spins_in_structure_order(structure, castep) == (2.2, -0.02, 2.1)
    assert mulliken_spins_in_structure_order(structure, tmp_path / "missing.castep") is None
    short = _mulliken_castep(tmp_path / "short.castep", [("Fe", 1, 2.2)])
    assert mulliken_spins_in_structure_order(structure, short) is None


def test_contact_shell_inputs_separate_the_muon_from_the_host() -> None:
    shell = contact_shell_inputs(_structure(), (2.2, -0.02, 2.1))

    assert shell is not None
    assert shell["muon_cartesian_ang"] == pytest.approx([1.25, 1.25, 1.25])
    assert shell["host_moments_muB"] == [2.2, 2.1]
    assert shell["host_cartesian_ang"][1] == pytest.approx([0.5, 0.0, 0.0])
    assert contact_shell_inputs(_structure(), None) is None
    assert contact_shell_inputs(_structure(("Fe", "Fe")), (1.0, 1.0)) is None


def _result(shell: dict | None, *, site_job: str, structure_id: str) -> Result:
    tensor = hyperfine_tensor_for_species(
        parse_magres_hyperfine(FIXTURES / "fe_tetrahedral_muon.magres"), "H:mu"
    )
    payload = hyperfine_result_payload(
        tensor,
        pseudo_spin_density_angstrom3=None,
        shell=shell,
        source={
            "source_site_id": str(uuid4()),
            "source_relaxation_job_id": site_job,
            "relaxed_structure_id": structure_id,
            "charge_state": "positive_muon",
        },
    )
    return Result(result_type=HYPERFINE_CALC_TYPE, data=payload)


def _document(tmp_path: Path):
    return ProjectService().create_project(tmp_path / "h.mufinder", "H")


def _site_state(*, job_id: str | None = None, structure_id: str | None = None) -> dict:
    return {
        "source": "relaxation_clustering",
        "label": "Analysis cluster 1",
        "fractional_position": [0.25, 0.25, 0.25],
        "distortion_structure_id": structure_id,
        "properties": {"cluster": {"representative_job_id": job_id}} if job_id else {},
    }


def test_handoff_fills_every_form_showing_the_site_with_its_own_shell(tmp_path: Path) -> None:
    document = _document(tmp_path)
    relaxation_job, relaxed_structure = str(uuid4()), str(uuid4())
    shell = contact_shell_inputs(_structure(), (2.2, -0.02, 2.0))
    result = _result(shell, site_job=relaxation_job, structure_id=relaxed_structure)
    document.ui_state["magnetic_field_form"] = {
        "selected_muon_site": _site_state(job_id=relaxation_job),
        "contact_nnn": 1,
    }
    document.ui_state["magnetic_field_workspaces"] = [
        {
            "id": "w1",
            "title": "Workspace",
            "form": {
                "selected_muon_site": _site_state(structure_id=relaxed_structure),
                "contact_nnn": 2,
                "hyperfine_coupling": "-0.5",
                "hyperfine_coupling_unit": "T_per_muB",
            },
        },
        {"id": "w2", "title": "Other site", "form": {"selected_muon_site": _site_state()}},
    ]

    targets = plan_magnetic_field_handoff(document, result)

    assert [target.title for target in targets] == ["Magnetic Fields", "Workspace"]
    rho = per_bohr3_to_per_angstrom3(result.data["spin_density_bohr3"])
    # nnn=1 takes the nearer Fe (0.5 Å along x is 1.92 Å away, moment 2.0).
    assert targets[0].coupling == pytest.approx(rho / 2.0)
    assert not targets[0].overwrites
    assert targets[1].overwrites
    assert targets[1].existing_display == "-0.5 T/μB"
    assert apply_magnetic_field_handoff(document, targets) == 2
    main = document.ui_state["magnetic_field_form"]
    assert main["hyperfine_coupling_unit"] == HANDOFF_COUPLING_UNIT
    assert float(main["hyperfine_coupling"]) == pytest.approx(rho / 2.0, rel=1e-5)
    other = document.ui_state["magnetic_field_workspaces"][1]["form"]
    assert "hyperfine_coupling" not in other


@pytest.mark.parametrize(
    ("existing_text", "existing_unit", "overwrites"),
    [
        pytest.param("", "angstrom^-3", False, id="empty"),
        pytest.param("0.0", "angstrom^-3", False, id="widget-default-zero"),
        pytest.param("0", "T_per_muB", False, id="zero-in-another-unit"),
        pytest.param("-0.5", "T_per_muB", True, id="different-value"),
        pytest.param("not a number", "angstrom^-3", True, id="unreadable"),
    ],
)
def test_handoff_confirms_only_replacing_a_nonzero_coupling(
    existing_text: str, existing_unit: str, overwrites: bool
) -> None:
    target = HandoffTarget(
        title="Magnetic Fields",
        workspace_id=None,
        existing_text=existing_text,
        existing_unit=existing_unit,
        coupling=-0.02,
    )

    assert target.overwrites is overwrites


def test_handoff_without_a_matching_form_changes_nothing(tmp_path: Path) -> None:
    document = _document(tmp_path)
    result = _result(
        contact_shell_inputs(_structure(), (2.2, 0.0, 2.0)),
        site_job=str(uuid4()),
        structure_id=str(uuid4()),
    )
    document.ui_state["magnetic_field_form"] = {"selected_muon_site": _site_state()}

    assert plan_magnetic_field_handoff(document, result) == []
    assert apply_magnetic_field_handoff(document, []) == 0
    assert "hyperfine_coupling" not in document.ui_state["magnetic_field_form"]


def test_handoff_skips_a_form_whose_shell_fails(tmp_path: Path) -> None:
    document = _document(tmp_path)
    relaxation_job = str(uuid4())
    result = _result(
        contact_shell_inputs(_structure(), (2.2, 0.0, 2.0)),
        site_job=relaxation_job,
        structure_id=str(uuid4()),
    )
    # A 0.5 Å contact radius reaches no magnetic ion.
    document.ui_state["magnetic_field_form"] = {
        "selected_muon_site": _site_state(job_id=relaxation_job),
        "contact_rcont": 0.5,
    }

    (target,) = plan_magnetic_field_handoff(document, result)

    assert target.coupling is None and target.error
    assert apply_magnetic_field_handoff(document, [target]) == 0


def test_a_result_without_mulliken_spins_cannot_be_handed_over(tmp_path: Path) -> None:
    result = _result(None, site_job=str(uuid4()), structure_id=str(uuid4()))

    assert handoff_unavailable_reason(result) == NO_MULLIKEN_SPINS_WARNING
    with pytest.raises(HyperfineError, match="no Mulliken spins"):
        plan_magnetic_field_handoff(_document(tmp_path), result)


def test_handoff_clears_field_results_only_of_forms_whose_coupling_changes(
    tmp_path: Path,
) -> None:
    from mufinder.domain.models import Result as StoredResult
    from mufinder.services.magnetic_field_results import (
        MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE,
        MAGNETIC_FIELD_PAGE_OWNER,
        MAGNETIC_FIELD_RESULT_TYPE,
        magnetic_field_result,
    )

    document = _document(tmp_path)
    relaxation_job = str(uuid4())
    result = _result(
        contact_shell_inputs(_structure(), (2.2, -0.02, 2.0)),
        site_job=relaxation_job,
        structure_id=str(uuid4()),
    )
    document.ui_state["magnetic_field_form"] = {
        "selected_muon_site": _site_state(job_id=relaxation_job),
        "contact_nnn": 1,
    }
    for result_type in (MAGNETIC_FIELD_RESULT_TYPE, MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE):
        stored = StoredResult(result_type=result_type, data={"owner": MAGNETIC_FIELD_PAGE_OWNER})
        document.results[stored.id] = stored

    targets = plan_magnetic_field_handoff(document, result)
    apply_magnetic_field_handoff(document, targets)

    # The field results used the old contact term; the dipolar tensors did not.
    assert (
        magnetic_field_result(document, MAGNETIC_FIELD_PAGE_OWNER, MAGNETIC_FIELD_RESULT_TYPE)
        is None
    )
    assert (
        magnetic_field_result(
            document, MAGNETIC_FIELD_PAGE_OWNER, MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE
        )
        is not None
    )

    # Handing the same coupling over again changes nothing, so a new result stays.
    fresh = StoredResult(
        result_type=MAGNETIC_FIELD_RESULT_TYPE, data={"owner": MAGNETIC_FIELD_PAGE_OWNER}
    )
    document.results[fresh.id] = fresh
    apply_magnetic_field_handoff(document, plan_magnetic_field_handoff(document, result))
    assert fresh.id in document.results
