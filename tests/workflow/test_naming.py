"""Tests for deterministic workflow job naming."""

from __future__ import annotations

import pytest

from mufinder.domain.charge_states import MUON_RELAXATION_CALC_TYPES
from mufinder.domain.enums import WorkflowStage
from mufinder.workflow.naming import (
    cluster_workflow_directory_fragment,
    default_cluster_job_name_prefix,
    job_name,
    sanitize_remote_path_component,
    sanitize_remote_path_fragment,
)
from tests.support.jobs import make_job


def test_job_name_is_deterministic() -> None:
    assert job_name("convergence", "ecut_300") == job_name("convergence", "ecut_300")


def test_job_name_sanitizes_special_characters() -> None:
    assert job_name("conv test", "ecut 300/eV") == "conv_test_ecut_300_eV"


def test_job_name_rejects_empty_result() -> None:
    try:
        job_name("***", "///")
    except ValueError as exc:
        assert "at least one valid character" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid job name input")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(" muon  site/1 ", "muon_site_1"), ("__x__", "x"), ("***", "workflow"), ("", "workflow")],
)
def test_sanitize_remote_path_component(value: str, expected: str) -> None:
    assert sanitize_remote_path_component(value) == expected


def test_sanitize_remote_path_fragment_keeps_separators_and_drops_empty_parts() -> None:
    assert sanitize_remote_path_fragment(" /muon relaxation//positive muon/ ") == "muon_relaxation/positive_muon"
    assert sanitize_remote_path_fragment(" / ") == "workflow"


@pytest.mark.parametrize(
    ("workflow_type", "expected"),
    [
        ("pristine_relaxation", "pristine_relax"),
        ("/uep/", "uep"),
        ("muon_relaxation/positivemuon", "positivemuon"),
        ("muon_relaxation/muonium", "muonium"),
        ("convergence_cutoff", "convergence"),
        ("mixed_workflows", "convergence"),
        (None, "convergence"),
    ],
)
def test_default_cluster_job_name_prefix(workflow_type: str | None, expected: str) -> None:
    assert default_cluster_job_name_prefix(workflow_type) == expected


def test_cluster_workflow_directory_fragment() -> None:
    positive = make_job(calc_type=MUON_RELAXATION_CALC_TYPES["positive_muon"])
    muonium = make_job(calc_type=MUON_RELAXATION_CALC_TYPES["muonium"])
    uep = make_job(workflow_stage=WorkflowStage.UEP, calc_type="castep_uep")
    pristine = make_job(workflow_stage=WorkflowStage.PRISTINE_RELAXATION, calc_type="castep_relax")

    assert cluster_workflow_directory_fragment([positive]) == "muon_relaxation/positivemuon"
    assert cluster_workflow_directory_fragment([muonium, muonium]) == "muon_relaxation/muonium"
    assert cluster_workflow_directory_fragment([positive, muonium]) == "muon_relaxation_mixed_charge_states"
    assert cluster_workflow_directory_fragment([uep]) == "uep"
    assert cluster_workflow_directory_fragment([positive, uep]) == "mixed_workflows"
    assert cluster_workflow_directory_fragment([uep, pristine]) == "mixed_workflows"
