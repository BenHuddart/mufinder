"""Tests for deterministic workflow job naming."""

from __future__ import annotations

from mufinder.workflow.naming import job_name


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
