"""Tests for the workflow expansion contract documentation module."""

from __future__ import annotations

from mufinder.workflow import (
    expand_convergence_sweep_jobs,
    expand_muon_site_jobs,
    expand_nudged_elastic_band_jobs,
    expand_relaxation_jobs,
)
from mufinder.workflow import base


def test_base_documents_function_expansion_contract() -> None:
    # base.py is documentation only: the former Workflow Protocol was retired.
    assert not hasattr(base, "Workflow")
    assert "expand_" in base.__doc__
    assert "MERGE_PRECEDENCE_ORDER" in base.__doc__


def test_expand_functions_are_public_and_callable() -> None:
    for expand in (
        expand_convergence_sweep_jobs,
        expand_muon_site_jobs,
        expand_nudged_elastic_band_jobs,
        expand_relaxation_jobs,
    ):
        assert callable(expand)
