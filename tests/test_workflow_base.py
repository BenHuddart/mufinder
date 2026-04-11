"""Tests for workflow protocol definitions."""

from __future__ import annotations

from mufinder.workflow.base import Workflow


class _SimpleWorkflow:
    def expand(self) -> list[object]:
        return []


def test_workflow_protocol_exposes_expand_contract() -> None:
    implementation = _SimpleWorkflow()

    assert callable(getattr(implementation, "expand"))
    assert implementation.expand() == []
    assert "Expand workflow inputs" in Workflow.expand.__doc__

