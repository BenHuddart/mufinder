"""Provenance records for Journal tests."""

from __future__ import annotations

from datetime import UTC, datetime

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import ProvenanceRecord


def generation_record(hour: int, *, day: int = 21, count: int = 2) -> ProvenanceRecord:
    """A muon-site generation record at ``hour``:00 UTC on ``day`` April 2026."""

    return ProvenanceRecord(
        timestamp=datetime(2026, 4, day, hour, 0, tzinfo=UTC),
        action="generate_candidates",
        workflow_stage=WorkflowStage.MUON_GENERATION,
        parameters={"method": "random", "count": count},
    )
