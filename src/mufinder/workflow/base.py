"""Workflow interfaces for resolved job expansion."""

from __future__ import annotations

from typing import Protocol

from .types import JobSpec


class Workflow(Protocol):
    """Protocol for workflow objects that expand into resolved jobs.

    Product-style expansion across multiple sweep axes is intentionally
    deferred. Current implementations use one-axis map expansion.
    """

    def expand(self) -> list[JobSpec]:
        """Expand workflow inputs into resolved job specifications."""
