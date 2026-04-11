"""Deterministic, filesystem-safe job naming helpers."""

from __future__ import annotations

import re


def _sanitize(token: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", token)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def job_name(prefix: str, suffix: str = "") -> str:
    """Create a deterministic filesystem-safe job name."""

    raw = f"{prefix}_{suffix}" if suffix else prefix
    normalized = _sanitize(raw)
    if not normalized:
        raise ValueError("Job name must contain at least one valid character.")
    return normalized
