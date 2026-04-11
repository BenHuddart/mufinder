"""Security helpers for remote cluster execution."""

from __future__ import annotations

import re
from typing import Iterable


_REMOTE_SAFE_JOB_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def is_remote_safe_job_label(label: str) -> bool:
    """Return whether a job label is safe to use in remote paths and commands."""

    return bool(_REMOTE_SAFE_JOB_LABEL_RE.fullmatch(label.strip()))


def invalid_remote_job_labels(labels: Iterable[str]) -> list[str]:
    """Return unique labels that are unsafe for remote execution."""

    invalid: list[str] = []
    seen: set[str] = set()
    for raw_label in labels:
        label = str(raw_label).strip()
        if not label or is_remote_safe_job_label(label) or label in seen:
            continue
        seen.add(label)
        invalid.append(label)
    return invalid
