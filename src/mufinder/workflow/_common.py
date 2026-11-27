"""Small shared helpers private to the workflow package.

These consolidate a handful of byte-identical fragments that had been copied
across the ``expand_*`` workflow modules (and one app-layer adapter). Keeping
them here avoids drift while leaving the domain-specific logic in each workflow
module untouched.

This module must not import from any other workflow submodule; the helpers are
deliberately self-contained so importing ``_common`` never risks a cycle.
"""

from __future__ import annotations

from collections.abc import Sequence


def format_value_token(value: float) -> str:
    """Format a numeric value as a filesystem/label-safe token.

    Renders ``value`` with ``f"{value:.12g}"`` and then replaces ``"."`` with
    ``"p"`` and ``"-"`` with ``"m"`` so the token is safe to embed in job
    labels and file names (e.g. ``-1.5`` -> ``m1p5``).
    """

    token = f"{value:.12g}"
    token = token.replace(".", "p")
    token = token.replace("-", "m")
    return token


def require_jobs(items: Sequence[object], what: str) -> None:
    """Raise ``ValueError`` if ``items`` is empty.

    Standardizes the "at least one X must be configured" guard shared by the
    workflow ``expand_*`` functions. ``what`` names the missing item, e.g.
    ``"NEB job"`` or ``"muon position"``.
    """

    if not items:
        raise ValueError(f"At least one {what} must be configured.")
