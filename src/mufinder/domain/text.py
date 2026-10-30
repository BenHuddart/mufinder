"""Shared helpers for text shown to the user.

The house style for that text is ``docs/ui_text.rst``. The helpers live in
``domain`` because services and science build user-facing messages as well as
the app.
"""

from __future__ import annotations

import re


def plural(n: int, singular: str, plural: str | None = None) -> str:
    """Return ``n`` followed by the word that agrees with it.

    ``plural(1, "job")`` is ``"1 job"`` and ``plural(3, "job")`` is ``"3 jobs"``.
    Pass ``plural`` when the plural is not the singular with an ``s``
    (``plural(2, "analysis", "analyses")``), or to make a verb agree as well
    (``plural(n, "job is", "jobs are")``). Write counts through this helper,
    never with ``(s)``.
    """

    word = singular if n == 1 else (plural if plural is not None else f"{singular}s")
    return f"{n} {word}"


#: ``Job.execution_target`` of a job run on the user's own computer; any other
#: target is the name of a cluster from ``Setup > HPC Clusters``.
LOCAL_EXECUTION_TARGET = "local"


def execution_target_text(target: str | None) -> str:
    """Return where a job runs, as text names it: ``"this computer"``, or the cluster's name."""

    name = str(target or "").strip()
    if not name or name.casefold() == LOCAL_EXECUTION_TARGET:
        return "this computer"
    return name


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def space_group_text(symbol: str) -> str:
    """Return a spglib international symbol as it is printed: ``"P4_2/mnm"`` is ``"P4₂/mnm"``.

    A screw axis's ``_n`` becomes a subscript digit; the rest is unchanged.
    """

    return re.sub(r"_(\d)", lambda match: match.group(1).translate(_SUBSCRIPT_DIGITS), symbol)


def natural_sort_key(label: str) -> tuple[tuple[int, object], ...]:
    """Return a key that orders labels the way a reader expects.

    Runs of digits compare as numbers and the rest compares case-insensitively,
    so ``site_2`` sorts before ``site_10`` and ``Site_1`` beside ``site_1``.
    """

    parts: list[tuple[int, object]] = []
    for part in re.split(r"(\d+)", str(label)):
        if not part:
            continue
        if part.isdecimal():
            parts.append((0, int(part)))
        else:
            parts.append((1, part.casefold()))
    return tuple(parts)
