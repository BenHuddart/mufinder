"""Shared helpers for text shown to the user.

The house style for that text is ``docs/ui_text.rst``. The helpers live in
``domain`` because services and science build user-facing messages as well as
the app.
"""

from __future__ import annotations


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
