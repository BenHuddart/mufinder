"""Shared helpers for text shown to the user.

The house style for that text is ``docs/ui_text.rst``. The helpers live in
``domain`` because services and science build user-facing messages as well as
the app.
"""

from __future__ import annotations

from collections.abc import Sequence
import errno
import re
from typing import NamedTuple


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


def plain_error(exc: BaseException) -> str:
    """Return why something failed in words, without an errno, a path or an end stop.

    ``str()`` of an ``OSError`` reads ``[Errno 13] Permission denied: '/the/whole/path'``.
    The message around this names the file. Any other exception's text is its own.
    """

    # A parser's "Failed to read '/the/path': ..." wraps the error worth reporting.
    if isinstance(exc.__cause__, (OSError, UnicodeError)):
        exc = exc.__cause__
    if isinstance(exc, UnicodeError):
        return "it is not a text file"
    if isinstance(exc, OSError):
        if exc.errno == errno.ENAMETOOLONG:
            return "the path is too long"
        if isinstance(exc, PermissionError):
            return "permission denied"
        if isinstance(exc, FileNotFoundError):
            return "it is no longer there"
        return (exc.strerror or str(exc) or "the file system refused it").lower()
    return str(exc).rstrip(" .")


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


def folder_phrase(verb: str, names: Sequence[str]) -> str:
    """Return what happened to named folders, in order: ``deleted folders 'a' and 'b'``."""

    quoted = [f"'{name}'" for name in sorted(names, key=natural_sort_key)]
    listed = quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"
    return f"{verb} {'folder' if len(names) == 1 else 'folders'} {listed}"


class KeptFolder(NamedTuple):
    """A folder a clean-up left in place, and why: the words for one folder, and for several.

    Empty words for several mean the reason only reads for one folder.
    """

    name: str
    because: tuple[str, str]


def folder_outcome_status(
    done: str, *, deleted: int = 0, archived: int = 0, cleaned: int = 0, kept: int = 0
) -> str:
    """Return the status for a clean-up: ``Removed 10 jobs; deleted 7 folders; kept 3``.

    Counts only, so it fits the status bar whatever the folders are called; the first count
    names the noun for the rest. The names of kept folders go in :func:`kept_folders_message`.
    """

    parts = [done]
    for verb, count in (
        ("deleted", deleted),
        ("archived", archived),
        ("cleaned", cleaned),
        ("kept", kept),
    ):
        if count:
            parts.append(
                f"{verb} {count}" if len(parts) > 1 else f"{verb} {plural(count, 'folder')}"
            )
    return "; ".join(parts)


def kept_folders_message(kept: Sequence[KeptFolder]) -> str:
    """Return why folders were kept: one line to show, then a line per folder for Details.

    Each entry is a folder of its own: two in different charge folders can share a name.
    """

    folders = sorted(kept, key=lambda folder: natural_sort_key(folder.name))
    count = plural(len(folders), "folder")
    reasons = {folder.because for folder in folders}
    one, several = next(iter(reasons))
    reason = one if len(folders) == 1 else several
    if len(reasons) == 1 and reason:
        summary = f"MuFinder kept {count} because {reason}."
    else:
        summary = f"MuFinder kept {count}. Details gives the reason for each."
    lines = [f"- '{folder.name}': {folder.because[0]}." for folder in folders]
    return "\n".join([summary, *lines])
