"""Packaged shell script templates.

Scripts MuFinder sends to a cluster as plain shell live in
``mufinder/assets/shell/`` so that they can be read, diffed and syntax-checked
as shell. Python only computes the values substituted into them. Every template
follows the same conventions, repeated at the top of each file:

- ``${tpl_name}`` is a placeholder. Only names starting ``tpl_`` are
  placeholders, so every other ``$`` is the shell's own and is left alone. A
  literal ``$$`` is written ``$$$$``.
- Values are inserted verbatim: the caller shell-quotes anything that needs it.
- Every placeholder must be given a value, and every value must be used.
- A placeholder alone on a line whose value is empty removes the line.
- Lines starting with ``##`` document the template and are removed.
- The file's final newline is not part of the rendered text.

Rendering uses :class:`string.Template`, as the job scripts of
:mod:`mufinder.services.script_template` do.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
import re
from string import Template


class _ShellTemplate(Template):
    # Case-sensitive, so ``$TIMEFORMAT`` or ``${MUFINDER_PARALLEL:-4}`` can
    # never be taken for a placeholder.
    flags = 0
    idpattern = r"tpl_[a-z0-9_]+"


_DOC_LINE_RE = re.compile(r"^[ \t]*##.*\n", re.MULTILINE)
_STANDALONE_PLACEHOLDER_RE = re.compile(r"^[ \t]*\$\{(tpl_[a-z0-9_]+)\}[ \t]*\n", re.MULTILINE)


@cache
def _load(name: str) -> _ShellTemplate:
    text = files("mufinder.assets").joinpath("shell", name).read_text(encoding="utf-8")
    return _ShellTemplate(_DOC_LINE_RE.sub("", text))


def render_shell_template(name: str, /, **values: object) -> str:
    """Return the template ``name`` from ``assets/shell`` with ``values`` filled in.

    Each keyword names a placeholder without its ``tpl_`` prefix, so
    ``castep=...`` fills ``${tpl_castep}``. Values are converted with ``str``.

    Raises:
        ValueError: A placeholder has no value, or a value has no placeholder.
    """

    template = _load(name)
    mapping = {f"tpl_{key}": str(value) for key, value in values.items()}
    placeholders = set(template.get_identifiers())
    if placeholders != set(mapping):
        missing = ", ".join(sorted(placeholders - set(mapping))) or "none"
        unused = ", ".join(sorted(set(mapping) - placeholders)) or "none"
        raise ValueError(f"Shell template {name}: missing values for {missing}; unused values {unused}.")
    text = _STANDALONE_PLACEHOLDER_RE.sub(
        lambda match: "" if mapping[match.group(1)] == "" else match.group(0),
        template.template,
    )
    return _ShellTemplate(text).safe_substitute(mapping).removesuffix("\n")
