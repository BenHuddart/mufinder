"""Validation and safe writing of ``castep_keywords.json`` catalog files.

A keyword catalog is the JSON document CASTEP's ``-help`` output is distilled
into (the format ASE's ``create_castep_keywords`` writes). MuFinder never bundles
one, so every catalog it uses was either generated on the user's machine or
cluster, or adopted from a file the user pointed at. Both routes funnel through
this module so that a truncated, hand-edited, or unrelated JSON file is rejected
before it can replace a working catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any


#: Smallest keyword counts a complete catalog can plausibly have. CASTEP 26.11
#: reports 118 ``cell`` and 449 ``param`` keywords; releases back to CASTEP 16
#: stay well above these floors, while a generation run that lost most of its
#: ``-help`` calls falls below them.
MIN_COMPLETE_CELL_KEYWORDS = 40
MIN_COMPLETE_PARAM_KEYWORDS = 150

#: Keywords every real CASTEP catalog documents. Their absence means the file is
#: not a CASTEP keyword catalog, whatever its shape.
_SENTINEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cell": ("lattice_cart", "positions_frac", "species_pot"),
    "param": ("task", "cut_off_energy", "xc_functional"),
}
_REQUIRED_ENTRY_FIELDS = ("option_type", "level", "docstring")


class KeywordCatalogValidationError(ValueError):
    """Raised when a file or payload is not a usable CASTEP keyword catalog."""


@dataclass(frozen=True, slots=True)
class KeywordCatalogSummary:
    """What a validated catalog contains, for display and confirmation."""

    castep_version: str
    cell_count: int
    param_count: int

    def describe(self) -> str:
        """Return a one-line human-readable description."""

        version = (
            f"CASTEP {self.castep_version}"
            if self.castep_version
            else "CASTEP (version not recorded)"
        )
        return f"{version}: {self.cell_count} cell and {self.param_count} param keywords"


def validate_keyword_catalog_payload(
    payload: object, *, require_complete: bool = True
) -> KeywordCatalogSummary:
    """Check that ``payload`` is a CASTEP keyword catalog and summarise it.

    Args:
        payload: The decoded JSON document.
        require_complete: When true, also require the keyword counts and
            sentinel keywords of a full CASTEP catalog. The runtime loader
            passes ``False`` so that a deliberately small catalog still loads;
            adopting or generating a catalog always passes ``True``.

    Raises:
        KeywordCatalogValidationError: If the payload is malformed or, with
            ``require_complete``, implausibly small.
    """

    if not isinstance(payload, dict):
        raise KeywordCatalogValidationError("The file is not a JSON object.")

    counts: dict[str, int] = {}
    for section in ("cell", "param"):
        entries = payload.get(section)
        if not isinstance(entries, dict):
            raise KeywordCatalogValidationError(
                f"The file has no '{section}' keyword section, so it is not a CASTEP keyword catalog."
            )
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                raise KeywordCatalogValidationError(
                    f"The '{section}' entry for '{name}' is not a keyword description."
                )
            if require_complete:
                missing = [field for field in _REQUIRED_ENTRY_FIELDS if field not in entry]
                if missing:
                    raise KeywordCatalogValidationError(
                        f"The '{section}' entry for '{name}' is missing {', '.join(missing)}."
                    )
        counts[section] = len(entries)

    if require_complete:
        for section, sentinels in _SENTINEL_KEYWORDS.items():
            present = {str(name).lower() for name in payload[section]}
            absent = [name.upper() for name in sentinels if name not in present]
            if absent:
                raise KeywordCatalogValidationError(
                    f"The '{section}' section does not describe {', '.join(absent)}, "
                    "which every CASTEP keyword catalog contains."
                )
        if (
            counts["cell"] < MIN_COMPLETE_CELL_KEYWORDS
            or counts["param"] < MIN_COMPLETE_PARAM_KEYWORDS
        ):
            raise KeywordCatalogValidationError(
                f"The catalog is incomplete: it describes {counts['cell']} cell and {counts['param']} param "
                f"keywords, but a full CASTEP catalog has at least {MIN_COMPLETE_CELL_KEYWORDS} and "
                f"{MIN_COMPLETE_PARAM_KEYWORDS}."
            )

    return KeywordCatalogSummary(
        castep_version=_version_text(payload.get("castep_version")),
        cell_count=counts["cell"],
        param_count=counts["param"],
    )


def read_keyword_catalog_file(
    path: str | Path,
    *,
    require_complete: bool = True,
) -> tuple[dict[str, Any], KeywordCatalogSummary]:
    """Read and validate a ``castep_keywords.json`` file.

    Raises:
        KeywordCatalogValidationError: If the file cannot be read, is not JSON,
            or fails :func:`validate_keyword_catalog_payload`.
    """

    source = Path(path).expanduser()
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise KeywordCatalogValidationError(f"Could not read '{source}': {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise KeywordCatalogValidationError(f"'{source.name}' is not valid JSON: {exc}") from exc
    summary = validate_keyword_catalog_payload(payload, require_complete=require_complete)
    return payload, summary


def write_keyword_catalog_file(payload: dict[str, Any], path: str | Path) -> Path:
    """Write ``payload`` to ``path`` atomically and return the resolved path.

    The document is written to a sibling temporary file and moved into place, so
    an interrupted write never leaves a truncated catalog where a good one was.
    """

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=4)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination.resolve()


def _version_text(raw: object) -> str:
    if raw is None:
        return ""
    return str(raw).strip()
