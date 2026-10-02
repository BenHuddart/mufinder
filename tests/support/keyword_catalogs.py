"""Synthetic CASTEP keyword catalogs and ``-help`` output for tests.

A real ``castep_keywords.json`` is licensed CASTEP material and must never be
committed, so tests that need a *complete* catalog (one that passes
``validate_keyword_catalog_payload(require_complete=True)``) build a synthetic
one here: the sentinel keywords plus enough filler to clear the size floors.
"""

from __future__ import annotations

import json
from pathlib import Path


_CELL_SENTINELS = ("lattice_cart", "positions_frac", "species_pot")
_PARAM_SENTINELS = ("task", "cut_off_energy", "xc_functional")


def complete_keyword_names() -> dict[str, tuple[str, ...]]:
    """Return keyword names for a catalog large enough to count as complete."""

    return {
        "cell": (*_CELL_SENTINELS, *(f"cell_filler_{index}" for index in range(45))),
        "param": (*_PARAM_SENTINELS, *(f"param_filler_{index}" for index in range(160))),
    }


def complete_catalog_payload(castep_version: str = "26.11") -> dict[str, object]:
    """Return a synthetic payload that validates as a complete catalog."""

    names = complete_keyword_names()
    return {
        "castep_version": castep_version,
        "cell": {name: _entry(name) for name in names["cell"]},
        "param": {name: _entry(name) for name in names["param"]},
        "types": ["String"],
        "levels": ["Basic"],
    }


def write_complete_catalog(path: Path, castep_version: str = "26.11") -> Path:
    """Write a synthetic complete catalog to ``path`` and return it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(complete_catalog_payload(castep_version)), encoding="utf-8")
    return path


def help_all_text() -> str:
    """Return ``castep -help all`` style output listing the synthetic keywords."""

    names = complete_keyword_names()
    lines = ["Help information on CELL keywords:", ""]
    lines.extend(f"{name.upper():<32} Description" for name in names["cell"])
    lines.extend(["", "Help information on PARAMETERS keywords:", ""])
    lines.extend(f"{name.upper():<32} Description" for name in names["param"])
    return "\n".join(lines) + "\n"


def help_keyword_text(name: str) -> str:
    """Return ``castep -help <NAME>`` style output for a synthetic keyword."""

    names = complete_keyword_names()
    description = f"Type: String                      Level: Basic\n\nHelp for {name.lower()}.\n\n"
    if name.lower() in names["cell"]:
        return (
            f"Help information on CELL keywords:\n\n {name.upper()}\n\n{description}"
            "Help information on PARAMETERS keywords:\n\nNone found\n"
        )
    return (
        "Help information on CELL keywords:\n\nNone found\n\n"
        f"Help information on PARAMETERS keywords:\n\n {name.upper()}\n\n{description}"
    )


def _entry(name: str) -> dict[str, str]:
    return {
        "keyword": name,
        "option_type": "String",
        "level": "Basic",
        "docstring": f"Help for {name}.",
    }
