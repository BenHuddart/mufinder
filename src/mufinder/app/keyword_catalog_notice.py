"""Shared wording for the state of the CASTEP keyword catalog.

Several places tell the user which keyword metadata the session is running on:
the status bar, the CASTEP setup dialog, the parameter dialog, and the warnings
shown at startup and at submission. They all take their text from here so the
explanation of what a keyword catalog is, and why it matters, stays consistent.
"""

from __future__ import annotations

from mufinder.castep_io.schema import OVERRIDE_KEYWORDS_JSON_ENV_VAR, KeywordCatalogStatus


LIMITED_CATALOG_EXPLANATION = (
    "MuFinder reads the list of keywords your CASTEP version understands from a keyword catalog "
    "(castep_keywords.json). The catalog is generated from CASTEP itself, so it is not shipped with "
    "MuFinder. Without one, MuFinder falls back to a small built-in set: fewer parameters are offered, "
    "and results can fail to load when CASTEP writes a keyword that the built-in set does not list "
    "(for example QUANTISATION_AXIS after a Hubbard U calculation)."
)
CATALOG_SETUP_HINT = (
    "Open Setup > CASTEP to generate the catalog from your cluster or a local CASTEP, "
    "or to import an existing castep_keywords.json."
)


def catalog_short_label(status: KeywordCatalogStatus) -> str:
    """Return a status-bar sized label for the active catalog."""

    if status.is_starter:
        return "CASTEP keywords: limited built-in set"
    if status.castep_version:
        return f"CASTEP keywords: {status.castep_version}"
    return "CASTEP keywords: loaded"


def catalog_source_description(status: KeywordCatalogStatus) -> str:
    """Return a sentence saying where the active catalog came from."""

    if status.is_starter:
        return "No keyword catalog is loaded; MuFinder is using its limited built-in keyword set."

    version = (
        f"CASTEP {status.castep_version}"
        if status.castep_version
        else "an unrecorded CASTEP version"
    )
    counts = f"{status.cell_count} cell and {status.param_count} param keywords"
    if status.origin == "setup":
        source = "installed through CASTEP setup"
    else:
        source = (
            f"set by the {OVERRIDE_KEYWORDS_JSON_ENV_VAR} environment variable, "
            "which overrides CASTEP setup"
        )
    return f"Keyword catalog for {version} ({counts}), {source}:\n{status.path}"


def catalog_detail_text(status: KeywordCatalogStatus) -> str:
    """Return the full explanation, including why a configured catalog was skipped."""

    parts = [catalog_source_description(status)]
    if status.issue:
        parts.append(status.issue)
    if status.is_starter:
        parts.append(LIMITED_CATALOG_EXPLANATION)
        parts.append(CATALOG_SETUP_HINT)
    return "\n\n".join(parts)
