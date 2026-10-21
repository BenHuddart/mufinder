"""Sphinx configuration for MuFinder."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mufinder._version import current_version

project = "MuFinder"
author = "MuFinder Developers"
release = current_version()

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

exclude_patterns = ["_build", "_generated", "Thumbs.db", ".DS_Store"]

templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
html_logo = "../src/mufinder/assets/mufinder_logo.png"
html_css_files = ["mufinder-docs.css"]
html_theme_options = {
    "style_nav_header_background": "#2C2418",
}

# Pages that moved when the site was split into audience sections. The old
# document names stay reachable: ``build-finished`` writes a stub at each old
# name that forwards the browser to the new page, so bookmarks and inbound
# links keep working. ``tests/docs/test_docs_redirects.py`` checks the table.
REDIRECTS: dict[str, str] = {
    "api": "api/index",
    "implemented_functionality": "user_guide/overview",
    "project_home": "user_guide/project_home",
    "host_structure": "user_guide/host_structure",
    "supercells": "user_guide/supercells",
    "convergence_testing": "user_guide/convergence_testing",
    "comparison_table": "user_guide/comparison_table",
    "uep_workflow": "user_guide/uep_workflow",
    "site_generation": "user_guide/site_generation",
    "structural_relaxation_analysis": "user_guide/structural_relaxation_analysis",
    "magnetic_fields": "user_guide/magnetic_fields",
    "quantum_corrections": "user_guide/quantum_corrections",
    "energy_barriers": "user_guide/energy_barriers",
    "journal": "user_guide/journal",
}

_REDIRECT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={target}">
<link rel="canonical" href="{target}">
<title>Redirecting to {target}</title>
</head>
<body>
<p>This page has moved to <a href="{target}">{target}</a>.</p>
</body>
</html>
"""


def _write_redirects(app, exception) -> None:
    if exception is not None or app.builder.name != "html":
        return
    outdir = Path(app.outdir)
    for old, new in REDIRECTS.items():
        target = f"{new}.html"
        (outdir / f"{old}.html").write_text(_REDIRECT_PAGE.format(target=target), encoding="utf-8")


def setup(app):
    app.connect("build-finished", _write_redirects)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
