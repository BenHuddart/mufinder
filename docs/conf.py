"""Sphinx configuration for MuFinder."""

from __future__ import annotations

from pathlib import Path
import re
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


# The documentation screenshots are lossless WebP images rendered at twice
# their logical size (see "Documentation screenshots" in
# quality_and_automation.rst). ``build-finished`` gives each one's ``<img>``
# explicit ``width`` and ``height`` in CSS pixels, half the file's pixel size,
# so the browser reserves its space before it loads, and ``loading="lazy"``
# and ``decoding="async"`` so a page with many screenshots loads only those
# near the viewport. The theme's ``max-width: 100%; height: auto`` still
# scales an image down to the content column. ``tests/docs/test_docs_images.py``
# checks the hook.

# The renderer's SCALE_FACTOR (scripts/docs_screenshots/render.py); a test keeps them equal.
SCREENSHOT_SCALE = 2
_IMG_TAG = re.compile(r"<img\b[^>]*>")
_SRC_ATTRIBUTE = re.compile(r'\bsrc="([^"]+)"')


def _sized_img_tag(tag: str, page: Path, outdir: Path) -> str:
    source = _SRC_ATTRIBUTE.search(tag)
    if source is None or not source.group(1).endswith(".webp"):
        return tag
    image = (page.parent / source.group(1)).resolve()
    if image.parent != (outdir / "_images").resolve() or not image.is_file():
        return tag
    attributes = {"loading": "lazy", "decoding": "async"}
    # A figure given a :width: or :height: keeps the size its page chose.
    if re.search(r"\b(?:width|height)=", tag) is None:
        from PIL import Image

        with Image.open(image) as opened:
            width, height = opened.size
        attributes = {"width": str(width // SCREENSHOT_SCALE), "height": str(height // SCREENSHOT_SCALE), **attributes}
    added = "".join(
        f' {name}="{value}"' for name, value in attributes.items() if re.search(rf"\b{name}=", tag) is None
    )
    end = -2 if tag.endswith("/>") else -1
    return tag[:end].rstrip() + added + (" />" if end == -2 else ">")


def _size_screenshots(app, exception) -> None:
    if exception is not None or app.builder.name != "html":
        return
    outdir = Path(app.outdir)
    for page in outdir.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        if "_images/" not in text:
            continue
        sized = _IMG_TAG.sub(lambda match: _sized_img_tag(match.group(0), page, outdir), text)
        if sized != text:
            page.write_text(sized, encoding="utf-8")


def setup(app):
    app.connect("build-finished", _write_redirects)
    app.connect("build-finished", _size_screenshots)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
