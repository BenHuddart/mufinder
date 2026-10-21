"""The ``build-finished`` hook in ``docs/conf.py`` that sizes and lazy-loads the screenshots.

A one-page Sphinx project in ``tmp_path`` shows a tiny generated WebP and a PNG
in figures, with the repository's hook registered on its application.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re

from PIL import Image
from sphinx.application import Sphinx

CONF = Path(__file__).resolve().parents[2] / "docs" / "conf.py"
RENDER = Path(__file__).resolve().parents[2] / "scripts" / "docs_screenshots" / "render.py"


def _conf():
    spec = importlib.util.spec_from_file_location("mufinder_docs_conf", CONF)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hook():
    return _conf()._size_screenshots


def _renderer_scale_factor() -> int:
    """The renderer's ``SCALE_FACTOR``, read from its source: importing it configures Qt."""

    for node in ast.parse(RENDER.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SCALE_FACTOR" for t in node.targets):
            return int(ast.literal_eval(node.value))
    raise AssertionError(f"{RENDER.name} does not assign SCALE_FACTOR")


def _build(tmp_path: Path, page: str) -> str:
    source = tmp_path / "source"
    (source / "_generated").mkdir(parents=True)
    Image.new("RGB", (64, 40), (200, 120, 40)).save(source / "_generated" / "shot.webp", "WEBP", lossless=True)
    Image.new("RGB", (30, 20), (40, 120, 200)).save(source / "_generated" / "logo.png")
    (source / "conf.py").write_text("", encoding="utf-8")
    (source / "index.rst").write_text(page, encoding="utf-8")
    output = tmp_path / "html"
    app = Sphinx(str(source), str(source), str(output), str(tmp_path / "doctrees"), "html", status=None, warning=None)
    app.connect("build-finished", _hook())
    app.build()
    return (output / "index.html").read_text(encoding="utf-8")


def _img(html: str, name: str) -> str:
    return next(tag for tag in re.findall(r"<img\b[^>]*>", html) if name in tag)


PAGE = """Images
======

.. figure:: _generated/shot.webp
   :alt: A screenshot.

   A caption.

.. image:: _generated/logo.png
   :alt: A logo.
"""


def test_a_screenshot_is_lazy_loaded_at_half_its_pixel_size(tmp_path: Path) -> None:
    tag = _img(_build(tmp_path, PAGE), "shot.webp")

    assert 'src="_images/shot.webp"' in tag
    assert 'width="32"' in tag and 'height="20"' in tag
    assert 'loading="lazy"' in tag and 'decoding="async"' in tag
    assert 'alt="A screenshot."' in tag


def test_an_image_that_is_not_a_screenshot_is_left_alone(tmp_path: Path) -> None:
    tag = _img(_build(tmp_path, PAGE), "logo.png")

    assert "loading=" not in tag and "width=" not in tag


def test_a_screenshot_with_an_explicit_width_keeps_it(tmp_path: Path) -> None:
    page = PAGE.replace("   :alt: A screenshot.", "   :alt: A screenshot.\n   :width: 50")

    tag = _img(_build(tmp_path, page), "shot.webp")

    assert 'loading="lazy"' in tag
    assert 'width="50"' in tag
    assert "height=" not in tag


def test_a_failed_build_is_not_post_processed(tmp_path: Path) -> None:
    class App:
        builder = type("Builder", (), {"name": "html"})()
        outdir = str(tmp_path)

    page = tmp_path / "index.html"
    page.write_text('<img src="_images/shot.webp" />', encoding="utf-8")

    _hook()(App(), RuntimeError("build failed"))

    assert page.read_text(encoding="utf-8") == '<img src="_images/shot.webp" />'


def test_the_hook_divides_by_the_scale_the_renderer_draws_at() -> None:
    assert _conf().SCREENSHOT_SCALE == _renderer_scale_factor()
