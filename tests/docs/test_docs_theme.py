from __future__ import annotations

from pathlib import Path


def test_sphinx_conf_loads_custom_brand_css() -> None:
    conf_path = Path(__file__).resolve().parents[2] / "docs" / "conf.py"
    conf_text = conf_path.read_text(encoding="utf-8")

    assert 'html_static_path = ["_static"]' in conf_text
    assert 'html_css_files = ["mufinder-docs.css"]' in conf_text


def test_docs_theme_css_uses_mufinder_palette_tokens() -> None:
    css_path = Path(__file__).resolve().parents[2] / "docs" / "_static" / "mufinder-docs.css"
    css_text = css_path.read_text(encoding="utf-8")

    assert "--mf-bg-primary: #f5ede0;" in css_text
    assert "--mf-accent: #e02020;" in css_text
    assert '--mf-heading-font: "Playfair Display", "Libre Baskerville", Georgia, serif;' in css_text
    assert "@media (prefers-color-scheme: dark)" in css_text
