from __future__ import annotations

from pathlib import Path


def test_sphinx_conf_declares_html_logo() -> None:
    conf_path = Path(__file__).resolve().parents[2] / "docs" / "conf.py"
    conf_text = conf_path.read_text(encoding="utf-8")

    assert 'html_logo = "branding/mufinder_logo.png"' in conf_text


def test_docs_index_displays_logo() -> None:
    index_path = Path(__file__).resolve().parents[2] / "docs" / "index.rst"
    index_text = index_path.read_text(encoding="utf-8")

    assert ".. image:: branding/mufinder_logo.png" in index_text
