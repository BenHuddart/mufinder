from __future__ import annotations

from pathlib import Path


_DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"
_PACKAGED_LOGO = (_DOCS_ROOT.parent / "src" / "mufinder" / "assets" / "mufinder_logo.png").resolve()


def test_sphinx_conf_declares_html_logo() -> None:
    conf_text = (_DOCS_ROOT / "conf.py").read_text(encoding="utf-8")

    assert 'html_logo = "../src/mufinder/assets/mufinder_logo.png"' in conf_text
    assert (_DOCS_ROOT / "../src/mufinder/assets/mufinder_logo.png").resolve() == _PACKAGED_LOGO
    assert _PACKAGED_LOGO.is_file()


def test_docs_index_displays_logo() -> None:
    index_text = (_DOCS_ROOT / "index.rst").read_text(encoding="utf-8")

    assert ".. image:: ../src/mufinder/assets/mufinder_logo.png" in index_text


def test_docs_use_the_packaged_logo_instead_of_a_copy() -> None:
    assert not (_DOCS_ROOT / "branding" / "mufinder_logo.png").exists()
