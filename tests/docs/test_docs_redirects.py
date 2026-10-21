"""The redirect table in ``docs/conf.py`` for pages that moved."""

from __future__ import annotations

from pathlib import Path
import runpy


DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"


def _redirects() -> dict[str, str]:
    namespace = runpy.run_path(str(DOCS_ROOT / "conf.py"))
    return dict(namespace["REDIRECTS"])


def test_every_redirect_points_at_a_current_page() -> None:
    for old, new in _redirects().items():
        assert (DOCS_ROOT / f"{new}.rst").is_file(), f"redirect {old!r} targets missing page {new!r}"


def test_no_redirect_shadows_a_current_page() -> None:
    for old in _redirects():
        assert not (DOCS_ROOT / f"{old}.rst").exists(), f"{old!r} is still a page; the redirect would overwrite it"


def test_redirect_stub_forwards_to_the_target(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(DOCS_ROOT / "conf.py"))

    class _Builder:
        name = "html"

    class _App:
        outdir = tmp_path
        builder = _Builder()

    namespace["_write_redirects"](_App(), None)
    stub = (tmp_path / "magnetic_fields.html").read_text(encoding="utf-8")
    assert 'url=user_guide/magnetic_fields.html' in stub
    assert len(list(tmp_path.glob("*.html"))) == len(namespace["REDIRECTS"])
