"""Tests for the packaged shell script templates and their loader."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import shutil
import subprocess

import pytest

import mufinder.services.shell_templates as shell_templates
from mufinder.services.shell_templates import render_shell_template


_SHELL_DIR = files("mufinder.assets").joinpath("shell")
_TEMPLATE_NAMES = sorted(entry.name for entry in _SHELL_DIR.iterdir() if entry.name.endswith(".sh"))


@pytest.fixture
def template_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Serve templates from ``tmp_path/shell`` instead of the packaged ones."""

    shell = tmp_path / "shell"
    shell.mkdir()
    monkeypatch.setattr(shell_templates, "files", lambda _package: tmp_path)
    shell_templates._load.cache_clear()
    yield shell
    shell_templates._load.cache_clear()


def test_placeholders_are_filled_and_the_shell_keeps_its_own_dollars(template_dir: Path) -> None:
    (template_dir / "t.sh").write_text(
        'echo ${tpl_word} "$HOME" ${MUFINDER_X:-${tpl_default}} $((1 + 2)) $TIMEFORMAT $tpl_bare\n',
        encoding="utf-8",
    )

    rendered = render_shell_template("t.sh", word="'a b'", default=4, bare="x")

    assert rendered == "echo 'a b' \"$HOME\" ${MUFINDER_X:-4} $((1 + 2)) $TIMEFORMAT x"


def test_documentation_lines_and_empty_standalone_placeholders_are_removed(template_dir: Path) -> None:
    (template_dir / "t.sh").write_text(
        "## About this template.\nfirst\n  ## Indented note.\n${tpl_optional}\n  ${tpl_indented}\n"
        "inline ${tpl_optional}end\n${tpl_block}\nlast\n",
        encoding="utf-8",
    )

    rendered = render_shell_template("t.sh", optional="", indented="", block="one\ntwo")

    assert rendered == "first\ninline end\none\ntwo\nlast"


def test_missing_and_unused_values_are_rejected(template_dir: Path) -> None:
    (template_dir / "t.sh").write_text("${tpl_a} ${tpl_b}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing values for tpl_b; unused values tpl_c"):
        render_shell_template("t.sh", a="1", c="3")


def test_substituted_values_are_not_substituted_again(template_dir: Path) -> None:
    (template_dir / "t.sh").write_text("${tpl_a}\n", encoding="utf-8")

    assert render_shell_template("t.sh", a="${tpl_a} $$") == "${tpl_a} $$"


@pytest.mark.parametrize("name", _TEMPLATE_NAMES)
def test_packaged_templates_are_valid_bash_and_document_their_conventions(name: str) -> None:
    text = _SHELL_DIR.joinpath(name).read_text(encoding="utf-8")
    assert "Template conventions (see mufinder.services.shell_templates)" in text
    assert text.endswith("\n")

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    # A placeholder reads as a variable expansion, so the raw template parses too.
    completed = subprocess.run([bash, "-n"], input=text, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_the_packaged_templates_ship_with_the_assets() -> None:
    assert {
        "help_loop.sh",
        "help_loop_timing.sh",
        "job_help_loop.sh",
        "keyword_job.sh",
        "keyword_job_strategy_bare.sh",
        "keyword_job_strategy_job_steps.sh",
        "keyword_job_strategy_launcher_step.sh",
        "login_command.sh",
    } == set(_TEMPLATE_NAMES)
