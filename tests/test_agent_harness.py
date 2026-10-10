"""Repository harness checks for agent-legible development."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "mufinder"


REQUIRED_AGENT_DOCS = (
    "docs/agent_harness.rst",
    "docs/architecture.rst",
    "docs/project_specification.rst",
    "docs/quality_and_automation.rst",
    "docs/compatibility_policy.rst",
)

INDEXED_DOCS = (
    "agent_harness",
    "architecture",
    "quality_and_automation",
    "compatibility_policy",
)

PR_TEMPLATE_PATH = ".github/PULL_REQUEST_TEMPLATE.md"

PR_TEMPLATE_SECTIONS = (
    "## Why",
    "## What changed",
    "## Reviewer notes",
    "## Verification",
    "## Gates",
)

FORBIDDEN_LAYER_IMPORTS = {
    "domain": {
        "app",
        "castep_io",
        "persistence",
        "science",
        "services",
        "workflow",
    },
    "castep_io": {"app", "persistence", "science", "services", "workflow"},
    "persistence": {"app", "castep_io", "science", "services", "workflow"},
    "science": {"app", "persistence", "services", "workflow"},
    "workflow": {"app", "science", "services"},
    "services": {"app"},
}


# Submit/materialize handlers allowed on ``MainWindow``.
#
# Every workflow's submit -> materialize -> result -> refresh cycle lives on a
# ``mufinder.app.workflow_controllers`` controller (docs/architecture.rst,
# Workflow controllers). The only ``_on_submit_*`` / ``_materialize_*`` names
# that may remain here are the two cross-workflow dashboard entry points,
# which fan out to every controller. This set is a ratchet: it may only
# shrink, and a new per-workflow handler on ``MainWindow`` fails the test.
PINNED_MAIN_WINDOW_SUBMIT_HANDLERS = frozenset(
    {
        "_on_submit_from_dashboard",
        "_on_submit_local_from_dashboard",
    }
)


# App modules that still build their matplotlib canvas directly instead of
# through ``mufinder.app.plot_canvas.create_plot_canvas``. Each of these already
# uses constrained or tight layout (or, for the UEP slice colourbar, manual axes
# with no tick labels), so they are not clipping today. The set may only shrink:
# a new embedded plot must use the shared helper.
PINNED_DIRECT_FIGURE_CANVAS_MODULES = frozenset(
    {
        "convergence_examiner.py",
        "convergence_overview_widget.py",
        "quantum_analysis_dialog.py",
        "uep_wizard.py",
    }
)

def _python_files(package: str) -> list[Path]:
    return sorted((SOURCE_ROOT / package).rglob("*.py"))


def _main_window_submit_handler_methods() -> set[str]:
    """Return ``MainWindow`` method names matching the submit/materialize patterns."""

    path = SOURCE_ROOT / "app" / "main_window.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main_window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    return {
        member.name
        for member in main_window.body
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (member.name.startswith("_on_submit_") or member.name.startswith("_materialize_"))
    }


def _imports_figure_canvas(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("matplotlib.backends.backend_qt")
        and any(alias.name == "FigureCanvasQTAgg" for alias in node.names)
        for node in ast.walk(tree)
    )

def _mufinder_import_layers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    layers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 2 and parts[0] == "mufinder":
                    layers.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            parts = module.split(".")
            if len(parts) >= 2 and parts[0] == "mufinder":
                layers.add(parts[1])
    return layers


def test_agents_md_is_short_map_to_repository_system_of_record() -> None:
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert len(agents.splitlines()) <= 160
    for doc in REQUIRED_AGENT_DOCS:
        assert doc in agents
        assert (PROJECT_ROOT / doc).is_file()


def test_harness_docs_are_indexed_by_sphinx() -> None:
    index = (PROJECT_ROOT / "docs" / "index.rst").read_text(encoding="utf-8")
    for doc in INDEXED_DOCS:
        assert f"\n   {doc}\n" in f"\n{index}\n"


def test_architecture_doc_lists_layer_packages() -> None:
    architecture = (PROJECT_ROOT / "docs" / "architecture.rst").read_text(encoding="utf-8")
    for layer in (
        "mufinder.domain",
        "mufinder.castep_io",
        "mufinder.persistence",
        "mufinder.science",
        "mufinder.workflow",
        "mufinder.services",
        "mufinder.app",
    ):
        assert layer in architecture


def test_package_layers_do_not_import_forbidden_higher_layers() -> None:
    violations: list[str] = []
    for package, forbidden_layers in FORBIDDEN_LAYER_IMPORTS.items():
        for path in _python_files(package):
            imported_layers = _mufinder_import_layers(path)
            illegal_layers = sorted(imported_layers & forbidden_layers)
            if illegal_layers:
                rel_path = path.relative_to(PROJECT_ROOT)
                violations.append(f"{rel_path}: imports {', '.join(illegal_layers)}")

    assert not violations, "\n".join(violations)


def test_main_window_submit_handlers_do_not_grow() -> None:
    """Pin MainWindow's per-workflow submit/materialize handler set as a ratchet.

    New ``_on_submit_*`` / ``_materialize_*`` methods must not appear on
    ``MainWindow``: workflow submission logic belongs on a
    ``mufinder.app.workflow_controllers`` controller. A handler that moves off
    ``MainWindow`` is removed from ``PINNED_MAIN_WINDOW_SUBMIT_HANDLERS`` (the
    set may only shrink).
    """

    actual = _main_window_submit_handler_methods()
    new_handlers = sorted(actual - PINNED_MAIN_WINDOW_SUBMIT_HANDLERS)
    assert not new_handlers, (
        "New per-workflow submit/materialize handler(s) appeared on MainWindow: "
        f"{new_handlers}. Put submission logic on a workflow_controllers controller "
        "instead of MainWindow."
    )
    removed_handlers = sorted(PINNED_MAIN_WINDOW_SUBMIT_HANDLERS - actual)
    assert not removed_handlers, (
        "Handler(s) removed from MainWindow but still pinned: "
        f"{removed_handlers}. Update PINNED_MAIN_WINDOW_SUBMIT_HANDLERS (shrink it) "
        "when a workflow migrates onto a controller."
    )


def test_app_figures_do_not_use_fractional_subplot_margins() -> None:
    """Forbid ``subplots_adjust`` margins in widgets; they clip labels when small.

    A figure-fraction margin reserves a share of the canvas while tick and axis
    labels need fixed pixels, so a small dialog cuts them off (minus signs
    included). Use ``mufinder.app.plot_canvas.create_plot_canvas``, whose
    constrained layout sizes margins from the rendered labels.
    """

    offenders = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in _python_files("app")
        if "subplots_adjust(" in path.read_text(encoding="utf-8")
    )
    assert not offenders, (
        f"subplots_adjust() margins found in {offenders}. Build the figure with "
        "mufinder.app.plot_canvas.create_plot_canvas instead."
    )


def test_direct_figure_canvas_modules_do_not_grow() -> None:
    """Ratchet the app modules that build a matplotlib canvas without the helper."""

    actual = {
        path.name
        for path in _python_files("app")
        if path.name != "plot_canvas.py" and _imports_figure_canvas(path)
    }
    new_modules = sorted(actual - PINNED_DIRECT_FIGURE_CANVAS_MODULES)
    assert not new_modules, (
        f"{new_modules} build a FigureCanvasQTAgg directly. Use "
        "mufinder.app.plot_canvas.create_plot_canvas so the plot keeps its labels visible."
    )
    migrated_modules = sorted(PINNED_DIRECT_FIGURE_CANVAS_MODULES - actual)
    assert not migrated_modules, (
        f"{migrated_modules} no longer build a canvas directly. Shrink "
        "PINNED_DIRECT_FIGURE_CANVAS_MODULES."
    )

def test_pull_request_template_is_discoverable_by_agents() -> None:
    """Guard the PR template and the pointers agents rely on to find it.

    ``gh pr create --body`` bypasses GitHub's web-form template injection, so
    agents only follow the template if the file they load at session start
    (``AGENTS.md``) points at it. ``CLAUDE.md`` must keep importing ``AGENTS.md``
    so Claude Code sees the same rules. This pins the template's section
    headings and both pointers so neither can silently rot.
    """

    template = (PROJECT_ROOT / PR_TEMPLATE_PATH).read_text(encoding="utf-8")
    missing_sections = [
        section for section in PR_TEMPLATE_SECTIONS if section not in template
    ]
    assert not missing_sections, (
        f"{PR_TEMPLATE_PATH} is missing section heading(s): {missing_sections}"
    )
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert PR_TEMPLATE_PATH in agents, (
        f"AGENTS.md no longer references {PR_TEMPLATE_PATH}; agents discover the "
        "PR template through that pointer."
    )
    claude = (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in claude, (
        "CLAUDE.md no longer imports AGENTS.md; Claude Code would stop seeing the "
        "canonical agent guide."
    )


def test_repository_does_not_track_python_bytecode_artifacts() -> None:
    result = subprocess.run(
        ["git", "ls-files", "src", "tests", "docs"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_bytecode = sorted(
        line
        for line in result.stdout.splitlines()
        if "/__pycache__/" in line or line.endswith((".pyc", ".pyo"))
    )
    assert not tracked_bytecode, "\n".join(tracked_bytecode)
