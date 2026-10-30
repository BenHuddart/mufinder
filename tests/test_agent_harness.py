"""Repository harness checks for agent-legible development."""

from __future__ import annotations

import ast
import math
import re
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
    "docs/ui_text.rst",
)

INDEXED_DOCS = (
    "agent_harness",
    "architecture",
    "quality_and_automation",
    "compatibility_policy",
    "ui_text",
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
        "uep_wizard.py",
    }
)


# Modules whose job-staging widgets are expected to share one implementation of
# each cross-workflow concern (docs/architecture.rst, Shared workflow
# infrastructure). The ratchet below pins, for every private method name that
# appears in two or more of them, how many of them define it. No new shared name
# may appear, and the fix for a missing capability is to extend the helper in
# the shared home, not to copy a sibling's method.
WORKFLOW_WIDGET_MODULES = (
    "convergence_widget.py",
    "kpoint_convergence_widget.py",
    "relaxation_widget.py",
    "muon_relaxation_widget.py",
    "uep_widget.py",
    "hyperfine_coupling_widget.py",
    "quantum_corrections_widget.py",
    "energy_barriers_widget.py",
    "magnetic_field_widget.py",
    "magnetic_component_dialogs.py",
    "magnetic_texture_dialog.py",
    "magnetic_structure_builder.py",
    "magnetic_site_dialog.py",
    "magnetic_field_distribution_dialog.py",
    "demagnetizing_dialogs.py",
    "magnetic_analysis_dialogs.py",
)

# The Magnetic Fields page and dialog modules share these names only: each
# definition builds, loads or refreshes its own page or dialog, so there is no
# common behaviour to move.
# The dipolar-tensor dialog's ``_set_selector`` also fills its atom-index field
# and reports whether the selector matched, unlike the component dialogs' one.
PINNED_DUPLICATED_WIDGET_METHODS: dict[str, int] = {
    "_build_layout": 6,
    "_load_component": 2,
    "_refresh_preview": 2,
    "_set_selector": 2,
    "_sync_enabled_state": 2,
}


# Qt widget and GUI imports are confined to ``mufinder.app``. The layer packages
# may use ``PySide6.QtCore`` (signals, threads, settings) but never widgets, so
# services, workflow and science code stays testable without a window and the
# documented "services must not import PySide widgets" rule is executable.
# ``mufinder.main`` and ``mufinder.branding`` sit outside the layers and are
# allowed to (application entry point and icons).
FORBIDDEN_QT_MODULES_OUTSIDE_APP = ("PySide6.QtWidgets", "PySide6.QtGui")

NON_APP_LAYER_PACKAGES = ("domain", "castep_io", "persistence", "science", "workflow", "services")


# Coverage-companion test files that exist today. Coverage gaps are closed by
# behaviour tests in the module's own test file (AGENTS.md, Tests), so this
# list may only shrink as the files are merged into their siblings.
PINNED_COVERAGE_TEST_FILES = frozenset({})


# Physical constants have one home, ``mufinder.domain.constants``, which takes
# them from ``scipy.constants`` (AGENTS.md, Reuse before you write). It is also
# the only module that imports ``scipy.constants`` or ``ase.units``.
CONSTANTS_HOME = SOURCE_ROOT / "domain" / "constants.py"

# A number in source within this relative distance of a physical constant is a
# typed-out copy of it. Loose enough to catch a rounded copy such as 27.211.
CONSTANT_LITERAL_RTOL = 1e-4

# Modules that may type a constant out, each with its reason.
CONSTANT_LITERAL_ALLOWLIST = {
    # A saved-project migration must reproduce the factor older projects were
    # written with, so it keeps its historical value rather than tracking CODATA.
    SOURCE_ROOT / "persistence" / "migrations.py",
}

CONSTANT_SOURCE_MODULES = ("scipy.constants", "ase.units")

# Names that held a typed-out constant before the home existed. A module that
# assigns one of these, or a name the home defines, is redefining a constant.
RETIRED_CONSTANT_NAMES = frozenset(
    {"ANGSTROM_TO_M", "AMU_TO_KG", "BOHR_TO_ANG", "HBAR", "MU0", "MU0_OVER_4PI", "MU_B", "MU_N"}
)


def _python_files(package: str) -> list[Path]:
    return sorted((SOURCE_ROOT / package).rglob("*.py"))


def _main_window_submit_handler_methods() -> set[str]:
    """Return ``MainWindow`` method names matching the submit/materialize patterns."""

    path = SOURCE_ROOT / "app" / "main_window.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main_window = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
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
    assert len(agents.splitlines()) <= 140
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


_RST_HEADING_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{3,}$")


def _rst_section_body(text: str, title: str) -> str:
    """Return the text under heading ``title`` up to the next heading of any level."""

    lines = text.splitlines()
    start = next(
        index + 2
        for index in range(len(lines) - 1)
        if lines[index] == title and _RST_HEADING_UNDERLINE.match(lines[index + 1])
    )
    end = next(
        (
            index
            for index in range(start, len(lines) - 1)
            if lines[index].strip() and _RST_HEADING_UNDERLINE.match(lines[index + 1])
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _is_mufinder_module_path(dotted: str) -> bool:
    path = SOURCE_ROOT.parent.joinpath(*dotted.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def test_shared_homes_table_names_existing_modules() -> None:
    """Every shared home in docs/architecture.rst names a real module or package.

    The table under "Shared workflow infrastructure" is the inventory ``AGENTS.md``
    sends agents to before they add a helper, so a renamed or deleted home must
    not linger in it. The check reads the file system only; nothing is imported.
    """

    architecture = (PROJECT_ROOT / "docs" / "architecture.rst").read_text(encoding="utf-8")
    section = _rst_section_body(architecture, "Shared workflow infrastructure")
    homes = sorted(set(re.findall(r"``(mufinder(?:\.\w+)+)``", section)))
    assert homes, "No ``mufinder.*`` homes found under Shared workflow infrastructure."
    missing = [home for home in homes if not _is_mufinder_module_path(home)]
    assert not missing, (
        f"The shared-homes table in docs/architecture.rst (Shared workflow "
        f"infrastructure) has rotted: {missing} name no module or package under "
        "src/mufinder. Update the row to the module's current dotted path, or drop "
        "it if the home is gone."
    )


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


#: The modules that define the look of the application and may hold colour literals.
THEME_STYLING_HOMES = frozenset({"app/theme.py", "app/splash_screen.py"})

HEX_COLOUR_LITERAL = re.compile(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?\b")
TRANSPARENT_STYLESHEET = re.compile(r"\bbackground(?:-color)?\s*:\s*transparent\b")

#: Hex colours that must stay outside ``theme.py``, per module, with the reason.
PINNED_HEX_COLOUR_LITERALS: dict[str, tuple[frozenset[str], str]] = {
    "app/project_home_trail.py": (
        frozenset(
            {
                "#FFFFFF",
                "#5C341E",
                "#7A4728",
                "#A66A3A",
                "#9B5F32",
                "#B3723D",
                "#8A512C",
                "#5F371F",
                "#C88B55",
                "#D2A169",
                "#FFF8EA",
            }
        ),
        "the wooden signpost is a fixed illustration, like the splash screen, in both themes",
    ),
    "app/uep_wizard.py": (
        frozenset({"#dc2626", "#ffffff", "#1f1a17", "#FFFFFF", "#1d4ed8", "#0f766e", "#b91c1c"}),
        "#312 removes this wizard's plots and overlays; their colours are not worth moving first",
    ),
}

#: Modules that may still set a transparent stylesheet, with the reason.
PINNED_TRANSPARENT_STYLESHEETS: dict[str, str] = {
    "app/uep_wizard.py": "#312 removes this wizard's copies; the rule in theme.py covers the rest",
}


def test_styling_stays_in_the_theme_module() -> None:
    """Keep colours and transparency workarounds in ``theme.py``.

    A ``#rrggbb`` literal elsewhere does not follow the theme, and labels and
    checkboxes are already transparent through the application stylesheet.
    Containers use ``style_transparent_surface``. A module on an allow-list
    must still need it, so the lists can only shrink.
    """

    hex_offenders: dict[str, set[str]] = {}
    hex_seen: dict[str, set[str]] = {}
    transparent_offenders: list[str] = []
    transparent_seen: set[str] = set()
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        module = path.relative_to(SOURCE_ROOT).as_posix()
        if module in THEME_STYLING_HOMES:
            continue
        text = path.read_text(encoding="utf-8")
        literals = set(HEX_COLOUR_LITERAL.findall(text))
        allowed = PINNED_HEX_COLOUR_LITERALS.get(module, (frozenset(), ""))[0]
        if literals - allowed:
            hex_offenders[module] = literals - allowed
        if literals:
            hex_seen[module] = literals
        if TRANSPARENT_STYLESHEET.search(text):
            transparent_seen.add(module)
            if module not in PINNED_TRANSPARENT_STYLESHEETS:
                transparent_offenders.append(module)

    assert not hex_offenders, (
        f"Hex colours outside theme.py: {hex_offenders}. Take plot colours from "
        "plot_series_colours() and everything else from the palette in mufinder.app.theme."
    )
    assert not transparent_offenders, (
        f"Transparent stylesheets outside theme.py: {transparent_offenders}. Labels and "
        "checkboxes need none; use style_transparent_surface for a container."
    )
    stale_hex = {
        module: sorted(allowed - hex_seen.get(module, set()))
        for module, (allowed, _reason) in PINNED_HEX_COLOUR_LITERALS.items()
        if allowed - hex_seen.get(module, set())
    }
    assert not stale_hex, f"Remove these from PINNED_HEX_COLOUR_LITERALS: {stale_hex}"
    stale_transparent = set(PINNED_TRANSPARENT_STYLESHEETS) - transparent_seen
    assert not stale_transparent, (
        f"Remove these from PINNED_TRANSPARENT_STYLESHEETS: {sorted(stale_transparent)}"
    )


#: ``QMessageBox.question`` and the Yes/No standard buttons (``YesToAll`` and
#: ``NoToAll`` included) answer a question with "Yes" or "No" instead of the
#: action, and the static ``question`` keeps Qt's default button, which is
#: "Yes" when none is given.
YES_NO_PROMPT_PATTERN = re.compile(
    r"QMessageBox\.question\b|StandardButton\.(?:Yes|No)|QMessageBox\.(?:Yes|No)\b"
)


def test_app_confirmations_name_their_actions() -> None:
    """Forbid Yes/No message boxes in ``mufinder.app``; use the shared helpers.

    ``workflow_table_utils.confirm_destructive`` (delete, remove, reset, cancel,
    dismiss) and ``workflow_table_utils.confirm_choice`` (any other question)
    label every button with its action and make the safe choice the default
    and the Escape answer.
    """

    offenders = sorted(
        f"{path.relative_to(PROJECT_ROOT)}:{number}"
        for path in _python_files("app")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if YES_NO_PROMPT_PATTERN.search(line)
    )
    assert not offenders, (
        f"Yes/No message boxes found at {offenders}. Ask through "
        "workflow_table_utils.confirm_destructive or confirm_choice, with buttons "
        "that name the action."
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
    missing_sections = [section for section in PR_TEMPLATE_SECTIONS if section not in template]
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


def _private_method_names(path: Path) -> set[str]:
    """Return the single-underscore method names defined on classes in ``path``."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for member in node.body:
            if (
                isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and member.name.startswith("_")
                and not member.name.startswith("__")
            ):
                names.add(member.name)
    return names


def _duplicated_widget_methods() -> dict[str, int]:
    counts: dict[str, int] = {}
    for module in WORKFLOW_WIDGET_MODULES:
        for name in _private_method_names(SOURCE_ROOT / "app" / module):
            counts[name] = counts.get(name, 0) + 1
    return {name: count for name, count in counts.items() if count >= 2}


def test_workflow_widgets_do_not_grow_duplicated_private_methods() -> None:
    """Ratchet the private methods copied between workflow widgets.

    Each cross-workflow concern has one shared home (docs/architecture.rst,
    Shared workflow infrastructure). A private method that two or more widget
    modules define is a copy; the pinned count per name may only fall, and no
    new shared name may appear.
    """

    actual = _duplicated_widget_methods()
    new_names = sorted(set(actual) - set(PINNED_DUPLICATED_WIDGET_METHODS))
    assert not new_names, (
        f"Private method(s) {new_names} are now defined by two or more workflow "
        "widgets. Move the shared behaviour into its home module (see "
        "docs/architecture.rst, Shared workflow infrastructure) instead of copying it."
    )
    grown = sorted(
        f"{name}: {PINNED_DUPLICATED_WIDGET_METHODS[name]} -> {count}"
        for name, count in actual.items()
        if count > PINNED_DUPLICATED_WIDGET_METHODS[name]
    )
    assert not grown, (
        f"More workflow widgets now define {grown}. Use the shared helper instead of "
        "adding another copy."
    )
    stale = sorted(
        f"{name}: pinned {pinned}, actual {actual.get(name, 0)}"
        for name, pinned in PINNED_DUPLICATED_WIDGET_METHODS.items()
        if actual.get(name, 0) < pinned
    )
    assert not stale, (
        f"Duplication shrank for {stale}. Lower or remove the pinned counts in "
        "PINNED_DUPLICATED_WIDGET_METHODS so the ratchet keeps its bite."
    )


def _qt_modules_imported(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "PySide6":
                modules.update(f"PySide6.{alias.name}" for alias in node.names)
            else:
                modules.add(module)
    return modules


def test_qt_widgets_are_only_imported_by_the_app_layer() -> None:
    violations: list[str] = []
    for package in NON_APP_LAYER_PACKAGES:
        for path in _python_files(package):
            offending = sorted(
                module
                for module in _qt_modules_imported(path)
                if module.startswith(FORBIDDEN_QT_MODULES_OUTSIDE_APP)
            )
            if offending:
                rel_path = path.relative_to(PROJECT_ROOT)
                violations.append(f"{rel_path}: imports {', '.join(offending)}")

    assert not violations, "Qt widget/GUI imports outside mufinder.app:\n" + "\n".join(violations)


def test_coverage_companion_test_files_do_not_grow() -> None:
    """Pin the ``*_coverage.py`` test files; the set may only shrink."""

    actual = {
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "tests").rglob("*_coverage.py")
    }
    new_files = sorted(actual - PINNED_COVERAGE_TEST_FILES)
    assert not new_files, (
        f"New coverage-companion test file(s) {new_files}. Close coverage gaps with "
        "behaviour tests in the module's own test file instead."
    )
    merged_files = sorted(PINNED_COVERAGE_TEST_FILES - actual)
    assert not merged_files, f"{merged_files} no longer exist. Shrink PINNED_COVERAGE_TEST_FILES."


def _physical_constant_references() -> list[tuple[float, str]]:
    """Each physical constant MuFinder uses, with the home name to import instead."""

    from scipy.constants import angstrom, c, e, epsilon_0, h, hbar, k, mu_0, physical_constants

    def codata(name: str) -> float:
        return physical_constants[name][0]

    muon_gamma = abs(2.0 * codata("muon mag. mom.") / hbar)
    cm_inv_to_ev = h * c * 100.0 / e
    return [
        (h, "PLANCK_CONSTANT_J_S"),
        (hbar, "HBAR_J_S"),
        (c, "LIGHT_SPEED_M_S"),
        (e, "ELEMENTARY_CHARGE_C"),
        (k, "BOLTZMANN_J_K"),
        (k / e, "BOLTZMANN_J_K / ELEMENTARY_CHARGE_C"),
        (mu_0, "MU0_T_M_PER_A"),
        (epsilon_0, "VACUUM_PERMITTIVITY_F_PER_M"),
        (codata("Bohr magneton"), "BOHR_MAGNETON_J_PER_T"),
        (codata("nuclear magneton"), "NUCLEAR_MAGNETON_J_PER_T"),
        (codata("atomic mass constant"), "ATOMIC_MASS_KG"),
        (codata("Bohr radius"), "BOHR_RADIUS_ANGSTROM * METRES_PER_ANGSTROM"),
        (codata("Bohr radius") / angstrom, "BOHR_RADIUS_ANGSTROM"),
        (angstrom / codata("Bohr radius"), "1 / BOHR_RADIUS_ANGSTROM"),
        (codata("Hartree energy in eV"), "HARTREE_EV"),
        (codata("muon mass in u"), "MUON_MASS_AMU"),
        (codata("muon mass"), "MUON_MASS_KG"),
        (muon_gamma, "MUON_GAMMA_RAD_S_T"),
        (muon_gamma / (2.0 * math.pi) / 1e6, "MUON_GAMMA_OVER_2PI_MHZ_T"),
        (cm_inv_to_ev, "PLANCK_CONSTANT_J_S * LIGHT_SPEED_M_S * 100 / ELEMENTARY_CHARGE_C"),
        (1.0 / cm_inv_to_ev, "ELEMENTARY_CHARGE_C / (PLANCK_CONSTANT_J_S * LIGHT_SPEED_M_S * 100)"),
    ]


def _is_power_of_ten(value: float) -> bool:
    exponent = math.log10(abs(value))
    return math.isclose(exponent, round(exponent), abs_tol=1e-12)


def _typed_out_constants(path: Path, references: list[tuple[float, str]]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or type(node.value) not in (int, float):
            continue
        value = float(node.value)
        # Powers of ten (1e-7, 1e-10) double as tolerances and SI prefixes.
        if value == 0.0 or _is_power_of_ten(value):
            continue
        for reference, home_name in references:
            if math.isclose(value, reference, rel_tol=CONSTANT_LITERAL_RTOL):
                found.append(f"line {node.lineno}: {node.value!r} is {home_name}")
                break
    return found


def _constant_scan_files() -> list[Path]:
    """The application source and the scripts that build its documentation data."""

    return sorted(SOURCE_ROOT.rglob("*.py")) + sorted((PROJECT_ROOT / "scripts").rglob("*.py"))


def test_physical_constants_are_not_typed_out_outside_their_home() -> None:
    """No module but ``mufinder.domain.constants`` types out a physical constant."""

    references = _physical_constant_references()
    violations: list[str] = []
    for path in _constant_scan_files():
        if path == CONSTANTS_HOME or path in CONSTANT_LITERAL_ALLOWLIST:
            continue
        rel_path = path.relative_to(PROJECT_ROOT)
        violations.extend(
            f"{rel_path} {finding}" for finding in _typed_out_constants(path, references)
        )
    assert not violations, (
        "Physical constants typed out outside mufinder.domain.constants. Import the "
        "named constant from mufinder.domain.constants (or build the expression from "
        "its constants) instead:\n" + "\n".join(violations)
    )


def _constant_source_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            modules = [module, *(f"{module}.{alias.name}" for alias in node.names)]
        else:
            continue
        if any(module in CONSTANT_SOURCE_MODULES for module in modules):
            found.append(f"line {node.lineno}")
    return found


def test_constant_libraries_are_only_imported_by_the_constants_home() -> None:
    """``scipy.constants`` and ``ase.units`` are read in ``mufinder.domain.constants`` only."""

    violations = [
        f"{path.relative_to(PROJECT_ROOT)} {finding}"
        for path in _constant_scan_files()
        if path != CONSTANTS_HOME
        for finding in _constant_source_imports(path)
    ]
    assert not violations, (
        "scipy.constants or ase.units imported outside mufinder.domain.constants. Add "
        "the constant there and import it from the home instead:\n" + "\n".join(violations)
    )


def _module_level_names(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[tuple[str, int]] = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        names.extend((target.id, node.lineno) for target in targets if isinstance(target, ast.Name))
    return names


def test_constant_names_are_only_defined_in_their_home() -> None:
    """No module redefines a constant the home holds, whatever value it writes.

    The value scan skips powers of ten and expressions, so ``MU0_OVER_4PI = 1e-7``
    or ``MU0 = 4 * pi * 1e-7`` would pass it; this check catches them by name.
    """

    home_names = {name for name, _ in _module_level_names(CONSTANTS_HOME) if name.isupper()}
    forbidden = home_names | RETIRED_CONSTANT_NAMES
    violations = [
        f"{path.relative_to(PROJECT_ROOT)} line {lineno}: {name}"
        for path in _constant_scan_files()
        if path != CONSTANTS_HOME and path not in CONSTANT_LITERAL_ALLOWLIST
        for name, lineno in _module_level_names(path)
        if name.lstrip("_") in forbidden
    ]
    assert not violations, (
        "Module-level names that redefine a physical constant outside "
        "mufinder.domain.constants. Import the constant from the home instead:\n"
        + "\n".join(violations)
    )


#: Text the UI style (``docs/ui_text.rst``) rules out: ``(s)`` instead of a count
#: that agrees with its number (``mufinder.domain.text.plural``), three dots
#: instead of the one-character ellipsis, and ``->`` instead of ``Setup > CASTEP``
#: in a menu path (or ``→`` for a change from one value to another).
UI_TEXT_FORBIDDEN = (
    (re.compile(r"\w\(s\)"), "'(s)': write the count with mufinder.domain.text.plural"),
    (re.compile(r"\.\.\."), "'...': use the one-character ellipsis '…'"),
    (re.compile(r"->"), "'->': write menu paths as 'Setup > CASTEP', a change as '→'"),
)

#: ``numpy.einsum`` subscripts use ``...`` and ``->`` as syntax, not as text.
EINSUM_SUBSCRIPTS = re.compile(r"^[a-z.,]+->[a-z.]*$")


def _ui_text_violations(source: str) -> list[tuple[int, str]]:
    """Return ``(line, problem)`` for each string literal that breaks the UI style.

    Every string literal counts except docstrings and einsum subscripts, so a
    message built in ``services`` or ``science`` and shown by the app is held to
    the same rules as a label in ``app``. An f-string is read whole, with each
    replacement field standing in as a word, so ``f"{noun}(s)"`` is caught.
    """

    tree = ast.parse(source)
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skipped.add(id(first.value))
        elif isinstance(node, ast.JoinedStr):
            skipped.update(id(value) for value in node.values)
    texts: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            texts.append(
                (
                    node.lineno,
                    "".join(
                        value.value if isinstance(value, ast.Constant) else "x"
                        for value in node.values
                    ),
                )
            )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skipped
        ):
            texts.append((node.lineno, node.value))
    return [
        (line, problem)
        for line, text in texts
        if not EINSUM_SUBSCRIPTS.match(text)
        for pattern, problem in UI_TEXT_FORBIDDEN
        if pattern.search(text)
    ]


def test_ui_text_check_flags_text_and_skips_docstrings_and_einsum() -> None:
    source = (
        '"""Module docstring with job(s)... and a -> b."""\n'
        "def f(n):\n"
        '    """Docstring: Setup -> CASTEP..."""\n'
        '    np.einsum("...i,ij->...j", a, b)\n'
        '    label = "Export..."\n'
        '    return f"Submitted {n} job(s) in Setup -> CASTEP"\n'
        'deleted = f"Deleted {n} {noun}(s)"\n'
        'swap = "Fe->Co"\n'
    )

    assert sorted(_ui_text_violations(source)) == sorted(
        [
            (5, UI_TEXT_FORBIDDEN[1][1]),
            (6, UI_TEXT_FORBIDDEN[0][1]),
            (6, UI_TEXT_FORBIDDEN[2][1]),
            (7, UI_TEXT_FORBIDDEN[0][1]),
            (8, UI_TEXT_FORBIDDEN[2][1]),
        ]
    )


def test_ui_text_follows_the_style_page() -> None:
    """No ``(s)``, ``...`` or ``->`` in any string MuFinder can show (``docs/ui_text.rst``)."""

    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line}: {problem}"
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        for line, problem in _ui_text_violations(path.read_text(encoding="utf-8"))
    ]
    assert not violations, "UI text breaks docs/ui_text.rst:\n" + "\n".join(violations)
