"""Helpers for persisted CASTEP executable and keyword metadata configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil

from PySide6.QtCore import QSettings, QStandardPaths

from mufinder.castep_io.keyword_catalog_file import (
    KeywordCatalogSummary,
    read_keyword_catalog_file,
    write_keyword_catalog_file,
)
from mufinder.castep_io.schema import SETUP_KEYWORDS_JSON_ENV_VAR, refresh_keyword_catalog


CASTEP_EXECUTABLE_KEY = "castep/executable_path"
CASTEP_KEYWORDS_JSON_KEY = "castep/keywords_json_path"
CASTEP_MPIRUN_KEY = "castep/mpirun_path"
MANAGED_KEYWORDS_JSON_NAME = "castep_keywords.json"


@dataclass(frozen=True, slots=True)
class CastepSetup:
    """Persisted CASTEP integration paths."""

    executable_path: str = ""
    keywords_json_path: str = ""
    mpirun_path: str = ""


def load_castep_setup(settings: QSettings) -> CastepSetup:
    """Load CASTEP integration settings from QSettings."""

    executable_path = str(settings.value(CASTEP_EXECUTABLE_KEY, "", str) or "").strip()
    keywords_json_path = str(settings.value(CASTEP_KEYWORDS_JSON_KEY, "", str) or "").strip()
    mpirun_path = str(settings.value(CASTEP_MPIRUN_KEY, "", str) or "").strip()
    return CastepSetup(
        executable_path=executable_path,
        keywords_json_path=keywords_json_path,
        mpirun_path=mpirun_path,
    )


def save_castep_setup(settings: QSettings, setup: CastepSetup) -> None:
    """Persist CASTEP integration settings to QSettings."""

    settings.setValue(CASTEP_EXECUTABLE_KEY, setup.executable_path.strip())
    settings.setValue(CASTEP_KEYWORDS_JSON_KEY, setup.keywords_json_path.strip())
    settings.setValue(CASTEP_MPIRUN_KEY, setup.mpirun_path.strip())


def discover_mpirun() -> str | None:
    """Locate an ``mpirun`` executable on this machine if available."""

    discovered = shutil.which("mpirun")
    if discovered:
        return str(Path(discovered).expanduser().resolve())

    candidates = (
        "/usr/bin/mpirun",
        "/usr/local/bin/mpirun",
        "/opt/homebrew/bin/mpirun",
        "/opt/local/bin/mpirun",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    return None


def _app_data_directory() -> Path:
    """Return the per-user directory MuFinder keeps its own data files in."""

    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if location:
        return Path(location)
    return Path.home() / ".mufinder"


def managed_keywords_json_path() -> Path:
    """Return where MuFinder keeps its own copy of the CASTEP keyword catalog.

    Generated and imported catalogs are both stored here, so users never have to
    choose a location and the catalog cannot disappear because a file elsewhere
    was moved or deleted.
    """

    return _app_data_directory() / MANAGED_KEYWORDS_JSON_NAME


def apply_castep_environment(setup: CastepSetup) -> None:
    """Expose persisted CASTEP configuration through environment variables.

    The saved catalog path is published through a setup-specific variable, not
    ``MUFINDER_CASTEP_KEYWORDS_JSON``: that one stays a user-controlled override
    which takes precedence over the catalog MuFinder manages.
    """

    keywords_json = setup.keywords_json_path.strip()

    if keywords_json:
        os.environ[SETUP_KEYWORDS_JSON_ENV_VAR] = keywords_json
    else:
        os.environ.pop(SETUP_KEYWORDS_JSON_ENV_VAR, None)


def install_keyword_catalog(source_path: str | Path, settings: QSettings) -> KeywordCatalogSummary:
    """Validate a keyword catalog, copy it into MuFinder's data directory, and use it.

    This is the single route by which a catalog becomes the configured one,
    whether it was just generated or is an existing file the user chose. The
    copy is written atomically, the saved setup is pointed at it, and the
    running schema is rebuilt, so the change takes effect immediately.

    Raises:
        KeywordCatalogValidationError: If ``source_path`` is not a complete
            CASTEP keyword catalog. Nothing is changed in that case.
    """

    payload, summary = read_keyword_catalog_file(source_path, require_complete=True)
    managed_path = write_keyword_catalog_file(payload, managed_keywords_json_path())

    setup = load_castep_setup(settings)
    updated = CastepSetup(
        executable_path=setup.executable_path,
        keywords_json_path=str(managed_path),
        mpirun_path=setup.mpirun_path,
    )
    save_castep_setup(settings, updated)
    apply_castep_environment(updated)
    refresh_keyword_catalog(allow_fallback=True)
    return summary


def apply_saved_castep_environment(settings: QSettings | None = None) -> CastepSetup:
    """Load persisted CASTEP settings, publish them, and load the saved catalog."""

    effective_settings = settings if settings is not None else QSettings("MuFinder", "MuFinder")
    setup = load_castep_setup(effective_settings)
    apply_castep_environment(setup)
    # Importing this module already imported the schema, which built itself
    # before the saved catalog path was published. Rebuild it now, or the
    # application starts on the starter catalog despite an installed one.
    refresh_keyword_catalog(allow_fallback=True)
    return setup


def build_castep_keyword_command(executable_path: str, mpirun_path: str = "") -> str:
    """Build the shell command ASE runs to query CASTEP help.

    ASE's ``create_castep_keywords`` invokes ``<command> -help ...`` through a
    shell. MPI-built CASTEP binaries (e.g. ``castep.mpi``) hang indefinitely when
    executed directly, because they block waiting for an MPI runtime. When an MPI
    launcher is configured we therefore wrap the executable in ``mpirun -np 1`` —
    which also runs serial binaries correctly — mirroring the cluster keyword
    generation path.
    """

    executable = executable_path.strip()
    mpirun = mpirun_path.strip()
    if mpirun:
        return " ".join(shlex.quote(part) for part in (mpirun, "-np", "1", executable))
    return executable


def generate_castep_keywords_json(
    *, executable_path: str, output_path: str | Path, mpirun_path: str = ""
) -> Path:
    """Generate a CASTEP keyword JSON file using ASE's supported extraction helper."""

    executable = executable_path.strip()
    if not executable:
        raise ValueError("A CASTEP executable path is required to generate castep_keywords.json.")

    output = Path(output_path).expanduser()
    if output.suffix.lower() != ".json":
        raise ValueError("CASTEP keywords output path must end with .json.")

    output.parent.mkdir(parents=True, exist_ok=True)

    from ase.calculators.castep import create_castep_keywords

    create_castep_keywords(
        build_castep_keyword_command(executable, mpirun_path),
        filename=output.name,
        path=str(output.parent),
        force_write=True,
    )

    if not output.exists():
        raise RuntimeError(f"CASTEP keyword generation did not produce {output}.")

    return output.resolve()
