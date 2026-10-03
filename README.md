# MuFinder

MuFinder is a PySide6 desktop application for muon stopping-site workflows, CASTEP-oriented simulation setup, and structure-comparison analysis.

## Documentation and Releases

- Documentation: <https://benhuddart.github.io/mufinder/>
- Releases: <https://github.com/BenHuddart/mufinder/releases>
- Repository: <https://github.com/BenHuddart/mufinder>

## What MuFinder Includes

- Host-structure import, a reusable supercell editor, and project workflow management
- Cut-off energy and k-point convergence testing with project-wide default settings
- Pristine host relaxation with experiment-versus-DFT comparison tables and CSV export
- Unperturbed electrostatic potential (UEP) screening with minimum clustering
- Muon candidate-site generation
- Structural muon relaxation workflows with project-integrated result tracking,
	clustering-based relaxation analysis, and import of relaxations from the legacy MuFinder
- Magnetic Fields: magnetic-structure building, dipolar local fields and field
	distributions, dipolar tensors, and nuclear dipolar linewidths
- Quantum Corrections for Gamma-point phonon calculations on relaxed muon sites,
	including per-row parameter configuration and parsed zero-point/witness output
- Energy Barriers workflows for symmetry-distinct path generation, climbing-image
	NEB calculations seeded directly from two relaxed endpoints, and inline path inspection
- CASTEP parameter and cell editing for the supported MuFinder subset
- Local and remote (HPC cluster) CASTEP job orchestration with a job dashboard
- A project journal of provenance events
- Responsive 3D structure viewing and UEP isosurface rendering via the PyVista/VTK backend

## Workflow Coverage

MuFinder supports the core end-to-end workflow from relaxed muon-site
analysis into higher-level follow-on calculations.

- Quantum Corrections turns completed relaxed muon sites into Gamma-point
	phonon jobs, exposes the most relevant phonon controls directly in the UI,
	and keeps additional supported CASTEP parameters available through a
	searchable advanced parameter surface.
- Energy Barriers starts from two completed relaxed endpoints, generates
	symmetry-distinct migration paths for that ordered pair, lets you preview the
	hop in the matching undistorted supercell, and then builds climbing-image NEB
	jobs for those paths from the project page (no separate transition-state
	search step is needed).
- Completed Energy Barriers NEB jobs can be inspected inside the application
	with a synchronized reaction-coordinate profile, slider-driven image
	navigation, and shared 3D structure viewer.

## CASTEP Metadata Contract

`castep_keywords.json` is intentionally not bundled with MuFinder and must not be committed to this repository.

MuFinder needs a keyword catalog to recognise every keyword your CASTEP version reads and writes. Without one it runs on a small built-in starter set, offers fewer parameters, and can fail to load results that contain keywords outside that set (for example `QUANTISATION_AXIS`, which CASTEP writes after a Hubbard U calculation).

Open `Setup > CASTEP` and either generate the catalog from a cluster or a local CASTEP, or import a `castep_keywords.json` you already have. MuFinder validates it and keeps its own copy, so there is no location to choose. The status bar shows which catalog is in use, and MuFinder warns at startup and before submitting jobs when none is loaded.

Developers can override the installed catalog with environment variables:

- `MUFINDER_CASTEP_KEYWORDS_JSON` or `CASTEP_KEYWORDS_JSON` (a catalog file)
- `MUFINDER_CASTEP_ROOT` or `CASTEP_ROOT` (a CASTEP source tree)

See `docs/castep_keyword_catalog.rst` for the full behaviour.

## Install From Source

Clone and install locally:

```bash
git clone git@github.com:BenHuddart/mufinder.git
cd mufinder
python -m pip install .
```

Install directly from Git for users with repository access:

```bash
python -m pip install "git+ssh://git@github.com/BenHuddart/mufinder.git"
```

Install with development tooling. The test suite also needs the
`scientific-backends` extra (`muesr`, `muLFC`, `Soprano`), which the Magnetic
Fields, UEP, and relaxation-analysis features use at runtime:

```bash
uv sync --extra dev --extra scientific-backends
# or: python -m pip install -e ".[dev,scientific-backends]"
```

The structure viewer uses the PyVista/VTK + `pyvistaqt` stack.

Run the app:

```bash
python -m mufinder.main
```

## Development Checks

The project follows GitHub Flow. Branch short-lived `feature/*` branches off `main` and open pull requests **directly into `main`**; CI runs the full quality gates on every PR and merges are blocked until they pass.

Run the full test suite manually when needed. Locally, cap parallelism at four
workers (each worker loads Qt, and `-n auto` can exhaust the machine); CI uses
`-n auto`:

```bash
python -m pytest -n 4 --dist load
```

Run a single test file without coverage while iterating:

```bash
python -m pytest --no-cov tests/science/test_supercell.py
```

Build the Sphinx documentation:

```bash
python -m sphinx -W -b html docs docs/_build/html
```

Cut a release from `main` (branch protection requires a PR for the version bump commit):

```bash
git checkout main && git pull
uvx "python-semantic-release>=10.5,<11" version   # bumps pyproject.toml + CHANGELOG.md, commits + tags locally (push fails — expected)
git checkout -b release/vX.Y.Z                    # branch off the version bump commit
git checkout main && git reset --hard origin/main  # restore local main
git push origin release/vX.Y.Z                    # open a PR, wait for CI + merge
git push origin vX.Y.Z                            # push the tag — triggers the Release Publish workflow
```

Pushing the `vX.Y.Z` tag triggers the `Release Publish` workflow, which builds the Windows and macOS binaries and publishes the GitHub release from the tag.

## Binary Distributions

GitHub Releases publish:

- Windows installers built from PyInstaller `onedir` bundles and wrapped with Inno Setup
- macOS `.app` bundles packaged into drag-to-Applications DMGs (**Apple Silicon / arm64 only**)

These binaries ship MuFinder itself, but CASTEP and MPI remain external user-supplied dependencies.

> **Intel Mac (x86_64) users:** pre-built binaries are not provided for Intel Macs. Install from source instead — see the [Install From Source](#install-from-source) section above.

For ad hoc validation on any branch, use the `Preview Build` workflow to produce downloadable artifacts without publishing a release.
