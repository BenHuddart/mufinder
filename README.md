# MuFinder

MuFinder is a PySide6 desktop application for muon stopping-site workflows, CASTEP-oriented simulation setup, and structure-comparison analysis.

## Documentation and Releases

- Documentation: <https://benhuddart.github.io/mufinder/>
- Releases: <https://github.com/BenHuddart/mufinder/releases>
- Repository: <https://github.com/BenHuddart/mufinder>

## What MuFinder Includes

- Host-structure import and workflow management
- Muon candidate-site generation
- Structural muon relaxation workflows with project-integrated result tracking
- Quantum Corrections for Gamma-point phonon calculations on relaxed muon sites,
	including per-row parameter configuration and parsed zero-point/witness output
- Energy Barriers workflows for symmetry-distinct path generation,
	transition-state search setup, NEB refinement, and inline path inspection
- CASTEP parameter and cell editing for the supported MuFinder subset
- Local and remote CASTEP job orchestration
- Pristine-relaxation comparison tables and CSV export
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
	hop in the matching undistorted supercell, and then drives both
	transition-state search and NEB job generation from the project page.
- Completed Energy Barriers NEB jobs can be inspected inside the application
	with a synchronized reaction-coordinate profile, slider-driven image
	navigation, and shared 3D structure viewer.

## CASTEP Metadata Contract

`castep_keywords.json` is intentionally not bundled with MuFinder and must not be committed to this repository.

MuFinder supports two user-managed runtime inputs for CASTEP metadata discovery:

- `MUFINDER_CASTEP_KEYWORDS_JSON` or `CASTEP_KEYWORDS_JSON`
- `MUFINDER_CASTEP_ROOT` or `CASTEP_ROOT`

Users can either point MuFinder at an existing generated `castep_keywords.json` or generate one locally from a CASTEP installation. The packaged application includes only MuFinder's conservative starter catalog fallback for supported keywords.

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

Install with development tooling:

```bash
python -m pip install -e ".[dev]"
```

The structure viewer uses the PyVista/VTK + `pyvistaqt` stack.

Run the app:

```bash
python -m mufinder.main
```

## Development Checks

The project follows GitHub Flow. Branch short-lived `feature/*` branches off `main` and open pull requests **directly into `main`**; CI runs the full quality gates on every PR and merges are blocked until they pass.

Run the full test suite manually when needed:

```bash
python -m pytest
```

Build the Sphinx documentation:

```bash
python -m sphinx -W -b html docs docs/_build/html
```

Cut a release by bumping the version and pushing a tag from `main`:

```bash
git checkout main
git pull
semantic-release version          # stamps pyproject.toml + CHANGELOG.md, tags vX.Y.Z
git push --follow-tags
```

Pushing the `vX.Y.Z` tag triggers the `Release Publish` workflow, which builds the Windows and macOS binaries and publishes the GitHub release from the tag.

## Binary Distributions

GitHub Releases publish:

- Windows installers built from PyInstaller `onedir` bundles and wrapped with Inno Setup
- macOS `.app` bundles packaged into drag-to-Applications DMGs (**Apple Silicon / arm64 only**)

These binaries ship MuFinder itself, but CASTEP and MPI remain external user-supplied dependencies.

> **Intel Mac (x86_64) users:** pre-built binaries are not provided for Intel Macs. Install from source instead — see the [Install From Source](#install-from-source) section above.

For ad hoc validation on any branch, use the `Preview Build` workflow to produce downloadable artifacts without publishing a release.
