# MuFinder

MuFinder is a PySide6 desktop application for muon stopping-site workflows, CASTEP-oriented simulation setup, and structure-comparison analysis.

## Documentation and Releases

- Documentation: <https://benhuddart.github.io/mufinder/>
- Releases: <https://github.com/BenHuddart/mufinder/releases>
- Repository: <https://github.com/BenHuddart/mufinder>

## What MuFinder Includes

- Host-structure import and workflow management
- Muon candidate-site generation
- CASTEP parameter and cell editing for the supported MuFinder subset
- Local and remote CASTEP job orchestration
- Pristine-relaxation comparison tables and CSV export
- Responsive 3D structure viewing and UEP isosurface rendering via the PyVista/VTK backend

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

The structure viewer uses the PyVista/VTK + `pyvistaqt` stack. The legacy
Matplotlib atom-viewer backend has been removed.

Run the app:

```bash
python -m mufinder.main
```

## Development Checks

Daily work is expected to land on `develop`. Full quality gates are reserved for the final release PR into `main`.

Run the full test suite manually when needed:

```bash
python -m pytest
```

Build the Sphinx documentation:

```bash
python -m sphinx -W -b html docs docs/_build/html
```

Prepare a release branch once `develop` is stable:

```bash
git checkout develop
git pull
```

Then use the `Prepare Release` GitHub Actions workflow to create `release/vX.Y.Z`, stamp the version, open the PR to `main`, and enable auto-merge once the required checks pass.
That workflow expects a repository variable named `RELEASE_APP_ID` and a repository secret named `RELEASE_APP_PRIVATE_KEY`.
Once the release PR merges, the publish workflow tags and publishes from `main`, merges `main` back into `develop`, and deletes the merged `release/vX.Y.Z` branch automatically.

## Binary Distributions

GitHub Releases will publish:

- Windows installers built from PyInstaller `onedir` bundles and wrapped with NSIS
- macOS `.app` bundles packaged into drag-to-Applications DMGs

These binaries ship MuFinder itself, but CASTEP and MPI remain external user-supplied dependencies.

For ad hoc validation on `develop` or a release branch, use the `Preview Build` workflow to produce downloadable artifacts without publishing a release.
