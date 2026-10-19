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
git clone https://github.com/BenHuddart/mufinder.git
cd mufinder
python -m pip install .
```

Install directly from Git:

```bash
python -m pip install "git+https://github.com/BenHuddart/mufinder.git"
```

Install with development tooling. The test suite also needs the
`scientific-backends` extra (`muesr`, `muLFC`, `Soprano`), which the Magnetic
Fields, UEP, and relaxation-analysis features use at runtime:

```bash
uv sync --extra dev --extra scientific-backends
# or: make install
```

The structure viewer uses the PyVista/VTK + `pyvistaqt` stack.

Run the app:

```bash
python -m mufinder.main
```

## Development Checks

The project follows GitHub Flow. Branch short-lived `feature/*` branches off `main` and open pull requests **directly into `main`**; CI runs the full quality gates on every PR and merges are blocked until they pass.

The `Makefile` defines the local commands; see
[Quality and Automation](https://benhuddart.github.io/mufinder/quality_and_automation.html)
for every target.

```bash
make test                                              # full suite with coverage; foreground, 4 workers
make test-fast FILES="tests/science/test_supercell.py"  # focused tests without coverage
make docs                                              # Sphinx build, warnings as errors
```

Releases are cut by pushing a `vX.Y.Z` tag, which triggers the `Release Publish` workflow to build the Windows and macOS binaries and publish the GitHub release. The version-bump steps are in the [Automation policy](https://benhuddart.github.io/mufinder/quality_and_automation.html#automation-policy).

## Binary Distributions

GitHub Releases publish:

- Windows installers built from PyInstaller `onedir` bundles and wrapped with Inno Setup
- macOS `.app` bundles packaged into drag-to-Applications DMGs (**Apple Silicon / arm64 only**)

These binaries ship MuFinder itself, but CASTEP and MPI remain external user-supplied dependencies.

> **Intel Mac (x86_64) users:** pre-built binaries are not provided for Intel Macs. Install from source instead — see the [Install From Source](#install-from-source) section above.

For ad hoc validation on any branch, use the `Preview Build` workflow to produce downloadable artifacts without publishing a release.

## Contributing

Bug reports and feature requests are welcome from everyone: please
[open an issue](https://github.com/BenHuddart/mufinder/issues/new/choose).
Pull requests are currently accepted only from repository collaborators. See
[CONTRIBUTING.md](CONTRIBUTING.md) for details and for how to get in touch
about becoming a collaborator.

## License

MuFinder is free software released under the GNU General Public License,
version 3 or (at your option) any later version. See [LICENSE](LICENSE).
CASTEP is separate, proprietary software that MuFinder drives but does not
include; obtain it under its own licence.
