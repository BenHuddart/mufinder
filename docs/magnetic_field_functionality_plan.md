# Magnetic Field Functionality Plan

This plan tracks the staged implementation of MuFinder's magnetic-field workflow. The workflow calculates magnetic fields, or distributions of magnetic fields, at candidate muon stopping sites using magnetic structures built with the propagation-vector formalism.

## Goals

- Reach feature parity with the first MuFinder magnetic-field workflow.
- Let users define magnetic structures from propagation vectors and complex Fourier coefficients.
- Support commensurate local-field calculations and incommensurate field distributions.
- Allow symmetry-equivalent muon sites and muon-induced distortion corrections to be used together.
- Move beyond single-q incommensurate structures.
- Use `muesr` and `muLFC` as optional local-field backends, with clear user-facing errors when a platform or environment does not provide them.
- Provide comprehensive tests and theory documentation.

## Phase 1: Magnetic Structure Model and Builder

Status: in progress.

Implemented:

- A JSON-safe propagation-vector magnetic-structure model in `mufinder.science.magnetic_structure`.
- Complex Fourier coefficients, q-basis vectors, component selectors, atom-resolved Fourier expansion, supercell moment evaluation, and muesr export data.
- Legacy conversion from the current host-structure scalar spin metadata.
- A Magnetic Fields workflow page that launches a separate Magnetic Structure Builder dialog.
- A builder dialog with a component table, legacy import, and component/full-structure preview.
- Viewer support for vector magnetic-moment arrows, so helices, cycloids, and conical helices can be previewed as real vector moments.
- Delete and Backspace shortcuts for component removal, matching the project macOS delete-key policy.
- Multi-q texture creation with square and hexagonal skyrmion presets, custom Fourier texture terms, vector/magnitude preview, opt-in normalization, and pruning.

Next:

- Add duplication, enable/disable toggles, and q-basis editing/inspection.
- Store the selected demagnetizing-field option in durable project state.
- Continue the builder review tracked in `docs/magnetic_structure_builder_review.md`.

## Phase 2: Field Calculation Engine

Status: foundation implemented.

Implemented:

- Guarded imports for `muesr` field engines, with `muesr` and `muLFC` installed as optional scientific backends when the target platform provides them.
- Commensurability classification.
- Finite dipole-field helper for tests and distortion corrections.
- Commensurate field calculation wrapper, dipolar tensor wrapper, equivalent muon-site expansion, and incommensurate sampling scaffolding.
- Magnetic Fields UI support for selecting muon sites from UEP clustering, structural-relaxation clustering, or a custom fractional coordinate.
- Structural-relaxation cluster selections can carry a distorted-structure reference for future muon-induced displacement corrections.
- Magnetic Fields UI dispatch for commensurate field values and incommensurate distribution summaries, with dependency errors surfaced in-page.
- Advanced demagnetizing-field options for ellipsoids and rectangular prisms, including tensor calculation, state persistence, and PyVista sample-shape preview.
- Demagnetizing-field tensors are applied to both commensurate finite fields and incommensurate sampled distributions, using the sampled Lorentz field at each phase point.

Next:

- Complete the production UI for calculation output modes, distortion correction execution, and run/result persistence.
- Add contact-field and Lorentz-field options to the calculation form.
- Add tensor calculation mode for commensurate structures.
- Surface dependency errors as actionable UI messages.
- Persist calculation results as structured MuFinder results.

## Phase 3: Distortion Corrections

Status: not yet wired into the application UI.

Planned:

- Borrow and improve the `mag2spectra` distortion-correction machinery without adding `mag2spectra` as a dependency.
- Allow distortion corrections and symmetry-equivalent muon sites together.
- Add validation for matching undistorted/distorted structures, finite correction cells, and atom mapping.
- Test against simple analytic and regression structures.

## Phase 4: Multi-q and Incommensurate Distributions

Status: multi-q phase sampling and UI controls implemented; remaining work focuses on convergence validation and result persistence.

Implemented:

- Generalized phase sampling for multiple independent q vectors.
- Per-component incommensurate lookup tables are combined through each component's `q_coeff`, so independent q-basis rows are sampled as independent phase dimensions.
- `Auto`, `Grid`, and `Monte Carlo` sampling controls are exposed in the calculation settings dialog.
- Incommensurate distribution results include sampled field vectors, magnitude histograms, active q-index metadata, sampling mode, and actual phase-sample count.

Planned:

- Detect commensurate approximants and report finite field lists when appropriate.
- Add user-facing convergence guidance for phase lookup resolution, Monte Carlo sample count, and histogram bins.
- Persist calculation results as structured MuFinder results.

## Phase 5: Documentation and Validation

Status: initial documentation added.

Planned:

- Expand user-facing theory notes with worked examples.
- Add screenshots once the calculation UI is complete.
- Add regression tests for muesr-backed calculations behind dependency guards.
- Add compatibility tests for old MuFinder-style magnetic metadata.
