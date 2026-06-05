# Magnetic Structure Builder Review

This review captures the state after splitting the magnetic-structure builder out of the Magnetic Fields workflow page.

## Current Shape

- The Magnetic Fields page is now an entry point for field-calculation setup.
- The magnetic-structure builder opens as a separate dialog from that page.
- The builder owns component creation, component removal, legacy host-spin import, the component table, supercell preview controls, atom selection, and the live structure viewer.
- The builder follows the same explicit viewer shutdown pattern used by the supercell dialogs and is created fresh each time it is opened.
- Changes made in the builder are applied to the project only when the dialog is accepted.
- The scientific data model still stores complex Fourier coefficients and propagation-vector basis directions in a muesr-compatible form.

## Alignment With Initial Requirements

Implemented or partly implemented:

- Components can be added to a table.
- Existing components can be edited from the table, including label, type, target selector, q vector, phase, and preset-specific parameters.
- Selecting rows previews those components; no selection previews the full magnetic structure.
- Component targets can include all atoms, species, symmetry-equivalent atom groups, or currently selected atoms.
- The add-component dialog now presents those target choices as adaptive radio-card options.
- The builder can seed a model from existing host spin metadata.
- Collinear, spin-density-wave, helix, cycloid, and conical presets are separate builder paths with regression tests for their Fourier coefficients.
- The builder can preview diagonal magnetic supercells to check longer-wavelength structures.
- The preview uses vector magnetic moments rather than scalar z-only arrows.
- Internal storage uses complex Fourier coefficients and propagation vectors.
- Delete and Backspace both remove selected components.
- A multi-q texture path now supports square and hexagonal skyrmion lattices plus custom Fourier texture terms.
- Texture rows are grouped components with metadata, so a skyrmion lattice remains one editable assignment rather than several unrelated rows.
- The texture dialog includes a compact scrollable workflow, repeated skyrmion-style quiver preview, opt-in moment normalization, and component pruning controls.

Not yet implemented:

- The preset builders still expose only the first usable parameter set for each texture; they are not yet at the quality level of `mag2spectra`.
- The q-basis is assigned implicitly and cannot yet be edited or inspected as a first-class part of the model.
- Components cannot yet be duplicated, disabled/enabled from the table, or reordered.
- The guidance/status surface is still thin; it reports preview state but does not yet provide a polished stepwise workflow.
- The field-calculation page has only a magnetic-structure summary and primitive demagnetizing-field controls.
- Field calculation, distortion correction, symmetry-equivalent muon-site expansion, and multi-q incommensurate distributions still need dedicated user-facing workflows.

## Workflow Quality Review

The split is an improvement because it gives the magnetic-structure builder a clear boundary and keeps the Magnetic Fields page from becoming a dense mixed-purpose tool. It also matches existing MuFinder patterns where complex configuration happens in a dialog and the workflow page summarizes the accepted state.

Remaining UX issues:

- The add-component dialog is too generic. It asks for amplitude, axes, q, and phase without enough structure-specific affordances.
- The builder table is currently read-only and action-light. A user will expect edit, duplicate, enable/disable, and inspect actions.
- The preview state depends on table selection, but the table does not yet make the selected-preview/full-preview distinction visually obvious enough.
- The model summary on the Magnetic Fields page is too sparse for confident review before running calculations.
- Demagnetizing-field controls are visible before field calculations exist; they may move into the calculation setup section later.

## Recommended Next Slice

1. Keep the builder as a separate dialog with its own live preview, matching the supercell editor workflow.
2. Add a first-class q-basis inspection panel so users can see how grouped textures map onto the stored propagation-vector basis.
3. Add component duplication and enable/disable toggles.
4. Add a q-basis panel that shows the current basis and how each component maps onto it.
5. Keep field-calculation options out of the builder except where they affect preview.
