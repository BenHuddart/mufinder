# CASTEP `.cell` Syntax Guide

This package implements a conservative subset of CASTEP `.cell` syntax for library use and editor assistance. The contributor-facing source of truth for implemented support is [`keyword_catalog.yaml`](keyword_catalog.yaml).

## Formatting Conventions

- Render keywords and block names in uppercase.
- Render blocks as:

```text
%BLOCK NAME
...
%ENDBLOCK NAME
```

- Separate major sections with a single blank line.
- Render scalar `.cell` keywords as whitespace-separated tokens, for example `KPOINT_MP_GRID 4 4 4`.
- Emit optional unit lines only when the structured input includes a unit.

## Validation vs Serialization

- `src/mufinder/castep_io/schema.py` defines the supported starter subset.
- `src/mufinder/castep_io/validators.py` performs structure, type, exclusivity, and task-aware checks.
- `src/mufinder/castep_io/cell_writer.py` only serializes already-validated objects.

This separation is intentional: extending support should start with schema and docs, then validator logic, then serializer updates if formatting is keyword-specific.

## Implemented Starter Subset

- Lattice blocks: `LATTICE_CART`, `LATTICE_ABC`
- Position blocks: `POSITIONS_FRAC`, `POSITIONS_ABS`
- Common blocks: `SPECIES_POT`, `KPOINTS_LIST`, `HUBBARD_U`
- Common scalars/flags: `KPOINT_MP_GRID`, `KPOINT_MP_SPACING`, `SYMMETRY_GENERATE`
- Spin decoration (structure-driven): optional inline `SPIN=<value>` token on `POSITIONS_FRAC` rows for spin-polarized workflows
- DFT+U setup: `HUBBARD_U` is represented as a `.cell` block, with an optional `eV` or `Ha` unit line and guided-editor rows that target all atoms of a species for one orbital, such as `Ni d: 2.4`.

Anything outside this subset should be treated as unsupported until it is added to the catalog, schema, and tests.

## Example

```text
%BLOCK LATTICE_CART
ang
5.43 0 0
0 5.43 0
0 0 5.43
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Fe 0 0 0 SPIN=2
Fe 0.5 0.5 0.5 SPIN=-2
%ENDBLOCK POSITIONS_FRAC

KPOINT_MP_GRID 4 4 4
SYMMETRY_GENERATE
```
