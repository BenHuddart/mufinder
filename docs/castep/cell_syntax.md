# CASTEP `.cell` Syntax Guide

This package implements a conservative subset of CASTEP `.cell` syntax for library use and editor assistance. The contributor-facing source of truth for implemented support is [`keyword_catalog.yaml`](keyword_catalog.yaml); a byte-identical copy ships as `src/mufinder/assets/keyword_catalog.yaml` and serves as the starter catalog (`tests/castep_io/test_schema.py` checks the two agree).

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

- `src/mufinder/castep_io/schema.py` builds the runtime schema from a discovered `castep_keywords.json` plus `catalog_overrides.py`, and falls back to the starter catalog when none is found (see `docs/castep_keyword_catalog.rst`).
- `src/mufinder/castep_io/validators.py` performs structure, type, exclusivity, and task-aware checks.
- `src/mufinder/castep_io/cell_writer.py` only serializes already-validated objects.

This separation is intentional: extending support should start with schema and docs, then validator logic, then serializer updates if formatting is keyword-specific.

## Parsing

`src/mufinder/castep_io/cell_parser.py` reads `.cell` and `-out.cell` text through `parse_cell_text` and `parse_cell_file`; it is the only `.cell` reader in the package.

- Comments (`!`, `#`) are stripped and block names are case-insensitive.
- An optional leading length-unit line (`ang`, `bohr`, `nm`, `pm`, `cm`, `m`) on `LATTICE_CART`, `LATTICE_ABC` and `POSITIONS_ABS` converts lengths to Angstrom. Any other single non-numeric token in that position is rejected as an unsupported length unit.
- `POSITIONS_ABS` rows are converted to fractional coordinates wrapped into `[0, 1)`; `POSITIONS_FRAC` coordinates are kept as written. Species labels such as `H:mu` are kept verbatim.
- `SPIN=<value>` tokens are read from `POSITIONS_FRAC` rows, including the padded `SPIN=  1.000` form CASTEP writes. Rows without a token get spin 0, an empty `SPIN=` marker counts as no token, and `atomic_spins` is `None` unless at least one row carries a value.
- Both lattice blocks, both position blocks, a repeated geometry block, and an unterminated or mismatched block are errors.
- A full parse validates the whole file against the loaded keyword catalog, so a keyword outside it raises `CellParseError`. With `geometry_only=True` only the lattice and position blocks are read and validated, and every other keyword and block is skipped. Structure import (`mufinder.science.structure_io.load_structure_file`) uses this mode, so a CASTEP `-out.cell` imports regardless of the calculation keywords it carries.

## Implemented Starter Subset

The `.cell` entries in the starter catalog are:

- Lattice blocks: `LATTICE_CART`, `LATTICE_ABC`
- Position blocks: `POSITIONS_FRAC`, `POSITIONS_ABS`, and the transition-state endpoint blocks `POSITIONS_FRAC_PRODUCT`, `POSITIONS_FRAC_INTERMEDIATE`, `POSITIONS_ABS_PRODUCT`, `POSITIONS_ABS_INTERMEDIATE`
- Species and k-point blocks: `SPECIES_POT`, `SPECIES_MASS`, `SPECIES_GAMMA`, `KPOINTS_LIST`, `HUBBARD_U`
- Constraint block: `CELL_CONSTRAINTS`
- Scalars/flags: `KPOINT_MP_GRID`, `KPOINT_MP_SPACING`, `PHONON_KPOINT_MP_GRID`, `SYMMETRY_GENERATE`, `SYMMETRY_TOL`, `FIX_ALL_CELL`, `FIX_COM`, and the opaque text-valued `UNIT_CELL`
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
