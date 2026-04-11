# CASTEP Support Instructions

- Treat [`docs/castep/keyword_catalog.yaml`](docs/castep/keyword_catalog.yaml) as the authoritative contributor-facing source for implemented CASTEP keywords, formatting rules, and examples.
- Never invent CASTEP keywords, block names, allowed values, units, or syntax. If support is unclear from local CASTEP source or docs, leave the keyword unimplemented and add a conservative note to the catalog or docs instead.
- Do not commit or distribute `castep_keywords.json` or generated copies of CASTEP metadata in this repository. Use a local CASTEP checkout via runtime discovery or environment variables instead.
- Keep schema, validation, and serialization separate:
  - `src/mufinder/castep_io/schema.py` discovers local CASTEP metadata and defines the supported runtime subset.
  - `src/mufinder/castep_io/validators.py` enforces explicit validation.
  - `src/mufinder/castep_io/cell_writer.py` and `src/mufinder/castep_io/param_writer.py` only render validated data.
- Prefer explicit, descriptive validation errors over permissive behavior or silent fallback.
- Preserve exact CASTEP formatting conventions:
  - uppercase keyword rendering
  - deterministic ordering
  - `%BLOCK ...` / `%ENDBLOCK ...` delimiters
  - stable scalar formatting
- Extend CASTEP support by updating docs and schema first, then targeted validator/writer logic, then tests.
- Every new keyword or block must come with focused tests, including serialization and validation coverage where relevant.
