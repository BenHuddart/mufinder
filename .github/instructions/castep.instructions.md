---
applyTo: "src/mufinder/castep_io/*.py,tests/test_castep_*.py,tests/golden/*.cell,tests/golden/*.param,docs/castep/**/*.md,docs/castep/**/*.yaml"
---

# CASTEP Package Editing Guidance

- Keep serializers deterministic. The same structured object should always render to byte-identical CASTEP text.
- Preserve exact CASTEP block formatting with `%BLOCK NAME` and `%ENDBLOCK NAME`.
- Keep validation in `validators.py`; do not bury keyword checks inside writers.
- Keep keyword metadata in `schema.py` aligned with `docs/castep/keyword_catalog.yaml`.
- Never vendor `castep_keywords.json` or generated copies of CASTEP metadata into the repo; depend on local discovery instead.
- Do not silently ignore unsupported or conflicting options. Raise descriptive exceptions instead.
- Prefer small, schema-driven additions. If a new keyword needs special rendering, isolate that logic to the smallest possible change.
- When editing tests, keep them lightweight and readable. Favor golden-file comparisons for stable formatting behavior.
- When uncertain about CASTEP behavior, leave a clear note in the docs or catalog rather than guessing.
