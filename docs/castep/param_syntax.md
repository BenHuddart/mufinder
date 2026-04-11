# CASTEP `.param` Syntax Guide

This package implements a small, explicit subset of CASTEP `.param` keywords. The contributor-facing source of truth for what MuFinder currently supports is [`keyword_catalog.yaml`](keyword_catalog.yaml).

## Formatting Conventions

- Render keyword names in uppercase.
- Render scalar assignments as `KEYWORD : value`.
- Render booleans as `TRUE` or `FALSE`.
- Preserve explicit units when the structured value includes them.
- Render block-style keywords with `%BLOCK ...` / `%ENDBLOCK ...` markers.

## Validation vs Serialization

- `schema.py` declares the supported starter subset and common relationships.
- `validators.py` checks types, mutually exclusive keywords, missing requirements, and a small set of task-specific rules backed by local CASTEP evidence.
- `param_writer.py` handles deterministic text rendering only.

This makes it possible to add new keywords by extending the schema and validator with small, focused changes rather than scattering logic through the writers.

## Implemented Starter Subset

- Task and XC settings: `TASK`, `XC_FUNCTIONAL`, `BASIS_PRECISION`
- Common scalar controls: `CUT_OFF_ENERGY`, `MAX_SCF_CYCLES`, `ELEC_ENERGY_TOL`
- Spin settings: `SPIN_POLARIZED`, `SPIN_POLARISED`, `SPIN`
- Geometry optimization settings: `GEOM_ENERGY_TOL`, `GEOM_FORCE_TOL`, `GEOM_STRESS_TOL`, `GEOM_MAX_ITER`
- Molecular dynamics settings: `MD_DELTA_T`, `MD_NUM_ITER`, `MD_TEMPERATURE`

## Example

```text
TASK : GEOMETRYOPTIMIZATION
XC_FUNCTIONAL : PBE
CUT_OFF_ENERGY : 340 eV
MAX_SCF_CYCLES : 60
GEOM_FORCE_TOL : 0.05 eV/ang
GEOM_MAX_ITER : 30
```
