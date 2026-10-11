# CASTEP `.param` Syntax Guide

This package implements a small, explicit subset of CASTEP `.param` keywords. The contributor-facing source of truth for what MuFinder currently supports is [`keyword_catalog.yaml`](keyword_catalog.yaml), mirrored as the packaged starter catalog `src/mufinder/assets/keyword_catalog.yaml`.

## Formatting Conventions

- Render keyword names in uppercase.
- Render scalar assignments as `KEYWORD : value`.
- Render booleans as `TRUE` or `FALSE`.
- Preserve explicit units when the structured value includes them.
- Render block-style keywords with `%BLOCK ...` / `%ENDBLOCK ...` markers.

## Validation vs Serialization

- `schema.py` builds the runtime schema from a discovered `castep_keywords.json` plus `catalog_overrides.py`, falling back to the starter catalog (translated by `starter_catalog.py`) when none is found (see `docs/castep_keyword_catalog.rst`).
- `validators.py` checks types, mutually exclusive keywords, missing requirements, and a small set of task-specific rules backed by local CASTEP evidence.
- `param_writer.py` handles deterministic text rendering only.

This makes it possible to add new keywords by extending the schema and validator with small, focused changes rather than scattering logic through the writers.

## Implemented Starter Subset

The `.param` entries in the starter catalog are:

- Task, run control and XC settings: `TASK`, `CONTINUATION`, `XC_FUNCTIONAL`, `XC_DEFINITION` (block), `BASIS_PRECISION`, `NUM_FARMS`, `IPRINT`, `LENGTH_UNIT`, `DEVEL_CODE` (block)
- Electronic controls: `CUT_OFF_ENERGY`, `FINE_CUT_OFF_ENERGY`, `MAX_SCF_CYCLES`, `ELEC_ENERGY_TOL`, `NONSCF_EIGENVALUE_TOL`, `SMEARING_WIDTH`, `NEXTRA_BANDS`, `CHARGE`
- Spin settings: `SPIN_POLARIZED`, `SPIN_POLARISED`, `SPIN`
- Geometry optimization settings: `GEOM_ENERGY_TOL`, `GEOM_FORCE_TOL`, `GEOM_STRESS_TOL`, `GEOM_DISP_TOL`, `GEOM_MAX_ITER`, `GEOM_FREQUENCY_EST`
- Transition-state search and NEB settings: `TSSEARCH_METHOD`, `TSSEARCH_ENERGY_TOL`, `TSSEARCH_FORCE_TOL`, `TSSEARCH_DISP_TOL`, `TSSEARCH_LST_MAX_ITER`, `TSSEARCH_QST_MAX_ITER`, `TSSEARCH_MAX_PATH_POINTS`, `TSSEARCH_NEB_MAX_ITER`, `TSSEARCH_NEB_SPRING_CONSTANT`, `TSSEARCH_NEB_METHOD`, `TSSEARCH_NEB_TANGENT_MODE`, `TSSEARCH_NEB_CLIMBING`, `TSSEARCH_NEB_NORMED`
- Phonon settings: `PHONON_METHOD`
- Output and property flags: `WRITE_CELL_STRUCTURE`, `WRITE_FORMATTED_DENSITY`, `CALCULATE_STRESS`, `CALCULATE_POLARIZATION`
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
