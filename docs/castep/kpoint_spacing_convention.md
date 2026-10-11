# K-point spacing convention: findings and open questions

> **Status**: reference note — captures findings from a 2026-05 investigation.
> Three fixes plus a sweep-generator rewrite have been merged; remaining open
> questions are deferred.

---

## TL;DR — current conventions

**Direction 1 — spacing → grid** (CASTEP, mirrored in
`grid_from_spacing`): `n_i = floor(|b_i| / (2*pi*s)) + 1`. Returned grid
matches CASTEP exactly.

**Direction 2 — grid → spacing** (MuFinder-only;
`spacing_from_grid`): `s = min_i(|b_i| / (2*pi*(N_i - 1)))` over axes with
`N_i >= 2`. This is the *inclusive upper bound* of the CASTEP-valid range
for the grid; feeding the value back to `grid_from_spacing` reproduces the
grid. Raises `ValueError` for `(1, 1, 1)`.

**Sweep generator** (`generate_kpoint_grids`): walks CASTEP's spacing
trajectory by probing at every transition spacing `|b_i|/(2*pi*n)` and
emitting each distinct grid produced. Every grid is round-trip-reachable.

**One-line guarantee**: for any grid returned by `generate_kpoint_grids`,
`grid_from_spacing(lattice, spacing_from_grid(lattice, grid)) == grid`.

**Floating-point note**: `grid_from_spacing` uses `floor(|b_i|/(2π·s)) + 1`.
When `s` is the output of `spacing_from_grid` — computed as `|b_i|/(2π·(N_i-1))`
— the ratio `|b_i|/(2π·s)` should be exactly `N_i-1`, but two-step IEEE 754
division can produce `N_i-1 - ε` (e.g. 6.999…9991 instead of 7.0), causing
`floor` to snap down by one. `grid_from_spacing` corrects this by adding `1e-9`
inside the `floor` call, which is large enough to absorb ULP rounding error yet
far below 1, so it never advances a genuinely non-integer ratio. The fix is
tested by `test_grid_from_spacing_round_trips_lif_high_n` (N=2 through N=13).

**Where these values surface in the GUI**: results-pane "Recommended
k-point grid: ... (spacing X 1/Ang)" line; per-grid spacing annotations on the
examiner and overview plots; "Use spacing" toggle defaults in the relaxation,
muon-relaxation, and UEP workflow configure-parameters dialogs. All three take
it from `SupercellColumn.suggested_spacing` (`app/workflow_table_utils.py`),
which applies `spacing_from_grid` to the project grid and its reference lattice. The examiner's secondary spacing axis is
different: its ticks are round spacing values placed with
`kpoint_count_for_spacing` (`_regular_spacing_ticks`), not per-grid
`spacing_from_grid` values.

**What is NOT round-trip-safe**: a grid not in the sweep (e.g. a manually
entered one off CASTEP's curve, or a converged grid persisted before the
generator rewrite). `spacing_from_grid` still returns a value, but feeding
it to CASTEP will produce a different grid. Re-running convergence with
the current generator avoids this.

---

## Background

MuFinder's convergence sweep reports a "spacing" for each k-point grid in the
log text and in `state.json`.  A bug was discovered and fixed where the reported
spacing was computed with the wrong formula, causing:

1. All grids in the `6×6×5`, `7×7×5`, …, `12×12×5` series to report the
   **same** spacing (0.02005 1/Å) for hexagonal AlF₃, despite very different
   in-plane sampling.
2. Spacings that were approximately **half** the physically correct value.

---

## CASTEP's formula (confirmed from source)

`E:\CASTEP-22.11\Source\Fundamental\cell.f90`, subroutine
`cell_kpoints_spacing_wrapped`, lines ≈ 20765–20773:

```fortran
rx = sqrt(sum(mycell%recip_lattice(1,:)**2))   ! |b₁|, already 2π-scaled (1/Å)
ry = sqrt(sum(mycell%recip_lattice(2,:)**2))
rz = sqrt(sum(mycell%recip_lattice(3,:)**2))
! "add one since mp set is minimum required and int() rounds down"
mp_nx = int(rx / (mp_spacing * two_pi)) + 1
mp_ny = int(ry / (mp_spacing * two_pi)) + 1
mp_nz = int(rz / (mp_spacing * two_pi)) + 1
```

CASTEP's forward direction is therefore:

```
n_i = floor( |b_i| / (s · 2π) ) + 1
```

where `|b_i|` includes the 2π factor and `s` is `KPOINT_MP_SPACING` in 1/Å.
This formula is also used by MuFinder's `grid_from_spacing`.

The physical k-point spacing of the resulting grid is:

```
physical spacing = max_i ( |b_i| / (2π · n_i) )
```

---

## `spacing_from_grid`: from physical density to round-trip-safe

`src/mufinder/science/kpoint_grids.py::spacing_from_grid`

CASTEP reproduces grid `(N₁, N₂, N₃)` exactly when `KPOINT_MP_SPACING = s`
lies in the half-open range

```
( max_i |b_i|/(2π·N_i),  min_i |b_i|/(2π·(N_i−1)) ]
     ↑ lower bound (exclusive)    ↑ upper bound (inclusive)
```

(For axes with `N_i = 1` the lower-bound term collapses to zero and the
upper-bound term is +∞; both bounds are taken over the remaining axes.)

The function has been revised twice during this investigation.

**Stage 1 — physical-density formula** (`max_i(|b_i|/(2π·N_i))`). Adopted to
replace the original `min/(N−1)` formula which collapsed anisotropic grids
onto the same value. Returned the exclusive lower bound of the valid
CASTEP range; mathematically meaningful as "average reciprocal-space
separation" but did **not** round-trip — CASTEP's `floor + 1` at that boundary
emits `N_i + 1` on the axis attaining the max. For LiF cubic 5×5×5 → reported
0.04898 → CASTEP would emit 6×6×6.

**Stage 2 (current) — round-trip upper bound** (`min_i(|b_i|/(2π·(N_i−1)))`,
over axes with `N_i ≥ 2`). The inclusive upper bound of the valid CASTEP
range. Every reported spacing reproduces its grid exactly. Raises
`ValueError` for `(1, 1, 1)` because no single spacing characterises
Gamma-only sampling.

Stage 2 was adopted after confirming that every production caller of
`spacing_from_grid` (the convergence examiner's results pane, the overview
chart annotations, and the grid↔spacing toggle defaults in the relaxation,
muon-relaxation, and UEP widgets) wants a value that, when written into a
`.cell` file's `KPOINT_MP_SPACING` line, reproduces the same grid in CASTEP.
The displayed spacing is now a value a user can copy directly into a
`.cell` file with no surprises.

### Round-trip table after Stage 2 + Fix 4 (sweep generator rewrite)

| Cell | Sweep grids | round-trip via current `spacing_from_grid` |
|---|---|---|
| AlF₃ hex (s_min=0.02) | 23 grids | 23/23 |
| LiF cubic (s_min=0.04) | 7 grids | 7/7 |
| Orthorhombic 4×6×8 (s_min=0.1) | 4 grids | 4/4 |
| Slab 3×3×30 (s_min=0.05) | 7 grids | 7/7 |
| Anisotropic 3×6×20 (s_min=0.03) | 13 grids | 13/13 |

---

## Impact of the rewrites on state.json values

For AlF₃ hexagonal with a 3×3×3 grid:

- Original buggy: 0.0401 1/Å (c-direction, N−1 denominator)
- Stage 1 (physical density): 0.0790 1/Å (ab-plane, N denominator)
- **Stage 2 (current, round-trip)**: 0.0401 1/Å (c-direction, N−1 denominator,
  upper bound of valid range)

Note the Stage 2 numerical value coincides with the original buggy formula for
this isotropic grid — but the mathematical meaning is now "upper bound of
CASTEP's valid input range" rather than "average separation". For anisotropic
grids the two diverge: the buggy `min/(N−1)` would collapse 6×6×5, 7×7×5, …,
12×12×5 onto the same value (because it ignored which axis was being indexed),
whereas the current formula respects per-axis transitions and gives distinct
values where the upper bound actually differs.

---

## Follow-up fixes (also merged)

### Fix 2: `_compute_spacings` — plateau guard

`src/mufinder/app/convergence_examiner.py::ConvergenceExaminerDialog._compute_spacings`

For cells with a very short direct axis in one direction and a very long axis in
another (e.g. a thin slab), successive grids can report the same spacing while
other axes keep growing, because one axis dominates the bound. A run of
identical spacings misleads the user about convergence.

The guard was first written for the Stage 1 formula, skipping a grid when a
frozen `n_i = 1` direction attained `max_i(|b_i|/(2π·N_i))`. With Stage 2 the
current guard is simpler: `_compute_spacings` walks the grids in order, skips
`(1, 1, 1)` (for which `spacing_from_grid` raises), and skips any grid whose
spacing equals the previous annotated spacing (`math.isclose`,
`rel_tol=1e-9`), so the annotation series stays strictly decreasing. The
overview log and convergence plot show no spacing annotation for skipped grids
rather than a misleading repeated value.

### Fix 4: spacing-driven sweep generator

`src/mufinder/science/kpoint_grids.py::generate_kpoint_grids`

A separate investigation confirmed CASTEP has no grid → spacing path at all —
every `kpoint_mp_spacing` assignment in CASTEP 25.12 is either init, copy, or
user input; the only conversion routine is `cell_kpoints_spacing_wrapped`
(spacing → grid). Consequently MuFinder's spacing label is purely informational
*and* MuFinder is free to choose its own sweep trajectory.

The previous sweep generator walked an isotropic level ramp with per-axis
clamping:

```python
for level in range(1, max(max_grid) + 1):
    grid = (min(level, max_grid[0]), min(level, max_grid[1]), min(level, max_grid[2]))
```

For anisotropic cells this trajectory diverges sharply from CASTEP's
spacing-locked trajectory — for AlF₃ only 2 of 12 ramp grids were
CASTEP-reachable (the round-trip table above).

**Fix**: enumerate CASTEP's piecewise-constant spacing curve directly. The grid
produced by `grid_from_spacing` changes only at breakpoints `|b_i|/(2π·n)`;
probing at each breakpoint yields every distinct CASTEP-reachable grid between
`(1, 1, 1)` and the cap. Effects on the example sweeps:

| Cell | Old sweep length / CASTEP-reachable | New sweep length / CASTEP-reachable |
|---|---|---|
| AlF₃ hex (a=4.872, c=12.466), s_min=0.02 | 12 / 2 | 23 / 23 |
| LiF cubic (a=4.0834), s_min=0.04 | 7 / 7 | 7 / 7 |
| Slab (3, 3, 30), s_min=0.05 | 7 / 7 | 7 / 7 |
| Anisotropic (3, 6, 20), s_min=0.03 | 12 / 3 | 13 / 13 |

Isotropic and frozen-axis (slab) cells are unchanged because CASTEP's curve
coincides with the isotropic ramp for those geometries. Anisotropic cells
typically gain grids because each axis transitions independently — the trade-off
is more convergence jobs per sweep.

Resolves the sweep side of the round-trip concern: every generated grid is
reproducible from some `KPOINT_MP_SPACING`. Stage 2 of `spacing_from_grid`
(above) then ensures the *displayed* spacing for any grid is a value that
CASTEP rounds back to that grid — so values shown in the GUI can be copied
directly into `.cell` files.

### Fix 3: stale `point_spacings` in overview snapshots

`src/mufinder/app/main_window.py::MainWindow._validated_kpoint_overview_snapshot`  
`src/mufinder/app/convergence_examiner.py::ConvergenceExaminerDialog.recompute_spacings_in_snapshot`

The convergence overview widget reads `point_spacings` from the snapshot that
was persisted in `state.json` at adopt-time. After a spacing-formula fix,
projects adopted before the fix continued to display old (wrong) spacings in
the overview panel indefinitely, because the persisted snapshot was never
refreshed.

**Fix**: `_validated_kpoint_overview_snapshot` now calls
`ConvergenceExaminerDialog.recompute_spacings_in_snapshot(snapshot,
host.lattice_vectors)` before handing the snapshot to the overview widget.
This always re-derives spacings from the current host structure's reciprocal
lattice, so both old and new projects show correct values on every load.

The recomputation is lightweight (one call to `spacing_from_grid` per grid
point) and happens only when the validated snapshot is about to be displayed.
The persisted state.json is not modified; only the in-memory copy used for
rendering is updated.

---

## Open questions (deferred)

### 1. Using spacing for cross-system comparability

`KPOINT_MP_SPACING` is attractive when running calculations across multiple
structures with different cell shapes, because it keeps the k-point density
comparable rather than requiring manual grid specification per system. After
Fix 4 (sweep generator) and Stage 2 of `spacing_from_grid` (round-trip upper
bound), MuFinder's spacing convention is well-defined for a *single* cell.
For cross-system propagation the remaining question is which property to hold
constant when moving the recommended spacing to a different cell:

- Round-trip upper bound on the source cell, then apply the same value on the
  target cell. Reproduces the source grid exactly on the source cell, but on
  the target cell may give a noticeably different `(N₁′, N₂′, N₃′)` because the
  upper bound depends on the source cell's reciprocal lattice, not on
  physical density.
- Physical density (`max_i(|b_i|/(2π·N_i))`) on the source cell, then apply
  on the target cell. Preserves k-point spacing in absolute terms but produces
  a grid one step finer than the source on the source cell itself.
- An explicit per-cell grid computed via `equivalent_grid_for_supercell` and
  the like (already implemented for supercells).

No decision required yet — cross-system spacing input is not exposed in any
workflow today.

### 2. K-point convergence transfer to supercells

MuFinder already has `equivalent_grid_for_supercell` in `kpoint_grids.py`,
which maps a primitive-cell grid to an equivalent supercell grid by scaling
each axis by `|b_i^{super}| / |b_i^{prim}|`. The spacing-based analogy would
be: adopt the primitive-cell spacing and let CASTEP compute the supercell grid
directly from `KPOINT_MP_SPACING`.

Complication: CASTEP's `floor+1` formula means specifying the primitive-cell
physical spacing produces a supercell grid that is one step finer per axis than
the minimum required. Whether this is desirable (conservative) or wasteful
(extra k-points) depends on context.

The round-trip analysis here provides the numerical foundation for reasoning
about these trade-offs in a future session.

---

## Key files

| Path | Role |
|---|---|
| `src/mufinder/science/kpoint_grids.py` | `spacing_from_grid`, `grid_from_spacing`, `generate_kpoint_grids`, `kpoint_count_for_spacing`, `equivalent_grid_for_supercell` |
| `src/mufinder/app/convergence_examiner.py` | `_compute_spacings` (uses `spacing_from_grid`); `recompute_spacings_in_snapshot`; `_regular_spacing_ticks` (uses `kpoint_count_for_spacing`) |
| `src/mufinder/app/workflow_row_widget.py`, `workflow_table_utils.py` | "Use spacing" toggle defaults for the relaxation, muon-relaxation and UEP widgets: `RowTableWidget` seeds the dialog from `SupercellColumn.suggested_spacing` (uses `spacing_from_grid`) |
| `src/mufinder/app/sweep_convergence_widget.py`, `kpoint_convergence_widget.py` | The k-point sweep's "Generate From Spacing..." rows: `KPointConvergenceWidget.generate_keys` calls `generate_kpoint_grids` for the `SweepConvergenceWidget` table |
| `tests/science/test_kpoint_grids.py` | Unit tests including AlF₃ and LiF sanity checks added with this fix |
| `E:\CASTEP-22.11\Source\Fundamental\cell.f90` | CASTEP source — `cell_kpoints_spacing_wrapped` ≈ line 20698 |
