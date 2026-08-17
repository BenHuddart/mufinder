Legacy Relaxation Import
========================

This document is the design contract for importing muon structural relaxations
produced by the **legacy MuFinder** (the Tkinter-era GitLab application) into a
project in the current application, so that those relaxed muoniated structures
can be used as inputs for Quantum Corrections and Energy Barriers calculations.

Scope is deliberately narrow: **muon structural relaxations only**. Convergence
sweeps, pristine relaxations, UEP, and dipolar-field data are out of scope.

Status: design agreed; implementation in progress. Sections marked *planned*
describe behaviour that is specified but not yet implemented.

Research data boundary
----------------------

**No research data, and no reference to it, belongs in this repository.** That
covers more than the files themselves:

- no absolute or relative paths to research directories;
- no project, sample, or material names, whether in prose, code, docstrings,
  test identifiers, comments, ``.gitignore`` entries, or launch configurations;
- no copied output files. A real ``.castep`` is megabytes of research results.

The datasets from which this design was derived are referred to throughout as
**dataset A** (36 relaxations, ``POSITIONS_FRAC``) and **dataset B** (79
relaxations, ``POSITIONS_ABS``, magnetic host). Their identities and locations
are deliberately absent and must stay absent.

**Every test fixture is hand-authored and synthetic** — a minimal file written
to a documented file shape, never a trimmed copy of a real run. This is not
merely a compliance cost:

- a synthetic ``.castep`` is a few dozen lines carrying exactly the header block
  under test, rather than megabytes of SCF history;
- the cases that matter most — a stale input tree, a spin-polarised run, a
  ``HUBBARD_U`` block, a muonium run, a multi-restart output — are absent from
  both real datasets anyway, so they must be synthesised regardless;
- a fixture written to a documented shape fails loudly when that shape is wrong,
  whereas a copied file quietly encodes one machine's CASTEP build.

Where this document cites a concrete observation from dataset A or B — a file
count, a parameter disagreement, a token tally — that observation is evidence
for a design decision, reproduced as a number and never as data.

Motivation
----------

Relaxed muon sites produced by the legacy application cannot currently feed
Quantum Corrections (Gamma-point phonons) or Energy Barriers (transition-state
search and NEB). Both gate their source lists on relaxations that this
application performed. The import exists to make a legacy relaxation
indistinguishable — *to those two workflows* — from a native one, without
pretending it was run here.

**Priority ordering.** The deliverable is the relaxed muoniated structure, usable
downstream. Recovered CASTEP parameters are informational: they let the user see
what produced the structure, and they seed the two values downstream jobs
inherit. Where the two goals conflict, structure availability wins. No
degradation in parameter recovery may block an import.

The legacy layout
-----------------

The legacy application has no project file. Everything is filesystem
convention::

    <work_dir>/
      <name>.cif                       host structure (sometimes; .mag_array alongside)
      input/  <seed>_<N>.cell          host supercell + H:mu at candidate N
              jobs.pkl                 transient submit queue, not a record
      run/    site<N>/<seed>_<N>.cell
              site<N>/<seed>_<N>.param
      results/<seed>_<N>.castep
              <seed>_<N>-out.cell

Properties that constrain the design:

- **The site index lives only in the filename.** Legacy parses it by splitting
  the name on ``_``, space, and ``.`` and taking the second-to-last field.
- **Site 0 is the pristine/neutral reference**, not a muon site. It is skipped
  on import.
- **No supercell matrix is recorded anywhere.** Legacy recovers it as
  ``L_relaxed @ inv(L_host)``. The importer must do the same.
- **``jobs.pkl`` is a transient queue**, not a provenance source. It is empty in
  dataset A.
- **The local input tree is not trustworthy.** In dataset A the local
  ``run/site1/<seed>_1.param`` declares an LDA functional, ``basis_precision:
  FINE`` and ``elec_energy_tol: 1E-10``, while the corresponding
  ``results/<seed>_1.castep`` reports PBE, a 900 eV cutoff, and a 3x3x3
  Monkhorst-Pack grid that appears in no local input file at all. The runs
  executed on an HPC system; the local input tree is a stale sibling.

.. important::

   The ``.castep`` header dump is the only authoritative record of what ran.
   Everything else is a hint.

The eligibility contract
------------------------

:func:`mufinder.workflow.quantum.quantum_source_eligibility` is called by
Quantum Corrections *and*, indirectly, by the Energy Barriers endpoint list.
Together with its caller it forms the entire contract an imported relaxation
must satisfy.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Entity
     - Required state
     - Enforced by
   * - ``MuonSite``
     - ``status == "relaxed"``, ``relaxation_run_id`` set
     - the gate
   * - ``CalculationRun``
     - id matches the site, ``status == COMPLETED``
     - the gate
   * - ``Job``
     - ``workflow_stage == MUON_RELAXATION``, ``status == COMPLETED``
     - the gate
   * - ``Structure``
     - ``origin == "muon_relaxation"``, ``properties["source_workflow_stage"]``
       is ``"muon_relaxation"``, species contains ``H:mu``
     - the gate
   * - run ``job_id`` / ``result_id``, result ``derived_structure_ids[0]``
     - present and resolvable
     - the caller, which skips a candidate on any ``None``

Nothing in either inspects *how* the job ran. An import that builds these five
entities honestly is automatically eligible downstream, requiring **no changes
to the quantum or energy-barrier layers**.

Two requirements sit just outside the gate and are easy to miss.

**Muon calc type** is not checked for eligibility, but it *is* required for the
job's output directory to resolve to ``<base>/<prefix>/<seed>``, and therefore
for parameter inheritance to find the files at all. It also drives the
charge-state label. Set it regardless.

**Structure parent** must point at the active host. Submitting a downstream job
runs ``validate_structures_match_active_host``, which calls
``resolve_underlying_host`` to walk ``parent_structure_id`` for an ancestor whose
origin is ``"imported"`` or ``"pristine_relaxation"``. An imported structure has
origin ``"muon_relaxation"``, so with no parent the walk returns ``None``, which
compares unequal to the host id and raises a host-mismatch dialog. The site would
pass every visible check and then fail at the exact moment the feature exists to
serve. Native code sets this parent when deriving the relaxed structure; the
importer must too.

Parameter inheritance happens on disk
-------------------------------------

Downstream jobs do not read the source relaxation's parameters from the project
document. They re-read them from disk::

    kpoint_mp_grid  <-  parse_cell_file(<jobdir>/<seed>.cell)
    cut_off_energy  <-  <jobdir>/<seed>.param

NEB inherits the same way, via the reactant endpoint. For dataset A a verbatim
copy of the legacy files would supply **neither** value: the input ``.cell``
carries no ``kpoint_mp_grid``, and the ``.param`` uses ``basis_precision``
instead of ``cut_off_energy``. A phonon job seeded from it would silently fall
back to project defaults and run at the wrong cutoff on the wrong grid —
precisely the failure this feature exists to prevent. This constraint drives the
on-disk layout below.

Decisions
---------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Decision
     - Choice
   * - User interface surface
     - Imported rows merge into the main Structural Relaxations table, flagged
       and read-only.
   * - Seed files on disk
     - Normalise from the ``.castep`` record; archive byte-exact originals.
   * - Host structure
     - Require a matching host; offer to adopt the legacy CIF when the project
       has none.
   * - Removing an imported row
     - Cascade-check references, then delete row, job, run, result, structure,
       site and files together.
   * - Failed or unconverged legacy runs
     - The wizard lists all runs, pre-ticks converged ones, and allows importing
       the rest as ``FAILED``.
   * - No active definition
     - The import creates one and makes it active.
   * - Legacy site 0
     - Skipped; it is the pristine reference, not a muon site.
   * - Site labelling
     - ``Legacy Site <N>`` from the filename index, suffixed on collision.

Entity construction
-------------------

Per imported site, one five-entity chain satisfying the eligibility contract.

``MuonSite``
   ``status = "relaxed"``, ``generation_method = "imported_legacy"``,
   ``source_structure_id`` the active host, ``fractional_position`` taken from
   the **input** cell's ``H:mu`` folded to host coordinates,
   ``cartesian_position`` derived from it, and ``relative_energy_eV`` the
   absolute total energy from the ``.castep`` — matching what the native
   completion path stores, despite the field name.

``Job``
   ``MUON_RELAXATION``, ``COMPLETED``, calc type by detected charge state, tagged
   ``imported`` and ``legacy_mufinder``. ``execution_target`` stays ``"local"``;
   a completed job is never reattached, so the value is inert.

``CalculationRun``
   ``COMPLETED``, with ``definition_id``, ``job_id`` and ``result_id`` set,
   ``convergence_achieved`` from the geometry-optimisation success marker,
   ``started_at`` from the ``.castep`` run banner, and ``completed_at`` from the
   file modification time.

``Result``
   The payload is **not** only the muon-relaxation controller's contribution.
   Native completion merges that contribution onto a standard payload, and both
   halves are required:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Source
        - Keys
      * - standard payload
        - ``completed``, ``total_energy``, ``enthalpy``, ``forces``, ``stress``,
          ``max_force``, ``max_stress``, ``num_scf_cycles``, ``wall_time``,
          ``warnings``, ``geom_converged``
      * - controller contribution
        - ``lattice_a`` / ``lattice_b`` / ``lattice_c``,
          ``muon_final_fractional_position_supercell``,
          ``muon_final_fractional_position_host``
      * - import marker
        - ``imported``

   ``total_energy`` in particular is what the table's Total Energy column reads
   and what the completed-source list sorts on. Omitting it fails the
   verification checklist below.

``Structure``
   The relaxed cell. ``origin`` **must** stay ``"muon_relaxation"`` and
   ``properties["source_workflow_stage"]`` must stay ``"muon_relaxation"`` for
   eligibility, so the import marker lives in ``properties["import_source"]``.
   ``parent_structure_id`` **must** be the active host id, or every downstream
   submission fails its host check.

Merging into the live table
---------------------------

The Structural Relaxations table renders the intersection of the widget's form
state and the **active** definition's rows, matched on the key
``{charge_prefix}:{label}`` *and* on a row signature. Three constraints follow.

**Signature equality is load-bearing.** The row signature hashes the site id,
label, initial position, charge state, supercell matrix, *normalised* parameter
values, and the added-advanced list. Any disagreement between the form row and
the definition row drops the row to a fallback that only accepts jobs in an
active status — and an imported job is ``COMPLETED``, not active — so the row
renders as "ready" with no energy. The importer must therefore store *the
widget's normalised values*: push rows in, read the widget's state back, and
persist what comes back.

**Label uniqueness is mandatory.** The job mapping is keyed on
``{prefix}:{label}``, so an imported label colliding with a native one in the
same charge state silently overwrites a mapping. De-duplicate against the active
definition's existing labels at import time.

**Mutate the active definition in place.** Every existing caller of the
muon-relaxation definition registrar leaves the job mapping empty and relies on
a later synchronisation pass to re-adopt bindings. That pass cannot fire for an
import, because it only matches rows whose signatures already agree. Import
therefore appends to the live definition's rows, job mapping, seed-name mapping
and input site ids. Only when no active definition exists is one created.

Safe without further work, **for completed imports only**: submission is already
blocked, because the row's effective status reads ``"completed"`` and the ready
check accepts only ready, defined, and not-generated. The dashboard
synchronisation pass adopts only jobs in an active status, so it cannot hijack
imported mappings.

Two hazards must be gated explicitly.

**Continuation.** A completed job is continuable, so "Continue relaxation" would
appear on imported rows. It rebuilds the cell from the *initial* muon position
and relies on a checkpoint file that an import does not have — a silent restart
from scratch dressed as a warm start.

**Failed imports.** The "already blocked" reasoning above holds only for
``COMPLETED``. A ``FAILED`` job is visible in the Job Dashboard with Resubmit,
Reset and Dismiss actions. One dismissal makes the row's effective status read
"ready", after which the ordinary submit cycle resets the job to ``DEFINED`` and
launches a real local calculation in the imported directory. Worse, the reset
cleanup keeps only ``.cell`` and ``.param`` and recursively removes
subdirectories — destroying the byte-exact ``.castep``, the relaxed output cell,
**and the entire archive of legacy originals**. Resubmit, reset, and directory
cleanup must all be gated on the imported marker; failing that, failed runs must
be imported in a non-resubmittable representation.

Row model
~~~~~~~~~

``MuonRelaxationRow`` gains an ``imported`` flag and an import-batch id. The
widget honours them to render a badge and to disable in-place editing, row
removal, inclusion in a batch parameter edit, and continuation. The parameter
dialog is opened read-only, which it already supports.

**The flag must survive reconfiguration** — *implemented*. Whenever the user
configures rows, a *new* definition is registered from the widget's current
configuration. Unless the flag and batch id are threaded through the widget's
internal row state, its state collection and restoration, *and* the
configuration serialiser and deserialiser, the flag is silently dropped on the
first reconfigure — every read-only gate dissolves while the imported entities
and files remain.

Both fields are additive with defaults, so a project saved before the import
existed loads with every row treated as natively staged, which is correct. They
carry a ledger entry in ``compatibility_policy``: neither may be dropped without
a migration that first relocates the marker, because a row that loses its flag
becomes editable and resubmittable while its imported entities and files remain.

The row signature includes both fields deliberately. Signature equality gates
whether a row renders with its real status, so a marker dropped from one side of
the form/definition pair makes the row visibly fall back to "ready" — far better
than rendering correctly while silently becoming editable.

.. warning::

   The widget's state shape is **not** the definition's configuration shape.
   The widget names the position ``fractional_position`` where the configuration
   says ``initial_fractional_position``, so feeding one directly to the other
   parses without error and silently zeroes every position. The import commit
   writes both shapes and must build each explicitly.

**Recovered parameters are filtered by the widget.** Row normalisation discards
names outside the widget's spec set and values equal to implicit project
defaults, so some recovered keywords never appear as row parameters or in the
dialog. That is acceptable under the priority ordering, but the wizard must not
imply the row carries the complete recovered set: the on-disk ``.param`` and the
provenance record are the authoritative copies.

Files on disk
-------------

::

    files/muon_relaxation/<charge_prefix>/<legacy_seed>/
        <legacy_seed>.cell        rendered: true initial structure plus the
        <legacy_seed>.param       effective parameters read from the .castep
        <legacy_seed>.castep      byte-exact copy
        <legacy_seed>-out.cell    byte-exact copy
        legacy_source/            byte-exact originals + SHA-256 manifest

The **legacy seed name is retained**. It keeps the ``.castep`` self-consistent
with its own filename, requires no content rewriting, and makes imported job
directories visibly distinct, which is a provenance feature rather than a cost.
The seed-name mapping already supports arbitrary seeds.

Rendering the ``.cell`` and ``.param`` from the ``.castep`` record is what makes
inheritance correct. It is not fabrication: the same atoms, with the grid and
cutoff that actually ran made explicit. The rendered ``.cell`` must contain the
**initial** structure, not the relaxed one, so that initial-versus-relaxed
comparisons stay truthful.

Parameter recovery
------------------

Two readers are required, both in :mod:`mufinder.castep_io`. Both are
**implemented**:

- :mod:`mufinder.castep_io.param_parser` — a ``.param`` reader. None existed
  previously, which is why the continue-relaxation keyword diff is computed by
  re-rendering rather than parsing. ``parse_param_text`` returns a
  ``ParamFile`` the writer can render again; ``parse_param_assignments`` returns
  a faithful ``{canonical name: value}`` view including keywords the runtime
  schema does not recognise; ``parse_param_observations`` splits mapped from
  unmapped. The spec-driven value coercion formerly living on ``MainWindow``
  moved here as ``parse_param_value``, which both the importer and the existing
  quantum-inheritance path now share.
- :mod:`mufinder.castep_io.run_parameters` — the ``.castep`` header-dump reader,
  covering the parameter blocks CASTEP echoes near the top of its output.

Both were validated against real legacy output from two CASTEP versions, and
correctly reproduce the input-tree disagreement described above: the echo
reports PBE with a 900 eV cutoff on a 3x3x3 grid while the stale local
``.param`` claims LDA and ``BASIS_PRECISION: FINE``. That disagreement is now
machine-detectable, which is what the wizard needs to surface it.

Value aliases must be folded the way CASTEP folds them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Legacy MuFinder wrote ``task: geometryoptimisation``. CASTEP accepts that and
normalises it to ``GEOMETRYOPTIMIZATION`` on read — its source performs the
substitution with the comment *"English spelling"* — but MuFinder's own
validator only accepts the American spelling. Parsed naively, a legacy
``.param`` therefore yields a value MuFinder then refuses to write, so the
rendered ``.param`` for an imported job could not be produced at all.

``param_parser`` folds enum values exactly as CASTEP does: uppercase, then a
transcribed table of the value aliases CASTEP itself rewrites (``GEOM``,
``GEOMOPT``, ``SPE``, ``MD`` and the rest). A value matching nothing is left
untouched rather than coerced, so the caller still sees what the file said. This
is transcription from the CASTEP source, not inference — a spelling CASTEP does
not fold must not be folded here.

The echo is lossy for some keywords
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CASTEP prints one display string for several distinct keyword values: both
``LDA`` and ``LDA_PZ`` print *Perdew-Zunger Local Density Approximation*, and
both ``SHF`` and ``SX`` print *Screened Hartree-Fock*. Reading the echo back
therefore cannot always recover which value was set, and no amount of parsing
effort will change that.

Such an observation reports the most canonical value and carries the remaining
candidates in ``ambiguous_with``, so the importer can record the ambiguity in
provenance rather than silently asserting one reading. Under the priority
ordering this is a cosmetic loss: no ambiguous keyword is one downstream jobs
inherit.

The header dump is not keyword-named
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Its lines are descriptive English, not CASTEP keywords::

     type of calculation                            : geometry optimization
     using functional                               : Perdew Burke Ernzerhof
     plane wave basis set cut-off                   :   900.0000   eV
                           MP grid size for SCF calculation is  3  3  3

So a keyword spec lookup cannot be applied until a **label to keyword-and-value**
translation table exists. That table, not the line parsing, is the substance of
this reader. It is enumerated explicitly from CASTEP's own documentation and
never inferred: the project rule forbids inventing keywords *or* values, so a
label with no confident mapping is reported as unmapped rather than guessed. Its
coverage is deliberately narrow — the legacy keyword set plus what inheritance
needs — and everything outside it is an unmapped observation.

Merge order and coercion
~~~~~~~~~~~~~~~~~~~~~~~~

Highest priority first: ``.castep``, then ``.param``, then ``.cell``, then the
relaxed output cell. A keyword is mapped when a runtime keyword spec exists for
it, and its value is coerced using that spec's value type. Anything unresolved is
recorded in provenance as an unmapped observation.

Every parsed value carries the file it came from. Disagreements are surfaced
before the user commits, with the ``.castep`` value shown as the one that will be
imported.

The runtime schema is environment-dependent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`mufinder.castep_io.schema` builds from the full CASTEP keyword metadata
discovered at runtime. The fallback used when no CASTEP installation is found is
the bundled starter catalog, which is far smaller. Measured against CASTEP 26.11,
the full catalog yields 447 ``param`` specs against the starter catalog's 47.

That gap looks alarming but barely intersects this feature. The legacy
application emits a narrow, fixed keyword set. Checking that entire set, plus the
cell keywords it writes, against the **starter** catalog, exactly one is missing:
``write_cif_structure``, a cosmetic output-format flag with no physics content
and no downstream role. Decisively, ``cut_off_energy`` and ``kpoint_mp_grid`` —
the two values inheritance actually reads — resolve under **both** catalogs. The
functionally critical subset is catalog-independent.

**Therefore: warn, never refuse.** The feature's purpose is to make relaxed
structures usable downstream; parsed parameters are informational alongside that.
Blocking an import over an informational-fidelity difference would trade the
feature's whole value against a cosmetic loss. The wizard notes the catalog in
use and flags degraded fidelity when running on the starter catalog, pointing at
the ``MUFINDER_CASTEP_KEYWORDS_JSON`` environment variable as the remedy. The
catalog source and its CASTEP version are recorded in the import provenance
regardless, so a completed import always names the catalog that governed its
parse.

.. note::

   Keyword overrides can retype a keyword relative to CASTEP's own metadata —
   ``charge`` is declared real by CASTEP but surfaces as an integer. Parsing must
   go through the resolved spec, never through CASTEP's raw option type.

Host structure and supercell inference
--------------------------------------

1. Infer ``S = round(L_input @ inv(L_host))`` from the **input** cell; the
   relaxed cell equals ``S * L_host`` only when the cell was held fixed.
2. Verify that ``S`` is integer within tolerance and that ``det(S) > 0``.
3. Verify the atom count equals ``det(S) * N_host + 1``.
4. Verify the host sublattice matches, modulo the muon.

On failure, refuse with a readable difference report. When the project has no
host structure, offer to import the legacy folder's CIF as the host.

**There may be no CIF to adopt.** Dataset A contains only the input, run and
results directories, with no CIF at all, so the adoption path does not exist for
it. When the project has no host *and* the folder has no CIF, block the import
with a message asking the user to import a host structure first. Do **not**
synthesise a host from an input cell: the muon-bearing supercell is not the host,
and inferring one would guess at the primitive cell and lose the magnetic
structure.

Charge-state detection
----------------------

**Charge alone determines the charge state. Spin polarisation is orthogonal to
it** — either charge state may have been run spin-polarised or not.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Source
     - Positive muon
     - Muonium
   * - ``.param``
     - ``charge: +1``
     - no ``charge`` line; CASTEP defaults to 0
   * - ``.castep``
     - ``net charge of system : +1.000``
     - ``net charge of system : 0.000``

This follows from the legacy parameter writer, which emits a charge line only for
the positive-muon case and takes spin polarisation as a separate, independent
argument.

One irreducible ambiguity: legacy writes no charge line for **either** muonium or
neutral, and nothing in the files distinguishes them. The resolution is
structural rather than parametric — neutral is used only for site 0, the pristine
reference, which has no muon and is skipped anyway. So a cell containing ``H:mu``
at charge 0 is muonium.

Magnetic and spin-polarised hosts
---------------------------------

Established by reading the legacy source rather than assumed.

**Legacy CASTEP inputs carry no per-atom spins whatsoever.** Legacy site
generation writes cells through ASE, and nothing anywhere in that codebase sets
initial magnetic moments, so ASE has no moments to emit and never writes ``SPIN=``
tokens. Verified against dataset B, whose host is a genuinely magnetic material
with rare-earth moments of several Bohr magnetons in its magnetic-structure file:
**0 of 83 input cells contain a spin token**.

The legacy magnetic-structure editor and its ``.mag_array`` file belong to a
different pipeline entirely. That file is a Fourier-coefficient CSV feeding the
**dipolar-field** calculation. It is not a DFT input and must not be read as one.

A magnetic legacy relaxation is therefore characterised by exactly two
recoverable things:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Where
     - What
     - Recoverable via
   * - ``.param``
     - ``spin_polarized : true`` — a single global flag, with no per-atom
       moments and no total spin keyword
     - the new ``.param`` reader; also the ``.castep`` treatment line
   * - ``.cell``
     - ``%BLOCK HUBBARD_U`` with element, orbital and U rows, appended from the
       legacy GUI
     - the cell parser: not a known block, so it lands among the generic parsed
       blocks

**Consequence for atomic spins.** There is nothing to reconstruct from the legacy
inputs. The available source is the converged Mulliken moments in the
``.castep``, which the output parser already extracts. Those are output rather
than input spins, which is if anything the better seed for a downstream phonon
job. Populate the structure's atomic spins from them when the run was
spin-polarised, and record in provenance that they are Mulliken-derived rather
than user-specified.

**Muonium remains a special case.** The quantum workflow raises if a muonium
muon carries no seeded moment, because an odd-electron cell with zero net moment
silently converges closed-shell. That seeding is driven by the charge state, so
an imported muonium site inherits it correctly *provided* charge detection ran
correctly — another reason charge detection must not be conflated with spin
polarisation.

.. note::

   ``phonon_method_compatibility`` accepts a Hubbard-U flag and refuses
   DFPT/linear-response when it is set, but no caller currently passes it. Legacy
   magnetic hosts routinely used a Hubbard U, so imports make that dormant guard
   considerably more reachable. Wiring it is separate work; the importer should
   at least record the recovered ``HUBBARD_U`` block in provenance.

Provenance
----------

A new ``import_legacy_relaxations`` action, plus a journal entry builder. The
journal maps action names to builders and renders nothing for an unmapped action,
so the builder is required rather than optional.

Recorded: source folder, per-file SHA-256, host-match verdict, inferred supercell
matrix, detected charge states, each recovered keyword value **and its source
file**, every conflict found, the set of runs offered versus imported, the
keyword catalog source and CASTEP version, and the application version doing the
import.

Idempotency key: the source folder, the seed name, and the ``.castep`` digest. A
repeat import offers to skip or re-import as new rather than duplicating
silently.

Removal
-------

Refuse while anything still references the site, run or structure; otherwise
delete the row, mapping entries, job, run, result, structure, site and copied
files as one unit, and append a provenance record.

The reference scan must be wider than the two obvious workflows. Check at least
quantum-corrections and energy-barrier definitions, comparison tables,
relaxation-analysis workspaces held in interface state, and muon-sites summary
snapshots. Missing any of these turns a removal into a dangling reference rather
than a refusal.

Layering
--------

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Concern
     - Layer
   * - ``.param`` reader, ``.castep`` header-dump reader
     - ``castep_io``
   * - Supercell inference, host matching
     - ``science``
   * - Folder scan, entity construction, file copy, provenance
     - ``services``
   * - Import wizard, row flags, read-only rendering, form-state mirroring
     - ``app``

**The commit necessarily splits across two layers.** The signature-equality
discipline requires a live widget, so the round trip through the widget's state
cannot happen in ``services``. The ``services`` layer builds the entities, files
and provenance; the ``app`` layer mirrors the new rows through the widget and
persists what comes back into *both* the definition and the stored form state.
Note that the form-configuration reader prefers the live widget's state over the
persisted copy, so writing only the persisted state is not sufficient. A
services-only API would appear to complete an import while leaving every row
rendering "ready".

Phasing
-------

1. **Parsers** — *done*. ``param_parser`` (including the scalar-value coercion
   moved down out of the app layer) and ``run_parameters``, the ``.castep``
   header-dump reader, with synthetic fixtures. Independently useful: it
   unblocks a parsed rather than re-rendered continue-relaxation diff.
2. **Scanner and inference** — *done*.
   :mod:`mufinder.services.legacy_import` turns a folder into a
   ``LegacyFolderScan``: site index, charge state, convergence, supercell
   matrix, merged parameters and conflicts per run. Pure with respect to the
   project — it mutates no document, writes no file, creates no entity — so the
   whole detection path is testable without a project on disk and the wizard can
   show a user what was found before anything is committed.

   Two properties are load-bearing. **Nothing raises for one bad run**: a
   directory of third-party output contains surprises, and the honest outcome is
   a report saying which runs are importable and why the others are not. And
   **blockers are distinct from warnings**: a blocker means the run cannot be
   imported (no ``.castep``, an unparseable input cell, a lattice that is not an
   integer supercell of the host), while a warning is something the user should
   know but that does not prevent import (an undetermined charge state, a muon
   labelled with a bare ``H``).

   Supercell inference reuses ``science.supercell_matching`` through a new
   lattice-level entry point, but with a **much tighter tolerance**. Structure
   comparison defaults to 0.2 because an experimental and a DFT lattice
   legitimately differ by a percent or so; a legacy input cell was constructed
   as an exact integer supercell of its host, so the only expected residual is
   ``.cell`` rounding, of order 1e-6. Keeping comparison's default here would
   accept a 7% lattice mismatch as a supercell and import a run against the
   wrong host.
3. **Row model** — *done*. The ``imported`` flag and batch id on
   ``MuonRelaxationRow``, threaded through the widget's per-row state, its state
   collection and restoration, the configuration serialiser pair, and the row
   signature. Deliberately ahead of entity construction, which writes definition
   rows that must already carry the flag.
4. **Entity construction and file materialisation** — *done*.
   :mod:`mufinder.services.legacy_import_commit` builds the five-entity chain,
   renders the seed files, archives the originals with digests, and appends one
   provenance record per batch with a matching journal entry. Tested against the
   eligibility gate *and* host resolution — but not "done" on eligibility alone,
   since a row cannot render completed with an energy until the next phase.

   The rendered ``.param`` resolves mutually exclusive keywords rather than
   merging them. A legacy run typically declares ``BASIS_PRECISION`` while the
   output echo reports the ``CUT_OFF_ENERGY`` CASTEP derived from it, so a
   merged parameter set legitimately holds both — and CASTEP treats them as
   exclusive. The resolved cutoff wins, because it is what the calculation
   actually used and what downstream jobs inherit; the superseded keyword is
   reported rather than dropped silently.

   Because ``services`` may not import ``app``, the definition rows this stage
   produces are plain payload dictionaries in the persisted configuration shape.
   Merging them into the active definition needs the widget and so belongs to
   the next phase.
5. **Table integration** — *done*. ``MainWindow.merge_imported_relaxation_rows``
   attaches a batch's rows to the active definition and mirrors them through the
   widget; imported rows render read-only; continuation, editing, in-table
   removal and batch configuration are gated; and the dashboard reset,
   resubmission and directory-cleanup paths refuse imported jobs.

   The gates key on an ``imported`` tag carried by the ``Job`` rather than on a
   lookup through its definition row, because every guard that must respect it
   already holds the job and may not hold the row. The reset guard sits inside
   the shared reset core, not only at its call sites: that core is reached both
   from the dashboard Reset action and from the lazy reset a dismissed row
   performs on resubmit, and its cleanup is irreversible.

   Read-only rendering reuses the widget's existing lock predicates rather than
   adding parallel checks, so an imported row locks exactly where a queued row
   already does. The marker is applied by style and tooltip, never by editing the
   label text: the label is read back as the row's identity, so decorating it
   would corrupt the row key.
6. **Wizard.** Folder picker, detected-run table with selection, conflict,
   catalog-fidelity and host-match warnings, commit.
7. **Removal cascade.**

Verification checklist
----------------------

- An imported site passes the eligibility gate and appears in both the Quantum
  Corrections source list and the Energy Barriers endpoint list.
- A quantum-corrections job and an NEB job generated from imported sites
  **actually submit**: host resolution returns the active host and no mismatch
  dialog appears. This is distinct from the eligibility check above — a site can
  pass every visible check and still fail here.
- A phonon job generated from an imported site inherits the legacy cutoff and
  k-point grid rather than project defaults.
- An imported row renders with its correct status, **total energy** and final
  fractional position.
- An imported row cannot be submitted, continued, edited in place, or included in
  a batch parameter edit.
- The imported flag survives a reconfigure: change an unrelated native row,
  re-register the definition, and confirm imported rows are still gated.
- A run imported as failed cannot be dismissed and then resubmitted, and no reset
  path can delete its ``.castep``, relaxed output cell, or archive. Assert the
  files still exist afterwards.
- A label collision with an existing native row does not overwrite either
  mapping.
- Re-importing the same folder does not duplicate entities.
- Removal is refused while a phonon, NEB, comparison table, or relaxation
  analysis workspace references the site.
- Round trip: save, close and reopen the project; imported rows, their statuses
  and their flags survive.

Open risks
----------

- **Legacy datasets using a bare ``H``** rather than ``H:mu``. The legacy muon
  label detector accepts both, but the eligibility gate hard-requires ``H:mu``,
  so the importer must relabel and record that it did.
- **Relaxed-cell legacy runs**, where the cell was not held fixed, break the
  ``S * L_host`` identity. Inference uses the input cell for this reason, but such
  runs also violate the native muon-relaxation invariant of a fixed cell. Detect
  and warn.
- **Multi-restart ``.castep`` files.** The existing parser reads the last run
  segment for status, but the header dump repeats per restart. Read the *last*
  dump.
- **Large folders.** Parsing dozens of multi-megabyte outputs must not run on the
  Qt main thread.
- **Catalog drift between imports.** Two imports into the same project under
  different CASTEP versions can map different keyword sets. The recorded CASTEP
  version per import makes this diagnosable rather than invisible.
- **No available dataset exercises spin polarisation, muonium, or Hubbard U.**
  Both real datasets are entirely positive-muon and non-spin-polarised with no
  Hubbard-U block. Fixtures are synthetic in any case, so this costs nothing
  structurally — but it does mean the magnetic-host file shapes are inferred from
  the legacy *source* rather than observed in output. Review them against CASTEP
  documentation before relying on them, and do not claim them verified against
  real data.

  This is itself informative rather than a gap in the samples: dataset B's host
  *is* a magnetic material, yet its relaxations were run non-spin-polarised, with
  the magnetism entering only through the dipolar-field step. In the legacy
  workflow a magnetic host usually does **not** imply a spin-polarised DFT
  relaxation, so the common import case is the simple one.
- **Position block variants.** Dataset A's cells use ``POSITIONS_FRAC`` and
  dataset B's use ``POSITIONS_ABS``. Exercising both found a real defect: the
  absolute-position branch of the cell parser converted coordinates by
  round-tripping through ``ase.Atoms``, which resolves every symbol against the
  periodic table and so raised on the ``H:mu`` species *label*. Every muonated
  cell written with ``POSITIONS_ABS`` was unparseable, which affected far more
  than this feature. The conversion is now arithmetic and label-agnostic. Both
  variants remain exercised by the scanner tests.
