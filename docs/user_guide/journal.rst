Journal
=======

The ``Journal`` is a curated, day-grouped narrative of what happened in a
project: what was imported, configured, submitted and finished, and what was
found. Use it to
reconstruct the history of a project, for example when writing up a study. It
is the user-facing view of the project's structured provenance stream.

Before you start
----------------

Open a project. The Journal fills in as you work, so it is empty on a new
project.

Reading and exporting the Journal
---------------------------------

1. **Open it.** Select ``Journal`` at the bottom of the project navigator, or
   click ``Open Journal`` on the :doc:`/user_guide/project_home`.
2. **Read the entries.** Entries are grouped by local calendar day under
   date headers, one compact card each (see `The Journal page`_).
3. **Expand an entry.** Click a card's top line, or press ``Enter`` or ``Space``
   while it has focus, to show what was logged (see `Expanded cards`_). Do the
   same again to close it.
4. **Go to the work.** On an expanded card, click ``Open in <workflow>`` to open
   that workflow's page.
5. **Choose a format.** Pick ``Plain text (.txt)`` or ``Markdown (.md)`` in the
   ``Export format`` selector.
6. **Export.** Click ``Export…`` and choose where to save the file. The
   suggested filename is ``<project-name>-journal`` with the extension for the
   selected format; characters other than letters, digits, ``-`` and ``_`` in
   the project name are replaced by underscores. Both formats give each
   entry's curated summary, not the details of an expanded card or the raw
   provenance records.

The Journal page
----------------

.. figure:: ../_generated/screenshots/journal_page.webp
   :alt: The Journal with milestone entries for the convergence setup and the
         generated muon sites.

   The Journal of the CoF₂ project (:doc:`/user_guide/examples/cof2`) with configured
   sweeps and generated sites.

Entries are grouped by local calendar day under human-readable date headers
such as ``Tuesday 21 April 2026``.

Each entry is a card of two lines. The first has a stamp dot, the title, the
workflow if the entry is tied to one (for example ``Cutoff Convergence`` or
``Muon Relaxations``) and the local time. The second is a one-line summary, cut
short with ``…`` when it does not fit; expanding the card shows all of it.

The dot and the card's edge say what kind of entry it is. Holding the pointer
over the dot names the kind.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Stamp
     - Colour
     - Marks
   * - ``Milestone``
     - Neutral
     - Routine steps: imports, configurations, launches
   * - ``Completed``
     - Gold
     - Adopted defaults, findings (`Findings`_), new comparison tables, finished
       calculations
   * - ``Attention``
     - Amber
     - Something to check: a stopped or cancelled calculation, a generation
       that fell short, an import with skipped runs or conflicts
   * - ``Failed``
     - Red
     - A failed calculation; red is used for nothing else

Expanded cards
~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/journal_expanded_card.webp
   :alt: An expanded Journal card for a muon relaxation batch, with the supercell,
         one row per site, the exact time and an Open in Muon Relaxations button.

   An expanded batch card of the CoF₂ project (:doc:`/user_guide/examples/cof2`).

An expanded card shows the entry's whole title and summary, the logged details
below, and the exact time with its time zone.

- **Configurations** show the supercell and the shared parameters that were
  recorded: spin, functional, cutoff and k-point grid. A parameter left at the
  project default is not in the record, and one that differs between rows is not
  shared, so neither appears. A convergence sweep gives its range, and a row list
  gives its count and first three names.
- **Batches** show the parameters their rows share, as above, and one row per
  calculation with its status, ``ΔE (meV)``, ``Max force (eV/Å)`` and warnings.
  ``ΔE`` is relative to the lowest total energy in that batch, whatever the
  calculations in it are. A failed calculation's reason is listed above the table.
- **Adopted defaults** show the tolerances they met and the functional.
- **Muon site generation** shows the run's inputs and random seed in a full
  sentence, ready for a methods section.
- **Legacy imports** show the source folder, each skipped run with its reason,
  and each parameter conflict with how it was resolved.
- **Findings** show their numbers; see `Findings`_.

``Open in <workflow>`` opens the entry's workflow page, ``Open in Host Structure``
an imported structure's page, and ``Open in '<table>'`` a comparison table. The
button is not shown when the page no longer exists, as for a deleted comparison
table.

Batch cards
~~~~~~~~~~~

One submission gives one card. It stays on the day the batch was launched and
says how the batch stands: ``3 of 10 completed`` while jobs are still running,
``10 of 10 completed`` or ``8 completed, 2 failed`` once they have all finished.

- A failed, stopped or cancelled job also keeps its own card, on the day it
  happened and with its reason, so a problem stands out.
- A batch that finishes on a later day adds a short line under that day, such as
  ``Finished: 10 of 10 completed``, saying which launch it belongs to.
- Resets and continuations keep their own cards, and a continuation's new run is
  a submission of its own. A completion with no submission on record, as in an
  older project, also has its own card.
- Dismissing several calculations gives one card (``Calculations dismissed``);
  dismissals in a row on one day read as one action.

Entries use the names the rest of MuFinder shows. A calculation is named by
its workflow and its row in that workflow's table (``pristine relaxation
relax_1``, ``muonium relaxation Random Site 1``, ``phonon calculation for
Attached-F Site 1``); a calculation whose record names no row, such as a
convergence point, is named by its job name. Where it ran is ``this
computer`` or the cluster's name, and a generation method is named as the
``Generate Sites`` method picker names it.

Findings
~~~~~~~~

Each result that concludes a step of a study gives one entry, with the measured
numbers in its summary line and in full on the expanded card.

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Entry
     - Written when
     - Example summary and details
   * - ``UEP minima found``
     - A UEP batch search is grouped
     - ``UEP: 3 distinct minima; lowest at (0.000, 0.500, 0.000)``. Each minimum's
       position, starts and ``ΔE (meV)``, and how the search was grouped.
   * - ``Relaxed sites grouped``
     - A grouping is applied to an analysis workspace
     - ``Positive muon: 4 distinct sites; lowest 4c, next cluster 2 +112 meV``. Each
       site's position and ``ΔE (meV)``, the method, its tolerances and the space
       group. A site is named by its Wyckoff label, or by its cluster without one.
   * - ``Magnetic field calculated``
     - A Magnetic Fields calculation gives new values
     - ``Field at 4c: 0.42 T (dipolar + contact)``, a range over equivalent sites
       that differ, or the RMS of an incommensurate distribution. Each site's
       ``|B|`` and field vector, or RMS and mean field, and the applied field.
   * - ``Polarisation simulated``
     - A polarisation simulation gives new traces
     - ``ZF polycrystal simulation from 2 sites``. Its title, its field sources with
       their weights, its end time and damping.
   * - ``Zero-point energy calculated``
     - A Quantum Corrections phonon job completes
     - ``ZPE at 8g · cluster 1: 0.558 eV; ordering unchanged``. The charge state, muon modes,
       zero-point widths and ``w1``. With two or more comparable sites, a table of
       ``ΔE``, ``ZPE`` and ``ΔE + ZPE``. A phonon warning, such as the muon-mass
       check, stamps the entry ``Attention``.
   * - ``Energy barrier calculated``
     - An Energy Barriers NEB job completes
     - ``Barrier 8g · cluster 1 → 16m · cluster 2: 0.495 eV``. The charge state, reverse barrier, endpoint
       ``ΔE``, images and path length.

``ordering unchanged`` compares the sites of the same charge state and cell that
have a zero-point energy so far: it says whether adding it reorders them. With
one such site the phrase is left out, so the first site of a batch never has it.

A relaxed site is named by its label on the Muon Sites summary when it is pinned
(``8g · cluster 1``), and otherwise by its own label. A field or simulation
calculated at an unrelaxed UEP site says it is an estimate.

Recalculating with identical inputs does not add an entry when it repeats the
latest finding about the same thing (the same UEP job, workspace, Magnetic Fields
page, simulation or row) on the same day; on a later day it does. Each record keeps
a fingerprint of its inputs for this. A UEP
grouping, a site grouping and a magnetic field also fingerprint what they found, so
adding the ZPE to a UEP search, or regrouping after a relaxation was run again,
is a new entry. A Magnetic Fields calculation that gives the values already stored
adds no entry on any day.

Going back to an earlier result adds an entry: calculating ``A``, ``B`` and ``A``
again on one day gives three, so the last entry about anything is always current. A
new polarisation simulation has its own entry even with the same settings as another.

What the Journal includes
~~~~~~~~~~~~~~~~~~~~~~~~~

The Journal highlights the major workflow milestones that are most useful for
reconstructing what happened during a project, including:

- structure import, and a change of the active host structure (naming the
  structure)
- workflow configuration for convergence, pristine relaxation, UEP, muon
  relaxations (the number of relaxations in each charge state, for example
  ``Prepared 14 muon relaxations for later submission (7 positive muon,
  7 muonium).``), Quantum Corrections and Energy Barriers
- adopted project defaults, including the convergence criterion they met
  (for example ``energy within 0.001 eV/atom and max force within 0.05 eV/Å
  of the finest cutoff``), the functional a cutoff or k-point default was
  converged with, any change to the project default XC functional, and
  defaults entered or cleared by hand
- muon candidate generation, with the run's parameters in one line and its
  random seed (for example ``Random, 6 starts, ≥ 1.79 Å apart, ≥ 1.0 Å from
  atoms, P4₂/mnm (16 ops), seed 418273.``), an ``Attention`` entry when fewer
  sites were generated than requested, and one when the host's symmetry could
  not be detected so generation treated it as having none; UEP promotion
- ``Cluster Positions…`` (the anchor and the starts moved), staging generated
  sites for relaxation, and removing generated sites
- creating a relaxation analysis workspace
- findings: UEP minima, grouped sites, magnetic fields, polarisation simulations,
  zero-point energies and energy barriers (see `Findings`_)
- calculation submission milestones with each batch's outcome, and continuation
  runs. A muon-relaxation batch names its charge state (``Muon relaxations
  launched: muonium``), so a positive-muon run and a muonium run of the same
  sites read differently
- failed, stopped and cancelled calculations with their reason, resets, and
  calculations dismissed from the Job Dashboard. A muon relaxation's outcome
  names its charge state (``Muonium relaxation failed``); where a calculation
  with its own card has a total energy, it is given to 0.1 meV and its largest
  force to three significant figures
- comparison-table creation
- import of legacy relaxations run outside MuFinder, and removal of imported
  relaxations

What the Journal omits
~~~~~~~~~~~~~~~~~~~~~~

The Journal is a curated narrative rather than a raw event ledger. It
intentionally omits:

- intermediate job states such as ``queued``, ``running``, and ``retrieving``
- inferred or backfilled history for older projects that lack the newer
  provenance milestones
- a UEP batch search that was never grouped, whose distinct minima are not known
- dipolar-tensor and nuclear-relaxation calculations on the Magnetic Fields page

Limitations
-----------

The Journal reflects only what has been explicitly recorded as the work
happened. History is not inferred for older projects, which may therefore
appear sparse until new work is recorded.

For developers
--------------

The Journal is built from the project's ``provenance.jsonl`` stream; see
:doc:`/architecture`.
