Journal
=======

The ``Journal`` is a curated, day-grouped narrative of what happened in a
project: what was imported, configured, submitted and finished. Use it to
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
   date headers (see `The Journal page`_).
3. **Choose a format.** Pick ``Plain text (.txt)`` or ``Markdown (.md)`` in the
   ``Export format`` selector.
4. **Export.** Click ``Export…`` and choose where to save the file. The
   suggested filename is ``<project-name>-journal`` with the extension for the
   selected format; characters other than letters, digits, ``-`` and ``_`` in
   the project name are replaced by underscores. Both formats mirror the
   curated Journal content rather than the raw provenance records.

The Journal page
----------------

.. figure:: ../_generated/screenshots/journal_page.webp
   :alt: The Journal with milestone entries for the convergence setup and the
         generated muon sites.

   The Journal of the CoF₂ project (:doc:`/user_guide/examples/cof2`) with configured
   sweeps and generated sites.

Entries are grouped by local calendar day under human-readable date headers
such as ``Tuesday 21 April 2026``. The Journal intentionally does not show
precise clock times.

Each entry carries a stamp: ``Milestone``, ``Completed``, ``Attention``, or
``Failed``. Entries tied to a workflow also name it (for example
``Cutoff Convergence`` or ``Muon Relaxations``).

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
- creating a relaxation analysis workspace, and applying a grouping to it (the
  number of distinct sites, the method, its tolerances and the space group)
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
- some analysis-only artifacts that are not yet recorded as dedicated
  provenance events

Limitations
-----------

The Journal reflects only what has been explicitly recorded as the work
happened. History is not inferred for older projects, which may therefore
appear sparse until new work is recorded.

For developers
--------------

The Journal is built from the project's ``provenance.jsonl`` stream; see
:doc:`/architecture`.
