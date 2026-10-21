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
4. **Export.** Click ``Export...`` and choose where to save the file. The
   suggested filename is ``<project-name>-journal`` with the extension for the
   selected format; characters other than letters, digits, ``-`` and ``_`` in
   the project name are replaced by underscores. Both formats mirror the
   curated Journal content rather than the raw provenance records.

The Journal page
----------------

.. figure:: ../_generated/screenshots/journal_page.png
   :alt: The Journal with milestone entries for the convergence setup and the
         generated muon sites.

   The Journal of a project with configured sweeps and generated sites.

Entries are grouped by local calendar day under human-readable date headers
such as ``Tuesday 21 April 2026``. The Journal intentionally does not show
precise clock times.

Each entry carries a stamp: ``Milestone``, ``Completed``, ``Attention``, or
``Failed``. Entries tied to a workflow also name it (for example
``Cutoff Convergence`` or ``Muon Relaxations``).

What the Journal includes
~~~~~~~~~~~~~~~~~~~~~~~~~

The Journal highlights the major workflow milestones that are most useful for
reconstructing what happened during a project, including:

- structure import
- workflow configuration for convergence, pristine relaxation, and UEP
- adopted project defaults, including the functional a cutoff or k-point
  default was converged with, any change to the project default XC
  functional, and defaults entered or cleared by hand
- muon candidate generation (with an ``Attention`` entry when fewer sites were
  generated than requested) and UEP promotion
- calculation submission milestones and continuation runs
- terminal calculation outcomes such as completed, failed, stopped, cancelled,
  and reset
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
