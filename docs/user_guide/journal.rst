Journal
=======

MuFinder's ``Journal`` is the user-facing view of the project's structured
provenance stream. Open it from the ``Journal`` entry at the bottom of the
project navigator, or with ``Open Journal`` on the :doc:`/user_guide/project_home`.

What the Journal includes
-------------------------

.. figure:: ../_generated/screenshots/journal_page.png
   :alt: The Journal with milestone entries for the convergence setup and the
         generated muon sites.

   The Journal of a project with configured sweeps and generated sites.

The Journal currently highlights the major workflow milestones that are most
useful for reconstructing what happened during a project, including:

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

Each entry carries a stamp: ``Milestone``, ``Completed``, ``Attention``, or
``Failed``. Entries tied to a workflow also name it (for example
``Cutoff Convergence`` or ``Muon Relaxations``).

Entries are grouped by local calendar day under human-readable date headers
such as ``Tuesday 21 April 2026``. The Journal intentionally does not show
precise clock times.

What the Journal omits
----------------------

The Journal is a curated narrative rather than a raw event ledger.

It intentionally omits:

- intermediate job states such as ``queued``, ``running``, and ``retrieving``
- inferred or backfilled history for older projects that lack the newer
  provenance milestones
- some analysis-only artifacts that are not yet recorded as dedicated
  provenance events

Export formats
--------------

Choose a format in the ``Export format`` selector and click ``Export...``.
Two formats are available:

- ``Plain text (.txt)``
- ``Markdown (.md)``

Both mirror the curated Journal content rather than the raw provenance records.

The suggested filename is ``<project-name>-journal`` with the extension for the
selected format; characters other than letters, digits, ``-`` and ``_`` in the
project name are replaced by underscores.

Known limitations
-----------------

The Journal reflects only what has been explicitly written into
``provenance.jsonl``.

Older projects created before the journal-specific submission and terminal
milestones were added may therefore appear sparse until new work is recorded.
