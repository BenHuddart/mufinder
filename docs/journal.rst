Journal
=======

MuFinder's ``Journal`` is the user-facing view of the project's structured
provenance stream.

What the Journal includes
-------------------------

The Journal currently highlights the major workflow milestones that are most
useful for reconstructing what happened during a project, including:

- structure import
- workflow configuration for convergence, pristine relaxation, and UEP
- adopted project defaults, including the functional a cutoff or k-point
  default was converged with, any change to the project default XC
  functional, and defaults entered or cleared by hand
- muon candidate generation and UEP promotion
- calculation submission milestones
- terminal calculation outcomes such as completed, failed, stopped, cancelled,
  and reset
- comparison-table creation

Entries are grouped by local calendar day and use human-readable date headers.
The Journal intentionally does not show precise clock times.

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

The Journal can be exported from the GUI in two formats:

- plain text ``.txt`` files
- Markdown ``.md`` files

The default filename format is ``<project-name>-journal`` with the extension
for the selected format.

Known limitations
-----------------

The Journal reflects only what has been explicitly written into
``provenance.jsonl``.

Older projects created before the journal-specific submission and terminal
milestones were added may therefore appear sparse until new work is recorded.
