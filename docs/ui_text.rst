UI text style
=============

These rules cover every string MuFinder shows: buttons, menu items, tabs,
window and dialog titles, message boxes, labels, tooltips, status messages,
progress text and the journal. A message that a service or science module
builds and the app displays follows them too.

Capitalisation
--------------

- **Sentence case everywhere**: buttons, menu items, tabs, window and dialog
  titles, message boxes and labels. Capitalise the first word and proper
  nouns only (``Delete supercell``, ``Import legacy relaxations…``).
- Proper nouns keep their capitals: product and program names (MuFinder,
  CASTEP, SLURM), abbreviations (UEP, NEB, DFT, MPI, SSH) and the names of
  workflow pages when the text refers to the page (Magnetic Fields, Quantum
  Corrections, Energy Barriers).

Punctuation
-----------

- **Ellipsis.** A command that opens a dialog needing more input ends in
  ``…``, the one-character ellipsis (``Export CSV…``, ``Configure…``). A
  command that acts at once, or only asks to confirm, has none. Progress text
  ends in ``…`` too (``Connecting to SCARF…``), and so does text cut short.
  Never write three dots.
- **Status messages** are in the past tense and have no final period
  (``Submitted 3 jobs``, ``Saved project``).
- **Tooltips and message text** are full sentences that end in a period. A
  tooltip that only names a value (a column header, ``Relaxed site: Mu-3``)
  stays a label.
- **Menu paths** are written ``Setup > CASTEP`` and ``Setup > HPC Clusters``,
  with ``>`` between the levels. A change from one value to another is written
  with ``→`` (``2 terms → 4 terms``), never ``->``.
- **Names** of projects, structures, workspaces and other things the user
  named, and of buttons or options the text refers to, go in single quotes:
  ``Delete analysis 'Fe sites'?``, ``Use 'Go to Analysis' to cluster…``.

Counts
------

A count agrees with its number: ``1 job``, ``3 jobs``. Write it with
``mufinder.domain.text.plural`` (``plural(n, "job")``; pass the plural when it
is not the singular with an ``s``, ``plural(n, "analysis", "analyses")``, or to
make a verb agree, ``plural(n, "job was", "jobs were")``). Never write ``(s)``.

Actions and confirmations
-------------------------

- Accept and confirmation buttons name the action (``Delete supercell``,
  ``Run 6 jobs``, ``Save``), never ``Yes``, ``No`` or ``OK`` for a choice. The
  safe answer is ``Cancel`` and is the default button. Ask through
  ``confirm_destructive`` or ``confirm_choice`` in
  ``mufinder.app.workflow_table_utils``.
- **Remove** takes something out of a list or out of the project and keeps its
  files, or asks what to do with them (``Remove rows``, ``Remove cluster``).
- **Delete** destroys saved data or files (``Delete jobs``, ``Delete
  supercell``, ``Delete workspace``). The button and its confirmation use the
  same word.

Units and terms
---------------

- Units are ``Å``, ``Å⁻¹`` and ``eV``, with a space between the number and the
  unit (``0.05 Å⁻¹``, ``500 eV``).
- Terms are spelled one way: ``k-point``, ``cutoff``, ``supercell`` and
  ``cluster`` (the menu is ``Setup > HPC Clusters``).

What not to show
----------------

- Never show internal identifiers: enum values, calculation types, field
  names, UUIDs or ``ui_state`` keys. Show the name the user knows.
- Show CASTEP keywords only where the user needs them, such as the parameter
  Info panel and keyword search.

How it is checked
-----------------

``tests/test_agent_harness.py`` reads every string literal under
``src/mufinder`` (docstrings and ``numpy.einsum`` subscripts excepted) and fails
on ``(s)`` after a word, on three dots and on ``->``. The other rules are kept
in review; :doc:`/quality_and_automation` lists the harness checks.
