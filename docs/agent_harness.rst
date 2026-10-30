Agent Harness
=============

MuFinder is developed largely by coding agents steered by humans. The repository, not
the chat transcript, records intent: ``AGENTS.md`` states the rules, ``docs/`` records
the contracts, and ``tests/`` makes both executable. An agent starting from a clean
checkout should be able to find the right source of truth, make a scoped change, and
validate it without hidden context.

Harness contract
----------------

The harness has four parts:

- ``AGENTS.md`` is the single instruction file for every agent, whichever vendor
  provides it. ``CLAUDE.md`` imports it (``@AGENTS.md``) and
  ``.github/copilot-instructions.md`` points to it, so no rule lives only in a
  vendor-specific file. It states the rules and links to the docs for the rationale,
  and stays within 140 lines. Inventories that grow with the code, such as the
  shared home for each cross-workflow concern, live in ``docs/`` (:doc:`/architecture`,
  Shared workflow infrastructure), where a harness test checks that each listed
  module still exists.
- ``docs/`` is the system of record for product behavior, architecture, quality gates,
  and compatibility policy.
- ``tests/`` contains both feature tests and structural tests that enforce architecture
  and harness invariants.
- CI runs the same core gates that local agents are expected to run before a change is
  ready for review.

Agent workflow
--------------

For non-trivial changes, use this loop:

1. Read ``AGENTS.md`` and the relevant document in ``docs/``.
2. Identify the package layer that owns the behavior in :doc:`/architecture`, and the
   shared home for any cross-workflow concern (:doc:`/architecture`, Shared
   workflow infrastructure).
3. Update implementation, tests, and documentation together.
4. Run focused tests for the touched area.
5. Run the full quality gates before merging or publishing.

When an agent finds a missing capability, stale assumption, repeated pattern, or
architectural ambiguity, prefer turning that lesson into a rule in ``AGENTS.md``, a
contract in ``docs/``, or an executable test. The durable fix is the harness, not a
one-off prompt.

Sources of truth
----------------

Use these documents as the first stop for common tasks:

- :doc:`/project_specification` defines product and workflow requirements.
- :doc:`/architecture` defines package responsibilities, dependency directions,
  the shared homes for cross-workflow code and the workflow-controller contract.
- :doc:`/user_guide/overview` summarizes the current feature surface.
- :doc:`/quality_and_automation` lists quality gates, local commands, and targeted checks.
- :doc:`/compatibility_policy` records compatibility aliases and expiry rules.
- :doc:`/job_lifecycle` is the job state-transition contract.
- :doc:`/ui_text` is the style for every string the application shows.

Executable guardrails
---------------------

``tests/test_agent_harness.py`` holds the structural checks; :doc:`/quality_and_automation`
(Agent harness checks) lists what they currently verify. Feature-specific tests live next
to the relevant domain, service, app, docs, or packaging area.

Add a harness test when a review finding identifies a pattern that should not regress
across the codebase. Ratchets (a pinned set that may only shrink) are the preferred shape
for retiring an existing pattern gradually.

Entropy management
------------------

MuFinder prefers small, continuous cleanup over large occasional rewrites. Good cleanup
candidates are:

- per-workflow widget or controller code that duplicates a sibling workflow and belongs
  in one of the shared homes listed in :doc:`/architecture`
- feature-local styling that belongs in ``mufinder.app.theme``
- compatibility aliases without an entry in :doc:`/compatibility_policy`, or past their
  expiry condition
- UI behavior that depends on platform-specific input handling
- code that crosses package layers without a deliberate architecture update
- documentation written as history rather than as a current-state contract

If a rule is important enough to mention repeatedly in review, promote it into
``AGENTS.md`` or a test so future work inherits it automatically.
