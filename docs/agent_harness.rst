Agent Harness
=============

MuFinder uses an agent-oriented development harness inspired by OpenAI's
`Harness engineering <https://openai.com/index/harness-engineering/>`_ article:
humans steer, the repository records durable intent, and executable checks make
future agent work reliable.

The goal is practical agent legibility. A future Codex run should be able to
enter the repository, find the right source of truth, make a scoped change, and
validate it without depending on hidden chat context.

Harness contract
----------------

The harness has four parts:

- ``AGENTS.md`` is the short map that points to deeper repository knowledge.
- ``docs/`` is the system of record for product behavior, architecture,
  quality gates, and compatibility policy.
- ``tests/`` contains both feature tests and structural tests that enforce
  architecture and harness invariants.
- CI runs the same core gates that local agents are expected to run before a
  change is ready for review.

Agent workflow
--------------

For non-trivial changes, use this loop:

1. Read ``AGENTS.md`` and the relevant document in ``docs/``.
2. Identify the package layer that owns the behavior in
   :doc:`architecture`.
3. Update implementation, tests, and documentation together.
4. Run focused tests for the touched area.
5. Run the full quality gates before merging or publishing.

When an agent finds a missing capability, stale assumption, repeated pattern,
or architectural ambiguity, prefer turning that lesson into repository-local
documentation or an executable test. The durable fix is the harness, not a
one-off prompt.

Sources of truth
----------------

Use these documents as the first stop for common tasks:

- :doc:`project_specification` defines product and workflow requirements.
- :doc:`architecture` defines package responsibilities and dependency
  directions.
- :doc:`implemented_functionality` summarizes the current feature surface.
- :doc:`quality_and_automation` lists quality gates and targeted local checks.
- :doc:`compatibility_policy` records compatibility aliases and expiry rules.
- :doc:`magnetic_fields` defines the current magnetic-field workflow behavior.

Executable guardrails
---------------------

The harness is intentionally checked by tests:

- ``tests/test_agent_harness.py`` verifies that ``AGENTS.md`` remains a short
  map, required docs are indexed, package import boundaries are respected, no
  new per-workflow submit/materialize handlers appear on ``MainWindow``, app
  plots avoid ``subplots_adjust`` and do not add new direct
  ``FigureCanvasQTAgg`` construction, the pull-request template stays
  discoverable, and tracked source, test, and docs trees contain no Python
  bytecode artifacts (see :doc:`quality_and_automation` for the full list).
- Feature-specific tests continue to live next to the relevant domain, service,
  app, docs, or packaging area.

Add new harness tests when a review finding identifies a pattern that should
not regress across the codebase.

Entropy management
------------------

MuFinder should prefer small, continuous cleanup over large occasional
rewrites. Good cleanup candidates are:

- duplicated workflow orchestration that can move into a shared controller
- feature-local styling that belongs in ``mufinder.app.theme``
- compatibility aliases without an entry in :doc:`compatibility_policy`
- UI behavior that depends on platform-specific input handling
- code that crosses package layers without a deliberate architecture update

If a rule is important enough to mention repeatedly in review, promote it into
documentation or a test so future work inherits it automatically.
