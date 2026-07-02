Compatibility and Deprecation Policy
====================================

MuFinder keeps a small number of compatibility aliases and persisted-field
shims so older projects, tests, and user workflows continue to load while the
workflow model evolves. New aliases should be added only when they protect a
saved-project migration path or a documented public import path.

Compatibility ledger
--------------------

.. list-table::
   :header-rows: 1
   :widths: 24 20 36 20

   * - Name
     - Location
     - Purpose
     - Expiry policy
   * - ``cutoff_energies``
     - ``ConvergenceConfig``
     - Legacy cutoff-only configuration field retained while convergence
       workflows move to generic ``sweep_parameter`` / ``sweep_values`` data.
     - Keep until the next saved-project schema version bump includes an
       automatic migration from ``cutoff_energies`` to ``sweep_values``.
   * - ``CutoffRunState``
     - ``mufinder.app.convergence_types`` and ``mufinder.app.main_window``
     - Import alias for ``SweepPointRunState`` used by older cutoff tests and
       external automation.
     - Keep through the same schema bump as ``cutoff_energies``; remove only
       after app-level call sites and tests import ``SweepPointRunState``.
   * - ``ConvergenceWidget``
     - ``mufinder.app.convergence_widget``
     - Import alias for ``CutoffConvergenceWidget`` kept for public app imports
       and old tests.
     - Public compatibility alias. Keep until a major application release or an
       explicitly documented API break.
   * - ``recommended_cutoff``
     - ``mufinder.science.convergence_analysis``
     - Backward-compatible helper name for cutoff-only analysis callers.
     - Public science helper. Keep until a major release provides a documented
       replacement and migration note.
   * - ``recommended_cutoff_*`` project fields
     - ``Project`` / project manifest
     - Persisted project defaults used by current cutoff, k-point, relaxation,
       and UEP setup flows.
     - Current project data, not deprecated. Do not remove without a schema
       migration and replacement project-default model.
   * - ``ConvergenceWorkflow``
     - ``mufinder.workflow.convergence`` (re-exported from
       ``mufinder.workflow``)
     - Deprecated class wrapper kept for public imports while the workflow layer
       converges on ``expand_convergence_sweep_jobs``.
     - Public import alias. Keep until a major release or one minor release
       after ``expand_convergence_sweep_jobs`` has shipped as the documented
       replacement.
   * - ``build_convergence_workflow``
     - ``mufinder.app.convergence_adapter`` (re-exported from ``mufinder.app``)
     - Deprecated public helper that returns the ``ConvergenceWorkflow`` shim.
       Its happy path is orphaned now that the GUI calls
       ``expand_convergence_jobs`` /
       ``expand_convergence_sweep_jobs`` directly; kept only for external
       callers.
     - Public helper alias. Retires together with ``ConvergenceWorkflow``: keep
       until a major release or one minor release after
       ``expand_convergence_sweep_jobs`` has shipped as the documented
       replacement.
   * - ``MuonSiteWorkflow``
     - ``mufinder.workflow.muon_site`` (re-exported from ``mufinder.workflow``)
     - Deprecated class wrapper kept for public imports while the workflow layer
       converges on ``expand_muon_site_jobs``.
     - Public import alias. Keep until a major release or one minor release
       after ``expand_muon_site_jobs`` has shipped as the documented
       replacement.

Rules for new shims
-------------------

- Every new compatibility alias should have a row in this ledger.
- Project-file shims must state the schema version or migration event that can
  remove them.
- Public import aliases should be removed only in major releases or after a
  documented replacement has existed for at least one minor release.
- Tests that exercise legacy data should name the legacy field explicitly, so
  future cleanup work can find the remaining callers with ``rg``.

