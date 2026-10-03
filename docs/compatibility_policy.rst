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
   * - ``imported`` / ``import_batch_id`` row fields
     - ``MuonRelaxationRow`` / muon-relaxation definition ``configuration``
     - Mark a relaxation row ingested from a calculation performed outside
       MuFinder, and group the rows one import produced. Both are additive with
       defaults, so a project saved before the legacy import existed loads with
       every row treated as natively staged, which is correct.
     - Current project data, not deprecated. These fields are load-bearing for
       read-only enforcement: a row whose ``imported`` flag is lost becomes
       editable and resubmittable while its imported entities and files remain,
       so neither field may be dropped without a schema migration that first
       relocates the marker.
   * - Task-only ``cutoff_parameter_rows`` entries
     - ``convergence_form`` UI state, read by
       ``CutoffConvergenceWidget._set_cutoff_parameter_values``
     - A cutoff row saved with parameters of exactly ``{"task": ...}`` and no
       advanced keywords is read back as "uses the defaults". A restore bug
       wrote such entries for default rows, which made the rows stop matching
       the definition their jobs ran under, so they were submitted again. The
       parameter dialog cannot produce a task-only row, so nothing a user
       configured is reinterpreted.
     - Read-side repair. Keep until a saved-project schema version bump
       includes a migration that drops task-only rows from ``convergence_form``.
   * - ``recommended_cutoff_*`` project fields
     - ``Project`` / project manifest
     - Persisted project defaults used by current cutoff, k-point, relaxation,
       and UEP setup flows.
     - Current project data, not deprecated. Do not remove without a schema
       migration and replacement project-default model.
   * - ``recommended_xc_functional`` /
       ``recommended_cutoff_xc_functional`` /
       ``recommended_kpoint_xc_functional`` project fields
     - ``Project`` / project manifest
     - The project default XC functional, and the functional each convergence
       default was converged with. All three are additive and default to
       ``None``, so a project saved before they existed loads with no project
       functional and no recorded converged-with functional: workflows keep
       their schema-default functional and nothing is flagged as stale, which
       matches the pre-existing behaviour. ``recommended_cutoff_method`` /
       ``recommended_kpoint_method`` may now also hold ``manual`` for a
       hand-entered default; older builds display that string verbatim.
     - Current project data, not deprecated. Do not remove without a schema
       migration and replacement project-default model.
   * - ``execution_target`` / ``num_processes`` /
       ``remote_job_id`` / ``remote_directory`` job fields
     - ``Job`` (``mufinder.domain.models``) / job entity files
     - Load-bearing persisted job-contract state, not deprecation shims.
       ``execution_target`` and ``num_processes`` drive resubmission and
       dashboard columns; ``remote_job_id`` and ``remote_directory`` are the
       metadata that lets ``_sync_cluster_tracking_from_document`` re-register
       persisted active cluster jobs after a project is reopened. Orphaned
       local-job reconciliation also reads ``execution_target`` to decide which
       active jobs to stop on project open.
     - Current job data, not deprecated. Renaming or removing any of these
       fields breaks reload of saved projects and cluster re-attachment, so it
       requires a schema-version bump with a migration entry (and an update to
       the persistence codec and the reconciliation/tracking call sites).
   * - ``last_checked_at`` / ``last_status_source`` /
       ``last_status_message`` job fields
     - ``Job`` (``mufinder.domain.models``) / job entity files
     - Persisted monitor metadata, not deprecation shims. ``last_checked_at``
       records when a job's status was last confirmed (scheduler poll, local
       runner event, reconciliation, or watchdog); ``last_status_source`` names
       that origin; ``last_status_message`` carries non-failure context such as
       the raw scheduler state or retrieval-retry progress. The dashboard uses
       them to distinguish "still queued" from "not checked recently" and to
       render stale/offline markers. All three are optional and default to
       ``None``, so projects saved before their introduction load unchanged.
     - Current job data, not deprecated. Removing or renaming any of these
       fields silently degrades staleness display for existing projects, so it
       requires a schema-version bump with a migration entry and updates to the
       persistence codec and dashboard rendering.
   * - ``dismissed_at`` job field
     - ``Job`` (``mufinder.domain.models``) / job entity files
     - Persisted dismissal timestamp for terminal jobs (failed, stopped, or
       cancelled) hidden from the dashboard via the Dismiss action. The job
       record, its runs, results, and provenance remain in the project;
       only dashboard visibility changes. Optional with a ``None`` default, so
       older projects load unchanged and are fully visible. The app layer also
       derives a submittable workflow-row status from the stamp (see
       ``docs/job_lifecycle.rst``, Dismissed rows stay resubmittable) — a
       read-time projection, not extra persisted state.
     - Current job data, not deprecated. Removing the field un-dismisses jobs
       in existing projects, so it requires a schema-version bump with a
       migration entry and an update to the persistence codec, the dashboard
       visibility filter, and the workflow-row projection.
   * - ``created_at`` job field
     - ``Job`` (``mufinder.domain.models``) / job entity files
     - Persisted creation timestamp that fixes a job's position in the
       dashboard across reloads (``load_project`` sorts jobs by
       ``(created_at, id)``). Job files written before the field existed are
       backfilled on load by ``ProjectService._legacy_job_created_at``: the
       job's first provenance mention (plus its position within a batch
       record), else its earliest lifecycle timestamp, else the load time so
       it sorts last. Backfilled jobs are left out of the dirty-tracking cache
       so the next save persists the value.
     - The field is current job data, not deprecated. The backfill is the
       shim: keep it until the next saved-project schema version bump, whose
       migration can stamp ``created_at`` on every job file that lacks it.
   * - ``castep/keywords_json_path`` setting pointing outside the
       application-data directory
     - ``mufinder.castep_setup`` / application settings (``QSettings``)
     - Before catalogs were installed as a managed copy, this setting held a
       path the user chose. Such a path is still honoured as the configured
       catalog, so existing installations keep working without re-running
       setup. Generating or importing a catalog repoints the setting at the
       managed copy.
     - Application setting, not project data. Keep indefinitely: honouring an
       arbitrary path costs nothing, and dropping it would silently return
       existing users to the starter catalog.
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


Removed settings
----------------

- ``clusters/<name>/keyword_python_command`` (application setting, removed with
  the shell-loop keyword generator described in :doc:`castep_keyword_catalog`).
  Cluster keyword generation no longer runs Python on the cluster, so the value
  has no meaning. It was never project data; a value left in saved settings is
  ignored and needs no migration.
