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
   * - ``ConvergenceWidget``
     - ``mufinder.app.convergence_widget``
     - Import alias for ``CutoffConvergenceWidget``, also exported from
       ``mufinder.app``, kept for external imports. No in-repo code or test
       imports it; one test checks that it still resolves. Not persisted.
     - Public import alias. Keep for one minor release (through 0.6.x) or until
       the next major release, whichever comes first; then drop the alias and
       its ``mufinder.app`` export.
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
       ``CutoffConvergenceWidget._is_default_parameter_set``
     - A cutoff row saved with parameters of exactly ``{"task": ...}`` and no
       advanced keywords is read back as "uses the defaults". A restore bug
       wrote such entries for default rows, which made the rows stop matching
       the definition their jobs ran under, so they were submitted again. The
       parameter dialog cannot produce a task-only row, so nothing a user
       configured is reinterpreted.
     - Read-side repair. Keep until a saved-project schema version bump
       includes a migration that drops task-only rows from ``convergence_form``.
   * - Bare site-id source keys and Quantum ``sources`` without ``key``
     - Quantum Corrections and Energy Barriers row keys;
       ``mufinder.services.source_candidates`` (``quantum_source_key``) and the
       Quantum Corrections definition ``configuration["sources"]``
     - A relaxed site's source key is its bare site id for the relaxation the
       site records, which is every source key a project held before one site
       could reach those stages in both charge states; a relaxation in the
       other charge state keys by ``"{site id}:{charge-state prefix}"``. The
       owning charge state is ``MuonSite.properties["source_key_charge_state"]``,
       written when the site is first relaxed; a site saved before the property
       existed takes it from the charge state of its linked relaxation, and the
       property is written at its next relaxation. A stored Quantum Corrections
       source entry without ``key`` is read as keyed by its ``site_id``.
     - Current project data, not deprecated. Row bindings, seed names and
       generated paths are stored under these keys, so the bare-id form may
       not change without a schema migration that rewrites them.
   * - Pinned sites without ``representative_job_id``
     - ``muon_sites_summary_snapshots`` UI state, read by
       ``mufinder.services.muon_sites_summary`` (``pinned_muon_sites``)
     - A Muon Sites summary snapshot pinned before sites recorded their
       cluster's representative relaxation job has no ``representative_job_id``
       key. Such a site takes the job id from the cluster with the same index in
       the pinned workspace while that workspace exists; otherwise it is
       reported as no longer resolving, and re-pinning the analysis writes the
       key. A site whose key is present but ``null`` is never repaired this way.
     - Read-side repair. Keep until a saved-project schema version bump
       includes a migration that adds the key to stored snapshots.
   * - ``relative_energy_eV`` muon-site key
     - ``mufinder.persistence.project_codec`` (``site_from_dict``)
     - ``MuonSite`` stored the source relaxation's absolute final total energy
       under the misleading name ``relative_energy_eV``. The field is now
       ``total_energy_eV`` and new saves write only that key; a project saved
       before the rename is read through the old key, which holds the same
       value. Relative energies are computed for display and never stored on
       the site.
     - Read-side repair. Keep until a saved-project schema version bump
       includes a migration that renames the key in stored sites.
   * - ``castep_transition_state_search`` /
       ``castep_transition_state_search_set`` calc types
     - ``TRANSITION_STATE_SEARCH_CALC_TYPE`` /
       ``TRANSITION_STATE_SEARCH_SET_CALC_TYPE``
       (``mufinder.workflow.energy_barriers``, re-exported from
       ``mufinder.workflow``) / job, result and calculation entity files
     - Calc types of the LST/QST transition-state search, which Energy
       Barriers no longer creates; it expands only NEB jobs. They are read so
       an older project's records still load:
       ``EnergyBarriersController`` owns jobs of the search calc type,
       ``RetrievalFlags.for_calc_type`` retrieves their ``.ts`` output, and the
       Energy Barriers completion rule and stage facts in
       ``mufinder.services.project_summary`` count their results.
       The diffusion-path ``sources`` in a set definition's ``configuration``
       join the generated paths
       (``mufinder.services.source_candidates.energy_barrier_generated_sources``)
       and are removed with them (``EnergyBarriersController.path_definitions``).
     - Project-file shim. The set calc type can go once a saved-project schema
       version bump includes a migration that moves each set's ``sources``
       into the Energy Barriers form state. The search calc type names what
       CASTEP ran, so a stored job or result keeps it; drop its readers only
       when projects holding such records are no longer supported.
   * - ``castep_quantum_singlepoint`` calc type
     - ``QUANTUM_SINGLEPOINT_CALC_TYPE`` (``mufinder.workflow.quantum``) / job
       and result entity files
     - Calc type of the singlepoint job Quantum Corrections ran before each
       phonon job; it now runs one phonon job per row and creates no such
       job. It is read so an older project's records still load and collect:
       ``QuantumCorrectionsController`` owns jobs of the type, contributes an
       empty result for them, shows a row whose singlepoint is still active
       with that job's status and refuses to submit it, and names a finished
       one as the phonon job's directory companion for the shared reset.
     - Project-file shim. The calc type names what CASTEP ran, so a stored
       job or result keeps it; drop its readers only when projects holding
       such records are no longer supported.
   * - ``grid_size`` / ``padding`` arguments of
       ``magnetometric_rectangular_prism_tensor``
     - ``mufinder.science.demagnetization``, also exported from
       ``mufinder.science``
     - The function returns the exact closed-form tensor of a rectangular
       prism, so the two keyword arguments of its sampled predecessor are
       ignored; passing either raises a ``DeprecationWarning``. One test checks
       the warning. The persisted ``fourier_grid`` and ``padding`` fields of a
       ``DemagnetizingSample`` stay current: the Fourier method uses them.
     - Deprecated public arguments. Keep for one minor release (through 0.9.x)
       or until the next major release, whichever comes first; then drop them.
   * - ``generate_random_sites_by_distance``
     - ``mufinder.science.site_generation``, also exported from
       ``mufinder.science``
     - Public helper that filled the cell with random sites at a minimum
       spacing. ``generate_random_sites`` with ``count=None`` does the same, and
       the helper now calls it and raises a ``DeprecationWarning``. No in-repo
       code calls it; one test checks the warning and the result. Not persisted.
     - Public helper, deprecated. Keep for one minor release (through 0.9.x)
       or until the next major release, whichever comes first; then drop it.
   * - ``singlepoint_job_ids_by_label`` /
       ``singlepoint_seed_names_by_label`` definition keys
     - Quantum Corrections ``castep_quantum_phonon_set`` definitions;
       ``QUANTUM_SINGLEPOINT_JOB_MAPPING`` / ``QUANTUM_SINGLEPOINT_SEED_MAPPING``
       (``mufinder.services.job_paths``)
     - ``singlepoint_job_ids_by_label`` binds a row to its legacy singlepoint
       job; new definitions no longer write it. It is read by
       ``QuantumCorrectionsController`` (``legacy_singlepoint_job_for_key``,
       ``definition_and_key_for_job``), by ``job_output_directory`` to find
       the job's directory and by ``JobFileCleanup.delete_job_records``, which
       unbinds a deleted job. ``singlepoint_seed_names_by_label`` is written
       only when such a row's new phonon job is given a fresh seed because the
       shared seed directory already exists; it keeps the singlepoint's
       directory resolving to the seed it ran under.
     - Project-file shim, read only for rows that hold a legacy singlepoint
       job. Keep both keys with the ``castep_quantum_singlepoint`` readers.
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
   * - ``recommended_cutoff_method`` / ``recommended_kpoint_method`` values
       ``delta-threshold``, ``derivative`` and ``asymptotic``
     - ``Project`` / project manifest; read by ``project_codec`` through
       ``normalize_default_method`` (``mufinder.domain.models``)
     - Builds that offered a choice of energy method stored its name. There is
       now one convergence criterion, so a project default is either
       ``converged`` or ``manual``: any stored value other than ``manual``
       (these three included) loads as ``converged``, and new adoptions write
       ``converged``. The manifest keeps the old string until the project is
       next saved. Older builds display ``converged`` verbatim.
     - Keep until the next saved-project schema version bump, whose migration
       can rewrite the values to ``converged``.
   * - ``execution_target`` / ``num_processes`` /
       ``remote_job_id`` / ``remote_directory`` job fields
     - ``Job`` (``mufinder.domain.models``) / job entity files
     - Load-bearing persisted job-contract state, not deprecation shims.
       ``execution_target`` and ``num_processes`` drive resubmission and
       dashboard columns; ``remote_job_id`` and ``remote_directory`` are the
       metadata that lets the main window re-register persisted active
       cluster jobs with the cluster runner after a project is reopened. Orphaned
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
   * - ``failure_reason`` job field
     - ``Job`` (``mufinder.domain.models``) / job entity files
     - Persisted cause of a failed or stopped job (a ``FailureReason`` value
       such as ``walltime`` or ``out_of_memory``), shown beside the dashboard
       status; see ``docs/job_lifecycle.rst`` (Why a job failed). Optional
       with a ``None`` default, and an unknown value loads as ``None``, so
       older projects load unchanged and their failures show the bare status.
     - Current job data, not deprecated. Removing the field or renaming a
       ``FailureReason`` value drops recorded causes from existing projects, so
       it requires a schema-version bump with a migration entry and updates to
       the persistence codec and dashboard rendering.
   * - ``created_at`` job field
     - ``Job`` (``mufinder.domain.models``) / job entity files
     - Persisted creation timestamp that fixes a job's position in the
       dashboard across reloads (``load_project`` sorts jobs by
       ``(created_at, id)``). Job files written before the field existed are
       backfilled on load by
       ``mufinder.persistence.project_codec.jobs_in_creation_order``: the
       job's first provenance mention (plus its position within a batch
       record), else its earliest lifecycle timestamp, else the load time so
       it sorts last. Backfilled jobs are left out of the dirty-tracking cache
       so the next save persists the value.
     - The field is current job data, not deprecated. The backfill is the
       shim: keep it until the next saved-project schema version bump, whose
       migration can stamp ``created_at`` on every job file that lacks it.
   * - ``potential_grid`` key in UEP analysis UI state
     - ``uep_analysis_states`` UI state; read by the UEP analysis open path
       in ``mufinder.app.main_window`` and by
       ``UepAnalysisDialog._restore_initial_state``
     - Older UEP analysis snapshots repeated the full potential grid, which is
       also persisted as the job's ``uep_potential_grid`` result. Snapshots no
       longer write it. Opening the analysis moves a grid found only in UI
       state into that result and strips the key; the wizard still restores
       from the key when no grid result is supplied.
     - Keep until the next saved-project schema version bump, whose migration
       can move the key into a ``uep_potential_grid`` result and drop it.
   * - Top-level ``isosurface_mesh`` key in UEP analysis UI state
     - ``uep_analysis_states`` UI state; read by
       ``UepAnalysisDialog._restore_initial_state``
     - Single-isosurface snapshots stored the mesh at the top level. Snapshots
       now keep every mesh only under ``isosurface_layers``; the key is still
       read when a snapshot has no ``isosurface_layers``.
     - Keep until the next saved-project schema version bump, whose migration
       can wrap the key into a one-entry ``isosurface_layers`` list.
   * - ``default_cut_off_*`` / ``default_kpoint_stress_threshold`` keys in
       project UI state
     - Top-level ``ui_state`` (``state.json``); removed by
       ``mufinder.persistence.migrations.drop_retired_ui_state_keys``, which
       ``ProjectService.load_project`` calls on every load
     - Projects saved by earlier releases may carry
       ``default_cut_off_energy``, ``default_cut_off_energy_method``,
       ``default_cut_off_energy_threshold``, ``default_cut_off_force_threshold``,
       ``default_cut_off_stress_threshold`` and
       ``default_kpoint_stress_threshold``, copies of the adopted convergence
       defaults that the project's ``recommended_*`` fields record. Nothing
       reads or writes them. Loading removes them (``RETIRED_UI_STATE_KEYS``),
       and the next save writes ``state.json`` without them.
     - Keep until the next saved-project schema version bump, whose migration
       can drop the keys; then remove the load-time helper.
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
   * - ``candidate`` muon-site status
     - ``MuonSite.status`` / site entity files; read by
       ``mufinder.persistence.project_codec.site_from_dict`` and
       ``MuonGenerationWidget._display_site_status``
     - Older projects stored freshly generated sites as ``candidate``. Loading
       maps the value to ``generated``, the display shows it as
       ``generated``, and ``mufinder.services.site_generation_service`` still
       treats ``candidate`` as a generated, stageable status
       (``LEGACY_GENERATED_SITE_STATUSES``).
     - Keep until the next saved-project schema version bump, whose migration
       can rewrite ``candidate`` to ``generated`` in every site file.
   * - ``converged_cutoff`` key in convergence-examiner result snapshots
     - Examiner UI state; written and read by
       ``ConvergenceExaminerDialog._serialize_result`` /
       ``_deserialize_result``
     - Cutoff-only snapshots stored the converged value as
       ``converged_cutoff``. Snapshots now write ``converged_value`` and still
       write ``converged_cutoff`` alongside it; loading falls back to
       ``converged_cutoff`` when ``converged_value`` is absent.
     - Keep until the next saved-project schema version bump, whose migration
       can rename the key to ``converged_value``.
   * - ``fractional_position`` / ``cartesian_position`` attributes
     - ``MuonSite`` (``mufinder.domain.models``)
     - Read-write attribute aliases of ``initial_fractional_position`` /
       ``initial_cartesian_position``, the names those fields had before the
       relaxed position was stored beside them. They are properties, not
       dataclass fields: the constructor takes only the new names and the
       site entity file stores only the new keys (the 2 to 3 schema step
       below renames stored keys). No MuFinder code reads them; one test
       checks that they still resolve.
     - Public attribute alias. Keep for one minor release (through 0.6.x);
       then drop both properties.
   * - ``ConstructionSignWidget`` / ``WorkflowDefinition`` /
       ``MuonChargeStateDialog``
     - ``mufinder.app.main_window``
     - Import path kept for names that live in
       ``mufinder.app.project_home_trail`` and
       ``mufinder.app.muon_generation_support``. ``ConstructionSignWidget`` is
       imported only to keep the path; ``MainWindow`` uses the others. Not
       persisted.
     - Public import alias. Keep for one minor release (through 0.6.x); then
       drop the ``ConstructionSignWidget`` import and move test and external
       imports of these names to their defining modules.
   * - ``task_farming_for_job`` / ``retrieve_flags_for_job`` keywords
     - ``JobExecutionController.submit_local_jobs``
     - The two-callback form of ``retrieval_for_job``: one callback returns
       ``(task_farming_enabled, num_farms)`` and the other a mapping of
       ``retrieve_*`` flags. They are read only when ``retrieval_for_job`` is
       ``None`` and are combined into one
       ``mufinder.services.job_artifacts.RetrievalFlags``. The application
       passes ``retrieval_for_job``. Not persisted.
     - Public keyword alias. Keep for one minor release (through 0.6.x); then
       move the remaining test callers to ``retrieval_for_job`` and drop both
       keywords.
   * - ``lattice_frac_to_cart`` / ``lattice_cart_to_frac``
     - ``mufinder.science.site_generation``
     - Public names for the scalar conversions that live in
       ``mufinder.science.lattice``. The ``site_generation`` versions are thin
       wrappers that raise ``SiteGenerationError`` for a malformed or singular
       lattice, which callers in ``mufinder.app``, ``mufinder.services`` and
       ``mufinder.science`` catch. Not persisted.
     - Public import path. Keep until the callers that catch
       ``SiteGenerationError`` around these conversions move to
       ``mufinder.science.lattice`` and ``LatticeError``; then drop both
       wrappers.
   * - ``structure_to_atoms``
     - ``mufinder.science.nuclear_relaxation``
     - Thin wrapper over ``mufinder.science.ase_bridge.structure_to_ase_atoms``
       that keeps the muon-skipping default, sanitised symbols, recorded
       species labels and ``NuclearRelaxationError``. Imported by the nuclear
       relaxation tests. Not persisted.
     - Public helper alias. Keep for one minor release (through 0.6.x); then
       move the tests to ``structure_to_ase_atoms`` and make the wrapper
       private.
   * - ``ClusterConfig``
     - ``mufinder.cluster_setup``
     - Re-export of ``mufinder.domain.cluster.ClusterConfig``, so the cluster
       record carries no Qt dependency into ``mufinder.services``. MuFinder
       code imports it from the domain module. Not persisted: the ``QSettings``
       keys that ``cluster_setup`` reads and writes are unchanged.
     - Public import alias. Keep for one minor release (through 0.6.x); then
       drop the re-export and its ``__all__`` entry.
   * - ``FormattedDensity`` / ``IonicPseudopotential`` /
       ``castep_total_charge``
     - ``mufinder.science.uep``
     - Re-exports of the CASTEP readers' records and charge reader in
       ``mufinder.castep_io.charge_density``. Imported by the UEP science
       tests. Not persisted.
     - Public import alias. Keep for one minor release (through 0.6.x); then
       move the tests to ``mufinder.castep_io.charge_density`` and drop the
       re-exports.
   * - ``parse_formatted_density`` / ``parse_castep_pseudopotentials``
     - ``mufinder.science.uep``
     - Thin wrappers over the ``mufinder.castep_io.charge_density`` readers
       that turn ``ChargeDensityInputError`` into ``UepError``. Callers of
       ``load_uep_source`` (the UEP wizard and job runner) and the UEP science
       tests catch ``UepError``. Not persisted.
     - Public import path. Keep until the callers that catch ``UepError``
       around these readers catch ``ChargeDensityInputError`` instead; then
       drop both wrappers.
   * - ``UepClusterSummary`` / ``UepClusterSimilaritySummary`` /
       ``UepBatchSummary`` / ``UepClusteringConfig`` / ``UepClusteringResult``
       / ``UepPreparedClusteringDataset`` / ``cluster_optimized_positions`` /
       ``relaxed_site_number_map`` / ``prepare_uep_clustering_dataset`` /
       ``cluster_prepared_uep_dataset`` / ``cluster_uep_optimizations``
     - ``mufinder.science.uep``
     - Names that live in ``mufinder.science.uep_clustering``. Because that
       module imports ``mufinder.science.uep``, the module-level
       ``__getattr__`` in ``uep`` resolves them lazily. No in-repo caller
       reaches them through ``uep``; MuFinder code and tests import them from
       ``mufinder.science.uep_clustering``. Not persisted.
     - Public import alias. Keep the ``__getattr__`` hook for one minor
       release (through 0.6.x); then drop it.
   * - ``clusters/<name>/default_queue_name`` without
       ``clusters/<name>/queues``
     - ``mufinder.cluster_setup`` (``load_cluster_configs``)
     - A cluster profile saved before profiles listed queues holds at most one
       queue, ``default_queue_name``, and no ``queues`` key. It loads with an
       empty ``queues`` tuple, so the submission dialog offers that one name
       with no overrides and every resource comes from the profile, as
       before. An ``addqueue`` profile with no queue at all still loads with
       ``redwood``. The setup dialog shows such a profile's queue as one row
       and saves it into ``queues`` the next time the profile is saved.
     - Application setting, not project data. Current behaviour, not
       deprecated: ``default_queue_name`` stays the name of the default queue,
       and a profile with no queues is valid.
   * - ``magnetic_structure`` moments with ``coordinate_system: "crystalaxis"``
     - ``mufinder.science.magnetic_structure``
       (``magnetic_model_from_legacy_structure``)
     - The MCIF importer stores each moment as a Cartesian ``vector`` in μB,
       with ``coordinate_system: "cartesian"`` and the components as written
       in ``source_components`` under ``source_coordinate_system``. A moment
       saved with ``coordinate_system: "crystalaxis"`` holds the file's
       components along the unit lattice directions, so building the magnetic
       model converts it with the host lattice. The per-atom spins such a
       project already stores are not recomputed.
     - Read-side repair. Keep until a saved-project schema version bump
       includes a migration that converts stored crystal-axis moments.
   * - ``magnetic_field`` records without ``site_source``
     - ``mufinder.services.magnetic_field_results``
       (``field_result_site_source``, ``attach_magnetic_field_results``)
     - A dipolar-field record names the source of the muon site it was
       computed at in ``data["site_source"]``, so fields at an unrelaxed UEP
       site are shown as estimates. A record saved before the key existed
       takes the ``source`` of its form's ``selected_muon_site``, which is the
       site the results were computed at because choosing another site clears
       them. A record whose form has no selected site shows no estimate
       marker. The next calculation writes the key.
     - Read-side repair. Keep until a saved-project schema version bump
       includes a migration that copies the selected site's source into each
       stored ``magnetic_field`` record.
   * - ``magnetic_field`` distribution entries without ``terms``
     - ``mufinder.science.magnetic_fields``
       (``MagneticFieldDistributionResult.from_json``,
       ``regenerate_phase_samples``), read by the Magnetic Fields results
       table
     - A distribution entry stores ``terms``, ``total_rms_T``, ``applied_T``
       and ``demag_T``, and its ``calculation_parameters`` hold
       ``phase_sampling_seed`` and ``component_phase_coefficients``. An entry
       saved before these keys existed loads with empty ``terms`` and ``None``
       for the rest; selecting its row shows its convergence checks and asks
       for a recalculation instead of a breakdown, and
       ``regenerate_phase_samples`` refuses it. The next calculation writes
       the keys. Nothing is recomputed on load.
     - Read-side repair. Current behaviour for any entry without the keys;
       keep until a saved-project schema version bump includes a migration
       that drops such entries or the page stops reading them.
   * - ``magnetic_field`` commensurate entries without
       ``calculation_parameters``, and Magnetic Fields form state without
       ``contact_nnn`` or ``contact_rcont``
     - ``mufinder.science.magnetic_fields`` (``MagneticFieldResult.from_json``)
       and ``mufinder.app.magnetic_field_widget`` (the form-state restore)
     - A commensurate entry records the ``contact_coupling``, ``nnn`` and
       ``rcont`` its contact term used in ``calculation_parameters``, and the
       form state saves the contact shell as ``contact_nnn`` and
       ``contact_rcont``. An entry saved before the key existed loads with
       empty ``calculation_parameters``; its fields were computed with
       ``muesr``'s shell of 2 ions within 10 Å. A form without the keys, or
       with values outside the dialog's ranges, takes those defaults. The
       next calculation writes the keys. ``contact_shell_warnings`` is
       written only when a site's shell splits a tie, so an entry without it,
       including one saved before it existed, shows no tie warning.
     - Read-side repair. Current behaviour for any entry or form without the
       keys; keep until a saved-project schema version bump includes a
       migration that writes the default shell into stored entries and forms.
   * - ``w1_entanglement_witness`` result payloads without ``w1_definition``
     - ``mufinder.services.quantum_results`` (``w1_from_result_data``), read
       by the Quantum Corrections results table
     - A quantum-phonon result stores the ``w1`` witness with
       ``w1_definition: "participation_minus_3"``, the ``P - 3`` definition
       of Gomilšek et al. A payload without ``w1_definition`` holds the value
       as ``-sqrt(3 - P)``, so the witness shown is recomputed as the summed
       ``participation`` of its ``muon_zpe_modes`` minus 3. When the payload
       has no usable ``muon_zpe_modes`` the stored value is shown as it is.
       The stored value is never rewritten.
     - Read-side repair. Keep until a saved-project schema version bump
       includes a migration that rewrites the stored witness of each
       quantum-phonon result, or indefinitely.
   * - Relaxed structures without ``atomic_spins`` in a project stored below
       schema 6
     - ``mufinder.services.relaxed_structure_spins``
       (``backfill_relaxed_structure_spins``), called by
       ``ProjectService.load_project``
     - The job-file half of the 5 to 6 step (see the table below), which a
       payload migration cannot do because it never sees the job folders.
     - Keep while schema 5 projects may still be opened, or indefinitely.

Saved-project schema migrations
-------------------------------

``project.json`` records the ``schema_version`` a project was saved at, and
every save writes ``PROJECT_SCHEMA_VERSION`` (``mufinder.domain.models``).
``ProjectService.load_project`` passes the manifest and the raw entity payloads
to ``mufinder.persistence.migrations.migrate_project_payloads`` before it
builds any domain object. That function applies one step per version, from the
stored version up to the current one, and sets the manifest's version to the
result. The entity files a step changed are left out of the dirty-tracking
cache, so the next save rewrites them together with the new manifest version.
Provenance is history: no step rewrites it, and readers of provenance accept
every name a record was ever written with. A project saved at the current
version or a newer one is loaded as it is. A manifest without
``schema_version`` is read as schema 1.

.. list-table::
   :header-rows: 1
   :widths: 12 44 44

   * - Step
     - Payloads changed
     - Readers that rely on it
   * - 1 to 2
     - Convergence-cutoff and convergence-k-point definitions
       (``calculations`` entities with ``workflow_stage``
       ``convergence_cutoff`` or ``convergence_kpoint``). Schema 1 stored the
       sweep in ``parameters.configuration`` under both ``sweep_values`` and
       the legacy ``cutoff_energies`` key, and read a cutoff sweep from
       ``cutoff_energies`` first. The step moves the values that reader used
       into ``sweep_values`` and removes ``cutoff_energies``.
     - ``deserialize_convergence_config`` and
       ``remove_cutoff_from_definition`` read only ``sweep_values``;
       ``ConvergenceConfig`` has no ``cutoff_energies`` field. The journal
       counts a ``configure_convergence_cutoff`` record's cutoffs from
       ``sweep_values``, else from ``cutoff_energies``. The widget form state
       (``ui_state["convergence_form"]``) is not part of the step and keeps its
       own ``cutoff_energies`` key; see :doc:`/architecture`.
   * - 2 to 3
     - Muon sites (``sites`` entities). Schema 2 stored a site's generated
       position as ``fractional_position`` / ``cartesian_position`` and kept
       it unchanged after relaxation. The step renames the keys to
       ``initial_fractional_position`` / ``initial_cartesian_position``. A
       site whose ``relaxation_run_id`` names a run with a result recording
       ``muon_final_fractional_position_host`` gains that triplet as
       ``relaxed_fractional_position``; other sites have none.
     - ``site_from_dict`` reads only the new keys, and
       ``MuonSite.current_fractional_position`` reads
       ``relaxed_fractional_position``. Muon-relaxation completion sets the
       relaxed position and removing the run's records clears it, so after
       the step every relaxed site carries it when its result does.
   * - 3 to 4
     - Results (``results`` entities) and the Magnetic Fields form state in
       ``ui_state``: ``magnetic_field_form`` for the page and the ``form`` of
       each entry in ``magnetic_field_workspaces``. Schema 3 kept the latest
       field, dipolar-tensor and nuclear-relaxation results inline in those
       forms under ``magnetic_field_results``, ``dipolar_tensor_results`` and
       ``nuclear_relaxation_results``. The step removes the three keys from
       every form and turns each non-empty value into a result entity of type
       ``magnetic_field``, ``magnetic_dipolar_tensor`` or
       ``nuclear_relaxation`` with no ``calculation_run_id``, whose
       ``data["owner"]`` is ``magnetic_fields`` for the page or the workspace
       id. A workspace without an id has no owner and keeps its form.
     - ``mufinder.services.magnetic_field_results`` reads the page's results
       only from result records and never from the persisted form; the
       project survey reads a workspace's field results from its record.
       ``migrate_project_payloads`` receives ``ui_state`` and reports it
       changed, so the next save rewrites ``state.json``.
   * - 4 to 5
     - Dipolar-tensor results (``results`` entities of type
       ``magnetic_dipolar_tensor``). Schema 4 stored ``muesr``'s tensor under
       ``dipolar_tensor_T_per_muB`` without converting it from ``Å⁻³``. The
       step multiplies every numeric ``3 x 3`` tensor in ``data["results"]``
       by :math:`\mu_0 \mu_B / (4\pi\,\text{Å}^3) = 0.9274` T/μB and leaves
       any other entry as it is. A schema 3 project reaches this step after
       its inline tensors have become records, so they are converted too.
     - The dipolar tensor dialog reads ``dipolar_tensor_T_per_muB`` as T/μB,
       and ``calculate_dipolar_tensors_for_atoms`` applies the same factor to
       new tensors. Saved commensurate field results are not rewritten.
   * - 5 to 6
     - Relaxed structures (``structures`` entities). Schema 5 did not store a
       relaxation's ``SPIN=`` values on its relaxed structure. They live in the
       job folders, which payloads cannot reach, so the payload step changes
       nothing; ``ProjectService.load_project`` instead calls
       ``backfill_relaxed_structure_spins`` for a project stored below schema
       6. Each structure a pristine or muon relaxation job derived (through
       ``source_job_id`` or its run's ``derived_structure_ids``) and that has
       no ``atomic_spins`` key gains the nonzero ``SPIN=`` values of that job's
       ``-out.cell``, else its input ``.cell``, when the file has the
       structure's atom count. Those structures are rewritten by the next
       save. A structure whose job files are gone, or whose run had no spins,
       keeps no key and is not checked again once the project is saved at
       schema 6.
     - ``workflow.bridge.structure_to_spec`` prefers ``atomic_spins``, so the
       ``.cell`` export and every job started from a relaxed structure write
       the run's spins. Relaxation completion stores them for new runs.

A schema bump adds ``PROJECT_SCHEMA_VERSION + 1``, one entry in
``_MIGRATIONS`` keyed by the version it upgrades from, a row in the table
above, and a fixture project saved at the old version under
``tests/fixtures/projects/`` that loads, round-trips and reloads.

Rules for new shims
-------------------

- Every new compatibility alias should have a row in this ledger.
- Project-file shims must state the schema version or migration event that can
  remove them.
- Public import aliases should be removed only in major releases or after a
  documented replacement has existed for at least one minor release.
- Tests that exercise legacy data should name the legacy field explicitly, so
  future cleanup work can find the remaining callers with ``rg``.


Removed aliases and settings
----------------------------

Each public import alias below was removed after its documented replacement had
shipped for at least one minor release. None was persisted, so projects need no
migration; code that imported one switches to the replacement named.

- ``ConvergenceWorkflow`` (``mufinder.workflow.convergence``, re-exported from
  ``mufinder.workflow``): call ``expand_convergence_sweep_jobs``.
- ``MuonSiteWorkflow`` (``mufinder.workflow.muon_site``, re-exported from
  ``mufinder.workflow``): call ``expand_muon_site_jobs``.
- ``build_convergence_workflow`` (``mufinder.app.convergence_adapter``,
  re-exported from ``mufinder.app``): call ``expand_convergence_jobs``.
- ``RelaxationExpansion`` (``mufinder.app.relaxation_adapter``) and
  ``UepExpansion`` (``mufinder.app.uep_adapter``): use ``AdapterExpansion``
  from ``mufinder.app.adapter_common``.
- ``AdapterExpansion.kpoint_messages``: read ``AdapterExpansion.messages``.
- ``CutoffRunState`` (``mufinder.app.convergence_types`` and
  ``mufinder.app.main_window``): use ``SweepPointRunState`` from
  ``mufinder.app.convergence_types``.
- ``recommended_cutoff`` (``mufinder.science.convergence_analysis``,
  re-exported from ``mufinder.science``): call ``recommended_value``.
- ``ConvergenceResult.converged_cutoff``: read
  ``ConvergenceResult.converged_value``. The persisted ``converged_cutoff``
  key in convergence-examiner snapshots is a separate project-file shim and
  stays in the ledger above.
- ``workflow_tick_icon`` (``mufinder.app.icons``): call
  ``workflow_stage_icon("completed")`` or ``workflow_stage_icon("incomplete")``.
- ``clusters/<name>/keyword_python_command`` (application setting, removed with
  the shell-loop keyword generator described in :doc:`/castep_keyword_catalog`).
  Cluster keyword generation no longer runs Python on the cluster, so the value
  has no meaning. It was never project data; a value left in saved settings is
  ignored and needs no migration.

The public helper below was removed once its ledger expiry (through 0.7.x)
had passed. It was deprecated with a ``DeprecationWarning`` in 0.8.0, and its
replacement has shipped since 0.6.0. It was not persisted, so projects need no
migration; the ``castep_quantum_singlepoint`` jobs and results of older
projects still load through the calc-type shim in the ledger above.

- ``expand_quantum_singlepoint_jobs`` (``mufinder.workflow.quantum``):
  Quantum Corrections runs one phonon job per row, which converges its own
  ground state; call ``expand_quantum_phonon_jobs``.

The convergence functions below were removed without a deprecation period,
together with their ``mufinder.science`` re-exports, when the examiner moved to
one criterion relative to the finest sweep point. They implemented the
retired criteria, so keeping them would keep those criteria in the API; no
project data depends on them. Code that called one switches to the
replacement named.

- ``delta_threshold_energy``, ``derivative_energy``,
  ``asymptotic_extrapolation_energy``, ``delta_threshold_force`` and
  ``delta_threshold_stress`` (``mufinder.science.convergence_analysis``): call
  ``analyze_convergence`` (cutoff sweeps, with ``max_stresses`` and
  ``stress_threshold`` for stress) or ``analyze_kpoint_convergence``, which
  return a ``ConvergenceAnalysis``. ``check_sweep_length`` runs the power-law
  sweep-length check on its own.
- ``apply_kpoint_label_to_stress_result``: pass ``max_stresses``,
  ``stress_threshold`` and ``point_labels`` to ``analyze_kpoint_convergence``.
- The ``method`` argument of ``analyze_convergence``, and the ``method``,
  ``fallback_suggestion`` and ``deltas`` fields of ``ConvergenceResult``:
  results now carry ``quantity``, ``summary``, ``explanation``,
  ``reference_value`` and ``deviations`` (each point's distance from the
  reference); ``details`` is a property joining them.

The examiner's saved state (``ui_state``) no longer writes ``method``,
``selected_method_for_recommendation`` or ``results_text`` (the results are
stored as ``results_rows``), and overview snapshots no longer write
``method``. An examiner state that still has ``method`` was produced by the
retired criteria: the examiner keeps its thresholds and plot data, drops its
results and recommendation, and says so in an ``Earlier analysis`` row. Old overview snapshots render as before; their
``method`` is ignored. No shim or migration is needed for UI state.

The setting below was removed without a deprecation period, because nothing
read it and it has no replacement.

- ``job_name_prefix``: the cluster submission dialog's ``Job name prefix``
  field, the ``ClusterSubmissionConfig.job_name_prefix`` field (``mufinder.app``,
  not in the API reference), the key of the ``last_cluster_submission_config``
  entry in project ``ui_state`` that saved it, and the helper
  ``default_cluster_job_name_prefix`` (``mufinder.workflow.naming``, never
  re-exported from ``mufinder.workflow``). No submission read the value: the
  scheduler names each job after its CASTEP seed. A project that still carries
  the key loads unchanged; the key is ignored and dropped the next time a
  cluster submission saves its settings, so no shim or schema migration is
  needed. Code that built ``ClusterSubmissionConfig`` drops the argument; the
  fields after ``username`` move up one place, so pass them by keyword.

The private forwarders below were removed without a deprecation period. They
were one-line ``MainWindow`` methods, not part of any public import path, and
only MuFinder's own code and tests called them.

- ``_serialize_uep_config`` / ``_deserialize_uep_config`` /
  ``_serialize_pristine_relaxation_config`` /
  ``_deserialize_pristine_relaxation_config`` /
  ``_serialize_muon_relaxation_config`` /
  ``_deserialize_muon_relaxation_config`` / ``_serialize_convergence_config`` /
  ``_deserialize_convergence_config``: call the codec of the same name without
  the leading underscore in ``mufinder.app.workflow_config_codecs``. The stored
  configuration shape is unchanged.

The names below were removed without a deprecation period. Neither is in the
API reference and no project data depends on them: nothing constructed the
dialog, and the Supercell Editor was the only caller of the private forwarder.

- ``SupercellPreviewDialog`` (``mufinder.app.supercell_dialog``): open
  ``SupercellEditorDialog`` from the same module, which previews, names and
  saves supercells.
- ``ProjectService._supercell_asset``: call ``supercell_asset`` from
  ``mufinder.services.supercell_assets``.

The keywords below were removed without a deprecation period. They belonged to
an unpersisted record that is not in the API reference and that only MuFinder
constructs, so no project data or packaged build depends on them. Code that
passed one switches to the replacement named.

- ``retrieve_den_fmt`` / ``retrieve_out_cell`` / ``retrieve_phonon`` /
  ``retrieve_ts`` / ``task_farming_enabled`` / ``num_farms`` keywords of
  ``ClusterJobSubmission`` (``mufinder.services.job_execution_controller``,
  last accepted in 0.5.0): pass ``retrieval=RetrievalFlags(...)`` with the
  same keyword names, or ``retrieval=RetrievalFlags.for_calc_type(...)``, from
  ``mufinder.services.job_artifacts``.

The module below was removed without a deprecation period, together with the
analysis window tab that was its only caller. It was in the API reference, but
no workflow fed it and nothing it returned was persisted, so projects need no
migration and load unchanged. It has no replacement: each muon mode's harmonic
zero-point energy and width are in the Quantum Corrections results, from
``mufinder.science.quantum_phonons``.

- ``mufinder.science.quantum_adiabatic``, with ``DirectionalPotential``,
  ``SingleParticleResult``, ``solve_single_particle_ground_state`` and
  ``effective_mass_from_relaxed_scan``.

The module-level names below live in another module and keep no import alias
at their former path. That module is not in the API reference and no MuFinder
code imported these names from it. Code that did imports them from the module
named.

- ``REMOTE_STALE_OUTPUT_SLACK_SECONDS``
  (``mufinder.services.cluster_runner``): import it from
  ``mufinder.services.cluster_retrieval``.
