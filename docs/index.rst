MuFinder Documentation
======================

.. image:: ../src/mufinder/assets/mufinder_logo.png
   :alt: MuFinder logo
   :width: 180
   :align: center

MuFinder is a PySide6 desktop application for muon stopping-site workflows
with CASTEP. A project starts from a host crystal structure and works through
the stages listed in the project navigator: supercells, convergence testing,
pristine relaxation, UEP screening, muon-site generation, relaxation and
analysis, magnetic fields, quantum corrections, and energy barriers. Jobs run
locally or on an HPC cluster and are tracked in the job dashboard.

New users should start with :doc:`implemented_functionality` for an overview of
every workflow stage, then set up a CASTEP keyword catalog
(:doc:`castep_keyword_catalog`) before running calculations. The workflow
guides that follow are listed in the order the stages appear in the
application. The reference and developer sections describe internal contracts
and the quality standards that must be maintained as development expands.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   implemented_functionality
   castep_keyword_catalog

.. toctree::
   :maxdepth: 2
   :caption: User guide

   supercells
   convergence_testing
   comparison_table
   uep_workflow
   site_generation
   structural_relaxation_analysis
   legacy_import
   magnetic_fields
   quantum_corrections
   energy_barriers
   journal
   job_lifecycle

.. toctree::
   :maxdepth: 2
   :caption: Reference

   project_specification
   compatibility_policy
   hpc_security_audit
   api

.. toctree::
   :maxdepth: 2
   :caption: Developer guide

   agent_harness
   architecture
   quality_and_automation
