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

New users should start with :doc:`/getting_started/installation`, follow the
:doc:`/getting_started/first_project` tutorial from an empty project to a
relaxed muon site, then read :doc:`/getting_started/running_jobs` and the
:doc:`/user_guide/overview` of every workflow stage. The user guide follows
the order of the stages in the application, and the worked example follows
the positive muon and muonium in CoF₂ through every stage, the project the
screenshots come from. The reference section covers the
CASTEP input MuFinder writes, the Python API documents the library packages
for scripting, and the developer guide describes internal contracts and
quality standards.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
   getting_started/castep_setup
   getting_started/first_project
   getting_started/running_jobs

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/overview
   user_guide/project_home
   user_guide/host_structure
   user_guide/supercells
   user_guide/convergence_testing
   user_guide/comparison_table
   user_guide/uep_workflow
   user_guide/site_generation
   user_guide/structural_relaxation_analysis
   user_guide/legacy_import
   user_guide/magnetic_fields
   user_guide/quantum_corrections
   user_guide/quantum_corrections_theory
   user_guide/energy_barriers
   user_guide/journal

.. toctree::
   :maxdepth: 2
   :caption: Worked examples

   user_guide/examples/cof2
   user_guide/examples/fe
   user_guide/examples/mnsi
   user_guide/examples/alpha_n2
   user_guide/examples/gdru2si2

.. toctree::
   :maxdepth: 2
   :caption: Reference

   castep_keyword_catalog
   castep/cell_syntax
   castep/param_syntax
   castep/kpoint_spacing_convention

.. toctree::
   :maxdepth: 2
   :caption: Python API

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Developer guide

   agent_harness
   architecture
   quality_and_automation
   compatibility_policy
   project_specification
   job_lifecycle
   legacy_import
   branding
