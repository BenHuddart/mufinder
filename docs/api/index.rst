Python API
==========

This reference documents the MuFinder library packages, for developers and for
anyone scripting a bespoke workflow without the GUI:

- ``mufinder.domain`` holds the dataclasses, enums and durable project records;
- ``mufinder.castep_io`` parses, validates and writes CASTEP input and output;
- ``mufinder.persistence`` stores projects as JSON through its codecs;
- ``mufinder.science`` holds the numerical and crystallographic algorithms;
- ``mufinder.workflow`` expands resolved workflow state into jobs and inputs;
- ``mufinder.services`` manages project state, execution and scheduling.

Dependencies flow inward, as described in :doc:`/architecture`: lower layers
never import higher ones, and nothing outside ``mufinder.app`` imports Qt
widgets. ``mufinder.app``, the PySide6 user interface, is intentionally not
documented here.

:doc:`using_mufinder` walks through four scripts that call these packages
directly.

.. toctree::
   :maxdepth: 1

   using_mufinder

.. autosummary::
   :toctree: generated
   :recursive:

   mufinder.domain
   mufinder.castep_io
   mufinder.persistence
   mufinder.science
   mufinder.workflow
   mufinder.services
