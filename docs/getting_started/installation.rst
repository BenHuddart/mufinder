Installation
============

MuFinder is distributed as pre-built binaries for Windows and Apple Silicon
macOS, and can be installed from source on any platform with Python 3.12 or
later. CASTEP and MPI are not included: MuFinder drives a CASTEP installation
you supply, locally or on an HPC cluster.

Binary distributions
--------------------

`GitHub Releases <https://github.com/BenHuddart/mufinder/releases>`_ publish:

- Windows installers built from PyInstaller ``onedir`` bundles and wrapped with
  Inno Setup
- macOS ``.app`` bundles packaged into drag-to-Applications DMGs (**Apple
  Silicon / arm64 only**)

These binaries ship MuFinder itself, but CASTEP and MPI remain external
user-supplied dependencies.

.. note::

   Pre-built binaries are not provided for Intel Macs (x86_64). Install from
   source instead, as described below.

Install from source
-------------------

Clone and install locally:

.. code-block:: bash

   git clone https://github.com/BenHuddart/mufinder.git
   cd mufinder
   python -m pip install .

Install directly from Git:

.. code-block:: bash

   python -m pip install "git+https://github.com/BenHuddart/mufinder.git"

Install with development tooling. The test suite also needs the
``scientific-backends`` extra (``muesr``, ``muLFC``, ``Soprano``), which the
Magnetic Fields, UEP, and relaxation-analysis features use at runtime:

.. code-block:: bash

   uv sync --extra dev --extra scientific-backends
   # or: make install

The structure viewer uses the PyVista/VTK + ``pyvistaqt`` stack.

Running MuFinder
----------------

Run the app:

.. code-block:: bash

   python -m mufinder.main

A source install also provides a ``mufinder`` command that starts the same
application.

Setting up CASTEP
-----------------

Before running calculations, open ``Setup > CASTEP`` to set the local CASTEP
executable and MPI launcher and to install a CASTEP keyword catalog; the steps
are in :doc:`/getting_started/castep_setup`. Cluster profiles are configured
under ``Setup > HPC Clusters`` and described in
:doc:`/getting_started/running_jobs`.
