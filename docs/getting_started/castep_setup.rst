Setting up CASTEP
=================

MuFinder drives a CASTEP installation you supply; it does not include CASTEP
or MPI. ``Setup > CASTEP`` is where you tell MuFinder about it. The dialog has
two jobs: installing the *keyword catalog* that tells MuFinder which keywords
your CASTEP version understands, which every installation needs, and, only if
you run calculations on this computer, recording the local CASTEP executable
and MPI launcher. Cluster runs are configured separately under
``Setup > HPC Clusters`` (see :doc:`/getting_started/running_jobs`).

Before you start
----------------

- **A CASTEP installation.** Either a CASTEP executable on this computer, or an
  HPC cluster profile that already has CASTEP set up, or a
  ``castep_keywords.json`` file you already have.
- **For local runs, an MPI launcher.** CASTEP binaries built with MPI (such as
  ``castep.mpi``) need ``mpirun`` to start properly. See `Local executable and
  MPI launcher`_.
- **For cluster generation, a working cluster profile.** Set it up first in
  ``Setup > HPC Clusters``; MuFinder may ask for your cluster credentials when
  you generate.

Installing a keyword catalog
----------------------------

.. figure:: ../_generated/screenshots/castep_setup_dialog.webp
   :alt: The Setup > CASTEP dialog with no keyword catalog loaded, offering to
         generate one or use an existing file.

   ``Setup > CASTEP`` before a keyword catalog is installed.

The top of the dialog, ``Keyword catalog``, says which catalog is in use (or
that none is loaded) and why a configured one was skipped. There are two ways
to install one. Both end the same way: MuFinder checks the file, keeps its own
copy, and uses it straight away, so there is no location to choose and no
``OK`` to click for the catalog to take effect.

Generating the catalog
~~~~~~~~~~~~~~~~~~~~~~

1. **Choose the target.** In ``Generate from``, pick ``This computer`` or a
   cluster. A cluster that is set up is preselected when no local CASTEP
   executable is configured. The line under the controls says which settings
   the target will use.
2. **For a cluster, choose how to run it.** The ``using`` selector offers
   ``Scheduler job (recommended)`` and ``Login node``. The scheduler job is the
   safe choice where running CASTEP on a login node is forbidden or
   discouraged; the login node is quicker where it is permitted. Your choice is
   remembered for that cluster.
3. **Start it.** Click ``Generate Keyword Catalog``. A progress bar and status
   line follow the run, and ``Cancel`` stops it. A local run takes a minute or
   two; a scheduler job takes a few minutes plus any time in the queue; a
   login-node run takes a minute or two.
4. **Check the result.** When generation finishes the ``Keyword catalog``
   status updates with the CASTEP version and the number of keywords found.
   A failed run says so and offers ``Details`` with the full log.

A local run needs the CASTEP executable set below (see `Local executable and
MPI launcher`_), and a cluster run uses that cluster's own settings from
``Setup > HPC Clusters`` rather than the local paths. Generation always works
in a scratch folder first, so a failed or partial run never replaces a working
catalog. Close the dialog only after generation finishes or is cancelled.

Importing a file
~~~~~~~~~~~~~~~~

Click ``Use Existing File…`` and choose a ``castep_keywords.json`` you already
have, for example one generated on another machine. MuFinder checks that it is
a usable catalog; if not, it says why and leaves the catalog in use untouched.

The installed copy does not track later changes to the original file. After a
CASTEP upgrade, generate or import again.

Local executable and MPI launcher
---------------------------------

The ``CASTEP on this computer (optional)`` group holds two paths. They are
used only for calculations run on this computer and for generating the
catalog from ``This computer``; cluster jobs and cluster generation never use
them.

- ``CASTEP executable``: type the path or click ``Browse…``.
- ``MPI launcher (mpirun)``: type the path, click ``Browse…``, or click
  ``Auto-detect`` to search for ``mpirun`` on this computer.

The launcher matters for two reasons. MPI-built binaries such as
``castep.mpi`` hang when asked for their keyword help directly, so local
catalog generation wraps CASTEP in ``mpirun -np 1`` when a launcher is set;
without it, progress stalls at 0%. And multi-core local runs need a configured
or discovered ``mpirun``: without one, cores per job are limited to 1.

Click ``OK`` to save the two paths. They are checked when you generate with
``Generate Keyword Catalog``: if the ``mpirun`` path does not point to an
executable file, MuFinder says so.

What the status bar shows
-------------------------

A permanent indicator in the status bar shows the keyword catalog in use:
``CASTEP keywords: <version>`` for an installed catalog, or ``⚠ CASTEP
keywords: limited built-in set`` when none is loaded. It also flags a
configured catalog that was skipped. Click it to open ``Setup > CASTEP``;
hover it for the catalog's path and counts.

Without a catalog
-----------------

Running without a catalog is allowed, never blocked, but MuFinder makes it
visible. Without a catalog:

- Parameter dialogs offer only a small built-in subset of CASTEP parameters,
  and say so.
- Results can fail to load when CASTEP writes a keyword outside that subset.
  For example, CASTEP writes ``QUANTISATION_AXIS`` into ``-out.cell`` after a
  Hubbard U calculation, which makes the relaxed structure unreadable in the
  workflows that parse an output cell (UEP, pristine relaxation, muon
  relaxation). With a catalog loaded the same file loads.
- Recovering parameters from legacy relaxations is less complete (see
  :doc:`/user_guide/legacy_import`).
- At startup a prompt, ``CASTEP keyword catalog needed``, offers ``Open CASTEP
  Setup…`` or ``Not Now``; tick ``Don't show this again`` to dismiss it
  permanently.
- Before a local or cluster submission, a ``No CASTEP keyword catalog`` warning
  offers ``Open CASTEP Setup…``, ``Submit Anyway`` or ``Cancel``. ``Submit
  Anyway`` silences the warning for the rest of the session.

Limitations
-----------

- A catalog is generated for one CASTEP version; generate or import again
  after upgrading CASTEP.
- Generation needs CASTEP to be runnable: a local executable (with a launcher
  for MPI builds), or a cluster whose scheduler or login node can run it.

For developers
--------------

The order in which MuFinder looks for a catalog (including the
``MUFINDER_CASTEP_KEYWORDS_JSON`` environment variable, which overrides a
catalog installed here), the validation rules, and how cluster generation works
are described in :doc:`/castep_keyword_catalog`.
