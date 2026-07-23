=====================================
Installing PIPS-IPM++ (block solving)
=====================================

.. warning::

   This integration is **experimental** and not tested in CI. It requires an
   external C++/MPI build of PIPS-IPM++ and is aimed at contributors and users
   with genuinely block-separable models. The Python export/round-trip is
   stable; the C++ callback driver is under active development.

`PIPS-IPM++ <https://gitlab.com/pips-ipmpp/pips-ipmpp>`_ is a distributed
interior-point solver (MPI) for doubly-bordered block-diagonal ("arrowhead")
linear programs. linopy can export a model with block structure into the format
PIPS consumes and read the solution back.

The picture is two-part: **linopy exports** the block files in-process, an
**external PIPS build solves** them via a small callback driver, and linopy
reads the result. PIPS is not a pip-installable package — it must be compiled
against MPI and a linear solver. The instructions below use the fully-open
`MUMPS <https://mumps-solver.org/>`_ backend, which needs no proprietary
solver.

Using the ``pips`` solver
=========================

Once PIPS-IPM++ is installed and the callback driver is built (see the
:ref:`installation steps <pips-install>` below), solving is a one-liner. Point
linopy at the driver binary through two environment variables and call
``Model.solve`` with ``solver_name="pips"``:

.. code-block:: bash

    export PIPS_BINARY=/path/to/build-driver/pips_driver
    export PIPS_MPI_RANKS=2      # optional, default 1; must be <= number of blocks

.. code-block:: python

    m.blocks = ...              # assign the block structure (see "When to use it")
    m.solve(solver_name="pips")

    m.objective.value           # optimal objective
    m.solution                  # primal values, mapped back onto the variables

The solver exports the model, runs PIPS under ``mpirun`` and reads the primal,
duals and objective back onto ``m`` — the same interface as every other linopy
solver. It is strictly opt-in: without ``PIPS_BINARY`` set, ``pips`` is absent
from ``linopy.available_solvers`` and is never auto-selected, so ordinary
installs are unaffected.

When to use it
==============

PIPS only pays off for models that are genuinely **block-separable**: a large
set of local variables/constraints per block, coupled by a comparatively small
set of global (first-stage) variables and linking constraints. Assign the block
structure via ``m.blocks`` (a ``xarray.DataArray`` of block ids over one
dimension); everything independent of that dimension becomes global (block 0).
A model where everything ends up linking defeats the purpose.

.. _pips-install:

System dependencies
===================

Verified on **Ubuntu 24.04** (cmake 3.28, g++ 13). Install the toolchain and
the MUMPS backend:

.. code-block:: bash

    sudo apt-get install -y gfortran libopenmpi-dev openmpi-bin \
        libmumps-dev libmetis-dev libscalapack-openmpi-dev \
        libblas-dev liblapack-dev zlib1g-dev

Notes:

- Mainline PIPS needs **no Boost**. OpenMP ships with g++ (no package).
- Stick to **one MPI implementation** (OpenMPI here) — mixing OpenMPI and MPICH
  breaks the build.
- MUMPS is the default open backend. Panua-PARDISO (≥7.2) or HSL/MA57 are
  optional performance backends and are user-supplied at build time; they are
  not required to solve.

Building PIPS-IPM++ and the driver
==================================

Use the **mainline GitLab repository**, not the stale GitHub mirrors (those lack
the MUMPS backend):

.. code-block:: bash

    git clone --depth 1 https://gitlab.com/pips-ipmpp/pips-ipmpp.git

The mainline build tree unconditionally adds the GAMS ``gmspips`` driver, which
needs proprietary GDX sources that are not vendored and are irrelevant to the
callback path. Disable that one subdirectory before configuring:

.. code-block:: bash

    sed -i 's|^\([[:space:]]*\)add_subdirectory(gams/gmspips)|\1# &|' \
        pips-ipmpp/PIPS-IPM/Drivers/CMakeLists.txt

Then configure, build and install (MUMPS is auto-detected; ``WITH_MAKETEST=OFF``
avoids pulling GoogleTest). Mainline exports a clean ``pips-ipmpp`` CMake package
on install, so the callback driver can link against it with
``find_package(pips-ipmpp CONFIG)``:

.. code-block:: bash

    cmake -S pips-ipmpp -B pips-ipmpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PWD/pips-install" \
        -DWITH_MAKETEST=OFF
    cmake --build pips-ipmpp/build -j"$(nproc)"
    cmake --install pips-ipmpp/build

The callback driver (``pips_driver.cpp``) and a scripted end-to-end build
(``build.sh``, which performs all of the above) live under
``dev-scripts/pips/`` in the linopy repository.

Exporting and solving a model
=============================

Export in-process, then hand the directory to the driver:

.. code-block:: python

    import linopy, xarray as xr, numpy as np, pandas as pd

    m = linopy.Model()
    time = pd.Index(range(12), name="time")
    m.blocks = xr.DataArray(np.repeat([1, 2], 6), [time])   # 2 local blocks
    x = m.add_variables(lower=0, coords=[time], name="x")
    g = m.add_variables(lower=0, name="g")                  # global (block 0)
    m.add_constraints(x <= g, name="cap")
    m.add_objective(x.sum() + g)

    m.to_pips_files("export-dir")                           # writes the block files

.. code-block:: bash

    # N = number of local blocks (here 2); use at most N MPI ranks
    mpirun -np 2 ./build-driver/pips_driver export-dir

.. important::

   PIPS caps the number of MPI ranks at the **number of local blocks** ``N`` (it
   aborts with *"too many MPI processes"* above that). Use ``-np <= N``; PIPS
   assigns several blocks per rank when there are fewer ranks than blocks.

The driver writes the primal, objective and status back into ``export-dir``.
Read them with ``linopy.io.read_pips_solution``.

Validating the export without PIPS
==================================

You do not need a PIPS build to check the exporter. ``linopy.io.read_pips_files``
reconstructs a numerically equivalent, flat model straight from the export
directory; solving it with any ordinary solver reproduces the original optimum:

.. code-block:: python

    from linopy.io import read_pips_files

    m.solve(solver_name="highs")
    m2 = read_pips_files("export-dir")
    m2.solve(solver_name="highs")
    assert abs(m2.objective.value - m.objective.value) < 1e-6

This round-trip runs in linopy's test suite and validates the full arrowhead
carve-up (block partition, linking rows, bound encoding) with no external
dependency.

Limitations
===========

- **LP only** (no MIP/QP through this path yet).
- Requires a manual external build; no conda/pip package yet (a MUMPS-based
  conda-forge recipe is feasible and planned).
- Performance depends on genuine block separability and the linear-solver
  backend (MUMPS works; PARDISO/MA57 are faster, user-supplied).

References
==========

- `PIPS-IPM++ (mainline) <https://gitlab.com/pips-ipmpp/pips-ipmpp>`_
- `Block-structure annotation guide <https://gitlab.com/beam-me/bpg>`_
- `MUMPS <https://mumps-solver.org/>`_
