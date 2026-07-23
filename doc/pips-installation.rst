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

Diagnosing block quality
========================

Before committing to an HPC run, check *whether* a K-way split will actually
scale on PIPS. ``linopy.pips.assign_blocks`` builds the ``m.blocks`` assignment
for you by splitting one dimension into contiguous blocks, and
``linopy.pips.diagnose`` reports the resulting arrowhead structure — a cheap,
pure-Python analysis that needs no PIPS build:

.. code-block:: python

    import linopy.pips

    linopy.pips.assign_blocks(m, "time", 50)  # 50 contiguous blocks over "time"
    report = linopy.pips.diagnose(m)
    print(report)

.. code-block:: text

    BlockReport: 50 blocks | 412 340 vars | 388 900 cons | 2 140 552 nnz
      columns    global=1 240   per-block min/med/max = 8 180 / 8 220 / 8 260
      block nnz  min/med/max = 41 002 / 42 780 / 44 190   (max/med ratio 1.03)
      rows       local=386 400  global=12  linking=2 488  (adjacent=2 450  border=38)
      border     nnz=214 300 / 2 140 552 = 10.0%
      parallel   max_ranks=50  target_cores=200 -> ranks=50 threads=4
      warnings   (none)

``assign_blocks(m, dim, n_blocks)`` is also available as a bound method,
``m.assign_blocks("time", 50)``. It returns the model and only supports the
default ``boundary="contiguous"`` split for now.

How to read the report:

- **border_fraction** is the share of matrix nonzeros that sit in linking rows
  or global columns — the part PIPS handles through the root Schur complement.
  It must stay small: above ~15% the root work dominates and parallel speedup
  collapses, so reduce ``K`` or reformulate.
- **balance** (the block-nnz ``max/median`` ratio) measures how evenly work is
  spread across blocks. Synchronous interior-point iterations move at the pace
  of the slowest block, so a ratio far above 1 (roughly > 3) means stragglers
  stall every iteration.
- **adjacent vs. border linking rows**: *adjacent* rows touch exactly two
  neighbouring local blocks (e.g. storage state-of-charge continuity at block
  boundaries) and are cheap; *border* rows span many blocks (global budget /
  CO₂ / energy caps) and feed the root complement. Many border rows are the
  expensive case.
- **max_ranks = n_blocks**: MPI width is capped by the number of blocks, exactly
  as PIPS enforces at solve time. ``diagnose(m, target_cores=...)`` folds any
  cores beyond that cap into threads-per-rank and warns when the target exceeds
  the block count.

``diagnose`` also emits warnings for a non-decomposed model (``n_blocks == 1``),
high border fraction, block imbalance, and empty blocks.

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

Running on a cluster (detached / SLURM)
=======================================

On an HPC system you do not hold a Python process for a multi-hour, multi-node
solve. Split the work into three steps — export, submit, ingest — controlled by
a :class:`linopy.pips.PipsConfig`:

.. code-block:: python

    import linopy.pips as pips

    # 1. build on a login/build node
    pips.assign_blocks(m, "time", 50)
    m.to_pips_files("/lustre/run42")  # put the export on the parallel FS
    cfg = pips.PipsConfig(threads_per_rank=8, linear_solver="pardiso")
    pips.write_job(
        "/lustre/run42",
        cfg,
        binary="/opt/pips/pips_driver",
        nodes=13,
        time="04:00:00",
        partition="fat",
        account="psa",
    )

.. code-block:: bash

    # 2. submit the generated job
    sbatch /lustre/run42/pips.slurm

.. code-block:: python

    # 3. later, in a fresh session, load the result onto the model
    from linopy.io import read_pips_solution

    read_pips_solution("/lustre/run42", model=m)

``write_job`` writes a SLURM script that sets ``--ntasks`` to the block count
(or ``config.n_ranks``, capped at the number of blocks), ``--cpus-per-task`` to
``threads_per_rank``, exports ``OMP_NUM_THREADS``/``MKL_NUM_THREADS``, and runs
the driver under ``srun`` with the chosen ``linear_solver`` and any extra
options. For an interactive allocation, the inline path
(``m.solve(solver_name="pips", solver_options={...})``) uses the same
``PipsConfig`` keys and honours ``launcher="srun"``.

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
