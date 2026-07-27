
============
Contributing
============

We welcome anyone interested in contributing to this project,
be it with new ideas, suggestions, by filing bug reports or
contributing code.

You are invited to submit pull requests / issues to our
`Github repository <https://github.com/pypsa/linopy>`_.

AI-assisted contributions
=========================

We're happy for you to use AI tools when contributing to linopy, but the
conversation around a change — pull request and issue descriptions, comments and
review discussions — must stay human and honest. The rules (mark AI-generated
content, write your own intent by hand) apply to humans and AI agents alike and
are kept in a single place: see `AGENTS.md
<https://github.com/pypsa/linopy/blob/master/AGENTS.md>`_ in the repository
root. If you use an AI agent here, point it at that file.

Development Setup
=================

For linting and formatting, we use `ruff <https://docs.astral.sh/ruff/>`_
and run it via `pre-commit <https://pre-commit.com/index.html>`_:

* Install the git hook (once): ``pre-commit install``
* Run manually: ``pre-commit run --all-files``

Running Tests
=============

Testing is essential for maintaining code quality. We use pytest as our testing framework.

Basic Testing
-------------

To run the test suite:

.. code-block:: bash

    # Install development dependencies
    uv sync --extra dev --extra solvers

    # Run all tests
    pytest

    # Run tests with coverage
    pytest --cov=./ --cov-report=xml linopy --doctest-modules test

    # Run a specific test file
    pytest test/test_model.py

    # Run a specific test function
    pytest test/test_model.py::test_model_creation

GPU Testing
-----------

Tests for GPU-accelerated solvers (e.g., cuPDLPx) are automatically skipped by default since CI machines and most development environments don't have GPU hardware. This ensures tests pass in all environments.

To run GPU tests locally (requires GPU hardware and CUDA):

.. code-block:: bash

    # Run all tests including GPU tests
    pytest --run-gpu

    # Run only GPU tests
    pytest -m gpu --run-gpu

GPU tests are automatically detected based on solver capabilities - no manual marking is required. When you add a new GPU solver to linopy, tests using that solver will automatically be marked as GPU tests.

See the :doc:`gpu-acceleration` guide for more information about GPU solver setup and usage.

Performance Benchmarks
======================

When working on performance-sensitive code, use the internal benchmark suite in ``benchmarks/`` to check for regressions.

.. code-block:: bash

    # Install benchmark dependencies
    uv sync --extra benchmarks

    # Quick timing benchmarks
    pytest benchmarks/ --quick

    # Compare timing between branches
    pytest benchmarks/test_build.py --benchmark-save=master
    pytest benchmarks/test_build.py --benchmark-save=my-feature --benchmark-compare=0001_master

    # Compare peak memory between branches
    python benchmarks/memory.py save master --quick
    python benchmarks/memory.py save my-feature --quick
    python benchmarks/memory.py compare master my-feature

See ``benchmarks/README.md`` for full details on models, phases, and usage.

Contributing examples
=====================

Nice examples are always welcome.

You can even submit your `Jupyter notebook`_ (``.ipynb``) directly
as an example.
For contributing notebooks (and working with notebooks in `git`
in general) we have compiled a workflow for you which we suggest
you follow:

* Locally install `this precommit hook for git`_

This obviously has to be done only once.
The hook checks if any of the notebooks you are including in a commit
contain a non-empty output cells.

Then for every notebook:

1. Write the notebook (let's call it ``foo.ipynb``) and place it
   in ``examples/foo.ipynb``.
2. Ask yourself: Is the output in each of the notebook's cells
   relevant for to example?

    * Yes: Leave it there.
      Just make sure to keep the amount of pictures/... to a minimum.
    * No: Clear the output of all cells,
      e.g. `Edit -> Clear all output` in JupyterLab.

3. Provide a link to the documentation:
   Include a file ``foo.nblink`` located in ``doc/foo.nblink``
   with this content

   .. code-block:: json

      {
          "path": "../examples/foo.ipynb"
      }

   Adjust the path for your file's name.
   This ``nblink`` allows us to link your notebook into the documentation.

4. Link your file in the documentation:

   Either

   * Include your ``foo.nblink`` directly into one of
     the documentation's toctrees; or
   * Tell us where in the documentation you want your example to show up

5. Commit your changes.
   If the precommit hook you installed above kicks in, confirm
   your decision ('y') or go back ('n') and delete the output
   of the notebook's cells.
6. Create a pull request for us to accept your example.

The support for the the ``.ipynb`` notebook format in our documentation
is realised via the extensions `nbsphinx`_ and `nbsphinx_link`_.

.. _Jupyter notebook: https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html
.. _this precommit hook for git: https://jamesfolberth.org/articles/2017/08/07/git-commit-hook-for-jupyter-notebooks/
.. _nbsphinx: https://nbsphinx.readthedocs.io/en/0.4.2/installation.html
.. _nbsphinx_link: https://nbsphinx.readthedocs.io/en/latest/
