# Contributing

We welcome anyone interested in contributing to this project,
be it with new ideas, suggestions, by filing bug reports or
contributing code.

You are invited to submit pull requests / issues to our
[Github repository](https://github.com/pypsa/linopy).

## AI-assisted contributions

We're happy for you to use AI tools when contributing to linopy, but the
conversation around a change — pull request and issue descriptions, comments and
review discussions — must stay human and honest. The rules (mark AI-generated
content, write your own intent by hand) apply to humans and AI agents alike and
are kept in a single place: see [AGENTS.md](https://github.com/pypsa/linopy/blob/master/AGENTS.md) in the repository
root. If you use an AI agent here, point it at that file.

## Development Setup

For linting and formatting, we use [ruff](https://docs.astral.sh/ruff/)
and run it via [pre-commit](https://pre-commit.com/index.html):

- Install the git hook (once): `pre-commit install`
- Run manually: `pre-commit run --all-files`

### Conventions for the dense store

The dense store in `linopy/store.py` has numpy fast paths next to the
general xarray path. To keep them lean, follow three rules:

- A user value entering a fast path passes through `on_cells` (a NaN
    raises) or `laid_on` (a NaN kept). Don't add a scalar NaN check or a
    coordinate comparison at the call site.
- A fast path doesn't call `Dataset(...)` or `transpose` directly. It
    lays fields out with `on_coords`, `DenseStore._own`, `transposed`
    and `DenseStore.normalized`.
- A performance trick with a numeric threshold, e.g. a per-term loop below
    some number of terms, needs a measured table in the pull request. Without
    one, it goes.

## Running Tests

Testing is essential for maintaining code quality. We use pytest as our testing framework.

### Basic Testing

To run the test suite:

```bash
# Install development dependencies
uv sync --extra dev --extra solvers

# Also run the mathspec binder tests (needs Python >= 3.12)
uv sync --extra dev --extra solvers --extra spec

# Run all tests
pytest

# Run tests with coverage
pytest --cov=./ --cov-report=xml linopy --doctest-modules test

# Run a specific test file
pytest test/test_model.py

# Run a specific test function
pytest test/test_model.py::test_model_creation
```

### GPU Testing

Tests for GPU-accelerated solvers (e.g., cuOpt and cuPDLPx) are automatically skipped by default since CI machines and most development environments don't have GPU hardware. This ensures tests pass in all environments.

To run GPU tests locally (requires GPU hardware and CUDA):

```bash
# Run all tests including GPU tests
pytest --run-gpu

# Run only GPU tests
pytest -m gpu --run-gpu
```

GPU tests are automatically detected based on solver capabilities - no manual marking is required. When you add a new GPU solver to linopy, tests using that solver will automatically be marked as GPU tests.

See the [GPU-Accelerated Solving](../user-guide/gpu-acceleration.md) guide for more information about GPU solver setup and usage.

## Performance Benchmarks

When working on performance-sensitive code, use the internal benchmark suite in `benchmarks/` to check for regressions.

```bash
# Install benchmark dependencies (keep the dev extra: the benchmarks need both)
uv sync --extra dev --extra benchmarks

# Smoke run: every spec builds once, no timing
pytest benchmarks/ --benchmark-disable -q

# Compare timing between branches
pytest benchmarks/drivers/test_build.py --benchmark-save=master
pytest benchmarks/drivers/test_build.py --benchmark-save=my-feature --benchmark-compare=0001_master
```

Peak memory is tracked on every pull request by CodSpeed rather than by a local script.

See `benchmarks/README.md` for full details on models, phases, and usage.

## Contributing examples

Nice examples are always welcome. You can submit a
[Jupyter notebook](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html)
(`.ipynb`) directly. The documentation renders notebooks with
[mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) and executes them
on every build, so commit them without outputs; the `nbstripout` pre-commit hook
clears them for you.

For every notebook:

1.  Write the notebook (let's call it `foo.ipynb`) and place it in the
    documentation folder that fits it: `doc/examples/` for end-to-end
    examples, `doc/user-guide/` for guides on a specific feature.

2.  Add it to the `nav` section of `mkdocs.yml`, or tell us in the pull request
    where in the documentation it should show up.

3.  Check the result locally (see [Building the documentation](#building-the-documentation)).

4.  Commit your changes and open a pull request.

## Building the documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/) and the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme. The
configuration lives in `mkdocs.yml`, the pages in `doc/`. Install the `docs`
extra and serve the documentation locally:

```bash
uv pip install -e ".[docs]"
mkdocs serve
```

Notebooks are executed during the build. To skip execution while working on
the text, set `LINOPY_DOCS_EXECUTE=false`:

```bash
LINOPY_DOCS_EXECUTE=false mkdocs serve
```

`mkdocs build --strict` fails on broken links and unresolved API
cross-references; the Read the Docs build runs in strict mode.

API pages under `doc/api/` use [mkdocstrings](https://mkdocstrings.github.io/)
(`::: linopy.model.Model.solve`). Reference API objects from any page with
`` [`Model.solve`][linopy.model.Model.solve] ``.
