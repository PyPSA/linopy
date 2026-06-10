# Internal Performance Benchmarks

End-to-end performance tracking for `linopy` — build → solver handoff
→ netCDF (de)serialization → fixed PyPSA model. Solver algorithm
runtime is out of scope.

**The walkthrough is load-bearing.** Phase coverage, CLI introspection,
the two-snapshot regression workflow with inline Plotly views, and
how to extend the suite live in [`walkthrough.md`](walkthrough.md).
This README only covers install and how to open the walkthrough.

> `benchmark/` (singular) is the legacy external-framework suite.
> `benchmarks/` (plural) is this internal suite.

## Models vs patterns

Two kinds of benchmark spec, same harness (time *or* peak memory — a
`run`/`sweep` `--metric` flag, same phases), distinguished by their sweep axis:

- **Models** (`models/`, `REGISTRY`) — whole `linopy.Model`s swept over
  `size` (axis `n`): "how does cost scale with the problem?"
- **Patterns** (`patterns/`, `PATTERNS`) — fragments of realistic modelling
  code (a balance constraint, a KVL contraction) swept over `severity`
  (0–100, axis `severity`): "how does cost respond as one data shape goes
  from benign to pathological?" Each `PatternSpec.description` documents what
  its dial means (`"0: …, 100: …"`).

Both kinds build a complete `linopy.Model`, so both run the **same phases** and
share the phase drivers (`test_build.py`, `test_matrices.py`, …) and `memory`
grid — they're just more `(spec, value)` rows, tagged by `axis`. There is no
separate pattern driver. Running a pattern through `build` *and* `lp_write`
shows whether a dense-`_term` blow-up propagates to export or collapses.

Patterns target the operations where the dense-`_term` representation forces
materialisation — `groupby().sum()` padding, sparse `@` densification — so a
`severity` sweep draws the cost cliff, and a cross-version `compare` shows a
kernel change bending it. Adding either is one file: drop it in `models/` or
`patterns/`, call `register(...)` / `register_pattern(...)`.

## Install

```bash
uv sync --extra dev --extra benchmarks
source .venv/bin/activate
```

`pypsa` is optional — `pypsa_scigrid` and
`test_pypsa_carbon_management.py` skip gracefully without it. Install
when you need them: `uv pip install pypsa`.

The `[benchmarks]` extra in `pyproject.toml` pins every direct dep that
affects measurement (`numpy`, `scipy`, `xarray`, `pandas`, `polars`,
`dask`, etc.). `sweep` installs these into each per-version venv, so
"same deps, only linopy varies" comes for free without a separate
lockfile — bump the pins in pyproject and the next sweep picks them up.

## Open the walkthrough

```bash
python -m benchmarks notebook --build       # (re)generate walkthrough.ipynb
jupyter lab benchmarks/walkthrough.ipynb    # ...or PyCharm / VSCode
```

The `.md` is the source of truth; the `.ipynb` is a disposable,
gitignored build artifact. Edit the `.md`, re-run `--build`, re-open.
Same workflow in any editor.

CI executes the walkthrough end-to-end on every PR
(`python -m benchmarks notebook`) so the examples can't silently rot.
