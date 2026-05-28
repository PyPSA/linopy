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
