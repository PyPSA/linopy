---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Linopy benchmarks — CLI walkthrough

> ⚠️ **This file is the source. Don't edit the `.ipynb` directly.**
> Run `python -m benchmarks notebook --build` to (re)generate
> `walkthrough.ipynb` from this `.md`, then open the `.ipynb` in
> JupyterLab / PyCharm / VSCode to view and run cells. To change the
> walkthrough's content, edit the `.md`, then re-run `--build`. The
> `.ipynb` is gitignored.

Internal performance tracking for `linopy`. This notebook shows the
typer CLI working end-to-end: introspect what's registered, run a
timing snapshot, diff two snapshots, render the comparison views
inline.

For what this notebook deliberately doesn't duplicate:

- **Install + size tiers** → [`benchmarks/README.md`](README.md)
- **Every CLI flag** → `python -m benchmarks --help` (rich-rendered);
  `--help` on any subcommand drills in.

## What's measured

| Phase            | Test file                         | Measures                                                       |
| ---------------- | --------------------------------- | -------------------------------------------------------------- |
| `build`          | `test_build.py`                   | constructing variables / expressions / constraints / objective |
| `matrices`       | `test_matrices.py`                | `A`, `b`, `c`, bounds, labels, `Q` for QP                      |
| `to_lp`          | `test_to_lp.py`                   | `model.to_file(...)` — LP / MPS serialization                  |
| `to_netcdf` / `from_netcdf` | `test_netcdf.py`       | netCDF write / read round-trip                                 |
| `to_solver`      | `test_to_solver.py`               | `lp.io.to_highspy` / `to_gurobipy` / `to_mosek` / `to_xpress`  |
| end-to-end       | `test_pypsa_carbon_management.py` | fixed PyPSA model → highspy; sweeps `freeze_constraints`       |

Solver algorithm runtime is intentionally out of scope.

## Setup

Locate the repo so the shell cells below can run `python -m benchmarks`
regardless of where Jupyter was launched, and pick a tempdir for the
snapshot/plot files we'll produce.

```{code-cell} ipython3
import os
import sys
import tempfile
from pathlib import Path

# CI sets LINOPY_REPO_ROOT; locally we walk up from cwd.
_root = os.environ.get("LINOPY_REPO_ROOT") or next(
    (
        str(p) for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
        if (p / "benchmarks" / "registry.py").exists()
    ),
    None,
)
if _root is None:
    raise RuntimeError(
        "Could not locate linopy repo root. Set LINOPY_REPO_ROOT or launch "
        "Jupyter from somewhere inside the repo."
    )

# Subshells launched by ``!``-cells inherit cwd, env, and PYTHONPATH.
os.chdir(_root)
os.environ["PYTHONPATH"] = f"{_root}:{os.environ.get('PYTHONPATH', '')}"
# Rich/click disable colour when stdout isn't a TTY (and the ``!`` pipe
# isn't); ``FORCE_COLOR`` overrides that so typer's ``--help`` panels
# render with colour in the notebook output.
os.environ["FORCE_COLOR"] = "1"

_tmp = Path(tempfile.mkdtemp(prefix="bench-walkthrough-"))
baseline = _tmp / "baseline.json"
candidate = _tmp / "candidate.json"
scatter_html = _tmp / "scatter.html"
compare_html = _tmp / "compare.html"

print(f"repo root: {_root}")
print(f"tempdir:   {_tmp}")
```

## Introspect the registry

`list` enumerates registered specs. `--details` shows the feature tags
and size range each spec covers, so you can pick a focused target.

```{code-cell} ipython3
!python -m benchmarks list --details
```

`show <name>` drills into one spec — every attribute the registry
exposes, including which phases it's eligible for and the
`quick_threshold` / `long_threshold` gating its sizes.

```{code-cell} ipython3
!python -m benchmarks show basic
```

`filter` narrows by feature tag (`quadratic`, `integer`, `sos`, …) or
phase tag — useful when you only care about a subset of the suite.

```{code-cell} ipython3
!python -m benchmarks filter --feature quadratic
```

## Patterns — severity-swept idioms

Alongside whole-model specs, the suite registers **patterns**: fragments of
realistic modelling code (a nodal balance, a KVL contraction) parametrised by
`severity` (0–100) instead of `size`. `severity` dials one data shape from
benign (0) to pathological (100), so a sweep draws the cost cliff and a
cross-version `compare` shows a kernel change bending it. `list --kind patterns`
shows just the patterns; `show <name>` prints what a pattern's dial means.

```{code-cell} ipython3
!python -m benchmarks show nodal_balance
```

A pattern builds a complete model, so it runs the **same phases** as a model
and rides the same phase drivers — there is no separate pattern test file.
Patterns are tagged by the `severity` axis in their test id, so the usual tools
target them by filtering on it:

```bash
pytest benchmarks/ -k severity                        # all patterns, every phase
pytest benchmarks/ -k nodal_balance                   # one pattern
python -m benchmarks run --filter severity --quick    # patterns, timing
python -m benchmarks memory save mylabel --filter severity   # patterns, memory
```

(`--filter`/`-k` selects specs by name or id substring on both `run` and
`memory save` — `nodal_balance` for one spec, `severity` for all patterns,
`n=` for models. `list --kind {models,patterns}` browses them.)

## Run a timing snapshot

`run` is the main timing entry point. Below we run twice with
`--quick --phase build` (~10 s each) to get a baseline / candidate
pair we can diff. On a real PR you'd run once on `master` and once on
your branch.

```{code-cell} ipython3
!python -m benchmarks run --quick --phase build --json {baseline}
```

```{code-cell} ipython3
!python -m benchmarks run --quick --phase build --json {candidate}
```

The diff between two `--quick` runs of the same code is just
measurement noise — that's expected. On a real PR the numbers below
would actually move.

## Diff snapshots

### Text table — `compare`

`compare` wraps `pytest-benchmark compare` with opinionated defaults:
group by full test name, sort by `min`, show min + IQR. One mini-table
per test with the baseline + candidate rows and a relative-speedup
factor flagging the slower one. Scales to 30+ tests, just long output.

```{code-cell} ipython3
!python -m benchmarks compare {baseline} {candidate}
```

### Scatter view — exploratory plot

x = baseline cost on a log axis, y = ratio (candidate / baseline),
colour = absolute Δ. **Top-right = slow tests that got slower** —
the "fix this" zone. Top-left = cheap tests with big ratio swings
(noise, not real change). Bottom-right = already-slow tests that
didn't move. Resolves the absolute-vs-relative tension that either
axis alone has a blind spot for.

```{code-cell} ipython3
!python -m benchmarks plot --view scatter {baseline} {candidate} -o {scatter_html}

from IPython.display import HTML
HTML(scatter_html.read_text())
```

### Compare view — sorted-Δ bar chart

The "did this PR regress anything, ranked by impact" picture. Bars
sorted by absolute time delta by default (`--sort relative` switches
to percent). Diverging colour around zero.

```{code-cell} ipython3
!python -m benchmarks plot --view compare {baseline} {candidate} -o {compare_html}
HTML(compare_html.read_text())
```

### In Python — load straight from file

The CLI views above all sit on one function, `load_long_df`, which reads
snapshot json files (timing *or* memory) into a tidy frame —  `snapshot`,
`test_id`, `phase`, `model`, `size`, `value` — plus the unit. Re-exported
from the package so you can do your own analysis without pulling in
plotly:

```{code-cell} ipython3
from benchmarks import load_long_df

df, unit = load_long_df([baseline, candidate])
print(f"unit: {unit}")
df.head()
```

Pivot to one column per snapshot and the comparison is a couple of pandas
lines — the same baseline-vs-candidate diff the `compare` view draws,
here as a DataFrame you can sort, filter, or feed onward:

```{code-cell} ipython3
wide = df.pivot_table(
    index=["phase", "spec", "size"], columns="snapshot", values="value"
)
wide["ratio"] = wide["candidate"] / wide["baseline"]
wide.sort_values("ratio", ascending=False)
```

(Two `--quick` runs of the same code, so the ratios are ~1 ± noise; on a
real PR they'd move. The same frame feeds the plot views — pass the files
to `python -m benchmarks plot` for the rendered version.)

## Memory snapshots

`memory save <label>` runs benchmarks under `memray.Tracker` and
writes peak allocations (MiB) per `(phase, spec, size)` to
`.benchmarks/memory/<label>.json`. The model is built **outside** the
tracked region so peak reflects only the phase work, not model
construction.

```{code-cell} ipython3
!python -m benchmarks memory save baseline_mem --quick --phase build
```

```{code-cell} ipython3
!python -m benchmarks memory save candidate_mem --quick --phase build
```

`memory compare` prints a per-test table of the two labels with
percent change — same shape as the timing `compare`, different
metric. Tests present in only one snapshot show `—` for the missing
column.

```{code-cell} ipython3
!python -m benchmarks memory compare baseline_mem candidate_mem
```

For cross-version memory tracking (analogous to `sweep` for timing),
use `memory sweep <v1> <v2> ...` — same per-version venv shape, peak
RSS metric.

Those per-phase peaks are *marginal* — each tracker sees only its own phase, so
the resident model is excluded. The end-to-end peak a build-then-export session
hits can't be recovered by summing them, so it's measured directly by the
opt-in `pipeline` phase (build → matrices → lp_write in one tracker). It re-runs
those phases, so it's not in the default set — request it standalone:

```bash
python -m benchmarks memory save ceiling --phase pipeline
```

## Benchmarking custom things — the `bench` API

The CLI measures the fixed registry grid. When you want to time or
memory-profile *something the registry doesn't have* — a builder called
with odd arguments, a phase verb on a model you built by hand, a one-off
lambda — reach for `benchmarks.bench`. It measures in-process on the
**current** tree and hands back a result you can inspect or drop into a
snapshot the `plot` / `compare` machinery already reads. (It can't feed
`sweep`, which runs pytest in per-version subprocesses — promote a model
to `benchmarks/models/` to sweep it.)

`bench.time` times any callable with the suite's min-of-N convention. It
is *not* pytest-benchmark's calibrated timer, so compare `bench` numbers
only to other `bench` numbers:

```{code-cell} ipython3
from benchmarks import REGISTRY, bench

bench.time(REGISTRY["basic"].build, 100, rounds=5)
```

Any callable works — including a phase verb applied to a model the
registry has never heard of. `bench.memory` profiles peak RSS through
the same `memray` path the `memory` command uses:

```{code-cell} ipython3
import linopy
from benchmarks.phases import touch_matrices

m = linopy.Model()
x = m.add_variables(coords=[range(2000)], dims=["i"], name="x")
m.add_constraints(x >= 1)
m.add_objective(x.sum())

bench.memory(touch_matrices, m)
```

`bench.compare` runs several callables and collects a `ResultSet`.
`to_snapshot` writes it in the on-disk shape `load_long_df` reads — the
seam every plot view sits on — so in-process results round-trip through
the existing tooling without a detour:

```{code-cell} ipython3
from benchmarks import load_long_df

rs = bench.compare(
    {
        "listcomp": lambda: [i * i for i in range(10_000)],
        "map": lambda: list(map(lambda i: i * i, range(10_000))),
    },
    rounds=20,
)

bench_snap = _tmp / "bench.json"
rs.to_snapshot(bench_snap)

df, unit = load_long_df([bench_snap])
print(f"unit: {unit}")
df
```

Those label-keyed ids land in the `other` bucket. For a size-`scaling`
plot, write each result with `spec=` / `size=` / `phase=` so the id
parses into those columns — `plot` then treats it like any suite
snapshot:

    bench.time(REGISTRY["basic"].build, 100).to_snapshot(
        snap, spec="basic", size=100, phase="build"
    )

## Extending the suite

Add a new model:

1. Drop `benchmarks/models/<name>.py` with a `build_<name>(size) -> linopy.Model`.
2. Build a `ModelSpec`, call `register(...)` at module scope, declare
   realistic `quick_threshold` / `long_threshold` so the smoke run
   stays fast.
3. Import it in `benchmarks/models/__init__.py` so registration fires
   on first import.

Every phase test that lists `<name>` in its applicable phases picks it
up automatically via `iter_params(phase)`. The first introspection
section of this notebook will list your new spec on the next run.
