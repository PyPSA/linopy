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
| `lp_write`       | `test_lp_write.py`                | `model.to_file(...)` — LP / MPS serialization                  |
| `netcdf`         | `test_netcdf.py`                  | `to_netcdf` / `read_netcdf` round-trip                         |
| `solver_handoff` | `test_solver_handoff.py`          | `lp.io.to_highspy` / `to_gurobipy` / `to_mosek` / `to_xpress`  |
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

## Other CLI surfaces

| Command                            | Purpose                                                              |
| ---------------------------------- | -------------------------------------------------------------------- |
| `smoke`                            | CI smoke run — every model/phase at quickest size, no timings (~20s) |
| `run --long`                       | Full sweep including heaviest sizes (knapsack 1M, basic 1600); slow  |
| `sweep <v1> <v2> ...`              | Build fresh venv per linopy version and run the suite in each        |
| `memory sweep <v1> <v2> ...`       | Same shape as `sweep`, but tracks peak RSS per version               |
| `plot --view sweep <s1> <s2> ...`  | Heatmap of ratios across 3+ snapshots                                |
| `plot --view scaling <snap>`       | Log-log time vs `n` for size-parametrized tests, faceted by phase    |
| `notebook`                         | Re-execute this walkthrough end-to-end (what CI runs)                |

Each has its own `--help` with all flags.

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
