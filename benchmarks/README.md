# Benchmarks

Modular benchmark framework for linopy. All commands use [`just`](https://github.com/casey/just).

```
$ just --list
Available recipes:
    [benchmark]
    all label="dev" iterations=default_iterations
    compare ref model="basic" phase="all" iterations=default_iterations quick="True"
    list
    model model phase="build" label="dev" iterations=default_iterations quick="True"
    plot +files
    quick label="dev"
```

Start with `just list` to see available models and phases, then `just quick` for a smoke test.

## Examples

```bash
# Discover available models and phases
just list

# Quick smoke test (basic model, all phases, 5 iterations)
just quick

# Full suite
just all label="my-branch"

# Single model + phase
just model knapsack memory label="my-branch" iterations=20

# Compare current branch against master
just compare master

# Compare against a remote fork
just compare FBumann:perf/lp-write-speed model="basic" phase="lp_write"

# Plot existing result files
just plot benchmarks/results/master_basic_build.json benchmarks/results/feat_basic_build.json
```

## Output

Results are saved as JSON in `benchmarks/results/` (gitignored), named `{label}_{model}_{phase}.json`. Comparison plots are saved as PNG alongside.
