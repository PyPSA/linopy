# Benchmarks

Modular benchmark framework for linopy. All commands use [`just`](https://github.com/casey/just).

```
$ just --list
Available recipes:
    [benchmark]
    all name iterations=default_iterations
    compare ref="master" model=default_model phase=default_phase iterations=default_iterations quick="False"
    compare-all ref="master" iterations=default_iterations
    compare-quick ref="master"
    list
    model name model phase=default_phase iterations=default_iterations quick="False"
    plot +files
    quick name="quick"
```

Start with `just list` to see available models and phases, then `just quick` for a smoke test.

## Examples

```bash
# Discover available models and phases
just list

# Quick smoke test (basic model, all phases, 5 iterations)
just quick

# Full suite (all models, all phases)
just all my-branch

# Single model + phase
just model my-branch knapsack memory

# Compare current branch against master (basic model, all phases)
just compare

# Compare all models against master
just compare-all

# Quick compare (basic model, small sizes, 5 iterations)
just compare-quick perf/lp-write-speed

# Compare against a remote fork
just compare FBumann:perf/lp-write-speed

# Plot existing result files
just plot benchmarks/results/master_basic_build.json benchmarks/results/feat_basic_build.json
```

## Overriding defaults

Parameters showing `=default_*` reference top-level justfile variables. Override them with `--set`:

```bash
just --set default_phase lp_write compare perf/lp-write-speed
just --set default_model knapsack --set default_iterations 20 compare master
```

## Output

Results are saved as JSON in `benchmarks/results/` (gitignored), named `{name}_{model}_{phase}.json`. Comparison plots are saved as PNG alongside.
