# Benchmarks

Modular benchmark framework for linopy. All commands use [`just`](https://github.com/casey/just).

```
$ just --list
Available recipes:
    [benchmark]
    all name iterations=default_iterations
    compare ref="master" model=default_model phase=default_phase iterations=default_iterations quick=default_quick
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

# Full suite
just all my-branch

# Single model + phase
just model my-branch knapsack memory

# Compare current branch against master (all phases, basic model)
just compare

# Compare against another branch
just compare perf/lp-write-speed-combined-bench

# Compare against a remote fork
just compare FBumann:perf/lp-write-speed

# Plot existing result files
just plot benchmarks/results/master_basic_build.json benchmarks/results/feat_basic_build.json
```

## Overriding defaults

Recipe parameters that show `=default_*` reference top-level variables in the justfile.
Override them with `--set` on the command line:

```bash
# Run compare with quick mode
just --set default_quick True compare perf/lp-write-speed

# Compare only the lp_write phase
just --set default_phase lp_write compare perf/lp-write-speed

# Combine multiple overrides
just --set default_quick True --set default_phase build compare perf/lp-write-speed
```

## Output

Results are saved as JSON in `benchmarks/results/` (gitignored), named `{name}_{model}_{phase}.json`. Comparison plots are saved as PNG alongside.
