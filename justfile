default_iterations := "30"
results_dir := "benchmarks/results"

# Run all phases for all models
bench label="dev" iterations=default_iterations:
    python -c "from benchmarks.run import run_all; run_all('{{label}}', iterations={{iterations}}, output_dir='{{results_dir}}')"

# Benchmark build phase only
bench-build label="dev" iterations=default_iterations:
    python -c "from benchmarks.run import run_phase; run_phase('build', label='{{label}}', iterations={{iterations}}, output_dir='{{results_dir}}')"

# Benchmark memory phase only
bench-memory label="dev":
    python -c "from benchmarks.run import run_phase; run_phase('memory', label='{{label}}', output_dir='{{results_dir}}')"

# Benchmark LP write phase only
bench-write label="dev" iterations=default_iterations:
    python -c "from benchmarks.run import run_phase; run_phase('lp_write', label='{{label}}', iterations={{iterations}}, output_dir='{{results_dir}}')"

# Run a single model + phase
bench-model model phase="memory" label="dev" iterations=default_iterations:
    python -c "from benchmarks.run import run_single; run_single('{{model}}', '{{phase}}', label='{{label}}', iterations={{iterations}}, output_dir='{{results_dir}}')"

# Quick smoke test (basic model only, small sizes)
bench-quick label="dev":
    python -c "from benchmarks.run import run_single; run_single('basic', 'build', label='{{label}}', iterations=5, quick=True, output_dir='{{results_dir}}')"
    python -c "from benchmarks.run import run_single; run_single('basic', 'memory', label='{{label}}', iterations=5, quick=True, output_dir='{{results_dir}}')"
    python -c "from benchmarks.run import run_single; run_single('basic', 'lp_write', label='{{label}}', iterations=5, quick=True, output_dir='{{results_dir}}')"

# Benchmark a branch vs current: checkout ref, run bench-quick, return, run bench-quick here, compare
# Usage: just bench-branch FBumann:perf/lp-write-speed-combined
#        just bench-branch origin/master
#        just bench-branch my-local-branch
bench-branch ref:
    #!/usr/bin/env bash
    set -euo pipefail
    home_branch=$(git rev-parse --abbrev-ref HEAD)
    home_label=$(echo "$home_branch" | tr '/:' '--')
    ref_label=$(echo "{{ref}}" | tr '/:' '--')
    ref="{{ref}}"
    if [[ "$ref" == *:* ]]; then
        remote="${ref%%:*}"
        branch="${ref#*:}"
        git remote get-url "$remote" 2>/dev/null || git remote add "$remote" "https://github.com/$remote/linopy.git"
        git fetch "$remote" "$branch" --no-tags --no-recurse-submodules 2>&1 || true
        checkout_ref="FETCH_HEAD"
    else
        git fetch origin --no-tags --no-recurse-submodules 2>&1 || true
        checkout_ref="$ref"
    fi
    echo ">>> Checking out $checkout_ref ..."
    git checkout --detach "$checkout_ref"
    pip install -e . --quiet 2>/dev/null || true
    echo ">>> Running bench-quick on $ref_label ..."
    python -c "from benchmarks.run import run_single; run_single('basic', 'build', label='$ref_label', iterations=5, quick=True, output_dir='benchmarks/results')"
    python -c "from benchmarks.run import run_single; run_single('basic', 'memory', label='$ref_label', iterations=5, quick=True, output_dir='benchmarks/results')"
    python -c "from benchmarks.run import run_single; run_single('basic', 'lp_write', label='$ref_label', iterations=5, quick=True, output_dir='benchmarks/results')"
    echo ">>> Returning to $home_branch ..."
    git checkout "$home_branch"
    pip install -e . --quiet 2>/dev/null || true
    echo ">>> Running bench-quick on $home_label ..."
    python -c "from benchmarks.run import run_single; run_single('basic', 'build', label='$home_label', iterations=5, quick=True, output_dir='benchmarks/results')"
    python -c "from benchmarks.run import run_single; run_single('basic', 'memory', label='$home_label', iterations=5, quick=True, output_dir='benchmarks/results')"
    python -c "from benchmarks.run import run_single; run_single('basic', 'lp_write', label='$home_label', iterations=5, quick=True, output_dir='benchmarks/results')"
    echo ">>> Comparing results ..."
    for phase in build memory lp_write; do
        old="benchmarks/results/${ref_label}_basic_${phase}.json"
        new="benchmarks/results/${home_label}_basic_${phase}.json"
        if [[ -f "$old" && -f "$new" ]]; then
            python -c "from benchmarks.compare import compare; compare('$old', '$new')"
        fi
    done
    echo ">>> Done."

# Compare result JSON files across branches (2 or more)
bench-compare +files:
    python -c "import sys; from benchmarks.compare import compare; compare(*sys.argv[1:])" {{files}}

# List available models and phases
bench-list:
    python -c "from benchmarks.run import list_available; list_available()"
