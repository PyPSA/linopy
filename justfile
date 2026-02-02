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

# Benchmark a remote/local branch (checks it out, runs, returns)
# Usage: just bench-branch FBumann:perf/lp-write-speed-combined build
#        just bench-branch origin/master memory
#        just bench-branch my-local-branch lp_write
bench-branch ref phase="build" model="basic" iterations=default_iterations:
    #!/usr/bin/env bash
    set -euo pipefail
    home_branch=$(git rev-parse --abbrev-ref HEAD)
    tmp_bench=$(mktemp -d)
    # Preserve benchmarks/ and results/ across checkout
    cp -r benchmarks/ "$tmp_bench/benchmarks"
    # Sanitize label: replace / and : with -
    label=$(echo "{{ref}}" | tr '/:' '--')
    # Handle remote refs like "FBumann:perf/lp-write-speed-combined"
    ref="{{ref}}"
    if [[ "$ref" == *:* ]]; then
        remote="${ref%%:*}"
        branch="${ref#*:}"
        # Add remote if not present, fetch the branch
        git remote get-url "$remote" 2>/dev/null || git remote add "$remote" "https://github.com/$remote/linopy.git"
        git fetch "$remote" "$branch"
        checkout_ref="$remote/$branch"
    else
        # Local branch or origin ref
        git fetch --all 2>/dev/null || true
        checkout_ref="$ref"
    fi
    echo ">>> Checking out $checkout_ref ..."
    git checkout "$checkout_ref" --detach
    # Restore benchmarks/
    cp -r "$tmp_bench/benchmarks" benchmarks/
    # Install the checked-out linopy
    pip install -e . --quiet 2>/dev/null || true
    echo ">>> Running: model={{model}} phase={{phase}} label=$label ..."
    python -c "from benchmarks.run import run_single; run_single('{{model}}', '{{phase}}', label='$label', iterations={{iterations}}, output_dir='{{results_dir}}')"
    # Save results before switching back
    cp -r benchmarks/results/ "$tmp_bench/results"
    echo ">>> Returning to $home_branch ..."
    git checkout "$home_branch"
    # Restore results from the run
    cp -r "$tmp_bench/results/"* benchmarks/results/ 2>/dev/null || true
    rm -rf "$tmp_bench"
    # Reinstall current branch
    pip install -e . --quiet 2>/dev/null || true
    echo ">>> Done. Results saved with label=$label"

# Compare result JSON files across branches (2 or more)
bench-compare +files:
    python -c "import sys; from benchmarks.compare import compare; compare(*sys.argv[1:])" {{files}}

# List available models and phases
bench-list:
    python -c "from benchmarks.run import list_available; list_available()"
