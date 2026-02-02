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

# Quick smoke test (small sizes, few iterations)
bench-quick label="dev":
    python -c "from benchmarks.run import run_all; run_all('{{label}}', iterations=5, quick=True, output_dir='{{results_dir}}')"

# Compare two result JSON files
bench-compare old new:
    python -c "from benchmarks.compare import compare; compare('{{old}}', '{{new}}')"

# List available models and phases
bench-list:
    python -c "from benchmarks.run import list_available; list_available()"
