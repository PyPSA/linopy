default_iterations := "10"
results_dir := "benchmarks/results"

[group('run')]
bench label="dev" iterations=default_iterations:
    python -c "from benchmarks.run import run_all; run_all('{{label}}', iterations={{iterations}}, output_dir='{{results_dir}}')"

[group('run')]
bench-model model phase="build" label="dev" iterations=default_iterations quick="True":
    python -c "from benchmarks.run import run_single; run_single('{{model}}', '{{phase}}', label='{{label}}', iterations={{iterations}}, quick={{quick}}, output_dir='{{results_dir}}')"

[group('run')]
bench-quick label="dev":
    just bench-run basic build {{label}} 5 True
    just bench-run basic memory {{label}} 5 True
    just bench-run basic lp_write {{label}} 5 True

[group('compare')]
bench-branch ref model="basic" phase="all" iterations=default_iterations quick="True":
    #!/usr/bin/env bash
    set -euo pipefail
    home_branch=$(git rev-parse --abbrev-ref HEAD)
    home_label=$(echo "$home_branch" | tr '/:' '--')
    ref_label=$(echo "{{ref}}" | tr '/:' '--')

    if [[ "{{phase}}" == "all" ]]; then
        phases="build memory lp_write"
    else
        phases="{{phase}}"
    fi

    ref="{{ref}}"
    if [[ "$ref" == *:* ]]; then
        remote="${ref%%:*}"
        branch="${ref#*:}"
        git remote get-url "$remote" 2>/dev/null || git remote add "$remote" "https://github.com/$remote/linopy.git"
        git fetch "$remote" "$branch" --no-tags --no-recurse-submodules
        checkout_ref="FETCH_HEAD"
    else
        git fetch origin --no-tags --no-recurse-submodules 2>&1 || true
        checkout_ref="$ref"
    fi

    echo ">>> Checking out $checkout_ref ..."
    git checkout --detach "$checkout_ref"
    pip install -e . --quiet 2>/dev/null || true

    echo ">>> Benchmarking $ref_label (model={{model}}, phases=$phases, quick={{quick}}) ..."
    for phase in $phases; do
        just bench-run "{{model}}" "$phase" "$ref_label" "{{iterations}}" "{{quick}}"
    done

    echo ">>> Returning to $home_branch ..."
    git checkout "$home_branch"
    pip install -e . --quiet 2>/dev/null || true

    echo ">>> Benchmarking $home_label (model={{model}}, phases=$phases, quick={{quick}}) ..."
    for phase in $phases; do
        just bench-run "{{model}}" "$phase" "$home_label" "{{iterations}}" "{{quick}}"
    done

    echo ">>> Comparing results ..."
    for phase in $phases; do
        old="benchmarks/results/${ref_label}_{{model}}_${phase}.json"
        new="benchmarks/results/${home_label}_{{model}}_${phase}.json"
        if [[ -f "$old" && -f "$new" ]]; then
            python -c "from benchmarks.compare import compare; compare('$old', '$new')"
        fi
    done
    echo ">>> Done."

[group('compare')]
bench-compare +files:
    python -c "import sys; from benchmarks.compare import compare; compare(*sys.argv[1:])" {{files}}

[group('info')]
bench-list:
    python -c "from benchmarks.run import list_available; list_available()"

[private]
bench-run model phase label iterations quick:
    python -c "from benchmarks.run import run_single; run_single('{{model}}', '{{phase}}', label='{{label}}', iterations={{iterations}}, quick={{quick}}, output_dir='{{results_dir}}')"
