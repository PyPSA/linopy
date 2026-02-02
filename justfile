default_iterations := "10"
results_dir := "benchmarks/results"

[group('benchmark')]
all name iterations=default_iterations:
    python -c "from benchmarks.run import run_all; run_all(name='{{name}}', iterations={{iterations}}, output_dir='{{results_dir}}')"

[group('benchmark')]
model name model phase="all" iterations=default_iterations quick="False":
    python -c "from benchmarks.run import run_single; run_single('{{model}}', '{{phase}}', name='{{name}}', iterations={{iterations}}, quick={{quick}}, output_dir='{{results_dir}}')"

[group('benchmark')]
quick name="quick":
    just _run basic build {{name}} 5 True
    just _run basic memory {{name}} 5 True
    just _run basic lp_write {{name}} 5 True

[group('benchmark')]
compare ref="master" model="basic" phase="all" iterations=default_iterations quick="True":
    #!/usr/bin/env bash
    set -euo pipefail
    home_branch=$(git rev-parse --abbrev-ref HEAD)
    home_name=$(echo "$home_branch" | tr '/:' '--')
    ref_name=$(echo "{{ref}}" | tr '/:' '--')

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

    echo ">>> Benchmarking $ref_name (model={{model}}, phases=$phases, quick={{quick}}) ..."
    for phase in $phases; do
        just _run "{{model}}" "$phase" "$ref_name" "{{iterations}}" "{{quick}}"
    done

    echo ">>> Returning to $home_branch ..."
    git checkout "$home_branch"
    pip install -e . --quiet 2>/dev/null || true

    echo ">>> Benchmarking $home_name (model={{model}}, phases=$phases, quick={{quick}}) ..."
    for phase in $phases; do
        just _run "{{model}}" "$phase" "$home_name" "{{iterations}}" "{{quick}}"
    done

    echo ">>> Comparing results ..."
    for phase in $phases; do
        old="benchmarks/results/${ref_name}_{{model}}_${phase}.json"
        new="benchmarks/results/${home_name}_{{model}}_${phase}.json"
        if [[ -f "$old" && -f "$new" ]]; then
            python -c "from benchmarks.compare import compare; compare('$old', '$new')"
        fi
    done
    echo ">>> Done."

[group('benchmark')]
plot +files:
    python -c "import sys; from benchmarks.compare import compare; compare(*sys.argv[1:])" {{files}}

[group('benchmark')]
list:
    python -c "from benchmarks.run import list_available; list_available()"

[private]
_run model phase name iterations quick:
    python -c "from benchmarks.run import run_single; run_single('{{model}}', '{{phase}}', name='{{name}}', iterations={{iterations}}, quick={{quick}}, output_dir='{{results_dir}}')"
