# experiment — Benders decomposition of a linopy model with Plasmo.jl

Goal: reproduce the Benders decomposition from
[`Linopy2Plasmo.jl`](https://github.com/leonardgoeke/Linopy2Plasmo.jl) on
`testProblem.nc`, but driving everything from **Python** through the new
`linopy/contrib/plasmo.py` module instead of from the Julia REPL.

`testProblem.nc` is a linopy model (an energy-system capacity-expansion problem,
ZEN-garden-style: `capacity`, `flow_*`, `storage_level`, `carbon_emissions_*`,
`cost_*`, objective `net_present_cost`, sense `min`). It is the same file
Linopy2Plasmo.jl ships as `data/testProblem.nc` — only the suffix was fixed.

## How Linopy2Plasmo.jl works

It is a thin Julia layer. The *input parsing* is pure linopy-via-PythonCall;
the *graph building* is JuMP/Plasmo. Pipeline:

```
read_netcdf → extract sets/vars/cns into tables → assign to nodes
            → group nodes into subgraphs → build OptiGraph → BendersAlgorithm
```

### 1. Read the model (`lin2plasObj(path)`, `objects.jl`)

Loads the `.nc` via `linopy.io.read_netcdf` and flattens it into integer-keyed
tables. Three kinds of data come out:

- **`sets`** — `Dict{set_name => Dict{element_string => int}}`. Every coordinate
  value that appears on any variable/constraint dim whose name contains `"set"`
  is interned to a dense integer index. `revSets` is the inverse.
- **`var`** — one DataFrame per variable. Built by:
  - `data = model.variables[v].data`
  - drop unused entries: `.where(labels != -1, drop=True)`
  - `.stack(...).dropna()` → long/tidy form, one row per scalar variable
  - columns: the set dims (as int indices) + `lower`, `upper`, and
    `key` (= linopy's integer `labels`, the global variable id).
- **`cns`** — one DataFrame per constraint. Same flattening on
  `model.constraints[c].data`, dropping rows where
  `vars == -1 | coeffs == 0 | rhs == Inf`. Keeps `coeffs`, `rhs`, `sign`,
  `vars` (variable id), `labels` (constraint id). Rows sharing a `labels` are
  grouped into one `cnsObj{var::Vector{varid=>coeff}, rhs, sense}`.
- **`obj`** — `DataFrame(var = objective varids, coeff = coeffs)`, plus
  `objSense` from `model.objective.sense`.

Key linopy accessors used (these are what `plasmo.py` must reproduce on the
Python side):

| Julia (PythonCall) | meaning |
|---|---|
| `model.variables[v].data` | xarray Dataset with `labels`, `lower`, `upper` |
| `.labels` | global integer id per variable (−1 = absent) |
| `model.constraints[c].data` | xarray with `vars`, `coeffs`, `rhs`, `sign`, `labels`, plus a `_term` dim |
| `model.objective.expression.vars.data` / `.coeffs.data` | flat objective varids and coeffs |
| `model.objective.sense` | `"min"` / `"max"` |

So most of step 1 is **plain linopy/xarray work** and belongs in Python.

### 2. Assign to nodes (`structureIntoNodes!(obj, split_tup)`)

`split_tup` is an ordered tuple of set names defining the node hierarchy, e.g.
`(:set_time_steps_yearly, :set_nodes, :set_time_steps_operation, :set_time_steps_storage)`.

A **node** is a tuple of integers, one slot per set in `split_tup`. For each
constraint, the slot is the constraint's index in that set if the constraint is
dimensioned over it, else `0`. So `0` means "not resolved along this axis" —
i.e. a coupling/aggregate level. `nodes` = the set of all distinct node tuples
observed across constraints. `varNode[varid]` = the list of nodes a variable
appears in (collected from the constraints it participates in). Objective
variables are pinned to their highest (most-zeros) node via `sortNodes`.

`sortNodes(nodes)` groups node tuples by **how many zero slots** they have —
more zeros = higher in the hierarchy. Used to decide which node "owns" a linking
constraint (always the higher one).

### 3. Subgraph layout (`def_dic`)

A `Dict{Symbol => Vector{node tuple}}` partitioning `nodes` into named
subgraphs. For Benders you need a `:top` master plus one or more `:sub`s. The
README's example:

```julia
def_dic[:top] = filter(x -> x[1] in (0,) || x[2] == 0, nodes)   # year 0 OR node 0
for (i,j) in enumerate([1,2,3])
    def_dic[Symbol(:sub,i)] = filter(x -> x[1]==j && x[2]!=0, nodes)  # one subproblem per year
end
```

i.e. partition by the value of the first hierarchy slot (year): aggregate/year-0
nodes → master, each concrete year → its own subproblem.

### 4. Build the Plasmo OptiGraph (`createOptProblem!(obj, def_dic)`)

- `OptiGraph` `mainGraph`; one `OptiGraph` per subgraph, added as subgraphs.
- One `OptiNode` per node tuple, placed in its subgraph.
- **Variables**: each var row is expanded to every node it belongs to
  (`flatten(:subGraph)`); a JuMP variable with its `lower`/`upper` bounds is
  created on the owning node. `varMap[(varid, node)] => VariableRef`.
- **Constraints**: each `cnsObj` becomes `@constraint(node, Σ coeff·var ⋛ rhs)`,
  pulling the per-node `VariableRef` from `varMap`.
- **Linking constraints**: any variable living in >1 node gets equality links
  `var@nodeA == var@nodeB`. Across subgraphs → `@linkconstraint` on `mainGraph`;
  within a subgraph → on the subgraph. These are the complicating variables
  Benders cuts on.
- **Objective**: per-node objective (sum of its objective vars·coeffs), then
  `set_to_node_objectives` rolls them up.

### 5. Solve

```julia
benders = BendersAlgorithm(mainGraph, subGraphs[:top];
                           solver=optimizer_with_attributes(HiGHS.Optimizer),
                           add_slacks=true, max_iters=1000, regularize=true)
run_algorithm!(benders)
```

(Project ships Gurobi + HiGHS; HiGHS is the open default.)

### 6. Results

`replaceSetColumns(var[:capacity], revSets)` maps int indices back to strings,
then `Plasmo.value(benders, varMap[(key, subGraph)])` reads each variable's
solution.

### Known limitations

- Continuous variables only — no binary/integer support. This carries over to
  the Python port too, but it's a gap in *our* build step (it never declares
  integer/binary variables), not a PlasmoBenders limitation — PlasmoBenders
  itself supports MIPs (`is_MIP`, `strengthened` cuts).
- `structureIntoNodes!` is Linopy2Plasmo's bottleneck — but this does **not**
  carry over. It was slow because it built node tuples row-by-row in Julia
  DataFrames; in `plasmo.py` node assignment is vectorized numpy/linopy
  (`np.isin` masks over `clabels`, membership from a column scan of `mat.A`), so
  the step is cheap.

## Plan for the Python port (`linopy/contrib/plasmo.py`)

The experiment uses **juliacall** (`pyjuliacall` in `pixi.toml`) to call
Plasmo/PlasmoBenders from Python. We do all linopy-side extraction and the
node/subgraph assignment in Python (it's just pandas/xarray there), and hand the
problem to Julia as flat **numpy arrays shared in memory** for graph
construction and solving.

### Decided

1. **No Linopy2Plasmo.jl dependency.** The Julia we need lives in
   `linopy/contrib/plasmo/helpers.jl`, `include`d via juliacall. If it grows
   unwieldy, promote it to a package later — not now.
2. **Hand-off = in-memory shared numpy arrays.** juliacall wraps numpy arrays as
   Julia arrays without copying, so `helpers.jl` receives plain
   `Vector{Int}`/`Vector{Float64}` views. No temp files, no DataFrames across the
   boundary. Strings (set element names) stay in Python; only integer indices
   cross.
3. **Node/subgraph layout in the linopy API** is the main open task — see below.

### Where the data comes from (resolved)

Two sources, each for one job — we do *not* replay the Julia code's
stack-everything-and-rebuild dance:

1. **Matrix data → `m.matrices`** (built once). Gives `A` (CSR `csr_array`),
   `b`, `sense`, `lb`, `ub`, `c`, and the global-id arrays `vlabels`/`clabels`.
   This is the whole LP; per-node blocks are row-slices of it (see
   *Representation & construction*). No `.flat`, no manual xarray flattening.

2. **Node assignment → the `labels` coordinate array.** The one thing `matrices`
   doesn't carry is *which set-coords each constraint has* (needed by the
   partition predicates). That lives on `model.constraints[c].labels` —
   dimensioned over the constraint's set dims (no `_term`), value = constraint id
   (`-1` = absent). Stacking + dropping `-1` gives, per constraint id, its
   position in each set. The partition (below) evaluates its predicates against
   these coords and produces `node_of_cns` keyed by `clabels`.

**Label → matrix position** is O(1) via linopy's `label_index` (no `np.isin`
scan): `m.variables.label_index.label_to_pos[label]` = CSR **column**,
`m.constraints.label_index.label_to_pos[label]` = CSR **row**. Their `vlabels` /
`clabels` are identical to `matrices.vlabels`/`clabels` (verified), so
`node_of_cns` (keyed by `clabels`) is already in CSR row order.

### Variable membership & linking (one incidence matrix)

Membership is a **node × variable bool incidence** `M`, built globally without a
per-node loop. Every nonzero of `A` is a `(constraint-row, var-col)`; expand the
row to its node and scatter:

```python
nnz_node = np.repeat(node_of_cns, np.diff(A.indptr))  # node of each A nonzero
M = coo_array(
    (np.ones(A.nnz, bool), (nnz_node, A.indices)), shape=(n_nodes, n_vars)
).tocsr()  # dedup dup entries → incidence
Mc = M.tocsc()  # one conversion, used twice below
```

`M`/`Mc` yield everything downstream needs, verified on `testProblem.nc`:

- **membership per node** = row `k` of `M` (`M.indices[M.indptr[k]:M.indptr[k+1]]`
  = that node's variable columns) — the per-node column set, computed once.
- **linking variables** = column degree > 1, read straight off the CSC pointers
  (no reduction): `deg = np.diff(Mc.indptr); linking = np.flatnonzero(deg > 1)`.
  We convert to CSC for the topology anyway, so the degree is free here.
- **link topology** = for a linking var `v`,
  `Mc.indices[Mc.indptr[v]:Mc.indptr[v+1]]` = the nodes it lives in → pick owner
  (earliest node in the partition's declared order) and emit `(v, owner, other)`
  star links.

A variable *must* be recorded per node (not just "seen in a row-slice") precisely
because linking constraints need the full node set for every shared variable —
`M`/`Mc` is that record. Objective-only / unconstrained vars have no `A` nonzero
→ absent from `M` → assign to the **first-declared node** separately.

### Representation & construction (resolved)

Everything that decides *which constraint/variable goes in which node* stays in
**Python + linopy**. Julia (`helpers.jl`) receives only per-node **sparse
matrices** and a link list — no ids, no linopy, no xarray. `helpers.jl` is
"sparse matrix + link list → OptiGraph".

Terminology: **node** (Plasmo's term), not "region". One node = one subgraph in
the Benders layout for now (`ponytail:` one-node-per-subgraph; add finer nodes
only if a model needs them).

**Python (`plasmo.py`):**

1. **Partition** constraints → `node_of_cns` (label → node int) via the layout
   API below. Disjoint & exhaustive.
2. **Derive variable membership**: scan `mat.A`'s columns per node block; a
   variable in >1 node is a **linking variable** (see below). Objective-only /
   unconstrained variables → assigned to the first-declared node.
3. **Slice per-node blocks** from the *whole-model* `m.matrices` (built once).
   `mat.A` is **CSR** (`scipy.sparse.csr_array`, verified against 0.8-dev on
   `testProblem.nc`), so row-slicing is contiguous and cheap:
   ```python
   mat = m.matrices  # A (CSR), b, sense, lb, ub, c, vlabels, clabels
   row = np.isin(mat.clabels, node_cns_labels)  # this node's constraints
   Ablk = mat.A[row]  # cheap CSR row-slice
   cols = np.unique(Ablk.indices)  # vars this node touches
   pos = np.full(mat.A.shape[1], -1)
   pos[cols] = np.arange(cols.size)
   colval_local = pos[Ablk.indices]  # remap global → node-local cols
   ```
   No column *slice* and no CSC conversion: we hand Julia the CSR arrays and it
   walks rows, remapping columns via `pos`. (COO single-pass scatter is the
   fallback only if node count explodes into many tiny nodes.)
4. **Transfer to Julia as CSR** (`indptr, colval_local, nzval` + `b, sense, lb,
   ub, c, vlabels_node=cols`), as flat numpy arrays shared via juliacall.
5. **Read back** results by `vlabels` → linopy variable positions.

**Julia (`helpers.jl` via juliacall):** per node, `OptiNode` with `length(lb)`
variables (bounds `lb/ub`); add constraints (below); objective (see next); add
the cross-node equality links on `main`; `BendersAlgorithm(main, master)` where
`master` = the subgraph of the **first-declared node**; expose solution as numpy.

**Objective: each coefficient applied once, on the variable's owner node.** A
variable's objective term must land on **exactly one** node, or it is
double-counted across the Benders subproblems. For a linking variable that is the
**owner** node (the same canonical node the equality links point to); for a
node-local variable it is its only node; for an **objective-only** variable (in
`c` but no `A` nonzero → absent from `M`) it is the first-declared node.
Linopy2Plasmo does the former via `sortNodes(varSub_dic[x])[1][1]` (top of the
var's node set) but **crashes on objective-only vars** (`varSub_dic[x]` KeyError)
— assuming every objective var also sits in a constraint. Our fallback for those
is a new decision, not a port.

**Constraints are built row-by-row.** Tested empirically (`pixi run`):
`@constraint(model, A*x .<= b)` works on a plain JuMP `Model` but the
**vectorized/broadcast form fails on a Plasmo `OptiNode`** — `MethodError:
length(::OptiNode)` (the node isn't broadcastable; both `.<=` and `A*x - b in
Nonpositives(n)` hit it). So we walk the CSR rows and build one affine expression
per row:
```julia
T = GenericAffExpr{Float64, eltype(x)}   # NOTE: OptiNode vars are NodeVariableRef,
                                          # not VariableRef — AffExpr(0.0) errors
for i in 1:n_rows
    expr = zero(T)
    for k in indptr[i]:(indptr[i+1]-1)          # this row's nonzeros (CSR-contiguous)
        add_to_expression!(expr, nzval[k], x[colval_local[k]])
    end
    s = sense[i]
    set = s == '<' ? MOI.LessThan(b[i]) : s == '>' ? MOI.GreaterThan(b[i]) : MOI.EqualTo(b[i])
    add_constraint(node, build_constraint(error, expr, set))   # function form, no macro
end
```

*Benchmark (27000×22000, ~4 terms/row — real block size):* term-loop vs
vectorized `A*x` vs macro vs function form all land within ~2× on time (46–72ms)
and ~1.2× on memory (83–103 MB) — **constraint building is not a bottleneck**
(LP solves dominate). Two real findings, not perf: (1) the expression type must
be `GenericAffExpr{Float64, eltype(x)}`; (2) the non-macro `add_constraint` /
`build_constraint` form is both fastest-tier and lowest-memory (~20% less alloc
than the macro), so it's the default. Vectorized `A*x` does **not** blow up
memory (JuMP's sparse matvec is efficient) — but still needs the per-row
`@constraint` loop since broadcast fails on the node, so it buys nothing.

**ConstraintRefs are not stored.** Storing them costs ~nothing, but nothing reads
them (Benders needs variable *values*, not constraint refs/duals). Store only if
we later add an IIS/dual-readback feature. `ponytail:` drop refs; add when a dual
consumer exists.

**Linking variables (star, not clique).** A variable in nodes {A, B, C} is
instantiated once per node; pick a canonical **owner** = the node earliest in the
partition's declared order (the master is simply the **first-declared node** —
`:top` in our example, but the name is not special), and add
`@linkconstraint(main, x_owner == x_other)` for each other node. Star keeps link
count low and names the master copy Benders cuts on. This is Linopy2Plasmo's
`varNode` / `sortNodes`-owner idea, minus the hierarchy tuple.

### The layout API (resolved)

Replaces Linopy2Plasmo's `split_tup` + hand-written `filter` lambdas. There is
**no node hierarchy tuple** — the set of distinct node labels observed in the
data *is* the node/subgraph set, discovered, not declared.

**A partition is an ordered `{name: predicate}` over constraint dims.** A
predicate maps a constraint (via its `labels` set-coords) to node membership;
**first match wins**, so the partition is disjoint and exhaustive over
constraints (a constraint matching no node is a surfaced error).

Predicates are built from two kinds of atom, composed with `~ & |`:

- **scalar** — one node. `has(dim)` (is the constraint dimensioned over `dim`?)
  and `name(cnsname)` (select a specific constraint by its linopy name, e.g.
  force `constraint_carbon_emissions_budget` into `top` regardless of its dims).
- **scattering** — fans the node into one subnode per label:
  `by_size(dim, n)` (label `idx // n`, e.g. weekly slices) and `group(dim)`
  (label = coordinate value). A scattering node expands `name → name[label]`;
  crossing two scatterers (`&`) gives `name[(l1, l2)]`. Numbering is local to the
  node group (`enumerate` its labels) — no global counter.
  Scalar `& ` scatter: the scalar **gates** (filters rows out), the scatter labels
  the rest.

The Linopy2Plasmo example (`top` = masters — no year *or* no spatial dim; one
`sub` per year, gated on having a spatial dim) becomes:

```python
Partition(
    {
        "top": ~has("set_time_steps_yearly") | ~has("set_nodes"),
        "sub": group("set_time_steps_yearly") & has("set_nodes"),
    }
)
```

(`group` splits by the coordinate value → one sub per year, exactly their
`enumerate([1,2,3])` loop. `by_size` would be used instead to bucket a fine
dimension into slices, e.g. `by_size("set_time_steps_operation", 168)` for weeks
— the generalization `group` doesn't cover.)

**Constraints partition; variables overlap.** The partition above defines only
the *constraint* → node map. A variable belongs to *every* node whose
constraints reference it (derived by scanning terms, as Linopy2Plasmo's
`addToVarSubDic!` does). A variable in >1 node is a **linking variable** →
Julia adds `var@A == var@B` equality links and Benders cuts on them. So overlap
is expected and correct for variables even though constraints are disjoint.

Ships: `has`, `group`, `by_size` + `~ & |`. Escape hatch: a `sub` value may be a
callable `constraint → label` for arbitrary keys — no separate imperative API.

*Later (not now):* an explicit variable-side partition, for when the derived
membership isn't the desired linking structure. Additive; overlaps allowed there
by design.

## Implementation plan (hand-off)

Steps 0-4 are **done** — the module builds the graph, solves via Benders, and
matches a monolithic HiGHS solve of `testProblem.nc` exactly (see *Running*
below). Step 5 (regression against the Julia reference) is the remaining work.
The subsections below now describe what was actually built, not a sketch —
kept as a map from the design decisions above to the real files/names, for
whoever tackles Step 5 or extends the module.

### Files

```
linopy/contrib/
  __init__.py
  plasmo/
    __init__.py            # public API: PlasmoModel, Partition, has/name/group/by_size,
                            # topologies (flat/manual), optimize()/benders()/solve_benders()
    partition.py           # partition algebra (Predicate tree, Partition.assign)
    build.py                # Plan.from_model (membership/links) + Plan.iter_blocks (CSR streaming)
    topology.py             # subgraph nesting (flat / manual tree)
    LinopyPlasmo.jl         # Julia: GraphBuilder -- streamed blocks -> OptiGraph -> Benders/optimize
experiment/
  run_benders.py           # drives linopy.contrib.plasmo on testProblem.nc, cross-checks HiGHS
  juliapkg.json            # pins JuMP/Plasmo/PlasmoBenders/HiGHS (this step; see below)
```

Note the module lives one level deeper than originally sketched
(`linopy/contrib/plasmo/` is a package, not a single `plasmo.py`) — the
partition algebra, block-building, and topology concerns each earned their own
file as the design solidified.

### Step 0 — Julia deps, reproducibly (done)

`experiment/juliapkg.json` declares `JuMP`, `Plasmo`, `PlasmoBenders`, `HiGHS`
(open solver) with version bounds matching what's resolved in this
experiment's env. `pyjuliapkg` (the dependency-resolution half of
`juliacall`/`PythonCall.jl`) discovers a project's Julia dependencies by
scanning `juliapkg.json` files it finds via `sys.path`: its own bundled file,
`<path>/juliapkg.json` and `<path>/<subdir>/juliapkg.json` for every entry on
`sys.path` (including `''`, i.e. the current working directory when the
interpreter starts), plus one subdir under any `pip install -e` mapping. All
matching files are merged before `Pkg.resolve()` runs once, lazily, on first
`import juliacall`.

This file was placed at `experiment/juliapkg.json` rather than under
`linopy/contrib/` deliberately: `deps_files()`'s editable-install branch only
looks *one* subdirectory below the mapped package root (`linopy/`), so a file
under `linopy/contrib/plasmo/` (two levels down) would silently never be
found. `experiment/juliapkg.json` is picked up via the plain cwd/`sys.path`
branch instead, since `run_benders.py` is always run with `experiment/` as the
working directory (`pixi run python run_benders.py`). The trade-off: this pin
only takes effect for scripts run from `experiment/`, not for `linopy.contrib.plasmo`
imported from an arbitrary cwd. Fine for now since the module's only consumer
is this experiment; revisit if `linopy.contrib.plasmo` gets a non-experimental
Julia-dependent consumer elsewhere.

Verify: `pixi run python -c "from juliacall import Main as jl; jl.seval('using JuMP, Plasmo, PlasmoBenders, HiGHS')"`.

### Step 1 — Partition algebra (`plasmo/partition.py`) (done)

Predicate expression tree over constraint dims, evaluated axis-separably (see
the module's docstring for the rectangle-selection argument for why `~`/`|`
reject scattering predicates). Public atoms: `has`, `name`, `group`, `by_size`,
composed with `~ & |`. `Partition.assign(model)` returns `node_of_cns` (int per
CSR row, aligned to `model.constraints.label_index.clabels`) and the ordered
`node_keys` list (`node_keys[0]` is the Benders master).

- **Test:** on `testProblem.nc`, the Linopy2Plasmo example partition (`~has(year)|
  ~has(nodes)` / `group(year) & has(nodes)`) yields 1 master + 3 subs; assert the
  node count and that every constraint is assigned exactly once. (Still to be
  written as an automated test -- verified manually via `run_benders.py` so far.)

### Step 2 — Matrix + membership (`plasmo/build.py`) (done)

`Plan.from_model` builds the node×variable incidence (`M`/`Mc`), the per-node
membership, linking-variable owners, and the cross-node `Links`, all vectorized
over `model.matrices`' CSR arrays -- no per-node Python loop. `Plan.iter_blocks`
then *streams* one `NodeBlock` (CSR row-slice + column remap) at a time, so
peak Python memory is one block, not all of them; the caller (`PlasmoModel._build`
in `plasmo/__init__.py`) consumes and drops each block before the next is built.

- **Test:** membership of a node == `np.unique` of its row-slice columns; sum of
  per-node var counts minus linking overlaps is consistent; objective coeffs
  each assigned to exactly one node (owner). Also still to be written as an
  automated test.

### Step 3 — `LinopyPlasmo.jl`: arrays → OptiGraph (done)

A `GraphBuilder` (mutable struct holding the growing `OptiGraph`, its
subgraphs, and each node's `NodeVariableRef` vector) accumulates blocks one at
a time via `add_block!`, in parent-first order so a nested block's parent
subgraph already exists (`topology.py` decides nesting; `flat()` -- everything
sibling under the root -- is what `benders()` requires). Constraints are built
by walking each block's CSR rows and adding one scalar affine constraint via
the function-form `add_constraint`/`build_constraint` (no macro, no
`ConstraintRef`s stored -- see the design notes above for why the macro/
vectorized forms don't work on an `OptiNode`). `add_links!` adds the star
equality links on the root graph; `finalize!` rolls per-node objectives up
through every subgraph via `set_to_node_objectives`.

- **Test:** build a small partition and assert `num_constraints`/`num_variables`
  per node and link count against what `Plan` computed in Python. Not yet
  automated.

### Step 4 — Driver + solve (`plasmo/__init__.py` + `run_benders.py`) (done)

`PlasmoModel` wraps a `Plan` + `Topology`, builds the Julia graph lazily on
first use, and exposes `optimize()` (monolithic solve of the whole graph) and
`benders()` (requires a flat topology) as free functions over it --
`solve_benders()` is the one-call convenience wrapper. Read-back
(`PlasmoModel.result()`) asks Julia for each node's full value vector at once
(not a per-variable round-trip) and scatters by that node's global variable
labels into a dense primal array indexed by linopy label.

- **Acceptance (met):** `pixi run python run_benders.py` on `testProblem.nc` runs
  Benders to convergence and matches a monolithic HiGHS solve of the same model
  to the printed tolerance (observed: exact match, `2360.09` both sides,
  relative difference `0.00e+00`).

### Step 5 — Regression against Linopy2Plasmo.jl (the reference oracle)

Verified this session: Linopy2Plasmo.jl loads **in the same juliacall process** as
`plasmo.py` — `Pkg.develop(path=".../Linopy2Plasmo.jl"); using Linopy2Plasmo`,
and its `pyimport("linopy")` resolves to the same Python. So the reference can be
driven from `run_benders.py` and compared directly, at three levels (cheapest
first — each catches a different class of bug):

1. **Node assignment.** Reference: `structureIntoNodes!(obj, split_tup)` fills
   `obj.varNode` (var label → node tuples) and `obj.nodes`. Ours: the incidence
   `M` (step 2). Both key on the **same linopy integer `labels`**, so compare
   `set(our_nodes[label]) == set(ref_nodes[label])` per variable, after aligning
   their tuple-nodes to our named nodes via the partition. Catches
   partition/membership bugs before any solve. *This is the highest-value check —
   it's exactly the logic we rewrote from Julia DataFrames to numpy incidence.*
2. **Graph structure.** After `createOptProblem!(obj, def_dic)`: per-node
   variable/constraint counts and total linking-constraint count vs. our
   `helpers.jl` graph.
3. **Solution.** Both run `BendersAlgorithm`/`run_algorithm!` with HiGHS; compare
   objective (and, sampled, variable values). Redundant with the monolithic gate
   on the objective, but confirms the *decomposition* agrees, not just the LP.

Notes for whoever implements this:
- Build the equivalent `split_tup` + `def_dic` for the reference to match our
  example `Partition` (year/nodes split). The reference needs the ordered
  `split_tup` of set names and the `def_dic` filter lambdas; derive them from the
  same partition spec so both sides decompose identically.
- The reference reads the `.nc` **itself** (its own `lin2plasObj`), so it does an
  independent extraction — a real cross-check of our `matrices`-based path, not a
  shared-code tautology.
- Gurobi is only a dep, not load-required; HiGHS (open) drives the solve. No
  license needed.
- Keep this as a `pytest`-skippable regression (`@pytest.mark.skipif` if
  Linopy2Plasmo.jl isn't dev'd) so the core suite doesn't hard-depend on the
  reference repo being present.

### Out of scope (later)

- Explicit variable-side partition (overlaps by design).
- Binary/integer variables — PlasmoBenders itself supports MIPs; the gap is
  ours (`build.py`/`LinopyPlasmo.jl` never declare integer/binary variables).
- Promoting `LinopyPlasmo.jl` to a Julia package if it outgrows one file.

## Files

- `testProblem.nc` — the linopy model to decompose.
- `pixi.toml` — env: python, editable linopy (`../`), juliacall, pyjuliapkg.
- `juliapkg.json` — pins the Julia side (`JuMP`/`Plasmo`/`PlasmoBenders`/`HiGHS`);
  see Step 0 above for how `pyjuliapkg` finds it.

## Running

```sh
pixi run python run_benders.py
```

## Further reading

For a narrative walkthrough (not this design log), see linopy's own docs:
`doc/plasmo-benders.rst` and the companion notebook
`examples/plasmo-benders-decomposition.ipynb`.
