# SPDX-FileCopyrightText: Contributors to linopy <https://github.com/PyPSA/linopy>
#
# SPDX-License-Identifier: MIT
#
# Build a Plasmo OptiGraph incrementally from the per-node LP blocks produced by
# linopy/contrib/plasmo/build.py, then run a decomposition algorithm on it.
#
# The Python side streams one block at a time (`add_block`) so it never holds all
# blocks at once; each block becomes one subgraph containing one OptiNode. Blocks
# are attached into a subgraph *tree* -- a block's `parent` is the root (0) or the
# 1-based index of an already-added node whose subgraph should contain it. This
# lets the caller build flat or nested topologies.
#
# All index arrays arrive from Python (numpy) 0-based; converted to 1-based on
# entry. Constraints are built by walking each block's CSR rows and adding one
# scalar affine constraint per row via the function form `add_constraint` (no
# macro, no ConstraintRefs stored). The affine expression element type must be
# `GenericAffExpr{Float64, NodeVariableRef}` -- an OptiNode's variables are
# NodeVariableRef, so `AffExpr(0.0)` would error.
#
# Loaded once from Python via `jl.seval('include(".../LinopyPlasmo.jl")')`;
# reached as `jl.LinopyPlasmo.<fn>`. Wrapping in a module keeps JuMP/Plasmo and
# these names out of `Main`.

module LinopyPlasmo

using JuMP
using Plasmo
using PlasmoBenders

export GraphBuilder, new_builder, add_block!, add_links!, finalize!
export run_benders, run_optimize, node_values

"""
    GraphBuilder

Holds the growing OptiGraph while blocks stream in. `subgraphs[k]` is node `k`'s
subgraph, `xs[k]` its variable vector (NodeVariableRef), both 1-based.
"""
mutable struct GraphBuilder
    graph::OptiGraph
    subgraphs::Vector{OptiGraph}
    onodes::Vector{OptiNode}
    xs::Vector{Any}
end

new_builder() = GraphBuilder(OptiGraph(), OptiGraph[], OptiNode[], Any[])

"""
    add_block!(b, parent, indptr, colval, nzval, sense_b, sense, lb, ub, c)

Add one node's LP block. `parent` is 0 (attach the new subgraph under the root
graph) or the 1-based index of a previously added node (attach under that node's
subgraph, i.e. nest). All index arrays are 0-based. Returns the new node's index.
"""
function add_block!(
    b::GraphBuilder,
    parent::Integer,
    name::AbstractString,
    indptr::AbstractVector{<:Integer},
    colval::AbstractVector{<:Integer},
    nzval::AbstractVector{<:Real},
    rhs::AbstractVector{<:Real},
    sense::AbstractVector,
    lb::AbstractVector{<:Real},
    ub::AbstractVector{<:Real},
    c::AbstractVector{<:Real},
)
    par = parent == 0 ? b.graph : b.subgraphs[parent]
    sym = Symbol(name)
    sg = add_subgraph(par; name=sym)
    node = add_node(sg; label=sym)
    ncols = length(lb)

    @variable(node, x[1:ncols])
    for j in 1:ncols
        isfinite(lb[j]) && set_lower_bound(x[j], lb[j])
        isfinite(ub[j]) && set_upper_bound(x[j], ub[j])
    end

    # constraints: walk CSR rows, one scalar affine constraint per row
    T = GenericAffExpr{Float64,eltype(x)}
    for i in 1:length(rhs)
        expr = zero(T)
        # 0-based indptr: row i spans indptr[i]+1 .. indptr[i+1] (1-based)
        for p in (indptr[i] + 1):indptr[i + 1]
            add_to_expression!(expr, nzval[p], x[colval[p] + 1])
        end
        s = sense[i]
        set = s == "<" ? MOI.LessThan(rhs[i]) :
              s == ">" ? MOI.GreaterThan(rhs[i]) :
              MOI.EqualTo(rhs[i])
        add_constraint(node, build_constraint(error, expr, set))
    end

    # per-node objective: sum of owned coeffs * their variable
    obj = zero(T)
    for j in 1:ncols
        iszero(c[j]) || add_to_expression!(obj, c[j], x[j])
    end
    @objective(node, Min, obj)

    push!(b.subgraphs, sg)
    push!(b.onodes, node)
    push!(b.xs, x)
    return length(b.subgraphs)
end

"""
    add_links!(b, owner, owner_col, other, other_col)

Add equality links `x[owner] == x[other]` on the root graph. All 0-based.
"""
function add_links!(
    b::GraphBuilder,
    owner::AbstractVector{<:Integer},
    owner_col::AbstractVector{<:Integer},
    other::AbstractVector{<:Integer},
    other_col::AbstractVector{<:Integer},
)
    for l in 1:length(owner)
        xo = b.xs[owner[l] + 1][owner_col[l] + 1]
        xt = b.xs[other[l] + 1][other_col[l] + 1]
        @linkconstraint(b.graph, xo == xt)
    end
    return nothing
end

"""
    finalize!(b)

Roll node objectives up into every subgraph (recursively) and the root graph.
Without the per-subgraph roll-up, algorithms see feasibility-sense subgraphs.
Returns the root graph.
"""
function finalize!(b::GraphBuilder)
    for sg in all_subgraphs(b.graph)
        set_to_node_objectives(sg)
    end
    set_to_node_objectives(b.graph)
    return b.graph
end

"""
    node_values(result, b, node)

The solution values of every variable of 0-based `node`, as a `Vector{Float64}`
in local-column order. `result` is a `BendersAlgorithm` (Benders) or the graph
(after `run_optimize`) -- `Plasmo.value` (itself a `JuMP.value` extension) already
dispatches on either. Retrieving a whole node at once keeps read-back off the
Python per-variable loop.
"""
function node_values(result, b::GraphBuilder, node::Integer)
    return Float64[Plasmo.value(result, x) for x in b.xs[node + 1]]
end

# -- algorithms ---------------------------------------------------------------

"""
    run_benders(graph, master; solver, kwargs...)

PlasmoBenders on `graph` with `master` the top subgraph. `kwargs` are passed
through to `BendersAlgorithm` verbatim (e.g. `max_iters`, `tol`, `regularize`,
`add_slacks`, ...; see `BendersAlgorithm`'s own docstring for the full list),
with `add_slacks=true` and `regularize=true` as defaults -- callers can
override either via `kwargs`. Returns the algorithm object (query values via
`node_values(alg, b, node)`).
"""
function run_benders(graph, master; solver, kwargs...)
    alg = BendersAlgorithm(
        graph,
        master;
        solver=solver,
        add_slacks=true,
        regularize=true,
        kwargs...,
    )
    run_algorithm!(alg)
    return alg
end

"""
    run_optimize(graph; solver)

Solve the whole graph monolithically (all subgraphs, links enforced). Query
values afterwards via `node_values(graph, b, node)`.
"""
function run_optimize(graph; solver)
    set_optimizer(graph, solver)
    optimize!(graph)
    return graph
end

# juliacall-friendly aliases: Python attribute access can't reach `!` names.
const add_block_b = add_block!
const add_links_b = add_links!
const finalize_b = finalize!

end # module LinopyPlasmo
