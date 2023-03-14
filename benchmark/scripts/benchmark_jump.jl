using JuMP
using Gurobi
using DataFrames
using CSV
using Dates
using Random
Random.seed!(125)

function basic_model(n, solver)
    m = Model(Gurobi.Optimizer)
    N = 1:n
    M = 1:n
    @variable(m, x[N, M])
    @variable(m, y[N, M])
    @constraint(m, [i=N, j=M], x[i, j] - y[i, j] >= i-1)
    @constraint(m, [i=N, j=M], x[i, j] + y[i, j] >= 0)
    @objective(m, Min, sum(2 * x[i, j] + y[i, j] for i in N, j in M))
    optimize!(m)
    return objective_value(m)
end

function knapsack_model(n, solver)
    m = Model(solver)
    N = 1:n
    @variable(m, x[N], Bin)
    weight = rand(1:100, n)
    value = rand(1:100, n)
    @constraint(m, sum(weight[i] * x[i] for i in N) <= 200)
    @objective(m, Max, sum(value[i] * x[i] for i in N))
    optimize!(m)
    return objective_value(m)
end


if snakemake.config["solver"] == "gurobi"
    solver = Gurobi.Optimizer
elseif snakemake.config["solver"] == "cbc"
    using CBC
    solver = CBC.Optimizer
end

if snakemake.config["benchmark"] == "basic"
    model = basic_model
elseif snakemake.config["benchmark"] == "knapsack"
    model = knapsack_model
end

# jit compile everything
model(1, solver)

profile = DataFrame(N=Int[], Time=Float64[], Memory=Float64[], Objective=Float64[])

for N in snakemake.params[1]
    mem = @allocated(model(N, solver))/10^6
    time = @elapsed(model(N, solver))
    objective = model(N, solver)
    push!(profile, [N, time, mem, objective])
end
profile[!, :API] .= "jump"
insertcols!(profile, 1, :Row => 1:nrow(profile))

CSV.write(snakemake.output[1], profile)
