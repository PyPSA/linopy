using JuMP
using Gurobi
using DataFrames
using CSV
using Dates
using Random
Random.seed!(125)

function basic_model(n, solver)
    m = Model(solver)
    @variable(m, x[1:n, 1:n])
    @variable(m, y[1:n, 1:n])
    @constraint(m, x - y .>= 0:(n-1))
    @constraint(m, x + y .>= 0)
    @objective(m, Min, 2 * sum(x) + sum(y))
    optimize!(m)
    return objective_value(m)
end

function knapsack_model(n, solver)
    m = Model(solver)
    @variable(m, x[1:n], Bin)
    weight = rand(1:100, n)
    value = rand(1:100, n)
    @constraint(m, weight' * x <= 200)
    @objective(m, Max, value' * x)
    optimize!(m)
    return objective_value(m)
end


if snakemake.config["solver"] == "gurobi"
    solver = Gurobi.Optimizer
elseif snakemake.config["solver"] == "cbc"
    using Cbc
    solver = Cbc.Optimizer
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
