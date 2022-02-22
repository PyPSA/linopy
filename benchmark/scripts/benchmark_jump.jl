using JuMP
using DataFrames
using CSV
using Dates

function model(n, solver, integerlabel=false)
    m = Model(Gurobi.Optimizer)
    N = 1:n
    M = 1:n
    if !integerlabel
        N = float.(N)
        M = string.(M)
    end
    @variable(m, x[N, M])
    @variable(m, y[N, M])
    @constraint(m, [i=N, j=M], x[i, j] - y[i, j] >= i-1)
    @constraint(m, [i=N, j=M], x[i, j] + y[i, j] >= 0)
    @objective(m, Min, sum(2 * x[i, j] + y[i, j] for i in N, j in M))
    optimize!(m)
    return m
end

if snakemake.wildcards["solver"] == "gurobi"
    using Gurobi
    solver = Gurobi.Optimizer
elseif snakemake.wildcards["solver"] == "cbc"
    using CBC
    solver = CBC.Optimizer
end

# jit compile everything
model(1, solver)

profile = DataFrame(N=Int[], Time=Float64[], Memory=Float64[])

for N in snakemake.params[1]
    time = @elapsed(model(N, solver))
    mem = @allocated(model(N, solver))/10^6
    push!(profile, [N, time, mem])
end
profile[:API] = "jump"
insertcols!(profile, 1, :Row => 1:nrow(profile))

CSV.write(snakemake.output[1], profile)
