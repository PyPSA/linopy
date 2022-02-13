using JuMP
using Gurobi
using DataFrames
using CSV

function model(N)
    m = Model(Gurobi.Optimizer)
    @variable(m, x[1:N, 1:N])
    @variable(m, y[1:N, 1:N])
    @constraint(m, [i=1:N, j=1:N], x[i, j] - y[i, j] >= i-1)
    @constraint(m, [i=1:N, j=1:N], x[i, j] + y[i, j] >= 0)
    @objective(m, Min, sum(2 * x[i, j] + y[i, j] for i in 1:N, j in 1:N))
    optimize!(m)
    return m
end

model(1)

profile = DataFrame(N=Int[], Time=Float64[], Memory=Float64[])
for N in snakemake.params[1]
    time = @elapsed(model(N))
    mem = @allocated(model(N))/10^6
    push!(profile, [N, time, mem])
end

CSV.write(snakemake.output[1], profile)
