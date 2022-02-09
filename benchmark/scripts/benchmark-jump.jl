using JuMP

function create_model(N)
    m = Model()
    @variable(m, x[1:N, 1:N])
    @variable(m, y[1:N, 1:N])
    @constraint(m, [i=1:N, j=1:N], x[i, j] - y[i, j] >= i)
    @objective(m, Min, sum(2 * x[i, j] + y[i, j] for i in 1:N, j in 1:N))
    return m
end
