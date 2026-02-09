"""
Quadratic Constraints Example
"""

import linopy

m = linopy.Model()

# Variables
x = m.add_variables(lower=0, upper=10, name="x")
y = m.add_variables(lower=0, upper=10, name="y")

# Linear constraint
m.add_constraints(x + y <= 8, name="linear_budget")

# Quadratic constraints
m.add_quadratic_constraints(x * x + y * y, "<=", 25, name="circle")
m.add_quadratic_constraints(x * y, "<=", 10, name="mixed_term")

# Objective: maximize x + 2y
m.add_objective(x + 2 * y, sense="max")

# Solve
m.solve(solver_name="gurobi")

# Results
print(f"x = {x.solution.values.item():.4f}")
print(f"y = {y.solution.values.item():.4f}")
print(f"Objective = {m.objective.value:.4f}")
