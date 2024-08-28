"""
This module contains examples of linear programming models using the linopy library.
"""

from numpy import arange

from linopy import Model


def simple_two_single_variables_model() -> Model:
    """
    Creates a simple linear programming model with two single variables.

    Returns
    -------
        Model: The created linear programming model.
    """
    m = Model()

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y >= 10)
    m.add_constraints(4 * x + 2 * y >= 3)

    m.add_objective(2 * y + x)
    return m


def simple_two_array_variables_model() -> Model:
    """
    Creates a simple linear programming model with two array variables.

    Returns
    -------
        Model: The created linear programming model.
    """
    m = Model()

    lower = [-10, -5]
    upper = [10, 15]
    x = m.add_variables(lower, upper, name="x")

    lower = [4, 0]
    upper = [8, 15]
    y = m.add_variables(lower, upper, name="y")

    m.add_constraints(2 * x + 2 * y >= 10)
    m.add_constraints(6 * x + 2 * y <= 100)

    m.add_objective(y + 2 * x)
    return m


def benchmark_model(n: int = 10, integerlabels: bool = False) -> Model:
    """
    Creates a benchmark linear programming model used in https://doi.org/10.21105/joss.04823.

    Args:
    ----
        n (int): The size of the benchmark models dimensions.
        integerlabels (bool, optional): Whether to use integer labels for variables.
        Defaults to False.

    Returns:
    -------
        Model: The created linear programming model.
    """
    m = Model()
    if integerlabels:
        naxis, maxis = [arange(n), arange(n)]
    else:
        naxis, maxis = [arange(n).astype(float), arange(n).astype(str)]
    x = m.add_variables(coords=[naxis, maxis])  # type: ignore
    y = m.add_variables(coords=[naxis, maxis])  # type: ignore
    m.add_constraints(x - y >= naxis)
    m.add_constraints(x + y >= 0)
    m.add_objective((2 * x).sum() + y.sum())
    return m
