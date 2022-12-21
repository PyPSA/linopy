import pandas as pd

from linopy import Model

m = Model()

lower = pd.Series(0, range(10))
upper = pd.DataFrame(10, range(10), range(10))

u = m.add_variables(0, upper, name="u")
v = m.add_variables(lower, upper, name="v")
x = m.add_variables(lower, 10, coords=[lower.index], name="x")
y = m.add_variables(0, 10, name="y")
z = m.add_variables(name="z", binary=True)
a = m.add_variables(coords=[lower.index], name="a", binary=True)


def test_variable_repr_u():
    repr(u)


def test_variable_repr_v():
    repr(v)


def test_variable_repr_x():
    repr(x)


def test_variable_repr_y():
    repr(y)


def test_variable_repr_z():
    repr(z)


def test_variable_repr_a():
    repr(a)


def test_linear_expression_u():
    repr(u.to_linexpr())


def test_linear_expression_v():
    repr(v.to_linexpr())


def test_linear_expression_x():
    repr(x.to_linexpr())


def test_linear_expression_y():
    repr(y.to_linexpr())


def test_constraint_u():
    repr(u >= 0)


def test_constraint_v():
    repr(v >= 0)


def test_constraint_x():
    repr(x >= 0)


def test_constraint_y():
    repr(y >= 0)
