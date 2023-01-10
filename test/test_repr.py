import pandas as pd
import xarray as xr

from linopy import Model

m = Model()

lower = pd.Series(0, range(10))
upper = pd.DataFrame(10, range(10), range(10))
types = pd.Index(list("abcdefgh"), name="types")

u = m.add_variables(0, upper, name="u")
v = m.add_variables(lower, upper, name="v")
x = m.add_variables(lower, 10, coords=[lower.index], name="x")
y = m.add_variables(0, 10, name="y")
z = m.add_variables(name="z", binary=True)
a = m.add_variables(coords=[lower.index], name="a", binary=True)
b = m.add_variables(coords=[lower.index], name="b", integer=True)
c_mask = xr.DataArray(False, coords=upper.axes)
c_mask[:, 5:] = True
c = m.add_variables(lower, upper, name="c", mask=c_mask)
d = m.add_variables(0, 10, coords=[types], name="d")


# create linear expression for each variable
lu = 1 * u
lv = 1 * v
lx = 1 * x
ly = 1 * y
lz = 1 * z
la = 1 * a
lb = 1 * b
lc = 1 * c
ld = 1 * d
lav = 1 * a + 1 * v


# create anonymous constraint for linear expression
cu_ = lu >= 0
cv_ = lv >= 0
cx_ = lx >= 0
cy_ = ly >= 0
cz_ = lz >= 0
ca_ = la >= 0
cb_ = lb >= 0
cc_ = lc >= 0
cd_ = ld >= 0
cav_ = lav >= 0


# add constraint for each variable
cu = m.add_constraints(cu_, name="cu")
cv = m.add_constraints(cv_, name="cv")
cx = m.add_constraints(cx_, name="cx")
cy = m.add_constraints(cy_, name="cy")
cz = m.add_constraints(cz_, name="cz")
ca = m.add_constraints(ca_, name="ca")
cb = m.add_constraints(cb_, name="cb")
cc = m.add_constraints(cc_, name="cc")
cd = m.add_constraints(cd_, name="cd")
cav = m.add_constraints(cav_, name="cav")


def test_variable_repr():
    for var in [u, v, x, y, z, a, b, c, d]:
        repr(var)


def test_scalar_variable_repr():
    repr(u[0, 0])


def test_linear_expression_repr():
    for expr in [lu, lv, lx, ly, lz, la, lb, lc, ld, lav]:
        repr(expr)


def test_linear_expression_long():
    repr(x.sum())


def test_scalar_linear_expression_repr():
    repr(1 * u[0, 0])


def test_anonymous_constraint_repr():
    for con in [cu_, cv_, cx_, cy_, cz_, ca_, cb_, cc_, cd_, cav_]:
        repr(con)


def test_scalar_constraint_repr():
    repr(1 * u[0, 0] >= 0)


def test_constraint_repr():
    for con in [cu, cv, cx, cy, cz, ca, cb, cc, cd, cav]:
        repr(con)
