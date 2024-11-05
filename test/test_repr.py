import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model, options

m = Model()

lower = pd.Series(0, range(10))
upper = pd.DataFrame(np.arange(10, 110).reshape(10, 10), range(10), range(10))
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

# new behavior in v0.2, variable with dimension name and other
# coordinates are added without a warning
e = m.add_variables(0, upper[5:], name="e")

f_mask = np.full_like(upper[:5], True, dtype=bool)
f_mask[:3] = False
f = m.add_variables(0, upper[5:], name="f", mask=f_mask)


multiindex = pd.MultiIndex.from_product(
    [list("asdfhjkg"), list("asdfghj")], names=["level_0", "level_1"]
)
g = m.add_variables(coords=[multiindex], name="g")

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
luc = 1 * v + 10
lq = x * x
lq2 = x * x + 1 * x
lq3 = x * x + 1 * x + 1 + 1 * y + 1 * z
lg = 1 * g

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
cuc_ = luc >= 0
cg_ = lg >= 0

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
cuc = m.add_constraints(cuc_, name="cuc")
cu_masked = m.add_constraints(cu_, name="cu_masked", mask=xr.full_like(u.labels, False))
cg = m.add_constraints(cg_, name="cg")

variables = [u, v, x, y, z, a, b, c, d, e, f, g]
expressions = [lu, lv, lx, ly, lz, la, lb, lc, ld, lav, luc, lq, lq2, lq3, lg]
anonymous_constraints = [cu_, cv_, cx_, cy_, cz_, ca_, cb_, cc_, cd_, cav_, cuc_, cg_]
constraints = [cu, cv, cx, cy, cz, ca, cb, cc, cd, cav, cuc, cu_masked, cg]


@pytest.mark.parametrize("var", variables)
def test_variable_repr(var):
    repr(var)


@pytest.mark.parametrize("var", variables)
def test_scalar_variable_repr(var):
    coord = tuple(var.indexes[c][0] for c in var.dims)
    repr(var.at[coord])


@pytest.mark.parametrize("var", variables)
def test_single_variable_repr(var):
    coord = tuple(var.indexes[c][0] for c in var.dims)
    repr(var.loc[coord])


@pytest.mark.parametrize("expr", expressions)
def test_linear_expression_repr(expr):
    repr(expr)


def test_linear_expression_long():
    repr(x.sum())


@pytest.mark.parametrize("var", variables)
def test_scalar_linear_expression_repr(var):
    coord = tuple(var.indexes[c][0] for c in var.dims)
    repr(1 * var.at[coord])


@pytest.mark.parametrize("var", variables)
def test_single_linear_repr(var):
    coord = tuple(var.indexes[c][0] for c in var.dims)
    repr(1 * var.loc[coord])


@pytest.mark.parametrize("var", variables)
def test_single_array_linear_repr(var):
    coord = {c: [var.indexes[c][0]] for c in var.dims}
    repr(1 * var.sel(coord))


@pytest.mark.parametrize("con", anonymous_constraints)
def test_anonymous_constraint_repr(con):
    repr(con)


def test_scalar_constraint_repr():
    repr(u.at[0, 0] >= 0)


@pytest.mark.parametrize("var", variables)
def test_single_constraint_repr(var):
    coord = tuple(var.indexes[c][0] for c in var.dims)
    repr(var.loc[coord] == 0)
    repr(1 * var.loc[coord] - var.loc[coord] == 0)


@pytest.mark.parametrize("var", variables)
def test_single_array_constraint_repr(var):
    coord = {c: [var.indexes[c][0]] for c in var.dims}
    repr(var.sel(coord) == 0)
    repr(1 * var.sel(coord) - var.sel(coord) == 0)


@pytest.mark.parametrize("con", constraints)
def test_constraint_repr(con):
    repr(con)


def test_empty_repr():
    repr(u.loc[[]])
    repr(lu.sel(dim_0=[]))
    repr(lu.sel(dim_0=[]) >= 0)


@pytest.mark.parametrize("obj", [v, lv, cv_, cv])
def test_print_options(obj):
    default_repr = repr(obj)
    with options as opts:
        opts.set_value(display_max_rows=20)
        longer_repr = repr(obj)
    assert len(default_repr) < len(longer_repr)

    longer_repr = obj.print(display_max_rows=20)


def test_print_labels():
    m.variables.print_labels([1, 2, 3])
    m.constraints.print_labels([1, 2, 3])
    m.constraints.print_labels([1, 2, 3], display_max_terms=10)


def test_label_position_too_high():
    with pytest.raises(ValueError):
        m.variables.print_labels([1000])


def test_model_repr_empty():
    repr(Model())
