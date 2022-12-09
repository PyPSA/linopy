from functools import partialmethod, update_wrapper

from xarray import DataArray

from linopy import expressions, variables


def monkey_patch(cls, pass_unpatched_method=False):
    def deco(func):
        func_name = func.__name__
        wrapped = getattr(cls, func_name)
        update_wrapper(func, wrapped)
        if pass_unpatched_method:
            func = partialmethod(func, unpatched_method=wrapped)
        setattr(cls, func_name, func)
        return func

    return deco


@monkey_patch(DataArray, pass_unpatched_method=True)
def __mul__(da, other, unpatched_method):
    if isinstance(other, (variables.Variable, expressions.LinearExpression)):
        return NotImplemented
    return unpatched_method(da, other)
