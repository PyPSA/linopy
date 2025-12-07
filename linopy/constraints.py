"""
Linopy constraints module.

This module contains implementations for the Constraint{s} class.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Hashable, ItemsView, Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import (
    TYPE_CHECKING,
    Any,
    overload,
)

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse
import xarray as xr
from numpy import array, ndarray
from xarray import DataArray, Dataset
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates
from xarray.core.indexes import Indexes
from xarray.core.utils import Frozen

from linopy import expressions, variables
from linopy.common import (
    LabelPositionIndex,
    LocIndexer,
    align_lines_by_delimiter,
    assign_multiindex_safe,
    check_has_nulls,
    check_has_nulls_polars,
    filter_nulls_polars,
    format_string_as_variable_name,
    generate_indices_for_printout,
    get_dims_with_index_levels,
    get_label_position,
    group_terms_polars,
    has_optimized_model,
    infer_schema_polars,
    is_constant,
    iterate_slices,
    maybe_replace_signs,
    print_coord,
    print_single_constraint,
    print_single_expression,
    replace_by_map,
    save_join,
    to_dataframe,
    to_polars,
)
from linopy.config import options
from linopy.constants import (
    EQUAL,
    GREATER_EQUAL,
    HELPER_DIMS,
    LESS_EQUAL,
    TERM_DIM,
    SIGNS_pretty,
)
from linopy.types import (
    ConstantLike,
    CoordsLike,
    ExpressionLike,
    SignLike,
    VariableLike,
)

if TYPE_CHECKING:
    from linopy.model import Model

FILL_VALUE = {"labels": -1, "rhs": np.nan, "coeffs": 0, "vars": -1, "sign": "="}


def conwrap(
    method: Callable, *default_args: Any, **new_default_kwargs: Any
) -> Callable:
    @functools.wraps(method)
    def _conwrap(con: Constraint, *args: Any, **kwargs: Any) -> Constraint:
        for k, v in new_default_kwargs.items():
            kwargs.setdefault(k, v)
        return con.__class__(
            method(con.data, *default_args, *args, **kwargs), con.model, con.name
        )

    _conwrap.__doc__ = (
        f"Wrapper for the xarray {method.__qualname__} function for linopy.Constraint"
    )
    if new_default_kwargs:
        _conwrap.__doc__ += f" with default arguments: {new_default_kwargs}"

    return _conwrap


def _con_unwrap(con: Constraint | Dataset) -> Dataset:
    return con.data if isinstance(con, Constraint) else con


class Constraint:
    """
    Projection to a single constraint in a model.

    The Constraint class is a subclass of xr.DataArray hence most xarray
    functions can be applied to it.
    """

    __slots__ = ("_data", "_model", "_assigned")

    _fill_value = FILL_VALUE

    def __init__(
        self,
        data: Dataset,
        model: Model,
        name: str = "",
        skip_broadcast: bool = False,
    ) -> None:
        """
        Initialize the Constraint.

        Parameters
        ----------
        labels : xarray.DataArray
            labels of the constraint.
        model : linopy.Model
            Underlying model.
        name : str
            Name of the constraint.
        """

        from linopy.model import Model

        if not isinstance(data, Dataset):
            raise ValueError(f"data must be a Dataset, got {type(data)}")

        if not isinstance(model, Model):
            raise ValueError(f"model must be a Model, got {type(model)}")

        # check that `labels`, `lower` and `upper`, `sign` and `mask` are in data
        for attr in ("coeffs", "vars", "sign", "rhs"):
            if attr not in data:
                raise ValueError(f"missing '{attr}' in data")

        data = data.assign_attrs(name=name)

        if not skip_broadcast:
            (data,) = xr.broadcast(data, exclude=[TERM_DIM])

        self._assigned = "labels" in data
        self._data = data
        self._model = model

    def __getitem__(
        self, selector: str | int | slice | list | tuple | dict
    ) -> Constraint:
        """
        Get selection from the constraint.
        This is a wrapper around the xarray __getitem__ method. It returns a
        new object with the selected data.
        """
        data = Dataset({k: self.data[k][selector] for k in self.data}, attrs=self.attrs)
        return self.__class__(data, self.model, self.name)

    @property
    def attrs(self) -> dict[str, Any]:
        """
        Get the attributes of the constraint.
        """
        return self.data.attrs

    @property
    def coords(self) -> DatasetCoordinates:
        """
        Get the coordinates of the constraint.
        """
        return self.data.coords

    @property
    def indexes(self) -> Indexes:
        """
        Get the indexes of the constraint.
        """
        return self.data.indexes

    @property
    def dims(self) -> Frozen[Hashable, int]:
        """
        Get the dimensions of the constraint.
        """
        return self.data.dims

    @property
    def sizes(self) -> Frozen[Hashable, int]:
        """
        Get the sizes of the constraint.
        """
        return self.data.sizes

    @property
    def nterm(self) -> int:
        """
        Get the number of terms in the constraint.
        """
        return self.lhs.nterm

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of the constraint.
        """
        return self.rhs.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the constraint.
        """
        return self.rhs.shape

    @property
    def size(self) -> int:
        """
        Get the size of the constraint.
        """
        return self.rhs.size

    @property
    def loc(self) -> LocIndexer:
        return LocIndexer(self)

    @property
    def data(self) -> Dataset:
        """
        Get the underlying DataArray.
        """
        return self._data

    @property
    def labels(self) -> DataArray:
        """
        Get the labels of the constraint.
        """
        return self.data.get("labels", DataArray([]))

    @property
    def model(self) -> Model:
        """
        Get the model of the constraint.
        """
        return self._model

    @property
    def name(self) -> str:
        """
        Return the name of the constraint.
        """
        return self.attrs["name"]

    @property
    def coord_dims(self) -> tuple[Hashable, ...]:
        return tuple(k for k in self.dims if k not in HELPER_DIMS)

    @property
    def coord_sizes(self) -> dict[Hashable, int]:
        return {k: v for k, v in self.sizes.items() if k not in HELPER_DIMS}

    @property
    def coord_names(self) -> list[str]:
        """
        Get the names of the coordinates.
        """
        return get_dims_with_index_levels(self.data, self.coord_dims)

    @property
    def is_assigned(self) -> bool:
        return self._assigned

    def __repr__(self) -> str:
        """
        Print the constraint arrays.
        """
        max_lines = options["display_max_rows"]
        dims = list(self.coord_sizes.keys())
        ndim = len(dims)
        dim_names = self.coord_names
        dim_sizes = list(self.coord_sizes.values())
        size = np.prod(dim_sizes)  # that the number of theoretical printouts
        masked_entries = (~self.mask).sum().values if self.mask is not None else 0
        lines = []

        header_string = f"{self.type} `{self.name}`" if self.name else f"{self.type}"

        if size > 1 or ndim > 0:
            for indices in generate_indices_for_printout(dim_sizes, max_lines):
                if indices is None:
                    lines.append("\t\t...")
                else:
                    coord = [
                        self.data.indexes[dims[i]][int(ind)]
                        for i, ind in enumerate(indices)
                    ]
                    if self.mask is None or self.mask.values[indices]:
                        expr = print_single_expression(
                            self.coeffs.values[indices],
                            self.vars.values[indices],
                            0,
                            self.model,
                        )
                        sign = SIGNS_pretty[self.sign.values[indices]]
                        rhs = self.rhs.values[indices]
                        line = print_coord(coord) + f": {expr} {sign} {rhs}"
                    else:
                        line = print_coord(coord) + ": None"
                    lines.append(line)
            lines = align_lines_by_delimiter(lines, list(SIGNS_pretty.values()))

            shape_str = ", ".join(f"{d}: {s}" for d, s in zip(dim_names, dim_sizes))
            mask_str = f" - {masked_entries} masked entries" if masked_entries else ""
            underscore = "-" * (len(shape_str) + len(mask_str) + len(header_string) + 4)
            lines.insert(0, f"{header_string} [{shape_str}]{mask_str}:\n{underscore}")
        elif size == 1:
            expr = print_single_expression(
                self.coeffs.values, self.vars.values, 0, self.model
            )
            lines.append(
                f"{header_string}\n{'-' * len(header_string)}\n{expr} {SIGNS_pretty[self.sign.item()]} {self.rhs.item()}"
            )
        else:
            lines.append(f"{header_string}\n{'-' * len(header_string)}\n<empty>")

        return "\n".join(lines)

    def print(self, display_max_rows: int = 20, display_max_terms: int = 20) -> None:
        """
        Print the linear expression.

        Parameters
        ----------
        display_max_rows : int
            Maximum number of rows to be displayed.
        display_max_terms : int
            Maximum number of terms to be displayed.
        """
        with options as opts:
            opts.set_value(
                display_max_rows=display_max_rows, display_max_terms=display_max_terms
            )
            print(self)

    def __contains__(self, value: Any) -> bool:
        return self.data.__contains__(value)

    @property
    def type(self) -> str:
        """
        Get the type of the constraint.
        """
        return "Constraint" if self.is_assigned else "Constraint (unassigned)"

    @property
    def range(self) -> tuple[int, int]:
        """
        Return the range of the constraint.
        """
        return self.data.attrs["label_range"]

    @property
    def term_dim(self) -> str:
        """
        Return the term dimension of the constraint.
        """
        return TERM_DIM

    @property
    def mask(self) -> DataArray | None:
        """
        Get the mask of the constraint.

        The mask indicates on which coordinates the constraint is enabled
        (True) and disabled (False).

        Returns
        -------
        xr.DataArray
        """
        if self.is_assigned:
            return (self.data.labels != FILL_VALUE["labels"]).astype(bool)
        return None

    @property
    def coeffs(self) -> DataArray:
        """
        Get the left-hand-side coefficients of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.data.coeffs

    @coeffs.setter
    def coeffs(self, value: ConstantLike) -> None:
        value = DataArray(value).broadcast_like(self.vars, exclude=[self.term_dim])
        self._data = assign_multiindex_safe(self.data, coeffs=value)

    @property
    def vars(self) -> DataArray:
        """
        Get the left-hand-side variables of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.data.vars

    @vars.setter
    def vars(self, value: variables.Variable | DataArray) -> None:
        if isinstance(value, variables.Variable):
            value = value.labels
        if not isinstance(value, DataArray):
            raise TypeError("Expected value to be of type DataArray or Variable")
        value = value.broadcast_like(self.coeffs, exclude=[self.term_dim])
        self._data = assign_multiindex_safe(self.data, vars=value)

    @property
    def lhs(self) -> expressions.LinearExpression:
        """
        Get the left-hand-side linear expression of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        data = self.data[["coeffs", "vars"]].rename({self.term_dim: TERM_DIM})
        return expressions.LinearExpression(data, self.model)

    @lhs.setter
    def lhs(self, value: ExpressionLike | VariableLike | ConstantLike) -> None:
        value = expressions.as_expression(
            value, self.model, coords=self.coords, dims=self.coord_dims
        )
        self._data = self.data.drop_vars(["coeffs", "vars"]).assign(
            coeffs=value.coeffs, vars=value.vars, rhs=self.rhs - value.const
        )

    @property
    def sign(self) -> DataArray:
        """
        Get the signs of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.data.sign

    @sign.setter
    @is_constant
    def sign(self, value: SignLike) -> None:
        value = maybe_replace_signs(DataArray(value)).broadcast_like(self.sign)
        self._data = assign_multiindex_safe(self.data, sign=value)

    @property
    def rhs(self) -> DataArray:
        """
        Get the right hand side constants of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.data.rhs

    @rhs.setter
    def rhs(self, value: ExpressionLike) -> None:
        value = expressions.as_expression(
            value, self.model, coords=self.coords, dims=self.coord_dims
        )
        self.lhs = self.lhs - value.reset_const()
        self._data = assign_multiindex_safe(self.data, rhs=value.const)

    @property
    @has_optimized_model
    def dual(self) -> DataArray:
        """
        Get the dual values of the constraint.

        The function raises an error in case no model is set as a
        reference or the model status is not okay.
        """
        if "dual" not in self.data:
            raise AttributeError(
                "Underlying is optimized but does not have dual values stored."
            )
        return self.data["dual"]

    @dual.setter
    def dual(self, value: ConstantLike) -> None:
        """
        Get the dual values of the constraint.
        """
        value = DataArray(value).broadcast_like(self.labels)
        self._data = assign_multiindex_safe(self.data, dual=value)

    @classmethod
    def from_rule(cls, model: Model, rule: Callable, coords: CoordsLike) -> Constraint:
        """
        Create a constraint from a rule and a set of coordinates.

        This functionality mirrors the assignment of constraints as done by
        Pyomo.


        Parameters
        ----------
        model : linopy.Model
            Passed to function `rule` as a first argument.
        rule : callable
            Function to be called for each combinations in `coords`.
            The first argument of the function is the underlying `linopy.Model`.
            The following arguments are given by the coordinates for accessing
            the variables. The function has to return a
            `AnonymousScalarConstraint`. Therefore use the direct getter when
            indexing variables in the linear expression.
        coords : coordinate-like
            Coordinates to processed by `xarray.DataArray`.
            For each combination of coordinates, the function given by `rule` is called.
            The order and size of coords has to be same as the argument list
            followed by `model` in function `rule`.


        Returns
        -------
        linopy.Constraint

        Examples
        --------
        >>> from linopy import Model, LinearExpression, Constraint
        >>> m = Model()
        >>> coords = pd.RangeIndex(10), ["a", "b"]
        >>> x = m.add_variables(0, 100, coords)
        >>> def bound(m, i, j):
        ...     if i % 2:
        ...         return (i - 1) * x.at[i - 1, j] >= 0
        ...     else:
        ...         return i * x.at[i, j] >= 0
        ...
        >>> con = Constraint.from_rule(m, bound, coords)
        >>> con = m.add_constraints(con)
        """
        if not isinstance(coords, DataArrayCoordinates):
            coords = DataArray(coords=coords).coords
        shape = list(map(len, coords.values()))

        # test output type
        output = rule(model, *[c.values[0] for c in coords.values()])
        if not isinstance(output, AnonymousScalarConstraint) and output is not None:
            msg = f"`rule` has to return AnonymousScalarConstraint not {type(output)}."
            raise TypeError(msg)

        combinations = product(*[c.values for c in coords.values()])
        placeholder_lhs = expressions.ScalarLinearExpression((np.nan,), (-1,), model)
        placeholder = AnonymousScalarConstraint(placeholder_lhs, "=", np.nan)
        cons = [rule(model, *coord) or placeholder for coord in combinations]
        exprs = [con.lhs for con in cons]

        lhs = expressions.LinearExpression._from_scalarexpression_list(
            exprs, coords, model
        )
        sign = DataArray(array([c.sign for c in cons]).reshape(shape), coords)
        rhs = DataArray(array([c.rhs for c in cons]).reshape(shape), coords)
        data = lhs.data.assign(sign=sign, rhs=rhs)
        return cls(data, model=model)

    @property
    def flat(self) -> pd.DataFrame:
        """
        Convert the constraint to a pandas DataFrame.

        The resulting DataFrame represents a long table format of the all
        non-masked constraints with non-zero coefficients. It contains the
        columns `labels`, `coeffs`, `vars`, `rhs`, `sign`.

        Returns
        -------
        df : pandas.DataFrame
        """
        ds = self.data

        def mask_func(data: pd.DataFrame) -> pd.Series:
            mask = (data["vars"] != -1) & (data["coeffs"] != 0)
            if "labels" in data:
                mask &= data["labels"] != -1
            return mask

        df = to_dataframe(ds, mask_func=mask_func)

        # Group repeated variables in the same constraint
        agg_custom = {k: "first" for k in list(df.columns)}
        agg_standards = dict(coeffs="sum", rhs="first", sign="first")
        agg = {**agg_custom, **agg_standards}
        df = df.groupby(["labels", "vars"], as_index=False).aggregate(agg)
        check_has_nulls(df, name=f"{self.type} {self.name}")
        return df

    def to_polars(self) -> pl.DataFrame:
        """
        Convert the constraint to a polars DataFrame.

        The resulting DataFrame represents a long table format of the all
        non-masked constraints with non-zero coefficients. It typically
        contains the columns `labels`, `coeffs`, `vars`, `rhs`, `sign`.

        Returns
        -------
        df : polars.DataFrame
        """
        ds = self.data

        keys = [k for k in ds if ("_term" in ds[k].dims) or (k == "labels")]
        long = to_polars(ds[keys])

        long = filter_nulls_polars(long)
        long = group_terms_polars(long)
        check_has_nulls_polars(long, name=f"{self.type} {self.name}")

        short_ds = ds[[k for k in ds if "_term" not in ds[k].dims]]
        schema = infer_schema_polars(short_ds)
        schema["sign"] = pl.Enum(["=", "<=", ">="])
        short = to_polars(short_ds, schema=schema)
        short = filter_nulls_polars(short)
        check_has_nulls_polars(short, name=f"{self.type} {self.name}")

        df = pl.concat([short, long], how="diagonal_relaxed").sort(["labels", "rhs"])
        # delete subsequent non-null rhs (happens is all vars per label are -1)
        is_non_null = df["rhs"].is_not_null()
        prev_non_is_null = is_non_null.shift(1).fill_null(False)
        df = df.filter(is_non_null & ~prev_non_is_null | ~is_non_null)
        return df[["labels", "coeffs", "vars", "sign", "rhs"]]

    # Wrapped function which would convert variable to dataarray
    assign = conwrap(Dataset.assign)

    assign_multiindex_safe = conwrap(assign_multiindex_safe)

    assign_attrs = conwrap(Dataset.assign_attrs)

    assign_coords = conwrap(Dataset.assign_coords)

    # bfill = conwrap(Dataset.bfill)

    broadcast_like = conwrap(Dataset.broadcast_like)

    chunk = conwrap(Dataset.chunk)

    drop_sel = conwrap(Dataset.drop_sel)

    drop_isel = conwrap(Dataset.drop_isel)

    expand_dims = conwrap(Dataset.expand_dims)

    # ffill = conwrap(Dataset.ffill)

    sel = conwrap(Dataset.sel)

    isel = conwrap(Dataset.isel)

    shift = conwrap(Dataset.shift)

    swap_dims = conwrap(Dataset.swap_dims)

    set_index = conwrap(Dataset.set_index)

    reindex = conwrap(Dataset.reindex, fill_value=_fill_value)

    reindex_like = conwrap(Dataset.reindex_like, fill_value=_fill_value)

    rename = conwrap(Dataset.rename)

    rename_dims = conwrap(Dataset.rename_dims)

    roll = conwrap(Dataset.roll)

    stack = conwrap(Dataset.stack)

    unstack = conwrap(Dataset.unstack)

    iterate_slices = iterate_slices


@dataclass(repr=False)
class Constraints:
    """
    A constraint container used for storing multiple constraint arrays.
    """

    data: dict[str, Constraint]
    model: Model
    _label_position_index: LabelPositionIndex | None = None

    dataset_attrs = ["labels", "coeffs", "vars", "sign", "rhs"]
    dataset_names = [
        "Labels",
        "Left-hand-side coefficients",
        "Left-hand-side variables",
        "Signs",
        "Right-hand-side constants",
    ]

    def _formatted_names(self) -> dict[str, str]:
        """
        Get a dictionary of formatted names to the proper constraint names.
        This map enables a attribute like accession of variable names which
        are not valid python variable names.
        """
        return {format_string_as_variable_name(n): n for n in self}

    def __repr__(self) -> str:
        """
        Return a string representation of the linopy model.
        """
        r = "linopy.model.Constraints"
        line = "-" * len(r)
        r += f"\n{line}\n"

        for name, ds in self.items():
            coords = (
                " (" + ", ".join([str(c) for c in ds.coords.keys()]) + ")"
                if ds.coords
                else ""
            )
            r += f" * {name}{coords}\n"
        if not len(list(self)):
            r += "<empty>\n"
        return r

    @overload
    def __getitem__(self, names: str) -> Constraint: ...

    @overload
    def __getitem__(self, names: list[str]) -> Constraints: ...

    def __getitem__(self, names: str | list[str]) -> Constraint | Constraints:
        if isinstance(names, str):
            return self.data[names]
        return Constraints({name: self.data[name] for name in names}, self.model)

    def __getattr__(self, name: str) -> Constraint:
        # If name is an attribute of self (including methods and properties), return that
        if name in self.data:
            return self.data[name]
        else:
            if name in (formatted_names := self._formatted_names()):
                return self.data[formatted_names[name]]
        raise AttributeError(
            f"Constraints has no attribute `{name}` or the attribute is not accessible, e.g. raises an error."
        )

    def __getstate__(self) -> dict:
        return self.__dict__

    def __setstate__(self, d: dict) -> None:
        self.__dict__.update(d)

    def __dir__(self) -> list[str]:
        base_attributes = list(super().__dir__())
        formatted_names = [
            n for n in self._formatted_names() if n not in base_attributes
        ]
        return base_attributes + formatted_names

    def __len__(self) -> int:
        return self.data.__len__()

    def __iter__(self) -> Iterator[str]:
        return self.data.__iter__()

    def items(self) -> ItemsView[str, Constraint]:
        return self.data.items()

    def _ipython_key_completions_(self) -> list[str]:
        """
        Provide method for the key-autocompletions in IPython.

        See
        http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
        For the details.
        """
        return list(self)

    def add(self, constraint: Constraint) -> None:
        """
        Add a constraint to the constraints constrainer.
        """
        self.data[constraint.name] = constraint
        self._invalidate_label_position_index()

    def remove(self, name: str) -> None:
        """
        Remove constraint `name` from the constraints.
        """
        self.data.pop(name)
        self._invalidate_label_position_index()

    def _invalidate_label_position_index(self) -> None:
        """Invalidate the label position index cache."""
        if self._label_position_index is not None:
            self._label_position_index.invalidate()

    @property
    def labels(self) -> Dataset:
        """
        Get the labels of all constraints.
        """
        return save_join(
            *[v.labels.rename(k) for k, v in self.items()],
            integer_dtype=True,
        )

    @property
    def coeffs(self) -> Dataset:
        """
        Get the coefficients of all constraints.
        """
        return save_join(*[v.coeffs.rename(k) for k, v in self.items()])

    @property
    def vars(self) -> Dataset:
        """
        Get the variables of all constraints.
        """

        def rename_term_dim(ds: DataArray) -> DataArray:
            return ds.rename({TERM_DIM: str(ds.name) + TERM_DIM})

        return save_join(
            *[rename_term_dim(v.vars.rename(k)) for k, v in self.items()],
            integer_dtype=True,
        )

    @property
    def sign(self) -> Dataset:
        """
        Get the signs of all constraints.
        """
        return save_join(*[v.sign.rename(k) for k, v in self.items()])

    @property
    def rhs(self) -> Dataset:
        """
        Get the right-hand-side constants of all constraints.
        """
        return save_join(*[v.rhs.rename(k) for k, v in self.items()])

    @property
    def dual(self) -> Dataset:
        """
        Get the dual values of all constraints.
        """
        try:
            return save_join(*[v.dual.rename(k) for k, v in self.items()])
        except AttributeError:
            return Dataset()

    @property
    def coefficientrange(self) -> pd.DataFrame:
        """
        Coefficient range of the constraint.
        """
        d = {
            k: [self[k].coeffs.min().item(), self[k].coeffs.max().item()] for k in self
        }
        return pd.DataFrame(d, index=["min", "max"]).T

    @property
    def ncons(self) -> int:
        """
        Get the number all constraints effectively used by the model.

        This excludes constraints with missing labels or where all variables
        are masked (vars == -1).
        """
        total = 0
        for con in self.data.values():
            labels = con.labels.values
            vars_arr = con.vars.values

            # Handle scalar constraint (single constraint, labels is 0-d)
            if labels.ndim == 0:
                # Scalar: valid if label != -1 and any var != -1
                if labels != -1 and (vars_arr != -1).any():
                    total += 1
                continue

            # Array constraint: labels has constraint dimensions, vars has
            # constraint dimensions + _term dimension
            valid_labels = labels != -1

            # Check if any variable in each constraint is valid (not -1)
            # vars has shape (..., n_terms) where ... matches labels shape
            has_valid_var = (vars_arr != -1).any(axis=-1)

            active = valid_labels & has_valid_var

            if con.mask is not None:
                active = active & con.mask.values

            total += int(active.sum())
        return total

    @property
    def inequalities(self) -> Constraints:
        """
        Get the subset of constraints which are purely inequalities.
        """
        return self[[n for n, s in self.items() if (s.sign != EQUAL).all()]]

    @property
    def equalities(self) -> Constraints:
        """
        Get the subset of constraints which are purely equalities.
        """
        return self[[n for n, s in self.items() if (s.sign == EQUAL).all()]]

    def sanitize_zeros(self) -> None:
        """
        Filter out terms with zero and close-to-zero coefficient.
        """
        for name in self:
            not_zero = abs(self[name].coeffs) > 1e-10
            con = self[name]
            con.vars = self[name].vars.where(not_zero, -1)
            con.coeffs = self[name].coeffs.where(not_zero)

    def sanitize_missings(self) -> None:
        """
        Set constraints labels to -1 where all variables in the lhs are
        missing.
        """
        for name in self:
            con = self[name]
            contains_non_missing = (con.vars != -1).any(con.term_dim)
            labels = self[name].labels.where(contains_non_missing, -1)
            con._data = assign_multiindex_safe(con.data, labels=labels)

    def sanitize_infinities(self) -> None:
        """
        Replace infinite values in the constraints with a large value.
        """
        for name in self:
            con = self[name]
            valid_infinity_values = ((con.sign == LESS_EQUAL) & (con.rhs == np.inf)) | (
                (con.sign == GREATER_EQUAL) & (con.rhs == -np.inf)
            )
            labels = con.labels.where(~valid_infinity_values, -1)
            con._data = assign_multiindex_safe(con.data, labels=labels)

    def get_name_by_label(self, label: int | float) -> str:
        """
        Get the constraint name of the constraint containing the passed label.

        Parameters
        ----------
        label : int
            Integer label within the range [0, MAX_LABEL] where MAX_LABEL is the last assigned
            constraint label.

        Raises
        ------
        ValueError
            If label is not contained by any constraint.

        Returns
        -------
        name : str
            Name of the containing constraint.
        """
        if not isinstance(label, float | int) or label < 0:
            raise ValueError("Label must be a positive number.")
        for name, ds in self.items():
            if label in ds.labels:
                return name
        raise ValueError(f"No constraint found containing the label {label}.")

    def get_label_position(
        self, values: int | ndarray
    ) -> (
        tuple[str, dict]
        | tuple[None, None]
        | list[tuple[str, dict] | tuple[None, None]]
        | list[list[tuple[str, dict] | tuple[None, None]]]
    ):
        """
        Get tuple of name and coordinate for constraint labels.

        Uses an optimized O(log n) binary search implementation with a cached index.
        """
        if self._label_position_index is None:
            self._label_position_index = LabelPositionIndex(self)
        return get_label_position(self, values, self._label_position_index)

    def print_labels(
        self, values: Sequence[int], display_max_terms: int | None = None
    ) -> None:
        """
        Print a selection of labels of the constraints.

        Parameters
        ----------
        values : list, array-like
            One dimensional array of constraint labels.
        """
        with options as opts:
            if display_max_terms is not None:
                opts.set_value(display_max_terms=display_max_terms)
            res = [print_single_constraint(self.model, v) for v in values]
        print("\n".join(res))

    def set_blocks(self, block_map: np.ndarray) -> None:
        """
        Get a dataset of same shape as constraints.labels with block values.

        Let N be the number of blocks.
        The following ciases are considered:

            * where are all vars are -1, the block is -1
            * where are all vars are 0, the block is 0
            * where all vars are n, the block is n
            * where vars are n or 0 (both present), the block is n
            * N+1 otherwise
        """
        N = block_map.max()

        for name, constraint in self.items():
            res = xr.full_like(constraint.labels, N + 1, dtype=block_map.dtype)
            entries = replace_by_map(constraint.vars, block_map)

            not_zero = entries != 0
            not_missing = entries != -1
            for n in range(N + 1):
                not_n = entries != n
                mask = not_n & not_zero & not_missing
                res = res.where(mask.any(constraint.term_dim), n)

            res = res.where(not_missing.any(constraint.term_dim), -1)
            res = res.where(not_zero.any(constraint.term_dim), 0)
            constraint._data = assign_multiindex_safe(constraint.data, blocks=res)

    @property
    def flat(self) -> pd.DataFrame:
        """
        Convert all constraint to a single pandas Dataframe.

        The resulting dataframe is a long format with columns
        `labels`, `coeffs`, `vars`, `rhs`, `sign`.

        Returns
        -------
        pd.DataFrame
        """
        dfs = [self[k].flat for k in self]
        if not len(dfs):
            return pd.DataFrame(columns=["coeffs", "vars", "labels", "key"])
        df = pd.concat(dfs, ignore_index=True)
        unique_labels = df.labels.unique()
        map_labels = pd.Series(np.arange(len(unique_labels)), index=unique_labels)
        df["key"] = df.labels.map(map_labels)
        return df

    def to_matrix(self, filter_missings: bool = True) -> scipy.sparse.csc_matrix:
        """
        Construct a constraint matrix in sparse format.

        Missing values, i.e. -1 in labels and vars, are ignored filtered
        out.
        """
        # TODO: rename "filter_missings" to "~labels_as_coordinates"
        cons = self.flat

        if not len(self):
            raise ValueError("No constraints available to convert to matrix.")

        if filter_missings:
            vars = self.model.variables.flat
            shape = (cons.key.max() + 1, vars.key.max() + 1)
            cons["vars"] = cons.vars.map(vars.set_index("labels").key)
            return scipy.sparse.csc_matrix(
                (cons.coeffs, (cons.key, cons.vars)), shape=shape
            )
        else:
            shape = self.model.shape
            return scipy.sparse.csc_matrix(
                (cons.coeffs, (cons.labels, cons.vars)), shape=shape
            )

    def reset_dual(self) -> None:
        """
        Reset the stored solution of variables.
        """
        for k, c in self.items():
            if "dual" in c:
                c._data = c.data.drop_vars("dual")


class AnonymousScalarConstraint:
    """
    Container for anonymous scalar constraint.

    This contains a left-hand-side (lhs), a sign and a right-hand-side
    (rhs) for exactly one constraint.
    """

    _lhs: expressions.ScalarLinearExpression
    _sign: str
    _rhs: int | float | np.floating | np.integer

    def __init__(
        self,
        lhs: expressions.ScalarLinearExpression,
        sign: str,
        rhs: int | float | np.floating | np.integer,
    ):
        """
        Initialize a anonymous scalar constraint.
        """
        if not isinstance(rhs, int | float | np.floating | np.integer):
            raise TypeError(f"Assigned rhs must be a constant, got {type(rhs)}).")
        self._lhs = lhs
        self._sign = sign
        self._rhs = rhs

    def __repr__(self) -> str:
        """
        Get the representation of the AnonymousScalarConstraint.
        """
        expr_string = print_single_expression(
            np.array(self.lhs.coeffs), np.array(self.lhs.vars), 0, self.lhs.model
        )
        return f"AnonymousScalarConstraint: {expr_string} {self.sign} {self.rhs}"

    @property
    def lhs(self) -> expressions.ScalarLinearExpression:
        """
        Get the left hand side of the constraint.
        """
        return self._lhs

    @property
    def sign(self) -> str:
        """
        Get the sign of the constraint.
        """
        return self._sign

    @property
    def rhs(self) -> int | float | np.floating | np.integer:
        """
        Get the right hand side of the constraint.
        """
        return self._rhs

    def to_constraint(self) -> Constraint:
        data = self.lhs.to_linexpr().data.assign(sign=self.sign, rhs=self.rhs)
        return Constraint(data=data, model=self.lhs.model)
