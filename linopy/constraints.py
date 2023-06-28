# -*- coding: utf-8 -*-
"""
Linopy constraints module.

This module contains implementations for the Constraint{s} class.
"""

import functools
import re
import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Sequence, Union

import dask
import numpy as np
import pandas as pd
import xarray as xr
from deprecation import deprecated
from numpy import arange, array
from scipy.sparse import csc_matrix
from xarray import DataArray, Dataset

from linopy import expressions, variables
from linopy.common import (
    LocIndexer,
    align_lines_by_delimiter,
    forward_as_properties,
    generate_indices_for_printout,
    get_label_position,
    has_optimized_model,
    is_constant,
    maybe_replace_signs,
    print_coord,
    print_single_constraint,
    print_single_expression,
    replace_by_map,
    save_join,
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


def conwrap(method, *default_args, **new_default_kwargs):
    @functools.wraps(method)
    def _conwrap(con, *args, **kwargs):
        for k, v in new_default_kwargs.items():
            kwargs.setdefault(k, v)
        return con.__class__(
            method(con.data, *default_args, *args, **kwargs), con.model, con.name
        )

    _conwrap.__doc__ = f"Wrapper for the xarray {method} function for linopy.Constraint"
    if new_default_kwargs:
        _conwrap.__doc__ += f" with default arguments: {new_default_kwargs}"

    return _conwrap


def _con_unwrap(con):
    return con.data if isinstance(con, Constraint) else con


@forward_as_properties(
    data=[
        "attrs",
        "coords",
        "indexes",
        "dims",
        "sizes",
    ],
    labels=["values"],
    lhs=["nterm"],
    rhs=["ndim", "shape", "size"],
)
class Constraint:
    """
    Projection to a single constraint in a model.

    The Constraint class is a subclass of xr.DataArray hence most xarray
    functions can be applied to it.
    """

    __slots__ = ("_data", "_model", "_assigned")

    def __init__(self, data: Dataset, model: Any, name: str = ""):
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

        (data,) = xr.broadcast(data, exclude=[TERM_DIM])

        self._data = data
        self._model = model

    @property
    def data(self):
        """
        Get the underlying DataArray.
        """
        return self._data

    @property
    def labels(self):
        """
        Get the labels of the constraint.
        """
        return self.data.labels if self.is_assigned else None

    @property
    def model(self):
        """
        Get the model of the constraint.
        """
        return self._model

    @property
    def name(self):
        """
        Return the name of the constraint.
        """
        return self.attrs["name"]

    @property
    def coord_dims(self):
        return {k: self.data.dims[k] for k in self.dims if k not in HELPER_DIMS}

    @property
    def is_assigned(self):
        return "labels" in self.data

    def __repr__(self):
        """
        Print the constraint arrays.
        """
        max_lines = options["display_max_rows"]
        dims = list(self.dims)
        dim_sizes = list(self.sizes.values())[:-1]
        size = np.prod(dim_sizes)  # that the number of theoretical printouts
        masked_entries = self.mask.sum().values if self.mask is not None else 0
        lines = []

        header_string = f"{self.type} `{self.name}`" if self.name else f"{self.type}"

        if size > 1:
            for indices in generate_indices_for_printout(dim_sizes, max_lines):
                if indices is None:
                    lines.append("\t\t...")
                else:
                    coord = [
                        self.data[dims[i]].values[ind] for i, ind in enumerate(indices)
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
                        line = f"{print_coord(coord)}: {expr} {sign} {rhs}"
                    else:
                        line = f"{print_coord(coord)}: None"
                    lines.append(line)
            lines = align_lines_by_delimiter(lines, list(SIGNS_pretty.values()))

            shape_str = ", ".join(f"{d}: {s}" for d, s in zip(dims, dim_sizes))
            mask_str = f" - {masked_entries} masked entries" if masked_entries else ""
            underscore = "-" * (len(shape_str) + len(mask_str) + len(header_string) + 4)
            lines.insert(0, f"{header_string} ({shape_str}){mask_str}:\n{underscore}")
        elif size == 1:
            expr = print_single_expression(self.coeffs, self.vars, 0, self.model)
            lines.append(
                f"{header_string}\n{'-'*len(header_string)}\n{expr} {SIGNS_pretty[self.sign.item()]} {self.rhs.item()}"
            )
        else:
            lines.append(f"{header_string}\n{'-'*len(header_string)}\n<empty>")

        return "\n".join(lines)

    def print(self, display_max_rows=20, display_max_terms=20):
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

    def __contains__(self, value):
        return self.data.__contains__(value)

    @property
    def type(self):
        """
        Get the type of the constraint.
        """
        return "Constraint" if self.is_assigned else "Constraint (unassigned)"

    @property
    def range(self):
        """
        Return the range of the constraint.
        """
        return self.data.attrs["label_range"]

    @property
    def term_dim(self):
        """
        Return the term dimension of the constraint.
        """
        return TERM_DIM

    @property
    def mask(self):
        """
        Get the mask of the constraint.

        The mask indicates on which coordinates the constraint is enabled
        (True) and disabled (False).

        Returns
        -------
        xr.DataArray
        """
        return self.data.get("mask")

    @property
    def coeffs(self):
        """
        Get the left-hand-side coefficients of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.data.coeffs

    @coeffs.setter
    def coeffs(self, value):
        value = DataArray(value).broadcast_like(self.vars, exclude=[self.term_dim])
        self._data = self.data.assign(coeffs=value)
        self.data["coeffs"] = value

    @property
    def vars(self):
        """
        Get the left-hand-side variables of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.data.vars

    @vars.setter
    def vars(self, value):
        if isinstance(value, variables.Variable):
            value = value.labels
        if not isinstance(value, DataArray):
            raise TypeError("Expected value to be of type DataArray or Variable")
        value = value.broadcast_like(self.coeffs, exclude=[self.term_dim])
        self.data["vars"] = value

    @property
    def lhs(self):
        """
        Get the left-hand-side linear expression of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        data = self.data[["coeffs", "vars"]].rename({self.term_dim: TERM_DIM})
        return expressions.LinearExpression(data, self.model)

    @lhs.setter
    def lhs(self, value):
        value = expressions.as_expression(
            value, self.model, coords=self.coords, dims=self.coord_dims
        )
        self._data = self.data.drop_vars(["coeffs", "vars"]).assign(
            coeffs=value.coeffs, vars=value.vars, rhs=self.rhs - value.const
        )

    @property
    def sign(self):
        """
        Get the signs of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.data.sign

    @sign.setter
    @is_constant
    def sign(self, value):
        value = maybe_replace_signs(DataArray(value)).broadcast_like(self.sign)
        self.data["sign"] = value

    @property
    def rhs(self):
        """
        Get the right hand side constants of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.data.rhs

    @rhs.setter
    def rhs(self, value):
        value = expressions.as_expression(
            value, self.model, coords=self.coords, dims=self.coord_dims
        )
        self.lhs = self.lhs - value.reset_const()
        self.data["rhs"] = value.const

    @property
    @has_optimized_model
    def dual(self):
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
    def dual(self, value):
        """
        Get the dual values of the constraint.
        """
        value = DataArray(value).broadcast_like(self.labels)
        self.data["dual"] = value

    @property
    def shape(self):
        return self.labels.shape

    @classmethod
    def from_rule(cls, model, rule, coords):
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
        ...         return (i - 1) * x[i - 1, j] >= 0
        ...     else:
        ...         return i * x[i, j] >= 0
        ...
        >>> con = Constraint.from_rule(m, bound, coords)
        >>> con = m.add_constraints(con)
        """
        if not isinstance(coords, xr.core.dataarray.DataArrayCoordinates):
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
    def flat(self):
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
        # if keys is not None:
        #     if isinstance(keys, str):
        #         keys = [keys]
        #     ds = ds[keys]
        if not ds.sizes:
            # fallback for weird error raised due to missing index
            df = pd.DataFrame({k: ds[k].item() for k in ds}, index=[0])
        else:
            df = ds.to_dataframe()
        df = df[(df.labels != -1) & (df.vars != -1) & (df.coeffs != 0)]
        # Group repeated variables in the same constraint
        agg = dict(coeffs="sum", rhs="first", sign="first")
        agg.update({k: "first" for k in df.columns if k not in agg})
        df = df.groupby(["labels", "vars"], as_index=False).aggregate(agg)

        any_nan = df.isna().any()
        if any_nan.any():
            fields = ", ".join("`" + df.columns[any_nan] + "`")
            raise ValueError(
                f"Constraint `{self.name}` contains nan's in field(s) {fields}"
            )
        return df

    sel = conwrap(Dataset.sel)

    isel = conwrap(Dataset.isel)


@dataclass(repr=False)
class Constraints:
    """
    A constraint container used for storing multiple constraint arrays.
    """

    data: Dict[str, Constraint] = field(default_factory=dict)
    model: Any = None  # Model is not defined due to circular imports

    dataset_attrs = ["labels", "coeffs", "vars", "sign", "rhs"]
    dataset_names = [
        "Labels",
        "Left-hand-side coefficients",
        "Left-hand-side variables",
        "Signs",
        "Right-hand-side constants",
    ]

    def __repr__(self):
        """
        Return a string representation of the linopy model.
        """
        r = "linopy.model.Constraints"
        line = "-" * len(r)
        r += f"\n{line}\n"

        for name, ds in self.items():
            coords = " (" + ", ".join(ds.coords) + ")" if ds.coords else ""
            r += f" * {name}{coords}\n"
        if not len(list(self)):
            r += "<empty>\n"
        return r

    def __getitem__(
        self, names: Union[str, Sequence[str]]
    ) -> Union[Constraint, "Constraints"]:
        if isinstance(names, str):
            return self.data[names]

        return self.__class__({name: self.data[name] for name in names}, self.model)

    def __getattr__(self, name: str):
        # If name is an attribute of self (including methods and properties), return that
        if name in self.data:
            return self.data[name]
        else:
            raise AttributeError(
                f"Constraints has no attribute `{name}` or the attribute is not accessible, e.g. raises an error."
            )

    def __len__(self):
        return self.data.__len__()

    def __iter__(self):
        return self.data.__iter__()

    def items(self):
        return self.data.items()

    def _ipython_key_completions_(self):
        """
        Provide method for the key-autocompletions in IPython.

        See
        http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
        For the details.
        """
        return list(self)

    def add(self, constraint):
        """
        Add a constraint to the constraints constrainer.
        """
        self.data[constraint.name] = constraint

    def remove(self, name):
        """
        Remove constraint `name` from the constraints.
        """
        self.data.pop(name)

    @property
    def loc(self):
        return LocIndexer(self)

    @property
    def labels(self):
        """
        Get the labels of all constraints.
        """
        return save_join(*[v.labels.rename(k) for k, v in self.items()])

    @property
    def coeffs(self):
        """
        Get the coefficients of all constraints.
        """
        return save_join(*[v.coeffs.rename(k) for k, v in self.items()])

    @property
    def vars(self):
        """
        Get the variables of all constraints.
        """
        return save_join(*[v.vars.rename(k) for k, v in self.items()])

    @property
    def sign(self):
        """
        Get the signs of all constraints.
        """
        return save_join(*[v.sign.rename(k) for k, v in self.items()])

    @property
    def rhs(self):
        """
        Get the right-hand-side constants of all constraints.
        """
        return save_join(*[v.rhs.rename(k) for k, v in self.items()])

    @property
    def dual(self):
        """
        Get the dual values of all constraints.
        """
        try:
            return save_join(*[v.dual.rename(k) for k, v in self.items()])
        except AttributeError:
            return Dataset()

    @property
    def coefficientrange(self):
        """
        Coefficient range of the constraint.
        """
        d = {
            k: [self[k].coeffs.min().item(), self[k].coeffs.max().item()] for k in self
        }
        return pd.DataFrame(d, index=["min", "max"]).T

    @property
    def ncons(self):
        """
        Get the number all constraints effectively used by the model.

        These excludes constraints with missing labels.
        """
        return len(self.flat.labels.unique())

    @property
    def inequalities(self):
        """
        Get the subset of constraints which are purely inequalities.
        """
        return self[[n for n, s in self.items() if (s.sign != EQUAL).all()]]

    @property
    def equalities(self):
        """
        Get the subset of constraints which are purely equalities.
        """
        return self[[n for n, s in self.items() if (s.sign == EQUAL).all()]]

    def sanitize_zeros(self):
        """
        Filter out terms with zero and close-to-zero coefficient.
        """
        for name in self:
            not_zero = abs(self[name].coeffs) > 1e-10
            self[name].vars = self[name].vars.where(not_zero, -1)
            self[name].coeffs = self[name].coeffs.where(not_zero)

    def sanitize_missings(self):
        """
        Set constraints labels to -1 where all variables in the lhs are
        missing.
        """
        for name in self:
            contains_non_missing = (self[name].vars != -1).any(self[name].term_dim)
            self[name].data["labels"] = self[name].labels.where(
                contains_non_missing, -1
            )

    def get_name_by_label(self, label):
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
        if not isinstance(label, (float, int)) or label < 0:
            raise ValueError("Label must be a positive number.")
        for name, ds in self.items():
            if label in ds.labels:
                return name
        raise ValueError(f"No constraint found containing the label {label}.")

    def get_label_position(self, values):
        """
        Get tuple of name and coordinate for constraint labels.
        """
        return get_label_position(self, values)

    def print_labels(self, values, display_max_terms=None):
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

    def set_blocks(self, block_map):
        """
        Get a dataset of same shape as constraints.labels with block values.

        Let N be the number of blocks.
        The following cases are considered:

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
            constraint.data["blocks"] = res

    @deprecated("0.2", details="Use `to_dataframe` or `flat` instead.")
    def iter_ravel(self, key, broadcast_like="labels", filter_missings=False):
        """
        Create an generator which iterates over all arrays in `key` and
        flattens them.

        Parameters
        ----------
        key : str/Dataset
            Key to be iterated over. Optionally pass a dataset which is
            broadcastable to `broadcast_like`.
        broadcast_like : str, optional
            Name of the dataset to which the input data in `key` is aligned to.
            Must be one of "labels", "vars". The default is "labels".
        filter_missings : bool, optional
            Filter out values where `labels` data is -1. If broadcast is `vars`
            also values where `vars` is -1 are filtered. This will raise an
            error if the filtered data still contains nan's.
            When enabled, the data is load into memory. The default is False.


        Yields
        ------
        flat : np.array/dask.array
        """
        assert broadcast_like in ["labels", "vars"]

        for name, constraint in self.items():
            values = constraint.data[broadcast_like]
            ds = constraint.data[key]
            broadcasted = ds.broadcast_like(values)
            if values.chunks is not None:
                broadcasted = broadcasted.chunk(values.chunks)

            if filter_missings:
                flat = np.ravel(broadcasted)
                mask = np.ravel(values) != -1
                if broadcast_like != "labels":
                    labels = np.ravel(constraint.labels.broadcast_like(values))
                    mask &= labels != -1
                flat = flat[mask]
                if pd.isna(flat).any():
                    names = self.dataset_names
                    ds_name = names[self.dataset_attrs.index(key)]
                    bc_name = names[self.dataset_attrs.index(broadcast_like)]
                    err = (
                        f"{ds_name} of constraint '{name}' are missing (nan) "
                        f"where {bc_name.lower()} are defined (not -1)."
                    )
                    raise ValueError(err)
            else:
                flat = broadcasted.data.ravel()
            yield flat

    @deprecated("0.2", details="Use `to_dataframe` or `flat` instead.")
    def ravel(self, key, broadcast_like="labels", filter_missings=False, compute=True):
        """
        Ravel and concate all arrays in `key` while aligning to
        `broadcast_like`.

        Parameters
        ----------
        key : str/Dataset
            Key to be iterated over. Optionally pass a dataset which is
            broadcastable to `broadcast_like`.
        broadcast_like : str, optional
            Name of the dataset to which the input data in `key` is aligned to.
            The default is "labels".
        filter_missings : bool, optional
            Filter out values where `broadcast_like` data is -1.
            The default is False.
        compute : bool, optional
            Whether to compute lazy data. The default is False.

        Returns
        -------
        flat
            One dimensional data with all values in `key`.
        """
        res = list(self.iter_ravel(key, broadcast_like, filter_missings))
        res = np.concatenate(res)
        return dask.compute(res)[0] if compute else res

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

    def to_matrix(self, filter_missings=True):
        """
        Construct a constraint matrix in sparse format.

        Missing values, i.e. -1 in labels and vars, are ignored filtered
        out.
        """
        # TODO: rename "filter_missings" to "~labels_as_coordinates"
        cons = self.flat

        if not len(self):
            return None

        if filter_missings:
            vars = self.model.variables.flat
            shape = (cons.key.max() + 1, vars.key.max() + 1)
            cons["vars"] = cons.vars.map(vars.set_index("labels").key)
            return csc_matrix((cons.coeffs, (cons.key, cons.vars)), shape=shape)
        else:
            shape = self.model.shape
            return csc_matrix((cons.coeffs, (cons.labels, cons.vars)), shape=shape)

    def reset_dual(self):
        """
        Reset the stored solution of variables.
        """
        for k, c in self.items():
            if "dual" in c:
                c._data = c.data.drop_vars("dual")


# define AnonymousConstraint for backwards compatibility
class AnonymousConstraint(Constraint):
    def __init__(self, lhs, sign, rhs):
        """
        Initialize a anonymous constraint.
        """
        # raise deprecation warning
        warnings.warn(
            "AnonymousConstraint is deprecated, use Constraint instead.",
        )

        if not isinstance(lhs, expressions.LinearExpression):
            raise TypeError(
                f"Assigned lhs must be a LinearExpression, got {type(lhs)})."
            )
        data = lhs.data.assign(sign=sign, rhs=rhs)
        super().__init__(data, lhs.model)


class AnonymousScalarConstraint:
    """
    Container for anonymous scalar constraint.

    This contains a left-hand-side (lhs), a sign and a right-hand-side
    (rhs) for exactly one constraint.
    """

    _lhs: "expressions.ScalarLinearExpression"
    _sign: str
    _rhs: float

    def __init__(self, lhs, sign, rhs):
        """
        Initialize a anonymous scalar constraint.
        """
        if not isinstance(rhs, (int, float, np.float32, np.float64, np.integer)):
            raise TypeError(f"Assigned rhs must be a constant, got {type(rhs)}).")
        self._lhs = lhs
        self._sign = sign
        self._rhs = rhs

    def __repr__(self):
        """
        Get the representation of the AnonymousScalarConstraint.
        """
        expr_string = print_single_expression(
            self.lhs.coeffs, self.lhs.vars, 0, self.lhs.model
        )
        return f"AnonymousScalarConstraint: {expr_string} {self.sign} {self.rhs}"

    @property
    def lhs(self):
        """
        Get the left hand side of the constraint.
        """
        return self._lhs

    @property
    def sign(self):
        """
        Get the sign of the constraint.
        """
        return self._sign

    @property
    def rhs(self):
        """
        Get the right hand side of the constraint.
        """
        return self._rhs

    def to_constraint(self):
        data = self.lhs.to_linexpr().data.assign(sign=self.sign, rhs=self.rhs)
        return Constraint(data=data, model=self.lhs.model)

    @deprecated(details="Use to_constraint instead.")
    def to_anonymous_constraint(self):
        return self.to_constraint()
