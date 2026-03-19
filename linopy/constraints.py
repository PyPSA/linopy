"""
Linopy constraints module.

This module contains implementations for the Constraint{s} class.
"""

from __future__ import annotations

import functools
import warnings
from abc import ABC, abstractmethod
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
    has_optimized_model,
    iterate_slices,
    maybe_group_terms_polars,
    maybe_replace_signs,
    print_coord,
    print_single_constraint,
    print_single_expression,
    replace_by_map,
    require_constant,
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
    PerformanceWarning,
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
    def _conwrap(
        con: MutableConstraint, *args: Any, **kwargs: Any
    ) -> MutableConstraint:
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


def _con_unwrap(con: ConstraintBase | Dataset) -> Dataset:
    return con.data if isinstance(con, ConstraintBase) else con


class ConstraintBase(ABC):
    """
    Abstract base class for Constraint and MutableConstraint.

    Provides all read-only properties and methods shared by both the immutable
    Constraint (CSR-backed) and the mutable MutableConstraint (Dataset-backed).
    """

    _fill_value = FILL_VALUE

    @property
    @abstractmethod
    def data(self) -> Dataset:
        """Get the underlying xarray Dataset representation."""

    @property
    @abstractmethod
    def model(self) -> Model:
        """Get the model reference."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the constraint name."""

    @property
    @abstractmethod
    def is_assigned(self) -> bool:
        """Whether the constraint has been assigned labels by the model."""

    @property
    @abstractmethod
    def labels(self) -> DataArray:
        """Get the labels DataArray."""

    @property
    @abstractmethod
    def coeffs(self) -> DataArray:
        """Get the LHS coefficients DataArray."""

    @property
    @abstractmethod
    def vars(self) -> DataArray:
        """Get the LHS variable labels DataArray."""

    @property
    @abstractmethod
    def sign(self) -> DataArray:
        """Get the constraint sign DataArray."""

    @property
    @abstractmethod
    def rhs(self) -> DataArray:
        """Get the RHS DataArray."""

    @property
    @abstractmethod
    def dual(self) -> DataArray:
        """Get the dual values DataArray."""

    def __getitem__(
        self, selector: str | int | slice | list | tuple | dict
    ) -> MutableConstraint:
        """
        Get selection from the constraint.
        Returns a MutableConstraint with the selected data.
        """
        data = Dataset(
            {k: self.data[k][selector] for k in self.data}, attrs=self.data.attrs
        )
        return MutableConstraint(data, self.model, self.name)

    @property
    def attrs(self) -> dict[str, Any]:
        """Get the attributes of the constraint."""
        return self.data.attrs

    @property
    def coords(self) -> DatasetCoordinates:
        """Get the coordinates of the constraint."""
        return self.data.coords

    @property
    def indexes(self) -> Indexes:
        """Get the indexes of the constraint."""
        return self.data.indexes

    @property
    def dims(self) -> Frozen[Hashable, int]:
        """Get the dimensions of the constraint."""
        return self.data.dims

    @property
    def sizes(self) -> Frozen[Hashable, int]:
        """Get the sizes of the constraint."""
        return self.data.sizes

    @property
    def nterm(self) -> int:
        """Get the number of terms in the constraint."""
        return self.data.sizes.get(TERM_DIM, 1)

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the constraint."""
        return self.rhs.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the constraint."""
        return self.rhs.shape

    @property
    def size(self) -> int:
        """Get the size of the constraint."""
        return self.rhs.size

    @property
    def ncons(self) -> int:
        """
        Get the number of active constraints (non-masked, with at least one valid variable).
        """
        labels = self.labels.values
        vars_arr = self.vars.values
        if labels.ndim == 0:
            return int(labels != FILL_VALUE["labels"] and (vars_arr != -1).any())
        return int(
            ((labels != FILL_VALUE["labels"]) & (vars_arr != -1).any(axis=-1)).sum()
        )

    @property
    def coord_dims(self) -> tuple[Hashable, ...]:
        return tuple(k for k in self.dims if k not in HELPER_DIMS)

    @property
    def coord_sizes(self) -> dict[Hashable, int]:
        return {k: v for k, v in self.sizes.items() if k not in HELPER_DIMS}

    @property
    def coord_names(self) -> list[str]:
        """Get the names of the coordinates."""
        return get_dims_with_index_levels(self.data, self.coord_dims)

    @property
    def type(self) -> str:
        """Get the type string of the constraint."""
        return "Constraint" if self.is_assigned else "Constraint (unassigned)"

    @property
    def term_dim(self) -> str:
        """Return the term dimension of the constraint."""
        return TERM_DIM

    @property
    def mask(self) -> DataArray | None:
        """
        Get the mask of the constraint.

        The mask indicates on which coordinates the constraint is enabled
        (True) and disabled (False).
        """
        if self.is_assigned:
            return (self.labels != FILL_VALUE["labels"]).astype(bool)
        return None

    @property
    def lhs(self) -> expressions.LinearExpression:
        """Get the left-hand-side linear expression of the constraint."""
        data = self.data[["coeffs", "vars"]].rename({self.term_dim: TERM_DIM})
        return expressions.LinearExpression(data, self.model)

    def __contains__(self, value: Any) -> bool:
        return self.data.__contains__(value)

    def __repr__(self) -> str:
        """Print the constraint arrays."""
        max_lines = options["display_max_rows"]
        dims = list(self.coord_sizes.keys())
        ndim = len(dims)
        dim_names = self.coord_names
        dim_sizes = list(self.coord_sizes.values())
        size = np.prod(dim_sizes)
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

    @property
    def flat(self) -> pd.DataFrame:
        """
        Convert the constraint to a pandas DataFrame.

        The resulting DataFrame represents a long table format of the all
        non-masked constraints with non-zero coefficients. It contains the
        columns `labels`, `coeffs`, `vars`, `rhs`, `sign`.
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

    def to_matrix(self) -> scipy.sparse.csr_array:
        """
        Construct a CSR matrix representation of this constraint.

        All flat positions in the constraint grid are included as rows;
        masked entries (labels == -1) become empty rows. Shape is
        (len(labels_flat), model._xCounter).

        Returns
        -------
        matrix : scipy.sparse.csr_array
        """
        vars = self.vars.values
        labels_flat = self.labels.values.ravel()
        vars_2d = vars.reshape(len(labels_flat), -1)
        coeffs_flat = self.coeffs.values.ravel()

        valid_2d = (labels_flat != -1)[:, np.newaxis] & (vars_2d != -1)
        cols = vars_2d[valid_2d]
        data = coeffs_flat.reshape(vars_2d.shape)[valid_2d]

        counts = valid_2d.sum(axis=1)
        indptr = np.empty(len(labels_flat) + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])

        shape = (len(labels_flat), self.model._xCounter)
        return scipy.sparse.csr_array((data, cols, indptr), shape=shape)

    def to_netcdf_ds(self) -> Dataset:
        """Return a Dataset representation suitable for netcdf serialization."""
        return self.data

    iterate_slices = iterate_slices


class Constraint(ConstraintBase):
    """
    Immutable constraint backed by a CSR sparse matrix.

    Parameters
    ----------
    csr : scipy.sparse.csr_array
        Shape (n_flat, model._xCounter). Each row is a flat position in the
        constraint grid (including masked/empty rows).
    rhs : np.ndarray
        Shape (n_flat,). Right-hand-side values.
    sign : str
        Constraint sign: one of '=', '<=', '>='.
        Note: per-element signs are not supported (documented regression vs MutableConstraint).
    coords : list of pd.Index
        One index per coordinate dimension defining the constraint grid.
    model : Model
        The linopy model this constraint belongs to.
    name : str
        Name of the constraint.
    cindex : int or None
        Starting label assigned by the model. None if not yet assigned.
    dual : np.ndarray or None
        Shape (n_flat,). Dual values after solving, or None.
    """

    __slots__ = (
        "_csr",
        "_rhs",
        "_sign",
        "_coords",
        "_model",
        "_name",
        "_cindex",
        "_dual",
    )

    def __init__(
        self,
        csr: scipy.sparse.csr_array,
        rhs: np.ndarray,
        sign: str,
        coords: list[pd.Index],
        model: Model,
        name: str = "",
        cindex: int | None = None,
        dual: np.ndarray | None = None,
    ) -> None:
        self._csr = csr
        self._rhs = rhs
        self._sign = sign
        self._coords = coords
        self._model = model
        self._name = name
        self._cindex = cindex
        self._dual = dual

    @property
    def model(self) -> Model:
        return self._model

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_assigned(self) -> bool:
        return self._cindex is not None

    @property
    def range(self) -> tuple[int, int]:
        """Return the (start, end) label range of the constraint."""
        if self._cindex is None:
            raise AttributeError("Constraint has not been assigned labels yet.")
        return (self._cindex, self._cindex + self._csr.shape[0])

    @property
    def _nonempty(self) -> np.ndarray:
        """Boolean array of shape (n_flat,) — True where row is non-masked."""
        return np.diff(self._csr.indptr).astype(bool)

    @property
    def ncons(self) -> int:
        return int(np.diff(self._csr.indptr).astype(bool).sum())

    @property
    def attrs(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self._name}
        if self._cindex is not None:
            d["label_range"] = (self._cindex, self._cindex + self._csr.shape[0])
        return d

    @property
    def dims(self) -> Frozen[Hashable, int]:
        d: dict[Hashable, int] = {c.name: len(c) for c in self._coords}
        nterm = int(self._csr.indptr.max()) if self._csr.nnz > 0 else 1
        d[TERM_DIM] = nterm
        return Frozen(d)

    @property
    def sizes(self) -> Frozen[Hashable, int]:
        return self.dims

    @property
    def indexes(self) -> Indexes:
        return Indexes({c.name: c for c in self._coords})

    @property
    def nterm(self) -> int:
        return int(self._csr.indptr.max()) if self._csr.nnz > 0 else 1

    @property
    def coord_names(self) -> list[str]:
        return [c.name for c in self._coords]

    @property
    def labels(self) -> DataArray:
        """Get labels DataArray, shape (*coord_dims)."""
        if self._cindex is None:
            return DataArray([])
        n_flat = self._csr.shape[0]
        labels_flat = np.where(
            self._nonempty,
            np.arange(self._cindex, self._cindex + n_flat),
            -1,
        )
        shape = tuple(len(c) for c in self._coords)
        dim_names = [c.name for c in self._coords]
        xr_coords = {c.name: c for c in self._coords}
        return DataArray(
            labels_flat.reshape(shape) if shape else labels_flat,
            coords=xr_coords,
            dims=dim_names,
        )

    @property
    def coeffs(self) -> DataArray:
        """Get coefficients DataArray, shape (*coord_dims, _term)."""
        warnings.warn(
            "Accessing .coeffs on a Constraint triggers full Dataset reconstruction. "
            "Use .to_matrix() for efficient access.",
            PerformanceWarning,
            stacklevel=2,
        )
        return self.data.coeffs

    @property
    def vars(self) -> DataArray:
        """Get variable labels DataArray, shape (*coord_dims, _term)."""
        warnings.warn(
            "Accessing .vars on a Constraint triggers full Dataset reconstruction. "
            "Use .to_matrix() for efficient access.",
            PerformanceWarning,
            stacklevel=2,
        )
        return self.data.vars

    @property
    def sign(self) -> DataArray:
        """Get sign DataArray (scalar, same sign for all entries)."""
        shape = tuple(len(c) for c in self._coords)
        dim_names = [c.name for c in self._coords]
        xr_coords = {c.name: c for c in self._coords}
        return DataArray(
            np.full(shape, self._sign) if shape else np.array(self._sign),
            coords=xr_coords,
            dims=dim_names,
        )

    @property
    def rhs(self) -> DataArray:
        """Get RHS DataArray, shape (*coord_dims)."""
        shape = tuple(len(c) for c in self._coords)
        dim_names = [c.name for c in self._coords]
        xr_coords = {c.name: c for c in self._coords}
        return DataArray(
            self._rhs.reshape(shape) if shape else self._rhs,
            coords=xr_coords,
            dims=dim_names,
        )

    @property
    @has_optimized_model
    def dual(self) -> DataArray:
        """Get dual values DataArray, shape (*coord_dims)."""
        if self._dual is None:
            raise AttributeError(
                "Underlying is optimized but does not have dual values stored."
            )
        shape = tuple(len(c) for c in self._coords)
        dim_names = [c.name for c in self._coords]
        xr_coords = {c.name: c for c in self._coords}
        return DataArray(
            self._dual.reshape(shape) if shape else self._dual,
            coords=xr_coords,
            dims=dim_names,
        )

    def _to_dataset(self, nterm: int) -> Dataset:
        """
        Reconstruct labels/coeffs/vars Dataset from the CSR matrix.

        Parameters
        ----------
        nterm : int
            Number of terms per row (width of the dense term block).

        Returns
        -------
        Dataset with variables ``labels``, ``coeffs``, ``vars``.
        """
        csr = self._csr
        n_total = csr.shape[0]
        counts = np.diff(csr.indptr)
        nonempty = counts > 0
        coeffs_2d = np.zeros((n_total, nterm), dtype=csr.dtype)
        vars_2d = np.full((n_total, nterm), -1, dtype=np.int64)
        if csr.nnz > 0:
            row_indices = np.repeat(np.arange(n_total, dtype=np.int32), counts)
            term_cols = np.arange(csr.nnz, dtype=np.int32) - np.repeat(
                csr.indptr[:-1].astype(np.int32), counts
            )
            vars_2d[row_indices, term_cols] = csr.indices
            coeffs_2d[row_indices, term_cols] = csr.data
        shape = tuple(len(c) for c in self._coords)
        dim_names = [c.name for c in self._coords]
        xr_coords = {c.name: c for c in self._coords}
        term_coords = {TERM_DIM: np.arange(nterm)}
        dims_with_term = dim_names + [TERM_DIM]
        coords_with_term = {**xr_coords, **term_coords}
        coeffs_da = DataArray(
            coeffs_2d.reshape(shape + (nterm,)) if shape else coeffs_2d,
            coords=coords_with_term,
            dims=dims_with_term,
        )
        vars_da = DataArray(
            vars_2d.reshape(shape + (nterm,)) if shape else vars_2d,
            coords=coords_with_term,
            dims=dims_with_term,
        )
        ds = Dataset({"coeffs": coeffs_da, "vars": vars_da})
        if self._cindex is not None:
            labels_flat = np.where(
                nonempty, np.arange(self._cindex, self._cindex + n_total), -1
            )
            ds["labels"] = DataArray(
                labels_flat.reshape(shape) if shape else labels_flat,
                coords=xr_coords,
                dims=dim_names,
            )
        return ds

    @property
    def data(self) -> Dataset:
        """Reconstruct the xarray Dataset from the CSR representation."""
        nterm = int(self._csr.indptr.max()) if self._csr.nnz > 0 else 1
        ds = self._to_dataset(nterm)
        sign_arr = np.full(tuple(len(c) for c in self._coords) or (1,), self._sign)
        rhs_arr = self._rhs.reshape(tuple(len(c) for c in self._coords) or (1,))
        shape = tuple(len(c) for c in self._coords)
        dim_names = [c.name for c in self._coords]
        xr_coords = {c.name: c for c in self._coords}
        ds = ds.assign(
            sign=DataArray(
                sign_arr if shape else np.array(self._sign),
                coords=xr_coords,
                dims=dim_names,
            ),
            rhs=DataArray(
                rhs_arr if shape else self._rhs, coords=xr_coords, dims=dim_names
            ),
        )
        if self._dual is not None:
            ds = ds.assign(
                dual=DataArray(
                    self._dual.reshape(shape) if shape else self._dual,
                    coords=xr_coords,
                    dims=dim_names,
                )
            )
        return ds.assign_attrs(name=self._name)

    def __repr__(self) -> str:
        """Print the constraint without reconstructing the full Dataset."""
        max_lines = options["display_max_rows"]
        coords = self._coords
        shape = tuple(len(c) for c in coords)
        dim_names = [c.name for c in coords]
        dim_sizes = list(shape)
        size = int(np.prod(shape)) if shape else 1
        nonempty = self._nonempty  # shape (size,)
        nterm = int(self._csr.indptr.max()) if self._csr.nnz > 0 else 1

        # Dense arrays for coeffs/vars, built without going through data property
        csr = self._csr
        counts = np.diff(csr.indptr)
        coeffs_2d = np.zeros((size, nterm), dtype=csr.dtype)
        vars_2d = np.full((size, nterm), -1, dtype=np.int64)
        if csr.nnz > 0:
            row_idx = np.repeat(np.arange(size, dtype=np.int32), counts)
            term_cols = np.arange(csr.nnz, dtype=np.int32) - np.repeat(
                csr.indptr[:-1].astype(np.int32), counts
            )
            vars_2d[row_idx, term_cols] = csr.indices
            coeffs_2d[row_idx, term_cols] = csr.data

        coeffs_nd = coeffs_2d.reshape(shape + (nterm,)) if shape else coeffs_2d
        vars_nd = vars_2d.reshape(shape + (nterm,)) if shape else vars_2d
        rhs_nd = self._rhs.reshape(shape) if shape else self._rhs
        masked_entries = int((~nonempty).sum())

        header_string = f"{self.type} `{self._name}`" if self._name else f"{self.type}"
        lines = []

        if size > 1 or len(dim_sizes) > 0:
            for indices in generate_indices_for_printout(dim_sizes, max_lines):
                if indices is None:
                    lines.append("\t\t...")
                else:
                    coord = [coords[i][int(ind)] for i, ind in enumerate(indices)]
                    flat_idx = int(np.ravel_multi_index(indices, shape)) if shape else 0
                    if nonempty[flat_idx]:
                        expr = print_single_expression(
                            coeffs_nd[indices], vars_nd[indices], 0, self._model
                        )
                        sign = SIGNS_pretty[self._sign]
                        rhs = rhs_nd[indices]
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
                coeffs_nd.ravel(), vars_nd.ravel(), 0, self._model
            )
            lines.append(
                f"{header_string}\n{'-' * len(header_string)}\n{expr} {SIGNS_pretty[self._sign]} {self._rhs.item()}"
            )
        else:
            lines.append(f"{header_string}\n{'-' * len(header_string)}\n<empty>")

        return "\n".join(lines)

    def to_matrix(self) -> scipy.sparse.csr_array:
        """Return the stored CSR matrix directly (no reconstruction needed)."""
        return self._csr

    def to_netcdf_ds(self) -> Dataset:
        """Return a Dataset with raw CSR components for netcdf serialization."""
        from xarray import DataArray

        csr = self._csr
        data_vars: dict[str, DataArray] = {
            "indptr": DataArray(csr.indptr, dims=["_indptr"]),
            "indices": DataArray(csr.indices, dims=["_nnz"]),
            "data": DataArray(csr.data, dims=["_nnz"]),
            "rhs": DataArray(self._rhs, dims=["_flat"]),
        }
        for c in self._coords:
            data_vars[f"_coord_{c.name}"] = DataArray(
                np.array(c), dims=[f"_coorddim_{c.name}"]
            )
        if self._dual is not None:
            data_vars["dual"] = DataArray(self._dual, dims=["_flat"])
        dim_names = [c.name for c in self._coords]
        return Dataset(
            data_vars,
            attrs={
                "_linopy_format": "csr",
                "sign": self._sign,
                "cindex": self._cindex if self._cindex is not None else -1,
                "shape": list(csr.shape),
                "coord_dims": dim_names,
                "name": self._name,
            },
        )

    @classmethod
    def from_netcdf_ds(cls, ds: Dataset, model: Model, name: str) -> Constraint:
        """Reconstruct a Constraint from a netcdf Dataset (CSR format)."""
        attrs = ds.attrs
        shape = tuple(attrs["shape"])
        csr = scipy.sparse.csr_array(
            (ds["data"].values, ds["indices"].values, ds["indptr"].values),
            shape=shape,
        )
        rhs = ds["rhs"].values
        sign = attrs["sign"]
        cindex = int(attrs["cindex"])
        cindex = cindex if cindex >= 0 else None
        coord_dims = attrs["coord_dims"]
        if isinstance(coord_dims, str):
            coord_dims = [coord_dims]
        coords = [pd.Index(ds[f"_coord_{d}"].values, name=d) for d in coord_dims]
        dual = ds["dual"].values if "dual" in ds else None
        return cls(csr, rhs, sign, coords, model, name, cindex=cindex, dual=dual)

    def freeze(self) -> Constraint:
        """Return self (already immutable)."""
        return self

    def mutable(self) -> MutableConstraint:
        """Convert to a MutableConstraint."""
        return MutableConstraint(self.data, self._model, self._name)

    @classmethod
    def from_mutable(
        cls,
        con: MutableConstraint,
        cindex: int | None = None,
    ) -> Constraint:
        """
        Create a Constraint from a MutableConstraint.

        Parameters
        ----------
        con : MutableConstraint
        cindex : int or None
            Starting label index, if assigned.
        """
        csr = con.to_matrix()
        coords = [con.indexes[d] for d in con.coord_dims]
        rhs = con.rhs.values.ravel()
        sign_vals = con.sign.values.ravel()
        unique_signs = np.unique(sign_vals)
        if len(unique_signs) > 1:
            raise ValueError(
                "Constraint has per-element signs; cannot freeze to immutable Constraint. "
                "This is a known limitation — use MutableConstraint instead."
            )
        sign = str(unique_signs[0]) if len(unique_signs) == 1 else "="
        dual = con.data["dual"].values.ravel() if "dual" in con.data else None
        return cls(
            csr, rhs, sign, coords, con.model, con.name, cindex=cindex, dual=dual
        )


class MutableConstraint(ConstraintBase):
    """
    Mutable constraint backed by an xarray Dataset.

    This is the original Constraint implementation, renamed to MutableConstraint.
    Supports setters, xarray operations via conwrap, and from_rule construction.
    """

    __slots__ = ("_data", "_model", "_assigned")

    def __init__(
        self,
        data: Dataset,
        model: Model,
        name: str = "",
        skip_broadcast: bool = False,
    ) -> None:
        from linopy.model import Model

        if not isinstance(data, Dataset):
            raise ValueError(f"data must be a Dataset, got {type(data)}")

        if not isinstance(model, Model):
            raise ValueError(f"model must be a Model, got {type(model)}")

        for attr in ("coeffs", "vars", "sign", "rhs"):
            if attr not in data:
                raise ValueError(f"missing '{attr}' in data")

        data = data.assign_attrs(name=name)

        if not skip_broadcast:
            (data,) = xr.broadcast(data, exclude=[TERM_DIM])

        self._assigned = "labels" in data
        self._data = data
        self._model = model

    @property
    def data(self) -> Dataset:
        return self._data

    @property
    def model(self) -> Model:
        return self._model

    @property
    def name(self) -> str:
        return self.attrs["name"]

    @property
    def is_assigned(self) -> bool:
        return self._assigned

    @property
    def range(self) -> tuple[int, int]:
        """Return the range of the constraint."""
        return self.data.attrs["label_range"]

    @property
    def loc(self) -> LocIndexer:
        return LocIndexer(self)

    @property
    def labels(self) -> DataArray:
        return self.data.get("labels", DataArray([]))

    @property
    def coeffs(self) -> DataArray:
        return self.data.coeffs

    @coeffs.setter
    def coeffs(self, value: ConstantLike) -> None:
        value = DataArray(value).broadcast_like(self.vars, exclude=[self.term_dim])
        self._data = assign_multiindex_safe(self.data, coeffs=value)

    @property
    def vars(self) -> DataArray:
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
    def sign(self) -> DataArray:
        return self.data.sign

    @sign.setter
    @require_constant
    def sign(self, value: SignLike) -> None:
        value = maybe_replace_signs(DataArray(value)).broadcast_like(self.sign)
        self._data = assign_multiindex_safe(self.data, sign=value)

    @property
    def rhs(self) -> DataArray:
        return self.data.rhs

    @rhs.setter
    def rhs(self, value: ExpressionLike) -> None:
        value = expressions.as_expression(
            value, self.model, coords=self.coords, dims=self.coord_dims
        )
        self.lhs = self.lhs - value.reset_const()
        self._data = assign_multiindex_safe(self.data, rhs=value.const)

    @property
    def lhs(self) -> expressions.LinearExpression:
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
    @has_optimized_model
    def dual(self) -> DataArray:
        if "dual" not in self.data:
            raise AttributeError(
                "Underlying is optimized but does not have dual values stored."
            )
        return self.data["dual"]

    @dual.setter
    def dual(self, value: ConstantLike) -> None:
        value = DataArray(value).broadcast_like(self.labels)
        self._data = assign_multiindex_safe(self.data, dual=value)

    def freeze(self) -> Constraint:
        """Convert to an immutable Constraint."""
        cindex = (
            int(self.data.attrs["label_range"][0])
            if "label_range" in self.data.attrs
            else None
        )
        return Constraint.from_mutable(self, cindex=cindex)

    def mutable(self) -> MutableConstraint:
        """Return self (already mutable)."""
        return self

    @classmethod
    def from_rule(
        cls, model: Model, rule: Callable, coords: CoordsLike
    ) -> MutableConstraint:
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
        linopy.MutableConstraint

        Examples
        --------
        >>> from linopy import Model, LinearExpression, MutableConstraint
        >>> m = Model()
        >>> coords = pd.RangeIndex(10), ["a", "b"]
        >>> x = m.add_variables(0, 100, coords)
        >>> def bound(m, i, j):
        ...     if i % 2:
        ...         return (i - 1) * x.at[i - 1, j] >= 0
        ...     else:
        ...         return i * x.at[i, j] >= 0
        ...
        >>> con = MutableConstraint.from_rule(m, bound, coords)
        >>> con = m.add_constraints(con)
        """
        if not isinstance(coords, DataArrayCoordinates):
            coords = DataArray(coords=coords).coords
        shape = list(map(len, coords.values()))

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
        if ds.sizes.get("_term", 1) > 1:
            long = maybe_group_terms_polars(long)
        check_has_nulls_polars(long, name=f"{self.type} {self.name}")

        labels_flat = ds["labels"].values.reshape(-1)
        mask = labels_flat != -1
        labels_masked = labels_flat[mask]
        rhs_flat = np.broadcast_to(ds["rhs"].values, ds["labels"].shape).reshape(-1)

        sign_values = ds["sign"].values
        sign_flat = np.broadcast_to(sign_values, ds["labels"].shape).reshape(-1)
        all_same_sign = len(sign_flat) > 0 and (
            sign_flat[0] == sign_flat[-1] and (sign_flat[0] == sign_flat).all()
        )

        short_data: dict = {
            "labels": labels_masked,
            "rhs": rhs_flat[mask],
        }
        if all_same_sign:
            short = pl.DataFrame(short_data).with_columns(
                pl.lit(sign_flat[0]).cast(pl.Enum(["=", "<=", ">="])).alias("sign")
            )
        else:
            short_data["sign"] = pl.Series(
                "sign", sign_flat[mask], dtype=pl.Enum(["=", "<=", ">="])
            )
            short = pl.DataFrame(short_data)

        df = long.join(short, on="labels", how="inner")
        return df[["labels", "coeffs", "vars", "sign", "rhs"]]

    # Wrapped xarray methods — only available on MutableConstraint
    assign = conwrap(Dataset.assign)
    assign_multiindex_safe = conwrap(assign_multiindex_safe)
    assign_attrs = conwrap(Dataset.assign_attrs)
    assign_coords = conwrap(Dataset.assign_coords)
    broadcast_like = conwrap(Dataset.broadcast_like)
    chunk = conwrap(Dataset.chunk)
    drop_sel = conwrap(Dataset.drop_sel)
    drop_isel = conwrap(Dataset.drop_isel)
    expand_dims = conwrap(Dataset.expand_dims)
    sel = conwrap(Dataset.sel)
    isel = conwrap(Dataset.isel)
    shift = conwrap(Dataset.shift)
    swap_dims = conwrap(Dataset.swap_dims)
    set_index = conwrap(Dataset.set_index)
    reindex = conwrap(Dataset.reindex, fill_value=FILL_VALUE)
    reindex_like = conwrap(Dataset.reindex_like, fill_value=FILL_VALUE)
    rename = conwrap(Dataset.rename)
    rename_dims = conwrap(Dataset.rename_dims)
    roll = conwrap(Dataset.roll)
    stack = conwrap(Dataset.stack)
    unstack = conwrap(Dataset.unstack)


@dataclass(repr=False)
class Constraints:
    """
    A constraint container used for storing multiple constraint arrays.
    """

    data: dict[str, ConstraintBase]
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
    def __getitem__(self, names: str) -> ConstraintBase: ...

    @overload
    def __getitem__(self, names: list[str]) -> Constraints: ...

    def __getitem__(self, names: str | list[str]) -> ConstraintBase | Constraints:
        if isinstance(names, str):
            return self.data[names]
        return Constraints({name: self.data[name] for name in names}, self.model)

    def __getattr__(self, name: str) -> ConstraintBase:
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

    def items(self) -> ItemsView[str, ConstraintBase]:
        return self.data.items()

    def _ipython_key_completions_(self) -> list[str]:
        """
        Provide method for the key-autocompletions in IPython.

        See
        http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
        For the details.
        """
        return list(self)

    def add(self, constraint: ConstraintBase, freeze: bool = False) -> ConstraintBase:
        """
        Add a constraint to the constraints container.
        """
        if freeze and isinstance(constraint, MutableConstraint):
            constraint = constraint.freeze()
        self.data[constraint.name] = constraint
        self._invalidate_label_position_index()
        return constraint

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
        return sum(con.ncons for con in self.data.values())

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

        output = "\n".join(res)
        try:
            print(output)
        except UnicodeEncodeError:
            # Replace Unicode math symbols with ASCII equivalents for Windows console
            output = output.replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")
            print(output)

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

    def to_matrix(
        self, filter_missings: bool = True
    ) -> tuple[scipy.sparse.csc_array, np.ndarray, np.ndarray | None]:
        """
        Construct a constraint matrix in sparse format by stacking per-constraint CSR matrices.

        Parameters
        ----------
        filter_missings : bool, default True
            If True, also strip empty columns and return ``var_labels`` for
            remapping columns back to original variable labels.
            If False, return full-width CSC with shape
            ``(n_active_cons, model._xCounter)`` and ``var_labels=None``.
            ``con_labels`` is always returned.

        Returns
        -------
        matrix : scipy.sparse.csc_array
            Shape ``(n_active_cons, n_active_vars)`` when
            ``filter_missings=True``, or ``(n_active_cons,
            model._xCounter)`` when ``filter_missings=False``.
        con_labels : np.ndarray
            Shape ``(n_active_cons,)``, maps each matrix row to the
            original constraint label.
        var_labels : np.ndarray or None
            Shape ``(n_active_vars,)``, maps each matrix column to the
            original variable label.  ``None`` when
            ``filter_missings=False``.
        """
        if not len(self):
            raise ValueError("No constraints available to convert to matrix.")

        active_csrs = []
        con_labels_list = []
        for c in self.data.values():
            csr = c.to_matrix()
            nonempty = np.diff(csr.indptr).astype(bool)
            active_csrs.append(csr[nonempty])
            start = (
                c._cindex
                if isinstance(c, Constraint)
                else c.data.attrs["label_range"][0]
            )
            con_labels_list.append(np.flatnonzero(nonempty) + start)
        csc: scipy.sparse.csc_array = scipy.sparse.vstack(active_csrs).tocsc()
        csc.sum_duplicates()
        con_labels = np.concatenate(con_labels_list)

        if filter_missings:
            indptr = csc.indptr
            nonempty_cols = indptr[1:] != indptr[:-1]
            new_indptr = np.r_[0, indptr[1:][nonempty_cols]]
            (var_labels,) = np.nonzero(nonempty_cols)
            matrix = scipy.sparse.csc_array(
                (csc.data, csc.indices, new_indptr),
                shape=(csc.shape[0], len(var_labels)),
            )
            return matrix, con_labels, var_labels
        else:
            return csc, con_labels, None

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

    def to_constraint(self) -> MutableConstraint:
        data = self.lhs.to_linexpr().data.assign(sign=self.sign, rhs=self.rhs)
        return MutableConstraint(data=data, model=self.lhs.model)
