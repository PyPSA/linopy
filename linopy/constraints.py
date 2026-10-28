"""
Linopy constraints module.

This module contains implementations for the Constraint{s} class.
"""

from __future__ import annotations

import copy
import functools
from collections.abc import (
    Callable,
    Generator,
    Hashable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from itertools import product
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    overload,
)

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse
import xarray as xr
from numpy import array
from xarray import DataArray, Dataset
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates
from xarray.core.indexes import Indexes
from xarray.core.utils import Frozen

from linopy import expressions, variables
from linopy.alignment import as_dataarray, broadcast_to_coords
from linopy.common import (
    ConstraintLabelIndex,
    LocIndexer,
    VariableLabelIndex,
    align_lines_by_delimiter,
    assign_fields,
    assigned_labels,
    check_has_nulls,
    check_has_nulls_polars,
    contains_labels,
    filter_nulls_polars,
    format_coord,
    format_single_constraint,
    format_single_expression,
    generate_indices_for_printout,
    get_dims_with_index_levels,
    get_printout_labels,
    has_optimized_model,
    maybe_group_terms_polars,
    maybe_replace_signs,
    reassign_coords,
    replace_by_map,
    save_join,
    slice_indexers,
    to_dataframe,
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
from linopy.containers import LabelContainer
from linopy.csr import CSRStore, _densify_notice
from linopy.scaling import validate_scaling
from linopy.semantics import check_user_nan
from linopy.store import NETCDF_STORE_ATTRS, DenseStore, Reason, Store, as_sparse
from linopy.types import (
    ConstantLike,
    ConstraintLike,
    CoordsLike,
    ExpressionLike,
    SignLike,
    VariableLike,
)

if TYPE_CHECKING:
    from linopy.model import Model


FILL_VALUE = {
    "labels": -1,
    "rhs": np.nan,
    "coeffs": np.nan,
    "vars": -1,
    "sign": "=",
    "scaling": 1.0,
}


def conwrap(
    method: Callable, *default_args: Any, **new_default_kwargs: Any
) -> Callable:
    @functools.wraps(method)
    def _conwrap(con: Constraint, *args: Any, **kwargs: Any) -> Constraint:
        for k, v in new_default_kwargs.items():
            kwargs.setdefault(k, v)
        return con.from_dataset(
            method(con.data, *default_args, *args, **kwargs), con.model, con.name
        )

    _conwrap.__doc__ = (
        f"Wrapper for the xarray {method.__qualname__} function for linopy.Constraint"
    )
    if new_default_kwargs:
        _conwrap.__doc__ += f" with default arguments: {new_default_kwargs}"

    return _conwrap


def _con_unwrap(con: Constraint | Dataset) -> Dataset:
    return con._dataset() if isinstance(con, Constraint) else con


def _slack_names(attrs: Mapping[str, Any]) -> tuple[str, str] | None:
    """Names of the positive and negative slack variables recorded by ``soften``."""
    positive = attrs.get("slack_positive")
    if positive is None:
        return None
    return str(positive), str(attrs.get("slack_negative", ""))


def _sparse_if(store: Store, sparse: bool, reason: Reason | None = None) -> Store:
    """
    ``store`` converted to sparse if ``sparse`` and it converts, else as is;
    with a ``reason``, a store that stays dense emits the densify notice.
    """
    if not sparse:
        return store
    converted = store.to_sparse()
    if not isinstance(converted, str):
        return converted
    if reason is not None:
        _densify_notice(f"{reason} ({converted})", store.model)
    return store


def model_store(store: Store, model: Model) -> Store:
    """``store`` on the store class of ``model`` where it converts, without a notice."""
    return _sparse_if(store, model.store_cls is CSRStore)


def rhs_on_grid(store: CSRStore, rhs: Any) -> DataArray | Reason:
    """
    The constant ``rhs`` as a DataArray on the grid of ``store``, with dense
    parity: NaN in the rhs raises (§5) and a reordered or differing index on
    a shared dim raises (§8). An rhs with helper dims or dims outside the
    grid returns the reason to fall back to the dense path. An expression
    rhs is moved to the lhs before, see :meth:`LinearExpression.to_constraint`.
    """
    da = as_dataarray(rhs)
    grid = store.grid
    if set(da.dims) & set(HELPER_DIMS) or not set(da.dims) <= set(grid.dims):
        return "to_constraint(rhs over dimensions outside the grid)"
    if bool(da.isnull().any()):
        check_user_nan()
    for d in da.dims:
        if not da.get_index(d).equals(grid.indexes[str(d)]):
            raise ValueError(
                f"Coordinate mismatch on shared dimension {d!r} between "
                "the rhs and the grouped result. Align the rhs with "
                ".sel(...) / .reindex(...) before combining (§8)."
            )
    missing = {d: i for d, i in grid.indexes.items() if d not in da.dims}
    if missing:
        da = da.expand_dims(missing)
    values = da.transpose(*grid.dims).to_numpy().reshape(-1)
    return grid.dataarray(values.astype(float), name="rhs")


class Slack(NamedTuple):
    """
    Slack variable(s) added by :meth:`Constraint.soften`.

    `negative` is None for inequality constraints, since those only
    need one slack variable to absorb a violation in a single direction.
    """

    positive: variables.Variable
    negative: variables.Variable | None


_FIELDS = (
    "sign",
    "rhs",
    "labels",
    "scaling",
    "dual",
    "binary_var",
    "binary_val",
    "blocks",
)
_DATASET_FIELDS = frozenset(("coeffs", "vars", *_FIELDS))


def _on_grid(value: Any, template: DataArray, label: str) -> DataArray:
    """``value`` broadcast onto the dims of ``template`` and in their order."""
    da = value if isinstance(value, DataArray) else DataArray(value)
    if extra := set(da.dims) - set(template.dims):
        raise ValueError(
            f"Constraint {label} has dimensions {sorted(map(str, extra))} outside "
            f"the dimensions {list(map(str, template.dims))} of its lhs."
        )
    if not da.ndim and template.ndim:
        return template.copy(deep=False, data=np.full(template.shape, da.values[()]))
    indexes = template.indexes
    aligned = all(i.equals(indexes[k]) for k, i in da.indexes.items() if k in indexes)
    if aligned and da.dims != template.dims:
        da = da.broadcast_like(template).transpose(*template.dims)
    if not aligned or da.sizes != template.sizes:
        raise ValueError(
            f"Constraint {label} does not align with the coordinates of its lhs."
        )
    missing = {k: v for k, v in template.coords.items() if k not in da.coords}
    return da.assign_coords(missing) if missing else da


class Constraint:
    """
    Constraint ``lhs sign rhs`` on the grid of its left-hand side.

    The lhs is a :class:`~linopy.expressions.LinearExpression` with a zero
    constant, its store in canonical form: a
    :class:`~linopy.store.DenseStore`, or a :class:`~linopy.csr.CSRStore` in a
    sparse model (``Model(sparse=True)``). ``sign``, ``rhs``, ``labels``,
    ``scaling``, ``dual``, ``binary_var``, ``binary_val`` and ``blocks`` are
    DataArrays on its grid; :attr:`data` is the Dataset derived from them.
    Supports setters, xarray operations via conwrap, and from_rule
    construction.
    """

    __slots__ = (
        "_lhs",
        "_sign",
        "_rhs",
        "_labels",
        "_scaling",
        "_dual",
        "_binary_var",
        "_binary_val",
        "_blocks",
        "_grid",
        "_attrs",
        "_model",
        "_coef_dirty",
    )

    _lhs: expressions.LinearExpression
    _sign: DataArray
    _rhs: DataArray
    _labels: DataArray | None
    _scaling: DataArray
    _dual: DataArray | None
    _binary_var: DataArray | None
    _binary_val: DataArray | None
    _blocks: DataArray | None
    _grid: Dataset
    _attrs: dict[str, Any]

    def __init__(
        self,
        lhs: expressions.LinearExpression,
        sign: SignLike,
        rhs: ConstantLike,
        model: Model,
        name: str = "",
        labels: DataArray | None = None,
        scaling: ConstantLike = 1.0,
        dual: DataArray | None = None,
        binary_var: DataArray | None = None,
        binary_val: ConstantLike | None = None,
        blocks: DataArray | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        from linopy.model import Model

        if not isinstance(lhs, expressions.LinearExpression):
            raise TypeError(f"lhs must be a LinearExpression, got {type(lhs)}")
        if not isinstance(model, Model):
            raise ValueError(f"model must be a Model, got {type(model)}")

        self._model = model
        self._attrs = {**(attrs or {}), "name": name}
        self._coef_dirty = False
        fields = {
            "sign": sign,
            "rhs": rhs,
            "labels": labels,
            "scaling": scaling,
            "dual": dual,
            "binary_var": binary_var,
            "binary_val": binary_val,
            "blocks": blocks,
        }
        self._bind(lhs, {k: v for k, v in fields.items() if v is not None})

    @classmethod
    def from_dataset(cls, ds: Dataset, model: Model, name: str = "") -> Constraint:
        """
        Build a constraint from its dense Dataset layout, see :attr:`data`.

        The Dataset holds ``coeffs`` and ``vars`` on the grid dims followed
        by ``_term``, and ``sign``, ``rhs`` and optionally ``labels``,
        ``scaling``, ``dual``, ``binary_var``, ``binary_val`` and ``blocks``.
        The lhs is held on the store class of ``model`` where it converts.
        """
        if not isinstance(ds, Dataset):
            raise ValueError(f"data must be a Dataset, got {type(ds)}")
        for attr in ("coeffs", "vars", "sign", "rhs"):
            if attr not in ds:
                raise ValueError(f"missing '{attr}' in data")
        if unknown := set(map(str, ds.data_vars)) - _DATASET_FIELDS:
            raise ValueError(f"unknown constraint fields {sorted(unknown)} in data")

        (ds,) = xr.broadcast(ds, exclude=[TERM_DIM])
        dense = expressions.LinearExpression(ds[["coeffs", "vars"]], model)
        lhs = expressions.LinearExpression._from_store(
            model_store(dense._store, model), model
        )
        fields = {k: ds[k] for k in _FIELDS if k in ds}
        return cls(lhs, model=model, name=name, attrs=ds.attrs, **fields)

    @classmethod
    def from_netcdf_ds(cls, ds: Dataset, model: Model, name: str) -> Constraint:
        """Read a constraint written by :meth:`to_netcdf_ds`."""
        store = Store.from_netcdf_vars(ds, model)
        lhs = expressions.LinearExpression._from_store(store, model)
        fields = {k: ds[k] for k in _FIELDS if k in ds}
        attrs = {k: v for k, v in ds.attrs.items() if k not in NETCDF_STORE_ATTRS}
        return cls(lhs, model=model, name=name, attrs=attrs, **fields)

    def to_netcdf_ds(self) -> Dataset:
        """
        A Dataset for netcdf serialization: the lhs store arrays, see
        :meth:`Store.to_netcdf_vars`, next to the fields on the grid dims.
        """
        ds = self._lhs._store.to_netcdf_vars().assign(self._fields())
        return ds.assign_attrs(self._attrs)

    def _bind(
        self, lhs: expressions.LinearExpression, fields: Mapping[Any, Any]
    ) -> None:
        """
        Hold ``lhs`` in canonical form and ``fields`` on its grid. Callers
        replacing the lhs terms flip ``_coef_dirty``.
        """
        store = lhs._store.canonical()
        const = store.const
        if np.any((const != 0) & ~np.isnan(const)):
            raise ValueError(
                "The lhs of a constraint must have a zero constant; move it to the rhs."
            )
        template = store.dataarray(const, "const")
        self._grid = template.to_dataset()
        self._lhs = expressions.LinearExpression._from_store(store, self._model)
        self._sign = _on_grid(fields["sign"], template, "sign")
        self._rhs = _on_grid(fields["rhs"], template, "rhs")
        self._scaling = _on_grid(fields.get("scaling", 1.0), template, "scaling")
        self._labels = self._optional_on_grid(fields, "labels", template)
        self._dual = self._optional_on_grid(fields, "dual", template)
        self._binary_var = self._optional_on_grid(fields, "binary_var", template)
        self._binary_val = self._optional_on_grid(fields, "binary_val", template)
        self._blocks = self._optional_on_grid(fields, "blocks", template)

    def _added(
        self,
        model: Model,
        name: str,
        rhs: DataArray,
        labels: DataArray,
        scaling: DataArray,
        attrs: Mapping[str, Any],
    ) -> Constraint:
        """
        Internal: this anonymous constraint as ``model`` adds it, named, with
        ``rhs``, ``labels`` and ``scaling`` on its grid and its lhs on the
        model's store class; the lhs and sign already held stay bound.
        """
        store = model_store(self._lhs._store, model)
        if store is not self._lhs._store or model is not self._model:
            lhs = expressions.LinearExpression._from_store(store, model)
            return Constraint(
                lhs, self._sign, rhs, model, name, labels, scaling, attrs=attrs
            )
        added = copy.copy(self)
        template = self._grid_template()
        added._attrs = {**attrs, "name": name}
        added._rhs = _on_grid(rhs, template, "rhs")
        added._labels = _on_grid(labels, template, "labels")
        added._scaling = _on_grid(scaling, template, "scaling")
        return added

    def _rebind(self, store: Store, fields: Mapping[Any, Any]) -> None:
        """
        Internal: hold ``store`` as the lhs of this constraint, its grid dims
        kept in the current order so the labels stay contiguous. Raises if
        ``store`` adds dims.
        """
        current = self._grid_template().dims
        extra = [d for d in store.dims if d not in current]
        if extra:
            raise ValueError(
                f"Constraint.update with an expression that adds dimensions {extra} is unsupported."
            )
        store = store.transpose([d for d in current if d in store.dims])
        self._bind(expressions.LinearExpression._from_store(store, self._model), fields)

    def _replace_store(self, store: Store) -> None:
        """Internal: hold ``store`` as the lhs, the fields kept. Sets ``_coef_dirty``."""
        self._rebind(store, self._fields())
        self._coef_dirty = True

    @staticmethod
    def _optional_on_grid(
        fields: Mapping[Any, Any], key: str, template: DataArray
    ) -> DataArray | None:
        value = fields.get(key)
        return None if value is None else _on_grid(value, template, key)

    def _grid_template(self) -> DataArray:
        """A DataArray on the lhs grid, to broadcast a field onto."""
        return self._grid["const"]

    def _fields(self) -> dict[str, DataArray]:
        """The fields next to the lhs that are set, in their Dataset order."""
        fields = {
            "sign": self._sign,
            "rhs": self._rhs,
            "labels": self._labels,
            "scaling": self._scaling,
            "dual": self._dual,
            "binary_var": self._binary_var,
            "binary_val": self._binary_val,
            "blocks": self._blocks,
        }
        return {k: v for k, v in fields.items() if v is not None}

    def _on_store(
        self, store: Store, model: Model, fields: Mapping[str, DataArray]
    ) -> Constraint:
        """A constraint with this one's attrs, the lhs held by ``store`` and ``fields``."""
        lhs = expressions.LinearExpression._from_store(store, model)
        return Constraint(lhs, model=model, name=self.name, attrs=self._attrs, **fields)

    def _copy(self, model: Model, deep: bool, include_solution: bool) -> Constraint:
        """A copy on ``model``, without the solve artifacts unless ``include_solution``."""
        keep = self._fields().keys() if include_solution else set(self.data_attrs)
        fields = {k: v.copy(deep=deep) for k, v in self._fields().items() if k in keep}
        return self._on_store(self._lhs._store.copy(model, deep), model, fields)

    def _isel(self, indexers: Mapping[Hashable, slice]) -> Constraint:
        """The cells at the positional slices ``indexers`` of grid dims."""
        fields = {k: v.isel(indexers) for k, v in self._fields().items()}
        return self._on_store(self._lhs._store.isel(indexers), self._model, fields)

    def __getitem__(
        self, selector: str | int | slice | list | tuple | dict
    ) -> Constraint:
        """
        Get selection from the constraint.
        Returns a Constraint with the selected data.
        """
        terms = self._lhs._dense_store("__getitem__").ds
        selected = DenseStore(
            Dataset({k: terms[k][selector] for k in terms}), self._model
        )
        fields = {k: v[selector] for k, v in self._fields().items()}
        store = model_store(selected, self._model)
        return self._on_store(store, self._model, fields)

    @property
    def data(self) -> Dataset:
        """
        The dense Dataset derived from the lhs and the fields, for a sparse
        lhs with a densify notice.
        """
        return self._dataset("data(read)")

    def _dataset(self, reason: Reason | None = None) -> Dataset:
        """:attr:`data`, with a densify notice naming ``reason`` if given."""
        terms = self._lhs._dense_store(reason).ds
        fields = {"coeffs": terms.coeffs, "vars": terms.vars, **self._fields()}
        return Dataset(fields, attrs=self._attrs)[list(fields)]

    @property
    def data_attrs(self) -> list[str]:
        """Data variables that define this constraint's persistent state."""
        base = list(Constraints.dataset_attrs)
        return base + ["binary_var", "binary_val"] if self.is_indicator else base

    @property
    def model(self) -> Model:
        return self._model

    @property
    def name(self) -> str:
        return self._attrs["name"]

    @property
    def attrs(self) -> dict[str, Any]:
        return self._attrs

    @property
    def coords(self) -> DatasetCoordinates:
        return self._grid.coords

    @property
    def indexes(self) -> Indexes:
        return self._grid.indexes

    @property
    def dims(self) -> Frozen[Hashable, int]:
        return self.sizes

    @property
    def sizes(self) -> Frozen[Hashable, int]:
        return Frozen({**self._grid_template().sizes, TERM_DIM: self.nterm})

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
    def coord_dims(self) -> tuple[Hashable, ...]:
        return tuple(k for k in self.dims if k not in HELPER_DIMS)

    @property
    def coord_sizes(self) -> dict[Hashable, int]:
        return {k: self.sizes[k] for k in self.coord_dims}

    @property
    def coord_names(self) -> list[str]:
        return get_dims_with_index_levels(self._grid, self.coord_dims)

    @property
    def type(self) -> str:
        """Get the type string of the constraint."""
        return "Constraint" if self.is_assigned else "Constraint (unassigned)"

    @property
    def term_dim(self) -> str:
        """Return the term dimension of the constraint."""
        return TERM_DIM

    @property
    def is_assigned(self) -> bool:
        return self._labels is not None

    @property
    def nterm(self) -> int:
        return self._lhs._store.nterm

    @property
    def ncons(self) -> int:
        """Get the number of active constraints (non-masked, with at least one valid variable)."""
        return int(self.active_row_mask().sum())

    @property
    def range(self) -> tuple[int, int]:
        """Return the range of the constraint."""
        return self._attrs["label_range"]

    @property
    def mask(self) -> DataArray | None:
        """
        Get the mask of the constraint.

        The mask indicates on which coordinates the constraint is enabled
        (True) and disabled (False).
        """
        if self.is_assigned:
            result: DataArray = self.labels != FILL_VALUE["labels"]  # type: ignore[assignment]
            return result.astype(bool)
        return None

    @property
    def loc(self) -> LocIndexer:
        return LocIndexer(self)

    @property
    def labels(self) -> DataArray:
        return DataArray([]) if self._labels is None else self._labels

    @property
    def coeffs(self) -> DataArray:
        return self.lhs.coeffs

    @property
    def vars(self) -> DataArray:
        return self.lhs.vars

    @property
    def sign(self) -> DataArray:
        return self._sign

    @property
    def rhs(self) -> DataArray:
        return self._rhs

    @property
    def scaling(self) -> DataArray:
        return self._scaling

    @scaling.setter
    def scaling(self, value: ConstantLike) -> None:
        scaling = broadcast_to_coords(
            value,
            coords=self.labels.coords,
            dims=self.coord_dims,
            label=f"scaling for constraint '{self.name}'",
        )
        scaling = validate_scaling(scaling, f"scaling for constraint '{self.name}'")
        self._scaling = _on_grid(scaling, self._grid_template(), "scaling")

    @property
    def is_indicator(self) -> bool:
        return self._binary_var is not None

    @property
    def binary_var(self) -> DataArray | None:
        return self._binary_var

    @property
    def binary_val(self) -> np.ndarray | None:
        return None if self._binary_val is None else self._binary_val.values

    @property
    def lhs(self) -> expressions.LinearExpression:
        """The lhs, a new expression on the store the constraint holds."""
        return expressions.LinearExpression._from_store(self._lhs._store, self._model)

    def __contains__(self, value: Any) -> bool:
        return value in ("coeffs", "vars", *self._fields()) or value in self.coords

    def __repr__(self) -> str:
        """Print the constraint arrays."""
        max_lines = options["display_max_rows"]
        dims = list(self.coord_dims)
        ndim = len(dims)
        dim_names = self.coord_names
        dim_sizes = list(self.coord_sizes.values())
        size = np.prod(dim_sizes)
        mask = self.mask
        masked_entries = (~mask).sum().values if mask is not None else 0
        store = self._lhs._store
        lines = []

        header_string = f"{self.type} `{self.name}`" if self.name else f"{self.type}"

        def row(indices: tuple[Any, ...]) -> str:
            coeffs, vars_, _ = store.cell(indices)
            expr = format_single_expression(coeffs, vars_, 0, self.model)
            sign = SIGNS_pretty[self.sign.values[indices]]
            return f"{expr} {sign} {self.rhs.values[indices]}"

        if size > 1 or ndim > 0:
            row_labels = get_printout_labels(self._grid, dims)
            for indices in generate_indices_for_printout(dim_sizes, max_lines):
                if indices is None:
                    lines.append("\t\t...")
                else:
                    coord = [row_labels[i][int(ind)] for i, ind in enumerate(indices)]
                    if mask is None or mask.values[indices]:
                        line = format_coord(coord) + f": {row(indices)}"
                    else:
                        line = format_coord(coord) + ": None"
                    lines.append(line)
            lines = align_lines_by_delimiter(lines, list(SIGNS_pretty.values()))

            shape_str = ", ".join(f"{d}: {s}" for d, s in zip(dim_names, dim_sizes))
            mask_str = f" - {masked_entries} masked entries" if masked_entries else ""
            underscore = "-" * (len(shape_str) + len(mask_str) + len(header_string) + 4)
            lines.insert(0, f"{header_string} [{shape_str}]{mask_str}:\n{underscore}")
        elif size == 1:
            lines.append(f"{header_string}\n{'-' * len(header_string)}\n{row(())}")
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

    def _flat(self) -> pd.DataFrame:
        ds = self._dataset("flat")

        def mask_func(data: dict) -> pd.Series:
            mask = (data["vars"] != -1) & (data["coeffs"] != 0)
            if "labels" in data:
                mask &= data["labels"] != -1
            return mask

        df = to_dataframe(ds, mask_func=mask_func)

        agg_custom = {k: "first" for k in list(df.columns)}
        agg_standards = dict(coeffs="sum", rhs="first", sign="first")
        agg = {**agg_custom, **agg_standards}
        df = df.groupby(["labels", "vars"], as_index=False).aggregate(agg)
        check_has_nulls(df, name=f"{self.type} {self.name}")
        return df

    def _assign_lhs(
        self,
        expr: expressions.LinearExpression,
        rhs: DataArray | None = None,
        reason: Reason | None = None,
    ) -> None:
        """
        Internal: replace the lhs by ``expr``, moving its constant part to
        the rhs and keeping a sparse lhs sparse where ``expr`` converts, else
        with a densify notice naming ``reason`` if given. Sets ``_coef_dirty``.
        """
        base_rhs = self.rhs if rhs is None else rhs
        fields = self._fields() | {"rhs": base_rhs - expr.const}
        store = _sparse_if(expr.reset_const()._store, self._lhs.is_sparse, reason)
        self._rebind(store, fields)
        self._coef_dirty = True

    def _assign_terms(
        self,
        coeffs: ConstantLike | None,
        variables: variables.Variable | None,
    ) -> None:
        """
        Internal: replace ``coeffs`` and/or ``vars`` of the dense lhs
        layout, a sparse lhs converted with a notice and back. Sets
        ``_coef_dirty``.
        """
        from linopy.variables import Variable

        store = self._lhs._store
        ds = store.to_dense("update(coeffs=, variables=)").ds[["coeffs", "vars"]]
        if coeffs is not None:
            new_coeffs = DataArray(coeffs).broadcast_like(ds.vars, exclude=[TERM_DIM])
            ds = assign_fields(ds, coeffs=new_coeffs)
        if variables is not None:
            if not isinstance(variables, Variable):
                raise TypeError(
                    "Constraint.update(variables=...) expects a Variable; "
                    f"got {type(variables).__name__}."
                )
            new_vars = variables.labels.broadcast_like(ds.coeffs, exclude=[TERM_DIM])
            ds = assign_fields(ds, vars=new_vars)
        dense = expressions.LinearExpression(ds, self._model)._store
        self._replace_store(
            _sparse_if(dense, self._lhs.is_sparse, "update(coeffs=, variables=)")
        )

    def update(
        self,
        constraint: ConstraintLike | None = None,
        *,
        lhs: ExpressionLike | VariableLike | ConstantLike | None = None,
        rhs: ExpressionLike | VariableLike | ConstantLike | None = None,
        sign: SignLike | None = None,
        coeffs: ConstantLike | None = None,
        variables: variables.Variable | None = None,
    ) -> Constraint:
        """
        Update the constraint in place.

        The only mutation API; setters forward here. Two call shapes:

        * ``c.update(x + 5 <= 3)`` — pass a complete constraint
          expression (mirroring ``add_constraints``). Replaces lhs,
          sign, and rhs at once.
        * ``c.update(lhs=, rhs=, sign=, coeffs=, variables=)`` — pass
          only what you want to change.

        Use the keyword form for targeted changes — it skips the
        unchanged attributes entirely. The positional form always
        rewrites lhs / sign / rhs (and flips ``_coef_dirty``), so it
        is the wrong shape for hot loops that only touch one part:

        .. code-block:: python

            # Hot loop, rhs is the only thing changing per iteration:
            for k in scenarios:
                c.update(rhs=rhs_k)  # ← targeted, cheap

            # Same loop written positionally rebuilds lhs every
            # iteration even though it never changes:
            for k in scenarios:
                c.update(big_lhs_expr <= rhs_k)  # ← avoid

        Parameters
        ----------
        constraint : ConstraintLike, optional
            A complete constraint expression (e.g. ``x + 5 <= 3``).
            Mutually exclusive with the keyword arguments below.
        lhs : ExpressionLike / VariableLike / ConstantLike, optional
            Replace the LHS expression. Any constant part is moved to
            ``rhs`` so ``c.lhs`` stays pure-variable. Cannot be combined
            with ``coeffs`` / ``variables``. Sets the internal
            ``_coef_dirty`` flag.
        rhs : ExpressionLike / VariableLike / ConstantLike, optional
            New right-hand side.

            * Constant rhs (scalar, array, DataArray) → assigned directly
              to ``c.rhs``; ``c.lhs`` is untouched.
            * Variable / Expression rhs → rearranged onto the lhs to
              preserve the invariant that ``c.rhs`` is constant-only,
              matching ``add_constraints``. **This rewrites ``c.lhs``.**

            Example — the two calls below produce the same final state::

                # Form A: explicit, only changes rhs
                c.update(rhs=5)

                # Form B: rhs carries a variable, so lhs is rewritten too.
                # Starting from `2*x <= 3`, this gives `2*x - y <= 5`:
                c.update(rhs=y + 5)

            If you want the rewrite to be loud, use the positional form
            (``c.update(2*x - y <= 5)``) which makes both sides explicit.
        sign : SignLike, optional
            New sign. One of ``"<=" / "==" / ">="`` (or their ``< > =``
            aliases).
        coeffs : ConstantLike, optional
            Replace coefficient values (same sparsity / term structure).
            Lower-level than ``lhs=``; sets ``_coef_dirty``.
        variables : Variable, optional
            Replace variable label array (same sparsity / term
            structure). Lower-level than ``lhs=``; sets ``_coef_dirty``.

        Returns
        -------
        Constraint
            ``self`` for chaining.
        """
        if constraint is not None:
            if any(x is not None for x in (lhs, rhs, sign, coeffs, variables)):
                raise TypeError(
                    "Constraint.update: positional `constraint` argument "
                    "cannot be combined with keyword arguments."
                )
            con: Constraint
            if isinstance(constraint, AnonymousScalarConstraint):
                con = constraint.to_constraint()
            elif isinstance(constraint, Constraint):
                con = constraint
            else:
                raise TypeError(
                    "Constraint.update: positional argument must be a "
                    "ConstraintLike (e.g. `x + 5 <= 3`); got "
                    f"{type(constraint).__name__}."
                )
            lhs, sign, rhs = con.lhs, con.sign, con.rhs

        if all(v is None for v in (lhs, rhs, sign, coeffs, variables)):
            return self

        if lhs is not None and (coeffs is not None or variables is not None):
            raise TypeError(
                "Constraint.update: pass either `lhs=` (replace the whole "
                "expression) or `coeffs=` / `variables=` (partial array "
                "replacement), not both."
            )

        if lhs is not None:
            expr = expressions.as_expression(
                lhs, self.model, coords=self.coords, dims=self.coord_dims
            )
            if isinstance(expr, expressions.QuadraticExpression):
                raise TypeError(
                    "Constraint.update: lhs must be linear; got a quadratic expression."
                )
            self._assign_lhs(expr, reason="update(lhs=)")

        if rhs is not None:
            expr = expressions.as_expression(
                rhs, self.model, coords=self.coords, dims=self.coord_dims
            )
            residual = expr.reset_const()
            if residual.nterm != 0:
                self._assign_lhs(self.lhs - residual, rhs=expr.const)
            else:
                self._rhs = _on_grid(expr.const, self._grid_template(), "rhs")

        if coeffs is not None or variables is not None:
            self._assign_terms(coeffs, variables)

        if sign is not None:
            new_sign = maybe_replace_signs(DataArray(sign)).broadcast_like(self.sign)
            self._sign = _on_grid(new_sign, self._grid_template(), "sign")

        return self

    def _assign_coords(self, **coords: Any) -> Constraint:
        """
        Reassign coordinate values on the constraint, keeping the shape.

        Internal: values-only replacement of existing dimension coordinates,
        used by :meth:`linopy.Model.assign_coords`. No relabeling, no
        reindexing, no shape change, and the order of the underlying data is
        preserved.
        """
        fields = reassign_coords(Dataset(self._fields()), **coords)
        self._rebind(self.lhs._assign_coords(**coords)._store, fields)
        return self

    @property
    @has_optimized_model
    def dual(self) -> DataArray:
        if self._dual is None:
            raise AttributeError(
                "Underlying is optimized but does not have dual values stored."
            )
        return self._dual

    @dual.setter
    def dual(self, value: ConstantLike) -> None:
        self._dual = _on_grid(value, self._grid_template(), "dual")

    def has_variable(self, variable: variables.Variable) -> bool:
        """
        Check if the constraint references any of the given variable labels.

        Masked variable entries (label -1) are ignored: they are not part of
        the model and would otherwise match every empty term slot.
        """
        return self.has_labels(assigned_labels(variable.labels))

    def has_labels(self, labels: np.ndarray) -> bool:
        """
        Check if the constraint references any of the given variable labels.

        ``labels`` must not contain the ``-1`` sentinel, see
        :func:`linopy.common.assigned_labels`.
        """
        return contains_labels(self._lhs._store.variable_labels(), labels)

    def _label_values(self) -> np.ndarray:
        """The flat labels; raises for a constraint without labels."""
        if self._labels is None:
            raise ValueError(f"Constraint {self.name!r} has not been assigned labels.")
        return self._labels.values.ravel()

    def to_matrix(
        self, label_index: VariableLabelIndex
    ) -> tuple[scipy.sparse.csr_array, np.ndarray]:
        """
        Construct a CSR matrix for this constraint.

        Only active (non-masked) rows are included. Column indices are dense
        positions in the active variable array, as given by ``label_index``.

        Returns
        -------
        csr : scipy.sparse.csr_array
            Shape (n_active_cons, n_active_vars).
        con_labels : np.ndarray
            Active constraint labels in row order.
        """
        active = self.active_row_mask()
        csr = self._lhs._store.positional_csr(label_index, active, self.name)
        return csr, self._label_values()[active]

    def active_row_mask(self) -> np.ndarray:
        """Boolean mask over raveled rows: label set and at least one variable present."""
        return (self._label_values() != -1) & self._lhs._store.has_terms()

    def active_labels(self) -> np.ndarray:
        """Active constraint labels in build order, without building the CSR."""
        return self._label_values()[self.active_row_mask()]

    def to_matrix_with_rhs(
        self, label_index: VariableLabelIndex
    ) -> tuple[scipy.sparse.csr_array, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (csr, con_labels, b, sense) in one pass.

        Avoids computing the CSR matrix twice when both the matrix and
        the RHS/sense vectors are needed.
        """
        active = self.active_row_mask()
        csr = self._lhs._store.positional_csr(label_index, active, self.name)
        con_labels = self._label_values()[active]
        b = self.rhs.values.ravel()[active]
        sense = self.sign.values.ravel()[active].astype("U1")
        return csr, con_labels, b, sense

    def sanitize_zeros(self) -> Constraint:
        """Remove terms with zero or near-zero coefficients."""
        self._replace_store(self._lhs._store.prune(1e-10))
        return self

    def sanitize_missings(self) -> Constraint:
        """Mask out rows where all variables are missing (-1)."""
        has_terms = self._lhs._store.dataarray(self._lhs._store.has_terms(), "labels")
        self._labels = self.labels.where(has_terms, -1)
        return self

    def sanitize_infinities(self) -> Constraint:
        """Mask out rows with invalid infinite RHS values."""
        valid_infinity_values = ((self.sign == LESS_EQUAL) & (self.rhs == np.inf)) | (
            (self.sign == GREATER_EQUAL) & (self.rhs == -np.inf)
        )
        self._labels = self.labels.where(~valid_infinity_values, -1)
        return self

    def to_dense(self) -> Constraint:
        """Hold the lhs on a dense store, in place."""
        self._replace_store(self._lhs._store.to_dense())
        return self

    def to_sparse(self) -> Constraint:
        """Hold the lhs on a sparse (CSR) store, in place; raises where it cannot convert."""
        self._replace_store(as_sparse(self._lhs._store))
        return self

    @property
    def slack(self) -> Slack | None:
        """
        Slack variable(s) added via :meth:`soften`, or ``None`` if the
        constraint has never been softened.
        """
        names = _slack_names(self.attrs)
        if names is None:
            return None
        positive, negative = names
        return Slack(
            positive=self.model.variables[positive],
            negative=self.model.variables[negative] if negative else None,
        )

    def soften(
        self,
        penalty: ConstantLike,
        *,
        max_violation: ConstantLike | None = None,
        name: str | None = None,
    ) -> Slack:
        """
        Soften a constraint, adding a slack variable and a penalty to the objective function.

        Parameters
        ----------
        penalty : constant-like
            The penalty that will match the slack variable inside the objective function. Must be bigger than 0.
        max_violation: constant-like
            The max violation possible that caps the slack (upper bound). If None, the slack will be unbounded.
        name: string
            The name for the slack variable. If None, it will reuse the constraint name and add a '_slack'.

        Returns
        -------
        Slack
            Named tuple with the `positive` slack variable, and the `negative` one for equality constraints (`None`
            for inequality constraints).

        Notes
        -----
        On a sparse lhs the slack terms are merged into the sparse rows, without densifying.
        A detached copy (from `.sel()`, `.isel()` or indexing) is not registered in
        model.constraints, so soften raises ValueError on it.

        Softening an already-softened constraint raises ValueError instead of stacking a second, redundant slack term
        onto the same lhs.

        Examples
        --------
        >>> from linopy import Model
        >>> import pandas as pd

        >>> m = Model()
        >>> investments = pd.Index(["A", "B", "C"], name="investments")
        >>> expected_return = pd.Series(
        ...     [0.08, 0.03, 0.1], index=investments, name="expected_return"
        ... )
        >>> w = m.add_variables(lower=0, upper=1, coords=[investments], name="weights")
        >>> m.add_objective((expected_return * w).sum(), sense="max")
        >>> budget_penalty = 2

        >>> budget_constraint = m.add_constraints(w.sum() == 1, name="budget")
        >>> slack = budget_constraint.soften(penalty=budget_penalty)
        """
        if not bool(np.all(np.asarray(penalty) > 0)):
            raise ValueError("Penalty is not positive.")

        model = self.model
        if model.objective.expression.empty:
            raise ValueError(
                "Objective must be defined via `model.add_objective` before calling `soften` on constraints."
            )

        if model.constraints.data.get(self.name) is not self:
            raise ValueError(
                f"Constraint {self.name!r} is not the constraint registered in the model, so "
                "`soften` would not affect it (it may be a detached copy from `.sel()`, "
                "`.isel()` or indexing). Call `soften` on `model.constraints[name]` "
                "directly."
            )

        existing_slack = self.slack
        if existing_slack is not None:
            raise ValueError(
                f"Constraint {self.name!r} was already softened (existing slack "
                f"variable {existing_slack.positive.name!r})"
            )

        sign_values = pd.unique(self.sign.values.ravel())
        if len(sign_values) > 1:
            raise NotImplementedError(
                "Constraint.soften does not support constraints with mixed signs."
            )
        sign = sign_values.item()

        name = name or f"{self.name}_slack"
        upper = np.inf if max_violation is None else max_violation

        def add_slack(suffix: str) -> variables.Variable:
            return model.add_variables(
                lower=0,
                upper=upper,
                coords=self.coords,
                mask=self.mask,
                name=f"{name}_{suffix}",
            )

        positive = add_slack("pos")
        slack = Slack(positive, add_slack("neg") if sign == EQUAL else None)
        self._attach_slack(slack, sign)

        violation = positive if slack.negative is None else positive + slack.negative
        direction = 1 if model.sense == "min" else -1
        model.objective += direction * (penalty * violation).sum()
        return slack

    def _attach_slack(self, slack: Slack, sign: str) -> None:
        """Add the slack terms to the lhs and record the slack variable names."""
        positive = slack.positive
        lhs = self.lhs + positive if sign == GREATER_EQUAL else self.lhs - positive
        if slack.negative is not None:
            lhs = lhs + slack.negative
        self.update(lhs=lhs)
        self._attrs.update(
            slack_positive=slack.positive.name,
            slack_negative=slack.negative.name if slack.negative is not None else "",
        )

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
        return cls(lhs, sign, rhs, model)

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
        cells, vars_, coeffs = self._lhs._store.terms()
        labels_flat = self._label_values()
        long = pl.DataFrame(
            {"labels": labels_flat[cells], "coeffs": coeffs, "vars": vars_}
        )
        del cells, vars_, coeffs
        long = filter_nulls_polars(long)
        if self.nterm > 1:
            long = maybe_group_terms_polars(long)
        check_has_nulls_polars(long, name=f"{self.type} {self.name}")

        mask = labels_flat != -1
        short_data: dict = {
            "labels": labels_flat[mask],
            "rhs": self.rhs.values.ravel()[mask],
        }
        sign_flat = self.sign.values.ravel()
        all_same_sign = len(sign_flat) > 0 and (
            sign_flat[0] == sign_flat[-1] and (sign_flat[0] == sign_flat).all()
        )
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

    def iterate_slices(
        self,
        slice_size: int | None = 10_000,
        slice_dims: list | None = None,
    ) -> Generator[Constraint, None, None]:
        """
        Yield sub-constraints along the largest of ``slice_dims``, each with at
        most about ``slice_size`` term slots, see
        :func:`linopy.common.slice_indexers`. The lhs store is sliced by
        :meth:`~linopy.store.Store.isel` and keeps its backing.
        """
        sizes = dict(self.sizes)
        dims = list(self.coord_dims) if slice_dims is None else slice_dims
        for indexers in slice_indexers(sizes, slice_size, dims):
            yield self._isel(indexers) if indexers else self

    assign = conwrap(assign_fields)
    assign_attrs = conwrap(Dataset.assign_attrs)
    assign_coords = conwrap(Dataset.assign_coords)
    broadcast_like = conwrap(Dataset.broadcast_like, exclude=[TERM_DIM])
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
class Constraints(LabelContainer[Constraint]):
    """
    A constraint container used for storing multiple constraint arrays.
    """

    _constraint_label_index: ConstraintLabelIndex | None = None

    dataset_attrs = ["labels", "coeffs", "vars", "sign", "rhs", "scaling"]

    @overload
    def __getitem__(self, names: str) -> Constraint: ...

    @overload
    def __getitem__(self, names: list[str]) -> Constraints: ...

    def __getitem__(self, names: str | list[str]) -> Constraint | Constraints:
        return super().__getitem__(names)

    def add(self, constraint: Constraint) -> Constraint:
        """
        Add a constraint to the constraints container.
        """
        if not constraint.is_assigned:
            raise ValueError(
                f"Constraint {constraint.name!r} has not been assigned labels; "
                "use `Model.add_constraints`."
            )
        self.data[constraint.name] = constraint
        self._invalidate_label_position_index()
        return constraint

    def _invalidate_label_position_index(self) -> None:
        super()._invalidate_label_position_index()
        if self._constraint_label_index is not None:
            self._constraint_label_index.invalidate()

    @property
    def label_index(self) -> ConstraintLabelIndex:
        """Index for O(1) label->position mapping and compact clabels array."""
        if self._constraint_label_index is None:
            self._constraint_label_index = ConstraintLabelIndex(self)
        return self._constraint_label_index

    @property
    def labels(self) -> Dataset:
        """
        Get the labels of all constraints.
        """
        return save_join(
            *[v.labels.rename(k) for k, v in self.items()],
            fill_dtype=self.model._dtypes["labels"],
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
            fill_dtype=self.model._dtypes["labels"],
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
    def scaling(self) -> Dataset:
        """
        Get the scaling factors of all constraints.
        """
        return save_join(*[v.scaling.rename(k) for k, v in self.items()])

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
        return sum(con.ncons for con in self.data.values() if not con.is_indicator)

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

    @property
    def indicator(self) -> Constraints:
        """
        Get the subset of constraints which are indicator constraints.
        """
        return self[[n for n, c in self.items() if c.is_indicator]]

    @property
    def regular(self) -> Constraints:
        """
        Get the subset of constraints which are not indicator constraints.
        """
        return self[[n for n, c in self.items() if not c.is_indicator]]

    def sanitize_zeros(self) -> None:
        """
        Filter out terms with zero and close-to-zero coefficient.
        """
        for con in self.data.values():
            con.sanitize_zeros()

    def sanitize_missings(self) -> None:
        """
        Set constraints labels to -1 where all variables in the lhs are
        missing.
        """
        for con in self.data.values():
            con.sanitize_missings()

    def sanitize_infinities(self) -> None:
        """
        Remove constraints whose RHS is an invalid infinity.

        Constraints with ``rhs == inf`` and sign ``<=``, or ``rhs == -inf``
        and sign ``>=``, are trivially satisfied and are masked out (label set
        to -1).
        """
        for con in self.data.values():
            con.sanitize_infinities()

    def format_labels(
        self, values: Sequence[int], display_max_terms: int | None = None
    ) -> str:
        """
        Get a string representation of a selection of constraint labels.

        Parameters
        ----------
        values : list, array-like
            One dimensional array of constraint labels.
        display_max_terms : int, optional
            Maximum number of terms to display per constraint. If ``None``,
            uses the global ``linopy.options.display_max_terms`` setting.

        Returns
        -------
        str
            String representation of the selected constraints.
        """
        with options as opts:
            if display_max_terms is not None:
                opts.set_value(display_max_terms=display_max_terms)
            res = [format_single_constraint(self.model, v) for v in values]

        return "\n".join(res)

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

        for constraint in self.data.values():
            res = xr.full_like(constraint.labels, N + 1, dtype=block_map.dtype)
            terms = constraint._lhs._store.to_dense("`set_blocks`").ds
            entries = replace_by_map(terms.vars, block_map)

            not_zero = entries != 0
            not_missing = entries != -1
            for n in range(N + 1):
                not_n = entries != n
                mask = not_n & not_zero & not_missing
                res = res.where(mask.any(constraint.term_dim), n)

            res = res.where(not_missing.any(constraint.term_dim), -1)
            res = res.where(not_zero.any(constraint.term_dim), 0)
            constraint._blocks = res

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
        dfs = [self[k]._flat() for k in self]
        if not len(dfs):
            return pd.DataFrame(columns=["coeffs", "vars", "labels", "key"])
        return self._flat_with_keys(dfs)

    def to_matrix(self) -> tuple[scipy.sparse.csr_array, np.ndarray]:
        """
        Construct a constraint matrix in sparse format by stacking per-constraint CSR matrices.

        Each per-constraint CSR has active constraint rows only and dense
        variable positions as column indices. Shape is
        ``(n_active_cons, n_active_vars)``.

        Returns
        -------
        matrix : scipy.sparse.csr_array
            Shape ``(n_active_cons, n_active_vars)``.
        con_labels : np.ndarray
            Shape ``(n_active_cons,)``, maps each matrix row to the
            original constraint label.
        """
        if not len(self):
            raise ValueError("No constraints available to convert to matrix.")

        label_index = self.model.variables.label_index
        csrs = []
        con_labels_list = []
        for c in self.data.values():
            if c.is_indicator:
                continue
            csr, con_labels = c.to_matrix(label_index)
            csrs.append(csr)
            con_labels_list.append(con_labels)
        if not csrs:
            raise ValueError("No constraints available to convert to matrix.")
        return scipy.sparse.vstack(csrs, format="csr"), np.concatenate(con_labels_list)

    def reset_dual(self) -> None:
        """
        Reset the stored solution of variables.
        """
        for c in self.data.values():
            c._dual = None


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
        expr_string = format_single_expression(
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
        model = self.lhs.model
        return Constraint(self.lhs.to_linexpr(), self.sign, self.rhs, model)
