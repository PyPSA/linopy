"""
Linopy declarative math helper-functions module.

This module contains:
- the helper functions that can be called in declarative math `mask` and `expression` strings using their `NAME`;
- the abstract base class from which users can define their own helpers, and;
- the registry builder that makes them available to a model build.

This module is adapted from the calliope Apache-2.0 licensed helper function module:
- https://github.com/calliope-project/calliope/blob/9916116a06ec8c1feaf3c2606bdb8941b916ce85/src/calliope/backend/helper_functions.py
"""

from __future__ import annotations

import functools
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import xarray as xr

from linopy.declarative.schema import DTYPE_OPTIONS, MathModel
from linopy.expressions import LinearExpression

if TYPE_CHECKING:
    from linopy.declarative.nodes import Context

KIND_T = Literal["mask", "expression"]


def dim_iterator(math: MathModel, dim: str) -> str:
    """
    Return the LaTeX iterator name of dimension `dim`.

    Falls back to the dimension name itself when the dimension is not declared
    in the math or declares no iterator.
    """
    if dim in math.dimensions.root:
        iterator = math.dimensions[dim].iterator
        if iterator != "NEEDS_ITERATOR":
            return iterator
    return dim


def _to_str_list(vals: Any) -> list[str]:
    """Force `vals` to a list of strings, extracting names from any DataArray items."""
    if not isinstance(vals, list):
        vals = [vals]
    return [str(i.name) if isinstance(i, xr.DataArray) else str(i) for i in vals]


def _update_iterator(
    instring: str,
    iterator_converter: dict[str, str],
    method: Literal["add", "replace"],
) -> str:
    r"""
    Update iterators in the iterator substring of a LaTeX component string.

    Find an iterator in the iterator substring of the component string (anything
    wrapped in `_\text{}`, e.g. the standalone `foo` in
    `\textit{my_param}_\text{bar,foo,foo=bar,foo+1}`) and append to it
    (`method="add"`) or replace it (`method="replace"`).

    Parameters
    ----------
    instring : str
        String in which the iterator substring can be found.
    iterator_converter : dict[str, str]
        Mapping from the iterator to search for to the string to append/replace.
    method : Literal["add", "replace"]
        Whether to append to the iterator or replace it entirely.
    """

    def _replace_in_iterator(matched: re.Match) -> str:
        new_iterators = []
        for it in matched.group(2).split(","):
            if it in iterator_converter:
                it = (
                    it + iterator_converter[it]
                    if method == "add"
                    else iterator_converter[it]
                )
            new_iterators.append(it)
        return matched.group(1) + ",".join(new_iterators) + matched.group(3)

    return re.sub(r"(_\\text{)([^{}]*?)(})", _replace_in_iterator, instring)


class HelperFunction(ABC):
    """
    Abstract base class of all declarative math helper functions.

    Subclasses must define the class attributes `NAME` (the function name used in
    math strings) and `ALLOWED_IN` (the string types the function can be called
    from), and implement :meth:`as_math_string` and :meth:`as_raw`. To make a
    custom helper available to a model build, pass the subclass to
    :func:`linopy.declarative.build.declarative_model` via its `helpers` argument.
    """

    NAME: ClassVar[str]
    """Helper function name used in math `mask`/`expression` strings."""

    ALLOWED_IN: ClassVar[list[KIND_T]]
    """The parseable string types this function can be called from."""

    ignore_mask: ClassVar[bool] = False
    """If True, `mask` arrays are not applied to the function's incoming arguments."""

    def __init__(self, context: Context) -> None:
        """
        Initialise the helper.

        Parameters
        ----------
        context : Context
            The evaluation context (input data, math definition, config, ...).
        """
        self._context = context

    @abstractmethod
    def as_math_string(self, *args: Any, **kwargs: Any) -> str:
        """Return a LaTeX math string that includes the action applied by this function."""

    @abstractmethod
    def as_raw(self, *args: Any, **kwargs: Any) -> LinearExpression | xr.DataArray:
        """Apply the helper function to its evaluated argument arrays."""

    def as_expr(self, *args: Any, **kwargs: Any) -> LinearExpression | xr.DataArray:
        """
        Apply the helper function on the expression route.

        By default this delegates to :meth:`as_raw`; helpers that need distinct
        behaviour on the expression route (e.g. to return a linopy expression)
        should override it.
        """
        return self.as_raw(*args, **kwargs)

    def _dim_iterator(self, dim: str) -> str:
        """Return the LaTeX iterator name of dimension `dim`."""
        return dim_iterator(self._context.math, dim)

    def _instr(self, dim: str) -> str:
        r"""Return the LaTeX "iterator in dimension" string (e.g. `\text{n} \in \text{node}`)."""
        return rf"\text{{{self._dim_iterator(dim)}}} \in \text{{{dim}}}"

    def _get_dims_from_iterators(self, instring: str) -> list[str]:
        """Return the dimensions whose iterators appear in a LaTeX component string."""
        math = self._context.math

        def _extract_dims(matched: re.Match) -> str:
            return ",".join(
                dim_name
                for i in matched.group(2).split(",")
                for dim_name in math.dimensions.root
                if dim_iterator(math, dim_name) == i
            )

        dims = re.sub(r"^.*(_\\text{)([^{}]*?)(})", _extract_dims, instring)
        return [dim for dim in dims.split(",") if dim]


class MaskAny(HelperFunction):
    """Apply `any` over dimension(s) in a `mask` string."""

    # Class name doesn't match NAME to avoid a clash with typing.Any
    NAME = "any"
    ALLOWED_IN: ClassVar[list[KIND_T]] = ["mask"]

    def as_math_string(  # noqa: D102, override
        self, array: str, *, over: str | list[str | xr.DataArray]
    ) -> str:
        overstring = r" \\ ".join(self._instr(i) for i in _to_str_list(over))
        # Using bigvee for "collective-or"
        return rf"\bigvee\limits_{{\substack{{{overstring}}}}} ({array})"

    def as_raw(
        self, input_component: xr.DataArray, *, over: xr.DataArray | list[xr.DataArray]
    ) -> xr.DataArray:
        """
        Reduce a boolean mask array by applying `any` over some dimension(s).

        Parameters
        ----------
        input_component : xr.DataArray
            Boolean array to reduce.
        over : xr.DataArray | list[xr.DataArray]
            Dimension(s) over which to apply `any`.

        Returns
        -------
        xr.DataArray
            Boolean array with dimensions reduced by applying a boolean OR
            operation along the dimensions given in `over`.
        """
        if input_component.dtype.kind != "b":
            raise ValueError(
                "Input to `any` must be a boolean array. "
                f"Received {input_component.name} of dtype {input_component.dtype}"
            )
        available_dims = set(input_component.dims).intersection(_to_str_list(over))
        return input_component.any(dim=available_dims, keep_attrs=True)


class Sum(HelperFunction):
    """Apply a summation over dimension(s) in math expressions."""

    NAME = "sum"
    ALLOWED_IN: ClassVar[list[KIND_T]] = ["expression", "mask"]

    def as_math_string(  # noqa: D102, override
        self, array: str, *, over: str | list[str | xr.DataArray]
    ) -> str:
        overstring = r" \\ ".join(self._instr(i) for i in _to_str_list(over))
        return rf"\sum\limits_{{\substack{{{overstring}}}}} ({array})"

    def as_raw(
        self, array: xr.DataArray, *, over: xr.DataArray | list[xr.DataArray]
    ) -> xr.DataArray:
        """
        Sum an expression array over the given dimension(s).

        Parameters
        ----------
        array : xr.DataArray
            Expression array.
        over : xr.DataArray | list[xr.DataArray]
            Dimension(s) over which to apply `sum`.

        Returns
        -------
        xr.DataArray
            Array with dimensions reduced by summing over the dimensions given
            in `over`; dimensions not present in `array` are ignored.
        """
        filtered_over = set(_to_str_list(over)).intersection(array.dims)
        return array.sum(filtered_over)


class SelectFromLookupArrays(HelperFunction):
    """N-dimensional vectorised indexing via lookup arrays."""

    NAME = "select_from_lookup_arrays"
    ALLOWED_IN: ClassVar[list[KIND_T]] = ["expression"]

    def as_math_string(self, array: str, **lookup_arrays: str) -> str:  # noqa: D102, override
        new_strings = {
            (iterator := self._dim_iterator(dim)): rf"={array}[{iterator}]"
            for dim, array in lookup_arrays.items()
        }
        return _update_iterator(array, new_strings, "add")

    def as_raw(
        self, array: xr.DataArray, **lookup_arrays: xr.DataArray
    ) -> xr.DataArray:
        """
        Apply vectorised indexing on an arbitrary number of an input array's dimensions.

        Parameters
        ----------
        array : xr.DataArray
            Array on which to apply vectorised indexing.
        **lookup_arrays : xr.DataArray
            Keys are dimensions on which to apply vectorised indexing; values are
            arrays whose values are either NaN or values from that dimension.

        Returns
        -------
        xr.DataArray
            `array` with rearranged values (coordinates remain unchanged).
            Any NaN index coordinates in the lookup arrays will be NaN in the
            returned array.

        Raises
        ------
        ValueError
            If `array` or any lookup array is not indexed over all the
            dimensions given in the `lookup_arrays` keys.
        """
        # Inspired by https://github.com/pydata/xarray/issues/1553#issuecomment-748491929
        # Reindex does not presently support vectorized lookups: https://github.com/pydata/xarray/issues/1553
        # Sel does (e.g. https://github.com/pydata/xarray/issues/4630) but can't handle missing keys
        dims = set(lookup_arrays.keys())
        missing_dims_in_component = dims.difference(array.dims)
        missing_dims_in_lookup_tables = any(
            dim not in lookup.dims for dim in dims for lookup in lookup_arrays.values()
        )
        if missing_dims_in_component:
            raise ValueError(
                f"Cannot select items from `{array.name}` on the dimensions {dims} "
                f"since the array is not indexed over the dimensions {missing_dims_in_component}"
            )
        if missing_dims_in_lookup_tables:
            raise ValueError(
                f"All lookup arrays used to select items from `{array.name}` "
                f"must be indexed over the dimensions {dims}"
            )

        dim = "dim_0"
        ixs = {}
        masks = []

        # Turn string lookup values to numeric ones.
        # We stack the dimensions to handle multidimensional lookups
        for index_dim, index in lookup_arrays.items():
            stacked_lookup = self._context.input_data[index.name].stack(
                {dim: tuple(dims)}
            )
            ix = array.indexes[index_dim].get_indexer(stacked_lookup)
            if (ix == -1).all():
                received_lookup = (
                    self._context.input_data[index.name].to_series().dropna()
                )
                raise IndexError(
                    f"Trying to select items on the dimension {index_dim} from the "
                    f"{index.name} lookup array, but no matches found. Received: {received_lookup}"
                )
            ixs[index_dim] = xr.DataArray(
                np.fmax(0, ix), coords={dim: stacked_lookup[dim]}
            )
            masks.append(ix >= 0)

        # Nullify any lookup values that are not given (i.e., are NaN in the lookup array)
        mask = functools.reduce(lambda x, y: x & y, masks)
        result = array[ixs]
        if not mask.all():
            result[{dim: ~mask}] = np.nan
        return result.drop_vars(dims).unstack(dim)


class GetValAtIndex(HelperFunction):
    """Get the value of a dimension at a given integer index."""

    NAME = "get_val_at_index"
    ALLOWED_IN: ClassVar[list[KIND_T]] = ["expression", "mask"]

    def as_math_string(self, **dim_idx_mapping: str) -> str:  # noqa: D102, override
        dim, idx = self._mapping_to_dim_idx(**dim_idx_mapping)
        return f"{dim}[{idx}]"

    def as_raw(self, **dim_idx_mapping: int) -> xr.DataArray:
        """
        Get the value of a model dimension at a given integer index.

        This function is primarily useful for timeseries data, e.g.
        `get_val_at_index(snapshot=0)` is the first snapshot.

        Parameters
        ----------
        **dim_idx_mapping : int
            A single mapping from a model dimension name to the (zero-indexed)
            integer index of the value to extract.

        Returns
        -------
        xr.DataArray
            Dimensionless array containing one value.
        """
        dim, idx = self._mapping_to_dim_idx(**dim_idx_mapping)
        return self._context.input_data.coords[dim][int(idx)]

    @staticmethod
    def _mapping_to_dim_idx(**dim_idx_mapping: Any) -> tuple[str, Any]:
        if len(dim_idx_mapping) != 1:
            raise ValueError("Supply one (and only one) dimension:index mapping")
        return next(iter(dim_idx_mapping.items()))


class Roll(HelperFunction):
    """Roll (a.k.a. shift) items along ordered dimensions."""

    NAME = "roll"
    ALLOWED_IN: ClassVar[list[KIND_T]] = ["expression"]
    ignore_mask = True

    def as_math_string(self, array: str, **roll_kwargs: str) -> str:  # noqa: D102, override
        new_strings = {
            self._dim_iterator(k): f"{-1 * int(v):+d}" for k, v in roll_kwargs.items()
        }
        return _update_iterator(array, new_strings, "add")

    def as_raw(self, array: xr.DataArray, **roll_kwargs: int) -> xr.DataArray:
        """
        Roll the array along the given dimension(s) by the given number of places.

        Rolling keeps the array index labels in the same position, but moves the
        data by the given number of places, e.g. `roll(storage_level, snapshot=1)`
        aligns each snapshot with the previous snapshot's storage level.

        Parameters
        ----------
        array : xr.DataArray
            Array on which to roll data.
        **roll_kwargs : int
            Keys are dimension names on which to roll; values are the number of
            places to roll data.

        Returns
        -------
        xr.DataArray
            `array` with rolled data.
        """
        roll_kwargs_int: Mapping = {k: int(v) for k, v in roll_kwargs.items()}
        return array.roll(roll_kwargs_int)


class Mask(HelperFunction):
    """Apply a boolean condition to an array _within_ an expression string."""

    NAME = "mask"
    ALLOWED_IN: ClassVar[list[KIND_T]] = ["expression"]

    def as_math_string(self, array: str, condition: str) -> str:  # noqa: D102, override
        return rf"({array} \text{{if }} {condition} == True)"

    def as_raw(self, array: xr.DataArray, condition: xr.DataArray) -> xr.DataArray:
        """
        Apply a mask condition to a math array within an expression string.

        For example, `sum(mask(flow_cap, node_grouping), over=node)` sums only
        the group members flagged in a boolean `node_grouping` array.

        Parameters
        ----------
        array : xr.DataArray
            Math component array.
        condition : xr.DataArray
            Boolean mask array. If not `bool` dtype, NaN and 0 are taken as False
            and all other values as True.

        Returns
        -------
        xr.DataArray
            The input array with the condition applied, broadcast across any new
            dimensions provided by the condition.
        """
        return array.where(condition.fillna(False).astype(bool))


class GroupSum(HelperFunction):
    """Apply a summation over an arbitrary array grouping."""

    NAME = "group_sum"
    ALLOWED_IN: ClassVar[list[KIND_T]] = ["expression"]
    ignore_mask = True

    def as_math_string(self, array: str, groupby: str, group_dim: str) -> str:  # noqa: D102, override
        group_dim_singular = self._dim_iterator(group_dim)
        sum_lim_string = rf"\text{{ if }} {groupby} = \text{{{group_dim_singular}}}"
        over = [self._instr(i) for i in self._get_dims_from_iterators(groupby)]
        foreach_string = r" \\ ".join([*over, sum_lim_string])
        return rf"\sum\limits_{{\substack{{{foreach_string}}}}} ({array})"

    def as_raw(
        self, array: xr.DataArray, groupby: xr.DataArray, group_dim: xr.DataArray
    ) -> xr.DataArray:
        """
        Sum an array over the given groupings.

        For example, `group_sum(p * sign, bus, node)` sums the per-component
        `p * sign` expression into per-`node` totals using the `bus` lookup.

        Parameters
        ----------
        array : xr.DataArray
            Expression array.
        groupby : xr.DataArray
            Array with which to group `array`; all dimensions over which it is
            indexed are replaced by `group_dim` in the result.
        group_dim : xr.DataArray
            Dimension that the `groupby` values are members of. This becomes a
            new dimension over which the result is indexed.

        Returns
        -------
        xr.DataArray
            Array with dimension(s) aggregated over the `groupby`.

        See Also
        --------
        GroupDatetime : grouping over datetime periods without a separate `groupby` array.
        """
        grouping_dims = groupby.dims
        groups = array.stack(_stacked=grouping_dims).groupby(
            groupby.rename(group_dim.name).stack(_stacked=grouping_dims)
        )
        return groups.sum("_stacked")


class GroupDatetime(HelperFunction):
    """Apply a summation over a datetime group on a datetime dimension."""

    NAME = "group_datetime"
    ALLOWED_IN: ClassVar[list[KIND_T]] = ["expression"]
    ignore_mask = True

    def as_math_string(self, array: str, over: str, group: str) -> str:  # noqa: D102, override
        overstring = self._instr(over)
        foreach_string = (
            rf"{overstring} \text{{ if }} \text{{{group}}}"
            rf"(\text{{{self._dim_iterator(over)}}}) = \text{{{self._dim_iterator(group)}}}"
        )
        return rf"\sum\limits_{{\substack{{{foreach_string}}}}} ({array})"

    def as_raw(
        self, array: xr.DataArray, over: xr.DataArray, group: xr.DataArray
    ) -> xr.DataArray:
        """
        Sum an array over a datetime grouping of a datetime dimension.

        For example, `group_datetime(flow_in, snapshot, month) <= max_monthly`
        constrains the monthly sum of a timeseries variable.

        Parameters
        ----------
        array : xr.DataArray
            Expression array.
        over : xr.DataArray
            Datetime dimension over which to group.
        group : xr.DataArray
            Datetime grouper dimension; any xarray/pandas datetime accessor name
            ('date', 'dayofweek', 'month', ...). The `over` dimension is replaced
            by this dimension in the result.

        Returns
        -------
        xr.DataArray
            Array with the datetime dimension aggregated over the grouper.
        """
        group_name = str(group.name)
        dtype = DTYPE_OPTIONS[self._context.math.dimensions[group_name].dtype]
        group_sum_helper = GroupSum(self._context)
        return group_sum_helper.as_raw(
            array, getattr(array[str(over.name)].dt, group_name).astype(dtype), group
        )


class SumNextN(HelperFunction):
    """
    Sum the current and next N items in an array.

    Works best for ordered arrays (datetime, integer) and is equivalent to a
    summation over a rolling window.
    """

    NAME = "sum_next_n"
    ALLOWED_IN: ClassVar[list[KIND_T]] = ["expression"]

    def as_math_string(self, array: str, over: str, N: int) -> str:  # noqa: D102, override
        over_singular = rf"\text{{{self._dim_iterator(over)}}}"
        new_iterator = over[0]
        updated_iterator_array = _update_iterator(
            array, {self._dim_iterator(over): new_iterator}, "replace"
        )
        return (
            rf"\sum\limits_{{\text{{{new_iterator}}}={over_singular}}}"
            rf"^{{{over_singular}+{N}}} ({updated_iterator_array})"
        )

    def as_raw(self, array: xr.DataArray, over: xr.DataArray, N: int) -> xr.DataArray:
        """
        Sum values from the current up to N-from-current position on a dimension.

        For example, `sum_next_n(flow_in, snapshot, 4) == sum_next_n(demand, snapshot, 4)`
        requires flexible demand to be met within a 4-snapshot window.

        Parameters
        ----------
        array : xr.DataArray
            Math component array.
        over : xr.DataArray
            Dimension over which to sum.
        N : int
            Number of items beyond the current value to sum over.

        Returns
        -------
        xr.DataArray
            Array of rolling-window sums, indexed as `array`.

        Notes
        -----
        The rolling window does not wrap around to the start of the dimension when
        reaching the end, so the final N items sum over progressively shorter
        windows. This over-constrains a model unless the constraint is limited
        (using its `mask` string) to the first `len(over) - N` items, e.g.
        `mask: snapshot<=get_val_at_index(snapshot=-4)` if N == 4.
        """
        # We cannot use the xarray rolling window method as it doesn't like
        # operating on Python objects, which our optimisation problem components are.
        results: list[xr.DataArray] = []
        for i in range(len(over)):
            results.append(
                array.isel({str(over.name): slice(i, i + int(N))}).sum(
                    str(over.name), min_count=1
                )
            )
        return xr.concat(results, dim=over).broadcast_like(array)


BUILTIN_HELPERS: tuple[type[HelperFunction], ...] = (
    MaskAny,
    Sum,
    SelectFromLookupArrays,
    GetValAtIndex,
    Roll,
    Mask,
    GroupSum,
    GroupDatetime,
    SumNextN,
)
"""The helper functions shipped with linopy."""


def build_registry(
    extra: Iterable[type[HelperFunction]] = (),
) -> dict[KIND_T, dict[str, type[HelperFunction]]]:
    """
    Return a helper-function registry of the built-in helpers plus any `extra` ones.

    Parameters
    ----------
    extra : Iterable[type[HelperFunction]], optional
        User-defined :class:`HelperFunction` subclasses to add to the registry.

    Returns
    -------
    dict[Literal["mask", "expression"], dict[str, type[HelperFunction]]]
        Per string type, a mapping from helper `NAME` to helper class.

    Raises
    ------
    ValueError
        If an entry is not a :class:`HelperFunction` subclass, is missing
        `NAME`/`ALLOWED_IN`, or clashes with an already-registered name.
    """
    registry: dict[KIND_T, dict[str, type[HelperFunction]]] = {
        "mask": {},
        "expression": {},
    }
    for cls in (*BUILTIN_HELPERS, *extra):
        if not (isinstance(cls, type) and issubclass(cls, HelperFunction)):
            raise ValueError(
                "Helper function must be subclassed from "
                f"linopy.declarative.helpers.HelperFunction: {cls}"
            )
        name = getattr(cls, "NAME", None)
        allowed_in = getattr(cls, "ALLOWED_IN", None)
        if not isinstance(name, str) or not allowed_in:
            raise ValueError(
                f"Helper function {cls.__name__} must define `NAME` and `ALLOWED_IN`"
            )
        for kind in allowed_in:
            if name in registry[kind]:
                raise ValueError(
                    f"`{kind}` string helper function `{name}` already exists"
                )
            registry[kind][name] = cls
    return registry
