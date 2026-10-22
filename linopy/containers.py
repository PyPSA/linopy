"""
Shared base for the Variables and Constraints containers.
"""

from __future__ import annotations

from collections.abc import ItemsView, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, Self, TypeVar, overload

import numpy as np
import pandas as pd
from xarray import DataArray
from xarray.core.coordinates import DatasetCoordinates

from linopy.common import (
    LabelPositionIndex,
    format_string_as_variable_name,
    get_label_position,
)

if TYPE_CHECKING:
    from linopy.model import Model


class LabeledItem(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def labels(self) -> DataArray: ...

    @property
    def coords(self) -> DatasetCoordinates: ...

    @property
    def flat(self) -> pd.DataFrame: ...


T = TypeVar("T", bound=LabeledItem)


@dataclass(repr=False)
class LabelContainer(Generic[T]):
    data: dict[str, T]
    model: Model
    _label_position_index: LabelPositionIndex | None = None

    def _formatted_names(self) -> dict[str, str]:
        """
        Get a dictionary of formatted names to the proper item names.
        This map enables a attribute like accession of item names which
        are not valid python variable names.
        """
        return {format_string_as_variable_name(n): n for n in self}

    @overload
    def __getitem__(self, names: str) -> T: ...

    @overload
    def __getitem__(self, names: list[str]) -> Self: ...

    def __getitem__(self, names: str | list[str]) -> T | Self:
        if isinstance(names, str):
            return self.data[names]
        return type(self)({name: self.data[name] for name in names}, self.model)

    def __getattr__(self, name: str) -> T:
        if name in self.data:
            return self.data[name]
        if name in (formatted_names := self._formatted_names()):
            return self.data[formatted_names[name]]
        raise AttributeError(
            f"{type(self).__name__} has no attribute `{name}` or the attribute is not accessible / raises an error."
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

    def _item_suffix(self, item: T) -> str:
        return ""

    def _format_items(self, exclude: set[str] | None = None) -> str:
        """Format items, optionally excluding names in a group."""
        r = ""
        count = 0
        for name, item in self.items():
            if exclude and name in exclude:
                continue
            count += 1
            coords = (
                " (" + ", ".join(str(coord) for coord in item.coords) + ")"
                if item.coords
                else ""
            )
            r += f" * {name}{coords}{self._item_suffix(item)}\n"
        if count == 0:
            r += "<empty>\n"
        return r

    def __repr__(self) -> str:
        r = f"linopy.model.{type(self).__name__}"
        line = "-" * len(r)
        r += f"\n{line}\n"
        r += self._format_items()
        return r

    def __len__(self) -> int:
        return self.data.__len__()

    def __iter__(self) -> Iterator[str]:
        return self.data.__iter__()

    def items(self) -> ItemsView[str, T]:
        return self.data.items()

    def _ipython_key_completions_(self) -> list[str]:
        """
        Provide method for the key-autocompletions in IPython.

        See
        http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
        For the details.
        """
        return list(self)

    def remove(self, name: str) -> None:
        """
        Remove item `name` from the container.
        """
        self.data.pop(name)
        self._invalidate_label_position_index()

    def _invalidate_label_position_index(self) -> None:
        """Invalidate the label position index cache."""
        if self._label_position_index is not None:
            self._label_position_index.invalidate()

    def _ensure_label_position_index(self) -> LabelPositionIndex:
        if self._label_position_index is None:
            self._label_position_index = LabelPositionIndex(self)
        return self._label_position_index

    def get_name_by_label(self, label: int | float) -> str:
        """
        Get the name of the item containing the passed label.

        Parameters
        ----------
        label : int
            Integer label within the range [0, MAX_LABEL] where MAX_LABEL is the last assigned
            label.

        Raises
        ------
        ValueError
            If label is not contained by any item.

        Returns
        -------
        name : str
            Name of the containing item.
        """
        if not isinstance(label, float | int | np.integer) or label < 0:
            raise ValueError("Label must be a positive number.")
        for name, item in self.items():
            if label in item.labels:
                return name
        kind = type(self).__name__[:-1].lower()
        raise ValueError(f"No {kind} found containing the label {label}.")

    def get_label_position(self, values: int | np.ndarray) -> Any:
        """
        Get tuple of name and coordinate for labels.

        Uses an optimized O(log n) binary search implementation with a cached index.
        """
        return get_label_position(self, values, self._ensure_label_position_index())

    def _flat_with_keys(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(dfs, ignore_index=True)
        unique_labels = df.labels.unique()
        map_labels = pd.Series(
            np.arange(len(unique_labels), dtype=self.model._dtypes["labels"]),
            index=unique_labels,
        )
        df["key"] = df.labels.map(map_labels)
        return df
