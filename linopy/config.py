#!/usr/bin/env python3
"""
Created on Fri Jan 13 12:57:45 2023.

@author: fabian
"""

from __future__ import annotations

from typing import Any

SPARSE_DEPRECATION = "is deprecated and will be removed; use Model(sparse=True) instead"
SEMANTICS_DEPRECATION = (
    'linopy.options["semantics"] is deprecated and has no effect: the v1 '
    "convention is the only behaviour. Drop the setting; it is removed in "
    "linopy 1.1."
)
SEMANTICS_MIGRATION = (
    "The legacy semantics were removed in linopy 1.0 and v1 is the only "
    "behaviour. See https://linopy.readthedocs.io/en/latest/migrating-to-v1.html "
    "to migrate, or pin linopy<1 to keep the legacy behaviour."
)


class LinopySemanticsWarning(FutureWarning):
    """
    Deprecated, no longer emitted; removed in 1.1.

    Kept so that existing warning filters referencing it keep working.
    """


class OptionSettings:
    """Runtime configuration knobs (e.g. display widths). Use as a context manager, which restores the values active on entry on exit, or set values directly via ``options(key=value)``."""

    def __init__(self, **kwargs: Any) -> None:
        self._defaults = kwargs
        self._current_values = kwargs.copy()
        self._stack: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.set_value(**kwargs)

    def __getitem__(self, key: str) -> Any:
        return self.get_value(key)

    def __setitem__(self, key: str, value: Any) -> None:
        return self.set_value(**{key: value})

    def set_value(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k == "semantics":
                _set_semantics(v)
                continue
            if k not in self._defaults:
                raise KeyError(f"{k} is not a valid setting.")
            if k == "sparse_groupby" and v:
                from linopy.semantics import warn_outside_linopy

                warn_outside_linopy(
                    f'linopy.options["sparse_groupby"] {SPARSE_DEPRECATION}.',
                    FutureWarning,
                )
            self._current_values[k] = v

    def get_value(self, name: str) -> Any:
        if name == "semantics":
            _warn_semantics()
            return "v1"
        if name in self._defaults:
            return self._current_values[name]
        else:
            raise KeyError(f"{name} is not a valid setting.")

    def reset(self) -> None:
        self._current_values = self._defaults.copy()

    def __enter__(self) -> OptionSettings:
        self._stack.append(self._current_values.copy())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        self._current_values = self._stack.pop()

    def __repr__(self) -> str:
        settings = "\n ".join(
            f"{name}={value}" for name, value in self._current_values.items()
        )
        return f"OptionSettings:\n {settings}"


def _warn_semantics() -> None:
    from linopy.semantics import warn_outside_linopy

    warn_outside_linopy(SEMANTICS_DEPRECATION, FutureWarning)


def _set_semantics(value: Any) -> None:
    if value != "v1":
        raise ValueError(f"Invalid semantics: {value!r}. {SEMANTICS_MIGRATION}")
    _warn_semantics()


options = OptionSettings(
    display_max_rows=14,
    display_max_terms=6,
    sparse_groupby=False,
    warn_on_densify=None,
)
