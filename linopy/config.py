#!/usr/bin/env python3
"""
Created on Fri Jan 13 12:57:45 2023.

@author: fabian
"""

from __future__ import annotations

from typing import Any

LEGACY_SEMANTICS = "legacy"
V1_SEMANTICS = "v1"
VALID_SEMANTICS = {LEGACY_SEMANTICS, V1_SEMANTICS}

LEGACY_SEMANTICS_MESSAGE = (
    "The 'legacy' semantics are deprecated and will be removed in "
    "linopy 1.0. Set linopy.options['semantics'] = 'v1' to opt in "
    "to the new behaviour, or silence this warning with:\n"
    "  import warnings; warnings.filterwarnings("
    "'ignore', category=LinopySemanticsWarning)"
)


class LinopySemanticsWarning(FutureWarning):
    """
    Emitted when code runs under the legacy arithmetic semantics.

    Subclasses ``FutureWarning`` rather than ``DeprecationWarning`` so it is
    shown to end users by default; the legacy-to-v1 transition changes
    results, not just an API surface.
    """


class OptionSettings:
    """Runtime configuration knobs (e.g. display widths). Use as a context manager or set values directly via ``options(key=value)``."""

    def __init__(self, **kwargs: Any) -> None:
        self._defaults = kwargs
        self._current_values = kwargs.copy()

    def __call__(self, **kwargs: Any) -> None:
        self.set_value(**kwargs)

    def __getitem__(self, key: str) -> Any:
        return self.get_value(key)

    def __setitem__(self, key: str, value: Any) -> None:
        return self.set_value(**{key: value})

    def set_value(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k not in self._defaults:
                raise KeyError(f"{k} is not a valid setting.")
            if k == "semantics" and v not in VALID_SEMANTICS:
                raise ValueError(
                    f"Invalid semantics: {v!r}. "
                    f"Must be one of {sorted(VALID_SEMANTICS)}."
                )
            self._current_values[k] = v

    def get_value(self, name: str) -> Any:
        if name in self._defaults:
            return self._current_values[name]
        else:
            raise KeyError(f"{name} is not a valid setting.")

    def reset(self) -> None:
        self._current_values = self._defaults.copy()

    def __enter__(self) -> OptionSettings:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        self.reset()

    def __repr__(self) -> str:
        settings = "\n ".join(
            f"{name}={value}" for name, value in self._current_values.items()
        )
        return f"OptionSettings:\n {settings}"


options = OptionSettings(
    display_max_rows=14,
    display_max_terms=6,
    semantics=LEGACY_SEMANTICS,
)
