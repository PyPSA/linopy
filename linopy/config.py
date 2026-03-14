#!/usr/bin/env python3
"""
Created on Fri Jan 13 12:57:45 2023.

@author: fabian
"""

from __future__ import annotations

from typing import Any

import numpy as np

_VALID_LABEL_DTYPES = {np.int32, np.int64}


class OptionSettings:
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
            if k == "label_dtype" and v not in _VALID_LABEL_DTYPES:
                raise ValueError(
                    f"label_dtype must be one of {_VALID_LABEL_DTYPES}, got {v}"
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


options = OptionSettings(display_max_rows=14, display_max_terms=6, label_dtype=np.int32)
