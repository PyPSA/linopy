#!/usr/bin/env python3
"""
Created on Fri Jan 13 12:57:45 2023.

@author: fabian
"""

from __future__ import annotations


class OptionSettings:
    def __init__(self, **kwargs) -> None:
        self._defaults = kwargs
        self._current_values = kwargs.copy()

    def __call__(self, **kwargs) -> None:
        self.set_value(**kwargs)

    def __getitem__(self, key: str) -> int:
        return self.get_value(key)

    def __setitem__(self, key: str, value: int) -> None:
        return self.set_value(**{key: value})

    def set_value(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k not in self._defaults:
                raise KeyError(f"{k} is not a valid setting.")
            self._current_values[k] = v

    def get_value(self, name: str) -> int:
        if name in self._defaults:
            return self._current_values[name]
        else:
            raise KeyError(f"{name} is not a valid setting.")

    def reset(self) -> None:
        self._current_values = self._defaults.copy()

    def __enter__(self) -> OptionSettings:
        return self

    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        self.reset()

    def __repr__(self) -> str:
        settings = "\n ".join(
            f"{name}={value}" for name, value in self._current_values.items()
        )
        return f"OptionSettings:\n {settings}"


options = OptionSettings(display_max_rows=14, display_max_terms=6)
