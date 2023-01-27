#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:57:45 2023.

@author: fabian
"""


class OptionSettings:
    def __init__(self, **kwargs):
        self._defaults = kwargs
        self._current_values = kwargs.copy()

    def __call__(self, **kwargs):
        self.set_value(**kwargs)

    def __getitem__(self, key):
        return self.get_value(key)

    def __setitem__(self, key, value):
        return self.set_value(**{key: value})

    def set_value(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._defaults:
                raise KeyError(f"{k} is not a valid setting.")
            self._current_values[k] = v

    def get_value(self, name):
        if name in self._defaults:
            return self._current_values[name]
        else:
            raise KeyError(f"{name} is not a valid setting.")

    def reset(self):
        self._current_values = self._defaults.copy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()

    def __repr__(self):
        settings = "\n ".join(
            f"{name}={value}" for name, value in self._current_values.items()
        )
        return f"OptionSettings:\n {settings}"


options = OptionSettings(display_max_rows=14, display_max_terms=6)
