# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Parsing evaluation attributes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import xarray as xr

from linopy.declarative.schema import ConfigModel, MathModel

if TYPE_CHECKING:
    from linopy.model import Model
TRUE_ARRAY = xr.DataArray(True)

LOGGER = logging.getLogger(__name__)


@dataclass
class EvalAttrs:
    """Attributes required for evaluating parsed expressions."""

    model: Model
    """Backend interface component dataset."""

    equation_name: str = ""
    """Name of the equation being evaluated."""

    helper_functions: dict[str, Callable] = field(default_factory=dict)
    """Helper functions available for evaluations."""

    input_data: xr.Dataset = field(default_factory=xr.Dataset)
    """Model input data."""

    math: MathModel = field(default_factory=MathModel)
    """Linopy math definitions."""

    apply_mask: bool = True
    """Whether to apply the 'where' condition."""

    as_values: bool = False
    """If True, return with the array contents evaluated to base Python objects.
    If False, return with the array contents as they are in the backend dataset."""

    config: ConfigModel = field(default_factory=ConfigModel)
    """Build configuration options."""

    references: set[str] = field(default_factory=set)
    """References to dimensions/lookups/parameters/variables/global expressions used in the expression."""

    slice_dict: dict = field(default_factory=dict)
    """Dictionary to look up array slice expressions if referenced in the evaluated string."""

    sub_expression_dict: dict = field(default_factory=dict)
    """Dictionary to look up sub-expressions if referenced in the evaluated string."""

    mask: xr.DataArray = field(default_factory=lambda: xr.DataArray(True))
    """Boolean array defining where the expression should be applied."""
