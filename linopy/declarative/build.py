"""
Linopy declarative model-build module.

This module contains the entry point to build a linopy optimisation model from a declarative math definition and an
xarray dataset of input data.

This module is adapted from the calliope Apache-2.0 licensed math backend model:
https://github.com/calliope-project/calliope/blob/9916116a06ec8c1feaf3c2606bdb8941b916ce85/src/calliope/backend/backend_model.py
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from typing import Any

import xarray as xr
from tqdm.auto import tqdm

from linopy.declarative import parsing
from linopy.declarative.helpers import HelperFunction, build_registry
from linopy.declarative.nodes import Component, Context, find_refs
from linopy.declarative.schema import (
    BUILD_ORDER,
    DTYPE_OPTIONS,
    EQUATION_GROUP_T,
    ConfigModel,
    ConstraintDef,
    ExpressionDef,
    MathModel,
    ObjectiveDef,
    VariableDef,
)
from linopy.expressions import LinearExpression, merge
from linopy.io import TQDM_COLOR
from linopy.model import Model

LOGGER = logging.getLogger(__name__)

_SKIP_MESSAGE = "No valid data points after applying mask. Not added to model."


def declarative_model(
    math_def: dict,
    input_data: xr.Dataset,
    config: dict,
    helpers: Iterable[type[HelperFunction]] = (),
) -> Model:
    """
    Build a linopy Model from a declarative math definition and input data.

    Parameters
    ----------
    math_def : dict
        Declarative math definition (see
        :class:`linopy.declarative.schema.MathModel` for the expected structure).
    input_data : xr.Dataset
        Model input data (parameters, lookups, dimensions).
    config : dict
        Build configuration options.
    helpers : Iterable[type[HelperFunction]], optional
        User-defined helper functions to make available in math strings, in
        addition to the built-in ones.

    Returns
    -------
    Model
        The built linopy model, ready to solve.
    """
    return DeclarativeModelBuilder(math_def, input_data, config, helpers).build()


class _DeclarativeBase:
    """Shared validation and context setup of the model and LaTeX builders."""

    def __init__(
        self,
        math_def: dict,
        input_data: xr.Dataset | None,
        config: dict | None,
        helpers: Iterable[type[HelperFunction]] = (),
        *,
        math_reprs: dict[str, str] | None = None,
    ) -> None:
        """
        Validate the math definition, input data, and config.

        Parameters
        ----------
        math_def : dict
            Declarative math definition.
        input_data : xr.Dataset, optional
            Model input data.
        config : dict, optional
            Build configuration options.
        helpers : Iterable[type[HelperFunction]], optional
            User-defined helper functions, in addition to the built-in ones.
        math_reprs : dict[str, str], optional
            Custom LaTeX representations per component name.
        """
        self.model = Model()
        self.math = MathModel.model_validate(math_def)
        self.parsed = parsing.parse_math(self.math)
        self.input_data = input_data if input_data is not None else xr.Dataset()
        self.config = ConfigModel.model_validate(config or {})
        self._ctx = Context(
            model=self.model,
            input_data=self.input_data,
            math=self.math,
            config=self.config,
            helpers=build_registry(helpers),
            math_reprs=math_reprs or {},
        )

    def _references(self, parsed: parsing.ParsedComponent) -> list[str]:
        """Return the sorted names of all math components a component references."""
        refs = find_refs(parsed.mask, Component)
        for equation in parsed.equations:
            refs |= equation.references()
        return sorted(refs)


class DeclarativeModelBuilder(_DeclarativeBase):
    """Builder turning a declarative math definition into a linopy Model."""

    def __init__(
        self,
        math_def: dict,
        input_data: xr.Dataset,
        config: dict,
        helpers: Iterable[type[HelperFunction]] = (),
    ) -> None:
        """
        Validate the math definition, input data, and config, ready to `build()`.

        Parameters
        ----------
        math_def : dict
            Declarative math definition.
        input_data : xr.Dataset
            Model input data.
        config : dict
            Build configuration options.
        helpers : Iterable[type[HelperFunction]], optional
            User-defined helper functions, in addition to the built-in ones.
        """
        super().__init__(math_def, input_data, config, helpers)
        self.input_data = self._update_dtypes(self.input_data)
        self._ctx = replace(self._ctx, input_data=self.input_data)
        self._check_inputs()

    def _update_dtypes(self, ds: xr.Dataset, id_: str = "") -> xr.Dataset:
        """
        Coerce dataset variables to the dtypes given by their math definitions.

        Variables not defined in the math are left unchanged (with an INFO log);
        datetime/date variables pass through uncoerced.
        """
        prefix = f"{id_} | " if id_ else ""
        for var_name, var_data in ds.items():
            try:
                math_def = self.math.find(
                    str(var_name), subset=["lookups", "parameters", "dimensions"]
                )
            except KeyError:
                LOGGER.info(
                    f"{prefix}input data `{var_name}` not defined in model math; "
                    "it will not be available in the optimisation problem."
                )
                continue

            dtype_str: str = math_def["dtype"]
            if dtype_str in ("datetime", "date"):
                continue
            dtype = DTYPE_OPTIONS[dtype_str]
            LOGGER.debug(
                f"{prefix}{math_def._group} | Updating values of `{var_name}` to {dtype_str} type"
            )
            match dtype_str:
                case "string":
                    updated_var = (
                        var_data.astype(dtype)
                        .where(var_data.notnull())
                        .where(var_data != "")
                    )
                case "bool":
                    updated_var = var_data.fillna(False).astype(dtype)
                case _:
                    updated_var = var_data.astype(dtype)

            ds[var_name] = updated_var
        return ds

    def _check_inputs(self) -> None:
        """Run the math's input-data checks, warning or raising on triggered ones."""
        warn_msgs: list[str] = []
        error_msgs: list[str] = []
        active = self.input_data.get("active", xr.DataArray(True))
        for name in self.math.checks._active:
            check = self.math.checks[name]
            mask_node = self.parsed.checks[name]
            check_ctx = replace(self._ctx, mode="mask", equation_name=name)
            evaluated = mask_node.evaluate(check_ctx)
            if (evaluated & active).any():
                messages = error_msgs if check.errors == "raise" else warn_msgs
                messages.append(check.message)

        if warn_msgs:
            bullets = "\n".join(f" * {msg}" for msg in sorted(set(warn_msgs)))
            LOGGER.info(
                f"Possible issues found during model input data checks:\n{bullets}"
            )
        if error_msgs:
            bullets = "\n".join(f" * {msg}" for msg in sorted(set(error_msgs)))
            raise ValueError(f"Errors during model input data checks:\n{bullets}")

    @staticmethod
    def _sorted_by_order(root: Mapping[str, Any]) -> list[tuple[str, Any]]:
        """Return (name, definition) pairs from a root mapping, sorted by definition order."""
        return sorted(root.items(), key=lambda item: getattr(item[1], "order", 0))

    def _iter_equations(
        self,
        equations: list[parsing.Equation],
        group: EQUATION_GROUP_T,
        mask: xr.DataArray,
    ) -> Iterator[tuple[parsing.Equation, xr.DataArray]]:
        """
        Yield each parsed equation with its evaluated, foreach-aligned sub-mask.

        Equations whose mask leaves no valid data point are skipped (with an INFO log).
        """
        for equation in equations:
            sub_mask = parsing.as_mask(equation, self._ctx, initial_mask=mask)
            if not sub_mask.any():
                LOGGER.info(f"{group}:{equation.name} | {_SKIP_MESSAGE}")
                continue
            yield equation, parsing.drop_dims_not_in_foreach(sub_mask, equation.sets)

    def add_variable(self, name: str, definition: VariableDef) -> None:
        """Add a decision variable to the model, masked by its math definition."""
        parsed = self.parsed["variables"][name]
        mask = parsing.component_mask(
            "variables", name, definition, parsed.mask, self._ctx
        )
        if not mask.any():
            LOGGER.info(f"variables:{name} | {_SKIP_MESSAGE}")
            return
        self.model.add_variables(
            coords=mask.coords,
            name=name,
            mask=mask,
            upper=definition.bounds.upper,
            lower=definition.bounds.lower,
            integer=definition.domain == "integer",
        )
        # Variable.attrs values are typed Hashable, but a sorted list serializes best.
        self.model.variables[name].attrs["references"] = self._references(parsed)  # type: ignore[assignment]

    def add_expression(self, name: str, definition: ExpressionDef) -> None:
        """Add a named expression to the model, merging its equation variants."""
        parsed = self.parsed["expressions"][name]
        mask = parsing.component_mask(
            "expressions", name, definition, parsed.mask, self._ctx
        )
        if not mask.any():
            LOGGER.info(f"expressions:{name} | {_SKIP_MESSAGE}")
            return
        expr: Any = LinearExpression(float("nan"), self.model).where(mask)
        filled = xr.DataArray(False)
        for equation, sub_mask in self._iter_equations(
            parsed.equations, "expressions", mask
        ):
            if (filled & sub_mask).any():
                raise ValueError(
                    f"expressions:{name} | Overlapping 'mask' conditions between "
                    "equations are not allowed. Please revise the 'mask' conditions "
                    "to ensure they are mutually exclusive."
                )
            filled = filled | sub_mask
            expr_to_fill = parsing.as_expression(equation, self._ctx, mask=sub_mask)
            expr = merge([expr, expr_to_fill.where(sub_mask)])
        if not filled.any():
            LOGGER.info(f"expressions:{name} | {_SKIP_MESSAGE}")
            return
        self.model.add_expressions(name=name, data=expr, mask=mask)
        self.model.expressions[name].attrs["references"] = self._references(parsed)

    def add_constraint(self, name: str, definition: ConstraintDef) -> None:
        """Add a constraint to the model, merging its equation variants."""
        parsed = self.parsed["constraints"][name]
        mask = parsing.component_mask(
            "constraints", name, definition, parsed.mask, self._ctx
        )
        if not mask.any():
            LOGGER.info(f"constraints:{name} | {_SKIP_MESSAGE}")
            return
        lhs: Any = LinearExpression(float("nan"), self.model).where(mask)
        rhs: Any = LinearExpression(float("nan"), self.model).where(mask)
        sign = xr.DataArray().where(mask)
        for equation, sub_mask in self._iter_equations(
            parsed.equations, "constraints", mask
        ):
            if (sign.notnull() & sub_mask).any():
                raise ValueError(
                    f"constraints:{name} | Overlapping 'mask' conditions between "
                    "equations are not allowed. Please revise the 'mask' conditions "
                    "to ensure they are mutually exclusive."
                )
            lhs_to_fill, sign_to_fill, rhs_to_fill = parsing.as_constraint(
                equation, self._ctx, mask=sub_mask
            )
            lhs = merge([lhs, lhs_to_fill])
            rhs = merge([rhs, rhs_to_fill])
            sign = sign.fillna(sign_to_fill)

        if sign.isnull().all():
            LOGGER.info(f"constraints:{name} | {_SKIP_MESSAGE}")
            return
        self.model.add_constraints(
            coords=mask.coords,
            name=name,
            lhs=lhs,
            # Default to equality to avoid errors on masked-out points.
            sign=sign.fillna("=="),
            rhs=rhs,
            mask=mask,
        )
        self.model.constraints[name].attrs["references"] = self._references(parsed)

    def add_objective(self, name: str, definition: ObjectiveDef) -> None:
        """Set the model objective, merging its equation variants."""
        parsed = self.parsed["objectives"][name]
        mask = parsing.component_mask(
            "objectives", name, definition, parsed.mask, self._ctx
        )
        if not mask.any():
            LOGGER.info(f"objectives:{name} | {_SKIP_MESSAGE}")
            return
        pieces: list[tuple[LinearExpression, xr.DataArray]] = []
        filled = xr.DataArray(False)
        for equation, sub_mask in self._iter_equations(
            parsed.equations, "objectives", mask
        ):
            if (filled & sub_mask).any():
                raise ValueError(
                    f"objectives:{name} | Overlapping 'mask' conditions between "
                    "equations are not allowed. Please revise the 'mask' conditions "
                    "to ensure they are mutually exclusive."
                )
            filled = filled | sub_mask
            pieces.append(
                (parsing.as_expression(equation, self._ctx, mask=sub_mask), sub_mask)
            )
        if not pieces:
            LOGGER.info(f"objectives:{name} | {_SKIP_MESSAGE}")
            return
        expr: Any = pieces[0][0]
        if len(pieces) > 1:
            expr = merge([piece.where(sub_mask) for piece, sub_mask in pieces])
        self.model.add_objective(expr=expr, sense=definition.sense)
        self.model.objective.attrs["references"] = self._references(parsed)

    def build(self) -> Model:
        """
        Build all math components into the linopy model.

        Components are built in group order (variables, expressions, constraints,
        objectives) and, within a group, by their `order` attribute where defined.

        Returns
        -------
        Model
            The built linopy model.
        """
        active_objectives = list(self.math.objectives._active)
        if len(active_objectives) > 1:
            raise ValueError(
                f"Only one active objective is supported, found: {active_objectives}"
            )
        for group in BUILD_ORDER:
            component = group.removesuffix("s")
            ordered_items = self._sorted_by_order(self.math[group]._active)
            for name, definition in tqdm(
                ordered_items, desc=f"Building {group}.", colour=TQDM_COLOR
            ):
                start = time.time()
                getattr(self, f"add_{component}")(name, definition)
                LOGGER.debug(f"{group}:{name} | Built in {time.time() - start:.4f}s")
            LOGGER.info(f"{group} | Generated.")
        return self.model
