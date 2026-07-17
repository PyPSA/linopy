import textwrap
import time
import typing

import numpy as np
import xarray as xr
from tqdm.asyncio import tqdm

from linopy.declarative import eval_attrs, helper_functions, parsing
from linopy.declarative.schema import (
    LOGGER,
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

ORDERED_COMPONENTS_T = typing.Literal[
    "variables",
    "expressions",
    "constraints",
    # "piecewise_constraints",
    "objectives",
]

DTYPE_OPTIONS = {
    "string": str,
    "float": float,
    "bool": bool,
    "datetime": np.datetime64,
    "date": np.datetime64,
    "integer": int,
}

DATETIME_DTYPE = "M"
"""Numpy type kind for datetime arrays"""


def declarative_model(math_def: dict, input_data: xr.Dataset, config: dict) -> Model:
    """Build a Linopy Model from declarative math definitions and input data."""
    builder = DeclarativeModelBuilder(math_def, input_data, config)
    return builder.build()


class DeclarativeModelBuilder:
    def __init__(self, math_def: dict, input_data: xr.Dataset, config: dict):
        self.model = Model()
        self.math = MathModel.model_validate(math_def)
        self.input_data = self._update_dtypes(input_data)
        self.config = ConfigModel.model_validate(config)

        self._check_inputs()

    def _update_dtypes(self, ds: xr.Dataset, id_: str = "") -> xr.Dataset:
        """
        Update data types of coordinates or data variables in the dataset.

        Args:
            ds (xr.Dataset): Dataset to update.
            math (math_schema.CalliopeBuildMath): Model math definition.
            id_ (str, optional): ID of the dataset being updated, for logging purposes. Defaults to an empty string.

        Raises:
            ValueError: If there is a mismatch between the provided variable and its definition in the model math.

        Returns:
            xr.Dataset: `ds` with data types updated.
        """
        prefix = f"{id_} | " if id_ else ""
        for var_name, var_data in ds.items():
            try:
                math_def = self.math.find(
                    var_name, subset=["lookups", "parameters", "dimensions"]
                )
            except KeyError:
                LOGGER.info(
                    f"{prefix}input data `{var_name}` not defined in model math; "
                    "it will not be available in the optimisation problem."
                )
                continue

            dtype_str = math_def.dtype  # type: ignore
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
                case "datetime":
                    updated_var = time._datetime_index(
                        var_data.to_series(), self.config.datetime_format
                    ).to_xarray()
                case "date":
                    updated_var = (
                        time._datetime_index(
                            var_data.to_series(), self.config.date_format
                        )
                        .to_xarray()
                        .assign_attrs(var_data.attrs)
                    )
                case "bool":
                    updated_var = var_data.fillna(False).astype(dtype)
                case _:
                    updated_var = var_data.astype(dtype)

            ds[var_name] = updated_var
        return ds

    @staticmethod
    def _sorted_by_order(
        root: typing.Mapping[str, typing.Any],
    ) -> list[tuple[str, typing.Any]]:
        """Return (name, obj) pairs from a root mapping, sorted by obj.order."""
        return sorted(root.items(), key=lambda item: getattr(item[1], "order", 0))

    def _check_inputs(self) -> None:
        data_checks = self.math.checks
        check_results: dict[str, list[str]] = {"raise": [], "warn": []}
        parser_ = parsing.mask_parser.generate_mask_string_parser(
            **self.math.parsing_components["mask"]
        )
        eval_kwargs = {
            "model": self.model,
            "math": self.math,
            "input_data": self.input_data,
            "config": self.config,
            "helper_functions": helper_functions._registry["mask"],
        }
        active = self.input_data.get("active", xr.DataArray(True))
        for name, check in data_checks.root.items():
            if check.active:
                parsed_ = parser_.parse_string(check.mask, parse_all=True)
                eval_attrs_ = eval_attrs.EvalAttrs(equation_name=name, **eval_kwargs)
                evaluated = parsed_[0].eval("raw", eval_attrs_)
                if (evaluated & active).any():
                    check_results[check.errors].append(check.message)

        print_warnings_and_raise_errors(
            check_results["warn"],
            check_results["raise"],
            during="model input data checks",
        )

    def add_variable(self, name: str, definition: VariableDef) -> None:
        references: set[str] = set()
        parsed_component = parsing.ParsedBackendComponent(
            "variables", name, definition, self.math.parsing_components
        )
        mask = parsed_component.generate_top_level_mask(
            self.input_data,
            self.model,
            self.math,
            self.config,
            align_to_foreach_sets=True,
            break_early=True,
            references=references,
        )
        kwargs = {
            "upper": definition.bounds.upper,
            "lower": definition.bounds.lower,
            "integer": definition.domain == "integer",
            "binary": definition.domain == "binary",
        }
        if mask.any():
            self.model.add_variables(coords=mask.coords, name=name, mask=mask, **kwargs)
            self.model.variables[name].attrs["references"] = references
        else:
            LOGGER.info(
                f"variables:{name} | No valid data points after applying mask. Variable not added to model."
            )

    def add_expression(self, name: str, definition: ExpressionDef) -> None:
        references: set[str] = set()
        parsed_component = parsing.ParsedBackendComponent(
            "expressions", name, definition, self.math.parsing_components
        )
        mask = parsed_component.generate_top_level_mask(
            self.input_data,
            self.model,
            self.math,
            self.config,
            align_to_foreach_sets=True,
            break_early=True,
            references=references,
        )
        expr = LinearExpression(float("nan"), self.model).where(mask)
        all_mask = mask.copy()
        if mask.any():
            equations = parsed_component.parse_equations()
            for equation in equations:
                sub_mask = equation.evaluate_mask(
                    self.input_data,
                    self.model,
                    self.math,
                    self.config,
                    initial_mask=mask,
                    references=references,
                )
                if not sub_mask.any():
                    continue
                sub_mask = parsed_component.drop_dims_not_in_foreach(sub_mask)
                if (~expr.isnull() & sub_mask).any():
                    raise ValueError(
                        f"expressions:{name} | Overlapping 'mask' conditions between equations are not allowed. "
                        "Please revise the 'mask' conditions to ensure they are mutually exclusive."
                    )
                expr_to_fill = equation.evaluate_expression(
                    self.input_data,
                    self.model,
                    self.math,
                    mask=sub_mask,
                    references=references,
                )
                expr = merge([expr, expr_to_fill.where(sub_mask)])
            if not expr.isnull().all():
                self.model.add_expressions(name=name, data=expr, mask=all_mask)
                self.model.expressions[name].attrs["references"] = references
            else:
                LOGGER.info(
                    f"expressions:{name} | No valid data points after applying mask. Expression not added to model."
                )
        else:
            LOGGER.info(
                f"expressions:{name} | No valid data points after applying mask. Expression not added to model."
            )

    def add_constraint(self, name: str, definition: ConstraintDef) -> None:
        references: set[str] = set()
        parsed_component = parsing.ParsedBackendComponent(
            "constraints", name, definition, self.math.parsing_components
        )
        mask = parsed_component.generate_top_level_mask(
            self.input_data,
            self.model,
            self.math,
            self.config,
            align_to_foreach_sets=True,
            break_early=True,
            references=references,
        )
        lhs = LinearExpression(float("nan"), self.model).where(mask)
        sign = xr.DataArray().where(parsed_component.drop_dims_not_in_foreach(mask))
        rhs = LinearExpression(float("nan"), self.model).where(mask)
        all_mask = mask.copy()
        if not mask.any():
            LOGGER.info(
                f"constraints:{name} | No valid data points after applying mask. Constraint not added to model."
            )
            return None

        equations = parsed_component.parse_equations()
        for equation in equations:
            sub_mask = equation.evaluate_mask(
                self.input_data,
                self.model,
                self.math,
                self.config,
                initial_mask=mask,
                references=references,
            )
            if not sub_mask.any():
                LOGGER.info(
                    f"constraints:{equation.name} | No valid data points after applying mask. Constraint not added to model."
                )
                continue
            sub_mask = parsed_component.drop_dims_not_in_foreach(sub_mask)
            if (sign.notnull() & sub_mask).any():
                raise ValueError(
                    f"constraints:{name} | "
                    "Overlapping 'mask' conditions between equations are not allowed. "
                    "Please revise the 'mask' conditions to ensure they are mutually exclusive."
                )
            lhs_to_fill, sign_to_fill, rhs_to_fill = equation.evaluate_equation(
                self.input_data,
                self.model,
                self.math,
                mask=sub_mask,
                references=references,
            )
            lhs = merge([lhs, lhs_to_fill])
            rhs = merge([rhs, rhs_to_fill])
            sign = sign.fillna(sign_to_fill)

        if sign.isnull().all():
            LOGGER.info(
                f"constraints:{name} | No valid data points after applying mask. Constraint not added to model."
            )
            return None

        self.model.add_constraints(
            coords=all_mask.coords,
            name=name,
            lhs=lhs,
            sign=sign.fillna(
                "=="
            ),  # Default to equality to avoid errors; will be masked.
            rhs=rhs,
            mask=all_mask,
        )
        self.model.constraints[name].attrs["references"] = references

    def add_objective(self, name: str, definition: ObjectiveDef) -> None:
        references: set[str] = set()
        parsed_component = parsing.ParsedBackendComponent(
            "objectives", name, definition, self.math.parsing_components
        )
        mask = parsed_component.generate_top_level_mask(
            self.input_data,
            self.model,
            self.math,
            self.config,
            align_to_foreach_sets=True,
            break_early=True,
            references=references,
        )
        expr = LinearExpression(float("nan"), self.model).where(mask)
        if mask.any():
            equations = parsed_component.parse_equations()
            for equation in equations:
                sub_mask = equation.evaluate_mask(
                    self.input_data,
                    self.model,
                    self.math,
                    self.config,
                    initial_mask=mask,
                    references=references,
                )
                if not sub_mask.any():
                    continue
                sub_mask = parsed_component.drop_dims_not_in_foreach(sub_mask)
                if (~expr.isnull() & sub_mask).any():
                    raise ValueError(
                        f"objectives:{name} | Overlapping 'mask' conditions between equations are not allowed. "
                        "Please revise the 'mask' conditions to ensure they are mutually exclusive."
                    )
                expr_to_fill = equation.evaluate_expression(
                    self.input_data,
                    self.model,
                    self.math,
                    mask=sub_mask,
                    references=references,
                )
                expr = expr_to_fill
            self.model.add_objective(expr=expr, sense=definition.sense)
            self.model.objective.attrs["references"] = references

    def build(self) -> Model:
        for components in typing.get_args(ORDERED_COMPONENTS_T):
            component = components.removesuffix("s")
            ordered_items = self._sorted_by_order(self.math[components].root)
            ordered_items_tqdm = tqdm(
                ordered_items,
                desc=f"Building {components}.",
                colour=TQDM_COLOR,
            )
            for name, definition in ordered_items_tqdm:
                start = time.time()
                getattr(self, f"add_{component}")(name, definition)
                end = time.time() - start
                LOGGER.debug(f"{components}:{name} | Built in {end:.4f}s")
            LOGGER.info(f"{components} | Generated.")
        return self.model


def print_warnings_and_raise_errors(
    warnings: list[str] | dict[str, list[str]] | None = None,
    errors: list[str] | dict[str, list[str]] | None = None,
    during: str = "model processing",
    bullet: str = " * ",
) -> None:
    """
    Process collections of warnings/errors.

    Prints warnings / raises errors with a bullet point list of the concatenated
    collections.

    Lists will return simple bullet lists:
    E.g. warnings=["foo", "bar"] becomes:

        Possible issues found during model processing:
        * foo
        * bar

    Dicts of lists will return nested bullet lists:
    E.g. errors={"foo": ["foobar", "foobaz"]} becomes:

        Errors during model processing:
        * foo
            * foobar
            * foobaz

    Args:
        warnings (list[str] | dict[str, list[str]] | None, optional):
            List of warning strings or dictionary of warning strings.
            If None or an empty list, no warnings will be printed.
            Defaults to None.
        errors (list[str] | dict[str, list[str]] | None, optional):
            List of error strings or dictionary of error strings.
            If None or an empty list, no errors will be raised.
            Defaults to None.
        during (str, optional):
            Substring that will be placed at the top of the concatenated list of warnings/errors to point to during which phase of data processing they occurred.
            Defaults to "model processing".
        bullet (str, optional): Type of bullet points to use. Defaults to " * ".

    Raises:
        ModelError: If errors is not None or is a non-empty list/dict

    """
    spacer = " " * len(bullet)

    def _sort_strings(stringlist: list[str]) -> list[str]:
        return sorted(list(set(stringlist)))

    def _predicate(string_: str) -> bool:
        return not string_.startswith((bullet, spacer))

    def _indenter(strings: list[str] | dict[str, list[str]]) -> str:
        if isinstance(strings, dict):
            sorted_strings = []
            for k, v in strings.items():
                sorted_strings.append(str(k) + ":")
                sorted_strings.extend(_sort_strings([spacer + bullet + i for i in v]))
        else:
            sorted_strings = _sort_strings(strings)
        return textwrap.indent("\n".join(sorted_strings), bullet, predicate=_predicate)

    if warnings:
        LOGGER.info(f"Possible issues found during {during}:\n" + _indenter(warnings))

    if errors:
        raise ValueError(f"Errors during {during}:\n" + _indenter(errors))
