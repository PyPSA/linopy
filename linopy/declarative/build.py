import time
import typing

import xarray as xr

from linopy import merge
from linopy.declarative import parsing
from linopy.declarative.schema import (
    LOGGER,
    ConfigModel,
    ConstraintDef,
    MathModel,
    ObjectiveDef,
    VariableDef,
)
from linopy.expressions import LinearExpression
from linopy.model import Model

ORDERED_COMPONENTS_T = typing.Literal[
    "variables",
    # "global_expressions",
    "constraints",
    # "piecewise_constraints",
    "objectives",
]


def declarative_model(math_def: dict, input_data: xr.Dataset, config: dict) -> Model:
    """Build a Linopy Model from declarative math definitions and input data."""
    builder = DeclarativeModelBuilder(math_def, input_data, config)
    return builder.build()


class DeclarativeModelBuilder:
    def __init__(self, math_def: dict, input_data: xr.Dataset, config: dict):
        self.model = Model()
        self.math = MathModel.model_validate(math_def)
        self.input_data = input_data
        self.config = ConfigModel.model_validate(config)

    @staticmethod
    def _sorted_by_order(
        root: typing.Mapping[str, typing.Any],
    ) -> list[tuple[str, typing.Any]]:
        """Return (name, obj) pairs from a root mapping, sorted by obj.order."""
        return sorted(root.items(), key=lambda item: getattr(item[1], "order", 0))

    def add_variable(self, name: str, definition: VariableDef):
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
            LOGGER.warning(
                f"Optimisation Model | variables:{name} | No valid data points after applying 'where' condition. Variable not added to model."
            )

    def add_constraint(self, name: str, definition: ConstraintDef):
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
                if (sign.notnull() & sub_mask).any():
                    raise ValueError(
                        f"Optimisation Model | constraints:{name} | Overlapping 'mask' conditions between equations are not allowed. Please revise the 'mask' conditions to ensure they are mutually exclusive."
                    )
                lhs_to_fill, sign_to_fill, rhs_to_fill = equation.evaluate_expression(
                    self.input_data,
                    self.model,
                    self.math,
                    mask=sub_mask,
                    references=references,
                )
                all_mask = all_mask | sub_mask
                if isinstance(lhs_to_fill, xr.DataArray):
                    lhs = lhs.fillna(lhs_to_fill)
                else:
                    lhs = merge([lhs, lhs_to_fill]).where(all_mask)
                sign = sign.fillna(sign_to_fill)
                if isinstance(rhs_to_fill, xr.DataArray):
                    rhs = rhs.fillna(rhs_to_fill)
                else:
                    rhs = merge([rhs, rhs_to_fill]).where(all_mask)

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

    def add_objective(self, name: str, definition: ObjectiveDef):
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
                        f"Optimisation Model | objectives:{name} | Overlapping 'mask' conditions between equations are not allowed. Please revise the 'mask' conditions to ensure they are mutually exclusive."
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
            for name, definition in ordered_items:
                start = time.time()
                getattr(self, f"add_{component}")(name, definition)
                end = time.time() - start
                LOGGER.debug(
                    f"Optimisation Model | {components}:{name} | Built in {end:.4f}s"
                )
            LOGGER.info(f"Optimisation Model | {components} | Generated.")
        return self.model
