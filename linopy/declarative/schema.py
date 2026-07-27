"""
Linopy declarative math schema module.

This module contains the pydantic models that validate declarative math and
build-configuration definitions.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterable
from functools import cached_property
from typing import Annotated, Any, ClassVar, Literal, Self, TypeVar, get_args

import numpy as np
from annotated_types import Len
from pydantic import AfterValidator, BaseModel, Field, RootModel, model_validator
from pydantic_core import PydanticCustomError

LOGGER = logging.getLogger(__name__)

# Modified from https://github.com/pydantic/pydantic-core/pull/820#issuecomment-1670475909
T = TypeVar("T", bound=Hashable | list)

DTYPE_OPTIONS: dict[str, type] = {
    "string": str,
    "float": float,
    "bool": bool,
    "integer": int,
    "datetime": np.datetime64,
    "date": np.datetime64,
}
"""Mapping from math-schema dtype names to Python/numpy types."""

COMPONENTS_T = Literal[
    "dimensions",
    "parameters",
    "lookups",
    "variables",
    "expressions",
    "constraints",
    "piecewise_constraints",
    "objectives",
    "postprocessed",
]

EQUATION_GROUP_T = Literal["expressions", "constraints", "objectives"]

BUILD_ORDER_T = Literal["variables", "expressions", "constraints", "objectives"]
BUILD_ORDER: tuple[BUILD_ORDER_T, ...] = get_args(BUILD_ORDER_T)

"""Component groups in the order they are built into a linopy model."""


def _validate_unique_list(v: list) -> list:
    try:
        unique = set(v)
    except TypeError:
        unique = set([tuple(i) for i in v])
    if len(v) != len(unique):
        raise PydanticCustomError("unique_list", "List must be unique")
    return v


UniqueList = Annotated[
    list[T],
    AfterValidator(_validate_unique_list),
    Field(json_schema_extra={"uniqueItems": True}),
]
"""A list with no repeated values."""
# ==
NonEmptyList = Annotated[list[T], Len(min_length=1)]
"""A list with at least one value in it."""
NonEmptyUniqueList = Annotated[UniqueList[T], Len(min_length=1)]
"""A list with at least one value in it and no repeated values."""
AttrStr = Annotated[str, Field(pattern=r"^[^_^\d][\w]*$")]
"""Single word string in snake_case (e.g., wind_offshore)."""
NumericVal = int | Annotated[float, Field(allow_inf_nan=True)]
"""Numerical integer or float value. Can be `nan` or infinite (`float(inf)`)."""


class LinopyDictModel(RootModel):
    """Pydantic model storing a dictionary of user-named component definitions."""

    def __setitem__(self, *args: Any, **kwargs: Any) -> None:
        """Do not allow direct item setting."""
        raise PydanticCustomError(
            "no_extra_dict",
            f"Cannot set a {self.__class__.__name__} item directly. Re-validate a new definition dictionary instead.",
        )

    def __getitem__(self, key: str) -> Any:
        """Expose the root attribute when getting an item by key."""
        return self.root[key]

    def __repr__(self, *args: Any, **kwargs: Any) -> str:
        """Show the __repr__ of the root attribute when requesting the __repr__ of the class."""
        return self.root.__repr__(*args, **kwargs)

    def __rich_repr__(self) -> Iterable:
        """Prettyprint the __repr__ of the root attribute when requesting the prettyprint of the class."""
        yield from self.root.items()

    @cached_property
    def _active(self) -> dict[str, BaseModel]:
        """Return only active components."""
        return {k: v for k, v in self.root.items() if v.active}


class LinopyListModel(RootModel):
    """Pydantic model storing a list of definitions."""

    def __iter__(self) -> Any:
        """Iterate over root attribute contents when iterating over class."""
        return iter(self.root)

    def __getitem__(self, item: int) -> Any:
        """Expose the root attribute when getting an item by index value."""
        return self.root[item]

    def __repr__(self, *args: Any, **kwargs: Any) -> str:
        """Show the __repr__ of the root attribute when requesting the __repr__ of the class."""
        return self.root.__repr__(*args, **kwargs)

    def __rich_repr__(self) -> Iterable:
        """Prettyprint the __repr__ of the root attribute when requesting the prettyprint of the class."""
        yield from self.root


class LinopyBaseModel(BaseModel):
    """Base class for declarative math pydantic models."""

    model_config = {
        "extra": "forbid",
        "frozen": True,
        "revalidate_instances": "always",
        "use_attribute_docstrings": True,
    }

    def __getitem__(self, item: str) -> Any:
        """Allow attribute access via item lookup."""
        return getattr(self, item)


class _ExpressionItem(LinopyBaseModel):
    """Schema for equations, _subexpressions and slices."""

    mask: str = "True"
    """Condition to determine whether the accompanying expression is built."""
    expression: str
    """Expression for this component.
    - _Equations: LHS OPERATOR RHS, where LHS and RHS are math expressions and OPERATOR is one of [==, <=, >=].
    - _Subexpressions: be one term or a combination of terms using the operators [+, -, *, /, **].
    - Slices: a list of set items or a call to a helper function.
    """


class _MathComponent(LinopyBaseModel):
    """Generic math component class."""

    title: str = ""
    """The component long name, for use in visualisation."""
    description: str = ""
    """A verbose description of the component."""
    active: bool = True
    """If False, this component will be ignored during the build phase."""

    _group: ClassVar[COMPONENTS_T]
    """Return the component group this component belongs to."""


class DimensionDef(_MathComponent):
    """Schema for named dimension."""

    dtype: Literal["string", "datetime", "date", "float", "integer"] = "string"
    """The data type of this dimension's items."""
    ordered: bool = False
    """If True, the order of the dimension items is meaningful (e.g. chronological time)."""
    iterator: str = "NEEDS_ITERATOR"
    """The name of the iterator to use in the LaTeX math formulation for this dimension."""

    _group: ClassVar[COMPONENTS_T] = "dimensions"

    @property
    def default(self) -> float:
        """Dummy field to align with lookups and dims."""
        return float("nan")


class ParameterDef(_MathComponent):
    """Schema for named parameter."""

    default: float | int = float("nan")
    """The default value for the parameter, if not set in the data."""
    unit: str = ""
    """The unit of the parameter, e.g. 'kW', 'm', 'kg', 'energy', 'power', ..."""

    @property
    def dtype(self) -> Literal["float"]:
        """Dummy field to align with lookups and dims."""
        return "float"

    _group: ClassVar[COMPONENTS_T] = "parameters"


class LookupDef(_MathComponent):
    """Schema for named lookup arrays."""

    default: AttrStr | float | int | bool = float("nan")
    """The default value for the lookup, if not set in the data."""
    dtype: Literal["float", "string", "bool", "datetime", "date"] = "string"
    """The lookup data type."""
    one_of: list | None = None
    """If given, the lookup values must be one of these items."""

    _group: ClassVar[COMPONENTS_T] = "lookups"


class _MathIndexedComponent(_MathComponent):
    """Generic indexed component class."""

    foreach: UniqueList[AttrStr] = Field(default_factory=list)
    """Sets (a.k.a. dimensions) of the model over which the math formulation component
    will be built."""
    mask: str = "True"
    """Top-level condition to determine whether the component exists in this
    optimisation problem. At all if `foreach` is not given, or for specific index items
    within the product of the sets given by `foreach`."""


class _Equations(LinopyListModel):
    """List of equations that can be updated when a parent pydantic model is updated."""

    root: list[_ExpressionItem] = Field(default_factory=list)


class _SubExpressions(LinopyDictModel):
    """Dictionary of sub-expressions that can be updated when a parent pydantic model is updated."""

    root: dict[AttrStr, _Equations] = Field(default_factory=dict)


class _MathEquationComponent(_MathComponent):
    """Components necessary to generate math expressions."""

    equations: _Equations = _Equations()
    """Constraint math equations."""
    sub_expressions: _SubExpressions = _SubExpressions()
    """Named sub-expressions."""
    slices: _SubExpressions = _SubExpressions()
    """Named index slices."""

    @model_validator(mode="after")
    def must_have_equations_if_active(self) -> Self:
        """Ensure that equations are defined if the component is active."""
        if self.active and not self.equations.root:
            raise ValueError("Must have equations defined if component is active.")
        return self


class ConstraintDef(_MathIndexedComponent, _MathEquationComponent):
    """Schema for named constraints."""

    _group: ClassVar[COMPONENTS_T] = "constraints"


class PiecewiseConstraintDef(_MathIndexedComponent):
    """
    Schema for named piece-wise constraints.

    These link an `x`-axis decision variable with a `y`-axis decision variable with
    values at specified breakpoints.
    """

    x_expression: str
    """X variable name whose values are assigned at each breakpoint."""
    y_expression: str
    """Y variable name whose values are assigned at each breakpoint."""
    x_values: str
    """X parameter name containing data, indexed over the `breakpoints` dimension."""
    y_values: str
    """Y parameter name containing data, indexed over the `breakpoints` dimension."""

    _group: ClassVar[COMPONENTS_T] = "piecewise_constraints"


class ExpressionDef(_MathIndexedComponent, _MathEquationComponent):
    """
    Schema for named expressions.

    Can be used to combine parameters and variables and then used in one or more
    expressions elsewhere in the math formulation (i.e., in constraints, objectives,
    and other expressions).

    NOTE: If expecting to use expression `A` in expression `B`, `A` must
    be defined above `B`.
    """

    unit: str = ""
    """Generalised unit of the component (e.g., length, time, quantity_per_hour, ...)."""
    default: NumericVal = float("nan")
    """If set, will be the default value for the expression."""
    equations: _Equations = _Equations()
    """Expression math equations."""
    sub_expressions: _SubExpressions = _SubExpressions()
    """Expression named sub-expressions."""
    slices: _SubExpressions = _SubExpressions()
    """Expression named index slices."""
    order: int = 0
    """Order in which to apply this expression relative to all others, if different to its definition order."""

    _group: ClassVar[COMPONENTS_T] = "expressions"


class _Bounds(LinopyBaseModel):
    """
    Bounds of decision variables.

    Either derived per-index item from a multi-dimensional input parameter, or given as
    a single value that is applied across all decision variable index items.
    """

    upper: AttrStr | NumericVal = float("inf")
    """Decision variable upper bound, either as a reference to an input parameter or as a number."""
    lower: AttrStr | NumericVal = float("-inf")
    """Decision variable lower bound, either as a reference to an input parameter or as a number."""


class VariableDef(_MathIndexedComponent):
    """
    Schema for optimisation problem variables.

    A decision variable must be referenced in at least one constraint or in the
    objective for it to exist in the optimisation problem that is sent to the solver.
    """

    unit: str = ""
    """Generalised unit of the component (e.g., length, time, quantity_per_hour, ...)."""
    default: NumericVal = float("nan")
    """If set, will be the default value for the variable."""
    domain: Literal["real", "integer"] = "real"
    """Allowed values that the decision variable can take.
    Either real (a.k.a. continuous) or integer."""
    bounds: _Bounds = _Bounds()

    _group: ClassVar[COMPONENTS_T] = "variables"


class ObjectiveDef(_MathEquationComponent):
    """
    Schema for optimisation problem objectives.

    Only one objective, the one referenced in model configuration `build.objective`
    will be activated for the optimisation problem.
    """

    sense: Literal["min", "max"]
    """Whether the objective function should be minimised or maximised in the
    optimisation."""

    _group: ClassVar[COMPONENTS_T] = "objectives"

    @property
    def foreach(self) -> UniqueList:
        """Dummy field to align with other math components."""
        return []

    @property
    def mask(self) -> str:
        """Dummy field to align with other math components."""
        return "True"


class PostprocessedExpressionDef(ExpressionDef):
    """
    Schema for postprocessed expressions.

    Can be used to combine parameters, variables, and expressions into a single expression solving the model.

    NOTE: If expecting to use postprocessed array `A` in postprocessed array `B`, `A` must
    be defined above `B`.
    """

    _group: ClassVar[COMPONENTS_T] = "postprocessed"


class CheckDef(LinopyBaseModel):
    """Schema for input data checks."""

    mask: str
    """Top-level condition to check"""
    message: str
    """Message to display when the `mask` array returns True, if raising or warning on error."""
    errors: Literal["raise", "warn"] = "raise"
    """How to respond to any instances in which the `mask` array returns True."""
    active: bool = True
    """If False, this check will be ignored during the build phase."""


class DimensionDefs(LinopyDictModel):
    """Linopy model dimensions dictionary."""

    root: dict[AttrStr, DimensionDef] = Field(default_factory=dict)


class ParameterDefs(LinopyDictModel):
    """Linopy model parameters dictionary."""

    root: dict[AttrStr, ParameterDef] = Field(default_factory=dict)


class LookupDefs(LinopyDictModel):
    """Linopy model lookup dictionary."""

    root: dict[AttrStr, LookupDef] = Field(default_factory=dict)


class VariableDefs(LinopyDictModel):
    """Linopy model variables dictionary."""

    root: dict[AttrStr, VariableDef] = Field(default_factory=dict)


class ExpressionDefs(LinopyDictModel):
    """Linopy model expressions dictionary."""

    root: dict[AttrStr, ExpressionDef] = Field(default_factory=dict)


class ConstraintDefs(LinopyDictModel):
    """Linopy model constraints dictionary."""

    root: dict[AttrStr, ConstraintDef] = Field(default_factory=dict)


class PiecewiseConstraintDefs(LinopyDictModel):
    """Linopy model piecewise_constraints dictionary."""

    root: dict[AttrStr, PiecewiseConstraintDef] = Field(default_factory=dict)


class ObjectiveDefs(LinopyDictModel):
    """Linopy model objectives dictionary."""

    root: dict[AttrStr, ObjectiveDef] = Field(default_factory=dict)


class PostprocessedExpressionDefs(LinopyDictModel):
    """Linopy model postprocessed expressions dictionary."""

    root: dict[AttrStr, PostprocessedExpressionDef] = Field(default_factory=dict)


class Checks(LinopyDictModel):
    """Linopy math checks dictionary."""

    root: dict[AttrStr, CheckDef] = Field(default_factory=dict)


class MathModel(LinopyBaseModel):
    """
    Declarative definition of a linopy optimisation problem.

    Contains all mathematical programming components from which a linopy model
    can be built.
    """

    model_config = {"title": "Model math schema"}

    dimensions: DimensionDefs = DimensionDefs()
    """All dimensions to include in the optimisation problem."""
    parameters: ParameterDefs = ParameterDefs()
    """All parameters to include in the optimisation problem."""
    lookups: LookupDefs = LookupDefs()
    """All lookups to include in the optimisation problem."""
    variables: VariableDefs = VariableDefs()
    """All decision variables to include in the optimisation problem."""
    expressions: ExpressionDefs = ExpressionDefs()
    """All expressions that can be applied to the optimisation problem."""
    constraints: ConstraintDefs = ConstraintDefs()
    """All constraints to apply to the optimisation problem."""
    piecewise_constraints: PiecewiseConstraintDefs = PiecewiseConstraintDefs()
    """All _piecewise_ constraints to apply to the optimisation problem."""
    objectives: ObjectiveDefs = ObjectiveDefs()
    """Possible objectives to apply to the optimisation problem."""
    postprocessed: PostprocessedExpressionDefs = PostprocessedExpressionDefs()
    """All postprocessed expressions generated after math has completed."""
    checks: Checks = Checks()
    """Checks to apply before building the optimisation problem."""

    @model_validator(mode="after")
    def unique_component_names(self) -> Self:
        """Ensure all component names are unique."""
        groups = sorted(
            (
                {name for name in getattr(self, field)._active}
                for field in type(self).model_fields
            ),
            key=len,
        )
        seen: set[str] = set()
        duplicates: set[str] = set()
        for field_names in groups:
            duplicates |= field_names & seen
            seen |= field_names
        if duplicates:
            raise ValueError(
                f"Non-unique names in math components: {sorted(duplicates)}."
            )

        return self

    @cached_property
    def parsing_components(self) -> dict[str, dict[str, set[str]]]:
        """
        Return the valid component names available to each parser.

        Returns
        -------
        dict[str, dict[str, set[str]]]
            Per parser kind ("expression" / "mask"), the valid names grouped by
            where they are defined in the math ("dimensions" / "inputs" / "results").
        """
        parsing_components = {
            "dimensions": ["dimensions"],
            "inputs": ["lookups", "parameters"],
            "results": ["variables", "expressions"],
        }

        def _names() -> dict[str, set[str]]:
            return {
                k: set().union(*[getattr(self, i)._active for i in v])
                for k, v in parsing_components.items()
            }

        mask_names = _names()
        all_active = mask_names["results"].union(mask_names["inputs"])
        for component in ["inputs", "results"]:
            all_names = set().union(
                *(getattr(self, k).root for k in parsing_components[component])
            )
            mask_names[component] |= all_names - all_active
        all_components = {"expression": _names(), "mask": mask_names}

        return all_components

    def find(
        self, component: str, subset: Iterable[COMPONENTS_T] | None = None
    ) -> _MathComponent:
        """Find a component in the math schema."""
        fields: Iterable = subset or (set(type(self).model_fields) - {"checks"})

        found = {f for f in fields if component in getattr(self, f)._active}
        if not found:
            raise KeyError(f"Component name `{component}` not found in math schema.")
        if len(found) > 1:
            raise ValueError(
                f"Component name `{component}` found in multiple places: {found}."
            )
        return getattr(self, found.pop())[component]


MATH_DEFS_T = (
    ConstraintDef | VariableDef | ExpressionDef | ObjectiveDef | PiecewiseConstraintDef
)


class ConfigModel(LinopyBaseModel):
    """Base configuration options used when building a Linopy optimisation problem."""

    model_config = {"title": "Model build configuration", "extra": "allow"}
