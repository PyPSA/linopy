"""
Tests for the declarative math interface.

Covers the grammar (string -> AST), the three evaluation routes (boolean mask
array / linopy expression / LaTeX math string), `$name` sub-expression and slicer
resolution, helper-function registration, and the model builder.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml

from linopy.declarative import grammar, nodes, parsing
from linopy.declarative.build import DeclarativeModelBuilder, declarative_model
from linopy.declarative.helpers import HelperFunction, build_registry
from linopy.declarative.latex import (
    LatexModelBuilder,
    _escape_text_mode,
    latex_math_doc,
)
from linopy.declarative.schema import COMPONENTS_T, ConfigModel, MathModel
from linopy.expressions import LinearExpression
from linopy.variables import Variable

NODES = ["a", "b", "c"]


def _math() -> dict:
    """Minimal but representative math definition exercising every route."""
    return {
        "dimensions": {"node": {"dtype": "string", "iterator": "n"}},
        "parameters": {
            "cost": {"default": 0},
            "cap_max": {"default": float("inf")},
        },
        "variables": {
            "flow": {
                "foreach": ["node"],
                "bounds": {"lower": 0, "upper": float("inf")},
            },
        },
        "expressions": {
            "total_cost": {
                "foreach": ["node"],
                "equations": [{"expression": "flow * cost"}],
            },
            # Pure-parameter expression: evaluates to a DataArray, must be coerced.
            "cost_plus_one": {
                "foreach": ["node"],
                "equations": [{"expression": "cost + 1"}],
            },
            # Sub-expression reference to a variable (regression for the expr route).
            "sub_expr_test": {
                "foreach": ["node"],
                "equations": [{"expression": "$foo * cost"}],
                "sub_expressions": {"foo": [{"expression": "flow"}]},
            },
        },
        "constraints": {
            "cap": {
                "foreach": ["node"],
                "equations": [{"expression": "flow <= cap_max"}],
            },
        },
        "objectives": {
            "obj": {
                "equations": [{"expression": "sum(total_cost, over=node)"}],
                "sense": "min",
            },
        },
    }


def _inputs() -> xr.Dataset:
    return xr.Dataset(
        {
            "cost": ("node", [1.0, 2.0, 3.0]),
            "cap_max": ("node", [10.0, 20.0, 30.0]),
        },
        coords={"node": NODES},
    )


def _ctx(builder: DeclarativeModelBuilder, **kwargs: Any) -> nodes.Context:
    """Build a fresh evaluation context from a builder's validated components."""
    return nodes.Context(
        model=builder.model,
        input_data=builder.input_data,
        math=builder.math,
        config=kwargs.pop("config", builder.config),
        helpers=kwargs.pop("helpers", build_registry()),
        **kwargs,
    )


def _first_equation(
    builder: DeclarativeModelBuilder, group: COMPONENTS_T, name: str
) -> tuple[parsing.Equation, xr.DataArray, nodes.Context]:
    """Return a pre-parsed component's first equation, sub-mask, and context."""
    ctx = _ctx(builder)
    definition = getattr(builder.math, group)[name]
    parsed = builder.parsed[group][name]
    mask = parsing.component_mask(group, name, definition, parsed.mask, ctx)
    equation = parsed.equations[0]
    sub_mask = parsing.as_mask(equation, ctx, initial_mask=mask)
    sub_mask = parsing.drop_dims_not_in_foreach(sub_mask, equation.sets)
    return equation, sub_mask, ctx


@pytest.fixture
def builder_with_flow() -> DeclarativeModelBuilder:
    """A builder with the `flow` variable already added to the model."""
    builder = DeclarativeModelBuilder(_math(), _inputs(), {})
    builder.add_variable("flow", builder.math.variables["flow"])
    return builder


class TestGrammar:
    """String -> AST parsing."""

    NAMES = frozenset({"flow", "cost", "cap_max", "node"})

    def test_equation_tree_shape(self) -> None:
        tree = grammar.equation_grammar(self.NAMES).parse_string(
            "flow <= cap_max", parse_all=True
        )[0]
        assert isinstance(tree, grammar.Compare)
        assert tree.op == "<="
        assert isinstance(tree.lhs, grammar.Component) and tree.lhs.name == "flow"

    def test_equation_rejects_mask_only_operators(self) -> None:
        import pyparsing as pp

        with pytest.raises(pp.ParseException):
            grammar.equation_grammar(self.NAMES).parse_string(
                "flow < cap_max", parse_all=True
            )

    def test_arithmetic_tree_shape(self) -> None:
        tree = grammar.arithmetic_grammar(self.NAMES).parse_string(
            "flow * cost + 1", parse_all=True
        )[0]
        assert isinstance(tree, grammar.Arith)
        assert tree.rest[0][0] == "+"

    def test_sliced_component(self) -> None:
        tree = grammar.arithmetic_grammar(self.NAMES).parse_string(
            "flow[node=$n]", parse_all=True
        )[0]
        assert isinstance(tree, grammar.Sliced)
        assert isinstance(tree.slices["node"], grammar.SliceRef)

    def test_sub_expression_grammar_rejects_refs(self) -> None:
        import pyparsing as pp

        with pytest.raises(pp.ParseException):
            grammar.sub_expression_grammar(self.NAMES).parse_string(
                "$foo + 1", parse_all=True
            )

    def test_find_refs_in_call_kwargs(self) -> None:
        tree = grammar.arithmetic_grammar(self.NAMES).parse_string(
            "sum($foo, over=node) + flow[node=$n]", parse_all=True
        )[0]
        assert grammar.find_refs(tree, grammar.SubExprRef) == {"foo"}
        assert grammar.find_refs(tree, grammar.SliceRef) == {"n"}
        assert grammar.find_refs(tree, grammar.Component) == {"node", "flow"}

    def test_node_repr_is_clean(self) -> None:
        tree = grammar.arithmetic_grammar(self.NAMES).parse_string(
            "flow * cost + 1", parse_all=True
        )[0]
        rendered = repr(tree)
        assert "instring=" not in rendered
        assert "loc=" not in rendered

    def test_parse_error_carries_position_marker(self) -> None:
        math = _math()
        math["constraints"]["cap"]["equations"][0]["expression"] = "flow <= <="
        with pytest.raises(ValueError, match="equations\\[0\\].expression") as excinfo:
            DeclarativeModelBuilder(math, _inputs(), {})
        message = str(excinfo.value)
        assert "constraints:cap:" in message
        assert "^" in message


class TestParseWalkthrough:
    """Whole-dict parse walkthrough with aggregated errors."""

    def test_errors_aggregate_across_components(self) -> None:
        """Broken strings in two components raise as one grouped error."""
        math = _math()
        math["constraints"]["cap"]["equations"][0]["expression"] = "flow <= <="
        math["expressions"]["total_cost"]["equations"][0]["expression"] = (
            "flow * * cost"
        )
        with pytest.raises(ValueError) as excinfo:
            DeclarativeModelBuilder(math, _inputs(), {})
        message = str(excinfo.value)
        assert "constraints:cap:" in message
        assert "expressions:total_cost:" in message
        assert message.count("^") == 2

    def test_undefined_ref_collected_alongside_syntax_errors(self) -> None:
        """A syntax error does not short-circuit undefined `$ref` collection."""
        math = _math()
        math["constraints"]["cap"]["equations"][0]["expression"] = "flow <= <="
        math["expressions"]["sub_expr_test"]["sub_expressions"] = {
            "bar": [{"expression": "flow"}]
        }
        with pytest.raises(ValueError) as excinfo:
            DeclarativeModelBuilder(math, _inputs(), {})
        message = str(excinfo.value)
        assert "constraints:cap:" in message
        assert "expressions:sub_expr_test:" in message
        assert "Undefined sub_expressions" in message

    def test_inactive_components_are_skipped(self) -> None:
        """Inactive components are neither parsed nor built."""
        math = _math()
        math["expressions"]["broken"] = {
            "active": False,
            "foreach": ["node"],
            "equations": [{"expression": "flow * * cost"}],
        }
        builder = DeclarativeModelBuilder(math, _inputs(), {})
        assert "broken" not in builder.parsed["expressions"]
        model = builder.build()
        assert "broken" not in model.expressions

    def test_check_masks_are_parsed(self) -> None:
        math = _math()
        math["checks"] = {"bad": {"mask": "cost > >", "message": "boom"}}
        with pytest.raises(ValueError, match="checks:bad"):
            DeclarativeModelBuilder(math, _inputs(), {})

    def test_inactive_check_masks_are_skipped(self) -> None:
        math = _math()
        math["checks"] = {
            "bad": {"mask": "cost > >", "message": "boom", "active": False}
        }
        builder = DeclarativeModelBuilder(math, _inputs(), {})
        assert "bad" not in builder.parsed.checks

    def test_parsed_math_shape(self) -> None:
        builder = DeclarativeModelBuilder(_math(), _inputs(), {})
        assert set(builder.parsed.components) == {
            "variables",
            "expressions",
            "constraints",
            "objectives",
        }
        assert builder.parsed["variables"]["flow"].equations == []
        assert builder.parsed["constraints"]["cap"].equations


class TestMaskRoute:
    """Mask strings evaluate to boolean arrays."""

    def test_top_level_mask_returns_boolean_dataarray(
        self, builder_with_flow: DeclarativeModelBuilder
    ) -> None:
        _, sub_mask, _ = _first_equation(builder_with_flow, "constraints", "cap")
        assert isinstance(sub_mask, xr.DataArray)
        assert sub_mask.dtype == bool
        assert bool(sub_mask.all())

    def test_mask_comparison_and_subset_and_helper_return_bool(self) -> None:
        math = _math()
        math["constraints"]["cap"]["equations"][0]["mask"] = (
            "cost > 1 and [a, b] in node and any(cap_max, over=node)"
        )
        builder = DeclarativeModelBuilder(math, _inputs(), {})
        builder.add_variable("flow", builder.math.variables["flow"])
        _, sub_mask, _ = _first_equation(builder, "constraints", "cap")
        assert sub_mask.dtype == bool
        # `cost > 1` only holds for nodes b and c.
        assert sub_mask.values.tolist() == [False, True, False]

    @pytest.mark.parametrize(
        "mask_string",
        ["True", "not cost > 1", "cost > 1 or cap_max <= 10", "config.foo == bar"],
    )
    def test_mask_atoms_return_bool(
        self, builder_with_flow: DeclarativeModelBuilder, mask_string: str
    ) -> None:
        node = parsing.parse_mask(mask_string, builder_with_flow.math)
        config = ConfigModel.model_validate({"foo": "bar"})
        result = nodes.evaluate(
            node, _ctx(builder_with_flow, mode="mask", config=config)
        )
        assert isinstance(result, xr.DataArray)
        assert result.dtype == bool

    def test_existence_coercion(
        self, builder_with_flow: DeclarativeModelBuilder
    ) -> None:
        """Bare input references coerce to existence booleans on the mask route."""
        node = parsing.parse_mask("cap_max", builder_with_flow.math)
        result = nodes.evaluate(node, _ctx(builder_with_flow, mode="mask"))
        assert result.values.tolist() == [True, True, True]


class TestExpressionRoute:
    """Expression strings evaluate to linopy expressions."""

    def test_expression_with_variable_returns_linexpr(
        self, builder_with_flow: DeclarativeModelBuilder
    ) -> None:
        equation, sub_mask, ctx = _first_equation(
            builder_with_flow, "expressions", "total_cost"
        )
        result = parsing.as_expression(equation, ctx, mask=sub_mask)
        assert isinstance(result, LinearExpression)

    def test_pure_parameter_expression_coerced_to_linexpr(
        self, builder_with_flow: DeclarativeModelBuilder
    ) -> None:
        equation, sub_mask, ctx = _first_equation(
            builder_with_flow, "expressions", "cost_plus_one"
        )
        result = parsing.as_expression(equation, ctx, mask=sub_mask)
        # No decision variable is involved, but the contract is still LinearExpression.
        assert isinstance(result, LinearExpression)

    def test_sub_expression_reference_returns_linexpr(
        self, builder_with_flow: DeclarativeModelBuilder
    ) -> None:
        equation, sub_mask, ctx = _first_equation(
            builder_with_flow, "expressions", "sub_expr_test"
        )
        assert set(equation.sub_expressions) == {"foo"}
        result = parsing.as_expression(equation, ctx, mask=sub_mask)
        assert isinstance(result, LinearExpression)

    def test_sub_expression_variants_expand_to_cartesian_product(self) -> None:
        """Two variants of one sub-expression yield two equations with merged masks."""
        math = _math()
        math["expressions"]["sub_expr_test"]["sub_expressions"]["foo"] = [
            {"mask": "cost > 1", "expression": "flow"},
            {"mask": "not cost > 1", "expression": "flow * 2"},
        ]
        builder = DeclarativeModelBuilder(math, _inputs(), {})
        builder.add_variable("flow", builder.math.variables["flow"])
        definition = builder.math.expressions["sub_expr_test"]
        equations = parsing.parse_component(
            "expressions", "sub_expr_test", definition, builder.math
        )
        assert len(equations) == 2
        assert {eq.name for eq in equations} == {
            "expressions:sub_expr_test:0-foo:0",
            "expressions:sub_expr_test:0-foo:1",
        }
        # Each equation carries its own mask plus the chosen variant's mask.
        assert all(len(eq.masks) == 2 for eq in equations)
        ctx = _ctx(builder)
        masks = [parsing.as_mask(eq, ctx) for eq in equations]
        # The variant masks are complementary.
        assert not (masks[0] & masks[1]).any()
        assert (masks[0] | masks[1]).all()
        # And the whole component still builds end-to-end.
        builder.add_expression("sub_expr_test", definition)
        assert "sub_expr_test" in builder.model.expressions

    def test_undefined_sub_expression_reference_raises(self) -> None:
        math = _math()
        math["expressions"]["sub_expr_test"]["sub_expressions"] = {
            "bar": [{"expression": "flow"}]
        }
        with pytest.raises(ValueError, match="Undefined sub_expressions"):
            DeclarativeModelBuilder(math, _inputs(), {})

    def test_plain_and_list_slices(
        self, builder_with_flow: DeclarativeModelBuilder
    ) -> None:
        ctx = _ctx(
            builder_with_flow,
            mode="expr",
            mask=xr.full_like(_inputs()["cost"], True, bool),
        )
        arith = grammar.arithmetic_grammar(
            frozenset({"flow", "cost", "cap_max", "node"})
        )
        scalar_sliced = arith.parse_string("flow[node=a] * cost", parse_all=True)[0]
        result = nodes.evaluate(scalar_sliced, ctx)
        assert isinstance(result, LinearExpression)

        list_sliced = arith.parse_string("flow[node=[a, b]]", parse_all=True)[0]
        result = nodes.evaluate(list_sliced, ctx)
        assert result.data.sizes["node"] == 2

    def test_slicer_reference(self, builder_with_flow: DeclarativeModelBuilder) -> None:
        """`$name` slicer references resolve like sub-expressions (feature parity)."""
        math = _math()
        math["expressions"]["sliced"] = {
            "foreach": ["node"],
            "equations": [{"expression": "flow[node=$n] * cost"}],
            "slices": {"n": [{"expression": "a"}]},
        }
        builder = DeclarativeModelBuilder(math, _inputs(), {})
        builder.add_variable("flow", builder.math.variables["flow"])
        definition = builder.math.expressions["sliced"]
        equations = parsing.parse_component(
            "expressions", "sliced", definition, builder.math
        )
        assert len(equations) == 1
        assert set(equations[0].slices) == {"n"}
        builder.add_expression("sliced", definition)
        assert "sliced" in builder.model.expressions


class TestConstraintRoute:
    """Constraint equations evaluate to (lhs, sign, rhs) tuples."""

    def test_equation_returns_lhs_sign_rhs_tuple(
        self, builder_with_flow: DeclarativeModelBuilder
    ) -> None:
        equation, sub_mask, ctx = _first_equation(
            builder_with_flow, "constraints", "cap"
        )
        lhs, sign, rhs = parsing.as_constraint(equation, ctx, mask=sub_mask)
        assert isinstance(lhs, LinearExpression)  # decision variable side
        assert isinstance(rhs, LinearExpression)  # pure-parameter side, coerced
        assert isinstance(sign, xr.DataArray)
        assert set(np.unique(sign.values)) <= {"<="}

    def test_foreach_dim_mismatch_raises(self) -> None:
        math = _math()
        # `sum` removed: the equation is indexed over `node` but foreach is empty.
        math["constraints"]["cap"]["foreach"] = []
        builder = DeclarativeModelBuilder(math, _inputs(), {})
        builder.add_variable("flow", builder.math.variables["flow"])
        equation, sub_mask, ctx = _first_equation(builder, "constraints", "cap")
        with pytest.raises(ValueError, match="not present in `foreach`"):
            parsing.as_constraint(equation, ctx, mask=sub_mask)


class TestLatexRoute:
    """Math strings render as LaTeX."""

    def test_equation_latex(self, builder_with_flow: DeclarativeModelBuilder) -> None:
        equation, _, ctx = _first_equation(builder_with_flow, "constraints", "cap")
        assert parsing.as_latex_expression(equation, ctx) == r"flow \leq cap_max"

    def test_sum_latex(self, builder_with_flow: DeclarativeModelBuilder) -> None:
        equation, _, ctx = _first_equation(builder_with_flow, "objectives", "obj")
        assert (
            parsing.as_latex_expression(equation, ctx)
            == r"\sum\limits_{\substack{\text{n} \in \text{node}}} (total_cost)"
        )

    def test_mask_latex(self) -> None:
        math = _math()
        math["constraints"]["cap"]["equations"][0]["mask"] = (
            "cost > 1 and [a, b] in node"
        )
        builder = DeclarativeModelBuilder(math, _inputs(), {})
        builder.add_variable("flow", builder.math.variables["flow"])
        equation, _, ctx = _first_equation(builder, "constraints", "cap")
        assert parsing.as_latex_mask(equation, ctx) == (
            r"(\textit{cost}\mathord{>}\text{1} \land \text{n} \in \text{[a,b]})"
        )

    def test_mask_infinity_latex_not_wrapped_in_text(self) -> None:
        # `\infty` (and other bare LaTeX commands) must stay in math mode; only
        # plain-text tokens (numbers, coordinate labels, booleans) get `\text{}`.
        math = _math()
        math["constraints"]["cap"]["equations"][0]["mask"] = "cap_max == inf"
        builder = DeclarativeModelBuilder(math, _inputs(), {})
        builder.add_variable("flow", builder.math.variables["flow"])
        equation, _, ctx = _first_equation(builder, "constraints", "cap")
        rendered = parsing.as_latex_mask(equation, ctx)
        assert r"\mathord{==}\infty" in rendered
        assert r"\text{\infty}" not in rendered

    def test_sliced_component_latex(
        self, builder_with_flow: DeclarativeModelBuilder
    ) -> None:
        ctx = _ctx(builder_with_flow)
        arith = grammar.arithmetic_grammar(frozenset({"flow", "node"}))
        tree = arith.parse_string("flow[node=a]", parse_all=True)[0]
        assert nodes.to_math_string(tree, ctx) == r"flow_\text{n=a}"

    def test_identity_operands_are_skipped(
        self, builder_with_flow: DeclarativeModelBuilder
    ) -> None:
        ctx = _ctx(builder_with_flow)
        arith = grammar.arithmetic_grammar(frozenset({"flow"}))
        tree = arith.parse_string("0 + flow", parse_all=True)[0]
        assert nodes.to_math_string(tree, ctx) == "flow"


class _RecordArgs(HelperFunction):
    """Test-only helper that records the types of the arguments it receives."""

    NAME = "record_args"
    ALLOWED_IN = ["expression"]
    received: list[type] = []

    def as_math_string(self, *args: Any, **kwargs: Any) -> str:  # noqa: D102
        return "record_args"

    def as_raw(self, *args: Any, **kwargs: Any) -> LinearExpression | xr.DataArray:  # noqa: D102
        type(self).received.extend(type(a) for a in args)
        # Return the first argument so the enclosing expression stays valid.
        return args[0]


class _Double(HelperFunction):
    """Test-only helper doubling its argument."""

    NAME = "double"
    ALLOWED_IN = ["expression"]

    def as_math_string(self, array: Any) -> str:  # noqa: D102
        return rf"2 \times {array}"

    def as_raw(self, array: Any) -> LinearExpression | xr.DataArray:  # noqa: D102
        return 2 * array


class TestHelpers:
    """Helper-function registration and argument evaluation."""

    def test_helper_arguments_are_evaluated_raw(self) -> None:
        """Helper args arrive un-normalised: raw Variable/DataArray, not LinearExpression."""
        math = _math()
        math["expressions"]["total_cost"]["equations"][0]["expression"] = (
            "record_args(flow, cost)"
        )
        builder = DeclarativeModelBuilder(math, _inputs(), {})
        builder.add_variable("flow", builder.math.variables["flow"])
        ctx = _ctx(builder, helpers=build_registry([_RecordArgs]))
        definition = builder.math.expressions["total_cost"]
        equation = parsing.parse_component(
            "expressions", "total_cost", definition, builder.math
        )[0]
        _RecordArgs.received = []
        parsing.as_expression(equation, ctx)
        assert _RecordArgs.received, "helper was not called"
        assert Variable in _RecordArgs.received
        assert xr.DataArray in _RecordArgs.received
        assert LinearExpression not in _RecordArgs.received

    def test_non_subclass_rejected_by_registry(self) -> None:
        with pytest.raises(ValueError, match="must be subclassed"):
            build_registry([str])  # type: ignore[list-item]

    def test_unknown_helper_rejected(self) -> None:
        math = _math()
        math["expressions"]["total_cost"]["equations"][0]["expression"] = (
            "unknown_helper(flow)"
        )
        builder = DeclarativeModelBuilder(math, _inputs(), {})
        builder.add_variable("flow", builder.math.variables["flow"])
        definition = builder.math.expressions["total_cost"]
        equation = parsing.parse_component(
            "expressions", "total_cost", definition, builder.math
        )[0]
        with pytest.raises(ValueError, match="Invalid helper function"):
            parsing.as_expression(equation, _ctx(builder))

    def test_eval_error_carries_caret(self) -> None:
        """Evaluation errors point a caret at the failing node in the source string."""
        math = _math()
        math["expressions"]["total_cost"]["equations"][0]["expression"] = (
            "unknown_helper(flow)"
        )
        builder = DeclarativeModelBuilder(math, _inputs(), {})
        builder.add_variable("flow", builder.math.variables["flow"])
        definition = builder.math.expressions["total_cost"]
        equation = parsing.parse_component(
            "expressions", "total_cost", definition, builder.math
        )[0]
        with pytest.raises(ValueError) as excinfo:
            parsing.as_expression(equation, _ctx(builder))
        message = str(excinfo.value)
        assert "unknown_helper(flow)" in message
        source_line, caret_line = message.splitlines()[-2:]
        assert caret_line.strip() == "^"
        assert caret_line.index("^") == source_line.index("unknown_helper")

    def test_duplicate_name_rejected(self) -> None:
        class _ClashingSum(HelperFunction):
            NAME = "sum"
            ALLOWED_IN = ["expression"]

            def as_math_string(self, *args: Any, **kwargs: Any) -> str:  # noqa: D102
                return ""

            def as_raw(
                self, *args: Any, **kwargs: Any
            ) -> LinearExpression | xr.DataArray:  # noqa: D102
                return xr.DataArray()

        with pytest.raises(ValueError, match="already exists"):
            build_registry([_ClashingSum])

    def test_custom_helper_end_to_end(self) -> None:
        math = _math()
        math["expressions"]["total_cost"]["equations"][0]["expression"] = (
            "double(flow) * cost"
        )
        model = declarative_model(math, _inputs(), {}, helpers=[_Double])
        assert "total_cost" in model.expressions

    def test_get_val_at_index(self, builder_with_flow: DeclarativeModelBuilder) -> None:
        ctx = _ctx(builder_with_flow)
        arith = grammar.arithmetic_grammar(frozenset({"flow", "node"}))
        tree = arith.parse_string("get_val_at_index(node=0)", parse_all=True)[0]
        assert nodes.evaluate(tree, ctx).item() == "a"


class TestBuilder:
    """Model assembly from parsed math."""

    def test_overlapping_equation_masks_rejected(self) -> None:
        math = _math()
        math["constraints"]["cap"]["equations"] = [
            {"mask": "cost > 0", "expression": "flow <= cap_max"},
            {"mask": "cost > 1", "expression": "flow <= 2 * cap_max"},
        ]
        with pytest.raises(ValueError, match="Overlapping 'mask' conditions"):
            declarative_model(math, _inputs(), {})

    def test_multiple_active_objectives_rejected(self) -> None:
        math = _math()
        math["objectives"]["obj2"] = math["objectives"]["obj"].copy()
        with pytest.raises(ValueError, match="Only one active objective"):
            declarative_model(math, _inputs(), {})

    def test_references_are_sorted_lists(self) -> None:
        model = declarative_model(_math(), _inputs(), {})
        refs = model.constraints["cap"].attrs["references"]
        assert refs == sorted(refs)
        assert isinstance(refs, list)
        assert set(refs) == {"cap_max", "flow"}

    def test_dtype_coercion(self) -> None:
        math = _math()
        math["lookups"] = {
            "flag": {"dtype": "bool", "default": False},
            "label": {"dtype": "string"},
        }
        inputs = _inputs()
        inputs["flag"] = ("node", [1.0, float("nan"), 0.0])
        inputs["label"] = ("node", ["x", "", "z"])
        builder = DeclarativeModelBuilder(math, inputs, {})
        assert builder.input_data["flag"].dtype == bool
        assert builder.input_data["flag"].values.tolist() == [True, False, False]
        # Empty strings are coerced to missing values.
        assert builder.input_data["label"].isnull().sum() == 1

    def test_checks_run_without_active_variable(self) -> None:
        """Input checks must not require an `active` variable in the input data."""
        math = _math()
        math["checks"] = {
            "too_expensive": {
                "mask": "cost > 100",
                "message": "cost too high",
                "errors": "raise",
            }
        }
        # No `active` variable in the inputs, and the check does not trigger.
        declarative_model(math, _inputs(), {})

    def test_check_raises_when_triggered_without_active(self) -> None:
        math = _math()
        math["checks"] = {
            "too_expensive": {
                "mask": "cost > 0",
                "message": "cost too high",
                "errors": "raise",
            }
        }
        with pytest.raises(ValueError, match="cost too high"):
            declarative_model(math, _inputs(), {})

    def test_check_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        math = _math()
        math["checks"] = {
            "pricey": {"mask": "cost > 0", "message": "prices!", "errors": "warn"}
        }
        with caplog.at_level("INFO", logger="linopy.declarative.build"):
            declarative_model(math, _inputs(), {})
        assert "prices!" in caplog.text


class TestLatexDoc:
    """LaTeX math documentation building."""

    def test_components_render_with_decorated_reprs(self) -> None:
        builder = LatexModelBuilder(_math(), _inputs(), {}).build()
        cap = builder.components["constraints"]["cap"]
        assert cap.foreach == r"\forall{} \text{n} \in \text{node}"
        assert cap.equations == [
            {
                "mask": "",
                "expression": r"\textbf{flow}_\text{n} \leq \textit{cap\_max}_\text{n}",
            }
        ]

    def test_variable_bounds_equation(self) -> None:
        math = _math()
        math["variables"]["flow"]["bounds"] = {"lower": 0, "upper": "cap_max"}
        builder = LatexModelBuilder(math, _inputs(), {}).build()
        flow = builder.components["variables"]["flow"]
        assert flow.equations[0]["expression"] == (
            r"0 \leq \textbf{flow}_\text{n} \leq \textit{cap\_max}_\text{n}"
        )
        assert flow.uses == ["cap_max"]

    def test_underscores_escaped_in_text_mode(self) -> None:
        # `cap_max` is a parameter (rendered `\textit{...}`); its underscore must
        # be escaped so KaTeX does not read it as a subscript operator.
        builder = LatexModelBuilder(_math(), _inputs(), {}).build()
        cap = builder.components["constraints"]["cap"]
        assert r"\textit{cap\_max}" in cap.equations[0]["expression"]
        # The subscript operator between the name and its dimension is preserved.
        assert r"\textbf{flow}_\text{n}" in cap.equations[0]["expression"]

    def test_escape_text_mode_escapes_content_only(self) -> None:
        # Underscores inside text commands (names and coordinate values) are
        # escaped, while the subscript operator between them is left intact.
        assert _escape_text_mode(r"\text{storage_units}") == r"\text{storage\_units}"
        assert (
            _escape_text_mode(r"\textbf{p_nom}_\text{n}") == r"\textbf{p\_nom}_\text{n}"
        )
        # Already-escaped underscores are not doubled up.
        assert _escape_text_mode(r"\text{a\_b}") == r"\text{a\_b}"

    def test_no_unescaped_underscore_in_math_text(self) -> None:
        # `cap_max` is a parameter whose underscore is rendered inside `\textit`;
        # no `\text*{...}` argument in the document may hold an unescaped one.
        doc = latex_math_doc(_math(), _inputs(), format="md")
        for arg in re.findall(r"\\text(?:bf|it)?\{([^{}]*)\}", doc):
            assert "_" not in arg.replace(r"\_", "")

    def test_cross_references(self) -> None:
        builder = LatexModelBuilder(_math(), _inputs(), {}).build()
        cost = builder.components["parameters"]["cost"]
        assert set(cost.used_in) == {"cost_plus_one", "sub_expr_test", "total_cost"}
        # Dimensions are not cross-referenced.
        obj = builder.components["objectives"]["obj"]
        assert obj.uses == ["total_cost"]
        assert obj.extras["Sense"] == "minimise"

    def test_equation_masks_render_as_if_conditions(self) -> None:
        math = _math()
        math["constraints"]["cap"]["equations"][0]["mask"] = "cost > 1"
        doc = latex_math_doc(math, _inputs(), format="md")
        assert r"\text{if } (\textit{cost}_\text{n}\mathord{>}\text{1})" in doc

    def test_infinity_not_wrapped_in_text(self) -> None:
        # `cap_max`'s default is `inf`; a mask comparing against it must render
        # `\infty` as a bare math token, never `\text{\infty}` (which KaTeX
        # would print as the literal string, not the symbol).
        math = _math()
        math["constraints"]["cap"]["equations"][0]["mask"] = "cap_max == inf"
        doc = latex_math_doc(math, _inputs(), format="md")
        assert r"\infty" in doc
        assert r"\text{\infty}" not in doc

    def test_sub_expression_variants_produce_multiple_equations(self) -> None:
        math = _math()
        math["expressions"]["sub_expr_test"]["sub_expressions"]["foo"] = [
            {"mask": "cost > 1", "expression": "flow"},
            {"mask": "not cost > 1", "expression": "flow * 2"},
        ]
        builder = LatexModelBuilder(math, _inputs(), {}).build()
        equations = builder.components["expressions"]["sub_expr_test"].equations
        assert len(equations) == 2
        assert equations[1]["expression"] == (
            r"\textbf{flow}_\text{n} \times 2 \times \textit{cost}_\text{n}"
        )

    def test_multiple_equations_render_as_one_block_with_cases(self) -> None:
        # Sub-clauses of a component share one `foreach`/top-level mask and
        # should render as `cases` rows at the same nesting level, not as
        # separate top-level math blocks.
        math = _math()
        math["constraints"]["cap"]["equations"] = [
            {"mask": "cost > 1", "expression": "flow <= cap_max"},
            {"mask": "not cost > 1", "expression": "flow <= 0"},
        ]
        doc = latex_math_doc(math, _inputs(), format="md")
        section = doc.split("### cap\n")[1].split("### ")[0]
        assert section.count(r"\begin{array}{l}") == 1
        assert section.count(r"\begin{cases}") == 1
        assert section.count("$$") == 2  # one opening, one closing delimiter
        assert r"\text{if } (\textit{cost}_\text{n}\mathord{>}\text{1})" in section
        assert (
            r"\text{if } (\neg (\textit{cost}_\text{n}\mathord{>}\text{1}))" in section
        )

    def test_single_equation_renders_inline_without_cases(self) -> None:
        doc = latex_math_doc(_math(), _inputs(), format="md")
        section = doc.split("### total_cost\n")[1].split("### ")[0]
        assert r"\begin{cases}" not in section
        assert r"\begin{array}{l}" in section

    def test_markdown_document_structure(self) -> None:
        doc = latex_math_doc(_math(), _inputs(), format="md")
        assert doc.startswith("# Math formulation")
        for heading in ("## Parameters", "## Variables", "## Constraints", "### cap"):
            assert heading in doc
        assert "$$" in doc

    def test_rst_document_structure(self) -> None:
        doc = latex_math_doc(_math(), _inputs(), format="rst")
        assert ".. math::" in doc
        assert "Math formulation\n================" in doc

    def test_tex_document_structure(self) -> None:
        doc = latex_math_doc(_math(), _inputs(), format="tex")
        assert r"\section{Math formulation}" in doc
        assert r"\begin{equation}" in doc
        # Underscores are escaped in text-mode headings.
        assert r"\paragraph{cap\_max}" in doc


class TestEndToEnd:
    """Full builds from math definitions."""

    def test_declarative_model_end_to_end(self) -> None:
        model = declarative_model(_math(), _inputs(), {})
        assert "flow" in model.variables
        assert "total_cost" in model.expressions
        assert "cap" in model.constraints
        assert model.objective is not None
        # flow is indexed over the node dimension.
        assert set(model.variables["flow"].dims) == {"node"}

    def test_repo_math_yaml_validates_and_parses(self) -> None:
        """The demo math.yaml at the repo root validates and every component parses."""
        math_path = Path(__file__).parent.parent / "math.yaml"
        if not math_path.exists():
            pytest.skip("repo-root math.yaml not present")
        math = MathModel.model_validate(yaml.safe_load(math_path.read_text()))
        for group in ("expressions", "constraints", "objectives"):
            for name, definition in getattr(math, group).root.items():
                equations = parsing.parse_component(group, name, definition, math)
                assert equations, f"{group}:{name} produced no equations"
        # And the full LaTeX math documentation generates in every format.
        for fmt in ("md", "rst", "tex"):
            doc = latex_math_doc(yaml.safe_load(math_path.read_text()), format=fmt)
            name = r"storage\_balance" if fmt == "tex" else "storage_balance"
            assert name in doc
