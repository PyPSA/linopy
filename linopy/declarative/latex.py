"""
Linopy declarative LaTeX math documentation module.

This module builds a human-readable mathematical formulation document from a declarative math definition.
Every active math component is rendered to LaTeX (equations, mask conditions, foreach sets, bounds) without building an optimisation problem.
Metadata and cross-references are then combined together with the LaTeX rendering to produce a complete mathematical formulation document.

This module is adapted from the calliope Apache-2.0 licensed latex backend module:
- https://github.com/calliope-project/calliope/blob/9916116a06ec8c1feaf3c2606bdb8941b916ce85/src/calliope/backend/latex_backend.py
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

import pandas as pd
import xarray as xr

from linopy.declarative import parsing
from linopy.declarative.build import _DeclarativeBase
from linopy.declarative.helpers import HelperFunction, dim_iterator
from linopy.declarative.nodes import Component, Node, find_refs, latex_number
from linopy.declarative.schema import (
    ConstraintDef,
    ExpressionDef,
    LookupDef,
    ObjectiveDef,
    ParameterDef,
    VariableDef,
)

FORMAT_T = Literal["md", "rst", "tex"]

_REPR_STYLES = {
    "parameters": "textit",
    "lookups": "textit",
    "variables": "textbf",
    "expressions": "textbf",
}

DOCUMENTED_GROUPS_T = (
    ParameterDef
    | LookupDef
    | VariableDef
    | ExpressionDef
    | ConstraintDef
    | ObjectiveDef
)
DOCUMENTED_LINOPY_OBJ_GROUPS_T = (
    VariableDef | ExpressionDef | ConstraintDef | ObjectiveDef
)
DOCUMENTED_EXPRESSION_GROUPS_T = ExpressionDef | ConstraintDef | ObjectiveDef
DOCUMENTED_GROUPS: dict[str, str] = {
    "parameters": "Parameters",
    "lookups": "Lookups",
    "variables": "Variables",
    "expressions": "Expressions",
    "constraints": "Constraints",
    "objectives": "Objectives",
}
"""Component groups documented in LaTeX math docs, with their section titles."""


@dataclass
class RenderedComponent:
    """One math component rendered to LaTeX, ready for document generation."""

    group: str
    """Component group (e.g. "constraints")."""

    name: str
    """Component name."""

    title: str = ""
    """The component's long name."""

    description: str = ""
    """The component's verbose description."""

    foreach: str = ""
    """LaTeX `\\forall` line body over the component's sets ("" if adimensional)."""

    mask: str = ""
    """LaTeX rendering of the component's top-level mask ("" if trivially true)."""

    equations: list[dict[str, str]] = field(default_factory=list)
    """Per equation variant, its "expression" LaTeX and (possibly empty) "mask" LaTeX."""

    uses: list[str] = field(default_factory=list)
    """Names of the math components this component references."""

    used_in: list[str] = field(default_factory=list)
    """Names of the math components referencing this component (filled by `build`)."""

    extras: dict[str, str] = field(default_factory=dict)
    """Additional metadata to document (unit, default, sense, ...)."""


class LatexModelBuilder(_DeclarativeBase):
    """
    Builder turning a declarative math definition into LaTeX math documentation.

    The counterpart of :class:`linopy.declarative.build.DeclarativeModelBuilder`
    on the LaTeX route: instead of adding components to a linopy model, it
    renders each component's math strings via the same parsing machinery.

    Examples
    --------
    ```python
    doc = LatexModelBuilder(math_def, input_data).build().generate_math_doc("md")
    ```
    """

    def __init__(
        self,
        math_def: dict,
        input_data: xr.Dataset | None = None,
        config: dict | None = None,
        helpers: Iterable[type[HelperFunction]] = (),
    ) -> None:
        """
        Validate the math definition, ready to `build()` the documentation.

        Parameters
        ----------
        math_def : dict
            Declarative math definition.
        input_data : xr.Dataset, optional
            Model input data; only used to derive the dimensions of parameters
            and lookups for their LaTeX subscripts.
        config : dict, optional
            Build configuration options (available to `config.` mask references).
        helpers : Iterable[type[HelperFunction]], optional
            User-defined helper functions, in addition to the built-in ones.
        """
        super().__init__(math_def, input_data, config, helpers)
        self.components: dict[str, dict[str, RenderedComponent]] = {}
        self._ctx = replace(self._ctx, math_reprs=self._build_math_reprs())

    def _build_math_reprs(self) -> dict[str, str]:
        r"""
        Return the decorated LaTeX representation of every referenceable component.

        Parameters/lookups render as `\textit{name}` and variables/expressions as
        `\textbf{name}`, subscripted with the iterators of the dimensions they are
        indexed over (a variable's `foreach`; an input's dimensions in the data).
        """
        reprs: dict[str, str] = {}
        for group, style in _REPR_STYLES.items():
            for name, definition in getattr(self.math, group)._active.items():
                if isinstance(definition, DOCUMENTED_LINOPY_OBJ_GROUPS_T):
                    dims = definition.foreach
                else:
                    dims = (
                        list(self.input_data[name].dims)
                        if name in self.input_data
                        else []
                    )

                iterators = ",".join(dim_iterator(self.math, str(dim)) for dim in dims)
                subscript = rf"_\text{{{iterators}}}" if iterators else ""
                reprs[name] = rf"\{style}{{{name}}}{subscript}"
        return reprs

    def _foreach_string(self, definition: DOCUMENTED_LINOPY_OBJ_GROUPS_T) -> str:
        r"""Return the LaTeX `\forall` line body for a component's `foreach` sets."""
        if not (sets := definition.foreach):
            return ""
        instrs = ", ".join(
            rf"\text{{{dim_iterator(self.math, dim)}}} \in \text{{{dim}}}"
            for dim in sets
        )
        return rf"\forall{{}} {instrs}"

    def _mask_string(self, mask_node: Node, name: str) -> str:
        """Return the LaTeX rendering of a component's parsed top-level mask ("" if true)."""
        rendered = mask_node.to_latex(
            replace(self._ctx, mode="mask", equation_name=name)
        )
        return "" if rendered == "true" else rendered

    def _render_metadata(self, definition: DOCUMENTED_GROUPS_T) -> dict[str, str]:
        """Return the documentable metadata (unit, default, ...) of a definition."""
        extras: dict[str, str] = {}
        if unit := getattr(definition, "unit", None):
            extras["Unit"] = unit
        if pd.notnull(default := getattr(definition, "default", None)):
            extras["Default"] = str(default)
        return extras

    def add_component(
        self, group: str, name: str, definition: DOCUMENTED_GROUPS_T
    ) -> None:
        """Render one math component and store it under `self.components`."""
        rendered = RenderedComponent(
            group=group,
            name=name,
            title=definition.title,
            description=definition.description,
            extras=self._render_metadata(definition),
        )
        if isinstance(definition, DOCUMENTED_LINOPY_OBJ_GROUPS_T):
            self._add_linopy_obj_component(group, name, rendered, definition)
        self.components.setdefault(definition._group, {})[name] = rendered

    def _add_linopy_obj_component(
        self,
        group: str,
        name: str,
        rendered: RenderedComponent,
        definition: DOCUMENTED_LINOPY_OBJ_GROUPS_T,
    ) -> None:
        parsed = self.parsed[group][name]
        rendered.mask = self._mask_string(parsed.mask, f"{group}:{name}")
        rendered.foreach = self._foreach_string(definition)

        uses = find_refs(parsed.mask, Component)
        if isinstance(definition, DOCUMENTED_EXPRESSION_GROUPS_T):
            self._add_expr_component(rendered, uses, parsed)
        if isinstance(definition, VariableDef):
            self._add_var_component(name, rendered, uses, definition)
        if isinstance(definition, ObjectiveDef):
            self._add_obj_component(rendered, definition)
        # Only cross-reference documented components (not dimensions or strings).
        rendered.uses = sorted(uses & set(self._ctx.math_reprs))

        # Escape special characters in text-mode LaTeX so KaTeX can render names
        # and coordinate values containing e.g. underscores. Done here, on the
        # final display strings, so `math_reprs` stay unescaped for evaluation.
        for rendered_eq in rendered.equations:
            rendered_eq["mask"] = _escape_text_mode(rendered_eq["mask"])
            rendered_eq["expression"] = _escape_text_mode(rendered_eq["expression"])
        rendered.foreach = _escape_text_mode(rendered.foreach)
        rendered.mask = _escape_text_mode(rendered.mask)

    def _add_expr_component(
        self,
        rendered: RenderedComponent,
        uses: set[str],
        parsed: parsing.ParsedComponent,
    ) -> None:
        for equation in parsed.equations:
            equation_ctx = replace(self._ctx, equation_name=equation.name)
            rendered.equations.append(
                {
                    "mask": parsing.as_latex_mask(equation, equation_ctx),
                    "expression": parsing.as_latex_expression(equation, equation_ctx),
                }
            )
            uses |= equation.references()

    def _add_var_component(
        self,
        name: str,
        rendered: RenderedComponent,
        uses: set[str],
        definition: VariableDef,
    ) -> None:
        rendered.extras["Domain"] = definition.domain
        rendered.equations.append(
            {"mask": "", "expression": self._bounds_string(name, definition)}
        )
        uses |= {
            bound
            for bound in (definition.bounds.lower, definition.bounds.upper)
            if isinstance(bound, str)
        }

    def _bounds_string(self, name: str, definition: VariableDef) -> str:
        """Return the LaTeX bounds equation of a decision variable."""
        bounds = definition.bounds
        reprs = self._ctx.math_reprs
        lower, upper = (
            reprs.get(bound, rf"\textit{{{bound}}}")
            if isinstance(bound, str)
            else latex_number(bound)
            for bound in (bounds.lower, bounds.upper)
        )
        return rf"{lower} \leq {reprs[name]} \leq {upper}"

    def _add_obj_component(
        self, rendered: RenderedComponent, definition: ObjectiveDef
    ) -> None:
        rendered.extras["Sense"] = (
            "minimise" if definition.sense == "min" else "maximise"
        )

    def build(self) -> LatexModelBuilder:
        """
        Render all active math components and resolve their cross-references.

        Returns
        -------
        LatexModelBuilder
            Itself, with `self.components` filled, so that document generation
            can be chained (`builder.build().generate_math_doc()`).
        """
        for group in DOCUMENTED_GROUPS:
            for name, definition in getattr(self.math, group)._active.items():
                self.add_component(group, name, definition)

        used_in: dict[str, set[str]] = {}
        for group_components in self.components.values():
            for component in group_components.values():
                for ref in component.uses:
                    used_in.setdefault(ref, set()).add(component.name)
        for group_components in self.components.values():
            for component in group_components.values():
                component.used_in = sorted(used_in.get(component.name, set()))
        return self

    def generate_math_doc(self, format: FORMAT_T = "md") -> str:
        """
        Generate a math documentation string from the rendered components.

        Parameters
        ----------
        format : Literal["md", "rst", "tex"], default: "md"
            Output format: Markdown, reStructuredText, or LaTeX source.

        Returns
        -------
        str
            The full documentation document.
        """
        if not self.components:
            self.build()
        blocks = [_heading(format, 1, "Math formulation"), ""]
        for group, group_title in DOCUMENTED_GROUPS.items():
            group_components = self.components.get(group)
            if not group_components:
                continue
            blocks.extend([_heading(format, 2, group_title), ""])
            for component in group_components.values():
                blocks.extend(_component_doc(format, component))
        return "\n".join(blocks).rstrip() + "\n"


def latex_math_doc(
    math_def: dict,
    input_data: xr.Dataset | None = None,
    config: dict | None = None,
    format: FORMAT_T = "md",
    helpers: Iterable[type[HelperFunction]] = (),
) -> str:
    """
    Generate LaTeX math documentation from a declarative math definition.

    Parameters
    ----------
    math_def : dict
        Declarative math definition (see
        :class:`linopy.declarative.schema.MathModel` for the expected structure).
    input_data : xr.Dataset, optional
        Model input data; only used to derive the dimensions of parameters and
        lookups for their LaTeX subscripts.
    config : dict, optional
        Build configuration options (available to `config.` mask references).
    format : Literal["md", "rst", "tex"], default: "md"
        Output format: Markdown, reStructuredText, or LaTeX source.
    helpers : Iterable[type[HelperFunction]], optional
        User-defined helper functions, in addition to the built-in ones.

    Returns
    -------
    str
        The full documentation document.
    """
    builder = LatexModelBuilder(math_def, input_data, config, helpers)
    return builder.build().generate_math_doc(format)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------


def _tex_text(text: str) -> str:
    """Escape underscores for LaTeX text mode."""
    return text.replace("_", r"\_")


# Matches a `\text`/`\textbf`/`\textit`/... command and captures its (brace-free)
# argument; the special characters within are escaped by `_escape_text_mode`.
_TEXT_CMD_RE = re.compile(r"(\\text(?:bf|it|rm|sf|tt|normal)?)\{([^{}]*)\}")
_TEXT_SPECIAL_RE = re.compile(r"(?<!\\)([_^#%&$])")


def _escape_text_mode(latex: str) -> str:
    r"""
    Escape LaTeX-special characters inside `\text*{...}` arguments.

    KaTeX rejects bare `_`, `^`, ... in text mode, so a component or coordinate
    named e.g. `storage_units` renders as `\text{storage_units}` and raises a
    parse error. Only the *contents* of text commands are escaped: subscript
    operators such as the `_` in `\textbf{x}_\text{n}` sit outside the braces and
    are preserved, and characters already escaped (`\_`) are left untouched.
    """

    def _escape(match: re.Match[str]) -> str:
        cmd, content = match.group(1), match.group(2)
        escaped = _TEXT_SPECIAL_RE.sub(r"\\\1", content)
        return f"{cmd}{{{escaped}}}"

    return _TEXT_CMD_RE.sub(_escape, latex)


def _heading(format: FORMAT_T, level: int, text: str) -> str:
    """Return a document heading at the given level."""
    if format == "md":
        return f"{'#' * level} {text}"
    if format == "rst":
        return f"{text}\n{'=-^'[level - 1] * len(text)}"
    tex_levels = {1: "section", 2: "subsection", 3: "paragraph"}
    return rf"\{tex_levels[level]}{{{_tex_text(text)}}}"


def _metadata_lines(format: FORMAT_T, key: str, value: str) -> list[str]:
    """Return the lines of a "key: value" metadata entry."""
    if format == "tex":
        # A blank line so that each entry renders as its own paragraph.
        return [rf"\textbf{{{key}}}: {_tex_text(value)}", ""]
    return [f"- **{key}**: {value}"]


def _array_block(lines: list[str]) -> str:
    """Return a LaTeX `array` environment of the given lines."""
    joined = " \\\\\n    ".join(lines)
    return f"\\begin{{array}}{{l}}\n    {joined}\n\\end{{array}}"


def _cases_block(equations: list[dict[str, str]]) -> str:
    r"""
    Return a LaTeX `cases` environment, one row per equation variant.

    Each row is the variant's expression, with its own mask (if any) as the
    row's `\text{if }` condition, so that variants sharing a component's
    `foreach`/top-level mask render as sub-clauses at the same nesting level.
    """
    rows = []
    for equation in equations:
        row = equation["expression"]
        if equation["mask"]:
            row += rf" & \quad \text{{if }} {equation['mask']}"
        rows.append(row)
    joined = " \\\\\n    ".join(rows)
    return f"\\begin{{cases}}\n    {joined}\n\\end{{cases}}"


def _math_block(format: FORMAT_T, inner: str) -> list[str]:
    """Return a display-math block wrapping the given LaTeX body."""
    if format == "md":
        return ["$$", inner, "$$", ""]
    if format == "rst":
        indented = "\n".join(f"    {line}" for line in inner.split("\n"))
        return [".. math::", "", indented, ""]
    return [r"\begin{equation}", inner, r"\end{equation}", ""]


def _component_doc(format: FORMAT_T, component: RenderedComponent) -> list[str]:
    """Return the documentation lines of one rendered component."""
    blocks = [_heading(format, 3, component.name), ""]
    if component.title:
        blocks.extend([component.title, ""])
    if component.description:
        blocks.extend([component.description, ""])
    metadata = dict(component.extras)
    if component.uses:
        metadata["Uses"] = ", ".join(component.uses)
    if component.used_in:
        metadata["Used in"] = ", ".join(component.used_in)
    for key, value in metadata.items():
        blocks.extend(_metadata_lines(format, key, value))
    if metadata:
        blocks.append("")
    if component.equations:
        header = []
        if component.foreach:
            header.append(component.foreach)
        if component.mask:
            header.append(rf"\text{{if }} {component.mask}")

        if len(component.equations) == 1:
            equation = component.equations[0]
            lines = [*header]
            if equation["mask"]:
                lines.append(rf"\text{{if }} {equation['mask']}")
            lines.append(equation["expression"])
            inner = _array_block(lines)
        else:
            cases = _cases_block(component.equations)
            inner = f"{_array_block(header)}\n{cases}" if header else cases
        blocks.extend(_math_block(format, inner))
    return blocks
