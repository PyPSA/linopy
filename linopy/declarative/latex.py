"""
Linopy declarative LaTeX math documentation module.

This module builds a human-readable mathematical formulation document from a
declarative math definition, without building an optimisation problem: every
active math component is rendered to LaTeX (equations, mask conditions, foreach
sets, bounds) together with its metadata and cross-references, and the result
can be generated as Markdown, reStructuredText, or LaTeX source.
"""

from __future__ import annotations

import math as pymath
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
from linopy.declarative.schema import DOCUMENTED_GROUPS, EQUATION_GROUPS

FORMAT_T = Literal["md", "rst", "tex"]

_REPR_STYLES = {
    "parameters": "textit",
    "lookups": "textit",
    "variables": "textbf",
    "expressions": "textbf",
}


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
                dims = getattr(definition, "foreach", None)
                if dims is None:
                    dims = (
                        list(self.input_data[name].dims)
                        if name in self.input_data
                        else []
                    )
                iterators = ",".join(dim_iterator(self.math, str(dim)) for dim in dims)
                subscript = rf"_\text{{{iterators}}}" if iterators else ""
                reprs[name] = rf"\{style}{{{name}}}{subscript}"
        return reprs

    def _foreach_string(self, definition: object) -> str:
        r"""Return the LaTeX `\forall` line body for a component's `foreach` sets."""
        sets = getattr(definition, "foreach", [])
        if not sets:
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

    def _render_metadata(self, definition: object) -> dict[str, str]:
        """Return the documentable metadata (unit, default, ...) of a definition."""
        extras: dict[str, str] = {}
        unit = getattr(definition, "unit", "")
        if unit:
            extras["Unit"] = unit
        default = getattr(definition, "default", None)
        if default is not None and pd.notna(default):
            if isinstance(default, int | float) and pymath.isinf(default):
                extras["Default"] = "inf" if default > 0 else "-inf"
            else:
                extras["Default"] = str(default)
        return extras

    def add_component(self, group: str, name: str, definition: object) -> None:
        """Render one math component and store it under `self.components`."""
        mask_node = parsing.parse_mask(
            getattr(definition, "mask", "True"), self.math, f"{group}:{name}"
        )
        rendered = RenderedComponent(
            group=group,
            name=name,
            title=getattr(definition, "title", ""),
            description=getattr(definition, "description", ""),
            foreach=self._foreach_string(definition),
            mask=self._mask_string(mask_node, f"{group}:{name}"),
            extras=self._render_metadata(definition),
        )
        uses = find_refs(mask_node, Component)

        if group in EQUATION_GROUPS:
            equations = parsing.parse_component(group, name, definition, self.math)  # type: ignore[arg-type]
            for equation in equations:
                equation_ctx = replace(self._ctx, equation_name=equation.name)
                rendered.equations.append(
                    {
                        "mask": parsing.as_latex(equation, equation_ctx, what="mask"),
                        "expression": parsing.as_latex(equation, equation_ctx),
                    }
                )
                uses |= equation.references()
        elif group == "variables":
            rendered.extras["Domain"] = definition.domain  # type: ignore[attr-defined]
            rendered.equations.append(
                {"mask": "", "expression": self._bounds_string(name, definition)}
            )
            uses |= {
                bound
                for bound in (definition.bounds.lower, definition.bounds.upper)  # type: ignore[attr-defined]
                if isinstance(bound, str)
            }
        if group == "objectives":
            rendered.extras["Sense"] = (
                "minimise" if definition.sense == "min" else "maximise"  # type: ignore[attr-defined]
            )

        # Only cross-reference documented components (not dimensions or strings).
        rendered.uses = sorted(uses & set(self._ctx.math_reprs))

        # Escape special characters in text-mode LaTeX so KaTeX can render names
        # and coordinate values containing e.g. underscores. Done here, on the
        # final display strings, so `math_reprs` stay unescaped for evaluation.
        rendered.foreach = _escape_text_mode(rendered.foreach)
        rendered.mask = _escape_text_mode(rendered.mask)
        for rendered_eq in rendered.equations:
            rendered_eq["mask"] = _escape_text_mode(rendered_eq["mask"])
            rendered_eq["expression"] = _escape_text_mode(rendered_eq["expression"])

        self.components.setdefault(group, {})[name] = rendered

    def _bounds_string(self, name: str, definition: object) -> str:
        """Return the LaTeX bounds equation of a decision variable."""
        bounds = definition.bounds  # type: ignore[attr-defined]
        reprs = self._ctx.math_reprs
        lower, upper = (
            reprs.get(bound, rf"\textit{{{bound}}}")
            if isinstance(bound, str)
            else latex_number(bound)
            for bound in (bounds.lower, bounds.upper)
        )
        return rf"{lower} \leq {reprs[name]} \leq {upper}"

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


def _math_block(format: FORMAT_T, lines: list[str]) -> list[str]:
    """Return a display-math block wrapping an `array` of the given LaTeX lines."""
    joined = " \\\\\n    ".join(lines)
    array = f"\\begin{{array}}{{l}}\n    {joined}\n\\end{{array}}"
    if format == "md":
        return ["$$", array, "$$", ""]
    if format == "rst":
        indented = "\n".join(f"    {line}" for line in array.split("\n"))
        return [".. math::", "", indented, ""]
    return [r"\begin{equation}", array, r"\end{equation}", ""]


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
    for equation in component.equations:
        lines = []
        if component.foreach:
            lines.append(component.foreach)
        for mask in (component.mask, equation["mask"]):
            if mask:
                lines.append(rf"\text{{if }} {mask}")
        lines.append(equation["expression"])
        blocks.extend(_math_block(format, lines))
    return blocks
