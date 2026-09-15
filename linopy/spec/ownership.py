"""
Which spec layer owns which name of a model.

A layer builds variables, constraints and expressions, binds model variables
it reads instead of building, declares special-ordered sets and may own the
objective. The model's containers consult this registry before they drop or
shadow a name, so a layer is never left referencing something the model no
longer holds. Names are the unit: what a name holds can drift without the
registry seeing it.

This module imports nothing of ``math_spec``, so ``linopy.model`` can name
the registry without pulling the spec package in.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["variable", "constraint", "expression", "sos"]


def joined(parts: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(parts) < 3:
        return " and ".join(parts)
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


@dataclass
class Ownership:
    """
    Model names by the spec layer that owns them.

    Attributes
    ----------
    variables
        Built variable name to layer.
    bound
        Model variable name to the layer that binds it.
    constraints, expressions, sos
        Constraint, named expression and special-ordered-set variable name to
        layer. An expression is recorded whether it was built into the model
        or folds lazily on read.
    objective
        The layer whose objective the model holds, ``None`` where the
        objective is none of theirs.
    """

    variables: dict[str, str] = field(default_factory=dict)
    bound: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, str] = field(default_factory=dict)
    expressions: dict[str, str] = field(default_factory=dict)
    sos: dict[str, str] = field(default_factory=dict)
    objective: str | None = None

    def _records(self, kind: Kind) -> dict[str, str]:
        return {
            "variable": self.variables,
            "constraint": self.constraints,
            "expression": self.expressions,
            "sos": self.sos,
        }[kind]

    def owner(self, kind: Kind, name: str) -> str | None:
        """The layer owning *name* as a *kind*; for a variable, the one binding it counts too."""
        layer = self._records(kind).get(name)
        if layer is None and kind == "variable":
            return self.bound.get(name)
        return layer

    def claim(
        self,
        layer: str,
        *,
        variables: Iterable[str] = (),
        bound: Iterable[str] = (),
        constraints: Iterable[str] = (),
        expressions: Iterable[str] = (),
        sos: Iterable[str] = (),
        objective: bool = False,
    ) -> None:
        """Record every name *layer* owns."""
        self.variables.update(dict.fromkeys(variables, layer))
        self.bound.update(dict.fromkeys(bound, layer))
        self.constraints.update(dict.fromkeys(constraints, layer))
        self.expressions.update(dict.fromkeys(expressions, layer))
        self.sos.update(dict.fromkeys(sos, layer))
        if objective:
            self.objective = layer

    def refuse_removal(self, kind: Kind, names: Iterable[str]) -> None:
        """Refuse to drop a name a layer builds or binds, which would strand its layer."""
        hit = sorted(n for n in names if self.owner(kind, n) is not None)
        if hit:
            raise ValueError(
                f"{joined(hit)} is declared or bound by a spec layer; a layer "
                "cannot be left referencing a name the model no longer holds."
            )

    def refuse_addition(self, kind: Kind, name: str) -> None:
        """Refuse a name a layer already owns as a *kind*, built, bound or declared lazily."""
        layer = self.owner(kind, name)
        if layer is not None:
            raise ValueError(
                f"{kind} '{name}' is declared or bound by spec layer '{layer}'; "
                f"a layer's names are its own. Pick another name."
            )
