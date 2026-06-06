"""Registry introspection commands: ``list``, ``show``, ``filter``."""

from __future__ import annotations

from typing import Annotated

import typer

from benchmarks import PATTERNS, REGISTRY, filter_by, get, get_pattern
from benchmarks.cli._base import SpecKind, app


@app.command("list")
def list_(
    details: Annotated[
        bool,
        typer.Option("--details", "-d", help="Show features and size range."),
    ] = False,
    kind: Annotated[
        SpecKind,
        typer.Option(help="Which specs to list: all (default), models, patterns."),
    ] = "all",
) -> None:
    """
    List the registered specs — models and patterns.

    By default emits one name per line (both kinds) — suitable for piping into
    other tools. ``--kind models`` / ``--kind patterns`` narrows to one;
    ``--details`` shows a per-kind table (features + sizes for models, the
    severity dial's description for patterns).
    """
    show_models = kind in ("all", "models")
    show_patterns = kind in ("all", "patterns")

    if not details:
        if show_models:
            for name in sorted(REGISTRY):
                typer.echo(name)
        if show_patterns:
            for name in sorted(PATTERNS):
                typer.echo(name)
        return

    if show_models:
        rows = [
            (
                spec.name,
                ",".join(sorted(spec.features)),
                f"{spec.sizes[0]}..{spec.sizes[-1]}",
            )
            for spec in REGISTRY.values()
        ]
        name_w = max(len(r[0]) for r in rows)
        feat_w = max(len(r[1]) for r in rows)
        typer.secho("models (sweep size, axis n)", bold=True)
        typer.secho(f"{'name':<{name_w}}  {'features':<{feat_w}}  sizes", dim=True)
        typer.secho("-" * (name_w + feat_w + 20), dim=True)
        for name, feats, sizes in rows:
            typer.secho(f"{name:<{name_w}}", fg=typer.colors.CYAN, nl=False)
            typer.echo(f"  {feats:<{feat_w}}  {sizes}")

    if show_patterns:
        if show_models:
            typer.echo()
        pat_w = max(len(p) for p in PATTERNS)
        typer.secho("patterns (sweep severity 0-100, axis severity)", bold=True)
        typer.secho(f"{'name':<{pat_w}}  description", dim=True)
        typer.secho("-" * (pat_w + 40), dim=True)
        for name in sorted(PATTERNS):
            typer.secho(f"{name:<{pat_w}}", fg=typer.colors.CYAN, nl=False)
            typer.echo(f"  {PATTERNS[name].description}")


@app.command()
def show(
    name: Annotated[str, typer.Argument(help="Spec name (see ``list``).")],
) -> None:
    """
    Print full attributes of one model or pattern spec.

    For a model: sizes, feature tags, applicable phases, the quick / long size
    thresholds, and any ``requires=`` deps. For a pattern: severities, its
    ``description`` (what the dial means), phases, thresholds, and requires.
    """

    def _row(label: str, value: object) -> None:
        typer.secho(f"  {label:<17}", dim=True, nl=False)
        typer.echo(value)

    if name in REGISTRY:
        spec = get(name)
        typer.echo(repr(spec))
        _row("sizes:", spec.sizes)
        _row("features:", sorted(spec.features))
        _row("phases:", sorted(spec.phases))
        _row("quick:", spec.quick_subset)
        _row("long_threshold:", spec.long_threshold)
        if spec.requires:
            _row("requires:", list(spec.requires))
        return

    if name in PATTERNS:
        pattern = get_pattern(name)
        typer.echo(repr(pattern))
        _row("severities:", pattern.severities)
        _row("description:", pattern.description)
        _row("phases:", sorted(pattern.phases))
        _row("quick:", pattern.quick_subset)
        _row("long_threshold:", pattern.long_threshold)
        if pattern.requires:
            _row("requires:", list(pattern.requires))
        return

    typer.secho(f"unknown spec: {name!r}", fg=typer.colors.RED, err=True)
    available = sorted(REGISTRY) + sorted(PATTERNS)
    typer.echo(f"available: {', '.join(available)}", err=True)
    raise typer.Exit(code=2)


@app.command("filter")
def filter_(
    feature: Annotated[
        str | None,
        typer.Option(help="Feature tag, e.g. 'quadratic', 'integer', 'sos'."),
    ] = None,
    phase: Annotated[
        str | None,
        typer.Option(help="Phase tag, e.g. 'to_gurobipy', 'lp_write'."),
    ] = None,
) -> None:
    """
    Filter specs by feature or phase tag.

    Both filters can be combined; the result is the intersection.
    At least one of ``--feature`` / ``--phase`` must be supplied.
    """
    if feature is None and phase is None:
        typer.secho("pass --feature and/or --phase", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    matches = filter_by(has_feature=feature, has_phase=phase)
    for spec in matches:
        typer.echo(repr(spec))
