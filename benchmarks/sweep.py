"""
Cross-version sweep orchestration — build a fresh per-version uv venv,
install the pinned benchmark infra plus a target ``linopy``, and run the
suite (timing) or ``memory save`` (peak RSS) inside it.

The heavy provisioning loop and the two sweep bodies live here so
``cli.py`` stays a thin layer of typer command shims. The CLI resolves
its options (phase → test file, smoke args) and calls :func:`run_sweep`
/ :func:`run_memory_sweep`; everything else — venv creation, isolation,
the per-version subprocess — is internal to this module.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import typer

_PLAIN_VERSION_RE = re.compile(r"^\d+(\.\d+)*([a-z]+\d*)?$")


def _benchmarks_extra_pins() -> list[str]:
    """
    Return the pins from ``pyproject.toml``'s ``[benchmarks]`` extra.

    Both ``sweep`` and ``memory sweep`` install these into each
    per-version venv. Direct pins are kept in pyproject as the single
    source of truth — bump them there and both sweeps pick up the
    change. Transitive deps resolve fresh per venv; uv's deterministic
    resolution gives identical results across versions within one sweep.
    """
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    return list(data["project"]["optional-dependencies"]["benchmarks"])


def _linopy_install_spec(version: str) -> str:
    """Turn ``0.4.0`` → ``linopy==0.4.0``, leave anything URL-y untouched."""
    if _PLAIN_VERSION_RE.match(version):
        return f"linopy=={version}"
    return version


def _snapshot_label(version: str) -> str:
    """
    Filesystem-safe label for a snapshot filename, derived from a spec.

    Plain releases pass through (``0.6.1`` → ``0.6.1``). For a pip spec
    with a ref — ``git+https://…/linopy.git@<sha>`` or ``linopy @ <url>``
    — take the part after the last ``@`` (the sha / tag / branch) so a
    pinned commit writes a clean ``linopy-<sha>.json`` instead of a
    slash-laden, unwritable name. Whatever's chosen is then sanitised to
    ``[0-9A-Za-z._-]``.
    """
    label = version.rsplit("@", 1)[-1] if "@" in version else version
    label = re.sub(r"[^0-9A-Za-z._-]+", "-", label).strip("-._")
    return label or "spec"


def _venv_python(venv: Path) -> Path:
    return (
        venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"
    )


@dataclass(frozen=True)
class _ProvisionedVenv:
    """
    One fresh per-version venv from :func:`_provision_venvs`.

    On success, ``python``, ``env``, and ``import_dir`` are populated
    and ``failed_at`` is ``None``. The caller MUST use ``import_dir``
    as cwd for per-version subprocesses — see :func:`_provision_venvs`
    for why. On failure, ``failed_at`` names the step that failed
    (``"venv"``, ``"install"``, or ``"isolation"``); the caller skips
    its per-version action and records the failure.
    """

    version: str
    python: Path | None
    env: dict[str, str] | None
    import_dir: Path | None
    failed_at: str | None


def _provision_venvs(
    versions: list[str], tmp_prefix: str, as_of: str | None = None
) -> Iterator[_ProvisionedVenv]:
    """
    Yield one fresh per-version uv venv for each linopy version.

    Used by both ``sweep`` and ``memory sweep`` so the venv plumbing
    (uv venv → install ``[benchmarks]`` pins + the target linopy →
    set up an isolated import root) lives in one place. The caller
    supplies the tempdir prefix (so ``ps``/``lsof`` can distinguish
    concurrent runs) and does whatever per-version action it needs.

    **Isolation:** the repo root contains a ``linopy/`` package (the
    one we're developing). Running the per-version pytest with the
    repo root on ``sys.path`` — either via ``PYTHONPATH=repo`` or via
    ``cwd=repo`` (Python prepends cwd as ``''``) — shadows the venv's
    installed linopy with the dev tree. The whole sweep then measures
    the dev linopy against itself instead of the requested version.
    To avoid this, ``import_dir`` is a fresh tempdir per version that
    holds a filtered *copy* of ``benchmarks/`` and nothing else — a
    copy rather than a symlink so the sweep runs on Windows without
    symlink privileges and so no per-version subprocess (nor its
    ``__pycache__`` writes) ever touches the working tree. Running
    subprocesses with ``cwd=import_dir`` and no ``PYTHONPATH`` makes
    ``import benchmarks`` resolve to that copy while ``import linopy``
    falls through to the venv's site-packages — i.e. the requested
    version. The preflight below asserts that resolution actually held.

    Each version's tempdir is cleaned up when the generator advances
    (or exits). The caller can break the loop early — Python's
    generator close protocol fires the ``with`` teardown.

    **Cross-time reproducibility:** if ``as_of`` is a date string
    (``YYYY-MM-DD`` or any ISO 8601 timestamp), passes
    ``--exclude-newer`` to uv so the entire transitive resolution is
    frozen to releases on or before that date. Pinning direct deps
    alone (current default) keeps results reproducible *within* one
    sweep call, but unpinned transitives can drift between sweep calls
    days apart; ``as_of`` closes that gap.
    """
    if shutil.which("uv") is None:
        typer.secho(
            "uv not found on PATH — install via https://docs.astral.sh/uv/",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    repo_root = Path.cwd()
    for version in versions:
        typer.secho(f"\n=== linopy {version} ===", fg=typer.colors.CYAN, bold=True)
        with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmp:
            venv = Path(tmp) / "venv"

            r = subprocess.run(
                ["uv", "venv", "--python", sys.executable, str(venv)],
                check=False,
            )
            if r.returncode != 0:
                typer.secho(
                    f"venv creation failed: {version}",
                    fg=typer.colors.RED,
                    err=True,
                )
                yield _ProvisionedVenv(version, None, None, None, "venv")
                continue

            vpy = _venv_python(venv)
            spec = _linopy_install_spec(version)

            # One install pass: pinned infra + linopy; uv resolves the pinned
            # inputs deterministically into each per-version venv.
            install_args = [
                "uv",
                "pip",
                "install",
                "--python",
                str(vpy),
                *(["--exclude-newer", as_of] if as_of else []),
                *_benchmarks_extra_pins(),
                spec,
            ]
            r = subprocess.run(install_args, check=False)
            if r.returncode != 0:
                typer.secho(f"install failed: {version}", fg=typer.colors.RED, err=True)
                yield _ProvisionedVenv(version, None, None, None, "install")
                continue

            # Isolated import root: a filtered copy of ``benchmarks/`` (docstring).
            import_dir = Path(tmp) / "iso"
            import_dir.mkdir()
            shutil.copytree(
                repo_root / "benchmarks",
                import_dir / "benchmarks",
                ignore=shutil.ignore_patterns("__pycache__", "*.ipynb", ".DS_Store"),
            )

            # Drop PYTHONPATH so the repo's ``linopy/`` can't shadow the venv copy.
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)

            # Preflight: assert the venv's linopy is the one that imports, so a
            # reintroduced shadow bug fails loudly here, not silently per-run.
            preflight = subprocess.run(
                [
                    str(vpy),
                    "-c",
                    (
                        "import linopy; "
                        f"assert {str(venv)!r} in linopy.__file__, "
                        "f'isolation leak: linopy resolved to "
                        "{linopy.__file__}, not the venv'"
                    ),
                ],
                cwd=str(import_dir),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if preflight.returncode != 0:
                typer.secho(
                    f"isolation preflight failed: {version}",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo(preflight.stderr.strip(), err=True)
                yield _ProvisionedVenv(version, None, None, None, "isolation")
                continue

            yield _ProvisionedVenv(version, vpy, env, import_dir, None)


def run_sweep(
    versions: list[str],
    *,
    output_dir: Path,
    test_target: str,
    smoke_args: list[str],
    long: bool = False,
    quick: bool = False,
    rounds: int | None = None,
    filter_expr: str | None = None,
    smoke: bool = False,
    as_of: str | None = None,
    extra_args: list[str] | None = None,
) -> None:
    """
    Timing sweep: run the benchmark suite in each per-version venv.

    ``test_target`` is the pytest target the caller resolved from
    ``--phase`` (or ``benchmarks/``); ``smoke_args`` is the shared smoke
    invocation; ``extra_args`` are trailing args forwarded to pytest. The
    pytest-benchmark JSON snapshot lands in
    ``<output_dir>/linopy-<version>.json``.
    """
    extra_args = extra_args or []

    if quick and long:
        typer.secho(
            "--quick and --long are mutually exclusive",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if smoke and (long or rounds is not None):
        typer.secho(
            "--smoke can't be combined with --long or --rounds "
            "(no timings are recorded in smoke mode).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if not smoke:
        output_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for prov in _provision_venvs(versions, "linopy-bench-", as_of=as_of):
        if prov.failed_at:
            failed.append(prov.version)
            continue

        if smoke:
            # Smoke mode: reuse the same pytest args as the top-level
            # ``smoke`` command. No JSON snapshot, return code is the
            # signal.
            pytest_cmd = [str(prov.python), "-m", "pytest", *smoke_args]
            if filter_expr:
                pytest_cmd.extend(["-k", filter_expr])
            pytest_cmd.extend(extra_args)

            typer.secho(f"$ {' '.join(pytest_cmd)}", fg=typer.colors.BRIGHT_BLACK)
            r = subprocess.run(
                pytest_cmd, env=prov.env, cwd=str(prov.import_dir), check=False
            )
            if r.returncode != 0:
                typer.secho(
                    f"smoke failed: {prov.version}", fg=typer.colors.RED, err=True
                )
                failed.append(prov.version)
            else:
                typer.secho(f"smoke ok: {prov.version}", fg=typer.colors.GREEN)
            continue

        snapshot = (
            output_dir / f"linopy-{_snapshot_label(prov.version)}.json"
        ).resolve()
        pytest_cmd = [
            str(prov.python),
            "-m",
            "pytest",
            test_target,
            "--benchmark-only",
            "--benchmark-json",
            str(snapshot),
        ]
        if quick:
            pytest_cmd.append("--quick")
        elif long:
            pytest_cmd.append("--long")
        if rounds is not None:
            pytest_cmd.extend(
                [f"--benchmark-min-rounds={rounds}", "--benchmark-max-time=0"]
            )

        if filter_expr:
            pytest_cmd.extend(["-k", filter_expr])

        pytest_cmd.extend(extra_args)

        typer.secho(f"$ {' '.join(pytest_cmd)}", fg=typer.colors.BRIGHT_BLACK)
        subprocess.run(pytest_cmd, env=prov.env, cwd=str(prov.import_dir), check=False)

        if snapshot.exists():
            typer.secho(f"saved {snapshot}", fg=typer.colors.GREEN)
        else:
            typer.secho(
                f"no snapshot produced for {prov.version}",
                fg=typer.colors.RED,
                err=True,
            )
            failed.append(prov.version)

    if failed:
        typer.secho(f"\nFailed versions: {failed}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


def run_memory_sweep(
    versions: list[str],
    *,
    output_dir: Path,
    quick: bool = False,
    phases: list[str] | None = None,
    repeats: int = 1,
    as_of: str | None = None,
) -> None:
    """
    Memory sweep: invoke ``memory save`` in each per-version venv.

    Mirrors :func:`run_sweep` but tracks peak RSS. Each version's
    snapshot lands at ``<output_dir>/linopy-<version>.json``.
    """
    from benchmarks.memory import MEMORY_PHASES

    if phases:
        unknown = [p for p in phases if p not in MEMORY_PHASES]
        if unknown:
            typer.secho(
                f"unknown phase(s): {unknown}; valid options: {list(MEMORY_PHASES)}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

    output_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for prov in _provision_venvs(versions, "linopy-mem-", as_of=as_of):
        if prov.failed_at:
            failed.append(prov.version)
            continue
        # ``failed_at is None`` guarantees these are populated (see
        # ``_ProvisionedVenv``); narrow for the type checker.
        assert prov.python is not None and prov.import_dir is not None

        label = f"linopy-{_snapshot_label(prov.version)}"
        mem_cmd = [
            str(prov.python),
            "-m",
            "benchmarks",
            "memory",
            "save",
            label,
        ]
        if quick:
            mem_cmd.append("--quick")
        for ph in phases or []:
            mem_cmd.extend(["--phase", ph])
        if repeats > 1:
            mem_cmd.extend(["--repeats", str(repeats)])

        typer.secho(f"$ {' '.join(mem_cmd)}", fg=typer.colors.BRIGHT_BLACK)
        subprocess.run(mem_cmd, env=prov.env, cwd=str(prov.import_dir), check=False)

        # ``memory save`` writes to ``.benchmarks/memory/<label>.json``
        # relative to its cwd — here, the isolated import_dir. Move it
        # under the user's chosen output_dir (resolves under repo_root
        # by default).
        default_path = prov.import_dir / ".benchmarks" / "memory" / f"{label}.json"
        target = output_dir / f"{label}.json"
        if default_path.exists() and default_path.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            default_path.replace(target)

        if target.exists():
            typer.secho(f"saved {target}", fg=typer.colors.GREEN)
        else:
            typer.secho(
                f"no snapshot produced for {prov.version}",
                fg=typer.colors.RED,
                err=True,
            )
            failed.append(prov.version)

    if failed:
        typer.secho(f"\nFailed versions: {failed}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
