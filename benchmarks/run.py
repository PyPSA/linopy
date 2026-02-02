"""Benchmark orchestrator — main entry point for running benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.models import get_model, list_models
from benchmarks.runners import get_runner, list_phases


def run_single(
    model_name: str,
    phase: str,
    label: str = "dev",
    iterations: int = 30,
    quick: bool = False,
    output_dir: str = "benchmarks/results",
) -> dict:
    """Run one model x one phase, save JSON, return results."""
    model_mod = get_model(model_name)
    runner = get_runner(phase)
    sizes = (
        model_mod.QUICK_SIZES
        if quick and hasattr(model_mod, "QUICK_SIZES")
        else model_mod.SIZES
    )

    results = {
        "label": label,
        "model": model_name,
        "phase": phase,
        "runs": [],
    }

    for kwargs in sizes:
        desc = model_mod.LABEL.format(**kwargs)
        print(f"  {desc} ... ", end="", flush=True)
        res = runner.run(
            label=label,
            builder=model_mod.build,
            builder_args=kwargs,
            iterations=iterations,
        )
        if res is None:
            print("skipped")
            continue
        results["runs"].append(res)
        # Print a compact summary
        summary_parts = []
        for key, val in res.items():
            if key in ("phase", "label", "params", "iterations"):
                continue
            if isinstance(val, float):
                summary_parts.append(f"{key}={val:.3f}")
            elif isinstance(val, int):
                summary_parts.append(f"{key}={val}")
        print(", ".join(summary_parts))

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / f"{label}_{model_name}_{phase}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {filename}")
    return results


def run_phase(
    phase: str,
    label: str = "dev",
    iterations: int = 30,
    quick: bool = False,
    output_dir: str = "benchmarks/results",
) -> list[dict]:
    """Run all models for one phase."""
    all_results = []
    for model_name in list_models():
        print(f"\n[{phase}] Model: {model_name}")
        res = run_single(
            model_name,
            phase,
            label=label,
            iterations=iterations,
            quick=quick,
            output_dir=output_dir,
        )
        all_results.append(res)
    return all_results


def run_all(
    label: str = "dev",
    iterations: int = 30,
    quick: bool = False,
    output_dir: str = "benchmarks/results",
) -> list[dict]:
    """Run all phases x all models."""
    all_results = []
    for phase in list_phases():
        print(f"\n{'=' * 60}")
        print(f"Phase: {phase}")
        print(f"{'=' * 60}")
        results = run_phase(
            phase,
            label=label,
            iterations=iterations,
            quick=quick,
            output_dir=output_dir,
        )
        all_results.extend(results)
    return all_results


def list_available() -> None:
    """Print available models and phases."""
    print("Models:")
    for name in list_models():
        mod = get_model(name)
        desc = getattr(mod, "DESCRIPTION", "")
        print(f"  {name:20s} {desc}")

    print("\nPhases:")
    for phase in list_phases():
        runner = get_runner(phase)
        doc = (runner.run.__doc__ or "").strip().split("\n")[0]
        print(f"  {phase:20s} {doc}")
