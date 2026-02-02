"""Model registry for benchmarks."""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

_MODELS: dict[str, ModuleType] = {}


def _discover() -> None:
    """Auto-discover model modules in this package."""
    if _MODELS:
        return
    package = importlib.import_module("benchmarks.models")
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"benchmarks.models.{info.name}")
        if hasattr(mod, "build") and hasattr(mod, "SIZES"):
            _MODELS[info.name] = mod


def get_model(name: str) -> ModuleType:
    """Return a model module by name."""
    _discover()
    return _MODELS[name]


def list_models() -> list[str]:
    """Return sorted list of available model names."""
    _discover()
    return sorted(_MODELS)
