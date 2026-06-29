#!/usr/bin/env python3
"""
Linopy module for solving lp files with different solvers.
"""

from __future__ import annotations

import contextlib
import enum
import functools
import io
import logging
import os
import re
import shutil
import subprocess as sub
import sys
import threading
import warnings
from abc import ABC
from collections import namedtuple
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, NamedTuple, TypeVar

import numpy as np
import pandas as pd
from packaging.specifiers import SpecifierSet
from packaging.version import parse as parse_version
from scipy.sparse import tril, triu

import linopy.io
from linopy.common import count_initial_letters, values_to_lookup_array
from linopy.constants import (
    EQUAL,
    SOS_DIM_ATTR,
    SOS_TYPE_ATTR,
    Result,
    Solution,
    SolverReport,
    SolverStatus,
    Status,
    TerminationCondition,
    short_GREATER_EQUAL,
    short_LESS_EQUAL,
)
from linopy.persistent import (
    ModelDiff,
    ModelSnapshot,
    RebuildReason,
    RebuildRequiredError,
    UnsupportedUpdate,
    UpdatesDisabledError,
    VarKind,
    clear_coef_dirty,
)


def _int_list(arr: np.ndarray, dtype: type = np.int64) -> list[int]:
    return arr.astype(dtype, copy=False).tolist()


def _float_list(arr: np.ndarray) -> list[float]:
    return arr.astype(float, copy=False).tolist()


def _parse_int_label(name: str) -> int:
    """Strip leading non-digits and parse the integer label."""
    s = str(name)
    cutoff = count_initial_letters(s)
    try:
        return int(s[cutoff:])
    except ValueError:
        return int(re.sub(r".*#", "", s))


def _names_to_labels(names: Any) -> np.ndarray:
    """Vectorised conversion of solver-provided names to integer labels."""
    if len(names) == 0:
        return np.array([], dtype=np.int64)
    index = pd.Index(names)
    if pd.api.types.is_integer_dtype(index.dtype):
        return index.to_numpy(dtype=np.int64)
    string_index = index.astype(str)
    cutoff = count_initial_letters(str(string_index[0]))
    try:
        return string_index.str[cutoff:].astype(np.int64).to_numpy(dtype=np.int64)
    except (TypeError, ValueError):
        try:
            return (
                string_index.str.replace(r".*#", "", regex=True)
                .astype(np.int64)
                .to_numpy(dtype=np.int64)
            )
        except (TypeError, ValueError):
            return np.fromiter(
                (_parse_int_label(n) for n in names), dtype=np.int64, count=len(names)
            )


def _solution_from_names(values: np.ndarray, names: Any, size: int) -> np.ndarray:
    """
    Build a label-indexed dense solution array of length ``size`` from
    solver-side names. Used by paths where the solver may iterate in arbitrary
    order or drop unused entities (file-based LP solvers, the ``from_file``
    paths of Highs/Gurobi).
    """
    if not size:
        return np.array([], dtype=float)
    return values_to_lookup_array(
        np.asarray(values, dtype=float), _names_to_labels(names), size=size
    )


def _solution_from_labels(
    values: np.ndarray, labels: np.ndarray | None, size: int
) -> np.ndarray:
    """Scatter solver-side values into a label-indexed dense array of length ``size``."""
    if not size:
        return np.array([], dtype=float)
    assert labels is not None
    return values_to_lookup_array(np.asarray(values, dtype=float), labels, size=size)


def _iter_sos_sets(model: Model) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(sos_type, positions, weights)`` per active SOS set in ``model``."""
    label_to_pos = model.variables.label_index.label_to_pos
    for var_name in model.variables.sos:
        var = model.variables.sos[var_name]
        sos_type = int(var.attrs[SOS_TYPE_ATTR])  # type: ignore[call-overload]
        sos_dim = str(var.attrs[SOS_DIM_ATTR])

        labels = var.labels.transpose(sos_dim, ...)
        weights = labels.coords[sos_dim].values
        arr = labels.values.reshape(labels.shape[0], -1)

        for i in range(arr.shape[1]):
            col = arr[:, i]
            mask = col != -1
            if mask.any():
                yield sos_type, label_to_pos[col[mask]], weights[mask]


class SolverFeature(Enum):
    """Enumeration of all solver capabilities tracked by linopy."""

    INTEGER_VARIABLES = auto()
    QUADRATIC_OBJECTIVE = auto()
    DIRECT_API = auto()
    LP_FILE_NAMES = auto()
    READ_MODEL_FROM_FILE = auto()
    SOLUTION_FILE_NOT_NEEDED = auto()
    GPU_ACCELERATION = auto()
    GPU_ONLY = auto()
    IIS_COMPUTATION = auto()
    SOS_CONSTRAINTS = auto()
    INDICATOR_CONSTRAINTS = auto()
    SEMI_CONTINUOUS_VARIABLES = auto()
    SOLVER_ATTRIBUTE_ACCESS = auto()
    MIP_DUAL_BOUND_REPORT = auto()


def _installed_version_in(pkg: str, spec: str) -> bool:
    """Check whether the installed version of `pkg` satisfies `spec`."""
    try:
        return package_version(pkg) in SpecifierSet(spec)
    except PackageNotFoundError:
        return False


if TYPE_CHECKING:
    import cupdlpx
    import gurobipy
    import highspy
    import mosek

    from linopy.model import Model

EnvType = TypeVar("EnvType")

FILE_IO_APIS = ["lp", "lp-polars", "mps"]
IO_APIS = FILE_IO_APIS + ["direct"]


def _run_highs_with_keyboard_interrupt(h: Any) -> None:
    """
    Run `highspy.Highs.run()` while ensuring Ctrl-C cancels the solve.

    HiGHS can run for a long time inside a C-extension call. Running it in a
    worker thread allows the main thread to reliably receive KeyboardInterrupt
    and signal HiGHS to stop via `cancelSolve()`.
    """

    handle_keyboard_interrupt = getattr(h, "HandleKeyboardInterrupt", None)
    handle_user_interrupt = getattr(h, "HandleUserInterrupt", None)

    old_handle_keyboard_interrupt = (
        handle_keyboard_interrupt if not callable(handle_keyboard_interrupt) else None
    )
    old_handle_user_interrupt = (
        handle_user_interrupt if not callable(handle_user_interrupt) else None
    )

    try:
        if callable(handle_keyboard_interrupt):
            handle_keyboard_interrupt(True)
        elif handle_keyboard_interrupt is not None:
            h.HandleKeyboardInterrupt = True

        if callable(handle_user_interrupt):
            handle_user_interrupt(True)
        elif handle_user_interrupt is not None:
            h.HandleUserInterrupt = True

        finished = threading.Event()
        run_error: BaseException | None = None

        def _target() -> None:
            nonlocal run_error
            try:
                h.run()
            except BaseException as exc:  # pragma: no cover
                run_error = exc
            finally:
                finished.set()

        thread = threading.Thread(target=_target, name="linopy-highs-run", daemon=True)
        thread.start()

        try:
            while not finished.wait(0.1):
                pass
        except KeyboardInterrupt:
            cancel_solve = getattr(h, "cancelSolve", None)
            if callable(cancel_solve):
                with contextlib.suppress(Exception):
                    cancel_solve()
            while not finished.wait(0.1):
                pass
            raise

        if run_error is not None:
            raise run_error
    finally:
        if old_handle_keyboard_interrupt is not None:
            h.HandleKeyboardInterrupt = old_handle_keyboard_interrupt
        if old_handle_user_interrupt is not None:
            h.HandleUserInterrupt = old_handle_user_interrupt


# xpress.Namespaces was added in xpress 9.6. Importing xpress is pure-Python
# and does not acquire a license, so this shim stays eager so downstream code
# can ``from linopy.solvers import xpress_Namespaces``.
with contextlib.suppress(ModuleNotFoundError, ImportError):
    import xpress  # noqa: F401

    try:
        from xpress import Namespaces as xpress_Namespaces
    except ImportError:

        class xpress_Namespaces:  # type: ignore[no-redef]
            ROW = 1
            COLUMN = 2
            SET = 3


class _LazyModule:
    """
    Module proxy that imports the underlying package on first attribute access.

    Lets us keep ``gurobipy.Env`` / ``mindoptpy.read`` references throughout the
    file while deferring the actual ``import`` (and its license-server side
    effects, for mindoptpy/coptpy) until a Solver subclass really needs them.
    """

    __slots__ = ("_name", "_module")

    def __init__(self, name: str) -> None:
        self._name = name
        self._module: Any = None

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        if self._module is None:
            import importlib

            self._module = importlib.import_module(self._name)
        return getattr(self._module, attr)


gurobipy = _LazyModule("gurobipy")  # type: ignore[assignment]
highspy = _LazyModule("highspy")  # type: ignore[assignment]
scip = _LazyModule("pyscipopt")
cplex = _LazyModule("cplex")
knitro = _LazyModule("knitro")
mosek = _LazyModule("mosek")
mindoptpy = _LazyModule("mindoptpy")
coptpy = _LazyModule("coptpy")
cupdlpx = _LazyModule("cupdlpx")


def _has_module(name: str) -> bool:
    """True if ``name`` is importable, without executing its ``__init__``."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@functools.cache
def _new_highspy_mps_layout() -> bool:
    """True for highspy >= 1.7.1 (new MPS coefficient layout)."""
    if not _has_module("highspy"):
        return False
    try:
        return parse_version(package_version("highspy")) >= parse_version("1.7.1")
    except PackageNotFoundError:
        return False


logger = logging.getLogger(__name__)


io_structure = dict(
    lp_file={
        "gurobi",
        "xpress",
        "cbc",
        "glpk",
        "cplex",
        "mosek",
        "mindopt",
    },
    blocks={"pips"},
)


# using enum to match solver subclasses with names
class SolverName(enum.Enum):
    CBC = "cbc"
    GLPK = "glpk"
    Highs = "highs"
    Cplex = "cplex"
    Gurobi = "gurobi"
    SCIP = "scip"
    Xpress = "xpress"
    Knitro = "knitro"
    Mosek = "mosek"
    COPT = "copt"
    MindOpt = "mindopt"
    PIPS = "pips"
    cuPDLPx = "cupdlpx"


def path_to_string(path: Path) -> str:
    """
    Convert a pathlib.Path to a string.
    """
    return str(path.resolve())


def read_sense_from_problem_file(problem_fn: Path | str) -> str:
    with open(problem_fn) as file:
        f = file.read()
    file_format = read_io_api_from_problem_file(problem_fn)
    if file_format == "lp":
        return "min" if "min" in f.lower() else "max"
    elif file_format == "mps":
        return "max" if "OBJSENSE\n  MAX\n" in f else "min"
    else:
        msg = "Unsupported problem file format."
        raise ValueError(msg)


def read_io_api_from_problem_file(problem_fn: Path | str) -> str:
    if isinstance(problem_fn, Path):
        return problem_fn.suffix[1:]
    else:
        return problem_fn.split(".")[-1]


def maybe_adjust_objective_sign(
    solution: Solution, io_api: str | None, sense: str | None
) -> Solution:
    if sense == "min":
        return solution
    if np.isnan(solution.objective):
        return solution
    if io_api == "mps" and not _new_highspy_mps_layout():
        logger.info(
            "Adjusting objective sign due to switched coefficients in MPS file."
        )
        solution.objective *= -1
    return solution


@dataclass(frozen=True)
class LicenseStatus:
    """Result of :meth:`Solver.license_status` — license/runtime probe outcome."""

    solver: str
    ok: bool
    message: str | None = None

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class Solver(ABC, Generic[EnvType]):
    """
    Abstract base class for solving a given linear problem.

    Subclasses provide ``_build_direct`` / ``_run_direct`` (when supporting the
    direct API) and ``_run_file`` (when supporting LP/MPS files). Construction
    goes via :meth:`Solver.from_name` or :meth:`Solver.from_model`.

    ``track_updates`` toggles persistent-update support:

    * ``False`` (default) — one-shot mode. No :class:`ModelSnapshot` is
      captured at build time; any later ``solve(model=...)`` or
      ``update(model)`` raises :class:`UpdatesDisabledError`. Use for
      throw-away solver instances and high-level ``Model.solve(...)``.
    * ``True`` — long-lived mode. A snapshot is captured at build time and
      re-captured after each successful in-place update, enabling
      diff-based ``solve(model=...)`` / ``update(model)`` across iterations.
    """

    model: Model | None = None
    io_api: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    track_updates: bool = False

    # Runtime state — never set via constructor.
    status: Status | None = field(init=False, default=None, repr=False)
    solution: Solution | None = field(init=False, default=None, repr=False)
    report: SolverReport | None = field(init=False, default=None, repr=False)
    solver_model: Any = field(init=False, default=None, repr=False)
    sense: str | None = field(init=False, default=None, repr=False)
    env: Any = field(init=False, default=None, repr=False)
    _env_stack: contextlib.ExitStack | None = field(
        init=False, default=None, repr=False
    )
    _vlabels: np.ndarray | None = field(init=False, default=None, repr=False)
    _clabels: np.ndarray | None = field(init=False, default=None, repr=False)
    _n_vars: int = field(init=False, default=0, repr=False)
    _n_cons: int = field(init=False, default=0, repr=False)
    _problem_fn: Path | None = field(init=False, default=None, repr=False)

    snapshot: ModelSnapshot | None = field(init=False, default=None, repr=False)
    _rebuilds: int = field(init=False, default=0, repr=False)
    _in_place_updates: int = field(init=False, default=0, repr=False)
    _last_rebuild_reason: RebuildReason | None = field(
        init=False, default=None, repr=False
    )

    display_name: ClassVar[str] = ""
    features: ClassVar[frozenset[SolverFeature]] = frozenset()
    accepted_io_apis: ClassVar[frozenset[str]] = frozenset()
    supports_persistent_update: ClassVar[bool] = False
    supports_sign_update: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if type(self) is Solver:
            raise TypeError(
                "Solver is abstract; instantiate a concrete subclass instead."
            )
        if not type(self).is_available():
            msg = (
                f"Solver package for '{self.solver_name.value}' is not installed. "
                "Please install first to initialize solver instance."
            )
            raise ImportError(msg)
        self._lock: threading.Lock = threading.Lock()

    def apply_update(
        self,
        diff: ModelDiff,
        var_label_index: Any,
        con_label_index: Any,
    ) -> None:
        """
        Apply an in-place :class:`ModelDiff` to the built native model.

        Template method: validates the diff up front (a rejected update
        leaves the native model untouched), then walks the sections in a
        fixed order, dispatching to the per-backend ``_apply_*`` hooks.
        """
        if not self.supports_persistent_update:
            raise UnsupportedUpdate(type(self).__name__)
        self._validate_update(diff)
        ctx = self._apply_begin(var_label_index, con_label_index)
        if diff.var_bounds_indices.size:
            self._apply_var_bounds(
                ctx,
                diff.var_bounds_indices,
                diff.var_bounds_lower,
                diff.var_bounds_upper,
            )
        if diff.var_type_positions.size:
            self._apply_var_types(ctx, diff.var_type_positions, diff.var_type_kinds)
            self._reclamp_binary_bounds(
                ctx, diff.var_type_positions, diff.var_type_kinds
            )
        if diff.con_rhs_indices.size:
            self._apply_con_rhs(ctx, diff)
        if diff.con_sign_indices.size:
            self._apply_con_signs(ctx, diff.con_sign_indices, diff.con_sign_values)
        if diff.n_coef_updates:
            self._apply_con_coefs(
                ctx, diff.con_coef_rows, diff.con_coef_cols, diff.con_coef_vals
            )
        if diff.obj_c_indices is not None:
            assert diff.obj_c_values is not None
            self._apply_obj_linear(ctx, diff.obj_c_indices, diff.obj_c_values)
        if diff.obj_sense is not None:
            self._apply_obj_sense(ctx, diff.obj_sense)
            self.sense = diff.obj_sense
        self._apply_end(ctx)

    def _validate_update(self, diff: ModelDiff) -> None:
        """Reject unsupported diff content before any native mutation."""
        if diff.con_sign_indices.size and not self.supports_sign_update:
            raise UnsupportedUpdate(
                f"{self.display_name} does not support in-place constraint sign change"
            )

    def _apply_begin(self, var_label_index: Any, con_label_index: Any) -> Any:
        """Backend prep + validation; the return value is passed to every hook."""
        return self.solver_model

    def _apply_end(self, ctx: Any) -> None:
        return None

    def _apply_var_bounds(
        self, ctx: Any, indices: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> None:
        raise NotImplementedError

    def _apply_var_types(
        self, ctx: Any, positions: np.ndarray, kinds: np.ndarray
    ) -> None:
        raise NotImplementedError

    def _reclamp_binary_bounds(
        self, ctx: Any, positions: np.ndarray, kinds: np.ndarray
    ) -> None:
        """
        Re-clamp variables switched to BINARY to [0, 1].

        Compensates for backends whose native type system only has a generic
        integer kind; backends where the binary type implies the bounds
        (Gurobi) override with a no-op.
        """
        binary_mask = kinds == VarKind.BINARY
        if binary_mask.any():
            bin_positions = positions[binary_mask]
            n = bin_positions.size
            self._apply_var_bounds(ctx, bin_positions, np.zeros(n), np.ones(n))

    def _apply_con_rhs(self, ctx: Any, diff: ModelDiff) -> None:
        raise NotImplementedError

    def _apply_con_signs(
        self, ctx: Any, indices: np.ndarray, signs: np.ndarray
    ) -> None:
        raise NotImplementedError

    def _apply_con_coefs(
        self, ctx: Any, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray
    ) -> None:
        raise NotImplementedError

    def _apply_obj_linear(
        self, ctx: Any, indices: np.ndarray, values: np.ndarray
    ) -> None:
        raise NotImplementedError

    def _apply_obj_sense(self, ctx: Any, sense: str) -> None:
        raise NotImplementedError

    @property
    def solver_options(self) -> dict[str, Any]:
        return self.options

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        """
        Return True if this solver's package/binary is importable.

        Must not acquire a license. Subclasses override with the cheapest
        possible probe. Base returns False so a forgotten override fails
        safe (the solver simply does not show up in ``available_solvers``).
        """
        return False

    @classmethod
    def license_status(cls) -> LicenseStatus:
        """
        Probe license/runtime availability. May acquire a license slot.

        Not cached — license state is mutable (server reachability, expiry).
        """
        name = SolverName[cls.__name__].value
        if not cls.is_available():
            return LicenseStatus(name, ok=False, message="package not installed")
        try:
            cls._license_probe()
        except Exception as e:
            return LicenseStatus(name, ok=False, message=f"{type(e).__name__}: {e}")
        return LicenseStatus(name, ok=True)

    @classmethod
    def _license_probe(cls) -> None:
        """Subclass hook. Default no-op. Raises on failure."""
        return None

    @classmethod
    def runtime_features(cls) -> frozenset[SolverFeature]:
        """
        Features whose availability depends on the installed solver version
        or runtime environment. Override in subclasses; the default is empty.
        """
        return frozenset()

    @classmethod
    def supported_features(cls) -> frozenset[SolverFeature]:
        """All features supported by this solver, static plus runtime."""
        return cls.features | cls.runtime_features()

    @classmethod
    def supports(cls, feature: SolverFeature) -> bool:
        """Check if this solver supports a given feature."""
        return feature in cls.features or feature in cls.runtime_features()

    @staticmethod
    def from_name(
        name: str,
        model: Model | None = None,
        io_api: str | None = None,
        options: dict[str, Any] | None = None,
        track_updates: bool = False,
        **build_kwargs: Any,
    ) -> Solver:
        """
        Construct the solver subclass registered as ``name``.

        With ``model`` supplied, the solver is built immediately. Without it,
        an unbuilt instance is returned and the first ``solve(model, ...)``
        call performs the build. See :class:`Solver` for ``track_updates``.
        """
        cls = _solver_class_for(name)
        if cls is None:
            raise ValueError(f"unknown solver: {name}")
        if model is None:
            return cls(
                model=None,
                io_api=io_api,
                options=options or {},
                track_updates=track_updates,
            )
        return cls.from_model(
            model,
            io_api=io_api,
            options=options or {},
            track_updates=track_updates,
            **build_kwargs,
        )

    @classmethod
    def from_model(
        cls,
        model: Model,
        io_api: str | None = None,
        options: dict[str, Any] | None = None,
        track_updates: bool = False,
        **build_kwargs: Any,
    ) -> Solver:
        """Instantiate and build the solver against ``model``."""
        instance = cls(
            model=model,
            io_api=io_api,
            options=options or {},
            track_updates=track_updates,
        )
        instance._build(**build_kwargs)
        return instance

    def _build(self, **build_kwargs: Any) -> None:
        """
        Dispatch to direct or file build based on ``io_api``.

        The Solver never mutates ``self.model``. Constraint sanitization
        (``model.constraints.sanitize_zeros()`` /
        ``.sanitize_infinities()``) and SOS reformulation
        (``model.apply_sos_reformulation()``) are Model-level operations
        the caller applies first; this builder consumes whatever shape it
        is handed.
        """
        if self.model is None:
            raise RuntimeError("Solver has no model attached; cannot build.")
        self._validate_model()
        if self.io_api == "direct":
            self._build_direct(**build_kwargs)
            if self.track_updates:
                self.snapshot = ModelSnapshot.capture(self.model)
                clear_coef_dirty(self.model)
        else:
            self._build_file(**build_kwargs)

    def _validate_model(self) -> None:
        """Pre-build checks on whether this solver can handle ``self.model``."""
        model = self.model
        assert model is not None
        solver_name = self.solver_name.value
        cls = type(self)

        if model.is_quadratic and not cls.supports(SolverFeature.QUADRATIC_OBJECTIVE):
            raise ValueError(
                f"Solver {solver_name} does not support quadratic problems."
            )

        if model.variables.semi_continuous and not cls.supports(
            SolverFeature.SEMI_CONTINUOUS_VARIABLES
        ):
            raise ValueError(
                f"Solver {solver_name} does not support semi-continuous variables. "
                "Use a solver that supports them (gurobi, cplex, highs)."
            )

        if model.variables.sos and not cls.supports(SolverFeature.SOS_CONSTRAINTS):
            raise ValueError(
                f"Solver {solver_name} does not support SOS constraints. "
                "Reformulate first via `Model.solve(reformulate_sos=True)` or "
                "`model.apply_sos_reformulation()`, or use a solver that supports SOS."
            )

        if model.indicator_constraints and not cls.supports(
            SolverFeature.INDICATOR_CONSTRAINTS
        ):
            raise ValueError(
                f"Solver {solver_name} does not support indicator constraints. "
                "Use a solver that supports them."
            )

    def _build_direct(self, **build_kwargs: Any) -> None:
        """Build the native solver model from ``self.model``. Override per-solver."""
        raise NotImplementedError(
            f"Solver {self.solver_name.value} does not support direct API model export."
        )

    def _build_file(self, **build_kwargs: Any) -> None:
        """Write the LP/MPS file for ``self.model`` and cache its path."""
        model = self.model
        assert model is not None
        io_api = self.io_api
        if io_api is not None and io_api not in FILE_IO_APIS:
            raise ValueError(
                f"Keyword argument `io_api` has to be one of {IO_APIS} or None"
            )
        explicit_coordinate_names = build_kwargs.pop("explicit_coordinate_names", False)
        slice_size = build_kwargs.pop("slice_size", 2_000_000)
        progress = build_kwargs.pop("progress", None)
        problem_fn = build_kwargs.pop("problem_fn", None)
        if problem_fn is None:
            problem_fn = model.get_problem_file(io_api=io_api)
        if not self.supports(SolverFeature.LP_FILE_NAMES) and explicit_coordinate_names:
            logger.warning(
                f"{self.solver_name.value} does not support writing names to "
                "lp files, disabling it."
            )
            explicit_coordinate_names = False
        problem_fn = model.to_file(
            Path(problem_fn) if not isinstance(problem_fn, Path) else problem_fn,
            io_api=io_api,
            explicit_coordinate_names=explicit_coordinate_names,
            slice_size=slice_size,
            progress=progress,
        )
        self._problem_fn = problem_fn
        if self.io_api is None:
            self.io_api = read_io_api_from_problem_file(problem_fn)
        self._cache_model_sizes(model)

    def solve(
        self,
        model: Model | None = None,
        assign: bool = False,
        ignore_dims: Iterable[str] = (),
        disallow_rebuild: bool = False,
        **run_kwargs: Any,
    ) -> Result:
        """
        Run the prepared solver and return a :class:`Result`.

        With ``model`` supplied, diff against the previous build and either
        apply in place or rebuild before running. Requires ``io_api='direct'``.
        With ``assign=True`` the Result is written back to the target Model
        via :meth:`Model.assign_result`.

        Coordinate alignment is checked on every dim by default. Pass
        ``ignore_dims`` to exclude dims whose coord values legitimately shift
        between solves.

        Pass ``disallow_rebuild=True`` to guarantee that an existing solver
        model is updated in place — any condition that would force a rebuild
        (structural change, sparsity change, backend rejection, …) raises
        :class:`RebuildRequiredError` instead. The initial build on the first
        ``solve(model, ...)`` is still allowed.

        Thread safety: the solver lock is held for the entire call,
        including the native run. This is deliberate — diff/apply and the
        run must be atomic (otherwise a concurrent apply would change the
        problem between apply and run), and native solver handles are not
        thread-safe. Concurrent solves therefore serialize per Solver
        instance; use separate instances for parallelism. Pure diff
        computation (``update(model, apply=False)``) does not take the lock.
        """
        if model is not None and self.io_api != "direct":
            raise ValueError("solve(model=...) requires io_api='direct'")

        with self._lock:
            if model is not None:
                if self.solver_model is None:
                    self.model = model
                    self._build()
                else:
                    if not self.track_updates and model is self.model:
                        raise UpdatesDisabledError(
                            "Solver was constructed with track_updates=False; "
                            "in-place mutations of the build-time Model cannot "
                            "be detected without a snapshot. Pass a freshly "
                            "built Model instance, or reconstruct the solver "
                            "with Solver.from_name(..., track_updates=True)."
                        )
                    self._apply_locked(
                        model,
                        ignore_dims=ignore_dims,
                        disallow_rebuild=disallow_rebuild,
                    )
                target = model
            else:
                target = self.model  # type: ignore[assignment]

            if self.model is not None and self.model.objective.expression.empty:
                raise ValueError(
                    "No objective has been set on the model. Use `m.add_objective(...)` "
                    "first (e.g. `m.add_objective(0 * x)` for a pure feasibility problem)."
                )
            if self.io_api == "direct" or self.solver_model is not None:
                result = self._run_direct(**run_kwargs)
            elif self._problem_fn is not None:
                result = self._run_file(**run_kwargs)
            else:
                raise RuntimeError(
                    "Solver has not been built; call Solver.from_name(...) or _build() first."
                )

            if assign and target is not None:
                target.assign_result(result, solver=self)
        return result

    def update(
        self,
        model: Model,
        apply: bool = True,
        ignore_dims: Iterable[str] = (),
    ) -> ModelDiff | RebuildReason:
        """
        Diff ``model`` against the solver state and optionally apply it.

        With ``apply=False`` the diff is computed without taking the solver
        lock, so it can overlap a concurrently running solve. The preview
        always runs a full comparison (no ``_coef_dirty`` shortcut — a
        concurrent apply may clear the flag against a newer snapshot), so it
        can report raw in-place ``.values[...]`` mutations that the apply
        path, which trusts the flag for the build-time model, would miss.
        """
        if self.io_api != "direct":
            raise ValueError("update requires io_api='direct'")
        if self.solver_model is None:
            raise RuntimeError("Solver has not been built")
        if not self.track_updates and model is self.model:
            raise UpdatesDisabledError(
                "Solver was constructed with track_updates=False; "
                "in-place mutations of the build-time Model cannot be "
                "detected without a snapshot. Pass a freshly built Model "
                "instance, or reconstruct the solver with "
                "Solver.from_name(..., track_updates=True)."
            )
        if not apply:
            return self._compute_diff(model, ignore_dims, same_model=False)
        with self._lock:
            return self._apply_locked(model, ignore_dims=ignore_dims)

    def _compute_diff(
        self, model: Model, ignore_dims: Iterable[str], same_model: bool
    ) -> ModelDiff | RebuildReason:
        """
        Diff ``model`` against the solver baseline (the captured snapshot, or
        the build-time Model when no snapshot is tracked).

        ``same_model=True`` lets ``from_snapshot`` trust the ``_coef_dirty``
        flag and skip the coefficient re-walk; ``same_model=False`` forces a
        full comparison. Snapshot and baseline refs are read once, so the walk
        stays consistent even while a concurrent apply swaps ``self.snapshot``;
        the ``from_models`` fallback is only consistent if no thread
        concurrently mutates either Model.
        """
        snapshot = self.snapshot
        if snapshot is not None:
            return ModelDiff.from_snapshot(
                snapshot, model, same_model=same_model, ignore_dims=ignore_dims
            )
        baseline = self.model
        assert baseline is not None
        return ModelDiff.from_models(baseline, model, ignore_dims=ignore_dims)

    def _apply_locked(
        self,
        model: Model,
        ignore_dims: Iterable[str] = (),
        disallow_rebuild: bool = False,
    ) -> ModelDiff | RebuildReason:
        if not self.supports_persistent_update:
            if disallow_rebuild:
                raise RebuildRequiredError(RebuildReason.BACKEND_REJECTED)
            self._rebuild(model, RebuildReason.BACKEND_REJECTED)
            return RebuildReason.BACKEND_REJECTED
        diff = self._compute_diff(model, ignore_dims, same_model=model is self.model)
        if isinstance(diff, RebuildReason):
            if disallow_rebuild:
                raise RebuildRequiredError(diff)
            self._rebuild(model, diff)
            return diff
        try:
            self.apply_update(
                diff,
                model.variables.label_index,
                model.constraints.label_index,
            )
        except Exception as exc:
            if disallow_rebuild:
                raise RebuildRequiredError(
                    RebuildReason.BACKEND_REJECTED, str(exc)
                ) from exc
            self._last_rebuild_reason = RebuildReason.BACKEND_REJECTED
            self._rebuild(model, RebuildReason.BACKEND_REJECTED)
            return diff
        self.model = model
        if self.track_updates:
            self.snapshot = diff.snapshot
            clear_coef_dirty(model)
        self._in_place_updates += 1
        self._last_rebuild_reason = None
        return diff

    def _rebuild(self, model: Model, reason: RebuildReason) -> None:
        self.close()
        self.model = model
        self._build()
        self._rebuilds += 1
        self._last_rebuild_reason = reason

    def _run_direct(self, **run_kwargs: Any) -> Result:
        """Run the pre-built native solver model. Override per-solver."""
        raise NotImplementedError(
            f"Direct API not implemented for {self.solver_name.value}"
        )

    def _run_file(self, **run_kwargs: Any) -> Result:
        """Invoke the solver binary on ``self._problem_fn``. Override per-solver."""
        raise NotImplementedError(
            f"File-based API not implemented for {self.solver_name.value}"
        )

    def solve_problem(
        self,
        model: Model | None = None,
        problem_fn: Path | None = None,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: EnvType | None = None,
        explicit_coordinate_names: bool = False,
    ) -> Result:
        """Deprecated. Use ``Solver.from_name(...).solve(...)`` or ``Model.solve(...)``."""
        warnings.warn(
            "Solver.solve_problem is deprecated and will be removed in a future "
            "release. Use Solver.from_name(name, model, ...).solve(...) or "
            "Model.solve(...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if problem_fn is not None and model is not None:
            raise ValueError(
                "Both problem file and model are given. Please specify only one."
            )
        if model is not None:
            return self.solve_problem_from_model(
                model=model,
                solution_fn=solution_fn,
                log_fn=log_fn,
                warmstart_fn=warmstart_fn,
                basis_fn=basis_fn,
                env=env,
                explicit_coordinate_names=explicit_coordinate_names,
            )
        if problem_fn is not None:
            return self.solve_problem_from_file(
                problem_fn=problem_fn,
                solution_fn=solution_fn,
                log_fn=log_fn,
                warmstart_fn=warmstart_fn,
                basis_fn=basis_fn,
                env=env,
            )
        raise ValueError("No problem file or model specified.")

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: EnvType | None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        """Deprecated shim that builds via ``_build_direct`` and runs via ``_run_direct``."""
        warnings.warn(
            "Solver.solve_problem_from_model is deprecated and will be removed in a "
            "future release. Use Solver.from_name(name, model, io_api='direct', ...)"
            ".solve(...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.supports(SolverFeature.DIRECT_API):
            raise NotImplementedError(
                f"Direct API not implemented for {self.solver_name.value}"
            )
        self.model = model
        build_kwargs: dict[str, Any] = {
            "explicit_coordinate_names": explicit_coordinate_names,
            "set_names": set_names,
            "log_fn": log_fn,
        }
        if env is not None:
            build_kwargs["env"] = env
        self._build_direct(**build_kwargs)
        return self._run_direct(
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            env=env,
        )

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: EnvType | None = None,
    ) -> Result:
        """Deprecated shim that caches ``problem_fn`` and runs via ``_run_file``."""
        warnings.warn(
            "Solver.solve_problem_from_file is deprecated and will be removed in a "
            "future release. Use Solver.from_name(name, model, problem_fn=..., ...)"
            ".solve(...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        problem_fn = (
            Path(problem_fn) if not isinstance(problem_fn, Path) else problem_fn
        )
        self._problem_fn = problem_fn
        self.io_api = read_io_api_from_problem_file(problem_fn)
        return self._run_file(
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            env=env,
        )

    def _cache_model_labels(self, model: Model) -> None:
        """Cache vlabels/clabels and total label counts for label-indexed solutions."""
        self._vlabels = model.variables.label_index.vlabels
        self._clabels = model.constraints.label_index.clabels
        self._n_vars = model._xCounter
        self._n_cons = model._cCounter

    def _cache_model_sizes(self, model: Model) -> None:
        """Cache total label counts only (file-based solvers parse names)."""
        self._n_vars = model._xCounter
        self._n_cons = model._cCounter

    def update_solver_model(self, model: Model, **kwargs: Any) -> None:
        raise NotImplementedError

    def close(self) -> None:
        if self._env_stack is not None:
            self._env_stack.close()
        self.env = None
        self.solver_model = None
        self._env_stack = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def __getstate__(self) -> dict[str, Any]:
        drop = {"solver_model", "env", "_env_stack", "snapshot", "_lock"}
        return {k: v for k, v in self.__dict__.items() if k not in drop}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.solver_model = None
        self.env = None
        self._env_stack = None
        self.snapshot = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        status = self.status.status.value if self.status is not None else "unsolved"
        parts = [f"name={self.solver_name.value!r}", f"status={status!r}"]
        if self.io_api is not None:
            parts.append(f"io_api={self.io_api!r}")
        if self.solver_model is not None:
            parts.append("solver_model=loaded")
        if self.env is not None:
            parts.append("env=active")
        if self.solution is not None:
            parts.append(f"objective={self.solution.objective:.4g}")
        if self.report is not None and self.report.runtime is not None:
            parts.append(f"runtime={self.report.runtime:.3g}s")
        return f"{type(self).__name__}({', '.join(parts)})"

    def _make_result(
        self,
        status: Status,
        solution: Solution | None,
        solver_model: Any = None,
        report: SolverReport | None = None,
    ) -> Result:
        self.status = status
        self.solution = solution
        self.report = report
        if solver_model is not None:
            self.solver_model = solver_model
        return Result(
            status=status,
            solution=solution,
            solver_model=solver_model,
            solver_name=self.solver_name.value,
            report=report,
        )

    def safe_get_solution(
        self, status: Status, func: Callable[[], Solution]
    ) -> Solution:
        """
        Get solution from function call, if status is unknown still try to run it.
        """
        if status.is_ok:
            return func()
        elif status.status == SolverStatus.unknown:
            try:
                logger.warning("Solution status unknown. Trying to parse solution.")
                sol = func()
                status.status = SolverStatus.ok
                logger.warning("Solution parsed successfully.")
                return sol
            except Exception as e:
                logger.error(f"Failed to parse solution: {e}")
        return Solution()

    @property
    def solver_name(self) -> SolverName:
        return SolverName[self.__class__.__name__]


class CBC(Solver[None]):
    """
    Solver subclass for the CBC solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "CBC"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.READ_MODEL_FROM_FILE,
        }
    )

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return shutil.which("cbc") is not None

    def _run_file(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        **kw: Any,
    ) -> Result:
        problem_fn = self._problem_fn
        assert problem_fn is not None
        sense = read_sense_from_problem_file(problem_fn)
        io_api = read_io_api_from_problem_file(problem_fn)

        if solution_fn is None:
            msg = "No solution file specified. For solving with CBC this is necessary."
            raise ValueError(msg)

        if io_api == "mps" and sense == "max" and _new_highspy_mps_layout():
            msg = (
                "CBC does not support maximization in MPS format highspy versions "
                " >=1.7.1"
            )
            raise ValueError(msg)

        # printingOptions is about what goes in solution file
        command = f"cbc -printingOptions all -import {problem_fn} "

        if warmstart_fn:
            command += f"-basisI {warmstart_fn} "

        if self.solver_options:
            command += (
                " ".join(
                    "-" + " ".join([k, str(v)]) for k, v in self.solver_options.items()
                )
                + " "
            )
        command += f"-solve -solu {solution_fn} "

        if basis_fn:
            command += f"-basisO {basis_fn} "

        Path(solution_fn).parent.mkdir(exist_ok=True)

        command = command.strip()

        if log_fn is None:
            p = sub.Popen(command.split(" "), stdout=sub.PIPE, stderr=sub.PIPE)

            if p.stdout is None:
                msg = (
                    f"Command `{command}` did not run successfully. Check if cbc is "
                    " installed and in PATH."
                )
                raise ValueError(msg)

            output = ""
            for line in iter(p.stdout.readline, b""):
                output += line.decode()
            logger.info(output)
            p.stdout.close()
            p.wait()
        else:
            with open(log_fn, "w") as log_f:
                p = sub.Popen(command.split(" "), stdout=log_f, stderr=log_f)
                p.wait()

        with open(solution_fn) as f:
            first_line = f.readline()

        if first_line.startswith("Optimal "):
            status = Status.from_termination_condition("optimal")
        elif "Infeasible" in first_line:
            status = Status.from_termination_condition("infeasible")
        else:
            status = Status(SolverStatus.warning, TerminationCondition.unknown)
        status.legacy_status = first_line

        # Use HiGHS to parse the problem file and find the set of variable names, needed to parse solution
        if "highs" not in available_solvers:
            raise ModuleNotFoundError(
                f"highspy is not installed. Please install it to use {self.solver_name.name} solver."
            )
        h = highspy.Highs()
        h.silent()
        h.readModel(path_to_string(problem_fn))
        variables = {v.name for v in h.getVariables()}

        def get_solver_solution() -> Solution:
            m = re.match(r"Optimal.* - objective value (-?\d+\.?\d*)$", first_line)
            if m and len(m.groups()) == 1:
                objective = float(m.group(1))
            else:
                objective = np.nan

            with open(solution_fn, "rb") as f:
                trimmed_sol_fn = re.sub(rb"\*\*\s+", b"", f.read())

            df = pd.read_csv(
                io.BytesIO(trimmed_sol_fn),
                header=None,
                skiprows=[0],
                sep=r"\s+",
                usecols=[1, 2, 3],
                index_col=0,
            )
            variables_b = df.index.isin(variables)

            sol_df = df[variables_b]
            dual_df = df[~variables_b]
            sol = _solution_from_names(
                sol_df[2].to_numpy(dtype=float),
                sol_df.index.tolist(),
                self._n_vars,
            )
            dual = _solution_from_names(
                dual_df[3].to_numpy(dtype=float),
                dual_df.index.tolist(),
                self._n_cons,
            )
            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        # Parse the output and get duality gap and solver runtime
        mip_gap, runtime = None, None
        if log_fn is not None:
            with open(log_fn) as log_f:
                output = "".join(log_f.readlines())
        m = re.search(r"\nGap: +(\d+\.?\d*)\n", output)
        if m and len(m.groups()) == 1:
            mip_gap = float(m.group(1))
        m = re.search(r"\nTime \(Wallclock seconds\): +(\d+\.?\d*)\n", output)
        if m and len(m.groups()) == 1:
            runtime = float(m.group(1))
        CbcModel = namedtuple("CbcModel", ["mip_gap", "runtime"])

        self.io_api = io_api
        return self._make_result(
            status,
            solution,
            solver_model=CbcModel(mip_gap, runtime),
            report=SolverReport(runtime=runtime, mip_gap=mip_gap),
        )


class GLPK(Solver[None]):
    """
    Solver subclass for the GLPK solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "GLPK"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.READ_MODEL_FROM_FILE,
        }
    )

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return shutil.which("glpsol") is not None

    def _run_file(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        **kw: Any,
    ) -> Result:
        problem_fn = self._problem_fn
        assert problem_fn is not None
        CONDITION_MAP = {
            "integer optimal": "optimal",
            "integer undefined": "infeasible_or_unbounded",
            "undefined": "infeasible_or_unbounded",
        }
        sense = read_sense_from_problem_file(problem_fn)
        io_api = read_io_api_from_problem_file(problem_fn)
        if solution_fn is None:
            msg = "No solution file specified. For solving with GLPK this is necessary."
            raise ValueError(msg)

        if io_api == "mps" and sense == "max" and _new_highspy_mps_layout():
            msg = (
                "GLPK does not support maximization in MPS format highspy versions "
                " >=1.7.1"
            )
            raise ValueError(msg)

        Path(solution_fn).parent.mkdir(exist_ok=True)

        # TODO use --nopresol argument for non-optimal solution output
        io_api_arg = "freemps" if io_api == "mps" else io_api
        command = f"glpsol --{io_api_arg} {problem_fn} --output {solution_fn} "
        if log_fn is not None:
            command += f"--log {log_fn} "
        if warmstart_fn:
            command += f"--ini {warmstart_fn} "
        if basis_fn:
            command += f"-w {basis_fn} "
        if self.solver_options:
            command += (
                " ".join(
                    "--" + " ".join([k, str(v)]) for k, v in self.solver_options.items()
                )
                + " "
            )
        command = command.strip()

        p = sub.Popen(command.split(" "), stdout=sub.PIPE, stderr=sub.PIPE)
        if log_fn is None:
            output = ""

            if p.stdout is None:
                msg = (
                    f"Command `{command}` did not run successfully. Check if glpsol is "
                    "installed and in PATH."
                )
                raise ValueError(msg)

            for line in iter(p.stdout.readline, b""):
                output += line.decode()
            logger.info(output)
            p.stdout.close()
            p.wait()
        else:
            p.wait()

        if not os.path.exists(solution_fn):
            status = Status(SolverStatus.warning, TerminationCondition.unknown)
            self.io_api = io_api
            return self._make_result(status, Solution())

        f = open(solution_fn)

        def read_until_break(f: io.TextIOWrapper) -> Generator[str, None, None]:
            while True:
                line = f.readline()
                if line in ["\n", ""]:
                    break
                yield line

        info_io = io.StringIO("".join(read_until_break(f))[:-2])
        info = pd.read_csv(info_io, sep=":", index_col=0, header=None)[1]
        condition = info.Status.lower().strip()
        objective = float(re.sub(r"[^0-9\.\+\-e]+", "", info.Objective))

        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            dual_io = io.StringIO("".join(read_until_break(f))[:-2])
            dual_ = pd.read_fwf(dual_io)[1:].set_index("Row name")
            if "Marginal" in dual_:
                dual = _solution_from_names(
                    pd.to_numeric(dual_["Marginal"], "coerce")
                    .fillna(0)
                    .to_numpy(dtype=float),
                    dual_.index.tolist(),
                    self._n_cons,
                )
            else:
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = np.array([], dtype=float)

            sol_io = io.StringIO("".join(read_until_break(f))[:-2])
            sol_df = pd.read_fwf(sol_io)[1:].set_index("Column name")
            sol = _solution_from_names(
                sol_df["Activity"].astype(float).to_numpy(),
                sol_df.index.tolist(),
                self._n_vars,
            )
            f.close()
            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)
        self.io_api = io_api
        return self._make_result(status, solution)


class Highs(Solver[None]):
    """
    Solver subclass for the HiGHS solver. HiGHS must be installed
    for usage. Find the documentation at https://highs.dev/.

    The full list of solver options is documented at https://ergo-code.github.io/HiGHS/stable/options/definitions/.

    Some exemplary options are:

    * presolve : "choose" by default - "on"/"off" are alternatives.
    * solver :"choose" by default - "simplex"/"ipm"/"pdlp" are alternatives. Only "choose" solves MIP / QP!
    * parallel : "choose" by default - "on"/"off" are alternatives.
    * time_limit : inf by default.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "HiGHS"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.DIRECT_API,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            SolverFeature.SEMI_CONTINUOUS_VARIABLES,
            SolverFeature.MIP_DUAL_BOUND_REPORT,
        }
    )
    supports_persistent_update: ClassVar[bool] = True

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("highspy")

    @classmethod
    @functools.cache
    def _vtype_map(cls) -> dict[VarKind, Any]:
        return {
            VarKind.CONTINUOUS: highspy.HighsVarType.kContinuous,
            VarKind.BINARY: highspy.HighsVarType.kInteger,
            VarKind.INTEGER: highspy.HighsVarType.kInteger,
            VarKind.SEMI_CONTINUOUS: highspy.HighsVarType.kSemiContinuous,
        }

    def _apply_var_bounds(
        self, ctx: Any, indices: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> None:
        ctx.changeColsBounds(indices.size, indices, lower, upper)

    def _apply_var_types(
        self, ctx: Any, positions: np.ndarray, kinds: np.ndarray
    ) -> None:
        type_map = self._vtype_map()
        integrality = np.fromiter(
            (int(type_map[k]) for k in kinds),
            dtype=np.uint8,
            count=positions.size,
        )
        ctx.changeColsIntegrality(positions.size, positions, integrality)

    def _apply_con_rhs(self, ctx: Any, diff: ModelDiff) -> None:
        lower, upper = diff.con_rhs_as_bounds()
        for pos, lo, up in zip(diff.con_rhs_indices, lower, upper):
            ctx.changeRowBounds(int(pos), float(lo), float(up))

    def _apply_con_coefs(
        self, ctx: Any, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray
    ) -> None:
        for i in range(rows.size):
            ctx.changeCoeff(int(rows[i]), int(cols[i]), float(vals[i]))

    def _apply_obj_linear(
        self, ctx: Any, indices: np.ndarray, values: np.ndarray
    ) -> None:
        ctx.changeColsCost(indices.size, indices, values)

    def _apply_obj_sense(self, ctx: Any, sense: str) -> None:
        native = (
            highspy.ObjSense.kMaximize if sense == "max" else highspy.ObjSense.kMinimize
        )
        ctx.changeObjectiveSense(native)

    def _build_direct(
        self,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
        log_fn: Path | None = None,
        **kwargs: Any,
    ) -> None:
        model = self.model
        assert model is not None
        if self.solver_options.get("solver") in [
            "simplex",
            "ipm",
            "pdlp",
        ] and model.type in [
            "QP",
            "MILP",
        ]:
            logger.warning(
                "The HiGHS solver ignores quadratic terms / integrality if the solver is set to 'simplex', 'ipm' or 'pdlp'. "
                "Drop the solver option or use 'choose' to enable quadratic terms / integrality."
            )

        h = self._build_solver_model(
            model,
            explicit_coordinate_names=explicit_coordinate_names,
            set_names=set_names,
        )
        self._set_solver_params(h, log_fn)
        self.solver_model = h
        self.io_api = "direct"
        self.sense = model.sense
        self._cache_model_labels(model)

    @staticmethod
    def _build_solver_model(
        model: Model,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> highspy.Highs:
        """Build a highspy.Highs instance that mirrors the linopy `model`."""
        if model.variables.sos:
            raise NotImplementedError(
                "SOS constraints are not supported by the HiGHS direct API. "
                "Use io_api='lp' instead."
            )

        M = model.matrices
        h = highspy.Highs()
        h.addVars(len(M.vlabels), M.lb, M.ub)
        if (
            len(model.binaries)
            + len(model.integers)
            + len(list(model.variables.semi_continuous))
        ):
            vtypes = M.vtypes
            integrality_map = {"C": 0, "B": 1, "I": 1, "S": 2}
            int_mask = (vtypes == "B") | (vtypes == "I") | (vtypes == "S")
            labels = np.arange(len(vtypes))[int_mask]
            integrality = np.array(
                [integrality_map[v] for v in vtypes[int_mask]], dtype=np.uint8
            )
            h.changeColsIntegrality(len(labels), labels, integrality)

        c = M.c
        h.changeColsCost(len(c), np.arange(len(c), dtype=np.int32), c)

        A = M.A
        if A is not None:
            A = A.tocsr()
            num_cons = A.shape[0]
            lower = np.where(M.sense != "<", M.b, -np.inf)
            upper = np.where(M.sense != ">", M.b, np.inf)
            h.addRows(num_cons, lower, upper, A.nnz, A.indptr, A.indices, A.data)

        if set_names:
            print_variables, print_constraints = linopy.io.get_printers_scalar(
                model, explicit_coordinate_names=explicit_coordinate_names
            )
            lp = h.getLp()
            lp.col_names_ = print_variables(M.vlabels)
            if len(M.clabels):
                lp.row_names_ = print_constraints(M.clabels)
            h.passModel(lp)

        Q = M.Q
        if Q is not None:
            Q = triu(Q).tocsr()
            num_vars = Q.shape[0]
            h.passHessian(num_vars, Q.nnz, 1, Q.indptr, Q.indices, Q.data)

        if model.objective.sense == "max":
            h.changeObjectiveSense(highspy.ObjSense.kMaximize)

        return h

    def _run_direct(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: Any = None,
        **kw: Any,
    ) -> Result:
        return self._solve(
            self.solver_model,
            solution_fn=solution_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api=self.io_api,
            sense=self.sense,
        )

    def _run_file(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        **kw: Any,
    ) -> Result:
        problem_fn = self._problem_fn
        assert problem_fn is not None
        problem_fn_ = path_to_string(problem_fn)
        h = highspy.Highs()
        self._set_solver_params(h, log_fn)

        h.readModel(problem_fn_)
        self.solver_model = h
        self.io_api = read_io_api_from_problem_file(problem_fn)

        return self._solve(
            h,
            solution_fn,
            warmstart_fn,
            basis_fn,
            io_api=self.io_api,
            sense=read_sense_from_problem_file(problem_fn),
            from_file=True,
        )

    def _set_solver_params(
        self,
        highs_solver: highspy.Highs,
        log_fn: Path | None = None,
    ) -> None:
        if log_fn is not None:
            self.solver_options["log_file"] = path_to_string(log_fn)
            logger.info(f"Log file at {self.solver_options['log_file']}")

        for k, v in self.solver_options.items():
            highs_solver.setOptionValue(k, v)

    def _solve(
        self,
        h: highspy.Highs,
        solution_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        io_api: str | None = None,
        sense: str | None = None,
        from_file: bool = False,
    ) -> Result:
        """
        Solve a linear problem from a HiGHS object.


        Parameters
        ----------
        h : highspy.Highs
            HiGHS object.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        io_api: str
            io_api of the problem. For direct API from linopy model this is "direct".
        sense: str
            "min" or "max"
        from_file: bool
            ``True`` when ``h`` was populated via ``readModel`` — HiGHS may have
            reordered columns/rows, so values are re-permuted using parsed names.

        Returns
        -------
        Result
        """
        # https://ergo-code.github.io/HiGHS/dev/structures/enums/#HighsModelStatus
        CONDITION_MAP: dict[highspy.HighsModelStatus, TerminationCondition] = {
            highspy.HighsModelStatus.kNotset: TerminationCondition.unknown,
            highspy.HighsModelStatus.kLoadError: TerminationCondition.internal_solver_error,
            highspy.HighsModelStatus.kModelError: TerminationCondition.internal_solver_error,
            highspy.HighsModelStatus.kPresolveError: TerminationCondition.internal_solver_error,
            highspy.HighsModelStatus.kSolveError: TerminationCondition.internal_solver_error,
            highspy.HighsModelStatus.kPostsolveError: TerminationCondition.internal_solver_error,
            highspy.HighsModelStatus.kModelEmpty: TerminationCondition.unknown,
            highspy.HighsModelStatus.kMemoryLimit: TerminationCondition.resource_interrupt,
            highspy.HighsModelStatus.kOptimal: TerminationCondition.optimal,
            highspy.HighsModelStatus.kInfeasible: TerminationCondition.infeasible,
            highspy.HighsModelStatus.kUnboundedOrInfeasible: TerminationCondition.infeasible_or_unbounded,
            highspy.HighsModelStatus.kUnbounded: TerminationCondition.unbounded,
            highspy.HighsModelStatus.kObjectiveBound: TerminationCondition.terminated_by_limit,
            highspy.HighsModelStatus.kObjectiveTarget: TerminationCondition.terminated_by_limit,
            highspy.HighsModelStatus.kTimeLimit: TerminationCondition.time_limit,
            highspy.HighsModelStatus.kIterationLimit: TerminationCondition.iteration_limit,
            highspy.HighsModelStatus.kSolutionLimit: TerminationCondition.terminated_by_limit,
            highspy.HighsModelStatus.kInterrupt: TerminationCondition.user_interrupt,
            highspy.HighsModelStatus.kUnknown: TerminationCondition.unknown,
        }

        if warmstart_fn is not None and warmstart_fn.suffix == ".sol":
            h.readSolution(path_to_string(warmstart_fn), 0)
        elif warmstart_fn:
            h.readBasis(path_to_string(warmstart_fn))

        _run_highs_with_keyboard_interrupt(h)

        condition = h.getModelStatus()
        termination_condition = CONDITION_MAP.get(
            condition, TerminationCondition.unknown
        )
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = h.modelStatusToString(condition)

        if basis_fn:
            h.writeBasis(path_to_string(basis_fn))

        if solution_fn:
            h.writeSolution(path_to_string(solution_fn), 0)

        def get_solver_solution() -> Solution:
            objective = h.getObjectiveValue()
            solution = h.getSolution()
            sol = np.asarray(solution.col_value, dtype=float)
            dual = np.asarray(solution.row_dual, dtype=float)
            if from_file:
                lp = h.getLp()
                sol = _solution_from_names(sol, lp.col_names_, self._n_vars)
                dual = _solution_from_names(dual, lp.row_names_, self._n_cons)
            else:
                sol = _solution_from_labels(sol, self._vlabels, self._n_vars)
                dual = _solution_from_labels(dual, self._clabels, self._n_cons)
            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        runtime: float | None = None
        mip_gap: float | None = None
        dual_bound: float | None = None
        with contextlib.suppress(Exception):
            runtime = float(h.getRunTime())
        with contextlib.suppress(Exception):
            mip_gap = float(h.getInfo().mip_gap)
        with contextlib.suppress(Exception):
            dual_bound = float(h.getInfo().mip_dual_bound)

        self.io_api = io_api
        return self._make_result(
            status,
            solution,
            solver_model=h,
            report=SolverReport(
                runtime=runtime, mip_gap=mip_gap, dual_bound=dual_bound
            ),
        )


class _GurobiApplyCtx(NamedTuple):
    gm: Any
    gvars: list[Any]
    gcons: list[Any]


class Gurobi(Solver["gurobipy.Env | dict[str, Any] | None"]):
    """
    Solver subclass for the gurobi solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "Gurobi"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.DIRECT_API,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            SolverFeature.IIS_COMPUTATION,
            SolverFeature.SOS_CONSTRAINTS,
            SolverFeature.INDICATOR_CONSTRAINTS,
            SolverFeature.SEMI_CONTINUOUS_VARIABLES,
            SolverFeature.SOLVER_ATTRIBUTE_ACCESS,
            SolverFeature.MIP_DUAL_BOUND_REPORT,
        }
    )
    supports_persistent_update: ClassVar[bool] = True
    supports_sign_update: ClassVar[bool] = True

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("gurobipy")

    @classmethod
    def _license_probe(cls) -> None:
        with gurobipy.Env():
            pass

    def _resolve_env(self, env: gurobipy.Env | dict[str, Any] | None) -> gurobipy.Env:
        self.close()
        self._env_stack = contextlib.ExitStack()
        if env is None:
            resolved = self._env_stack.enter_context(gurobipy.Env())
        elif isinstance(env, dict):
            resolved = self._env_stack.enter_context(gurobipy.Env(params=env))
        else:
            resolved = env
        self.env = resolved
        return resolved

    def _build_direct(
        self,
        explicit_coordinate_names: bool = False,
        env: gurobipy.Env | dict[str, Any] | None = None,
        set_names: bool = True,
        **kwargs: Any,
    ) -> None:
        model = self.model
        assert model is not None
        env_ = self._resolve_env(env)
        m = self._build_solver_model(
            model,
            env=env_,
            explicit_coordinate_names=explicit_coordinate_names,
            set_names=set_names,
        )
        self.solver_model = m
        self.io_api = "direct"
        self.sense = model.sense
        self._cache_model_labels(model)

    @staticmethod
    def _build_solver_model(
        model: Model,
        env: gurobipy.Env | None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> gurobipy.Model:
        """Build a gurobipy.Model that mirrors the linopy `model`."""
        model.constraints.sanitize_missings()
        gm = gurobipy.Model(env=env)

        M = model.matrices

        kwargs: dict[str, Any] = {}
        if set_names:
            print_variables, print_constraints = linopy.io.get_printers_scalar(
                model, explicit_coordinate_names=explicit_coordinate_names
            )
            kwargs["name"] = print_variables(M.vlabels)
        if (
            len(model.binaries.labels)
            + len(model.integers.labels)
            + len(list(model.variables.semi_continuous))
        ):
            kwargs["vtype"] = M.vtypes
        x = gm.addMVar(M.vlabels.shape, M.lb, M.ub, **kwargs)

        if model.is_quadratic:
            assert M.Q is not None
            gm.setObjective(0.5 * x.T @ M.Q @ x + M.c @ x)
        else:
            gm.setObjective(M.c @ x)

        if model.objective.sense == "max":
            gm.ModelSense = -1

        if M.A is not None:
            c = gm.addMConstr(M.A, x, M.sense, M.b)
            if set_names:
                names = print_constraints(M.clabels)
                c.setAttr("ConstrName", names)

        for sos_type, positions, weights in _iter_sos_sets(model):
            gm.addSOS(sos_type, x[positions.tolist()].tolist(), weights.tolist())

        if M.indicator_A is not None:
            sense_map = {
                "<": gurobipy.GRB.LESS_EQUAL,
                ">": gurobipy.GRB.GREATER_EQUAL,
                "=": gurobipy.GRB.EQUAL,
            }
            x_list = x.tolist()
            A = M.indicator_A
            for i in range(A.shape[0]):
                lhs = gurobipy.LinExpr()
                start, end = A.indptr[i], A.indptr[i + 1]
                for col, coeff in zip(A.indices[start:end], A.data[start:end]):
                    lhs.add(x_list[int(col)], float(coeff))
                gm.addGenConstrIndicator(
                    x_list[int(M.indicator_binvar[i])],
                    bool(M.indicator_binval[i]),
                    lhs,
                    sense_map[str(M.indicator_sense[i])],
                    float(M.indicator_b[i]),
                )

        gm.update()
        return gm

    _GUROBI_VTYPE_MAP: ClassVar[dict[VarKind, str]] = {
        VarKind.CONTINUOUS: "C",
        VarKind.BINARY: "B",
        VarKind.INTEGER: "I",
        VarKind.SEMI_CONTINUOUS: "S",
    }
    _GUROBI_SIGN_MAP: ClassVar[dict[str, str]] = {
        short_LESS_EQUAL: "<",
        short_GREATER_EQUAL: ">",
        EQUAL: "=",
    }
    _GUROBI_SENSE_MAP: ClassVar[dict[str, int]] = {"min": 1, "max": -1}

    def _apply_begin(self, var_label_index: Any, con_label_index: Any) -> Any:
        gm = self.solver_model
        gurobi_vars = gm.getVars()
        gurobi_cons = gm.getConstrs()
        if len(gurobi_vars) != var_label_index.n_active_vars:
            raise UnsupportedUpdate("gurobi var count mismatch")
        if len(gurobi_cons) != con_label_index.n_active_cons:
            raise UnsupportedUpdate("gurobi con count mismatch")
        return _GurobiApplyCtx(gm, gurobi_vars, gurobi_cons)

    def _apply_end(self, ctx: Any) -> None:
        ctx.gm.update()

    def _apply_var_bounds(
        self, ctx: Any, indices: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> None:
        gm, gvars, _ = ctx
        subset = [gvars[int(i)] for i in indices]
        gm.setAttr("LB", subset, lower.tolist())
        gm.setAttr("UB", subset, upper.tolist())

    def _apply_var_types(
        self, ctx: Any, positions: np.ndarray, kinds: np.ndarray
    ) -> None:
        gm, gvars, _ = ctx
        subset = [gvars[int(p)] for p in positions]
        vtypes = [self._GUROBI_VTYPE_MAP[k] for k in kinds]
        gm.setAttr("VType", subset, vtypes)

    def _reclamp_binary_bounds(
        self, ctx: Any, positions: np.ndarray, kinds: np.ndarray
    ) -> None:
        # Gurobi's VType 'B' natively implies [0, 1]; no bound writes needed.
        return None

    def _apply_con_rhs(self, ctx: Any, diff: ModelDiff) -> None:
        gm, _, gcons = ctx
        subset = [gcons[int(r)] for r in diff.con_rhs_indices]
        gm.setAttr("RHS", subset, diff.con_rhs_values.tolist())

    def _apply_con_signs(
        self, ctx: Any, indices: np.ndarray, signs: np.ndarray
    ) -> None:
        gm, _, gcons = ctx
        senses = []
        for s in signs:
            s_str = str(s)
            if s_str not in self._GUROBI_SIGN_MAP:
                raise UnsupportedUpdate(f"unknown sign {s_str!r}")
            senses.append(self._GUROBI_SIGN_MAP[s_str])
        subset = [gcons[int(r)] for r in indices]
        gm.setAttr("Sense", subset, senses)

    def _apply_con_coefs(
        self, ctx: Any, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray
    ) -> None:
        gm, gvars, gcons = ctx
        for i in range(rows.size):
            gm.chgCoeff(gcons[int(rows[i])], gvars[int(cols[i])], float(vals[i]))

    def _apply_obj_linear(
        self, ctx: Any, indices: np.ndarray, values: np.ndarray
    ) -> None:
        gm, gvars, _ = ctx
        subset = [gvars[int(i)] for i in indices]
        gm.setAttr("Obj", subset, values.tolist())

    def _apply_obj_sense(self, ctx: Any, sense: str) -> None:
        if sense not in self._GUROBI_SENSE_MAP:
            raise UnsupportedUpdate(f"unknown obj sense {sense!r}")
        ctx.gm.ModelSense = self._GUROBI_SENSE_MAP[sense]

    def _run_direct(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: Any = None,
        **kw: Any,
    ) -> Result:
        return self._solve(
            self.solver_model,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api=self.io_api,
            sense=self.sense,
        )

    def _run_file(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: gurobipy.Env | dict[str, Any] | None = None,
        **kw: Any,
    ) -> Result:
        problem_fn = self._problem_fn
        assert problem_fn is not None
        sense = read_sense_from_problem_file(problem_fn)
        io_api = read_io_api_from_problem_file(problem_fn)
        problem_fn_ = path_to_string(problem_fn)

        env_ = self._resolve_env(env)
        m = gurobipy.read(problem_fn_, env=env_)
        self.solver_model = m
        self.io_api = io_api

        return self._solve(
            m,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api=io_api,
            sense=sense,
            from_file=True,
        )

    def _solve(
        self,
        m: gurobipy.Model,
        solution_fn: Path | None,
        log_fn: Path | None,
        warmstart_fn: Path | None,
        basis_fn: Path | None,
        io_api: str | None,
        sense: str | None,
        from_file: bool = False,
    ) -> Result:
        """
        Solve a linear problem from a Gurobi object.


        Parameters
        ----------
        m
            Gurobi object.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        io_api: str
            io_api of the problem. For direct API from linopy model this is "direct".
        sense: str
            "min" or "max"

        Returns
        -------
        Result
        """
        # see https://www.gurobi.com/documentation/10.0/refman/optimization_status_codes.html
        CONDITION_MAP = {
            1: "unknown",
            2: "optimal",
            3: "infeasible",
            4: "infeasible_or_unbounded",
            5: "unbounded",
            6: "other",
            7: "iteration_limit",
            8: "terminated_by_limit",
            9: "time_limit",
            10: "optimal",
            11: "user_interrupt",
            12: "other",
            13: "suboptimal",
            14: "unknown",
            15: "terminated_by_limit",
            16: "internal_solver_error",
            17: "internal_solver_error",
        }

        if self.solver_options is not None:
            for key, value in self.solver_options.items():
                m.setParam(key, value)
        if log_fn is not None:
            m.setParam("logfile", path_to_string(log_fn))

        if warmstart_fn is not None:
            m.read(path_to_string(warmstart_fn))
        m.optimize()

        if basis_fn is not None:
            try:
                m.write(path_to_string(basis_fn))
            except gurobipy.GurobiError as err:
                logger.info("No model basis stored. Raised error: %s", err)

        if solution_fn is not None and solution_fn.suffix == ".sol":
            try:
                m.write(path_to_string(solution_fn))
            except gurobipy.GurobiError as err:
                logger.info("Unable to save solution file. Raised error: %s", err)

        condition = m.status
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            objective = m.ObjVal

            vars_ = m.getVars()
            sol = np.array([v.X for v in vars_], dtype=float)
            if from_file:
                sol = _solution_from_names(
                    sol, [v.VarName for v in vars_], self._n_vars
                )
            else:
                sol = _solution_from_labels(sol, self._vlabels, self._n_vars)

            try:
                constrs = m.getConstrs()
                dual = np.array([c.Pi for c in constrs], dtype=float)
                if from_file:
                    dual = _solution_from_names(
                        dual,
                        [c.ConstrName for c in constrs],
                        self._n_cons,
                    )
                else:
                    dual = _solution_from_labels(dual, self._clabels, self._n_cons)
            except AttributeError:
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = np.array([], dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        runtime: float | None = None
        mip_gap: float | None = None
        dual_bound: float | None = None
        with contextlib.suppress(Exception):
            runtime = float(m.Runtime)
        with contextlib.suppress(Exception):
            mip_gap = float(m.MIPGap)
        with contextlib.suppress(Exception):
            dual_bound = float(m.ObjBound)

        self.io_api = io_api
        return self._make_result(
            status,
            solution,
            solver_model=m,
            report=SolverReport(
                runtime=runtime, mip_gap=mip_gap, dual_bound=dual_bound
            ),
        )


class Cplex(Solver[None]):
    """
    Solver subclass for the Cplex solver.

    Note if you pass additional solver_options, the key can specify deeper
    layered parameters, use a dot as a separator here,
    i.e. `**{'aa.bb.cc' : x}`.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "CPLEX"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOS_CONSTRAINTS,
            SolverFeature.INDICATOR_CONSTRAINTS,
            SolverFeature.SEMI_CONTINUOUS_VARIABLES,
        }
    )

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("cplex")

    def _run_file(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        **kw: Any,
    ) -> Result:
        problem_fn = self._problem_fn
        assert problem_fn is not None
        CONDITION_MAP = {
            "integer optimal solution": "optimal",
            "integer optimal, tolerance": "optimal",
            "integer infeasible": "infeasible",
            "time limit exceeded": "time_limit",
            "time limit exceeded, no integer solution": "infeasible",
            "error termination": "error",
            "error termination, no integer solution": "error",
            "memory limit exceeded": "internal_solver_error",
            "memory limit exceeded, no integer solution": "internal_solver_error",
            "aborted": "user_interrupt",
            "integer unbounded": "unbounded",
            "integer infeasible or unbounded": "infeasible_or_unbounded",
            "Unknown status value": "unknown",
        }
        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        m = cplex.Cplex()

        if log_fn is not None:
            log_f = open(path_to_string(log_fn), "w")
            m.set_results_stream(log_f)
            m.set_warning_stream(log_f)
            m.set_error_stream(log_f)
            m.set_log_stream(log_f)

        if self.solver_options is not None:
            for key, value in self.solver_options.items():
                param = m.parameters
                for key_layer in key.split("."):
                    param = getattr(param, key_layer)
                param.set(value)

        m.read(path_to_string(problem_fn))

        if warmstart_fn is not None:
            m.start.read_basis(path_to_string(warmstart_fn))

        is_lp = m.problem_type[m.get_problem_type()] == "LP"

        with contextlib.suppress(cplex.exceptions.errors.CplexSolverError):
            m.solve()

        if solution_fn is not None:
            try:
                m.solution.write(path_to_string(solution_fn))
            except cplex.exceptions.errors.CplexSolverError as err:
                logger.info("Unable to save solution file. Raised error: %s", err)

        condition = m.solution.get_status_string()
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        if log_fn is not None:
            log_f.close()

        def get_solver_solution() -> Solution:
            if basis_fn and is_lp:
                try:
                    m.solution.basis.write(path_to_string(basis_fn))
                except cplex.exceptions.errors.CplexSolverError:
                    logger.info("No model basis stored")

            objective = m.solution.get_objective_value()

            solution = _solution_from_names(
                np.asarray(m.solution.get_values(), dtype=float),
                m.variables.get_names(),
                self._n_vars,
            )

            try:
                dual = _solution_from_names(
                    np.asarray(m.solution.get_dual_values(), dtype=float),
                    m.linear_constraints.get_names(),
                    self._n_cons,
                )
            except Exception:
                logger.warning(
                    "Dual values not available (e.g. barrier solution without crossover)"
                )
                dual = np.array([], dtype=float)
            return Solution(solution, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        self.io_api = io_api
        return self._make_result(status, solution, solver_model=m)


class SCIP(Solver[None]):
    """
    Solver subclass for the SCIP solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "SCIP"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }
    )

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("pyscipopt")

    def _run_file(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        **kw: Any,
    ) -> Result:
        problem_fn = self._problem_fn
        assert problem_fn is not None
        CONDITION_MAP: dict[str, TerminationCondition] = {
            # https://github.com/scipopt/scip/blob/b2bac412222296ff2b7f2347bb77d5fc4e05a2a1/src/scip/type_stat.h#L40
            "inforunbd": TerminationCondition.infeasible_or_unbounded,
            "userinterrupt": TerminationCondition.user_interrupt,
            "terminate": TerminationCondition.user_interrupt,
            "nodelimit": TerminationCondition.terminated_by_limit,
            "totalnodelimit": TerminationCondition.terminated_by_limit,
            "stallnodelimit": TerminationCondition.terminated_by_limit,
            "timelimit": TerminationCondition.time_limit,
            "memlimit": TerminationCondition.terminated_by_limit,
            "gaplimit": TerminationCondition.optimal,
            "primallimit": TerminationCondition.terminated_by_limit,
            "duallimit": TerminationCondition.terminated_by_limit,
            "sollimit": TerminationCondition.terminated_by_limit,
            "bestsollimit": TerminationCondition.terminated_by_limit,
            "restartlimit": TerminationCondition.terminated_by_limit,
        }

        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        m = scip.Model()
        m.readProblem(path_to_string(problem_fn))

        if self.solver_options is not None:
            emphasis = self.solver_options.pop("setEmphasis", None)
            if emphasis is not None:
                m.setEmphasis(getattr(scip.SCIP_PARAMEMPHASIS, emphasis.upper()))

            heuristics = self.solver_options.pop("setHeuristics", None)
            if heuristics is not None:
                m.setEmphasis(getattr(scip.SCIP_PARAMSETTING, heuristics.upper()))

            presolve = self.solver_options.pop("setPresolve", None)
            if presolve is not None:
                m.setEmphasis(getattr(scip.SCIP_PARAMSETTING, presolve.upper()))

            m.setParams(self.solver_options)

        if log_fn is not None:
            m.setLogfile(path_to_string(log_fn))

        if warmstart_fn:
            logger.warning("Warmstart not implemented for SCIP")

        m.optimize()

        if basis_fn:
            logger.warning("Basis not implemented for SCIP")

        if solution_fn:
            try:
                m.writeSol(m.getBestSol(), filename=path_to_string(solution_fn))
            except FileNotFoundError as err:
                logger.warning("Unable to save solution file. Raised error: %s", err)

        condition = m.getStatus()
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            objective = m.getObjVal()
            vars_to_ignore = {"quadobjvar", "qmatrixvar", "quadobj", "qmatrix"}

            s = m.getSols()[0]
            kept_vars = [v for v in m.getVars() if v.name not in vars_to_ignore]
            sol = _solution_from_names(
                np.array([s[v] for v in kept_vars], dtype=float),
                [v.name for v in kept_vars],
                self._n_vars,
            )

            cons = m.getConss(False)
            if len(cons) != 0:
                kept_cons = [c for c in cons if c.name not in vars_to_ignore]
                dual = _solution_from_names(
                    np.array([m.getDualSolVal(c) for c in kept_cons], dtype=float),
                    [c.name for c in kept_cons],
                    self._n_cons,
                )
            else:
                logger.warning("Dual values not available (is this an MILP?)")
                dual = np.array([], dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        self.io_api = io_api
        return self._make_result(status, solution, solver_model=m)


class Xpress(Solver[None]):
    """
    Solver subclass for the xpress solver.

    For more information on solver options, see
    https://www.fico.com/fico-xpress-optimization/docs/latest/solver/GUID-ACD7E60C-7852-36B7-A78A-CED0EA291CDD.html

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "FICO Xpress"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.DIRECT_API,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            SolverFeature.IIS_COMPUTATION,
            SolverFeature.SOS_CONSTRAINTS,
        }
    )
    supports_persistent_update: ClassVar[bool] = True
    supports_sign_update: ClassVar[bool] = True

    _XPRESS_VTYPE_MAP: ClassVar[dict[VarKind, str]] = {
        VarKind.CONTINUOUS: "C",
        VarKind.BINARY: "B",
        VarKind.INTEGER: "I",
        VarKind.SEMI_CONTINUOUS: "S",
    }
    _XPRESS_ROWTYPE_MAP: ClassVar[dict[str, str]] = {
        short_LESS_EQUAL: "L",
        short_GREATER_EQUAL: "G",
        EQUAL: "E",
    }

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("xpress")

    def _apply_var_bounds(
        self, ctx: Any, indices: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> None:
        cols = np.concatenate([indices, indices]).astype(np.int64, copy=False)
        btypes = ["L"] * indices.size + ["U"] * indices.size
        lb = np.where(np.isneginf(lower), -xpress.infinity, lower)
        ub = np.where(np.isposinf(upper), xpress.infinity, upper)
        vals = np.concatenate([lb, ub]).astype(float, copy=False)
        ctx.chgbounds(cols.tolist(), btypes, vals.tolist())

    def _apply_var_types(
        self, ctx: Any, positions: np.ndarray, kinds: np.ndarray
    ) -> None:
        coltypes = [self._XPRESS_VTYPE_MAP[k] for k in kinds]
        ctx.chgcoltype(positions.tolist(), coltypes)

    def _apply_con_rhs(self, ctx: Any, diff: ModelDiff) -> None:
        ctx.chgrhs(_int_list(diff.con_rhs_indices), _float_list(diff.con_rhs_values))

    def _apply_con_signs(
        self, ctx: Any, indices: np.ndarray, signs: np.ndarray
    ) -> None:
        rowtypes = []
        for s in signs:
            s_str = str(s)
            if s_str not in self._XPRESS_ROWTYPE_MAP:
                raise UnsupportedUpdate(f"unknown sign {s_str!r}")
            rowtypes.append(self._XPRESS_ROWTYPE_MAP[s_str])
        ctx.chgrowtype(_int_list(indices), rowtypes)

    def _apply_con_coefs(
        self, ctx: Any, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray
    ) -> None:
        ctx.chgmcoef(_int_list(rows), _int_list(cols), _float_list(vals))

    def _apply_obj_linear(
        self, ctx: Any, indices: np.ndarray, values: np.ndarray
    ) -> None:
        ctx.chgobj(_int_list(indices), _float_list(values))

    def _apply_obj_sense(self, ctx: Any, sense: str) -> None:
        if sense == "max":
            ctx.chgobjsense(xpress.maximize)
        elif sense == "min":
            ctx.chgobjsense(xpress.minimize)
        else:
            raise UnsupportedUpdate(f"unknown obj sense {sense!r}")

    def _build_direct(
        self,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
        **kwargs: Any,
    ) -> None:
        model = self.model
        assert model is not None
        self.close()
        self._env_stack = contextlib.ExitStack()
        problem = self._build_solver_model(
            model,
            explicit_coordinate_names=explicit_coordinate_names,
            set_names=set_names,
        )
        self._env_stack.enter_context(problem)
        self.solver_model = problem
        self.io_api = "direct"
        self.sense = model.sense
        self._cache_model_labels(model)

    @staticmethod
    def _build_solver_model(
        model: Model,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> xpress.problem:
        """
        Build an ``xpress.problem`` that mirrors the linopy ``model`` via ``loadproblem``.

        ``loadproblem`` is Xpress' universal native-array entry point loading LP/QP/MIQP
        in a single call; see the parameter reference at
        https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/python/HTML/problem.loadproblem.html.
        SOS arguments are left ``None`` and sets are added afterwards via ``addSOS`` so
        multi-dim ``add_sos_constraints`` can be grouped natively.
        """
        model.constraints.sanitize_missings()
        problem = xpress.problem()

        M = model.matrices
        A = M.A
        Q = M.Q

        if A is not None and A.nnz:
            if A.format != "csc":
                A = A.tocsc()
            start = A.indptr.astype(np.int64, copy=False)
            rowind = A.indices.astype(np.int64, copy=False)
            rowcoef = A.data.astype(float, copy=False)
        else:
            start = np.zeros(len(M.vlabels) + 1, dtype=np.int64)
            rowind = np.empty(0, dtype=np.int64)
            rowcoef = np.empty(0, dtype=float)

        lb = np.asarray(M.lb, dtype=float)
        ub = np.asarray(M.ub, dtype=float)
        np.place(lb, np.isneginf(lb), -xpress.infinity)
        np.place(ub, np.isposinf(ub), xpress.infinity)

        rowtype: np.ndarray
        rhs: np.ndarray
        if len(M.clabels):
            sense = M.sense
            rowtype = np.full(sense.shape, "E", dtype="U1")
            rowtype[sense == "<"] = "L"
            rowtype[sense == ">"] = "G"
            rhs = np.asarray(M.b, dtype=float)
        else:
            rowtype = np.empty(0, dtype="U1")
            rhs = np.empty(0, dtype=float)

        objqcol1: np.ndarray | None
        objqcol2: np.ndarray | None
        objqcoef: np.ndarray | None
        if Q is not None and Q.nnz:
            Qt = Q if Q.format == "coo" else triu(Q, format="coo")  # codespell:ignore
            mask = Qt.row <= Qt.col
            objqcol1 = Qt.row[mask].astype(np.int64, copy=False)
            objqcol2 = Qt.col[mask].astype(np.int64, copy=False)
            objqcoef = Qt.data[mask].astype(float, copy=False)
        else:
            objqcol1 = None
            objqcol2 = None
            objqcoef = None

        vtypes = M.vtypes
        integer_mask = (vtypes == "B") | (vtypes == "I")
        if integer_mask.any():
            entind = np.flatnonzero(integer_mask).astype(np.int64, copy=False)
            coltype = vtypes[entind]
        else:
            entind = None
            coltype = None

        objcoef = np.asarray(M.c, dtype=float)
        has_q = objqcol1 is not None
        has_int = coltype is not None
        base_kwargs: dict[str, Any] = dict(
            probname="linopy",
            rowtype=rowtype,
            rhs=rhs,
            rng=None,
            objcoef=objcoef,
            start=start,
            collen=None,
            rowind=rowind,
            rowcoef=rowcoef,
            lb=lb,
            ub=ub,
        )
        try:  # Try new API first (Xpress 9.8+)
            if has_q and has_int:
                problem.loadMIQP(
                    **base_kwargs,
                    objqcol1=objqcol1,
                    objqcol2=objqcol2,
                    objqcoef=objqcoef,
                    coltype=coltype,
                    entind=entind,
                )
            elif has_q:
                problem.loadQP(
                    **base_kwargs,
                    objqcol1=objqcol1,
                    objqcol2=objqcol2,
                    objqcoef=objqcoef,
                )
            elif has_int:
                problem.loadMIP(
                    **base_kwargs,
                    coltype=coltype,
                    entind=entind,
                )
            else:
                problem.loadLP(**base_kwargs)
        except AttributeError:  # Fallback to old API
            problem.loadproblem(
                probname="linopy",
                rowtype=rowtype,
                rhs=rhs,
                rng=None,
                objcoef=objcoef,
                start=start,
                collen=None,
                rowind=rowind,
                rowcoef=rowcoef,
                lb=lb,
                ub=ub,
                objqcol1=objqcol1,
                objqcol2=objqcol2,
                objqcoef=objqcoef,
                qrowind=None,
                nrowqcoefs=None,
                rowqcol1=None,
                rowqcol2=None,
                rowqcoef=None,
                coltype=coltype,
                entind=entind,
                limit=None,
                settype=None,
                setstart=None,
                setind=None,
                refval=None,
            )

        if model.objective.sense == "max":
            problem.chgobjsense(xpress.maximize)

        if set_names:
            print_variable, print_constraint = linopy.io.get_printers_scalar(
                model, explicit_coordinate_names=explicit_coordinate_names
            )
            vnames = print_variable(M.vlabels)
            if vnames:
                try:  # Try new API first (Xpress 9.8+)
                    problem.addNames(
                        xpress_Namespaces.COLUMN, vnames, 0, len(vnames) - 1
                    )
                except AttributeError:  # Fallback to old API
                    problem.addnames(
                        xpress_Namespaces.COLUMN, vnames, 0, len(vnames) - 1
                    )
            cnames = print_constraint(M.clabels)
            if cnames:
                try:  # Try new API first (Xpress 9.8+)
                    problem.addNames(xpress_Namespaces.ROW, cnames, 0, len(cnames) - 1)
                except AttributeError:  # Fallback to old API
                    problem.addnames(xpress_Namespaces.ROW, cnames, 0, len(cnames) - 1)

        for sos_type, positions, weights in _iter_sos_sets(model):
            problem.addSOS(positions.tolist(), weights.tolist(), type=sos_type)

        return problem

    @classmethod
    def runtime_features(cls) -> frozenset[SolverFeature]:
        if _installed_version_in("xpress", ">=9.8.0"):
            return frozenset({SolverFeature.GPU_ACCELERATION})
        return frozenset()

    def _run_direct(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        **kw: Any,
    ) -> Result:
        return self._solve(
            self.solver_model,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api=self.io_api,
            sense=self.sense,
        )

    def _run_file(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        **kw: Any,
    ) -> Result:
        problem_fn = self._problem_fn
        assert problem_fn is not None
        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        self.close()
        self._env_stack = contextlib.ExitStack()
        m = self._env_stack.enter_context(xpress.problem())
        try:  # Try new API first
            m.readProb(path_to_string(problem_fn))
        except AttributeError:  # Fallback to old API
            m.read(path_to_string(problem_fn))

        return self._solve(
            m,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api=io_api,
            sense=sense,
            from_file=True,
        )

    def _solve(
        self,
        m: xpress.problem,
        solution_fn: Path | None,
        log_fn: Path | None,
        warmstart_fn: Path | None,
        basis_fn: Path | None,
        io_api: str | None,
        sense: str | None,
        from_file: bool = False,
    ) -> Result:
        CONDITION_MAP = {
            xpress.SolStatus.NOTFOUND: "unknown",
            xpress.SolStatus.OPTIMAL: "optimal",
            xpress.SolStatus.FEASIBLE: "terminated_by_limit",
            xpress.SolStatus.INFEASIBLE: "infeasible",
            xpress.SolStatus.UNBOUNDED: "unbounded",
        }

        if self.solver_options is not None:
            m.setControl(self.solver_options)

        if log_fn is not None:
            try:  # Try new API first
                m.setLogFile(path_to_string(log_fn))
            except AttributeError:  # Fallback to old API
                m.setlogfile(path_to_string(log_fn))

        if warmstart_fn is not None:
            try:  # Try new API first
                m.readBasis(path_to_string(warmstart_fn))
            except AttributeError:  # Fallback to old API
                m.readbasis(path_to_string(warmstart_fn))

        m.optimize()

        if m.attributes.solvestatus == xpress.enums.SolveStatus.STOPPED:
            try:  # Try new API first
                m.postSolve()
            except AttributeError:  # Fallback to old API
                m.postsolve()

        if basis_fn is not None:
            try:
                try:  # Try new API first
                    m.writeBasis(path_to_string(basis_fn))
                except AttributeError:  # Fallback to old API
                    m.writebasis(path_to_string(basis_fn))
            except (xpress.SolverError, xpress.ModelError) as err:  # pragma: no cover
                logger.info("No model basis stored. Raised error: %s", err)

        if solution_fn is not None:
            try:
                try:  # Try new API first
                    m.writeBinSol(path_to_string(solution_fn))
                except AttributeError:  # Fallback to old API
                    m.writebinsol(path_to_string(solution_fn))
            except (xpress.SolverError, xpress.ModelError) as err:  # pragma: no cover
                logger.info("Unable to save solution file. Raised error: %s", err)

        condition = m.attributes.solstatus
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            objective = m.attributes.objval

            sol_values = np.asarray(m.getSolution(), dtype=float)
            if from_file:
                sol = _solution_from_names(
                    sol_values,
                    [v.name for v in m.getVariable()],
                    self._n_vars,
                )
            else:
                sol = _solution_from_labels(sol_values, self._vlabels, self._n_vars)

            try:
                if m.attributes.rows == 0:
                    dual = np.array([], dtype=float)
                else:
                    try:  # getDuals introduced in 9.5; fallback for 9.4
                        dual_values = np.asarray(m.getDuals(), dtype=float)
                    except AttributeError:
                        dual_values = np.asarray(m.getDual(), dtype=float)
                    if from_file:
                        dual = _solution_from_names(
                            dual_values,
                            [c.name for c in m.getConstraint()],
                            self._n_cons,
                        )
                    else:
                        dual = _solution_from_labels(
                            dual_values, self._clabels, self._n_cons
                        )
            except (
                xpress.SolverError,
                xpress.ModelError,
                SystemError,
            ):  # pragma: no cover
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = np.array([], dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        self.io_api = io_api
        return self._make_result(status, solution, solver_model=m)


KnitroResult = namedtuple(
    "KnitroResult",
    "reported_runtime mip_relaxation_bnd mip_number_nodes mip_number_solves mip_rel_gap mip_abs_gap abs_feas_error rel_feas_error abs_opt_error rel_opt_error n_vars n_cons n_integer_vars n_continuous_vars",
)


class Knitro(Solver[None]):
    """
    Solver subclass for the Knitro solver.

    For more information on solver options, see
    https://www.artelys.com/app/docs/knitro/3_referenceManual/knitroPythonReference.html

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "Artelys Knitro"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            SolverFeature.MIP_DUAL_BOUND_REPORT,
        }
    )

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("knitro")

    @classmethod
    def _license_probe(cls) -> None:
        kc = knitro.KN_new()
        knitro.KN_free(kc)

    @staticmethod
    def _set_option(kc: Any, name: str, value: Any) -> None:
        param_id = knitro.KN_get_param_id(kc, name)

        if isinstance(value, bool):
            value = int(value)

        if isinstance(value, int):
            knitro.KN_set_int_param(kc, param_id, value)
        elif isinstance(value, float):
            knitro.KN_set_double_param(kc, param_id, value)
        elif isinstance(value, str):
            knitro.KN_set_char_param(kc, param_id, value)
        else:
            msg = f"Unsupported Knitro option type for {name!r}: {type(value).__name__}"
            raise TypeError(msg)

    @staticmethod
    def _extract_values(
        kc: Any,
        get_count_fn: Callable[..., Any],
        get_values_fn: Callable[..., Any],
    ) -> np.ndarray:
        n = int(get_count_fn(kc))
        if n == 0:
            return np.array([], dtype=float)

        try:
            # Compatible with KNITRO >= 15
            values = get_values_fn(kc)
        except TypeError:
            # Fallback for older wrappers requiring explicit indices
            values = get_values_fn(kc, list(range(n)))

        return np.asarray(values, dtype=float)

    def _run_file(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        **kw: Any,
    ) -> Result:
        problem_fn = self._problem_fn
        assert problem_fn is not None
        CONDITION_MAP: dict[int, TerminationCondition] = {
            0: TerminationCondition.optimal,
            -100: TerminationCondition.suboptimal,
            -101: TerminationCondition.infeasible,
            -102: TerminationCondition.suboptimal,
            -200: TerminationCondition.unbounded,
            -201: TerminationCondition.infeasible_or_unbounded,
            -202: TerminationCondition.iteration_limit,
            -203: TerminationCondition.time_limit,
            -204: TerminationCondition.terminated_by_limit,
            -300: TerminationCondition.unbounded,
            -400: TerminationCondition.iteration_limit,
            -401: TerminationCondition.time_limit,
            -410: TerminationCondition.terminated_by_limit,
            -411: TerminationCondition.terminated_by_limit,
        }

        READ_OPTIONS: dict[str, str] = {".lp": "l", ".mps": "m"}

        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        suffix = problem_fn.suffix.lower()
        if suffix not in READ_OPTIONS:
            msg = f"Unsupported problem file format: {suffix}"
            raise ValueError(msg)

        kc = knitro.KN_new()
        try:
            knitro.KN_read_problem(
                kc,
                path_to_string(problem_fn),
                read_options=READ_OPTIONS[suffix],
            )

            if log_fn is not None:
                logger.warning("Log file output not implemented for Knitro")

            for k, v in self.solver_options.items():
                self._set_option(kc, k, v)

            ret = int(knitro.KN_solve(kc))

            reported_runtime: float | None = None
            mip_relaxation_bnd: float | None = None
            mip_number_nodes: int | None = None
            mip_number_solves: int | None = None
            mip_rel_gap: float | None = None
            mip_abs_gap: float | None = None
            abs_feas_error: float | None = None
            rel_feas_error: float | None = None
            abs_opt_error: float | None = None
            rel_opt_error: float | None = None
            n_vars: int | None = None
            n_cons: int | None = None
            n_integer_vars: int | None = None
            n_continuous_vars: int | None = None
            with contextlib.suppress(Exception):
                reported_runtime = float(knitro.KN_get_solve_time_real(kc))
                mip_relaxation_bnd = float(knitro.KN_get_mip_relaxation_bnd(kc))
                mip_number_nodes = int(knitro.KN_get_mip_number_nodes(kc))
                mip_number_solves = int(knitro.KN_get_mip_number_solves(kc))
                mip_rel_gap = float(knitro.KN_get_mip_rel_gap(kc))
                mip_abs_gap = float(knitro.KN_get_mip_abs_gap(kc))
                abs_feas_error = float(knitro.KN_get_abs_feas_error(kc))
                rel_feas_error = float(knitro.KN_get_rel_feas_error(kc))
                abs_opt_error = float(knitro.KN_get_abs_opt_error(kc))
                rel_opt_error = float(knitro.KN_get_rel_opt_error(kc))
                n_vars = int(knitro.KN_get_number_vars(kc))
                n_cons = int(knitro.KN_get_number_cons(kc))
                var_types = list(knitro.KN_get_var_types(kc))
                n_integer_vars = int(
                    var_types.count(knitro.KN_VARTYPE_INTEGER)
                    + var_types.count(knitro.KN_VARTYPE_BINARY)
                )
                n_continuous_vars = int(var_types.count(knitro.KN_VARTYPE_CONTINUOUS))

            if ret in CONDITION_MAP:
                termination_condition = CONDITION_MAP[ret]
            elif ret > 0:
                termination_condition = TerminationCondition.internal_solver_error
            else:
                termination_condition = TerminationCondition.unknown

            status = Status.from_termination_condition(termination_condition)
            status.legacy_status = str(ret)

            def get_solver_solution() -> Solution:
                objective = float(knitro.KN_get_obj_value(kc))

                sol = self._extract_values(
                    kc,
                    knitro.KN_get_number_vars,
                    knitro.KN_get_var_primal_values,
                )
                n_vars = int(knitro.KN_get_number_vars(kc))
                var_names = [knitro.KN_get_var_names(kc, i) for i in range(n_vars)]
                sol = _solution_from_names(sol, var_names, self._n_vars)

                try:
                    dual = self._extract_values(
                        kc,
                        knitro.KN_get_number_cons,
                        knitro.KN_get_con_dual_values,
                    )
                    n_cons = int(knitro.KN_get_number_cons(kc))
                    con_names = [knitro.KN_get_con_names(kc, i) for i in range(n_cons)]
                    dual = _solution_from_names(dual, con_names, self._n_cons)
                except Exception:
                    logger.warning("Dual values couldn't be parsed")
                    dual = np.array([], dtype=float)

                return Solution(sol, dual, objective)

            solution = self.safe_get_solution(status=status, func=get_solver_solution)
            solution = maybe_adjust_objective_sign(solution, io_api, sense)

            if solution_fn is not None:
                solution_fn.parent.mkdir(exist_ok=True)
                knitro.KN_write_mps_file(kc, path_to_string(solution_fn))

            knitro_model = KnitroResult(
                reported_runtime=reported_runtime,
                mip_relaxation_bnd=mip_relaxation_bnd,
                mip_number_nodes=mip_number_nodes,
                mip_number_solves=mip_number_solves,
                mip_rel_gap=mip_rel_gap,
                mip_abs_gap=mip_abs_gap,
                abs_feas_error=abs_feas_error,
                rel_feas_error=rel_feas_error,
                abs_opt_error=abs_opt_error,
                rel_opt_error=rel_opt_error,
                n_vars=n_vars,
                n_cons=n_cons,
                n_integer_vars=n_integer_vars,
                n_continuous_vars=n_continuous_vars,
            )
            self.io_api = io_api
            return self._make_result(
                status,
                solution,
                solver_model=knitro_model,
                report=SolverReport(runtime=reported_runtime, mip_gap=mip_rel_gap),
            )
        finally:
            with contextlib.suppress(Exception):
                knitro.KN_free(kc)


mosek_bas_re = re.compile(r" (XL|XU)\s+([^ \t]+)\s+([^ \t]+)| (LL|UL|BS)\s+([^ \t]+)")


class Mosek(Solver[None]):
    """
    Solver subclass for the Mosek solver.

    https://www.mosek.com/

    For more information on solver options, see
    https://docs.mosek.com/latest/pythonapi/parameters.html#doc-all-parameter-list


    For remote optimization of smaller problems, which do not require a license,
    set the following solver_options:
    {"MSK_SPAR_REMOTE_OPTSERVER_HOST": "http://solve.mosek.com:30080"}

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "MOSEK"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.DIRECT_API,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }
    )
    supports_persistent_update: ClassVar[bool] = True

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("mosek")

    @classmethod
    def _license_probe(cls) -> None:
        with mosek.Env() as env, env.Task(0, 0) as task:
            task.optimize()

    def _validate_update(self, diff: ModelDiff) -> None:
        super()._validate_update(diff)
        if (diff.var_type_kinds == VarKind.SEMI_CONTINUOUS).any():
            raise UnsupportedUpdate("MOSEK does not support semi-continuous variables")

    def _apply_var_bounds(
        self, ctx: Any, indices: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> None:
        for k in range(indices.size):
            j = int(indices[k])
            lb = float(lower[k])
            ub = float(upper[k])
            ctx.chgvarbound(j, 1, int(np.isfinite(lb)), lb)
            ctx.chgvarbound(j, 0, int(np.isfinite(ub)), ub)

    def _apply_var_types(
        self, ctx: Any, positions: np.ndarray, kinds: np.ndarray
    ) -> None:
        integer_mask = (kinds == VarKind.BINARY) | (kinds == VarKind.INTEGER)
        vartypes = np.where(
            integer_mask,
            mosek.variabletype.type_int,
            mosek.variabletype.type_cont,
        ).tolist()
        ctx.putvartypelist(_int_list(positions, np.int32), vartypes)

    def _apply_con_rhs(self, ctx: Any, diff: ModelDiff) -> None:
        lower, upper = diff.con_rhs_as_bounds()
        for k, i in enumerate(diff.con_rhs_indices):
            lo = float(lower[k])
            up = float(upper[k])
            ctx.chgconbound(int(i), 1, int(np.isfinite(lo)), lo)
            ctx.chgconbound(int(i), 0, int(np.isfinite(up)), up)

    def _apply_con_coefs(
        self, ctx: Any, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray
    ) -> None:
        ctx.putaijlist(
            _int_list(rows, np.int32), _int_list(cols, np.int32), _float_list(vals)
        )

    def _apply_obj_linear(
        self, ctx: Any, indices: np.ndarray, values: np.ndarray
    ) -> None:
        ctx.putclist(_int_list(indices, np.int32), _float_list(values))

    def _apply_obj_sense(self, ctx: Any, sense: str) -> None:
        if sense == "max":
            ctx.putobjsense(mosek.objsense.maximize)
        elif sense == "min":
            ctx.putobjsense(mosek.objsense.minimize)
        else:
            raise UnsupportedUpdate(f"unknown obj sense {sense!r}")

    def _run_direct(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: Any = None,
        **kw: Any,
    ) -> Result:
        return self._solve(
            self.solver_model,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api=self.io_api,
            sense=self.sense,
        )

    def _build_direct(
        self,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
        **kwargs: Any,
    ) -> None:
        model = self.model
        assert model is not None
        self.close()
        self._env_stack = contextlib.ExitStack()
        env = self._env_stack.enter_context(mosek.Env())
        task = self._env_stack.enter_context(env.Task(0, 0))
        m = self._build_solver_model(
            model,
            task,
            explicit_coordinate_names=explicit_coordinate_names,
            set_names=set_names,
        )
        self.solver_model = m
        self.io_api = "direct"
        self.sense = model.sense
        self._cache_model_labels(model)

    @staticmethod
    def _build_solver_model(
        model: Model,
        task: mosek.Task,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> mosek.Task:
        """Populate an empty MOSEK task with the contents of `model`."""
        if model.variables.sos:
            raise NotImplementedError("SOS constraints are not supported by MOSEK.")
        if model.variables.semi_continuous:
            raise NotImplementedError(
                "Semi-continuous variables are not supported by MOSEK. "
                "Use a solver that supports them (gurobi, cplex, highs)."
            )

        task.appendvars(model.nvars)
        task.appendcons(model.ncons)

        M = model.matrices

        if set_names:
            print_variables, print_constraints = linopy.io.get_printers_scalar(
                model, explicit_coordinate_names=explicit_coordinate_names
            )
            labels = print_variables(M.vlabels)
            task.generatevarnames(
                np.arange(0, len(labels)), "%0", [len(labels)], None, [0], labels
            )

        bkx = [
            (
                (
                    (mosek.boundkey.ra if lb < ub else mosek.boundkey.fx)
                    if ub < np.inf
                    else mosek.boundkey.lo
                )
                if (lb > -np.inf)
                else (mosek.boundkey.up if (ub < np.inf) else mosek.boundkey.fr)
            )
            for (lb, ub) in zip(M.lb, M.ub)
        ]
        blx = [b if b > -np.inf else 0.0 for b in M.lb]
        bux = [b if b < np.inf else 0.0 for b in M.ub]
        task.putvarboundslice(0, model.nvars, bkx, blx, bux)

        if len(model.binaries.labels) + len(model.integers.labels) > 0:
            idx = [i for (i, v) in enumerate(M.vtypes) if v in ["B", "I"]]
            task.putvartypelist(idx, [mosek.variabletype.type_int] * len(idx))

        if len(model.constraints) > 0:
            if set_names:
                names = print_constraints(M.clabels)
                for i, n in enumerate(names):
                    task.putconname(i, n)
            bkc = [
                (
                    (mosek.boundkey.up if b < np.inf else mosek.boundkey.fr)
                    if s == "<"
                    else (
                        (mosek.boundkey.lo if b > -np.inf else mosek.boundkey.up)
                        if s == ">"
                        else mosek.boundkey.fx
                    )
                )
                for s, b in zip(M.sense, M.b)
            ]
            blc = [b if b > -np.inf else 0.0 for b in M.b]
            buc = [b if b < np.inf else 0.0 for b in M.b]
            if M.A is not None:
                A = M.A.tocsr()
                task.putarowslice(
                    0, model.ncons, A.indptr[:-1], A.indptr[1:], A.indices, A.data
                )
                task.putconboundslice(0, model.ncons, bkc, blc, buc)

        if M.Q is not None:
            Q = (0.5 * tril(M.Q + M.Q.transpose())).tocoo()
            task.putqobj(Q.row, Q.col, Q.data)
        task.putclist(list(np.arange(model.nvars)), M.c)

        if model.objective.sense == "max":
            task.putobjsense(mosek.objsense.maximize)
        else:
            task.putobjsense(mosek.objsense.minimize)
        return task

    @staticmethod
    def _choose_solution(task: mosek.Task) -> mosek.soltype | None:
        """
        Pick the Mosek solution with the best status available.

        Mosek may return up to three solutions per task: interior-point
        (``soltype.itr``), basic (``soltype.bas``), and integer
        (``soltype.itg``). Each carries its own ``solsta``: on a numerically
        marginal LP solved with the default IPM+crossover, the interior-point
        solver may terminate with ``solsta.dual_infeas_cer`` while crossover
        recovers ``solsta.optimal`` for the basic solution. Reading only the
        interior-point solution would discard the actual optimum.

        Ranking, best to worst: ``solsta.optimal`` / ``solsta.integer_optimal``
        > any other defined status > undefined. On a tie between ``bas`` and
        ``itr`` (e.g. both ``optimal``) we prefer ``itr`` to preserve historical
        behaviour. If ``itg`` is defined it always wins, since integer and
        continuous solutions do not coexist for a well-posed task.

        Returns ``None`` if no solution is defined at all (e.g. the optimizer
        crashed before producing one).
        """

        def _is_defined(soltype: mosek.soltype) -> bool:
            try:
                return bool(task.solutiondef(soltype))
            except mosek.Error:
                return False

        if _is_defined(mosek.soltype.itg):
            return mosek.soltype.itg

        optimal_statuses = {mosek.solsta.optimal, mosek.solsta.integer_optimal}

        best: mosek.soltype | None = None
        best_score = -1
        # Iterate bas first and only then itr so that on a score tie
        # itr wins, preserving the historical default for the common LP case.
        for candidate in [mosek.soltype.bas, mosek.soltype.itr]:
            if not _is_defined(candidate):
                continue
            try:
                solsta = task.getsolsta(candidate)
            except mosek.Error:
                continue
            score = 1 if solsta in optimal_statuses else 0
            if score >= best_score:
                best = candidate
                best_score = score

        return best

    def _run_file(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        **kw: Any,
    ) -> Result:
        problem_fn = self._problem_fn
        assert problem_fn is not None
        self.close()
        self._env_stack = contextlib.ExitStack()
        mosek_env = self._env_stack.enter_context(mosek.Env())
        m = self._env_stack.enter_context(mosek_env.Task(0, 0))
        sense = read_sense_from_problem_file(problem_fn)
        io_api = read_io_api_from_problem_file(problem_fn)
        problem_fn_ = path_to_string(problem_fn)
        m.readdata(problem_fn_)
        self.solver_model = m
        self.io_api = io_api

        return self._solve(
            m,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api=io_api,
            sense=sense,
            from_file=True,
        )

    def _solve(
        self,
        m: mosek.Task,
        solution_fn: Path | None,
        log_fn: Path | None,
        warmstart_fn: Path | None,
        basis_fn: Path | None,
        io_api: str | None,
        sense: str | None,
        from_file: bool = False,
    ) -> Result:
        """
        Solve a linear problem from a Mosek task object.

        Parameters
        ----------
        m : mosek.Task
            Mosek task object.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        io_api: str
            io_api of the problem. For direct API from linopy model this is "direct".
        sense: str
            "min" or "max"

        Returns
        -------
        Result
        """
        CONDITION_MAP = {
            "solsta.unknown": "unknown",
            "solsta.optimal": "optimal",
            "solsta.integer_optimal": "optimal",
            "solsta.prim_infeas_cer": "infeasible",
            "solsta.dual_infeas_cer": "infeasible_or_unbounded",
        }

        for k, v in self.solver_options.items():
            m.putparam(k, str(v))

        if log_fn is not None:
            m.linkfiletostream(mosek.streamtype.log, path_to_string(log_fn), 0)
        else:
            m.set_Stream(mosek.streamtype.log, sys.stdout.write)

        if warmstart_fn is not None:
            m.putintparam(mosek.iparam.sim_hotstart, mosek.simhotstart.status_keys)
            skx = [mosek.stakey.low] * m.getnumvar()
            skc = [mosek.stakey.bas] * m.getnumcon()

            with open(path_to_string(warmstart_fn)) as f:
                for line in f:
                    if line.startswith("NAME "):
                        break

                for line in f:
                    if line.startswith("ENDATA"):
                        break

                    o = mosek_bas_re.match(line)
                    if o is not None:
                        if o.group(1) is not None:
                            key = o.group(1)
                            try:
                                skx[m.getvarnameindex(o.group(2))] = mosek.stakey.basis
                            except:  # noqa: E722
                                pass
                            try:
                                skc[m.getvarnameindex(o.group(3))] = (
                                    mosek.stakey.low if key == "XL" else "XU"
                                )
                            except:  # noqa: E722
                                pass
                        else:
                            key = o.group(4)
                            name = o.group(5)
                            stakey = (
                                mosek.stakey.low
                                if key == "LL"
                                else (
                                    mosek.stakey.upr
                                    if key == "UL"
                                    else mosek.stakey.bas
                                )
                            )

                            try:
                                skx[m.getvarnameindex(name)] = stakey
                            except:  # noqa: E722
                                try:
                                    skc[m.getvarnameindex(name)] = stakey
                                except:  # noqa: E722
                                    pass
            m.putskc(mosek.soltype.bas, skc)
            m.putskx(mosek.soltype.bas, skx)
        m.optimize()

        m.solutionsummary(mosek.streamtype.log)

        if basis_fn is not None:
            if m.solutiondef(mosek.soltype.bas):
                with open(path_to_string(basis_fn), "w") as f:
                    f.write(f"NAME {basis_fn}\n")

                    skc = [
                        (0 if sk != mosek.stakey.bas else 1, i, sk)
                        for (i, sk) in enumerate(m.getskc(mosek.soltype.bas))
                    ]
                    skx = [
                        (0 if sk == mosek.stakey.bas else 1, j, sk)
                        for (j, sk) in enumerate(m.getskx(mosek.soltype.bas))
                    ]
                    skc.sort()
                    skc.reverse()
                    skx.sort()
                    skx.reverse()
                    while skx and skc and skx[-1][0] == 0 and skc[-1][0] == 0:
                        (_, i, kc) = skc.pop()
                        (_, j, kx) = skx.pop()

                        namex = m.getvarname(j)
                        namec = m.getconname(i)

                        if kc in [mosek.stakey.low, mosek.stakey.fix]:
                            f.write(f" XL {namex} {namec}\n")
                        else:
                            f.write(f" XU {namex} {namec}\n")
                    while skc and skc[-1][0] == 0:
                        (_, i, kc) = skc.pop()
                        namec = m.getconname(i)
                        if kc in [mosek.stakey.low, mosek.stakey.fix]:
                            f.write(f" LL {namex}\n")
                        else:
                            f.write(f" UL {namex}\n")
                    while skx:
                        (_, j, kx) = skx.pop()
                        namex = m.getvarname(j)
                        if kx == mosek.stakey.bas:
                            f.write(f" BS {namex}\n")
                        elif kx in [mosek.stakey.low, mosek.stakey.fix]:
                            f.write(f" LL {namex}\n")
                        elif kx == mosek.stakey.upr:
                            f.write(f" UL {namex}\n")
                    f.write("ENDATA\n")

        # Inspect both bas and itr (and itg for MILPs) and pick the
        # solution with the best status. Reading only the interior-point
        # solution may discard a valid crossover optimum.
        soltype = Mosek._choose_solution(m)

        if solution_fn is not None and soltype is not None:
            try:
                m.writesolution(soltype, path_to_string(solution_fn))
            except mosek.Error as err:
                logger.info("Unable to save solution file. Raised error: %s", err)

        if soltype is None:
            condition = "no solution available"
            status = Status.from_termination_condition(
                TerminationCondition.internal_solver_error
            )
            status.legacy_status = condition
            return self._make_result(status, None)

        condition = str(m.getsolsta(soltype))
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            objective = m.getprimalobj(soltype)

            sol_values = np.asarray(m.getxx(soltype), dtype=float)
            if from_file:
                sol = _solution_from_names(
                    sol_values,
                    [m.getvarname(i) for i in range(m.getnumvar())],
                    self._n_vars,
                )
            else:
                sol = _solution_from_labels(sol_values, self._vlabels, self._n_vars)

            try:
                dual_values = np.asarray(m.gety(soltype), dtype=float)
                if from_file:
                    dual = _solution_from_names(
                        dual_values,
                        [m.getconname(i) for i in range(m.getnumcon())],
                        self._n_cons,
                    )
                else:
                    dual = _solution_from_labels(
                        dual_values,
                        self._clabels,
                        self._n_cons,
                    )
            except (mosek.Error, AttributeError):
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = np.array([], dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        self.io_api = io_api
        return self._make_result(status, solution, solver_model=m)


class COPT(Solver[None]):
    """
    Solver subclass for the COPT solver.

    https://guide.coap.online/copt/en-doc/index.html

    For more information on solver options, see
    https://guide.coap.online/copt/en-doc/parameter.html

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "COPT"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }
    )

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("coptpy")

    @classmethod
    def _license_probe(cls) -> None:
        env = coptpy.Envr()
        env.close()

    def _run_file(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        **kw: Any,
    ) -> Result:
        problem_fn = self._problem_fn
        assert problem_fn is not None
        # conditions: https://guide.coap.online/copt/en-doc/constant.html#chapconst-solstatus
        CONDITION_MAP = {
            0: "unstarted",
            1: "optimal",
            2: "infeasible",
            3: "unbounded",
            4: "infeasible_or_unbounded",
            5: "numerical",
            6: "node_limit",
            7: "imprecise",
            8: "time_limit",
            9: "unfinished",
            10: "interrupted",
        }

        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        if env is None:
            env_ = coptpy.Envr()

        try:
            m = env_.createModel()

            m.read(path_to_string(problem_fn))

            if log_fn is not None:
                m.setLogFile(path_to_string(log_fn))

            for k, v in self.solver_options.items():
                m.setParam(k, v)

            if warmstart_fn is not None:
                m.readBasis(path_to_string(warmstart_fn))

            m.solve()

            if basis_fn and m.HasBasis:
                try:
                    m.write(path_to_string(basis_fn))
                except coptpy.CoptError as err:
                    logger.warning("No model basis stored. Raised error: %s", err)

            if solution_fn:
                try:
                    m.write(path_to_string(solution_fn))
                except coptpy.CoptError as err:
                    logger.warning("No model solution stored. Raised error: %s", err)

            # TODO: check if this suffices
            condition = m.MipStatus if m.ismip else m.LpStatus
            termination_condition = CONDITION_MAP.get(condition, str(condition))
            status = Status.from_termination_condition(termination_condition)
            status.legacy_status = str(condition)

            def get_solver_solution() -> Solution:
                # TODO: check if this suffices
                objective = m.BestObj if m.ismip else m.LpObjVal

                vars_ = m.getVars()
                sol = _solution_from_names(
                    np.array([v.x for v in vars_], dtype=float),
                    [v.name for v in vars_],
                    self._n_vars,
                )

                try:
                    cons = m.getConstrs()
                    dual = _solution_from_names(
                        np.array([c.pi for c in cons], dtype=float),
                        [c.name for c in cons],
                        self._n_cons,
                    )
                except (coptpy.CoptError, AttributeError):
                    logger.warning("Dual values of MILP couldn't be parsed")
                    dual = np.array([], dtype=float)

                return Solution(sol, dual, objective)

            solution = self.safe_get_solution(status=status, func=get_solver_solution)
            solution = maybe_adjust_objective_sign(solution, io_api, sense)

            self.io_api = io_api
            return self._make_result(status, solution, solver_model=m)
        finally:
            env_.close()


class MindOpt(Solver[None]):
    """
    Solver subclass for the MindOpt solver.

    https://solver.damo.alibaba.com/doc/en/html/index.html

    For more information on solver options, see
    https://solver.damo.alibaba.com/doc/en/html/API2/param/index.html

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "MindOpt"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }
    )

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("mindoptpy")

    @classmethod
    def _license_probe(cls) -> None:
        env = mindoptpy.Env()
        env.dispose()

    def _run_file(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        **kw: Any,
    ) -> Result:
        problem_fn = self._problem_fn
        assert problem_fn is not None
        CONDITION_MAP = {
            -1: "error",
            0: "unknown",
            1: "optimal",
            2: "infeasible",
            3: "unbounded",
            4: "infeasible_or_unbounded",
            5: "suboptimal",
        }
        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        if io_api == "lp":
            # for model type "QP", lp file with have "[" and "]" in objective function
            if "[" in open(problem_fn).read() and "]" in open(problem_fn).read():
                msg = (
                    "MindOpt does not support QP problems in LP format. Use MPS file "
                    "format instead."
                )
                raise ValueError(msg)

        if env is None:
            env_ = mindoptpy.Env(path_to_string(log_fn) if log_fn else "")

        m = None
        try:
            env_.start()

            m = mindoptpy.read(path_to_string(problem_fn), env_)

            for k, v in self.solver_options.items():
                m.setParam(k, v)

            if warmstart_fn:
                try:
                    m.read(path_to_string(warmstart_fn))
                except mindoptpy.MindoptError as err:
                    logger.info("Model basis could not be read. Raised error: %s", err)

            m.optimize()

            if basis_fn:
                try:
                    m.write(path_to_string(basis_fn))
                except mindoptpy.MindoptError as err:
                    logger.info("No model basis stored. Raised error: %s", err)

            if solution_fn:
                try:
                    m.write(path_to_string(solution_fn))
                except mindoptpy.MindoptError as err:
                    logger.info("No model solution stored. Raised error: %s", err)

            condition = m.status
            termination_condition = CONDITION_MAP.get(condition, condition)
            status = Status.from_termination_condition(termination_condition)
            status.legacy_status = condition

            def get_solver_solution() -> Solution:
                assert m is not None
                objective = m.objval

                vars_ = m.getVars()
                sol = _solution_from_names(
                    np.array([v.X for v in vars_], dtype=float),
                    [v.VarName for v in vars_],
                    self._n_vars,
                )

                try:
                    cons = m.getConstrs()
                    dual = _solution_from_names(
                        np.array([c.DualSoln for c in cons], dtype=float),
                        [c.ConstrName for c in cons],
                        self._n_cons,
                    )
                except (mindoptpy.MindoptError, AttributeError):
                    logger.warning("Dual values of MILP couldn't be parsed")
                    dual = np.array([], dtype=float)

                return Solution(sol, dual, objective)

            solution = self.safe_get_solution(status=status, func=get_solver_solution)
            solution = maybe_adjust_objective_sign(solution, io_api, sense)

            self.io_api = io_api
            return self._make_result(status, solution, solver_model=m)
        finally:
            if m is not None:
                m.dispose()
            env_.dispose()


class PIPS(Solver[None]):
    """
    Solver subclass for the PIPS solver.
    """

    def __post_init__(self) -> None:
        msg = "The PIPS solver interface is not yet implemented."
        raise NotImplementedError(msg)


class cuPDLPx(Solver[None]):
    """
    Solver subclass for the cuPDLPx solver. cuPDLPx must be installed
    with working GPU support for usage. Find the installation instructions
    at https://github.com/MIT-Lu-Lab/cuPDLPx.

    The full list of solver options provided with the python interface
    is documented at https://github.com/MIT-Lu-Lab/cuPDLPx/tree/main/python.

    Some example options are:
    * LogToConsole : False by default.
    * TimeLimit : 3600.0 by default.
    * IterationLimit : 2147483647 by default.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "cuPDLPx"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.DIRECT_API,
            SolverFeature.GPU_ACCELERATION,
            SolverFeature.GPU_ONLY,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }
    )

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("cupdlpx")

    @classmethod
    def _license_probe(cls) -> None:
        cupdlpx.Model(np.array([0.0]), np.array([[0.0]]), None, None)

    def _run_file(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: EnvType | None = None,
        **kw: Any,
    ) -> Result:
        problem_fn = self._problem_fn
        assert problem_fn is not None
        logger.warning(
            "cuPDLPx doesn't currently support file IO. Building model from file using linopy."
        )
        problem_fn_ = path_to_string(problem_fn)

        if problem_fn_.endswith(".netcdf"):
            model: Model = linopy.io.read_netcdf(problem_fn_)
        else:
            msg = "linopy currently only supports reading models from netcdf files. Try using io_api='direct' instead."
            raise NotImplementedError(msg)

        self.model = model
        self._build_direct()
        return self._run_direct(
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            env=env,
        )

    def _build_direct(self, **kwargs: Any) -> None:
        model = self.model
        assert model is not None
        if model.type in ["QP", "MILP"]:
            msg = "cuPDLPx does not currently support QP or MILP problems."
            raise NotImplementedError(msg)
        if kwargs.get("explicit_coordinate_names"):
            warnings.warn(
                "cuPDLPx does not support named variables/constraints. "
                "The explicit_coordinate_names parameter is ignored.",
                UserWarning,
                stacklevel=2,
            )
        cu_model = self._build_solver_model(model)
        self.solver_model = cu_model
        self.io_api = "direct"
        self.sense = model.sense
        self._cache_model_labels(model)

    @staticmethod
    def _build_solver_model(model: Model) -> cupdlpx.Model:
        """Build a cupdlpx.Model that mirrors the linopy `model`."""
        if model.variables.semi_continuous:
            raise NotImplementedError(
                "Semi-continuous variables are not supported by cuPDLPx. "
                "Use a solver that supports them (gurobi, cplex, highs)."
            )

        M = model.matrices
        if M.A is None:
            raise ValueError("Model has no constraints, cannot export to cuPDLPx.")
        A = M.A.tocsr()
        lower = np.where(
            np.logical_or(np.equal(M.sense, ">"), np.equal(M.sense, "=")),
            M.b,
            -np.inf,
        )
        upper = np.where(
            np.logical_or(np.equal(M.sense, "<"), np.equal(M.sense, "=")),
            M.b,
            np.inf,
        )

        cu_model = cupdlpx.Model(
            objective_vector=M.c,
            constraint_matrix=A,
            constraint_lower_bound=lower,
            constraint_upper_bound=upper,
            variable_lower_bound=M.lb,
            variable_upper_bound=M.ub,
        )

        if model.objective.sense == "max":
            cu_model.ModelSense = cupdlpx.PDLP.MAXIMIZE

        return cu_model

    def _run_direct(
        self,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: Any = None,
        **kw: Any,
    ) -> Result:
        return self._solve(
            self.solver_model,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api=self.io_api,
            sense=self.sense,
        )

    def _solve(
        self,
        cu_model: cupdlpx.Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        io_api: str | None = None,
        sense: str | None = None,
    ) -> Result:
        """
        Solve a linear problem from a cupdlpx.Model object.

        Parameters
        ----------
        cu_model: cupdlpx.Model
            cupdlpx object.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        model : linopy.model, optional
            Linopy model for the problem.
        io_api: str
            io_api of the problem. For direct API from linopy model this is "direct".
        sense: str
            "min" or "max"

        Returns
        -------
        Result
        """

        # see https://github.com/MIT-Lu-Lab/cuPDLPx/blob/main/python/cupdlpx/PDLP.py
        CONDITION_MAP: dict[int, TerminationCondition] = {
            cupdlpx.PDLP.OPTIMAL: TerminationCondition.optimal,
            cupdlpx.PDLP.PRIMAL_INFEASIBLE: TerminationCondition.infeasible,
            cupdlpx.PDLP.DUAL_INFEASIBLE: TerminationCondition.infeasible_or_unbounded,
            cupdlpx.PDLP.TIME_LIMIT: TerminationCondition.time_limit,
            cupdlpx.PDLP.ITERATION_LIMIT: TerminationCondition.iteration_limit,
            cupdlpx.PDLP.UNSPECIFIED: TerminationCondition.unknown,
        }

        self._set_solver_params(cu_model)

        if warmstart_fn is not None:
            # cuPDLPx supports warmstart, but there currently isn't the tooling
            # to read it in from a file
            raise NotImplementedError("Warmstarting not yet implemented for cuPDLPx.")
        else:
            cu_model.clearWarmStart()

        if basis_fn is not None:
            logger.warning("Basis files are not supported by cuPDLPx. Ignoring.")

        if log_fn is not None:
            logger.warning("Log files are not supported by cuPDLPx. Ignoring.")

        # solve
        cu_model.optimize()

        # parse solution and output
        if solution_fn is not None:
            raise NotImplementedError(
                "Solution file output not yet implemented for cuPDLPx."
            )

        termination_condition = CONDITION_MAP.get(
            cu_model.StatusCode, cu_model.StatusCode
        )
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = cu_model.Status  # cuPDLPx status message

        def get_solver_solution() -> Solution:
            objective = cu_model.ObjVal
            sol = np.asarray(cu_model.X, dtype=float)
            dual = np.asarray(cu_model.Pi, dtype=float)

            if cu_model.ModelSense == cupdlpx.PDLP.MAXIMIZE:
                dual = -dual

            sol = _solution_from_labels(sol, self._vlabels, self._n_vars)
            dual = _solution_from_labels(dual, self._clabels, self._n_cons)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        runtime: float | None = None
        with contextlib.suppress(Exception):
            runtime = float(cu_model.Runtime)

        self.io_api = io_api
        return self._make_result(
            status,
            solution,
            solver_model=cu_model,
            report=SolverReport(runtime=runtime),
        )

    def _set_solver_params(self, cu_model: cupdlpx.Model) -> None:
        """
        Set solver options for cuPDLPx model.

        For list of available options, see
        https://github.com/MIT-Lu-Lab/cuPDLPx/tree/main/python#parameters
        """
        for k, v in self.solver_options.items():
            cu_model.setParam(k, v)


def _solver_class_for(name: str) -> type[Solver] | None:
    try:
        return globals().get(SolverName(name).name)
    except ValueError:
        return None


QUADRATIC_SOLVERS = [
    n.value
    for n in SolverName
    if (cls := _solver_class_for(n.value)) is not None
    and cls.supports(SolverFeature.QUADRATIC_OBJECTIVE)
]
NO_SOLUTION_FILE_SOLVERS = [
    n.value
    for n in SolverName
    if (cls := _solver_class_for(n.value)) is not None
    and cls.supports(SolverFeature.SOLUTION_FILE_NOT_NEEDED)
]


# Defines the iteration order of ``available_solvers`` — the first installed
# entry is the default solver in :meth:`Model.solve`. Matches the historical
# eager-probe order from before lazy availability landed.
_SOLVER_PROBE_ORDER: tuple[str, ...] = (
    "gurobi",
    "highs",
    "glpk",
    "cbc",
    "scip",
    "cplex",
    "xpress",
    "knitro",
    "mosek",
    "mindopt",
    "copt",
    "cupdlpx",
    "pips",
)


class _AvailableSolvers(Sequence[str]):
    """
    Lazy sequence of installed solver names.

    Probes each solver's :meth:`Solver.is_available` on first access and caches
    the result. Membership means the solver's Python package or binary is
    importable — it does **not** mean a working license exists. Call
    :func:`check_solver_licenses` for an opt-in eager license probe.

    :meth:`refresh` clears the cache (and each per-class ``is_available``
    cache) so the probe re-runs.
    """

    _filter: ClassVar[frozenset[str] | None] = None

    @functools.cached_property
    def _names(self) -> list[str]:
        names: list[str] = []
        for name in _SOLVER_PROBE_ORDER:
            if self._filter is not None and name not in self._filter:
                continue
            cls = _solver_class_for(name)
            if cls is not None and cls.is_available():
                names.append(name)
        return names

    def __contains__(self, item: object) -> bool:
        return item in self._names

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, idx: int | slice) -> Any:
        return self._names[idx]

    def __repr__(self) -> str:
        return repr(self._names)

    def __bool__(self) -> bool:
        return bool(self._names)

    def refresh(self) -> None:
        self.__dict__.pop("_names", None)
        seen: set[int] = set()
        for name in _SOLVER_PROBE_ORDER:
            cls = _solver_class_for(name)
            if cls is None:
                continue
            fn = cls.__dict__.get("is_available")
            if fn is None:
                continue
            cache_clear = getattr(fn, "cache_clear", None)
            if cache_clear is not None and id(fn) not in seen:
                cache_clear()
                seen.add(id(fn))


class _QuadraticSolvers(_AvailableSolvers):
    _filter: ClassVar[frozenset[str] | None] = frozenset(QUADRATIC_SOLVERS)


class _LicensedSolvers(_AvailableSolvers):
    """Installed solvers whose ``license_status()`` probe currently succeeds."""

    @functools.cached_property
    def _names(self) -> list[str]:
        names: list[str] = []
        for name in _SOLVER_PROBE_ORDER:
            cls = _solver_class_for(name)
            if cls is None or not cls.is_available():
                continue
            if cls.license_status().ok:
                names.append(name)
        return names


available_solvers = _AvailableSolvers()
quadratic_solvers = _QuadraticSolvers()
licensed_solvers = _LicensedSolvers()


def check_solver_licenses(*names: str) -> dict[str, LicenseStatus]:
    """Probe license status for the given solvers, or all installed ones."""
    targets = names or tuple(available_solvers)
    out: dict[str, LicenseStatus] = {}
    for n in targets:
        cls = _solver_class_for(n)
        if cls is None:
            raise ValueError(f"unknown solver: {n!r}")
        out[n] = cls.license_status()
    return out
