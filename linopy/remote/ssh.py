#!/usr/bin/env python3
"""
Created on Sun Feb 13 21:34:55 2022.

@author: fabian
"""

import logging
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Union

from linopy.constants import Result
from linopy.io import read_netcdf
from linopy.sos_reformulation import (
    sos_reformulation_context,
    suppress_serialization_warning,
)

if TYPE_CHECKING:
    from linopy.model import Model

paramiko_present = True
try:
    import paramiko
except ImportError:
    paramiko_present = False
logger = logging.getLogger(__name__)

command = """
import linopy

m = linopy.read_netcdf("{model_unsolved_file}")
m.solve({solve_kwargs})
m.to_netcdf("{model_solved_file}")
"""


@dataclass
class SshSettings:
    """
    Transport-only config for the :class:`linopy.solvers.SSH` solver.

    Inner solver name and solver options come from :meth:`Model.solve` —
    ``m.solve("gurobi", remote=SshSettings(hostname=...), presolve="on")``.

    Use ``setup_commands`` to prepare the remote shell before the solve —
    e.g. activate a conda environment or set ``PATH``::

        SshSettings(hostname=..., setup_commands=["conda activate linopy-env"])
    """

    hostname: str
    port: int = 22
    username: str | None = None
    password: str | None = None
    python_executable: str = "python"
    python_file: str = "/tmp/linopy-execution.py"
    model_unsolved_file: str = "/tmp/linopy-unsolved-model.nc"
    model_solved_file: str = "/tmp/linopy-solved-model.nc"
    setup_commands: list[str] = field(default_factory=list)


@dataclass
class RemoteHandler:
    """
    Handler class for solving models on a remote machine via an SSH connection.

    .. deprecated::
        ``RemoteHandler`` is the legacy low-level entry point and will be
        removed in a future release. Prefer
        ``Model.solve("gurobi", remote=SshSettings(hostname=...))`` or
        instantiate :class:`SSH` directly.

    The basic idea of the handler is to provide a workflow that:

        1. defines a model on the local machine
        2. saves it to a file on the local machine
        3. copies that file to the remote machine
        4. loads, solves and writes out the model, all on the remote machine
        5. copies the solved model to the local machine
        6. loads the solved model on the local machine


    The Handler opens an interactive shell in which the commands are executed.
    All standard outputs of the remote are directly displayed in the local prompt.
    You can directly set a connected SSH client for the RemoteHandler if you
    don't want to use the default connection parameters `host`, `username` and
    `password`.

    If the SSH keys are stored in a default location, the keys are autodetected
    and the RemoteHandler does not require a password argument.

    Parameters
    ----------
    hostname : str
        Name of the server to connect to. This is used if client is None.
    port : int
        The server port to connect to. This is used if client is None.
    username : str
        The username to authenticate as (defaults to the current local username).
        This is used if client is None.
    password : str
        Used for password authentication; is also used for private key
        decryption. Not necessary if ssh keys are auto-detectable.
        This is used if client is None.
    client : paramiko.SSHClient
        Already connected client to use instead of initializing a one with
        the above arguments.
    python_script : callable
        Format function which takes the arguments `model_unsolved_file`,
        `solve_kwargs` and `model_solved_files`. Defaults to
        `linopy.remote.command.format`, where `linopy.remote.command` is the
        string of the python command.
    python_executable : str
        Python executable to use on the remote machine.
    python_file : str
        Path where to store the python script on the remote machine.
    model_unsolved_file : str
        Path where to temporarily store the unsolved model on the local machine
        before copying it over.
    model_solved_file : str
        Path where to temporarily store the solved model on the remote machine.


    Example
    -------

    >>> import linopy
    >>> from linopy import Model
    >>> from numpy import arange
    >>> from xarray import DataArray
    >>>
    >>> N = 10
    >>> m = Model()
    >>> coords = [arange(N), arange(N)]
    >>> x = m.add_variables(coords=coords)
    >>> y = m.add_variables(coords=coords)
    >>> con1 = m.add_constraints(x - y >= DataArray(arange(N)))
    >>> con2 = m.add_constraints(x + y >= 0)
    >>> obj = m.add_objective((2 * x + y).sum())
    >>>
    >>> host = "my-remote-machine.com"
    >>> username = "my-username"
    >>> handler = linopy.remote.RemoteHandler(host, username=username)  # doctest: +SKIP
    >>>
    >>> # optionally activate a conda environment
    >>> handler.execute("conda activate my-linopy-env")  # doctest: +SKIP
    >>>
    >>> m = handler.solve_on_remote(m)  # doctest: +SKIP
    """

    hostname: str
    port: int = 22
    username: str | None = None
    password: str | None = None
    client: Union["paramiko.SSHClient", None] = None

    python_script: Callable = command.format
    python_executable: str = "python"
    python_file: str = "/tmp/linopy-execution.py"

    model_unsolved_file: str = "/tmp/linopy-unsolved-model.nc"
    model_solved_file: str = "/tmp/linopy-solved-model.nc"

    _internal: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        assert paramiko_present, "The required paramiko package is not installed."

        if not self._internal:
            warnings.warn(
                "`RemoteHandler` is deprecated; use `SSH(settings, solver_name, "
                "options)` from `linopy.remote` or `Model.solve(remote=SshSettings"
                "(hostname=...))`. `RemoteHandler` will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.client is None:
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.connect(self.hostname, self.port, self.username, self.password)
            self.client = client

        logger.info("Open interactive shell session.")
        self.channel = self.client.invoke_shell()
        self.stdin = self.channel.makefile("wb", -1)
        self.stdout = self.channel.makefile("r", -1)
        self.stderr = self.channel.makefile("r", -1)

        logger.info("Open an SFTP session on the SSH server")
        self.sftp_client = self.client.open_sftp()

    def __del__(self) -> None:
        if self.client is not None:
            self.client.close()

    def write_python_file_on_remote(self, **solve_kwargs: Any) -> None:
        """
        Write the python file of the RemoteHandler on the remote machine under
        `self.python_file`.
        """
        logger.info(f"Saving python script at {self.python_file} on remote")
        script_kwargs: dict[str, str] = dict(
            model_unsolved_file=self.model_unsolved_file,
            solve_kwargs=f"**{solve_kwargs}",
            model_solved_file=self.model_solved_file,
        )
        with self.sftp_client.open(self.python_file, "w") as fn:
            fn.write(self.python_script(**script_kwargs))

    def write_model_on_remote(self, model: "Model") -> None:
        """
        Write a model on the remote machine under `self.model_unsolved_file`.
        """
        logger.info(f"Saving unsolved model at {self.model_unsolved_file} on remote")
        with tempfile.NamedTemporaryFile(prefix="linopy", suffix=".nc") as fn:
            model.to_netcdf(fn.name)
            self.sftp_client.put(fn.name, self.model_unsolved_file)

    def execute(self, cmd: str) -> None:
        """
        Execute a shell command on the remote machine.
        """
        cmd = cmd.strip("\n")
        self.stdin.write(cmd + "\n")
        finish: str = "End of stdout. Exit Status"
        echo_cmd: str = f"echo {finish} $?"
        self.stdin.write(echo_cmd + "\n")
        self.stdin.flush()

        print_stdout = False
        exit_status = 0
        for line in self.stdout:
            line = str(line).strip("\n").strip()
            if line.endswith(cmd):
                # up to now everything was login and stdin
                print_stdout = True
            elif line.startswith(finish):
                exit_status = int(line.rsplit(maxsplit=1)[1])
                break
            elif finish not in line and print_stdout:
                print(line)

        if exit_status:
            raise OSError("Execution on remote raised an error, see above.")

    def solve_on_remote(
        self,
        model: "Model",
        *,
        reformulate_sos: bool | Literal["auto"] = False,
        **kwargs: Any,
    ) -> "Model":
        """
        Solve a linopy model on the remote machine.

        Reformulates SOS constraints locally before serialization when
        requested, so the worker just solves a plain MILP and the SOS
        lifecycle stays on the caller's model.

        Parameters
        ----------
        model : linopy.model.Model
        reformulate_sos : bool | "auto", optional
            Forwarded to ``Model._resolve_sos_reformulation`` to decide
            whether to apply SOS reformulation locally before transfer.
        **kwargs :
            Keyword arguments passed to `linopy.model.Model.solve` on the
            remote worker.

        Returns
        -------
        linopy.model.Model
            Solved model.
        """
        solver_name = kwargs.get("solver_name")
        with sos_reformulation_context(model, solver_name, reformulate_sos) as applied:
            self.write_python_file_on_remote(**kwargs)
            with suppress_serialization_warning(active=applied):
                self.write_model_on_remote(model)

            command = f"{self.python_executable} {self.python_file}"

            logger.info("Solving model on remote.")
            self.execute(command)

            logger.info("Retrieve solved model from remote.")
            with tempfile.NamedTemporaryFile(prefix="linopy", suffix=".nc") as fn:
                self.sftp_client.get(self.model_solved_file, fn.name)
                solved = read_netcdf(fn.name)

            self.sftp_client.remove(self.python_file)
            self.sftp_client.remove(self.model_solved_file)

            return solved


@dataclass
class SSH:
    """
    Remote handler that solves a linopy model on a remote machine over SSH.

    This is a standalone class — *not* a :class:`linopy.solvers.Solver`
    subclass. It ships the model to a remote host and runs
    ``read_netcdf(...).solve(solver_name=...)`` there, pulling the solved
    netcdf back.

    Parameters
    ----------
    settings : SshSettings
        Connection + remote-execution paths.
    solver_name : str
        Inner solver to run on the remote (e.g. ``"gurobi"``).
    options : dict, optional
        Solver options passed through to the inner solver.

    Notes
    -----
    Synchronous; unlike OETC the remote shell job is short-lived and
    doesn't expose a useful submit/collect seam.
    """

    settings: SshSettings
    solver_name: str
    options: dict[str, Any] = field(default_factory=dict)

    _handler: "RemoteHandler | None" = field(init=False, default=None, repr=False)
    _solved_model: Any = field(init=False, default=None, repr=False)

    @classmethod
    def is_available(cls) -> bool:
        """Return True iff paramiko is importable."""
        return paramiko_present

    def solve(self, model: "Model") -> Result:
        """Ship the model, run the inner solver on the remote, return a Result."""
        from linopy.constants import Status
        from linopy.remote._common import (
            _scatter_solution_from_solved_model,
            _validate_inner_solver,
        )

        _validate_inner_solver(self.solver_name, model)

        if self._handler is None:
            self._handler = RemoteHandler(
                hostname=self.settings.hostname,
                port=self.settings.port,
                username=self.settings.username,
                password=self.settings.password,
                python_executable=self.settings.python_executable,
                python_file=self.settings.python_file,
                model_unsolved_file=self.settings.model_unsolved_file,
                model_solved_file=self.settings.model_solved_file,
                _internal=True,
            )
            for cmd in self.settings.setup_commands:
                self._handler.execute(cmd)

        solve_kwargs: dict[str, Any] = {"solver_name": self.solver_name}
        if self.options:
            solve_kwargs.update(self.options)
        solved = self._handler.solve_on_remote(model, **solve_kwargs)
        self._solved_model = solved

        status = Status.from_termination_condition(solved.termination_condition)
        solution = _scatter_solution_from_solved_model(
            model, solved, model._xCounter, model._cCounter
        )
        return Result(
            status=status,
            solution=solution,
            solver_name=self.solver_name,
        )
