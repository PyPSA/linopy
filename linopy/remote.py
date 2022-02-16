#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:34:55 2022.

@author: fabian
"""

import logging
import tempfile
from dataclasses import dataclass
from typing import Union

import paramiko
from paramiko.sftp_client import SFTPClient

from linopy.io import read_netcdf

logger = logging.getLogger(__name__)

command = """
import linopy

m = linopy.read_netcdf("{model_unsolved_file}")
m.solve({solve_kwargs})
m.to_netcdf("{model_solved_file}")
"""


@dataclass
class RemoteScheduler:

    hostname: str = None
    port: int = 22
    username: str = None
    password: str = None
    client: paramiko.SSHClient = None
    _sftp_client: SFTPClient = None

    pre_execution: str = "source ~/.bashrc"  # force login shell
    post_execution: str = None

    python_script: Union[callable, str] = command.format
    python_executable: str = "python"
    python_file: str = "/tmp/linopy-execution.py"

    solve_kwargs = {}
    model_unsolved_file: str = "/tmp/linopy-unsolved-model.nc"
    model_solved_file: str = "/tmp/linopy-solved-model.nc"

    def __post_init__(self):
        if self.client is None:
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.connect(self.hostname, self.port, self.username, self.password)
            self.client = client
        logger.info("Open an SFTP session on the SSH server")
        self._sftp_client = self.client.open_sftp()

    def write_python_file_on_remote(self):
        logger.info(f"Writing python script at {self.python_file} on remote")
        script_kwargs = dict(
            model_unsolved_file=self.model_unsolved_file,
            solve_kwargs="**" + str(self.solve_kwargs),
            model_solved_file=self.model_solved_file,
        )
        with self._sftp_client.open(self.python_file, "w") as fn:
            fn.write(self.python_script(**script_kwargs))

    def write_model_on_remote(self, model):
        logger.info(f"Writing unsolved model at {self.model_unsolved_file} on remote")
        with tempfile.NamedTemporaryFile(prefix="linopy", suffix=".nc") as fn:
            model.to_netcdf(fn.name)
            self._sftp_client.put(fn.name, self.model_unsolved_file)

    def execute(self, command):
        stdin, stdout, stderr = self.client.exec_command(command, get_pty=True)
        stdin.close()

        for line in iter(stdout.readline, ""):
            print(line, end="")

        raised_error = False
        for line in iter(stderr.readline, ""):
            print(line, end="")
            raised_error = True
        if raised_error:
            raise OSError("Execution on remote raised an error, see above.")

    def solve_on_remote(self, model, **kwargs):

        if self.pre_execution:
            self.execute(self.pre_execution)

        if kwargs:
            self.solve_kwargs = kwargs
        self.write_python_file_on_remote()
        self.write_model_on_remote(model)

        command = self.python_executable + " " + self.python_file
        self.execute(command)

        if self.post_execution:
            self.execute(self.post_execution)

        with tempfile.NamedTemporaryFile(prefix="linopy", suffix=".nc") as fn:
            self._sftp_client.get(self.model_solved_file, fn.name)
            return read_netcdf(fn.name)
