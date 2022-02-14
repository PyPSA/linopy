#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:34:55 2022.

@author: fabian
"""

from dataclasses import dataclass
from typing import Union

import paramiko
from paramiko.sftp_client import SFTPClient

from linopy.io import read_netcdf

command = """
import linopy

m = linopy.read_netcdf({model_unsolved_file})
m.solve({solve_kwargs})
m.to_netcdf({model_solved_file})
"""


@dataclass
class RemoteScheduler:

    hostname: str = None
    port: int = 22
    username: str = None
    password: str = None
    client: paramiko.SSHClient = None
    _sftp_client: SFTPClient = None

    pre_execution: str = None
    post_execution: str = None

    python_script: Union[callable, str] = command.format
    python_executable: str = "python"
    python_file: str = "/tmp/linopy-execution.py"

    model_unsolved_file: str = "/tmp/linopy-unsolved-model.nc"
    model_solved_file: str = "/tmp/linopy-solved-model.nc"

    def __post_init__(self):
        if self.client is None:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, self.port, self.username, self.password)
        self._sftp_client = self.open_sftp()

    def write_python_script_on_remote(self):
        script_kwargs = dict(
            model_unsolved_file=self.model_unsolved_file,
            solve_kwargs=self.solve_kwargs,
            model_solved_file=self.model_solved_file,
        )
        with self.client.open(self.python_script_file, "w") as fn:
            fn.write(self.python_script(**script_kwargs))

    def write_model_on_remote(self, model):
        with self.client.open(self.model_unsolved_file, "bw") as fn:
            fn.write(model)

    def execute(self, command):
        stdin, stdout, stderr = self.client.exec_command(command)
        stdin.close()

        for line in iter(stdout.readline, ""):
            print(line, end="")

        for line in iter(stderr.readline, ""):
            print(line, end="")
            raise OSError("Execution on remote raised an error, see above.")

    def solve_on_remote(self, model):

        if self.pre_execution:
            self.execute(self.pre_execution)

        self.write_python_file_on_remote()
        self.write_model_on_remote()

        command = self.python_executable + " " + self.python_file
        self.execute(command)

        if self.post_execution:
            self.execute(self.post_execution)

        with self.client.open(self.model_solved_file) as fn:
            return read_netcdf(fn)
