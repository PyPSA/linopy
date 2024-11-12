#!/usr/bin/env python3
"""
Created on Sun Feb 13 21:34:55 2022.

@author: fabian
"""
import gzip
import json
import logging
import tempfile
from dataclasses import dataclass
import os
from pathlib import Path

from google.cloud import storage
from google.cloud.storage import Client, Bucket, Blob
from google.oauth2 import service_account
from requests import Response, post, get
from time import sleep
from typing import Callable, Union

import linopy
from linopy.io import read_netcdf

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
class RemoteHandler:
    """
    Handler class for solving models on a remote machine via an SSH connection.

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
    username: Union[str, None] = None
    password: Union[str, None] = None
    client: Union["paramiko.SSHClient", None] = None

    python_script: Callable = command.format
    python_executable: str = "python"
    python_file: str = "/tmp/linopy-execution.py"

    model_unsolved_file: str = "/tmp/linopy-unsolved-model.nc"
    model_solved_file: str = "/tmp/linopy-solved-model.nc"

    def __post_init__(self):
        assert paramiko_present, "The required paramiko package is not installed."

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

    def __del__(self):
        if self.client is not None:
            self.client.close()

    def write_python_file_on_remote(self, **solve_kwargs):
        """
        Write the python file of the RemoteHandler on the remote machine under
        `self.python_file`.
        """
        logger.info(f"Saving python script at {self.python_file} on remote")
        script_kwargs = dict(
            model_unsolved_file=self.model_unsolved_file,
            solve_kwargs=f"**{solve_kwargs}",
            model_solved_file=self.model_solved_file,
        )
        with self.sftp_client.open(self.python_file, "w") as fn:
            fn.write(self.python_script(**script_kwargs))

    def write_model_on_remote(self, model):
        """
        Write a model on the remote machine under `self.model_unsolved_file`.
        """
        logger.info(f"Saving unsolved model at {self.model_unsolved_file} on remote")
        with tempfile.NamedTemporaryFile(prefix="linopy", suffix=".nc") as fn:
            model.to_netcdf(fn.name)
            self.sftp_client.put(fn.name, self.model_unsolved_file)

    def execute(self, cmd):
        """
        Execute a shell command on the remote machine.
        """
        cmd = cmd.strip("\n")
        self.stdin.write(cmd + "\n")
        finish = "End of stdout. Exit Status"
        echo_cmd = f"echo {finish} $?"
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

    def solve_on_remote(self, model, **kwargs):
        """
        Solve a linopy model on the remote machine.

        This function

            1. saves the model to a file on the local machine.
            2. copies that file to the remote machine.
            3. loads, solves and writes out the model, all on the remote machine.
            4. copies the solved model to the local machine.
            5. loads and returns the solved model.

        Parameters
        ----------
        model : linopy.model.Model
        **kwargs :
            Keyword arguments passed to `linopy.model.Model.solve`.

        Returns
        -------
        linopy.model.Model
            Solved model.
        """
        self.write_python_file_on_remote(**kwargs)
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


# TODO perhaps we make RemoteHandler an abstract base class, and rename the class above as SshHandler,
#  and have that and OetCloudHandler inherit from RemoteHandler?
class OetCloudHandler:

    def __init__(self):
        self.oetc_url = os.environ.get('OETC_URL', 'http://127.0.0.1:5000')
        # TODO: Temporary solution, in the future read service key from auth API
        with open(
            os.environ["GCP_SERVICE_KEY_PATH"], "r"
        ) as service_key_file:
            self.gcp_service_key = json.load(service_key_file)
        # TODO: Temporary solution, in the future use auth login to obtain the JWT
        self.jwt = os.environ["OETC_JWT"]

    def solve_on_remote(self, model, **kwargs):
        """
        Solve a linopy model on the OET Cloud compute app.

        Parameters
        ----------
        model : linopy.model.Model
        **kwargs :
            Keyword arguments passed to `linopy.model.Model.solve`.

        Returns
        -------
        linopy.model.Model
            Solved model.
        """
        logger.warning(f'Ignoring these kwargs for now: {kwargs}')  # TODO

        with tempfile.NamedTemporaryFile(prefix="linopy-", suffix=".nc") as fn:
            model.to_netcdf(fn.name)
            logger.info(f'Model written to: {fn.name}')
            input_file_name = self._upload_file_to_gcp(fn.name)

            job_uuid = self._submit_job(input_file_name)
            job_data = self._wait_and_get_job_data(job_uuid)

            out_file_name = job_data['output_files'][0]['name']
            out_file_name_path = self._download_file_from_gcp(out_file_name)

            solution = linopy.read_netcdf(out_file_name_path)
            os.remove(out_file_name_path)
            logger.info(
                f'OETC result: {solution.status}, {solution.termination_condition}, Objective: {solution.objective.value:.2e}'
            )

            return solution

    def _submit_job(self, input_file_name):
        logger.info('Calling OETC...')
        response: Response = post(
            f"{self.oetc_url}/compute-job/create",
            headers={"Authorization": f"Bearer {self.jwt}"},
            json={
                "input_file_name": input_file_name,
            },
        )
        response.raise_for_status()
        content = response.json()
        logger.info(f'OETC job submitted successfully. ID: {content["uuid"]}')
        return content['uuid']

    def _wait_and_get_job_data(self, uuid: str, retry_every_s=60) -> dict:
        """Waits for job completion until it completes or an error occurs,
        waiting `retry_every_s` seconds in between retries; returns job data including output file download links."""
        while True:
            logger.info('Checking job status...')
            response: Response = get(
                f"{self.oetc_url}/compute-job/{uuid}",
                headers={"Authorization": f"Bearer {self.jwt}"},
            )
            if not response.ok:
                raise ValueError(f'OETC Error: {response.text}')
            content = response.json()
            if not "status" in content:
                raise ValueError(f'Unexpected response: {response.text}')
            if content['status'] == 'FINISHED':
                logger.info('OETC completed job execution')
                return content
            elif content['status'] not in {'RUNNING', 'PENDING'}:
                raise ValueError(f"Unexpected status: {content['status']}")
            logger.info('OETC still crunching...')
            sleep(retry_every_s)

    def _gzip_compress(self, source_path: str) -> str:
        output_path = source_path + ".gz"
        chunk_size = 1024 * 1024

        with open(source_path, "rb") as f_in:
            with gzip.open(output_path, "wb", compresslevel=9) as f_out:
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)

        return output_path

    def _gzip_decompress(self, input_path: str) -> str:
        output_path = str(Path(input_path).with_suffix(""))

        chunk_size = 1024 * 1024

        with gzip.open(input_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)

        return output_path

    def _upload_file_to_gcp(self, file_path: str) -> str:
        logger.info(f'Compressing model...')
        compressed_file_path = self._gzip_compress(file_path)
        logger.info(f'Model compressed to: {compressed_file_path}')
        compressed_file_name = os.path.basename(compressed_file_path)

        credentials = service_account.Credentials.from_service_account_info(
            self.gcp_service_key,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        storage_client: Client = storage.Client(credentials=credentials)
        bucket: Bucket = storage_client.bucket(
            "oetc_files"  # TODO: Make bucket name dynamic if necessary
        )
        blob: Blob = bucket.blob(compressed_file_name)

        logger.info(f"Uploading model from {compressed_file_path}...")
        blob.upload_from_filename(compressed_file_path)
        logger.info(f"Uploaded {compressed_file_name}")

        return compressed_file_name

    def _download_file_from_gcp(self, file_name: str) -> str:
        credentials = service_account.Credentials.from_service_account_info(
            self.gcp_service_key,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        storage_client: Client = storage.Client(credentials=credentials)
        bucket: Bucket = storage_client.bucket(
            "oetc_files"
        )  # TODO: Make bucket name dynamic if necessary
        blob: Blob = bucket.blob(file_name)

        file_path: str = f"/tmp/{file_name}"

        logger.info(f"Downloading model {file_name}")
        blob.download_to_filename(file_path)
        logger.info(f"Model saved at {file_path}")

        logger.info(f"Decompressing .gz file {file_name}")
        decompressed_file_path = self._gzip_decompress(file_path)
        logger.info(f"Decompressed input file: {decompressed_file_path}")

        return decompressed_file_path
