from __future__ import annotations

import base64
import contextlib
import gzip
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from linopy.model import Model

try:
    import requests
    from google.cloud import storage
    from google.oauth2 import service_account
    from requests import RequestException

    _oetc_deps_available = True
except ImportError:
    _oetc_deps_available = False

import warnings

import linopy
from linopy.sos_reformulation import (
    sos_reformulation_context,
    suppress_serialization_warning,
)

logger = logging.getLogger(__name__)


class ComputeProvider(str, Enum):
    GCP = "GCP"


@dataclass
class OetcCredentials:
    """
    .. deprecated::
        Pass ``email`` and ``password`` directly to :class:`OetcSettings`
        instead of wrapping them in ``OetcCredentials``. This class will be
        removed in a future release.
    """

    email: str
    password: str

    def __post_init__(self) -> None:
        warnings.warn(
            "`OetcCredentials` is deprecated; pass `email=` and `password=` "
            "directly to `OetcSettings`. `OetcCredentials` will be removed "
            "in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )


@dataclass
class OetcSettings:
    """
    Connection config for the OET Cloud (OETC) remote service.

    Carries the auth/orchestrator endpoints and the worker resource
    sizing. The solver is chosen per call — pass it to
    :meth:`Model.solve` or :meth:`Oetc.submit`:

    >>> m.solve("gurobi", remote=OetcSettings(...), Method=2)  # doctest: +SKIP
    """

    name: str
    authentication_server_url: str
    orchestrator_server_url: str
    email: str | None = None
    password: str | None = None
    credentials: OetcCredentials | None = None
    compute_provider: ComputeProvider = ComputeProvider.GCP
    solver: str | None = None
    solver_options: dict[str, Any] | None = None
    cpu_cores: int = 2
    disk_space_gb: int = 10
    delete_worker_on_error: bool = False

    def __post_init__(self) -> None:
        if self.credentials is not None:
            # `credentials=` warns from its own __post_init__; carry its
            # values over unless `email` / `password` were also explicitly
            # given (in which case the call site wins).
            if self.email is None:
                self.email = self.credentials.email
            if self.password is None:
                self.password = self.credentials.password
            self.credentials = None
        if not self.email or not self.password:
            raise ValueError("`OetcSettings` requires `email` and `password`.")
        if self.solver is not None or self.solver_options is not None:
            warnings.warn(
                "`OetcSettings.solver` and `OetcSettings.solver_options` are "
                "deprecated and consulted only by the deprecated `OetcHandler`. "
                "Pass the solver to `Model.solve(solver_name, remote=...)` or "
                "`Oetc.submit(model, solver_name, ...)`. These fields will be "
                "removed together with `OetcHandler`.",
                DeprecationWarning,
                stacklevel=2,
            )

    @classmethod
    def from_env(
        cls,
        *,
        email: str | None = None,
        password: str | None = None,
        name: str | None = None,
        authentication_server_url: str | None = None,
        orchestrator_server_url: str | None = None,
        cpu_cores: int | None = None,
        disk_space_gb: int | None = None,
        delete_worker_on_error: bool | None = None,
    ) -> OetcSettings:
        required_fields = {
            "email": ("OETC_EMAIL", email),
            "password": ("OETC_PASSWORD", password),
            "name": ("OETC_NAME", name),
            "authentication_server_url": ("OETC_AUTH_URL", authentication_server_url),
            "orchestrator_server_url": (
                "OETC_ORCHESTRATOR_URL",
                orchestrator_server_url,
            ),
        }

        resolved: dict[str, Any] = {}
        missing: list[str] = []

        for field_name, (env_var, kwarg) in required_fields.items():
            if kwarg is not None:
                resolved[field_name] = kwarg
            else:
                env_val = os.environ.get(env_var, "").strip()
                if env_val:
                    resolved[field_name] = env_val
                else:
                    missing.append(env_var)

        if missing:
            raise ValueError(
                f"Missing required OETC configuration: {', '.join(missing)}"
            )

        kwargs: dict[str, Any] = {
            "email": resolved["email"],
            "password": resolved["password"],
            "name": resolved["name"],
            "authentication_server_url": resolved["authentication_server_url"],
            "orchestrator_server_url": resolved["orchestrator_server_url"],
        }

        if cpu_cores is not None:
            kwargs["cpu_cores"] = cpu_cores
        elif (cpu_env := os.environ.get("OETC_CPU_CORES")) is not None:
            try:
                kwargs["cpu_cores"] = int(cpu_env)
            except ValueError as e:
                raise ValueError(
                    f"OETC_CPU_CORES is not a valid integer: {cpu_env}"
                ) from e

        if disk_space_gb is not None:
            kwargs["disk_space_gb"] = disk_space_gb
        elif (disk_env := os.environ.get("OETC_DISK_SPACE_GB")) is not None:
            try:
                kwargs["disk_space_gb"] = int(disk_env)
            except ValueError as e:
                raise ValueError(
                    f"OETC_DISK_SPACE_GB is not a valid integer: {disk_env}"
                ) from e

        if delete_worker_on_error is not None:
            kwargs["delete_worker_on_error"] = delete_worker_on_error
        elif (del_env := os.environ.get("OETC_DELETE_WORKER_ON_ERROR")) is not None:
            low = del_env.lower()
            if low in ("true", "1", "yes"):
                kwargs["delete_worker_on_error"] = True
            elif low in ("false", "0", "no"):
                kwargs["delete_worker_on_error"] = False
            else:
                raise ValueError(
                    f"OETC_DELETE_WORKER_ON_ERROR has invalid value: {del_env}"
                )

        return cls(**kwargs)


@dataclass
class GcpCredentials:
    gcp_project_id: str
    gcp_service_key: str
    input_bucket: str
    solution_bucket: str


@dataclass
class AuthenticationResult:
    token: str
    token_type: str
    expires_in: int  # value represented in seconds
    authenticated_at: datetime

    @property
    def expires_at(self) -> datetime:
        """Calculate when the token expires"""
        return self.authenticated_at + timedelta(seconds=self.expires_in)

    @property
    def is_expired(self) -> bool:
        """Check if the token has expired"""
        return datetime.now() >= self.expires_at


@dataclass
class JobResult:
    uuid: str
    status: str
    name: str | None = None
    owner: str | None = None
    solver: str | None = None
    duration_in_seconds: int | None = None
    solving_duration_in_seconds: int | None = None
    input_files: list | None = None
    output_files: list | None = None
    created_at: str | None = None


class OetcHandler:
    """
    .. deprecated::
        Use :class:`~linopy.remote.Oetc` or :meth:`Model.solve(remote=OetcSettings(...))
        <linopy.model.Model.solve>` instead. This class will be removed in a
        future release. The new :class:`Oetc` class owns the public lifecycle
        (``submit`` / ``status`` / ``collect`` / ``solve``); ``OetcHandler``
        remains only for back-compat with code that holds a long-lived
        handler instance.
    """

    def __init__(self, settings: OetcSettings, *, _internal: bool = False) -> None:
        if not _oetc_deps_available:
            raise ImportError(
                "The 'google-cloud-storage' and 'requests' packages are required "
                "for OetcHandler. Install them with: pip install linopy[oetc]"
            )
        if not _internal:
            warnings.warn(
                "`OetcHandler` is deprecated; use `Oetc(settings)` from "
                "`linopy.remote` or `Model.solve(remote=OetcSettings(...))`. "
                "`OetcHandler` will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.settings = settings
        self.jwt = self.__sign_in()
        self.cloud_provider_credentials = self.__get_cloud_provider_credentials()

    def __sign_in(self) -> AuthenticationResult:
        """
        Authenticate with the server and return the authentication result.

        Returns:
            AuthenticationResult: The complete authentication result including token and expiration info

        Raises:
            Exception: If authentication fails or response is invalid
        """
        try:
            logger.info("OETC - Signing in...")
            payload = {
                "email": self.settings.email,
                "password": self.settings.password,
            }

            response = requests.post(
                f"{self.settings.authentication_server_url}/sign-in",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            response.raise_for_status()
            jwt_result = response.json()

            logger.info("OETC - Signed in")

            return AuthenticationResult(
                token=jwt_result["token"],
                token_type=jwt_result["token_type"],
                expires_in=jwt_result["expires_in"],
                authenticated_at=datetime.now(),
            )

        except RequestException as e:
            raise Exception(f"Authentication request failed: {e}")
        except KeyError as e:
            raise Exception(f"Invalid response format: missing field {e}")
        except Exception as e:
            raise Exception(f"Authentication error: {e}")

    def _decode_jwt_payload(self, token: str) -> dict:
        """
        Decode JWT payload without verification to extract user information.

        Args:
            token: The JWT token

        Returns:
            dict: The decoded payload containing user information

        Raises:
            Exception: If token cannot be decoded
        """
        try:
            payload_part = token.split(".")[1]
            payload_part += "=" * (4 - len(payload_part) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_part)
            return json.loads(payload_bytes.decode("utf-8"))
        except (IndexError, json.JSONDecodeError, Exception) as e:
            raise Exception(f"Failed to decode JWT payload: {e}")

    def __get_cloud_provider_credentials(self) -> GcpCredentials:
        """
        Fetch cloud provider credentials based on the configured provider.

        Returns:
            GcpCredentials: The cloud provider credentials

        Raises:
            Exception: If the compute provider is not supported
        """
        if self.settings.compute_provider == ComputeProvider.GCP:
            return self.__get_gcp_credentials()
        else:
            raise Exception(
                f"Unsupported compute provider: {self.settings.compute_provider}"
            )

    def __get_gcp_credentials(self) -> GcpCredentials:
        """
        Fetch GCP credentials for the authenticated user.

        Returns:
            GcpCredentials: The GCP credentials including project ID, service key, and bucket information

        Raises:
            Exception: If credentials fetching fails or response is invalid
        """
        try:
            logger.info("OETC - Fetching user GCP credentials...")
            payload = self._decode_jwt_payload(self.jwt.token)
            user_uuid = payload.get("sub")

            if not user_uuid:
                raise Exception("User UUID not found in JWT token")

            response = requests.get(
                f"{self.settings.authentication_server_url}/users/{user_uuid}/gcp-credentials",
                headers={
                    "Authorization": f"{self.jwt.token_type} {self.jwt.token}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )

            response.raise_for_status()
            credentials_data = response.json()

            logger.info("OETC - Fetched user GCP credentials")

            return GcpCredentials(
                gcp_project_id=credentials_data["gcp_project_id"],
                gcp_service_key=credentials_data["gcp_service_key"],
                input_bucket=credentials_data["input_bucket"],
                solution_bucket=credentials_data["solution_bucket"],
            )

        except RequestException as e:
            raise Exception(f"Failed to fetch GCP credentials: {e}")
        except KeyError as e:
            raise Exception(f"Invalid credentials response format: missing field {e}")
        except Exception as e:
            raise Exception(f"Error fetching GCP credentials: {e}")

    def _submit_job_to_compute_service(
        self, input_file_name: str, solver: str, solver_options: dict[str, Any]
    ) -> str:
        """
        Submit a job to the compute service.

        Args:
            input_file_name: Name of the input file uploaded to GCP
            solver: Solver name to use
            solver_options: Solver options dict

        Returns:
            CreateComputeJobResult: The job creation result with UUID

        Raises:
            Exception: If job submission fails
        """
        try:
            logger.info("OETC - Submitting compute job...")
            payload = {
                "name": self.settings.name,
                "solver": solver,
                "solver_options": solver_options,
                "provider": self.settings.compute_provider.value,
                "cpu_cores": self.settings.cpu_cores,
                "disk_space_gb": self.settings.disk_space_gb,
                "input_file_name": input_file_name,
                "delete_worker_on_error": self.settings.delete_worker_on_error,
            }

            response = requests.post(
                f"{self.settings.orchestrator_server_url}/compute-job/create",
                json=payload,
                headers={
                    "Authorization": f"{self.jwt.token_type} {self.jwt.token}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )

            response.raise_for_status()
            job_result = response.json()

            logger.info(f"OETC - Compute job {job_result['uuid']} started")

            return job_result["uuid"]

        except RequestException as e:
            raise Exception(f"Failed to submit job to compute service: {e}")
        except KeyError as e:
            raise Exception(
                f"Invalid job submission response format: missing field {e}"
            )
        except Exception as e:
            raise Exception(f"Error submitting job to compute service: {e}")

    def _get_job_logs(self, job_uuid: str) -> str:
        """
        Fetch logs for a compute job.

        Args:
            job_uuid: UUID of the job to fetch logs for

        Returns:
            str: The job logs content as a string

        Raises:
            Exception: If fetching logs fails
        """
        try:
            logger.info(f"OETC - Fetching logs for job {job_uuid}...")

            response = requests.get(
                f"{self.settings.orchestrator_server_url}/compute-job/{job_uuid}/get-logs",
                headers={
                    "Authorization": f"{self.jwt.token_type} {self.jwt.token}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )

            response.raise_for_status()
            logs_data = response.json()

            # Extract content from the response structure
            logs_content = logs_data.get("content", "")

            logger.info(f"OETC - Successfully fetched logs for job {job_uuid}")
            return logs_content

        except RequestException as e:
            logger.warning(f"OETC - Failed to fetch logs for job {job_uuid}: {e}")
            return f"[Unable to fetch logs: {e}]"
        except Exception as e:
            logger.warning(f"OETC - Error fetching logs for job {job_uuid}: {e}")
            return f"[Error fetching logs: {e}]"

    def _get_job(self, job_uuid: str) -> JobResult:
        """
        Fetch the current job record in a single request (no polling).

        Raises ``RequestException`` on a failed request and ``KeyError``
        if the response is missing required fields.
        """
        response = requests.get(
            f"{self.settings.orchestrator_server_url}/compute-job/{job_uuid}",
            headers={
                "Authorization": f"{self.jwt.token_type} {self.jwt.token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()
        job_data_dict = response.json()
        return JobResult(
            uuid=job_data_dict["uuid"],
            status=job_data_dict["status"],
            name=job_data_dict.get("name"),
            owner=job_data_dict.get("owner"),
            solver=job_data_dict.get("solver"),
            duration_in_seconds=job_data_dict.get("duration_in_seconds"),
            solving_duration_in_seconds=job_data_dict.get(
                "solving_duration_in_seconds"
            ),
            input_files=job_data_dict.get("input_files", []),
            output_files=job_data_dict.get("output_files", []),
            created_at=job_data_dict.get("created_at"),
        )

    def wait_and_get_job_data(
        self,
        job_uuid: str,
        initial_poll_interval: int = 30,
        max_poll_interval: int = 300,
    ) -> JobResult:
        """
        Wait for job completion and get job data by polling the orchestrator service.

        This method will poll indefinitely until the job finishes (FINISHED) or
        encounters an error (SETUP_ERROR, RUNTIME_ERROR).

        Args:
            job_uuid: UUID of the job to wait for
            initial_poll_interval: Initial polling interval in seconds (default: 30)
            max_poll_interval: Maximum polling interval in seconds (default: 300)

        Returns:
            JobResult: The job result when complete

        Raises:
            Exception: If job encounters errors or network requests consistently fail
        """
        poll_interval = initial_poll_interval
        consecutive_failures = 0
        max_network_retries = 10

        logger.info(f"OETC - Waiting for job {job_uuid} to complete...")

        while True:
            if self.jwt.is_expired:
                logger.info("OETC - Auth token expired; re-authenticating.")
                self.jwt = self.__sign_in()
            try:
                job_result = self._get_job(job_uuid)

                consecutive_failures = 0

                if job_result.status == "FINISHED":
                    logger.info(f"OETC - Job {job_uuid} completed successfully!")
                    if not job_result.output_files:
                        logger.warning(
                            "OETC - Warning: Job completed but no output files found"
                        )
                    return job_result

                elif job_result.status == "SETUP_ERROR":
                    error_msg = f"Job failed during setup phase (status: {job_result.status}). Please check the OETC logs for details."
                    logger.error(f"OETC Error: {error_msg}")
                    raise Exception(error_msg)

                elif job_result.status == "RUNTIME_ERROR":
                    # Fetch and display logs
                    logs = self._get_job_logs(job_uuid)
                    logger.error(f"OETC - Job {job_uuid} logs:\n{logs}")

                    error_msg = (
                        f"Job failed during execution (status: {job_result.status}).\n"
                        f"Logs:\n{logs}"
                    )
                    logger.error(f"OETC Error: {error_msg}")
                    raise Exception(error_msg)

                elif job_result.status in ["PENDING", "STARTING", "RUNNING"]:
                    status_msg = f"Job {job_uuid} status: {job_result.status}"
                    if job_result.duration_in_seconds:
                        status_msg += (
                            f" (running for {job_result.duration_in_seconds}s)"
                        )
                    status_msg += f", checking again in {poll_interval} seconds..."
                    logger.info(f"OETC - {status_msg}")

                    time.sleep(poll_interval)

                    # Exponential backoff for polling interval, capped at max_poll_interval
                    poll_interval = min(int(poll_interval * 1.5), max_poll_interval)

                else:
                    # Unknown status
                    error_msg = f"Unknown job status: {job_result.status}. Please check the OETC logs for details."
                    logger.error(f"OETC Error: {error_msg}")
                    raise Exception(error_msg)

            except RequestException as e:
                consecutive_failures += 1

                if consecutive_failures >= max_network_retries:
                    raise Exception(
                        f"Failed to get job status after {max_network_retries} network retries: {e}"
                    )

                # Wait before retrying network request
                retry_wait = min(consecutive_failures * 10, 60)
                logger.error(
                    f"OETC - Network error getting job status (attempt {consecutive_failures}/{max_network_retries}), "
                    f"retrying in {retry_wait} seconds: {e}"
                )
                time.sleep(retry_wait)

            except KeyError as e:
                raise Exception(
                    f"Invalid job status response format: missing field {e}"
                )
            except Exception as e:
                if "status:" in str(e) or "OETC logs" in str(e):
                    raise
                else:
                    raise Exception(f"Error getting job status: {e}")

    def _gzip_decompress(self, input_path: str) -> str:
        """
        Decompress a gzip-compressed file.

        Args:
            input_path: Path to the compressed file

        Returns:
            str: Path to the decompressed file

        Raises:
            Exception: If decompression fails
        """
        try:
            logger.info(f"OETC - Decompressing file: {input_path}")
            output_path = input_path[:-3]
            chunk_size = 1024 * 1024

            with gzip.open(input_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    while True:
                        chunk = f_in.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)

            logger.info(f"OETC - File decompressed successfully: {output_path}")
            return output_path
        except Exception as e:
            raise Exception(f"Failed to decompress file: {e}")

    def _download_file_from_gcp(self, file_name: str) -> str:
        """
        Download a file from GCP storage bucket.

        Args:
            file_name: Name of the file to download from the solution bucket

        Returns:
            str: Path to the downloaded and decompressed file

        Raises:
            Exception: If download or decompression fails
        """
        try:
            logger.info(f"OETC - Downloading file from GCP: {file_name}")

            # Create GCP credentials from service key
            service_key_dict = json.loads(
                self.cloud_provider_credentials.gcp_service_key
            )
            credentials = service_account.Credentials.from_service_account_info(
                service_key_dict,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            # Download from GCP solution bucket
            storage_client = storage.Client(
                credentials=credentials,
                project=self.cloud_provider_credentials.gcp_project_id,
            )
            bucket = storage_client.bucket(
                self.cloud_provider_credentials.solution_bucket
            )
            blob = bucket.blob(file_name)

            # Create temporary file for download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as temp_file:
                compressed_file_path = temp_file.name

            blob.download_to_filename(compressed_file_path)
            logger.info(f"OETC - File downloaded from GCP successfully: {file_name}")

            # Decompress the downloaded file
            decompressed_file_path = self._gzip_decompress(compressed_file_path)

            # Clean up compressed file
            os.remove(compressed_file_path)

            return decompressed_file_path

        except Exception as e:
            raise Exception(f"Failed to download file from GCP: {e}")

    def solve_on_oetc(
        self,
        model: Model,
        solver_name: str | None = None,
        *,
        reformulate_sos: bool | Literal["auto"] = False,
        **solver_options: Any,
    ) -> Model:
        """
        Solve a linopy model on the OET Cloud compute app.

        .. deprecated::
            Use :class:`Oetc` or
            :meth:`Model.solve(remote=OetcSettings(...)) <linopy.model.Model.solve>`.

        Parameters
        ----------
        model : linopy.model.Model
        solver_name : str, optional
            Override the solver from settings.
        reformulate_sos : bool | "auto", optional
            See :meth:`linopy.model.Model.solve`.
        **solver_options
            Override/extend solver_options from settings.

        Returns
        -------
        linopy.model.Model
            Solved model.
        """
        # Delegates to ``Oetc`` so the upload→submit→poll→download
        # orchestration lives in one place.
        effective_solver = solver_name or self.settings.solver or "highs"
        merged_solver_options = {
            **(self.settings.solver_options or {}),
            **solver_options,
        }

        oetc = Oetc(settings=self.settings)
        oetc._handler = self  # reuse this handler so auth is not refetched
        try:
            with sos_reformulation_context(
                model, effective_solver, reformulate_sos
            ) as applied:
                with suppress_serialization_warning(active=applied):
                    job_uuid = oetc.submit(
                        model, effective_solver, **merged_solver_options
                    )
                solved_model = oetc.collect(job_uuid)
        except Exception as e:
            raise Exception(f"Error solving model on OETC: {e}") from e

        logger.info(f"OETC - Model solved successfully. Status: {solved_model.status}")
        if solved_model.objective.value is not None:
            logger.info(f"OETC - Objective value: {solved_model.objective.value:.2e}")
        return solved_model

    def _gzip_compress(self, source_path: str) -> str:
        """
        Compress a file using gzip compression.

        Args:
            source_path: Path to the source file to compress

        Returns:
            str: Path to the compressed file

        Raises:
            Exception: If compression fails
        """
        try:
            logger.info(f"OETC - Compressing file: {source_path}")
            output_path = source_path + ".gz"
            chunk_size = 1024 * 1024

            with open(source_path, "rb") as f_in:
                with gzip.open(output_path, "wb", compresslevel=9) as f_out:
                    while True:
                        chunk = f_in.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)

            logger.info(f"OETC - File compressed successfully: {output_path}")
            return output_path
        except Exception as e:
            raise Exception(f"Failed to compress file: {e}")

    def _upload_file_to_gcp(self, file_path: str) -> str:
        """
        Upload a file to GCP storage bucket after compression.

        Args:
            file_path: Path to the file to upload

        Returns:
            str: Name of the uploaded file in the bucket

        Raises:
            Exception: If upload fails
        """
        try:
            compressed_file_path = self._gzip_compress(file_path)
            compressed_file_name = os.path.basename(compressed_file_path)

            logger.info(f"OETC - Uploading file to GCP: {compressed_file_name}")

            # Create GCP credentials from service key
            service_key_dict = json.loads(
                self.cloud_provider_credentials.gcp_service_key
            )
            credentials = service_account.Credentials.from_service_account_info(
                service_key_dict,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            # Upload to GCP bucket
            storage_client = storage.Client(
                credentials=credentials,
                project=self.cloud_provider_credentials.gcp_project_id,
            )
            bucket = storage_client.bucket(self.cloud_provider_credentials.input_bucket)
            blob = bucket.blob(compressed_file_name)

            blob.upload_from_filename(compressed_file_path)

            logger.info(
                f"OETC - File uploaded to GCP successfully: {compressed_file_name}"
            )

            # Clean up compressed file
            os.remove(compressed_file_path)

            return compressed_file_name

        except Exception as e:
            raise Exception(f"Failed to upload file to GCP: {e}")


@dataclass
class Oetc:
    """
    A session with the OET Cloud (OETC) managed compute service.

    This is a standalone class — *not* a :class:`linopy.solvers.Solver`
    subclass. An ``Oetc`` instance is a *session*, not a job: it holds an
    auth token — not a persistent connection — and can submit and collect
    any number of jobs over HTTPS.

    A job is identified solely by the uuid string returned from
    :meth:`submit`. Because the handle is just a string, the lifecycle is
    async-friendly — submit many models, hold their uuids, and
    :meth:`collect` each when convenient, even from a different process
    (a fresh ``Oetc(settings)`` re-authenticates and collects by uuid).

    Parameters
    ----------
    settings : OetcSettings
        Auth + orchestrator config (where to talk to).
    """

    settings: OetcSettings

    _handler: OetcHandler | None = field(init=False, default=None, repr=False)

    @classmethod
    def is_available(cls) -> bool:
        """Return True iff the OETC network deps are importable."""
        return _oetc_deps_available

    def _session(self) -> OetcHandler:
        """
        Return the authenticated handler.

        Builds it on first use, and rebuilds it (re-authenticating) once
        the previous auth token has expired — so a long-lived ``Oetc``
        keeps working across the token lifetime.
        """
        if self._handler is None or self._handler.jwt.is_expired:
            self._handler = OetcHandler(self.settings, _internal=True)
        return self._handler

    def submit(self, model: Model, solver_name: str, **options: Any) -> str:
        """
        Serialize and upload the model, submit the job, and return its uuid.

        The uuid is the only handle a job needs — persist it and
        :meth:`collect` later, from this or any other process.
        """
        handler = self._session()
        with tempfile.NamedTemporaryFile(prefix="linopy-", suffix=".nc") as fn:
            fn.file.close()
            model.to_netcdf(fn.name)
            input_file_name = handler._upload_file_to_gcp(fn.name)
        return handler._submit_job_to_compute_service(
            input_file_name, solver_name, dict(options)
        )

    def status(self, job_uuid: str) -> str:
        """Return the current job status in a single, non-blocking request."""
        return self._session()._get_job(job_uuid).status

    def collect(self, job_uuid: str) -> Model:
        """
        Block until the job finishes, download, and return the solved model.

        Needs only the uuid and the session, so it can run in a
        different process than the one that called :meth:`submit`.
        """
        handler = self._session()
        job_result = handler.wait_and_get_job_data(job_uuid)
        if not job_result.output_files:
            raise Exception("No output files found in completed job")
        output_file_name = job_result.output_files[0]
        if isinstance(output_file_name, dict) and "name" in output_file_name:
            output_file_name = output_file_name["name"]

        solution_file_path = handler._download_file_from_gcp(output_file_name)
        try:
            return linopy.read_netcdf(solution_file_path)
        finally:
            with contextlib.suppress(OSError):
                os.remove(solution_file_path)

    def solve(self, model: Model, solver_name: str, **options: Any) -> Model:
        """Submit the model and block until the solved model is back."""
        from linopy.remote._common import _validate_inner_solver

        _validate_inner_solver(solver_name, model)
        job_uuid = self.submit(model, solver_name, **options)
        return self.collect(job_uuid)
