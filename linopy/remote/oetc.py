import base64
import gzip
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import requests
from google.cloud import storage
from google.oauth2 import service_account
from requests import RequestException

import linopy

logger = logging.getLogger(__name__)


class ComputeProvider(str, Enum):
    GCP = "GCP"


@dataclass
class OetcCredentials:
    email: str
    password: str


@dataclass
class OetcSettings:
    credentials: OetcCredentials
    name: str
    authentication_server_url: str
    orchestrator_server_url: str
    compute_provider: ComputeProvider = ComputeProvider.GCP
    solver: str = "highs"
    solver_options: dict = field(default_factory=dict)
    cpu_cores: int = 2
    disk_space_gb: int = 10
    delete_worker_on_error: bool = False


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
    def __init__(self, settings: OetcSettings) -> None:
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
                "email": self.settings.credentials.email,
                "password": self.settings.credentials.password,
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

    def _submit_job_to_compute_service(self, input_file_name: str) -> str:
        """
        Submit a job to the compute service.

        Args:
            input_file_name: Name of the input file uploaded to GCP

        Returns:
            CreateComputeJobResult: The job creation result with UUID

        Raises:
            Exception: If job submission fails
        """
        try:
            logger.info("OETC - Submitting compute job...")
            payload = {
                "name": self.settings.name,
                "solver": self.settings.solver,
                "solver_options": self.settings.solver_options,
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
            try:
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

                job_result = JobResult(
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

    def solve_on_oetc(self, model):  # type: ignore
        """
        Solve a linopy model on the OET Cloud compute app.

        Parameters
        ----------
        model : linopy.model.Model

        Returns
        -------
        linopy.model.Model
            Solved model.

        Raises
        ------
            Exception: If solving fails at any stage
        """
        try:
            # Save model to temporary file and upload
            with tempfile.NamedTemporaryFile(prefix="linopy-", suffix=".nc") as fn:
                fn.file.close()
                model.to_netcdf(fn.name)
                input_file_name = self._upload_file_to_gcp(fn.name)

            # Submit job and wait for completion
            job_uuid = self._submit_job_to_compute_service(input_file_name)
            job_result = self.wait_and_get_job_data(job_uuid)

            # Download and load the solution
            if not job_result.output_files:
                raise Exception("No output files found in completed job")

            output_file_name = job_result.output_files[0]
            if isinstance(output_file_name, dict) and "name" in output_file_name:
                output_file_name = output_file_name["name"]

            solution_file_path = self._download_file_from_gcp(output_file_name)

            # Load the solved model
            solved_model = linopy.read_netcdf(solution_file_path)

            # Clean up downloaded file
            os.remove(solution_file_path)

            logger.info(
                f"OETC - Model solved successfully. Status: {solved_model.status}"
            )
            if hasattr(solved_model, "objective") and hasattr(
                solved_model.objective, "value"
            ):
                logger.info(
                    f"OETC - Objective value: {solved_model.objective.value:.2e}"
                )

            return solved_model

        except Exception as e:
            raise Exception(f"Error solving model on OETC: {e}")

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
