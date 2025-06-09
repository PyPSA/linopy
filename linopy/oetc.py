from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Union
import json
import base64

import requests
from requests import RequestException


class ComputeProvider(str, Enum):
    GCP = "GCP"


@dataclass
class OetcCredentials:
    email: str
    password: str


@dataclass
class OetcSettings:
    credentials: OetcCredentials
    authentication_server_url: str
    compute_provider: ComputeProvider = ComputeProvider.GCP


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
            payload = {
                "email": self.settings.credentials.email,
                "password": self.settings.credentials.password
            }

            response = requests.post(
                f"{self.settings.authentication_server_url}/sign-in",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            response.raise_for_status()
            jwt_result = response.json()

            return AuthenticationResult(
                token=jwt_result["token"],
                token_type=jwt_result["token_type"],
                expires_in=jwt_result["expires_in"],
                authenticated_at=datetime.now()
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
            payload_part = token.split('.')[1]
            payload_part += '=' * (4 - len(payload_part) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_part)
            return json.loads(payload_bytes.decode('utf-8'))
        except (IndexError, json.JSONDecodeError, Exception) as e:
            raise Exception(f"Failed to decode JWT payload: {e}")

    def __get_cloud_provider_credentials(self) -> Union[GcpCredentials, None]:
        """
        Fetch cloud provider credentials based on the configured provider.

        Returns:
            Union[GcpCredentials, None]: The cloud provider credentials

        Raises:
            Exception: If the compute provider is not supported
        """
        if self.settings.compute_provider == ComputeProvider.GCP:
            return self.__get_gcp_credentials()
        else:
            raise Exception(f"Unsupported compute provider: {self.settings.compute_provider}")

    def __get_gcp_credentials(self) -> GcpCredentials:
        """
        Fetch GCP credentials for the authenticated user.

        Returns:
            GcpCredentials: The GCP credentials including project ID, service key, and bucket information

        Raises:
            Exception: If credentials fetching fails or response is invalid
        """
        try:
            payload = self._decode_jwt_payload(self.jwt.token)
            user_uuid = payload.get('sub')

            if not user_uuid:
                raise Exception("User UUID not found in JWT token")

            response = requests.get(
                f"{self.settings.authentication_server_url}/users/{user_uuid}/gcp-credentials",
                headers={
                    "Authorization": f"{self.jwt.token_type} {self.jwt.token}",
                    "Content-Type": "application/json"
                },
                timeout=30
            )

            response.raise_for_status()
            credentials_data = response.json()

            return GcpCredentials(
                gcp_project_id=credentials_data["gcp_project_id"],
                gcp_service_key=credentials_data["gcp_service_key"],
                input_bucket=credentials_data["input_bucket"],
                solution_bucket=credentials_data["solution_bucket"]
            )

        except RequestException as e:
            raise Exception(f"Failed to fetch GCP credentials: {e}")
        except KeyError as e:
            raise Exception(f"Invalid credentials response format: missing field {e}")
        except Exception as e:
            raise Exception(f"Error fetching GCP credentials: {e}")