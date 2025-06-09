from dataclasses import dataclass
from datetime import datetime, timedelta

import requests
from requests import RequestException


@dataclass
class OetcCredentials:
    email: str
    password: str

@dataclass
class OetcSettings:
    credentials: OetcCredentials
    authentication_server_url: str


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