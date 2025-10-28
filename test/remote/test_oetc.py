import base64
import json
from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest
import requests
from requests import RequestException

from linopy.remote.oetc import (
    AuthenticationResult,
    ComputeProvider,
    GcpCredentials,
    JobResult,
    OetcCredentials,
    OetcHandler,
    OetcSettings,
)


@pytest.fixture
def sample_jwt_token() -> str:
    """Create a sample JWT token with test payload"""
    payload = {
        "iss": "OETC",
        "sub": "user-uuid-123",
        "exp": 1640995200,
        "jti": "jwt-id-456",
        "email": "test@example.com",
        "firstname": "Test",
        "lastname": "User",
    }

    # Create a simple JWT-like token (header.payload.signature)
    header = (
        base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        .decode()
        .rstrip("=")
    )
    payload_encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    )
    signature = "fake_signature"

    return f"{header}.{payload_encoded}.{signature}"


@pytest.fixture
def mock_gcp_credentials_response() -> dict:
    """Create a mock GCP credentials response"""
    return {
        "gcp_project_id": "test-project-123",
        "gcp_service_key": "test-service-key-content",
        "input_bucket": "test-input-bucket",
        "solution_bucket": "test-solution-bucket",
    }


@pytest.fixture
def mock_settings() -> OetcSettings:
    """Create mock settings for testing"""
    credentials = OetcCredentials(email="test@example.com", password="test_password")
    return OetcSettings(
        credentials=credentials,
        name="Test Job",
        authentication_server_url="https://auth.example.com",
        orchestrator_server_url="https://orchestrator.example.com",
        compute_provider=ComputeProvider.GCP,
    )


class TestOetcHandler:
    @pytest.fixture
    def mock_jwt_response(self) -> dict:
        """Create a mock JWT response"""
        return {
            "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

    @patch("linopy.remote.oetc.requests.post")
    @patch("linopy.remote.oetc.requests.get")
    @patch("linopy.remote.oetc.datetime")
    def test_successful_authentication(
        self,
        mock_datetime: Mock,
        mock_get: Mock,
        mock_post: Mock,
        mock_settings: OetcSettings,
        mock_jwt_response: dict,
        mock_gcp_credentials_response: dict,
        sample_jwt_token: str,
    ) -> None:
        """Test successful authentication flow"""
        # Setup mocks
        fixed_time = datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.now.return_value = fixed_time

        # Mock authentication response
        mock_auth_response = Mock()
        mock_jwt_response["token"] = sample_jwt_token
        mock_auth_response.json.return_value = mock_jwt_response
        mock_auth_response.raise_for_status.return_value = None
        mock_post.return_value = mock_auth_response

        # Mock GCP credentials response
        mock_gcp_response = Mock()
        mock_gcp_response.json.return_value = mock_gcp_credentials_response
        mock_gcp_response.raise_for_status.return_value = None
        mock_get.return_value = mock_gcp_response

        # Execute
        handler = OetcHandler(mock_settings)

        # Verify authentication request
        mock_post.assert_called_once_with(
            "https://auth.example.com/sign-in",
            json={"email": "test@example.com", "password": "test_password"},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        # Verify GCP credentials request
        mock_get.assert_called_once_with(
            "https://auth.example.com/users/user-uuid-123/gcp-credentials",
            headers={
                "Authorization": f"Bearer {sample_jwt_token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )

        # Verify AuthenticationResult
        assert isinstance(handler.jwt, AuthenticationResult)
        assert handler.jwt.token == sample_jwt_token
        assert handler.jwt.token_type == "Bearer"
        assert handler.jwt.expires_in == 3600
        assert handler.jwt.authenticated_at == fixed_time

        # Verify GcpCredentials
        assert isinstance(handler.cloud_provider_credentials, GcpCredentials)
        assert handler.cloud_provider_credentials.gcp_project_id == "test-project-123"
        assert (
            handler.cloud_provider_credentials.gcp_service_key
            == "test-service-key-content"
        )
        assert handler.cloud_provider_credentials.input_bucket == "test-input-bucket"
        assert (
            handler.cloud_provider_credentials.solution_bucket == "test-solution-bucket"
        )

    @patch("linopy.remote.oetc.requests.post")
    def test_authentication_http_error(
        self, mock_post: Mock, mock_settings: OetcSettings
    ) -> None:
        """Test authentication failure with HTTP error"""
        # Setup mock to raise HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "401 Unauthorized"
        )
        mock_post.return_value = mock_response

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            OetcHandler(mock_settings)

        assert "Authentication request failed" in str(exc_info.value)


class TestJwtDecoding:
    @pytest.fixture
    def handler_with_mocked_auth(self) -> OetcHandler:
        """Create handler with mocked authentication for testing JWT decoding"""
        with (
            patch("linopy.remote.oetc.requests.post"),
            patch("linopy.remote.oetc.requests.get"),
        ):
            credentials = OetcCredentials(
                email="test@example.com", password="test_password"
            )
            settings = OetcSettings(
                credentials=credentials,
                name="Test Job",
                authentication_server_url="https://auth.example.com",
                orchestrator_server_url="https://orchestrator.example.com",
                compute_provider=ComputeProvider.GCP,
            )

            # Mock the authentication and credentials fetching
            mock_auth_result = AuthenticationResult(
                token="fake.token.here",
                token_type="Bearer",
                expires_in=3600,
                authenticated_at=datetime.now(),
            )

            handler = OetcHandler.__new__(OetcHandler)
            handler.settings = settings
            handler.jwt = mock_auth_result
            handler.cloud_provider_credentials = None  # type: ignore

            return handler

    def test_decode_jwt_payload_success(
        self, handler_with_mocked_auth: OetcHandler, sample_jwt_token: str
    ) -> None:
        """Test successful JWT payload decoding"""
        result = handler_with_mocked_auth._decode_jwt_payload(sample_jwt_token)

        assert result["iss"] == "OETC"
        assert result["sub"] == "user-uuid-123"
        assert result["email"] == "test@example.com"
        assert result["firstname"] == "Test"
        assert result["lastname"] == "User"

    def test_decode_jwt_payload_invalid_token(
        self, handler_with_mocked_auth: OetcHandler
    ) -> None:
        """Test JWT payload decoding with invalid token"""
        with pytest.raises(Exception) as exc_info:
            handler_with_mocked_auth._decode_jwt_payload("invalid.token")

        assert "Failed to decode JWT payload" in str(exc_info.value)

    def test_decode_jwt_payload_malformed_token(
        self, handler_with_mocked_auth: OetcHandler
    ) -> None:
        """Test JWT payload decoding with malformed token"""
        with pytest.raises(Exception) as exc_info:
            handler_with_mocked_auth._decode_jwt_payload("not_a_jwt_token")

        assert "Failed to decode JWT payload" in str(exc_info.value)


class TestCloudProviderCredentials:
    @pytest.fixture
    def handler_with_mocked_auth(self, sample_jwt_token: str) -> OetcHandler:
        """Create handler with mocked authentication for testing credentials fetching"""
        credentials = OetcCredentials(
            email="test@example.com", password="test_password"
        )
        settings = OetcSettings(
            credentials=credentials,
            name="Test Job",
            authentication_server_url="https://auth.example.com",
            orchestrator_server_url="https://orchestrator.example.com",
            compute_provider=ComputeProvider.GCP,
        )

        # Mock the authentication result
        mock_auth_result = AuthenticationResult(
            token=sample_jwt_token,
            token_type="Bearer",
            expires_in=3600,
            authenticated_at=datetime.now(),
        )

        handler = OetcHandler.__new__(OetcHandler)
        handler.settings = settings
        handler.jwt = mock_auth_result

        return handler

    @patch("linopy.remote.oetc.requests.get")
    def test_get_gcp_credentials_success(
        self,
        mock_get: Mock,
        handler_with_mocked_auth: OetcHandler,
        mock_gcp_credentials_response: dict,
    ) -> None:
        """Test successful GCP credentials fetching"""
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = mock_gcp_credentials_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Execute
        result = handler_with_mocked_auth._OetcHandler__get_gcp_credentials()  # type: ignore[attr-defined]

        # Verify request
        mock_get.assert_called_once_with(
            "https://auth.example.com/users/user-uuid-123/gcp-credentials",
            headers={
                "Authorization": f"Bearer {handler_with_mocked_auth.jwt.token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )

        # Verify result
        assert isinstance(result, GcpCredentials)
        assert result.gcp_project_id == "test-project-123"
        assert result.gcp_service_key == "test-service-key-content"
        assert result.input_bucket == "test-input-bucket"
        assert result.solution_bucket == "test-solution-bucket"

    @patch("linopy.remote.oetc.requests.get")
    def test_get_gcp_credentials_http_error(
        self, mock_get: Mock, handler_with_mocked_auth: OetcHandler
    ) -> None:
        """Test GCP credentials fetching with HTTP error"""
        # Setup mock to raise HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        mock_get.return_value = mock_response

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_mocked_auth._OetcHandler__get_gcp_credentials()  # type: ignore[attr-defined]

        assert "Failed to fetch GCP credentials" in str(exc_info.value)

    @patch("linopy.remote.oetc.requests.get")
    def test_get_gcp_credentials_missing_field(
        self, mock_get: Mock, handler_with_mocked_auth: OetcHandler
    ) -> None:
        """Test GCP credentials fetching with missing response field"""
        # Setup mock with invalid response
        mock_response = Mock()
        mock_response.json.return_value = {
            "gcp_project_id": "test-project-123",
            "gcp_service_key": "test-service-key-content",
            "input_bucket": "test-input-bucket",
            # Missing "solution_bucket" field
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_mocked_auth._OetcHandler__get_gcp_credentials()  # type: ignore[attr-defined]

        assert (
            "Invalid credentials response format: missing field 'solution_bucket'"
            in str(exc_info.value)
        )

    def test_get_cloud_provider_credentials_unsupported_provider(
        self, handler_with_mocked_auth: OetcHandler
    ) -> None:
        """Test cloud provider credentials with unsupported provider"""
        # Change to unsupported provider
        handler_with_mocked_auth.settings.compute_provider = "AWS"  # type: ignore  # Not in enum, but for testing

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_mocked_auth._OetcHandler__get_cloud_provider_credentials()  # type: ignore[attr-defined]

        assert "Unsupported compute provider: AWS" in str(exc_info.value)

    def test_get_gcp_credentials_no_user_uuid_in_token(
        self, handler_with_mocked_auth: OetcHandler
    ) -> None:
        """Test GCP credentials fetching when JWT token has no user UUID"""
        # Create token without 'sub' field
        payload = {"iss": "OETC", "email": "test@example.com"}
        payload_encoded = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )
        token_without_sub = f"header.{payload_encoded}.signature"

        handler_with_mocked_auth.jwt.token = token_without_sub

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_mocked_auth._OetcHandler__get_gcp_credentials()  # type: ignore[attr-defined]

        assert "User UUID not found in JWT token" in str(exc_info.value)


class TestGcpCredentials:
    def test_gcp_credentials_creation(self) -> None:
        """Test GcpCredentials dataclass creation"""
        credentials = GcpCredentials(
            gcp_project_id="test-project-123",
            gcp_service_key="test-service-key-content",
            input_bucket="test-input-bucket",
            solution_bucket="test-solution-bucket",
        )

        assert credentials.gcp_project_id == "test-project-123"
        assert credentials.gcp_service_key == "test-service-key-content"
        assert credentials.input_bucket == "test-input-bucket"
        assert credentials.solution_bucket == "test-solution-bucket"


class TestComputeProvider:
    def test_compute_provider_enum(self) -> None:
        """Test ComputeProvider enum values"""
        assert ComputeProvider.GCP == "GCP"
        assert ComputeProvider.GCP.value == "GCP"

    @patch("linopy.remote.oetc.requests.post")
    def test_authentication_network_error(
        self, mock_post: Mock, mock_settings: OetcSettings
    ) -> None:
        """Test authentication failure with network error"""
        # Setup mock to raise network error
        mock_post.side_effect = RequestException("Connection timeout")

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            OetcHandler(mock_settings)

        assert "Authentication request failed" in str(exc_info.value)

    @patch("linopy.remote.oetc.requests.post")
    def test_authentication_invalid_response_missing_token(
        self, mock_post: Mock, mock_settings: OetcSettings
    ) -> None:
        """Test authentication failure with missing token in response"""
        # Setup mock with invalid response
        mock_response = Mock()
        mock_response.json.return_value = {
            "token_type": "Bearer",
            "expires_in": 3600,
            # Missing "token" field
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            OetcHandler(mock_settings)

        assert "Invalid response format: missing field 'token'" in str(exc_info.value)

    @patch("linopy.remote.oetc.requests.post")
    def test_authentication_invalid_response_missing_expires_in(
        self, mock_post: Mock, mock_settings: OetcSettings
    ) -> None:
        """Test authentication failure with missing expires_in in response"""
        # Setup mock with invalid response
        mock_response = Mock()
        mock_response.json.return_value = {
            "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "Bearer",
            # Missing "expires_in" field
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            OetcHandler(mock_settings)

        assert "Invalid response format: missing field 'expires_in'" in str(
            exc_info.value
        )

    @patch("linopy.remote.oetc.requests.post")
    def test_authentication_timeout_error(
        self, mock_post: Mock, mock_settings: OetcSettings
    ) -> None:
        """Test authentication failure with timeout"""
        # Setup mock to raise timeout error
        mock_post.side_effect = requests.Timeout("Request timeout")

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            OetcHandler(mock_settings)

        assert "Authentication request failed" in str(exc_info.value)


class TestAuthenticationResult:
    @pytest.fixture
    def auth_result(self) -> AuthenticationResult:
        """Create an AuthenticationResult for testing"""
        return AuthenticationResult(
            token="test_token",
            token_type="Bearer",
            expires_in=3600,  # 1 hour
            authenticated_at=datetime(2024, 1, 15, 12, 0, 0),
        )

    def test_expires_at_calculation(self, auth_result: AuthenticationResult) -> None:
        """Test that expires_at correctly calculates expiration time"""
        expected_expiry = datetime(2024, 1, 15, 13, 0, 0)  # 1 hour later
        assert auth_result.expires_at == expected_expiry

    @patch("linopy.remote.oetc.datetime")
    def test_is_expired_false_when_not_expired(
        self, mock_datetime: Mock, auth_result: AuthenticationResult
    ) -> None:
        """Test is_expired returns False when token is still valid"""
        # Set current time to before expiration
        mock_datetime.now.return_value = datetime(2024, 1, 15, 12, 30, 0)

        assert auth_result.is_expired is False

    @patch("linopy.remote.oetc.datetime")
    def test_is_expired_true_when_expired(
        self, mock_datetime: Mock, auth_result: AuthenticationResult
    ) -> None:
        """Test is_expired returns True when token has expired"""
        # Set current time to after expiration
        mock_datetime.now.return_value = datetime(2024, 1, 15, 14, 0, 0)

        assert auth_result.is_expired is True

    @patch("linopy.remote.oetc.datetime")
    def test_is_expired_true_when_exactly_expired(
        self, mock_datetime: Mock, auth_result: AuthenticationResult
    ) -> None:
        """Test is_expired returns True when token expires exactly now"""
        # Set current time to exact expiration time
        mock_datetime.now.return_value = datetime(2024, 1, 15, 13, 0, 0)

        assert auth_result.is_expired is True


class TestGetJobLogs:
    @pytest.fixture
    def handler_with_auth_setup(self, sample_jwt_token: str) -> OetcHandler:
        """Create handler with authentication setup for testing log fetching"""
        credentials = OetcCredentials(
            email="test@example.com", password="test_password"
        )
        settings = OetcSettings(
            credentials=credentials,
            name="Test Job",
            authentication_server_url="https://auth.example.com",
            orchestrator_server_url="https://orchestrator.example.com",
            compute_provider=ComputeProvider.GCP,
        )

        mock_auth_result = AuthenticationResult(
            token=sample_jwt_token,
            token_type="Bearer",
            expires_in=3600,
            authenticated_at=datetime.now(),
        )

        handler = OetcHandler.__new__(OetcHandler)
        handler.settings = settings
        handler.jwt = mock_auth_result
        handler.cloud_provider_credentials = Mock()

        return handler

    @patch("linopy.remote.oetc.requests.get")
    def test_get_job_logs_success(
        self, mock_get: Mock, handler_with_auth_setup: OetcHandler
    ) -> None:
        """Test successful job logs fetching"""
        # Setup
        job_uuid = "test-job-uuid-123"
        expected_logs = "Error: Solver failed\nTraceback: ...\nSolver output: ..."

        mock_response = Mock()
        mock_response.json.return_value = {"content": expected_logs}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Execute
        result = handler_with_auth_setup._get_job_logs(job_uuid)

        # Verify request
        mock_get.assert_called_once_with(
            f"https://orchestrator.example.com/compute-job/{job_uuid}/get-logs",
            headers={
                "Authorization": f"Bearer {handler_with_auth_setup.jwt.token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )

        # Verify result
        assert result == expected_logs

    @patch("linopy.remote.oetc.requests.get")
    def test_get_job_logs_http_error(
        self, mock_get: Mock, handler_with_auth_setup: OetcHandler
    ) -> None:
        """Test job logs fetching with HTTP error"""
        # Setup
        job_uuid = "test-job-uuid-123"
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        # Execute
        result = handler_with_auth_setup._get_job_logs(job_uuid)

        # Verify - should return error message instead of raising
        assert "[Unable to fetch logs:" in result
        assert "404 Not Found" in result

    @patch("linopy.remote.oetc.requests.get")
    def test_get_job_logs_empty_content(
        self, mock_get: Mock, handler_with_auth_setup: OetcHandler
    ) -> None:
        """Test job logs fetching with empty content"""
        # Setup
        job_uuid = "test-job-uuid-123"
        mock_response = Mock()
        mock_response.json.return_value = {"content": ""}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Execute
        result = handler_with_auth_setup._get_job_logs(job_uuid)

        # Verify
        assert result == ""


class TestWaitAndGetJobDataWithLogs:
    @pytest.fixture
    def handler_with_auth_setup(self, sample_jwt_token: str) -> OetcHandler:
        """Create handler with authentication setup for testing job waiting"""
        credentials = OetcCredentials(
            email="test@example.com", password="test_password"
        )
        settings = OetcSettings(
            credentials=credentials,
            name="Test Job",
            authentication_server_url="https://auth.example.com",
            orchestrator_server_url="https://orchestrator.example.com",
            compute_provider=ComputeProvider.GCP,
        )

        mock_auth_result = AuthenticationResult(
            token=sample_jwt_token,
            token_type="Bearer",
            expires_in=3600,
            authenticated_at=datetime.now(),
        )

        handler = OetcHandler.__new__(OetcHandler)
        handler.settings = settings
        handler.jwt = mock_auth_result
        handler.cloud_provider_credentials = Mock()

        return handler

    @patch("linopy.remote.oetc.requests.get")
    def test_wait_runtime_error_with_logs(
        self, mock_get: Mock, handler_with_auth_setup: OetcHandler
    ) -> None:
        """Test wait_and_get_job_data for RUNTIME_ERROR with successful log fetching"""
        # Setup
        job_uuid = "test-job-uuid-123"
        expected_logs = "Error: Solver crashed\nMemory limit exceeded\nExit code: 137"

        # First call returns RUNTIME_ERROR status
        mock_status_response = Mock()
        mock_status_response.json.return_value = {
            "uuid": job_uuid,
            "status": "RUNTIME_ERROR",
            "name": "Test Job",
        }
        mock_status_response.raise_for_status.return_value = None

        # Second call returns logs
        mock_logs_response = Mock()
        mock_logs_response.json.return_value = {"content": expected_logs}
        mock_logs_response.raise_for_status.return_value = None

        mock_get.side_effect = [mock_status_response, mock_logs_response]

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_auth_setup.wait_and_get_job_data(job_uuid)

        # Verify exception contains logs
        error_message = str(exc_info.value)
        assert "Job failed during execution" in error_message
        assert "RUNTIME_ERROR" in error_message
        assert expected_logs in error_message

        # Verify both API calls were made
        assert mock_get.call_count == 2
        # First call - get job status
        assert (
            f"https://orchestrator.example.com/compute-job/{job_uuid}"
            in mock_get.call_args_list[0][0][0]
        )
        # Second call - get logs
        assert (
            f"https://orchestrator.example.com/compute-job/{job_uuid}/get-logs"
            in mock_get.call_args_list[1][0][0]
        )

    @patch("linopy.remote.oetc.requests.get")
    def test_wait_runtime_error_logs_fetch_fails(
        self, mock_get: Mock, handler_with_auth_setup: OetcHandler
    ) -> None:
        """Test wait_and_get_job_data for RUNTIME_ERROR when log fetching fails"""
        # Setup
        job_uuid = "test-job-uuid-123"

        # First call returns RUNTIME_ERROR status
        mock_status_response = Mock()
        mock_status_response.json.return_value = {
            "uuid": job_uuid,
            "status": "RUNTIME_ERROR",
            "name": "Test Job",
        }
        mock_status_response.raise_for_status.return_value = None

        # Second call for logs fails
        def side_effect_func(*args: Any, **kwargs: Any) -> Mock:
            if "get-logs" in args[0]:
                raise RequestException("Log service unavailable")
            return mock_status_response

        mock_get.side_effect = side_effect_func

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_auth_setup.wait_and_get_job_data(job_uuid)

        # Verify exception contains error message about log fetching failure
        error_message = str(exc_info.value)
        assert "Job failed during execution" in error_message
        assert "RUNTIME_ERROR" in error_message
        assert "[Unable to fetch logs:" in error_message
        assert "Log service unavailable" in error_message

    @patch("linopy.remote.oetc.requests.get")
    def test_wait_setup_error_no_logs_fetched(
        self, mock_get: Mock, handler_with_auth_setup: OetcHandler
    ) -> None:
        """Test wait_and_get_job_data for SETUP_ERROR does not fetch logs"""
        # Setup
        job_uuid = "test-job-uuid-123"

        # First call returns SETUP_ERROR status
        mock_status_response = Mock()
        mock_status_response.json.return_value = {
            "uuid": job_uuid,
            "status": "SETUP_ERROR",
            "name": "Test Job",
        }
        mock_status_response.raise_for_status.return_value = None
        mock_get.return_value = mock_status_response

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_auth_setup.wait_and_get_job_data(job_uuid)

        # Verify exception does not try to fetch logs
        error_message = str(exc_info.value)
        assert "Job failed during setup phase" in error_message
        assert "SETUP_ERROR" in error_message

        # Verify only one API call was made (status check, no logs fetch)
        assert mock_get.call_count == 1


class TestFileCompression:
    @pytest.fixture
    def handler_with_mocked_auth(self) -> OetcHandler:
        """Create handler with mocked authentication for testing file operations"""
        with (
            patch("linopy.remote.oetc.requests.post"),
            patch("linopy.remote.oetc.requests.get"),
        ):
            credentials = OetcCredentials(
                email="test@example.com", password="test_password"
            )
            settings = OetcSettings(
                credentials=credentials,
                name="Test Job",
                authentication_server_url="https://auth.example.com",
                orchestrator_server_url="https://orchestrator.example.com",
                compute_provider=ComputeProvider.GCP,
            )

            handler = OetcHandler.__new__(OetcHandler)
            handler.settings = settings
            handler.jwt = Mock()
            handler.cloud_provider_credentials = Mock()

            return handler

    @patch("linopy.remote.oetc.gzip.open")
    @patch("linopy.remote.oetc.os.path.exists")
    @patch("builtins.open")
    def test_gzip_compress_success(
        self,
        mock_open: Mock,
        mock_exists: Mock,
        mock_gzip_open: Mock,
        handler_with_mocked_auth: OetcHandler,
    ) -> None:
        """Test successful file compression"""
        # Setup
        source_path = "/tmp/test_file.nc"
        expected_output = "/tmp/test_file.nc.gz"

        # Mock file operations
        mock_exists.return_value = True
        mock_file_in = Mock()
        mock_file_out = Mock()
        mock_open.return_value.__enter__.return_value = mock_file_in
        mock_gzip_open.return_value.__enter__.return_value = mock_file_out

        # Mock file reading
        mock_file_in.read.side_effect = [
            b"test_data_chunk",
            b"",
        ]  # First read returns data, second returns empty

        # Execute
        result = handler_with_mocked_auth._gzip_compress(source_path)

        # Verify
        assert result == expected_output
        mock_open.assert_called_once_with(source_path, "rb")
        mock_gzip_open.assert_called_once_with(expected_output, "wb", compresslevel=9)
        mock_file_out.write.assert_called_once_with(b"test_data_chunk")

    @patch("builtins.open")
    def test_gzip_compress_file_read_error(
        self, mock_open: Mock, handler_with_mocked_auth: OetcHandler
    ) -> None:
        """Test file compression with read error"""
        # Setup
        source_path = "/tmp/test_file.nc"
        mock_open.side_effect = OSError("File not found")

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_mocked_auth._gzip_compress(source_path)

        assert "Failed to compress file" in str(exc_info.value)


class TestGcpUpload:
    @pytest.fixture
    def handler_with_gcp_credentials(
        self, mock_gcp_credentials_response: dict
    ) -> OetcHandler:
        """Create handler with GCP credentials for testing upload"""
        with (
            patch("linopy.remote.oetc.requests.post"),
            patch("linopy.remote.oetc.requests.get"),
        ):
            credentials = OetcCredentials(
                email="test@example.com", password="test_password"
            )
            settings = OetcSettings(
                credentials=credentials,
                name="Test Job",
                authentication_server_url="https://auth.example.com",
                orchestrator_server_url="https://orchestrator.example.com",
                compute_provider=ComputeProvider.GCP,
            )

            # Create proper GCP credentials
            gcp_creds = GcpCredentials(
                gcp_project_id="test-project-123",
                gcp_service_key='{"type": "service_account", "project_id": "test-project-123"}',
                input_bucket="test-input-bucket",
                solution_bucket="test-solution-bucket",
            )

            handler = OetcHandler.__new__(OetcHandler)
            handler.settings = settings
            handler.jwt = Mock()
            handler.cloud_provider_credentials = gcp_creds

            return handler

    @patch("linopy.remote.oetc.os.remove")
    @patch("linopy.remote.oetc.os.path.basename")
    @patch("linopy.remote.oetc.storage.Client")
    @patch("linopy.remote.oetc.service_account.Credentials.from_service_account_info")
    def test_upload_file_to_gcp_success(
        self,
        mock_creds_from_info: Mock,
        mock_storage_client: Mock,
        mock_basename: Mock,
        mock_remove: Mock,
        handler_with_gcp_credentials: OetcHandler,
    ) -> None:
        """Test successful file upload to GCP"""
        # Setup
        file_path = "/tmp/test_file.nc"
        compressed_path = "/tmp/test_file.nc.gz"
        compressed_name = "test_file.nc.gz"

        # Mock compression
        with patch.object(
            handler_with_gcp_credentials, "_gzip_compress", return_value=compressed_path
        ):
            # Mock path operations
            mock_basename.return_value = compressed_name

            # Mock GCP components
            mock_credentials = Mock()
            mock_creds_from_info.return_value = mock_credentials

            mock_client = Mock()
            mock_storage_client.return_value = mock_client

            mock_bucket = Mock()
            mock_client.bucket.return_value = mock_bucket

            mock_blob = Mock()
            mock_bucket.blob.return_value = mock_blob

            # Execute
            result = handler_with_gcp_credentials._upload_file_to_gcp(file_path)

            # Verify
            assert result == compressed_name

            # Verify GCP credentials creation
            mock_creds_from_info.assert_called_once_with(
                {"type": "service_account", "project_id": "test-project-123"},
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            # Verify GCP client creation
            mock_storage_client.assert_called_once_with(
                credentials=mock_credentials, project="test-project-123"
            )

            # Verify bucket access
            mock_client.bucket.assert_called_once_with("test-input-bucket")

            # Verify blob operations
            mock_bucket.blob.assert_called_once_with(compressed_name)
            mock_blob.upload_from_filename.assert_called_once_with(compressed_path)

            # Verify cleanup
            mock_remove.assert_called_once_with(compressed_path)

    @patch("linopy.remote.oetc.json.loads")
    def test_upload_file_to_gcp_invalid_service_key(
        self, mock_json_loads: Mock, handler_with_gcp_credentials: OetcHandler
    ) -> None:
        """Test upload failure with invalid service key"""
        # Setup
        file_path = "/tmp/test_file.nc"
        mock_json_loads.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_gcp_credentials._upload_file_to_gcp(file_path)

        assert "Failed to upload file to GCP" in str(exc_info.value)

    @patch("linopy.remote.oetc.storage.Client")
    @patch("linopy.remote.oetc.service_account.Credentials.from_service_account_info")
    def test_upload_file_to_gcp_upload_error(
        self,
        mock_creds_from_info: Mock,
        mock_storage_client: Mock,
        handler_with_gcp_credentials: OetcHandler,
    ) -> None:
        """Test upload failure during blob upload"""
        # Setup
        file_path = "/tmp/test_file.nc"
        compressed_path = "/tmp/test_file.nc.gz"

        # Mock compression
        with patch.object(
            handler_with_gcp_credentials, "_gzip_compress", return_value=compressed_path
        ):
            # Mock GCP setup
            mock_credentials = Mock()
            mock_creds_from_info.return_value = mock_credentials

            mock_client = Mock()
            mock_storage_client.return_value = mock_client

            mock_bucket = Mock()
            mock_client.bucket.return_value = mock_bucket

            mock_blob = Mock()
            mock_blob.upload_from_filename.side_effect = Exception("Upload failed")
            mock_bucket.blob.return_value = mock_blob

            # Execute and verify exception
            with pytest.raises(Exception) as exc_info:
                handler_with_gcp_credentials._upload_file_to_gcp(file_path)

            assert "Failed to upload file to GCP" in str(exc_info.value)


class TestFileDecompression:
    @pytest.fixture
    def handler_with_mocked_auth(self) -> OetcHandler:
        """Create handler with mocked authentication for testing file operations"""
        with (
            patch("linopy.remote.oetc.requests.post"),
            patch("linopy.remote.oetc.requests.get"),
        ):
            credentials = OetcCredentials(
                email="test@example.com", password="test_password"
            )
            settings = OetcSettings(
                credentials=credentials,
                name="Test Job",
                authentication_server_url="https://auth.example.com",
                orchestrator_server_url="https://orchestrator.example.com",
                compute_provider=ComputeProvider.GCP,
            )

            handler = OetcHandler.__new__(OetcHandler)
            handler.settings = settings
            handler.jwt = Mock()
            handler.cloud_provider_credentials = Mock()

            return handler

    @patch("linopy.remote.oetc.gzip.open")
    @patch("builtins.open")
    def test_gzip_decompress_success(
        self,
        mock_open_file: Mock,
        mock_gzip_open: Mock,
        handler_with_mocked_auth: OetcHandler,
    ) -> None:
        """Test successful file decompression"""
        # Setup
        input_path = "/tmp/test_file.nc.gz"
        expected_output = "/tmp/test_file.nc"

        # Mock file operations
        mock_file_in = Mock()
        mock_file_out = Mock()
        mock_gzip_open.return_value.__enter__.return_value = mock_file_in
        mock_open_file.return_value.__enter__.return_value = mock_file_out

        # Mock file reading - simulate reading compressed data in chunks
        mock_file_in.read.side_effect = [
            b"decompressed_data_chunk",
            b"",
        ]  # First read returns data, second returns empty

        # Execute
        result = handler_with_mocked_auth._gzip_decompress(input_path)

        # Verify
        assert result == expected_output
        mock_gzip_open.assert_called_once_with(input_path, "rb")
        mock_open_file.assert_called_once_with(expected_output, "wb")
        mock_file_out.write.assert_called_once_with(b"decompressed_data_chunk")

    @patch("linopy.remote.oetc.gzip.open")
    def test_gzip_decompress_gzip_open_error(
        self, mock_gzip_open: Mock, handler_with_mocked_auth: OetcHandler
    ) -> None:
        """Test file decompression with gzip open error"""
        # Setup
        input_path = "/tmp/test_file.nc.gz"
        mock_gzip_open.side_effect = OSError("Failed to open gzip file")

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_mocked_auth._gzip_decompress(input_path)

        assert "Failed to decompress file" in str(exc_info.value)

    @patch("linopy.remote.oetc.gzip.open")
    @patch("builtins.open")
    def test_gzip_decompress_write_error(
        self,
        mock_open_file: Mock,
        mock_gzip_open: Mock,
        handler_with_mocked_auth: OetcHandler,
    ) -> None:
        """Test file decompression with write error"""
        # Setup
        input_path = "/tmp/test_file.nc.gz"

        # Mock file operations
        mock_file_in = Mock()
        mock_gzip_open.return_value.__enter__.return_value = mock_file_in
        mock_open_file.side_effect = OSError("Permission denied")

        # Mock file reading
        mock_file_in.read.return_value = b"test_data"

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_mocked_auth._gzip_decompress(input_path)

        assert "Failed to decompress file" in str(exc_info.value)

    def test_gzip_decompress_output_path_generation(
        self, handler_with_mocked_auth: OetcHandler
    ) -> None:
        """Test correct output path generation for decompression"""
        # Test first path
        with patch("linopy.remote.oetc.gzip.open") as mock_gzip_open:
            with patch("builtins.open") as mock_open_file:
                mock_file_in = Mock()
                mock_file_out = Mock()
                mock_gzip_open.return_value.__enter__.return_value = mock_file_in
                mock_open_file.return_value.__enter__.return_value = mock_file_out
                mock_file_in.read.side_effect = [b"test", b""]

                result = handler_with_mocked_auth._gzip_decompress("/tmp/file.nc.gz")
                assert result == "/tmp/file.nc"

        # Test second path with fresh mocks
        with patch("linopy.remote.oetc.gzip.open") as mock_gzip_open:
            with patch("builtins.open") as mock_open_file:
                mock_file_in = Mock()
                mock_file_out = Mock()
                mock_gzip_open.return_value.__enter__.return_value = mock_file_in
                mock_open_file.return_value.__enter__.return_value = mock_file_out
                mock_file_in.read.side_effect = [b"test", b""]

                result = handler_with_mocked_auth._gzip_decompress(
                    "/path/to/model.data.gz"
                )
                assert result == "/path/to/model.data"


class TestGcpDownload:
    @pytest.fixture
    def handler_with_gcp_credentials(
        self, mock_gcp_credentials_response: dict
    ) -> OetcHandler:
        """Create handler with GCP credentials for testing download"""
        with (
            patch("linopy.remote.oetc.requests.post"),
            patch("linopy.remote.oetc.requests.get"),
        ):
            credentials = OetcCredentials(
                email="test@example.com", password="test_password"
            )
            settings = OetcSettings(
                credentials=credentials,
                name="Test Job",
                authentication_server_url="https://auth.example.com",
                orchestrator_server_url="https://orchestrator.example.com",
                compute_provider=ComputeProvider.GCP,
            )

            # Create proper GCP credentials
            gcp_creds = GcpCredentials(
                gcp_project_id="test-project-123",
                gcp_service_key='{"type": "service_account", "project_id": "test-project-123"}',
                input_bucket="test-input-bucket",
                solution_bucket="test-solution-bucket",
            )

            handler = OetcHandler.__new__(OetcHandler)
            handler.settings = settings
            handler.jwt = Mock()
            handler.cloud_provider_credentials = gcp_creds

            return handler

    @patch("linopy.remote.oetc.os.remove")
    @patch("linopy.remote.oetc.tempfile.NamedTemporaryFile")
    @patch("linopy.remote.oetc.storage.Client")
    @patch("linopy.remote.oetc.service_account.Credentials.from_service_account_info")
    def test_download_file_from_gcp_success(
        self,
        mock_creds_from_info: Mock,
        mock_storage_client: Mock,
        mock_tempfile: Mock,
        mock_remove: Mock,
        handler_with_gcp_credentials: OetcHandler,
    ) -> None:
        """Test successful file download from GCP"""
        # Setup
        file_name = "solution_file.nc.gz"
        compressed_path = "/tmp/tmpfile.gz"
        decompressed_path = "/tmp/tmpfile"

        # Mock temporary file creation
        mock_temp_file = Mock()
        mock_temp_file.name = compressed_path
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        # Mock decompression
        with patch.object(
            handler_with_gcp_credentials,
            "_gzip_decompress",
            return_value=decompressed_path,
        ):
            # Mock GCP components
            mock_credentials = Mock()
            mock_creds_from_info.return_value = mock_credentials

            mock_client = Mock()
            mock_storage_client.return_value = mock_client

            mock_bucket = Mock()
            mock_client.bucket.return_value = mock_bucket

            mock_blob = Mock()
            mock_bucket.blob.return_value = mock_blob

            # Execute
            result = handler_with_gcp_credentials._download_file_from_gcp(file_name)

            # Verify
            assert result == decompressed_path

            # Verify GCP credentials creation
            mock_creds_from_info.assert_called_once_with(
                {"type": "service_account", "project_id": "test-project-123"},
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            # Verify GCP client creation
            mock_storage_client.assert_called_once_with(
                credentials=mock_credentials, project="test-project-123"
            )

            # Verify bucket access (solution bucket, not input bucket)
            mock_client.bucket.assert_called_once_with("test-solution-bucket")

            # Verify blob operations
            mock_bucket.blob.assert_called_once_with(file_name)
            mock_blob.download_to_filename.assert_called_once_with(compressed_path)

            # Verify cleanup
            mock_remove.assert_called_once_with(compressed_path)

    @patch("linopy.remote.oetc.json.loads")
    def test_download_file_from_gcp_invalid_service_key(
        self, mock_json_loads: Mock, handler_with_gcp_credentials: OetcHandler
    ) -> None:
        """Test download failure with invalid service key"""
        # Setup
        file_name = "solution_file.nc.gz"
        mock_json_loads.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_gcp_credentials._download_file_from_gcp(file_name)

        assert "Failed to download file from GCP" in str(exc_info.value)

    @patch("linopy.remote.oetc.tempfile.NamedTemporaryFile")
    @patch("linopy.remote.oetc.storage.Client")
    @patch("linopy.remote.oetc.service_account.Credentials.from_service_account_info")
    def test_download_file_from_gcp_download_error(
        self,
        mock_creds_from_info: Mock,
        mock_storage_client: Mock,
        mock_tempfile: Mock,
        handler_with_gcp_credentials: OetcHandler,
    ) -> None:
        """Test download failure during blob download"""
        # Setup
        file_name = "solution_file.nc.gz"
        compressed_path = "/tmp/tmpfile.gz"

        # Mock temporary file creation
        mock_temp_file = Mock()
        mock_temp_file.name = compressed_path
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        # Mock GCP setup
        mock_credentials = Mock()
        mock_creds_from_info.return_value = mock_credentials

        mock_client = Mock()
        mock_storage_client.return_value = mock_client

        mock_bucket = Mock()
        mock_client.bucket.return_value = mock_bucket

        mock_blob = Mock()
        mock_blob.download_to_filename.side_effect = Exception("Download failed")
        mock_bucket.blob.return_value = mock_blob

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_gcp_credentials._download_file_from_gcp(file_name)

        assert "Failed to download file from GCP" in str(exc_info.value)

    @patch("linopy.remote.oetc.os.remove")
    @patch("linopy.remote.oetc.tempfile.NamedTemporaryFile")
    @patch("linopy.remote.oetc.storage.Client")
    @patch("linopy.remote.oetc.service_account.Credentials.from_service_account_info")
    def test_download_file_from_gcp_decompression_error(
        self,
        mock_creds_from_info: Mock,
        mock_storage_client: Mock,
        mock_tempfile: Mock,
        mock_remove: Mock,
        handler_with_gcp_credentials: OetcHandler,
    ) -> None:
        """Test download failure during decompression"""
        # Setup
        file_name = "solution_file.nc.gz"
        compressed_path = "/tmp/tmpfile.gz"

        # Mock temporary file creation
        mock_temp_file = Mock()
        mock_temp_file.name = compressed_path
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        # Mock successful GCP download but failed decompression
        mock_credentials = Mock()
        mock_creds_from_info.return_value = mock_credentials

        mock_client = Mock()
        mock_storage_client.return_value = mock_client

        mock_bucket = Mock()
        mock_client.bucket.return_value = mock_bucket

        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob

        # Mock decompression failure
        with patch.object(
            handler_with_gcp_credentials,
            "_gzip_decompress",
            side_effect=Exception("Decompression failed"),
        ):
            # Execute and verify exception
            with pytest.raises(Exception) as exc_info:
                handler_with_gcp_credentials._download_file_from_gcp(file_name)

            assert "Failed to download file from GCP" in str(exc_info.value)

    @patch("linopy.remote.oetc.service_account.Credentials.from_service_account_info")
    def test_download_file_from_gcp_credentials_error(
        self, mock_creds_from_info: Mock, handler_with_gcp_credentials: OetcHandler
    ) -> None:
        """Test download failure during credentials creation"""
        # Setup
        file_name = "solution_file.nc.gz"
        mock_creds_from_info.side_effect = Exception("Invalid credentials")

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_gcp_credentials._download_file_from_gcp(file_name)

        assert "Failed to download file from GCP" in str(exc_info.value)


class TestJobSubmission:
    @pytest.fixture
    def handler_with_auth_setup(self, sample_jwt_token: str) -> OetcHandler:
        """Create handler with authentication setup for testing job submission"""
        credentials = OetcCredentials(
            email="test@example.com", password="test_password"
        )
        settings = OetcSettings(
            credentials=credentials,
            name="Test Optimization Job",
            authentication_server_url="https://auth.example.com",
            orchestrator_server_url="https://orchestrator.example.com",
            compute_provider=ComputeProvider.GCP,
            solver="gurobi",
            cpu_cores=4,
            disk_space_gb=20,
        )

        # Mock the authentication result
        mock_auth_result = AuthenticationResult(
            token=sample_jwt_token,
            token_type="Bearer",
            expires_in=3600,
            authenticated_at=datetime.now(),
        )

        handler = OetcHandler.__new__(OetcHandler)
        handler.settings = settings
        handler.jwt = mock_auth_result
        handler.cloud_provider_credentials = Mock()

        return handler

    @patch("linopy.remote.oetc.requests.post")
    def test_submit_job_success(
        self, mock_post: Mock, handler_with_auth_setup: OetcHandler
    ) -> None:
        """Test successful job submission to compute service"""
        # Setup
        input_file_name = "test_model.nc.gz"
        expected_job_uuid = "job-uuid-123"

        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {"uuid": expected_job_uuid}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Execute
        result = handler_with_auth_setup._submit_job_to_compute_service(input_file_name)

        # Verify request
        expected_payload = {
            "name": "Test Optimization Job",
            "solver": "gurobi",
            "solver_options": {},
            "provider": "GCP",
            "cpu_cores": 4,
            "disk_space_gb": 20,
            "input_file_name": input_file_name,
            "delete_worker_on_error": False,
        }

        mock_post.assert_called_once_with(
            "https://orchestrator.example.com/compute-job/create",
            json=expected_payload,
            headers={
                "Authorization": f"Bearer {handler_with_auth_setup.jwt.token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )

        # Verify result
        assert result == expected_job_uuid

    @patch("linopy.remote.oetc.requests.post")
    def test_submit_job_http_error(
        self, mock_post: Mock, handler_with_auth_setup: OetcHandler
    ) -> None:
        """Test job submission with HTTP error"""
        # Setup
        input_file_name = "test_model.nc.gz"
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "400 Bad Request"
        )
        mock_post.return_value = mock_response

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_auth_setup._submit_job_to_compute_service(input_file_name)

        assert "Failed to submit job to compute service" in str(exc_info.value)

    @patch("linopy.remote.oetc.requests.post")
    def test_submit_job_missing_uuid_in_response(
        self, mock_post: Mock, handler_with_auth_setup: OetcHandler
    ) -> None:
        """Test job submission with missing UUID in response"""
        # Setup
        input_file_name = "test_model.nc.gz"
        mock_response = Mock()
        mock_response.json.return_value = {}  # Missing "uuid" field
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_auth_setup._submit_job_to_compute_service(input_file_name)

        assert "Invalid job submission response format: missing field 'uuid'" in str(
            exc_info.value
        )

    @patch("linopy.remote.oetc.requests.post")
    def test_submit_job_network_error(
        self, mock_post: Mock, handler_with_auth_setup: OetcHandler
    ) -> None:
        """Test job submission with network error"""
        # Setup
        input_file_name = "test_model.nc.gz"
        mock_post.side_effect = RequestException("Connection timeout")

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            handler_with_auth_setup._submit_job_to_compute_service(input_file_name)

        assert "Failed to submit job to compute service" in str(exc_info.value)


class TestSolveOnOetc:
    @pytest.fixture
    def handler_with_complete_setup(
        self, mock_gcp_credentials_response: dict
    ) -> OetcHandler:
        """Create handler with complete setup for testing solve functionality"""
        with (
            patch("linopy.remote.oetc.requests.post"),
            patch("linopy.remote.oetc.requests.get"),
        ):
            credentials = OetcCredentials(
                email="test@example.com", password="test_password"
            )
            settings = OetcSettings(
                credentials=credentials,
                name="Test Job",
                authentication_server_url="https://auth.example.com",
                orchestrator_server_url="https://orchestrator.example.com",
                compute_provider=ComputeProvider.GCP,
            )

            gcp_creds = GcpCredentials(
                gcp_project_id="test-project-123",
                gcp_service_key='{"type": "service_account", "project_id": "test-project-123"}',
                input_bucket="test-input-bucket",
                solution_bucket="test-solution-bucket",
            )

            handler = OetcHandler.__new__(OetcHandler)
            handler.settings = settings
            handler.jwt = Mock()
            handler.cloud_provider_credentials = gcp_creds

            return handler

    @patch("linopy.remote.oetc.linopy.read_netcdf")
    @patch("linopy.remote.oetc.os.remove")
    @patch("linopy.remote.oetc.tempfile.NamedTemporaryFile")
    def test_solve_on_oetc_file_upload(
        self,
        mock_tempfile: Mock,
        mock_remove: Mock,
        mock_read_netcdf: Mock,
        handler_with_complete_setup: OetcHandler,
    ) -> None:
        """Test solve_on_oetc method complete workflow"""
        # Setup
        mock_model = Mock()
        mock_solved_model = Mock()
        mock_solved_model.status = "optimal"
        mock_solved_model.objective.value = 42.0

        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/linopy-abc123.nc"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        mock_read_netcdf.return_value = mock_solved_model

        # Mock file upload, job submission, job waiting, and download
        with patch.object(
            handler_with_complete_setup,
            "_upload_file_to_gcp",
            return_value="uploaded_file.nc.gz",
        ) as mock_upload:
            with patch.object(
                handler_with_complete_setup,
                "_submit_job_to_compute_service",
                return_value="test-job-uuid",
            ) as mock_submit:
                with patch.object(
                    handler_with_complete_setup,
                    "wait_and_get_job_data",
                    return_value=JobResult(
                        uuid="test-job-uuid",
                        status="FINISHED",
                        output_files=[{"name": "result.nc.gz"}],
                    ),
                ) as mock_wait:
                    with patch.object(
                        handler_with_complete_setup,
                        "_download_file_from_gcp",
                        return_value="/tmp/downloaded_result.nc",
                    ) as mock_download:
                        # Execute
                        result = handler_with_complete_setup.solve_on_oetc(mock_model)

                        # Verify
                        assert (
                            result == mock_solved_model
                        )  # Now returns the solved model
                        mock_model.to_netcdf.assert_called_once_with(
                            "/tmp/linopy-abc123.nc"
                        )
                        mock_upload.assert_called_once_with("/tmp/linopy-abc123.nc")
                        mock_submit.assert_called_once_with("uploaded_file.nc.gz")
                        mock_wait.assert_called_once_with("test-job-uuid")
                        mock_download.assert_called_once_with("result.nc.gz")
                        mock_read_netcdf.assert_called_once_with(
                            "/tmp/downloaded_result.nc"
                        )
                        mock_remove.assert_called_once_with("/tmp/downloaded_result.nc")

    @patch("linopy.remote.oetc.tempfile.NamedTemporaryFile")
    def test_solve_on_oetc_upload_failure(
        self, mock_tempfile: Mock, handler_with_complete_setup: OetcHandler
    ) -> None:
        """Test solve_on_oetc method with upload failure"""
        # Setup
        mock_model = Mock()
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/linopy-abc123.nc"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        # Mock upload failure
        with patch.object(
            handler_with_complete_setup,
            "_upload_file_to_gcp",
            side_effect=Exception("Upload failed"),
        ):
            # Execute and verify exception
            with pytest.raises(Exception) as exc_info:
                handler_with_complete_setup.solve_on_oetc(mock_model)

            assert "Upload failed" in str(exc_info.value)


class TestSolveOnOetcWithJobSubmission:
    @pytest.fixture
    def handler_with_full_setup(self) -> OetcHandler:
        """Create handler with full setup for testing complete solve flow"""
        credentials = OetcCredentials(
            email="test@example.com", password="test_password"
        )
        settings = OetcSettings(
            credentials=credentials,
            name="Linopy Solve Job",
            authentication_server_url="https://auth.example.com",
            orchestrator_server_url="https://orchestrator.example.com",
            compute_provider=ComputeProvider.GCP,
            solver="highs",
            cpu_cores=2,
            disk_space_gb=15,
        )

        gcp_creds = GcpCredentials(
            gcp_project_id="test-project-123",
            gcp_service_key='{"type": "service_account", "project_id": "test-project-123"}',
            input_bucket="test-input-bucket",
            solution_bucket="test-solution-bucket",
        )

        handler = OetcHandler.__new__(OetcHandler)
        handler.settings = settings
        handler.jwt = Mock()
        handler.cloud_provider_credentials = gcp_creds

        return handler

    @patch("linopy.remote.oetc.linopy.read_netcdf")
    @patch("linopy.remote.oetc.os.remove")
    @patch("linopy.remote.oetc.tempfile.NamedTemporaryFile")
    def test_solve_on_oetc_with_job_submission(
        self,
        mock_tempfile: Mock,
        mock_remove: Mock,
        mock_read_netcdf: Mock,
        handler_with_full_setup: OetcHandler,
    ) -> None:
        """Test solve_on_oetc method including job submission, waiting, and download"""
        # Setup
        mock_model = Mock()
        mock_solved_model = Mock()
        mock_solved_model.status = "optimal"
        mock_solved_model.objective.value = 100.5

        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/linopy-abc123.nc"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        mock_read_netcdf.return_value = mock_solved_model

        uploaded_file_name = "model_file.nc.gz"
        job_uuid = "job-uuid-456"

        # Mock complete workflow
        with patch.object(
            handler_with_full_setup,
            "_upload_file_to_gcp",
            return_value=uploaded_file_name,
        ) as mock_upload:
            with patch.object(
                handler_with_full_setup,
                "_submit_job_to_compute_service",
                return_value=job_uuid,
            ) as mock_submit:
                with patch.object(
                    handler_with_full_setup,
                    "wait_and_get_job_data",
                    return_value=JobResult(
                        uuid=job_uuid,
                        status="FINISHED",
                        output_files=[{"name": "result.nc.gz"}],
                    ),
                ) as mock_wait:
                    with patch.object(
                        handler_with_full_setup,
                        "_download_file_from_gcp",
                        return_value="/tmp/solution_file.nc",
                    ) as mock_download:
                        # Execute
                        result = handler_with_full_setup.solve_on_oetc(mock_model)

                        # Verify
                        assert (
                            result == mock_solved_model
                        )  # Now returns the solved model
                        mock_model.to_netcdf.assert_called_once_with(
                            "/tmp/linopy-abc123.nc"
                        )
                        mock_upload.assert_called_once_with("/tmp/linopy-abc123.nc")
                        mock_submit.assert_called_once_with(uploaded_file_name)
                        mock_wait.assert_called_once_with(job_uuid)
                        mock_download.assert_called_once_with("result.nc.gz")
                        mock_read_netcdf.assert_called_once_with(
                            "/tmp/solution_file.nc"
                        )
                        mock_remove.assert_called_once_with("/tmp/solution_file.nc")

    @patch("linopy.remote.oetc.tempfile.NamedTemporaryFile")
    def test_solve_on_oetc_job_submission_failure(
        self, mock_tempfile: Mock, handler_with_full_setup: OetcHandler
    ) -> None:
        """Test solve_on_oetc method with job submission failure"""
        # Setup
        mock_model = Mock()
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/linopy-abc123.nc"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        uploaded_file_name = "model_file.nc.gz"

        # Mock successful upload but failed job submission - no need to mock waiting since submission fails
        with patch.object(
            handler_with_full_setup,
            "_upload_file_to_gcp",
            return_value=uploaded_file_name,
        ):
            with patch.object(
                handler_with_full_setup,
                "_submit_job_to_compute_service",
                side_effect=Exception("Job submission failed"),
            ):
                # Execute and verify exception
                with pytest.raises(Exception) as exc_info:
                    handler_with_full_setup.solve_on_oetc(mock_model)

                assert "Job submission failed" in str(exc_info.value)

    @patch("linopy.remote.oetc.tempfile.NamedTemporaryFile")
    def test_solve_on_oetc_job_waiting_failure(
        self, mock_tempfile: Mock, handler_with_full_setup: OetcHandler
    ) -> None:
        """Test solve_on_oetc method with job waiting failure"""
        # Setup
        mock_model = Mock()
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/linopy-abc123.nc"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        uploaded_file_name = "model_file.nc.gz"
        job_uuid = "job-uuid-failed"

        # Mock successful upload and job submission but failed job waiting
        with patch.object(
            handler_with_full_setup,
            "_upload_file_to_gcp",
            return_value=uploaded_file_name,
        ):
            with patch.object(
                handler_with_full_setup,
                "_submit_job_to_compute_service",
                return_value=job_uuid,
            ):
                with patch.object(
                    handler_with_full_setup,
                    "wait_and_get_job_data",
                    side_effect=Exception("Job failed: solver error"),
                ):
                    # Execute and verify exception
                    with pytest.raises(Exception) as exc_info:
                        handler_with_full_setup.solve_on_oetc(mock_model)

                    assert "Job failed: solver error" in str(exc_info.value)

    @patch("linopy.remote.oetc.tempfile.NamedTemporaryFile")
    def test_solve_on_oetc_no_output_files_error(
        self, mock_tempfile: Mock, handler_with_full_setup: OetcHandler
    ) -> None:
        """Test solve_on_oetc method when job completes but has no output files"""
        # Setup
        mock_model = Mock()
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/linopy-abc123.nc"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        uploaded_file_name = "model_file.nc.gz"
        job_uuid = "job-uuid-456"

        # Mock successful workflow until job completion with no output files
        with patch.object(
            handler_with_full_setup,
            "_upload_file_to_gcp",
            return_value=uploaded_file_name,
        ):
            with patch.object(
                handler_with_full_setup,
                "_submit_job_to_compute_service",
                return_value=job_uuid,
            ):
                with patch.object(
                    handler_with_full_setup,
                    "wait_and_get_job_data",
                    return_value=JobResult(
                        uuid=job_uuid, status="FINISHED", output_files=[]
                    ),
                ):  # No output files
                    # Execute and verify exception
                    with pytest.raises(Exception) as exc_info:
                        handler_with_full_setup.solve_on_oetc(mock_model)

                    assert "No output files found in completed job" in str(
                        exc_info.value
                    )


# Additional integration-style test
class TestOetcHandlerIntegration:
    @patch("linopy.remote.oetc.requests.post")
    @patch("linopy.remote.oetc.requests.get")
    @patch("linopy.remote.oetc.datetime")
    def test_complete_authentication_flow(
        self, mock_datetime: Mock, mock_get: Mock, mock_post: Mock
    ) -> None:
        """Test complete authentication and credentials flow with realistic data"""
        # Setup
        fixed_time = datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.now.return_value = fixed_time

        credentials = OetcCredentials(
            email="user@company.com", password="secure_password_123"
        )
        settings = OetcSettings(
            credentials=credentials,
            name="Integration Test Job",
            authentication_server_url="https://api.company.com/auth",
            orchestrator_server_url="https://api.company.com/orchestrator",
            compute_provider=ComputeProvider.GCP,
        )

        # Create realistic JWT token
        payload = {
            "iss": "OETC",
            "sub": "user-uuid-456",
            "exp": 1640995200,
            "jti": "jwt-id-789",
            "email": "user@company.com",
            "firstname": "John",
            "lastname": "Doe",
        }
        header = (
            base64.urlsafe_b64encode(
                json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
            )
            .decode()
            .rstrip("=")
        )
        payload_encoded = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )
        realistic_token = f"{header}.{payload_encoded}.realistic_signature"

        # Mock authentication response
        mock_auth_response = Mock()
        mock_auth_response.json.return_value = {
            "token": realistic_token,
            "token_type": "Bearer",
            "expires_in": 7200,  # 2 hours
        }
        mock_auth_response.raise_for_status.return_value = None
        mock_post.return_value = mock_auth_response

        # Mock GCP credentials response
        mock_gcp_response = Mock()
        mock_gcp_response.json.return_value = {
            "gcp_project_id": "production-project-456",
            "gcp_service_key": "production-service-key-content",
            "input_bucket": "prod-input-bucket",
            "solution_bucket": "prod-solution-bucket",
        }
        mock_gcp_response.raise_for_status.return_value = None
        mock_get.return_value = mock_gcp_response

        # Execute
        handler = OetcHandler(settings)

        # Verify authentication
        assert handler.jwt.token == realistic_token
        assert handler.jwt.token_type == "Bearer"
        assert handler.jwt.expires_in == 7200
        assert handler.jwt.authenticated_at == fixed_time
        assert handler.jwt.expires_at == datetime(
            2024, 1, 15, 14, 0, 0
        )  # 2 hours later
        assert handler.jwt.is_expired is False

        # Verify GCP credentials
        assert isinstance(handler.cloud_provider_credentials, GcpCredentials)
        assert (
            handler.cloud_provider_credentials.gcp_project_id
            == "production-project-456"
        )
        assert (
            handler.cloud_provider_credentials.gcp_service_key
            == "production-service-key-content"
        )
        assert handler.cloud_provider_credentials.input_bucket == "prod-input-bucket"
        assert (
            handler.cloud_provider_credentials.solution_bucket == "prod-solution-bucket"
        )

        # Verify correct API calls were made
        mock_post.assert_called_once_with(
            "https://api.company.com/auth/sign-in",
            json={"email": "user@company.com", "password": "secure_password_123"},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        mock_get.assert_called_once_with(
            "https://api.company.com/auth/users/user-uuid-456/gcp-credentials",
            headers={
                "Authorization": f"Bearer {realistic_token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
