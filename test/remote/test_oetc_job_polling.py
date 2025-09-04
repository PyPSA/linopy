"""
Tests for OetcHandler job polling and status monitoring.

This module tests the wait_and_get_job_data method which polls for job completion
and handles various job states and error conditions.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from requests import RequestException

from linopy.remote.oetc import (
    AuthenticationResult,
    ComputeProvider,
    OetcCredentials,
    OetcHandler,
    OetcSettings,
)


@pytest.fixture
def mock_settings() -> OetcSettings:
    """Create mock settings for testing."""
    credentials = OetcCredentials(email="test@example.com", password="test_password")
    return OetcSettings(
        credentials=credentials,
        name="Test Job",
        authentication_server_url="https://auth.example.com",
        orchestrator_server_url="https://orchestrator.example.com",
        compute_provider=ComputeProvider.GCP,
    )


@pytest.fixture
def mock_auth_result() -> AuthenticationResult:
    """Create mock authentication result."""
    return AuthenticationResult(
        token="mock_token",
        token_type="Bearer",
        expires_in=3600,
        authenticated_at=datetime.now(),
    )


@pytest.fixture
def oetc_handler(
    mock_settings: OetcSettings, mock_auth_result: AuthenticationResult
) -> OetcHandler:
    """Create OetcHandler with mocked authentication."""
    with patch(
        "linopy.remote.oetc.OetcHandler._OetcHandler__sign_in",
        return_value=mock_auth_result,
    ):
        with patch(
            "linopy.remote.oetc.OetcHandler._OetcHandler__get_cloud_provider_credentials"
        ):
            handler = OetcHandler(mock_settings)
            return handler


class TestJobPollingSuccess:
    """Test successful job polling scenarios."""

    def test_job_completes_immediately(self, oetc_handler: OetcHandler) -> None:
        """Test job that completes on first poll."""
        job_data = {
            "uuid": "job-123",
            "status": "FINISHED",
            "name": "test-job",
            "owner": "test-user",
            "solver": "highs",
            "duration_in_seconds": 120,
            "solving_duration_in_seconds": 90,
            "input_files": ["input.nc"],
            "output_files": ["output.nc"],
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = job_data
            mock_get.return_value = mock_response

            result = oetc_handler.wait_and_get_job_data("job-123")

            assert result.uuid == "job-123"
            assert result.status == "FINISHED"
            assert result.output_files == ["output.nc"]
            mock_get.assert_called_once()

    def test_job_completes_with_no_output_files_warning(
        self, oetc_handler: OetcHandler
    ) -> None:
        """Test job completion with no output files generates warning."""
        job_data = {"uuid": "job-123", "status": "FINISHED", "output_files": []}

        with patch("requests.get") as mock_get:
            with patch("linopy.remote.oetc.logger.warning") as mock_warning:
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = job_data
                mock_get.return_value = mock_response

                result = oetc_handler.wait_and_get_job_data("job-123")

                assert result.status == "FINISHED"
                mock_warning.assert_called_once_with(
                    "OETC - Warning: Job completed but no output files found"
                )

    @patch("time.sleep")  # Mock sleep to speed up test
    def test_job_polling_progression(
        self, mock_sleep: Mock, oetc_handler: OetcHandler
    ) -> None:
        """Test job progresses through multiple states before completion."""
        responses = [
            {"uuid": "job-123", "status": "PENDING"},
            {"uuid": "job-123", "status": "STARTING"},
            {"uuid": "job-123", "status": "RUNNING", "duration_in_seconds": 30},
            {"uuid": "job-123", "status": "RUNNING", "duration_in_seconds": 60},
            {"uuid": "job-123", "status": "FINISHED", "output_files": ["output.nc"]},
        ]

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.side_effect = responses
            mock_get.return_value = mock_response

            result = oetc_handler.wait_and_get_job_data(
                "job-123", initial_poll_interval=1
            )

            assert result.status == "FINISHED"
            assert mock_get.call_count == 5
            assert mock_sleep.call_count == 4  # Sleep called 4 times between 5 polls

    @patch("time.sleep")
    def test_polling_interval_backoff(
        self, mock_sleep: Mock, oetc_handler: OetcHandler
    ) -> None:
        """Test polling interval increases with exponential backoff."""
        responses = [
            {"uuid": "job-123", "status": "PENDING"},
            {"uuid": "job-123", "status": "RUNNING"},
            {"uuid": "job-123", "status": "FINISHED", "output_files": ["output.nc"]},
        ]

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.side_effect = responses
            mock_get.return_value = mock_response

            oetc_handler.wait_and_get_job_data(
                "job-123", initial_poll_interval=10, max_poll_interval=100
            )

            # Verify sleep was called with increasing intervals
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert sleep_calls[0] == 10  # Initial interval
            assert sleep_calls[1] == 15  # 10 * 1.5 = 15


class TestJobPollingErrors:
    """Test job polling error scenarios."""

    def test_setup_error_status(self, oetc_handler: OetcHandler) -> None:
        """Test job with SETUP_ERROR status raises exception."""
        job_data = {"uuid": "job-123", "status": "SETUP_ERROR"}

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = job_data
            mock_get.return_value = mock_response

            with pytest.raises(Exception, match="Job failed during setup phase"):
                oetc_handler.wait_and_get_job_data("job-123")

    def test_runtime_error_status(self, oetc_handler: OetcHandler) -> None:
        """Test job with RUNTIME_ERROR status raises exception."""
        job_data = {"uuid": "job-123", "status": "RUNTIME_ERROR"}

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = job_data
            mock_get.return_value = mock_response

            with pytest.raises(Exception, match="Job failed during execution"):
                oetc_handler.wait_and_get_job_data("job-123")

    def test_unknown_status_error(self, oetc_handler: OetcHandler) -> None:
        """Test job with unknown status raises exception."""
        job_data = {"uuid": "job-123", "status": "UNKNOWN_STATUS"}

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = job_data
            mock_get.return_value = mock_response

            with pytest.raises(Exception, match="Unknown job status: UNKNOWN_STATUS"):
                oetc_handler.wait_and_get_job_data("job-123")


class TestJobPollingNetworkErrors:
    """Test network error handling during job polling."""

    @patch("time.sleep")
    def test_network_retry_success(
        self, mock_sleep: Mock, oetc_handler: OetcHandler
    ) -> None:
        """Test network errors are retried and eventually succeed."""
        successful_response = {
            "uuid": "job-123",
            "status": "FINISHED",
            "output_files": ["output.nc"],
        }

        with patch("requests.get") as mock_get:
            # First two calls fail, third succeeds
            mock_get.side_effect = [
                RequestException("Network error 1"),
                RequestException("Network error 2"),
                Mock(
                    raise_for_status=Mock(), json=Mock(return_value=successful_response)
                ),
            ]

            result = oetc_handler.wait_and_get_job_data("job-123")

            assert result.status == "FINISHED"
            assert mock_get.call_count == 3
            assert mock_sleep.call_count == 2  # Retry delays

            # Verify retry delays increase
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert sleep_calls[0] == 10  # First retry: 1 * 10 = 10
            assert sleep_calls[1] == 20  # Second retry: 2 * 10 = 20

    @patch("time.sleep")
    def test_max_network_retries_exceeded(
        self, mock_sleep: Mock, oetc_handler: OetcHandler
    ) -> None:
        """Test max network retries causes exception."""
        with patch("requests.get") as mock_get:
            # All calls fail with RequestException
            mock_get.side_effect = RequestException("Network error")

            with pytest.raises(
                Exception, match="Failed to get job status after 10 network retries"
            ):
                oetc_handler.wait_and_get_job_data("job-123")

            # Should retry exactly 10 times before failing
            assert mock_get.call_count == 10

    @patch("time.sleep")
    def test_network_retry_delay_cap(
        self, mock_sleep: Mock, oetc_handler: OetcHandler
    ) -> None:
        """Test network retry delay is capped at 60 seconds."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = RequestException("Network error")

            with pytest.raises(Exception):
                oetc_handler.wait_and_get_job_data("job-123")

            # Check that delay is capped at 60 seconds
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert all(delay <= 60 for delay in sleep_calls)

    def test_keyerror_in_response(self, oetc_handler: OetcHandler) -> None:
        """Test KeyError in response parsing raises exception."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {}  # Missing required 'uuid' field
            mock_get.return_value = mock_response

            with pytest.raises(
                Exception, match="Invalid job status response format: missing field"
            ):
                oetc_handler.wait_and_get_job_data("job-123")

    def test_generic_exception_handling(self, oetc_handler: OetcHandler) -> None:
        """Test generic exception handling in polling loop."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = ValueError("Unexpected error")
            mock_get.return_value = mock_response

            with pytest.raises(
                Exception, match="Error getting job status: Unexpected error"
            ):
                oetc_handler.wait_and_get_job_data("job-123")

    def test_status_error_exception_preserved(self, oetc_handler: OetcHandler) -> None:
        """Test that status-related exceptions are preserved."""
        with patch("requests.get") as mock_get:
            # Simulate an exception that mentions "status:" - should be re-raised as-is
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = Exception(
                "Custom status: error"
            )
            mock_get.return_value = mock_response

            with pytest.raises(Exception, match="Custom status: error"):
                oetc_handler.wait_and_get_job_data("job-123")

    def test_oetc_logs_exception_preserved(self, oetc_handler: OetcHandler) -> None:
        """Test that OETC logs exceptions are preserved."""
        with patch("requests.get") as mock_get:
            # Simulate an exception that mentions "OETC logs" - should be re-raised as-is
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = Exception(
                "Check the OETC logs for details"
            )
            mock_get.return_value = mock_response

            with pytest.raises(Exception, match="Check the OETC logs for details"):
                oetc_handler.wait_and_get_job_data("job-123")
