import pytest
from datetime import datetime
from unittest.mock import patch, Mock

import requests
from requests import RequestException

from linopy.oetc import OetcCredentials, OetcSettings, OetcHandler, AuthenticationResult


class TestOetcHandler:

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing"""
        credentials = OetcCredentials(
            email="test@example.com",
            password="test_password"
        )
        return OetcSettings(
            credentials=credentials,
            authentication_server_url="https://auth.example.com"
        )

    @pytest.fixture
    def mock_jwt_response(self):
        """Create a mock JWT response"""
        return {
            "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "Bearer",
            "expires_in": 3600
        }

    @patch('linopy.oetc.requests.post')
    @patch('linopy.oetc.datetime')
    def test_successful_authentication(self, mock_datetime, mock_post, mock_settings, mock_jwt_response):
        """Test successful authentication flow"""
        # Setup mocks
        fixed_time = datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.now.return_value = fixed_time

        mock_response = Mock()
        mock_response.json.return_value = mock_jwt_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Execute
        handler = OetcHandler(mock_settings)

        # Verify requests.post was called correctly
        mock_post.assert_called_once_with(
            "https://auth.example.com/sign-in",
            json={
                "email": "test@example.com",
                "password": "test_password"
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        # Verify AuthenticationResult
        assert isinstance(handler.jwt, AuthenticationResult)
        assert handler.jwt.token == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        assert handler.jwt.token_type == "Bearer"
        assert handler.jwt.expires_in == 3600
        assert handler.jwt.authenticated_at == fixed_time

    @patch('linopy.oetc.requests.post')
    def test_authentication_http_error(self, mock_post, mock_settings):
        """Test authentication failure with HTTP error"""
        # Setup mock to raise HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
        mock_post.return_value = mock_response

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            OetcHandler(mock_settings)

        assert "Authentication request failed" in str(exc_info.value)

    @patch('linopy.oetc.requests.post')
    def test_authentication_network_error(self, mock_post, mock_settings):
        """Test authentication failure with network error"""
        # Setup mock to raise network error
        mock_post.side_effect = RequestException("Connection timeout")

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            OetcHandler(mock_settings)

        assert "Authentication request failed" in str(exc_info.value)

    @patch('linopy.oetc.requests.post')
    def test_authentication_invalid_response_missing_token(self, mock_post, mock_settings):
        """Test authentication failure with missing token in response"""
        # Setup mock with invalid response
        mock_response = Mock()
        mock_response.json.return_value = {
            "token_type": "Bearer",
            "expires_in": 3600
            # Missing "token" field
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            OetcHandler(mock_settings)

        assert "Invalid response format: missing field 'token'" in str(exc_info.value)

    @patch('linopy.oetc.requests.post')
    def test_authentication_invalid_response_missing_expires_in(self, mock_post, mock_settings):
        """Test authentication failure with missing expires_in in response"""
        # Setup mock with invalid response
        mock_response = Mock()
        mock_response.json.return_value = {
            "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "Bearer"
            # Missing "expires_in" field
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            OetcHandler(mock_settings)

        assert "Invalid response format: missing field 'expires_in'" in str(exc_info.value)

    @patch('linopy.oetc.requests.post')
    def test_authentication_timeout_error(self, mock_post, mock_settings):
        """Test authentication failure with timeout"""
        # Setup mock to raise timeout error
        mock_post.side_effect = requests.Timeout("Request timeout")

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            OetcHandler(mock_settings)

        assert "Authentication request failed" in str(exc_info.value)


class TestAuthenticationResult:

    @pytest.fixture
    def auth_result(self):
        """Create an AuthenticationResult for testing"""
        return AuthenticationResult(
            token="test_token",
            token_type="Bearer",
            expires_in=3600,  # 1 hour
            authenticated_at=datetime(2024, 1, 15, 12, 0, 0)
        )

    def test_expires_at_calculation(self, auth_result):
        """Test that expires_at correctly calculates expiration time"""
        expected_expiry = datetime(2024, 1, 15, 13, 0, 0)  # 1 hour later
        assert auth_result.expires_at == expected_expiry

    @patch('linopy.oetc.datetime')
    def test_is_expired_false_when_not_expired(self, mock_datetime, auth_result):
        """Test is_expired returns False when token is still valid"""
        # Set current time to before expiration
        mock_datetime.now.return_value = datetime(2024, 1, 15, 12, 30, 0)

        assert auth_result.is_expired is False

    @patch('linopy.oetc.datetime')
    def test_is_expired_true_when_expired(self, mock_datetime, auth_result):
        """Test is_expired returns True when token has expired"""
        # Set current time to after expiration
        mock_datetime.now.return_value = datetime(2024, 1, 15, 14, 0, 0)

        assert auth_result.is_expired is True

    @patch('linopy.oetc.datetime')
    def test_is_expired_true_when_exactly_expired(self, mock_datetime, auth_result):
        """Test is_expired returns True when token expires exactly now"""
        # Set current time to exact expiration time
        mock_datetime.now.return_value = datetime(2024, 1, 15, 13, 0, 0)

        assert auth_result.is_expired is True


class TestOetcHandlerIntegration:
    @patch('linopy.oetc.requests.post')
    @patch('linopy.oetc.datetime')
    def test_complete_authentication_flow(self, mock_datetime, mock_post):
        """Test complete authentication flow with realistic data"""
        # Setup
        fixed_time = datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.now.return_value = fixed_time

        credentials = OetcCredentials(
            email="user@company.com",
            password="secure_password_123"
        )
        settings = OetcSettings(
            credentials=credentials,
            authentication_server_url="https://api.company.com/auth"
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "token_type": "Bearer",
            "expires_in": 7200  # 2 hours
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Execute
        handler = OetcHandler(settings)

        # Verify
        assert handler.jwt.token.startswith("eyJhbGciOiJIUzI1NiI")
        assert handler.jwt.token_type == "Bearer"
        assert handler.jwt.expires_in == 7200
        assert handler.jwt.authenticated_at == fixed_time
        assert handler.jwt.expires_at == datetime(2024, 1, 15, 14, 0, 0)  # 2 hours later

        # Test that token is not expired immediately after authentication
        assert handler.jwt.is_expired is False