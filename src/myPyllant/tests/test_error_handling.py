"""
Tests for error handling in HTTP requests.
"""
import asyncio
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from aioresponses import aioresponses

from myPyllant.api import MyPyllantAPI
from myPyllant.http_client import AuthenticationFailed
from myPyllant.http_helpers import (
    TransientError,
    PermanentError,
    _request_json,
)
from myPyllant.tests.utils import _mocked_api, _mypyllant_aioresponses


class TestHttpHelpers:
    """Tests for the centralized HTTP request helper."""

    @pytest.fixture
    async def mock_session(self):
        """Create a mock aiohttp ClientSession."""
        return AsyncMock(spec=aiohttp.ClientSession)

    async def test_request_json_success(self, mock_session):
        """Test successful JSON request."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "test"})
        mock_response.raise_for_status = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Mock request to return a context manager
        mock_session.request.return_value = mock_response

        result = await _request_json(
            mock_session,
            "GET",
            "http://test.com/api",
        )

        assert result == {"data": "test"}
        mock_session.request.assert_called_once_with(
            "GET",
            "http://test.com/api",
        )

    async def test_request_json_401_raises_auth_failed(self, mock_session):
        """Test that 401 status raises AuthenticationFailed."""
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Unauthorized")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.request.return_value = mock_response

        with pytest.raises(AuthenticationFailed) as exc_info:
            await _request_json(
                mock_session,
                "GET",
                "http://test.com/api",
            )

        assert "Authentication failed" in str(exc_info.value)
        assert "401" in str(exc_info.value)

    async def test_request_json_403_raises_auth_failed(self, mock_session):
        """Test that 403 status raises AuthenticationFailed."""
        mock_response = AsyncMock()
        mock_response.status = 403
        mock_response.text = AsyncMock(return_value="Forbidden")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.request.return_value = mock_response

        with pytest.raises(AuthenticationFailed) as exc_info:
            await _request_json(
                mock_session,
                "GET",
                "http://test.com/api",
            )

        assert "Authentication failed" in str(exc_info.value)
        assert "403" in str(exc_info.value)

    async def test_request_json_500_raises_transient_error(self, mock_session):
        """Test that 5xx status raises TransientError."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.request.return_value = mock_response

        with pytest.raises(TransientError) as exc_info:
            await _request_json(
                mock_session,
                "GET",
                "http://test.com/api",
            )

        assert "Server error" in str(exc_info.value)
        assert "500" in str(exc_info.value)

    async def test_request_json_503_raises_transient_error(self, mock_session):
        """Test that 503 status raises TransientError."""
        mock_response = AsyncMock()
        mock_response.status = 503
        mock_response.text = AsyncMock(return_value="Service Unavailable")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.request.return_value = mock_response

        with pytest.raises(TransientError) as exc_info:
            await _request_json(
                mock_session,
                "GET",
                "http://test.com/api",
            )

        assert "Server error" in str(exc_info.value)
        assert "503" in str(exc_info.value)

    async def test_request_json_invalid_json_raises_permanent_error(
        self, mock_session
    ):
        """Test that invalid JSON raises PermanentError."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            side_effect=aiohttp.ContentTypeError(
                Mock(), Mock(), message="Invalid JSON"
            )
        )
        mock_response.text = AsyncMock(return_value="Not JSON")
        mock_response.raise_for_status = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.request.return_value = mock_response

        with pytest.raises(PermanentError) as exc_info:
            await _request_json(
                mock_session,
                "GET",
                "http://test.com/api",
            )

        assert "Invalid JSON response" in str(exc_info.value)

    async def test_request_json_connection_error_raises_transient_error(
        self, mock_session
    ):
        """Test that connection errors raise TransientError."""
        mock_session.request.side_effect = aiohttp.ClientConnectorError(
            Mock(), Mock()
        )

        with pytest.raises(TransientError) as exc_info:
            await _request_json(
                mock_session,
                "GET",
                "http://test.com/api",
            )

        assert "Connection error" in str(exc_info.value)

    async def test_request_json_timeout_raises_transient_error(
        self, mock_session
    ):
        """Test that timeouts raise TransientError."""
        mock_session.request.side_effect = asyncio.TimeoutError()

        with pytest.raises(TransientError) as exc_info:
            await _request_json(
                mock_session,
                "GET",
                "http://test.com/api",
            )

        assert "Request timeout" in str(exc_info.value)

    async def test_request_json_404_raises_permanent_error(self, mock_session):
        """Test that 404 status raises PermanentError."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not Found")
        mock_response.raise_for_status = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(),
                history=(),
                status=404,
                message="Not Found",
            )
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.request.return_value = mock_response

        with pytest.raises(PermanentError) as exc_info:
            await _request_json(
                mock_session,
                "GET",
                "http://test.com/api",
            )

        assert "Client error" in str(exc_info.value)


class TestAPIErrorHandling:
    """Tests for API methods error handling."""

    @pytest.fixture
    async def mocked_api(self) -> MyPyllantAPI:
        """Create a mocked API instance."""
        return await _mocked_api()

    async def test_get_diagnostic_trouble_codes_returns_none_on_error(
        self, mocked_api, mypyllant_aioresponses
    ):
        """Test that get_diagnostic_trouble_codes returns None on error."""
        with mypyllant_aioresponses(
            raise_exception=aiohttp.ClientConnectorError(Mock(), Mock())
        ):
            result = await mocked_api.get_diagnostic_trouble_codes("system-id")
            assert result is None

    async def test_get_rts_returns_default_on_error(
        self, mocked_api, mypyllant_aioresponses
    ):
        """Test that get_rts returns default dict on error."""
        with mypyllant_aioresponses(
            raise_exception=asyncio.TimeoutError()
        ):
            result = await mocked_api.get_rts("system-id")
            assert result == {"statistics": []}

    async def test_get_mpc_returns_default_on_error(
        self, mocked_api, mypyllant_aioresponses
    ):
        """Test that get_mpc returns default dict on error."""
        with mypyllant_aioresponses(
            raise_exception=aiohttp.ClientConnectorError(Mock(), Mock())
        ):
            result = await mocked_api.get_mpc("system-id")
            assert result == {"devices": []}

    async def test_get_energy_management_returns_default_on_error(
        self, mocked_api, mypyllant_aioresponses
    ):
        """Test that get_energy_management returns default dict on error."""
        with mypyllant_aioresponses(
            raise_exception=asyncio.TimeoutError()
        ):
            result = await mocked_api.get_energy_management("system-id")
            assert result == {}

    async def test_get_eebus_returns_default_on_error(
        self, mocked_api, mypyllant_aioresponses
    ):
        """Test that get_eebus returns default dict on error."""
        with mypyllant_aioresponses(
            raise_exception=aiohttp.ClientConnectorError(Mock(), Mock())
        ):
            result = await mocked_api.get_eebus("system-id")
            assert result == {}

    async def test_get_ambisense_capability_returns_false_on_error(
        self, mocked_api, mypyllant_aioresponses
    ):
        """Test that get_ambisense_capability returns False on error."""
        with mypyllant_aioresponses(
            raise_exception=asyncio.TimeoutError()
        ):
            result = await mocked_api.get_ambisense_capability("system-id")
            assert result is False

    async def test_get_ambisense_rooms_returns_empty_list_on_error(
        self, mocked_api, mypyllant_aioresponses
    ):
        """Test that get_ambisense_rooms returns empty list on error."""
        with mypyllant_aioresponses(
            raise_exception=aiohttp.ClientConnectorError(Mock(), Mock())
        ):
            result = await mocked_api.get_ambisense_rooms("system-id")
            assert result == []


class TestRetryBehavior:
    """Tests for retry behavior on transient errors."""

    async def test_request_json_retries_on_transient_error(self):
        """Test that _request_json retries on TransientError."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        
        # First two attempts fail with 503, third succeeds
        mock_response_fail = AsyncMock()
        mock_response_fail.status = 503
        mock_response_fail.text = AsyncMock(return_value="Service Unavailable")
        mock_response_fail.__aenter__ = AsyncMock(return_value=mock_response_fail)
        mock_response_fail.__aexit__ = AsyncMock(return_value=None)

        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={"data": "success"})
        mock_response_success.raise_for_status = AsyncMock()
        mock_response_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_response_success.__aexit__ = AsyncMock(return_value=None)

        mock_session.request.side_effect = [
            mock_response_fail,
            mock_response_fail,
            mock_response_success,
        ]

        result = await _request_json(
            mock_session,
            "GET",
            "http://test.com/api",
        )

        assert result == {"data": "success"}
        assert mock_session.request.call_count == 3

    async def test_request_json_gives_up_after_max_retries(self):
        """Test that _request_json gives up after max retries."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        
        # All attempts fail with 503
        mock_response_fail = AsyncMock()
        mock_response_fail.status = 503
        mock_response_fail.text = AsyncMock(return_value="Service Unavailable")
        mock_response_fail.__aenter__ = AsyncMock(return_value=mock_response_fail)
        mock_response_fail.__aexit__ = AsyncMock(return_value=None)

        mock_session.request.return_value = mock_response_fail

        with pytest.raises(TransientError):
            await _request_json(
                mock_session,
                "GET",
                "http://test.com/api",
            )

        # Should attempt 3 times (initial + 2 retries)
        assert mock_session.request.call_count == 3

    async def test_request_json_does_not_retry_permanent_errors(self):
        """Test that _request_json does not retry on permanent errors."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        
        # Return 401 (authentication error)
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Unauthorized")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.request.return_value = mock_response

        with pytest.raises(AuthenticationFailed):
            await _request_json(
                mock_session,
                "GET",
                "http://test.com/api",
            )

        # Should only attempt once (no retries for auth errors)
        assert mock_session.request.call_count == 1
