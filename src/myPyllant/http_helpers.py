"""
HTTP helpers for centralized error handling and retries.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from myPyllant.http_client import AuthenticationFailed

logger = logging.getLogger(__name__)


class TransientError(ConnectionError):
    """
    Raised for temporary errors that may succeed on retry.
    Examples: network timeouts, 5xx server errors, connection failures.
    """
    pass


class PermanentError(ConnectionError):
    """
    Raised for permanent errors that won't succeed on retry.
    Examples: invalid JSON, 4xx client errors (except auth).
    """
    pass


@retry(
    retry=retry_if_exception_type(TransientError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _request_json(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    **kwargs: Any,
) -> dict | list:
    """
    Make an HTTP request and return JSON response with proper error handling.

    Args:
        session: aiohttp ClientSession to use for the request
        method: HTTP method (GET, POST, etc.)
        url: URL to request
        **kwargs: Additional arguments to pass to session.request

    Returns:
        Parsed JSON response as dict or list

    Raises:
        AuthenticationFailed: For 401/403 responses
        TransientError: For network errors, timeouts, 5xx errors (retried)
        PermanentError: For invalid JSON or other permanent failures
    """
    try:
        async with session.request(method, url, **kwargs) as resp:
            # Check for authentication errors first
            if resp.status in (401, 403):
                text = await resp.text()
                logger.error(
                    "Authentication failed for %s %s: %s (status %d)",
                    method,
                    url,
                    text,
                    resp.status,
                )
                raise AuthenticationFailed(
                    f"Authentication failed: {resp.status} {text}"
                )

            # Check for server errors (transient)
            if resp.status >= 500:
                text = await resp.text()
                logger.warning(
                    "Server error for %s %s: %s (status %d) - will retry",
                    method,
                    url,
                    text,
                    resp.status,
                )
                raise TransientError(
                    f"Server error: {resp.status} {text}"
                )

            # Check for other client errors (4xx)
            if 400 <= resp.status < 500:
                text = await resp.text()
                logger.error(
                    "Client error for %s %s: %s (status %d)",
                    method,
                    url,
                    text,
                    resp.status,
                )
                raise PermanentError(f"Client error: {resp.status} {text}")

            # Parse JSON
            try:
                return await resp.json()
            except (aiohttp.ContentTypeError, ValueError) as e:
                text = await resp.text()
                logger.error(
                    "Invalid JSON response for %s %s: %s",
                    method,
                    url,
                    text[:500],  # Limit logged text
                )
                raise PermanentError(
                    f"Invalid JSON response: {text[:200]}"
                ) from e

    except aiohttp.ClientConnectorError as e:
        logger.warning(
            "Connection error for %s %s: %s - will retry",
            method,
            url,
            str(e),
        )
        raise TransientError(f"Connection error: {e}") from e
    except asyncio.TimeoutError as e:
        logger.warning(
            "Timeout for %s %s - will retry",
            method,
            url,
        )
        raise TransientError(f"Request timeout: {url}") from e
    except aiohttp.ClientError as e:
        # Catch other aiohttp errors as transient
        logger.warning(
            "Client error for %s %s: %s - will retry",
            method,
            url,
            str(e),
        )
        raise TransientError(f"Client error: {e}") from e
